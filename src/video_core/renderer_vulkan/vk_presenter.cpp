// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/debug.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "core/emulator_settings.h"
#include "core/libraries/system/systemservice.h"
#include "imgui/friends_layer.h"
#include "imgui/invitation_prompt_layer.h"
#include "imgui/notifications_layer.h"
#include "imgui/renderer/imgui_core.h"
#include "imgui/renderer/imgui_impl_vulkan.h"
#include "imgui/shadnet_notifications_layer.h"
#include "sdl_window.h"
#include "video_core/bb/bb_native_mv.h"
#include "video_core/bb/bb_objmv2.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/bb/bb_fg_internal.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <sstream>
#include <system_error>
#include <vector>
#include <imgui.h>
#include <stb_image_write.h>
#include <vk_mem_alloc.h>

namespace Vulkan {

bool CanBlitToSwapchain(const vk::PhysicalDevice physical_device, vk::Format format) {
    const vk::FormatProperties props{physical_device.getFormatProperties(format)};
    return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);
}

[[nodiscard]] vk::ImageSubresourceLayers MakeImageSubresourceLayers() {
    return vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlit(s32 frame_width, s32 frame_height, s32 dst_width,
                                          s32 dst_height, s32 offset_x, s32 offset_y) {
    return vk::ImageBlit{
        .srcSubresource = MakeImageSubresourceLayers(),
        .srcOffsets =
            std::array{
                vk::Offset3D{
                    .x = 0,
                    .y = 0,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = frame_width,
                    .y = frame_height,
                    .z = 1,
                },
            },
        .dstSubresource = MakeImageSubresourceLayers(),
        .dstOffsets =
            std::array{
                vk::Offset3D{
                    .x = offset_x,
                    .y = offset_y,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = offset_x + dst_width,
                    .y = offset_y + dst_height,
                    .z = 1,
                },
            },
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlitStretch(s32 frame_width, s32 frame_height,
                                                 s32 swapchain_width, s32 swapchain_height) {
    return MakeImageBlit(frame_width, frame_height, swapchain_width, swapchain_height, 0, 0);
}

static vk::Rect2D FitImage(s32 frame_width, s32 frame_height, s32 swapchain_width,
                           s32 swapchain_height) {
    float frame_aspect = static_cast<float>(frame_width) / frame_height;
    float swapchain_aspect = static_cast<float>(swapchain_width) / swapchain_height;

    u32 dst_width = swapchain_width;
    u32 dst_height = swapchain_height;

    if (frame_aspect > swapchain_aspect) {
        dst_height = static_cast<s32>(swapchain_width / frame_aspect);
    } else {
        dst_width = static_cast<s32>(swapchain_height * frame_aspect);
    }

    const s32 offset_x = (swapchain_width - dst_width) / 2;
    const s32 offset_y = (swapchain_height - dst_height) / 2;

    return vk::Rect2D{{offset_x, offset_y}, {dst_width, dst_height}};
}

[[nodiscard]] vk::ImageBlit MakeImageBlitFit(s32 frame_width, s32 frame_height, s32 swapchain_width,
                                             s32 swapchain_height) {
    const auto& dst_rect = FitImage(frame_width, frame_height, swapchain_width, swapchain_height);

    return MakeImageBlit(frame_width, frame_height, dst_rect.extent.width, dst_rect.extent.height,
                         dst_rect.offset.x, dst_rect.offset.y);
}

enum class ScreenshotKind : u8 {
    GameOnly,
    WithOverlays,
};

struct ScreenshotReadback {
    ScreenshotKind kind{};
    std::vector<std::filesystem::path> paths{};
    VideoCore::Buffer buffer;
    u32 width{};
    u32 height{};
    vk::Format format{};
    bool hdr_encoded{};

    ScreenshotReadback(const Instance& instance, ScreenshotKind kind_,
                       std::vector<std::filesystem::path> paths_, const u32 width_,
                       const u32 height_, const vk::Format format_, const bool hdr_encoded_)
        : kind{kind_}, paths{std::move(paths_)},
          buffer{instance, 0, static_cast<u64>(width_) * static_cast<u64>(height_) * 4,
                 VideoCore::MemoryType::HostCached},
          width{width_}, height{height_}, format{format_}, hdr_encoded{hdr_encoded_} {}
};

static std::string SanitizeFilenameComponent(std::string value) {
    for (char& c : value) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_' && c != '-') {
            c = '_';
        }
    }
    if (value.empty()) {
        return "UNKNOWN";
    }
    return value;
}

static std::vector<std::filesystem::path> BuildScreenshotPaths(const ScreenshotKind kind,
                                                               const u32 count) {
    static std::atomic<u64> screenshot_sequence{0};
    std::vector<std::filesystem::path> paths{};
    if (count == 0) {
        return paths;
    }

    const auto& screenshots_dir = Common::FS::GetUserPath(Common::FS::PathType::ScreenshotsDir);
    std::filesystem::create_directories(screenshots_dir);

    const auto game_id =
        SanitizeFilenameComponent(std::string(Common::ElfInfo::Instance().GameSerial()));
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;

    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream stamp;
    stamp << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << '_' << std::setw(3) << std::setfill('0')
          << ms;

    const char* suffix = kind == ScreenshotKind::GameOnly ? "game" : "hud";
    const auto first_sequence = screenshot_sequence.fetch_add(count, std::memory_order_relaxed);

    paths.reserve(count);
    const auto stamp_str = stamp.str();
    for (u32 i = 0; i < count; ++i) {
        paths.emplace_back(screenshots_dir / fmt::format("{}_{}_{}_{:06}.png", game_id, stamp_str,
                                                         suffix, first_sequence + i));
    }

    return paths;
}

static float PqToNits(const float encoded) {
    // ST.2084 inverse EOTF
    constexpr float m1 = 2610.0f / 16384.0f;
    constexpr float m2 = 2523.0f / 32.0f;
    constexpr float c1 = 3424.0f / 4096.0f;
    constexpr float c2 = 2413.0f / 128.0f;
    constexpr float c3 = 2392.0f / 128.0f;

    const float v = std::clamp(encoded, 0.0f, 1.0f);
    const float vp = std::pow(v, 1.0f / m2);
    const float num = std::max(vp - c1, 0.0f);
    const float den = std::max(c2 - c3 * vp, 1e-6f);
    return 10000.0f * std::pow(num / den, 1.0f / m1);
}

static float ToneMapToSdrLinear(const float nits) {
    // Map absolute HDR luminance into SDR [0,1], preserving 100-nit white.
    constexpr float sdr_white_nits = 100.0f;
    const float x = std::max(nits, 0.0f) / sdr_white_nits;
    const float mapped = (2.0f * x) / (1.0f + x);
    return std::clamp(mapped, 0.0f, 1.0f);
}

static float LinearToSrgb(const float linear) {
    const float x = std::clamp(linear, 0.0f, 1.0f);
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

static const std::array<float, 1024>& GetPqDecodeNitsLut() {
    static const std::array<float, 1024> lut = [] {
        std::array<float, 1024> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = PqToNits(static_cast<float>(i) / 1023.0f);
        }
        return values;
    }();
    return lut;
}

static const std::array<u8, 1024>& GetUnorm10ToU8Lut() {
    static const std::array<u8, 1024> lut = [] {
        std::array<u8, 1024> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<u8>((i * 255u + 511u) / 1023u);
        }
        return values;
    }();
    return lut;
}

static void CopyImageToReadback(const vk::CommandBuffer& cmdbuf, const vk::Image image,
                                const vk::ImageLayout layout, ScreenshotReadback& readback) {
    const vk::BufferImageCopy copy_region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {readback.width, readback.height, 1},
    };
    cmdbuf.copyImageToBuffer(image, layout, readback.buffer.Handle(), copy_region);
}

static bool ConvertReadbackToRgba8(const ScreenshotReadback& readback, std::vector<u8>& out_rgba) {
    const u64 pixel_count = static_cast<u64>(readback.width) * static_cast<u64>(readback.height);
    const u64 byte_size = pixel_count * 4;
    if (readback.buffer.mapped_data.size() < byte_size) {
        LOG_ERROR(Render_Vulkan, "Screenshot readback buffer size mismatch (have {}, need {})",
                  readback.buffer.mapped_data.size(), byte_size);
        return false;
    }

    const auto src =
        std::span<const u8>{readback.buffer.mapped_data.data(), static_cast<size_t>(byte_size)};
    out_rgba.resize(static_cast<size_t>(byte_size));

    switch (readback.format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        std::memcpy(out_rgba.data(), src.data(), out_rgba.size());
        for (u64 i = 0; i < pixel_count; ++i) {
            out_rgba[static_cast<size_t>(i) * 4 + 3] = 255;
        }
        return true;
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            out_rgba[o + 0] = src[o + 2];
            out_rgba[o + 1] = src[o + 1];
            out_rgba[o + 2] = src[o + 0];
            out_rgba[o + 3] = 255;
        }
        return true;
    case vk::Format::eA2R10G10B10UnormPack32: {
        const auto& pq_decode_lut = GetPqDecodeNitsLut();
        const auto& unorm10_to_u8 = GetUnorm10ToU8Lut();

        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            const u32 packed = static_cast<u32>(src[o + 0]) | (static_cast<u32>(src[o + 1]) << 8) |
                               (static_cast<u32>(src[o + 2]) << 16) |
                               (static_cast<u32>(src[o + 3]) << 24);
            const u32 b = (packed >> 0) & 0x3FF;
            const u32 g = (packed >> 10) & 0x3FF;
            const u32 r = (packed >> 20) & 0x3FF;

            if (readback.hdr_encoded) {
                // Rec.2020 + PQ. Convert to SDR Rec.709 for PNG output.
                const float r2020 = pq_decode_lut[r];
                const float g2020 = pq_decode_lut[g];
                const float b2020 = pq_decode_lut[b];

                const float r709_nits = 1.6605f * r2020 - 0.5876f * g2020 - 0.0728f * b2020;
                const float g709_nits = -0.1246f * r2020 + 1.1329f * g2020 - 0.0083f * b2020;
                const float b709_nits = -0.0182f * r2020 - 0.1006f * g2020 + 1.1187f * b2020;

                const float r_srgb = LinearToSrgb(ToneMapToSdrLinear(r709_nits));
                const float g_srgb = LinearToSrgb(ToneMapToSdrLinear(g709_nits));
                const float b_srgb = LinearToSrgb(ToneMapToSdrLinear(b709_nits));

                out_rgba[o + 0] = static_cast<u8>(std::clamp(r_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 1] = static_cast<u8>(std::clamp(g_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 2] = static_cast<u8>(std::clamp(b_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
            } else {
                out_rgba[o + 0] = unorm10_to_u8[r];
                out_rgba[o + 1] = unorm10_to_u8[g];
                out_rgba[o + 2] = unorm10_to_u8[b];
            }
            out_rgba[o + 3] = 255;
        }
        return true;
    }
    case vk::Format::eA2B10G10R10UnormPack32: {
        const auto& pq_decode_lut = GetPqDecodeNitsLut();
        const auto& unorm10_to_u8 = GetUnorm10ToU8Lut();

        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            const u32 packed = static_cast<u32>(src[o + 0]) | (static_cast<u32>(src[o + 1]) << 8) |
                               (static_cast<u32>(src[o + 2]) << 16) |
                               (static_cast<u32>(src[o + 3]) << 24);
            const u32 r = (packed >> 0) & 0x3FF;
            const u32 g = (packed >> 10) & 0x3FF;
            const u32 b = (packed >> 20) & 0x3FF;

            if (readback.hdr_encoded) {
                // HDR swapchain path is Rec.2020 + PQ. Convert to SDR Rec.709 for PNG output.
                const float r2020 = pq_decode_lut[r];
                const float g2020 = pq_decode_lut[g];
                const float b2020 = pq_decode_lut[b];

                const float r709_nits = 1.6605f * r2020 - 0.5876f * g2020 - 0.0728f * b2020;
                const float g709_nits = -0.1246f * r2020 + 1.1329f * g2020 - 0.0083f * b2020;
                const float b709_nits = -0.0182f * r2020 - 0.1006f * g2020 + 1.1187f * b2020;

                const float r_srgb = LinearToSrgb(ToneMapToSdrLinear(r709_nits));
                const float g_srgb = LinearToSrgb(ToneMapToSdrLinear(g709_nits));
                const float b_srgb = LinearToSrgb(ToneMapToSdrLinear(b709_nits));

                out_rgba[o + 0] = static_cast<u8>(std::clamp(r_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 1] = static_cast<u8>(std::clamp(g_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 2] = static_cast<u8>(std::clamp(b_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
            } else {
                out_rgba[o + 0] = unorm10_to_u8[r];
                out_rgba[o + 1] = unorm10_to_u8[g];
                out_rgba[o + 2] = unorm10_to_u8[b];
            }
            out_rgba[o + 3] = 255;
        }
        return true;
    }
    default:
        LOG_WARNING(Render_Vulkan, "Unsupported screenshot format: {}",
                    vk::to_string(readback.format));
        return false;
    }
}

static bool WritePng(const std::filesystem::path& path, const std::span<const u8> rgba,
                     const u32 width, const u32 height) {
    Common::FS::IOFile file(path, Common::FS::FileAccessMode::Create);
    if (!file.IsOpen()) {
        return false;
    }

    auto callback = [](void* context, void* data, int size) {
        const auto* f = static_cast<Common::FS::IOFile*>(context);
        f->WriteRaw<u8>(data, size);
    };
    return stbi_write_png_to_func(callback, &file, width, height, 4, rgba.data(), 0);
}

static void SavePendingScreenshot(const ScreenshotReadback& readback) {
    if (readback.paths.empty()) {
        return;
    }

    std::vector<u8> rgba;
    if (!ConvertReadbackToRgba8(readback, rgba)) {
        return;
    }

    const auto& primary_path = readback.paths.front();
    if (!WritePng(primary_path, rgba, readback.width, readback.height)) {
        LOG_ERROR(Render_Vulkan, "Failed saving screenshot to {}", primary_path.string());
        return;
    }

    LOG_INFO(Render_Vulkan, "Saved screenshot: {}", primary_path.string());

    std::ifstream file(primary_path, std::ios::binary);
    std::vector<u8> imgdata;
    if (file) {
        imgdata =
            std::vector<u8>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    shadNotifications::QueueNotification("Saved screenshot:\n" + primary_path.string(), 3.0f,
                                         shadNotifications::position::BottomRight, imgdata);

    for (size_t i = 1; i < readback.paths.size(); ++i) {
        const auto& path = readback.paths[i];
        std::error_code ec{};
        std::filesystem::copy_file(primary_path, path, std::filesystem::copy_options::none, ec);
        if (ec) {
            // Fallback for platforms/filesystems where copy_file can fail for transient
            // reasons.
            if (!WritePng(path, rgba, readback.width, readback.height)) {
                LOG_ERROR(Render_Vulkan, "Failed saving screenshot to {}", path.string());
                continue;
            }
        }

        LOG_INFO(Render_Vulkan, "Saved screenshot: {}", path.string());
        std::ifstream file(path, std::ios::binary);
        std::vector<u8> imgdata;
        if (file) {
            imgdata = std::vector<u8>(std::istreambuf_iterator<char>(file),
                                      std::istreambuf_iterator<char>());
        }
        shadNotifications::QueueNotification("Saved screenshot:\n" + path.string(), 3.0f,
                                             shadNotifications::position::BottomRight, imgdata);
    }
}

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_},
      instance{window, EmulatorSettings.GetGpuId(), EmulatorSettings.IsVkValidationEnabled(),
               EmulatorSettings.IsVkCrashDiagnosticEnabled()},
      draw_scheduler{instance}, present_scheduler{instance}, flip_scheduler{instance},
      swapchain{instance, window}, runtime{instance, draw_scheduler},
      rasterizer{std::make_unique<Rasterizer>(instance, draw_scheduler, runtime, liverpool)},
      texture_cache{rasterizer->GetTextureCache()} {
    const u32 num_images = swapchain.GetImageCount();
    const vk::Device device = instance.GetDevice();

    // Create presentation frames.
    present_frames.resize(num_images);
    for (u32 i = 0; i < num_images; i++) {
        Frame& frame = present_frames[i];
        frame.id = i;
        auto fence = Check<"create present done fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        frame.present_done = fence;
        free_queue.push(&frame);
    }

    fsr_settings.enable = EmulatorSettings.IsFsrEnabled();
    fsr_settings.use_rcas = EmulatorSettings.IsRcasEnabled();
    fsr_settings.rcas_attenuation =
        static_cast<float>(EmulatorSettings.GetRcasAttenuation() / 1000.f);

    fsr_pass.Create(device, instance.GetAllocator(), num_images);
    bb_sr = std::make_unique<BB::PresentSr>(instance);
    bb_hud_fg = std::make_unique<BB::HudFg>(instance);
    rasterizer->bb_cut.SetUpscaler(bb_sr.get()); // bb-clean-021-bbport: the HUD cut runs it
    pp_pass.Create(device, swapchain.GetSurfaceFormat().format);

    ImGui::Layer::AddLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
    ImGui::Friends::Register();
    ImGui::ShadNetNotify::Register();
    ImGui::InvitationPrompt::Register();
}

Presenter::~Presenter() {
    ImGui::InvitationPrompt::Unregister();
    ImGui::ShadNetNotify::Unregister();
    ImGui::Friends::Unregister();
    ImGui::Layer::RemoveLayer(Common::Singleton<Core::Devtools::Layer>::Instance());

    draw_scheduler.Finish();
    present_scheduler.Finish();
    flip_scheduler.Finish();
    Check(draw_scheduler.CommandBuffer().reset());
    Check(present_scheduler.CommandBuffer().reset());
    Check(flip_scheduler.CommandBuffer().reset());

    const vk::Device device = instance.GetDevice();
    for (auto& frame : present_frames) {
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        device.destroyImageView(frame.image_view);
        device.destroyFence(frame.present_done);
        if (frame.hudless_view) device.destroyImageView(frame.hudless_view);
        if (frame.hudless_rgba_view) device.destroyImageView(frame.hudless_rgba_view);
        if (frame.hudless_image) {
            vmaDestroyImage(instance.GetAllocator(), frame.hudless_image,
                            frame.hudless_allocation);
        }
    }
}

// bb-hudless: (re)creates a frame's HUD-less image at the frame's size and format.
static bool BbEnsureHudless(const Vulkan::Instance& instance, Frame& frame, vk::Format format) {
    if (frame.hudless_image && frame.hudless_width == frame.width &&
        frame.hudless_height == frame.height && frame.hudless_format == format) {
        return true;
    }
    const vk::Device device = instance.GetDevice();
    if (frame.hudless_image) {
        (void)device.waitIdle(); // a resize only
        device.destroyImageView(frame.hudless_view);
        if (frame.hudless_rgba_view) device.destroyImageView(frame.hudless_rgba_view);
        vmaDestroyImage(instance.GetAllocator(), frame.hudless_image, frame.hudless_allocation);
        frame.hudless_image = vk::Image{};
        frame.hudless_view = vk::ImageView{};
        frame.hudless_rgba_view = vk::ImageView{};
    }
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {frame.width, frame.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image{};
    bool storage = true;
    if (vmaCreateImage(instance.GetAllocator(), &ici, &aci, &image, &frame.hudless_allocation,
                       nullptr) != VK_SUCCESS) {
        // Storage may be unsupported for this format: retry without it.
        storage = false;
        VkImageCreateInfo ici2 = ici;
        ici2.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
        if (vmaCreateImage(instance.GetAllocator(), &ici2, &aci, &image,
                           &frame.hudless_allocation, nullptr) != VK_SUCCESS) {
            return false;
        }
    }
    frame.hudless_image = vk::Image{image};
    auto [vr, view] = device.createImageView({
        .image = frame.hudless_image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (vr != vk::Result::eSuccess) {
        vmaDestroyImage(instance.GetAllocator(), frame.hudless_image, frame.hudless_allocation);
        frame.hudless_image = vk::Image{};
        return false;
    }
    frame.hudless_view = view;
    if (storage && (format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eR8G8B8A8Unorm)) {
        auto [rr, rgba] = device.createImageView({
            .image = frame.hudless_image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (rr == vk::Result::eSuccess) frame.hudless_rgba_view = rgba;
    }
    frame.hudless_width = frame.width;
    frame.hudless_height = frame.height;
    frame.hudless_format = format;
    return true;
}

// bb-clean-021-bbport: Supermedo's DestroyGenerated / EnsureGenerated (vk_presenter.cpp, GPL-2.0):
// only as many as asked; every presentation frame has its own (the frame's fence was waited for
// in GetRenderFrame, so the GPU is done with them).
static vk::Format BbUnorm(vk::Format f) {
    switch (f) {
    case vk::Format::eB8G8R8A8Srgb:
        return vk::Format::eB8G8R8A8Unorm;
    case vk::Format::eR8G8B8A8Srgb:
        return vk::Format::eR8G8B8A8Unorm;
    default:
        return f;
    }
}

void Presenter::BbDestroyGenerated(Frame* frame, u32 from) {
    const vk::Device device = instance.GetDevice();
    for (u32 i = from; i < frame->generated.size(); ++i) {
        auto& generated = frame->generated[i];
        if (generated.view) device.destroyImageView(generated.view);
        if (generated.rgba_view) device.destroyImageView(generated.rgba_view);
        if (generated.image) {
            vmaDestroyImage(instance.GetAllocator(), generated.image, generated.allocation);
        }
        generated = {};
    }
    frame->generated_count = std::min(frame->generated_count, from);
    if (from == 0 && frame->unorm_view) {
        device.destroyImageView(frame->unorm_view);
        frame->unorm_view = vk::ImageView{};
    }
}

u32 Presenter::BbEnsureGenerated(Frame* frame, u32 count) {
    count = std::min<u32>(count, u32(frame->generated.size()));
    BbDestroyGenerated(frame, count == 0 ? 0 : count);
    if (count == 0) return 0;
    const vk::Device device = instance.GetDevice();
    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::Format unorm = BbUnorm(format);
    const auto props = instance.GetPhysicalDevice().getFormatProperties(unorm);
    if (!(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage)) {
        static bool said = false;
        if (!said) {
            said = true;
            BB::Upscale::Checkpoint(fmt::format("DLSS-G: {} has no storage support: off",
                                                vk::to_string(unorm)));
        }
        return 0;
    }
    if (!frame->unorm_view) {
        auto [r, view] = device.createImageView({
            .image = frame->image,
            .viewType = vk::ImageViewType::e2D,
            .format = unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (r != vk::Result::eSuccess) return 0;
        frame->unorm_view = view;
        frame->unorm_format = unorm;
    }
    for (u32 i = 0; i < count; ++i) {
        auto& generated = frame->generated[i];
        if (generated.image) continue;
        const VkImageCreateInfo image_info = static_cast<VkImageCreateInfo>(vk::ImageCreateInfo{
            .flags = vk::ImageCreateFlagBits::eMutableFormat |
                     vk::ImageCreateFlagBits::eExtendedUsage,
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {frame->width, frame->height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        });
        const VmaAllocationCreateInfo alloc_info = {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        };
        VkImage image{};
        if (vmaCreateImage(instance.GetAllocator(), &image_info, &alloc_info, &image,
                           &generated.allocation, nullptr) != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "Frame generation: no memory for the generated frames");
            BbDestroyGenerated(frame, i == 0 ? 1 : i);
            return i;
        }
        generated.image = vk::Image{image};
        generated.format = format;
        auto [r, view] = device.createImageView({
            .image = generated.image,
            .viewType = vk::ImageViewType::e2D,
            .format = unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (r != vk::Result::eSuccess) {
            BbDestroyGenerated(frame, i == 0 ? 1 : i);
            return i;
        }
        generated.view = view;
        // bb-clean: an R8G8B8A8 view of the same memory (mutable) for the HUD overlay pass.
        auto [rr, rgba] = device.createImageView({
            .image = generated.image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (rr == vk::Result::eSuccess) generated.rgba_view = rgba;
    }
    return count;
}

// Supermedo's UpdateFrameGenLimit: the images the display shows per game frame (60 FPS on 144 Hz
// has room for one generated frame, on 240 Hz for three), decided once a second from that
// second's game frame rate: down at once when 10% short, up after two seconds with 5% to spare;
// paused for a while when presenting holds the game back.
void Presenter::BbUpdateFrameGenLimit(bool wanted, double present_ms) {
    if (!wanted) {
        bb_frame_gen_limit = 0;
        bb_present_busy = 0.0;
        return;
    }
    const double interval = bb_game_frame_ms.load(std::memory_order_relaxed);
    const double hz_d = BB::FG::Amd::DisplayHz();
    const u32 hz = hz_d > 1.0 ? u32(std::lround(hz_d)) : 60u;
    const auto now = std::chrono::steady_clock::now();
    const u32 before = bb_frame_gen_limit.load(std::memory_order_relaxed);
    u32 limit = before;
    ++bb_limit_frames;
    const double window_s = std::chrono::duration<double>(now - bb_limit_window).count();
    if (window_s >= 1.0) {
        const double ratio = hz * window_s / bb_limit_frames;
        bb_limit_window = now;
        bb_limit_frames = 0;
        if (window_s < 2.0) { // not the first, nor after a pause of the game
            if (limit > 0 && ratio < (limit + 1) * 0.9) {
                while (limit > 0 && ratio < (limit + 1) * 0.9) {
                    --limit;
                }
                bb_limit_up_seconds = 0;
            } else if (limit < 3 && ratio >= (limit + 2) * 0.95) {
                if (++bb_limit_up_seconds >= 2) {
                    ++limit;
                    bb_limit_up_seconds = 0;
                }
            } else {
                bb_limit_up_seconds = 0;
            }
        }
    }
    // Fixed multipliers (dlss-x2..x4) as chosen, past the display's refresh rate (Supermedo's
    // frame_gen_fit_display off); dlss-dyn fits the display. (Supermedo caps the game under FIFO
    // through VideoOut; this build has no such cap, so FIFO keeps the fitted limit.)
    if (!BB::FG::Cfg().dynamic && !swapchain.IsFifo()) {
        limit = 3;
    }
    bb_present_busy += (std::min(present_ms / interval, 1.2) - bb_present_busy) * 0.1;
    // bb: the "presenting does not keep up" pause measured the whole present call, which by design
    // spans the game frame (the generated frames are spaced over it): 8.9 of 8.9 ms every time, so
    // DLSS-G was paused 5, 10, 20, 40 s in a row on a 240 Hz display with mailbox. Off unless
    // BB_FG_BUSY_PAUSE=1.
    static const bool busy_pause = [] {
        const char* v = std::getenv("BB_FG_BUSY_PAUSE");
        return v != nullptr && v[0] == '1';
    }();
    if (busy_pause && before && bb_present_busy > 0.97) {
        bb_frame_gen_hold_seconds = now - bb_frame_gen_hold_until > std::chrono::seconds(60)
                                        ? 5
                                        : std::min(bb_frame_gen_hold_seconds * 2, 120);
        bb_frame_gen_hold_until = now + std::chrono::seconds(bb_frame_gen_hold_seconds);
        BB::Upscale::Checkpoint(fmt::format(
            "Frame generation: paused for {} s, presenting does not keep up ({:.1f} of {:.1f} ms "
            "per game frame; FIFO present mode or a slower display?)",
            bb_frame_gen_hold_seconds, present_ms, interval));
        bb_present_busy = 0.0;
    }
    if (now < bb_frame_gen_hold_until) {
        limit = 0;
    }
    if (limit != before && BB::FG::Cfg().dynamic) {
        BB::Upscale::Checkpoint(fmt::format("Frame generation: {} generated per game frame "
                                            "(display {} Hz, game {:.0f} FPS)",
                                            limit, hz, 1000.0 / interval));
    }
    bb_frame_gen_limit = limit;
}

// Supermedo's PresentImage for a generated frame: acquire, blit into the swapchain image (the
// frame's place in the window), submit after the frame's own work, present.
bool Presenter::BbPresentGenerated(Frame* frame, u32 index, bool first) {
    if (!swapchain.AcquireNextImage()) {
        return false;
    }
    auto& scheduler = present_scheduler;
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::Image dst = swapchain.Image();
    const vk::Extent2D extent = swapchain.GetExtent();
    const auto& generated = frame->generated[index];
    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    const std::array pre{
        vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = dst,
            .subresourceRange = range,
        },
        vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .image = generated.image,
            .subresourceRange = range,
        },
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(pre.size()),
                             .pImageMemoryBarriers = pre.data()});
    cmdbuf.clearColorImage(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::ClearColorValue{std::array{0.0f, 0.0f, 0.0f, 1.0f}}, range);
    const vk::MemoryBarrier2 cleared{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                     .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                     .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                     .dstAccessMask = vk::AccessFlagBits2::eTransferWrite};
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &cleared});
    const auto rect = FitImage(s32(frame->width), s32(frame->height), s32(extent.width),
                               s32(extent.height));
    const vk::ImageBlit region{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(frame->width), s32(frame->height), 1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{rect.offset.x, rect.offset.y, 0},
                                 vk::Offset3D{rect.offset.x + s32(rect.extent.width),
                                              rect.offset.y + s32(rect.extent.height), 1}},
    };
    cmdbuf.blitImage(generated.image, vk::ImageLayout::eTransferSrcOptimal, dst,
                     vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);
    // bb-dlssg-ui2: the last game frame's ImGui (port menus, F8 metrics) drawn over the generated
    // frame too, without the window that shows the game image: no flicker, and DLSS-G stays on.
    bool ui_drawn = false;
    if (ImDrawData* dd = ImGui::GetDrawData(); bb_imgui_ui_shown && dd != nullptr && dd->Valid &&
                                              dd->CmdListsCount > 0) {
        const vk::ImageMemoryBarrier2 to_att{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                             vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = dst,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_att});
        std::vector<std::pair<ImDrawCmd*, unsigned int>> hidden;
        for (int l = 0; l < dd->CmdListsCount; ++l) {
            ImDrawList* list = dd->CmdLists[l];
            bool game = false;
            for (const ImDrawCmd& c : list->CmdBuffer) {
                if (c.UserCallback == nullptr && c.GetTexID() == frame->imgui_texture) game = true;
            }
            if (!game) continue;
            for (ImDrawCmd& c : list->CmdBuffer) {
                if (c.UserCallback == nullptr && c.ElemCount != 0) {
                    hidden.emplace_back(&c, c.ElemCount);
                    c.ElemCount = 0;
                }
            }
        }
        vk::RenderingAttachmentInfo att{
            .imageView = swapchain.ImageView(),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
        vk::RenderingInfo ri{};
        ri.renderArea = vk::Rect2D{.offset = {0, 0}, .extent = extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &att;
        cmdbuf.beginRendering(ri);
        ImGui::Vulkan::RenderDrawData(*dd, cmdbuf);
        cmdbuf.endRendering();
        for (auto& [c, n] : hidden) c->ElemCount = n;
        ui_drawn = true;
    }
    const std::array post{
        vk::ImageMemoryBarrier2{
            .srcStageMask = ui_drawn ? vk::PipelineStageFlagBits2::eColorAttachmentOutput
                                     : vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = ui_drawn ? vk::AccessFlagBits2::eColorAttachmentWrite
                                      : vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eNone,
            .oldLayout = ui_drawn ? vk::ImageLayout::eColorAttachmentOptimal
                                  : vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .image = dst,
            .subresourceRange = range,
        },
        vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = generated.image,
            .subresourceRange = range,
        },
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(post.size()),
                             .pImageMemoryBarriers = post.data()});
    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    if (first) info.AddWait(frame->ready_semaphore, frame->ready_tick);
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    scheduler.Flush(info);
    std::scoped_lock submit_lock{Scheduler::submit_mutex};
    if (!swapchain.Present()) {
        return false;
    }
    return true;
}


bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

void Presenter::RecreateFrame(Frame* frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    BbDestroyGenerated(frame); // bb-clean-021-bbport: DLSS-G frames of the old size
    if (frame->imgui_texture) {
        ImGui::Vulkan::RemoveTexture(frame->imgui_texture);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &frame->allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}",
                     vk::to_string(vk::Result{result}));
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};
    SetObjectName(device, frame->image, "Frame image #{}", frame->id);

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    auto view = Check<"create frame image view">(device.createImageView(view_info));
    frame->image_view = view;
    frame->width = width;
    frame->height = height;

    frame->imgui_texture = ImGui::Vulkan::AddTexture(view, vk::ImageLayout::eShaderReadOnlyOptimal);
    frame->is_hdr = swapchain.GetHDR();
}

Frame* Presenter::PrepareLastFrame() {
    if (last_submit_frame == nullptr) {
        return nullptr;
    }

    Frame* frame = last_submit_frame;

    while (true) {
        vk::Result result = instance.GetDevice().waitForFences(frame->present_done, false,
                                                               std::numeric_limits<u64>::max());
        if (result == vk::Result::eSuccess) {
            break;
        }
        if (result == vk::Result::eTimeout) {
            continue;
        }
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
    }

    auto& scheduler = flip_scheduler;
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier =
        vk::ImageMemoryBarrier2{.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
                                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                                .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                .newLayout = vk::ImageLayout::eGeneral,
                                .image = frame->image,
                                .subresourceRange{frame_subresources}};

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetWorkSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return frame;
}

static vk::Format GetFrameViewFormat(const Libraries::VideoOut::PixelFormat format) {
    switch (format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    const auto image_id = texture_cache.FindImage(desc);
    texture_cache.UpdateImage(image_id);
    const auto bb_t0 = std::chrono::steady_clock::now();
    Frame* frame = GetRenderFrame();
    bb_wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - bb_t0)
                     .count();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange{frame_subresources},
    };

    draw_scheduler.EndRendering();
    const auto cmdbuf = draw_scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    VideoCore::ImageViewInfo view_info{};
    view_info.format = GetFrameViewFormat(attribute.attrib.pixel_format);
    // Exclude alpha from output frame to avoid blending with UI.
    view_info.mapping.a = AmdGpu::CompSwizzle::One;

    auto& image = texture_cache.GetImage(image_id);
    auto image_view = *image.FindView(view_info).image_view;
    vk::Extent2D image_size = {image.info.size.width, image.info.size.height};
    expected_ratio = static_cast<float>(image_size.width) / static_cast<float>(image_size.height);

    const u32 capture_game_only_count = VideoCore::ConsumeGameOnlyScreenshotRequests();
    std::optional<ScreenshotReadback> pending_screenshot;

    // Capture the guest output before any host-side scaling (FSR/PP) is applied.
    if (capture_game_only_count > 0) {
        const bool hdr_encoded =
            attribute.attrib.pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq;
        auto& readback = pending_screenshot.emplace(
            instance, ScreenshotKind::GameOnly,
            BuildScreenshotPaths(ScreenshotKind::GameOnly, capture_game_only_count),
            image_size.width, image_size.height, view_info.format, hdr_encoded);
        const vk::BufferImageCopy copy_region = {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {readback.width, readback.height, 1},
        };
        runtime.DownloadImage(&image, &readback.buffer, std::span{&copy_region, 1});
    }

    // Continue with host-side passes that draw the displayed (scaled) frame.

    runtime.Transit(&image, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();

    // bb-clean-021: DLSS / FSR 4 on the presented image (its depth and camera motion), recorded
    // into this command buffer: no wait of its own.
    const bool srgb10 =
        attribute.attrib.pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb;
    vk::Extent2D bb_in_size = image_size;
    bool bb_upscaled = false;
    vk::Extent2D bb_sr_size{};
    vk::ImageView bb_hudless_scene{};
    vk::ImageView bb_hud_raw{}; // bb-fsr3-hudless: the UI target with the HUD (raw copy)
    bool bb_hudless_swapped = false; // bb-artifact-fix: both raw views swap red and blue
    frame->fg = {};
    // bb-clean-021-bbport: Supermedo's HUD cut (bb_ui_cut.h). A frame upscaled at its first UI
    // draw has an output-size host copy of this display buffer (the game's display pass drew it,
    // UI included, at the output's pixels): it is shown as it is, with the inputs and the
    // picture without the HUD taken at the cut for frame generation.
    BB::UiCut::NoteDisplayBuffer(image.info.guest_address);
    BB::UiCut::Display bb_display{};
    const bool bb_cut_frame =
        BB::UiCut::Active() &&
        rasterizer->bb_cut.DisplayOverride(image.info.guest_address, view_info.format, bb_display);
    if (bb_cut_frame) {
        const vk::ImageMemoryBarrier2 to_read{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = bb_display.image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
        image_view = bb_display.view;
        image_size = {bb_display.width, bb_display.height};
        bb_sr_size = image_size;
        frame->fg = bb_display.fg;
        bb_hudless_scene = bb_display.hudless_view;
        bb_hud_raw = bb_display.hud_view;
        bb_hudless_swapped = bb_display.hudless_swapped;
        bb_upscaled = true;
        ++bb_sec_cut;
    } else if (BB::Upscale::Capture()) {
        // No cut this frame (menus without the 3D scene, BB_FG without an upscaler): the
        // presented image as before.
        vk::Extent2D out_size{frame->width, frame->height};
        if (u32 ow = 0, oh = 0; BB::Upscale::Enabled() && BB::Upscale::OutputSize(ow, oh) &&
                                ow >= image_size.width && oh >= image_size.height) {
            out_size = {ow, oh};
        }
        if (const auto v = bb_sr->Run(cmdbuf, texture_cache, runtime, image, image_view,
                                      image_size, out_size, rasterizer->bb_scene_depth, srgb10,
                                      &frame->fg)) {
            image_view = v;
            image_size = out_size;
            bb_sr_size = out_size;
            bb_upscaled = true;
        }
    }

    // bb-clean-021-scene: a frame the upscaler did not take (title screen, menus, loading: no
    // scene depth) is still drawn at the resolution patch's reduced size; it is scaled to the
    // window with FSR 1 EASU + RCAS instead of the bilinear stretch (BB_MENU_SHARP=0: as before).
    // One extra pass only on those frames, in the presenter: nothing on the GPU command thread.
    auto bb_fsr = fsr_settings;
    {
        static const bool menu_sharp = [] {
            const char* v = std::getenv("BB_MENU_SHARP");
            const bool on = v == nullptr || v[0] != '0';
            BB::Upscale::Checkpoint(on ? "[menu-sharp] on: frames without the 3D scene (title, "
                                         "menus) scaled with FSR 1 EASU + RCAS"
                                       : "[menu-sharp] off (BB_MENU_SHARP=0)");
            return on;
        }();
        static u32 sec_frames = 0, sec_sharp = 0;
        static auto sec_start = std::chrono::steady_clock::now();
        const bool sharp = menu_sharp && BB::Upscale::Enabled() && !bb_upscaled &&
                           (image_size.width < frame->width || image_size.height < frame->height);
        if (sharp) {
            bb_fsr.enable = true;
            bb_fsr.use_rcas = true;
            if (!fsr_settings.enable) bb_fsr.rcas_attenuation = 0.25f;
            ++sec_sharp;
        }
        ++sec_frames;
        const auto now = std::chrono::steady_clock::now();
        if (menu_sharp && BB::Upscale::Enabled() && now - sec_start >= std::chrono::seconds(5)) {
            BB::Upscale::Checkpoint(fmt::format(
                "[menu-sharp] last 5 s: {} of {} frames without the 3D scene scaled with FSR 1 "
                "({}x{} -> {}x{})",
                sec_sharp, sec_frames, image_size.width, image_size.height, frame->width,
                frame->height));
            sec_frames = sec_sharp = 0;
            sec_start = now;
        }
    }
    image_view = fsr_pass.Render(cmdbuf, image_view, image_size, {frame->width, frame->height},
                                 bb_fsr, frame->is_hdr);

    // Vulkan has no sRGB variant of the 10-bit format, so an A2R10G10B10Srgb buffer reaches
    // the post process pass still sRGB encoded and has to be decoded there instead.
    // (The host display copy holds what the game's display pass wrote: the guest's encoding.)
    pp_settings.srgb_input = srgb10 && (!bb_upscaled || bb_cut_frame);
    // bb-overscan only on frames frame generation does not get (boot/loading/menus): FG's depth
    // and motion vectors are uncropped, so a cropped colour would sit up to 0.2 % off them and
    // smear the generated frames' edges. The HUD-less copy below takes the same setting.
    pp_settings.uv_scale =
        frame->fg.valid ? 1.0f : HostPasses::PostProcessingPass::OverscanScale();
    {
        // bb-hudless: the HUD-less colour for frame generation (Supermedo's CaptureHudlessUi copy
        // at the HUD cut) through the same post-process pass into an image of the frame's size
        // and format (the backbuffer's): AMD FSR 3 FG's HUD-less input and DLSS-G's.
        u32 failed = 0;
        bool ok = false;
        if (bb_hudless_scene && frame->fg.valid) {
            const vk::Format fmt = swapchain.GetSurfaceFormat().format;
            if (BbEnsureHudless(instance, *frame, fmt)) {
                const vk::ImageMemoryBarrier2 to_att{
                    .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
                    .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .image = frame->hudless_image,
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                };
                cmdbuf.pipelineBarrier2(
                    vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_att});
                Frame target = *frame;
                target.image = frame->hudless_image;
                target.image_view = frame->hudless_view;
                auto hs = pp_settings;
                hs.srgb_input = srgb10; // the UI target's copy, encoded like the display
                pp_pass.Render(cmdbuf, bb_hudless_scene, image_size, target, hs);
                const vk::ImageMemoryBarrier2 to_read{
                    .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .image = frame->hudless_image,
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                };
                cmdbuf.pipelineBarrier2(
                    vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
                frame->fg.hudless = {frame->hudless_image, frame->hudless_view, fmt,
                                     {frame->width, frame->height},
                                     vk::ImageUsageFlagBits::eColorAttachment |
                                         vk::ImageUsageFlagBits::eSampled};
                frame->fg.has_hudless = true;
                ok = true;
            } else {
                ++failed;
            }
        }
        bb_sec_hudless += ok ? 1u : 0u;
        bb_sec_hudless_failed += failed;
    }
    pp_pass.Render(cmdbuf, image_view, image_size, *frame, pp_settings);

    // bb-fsr3-hudless: FSR 3 FG takes every pixel where the back buffer and the HUD-less image
    // differ as UI (the real frame's pixel in generated frames). The HUD-less copy went through
    // our post-process pass but not the game's display pass, so it differed everywhere and the
    // generated frames repeated the real one. Outside the HUD it now is the finished frame itself.
    // FSR 3 only (DLSS-G keeps its input as before).
    if (frame->fg.has_hudless && BB::FG::Fsr3Active() && !frame->is_hdr) {
        const vk::Format fmt = swapchain.GetSurfaceFormat().format;
        const vk::MemoryBarrier2 written{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        };
        cmdbuf.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &written});
        bb_hud_fg->FixHudless(cmdbuf, frame->image_view, bb_hud_raw, bb_hudless_scene,
                              frame->hudless_image, frame->hudless_rgba_view,
                              {frame->width, frame->height}, fmt == vk::Format::eB8G8R8A8Unorm,
                              bb_hudless_swapped);
    }

    // bb-clean: the HUD as a separate UI layer for frame generation (bb_hud_fg.h): built from the
    // finished frame, the HUD-less copy (when the HUD cut ran) and the previous real frame; drawn
    // over DLSS-G's generated frames below and given to FSR 3 as its UI resource at the present.
    const BB::FG::Image* bb_ui_layer = nullptr;
    // bb-clean-2: the UI layer only feeds DLSS-G (and FSR 3 when BB_FSR3_UI_LAYER=1).
    const bool bb_fg_on = !frame->is_hdr && frame->fg.valid &&
                          (BB::FG::Dlss::Ready() ||
                           (BB::FG::Fsr3Active() && std::getenv("BB_FSR3_UI_LAYER") != nullptr));
    // bb-clean-3: opt-in (BB_HUD_FG=1). With DLSS-G x2 on (31 of 31 generated and presented)
    // the user saw no frame generation: the HUD-less diff can cover the whole picture (the copy
    // at the cut is not bit-equal to the finished frame), and the overlay then replaced every
    // generated frame with the real one. DLSS-G / FSR 3 get the HUD-less image as in 370.
    static const bool hud_fg = [] {
        const char* v = std::getenv("BB_HUD_FG");
        return v != nullptr && v[0] == '1';
    }();
    if (bb_fg_on && hud_fg) {
        bb_ui_layer = bb_hud_fg->Build(cmdbuf, frame->image, frame->image_view,
                                       swapchain.GetSurfaceFormat().format,
                                       {frame->width, frame->height},
                                       frame->fg.has_hudless ? frame->fg.hudless.view
                                                             : vk::ImageView{});
    }
    if (bb_fg_on) {
        bb_hud_fg->NoteFrame(true, BB::FG::Name(), false);
    }
    frame->has_ui_layer = bb_ui_layer != nullptr;
    if (bb_ui_layer) frame->ui_layer = *bb_ui_layer;

    // bb-clean-021-bbport: DLSS Frame Generation (Supermedo's PrepareFrame): the frames between
    // the previous frame and this one, from this frame (UI included) and its inputs.
    frame->generated_count = 0;
    if (BB::FG::Dlss::Ready() && !frame->is_hdr && frame->fg.valid) {
        // At most the frames the display can show between two of the game's.
        const u32 count = std::min(BbEnsureGenerated(frame, BB::FG::Dlss::WantedCount()),
                                   bb_frame_gen_limit.load(std::memory_order_relaxed));
        if (count && frame->unorm_view) {
            const auto range = [](vk::ImageAspectFlags a) {
                return vk::ImageSubresourceRange{a, 0, 1, 0, 1};
            };
            const auto depth_aspect = frame->fg.depth.format == vk::Format::eD32SfloatS8Uint ||
                                              frame->fg.depth.format == vk::Format::eD24UnormS8Uint
                                          ? vk::ImageAspectFlagBits::eDepth |
                                                vk::ImageAspectFlagBits::eStencil
                                          : vk::ImageAspectFlags{vk::ImageAspectFlagBits::eDepth};
            std::vector<vk::ImageMemoryBarrier2> to_general;
            const auto add = [&](vk::Image img, vk::ImageLayout from, vk::ImageAspectFlags a) {
                to_general.push_back({.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                      .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                      .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                      .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
                                                       vk::AccessFlagBits2::eMemoryWrite,
                                      .oldLayout = from,
                                      .newLayout = vk::ImageLayout::eGeneral,
                                      .image = img,
                                      .subresourceRange = range(a)});
            };
            add(frame->image, vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
            add(frame->fg.depth.image, vk::ImageLayout::eShaderReadOnlyOptimal, depth_aspect);
            add(frame->fg.motion.image, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::ImageAspectFlagBits::eColor);
            if (frame->fg.has_hudless) {
                add(frame->fg.hudless.image, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::ImageAspectFlagBits::eColor);
            }
            for (u32 i = 0; i < count; ++i) {
                add(frame->generated[i].image, vk::ImageLayout::eUndefined,
                    vk::ImageAspectFlagBits::eColor);
            }
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(to_general.size()),
                                     .pImageMemoryBarriers = to_general.data()});
            std::array<BB::FG::Image, 3> outputs{};
            for (u32 i = 0; i < count; ++i) {
                outputs[i] = {frame->generated[i].image, frame->generated[i].view,
                              frame->unorm_format, {frame->width, frame->height},
                              vk::ImageUsageFlagBits::eStorage};
            }
            const BB::FG::Image backbuffer{frame->image, frame->unorm_view, frame->unorm_format,
                                           {frame->width, frame->height},
                                           vk::ImageUsageFlagBits::eSampled};
            frame->generated_count = BB::FG::Dlss::Generate(cmdbuf, backbuffer, frame->fg,
                                                            outputs.data(), count,
                                                            instance.GetDevice());
            // A DLSS interposer may continue the recording in another command buffer.
            draw_scheduler.GetDynamicState().Invalidate();
            if (bb_ui_layer) {
                // bb-clean: the real frame's HUD over each generated one (never interpolated).
                const bool bgra = frame->unorm_format == vk::Format::eB8G8R8A8Unorm;
                for (u32 i = 0; i < frame->generated_count; ++i) {
                    bb_hud_fg->Overlay(draw_scheduler.CommandBuffer(), frame->generated[i].rgba_view,
                                       {frame->width, frame->height}, bgra);
                }
            }
            std::vector<vk::ImageMemoryBarrier2> back;
            for (auto b : to_general) {
                if (b.image == frame->image ||
                    std::any_of(frame->generated.begin(), frame->generated.end(),
                                [&](const Frame::Generated& g) { return g.image == b.image; })) {
                    continue; // left in General
                }
                std::swap(b.oldLayout, b.newLayout);
                back.push_back(b);
            }
            draw_scheduler.CommandBuffer().pipelineBarrier2(
                {.imageMemoryBarrierCount = u32(back.size()), .pImageMemoryBarriers = back.data()});
            bb_sec_generated_made += frame->generated_count;
        }
    } else if (!BB::FG::Dlss::Ready() && frame->generated[0].image) {
        BbDestroyGenerated(frame);
    }
    bb_sr->FrameEnd(draw_scheduler.CommandBuffer());
    // bb-velocity: the guest frame ended, after its compose pass (bb_sr->Run) was recorded.
    BB::ObjMv2::OnFlip();
    BB::NativeMv::OnPresent();

    DebugState.game_resolution = {image_size.width, image_size.height};
    DebugState.output_resolution = {frame->width, frame->height};

    if (pending_screenshot) {
        draw_scheduler.DeferPriorityOperation(
            [deferred_screenshot = std::move(pending_screenshot)]() {
                SavePendingScreenshot(deferred_screenshot.value());
            });
    }

    // Flush frame creation commands.
    frame->ready_semaphore = draw_scheduler.GetWorkSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    SubmitInfo info{};
    draw_scheduler.Flush(info);
    {
        // bb-clean-021 [perf]: frame begin stamp of the next frame (top of its command buffer),
        // the GPU times of a frame four flips back (no wait), and the CPU side of this one.
        bb_sr->FrameBegin(draw_scheduler.CommandBuffer());
        double gpu = 0, sr = 0, mv = 0;
        if (!bb_sr->Poll(gpu, sr, mv)) gpu = 0;
        const auto now = std::chrono::steady_clock::now();
        const double frame_ms =
            bb_last_flip.time_since_epoch().count() != 0
                ? std::chrono::duration<double, std::milli>(now - bb_last_flip).count()
                : 0.0;
        bb_last_flip = now;
        BB::Upscale::NoteFrame(frame_ms - bb_wait_ms, bb_wait_ms, gpu, sr, mv, bb_in_size.width,
                               bb_in_size.height, bb_sr_size.width, bb_sr_size.height,
                               frame->width, frame->height, bb_upscaled);
    }
    return frame;
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    // Request a free presentation frame.
    Frame* frame = GetRenderFrame();

    auto& scheduler = present_thread ? present_scheduler : draw_scheduler;
    scheduler.EndRendering();

    const auto cmdbuf = scheduler.CommandBuffer();

    constexpr vk::ImageSubresourceRange simple_subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const auto post_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const vk::RenderingAttachmentInfo attachment = {
        .imageView = frame->image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .extent = {frame->width, frame->height},
            },
        .layerCount = 1,
        .colorAttachmentCount = 1u,
        .pColorAttachments = &attachment,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    cmdbuf.beginRendering(rendering_info);
    cmdbuf.endRendering();

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &post_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetWorkSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return frame;
}

void Presenter::Present(Frame* frame, bool is_reusing_frame, bool is_game_frame) {
    // Free the frame for reuse
    const auto free_frame = [&] {
        if (!is_reusing_frame) {
            last_submit_frame = frame;
            std::scoped_lock fl{free_mutex};
            free_queue.push(frame);
            free_cv.notify_one();
        }
    };

    // Recreate the swapchain if the window was resized.
    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }

    // bb-clean-021-bbport: Supermedo's frame generation present loop (DLSS-G): the generated
    // frames first, then the game's own, evenly spaced over the interval between the game's
    // frames (the generated ones belong to the time before this frame). A frame shown again has
    // none.
    u32 bb_generated = is_reusing_frame ? 0 : frame->generated_count;
    const auto bb_start = std::chrono::steady_clock::now();
    if (!is_reusing_frame && is_game_frame) {
        const double ms =
            std::chrono::duration<double, std::milli>(bb_start - bb_last_game_present).count();
        if (ms > 1.0 && ms < 100.0) {
            bb_game_frame_ms = bb_game_frame_ms + (ms - bb_game_frame_ms) * 0.1;
        }
        bb_last_game_present = bb_start;
        ++bb_sec_game;
    }
    const auto bb_step = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double, std::milli>(bb_game_frame_ms / (bb_generated + 1)));
    // The game must never wait for generated frames: when every frame is in use they are
    // dropped and the game's frame goes out at once.
    const auto bb_behind = [this] {
        std::scoped_lock lock{free_mutex};
        return free_queue.empty();
    };
    if (bb_generated && bb_behind()) {
        bb_generated = 0;
    }
    const auto bb_wait_until = [](std::chrono::steady_clock::time_point at) {
        // The system sleep is coarse: sleep most of the way, then spin.
        constexpr auto spin = std::chrono::microseconds(1500);
        const auto now = std::chrono::steady_clock::now();
        if (at - now > spin) {
            std::this_thread::sleep_for(at - now - spin);
        }
        while (std::chrono::steady_clock::now() < at) {
            std::this_thread::yield();
        }
    };
    u32 bb_shown = 0;
    for (; bb_shown < bb_generated; ++bb_shown) {
        if (bb_shown > 0) {
            if (bb_behind()) break;
            bb_wait_until(bb_start + bb_step * bb_shown);
        }
        if (!BbPresentGenerated(frame, bb_shown, bb_shown == 0)) break;
    }
    if (bb_shown && !bb_behind()) {
        bb_wait_until(bb_start + bb_step * bb_shown);
    }
    bb_sec_generated_shown += bb_shown;

    if (!swapchain.AcquireNextImage()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
        if (!swapchain.AcquireNextImage()) {
            // User resizes the window too fast and GPU can't keep up. Skip this frame.
            LOG_WARNING(Render_Vulkan, "Skipping frame!");
            if (bb_shown) {
                // The frame's fence does not cover the generated frames' blits.
                present_scheduler.Finish();
            }
            free_frame();
            return;
        }
    }

    // Reset fence for queue submission. Do it here instead of GetRenderFrame() because we may
    // skip frame because of slow swapchain recreation. If a frame skip occurs, we skip signal
    // the frame's present fence and future GetRenderFrame() call will hang waiting for this frame.
    const auto reset_result = instance.GetDevice().resetFences(frame->present_done);
    ASSERT_MSG(reset_result == vk::Result::eSuccess,
               "Unexpected error resetting present done fence: {}", vk::to_string(reset_result));

    ImGuiID dockId = ImGui::Core::NewFrame(is_reusing_frame);
    BB::Upscale::DrawOverlay(); // bb-clean-021 F8 (nothing while hidden)

    const vk::Image swapchain_image = swapchain.Image();
    const vk::ImageView swapchain_image_view = swapchain.ImageView();

    auto& scheduler = present_scheduler;
    const auto cmdbuf = scheduler.CommandBuffer();
    const u32 capture_with_overlays_count = VideoCore::ConsumeWithOverlaysScreenshotRequests();
    std::optional<ScreenshotReadback> pending_screenshot;

    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
            .pLabelName = "Present",
        });
    }

    {
        auto* profiler_ctx = instance.GetProfilerContext();
        TracyVkNamedZoneC(profiler_ctx, renderer_gpu_zone, cmdbuf, "Host frame",
                          MarkersPalette::GpuMarkerColor, profiler_ctx != nullptr);

        const vk::Extent2D extent = swapchain.GetExtent();
        const std::array pre_barriers{
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = frame->image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
        };

        bool swapchain_copied_for_screenshot = false;

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput |
                                   vk::PipelineStageFlagBits::eFragmentShader,
                               {}, {}, {}, pre_barriers);

        { // Draw the game
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
            ImGui::SetNextWindowDockID(dockId, ImGuiCond_Once);
            if (ImGui::Begin("Display##game_display", nullptr, ImGuiWindowFlags_NoNav)) {
                auto game_texture = frame->imgui_texture;
                auto game_width = frame->width;
                auto game_height = frame->height;

                if (Libraries::SystemService::IsSplashVisible()) { // draw splash
                    if (!splash_img.has_value()) {
                        splash_img.emplace();
                        const auto& splash_data = Common::ElfInfo::Instance().GetSplashData();
                        if (!splash_data.empty()) {
                            splash_img = ImGui::RefCountedTexture::DecodePngTexture(splash_data);
                        }
                    }
                    if (auto& splash_image = this->splash_img.value()) {
                        auto [im_id, width, height] = splash_image.GetTexture();
                        game_texture = im_id;
                        game_width = width;
                        game_height = height;
                    }
                }

                ImVec2 contentArea = ImGui::GetContentRegionAvail();
                {
                    // bb: the game area should be the whole window (was 2px short: see imgui_core.cpp).
                    static s32 said_w = -1, said_h = -1;
                    if ((s32)contentArea.x != said_w || (s32)contentArea.y != said_h) {
                        said_w = (s32)contentArea.x;
                        said_h = (s32)contentArea.y;
                        BB::Upscale::Checkpoint(fmt::format("game area {}x{} in a {}x{} swapchain", said_w,
                                                            said_h, extent.width, extent.height));
                    }
                }
                SetExpectedGameSize((s32)contentArea.x, (s32)contentArea.y);

                const auto imgRect =
                    FitImage(game_width, game_height, (s32)contentArea.x, (s32)contentArea.y);
                ImVec2 offset{
                    static_cast<float>(imgRect.offset.x),
                    static_cast<float>(imgRect.offset.y),
                };
                ImVec2 size{
                    static_cast<float>(imgRect.extent.width),
                    static_cast<float>(imgRect.extent.height),
                };

                ImGui::SetCursorPos(ImGui::GetCursorStartPos() + offset);
                ImGui::Image(game_texture, size);

                if (EmulatorSettings.IsNullGPU()) {
                    Core::Devtools::Layer::DrawNullGpuNotice();
                }
            }
            ImGui::End();
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor();
        }
        ImGui::Core::Render(cmdbuf, swapchain_image_view, swapchain.GetExtent());
        // bb-dlssg-ui: anything ImGui drew besides the game image (the port's menus, the F8
        // overlay): DLSS-G's generated frames are blitted without it, so it flickered at 30 Hz.
        // While it is up the game's frames go out alone (FSR 3 draws it into its own).
        if (const ImDrawData* dd = ImGui::GetDrawData(); dd != nullptr) {
            bb_imgui_ui_shown = dd->TotalVtxCount > 48;
        }

        if (capture_with_overlays_count > 0) {
            auto& readback = pending_screenshot.emplace(
                instance, ScreenshotKind::WithOverlays,
                BuildScreenshotPaths(ScreenshotKind::WithOverlays, capture_with_overlays_count),
                extent.width, extent.height,
                swapchain.GetHDR() ? vk::Format::eA2B10G10R10UnormPack32
                                   : swapchain.GetSurfaceFormat().format,
                swapchain.GetHDR());

            const vk::ImageMemoryBarrier to_transfer{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, to_transfer);
            CopyImageToReadback(cmdbuf, swapchain_image, vk::ImageLayout::eTransferSrcOptimal,
                                readback);
            swapchain_copied_for_screenshot = true;
        }

        const vk::AccessFlags post_src_access_mask =
            swapchain_copied_for_screenshot ? vk::AccessFlagBits::eTransferRead
                                            : vk::AccessFlagBits::eColorAttachmentWrite;
        const vk::ImageLayout post_old_layout = swapchain_copied_for_screenshot
                                                    ? vk::ImageLayout::eTransferSrcOptimal
                                                    : vk::ImageLayout::eColorAttachmentOptimal;
        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = post_src_access_mask,
            .dstAccessMask = vk::AccessFlagBits::eNone,
            .oldLayout = post_old_layout,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);

        if (profiler_ctx) {
            TracyVkCollect(profiler_ctx, cmdbuf);
        }
    }
    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }

    // Flush vulkan commands.
    if (pending_screenshot) {
        scheduler.DeferPriorityOperation([deferred_screenshot = std::move(pending_screenshot)]() {
            SavePendingScreenshot(deferred_screenshot.value());
        });
    }

    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    info.AddWait(frame->ready_semaphore, frame->ready_tick);
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    info.AddSignal(frame->present_done);
    scheduler.Flush(info);

    // bb-clean-021-fg: FSR 3 frame generation's configure and prepare for this present (the
    // frame's own inputs for a new game frame; none for a repeat). Nothing with BB_FG off.
    if (BB::FG::Fsr3Active()) {
        const bool fg_frame = !is_reusing_frame && is_game_frame && frame->fg.valid;
        // bb-clean: this frame's HUD layer as the FG swapchain's UI resource (composited over the
        // real and the generated frames after interpolation; registered in BeforePresent).
        // bb-clean-2: only opt-in (BB_FSR3_UI_LAYER=1) and only at the backbuffer's exact size:
        // 854db18 hung with FSR 3 x2 right after registering a 2560x1438 layer on a 2560x1440
        // FG swapchain. By default FSR 3 gets the HUD-less colour (HUDLessColor) only.
        static const bool fsr3_ui = [] {
            const char* v = std::getenv("BB_FSR3_UI_LAYER");
            return v != nullptr && v[0] == '1';
        }();
        if (fsr3_ui && fg_frame && frame->has_ui_layer &&
            frame->ui_layer.extent == swapchain.GetExtent()) {
            BB::FG::SetUiLayer(&frame->ui_layer);
            bb_hud_fg->NoteFrame(false, BB::FG::Name(), true);
        }
        BB::FG::BeforePresent(fg_frame ? &frame->fg : nullptr, swapchain.GetHandle(),
                              swapchain.GetExtent(),
                              swapchain.GetHDR() ? vk::Format::eA2B10G10R10UnormPack32
                                                 : swapchain.GetSurfaceFormat().format);
    }

    // Present to swapchain.
    {
        // (FSR 3's frame interpolation swapchain submits to the game queue itself, under this
        // lock: not held here then.)
        std::unique_lock submit_lock{Scheduler::submit_mutex, std::defer_lock};
        if (!BB::FG::Fsr3Active()) {
            submit_lock.lock();
        }
        if (!swapchain.Present()) {
            swapchain.Recreate(window.GetWidth(), window.GetHeight());
        }
    }

    ++bb_sec_presents;
    if (!is_reusing_frame && is_game_frame) {
        BbUpdateFrameGenLimit(BB::FG::Dlss::Ready(),
                              std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - bb_start)
                                  .count());
    }
    {
        // bb-clean-021-bbport: one line a second confirming each piece: HUD cut, HUD-less copy
        // given to frame generation, DLSS-G frames presented, real and shown fps (measured).
        const auto now = std::chrono::steady_clock::now();
        if (bb_sec_start.time_since_epoch().count() == 0) bb_sec_start = now;
        const double sec = std::chrono::duration<double>(now - bb_sec_start).count();
        if (sec >= 1.0 && (BB::Upscale::Enabled() || BB::FG::On())) {
            const auto cut = rasterizer->bb_cut.TakeCounters();
            const u32 game = bb_sec_game.exchange(0), presents = bb_sec_presents.exchange(0);
            const u32 made = bb_sec_generated_made.exchange(0);
            const u32 shown = bb_sec_generated_shown.exchange(0);
            const u32 frames_cut = bb_sec_cut.exchange(0), hudless = bb_sec_hudless.exchange(0);
            const u32 hudless_failed = bb_sec_hudless_failed.exchange(0);
            BB::Upscale::Checkpoint(fmt::format(
                "[bbport] 1 s: HUD cut {} ({} of {} frames shown cut; {} cuts, {} UI-only, {} "
                "display copies; HUD layer: {} cuts, {} drawn over, {} failed) | HUD-less to FG {} "
                "({} copies at the cut, {} given, {} "
                "failed) | DLSS-G {} ({} generated, {} presented, limit {}) | real {:.1f} fps, "
                "shown {:.1f} fps",
                frames_cut ? "yes" : "no", frames_cut, game, cut.cuts, cut.ui_only, cut.displays,
                cut.layer_cuts, cut.layer_composites, cut.layer_failed, hudless ? "yes" : "no", cut.hudless, hudless, hudless_failed,
                BB::FG::Dlss::Ready() ? (shown ? "on" : "idle") : "off", made, shown,
                bb_frame_gen_limit.load(), game / sec, (presents + shown) / sec));
            bb_hud_fg->LogSecond(sec);
            bb_hud_fg->LogHudlessSecond(sec);
            bb_sec_start = now;
        }
    }
    free_frame();
    if (!is_reusing_frame && is_game_frame) {
        DebugState.IncFlipFrameNum();
    }
}

Frame* Presenter::GetRenderFrame() {
    // Wait for free presentation frames
    Frame* frame;
    {
        std::unique_lock lock{free_mutex};
        free_cv.wait(lock, [this] { return !free_queue.empty(); });
        LOG_DEBUG(Render_Vulkan, "Got render frame, remaining {}", free_queue.size() - 1);

        // Take the frame from the queue
        frame = free_queue.front();
        free_queue.pop();
    }

    const vk::Device device = instance.GetDevice();
    vk::Result result{};

    const auto wait = [&]() {
        result = device.waitForFences(frame->present_done, false, std::numeric_limits<u64>::max());
        return result;
    };

    // Wait for the presentation to be finished so all frame resources are free
    while (wait() != vk::Result::eSuccess) {
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
        // Retry if the waiting times out
        if (result == vk::Result::eTimeout) {
            continue;
        }
    }

    if (frame->width != expected_frame_width || frame->height != expected_frame_height ||
        frame->is_hdr != swapchain.GetHDR()) {
        RecreateFrame(frame, expected_frame_width, expected_frame_height);
    }

    return frame;
}

void Presenter::SetExpectedGameSize(s32 width, s32 height) {
    const float ratio = (float)width / (float)height;

    expected_frame_height = height;
    expected_frame_width = width;
    if (ratio > expected_ratio) {
        expected_frame_width = static_cast<s32>(std::lround(height * expected_ratio));
    } else {
        expected_frame_height = static_cast<s32>(std::lround(width / expected_ratio));
    }
    // bb-clean-2: the game's render size (1708x960) is not exactly 16:9, which made a 2560x1438
    // frame in a 2560x1440 window: the output-size HUD (display copy 2560x1440) was resampled to
    // 1438 rows and again to 1440 by the swapchain (soft HUD). Within 0.5 % of the window: the
    // window's own size, pixel for pixel.
    if (std::abs(static_cast<s32>(expected_frame_width) - width) * 200 <= width) expected_frame_width = width;
    if (std::abs(static_cast<s32>(expected_frame_height) - height) * 200 <= height) expected_frame_height = height;
}

} // namespace Vulkan
