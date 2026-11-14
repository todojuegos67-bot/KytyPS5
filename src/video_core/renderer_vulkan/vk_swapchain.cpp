// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include <cstdlib>
#include <string>
#include "video_core/bb/bb_fg.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

namespace Vulkan {

static constexpr vk::SurfaceFormatKHR SURFACE_FORMAT_HDR = {
    .format = vk::Format::eA2B10G10R10UnormPack32,
    .colorSpace = vk::ColorSpaceKHR::eHdr10St2084EXT,
};

Swapchain::Swapchain(const Instance& instance_, const Frontend::WindowSDL& window_)
    : instance{instance_}, window{window_}, surface{CreateSurface(instance.GetInstance(), window)} {
    FindPresentFormat();
    FindPresentMode();

    Create(window.GetWidth(), window.GetHeight());
    ImGui::Core::Initialize(instance, window, image_count, surface_format.format);
}

Swapchain::~Swapchain() {
    Destroy();
    instance.GetInstance().destroySurfaceKHR(surface);
}

void Swapchain::Create(u32 width_, u32 height_) {
    width = width_;
    height = height_;
    needs_recreation = false;

    Destroy();

    SetSurfaceProperties();

    // bb-alttab: a minimized window (Alt+Tab out of fullscreen) has a 0x0 surface. A swapchain
    // made then (the FSR 3 FG one accepted it) stayed 0x0 after the window came back, because the
    // window's own size never changed: black screen with sound. Make none now; the presenter
    // skips frames until the surface has a size again and then makes it (SurfaceChanged()).
    if (extent.width == 0 || extent.height == 0) {
        needs_recreation = true;
        LOG_INFO(Render_Vulkan, "Swapchain not made: the surface is 0x0 (window minimized), "
                                "waiting for it to come back ({}x{} window)",
                 width, height);
        return;
    }

    const std::array queue_family_indices = {
        instance.GetGraphicsQueueFamilyIndex(),
        instance.GetPresentQueueFamilyIndex(),
    };

    const bool exclusive = queue_family_indices[0] == queue_family_indices[1];
    const u32 queue_family_indices_count = exclusive ? 1u : 2u;
    const vk::SharingMode sharing_mode =
        exclusive ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent;
    const auto format = needs_hdr ? SURFACE_FORMAT_HDR : surface_format;
    const vk::SwapchainCreateInfoKHR swapchain_info = {
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = sharing_mode,
        .queueFamilyIndexCount = queue_family_indices_count,
        .pQueueFamilyIndices = queue_family_indices.data(),
        .preTransform = transform,
        .compositeAlpha = composite_alpha,
        .presentMode = present_mode,
        .clipped = true,
        .oldSwapchain = nullptr,
    };

    // bb-clean-021-fg: FSR 3 frame generation makes its frame interpolation swapchain instead
    // (BB_FG=amd-x2; nothing else with BB_FG off).
    vk::SwapchainKHR fg_chain{};
    if (BB::FG::CreateSwapchain(swapchain_info, &fg_chain)) {
        swapchain = fg_chain;
    } else {
        auto [swapchain_result, chain] = instance.GetDevice().createSwapchainKHR(swapchain_info);
        ASSERT_MSG(swapchain_result == vk::Result::eSuccess, "Failed to create swapchain: {}",
                   vk::to_string(swapchain_result));
        swapchain = chain;
    }

    SetupImages();
    RefreshSemaphores();
}

void Swapchain::Recreate(u32 width_, u32 height_) {
    LOG_DEBUG(Render_Vulkan, "Recreate the swapchain: width={} height={} HDR={}", width_, height_,
              needs_hdr);
    Create(width_, height_);
    if (swapchain) {
        LOG_INFO(Render_Vulkan, "Swapchain made {}x{} (window {}x{}, HDR {}, FSR 3 FG {})",
                 extent.width, extent.height, width_, height_, needs_hdr, BB::FG::Fsr3Active());
    }
}

bool Swapchain::SurfaceChanged() const {
    const auto [result, caps] = instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(surface);
    if (result != vk::Result::eSuccess) {
        return !swapchain;
    }
    const vk::Extent2D cur = caps.currentExtent;
    if (cur.width == std::numeric_limits<u32>::max()) {
        return !swapchain; // the surface takes the swapchain's size
    }
    if (cur.width == 0 || cur.height == 0) {
        return false; // still minimized: nothing to make yet
    }
    return !swapchain || cur.width != extent.width || cur.height != extent.height;
}

void Swapchain::SetHDR(bool hdr) {
    if (needs_hdr == hdr) {
        return;
    }

    auto result = instance.GetDevice().waitIdle();
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(ImGui, "Failed to wait for Vulkan device idle on mode change: {}",
                    vk::to_string(result));
    }

    needs_hdr = hdr;
    Recreate(width, height);
    ImGui::Core::OnSurfaceFormatChange(needs_hdr ? SURFACE_FORMAT_HDR.format
                                                 : surface_format.format);
}

bool Swapchain::AcquireNextImage() {
    if (!swapchain) {
        needs_recreation = true;
        return false;
    }
    vk::Device device = instance.GetDevice();
    vk::Result result =
        BB::FG::Fsr3Active()
            ? BB::FG::AcquireNextImage(swapchain, image_acquired[frame_index], &image_index)
            : device.acquireNextImageKHR(swapchain, std::numeric_limits<u64>::max(),
                                         image_acquired[frame_index], VK_NULL_HANDLE,
                                         &image_index);

    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eSuboptimalKHR:
    case vk::Result::eErrorSurfaceLostKHR:
    case vk::Result::eErrorOutOfDateKHR:
    case vk::Result::eErrorUnknown:
        needs_recreation = true;
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Swapchain acquire returned unknown result {}",
                     vk::to_string(result));
        UNREACHABLE();
        break;
    }

    return !needs_recreation;
}

bool Swapchain::Present() {
    if (!swapchain) {
        needs_recreation = true;
        return false;
    }
    const vk::PresentInfoKHR present_info = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &present_ready[image_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    };

    auto result = BB::FG::Fsr3Active()
                      ? BB::FG::QueuePresent(instance.GetPresentQueue(), present_info)
                      : instance.GetPresentQueue().presentKHR(present_info);
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        needs_recreation = true;
    } else {
        ASSERT_MSG(result == vk::Result::eSuccess, "Swapchain presentation failed: {}",
                   vk::to_string(result));
    }

    frame_index = (frame_index + 1) % image_count;

    return !needs_recreation;
}

void Swapchain::FindPresentFormat() {
    const auto [formats_result, formats] =
        instance.GetPhysicalDevice().getSurfaceFormatsKHR(surface);
    ASSERT_MSG(formats_result == vk::Result::eSuccess, "Failed to query surface formats: {}",
               vk::to_string(formats_result));

    // Check if the device supports HDR formats. Here we care of Rec.2020 PQ only as it is expected
    // game output. Other variants as e.g. linear Rec.2020 will require additional color space
    // rotation
    supports_hdr =
        std::find_if(formats.begin(), formats.end(), [](const vk::SurfaceFormatKHR& format) {
            return format == SURFACE_FORMAT_HDR;
        }) != formats.end();
    // Also make sure that user allowed us to use HDR
    supports_hdr &= EmulatorSettings.IsHdrAllowed();

    // If there is a single undefined surface format, the device doesn't care, so we'll just use
    // RGBA sRGB.
    if (formats[0].format == vk::Format::eUndefined) {
        surface_format.format = vk::Format::eR8G8B8A8Unorm;
        surface_format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
        return;
    }

    // Try to find a suitable format.
    for (const vk::SurfaceFormatKHR& sformat : formats) {
        vk::Format format = sformat.format;
        if (format != vk::Format::eR8G8B8A8Unorm && format != vk::Format::eB8G8R8A8Unorm) {
            continue;
        }

        surface_format.format = format;
        surface_format.colorSpace = sformat.colorSpace;
        return;
    }

    UNREACHABLE_MSG("Unable to find required swapchain format!");
}

void Swapchain::FindPresentMode() {
    const auto [modes_result, modes] =
        instance.GetPhysicalDevice().getSurfacePresentModesKHR(surface);
    if (modes_result != vk::Result::eSuccess) {
        LOG_ERROR(Render, "Failed to query available present modes, falling back to Fifo as "
                          "guaranteed supported option.");
        present_mode = vk::PresentModeKHR::eFifo;
        return;
    }

    const auto requested_mode = EmulatorSettings.GetPresentMode();
    if (requested_mode == "Mailbox") {
        present_mode = vk::PresentModeKHR::eMailbox;
    } else if (requested_mode == "Fifo") {
        present_mode = vk::PresentModeKHR::eFifo;
    } else if (requested_mode == "Immediate") {
        present_mode = vk::PresentModeKHR::eImmediate;
    } else {
        LOG_ERROR(Render_Vulkan, "Unknown present mode {}, defaulting to Mailbox.",
                  EmulatorSettings.GetPresentMode());
        present_mode = vk::PresentModeKHR::eMailbox;
    }

    if (std::ranges::find(modes, present_mode) == modes.cend()) {
        // FIFO is guaranteed to be supported by the Vulkan spec.
        constexpr auto fallback = vk::PresentModeKHR::eFifo;
        LOG_WARNING(Render, "Requested present mode {} is not supported, falling back to {}.",
                    vk::to_string(present_mode), vk::to_string(fallback));
        present_mode = fallback;
    }
    // bb-clean-021-fg: FSR 3 FG paces its own presents (real + generated). Under FIFO the
    // present of the real frame waited a whole refresh (real fps halved, no gain over FG off):
    // with FG on, mailbox (else immediate) unless BB_FG_PRESENT=fifo.
    // bb-clean-021-bbport: DLSS-G presents its generated frames itself (Supermedo's loop): the
    // same choice.
    if ((BB::FG::Fsr3Requested() || BB::FG::Dlss::Requested()) &&
        (present_mode == vk::PresentModeKHR::eFifo ||
         present_mode == vk::PresentModeKHR::eFifoRelaxed)) {
        const char* v = std::getenv("BB_FG_PRESENT");
        const std::string want = v != nullptr ? v : "";
        const auto has = [&](vk::PresentModeKHR m) {
            return std::ranges::find(modes, m) != modes.cend();
        };
        vk::PresentModeKHR chosen = present_mode;
        if (want == "immediate" && has(vk::PresentModeKHR::eImmediate)) {
            chosen = vk::PresentModeKHR::eImmediate;
        } else if (want != "fifo") {
            chosen = has(vk::PresentModeKHR::eMailbox)     ? vk::PresentModeKHR::eMailbox
                     : has(vk::PresentModeKHR::eImmediate) ? vk::PresentModeKHR::eImmediate
                                                           : present_mode;
        }
        LOG_INFO(Render_Vulkan, "Frame generation: present mode {} (configured {}; BB_FG_PRESENT={})",
                 vk::to_string(chosen), vk::to_string(present_mode), want.empty() ? "auto" : want);
        present_mode = chosen;
    }
}

void Swapchain::SetSurfaceProperties() {
    const auto [capabilities_result, capabilities] =
        instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(surface);
    ASSERT_MSG(capabilities_result == vk::Result::eSuccess,
               "Failed to query surface capabilities: {}", vk::to_string(capabilities_result));

    extent = capabilities.currentExtent;
    if (capabilities.currentExtent.width == std::numeric_limits<u32>::max()) {
        extent.width = std::max(capabilities.minImageExtent.width,
                                std::min(capabilities.maxImageExtent.width, width));
        extent.height = std::max(capabilities.minImageExtent.height,
                                 std::min(capabilities.maxImageExtent.height, height));
    }

    // Select number of images in swap chain, we prefer one buffer in the background to work on
    image_count = capabilities.minImageCount + 1;
    // bb-clean-021-bbport (Supermedo): frame generation presents several images per game frame:
    // one more for them.
    if (BB::FG::Dlss::Requested()) {
        ++image_count;
    }
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    // Prefer identity transform if possible
    transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (!(capabilities.supportedTransforms & transform)) {
        transform = capabilities.currentTransform;
    }

    // Opaque is not supported everywhere.
    composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    if (!(capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque)) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
    }
}

void Swapchain::Destroy() {
    vk::Device device = instance.GetDevice();
    const auto wait_result = device.waitIdle();
    if (wait_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Failed to wait for device to become idle: {}",
                    vk::to_string(wait_result));
    }

    for (auto& image_view : images_view) {
        device.destroyImageView(image_view);
    }
    images_view.clear();

    if (swapchain) {
        if (BB::FG::Fsr3Active()) {
            BB::FG::DestroySwapchain(swapchain);
        } else {
            device.destroySwapchainKHR(swapchain);
        }
        swapchain = vk::SwapchainKHR{};
    }

    for (const auto& sem : image_acquired) {
        device.destroySemaphore(sem);
    }
    for (const auto& sem : present_ready) {
        device.destroySemaphore(sem);
    }

    image_acquired.clear();
    present_ready.clear();
}

void Swapchain::RefreshSemaphores() {
    const vk::Device device = instance.GetDevice();
    image_acquired.resize(image_count);
    present_ready.resize(image_count);

    for (vk::Semaphore& semaphore : image_acquired) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create image acquired semaphore: {}",
                   vk::to_string(semaphore_result));
        semaphore = sem;
    }
    for (vk::Semaphore& semaphore : present_ready) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create present ready semaphore: {}", vk::to_string(semaphore_result));
        semaphore = sem;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, image_acquired[i], "Swapchain Semaphore: image_acquired {}", i);
        SetObjectName(device, present_ready[i], "Swapchain Semaphore: present_ready {}", i);
    }
}

void Swapchain::SetupImages() {
    vk::Device device = instance.GetDevice();
    if (BB::FG::Fsr3Active()) {
        u32 count = 0;
        auto result = BB::FG::GetSwapchainImages(swapchain, &count, nullptr);
        std::vector<vk::Image> imgs(count);
        if (result == vk::Result::eSuccess) {
            result = BB::FG::GetSwapchainImages(swapchain, &count, imgs.data());
        }
        ASSERT_MSG(result == vk::Result::eSuccess, "Failed to get the FSR 3 swapchain images: {}",
                   vk::to_string(result));
        imgs.resize(count);
        images = std::move(imgs);
    } else {
        auto [images_result, imgs] = device.getSwapchainImagesKHR(swapchain);
        ASSERT_MSG(images_result == vk::Result::eSuccess, "Failed to create swapchain images: {}",
                   vk::to_string(images_result));
        images = std::move(imgs);
    }
    image_count = static_cast<u32>(images.size());
    images_view.resize(image_count);
    for (u32 i = 0; i < image_count; ++i) {
        if (images_view[i]) {
            device.destroyImageView(images_view[i]);
        }
        auto [im_view_result, im_view] = device.createImageView(vk::ImageViewCreateInfo{
            .image = images[i],
            .viewType = vk::ImageViewType::e2D,
            .format = needs_hdr ? SURFACE_FORMAT_HDR.format : surface_format.format,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .levelCount = 1,
                    .layerCount = 1,
                },
        });
        ASSERT_MSG(im_view_result == vk::Result::eSuccess, "Failed to create image view: {}",
                   vk::to_string(im_view_result));
        images_view[i] = im_view;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, images[i], "Swapchain Image {}", i);
        SetObjectName(device, images_view[i], "Swapchain ImageView {}", i);
    }
}

} // namespace Vulkan
