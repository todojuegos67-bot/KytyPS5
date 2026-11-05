// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/videoout/driver.h"
#include "video_core/bb/bb_antilag.h"
#include "video_core/bb/bb_fg.h"
#include "core/libraries/videoout/videoout_error.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Libraries::VideoOut {

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    if (main_port.is_open) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.is_open = true;
    liverpool->SetVoPort(&main_port);
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock lock{mutex};

    // Mark as closed
    main_port.is_open = false;
    main_port.flip_rate = 0;
    main_port.prev_index = -1;

    // Clear port information
    std::memset(main_port.buffer_labels.data(), 0, sizeof(main_port.buffer_labels));
    std::memset(main_port.groups.data(), 0, sizeof(main_port.groups));
    std::memset(&main_port.vblank_status, 0, sizeof(main_port.vblank_status));
    main_port.flip_status = FlipStatus{};

    // Re-initialize buffers
    std::memset(main_port.buffer_slots.data(), 0, sizeof(main_port.buffer_slots));
    for (auto& buffer : main_port.buffer_slots) {
        buffer.group_index = -1;
    }

    // Clear events
    for (auto event : main_port.flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.flip_events.clear();
    for (auto event : main_port.vblank_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.vblank_events.clear();
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };

        // Reset flip label also when registering buffer
        port->buffer_labels[startIndex + i] = 0;
        port->SignalVoLabel();

        presenter->RegisterVideoOutSurface(group, address);
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

int VideoOutDriver::ChangeBufferAttribute(VideoOutPort* port, s32 attributeIndex,
                                          const BufferAttribute* attribute) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "attributeIndex = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             attributeIndex, GetPixelFormatString(attribute->pixel_format), attribute->aspect_ratio,
             static_cast<u32>(attribute->tiling_mode), attribute->width, attribute->height,
             attribute->pitch_in_pixel, attribute->option);

    std::unique_lock lock{port->port_mutex};
    std::memcpy(&port->groups[attributeIndex].attrib, attribute, sizeof(BufferAttribute));
    return 0;
}

void VideoOutDriver::Flip(const Request& req) {
    // Update HDR status before presenting.
    presenter->SetHDR(req.port->is_hdr);

    // Present the frame (BB::AntiLag: its guest flip's present markers around it).
    const u64 lat_index = req.frame != nullptr ? req.frame->lat_index : 0;
    BB::AntiLag::OnPresent(lat_index);
    BB::FG::SetPresentSlot({req.pace_qpc, req.pace_interval});
    presenter->Present(req.frame);
    BB::FG::SetPresentSlot({});
    BB::AntiLag::AfterPresent(lat_index);
    if (BB::AntiLag::ReflexActive() && req.frame != nullptr) {
        const int vblanks = req.port->flip_rate + 1;
        double game_ms =
            vblanks > 1 ? 1000.0 / EmulatorSettings.GetVblankFrequency() * vblanks : 0.0;
        // 0.72 one limiter: when the frame generation pacer (PaceGuestFlip, BB_FG_PACE=guest)
        // already holds the flips to an interval at least as long as the game's own, it stays the
        // only limit (0.70 pacing and the FSR 3 midpoint slots unchanged) and Reflex gets no
        // minimum interval; otherwise the pacer never waits and Reflex holds the game's cadence.
        const double pace_ms = BB::FG::PacerIntervalMs();
        const bool pacer_holds = pace_ms > 0.0 && game_ms > 0.0 && pace_ms >= game_ms * 0.995;
        static int said = -1;
        if (game_ms > 0.0 && static_cast<int>(pacer_holds) != said) {
            said = pacer_holds ? 1 : 0;
            BB::FG::Printf("[reflex] one limiter: %s (game flip interval %.2f ms, pacer grid %.2f ms)",
                           pacer_holds ? "the frame generation pacer holds the flips; Reflex game limit off"
                                       : "Reflex holds the game's flip interval; the pacer grid is shorter",
                           game_ms, pace_ms);
        }
        if (pacer_holds) {
            game_ms = 0.0;
        }
        BB::AntiLag::SetGameInterval(game_ms);
    }
    if (BB::AntiLag::PaceOn() || BB::AntiLag::PhaseOn() || BB::AntiLag::PadLateOn()) {
        // KYTY_LOWLAT pacing / Anti-Lag+ pad late (the DS videoOut rule): the game's own flip
        // interval (flip rate + 1 vblanks) when it is above one vblank, else 0 (the pacing off).
        const double vblank_ms = 1000.0 / EmulatorSettings.GetVblankFrequency();
        const int vblanks = req.port->flip_rate + 1;
        BB::AntiLag::SetPaceCap(req.frame != nullptr && vblanks > 1 ? vblank_ms * vblanks : 0.0,
                                vblank_ms);
    }
    BB::AntiLag::OnFlipReleased(lat_index);
    if (BB::AntiLag::PhaseOn()) {
        BB::AntiLag::PhaseRelease(-1.0);
    }
    BB::AntiLag::OnPresented(lat_index);

    // Update flip status.
    auto* port = req.port;
    {
        std::unique_lock lock{port->port_mutex};
        auto& flip_status = port->flip_status;
        flip_status.count++;
        flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        flip_status.flip_arg = req.flip_arg;
        flip_status.current_buffer = req.index;
        if (req.eop) {
            --flip_status.gc_queue_num;
        }
        --flip_status.flip_pending_num;
    }

    // Trigger flip events for the port.
    for (auto event : port->flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    // Reset prev flip label
    if (port->prev_index != -1) {
        port->buffer_labels[port->prev_index] = 0;
        port->SignalVoLabel();
    }
    // save to prev buf index
    port->prev_index = req.index;
}

void VideoOutDriver::DrawBlankFrame() {
    const auto empty_frame = presenter->PrepareBlankFrame(true);
    presenter->Present(empty_frame, false, false);
}

void VideoOutDriver::DrawLastFrame() {
    const auto frame = presenter->PrepareLastFrame();
    if (frame != nullptr) {
        presenter->Present(frame, true);
    }
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg,
                                bool is_eop /*= false*/) {
    {
        std::unique_lock lock{port->port_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        ++port->flip_status.flip_pending_num; // integral GPU and CPU pending flips counter
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
    }
    // Frame generation: the real frames held to its limit at the game's own flip (bb_fg.h); the
    // flip's grid slot goes with its request (the frame generation presents).
    BB::FG::PresentSlot pace{};
    if (index != -1) {
        pace = BB::FG::PaceGuestFlip();
        BB::FG::GameFlipReserved();
    }
    // BB::AntiLag: the game's CPU frame ends at its flip.
    const u64 lat_index = BB::AntiLag::OnFlipReserved();

    if (!is_eop) {
        // Non EOP flips can arrive from any thread so ask GPU thread to perform them
        liverpool->SendCommand([=, this]() {
            SubmitFlipInternal(port, index, flip_arg, is_eop, lat_index, pace.qpc, pace.interval);
        });
    } else {
        SubmitFlipInternal(port, index, flip_arg, is_eop, lat_index, pace.qpc, pace.interval);
    }

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop,
                                        u64 lat_index, s64 pace_qpc, s64 pace_interval) {
    Vulkan::Frame* frame;
    if (index == -1) {
        frame = presenter->PrepareBlankFrame(false);
    } else {
        const auto& buffer = port->buffer_slots[index];
        ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
        const auto& group = port->groups[buffer.group_index];
        frame = presenter->PrepareFrame(group, buffer.address_left);
    }
    if (frame != nullptr) {
        frame->lat_index = lat_index;
        BB::AntiLag::OnFlipReady(lat_index);
        // KYTY_LOWLAT (Anti-Lag / Anti-Lag+): this frame's last host GPU tick (the DS
        // FlipPreparation hook); the timeline wait at the next controller read uses it.
        if (BB::AntiLag::OwnLowLatencyOn() && frame->ready_tick > 0) {
            BB::AntiLag::OnFrameSubmitted(VK_NULL_HANDLE,
                                          static_cast<VkSemaphore>(frame->ready_semaphore),
                                          frame->ready_tick);
        }
    }

    {
        std::scoped_lock lock{mutex};
        requests.push({
            .frame = frame,
            .port = port,
            .flip_arg = flip_arg,
            .index = index,
            .eop = is_eop,
            .pace_qpc = pace_qpc,
            .pace_interval = pace_interval,
        });
    }
    request_cv.notify_one();
}

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 /
                                                 EmulatorSettings.GetVblankFrequency());

    Common::SetCurrentThreadName("shadPS4:PresentThread");
    Common::SetCurrentThreadRealtime(vblank_period);

    Common::AccurateTimer timer{vblank_period};

    const auto receive_request = [this] -> Request {
        std::scoped_lock lk{mutex};
        if (!requests.empty()) {
            const auto request = requests.front();
            requests.pop();
            return request;
        }
        return {};
    };

    // BB::FG::PresentOnArrival (bb-pace-049): under the real fps cap the game's flips are already
    // held to their grid at the submit (PaceGuestFlip), so a flip is presented as it arrives
    // instead of at the next emulated vblank (240 Hz: up to 4.17 ms later on the vblank timer's own
    // phase, 16.67 + 4.17 = 20.8 ms intervals); the vblanks keep their own timer.
    const auto wait_tick_or_flip = [&](std::chrono::nanoseconds wait) {
        const auto deadline = std::chrono::steady_clock::now() + wait;
        while (!token.stop_requested()) {
            Request request{};
            bool got = false;
            {
                std::unique_lock lk{mutex};
                request_cv.wait_until(lk, deadline, [&] {
                    return !requests.empty() || token.stop_requested();
                });
                if (!requests.empty()) {
                    request = requests.front();
                    requests.pop();
                    got = true;
                }
            }
            if (!got) {
                break;
            }
            if (request) {
                Flip(request);
                FRAME_END;
            }
        }
        // The condition variable's timeout is as coarse as the system timer: the rest accurately.
        const auto left = deadline - std::chrono::steady_clock::now();
        if (left > std::chrono::nanoseconds{0} && !token.stop_requested()) {
            Common::AccurateSleep(std::chrono::duration_cast<std::chrono::nanoseconds>(left),
                                  nullptr, false);
        }
    };

    while (!token.stop_requested()) {
        if (BB::FG::PresentOnArrival() && main_port.flip_rate == 0 && main_port.is_open &&
            !DebugState.IsGuestThreadsPaused()) {
            timer.Start(wait_tick_or_flip);
        } else {
            timer.Start();
        }

        if (DebugState.IsGuestThreadsPaused()) {
            DrawLastFrame();
            timer.End();
            continue;
        }

        // Check if it's time to take a request.
        auto& vblank_status = main_port.vblank_status;
        if (vblank_status.count % (main_port.flip_rate + 1) == 0) {
            const auto request = receive_request();
            if (!request) {
                if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                    if (!main_port.is_open) {
                        DrawBlankFrame();
                    } else if (ImGui::Core::MustKeepDrawing()) {
                        DrawLastFrame();
                    }
                }
            } else {
                Flip(request);
                FRAME_END;
            }
        }

        {
            // Needs lock here as can be concurrently read by `sceVideoOutGetVblankStatus`
            std::scoped_lock lock{main_port.vo_mutex};

            // Trigger flip events for the port
            for (auto event : main_port.vblank_events) {
                auto equeue = Kernel::GetEqueue(event);
                if (equeue != nullptr) {
                    equeue->TriggerEvent(
                        static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                        reinterpret_cast<void*>(
                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                            (vblank_status.count << 16)));
                }
            }

            // Update vblank status
            vblank_status.count++;
            vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
        }

        timer.End();
    }
}

} // namespace Libraries::VideoOut
