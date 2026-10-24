#ifndef EMULATOR_SRC_LOCAL_METRICS_H_
#define EMULATOR_SRC_LOCAL_METRICS_H_
// bruno-astro-1025: the todojuegos67-bot fork's log metrics that its metrics overlay (metricas.ps1) and its users'
// A/B runs read, on brunoKytyPs5's presenter:
// - KYTY_FPS_LOG=1 (the configs set it): once a second, at a guest flip,
//     [fps] 60 frames: mean 16.7 ms, max 18.0; over 17 ms: 3, over 25 ms: 0; flip->present mean 4.10 sd 0.80 ms,
//     present interval mean 16.67 sd 0.40 ms
//   (the game's frames: guest flip to guest flip; flip->present: a frame's guest flip to its vkQueuePresentKHR return;
//   present interval: between those presents of new frames).
// - KYTY_LATENCY=1 (the launcher's metrics option): input -> present of that real frame, once a second,
//     [lat] input->present mean 38.2 ms, p95 45.1, max 52.0; +scanout 2.1; fg +16.7 (n 60); total 57.0, p95 63.9;
//     mode off, fg x2
//   input: the game's first controller read (scePadRead / scePadReadState) after its last flip; present: the return of
//   vkQueuePresentKHR of the frame that flip showed (with DLSS-G / FSR 3 the real frame handed to the generating
//   swapchain). +scanout: half the display's refresh period (an estimate); fg: the frame generation hold, estimated as
//   the real present interval x g / (g + 1) for g generated frames per real one.
// Off: a relaxed load per call.
#include <cstdint>

namespace Kyty::Metrics {

[[nodiscard]] bool FpsLogOn();
[[nodiscard]] bool LatencyOn();
// Guest threads: the game reads the controller (controller.cpp).
void OnPadRead();
// Renderer thread, at the guest flip (Presenter::PrepareFrame): the frame's input stamp (0: none) and flip time.
struct FrameStamp {
	int64_t input_ns = 0;
	int64_t flip_ns  = 0;
};
[[nodiscard]] FrameStamp OnGuestFlip();
// Present thread, right after vkQueuePresentKHR of a new frame returned successfully; `generated`: frame generation
// frames per real one now (0: none).
void OnPresented(const FrameStamp& stamp, int generated);

} // namespace Kyty::Metrics

#endif // EMULATOR_SRC_LOCAL_METRICS_H_
