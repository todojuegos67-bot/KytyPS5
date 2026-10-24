#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_

#include "common/common.h"
// #include "common/subsystems.h"

#include "common/abi.h"
#include "kernel/eventQueue.h"

#include <chrono>
#include <memory>

namespace Libs::Graphics {
class CommandBuffer;
class Presenter;
} // namespace Libs::Graphics

namespace Libs::VideoOut {

// A movie frame was decoded (VideoDec2, AvPlayer): flips keep 60 per second meanwhile.
void NoteMovieFrame() noexcept;
// A read of one of the game's cinematic files (`name` on its open, logged): MoviePlaying() while they are read.
void NoteCinematic(const char* name) noexcept;
// The game opened one of a world's files (levels/...): the next 3D view after it is the world (the world gate).
void NoteWorldFile(const char* name) noexcept;
// A movie frame was decoded within the last half second.
bool MoviePlaying() noexcept;
// An in-engine cinematic's files were read lately (KYTY_CINEMATIC_HOLD_MS): its frames keep 60.
bool CinematicPlaying() noexcept;
// A held in-engine cutscene (KYTY_CUTSCENE_HOLD; until the player moves after its files were read).
bool CutsceneHeld() noexcept;
// Dynamic frame generation (10-21): the game's 3D view flipped at most `fps` a second, switched at runtime (0: none).
// Takes effect only when the virtual vblank is a whole multiple of fps (840 Hz: 60 = 14 vblanks, 56 = 15), and never
// during cinematics, movies or menus. Returns the fps held (0 when refused or none).
int SetDynamicFlipFps(int fps) noexcept;
// The largest whole fps f, at least 54 and at most 60, that divides the virtual vblank evenly and with f * multiplier
// at most target + 1 (0: none; 840 Hz, 224 target: x3 -> 60, x4 -> 56).
int ReachFps(double target, int multiplier) noexcept;
// The first button the player pressed (controller.cpp): until then (the PlayStation Studios logo, the intro and the
// title screen's "press a button") flips keep 60 a second above 60 Hz and frame generation stays off.
void NotePlayerInput() noexcept;
// Any button or a stick off centre (controller.cpp): ends a held cutscene (CutsceneHeld).
void NotePlayerActivity() noexcept;
bool PlayerInputSeen() noexcept;
// The player's first button was pressed (NotePlayerInput), whatever the world gate.
bool PlayerButtonSeen() noexcept;
// KYTY_PACE_STATS (default 1): every 5 s in a world, how evenly the game asks for flips (kind 0) and how evenly
// they reach the screen (kind 1), and how far the two disagree frame by frame (what the eye sees as roughness).
void NotePace(int kind) noexcept;
// KYTY_FLIP_WHEN_READY=1: guest flips presented when their GPU work completes, not on virtual vblank ticks.
bool FlipWhenReady() noexcept;
// [fps] diagnostics: guest flip submit -> vkQueuePresentKHR return, and the interval between those presents.
void NoteFlipSubmitted(std::chrono::steady_clock::time_point submit_time) noexcept;
void NotePresentDone() noexcept;
bool TakePresentTiming(double& lat_mean, double& lat_sd, double& int_mean, double& int_sd) noexcept;

struct VideoOutBufferAttribute2;
struct VideoOutFlipStatus;
struct VideoOutVblankStatus;
struct VideoOutOutputStatus;
struct VideoOutOutputOptions;
struct VideoOutBuffers;
struct VideoOutColorSettings;

class VideoOutDriver final {
public:
	struct Impl;

	VideoOutDriver(uint32_t width, uint32_t height, Graphics::Presenter& presenter);
	~VideoOutDriver();
	KYTY_CLASS_NO_COPY(VideoOutDriver);

	int  SubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index, int flip_mode,
	                       int64_t flip_arg, uint64_t& request_id);
	void PrepareFlip(uint64_t request_id, Graphics::CommandBuffer& buffer);
	void CompleteFlip(uint64_t request_id);
	void SubmitFlipPreparation(uint64_t request_id);
	void WaitForSubmitSlot();
	void WaitFlipDone(int handle, int index);

	[[nodiscard]] Impl& State() noexcept;

private:
	std::unique_ptr<Impl> m_impl;
};

[[nodiscard]] VideoOutDriver& VideoOutInit(uint32_t width, uint32_t height,
                                           Graphics::Presenter& presenter);
void                          VideoOutShutdown();

KYTY_SYSV_ABI int  VideoOutOpen(int user_id, int bus_type, int index, const void* param);
KYTY_SYSV_ABI int  VideoOutClose(int handle);
KYTY_SYSV_ABI void VideoOutSetBufferAttribute2(VideoOutBufferAttribute2* attribute,
                                               uint64_t pixel_format, uint32_t tiling_mode,
                                               uint32_t width, uint32_t height, uint64_t option,
                                               uint32_t dcc_control,
                                               uint64_t dcc_cb_register_clear_color);
KYTY_SYSV_ABI int  VideoOutSetFlipRate(int handle, int rate);
KYTY_SYSV_ABI int  VideoOutAddFlipEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                        void* udata);
KYTY_SYSV_ABI int  VideoOutAddVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                          void* udata);
KYTY_SYSV_ABI int VideoOutAddPreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                                 void* udata);
KYTY_SYSV_ABI int VideoOutAddOutputModeEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                             void* udata);
KYTY_SYSV_ABI int VideoOutDeleteFlipEvent(LibKernel::EventQueue::KernelEqueue eq, int handle);
KYTY_SYSV_ABI int VideoOutDeleteVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle);
KYTY_SYSV_ABI int VideoOutDeletePreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq,
                                                    int                                 handle);
KYTY_SYSV_ABI int VideoOutRegisterBuffers2(int handle, int set_index, int buffer_index_start,
                                           const VideoOutBuffers* buffers, int buffer_num,
                                           const VideoOutBufferAttribute2* attribute, int category,
                                           void* option);
KYTY_SYSV_ABI int VideoOutSubmitChangeBufferAttribute2(int handle, int set_index,
                                                       const VideoOutBufferAttribute2* attribute,
                                                       void*                           option);
KYTY_SYSV_ABI int VideoOutUnregisterBuffers(int handle, int set_index);
KYTY_SYSV_ABI int VideoOutSubmitFlip(int handle, int index, int flip_mode, int64_t flip_arg);
KYTY_SYSV_ABI int VideoOutGetFlipStatus(int handle, VideoOutFlipStatus* status);
KYTY_SYSV_ABI int VideoOutIsFlipPending(int handle);
KYTY_SYSV_ABI int VideoOutGetVblankStatus(int handle, VideoOutVblankStatus* status);
KYTY_SYSV_ABI int VideoOutGetEventId(const LibKernel::EventQueue::KernelEvent* ev);
KYTY_SYSV_ABI int VideoOutGetEventData(const LibKernel::EventQueue::KernelEvent* ev, int64_t* data);
KYTY_SYSV_ABI int VideoOutGetEventCount(const LibKernel::EventQueue::KernelEvent* ev);
KYTY_SYSV_ABI int VideoOutWaitVblank(int handle);
KYTY_SYSV_ABI int VideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status);
KYTY_SYSV_ABI int VideoOutInitializeOutputOptions(VideoOutOutputOptions* options);
KYTY_SYSV_ABI int VideoOutIsOutputSupported(int handle, uint64_t mode,
                                            const VideoOutOutputOptions* options,
                                            void* reserved_ptr, uint64_t reserved);
KYTY_SYSV_ABI int VideoOutConfigureOutput(int handle, uint64_t mode,
                                          const VideoOutOutputOptions* options, void* reserved_ptr,
                                          uint64_t reserved);
KYTY_SYSV_ABI int VideoOutSetWindowModeMargins(int handle, int top, int bottom);
KYTY_SYSV_ABI int VideoOutLatencyControlWaitBeforeInput(int handle);
KYTY_SYSV_ABI int VideoOutLatencyMeasureSetStartPoint(int handle, uint32_t point);
KYTY_SYSV_ABI int VideoOutColorSettingsSetGamma(VideoOutColorSettings* settings, float gamma);
KYTY_SYSV_ABI int VideoOutAdjustColor(int handle, const VideoOutColorSettings* settings);

} // namespace Libs::VideoOut

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_ */
