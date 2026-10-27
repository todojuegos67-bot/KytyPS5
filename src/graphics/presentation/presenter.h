#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_

#include "common/common.h"

#include <cstdint>
#include <memory>
#include <stop_token>

namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;
struct ImageInfo;
struct WindowContext;

class Presenter final {
public:
	struct Frame;

	explicit Presenter(WindowContext& window);
	~Presenter();
	KYTY_CLASS_NO_COPY(Presenter);

	// `hdr`: the guest buffer holds BT.2100 PQ pixels (shown on an HDR10 swapchain when there is one).
	[[nodiscard]] Frame&         PrepareFrame(CommandBuffer& command, const ImageInfo& info,
	                                         bool hdr = false);
	// `producer`: the GPU thread's command buffer (a guest flip of a blank buffer).
	[[nodiscard]] Frame&         PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque, CommandBuffer* producer);
	[[nodiscard]] Frame*         PrepareLastFrame();
	// The vblank thread hands a frame to the presenter's own thread ("Kyty.Present"), which presents it once the display
	// can take it: a present that waits for the display (VSync, a full queue, a driver's forced vertical sync) no longer
	// holds up the guest's vblanks, and the game keeps its speed. The newest frame wins: one still waiting when a newer
	// comes is not shown. `reuse`: the last frame again (PrepareLastFrame).
	void                         Submit(Frame& frame, bool reuse = false);
	// A cleared frame for the presenter's thread to make and show (the vblank thread's own, without a guest flip).
	void                         SubmitBlank(uint32_t width, uint32_t height, bool opaque);
	// Nothing submitted waits or is being presented: the vblank thread's idle refreshes (the last frame again, a blank
	// one) go only then, never in place of a guest frame.
	[[nodiscard]] bool           PresentIdle();
	// VSync (FIFO presents) on a display refreshing at a whole multiple of the guest's vblank rate: how many of the
	// display's vblanks make one of the guest's (KYTY_VSYNC_LOCK=0: none), else 0. The vblank thread then takes its
	// beat from the display (WaitForDisplayVblank) instead of its clock: a 59.94 Hz TV no longer drifts against 60 Hz
	// vblanks (a frame dropped or doubled every 17 s).
	[[nodiscard]] uint32_t       DisplayVblanksPerTick(uint32_t vblank_hz);
	[[nodiscard]] bool           WaitForDisplayVblank(uint64_t& seen, uint32_t timeout_us);
	[[nodiscard]] bool           IsGuestPaused() const noexcept;
	// --hdr and the display takes HDR10: the game is told it may output HDR.
	[[nodiscard]] bool           IsHdrSupported() const noexcept;
	[[nodiscard]] bool           NeedsSystemOverlayRefresh() const noexcept;
	// Only the overlay's content changed (not merely shown).
	[[nodiscard]] bool           SystemOverlayChanged() const noexcept;
	[[nodiscard]] RenderContext& Renderer() const noexcept;
	// KYTY_FLIP_RATE caps the frame's flip. With frame generation only frames carrying its inputs
	// are capped (the game's 3D view): movies and menus keep a flip per vblank.
	[[nodiscard]] static bool FlipRateApplies(const Frame& frame) noexcept;
	// A frame of the game's 3D view (its TAA pass was seen: gameplay or an in-engine scene), not a menu, load or movie.
	[[nodiscard]] static bool IsGameView(const Frame& frame) noexcept;
	// The frame shows the game's 3D view (its TAA pass ran): above 60 Hz the others (movies, menus, loading screens)
	// keep 60 (videoOut.cpp).
	[[nodiscard]] static bool IsSceneFrame(const Frame& frame) noexcept;
	// Frame generation is on: it paces the frames it makes from the intervals between the game's (videoOut.cpp).
	[[nodiscard]] static bool GeneratesFrames() noexcept;
	// KYTY_ANTILAG: the guest frame's index (its flip), for its PRESENT mark.
	static void               SetLatencyIndex(Frame& frame, uint64_t index) noexcept;
	// KYTY_KBM_WARP: the game-frame pad reads the frame shows (HostInputKbmOnFlip at its flip).
	static void               SetKbmSeq(Frame& frame, uint64_t seq) noexcept;
	void                         Discard(Frame& frame);

private:
	struct Impl;
	// The presenter's thread: Submit's frames presented (Present), one at a time.
	void                  PresentLoop(std::stop_token token);
	void                  Present(Frame& frame, bool reuse);
	std::unique_ptr<Impl> m_impl;
};
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
