#ifndef EMULATOR_SRC_LOCAL_FRAME_GEN_H_
#define EMULATOR_SRC_LOCAL_FRAME_GEN_H_
// DLSS Frame Generation through NVIDIA Streamline (KYTY_FRAMEGEN=<generated frames, 1..5>), for
// Demon's Souls PPSA01341 01.007.000 and 01.005.000. The inputs come from the game's own passes, found by frame
// capture (docs/RE-DEMONS-SOULS.md): the G-buffer vertex shaders' constants hold the camera
// (world->clip matrix, the previous frame's, and the TAA jitter), the TAA dispatch reads the depth
// and motion-vector images. The renderer thread gathers them per guest frame; at the guest flip
// they are copied next to the presenter's frame; the present thread hands them to Streamline and
// presents through it (sl.interposer.dll is the process' Vulkan loader, see Initialize()).
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/image.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace Libs::Graphics {
struct RenderState;
struct TextureBinding;
class CommandBuffer;
class TextureCache;
namespace FrameGen {

// The TAA compute shader, and the G-buffer vertex shaders sharing the constant layout below (flattened SRT
// words), of 01.007.000 and of 01.005.000: the same shaders recompiled (KYTY_SHADER_DUMP: the same resources,
// the vertex shaders the same translated program; the TAA images are what is read from its dispatch).
constexpr std::array<uint64_t, 2> TaaShaders {0xa44e580f4f0cfb2bULL, 0x934a6e1197b99148ULL};
constexpr std::array<uint64_t, 6> GbufferVertexShaders {0xf4d2e78cd1769fc7ULL, 0x08e16923875aa8e9ULL, 0x0ffcbf08be26e74eULL,
                                                        0x1d69b5d27198446aULL, 0x5931e753b757f3bcULL, 0xa97f610824942162ULL};
inline bool IsTaaShader(uint64_t hash) {
	return std::find(TaaShaders.begin(), TaaShaders.end(), hash) != TaaShaders.end();
}
inline bool IsGbufferVertexShader(uint64_t hash) {
	return std::find(GbufferVertexShaders.begin(), GbufferVertexShaders.end(), hash) != GbufferVertexShaders.end();
}
constexpr uint32_t SrtViewProj     = 29; // 16 floats, row-major, clip = M * world
constexpr uint32_t SrtPrevViewProj = 45; // 12 floats: the previous frame's rows x, y and w
constexpr uint32_t SrtJitter       = 57; // 2 floats: clip-space offset (NDC units)

struct Camera {
	std::array<float, 16> view_proj {};
	std::array<float, 16> prev_view_proj {};
	std::array<float, 2>  jitter_ndc {};
	bool                  valid = false;
};

// KYTY_FG_HUDLESS (frame-gen-hudless.h): a frame's game image without its HUD, its pixels as the frame image's
// (`image`), and that laid out as the backbuffer (`display`, the present's: DLSS-G's HUD-less color).
struct HudlessImages {
	std::unique_ptr<VulkanImage> image;
	VulkanImage                  display;
	vk::ImageView                display_view = nullptr;
	bool                         has_image = false, has_display = false;
};

// One presented frame's inputs, owned by the presenter's frame.
struct FrameInputs {
	VulkanImage   depth, motion;
	vk::ImageView depth_view = nullptr, motion_view = nullptr;
	vk::Extent2D  render_extent {};
	Camera        camera;
	uint64_t      guest_frame = 0;
	bool          motion_attachment = false; // the motion copy is a color attachment too
	// FSR 3: the game's image before its HUD (the flip image as the first render pass after the 3D scene found it),
	// copied at the guest flip; its pixels as the flip image's (a UNORM format).
	VulkanImage   hudless;
	vk::ImageView hudless_view = nullptr;
	bool          has_hudless  = false;
	bool          hudless_composite = false; // the game's composite drawn without its HUD texture (not a copy with it)
	// ... scaled to the backbuffer (DLSS-G wants it at the backbuffer's size): ScaleHudless.
	VulkanImage   hudless_display;
	vk::ImageView hudless_display_view = nullptr;
	bool          has_hudless_display  = false;
	bool          valid       = false;
};

// Process start, before the window: loads Streamline when KYTY_FRAMEGEN or KYTY_UPSCALER=dlss is set (it then
// becomes SDL's Vulkan loader, and so the proxy of instance, device and swapchain calls).
void Initialize();
// After Initialize: Streamline is loaded (device creation then adds what it relies on but does
// not add itself, VK_KHR_present_id for its VK_NV_low_latency2).
[[nodiscard]] bool Requested();
// Streamline is loaded (DLSS frame generation or DLSS super resolution): it is the Vulkan loader then.
[[nodiscard]] bool StreamlineLoaded();
// DLSS-G under the G-SYNC cap (KYTY_VRR_CAP_FPS): the real frames' least interval in seconds, 0 when none. The flip
// queue holds the game to it (videoOut.cpp): Reflex's limit alone held the presents, and the game made 60 a second at
// x6 (37.3 shown real): the rest dropped, a stutter.
[[nodiscard]] double RealFrameInterval();
// Frames to generate per rendered frame (KYTY_FRAMEGEN; 0 when off).
[[nodiscard]] int  GeneratedFrames();
// The present thread's stages around a DLSS-G present, timed into the [dlssg] line: the swapchain image acquire, the
// wait for the frame's own GPU work (present_scheduler) and the flip's whole Present (the flip is done when it returns).
void NoteAcquireMs(double ms);
void NoteGpuWaitMs(double ms);
void NoteFlipPresentMs(double ms);
void NoteSubmitMs(double ms);        // present thread: record + submit of the copy into the swapchain
void NoteBeforePresentMs(double ms); // BeforePresent (Streamline tags and options)
// Before the device is created: why Streamline says DLSS-G cannot run with this GPU, driver and Windows (e.g.
// hardware-accelerated GPU scheduling turned off), logged; empty when it can, or when frame generation is off. A refused
// device's error repeats it.
[[nodiscard]] std::string SupportProblem(vk::PhysicalDevice physical_device);
// A refused device's advice when this process runs without bottom-up ASLR (issue #45: forcing it on for the emulator in
// Windows' Exploit protection let the driver make the device); empty otherwise.
[[nodiscard]] std::string RefusedDeviceAdvice();
// After the device exists: checks that DLSS-G is available on it.
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device);
// Right after the dispatcher is loaded for the device (before the recording layer copies it):
// the command-buffer calls Streamline intercepts only for its command-list state tracking
// (disabled here) go straight to the driver.
void BypassCommandHooks(vk::Device device);
[[nodiscard]] bool Enabled();
// The TAA pass's inputs are gathered at the guest flip: DLSS-G (Enabled) or FSR 3 frame generation (src/local/fsr3-fg.h).
[[nodiscard]] bool Capturing();

// Present thread, around vkQueuePresentKHR of the image showing `inputs` (null: a frame
// without inputs, e.g. a menu). `backbuffer` is the swapchain extent, `region` the part the
// game image covers.
void BeforePresent(const FrameInputs* inputs, vk::Extent2D backbuffer, vk::Rect2D region,
                   vk::Format format, uint32_t image_count);
void AfterPresent();
// Present thread: the swapchain is made again (DLSS-G off until frames with inputs turn it on again).
void SwapchainChanging();
// Any thread: a frame with inputs was left unshown (the presenter kept a newer one): the next one resets DLSS-G's history.
void NoteDroppedFrame();
// Present thread: the frames DLSS-G showed per presented frame (1 without generation), as last
// queried (about once a second), for the KYTY_FPS_HUD panel.
[[nodiscard]] uint32_t PresentedFrames();
// KYTY_FPS_HUD: the panel's rectangle, normalized to the game image. Its motion vectors are
// zeroed so generated frames keep the panel in place.
void SetStaticRect(float x0, float y0, float x1, float y1);

// Renderer thread (Kyty.Gpu).
// The guest frame ran the TAA pass: the game's 3D view (play, or a cutscene in the engine), not a movie, a menu or a
// loading screen; with or without frame generation, by the normal or the table way. The guest flip takes it
// (Presenter::PrepareFrame): above 60 Hz the other frames keep 60 (videoOut.cpp).
void NoteScene();
[[nodiscard]] bool TakeScene();
// The current guest frame ran the TAA pass so far, or the last one flipped did (HudAnchor: the game queues a frame's
// interface before the flip of the one before).
[[nodiscard]] bool SceneNoted();
// The guest flips taken so far (TakeScene).
[[nodiscard]] uint64_t GuestFlips();
// The current guest frame has no camera yet: its next G-buffer draw goes the normal way (the table way's draws have
// no constants on the CPU), whose hook reads it.
[[nodiscard]] bool WantsCamera();
// The current guest frame has DLSS-G's inputs so far (the camera, the TAA pass's or the upscale's depth and motion),
// which its flip takes (PrepareInputs).
[[nodiscard]] bool InputsGathered();
void OnDraw(uint64_t vs_hash, std::span<const uint32_t> flattened_srt);
// KYTY_SHADER_CENSUS=1 (frame-gen.cpp): a program the game ran (stage v, p or c).
void NoteShader(uint64_t hash, char stage);
void OnDispatch(uint64_t cs_hash, ImageId depth, uint64_t depth_address, ImageId motion,
                uint64_t motion_address);
// The game's super resolution dispatch in place of its TAA (renderSuperResolution.cpp): the same depth and motion.
void OnUpscale(ImageId depth, uint64_t depth_address, ImageId motion, uint64_t motion_address);
// At the guest flip, in the frame's command buffer: copy the gathered images into `inputs`.
// `flip` (`flip_id`): the image this flip shows (the HUD-less image is taken from the next ones like it).
void PrepareInputs(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command,
                   FrameInputs& inputs, ImageId flip_id, Image& flip);
// Renderer thread, as a render pass begins (CommandBuffer::BeginRendering, after the previous one ended): a pass into
// a recent flip image; the HUD-less copy is taken before the one the HUD starts with.
void OnBeginRendering(CommandBuffer& command, const RenderState& state);
// Renderer thread, after a draw was recorded: in the flip image's render pass, the HUD-less copy is taken after its
// first draw (the scene's full-screen pass; the HUD is drawn after it in the same pass). KYTY_FSR3_HUDLESS_DRAW: after
// which draw.
void AfterDraw(CommandBuffer& command);
// Normal draw path. The game draws its final image (scene and HUD from separate textures) into the flip image with
// one draw. NoteDraw counts the draws into each color target this frame. FlipSlot: the slot of `targets` that is a
// recent flip image (-1: none). BeforeFlipDraw, with everything bound and before the pass begins: that draw once
// more with the HUD's texture (of `images`, the one most draws went to this frame) cleared to transparent, the
// result copied as the HUD-less image, then the flip image and the HUD texture put back as they were.
void NoteDraw(ImageId target);
// The table and native XPR paths: a draw into a recent flip image is left to the normal path (FlipSlot, BeforeFlipDraw).
[[nodiscard]] bool IsFlipImage(ImageId id);
[[nodiscard]] int FlipSlot(std::span<const ImageId> targets);
void BeforeFlipDraw(CommandBuffer& command, const RenderState& rendering, ImageId target, uint32_t slot,
                    std::span<const TextureBinding> images, const std::function<void()>& redraw);
// Upscaling (KYTY_UPSCALE=fsr): with everything bound for the game's composite into the flip image and before the
// descriptors are written, its 3D image (render size, RGBA16F) upscaled by FSR 3.1 to KYTY_UPSCALE_HEIGHT (default
// 1440) with the TAA pass's depth and motion vectors, and the composite's binding pointed at the result. Its draw
// samples it by coordinates: the bigger image takes the place of the render-size one.
void UpscaleComposite(CommandBuffer& command, const RenderState& rendering, std::span<TextureBinding> images);
// KYTY_TAA_DIAG=1: true for the TAA dispatches of the frames logged ([taa] lines: the first 6 scene frames, then every
// 600th); the composite's images are logged in those frames too.
[[nodiscard]] bool TaaDiagFrame();
// KYTY_TAA_REPLACE=dlss: the game's TAA dispatch replaced by DLSS (DLAA at the render size) on the dispatch's own images:
// color 0 (jittered), depth 1, motion 3, output 6 (renderCompute.cpp, in place of the dispatch, its descriptors bound).
// KYTY_TAA_REPLACE=upscaler: the dispatch replaced by a copy of its jittered color to its output, for the upscaler.
// False: the dispatch runs as usual.
[[nodiscard]] bool TaaReplaceWanted();
bool ReplaceTaa(CommandBuffer& command, vk::CommandBuffer vk_command, std::span<const TextureBinding> images);
// KYTY_TAA_REPLACE=upscaler (KYTY_TAA_BYPASS_MODE not copy): right after the TAA dispatch, its output's color replaced
// by the raw jittered frame's, its alpha (flags later passes read) kept. False: nothing done.
bool AfterTaa(CommandBuffer& command, vk::CommandBuffer vk_command, std::span<const TextureBinding> images);
void DestroyInputs(GraphicContext& graphics, FrameInputs& inputs);
// Present thread, in the present's command buffer: the HUD-less image scaled to the backbuffer (DLSS-G).
void ScaleHudless(GraphicContext& graphics, vk::CommandBuffer command, FrameInputs& inputs, vk::Extent2D backbuffer,
                  vk::Format format);

// Frame generation's own images (not the texture cache's; frame-gen-hudless.cpp's too), on the thread that records
// them: `image` (re)made as a sampled copy target like `source`, with its view (with `attachment`, a color attachment
// too where the format allows it, *attachment tells); a barrier from `image`'s state to `layout`, `access` at `stage`.
void Configure(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view, const VulkanImage& source,
               bool* attachment = nullptr);
void Transition(vk::CommandBuffer command, VulkanImage& image, vk::ImageAspectFlags aspect, vk::ImageLayout layout,
                vk::AccessFlags2 access, vk::PipelineStageFlags2 stage);

// DLSS super resolution (KYTY_UPSCALER=dlss) in place of Sony's PS5 Pro upscaler the game dispatches
// (graphics/guest_gpu/superResolution.h, renderer/renderSuperResolution.cpp).
struct UpscaleImage {
	vk::Image           image  = nullptr;
	vk::ImageView       view   = nullptr;
	vk::ImageLayout     layout = vk::ImageLayout::eUndefined; // as the command buffer leaves it for the upscaler
	vk::Format          format = vk::Format::eUndefined;
	vk::Extent2D        extent {};
	vk::ImageUsageFlags usage {};
	uint32_t            level = 0, layer = 0; // the view's subresource
};
struct UpscaleInputs {
	UpscaleImage color, depth, motion, output; // color, depth and motion at the render size
	float        jitter_x = 0.0f, jitter_y = 0.0f;       // the projection's offset, in render pixels
	float        motion_scale_x = 0.0f, motion_scale_y = 0.0f; // motion vector * scale = render pixels to the previous
	bool         reset = false;                           // no history (a cut, a level load)
	std::array<float, 16> view_to_clip {};                // row vectors (p' = p * M)
	float        near_plane = 0.1f, far_plane = 3000.0f;
};
// Streamline offers DLSS super resolution on this device.
[[nodiscard]] bool UpscalerReady();
// Records DLSS into `command` (outside a render pass; the images in their `layout`s): false when it could not.
[[nodiscard]] bool Upscale(vk::CommandBuffer command, const UpscaleInputs& inputs);

} // namespace FrameGen
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FRAME_GEN_H_
