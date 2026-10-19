#include "frame-gen-hudless.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics::FrameGen {

bool g_hudless = false;

namespace {

// The game's final composite (01.007.000), as frame captures show it: the frame's last draw and its only one into the
// flip image, in menus and in play. Its pixel shader samples the HUD layer at CompositeHudBinding (3840x2160 RGBA16F,
// the flip image's size) and the 3D scene after it, of the render size or, at KYTY_RENDER_HEIGHT=2160 and on the PS5
// Pro path (its upscale's output), of the flip image's size too. The first composite is the one of a smaller scene
// (the PS5's 1440 and the lower render heights), the second the one of a scene of the output's size (the PS5 Pro
// path, KYTY_UPSCALER=dlss, and KYTY_RENDER_HEIGHT=2160): both samples 3840x2160 RGBA16F there, the HUD at binding 0.
constexpr std::array<uint64_t, 2> CompositeShaders {0x3359df0b33704924ULL, 0x1a178683189cc68bULL};
constexpr size_t                  CompositeHudBinding = 0;

bool IsCompositeShader(uint64_t hash) {
	return std::find(CompositeShaders.begin(), CompositeShaders.end(), hash) != CompositeShaders.end();
}

// The game's flip images by guest address (two or three alternate; four kept): a draw into one is the composite's
// candidate. Written at the guest flip, read by any translating thread (IsFlipTarget).
std::array<std::atomic<uint64_t>, 4> g_flips {};
uint32_t                             g_flip_next = 0;
// This frame's composite was drawn without its HUD: later draws into a flip image are the game's alone.
std::atomic<bool> g_taken {false};

// Kyty.Gpu.
int  g_ui_override = -1;    // KYTY_FG_HUDLESS_UI
bool g_ui_debug    = false; // KYTY_FG_HUDLESS_DEBUG
// The composite without its HUD (the flip image's pixels, as the frame image has them) and the flip image it was
// drawn into; the flip image and the HUD texture as the composite found them.
std::unique_ptr<VulkanImage> g_capture;
uint64_t                     g_capture_flip = 0;
std::unique_ptr<VulkanImage> g_flip_backup, g_ui_backup;

// How the HUD texture was told: by KYTY_FG_HUDLESS_UI, by the composite's pixel shader (CompositeShaders) or, for
// another game version's composite, by the guess.
enum class Way : uint8_t { Switch, Shader, Guess };

// The HUD texture a composite gave (candidate -1: none) and what it was told from.
struct Pick {
	uint64_t     shader = 0; // the composite's pixel shader
	vk::Extent3D flip_extent {}, extent {};
	vk::Format   flip_format = vk::Format::eUndefined, format = vk::Format::eUndefined;
	size_t       sampled = 0, candidates = 0, binding = 0;
	int          candidate = -1;
	Way          way       = Way::Guess;

	bool operator==(const Pick&) const = default;
};

// The stats line, every 600 guest flips: the frames given the HUD-less image (of those with DLSS-G's inputs), the
// composite draws seen and why one gave none; with the HUD texture taken last.
struct Stats {
	uint32_t frames = 0, scene = 0, captured = 0, composites = 0, refused = 0, no_hud = 0;
};
Stats g_stats;
Pick  g_last;
// The candidates line: at the first composite, then when the pick changes, at most once every 600 guest flips.
Pick     g_logged;
uint64_t g_logged_flips = 0; // GuestFlips() at it
bool     g_logged_once  = false;

// A sampled image of the composite that may be its HUD texture.
struct Candidate {
	size_t binding = 0;
	Image* image   = nullptr;
	bool   full    = false; // the flip image's size
};

// A texture cache image's barrier state as it was, put back once its copies are done.
struct SavedState {
	void Save(const Image& image) {
		whole = image.CurrentState();
		parts = image.backing.subresource_states;
	}
	void Restore(Image& image, vk::CommandBuffer command) const { image.RestoreState(whole, parts, command); }

	VulkanImageState              whole;
	std::vector<VulkanImageState> parts;
};

// The composite being drawn without its HUD (BeginHudlessDraw to EndHudlessDraw): its flip image and HUD texture,
// their barrier states as the composite found them, whether the flip image's pixels are needed again and whether its
// pass was in progress.
struct Redraw {
	Image*     flip = nullptr;
	Image*     ui   = nullptr;
	SavedState flip_state, ui_state;
	bool       keep_flip = false, in_pass = false;
};
Redraw g_redraw;

bool IsFlipAddress(uint64_t address) {
	return address != 0 && std::any_of(g_flips.begin(), g_flips.end(), [&](const std::atomic<uint64_t>& flip) {
		       return flip.load(std::memory_order_relaxed) == address;
	       });
}

// The frame image's format (Presenter::PrepareFrame): the flip image's pixels, which a copy or a blit then keeps.
vk::Format UnormOf(vk::Format format) {
	switch (format) {
		case vk::Format::eR8G8B8A8Srgb: return vk::Format::eR8G8B8A8Unorm;
		case vk::Format::eB8G8R8A8Srgb: return vk::Format::eB8G8R8A8Unorm;
		default: return format;
	}
}

// A render target of one level and layer, single-sampled, that can be copied whole and back (and so cleared).
bool IsCopyable(const VulkanImage& image) {
	const vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	                                  vk::ImageUsageFlagBits::eColorAttachment;
	return image.mip_levels == 1 && image.layers == 1 && image.samples == 1 && (image.usage & usage) == usage;
}

// One of the capture's images, a transfer source and destination of that format and size: made anew when it is not
// one, the image it replaces deleted once the GPU is past the commands recorded so far (the renderer thread's, which
// used it).
VulkanImage& Reserve(CommandBuffer& command, std::unique_ptr<VulkanImage>& image, vk::Format format,
                     vk::Extent3D extent) {
	if (image != nullptr && image->format == format && image->extent == extent) {
		return *image;
	}
	auto& graphics = command.GetGraphics();
	if (image != nullptr) {
		command.GetContext().GetCommandScheduler().DeferOperation(
		    [&graphics, retired = std::move(image)]() mutable { graphics.DeleteImage(*retired); });
	}
	image = std::make_unique<VulkanImage>();
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, *image)) {
		EXIT("HUD-less image: cannot allocate a %ux%u image\n", extent.width, extent.height);
	}
	return *image;
}

void Delete(GraphicContext& graphics, std::unique_ptr<VulkanImage>& image) {
	if (image != nullptr && image->image != nullptr) {
		graphics.DeleteImage(*image);
	}
	image.reset();
}

// The image into `copy`, which is left a transfer source.
void CopyOut(vk::CommandBuffer command, Image& image, VulkanImage& copy) {
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	Transition(command, copy, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageCopy region {};
	region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.extent         = copy.extent;
	command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, copy.image,
	                  vk::ImageLayout::eTransferDstOptimal, region);
	Transition(command, copy, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
	           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
}

// CopyOut's `copy` back into the image.
void CopyBack(vk::CommandBuffer command, const VulkanImage& copy, Image& image) {
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	vk::ImageCopy region {};
	region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.extent         = copy.extent;
	command.copyImage(copy.image, vk::ImageLayout::eTransferSrcOptimal, image.backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, region);
}

// The HUD texture (-1: none): the composite's candidate KYTY_FG_HUDLESS_UI, else the one at its pixel shader's HUD
// binding, else the guess: of the candidates of the flip image's size, the one of the lowest binding (the scene, at
// times of that size too, comes after the HUD in the composite above).
int ChooseHud(const std::vector<Candidate>& candidates, Way way) {
	if (way == Way::Switch) {
		return g_ui_override < static_cast<int>(candidates.size()) ? g_ui_override : -1;
	}
	for (size_t i = 0; i < candidates.size(); ++i) {
		if (way == Way::Shader ? candidates[i].binding == CompositeHudBinding : candidates[i].full) {
			return static_cast<int>(i);
		}
	}
	return -1;
}

const char* WayName(Way way) {
	switch (way) {
		case Way::Switch: return "KYTY_FG_HUDLESS_UI";
		case Way::Shader: return "the composite's pixel shader";
		default: return "the guess";
	}
}

// A line of the log and of stdout (the run log), as frame-gen.cpp's.
void Report(const char* line) {
	LOGF("frame generation: %s\n", line);
	std::printf("Frame generation: %s\n", line);
	std::fflush(stdout);
}

void LogCandidates(const Pick& pick, const std::vector<Candidate>& candidates) {
	char part[192];
	std::snprintf(part, sizeof(part),
	              "HUD-less image: the composite (PS 0x%016" PRIx64 ") into %ux%u %s samples %zu images; candidates:",
	              pick.shader, pick.flip_extent.width, pick.flip_extent.height, vk::to_string(pick.flip_format).c_str(),
	              pick.sampled);
	std::string line = part;
	for (size_t i = 0; i < candidates.size(); ++i) {
		const auto& image = candidates[i].image->backing;
		std::snprintf(part, sizeof(part), " [%zu] binding %zu %ux%u %s%s", i, candidates[i].binding,
		              image.extent.width, image.extent.height, vk::to_string(image.format).c_str(),
		              static_cast<int>(i) == pick.candidate ? " (HUD)" : "");
		line += part;
	}
	line += candidates.empty() ? " none" : "";
	line += pick.candidate >= 0 ? std::string("; the HUD texture by ") + WayName(pick.way) : "; no HUD texture";
	Report(line.c_str());
}

// The candidates line at the first composite, then when the pick changes, at most once every 600 guest flips.
void NoteCandidates(const Pick& pick, const std::vector<Candidate>& candidates) {
	const uint64_t flips = GuestFlips();
	if (g_logged_once && (pick == g_logged || flips - g_logged_flips < 600)) {
		return;
	}
	LogCandidates(pick, candidates);
	g_logged       = pick;
	g_logged_flips = flips;
	g_logged_once  = true;
}

void LogStats() {
	const auto& stats = g_stats;
	char        hud[192] {"none yet"};
	if (g_last.candidate >= 0) {
		std::snprintf(hud, sizeof(hud), "candidate %d of %zu, binding %zu, %ux%u %s (%s)", g_last.candidate,
		              g_last.candidates, g_last.binding, g_last.extent.width, g_last.extent.height,
		              vk::to_string(g_last.format).c_str(), WayName(g_last.way));
	}
	char line[448];
	std::snprintf(line, sizeof(line),
	              "HUD-less image: %u frames, %u given it of %u with DLSS-G's inputs; composite draws %u (pass refused "
	              "%u, no HUD texture %u); HUD texture: %s",
	              stats.frames, stats.captured, stats.scene, stats.composites, stats.refused, stats.no_hud, hud);
	Report(line);
}

} // namespace

void InitializeHudless() {
	const char* value = std::getenv("KYTY_FG_HUDLESS");
	g_hudless         = value != nullptr && std::string_view(value) == "1";
	if (!g_hudless) {
		return;
	}
	if (const char* ui = std::getenv("KYTY_FG_HUDLESS_UI"); ui != nullptr && *ui != '\0') {
		g_ui_override = std::max(std::atoi(ui), -1);
	}
	const char* debug = std::getenv("KYTY_FG_HUDLESS_DEBUG");
	g_ui_debug        = debug != nullptr && std::string_view(debug) == "1";
	Report(g_ui_debug ? "the HUD-less image (KYTY_FG_HUDLESS=1), the HUD texture left cleared on screen "
	                    "(KYTY_FG_HUDLESS_DEBUG=1)"
	                  : "the HUD-less image (KYTY_FG_HUDLESS=1)");
}

int FlipSlot(std::span<const RenderColorInfo> targets) {
	if (g_taken.load(std::memory_order_relaxed)) {
		return -1;
	}
	for (size_t i = 0; i < targets.size(); ++i) {
		if (IsFlipAddress(targets[i].desc.info.data.address)) {
			return static_cast<int>(i);
		}
	}
	return -1;
}

bool IsFlipTarget(uint64_t address) {
	return !g_taken.load(std::memory_order_relaxed) && IsFlipAddress(address);
}

bool BeginHudlessDraw(CommandBuffer& command, const RenderState& rendering, const DrawRenderState& state, uint32_t slot,
                      std::span<const TextureBinding> images) {
	// (A frame without DLSS-G's inputs so far: its flip would not hand the image over, PrepareHudless.)
	if (!InputsGathered()) {
		return false;
	}
	auto& cache = command.GetContext().GetTextureCache();
	auto* flip  = cache.TryGetImage(state.color_info[slot].image_id);
	++g_stats.composites;
	// Drawn twice, the draw may write nothing but the flip image, whose copy alone undoes the first draw: no other
	// attachment, no buffer or storage image its shaders write.
	const bool writes = HasShaderBufferWrites(state.vs_input_info.stage) ||
	                    (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) ||
	                    std::any_of(images.begin(), images.end(), [](const TextureBinding& binding) {
		                    return binding.desc.type == TextureCache::BindingType::Storage;
	                    });
	if (flip == nullptr || writes || rendering.num_color_attachments != 1 || slot != 0 ||
	    rendering.depth_stencil_attachment.image_view != nullptr || !IsCopyable(flip->backing)) {
		++g_stats.refused;
		return false;
	}
	const auto& extent = flip->backing.extent;
	// Of the images the composite samples, the color render targets at least half the flip image's size.
	std::vector<Candidate> candidates;
	for (size_t i = 0; i < images.size(); ++i) {
		auto* image = cache.TryGetImage(images[i].image_id);
		if (image == nullptr || image->backing.image == flip->backing.image ||
		    images[i].desc.type != TextureCache::BindingType::Texture || image->info.IsDepth() ||
		    !IsCopyable(image->backing) || image->backing.extent.width * 2 < extent.width ||
		    image->backing.extent.height * 2 < extent.height ||
		    std::any_of(candidates.begin(), candidates.end(), [&](const Candidate& c) { return c.image == image; })) {
			continue;
		}
		candidates.push_back(
		    {i, image, image->backing.extent.width == extent.width && image->backing.extent.height == extent.height});
	}
	const auto* program = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	Pick        pick;
	pick.shader      = program != nullptr ? program->shader_hash : 0;
	pick.flip_extent = extent;
	pick.flip_format = flip->backing.format;
	pick.sampled     = images.size();
	pick.candidates  = candidates.size();
	pick.way         = g_ui_override >= 0 ? Way::Switch : IsCompositeShader(pick.shader) ? Way::Shader : Way::Guess;
	pick.candidate   = ChooseHud(candidates, pick.way);
	if (pick.candidate >= 0) {
		const auto& backing = candidates[pick.candidate].image->backing;
		pick.binding        = candidates[pick.candidate].binding;
		pick.extent         = backing.extent;
		pick.format         = backing.format;
	}
	NoteCandidates(pick, candidates);
	if (pick.candidate < 0) {
		++g_stats.no_hud;
		return false;
	}
	auto& redraw = g_redraw;
	auto& ui     = *candidates[pick.candidate].image;
	redraw.flip  = flip;
	redraw.ui    = &ui;
	// The flip image as the draw finds it is needed again unless its pass begins by clearing it.
	redraw.in_pass   = command.RenderingWith(rendering);
	redraw.keep_flip = redraw.in_pass || !rendering.color_attachments[slot].is_clear;
	redraw.flip_state.Save(*flip);
	redraw.ui_state.Save(ui);
	g_last = pick;
	command.EndRendering();
	auto vk_command = command.Handle();
	// Both as they are now (KYTY_FG_HUDLESS_DEBUG: the HUD texture is left cleared, so the game's own draw shows what
	// was taken for the HUD: whatever vanishes from the screen is what DLSS-G gets without it).
	if (redraw.keep_flip) {
		CopyOut(vk_command, *flip, Reserve(command, g_flip_backup, flip->backing.format, extent));
	}
	if (!g_ui_debug) {
		CopyOut(vk_command, ui, Reserve(command, g_ui_backup, ui.backing.format, ui.backing.extent));
	}
	// No HUD: its texture transparent, back in the layout the composite samples it in, the flip image an attachment.
	ui.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, vk_command);
	vk::ClearColorValue transparent {};
	transparent.uint32 = std::array<uint32_t, 4> {0u, 0u, 0u, 0u};
	const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk_command.clearColorImage(ui.backing.image, vk::ImageLayout::eTransferDstOptimal, &transparent, 1, &whole);
	redraw.ui_state.Restore(ui, vk_command);
	if (redraw.keep_flip) {
		redraw.flip_state.Restore(*flip, vk_command);
	}
	// The draw without it, in its pass as the game's draw finds it (continued, or begun as the game's begins).
	if (redraw.in_pass) {
		command.ResumeRendering(rendering);
	} else {
		command.BeginRendering(rendering);
	}
	return true;
}

void EndHudlessDraw(CommandBuffer& command, const RenderState& rendering) {
	auto& redraw = g_redraw;
	auto& flip   = *redraw.flip;
	auto& ui     = *redraw.ui;
	command.EndRendering();
	auto vk_command = command.Handle();
	CopyOut(vk_command, flip, Reserve(command, g_capture, UnormOf(flip.backing.format), flip.backing.extent));
	// Everything back, the pass again in progress for the game's draw if it was.
	if (redraw.keep_flip) {
		CopyBack(vk_command, *g_flip_backup, flip);
	}
	if (!g_ui_debug) {
		CopyBack(vk_command, *g_ui_backup, ui);
	}
	redraw.flip_state.Restore(flip, vk_command);
	redraw.ui_state.Restore(ui, vk_command);
	if (redraw.in_pass) {
		command.ResumeRendering(rendering);
	}
	g_capture_flip = flip.info.data.address;
	g_taken.store(true, std::memory_order_relaxed);
	redraw.flip = redraw.ui = nullptr;
}

void PrepareHudless(FrameInputs& inputs, const Image& flip) {
	// This flip image among the recent ones (a new one in the oldest's place).
	const uint64_t address = flip.info.data.address;
	if (!IsFlipAddress(address)) {
		g_flips[g_flip_next].store(address, std::memory_order_relaxed);
		g_flip_next = (g_flip_next + 1) % static_cast<uint32_t>(g_flips.size());
	}
	const bool taken = g_taken.exchange(false, std::memory_order_relaxed);
	// The composite drawn into this flip image goes with a frame DLSS-G generates from: the frame's image and the
	// capture's trade places (the frame's former one, idle since the frame was acquired, takes the next capture).
	auto& hudless       = inputs.hudless;
	hudless.has_image   = taken && inputs.valid && g_capture != nullptr && g_capture_flip == address;
	hudless.has_display = false;
	if (hudless.has_image) {
		std::swap(hudless.image, g_capture);
	}
	++g_stats.frames;
	g_stats.scene += inputs.valid ? 1u : 0u;
	g_stats.captured += hudless.has_image ? 1u : 0u;
	if (g_stats.frames == 600) {
		LogStats();
		g_stats = {};
	}
}

void ScaleHudless(GraphicContext& graphics, vk::CommandBuffer command, FrameInputs& inputs, vk::Extent2D backbuffer,
                  vk::Rect2D region, vk::Format format) {
	auto& hudless       = inputs.hudless;
	hudless.has_display = false;
	if (!inputs.valid || !hudless.has_image) {
		return;
	}
	auto& source  = *hudless.image;
	auto& display = hudless.display;
	// A sampled copy target of the backbuffer's size and format, as DLSS-G's other inputs (one made for another went
	// with the swapchain's recreation, which waited for its queues).
	VulkanImage backbuffer_like;
	backbuffer_like.format = format;
	backbuffer_like.extent = vk::Extent3D {backbuffer.width, backbuffer.height, 1};
	Configure(graphics, display, hudless.display_view, backbuffer_like);
	Transition(command, source, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
	           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
	Transition(command, display, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	// Where RecordPresentCommands put the frame image, black around it as there.
	if (region.offset.x != 0 || region.offset.y != 0 || region.extent != backbuffer) {
		const vk::ClearColorValue       black {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}};
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(display.image, vk::ImageLayout::eTransferDstOptimal, &black, 1, &whole);
		Transition(command, display, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
		           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	}
	vk::ImageBlit blit {};
	blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(source.extent.width),
	                                    static_cast<int32_t>(source.extent.height), 1};
	blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.dstOffsets[0]  = vk::Offset3D {region.offset.x, region.offset.y, 0};
	blit.dstOffsets[1]  = vk::Offset3D {region.offset.x + static_cast<int32_t>(region.extent.width),
	                                    region.offset.y + static_cast<int32_t>(region.extent.height), 1};
	command.blitImage(source.image, vk::ImageLayout::eTransferSrcOptimal, display.image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);
	Transition(command, display, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
	hudless.has_display = true;
}

void DestroyHudlessInputs(GraphicContext& graphics, FrameInputs& inputs) {
	auto& hudless = inputs.hudless;
	if (hudless.display_view != nullptr) {
		graphics.device.destroyImageView(hudless.display_view, nullptr);
		hudless.display_view = nullptr;
	}
	if (hudless.display.image != nullptr) {
		graphics.DeleteImage(hudless.display);
	}
	Delete(graphics, hudless.image);
	hudless.has_image = hudless.has_display = false;
}

void DestroyHudless(GraphicContext& graphics) {
	for (auto* image: {&g_capture, &g_flip_backup, &g_ui_backup}) {
		Delete(graphics, *image);
	}
}

} // namespace Libs::Graphics::FrameGen
