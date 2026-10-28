#include "analyzer.h"

#include "overlay.h"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace inj::analyzer {
namespace {

bool IsDepthFormat(VkFormat f) {
	switch (f) {
	case VK_FORMAT_D16_UNORM:
	case VK_FORMAT_X8_D24_UNORM_PACK32:
	case VK_FORMAT_D32_SFLOAT:
	case VK_FORMAT_D16_UNORM_S8_UINT:
	case VK_FORMAT_D24_UNORM_S8_UINT:
	case VK_FORMAT_D32_SFLOAT_S8_UINT: return true;
	default: return false;
	}
}
bool HasStencil(VkFormat f) {
	return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}
// How much a format looks like motion vectors (0: not at all).
float MotionFormatScore(VkFormat f) {
	switch (f) {
	case VK_FORMAT_R16G16_SFLOAT: return 1.0f;
	case VK_FORMAT_R32G32_SFLOAT: return 0.9f;
	case VK_FORMAT_R16G16_SNORM: return 0.85f;
	case VK_FORMAT_R16G16_UNORM: return 0.5f;
	case VK_FORMAT_R8G8_SNORM: return 0.3f;
	default: return 0.0f;
	}
}

struct Img {
	VkImageCreateInfo info {};
	VkImageLayout     layout = VK_IMAGE_LAYOUT_UNDEFINED;  // last layout seen in submitted work
	bool              interesting = false;
	bool              swapchain   = false;
};

enum class Kind : uint8_t { Layout, Rendering };
struct Ev {
	Kind          kind;
	uint8_t       ncolors = 0;
	VkImageLayout layout  = VK_IMAGE_LAYOUT_UNDEFINED;
	float         depth_clear = -1.0f;  // a cleared depth attachment's value, -1 none
	VkImage       image   = VK_NULL_HANDLE;  // Layout: the image; Rendering: the depth attachment
	VkImageLayout depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImage       colors[8] {};
	VkImageLayout color_layouts[8] {};
};

std::mutex                                        g_mutex;
std::unordered_map<VkImage, Img>                  g_images;
std::unordered_map<VkImageView, VkImage>          g_views;
std::unordered_map<VkCommandBuffer, std::vector<Ev>> g_cb;
std::vector<Ev>                                   g_frame;
std::unordered_set<VkImage>                       g_swap;

// A selected resource and how sure we are.
struct Pick {
	VkImage    image = VK_NULL_HANDLE;
	VkFormat   format = VK_FORMAT_UNDEFINED;
	VkExtent2D extent {};
	uint32_t   stable = 0;     // frames in a row with the same signature
	float      confidence = 0;
	float      score = 0;
	std::string why;
};
Pick  g_depth, g_motion, g_backbuffer;
bool  g_reversed = false, g_reversed_known = false;
int   g_hud_after = -1, g_hud_stable = 0, g_ui_passes = 0, g_passes_last = 0;
float g_hud_conf = 0;
uint64_t g_frames = 0;

// The profile: signatures the last run settled on.
struct Sig { VkFormat format = VK_FORMAT_UNDEFINED; uint32_t w = 0, h = 0; bool valid() const { return w != 0; } };
Sig  g_prof_depth, g_prof_motion;
bool g_prof_loaded = false;
std::chrono::steady_clock::time_point g_last_save {};
std::string g_saved_text;

// Our copies.
struct Copy {
	VkImage        image = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkImageView    view = VK_NULL_HANDLE;
	VkFormat       format = VK_FORMAT_UNDEFINED;
	VkExtent2D     extent {};
	bool           written = false;
};
Copy g_copy_depth, g_copy_motion, g_copy_color;

bool SameSig(const Img& i, const Sig& s) {
	return s.valid() && i.info.format == s.format && i.info.extent.width == s.w && i.info.extent.height == s.h;
}

std::wstring ProfilePath() {
	return ExeDir() + L"injector_profiles\\" + std::wstring(ExeName().begin(), ExeName().end()) + L".ini";
}

void LoadProfile() {
	g_prof_loaded = true;
	std::ifstream in(ProfilePath());
	if (!in) { Log("[analyze] no profile yet for %s: learning live", ExeName().c_str()); return; }
	std::string line, section;
	while (std::getline(in, line)) {
		if (line.empty() || line[0] == ';') continue;
		if (line[0] == '[') { section = line.substr(1, line.find(']') - 1); continue; }
		const auto eq = line.find('=');
		if (eq == std::string::npos) continue;
		const std::string key = line.substr(0, eq);
		const long value = std::strtol(line.c_str() + eq + 1, nullptr, 10);
		Sig* s = section == "depth" ? &g_prof_depth : section == "motion" ? &g_prof_motion : nullptr;
		if (s != nullptr) {
			if (key == "format") s->format = static_cast<VkFormat>(value);
			else if (key == "width") s->w = static_cast<uint32_t>(value);
			else if (key == "height") s->h = static_cast<uint32_t>(value);
		}
		if (section == "fg") {
			if (key == "mv_sign") Cfg().mv_sign = value < 0 ? -1 : 1;
			else if (key == "mv_units") Cfg().mv_units = value != 0 ? 1 : 0;
		}
	}
	Log("[analyze] profile loaded: depth fmt %d %ux%u, motion fmt %d %ux%u", g_prof_depth.format, g_prof_depth.w,
	    g_prof_depth.h, g_prof_motion.format, g_prof_motion.w, g_prof_motion.h);
}

void SaveProfileIfChanged() {
	const float need = Cfg().min_confidence;
	if (g_depth.confidence < need && g_motion.confidence < need) return;
	const auto now = std::chrono::steady_clock::now();
	if (now - g_last_save < std::chrono::seconds(5)) return;
	std::ostringstream o;
	o << "; written live by the injector's analyzer (F1 > Analizar vectores). Signatures only: format, size.\n";
	if (g_depth.confidence >= need)
		o << "[depth]\nformat=" << g_depth.format << "\nwidth=" << g_depth.extent.width << "\nheight=" << g_depth.extent.height
		  << "\nreversed=" << (g_reversed ? 1 : 0) << "\nconfidence=" << static_cast<int>(g_depth.confidence * 100) << "\n";
	if (g_motion.confidence >= need)
		o << "[motion]\nformat=" << g_motion.format << "\nwidth=" << g_motion.extent.width << "\nheight=" << g_motion.extent.height
		  << "\nconfidence=" << static_cast<int>(g_motion.confidence * 100) << "\n";
	if (g_backbuffer.image != VK_NULL_HANDLE)
		o << "[hud]\nbackbuffer_format=" << g_backbuffer.format << "\nwidth=" << g_backbuffer.extent.width << "\nheight="
		  << g_backbuffer.extent.height << "\nhud_after_pass=" << g_hud_after << "\nui_passes=" << g_ui_passes
		  << "\nconfidence=" << static_cast<int>(g_hud_conf * 100) << "\n";
	o << "[fg]\nmv_sign=" << Cfg().mv_sign << "\nmv_units=" << Cfg().mv_units << "\n";
	const std::string text = o.str();
	if (text == g_saved_text) return;
	CreateDirectoryW((ExeDir() + L"injector_profiles").c_str(), nullptr);
	std::ofstream out(ProfilePath(), std::ios::trunc);
	if (!out) return;
	out << text;
	g_saved_text = text;
	g_last_save  = now;
	Log("[analyze] profile updated: depth %.2f, motion %.2f, hud %.2f", g_depth.confidence, g_motion.confidence, g_hud_conf);
}

// Stability: a pick that keeps the same signature over ~1 s at 60 fps gets the full share.
void Settle(Pick& pick, const Img* best, VkImage image, float score) {
	if (best == nullptr) {
		pick.stable = 0;
		pick.image  = VK_NULL_HANDLE;
		pick.confidence *= 0.95f;
		return;
	}
	const bool same = pick.format == best->info.format && pick.extent.width == best->info.extent.width &&
	                  pick.extent.height == best->info.extent.height;
	pick.stable = same ? pick.stable + 1 : 0;
	pick.image  = image;
	pick.format = best->info.format;
	pick.extent = {best->info.extent.width, best->info.extent.height};
	pick.score  = score;
}

} // namespace

void PatchImageCreate(VkImageCreateInfo& info) {
	if (info.imageType != VK_IMAGE_TYPE_2D) return;
	const bool depth  = IsDepthFormat(info.format) && (info.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	const bool motion = MotionFormatScore(info.format) > 0 &&
	                    (info.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT));
	const bool color  = (info.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && info.extent.width >= 640;
	if ((depth || motion || color) && info.samples == VK_SAMPLE_COUNT_1_BIT) info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
}

void OnImageCreated(VkImage image, const VkImageCreateInfo& info) {
	Img img;
	img.info       = info;
	img.info.pNext = nullptr;
	img.info.pQueueFamilyIndices = nullptr;
	img.interesting = info.imageType == VK_IMAGE_TYPE_2D && info.samples == VK_SAMPLE_COUNT_1_BIT &&
	                  (info.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) && info.extent.width >= 320 && info.extent.height >= 180 &&
	                  (info.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
	                                 VK_IMAGE_USAGE_STORAGE_BIT));
	std::lock_guard lock(g_mutex);
	g_images[image] = img;
}

void OnImageDestroyed(VkImage image) {
	std::lock_guard lock(g_mutex);
	g_images.erase(image);
	for (Pick* p: {&g_depth, &g_motion, &g_backbuffer})
		if (p->image == image) p->image = VK_NULL_HANDLE;
}

void OnViewCreated(VkImageView view, const VkImageViewCreateInfo& info) {
	std::lock_guard lock(g_mutex);
	g_views[view] = info.image;
}
void OnViewDestroyed(VkImageView view) {
	std::lock_guard lock(g_mutex);
	g_views.erase(view);
}

void SetSwapchainImages(const VkImage* images, uint32_t count) {
	std::lock_guard lock(g_mutex);
	g_swap.clear();
	for (uint32_t i = 0; i < count; ++i) g_swap.insert(images[i]);
}

void OnBeginCommandBuffer(VkCommandBuffer cb) {
	if (!Cfg().analyze && !Cfg().use_game_inputs) return;
	std::lock_guard lock(g_mutex);
	auto it = g_cb.find(cb);
	if (it != g_cb.end()) it->second.clear();
}

void OnBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info) {
	if (info == nullptr || (!Cfg().analyze && !Cfg().use_game_inputs)) return;
	Ev ev {Kind::Rendering};
	std::lock_guard lock(g_mutex);
	const auto image_of = [](VkImageView view) {
		auto it = g_views.find(view);
		return it == g_views.end() ? VK_NULL_HANDLE : it->second;
	};
	for (uint32_t i = 0; i < info->colorAttachmentCount && ev.ncolors < 8; ++i) {
		const auto& a = info->pColorAttachments[i];
		if (a.imageView == VK_NULL_HANDLE) continue;
		ev.color_layouts[ev.ncolors] = a.imageLayout;
		ev.colors[ev.ncolors++]      = image_of(a.imageView);
	}
	if (info->pDepthAttachment != nullptr && info->pDepthAttachment->imageView != VK_NULL_HANDLE) {
		ev.image        = image_of(info->pDepthAttachment->imageView);
		ev.depth_layout = info->pDepthAttachment->imageLayout;
		if (info->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
			ev.depth_clear = info->pDepthAttachment->clearValue.depthStencil.depth;
	}
	g_cb[cb].push_back(ev);
}

static void RecordLayout(VkCommandBuffer cb, VkImage image, VkImageLayout layout) {
	auto it = g_images.find(image);
	if (it == g_images.end() || !it->second.interesting) return;
	Ev ev {Kind::Layout};
	ev.image  = image;
	ev.layout = layout;
	g_cb[cb].push_back(ev);
}

void OnBarrier2(VkCommandBuffer cb, const VkDependencyInfo* info) {
	if (info == nullptr || (!Cfg().analyze && !Cfg().use_game_inputs)) return;
	std::lock_guard lock(g_mutex);
	for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i)
		RecordLayout(cb, info->pImageMemoryBarriers[i].image, info->pImageMemoryBarriers[i].newLayout);
}

void OnBarrier(VkCommandBuffer cb, uint32_t count, const VkImageMemoryBarrier* barriers) {
	if (barriers == nullptr || (!Cfg().analyze && !Cfg().use_game_inputs)) return;
	std::lock_guard lock(g_mutex);
	for (uint32_t i = 0; i < count; ++i) RecordLayout(cb, barriers[i].image, barriers[i].newLayout);
}

void OnSubmit(const VkCommandBuffer* cbs, uint32_t count) {
	if (!Cfg().analyze && !Cfg().use_game_inputs) return;
	std::lock_guard lock(g_mutex);
	for (uint32_t i = 0; i < count; ++i) {
		auto it = g_cb.find(cbs[i]);
		if (it == g_cb.end()) continue;
		for (const auto& ev: it->second) {
			g_frame.push_back(ev);
			// Execution order: the layout each image is left in.
			auto set = [](VkImage image, VkImageLayout layout) {
				auto img = g_images.find(image);
				if (img != g_images.end()) img->second.layout = layout;
			};
			if (ev.kind == Kind::Layout) set(ev.image, ev.layout);
			else {
				if (ev.image != VK_NULL_HANDLE) set(ev.image, ev.depth_layout);
				for (uint8_t c = 0; c < ev.ncolors; ++c) set(ev.colors[c], ev.color_layouts[c]);
			}
		}
		// (A command buffer resubmitted without re-recording keeps its events.)
	}
	if (g_frame.size() > 200000) g_frame.erase(g_frame.begin(), g_frame.begin() + 100000);
}

void EndFrame(VkExtent2D display) {
	if (!g_prof_loaded) LoadProfile();
	if (!Cfg().analyze && !Cfg().use_game_inputs) return;
	std::lock_guard lock(g_mutex);
	++g_frames;
	struct Use { uint32_t depth = 0, color = 0, layout_general = 0; int first = -1, last = -1; float clear = -1; };
	std::unordered_map<VkImage, Use> uses;
	int passes = 0;
	std::vector<VkImage> pass_target;  // the first color target of each pass (to find the HUD point)
	for (const auto& ev: g_frame) {
		if (ev.kind == Kind::Layout) {
			if (ev.layout == VK_IMAGE_LAYOUT_GENERAL) uses[ev.image].layout_general++;
			continue;
		}
		if (ev.image != VK_NULL_HANDLE) {
			auto& u = uses[ev.image];
			u.depth++;
			if (ev.depth_clear >= 0) u.clear = ev.depth_clear;
		}
		VkImage first_color = VK_NULL_HANDLE;
		for (uint8_t c = 0; c < ev.ncolors; ++c) {
			if (ev.colors[c] == VK_NULL_HANDLE || g_swap.count(ev.colors[c])) continue;
			auto& u = uses[ev.colors[c]];
			u.color++;
			if (u.first < 0) u.first = passes;
			u.last = passes;
			if (first_color == VK_NULL_HANDLE) first_color = ev.colors[c];
		}
		pass_target.push_back(first_color);
		++passes;
	}
	g_frame.clear();
	g_passes_last = passes;
	if (passes == 0) return;
	const double display_area = std::max(1.0, static_cast<double>(display.width) * display.height);

	// Depth.
	const Img* best_d = nullptr; VkImage best_d_image = VK_NULL_HANDLE; float best_ds = 0; float best_clear = -1;
	for (const auto& [image, u]: uses) {
		if (u.depth == 0) continue;
		auto it = g_images.find(image);
		if (it == g_images.end() || !it->second.interesting || !IsDepthFormat(it->second.info.format)) continue;
		const double area = static_cast<double>(it->second.info.extent.width) * it->second.info.extent.height;
		float s = static_cast<float>(std::min(1.0, area / display_area)) + std::min(1.0f, 4.0f * u.depth / passes) * 0.5f;
		if (SameSig(it->second, g_prof_depth)) s += 0.5f;
		if (s > best_ds) { best_ds = s; best_d = &it->second; best_d_image = image; best_clear = u.clear; }
	}
	Settle(g_depth, best_d, best_d_image, best_ds);
	if (best_d != nullptr) {
		const double area = static_cast<double>(g_depth.extent.width) * g_depth.extent.height;
		float conf = 0.35f * static_cast<float>(std::min(1.0, area / display_area)) + 0.25f * std::min(1.0f, best_ds / 1.5f) +
		             0.40f * std::min(1.0f, g_depth.stable / 60.0f);
		if (SameSig(*best_d, g_prof_depth)) conf = std::max(conf, 0.5f + 0.5f * std::min(1.0f, g_depth.stable / 10.0f));
		g_depth.confidence = std::min(1.0f, conf);
		if (best_clear >= 0) { g_reversed = best_clear < 0.5f; g_reversed_known = true; }
		g_depth.why = best_clear >= 0 ? (best_clear < 0.5f ? "cleared to 0: reversed Z" : "cleared to 1: normal Z") : "no clear seen";
	}

	// Motion vectors: written this frame (a pass target, or moved to GENERAL: a compute write), 2-channel.
	const Img* best_m = nullptr; VkImage best_m_image = VK_NULL_HANDLE; float best_ms = 0;
	for (const auto& [image, u]: uses) {
		if (u.color == 0 && u.layout_general == 0) continue;
		auto it = g_images.find(image);
		if (it == g_images.end() || !it->second.interesting) continue;
		const float f = MotionFormatScore(it->second.info.format);
		if (f <= 0) continue;
		const auto& e = it->second.info.extent;
		float size = 0.2f;
		if (g_depth.image != VK_NULL_HANDLE && e.width == g_depth.extent.width && e.height == g_depth.extent.height) size = 1.0f;
		else if (std::abs(static_cast<double>(e.width) * e.height / display_area - 1.0) < 0.05) size = 0.8f;
		else if (e.width * 2 >= display.width) size = 0.5f;
		float s = f + size + (u.color != 0 ? 0.2f : 0.0f);
		if (SameSig(it->second, g_prof_motion)) s += 0.5f;
		if (s > best_ms) { best_ms = s; best_m = &it->second; best_m_image = image; }
	}
	Settle(g_motion, best_m, best_m_image, best_ms);
	if (best_m != nullptr) {
		float conf = 0.6f * std::min(1.0f, best_ms / 2.2f) + 0.4f * std::min(1.0f, g_motion.stable / 60.0f);
		if (SameSig(*best_m, g_prof_motion)) conf = std::max(conf, 0.5f + 0.5f * std::min(1.0f, g_motion.stable / 10.0f));
		g_motion.confidence = std::min(1.0f, conf);
		g_motion.why = best_m->info.format == VK_FORMAT_R16G16_SFLOAT ? "RG16F written each frame" : "2-channel target";
	}

	// The backbuffer: the display-size color image the frame's last pass (ours excluded) wrote; the HUD point: its
	// first pass of the frame (the scene composite), everything after it into the same image counted as UI passes.
	VkImage bb = VK_NULL_HANDLE;
	for (int p = passes - 1; p >= 0 && bb == VK_NULL_HANDLE; --p) {
		VkImage t = pass_target[p];
		if (t == VK_NULL_HANDLE) continue;
		auto it = g_images.find(t);
		if (it != g_images.end() && it->second.info.extent.width == display.width && it->second.info.extent.height == display.height)
			bb = t;
	}
	auto bbit = bb != VK_NULL_HANDLE ? g_images.find(bb) : g_images.end();
	Settle(g_backbuffer, bbit != g_images.end() ? &bbit->second : nullptr, bb, 1.0f);
	if (bb != VK_NULL_HANDLE) {
		int count = 0;
		for (int p = 0; p < passes; ++p) count += pass_target[p] == bb;
		const int hud_after = uses[bb].first;
		g_hud_stable = (hud_after == g_hud_after) ? g_hud_stable + 1 : 0;
		g_hud_after  = hud_after;
		g_ui_passes  = count - 1;
		g_hud_conf   = (g_ui_passes > 0 ? 0.4f : 0.1f) + 0.3f * std::min(1.0f, g_backbuffer.stable / 60.0f) +
		             0.3f * std::min(1.0f, g_hud_stable / 60.0f);
	} else {
		g_hud_conf *= 0.95f;
	}
	if (g_frames % 300 == 0)
		Log("[analyze] frame %llu: %d passes; depth %s %.2f (fmt %d %ux%u, %s); motion %s %.2f (fmt %d %ux%u); hud %.2f "
		    "(after pass %d, %d UI passes)",
		    static_cast<unsigned long long>(g_frames), passes, g_depth.image ? "found" : "none", g_depth.confidence,
		    g_depth.format, g_depth.extent.width, g_depth.extent.height, g_depth.why.c_str(),
		    g_motion.image ? "found" : "none", g_motion.confidence, g_motion.format, g_motion.extent.width,
		    g_motion.extent.height, g_hud_conf, g_hud_after, g_ui_passes);
	SaveProfileIfChanged();
}

namespace {

void DestroyCopy(Copy& c) {
	if (c.view) R().vkDestroyImageView(R().device, c.view, nullptr);
	if (c.image) R().vkDestroyImage(R().device, c.image, nullptr);
	if (c.memory) R().vkFreeMemory(R().device, c.memory, nullptr);
	overlay::ForgetTexture(c.view);
	c = {};
}

bool EnsureCopy(Copy& c, VkFormat format, VkExtent2D extent) {
	if (c.image != VK_NULL_HANDLE && c.format == format && c.extent.width == extent.width && c.extent.height == extent.height)
		return true;
	DestroyCopy(c);
	VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ci.imageType   = VK_IMAGE_TYPE_2D;
	ci.format      = format;
	ci.extent      = {extent.width, extent.height, 1};
	ci.mipLevels   = 1;
	ci.arrayLayers = 1;
	ci.samples     = VK_SAMPLE_COUNT_1_BIT;
	ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
	ci.usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	if (R().vkCreateImage(R().device, &ci, nullptr, &c.image) != VK_SUCCESS) { c.image = VK_NULL_HANDLE; return false; }
	VkMemoryRequirements req {};
	R().vkGetImageMemoryRequirements(R().device, c.image, &req);
	VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	ai.allocationSize  = req.size;
	ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (R().vkAllocateMemory(R().device, &ai, nullptr, &c.memory) != VK_SUCCESS ||
	    R().vkBindImageMemory(R().device, c.image, c.memory, 0) != VK_SUCCESS) {
		DestroyCopy(c);
		return false;
	}
	VkImageViewCreateInfo vi {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	vi.image    = c.image;
	vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vi.format   = format;
	vi.subresourceRange = {static_cast<VkImageAspectFlags>(IsDepthFormat(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
	if (R().vkCreateImageView(R().device, &vi, nullptr, &c.view) != VK_SUCCESS) c.view = VK_NULL_HANDLE;
	c.format  = format;
	c.extent  = extent;
	c.written = false;
	Log("[analyze] copy image %dx%d format %d made", extent.width, extent.height, format);
	return true;
}

void Barrier(VkCommandBuffer cb, VkImage image, VkFormat format, VkImageLayout from, VkImageLayout to) {
	VkImageMemoryBarrier b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	b.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
	b.dstAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
	b.oldLayout           = from;
	b.newLayout           = to;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = image;
	VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
	if (IsDepthFormat(format)) aspect = VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
	b.subresourceRange = {aspect, 0, 1, 0, 1};
	R().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
	                         nullptr, 1, &b);
}

// The source image (left in the layout the game had it in) into our copy (left in SHADER_READ_ONLY_OPTIMAL).
bool CopyInto(VkCommandBuffer cb, const Pick& pick, Copy& c) {
	if (pick.image == VK_NULL_HANDLE) return false;
	auto it = g_images.find(pick.image);
	if (it == g_images.end()) return false;
	const Img& src = it->second;
	if (!(src.info.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return false;
	const VkImageLayout layout = src.layout;
	if (layout == VK_IMAGE_LAYOUT_UNDEFINED || layout == VK_IMAGE_LAYOUT_PREINITIALIZED) return false;
	if (!EnsureCopy(c, src.info.format, {src.info.extent.width, src.info.extent.height})) return false;
	Barrier(cb, pick.image, src.info.format, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	Barrier(cb, c.image, c.format, c.written ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
	        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkImageCopy region {};
	region.srcSubresource = {static_cast<VkImageAspectFlags>(IsDepthFormat(c.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
	region.dstSubresource = region.srcSubresource;
	region.extent         = {c.extent.width, c.extent.height, 1};
	R().vkCmdCopyImage(cb, pick.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	Barrier(cb, pick.image, src.info.format, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout);
	Barrier(cb, c.image, c.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	c.written = true;
	return true;
}

} // namespace

Inputs RecordCopies(VkCommandBuffer cb) {
	Inputs in;
	std::lock_guard lock(g_mutex);
	const float need  = Cfg().min_confidence;
	const bool  feed  = Cfg().use_game_inputs;
	const bool  show  = Cfg().analyze && Cfg().previews && overlay::MenuVisible();
	if ((feed && g_depth.confidence >= need) || show) in.depth_ok = CopyInto(cb, g_depth, g_copy_depth) && g_depth.confidence >= need;
	if ((feed && g_motion.confidence >= need) || show) in.motion_ok = CopyInto(cb, g_motion, g_copy_motion) && g_motion.confidence >= need;
	if (show) CopyInto(cb, g_backbuffer, g_copy_color);
	in.depth         = g_copy_depth.image;
	in.depth_format  = g_copy_depth.format;
	in.depth_extent  = g_copy_depth.extent;
	in.motion        = g_copy_motion.image;
	in.motion_format = g_copy_motion.format;
	in.motion_extent = g_copy_motion.extent;
	const int mode   = Cfg().depth_mode;
	in.reversed_depth = mode == 2 || (mode == 0 && g_reversed_known && g_reversed);
	if (!feed) in.depth_ok = in.motion_ok = false;
	return in;
}

void DrawMenu() {
	std::lock_guard lock(g_mutex);
	auto& s = Cfg();
	ImGui::Text("Frames analizados: %llu, pases este frame: %d", static_cast<unsigned long long>(g_frames), g_passes_last);
	const auto row = [&](const char* name, const Pick& p, const Copy& c, const char* extra) {
		const ImVec4 col = p.confidence >= s.min_confidence ? ImVec4(0.4f, 1, 0.4f, 1) : ImVec4(1, 0.8f, 0.3f, 1);
		ImGui::TextColored(col, "%-8s %3.0f%%", name, p.confidence * 100.0f);
		ImGui::SameLine();
		if (p.image != VK_NULL_HANDLE) ImGui::Text("fmt %d  %ux%u  %s", p.format, p.extent.width, p.extent.height, extra);
		else ImGui::TextDisabled("buscando...");
		if (s.previews && c.written && c.view != VK_NULL_HANDLE) {
			if (void* tex = overlay::Texture(c.view)) ImGui::Image(reinterpret_cast<ImTextureID>(tex), ImVec2(192, 108));
		}
	};
	row("Depth", g_depth, g_copy_depth, g_depth.why.c_str());
	row("Motion", g_motion, g_copy_motion, g_motion.why.c_str());
	char hud[96];
	std::snprintf(hud, sizeof hud, "HUD tras pase %d, %d pases UI", g_hud_after, g_ui_passes);
	Pick hp = g_backbuffer;
	hp.confidence = g_hud_conf;
	row("HUD", hp, g_copy_color, hud);
	ImGui::SliderFloat("Confianza minima", &s.min_confidence, 0.3f, 0.95f, "%.2f");
	ImGui::Checkbox("Usar depth/vectores del juego en FG", &s.use_game_inputs);
	ImGui::Checkbox("Previews", &s.previews);
	ImGui::Combo("Unidades MV", &s.mv_units, "UV (x tamano)\0Pixeles\0");
	bool flip = s.mv_sign > 0;
	if (ImGui::Checkbox("Invertir signo MV", &flip)) s.mv_sign = flip ? 1 : -1;
	ImGui::Combo("Depth", &s.depth_mode, "Auto\0Normal\0Invertido (reversed Z)\0");
	ImGui::TextDisabled("Perfil: injector_profiles\\%s.ini", ExeName().c_str());
}

void Shutdown() {
	std::lock_guard lock(g_mutex);
	DestroyCopy(g_copy_depth);
	DestroyCopy(g_copy_motion);
	DestroyCopy(g_copy_color);
	// Everything learned about this device's images goes with it (the profile stays).
	g_images.clear();
	g_views.clear();
	g_cb.clear();
	g_frame.clear();
	g_swap.clear();
	g_depth = g_motion = g_backbuffer = Pick {};
	g_hud_after = -1;
	g_hud_stable = 0;
	g_hud_conf = 0;
}

} // namespace inj::analyzer
