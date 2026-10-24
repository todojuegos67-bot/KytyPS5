#include "local/fg-discover.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::FgDiscover {

namespace {

using Clock = std::chrono::steady_clock;

struct ImageRef {
	uint32_t slot = 0;
	uint64_t address = 0;
	uint32_t format = 0, width = 0, height = 0;
	bool     storage = false, depth = false, motion = false;
};

struct Dispatch {
	uint64_t              cs = 0;
	uint32_t              gx = 0, gy = 0;
	std::vector<ImageRef> images;
};

struct FlipDraw {
	uint64_t              vs = 0, ps = 0;
	std::vector<ImageRef> sampled;
};

struct State {
	std::mutex                    mutex;
	std::vector<double>           at_s;           // when, after the first button
	size_t                        next_at = 0;
	Clock::time_point             first_button {};
	uint32_t                      frames_left = 0; // frames still to record of the current capture
	uint32_t                      frame_serial = 0;
	bool                          recording = false;
	std::vector<Dispatch>         dispatches;
	std::map<uint64_t, uint32_t>  target_draws;    // render target address -> draws
	std::map<uint64_t, Target>    targets;
	std::map<uint64_t, Target>    depth_targets;
	std::vector<std::pair<std::vector<Target>, FlipDraw>> draws; // every draw's colors (the flip ones kept at OnFlip)
	uint64_t                      last_flip = 0;
};

State& G() {
	static State state;
	return state;
}

bool MotionLike(vk::Format f) {
	switch (f) {
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16Snorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16Sint:
		case vk::Format::eR32G32Sfloat: return true;
		default: return false;
	}
}

bool DepthLike(const TextureBinding& b) {
	return b.desc.info.IsDepth() || b.desc.view_info.format == vk::Format::eD32Sfloat ||
	       b.desc.view_info.format == vk::Format::eD16Unorm || b.desc.view_info.format == vk::Format::eD32SfloatS8Uint;
}

ImageRef Ref(uint32_t slot, const TextureBinding& b) {
	ImageRef r;
	r.slot    = slot;
	r.address = b.desc.info.data.address;
	r.format  = static_cast<uint32_t>(b.desc.view_info.format != vk::Format::eUndefined ? b.desc.view_info.format
	                                                                                    : b.desc.info.pixel_format);
	r.width   = b.desc.info.extent.width;
	r.height  = b.desc.info.extent.height;
	r.storage = b.desc.type == TextureCache::BindingType::Storage;
	r.depth   = DepthLike(b);
	r.motion  = MotionLike(static_cast<vk::Format>(r.format)) || MotionLike(b.desc.info.pixel_format);
	return r;
}

std::string Describe(const ImageRef& r) {
	char text[160];
	std::snprintf(text, sizeof(text), " [%u] 0x%010" PRIx64 " fmt %u %ux%u %s%s%s", r.slot, r.address, r.format,
	              r.width, r.height, r.storage ? "s" : "t", r.depth ? " D" : "", r.motion ? " M" : "");
	return text;
}

void Print(State& s, uint64_t flip_address, uint32_t w, uint32_t h, uint32_t format) {
	const uint32_t frame = s.frame_serial;
	std::printf("[fgdisc] frame %u: flip image 0x%010" PRIx64 " fmt %u %ux%u; %zu dispatches, %zu draws\n", frame,
	            flip_address, format, w, h, s.dispatches.size(), s.draws.size());
	// The TAA-like passes first.
	uint32_t listed = 0;
	for (const auto& d: s.dispatches) {
		const ImageRef* out    = nullptr;
		const ImageRef* depth  = nullptr;
		const ImageRef* motion = nullptr;
		for (const auto& i: d.images) {
			if (i.storage && out == nullptr && !i.motion && !i.depth) out = &i;
		}
		if (out == nullptr) continue;
		std::vector<const ImageRef*> colors;
		for (const auto& i: d.images) {
			if (i.storage || i.width != out->width || i.height != out->height) continue;
			if (i.depth || s.depth_targets.count(i.address) != 0) {
				if (depth == nullptr) depth = &i;
			} else if (i.motion) {
				if (motion == nullptr) motion = &i;
			} else {
				colors.push_back(&i);
			}
		}
		if (depth == nullptr || motion == nullptr) continue;
		std::string line;
		for (const auto& i: d.images) line += Describe(i);
		std::printf("[fgdisc] frame %u: TAA-like cs %016" PRIx64 " (%ux%u groups):%s\n", frame, d.cs, d.gx, d.gy,
		            line.c_str());
		const uint32_t color   = colors.empty() ? 0u : colors[0]->slot;
		const uint32_t history = colors.size() > 1 ? colors[1]->slot : color;
		std::printf("[fgdisc] frame %u:   try KYTY_FG_TAA_SHADERS=%016" PRIx64 " KYTY_FG_TAA_SLOTS=%u,%u,%u,%u,%u\n", frame,
		            d.cs, color, depth->slot, motion->slot, history, out->slot);
		++listed;
	}
	if (listed == 0) std::printf("[fgdisc] frame %u: no TAA-like dispatch\n", frame);
	uint32_t shown = 0;
	for (const auto& d: s.dispatches) {
		if (d.images.size() < 3 || shown >= 48) continue;
		std::string line;
		for (const auto& i: d.images) line += Describe(i);
		std::printf("[fgdisc] frame %u: cs %016" PRIx64 " (%ux%u groups):%s\n", frame, d.cs, d.gx, d.gy, line.c_str());
		++shown;
	}
	std::vector<std::pair<uint32_t, uint64_t>> by_draws;
	for (const auto& [address, n]: s.target_draws) by_draws.emplace_back(n, address);
	std::sort(by_draws.rbegin(), by_draws.rend());
	for (size_t i = 0; i < by_draws.size() && i < 16; ++i) {
		const auto& t = s.targets[by_draws[i].second];
		std::printf("[fgdisc] frame %u: render target 0x%010" PRIx64 " fmt %u %ux%u: %u draws%s\n", frame, t.address,
		            t.format, t.width, t.height, by_draws[i].first, t.address == flip_address ? " (the flip image)" : "");
	}
	for (const auto& [address, t]: s.depth_targets) {
		std::printf("[fgdisc] frame %u: depth target 0x%010" PRIx64 " fmt %u %ux%u\n", frame, address, t.format, t.width,
		            t.height);
	}
	uint32_t flip_draws = 0;
	for (const auto& [colors, draw]: s.draws) {
		const bool into_flip = std::any_of(colors.begin(), colors.end(), [&](const Target& t) { return t.address == flip_address; });
		if (!into_flip || flip_draws >= 24) continue;
		std::string line;
		for (const auto& i: draw.sampled) line += Describe(i);
		std::printf("[fgdisc] frame %u: draw into the flip image vs %016" PRIx64 " ps %016" PRIx64 " samples:%s\n", frame,
		            draw.vs, draw.ps, line.c_str());
		++flip_draws;
	}
	std::fflush(stdout);
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FG_DISCOVER");
		const bool  enabled = value != nullptr && *value == '1';
		if (enabled) {
			auto& s = G();
			std::string at = std::getenv("KYTY_FG_DISCOVER_AT") != nullptr ? std::getenv("KYTY_FG_DISCOVER_AT") : "20,60,180";
			for (size_t begin = 0; begin <= at.size();) {
				const auto end = std::min(at.find(',', begin), at.size());
				if (end > begin) s.at_s.push_back(std::atof(at.substr(begin, end - begin).c_str()));
				begin = end + 1;
			}
			std::sort(s.at_s.begin(), s.at_s.end());
			std::printf("[fgdisc] on: two frames summarized %s s after the first button\n", at.c_str());
			std::fflush(stdout);
		}
		return enabled;
	}();
	return on;
}

bool Recording() {
	return Enabled() && G().recording;
}

void OnFlip(uint64_t address, uint32_t width, uint32_t height, uint32_t format, bool player_button_seen) {
	if (!Enabled()) return;
	auto&           s = G();
	std::lock_guard lock(s.mutex);
	if (s.recording) {
		Print(s, address, width, height, format);
		s.dispatches.clear();
		s.target_draws.clear();
		s.targets.clear();
		s.depth_targets.clear();
		s.draws.clear();
		s.recording = false;
		++s.frame_serial;
	}
	if (s.frames_left > 0) {
		--s.frames_left;
		s.recording = true;
		return;
	}
	if (!player_button_seen || s.next_at >= s.at_s.size()) return;
	const auto now = Clock::now();
	if (s.first_button == Clock::time_point {}) s.first_button = now;
	if (std::chrono::duration<double>(now - s.first_button).count() >= s.at_s[s.next_at]) {
		std::printf("[fgdisc] capture at %.0f s after the first button\n", s.at_s[s.next_at]);
		++s.next_at;
		s.recording   = true;
		s.frames_left = 1; // (this frame and the next)
	}
}

void OnDispatch(uint64_t cs_hash, std::span<const TextureBinding> images, uint32_t groups_x, uint32_t groups_y) {
	if (!Recording()) return;
	auto&           s = G();
	std::lock_guard lock(s.mutex);
	if (s.dispatches.size() >= 4096) return;
	Dispatch d;
	d.cs = cs_hash;
	d.gx = groups_x;
	d.gy = groups_y;
	for (uint32_t i = 0; i < images.size(); ++i) {
		if (!images[i].image_id) continue;
		d.images.push_back(Ref(i, images[i]));
	}
	s.dispatches.push_back(std::move(d));
}

void OnDraw(uint64_t vs_hash, uint64_t ps_hash, std::span<const Target> colors, const Target& depth,
            std::span<const TextureBinding> sampled) {
	if (!Recording()) return;
	auto&           s = G();
	std::lock_guard lock(s.mutex);
	for (const auto& c: colors) {
		if (c.address == 0) continue;
		++s.target_draws[c.address];
		s.targets[c.address] = c;
	}
	if (depth.address != 0) s.depth_targets[depth.address] = depth;
	if (s.draws.size() >= 30000) return;
	FlipDraw draw;
	draw.vs = vs_hash;
	draw.ps = ps_hash;
	for (uint32_t i = 0; i < sampled.size(); ++i) {
		if (!sampled[i].image_id) continue;
		draw.sampled.push_back(Ref(i, sampled[i]));
	}
	s.draws.emplace_back(std::vector<Target>(colors.begin(), colors.end()), std::move(draw));
}

} // namespace Libs::Graphics::FgDiscover
