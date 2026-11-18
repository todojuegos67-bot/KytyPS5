#include "graphics/host_gpu/renderer/cache/samplerCache.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

SamplerCache::~SamplerCache() {
	// The S# map's samplers are these (or the nearest of them): each is destroyed once.
	for (const auto& [key, sampler]: m_vulkan_samplers) {
		(void)key;
		m_graphics.device.destroySampler(sampler, nullptr);
	}
}

// KYTY_UNNORMALIZED_LOD=1, or KYTY_BISECT's unorm:lod (round 76): a sampler with unnormalized
// coordinates keeps the guest's LOD range. Vulkan allows such a sampler only LOD 0 (and a view of
// one mip), so it was made with 0 to 0, which clamps every explicit LOD to the first mip. The
// guest GPU samples the mip the shader names, at coordinates in that mip's texels: GT7's hull
// shaders read their depth pyramid that way, a level picked from each patch's size on screen,
// and the patches of car bodies close to the camera read the wrong level and were culled.
static bool UnnormalizedLodEnabled() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_UNNORMALIZED_LOD");
		return value != nullptr && value[0] == '1';
	}();
	return always || Bisect::UnnormalizedLod();
}

uint64_t SamplerCache::NextGeneration() {
	static std::atomic<uint64_t> next {1};
	return next.fetch_add(1, std::memory_order_relaxed);
}

vk::Sampler SamplerCache::FindSampler(const ShaderSamplerResource& r, bool integer_border) {
	const bool       keep_lod = r.ForceUnormCoords() && UnnormalizedLodEnabled();
	const SamplerKey key {r.fields[0], r.fields[1],    r.fields[2],
	                      r.fields[3], integer_border, keep_lod};
	struct Front {
		uint64_t    generation = 0;
		SamplerKey  key {};
		vk::Sampler sampler = nullptr;
	};
	thread_local std::array<Front, 64> front {};
	auto&                              slot = front[SamplerKeyHash {}(key) % front.size()];
	if (slot.generation == m_generation && slot.key == key) {
		return slot.sampler;
	}
	Common::LockGuard lock(m_mutex);
	if (auto iter = m_samplers.find(key); iter != m_samplers.end()) {
		slot = {m_generation, key, iter->second};
		return iter->second;
	}
	return nullptr;
}

vk::Sampler SamplerCache::GetSampler(const ShaderSamplerResource& r, bool integer_border) {
	const bool       keep_lod = r.ForceUnormCoords() && UnnormalizedLodEnabled();
	const SamplerKey key {r.fields[0], r.fields[1],    r.fields[2],
	                      r.fields[3], integer_border, keep_lod};
	// Round 80: each thread's front cache of the samplers it asked for (a sampler lives as long
	// as its cache): every sampler of every draw took the lock and a map search, ~0.5 % of the GPU
	// thread in a race.
	struct Front {
		uint64_t    generation = 0;
		SamplerKey  key {};
		vk::Sampler sampler = nullptr;
	};
	thread_local std::array<Front, 64> front {};
	auto&                              slot = front[SamplerKeyHash {}(key) % front.size()];
	if (slot.generation == m_generation && slot.key == key) {
		return slot.sampler;
	}

	Common::LockGuard lock(m_mutex);
	if (auto iter = m_samplers.find(key); iter != m_samplers.end()) {
		slot = {m_generation, key, iter->second};
		return iter->second;
	}

	float      aniso_ratio = 1.0f;
	const auto mag_filter  = r.XyMagFilter();
	const auto min_filter  = r.XyMinFilter();

	auto is_aniso_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kAnisoPoint:
			case Prospero::SamplerFilter::kAnisoLinear: return true;
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kBilinear: return false;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return false;
	};

	auto to_vk_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kAnisoPoint: return vk::Filter::eNearest;
			case Prospero::SamplerFilter::kBilinear:
			case Prospero::SamplerFilter::kAnisoLinear: return vk::Filter::eLinear;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return vk::Filter::eNearest;
	};

	const bool aniso = is_aniso_filter(mag_filter) || is_aniso_filter(min_filter);
	if (aniso) {
		switch (static_cast<Prospero::SamplerAnisoRatio>(r.MaxAnisoRatio())) {
			case Prospero::SamplerAnisoRatio::kOne: aniso_ratio = 1.0f; break;
			case Prospero::SamplerAnisoRatio::kTwo: aniso_ratio = 2.0f; break;
			case Prospero::SamplerAnisoRatio::kFour: aniso_ratio = 4.0f; break;
			case Prospero::SamplerAnisoRatio::kEight: aniso_ratio = 8.0f; break;
			case Prospero::SamplerAnisoRatio::kSixteen:
			// Reserved encodings 5-7 fall back to the maximum ratio.
			default: aniso_ratio = 16.0f; break;
		}
	}

	const auto mip_filter = r.MipFilter();
	float      min_lod    = 0.0f;
	float      max_lod    = 0.0f;
	if (static_cast<Prospero::SamplerMipFilter>(mip_filter) != Prospero::SamplerMipFilter::kNone) {
		min_lod = static_cast<float>(r.MinLod()) / 256.0f;
		max_lod = static_cast<float>(r.MaxLod()) / 256.0f;
	}

	vk::SamplerCreateInfo sampler_info {};

	auto to_vk_address_mode = [](uint8_t clamp) {
		switch (static_cast<Prospero::SamplerClampMode>(clamp)) {
			case Prospero::SamplerClampMode::kWrap: return vk::SamplerAddressMode::eRepeat;
			case Prospero::SamplerClampMode::kMirror:
				return vk::SamplerAddressMode::eMirroredRepeat;
			case Prospero::SamplerClampMode::kClampLastTexel:
				return vk::SamplerAddressMode::eClampToEdge;
			case Prospero::SamplerClampMode::kMirrorOnceLastTexel:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampHalfBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceHalfBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			default: EXIT("unknown clamp: %u\n", clamp);
		}
		return vk::SamplerAddressMode::eClampToBorder;
	};

	vk::BorderColor border = integer_border ? vk::BorderColor::eIntTransparentBlack
	                                       : vk::BorderColor::eFloatTransparentBlack;
	switch (static_cast<Prospero::SamplerBorderColor>(r.BorderColorType())) {
		case Prospero::SamplerBorderColor::kTransBlack: break;
		case Prospero::SamplerBorderColor::kOpaqueBlack:
			border = integer_border ? vk::BorderColor::eIntOpaqueBlack
			                        : vk::BorderColor::eFloatOpaqueBlack;
			break;
		case Prospero::SamplerBorderColor::kOpaqueWhite:
			border = integer_border ? vk::BorderColor::eIntOpaqueWhite
			                        : vk::BorderColor::eFloatOpaqueWhite;
			break;
		case Prospero::SamplerBorderColor::kFromTable:
			LOGF(
			    "temporary: approximating table border color as transparent black, index = %" PRIu16
			    "\n",
			    r.BorderColorPtr());
			break;
		default: EXIT("unknown border color: %d", static_cast<int>(r.BorderColorType()));
	}

	sampler_info.magFilter = to_vk_filter(mag_filter);
	sampler_info.minFilter = to_vk_filter(min_filter);
	sampler_info.mipmapMode =
	    (static_cast<Prospero::SamplerMipFilter>(mip_filter) == Prospero::SamplerMipFilter::kLinear
	         ? vk::SamplerMipmapMode::eLinear
	         : vk::SamplerMipmapMode::eNearest);
	sampler_info.addressModeU = to_vk_address_mode(r.ClampX());
	sampler_info.addressModeV = to_vk_address_mode(r.ClampY());
	sampler_info.addressModeW = to_vk_address_mode(r.ClampZ());
    const auto max_lod_bias = m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerLodBias;
    sampler_info.mipLodBias = std::clamp(
        static_cast<float>(static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u)) / 256.0f,
        -max_lod_bias, max_lod_bias
    );
	sampler_info.anisotropyEnable        = (aniso ? VK_TRUE : VK_FALSE);
	sampler_info.maxAnisotropy           = aniso_ratio;
	sampler_info.compareEnable           = (r.DepthCompareFunc() != 0 ? VK_TRUE : VK_FALSE);
	sampler_info.compareOp               = static_cast<vk::CompareOp>(r.DepthCompareFunc());
	sampler_info.minLod                  = min_lod;
	sampler_info.maxLod                  = max_lod;
	sampler_info.borderColor             = border;
	sampler_info.unnormalizedCoordinates = (r.ForceUnormCoords() ? VK_TRUE : VK_FALSE);

	if (r.ForceUnormCoords()) {
		sampler_info.addressModeU     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeV     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeW     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.mipmapMode       = vk::SamplerMipmapMode::eNearest;
		sampler_info.minLod           = keep_lod ? min_lod : 0.0f;
		sampler_info.maxLod           = keep_lod ? std::max(max_lod, min_lod) : 0.0f;
		sampler_info.anisotropyEnable = VK_FALSE;
		sampler_info.maxAnisotropy    = 1.0f;
		sampler_info.compareEnable    = VK_FALSE;
		sampler_info.mipLodBias       = 0.0f;
	}

	const auto vk_sampler = FindOrCreate(sampler_info);
	// The S# map only finds the sampler for words seen before: past this many it starts again
	// (the samplers stay, in m_vulkan_samplers).
	constexpr size_t MaxSamplerKeys = 65536;
	if (m_samplers.size() >= MaxSamplerKeys) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			LOGF("Samplers: %zu S# variants seen; their map starts again (%zu Vulkan samplers)\n",
			     m_samplers.size(), m_vulkan_samplers.size());
		}
		m_samplers.clear();
	}
	m_samplers.emplace(key, vk_sampler);
	slot = {m_generation, key, vk_sampler};
	return vk_sampler;
}

namespace {

uint32_t FloatBits(float value) {
	uint32_t bits = 0;
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

// Bits 25 to 29 of a discrete key's first word: compare enable, unnormalized coordinates and the
// compare op, which a sampler standing in for another must share (FindOrCreate).
constexpr uint32_t SamplerKindBits = 0x3e000000u;

} // namespace

// Round 128: a VkSampler for each distinct create info, not for each distinct S#. Every S# the
// game used got a sampler of its own and none was freed; S#s that differ only where the Vulkan
// sampler does not (or by a LOD bias the game moves) piled up past the driver's limit
// (maxSamplerAllocationCount, 4000 on NVIDIA), and creating the next one ended the emulator after
// ten minutes of Circuit Experience retries (issue 9). Past a budget under that limit, a new LOD
// range or bias takes the nearest sampler of the same discrete state.
vk::Sampler SamplerCache::FindOrCreate(const vk::SamplerCreateInfo& info) {
	const DiscreteKey discrete {
	    static_cast<uint32_t>(info.magFilter) | (static_cast<uint32_t>(info.minFilter) << 4u) |
	        (static_cast<uint32_t>(info.mipmapMode) << 8u) |
	        (static_cast<uint32_t>(info.addressModeU) << 12u) |
	        (static_cast<uint32_t>(info.addressModeV) << 16u) |
	        (static_cast<uint32_t>(info.addressModeW) << 20u) |
	        (static_cast<uint32_t>(info.anisotropyEnable) << 24u) |
	        (static_cast<uint32_t>(info.compareEnable) << 25u) |
	        (static_cast<uint32_t>(info.unnormalizedCoordinates) << 26u) |
	        ((static_cast<uint32_t>(info.compareOp) & 7u) << 27u),
	    static_cast<uint32_t>(info.borderColor), FloatBits(info.maxAnisotropy)};
	const VulkanKey key {discrete[0],
	                     discrete[1],
	                     discrete[2],
	                     FloatBits(info.mipLodBias),
	                     FloatBits(info.minLod),
	                     FloatBits(info.maxLod)};
	if (const auto found = m_vulkan_samplers.find(key); found != m_vulkan_samplers.end()) {
		return found->second;
	}
	if (m_budget == 0) {
		// Room for the samplers made elsewhere (the blit helper's, the presenter's, the overlay's).
		const auto limit =
		    m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerAllocationCount;
		m_budget = limit > 1024u ? limit - 256u : std::max(limit / 2u, 1u);
		LOGF("Samplers: one Vulkan sampler for each create info, %u at most (the device allows "
		     "%u); past that a new LOD range takes the nearest of the same kind (round 128)\n",
		     m_budget, limit);
	}
	auto&      variants = m_variants[discrete];
	const auto nearest  = [&info](const std::vector<LodVariant>& list) -> vk::Sampler {
        const LodVariant* best          = nullptr;
        float             best_distance = 0.0f;
        for (const auto& variant: list) {
            const float distance = std::abs(variant.bias - info.mipLodBias) +
                                   std::abs(variant.min_lod - info.minLod) +
                                   std::abs(variant.max_lod - info.maxLod);
            if (best == nullptr || distance < best_distance) {
                best          = &variant;
                best_distance = distance;
            }
        }
        return best != nullptr ? best->sampler : nullptr;
	};
	const auto note_reuse = [this](const char* why) {
		m_nearest_reuses++;
		if (m_nearest_reuses <= 8 || (m_nearest_reuses & (m_nearest_reuses - 1u)) == 0) {
			LOGF("Samplers: %s; the nearest of the same kind stands in (%" PRIu64
			     " so far, %zu Vulkan samplers, %zu kinds)\n",
			     why, m_nearest_reuses, m_vulkan_samplers.size(), m_variants.size());
		}
	};
	// Any sampler of the same kind (compare, op and coordinates), the nearest in LOD of the first
	// discrete state that has one.
	const auto same_kind = [&]() -> vk::Sampler {
		for (const auto& [other, list]: m_variants) {
			if ((other[0] & SamplerKindBits) == (discrete[0] & SamplerKindBits) && !list.empty()) {
				return nearest(list);
			}
		}
		return nullptr;
	};
	if (m_vulkan_samplers.size() >= m_budget) {
		// Round 130: a new discrete state takes one of the same kind too, so the count stays at
		// the budget (round 128 made those, up to the driver's limit: in a tester's 15 minutes of
		// retries the game asked for some 2000 discrete states).
		auto stand_in = nearest(variants);
		if (stand_in == nullptr) {
			stand_in = same_kind();
		}
		if (stand_in != nullptr) {
			note_reuse("the budget is used up");
			return stand_in;
		}
	}
	vk::Sampler vk_sampler = nullptr;
	const auto  result     = m_graphics.device.createSampler(&info, nullptr, &vk_sampler);
	if (result != vk::Result::eSuccess || vk_sampler == nullptr) {
		// The nearest of the same discrete state, or else any of the same kind (compare and
		// coordinates), rather than ending the emulator.
		auto stand_in = nearest(variants);
		if (stand_in == nullptr) {
			stand_in = same_kind();
		}
		if (stand_in == nullptr) {
			EXIT("vkCreateSampler failed: %s (%zu samplers made)\n", vk::to_string(result).c_str(),
			     m_vulkan_samplers.size());
		}
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Samplers: vkCreateSampler failed: %s, with %zu made\n",
			     vk::to_string(result).c_str(), m_vulkan_samplers.size());
		}
		note_reuse("a sampler could not be made");
		return stand_in;
	}
	m_vulkan_samplers.emplace(key, vk_sampler);
	variants.push_back({info.mipLodBias, info.minLod, info.maxLod, vk_sampler});
	if (m_vulkan_samplers.size() >= m_next_count_log) {
		LOGF("Samplers: %zu Vulkan samplers made (%zu kinds), for %zu S# variants\n",
		     m_vulkan_samplers.size(), m_variants.size(), m_samplers.size() + 1u);
		m_next_count_log *= 2u;
	}
	return vk_sampler;
}

} // namespace Libs::Graphics
