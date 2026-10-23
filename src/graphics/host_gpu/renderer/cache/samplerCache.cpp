#include "graphics/host_gpu/renderer/cache/samplerCache.h"

#include "common/assert.h"

#include <cstdio>
#include <cstdlib>
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "loader/demonsSoulsRenderSize.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Libs::Graphics {

// KYTY_TEXTURE_LOD_BIAS=<float|auto> (default off): an extra mip LOD bias for scene textures when the 3D renders below
// the display and an upscaler (DLSS/XeSS/FSR) makes the display image - the vendors' log2(render/display), so the mips
// are picked for the output's resolution, not the render's. "auto": log2(render height / display height) from what
// KYTY_UPSCALE_RATIO resolved to (loader/demonsSoulsRenderSize), clamped to [-2, 0]; a number: that bias exactly
// (e.g. -1 more for DLSS). Process-constant, so the guest-field sampler key stays valid.
static float TextureLodBias() {
	static const float bias = [] {
		const char* e = std::getenv("KYTY_TEXTURE_LOD_BIAS");
		if (e == nullptr || *e == 0) return 0.0f;
		float v = 0.0f;
		if (std::strcmp(e, "auto") == 0) {
			uint32_t render  = Loader::DemonsSoulsRenderSize::g_lod_render_height;
			uint32_t display = Loader::DemonsSoulsRenderSize::g_lod_display_height;
			if (render == 0 || display == 0) {
				const char* r = std::getenv("KYTY_UPSCALE_RATIO");
				const double ratio = r != nullptr ? std::strtod(r, nullptr) : 0.0;
				if (ratio >= 1.0) {
					display = 2160;
					render  = static_cast<uint32_t>(std::lround(display / ratio));
				}
			}
			if (render != 0 && display != 0)
				v = std::clamp(static_cast<float>(std::log2(static_cast<double>(render) / display)), -2.0f, 0.0f);
			std::printf("Texture LOD bias: %.3f (auto from render %u / display %u)\n", v, render, display);
		} else {
			v = static_cast<float>(std::atof(e));
			std::printf("Texture LOD bias: %.3f (KYTY_TEXTURE_LOD_BIAS)\n", v);
		}
		std::fflush(stdout);
		return v;
	}();
	return bias;
}

SamplerCache::~SamplerCache() {
	for (const auto& [key, sampler]: m_samplers) {
		(void)key;
		m_graphics.device.destroySampler(sampler, nullptr);
	}
}

vk::Sampler SamplerCache::GetSampler(const ShaderSamplerResource& r) {
	const SamplerKey key {r.fields[0], r.fields[1], r.fields[2], r.fields[3]};
	// Samplers live as long as the cache: a thread's recent lookups need neither the lock nor
	// the map.
	struct Recent {
		const SamplerCache* owner = nullptr;
		SamplerKey          key {};
		vk::Sampler         sampler;
	};
	thread_local std::array<Recent, 64> recent {};
	auto& slot = recent[SamplerKeyHash {}(key) % recent.size()];
	if (slot.owner == this && slot.key == key) return slot.sampler;

	Common::LockGuard lock(m_mutex);
	if (auto iter = m_samplers.find(key); iter != m_samplers.end()) {
		slot = {this, key, iter->second};
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
	// set below: the guest itself asks for anisotropic filtering (ratio > 1) - Demon's Souls' surface materials do,
	// its fog/noise/particle textures don't
	bool guest_aniso = false;
	if (aniso) {
		switch (static_cast<Prospero::SamplerAnisoRatio>(r.MaxAnisoRatio())) {
			case Prospero::SamplerAnisoRatio::kOne: aniso_ratio = 1.0f; break;
			case Prospero::SamplerAnisoRatio::kTwo: aniso_ratio = 2.0f; break;
			case Prospero::SamplerAnisoRatio::kFour: aniso_ratio = 4.0f; break;
			case Prospero::SamplerAnisoRatio::kEight: aniso_ratio = 8.0f; break;
			case Prospero::SamplerAnisoRatio::kSixteen: aniso_ratio = 16.0f; break;
			default:
				EXIT("unknown ratio: %d dwords=%08x,%08x,%08x,%08x\n",
				     static_cast<int>(r.MaxAnisoRatio()), r.fields[0], r.fields[1], r.fields[2],
				     r.fields[3]);
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

	// The border as a float color: the game's sampled textures are UNORM, float and depth formats, which Vulkan gives
	// no defined value from an INT border (its bits as a float: opaque white read as ~0, so a shadow map's compare
	// sampler clamped to a white border - "lit" past the map - came back "in shadow"; the 'low' quality convars of
	// -Turbo shorten sunShadowMaxDistance, and everything past it sampled that border: distant surfaces dark, lit up
	// when the camera came close). Black is zero either way. KYTY_SAMPLER_INT_BORDER=1: the INT colors as before.
	static const bool int_border = [] {
		const char* value = std::getenv("KYTY_SAMPLER_INT_BORDER");
		return value != nullptr && value[0] == '1';
	}();
	vk::BorderColor border = int_border ? vk::BorderColor::eIntTransparentBlack : vk::BorderColor::eFloatTransparentBlack;
	const auto      note_border = [&](const char* what) {
		static uint32_t border_count = 0; // (under m_mutex)
		const uint32_t  n = ++border_count;
		if (n <= 4 || (n & (n - 1)) == 0) {
			std::printf("Samplers: %u with a %s border (%s colors; KYTY_SAMPLER_INT_BORDER=1 for INT)\n", n, what,
			            int_border ? "INT" : "float");
			std::fflush(stdout);
		}
	};
	switch (static_cast<Prospero::SamplerBorderColor>(r.BorderColorType())) {
		case Prospero::SamplerBorderColor::kTransBlack:
			border = int_border ? vk::BorderColor::eIntTransparentBlack : vk::BorderColor::eFloatTransparentBlack;
			break;
		case Prospero::SamplerBorderColor::kOpaqueBlack:
			border = int_border ? vk::BorderColor::eIntOpaqueBlack : vk::BorderColor::eFloatOpaqueBlack;
			note_border("opaque black");
			break;
		case Prospero::SamplerBorderColor::kOpaqueWhite:
			border = int_border ? vk::BorderColor::eIntOpaqueWhite : vk::BorderColor::eFloatOpaqueWhite;
			note_border("opaque white");
			break;
		case Prospero::SamplerBorderColor::kFromTable:
			LOGF(
			    "temporary: approximating table border color as transparent black, index = %" PRIu16
			    "\n",
			    r.BorderColorPtr());
			border = int_border ? vk::BorderColor::eIntTransparentBlack : vk::BorderColor::eFloatTransparentBlack;
			note_border("table (as transparent black)");
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
	sampler_info.mipLodBias =
	    static_cast<float>(static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u)) / 256.0f;
	guest_aniso = aniso && aniso_ratio > 1.0f;
	sampler_info.anisotropyEnable        = (aniso ? VK_TRUE : VK_FALSE);
	sampler_info.maxAnisotropy           = aniso_ratio;
	// KYTY_FORCE_AF=2..16 (launcher "Filtrado x16"): forced only on scene texture samplers - the same filter as
	// KYTY_TEXTURE_LOD_BIAS: linear min filter + linear mips, MaxLod >= 1 and > MinLod, no depth compare, normalized
	// coordinates. Point/nearest, unnormalized, shadow and no-mip (render target / post-process / fog lookup) samplers
	// keep the guest's settings: forcing AF on them broke Demon's Souls' ground mist (white puffs along walls).
	// The image (3D/volume or not) is not known here - samplers are cached by descriptor alone.
	{
		static const float force_af = []() {
			const char* e = std::getenv("KYTY_FORCE_AF");
			float v = (e != nullptr ? static_cast<float>(std::atof(e)) : 0.0f);
			return (v >= 2.0f ? (v > 16.0f ? 16.0f : v) : 0.0f);
		}();
		if (force_af > 0.0f) {
			// counters under m_mutex: 0 applied, 1 min/mag not linear, 2 mip not linear, 3 depth compare,
			// 4 unnormalized, 5 no mips (MaxLod < 1 or <= MinLod), 6 guest aniso already >= forced,
			// 7 KYTY_FORCE_AF_STRICT=1 and the guest asks for no aniso
			static const bool strict = [] {
				const char* e = std::getenv("KYTY_FORCE_AF_STRICT");
				return e != nullptr && e[0] == '1';
			}();
			static uint32_t cnt[8] = {};
			static const char* const names[8] = {"applied", "filter-not-linear", "mip-not-linear", "depth-compare",
			                                     "unnormalized", "no-mips", "guest-aniso", "no-guest-aniso"};
			int reason = 0;
			if (sampler_info.minFilter != vk::Filter::eLinear || sampler_info.magFilter != vk::Filter::eLinear) {
				reason = 1;
			} else if (sampler_info.mipmapMode != vk::SamplerMipmapMode::eLinear) {
				reason = 2;
			} else if (r.DepthCompareFunc() != 0) {
				reason = 3;
			} else if (r.ForceUnormCoords()) {
				reason = 4;
			} else if (!(max_lod >= 1.0f && max_lod > min_lod)) {
				reason = 5;
			} else if (strict && !guest_aniso) {
				reason = 7;
			} else if (!(force_af > sampler_info.maxAnisotropy)) {
				reason = 6;
			}
			if (reason == 0) {
				sampler_info.anisotropyEnable = VK_TRUE;
				sampler_info.maxAnisotropy    = force_af;
			}
			const uint32_t total = 0u + cnt[0] + cnt[1] + cnt[2] + cnt[3] + cnt[4] + cnt[5] + cnt[6] + cnt[7] + 1u;
			cnt[reason]++;
			if (total <= 64 || (total & (total - 1)) == 0) {
				std::printf("Force AF x%.0f%s: %u samplers - %s %u, skipped: %s %u, %s %u, %s %u, %s %u, %s %u, %s %u, "
				            "%s %u\n",
				            force_af, strict ? " strict" : "", total, names[0], cnt[0], names[1], cnt[1], names[2], cnt[2],
				            names[3], cnt[3], names[4], cnt[4], names[5], cnt[5], names[6], cnt[6], names[7], cnt[7]);
				std::fflush(stdout);
			}
		}
	}
	// KYTY_TEXTURE_LOD_BIAS: scene textures only - mipmapped (mip filter on, a max LOD of at least 1), linear, not
	// depth-compare (shadow maps) nor unnormalized: render targets and post-process buffers are sampled with no mips
	// (max LOD 0) and keep the guest's bias. Added to the guest's own, clamped to the device limit.
	// Also skipped: clamp-to-edge/border on both U and V - particle, smoke and decal sprites clamp while scene materials
	// wrap or mirror; a negative bias sharpened the blurred low mips of Demon's Souls' ground mist into bright puffs.
	// (The image's mip count/size is not known here - samplers are cached by descriptor alone.)
	if (const float extra = TextureLodBias(); extra != 0.0f) {
		const auto clamps = [](vk::SamplerAddressMode m) {
			return m == vk::SamplerAddressMode::eClampToEdge || m == vk::SamplerAddressMode::eClampToBorder;
		};
		// Default: only samplers where the guest asks for aniso (ratio > 1) - fog/noise textures wrap, so the clamp
		// rule alone missed them. KYTY_TEXTURE_LOD_BIAS_ALL=1: the older selection (scene filter + not UV-clamped).
		static const bool all = [] {
			const char* e = std::getenv("KYTY_TEXTURE_LOD_BIAS_ALL");
			return e != nullptr && e[0] == '1';
		}();
		// counters under m_mutex: 0 applied, 1 not linear min/mip, 2 depth compare, 3 unnormalized, 4 no mips,
		// 5 clamped UV (ALL mode), 6 no guest aniso (default mode)
		static uint32_t cnt[7] = {};
		int reason = 0;
		if (sampler_info.mipmapMode != vk::SamplerMipmapMode::eLinear || sampler_info.minFilter != vk::Filter::eLinear) {
			reason = 1;
		} else if (r.DepthCompareFunc() != 0) {
			reason = 2;
		} else if (r.ForceUnormCoords()) {
			reason = 3;
		} else if (!(max_lod >= 1.0f && max_lod > min_lod)) {
			reason = 4;
		} else if (!all && !guest_aniso) {
			reason = 6;
		} else if (all && clamps(sampler_info.addressModeU) && clamps(sampler_info.addressModeV)) {
			reason = 5;
		}
		if (reason == 0) {
			static const float limit = m_graphics.physical_device_properties.limits.maxSamplerLodBias;
			sampler_info.mipLodBias = std::clamp(sampler_info.mipLodBias + extra, -limit, limit);
		}
		cnt[reason]++;
		const uint32_t total = cnt[0] + cnt[1] + cnt[2] + cnt[3] + cnt[4] + cnt[5] + cnt[6];
		if (total <= 64 || (total & (total - 1)) == 0) {
			std::printf("Texture LOD bias %.3f (%s): %u samplers - applied %u, skipped: filter-not-linear %u, "
			            "depth-compare %u, unnormalized %u, no-mips %u, clamped-uv %u, no-guest-aniso %u\n",
			            extra, all ? "all" : "guest-aniso", total, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6]);
			std::fflush(stdout);
		}
	}
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
		sampler_info.minLod           = 0.0f;
		sampler_info.maxLod           = 0.0f;
		sampler_info.anisotropyEnable = VK_FALSE;
		sampler_info.maxAnisotropy    = 1.0f;
		sampler_info.compareEnable    = VK_FALSE;
		sampler_info.mipLodBias       = 0.0f;
	}

	vk::Sampler vk_sampler = nullptr;
	const auto  result     = m_graphics.device.createSampler(&sampler_info, nullptr, &vk_sampler);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || vk_sampler == nullptr);

	m_samplers.emplace(key, vk_sampler);
	slot = {this, key, vk_sampler};
	return vk_sampler;
}

} // namespace Libs::Graphics
