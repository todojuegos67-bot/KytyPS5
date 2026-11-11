#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstddef>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;

// The texture LOD bias of an upscaler's 3D below its output (Loader::DemonsSoulsRenderSize: the PS5 Pro path with DLSS or
// FSR 3), log2(render height / output height), added to the materials' samplers made from then on (GetSampler). Set
// before the game makes any.
void SetUpscaleLodBias(float bias);

// Whether a shader's samplers are the materials' (the upscaler's LOD bias applies): a pixel shader writing at least
// KYTY_UPSCALE_LOD_BIAS_TARGETS color targets (default 3: the G-buffer's, 5 or 6 at 1-1) or none (the depth passes,
// whose alpha tests must keep the G-buffer's coverage); not a forward one of one or two (flames, smoke, glows drawn over
// the scene), whose finer texture levels flickered under DLSS and FSR 3 (issues #232, #85). 0: every pixel shader's.
// (1-1: 224 pixel shaders without a color target, 195 of 5-8, 21 of one and 6 of two.)
[[nodiscard]] bool UpscaleBiasApplies(bool pixel, uint32_t color_targets);

class SamplerCache {
public:
	explicit SamplerCache(GraphicContext& graphics): m_graphics(graphics) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	}
	~SamplerCache();
	KYTY_CLASS_NO_COPY(SamplerCache);

	// lod_bias: the KYTY_TEXTURE_LOD_BIAS variant (descriptors.cpp decides it per image + sampler pair).
	vk::Sampler GetSampler(const ShaderSamplerResource& r, bool lod_bias = false);

	// KYTY_TEXTURE_LOD_BIAS (auto|float), 0 when off; process-constant.
	static float TextureLodBias();
	// KYTY_TEXTURE_LOD_BIAS_ALL=1: legacy sampler-level selection.
	static bool LodBiasLegacyAll();
	static bool IsSceneSampler(const ShaderSamplerResource& r);

private:
	using SamplerKey = std::array<uint32_t, 5>;

	struct SamplerKeyHash {
		std::size_t operator()(const SamplerKey& key) const {
			std::size_t hash = 0;
			for (auto value: key) {
				hash ^= static_cast<std::size_t>(value) +
				        static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
				        (hash >> 2u);
			}
			return hash;
		}
	};

	GraphicContext&                                             m_graphics;
	Common::Mutex                                               m_mutex;
	std::unordered_map<SamplerKey, vk::Sampler, SamplerKeyHash> m_samplers;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
