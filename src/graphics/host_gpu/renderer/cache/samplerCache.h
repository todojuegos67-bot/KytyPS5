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
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

class SamplerCache {
public:
	explicit SamplerCache(GraphicContext& graphics)
	    : m_graphics(graphics), m_generation(NextGeneration()) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	}
	~SamplerCache();
	KYTY_CLASS_NO_COPY(SamplerCache);

	vk::Sampler GetSampler(const ShaderSamplerResource& r, bool integer_border);
	// The sampler GetSampler would give, if it has made it already (null otherwise): never makes
	// one (KYTY_MT_GPU=1/4's translation workers).
	vk::Sampler FindSampler(const ShaderSamplerResource& r, bool integer_border);

private:
	// The S# words, the integer border variant, and whether an unnormalized sampler keeps its LOD
	// range (UnnormalizedLodEnabled).
	using SamplerKey = std::array<uint32_t, 6>;
	// Round 128: a Vulkan sampler's create info, packed: the discrete state (filters, address
	// modes, anisotropy, compare, border, unnormalized coordinates) in the first three words, then
	// the LOD bias, the minimum and the maximum LOD as float bits.
	using VulkanKey   = std::array<uint32_t, 6>;
	using DiscreteKey = std::array<uint32_t, 3>;

	struct SamplerKeyHash {
		template <size_t N>
		std::size_t operator()(const std::array<uint32_t, N>& key) const {
			std::size_t hash = 0;
			for (auto value: key) {
				hash ^= static_cast<std::size_t>(value) +
				        static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
				        (hash >> 2u);
			}
			return hash;
		}
	};

	struct LodVariant {
		float       bias    = 0.0f;
		float       min_lod = 0.0f;
		float       max_lod = 0.0f;
		vk::Sampler sampler = nullptr;
	};

	// Each cache's own number, for the threads' front caches of its samplers (GetSampler).
	static uint64_t NextGeneration();

	// Round 128: the sampler for a create info (m_mutex held): one already made for the same
	// state, a new one while the budget allows, or else the nearest in LOD of the same discrete
	// state.
	vk::Sampler FindOrCreate(const vk::SamplerCreateInfo& info);

	GraphicContext&                                             m_graphics;
	uint64_t                                                    m_generation = 0;
	Common::Mutex                                               m_mutex;
	std::unordered_map<SamplerKey, vk::Sampler, SamplerKeyHash> m_samplers;
	// Round 128: the Vulkan samplers made, by create info (each made once; never destroyed before
	// the cache, as descriptor sets and the bindless sampler arrays keep them), and those of each
	// discrete state.
	std::unordered_map<VulkanKey, vk::Sampler, SamplerKeyHash>               m_vulkan_samplers;
	std::unordered_map<DiscreteKey, std::vector<LodVariant>, SamplerKeyHash> m_variants;
	uint32_t                                                                 m_budget         = 0;
	uint64_t                                                                 m_nearest_reuses = 0;
	uint32_t                                                                 m_next_count_log = 256;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
