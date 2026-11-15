#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEAHEAD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEAHEAD_H_

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Graphics and compute ahead (pipelineCache.cpp): the parts that need no GPU, apart for their tests.

// The ahead workers on a machine with this many hardware threads: all but four, which stay for the game's own threads,
// the GPU thread and the scanner, and at most 16; never fewer than the half (at most 8) the pool had before.
constexpr uint32_t AheadWorkerCount(uint32_t hardware_threads) {
	const uint32_t half = std::min(hardware_threads / 2u, 8u);
	const uint32_t most = hardware_threads > 4u ? std::min(hardware_threads - 4u, 16u) : 0u;
	return std::max({half, most, 1u});
}

// One translation of a shader program among the scanner and the workers. The first item to need it on a worker
// translates it; the items that need it meanwhile wait here, and are queued again once it is done, instead of
// translating it a second time or holding a worker. The scanner never translates: it hands the item over.
// Every call is under the owner's lock; the translation itself is not.
template <typename Item>
class AheadTranslation {
public:
	enum class Step : uint8_t {
		Ready,       // translated: go on
		Translate,   // this caller translates it, then calls Finish
		Waiting,     // being translated: the item waits here
		Unsupported, // the translation failed or cannot be done ahead
	};

	[[nodiscard]] Step Take(Item item) {
		if (m_unsupported) return Step::Unsupported;
		if (m_ready) return Step::Ready;
		if (m_translating) {
			m_waiting.push_back(std::move(item));
			return Step::Waiting;
		}
		m_translating = true;
		return Step::Translate;
	}
	// Without an item (the scanner's thread): Waiting means "not translated yet", and nothing is kept.
	[[nodiscard]] Step Peek() const {
		if (m_unsupported) return Step::Unsupported;
		return m_ready ? Step::Ready : Step::Waiting;
	}
	// The translation is done, or failed: the items that waited, to be queued again.
	[[nodiscard]] std::vector<Item> Finish(bool supported) {
		m_translating = false;
		m_unsupported = m_unsupported || !supported;
		m_ready       = !m_unsupported;
		return std::exchange(m_waiting, {});
	}
	// A program that cannot be built ahead, found outside a translation.
	void SetUnsupported() { m_unsupported = true; }

	[[nodiscard]] bool Ready() const { return m_ready; }
	[[nodiscard]] bool Unsupported() const { return m_unsupported; }

private:
	bool              m_ready       = false;
	bool              m_translating = false;
	bool              m_unsupported = false;
	std::vector<Item> m_waiting;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEAHEAD_H_
