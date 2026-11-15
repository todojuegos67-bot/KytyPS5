#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_GROUPNOTICES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_GROUPNOTICES_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

// What happened to the images bound through storage groups since a group last looked. A group's draw visits about 2,000
// candidates, nearly all of them images where FindTexture and the transit would do nothing: Marvel's Wolverine's Trask
// outpost has about 15,000 such visits a frame against about 200 changes. So an image in a group lists every change
// that gives those two something to do, and a walk visits only the listed images (RenderExecutor::RebindImages,
// CommitBindings). Each group keeps its own position, since one change concerns all of them.
class GroupNotices {
public:
	enum class Kind : uint32_t {
		Contents, // what FindTexture acts on: dirty, GPU ownership, read-back enrolment, tracking
		State,    // what a transit acts on: layout or access
	};

	// From any thread: page faults mark images dirty.
	void Add(uint32_t image, Kind kind) {
		std::scoped_lock lock {m_lock};
		const auto       end = m_end.load(std::memory_order_relaxed);
		m_entries[end % Capacity] = {image, kind};
		m_end.store(end + 1u, std::memory_order_release);
	}

	[[nodiscard]] uint64_t End() const noexcept { return m_end.load(std::memory_order_acquire); }

	// Calls visit(image, kind) for the entries [from, to). False when the oldest are gone: the caller walks everything.
	template <typename Visit>
	[[nodiscard]] bool Since(uint64_t from, uint64_t to, Visit&& visit) const {
		if (from == to) return true;
		std::scoped_lock lock {m_lock};
		if (m_end.load(std::memory_order_relaxed) - from > Capacity) return false;
		for (auto at = from; at < to; ++at) {
			const auto& entry = m_entries[at % Capacity];
			visit(entry.image, entry.kind);
		}
		return true;
	}

private:
	static constexpr uint64_t Capacity = 1u << 16u;
	struct Entry {
		uint32_t image = 0;
		Kind     kind  = Kind::Contents;
	};
	mutable std::mutex              m_lock;
	std::vector<Entry>              m_entries = std::vector<Entry>(Capacity); // half a MB, and a TextureCache can live on a stack
	std::atomic<uint64_t>           m_end {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_GROUPNOTICES_H_
