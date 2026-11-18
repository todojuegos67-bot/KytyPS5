#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TEXTUREWORKERVIEW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TEXTUREWORKERVIEW_H_

#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// KYTY_MT_GPU=1/4 (mtGpuWorkers.h): what the translation workers read of the texture cache,
// without its locks. Read-copy-update: every entry is an immutable record behind an atomic
// shared_ptr; the GPU thread (the only writer, under the cache's locks) publishes a new record
// when it registers, unregisters or changes an image, and always BEFORE it stamps the write
// registry over the image (MtGpu::NoteWrite). A worker reads the registry's epoch before it reads
// anything here, so a record it read stale is one whose change was stamped after its epoch: the
// GPU thread rejects that translation (ZonesWrittenSince). A worker never creates anything here:
// what it does not find fails its draw.
namespace Libs::Graphics {

class TextureWorkerView {
public:
	// What a worker reads of one image (registered images, null images).
	struct Record {
		ImageId           id;
		ImageInfo         info;
		ImageId           depth_id; // a depth image's stencil plane: the depth image
		uint32_t          samples        = 0; // the backing's (Image::backing)
		uint32_t          backing_layers = 0;
		vk::ImageType     backing_type   = vk::ImageType::e2D;
		vk::Format        backing_format = vk::Format::eUndefined;
		bool              has_backing     = false;
		bool              read_only_alias = false;
	};
	// A lookup the GPU thread's FindImage resolved through the overlap loop with no change to the
	// image set (an image found that starts at the descriptor's address with another size or mips,
	// or one the descriptor lies in): its result, for the same description again.
	struct Resolution {
		uint64_t      hash = 0;
		ImageInfo     info;
		ImageViewInfo view_info;
		uint8_t       type        = 0; // TextureCache::BindingType
		bool          exact       = false;
		ImageId       result;
		uint32_t      base_level  = 0; // the view base FindImage gave the description
		uint32_t      base_layer  = 0;
		uint64_t      range_stamp = 0; // TextureCache::RangeStamp of the description's memory then
		uint64_t      epoch       = 0; // the write registry's then (MtGpu::Epoch)
	};
	// What a worker's FindImage calls found, for the GPU thread: the images (to touch them as
	// FindImage does), and for each overlap result taken from the view, the ranges that must
	// not have been written since it was kept (GPU or host writes change how an overlap
	// resolves: the GPU thread rejects the translation then).
	struct FoundLog {
		struct Since {
			uint64_t address = 0;
			uint64_t size    = 0;
			uint64_t epoch   = 0;
		};
		std::vector<ImageId> images;
		std::vector<Since>   unwritten;
		void                 Clear() {
            images.clear();
            unwritten.clear();
		}
	};

	TextureWorkerView() {
		for (auto& c: m_chunks) {
			c.store(nullptr, std::memory_order_relaxed);
		}
	}
	~TextureWorkerView() {
		for (auto& c: m_chunks) {
			delete c.load(std::memory_order_relaxed);
		}
	}
	TextureWorkerView(const TextureWorkerView&)            = delete;
	TextureWorkerView& operator=(const TextureWorkerView&) = delete;

	// Worker: the record of image `id` (this slot generation), or null.
	[[nodiscard]] std::shared_ptr<const Record> Get(ImageId id) const {
		if (!id || (id.index >> ChunkBits) >= MaxChunks) {
			return {};
		}
		const auto* chunk = m_chunks[id.index >> ChunkBits].load(std::memory_order_acquire);
		if (chunk == nullptr) {
			return {};
		}
		auto record = chunk->slots[id.index & (ChunkSize - 1u)].load(std::memory_order_acquire);
		if (record == nullptr || record->id != id) {
			return {};
		}
		return record;
	}
	// Worker: the images that start at `address` (registered), in no particular order.
	[[nodiscard]] std::shared_ptr<const std::vector<std::pair<uint64_t, ImageId>>>
	StartsBucket(uint64_t address) const {
		return m_starts[StartSlot(address)].load(std::memory_order_acquire);
	}
	// Worker: the null image for a format and type, if made.
	[[nodiscard]] ImageId NullImage(vk::Format format, Prospero::ImageType type) const {
		const auto list = m_nulls.load(std::memory_order_acquire);
		if (list != nullptr) {
			for (const auto& [key, id]: *list) {
				if (key.first == format && key.second == type) {
					return id;
				}
			}
		}
		return {};
	}
	// Worker: the resolutions kept for descriptions with this hash.
	[[nodiscard]] std::shared_ptr<const std::vector<Resolution>> Resolutions(uint64_t hash) const {
		return m_resolutions[hash & (ResolutionSlots - 1u)].load(std::memory_order_acquire);
	}

	// GPU thread.
	void Set(ImageId id, std::shared_ptr<const Record> record) {
		if (!id || (id.index >> ChunkBits) >= MaxChunks) {
			return;
		}
		auto& slot  = m_chunks[id.index >> ChunkBits];
		auto* chunk = slot.load(std::memory_order_relaxed);
		if (chunk == nullptr) {
			if (record == nullptr) {
				return;
			}
			chunk = new Chunk;
			slot.store(chunk, std::memory_order_release);
		}
		chunk->slots[id.index & (ChunkSize - 1u)].store(std::move(record),
		                                                std::memory_order_release);
	}
	void AddStart(uint64_t address, ImageId id) {
		auto&      slot = m_starts[StartSlot(address)];
		const auto old  = slot.load(std::memory_order_relaxed);
		auto       next = old != nullptr
		                      ? std::make_shared<std::vector<std::pair<uint64_t, ImageId>>>(*old)
		                      : std::make_shared<std::vector<std::pair<uint64_t, ImageId>>>();
		next->emplace_back(address, id);
		slot.store(std::move(next), std::memory_order_release);
	}
	void RemoveStart(uint64_t address, ImageId id) {
		auto&      slot = m_starts[StartSlot(address)];
		const auto old  = slot.load(std::memory_order_relaxed);
		if (old == nullptr) {
			return;
		}
		auto next = std::make_shared<std::vector<std::pair<uint64_t, ImageId>>>();
		next->reserve(old->size());
		for (const auto& entry: *old) {
			if (!(entry.first == address && entry.second == id)) {
				next->push_back(entry);
			}
		}
		slot.store(next->empty() ? nullptr : std::move(next), std::memory_order_release);
	}
	void AddNull(vk::Format format, Prospero::ImageType type, ImageId id) {
		const auto old  = m_nulls.load(std::memory_order_relaxed);
		auto       next = old != nullptr ? std::make_shared<NullList>(*old)
		                                 : std::make_shared<NullList>();
		next->push_back({{format, type}, id});
		m_nulls.store(std::move(next), std::memory_order_release);
	}
	// Kept per slot: the newest few (a description that resolves again replaces its own).
	void AddResolution(Resolution resolution,
	                   bool (*same)(const Resolution&, const Resolution&)) {
		auto&      slot = m_resolutions[resolution.hash & (ResolutionSlots - 1u)];
		const auto old  = slot.load(std::memory_order_relaxed);
		auto       next = std::make_shared<std::vector<Resolution>>();
		next->reserve(MaxResolutionsPerSlot);
		next->push_back(std::move(resolution));
		if (old != nullptr) {
			for (const auto& kept: *old) {
				if (next->size() < MaxResolutionsPerSlot && !same(kept, next->front())) {
					next->push_back(kept);
				}
			}
		}
		slot.store(std::move(next), std::memory_order_release);
	}

private:
	static constexpr uint32_t ChunkBits             = 10;
	static constexpr uint32_t ChunkSize             = 1u << ChunkBits;
	static constexpr uint32_t MaxChunks             = 4096; // 4M image slots
	static constexpr size_t   StartSlots            = size_t {1} << 14u;
	static constexpr size_t   ResolutionSlots       = size_t {1} << 12u;
	static constexpr size_t   MaxResolutionsPerSlot = 4;

	struct Chunk {
		std::array<std::atomic<std::shared_ptr<const Record>>, ChunkSize> slots;
	};
	using NullList = std::vector<std::pair<std::pair<vk::Format, Prospero::ImageType>, ImageId>>;

	static size_t StartSlot(uint64_t address) {
		const uint64_t h = (address >> 8u) * 0x9E3779B97F4A7C15ull;
		return static_cast<size_t>(h >> 50u) & (StartSlots - 1u);
	}

	std::array<std::atomic<Chunk*>, MaxChunks>                     m_chunks;
	std::array<std::atomic<std::shared_ptr<const std::vector<std::pair<uint64_t, ImageId>>>>,
	           StartSlots>                                         m_starts {};
	std::atomic<std::shared_ptr<const NullList>>                   m_nulls;
	std::array<std::atomic<std::shared_ptr<const std::vector<Resolution>>>, ResolutionSlots>
	    m_resolutions {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TEXTUREWORKERVIEW_H_
