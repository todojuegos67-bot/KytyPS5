// (onda4) The GPU side of local/onda-dump.h: included by renderCompute.cpp at file scope, after its includes
// (TextureBinding, TextureCache, GraphicContext, CommandScheduler, VMA). Kyty.Gpu only, except the writer thread.
#include "local/onda-dump.h"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <span>
#include <tuple>
#include <thread>

namespace OndaDump {

struct Pending {
	VkBuffer      buffer = VK_NULL_HANDLE;
	VmaAllocation mem    = nullptr;
	uint64_t      bytes = 0, tick = 0, frame = 0;
	std::string   file, meta; // file name in the session folder, its index line
	std::string   dir;
};

struct GpuState {
	VmaAllocator         allocator = nullptr;
	int                  session   = 0;
	std::string          dir;
	uint64_t             base_frame = 0, last_frame = ~0ull;
	uint32_t             frame_index = 0; // captured frames so far
	uint32_t             seq         = 0; // copies in the current captured frame
	bool                 capture_now = false;
	std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> pre_seen; // (address, level, layer) dumped before, this frame
	std::vector<std::pair<uint64_t, uint32_t>>            addr_writes;
	std::deque<Pending>  in_flight;
	std::atomic<uint64_t> held {0};
	uint64_t             copies = 0, skipped_budget = 0, skipped_format = 0, bytes_total = 0;
	// writer
	std::mutex              mutex;
	std::condition_variable cv;
	std::deque<Pending>     ready;
	bool                    writer_started = false;
	std::atomic<uint64_t>   written {0};
};

inline GpuState& G() {
	static GpuState* g = new GpuState; // (never destroyed: the writer thread may outlive static destruction)
	return *g;
}

// Bytes per texel of a copy of `aspect` (0: not dumped: block-compressed, multi-planar, unknown).
inline uint32_t TexelBytes(vk::Format f, vk::ImageAspectFlags& aspect) {
	aspect = vk::ImageAspectFlagBits::eColor;
	switch (f) {
		case vk::Format::eD32Sfloat:
		case vk::Format::eD32SfloatS8Uint: aspect = vk::ImageAspectFlagBits::eDepth; return 4;
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD24UnormS8Uint: aspect = vk::ImageAspectFlagBits::eDepth; return 4;
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: aspect = vk::ImageAspectFlagBits::eDepth; return 2;
		case vk::Format::eR8Unorm:
		case vk::Format::eR8Uint:
		case vk::Format::eR8Snorm: return 1;
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8Uint:
		case vk::Format::eR8G8Snorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16Uint:
		case vk::Format::eR16Snorm: return 2;
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eR8G8B8A8Uint:
		case vk::Format::eR8G8B8A8Snorm:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA2B10G10R10UintPack32:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eE5B9G9R9UfloatPack32:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16Snorm:
		case vk::Format::eR16G16Uint:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32Uint:
		case vk::Format::eR32Sint: return 4;
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16G16B16A16Snorm:
		case vk::Format::eR16G16B16A16Uint:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32Uint: return 8;
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eR32G32B32A32Uint: return 16;
		default: return 0;
	}
}

inline void WriterLoop() {
	auto& g = G();
	for (;;) {
		Pending p;
		{
			std::unique_lock lock(g.mutex);
			g.cv.wait(lock, [&] { return !g.ready.empty(); });
			p = std::move(g.ready.front());
			g.ready.pop_front();
		}
		void* data = nullptr;
		if (vmaInvalidateAllocation(g.allocator, p.mem, 0, VK_WHOLE_SIZE) == VK_SUCCESS &&
		    vmaMapMemory(g.allocator, p.mem, &data) == VK_SUCCESS) {
			if (FILE* f = std::fopen((p.dir + "/" + p.file).c_str(), "wb"); f != nullptr) {
				std::fwrite(data, 1, p.bytes, f);
				std::fclose(f);
				if (FILE* index = std::fopen((p.dir + "/index.jsonl").c_str(), "ab"); index != nullptr) {
					std::fwrite(p.meta.data(), 1, p.meta.size(), index);
					std::fclose(index);
				}
			}
			vmaUnmapMemory(g.allocator, p.mem);
		}
		vmaDestroyBuffer(g.allocator, p.buffer, p.mem);
		g.held.fetch_sub(p.bytes, std::memory_order_relaxed);
		g.written.fetch_add(1, std::memory_order_relaxed);
	}
}

// Every dispatch (cheap when idle): copies whose GPU work is done go to the writer.
inline void Poll(Libs::Graphics::CommandScheduler& scheduler, uint64_t frame) {
	auto& g = G();
	if (g.in_flight.empty()) return;
	while (!g.in_flight.empty()) {
		auto& p = g.in_flight.front();
		if (frame < p.frame + 8 || !scheduler.IsFree(p.tick)) break;
		{
			const std::lock_guard lock(g.mutex);
			g.ready.push_back(std::move(p));
		}
		g.cv.notify_one();
		g.in_flight.pop_front();
	}
}

// Kyty.Gpu, at the top of a dispatch's recording: starts / advances / ends the session. True: this frame is dumped.
inline bool BeginDispatch(Libs::Graphics::CommandScheduler& scheduler, uint64_t frame) {
	auto& g = G();
	Poll(scheduler, frame);
	if (const int armed = g_armed.load(std::memory_order_acquire); armed != 0 && !g_recording.load(std::memory_order_relaxed)) {
		{
			const std::lock_guard lock(g_dir_mutex);
			g.dir = g_dir;
		}
		std::error_code error;
		std::filesystem::create_directories(g.dir, error);
		g.session     = armed;
		g.base_frame  = frame;
		g.last_frame  = ~0ull;
		g.frame_index = 0;
		g.copies = g.skipped_budget = g.skipped_format = g.bytes_total = 0;
		g_recording.store(true, std::memory_order_relaxed);
		g_armed.store(0, std::memory_order_release);
		if (!g.writer_started) {
			g.writer_started = true;
			std::thread(WriterLoop).detach();
		}
	}
	if (!g_recording.load(std::memory_order_relaxed)) return false;
	if (frame != g.last_frame) {
		g.last_frame = frame;
		const uint64_t rel = frame - g.base_frame;
		const auto&    cfg = Cfg();
		if (rel >= static_cast<uint64_t>(cfg.frames) * cfg.step) {
			g_recording.store(false, std::memory_order_relaxed);
			g.capture_now = false;
			std::printf("[onda4] sesion %d acabada: %u frames, %llu copias, %.1f MB; saltadas: %llu por tope de memoria, %llu por "
			            "formato -> %s\n",
			            g.session, g.frame_index, static_cast<unsigned long long>(g.copies), g.bytes_total / 1048576.0,
			            static_cast<unsigned long long>(g.skipped_budget), static_cast<unsigned long long>(g.skipped_format),
			            g.dir.c_str());
			std::fflush(stdout);
			Loader::DemonsSoulsWarp::ShowNotice("Volcado onda " + std::to_string(g.session) + " hecho (" +
			                                        std::to_string(g.frame_index) + " frames). Se sigue escribiendo a disco unos segundos",
			                                    4);
			return false;
		}
		g.capture_now = rel % cfg.step == 0;
		if (g.capture_now) {
			++g.frame_index;
			g.seq = 0;
			g.pre_seen.clear();
			g.addr_writes.clear();
		}
	}
	return g.capture_now;
}

inline bool Listed(uint64_t hash) {
	if (Libs::Graphics::FrameGen::IsTaaShader(hash)) return true;
	const auto& h = Cfg().hashes;
	return std::find(h.begin(), h.end(), hash) != h.end();
}

inline bool ListedAddress(uint64_t address) {
	const auto& a = Cfg().addrs;
	return address != 0 && std::find(a.begin(), a.end(), address) != a.end();
}

// One image of a binding (its view's base level, layer `layer`) into a new staging buffer, recorded into `command`.
inline void CopyOne(Libs::Graphics::GraphicContext& graphics, Libs::Graphics::CommandScheduler& scheduler,
                    Libs::Graphics::TextureCache& cache, vk::CommandBuffer command, uint64_t frame, uint64_t cs_hash,
                    const char* when, uint32_t slot, const Libs::Graphics::TextureBinding& binding, uint32_t layer,
                    bool written) {
	using namespace Libs::Graphics;
	auto& g = G();
	if (!binding.image_id) return;
	auto& image = cache.GetImage(binding.image_id);
	if (image.backing.image == nullptr || image.backing.sparse != nullptr || image.backing.samples > 1 ||
	    image.backing.image_type != vk::ImageType::e2D || !(image.backing.usage & vk::ImageUsageFlagBits::eTransferSrc))
		return;
	vk::ImageAspectFlags aspect;
	const uint32_t       texel = TexelBytes(image.backing.format, aspect);
	if (texel == 0) {
		++g.skipped_format;
		return;
	}
	const uint32_t level = binding.desc.view_info.base_level;
	if (level >= image.backing.mip_levels || layer >= image.backing.layers) return;
	const uint32_t width  = std::max(1u, image.backing.extent.width >> level);
	const uint32_t height = std::max(1u, image.backing.extent.height >> level);
	const uint64_t bytes  = static_cast<uint64_t>(width) * height * texel;
	if (g.held.load(std::memory_order_relaxed) + bytes > Cfg().budget) {
		++g.skipped_budget;
		return;
	}
	g.allocator = graphics.allocator;
	VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	create.size  = bytes;
	create.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	Pending p;
	if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &p.buffer, &p.mem, nullptr) != VK_SUCCESS) {
		++g.skipped_budget;
		return;
	}
	const ImageSubresourceRange range {level, 1, layer, 1};
	// (the dispatch's or an earlier pass's writes before the copy; Transit adds the image's own barrier too)
	{
		vk::MemoryBarrier2 mb {};
		mb.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		mb.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
		mb.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		mb.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &mb;
		command.pipelineBarrier2(dependency);
	}
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, command);
	vk::BufferImageCopy copy {};
	copy.bufferOffset      = 0;
	copy.bufferRowLength   = 0;
	copy.bufferImageHeight = 0;
	copy.imageSubresource  = vk::ImageSubresourceLayers {aspect, level, layer, 1};
	copy.imageOffset       = vk::Offset3D {0, 0, 0};
	copy.imageExtent       = vk::Extent3D {width, height, 1};
	command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Buffer(p.buffer), 1, &copy);
	{
		vk::MemoryBarrier2 mb {};
		mb.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		mb.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		mb.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
		mb.dstAccessMask = vk::AccessFlagBits2::eHostRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &mb;
		command.pipelineBarrier2(dependency);
	}
	const vk::ImageLayout back = binding.layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eGeneral : binding.layout;
	image.Transit(back, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, range, command);

	char name[160];
	std::snprintf(name, sizeof(name), "f%03u_%03u_%016llx_%s_s%u_%llx_l%u.raw", g.frame_index, g.seq,
	              static_cast<unsigned long long>(cs_hash), when, slot,
	              static_cast<unsigned long long>(binding.desc.info.data.address), layer);
	char meta[640];
	std::snprintf(meta, sizeof(meta),
	              "{\"file\":\"%s\",\"frame\":%u,\"emu_frame\":%llu,\"seq\":%u,\"cs\":\"%016llx\",\"when\":\"%s\",\"slot\":%u,"
	              "\"addr\":\"%llx\",\"vkformat\":%d,\"w\":%u,\"h\":%u,\"level\":%u,\"layer\":%u,\"layers\":%u,\"texel\":%u,"
	              "\"aspect\":\"%s\",\"written\":%d,\"storage\":%d}\n",
	              name, g.frame_index, static_cast<unsigned long long>(frame), g.seq, static_cast<unsigned long long>(cs_hash),
	              when, slot, static_cast<unsigned long long>(binding.desc.info.data.address),
	              static_cast<int>(image.backing.format), width, height, level, layer, image.backing.layers, texel,
	              aspect == vk::ImageAspectFlagBits::eDepth ? "depth" : "color", written ? 1 : 0,
	              binding.desc.type == TextureCache::BindingType::Storage ? 1 : 0);
	++g.seq;
	p.bytes = bytes;
	p.tick  = scheduler.CurrentTick();
	p.frame = frame;
	p.file  = name;
	p.meta  = meta;
	p.dir   = g.dir;
	g.held.fetch_add(bytes, std::memory_order_relaxed);
	g.bytes_total += bytes;
	++g.copies;
	g.in_flight.push_back(std::move(p));
}

// Layers of a binding to copy: its view's, up to 4 (the G-buffer array: normal / roughness / material layers).
template <typename F>
inline void ForLayers(const Libs::Graphics::TextureBinding& binding, F&& f) {
	const uint32_t first = binding.desc.view_info.base_layer;
	const uint32_t count = std::clamp(binding.desc.view_info.layer_count, 1u, 4u);
	for (uint32_t i = 0; i < count; ++i) f(first + i);
}

// Before a listed shader's dispatch (after its bindings are resolved, before they are committed): every bound image,
// once per frame and (address, level, layer).
inline void BeforeDispatch(Libs::Graphics::GraphicContext& graphics, Libs::Graphics::CommandScheduler& scheduler,
                           Libs::Graphics::TextureCache& cache, vk::CommandBuffer command, uint64_t frame, uint64_t cs_hash,
                           std::span<const Libs::Graphics::TextureBinding> images, uint64_t written_mask) {
	if (!Listed(cs_hash)) return;
	auto& g = G();
	for (uint32_t i = 0; i < images.size() && i < 64; ++i) {
		const auto& b = images[i];
		if (!b.image_id || b.desc.info.data.address == 0) continue;
		ForLayers(b, [&](uint32_t layer) {
			const auto key = std::make_tuple(b.desc.info.data.address, b.desc.view_info.base_level, layer);
			if (std::find(g.pre_seen.begin(), g.pre_seen.end(), key) != g.pre_seen.end()) return;
			g.pre_seen.push_back(key);
			CopyOne(graphics, scheduler, cache, command, frame, cs_hash, "pre", i, b, layer, ((written_mask >> i) & 1u) != 0);
		});
	}
}

// After a dispatch: a listed shader's written images; any dispatch's written image at a listed address (up to 3 a
// frame per address).
inline void AfterDispatch(Libs::Graphics::GraphicContext& graphics, Libs::Graphics::CommandScheduler& scheduler,
                          Libs::Graphics::TextureCache& cache, vk::CommandBuffer command, uint64_t frame, uint64_t cs_hash,
                          std::span<const Libs::Graphics::TextureBinding> images, uint64_t written_mask) {
	auto&      g      = G();
	const bool listed = Listed(cs_hash);
	for (uint32_t i = 0; i < images.size() && i < 64; ++i) {
		if (((written_mask >> i) & 1u) == 0) continue;
		const auto& b = images[i];
		if (!b.image_id) continue;
		const uint64_t address = b.desc.info.data.address;
		if (!listed) {
			if (!ListedAddress(address)) continue;
			auto it = std::find_if(g.addr_writes.begin(), g.addr_writes.end(), [&](const auto& e) { return e.first == address; });
			if (it == g.addr_writes.end()) it = g.addr_writes.insert(g.addr_writes.end(), {address, 0u});
			if (it->second >= 3) continue;
			++it->second;
		}
		ForLayers(b, [&](uint32_t layer) {
			CopyOne(graphics, scheduler, cache, command, frame, cs_hash, "post", i, b, layer, true);
		});
	}
}

} // namespace OndaDump
