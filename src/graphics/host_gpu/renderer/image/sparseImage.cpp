#include "graphics/host_gpu/renderer/image/sparseImage.h"

#include "common/assert.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "kernel/memory.h"
#include "async-upload.h"
#include "device-fault.h"
#include "ram-native-tex.h"
#include "vulkan-recording.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>

// (ram1b) KYTY_RAM_NATIVE_TEX=1|2 (local/ram-native-tex.h). Ghidra 01.007.000: the game commits a texture array's layer
// slot whole (0x901030: faces x slice at base + layer x faces x slice) and decommits it whole (0x901110, from the
// deferred free list 0xac4e00); its GPU samples only the layers it committed. The PC image of such an array was dense:
// 972 MiB of guest range over 12 images with 414-511 MiB committed (11-17/11-18 logs, "Offload differs"). Here the image
// is sparse (VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT): each layer gets video memory - its levels above the mip tail through
// image binds, its own mip tail through an opaque bind (one tail for all with SINGLE_MIPTAIL) - while its guest slice is
// committed. Layers' memory comes from chunks of a few slots (64 MiB at most a chunk), freed slots reused; the chunks
// are freed with the image (the GPU done with it: GraphicContext::DeleteImage). Bindings are queued on the graphics
// queue (as Buffer::MakeResident does) with one timeline semaphore for all, which the scheduler's next submission waits
// for: what is recorded after a bind sees it. An unbind lands before the work recorded but not yet submitted: the game
// releases a slot only through its deferred free list (after its GPU is done with it), and an unbound layer reads zero or
// undefined, never faults (sparse residency). A layer left without memory (none to be had) is counted and logged.

namespace Libs::Graphics {

struct SparseImageMemory {
	uint32_t              layers = 0, levels = 0;
	uint32_t              tail_first_lod = 0; // levels >= it are in the mip tail
	bool                  single_tail    = false;
	bool                  tail_bound     = false; // (single tail: bound once, kept)
	uint64_t              tail_offset = 0, tail_stride = 0, tail_size = 0;
	VkExtent3D            granularity {1, 1, 1};
	VkExtent3D            extent {1, 1, 1};
	uint64_t              block       = 0; // sparse block size (memory alignment)
	uint64_t              layer_bytes = 0; // memory one layer takes
	std::vector<uint64_t> level_bytes;     // memory of each level above the tail, in one layer
	uint32_t              memory_bits = 0;
	uint32_t              memory_type = 0;
	uint32_t              slots_per_chunk = 1;
	std::vector<VkDeviceMemory> chunks;
	VkDeviceMemory        tail_memory = VK_NULL_HANDLE; // (single tail)
	std::vector<uint32_t> free_slots;
	std::vector<int64_t>  slot_of_layer; // -1: unbound
	uint64_t              bound_bytes = 0, dense_bytes = 0, chunk_bytes = 0;
};

namespace {

thread_local int t_force_dense = 0;

// One timeline semaphore for every sparse image's bindings (never destroyed: the scheduler may hold it as its next wait).
VkSemaphore  g_timeline = VK_NULL_HANDLE;
uint64_t     g_timeline_value = 0;
std::mutex   g_timeline_mutex;

VkSemaphore Timeline(VkDevice device) {
	if (g_timeline != VK_NULL_HANDLE) return g_timeline;
	VkSemaphoreTypeCreateInfo timeline {};
	timeline.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	VkSemaphoreCreateInfo semaphore {};
	semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphore.pNext = &timeline;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateSemaphore(device, &semaphore, nullptr, &g_timeline) != VK_SUCCESS) g_timeline = VK_NULL_HANDLE;
	return g_timeline;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
	return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

void Refuse(const char* why, const vk::ImageCreateInfo& create) {
	RamNativeTex::Sparse().refused.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 24) {
		std::printf("[rnt] array denso (no disperso): %s - %ux%u capas=%u niveles=%u formato=%d\n", why, create.extent.width,
		            create.extent.height, create.arrayLayers, create.mipLevels, static_cast<int>(create.format));
		std::fflush(stdout);
	}
}

} // namespace

ImageForceDense::ImageForceDense() {
	t_force_dense++;
}
ImageForceDense::~ImageForceDense() {
	t_force_dense--;
}

bool SparseImageWanted(const GraphicContext& graphics, const ImageInfo& info, const vk::ImageCreateInfo& create) {
	if (!RamNativeTex::On() || t_force_dense != 0 || info.data.Empty()) return false;
	if (!RamNativeTex::InZone(info.data.address, info.data.End())) return false;
	if (info.type != Prospero::ImageType::kColor2D || info.IsVolume() || info.resources.layers < 8 || info.samples != 1 ||
	    info.HasStencil() || info.IsDepth() || info.tile_mode == Prospero::TileMode::kDepth || info.data.size % info.resources.layers != 0)
		return false;
	if (create.imageType != vk::ImageType::e2D || create.arrayLayers != info.resources.layers ||
	    static_cast<bool>(create.flags & vk::ImageCreateFlagBits::eCubeCompatible) ||
	    static_cast<bool>(create.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment))
		return false;
	// (Thread: the bindings wait on the GPU thread's scheduler; a speculation's recorder, the upload worker and the
	// deferred operations make dense images.)
	if (!GuestGpu::IsGpuThread() || CommandScheduler::ThreadRecorder() != nullptr || CommandScheduler::InDeferredOperation() ||
	    AsyncUpload::t_worker) {
		Refuse("hilo que no es el de la GPU", create);
		return false;
	}
	if (!graphics.sparse_residency_images) {
		static std::atomic<bool> once {false};
		if (!once.exchange(true)) Refuse("el dispositivo no tiene sparseResidencyImage2D en la cola grafica", create);
		return false;
	}
	return true;
}

bool CreateSparseImage(GraphicContext& graphics, const vk::ImageCreateInfo& create_in, VulkanImage& image) {
	EXIT_IF(image.image != nullptr || image.sparse != nullptr);
	const auto native_device   = static_cast<VkDevice>(graphics.device);
	const auto native_physical = static_cast<VkPhysicalDevice>(graphics.physical_device);
	auto       create          = create_in;
	create.flags |= vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency;
	// The format's sparse support (color aspect, standard or not).
	{
		uint32_t count = 0;
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceSparseImageFormatProperties(
		    native_physical, static_cast<VkFormat>(create.format), VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT,
		    static_cast<VkImageUsageFlags>(create.usage), VK_IMAGE_TILING_OPTIMAL, &count, nullptr);
		if (count == 0) {
			Refuse("formato sin soporte disperso", create);
			return false;
		}
		vk::ImageFormatProperties properties {};
		if (graphics.physical_device.getImageFormatProperties(create.format, create.imageType, create.tiling, create.usage,
		                                                      create.flags, &properties) != vk::Result::eSuccess ||
		    create.arrayLayers > properties.maxArrayLayers) {
			Refuse("formato/uso sin imagen dispersa", create);
			return false;
		}
	}
	VkImage    native_image = VK_NULL_HANDLE;
	const auto raw          = static_cast<VkImageCreateInfo>(create);
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage(native_device, &raw, nullptr, &native_image) != VK_SUCCESS) {
		Refuse("vkCreateImage disperso rechazado", create);
		return false;
	}
	VkMemoryRequirements requirements {};
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements(native_device, native_image, &requirements);
	uint32_t count = 0;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageSparseMemoryRequirements(native_device, native_image, &count, nullptr);
	std::vector<VkSparseImageMemoryRequirements> sparse_requirements(count);
	if (count != 0)
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageSparseMemoryRequirements(native_device, native_image, &count, sparse_requirements.data());
	const VkSparseImageMemoryRequirements* color = nullptr;
	bool                                   other = false;
	for (const auto& item: sparse_requirements) {
		if (item.formatProperties.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT) color = &item;
		else other = true; // (metadata or another aspect: not handled here)
	}
	auto memory = std::make_unique<SparseImageMemory>();
	auto& s     = *memory;
	s.layers      = create.arrayLayers;
	s.levels      = create.mipLevels;
	s.extent      = static_cast<VkExtent3D>(create.extent);
	s.block       = requirements.alignment;
	s.memory_bits = requirements.memoryTypeBits;
	s.dense_bytes = requirements.size;
	if (color != nullptr) {
		s.granularity    = color->formatProperties.imageGranularity;
		s.single_tail    = (color->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0;
		s.tail_first_lod = std::min(color->imageMipTailFirstLod, s.levels);
		s.tail_offset    = color->imageMipTailOffset;
		s.tail_stride    = color->imageMipTailStride;
		s.tail_size      = AlignUp(color->imageMipTailSize, s.block);
	}
	if (color == nullptr || other || s.block == 0 || s.granularity.width == 0 || s.granularity.height == 0 ||
	    s.granularity.depth == 0 || (s.tail_first_lod < s.levels && s.tail_size == 0)) {
		VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(native_device, native_image, nullptr);
		Refuse(other ? "requisitos de metadatos u otro aspecto" : "requisitos dispersos incompletos", create);
		return false;
	}
	for (uint32_t level = 0; level < s.tail_first_lod; level++) {
		const uint64_t w = std::max(s.extent.width >> level, 1u), h = std::max(s.extent.height >> level, 1u);
		const uint64_t blocks = ((w + s.granularity.width - 1) / s.granularity.width) * ((h + s.granularity.height - 1) / s.granularity.height);
		s.level_bytes.push_back(blocks * s.block);
		s.layer_bytes += blocks * s.block;
	}
	if (!s.single_tail && s.tail_first_lod < s.levels) s.layer_bytes += s.tail_size;
	// (A layer smaller than a few blocks gains nothing: dense. Nor one whose sparse layers would take more than the dense
	// image's share.)
	if (s.layer_bytes == 0 || s.layer_bytes > std::max<uint64_t>(requirements.size / s.layers, 1) * 2) {
		VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(native_device, native_image, nullptr);
		Refuse("capa dispersa mayor que el doble de su parte densa", create);
		return false;
	}
	// Memory type: device-local only first.
	const auto& types = graphics.physical_device_memory_properties;
	s.memory_type     = UINT32_MAX;
	for (uint32_t i = 0; i < types.memoryTypeCount && s.memory_type == UINT32_MAX; ++i)
		if ((s.memory_bits & (1u << i)) != 0 &&
		    static_cast<VkMemoryPropertyFlags>(types.memoryTypes[i].propertyFlags) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			s.memory_type = i;
	for (uint32_t i = 0; i < types.memoryTypeCount && s.memory_type == UINT32_MAX; ++i)
		if ((s.memory_bits & (1u << i)) != 0 &&
		    (static_cast<VkMemoryPropertyFlags>(types.memoryTypes[i].propertyFlags) & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
			s.memory_type = i;
	if (s.memory_type == UINT32_MAX || Timeline(native_device) == VK_NULL_HANDLE) {
		VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(native_device, native_image, nullptr);
		Refuse("sin tipo de memoria de video o semaforo", create);
		return false;
	}
	s.slots_per_chunk = static_cast<uint32_t>(std::clamp<uint64_t>((uint64_t {64} << 20u) / s.layer_bytes, 1, s.layers));
	s.slot_of_layer.assign(s.layers, -1);

	image.image      = vk::Image(native_image);
	image.allocation = nullptr;
	image.format     = create_in.format;
	image.image_type = create_in.imageType;
	image.extent     = create_in.extent;
	image.layers     = create_in.arrayLayers;
	image.mip_levels = create_in.mipLevels;
	image.samples    = static_cast<uint32_t>(create_in.samples);
	image.usage      = create_in.usage;
	image.flags      = create.flags;
	image.state      = {.layout = create_in.initialLayout};
	image.subresource_states.clear();
	image.sparse = memory.release();
	auto& counters = RamNativeTex::Sparse();
	counters.images.fetch_add(1, std::memory_order_relaxed);
	counters.made.fetch_add(1, std::memory_order_relaxed);
	counters.dense_bytes.fetch_add(s.dense_bytes, std::memory_order_relaxed);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 24) {
		std::printf("[rnt] array disperso: %ux%u capas=%u niveles=%u formato=%d: denso %.1f MiB, %.2f MiB por capa (bloque %llu KiB, "
		            "granularidad %ux%u, cola de mips desde %u%s)\n",
		            s.extent.width, s.extent.height, s.layers, s.levels, static_cast<int>(create.format),
		            static_cast<double>(s.dense_bytes) / 1048576.0, static_cast<double>(s.layer_bytes) / 1048576.0,
		            static_cast<unsigned long long>(s.block >> 10u), s.granularity.width, s.granularity.height, s.tail_first_lod,
		            s.single_tail ? ", una para todas" : "");
		std::fflush(stdout);
	}
	return true;
}

void DestroySparseImage(GraphicContext& graphics, VulkanImage& image) {
	if (image.sparse == nullptr) return;
	const auto native_device = static_cast<VkDevice>(graphics.device);
	auto*      s             = image.sparse;
	if (image.image != nullptr) VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(native_device, static_cast<VkImage>(image.image), nullptr);
	for (auto chunk: s->chunks) VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(native_device, chunk, nullptr);
	if (s->tail_memory != VK_NULL_HANDLE) VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(native_device, s->tail_memory, nullptr);
	auto& counters = RamNativeTex::Sparse();
	counters.images.fetch_sub(1, std::memory_order_relaxed);
	counters.dense_bytes.fetch_sub(s->dense_bytes, std::memory_order_relaxed);
	counters.bound_bytes.fetch_sub(s->bound_bytes, std::memory_order_relaxed);
	counters.chunk_bytes.fetch_sub(s->chunk_bytes, std::memory_order_relaxed);
	delete s;
	image.sparse     = nullptr;
	image.image      = nullptr;
	image.allocation = nullptr;
}

namespace {

// A memory object of `bytes` in the image's type, or another type it allows (null: none could be had).
VkDeviceMemory AllocateSparse(GraphicContext& graphics, SparseImageMemory& s, uint64_t bytes) {
	const auto           native_device = static_cast<VkDevice>(graphics.device);
	VkMemoryAllocateInfo allocate {};
	allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.allocationSize  = bytes;
	allocate.memoryTypeIndex = s.memory_type;
	VkDeviceMemory memory    = VK_NULL_HANDLE;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(native_device, &allocate, nullptr, &memory) == VK_SUCCESS) return memory;
	const auto& types = graphics.physical_device_memory_properties;
	for (uint32_t i = 0; i < types.memoryTypeCount; ++i) {
		if ((s.memory_bits & (1u << i)) == 0 || i == s.memory_type) continue;
		allocate.memoryTypeIndex = i;
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(native_device, &allocate, nullptr, &memory) == VK_SUCCESS) return memory;
	}
	return VK_NULL_HANDLE;
}

} // namespace

void SparseUpdateLayers(GraphicContext& graphics, CommandScheduler& scheduler, VulkanImage& image, uint64_t address,
                        uint64_t stride, uint64_t begin, uint64_t end, uint64_t gone_begin, uint64_t gone_end) {
	if (image.sparse == nullptr || image.image == nullptr || stride == 0 || end <= begin) return;
	auto&          s     = *image.sparse;
	const uint64_t limit = address + stride * s.layers;
	begin                = std::max(begin, address);
	end                  = std::min(end, limit);
	if (end <= begin) return;
	const auto start = std::chrono::steady_clock::now();
	const auto first = static_cast<uint32_t>((begin - address) / stride);
	const auto last  = static_cast<uint32_t>((end - 1 - address) / stride);
	thread_local std::vector<std::pair<uint64_t, uint64_t>> parts;
	std::vector<uint32_t>                                    bind, unbind;
	for (uint32_t layer = first; layer <= last && layer < s.layers; layer++) {
		const uint64_t slice = address + stride * layer;
		Libs::LibKernel::Memory::MappedParts(slice, stride, &parts);
		bool committed = false;
		for (const auto& [at, bytes]: parts) {
			const uint64_t lo = std::max(at, slice), hi = std::min(at + bytes, slice + stride);
			if (lo >= hi) continue;
			// (What is about to be unmapped does not count.)
			if (gone_end <= gone_begin || lo < gone_begin || hi > gone_end) {
				committed = true;
				break;
			}
		}
		if (committed && s.slot_of_layer[layer] < 0) bind.push_back(layer);
		if (!committed && s.slot_of_layer[layer] >= 0) unbind.push_back(layer);
	}
	if (bind.empty() && unbind.empty()) return;
	std::vector<VkSparseImageMemoryBind> image_binds;
	std::vector<VkSparseMemoryBind>      opaque_binds;
	auto&                                counters = RamNativeTex::Sparse();
	const auto add_layer = [&](uint32_t layer, VkDeviceMemory memory, uint64_t base) {
		uint64_t offset = base;
		for (uint32_t level = 0; level < s.tail_first_lod; level++) {
			VkSparseImageMemoryBind bind_item {};
			bind_item.subresource  = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer};
			bind_item.offset       = {0, 0, 0};
			bind_item.extent       = {std::max(s.extent.width >> level, 1u), std::max(s.extent.height >> level, 1u), 1u};
			bind_item.memory       = memory;
			bind_item.memoryOffset = memory != VK_NULL_HANDLE ? offset : 0;
			image_binds.push_back(bind_item);
			offset += s.level_bytes[level];
		}
		if (!s.single_tail && s.tail_first_lod < s.levels) {
			VkSparseMemoryBind tail {};
			tail.resourceOffset = s.tail_offset + static_cast<uint64_t>(layer) * s.tail_stride;
			tail.size           = s.tail_size;
			tail.memory         = memory;
			tail.memoryOffset   = memory != VK_NULL_HANDLE ? offset : 0;
			opaque_binds.push_back(tail);
		}
	};
	for (const auto layer: unbind) {
		add_layer(layer, VK_NULL_HANDLE, 0);
		s.free_slots.push_back(static_cast<uint32_t>(s.slot_of_layer[layer]));
		s.slot_of_layer[layer] = -1;
		s.bound_bytes -= std::min(s.bound_bytes, s.layer_bytes);
		counters.bound_bytes.fetch_sub(s.layer_bytes, std::memory_order_relaxed);
		counters.layers_unbound.fetch_add(1, std::memory_order_relaxed);
	}
	// (Single tail: bound once, with the first layer.)
	if (!bind.empty() && s.single_tail && s.tail_first_lod < s.levels && !s.tail_bound) {
		s.tail_memory = AllocateSparse(graphics, s, s.tail_size);
		if (s.tail_memory != VK_NULL_HANDLE) {
			opaque_binds.push_back({s.tail_offset, s.tail_size, s.tail_memory, 0, 0});
			s.tail_bound = true;
			s.chunk_bytes += s.tail_size;
			counters.chunk_bytes.fetch_add(s.tail_size, std::memory_order_relaxed);
		}
	}
	for (const auto layer: bind) {
		if (s.free_slots.empty()) {
			const uint64_t bytes = s.layer_bytes * s.slots_per_chunk;
			const auto     chunk = AllocateSparse(graphics, s, bytes);
			if (chunk == VK_NULL_HANDLE) {
				counters.alloc_failed.fetch_add(1, std::memory_order_relaxed);
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::printf("[rnt] FALLO: sin memoria de video para la capa %u de un array disperso (%.1f MiB): queda sin ligar\n",
					            layer, static_cast<double>(bytes) / 1048576.0);
					std::fflush(stdout);
				}
				continue;
			}
			const auto index = static_cast<uint32_t>(s.chunks.size());
			s.chunks.push_back(chunk);
			for (uint32_t k = s.slots_per_chunk; k-- > 0;) s.free_slots.push_back(index * s.slots_per_chunk + k);
			s.chunk_bytes += bytes;
			counters.chunk_bytes.fetch_add(bytes, std::memory_order_relaxed);
		}
		const uint32_t slot = s.free_slots.back();
		s.free_slots.pop_back();
		s.slot_of_layer[layer] = slot;
		add_layer(layer, s.chunks[slot / s.slots_per_chunk], static_cast<uint64_t>(slot % s.slots_per_chunk) * s.layer_bytes);
		s.bound_bytes += s.layer_bytes;
		counters.bound_bytes.fetch_add(s.layer_bytes, std::memory_order_relaxed);
		counters.layers_bound.fetch_add(1, std::memory_order_relaxed);
	}
	if (image_binds.empty() && opaque_binds.empty()) return;
	VkSparseImageMemoryBindInfo       image_info {static_cast<VkImage>(image.image), static_cast<uint32_t>(image_binds.size()),
	                                              image_binds.data()};
	VkSparseImageOpaqueMemoryBindInfo opaque_info {static_cast<VkImage>(image.image), static_cast<uint32_t>(opaque_binds.size()),
	                                               opaque_binds.data()};
	const auto                        native_device = static_cast<VkDevice>(graphics.device);
	const VkSemaphore                 semaphore     = Timeline(native_device);
	uint64_t                          signal        = 0;
	VkResult                          result        = VK_SUCCESS;
	{
		std::lock_guard timeline_lock(g_timeline_mutex);
		signal = ++g_timeline_value;
		VkTimelineSemaphoreSubmitInfo timeline {};
		timeline.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
		timeline.signalSemaphoreValueCount = 1;
		timeline.pSignalSemaphoreValues    = &signal;
		VkBindSparseInfo info {};
		info.sType                = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
		info.pNext                = &timeline;
		info.imageBindCount       = image_binds.empty() ? 0u : 1u;
		info.pImageBinds          = image_binds.empty() ? nullptr : &image_info;
		info.imageOpaqueBindCount = opaque_binds.empty() ? 0u : 1u;
		info.pImageOpaqueBinds    = opaque_binds.empty() ? nullptr : &opaque_info;
		info.signalSemaphoreCount = 1;
		info.pSignalSemaphores    = &semaphore;
		// (The graphics queue, under its lock, outside the recording layer: as Buffer::MakeResident.)
		Common::LockGuard lock(graphics.queue_mutex);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		const auto& dispatch = LocalVulkanRecording::DirectDispatch();
#else
		const auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
#endif
		result = dispatch.vkQueueBindSparse(static_cast<VkQueue>(graphics.queue), 1, &info, VK_NULL_HANDLE);
	}
	if (result == VK_ERROR_DEVICE_LOST) DeviceFault::Report();
	EXIT_NOT_IMPLEMENTED(result != VK_SUCCESS);
	// What is recorded from here on runs after the binding (binding operations keep no order with command buffers but
	// through semaphores).
	scheduler.WaitBeforeNextSubmit(vk::Semaphore(semaphore), signal);
	counters.bind_calls.fetch_add(1, std::memory_order_relaxed);
	counters.bind_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                                     std::chrono::steady_clock::now() - start)
	                                                     .count()),
	                           std::memory_order_relaxed);
}

void SparseBoundLayers(const VulkanImage& image, std::vector<uint32_t>* layers) {
	layers->clear();
	if (image.sparse == nullptr) return;
	for (uint32_t layer = 0; layer < image.sparse->layers; layer++)
		if (image.sparse->slot_of_layer[layer] >= 0) layers->push_back(layer);
}

} // namespace Libs::Graphics
