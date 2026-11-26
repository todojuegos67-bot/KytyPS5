#include "graphics/host_gpu/renderer/image/sparseImage.h"

#include "common/assert.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "kernel/memory.h"
#include "async-upload.h"
#include "device-fault.h"
#include "ram-native-tex.h"
#include "tex-nativa.h" // (N1, KYTY_TEX_NATIVA)
#include "vulkan-recording.h"
#include "local/vram-drv.h" // (VRAM-DRV)
#include "local/vram-tope.h" // (W16) solo lectura: uso y tope efectivo
#include "local/gestor-memoria.h" // (M7) registro de las texturas nativas en el gestor central

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <memory>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
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

struct SparsePoolBlock; // (SPARSE-POOL) bloque grande del que salen los mips (null: memoria propia)

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
	// (N1, KYTY_TEX_NATIVA) Una textura (1 capa) con toda su cadena de mips: memoria por mip (los mips sobre la cola),
	// ligada mientras el juego tiene comprometido el rango de ese mip; la cola de mips, ligada una vez.
	bool                        per_level = false;
	std::vector<VkDeviceMemory> level_memory;
	std::vector<uint32_t>       level_type; // (SPARSE-HILO) tipo de memoria de cada mip (para devolverla al pool)
	uint32_t                    tail_type = 0; // (SPARSE-HILO)
	std::vector<SparsePoolBlock*> level_block;      // (SPARSE-POOL) bloque de cada mip (null: memoria propia)
	std::vector<uint64_t>         level_offset;     // (SPARSE-POOL) desplazamiento en el bloque
	SparsePoolBlock*              tail_block = nullptr; // (SPARSE-POOL)
	uint64_t                      tail_mem_offset = 0;  // (SPARSE-POOL)
	struct Retired {
		VkDeviceMemory   memory;
		uint64_t         bytes;
		uint64_t         tick;
		uint32_t         type; // (SPARSE-HILO)
		SparsePoolBlock* block  = nullptr; // (SPARSE-POOL)
		uint64_t         offset = 0;
	};
	std::vector<Retired> retired; // desligada; se libera cuando la GPU acabo con lo grabado antes (scheduler.IsFree)
	// (VRAM-PRES) El scheduler de lo retirado y si la imagen esta en g_live (el recolector la recorre para devolver lo
	// retirado aunque la imagen no vuelva a pasar por SparseUpdateLevels).
	CommandScheduler* retired_scheduler = nullptr;
	bool              in_live           = false;
	// (P5) Ligados aun no enviados (bajo g_pending_mutex): van en el siguiente SparseFlushPendingBinds de pending_scheduler.
	VkImage                              vk_image          = VK_NULL_HANDLE;
	CommandScheduler*                    pending_scheduler = nullptr;
	std::vector<VkSparseImageMemoryBind> pending_image;
	std::vector<VkSparseMemoryBind>      pending_opaque;
	// (W14, SPARSE-COMPACTAR) Mips en traslado a otro bloque (copia GPU hecha, ligado nuevo aun no) y si se cancelan
	// (la imagen se uso o cambio su ligado mientras: el traslado se descarta y se queda donde estaba).
	std::atomic<uint32_t> moving {0};
	std::atomic<bool>     move_cancel {false};
	// (W16, TEX-NATIVA-RAM) Ultimo uso (ms) y usos (TouchImage -> SparseMoverCancelar): orden de lo que va a la RAM.
	std::atomic<uint64_t> ultimo_uso_ms {0};
	std::atomic<uint32_t> usos {0};
	// (W17) Bytes de esta imagen (mips y cola) en bloques de RAM: SparseBoundBytes y c_bound_bytes no los cuentan como VRAM.
	uint64_t ram_bytes = 0;
	uint64_t gestor_id = 0, gestor_bytes = 0; // (M7) GestorMem: id y ultimo tamano comunicado (VRAM ligada)
};

namespace {

thread_local int t_force_dense = 0;

// One timeline semaphore for every sparse image's bindings (never destroyed: the scheduler may hold it as its next wait).
VkSemaphore  g_timeline = VK_NULL_HANDLE;
uint64_t     g_timeline_value = 0;
std::mutex   g_timeline_mutex;

// (P5) Imagenes per_level con ligados pendientes (sin repetir).
std::mutex                      g_pending_mutex;
std::vector<SparseImageMemory*> g_pending;

// (VRAM-PRES) Imagenes nativas con memoria retirada pendiente (bajo g_live_mutex, que tambien guarda su `retired`),
// bytes retirados aun sin devolver, presion de presupuesto (el hilo no rellena el pool y lo vacia) y lo devuelto.
std::mutex                      g_live_mutex;
std::vector<SparseImageMemory*> g_live;
// (W17) Todas las imagenes nativas (per_level): para saber que mips hay en cada bloque y vaciarlo entero.
std::mutex                      g_todas_mutex;
std::vector<SparseImageMemory*> g_todas;
std::atomic<uint64_t>           g_retired_bytes {0};
std::atomic<bool>               g_pres_pressure {false};

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

void Refuse(const char* why, const vk::ImageCreateInfo& create, int reason) {
	if (create.arrayLayers == 1) { // (N1) textura nativa rechazada: densa como siempre
		TexNativa::c_refused.fetch_add(1, std::memory_order_relaxed);
		TexNativa::c_rechazo[reason].fetch_add(1, std::memory_order_relaxed); // (P2)
		return;
	}
	RamNativeTex::Sparse().refused.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 24) {
		std::printf("[rnt] array denso (no disperso): %s - %ux%u capas=%u niveles=%u formato=%d\n", why, create.extent.width,
		            create.extent.height, create.arrayLayers, create.mipLevels, static_cast<int>(create.format));
		std::fflush(stdout);
	}
}

// (W14) Definida mas abajo: los traslados en curso de una imagen que se destruye quedan sin imagen.
void CompactarSoltarImagen(SparseImageMemory* s);
// (SPARSE-HILO) Definidas mas abajo.
bool HiloOn();
void GiveMemory(VkDevice device, VkDeviceMemory memory, uint64_t bytes, uint32_t type);
void GiveRange(VkDevice device, VkDeviceMemory memory, SparsePoolBlock* block, uint64_t offset, uint64_t bytes, uint32_t type);
extern std::mutex g_inflight_mutex;

} // namespace

ImageForceDense::ImageForceDense() {
	t_force_dense++;
}
ImageForceDense::~ImageForceDense() {
	t_force_dense--;
}

// (N1) Una textura de una CTexture (TexNativa::Register, salida de 0x8fe750) con su cadena de mips completa.
static bool NativeTextureWanted(const GraphicContext& graphics, const ImageInfo& info, const vk::ImageCreateInfo& create) {
	if (!TexNativa::On() || t_force_dense != 0 || info.data.Empty() || !TexNativa::IsNative(info.data.address)) return false;
	// (P2) Cada motivo, contado (fuera: no se intenta y no cuenta en c_refused). Un solo mip tambien entra (UnMip): la
	// misma imagen dispersa por mip con un nivel (o solo la cola), mismo camino de trozos directos.
	const auto fuera = [](int reason) {
		TexNativa::c_rechazo[reason].fetch_add(1, std::memory_order_relaxed);
		return false;
	};
	if (info.resources.layers != 1 || create.arrayLayers != 1) return fuera(TexNativa::kRCapas);
	if (info.resources.levels < (TexNativa::UnMip() ? 1u : 2u) || info.resources.levels > info.mip_layout.size() ||
	    create.mipLevels != info.resources.levels)
		return fuera(TexNativa::kRMips);
	if (static_cast<bool>(create.usage & (vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eColorAttachment)))
		return fuera(TexNativa::kRAdjunto);
	if (info.type != Prospero::ImageType::kColor2D || info.IsVolume() || info.samples != 1 || info.HasStencil() || info.IsDepth() ||
	    info.tile_mode == Prospero::TileMode::kDepth || create.imageType != vk::ImageType::e2D ||
	    static_cast<bool>(create.flags & vk::ImageCreateFlagBits::eCubeCompatible))
		return fuera(TexNativa::kRForma);
	if (!GuestGpu::IsGpuThread() || CommandScheduler::ThreadRecorder() != nullptr || CommandScheduler::InDeferredOperation() ||
	    AsyncUpload::t_worker || !graphics.sparse_residency_images) {
		TexNativa::c_refused.fetch_add(1, std::memory_order_relaxed);
		TexNativa::c_rechazo[graphics.sparse_residency_images ? TexNativa::kRHilo : TexNativa::kRDisp].fetch_add(
		    1, std::memory_order_relaxed);
		return false;
	}
	if (info.resources.levels == 1) TexNativa::c_un_mip.fetch_add(1, std::memory_order_relaxed); // (P2) intentadas
	return true;
}

bool SparseImageWanted(const GraphicContext& graphics, const ImageInfo& info, const vk::ImageCreateInfo& create) {
	if (NativeTextureWanted(graphics, info, create)) return true;
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
		Refuse("hilo que no es el de la GPU", create, TexNativa::kRHilo);
		return false;
	}
	if (!graphics.sparse_residency_images) {
		static std::atomic<bool> once {false};
		if (!once.exchange(true)) Refuse("el dispositivo no tiene sparseResidencyImage2D en la cola grafica", create, TexNativa::kRDisp);
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
			Refuse("formato sin soporte disperso", create, TexNativa::kRFormato);
			return false;
		}
		vk::ImageFormatProperties properties {};
		if (graphics.physical_device.getImageFormatProperties(create.format, create.imageType, create.tiling, create.usage,
		                                                      create.flags, &properties) != vk::Result::eSuccess ||
		    create.arrayLayers > properties.maxArrayLayers) {
			Refuse("formato/uso sin imagen dispersa", create, TexNativa::kRFormatoUso);
			return false;
		}
	}
	VkImage    native_image = VK_NULL_HANDLE;
	const auto raw          = static_cast<VkImageCreateInfo>(create);
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage(native_device, &raw, nullptr, &native_image) != VK_SUCCESS) {
		Refuse("vkCreateImage disperso rechazado", create, TexNativa::kRCrear);
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
		Refuse(other ? "requisitos de metadatos u otro aspecto" : "requisitos dispersos incompletos", create, TexNativa::kRReqs);
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
	// (TEX-DOBLE) El log del 24/11 da 249 rechazos "doble" (texturas pequenas: los bloques de 64 KB de la imagen
	// dispersa ocupan mas del doble que la densa). Ahora tambien son nativas: gastan algo mas de VRAM, pero sus bytes
	// salen de la RAM. KYTY_TEX_DOBLE=0 vuelve al rechazo de antes.
	static const bool doble_nativa = [] {
		const char* v = std::getenv("KYTY_TEX_DOBLE");
		return v == nullptr || v[0] != '0';
	}();
	// (TEX-COLA) Las 388 "doble" del log 082750 son texturas enteras en la cola de mips (NVIDIA: SINGLE_MIPTAIL, cola
	// desde el mip 0): layer_bytes 0. Para una textura nativa (1 capa, memoria por mip) la cola se liga una vez en
	// SparseUpdateLevels; no hay mips sueltos. Nativa tambien. KYTY_TEX_COLA=0 vuelve al rechazo.
	static const bool cola_nativa = [] {
		const char* v = std::getenv("KYTY_TEX_COLA");
		return v == nullptr || v[0] != '0';
	}();
	const bool solo_cola = cola_nativa && s.layers == 1 && s.layer_bytes == 0 && s.tail_first_lod == 0 && s.tail_size != 0;
	if ((s.layer_bytes == 0 && !solo_cola) || (!doble_nativa && s.layer_bytes > std::max<uint64_t>(requirements.size / s.layers, 1) * 2)) {
		VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(native_device, native_image, nullptr);
		Refuse("capa dispersa mayor que el doble de su parte densa", create, TexNativa::kRDoble);
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
		Refuse("sin tipo de memoria de video o semaforo", create, TexNativa::kRMemoria);
		return false;
	}
	// (VRAM-AJUSTE) Trozos de 16 MiB (antes 64): log 20261125-104537 "texturas nativas 803 en 16" = 9 bloques del pool (576)
	// + 7 trozos de arrays dispersos (227 MiB) con capas sin ligar dentro; un trozo mas pequeno reserva solo lo que se liga.
	s.slots_per_chunk = static_cast<uint32_t>(std::clamp<uint64_t>((uint64_t {16} << 20u) / std::max<uint64_t>(s.layer_bytes, 1), 1, s.layers));
	s.slot_of_layer.assign(s.layers, -1);
	if (s.layers == 1) { // (N1) textura nativa: memoria por mip
		s.per_level = true;
		s.level_memory.assign(s.tail_first_lod, VK_NULL_HANDLE);
		s.level_type.assign(s.tail_first_lod, 0); // (SPARSE-HILO)
		s.level_block.assign(s.tail_first_lod, nullptr); // (SPARSE-POOL)
		s.level_offset.assign(s.tail_first_lod, 0);
		TexNativa::c_images.fetch_add(1, std::memory_order_relaxed);
		std::lock_guard todas_lock(g_todas_mutex); // (W17)
		g_todas.push_back(&s);
	}

	image.image      = vk::Image(native_image);
	s.vk_image       = native_image; // (P5)
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
	// (M7) Al gestor central: textura nativa / array disperso, VRAM = lo ligado (0 al crearse; CambiarTamano al ligar/soltar).
	memory->gestor_id = GestorMem::Registrar(GestorMem::TipoNativa, GestorMem::Vram, 0, 0, memory.get());
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
	// (SPARSE-HILO) Si el hilo esta enviando ligados (quiza de esta imagen), se espera a que acabe el envio.
	std::unique_lock<std::mutex> inflight_lock;
	if (s->per_level && HiloOn()) inflight_lock = std::unique_lock<std::mutex>(g_inflight_mutex);
	if (s->per_level) { // (P5) sus ligados pendientes no se envian: la imagen desaparece
		std::lock_guard pending_lock(g_pending_mutex);
		std::erase(g_pending, s);
		s->pending_scheduler = nullptr;
		s->pending_image.clear();
		s->pending_opaque.clear();
	}
	if (s->per_level) { // (W17) fuera del registro antes de soltar sus traslados (la pasada de vaciado lo tiene bloqueado)
		std::lock_guard todas_lock(g_todas_mutex);
		std::erase(g_todas, s);
	}
	if (s->per_level && s->moving.load(std::memory_order_relaxed) != 0) CompactarSoltarImagen(s); // (W14)
	if (image.image != nullptr) VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(native_device, static_cast<VkImage>(image.image), nullptr);
	for (auto chunk: s->chunks) VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(native_device, chunk, nullptr);
	// (SPARSE-HILO) La memoria por mip, la retirada y la cola: al pool (la GPU acabo con la imagen).
	for (size_t level = 0; level < s->level_memory.size(); level++)
		if (s->level_memory[level] != VK_NULL_HANDLE)
			GiveRange(native_device, s->level_memory[level], s->level_block[level], s->level_offset[level], s->level_bytes[level],
			          s->level_type[level]); // (SPARSE-POOL)
	{ // (VRAM-PRES)
		std::lock_guard live_lock(g_live_mutex);
		if (s->in_live) std::erase(g_live, s);
		for (const auto& r: s->retired) {
			GiveRange(native_device, r.memory, r.block, r.offset, r.bytes, r.type); // (SPARSE-POOL)
			g_retired_bytes.fetch_sub(r.bytes, std::memory_order_relaxed);
		}
		s->retired.clear();
	}
	if (s->per_level) {
		TexNativa::c_images.fetch_sub(1, std::memory_order_relaxed);
		TexNativa::c_bound_bytes.fetch_sub(s->bound_bytes - std::min(s->bound_bytes, s->ram_bytes), std::memory_order_relaxed); // (W17)
	}
	if (s->tail_memory != VK_NULL_HANDLE) {
		if (s->per_level) GiveRange(native_device, s->tail_memory, s->tail_block, s->tail_mem_offset, s->tail_size, s->tail_type); // (SPARSE-POOL)
		else VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(native_device, s->tail_memory, nullptr);
	}
	GestorMem::Liberar(s->gestor_id); // (M7)
	s->gestor_id = 0;
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
VkDeviceMemory AllocateSparse(GraphicContext& graphics, SparseImageMemory& s, uint64_t bytes, uint32_t* type_out = nullptr) {
	const auto           native_device = static_cast<VkDevice>(graphics.device);
	VkMemoryAllocateInfo allocate {};
	allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.allocationSize  = bytes;
	allocate.memoryTypeIndex = s.memory_type;
	VkDeviceMemory memory    = VK_NULL_HANDLE;
	VramDrv::Scope vram_drv(VramDrv::CatSparseImg); // (VRAM-DRV)
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(native_device, &allocate, nullptr, &memory) == VK_SUCCESS) {
		if (type_out != nullptr) *type_out = s.memory_type; // (SPARSE-HILO)
		return memory;
	}
	const auto& types = graphics.physical_device_memory_properties;
	for (uint32_t i = 0; i < types.memoryTypeCount; ++i) {
		if ((s.memory_bits & (1u << i)) == 0 || i == s.memory_type) continue;
		allocate.memoryTypeIndex = i;
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(native_device, &allocate, nullptr, &memory) == VK_SUCCESS) {
			if (type_out != nullptr) *type_out = i; // (SPARSE-HILO)
			return memory;
		}
	}
	return VK_NULL_HANDLE;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// (SPARSE-HILO) KYTY_SPARSE_HILO=1 (por defecto; =0: como antes). Medido en el log ([tex-nativa-coste], 30 s): "mips
// ligados/desligados" 520-968 ms (vkAllocateMemory por mip + vkFreeMemory de lo retirado + consulta de rangos, en el hilo
// de render) frente a vkQueueBindSparse 143-292 ms. Aqui:
//  1) Pool de VkDeviceMemory por (tamano de mip, tipo): desligar devuelve la memoria al pool (cuando la GPU acabo:
//     scheduler.IsFree, como antes) y ligar la toma del pool. Un hilo propio rellena el pool con lo que se pidio hace poco
//     y libera lo que pase del tope (KYTY_SPARSE_POOL_MB, por defecto 256): vkAllocateMemory/vkFreeMemory fuera del render
//     (salvo un fallo de pool: reserva sincrona, contada).
//  2) El mismo hilo envia los vkQueueBindSparse pendientes en cuanto los hay (misma cola grafica, bajo queue_mutex; cada
//     envio espera al anterior en el semaforo timeline para conservar el orden). SubmitBuffers solo espera a que lo pedido
//     antes este ENVIADO (nunca espera-antes-de-señal) y añade la espera GPU al valor timeline de ese scheduler.
// Log: [sparse-hilo] cada 30 s.
namespace {

bool HiloOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_SPARSE_HILO");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

uint64_t PoolCapBytes() {
	static const uint64_t cap = [] {
		const char* v  = std::getenv("KYTY_SPARSE_POOL_MB");
		const long  mb = v != nullptr ? std::atol(v) : 32; // (log 044349: con 256 la VRAM se pasaba 0,3-0,7 GB)
		return static_cast<uint64_t>(mb < 0 ? 0 : mb) << 20u;
	}();
	return cap;
}

uint64_t UsSince(std::chrono::steady_clock::time_point t) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count());
}

// Contadores (us y llamadas). render_*: en el hilo de render; hilo_*: en el hilo propio.
std::atomic<uint64_t> h_render_alloc_us {0}, h_render_alloc_n {0}, h_render_free_us {0}, h_render_free_n {0};
std::atomic<uint64_t> h_render_wait_us {0}, h_render_wait_n {0}, h_render_wait_max_us {0};
std::atomic<uint64_t> h_pool_hit {0}, h_pool_miss {0};
std::atomic<uint64_t> h_hilo_bind_us {0}, h_hilo_bind_n {0}, h_hilo_alloc_us {0}, h_hilo_alloc_n {0}, h_hilo_free_us {0},
    h_hilo_free_n {0};

void AtomicMax(std::atomic<uint64_t>& a, uint64_t v) {
	uint64_t cur = a.load(std::memory_order_relaxed);
	while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
}

struct PoolEntry {
	VkDeviceMemory memory;
	uint32_t       type;
};
struct PoolClass {
	std::vector<PoolEntry> free;
	uint32_t               want_type = 0, want_bits = 0; // del ultimo que pidio (para rellenar)
	uint32_t               demand    = 0;                // tomas recientes (se reduce a la mitad cada segundo)
};
std::mutex                              g_pool_mutex;
std::unordered_map<uint64_t, PoolClass> g_pool; // por tamano
uint64_t                                g_pool_idle_bytes = 0;

GraphicContext*         g_hilo_graphics = nullptr;
std::mutex              g_hilo_mutex; // g_hilo_* y despertar
std::condition_variable g_hilo_cv, g_hilo_done_cv;
bool                    g_hilo_urgent = false, g_hilo_pool_work = false;
uint64_t                g_enqueued_seq = 0, g_sent_seq = 0;                    // (bajo g_pending_mutex / g_hilo_mutex)
std::vector<std::pair<CommandScheduler*, uint64_t>> g_sched_signal, g_sched_waited; // (bajo g_hilo_mutex)
std::mutex              g_inflight_mutex; // el hilo lo tiene mientras envia: DestroySparseImage espera (imagen viva)

void StartHilo(GraphicContext& graphics);

// Memoria de `bytes` para `s`: del pool si hay (KYTY_SPARSE_HILO), si no reservada aqui.
VkDeviceMemory TakeMemory(GraphicContext& graphics, SparseImageMemory& s, uint64_t bytes, uint32_t* type_out) {
	if (HiloOn()) {
		StartHilo(graphics);
		std::lock_guard lock(g_pool_mutex);
		auto& c     = g_pool[bytes];
		c.want_type = s.memory_type;
		c.want_bits = s.memory_bits;
		c.demand++;
		for (size_t i = c.free.size(); i-- > 0;) {
			if ((s.memory_bits & (1u << c.free[i].type)) == 0) continue;
			const auto e = c.free[i];
			c.free.erase(c.free.begin() + static_cast<ptrdiff_t>(i));
			g_pool_idle_bytes -= bytes;
			h_pool_hit.fetch_add(1, std::memory_order_relaxed);
			*type_out = e.type;
			{
				std::lock_guard h(g_hilo_mutex);
				g_hilo_pool_work = true;
			}
			g_hilo_cv.notify_one();
			return e.memory;
		}
		h_pool_miss.fetch_add(1, std::memory_order_relaxed);
		{
			std::lock_guard h(g_hilo_mutex);
			g_hilo_pool_work = true;
		}
		g_hilo_cv.notify_one();
	}
	const auto t      = std::chrono::steady_clock::now();
	const auto memory = AllocateSparse(graphics, s, bytes, type_out);
	h_render_alloc_us.fetch_add(UsSince(t), std::memory_order_relaxed);
	h_render_alloc_n.fetch_add(1, std::memory_order_relaxed);
	return memory;
}

// Memoria sin uso en la GPU: al pool (el hilo libera lo que sobre) o, sin hilo, liberada aqui.
void GiveMemory(VkDevice device, VkDeviceMemory memory, uint64_t bytes, uint32_t type) {
	if (memory == VK_NULL_HANDLE) return;
	if (HiloOn() && g_hilo_graphics != nullptr) {
		{
			std::lock_guard lock(g_pool_mutex);
			g_pool[bytes].free.push_back({memory, type});
			g_pool_idle_bytes += bytes;
		}
		{
			std::lock_guard h(g_hilo_mutex);
			g_hilo_pool_work = true;
		}
		g_hilo_cv.notify_one();
		return;
	}
	const auto t = std::chrono::steady_clock::now();
	VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(device, memory, nullptr);
	h_render_free_us.fetch_add(UsSince(t), std::memory_order_relaxed);
	h_render_free_n.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// (SPARSE-POOL) KYTY_SPARSE_POOL=1 (por defecto; =0: pool por tamano de SPARSE-HILO, como antes). Log 20261125-101818:
// [sparse-hilo] "pool aciertos 6-268 fallos 2830-6669, libre 0 MB": cada mip tiene su tamano y el pool por tamano casi
// nunca acierta (y la presion lo vacia): 2830-6669 vkAllocateMemory en el render (273-760 ms/30 s) y 4166-6246
// vkFreeMemory en el hilo (821-1254 ms/30 s). Aqui la memoria de los mips sale de bloques de 64 MiB por tipo, en paginas
// de 64 KiB contiguas (un mip = un rango: el ligado sigue siendo uno por mip, con memoryOffset); desligar devuelve las
// paginas al bloque. Un bloque vacio se libera tras 2 s (se guarda uno de reserva por tipo); bajo presion de VRAM los
// vacios se liberan ya y no hay reserva. Lo libre dentro de los bloques cuenta como "pool" en SparseNativeStatsNow.
struct SparsePoolBlock {
	VkDeviceMemory                        memory = VK_NULL_HANDLE;
	uint32_t                              type   = 0;
	uint32_t                              used   = 0; // paginas usadas
	std::vector<uint8_t>                  page;       // 1: usada
	std::chrono::steady_clock::time_point empty_since {};
	VkBuffer                              copy_buffer = VK_NULL_HANDLE; // (W14) bufer sobre todo el bloque, para trasladar mips
	bool                                  no_movible  = false;          // (W14) el tipo de memoria no admite bufer
	bool                                  ram         = false;          // (W16) bloque en memoria del sistema (no VRAM)
	bool                                  vaciar      = false;          // (W17) bloque de VRAM que se vacia entero: nada nuevo entra
};

namespace {

constexpr uint64_t kPoolPage       = uint64_t {64} << 10u;
// (VRAM-AJUSTE) 16 MiB (antes 64): log 20261125-104537 "[sparse-pool] 9 bloques (576 MB), libre dentro 93 MB" + un bloque
// vacio de reserva por tipo; con bloques de 16 MiB y el bloque mas lleno primero, el hueco interno y la reserva bajan a
// ~1/4. Mips de mas de 8 MiB: memoria propia de su tamano justo (como antes los de mas de 32 MiB).
constexpr uint64_t kPoolBlockBytes = uint64_t {16} << 20u;
constexpr uint32_t kPoolPages      = static_cast<uint32_t>(kPoolBlockBytes / kPoolPage);

bool BlockPoolOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_SPARSE_POOL");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

std::vector<std::unique_ptr<SparsePoolBlock>> g_blocks; // (bajo g_pool_mutex)
uint64_t                                      g_blk_free_bytes = 0; // libre dentro de los bloques (bajo g_pool_mutex)
uint64_t                                      g_ram_free_bytes = 0; // (W16) idem, bloques en RAM (no cuentan como VRAM)
std::atomic<uint64_t> v_lib_n {0}, v_lib_bytes {0}; // (W17) bloques de VRAM vaciados y liberados (vkFreeMemory)
std::atomic<uint64_t> p_take_n {0}, p_take_us {0}, p_new_n {0}, p_new_us {0}, p_freed_n {0}, p_freed_us {0}, p_big_n {0};

// (SPARSE-ALINEADO) KYTY_SPARSE_ALINEADO=1 (por defecto; =0: primer hueco, como antes). Log 20261125-105844 [vram-pres]:
// "nativa 768 MB (mips ligados 471-477, pool 290-296)": el 38 % de los bloques es hueco suelto entre mips de tamanos
// distintos (primer hueco que cabe), y la VRAM se queda en 6281-6504 / 5416 MiB (113-120 %, Windows pagina VRAM;
// flip->present 12.8 ms frente a 7.8 de la base 20261125-101543 a 5566/5591). Con el rango empezando en un multiplo de
// su tamano redondeado a potencia de 2 (como un buddy; los mips de DS son potencias de 2 en paginas de 64 KiB), los
// huecos que deja un mip al soltarse son del mismo tamano y alineados: los reusa el siguiente mip igual o menor y los
// bloques se vacian enteros (FreeEmptyBlocks los devuelve) en vez de quedar medio llenos.
bool AlineadoOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_SPARSE_ALINEADO");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

// Un rango de `n` paginas libres en `b` (UINT32_MAX: no hay).
uint32_t FindRun(const SparsePoolBlock& b, uint32_t n) {
	if (AlineadoOn()) {
		uint32_t a = 1;
		while (a < n) a <<= 1u;
		for (uint32_t first = 0; first + n <= kPoolPages; first += a) {
			uint32_t i = first;
			while (i < first + n && b.page[i] == 0) i++;
			if (i == first + n) return first;
		}
		return UINT32_MAX;
	}
	uint32_t run = 0;
	for (uint32_t i = 0; i < kPoolPages; i++) {
		run = b.page[i] == 0 ? run + 1 : 0;
		if (run == n) return i + 1 - n;
	}
	return UINT32_MAX;
}

void Carve(SparsePoolBlock& b, uint32_t first, uint32_t n) {
	std::fill(b.page.begin() + first, b.page.begin() + first + n, uint8_t {1});
	b.used += n;
	(b.ram ? g_ram_free_bytes : g_blk_free_bytes) -= static_cast<uint64_t>(n) * kPoolPage;
}

// Memoria de `bytes` para `s`: un rango de un bloque (blk/off) o, si no cabe en bloques, memoria propia (blk null).
VkDeviceMemory TakeRange(GraphicContext& graphics, SparseImageMemory& s, uint64_t bytes, uint32_t* type_out, SparsePoolBlock** blk,
                         uint64_t* off) {
	*blk = nullptr;
	*off = 0;
	if (!BlockPoolOn() || bytes == 0 || bytes > kPoolBlockBytes / 2 || s.block == 0 || kPoolPage % s.block != 0) {
		if (BlockPoolOn()) p_big_n.fetch_add(1, std::memory_order_relaxed);
		return TakeMemory(graphics, s, bytes, type_out);
	}
	const auto     t = std::chrono::steady_clock::now();
	const uint32_t n = static_cast<uint32_t>((bytes + kPoolPage - 1) / kPoolPage);
	{
		std::lock_guard lock(g_pool_mutex);
		// (VRAM-AJUSTE) El bloque mas lleno donde quepa: los demas se vacian antes y se liberan (FreeEmptyBlocks).
		SparsePoolBlock* best = nullptr;
		uint32_t         best_first = 0;
		for (auto& b: g_blocks) {
			if (b->ram || b->vaciar || (s.memory_bits & (1u << b->type)) == 0 || kPoolPages - b->used < n) continue; // (W16/W17)
			if (best != nullptr && b->used <= best->used) continue;
			const uint32_t first = FindRun(*b, n);
			if (first == UINT32_MAX) continue;
			best       = b.get();
			best_first = first;
		}
		if (best != nullptr) {
			auto* b = best;
			const uint32_t first = best_first;
			Carve(*b, first, n);
			*blk      = b;
			*off      = static_cast<uint64_t>(first) * kPoolPage;
			*type_out = b->type;
			p_take_n.fetch_add(1, std::memory_order_relaxed);
			p_take_us.fetch_add(UsSince(t), std::memory_order_relaxed);
			return b->memory;
		}
	}
	// Ningun bloque con sitio: uno nuevo (fuera del candado).
	const auto t_new = std::chrono::steady_clock::now();
	uint32_t   type  = s.memory_type;
	const auto mem   = AllocateSparse(graphics, s, kPoolBlockBytes, &type);
	p_new_us.fetch_add(UsSince(t_new), std::memory_order_relaxed);
	if (mem == VK_NULL_HANDLE) return TakeMemory(graphics, s, bytes, type_out); // (sin bloque: memoria propia, como antes)
	p_new_n.fetch_add(1, std::memory_order_relaxed);
	auto b    = std::make_unique<SparsePoolBlock>();
	b->memory = mem;
	b->type   = type;
	b->page.assign(kPoolPages, 0);
	std::lock_guard lock(g_pool_mutex);
	g_blk_free_bytes += kPoolBlockBytes;
	Carve(*b, 0, n);
	*blk      = b.get();
	*off      = 0;
	*type_out = type;
	g_blocks.push_back(std::move(b));
	return mem;
}

// Bloques vacios liberados: todos con `all`, si no los vacios hace mas de 2 s salvo uno por tipo. Devuelve los bytes.
uint64_t FreeEmptyBlocks(VkDevice device, bool all) {
	std::vector<VkDeviceMemory> to_free;
	{
		std::lock_guard lock(g_pool_mutex);
		const auto            now = std::chrono::steady_clock::now();
		std::vector<uint32_t> kept_type;
		std::erase_if(g_blocks, [&](const std::unique_ptr<SparsePoolBlock>& b) {
			if (b->used != 0) return false;
			if (!all && !b->vaciar) { // (W17) un bloque vaciado se libera ya, sin reserva ni espera
				if (std::find(kept_type.begin(), kept_type.end(), b->type) == kept_type.end()) {
					kept_type.push_back(b->type); // reserva de este tipo
					return false;
				}
				if (now - b->empty_since < std::chrono::seconds(2)) return false;
			}
			if (b->copy_buffer != VK_NULL_HANDLE) VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer(device, b->copy_buffer, nullptr); // (W14)
			to_free.push_back(b->memory);
			if (b->vaciar && !b->ram) {
				v_lib_n.fetch_add(1, std::memory_order_relaxed);
				v_lib_bytes.fetch_add(kPoolBlockBytes, std::memory_order_relaxed);
			}
			(b->ram ? g_ram_free_bytes : g_blk_free_bytes) -= kPoolBlockBytes;
			return true;
		});
	}
	for (auto memory: to_free) {
		const auto t = std::chrono::steady_clock::now();
		VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(device, memory, nullptr);
		p_freed_us.fetch_add(UsSince(t), std::memory_order_relaxed);
		p_freed_n.fetch_add(1, std::memory_order_relaxed);
	}
	return static_cast<uint64_t>(to_free.size()) * kPoolBlockBytes;
}

// Memoria sin uso en la GPU: sus paginas al bloque, o (memoria propia) al pool por tamano.
void GiveRange(VkDevice device, VkDeviceMemory memory, SparsePoolBlock* block, uint64_t offset, uint64_t bytes, uint32_t type) {
	if (memory == VK_NULL_HANDLE) return;
	if (block == nullptr) {
		GiveMemory(device, memory, bytes, type);
		return;
	}
	const uint32_t first = static_cast<uint32_t>(offset / kPoolPage), n = static_cast<uint32_t>((bytes + kPoolPage - 1) / kPoolPage);
	std::lock_guard lock(g_pool_mutex);
	std::fill(block->page.begin() + first, block->page.begin() + first + n, uint8_t {0});
	block->used -= std::min(block->used, n);
	(block->ram ? g_ram_free_bytes : g_blk_free_bytes) += static_cast<uint64_t>(n) * kPoolPage;
	if (block->used == 0) block->empty_since = std::chrono::steady_clock::now();
}

} // namespace

namespace {
// Los ligados de memoria encolados en la cola grafica; la siguiente entrega del scheduler los espera.
void SubmitBindInfos(GraphicContext& graphics, CommandScheduler& scheduler,
                     const std::vector<VkSparseImageMemoryBindInfo>&       image_infos,
                     const std::vector<VkSparseImageOpaqueMemoryBindInfo>& opaque_infos,
                     std::chrono::steady_clock::time_point                 start) {
	auto& counters = RamNativeTex::Sparse();
	if (image_infos.empty() && opaque_infos.empty()) return;
	const auto        native_device = static_cast<VkDevice>(graphics.device);
	const VkSemaphore semaphore     = Timeline(native_device);
	uint64_t          signal        = 0;
	VkResult          result        = VK_SUCCESS;
	{
		std::lock_guard timeline_lock(g_timeline_mutex);
		// (SPARSE-HILO) Con el hilo, este envio (arrays) tambien espera al anterior: los valores del timeline se señalan en orden.
		const uint64_t wait = HiloOn() ? g_timeline_value : 0;
		signal = ++g_timeline_value;
		VkTimelineSemaphoreSubmitInfo timeline {};
		timeline.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
		timeline.waitSemaphoreValueCount   = wait != 0 ? 1 : 0;
		timeline.pWaitSemaphoreValues      = wait != 0 ? &wait : nullptr;
		timeline.signalSemaphoreValueCount = 1;
		timeline.pSignalSemaphoreValues    = &signal;
		VkBindSparseInfo info {};
		info.sType                = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
		info.pNext                = &timeline;
		info.waitSemaphoreCount   = wait != 0 ? 1 : 0; // (SPARSE-HILO)
		info.pWaitSemaphores      = wait != 0 ? &semaphore : nullptr;
		info.imageBindCount       = static_cast<uint32_t>(image_infos.size());
		info.pImageBinds          = image_infos.empty() ? nullptr : image_infos.data();
		info.imageOpaqueBindCount = static_cast<uint32_t>(opaque_infos.size());
		info.pImageOpaqueBinds    = opaque_infos.empty() ? nullptr : opaque_infos.data();
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

void SubmitBinds(GraphicContext& graphics, CommandScheduler& scheduler, VulkanImage& image,
                 std::vector<VkSparseImageMemoryBind>& image_binds, std::vector<VkSparseMemoryBind>& opaque_binds,
                 std::chrono::steady_clock::time_point start) {
	std::vector<VkSparseImageMemoryBindInfo>       image_infos;
	std::vector<VkSparseImageOpaqueMemoryBindInfo> opaque_infos;
	if (!image_binds.empty())
		image_infos.push_back({static_cast<VkImage>(image.image), static_cast<uint32_t>(image_binds.size()), image_binds.data()});
	if (!opaque_binds.empty())
		opaque_infos.push_back({static_cast<VkImage>(image.image), static_cast<uint32_t>(opaque_binds.size()), opaque_binds.data()});
	SubmitBindInfos(graphics, scheduler, image_infos, opaque_infos, start);
}

} // namespace

void SparseUpdateLayers(GraphicContext& graphics, CommandScheduler& scheduler, VulkanImage& image, uint64_t address,
                        uint64_t stride, uint64_t begin, uint64_t end, uint64_t gone_begin, uint64_t gone_end) {
	if (image.sparse == nullptr || image.image == nullptr || stride == 0 || end <= begin) return;
	auto&          s     = *image.sparse;
	struct GestorTam { // (M7) al salir (por cualquier return): VRAM ligada de verdad al gestor, solo si cambio
		SparseImageMemory& m;
		~GestorTam() {
			const uint64_t b = m.bound_bytes - std::min(m.bound_bytes, m.ram_bytes);
			if (m.gestor_id != 0 && b != m.gestor_bytes) {
				m.gestor_bytes = b;
				GestorMem::CambiarTamano(m.gestor_id, b);
			}
		}
	} gestor_tam {s};
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
	SubmitBinds(graphics, scheduler, image, image_binds, opaque_binds, start);
}

// (N1, KYTY_TEX_NATIVA) Los mips de una textura nativa cuyo rango [address + offset, + size) el juego tiene comprometido
// ahora (menos [gone_begin, gone_end), a punto de liberarse) quedan ligados, los demas desligados, para los mips que tocan
// [begin, end). La cola de mips se liga una vez. La memoria desligada se libera cuando la GPU termino lo grabado antes.
void SparseUpdateLevels(GraphicContext& graphics, CommandScheduler& scheduler, VulkanImage& image, uint64_t address,
                        const uint64_t* offsets, const uint64_t* sizes, uint32_t levels, uint32_t min_level, uint64_t begin,
                        uint64_t end, uint64_t gone_begin, uint64_t gone_end) {
	if (image.sparse == nullptr || image.image == nullptr || !image.sparse->per_level || end <= begin) return;
	TexNativa::Coste coste(TexNativa::kCLevels); // (N2+N3f)
	auto&      s           = *image.sparse;
	struct GestorTam { // (M7) al salir (por cualquier return): VRAM ligada de verdad al gestor, solo si cambio
		SparseImageMemory& m;
		~GestorTam() {
			const uint64_t b = m.bound_bytes - std::min(m.bound_bytes, m.ram_bytes);
			if (m.gestor_id != 0 && b != m.gestor_bytes) {
				m.gestor_bytes = b;
				GestorMem::CambiarTamano(m.gestor_id, b);
			}
		}
	} gestor_tam {s};
	if (s.moving.load(std::memory_order_relaxed) != 0) s.move_cancel.store(true, std::memory_order_relaxed); // (W14)
	const auto start       = std::chrono::steady_clock::now();
	const auto native_device = static_cast<VkDevice>(graphics.device);
	// Memoria desligada antes cuyo trabajo de GPU ya acabo.
	{
		std::lock_guard live_lock(g_live_mutex); // (VRAM-PRES)
		std::erase_if(s.retired, [&](const SparseImageMemory::Retired& r) {
			if (!scheduler.IsFree(r.tick)) return false;
			GiveRange(native_device, r.memory, r.block, r.offset, r.bytes, r.type); // (SPARSE-POOL) al bloque / pool
			TexNativa::c_freed_bytes.fetch_add(r.bytes, std::memory_order_relaxed);
			g_retired_bytes.fetch_sub(r.bytes, std::memory_order_relaxed); // (VRAM-PRES)
			return true;
		});
	}
	thread_local std::vector<std::pair<uint64_t, uint64_t>> parts;
	std::vector<VkSparseImageMemoryBind>                     image_binds;
	std::vector<VkSparseMemoryBind>                          opaque_binds;
	const uint32_t top = std::min({s.tail_first_lod, levels, static_cast<uint32_t>(s.level_memory.size())});
	for (uint32_t level = 0; level < top; level++) {
		const uint64_t lo_l = address + offsets[level], hi_l = lo_l + sizes[level];
		if (sizes[level] == 0 || hi_l <= begin || lo_l >= end) continue;
		bool committed = false;
		if (level >= min_level) {
			Libs::LibKernel::Memory::MappedParts(lo_l, sizes[level], &parts);
			for (const auto& [at, bytes]: parts) {
				const uint64_t lo = std::max(at, lo_l), hi = std::min(at + bytes, hi_l);
				if (lo >= hi) continue;
				if (gone_end <= gone_begin || lo < gone_begin || hi > gone_end) {
					committed = true;
					break;
				}
			}
		}
		const bool bound = s.level_memory[level] != VK_NULL_HANDLE;
		if (committed == bound) continue;
		VkSparseImageMemoryBind item {};
		item.subresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0};
		item.offset      = {0, 0, 0};
		item.extent      = {std::max(s.extent.width >> level, 1u), std::max(s.extent.height >> level, 1u), 1u};
		if (committed) {
			uint32_t   type   = s.memory_type;
			SparsePoolBlock* blk = nullptr;
			uint64_t         off = 0;
			const auto memory = TakeRange(graphics, s, s.level_bytes[level], &type, &blk, &off); // (SPARSE-POOL)
			if (memory == VK_NULL_HANDLE) {
				TexNativa::c_alloc_failed.fetch_add(1, std::memory_order_relaxed);
				continue; // sin ligar: se lee como cero, nunca falla (sparse residency)
			}
			s.level_memory[level] = memory;
			s.level_type[level]   = type; // (SPARSE-HILO)
			s.level_block[level]  = blk;  // (SPARSE-POOL)
			s.level_offset[level] = off;
			item.memory           = memory;
			item.memoryOffset     = off;
			s.bound_bytes += s.level_bytes[level];
			TexNativa::c_bound_bytes.fetch_add(s.level_bytes[level], std::memory_order_relaxed);
			TexNativa::c_levels_bound.fetch_add(1, std::memory_order_relaxed);
		} else {
			{
				std::lock_guard live_lock(g_live_mutex); // (VRAM-PRES)
				s.retired.push_back({s.level_memory[level], s.level_bytes[level], scheduler.CurrentTick(), s.level_type[level],
				                     s.level_block[level], s.level_offset[level]}); // (SPARSE-POOL)
				s.retired_scheduler = &scheduler;
				g_retired_bytes.fetch_add(s.level_bytes[level], std::memory_order_relaxed);
				if (!s.in_live) {
					s.in_live = true;
					g_live.push_back(&s);
				}
			}
			const bool en_ram     = s.level_block[level] != nullptr && s.level_block[level]->ram; // (W17)
			s.level_memory[level] = VK_NULL_HANDLE;
			s.level_block[level]  = nullptr;
			s.level_offset[level] = 0;
			item.memory           = VK_NULL_HANDLE;
			s.bound_bytes -= std::min(s.bound_bytes, s.level_bytes[level]);
			if (en_ram) s.ram_bytes -= std::min(s.ram_bytes, s.level_bytes[level]); // (W17) no estaba en c_bound_bytes
			else TexNativa::c_bound_bytes.fetch_sub(s.level_bytes[level], std::memory_order_relaxed);
			TexNativa::c_levels_unbound.fetch_add(1, std::memory_order_relaxed);
		}
		image_binds.push_back(item);
	}
	if (s.tail_first_lod < s.levels && !s.tail_bound) {
		s.tail_memory = TakeRange(graphics, s, s.tail_size, &s.tail_type, &s.tail_block, &s.tail_mem_offset); // (SPARSE-POOL)
		if (s.tail_memory != VK_NULL_HANDLE) {
			opaque_binds.push_back({s.tail_offset, s.tail_size, s.tail_memory, s.tail_mem_offset, 0});
			s.tail_bound = true;
			s.bound_bytes += s.tail_size;
			TexNativa::c_bound_bytes.fetch_add(s.tail_size, std::memory_order_relaxed);
		} else {
			TexNativa::c_alloc_failed.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (image_binds.empty() && opaque_binds.empty()) return;
	// (P5) A la cola pendiente; SparseFlushPendingBinds los envia todos antes de la siguiente entrega de `scheduler`.
	std::lock_guard pending_lock(g_pending_mutex);
	if (s.pending_scheduler != nullptr && s.pending_scheduler != &scheduler) {
		// (otro scheduler: lo pendiente se envia ya, como antes, para no mezclar esperas)
		auto* other = s.pending_scheduler;
		std::erase(g_pending, &s);
		s.pending_scheduler = nullptr;
		SubmitBinds(graphics, *other, image, s.pending_image, s.pending_opaque, start);
		s.pending_image.clear();
		s.pending_opaque.clear();
	}
	if (s.pending_scheduler == nullptr) {
		s.pending_scheduler = &scheduler;
		g_pending.push_back(&s);
	}
	for (const auto& item: image_binds) {
		// Un mip ya pendiente: gana el ultimo estado (no dos ligados de la misma region en un envio).
		auto it = std::find_if(s.pending_image.begin(), s.pending_image.end(), [&](const VkSparseImageMemoryBind& b) {
			return b.subresource.mipLevel == item.subresource.mipLevel;
		});
		if (it != s.pending_image.end()) *it = item;
		else s.pending_image.push_back(item);
	}
	s.pending_opaque.insert(s.pending_opaque.end(), opaque_binds.begin(), opaque_binds.end());
	(void)start;
	if (HiloOn()) { // (SPARSE-HILO) el hilo los envia ya (el render no espera en SubmitBuffers si llego a tiempo)
		g_enqueued_seq++;
		{
			std::lock_guard h(g_hilo_mutex);
			g_hilo_urgent = true;
		}
		g_hilo_cv.notify_one();
	}
}

namespace {

void PrintHilo(); // (SPARSE-HILO)

uint64_t& SchedValue(std::vector<std::pair<CommandScheduler*, uint64_t>>& v, CommandScheduler* s) {
	for (auto& [k, value]: v)
		if (k == s) return value;
	v.emplace_back(s, 0);
	return v.back().second;
}

// (SPARSE-HILO) Un VkBindSparseInfo en la cola grafica (bajo queue_mutex), que espera al envio anterior del timeline y
// señala uno nuevo. Devuelve el valor señalado.
uint64_t SubmitBindRaw(GraphicContext& graphics, const std::vector<VkSparseImageMemoryBindInfo>& image_infos,
                       const std::vector<VkSparseImageOpaqueMemoryBindInfo>& opaque_infos) {
	const auto        native_device = static_cast<VkDevice>(graphics.device);
	const VkSemaphore semaphore     = Timeline(native_device);
	uint64_t          signal = 0, wait = 0;
	VkResult          result = VK_SUCCESS;
	{
		std::lock_guard timeline_lock(g_timeline_mutex);
		wait   = g_timeline_value; // ya enviado (bajo este mismo mutex): nunca espera-antes-de-señal
		signal = ++g_timeline_value;
		VkTimelineSemaphoreSubmitInfo timeline {};
		timeline.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
		timeline.waitSemaphoreValueCount   = wait != 0 ? 1 : 0;
		timeline.pWaitSemaphoreValues      = wait != 0 ? &wait : nullptr;
		timeline.signalSemaphoreValueCount = 1;
		timeline.pSignalSemaphoreValues    = &signal;
		VkBindSparseInfo info {};
		info.sType                = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
		info.pNext                = &timeline;
		info.waitSemaphoreCount   = wait != 0 ? 1 : 0;
		info.pWaitSemaphores      = wait != 0 ? &semaphore : nullptr;
		info.imageBindCount       = static_cast<uint32_t>(image_infos.size());
		info.pImageBinds          = image_infos.empty() ? nullptr : image_infos.data();
		info.imageOpaqueBindCount = static_cast<uint32_t>(opaque_infos.size());
		info.pImageOpaqueBinds    = opaque_infos.empty() ? nullptr : opaque_infos.data();
		info.signalSemaphoreCount = 1;
		info.pSignalSemaphores    = &semaphore;
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
	return signal;
}

// (SPARSE-HILO) Envia todo lo pendiente (de todos los schedulers). Solo el hilo propio.
void HiloSendPending(GraphicContext& graphics) {
	struct Item {
		CommandScheduler*                    scheduler;
		VkImage                              image;
		std::vector<VkSparseImageMemoryBind> binds;
		std::vector<VkSparseMemoryBind>      opaque;
	};
	thread_local std::vector<Item> items;
	std::lock_guard inflight(g_inflight_mutex);
	uint64_t        seq = 0;
	items.clear();
	{
		std::lock_guard pending_lock(g_pending_mutex);
		seq = g_enqueued_seq;
		for (auto* s: g_pending) {
			items.push_back({s->pending_scheduler, s->vk_image, std::move(s->pending_image), std::move(s->pending_opaque)});
			s->pending_image.clear();
			s->pending_opaque.clear();
			s->pending_scheduler = nullptr;
		}
		g_pending.clear();
	}
	if (!items.empty()) {
		const auto t = std::chrono::steady_clock::now();
		std::vector<CommandScheduler*> scheds;
		for (const auto& it: items)
			if (std::find(scheds.begin(), scheds.end(), it.scheduler) == scheds.end()) scheds.push_back(it.scheduler);
		std::vector<VkSparseImageMemoryBindInfo>       image_infos;
		std::vector<VkSparseImageOpaqueMemoryBindInfo> opaque_infos;
		for (auto* sched: scheds) {
			image_infos.clear();
			opaque_infos.clear();
			for (const auto& it: items) {
				if (it.scheduler != sched) continue;
				if (!it.binds.empty()) image_infos.push_back({it.image, static_cast<uint32_t>(it.binds.size()), it.binds.data()});
				if (!it.opaque.empty()) opaque_infos.push_back({it.image, static_cast<uint32_t>(it.opaque.size()), it.opaque.data()});
			}
			if (image_infos.empty() && opaque_infos.empty()) continue;
			const uint64_t signal = SubmitBindRaw(graphics, image_infos, opaque_infos);
			std::lock_guard h(g_hilo_mutex);
			SchedValue(g_sched_signal, sched) = signal;
		}
		const uint64_t us = UsSince(t);
		h_hilo_bind_us.fetch_add(us, std::memory_order_relaxed);
		h_hilo_bind_n.fetch_add(1, std::memory_order_relaxed);
		auto& counters = RamNativeTex::Sparse();
		counters.bind_calls.fetch_add(1, std::memory_order_relaxed);
		counters.bind_us.fetch_add(us, std::memory_order_relaxed);
	}
	{
		std::lock_guard h(g_hilo_mutex);
		g_sent_seq = std::max(g_sent_seq, seq);
	}
	g_hilo_done_cv.notify_all();
}

// (SPARSE-HILO) Rellena el pool con lo pedido hace poco y libera lo que pase del tope. Solo el hilo propio.
void HiloPoolWork(GraphicContext& graphics, bool decay) {
	const auto device = static_cast<VkDevice>(graphics.device);
	const bool pressure = g_pres_pressure.load(std::memory_order_relaxed); // (VRAM-PRES) sin rellenar, vaciado
	const auto cap      = pressure ? uint64_t {0} : PoolCapBytes();
	// Liberar lo que sobra.
	std::vector<VkDeviceMemory> to_free;
	struct Need {
		uint64_t bytes;
		uint32_t type, bits, count;
	};
	std::vector<Need> needs;
	{
		std::lock_guard lock(g_pool_mutex);
		for (auto& [bytes, c]: g_pool) {
			while (g_pool_idle_bytes > cap && !c.free.empty()) {
				to_free.push_back(c.free.back().memory);
				c.free.pop_back();
				g_pool_idle_bytes -= bytes;
			}
		}
		// Rellenar: tantas libres como tomas recientes (minimo 2 si se usa la clase), dentro del tope.
		uint64_t budget = cap > g_pool_idle_bytes ? cap - g_pool_idle_bytes : 0;
		for (auto& [bytes, c]: g_pool) {
			const uint32_t target = (c.demand == 0 || pressure) ? 0 : std::min<uint32_t>(std::max<uint32_t>(c.demand, 2), 64);
			if (c.free.size() < target) {
				uint32_t n = target - static_cast<uint32_t>(c.free.size());
				n          = static_cast<uint32_t>(std::min<uint64_t>(n, bytes == 0 ? 0 : budget / bytes));
				if (n != 0) needs.push_back({bytes, c.want_type, c.want_bits, n});
				budget -= static_cast<uint64_t>(n) * bytes;
			}
			if (decay) c.demand /= 2;
		}
	}
	if (BlockPoolOn() && (decay || pressure)) FreeEmptyBlocks(device, pressure); // (SPARSE-POOL) fuera del render
	for (auto memory: to_free) {
		const auto t = std::chrono::steady_clock::now();
		VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(device, memory, nullptr);
		h_hilo_free_us.fetch_add(UsSince(t), std::memory_order_relaxed);
		h_hilo_free_n.fetch_add(1, std::memory_order_relaxed);
	}
	for (const auto& need: needs) {
		SparseImageMemory probe; // solo tipo y bits para AllocateSparse
		probe.memory_type = need.type;
		probe.memory_bits = need.bits;
		for (uint32_t i = 0; i < need.count; i++) {
			// (Lo urgente primero: si el render espera ligados, se corta el relleno.)
			{
				std::lock_guard h(g_hilo_mutex);
				if (g_hilo_urgent) return;
			}
			const auto t    = std::chrono::steady_clock::now();
			uint32_t   type = need.type;
			const auto mem  = AllocateSparse(graphics, probe, need.bytes, &type);
			h_hilo_alloc_us.fetch_add(UsSince(t), std::memory_order_relaxed);
			h_hilo_alloc_n.fetch_add(1, std::memory_order_relaxed);
			if (mem == VK_NULL_HANDLE) break;
			std::lock_guard lock(g_pool_mutex);
			g_pool[need.bytes].free.push_back({mem, type});
			g_pool_idle_bytes += need.bytes;
		}
	}
}

void StartHilo(GraphicContext& graphics) {
	static std::once_flag once;
	std::call_once(once, [&] {
		g_hilo_graphics = &graphics;
		std::thread([] {
			auto& graphics   = *g_hilo_graphics;
			auto  last_decay = std::chrono::steady_clock::now();
			for (;;) {
				bool urgent = false, pool = false;
				{
					std::unique_lock h(g_hilo_mutex);
					g_hilo_cv.wait_for(h, std::chrono::milliseconds(100), [] { return g_hilo_urgent || g_hilo_pool_work; });
					urgent           = g_hilo_urgent;
					pool             = g_hilo_pool_work;
					g_hilo_urgent    = false;
					g_hilo_pool_work = false;
				}
				if (urgent) HiloSendPending(graphics);
				const bool decay = std::chrono::steady_clock::now() - last_decay > std::chrono::seconds(1);
				if (decay) last_decay = std::chrono::steady_clock::now();
				if (pool || decay) HiloPoolWork(graphics, decay);
			}
		}).detach();
	});
}

void PrintHilo() {
	static auto last = std::chrono::steady_clock::now();
	if (std::chrono::steady_clock::now() - last < std::chrono::seconds(30)) return;
	last = std::chrono::steady_clock::now();
	const auto x = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
	const double ra = x(h_render_alloc_us) / 1000.0, rf = x(h_render_free_us) / 1000.0, rw = x(h_render_wait_us) / 1000.0;
	const double hb = x(h_hilo_bind_us) / 1000.0, ha = x(h_hilo_alloc_us) / 1000.0, hf = x(h_hilo_free_us) / 1000.0;
	const auto   ran = x(h_render_alloc_n), rfn = x(h_render_free_n), rwn = x(h_render_wait_n), hbn = x(h_hilo_bind_n),
	           han = x(h_hilo_alloc_n), hfn = x(h_hilo_free_n), hit = x(h_pool_hit), miss = x(h_pool_miss);
	const double rmax = x(h_render_wait_max_us) / 1000.0;
	uint64_t     idle = 0;
	{
		std::lock_guard lock(g_pool_mutex);
		idle = g_pool_idle_bytes;
	}
	std::printf("[sparse-hilo] 30 s (%s): render: reservas %.1f ms (%llu), liberaciones %.1f ms (%llu), espera de ligados %.1f ms "
	            "(%llu entregas, max %.2f ms); hilo: vkQueueBindSparse %.1f ms (%llu), reservas %.1f ms (%llu), liberaciones %.1f "
	            "ms (%llu); pool aciertos %llu fallos %llu, libre %llu MB; render ahorrado ~%.1f ms\n",
	            HiloOn() ? "hilo" : "sin hilo", ra, static_cast<unsigned long long>(ran), rf, static_cast<unsigned long long>(rfn), rw,
	            static_cast<unsigned long long>(rwn), rmax, hb, static_cast<unsigned long long>(hbn), ha,
	            static_cast<unsigned long long>(han), hf, static_cast<unsigned long long>(hfn), static_cast<unsigned long long>(hit),
	            static_cast<unsigned long long>(miss), static_cast<unsigned long long>(idle >> 20u),
	            std::max(0.0, hb + ha + hf - rw));
	if (BlockPoolOn()) { // (SPARSE-POOL)
		size_t   blocks = 0;
		uint64_t inside = 0;
		{
			std::lock_guard lock(g_pool_mutex);
			blocks = g_blocks.size();
			inside = g_blk_free_bytes;
		}
		std::printf("[sparse-pool] 30 s: rangos tomados %llu en %.1f ms; bloques nuevos %llu en %.1f ms (render); bloques liberados %llu "
		            "en %.1f ms; fuera de bloques %llu; ahora %zu bloques (%llu MB), libre dentro %llu MB\n",
		            static_cast<unsigned long long>(x(p_take_n)), x(p_take_us) / 1000.0, static_cast<unsigned long long>(x(p_new_n)),
		            x(p_new_us) / 1000.0, static_cast<unsigned long long>(x(p_freed_n)), x(p_freed_us) / 1000.0,
		            static_cast<unsigned long long>(x(p_big_n)), blocks, static_cast<unsigned long long>((blocks * kPoolBlockBytes) >> 20u),
		            static_cast<unsigned long long>(inside >> 20u));
	}
	std::fflush(stdout);
}

} // namespace

void SparseFlushPendingBinds(GraphicContext& graphics, CommandScheduler& scheduler) {
	PrintHilo(); // (SPARSE-HILO)
	if (HiloOn() && g_hilo_graphics != nullptr) {
		// (SPARSE-HILO) Esperar solo a que lo encolado antes este ENVIADO por el hilo; luego, la espera GPU en la entrega.
		uint64_t target = 0;
		{
			std::lock_guard pending_lock(g_pending_mutex);
			target = g_enqueued_seq;
		}
		uint64_t value = 0;
		{
			std::unique_lock h(g_hilo_mutex);
			if (g_sent_seq < target) {
				TexNativa::Coste coste_bind(TexNativa::kCBind); // (en este modo: espera del render)
				const auto       t = std::chrono::steady_clock::now();
				g_hilo_urgent      = true;
				g_hilo_cv.notify_one();
				g_hilo_done_cv.wait(h, [&] { return g_sent_seq >= target; });
				const uint64_t us = UsSince(t);
				h_render_wait_us.fetch_add(us, std::memory_order_relaxed);
				h_render_wait_n.fetch_add(1, std::memory_order_relaxed);
				AtomicMax(h_render_wait_max_us, us);
			}
			const uint64_t signal = SchedValue(g_sched_signal, &scheduler);
			uint64_t&      waited = SchedValue(g_sched_waited, &scheduler);
			if (signal > waited) value = waited = signal;
		}
		if (value != 0) scheduler.WaitBeforeNextSubmit(vk::Semaphore(Timeline(static_cast<VkDevice>(graphics.device))), value);
		return;
	}
	thread_local std::vector<SparseImageMemory*> mine;
	{
		std::lock_guard pending_lock(g_pending_mutex);
		if (g_pending.empty()) return;
		TexNativa::Coste coste_bind(TexNativa::kCBind); // (N2+N3f) vkQueueBindSparse de nativas solo desde aqui
		const auto       start = std::chrono::steady_clock::now();
		mine.clear();
		std::erase_if(g_pending, [&](SparseImageMemory* s) {
			if (s->pending_scheduler != &scheduler) return false;
			mine.push_back(s);
			return true;
		});
		if (mine.empty()) return;
		// Un VkBindSparseInfo con la info de cada imagen (los vectores viven en cada imagen hasta aqui: bajo el mutex,
		// DestroySparseImage no puede borrarlas a la vez).
		std::vector<VkSparseImageMemoryBindInfo>       image_infos;
		std::vector<VkSparseImageOpaqueMemoryBindInfo> opaque_infos;
		for (auto* s: mine) {
			if (!s->pending_image.empty())
				image_infos.push_back({s->vk_image, static_cast<uint32_t>(s->pending_image.size()), s->pending_image.data()});
			if (!s->pending_opaque.empty())
				opaque_infos.push_back({s->vk_image, static_cast<uint32_t>(s->pending_opaque.size()), s->pending_opaque.data()});
		}
		SubmitBindInfos(graphics, scheduler, image_infos, opaque_infos, start);
		for (auto* s: mine) {
			s->pending_scheduler = nullptr;
			s->pending_image.clear();
			s->pending_opaque.clear();
		}
	}
	mine.clear();
}

bool SparseLevelBound(const VulkanImage& image, uint32_t level) {
	if (image.sparse == nullptr || !image.sparse->per_level) return true;
	const auto& s = *image.sparse;
	if (level >= s.tail_first_lod) return s.tail_bound;
	return level < s.level_memory.size() && s.level_memory[level] != VK_NULL_HANDLE;
}

bool SparseIsPerLevel(const VulkanImage& image) {
	return image.sparse != nullptr && image.sparse->per_level;
}

void SparseBoundLayers(const VulkanImage& image, std::vector<uint32_t>* layers) {
	layers->clear();
	if (image.sparse == nullptr) return;
	for (uint32_t layer = 0; layer < image.sparse->layers; layer++)
		if (image.sparse->slot_of_layer[layer] >= 0) layers->push_back(layer);
}

} // namespace Libs::Graphics

// ---------------------------------------------------------------------------------------------------------------------
// (VRAM-PRES) KYTY_TEX_NATIVA_PRESUPUESTO=1 (por defecto; =0: como antes). Log 20261124-044349: la memoria por mip de las
// texturas nativas (vkAllocateMemory propio, fuera de VMA) sale en la parte "driver" de la VRAM (2077-2114 MiB): VRAM
// ligada 414-503 MB + pool 184-255 MB + lo retirado que solo se devolvia cuando la misma imagen volvia a SparseUpdateLevels.
// Aqui el recolector lo ve y lo devuelve.
namespace Libs::Graphics {

bool SparsePresupuestoOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_NATIVA_PRESUPUESTO");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

SparseNativeStats SparseNativeStatsNow() {
	SparseNativeStats st;
	st.bound   = TexNativa::c_bound_bytes.load(std::memory_order_relaxed);
	st.retired = g_retired_bytes.load(std::memory_order_relaxed);
	std::lock_guard lock(g_pool_mutex);
	st.pool = g_pool_idle_bytes + g_blk_free_bytes; // (SPARSE-POOL) lo libre dentro de los bloques tambien es VRAM
	return st;
}

uint64_t SparseBoundBytes(const VulkanImage& image) {
	if (image.sparse == nullptr || !image.sparse->per_level) return 0;
	const auto* s = image.sparse; // (W17) solo lo que esta en VRAM
	return s->bound_bytes - std::min(s->bound_bytes, s->ram_bytes);
}

uint64_t SparsePresupuestoReclaim(GraphicContext& graphics, bool pressure) {
	g_pres_pressure.store(pressure, std::memory_order_relaxed);
	const auto device = static_cast<VkDevice>(graphics.device);
	uint64_t   freed  = 0;
	{ // Lo retirado cuya GPU ya acabo, de todas las imagenes (no solo de la que vuelve a ligar).
		std::lock_guard live_lock(g_live_mutex);
		std::erase_if(g_live, [&](SparseImageMemory* s) {
			if (s->retired_scheduler != nullptr) {
				std::erase_if(s->retired, [&](const SparseImageMemory::Retired& r) {
					if (!s->retired_scheduler->IsFree(r.tick)) return false;
					GiveRange(device, r.memory, r.block, r.offset, r.bytes, r.type); // (SPARSE-POOL)
					TexNativa::c_freed_bytes.fetch_add(r.bytes, std::memory_order_relaxed);
					g_retired_bytes.fetch_sub(r.bytes, std::memory_order_relaxed);
					if (!HiloOn()) freed += r.bytes; // (sin hilo GiveMemory libera; con hilo va al pool y cuenta al vaciarlo)
					return true;
				});
			}
			if (!s->retired.empty()) return false;
			s->in_live = false;
			return true;
		});
	}
	if (pressure && BlockPoolOn()) freed += FreeEmptyBlocks(device, true); // (SPARSE-POOL) vacios fuera ya
	if (!pressure || !HiloOn()) return freed;
	// Bajo presion: el pool entero liberado ya (vkFreeMemory aqui; el hilo tampoco lo rellena mientras dure).
	std::vector<VkDeviceMemory> to_free;
	{
		std::lock_guard lock(g_pool_mutex);
		for (auto& [bytes, c]: g_pool) {
			for (const auto& e: c.free) to_free.push_back(e.memory);
			freed += bytes * c.free.size();
			c.free.clear();
		}
		g_pool_idle_bytes = 0;
	}
	for (auto memory: to_free) VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(device, memory, nullptr);
	return freed;
}

} // namespace Libs::Graphics

// ---------------------------------------------------------------------------------------------------------------------
// (W14, SPARSE-COMPACTAR) KYTY_SPARSE_COMPACTAR=1 (por defecto; =0 apagado). Log 20261125-120123 [sparse-pool]: 43 bloques
// de 16 MiB (688 MB) con 170-201 MB libres en huecos sueltos dentro: ni el bloque mas lleno primero ni el ajuste alineado
// vacian bloques enteros, porque los mips que quedan en los bloques casi vacios no se mueven nunca. Aqui, cada pasada:
// 1) los traslados cuya copia la GPU ya acabo se ligan a su sitio nuevo (cola pendiente normal, P5/SPARSE-HILO) y la
//    memoria vieja se retira (vuelve al bloque cuando la GPU acabo con lo grabado antes: `retired`, como al desligar);
// 2) si no hay traslados en curso y lo libre dentro pasa de 2 bloques: el bloque menos lleno cuyos mips caben en otros
//    mas llenos se vacia: rango nuevo en el bloque mas lleno que tenga sitio, vkCmdCopyBuffer entre los buferes que
//    cubren cada bloque (misma imagen, mismo mip, mismo tamano: los mismos bytes), sin ligar aun.
// Solo mips de imagenes que el recolector da por no usadas (`sin_uso`); si se usan o cambian su ligado antes del paso 1,
// el traslado se descarta (SparseMoverCancelar / SparseUpdateLevels) y el rango nuevo se devuelve. Una sola copia: la
// memoria vieja sale en cuanto la GPU deja de leerla. Linea [sparse-compactar] cada 30 s.
namespace Libs::Graphics {
namespace {

struct Traslado {
	SparseImageMemory* s;
	uint32_t           level;
	SparsePoolBlock*   src;
	uint64_t           src_off;
	SparsePoolBlock*   dst;
	uint64_t           dst_off;
	uint64_t           bytes;
	uint64_t           tick;
	CommandScheduler*  scheduler;
	bool               a_ram = false; // (W16)
};
std::mutex            g_move_mutex;
std::vector<Traslado> g_moves; // (bajo g_move_mutex)
std::atomic<uint64_t> c_mov_n {0}, c_mov_bytes {0}, c_mov_cancel {0}, c_mov_sinbufer {0}, c_pasadas {0};
// (W16) contadores de TEX-NATIVA-RAM
std::atomic<uint64_t> r_mov_n {0}, r_mov_bytes {0}, r_cancel {0}, r_total_bytes {0}, r_usos {0}, r_usos_bytes {0};
std::atomic<uint64_t> r_vaciando {0}; // (W17) bloques de VRAM marcados para vaciar

bool CompactarOn() {
	static const bool on = [] {
		// (W17) Apagado por defecto: prueba 52 (log 20261125-125941) dio texturas negras. Copiar en crudo la memoria de un
		// mip sparse (tiling optimo) a otra memoria y religarlo no tiene contenido definido en Vulkan. =1: prueba.
		const char* v = std::getenv("KYTY_SPARSE_COMPACTAR");
		const bool  r = v != nullptr && v[0] == '1';
		std::printf("[sparse-compactar] %s (KYTY_SPARSE_COMPACTAR=1: encendido, copia en crudo)\n", r ? "encendido" : "apagado");
		std::fflush(stdout);
		return r;
	}();
	return on;
}

// Bufer de copia sobre todo el bloque (bajo g_pool_mutex). False: ese tipo de memoria no admite buferes.
bool BloqueBufer(VkDevice device, SparsePoolBlock& b) {
	if (b.copy_buffer != VK_NULL_HANDLE) return true;
	if (b.no_movible) return false;
	VkBufferCreateInfo info {};
	info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size        = kPoolBlockBytes;
	info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer buffer  = VK_NULL_HANDLE;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) {
		b.no_movible = true;
		return false;
	}
	VkMemoryRequirements req {};
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements(device, buffer, &req);
	if ((req.memoryTypeBits & (1u << b.type)) == 0 || req.size > kPoolBlockBytes ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory(device, buffer, b.memory, 0) != VK_SUCCESS) {
		VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer(device, buffer, nullptr);
		b.no_movible = true;
		c_mov_sinbufer.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	b.copy_buffer = buffer;
	return true;
}

void DevolverDestino(VkDevice device, const Traslado& m) {
	GiveRange(device, m.dst->memory, m.dst, m.dst_off, m.bytes, m.dst->type);
}

void InformeCompactar() {
	static auto last = std::chrono::steady_clock::now();
	if (std::chrono::steady_clock::now() - last < std::chrono::seconds(30)) return;
	last = std::chrono::steady_clock::now();
	size_t   blocks = 0;
	uint64_t inside = 0;
	{
		std::lock_guard lock(g_pool_mutex);
		blocks = g_blocks.size();
		inside = g_blk_free_bytes;
	}
	std::printf("[sparse-compactar] 30 s: mips trasladados %llu (%llu MiB), descartados por uso %llu, bloques sin bufer %llu, "
	            "pasadas %llu; ahora %zu bloques (%llu MiB), libre dentro %llu MiB (los vacios los libera [sparse-pool])\n",
	            static_cast<unsigned long long>(c_mov_n.exchange(0)), static_cast<unsigned long long>(c_mov_bytes.exchange(0) >> 20u),
	            static_cast<unsigned long long>(c_mov_cancel.exchange(0)), static_cast<unsigned long long>(c_mov_sinbufer.load()),
	            static_cast<unsigned long long>(c_pasadas.exchange(0)), blocks,
	            static_cast<unsigned long long>((blocks * kPoolBlockBytes) >> 20u), static_cast<unsigned long long>(inside >> 20u));
	std::fflush(stdout);
}

void CompactarSoltarImagen(SparseImageMemory* s) {
	std::lock_guard lock(g_move_mutex);
	for (auto& m: g_moves)
		if (m.s == s) m.s = nullptr; // el rango nuevo vuelve al bloque cuando la copia acabo (paso 1)
	s->moving.store(0, std::memory_order_relaxed);
}


// ---------------------------------------------------------------------------------------------------------------------
// (W16, TEX-NATIVA-RAM) KYTY_TEX_NATIVA_RAM=1 (por defecto apagado). Que la VRAM no desborde pasando texturas nativas
// a la RAM del sistema, UNA sola copia. Objetivo: el tope efectivo (VramTope::g_tope_bytes, solo lectura) o, con
// KYTY_VRAM_MITAD=1 (prueba; implica KYTY_TEX_NATIVA_RAM=1 y sin tope de MiB salvo KYTY_TEX_NATIVA_RAM_MB), la mitad.
// Si el uso del proceso (VramTope::g_uso_bytes) pasa del objetivo, cada pasada del recolector traslada mips de las
// nativas sin usar >= 120 fotogramas (`sin_uso`: nunca nada del fotograma actual), uso mas antiguo y menos usos
// primero, mips grandes (0, 1) primero, a bloques de 16 MiB de un tipo de memoria que la imagen admite y cuyo heap no
// es DEVICE_LOCAL (la GPU lo lee por PCIe). Maquinaria de SPARSE-COMPACTAR: copia GPU bufer a bufer, ligado nuevo al
// acabar la copia, la memoria de VRAM se retira y el bloque vacio lo libera FreeEmptyBlocks. Bajo el objetivo no se
// mueve nada. Un mip en RAM que se desliga y vuelve a ligar vuelve a la VRAM (TakeRange no usa bloques de RAM).
// Linea [tex-nativa-ram] cada 30 s; aviso si los usos de mips en RAM pasan de 256 MiB/s (coste PCIe).
uint64_t NowMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool MitadOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_MITAD");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

bool NativaRamOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_NATIVA_RAM");
		const bool  r = (v != nullptr && v[0] == '1') || MitadOn();
		std::printf("[tex-nativa-ram] %s%s (KYTY_TEX_NATIVA_RAM=1 / KYTY_VRAM_MITAD=1)\n", r ? "encendido" : "apagado",
		            MitadOn() ? ", objetivo = mitad del tope efectivo" : "");
		std::fflush(stdout);
		return r;
	}();
	return on;
}

uint64_t NativaRamCap() {
	static const uint64_t cap = [] {
		const char* v = std::getenv("KYTY_TEX_NATIVA_RAM_MB");
		if (v == nullptr && MitadOn()) return uint64_t {1} << 40u; // sin limite en la prueba de la mitad
		const long mb = v != nullptr ? std::atol(v) : 2048;
		return static_cast<uint64_t>(mb < 0 ? 0 : mb) << 20u;
	}();
	return cap;
}

uint64_t ObjetivoBytes() {
	const uint64_t tope = VramTope::g_tope_bytes.load(std::memory_order_relaxed);
	return MitadOn() ? tope / 2 : tope;
}

// Tipo de memoria del sistema (heap no DEVICE_LOCAL) que admiten los bits de la imagen; UINT32_MAX: ninguno.
uint32_t TipoRam(const GraphicContext& graphics, uint32_t bits) {
	const auto& t    = graphics.physical_device_memory_properties;
	uint32_t    best = UINT32_MAX;
	for (uint32_t i = 0; i < t.memoryTypeCount; ++i) {
		if ((bits & (1u << i)) == 0) continue;
		const auto& heap = t.memoryHeaps[t.memoryTypes[i].heapIndex];
		if ((static_cast<VkMemoryHeapFlags>(heap.flags) & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) continue;
		const auto f = static_cast<VkMemoryPropertyFlags>(t.memoryTypes[i].propertyFlags);
		if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) continue;
		const bool cached_best =
		    best != UINT32_MAX &&
		    (static_cast<VkMemoryPropertyFlags>(t.memoryTypes[best].propertyFlags) & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
		if (best == UINT32_MAX || ((f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) == 0 && cached_best)) best = i;
	}
	return best;
}

void InformeRam() {
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(30)) return;
	const double secs = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count()) / 1000.0;
	last              = now;
	uint64_t ram_used = 0, vram_used = 0;
	size_t   ram_blk = 0, vram_blk = 0;
	{
		std::lock_guard lock(g_pool_mutex);
		for (const auto& b: g_blocks) {
			if (b->ram) {
				ram_blk++;
				ram_used += static_cast<uint64_t>(b->used) * kPoolPage;
			} else {
				vram_blk++;
				vram_used += static_cast<uint64_t>(b->used) * kPoolPage;
			}
		}
	}
	const uint64_t usos = r_usos.exchange(0), usos_b = r_usos_bytes.exchange(0);
	const double   mbs  = secs > 0 ? static_cast<double>(usos_b >> 20u) / secs : 0.0;
	std::printf("[tex-nativa-ram] 30 s: texturas nativas en RAM %llu MiB (%zu bloques), en VRAM %llu MiB (%zu bloques); movidos "
	            "%llu mips (%llu MiB), descartados %llu; bloques de VRAM puestos a vaciar %llu, liberados %llu (%llu MiB devueltos); usos de mips en RAM %llu (%.0f MiB/s); bufferes en RAM (BUF-RAM, de "
	            "vram-tope) %llu MiB (%llu); uso %llu / objetivo %llu MiB (tope %llu%s); RAM Vulkan del sistema: fijada por el "
	            "driver, Windows no la comprime ni la pagina (no medido)\n",
	            static_cast<unsigned long long>(ram_used >> 20u), ram_blk, static_cast<unsigned long long>(vram_used >> 20u), vram_blk,
	            static_cast<unsigned long long>(r_mov_n.exchange(0)), static_cast<unsigned long long>(r_mov_bytes.exchange(0) >> 20u),
	            static_cast<unsigned long long>(r_cancel.exchange(0)), static_cast<unsigned long long>(r_vaciando.exchange(0)),
	            static_cast<unsigned long long>(v_lib_n.exchange(0)), static_cast<unsigned long long>(v_lib_bytes.exchange(0) >> 20u),
	            static_cast<unsigned long long>(usos), mbs,
	            static_cast<unsigned long long>(VramTope::g_bufram_bytes.load() >> 20u),
	            static_cast<unsigned long long>(VramTope::g_bufram_n.load()),
	            static_cast<unsigned long long>(VramTope::g_uso_bytes.load() >> 20u), static_cast<unsigned long long>(ObjetivoBytes() >> 20u),
	            static_cast<unsigned long long>(VramTope::g_tope_bytes.load() >> 20u), MitadOn() ? ", modo mitad" : "");
	// (W17) Reparto por tipo (VRAM / RAM) y objetivo, para ver cuanto baja con la mitad.
	std::printf("[vram-mitad] VRAM: texturas %llu MiB, RT %llu MiB, bufferes %llu MiB, nativas %llu MiB | RAM: bufferes %llu MiB, "
	            "nativas %llu MiB | uso %llu / objetivo %llu MiB%s\n",
	            static_cast<unsigned long long>(VramTope::g_rep_tex.load() >> 20u), static_cast<unsigned long long>(VramTope::g_rep_rt.load() >> 20u),
	            static_cast<unsigned long long>(VramTope::g_rep_buf.load() >> 20u), static_cast<unsigned long long>(vram_used >> 20u),
	            static_cast<unsigned long long>(VramTope::g_bufram_bytes.load() >> 20u), static_cast<unsigned long long>(ram_used >> 20u),
	            static_cast<unsigned long long>(VramTope::g_uso_bytes.load() >> 20u), static_cast<unsigned long long>(ObjetivoBytes() >> 20u),
	            MitadOn() ? " (modo mitad)" : "");
	if (mbs > 256.0)
		std::printf("[tex-nativa-ram] AVISO coste: la GPU usa %.0f MiB/s de mips que estan en RAM (lectura por PCIe); si los fps "
		            "bajan, bajar KYTY_TEX_NATIVA_RAM_MB o apagarlo\n",
		            mbs);
	std::fflush(stdout);
}

// Copias de `nuevos` en el comando actual: despues de todo lo grabado antes (escrituras de subida incluidas).
void GrabarCopias(CommandScheduler& scheduler, const std::vector<Traslado>& nuevos) {
	scheduler.EndRendering();
	auto cmd = scheduler.Current().Handle();
#pragma push_macro("MemoryBarrier")
#undef MemoryBarrier
	vk::MemoryBarrier antes {};
	antes.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	antes.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1, &antes, 0, nullptr, 0,
	                    nullptr);
	for (const auto& m: nuevos) {
		vk::BufferCopy region {m.src_off, m.dst_off, m.bytes};
		cmd.copyBuffer(vk::Buffer(m.src->copy_buffer), vk::Buffer(m.dst->copy_buffer), 1, &region);
	}
	vk::MemoryBarrier despues {};
	despues.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	despues.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 1, &despues, 0, nullptr,
	                    0, nullptr);
#pragma pop_macro("MemoryBarrier")
}

// (W17) Log 20261125-125941: W16 movio 882 mips (190 MiB) a la RAM pero la VRAM no bajo (34 bloques, 501 MiB): los mips
// movidos dejaban huecos en bloques que nunca quedaban vacios y FreeEmptyBlocks no liberaba nada. Ahora se vacian bloques
// ENTEROS: el bloque de VRAM menos usado (ultimo uso mas reciente de sus mips, el mas antiguo primero; los ya marcados
// `vaciar` antes) se marca `vaciar` (TakeRange y los traslados ya no meten nada en el) y todos sus mips y su cola salen:
// los de imagenes sin uso a la RAM, los de imagenes en uso a otro bloque de VRAM con hueco (si no hay, a la RAM). Al
// volver lo retirado, el bloque queda vacio y FreeEmptyBlocks lo libera ya (vkFreeMemory, sin reserva ni espera).
// Se repite por pasadas hasta el objetivo (mitad del tope con KYTY_VRAM_MITAD, el tope efectivo sin ella).
constexpr uint32_t kCola = UINT32_MAX; // Traslado.level de la cola de mips (ligado opaco)

// Rango de `n` paginas en un bloque de RAM del tipo `tipo` (uno nuevo si hace falta). Sin g_pool_mutex tomado.
SparsePoolBlock* RangoRam(VkDevice device, uint32_t tipo, uint32_t n, uint32_t* first, bool* parar) {
	{
		std::lock_guard lock(g_pool_mutex);
		SparsePoolBlock* dst = nullptr;
		uint32_t         f0  = 0;
		for (auto& b: g_blocks) {
			if (!b->ram || b->type != tipo || kPoolPages - b->used < n) continue;
			if (dst != nullptr && b->used <= dst->used) continue;
			const uint32_t f = FindRun(*b, n);
			if (f == UINT32_MAX || !BloqueBufer(device, *b)) continue;
			dst = b.get();
			f0  = f;
		}
		if (dst != nullptr) {
			Carve(*dst, f0, n);
			*first = f0;
			return dst;
		}
	}
	VkMemoryAllocateInfo allocate {};
	allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.allocationSize  = kPoolBlockBytes;
	allocate.memoryTypeIndex = tipo;
	VkDeviceMemory mem       = VK_NULL_HANDLE;
	{
		VramDrv::Scope vram_drv(VramDrv::CatSparseImg);
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(device, &allocate, nullptr, &mem) != VK_SUCCESS) mem = VK_NULL_HANDLE;
	}
	if (mem == VK_NULL_HANDLE) {
		*parar = true;
		return nullptr;
	}
	auto b    = std::make_unique<SparsePoolBlock>();
	b->memory = mem;
	b->type   = tipo;
	b->ram    = true;
	b->page.assign(kPoolPages, 0);
	b->empty_since = std::chrono::steady_clock::now();
	std::lock_guard lock(g_pool_mutex);
	g_ram_free_bytes += kPoolBlockBytes;
	if (!BloqueBufer(device, *b)) {
		static std::atomic<bool> once {false};
		if (!once.exchange(true)) {
			std::printf("[tex-nativa-ram] el tipo de memoria %u no admite bufer de copia: no se mueve nada\n", tipo);
			std::fflush(stdout);
		}
		g_blocks.push_back(std::move(b)); // vacio: FreeEmptyBlocks lo libera
		*parar = true;
		return nullptr;
	}
	Carve(*b, 0, n);
	*first   = 0;
	auto* rb = b.get();
	g_blocks.push_back(std::move(b));
	return rb;
}

// Una tanda de vaciado de bloques de VRAM. True: se encolo algo; false: no hace falta o no hay que mover.
bool NativaRamPasada(GraphicContext& graphics, CommandScheduler& scheduler, const std::vector<VulkanImage*>& sin_uso) {
	if (!NativaRamOn()) return false;
	InformeRam();
	// (W17) Texturas negras en la prueba 52: la copia en crudo de mips sparse no es segura. Sin traslados de nativas
	// salvo KYTY_TEX_NATIVA_RAM_CRUDO=1 (prueba); la mitad la hacen los bufferes (BUF-RAM sin limite en modo mitad).
	static const bool crudo = [] {
		const char* v = std::getenv("KYTY_TEX_NATIVA_RAM_CRUDO");
		return v != nullptr && v[0] == '1';
	}();
	if (!crudo) return false;
	const auto device = static_cast<VkDevice>(graphics.device);
	FreeEmptyBlocks(device, false); // (W17) los bloques ya vaciados, fuera (vkFreeMemory)
	const uint64_t uso      = VramTope::g_uso_bytes.load(std::memory_order_relaxed);
	const uint64_t objetivo = ObjetivoBytes();
	if (objetivo == 0 || uso <= objetivo) return false; // bajo el objetivo: nada
	// El uso se mide cada pocos segundos: lo vaciado desde la ultima medida cuenta ya como bajado.
	static uint64_t uso_visto = 0, movido_desde = 0;
	if (uso != uso_visto) {
		uso_visto    = uso;
		movido_desde = 0;
	}
	const uint64_t falta = uso - objetivo;
	if (movido_desde >= falta || r_total_bytes.load() >= NativaRamCap()) return false;
	std::vector<const SparseImageMemory*> libres; // imagenes sin uso (ordenadas, para buscar)
	for (auto* vi: sin_uso)
		if (vi != nullptr && vi->sparse != nullptr) libres.push_back(vi->sparse);
	std::sort(libres.begin(), libres.end());
	struct Ocup {
		SparseImageMemory* s;
		uint32_t           level;
		uint64_t           off;
		uint64_t           bytes;
	};
	// Bajo g_todas_mutex toda la pasada: DestroySparseImage no puede borrar una imagen hasta que su `moving` este puesto.
	std::lock_guard todas_lock(g_todas_mutex);
	std::unordered_map<SparsePoolBlock*, std::vector<Ocup>> por_bloque;
	std::unordered_map<SparsePoolBlock*, uint64_t>          ultimo; // uso mas reciente de lo que hay en el bloque
	std::vector<SparsePoolBlock*>                           ocupado; // con mips de imagenes en traslado: no se elige
	for (auto* s: g_todas) {
		const bool moviendo = s->moving.load(std::memory_order_relaxed) != 0;
		const uint64_t u    = s->ultimo_uso_ms.load(std::memory_order_relaxed);
		auto anota = [&](SparsePoolBlock* b, uint32_t level, uint64_t off, uint64_t bytes) {
			if (b == nullptr || b->ram) return;
			if (moviendo) {
				ocupado.push_back(b);
				return;
			}
			por_bloque[b].push_back({s, level, off, bytes});
			auto& x = ultimo[b];
			x       = std::max(x, u);
		};
		for (uint32_t level = 0; level < s->level_block.size() && level < s->level_bytes.size(); level++)
			if (s->level_memory[level] != VK_NULL_HANDLE) anota(s->level_block[level], level, s->level_offset[level], s->level_bytes[level]);
		if (s->tail_memory != VK_NULL_HANDLE) anota(s->tail_block, kCola, s->tail_mem_offset, s->tail_size);
	}
	std::sort(ocupado.begin(), ocupado.end());
	std::vector<SparsePoolBlock*> victimas;
	{
		std::lock_guard lock(g_pool_mutex);
		for (auto& b: g_blocks) {
			if (b->ram || b->used == 0 || b->no_movible || por_bloque.find(b.get()) == por_bloque.end()) continue;
			if (std::binary_search(ocupado.begin(), ocupado.end(), b.get())) continue;
			victimas.push_back(b.get());
		}
		std::sort(victimas.begin(), victimas.end(), [&](SparsePoolBlock* x, SparsePoolBlock* y) {
			if (x->vaciar != y->vaciar) return x->vaciar; // seguir con los empezados
			return ultimo[x] < ultimo[y];
		});
	}
	if (victimas.empty()) return false;
	const uint64_t        MaxPorPasada = (MitadOn() ? uint64_t {96} : uint64_t {32}) << 20u;
	const uint64_t        limite = std::min<uint64_t>({MaxPorPasada, falta - movido_desde + kPoolBlockBytes, NativaRamCap() - r_total_bytes.load()});
	std::vector<Traslado> nuevos;
	uint64_t              total = 0, bloques = 0;
	bool                  parar = false;
	for (auto* src: victimas) {
		if (total >= limite || parar) break;
		{
			std::lock_guard lock(g_pool_mutex);
			if (!BloqueBufer(device, *src)) continue;
			src->vaciar = true;
		}
		bloques++;
		for (const auto& o: por_bloque[src]) {
			const uint32_t n      = static_cast<uint32_t>((o.bytes + kPoolPage - 1) / kPoolPage);
			const bool     en_uso = !std::binary_search(libres.begin(), libres.end(), static_cast<const SparseImageMemory*>(o.s));
			const uint32_t tipo   = TipoRam(graphics, o.s->memory_bits);
			SparsePoolBlock* dst  = nullptr;
			uint32_t         first = 0;
			if (en_uso || tipo == UINT32_MAX) { // a otro bloque de VRAM con hueco (el mas lleno)
				std::lock_guard lock(g_pool_mutex);
				for (auto& b: g_blocks) {
					if (b.get() == src || b->ram || b->vaciar || b->type != src->type || kPoolPages - b->used < n) continue;
					if (dst != nullptr && b->used <= dst->used) continue;
					const uint32_t f = FindRun(*b, n);
					if (f == UINT32_MAX || !BloqueBufer(device, *b)) continue;
					dst   = b.get();
					first = f;
				}
				if (dst != nullptr) Carve(*dst, first, n);
			}
			if (dst == nullptr && tipo != UINT32_MAX && !parar) dst = RangoRam(device, tipo, n, &first, &parar);
			if (dst == nullptr) {
				if (tipo == UINT32_MAX) {
					static std::atomic<bool> once {false};
					if (!once.exchange(true)) {
						std::printf("[tex-nativa-ram] la imagen sparse no admite memoria del sistema (memoryTypeBits 0x%x)\n",
						            o.s->memory_bits);
						std::fflush(stdout);
					}
				}
				continue; // se queda: el bloque se vaciara cuando se suelte
			}
			Traslado m {o.s, o.level, src, o.off, dst, static_cast<uint64_t>(first) * kPoolPage, o.bytes, 0, &scheduler};
			m.a_ram = dst->ram;
			nuevos.push_back(m);
		}
		total += kPoolBlockBytes;
	}
	if (nuevos.empty()) return false;
	static std::atomic<bool> tipo_dicho {false};
	for (const auto& m: nuevos) {
		if (!m.a_ram || tipo_dicho.exchange(true)) continue;
		const auto&    t    = graphics.physical_device_memory_properties;
		const uint32_t tipo = m.dst->type;
		std::printf("[tex-nativa-ram] memoryTypeBits de la imagen 0x%x; tipo de RAM %u (flags 0x%x, heap %u flags 0x%x)\n",
		            m.s->memory_bits, tipo, static_cast<unsigned>(t.memoryTypes[tipo].propertyFlags), t.memoryTypes[tipo].heapIndex,
		            static_cast<unsigned>(t.memoryHeaps[t.memoryTypes[tipo].heapIndex].flags));
		std::fflush(stdout);
		break;
	}
	GrabarCopias(scheduler, nuevos);
	movido_desde += total;
	r_vaciando.fetch_add(bloques, std::memory_order_relaxed);
	const uint64_t  tick = scheduler.CurrentTick();
	std::lock_guard move_lock(g_move_mutex);
	for (auto& m: nuevos) {
		m.tick = tick;
		m.s->moving.fetch_add(1, std::memory_order_relaxed);
		g_moves.push_back(m);
	}
	return true;
}

} // namespace

void SparseMoverCancelar(const VulkanImage& image) {
	auto* s = image.sparse;
	if (s == nullptr) return;
	if (s->moving.load(std::memory_order_relaxed) != 0) s->move_cancel.store(true, std::memory_order_relaxed);
	// (W16) Marca de uso (orden de TEX-NATIVA-RAM) y coste: usos de imagenes con mips en la RAM (la GPU lee por PCIe).
	s->ultimo_uso_ms.store(NowMs(), std::memory_order_relaxed);
	s->usos.fetch_add(1, std::memory_order_relaxed);
	if (NativaRamOn() && s->per_level) {
		uint64_t en_ram = 0;
		for (size_t l = 0; l < s->level_block.size(); l++)
			if (s->level_block[l] != nullptr && s->level_block[l]->ram && l < s->level_bytes.size()) en_ram += s->level_bytes[l];
		if (en_ram != 0) {
			r_usos.fetch_add(1, std::memory_order_relaxed);
			r_usos_bytes.fetch_add(en_ram, std::memory_order_relaxed);
		}
	}
}

void SparseCompactar(GraphicContext& graphics, CommandScheduler& scheduler, const std::vector<VulkanImage*>& sin_uso) {
	if (!CompactarOn() || !BlockPoolOn()) {
		if (NativaRamOn()) { // (W17) la linea [vram-mitad] / [tex-nativa-ram] sale igual
			InformeRam();
			FreeEmptyBlocks(static_cast<VkDevice>(graphics.device), false);
		}
		return;
	}
	InformeCompactar();
	const auto device = static_cast<VkDevice>(graphics.device);
	c_pasadas.fetch_add(1, std::memory_order_relaxed);
	// 1) Traslados con la copia acabada: ligado nuevo o descarte.
	{
		std::lock_guard move_lock(g_move_mutex);
		std::erase_if(g_moves, [&](const Traslado& m) {
			if (!m.scheduler->IsFree(m.tick)) return false;
			auto* s = m.s;
			const bool cola = m.level == kCola; // (W17)
			const bool vale =
			    s != nullptr && !s->move_cancel.load(std::memory_order_relaxed) &&
			    (cola ? (s->tail_block == m.src && s->tail_mem_offset == m.src_off && s->tail_memory == m.src->memory)
			          : (m.level < s->level_block.size() && s->level_block[m.level] == m.src && s->level_offset[m.level] == m.src_off &&
			             s->level_memory[m.level] == m.src->memory));
			if (!vale) {
				DevolverDestino(device, m);
				c_mov_cancel.fetch_add(1, std::memory_order_relaxed);
					if (m.a_ram) r_cancel.fetch_add(1, std::memory_order_relaxed);
			} else {
				bool encolado = false;
				{
					std::lock_guard pending_lock(g_pending_mutex);
					if (cola && (s->pending_scheduler == nullptr || s->pending_scheduler == m.scheduler)) { // (W17) cola: ligado opaco
						auto it = std::find_if(s->pending_opaque.begin(), s->pending_opaque.end(),
						                       [&](const VkSparseMemoryBind& b) { return b.resourceOffset == s->tail_offset; });
						if (it == s->pending_opaque.end()) {
							if (s->pending_scheduler == nullptr) {
								s->pending_scheduler = m.scheduler;
								g_pending.push_back(s);
							}
							s->pending_opaque.push_back({s->tail_offset, s->tail_size, m.dst->memory, m.dst_off, 0});
							if (HiloOn()) g_enqueued_seq++;
							encolado = true;
						}
					} else if (s->pending_scheduler == nullptr || s->pending_scheduler == m.scheduler) {
						VkSparseImageMemoryBind item {};
						item.subresource  = {VK_IMAGE_ASPECT_COLOR_BIT, m.level, 0};
						item.offset       = {0, 0, 0};
						item.extent       = {std::max(s->extent.width >> m.level, 1u), std::max(s->extent.height >> m.level, 1u), 1u};
						item.memory       = m.dst->memory;
						item.memoryOffset = m.dst_off;
						auto it = std::find_if(s->pending_image.begin(), s->pending_image.end(),
						                       [&](const VkSparseImageMemoryBind& b) { return b.subresource.mipLevel == m.level; });
						if (it == s->pending_image.end()) {
							if (s->pending_scheduler == nullptr) {
								s->pending_scheduler = m.scheduler;
								g_pending.push_back(s);
							}
							s->pending_image.push_back(item);
							if (HiloOn()) g_enqueued_seq++;
							encolado = true;
						}
					}
				}
				if (!encolado) {
					DevolverDestino(device, m);
					c_mov_cancel.fetch_add(1, std::memory_order_relaxed);
					if (m.a_ram) r_cancel.fetch_add(1, std::memory_order_relaxed);
				} else {
					{
						std::lock_guard live_lock(g_live_mutex);
						s->retired.push_back({m.src->memory, m.bytes, m.scheduler->CurrentTick(), cola ? s->tail_type : s->level_type[m.level],
						                      m.src, m.src_off});
						s->retired_scheduler = m.scheduler;
						g_retired_bytes.fetch_add(m.bytes, std::memory_order_relaxed);
						if (!s->in_live) {
							s->in_live = true;
							g_live.push_back(s);
						}
					}
					if (cola) {
						s->tail_memory     = m.dst->memory;
						s->tail_block      = m.dst;
						s->tail_mem_offset = m.dst_off;
						s->tail_type       = m.dst->type;
					} else {
						s->level_memory[m.level] = m.dst->memory;
						s->level_block[m.level]  = m.dst;
						s->level_offset[m.level] = m.dst_off;
						s->level_type[m.level]   = m.dst->type;
					}
					if (m.dst->ram && !m.src->ram) { // (W17) contabilidad: ya no es VRAM
						s->ram_bytes += m.bytes;
						TexNativa::c_bound_bytes.fetch_sub(m.bytes, std::memory_order_relaxed);
					}
					c_mov_n.fetch_add(1, std::memory_order_relaxed);
					c_mov_bytes.fetch_add(m.bytes, std::memory_order_relaxed);
					if (m.a_ram) {
						r_mov_n.fetch_add(1, std::memory_order_relaxed);
						r_mov_bytes.fetch_add(m.bytes, std::memory_order_relaxed);
						r_total_bytes.fetch_add(m.bytes, std::memory_order_relaxed);
					}
					if (HiloOn()) {
						{
							std::lock_guard h(g_hilo_mutex);
							g_hilo_urgent = true;
						}
						g_hilo_cv.notify_one();
					}
				}
			}
			if (s != nullptr && s->moving.fetch_sub(1, std::memory_order_relaxed) == 1)
				s->move_cancel.store(false, std::memory_order_relaxed);
			return true;
		});
		if (!g_moves.empty()) return; // una tanda cada vez
	}
	// (W16) Sobre el objetivo: lo menos usado a la RAM del sistema; si no hace falta, compactar como antes.
	if (NativaRamPasada(graphics, scheduler, sin_uso)) return;
	// 2) Nueva tanda: el bloque menos lleno, si sus mips caben en bloques mas llenos.
	constexpr uint64_t MaxPorPasada = uint64_t {32} << 20u;
	struct Mip {
		SparseImageMemory* s;
		uint32_t           level;
	};
	std::vector<Traslado> nuevos;
	{
		std::lock_guard lock(g_pool_mutex);
		if (g_blk_free_bytes < 2 * kPoolBlockBytes) return;
		// Bloques de origen: los de como mucho media ocupacion, de menos a mas llenos; el primero con mips de imagenes sin
		// uso. (Los mips de imagenes en uso se quedan; al soltarse, el bloque se vacia y FreeEmptyBlocks lo devuelve.)
		std::vector<SparsePoolBlock*> origen;
		for (auto& b: g_blocks)
			if (b->used != 0 && b->used <= kPoolPages / 2 && !b->no_movible && !b->ram && !b->vaciar) origen.push_back(b.get());
		std::sort(origen.begin(), origen.end(), [](const SparsePoolBlock* x, const SparsePoolBlock* y) { return x->used < y->used; });
		SparsePoolBlock* src = nullptr;
		std::vector<Mip> mips;
		for (auto* cand: origen) {
			for (auto* vi: sin_uso) {
				auto* s = vi != nullptr ? vi->sparse : nullptr;
				if (s == nullptr || !s->per_level || s->moving.load(std::memory_order_relaxed) != 0) continue;
				for (uint32_t level = 0; level < s->level_block.size(); level++)
					if (s->level_block[level] == cand && s->level_memory[level] != VK_NULL_HANDLE) mips.push_back({s, level});
			}
			if (!mips.empty() && BloqueBufer(device, *cand)) {
				src = cand;
				break;
			}
			mips.clear();
		}
		if (src == nullptr) return;
		uint64_t total = 0;
		for (const auto& [s, level]: mips) {
			const uint64_t bytes = s->level_bytes[level];
			const uint32_t n     = static_cast<uint32_t>((bytes + kPoolPage - 1) / kPoolPage);
			SparsePoolBlock* best = nullptr;
			uint32_t         best_first = 0;
			for (auto& b: g_blocks) {
				if (b.get() == src || b->ram || b->vaciar || b->type != src->type || b->used < src->used || kPoolPages - b->used < n) continue;
				if (best != nullptr && b->used <= best->used) continue;
				const uint32_t first = FindRun(*b, n);
				if (first == UINT32_MAX) continue;
				best       = b.get();
				best_first = first;
			}
			if (best == nullptr || !BloqueBufer(device, *best)) break;
			Carve(*best, best_first, n);
			nuevos.push_back({s, level, src, s->level_offset[level], best, static_cast<uint64_t>(best_first) * kPoolPage, bytes,
			                  0, &scheduler});
			total += bytes;
			if (total >= MaxPorPasada) break;
		}
	}
	if (nuevos.empty()) return;
	GrabarCopias(scheduler, nuevos);
	const uint64_t tick = scheduler.CurrentTick();
	std::lock_guard move_lock(g_move_mutex);
	for (auto& m: nuevos) {
		m.tick = tick;
		m.s->moving.fetch_add(1, std::memory_order_relaxed);
		g_moves.push_back(m);
	}
}

} // namespace Libs::Graphics
