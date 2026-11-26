#pragma once
// (VRAM-EMU, 11-25) KYTY_VRAM_EMU=1 (por defecto; =0 lo apaga: no cuenta nada, todo como antes).
// Log 20261125-202747 (cmd 55): "images and other device-local" 3547 MiB, de ellos 1253 MiB que no son imagenes
// registradas del juego, sin desglose. Aqui cada asignacion de VMA que hace el emulador (vmaCreateBuffer/vmaCreateImage
// fuera del pool de bufferes del juego, y GraphicContext::CreateImage) se apunta con su dueno y su tamano real
// (vmaGetAllocationInfo), solo si es memoria de video (DEVICE_LOCAL). Linea [vram-emu] cada 30 s (vma.cpp, junto a
// "Video memory by category"): MiB por dueno y lo que queda sin dueno. No cambia ninguna asignacion.
//
// Uso en una unidad de compilacion: despues de TODOS sus #include,
//     #define VRAM_EMU_OWNER ::VramEmu::kFsr4
//     #include "local/vram-emu.h"
// y sus vmaCreateBuffer/vmaCreateImage/vmaDestroyBuffer/vmaDestroyImage/vmaFreeMemory y graphics.CreateImage(...) pasan
// por aqui (macros). vma.cpp lo incluye sin VRAM_EMU_OWNER (solo la API).
#include "graphics/host_gpu/graphicContext.h" // (antes de las macros: su declaracion de CreateImage queda intacta)

#include <vk_mem_alloc.h>

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace VramEmu {

enum Owner : int {
	kCacheImg = 0, // GraphicContext::CreateImage desde la cache de texturas (imagenes del juego y auxiliares de la cache)
	kFsr4,         // FSR 4: scratch del modelo, constantes, salida rgba16f
	kFg,           // frame-gen.cpp / fsr3-fg.cpp / frame-gen-hudless.cpp: entradas de FG, HUD, nitidez, efectos
	kTiler,        // scratch de tile/detile (en uso + guardado en su pool)
	kAnillos,      // StreamBuffer en VRAM (MemoryUsage::Stream: anillos de subida/stream, GDS)
	kBufJuegoFuera,// bufferes del juego (con direccion del juego) fuera del pool de bufferes (no cupo / sin BDA)
	kBufEmu,       // bufferes DeviceLocal del emulador sin direccion del juego (tablas, scratch de copias, anillo VRAM-BUF)
	kBufCache,     // vmaCreateBuffer propios de bufferCache.cpp (mallas nativas, staging)
	kCompute,      // renderCompute.cpp (snapshots, imagen propia)
	kOtro,         // cualquier otra unidad con la macro
	kCount
};

inline const char* Name(int owner) {
	static const char* const names[kCount] = {"cache de texturas (CreateImage)", "FSR4", "FG/efectos", "tiler",
	                                          "anillos stream", "bufferes del juego fuera del pool", "bufferes del emulador",
	                                          "bufferCache propios", "renderCompute", "otros"};
	return owner >= 0 && owner < kCount ? names[owner] : "?";
}

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_EMU");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

struct Entry {
	int      owner;
	uint64_t bytes;
};

inline std::atomic<int64_t>  g_bytes[kCount] {};
inline std::atomic<int64_t>  g_count[kCount] {};
inline std::mutex            g_mutex;
inline std::unordered_map<VmaAllocation, Entry>* g_map = nullptr;
inline thread_local int      t_next = -1; // dueno de la PROXIMA creacion de este hilo (SetNext), -1: el de la unidad

// Para la macro de CreateImage (operador coma): fija el dueno de la siguiente creacion de este hilo.
inline int SetNext(int owner) {
	t_next = owner;
	return owner;
}
inline int TakeNext(int fallback) {
	const int o = t_next;
	t_next      = -1;
	return o >= 0 ? o : fallback;
}

inline void Untrack(VmaAllocation allocation) {
	if (!On() || allocation == nullptr) return;
	std::lock_guard lock(g_mutex);
	if (g_map == nullptr) return;
	const auto it = g_map->find(allocation);
	if (it == g_map->end()) return;
	g_bytes[it->second.owner].fetch_sub(static_cast<int64_t>(it->second.bytes), std::memory_order_relaxed);
	g_count[it->second.owner].fetch_sub(1, std::memory_order_relaxed);
	g_map->erase(it);
}

// Solo memoria de video (tipo DEVICE_LOCAL): la de sistema no quita VRAM.
inline void Track(VmaAllocator allocator, VmaAllocation allocation, int owner) {
	if (!On() || allocator == nullptr || allocation == nullptr || owner < 0 || owner >= kCount) return;
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, allocation, &info);
	VkMemoryPropertyFlags flags = 0;
	vmaGetMemoryTypeProperties(allocator, info.memoryType, &flags);
	std::lock_guard lock(g_mutex);
	if (g_map == nullptr) g_map = new std::unordered_map<VmaAllocation, Entry>();
	// (VMA reutiliza los objetos VmaAllocation: una entrada vieja con la misma direccion se quita antes.)
	if (const auto it = g_map->find(allocation); it != g_map->end()) {
		g_bytes[it->second.owner].fetch_sub(static_cast<int64_t>(it->second.bytes), std::memory_order_relaxed);
		g_count[it->second.owner].fetch_sub(1, std::memory_order_relaxed);
		g_map->erase(it);
	}
	if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) return;
	(*g_map)[allocation] = {owner, static_cast<uint64_t>(info.size)};
	g_bytes[owner].fetch_add(static_cast<int64_t>(info.size), std::memory_order_relaxed);
	g_count[owner].fetch_add(1, std::memory_order_relaxed);
}

inline uint64_t Bytes(int owner) {
	const int64_t v = g_bytes[owner].load(std::memory_order_relaxed);
	return v > 0 ? static_cast<uint64_t>(v) : 0;
}

// Envolturas de VMA (las macros de abajo llevan aqui). Una asignacion de un pool propio (el de bufferes del juego) no se
// apunta: ese pool sale aparte en "Video memory by category".
inline VkResult EmuCreateBuffer(int owner, VmaAllocator allocator, const VkBufferCreateInfo* buffer_info,
                             const VmaAllocationCreateInfo* allocation_info, VkBuffer* buffer, VmaAllocation* allocation,
                             VmaAllocationInfo* result_info) {
	const VkResult r = ::vmaCreateBuffer(allocator, buffer_info, allocation_info, buffer, allocation, result_info);
	if (r == VK_SUCCESS) {
		const int o = TakeNext(owner);
		if (allocation != nullptr && (allocation_info == nullptr || allocation_info->pool == VK_NULL_HANDLE))
			Track(allocator, *allocation, o);
	}
	return r;
}
inline VkResult EmuCreateImage(int owner, VmaAllocator allocator, const VkImageCreateInfo* image_info,
                            const VmaAllocationCreateInfo* allocation_info, VkImage* image, VmaAllocation* allocation,
                            VmaAllocationInfo* result_info) {
	const VkResult r = ::vmaCreateImage(allocator, image_info, allocation_info, image, allocation, result_info);
	if (r == VK_SUCCESS) {
		const int o = TakeNext(owner);
		if (allocation != nullptr && (allocation_info == nullptr || allocation_info->pool == VK_NULL_HANDLE))
			Track(allocator, *allocation, o);
	}
	return r;
}
inline void EmuDestroyBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
	Untrack(allocation);
	::vmaDestroyBuffer(allocator, buffer, allocation);
}
inline void EmuDestroyImage(VmaAllocator allocator, VkImage image, VmaAllocation allocation) {
	Untrack(allocation);
	::vmaDestroyImage(allocator, image, allocation);
}
inline void EmuFreeMemory(VmaAllocator allocator, VmaAllocation allocation) {
	Untrack(allocation);
	::vmaFreeMemory(allocator, allocation);
}

// La linea [vram-emu] (vma.cpp, cada 30 s). other_local: "images and other device-local" de VMA (sin el pool de bufferes
// del juego ni el pool de imagenes); image_pool: el pool de imagenes; tiler_pool: lo guardado en el pool del tiler (ya
// dentro de "tiler"); free_pool / free_rest: huecos libres dentro de los bloques (pool de bufferes / el resto).
inline void Report(uint64_t other_local, uint64_t image_pool, uint64_t tiler_pool, uint64_t free_pool, uint64_t free_rest) {
	if (!On()) return;
	uint64_t tracked = 0;
	for (int i = 0; i < kCount; i++) tracked += Bytes(i);
	const uint64_t rest = other_local > tracked ? other_local - tracked : 0;
	const auto     mib  = [](uint64_t v) { return static_cast<unsigned long long>(v >> 20u); };
	std::printf("[vram-emu] 30 s, MiB: imagenes y otros de VRAM %llu = %s %llu (%lld) + %s %llu (%lld) + %s %llu (%lld) + %s "
	            "%llu (%lld; de ello en su pool %llu) + %s %llu (%lld) + %s %llu (%lld) + %s %llu (%lld) + %s %llu (%lld) + %s "
	            "%llu (%lld) + %s %llu (%lld) + sin dueno %llu; aparte: pool de imagenes %llu, huecos libres en bloques: pool "
	            "de bufferes %llu, resto %llu (KYTY_VRAM_EMU=0 la apaga)\n",
	            mib(other_local), Name(kCacheImg), mib(Bytes(kCacheImg)), static_cast<long long>(g_count[kCacheImg].load()),
	            Name(kFsr4), mib(Bytes(kFsr4)), static_cast<long long>(g_count[kFsr4].load()), Name(kFg), mib(Bytes(kFg)),
	            static_cast<long long>(g_count[kFg].load()), Name(kTiler), mib(Bytes(kTiler)),
	            static_cast<long long>(g_count[kTiler].load()), mib(tiler_pool), Name(kAnillos), mib(Bytes(kAnillos)),
	            static_cast<long long>(g_count[kAnillos].load()), Name(kBufJuegoFuera), mib(Bytes(kBufJuegoFuera)),
	            static_cast<long long>(g_count[kBufJuegoFuera].load()), Name(kBufEmu), mib(Bytes(kBufEmu)),
	            static_cast<long long>(g_count[kBufEmu].load()), Name(kBufCache), mib(Bytes(kBufCache)),
	            static_cast<long long>(g_count[kBufCache].load()), Name(kCompute), mib(Bytes(kCompute)),
	            static_cast<long long>(g_count[kCompute].load()), Name(kOtro), mib(Bytes(kOtro)),
	            static_cast<long long>(g_count[kOtro].load()), mib(rest), mib(image_pool), mib(free_pool), mib(free_rest));
	std::fflush(stdout);
}

} // namespace VramEmu

// Macros por unidad (ver arriba). vma.cpp no define VRAM_EMU_OWNER: alli se llama a Track/Untrack a mano.
#if defined(VRAM_EMU_OWNER) && !defined(VMA_IMPLEMENTATION) && !defined(KYTY_VRAM_EMU_MACROS)
#define KYTY_VRAM_EMU_MACROS 1
#define vmaCreateBuffer(a, b, c, d, e, f) ::VramEmu::EmuCreateBuffer(VRAM_EMU_OWNER, a, b, c, d, e, f)
#define vmaCreateImage(a, b, c, d, e, f)  ::VramEmu::EmuCreateImage(VRAM_EMU_OWNER, a, b, c, d, e, f)
#define vmaDestroyBuffer(a, b, c)         ::VramEmu::EmuDestroyBuffer(a, b, c)
#define vmaDestroyImage(a, b, c)          ::VramEmu::EmuDestroyImage(a, b, c)
#define vmaFreeMemory(a, b)               ::VramEmu::EmuFreeMemory(a, b)
// graphics.CreateImage(info, image): el dueno de esta unidad para la imagen (GraphicContext::CreateImage lo toma).
#define CreateImage(info, image) CreateImage((static_cast<void>(::VramEmu::SetNext(VRAM_EMU_OWNER)), (info)), image)
#endif
