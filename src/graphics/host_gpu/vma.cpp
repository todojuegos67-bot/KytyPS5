#include "graphics/host_gpu/vulkanCommon.h"
#include "local/mem-guard.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/sparseImage.h" // (ram1b)
#include "live-counters.h"
#include "local-platform.h"
#include "soft-release.h"
#include "ram-guard.h"
#include "vram-drv.h" // (VRAM-DRV: vkAllocateMemory por categoria)
#include "vram-store.h" // (mem1 etapa 2: the store's video memory added to the caches' cap)
#include "vram-recursos.h" // (RECURSOS-JUSTOS)
#include "gestor-memoria.h" // (GESTOR-MEM) arbitro unico de RAM y VRAM
#include "local/vram-emu.h" // (VRAM-EMU) VRAM del emulador por dueno, linea [vram-emu]
#if __has_include("tex-stream.h")
#include "tex-stream.h" // (m2) limite de mips del juego
#define KYTY_VMA_TEXSTREAM() TexStream::On()
#else
#define KYTY_VMA_TEXSTREAM() false
#endif

#include <cstdio>
#include "kytyGitVersion.h"

#include <algorithm>
#include <cinttypes>

namespace Libs::Graphics {

void FlushBufferReclaimer(); // streamBuffer.cpp (KYTY_BUFFER_RECLAIM)
uint64_t TilerTrimIdle();     // tiler.cpp (VRAM-EMU, KYTY_EMU_TILER_POOL): scratch guardado sin uso, liberado
uint64_t TilerPoolBytes();    // tiler.cpp: lo guardado ahora en su pool
void VramStoreRegister(GraphicContext& graphics); // vramStore.cpp (mem1 etapa 2, KYTY_MEM1_STAGE2)

// Local diagnostic (live `vma <path>`): VMA is thread-safe, so the live thread writes it.
static VmaAllocator          g_report_allocator = nullptr;
static const GraphicContext* g_report_context   = nullptr;
// Freed images kept for reuse (KYTY_IMAGE_POOL): at most 1 GiB, and a 16th of the GPU's memory budget.
static uint64_t g_image_pool_limit = 1024ull << 20;
// (vram-budget) KYTY_SIMULATE_VRAM_MB's size, bytes (0: no simulation): the budget is that card's whatever the ballast got.
static uint64_t g_simulated_vram = 0;
// (vram-budget) What the driver of a card with `bytes` of video memory leaves the game: Windows and the desktop keep ~10%
// (an 8 GB RX 6650 XT's budget 7378 MiB at start-up, issue #276): 8192 -> 7373 MiB.
static uint64_t SimulatedDriverBudget(uint64_t bytes) { return bytes - bytes / 10; }
// (VRAM8-SIN-LASTRE, 11-24) Size of the real largest device-local heap, bytes (for the simulated budget below).
static uint64_t g_real_vram_heap = 0;
// (VRAM8-SIN-LASTRE) The driver budget an 8 GB (KYTY_SIMULATE_VRAM_MB) card would report: its ~90% less what the rest of
// the system uses now. What the rest uses is read from the real card: its ~90% less the real budget (VK_EXT_memory_budget).
// No ballast: the old 7785 MiB ballast was charged by NVIDIA as process private commit (~7.8 GB) and pushed ~2.6 GB to
// system RAM (log 20261124-130247: 240 MiB free), which a real 8 GB card does not do.
// (VRAM-REAL, 11-24; KYTY_VRAM_REAL=0 lo apaga) Lo que el resto de procesos ocupa de verdad en la VRAM de la tarjeta
// (contador de Windows "GPU Adapter Memory\Dedicated Usage" menos el uso DXGI de este proceso, cada 2 s). Log
// 20261124-201421: el proceso 6480 MiB (DXGI, lo mismo que cuenta [vram-pres]), presupuesto 7116 = 7372 - 256 de "otros"
// sacados del hueco del presupuesto de NVIDIA, que en una tarjeta de 16 GB apenas baja por el escritorio y el navegador:
// el total de la tarjeta pasaba de 8 GB. Una de 8 GB de verdad no los tiene: ahora el presupuesto descuenta lo medido.
// Encendido con la simulacion de 8 GB o con tarjetas reales de 10 GB o menos.
static bool g_vram_real_small = false;
static bool VramRealOn() {
	static const bool on = [] { const char* v = std::getenv("KYTY_VRAM_REAL"); return v == nullptr || *v != '0'; }();
	return on && (g_simulated_vram != 0 || g_vram_real_small);
}
static uint64_t MeasuredOthers() {
	uint64_t all = 0, others = 0;
	if (!VramRealOn() || !LocalPlatform::QueryAdapterDedicatedUsage(&all, &others)) return 0;
	return others;
}
static uint64_t SimulatedBudgetFromReal(uint64_t real_budget) {
	const uint64_t cap = SimulatedDriverBudget(g_simulated_vram);
	if (g_real_vram_heap == 0 || real_budget == 0) return cap - std::min(cap, MeasuredOthers());
	const uint64_t real_cap = SimulatedDriverBudget(g_real_vram_heap);
	const uint64_t others   = std::max<uint64_t>(real_cap > real_budget ? real_cap - real_budget : 0, MeasuredOthers());
	// (11-25 noche, usuario) Tarjeta de 8 GB: el tope es 6,3 GB fijo (+128 MiB del margen del gestor), no menos.
	const uint64_t minimo   = (uint64_t {6451} + 128) << 20u;
	return std::max(cap - std::min(cap, others), std::min(cap, minimo));
}
// (VRAM-REAL) Real card of 10 GB or less, no simulation: its ~90% less what the other processes really hold.
static uint64_t RealSmallCardBudget(uint64_t budget) {
	if (g_simulated_vram != 0 || !g_vram_real_small || g_real_vram_heap == 0 || !VramRealOn()) return budget;
	const uint64_t cap = SimulatedDriverBudget(g_real_vram_heap);
	return std::min(budget, cap - std::min(cap, MeasuredOthers()));
}
static std::mutex g_image_pool_mutex;
static uint64_t   g_image_pool_bytes = 0;
static uint64_t   FlushImagePool(uint64_t budget, uint64_t* held_us);
static int        g_image_pool_releaser = 0; // MemGuard's id
static void       DestroyImagePool(VmaAllocator allocator);
static void WriteVmaReport(const char* path) {
	if (g_report_allocator == nullptr) return;
	if (g_report_context != nullptr) g_report_context->LogVideoMemory("live");
	char* json = nullptr;
	vmaBuildStatsString(g_report_allocator, &json, VK_TRUE);
	if (FILE* file = std::fopen(path, "wb"); file != nullptr) {
		std::fputs(json, file);
		std::fclose(file);
	}
	vmaFreeStatsString(g_report_allocator, json);
}

// Tagged commit of Vulkan memory (LocalPlatform::MemoryTagAdd): every vkAllocateMemory (VMA's, the ballast's, the
// stream buffers') with the process's private commit before and after it, serialized so that the delta is its own
// (other threads' allocations meanwhile still add noise). NVIDIA's Windows driver can charge device-local memory to
// the process's commit as well: this tells how much, by kind, in the "Private breakdown:" line.
namespace {
PFN_vkAllocateMemory           g_real_allocate = nullptr;
PFN_vkFreeMemory               g_real_free     = nullptr;
VkPhysicalDeviceMemoryProperties g_memory_properties {};
std::mutex                     g_tag_mutex;
std::atomic<bool>              g_tag_ballast {false};
struct TaggedMemory {
	int     tag;
	int64_t requested, committed;
	int     cat;   // (VRAM-DRV) VramDrv::Cat del punto de llamada
	bool    local; // (VRAM-DRV) tipo device-local
};
std::unordered_map<uint64_t, TaggedMemory>* g_tagged = nullptr;

// (VRAM-DRV fix 11-25) Log 20261125-061424: con KYTY_VRAM_DRV=1 el juego pinta ~5 s a 60 fps y se queda en 0 flips/s,
// sin ninguna linea [vram-drv]; con =0 (gancho sin poner) juega. El gancho tenia g_tag_mutex tomado DURANTE la llamada
// real al driver: todas las vkAllocateMemory del proceso (VMA, texturas nativas, bufferes sparse, memoria importada del
// juego, de cualquier hilo) quedaban en fila detras de una sola, y una que reentre en el mismo hilo (fallo de pagina
// atendido dentro del driver al importar memoria del juego, VMA dentro de otra) se bloqueaba contra si misma
// (std::mutex no es recursivo). Ahora sin KYTY_MEMORY_TAGS el driver se llama SIN el mutex (solo el mapa va con el), y
// una llamada anidada en el mismo hilo va directa al driver sin contarse dos veces. Con KYTY_MEMORY_TAGS=1 sigue en
// serie (la diferencia de commit antes/despues debe ser solo suya), pero tambien sin reentrar.
thread_local int t_tag_depth = 0;

VKAPI_ATTR VkResult VKAPI_CALL TaggedAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                                                    const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
	if (t_tag_depth != 0) return g_real_allocate(device, info, callbacks, memory); // anidada (mismo hilo): directa, sin contar
	struct Depth {
		Depth() { t_tag_depth++; }
		~Depth() { t_tag_depth--; }
	} depth;
	const bool tags = LocalPlatform::MemoryTagsEnabled();
	std::unique_lock serial(g_tag_mutex, std::defer_lock);
	if (tags) serial.lock(); // solo para medir el commit de esta llamada sola
	const auto before = tags ? LocalPlatform::ProcessMemory().private_bytes : 0;
	const auto result = g_real_allocate(device, info, callbacks, memory);
	const auto after  = tags ? LocalPlatform::ProcessMemory().private_bytes : 0;
	if (result == VK_SUCCESS && memory != nullptr && *memory != VK_NULL_HANDLE && info != nullptr) {
		const auto flags = info->memoryTypeIndex < g_memory_properties.memoryTypeCount
		                       ? g_memory_properties.memoryTypes[info->memoryTypeIndex].propertyFlags
		                       : 0u;
		const int tag = g_tag_ballast.load() ? LocalPlatform::MemoryTagVulkanBallast
		                : (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 ? LocalPlatform::MemoryTagVulkanDeviceLocal
		                                                                     : LocalPlatform::MemoryTagVulkanHost;
		const bool local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
		const int  cat   = g_tag_ballast.load() ? VramDrv::CatLastre : VramDrv::t_cat;
		const TaggedMemory entry {tag, static_cast<int64_t>(info->allocationSize),
		                          after > before ? static_cast<int64_t>(after - before) : 0, cat, local};
		if (!tags) serial.lock(); // el mapa, solo un momento
		if (g_tagged == nullptr) g_tagged = new std::unordered_map<uint64_t, TaggedMemory>();
		(*g_tagged)[reinterpret_cast<uint64_t>(*memory)] = entry;
		serial.unlock();
		if (tags) LocalPlatform::MemoryTagAdd(entry.tag, entry.requested, entry.committed);
		VramDrv::Add(cat, local, entry.requested, 1);
	}
	return result;
}

VKAPI_ATTR void VKAPI_CALL TaggedFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
	if (memory != VK_NULL_HANDLE) {
		bool         found = false;
		TaggedMemory entry {};
		{
			std::lock_guard lock(g_tag_mutex);
			if (g_tagged != nullptr) {
				if (auto it = g_tagged->find(reinterpret_cast<uint64_t>(memory)); it != g_tagged->end()) {
					entry = it->second;
					found = true;
					g_tagged->erase(it);
				}
			}
		}
		if (found) {
			if (LocalPlatform::MemoryTagsEnabled()) LocalPlatform::MemoryTagAdd(entry.tag, -entry.requested, -entry.committed);
			VramDrv::Add(entry.cat, entry.local, -entry.requested, -1);
		}
	}
	g_real_free(device, memory, callbacks);
}

// (VRAM-DRV) VMA's own calls: marked as VMA (its blocks and dedicated allocations).
VKAPI_ATTR VkResult VKAPI_CALL VmaAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                                                 const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
	VramDrv::Scope scope(VramDrv::CatVma);
	return TaggedAllocateMemory(device, info, callbacks, memory);
}

void InstallMemoryTags(VkPhysicalDevice physical_device) {
	if ((!LocalPlatform::MemoryTagsEnabled() && !VramDrv::On()) || g_real_allocate != nullptr) return;
	auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (dispatcher.vkAllocateMemory == nullptr || dispatcher.vkFreeMemory == nullptr) return;
	dispatcher.vkGetPhysicalDeviceMemoryProperties(physical_device, &g_memory_properties);
	g_real_allocate             = dispatcher.vkAllocateMemory;
	g_real_free                 = dispatcher.vkFreeMemory;
	dispatcher.vkAllocateMemory = TaggedAllocateMemory;
	dispatcher.vkFreeMemory     = TaggedFreeMemory;
}
} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	InstallMemoryTags(static_cast<VkPhysicalDevice>(physical_device));
	if (g_real_allocate != nullptr) {
		functions.vkAllocateMemory = VmaAllocateMemory;
		functions.vkFreeMemory     = TaggedFreeMemory;
	}

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}
	// (vram-budget) The small video memory heap (graphicContext.h): known before the allocator (block size), from the heaps.
	{
		uint64_t largest = 0;
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			if (physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
				largest = std::max<uint64_t>(largest, physical_device_memory_properties.memoryHeaps[heap].size);
		}
		if (const char* simulate = std::getenv("KYTY_SIMULATE_VRAM_MB"); simulate != nullptr)
			g_simulated_vram = std::strtoull(simulate, nullptr, 10) << 20u;
		if (g_simulated_vram >= largest) g_simulated_vram = 0; // (no smaller card to simulate)
		g_real_vram_heap = largest;
		constexpr uint64_t SmallHeap = uint64_t {10} << 30u;
		g_vram_real_small = largest != 0 && largest <= SmallHeap;
		small_vram_heap = (largest != 0 && largest <= SmallHeap) || (g_simulated_vram != 0 && g_simulated_vram <= SmallHeap);
	}
	// Memory blocks of 64 MiB (VMA's default is 256): a block stays allocated while any allocation of it lives, so
	// the game's streaming left 21 blocks of 256 MiB with 2.7 GB free inside them at Boletaria (11.9 GB held of
	// which 8.9 in use); smaller blocks hold less around each straggler. KYTY_VMA_BLOCK_MB=<n> for an A/B.
	{
		// (vram-budget) 32 MiB on a small video memory heap too: "free inside VMA blocks" was 500-655 MiB at 64. VMA gives
		// an allocation over half a block its own memory (dedicated), so images over 16 MiB still get theirs.
		// (VRAM-HUECO, 26/11) 32 MiB en todas las tarjetas: log 20261126-230735 (16 GB, bloques de 64) el hueco dentro de
		// los bloques subia 95 -> 704 MiB en 11 min (132 -> 222 bloques) con lo usado plano. KYTY_VMA_BLOCK_MB=64 como antes.
		uint64_t block_mb = 32;
		if (const char* text = std::getenv("KYTY_VMA_BLOCK_MB"); text != nullptr && std::strtoull(text, nullptr, 10) > 0)
			block_mb = std::strtoull(text, nullptr, 10);
		info.preferredLargeHeapBlockSize = block_mb << 20u;
		std::printf("Video memory: allocator blocks of %llu MiB\n", static_cast<unsigned long long>(block_mb));
	}
	// KYTY_VRAM_LIMIT_MB=<n>: video memory as on a GPU with n MiB (VMA's heap size limit): its budget,
	// and allocations past it failing, for tests of memory pressure on smaller GPUs.
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heap_limits {};
	if (const char* text = std::getenv("KYTY_VRAM_LIMIT_MB"); text != nullptr && std::strtoull(text, nullptr, 10) > 0) {
		heap_limits.fill(VK_WHOLE_SIZE);
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			if (physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				heap_limits[heap] = std::strtoull(text, nullptr, 10) << 20u;
			}
		}
		info.pHeapSizeLimit = heap_limits.data();
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	// KYTY_SIMULATE_VRAM_MB=<n>: a GPU with n MiB of video memory, for tests on a bigger one. Unlike
	// KYTY_VRAM_LIMIT_MB (an allocator limit: what it puts in system memory types the driver still places in video
	// memory while there is room), the rest of the largest device-local heap is taken here at the highest priority
	// and kept, so the driver and OS move the game's memory to system memory as on such a GPU. Budget and usage
	// (GetTotalMemoryBudget, GetDeviceMemoryUsage) leave it out.
	// (VRAM8-SIN-LASTRE) The ballast only with KYTY_SIMULATE_VRAM_LASTRE=1 (old way); by default the simulation is the
	// budget alone (GetTotalMemoryBudget, GetVideoMemoryRoom, QueryLocalVideoMemory), no video memory taken.
	static const bool lastre = [] { const char* v = std::getenv("KYTY_SIMULATE_VRAM_LASTRE"); return !(v != nullptr && *v == '0'); }(); // (11-26 limpieza) lastre por defecto, como la release
	if (const char* simulate = std::getenv("KYTY_SIMULATE_VRAM_MB"); simulate != nullptr && lastre) {
		const uint64_t simulated = std::strtoull(simulate, nullptr, 10) << 20u;
		uint32_t       heap      = 0;
		for (uint32_t i = 1; i < physical_device_memory_properties.memoryHeapCount; i++) {
			if (physical_device_memory_properties.memoryHeaps[i].size > physical_device_memory_properties.memoryHeaps[heap].size &&
			    (physical_device_memory_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal))
				heap = i;
		}
		uint32_t type = UINT32_MAX;
		for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount && type == UINT32_MAX; i++) {
			const auto& candidate = physical_device_memory_properties.memoryTypes[i];
			if (candidate.heapIndex == heap && candidate.propertyFlags == vk::MemoryPropertyFlagBits::eDeviceLocal) type = i;
		}
		const uint64_t size = physical_device_memory_properties.memoryHeaps[heap].size;
		g_tag_ballast = true;
		for (uint64_t left = simulated > 0 && simulated < size ? size - simulated : 0; left > 0 && type != UINT32_MAX;) {
			const uint64_t                    chunk = std::min<uint64_t>(left, uint64_t {1} << 30u);
			vk::MemoryPriorityAllocateInfoEXT priority {};
			priority.priority = 1.0f;
			vk::MemoryAllocateInfo info {};
			info.pNext           = memory_priority_enabled ? &priority : nullptr;
			info.allocationSize  = chunk;
			info.memoryTypeIndex = type;
			vk::DeviceMemory memory = nullptr;
			if (device.allocateMemory(&info, nullptr, &memory) != vk::Result::eSuccess) break;
			simulation_ballast.push_back(memory);
			simulation_ballast_bytes += chunk;
			left -= chunk;
		}
		g_tag_ballast = false;
		LocalPlatform::SetSimulationBallast(simulation_ballast_bytes);
		std::printf("Simulated video memory: %" PRIu64 " MiB of heap %u (%" PRIu64 " MiB) taken, priorities %s\n",
		            simulation_ballast_bytes >> 20u, heap, size >> 20u, memory_priority_enabled ? "on" : "off");
	}
	if (g_simulated_vram != 0 && !lastre) {
		std::printf("simulaci\xC3\xB3n 8 GB sin lastre: presupuesto %llu MiB (tarjeta simulada %llu MiB, caches %llu MiB)\n",
		            static_cast<unsigned long long>(GetTotalMemoryBudget(false) >> 20u),
		            static_cast<unsigned long long>(g_simulated_vram >> 20u),
		            static_cast<unsigned long long>(GetTotalMemoryBudget() >> 20u));
		std::fflush(stdout);
	}
	{
		vk::PhysicalDeviceIDProperties id {};
		vk::PhysicalDeviceProperties2  device_properties {};
		device_properties.pNext = &id;
		physical_device.getProperties2(&device_properties);
		if (id.deviceLUIDValid) (void)LocalPlatform::OpenVideoMemoryAdapter(id.deviceLUID.data());
	}
	// (The GPU's own budget, not the caches' cap: a 16 GB card is not short of video memory.)
	// KYTY_SMALL_VRAM=1 (run-windows.ps1 -Turbo): as on a small GPU whatever this one has (textures yield, no warmup).
	const char* small_env = std::getenv("KYTY_SMALL_VRAM");
	small_video_memory = GetTotalMemoryBudget(false) < (uint64_t {12} << 30u) || (small_env != nullptr && *small_env == '1');
	LocalPlatform::StartWorkingSetTrim();
	SoftRelease::Start();
	RamGuard::Start();
	LocalPlatform::KeepResident();
	LocalPlatform::StartMemoryBreakdown();
	// The game buffers' pool (graphicContext.h): VMA's device-local type for them, VMA's block sizes (dedicated memory
	// for big ones, as in the default pools).
	if (const char* defrag = std::getenv("KYTY_BUFFER_DEFRAG");
	    defrag == nullptr || std::strcmp(defrag, "0") != 0) {
		// (VRAM-HUECO, 26/11) En todas las tarjetas (antes solo < 12 GB): en 16 GB los ~2000-5000 bufferes del juego
		// (1,2-1,7 GB) iban mezclados con las imagenes en los bloques comunes y dejaban huecos que nadie compactaba.
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = uint64_t {64} << 10u;
		buffer.usage = static_cast<VkBufferUsageFlags>(AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress);
		VmaAllocationCreateInfo allocation {};
		allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		uint32_t type    = 0;
		VmaPoolCreateInfo pool {};
		if (vmaFindMemoryTypeIndexForBufferInfo(allocator, &buffer, &allocation, &type) == VK_SUCCESS) {
			pool.memoryTypeIndex = type;
			if (vmaCreatePool(allocator, &pool, &game_buffer_pool) != VK_SUCCESS) game_buffer_pool = nullptr;
		}
		std::printf("Game buffers: %s\n", game_buffer_pool != nullptr ? "own pool, compacted (KYTY_BUFFER_DEFRAG=0 turns it off)"
		                                                               : "pool failed, default pools");
	}
	LogVideoMemory("start");
	g_report_allocator          = allocator;
	g_report_context            = this;
	LiveCounters::g_vma_report = WriteVmaReport;
	g_image_pool_limit         = std::min<uint64_t>(LocalPlatform::LowMemVram() ? 256ull << 20 : 1024ull << 20, GetTotalMemoryBudget() / 32);
	// (vram-budget) On a small video memory heap at most 64 MiB: enough for the transient targets the game deletes and
	// makes again every frame (a 2560x1440 RGBA16F target is 28 MiB), not for the area's leftovers (234-255 MiB held).
	// Not 0: every freed image would then be made again at its next use, a vkAllocateMemory each.
	if (small_vram_heap) g_image_pool_limit = std::min<uint64_t>(g_image_pool_limit, 64ull << 20);
	std::printf("Video memory: image reuse pool of %llu MiB at most%s\n", static_cast<unsigned long long>(g_image_pool_limit >> 20u),
	            small_vram_heap ? " (small video memory heap)" : "");
	// Out of memory (MemGuard): the freed images kept for reuse go (made again when needed). A system-memory image's
	// pages are commit on NVIDIA.
	// The periodic RAM flush (ram-guard.h, KYTY_RAM_FLUSH_SECONDS): the same images, a budget's worth, destroyed outside
	// the pool's lock.
	{
		static std::once_flag once;
		std::call_once(once, [] { RamGuard::RegisterBudgetFlush("image reuse pool", RamGuard::FlushCaches, FlushImagePool); });
	}
	g_image_pool_releaser = MemGuard::RegisterReleaser("image reuse pool", [vma = allocator] {
		uint64_t bytes = 0;
		{
			std::scoped_lock lock(g_image_pool_mutex);
			bytes = g_image_pool_bytes;
		}
		DestroyImagePool(vma);
		return bytes;
	});
	{
		// The caches' video memory budget, in the run log with the build (to tell builds apart in a log).
		uint64_t local = 0;
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			if (physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				local += physical_device_memory_properties.memoryHeaps[heap].size;
			}
		}
		std::printf("Video memory: %llu MiB on the GPU; the caches' budget is %llu MiB (build %s)\n",
		            static_cast<unsigned long long>(local >> 20u),
		            static_cast<unsigned long long>(GetTotalMemoryBudget() >> 20u), KYTY_GIT_HASH);
		std::fflush(stdout);
	}
	VramStoreRegister(*this); // (mem1 etapa 2: only registered; its resources come at its first probe)
	return true;
}

static void DestroyImagePool(VmaAllocator allocator);

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	LiveCounters::g_vma_report = nullptr;
	g_report_allocator         = nullptr;
	g_report_context           = nullptr;
	FlushBufferReclaimer();
	if (g_image_pool_releaser != 0) MemGuard::UnregisterReleaser(g_image_pool_releaser);
	g_image_pool_releaser = 0;
	DestroyImagePool(allocator);
	if (game_buffer_pool != nullptr) {
		// (Only empty: a buffer still alive frees its memory into it later.)
		VmaStatistics statistics {};
		vmaGetPoolStatistics(allocator, game_buffer_pool, &statistics);
		if (statistics.allocationCount == 0) vmaDestroyPool(allocator, game_buffer_pool);
		game_buffer_pool = nullptr;
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
	for (const auto memory: simulation_ballast) device.freeMemory(memory);
	simulation_ballast.clear();
	simulation_ballast_bytes = 0;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

// (ram4) The "[ram+]" line (ram-guard.cpp, KYTY_RAM_LINE=1; its own thread): the emulator's own Vulkan memory blocks,
// bytes, in the device-local heaps (video memory; the ballast is not VMA's and comes apart) and in the others (system
// memory: the host-visible upload and readback rings, and the driver's plain system type). VMA's per-heap counters
// only (atomics; no walk of its blocks). False before the allocator is made or after it is gone.
bool RamLineVulkan(uint64_t* video_blocks, uint64_t* system_blocks, uint64_t* ballast);
bool RamLineVulkan(uint64_t* video_blocks, uint64_t* system_blocks, uint64_t* ballast) {
	const VmaAllocator    allocator = g_report_allocator;
	const GraphicContext* context   = g_report_context;
	if (allocator == nullptr || context == nullptr) return false;
	const auto& properties = context->GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	uint64_t video = 0, host_side = 0;
	for (uint32_t heap = 0; heap < properties.memoryHeapCount && heap < VK_MAX_MEMORY_HEAPS; heap++) {
		const bool local = static_cast<bool>(properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		(local ? video : host_side) += budgets[heap].statistics.blockBytes;
	}
	*video_blocks  = video;
	*system_blocks = host_side;
	*ballast       = LocalPlatform::SimulationBallast();
	return true;
}

uint64_t VramCardBytes(); // abajo

void GraphicContext::RefreshMemoryBudget() {
	static std::atomic<uint32_t> frame {0};
	if (allocator != nullptr) vmaSetCurrentFrameIndex(allocator, frame.fetch_add(1, std::memory_order_relaxed) + 1);
	{
		// (VRAM-EMU) Una vez por segundo: el scratch del tiler guardado y sin uso desde hace 5 s vuelve al driver
		// (KYTY_EMU_TILER_POOL=0: como antes, solo el vaciado periodico de 30 s).
		static std::atomic<int64_t> last_trim_ms {0};
		const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		                           std::chrono::steady_clock::now().time_since_epoch()).count();
		int64_t last = last_trim_ms.load(std::memory_order_relaxed);
		if (now_ms - last >= 1000 && last_trim_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed))
			(void)TilerTrimIdle();
	}
	// (GESTOR-MEM) Una vez por fotograma: medidas en caliente y, si hace falta, expulsion por prioridad.
	if (GestorMem::On() && allocator != nullptr) {
		static std::atomic<bool> dentro {false};
		if (dentro.exchange(true)) return; // RefreshMemoryBudget se llama desde dos sitios: una sola pasada a la vez
		GestorMem::Medidas m;
		m.now_ms      = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch()).count());
		m.card_bytes  = VramCardBytes();
		m.vram_uso    = GetDeviceMemoryUsage();
		(void)GetVideoMemoryRoom(); // escribe VramDrv::g_drv_budget_bytes (presupuesto real del driver)
		m.vram_driver = VramDrv::g_drv_budget_bytes.load(std::memory_order_relaxed);
		static uint64_t ram_ms = 0;
		static GestorMem::Medidas ram {};
		if (m.now_ms - ram_ms >= 250) { // RAM: 4 veces por segundo basta (llamadas al sistema)
			ram_ms             = m.now_ms;
			ram.ram_fisica     = LocalPlatform::PhysicalMemory();
			ram.ram_libre      = LocalPlatform::AvailablePhysicalMemory();
			const auto pm      = LocalPlatform::ProcessMemory();
			ram.ram_ws         = pm.working_set;
			ram.ram_privada    = pm.private_bytes;
		}
		m.ram_fisica  = ram.ram_fisica;
		m.ram_libre   = ram.ram_libre;
		m.ram_ws      = ram.ram_ws;
		m.ram_privada = ram.ram_privada;
		m.ram_baja    = ram.ram_libre != 0 && ram.ram_libre < (GestorMem::RamCriticaMiB << 20u);
		GestorMem::g_cats.vma        = static_cast<uint64_t>(std::max<int64_t>(VramDrv::g_bytes[VramDrv::CatVma][0].load(), 0));
		GestorMem::g_cats.nativas    = static_cast<uint64_t>(std::max<int64_t>(VramDrv::g_bytes[VramDrv::CatSparseImg][0].load(), 0));
		GestorMem::g_cats.sparse_buf = static_cast<uint64_t>(std::max<int64_t>(VramDrv::g_bytes[VramDrv::CatSparseBuf][0].load(), 0));
		GestorMem::g_cats.store      = static_cast<uint64_t>(std::max<int64_t>(VramDrv::g_bytes[VramDrv::CatVramStore][0].load(), 0));
		GestorMem::g_cats.otro       = static_cast<uint64_t>(std::max<int64_t>(VramDrv::g_bytes[VramDrv::CatOtro][0].load(), 0));
		GestorMem::Tick(m);
		dentro.store(false);
	}
}

bool GraphicContext::QueryLocalVideoMemory(uint64_t* usage, uint64_t* budget) const {
	if (!LocalPlatform::QueryVideoMemory(usage, budget)) return false;
	*usage -= std::min(*usage, simulation_ballast_bytes);
	*budget -= std::min(*budget, simulation_ballast_bytes);
	// (VRAM8-SIN-LASTRE) The simulated card's driver budget, not the big card's.
	if (g_simulated_vram != 0 && simulation_ballast_bytes == 0) *budget = std::min(*budget, SimulatedBudgetFromReal(*budget));
	*budget = RealSmallCardBudget(*budget); // (VRAM-REAL)
	return true;
}

void GraphicContext::LogVideoMemory(const char* label) const {
	uint64_t video = 0, budget = 0, shared = 0;
	if (allocator == nullptr || !QueryLocalVideoMemory(&video, &budget) || !LocalPlatform::QuerySharedGpuMemory(&shared)) return;
	// VMA's memory by kind of memory type: device-local (video memory), the driver's system memory type without
	// flags (NVIDIA places it in video memory while there is room), host-visible.
	VmaTotalStatistics statistics {};
	vmaCalculateStatistics(allocator, &statistics);
	uint64_t local = 0, plain = 0, host = 0;
	const auto& properties = GetPhysicalDeviceMemoryProperties();
	for (uint32_t type = 0; type < properties.memoryTypeCount; type++) {
		const auto flags = properties.memoryTypes[type].propertyFlags;
		const auto bytes = statistics.memoryType[type].statistics.blockBytes;
		if (flags & vk::MemoryPropertyFlagBits::eDeviceLocal) local += bytes;
		else if (flags & vk::MemoryPropertyFlagBits::eHostVisible) host += bytes;
		else plain += bytes;
	}
	// What of the plain type the GPU's system memory does not hold is in video memory; the rest of the video memory in
	// use is the driver's own (pipelines, shader local memory, descriptors).
	const uint64_t plain_in_system = std::min(plain, shared - std::min(shared, host));
	const uint64_t plain_in_video  = plain - plain_in_system;
	const uint64_t driver          = video - std::min(video, local + plain_in_video);
	std::printf("Video memory (%s): %" PRIu64 " MiB in use of %" PRIu64 " MiB (VMA device-local %" PRIu64 " MiB, plain type %" PRIu64
	            " MiB in video memory, driver %" PRIu64 " MiB); GPU system memory %" PRIu64 " MiB (plain type %" PRIu64
	            " MiB, host-visible %" PRIu64 " MiB)\n",
	            label, video >> 20u, budget >> 20u, local >> 20u, plain_in_video >> 20u, driver >> 20u, shared >> 20u,
	            plain_in_system >> 20u, host >> 20u);
	if (std::strcmp(label, "30 s") == 0) {
		// (VRAM-REAL) Uso real: el del proceso (DXGI, lo que Task Manager da al proceso), lo que son caches/VMA, lo que no
		// (driver: pipelines, swapchain, memoria interna) y lo que tienen los demas procesos en la misma tarjeta.
		uint64_t all = 0, others = 0;
		const bool adapter = LocalPlatform::QueryAdapterDedicatedUsage(&all, &others);
		std::printf("[vram-real] 30 s, MiB: proceso %" PRIu64 " (caches/VMA %" PRIu64 " + fuera de las caches %" PRIu64
		            "); otros procesos %s%" PRIu64 ", tarjeta entera %" PRIu64 "; presupuesto efectivo %" PRIu64
		            " (proceso + otros %" PRIu64 " de %" PRIu64 " del driver %s); KYTY_VRAM_REAL %s\n",
		            video >> 20u, (local + plain_in_video) >> 20u, driver >> 20u, adapter ? "" : "(sin medir) ",
		            others >> 20u, all >> 20u, budget >> 20u, (video + others) >> 20u,
		            (g_simulated_vram != 0 ? SimulatedDriverBudget(g_simulated_vram) : SimulatedDriverBudget(g_real_vram_heap)) >> 20u,
		            g_simulated_vram != 0 ? "simulado" : "real", VramRealOn() ? "encendido" : "apagado");
		// By category: the freed images kept for reuse, the game buffers' own pool, the rest of VMA's device-local
		// memory (textures and render targets of the texture cache, effect and upscaler images, stream rings).
		uint64_t pool_bytes = 0;
		{
			std::scoped_lock lock(g_image_pool_mutex);
			pool_bytes = g_image_pool_bytes;
		}
		VmaStatistics buffers {};
		if (game_buffer_pool != nullptr) vmaGetPoolStatistics(allocator, game_buffer_pool, &buffers);
		const uint64_t used_local = [&] {
			uint64_t used = 0;
			for (uint32_t type = 0; type < properties.memoryTypeCount; type++)
				if (properties.memoryTypes[type].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)
					used += statistics.memoryType[type].statistics.allocationBytes;
			return used;
		}();
		const uint64_t other = used_local - std::min(used_local, pool_bytes + buffers.allocationBytes);
		std::printf("Video memory by category (30 s): game buffer pool %" PRIu64 " MiB (blocks %" PRIu64
		            "), image reuse pool %" PRIu64 " MiB, images and other device-local %" PRIu64
		            " MiB, free inside VMA blocks %" PRIu64 " MiB, driver %" PRIu64 " MiB, room under the budget %" PRId64
		            " MiB\n",
		            buffers.allocationBytes >> 20u, buffers.blockBytes >> 20u, pool_bytes >> 20u, other >> 20u,
		            (local - std::min(local, used_local)) >> 20u, driver >> 20u,
		            (static_cast<int64_t>(budget) - static_cast<int64_t>(video)) / (1 << 20));
		{
			// (VRAM-EMU) Los "images and other device-local" de arriba por dueno; los huecos libres, del pool de bufferes y del
			// resto de bloques.
			const uint64_t free_all  = local - std::min(local, used_local);
			const uint64_t free_pool = buffers.blockBytes - std::min(buffers.blockBytes, buffers.allocationBytes);
			VramEmu::Report(other, pool_bytes, TilerPoolBytes(), free_pool, free_all - std::min(free_all, free_pool));
		}
		// (DRV-PIPE, 11-25) Log 20261125-201501: "driver" 1457 -> 1909 MiB en escalones al cambiar de mundo, sin linea que
		// diga cuantos pipelines habia. Esta pone al lado los pipelines de la cache (graficos + compute en sus mapas, con
		// los expulsados) para ver si crecen a la par; la cifra a comparar es la primera y la ultima de la partida.
		std::printf("[drv-pipe] 30 s: driver %" PRIu64 " MiB (fuera de VMA: pipelines, descriptores, texturas dispersas, pool "
		            "sparse); pipelines en la cache %llu graficos + %llu compute, expulsados %llu, rehechos %llu\n",
		            driver >> 20u, static_cast<unsigned long long>(VramRecursos::g_graphics.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(VramRecursos::g_compute.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(VramRecursos::g_expulsados.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(VramRecursos::g_rehechos.load(std::memory_order_relaxed)));
		// The emulator's private commit split into what NVIDIA's Windows driver charges for video memory and the rest.
		// The logs show private minus VMA's device-local blocks flat while video memory moves (16 GB run ~4.8 GB,
		// 64 GB run ~6.1 GB), so the driver is taken to charge each device-local block's full size. An estimate from
		// totals, not measured per allocation (KYTY_MEMORY_TAGS=1 measures it per vkAllocateMemory).
		const uint64_t private_bytes = LocalPlatform::ProcessMemory().private_bytes;
		const uint64_t vulkan_blocks = local + plain + host;
		std::printf("Private split (30 s): emulator private %" PRIu64 " MiB; video memory in use %" PRIu64
		            " MiB (VMA device-local blocks %" PRIu64 " MiB, plain type blocks %" PRIu64
		            " MiB, host-visible blocks %" PRIu64 " MiB); non-driver remainder %" PRIu64
		            " MiB (private minus VMA blocks), %" PRIu64 " MiB (private minus video memory in use)\n",
		            private_bytes >> 20u, video >> 20u, local >> 20u, plain >> 20u, host >> 20u,
		            (private_bytes - std::min(private_bytes, vulkan_blocks)) >> 20u,
		            (private_bytes - std::min(private_bytes, video)) >> 20u);
	}
	std::fflush(stdout);
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage - std::min(usage, simulation_ballast_bytes);
}

uint64_t GraphicContext::GetVideoMemoryRoom() const {
	// (Without VK_EXT_memory_budget VMA counts only its own blocks against a guessed budget: no figure to go by.)
	if (allocator == nullptr || !CanReportMemoryUsage()) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto& heaps = physical_device_memory_properties.memoryHeaps;
	uint32_t    video = UINT32_MAX;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool larger = video == UINT32_MAX || heaps[heap].size > heaps[video].size;
		if ((heaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) && larger) video = heap;
	}
	if (video == UINT32_MAX) {
		return 0;
	}
	// (The driver's usage holds KYTY_SIMULATE_VRAM_MB's ballast: the room left is the same.)
	const auto& budget = budgets[video];
	uint64_t    room   = budget.budget > budget.usage ? budget.budget - budget.usage : 0;
	// (N3-REAL) The budget itself, in the process usage's units (GetDeviceMemoryUsage: ballast taken off), so that
	// vram-tope.h sees how far OVER it the process is (room stops at 0).
	uint64_t real_budget = budget.budget - std::min<uint64_t>(budget.budget, simulation_ballast_bytes);
	// (vram-budget) KYTY_SIMULATE_VRAM_MB: no more than that card's driver would leave (the ballast may have got less).
	if (g_simulated_vram != 0) {
		const uint64_t usage = budget.usage - std::min<uint64_t>(budget.usage, simulation_ballast_bytes);
		const uint64_t cap   = simulation_ballast_bytes == 0 ? SimulatedBudgetFromReal(budget.budget) : SimulatedDriverBudget(g_simulated_vram);
		room                 = std::min(room, cap > usage ? cap - usage : uint64_t {0});
		real_budget          = std::min(real_budget, cap);
	} else if (g_vram_real_small) {
		const uint64_t cap = RealSmallCardBudget(budget.budget); // (VRAM-REAL)
		room               = std::min(room, cap > budget.usage ? cap - budget.usage : uint64_t {0});
		real_budget        = std::min(real_budget, cap);
	}
	VramDrv::g_drv_budget_bytes.store(real_budget, std::memory_order_relaxed);
	return room;
}

// (vram8 gc) VMA's device-local blocks less what is allocated in them, less the game buffers' pool's own free space
// (its blocks take only game buffers).
uint64_t GraphicContext::GetVideoBlockSlack() const {
	if (allocator == nullptr) return 0;
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	uint64_t slack = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount && heap < VK_MAX_MEMORY_HEAPS; heap++) {
		if (!(physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) continue;
		const auto& st = budgets[heap].statistics;
		slack += st.blockBytes - std::min(st.blockBytes, st.allocationBytes);
	}
	if (game_buffer_pool != nullptr) {
		VmaStatistics pool {};
		vmaGetPoolStatistics(allocator, game_buffer_pool, &pool);
		slack -= std::min(slack, pool.blockBytes - std::min(pool.blockBytes, pool.allocationBytes));
	}
	return slack;
}

// Allocations in the heaps that are not device-local (system RAM the driver maps for the GPU): staging and
// readback buffers, and anything the budget pushed out of video memory.
uint64_t GraphicContext::GetHostMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!device_local) usage += budgets[heap].usage;
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget(bool capped) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	budget -= std::min(budget, simulation_ballast_bytes);
	usage -= std::min(usage, simulation_ballast_bytes);
	local -= std::min(local, simulation_ballast_bytes);
	// (vram-budget) KYTY_SIMULATE_VRAM_MB: that card's memory and driver budget whatever the ballast got (a run on a 16 GB
	// card logged "in use of 15209 MiB" and a caches' budget of 8192: without a ballast it is the big card's budget).
	if (g_simulated_vram != 0) {
		local  = std::min(local, g_simulated_vram);
		budget = std::min(budget, simulation_ballast_bytes == 0 && CanReportMemoryUsage() ? SimulatedBudgetFromReal(budget)
		                                                                                    : SimulatedDriverBudget(g_simulated_vram));
	} else if (CanReportMemoryUsage()) {
		budget = RealSmallCardBudget(budget); // (VRAM-REAL)
	}
	if (discrete) {
		uint64_t result = budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
		// (VRAM-REPARTO, 11-24; KYTY_VRAM_REPARTO=0 lo apaga; KYTY_VRAM_MARGEN_MB=<n>, 256 por defecto) Solo con la VRAM del
		// adaptador <= 8 GB. Log 20261124-113637 (8 GB simulados): driver 6937 MiB -> [vram] 6070 (- 1/8) -> caches 5120
		// (local - 3 GiB): 1817 MiB de un presupuesto ya medido por el driver se perdian en dos margenes fijos; el juego usa
		// 5,6-5,8 GB en Boletaria y el recolector expulsaba sin parar (20-23 ms). Ahora caches y [vram] = driver - margen
		// medido (256); VRAM-DESG sigue dejando 384+256 bajo el driver, asi que no se le llega a pasar.
		static const bool reparto = [] { const char* v = std::getenv("KYTY_VRAM_REPARTO"); return v == nullptr || *v != '0'; }();
		if (reparto && small_vram_heap && CanReportMemoryUsage()) {
			static const uint64_t margen = [] {
				const char* v = std::getenv("KYTY_VRAM_MARGEN_MB");
				return (v != nullptr ? std::strtoull(v, nullptr, 10) : 256ull) << 20u;
			}();
			const uint64_t r = budget - std::min(budget, margen);
			if (!capped) return r;
			static const uint64_t forced8 = [] {
				const char* text = std::getenv("KYTY_VRAM_BUDGET_MB");
				return text != nullptr ? std::strtoull(text, nullptr, 10) << 20u : uint64_t {0};
			}();
			return forced8 != 0 ? std::min(r, forced8) : r;
		}
		if (!capped) return result;
		// (mem1 etapa 2) The offload store's video memory (vramStore.cpp) on top of the caches' cap: the caches keep their
		// room; it only grows within the driver's budget (its probe and each block check that).
		const uint64_t uncapped_result = result;
		const uint64_t store_bytes     = VramStore::Get().allocated.load(std::memory_order_relaxed);
		const auto     with_store      = [&](uint64_t cap) { return std::min(uncapped_result, cap + store_bytes); };
		// KYTY_VRAM_BUDGET_MB=<n> sets the caches' budget outright.
		static const uint64_t forced = [] {
			const char* text = std::getenv("KYTY_VRAM_BUDGET_MB");
			return text != nullptr ? std::strtoull(text, nullptr, 10) << 20u : uint64_t {0};
		}();
		if (forced != 0) return with_store(std::min(result, forced));
		// The caches collect from the GPU's memory less 3 GB, at most 8 GB: 8 GB on a 12 GB card and up
		// one, so the whole process (pipelines and the driver's own allocations on top) stays near 11-12 GB with
		// room for Windows, the desktop and a browser. (13.4 GB on a 16 GB card: 13.3 GB in use after 8 minutes,
		// near 15 GB with the rest of the system.) small_video_memory reads the driver's budget, not this cap.
		constexpr uint64_t GiB = 1024ull * 1024 * 1024;
		// 8 GB at most: a 16 GB card with the desktop, an overlay and a browser stayed near 15 GB in use at 10 GB.
		// (11-26) Tope por tarjeta, como pidio el usuario: 8 GB -> 6,3 GB; 10 GB -> 7,5; 12 GB -> 10; 16 GB y mas -> 14,5.
		// (Antes: la memoria menos 3 GB, como mucho 8 GB en todas.) KYTY_VRAM_BUDGET_MB sigue mandando si se pone.
		if (local > 4 * GiB) {
			const uint64_t MiB  = 1024ull * 1024;
			const uint64_t tope = local <= 8 * GiB + GiB / 2 ? 6451 * MiB : local <= 10 * GiB + GiB / 2 ? 7680 * MiB
			                    : local <= 12 * GiB + GiB / 2 ? 10240 * MiB : 14848 * MiB;
			result = std::min(result, tope);
		}
		// KYTY_LOWMEM_VRAM=1: at most 6 GB, the GPU's memory less 4 GB (not below 3 GB), only on cards above 8.5 GB.
		// On 8 GB it gave 4096 instead of the default 5120 and hitched far more (simulated 8 GB: 25-97 hitch windows
		// at 4096 against 3-17 in most runs at 5120): the collector then deletes and re-uploads textures mid-play.
		if (LocalPlatform::LowMemVram() && local > 8 * GiB + GiB / 2)
			result = std::min(result, std::max(std::min(local - 4 * GiB, 6 * GiB), 3 * GiB));
		return with_store(result);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

// Recycling of freed images (KYTY_IMAGE_POOL). The game aliases transient render
// targets in one heap, so the texture cache deletes and re-creates the same few
// images every frame. An image reaches DeleteImage only after the GPU is done
// with it; a new image always starts in eUndefined layout, so reusing the VkImage
// and its memory for identical creation parameters is equivalent to a fresh
// allocation.
extern "C" {
volatile std::atomic<uint32_t> kyty_local_image_pool_mode {0};
}
namespace {
struct PooledImage {
	std::array<uint32_t, 10> key {};
	VkImage                  image      = VK_NULL_HANDLE;
	VmaAllocation            allocation = nullptr;
	uint64_t                 bytes      = 0;
	// (When it was pooled: the periodic RAM flush takes only images no CreateImage reused for a while.)
	std::chrono::steady_clock::time_point pooled_at = std::chrono::steady_clock::now();
};
std::vector<PooledImage> g_image_pool;

std::array<uint32_t, 10> ImagePoolKey(const vk::ImageCreateInfo& info) {
	return {static_cast<uint32_t>(static_cast<VkImageCreateFlags>(info.flags)), static_cast<uint32_t>(info.imageType),
	        static_cast<uint32_t>(info.format), info.extent.width, info.extent.height, info.extent.depth,
	        info.mipLevels, info.arrayLayers,
	        static_cast<uint32_t>(info.samples) | (static_cast<uint32_t>(info.tiling) << 16u),
	        static_cast<uint32_t>(static_cast<VkImageUsageFlags>(info.usage))};
}
std::array<uint32_t, 10> ImagePoolKey(const VulkanImage& image) {
	vk::ImageCreateInfo info {};
	info.flags       = image.flags;
	info.imageType   = image.image_type;
	info.format      = image.format;
	info.extent      = image.extent;
	info.mipLevels   = image.mip_levels;
	info.arrayLayers = image.layers;
	info.samples     = static_cast<vk::SampleCountFlagBits>(image.samples);
	info.tiling      = vk::ImageTiling::eOptimal;
	info.usage       = image.usage;
	return ImagePoolKey(info);
}
} // namespace

static void DestroyImagePool(VmaAllocator allocator) {
	std::scoped_lock lock(g_image_pool_mutex);
	for (const auto& pooled: g_image_pool) {
		vmaDestroyImage(allocator, pooled.image, pooled.allocation);
	}
	g_image_pool.clear();
	g_image_pool_bytes = 0;
}

// The periodic RAM flush's share (ram-guard.h): up to `budget` bytes of the images pooled 5 s ago or more; the pool's lock is
// held only to take them out (CreateImage takes it), the images are destroyed after (the GPU is done with them).
static uint64_t FlushImagePool(uint64_t budget, uint64_t* held_us) {
	std::vector<PooledImage> taken;
	uint64_t                 bytes     = 0;
	VmaAllocator             allocator = nullptr;
	{
		const auto start = std::chrono::steady_clock::now();
		std::scoped_lock lock(g_image_pool_mutex);
		allocator = g_report_allocator;
		if (allocator == nullptr) return 0;
		// (Pooled 5 s ago or more and not reused since: not one of the transient targets the game makes every frame,
		// which would only be made again at once.)
		const auto old = start - std::chrono::seconds(5);
		for (auto it = g_image_pool.begin(); it != g_image_pool.end() && bytes < budget;) {
			if (it->pooled_at > old) {
				++it;
				continue;
			}
			bytes += it->bytes;
			taken.push_back(*it);
			it = g_image_pool.erase(it);
		}
		g_image_pool_bytes -= std::min(g_image_pool_bytes, bytes);
		if (held_us != nullptr)
			*held_us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	}
	for (const auto& pooled: taken) vmaDestroyImage(allocator, pooled.image, pooled.allocation);
	return bytes;
}

static std::atomic<uint64_t> g_vram_saver_counters[VramCounterCount] {};
void VramSaverAdd(int counter, uint64_t value) {
	if (counter >= 0 && counter < VramCounterCount) g_vram_saver_counters[counter].fetch_add(value, std::memory_order_relaxed);
}
uint64_t VramSaverGet(int counter) {
	return counter >= 0 && counter < VramCounterCount ? g_vram_saver_counters[counter].load(std::memory_order_relaxed) : 0;
}

bool GraphicContext::VramSaver() const {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_VRAM_SAVER");
		return env == nullptr || std::strcmp(env, "0") != 0;
	}();
	return enabled && small_video_memory;
}

// (DESTRUIR-HILO, KYTY_DESTRUIR_HILO; por defecto 0 desde 04:06, =1 lo enciende) Log 20261125-021048 (KYTY_TIRON3, mapa de la
// compilacion de las 02:07): en los fotogramas de 62.2 y 63.8 ms el hilo de render pasaba el 40 % en vkDestroyImage de
// imagenes dispersas (DestroySparseImage, con el Stream::Drain de la grabacion que fuerza esa llamada directa) y el
// 20-29 % en vmaDestroyImage -> FreeVulkanMemory (driver, win32u). Las imagenes llegan aqui ya retiradas (la GPU acabo con
// ellas: TextureCache::EraseRetired / RetireImage): destruirlas puede hacerse en otro hilo. Ese hilo no es productor de la
// grabacion (no vacia la del render) y VMA esta creado sin EXTERNALLY_SYNCHRONIZED (seguro entre hilos). Quien necesita la
// memoria ya liberada (ReclaimImageMemory, TrimImagePool en los reintentos de crear) llama a DrainImageDestroys.
namespace {
struct DestroyJob {
	VmaAllocator       allocator = nullptr;
	GraphicContext*    graphics  = nullptr;
	VkImage            image     = VK_NULL_HANDLE;
	VmaAllocation      allocation = nullptr;
	SparseImageMemory* sparse    = nullptr;
};
std::mutex              g_destroy_mutex;
std::condition_variable g_destroy_cv, g_destroy_done_cv;
std::deque<DestroyJob>  g_destroy_queue;
uint64_t                g_destroy_queued = 0, g_destroy_done = 0;
std::once_flag          g_destroy_start;
bool DestruirHiloOn() {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_DESTRUIR_HILO");
		const bool  set = v != nullptr && v[0] == '1'; // (04:06) apagado por defecto: con el encendido la VRAM pasaba el presupuesto (log 022328)
		std::printf("[destruir-hilo] destruccion de imagenes fuera del hilo de render: %s\n",
		            set ? "ON (KYTY_DESTRUIR_HILO=1)" : "off (KYTY_DESTRUIR_HILO=0)");
		std::fflush(stdout);
		return set;
	}();
	return on;
}
void RunDestroy(DestroyJob& job) {
	if (job.sparse != nullptr) {
		VulkanImage image;
		image.image  = vk::Image(job.image);
		image.sparse = job.sparse;
		DestroySparseImage(*job.graphics, image);
	} else if (job.image != VK_NULL_HANDLE) {
		vmaDestroyImage(job.allocator, job.image, job.allocation);
	}
}
void DestroyThread() {
	LocalPlatform::SetThreadName("Kyty.Destruir");
	for (;;) {
		DestroyJob job;
		{
			std::unique_lock lock(g_destroy_mutex);
			g_destroy_cv.wait(lock, [] { return !g_destroy_queue.empty(); });
			job = g_destroy_queue.front();
			g_destroy_queue.pop_front();
		}
		RunDestroy(job);
		{
			std::lock_guard lock(g_destroy_mutex);
			++g_destroy_done;
		}
		g_destroy_done_cv.notify_all();
	}
}
bool QueueDestroy(const DestroyJob& job) {
	if (!DestruirHiloOn()) return false;
	std::call_once(g_destroy_start, [] { std::thread(DestroyThread).detach(); });
	{
		std::lock_guard lock(g_destroy_mutex);
		g_destroy_queue.push_back(job);
		++g_destroy_queued;
	}
	g_destroy_cv.notify_one();
	return true;
}
} // namespace

// Waits until every image destruction queued so far is done (their memory is free).
void DrainImageDestroys() {
	std::unique_lock lock(g_destroy_mutex);
	const uint64_t target = g_destroy_queued;
	g_destroy_done_cv.wait(lock, [&] { return g_destroy_done >= target; });
}

void GraphicContext::TrimImagePool() {
	DrainImageDestroys(); // (DESTRUIR-HILO: the retries that call this want the memory back)
	if (allocator != nullptr) DestroyImagePool(allocator);
}

// Out of video memory: the first few times say so in the log, with the heaps' budgets.
void GraphicContext::ReportMemoryFallback(const char* what, uint64_t bytes) const {
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) >= 8) return;
	std::printf("Vulkan: video memory full, %s (%" PRIu64 " bytes)\n", what, bytes);
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < GetPhysicalDeviceMemoryProperties().memoryHeapCount; i++) {
		std::printf("  heap %u: usage %" PRIu64 " MiB of budget %" PRIu64 " MiB\n", i, budgets[i].usage >> 20u,
		            budgets[i].budget >> 20u);
	}
	std::fflush(stdout);
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);
	const int emu_owner = VramEmu::TakeNext(VramEmu::kCacheImg); // (VRAM-EMU) dueno: el de la macro del llamador

	if (kyty_local_image_pool_mode.load(std::memory_order_relaxed) != 0 && image_info.pNext == nullptr &&
	    image_info.tiling == vk::ImageTiling::eOptimal &&
	    image_info.initialLayout == vk::ImageLayout::eUndefined) {
		const auto key = ImagePoolKey(image_info);
		std::scoped_lock lock(g_image_pool_mutex);
		for (auto it = g_image_pool.begin(); it != g_image_pool.end(); ++it) {
			if (it->key != key) {
				continue;
			}
			image.image      = it->image;
			image.allocation = it->allocation;
			g_image_pool_bytes -= it->bytes;
			g_image_pool.erase(it);
			image.format     = image_info.format;
			image.image_type = image_info.imageType;
			image.extent     = image_info.extent;
			image.layers     = image_info.arrayLayers;
			image.mip_levels = image_info.mipLevels;
			image.samples    = static_cast<uint32_t>(image_info.samples);
			image.usage      = image_info.usage;
			image.flags      = image_info.flags;
			image.state      = {.layout = image_info.initialLayout};
			image.subresource_states.clear();
			VramEmu::Track(allocator, image.allocation, emu_owner); // (VRAM-EMU) sale del pool de imagenes
			return true;
		}
	}

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	// On a GPU that cannot hold the game's working set (small_video_memory: Boletaria uses ~11 GB), an image only
	// sampled and copied (a block-compressed texture) goes to the driver's memory type without flags once half the
	// budget is in use: NVIDIA places it in video memory while there is room and in system memory after, so render
	// targets, depth, storage images and buffers, which the GPU reads and writes every frame, keep video memory (it
	// fails device-local allocations past its video memory: what came last went to system memory, 8 GB at 1-1:
	// 22 fps, textures yielding: ~45). It counts that type under the video memory heap: on 8 GB, half is reached
	// before the first texture.
	const bool texture = IsTextureUsage(image_info.usage);
	// KYTY_LOWMEM_VRAM=1: no redirect. The plain type's allocations are a second pool outside the caches' budget
	// (2593 MiB of it at 4096 budget on a simulated 8 GB card, the process's private commit +4.6 GB, NVIDIA charging
	// it to commit); device-local textures stay under the budget the caches collect to.
	// (GESTOR-MEM) Con el gestor no hay segundo pool: esas texturas "plain" (833-976 MiB medidos) NVIDIA las dejaba en
	// VRAM fuera de todo presupuesto y ademas las cargaba al commit (+4,6 GB). Ahora van device-local, cuentan en el
	// presupuesto y el gestor las expulsa por prioridad (tienen respaldo en la RAM del juego).
	if (texture && small_video_memory && !LocalPlatform::LowMemVram() && !GestorMem::On() && !KYTY_VMA_TEXSTREAM() &&
	    GetDeviceMemoryUsage() >= GetTotalMemoryBudget(false) / 2) {
		alloc_info.requiredFlags = 0;
		alloc_info.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	vk::Result result = vk::Result::eSuccess;
	const auto create = [&] {
		vk::Image::CType native_image = VK_NULL_HANDLE;
		result                        = static_cast<vk::Result>(
		    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
		                   &alloc_info, &native_image, &image.allocation, nullptr));
		image.image = native_image;
		return result == vk::Result::eSuccess;
	};
	// Out of video memory: without the freed images kept for reuse, then in system memory (slower
	// to sample, but the game goes on).
	if (!create()) {
		TrimImagePool();
		if (!create()) {
			alloc_info.requiredFlags  = 0;
			alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			bool created              = create();
			VmaAllocationInfo allocated {};
			if (created) vmaGetAllocationInfo(allocator, image.allocation, &allocated);
			// Refused even in system memory: one release pass (MemGuard: caches, pools, the heaps) and a last try.
			if (!created && MemGuard::ShouldRetry("an image (vmaCreateImage)", 0, 0)) {
				TrimImagePool();
				created = create();
				if (created) vmaGetAllocationInfo(allocator, image.allocation, &allocated);
			}
			if (created) {
				VramSaverAdd(VramFallbackHost, 1);
				ReportMemoryFallback("an image is in system memory", allocated.size);
			} else {
				// (Which memory ran out: OUT_OF_HOST_MEMORY is the system's commit, OUT_OF_DEVICE_MEMORY the GPU's.)
				const auto what = "an image could not be created (" + vk::to_string(result) + ")";
				ReportMemoryFallback(what.c_str(), 0);
				VramSaverAdd(VramFailed, 1);
				return false;
			}
		}
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();
	VramEmu::Track(allocator, image.allocation, emu_owner); // (VRAM-EMU)

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	// (ram1b, KYTY_RAM_NATIVE_TEX) A sparse image: no VMA allocation, never pooled; its layers' memory freed with it.
	if (image.sparse != nullptr) {
		if (QueueDestroy({.graphics = this, .image = static_cast<VkImage>(image.image), .sparse = image.sparse})) { // (DESTRUIR-HILO)
			image.sparse     = nullptr;
			image.image      = nullptr;
			image.allocation = nullptr;
			return;
		}
		DestroySparseImage(*this, image);
		return;
	}
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);
	VramEmu::Untrack(image.allocation); // (VRAM-EMU) al pool de imagenes (cuenta aparte) o destruida

	if (kyty_local_image_pool_mode.load(std::memory_order_relaxed) != 0) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		std::scoped_lock lock(g_image_pool_mutex);
		g_image_pool.push_back({ImagePoolKey(image), image.image, image.allocation, allocation_info.size});
		g_image_pool_bytes += allocation_info.size;
		while (g_image_pool_bytes > g_image_pool_limit && !g_image_pool.empty()) {
			auto& oldest = g_image_pool.front();
			vmaDestroyImage(allocator, oldest.image, oldest.allocation);
			g_image_pool_bytes -= oldest.bytes;
			g_image_pool.erase(g_image_pool.begin());
		}
		image.image      = nullptr;
		image.allocation = nullptr;
		return;
	}
	if (!QueueDestroy({.allocator = allocator, .image = static_cast<VkImage>(image.image), .allocation = image.allocation})) // (DESTRUIR-HILO)
		vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}


// (VRAM-TOPE) Tamano de la tarjeta para la tabla de topes (vram-tope.h): la simulada con KYTY_SIMULATE_VRAM_MB, si no
// el mayor heap device-local real.
uint64_t VramCardBytes() { return g_simulated_vram != 0 ? g_simulated_vram : g_real_vram_heap; }

// (VRAM-DRV) Desglose medido de la VRAM del proceso, cada 5 s (textureCache.cpp). Todo vkAllocateMemory lo cuenta el
// gancho de arriba por categoria; lo que el proceso usa sin ninguna es memoria interna del driver.
// (RECURSOS-JUSTOS) Cada 10 s: lo que el emulador tiene en RAM y VRAM, por consumidor (KYTY_RECURSOS_JUSTOS=0: nada).
static void RecursosReport(const GraphicContext& g) {
	if (!VramRecursos::On() || g.allocator == nullptr) return;
	static std::atomic<int64_t> last {0};
	const int64_t now  = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	int64_t       prev = last.load(std::memory_order_relaxed);
	if (now - prev < 10 || !last.compare_exchange_strong(prev, now)) return;
	uint64_t video = 0, budget = 0, shared = 0;
	if (!g.QueryLocalVideoMemory(&video, &budget)) return;
	(void)LocalPlatform::QuerySharedGpuMemory(&shared);
	VmaTotalStatistics st {};
	vmaCalculateStatistics(g.allocator, &st);
	uint64_t local = 0, used_local = 0, plain = 0, host = 0;
	uint32_t blocks = 0;
	const auto& props = g.GetPhysicalDeviceMemoryProperties();
	for (uint32_t t = 0; t < props.memoryTypeCount; t++) {
		const auto  f = props.memoryTypes[t].propertyFlags;
		const auto& s = st.memoryType[t].statistics;
		if (f & vk::MemoryPropertyFlagBits::eDeviceLocal) {
			local += s.blockBytes;
			used_local += s.allocationBytes;
			blocks += s.blockCount;
		} else if (f & vk::MemoryPropertyFlagBits::eHostVisible) host += s.blockBytes;
		else plain += s.blockBytes;
	}
	const uint64_t plain_sys = std::min(plain, shared - std::min(shared, host));
	const uint64_t plain_vid = plain - plain_sys;
	uint64_t       pool_bytes = 0;
	{
		std::scoped_lock lock(g_image_pool_mutex);
		pool_bytes = g_image_pool_bytes;
	}
	VmaStatistics buffers {};
	if (g.game_buffer_pool != nullptr) vmaGetPoolStatistics(g.allocator, g.game_buffer_pool, &buffers);
	const uint64_t images = used_local - std::min(used_local, pool_bytes + buffers.allocationBytes);
	int64_t fuera = 0;
	for (int c = 0; c < VramDrv::CatCount; c++)
		if (c != VramDrv::CatVma && c != VramDrv::CatLastre) fuera += VramDrv::g_bytes[c][0].load(std::memory_order_relaxed);
	const int64_t nativas  = VramDrv::g_bytes[VramDrv::CatSparseImg][0].load(std::memory_order_relaxed);
	const int64_t interno  = static_cast<int64_t>(video) - static_cast<int64_t>(local + plain_vid) - fuera;
	const auto    mem      = LocalPlatform::ProcessMemory();
	const uint64_t gfx = VramRecursos::g_graphics.load(), cmp = VramRecursos::g_compute.load();
	const uint64_t exp = VramRecursos::g_expulsados.load(), reh = VramRecursos::g_rehechos.load();
	const uint64_t vivos = gfx + cmp - std::min(gfx + cmp, exp - std::min(exp, reh));
	auto mib = [](int64_t b) { return static_cast<long long>((b < 0 ? 0 : b) >> 20); };
	std::printf("[recursos] 10 s, MiB. VRAM: proceso %lld / presupuesto %lld = VMA %lld en %u bloques (bufferes del juego %lld, "
	            "reutilizacion %lld, imagenes/RT/anillos %lld, hueco %lld) + plain en VRAM %lld + texturas nativas %lld + otro "
	            "fuera de VMA %lld + driver (pipelines, descriptores, swapchain) %lld. RAM: privado %lld, en RAM %lld; Vulkan en "
	            "RAM: host-visible %lld + plain %lld. Pipelines: %llu vivos (graficos %llu, compute %llu; expulsados %llu, "
	            "rehechos %llu; de la precarga expulsables %llu)\n",
	            mib(video), mib(budget), mib(local), blocks, mib(buffers.allocationBytes), mib(pool_bytes), mib(images),
	            mib(local - std::min(local, used_local)), mib(plain_vid), mib(nativas), mib(fuera - nativas), mib(interno),
	            mib(mem.private_bytes), mib(mem.working_set), mib(host), mib(plain_sys),
	            static_cast<unsigned long long>(vivos), static_cast<unsigned long long>(gfx), static_cast<unsigned long long>(cmp),
	            static_cast<unsigned long long>(exp), static_cast<unsigned long long>(reh),
	            static_cast<unsigned long long>(VramRecursos::g_warm_expulsables.load()));
	std::fflush(stdout);
}

void VramDrvReport(const GraphicContext& g) {
	RecursosReport(g);
	if (!VramDrv::On() || g.allocator == nullptr) return;
	uint64_t video = 0, budget = 0, shared = 0;
	if (!g.QueryLocalVideoMemory(&video, &budget)) return;
	(void)LocalPlatform::QuerySharedGpuMemory(&shared);
	VmaTotalStatistics statistics {};
	vmaCalculateStatistics(g.allocator, &statistics);
	uint64_t local = 0, used_local = 0, plain = 0, host = 0;
	uint32_t local_blocks = 0, local_allocs = 0;
	const auto& properties = g.GetPhysicalDeviceMemoryProperties();
	for (uint32_t type = 0; type < properties.memoryTypeCount; type++) {
		const auto  flags = properties.memoryTypes[type].propertyFlags;
		const auto& st    = statistics.memoryType[type].statistics;
		if (flags & vk::MemoryPropertyFlagBits::eDeviceLocal) {
			local += st.blockBytes;
			used_local += st.allocationBytes;
			local_blocks += st.blockCount;
			local_allocs += st.allocationCount;
		} else if (flags & vk::MemoryPropertyFlagBits::eHostVisible) host += st.blockBytes;
		else plain += st.blockBytes;
	}
	const uint64_t plain_in_system = std::min(plain, shared - std::min(shared, host));
	const uint64_t plain_in_video  = plain - plain_in_system;
	uint64_t       pool_bytes      = 0;
	{
		std::scoped_lock lock(g_image_pool_mutex);
		pool_bytes = g_image_pool_bytes;
	}
	VmaStatistics buffers {};
	if (g.game_buffer_pool != nullptr) vmaGetPoolStatistics(g.allocator, g.game_buffer_pool, &buffers);
	const uint64_t images = used_local - std::min(used_local, pool_bytes + buffers.allocationBytes);
	auto mib = [](int64_t b) { return static_cast<long long>((b < 0 ? 0 : b) >> 20); };
	auto lv  = [](int c) { return VramDrv::g_bytes[c][0].load(std::memory_order_relaxed); };
	auto hv  = [](int c) { return VramDrv::g_bytes[c][1].load(std::memory_order_relaxed); };
	auto cnt = [](int c) { return static_cast<long long>(VramDrv::g_count[c][0].load(std::memory_order_relaxed)); };
	auto pk  = [](int c) { return VramDrv::g_peak[c][0].load(std::memory_order_relaxed); };
	// Fuera de VMA, device-local (lo que el gancho vio, sin el lastre de la simulacion).
	int64_t fuera = 0;
	for (int c = 0; c < VramDrv::CatCount; c++)
		if (c != VramDrv::CatVma && c != VramDrv::CatLastre) fuera += lv(c);
	const int64_t interno = static_cast<int64_t>(video) - static_cast<int64_t>(local + plain_in_video) - fuera;
	const uint64_t vbudget = g.GetTotalMemoryBudget(false);
	std::printf("[vram-drv] 5 s, MiB: proceso %lld / presupuesto driver %lld ([vram] %lld, techo %llu %% = %lld); "
	            "VMA device-local %lld en %u bloques (gancho %lld) = bufferes del juego %lld + reutilizacion %lld + imagenes y "
	            "resto (texturas, RT, FSR4/FG, anillos) %lld + hueco en bloques %lld; plain en VRAM %lld; fuera de VMA: "
	            "texturas nativas %lld en %lld (pico %lld), bufferes sparse %lld en %lld, vramStore %lld, otro %lld en %lld; "
	            "interno del driver (pipelines, descriptores, swapchain; sin vkAllocateMemory) %lld; no VRAM: importada "
	            "del juego %lld, host-visible %lld\n",
	            mib(video), mib(budget), mib(vbudget), static_cast<unsigned long long>(VramDrv::TechoPct),
	            mib(vbudget * VramDrv::TechoPct / 100), mib(local), local_blocks, mib(lv(VramDrv::CatVma)),
	            mib(buffers.allocationBytes), mib(pool_bytes), mib(images), mib(local - std::min(local, used_local)),
	            mib(plain_in_video), mib(lv(VramDrv::CatSparseImg)), cnt(VramDrv::CatSparseImg), mib(pk(VramDrv::CatSparseImg)),
	            mib(lv(VramDrv::CatSparseBuf)), cnt(VramDrv::CatSparseBuf), mib(lv(VramDrv::CatVramStore)),
	            mib(lv(VramDrv::CatOtro)), cnt(VramDrv::CatOtro), mib(interno), mib(hv(VramDrv::CatHostImport)), mib(host));
	(void)local_allocs;
	std::fflush(stdout);
}

} // namespace Libs::Graphics
