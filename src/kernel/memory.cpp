#include "kernel/memory.h"
#include "local-platform.h"
#include "live-census.h"
#include "live-counters.h"
#include "live-trace.h"
#include "time-census.h"
#include "writeback-verify.h"
#include "fresh-memory.h"
#include "guest-lazy-commit.h"
#include "f0-measure.h" // (f0, KYTY_F0: memoryAddressSpace.inc)
#include "native/g1-probe.h" // (g1, KYTY_G1_PROBE: memoryAddressSpace.inc)
#include "frame-nativo.h" // (FRAME-NATIVO) protection changes by label (memoryAddressSpace.inc)
#include "frame-nativo2.h" // (FRAME-NATIVO2 v2) imported arenas excluded from protection (memoryAddressSpace.inc)
#include "mem-guard.h"
#include "ram-guard.h"
#include "common/guardedCopy.h" // (raiz2: NativeGuardedRead, the read probes)
#include "raiz.h" // (raiz1: KYTY_RAIZ, private guest memory)
#include "parpadeo6.h" // (SIN-VIGILAR-GANCHO: "asentados sin vigilante" en la linea [parpadeo6])
#include "raiz-wb-guard.h" // (crash-ba2b1f: KYTY_RAIZ_WB_GUARD)
#include "arena-dueno.h" // (ARENA-DUENO: KYTY_ARENA_DUENO=1, los [wb-guard] por instancia duena)
#include "tex-guarda.h" // (TEX-GUARDA: incluido fuera del namespace; memoryAddressSpace.inc lo usa)

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/magicEnum.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/cache/gpuResourceManager.h"
#include "graphics/host_gpu/renderer/demonsSouls.h" // (ram1: KYTY_DS_FREED_DISCARD)
#include "libs/errno.h"
#include "libs/libs.h"
#include "loader/demonsSoulsMemMap.h" // (ram2: PageAllocatorFree for the "[ram]" line)
#include "loader/demonsSoulsTextureProbe.h"
#include "native-resource-state.h"
#include "speculation-state.h"

#include <algorithm>
#include <unordered_set>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <immintrin.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <winioctl.h> // FSCTL_SET_SPARSE (KYTY_BACKING_FILE)
#include <psapi.h>    // (ram6: K32QueryWorkingSetEx, KYTY_DS_FREED_RELEASE)
#include <deque>      // (ram9: StoreTrim)
#include <random>     // (ram9: StoreTrim)
#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif
#ifndef MEM_COALESCE_PLACEHOLDERS
#define MEM_COALESCE_PLACEHOLDERS 0x00000001
#endif
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::LibKernel::Memory {

extern "C" {
// 1: small host-copy reads of guest backing take a lock-free read transaction
// with a cached translation instead of the backing mutex.
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_backing_read_mode {0};
// 1: the other parts of a view a partial unmap maps again (Windows) get the GPU trackers' protection of their watched
// pages at once, before the address-space lock is released (GuestBackingStore::ProtectRemapped). 0: given back after
// the lock (ReapplyRemappedViews): a CPU write to a watched page in between went unseen (a GPU copy left stale, or a
// GPU-owned page later written back over the write).
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_protective_remap {1};
// 1: a partial unmap of a view (Windows) leaves the range parked in the view instead of unmapping the whole view and
// mapping its other parts again (GuestBackingStore::UnmapOne); 2: parked no-access (a stray access faults, for
// debugging). 0: views with nothing parked yet are split at once, as before (a view holding parked ranges keeps parking).
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_park_unmaps {1};
// (Diagnostic: microseconds UnmapBacking waits after it mapped views again, before ReapplyRemappedViews: the window
// kyty_local_protective_remap 0 leaves, widened.)
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_remap_window_us {0};
}

namespace VirtualMemory = Common::VirtualMemory;

LIB_NAME("libkernel", "libkernel");

constexpr int PROT_CPU_READ  = 0x01;
constexpr int PROT_CPU_WRITE = 0x02;
constexpr int PROT_CPU_EXEC  = 0x04;
constexpr int PROT_GPU_READ  = 0x10;
constexpr int PROT_GPU_WRITE = 0x20;

enum class GpuAccessMode { NoAccess, Read, Write, ReadWrite };

constexpr uint64_t PAGE_TABLE_POOL_SIZE   = 4ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t PAGE_TABLE_GRANULARITY = 2ull * 1024ull * 1024ull;
constexpr int      PAGE_TABLE_POOL_ENTRIES =
    static_cast<int>(PAGE_TABLE_POOL_SIZE / PAGE_TABLE_GRANULARITY);
constexpr uint64_t DEFAULT_FLEXIBLE_MEMORY_SIZE = 1ull * 1024ull * 1024ull * 1024ull;

static uint64_t                      g_flexible_memory_size        = DEFAULT_FLEXIBLE_MEMORY_SIZE;
static bool                          g_flexible_memory_size_frozen = false;
static Graphics::GpuResourceManager* g_gpu_resources               = nullptr;

// Every out-of-memory answer to the guest, on stdout (the first 64): a game that does not expect one
// can carry a null pointer into its data and crash much later, elsewhere.
static int NoMemory(const char* function, int line) {
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
		std::printf("Memory: ENOMEM to the guest from %s (memory.cpp:%d)\n", function, line);
		std::fflush(stdout);
	}
	return KERNEL_ERROR_ENOMEM;
}

// SetAvailableDirectClamp (memory.h): written once at load, before any guest thread runs.
namespace AvailableClamp {
std::atomic<uint64_t> g_return {0};
uint64_t              g_limit = 0, g_trace_begin = 0, g_trace_end = 0, g_base = 0;
std::atomic<uint32_t> g_printed {0};

bool Traced(uint64_t caller) {
	return caller >= g_trace_begin && caller < g_trace_end;
}

void Print(const char* what, uint64_t caller, uint64_t len, uint64_t phys, const char* outcome) {
	if (g_printed.fetch_add(1, std::memory_order_relaxed) >= 32) return;
	std::printf("[port] ds-alloc: eboot+0x%llx %s len=0x%llx (%llu MiB) phys=0x%llx: %s\n",
	            static_cast<unsigned long long>(caller - g_base), what, static_cast<unsigned long long>(len),
	            static_cast<unsigned long long>(len >> 20u), static_cast<unsigned long long>(phys), outcome);
	std::fflush(stdout);
}
} // namespace AvailableClamp

// (MAPA-GRANDE) KYTY_MAPA_GRANDE=1 (apagado por defecto; solo mide): cada memoria del juego de 64 MiB o mas que el
// emulador pone en una direccion del juego, con la funcion del juego que la pidio. Motivo: el bloque 0x33fc00000 (320 MiB,
// 301 en RAM, log 20261124-221631) esta en direcciones del juego pero fuera de los 41 asignadores que registra el motor
// (tabla eboot+0x2ea0df0, Ghidra 0x7f8a10); un bloque de 64 KiB del PlatformPageAllocator (0x801d00) no sale entero en
// un solo mapeo, asi que es un mapeo directo grande. Cada funcion de mapeo del juego guarda su direccion de vuelta al
// entrar (el primer nivel: las llamadas anidadas del emulador no la pisan) y el punto que hace la memoria nativa
// (ReplacePlaceholderNative y la memoria privada) imprime:
//   [mapa-grande] 0x33fc00000 320 MiB <- sceKernelMapDirectMemory desde eboot+0x...; en la pila: eboot+0x.. eboot+0x..
// (los marcos de la pila: valores de la pila del hilo que caen en el codigo del eboot, los primeros 8, para ver quien
// esta por encima del envoltorio del motor).
namespace MapaGrande {
constexpr uint64_t    kEboot = 0x900000000ull, kEbootSize = 0x3000000ull;
thread_local uint64_t t_caller = 0;
thread_local const char* t_what = nullptr;
[[nodiscard]] bool On() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_MAPA_GRANDE");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}
struct Scope {
	bool own;
	Scope(const char* what, uint64_t caller) noexcept: own(t_caller == 0) {
		if (own) {
			t_caller = caller;
			t_what   = what;
		}
	}
	~Scope() {
		if (own) {
			t_caller = 0;
			t_what   = nullptr;
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
};
void Note(uint64_t vaddr, uint64_t size) noexcept {
	if (size < (uint64_t {64} << 20u) || !On()) return;
	char   text[600];
	size_t used = 0;
	const auto add = [&](const char* format, auto... args) {
		if (used + 1 >= sizeof(text)) return;
		const int n = std::snprintf(text + used, sizeof(text) - used, format, args...);
		if (n > 0) used = std::min(sizeof(text) - 1, used + static_cast<size_t>(n));
	};
	const uint64_t caller = t_caller;
	if (caller >= kEboot && caller < kEboot + kEbootSize)
		add("[mapa-grande] 0x%llx %llu MiB <- %s desde eboot+0x%llx; en la pila:", static_cast<unsigned long long>(vaddr),
		    static_cast<unsigned long long>(size >> 20u), t_what != nullptr ? t_what : "?",
		    static_cast<unsigned long long>(caller - kEboot));
	else
		add("[mapa-grande] 0x%llx %llu MiB <- %s desde 0x%llx (fuera del eboot); en la pila:", static_cast<unsigned long long>(vaddr),
		    static_cast<unsigned long long>(size >> 20u), t_what != nullptr ? t_what : "(sin funcion de mapeo)",
		    static_cast<unsigned long long>(caller));
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// (the stack the guest call runs on, host or guest: up to the end of the committed region that holds this frame)
	const auto*              at = static_cast<const uint64_t*>(__builtin_frame_address(0));
	MEMORY_BASIC_INFORMATION region {};
	uint64_t                 high = 0;
	if (VirtualQuery(at, &region, sizeof(region)) == sizeof(region))
		high = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
	int got = 0;
	for (uint64_t p = reinterpret_cast<uint64_t>(at); p + 8 <= high && p < reinterpret_cast<uint64_t>(at) + 0x4000 && got < 8; p += 8) {
		const uint64_t v = *reinterpret_cast<const uint64_t*>(p);
		if (v > kEboot && v < kEboot + kEbootSize) {
			add(" eboot+0x%llx", static_cast<unsigned long long>(v - kEboot));
			got++;
		}
	}
#endif
	std::printf("%s\n", text);
	std::fflush(stdout);
}
} // namespace MapaGrande
#define KYTY_MAPA_GRANDE_SCOPE(what) \
	MapaGrande::Scope mapa_grande_scope((what), reinterpret_cast<uint64_t>(__builtin_return_address(0)))

void SetAvailableDirectClamp(uint64_t return_address, uint64_t limit, uint64_t trace_begin, uint64_t trace_end,
                             uint64_t image_base) {
	AvailableClamp::g_limit       = limit;
	AvailableClamp::g_trace_begin = trace_begin;
	AvailableClamp::g_trace_end   = trace_end;
	AvailableClamp::g_base        = image_base;
	AvailableClamp::g_return.store(return_address, std::memory_order_release);
}

// The latest GPU write-backs into guest memory, for crash reports: a pointer a game structure lost could
// have been overwritten by one.
struct WriteBackRecord {
	int64_t     ns;
	uint64_t    vaddr, size;
	const char* source;
};
static constexpr size_t      WriteBackRecords = 8192;
static WriteBackRecord        g_write_backs[WriteBackRecords];
static std::atomic<uint64_t> g_write_back_count {0};

static Graphics::GpuResourceManager& GetGpuResources() {
	EXIT_IF(g_gpu_resources == nullptr);
	return *g_gpu_resources;
}

static bool IsGpuAddressRange(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t GPU_ADDRESS_LIMIT = 1ull << 40u;
	return vaddr != 0 && size != 0 && vaddr < GPU_ADDRESS_LIMIT && size < GPU_ADDRESS_LIMIT - vaddr;
}

static void MapGpuRange(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return;
	}
	const auto start = __rdtsc();
	GetGpuResources().MapMemory(vaddr, size);
	LiveCounters::Add(LiveCounters::GpuMapKt, (__rdtsc() - start) >> 10u);
}

// `releasing`: the range is unmapped or decommitted right after (GpuResourceManager::UnmapMemory).
static void UnmapGpuRange(uint64_t vaddr, uint64_t size, bool releasing = false) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return;
	}
	const auto start = __rdtsc();
	GetGpuResources().UnmapMemory(vaddr, size, releasing);
	LiveCounters::Add(LiveCounters::GpuUnmapKt, (__rdtsc() - start) >> 10u);
}

// A releasing UnmapGpuRange left the pages' host protection to the unmap; when the unmap fails
// they stay mapped, so they get their trackers' protection back.
static void RestoreGpuRangeProtection(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return;
	}
	GetGpuResources().SyncProtection(vaddr, size);
}

static bool DecodeMemoryProtection(int prot, VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	EXIT_IF(mode == nullptr);
	EXIT_IF(gpu_mode == nullptr);

	bool cpu_read  = (prot & PROT_CPU_READ) != 0;
	bool cpu_write = (prot & PROT_CPU_WRITE) != 0;
	bool cpu_exec  = (prot & PROT_CPU_EXEC) != 0;
	bool gpu_read  = (prot & PROT_GPU_READ) != 0;
	bool gpu_write = (prot & PROT_GPU_WRITE) != 0;

	if ((prot &
	     (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC | PROT_GPU_READ | PROT_GPU_WRITE)) == 0 &&
	    prot != 0) {
		return false;
	}

	if (gpu_read && gpu_write) {
		*gpu_mode = GpuAccessMode::ReadWrite;
	} else if (gpu_read) {
		*gpu_mode = GpuAccessMode::Read;
	} else if (gpu_write) {
		*gpu_mode = GpuAccessMode::Write;
	} else {
		*gpu_mode = GpuAccessMode::NoAccess;
	}

	bool host_read  = cpu_read || gpu_read;
	bool host_write = cpu_write || gpu_write;

	if (host_write) {
		host_read = true;
	}

	*mode = VirtualMemory::Mode::NoAccess;
	if (cpu_exec) {
		*mode = (host_write ? VirtualMemory::Mode::ExecuteReadWrite
		                    : (host_read ? VirtualMemory::Mode::ExecuteRead
		                                 : VirtualMemory::Mode::Execute));
	} else if (host_write) {
		*mode = VirtualMemory::Mode::ReadWrite;
	} else if (host_read) {
		*mode = VirtualMemory::Mode::Read;
	}

	return true;
}

static void CopyVirtualRangeName(char* dst, const char* name) {
	EXIT_IF(dst == nullptr);

	std::memset(dst, 0, KERNEL_MAXIMUM_NAME_LENGTH);

	if (name != nullptr) {
		std::strncpy(dst, name, KERNEL_MAXIMUM_NAME_LENGTH - 1);
	}
}

static bool VirtualRangesOverlap(uint64_t left_start, uint64_t left_size, uint64_t right_start,
                                 uint64_t right_size) {
	if (left_size == 0 || right_size == 0) {
		return false;
	}

	auto left_end = (UINT64_MAX - left_start < left_size ? UINT64_MAX : left_start + left_size);
	auto right_end =
	    (UINT64_MAX - right_start < right_size ? UINT64_MAX : right_start + right_size);

	return left_start < right_end && right_start < left_end;
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
static uint32_t g_test_backing_store_unmaps_before_failure = UINT32_MAX;
#endif

// (ram15, KYTY_NATIVE_TEXTURE_OFFLOAD) Called first by every window over guest memory (NativeGuardedAccess in the .inc):
// the texture offload gives back the dropped bytes a window would read (SetNativeWindowHook).
static std::atomic<native_window_hook_t> g_native_window_hook {nullptr};

#include "memoryAddressSpace.inc"

void ReleaseFrameGenAddressHole() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (g_frame_gen_hole != nullptr) {
		VirtualFree(std::exchange(g_frame_gen_hole, nullptr), 0, MEM_RELEASE);
		std::printf("Memory: 64 GiB above the guest released for the Vulkan device (KYTY_FG_ADDRESS_HOLE=0: none)\n");
		std::fflush(stdout);
	}
#endif
}

enum class VirtualRangeType {
	Reserved,
	PoolReserved,
	Direct,
	Flexible,
	Pooled,
	Stack,
	Code,
	Runtime,
};

static bool IsReservedRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Reserved || type == VirtualRangeType::PoolReserved;
}

static bool IsPooledRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Pooled || type == VirtualRangeType::PoolReserved;
}

static bool IsCommittedRangeType(VirtualRangeType type) {
	return !IsReservedRangeType(type);
}

static bool IsPrivateCommittedRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Stack || type == VirtualRangeType::Code ||
	       type == VirtualRangeType::Runtime;
}

// The guest's virtual ranges, sorted and disjoint, keyed by their start. A tree: a streamed texture pool leaves tens of
// thousands of ranges (one per direct mapping, never merged), and as a sorted vector each insertion or removal moved
// the ranges after it (~80 bytes each): up to ~10 ms per munmap, 70 ms of a frame when the game unmapped and mapped
// hundreds of pages at once, growing with play time (0.3-0.8 s stalls after world switches).
class VirtualRanges {
public:
	VirtualRanges() = default;
	~VirtualRanges() {
		for (auto& chunk: m_committed) delete[] chunk.load(std::memory_order_relaxed);
	}
	KYTY_CLASS_NO_COPY(VirtualRanges);

	struct Range {
		uint64_t         start          = 0;
		uint64_t         size           = 0;
		uint64_t         offset         = 0;
		int              protection     = 0;
		int              memory_type    = 0;
		VirtualRangeType type           = VirtualRangeType::Reserved;
		bool             disallow_merge = false;
		char             name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	// Diagnostics (live `ranges`): every range, one tab-separated row each.
	void Dump(FILE* file) {
		Common::LockGuard lock(m_mutex);
		std::fprintf(file, "start\tsize\toffset\tprotection\tmemory_type\ttype\tname\n");
		for (const auto& [start, range]: m_ranges)
			std::fprintf(file, "0x%llx\t0x%llx\t0x%llx\t0x%x\t%d\t%d\t%s\n", static_cast<unsigned long long>(range.start),
			             static_cast<unsigned long long>(range.size), static_cast<unsigned long long>(range.offset),
			             static_cast<unsigned>(range.protection), range.memory_type, static_cast<int>(range.type), range.name);
	}

	bool Add(uint64_t start, uint64_t size, uint64_t offset, int protection, int memory_type,
	         VirtualRangeType type, const char* name, bool disallow_merge = false) {
		Common::LockGuard lock(m_mutex);
		Mutation          mutation(*this);

		if (start == 0 || size == 0) {
			return false;
		}
		auto position = m_ranges.lower_bound(start);
		if ((position != m_ranges.end() &&
		     VirtualRangesOverlap(start, size, position->second.start, position->second.size)) ||
		    (position != m_ranges.begin() &&
		     VirtualRangesOverlap(start, size, std::prev(position)->second.start,
		                          std::prev(position)->second.size))) {
			return false;
		}

		Range r {};
		r.start          = start;
		r.size           = size;
		r.offset         = offset;
		r.protection     = protection;
		r.memory_type    = memory_type;
		r.type           = type;
		r.disallow_merge = disallow_merge;
		CopyVirtualRangeName(r.name, name);
		MergeAroundUnlocked(m_ranges.emplace_hint(position, start, r));
		if (IsCommittedRangeType(type)) MarkCommittedUnlocked(start, size, true);
		return true;
	}

	bool Remove(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);
		Mutation          mutation(*this);

		const auto position = m_ranges.find(start);
		if (position != m_ranges.end() && position->second.size == size) {
			if (IsCommittedRangeType(position->second.type)) MarkCommittedUnlocked(start, size, false);
			m_ranges.erase(position);
			return true;
		}
		return RemoveUnlocked(start, size);
	}

	bool HasOverlap(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);

		return FindOverlap(start, size) != nullptr;
	}

	bool QueryOverlap(uint64_t start, uint64_t size, Range* out) {
		EXIT_IF(out == nullptr);
		Common::LockGuard lock(m_mutex);

		const auto* overlap = FindOverlap(start, size);
		if (overlap == nullptr) {
			return false;
		}
		*out = *overlap;
		return true;
	}

	bool ReleaseReserved(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);
		Mutation          mutation(*this);

		// Ranges start at distinct addresses: only the one starting at `start` can match.
		const auto position = m_ranges.find(start);
		if (position != m_ranges.end() && position->second.size == size &&
		    IsReservedRangeType(position->second.type)) {
			m_ranges.erase(position);
		}
		return true;
	}

	bool ConsumeReserved(uint64_t start, uint64_t size,
	                     VirtualRangeType type = VirtualRangeType::Reserved) {
		Common::LockGuard lock(m_mutex);
		Mutation          mutation(*this);

		// Disjoint ranges: only the one containing `start` can contain the request.
		auto        end = End(start, size);
		const auto* r   = FindContainingUnlocked(start, type);
		if (r != nullptr && end <= End(r->start, r->size)) {
			RemoveUnlocked(start, size);
			return true;
		}

		return false;
	}

	bool ConsumeReservedSpan(uint64_t start, uint64_t size, Range* first_range = nullptr,
	                         VirtualRangeType type = VirtualRangeType::Reserved) {
		Common::LockGuard lock(m_mutex);
		Mutation          mutation(*this);

		if (size == 0) {
			return false;
		}

		auto current = start;
		auto end     = End(start, size);
		while (current < end) {
			const Range* candidate = FindContainingUnlocked(current, type);
			if (candidate == nullptr) {
				return false;
			}
			if (current == start && first_range != nullptr) {
				*first_range = *candidate;
			}
			current = std::min(end, End(candidate->start, candidate->size));
		}

		RemoveUnlocked(start, size);
		return true;
	}

	void Rename(uint64_t start, uint64_t size, const char* name) {
		Common::LockGuard lock(m_mutex);
		Mutation          mutation(*this);

		const auto position = m_ranges.find(start);
		if (position != m_ranges.end() && position->second.size == size) {
			CopyVirtualRangeName(position->second.name, name);
			MergeAroundUnlocked(position);
			return;
		}
		EditUnlocked(start, size, [name](Range* r) { CopyVirtualRangeName(r->name, name); });
	}

	void Protect(uint64_t start, uint64_t size, int protection) {
		Common::LockGuard lock(m_mutex);
		Mutation          mutation(*this);

		EditUnlocked(start, size, [protection](Range* r) { r->protection = protection; });
	}

	void SetMemoryType(uint64_t start, uint64_t size, int memory_type) {
		Common::LockGuard lock(m_mutex);
		Mutation          mutation(*this);

		EditUnlocked(start, size, [memory_type](Range* r) { r->memory_type = memory_type; });
	}

	bool Query(uint64_t addr, int flags, Range* out) {
		EXIT_IF(out == nullptr);

		Common::LockGuard lock(m_mutex);

		auto next = m_ranges.upper_bound(addr);
		if (next != m_ranges.begin()) {
			const auto& current = std::prev(next)->second;
			if (addr < End(current.start, current.size)) {
				*out = current;
				return true;
			}
		}
		if (flags != 1 || next == m_ranges.end()) {
			return false;
		}

		*out = next->second;
		return true;
	}

	bool QuerySpan(uint64_t start, uint64_t size, std::vector<Range>* out) {
		EXIT_IF(out == nullptr);

		Common::LockGuard lock(m_mutex);
		out->clear();
		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return false;
		}

		const auto end     = start + size;
		auto       current = start;
		// Ranges are sorted and disjoint: start at the last one beginning at or before `start`.
		auto first = m_ranges.upper_bound(start);
		if (first != m_ranges.begin()) {
			--first;
		}
		for (auto it = first; it != m_ranges.end(); ++it) {
			const auto& range     = it->second;
			const auto  range_end = End(range.start, range.size);
			if (range_end <= current) {
				continue;
			}
			if (range.start > current) {
				break;
			}

			Range part = range;
			part.start = current;
			part.size  = std::min(end, range_end) - current;
			if (part.type == VirtualRangeType::Direct) {
				part.offset += current - range.start;
			}
			out->push_back(part);
			current += part.size;
			if (current == end) {
				return true;
			}
		}

		out->clear();
		return false;
	}

	// The end of the committed virtual range `virtual_addr` lies in (0: none), itself, not the ranges after it.
	uint64_t MappingEnd(uint64_t virtual_addr) {
		thread_local std::array<std::pair<uint64_t, uint64_t>, 16> cache {};       // (begin, end) by address
		thread_local std::array<std::pair<uint64_t, uint64_t>, 16> cache_stamp {}; // (identity, epoch)
		const auto slot  = ((virtual_addr >> 20) ^ (virtual_addr >> 32)) & 15u;
		const auto epoch = m_epoch.load(std::memory_order_acquire);
		if (!(epoch & 1) && cache_stamp[slot] == std::pair {m_identity, epoch} && virtual_addr >= cache[slot].first &&
		    virtual_addr < cache[slot].second)
			return cache[slot].second;
		Common::LockGuard lock(m_mutex);
		auto vma = m_ranges.upper_bound(virtual_addr);
		if (vma == m_ranges.begin()) return 0;
		--vma;
		const auto& range   = vma->second;
		const auto  vma_end = End(range.start, range.size);
		if (virtual_addr < range.start || virtual_addr >= vma_end || !IsCommittedRangeType(range.type)) return 0;
		cache[slot]       = {range.start, vma_end};
		cache_stamp[slot] = {m_identity, m_epoch.load(std::memory_order_relaxed)};
		return vma_end;
	}

	uint64_t ClampRangeSize(uint64_t virtual_addr, uint64_t size) {
		if (virtual_addr == 0 || size == 0 || size > UINT64_MAX - virtual_addr) {
			return 0;
		}
		if (FullyCommitted(virtual_addr, size)) {
			// KYTY_VERIFY_COMMITTED=1: the locked walk's answer compared (COMMITTED-MISMATCH lines; none over two
			// rounds of the user's 1-1 route).
			static const bool verify = std::getenv("KYTY_VERIFY_COMMITTED") != nullptr;
			if (verify) {
				Common::LockGuard lock(m_mutex);
				uint64_t          covered = 0;
				auto              it      = m_ranges.upper_bound(virtual_addr);
				if (it != m_ranges.begin()) {
					--it;
					uint64_t expected = virtual_addr;
					for (; it != m_ranges.end() && it->second.start <= expected && covered < size; ++it) {
						const auto range_end = End(it->second.start, it->second.size);
						if (range_end <= expected || !IsCommittedRangeType(it->second.type)) break;
						covered += std::min(size - covered, range_end - expected);
						expected = range_end;
					}
				}
				static std::atomic<uint32_t> mismatches {0};
				if (covered != size && mismatches.fetch_add(1, std::memory_order_relaxed) < 32)
					std::printf("COMMITTED-MISMATCH addr=0x%llx size=0x%llx covered=0x%llx\n",
					            static_cast<unsigned long long>(virtual_addr), static_cast<unsigned long long>(size),
					            static_cast<unsigned long long>(covered));
			}
			return size;
		}
		struct CachedRange {
			uint64_t identity = 0, epoch = 0, begin = 0, end = 0;
		};
		thread_local std::array<CachedRange, 128> cache;
		auto& entry = cache[((virtual_addr >> 18) ^ (virtual_addr >> 30)) & 127];
		{
			const auto epoch = m_epoch.load(std::memory_order_acquire);
			// Only fully contained committed requests can bypass the lookup.
			// Clamped/invalid requests still use the original locked walk.
			if (!(epoch & 1) && entry.identity == m_identity && entry.epoch == epoch &&
			    virtual_addr >= entry.begin && virtual_addr < entry.end &&
			    size <= entry.end - virtual_addr)
				return size;
		}
		Common::LockGuard lock(m_mutex);

		auto vma = m_ranges.upper_bound(virtual_addr);
		if (vma == m_ranges.begin()) {
			return 0;
		}
		--vma;

		const auto vma_end = End(vma->second.start, vma->second.size);
		if (virtual_addr < vma->second.start || virtual_addr >= vma_end ||
		    !IsCommittedRangeType(vma->second.type)) {
			return 0;
		}

		const auto cache_begin  = vma->second.start;
		auto       cache_end    = vma_end;
		uint64_t   clamped_size = std::min(size, vma_end - virtual_addr);
		uint64_t   expected     = virtual_addr + clamped_size;
		++vma;

		while (vma != m_ranges.end() && vma->second.start == expected && IsCommittedRangeType(vma->second.type) &&
		       clamped_size < size) {
			cache_end        = End(vma->second.start, vma->second.size);
			const auto chunk = std::min(size - clamped_size, vma->second.size);
			clamped_size += chunk;
			expected += chunk;
			++vma;
		}

		entry = {m_identity, m_epoch.load(std::memory_order_relaxed), cache_begin, cache_end};
		return clamped_size;
	}

	struct Part {
		uint64_t address        = 0;
		uint64_t size           = 0;
		bool     private_memory = false; // stacks, code, runtime memory: no backing view
	};
	// The committed parts of [virtual_addr, virtual_addr + size), in order, adjacent ones of a kind merged.
	void CommittedParts(uint64_t virtual_addr, uint64_t size, std::vector<Part>* out) {
		out->clear();
		if (size == 0 || size > UINT64_MAX - virtual_addr) {
			return;
		}
		const auto        end = virtual_addr + size;
		Common::LockGuard lock(m_mutex);
		auto              it  = m_ranges.upper_bound(virtual_addr);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->second.start < end; ++it) {
			const auto& range     = it->second;
			const auto  range_end = End(range.start, range.size);
			if (range_end <= virtual_addr || !IsCommittedRangeType(range.type)) {
				continue;
			}
			const auto begin          = std::max(range.start, virtual_addr);
			const auto finish         = std::min(range_end, end);
			const bool private_memory = IsPrivateCommittedRangeType(range.type);
			if (!out->empty() && out->back().address + out->back().size == begin &&
			    out->back().private_memory == private_memory) {
				out->back().size += finish - begin;
			} else {
				out->push_back({begin, finish - begin, private_memory});
			}
		}
	}

	// (11-13) Committed guest bytes by access: [0] CPU only, [1] GPU only, [2] both.
	void SumByAccess(uint64_t out[3]) {
		Common::LockGuard lock(m_mutex);
		out[0] = out[1] = out[2] = 0;
		for (const auto& [key, r]: m_ranges) {
			if (!IsCommittedRangeType(r.type) || r.size == 0) continue;
			const bool cpu = (r.protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC)) != 0;
			const bool gpu = (r.protection & (PROT_GPU_READ | PROT_GPU_WRITE)) != 0;
			out[cpu && gpu ? 2 : gpu ? 1 : 0] += r.size;
		}
	}

	uint64_t CountPageTableEntries(bool gpu) {
		Common::LockGuard lock(m_mutex);

		uint64_t used = 0;
		for (const auto& [key, r]: m_ranges) {
			if (!IsCommittedRangeType(r.type) || r.size == 0) {
				continue;
			}

			const bool has_cpu_access =
			    (r.protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC)) != 0;
			const bool has_gpu_access = (r.protection & (PROT_GPU_READ | PROT_GPU_WRITE)) != 0;
			if (gpu ? !has_gpu_access : !has_cpu_access) {
				continue;
			}

			const auto end         = End(r.start, r.size);
			const auto first_entry = r.start / PAGE_TABLE_GRANULARITY;
			const auto last_entry  = (end - 1) / PAGE_TABLE_GRANULARITY;
			used += last_entry - first_entry + 1;
		}

		return used;
	}

private:
	using Ranges   = std::map<uint64_t, Range>;
	using Iterator = Ranges::iterator;

	// Writers already hold m_mutex. Odd epochs make concurrent cache readers
	// take that lock; the final release invalidates every previous interval.
	struct Mutation {
		VirtualRanges& owner;
		explicit Mutation(VirtualRanges& ranges): owner(ranges) {
			owner.m_epoch.fetch_add(1, std::memory_order_acq_rel);
		}
		~Mutation() { owner.m_epoch.fetch_add(1, std::memory_order_release); }
	};
	inline static std::atomic<uint64_t> s_next_identity {1};
	const uint64_t        m_identity = s_next_identity.fetch_add(1, std::memory_order_relaxed);
	std::atomic<uint64_t> m_epoch {2};

	static uint64_t End(uint64_t start, uint64_t size) {
		return (UINT64_MAX - start < size ? UINT64_MAX : start + size);
	}

	static bool SameMergeKey(const Range& left, const Range& right) {
		if (left.disallow_merge || right.disallow_merge || left.type == VirtualRangeType::Direct ||
		    right.type == VirtualRangeType::Direct) {
			return false;
		}

		return left.type == right.type && left.protection == right.protection &&
		       left.memory_type == right.memory_type &&
		       std::strncmp(left.name, right.name, KERNEL_MAXIMUM_NAME_LENGTH) == 0;
	}

	static void AddPiece(std::vector<Range>* ranges, const Range& source, uint64_t start,
	                     uint64_t end) {
		EXIT_IF(ranges == nullptr);

		if (end <= start) {
			return;
		}

		Range piece = source;
		piece.start = start;
		piece.size  = end - start;
		if (piece.type == VirtualRangeType::Direct) {
			piece.offset += start - source.start;
		}
		ranges->push_back(piece);
	}

	// `it` with the adjacent ranges it can merge with (its predecessor first, then its successors).
	void MergeAroundUnlocked(Iterator it) {
		if (it == m_ranges.end()) {
			return;
		}
		if (it != m_ranges.begin()) {
			const auto previous = std::prev(it);
			if (End(previous->second.start, previous->second.size) == it->second.start &&
			    SameMergeKey(previous->second, it->second)) {
				previous->second.size += it->second.size;
				m_ranges.erase(it);
				it = previous;
			}
		}
		for (auto next = std::next(it); next != m_ranges.end(); next = std::next(it)) {
			if (End(it->second.start, it->second.size) != next->second.start ||
			    !SameMergeKey(it->second, next->second)) {
				break;
			}
			it->second.size += next->second.size;
			m_ranges.erase(next);
		}
	}

	// The ranges overlapping [start, start + size): [first, last) (sorted, disjoint, never empty).
	// Only the range containing `start` can begin before it.
	std::pair<Iterator, Iterator> OverlapSpan(uint64_t start, uint64_t size) {
		const auto end   = End(start, size);
		auto       first = m_ranges.upper_bound(start);
		if (first != m_ranges.begin() && End(std::prev(first)->second.start, std::prev(first)->second.size) > start) {
			--first;
		}
		auto last = first;
		while (last != m_ranges.end() && last->second.start < end) {
			++last;
		}
		return {first, last};
	}

	// Replaces [first, last) with `pieces` (in order, inside the replaced span), then merges what the change can have
	// made mergeable: the pieces and their two neighbours. Every other adjacent pair was already unmergeable (each
	// mutation merges around what it changed), so this equals merging the whole list.
	void SpliceUnlocked(Iterator first, Iterator last, const std::vector<Range>& pieces) {
		for (auto it = first; it != last; ++it) {
			if (IsCommittedRangeType(it->second.type)) MarkCommittedUnlocked(it->second.start, it->second.size, false);
		}
		for (const auto& piece: pieces) {
			if (IsCommittedRangeType(piece.type)) MarkCommittedUnlocked(piece.start, piece.size, true);
		}
		auto hint = m_ranges.erase(first, last);
		if (pieces.empty()) {
			return; // (a gap now separates the neighbours)
		}
		Iterator inserted_first = m_ranges.end();
		for (const auto& piece: pieces) {
			const auto it = m_ranges.emplace_hint(hint, piece.start, piece);
			if (inserted_first == m_ranges.end()) inserted_first = it;
			hint = std::next(it);
		}
		// Merge within [the range before the first piece, the range after the last piece].
		const auto stop_key = pieces.back().start;
		auto       it       = inserted_first != m_ranges.begin() ? std::prev(inserted_first) : inserted_first;
		while (it != m_ranges.end() && it->first <= stop_key) {
			const auto next = std::next(it);
			if (next == m_ranges.end()) {
				break;
			}
			if (End(it->second.start, it->second.size) == next->second.start &&
			    SameMergeKey(it->second, next->second)) {
				it->second.size += next->second.size;
				m_ranges.erase(next);
			} else {
				it = next;
			}
		}
	}

	template <typename EditFunc>
	void EditUnlocked(uint64_t start, uint64_t size, EditFunc edit) {
		if (size == 0) {
			return;
		}

		const auto [first, last] = OverlapSpan(start, size);
		if (first == last) {
			return;
		}
		std::vector<Range> pieces;
		auto               edit_end = End(start, size);
		for (auto i = first; i != last; ++i) {
			const auto r     = i->second;
			auto       r_end = End(r.start, r.size);

			auto mid_start = std::max(start, r.start);
			auto mid_end   = std::min(edit_end, r_end);

			AddPiece(&pieces, r, r.start, mid_start);

			Range mid = r;
			mid.start = mid_start;
			mid.size  = mid_end - mid_start;
			if (mid.type == VirtualRangeType::Direct) {
				mid.offset += mid_start - r.start;
			}
			edit(&mid);
			pieces.push_back(mid);

			AddPiece(&pieces, r, mid_end, r_end);
		}
		SpliceUnlocked(first, last, pieces);
	}

	bool RemoveUnlocked(uint64_t start, uint64_t size) {
		if (size == 0) {
			return false;
		}

		const auto [first, last] = OverlapSpan(start, size);
		if (first == last) {
			return false;
		}
		// Only the first overlapping range can keep a head and only the last one a tail.
		std::vector<Range> pieces;
		const auto         rem_end = End(start, size);
		const auto         head    = first->second;
		const auto         tail    = std::prev(last)->second;
		AddPiece(&pieces, head, head.start, std::max(start, head.start));
		AddPiece(&pieces, tail, std::min(rem_end, End(tail.start, tail.size)), End(tail.start, tail.size));
		SpliceUnlocked(first, last, pieces);
		return true;
	}

	// The range of `type` that contains `address`, or null.
	const Range* FindContainingUnlocked(uint64_t address, VirtualRangeType type) {
		const auto [first, last] = OverlapSpan(address, 1);
		if (first == last || first->second.type != type || first->second.start > address) {
			return nullptr;
		}
		return &first->second;
	}

	Range* FindOverlap(uint64_t start, uint64_t size) {
		auto position = m_ranges.lower_bound(start);
		if (position != m_ranges.end() &&
		    VirtualRangesOverlap(start, size, position->second.start, position->second.size)) {
			return &position->second;
		}
		if (position != m_ranges.begin()) {
			auto previous = std::prev(position);
			if (VirtualRangesOverlap(start, size, previous->second.start, previous->second.size)) {
				return &previous->second;
			}
		}
		return nullptr;
	}

	// Committed pages (a bit per 4 KiB page, set for pages a committed range covers whole, by 1 GiB chunk), read without
	// the lock: a request on set pages only is fully committed. Written under m_mutex: cleared for each committed range
	// that goes before set for each one that comes, so a reader racing a change sees the pages before or after it (or
	// clear: the locked path answers). Guest addresses past the chunks take the locked path.
	static constexpr uint64_t CommittedPageBits = 12, CommittedChunkBits = 30, CommittedChunks = 1024;
	static constexpr uint64_t CommittedChunkWords = uint64_t {1} << (CommittedChunkBits - CommittedPageBits - 6);
	std::array<std::atomic<std::atomic<uint64_t>*>, CommittedChunks> m_committed {};

	void MarkCommittedUnlocked(uint64_t start, uint64_t size, bool on) {
		const auto end = End(start, size);
		// Set: the pages the range covers whole; cleared: every page it touches.
		auto first = on ? (start + (uint64_t {1} << CommittedPageBits) - 1) >> CommittedPageBits : start >> CommittedPageBits;
		auto last  = on ? end >> CommittedPageBits : (end + (uint64_t {1} << CommittedPageBits) - 1) >> CommittedPageBits;
		last       = std::min<uint64_t>(last, CommittedChunks << (CommittedChunkBits - CommittedPageBits));
		for (auto page = first; page < last;) {
			const auto chunk_index = page >> (CommittedChunkBits - CommittedPageBits);
			auto*      chunk       = m_committed[chunk_index].load(std::memory_order_acquire);
			const auto chunk_end   = (chunk_index + 1) << (CommittedChunkBits - CommittedPageBits);
			const auto stop        = std::min(last, chunk_end);
			if (chunk == nullptr) {
				if (!on) {
					page = stop;
					continue;
				}
				chunk = new std::atomic<uint64_t>[CommittedChunkWords] {};
				m_committed[chunk_index].store(chunk, std::memory_order_release);
			}
			for (; page < stop;) {
				const auto local = page - (chunk_index << (CommittedChunkBits - CommittedPageBits));
				const auto bit   = local & 63u;
				const auto count = std::min<uint64_t>(64 - bit, stop - page);
				const auto mask  = (count == 64 ? ~uint64_t {0} : ((uint64_t {1} << count) - 1)) << bit;
				if (on)
					chunk[local >> 6].fetch_or(mask, std::memory_order_release);
				else
					chunk[local >> 6].fetch_and(~mask, std::memory_order_release);
				page += count;
			}
		}
	}

	[[nodiscard]] bool FullyCommitted(uint64_t start, uint64_t size) const {
		const auto first = start >> CommittedPageBits;
		const auto last  = (start + size - 1) >> CommittedPageBits; // (inclusive; size != 0, no overflow: checked)
		if (last >= (CommittedChunks << (CommittedChunkBits - CommittedPageBits))) return false;
		for (auto page = first; page <= last;) {
			const auto chunk_index = page >> (CommittedChunkBits - CommittedPageBits);
			const auto* chunk      = m_committed[chunk_index].load(std::memory_order_acquire);
			if (chunk == nullptr) return false;
			const auto stop = std::min(last + 1, (chunk_index + 1) << (CommittedChunkBits - CommittedPageBits));
			for (; page < stop;) {
				const auto local = page - (chunk_index << (CommittedChunkBits - CommittedPageBits));
				const auto bit   = local & 63u;
				const auto count = std::min<uint64_t>(64 - bit, stop - page);
				const auto mask  = (count == 64 ? ~uint64_t {0} : ((uint64_t {1} << count) - 1)) << bit;
				if ((chunk[local >> 6].load(std::memory_order_acquire) & mask) != mask) return false;
				page += count;
			}
		}
		return true;
	}

	Ranges        m_ranges;
	Common::Mutex m_mutex;
};

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
static uint32_t g_test_physical_memory_unmaps_before_failure = UINT32_MAX;
static bool     g_test_fail_next_fixed_reserve_range_add     = false;
#endif

// Bytes of the guest backing store allocated (direct + flexible); the peak is logged every 30 s.
static std::atomic<int64_t> g_guest_store_used {0};
static std::atomic<int64_t> g_guest_store_peak {0};
static void GuestStoreUsage(int64_t delta) {
	const int64_t used = g_guest_store_used.fetch_add(delta, std::memory_order_relaxed) + delta;
	int64_t       peak = g_guest_store_peak.load(std::memory_order_relaxed);
	while (used > peak && !g_guest_store_peak.compare_exchange_weak(peak, used, std::memory_order_relaxed)) {
	}
	using Clock = std::chrono::steady_clock;
	static std::atomic<int64_t> last {0};
	const int64_t now  = std::chrono::duration_cast<std::chrono::seconds>(Clock::now().time_since_epoch()).count();
	int64_t       prev = last.load(std::memory_order_relaxed);
	if (now - prev >= 30 && last.compare_exchange_strong(prev, now, std::memory_order_relaxed)) {
		std::printf("Guest store: %lld MiB allocated now, peak %lld MiB\n", static_cast<long long>(used >> 20),
		            static_cast<long long>(g_guest_store_peak.load(std::memory_order_relaxed) >> 20));
	}
}
static void GuestStoreTooSmall(const char* what, uint64_t len);

class PhysicalMemory {
public:
	struct AllocatedBlock {
		uint64_t            start_addr;
		uint64_t            size;
		uint64_t            map_vaddr;
		uint64_t            map_size;
		uint64_t            host_vaddr;
		uint64_t            host_size;
		int                 prot;
		VirtualMemory::Mode mode;
		GpuAccessMode       gpu_mode;
		int                 memory_type;
		bool                pool_expansion;
		char                name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	PhysicalMemory() {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		m_free.emplace(0, Size());
	}
	virtual ~PhysicalMemory() = default;

	KYTY_CLASS_NO_COPY(PhysicalMemory);

	// The guest backing store: 13824 MiB (direct memory + flexible memory), or KYTY_GUEST_STORE_MB (at least 11264).
	// The direct-memory size the game is told (sceKernelGetDirectMemorySize) shrinks with it.
	static uint64_t TotalSize() {
		static const uint64_t total = [] {
			constexpr uint64_t Default = 13824;
			constexpr uint64_t Minimum = 11264;
			uint64_t           mib     = Default;
			if (const char* v = std::getenv("KYTY_GUEST_STORE_MB"); v != nullptr && *v != 0) {
				const uint64_t asked = std::strtoull(v, nullptr, 10);
				mib = std::min(Default, std::max(Minimum, asked));
				std::printf("Guest store: KYTY_GUEST_STORE_MB=%s -> %llu MiB (clamped to %llu..%llu)\n", v,
				            static_cast<unsigned long long>(mib), static_cast<unsigned long long>(Minimum),
				            static_cast<unsigned long long>(Default));
			} else {
				std::printf("Guest store: %llu MiB (KYTY_GUEST_STORE_MB not set)\n", static_cast<unsigned long long>(mib));
			}
			std::fflush(stdout);
			return mib * 1024 * 1024;
		}();
		return total;
	}
	static uint64_t           Size() {
		EXIT_IF(g_flexible_memory_size >= TotalSize());
		return TotalSize() - g_flexible_memory_size;
	}

	bool Alloc(uint64_t search_start, uint64_t search_end, size_t len, size_t alignment,
	           uint64_t* phys_addr_out, int memory_type, bool pool_expansion = false);
	bool Available(uint64_t search_start, uint64_t search_end, size_t alignment,
	               uint64_t* phys_addr_out, uint64_t* size_out);
	bool Release(uint64_t start, size_t len, uint64_t* vaddr, uint64_t* size,
	             GpuAccessMode* gpu_mode);
	bool Map(uint64_t vaddr, uint64_t phys_addr, size_t len, int prot, VirtualMemory::Mode mode,
	         GpuAccessMode gpu_mode);
	bool Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
	           uint64_t* host_vaddr_to_release = nullptr);
	bool Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
	          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode);
	bool Find(uint64_t phys_addr, bool next, PhysicalMemory::AllocatedBlock* out);
	bool CanMapDirect(uint64_t phys_addr, size_t len);
	bool ReleasePoolExpansion(uint64_t phys_addr, size_t len);
	bool GetAllocatedSpan(uint64_t phys_addr, size_t len, std::vector<AllocatedBlock>* blocks);
	std::vector<AllocatedBlock> FindMappings(uint64_t phys_addr, size_t len);
	void ProtectMapping(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
	                    GpuAccessMode gpu_mode);
	void SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name);
	void SetVirtualRangeMemoryType(uint64_t vaddr, uint64_t len, int memory_type);

	[[nodiscard]] Common::Mutex&                            GetMutex() { return m_mutex; }
	[[nodiscard]] const std::map<uint64_t, AllocatedBlock>& GetPhysicalBlocks() const {
		return m_physical;
	}
	[[nodiscard]] const std::map<uint64_t, AllocatedBlock>& GetMappings() const { return m_mappings; }
	// (The periodic RAM flush, ram-guard.h: the free ranges as backing offsets, and whether a range is all free.)
	void FreeRanges(std::vector<std::pair<uint64_t, uint64_t>>* out) {
		Common::LockGuard lock(m_mutex);
		for (const auto& [start, size]: m_free) out->emplace_back(start, size);
	}
	// `call` with [start, start + size) free all along (no allocation can take it meanwhile: Alloc waits).
	template <typename F>
	bool WhileFree(uint64_t start, uint64_t size, F&& call) {
		Common::LockGuard lock(m_mutex);
		auto it = m_free.upper_bound(start);
		if (it == m_free.begin()) return false;
		--it;
		if (!(it->first <= start && it->first + it->second >= start + size)) return false;
		call();
		return true;
	}

private:
	void ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
	                      uint64_t size);
	void AddFreeRange(uint64_t start, uint64_t size);

	std::map<uint64_t, AllocatedBlock> m_physical;
	std::map<uint64_t, uint64_t>       m_free;
	// The mappings overlapping [vaddr, vaddr + len), in order (m_mutex held).
	template <typename Func>
	void ForEachMappingOverlapping(uint64_t vaddr, uint64_t len, Func&& func) {
		auto it = m_mappings.upper_bound(vaddr);
		if (it != m_mappings.begin() && std::prev(it)->second.map_vaddr + std::prev(it)->second.map_size > vaddr) {
			--it;
		}
		for (; it != m_mappings.end() && it->second.map_vaddr < vaddr + len; ++it) {
			if (VirtualRangesOverlap(vaddr, len, it->second.map_vaddr, it->second.map_size)) func(it->second);
		}
	}
	// The mappings by their virtual address (disjoint): a vector scanned at every map, unmap and protection change
	// held tens of thousands of streamed-texture mappings (~130 bytes each).
	std::map<uint64_t, AllocatedBlock> m_mappings;
	Common::Mutex                      m_mutex;
};

class FlexibleMemory {
public:
	struct AllocatedBlock {
		uint64_t            map_vaddr;
		uint64_t            map_size;
		uint64_t            backing_offset;
		uint64_t            host_vaddr;
		uint64_t            host_size;
		int                 prot;
		VirtualMemory::Mode mode;
		GpuAccessMode       gpu_mode;
		char                name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	FlexibleMemory(): m_backing_high_water(PhysicalMemory::Size()) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		m_free.emplace(PhysicalMemory::Size(), Size());
	}
	virtual ~FlexibleMemory() = default;

	KYTY_CLASS_NO_COPY(FlexibleMemory);

	static uint64_t Size() { return g_flexible_memory_size; }
	uint64_t        Available();

	bool Map(uint64_t vaddr, size_t len, int prot, VirtualMemory::Mode mode, GpuAccessMode gpu_mode,
	         const char* name);
	bool Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
	           uint64_t* host_vaddr_to_release = nullptr);
	bool Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
	          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode);
	bool Snapshot(uint64_t vaddr, uint64_t size, std::vector<AllocatedBlock>* blocks);
	bool Restore(const std::vector<AllocatedBlock>& blocks);
	void Protect(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
	             GpuAccessMode gpu_mode);
	void SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name);

	[[nodiscard]] Common::Mutex&                     GetMutex() { return m_mutex; }
	[[nodiscard]] const std::vector<AllocatedBlock>& GetBlocks() const { return m_allocated; }
	// (The periodic RAM flush, ram-guard.h: the free backing below the high-water mark - above it the section's pages
	// were never written - and whether a range is all free.)
	void FreeRanges(std::vector<std::pair<uint64_t, uint64_t>>* out) {
		Common::LockGuard lock(m_mutex);
		for (const auto& [start, size]: m_free) {
			const uint64_t end = std::min(start + size, m_backing_high_water);
			if (start < end) out->emplace_back(start, end - start);
		}
	}
	// `call` with [start, start + size) free all along (Map, which zeroes reused backing, waits).
	template <typename F>
	bool WhileFree(uint64_t start, uint64_t size, F&& call) {
		Common::LockGuard lock(m_mutex);
		auto it = m_free.upper_bound(start);
		if (it == m_free.begin()) return false;
		--it;
		if (!(it->first <= start && it->first + it->second >= start + size && start + size <= m_backing_high_water))
			return false;
		call();
		return true;
	}

private:
	void ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
	                      uint64_t size);
	void AddFreeRange(uint64_t start, uint64_t size);

	std::vector<AllocatedBlock>  m_allocated;
	std::map<uint64_t, uint64_t> m_free;
	uint64_t                     m_allocated_total = 0;
	// The end of the highest backing any mapping was given (Map: what is above it is still the section's zero pages).
	uint64_t                     m_backing_high_water = 0;
	Common::Mutex                m_mutex;
};

class PooledMemory {
public:
	struct Mapping {
		uint64_t      vaddr;
		uint64_t      size;
		uint64_t      phys_addr;
		GpuAccessMode gpu_mode;
	};

	void                   Expand(uint64_t phys_addr, uint64_t size);
	bool                   ReleaseExpansion(uint64_t phys_addr, uint64_t size);
	bool                   Allocate(uint64_t vaddr, uint64_t size, GpuAccessMode gpu_mode,
	                                std::vector<Mapping>* mappings);
	bool                   Query(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings);
	bool                   Release(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode);
	[[nodiscard]] uint64_t Available();
	[[nodiscard]] std::vector<Mapping> GetMappings();

private:
	struct PhysicalRange {
		uint64_t start;
		uint64_t size;
	};

	void AddFreeUnlocked(uint64_t start, uint64_t size);
	bool QueryUnlocked(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings) const;

	std::vector<PhysicalRange> m_free;
	std::vector<PhysicalRange> m_expansions;
	std::vector<Mapping>       m_mappings;
	Common::Mutex              m_mutex;
};

static std::unique_ptr<PhysicalMemory>    g_physical_memory;
static std::unique_ptr<FlexibleMemory>    g_flexible_memory;
static std::unique_ptr<PooledMemory>      g_pooled_memory;
static std::unique_ptr<VirtualRanges>     g_virtual_ranges;

// (11-13) A "Guest memory by access" line each time the mapped total grows by 512 MiB: how much of the
// game's memory the CPU never touches (GPU only), the part a PC port keeps in video memory alone.
void ReportGuestAccess(bool force);
void ReportGuestAccess(bool force) {
	static std::atomic<uint64_t> last {0};
	if (g_virtual_ranges == nullptr) return;
	uint64_t sum[3];
	g_virtual_ranges->SumByAccess(sum);
	const uint64_t total = sum[0] + sum[1] + sum[2];
	uint64_t prev = last.load();
	if (!force && total < prev + (512ull << 20) && !(total + (512ull << 20) < prev)) return;
	if (!last.compare_exchange_strong(prev, total) && !force) return;
	std::printf("Guest memory by access: total %llu MiB = CPU only %llu MiB, GPU only %llu MiB, CPU and GPU %llu MiB\n",
	     (unsigned long long)(total >> 20), (unsigned long long)(sum[0] >> 20), (unsigned long long)(sum[1] >> 20),
	     (unsigned long long)(sum[2] >> 20));
	std::fflush(stdout);
}
static std::unique_ptr<GuestAddressSpace> g_guest_address_space;
static callback_func_t                    g_alloc_callback        = nullptr;
static callback_func_t                    g_free_callback         = nullptr;
static std::atomic<uint64_t>              g_memory_pool_committed = 0;
static void                               MemoryPoolSubtractCommitted(uint64_t len);
// Keep host mappings, physical blocks, placeholders, and virtual ranges in step.
static std::recursive_mutex g_memory_operation_mutex;

// The base address the PS5 kernel hands out for hint-less user mappings. Guest code can
// assume mappings it did not place explicitly are at or above this (Sony's libc rejects a
// heap below it), so hint-less searches must not fall back to the low system-managed range.
static constexpr uint64_t GUEST_DEFAULT_MAP_BASE = 0x200000000ull;

static uint64_t FindGuestFreeRange(uint64_t search_addr, uint64_t size, uint64_t alignment) {
	EXIT_IF(g_guest_address_space == nullptr || g_virtual_ranges == nullptr);

	auto find_in = [&](uint64_t begin, uint64_t end) {
		auto current = begin;
		while (current < end && size <= end - current) {
			const auto candidate =
			    g_guest_address_space->FindFreeAligned(current, end, size, alignment);
			if (candidate == 0) {
				return uint64_t {0};
			}
			VirtualRanges::Range overlap {};
			if (!g_virtual_ranges->QueryOverlap(candidate, size, &overlap)) {
				return candidate;
			}
			const auto overlap_end = overlap.start + overlap.size;
			if (overlap_end <= current) {
				return uint64_t {0};
			}
			current = overlap_end;
		}
		return uint64_t {0};
	};

	if (search_addr != 0) {
		return find_in(search_addr, HOST_USER_MAX + 1u);
	}
	auto addr = find_in(GUEST_DEFAULT_MAP_BASE, HOST_SYSTEM_MANAGED_MAX + 1u);
	if (addr == 0) {
		addr = find_in(HOST_USER_MIN, HOST_USER_MAX + 1u);
	}
	return addr;
}

bool WriteGuestRanges(const char* path) {
	if (g_virtual_ranges == nullptr || path == nullptr) return false;
	FILE* file = std::fopen(path, "w");
	if (file == nullptr) return false;
	g_virtual_ranges->Dump(file);
	std::fclose(file);
	return true;
}

bool BackingView(uint64_t* base, uint64_t* size) {
	if (g_guest_address_space == nullptr || base == nullptr || size == nullptr) return false;
	*base = g_guest_address_space->GetBackingBase();
	*size = g_guest_address_space->GetBackingSize();
	return *base != 0 && *size != 0;
}

bool GuestRangeAccess(uint64_t vaddr, uint64_t size, bool* gpu, bool* cpu_write) {
	std::vector<VirtualRanges::Range> ranges;
	if (g_virtual_ranges == nullptr || gpu == nullptr || cpu_write == nullptr || !g_virtual_ranges->QuerySpan(vaddr, size, &ranges))
		return false;
	*gpu       = false;
	*cpu_write = false;
	for (const auto& range: ranges) {
		*gpu       = *gpu || (range.protection & (PROT_GPU_READ | PROT_GPU_WRITE)) != 0;
		*cpu_write = *cpu_write || (range.protection & PROT_CPU_WRITE) != 0;
	}
	return true;
}

// (11-14) Native memory: a write by the emulator itself (GPU results written back, offloaded textures restored) is
// not a game write: its pages' write-watch bits are cleared, or the caches would take the GPU's own results for new
// CPU data and upload stale bytes over them (corrupt, flickering textures).
void NativeForgetWrites(uint64_t vaddr, uint64_t size) {
#if defined(_WIN32)
	if (size == 0 || !NativeMemoryEnabled()) return;
	// (raiz1, KYTY_RAIZ: the game's writes are seen by faults, no sweep reads the write watch - a ResetWriteWatch per
	// write-back for nothing -, and KYTY_RAIZ_VERIFY's audit must see it as it is.)
	if (Raiz::On()) return;
	constexpr uint64_t Page = 0x1000;
	const uint64_t     begin = vaddr & ~(Page - 1u);
	const uint64_t     end   = (vaddr + size + Page - 1u) & ~(Page - 1u);
	// (ram4) ResetWriteWatch takes one allocation's pages: over a range the game mapped in several calls (pool blocks
	// side by side: a texture or a buffer written back across two of them) it fails whole and resets nothing, and the
	// emulator's own write was then taken for the game's at the next sweep (the GPU's results invalidated: uploaded
	// again from the bytes just written back, an image's GPU copy dropped). Then mapping by mapping, as the sweeps read
	// it (NativeWrittenPages: read and reset, the pages read thrown away; the caller kept the game's writes before its
	// own, PageManager::CaptureCpuWrites).
	if (ResetWriteWatch(reinterpret_cast<void*>(begin), end - begin) != 0) {
		static std::atomic<uint64_t> fallbacks {0};
		if (fallbacks.fetch_add(1, std::memory_order_relaxed) == 0) {
			std::printf("Native memory: a write-back over more than one allocation (0x%llx+0x%llx): its write watch reset "
			            "mapping by mapping (said once)\n",
			            static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin));
			std::fflush(stdout);
		}
		(void)NativeWriteWatch(begin, end - begin, nullptr, nullptr, true);
	}
#else
	(void)vaddr;
	(void)size;
#endif
}

// (ram15) KYTY_NATIVE_TEXTURE_OFFLOAD (local/texture-offload.h): memory.h.
void SetNativeWindowHook(native_window_hook_t hook) {
	g_native_window_hook.store(hook, std::memory_order_release);
}

uint64_t NativeOffloadDrop(uint64_t vaddr, uint64_t size, uint64_t* refused) {
	uint64_t done = 0, not_done = 0;
#if defined(_WIN32)
	if (size != 0 && UINT64_MAX - vaddr >= size && g_guest_address_space != nullptr && g_guest_address_space->IsNative()) {
		constexpr uint64_t Chunk = uint64_t {1} << 20u; // (a MiB at a time: the working-set lock a reset holds)
		const uint64_t     end   = vaddr + size;
		for (uint64_t chunk = vaddr; chunk < end; chunk += Chunk) {
			const uint64_t chunk_end = std::min(end, chunk + Chunk);
			uint64_t       reset = 0, refused_here = 0;
			const bool     ran   = g_guest_address_space->RunQuietWith([&] {
                for (uint64_t at = chunk; at < chunk_end;) {
                    MEMORY_BASIC_INFORMATION info {};
                    if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
                    const uint64_t stop =
                        std::min(chunk_end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
                    if (stop <= at) break;
                    if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE) {
                        if (VirtualAlloc(reinterpret_cast<void*>(at), stop - at, MEM_RESET, PAGE_READWRITE) != nullptr) {
                            reset += stop - at;
                            (void)VirtualUnlock(reinterpret_cast<void*>(at), stop - at); // (fails "not locked": out of the working set)
                        } else {
                            refused_here += stop - at;
                        }
                    }
                    at = stop;
                }
			});
			if (!ran) refused_here = chunk_end - chunk; // (the address space busy for 200 ms: left in RAM)
			done += reset;
			not_done += refused_here;
			SwitchToThread();
		}
	}
#else
	(void)vaddr;
	not_done = size;
#endif
	if (refused != nullptr) *refused = not_done;
	return done;
}

bool NativeOffloadWrite(uint64_t vaddr, const void* data, uint64_t size) {
	if (size == 0) return true;
#if defined(_WIN32)
	if (data == nullptr || g_guest_address_space == nullptr || !g_guest_address_space->IsNative()) return false;
	auto*    host = reinterpret_cast<uint8_t*>(vaddr);
	uint64_t left = size;
	// (A guarded copy inside the window: a page unmapped meanwhile stops it instead of faulting in the emulator.)
	NativeGuardedAccess(host, size, [&] { left = Common::GuardedCopy(host, data, size); }, true);
	return left == 0;
#else
	(void)vaddr, (void)data;
	return false;
#endif
}

bool NativeOffloadOpen(uint64_t vaddr, uint64_t size, bool write) {
#if defined(_WIN32)
	if (size == 0 || g_guest_address_space == nullptr || !g_guest_address_space->IsNative()) return false;
	NativeProtectLock guard(NativeReadFaultsRequested());
	DWORD old = 0;
	return VirtualProtect(reinterpret_cast<void*>(vaddr), size, write ? PAGE_READWRITE : PAGE_READONLY, &old) != 0;
#else
	(void)vaddr, (void)size, (void)write;
	return false;
#endif
}

// (anuncio1) KYTY_RAIZ_ANNOUNCE=1 + KYTY_RAIZ_VERIFY=1 (loader/demonsSoulsWriteAudit.cpp): the write watch of the pages
// no protection watches. No lock of the emulator's: callers hold a tracker region's lock or a cache's. A write-watch call
// takes one allocation's pages; a range over several is done VirtualQuery region by region (each inside one allocation).
// Memory unmapped meanwhile: the calls fail there and nothing is reported or reset.
void NativeAnnounceReset(uint64_t vaddr, uint64_t size) {
#if defined(_WIN32)
	if (size == 0 || !NativeMemoryEnabled()) return;
	constexpr uint64_t Page  = 0x1000;
	const uint64_t     begin = vaddr & ~(Page - 1u);
	const uint64_t     end   = (vaddr + size + Page - 1u) & ~(Page - 1u);
	if (ResetWriteWatch(reinterpret_cast<void*>(begin), end - begin) == 0) return;
	for (uint64_t at = begin; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
		const uint64_t stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && stop > at) (void)ResetWriteWatch(reinterpret_cast<void*>(at), stop - at);
		at = std::max(stop, at + Page);
	}
#else
	(void)vaddr;
	(void)size;
#endif
}

uint64_t NativeAnnounceWritten(uint64_t vaddr, uint64_t size, uint64_t* pages, uint64_t max) {
#if defined(_WIN32)
	if (size == 0 || !NativeMemoryEnabled()) return 0;
	constexpr uint64_t  Page  = 0x1000;
	constexpr ULONG_PTR Batch = 256;
	const uint64_t      begin = vaddr & ~(Page - 1u);
	const uint64_t      end   = (vaddr + size + Page - 1u) & ~(Page - 1u);
	void*               found[Batch];
	uint64_t            total = 0;
	// The written pages of [from, to) (inside one allocation); false: the call was refused.
	const auto take = [&](uint64_t from, uint64_t to) {
		while (from < to) {
			ULONG_PTR count       = Batch;
			DWORD     granularity = 0;
			if (GetWriteWatch(0, reinterpret_cast<void*>(from), to - from, found, &count, &granularity) != 0) return false;
			for (ULONG_PTR i = 0; i < count; ++i) {
				if (pages != nullptr && total < max) pages[total] = reinterpret_cast<uint64_t>(found[i]);
				++total;
			}
			if (count < Batch) break;
			from = reinterpret_cast<uint64_t>(found[count - 1]) + Page;
		}
		return true;
	};
	if (take(begin, end)) return total;
	total = 0;
	for (uint64_t at = begin; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
		const uint64_t stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && stop > at) (void)take(at, stop);
		at = std::max(stop, at + Page);
	}
	return total;
#else
	(void)vaddr;
	(void)size;
	(void)pages;
	(void)max;
	return 0;
#endif
}

// (nativo2) memory.h.
uint64_t NativeTakeWritten(uint64_t vaddr, uint64_t size, uint64_t* pages, uint64_t max, bool* failed) {
	if (failed != nullptr) *failed = false;
#if defined(_WIN32)
	if (size == 0 || !NativeMemoryEnabled()) return 0;
	constexpr uint64_t  Page  = 0x1000;
	constexpr ULONG_PTR Batch = 256;
	const uint64_t      begin = vaddr & ~(Page - 1u);
	const uint64_t      end   = (vaddr + size + Page - 1u) & ~(Page - 1u);
	void*               found[Batch];
	uint64_t            total = 0;
	// The written pages of [from, to) (inside one allocation); false: the call was refused.
	const auto take = [&](uint64_t from, uint64_t to) {
		while (from < to) {
			ULONG_PTR count       = Batch;
			DWORD     granularity = 0;
			if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<void*>(from), to - from, found, &count, &granularity) != 0) return false;
			for (ULONG_PTR i = 0; i < count; ++i) {
				if (pages != nullptr && total < max) pages[total] = reinterpret_cast<uint64_t>(found[i]);
				++total;
			}
			if (count < Batch) break;
			from = reinterpret_cast<uint64_t>(found[count - 1]) + Page;
		}
		return true;
	};
	if (take(begin, end)) return total;
	// (Refused over the whole range: it spans allocations. What a refused call reset is nothing - Windows fails it whole -,
	// and what earlier batches found stays counted: no page whose bit was reset is lost.)
	for (uint64_t at = begin; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) {
			if (failed != nullptr) *failed = true;
			break;
		}
		const uint64_t stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
		if (info.State == MEM_COMMIT && stop > at && (info.Type != MEM_PRIVATE || !take(at, stop)) && failed != nullptr)
			*failed = true;
		at = std::max(stop, at + Page);
	}
	return total;
#else
	(void)vaddr;
	(void)size;
	(void)pages;
	(void)max;
	if (failed != nullptr) *failed = true;
	return 0;
#endif
}

// (PROT-MENOS4) The written pages of [vaddr, vaddr + size) without resetting their write watch (one GetWriteWatch per 256
// pages found). *failed: the range is not one write-watched allocation (nothing read; the caller takes another path).
uint64_t NativePeekWritten(uint64_t vaddr, uint64_t size, uint64_t* pages, uint64_t max, bool* failed) {
	if (failed != nullptr) *failed = false;
#if defined(_WIN32)
	if (size == 0 || !NativeMemoryEnabled()) return 0;
	constexpr uint64_t  Page  = 0x1000;
	constexpr ULONG_PTR Batch = 256;
	uint64_t            from  = vaddr & ~(Page - 1u);
	const uint64_t      to    = (vaddr + size + Page - 1u) & ~(Page - 1u);
	void*               found[Batch];
	uint64_t            total = 0;
	while (from < to) {
		ULONG_PTR count       = Batch;
		DWORD     granularity = 0;
		if (GetWriteWatch(0, reinterpret_cast<void*>(from), to - from, found, &count, &granularity) != 0) {
			if (failed != nullptr) *failed = true;
			return 0;
		}
		for (ULONG_PTR i = 0; i < count; ++i) {
			if (pages != nullptr && total < max) pages[total] = reinterpret_cast<uint64_t>(found[i]);
			++total;
		}
		if (count < Batch) break;
		from = reinterpret_cast<uint64_t>(found[count - 1]) + Page;
	}
	return total;
#else
	(void)vaddr;
	(void)size;
	(void)pages;
	(void)max;
	if (failed != nullptr) *failed = true;
	return 0;
#endif
}

bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
	// (Native memory: the game's writes there since the last sweep are kept before NativeForgetWrites clears the
	// write watch of the emulator's own write: PageManager::CaptureCpuWrites.)
	if (size != 0 && NativeMemoryEnabled()) Libs::Graphics::PageManager::CaptureCpuWrites(vaddr, size);
	const bool done = g_guest_address_space != nullptr && g_guest_address_space->TryWriteBacking(vaddr, data, size);
	if (done) NativeForgetWrites(vaddr, size);
	return done;
}

// (raiz2) Memory::NativeGuardedRead / TryGetUploadSourcePointer (memory.h).
bool NativeGuardedRead(void* destination, const void* guest, uint64_t size) {
	if (size == 0) return true;
	auto*       to   = static_cast<uint8_t*>(destination);
	const auto* from = static_cast<const uint8_t*>(guest);
#if defined(_WIN32)
	if (Raiz::On()) {
		// Straight through the guest's own pages; where one is no-access the copy stops (no fault handler runs) and the
		// rest goes through a window (opened read-only: a game write meanwhile still faults).
		// (raiz11, KYTY_RAIZ_STALE_CHECK) The direct part checked against the put-ins of kept GPU bytes under way: one
		// overlapping it may have shown this copy the pages' old bytes (readable, the GPU's not in yet). Read again
		// through a window then (KYTY_RAIZ_STALE_FIX): the gate alone waits for the put-in to end.
		NativeReadTicket ticket;
		NativeReadTicketTake(reinterpret_cast<uint64_t>(from), size, &ticket);
		const auto stale_again = [&]() {
			if (NativeReadTicketClean(ticket)) return;
			const bool fix = NativeStaleFixOn();
			NativeNoteStale(Raiz::StaleUpload, reinterpret_cast<uint64_t>(from), size, static_cast<uint64_t>(Raiz::t_site), fix);
			if (fix) {
				NativeGuardedAccess(from, size, [&] {
					if (const uint64_t rest = Common::GuardedCopy(to, from, size); rest != 0) std::memset(to + size - rest, 0, rest);
				});
			}
		};
		const uint64_t left = Common::GuardedCopy(to, from, size);
		if (left == 0) {
			Raiz::g_counters.guarded_reads_direct.fetch_add(1, std::memory_order_relaxed);
			stale_again();
			return true;
		}
		const uint64_t done = size - left;
		Raiz::g_counters.guarded_reads_window.fetch_add(1, std::memory_order_relaxed);
		// (Guarded inside the window too: a page that is not committed - not a tracker's no-access - reads as zeros
		// instead of faulting on this thread; said once.)
		bool short_read = false;
		NativeGuardedAccess(from + done, left, [&] {
			if (const uint64_t rest = Common::GuardedCopy(to + done, from + done, left); rest != 0) {
				std::memset(to + size - rest, 0, rest);
				short_read = true;
			}
		});
		if (!short_read && done != 0) stale_again(); // (raiz11: the direct part)
		if (short_read) {
			static std::atomic<uint32_t> said {0};
			if (said.fetch_add(1, std::memory_order_relaxed) < 8) {
				std::printf("[raiz] guarded read of 0x%llx+0x%llx: part not readable even in a window (not committed?): "
				            "zeros\n",
				            static_cast<unsigned long long>(reinterpret_cast<uint64_t>(from)), static_cast<unsigned long long>(size));
				std::fflush(stdout);
			}
		}
		return !short_read;
	}
#endif
	NativeGuardedAccess(from, size, [&] { std::memcpy(to, from, size); });
	return true;
}

void NativeWindowRun(const void* guest, uint64_t size, void (*access)(void*), void* context) { // (tiron8)
	if (size == 0) return;
	const auto* from = static_cast<const uint8_t*>(guest);
#if defined(_WIN32)
	// (raiz11, as NativeGuardedRead) Checked against the put-ins of kept GPU bytes under way: one overlapping it may have
	// shown the copy the pages' old bytes; read again then (KYTY_RAIZ_STALE_FIX), the gate waiting for the put-in to end.
	NativeReadTicket ticket;
	if (Raiz::On()) NativeReadTicketTake(reinterpret_cast<uint64_t>(from), size, &ticket);
	NativeGuardedAccess(from, size, [&] { access(context); });
	if (Raiz::On() && !NativeReadTicketClean(ticket)) {
		const bool fix = NativeStaleFixOn();
		NativeNoteStale(Raiz::StaleUpload, reinterpret_cast<uint64_t>(from), size, static_cast<uint64_t>(Raiz::t_site), fix);
		if (fix) NativeGuardedAccess(from, size, [&] { access(context); });
	}
#else
	NativeGuardedAccess(from, size, [&] { access(context); });
#endif
}

const uint8_t* TryGetUploadSourcePointer(uint64_t vaddr, uint64_t size) {
	if (g_guest_address_space == nullptr) return nullptr;
	if (!Raiz::On()) return g_guest_address_space->BackingPointer(vaddr, size);
	t_native_allow_protected_pointer = true;
	const auto* pointer              = g_guest_address_space->BackingPointer(vaddr, size);
	t_native_allow_protected_pointer = false;
	return pointer;
}

bool TryWriteCpuBacking(uint64_t vaddr, const void* data, uint64_t size) {
	Raiz::SiteScope site(Raiz::SiteCpuWrite); // (raiz2: the "[raiz]" line's write windows by source)
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->TryWriteCpuBacking(vaddr, data, size);
}

[[gnu::noinline]] bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	// Diagnostic (live timecensus, source BackingRead): which code reads the backing, how much.
	if (size >= 0x10000) KYTY_TIME_CENSUS(BackingRead, size);
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->TryReadBacking(vaddr, data, size);
}

const uint8_t* TryGetBackingPointer(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space != nullptr ? g_guest_address_space->BackingPointer(vaddr, size) : nullptr;
}

bool NativeMemoryEnabled() {
	return NativeMemoryRequested();
}

bool NativeReadFaultsEnabled() {
	return NativeReadFaultsRequested();
}

void NativeProtectCounters(uint64_t* protects, uint64_t* copy_protects, uint64_t* copies) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (protects != nullptr) *protects = g_native_protects.load(std::memory_order_relaxed);
	if (copy_protects != nullptr) *copy_protects = g_native_copy_protects.load(std::memory_order_relaxed);
	if (copies != nullptr) *copies = g_native_guarded_copies.load(std::memory_order_relaxed);
#else
	if (protects != nullptr) *protects = 0;
	if (copy_protects != nullptr) *copy_protects = 0;
	if (copies != nullptr) *copies = 0;
#endif
}

bool NativeWriteWatch(uint64_t vaddr, uint64_t size, native_write_watch_func_t callback, void* user, bool reset) {
	if (g_guest_address_space == nullptr || !g_guest_address_space->IsNative()) return false;
	// (The pages gathered under the address space's lock, given to the callback after it.)
	thread_local std::vector<uint64_t> pages;
	pages.clear();
	const bool ok = g_guest_address_space->NativeWrittenPages(vaddr, size, reset, &pages);
	if (callback != nullptr && !pages.empty()) {
		const std::vector<uint64_t> written = std::move(pages); // (a callback may call NativeWriteWatch again)
		pages.clear();
		for (const auto page: written) callback(page, user);
	}
	return ok;
}

bool NativeWriteWatchRanges(const std::pair<uint64_t, uint64_t>* ranges, size_t count, bool reset, std::vector<uint64_t>* pages,
                            std::vector<uint32_t>* failed) {
	if (g_guest_address_space == nullptr || !g_guest_address_space->IsNative() || ranges == nullptr || pages == nullptr) return false;
	return g_guest_address_space->NativeWrittenRanges(ranges, count, reset, pages, failed);
}

void NativeWriteWatchCounters(uint64_t* calls, uint64_t* retries) {
	if (calls != nullptr) *calls = g_native_watch_calls.load(std::memory_order_relaxed);
	if (retries != nullptr) *retries = g_native_watch_retries.load(std::memory_order_relaxed);
}

bool PrefaultGuestRange(const void* address, uint64_t size) {
	return GuestLazyCommit::Prefault(address, size);
}

// (KYTY_GUEST_LAZY_COMMIT: the host-memory readability queries commit the never-touched guest pages they look at.)
[[maybe_unused]] static const bool g_host_memory_prefault_set = [] {
	Libs::Graphics::SetHostMemoryPrefault(&PrefaultGuestRange);
	return true;
}();

uint64_t BackingMappingEnd(uint64_t vaddr) {
	return g_guest_address_space != nullptr ? g_guest_address_space->BackingMappingEnd(vaddr) : 0;
}

void TrimGuestRange(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (size != 0) ParkedRangeTrimmer::Trim(vaddr, size);
#else
	(void)vaddr, (void)size;
#endif
}

bool TryGetBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
	return g_guest_address_space != nullptr && g_guest_address_space->BackingPieces(vaddr, size, pieces);
}

bool TryGetSparseBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
	return g_guest_address_space != nullptr && g_guest_address_space->SparseBackingPieces(vaddr, size, pieces);
}

bool TryReadBackingToHost(uint64_t vaddr, void* data, uint64_t size) {
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->TryReadBackingToHost(vaddr, data, size);
}

// A speculative translation's CPU read of guest memory no GPU work wrote: not memory its own work or its guest
// memory writes write (both wait for its commit), and noted (speculation-state.h: a hole that writes it stops the
// commit before the reading segment).
static bool SpeculationReadsClean(uint64_t vaddr, uint64_t size) {
	auto* spec = Graphics::Spec::Current();
	if (spec == nullptr) return true;
	if (spec->WrittenIntersects(vaddr, size) || spec->JournalIntersects(vaddr, size)) return false;
	spec->NoteCleanRead(vaddr, size);
	return true;
}

bool TryReadGpuCleanBackingToHost(uint64_t vaddr, void* data, uint64_t size) {
	// (FRAME-NATIVO2 v11b, =4) Bloques sin vigilar de instancias de solo lectura: la GPU no escribe alli (red de seguridad
	// en BufferCache), asi que un estado "escrito por la GPU" es de un dueno anterior: se lee siempre la memoria real del
	// juego (antes: rechazado -> "a descriptor source did not evaluate").
	if (::FrameNativo2::SoloLectura() && ::FrameNativo2::RoAll(vaddr, size) && SpeculationReadsClean(vaddr, size)) {
		::FrameNativo2::g_ro_desc_reads.fetch_add(1, std::memory_order_relaxed);
		return TryReadBackingToHost(vaddr, data, size);
	}
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		if (!Graphics::GuestGpu::IsTranslatingThread() ||
		    GetGpuResources().GetBufferCache().HasGpuDirtyBytes(vaddr, size) ||
		    GetGpuResources().GetTextureCache().IsRegionGpuModified(vaddr, size) || !SpeculationReadsClean(vaddr, size)) {
			return false;
		}
	}
	return TryReadBackingToHost(vaddr, data, size);
}

bool IsUniqueGuestBackingRange(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space != nullptr && g_guest_address_space->IsUniqueBackingRange(vaddr, size);
}

bool GuestBackingAliases(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>* aliases) {
	return g_guest_address_space != nullptr && g_guest_address_space->BackingAliases(vaddr, size, aliases);
}

bool IsUniqueGuestBackingRangeAcrossMappings(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space != nullptr && g_guest_address_space->IsUniqueBackingRangeAcrossMappings(vaddr, size);
}

// `data` is host memory (every caller reads into a local or a scratch vector): the lock-free
// host read applies; the guest-destination variant took both backing locks for each word.
bool TryReadGpuCleanBackingOnWatchedPage(uint64_t vaddr, void* data, uint64_t size) {
	return g_gpu_resources != nullptr && Graphics::GuestGpu::IsTranslatingThread() &&
	       IsGpuAddressRange(vaddr, size) && g_gpu_resources->HasReadWatchers(vaddr, size) &&
	       TryReadGpuCleanBackingToHost(vaddr, data, size);
}

bool ReadFaults(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) && g_gpu_resources->HasReadWatchers(vaddr, size);
}

bool TryReadGpuShaderSpan(uint64_t vaddr, void* data, uint64_t size, bool clean) {
	if (!data || size < 8 || size > (clean ? 4096u : 64u) || size % 4 != 0 || !g_gpu_resources ||
	    !Graphics::GuestGpu::IsTranslatingThread() || !IsGpuAddressRange(vaddr, size))
		return false;
	if (!clean && !g_gpu_resources->HasReadWatchers(vaddr, size) && SpeculationReadsClean(vaddr, size)) {
		// 8..64 bytes of whole words: fixed-size moves inline, a variable-size memcpy is a CRT call
		// (thousands of SRT span reads per frame).
		auto*       out = static_cast<uint8_t*>(data);
		const auto* in  = reinterpret_cast<const uint8_t*>(vaddr);
		uint64_t    at  = 0;
		for (; at + 8 <= size; at += 8) std::memcpy(out + at, in + at, 8);
		if (at < size) std::memcpy(out + at, in + at, 4);
		return true;
	}
	return TryReadGpuCleanBackingToHost(vaddr, data, size);
}

thread_local uint64_t g_srt_shader_hash = 0;

bool SyncGpuCleanBacking(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return true;
	}
	// (FRAME-NATIVO2 v11b, =4) En bloques sin vigilar nunca se descarga lo de la GPU: pisaria lo que el juego escribio.
	if (::FrameNativo2::SoloLectura() && ::FrameNativo2::RoAll(vaddr, size)) {
		::FrameNativo2::g_ro_sync_skips.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	if (Graphics::Spec::Current() != nullptr) {
		// A speculative translation (speculation-state.h) reads guest memory as it is: words the GPU wrote need the
		// GPU first (the translation that follows waits for it).
		if (GetGpuResources().GetTextureCache().IsRegionGpuModified(vaddr, size) ||
		    GetGpuResources().GetBufferCache().HasGpuDirtyBytes(vaddr, size) ||
		    (!Graphics::GuestGpu::IsGpuThread() && ReadFaults(vaddr, size)) || !SpeculationReadsClean(vaddr, size))
			Graphics::Spec::Refuse("GPU-written guest memory");
		return true;
	}
	if (!Graphics::GuestGpu::IsGpuThread() ||
	    GetGpuResources().GetTextureCache().IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	// No page of it read-watched: no GPU-written bytes of a buffer over it (their pages are, from the write on: a CPU
	// read faults), as NativeXprReader's direct reads take it. (The command processor asks for every indirect register
	// table, ~11K a frame at 1-1, in memory the CPU writes: each a search of the GPU-written ranges.)
	if (!GetGpuResources().HasReadWatchers(vaddr, size)) return true;
	if (GetGpuResources().GetBufferCache().HasGpuDirtyBytes(vaddr, size)) {
		// KYTY_SYNC_LOG: which shader's resource evaluation drains the GPU for its words.
		static const bool log = std::getenv("KYTY_SYNC_LOG") != nullptr;
		static std::atomic<uint32_t> logged {0};
		if (log && logged.fetch_add(1, std::memory_order_relaxed) < 400) {
			std::printf("SRT sync download: shader=0x%016llx addr=0x%llx size=0x%llx\n",
			            static_cast<unsigned long long>(g_srt_shader_hash), static_cast<unsigned long long>(vaddr),
			            static_cast<unsigned long long>(size));
		}
		GetGpuResources().GetBufferCache().ReadMemory(vaddr, size);
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// (raiz11, KYTY_RAIZ_STALE_CHECK) Read-watched bytes the caller reads straight after this (indirect arguments, tables):
	// a put-in of kept GPU bytes under way over them shows their OLD bytes (readable already, the GPU's not in yet).
	// Counted; waited for (KYTY_RAIZ_STALE_FIX: at most 20 ms, the put-in is microseconds).
	if (NativePutInBusy(vaddr, size)) {
		const bool fix = NativeStaleFixOn();
		NativeNoteStale(Raiz::StaleGpuSync, vaddr, size, g_srt_shader_hash, fix); // (first: whether it covers these pages)
		if (fix) {
			const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
			while (NativePutInBusy(vaddr, size) && std::chrono::steady_clock::now() < until) std::this_thread::yield();
		}
	}
#endif
	return true;
}

// (raiz12, KYTY_RAIZ_SETTLE; local/raiz.h) A game thread about to read [vaddr, vaddr + size) of GPU results (a hook at
// the reader's entry): with mode 1, what the GPU still owns of it read back (the readback a read fault would make, once
// for the range), its deferred releases applied by this thread (the put-ins of their kept bytes, under the page
// manager's locks: a put-in another thread holds is waited for there), then until no put-in is under way over it and
// no kept bytes are left over it (at most 4 ms; NativeSettle). After that no page of it shows old bytes: each is either
// in with the GPU's bytes or no-access (a read faults and gets them). A put-in starts only on pages with kept bytes; none
// is kept over the range until the GPU writes it again, which the game orders after its reads (it waits for the GPU
// work first). Mode 2: only the checks at entry, counted. `site`: Raiz::SettleSiteKind. Returns RaizSettle* flags.
uint32_t RaizSettleRange(uint64_t vaddr, uint64_t size, uint32_t site, int mode) noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static_assert(uint32_t {RaizSettleBusy} == uint32_t {NativeSettleBusy} && uint32_t {RaizSettlePending} == uint32_t {NativeSettlePending} &&
	              uint32_t {RaizSettleReadback} == uint32_t {NativeSettleReadback} && uint32_t {RaizSettleWaited} == uint32_t {NativeSettleWaited} &&
	              uint32_t {RaizSettleLeft} == uint32_t {NativeSettleLeft});
	if (!NativeReadFaultsRequested()) return 0;
	return NativeSettle(
	    vaddr, size, site,
	    [&] {
		    if (ReadFaults(vaddr, size)) return TryReadbackRange(vaddr, size);
		    // (SIN-VIGILAR-GANCHO) m_noread pages (CPUHEAP-/DEFAULTTAGG-SIN-VIGILAR): readable with old bytes, no fault
		    // reads them back; the XPR chunk bits 0xa1f290 reads live in DefaultTagg (RenderSingleFrameCPU, Ghidra 0xaa2050).
		    if (!TryReadbackRangeUnwatched(vaddr, size)) return false;
		    if (site == Raiz::SettleChunkBits) ::Parpadeo6::g_c.unwatched.fetch_add(1, std::memory_order_relaxed);
		    return true;
	    },
	    [&] { Graphics::PageManager::ApplyDeferredReleases(vaddr, size); }, mode);
#else
	(void)vaddr, (void)size, (void)site, (void)mode;
	return 0;
#endif
}

uint64_t MappingEnd(uint64_t vaddr) {
	return g_virtual_ranges != nullptr ? g_virtual_ranges->MappingEnd(vaddr) : 0;
}

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	EXIT_IF(g_virtual_ranges == nullptr);

	const auto clamped_size = g_virtual_ranges->ClampRangeSize(vaddr, size);
	if (clamped_size == 0) {
		// A descriptor over memory the guest has released (a dispatch in a newly streamed area
		// stopped the emulator here): the caller binds nothing, as for a null descriptor.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			std::printf("Memory: buffer range at an unmapped address ignored, addr=0x%016" PRIx64 " size=0x%016" PRIx64
			            "\n",
			            vaddr, size);
		}
		return 0;
	}
	if (clamped_size != size) {
		LOGF("Memory: clamped buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     " to 0x%016" PRIx64 "\n",
		     vaddr, size, clamped_size);
	}
	return clamped_size;
}

bool IsFullyMapped(uint64_t vaddr, uint64_t size) {
	return g_virtual_ranges == nullptr || g_virtual_ranges->ClampRangeSize(vaddr, size) == size;
}

void MappedParts(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>* parts) {
	parts->clear();
	if (g_virtual_ranges == nullptr) {
		parts->emplace_back(vaddr, size);
		return;
	}
	std::vector<VirtualRanges::Part> committed;
	g_virtual_ranges->CommittedParts(vaddr, size, &committed);
	for (const auto& part: committed) {
		if (!parts->empty() && parts->back().first + parts->back().second == part.address) {
			parts->back().second += part.size;
		} else {
			parts->emplace_back(part.address, part.size);
		}
	}
}

// Private memory has no backing view: written at its own address, each page made writable for the
// copy (and executable: it may be code another thread runs) and given its protection back.
static bool WritePrivate(uint64_t vaddr, const uint8_t* data, uint64_t size) {
	constexpr uint64_t PageSize = 0x1000;
	for (uint64_t at = vaddr, end = vaddr + size; at < end;) {
		const auto          next = std::min((at & ~(PageSize - 1)) + PageSize, end);
		VirtualMemory::Mode mode = VirtualMemory::Mode::NoAccess;
		if (!VirtualMemory::Protect(at, next - at, VirtualMemory::Mode::ExecuteReadWrite, &mode)) {
			return false;
		}
		std::memcpy(reinterpret_cast<void*>(at), data + (at - vaddr), next - at);
		if (mode != VirtualMemory::Mode::ExecuteReadWrite) VirtualMemory::Protect(at, next - at, mode);
		at = next;
	}
	return true;
}

bool PrintWriteBacksOverlapping(uint64_t vaddr, uint64_t size) {
	const auto count = g_write_back_count.load(std::memory_order_acquire);
	const auto now   = std::chrono::steady_clock::now().time_since_epoch().count();
	bool       any   = false;
	for (uint64_t i = count; i > 0 && count - i < WriteBackRecords; --i) {
		const auto record = g_write_backs[(i - 1) % WriteBackRecords];
		if (record.size == 0 || record.vaddr >= vaddr + size || vaddr >= record.vaddr + record.size) continue;
		std::printf("  write-back %s 0x%016" PRIx64 "+0x%" PRIx64 " %.1f ms before (#%" PRIu64 " of %" PRIu64 ")\n",
		            record.source != nullptr ? record.source : "?", record.vaddr, record.size,
		            static_cast<double>(now - record.ns) / 1e6, i, count);
		any = true;
	}
	return any;
}

void WriteBacking(uint64_t vaddr, const void* data, uint64_t size, const char* source) noexcept {
	Raiz::SiteScope site(Raiz::SiteOf(source)); // (raiz2: the "[raiz]" line's write windows by source)
	const auto index = g_write_back_count.fetch_add(1, std::memory_order_relaxed);
	g_write_backs[index % WriteBackRecords] = {std::chrono::steady_clock::now().time_since_epoch().count(),
	                                          vaddr, size, source};
	// (Diagnostic, KYTY_WRITEBACK_VERIFY: the GPU-owned pages checked before, hashed again after.)
	struct Verify {
		uint64_t vaddr, size;
		bool     on;
		~Verify() {
			if (on) WritebackVerify::Rehash(vaddr, size);
		}
	} verify {vaddr, size, WritebackVerify::Enabled()};
	// (crash-ba2b1f, KYTY_RAIZ_WB_GUARD) This write is the emulator's: the write watch of its pages forgets it (after it, on
	// every path out), so the next write-back over them does not take it for the game's.
	struct GuardForget {
		uint64_t vaddr, size;
		bool     on;
		~GuardForget() {
			if (on) RaizWbEmulatorWrote(vaddr, size);
		}
	} guard_forget {vaddr, size, RaizWbGuardOn()};
	if (verify.on) WritebackVerify::Check(vaddr, static_cast<const uint8_t*>(data), size, source);
	if (TryWriteBacking(vaddr, data, size)) {
		return;
	}
	// GPU results (readbacks, downloads) the backing view does not take whole: private memory (a
	// thread's stack or TLS, where the game can have the GPU write a fence or a result) is written at
	// its own address. What the guest unmapped meanwhile, or what takes neither write, is dropped.
	std::vector<VirtualRanges::Part> parts;
	if (g_virtual_ranges != nullptr) g_virtual_ranges->CommittedParts(vaddr, size, &parts);
	uint64_t written = 0;
	for (const auto& part: parts) {
		const auto* source = static_cast<const uint8_t*>(data) + (part.address - vaddr);
		bool        done   = part.private_memory ? WritePrivate(part.address, source, part.size)
		                                         : TryWriteBacking(part.address, source, part.size);
		if (!done && !part.private_memory && g_guest_address_space != nullptr) {
			// A partial unmap of the part's view (Windows) takes the mapping entries of its other parts away until they
			// are mapped again, under the address-space lock: written once that change is done.
			g_guest_address_space->WaitForChanges(part.address, part.size);
			done = TryWriteBacking(part.address, source, part.size);
			if (done) LiveCounters::Add(LiveCounters::WriteBackRetries);
		}
		if (done) written += part.size;
	}
	if (written == size) {
		return;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
		std::printf("Memory: GPU data for guest memory dropped, addr=0x%016" PRIx64 " size=0x%" PRIx64
		            " (0x%" PRIx64 " bytes written; committed parts %zu) caller=%p\n",
		            vaddr, size, written, parts.size(), __builtin_return_address(0));
	}
}

// ---- (crash-ba2b1f) KYTY_RAIZ_WB_GUARD (local/raiz-wb-guard.h) -------------------------------------------------------
namespace {
std::mutex                   g_wbg_mutex;  // (a leaf)
std::unordered_set<uint64_t> g_wbg_marked; // pages a game write was found on where it should have faulted
std::atomic<uint32_t>        g_wbg_marked_count {0};
std::atomic<uint64_t>        g_wbg_skipped {0}, g_wbg_overwritten {0}, g_wbg_marks {0}, g_wbg_race_pages {0},
    g_wbg_race_words {0}, g_wbg_race_after {0}, g_wbg_owned_calls {0}, g_wbg_emulator_calls {0};
std::atomic<int64_t>         g_wbg_last_summary {0};
std::atomic<RaizPagePeek>    g_wbg_peek {nullptr};
std::atomic<void*>           g_wbg_peek_user {nullptr};

// A line every 5 s at most, only when something was found (called from the paths that find it).
void WbgSummary() {
	const int64_t now  = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	int64_t       last = g_wbg_last_summary.load(std::memory_order_relaxed);
	if (now - last < 5 || !g_wbg_last_summary.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
	std::printf("[wb-guard] (crash-ba2b1f) totals: write-back pages left alone because the game wrote them while GPU-owned "
	            "%llu (counted only, KYTY_RAIZ_WB_GUARD=2: %llu); put-ins where a game write landed in the open page "
	            "%llu pages (%llu words kept the game's; %llu pages written after the copy); pages marked %llu; resets: "
	            "GPU took %llu, emulator wrote %llu\n",
	            static_cast<unsigned long long>(g_wbg_skipped.load()), static_cast<unsigned long long>(g_wbg_overwritten.load()),
	            static_cast<unsigned long long>(g_wbg_race_pages.load()), static_cast<unsigned long long>(g_wbg_race_words.load()),
	            static_cast<unsigned long long>(g_wbg_race_after.load()), static_cast<unsigned long long>(g_wbg_marks.load()),
	            static_cast<unsigned long long>(g_wbg_owned_calls.load()), static_cast<unsigned long long>(g_wbg_emulator_calls.load()));
	std::fflush(stdout);
}
} // namespace

int RaizWbGuardMode() noexcept {
	static const int mode = [] {
		if (!Raiz::On() || !NativeMemoryEnabled()) return 0;
		const char* v = std::getenv("KYTY_RAIZ_WB_GUARD");
		const int   m = (v == nullptr || v[0] == '\0') ? 1 : (v[0] == '0' ? 0 : v[0] == '2' ? 2 : 1);
		std::printf("[wb-guard] KYTY_RAIZ_WB_GUARD=%d (crash-ba2b1f): %s\n", m,
		            m == 0   ? "off"
		            : m == 1 ? "a GPU write-back leaves alone the GPU-owned pages the game wrote since the GPU took them (=2: "
		                       "only counted, =0: off)"
		                     : "the GPU-owned pages the game wrote since the GPU took them are counted (written back as before)");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

void RaizWbGpuOwned(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0 || !RaizWbGuardOn()) return;
	NativeAnnounceReset(vaddr, size);
	g_wbg_owned_calls.fetch_add(1, std::memory_order_relaxed);
	if (g_wbg_marked_count.load(std::memory_order_relaxed) == 0) return;
	std::lock_guard lock(g_wbg_mutex);
	for (uint64_t page = vaddr & ~uint64_t {0xfff}; page < vaddr + size; page += 0x1000) g_wbg_marked.erase(page);
	g_wbg_marked_count.store(static_cast<uint32_t>(g_wbg_marked.size()), std::memory_order_relaxed);
}

void RaizWbEmulatorWrote(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0 || !RaizWbGuardOn()) return;
	NativeAnnounceReset(vaddr, size);
	g_wbg_emulator_calls.fetch_add(1, std::memory_order_relaxed);
}

void RaizWbNoteGameWrote(uint64_t page, const char* where) noexcept {
	page &= ~uint64_t {0xfff};
	if (ArenaDueno::On()) ArenaDueno::OnWbGuard(page); // (ARENA-DUENO)
	{
		std::lock_guard lock(g_wbg_mutex);
		g_wbg_marked.insert(page);
		g_wbg_marked_count.store(static_cast<uint32_t>(g_wbg_marked.size()), std::memory_order_relaxed);
	}
	// (Its GPU copy lacks the write: uploaded again before the next submission reads it - as a write fault would have
	// made it -. The GPU thread's resync, GpuResourceManager; a page still GPU-owned is read back first, the mark keeps
	// the write-back off it.)
	Raiz::QueueResync(page, page + 0x1000);
	static std::atomic<uint32_t> said {0};
	g_wbg_marks.fetch_add(1, std::memory_order_relaxed);
	if (said.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("[wb-guard] page 0x%llx: a game write with no fault (%s): marked, queued as CPU-written\n",
		            static_cast<unsigned long long>(page), where != nullptr ? where : "?");
		std::fflush(stdout);
	}
	WbgSummary();
}

void RaizWbGameWritten(uint64_t vaddr, uint64_t size, std::vector<uint64_t>* pages) noexcept {
	pages->clear();
	if (size == 0 || !RaizWbGuardOn()) return;
	constexpr uint64_t Max = 1024;
	uint64_t           found[Max];
	const uint64_t     total = NativeAnnounceWritten(vaddr, size, found, Max);
	for (uint64_t i = 0; i < total && i < Max; ++i) pages->push_back(found[i]);
	if (g_wbg_marked_count.load(std::memory_order_relaxed) != 0) {
		std::lock_guard lock(g_wbg_mutex);
		const uint64_t  first = vaddr & ~uint64_t {0xfff};
		if (((vaddr + size - first + 0xfff) >> 12u) <= g_wbg_marked.size()) {
			for (uint64_t page = first; page < vaddr + size; page += 0x1000)
				if (g_wbg_marked.count(page) != 0) pages->push_back(page);
		} else {
			for (const auto page: g_wbg_marked)
				if (page >= first && page < vaddr + size) pages->push_back(page);
		}
	}
	if (pages->size() > 1) {
		std::sort(pages->begin(), pages->end());
		pages->erase(std::unique(pages->begin(), pages->end()), pages->end());
	}
}

void RaizWbNoteSkipped(uint64_t page, uint64_t address, uint64_t size, const char* source, bool skipped) noexcept {
	(skipped ? g_wbg_skipped : g_wbg_overwritten).fetch_add(1, std::memory_order_relaxed);
	if (ArenaDueno::On()) ArenaDueno::OnWbGuard(page); // (ARENA-DUENO)
	// (Left alone: the GPU's copy of the page lacks the game's bytes - the reason it is here -: uploaded again before the
	// next submission reads it, as a write fault would have made it. The GPU thread's resync, GpuResourceManager.)
	if (skipped) Raiz::QueueResync(page, page + 0x1000);
	static std::atomic<uint32_t> said {0};
	if (said.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("[wb-guard] %s write-back of 0x%llx+0x%llx: page 0x%llx written by the game while GPU-owned (no fault): "
		            "%s\n",
		            source != nullptr ? source : "?", static_cast<unsigned long long>(address),
		            static_cast<unsigned long long>(size), static_cast<unsigned long long>(page),
		            skipped ? "LEFT ALONE (the game's bytes kept)" : "written over as before (KYTY_RAIZ_WB_GUARD=2)");
		std::fflush(stdout);
	}
	WbgSummary();
}

void RaizWbNotePutInRace(uint64_t page, uint64_t words, bool after) noexcept {
	g_wbg_race_pages.fetch_add(1, std::memory_order_relaxed);
	if (ArenaDueno::On()) ArenaDueno::OnWbGuard(page); // (ARENA-DUENO)
	g_wbg_race_words.fetch_add(words, std::memory_order_relaxed);
	if (after) g_wbg_race_after.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> said {0};
	if (said.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("[wb-guard] put-in of kept GPU bytes into page 0x%llx: the game wrote the open page %s (%llu words of "
		            "8 bytes kept the game's)\n",
		            static_cast<unsigned long long>(page), after ? "after the copy, before its protection" : "during the copy",
		            static_cast<unsigned long long>(words));
		std::fflush(stdout);
	}
	WbgSummary();
}

void RaizWbSetPagePeek(RaizPagePeek peek, void* user) noexcept {
	g_wbg_peek_user.store(user, std::memory_order_relaxed);
	g_wbg_peek.store(peek, std::memory_order_release);
}

// (Crash reports: no lock waited for - the crashing thread may hold any of them -: try_lock, "?" when taken.)
void PrintGuestPageState(uint64_t page) noexcept {
	page &= ~uint64_t {0xfff};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	char                     host[96] = "?";
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<const void*>(page), &info, sizeof(info)) == sizeof(info)) {
		const DWORD p    = info.Protect & 0xffu;
		const char* prot = p == PAGE_NOACCESS ? "no-access" : p == PAGE_READONLY ? "read-only" : p == PAGE_READWRITE ? "read-write"
		                 : p == PAGE_EXECUTE_READ ? "exec-read" : p == PAGE_EXECUTE_READWRITE ? "exec-read-write" : "other";
		std::snprintf(host, sizeof(host), "%s %s%s (0x%lx)",
		              info.State == MEM_COMMIT ? "committed" : info.State == MEM_RESERVE ? "reserved" : "free",
		              info.Type == MEM_PRIVATE ? "private " : info.Type == MEM_MAPPED ? "mapped " : "", prot,
		              static_cast<unsigned long>(info.Protect));
	}
	char       tracker[128] = "untracked";
	uint32_t   bits         = 0;
	const auto peek         = g_wbg_peek.load(std::memory_order_acquire);
	if (peek != nullptr && peek(g_wbg_peek_user.load(std::memory_order_relaxed), page, &bits) && (bits & RaizPeekRegion) != 0) {
		std::snprintf(tracker, sizeof(tracker), "%s%s%s%s%s%s", (bits & RaizPeekCpuDirty) ? "cpu-dirty " : "clean ",
		              (bits & RaizPeekGpuDirty) ? "GPU-DIRTY " : "", (bits & RaizPeekWritable) ? "writable " : "write-watched ",
		              (bits & RaizPeekReadable) ? "readable" : "read-watched", (bits & RaizPeekNoWatch) ? " no-watch" : "",
		              (bits & RaizPeekNoSplit) ? " refused" : "");
	}
	const bool read_watch = g_gpu_resources != nullptr && g_gpu_resources->HasReadWatchers(page, 0x1000);
	const char* kept = "0";
	if (g_pending_count.load(std::memory_order_relaxed) != 0) {
		if (g_pending_mutex.try_lock()) {
			kept = g_pending.count(page) != 0 ? "YES" : "no";
			g_pending_mutex.unlock();
		} else {
			kept = "?";
		}
	}
	const char* watch = "n/a";
	{
		void*     found[1] {};
		ULONG_PTR count       = 1;
		DWORD     granularity = 0;
		if (GetWriteWatch(0, reinterpret_cast<void*>(page), 0x1000, found, &count, &granularity) == 0) watch = count != 0 ? "SET" : "clear";
	}
	const char* mark = "no";
	if (g_wbg_marked_count.load(std::memory_order_relaxed) != 0) {
		if (g_wbg_mutex.try_lock()) {
			mark = g_wbg_marked.count(page) != 0 ? "YES" : "no";
			g_wbg_mutex.unlock();
		} else {
			mark = "?";
		}
	}
	std::printf("  page 0x%016llx: host %s | tracker %s | page manager read watcher %s | kept GPU bytes %s | write watch %s "
	            "(since the GPU took it, with KYTY_RAIZ_WB_GUARD) | guard mark %s\n",
	            static_cast<unsigned long long>(page), host, tracker, read_watch ? "yes" : "no", kept, watch, mark);
#else
	std::printf("  page 0x%016llx\n", static_cast<unsigned long long>(page));
#endif
}

void InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	(void)GetGpuResources().InvalidateMemory(vaddr, size);
}

void InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	(void)GetGpuResources().InvalidateMemoryForWrite(vaddr, size);
}

bool PrepareHostOverwrite(uint64_t vaddr, uint64_t size) {
	if (size == 0 || UINT64_MAX - vaddr < size || g_gpu_resources == nullptr) return false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// From the first page a store would fault on to the end of the last one, of those the trackers protect: by the
	// pages' state. (A VirtualQuery walk cost more than the reads at start-up - the kernel scans a region's page table to
	// its end - 88% of the AMPR thread's time while the game loaded on a laptop.) A page the guest itself protects faults
	// as before.
	uint64_t first = 0, end = 0;
	if (g_gpu_resources->WatchedSpan(vaddr, size, &first, &end)) {
		first = std::max(first, vaddr);
		end   = std::min(end, vaddr + size);
		(void)g_gpu_resources->InvalidateMemoryForWrite(first, end - first);
		return true;
	}
	return false;
#else
	return false;
#endif
}

void FinishHostOverwrite(uint64_t vaddr, uint64_t size, bool prepared) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HOST_WRITE_REFRESH");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled || size == 0 || UINT64_MAX - vaddr < size || g_gpu_resources == nullptr) return;
	if (!prepared && size < 256u * 1024u) return;
	// Buffers too, not only images: the GPU-decoded archstone videos read their bitstream (streamed in by file reads)
	// as a storage buffer, and a stale copy of it decoded as garbage until the next key frame.
	(void)TryPrepareHostOverwrite(vaddr, size); // (the bytes there are the new ones: GPU copies are dropped, not read back)
}

bool TryPrepareHostWrite(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) &&
	       g_gpu_resources->InvalidateMemory(vaddr, size);
}

bool TryPrepareHostImageWrite(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) && g_gpu_resources->InvalidateImages(vaddr, size);
}

bool TryPrepareHostOverwrite(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) &&
	       g_gpu_resources->InvalidateMemoryForWrite(vaddr, size);
}

void InstallGpuResources(Graphics::GpuResourceManager* resources) noexcept {
	EXIT_IF(resources != nullptr && g_gpu_resources != nullptr);
	g_gpu_resources = resources;
}

bool IsGuestMappedAfterChanges(uint64_t vaddr) {
	return g_guest_address_space != nullptr && g_guest_address_space->IsMappedAfterChanges(vaddr);
}

bool InRemapWindow(uint64_t vaddr) noexcept {
	const auto begin = g_remap_window_begin.load(std::memory_order_acquire);
	return begin != 0 && vaddr >= begin && vaddr < g_remap_window_end.load(std::memory_order_relaxed);
}

bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	return g_gpu_resources != nullptr && g_gpu_resources->HandleFault(access, fault_vaddr);
}

// (faltas-emu, KYTY_FALTAS_EMU; local/faltas-emu.h) The emulator is about to write [vaddr, vaddr + size) of guest memory
// itself (an Agc packet in the game's command buffer): each page of it the trackers protect now gets what its write fault
// would get (HandleFault, this thread), page by page, before the write instead of from inside an exception. Pages whose
// state is not protected are left alone (their write would not fault). Without host protection of written pages
// (KYTY_NATIVE_MEMORY unless KYTY_WRITE_WATCH_MODE=protect) nothing is done: there a write faults only on read-watched
// pages, as before.
uint32_t OpenWatchedForEmulatorWrite(uint64_t vaddr, uint64_t size) noexcept {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) return 0;
	if (Graphics::PageManager::NativeWriteWatch() && !Graphics::PageManager::WatchProtects()) return 0;
	uint32_t       n    = 0;
	const uint64_t end  = vaddr + size;
	for (uint64_t page = vaddr & ~uint64_t {4095}; page < end; page += 4096) {
		const uint64_t at = page < vaddr ? vaddr : page;
		uint64_t       first = 0, last = 0;
		if (!g_gpu_resources->WatchedSpan(at, 1, &first, &last)) continue;
		if (g_gpu_resources->HandleFault(Graphics::PageFaultAccess::Write, at)) ++n;
	}
	return n;
}

bool TryReadbackRange(uint64_t vaddr, uint64_t size) noexcept {
	return g_gpu_resources != nullptr && size != 0 && IsGpuAddressRange(vaddr, size) && g_gpu_resources->HasReadWatchers(vaddr, size) &&
	       g_gpu_resources->ReadbackRange(vaddr, size);
}

// (SIN-VIGILAR-GANCHO) Cheap gate first: only granules labelled cpuheap / DefaultTagg with their SIN-VIGILAR switch on can
// hold m_noread pages (RegionManager::UpdateGpuProtection decides with the same FrameNativo::Classify), one relaxed load
// per 64 KiB granule; anything else returns before the page manager and the buffer cache are asked.
static bool MaybeUnwatchedGpuPages(uint64_t vaddr, uint64_t size) noexcept {
	// (11-26 03:28) APAGADO por defecto (KYTY_LECTURA_SIN_VIGILANTE=1 lo enciende): con el rompia el culling (geometria que desaparece).
	static const bool kOn = [] { const char* v = std::getenv("KYTY_LECTURA_SIN_VIGILANTE"); return v != nullptr && v[0] == '1'; }();
	if (!kOn) return false;
	const bool cpuheap = ::CpuheapSinVigilar::On(), tagg = ::DefaultTaggSinVigilar::On();
	if ((!cpuheap && !tagg) || size == 0 || size > (uint64_t {64} << 20u)) return false;
	for (uint64_t g = vaddr & ~0xffffull; g < vaddr + size; g += 0x10000) {
		const auto cls = FrameNativo::Classify(g);
		if ((cls == FrameNativo::CCpuheap && cpuheap) || (cls == FrameNativo::CDefaultTagg && tagg)) return true;
	}
	return false;
}

bool TryReadbackRangeUnwatched(uint64_t vaddr, uint64_t size) noexcept {
	return g_gpu_resources != nullptr && MaybeUnwatchedGpuPages(vaddr, size) && IsGpuAddressRange(vaddr, size) &&
	       !g_gpu_resources->HasReadWatchers(vaddr, size) &&
	       g_gpu_resources->ReadbackRange(vaddr, size);
}

struct PrtAperture {
	uint64_t address = 0;
	uint64_t size    = 0;
};

constexpr int      PRT_APERTURE_MAX_INDEX = 2;
constexpr uint64_t PRT_PAGE_SIZE          = 0x4000;
constexpr uint64_t PRT_APERTURE_START     = 0x0f00000000ull;
constexpr uint64_t PRT_APERTURE_END       = 0xfc00000000ull;

static std::array<PrtAperture, PRT_APERTURE_MAX_INDEX + 1> g_prt_apertures {};
static Common::Mutex                                       g_prt_aperture_mutex;
// PRT pool residency has two representations: logical ranges and host backing.
// Publish their transitions together. Never hold this across Map/UnmapGpuRange:
// the renderer can read PRT while a guest writer waits for SendCommandSync.
static std::mutex                                          g_prt_residency_mutex;

static bool IsInPrtAperture(uint64_t address, uint64_t size = 1) {
	if (size == 0 || UINT64_MAX - address < size) {
		return false;
	}
	Common::LockGuard lock(g_prt_aperture_mutex);

	for (const auto& aperture: g_prt_apertures) {
		if (address >= aperture.address) {
			const auto offset = address - aperture.address;
			if (offset < aperture.size && size <= aperture.size - offset) {
				return true;
			}
		}
	}

	return false;
}

bool RestoreGuestWritable(uint64_t vaddr, uint64_t size) {
	VirtualRanges::Range range {};
	if (g_virtual_ranges == nullptr || g_guest_address_space == nullptr ||
	    !g_virtual_ranges->Query(vaddr, 0, &range) || (range.protection & PROT_CPU_WRITE) == 0 ||
	    range.start > vaddr || vaddr + size > range.start + range.size) {
		return false;
	}
	const auto mode = (range.protection & PROT_CPU_EXEC) != 0 ? VirtualMemory::Mode::ExecuteReadWrite
	                                                          : VirtualMemory::Mode::ReadWrite;
	return g_guest_address_space->ProtectTransient(vaddr, size, mode);
}

bool TryReadMappedOrZero(uint64_t vaddr, void* data, uint64_t size) {
	return g_guest_address_space != nullptr && g_guest_address_space->TryReadSparseBacking(vaddr, data, size);
}

bool TryReadPrtBacking(uint64_t vaddr, void* data, uint64_t size, const char** failure_reason) {
	const auto fail = [failure_reason](const char* reason) {
		if (failure_reason != nullptr) *failure_reason = reason;
		return false;
	};
	std::lock_guard residency_lock(g_prt_residency_mutex);
	std::vector<VirtualRanges::Range> ranges;
	if (g_guest_address_space == nullptr || g_virtual_ranges == nullptr)
		return fail("uninitialized-memory");
	if (!IsInPrtAperture(vaddr, size)) return fail("outside-aperture");
	if (!g_virtual_ranges->QuerySpan(vaddr, size, &ranges)) return fail("missing-logical-ownership");
	if (std::any_of(ranges.begin(), ranges.end(), [](const auto& range) {
		    return !IsReservedRangeType(range.type) &&
		           !g_guest_address_space->BackingContains(range.start, range.size);
	    })) {
		return fail("missing-committed-backing");
	}
	if (!g_guest_address_space->TryReadSparseBacking(vaddr, data, size))
		return fail("sparse-backing-copy-failed");
	if (failure_reason != nullptr) *failure_reason = nullptr;
	return true;
}

static bool SelfTestSub64SharedPlaceholderAlias() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	constexpr uint64_t PageSize    = 0x4000;
	const auto         granularity = g_guest_address_space->GetGranularity();
	if (granularity < PageSize * 2u) {
		LOGF_COLOR(
		    Log::Color::Yellow,
		    "\t direct-memory sub-64K placeholder self-test skipped: granularity too small\n");
		return true;
	}

	const auto base = FindGuestFreeRange(0, granularity, granularity);
	if (base == 0) {
		LOGF_COLOR(Log::Color::Red,
		           "\t direct-memory sub-64K placeholder self-test: reserve unavailable\n");
		return false;
	}

	const auto alias          = base + PageSize;
	bool       ok             = false;
	auto       failure_reason = GuestBackingStore::FailureReason::None;
	if (g_guest_address_space->MapBacking(alias, PageSize, PageSize, VirtualMemory::Mode::ReadWrite,
	                                      &failure_reason)) {
		auto* ptr = reinterpret_cast<uint64_t*>(alias);
		*ptr      = 0x4b59545953553634ull; // "KYTYSU64"
		ok        = (*ptr == 0x4b59545953553634ull);
		std::memset(ptr, 0, PageSize);

		// A second mapping of the same backing bytes: each is the other's alias (GuestBackingAliases), the first unique
		// again once it is gone.
		const auto second = base + 2u * PageSize;
		if (g_guest_address_space->IsNative()) {
			// (KYTY_NATIVE_MEMORY: no aliases - the second mapping is refused -, and the first stays unique.)
			g_native_alias_selftest = true; // (raiz2: its refusal is said as the self-test's)
			const bool refused =
			    !g_guest_address_space->MapBacking(second, PageSize, PageSize, VirtualMemory::Mode::ReadWrite, &failure_reason) &&
			    failure_reason == GuestBackingStore::FailureReason::NativeAliasRefused;
			g_native_alias_selftest = false;
			ok = ok && refused && g_guest_address_space->IsUniqueBackingRange(alias, PageSize) &&
			     g_guest_address_space->BackingPointer(alias, PageSize) == reinterpret_cast<const uint8_t*>(alias);
			if (refused) failure_reason = GuestBackingStore::FailureReason::None;
		} else if (g_guest_address_space->MapBacking(second, PageSize, PageSize, VirtualMemory::Mode::ReadWrite, &failure_reason)) {
			std::vector<std::pair<uint64_t, uint64_t>> aliases;
			ok = ok && !g_guest_address_space->IsUniqueBackingRange(alias, PageSize);
			ok = ok && g_guest_address_space->BackingAliases(alias, PageSize, &aliases) && aliases.size() == 1 &&
			     aliases[0] == std::pair<uint64_t, uint64_t> {second, PageSize};
			ok = ok && g_guest_address_space->BackingAliases(second + 0x1000, 0x1000, &aliases) && aliases.size() == 1 &&
			     aliases[0] == std::pair<uint64_t, uint64_t> {alias + 0x1000, 0x1000};
			ok = g_guest_address_space->UnmapBacking(second, PageSize) && ok;
			ok = ok && g_guest_address_space->IsUniqueBackingRange(alias, PageSize);
		} else {
			ok = false;
		}

		ok = g_guest_address_space->UnmapBacking(alias, PageSize) && ok;
	}

	LOGF_COLOR(
	    ok ? Log::Color::Green : Log::Color::Red,
	    "\t direct-memory sub-64K placeholder self-test: %s%s%s\n", ok ? "ok" : "failed",
	    ok ? "" : ", reason = ", ok ? "" : GuestBackingStore::GetFailureReasonName(failure_reason));
	return ok;
#else
	return true;
#endif
}

static bool ReplaceFixedRangeWithReserved(uint64_t start, uint64_t size);

static void GuestStoreTooSmall(const char* what, uint64_t len) {
	if (PhysicalMemory::TotalSize() >= static_cast<uint64_t>(13824) * 1024 * 1024) {
		return; // full-size store: the game's own failure, not a shrunken store
	}
	std::printf("Guest store too small: raise KYTY_GUEST_STORE_MB (a %s allocation of %llu MiB did not fit in a %llu MiB "
	            "store; %lld MiB allocated, peak %lld MiB)\n",
	            what, static_cast<unsigned long long>(len >> 20),
	            static_cast<unsigned long long>(PhysicalMemory::TotalSize() >> 20),
	            static_cast<long long>(g_guest_store_used.load() >> 20),
	            static_cast<long long>(g_guest_store_peak.load() >> 20));
	std::fflush(stdout);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// The periodic RAM flush (ram-guard.h, KYTY_RAM_FLUSH_SECONDS), guest side.
namespace RamFlushGuest {

constexpr uint64_t Chunk = uint64_t {8} << 20u;

uint64_t Micros(std::chrono::steady_clock::time_point start) {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

// Guest memory the game released (direct memory released, flexible memory unmapped) still committed in the store:
// its pages dropped (DiscardVirtualMemory: their contents are no one's - no mapping of them is left - and the next
// owner gets them as an allocation does: flexible memory zeroed by Map, direct memory as it was). Each chunk under the
// owner's lock with the range checked free, so an allocation of it waits for the drop to end. The commit stays (a
// page-file section's pages cannot be decommitted): the RAM in use drops. Ranges dropped are not looked at again
// while they stay free. (Guard thread only.)
uint64_t FlushFreed(uint64_t budget, uint64_t* held_us) {
	static std::vector<std::pair<uint64_t, uint64_t>> done; // backing offsets [begin, end) dropped or not committed
	static uint64_t                                   failures = 0, passes = 0;
	if (g_guest_address_space == nullptr || g_physical_memory == nullptr || g_flexible_memory == nullptr) return 0;
	auto* base = reinterpret_cast<uint8_t*>(g_guest_address_space->GetBackingBase());
	if (base == nullptr) return 0;
	std::vector<std::pair<uint64_t, uint64_t>> physical, flexible;
	g_physical_memory->FreeRanges(&physical);
	g_flexible_memory->FreeRanges(&flexible);
	const auto inside = [](const std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t b, uint64_t e) {
		auto it = std::upper_bound(ranges.begin(), ranges.end(), std::pair<uint64_t, uint64_t> {b, UINT64_MAX});
		if (it == ranges.begin()) return false;
		--it;
		return it->first <= b && it->first + it->second >= e;
	};
	// (What was taken again since is looked at anew.)
	std::erase_if(done, [&](const auto& r) { return !inside(physical, r.first, r.second) && !inside(flexible, r.first, r.second); });
	std::sort(done.begin(), done.end());
	uint64_t dropped = 0, held = 0;
	const auto walk  = [&](const std::vector<std::pair<uint64_t, uint64_t>>& ranges, bool is_physical) {
		for (const auto& [start, size]: ranges) {
			for (uint64_t at = start, end = start + size; at < end && dropped < budget;) {
				// (Past what was done already.)
				auto d = std::upper_bound(done.begin(), done.end(), std::pair<uint64_t, uint64_t> {at, UINT64_MAX});
				if (d != done.begin() && std::prev(d)->second > at) {
					at = std::prev(d)->second;
					continue;
				}
				uint64_t stop = std::min(end, (at & ~(Chunk - 1u)) + Chunk);
				if (d != done.end() && d->first < stop) stop = d->first;
				MEMORY_BASIC_INFORMATION info {};
				if (VirtualQuery(base + at, &info, sizeof(info)) != sizeof(info)) return;
				const uint64_t region_end = static_cast<uint64_t>(static_cast<uint8_t*>(info.BaseAddress) - base) + info.RegionSize;
				stop = std::min(stop, std::max(region_end, at + 1));
				if (info.State == MEM_COMMIT) {
					const auto t0   = std::chrono::steady_clock::now();
					const auto drop = [&] {
						if (DiscardVirtualMemory(base + at, stop - at) != ERROR_SUCCESS &&
						    VirtualAlloc(base + at, stop - at, MEM_RESET, PAGE_READWRITE) == nullptr)
							failures++;
					};
					const bool free_now = is_physical ? g_physical_memory->WhileFree(at, stop - at, drop)
					                                  : g_flexible_memory->WhileFree(at, stop - at, drop);
					held = std::max(held, Micros(t0));
					if (!free_now) {
						at = stop;
						continue;
					}
					dropped += stop - at;
				}
				done.emplace_back(at, stop);
				std::sort(done.begin(), done.end());
				at = stop;
			}
		}
	};
	walk(physical, true);
	walk(flexible, false);
	if (held_us != nullptr) *held_us = held;
	if (++passes == 1 || (passes % 60) == 0)
		std::printf("RAM flush: released guest memory, pass %llu: %llu MiB of it committed and dropped (free: direct %zu "
		            "ranges, flexible %zu ranges below the high-water mark), %llu drops refused by Windows in all\n",
		            static_cast<unsigned long long>(passes),
		            static_cast<unsigned long long>(dropped >> 20u), physical.size(), flexible.size(),
		            static_cast<unsigned long long>(failures));
	std::fflush(stdout);
	return dropped;
}

// KYTY_RAM_FLUSH_TRIM=1: the store's alias (the emulator's view of all guest memory) taken out of the working set, a
// chunk at a time from where the last pass stopped, with the guest's address space quiet (no map, unmap or protection
// change meanwhile: paging a page while its protection changes is the bug check 0x1A/0x41202 reproducer). A page the
// game's own views hold stays in RAM; one only the alias held goes to the standby list (or, written, the page file).
uint64_t FlushTrim(uint64_t budget, uint64_t* held_us) {
	static uint64_t cursor = 0;
	if (g_guest_address_space == nullptr) return 0;
	auto* base = reinterpret_cast<uint8_t*>(g_guest_address_space->GetBackingBase());
	if (base == nullptr) return 0;
	constexpr uint64_t Piece = uint64_t {4} << 20u;
	const uint64_t     size  = PhysicalMemory::TotalSize();
	uint64_t           done = 0, walked = 0, held = 0;
	while (done < budget && walked < size) {
		if (cursor >= size) cursor = 0;
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(base + cursor, &info, sizeof(info)) != sizeof(info)) break;
		const uint64_t region_end = static_cast<uint64_t>(static_cast<uint8_t*>(info.BaseAddress) - base) + info.RegionSize;
		const uint64_t stop       = std::min({size, std::max(region_end, cursor + 1), (cursor & ~(Piece - 1u)) + Piece});
		if (info.State == MEM_COMMIT) {
			const auto t0 = std::chrono::steady_clock::now();
			if (!g_guest_address_space->RunQuietWith([&] { (void)VirtualUnlock(base + cursor, stop - cursor); })) break;
			held = std::max(held, Micros(t0));
			done += stop - cursor;
			// (Spread: the game's maps and protection changes get the address space between pieces.)
			std::this_thread::sleep_for(std::chrono::microseconds(200));
		}
		walked += stop - cursor;
		cursor = stop;
	}
	if (held_us != nullptr) *held_us = held;
	return done;
}

} // namespace RamFlushGuest

// (ram1) KYTY_DS_FREED_DISCARD=1 (Windows, Demon's Souls; off by default): the guest store's bytes of the direct memory
// the game's page allocator gives back dropped from RAM. Demon's Souls takes almost all direct memory once at start-up
// (PS5PlatformPageAllocator: AllocateDirect of up to 12 GiB, demonsSoulsAllocClamp.cpp) and maps it 64 KiB at a time;
// a decommit (VirtualAlloc op 2: texture mips dropped by CTexture::SetCurrentStreamingChunk 0x9014f4, heaps and pages
// given back) is UnmapPhysicalPages (0x80eb90): it sets the pages' bits in its free-page bitmap and unmaps them
// (sceKernelBatchMap op 1, or sceKernelReserveVirtualRange over them). Those bytes are no one's from then on: the next
// commit gets whichever free page the bitmap gives, and its new owner writes it before reading (the streamer memcpy's
// the mips in, 0x901459). Kyty kept them: the direct memory stays allocated, so the RAM flush's FlushFreed never sees
// them; the parked view's pages are trimmed to the modified list and written to the page file, the alias's stay
// resident. A native port's allocator decommits them. Here: each unmap of direct memory inside that pool is queued, and
// a thread of its own drops the store's bytes of it 1 MiB at a time (GuestBackingStore::DiscardUncovered: only while no
// guest mapping covers them, checked under the store's lock a map takes exclusive; DiscardVirtualMemory, MEM_RESET when
// refused). The commit charge stays (page-file section). Logged: "Freed discard ..." every 30 s while it drops.
namespace FreedDiscard {

std::atomic<uint64_t> g_pool_begin {0}, g_pool_end {0};
std::atomic<uint64_t> g_dropped {0}, g_covered {0}, g_reserved {0}, g_refused {0}, g_queued {0}, g_calls {0};

// (ram2) KYTY_DS_FREED_DISCARD: 1 = ram1's path (queued, dropped later through the emulator's alias by a thread);
// 2 = reset through the game's own view just before the unmap (GuestBackingStore::ResetSoleMapping), nothing queued,
// no thread, no VirtualQuery. Logs 11-15 04:54/04:57 (ram1, =1): 205-394 MiB "dropped" and nothing gained (available
// 249-856 MiB, the modified list up to 1460 MiB): the alias's pages of a freed range were never in the working set
// ("in RAM not mapped by the game 0-2 MiB"), so the discard there had nothing to act on while the game's view's pages
// had already been trimmed dirty to the modified list.
int RequestedMode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_DS_FREED_DISCARD");
		return v == nullptr ? 0 : v[0] == '2' ? 2 : v[0] == '1' ? 1 : 0;
	}();
	return mode;
}
bool Requested() {
	return RequestedMode() != 0;
}
// The mode in use once the pool is known (0 until then, and when refused).
std::atomic<int>      g_active {0};
std::atomic<uint64_t> g_reset_bytes {0}, g_reset_calls {0}, g_reset_us {0}, g_reset_max_us {0}, g_reset_shared {0},
    g_reset_refused {0};

struct Queue {
	std::mutex                                 mutex;
	std::condition_variable                    ready;
	std::vector<std::pair<uint64_t, uint64_t>> ranges; // physical (backing offset, size)
};
Queue& GetQueue() {
	static auto* queue = new Queue; // (never destroyed: its thread may outlive static destructors)
	return *queue;
}

void Report(bool force) {
	static auto     last       = std::chrono::steady_clock::now();
	static uint64_t last_drops = 0;
	const auto      now        = std::chrono::steady_clock::now();
	if (!force && now - last < std::chrono::seconds(30)) return;
	const uint64_t dropped = g_dropped.load(std::memory_order_relaxed);
	if (!force && dropped == last_drops) return;
	last = now;
	std::printf("Freed discard (KYTY_DS_FREED_DISCARD=1): %llu MiB of freed direct memory dropped from RAM in all (%llu MiB "
	            "since the last line), %llu MiB queued in %llu unmaps; kept: %llu MiB mapped again first, %llu MiB never "
	            "committed, %llu MiB refused by Windows\n",
	            static_cast<unsigned long long>(dropped >> 20u),
	            static_cast<unsigned long long>((dropped - last_drops) >> 20u),
	            static_cast<unsigned long long>(g_queued.load(std::memory_order_relaxed) >> 20u),
	            static_cast<unsigned long long>(g_calls.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(g_covered.load(std::memory_order_relaxed) >> 20u),
	            static_cast<unsigned long long>(g_reserved.load(std::memory_order_relaxed) >> 20u),
	            static_cast<unsigned long long>(g_refused.load(std::memory_order_relaxed) >> 20u));
	std::fflush(stdout);
	last_drops = dropped;
}

void Worker() {
	LocalPlatform::SetThreadName("Kyty.FreedDiscard");
	// (Normal priority: a discard holds the working-set lock over its pages, which the game's page faults wait for; a
	// descheduled low-priority thread must not hold it.)
	auto&                                      queue = GetQueue();
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	for (;;) {
		{
			std::unique_lock lock(queue.mutex);
			queue.ready.wait_for(lock, std::chrono::seconds(30), [&] { return !queue.ranges.empty(); });
		}
		// (A short wait gathers a batch: the streamer unmaps a texture's mips in a row, BatchMap 64 entries at a time.)
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		{
			std::lock_guard lock(queue.mutex);
			ranges.swap(queue.ranges);
		}
		std::sort(ranges.begin(), ranges.end());
		std::vector<std::pair<uint64_t, uint64_t>> merged; // [begin, end)
		for (const auto& [offset, size]: ranges) {
			if (!merged.empty() && offset <= merged.back().second) merged.back().second = std::max(merged.back().second, offset + size);
			else merged.emplace_back(offset, offset + size);
		}
		ranges.clear();
		constexpr uint64_t Chunk = uint64_t {1} << 20u;
		for (const auto& [begin, end]: merged) {
			for (uint64_t at = begin; at < end;) {
				const uint64_t stop    = std::min(end, (at & ~(Chunk - 1u)) + Chunk);
				uint64_t       covered = 0, reserved = 0, refused = 0;
				const uint64_t dropped =
				    g_guest_address_space != nullptr
				        ? g_guest_address_space->DiscardUncoveredBacking(at, stop - at, &covered, &reserved, &refused)
				        : 0;
				g_dropped.fetch_add(dropped, std::memory_order_relaxed);
				g_covered.fetch_add(covered, std::memory_order_relaxed);
				g_reserved.fetch_add(reserved, std::memory_order_relaxed);
				g_refused.fetch_add(refused, std::memory_order_relaxed);
				at = stop;
				SwitchToThread();
			}
		}
		Report(false);
	}
}

// (ram3) KYTY_DS_FREECACHE_RESET=1 (Windows, Demon's Souls 01.007.000; off by default; needs the backing view: not
// KYTY_NATIVE_MEMORY, a backing file or lazy commit). The game's page allocator keeps the blocks it frees MAPPED in a
// free list (ReleasePages 0x80f550; "bloques liberados que siguen mapeados" of the [ram] line: 709-876 MiB after 2.5
// min, log 11-15 06:00) and hands them out again as new memory (first fit, 0x80e7a0): their contents are dead, but
// the emulator sees no unmap, so Windows trims them like live data - to the page file or its compression store (3.6
// GB there in that log). A thread walks the free lists every KYTY_DS_FREECACHE_RESET_MS (default 2000) holding the
// game's own allocator spinlock (DemonsSoulsMemMap::WithPageAllocatorFreeBlocks: no block can be handed out
// meanwhile) and resets, through the game's view, the pool pages of the blocks found free at two passes in a row
// (VirtualAlloc MEM_RESET by GuestBackingStore::ResetSoleMapping, the same call KYTY_DS_FREED_DISCARD=2 makes before
// an unmap: "no longer of interest", dropped instead of written or compressed when trimmed; a native allocator would
// decommit them). The lock is held KYTY_DS_FREECACHE_RESET_US at most a pass (default 1000; the rest at the next
// pass, 250 ms later). A block reused and freed again between two passes is not seen as new (left as it is until it
// changes). Logged every 30 s when it did something: "Free cache reset ...".
namespace FreeCache {

bool Requested() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_DS_FREECACHE_RESET");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

struct Entry {
	uint64_t size = 0, mapped = 0, done = 0; // (done: bytes of it reset; == size: all)
	bool     old = false;                    // (seen at the pass before)
};
struct Work {
	std::unordered_map<uint64_t, Entry> before, now; // by base
	uint64_t budget_us = 1000;
	uint64_t reset = 0, shared = 0, refused = 0, blocks = 0, mapped = 0;
	bool     more = false; // (the budget ended first)
};

// With the game's allocator spinlock held (see above): short, no waits. The store's shared lock inside
// ResetBackingBeforeUnmap is taken exclusive only by a map or unmap, which a game thread makes with this spinlock held
// when it is the allocator's (FlushMemoryRequestQueue 0x80f2c0), or without any allocator lock otherwise.
void OnBlocks(void* user, const Loader::DemonsSoulsMemMap::FreeBlock* blocks, size_t count) {
	auto&          work  = *static_cast<Work*>(user);
	const auto     start = std::chrono::steady_clock::now();
	const uint64_t pool_begin = g_pool_begin.load(std::memory_order_relaxed);
	const uint64_t pool_end   = g_pool_end.load(std::memory_order_acquire);
	constexpr uint64_t Chunk  = uint64_t {4} << 20u;
	work.now.clear();
	work.more = false;
	work.blocks = count;
	work.mapped = 0;
	for (size_t i = 0; i < count; i++) {
		const auto& block = blocks[i];
		work.mapped += block.mapped;
		if (block.mapped == 0) continue; // (nothing of it is mapped)
		Entry entry {block.size, block.mapped, 0, true};
		const auto found = work.before.find(block.base);
		const bool same  = found != work.before.end() && found->second.size == block.size && found->second.mapped == block.mapped;
		if (same) entry.done = found->second.done;
		if (same && found->second.old && pool_end != 0 && g_guest_address_space != nullptr) {
			while (entry.done < entry.size) {
				const auto us = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
				if (us >= work.budget_us) {
					work.more = true;
					break;
				}
				const uint64_t piece = std::min(Chunk, entry.size - entry.done);
				work.reset += g_guest_address_space->ResetBackingBeforeUnmap(block.base + entry.done, piece, pool_begin, pool_end,
				                                                             &work.shared, &work.refused);
				entry.done += piece;
			}
		}
		work.now.emplace(block.base, entry);
	}
	work.before.swap(work.now);
}

void Worker() {
	LocalPlatform::SetThreadName("Kyty.FreeCache");
	// (Normal priority: it holds the game's allocator spinlock while it works; a descheduled low-priority holder would
	// make the game's threads spin.)
	const auto number = [](const char* name, uint64_t fallback, uint64_t lowest, uint64_t highest) {
		const char* v = std::getenv(name);
		const uint64_t n = v != nullptr && v[0] != '\0' ? static_cast<uint64_t>(std::strtoull(v, nullptr, 10)) : fallback;
		return std::min(std::max(n, lowest), highest);
	};
	const uint64_t period_ms = number("KYTY_DS_FREECACHE_RESET_MS", 2000, 250, 60000);
	Work           work;
	work.budget_us = number("KYTY_DS_FREECACHE_RESET_US", 1000, 100, 20000);
	uint64_t total = 0, shared = 0, refused = 0, passes = 0, busy = 0, invalid = 0, hold_us = 0, hold_max_us = 0, said_total = 0;
	uint64_t said_invalid = 0;
	auto     said = std::chrono::steady_clock::now();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(work.more ? 250 : period_ms));
		if (g_pool_end.load(std::memory_order_acquire) == 0) continue;
		work.reset = work.shared = work.refused = 0;
		work.more  = false;
		uint64_t   cached = 0;
		const auto t0     = std::chrono::steady_clock::now();
		const int  result = Loader::DemonsSoulsMemMap::WithPageAllocatorFreeBlocks(&OnBlocks, &work, &cached);
		const auto us     = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
		if (result == 0) {
			passes++;
			total += work.reset;
			shared += work.shared;
			refused += work.refused;
			hold_us += us;
			hold_max_us = std::max(hold_max_us, us);
		} else if (result == 2) {
			busy++;
		} else if (result == 3) {
			invalid++;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - said < std::chrono::seconds(30) || (total == said_total && invalid == said_invalid)) continue;
		said         = now;
		std::printf("Free cache reset (30 s, KYTY_DS_FREECACHE_RESET=1): %llu MiB of the game's free blocks reset in all (%llu "
		            "MiB since the last line); free cache now %llu MiB mapped in %llu blocks; %llu passes (lock busy %llu, "
		            "lists not valid %llu), the game's allocator lock held %.1f ms in all, at most %.2f ms; left: %llu MiB "
		            "mapped twice, %llu MiB refused by Windows\n",
		            static_cast<unsigned long long>(total >> 20u), static_cast<unsigned long long>((total - said_total) >> 20u),
		            static_cast<unsigned long long>(work.mapped >> 20u), static_cast<unsigned long long>(work.blocks),
		            static_cast<unsigned long long>(passes), static_cast<unsigned long long>(busy),
		            static_cast<unsigned long long>(invalid), static_cast<double>(hold_us) / 1000.0,
		            static_cast<double>(hold_max_us) / 1000.0, static_cast<unsigned long long>(shared >> 20u),
		            static_cast<unsigned long long>(refused >> 20u));
		std::fflush(stdout);
		said_total   = total;
		said_invalid = invalid;
		hold_max_us  = 0;
	}
}

} // namespace FreeCache

// (ram6) KYTY_DS_FREED_RELEASE=1 (Windows, Demon's Souls 01.007.000; off by default; the guest store's backing view
// in the page file - all committed or KYTY_BACKING_ON_DEMAND -, not KYTY_NATIVE_MEMORY, KYTY_BACKING_FILE or lazy
// commit, where it counts nothing). What the game's page allocator holds dead, given back to Windows:
//  - its free-block cache (Ghidra 01.007.000, ram6/NOTAS.md): ReleasePages 0x80f550 puts a fully mapped block, still
//    mapped, in its category's free list (sorted by address, merged with its neighbours, no limit: 2192 MiB right
//    after a zone unload, log 11-15 09:19); RequestVirtualPages 0x80d9a0 takes from there first (first fit 0x80e7a0,
//    unless flag 2) with no system call, and unmaps a cached block only when the bitmap has no page left (0x80e15c).
//    A reused block is new memory: its new owner writes it before reading it (pages from the bitmap come with the old
//    bytes of whoever had them, so nothing in the game can count on their contents);
//  - what it unmaps (UnmapPhysicalPages 0x80eb90: mips the streamer drops, partly mapped blocks).
// A thread walks the free lists every KYTY_DS_FREED_RELEASE_MS (default 1000) with the game's allocator spinlock held
// (DemonsSoulsMemMap::WithPageAllocatorFreeBlocks, as KYTY_DS_FREECACHE_RESET). What was free at two passes in a row is
// first made CPU-written for the GPU caches (TryPrepareHostOverwrite, outside the game's lock: no GPU copy of it stays
// "clean" with bytes that are about to change, none is read back into it), then - the lock taken again, the block
// checked still free - given back: GuestBackingStore::ReleaseSoleMapping (DiscardVirtualMemory through the game's
// view and the alias; MEM_RESET + out of the working set when refused). At most KYTY_DS_FREED_RELEASE_US (default
// 2000) of the game's lock a pass; the rest 50 ms later. A released range that comes back into RAM while still free
// (touched, or reused and freed between passes) is released again. KYTY_DS_FREED_RELEASE=2: the same walk, nothing
// invalidated or released - what would be, and how much of it is in RAM. Totals in the [ram] line (KYTY_RAM_LINE=1).
namespace FreedRelease {

int RequestedMode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_DS_FREED_RELEASE");
		return v == nullptr ? 0 : v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
	}();
	return mode;
}
bool Requested() {
	return RequestedMode() != 0;
}

std::atomic<int> g_active {0}; // (the mode in use once the pool is known; 0 off)

// Bytes unless said otherwise.
struct Counters {
	std::atomic<uint64_t> cache {0}, unmap {0};                 // released (or, mode 2, that would be): free blocks, unmaps
	std::atomic<uint64_t> discarded {0}, reset {0};              // (mode 1) by DiscardVirtualMemory / MEM_RESET + unlock
	std::atomic<uint64_t> reused {0}, touched {0};               // taken back by the game / back in RAM while still free
	std::atomic<uint64_t> shared {0}, refused {0}, partial {0};  // not released: mapped twice, Windows, block partly mapped
	std::atomic<uint64_t> now_free {0}, now_released {0};        // at the last pass
	std::atomic<uint64_t> checked {0}, still_game {0}, still_alias {0}; // samples right after release still in RAM
	std::atomic<uint64_t> sampled {0}, in_ram {0};               // (mode 2) the would-release set now: bytes, ~in RAM
	std::atomic<uint64_t> invalidated {0};
	std::atomic<uint64_t> passes {0}, busy {0}, invalid {0}, held_us {0}, held_max_us {0};
	std::atomic<uint64_t> unmap_calls {0}, unmap_us {0}, unmap_max_us {0};
};
Counters& Stat() {
	static Counters counters;
	return counters;
}
void Max(std::atomic<uint64_t>& to, uint64_t v) {
	uint64_t seen = to.load(std::memory_order_relaxed);
	while (v > seen && !to.compare_exchange_weak(seen, v, std::memory_order_relaxed)) {
	}
}

using Ranges = std::vector<std::pair<uint64_t, uint64_t>>; // sorted, disjoint [begin, end)

void Normalize(Ranges* r) {
	std::sort(r->begin(), r->end());
	size_t n = 0;
	for (const auto& x: *r) {
		if (x.first >= x.second) continue;
		if (n != 0 && x.first <= (*r)[n - 1].second) (*r)[n - 1].second = std::max((*r)[n - 1].second, x.second);
		else (*r)[n++] = x;
	}
	r->resize(n);
}
Ranges Intersect(const Ranges& a, const Ranges& b) {
	Ranges out;
	for (size_t i = 0, j = 0; i < a.size() && j < b.size();) {
		const uint64_t lo = std::max(a[i].first, b[j].first);
		const uint64_t hi = std::min(a[i].second, b[j].second);
		if (lo < hi) out.emplace_back(lo, hi);
		if (a[i].second < b[j].second) i++;
		else j++;
	}
	return out;
}
Ranges Subtract(const Ranges& a, const Ranges& b) {
	Ranges out;
	size_t j = 0;
	for (const auto& x: a) {
		uint64_t at = x.first;
		while (j < b.size() && b[j].second <= at) j++;
		for (size_t k = j; k < b.size() && b[k].first < x.second; k++) {
			if (b[k].first > at) out.emplace_back(at, b[k].first);
			at = std::max(at, b[k].second);
			if (at >= x.second) break;
		}
		if (at < x.second) out.emplace_back(at, x.second);
	}
	return out;
}
uint64_t Bytes(const Ranges& r) {
	uint64_t n = 0;
	for (const auto& x: r) n += x.second - x.first;
	return n;
}

constexpr size_t   MaxBlocks = 8192; // (WithPageAllocatorFreeBlocks gives at most this many)
constexpr size_t   MaxDone   = 4096;
constexpr uint64_t Chunk     = uint64_t {4} << 20u;

// (ram14) KYTY_DS_FREED_RELEASE with native guest memory (KYTY_NATIVE_MEMORY, which KYTY_RAIZ=1 uses). There the
// guest address IS the memory: private, committed when mapped, no section, no alias (the start-up self-test refuses
// one), and a guest unmap releases it (VirtualFree MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER): what the game unmaps is
// already given back. What it keeps dead is only its free-block cache (Ghidra, eboot 01.007.000, projelf_un3, ram14):
//  - ReleasePages 0x80f550: a fully mapped block goes, still mapped, into its category's free list (state + c*0xa0,
//    +0x28; nodes are PageAllocationInfo of 0x78 bytes from 0x80d0a0, never inside the memory they describe); only a
//    partly mapped one is unmapped (UnmapPhysicalPages 0x80eb90, flag 1);
//  - 0x80e7a0 (first fit, called by RequestVirtualPages 0x80d9a0 unless flag 2): splits a cached block and hands it
//    out with no system call, no zeroing: its new owner writes before reading (nothing can count on the old bytes).
// Logs 11-16 03:47/04:40/05:26 (16 GB, raiz): "bloques liberados que siguen mapeados" 423-545 MiB all session long;
// with private memory Windows can only keep them (RAM or its compression store: 4-5 GB there). Here each block free at
// two passes in a row is reset at its guest address (VirtualAlloc MEM_RESET: no write to the pages, whatever their
// protection; the pages stay committed and mapped; their RAM and compressed copies are dropped instead of kept) and
// trimmed (VirtualUnlock). Only committed private regions (VirtualQuery) are touched; the commit charge stays.
std::atomic<bool> g_native {false};

// (commit-real, KYTY_COMMIT_REAL=1 by default) With commit-at-touch on, a free block is decommitted instead
// (NativeLazyCommit::DecommitFree: MEM_DECOMMIT, the commit charge given back too, not only the RAM; the next touch
// commits it again, zero-filled - a reused block is written by its new owner before it is read, Ghidra above). Pages
// whose protection is not the mapping's (watched or taken out by the GPU caches or an offload) keep MEM_RESET below.
uint64_t NativeRelease(uint64_t vaddr, uint64_t size, bool act, GuestBackingStore::ReleaseCounts* counts) {
	if (size == 0 || UINT64_MAX - vaddr < size) return 0;
	const uint64_t end  = vaddr + size;
	uint64_t       done = 0;
	if (act) {
		uint64_t       rest        = 0;
		const uint64_t decommitted = NativeLazyCommit::DecommitFree(vaddr, size, &rest);
		counts->reset += decommitted;
		done += decommitted;
	}
	for (uint64_t at = vaddr; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
		const uint64_t region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize);
		const uint64_t stop       = std::min(end, region_end);
		if (stop <= at) break;
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE) {
			const uint64_t bytes = stop - at;
			if (!act) {
				done += bytes;
			} else if (VirtualAlloc(reinterpret_cast<void*>(at), bytes, MEM_RESET, PAGE_READWRITE) != nullptr) {
				counts->reset += bytes;
				done += bytes;
			} else {
				counts->refused += bytes;
				counts->Skip(at, stop);
			}
		}
		at = stop;
	}
	counts->done += done;
	return done;
}

// Out of the working set (nothing written: a reset page leaves it without a page-file or compression copy).
void NativeTrim(uint64_t vaddr, uint64_t size) {
	const uint64_t end = vaddr + size;
	for (uint64_t at = vaddr; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
		const uint64_t stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
		if (stop <= at) break;
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE) (void)VirtualUnlock(reinterpret_cast<void*>(at), stop - at);
		at = stop;
	}
}

// Bit 0 of flags[i]: the page at addresses[i] is in the working set (K32QueryWorkingSetEx, 1024 a call).
bool NativeSample(const uint64_t* addresses, size_t count, uint8_t* flags) {
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> query(std::min<size_t>(count, 1024));
	for (size_t begin = 0; begin < count; begin += query.size()) {
		const size_t n = std::min(query.size(), count - begin);
		for (size_t i = 0; i < n; i++) {
			query[i]                = {};
			query[i].VirtualAddress = reinterpret_cast<PVOID>(addresses[begin + i] & ~uint64_t {0xfff});
		}
		if (K32QueryWorkingSetEx(GetCurrentProcess(), query.data(), static_cast<DWORD>(n * sizeof(query[0]))) == 0) return false;
		for (size_t i = 0; i < n; i++) flags[begin + i] = query[i].VirtualAttributes.Valid != 0 ? 1u : 0u;
	}
	return true;
}

uint64_t ReleaseRange(uint64_t vaddr, uint64_t size, uint64_t pool_begin, uint64_t pool_end, bool act,
                      GuestBackingStore::ReleaseCounts* counts) {
	if (g_native.load(std::memory_order_relaxed)) return NativeRelease(vaddr, size, act, counts);
	return g_guest_address_space->ReleaseBackingSole(vaddr, size, pool_begin, pool_end, act, counts);
}
void TrimRange(uint64_t vaddr, uint64_t size, uint64_t pool_begin, uint64_t pool_end) {
	if (g_native.load(std::memory_order_relaxed)) NativeTrim(vaddr, size);
	else g_guest_address_space->TrimBackingSole(vaddr, size, pool_begin, pool_end);
}
bool SampleResidency(const uint64_t* addresses, size_t count, uint8_t* flags) {
	if (g_native.load(std::memory_order_relaxed)) return NativeSample(addresses, count, flags);
	return g_guest_address_space->SampleBackingResidency(addresses, count, flags);
}

// Filled with the game's allocator lock held: no allocation there (capacities reserved before).
struct Pass {
	std::vector<Loader::DemonsSoulsMemMap::FreeBlock> blocks; // the fully mapped free blocks, by base
	uint64_t                                          partial = 0;
	const Ranges*                                     todo    = nullptr;
	std::pair<uint64_t, uint64_t>                     done[MaxDone];
	size_t                                            done_count = 0;
	std::pair<uint64_t, uint64_t>                     skipped[MaxDone]; // (counts.skipped: mapped twice, refused)
	uint64_t                                          budget_us = 2000, pool_begin = 0, pool_end = 0;
	GuestBackingStore::ReleaseCounts                  counts;
	bool                                              more = false;
};

void TakeBlocks(Pass& p, const Loader::DemonsSoulsMemMap::FreeBlock* blocks, size_t count) {
	p.blocks.clear();
	p.partial = 0;
	for (size_t i = 0; i < count && p.blocks.size() < MaxBlocks; i++) {
		if (blocks[i].mapped == blocks[i].size) p.blocks.push_back(blocks[i]);
		else p.partial += blocks[i].size; // (a partly mapped block is on its way out: UnmapPhysicalPages)
	}
	std::sort(p.blocks.begin(), p.blocks.end(), [](const auto& a, const auto& b) { return a.base < b.base; });
}

void OnCollect(void* user, const Loader::DemonsSoulsMemMap::FreeBlock* blocks, size_t count) {
	TakeBlocks(*static_cast<Pass*>(user), blocks, count);
}

void OnRelease(void* user, const Loader::DemonsSoulsMemMap::FreeBlock* blocks, size_t count) {
	auto& p = *static_cast<Pass*>(user);
	const auto start = std::chrono::steady_clock::now();
	TakeBlocks(p, blocks, count);
	p.done_count = 0;
	p.more       = false;
	for (const auto& [b, e]: *p.todo) {
		// (Only if all of it is still in one fully mapped free block: taken or changed since, it waits for a pass.)
		auto it = std::upper_bound(p.blocks.begin(), p.blocks.end(), b, [](uint64_t v, const auto& f) { return v < f.base; });
		if (it == p.blocks.begin()) continue;
		--it;
		if (b < it->base || e > it->base + it->size) continue;
		for (uint64_t at = b; at < e;) {
			const auto us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			if (us >= p.budget_us || p.done_count == MaxDone) {
				p.more = true;
				return;
			}
			const uint64_t stop = std::min(e, at + Chunk);
			(void)ReleaseRange(at, stop - at, p.pool_begin, p.pool_end, true, &p.counts);
			p.done[p.done_count++] = {at, stop};
			at = stop;
		}
	}
}

// One sample per `stride` bytes of `set` (at most `max`): the addresses, and the stride used.
uint64_t SampleAddresses(const Ranges& set, size_t max, std::vector<uint64_t>* out) {
	out->clear();
	const uint64_t bytes  = Bytes(set);
	uint64_t       stride = 0x10000;
	while (bytes / stride > max) stride <<= 1u;
	for (const auto& [b, e]: set)
		for (uint64_t at = b; at < e && out->size() < max; at += stride) out->push_back(at);
	return stride;
}

void Worker() {
	LocalPlatform::SetThreadName("Kyty.FreedRelease");
	// (Normal priority: it holds the game's allocator spinlock while it works.)
	const auto number = [](const char* name, uint64_t fallback, uint64_t lowest, uint64_t highest) {
		const char*    v = std::getenv(name);
		const uint64_t n = v != nullptr && v[0] != '\0' ? static_cast<uint64_t>(std::strtoull(v, nullptr, 10)) : fallback;
		return std::min(std::max(n, lowest), highest);
	};
	const int      mode       = RequestedMode();
	const uint64_t period_ms  = number("KYTY_DS_FREED_RELEASE_MS", 1000, 200, 60000);
	const uint64_t budget_us  = number("KYTY_DS_FREED_RELEASE_US", 2000, 100, 20000);
	const bool     invalidate = [] {
		const char* v = std::getenv("KYTY_DS_FREED_RELEASE_INVALIDATE");
		return v == nullptr || v[0] != '0';
	}();
	// (ram14) KYTY_DS_FREED_RELEASE_QUIET=0: the resets and the trim without holding every guest map, unmap and
	// protection change off (ram9's caution, kept by default; with native memory there is no shared view to race).
	const bool quiet = [] {
		const char* v = std::getenv("KYTY_DS_FREED_RELEASE_QUIET");
		return v == nullptr || v[0] != '0';
	}();
	auto said = std::chrono::steady_clock::now();
	auto&                 st   = Stat();
	auto*                 pass = new Pass; // (never destroyed: the thread may outlive static destructors)
	pass->blocks.reserve(MaxBlocks);
	pass->budget_us = budget_us;
	Ranges   previous, released, invalidated_set;
	Ranges   stuck; // (free ranges left as they were - mapped twice, refused -: not released again and again)
	bool     more     = false;
	auto     scanned  = std::chrono::steady_clock::now();
	std::vector<uint64_t> addresses;
	std::vector<uint8_t>  flags;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(more ? 50 : period_ms));
		more = false;
		if (g_active.load(std::memory_order_acquire) == 0 || g_pool_end.load(std::memory_order_acquire) == 0) continue;
		pass->pool_begin = g_pool_begin.load(std::memory_order_relaxed);
		pass->pool_end   = g_pool_end.load(std::memory_order_acquire);
		// 1. What is free now.
		uint64_t   cached = 0;
		auto       t0     = std::chrono::steady_clock::now();
		const int  result = Loader::DemonsSoulsMemMap::WithPageAllocatorFreeBlocks(&OnCollect, pass, &cached);
		uint64_t   us     = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
		if (result != 0) {
			(result == 2 ? st.busy : st.invalid).fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		st.passes.fetch_add(1, std::memory_order_relaxed);
		st.held_us.fetch_add(us, std::memory_order_relaxed);
		Max(st.held_max_us, us);
		st.partial.store(pass->partial, std::memory_order_relaxed);
		Ranges now;
		now.reserve(pass->blocks.size());
		for (const auto& f: pass->blocks) now.emplace_back(f.base, f.base + f.size);
		Normalize(&now);
		// 2. Released bytes no longer free: the game took them back (or unmapped them).
		const Ranges kept = Intersect(released, now);
		st.reused.fetch_add(Bytes(released) - Bytes(kept), std::memory_order_relaxed);
		released        = kept;
		invalidated_set = Intersect(invalidated_set, now);
		stuck           = Intersect(stuck, now);
		// 3. Free at this pass and the one before, not released yet.
		Ranges todo = Subtract(Intersect(previous, now), released);
		previous    = now;
		st.now_free.store(Bytes(now), std::memory_order_relaxed);
		if (mode == 2) {
			// (Counted through the store as mode 1 would release it - the pool's, mapped once -; nothing changed.)
			GuestBackingStore::ReleaseCounts counts;
			counts.skipped     = pass->skipped;
			counts.skipped_max = MaxDone;
			if (g_guest_address_space != nullptr)
				for (const auto& [b, e]: todo)
					(void)ReleaseRange(b, e - b, pass->pool_begin, pass->pool_end, false, &counts);
			st.cache.fetch_add(counts.done, std::memory_order_relaxed);
			st.shared.fetch_add(counts.shared, std::memory_order_relaxed);
			for (size_t i = 0; i < counts.skipped_count; i++) stuck.push_back(pass->skipped[i]);
			Normalize(&stuck);
			for (const auto& r: todo) released.push_back(r);
			Normalize(&released);
		} else if (!todo.empty() && g_guest_address_space != nullptr) {
			// 4. The GPU caches told first (outside the game's lock: they take their own locks).
			if (invalidate) {
				for (const auto& [b, e]: Subtract(todo, invalidated_set)) {
					if (TryPrepareHostOverwrite(b, e - b)) st.invalidated.fetch_add(e - b, std::memory_order_relaxed);
					invalidated_set.emplace_back(b, e);
				}
				Normalize(&invalidated_set);
			}
			// 5. Given back, each range checked still free with the lock held again.
			pass->todo               = &todo;
			pass->counts             = {};
			pass->counts.skipped     = pass->skipped;
			pass->counts.skipped_max = MaxDone;
			t0           = std::chrono::steady_clock::now();
			// (ram9) With the guest memory quiet (no guest map, unmap or protection change meanwhile, as every working
			// set trim of the emulator): its MEM_RESETs ran beside the page manager's ~34,000 protection changes a
			// second (log 22:40, 16 GB, Windows restarted at ~10.5 min; bug check 0x1A seen before with a working-set
			// change beside a protection change). The game's lock is only tried (64 attempts): no wait on it here.
			int again = 2;
			if (!quiet) again = Loader::DemonsSoulsMemMap::WithPageAllocatorFreeBlocks(&OnRelease, pass, &cached);
			else if (!g_guest_address_space->RunQuietWith(
			             [&] { again = Loader::DemonsSoulsMemMap::WithPageAllocatorFreeBlocks(&OnRelease, pass, &cached); }))
				again = 2;
			us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
			if (again == 0) {
				st.held_us.fetch_add(us, std::memory_order_relaxed);
				Max(st.held_max_us, us);
				st.cache.fetch_add(pass->counts.done, std::memory_order_relaxed);
				st.discarded.fetch_add(pass->counts.discarded, std::memory_order_relaxed);
				st.reset.fetch_add(pass->counts.reset, std::memory_order_relaxed);
				st.shared.fetch_add(pass->counts.shared, std::memory_order_relaxed);
				st.refused.fetch_add(pass->counts.refused, std::memory_order_relaxed);
				Ranges done(pass->done, pass->done + pass->done_count);
				for (const auto& r: done) released.push_back(r);
				Normalize(&released);
				for (size_t i = 0; i < pass->counts.skipped_count; i++) stuck.push_back(pass->skipped[i]);
				Normalize(&stuck);
				more = pass->more;
				// (ram8) Out of the working set now (reset pages are dropped without a page-file write when they leave
				// it): out of the game's lock, with every guest map, unmap and protection change held off, as the RAM
				// flush's trim. A block the game took back meanwhile is only trimmed: nothing is written.
				if (!done.empty()) {
					const uint64_t begin_pool = pass->pool_begin, end_pool = pass->pool_end; // (ram9: quiet again, apart)
					if (quiet)
						(void)g_guest_address_space->RunQuietWith([&] {
							for (const auto& [b, e]: done) TrimRange(b, e - b, begin_pool, end_pool);
						});
					else
						for (const auto& [b, e]: done) TrimRange(b, e - b, begin_pool, end_pool);
				}
				// 6. Did it leave RAM? (A sample of what was just released: 512 pages at most, both views.)
				Normalize(&done);
				done = Subtract(done, stuck);
				if (!done.empty()) {
					(void)SampleAddresses(done, 512, &addresses);
					flags.resize(addresses.size());
					if (SampleResidency(addresses.data(), addresses.size(), flags.data())) {
						uint64_t game = 0, alias = 0;
						for (const uint8_t f: flags) {
							game += (f & 1u) != 0 ? 1 : 0;
							alias += (f & 2u) != 0 ? 1 : 0;
						}
						st.checked.fetch_add(addresses.size(), std::memory_order_relaxed);
						st.still_game.fetch_add(game, std::memory_order_relaxed);
						st.still_alias.fetch_add(alias, std::memory_order_relaxed);
					}
				}
			} else {
				(again == 2 ? st.busy : st.invalid).fetch_add(1, std::memory_order_relaxed);
				more = true;
			}
		}
		// 7. Every 5 s: what of the released (or would-be) set is in RAM. Mode 1: a page back in RAM while still free was
		// touched (or reused and freed between passes): released again at the next passes.
		const Ranges given = Subtract(released, stuck);
		if (std::chrono::steady_clock::now() - scanned >= std::chrono::seconds(5) && !given.empty() &&
		    g_guest_address_space != nullptr) {
			scanned             = std::chrono::steady_clock::now();
			const uint64_t step = SampleAddresses(given, 32768, &addresses);
			flags.resize(addresses.size());
			if (SampleResidency(addresses.data(), addresses.size(), flags.data())) {
				Ranges   back;
				uint64_t in_ram = 0;
				for (size_t i = 0; i < addresses.size(); i++) {
					if (flags[i] == 0) continue;
					in_ram += step;
					back.emplace_back(addresses[i], addresses[i] + step);
				}
				if (mode == 2) {
					st.sampled.store(Bytes(given), std::memory_order_relaxed);
					st.in_ram.store(in_ram, std::memory_order_relaxed);
				} else if (!back.empty()) {
					Normalize(&back);
					back = Intersect(back, given);
					st.touched.fetch_add(Bytes(back), std::memory_order_relaxed);
					released        = Subtract(released, back);
					invalidated_set = Subtract(invalidated_set, back);
				}
			}
		}
		st.now_released.store(Bytes(Subtract(released, stuck)), std::memory_order_relaxed);
		// (ram14) Every 30 s, its own line (with or without KYTY_RAM_LINE): what is given back now, in GB.
		if (std::chrono::steady_clock::now() - said >= std::chrono::seconds(30)) {
			said              = std::chrono::steady_clock::now();
			constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
			const auto get = [](const std::atomic<uint64_t>& a) { return static_cast<double>(a.load(std::memory_order_relaxed)); };
			const uint64_t   checked = st.checked.load(std::memory_order_relaxed);
			std::printf("Freed release (30 s, KYTY_DS_FREED_RELEASE=%d%s): RAM ahorrada ahora %.2f GB (bloques libres del juego "
			            "devueltos a Windows, sin reutilizar ni volver a RAM; libres ahora %.2f GB); devuelto en total %.2f GB, "
			            "reutilizado por el juego %.2f GB, vuelto a RAM estando libre %.2f GB, rechazado %.2f GB; tras devolver "
			            "seguian en RAM %llu de %llu muestras; candado del juego %llu pasadas (ocupado %llu), la mas larga "
			            "%.2f ms; memoria quieta %s\n",
			            mode, g_native.load(std::memory_order_relaxed) ? ", memoria nativa" : "",
			            mode == 1 ? get(st.now_released) / GiB : 0.0, get(st.now_free) / GiB, get(st.cache) / GiB,
			            get(st.reused) / GiB, get(st.touched) / GiB, get(st.refused) / GiB,
			            static_cast<unsigned long long>(st.still_game.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(checked), static_cast<unsigned long long>(get(st.passes)),
			            static_cast<unsigned long long>(get(st.busy)), get(st.held_max_us) / 1000.0, quiet ? "si" : "no");
			std::fflush(stdout);
		}
	}
}

// (nativa2) KYTY_RAM_NATIVA2=1 (off by default; =2 measures only): the free chunks of the game's CPU HeapAllocator
// ("cpuheap", 1 GiB; Ghidra 01.007.000: Free 0x802f90, size tree at heap+0x88, lock at [heap+0x80]; see
// DemonsSoulsMemMap::WithCpuHeapFreeChunks) keep their pages in RAM: the heap never gives a chunk back while its
// segment holds anything else, and the loads of the title screen / Nexus free their temporary buffers there. Log
// 11-24 09:34: cpuheap 0.85 of 1.00 GB in RAM, all of it without a GPU copy. Here the interior of each chunk free at two
// passes in a row (from chunk+0x40 to 0x40 before its end, whole pages: the heap's header, tree node and the next
// chunk's header are never touched) is reset (MEM_RESET; DiscardVirtualMemory through the view without native memory)
// and trimmed, with the heap's lock held and the chunk checked still free; the GPU caches told first. A reused chunk is
// new memory for its new owner (malloc: nothing reads it before writing it). A "[ram-nativa2]" line every 10 s.
int Nativa2Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_RAM_NATIVA2");
		if (v == nullptr) { // (RAM-NATIVA) KYTY_RAM_NATIVA tambien mide el monton CPU (solo medir)
			const char* t = std::getenv("KYTY_RAM_NATIVA");
			return t != nullptr && (t[0] == '1' || t[0] == '2') ? 2 : 0;
		}
		return v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
	}();
	return mode;
}

Counters& HeapStat() {
	static Counters counters;
	return counters;
}

std::atomic<uint64_t> g_heap_chunks {0};

void HeapWorker() {
	LocalPlatform::SetThreadName("Kyty.RamNativa2");
	const int mode = Nativa2Mode();
	auto&     st   = HeapStat();
	auto*     pass = new Pass; // (never destroyed: the thread may outlive static destructors)
	pass->blocks.reserve(MaxBlocks);
	pass->budget_us = 2000;
	Ranges                previous, released, invalidated_set, stuck;
	bool                  more    = false;
	auto                  scanned = std::chrono::steady_clock::now();
	auto                  said    = scanned;
	std::vector<uint64_t> addresses;
	std::vector<uint8_t>  flags;
	uint64_t              in_ram_now = 0;
	constexpr double      GiB        = 1024.0 * 1024.0 * 1024.0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(more ? 50 : 1000));
		more = false;
		if (g_pool_end.load(std::memory_order_acquire) == 0 || g_guest_address_space == nullptr) continue;
		pass->pool_begin = g_pool_begin.load(std::memory_order_relaxed);
		pass->pool_end   = g_pool_end.load(std::memory_order_acquire);
		// 1. What is free now (the heap's lock held while it is read).
		uint64_t  free_bytes = 0, chunks = 0;
		auto      t0         = std::chrono::steady_clock::now();
		const int result     = Loader::DemonsSoulsMemMap::WithCpuHeapFreeChunks(&OnCollect, pass, &free_bytes, &chunks);
		uint64_t  us         = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
		if (result != 0) {
			(result == 2 ? st.busy : st.invalid).fetch_add(1, std::memory_order_relaxed);
		} else {
			st.passes.fetch_add(1, std::memory_order_relaxed);
			st.held_us.fetch_add(us, std::memory_order_relaxed);
			Max(st.held_max_us, us);
			st.now_free.store(free_bytes, std::memory_order_relaxed);
			g_heap_chunks.store(chunks, std::memory_order_relaxed);
			Ranges now;
			now.reserve(pass->blocks.size());
			for (const auto& f: pass->blocks) now.emplace_back(f.base, f.base + f.size);
			Normalize(&now);
			st.sampled.store(Bytes(now), std::memory_order_relaxed);
			// 2. Released bytes no longer free: the game took them back.
			const Ranges kept = Intersect(released, now);
			st.reused.fetch_add(Bytes(released) - Bytes(kept), std::memory_order_relaxed);
			released        = kept;
			invalidated_set = Intersect(invalidated_set, now);
			stuck           = Intersect(stuck, now);
			// 3. Free at this pass and the one before, not released yet.
			const Ranges todo = Subtract(Subtract(Intersect(previous, now), released), stuck);
			previous          = now;
			if (mode == 1 && !todo.empty()) {
				// 4. The GPU caches told first (outside the heap's lock: they take their own locks).
				for (const auto& [b, e]: Subtract(todo, invalidated_set)) {
					if (TryPrepareHostOverwrite(b, e - b)) st.invalidated.fetch_add(e - b, std::memory_order_relaxed);
					invalidated_set.emplace_back(b, e);
				}
				Normalize(&invalidated_set);
				// 5. Given back, each range checked still inside a free chunk with the lock held again, the guest memory
				// quiet (no map, unmap or protection change meanwhile, as KYTY_DS_FREED_RELEASE).
				pass->todo               = &todo;
				pass->counts             = {};
				pass->counts.skipped     = pass->skipped;
				pass->counts.skipped_max = MaxDone;
				int again                = 2;
				t0                       = std::chrono::steady_clock::now();
				if (!g_guest_address_space->RunQuietWith([&] {
					    again = Loader::DemonsSoulsMemMap::WithCpuHeapFreeChunks(&OnRelease, pass, nullptr, nullptr);
				    }))
					again = 2;
				us = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
				if (again == 0) {
					st.held_us.fetch_add(us, std::memory_order_relaxed);
					Max(st.held_max_us, us);
					st.cache.fetch_add(pass->counts.done, std::memory_order_relaxed);
					st.discarded.fetch_add(pass->counts.discarded, std::memory_order_relaxed);
					st.reset.fetch_add(pass->counts.reset, std::memory_order_relaxed);
					st.shared.fetch_add(pass->counts.shared, std::memory_order_relaxed);
					st.refused.fetch_add(pass->counts.refused, std::memory_order_relaxed);
					Ranges done(pass->done, pass->done + pass->done_count);
					for (const auto& r: done) released.push_back(r);
					Normalize(&released);
					for (size_t i = 0; i < pass->counts.skipped_count; i++) stuck.push_back(pass->skipped[i]);
					Normalize(&stuck);
					more = pass->more;
					if (!done.empty()) {
						const uint64_t begin_pool = pass->pool_begin, end_pool = pass->pool_end;
						(void)g_guest_address_space->RunQuietWith([&] {
							for (const auto& [b, e]: done) TrimRange(b, e - b, begin_pool, end_pool);
						});
					}
				} else {
					(again == 2 ? st.busy : st.invalid).fetch_add(1, std::memory_order_relaxed);
					more = true;
				}
			}
			// 6. Every 5 s: released pages back in RAM while still free (touched) are released again later; and what
			// of the free interior is in RAM now (the measurement).
			if (std::chrono::steady_clock::now() - scanned >= std::chrono::seconds(5)) {
				scanned = std::chrono::steady_clock::now();
				const Ranges given = Subtract(released, stuck);
				if (!given.empty()) {
					const uint64_t step = SampleAddresses(given, 16384, &addresses);
					flags.resize(addresses.size());
					if (SampleResidency(addresses.data(), addresses.size(), flags.data())) {
						Ranges back;
						for (size_t i = 0; i < addresses.size(); i++)
							if (flags[i] != 0) back.emplace_back(addresses[i], addresses[i] + step);
						if (!back.empty()) {
							Normalize(&back);
							back = Intersect(back, given);
							st.touched.fetch_add(Bytes(back), std::memory_order_relaxed);
							released        = Subtract(released, back);
							invalidated_set = Subtract(invalidated_set, back);
						}
					}
				}
				const Ranges kept_free = Subtract(now, Subtract(released, stuck));
				in_ram_now             = 0;
				if (!kept_free.empty()) {
					const uint64_t step = SampleAddresses(kept_free, 16384, &addresses);
					flags.resize(addresses.size());
					if (SampleResidency(addresses.data(), addresses.size(), flags.data()))
						for (const uint8_t f: flags) in_ram_now += f != 0 ? step : 0;
				}
				st.in_ram.store(std::min(in_ram_now, Bytes(kept_free)), std::memory_order_relaxed);
			}
			st.now_released.store(Bytes(Subtract(released, stuck)), std::memory_order_relaxed);
		}
		// 7. Every 10 s: its line, GB.
		if (std::chrono::steady_clock::now() - said >= std::chrono::seconds(10)) {
			said           = std::chrono::steady_clock::now();
			const auto get = [](const std::atomic<uint64_t>& a) { return static_cast<double>(a.load(std::memory_order_relaxed)); };
			std::printf("[ram-nativa2] 10 s, GB (KYTY_RAM_NATIVA2=%d%s): monton CPU del juego (cpuheap): libre %.2f en %llu trozos, "
			            "interior devolvible %.2f, de lo no devuelto sigue en RAM %.2f; RAM ahorrada ahora %.2f (devuelto y sin "
			            "reutilizar); devuelto en total %.2f [MEM_RESET %.2f, Discard %.2f], reutilizado por el juego %.2f, "
			            "vuelto a RAM estando libre %.2f, rechazado %.2f; avisado a la GPU %.2f; candado del monton %llu pasadas "
			            "(ocupado %llu, arbol no valido %llu), la mas larga %.2f ms\n",
			            mode, mode == 2 ? ", solo mide" : "", get(st.now_free) / GiB,
			            static_cast<unsigned long long>(g_heap_chunks.load(std::memory_order_relaxed)), get(st.sampled) / GiB,
			            get(st.in_ram) / GiB, mode == 1 ? get(st.now_released) / GiB : 0.0, get(st.cache) / GiB,
			            get(st.reset) / GiB, get(st.discarded) / GiB, get(st.reused) / GiB, get(st.touched) / GiB,
			            get(st.refused) / GiB, get(st.invalidated) / GiB,
			            static_cast<unsigned long long>(get(st.passes)), static_cast<unsigned long long>(get(st.busy)),
			            static_cast<unsigned long long>(get(st.invalid)),
			            static_cast<double>(st.held_max_us.exchange(0, std::memory_order_relaxed)) / 1000.0);
			std::fflush(stdout);
		}
	}
}

void StartNativa2() {
	static std::atomic<bool> started {false};
	if (Nativa2Mode() == 0 || started.exchange(true)) return;
	std::thread(HeapWorker).detach();
	std::printf("[ram-nativa2] ON (KYTY_RAM_NATIVA2=%d): trozos libres del monton CPU del juego (Free 0x802f90, arbol heap+0x88, "
	            "candado [heap+0x80]) %s; linea [ram-nativa2] cada 10 s\n",
	            Nativa2Mode(), Nativa2Mode() == 1 ? "devueltos a Windows por dentro (MEM_RESET, sin tocar cabecera ni final)"
	                                              : "solo medidos");
	std::fflush(stdout);
}


// (RAM-NATIVA) KYTY_RAM_NATIVA=1 (apagado por defecto; =2 solo mide): los bloques libres de los TaggedFrameAllocator
// (arenas "DefaultTagg"). Ghidra 01.007.000 (projelf_par1, VA del juego):
//  - AllocTag 0x7fe350 (vtable+0xf8 del padre): candado *[padre+0x80] (0 -> -1, cmpxchg); bloques de [padre+0x88]
//    (1 MiB); trozos en la lista [padre+0x518] -> monton+0x10; [padre+0x90] += bloques dados y
//    [padre+0xa0] = max([padre+0xa0], [padre+0x90]) (el maximo historico del propio juego).
//  - Trozo nuevo 0x7fefc0: 0x801d00(padre, [padre+0x88]*0x78 = 120 MiB, 0x10000) y [padre+0x98] += 0x78 (reservado).
//  - FreeTag 0x7fe540 (vtable+0x100): [padre+0x90] -= bloques; mascara de uso del trozo [c+0x18] (128 bits, 120
//    validos) &= ~mascara; un trozo vacio que no es el primero (padre+0xe8) se devuelve: 0x801b80 / 0x801a10 ->
//    ReleasePages 0x80f550 (cache de bloques libres del PageAllocator, ya cubierta por KYTY_DS_FREED_RELEASE).
// Aqui, cada 1 s, con el candado del juego: los bloques con bit 0 en [c+0x18] (libres). Libres dos pasadas seguidas:
// la GPU avisada (TryPrepareHostOverwrite, fuera del candado) y, con el candado otra vez y el bit comprobado todavia
// a 0, MEM_RESET por la vista del juego (ReleaseRange: commit intacto, sin fallos de pagina, sin desmapear: un VkBuffer
// importado y la cache de buferes siguen validos). Cuando AllocTag vuelve a dar el bloque, el juego lo escribe antes
// de leerlo (memoria de fotograma) y Windows lo trae de nuevo a RAM al tocarlo. Linea [ram-nativa] cada 10 s, MiB
// exactos por padre (una consulta de working set por pagina).
// KYTY_RAM_NATIVA activo tambien enciende la medida del monton CPU ([ram-nativa2] =2, solo mide) si no esta puesta.
int TagMode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_RAM_NATIVA");
		return v == nullptr ? 0 : v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
	}();
	return mode;
}

namespace TagNativa {
constexpr int      MaxParents = 8;
constexpr uint64_t AllocTagVa = 0x7fe350, FreeTagVa = 0x7fe540;

uint64_t Rd(uint64_t a) {
	return *reinterpret_cast<const volatile uint64_t*>(a);
}
// Candado del juego *[padre+0x80] (como AllocTag 0x7fe350); false si sigue ocupado.
bool Lock(uint64_t parent) {
	auto* lock = reinterpret_cast<std::atomic<int64_t>*>(Rd(parent + 0x80));
	if (lock == nullptr) return true;
	for (int i = 0; i < 20000; i++) {
		int64_t z = 0;
		if (lock->compare_exchange_weak(z, -1, std::memory_order_acquire)) return true;
		_mm_pause();
	}
	return false;
}
void Unlock(uint64_t parent) {
	auto* lock = reinterpret_cast<std::atomic<int64_t>*>(Rd(parent + 0x80));
	if (lock != nullptr) lock->store(0, std::memory_order_release);
}

struct Snap {
	uint64_t block = 0, used = 0, reserved = 0, peak = 0;
	Ranges   all, free; // trozos enteros / bloques libres (sin reservar memoria con el candado: capacidad previa)
};
// Con el candado tomado. false si la estructura no es valida.
bool Read(uint64_t parent, Snap* s) {
	s->all.clear();
	s->free.clear();
	s->block    = Rd(parent + 0x88);
	s->used     = Rd(parent + 0x90);
	s->reserved = Rd(parent + 0x98);
	s->peak     = Rd(parent + 0xa0);
	if (s->block == 0 || s->block > (uint64_t {64} << 20u) || (s->block & 0xfff) != 0) return false;
	const uint64_t heads = parent + 0x518;
	int            guard = 0;
	for (uint64_t h = Rd(heads); h != heads && h != 0 && guard < 4096; h = Rd(h), ++guard) {
		const uint64_t chunks = h + 0x10;
		for (uint64_t c = Rd(chunks); c != chunks && c != 0 && guard < 4096; c = Rd(c), ++guard) {
			const uint64_t base = Rd(c + 0x10);
			if (base == 0 || s->all.size() == s->all.capacity()) continue;
			const uint64_t m[2] = {Rd(c + 0x18), Rd(c + 0x20) & ((uint64_t {1} << 0x38) - 1)};
			s->all.emplace_back(base, base + 120 * s->block);
			for (uint64_t i = 0; i < 120;) {
				if (((m[i >> 6] >> (i & 63)) & 1u) != 0) {
					++i;
					continue;
				}
				uint64_t j = i;
				while (j < 120 && ((m[j >> 6] >> (j & 63)) & 1u) == 0) ++j;
				if (s->free.size() < s->free.capacity()) s->free.emplace_back(base + i * s->block, base + j * s->block);
				i = j;
			}
		}
	}
	return guard < 4096;
}

// Bytes en RAM de un conjunto (una muestra por pagina de 4 KiB: exacto).
uint64_t InRam(const Ranges& set, std::vector<uint64_t>* addresses, std::vector<uint8_t>* flags) {
	addresses->clear();
	for (const auto& [b, e]: set)
		for (uint64_t a = b; a < e; a += 0x1000) addresses->push_back(a);
	if (addresses->empty()) return 0;
	flags->resize(addresses->size());
	if (!SampleResidency(addresses->data(), addresses->size(), flags->data())) return 0;
	uint64_t n = 0;
	for (const uint8_t f: *flags) n += f != 0 ? 0x1000 : 0;
	return n;
}

struct Parent {
	uint64_t addr = 0;
	Ranges   previous, released, invalidated, stuck;
	Snap     snap;
	uint64_t reset = 0, reused = 0, touched = 0, refused = 0, busy = 0, invalid = 0, held_max_us = 0;
	uint64_t ram_used = 0, ram_free = 0, released_now = 0;
};

void Worker() {
	LocalPlatform::SetThreadName("Kyty.RamNativa");
	const int mode = (TagMode() == 1 && ::FrameNativo2::Mode() != 0) ? 2 : TagMode(); // (FRAME-NATIVO2 importa arenas: solo medir)
	auto*     ps   = new Parent[MaxParents]; // (nunca se destruye)
	int       np   = 0;
	for (int i = 0; i < MaxParents; i++) {
		ps[i].snap.all.reserve(256);
		ps[i].snap.free.reserve(256 * 61);
	}
	std::vector<uint64_t>        addresses;
	std::vector<uint8_t>         flags;
	std::pair<uint64_t, uint64_t> skipped[MaxDone];
	auto                         said = std::chrono::steady_clock::now();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		if (g_pool_end.load(std::memory_order_acquire) == 0 || g_guest_address_space == nullptr) continue;
		const uint64_t pool_begin = g_pool_begin.load(std::memory_order_relaxed), pool_end = g_pool_end.load(std::memory_order_acquire);
		// 0. Padres nuevos (de los Init vistos por ARENA-DUENO), comprobados con su vtable: +0xf8 = AllocTag, +0x100 = FreeTag.
		const uint64_t eboot = ::ArenaDueno::g_eboot.load(std::memory_order_relaxed);
		const int      ni    = ::ArenaDueno::g_inst_count.load(std::memory_order_acquire);
		for (int i = 0; i < ni && eboot != 0; i++) {
			const uint64_t p = ::ArenaDueno::g_inst[i].parent;
			if (p == 0) continue;
			bool known = false;
			for (int k = 0; k < np; k++) known = known || ps[k].addr == p;
			if (known || np == MaxParents) continue;
			const uint64_t vt = Rd(p);
			if (vt == 0 || Rd(vt + 0xf8) != eboot + AllocTagVa || Rd(vt + 0x100) != eboot + FreeTagVa) continue;
			ps[np++].addr = p;
			std::printf("[ram-nativa] padre TaggedFrameAllocator 0x%llx (vtable AllocTag eboot+0x7fe350, FreeTag eboot+0x7fe540)\n",
			            static_cast<unsigned long long>(p));
			std::fflush(stdout);
		}
		const bool report = std::chrono::steady_clock::now() - said >= std::chrono::seconds(10);
		for (int k = 0; k < np; k++) {
			Parent& P = ps[k];
			// 1. Lo libre ahora (candado del juego).
			auto t0 = std::chrono::steady_clock::now();
			if (!Lock(P.addr)) {
				P.busy++;
				continue;
			}
			const bool ok = Read(P.addr, &P.snap);
			Unlock(P.addr);
			const auto us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
			P.held_max_us = std::max(P.held_max_us, us);
			if (!ok) {
				P.invalid++;
				continue;
			}
			Ranges now = P.snap.free;
			Normalize(&now);
			// 2. Devuelto que ya no esta libre: el juego lo volvio a pedir (AllocTag).
			const Ranges kept = Intersect(P.released, now);
			P.reused += Bytes(P.released) - Bytes(kept);
			P.released    = kept;
			P.invalidated = Intersect(P.invalidated, now);
			P.stuck       = Intersect(P.stuck, now);
			// 3. Libre en esta pasada y en la anterior, sin devolver.
			const Ranges todo = Subtract(Subtract(Intersect(P.previous, now), P.released), P.stuck);
			P.previous        = now;
			if (mode == 1 && !todo.empty()) {
				// 4. La GPU avisada antes (fuera del candado del juego: las caches toman los suyos). Nunca en el hilo de la GPU.
				for (const auto& [b, e]: Subtract(todo, P.invalidated)) {
					(void)TryPrepareHostOverwrite(b, e - b);
					P.invalidated.emplace_back(b, e);
				}
				Normalize(&P.invalidated);
				// 5. Devuelto con el candado otra vez y cada bloque comprobado todavia libre (bit 0 en [c+0x18]).
				GuestBackingStore::ReleaseCounts counts;
				counts.skipped     = skipped;
				counts.skipped_max = MaxDone;
				Ranges done;
				done.reserve(todo.size());
				int state = 2;
				(void)g_guest_address_space->RunQuietWith([&] {
					if (!Lock(P.addr)) return;
					Snap& s = P.snap;
					if (!Read(P.addr, &s)) {
						Unlock(P.addr);
						state = 1;
						return;
					}
					Ranges still = s.free;
					Normalize(&still);
					for (const auto& [b, e]: Intersect(todo, still)) {
						(void)ReleaseRange(b, e - b, pool_begin, pool_end, true, &counts);
						done.emplace_back(b, e);
					}
					Unlock(P.addr);
					state = 0;
				});
				if (state == 0) {
					P.reset += counts.done;
					P.refused += counts.refused;
					for (const auto& r: done) P.released.push_back(r);
					Normalize(&P.released);
					for (size_t i = 0; i < counts.skipped_count; i++) P.stuck.push_back(skipped[i]);
					Normalize(&P.stuck);
					if (!done.empty())
						(void)g_guest_address_space->RunQuietWith([&] {
							for (const auto& [b, e]: done) TrimRange(b, e - b, pool_begin, pool_end);
						});
				} else {
					(state == 2 ? P.busy : P.invalid)++;
				}
			}
			// 6. Cada 10 s: RAM exacta (pagina a pagina) de lo usado y de lo libre; lo devuelto que volvio a RAM estando libre.
			if (report) {
				const Ranges given = Subtract(P.released, P.stuck);
				const uint64_t back = InRam(given, &addresses, &flags);
				P.touched += back;
				if (back != 0) {
					Ranges b2;
					for (size_t i = 0; i < addresses.size(); i++)
						if (flags[i] != 0) b2.emplace_back(addresses[i], addresses[i] + 0x1000);
					Normalize(&b2);
					P.released    = Subtract(P.released, b2);
					P.invalidated = Subtract(P.invalidated, b2);
				}
				Ranges all = P.snap.all;
				Normalize(&all);
				P.ram_used     = InRam(Subtract(all, now), &addresses, &flags);
				P.ram_free     = InRam(now, &addresses, &flags);
				P.released_now = Bytes(Subtract(P.released, P.stuck));
			}
		}
		if (!report) continue;
		said                 = std::chrono::steady_clock::now();
		constexpr double MiB = 1024.0 * 1024.0;
		uint64_t         tr = 0, tu = 0, tp = 0, tram = 0, tsaved = 0;
		for (int k = 0; k < np; k++) {
			const Parent& P   = ps[k];
			const uint64_t blk = P.snap.block;
			const Ranges& now  = P.previous;
			std::printf("[ram-nativa] 10 s, MiB (KYTY_RAM_NATIVA=%d%s) padre 0x%llx (TaggedFrameAllocator, bloque %llu KiB): "
			            "reservado %.0f ([+0x98]), en uso %.0f ([+0x90]), maximo del juego %.0f ([+0xa0]), libre %.0f; "
			            "en RAM: usado %.1f + libre %.1f = %.1f; devuelto ahora %.1f, devuelto en total %.1f, reutilizado por el "
			            "juego %.1f, vuelto a RAM estando libre %.1f, rechazado %.1f; candado ocupado %llu, no valido %llu, "
			            "lectura mas larga %.2f ms\n",
			            mode, mode == 2 ? ", solo mide" : "", static_cast<unsigned long long>(P.addr),
			            static_cast<unsigned long long>(blk >> 10u), static_cast<double>(P.snap.reserved * blk) / MiB,
			            static_cast<double>(P.snap.used * blk) / MiB, static_cast<double>(P.snap.peak * blk) / MiB,
			            static_cast<double>(Bytes(now)) / MiB, static_cast<double>(P.ram_used) / MiB,
			            static_cast<double>(P.ram_free) / MiB, static_cast<double>(P.ram_used + P.ram_free) / MiB,
			            static_cast<double>(P.released_now) / MiB, static_cast<double>(P.reset) / MiB,
			            static_cast<double>(P.reused) / MiB, static_cast<double>(P.touched) / MiB,
			            static_cast<double>(P.refused) / MiB, static_cast<unsigned long long>(P.busy),
			            static_cast<unsigned long long>(P.invalid), static_cast<double>(P.held_max_us) / 1000.0);
			tr += P.snap.reserved * blk;
			tu += P.snap.used * blk;
			tp += P.snap.peak * blk;
			tram += P.ram_used + P.ram_free;
			tsaved += P.released_now;
		}
		std::printf("[ram-nativa] 10 s, MiB TOTAL TaggedFrameAllocator (%d padres): reservado %.0f, en uso %.0f, maximo %.0f, "
		            "en RAM %.1f, ahorrado ahora %.1f\n",
		            np, static_cast<double>(tr) / MiB, static_cast<double>(tu) / MiB, static_cast<double>(tp) / MiB,
		            static_cast<double>(tram) / MiB, static_cast<double>(tsaved) / MiB);
		std::fflush(stdout);
	}
}
} // namespace TagNativa

void StartTag() {
	static std::atomic<bool> started {false};
	if (TagMode() == 0 || started.exchange(true)) return;
	std::thread(TagNativa::Worker).detach();
	std::printf("[ram-nativa] ON (KYTY_RAM_NATIVA=%d): bloques libres de los TaggedFrameAllocator (AllocTag 0x7fe350, FreeTag "
	            "0x7fe540, mascara [trozo+0x18], candado [padre+0x80]) %s; linea [ram-nativa] cada 10 s%s\n",
	            TagMode(), TagMode() == 1 ? "devueltos a Windows (MEM_RESET tras 2 pasadas libres, commit intacto)" : "solo medidos",
	            (TagMode() == 1 && ::FrameNativo2::Mode() != 0) ? " (FRAME-NATIVO2 encendido: solo mide)" : "");
	std::fflush(stdout);
}

void Start(uint64_t start, uint64_t len) {
	StartNativa2(); // (nativa2)
	StartTag();     // (RAM-NATIVA)
	if (!Requested()) return;
	const int mode = RequestedMode();
	g_active.store(mode, std::memory_order_release);
	std::thread(Worker).detach();
	if (g_native.load(std::memory_order_relaxed))
		std::printf("Freed release: on (KYTY_DS_FREED_RELEASE=%d, native guest memory, ram14): the game's free-block cache "
		            "(ReleasePages 0x80f550 keeps freed blocks mapped; 0x80e7a0 hands them out again unzeroed), free at two "
		            "passes in a row, %s at its guest address (MEM_RESET + out of the working set; commit unchanged); "
		            "unmaps already give memory back; pool 0x%llx+%llu MiB; a 'Freed release (30 s' line with the GB\n",
		            mode, mode == 1 ? "reset" : "counted, nothing changed", static_cast<unsigned long long>(start),
		            static_cast<unsigned long long>(len >> 20u));
	else if (mode == 1)
		std::printf("Freed release: on (KYTY_DS_FREED_RELEASE=1): what the game's page allocator holds dead (its free-block "
		            "cache, free at two passes in a row, and what it unmaps) is given back to Windows (DiscardVirtualMemory "
		            "through the game's view and the alias; the commit charge stays: a page-file section); pool 0x%llx+%llu "
		            "MiB; totals in the [ram] line (KYTY_RAM_LINE=1)\n",
		            static_cast<unsigned long long>(start), static_cast<unsigned long long>(len >> 20u));
	else
		std::printf("Freed release: measuring only (KYTY_DS_FREED_RELEASE=2): what KYTY_DS_FREED_RELEASE=1 would give back, "
		            "nothing changed; pool 0x%llx+%llu MiB; totals in the [ram] line (KYTY_RAM_LINE=1)\n",
		            static_cast<unsigned long long>(start), static_cast<unsigned long long>(len >> 20u));
	std::fflush(stdout);
}

// Just before a guest unmap of direct memory at [vaddr, vaddr + len), on the unmapping thread: what of it is the pool's
// and mapped nowhere else, given back (mode 1) or counted (mode 2). Timed.
void BeforeUnmap(uint64_t vaddr, uint64_t len) {
	const int mode = g_active.load(std::memory_order_relaxed);
	if (mode == 0) return;
	// (ram9) KYTY_DS_FREED_RELEASE_UNMAP=0: not at the unmap (41 of the 53 GB of log 22:40; 73,044 calls): the free-block
	// cache only (its resets under the quiet guest memory). For an A/B of the restart.
	static const bool at_unmap = [] {
		const char* v = std::getenv("KYTY_DS_FREED_RELEASE_UNMAP");
		return v == nullptr || v[0] != '0';
	}();
	if (!at_unmap) return;
	const uint64_t end = g_pool_end.load(std::memory_order_acquire);
	if (end == 0 || g_guest_address_space == nullptr) return;
	if (g_native.load(std::memory_order_relaxed)) return; // (ram14: a native unmap releases the memory itself)
	const auto                       t0 = std::chrono::steady_clock::now();
	GuestBackingStore::ReleaseCounts counts;
	(void)g_guest_address_space->ReleaseBackingSole(vaddr, len, g_pool_begin.load(std::memory_order_relaxed), end, mode == 1,
	                                                &counts);
	if (counts.done == 0 && counts.shared == 0 && counts.refused == 0) return;
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
	auto& st = Stat();
	st.unmap.fetch_add(counts.done, std::memory_order_relaxed);
	st.discarded.fetch_add(counts.discarded, std::memory_order_relaxed);
	st.reset.fetch_add(counts.reset, std::memory_order_relaxed);
	st.shared.fetch_add(counts.shared, std::memory_order_relaxed);
	st.refused.fetch_add(counts.refused, std::memory_order_relaxed);
	st.unmap_calls.fetch_add(1, std::memory_order_relaxed);
	st.unmap_us.fetch_add(us, std::memory_order_relaxed);
	Max(st.unmap_max_us, us);
}

// The [ram] line's part (its thread only), MiB; empty when off.
void Line(char* out, size_t size) {
	out[0]         = 0;
	const int mode = g_active.load(std::memory_order_relaxed);
	if (mode == 0) return;
	auto&           st        = Stat();
	static uint64_t last      = 0;
	const auto      mib       = [](uint64_t b) { return static_cast<unsigned long long>(b >> 20u); };
	const auto      get       = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
	const uint64_t  total     = get(st.cache) + get(st.unmap);
	const uint64_t  checked   = get(st.checked);
	if (mode == 1) {
		std::snprintf(out, size,
		              "devuelto a Windows (KYTY_DS_FREED_RELEASE=1) %llu en total (+%llu): de bloques libres %llu, al "
		              "desmapear %llu [DiscardVirtualMemory %llu, MEM_RESET %llu]; reutilizado o desmapeado por el juego "
		              "%llu; vuelto a RAM estando libre %llu; rechazado: con otro mapeo %llu, por Windows %llu, bloques a "
		              "medio mapear ahora %llu; ahora libres %llu, de ellos devueltos %llu; tras devolver seguian en RAM %llu "
		              "de %llu muestras por la vista del juego y %llu por el alias; avisado a la GPU %llu; candado del juego "
		              "%llu pasadas (ocupado %llu, listas no validas %llu), %.1f ms en total, la mas larga %.2f ms; al "
		              "desmapear %llu llamadas, %.1f ms, la mas larga %.2f ms",
		              mib(total), mib(total - last), mib(get(st.cache)), mib(get(st.unmap)), mib(get(st.discarded)),
		              mib(get(st.reset)), mib(get(st.reused)), mib(get(st.touched)), mib(get(st.shared)), mib(get(st.refused)),
		              mib(get(st.partial)), mib(get(st.now_free)), mib(get(st.now_released)),
		              static_cast<unsigned long long>(get(st.still_game)), static_cast<unsigned long long>(checked),
		              static_cast<unsigned long long>(get(st.still_alias)), mib(get(st.invalidated)),
		              static_cast<unsigned long long>(get(st.passes)), static_cast<unsigned long long>(get(st.busy)),
		              static_cast<unsigned long long>(get(st.invalid)), static_cast<double>(get(st.held_us)) / 1000.0,
		              static_cast<double>(st.held_max_us.exchange(0, std::memory_order_relaxed)) / 1000.0,
		              static_cast<unsigned long long>(get(st.unmap_calls)), static_cast<double>(get(st.unmap_us)) / 1000.0,
		              static_cast<double>(st.unmap_max_us.exchange(0, std::memory_order_relaxed)) / 1000.0);
	} else {
		std::snprintf(out, size,
		              "se devolveria (KYTY_DS_FREED_RELEASE=2, solo mide) %llu en total (+%llu): de bloques libres %llu, al "
		              "desmapear %llu; reutilizado o desmapeado por el juego %llu; ahora libres %llu, ya contados %llu, en RAM "
		              "unos %llu (muestra cada 5 s); no se devolveria: con otro mapeo %llu, bloques a medio mapear ahora %llu; "
		              "candado del juego %llu pasadas (ocupado %llu, listas no validas %llu), %.1f ms en total, la mas larga "
		              "%.2f ms",
		              mib(total), mib(total - last), mib(get(st.cache)), mib(get(st.unmap)), mib(get(st.reused)),
		              mib(get(st.now_free)), mib(get(st.sampled)), mib(get(st.in_ram)), mib(get(st.shared)),
		              mib(get(st.partial)), static_cast<unsigned long long>(get(st.passes)),
		              static_cast<unsigned long long>(get(st.busy)), static_cast<unsigned long long>(get(st.invalid)),
		              static_cast<double>(get(st.held_us)) / 1000.0,
		              static_cast<double>(st.held_max_us.exchange(0, std::memory_order_relaxed)) / 1000.0);
	}
	last = total;
}

} // namespace FreedRelease

// KernelAllocateDirectMemory: the page allocator's pool is the one direct allocation of 4 GiB or more (12 GiB less
// 64 KiB with the allocator clamp; Kyty's store is 13.5 GiB, so there is only one).
void NotePool(uint64_t start, uint64_t len) {
	if ((!Requested() && !FreeCache::Requested() && !FreedRelease::Requested() && FreedRelease::Nativa2Mode() == 0) ||
	    len < (uint64_t {4} << 30u) ||
	    g_pool_end.load(std::memory_order_relaxed) != 0)
		return;
	// (ram14) With native guest memory, KYTY_DS_FREED_RELEASE alone (its native path); the other two still need the view.
	if (g_guest_address_space != nullptr && g_guest_address_space->IsNative() &&
	    (FreedRelease::Requested() || FreedRelease::Nativa2Mode() != 0) &&
	    Libs::Graphics::DemonsSouls::IsSupportedGame()) {
		if (Requested() || FreeCache::Requested()) {
			std::printf("Freed discard / free cache reset: off with native guest memory (KYTY_DS_FREED_RELEASE does it)\n");
			std::fflush(stdout);
		}
		g_pool_begin.store(start, std::memory_order_relaxed);
		g_pool_end.store(start + len, std::memory_order_release);
		FreedRelease::g_native.store(true, std::memory_order_relaxed);
		FreedRelease::Start(start, len);
		return;
	}
	if (g_guest_address_space == nullptr || g_guest_address_space->IsNative() ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame()) {
		std::printf("Freed discard / free cache reset / freed release: off (KYTY_DS_FREED_DISCARD, KYTY_DS_FREECACHE_RESET "
		            "and KYTY_DS_FREED_RELEASE need Demon's Souls and the guest store's backing view, not KYTY_NATIVE_MEMORY)\n");
		std::fflush(stdout);
		return;
	}
	g_pool_begin.store(start, std::memory_order_relaxed);
	g_pool_end.store(start + len, std::memory_order_release);
	FreedRelease::Start(start, len); // (ram6)
	if (FreeCache::Requested() && FreedRelease::Requested()) { // (ram6: the same blocks; one walker of the lists at a time)
		std::printf("Free cache reset: off (KYTY_DS_FREED_RELEASE covers the same blocks)\n");
		std::fflush(stdout);
	} else if (FreeCache::Requested()) { // (ram3)
		std::thread(FreeCache::Worker).detach();
		std::printf("Free cache reset: on (KYTY_DS_FREECACHE_RESET=1): the blocks the game's page allocator frees and keeps "
		            "mapped are reset (MEM_RESET through the game's view, under the game's allocator lock) once free at two "
		            "passes in a row; pool 0x%llx+%llu MiB\n",
		            static_cast<unsigned long long>(start), static_cast<unsigned long long>(len >> 20u));
		std::fflush(stdout);
	}
	if (!Requested()) return;
	if (RequestedMode() == 2) {
		// (ram2) No queue and no thread: BeforeUnmap resets each range as the game unmaps it.
		g_active.store(2, std::memory_order_release);
		std::printf("Freed discard: on (KYTY_DS_FREED_DISCARD=2): the page allocator's pool is direct memory 0x%llx+%llu "
		            "MiB; its pages are reset through the game's view as the game unmaps them (MEM_RESET: no page-file "
		            "write of them, commit charge unchanged); totals in the [ram] line (KYTY_RAM_LINE=1)\n",
		            static_cast<unsigned long long>(start), static_cast<unsigned long long>(len >> 20u));
		std::fflush(stdout);
		return;
	}
	g_active.store(1, std::memory_order_release);
	std::thread(Worker).detach();
	std::printf("Freed discard: on (KYTY_DS_FREED_DISCARD=1): the page allocator's pool is direct memory 0x%llx+%llu MiB; "
	            "its pages the game unmaps are dropped from RAM (commit charge unchanged)\n",
	            static_cast<unsigned long long>(start), static_cast<unsigned long long>(len >> 20u));
	std::fflush(stdout);
}

// ReleaseDirectMemoryInternal: the pool given back (never seen in Demon's Souls): off; the RAM flush owns released memory.
void NoteRelease(uint64_t start, uint64_t len) {
	const uint64_t end = g_pool_end.load(std::memory_order_acquire);
	if (end == 0 || start >= end || start + len <= g_pool_begin.load(std::memory_order_relaxed)) return;
	g_pool_end.store(0, std::memory_order_release);
	g_active.store(0, std::memory_order_release);
	FreedRelease::g_active.store(0, std::memory_order_release); // (ram6)
	std::printf("Freed discard: off (the page allocator's pool was released)\n");
	std::fflush(stdout);
}

// UnmapRange: direct memory [offset, offset + len) unmapped. Only the pool's whole 64 KiB granules.
void Note(uint64_t offset, uint64_t len) {
	if (g_active.load(std::memory_order_relaxed) != 1) return; // (ram2: mode 2 did its work in BeforeUnmap)
	const uint64_t end = g_pool_end.load(std::memory_order_acquire);
	if (end == 0) return;
	const uint64_t pool_begin = g_pool_begin.load(std::memory_order_relaxed);
	constexpr uint64_t Granule = 0x10000;
	const uint64_t     b       = std::max(offset, pool_begin);
	const uint64_t     e       = std::min(offset + len, end);
	const uint64_t     ab      = (b + Granule - 1) & ~(Granule - 1);
	const uint64_t     ae      = e & ~(Granule - 1);
	if (ae <= ab) return;
	g_queued.fetch_add(ae - ab, std::memory_order_relaxed);
	g_calls.fetch_add(1, std::memory_order_relaxed);
	auto& queue = GetQueue();
	{
		std::lock_guard lock(queue.mutex);
		if (queue.ranges.size() < (size_t {1} << 20u)) queue.ranges.emplace_back(ab, ae - ab);
	}
	queue.ready.notify_one();
}

// (ram2, KYTY_DS_FREED_DISCARD=2) Just before a guest unmap of direct memory at [vaddr, vaddr + len): what of it is
// the pool's and mapped nowhere else is reset through the game's view (the unmap that follows takes it out of the
// working set clean). On the unmapping thread: one shared lock of the store and one VirtualAlloc(MEM_RESET) a mapping,
// timed (the [ram] line gives calls, total and longest).
void BeforeUnmap(uint64_t vaddr, uint64_t len) {
	if (g_active.load(std::memory_order_relaxed) != 2) return;
	const uint64_t end = g_pool_end.load(std::memory_order_acquire);
	if (end == 0 || g_guest_address_space == nullptr) return;
	const auto     t0     = std::chrono::steady_clock::now();
	uint64_t       shared = 0, refused = 0;
	const uint64_t done   = g_guest_address_space->ResetBackingBeforeUnmap(
        vaddr, len, g_pool_begin.load(std::memory_order_relaxed), end, &shared, &refused);
	if (done == 0 && shared == 0 && refused == 0) return;
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
	g_reset_bytes.fetch_add(done, std::memory_order_relaxed);
	g_reset_shared.fetch_add(shared, std::memory_order_relaxed);
	g_reset_refused.fetch_add(refused, std::memory_order_relaxed);
	g_reset_calls.fetch_add(1, std::memory_order_relaxed);
	g_reset_us.fetch_add(us, std::memory_order_relaxed);
	uint64_t longest = g_reset_max_us.load(std::memory_order_relaxed);
	while (us > longest && !g_reset_max_us.compare_exchange_weak(longest, us, std::memory_order_relaxed)) {
	}
}

// (ram2) The counters for the [ram] line.
void Stats(uint64_t* v) {
	const int mode = g_active.load(std::memory_order_relaxed);
	v[0]           = static_cast<uint64_t>(mode);
	if (mode == 2) {
		v[1] = g_reset_bytes.load(std::memory_order_relaxed);
		v[3] = g_reset_shared.load(std::memory_order_relaxed);
		v[4] = g_reset_refused.load(std::memory_order_relaxed);
		v[5] = g_reset_calls.load(std::memory_order_relaxed);
		v[6] = g_reset_us.load(std::memory_order_relaxed);
		v[7] = g_reset_max_us.exchange(0, std::memory_order_relaxed);
	} else {
		v[1] = g_dropped.load(std::memory_order_relaxed);
		v[2] = g_queued.load(std::memory_order_relaxed);
		v[3] = g_covered.load(std::memory_order_relaxed);
		v[4] = g_refused.load(std::memory_order_relaxed);
		v[5] = g_calls.load(std::memory_order_relaxed);
	}
}


// (ram9) The guest store's pages in RAM ("almacen del juego": 6.1-6.3 GB of a 16 GB PC in log 22:40, available 330-1900
// MiB all along, free 0-8): how much of it the game uses, and what it does not use given back to Windows' standby list.
//
// KYTY_STORE_HOTSET=1 (or with KYTY_STORE_TRIM=1): every 30 s, 512 granules of 64 KiB picked at random among the game's
// mappings of the store; those in RAM (their first page valid in the working set) are trimmed (VirtualUnlock, through
// the game's view and the alias, guest memory quiet) and looked at again 1 s and 10 s later: back in RAM = touched in
// that time (a soft fault from the standby/modified list: no disk, no byte lost). Line:
//   Store hot set (30 s): mapped M MiB, in RAM ~R MiB (r of n sampled); touched again within 1 s a% (~A MiB), within
//   10 s b% (~B MiB); all-time ...
// A = what about 60 frames use; B what 10 s use. Cost: 32 MiB trimmed every 30 s.
//
// KYTY_STORE_TRIM=1 (off): while available memory is below KYTY_STORE_TRIM_BELOW_MB (1500), granules of 2 MiB of the
// store go out of the working set (round robin over the game's mappings, both views), up to KYTY_STORE_TRIM_MB_S (256)
// MiB a second, 8 granules per quiet section (~1 ms with every guest protection change held off), until available is
// KYTY_STORE_TRIM_BELOW_MB + 512 again. A granule the game touched again within 2 s of its trim (its first page back) is
// hot: left alone for 120 s. What leaves goes to standby (clean) or modified (dirty: Windows writes or compresses it):
// it stays in Windows' lists, counted as available, and comes back by a soft fault while it is there. No byte is lost;
// a page Windows gave away meanwhile comes back from the page file (a hard fault: only cold pages, by the rule above).
// Line every 30 s: "Store trim (30 s): ..." (MiB trimmed, hot granules skipped and found, quiet sections and time).
namespace StoreTrim {
using Granule = GuestBackingStore::Granule;

bool InRam(uint64_t address) {
	PSAPI_WORKING_SET_EX_INFORMATION page {};
	page.VirtualAddress = reinterpret_cast<void*>(address & ~uint64_t {0xfff});
	return K32QueryWorkingSetEx(GetCurrentProcess(), &page, sizeof(page)) != 0 && page.VirtualAttributes.Valid != 0;
}

// Granules out of the working set, both views, guest memory quiet. False when the quiet section was refused.
bool Trim(const Granule* granules, size_t count, uint64_t bytes_each) {
	if (g_guest_address_space == nullptr || count == 0) return false;
	return g_guest_address_space->RunQuietWith([&] {
		for (size_t i = 0; i < count; i++) {
			const uint64_t bytes = std::min(bytes_each, granules[i].bytes);
			(void)VirtualUnlock(reinterpret_cast<void*>(granules[i].guest), bytes);
			(void)VirtualUnlock(granules[i].alias, bytes);
		}
	});
}

// (ram10) Windows' paging I/O, pages a second (PDH "Memory" counters, English names; pdh.dll loaded here, no import):
// [0] Pages Input/sec (read from disk by hard faults), [1] Pages Output/sec (written to disk - page file or a mapped
// file such as KYTY_BACKING_FILE's - to free RAM), [2] Page Writes/sec (those writes, operations). False when missing.
bool PagingCounters(double* out) {
	using OpenQuery  = LONG(WINAPI*)(const wchar_t*, DWORD_PTR, HANDLE*);
	using AddCounter = LONG(WINAPI*)(HANDLE, const wchar_t*, DWORD_PTR, HANDLE*);
	using Collect    = LONG(WINAPI*)(HANDLE);
	using GetValue   = LONG(WINAPI*)(HANDLE, DWORD, DWORD*, void*);
	struct Value { // (PDH_FMT_COUNTERVALUE on x64)
		DWORD  status;
		DWORD  padding;
		double value;
	};
	static_assert(sizeof(Value) == 16);
	struct State {
		bool     ok      = false;
		Collect  collect = nullptr;
		GetValue get     = nullptr;
		HANDLE   query   = nullptr;
		HANDLE   counters[3] {};
	};
	static State state = [] {
		State   s;
		HMODULE pdh = LoadLibraryW(L"pdh.dll");
		if (pdh == nullptr) return s;
		const auto open = reinterpret_cast<OpenQuery>(reinterpret_cast<void*>(GetProcAddress(pdh, "PdhOpenQueryW")));
		const auto add  = reinterpret_cast<AddCounter>(reinterpret_cast<void*>(GetProcAddress(pdh, "PdhAddEnglishCounterW")));
		s.collect       = reinterpret_cast<Collect>(reinterpret_cast<void*>(GetProcAddress(pdh, "PdhCollectQueryData")));
		s.get           = reinterpret_cast<GetValue>(reinterpret_cast<void*>(GetProcAddress(pdh, "PdhGetFormattedCounterValue")));
		if (open == nullptr || add == nullptr || s.collect == nullptr || s.get == nullptr) return s;
		if (open(nullptr, 0, &s.query) != 0 || s.query == nullptr) return s;
		const wchar_t* names[3] = {L"\\Memory\\Pages Input/sec", L"\\Memory\\Pages Output/sec", L"\\Memory\\Page Writes/sec"};
		for (int i = 0; i < 3; i++)
			if (add(s.query, names[i], 0, &s.counters[i]) != 0) return s;
		(void)s.collect(s.query); // (a rate needs two samples)
		s.ok = true;
		return s;
	}();
	if (!state.ok || state.collect(state.query) != 0) return false;
	for (int i = 0; i < 3; i++) {
		constexpr DWORD FormatDouble = 0x00000200; // PDH_FMT_DOUBLE
		Value           v {};
		if (state.get(state.counters[i], FormatDouble, nullptr, &v) != 0) return false;
		out[i] = v.value;
	}
	return true;
}

uint64_t Number(const char* name, uint64_t fallback) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] != '\0' ? static_cast<uint64_t>(std::strtoull(v, nullptr, 10)) : fallback;
}

void Worker() {
	LocalPlatform::SetThreadName("Kyty.StoreTrim");
	std::this_thread::sleep_for(std::chrono::seconds(10)); // (the port sets its variables after the start)
	const bool trim_on = Number("KYTY_STORE_TRIM", 0) == 1;
	const bool hot_on  = trim_on || Number("KYTY_STORE_HOTSET", 0) == 1;
	if (!trim_on && !hot_on) return;
	const uint64_t below   = Number("KYTY_STORE_TRIM_BELOW_MB", 1500) << 20u;
	const uint64_t target  = below + (uint64_t {512} << 20u);
	const uint64_t per_s   = std::max<uint64_t>(Number("KYTY_STORE_TRIM_MB_S", 256), 16) << 20u;
	constexpr uint64_t Big = uint64_t {2} << 20u, Probe = uint64_t {64} << 10u;
	std::printf("Store trim: %s (KYTY_STORE_TRIM=%d; below %llu MiB available, %llu MiB/s at most); hot set measure %s; "
	            "store %s\n",
	            trim_on ? "on" : "off", trim_on ? 1 : 0, static_cast<unsigned long long>(below >> 20u),
	            static_cast<unsigned long long>(per_s >> 20u), hot_on ? "on (KYTY_STORE_HOTSET)" : "off",
	            g_guest_address_space->StoreInFile()
	                ? "in a file (KYTY_BACKING_FILE: trimmed dirty pages are written to it, not to memory compression)"
	                : "in the page file (trimmed dirty pages go to Windows' memory compression: KYTY_BACKING_FILE=1 sends them to disk)");
	std::fflush(stdout);
	std::vector<Granule> big, probe;
	std::unordered_map<uint64_t, uint64_t> hot_until; // granule guest address -> tick it is hot until
	struct Pending {
		uint64_t guest = 0, tick = 0;
	};
	std::deque<Pending> pending; // trimmed, first page looked at 2 s later
	size_t               cursor = 0;
	uint64_t             tick = 0, last_list = 0;
	uint64_t trimmed = 0, skipped = 0, found_hot = 0, quiet = 0, quiet_refused = 0, quiet_us = 0;
	uint64_t all_n = 0, all_1 = 0, all_10 = 0;
	std::mt19937_64 random(GetTickCount64());
	for (;; tick++) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		const uint64_t now_ms = GetTickCount64();
		if (big.empty() || tick - last_list >= 30) {
			g_guest_address_space->StoreGranules(Big, &big);
			last_list = tick;
			if (cursor >= big.size()) cursor = 0;
		}
		// Hot: back in RAM within 2 s of its trim.
		while (!pending.empty() && now_ms - pending.front().tick >= 2000) {
			if (InRam(pending.front().guest)) {
				hot_until[pending.front().guest] = now_ms + 120000;
				found_hot++;
			}
			pending.pop_front();
		}
		// The hot set measure, every 30 s.
		if (hot_on && tick % 30 == 5 && !big.empty()) {
			g_guest_address_space->StoreGranules(Probe, &probe);
			uint64_t mapped = 0;
			for (const auto& g: probe) mapped += g.bytes;
			std::vector<Granule> picked, resident;
			for (size_t i = 0; i < 512 && !probe.empty(); i++) picked.push_back(probe[random() % probe.size()]);
			for (const auto& g: picked)
				if (InRam(g.guest)) resident.push_back(g);
			if (!resident.empty() && Trim(resident.data(), resident.size(), Probe)) {
				size_t back_1 = 0, back_10 = 0;
				std::this_thread::sleep_for(std::chrono::seconds(1));
				for (const auto& g: resident) back_1 += InRam(g.guest) ? 1 : 0;
				std::this_thread::sleep_for(std::chrono::seconds(9));
				for (const auto& g: resident) back_10 += InRam(g.guest) ? 1 : 0;
				const double in_ram = static_cast<double>(mapped) * static_cast<double>(resident.size()) / picked.size();
				const double f1 = static_cast<double>(back_1) / resident.size(), f10 = static_cast<double>(back_10) / resident.size();
				all_n += resident.size();
				all_1 += back_1;
				all_10 += back_10;
				std::printf("Store hot set (30 s): mapped %llu MiB, in RAM ~%.0f MiB (%zu of %zu sampled); touched again within 1 s "
				            "%.0f%% (~%.0f MiB), within 10 s %.0f%% (~%.0f MiB); all-time %.0f%% / %.0f%% of %llu\n",
				            static_cast<unsigned long long>(mapped >> 20u), in_ram / 1048576.0, resident.size(), picked.size(),
				            f1 * 100.0, in_ram * f1 / 1048576.0, f10 * 100.0, in_ram * f10 / 1048576.0,
				            100.0 * static_cast<double>(all_1) / static_cast<double>(all_n),
				            100.0 * static_cast<double>(all_10) / static_cast<double>(all_n), static_cast<unsigned long long>(all_n));
				std::fflush(stdout);
			}
			tick += 10;
		}
		// The trim, while available memory is short.
		if (trim_on && !big.empty()) {
			const uint64_t available = RamGuard::AvailablePhysical();
			static bool    active    = false;
			if (available != 0 && available < below) active = true;
			if (available == 0 || available >= target) active = false;
			if (active) {
				uint64_t budget = per_s;
				std::vector<Granule> batch;
				for (size_t seen = 0; seen < big.size() && budget >= Big; seen++) {
					const Granule& g = big[cursor];
					cursor           = cursor + 1 < big.size() ? cursor + 1 : 0;
					const auto hot   = hot_until.find(g.guest);
					if (hot != hot_until.end()) {
						if (hot->second > now_ms) {
							skipped++;
							continue;
						}
						hot_until.erase(hot);
					}
					if (!InRam(g.guest)) continue;
					batch.push_back(g);
					budget -= Big;
					if (batch.size() == 8) {
						const auto t0 = std::chrono::steady_clock::now();
						if (Trim(batch.data(), batch.size(), Big)) {
							quiet++;
							for (const auto& b: batch) {
								trimmed += b.bytes;
								pending.push_back({b.guest, GetTickCount64()});
							}
						} else {
							quiet_refused++;
						}
						quiet_us += static_cast<uint64_t>(
						    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
						batch.clear();
						std::this_thread::sleep_for(std::chrono::milliseconds(2));
					}
				}
				if (!batch.empty() && Trim(batch.data(), batch.size(), Big)) {
					quiet++;
					for (const auto& b: batch) {
						trimmed += b.bytes;
						pending.push_back({b.guest, GetTickCount64()});
					}
				}
			}
			// (ram10) Windows' paging I/O beside it, averaged over the 30 s (sampled each second).
			static double paging_sum[3] {};
			static uint64_t paging_n = 0;
			double          paging[3] {};
			if (PagingCounters(paging)) {
				for (int i = 0; i < 3; i++) paging_sum[i] += paging[i];
				paging_n++;
			}
			if (tick % 30 == 0) {
				const double n = paging_n != 0 ? static_cast<double>(paging_n) : 1.0;
				std::printf("Store trim (30 s): %llu MiB out of the working set in all, %s now (available %llu MiB); hot granules "
				            "found %llu, skipped %llu; quiet sections %llu (refused %llu), %.1f ms in all; Windows paging (whole "
				            "PC, 30 s mean): read from disk %.0f pages/s (%.1f MiB/s), written to disk %.0f pages/s (%.1f MiB/s) "
				            "in %.0f writes/s\n",
				            static_cast<unsigned long long>(trimmed >> 20u), active ? "trimming" : "idle",
				            static_cast<unsigned long long>(available >> 20u), static_cast<unsigned long long>(found_hot),
				            static_cast<unsigned long long>(skipped), static_cast<unsigned long long>(quiet),
				            static_cast<unsigned long long>(quiet_refused), static_cast<double>(quiet_us) / 1000.0,
				            paging_sum[0] / n, paging_sum[0] / n / 256.0, paging_sum[1] / n, paging_sum[1] / n / 256.0,
				            paging_sum[2] / n);
				paging_sum[0] = paging_sum[1] = paging_sum[2] = 0;
				paging_n                                     = 0;
			}
		}
	}
}

void Start() {
	static std::once_flag once;
	std::call_once(once, [] { std::thread(Worker).detach(); });
}
} // namespace StoreTrim
} // namespace FreedDiscard
#else
namespace FreedDiscard {
inline void NotePool(uint64_t, uint64_t) {}
inline void NoteRelease(uint64_t, uint64_t) {}
inline void Note(uint64_t, uint64_t) {}
inline void BeforeUnmap(uint64_t, uint64_t) {}
inline void Stats(uint64_t*) {}
namespace FreedRelease { // (ram6)
inline void BeforeUnmap(uint64_t, uint64_t) {}
inline void Line(char* out, size_t) {
	out[0] = 0;
}
} // namespace FreedRelease
} // namespace FreedDiscard
#endif

// (ram2) For the "[ram]" line (ram-guard.cpp, KYTY_RAM_LINE=1), any thread: counters and four guarded reads of the
// game's page allocator, no lock and no walk. out[0] freed-discard mode in use (0 off); [1] bytes of freed direct
// memory dropped (mode 1) or reset (mode 2) in all; [2] bytes queued (mode 1); [3] bytes left because another mapping
// covered them; [4] bytes Windows refused; [5] calls; [6] microseconds in the resets, in all (mode 2); [7] the longest
// reset since the last call, microseconds (mode 2); [8] 1 when [9] and [10] are read; [9] bytes of the game's page
// allocator's released blocks still mapped (its free-block cache: the emulator sees no unmap of them); [10] bytes of
// its pool not mapped (free in its bitmap).
void RamLineStats(uint64_t* out, size_t count);
void RamLineStats(uint64_t* out, size_t count) {
	uint64_t v[11] {};
	FreedDiscard::Stats(v);
	uint64_t cached = 0, unmapped = 0;
	if (Loader::DemonsSoulsMemMap::PageAllocatorFree(&cached, &unmapped)) {
		v[8]  = 1;
		v[9]  = cached;
		v[10] = unmapped;
	}
	for (size_t i = 0; i < count && i < 11; i++) out[i] = v[i];
}

// (ram6) The [ram] line's KYTY_DS_FREED_RELEASE part (ram-guard.cpp, its thread), MiB; "" when off.
void RamLineFreedRelease(char* out, size_t size);
void RamLineFreedRelease(char* out, size_t size) {
	if (out == nullptr || size == 0) return;
	FreedDiscard::FreedRelease::Line(out, size);
}

// (ram16, KYTY_RAM16) For the "[ram16]" line (ram-guard.cpp, its thread), any thread, no lock: the bytes
// KYTY_DS_FREED_RELEASE=1 gave back to Windows that the game has not taken back nor touched since (its "RAM ahorrada
// ahora", at its last pass); 0 when it is off, only measuring (=2) or not on Windows.
uint64_t Ram16FreedReleasedNow();
uint64_t Ram16FreedReleasedNow() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (FreedDiscard::FreedRelease::g_active.load(std::memory_order_relaxed) != 1) return 0;
	return FreedDiscard::FreedRelease::Stat().now_released.load(std::memory_order_relaxed);
#else
	return 0;
#endif
}

void Initialize() {
	g_flexible_memory_size_frozen = true;
	VirtualMemory::Init();
	g_guest_address_space = std::make_unique<GuestAddressSpace>(PhysicalMemory::TotalSize());
	LocalPlatform::SetTrimRunner([](void (*call)()) {
		if (g_guest_address_space != nullptr) g_guest_address_space->RunQuiet(call);
		else call();
	});
	g_physical_memory     = std::make_unique<PhysicalMemory>();
	g_flexible_memory     = std::make_unique<FlexibleMemory>();
	g_pooled_memory       = std::make_unique<PooledMemory>();
	g_virtual_ranges      = std::make_unique<VirtualRanges>();
	EXIT_IF(!g_guest_address_space->SelfTest());
	EXIT_IF(!SelfTestSub64SharedPlaceholderAlias());
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	RamGuard::RegisterBudgetFlush("released guest memory", RamGuard::FlushGuestFreed, RamFlushGuest::FlushFreed);
	RamGuard::RegisterBudgetFlush("guest store alias", RamGuard::FlushTrim, RamFlushGuest::FlushTrim);
	FreedDiscard::StoreTrim::Start(); // (ram9: KYTY_STORE_TRIM / KYTY_STORE_HOTSET, read in its thread; nothing without)
#endif
}

void Shutdown() {
	g_pooled_memory.reset();
	g_flexible_memory.reset();
	g_physical_memory.reset();
	g_virtual_ranges.reset();
	g_guest_address_space.reset();
}

struct AlignedPos {
	uint64_t value = 0;
	bool     valid = false;
};

static constexpr AlignedPos GetAlignedPos(uint64_t pos, size_t alignment) {
	if (alignment == 0) {
		return {pos, true};
	}

	const auto remainder = pos % alignment;
	const auto increment = (remainder != 0 ? alignment - remainder : 0);
	if (increment > UINT64_MAX - pos) {
		return {};
	}

	return {pos + increment, true};
}

static_assert(!GetAlignedPos(UINT64_MAX - 1, 4).valid);

void RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func) {
	EXIT_IF(g_alloc_callback != nullptr || g_free_callback != nullptr);
	EXIT_IF(alloc_func == nullptr || free_func == nullptr);

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	g_alloc_callback = alloc_func;
	g_free_callback  = free_func;

	g_physical_memory->GetMutex().Lock();
	for (const auto& [mapped_vaddr, b]: g_physical_memory->GetMappings()) {
		if (b.map_vaddr != 0 && b.map_size != 0) {
			g_alloc_callback(b.map_vaddr, b.map_size);
		}
	}
	g_physical_memory->GetMutex().Unlock();

	g_flexible_memory->GetMutex().Lock();
	for (const auto& b: g_flexible_memory->GetBlocks()) {
		g_alloc_callback(b.map_vaddr, b.map_size);
	}
	g_flexible_memory->GetMutex().Unlock();

	for (const auto& mapping: g_pooled_memory->GetMappings()) {
		g_alloc_callback(mapping.vaddr, mapping.size);
	}
}

void SetFlexibleMemorySize(uint64_t size) {
	constexpr uint64_t GuestPageSize = 0x4000;
	EXIT_IF(g_flexible_memory_size_frozen || g_guest_address_space != nullptr);
	EXIT_IF(size == 0 || (size & (GuestPageSize - 1u)) != 0 || size >= PhysicalMemory::TotalSize());
	g_flexible_memory_size = size;
	LOGF("\t flexible memory size = 0x%016" PRIx64 " (%" PRIu64 " MiB)\n", size,
	     size / (1024ull * 1024ull));
}

bool PhysicalMemory::Alloc(uint64_t search_start, uint64_t search_end, size_t len, size_t alignment,
                           uint64_t* phys_addr_out, int memory_type, bool pool_expansion) {
	if (phys_addr_out == nullptr) {
		return false;
	}

	Common::LockGuard lock(m_mutex);

	search_end = std::min<uint64_t>(search_end, Size());
	if (search_start >= search_end) {
		return false;
	}

	auto range = m_free.upper_bound(search_start);
	if (range != m_free.begin()) {
		range--;
	}
	for (; range != m_free.end() && range->first < search_end; ++range) {
		const auto range_end   = std::min<uint64_t>(range->first + range->second, search_end);
		const auto lower_bound = std::max(range->first, search_start);
		const auto aligned     = GetAlignedPos(lower_bound, alignment);
		const auto free_pos    = aligned.value;
		if (!aligned.valid || free_pos < lower_bound || free_pos > range_end ||
		    len > range_end - free_pos) {
			continue;
		}

		AllocatedBlock b {};
		b.size           = len;
		b.start_addr     = free_pos;
		b.gpu_mode       = GpuAccessMode::NoAccess;
		b.map_size       = 0;
		b.map_vaddr      = 0;
		b.prot           = 0;
		b.mode           = VirtualMemory::Mode::NoAccess;
		b.memory_type    = memory_type;
		b.pool_expansion = pool_expansion;

		ConsumeFreeRange(range, free_pos, len);
		EXIT_IF(!m_physical.emplace(b.start_addr, b).second);

		*phys_addr_out = free_pos;
		return true;
	}

	return false;
}

bool PhysicalMemory::Available(uint64_t search_start, uint64_t search_end, size_t alignment,
                               uint64_t* phys_addr_out, uint64_t* size_out) {
	if (phys_addr_out == nullptr || size_out == nullptr) {
		return false;
	}

	Common::LockGuard lock(m_mutex);

	search_end = std::min<uint64_t>(search_end, Size());
	if (search_start >= search_end) {
		return false;
	}

	uint64_t best_addr = 0;
	uint64_t best_size = 0;

	for (const auto& [range_start, range_size]: m_free) {
		if (range_start >= search_end) {
			break;
		}
		const auto range_end   = std::min<uint64_t>(range_start + range_size, search_end);
		const auto lower_bound = std::max(range_start, search_start);
		const auto aligned     = GetAlignedPos(lower_bound, alignment);
		const auto free_pos    = aligned.value;
		if (aligned.valid && free_pos >= lower_bound && free_pos < range_end &&
		    range_end - free_pos > best_size) {
			best_addr = free_pos;
			best_size = range_end - free_pos;
		}
	}

	if (best_size == 0) {
		return false;
	}

	*phys_addr_out = best_addr;
	*size_out      = best_size;
	return true;
}

void PhysicalMemory::ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
                                      uint64_t size) {
	GuestStoreUsage(static_cast<int64_t>(size));
	const auto range_start = range->first;
	const auto range_end   = range->first + range->second;
	m_free.erase(range);
	if (range_start < start) {
		m_free.emplace(range_start, start - range_start);
	}
	if (start + size < range_end) {
		m_free.emplace(start + size, range_end - start - size);
	}
}

void PhysicalMemory::AddFreeRange(uint64_t start, uint64_t size) {
	GuestStoreUsage(-static_cast<int64_t>(size));
	auto end  = start + size;
	auto next = m_free.lower_bound(start);
	if (next != m_free.begin()) {
		auto previous = std::prev(next);
		if (previous->first + previous->second >= start) {
			start = previous->first;
			end   = std::max(end, previous->first + previous->second);
			next  = m_free.erase(previous);
		}
	}
	while (next != m_free.end() && next->first <= end) {
		end  = std::max(end, next->first + next->second);
		next = m_free.erase(next);
	}
	m_free.emplace(start, end - start);
}

bool PhysicalMemory::Release(uint64_t start, size_t len, uint64_t* vaddr, uint64_t* size,
                             GpuAccessMode* gpu_mode) {
	EXIT_IF(vaddr == nullptr);
	EXIT_IF(size == nullptr);
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	auto next = m_physical.upper_bound(start);
	if (next == m_physical.begin()) {
		return false;
	}
	auto  it = std::prev(next);
	auto& b  = it->second;
	if (b.pool_expansion || start < b.start_addr || start >= b.start_addr + b.size ||
	    len > b.start_addr + b.size - start) {
		return false;
	}

	if (start == b.start_addr && len == b.size) {
		*vaddr    = b.map_vaddr;
		*size     = b.map_size;
		*gpu_mode = b.gpu_mode;

		m_physical.erase(it);
		AddFreeRange(start, len);
		return true;
	}
	if (start > b.start_addr && start + len < b.start_addr + b.size) {
		auto old_start = b.start_addr;
		auto old_end   = b.start_addr + b.size;

		*vaddr    = (b.map_vaddr != 0 ? b.map_vaddr + (start - old_start) : 0);
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		AllocatedBlock right = b;
		right.start_addr     = start + len;
		right.size           = old_end - right.start_addr;
		if (right.map_vaddr != 0) {
			right.map_vaddr += right.start_addr - old_start;
			right.map_size = right.size;
		}

		b.size = start - old_start;
		if (b.map_vaddr != 0) {
			b.map_size = b.size;
		}

		m_physical.emplace(right.start_addr, right);
		AddFreeRange(start, len);
		return true;
	}
	if (start == b.start_addr && len < b.size) {
		*vaddr    = b.map_vaddr;
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		AllocatedBlock remaining = b;
		m_physical.erase(it);
		remaining.start_addr += len;
		remaining.size -= len;
		if (remaining.map_vaddr != 0) {
			remaining.map_vaddr += len;
			remaining.map_size -= len;
		}
		m_physical.emplace(remaining.start_addr, remaining);
		AddFreeRange(start, len);
		return true;
	}
	if (start > b.start_addr && start + len == b.start_addr + b.size) {
		*vaddr    = (b.map_vaddr != 0 ? b.map_vaddr + (start - b.start_addr) : 0);
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		b.size = start - b.start_addr;
		if (b.map_vaddr != 0) {
			b.map_size = b.size;
		}
		AddFreeRange(start, len);
		return true;
	}

	return false;
}

bool PhysicalMemory::Map(uint64_t vaddr, uint64_t phys_addr, size_t len, int prot,
                         VirtualMemory::Mode mode, GpuAccessMode gpu_mode) {
	Common::LockGuard lock(m_mutex);

	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	auto current = phys_addr;
	auto next    = m_physical.upper_bound(current);
	if (next == m_physical.begin()) {
		return false;
	}
	auto first = std::prev(next);
	while (current < phys_addr + len) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.pool_expansion || current < block->second.start_addr ||
		    current >= block_end) {
			return false;
		}
		current = std::min<uint64_t>(phys_addr + len, block_end);
		if (current < phys_addr + len) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				return false;
			}
		}
	}

	AllocatedBlock mapping = first->second;
	mapping.start_addr     = phys_addr;
	mapping.size           = len;
	mapping.map_vaddr      = vaddr;
	mapping.map_size       = len;
	mapping.host_vaddr     = vaddr;
	mapping.host_size      = len;
	mapping.prot           = prot;
	mapping.mode           = mode;
	mapping.gpu_mode       = gpu_mode;
	m_mappings.insert_or_assign(vaddr, mapping); // (a stale record at that address is replaced)

	return true;
}

bool PhysicalMemory::CanMapDirect(uint64_t phys_addr, size_t len) {
	Common::LockGuard lock(m_mutex);

	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	const auto end     = phys_addr + len;
	auto       current = phys_addr;
	while (current < end) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.pool_expansion || current < block->second.start_addr ||
		    current >= block_end) {
			return false;
		}
		current = std::min(end, block_end);
		if (current < end) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				return false;
			}
		}
	}
	return true;
}

bool PhysicalMemory::ReleasePoolExpansion(uint64_t phys_addr, size_t len) {
	Common::LockGuard lock(m_mutex);
	const auto        it = m_physical.find(phys_addr);
	if (it == m_physical.end() || !it->second.pool_expansion || it->second.size != len) {
		return false;
	}
	m_physical.erase(it);
	AddFreeRange(phys_addr, len);
	return true;
}

bool PhysicalMemory::GetAllocatedSpan(uint64_t phys_addr, size_t len,
                                      std::vector<AllocatedBlock>* blocks) {
	EXIT_IF(blocks == nullptr);
	blocks->clear();
	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	Common::LockGuard lock(m_mutex);
	const auto        end     = phys_addr + len;
	auto              current = phys_addr;
	while (current < end) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			blocks->clear();
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.pool_expansion || current < block->second.start_addr ||
		    current >= block_end) {
			blocks->clear();
			return false;
		}

		auto part       = block->second;
		part.start_addr = current;
		part.size       = std::min(end, block_end) - current;
		blocks->push_back(part);
		current += part.size;
		if (current < end) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				blocks->clear();
				return false;
			}
		}
	}
	return true;
}

bool PhysicalMemory::Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
                           uint64_t* host_vaddr_to_release) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	if (g_test_physical_memory_unmaps_before_failure == 0) {
		g_test_physical_memory_unmaps_before_failure = UINT32_MAX;
		return false;
	}
	if (g_test_physical_memory_unmaps_before_failure != UINT32_MAX) {
		g_test_physical_memory_unmaps_before_failure--;
	}
#endif

	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	if (host_vaddr_to_release != nullptr) {
		*host_vaddr_to_release = 0;
	}

	auto set_host_release_if_last = [this, host_vaddr_to_release](uint64_t host_vaddr,
	                                                              uint64_t host_size) {
		if (host_vaddr_to_release == nullptr || host_vaddr == 0 || host_size == 0) {
			return;
		}
		const bool still_mapped = std::any_of(m_mappings.begin(), m_mappings.end(), [host_vaddr, host_size](const auto& entry) {
			return entry.second.host_vaddr == host_vaddr && entry.second.host_size == host_size;
		});
		if (!still_mapped) {
			*host_vaddr_to_release = host_vaddr;
		}
	};

	// The one mapping containing `vaddr` (they are disjoint).
	auto it = m_mappings.upper_bound(vaddr);
	if (it == m_mappings.begin()) {
		return false;
	}
	--it;
	auto& b = it->second;
	if (b.map_vaddr == vaddr && b.map_size == size) {
		*gpu_mode             = b.gpu_mode;
		const auto host_vaddr = b.host_vaddr;
		const auto host_size  = b.host_size;

		m_mappings.erase(it);
		set_host_release_if_last(host_vaddr, host_size);

		return true;
	}
	if (vaddr > b.map_vaddr && vaddr + size < b.map_vaddr + b.map_size) {
		*gpu_mode = b.gpu_mode;

		AllocatedBlock right = b;
		right.start_addr += (vaddr + size) - b.map_vaddr;
		right.size      = b.map_vaddr + b.map_size - (vaddr + size);
		right.map_size  = right.size;
		right.map_vaddr = vaddr + size;

		b.size     = vaddr - b.map_vaddr;
		b.map_size = b.size;
		m_mappings.insert_or_assign(right.map_vaddr, right);
		return true;
	}
	if (vaddr == b.map_vaddr && size < b.map_size) {
		*gpu_mode = b.gpu_mode;

		AllocatedBlock rest = b;
		rest.start_addr += size;
		rest.size -= size;
		rest.map_vaddr += size;
		rest.map_size -= size;
		m_mappings.erase(it);
		m_mappings.insert_or_assign(rest.map_vaddr, rest);
		return true;
	}
	if (vaddr > b.map_vaddr && vaddr + size == b.map_vaddr + b.map_size) {
		*gpu_mode = b.gpu_mode;

		b.size     = vaddr - b.map_vaddr;
		b.map_size = b.size;
		return true;
	}

	return false;
}

void PhysicalMemory::ProtectMapping(uint64_t vaddr, uint64_t size, int prot,
                                    VirtualMemory::Mode mode, GpuAccessMode gpu_mode) {
	Common::LockGuard lock(m_mutex);
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return;
	}

	const auto                  end = vaddr + size;
	std::vector<AllocatedBlock> updated;
	// The mappings the range overlaps (disjoint: from the one containing `vaddr`), replaced by their pieces.
	auto it = m_mappings.upper_bound(vaddr);
	if (it != m_mappings.begin() && std::prev(it)->second.map_vaddr + std::prev(it)->second.map_size > vaddr) {
		--it;
	}
	while (it != m_mappings.end() && it->second.map_vaddr < end) {
		const auto block     = it->second;
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			++it;
			continue;
		}
		it                       = m_mappings.erase(it);
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.size     = overlap_start - block.map_vaddr;
			left.map_size = left.size;
			updated.push_back(left);
		}
		auto middle       = block;
		middle.start_addr = block.start_addr + overlap_start - block.map_vaddr;
		middle.size       = overlap_end - overlap_start;
		middle.map_vaddr  = overlap_start;
		middle.map_size   = middle.size;
		middle.prot       = prot;
		middle.mode       = mode;
		middle.gpu_mode   = gpu_mode;
		updated.push_back(middle);
		if (overlap_end < block_end) {
			auto right       = block;
			right.start_addr = block.start_addr + overlap_end - block.map_vaddr;
			right.size       = block_end - overlap_end;
			right.map_vaddr  = overlap_end;
			right.map_size   = right.size;
			updated.push_back(right);
		}
	}
	for (const auto& piece: updated) {
		m_mappings.insert_or_assign(piece.map_vaddr, piece);
	}
}

bool PhysicalMemory::Find(uint64_t phys_addr, bool next, AllocatedBlock* out) {
	EXIT_IF(out == nullptr);

	Common::LockGuard lock(m_mutex);

	auto following = m_physical.upper_bound(phys_addr);
	if (following != m_physical.begin()) {
		const auto& block = std::prev(following)->second;
		if (phys_addr < block.start_addr + block.size) {
			*out = block;
			return true;
		}
	}

	if (next && following != m_physical.end()) {
		*out = following->second;
		return true;
	}

	return false;
}

std::vector<PhysicalMemory::AllocatedBlock> PhysicalMemory::FindMappings(uint64_t phys_addr,
                                                                         size_t   len) {
	Common::LockGuard lock(m_mutex);

	std::vector<AllocatedBlock> mappings;
	for (const auto& [mapped_vaddr, m]: m_mappings) {
		if (!VirtualRangesOverlap(phys_addr, len, m.start_addr, m.size)) {
			continue;
		}

		const uint64_t overlap_start = std::max<uint64_t>(phys_addr, m.start_addr);
		const uint64_t overlap_end   = std::min<uint64_t>(phys_addr + len, m.start_addr + m.size);
		AllocatedBlock part          = m;
		part.start_addr              = overlap_start;
		part.size                    = overlap_end - overlap_start;
		part.map_vaddr += overlap_start - m.start_addr;
		part.map_size = part.size;
		mappings.push_back(part);
	}

	std::sort(mappings.begin(), mappings.end(),
	          [](const auto& a, const auto& b) { return a.map_vaddr < b.map_vaddr; });
	return mappings;
}

bool PhysicalMemory::Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
                          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	Common::LockGuard lock(m_mutex);

	auto it = m_mappings.upper_bound(vaddr);
	if (it == m_mappings.begin()) {
		return false;
	}
	const auto& b = std::prev(it)->second;
	if (vaddr < b.map_vaddr || vaddr >= b.map_vaddr + b.map_size) {
		return false;
	}
	if (base_addr != nullptr) {
		*base_addr = b.map_vaddr;
	}
	if (len != nullptr) {
		*len = b.map_size;
	}
	if (prot != nullptr) {
		*prot = b.prot;
	}
	if (mode != nullptr) {
		*mode = b.mode;
	}
	if (gpu_mode != nullptr) {
		*gpu_mode = b.gpu_mode;
	}
	return true;
}

void PhysicalMemory::SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name) {
	Common::LockGuard lock(m_mutex);

	ForEachMappingOverlapping(vaddr, len, [name](AllocatedBlock& b) { CopyVirtualRangeName(b.name, name); });
}

bool FlexibleMemory::Map(uint64_t vaddr, size_t len, int prot, VirtualMemory::Mode mode,
                         GpuAccessMode gpu_mode, const char* name) {
	Common::LockGuard lock(m_mutex);

	const auto available = (Size() >= m_allocated_total ? Size() - m_allocated_total : 0);
	if (len == 0 || len > available) {
		LOGF_COLOR(Log::Color::Red,
		           "\t flexible memory exhausted: configured = 0x%016" PRIx64
		           ", allocated = 0x%016" PRIx64 ", available = 0x%016" PRIx64
		           ", requested = 0x%016" PRIx64 "\n",
		           Size(), m_allocated_total, available, static_cast<uint64_t>(len));
		return false;
	}

	std::vector<AllocatedBlock> blocks;
	auto                        current   = vaddr;
	auto                        remaining = static_cast<uint64_t>(len);
	for (const auto& [backing_offset, free_size]: m_free) {
		if (remaining == 0) {
			break;
		}
		const auto     chunk = std::min(remaining, free_size);
		AllocatedBlock block {};
		block.map_vaddr      = current;
		block.map_size       = chunk;
		block.backing_offset = backing_offset;
		block.host_vaddr     = vaddr;
		block.host_size      = len;
		block.prot           = prot;
		block.mode           = mode;
		block.gpu_mode       = gpu_mode;
		CopyVirtualRangeName(block.name, name);
		blocks.push_back(block);
		current += chunk;
		remaining -= chunk;
	}
	if (remaining != 0) {
		return false;
	}

	std::vector<AllocatedBlock> mapped;
	for (const auto& block: blocks) {
		// Only backing an earlier mapping had needs zeroing: past the high-water mark of the blocks handed out, the
		// section's pages were never written and read zero. Demon's Souls maps 532 MiB at start-up: zeroing it took
		// 265 ms and made all of it resident at once.
		const uint64_t used = m_backing_high_water > block.backing_offset
		                          ? std::min(block.map_size, m_backing_high_water - block.backing_offset)
		                          : 0;
		if (used != 0 && !g_guest_address_space->ZeroBacking(block.backing_offset, used)) {
			for (auto it = mapped.rbegin(); it != mapped.rend(); ++it) {
				EXIT_IF(!g_guest_address_space->UnmapBacking(it->map_vaddr, it->map_size));
			}
			return false;
		}
		m_backing_high_water = std::max(m_backing_high_water, block.backing_offset + block.map_size);
		if (!g_guest_address_space->MapBacking(block.map_vaddr, block.map_size,
		                                       block.backing_offset, block.mode)) {
			for (auto it = mapped.rbegin(); it != mapped.rend(); ++it) {
				EXIT_IF(!g_guest_address_space->UnmapBacking(it->map_vaddr, it->map_size));
			}
			return false;
		}
		mapped.push_back(block);
	}

	for (const auto& block: blocks) {
		auto next = m_free.upper_bound(block.backing_offset);
		EXIT_IF(next == m_free.begin());
		auto range = std::prev(next);
		EXIT_IF(block.backing_offset < range->first ||
		        block.map_size > range->first + range->second - block.backing_offset);
		ConsumeFreeRange(range, block.backing_offset, block.map_size);
		m_allocated.push_back(block);
	}
	std::sort(m_allocated.begin(), m_allocated.end(),
	          [](const auto& left, const auto& right) { return left.map_vaddr < right.map_vaddr; });
	m_allocated_total += len;

	return true;
}

bool FlexibleMemory::Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
                           uint64_t* host_vaddr_to_release) {
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	if (host_vaddr_to_release != nullptr) {
		*host_vaddr_to_release = 0;
	}
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	const auto end     = vaddr + size;
	auto       current = vaddr;
	bool       found   = false;
	for (const auto& block: m_allocated) {
		if (block.map_vaddr + block.map_size <= current) {
			continue;
		}
		if (block.map_vaddr > current || block.map_vaddr >= end) {
			break;
		}
		if (!found) {
			*gpu_mode = block.gpu_mode;
			found     = true;
		}
		current = std::min(end, block.map_vaddr + block.map_size);
		if (current == end) {
			break;
		}
	}
	if (!found || current != end || !g_guest_address_space->UnmapBacking(vaddr, size)) {
		return false;
	}

	std::vector<AllocatedBlock> remaining;
	remaining.reserve(m_allocated.size() + 1);
	uint64_t removed = 0;
	for (const auto& block: m_allocated) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			remaining.push_back(block);
			continue;
		}
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		const auto overlap_size  = overlap_end - overlap_start;
		AddFreeRange(block.backing_offset + overlap_start - block.map_vaddr, overlap_size);
		removed += overlap_size;

		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.map_size = overlap_start - block.map_vaddr;
			remaining.push_back(left);
		}
		if (overlap_end < block_end) {
			auto right           = block;
			right.map_vaddr      = overlap_end;
			right.map_size       = block_end - overlap_end;
			right.backing_offset = block.backing_offset + overlap_end - block.map_vaddr;
			remaining.push_back(right);
		}
	}
	EXIT_IF(removed != size || removed > m_allocated_total);
	m_allocated = std::move(remaining);
	m_allocated_total -= removed;
	return true;
}

void FlexibleMemory::ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
                                      uint64_t size) {
	GuestStoreUsage(static_cast<int64_t>(size));
	const auto range_start = range->first;
	const auto range_end   = range->first + range->second;
	m_free.erase(range);
	if (range_start < start) {
		m_free.emplace(range_start, start - range_start);
	}
	if (start + size < range_end) {
		m_free.emplace(start + size, range_end - start - size);
	}
}

void FlexibleMemory::AddFreeRange(uint64_t start, uint64_t size) {
	GuestStoreUsage(-static_cast<int64_t>(size));
	auto end  = start + size;
	auto next = m_free.lower_bound(start);
	if (next != m_free.begin()) {
		auto previous = std::prev(next);
		if (previous->first + previous->second >= start) {
			start = previous->first;
			end   = std::max(end, previous->first + previous->second);
			next  = m_free.erase(previous);
		}
	}
	while (next != m_free.end() && next->first <= end) {
		end  = std::max(end, next->first + next->second);
		next = m_free.erase(next);
	}
	m_free.emplace(start, end - start);
}

bool FlexibleMemory::Snapshot(uint64_t vaddr, uint64_t size, std::vector<AllocatedBlock>* blocks) {
	EXIT_IF(blocks == nullptr);
	Common::LockGuard lock(m_mutex);
	blocks->clear();
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	const auto end     = vaddr + size;
	auto       current = vaddr;
	for (const auto& block: m_allocated) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (block_end <= current) {
			continue;
		}
		if (block.map_vaddr > current || block.map_vaddr >= end) {
			break;
		}
		const auto part_end = std::min(end, block_end);
		auto       part     = block;
		part.map_vaddr      = current;
		part.map_size       = part_end - current;
		part.backing_offset += current - block.map_vaddr;
		blocks->push_back(part);
		current = part_end;
		if (current == end) {
			return true;
		}
	}
	blocks->clear();
	return false;
}

bool FlexibleMemory::Restore(const std::vector<AllocatedBlock>& blocks) {
	Common::LockGuard lock(m_mutex);
	if (blocks.empty()) {
		return false;
	}

	for (const auto& block: blocks) {
		auto next = m_free.upper_bound(block.backing_offset);
		if (next == m_free.begin()) {
			return false;
		}
		const auto range = std::prev(next);
		if (block.backing_offset < range->first ||
		    block.map_size > range->first + range->second - block.backing_offset) {
			return false;
		}
	}

	std::vector<AllocatedBlock> mapped;
	for (const auto& block: blocks) {
		if (!g_guest_address_space->MapBacking(block.map_vaddr, block.map_size,
		                                       block.backing_offset, block.mode)) {
			for (auto it = mapped.rbegin(); it != mapped.rend(); ++it) {
				EXIT_IF(!g_guest_address_space->UnmapBacking(it->map_vaddr, it->map_size));
			}
			return false;
		}
		mapped.push_back(block);
	}

	for (const auto& block: blocks) {
		auto next = m_free.upper_bound(block.backing_offset);
		EXIT_IF(next == m_free.begin());
		auto range = std::prev(next);
		ConsumeFreeRange(range, block.backing_offset, block.map_size);
		m_allocated.push_back(block);
		m_allocated_total += block.map_size;
	}
	std::sort(m_allocated.begin(), m_allocated.end(),
	          [](const auto& left, const auto& right) { return left.map_vaddr < right.map_vaddr; });
	return true;
}

void FlexibleMemory::Protect(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
                             GpuAccessMode gpu_mode) {
	Common::LockGuard lock(m_mutex);
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return;
	}

	const auto                  end = vaddr + size;
	std::vector<AllocatedBlock> updated;
	updated.reserve(m_allocated.size() + 2);
	for (const auto& block: m_allocated) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			updated.push_back(block);
			continue;
		}
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.map_size = overlap_start - block.map_vaddr;
			updated.push_back(left);
		}
		auto middle           = block;
		middle.map_vaddr      = overlap_start;
		middle.map_size       = overlap_end - overlap_start;
		middle.backing_offset = block.backing_offset + overlap_start - block.map_vaddr;
		middle.prot           = prot;
		middle.mode           = mode;
		middle.gpu_mode       = gpu_mode;
		updated.push_back(middle);
		if (overlap_end < block_end) {
			auto right           = block;
			right.map_vaddr      = overlap_end;
			right.map_size       = block_end - overlap_end;
			right.backing_offset = block.backing_offset + overlap_end - block.map_vaddr;
			updated.push_back(right);
		}
	}
	m_allocated = std::move(updated);
}

bool FlexibleMemory::Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
                          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	Common::LockGuard lock(m_mutex);

	return std::any_of(m_allocated.begin(), m_allocated.end(),
	                   [vaddr, base_addr, len, prot, mode, gpu_mode](auto& b) {
		                   if (vaddr >= b.map_vaddr && vaddr < b.map_vaddr + b.map_size) {
			                   if (base_addr != nullptr) {
				                   *base_addr = b.map_vaddr;
			                   }
			                   if (len != nullptr) {
				                   *len = b.map_size;
			                   }
			                   if (prot != nullptr) {
				                   *prot = b.prot;
			                   }
			                   if (mode != nullptr) {
				                   *mode = b.mode;
			                   }
			                   if (gpu_mode != nullptr) {
				                   *gpu_mode = b.gpu_mode;
			                   }

			                   return true;
		                   }
		                   return false;
	                   });
}

void FlexibleMemory::SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name) {
	Common::LockGuard lock(m_mutex);

	for (auto& b: m_allocated) {
		if (VirtualRangesOverlap(vaddr, len, b.map_vaddr, b.map_size)) {
			CopyVirtualRangeName(b.name, name);
		}
	}
}

void PhysicalMemory::SetVirtualRangeMemoryType(uint64_t vaddr, uint64_t len, int memory_type) {
	Common::LockGuard lock(m_mutex);

	ForEachMappingOverlapping(vaddr, len, [memory_type](AllocatedBlock& b) { b.memory_type = memory_type; });
}

uint64_t FlexibleMemory::Available() {
	Common::LockGuard lock(m_mutex);

	return (Size() >= m_allocated_total ? Size() - m_allocated_total : 0);
}

void PooledMemory::AddFreeUnlocked(uint64_t start, uint64_t size) {
	if (size == 0) {
		return;
	}

	m_free.push_back({start, size});
	std::sort(m_free.begin(), m_free.end(),
	          [](const auto& left, const auto& right) { return left.start < right.start; });

	std::vector<PhysicalRange> merged;
	for (const auto& range: m_free) {
		if (!merged.empty() && range.start <= merged.back().start + merged.back().size) {
			const auto end =
			    std::max(merged.back().start + merged.back().size, range.start + range.size);
			merged.back().size = end - merged.back().start;
		} else {
			merged.push_back(range);
		}
	}
	m_free = std::move(merged);
}

void PooledMemory::Expand(uint64_t phys_addr, uint64_t size) {
	Common::LockGuard lock(m_mutex);
	m_expansions.push_back({phys_addr, size});
	AddFreeUnlocked(phys_addr, size);
}

bool PooledMemory::ReleaseExpansion(uint64_t phys_addr, uint64_t size) {
	Common::LockGuard lock(m_mutex);
	const auto        expansion = std::find_if(m_expansions.begin(), m_expansions.end(),
	                                           [phys_addr, size](const auto& range) {
		                                    return range.start == phys_addr && range.size == size;
	                                           });
	if (expansion == m_expansions.end()) {
		return false;
	}
	if (std::any_of(m_mappings.begin(), m_mappings.end(), [phys_addr, size](const auto& mapping) {
		    return VirtualRangesOverlap(phys_addr, size, mapping.phys_addr, mapping.size);
	    })) {
		return false;
	}

	const auto end = phys_addr + size;
	const auto free_range =
	    std::find_if(m_free.begin(), m_free.end(), [phys_addr, end](const auto& r) {
		    return phys_addr >= r.start && end <= r.start + r.size;
	    });
	if (free_range == m_free.end()) {
		return false;
	}

	const auto old        = *free_range;
	const auto old_end    = old.start + old.size;
	const auto left_size  = phys_addr - old.start;
	const auto right_size = old_end - end;
	m_free.erase(free_range);
	if (left_size != 0) {
		m_free.push_back({old.start, left_size});
	}
	if (right_size != 0) {
		m_free.push_back({end, right_size});
	}
	std::sort(m_free.begin(), m_free.end(),
	          [](const auto& left, const auto& right) { return left.start < right.start; });
	m_expansions.erase(expansion);
	return true;
}

bool PooledMemory::Allocate(uint64_t vaddr, uint64_t size, GpuAccessMode gpu_mode,
                            std::vector<Mapping>* mappings) {
	EXIT_IF(mappings == nullptr);
	mappings->clear();
	if (vaddr == 0 || size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	Common::LockGuard lock(m_mutex);
	auto              free      = m_free;
	auto              current   = vaddr;
	auto              remaining = size;
	for (auto& range: free) {
		if (remaining == 0 || range.size == 0) {
			continue;
		}
		const auto part_size = std::min(range.size, remaining);
		mappings->push_back({current, part_size, range.start, gpu_mode});
		range.start += part_size;
		range.size -= part_size;
		current += part_size;
		remaining -= part_size;
	}
	if (remaining != 0) {
		mappings->clear();
		return false;
	}

	free.erase(
	    std::remove_if(free.begin(), free.end(), [](const auto& range) { return range.size == 0; }),
	    free.end());
	m_free = std::move(free);
	m_mappings.insert(m_mappings.end(), mappings->begin(), mappings->end());
	return true;
}

bool PooledMemory::QueryUnlocked(uint64_t vaddr, uint64_t size,
                                 std::vector<Mapping>* mappings) const {
	EXIT_IF(mappings == nullptr);
	mappings->clear();
	if (vaddr == 0 || size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	const auto end     = vaddr + size;
	auto       current = vaddr;
	while (current < end) {
		const auto it =
		    std::find_if(m_mappings.begin(), m_mappings.end(), [current](const auto& m) {
			    return current >= m.vaddr && current < m.vaddr + m.size;
		    });
		if (it == m_mappings.end()) {
			mappings->clear();
			return false;
		}
		const auto part_size = std::min(end, it->vaddr + it->size) - current;
		mappings->push_back(
		    {current, part_size, it->phys_addr + (current - it->vaddr), it->gpu_mode});
		current += part_size;
	}
	return true;
}

bool PooledMemory::Query(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings) {
	Common::LockGuard lock(m_mutex);
	return QueryUnlocked(vaddr, size, mappings);
}

bool PooledMemory::Release(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode) {
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard    lock(m_mutex);
	std::vector<Mapping> released;
	if (!QueryUnlocked(vaddr, size, &released)) {
		return false;
	}

	*gpu_mode                = released.front().gpu_mode;
	const auto           end = vaddr + size;
	std::vector<Mapping> kept;
	for (const auto& mapping: m_mappings) {
		const auto mapping_end = mapping.vaddr + mapping.size;
		const auto cut_start   = std::max(vaddr, mapping.vaddr);
		const auto cut_end     = std::min(end, mapping_end);
		if (cut_start >= cut_end) {
			kept.push_back(mapping);
			continue;
		}
		if (mapping.vaddr < cut_start) {
			auto left = mapping;
			left.size = cut_start - mapping.vaddr;
			kept.push_back(left);
		}
		if (cut_end < mapping_end) {
			auto right      = mapping;
			right.vaddr     = cut_end;
			right.size      = mapping_end - cut_end;
			right.phys_addr = mapping.phys_addr + (cut_end - mapping.vaddr);
			kept.push_back(right);
		}
	}
	m_mappings = std::move(kept);
	for (const auto& mapping: released) {
		AddFreeUnlocked(mapping.phys_addr, mapping.size);
	}
	return true;
}

uint64_t PooledMemory::Available() {
	Common::LockGuard lock(m_mutex);
	uint64_t          available = 0;
	for (const auto& range: m_free) {
		available += range.size;
	}
	return available;
}

std::vector<PooledMemory::Mapping> PooledMemory::GetMappings() {
	Common::LockGuard lock(m_mutex);
	return m_mappings;
}

static bool UnmapPooledBackingTransactional(const std::vector<PooledMemory::Mapping>& mappings,
                                            VirtualMemory::Mode                       mode) {
	std::vector<PooledMemory::Mapping> removed;
	for (const auto& mapping: mappings) {
		if (!g_guest_address_space->UnmapBacking(mapping.vaddr, mapping.size)) {
			for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
				auto       failure_reason = GuestBackingStore::FailureReason::None;
				const bool restored       = g_guest_address_space->MapBacking(
				    it->vaddr, it->size, it->phys_addr, mode, &failure_reason);
				if (!restored) {
					EXIT("pooled-memory unmap rollback failed: %s\n",
					     GuestBackingStore::GetFailureReasonName(failure_reason));
				}
			}
			return false;
		}
		removed.push_back(mapping);
	}
	return true;
}

int32_t KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot,
                                                   int flags, const char* name) {
	KYTY_MAPA_GRANDE_SCOPE("sceKernelMapNamedFlexibleMemory");
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	EXIT_NOT_IMPLEMENTED(addr_in_out == nullptr);

	constexpr size_t   PAGE_SIZE                = 0x4000;
	constexpr size_t   MAXIMUM_NAME_SIZE        = 32;
	constexpr uint64_t DEFAULT_PS5_BASE         = 0x200000000;
	constexpr uint32_t GUEST_MAP_FIXED          = 0x10;
	constexpr uint32_t GUEST_MAP_NO_OVERWRITE   = 0x80;
	constexpr uint32_t GUEST_MAP_DMEM_COMPAT    = 0x400;
	constexpr uint32_t GUEST_MAP_UNKNOWN_8000   = 0x8000;
	constexpr uint32_t GUEST_MAP_NO_COALESCE    = 0x400000;
	constexpr uint32_t GUEST_MAP_ALIGNMENT_MASK = 0xff000000;
	constexpr uint32_t SUPPORTED_MAP_BITS       = GUEST_MAP_FIXED | GUEST_MAP_NO_OVERWRITE |
	                                              GUEST_MAP_DMEM_COMPAT | GUEST_MAP_UNKNOWN_8000 |
	                                              GUEST_MAP_NO_COALESCE | GUEST_MAP_ALIGNMENT_MASK;

	if (len == 0 || (len & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if (len > g_flexible_memory->Available()) {
		return NoMemory(__func__, __LINE__);
	}

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= MAXIMUM_NAME_SIZE) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	const auto map_flags       = static_cast<uint32_t>(flags);
	const auto alignment_shift = (map_flags & GUEST_MAP_ALIGNMENT_MASK) >> 24u;
	if ((map_flags & ~SUPPORTED_MAP_BITS) != 0 ||
	    (alignment_shift != 0 && (alignment_shift < 14 || alignment_shift > 31))) {
		LOGF_COLOR(Log::Color::Red, "\t unsupported flags = 0x%08" PRIx32 "\n", map_flags);
		return KERNEL_ERROR_EINVAL;
	}
	const uint64_t map_alignment =
	    alignment_shift != 0 ? uint64_t {1} << alignment_shift : PAGE_SIZE;

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		EXIT("unknown prot: %d\n", prot);
	}

	auto                 in_addr              = reinterpret_cast<uint64_t>(*addr_in_out);
	uint64_t             out_addr             = 0;
	bool                 consumed_reservation = false;
	VirtualRanges::Range consumed_range {};
	std::unique_lock residency_lock(g_prt_residency_mutex, std::defer_lock);

	if ((flags & GUEST_MAP_FIXED) != 0) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1)) != 0 ||
		    (in_addr & (map_alignment - 1u)) != 0) {
			return KERNEL_ERROR_EINVAL;
		}
		if ((flags & GUEST_MAP_NO_OVERWRITE) != 0 && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return NoMemory(__func__, __LINE__);
		}
		std::vector<VirtualRanges::Range> reserved_ranges;
		if (g_virtual_ranges->QuerySpan(in_addr, len, &reserved_ranges) &&
		    std::all_of(reserved_ranges.begin(), reserved_ranges.end(), [](const auto& range) {
			    return range.type == VirtualRangeType::Reserved;
		    })) {
			UnmapGpuRange(in_addr, len);
			residency_lock.lock();
			consumed_range = reserved_ranges.front();
			if (g_virtual_ranges->ConsumeReservedSpan(in_addr, len)) {
				consumed_reservation = true;
				out_addr             = in_addr;
			}
		}
		if (!consumed_reservation) {
			if (residency_lock.owns_lock()) residency_lock.unlock();
			if (ReplaceFixedRangeWithReserved(in_addr, len)) {
				residency_lock.lock();
				if (g_virtual_ranges->ConsumeReservedSpan(in_addr, len, &consumed_range)) {
					consumed_reservation = true;
					out_addr             = in_addr;
				}
			}
		}
	} else {
		const auto search_addr = (in_addr != 0 ? in_addr : DEFAULT_PS5_BASE);
		out_addr               = FindGuestFreeRange(search_addr, len, map_alignment);
		if (out_addr != 0) {
			UnmapGpuRange(out_addr, len);
			residency_lock.lock();
		}
	}

	*addr_in_out = reinterpret_cast<void*>(out_addr);

	if (out_addr == 0) {
		if (consumed_reservation) {
			g_virtual_ranges->Add(in_addr, len, 0, 0, 0, VirtualRangeType::Reserved,
			                      consumed_range.name);
		}
		return NoMemory(__func__, __LINE__);
	}

	if (!g_flexible_memory->Map(out_addr, len, prot, mode, gpu_mode, name)) {
		LOGF_COLOR(Log::Color::Red, "\t [Fail]\n");
		if (consumed_reservation) {
			EXIT_IF(!g_virtual_ranges->Add(out_addr, len, 0, 0, 0, VirtualRangeType::Reserved,
			                               consumed_range.name));
		}
		return NoMemory(__func__, __LINE__);
	}

	if (!g_virtual_ranges->Add(out_addr, len, 0, prot, 0, VirtualRangeType::Flexible, name,
	                           (map_flags & GUEST_MAP_NO_COALESCE) != 0)) {
		GpuAccessMode rollback_gpu_mode = GpuAccessMode::NoAccess;
		EXIT_IF(!g_flexible_memory->Unmap(out_addr, len, &rollback_gpu_mode));
		if (consumed_reservation) {
			EXIT_IF(!g_virtual_ranges->Add(out_addr, len, 0, 0, 0, VirtualRangeType::Reserved,
			                               consumed_range.name));
		}
		return KERNEL_ERROR_EBUSY;
	}

	LOGF("\t in_addr  = 0x%016" PRIx64 "\n"
	     "\t out_addr = 0x%016" PRIx64 "\n"
	     "\t size     = %" PRIu64 "\n"
	     "\t mode     = %s\n"
	     "\t flags    = 0x%08" PRIx32 "\n"
	     "\t name     = %s\n"
	     "\t gpu_mode = %s\n",
	     in_addr, out_addr, len, Common::EnumName(mode).c_str(), static_cast<uint32_t>(flags), name,
	     Common::EnumName(gpu_mode).c_str());

	residency_lock.unlock();
	MapGpuRange(out_addr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(out_addr, len);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags) {
	KYTY_MAPA_GRANDE_SCOPE("sceKernelMapFlexibleMemory");
	struct AccessReport { ~AccessReport() { ReportGuestAccess(false); } } access_report;
	return KernelMapNamedFlexibleMemory(addr_in_out, len, prot, flags, "");
}

int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	const auto address = reinterpret_cast<uint64_t>(addr);

	LOGF("\t index = %d\n"
	     "\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     index, address, static_cast<uint64_t>(len));

	if (index < 0 || index > PRT_APERTURE_MAX_INDEX) {
		return KERNEL_ERROR_EINVAL;
	}

	if (len != 0 && (address == 0 || (address & (PRT_PAGE_SIZE - 1u)) != 0 ||
	                 (len & (PRT_PAGE_SIZE - 1u)) != 0 || address < PRT_APERTURE_START ||
	                 len > PRT_APERTURE_END - address)) {
		return KERNEL_ERROR_EINVAL;
	}

	PrtAperture old {};
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		old = g_prt_apertures[static_cast<size_t>(index)];
	}
	if (old.size != 0) {
		UnmapGpuRange(old.address, old.size);
	}
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		g_prt_apertures[static_cast<size_t>(index)] =
		    len == 0 ? PrtAperture {} : PrtAperture {address, static_cast<uint64_t>(len)};
	}
	if (len != 0) {
		MapGpuRange(address, len);
	}

	LOGF_COLOR(Log::Color::Green, "\t[Ok]\n");

	return OK;
}

int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len) {
	PRINT_NAME();

	LOGF("\t index = %d\n"
	     "\t addr  = %p\n"
	     "\t len   = %p\n",
	     index, static_cast<void*>(addr), static_cast<void*>(len));

	if (index < 0 || index > PRT_APERTURE_MAX_INDEX) {
		return KERNEL_ERROR_EINVAL;
	}

	if (addr == nullptr || len == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	PrtAperture aperture {};
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		aperture = g_prt_apertures[static_cast<size_t>(index)];
	}

	*addr = reinterpret_cast<void*>(aperture.address);
	*len  = static_cast<size_t>(aperture.size);

	LOGF_COLOR(Log::Color::Green,
	           "\t *addr = 0x%016" PRIx64 "\n"
	           "\t *len  = 0x%016" PRIx64 "\n"
	           "\t[Ok]\n",
	           aperture.address, aperture.size);

	return OK;
}

int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n"
	     "\t name = %s\n",
	     vaddr, len, name != nullptr ? name : "(null)");

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= KERNEL_MAXIMUM_NAME_LENGTH) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	g_physical_memory->SetVirtualRangeName(vaddr, len, name);
	g_flexible_memory->SetVirtualRangeName(vaddr, len, name);
	g_virtual_ranges->Rename(vaddr, len, name);

	return OK;
}

int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n",
	     vaddr, len);

	g_physical_memory->SetVirtualRangeName(vaddr, len, "");
	g_flexible_memory->SetVirtualRangeName(vaddr, len, "");
	g_virtual_ranges->Rename(vaddr, len, "");

	return OK;
}

static bool FreeGuestMemoryOwner(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space->ReleaseCommitted(vaddr, size) &&
	       g_virtual_ranges->Remove(vaddr, size);
}

static int UnmapRange(uint64_t vaddr, size_t len, const VirtualRanges::Range& range);

static int UnmapMemoryRange(uint64_t vaddr, size_t len) {
	if (len == 0 || UINT64_MAX - vaddr < len) {
		return KERNEL_ERROR_EINVAL;
	}

	// The virtual ranges it spans, one after the other. (A call level for each, as this was, overflowed a job fiber's
	// stack: the game unmapped a streaming pool of 866 ranges in one call in the Nexus, 10-12.)
	for (;;) {
		VirtualRanges::Range range {};
		if (!g_virtual_ranges->Query(vaddr, 0, &range)) {
			return KERNEL_ERROR_EACCES;
		}
		const auto chunk_len = std::min<uint64_t>(len, range.size - (vaddr - range.start));
		if (chunk_len == 0) {
			return KERNEL_ERROR_EINVAL;
		}
		if (const int ret = UnmapRange(vaddr, chunk_len, range); ret != OK || chunk_len == len) {
			return ret;
		}
		vaddr += chunk_len;
		len -= chunk_len;
	}
}

// [vaddr, vaddr + len) of one virtual range.
static int UnmapRange(uint64_t vaddr, size_t len, const VirtualRanges::Range& range) {
	if (IsReservedRangeType(range.type)) {
		if (!g_guest_address_space->ReleaseFree(vaddr, len)) {
			return KERNEL_ERROR_EACCES;
		}
		g_virtual_ranges->Remove(vaddr, len);
		return OK;
	}
	if (range.type == VirtualRangeType::Code || range.type == VirtualRangeType::Runtime) {
		return FreeGuestMemoryOwner(vaddr, len) ? OK : KERNEL_ERROR_EACCES;
	}

	GpuAccessMode gpu_mode       = GpuAccessMode::NoAccess;
	bool          owner_unmapped = false;
	if (range.type == VirtualRangeType::Pooled) {
		std::vector<PooledMemory::Mapping> mappings;
		if (g_pooled_memory->Query(vaddr, len, &mappings)) {
			VirtualMemory::Mode mode        = VirtualMemory::Mode::NoAccess;
			GpuAccessMode       decoded_gpu = GpuAccessMode::NoAccess;
			owner_unmapped = DecodeMemoryProtection(range.protection, &mode, &decoded_gpu) &&
			                 UnmapPooledBackingTransactional(mappings, mode);
			if (owner_unmapped && !g_pooled_memory->Release(vaddr, len, &gpu_mode)) {
				EXIT("failed to release unmapped pooled-memory range\n");
			}
		}
		if (!owner_unmapped) {
			return KERNEL_ERROR_EACCES;
		}
	} else if (range.type == VirtualRangeType::Direct) {
		VirtualMemory::Mode direct_mode   = VirtualMemory::Mode::NoAccess;
		GpuAccessMode       direct_gpu    = GpuAccessMode::NoAccess;
		const auto          direct_offset = range.offset + vaddr - range.start;
		if (DecodeMemoryProtection(range.protection, &direct_mode, &direct_gpu) &&
		    g_guest_address_space->BackingContains(vaddr, len) &&
		    (FreedDiscard::BeforeUnmap(vaddr, len), true) && // (ram2, KYTY_DS_FREED_DISCARD=2: reset through this view)
		    (FreedDiscard::FreedRelease::BeforeUnmap(vaddr, len), true) && // (ram6, KYTY_DS_FREED_RELEASE)
		    g_guest_address_space->UnmapBacking(vaddr, len)) {
			if (g_physical_memory->Unmap(vaddr, len, &gpu_mode)) {
				owner_unmapped = true;
				FreedDiscard::Note(direct_offset, len); // (ram1, KYTY_DS_FREED_DISCARD: queued, dropped if still unmapped)
			} else {
				EXIT_IF(!g_guest_address_space->MapBacking(vaddr, len, direct_offset, direct_mode));
			}
		}
	} else if (range.type == VirtualRangeType::Stack) {
		owner_unmapped = g_guest_address_space->ReleaseCommitted(vaddr, len);
	} else {
		uint64_t ignored_host = 0;
		owner_unmapped        = g_flexible_memory->Unmap(vaddr, len, &gpu_mode, &ignored_host);
	}
	if (!owner_unmapped) {
		return KERNEL_ERROR_EACCES;
	}

	g_virtual_ranges->Remove(vaddr, len);

	if (g_free_callback != nullptr && IsCommittedRangeType(range.type)) {
		g_free_callback(vaddr, len);
	}
	if (range.type == VirtualRangeType::Pooled) {
		MemoryPoolSubtractCommitted(len);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     vaddr, len);

	if (len == 0 || UINT64_MAX - vaddr < len) {
		return KERNEL_ERROR_EINVAL;
	}
	std::vector<VirtualRanges::Range> ranges;
	if (!g_virtual_ranges->QuerySpan(vaddr, len, &ranges)) {
		return KERNEL_ERROR_EACCES;
	}
	UnmapGpuRange(vaddr, len, true);
	const int result = UnmapMemoryRange(vaddr, len);
	if (result != OK) {
		RestoreGpuRangeProtection(vaddr, len);
	} else {
		::Loader::DemonsSoulsTextureProbe::OnUnmap(vaddr, len); // KYTY_TEXTURE_PROBE (measurement only)
	}
	return result;
}

size_t KYTY_SYSV_ABI KernelGetDirectMemorySize() {
	PRINT_NAME();

	return PhysicalMemory::Size();
}

int KYTY_SYSV_ABI KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                  size_t alignment, int64_t* phys_addr_out,
                                                  size_t* size_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t search_start = 0x%016" PRIx64 "\n"
	     "\t search_end   = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n",
	     search_start, search_end, static_cast<uint64_t>(alignment));

	if (phys_addr_out == nullptr || size_out == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*phys_addr_out = 0;
	*size_out      = 0;

	if (search_start < 0 || search_end < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	if (search_end <= search_start) {
		LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		return NoMemory(__func__, __LINE__);
	}

	uint64_t phys_addr = 0;
	uint64_t size      = 0;
	if (!g_physical_memory->Available(static_cast<uint64_t>(search_start),
	                                  static_cast<uint64_t>(search_end), alignment, &phys_addr,
	                                  &size)) {
		LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		return NoMemory(__func__, __LINE__);
	}

	// KYTY_DS_ALLOC_PATCH: the one call in the game's page-allocator start-up is told at most the clamp (the rest of
	// direct memory stays free for later allocations; loader/demonsSoulsAllocClamp.cpp).
	if (const auto clamp_return = AvailableClamp::g_return.load(std::memory_order_acquire);
	    clamp_return != 0 && reinterpret_cast<uint64_t>(__builtin_return_address(0)) == clamp_return) {
		const uint64_t told = std::min(size, AvailableClamp::g_limit);
		std::printf("[port] ds-alloc: eboot+0x%llx sceKernelAvailableDirectMemorySize(0x%llx, 0x%llx, 0x%llx) = base "
		            "0x%llx size 0x%llx (%llu MiB) -> told 0x%llx (%llu MiB)%s\n",
		            static_cast<unsigned long long>(clamp_return - AvailableClamp::g_base),
		            static_cast<unsigned long long>(search_start), static_cast<unsigned long long>(search_end),
		            static_cast<unsigned long long>(alignment), static_cast<unsigned long long>(phys_addr),
		            static_cast<unsigned long long>(size), static_cast<unsigned long long>(size >> 20u),
		            static_cast<unsigned long long>(told), static_cast<unsigned long long>(told >> 20u),
		            told < size ? "" : " (not above the clamp: unchanged)");
		std::fflush(stdout);
		size = told;
	}

	*phys_addr_out = static_cast<int64_t>(phys_addr);
	*size_out      = static_cast<size_t>(size);

	LOGF_COLOR(Log::Color::Green,
	           "\t phys_addr = 0x%016" PRIx64 "\n"
	           "\t size      = 0x%016" PRIx64 "\n"
	           "\t[Ok]\n",
	           phys_addr, size);

	return OK;
}

int KYTY_SYSV_ABI KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                          int* gpu_available) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (cpu_total == nullptr || cpu_available == nullptr || gpu_total == nullptr ||
	    gpu_available == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	const auto cpu_used =
	    (g_virtual_ranges != nullptr ? g_virtual_ranges->CountPageTableEntries(false) : 0);
	const auto gpu_used =
	    (g_virtual_ranges != nullptr ? g_virtual_ranges->CountPageTableEntries(true) : 0);

	*cpu_total     = PAGE_TABLE_POOL_ENTRIES;
	*gpu_total     = PAGE_TABLE_POOL_ENTRIES;
	*cpu_available = PAGE_TABLE_POOL_ENTRIES -
	                 static_cast<int>(std::min<uint64_t>(cpu_used, PAGE_TABLE_POOL_ENTRIES));
	*gpu_available = PAGE_TABLE_POOL_ENTRIES -
	                 static_cast<int>(std::min<uint64_t>(gpu_used, PAGE_TABLE_POOL_ENTRIES));

	LOGF_COLOR(Log::Color::Green,
	           "\t cpu_total     = %d\n"
	           "\t cpu_available = %d\n"
	           "\t gpu_total     = %d\n"
	           "\t gpu_available = %d\n"
	           "\t[Ok]\n",
	           *cpu_total, *cpu_available, *gpu_total, *gpu_available);

	return OK;
}

int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t offset    = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t info_size = 0x%016" PRIx64 "\n",
	     offset, flags, info_size);

	struct QueryInfo {
		int64_t start;
		int64_t end;
		int     memory_type;
	};

	if (offset < 0 || (flags != 0 && flags != 1) || info_size != sizeof(QueryInfo) ||
	    info == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto* query_info = static_cast<QueryInfo*>(info);

	PhysicalMemory::AllocatedBlock block {};
	{
		Common::LockGuard lock(g_physical_memory->GetMutex());
		const auto&       blocks  = g_physical_memory->GetPhysicalBlocks();
		auto              current = blocks.upper_bound(static_cast<uint64_t>(offset));
		if (current != blocks.begin()) {
			auto previous = std::prev(current);
			if (static_cast<uint64_t>(offset) <
			    previous->second.start_addr + previous->second.size) {
				current = previous;
			}
		}
		if (current == blocks.end() ||
		    (flags == 0 && (static_cast<uint64_t>(offset) < current->second.start_addr ||
		                    static_cast<uint64_t>(offset) >=
		                        current->second.start_addr + current->second.size))) {
			if (flags == 1 && static_cast<uint64_t>(offset) < PhysicalMemory::Size()) {
				query_info->start       = static_cast<int64_t>(PhysicalMemory::Size());
				query_info->end         = static_cast<int64_t>(PhysicalMemory::Size());
				query_info->memory_type = 0;
				LOGF_COLOR(Log::Color::Green, "\t terminal    = true\n\t[Ok]\n");
				return OK;
			}

			LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
			return KERNEL_ERROR_EACCES;
		}

		block        = current->second;
		uint64_t end = block.start_addr + block.size;
		for (auto following = std::next(current);
		     following != blocks.end() && following->second.start_addr == end &&
		     following->second.memory_type == block.memory_type;
		     ++following) {
			end += following->second.size;
		}
		block.size = end - block.start_addr;
	}

	query_info->start       = static_cast<int64_t>(block.start_addr);
	query_info->end         = static_cast<int64_t>(block.start_addr + block.size);
	query_info->memory_type = block.memory_type;

	LOGF_COLOR(Log::Color::Green,
	           "\t start       = %016" PRIx64 "\n"
	           "\t end         = %016" PRIx64 "\n"
	           "\t memory_type = %d\n"
	           "\t[Ok]\n",
	           query_info->start, query_info->end, query_info->memory_type);

	return OK;
}

int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t search_start = 0x%016" PRIx64 "\n"
	     "\t search_end   = 0x%016" PRIx64 "\n"
	     "\t len          = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n"
	     "\t memory_type  = %d\n",
	     search_start, search_end, len, alignment, memory_type);

	constexpr uint64_t PAGE_SIZE = 0x4000;
	if (search_start < 0 || search_end <= search_start || len == 0 ||
	    (len & (PAGE_SIZE - 1u)) != 0 || (alignment != 0 && (alignment & (PAGE_SIZE - 1u)) != 0) ||
	    phys_addr_out == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	uint64_t   addr      = 0;
	const bool allocated = g_physical_memory->Alloc(search_start, search_end, len, alignment, &addr, memory_type);
	if (const auto caller = reinterpret_cast<uint64_t>(__builtin_return_address(0)); AvailableClamp::Traced(caller)) {
		AvailableClamp::Print("sceKernelAllocateDirectMemory", caller, len,
		                      allocated ? addr : static_cast<uint64_t>(search_start),
		                      allocated ? "allocated" : "FAILED (EAGAIN; phys = search start)");
	}
	if (!allocated && AvailableClamp::g_return.load(std::memory_order_relaxed) != 0) {
		std::printf("[port] ds-alloc: a direct allocation of 0x%llx (%llu MiB) in [0x%llx, 0x%llx) FAILED with the "
		            "allocator clamp on\n",
		            static_cast<unsigned long long>(len), static_cast<unsigned long long>(len >> 20u),
		            static_cast<unsigned long long>(search_start), static_cast<unsigned long long>(search_end));
		std::fflush(stdout);
	}
	if (!allocated) {
		GuestStoreTooSmall("direct memory", len);
		LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		return KERNEL_ERROR_EAGAIN;
	}

	*phys_addr_out = static_cast<int64_t>(addr);
	FreedDiscard::NotePool(addr, len); // (ram1, KYTY_DS_FREED_DISCARD)

	LOGF_COLOR(Log::Color::Green, "\tphys_addr    = %016" PRIx64 "\n\t[Ok]\n", addr);

	return OK;
}

int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t len          = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n"
	     "\t memory_type  = %d\n",
	     len, alignment, memory_type);

	return KernelAllocateDirectMemory(0, static_cast<int64_t>(PhysicalMemory::Size()), len,
	                                  alignment, memory_type, phys_addr_out);
}

static int ReleaseDirectMemoryInternal(int64_t start, size_t len) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);
	FreedDiscard::NoteRelease(static_cast<uint64_t>(start), len); // (ram1, KYTY_DS_FREED_DISCARD)

	if (g_pooled_memory->ReleaseExpansion(static_cast<uint64_t>(start), len)) {
		if (!g_physical_memory->ReleasePoolExpansion(static_cast<uint64_t>(start), len)) {
			EXIT("failed to release physical pool expansion\n");
		}
		return OK;
	}

	std::vector<PhysicalMemory::AllocatedBlock> allocated_span;
	if (!g_physical_memory->GetAllocatedSpan(static_cast<uint64_t>(start), len, &allocated_span)) {
		return KERNEL_ERROR_EACCES;
	}

	const auto mapped_aliases = g_physical_memory->FindMappings(start, len);
	for (const auto& alias: mapped_aliases) {
		VirtualRanges::Range range {};
		if (!g_guest_address_space->BackingContains(alias.map_vaddr, alias.map_size) ||
		    !g_virtual_ranges->Query(alias.map_vaddr, 0, &range) ||
		    range.type != VirtualRangeType::Direct ||
		    alias.map_size > range.start + range.size - alias.map_vaddr) {
			EXIT("direct-memory alias escaped guest address-space ownership\n");
		}
	}

	for (const auto& alias: mapped_aliases) {
		UnmapGpuRange(alias.map_vaddr, alias.map_size);
	}

	auto restore_gpu_aliases = [&mapped_aliases]() {
		for (const auto& alias: mapped_aliases) {
			MapGpuRange(alias.map_vaddr, alias.map_size);
		}
	};
	auto restore_owner_aliases = [](const std::vector<PhysicalMemory::AllocatedBlock>& aliases) {
		for (auto it = aliases.rbegin(); it != aliases.rend(); ++it) {
			EXIT_IF(!g_guest_address_space->MapBacking(it->map_vaddr, it->map_size, it->start_addr,
			                                           it->mode));
		}
	};

	std::vector<PhysicalMemory::AllocatedBlock> owner_unmapped;
	for (const auto& alias: mapped_aliases) {
		if (!g_guest_address_space->UnmapBacking(alias.map_vaddr, alias.map_size)) {
			restore_owner_aliases(owner_unmapped);
			restore_gpu_aliases();
			return KERNEL_ERROR_EACCES;
		}
		owner_unmapped.push_back(alias);
	}

	std::vector<PhysicalMemory::AllocatedBlock> metadata_unmapped;
	for (const auto& alias: mapped_aliases) {
		GpuAccessMode alias_gpu_mode = GpuAccessMode::NoAccess;
		if (!g_physical_memory->Unmap(alias.map_vaddr, alias.map_size, &alias_gpu_mode)) {
			for (const auto& removed: metadata_unmapped) {
				EXIT_IF(!g_physical_memory->Map(removed.map_vaddr, removed.start_addr,
				                                removed.map_size, removed.prot, removed.mode,
				                                removed.gpu_mode));
			}
			restore_owner_aliases(owner_unmapped);
			restore_gpu_aliases();
			return KERNEL_ERROR_EACCES;
		}
		metadata_unmapped.push_back(alias);
	}

	for (const auto& alias: mapped_aliases) {
		EXIT_IF(!g_virtual_ranges->Remove(alias.map_vaddr, alias.map_size));
	}

	for (const auto& block: allocated_span) {
		uint64_t      unused_vaddr = 0;
		uint64_t      unused_size  = 0;
		GpuAccessMode unused_gpu   = GpuAccessMode::NoAccess;
		EXIT_IF(!g_physical_memory->Release(block.start_addr, block.size, &unused_vaddr,
		                                    &unused_size, &unused_gpu));
	}

	if (g_free_callback != nullptr) {
		for (const auto& alias: mapped_aliases) {
			g_free_callback(alias.map_vaddr, alias.map_size);
		}
	}

	return OK;
}

static int ValidateDirectReleaseRange(int64_t start, size_t len) {
	constexpr uint64_t PAGE_SIZE = 0x4000;
	return start < 0 || (static_cast<uint64_t>(start) & (PAGE_SIZE - 1u)) != 0 ||
	               (len & (PAGE_SIZE - 1u)) != 0
	           ? KERNEL_ERROR_EINVAL
	           : OK;
}

int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len) {
	PRINT_NAME();

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     start, len);

	const int validation = ValidateDirectReleaseRange(start, len);
	if (validation != OK) {
		return validation;
	}
	if (len != 0) {
		(void)ReleaseDirectMemoryInternal(start, len);
	}
	return OK;
}

int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len) {
	PRINT_NAME();

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     start, len);

	const int validation = ValidateDirectReleaseRange(start, len);
	if (validation != OK || len == 0) {
		return validation;
	}

	const int result = ReleaseDirectMemoryInternal(start, len);
	return result == KERNEL_ERROR_EACCES ? KERNEL_ERROR_ENOENT : result;
}

int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment) {
	KYTY_MAPA_GRANDE_SCOPE("sceKernelMapDirectMemory");
	struct AccessReport { ~AccessReport() { ReportGuestAccess(false); } } access_report;
	PRINT_NAME();

	if (const auto caller = reinterpret_cast<uint64_t>(__builtin_return_address(0)); AvailableClamp::Traced(caller)) {
		AvailableClamp::Print("sceKernelMapDirectMemory", caller, len, static_cast<uint64_t>(direct_memory_start),
		                      "map requested");
	}

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (addr == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}
	constexpr uint64_t PAGE_SIZE              = 0x4000;
	constexpr int      GUEST_MAP_FIXED        = 0x10;
	constexpr int      GUEST_MAP_NO_OVERWRITE = 0x80;

	if (len == 0 || (len & (PAGE_SIZE - 1u)) != 0 || direct_memory_start < 0 ||
	    (static_cast<uint64_t>(direct_memory_start) & (PAGE_SIZE - 1u)) != 0 ||
	    (alignment != 0 && (alignment & (alignment - 1u)) != 0 &&
	     (alignment & (PAGE_SIZE - 1u)) != 0)) {
		return KERNEL_ERROR_EINVAL;
	}
	if ((prot & PROT_CPU_EXEC) != 0) {
		return KERNEL_ERROR_EACCES;
	}

	bool fixed        = ((flags & GUEST_MAP_FIXED) != 0);
	bool no_overwrite = ((flags & GUEST_MAP_NO_OVERWRITE) != 0);

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		return KERNEL_ERROR_EINVAL;
	}
	if (!g_physical_memory->CanMapDirect(static_cast<uint64_t>(direct_memory_start), len)) {
		return NoMemory(__func__, __LINE__);
	}

	auto                 in_addr              = reinterpret_cast<uint64_t>(*addr);
	uint64_t             out_addr             = 0;
	bool                 shared_backing       = false;
	bool                 consumed_reservation = false;
	VirtualRanges::Range consumed_range {};
	auto                 shared_failure = GuestBackingStore::FailureReason::None;
	std::unique_lock residency_lock(g_prt_residency_mutex, std::defer_lock);
	// Direct mappings must remain views of the single backing object. Anonymous fallbacks break
	// aliasing and lose direct-memory contents when a range is unmapped and mapped again.
	auto map_shared_fixed = [&](uint64_t target_addr) -> bool {
		return g_guest_address_space->MapBacking(target_addr, len, direct_memory_start, mode,
		                                         &shared_failure);
	};
	auto map_consumed_reserved_fixed = [&]() {
		if (map_shared_fixed(in_addr)) {
			out_addr       = in_addr;
			shared_backing = true;
		}
	};

	if (fixed) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1u)) != 0 ||
		    (alignment != 0 && in_addr % alignment != 0)) {
			return KERNEL_ERROR_EINVAL;
		}
		if (no_overwrite && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return NoMemory(__func__, __LINE__);
		}

		std::vector<VirtualRanges::Range> reserved_ranges;
		if (g_virtual_ranges->QuerySpan(in_addr, len, &reserved_ranges) &&
		    std::all_of(reserved_ranges.begin(), reserved_ranges.end(), [](const auto& range) {
			    return range.type == VirtualRangeType::Reserved;
		    })) {
			UnmapGpuRange(in_addr, len);
			residency_lock.lock();
			consumed_range = reserved_ranges.front();
			if (g_virtual_ranges->ConsumeReservedSpan(in_addr, len)) {
				consumed_reservation = true;
				map_consumed_reserved_fixed();
			}
		}
		if (!consumed_reservation) {
			if (residency_lock.owns_lock()) residency_lock.unlock();
			if (ReplaceFixedRangeWithReserved(in_addr, len)) {
				residency_lock.lock();
				if (g_virtual_ranges->ConsumeReservedSpan(in_addr, len, &consumed_range)) {
					consumed_reservation = true;
					map_consumed_reserved_fixed();
				}
			}
		}
		if (!consumed_reservation) {
			return NoMemory(__func__, __LINE__);
		}
	} else {
		constexpr size_t DEFAULT_ALIGNMENT = 0x4000;
		alignment                          = (alignment != 0 ? alignment : DEFAULT_ALIGNMENT);
		std::vector<VirtualRanges::Range> reserved_ranges;
		if (in_addr != 0 && g_virtual_ranges->QuerySpan(in_addr, len, &reserved_ranges) &&
		    std::all_of(reserved_ranges.begin(), reserved_ranges.end(), [](const auto& range) {
			    return range.type == VirtualRangeType::Reserved;
		    })) {
			UnmapGpuRange(in_addr, len);
			residency_lock.lock();
			consumed_range = reserved_ranges.front();
			if (g_virtual_ranges->ConsumeReservedSpan(in_addr, len)) {
				consumed_reservation = true;
				if (map_shared_fixed(in_addr)) {
					out_addr       = in_addr;
					shared_backing = true;
				}
			}
		}
		if (!consumed_reservation) {
			if (residency_lock.owns_lock()) residency_lock.unlock();
			out_addr = FindGuestFreeRange(in_addr, len, alignment);
			if (out_addr != 0) {
				UnmapGpuRange(out_addr, len);
				residency_lock.lock();
				shared_backing = map_shared_fixed(out_addr);
				if (!shared_backing) {
					out_addr = 0;
				}
			}
		}
	}

	*addr = reinterpret_cast<void*>(out_addr);

	const char* shared_reason = "n/a";
	if (!shared_backing) {
		shared_reason = GuestBackingStore::GetFailureReasonName(shared_failure);
	}

	LOGF("\t in_addr  = 0x%016" PRIx64 "\n"
	     "\t out_addr = 0x%016" PRIx64 "\n"
	     "\t dmem     = 0x%016" PRIx64 "\n"
	     "\t size     = 0x%016" PRIx64 "\n"
	     "\t mode     = %s\n"
	     "\t flags    = 0x%08" PRIx32 "\n"
	     "\t align    = 0x%016" PRIx64 "\n"
	     "\t gpu_mode = %s\n"
	     "\t shared   = %s\n"
	     "\t reason   = %s\n",
	     in_addr, out_addr, static_cast<uint64_t>(direct_memory_start), len,
	     Common::EnumName(mode).c_str(), static_cast<uint32_t>(flags), alignment,
	     Common::EnumName(gpu_mode).c_str(), shared_backing ? "yes" : "no", shared_reason);

	if (out_addr == 0) {
		if (consumed_reservation) {
			g_virtual_ranges->Add(in_addr, len, 0, 0, 0, VirtualRangeType::Reserved,
			                      consumed_range.name);
		}
		return NoMemory(__func__, __LINE__);
	}

	if (!g_physical_memory->Map(out_addr, direct_memory_start, len, prot, mode, gpu_mode)) {
		LOGF_COLOR(Log::Color::Red, "\t [Fail]\n");
		EXIT_IF(!g_guest_address_space->UnmapBacking(out_addr, len));
		if (consumed_reservation) {
			EXIT_IF(!g_virtual_ranges->Add(in_addr, len, 0, 0, 0, VirtualRangeType::Reserved,
			                               consumed_range.name));
		}
		return KERNEL_ERROR_EBUSY;
	}

	PhysicalMemory::AllocatedBlock mapped_block {};
	g_physical_memory->Find(direct_memory_start, false, &mapped_block);
	if (!g_virtual_ranges->Add(out_addr, len, direct_memory_start, prot, mapped_block.memory_type,
	                           VirtualRangeType::Direct, "")) {
		GpuAccessMode rollback_gpu_mode = GpuAccessMode::NoAccess;
		EXIT_IF(!g_physical_memory->Unmap(out_addr, len, &rollback_gpu_mode));
		EXIT_IF(!g_guest_address_space->UnmapBacking(out_addr, len));
		if (consumed_reservation) {
			EXIT_IF(!g_virtual_ranges->Add(in_addr, len, 0, 0, 0, VirtualRangeType::Reserved,
			                               consumed_range.name));
		}
		return KERNEL_ERROR_EBUSY;
	}

	residency_lock.unlock();
	MapGpuRange(out_addr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(out_addr, len);
	}

	LOGF_COLOR(Log::Color::Green, "\t [Ok]\n");

	return OK;
}

int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment) {
	KYTY_MAPA_GRANDE_SCOPE("sceKernelMapDirectMemory2");
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t type = %d\n", type);

	auto ret = KernelMapDirectMemory(addr, len, prot, flags, direct_memory_start, alignment);
	if (ret == OK && addr != nullptr && *addr != nullptr) {
		const auto out_addr = reinterpret_cast<uint64_t>(*addr);
		g_physical_memory->SetVirtualRangeMemoryType(out_addr, len, type);
		g_virtual_ranges->SetMemoryType(out_addr, len, type);
	}

	return ret;
}

int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name) {
	KYTY_MAPA_GRANDE_SCOPE("sceKernelMapNamedDirectMemory");
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t name = %s\n", name != nullptr ? name : "(null)");

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= KERNEL_MAXIMUM_NAME_LENGTH) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	auto ret = KernelMapDirectMemory(addr, len, prot, flags, direct_memory_start, alignment);
	if (ret == OK && addr != nullptr) {
		g_physical_memory->SetVirtualRangeName(reinterpret_cast<uint64_t>(*addr), len, name);
		g_virtual_ranges->Rename(reinterpret_cast<uint64_t>(*addr), len, name);
	}

	return ret;
}

int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled() {
	PRINT_NAME();

	return 0;
}

int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	EXIT_NOT_IMPLEMENTED(addr == nullptr);

	VirtualRanges::Range range {};
	if (!g_virtual_ranges->Query(reinterpret_cast<uint64_t>(addr), 0, &range)) {
		return KERNEL_ERROR_EACCES;
	}

	if (start != nullptr) {
		*start = reinterpret_cast<void*>(range.start);
	}
	if (end != nullptr) {
		*end = reinterpret_cast<void*>(range.start + range.size);
	}
	if (prot != nullptr) {
		*prot = range.protection;
	}

	return OK;
}

static bool ReplaceFixedRangeWithReserved(uint64_t start, uint64_t size) {

	struct ReplacedChunk {
		VirtualRanges::Range                        range {};
		VirtualMemory::Mode                         mode     = VirtualMemory::Mode::NoAccess;
		GpuAccessMode                               gpu_mode = GpuAccessMode::NoAccess;
		std::vector<FlexibleMemory::AllocatedBlock> flexible_blocks;
		bool                                        host_unmapped    = false;
		bool                                        backend_unmapped = false;
	};

	std::vector<ReplacedChunk> chunks;
	const auto                 end     = start + size;
	auto                       current = start;

	while (current < end) {
		VirtualRanges::Range range {};
		if (!g_virtual_ranges->Query(current, 1, &range)) {
			break;
		}
		if (range.start >= end) {
			break;
		}
		if (current < range.start) {
			current = std::min<uint64_t>(end, range.start);
			continue;
		}

		const auto range_end = range.start + range.size;
		const auto chunk     = std::min<uint64_t>(end, range_end) - current;

		ReplacedChunk replaced {};
		replaced.range       = range;
		replaced.range.start = current;
		replaced.range.size  = chunk;
		if (range.type == VirtualRangeType::Direct) {
			replaced.range.offset += current - range.start;
		}
		DecodeMemoryProtection(replaced.range.protection, &replaced.mode, &replaced.gpu_mode);
		if (range.type == VirtualRangeType::Direct &&
		    !g_guest_address_space->BackingContains(current, chunk)) {
			return false;
		}
		if (range.type == VirtualRangeType::Flexible &&
		    !g_flexible_memory->Snapshot(current, chunk, &replaced.flexible_blocks)) {
			return false;
		}
		if (range.type == VirtualRangeType::Pooled) {
			EXIT("reserve-fixed replacement of pooled memory is unsupported: addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n",
			     replaced.range.start, replaced.range.size);
		}
		chunks.push_back(replaced);

		current += chunk;
	}

	// Existing pages remain logically owned throughout a successful replacement
	// (or its rollback). Readers must not observe the temporary Remove/Add gap or
	// a backing view halfway through replacement. GPU drains precede this lock.
	UnmapGpuRange(start, size);
	std::unique_lock residency_lock(g_prt_residency_mutex);
	auto restore_chunks = [&chunks, &residency_lock]() -> bool {
		bool ok = true;
		for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
			const auto& chunk            = *it;
			bool        host_restored    = true;
			bool        backend_restored = true;

			if (chunk.range.type == VirtualRangeType::Direct) {
				if (chunk.host_unmapped) {
					host_restored = g_guest_address_space->MapBacking(
					    chunk.range.start, chunk.range.size, chunk.range.offset, chunk.mode);
				}
				if (chunk.backend_unmapped) {
					backend_restored = g_physical_memory->Map(
					    chunk.range.start, chunk.range.offset, chunk.range.size,
					    chunk.range.protection, chunk.mode, chunk.gpu_mode);
				}
			} else if (chunk.range.type == VirtualRangeType::Flexible && chunk.backend_unmapped) {
				host_restored = backend_restored =
				    g_flexible_memory->Restore(chunk.flexible_blocks);
			} else if (IsPrivateCommittedRangeType(chunk.range.type) && chunk.host_unmapped) {
				host_restored =
				    g_guest_address_space->Commit(chunk.range.start, chunk.range.size, chunk.mode);
			}
			ok = host_restored && backend_restored && ok;
		}

		for (const auto& chunk: chunks) {
			const bool range_restored = g_virtual_ranges->Add(
			    chunk.range.start, chunk.range.size, chunk.range.offset, chunk.range.protection,
			    chunk.range.memory_type, chunk.range.type, chunk.range.name,
			    chunk.range.disallow_merge);
			ok = range_restored && ok;
		}
		if (ok) {
			residency_lock.unlock();
			for (const auto& chunk: chunks) {
				if (IsCommittedRangeType(chunk.range.type)) {
					MapGpuRange(chunk.range.start, chunk.range.size);
				}
			}
		}
		return ok;
	};

	g_virtual_ranges->Remove(start, size);

	for (auto& chunk: chunks) {
		GpuAccessMode gpu_mode = GpuAccessMode::NoAccess;
		bool          unmapped = true;

		if (chunk.range.type == VirtualRangeType::Direct) {
			// (ram2, KYTY_DS_FREED_DISCARD=2: Demon's Souls' UnmapPhysicalPages unmaps this way too. A rollback below -
			// the reservation failing, fatal anyway - would map reset pages back.)
			FreedDiscard::BeforeUnmap(chunk.range.start, chunk.range.size);
			FreedDiscard::FreedRelease::BeforeUnmap(chunk.range.start, chunk.range.size); // (ram6, KYTY_DS_FREED_RELEASE)
			if (!g_guest_address_space->UnmapBacking(chunk.range.start, chunk.range.size)) {
				unmapped = false;
			} else {
				chunk.host_unmapped = true;
			}
			if (unmapped) {
				unmapped = g_physical_memory->Unmap(chunk.range.start, chunk.range.size, &gpu_mode);
				chunk.gpu_mode         = gpu_mode;
				chunk.backend_unmapped = unmapped;
			}
		} else if (chunk.range.type == VirtualRangeType::Flexible) {
			unmapped = g_flexible_memory->Unmap(chunk.range.start, chunk.range.size, &gpu_mode);
			chunk.host_unmapped    = unmapped;
			chunk.backend_unmapped = unmapped;
			chunk.gpu_mode         = gpu_mode;
		} else if (IsPrivateCommittedRangeType(chunk.range.type)) {
			unmapped = g_guest_address_space->ReleaseCommitted(chunk.range.start, chunk.range.size);
			chunk.host_unmapped    = unmapped;
			chunk.backend_unmapped = unmapped;
		} else if (IsReservedRangeType(chunk.range.type)) {
			unmapped = g_guest_address_space->ReleaseFree(chunk.range.start, chunk.range.size);
		} else {
			unmapped = false;
		}
		if (chunk.range.type == VirtualRangeType::Direct) {
			chunk.gpu_mode         = gpu_mode;
			chunk.backend_unmapped = chunk.backend_unmapped && unmapped;
		}

		if (!unmapped) {
			LOGF_COLOR(Log::Color::Red,
			           "\t reserve-fixed replace: backend unmap failed at 0x%016" PRIx64
			           ", size=0x%016" PRIx64 ", type=%s\n",
			           chunk.range.start, chunk.range.size,
			           Common::EnumName(chunk.range.type).c_str());
			if (!restore_chunks()) {
				EXIT("reserve-fixed backend-unmap rollback failed\n");
			}
			return false;
		}
	}

	if (!g_guest_address_space->ReserveFixed(start, size)) {
		if (!restore_chunks()) {
			EXIT("reserve-fixed host-reservation rollback failed\n");
		}
		return false;
	}

	bool range_added = false;
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	if (g_test_fail_next_fixed_reserve_range_add) {
		g_test_fail_next_fixed_reserve_range_add = false;
	} else
#endif
	{
		range_added =
		    g_virtual_ranges->Add(start, size, 0, 0, 0, VirtualRangeType::Reserved, "anon");
	}
	if (!range_added) {
		LOGF_COLOR(Log::Color::Red,
		           "\t reserve-fixed replace: range add failed at 0x%016" PRIx64
		           ", size=0x%016" PRIx64 "\n",
		           start, size);
		if (!restore_chunks()) {
			EXIT("reserve-fixed range-registration rollback failed\n");
		}
		auto free_start = start;
		for (const auto& chunk: chunks) {
			if (free_start < chunk.range.start &&
			    !g_guest_address_space->ReleaseFree(free_start, chunk.range.start - free_start)) {
				EXIT("reserve-fixed range-registration gap cleanup failed\n");
			}
			free_start = chunk.range.start + chunk.range.size;
		}
		if (free_start < start + size &&
		    !g_guest_address_space->ReleaseFree(free_start, start + size - free_start)) {
			EXIT("reserve-fixed range-registration tail cleanup failed\n");
		}
		return false;
	}

	residency_lock.unlock();
	for (const auto& chunk: chunks) {
		if (g_free_callback != nullptr && IsCommittedRangeType(chunk.range.type)) {
			g_free_callback(chunk.range.start, chunk.range.size);
		}
		// (ram1, KYTY_DS_FREED_DISCARD: Demon's Souls' UnmapPhysicalPages unmaps this way too, sceKernelReserveVirtualRange
		// over the pages it frees.)
		if (chunk.range.type == VirtualRangeType::Direct) FreedDiscard::Note(chunk.range.offset, chunk.range.size);
	}

	return true;
}

int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	const auto in_addr = (addr != nullptr ? reinterpret_cast<uint64_t>(*addr) : 0);

	LOGF("\t in_addr   = 0x%016" PRIx64 "\n"
	     "\t len       = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t alignment = 0x%016" PRIx64 "\n",
	     in_addr, len, flags, alignment);

	constexpr size_t PAGE_SIZE              = 0x4000;
	constexpr int    GUEST_MAP_FIXED        = 0x10;
	constexpr int    GUEST_MAP_NO_OVERWRITE = 0x80;

	if (addr == nullptr || len == 0 || (len & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if (alignment != 0 && (alignment & (alignment - 1)) != 0 &&
	    (alignment & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	uint64_t out_addr            = 0;
	bool     range_already_added = false;
	if ((flags & GUEST_MAP_FIXED) != 0) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1)) != 0) {
			return KERNEL_ERROR_EINVAL;
		}
		if ((flags & GUEST_MAP_NO_OVERWRITE) != 0 && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return NoMemory(__func__, __LINE__);
		}
		if (ReplaceFixedRangeWithReserved(in_addr, len)) {
			out_addr            = in_addr;
			range_already_added = true;
		}
	} else {
		alignment = (alignment != 0 ? alignment : PAGE_SIZE);
		out_addr  = FindGuestFreeRange(in_addr, len, alignment);
		if (out_addr != 0) {
			UnmapGpuRange(out_addr, len);
		}
	}

	if (out_addr == 0) {
		return NoMemory(__func__, __LINE__);
	}

	if (!range_already_added &&
	    !g_virtual_ranges->Add(out_addr, len, 0, 0, 0, VirtualRangeType::Reserved, "anon")) {
		return KERNEL_ERROR_EBUSY;
	}

	*addr = reinterpret_cast<void*>(out_addr);

	LOGF("\t out_addr  = 0x%016" PRIx64 "\n", out_addr);

	return OK;
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void TestFailNextPhysicalMemoryUnmap() {
	TestFailPhysicalMemoryUnmapAfter(0);
}

void TestFailPhysicalMemoryUnmapAfter(uint32_t successful_unmaps) {
	g_test_physical_memory_unmaps_before_failure = successful_unmaps;
}

void TestFailGuestBackingStoreUnmapAfter(uint32_t successful_unmaps) {
	g_test_backing_store_unmaps_before_failure = successful_unmaps;
}

void TestFailNextFixedReserveRangeRegistration() {
	g_test_fail_next_fixed_reserve_range_add = true;
}

bool TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space->TestContainsFree(vaddr, size);
}

bool TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space->Owns(vaddr, size);
}

bool TestGuestBackingOutsideAddressSpace() {
	return !g_guest_address_space->OverlapsOwned(g_guest_address_space->GetBackingBase(),
	                                             g_guest_address_space->GetBackingSize());
}

uint64_t TestGuestBackingSize() {
	return g_guest_address_space->GetBackingSize();
}

bool TestGuestFreeRangeBounds() {
	return GuestFreeRangeContains(0x10000, 0x20000, 0x18000, 0x4000) &&
	       !GuestFreeRangeContains(0x10000, 0x20000, 0x40000, 0x4000) &&
	       !GuestFreeRangeContains(UINT64_MAX - 0x1000, 0x2000, UINT64_MAX - 0x800, 0x400);
}
#endif

int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr      = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t info_size = 0x%016" PRIx64 "\n",
	     vaddr, flags, info_size);

	if (info == nullptr || info_size != sizeof(VirtualQueryInfo) || (flags != 0 && flags != 1)) {
		return KERNEL_ERROR_EINVAL;
	}

	VirtualRanges::Range candidate {};
	if (!g_virtual_ranges->Query(vaddr, flags, &candidate)) {
		return KERNEL_ERROR_EACCES;
	}

	std::memset(info, 0, sizeof(VirtualQueryInfo));
	info->start        = candidate.start;
	info->end          = candidate.start + candidate.size;
	info->offset       = candidate.offset;
	info->protection   = candidate.protection;
	info->memory_type  = candidate.memory_type;
	info->is_flexible  = (candidate.type == VirtualRangeType::Flexible ? 1 : 0);
	info->is_direct    = (candidate.type == VirtualRangeType::Direct ? 1 : 0);
	info->is_stack     = (candidate.type == VirtualRangeType::Stack ? 1 : 0);
	info->is_pooled    = (IsPooledRangeType(candidate.type) ? 1 : 0);
	info->is_committed = (IsCommittedRangeType(candidate.type) ? 1 : 0);
	info->is_gpu_prt   = IsInPrtAperture(vaddr);
	CopyVirtualRangeName(info->name, candidate.name);

	static std::atomic<uint32_t> log_count {0};
	if (log_count.fetch_add(1) < 64) {
		LOGF("\t start       = 0x%016" PRIx64 "\n"
		     "\t end         = 0x%016" PRIx64 "\n"
		     "\t offset      = 0x%016" PRIx64 "\n"
		     "\t protection  = 0x%08" PRIx32 "\n"
		     "\t memory_type = %d\n"
		     "\t flexible    = %d\n"
		     "\t direct      = %d\n"
		     "\t name        = %s\n",
		     static_cast<uint64_t>(info->start), static_cast<uint64_t>(info->end), info->offset,
		     info->protection, info->memory_type, static_cast<int>(info->is_flexible),
		     static_cast<int>(info->is_direct), info->name);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n", vaddr);

	VirtualRanges::Range candidate {};
	if (!g_virtual_ranges->Query(vaddr, 0, &candidate)) {
		return KERNEL_ERROR_EACCES;
	}

	uint64_t stack_start = 0;
	uint64_t stack_end   = 0;
	if (candidate.type == VirtualRangeType::Stack) {
		stack_start = candidate.start;
		stack_end   = candidate.start + candidate.size;
	}

	if (start != nullptr) {
		*start = reinterpret_cast<void*>(stack_start);
	}

	if (end != nullptr) {
		*end = reinterpret_cast<void*>(stack_end);
	}

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t end   = 0x%016" PRIx64 "\n",
	     stack_start, stack_end);

	return OK;
}

int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (size == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*size = g_flexible_memory->Available();

	LOGF("\t *size = 0x%016" PRIx64 "\n", *size);

	return OK;
}

int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size) {
	PRINT_NAME();

	if (size == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*size = FlexibleMemory::Size();

	LOGF("\t *size = 0x%016" PRIx64 "\n", *size);

	return OK;
}

static int ProgramProtection(VirtualMemory::Mode mode) {
	const auto protection = static_cast<int>(mode);
	if ((protection & ~(PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC)) != 0) {
		EXIT("unsupported program-memory protection: 0x%08x\n", protection);
	}
	return protection;
}

static std::vector<VirtualRanges::Range> RequireGuestRuntimeMemory(uint64_t vaddr, uint64_t size) {
	std::vector<VirtualRanges::Range> ranges;
	if (g_virtual_ranges == nullptr || !g_virtual_ranges->QuerySpan(vaddr, size, &ranges) ||
	    std::any_of(ranges.begin(), ranges.end(), [](const auto& range) {
		    return range.type != VirtualRangeType::Code && range.type != VirtualRangeType::Runtime;
	    })) {
		EXIT("guest runtime range is not fully mapped: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}
	return ranges;
}

static uint64_t AllocateGuestRuntimeMemory(uint64_t search_addr, uint64_t size,
                                           VirtualMemory::Mode mode, const char* name,
                                           VirtualRangeType type, bool fixed) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	constexpr uint64_t GuestPageSize = 0x4000;
	if (size == 0 || size > UINT64_MAX - (GuestPageSize - 1u) || name == nullptr ||
	    (fixed && (search_addr == 0 || (search_addr & (GuestPageSize - 1u)) != 0))) {
		return 0;
	}
	const auto mapped_size = (size + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	const auto vaddr =
	    fixed ? search_addr : FindGuestFreeRange(search_addr, mapped_size, GuestPageSize);
	if (vaddr == 0 || g_virtual_ranges->HasOverlap(vaddr, mapped_size)) {
		return 0;
	}
	UnmapGpuRange(vaddr, mapped_size);
	if (!g_guest_address_space->Commit(vaddr, mapped_size, mode)) {
		return 0;
	}
	if (!g_virtual_ranges->Add(vaddr, mapped_size, 0, ProgramProtection(mode), 0, type, name)) {
		EXIT_IF(!g_guest_address_space->ReleaseCommitted(vaddr, mapped_size));
		return 0;
	}
	MapGpuRange(vaddr, mapped_size);
	return vaddr;
}

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                               const char* name) {
	return AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Code, false);
}

void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	const auto ranges = RequireGuestRuntimeMemory(vaddr, size);
	if (std::any_of(ranges.begin(), ranges.end(),
	                [](const auto& range) { return range.type != VirtualRangeType::Code; })) {
		EXIT("program-memory range is not fully mapped: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}

	const auto host_mode = VirtualMemory::IsExecute(mode) ? VirtualMemory::Mode::ExecuteReadWrite
	                                                      : VirtualMemory::Mode::ReadWrite;
	EXIT_IF(!g_guest_address_space->Protect(vaddr, size, host_mode));
	g_virtual_ranges->Protect(vaddr, size, ProgramProtection(mode));
}

uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                               const char* name, bool fixed) {
	return AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Runtime,
	                                  fixed);
}

uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                                  const char* name) {
	const auto vaddr =
	    AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Stack, false);
	// (commit-al-tocar, KYTY_COMMIT_AL_TOCAR=1) A stack is committed whole now: a fault on a reserved stack page cannot be
	// handled (Windows writes the exception's frame on that same stack) and ends the process without a report. Fiber
	// stacks (MapGuestStack, never memset) were the ones: the first push of a fiber on its reserved stack, ~20 s.
	if (vaddr != 0) (void)PrefaultGuestRange(reinterpret_cast<void*>(vaddr), size);
	if (vaddr != 0 && g_alloc_callback != nullptr) {
		g_alloc_callback(vaddr, size);
	}
	return vaddr;
}

bool ProtectGuestMemory(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode,
                        VirtualMemory::Mode* old_mode) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);
	constexpr uint64_t                    GuestPageSize = 0x4000;
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - (vaddr & (GuestPageSize - 1u))) {
		return false;
	}
	const auto aligned_addr = vaddr & ~(GuestPageSize - 1u);
	const auto aligned_size =
	    (size + (vaddr - aligned_addr) + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	const auto ranges = RequireGuestRuntimeMemory(aligned_addr, aligned_size);
	if (old_mode != nullptr) {
		*old_mode = static_cast<VirtualMemory::Mode>(
		    ranges.front().protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC));
	}
	if (!g_guest_address_space->Protect(aligned_addr, aligned_size, mode)) {
		return false;
	}
	g_virtual_ranges->Protect(aligned_addr, aligned_size, ProgramProtection(mode));
	return true;
}

#if defined(_WIN32)
// (11-14) Watched pages pinned in RAM (KYTY_PIN_WATCHED=1 turns it on; KYTY_PIN_WATCHED_MB, default 6144: at most that
// much). Every reboot measured on 11-14 was bug check 0x1A / 0x41202: Windows paging a page out while its protection
// changes. The page manager changes protections tens of thousands of times a minute; when RAM runs low, Windows trims
// those very pages. A page is locked (VirtualLock) the first time its protection is changed, while still accessible,
// and stays locked until the game frees it: Windows can no longer page it, so the race cannot happen.
namespace PinWatched {
static std::mutex                   g_mutex;
static std::unordered_set<uint64_t> g_granules; // 64 KiB granules tried (locked or refused)
static uint64_t                     g_pinned = 0, g_budget = 0, g_refused = 0;
static bool                         g_on = false, g_full_logged = false;
static std::once_flag               g_once;
constexpr uint64_t                  Granule = uint64_t {1} << 16u;

static void Init() {
	const char* off = std::getenv("KYTY_PIN_WATCHED");
	g_on = off != nullptr && off[0] == '1'; // (11-14) off by default: it filled the RAM (6 GB pinned at once)
	if (!g_on) {
		std::printf("Pin watched pages: off (KYTY_PIN_WATCHED=0)\n");
		std::fflush(stdout);
		return;
	}
	const char* mb = std::getenv("KYTY_PIN_WATCHED_MB");
	g_budget       = (mb != nullptr && *mb != 0 ? std::strtoull(mb, nullptr, 10) : 6144ull) << 20u;
	SIZE_T wmin = 0, wmax = 0;
	GetProcessWorkingSetSize(GetCurrentProcess(), &wmin, &wmax);
	const BOOL ok = SetProcessWorkingSetSizeEx(GetCurrentProcess(), wmin + g_budget + (256ull << 20u),
	                                           std::max<SIZE_T>(wmax, wmin + g_budget + (1024ull << 20u)),
	                                           QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
	std::printf("Pin watched pages: on, up to %llu MiB locked (working set minimum %s; KYTY_PIN_WATCHED=1 turns it on)\n",
	            static_cast<unsigned long long>(g_budget >> 20u), ok ? "raised" : "NOT raised");
	std::fflush(stdout);
}

static void Before(uint64_t vaddr, uint64_t size) {
	std::call_once(g_once, Init);
	if (!g_on || size == 0) return;
	std::lock_guard lock(g_mutex);
	for (uint64_t g = vaddr & ~(Granule - 1u); g < vaddr + size; g += Granule) {
		if (!g_granules.insert(g).second) continue;
		if (g_pinned + Granule > g_budget) {
			if (!g_full_logged) {
				g_full_logged = true;
				std::printf("Pin watched pages: budget of %llu MiB used up; later watched pages are not pinned\n",
				            static_cast<unsigned long long>(g_budget >> 20u));
				std::fflush(stdout);
			}
			continue;
		}
		if (VirtualLock(reinterpret_cast<void*>(g), Granule) != 0) {
			g_pinned += Granule;
		} else {
			g_refused++;
		}
	}
}

// The game freed [vaddr, vaddr + size): its locks went with the mapping.
static void Forget(uint64_t vaddr, uint64_t size) {
	if (!g_on || size == 0) return;
	std::lock_guard lock(g_mutex);
	for (uint64_t g = vaddr & ~(Granule - 1u); g < vaddr + size; g += Granule) {
		if (g_granules.erase(g) != 0 && g_pinned >= Granule) g_pinned -= Granule;
	}
}
} // namespace PinWatched
#endif

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
#if defined(_WIN32)
	PinWatched::Before(vaddr, size);
#endif
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->ProtectTransient(vaddr, size, mode);
}

bool FreeGuestMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	constexpr uint64_t GuestPageSize = 0x4000;
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - (GuestPageSize - 1u)) {
		return false;
	}
	const auto mapped_size = (size + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	(void)RequireGuestRuntimeMemory(vaddr, mapped_size);
	UnmapGpuRange(vaddr, mapped_size, true);
	const bool freed = FreeGuestMemoryOwner(vaddr, mapped_size);
#if defined(_WIN32)
	if (freed) PinWatched::Forget(vaddr, mapped_size);
#endif
	if (!freed) {
		RestoreGpuRangeProtection(vaddr, mapped_size);
	}
	return freed;
}

int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n",
	     vaddr, static_cast<uint64_t>(len));

	constexpr uint64_t PAGE_SIZE    = 0x4000;
	auto               aligned_addr = vaddr & ~(PAGE_SIZE - 1);
	const auto         page_offset  = vaddr - aligned_addr;
	if (len > UINT64_MAX - page_offset || len + page_offset > UINT64_MAX - (PAGE_SIZE - 1)) {
		EXIT("memory-protection range overflows: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, static_cast<uint64_t>(len));
	}
	auto aligned_len = (len + page_offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

	if (aligned_len == 0) {
		return OK;
	}

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		return KERNEL_ERROR_EINVAL;
	}
	// (Fresh memory: bytes the CPU may write from now on without a write watch count as written, before it can.)
	if ((prot & PROT_CPU_WRITE) != 0) FreshMemory::MarkGuestRange(aligned_addr, aligned_len);
	std::vector<VirtualRanges::Range> old_ranges;
	if (!g_virtual_ranges->QuerySpan(aligned_addr, aligned_len, &old_ranges) ||
	    std::any_of(old_ranges.begin(), old_ranges.end(),
	                [](const auto& range) { return !IsCommittedRangeType(range.type); })) {
		EXIT("memory-protection range is not fully mapped: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     "\n",
		     aligned_addr, aligned_len);
	}
	const auto old_mode = static_cast<VirtualMemory::Mode>(
	    old_ranges.front().protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC));
	bool ok = g_guest_address_space->ProtectKeepingWatched(aligned_addr, aligned_len, mode);

	if (!ok) {
		EXIT("host memory-protection update failed: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     " prot=0x%08x\n",
		     aligned_addr, aligned_len, prot);
	}
	// A writable page the GPU tracker watches would stop reporting CPU writes.
	if (g_gpu_resources != nullptr && IsGpuAddressRange(aligned_addr, aligned_len)) {
		g_gpu_resources->ReapplyProtection(aligned_addr, aligned_len);
	}
	for (const auto& old_range: old_ranges) {
		if (old_range.type == VirtualRangeType::Direct) {
			g_physical_memory->ProtectMapping(old_range.start, old_range.size, prot, mode,
			                                  gpu_mode);
		} else if (old_range.type == VirtualRangeType::Flexible) {
			g_flexible_memory->Protect(old_range.start, old_range.size, prot, mode, gpu_mode);
		}
	}
	g_virtual_ranges->Protect(aligned_addr, aligned_len, prot);

	LOGF("\t prot: %s -> %s\n", Common::EnumName(old_mode).c_str(), Common::EnumName(mode).c_str());

	return OK;
}

int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n"
	     "\t type = 0x%08" PRIx32 "\n"
	     "\t prot = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(len), static_cast<uint32_t>(type),
	     static_cast<uint32_t>(prot));

	return KernelMprotect(addr, len, prot);
}

int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags) {
	KYTY_MAPA_GRANDE_SCOPE("sceKernelBatchMap");
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t entries         = %p\n"
	     "\t num_entries     = %d\n"
	     "\t num_entries_out = %p\n"
	     "\t flags           = 0x%08" PRIx32 "\n",
	     static_cast<void*>(entries), num_entries, static_cast<void*>(num_entries_out),
	     static_cast<uint32_t>(flags));

	enum MemoryOpTypes {
		MAP_OP_MAP_DIRECT   = 0,
		MAP_OP_UNMAP        = 1,
		MAP_OP_PROTECT      = 2,
		MAP_OP_MAP_FLEXIBLE = 3,
		MAP_OP_TYPE_PROTECT = 4,
	};

	if (entries == nullptr || num_entries < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	int processed = 0;

	for (int i = 0; i < num_entries; i++, processed++) {
		auto* entry = &entries[i];

		LOGF("\t [%d] start = %p, offset = 0x%016" PRIx64 ", length = 0x%016" PRIx64
		     ", prot = 0x%02" PRIx32 ", type = 0x%02" PRIx32 ", op = %d\n",
		     i, entry->start, entry->offset, entry->length,
		     static_cast<uint32_t>(entry->protection), static_cast<uint32_t>(entry->type),
		     entry->operation);

		if (entry->length == 0 || entry->operation < MAP_OP_MAP_DIRECT ||
		    entry->operation > MAP_OP_TYPE_PROTECT) {
			break;
		}

		int result = OK;
		switch (entry->operation) {
			case MAP_OP_MAP_DIRECT:
				result = KernelMapNamedDirectMemory(&entry->start, entry->length, entry->protection,
				                                    flags, static_cast<int64_t>(entry->offset), 0,
				                                    "anon");
				break;
			case MAP_OP_UNMAP:
				result = KernelMunmap(reinterpret_cast<uint64_t>(entry->start), entry->length);
				break;
			case MAP_OP_PROTECT:
				result = KernelMprotect(entry->start, entry->length, entry->protection);
				break;
			case MAP_OP_MAP_FLEXIBLE:
				result = KernelMapNamedFlexibleMemory(&entry->start, entry->length,
				                                      entry->protection, flags, "anon");
				break;
			case MAP_OP_TYPE_PROTECT:
				result =
				    KernelMtypeprotect(entry->start, entry->length, entry->type, entry->protection);
				break;
			default: result = KERNEL_ERROR_EINVAL; break;
		}

		if (result != OK) {
			if (num_entries_out != nullptr) {
				*num_entries_out = processed;
			}
			return result;
		}
	}

	if (num_entries_out != nullptr) {
		*num_entries_out = processed;
	}

	return (processed == num_entries ? OK : KERNEL_ERROR_EINVAL);
}

int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out) {
	constexpr int GUEST_MAP_FIXED = 0x10;
	return KernelBatchMap2(entries, num_entries, num_entries_out, GUEST_MAP_FIXED);
}

static bool IsAligned(uint64_t value, uint64_t alignment) {
	return alignment == 0 || (value & (alignment - 1u)) == 0;
}

static bool IsPowerOfTwo(uint64_t value) {
	return value != 0 && (value & (value - 1u)) == 0;
}

static void MemoryPoolSubtractCommitted(uint64_t len) {
	auto current = g_memory_pool_committed.load(std::memory_order_relaxed);
	while (current != 0) {
		const auto next = (current > len ? current - len : 0);
		if (g_memory_pool_committed.compare_exchange_weak(current, next,
		                                                  std::memory_order_relaxed)) {
			return;
		}
	}
}

int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	constexpr uint64_t POOL_PAGE_SIZE = 0x10000;
	if (search_start < 0 || search_end <= search_start || len == 0 ||
	    (len & (POOL_PAGE_SIZE - 1u)) != 0 || phys_addr_out == nullptr ||
	    (alignment != 0 &&
	     (!IsPowerOfTwo(alignment) || (alignment & (POOL_PAGE_SIZE - 1u)) != 0))) {
		return KERNEL_ERROR_EINVAL;
	}
	if (static_cast<uint64_t>(search_end - search_start) < len) {
		return NoMemory(__func__, __LINE__);
	}

	const auto effective_alignment = (alignment != 0 ? alignment : POOL_PAGE_SIZE);
	uint64_t   phys_addr           = 0;
	if (!g_physical_memory->Alloc(static_cast<uint64_t>(search_start),
	                              static_cast<uint64_t>(search_end), len, effective_alignment,
	                              &phys_addr, 0, true)) {
		return NoMemory(__func__, __LINE__);
	}

	g_pooled_memory->Expand(phys_addr, len);
	*phys_addr_out = static_cast<int64_t>(phys_addr);

	LOGF("\t search_start = 0x%016" PRIx64 "\n"
	     "\t search_end   = 0x%016" PRIx64 "\n"
	     "\t len          = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n"
	     "\t phys_addr    = 0x%016" PRIx64 "\n",
	     search_start, search_end, static_cast<uint64_t>(len), static_cast<uint64_t>(alignment),
	     phys_addr);
	return OK;
}

int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t addr_in   = 0x%016" PRIx64 "\n"
	     "\t len       = 0x%016" PRIx64 "\n"
	     "\t alignment = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t addr_out  = %p\n",
	     reinterpret_cast<uint64_t>(addr_in), static_cast<uint64_t>(len),
	     static_cast<uint64_t>(alignment), static_cast<uint32_t>(flags),
	     static_cast<void*>(addr_out));

	constexpr uint64_t POOL_RESERVE_ALIGNMENT = 0x200000;

	if (addr_out == nullptr || len == 0 || !IsAligned(len, POOL_RESERVE_ALIGNMENT)) {
		return KERNEL_ERROR_EINVAL;
	}
	if (alignment != 0 &&
	    (!IsPowerOfTwo(alignment) || !IsAligned(alignment, POOL_RESERVE_ALIGNMENT))) {
		return KERNEL_ERROR_EINVAL;
	}

	void*      out_addr          = addr_in;
	const auto reserve_alignment = (alignment != 0 ? alignment : POOL_RESERVE_ALIGNMENT);
	const int  ret = KernelReserveVirtualRange(&out_addr, len, flags, reserve_alignment);
	if (ret == OK) {
		// Retagging a valid reservation also removes and re-adds its logical span.
		// KernelReserveVirtualRange has already completed any required GPU drain.
		std::lock_guard residency_lock(g_prt_residency_mutex);
		const auto           out_vaddr = reinterpret_cast<uint64_t>(out_addr);
		VirtualRanges::Range reserved_range {};
		if (!g_virtual_ranges->Query(out_vaddr, 0, &reserved_range) ||
		    reserved_range.start != out_vaddr || reserved_range.size != len ||
		    !IsReservedRangeType(reserved_range.type)) {
			return KERNEL_ERROR_EBUSY;
		}
		g_virtual_ranges->Remove(out_vaddr, len);
		if (!g_virtual_ranges->Add(out_vaddr, len, 0, 0, 0, VirtualRangeType::PoolReserved,
		                           reserved_range.name)) {
			g_virtual_ranges->Add(out_vaddr, len, reserved_range.offset, reserved_range.protection,
			                      reserved_range.memory_type, reserved_range.type,
			                      reserved_range.name);
			return KERNEL_ERROR_EBUSY;
		}
		*addr_out = out_addr;
		LOGF("\t out_addr  = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(out_addr));
	}

	return ret;
}

int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags) {
	KYTY_MAPA_GRANDE_SCOPE("sceKernelMemoryPoolCommit");
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n"
	     "\t type  = 0x%08" PRIx32 "\n"
	     "\t prot  = 0x%08" PRIx32 "\n"
	     "\t flags = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(len), static_cast<uint32_t>(type),
	     static_cast<uint32_t>(prot), static_cast<uint32_t>(flags));

	constexpr uint64_t POOL_COMMIT_ALIGNMENT = 0x10000;
	constexpr int      PROT_CPU_EXEC         = 0x04;

	if (addr == nullptr || len == 0 || !IsAligned(len, POOL_COMMIT_ALIGNMENT) ||
	    (prot & PROT_CPU_EXEC) != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;
	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		return KERNEL_ERROR_EINVAL;
	}

	const auto vaddr = reinterpret_cast<uint64_t>(addr);

	VirtualRanges::Range old_range {};
	if (!g_virtual_ranges->Query(vaddr, 0, &old_range) ||
	    old_range.type != VirtualRangeType::PoolReserved) {
		return KERNEL_ERROR_EACCES;
	}

	std::unique_lock residency_lock(g_prt_residency_mutex);
	if (!g_virtual_ranges->ConsumeReserved(vaddr, len, VirtualRangeType::PoolReserved)) {
		return KERNEL_ERROR_EACCES;
	}

	std::vector<PooledMemory::Mapping> mappings;
	if (!g_pooled_memory->Allocate(vaddr, len, gpu_mode, &mappings)) {
		g_virtual_ranges->Add(vaddr, len, 0, 0, 0, VirtualRangeType::PoolReserved, old_range.name);
		return NoMemory(__func__, __LINE__);
	}

	std::vector<PooledMemory::Mapping> mapped;
	auto                               rollback = [&]() {
		if (!UnmapPooledBackingTransactional(mapped, mode)) {
			EXIT("failed to roll back pooled-memory backing maps\n");
		}
		GpuAccessMode rollback_gpu_mode = GpuAccessMode::NoAccess;
		if (!g_pooled_memory->Release(vaddr, len, &rollback_gpu_mode)) {
			EXIT("failed to release pooled-memory rollback allocation\n");
		}
		g_virtual_ranges->Add(vaddr, len, 0, 0, 0, VirtualRangeType::PoolReserved, old_range.name);
	};

	for (const auto& mapping: mappings) {
		auto       failure_reason = GuestBackingStore::FailureReason::None;
		const bool ok = g_guest_address_space->MapBacking(mapping.vaddr, mapping.size,
		                                                  mapping.phys_addr, mode, &failure_reason);
		if (!ok) {
			LOGF_COLOR(Log::Color::Red, "\t pool backing map failed: %s\n",
			           GuestBackingStore::GetFailureReasonName(failure_reason));
			rollback();
			return NoMemory(__func__, __LINE__);
		}
		mapped.push_back(mapping);
	}

	if (!g_virtual_ranges->Add(vaddr, len, 0, prot, type, VirtualRangeType::Pooled,
	                           old_range.name)) {
		rollback();
		return KERNEL_ERROR_EBUSY;
	}
	residency_lock.unlock();

	MapGpuRange(vaddr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(vaddr, len);
	}

	g_memory_pool_committed.fetch_add(len, std::memory_order_relaxed);

	return OK;
}

static int DecommitPoolRange(uint64_t vaddr, size_t len, const VirtualRanges::Range& old_range);

static int DecommitMemoryPoolRange(uint64_t vaddr, size_t len) {
	// The virtual ranges it spans, one after the other (as UnmapMemoryRange: not a call level each).
	for (;;) {
		VirtualRanges::Range old_range {};
		if (!g_virtual_ranges->Query(vaddr, 0, &old_range)) {
			return KERNEL_ERROR_EACCES;
		}
		const auto chunk_len = std::min<uint64_t>(len, old_range.size - (vaddr - old_range.start));
		if (chunk_len == 0) {
			return KERNEL_ERROR_EINVAL;
		}
		// (A reserved part has nothing committed.)
		if (old_range.type != VirtualRangeType::PoolReserved) {
			if (old_range.type != VirtualRangeType::Pooled) {
				return KERNEL_ERROR_EACCES;
			}
			if (const int ret = DecommitPoolRange(vaddr, chunk_len, old_range); ret != OK) {
				return ret;
			}
		}
		if (chunk_len == len) {
			return OK;
		}
		vaddr += chunk_len;
		len -= chunk_len;
	}
}

// [vaddr, vaddr + len) of one pooled virtual range.
static int DecommitPoolRange(uint64_t vaddr, size_t len, const VirtualRanges::Range& old_range) {
	std::vector<PooledMemory::Mapping> mappings;
	if (!g_pooled_memory->Query(vaddr, len, &mappings)) {
		return KERNEL_ERROR_EACCES;
	}

	VirtualMemory::Mode mode        = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       decoded_gpu = GpuAccessMode::NoAccess;
	if (!DecodeMemoryProtection(old_range.protection, &mode, &decoded_gpu)) {
		return KERNEL_ERROR_EACCES;
	}
	// KernelMemoryPoolDecommit already drained GPU work before entering here.
	std::unique_lock residency_lock(g_prt_residency_mutex);
	if (!UnmapPooledBackingTransactional(mappings, mode)) {
		EXIT("pooled-memory backing transaction failed after GPU unmap: addr=0x%016" PRIx64
		     " size=0x%016" PRIx64 "\n",
		     vaddr, len);
	}

	GpuAccessMode gpu_mode = GpuAccessMode::NoAccess;
	if (!g_pooled_memory->Release(vaddr, len, &gpu_mode)) {
		EXIT("failed to release decommitted pooled-memory range\n");
	}

	g_virtual_ranges->Remove(vaddr, len);
	g_virtual_ranges->Add(vaddr, len, 0, 0, 0, VirtualRangeType::PoolReserved, old_range.name);
	residency_lock.unlock();

	if (g_free_callback != nullptr) {
		g_free_callback(vaddr, len);
	}

	MemoryPoolSubtractCommitted(len);
	return OK;
}

int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n"
	     "\t flags = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(len),
	     static_cast<uint32_t>(flags));

	constexpr uint64_t POOL_COMMIT_ALIGNMENT = 0x10000;

	if (addr == nullptr || len == 0 || !IsAligned(len, POOL_COMMIT_ALIGNMENT)) {
		return KERNEL_ERROR_EINVAL;
	}

	const auto vaddr = reinterpret_cast<uint64_t>(addr);
	if (UINT64_MAX - vaddr < len) {
		return KERNEL_ERROR_EINVAL;
	}
	const auto end  = vaddr + len;
	auto       scan = vaddr;
	while (scan < end) {
		VirtualRanges::Range scan_range {};
		if (!g_virtual_ranges->Query(scan, 0, &scan_range) ||
		    (scan_range.type != VirtualRangeType::Pooled &&
		     scan_range.type != VirtualRangeType::PoolReserved)) {
			return KERNEL_ERROR_EACCES;
		}
		const auto next = std::min(end, scan_range.start + scan_range.size);
		if (next <= scan) {
			return KERNEL_ERROR_EACCES;
		}
		scan = next;
	}

	UnmapGpuRange(vaddr, len, true);
	const int result = DecommitMemoryPoolRange(vaddr, len);
	if (result != OK) {
		RestoreGpuRangeProtection(vaddr, len);
	}
	return result;
}

int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (entries == nullptr || num_entries < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	enum MemoryPoolOp {
		POOL_OP_COMMIT       = 1,
		POOL_OP_DECOMMIT     = 2,
		POOL_OP_PROTECT      = 3,
		POOL_OP_TYPE_PROTECT = 4,
		POOL_OP_MOVE         = 5,
	};

	int processed = 0;
	int result    = OK;

	for (int i = 0; i < num_entries; i++, processed++) {
		const auto& entry = entries[i];
		switch (entry.op) {
			case POOL_OP_COMMIT:
				result = KernelMemoryPoolCommit(entry.commit.addr, entry.commit.len,
				                                entry.commit.type, entry.commit.prot, entry.flags);
				break;
			case POOL_OP_DECOMMIT:
				result =
				    KernelMemoryPoolDecommit(entry.decommit.addr, entry.decommit.len, entry.flags);
				break;
			case POOL_OP_PROTECT:
				result = KernelMprotect(entry.protect.addr, entry.protect.len, entry.protect.prot);
				break;
			case POOL_OP_TYPE_PROTECT:
				result = KernelMtypeprotect(entry.type_protect.addr, entry.type_protect.len,
				                            entry.type_protect.type, entry.type_protect.prot);
				break;
			case POOL_OP_MOVE:
			default: result = KERNEL_ERROR_EINVAL; break;
		}

		if (result != OK) {
			break;
		}
	}

	if (num_entries_out != nullptr) {
		*num_entries_out = processed;
	}

	(void)flags;
	return result;
}

int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (output == nullptr && output_size != 0) {
		return KERNEL_ERROR_EFAULT;
	}

	KernelMemoryPoolBlockStats stats {};
	constexpr uint64_t         BLOCK_SIZE = 0x10000;
	const uint64_t             committed  = g_memory_pool_committed.load(std::memory_order_relaxed);
	const uint64_t available = (g_pooled_memory != nullptr ? g_pooled_memory->Available() : 0);

	stats.available_flushed_blocks = static_cast<int32_t>(available / BLOCK_SIZE);
	stats.available_cached_blocks  = 0;
	stats.allocated_flushed_blocks = static_cast<int32_t>(committed / BLOCK_SIZE);
	stats.allocated_cached_blocks  = 0;

	const auto copy_size = std::min(output_size, sizeof(stats));
	if (copy_size != 0) {
		std::memcpy(output, &stats, copy_size);
	}

	return OK;
}

} // namespace Libs::LibKernel::Memory
