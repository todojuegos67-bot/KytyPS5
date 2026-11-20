#include "ram-guard.h"

#include "fresh-memory.h"
#include "gpu-residency.h"
#include "guest-lazy-commit.h"
#include "local-platform.h"
#include "mem-guard.h"
#include "mem1.h" // (mem1: KYTY_MEM1)
#include "soft-release.h"
#include "texture-offload.h" // (ram16: the native offload's numbers for the "[ram16]" line)

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// (After windows.h.)
#include <psapi.h>
#include <intrin.h>
#endif

namespace Libs::LibKernel::Memory {
void ReportGuestAccess(bool force);
void RamLineStats(uint64_t* out, size_t count); // (ram2: memory.cpp, the "[ram]" line's guest counters)
void RamLineFreedRelease(char* out, size_t size); // (ram6: memory.cpp, KYTY_DS_FREED_RELEASE; "" when off)
uint64_t Ram16FreedReleasedNow(); // (ram16: memory.cpp, KYTY_DS_FREED_RELEASE=1's bytes given back now)
}
namespace Loader::DemonsSoulsMemMap {
// (ram16: loader/demonsSoulsMemMap.cpp) The last snapshot's ranges of the labels asked (lock-free, no guest read).
size_t RangesOf(const char* const* labels, size_t label_count, uint64_t* out, size_t max_pairs) noexcept;
} // namespace Loader::DemonsSoulsMemMap
namespace Libs::Graphics {
// (ram4: vma.cpp) The emulator's own Vulkan memory blocks by heap kind, and the KYTY_SIMULATE_VRAM_MB ballast, bytes.
bool RamLineVulkan(uint64_t* video_blocks, uint64_t* system_blocks, uint64_t* ballast);
} // namespace Libs::Graphics
namespace LocalPlatform {
// (ram4: local-platform.cpp) The process heaps as the last "Private breakdown" walk saw them.
void RamLineHeap(uint64_t* out, size_t count);
} // namespace LocalPlatform
namespace RamGuard {

namespace {

constexpr uint64_t MiB = uint64_t {1} << 20u;
constexpr uint64_t GiB = uint64_t {1} << 30u;

std::atomic<int>      g_level {None};
std::atomic<uint64_t> g_available {0};
std::atomic<uint64_t> g_released_count[ReleasedKinds] {};
std::atomic<uint64_t> g_released_bytes[ReleasedKinds] {};
std::atomic<uint64_t> g_released_count_total[ReleasedKinds] {};
std::atomic<uint64_t> g_released_bytes_total[ReleasedKinds] {};

std::atomic<const void*> g_store_alias {nullptr};
std::atomic<uint64_t>    g_store_size {0};
std::atomic<int>         g_store_charge {static_cast<int>(StoreCharge::AllCommitted)};

std::mutex                                           g_flush_mutex;
std::vector<std::pair<const char*, uint64_t (*)(int)>> g_flushes;
struct BudgetFlush {
	const char*   name;
	FlushClass    kind;
	BudgetFlushFn fn;
};
std::vector<BudgetFlush> g_budget_flushes;
std::atomic<bool>        g_periodic {false};
std::atomic<uint64_t>    g_mem1_ring {0}; // (mem1: the upload ring's bytes KYTY_MEM1 took off, 256 -> 128 MiB; 0 when set)

const char* Env(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

[[maybe_unused]] uint64_t EnvMiB(const char* name, uint64_t fallback_bytes) {
	const char* value = Env(name);
	return value != nullptr ? std::strtoull(value, nullptr, 10) * MiB : fallback_bytes;
}

bool SetDefault(const char* name, const char* value, std::string* set) {
	if (Env(name) != nullptr) return false;
#if defined(_WIN32)
	if (_putenv_s(name, value) != 0) return false;
	(void)SetEnvironmentVariableA(name, value);
#else
	if (setenv(name, value, 0) != 0) return false;
#endif
	if (!set->empty()) *set += ' ';
	*set += name;
	*set += '=';
	*set += value;
	return true;
}

[[maybe_unused]] const char* LevelName(int level) {
	return level == Critical ? "critical" : level == Low ? "low" : "none";
}

} // namespace

bool LowRam() {
	static const bool on = [] {
		if (const char* value = Env("KYTY_LOWRAM"); value != nullptr) return value[0] == '1';
		const uint64_t ram = LocalPlatform::PhysicalMemory();
		return ram != 0 && ram <= 20 * GiB;
	}();
	return on;
}

#if defined(_WIN32)
namespace {
std::atomic<uint64_t> g_ram_ballast {0};
// (11-12) KYTY_SIMULATE_RAM_MB=<n> on a bigger PC: hold (physical - n MiB - 1 GiB margin for the bigger PC's own
// OS headroom is NOT kept: the rest of the machine behaves as an n MiB PC) of real RAM, committed, touched and locked,
// so the game and Windows see the memory pressure of an n MiB PC, as the VRAM ballast does for KYTY_SIMULATE_VRAM_MB.
void HoldRamBallast() {
	const char* text = std::getenv("KYTY_SIMULATE_RAM_MB");
	if (text == nullptr || *text == '\0') return;
	const uint64_t want = std::strtoull(text, nullptr, 10) << 20u;
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (want == 0 || GlobalMemoryStatusEx(&status) == 0 || status.ullTotalPhys <= want) return;
	uint64_t ballast = status.ullTotalPhys - want;
	// Never more than what is free now less 2 GiB (the game must still start).
	if (status.ullAvailPhys < (2ull << 30u)) return;
	ballast = std::min<uint64_t>(ballast, status.ullAvailPhys - (2ull << 30u));
	SIZE_T wmin = 0, wmax = 0;
	GetProcessWorkingSetSize(GetCurrentProcess(), &wmin, &wmax);
	SetProcessWorkingSetSizeEx(GetCurrentProcess(), wmin + ballast + (64ull << 20u), wmax + ballast + (256ull << 20u),
	                           QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
	constexpr uint64_t kChunk = 256ull << 20u;
	uint64_t held = 0, locked = 0;
	for (; held + kChunk <= ballast; held += kChunk) {
		void* p = VirtualAlloc(nullptr, kChunk, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
		if (p == nullptr) break;
		for (uint64_t o = 0; o < kChunk; o += 4096) static_cast<volatile char*>(p)[o] = 1;
		if (VirtualLock(p, kChunk) != 0) locked += kChunk;
	}
	g_ram_ballast = held;
	std::printf("RAM simulation: %llu MiB PC on %llu MiB: %llu MiB of RAM held (%llu MiB locked)\n",
	            static_cast<unsigned long long>(want >> 20u), static_cast<unsigned long long>(status.ullTotalPhys >> 20u),
	            static_cast<unsigned long long>(held >> 20u), static_cast<unsigned long long>(locked >> 20u));
	std::fflush(stdout);
}
} // namespace
#endif

uint64_t RamBallast() {
#if defined(_WIN32)
	return g_ram_ballast.load();
#else
	return 0;
#endif
}

// (ram16) KYTY_RAM16=1 (off by default, read once, before anything reads the switches it sets): the RAM cuts measured to
// save RAM without lowering quality, as one switch, and the "[ram16]" line (every 5 s: what they save now) with a census
// of the game's memory in RAM (every KYTY_RAM16_CENSUS_S, default 30; 0 off). Each cut keeps its own switch, which wins
// when set (=0 turns it off):
//   KYTY_NATIVE_TEXTURE_OFFLOAD=1 (ram15): clean sampled textures' guest bytes out of RAM at their guest address, given
//     back from the image on any touch. Measured (11-16 07:41, 16 GB, 8 GB video memory simulated): 0.47-0.56 GB out of
//     RAM, 0 missed, its restores 2.0 s in 439 over the run (at most 9.6 ms, on the offload worker for evictions); the
//     650 ms frame of that run was a pipeline build (dispatch.pipeline 105 ms, programs 127 ms), not a restore.
//   KYTY_DS_FREED_RELEASE=1 (ram14): the game's free-block cache given back to Windows. Measured 0.41-0.57 GB.
//   KYTY_RAM16_RETAKE_S=60 (texture-offload.h): a texture touched once taken again after 60 s clean (140 MiB measured
//     held in RAM by the "never again" rule).
//   KYTY_PRIVATE_RESIDENCY=1 (ram3, local-platform.cpp): the census of the emulator's own allocations in RAM, a sample
//     per MiB every 60 s on a lowest-priority thread ("Private residency" line).
// Only does something with native guest memory (KYTY_RAIZ=1 / KYTY_NATIVE_MEMORY=1) for the textures; the freed release
// works with both.
static bool Ram16On() {
	static const bool on = [] {
		const char* v = Env("KYTY_RAM16");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}
static void StartRam16Line(); // (below: the "[ram16]" line's thread)
static void StartRam18();     // (ram18, below: the cold-range thread)
static void StartMem1();      // (mem1, below: the heap passes and the VirtualMemo census)

void ApplyLowRamDefaults() {
	static std::once_flag once;
	std::call_once(once, [] {
		// (mem1) KYTY_MEM1=1 (mem1.h): implies KYTY_RAM16=1 (the native offload's window hook stamps VirtualMemo, the
		// "[ram16]" thread prints "[mem1]") and sets the upload ring to 128 MiB unless KYTY_STAGING_RING_MB is set.
		std::string mem1_set;
		if (Mem1::On()) {
			(void)Mem1::Epoch();
			SetDefault("KYTY_RAM16", "1", &mem1_set);
			if (SetDefault("KYTY_STAGING_RING_MB", "128", &mem1_set)) g_mem1_ring.store(128 * MiB, std::memory_order_relaxed);
		}
		// (ram18) KYTY_RAM18=1: the game's cold memory to the disk store (texture-offload.h); implies KYTY_RAM17=1.
		const char* ram18    = Env("KYTY_RAM18");
		const bool  ram18_on = ram18 != nullptr && ram18[0] == '1';
		std::string ram18_set;
		if (ram18_on) SetDefault("KYTY_RAM17", "1", &ram18_set);
		// (ram20) KYTY_RAM20=1: the game's memory with a copy in video memory to the disk store and out of RAM (the RAM18
		// cold-range path, texture-offload.h); implies KYTY_RAM17=1 (and RAM16), and the "[ram19]" lines (gpu-residency.h).
		const char* ram20    = Env("KYTY_RAM20");
		const bool  ram20_on = ram20 != nullptr && ram20[0] == '1' && ram20[1] == '\0';
		std::string ram20_set;
		if (ram20_on) SetDefault("KYTY_RAM17", "1", &ram20_set);
		// (ram1b) KYTY_RAM_NATIVE_TEX=1: the guest copy of the arrays' layers and the render targets to the disk store (the
		// RAM18 cold-range path, texture-offload.h); implies KYTY_RAM17=1 (and RAM16). =2 drops nothing.
		const bool  rnt_on = TextureOffload::RntAsked();
		std::string rnt_set;
		if (rnt_on) SetDefault("KYTY_RAM17", "1", &rnt_set);
		// (ram17) KYTY_RAM17=1: the disk store of the native offload (texture-offload.h); implies KYTY_RAM16=1.
		const char* ram17 = Env("KYTY_RAM17");
		const bool  ram17_on = ram17 != nullptr && ram17[0] == '1';
		std::string ram17_set;
		if (ram17_on) SetDefault("KYTY_RAM16", "1", &ram17_set);
		if (Ram16On()) { // (ram16: first, whatever the RAM)
			std::string ram16;
			SetDefault("KYTY_NATIVE_TEXTURE_OFFLOAD", "1", &ram16);
			SetDefault("KYTY_DS_FREED_RELEASE", "1", &ram16);
			SetDefault("KYTY_PRIVATE_RESIDENCY", "1", &ram16);
			std::printf("RAM16: on (KYTY_RAM16=1): %s; textures touched once taken again after %d s clean "
			            "(KYTY_RAM16_RETAKE_S); \"[ram16]\" line every 5 s, census every %s s (KYTY_RAM16_CENSUS_S)\n",
			            ram16.empty() ? "every switch already set" : ram16.c_str(), TextureOffload::Ram16RetakeSeconds(),
			            Env("KYTY_RAM16_CENSUS_S") != nullptr ? Env("KYTY_RAM16_CENSUS_S") : "30");
			if (ram17_on)
				std::printf("RAM17: on (KYTY_RAM17=1%s%s): textures taken written to the disk store first; images gone leave "
				            "their bytes there (no download), any touch reads them back from it; read back = held in RAM %s s "
				            "(KYTY_RAM17_HOLD_S); \"[ram17]\" line every 5 s\n",
				            ram17_set.empty() ? "" : ", ", ram17_set.c_str(),
				            Env("KYTY_RAM17_HOLD_S") != nullptr ? Env("KYTY_RAM17_HOLD_S") : "30");
			if (ram18_on) {
				std::printf("RAM18: asked (KYTY_RAM18=1%s%s; works with the native offload and KYTY_RAM17): game memory "
				            "Windows found cold (out of the working set %s s, KYTY_RAM18_COLD_S) in %s (KYTY_RAM18_LABELS) written to the disk store and dropped, read back on "
				            "any touch; up to %s MiB a pass (KYTY_RAM18_MB); \"[ram18]\" line every 5 s\n",
				            ram18_set.empty() ? "" : ", ", ram18_set.c_str(),
				            Env("KYTY_RAM18_COLD_S") != nullptr ? Env("KYTY_RAM18_COLD_S") : "10",
				            Env("KYTY_RAM18_LABELS") != nullptr ? Env("KYTY_RAM18_LABELS") : "VirtualMemo",
				            Env("KYTY_RAM18_MB") != nullptr ? Env("KYTY_RAM18_MB") : "256");
				StartRam18();
			}
			if (ram20_on) {
				std::printf("RAM20: asked (KYTY_RAM20=1%s%s; works with the native offload and KYTY_RAM17): game memory with a copy "
				            "in video memory (last GPU residency pass) written to the disk store and dropped - out of RAM and of "
				            "Windows' compression -, read back on any touch; cold (out of the working set %s s, KYTY_RAM20_COLD_S) in "
				            "every allocator, in RAM too for images of the GPU-only zones (KYTY_RAM20_WARM=0 off; buffers: "
				            "KYTY_RAM20_WARM_BUF=1); up to %s MiB a pass (KYTY_RAM20_MB); \"[ram20]\" line every %s s "
				            "(KYTY_RAM20_LINE_S)\n",
				            ram20_set.empty() ? "" : ", ", ram20_set.c_str(),
				            Env("KYTY_RAM20_COLD_S") != nullptr ? Env("KYTY_RAM20_COLD_S") : "10",
				            Env("KYTY_RAM20_MB") != nullptr ? Env("KYTY_RAM20_MB") : "256",
				            Env("KYTY_RAM20_LINE_S") != nullptr ? Env("KYTY_RAM20_LINE_S") : "10");
				StartRam18();
			}
			if (Mem1::On()) {
				std::printf("MEM1: on (KYTY_MEM1=1%s%s): the C++ heap's free blocks given back at 60 s, 120 s, then every 180 s "
				            "(KYTY_MEM1_HEAP=0 off; periodic passes stop if the heap's lock is held over KYTY_MEM1_HEAP_MAX_MS, "
				            "default 20); \"[mem1-vm]\" VirtualMemo census every 30 s; \"[mem1]\" line every 10 s\n",
				            mem1_set.empty() ? "" : ", ", mem1_set.c_str());
				StartMem1();
			}
			if (rnt_on) { // (ram1b)
				std::printf("[rnt] RAM: copia guest de capas de arrays y render targets con imagen al almacen en disco y fuera de RAM "
				            "(KYTY_RAM_NATIVE_TEX=1%s%s; ruta RAM18 en frio: cualquier toque la lee de vuelta del disco y se cuenta)\n",
				            rnt_set.empty() ? "" : ", ", rnt_set.c_str());
				StartRam18();
			}
			std::fflush(stdout);
			StartRam16Line();
		}
		const uint64_t ram    = LocalPlatform::PhysicalMemory();
		const char*    forced = Env("KYTY_LOWRAM");
		if (!LowRam()) {
			std::printf("RAM: low-RAM profile off (%llu MiB of RAM%s; KYTY_LOWRAM=1 turns it on)\n",
			            static_cast<unsigned long long>(ram / MiB), forced != nullptr ? ", KYTY_LOWRAM=0" : "");
			std::fflush(stdout);
			return;
		}
		std::string set;
		SetDefault("KYTY_SOFT_RELEASE", "1", &set);
		SetDefault("KYTY_RAM_DIET", "1", &set);
		SetDefault("KYTY_PREFETCH_MIN_AVAIL_MB", "3072", &set);
		// (No video memory budget of its own: 5120 MiB, 10-24, left the video memory in use at 9.1-9.4 GB anyway - what
		// holds it is not the collector's images - and only made the collector and pipeline eviction churn.)
		// The guest store committed as it is mapped (the game maps ~11.3 of its 13.5 GiB): only for the page-file
		// section committed at launch, the default.
		const char* file     = Env("KYTY_BACKING_FILE");
		const bool  in_file  = file != nullptr && !(file[0] == '0' && file[1] == '\0');
		const char* lazy     = Env("KYTY_GUEST_LAZY_COMMIT");
		const char* demand   = Env("KYTY_LOWRAM_ON_DEMAND");
		if (!in_file && !(lazy != nullptr && lazy[0] == '1') && !(demand != nullptr && demand[0] == '0'))
			SetDefault("KYTY_BACKING_ON_DEMAND", "1", &set);
		std::printf("RAM: low-RAM profile on (%llu MiB of RAM%s; KYTY_LOWRAM=0 turns it off): %s\n",
		            static_cast<unsigned long long>(ram / MiB), forced != nullptr ? ", KYTY_LOWRAM=1" : ", 20 GiB or less",
		            set.empty() ? "every switch already set" : set.c_str());
		std::fflush(stdout);
	});
}

int PressureLevel() {
	return g_level.load(std::memory_order_relaxed);
}

uint64_t AvailablePhysical() {
	const uint64_t value = g_available.load(std::memory_order_relaxed);
	return value != 0 ? value : LocalPlatform::AvailablePhysicalMemory();
}

void NoteGuestStore(const void* alias, uint64_t size, StoreCharge charge) {
	g_store_alias.store(alias, std::memory_order_relaxed);
	g_store_size.store(size, std::memory_order_relaxed);
	g_store_charge.store(static_cast<int>(charge), std::memory_order_relaxed);
}

void NoteReleased(Released kind, uint64_t count, uint64_t bytes) {
	if (kind < 0 || kind >= ReleasedKinds) return;
	g_released_count[kind].fetch_add(count, std::memory_order_relaxed);
	g_released_bytes[kind].fetch_add(bytes, std::memory_order_relaxed);
	g_released_count_total[kind].fetch_add(count, std::memory_order_relaxed);
	g_released_bytes_total[kind].fetch_add(bytes, std::memory_order_relaxed);
}

void RegisterFlush(const char* name, uint64_t (*flush)(int level)) {
	std::lock_guard lock(g_flush_mutex);
	g_flushes.emplace_back(name, flush);
}

void RegisterBudgetFlush(const char* name, FlushClass kind, BudgetFlushFn flush) {
	if (flush == nullptr || kind < 0 || kind >= FlushClasses) return;
	std::lock_guard lock(g_flush_mutex);
	g_budget_flushes.push_back({name, kind, flush});
}

bool PeriodicFlush() {
	return g_periodic.load(std::memory_order_relaxed);
}

#if defined(_WIN32)
namespace {

// (render1, 11-15) KYTY_RAM_STORE_COUNTER=1 (off by default, read once): the guest store's committed bytes under
// KYTY_BACKING_ON_DEMAND from the store's own counter of mapped 64 KiB granules (GuestLazyCommit::NoteMapped, what the
// on-demand commit committed at map; a section's pages stay committed) instead of a VirtualQuery walk of the 13.5 GiB
// alias every 5 s. Logs 11-15 (03:52, 04:09): a 105-145 ms render-thread stall every 5.13 s (= the guard's 5 s plus the
// walk itself), in whatever call took the address space next (the upload worker's protections: upload_call_us 105-134
// ms; buffer_sync_us / region_sync_us ~120 ms), half the gameplay hitches over 60 ms. The figure is for the RAM lines
// and the simulated commit only: nothing else changes. (Walks still taken are timed: "SLOW StoreWalk" over 20 ms.)
bool StoreCounterEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAM_STORE_COUNTER");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

// The guest store's share of the commit charge.
uint64_t StoreCommitted() {
	const auto  charge = static_cast<StoreCharge>(g_store_charge.load(std::memory_order_relaxed));
	const auto  size   = g_store_size.load(std::memory_order_relaxed);
	const auto* alias  = static_cast<const uint8_t*>(g_store_alias.load(std::memory_order_relaxed));
	switch (charge) {
		case StoreCharge::File: return 0;
		case StoreCharge::Lazy: return GuestLazyCommit::Get().committed.load(std::memory_order_relaxed);
		case StoreCharge::OnDemand: {
			if (StoreCounterEnabled()) {
				const uint64_t mapped = GuestLazyCommit::Get().mapped.load(std::memory_order_relaxed);
				if (mapped != 0) return std::min(mapped, size);
			}
			// (A walk of the alias's regions: committed runs as the game mapped them, a few thousand at most.)
			if (alias == nullptr) return size;
			const auto               walk_start = std::chrono::steady_clock::now();
			uint64_t                 committed = 0, regions = 0;
			MEMORY_BASIC_INFORMATION info {};
			for (uint64_t at = 0; at < size;) {
				if (VirtualQuery(alias + at, &info, sizeof(info)) != sizeof(info)) return size;
				regions++;
				const uint64_t end  = static_cast<uint64_t>(static_cast<const uint8_t*>(info.BaseAddress) - alias) + info.RegionSize;
				const uint64_t stop = std::min(end, size);
				if (stop <= at) break;
				if (info.State == MEM_COMMIT) committed += stop - at;
				at = stop;
			}
			// (render1) Timed: on the SLOW Frame lines' TSC timeline, to match the frames it stalled.
			if (const double walk_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - walk_start).count();
			    walk_ms >= 20.0) {
				std::printf("[tsc %llu] SLOW StoreWalk %.1f ms regions=%llu committed=%llu MiB (KYTY_RAM_STORE_COUNTER=1: "
				            "no walk)\n",
				            static_cast<unsigned long long>(__rdtsc()), walk_ms, static_cast<unsigned long long>(regions),
				            static_cast<unsigned long long>(committed >> 20u));
				std::fflush(stdout);
			}
			return committed;
		}
		case StoreCharge::AllCommitted:
		default: return size;
	}
}

// KYTY_RESIDENCY_LOG=0 turns it off (default on): with the 30 s RAM line, where the working set is, from the process's
// whole working-set list (K32QueryWorkingSet, one walk on the guard's thread, ~10-40 ms for 4M pages): pages of the guest
// store's emulator-side alias, pages of the other mapped views (the game's own views of the guest store: direct and
// flexible memory), private pages (the emulator's own allocations and the driver's), images (DLLs); against what of the
// store is committed and what of it was ever written (fresh-memory granules: CPU, host or write-back writes). Committed
// but never touched pages of the store are commit charge only: no RAM. A page touched through both the alias and a game
// view is in the working set twice (one physical page): the alias and view figures can overlap.
bool ResidencyLogEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RESIDENCY_LOG");
		return v == nullptr || !(v[0] == '0' && v[1] == '\0');
	}();
	return on;
}

void PrintResidency(uint64_t store_committed) {
	if (!ResidencyLogEnabled()) return;
	const auto t0 = std::chrono::steady_clock::now();
	// (Freed after each line: ~8 bytes a resident page, not kept.)
	std::vector<ULONG_PTR> buffer(size_t {1} << 16u);
	HANDLE                 process = GetCurrentProcess();
	bool ok = false;
	for (int attempt = 0; attempt < 4; attempt++) {
		if (K32QueryWorkingSet(process, buffer.data(), static_cast<DWORD>(buffer.size() * sizeof(ULONG_PTR))) != 0) {
			ok = true;
			break;
		}
		if (GetLastError() != ERROR_BAD_LENGTH) break;
		const auto needed = static_cast<size_t>(buffer[0]); // NumberOfEntries
		buffer.resize(std::max(buffer.size() * 2, needed + needed / 8 + 4096 + 1));
	}
	if (!ok) {
		std::printf("RAM residency (30 s): K32QueryWorkingSet failed (0x%08x)\n", static_cast<unsigned>(GetLastError()));
		std::fflush(stdout);
		return;
	}
	const size_t count = std::min(static_cast<size_t>(buffer[0]), buffer.size() - 1);
	uint64_t     shared_pages = 0;
	// (In place: the entries become page addresses, sorted, so the region walk queries each region once.)
	for (size_t i = 1; i <= count; i++) {
		if ((buffer[i] & 0x100u) != 0) shared_pages++; // PSAPI_WORKING_SET_BLOCK.Shared (bit 8)
		buffer[i] &= ~static_cast<ULONG_PTR>(0xfff);
	}
	std::sort(buffer.begin() + 1, buffer.begin() + 1 + static_cast<std::ptrdiff_t>(count));
	const auto     alias      = reinterpret_cast<uint64_t>(g_store_alias.load(std::memory_order_relaxed));
	const uint64_t store_size = g_store_size.load(std::memory_order_relaxed);
	constexpr uint64_t Page   = 4096;
	uint64_t in_alias = 0, in_views = 0, in_private = 0, in_image = 0, in_other = 0;
	uint64_t region_begin = 0, region_end = 0;
	DWORD    region_type  = 0;
	for (size_t i = 1; i <= count; i++) {
		const auto page = static_cast<uint64_t>(buffer[i]);
		if (alias != 0 && page >= alias && page - alias < store_size) {
			in_alias += Page;
			continue;
		}
		if (page < region_begin || page >= region_end) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(page), &info, sizeof(info)) != sizeof(info)) {
				in_other += Page;
				region_begin = region_end = 0;
				continue;
			}
			region_begin = reinterpret_cast<uint64_t>(info.BaseAddress);
			region_end   = region_begin + info.RegionSize;
			region_type  = info.Type;
		}
		(region_type == MEM_MAPPED ? in_views : region_type == MEM_PRIVATE ? in_private : region_type == MEM_IMAGE ? in_image : in_other) += Page;
	}
	uint64_t written = 0;
	if (const auto* fresh = FreshMemory::Get(); fresh != nullptr) {
		const uint64_t words = (fresh->granules + 63) / 64;
		for (uint64_t w = 0; w < words; w++)
			for (uint64_t bits = fresh->written[w].load(std::memory_order_relaxed); bits != 0; bits &= bits - 1) written++;
		written <<= FreshMemory::GranuleBits;
	}
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::printf("RAM residency (30 s, MiB; KYTY_RESIDENCY_LOG=0 off): working set %llu (shared pages %llu) = guest store "
	            "through the emulator's alias %llu + other mapped views (the game's views of the guest store) %llu + private "
	            "%llu + images %llu + other %llu; guest store committed %llu, ever written %s%llu, so committed and never "
	            "written <= %llu (commit charge, no RAM unless read); %.1f ms\n",
	            static_cast<unsigned long long>(count * Page / MiB), static_cast<unsigned long long>(shared_pages * Page / MiB),
	            static_cast<unsigned long long>(in_alias / MiB), static_cast<unsigned long long>(in_views / MiB),
	            static_cast<unsigned long long>(in_private / MiB), static_cast<unsigned long long>(in_image / MiB),
	            static_cast<unsigned long long>(in_other / MiB), static_cast<unsigned long long>(store_committed / MiB),
	            FreshMemory::Enabled() ? "" : "(fresh memory off: unknown) ", static_cast<unsigned long long>(written / MiB),
	            static_cast<unsigned long long>((store_committed > written ? store_committed - written : 0) / MiB), ms);
	std::fflush(stdout);
}

struct Sample {
	uint64_t private_bytes = 0, working_set = 0;
	uint64_t total_phys = 0, available = 0;          // as counted (simulated when asked)
	uint64_t commit_total = 0, commit_limit = 0;     // as counted
	uint64_t real_available = 0, real_headroom = 0;  // Windows' own
	uint64_t store_committed = 0;
	bool     signal    = false;                      // Windows' low-memory notification
	bool     simulated = false;
	[[nodiscard]] uint64_t Headroom() const { return commit_limit > commit_total ? commit_limit - commit_total : 0; }
};

struct Settings {
	uint64_t sim_ram = 0, sim_commit_limit = 0, sim_os = 0, sim_os_commit = 0;
	uint64_t low_avail = 0, low_commit = 0;
	bool     sim_abort = false;                      // the simulated PC's end when its commit is used up
	int64_t  flush_seconds = 0;                      // KYTY_RAM_FLUSH_SECONDS: the periodic flush (0 off)
	uint64_t flush_bytes   = 0;                      // KYTY_RAM_FLUSH_MB
	bool     flush_trim    = false;                  // KYTY_RAM_FLUSH_TRIM=1: the working set trim too
	uint64_t budget        = 0;                      // KYTY_RAM_BUDGET_MB: the working set kept under it (0 off)
	uint64_t budget_step   = 0;                      // KYTY_RAM_BUDGET_STEP_MB: the most trimmed a second
	uint64_t budget_floor  = 0;                      // the hard working set minimum kept meanwhile (KYTY_RESIDENT_MB)
};

Settings ReadSettings() {
	Settings s;
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	const uint64_t real_ram = GlobalMemoryStatusEx(&status) != 0 ? status.ullTotalPhys : 0;
	s.sim_ram          = EnvMiB("KYTY_SIMULATE_RAM_MB", 0);
	if (real_ram != 0 && s.sim_ram >= real_ram) s.sim_ram = 0; // (not a smaller PC)
	s.sim_commit_limit = EnvMiB("KYTY_SIMULATE_COMMIT_MB", 0);
	if (s.sim_commit_limit == 0 && s.sim_ram != 0) s.sim_commit_limit = s.sim_ram + EnvMiB("KYTY_SIMULATE_PAGEFILE_MB", s.sim_ram);
	s.sim_os        = EnvMiB("KYTY_SIMULATE_OS_MB", 3 * GiB);
	s.sim_os_commit = EnvMiB("KYTY_SIMULATE_OS_COMMIT_MB", 4 * GiB);
	const uint64_t ram = s.sim_ram != 0 ? s.sim_ram : real_ram;
	s.low_avail  = EnvMiB("KYTY_RAM_LOW_AVAIL_MB", std::max<uint64_t>(GiB, ram / 16));
	s.low_commit = EnvMiB("KYTY_RAM_LOW_COMMIT_MB", 2 * GiB);
	{
		const char* abort_text = Env("KYTY_SIMULATE_RAM_ABORT");
		s.sim_abort            = s.sim_commit_limit != 0 && !(abort_text != nullptr && abort_text[0] == '0');
		const char* seconds    = Env("KYTY_RAM_FLUSH_SECONDS");
		s.flush_seconds        = seconds != nullptr ? std::max<int64_t>(0, std::strtoll(seconds, nullptr, 10)) : 0;
		s.flush_bytes          = std::max<uint64_t>(MiB, EnvMiB("KYTY_RAM_FLUSH_MB", 1024 * MiB));
		const char* trim       = Env("KYTY_RAM_FLUSH_TRIM");
		s.flush_trim           = trim != nullptr && trim[0] == '1';
	}
	s.budget      = EnvMiB("KYTY_RAM_BUDGET_MB", 0);
	if (s.budget != 0) s.budget = std::max<uint64_t>(s.budget, 2 * GiB);
	s.budget_step = std::clamp<uint64_t>(EnvMiB("KYTY_RAM_BUDGET_STEP_MB", 384 * MiB), 64 * MiB, 2 * GiB);
	// (KYTY_RESIDENT_MB's hard minimum kept, but under the budget's trim target.)
	s.budget_floor = s.budget != 0 ? std::min(EnvMiB("KYTY_RESIDENT_MB", 0), s.budget - GiB) : 0;
	return s;
}

Sample Take(const Settings& s, HANDLE low_memory, bool walk_store, uint64_t last_store) {
	Sample out;
	const auto use    = LocalPlatform::ProcessMemory();
	out.private_bytes = use.private_bytes;
	out.working_set   = use.working_set;
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (GlobalMemoryStatusEx(&status) != 0) {
		out.total_phys     = status.ullTotalPhys;
		out.available      = status.ullAvailPhys;
		out.real_available = status.ullAvailPhys;
		// (ullTotalPageFile/ullAvailPageFile: the commit limit and what is left of it.)
		out.commit_limit  = status.ullTotalPageFile;
		out.commit_total  = status.ullTotalPageFile - std::min(status.ullTotalPageFile, status.ullAvailPageFile);
		out.real_headroom = status.ullAvailPageFile;
	}
	out.store_committed = walk_store ? StoreCommitted() : last_store;
	if (s.sim_ram != 0) {
		out.simulated  = true;
		out.total_phys = s.sim_ram;
		const uint64_t used = s.sim_os + out.working_set;
		out.available  = std::min(out.real_available, s.sim_ram > used ? s.sim_ram - used : 0);
	}
	if (s.sim_commit_limit != 0) {
		out.simulated = true;
		// (Not the KYTY_SIMULATE_VRAM_MB ballast's driver commit: 7.8 GB of the private bytes in a simulated 8 GB card
		// run, 10-24, which a real 8 GB card does not have.)
		const uint64_t ballast  = LocalPlatform::SimulationBallast();
		const uint64_t charged  = s.sim_os_commit + (out.private_bytes - std::min(out.private_bytes, ballast)) + out.store_committed;
		const uint64_t headroom = std::min(out.real_headroom, s.sim_commit_limit > charged ? s.sim_commit_limit - charged : 0);
		out.commit_limit        = s.sim_commit_limit;
		out.commit_total        = s.sim_commit_limit - std::min(s.sim_commit_limit, headroom);
	}
	if (low_memory != nullptr) {
		BOOL state = FALSE;
		out.signal = QueryMemoryResourceNotification(low_memory, &state) != 0 && state != FALSE;
	}
	return out;
}

int LevelOf(const Settings& s, const Sample& sample) {
	const uint64_t headroom = sample.Headroom();
	const bool     known    = sample.total_phys != 0;
	if (!known) return None;
	if (sample.available < s.low_avail / 2 || (sample.commit_limit != 0 && headroom < s.low_commit / 2)) return Critical;
	if (sample.available < s.low_avail || (sample.commit_limit != 0 && headroom < s.low_commit) || sample.signal) return Low;
	return None;
}

std::string Figures(const Sample& sample) {
	char       ballast_text[96] = "";
	if (const uint64_t ballast = LocalPlatform::SimulationBallast(); ballast != 0)
		std::snprintf(ballast_text, sizeof(ballast_text), " (%llu MiB of the private: the KYTY_SIMULATE_VRAM_MB ballast, not counted)",
		              static_cast<unsigned long long>(ballast / MiB));
	char line[640];
	std::snprintf(line, sizeof(line),
	              "available %llu of %llu MiB, commit %llu of %llu MiB (headroom %llu MiB)%s%s; emulator private %llu MiB, "
	              "working set %llu MiB, guest store committed %llu MiB%s",
	              static_cast<unsigned long long>(sample.available / MiB),
	              static_cast<unsigned long long>(sample.total_phys / MiB),
	              static_cast<unsigned long long>(sample.commit_total / MiB),
	              static_cast<unsigned long long>(sample.commit_limit / MiB),
	              static_cast<unsigned long long>(sample.Headroom() / MiB), sample.signal ? ", Windows low-memory signal" : "",
	              sample.simulated ? " (simulated: KYTY_SIMULATE_RAM_MB / KYTY_SIMULATE_COMMIT_MB)" : "",
	              static_cast<unsigned long long>(sample.private_bytes / MiB),
	              static_cast<unsigned long long>(sample.working_set / MiB),
	              static_cast<unsigned long long>(sample.store_committed / MiB), ballast_text);
	return line;
}

// What owners released since the last line (pipelines, images), and clears it.
std::string TakeReleased(uint64_t* bytes) {
	std::string out;
	static const char* const names[ReleasedKinds] = {"idle pipelines", "unused images (video memory)"};
	for (int kind = 0; kind < ReleasedKinds; kind++) {
		const uint64_t count = g_released_count[kind].exchange(0, std::memory_order_relaxed);
		const uint64_t b     = g_released_bytes[kind].exchange(0, std::memory_order_relaxed);
		if (count == 0) continue;
		*bytes += b;
		char piece[128];
		std::snprintf(piece, sizeof(piece), ", %s %llu (~%llu MiB)", names[kind], static_cast<unsigned long long>(count),
		              static_cast<unsigned long long>(b / MiB));
		out += piece;
	}
	return out;
}

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point start) {
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// The simulated PC ran out of commit: the run ends as it would have there (an allocation refused, the game closed).
[[noreturn]] void SimulatedOutOfMemory(const Settings& s, const Sample& sample) {
	char pc[64];
	if (s.sim_ram != 0)
		std::snprintf(pc, sizeof(pc), "un PC de %llu GB", static_cast<unsigned long long>((s.sim_ram + GiB / 2) / GiB));
	else
		std::snprintf(pc, sizeof(pc), "un PC con %llu MiB de commit", static_cast<unsigned long long>(s.sim_commit_limit / MiB));
	std::printf("RAM simulada agotada: en %s el juego se habria cerrado (simulated RAM used up: on that PC the game "
	            "would have closed; KYTY_SIMULATE_RAM_ABORT=0 keeps running on the real memory). %s\n",
	            pc, Figures(sample).c_str());
	std::fflush(stdout);
	std::fflush(stderr);
	// (No exit handlers: the game's threads are running; STATUS_NO_MEMORY, as a refused allocation's end would be.)
	TerminateProcess(GetCurrentProcess(), 0xC0000017u);
	for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}

struct FlushState {
	uint64_t passes = 0;
	uint64_t released_total[FlushClasses] {};
	double   ms_total = 0.0, ms_max = 0.0;
	double   heaps_ms_last = 0.0;
	Clock::time_point heaps_last {};
};

// One periodic flush pass (ram-guard.h): the sources by class, in order, until s.flush_bytes are released.
void FlushPass(const Settings& s, HANDLE low_memory, uint64_t store, FlushState* state) {
	const auto   start  = Clock::now();
	const Sample before = Take(s, low_memory, false, store);
	std::vector<BudgetFlush> list;
	{
		std::lock_guard lock(g_flush_mutex);
		list = g_budget_flushes;
	}
	uint64_t    remaining = s.flush_bytes;
	uint64_t    released[FlushClasses] {};
	uint64_t    held_us = 0;
	std::string parts[FlushClasses];
	const auto  add_part = [&](int kind, const char* name, uint64_t bytes, double ms) {
		char piece[160];
		std::snprintf(piece, sizeof(piece), "%s%s %.1f MiB %.1f ms", parts[kind].empty() ? "" : ", ", name,
		              static_cast<double>(bytes) / 1048576.0, ms);
		parts[kind] += piece;
	};
	for (int kind = 0; kind < FlushClasses; kind++) {
		if (kind == FlushTrim && !s.flush_trim) break;
		const auto use_before = LocalPlatform::ProcessMemory();
		uint64_t   reported   = 0;
		for (const auto& source: list) {
			if (source.kind != kind || remaining <= reported) continue;
			uint64_t   held = 0;
			const auto t0   = Clock::now();
			const auto b    = source.fn(remaining - reported, &held);
			held_us         = std::max(held_us, held);
			reported += b;
			add_part(kind, source.name, b, MsSince(t0));
		}
		if (kind == FlushCaches && remaining > reported) {
			// Idle scratch buffers and the heaps' free blocks (soft-release.h). The compaction holds each heap's lock: when
			// it took more than 5 ms, it runs at most every 30 s.
			const auto t0 = Clock::now();
			if (state->heaps_ms_last <= 5.0 || t0 - state->heaps_last >= std::chrono::seconds(30)) {
				char detail[192];
				(void)SoftRelease::FlushNow(10'000, detail, sizeof(detail));
				state->heaps_ms_last = MsSince(t0);
				state->heaps_last    = t0;
				held_us              = std::max(held_us, static_cast<uint64_t>(state->heaps_ms_last * 1000.0));
				char piece[256];
				std::snprintf(piece, sizeof(piece), "%s%s %.1f ms", parts[kind].empty() ? "" : ", ", detail, state->heaps_ms_last);
				parts[kind] += piece;
			} else {
				parts[kind] += parts[kind].empty() ? "heaps skipped (slow last time)" : ", heaps skipped (slow last time)";
			}
		}
		// Caches: what the commit dropped (or the sources said); guest memory and the trim: what the working set dropped
		// (their commit stays).
		const auto use_after = LocalPlatform::ProcessMemory();
		const auto dropped   = kind == FlushCaches ? (use_before.private_bytes > use_after.private_bytes
		                                                  ? use_before.private_bytes - use_after.private_bytes : 0)
		                                           : (use_before.working_set > use_after.working_set
		                                                  ? use_before.working_set - use_after.working_set : 0);
		released[kind] = kind == FlushCaches ? std::max(reported, dropped) : dropped;
		remaining -= std::min(remaining, std::max(released[kind], kind == FlushCaches ? 0 : reported));
		state->released_total[kind] += released[kind];
		if (remaining == 0) break;
	}
	const double ms    = MsSince(start);
	const Sample after = Take(s, low_memory, false, store);
	state->passes++;
	state->ms_total += ms;
	state->ms_max = std::max(state->ms_max, ms);
	const auto mb = [](uint64_t b) { return static_cast<unsigned long long>(b / 1000000u); };
	const auto mi = [](uint64_t b) { return static_cast<unsigned long long>(b / MiB); };
	std::printf("RAM flush #%llu (every %lld s, up to %llu MB): released %llu MB: caches %llu MB [%s], freed guest memory "
	            "%llu MB of RAM [%s], trim %s%llu MB%s%s%s; %.1f ms (longest lock a game thread could wait on %.2f ms); "
	            "private %llu -> %llu MiB, working set %llu -> %llu MiB, commit %llu -> %llu of %llu MiB, available %llu -> "
	            "%llu MiB%s\n",
	            static_cast<unsigned long long>(state->passes), static_cast<long long>(s.flush_seconds), mb(s.flush_bytes),
	            mb(released[0] + released[1] + released[2]), mb(released[0]), parts[0].c_str(), mb(released[1]),
	            parts[1].empty() ? "none" : parts[1].c_str(), s.flush_trim ? "" : "off ", mb(released[2]),
	            s.flush_trim ? " [" : "", s.flush_trim ? (parts[2].empty() ? "none" : parts[2].c_str()) : "",
	            s.flush_trim ? "] (no commit change)" : "", ms, static_cast<double>(held_us) / 1000.0,
	            mi(before.private_bytes), mi(after.private_bytes), mi(before.working_set), mi(after.working_set),
	            mi(before.commit_total), mi(after.commit_total), mi(after.commit_limit), mi(before.available),
	            mi(after.available), after.simulated ? " (simulated)" : "");
	std::fflush(stdout);
}

void Watch() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.RamGuard");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	SoftRelease::LowerThreadMemoryPriority();
	const Settings s          = ReadSettings();
	const char*    log_text   = Env("KYTY_RAM_LOG_SECONDS");
	const int64_t  log_period = log_text != nullptr ? std::strtoll(log_text, nullptr, 10) : 30;
	HANDLE         low_memory = CreateMemoryResourceNotification(LowMemoryResourceNotification);
	std::printf("RAM guard: on (KYTY_RAM_GUARD=0 turns it off): pressure below %llu MiB available or %llu MiB of commit "
	            "headroom (critical at half)%s; low-RAM profile %s; a line every %lld s\n",
	            static_cast<unsigned long long>(s.low_avail / MiB), static_cast<unsigned long long>(s.low_commit / MiB),
	            s.sim_ram != 0 || s.sim_commit_limit != 0 ? ", simulated PC" : "", LowRam() ? "on" : "off",
	            static_cast<long long>(log_period));
	if (s.sim_ram != 0 || s.sim_commit_limit != 0)
		std::printf("RAM guard: simulating %llu MiB of RAM (%llu MiB for Windows) and a %llu MiB commit limit (%llu MiB for "
		            "Windows)\n",
		            static_cast<unsigned long long>(s.sim_ram / MiB), static_cast<unsigned long long>(s.sim_os / MiB),
		            static_cast<unsigned long long>(s.sim_commit_limit / MiB),
		            static_cast<unsigned long long>(s.sim_os_commit / MiB));
	if (s.sim_abort)
		std::printf("RAM guard: the simulated PC's end is simulated too: with no commit headroom left the run is closed "
		            "(\"RAM simulada agotada\"; KYTY_SIMULATE_RAM_ABORT=0 keeps running)\n");
	if (s.flush_seconds > 0)
		std::printf("RAM flush (vaciado): every %lld s up to %llu MB: caches (re-creatable), guest memory the game released%s "
		            "(KYTY_RAM_FLUSH_SECONDS / KYTY_RAM_FLUSH_MB / KYTY_RAM_FLUSH_TRIM)\n",
		            static_cast<long long>(s.flush_seconds), static_cast<unsigned long long>(s.flush_bytes / 1000000u),
		            s.flush_trim ? ", then the working set trim of cold guest pages" : "; no working set trim");
	std::fflush(stdout);
	if (s.budget != 0) {
		std::printf("RAM budget: working set kept under %llu MiB (KYTY_RAM_BUDGET_MB; 0 turns it off): over it, up to %llu "
		            "MiB a second trimmed to the page file / standby list in quiet windows (hard minimum %llu MiB), the "
		            "release passes run as under pressure (critical 1 GiB over), released guest memory dropped\n",
		            static_cast<unsigned long long>(s.budget / MiB), static_cast<unsigned long long>(s.budget_step / MiB),
		            static_cast<unsigned long long>(s.budget_floor / MiB));
	}
	std::fflush(stdout);
	g_periodic.store(s.flush_seconds > 0, std::memory_order_relaxed);
	FlushState flush_state;
	auto       next_periodic = Clock::now() + std::chrono::seconds(std::max<int64_t>(s.flush_seconds, 1));
	uint64_t   seconds_under_1g = 0, seconds_under_512 = 0;
	// The RAM budget: what was trimmed (in all and since the last line), the seconds over it, the trims refused.
	uint64_t   budget_trimmed = 0, budget_trimmed_line = 0, budget_seconds_over = 0, budget_refused = 0;
	uint64_t   budget_ws_max_line = 0;
	auto       budget_next_guest = Clock::now();
	int        sim_exhausted = 0;
	auto     next_log   = Clock::now() + std::chrono::seconds(std::max<int64_t>(log_period, 1));
	auto     next_flush = Clock::now();
	auto     next_walk  = Clock::now();
	int      backoff    = 0;
	int      logged     = None;
	uint64_t store      = 0, flushes = 0, freed_total = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		const auto now  = Clock::now();
		const bool walk = now >= next_walk;
		if (walk) next_walk = now + std::chrono::seconds(5);
		const Sample sample = Take(s, low_memory, walk, store);
		store               = sample.store_committed;
		g_available.store(sample.available, std::memory_order_relaxed);
		// (The simulated commit used up: as on that PC, a release pass first - what MemGuard does when an allocation is
		// refused - and the end only when a second later there is still none.)
		if (s.sim_abort && sample.commit_limit != 0 && sample.Headroom() == 0) {
			if (sim_exhausted++ != 0) SimulatedOutOfMemory(s, sample);
			std::printf("RAM simulada: sin commit libre (no simulated commit headroom left): a release pass, the run ends if "
			            "there is still none; %s\n",
			            Figures(sample).c_str());
			std::fflush(stdout);
			(void)MemGuard::ReleaseNow("simulated commit used up", 500);
		} else {
			sim_exhausted = 0;
		}
		if (s.sim_ram != 0) {
			seconds_under_1g += sample.available < GiB ? 1 : 0;
			seconds_under_512 += sample.available < GiB / 2 ? 1 : 0;
		}
		int level = LevelOf(s, sample);
		if (s.budget != 0) {
			budget_ws_max_line = std::max(budget_ws_max_line, sample.working_set);
			if (sample.working_set > s.budget) {
				budget_seconds_over++;
				// Over the budget: the release passes and the owners (pipelines, images) as under pressure; critical a GiB
				// over. Then the working set trimmed one step (a few hundred MiB, inside the guest memory's quiet window)
				// toward 256 MiB under the budget: cold pages first (Windows trims the least recently used).
				level = std::max(level, sample.working_set > s.budget + GiB ? static_cast<int>(Critical) : static_cast<int>(Low));
				const uint64_t goal   = s.budget - 256 * MiB;
				const uint64_t target = std::max(goal, sample.working_set - std::min(sample.working_set, s.budget_step));
				if (target > s.budget_floor) {
					if (LocalPlatform::TrimWorkingSetTo(target, s.budget_floor)) {
						const uint64_t after = LocalPlatform::ProcessMemory().working_set;
						const uint64_t cut   = sample.working_set > after ? sample.working_set - after : 0;
						budget_trimmed += cut;
						budget_trimmed_line += cut;
					} else {
						budget_refused++;
					}
				}
				// Guest memory the game released but still in RAM, dropped (no data lost), at most every 30 s.
				if (now >= budget_next_guest) {
					budget_next_guest = now + std::chrono::seconds(30);
					Settings guest  = s;
					guest.flush_trim = false;
					FlushPass(guest, low_memory, store, &flush_state);
				}
			}
		}
		g_level.store(level, std::memory_order_relaxed);
		if (s.flush_seconds > 0 && now >= next_periodic) {
			FlushPass(s, low_memory, store, &flush_state);
			next_periodic = Clock::now() + std::chrono::seconds(s.flush_seconds);
		}
		if (level != logged && level == None) {
			uint64_t    bytes    = 0;
			const auto  released = TakeReleased(&bytes);
			std::printf("RAM: pressure %s -> none; %s%s\n", LevelName(logged), Figures(sample).c_str(), released.c_str());
			std::fflush(stdout);
			backoff = 0;
		}
		logged = level;
		if (level != None && now >= next_flush) {
			// The safe flush: only free or re-creatable memory (soft-release.h's pass and the registered sources); the
			// owners of pipelines and images release theirs on their own threads while the level stays up.
			char           detail[256];
			const uint64_t before = sample.private_bytes;
			uint64_t       freed  = SoftRelease::FlushNow(level == Critical ? 500 : 2000, detail, sizeof(detail));
			std::string    sources;
			{
				std::vector<std::pair<const char*, uint64_t (*)(int)>> list;
				{
					std::lock_guard lock(g_flush_mutex);
					list = g_flushes;
				}
				for (const auto& [name, flush]: list) {
					const uint64_t bytes = flush(level);
					if (bytes == 0) continue;
					char piece[128];
					std::snprintf(piece, sizeof(piece), ", %s %.1f MiB", name, static_cast<double>(bytes) / 1048576.0);
					sources += piece;
					freed += bytes;
				}
			}
			uint64_t   owners   = 0;
			const auto released = TakeReleased(&owners);
			freed += owners;
			freed_total += freed;
			flushes++;
			const auto after = LocalPlatform::ProcessMemory().private_bytes;
			std::printf("RAM: pressure %s (%s), freed %llu MB (%s%s%s; private %llu -> %llu MiB); %llu passes, %llu MB in all\n",
			            LevelName(level), Figures(sample).c_str(), static_cast<unsigned long long>(freed / 1000000u), detail,
			            sources.c_str(), released.c_str(), static_cast<unsigned long long>(before / MiB),
			            static_cast<unsigned long long>(after / MiB), static_cast<unsigned long long>(flushes),
			            static_cast<unsigned long long>(freed_total / 1000000u));
			std::fflush(stdout);
			// Again in 5 s (critical) or 15 s (low); a pass that freed little waits twice as long, up to 2 minutes:
			// nothing left to give is not looked for every few seconds.
			backoff = freed < 32 * MiB ? std::min(backoff + 1, 4) : 0;
			const int base = level == Critical ? 5 : 15;
			next_flush     = now + std::chrono::seconds(std::min(120, base << backoff));
		}
		if (log_period > 0 && now >= next_log) {
			next_log = now + std::chrono::seconds(log_period);
			Libs::LibKernel::Memory::ReportGuestAccess(true);
			PrintResidency(sample.store_committed);
			std::printf("RAM (%lld s): %s; pressure %s, real available %llu MiB, real commit headroom %llu MiB; pressure "
			            "passes %llu (%llu MB), pipelines released %llu, images collected %llu (%llu MiB)\n",
			            static_cast<long long>(log_period), Figures(sample).c_str(), LevelName(level),
			            static_cast<unsigned long long>(sample.real_available / MiB),
			            static_cast<unsigned long long>(sample.real_headroom / MiB), static_cast<unsigned long long>(flushes),
			            static_cast<unsigned long long>(freed_total / 1000000u),
			            static_cast<unsigned long long>(g_released_count_total[Pipelines].load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(g_released_count_total[Images].load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(g_released_bytes_total[Images].load(std::memory_order_relaxed) / MiB));
			if (s.budget != 0) {
				const uint64_t ws_now = LocalPlatform::ProcessMemory().working_set;
				std::printf("RAM budget (%lld s): budget %llu MiB, working set %llu MiB now (peak %llu MiB this period), "
				            "trimmed %llu MiB this period, %llu MiB in all; %llu s over budget in all, %llu trims refused\n",
				            static_cast<long long>(log_period), static_cast<unsigned long long>(s.budget / MiB),
				            static_cast<unsigned long long>(ws_now / MiB), static_cast<unsigned long long>(budget_ws_max_line / MiB),
				            static_cast<unsigned long long>(budget_trimmed_line / MiB),
				            static_cast<unsigned long long>(budget_trimmed / MiB),
				            static_cast<unsigned long long>(budget_seconds_over), static_cast<unsigned long long>(budget_refused));
				budget_trimmed_line = 0;
				budget_ws_max_line  = 0;
			}
			if (s.flush_seconds > 0 || s.sim_ram != 0) {
				std::printf("RAM (%lld s): flush passes %llu (caches %llu MB, freed guest memory %llu MB, trim %llu MB; %.1f ms "
				            "a pass on average, at most %.1f ms); simulated PC under 1 GiB available %llu s, under 512 MiB "
				            "%llu s (a real one pages there: stutter)\n",
				            static_cast<long long>(log_period), static_cast<unsigned long long>(flush_state.passes),
				            static_cast<unsigned long long>(flush_state.released_total[0] / 1000000u),
				            static_cast<unsigned long long>(flush_state.released_total[1] / 1000000u),
				            static_cast<unsigned long long>(flush_state.released_total[2] / 1000000u),
				            flush_state.passes != 0 ? flush_state.ms_total / static_cast<double>(flush_state.passes) : 0.0,
				            flush_state.ms_max, static_cast<unsigned long long>(seconds_under_1g),
				            static_cast<unsigned long long>(seconds_under_512));
			}
			std::fflush(stdout);
		}
	}
}

// (ram2) KYTY_RAM_LINE=1 (off by default, read once): one "[ram]" line every 10 s that says where the PC's RAM is,
// from a thread of its own ("Kyty.RamLine") that takes none of the emulator's locks and walks none of its memory:
// GlobalMemoryStatusEx, GetPerformanceInfo, NtQuerySystemInformation(SystemMemoryListInformation) for the free /
// standby / modified lists, and one NtQuerySystemInformation(SystemProcessInformation) for every process's working
// set (this process's private part and hard faults, Windows' "Memory Compression" process - where trimmed pages are
// kept compressed, in RAM - and the three largest others). Why: logs 11-14/11-15 on 16 GiB, RAM outside the
// emulator's working set (total - available - working set) is 7.4 GiB (median of 135 lines, no VRAM ballast) / 8.0 GiB
// (121 lines, ballast) while standby is 0.2-1 GiB and the modified list 0-1.5 GiB: that much is in use by something the
// emulator's own lines cannot name. The guest store's RAM is this process's shared working set (its private one taken
// off: the game's views and the alias; mapped images are in it too, 24-100 MiB in the residency lines). The guest
// counters (freed memory reset, the game's allocator) are Memory::RamLineStats. Costs one system call of a few ms.
bool RamLineEnabled() {
	static const bool on = [] {
		const char* v = Env("KYTY_RAM_LINE");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

struct RamLineProcesses {
	bool     ok              = false;
	uint64_t own_ws          = 0;
	uint64_t own_private     = 0;
	uint64_t own_hard_faults = 0;
	uint64_t compression_ws  = 0;
	uint64_t others_ws       = 0; // (every other process but the compression store)
	uint32_t others          = 0;
	struct Top {
		char     name[24] {};
		uint64_t ws = 0;
	} top[3];
};

// SYSTEM_PROCESS_INFORMATION (x64), the fields read: +0 NextEntryOffset (u32), +8 WorkingSetPrivateSize (u64), +16
// HardFaultCount (u32), +56 ImageName.Length (u16, bytes), +64 ImageName.Buffer, +80 UniqueProcessId, +144
// WorkingSetSize.
void RamLineReadProcesses(std::vector<uint8_t>* buffer, RamLineProcesses* out) {
	using Query          = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
	static const auto fn = reinterpret_cast<Query>(
	    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQuerySystemInformation")));
	if (fn == nullptr || sizeof(void*) != 8) return;
	if (buffer->size() < (size_t {1} << 20u)) buffer->resize(size_t {1} << 20u);
	ULONG got    = 0;
	LONG  status = fn(5, buffer->data(), static_cast<ULONG>(buffer->size()), &got);
	if (status < 0 && got > buffer->size()) { // (STATUS_INFO_LENGTH_MISMATCH: the size wanted is returned)
		buffer->resize(static_cast<size_t>(got) + (size_t {256} << 10u));
		status = fn(5, buffer->data(), static_cast<ULONG>(buffer->size()), &got);
	}
	if (status < 0) return;
	const uint64_t self  = GetCurrentProcessId();
	const uint8_t* begin = buffer->data();
	const uint8_t* end   = begin + buffer->size();
	for (size_t at = 0; at + 152 <= buffer->size();) {
		const uint8_t* p = begin + at;
		uint32_t       next = 0, hard = 0;
		uint16_t       name_bytes = 0;
		uint64_t       private_ws = 0, name_at = 0, pid = 0, ws = 0;
		std::memcpy(&next, p, 4);
		std::memcpy(&private_ws, p + 8, 8);
		std::memcpy(&hard, p + 16, 4);
		std::memcpy(&name_bytes, p + 56, 2);
		std::memcpy(&name_at, p + 64, 8);
		std::memcpy(&pid, p + 80, 8);
		std::memcpy(&ws, p + 144, 8);
		char        name[24] {};
		const auto* text = reinterpret_cast<const uint8_t*>(name_at);
		if (text >= begin && text + name_bytes <= end) {
			for (size_t i = 0; i < name_bytes / 2u && i + 1 < sizeof(name); i++) {
				uint16_t c = 0;
				std::memcpy(&c, text + i * 2, 2);
				name[i] = c >= 32 && c < 127 ? static_cast<char>(c) : '?';
			}
		}
		if (pid == self) {
			out->own_ws          = ws;
			out->own_private     = private_ws;
			out->own_hard_faults = hard;
		} else if (std::strstr(name, "ompression") != nullptr) { // ("Memory Compression" / "MemCompression")
			out->compression_ws += ws;
		} else {
			out->others_ws += ws;
			out->others++;
			if (name[0] == '\0') std::snprintf(name, sizeof(name), "pid %llu", static_cast<unsigned long long>(pid));
			for (auto& slot: out->top) {
				if (ws <= slot.ws) continue;
				// (The smaller ones move down one.)
				RamLineProcesses::Top moved = slot;
				std::memcpy(slot.name, name, sizeof(slot.name));
				slot.ws = ws;
				std::memcpy(name, moved.name, sizeof(name));
				ws = moved.ws;
			}
		}
		if (next == 0) break;
		at += next;
	}
	out->ok = true;
}

// The system's page lists (NtQuerySystemInformation, SystemMemoryListInformation = 80), in pages; false when refused.
bool RamLinePageLists(uint64_t* free_pages, uint64_t* standby_pages, uint64_t* modified_pages) {
	struct MemoryLists {
		ULONG_PTR zeroed, free, modified, modified_no_write, bad;
		ULONG_PTR standby_by_priority[8];
		ULONG_PTR repurposed_by_priority[8];
		ULONG_PTR modified_page_file;
	};
	using Query          = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
	static const auto fn = reinterpret_cast<Query>(
	    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQuerySystemInformation")));
	if (fn == nullptr) return false;
	MemoryLists lists {};
	ULONG       got = 0;
	if (fn(80, &lists, sizeof(lists), &got) < 0) return false;
	uint64_t standby = 0;
	for (const auto n: lists.standby_by_priority) standby += n;
	*free_pages     = static_cast<uint64_t>(lists.free) + lists.zeroed;
	*standby_pages  = standby;
	*modified_pages = static_cast<uint64_t>(lists.modified) + lists.modified_no_write;
	return true;
}

// Appends to the line (truncated when full).
#if defined(__clang__) || defined(__GNUC__)
__attribute__((format(printf, 4, 5)))
#endif
void RamLineAdd(char* line, size_t size, size_t* used, const char* format, ...) {
	if (*used + 1 >= size) return;
	va_list args;
	va_start(args, format);
	const int n = std::vsnprintf(line + *used, size - *used, format, args);
	va_end(args);
	if (n > 0) *used = std::min(size - 1, *used + static_cast<size_t>(n));
}

// (ram4) The "[ram+]" line's measurements, all system calls (KYTY_RAM_LINE=1 only). What "sin atribuir" can be made of:
// the system's own working set (the file cache's mapped views, the paged pool, pageable kernel and driver code), the
// GPU's "shared" memory (system RAM the graphics kernel locks for every process: Vulkan's host-visible memory, video
// memory moved out of a full card), and what no counter names (page tables, the PFN database, drivers' locked pages).

// The system working set's parts in RAM, in pages (NtQuerySystemInformation, SystemPerformanceInformation = 2; x64
// offsets of SYSTEM_PERFORMANCE_INFORMATION: +44 AvailablePages, +112 PagedPoolPages, +140 ResidentSystemCodePage,
// +164 ResidentSystemCachePage, +168 ResidentPagedPoolPage, +172 ResidentSystemDriverPage, all ULONG). False when
// Windows refuses or the structure is not the one known: its AvailablePages must be the available memory (within
// 256 MiB), or nothing is printed from it.
struct RamLineSystem {
	uint64_t cache_pages = 0, paged_pool_pages = 0, code_pages = 0, driver_pages = 0, paged_pool_virtual_pages = 0;
};
bool RamLineSystemResident(uint64_t available_bytes, uint64_t page, RamLineSystem* out) {
	using Query          = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
	static const auto fn = reinterpret_cast<Query>(
	    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQuerySystemInformation")));
	if (fn == nullptr || sizeof(void*) != 8) return false;
	alignas(8) uint8_t data[1024] {};
	ULONG              got    = 0;
	LONG               status = fn(2, data, static_cast<ULONG>(sizeof(data)), &got);
	if (status < 0) status = fn(2, data, 344, &got); // (the structure's size on Windows 10/11, should it want it exact)
	if (status < 0 || got < 176) return false;
	const auto word = [&](size_t at) {
		uint32_t v = 0;
		std::memcpy(&v, data + at, 4);
		return static_cast<uint64_t>(v);
	};
	const uint64_t available = word(44) * page;
	const uint64_t apart     = available > available_bytes ? available - available_bytes : available_bytes - available;
	if (apart > (uint64_t {256} << 20u)) return false;
	out->paged_pool_virtual_pages = word(112);
	out->code_pages               = word(140);
	out->cache_pages              = word(164);
	out->paged_pool_pages         = word(168);
	out->driver_pages             = word(172);
	return true;
}

// The GPUs' memory in use by every process (Windows' "GPU Adapter Memory" performance counters, summed over the
// adapters; pdh.dll loaded here, no import): "Shared Usage" is system RAM, "Dedicated Usage" video memory. False when
// the counters are not there. The first call adds the counters (once); each call after collects them.
struct RamLineGpuCounters {
	uint64_t shared = 0, dedicated = 0;
};
bool RamLineGpuSystem(RamLineGpuCounters* out) {
	using OpenQuery  = LONG(WINAPI*)(const wchar_t*, DWORD_PTR, HANDLE*);
	using AddCounter = LONG(WINAPI*)(HANDLE, const wchar_t*, DWORD_PTR, HANDLE*);
	using Collect    = LONG(WINAPI*)(HANDLE);
	using GetArray   = LONG(WINAPI*)(HANDLE, DWORD, DWORD*, DWORD*, void*);
	// (PDH_FMT_COUNTERVALUE_ITEM_W on x64: the instance's name, the status, the value.)
	struct Item {
		wchar_t* name;
		uint32_t status;
		uint32_t padding;
		int64_t  value;
	};
	static_assert(sizeof(Item) == 24);
	struct State {
		bool     ok           = false;
		Collect  collect      = nullptr;
		GetArray get_array    = nullptr;
		HANDLE   query        = nullptr;
		HANDLE   shared_usage = nullptr, dedicated_usage = nullptr;
	};
	static const State state = [] {
		State   s;
		HMODULE pdh = LoadLibraryW(L"pdh.dll");
		if (pdh == nullptr) return s;
		const auto open = reinterpret_cast<OpenQuery>(reinterpret_cast<void*>(GetProcAddress(pdh, "PdhOpenQueryW")));
		const auto add  = reinterpret_cast<AddCounter>(reinterpret_cast<void*>(GetProcAddress(pdh, "PdhAddEnglishCounterW")));
		s.collect       = reinterpret_cast<Collect>(reinterpret_cast<void*>(GetProcAddress(pdh, "PdhCollectQueryData")));
		s.get_array     = reinterpret_cast<GetArray>(reinterpret_cast<void*>(GetProcAddress(pdh, "PdhGetFormattedCounterArrayW")));
		if (open == nullptr || add == nullptr || s.collect == nullptr || s.get_array == nullptr) return s;
		if (open(nullptr, 0, &s.query) != 0 || s.query == nullptr) return s;
		if (add(s.query, L"\\GPU Adapter Memory(*)\\Shared Usage", 0, &s.shared_usage) != 0) return s;
		if (add(s.query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &s.dedicated_usage) != 0) return s;
		s.ok = true;
		return s;
	}();
	if (!state.ok || state.collect(state.query) != 0) return false;
	static std::vector<uint8_t> buffer; // (this thread's only)
	const auto                  sum = [&](HANDLE counter, uint64_t* total) {
		constexpr DWORD FormatLarge = 0x00000400; // PDH_FMT_LARGE
		DWORD           bytes = 0, count = 0;
		(void)state.get_array(counter, FormatLarge, &bytes, &count, nullptr); // (PDH_MORE_DATA: the size wanted)
		if (bytes == 0 || bytes > (1u << 20u)) return false;
		if (buffer.size() < bytes) buffer.resize(bytes);
		count = 0;
		if (state.get_array(counter, FormatLarge, &bytes, &count, buffer.data()) != 0) return false;
		if (static_cast<uint64_t>(count) * sizeof(Item) > buffer.size()) return false;
		*total = 0;
		for (DWORD i = 0; i < count; i++) {
			Item item {};
			std::memcpy(&item, buffer.data() + static_cast<size_t>(i) * sizeof(Item), sizeof(Item));
			if (item.status <= 1 && item.value > 0) *total += static_cast<uint64_t>(item.value); // (valid or new data)
		}
		return true;
	};
	return sum(state.shared_usage, &out->shared) && sum(state.dedicated_usage, &out->dedicated);
}

void RamLineThread() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.RamLine");
	std::printf("RAM line: on (KYTY_RAM_LINE=1): a \"[ram]\" line every 10 s (system calls only: no lock of the emulator's, "
	            "no walk of its memory)\n");
	std::fflush(stdout);
	std::vector<uint8_t> buffer;
	uint64_t             last_faults = 0, last_dropped = 0, last_calls = 0, last_reset_us = 0;
	bool                 first = true;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(10));
		const auto   t0 = std::chrono::steady_clock::now();
		MEMORYSTATUSEX status {};
		status.dwLength = sizeof(status);
		if (GlobalMemoryStatusEx(&status) == 0) continue;
		PERFORMANCE_INFORMATION perf {};
		perf.cb = sizeof(perf);
		const bool     have_perf = GetPerformanceInfo(&perf, sizeof(perf)) != 0;
		const uint64_t page      = have_perf && perf.PageSize != 0 ? perf.PageSize : 4096;
		uint64_t       free_pages = 0, standby_pages = 0, modified_pages = 0;
		const bool     have_lists = RamLinePageLists(&free_pages, &standby_pages, &modified_pages);
		RamLineProcesses procs;
		RamLineReadProcesses(&buffer, &procs);
		uint64_t guest[11] {};
		Libs::LibKernel::Memory::RamLineStats(guest, 11);
		const uint64_t store = GuestLazyCommit::Get().mapped.load(std::memory_order_relaxed);
		const double   ms    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

		const auto mib = [](uint64_t bytes) { return static_cast<unsigned long long>(bytes / MiB); };
		char       line[2400]; // (ram6: 1400 -> 2400, the KYTY_DS_FREED_RELEASE part)
		size_t     used = 0;
		RamLineAdd(line, sizeof(line), &used, "[ram] 10 s, MiB (KYTY_RAM_LINE=1, %.1f ms): disponible %llu de %llu", ms, mib(status.ullAvailPhys),
		    mib(status.ullTotalPhys));
		if (have_lists)
			RamLineAdd(line, sizeof(line), &used, " (libre %llu, standby %llu, modificada %llu)", mib(free_pages * page), mib(standby_pages * page),
			    mib(modified_pages * page));
		else
			RamLineAdd(line, sizeof(line), &used, " (listas de paginas: Windows no las da)");
		int64_t rest      = 0; // ("sin atribuir", MiB; the "[ram+]" line takes its known parts off)
		bool    have_rest = false;
		if (procs.ok) {
			const uint64_t shared_ws = procs.own_ws > procs.own_private ? procs.own_ws - procs.own_private : 0;
			RamLineAdd(line, sizeof(line), &used, "; emulador en RAM %llu = privado %llu + almacen del juego %llu", mib(procs.own_ws), mib(procs.own_private),
			    mib(shared_ws));
			if (first) RamLineAdd(line, sizeof(line), &used, ", lecturas de disco por paginacion %llu desde el arranque", static_cast<unsigned long long>(procs.own_hard_faults));
			else
				RamLineAdd(line, sizeof(line), &used, ", lecturas de disco por paginacion %llu", static_cast<unsigned long long>(procs.own_hard_faults - last_faults));
			last_faults = procs.own_hard_faults;
			RamLineAdd(line, sizeof(line), &used, "; fuera del emulador: compresion de memoria de Windows %llu, otros %u procesos %llu (", mib(procs.compression_ws),
			    procs.others, mib(procs.others_ws));
			for (size_t i = 0; i < 3; i++)
				if (procs.top[i].ws != 0) RamLineAdd(line, sizeof(line), &used, "%s%s %llu", i == 0 ? "" : ", ", procs.top[i].name, mib(procs.top[i].ws));
			RamLineAdd(line, sizeof(line), &used, ")");
			if (have_perf) {
				// (What no working set, list or pool accounts for: drivers' locked memory - the GPU's "shared" memory -,
				// the paged pool and system cache in RAM. Pages shared by processes count in each: it can come out low.)
				rest = static_cast<int64_t>(status.ullTotalPhys / MiB) - static_cast<int64_t>(status.ullAvailPhys / MiB) -
				       static_cast<int64_t>(modified_pages * page / MiB) -
				       static_cast<int64_t>((procs.own_ws + procs.compression_ws + procs.others_ws) / MiB) -
				       static_cast<int64_t>(perf.KernelNonpaged * page / MiB);
				have_rest = true;
				RamLineAdd(line, sizeof(line), &used, ", nucleo no paginado %llu, sin atribuir %lld", mib(perf.KernelNonpaged * page), static_cast<long long>(rest));
			}
		} else {
			RamLineAdd(line, sizeof(line), &used, "; procesos: Windows no los da");
		}
		if (have_perf) RamLineAdd(line, sizeof(line), &used, "; commit del sistema %llu de %llu", mib(perf.CommitTotal * page), mib(perf.CommitLimit * page));
		if (store != 0) RamLineAdd(line, sizeof(line), &used, ", almacen del juego con commit %llu", mib(store));
		if (guest[0] != 0) {
			RamLineAdd(line, sizeof(line), &used, "; liberado por el juego y soltado %llu en total (+%llu; modo %llu", mib(guest[1]), mib(guest[1] - last_dropped),
			    static_cast<unsigned long long>(guest[0]));
			if (guest[0] == 2)
				RamLineAdd(line, sizeof(line), &used, ", %llu llamadas, %.1f ms, la mas larga %.2f ms", static_cast<unsigned long long>(guest[5] - last_calls),
				    static_cast<double>(guest[6] - last_reset_us) / 1000.0, static_cast<double>(guest[7]) / 1000.0);
			else
				RamLineAdd(line, sizeof(line), &used, ", en cola %llu", mib(guest[2]));
			RamLineAdd(line, sizeof(line), &used, "; no soltado: con otro mapeo %llu, rechazado %llu)", mib(guest[3]), mib(guest[4]));
			last_dropped  = guest[1];
			last_calls    = guest[5];
			last_reset_us = guest[6];
		} else {
			RamLineAdd(line, sizeof(line), &used, "; liberado por el juego y soltado: apagado (KYTY_DS_FREED_DISCARD)");
		}
		{
			// (ram6) KYTY_DS_FREED_RELEASE: what the game holds dead, given back (or, =2, what would be).
			char released[1024];
			Libs::LibKernel::Memory::RamLineFreedRelease(released, sizeof(released));
			if (released[0] != 0) RamLineAdd(line, sizeof(line), &used, "; %s", released);
		}
		if (guest[8] != 0)
			RamLineAdd(line, sizeof(line), &used, "; asignador del juego: bloques liberados que siguen mapeados %llu, paginas libres sin mapear %llu", mib(guest[9]),
			    mib(guest[10]));
		first = false;
		std::printf("%s\n", line);
		// (ram4) "[ram+]": what "sin atribuir" and the emulator's private memory are made of, as far as system calls
		// tell (no lock of the emulator's but VMA's own counters, no walk of memory).
		{
			const auto         t1 = std::chrono::steady_clock::now();
			RamLineSystem      sys;
			const bool         have_sys = RamLineSystemResident(status.ullAvailPhys, page, &sys);
			RamLineGpuCounters gpu;
			const bool         have_gpu = RamLineGpuSystem(&gpu);
			uint64_t           own_video = 0, own_budget = 0, own_shared = 0;
			const bool         have_own  = LocalPlatform::QueryVideoMemory(&own_video, &own_budget);
			const bool         have_own_shared = LocalPlatform::QuerySharedGpuMemory(&own_shared);
			uint64_t           vk_video = 0, vk_system = 0, vk_ballast = 0;
			const bool         have_vk = Libs::Graphics::RamLineVulkan(&vk_video, &vk_system, &vk_ballast);
			uint64_t           heap[4] {};
			LocalPlatform::RamLineHeap(heap, 4);
			const double more_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
			used                 = 0;
			RamLineAdd(line, sizeof(line), &used, "[ram+] MiB (%.1f ms)", more_ms);
			uint64_t system_resident = 0;
			if (have_sys) {
				system_resident = (sys.cache_pages + sys.paged_pool_pages + sys.code_pages + sys.driver_pages) * page;
				RamLineAdd(line, sizeof(line), &used,
				    ": sistema en RAM %llu = cache de archivos %llu + bloque paginado del nucleo %llu (de %llu reservados) + codigo del "
				    "nucleo %llu + drivers %llu",
				    mib(system_resident), mib(sys.cache_pages * page), mib(sys.paged_pool_pages * page), mib(sys.paged_pool_virtual_pages * page),
				    mib(sys.code_pages * page), mib(sys.driver_pages * page));
			} else {
				RamLineAdd(line, sizeof(line), &used, ": sistema en RAM: Windows no lo da");
			}
			if (have_perf)
				RamLineAdd(line, sizeof(line), &used, "; segun GetPerformanceInfo: cache del sistema %llu, nucleo paginado %llu, nucleo total %llu",
				    mib(perf.SystemCache * page), mib(perf.KernelPaged * page), mib(perf.KernelTotal * page));
			if (have_gpu)
				RamLineAdd(line, sizeof(line), &used, "; GPU de todos los procesos: compartida (RAM) %llu, dedicada %llu", mib(gpu.shared),
				    mib(gpu.dedicated));
			else
				RamLineAdd(line, sizeof(line), &used, "; GPU de todos los procesos: contadores no disponibles");
			if (have_own)
				RamLineAdd(line, sizeof(line), &used, "; GPU de este proceso: dedicada %llu de %llu", mib(own_video), mib(own_budget));
			if (have_own_shared) RamLineAdd(line, sizeof(line), &used, "%s compartida (RAM) %llu", have_own ? "," : "; GPU de este proceso:", mib(own_shared));
			if (have_vk)
				RamLineAdd(line, sizeof(line), &used, "; Vulkan del emulador: bloques en memoria de video %llu, en memoria del sistema %llu, lastre %llu",
				    mib(vk_video), mib(vk_system), mib(vk_ballast));
			if (heap[0] != 0)
				RamLineAdd(line, sizeof(line), &used, "; monton C++ del emulador (commit, hace %llu s): %llu en %llu segmentos, el mayor %llu",
				    static_cast<unsigned long long>((GetTickCount64() - heap[3]) / 1000), mib(heap[0]), static_cast<unsigned long long>(heap[1]),
				    mib(heap[2]));
			if (have_rest && have_sys) {
				// (What is left once the system's working set is taken off; the GPU's shared memory of all processes is
				// printed beside it, not taken off: whether a process's working set counts it is what this line is to show.)
				const int64_t left = rest - static_cast<int64_t>(system_resident / MiB);
				RamLineAdd(line, sizeof(line), &used, "; sin atribuir %lld - sistema en RAM = %lld", static_cast<long long>(rest),
				    static_cast<long long>(left));
				if (have_gpu)
					RamLineAdd(line, sizeof(line), &used, " (- GPU compartida = %lld)", static_cast<long long>(left - static_cast<int64_t>(gpu.shared / MiB)));
			}
			std::printf("%s\n", line);
		}
		std::fflush(stdout);
	}
}

// (ram16) The census of the game's memory in RAM, by allocator (the memory map's labels: KYTY_MEMMAP, on by default),
// into `out`: a sample per `stride` (64 KiB, more when the labels span more than 4 GiB a 65536 samples) of every range,
// K32QueryWorkingSetEx 1024 samples a call, the CPU yielded between calls (each call holds the working-set lock the game's
// page faults wait for: ~0.1-0.3 ms). With native guest memory a game page in the working set is the game's own RAM;
// compressed or paged-out pages are not in it. `game_ram`: the sum (the labels below do not overlap: the ring and the
// pool, inside cpupage / gpupage, are left out). False when the map has no snapshot.
bool Ram16Census(char* out, size_t size, uint64_t* game_ram, double* ms, size_t* samples_out, uint64_t* stride_out) {
	static const char* const labels[] = {"VirtualMemo", "cpuheap", "gpuheap", "cpupage", "gpupage", "cpubkt",
	                                     "gpubkt",      "DefaultTagg", "RTSharedAll", "RTPersistAl"};
	constexpr size_t         Labels   = sizeof(labels) / sizeof(labels[0]);
	constexpr size_t         MaxPairs = 8192;
	const auto               t0       = std::chrono::steady_clock::now();
	std::vector<uint64_t>    pairs(MaxPairs * 2);
	std::vector<std::pair<uint64_t, uint64_t>> ranges[Labels];
	uint64_t                 range_bytes[Labels] {}, total = 0;
	for (size_t l = 0; l < Labels; l++) {
		const size_t n = Loader::DemonsSoulsMemMap::RangesOf(&labels[l], 1, pairs.data(), MaxPairs);
		if (n == SIZE_MAX) return false;
		for (size_t i = 0; i < n; i++) {
			if (pairs[2 * i + 1] <= pairs[2 * i]) continue;
			ranges[l].emplace_back(pairs[2 * i], pairs[2 * i + 1]);
			range_bytes[l] += pairs[2 * i + 1] - pairs[2 * i];
		}
		total += range_bytes[l];
	}
	constexpr uint64_t Granule    = uint64_t {64} << 10u;
	constexpr uint64_t MaxSamples = uint64_t {1} << 16u;
	const uint64_t     stride     = std::max(Granule, ((total / MaxSamples) + Granule - 1) & ~(Granule - 1));
	struct Sample {
		uint64_t page   = 0;
		uint32_t weight = 0;
		uint16_t label  = 0;
	};
	std::vector<Sample> samples;
	samples.reserve(static_cast<size_t>(std::min<uint64_t>(total / stride + 1024, MaxSamples * 2)));
	for (size_t l = 0; l < Labels; l++)
		for (const auto& [begin, end]: ranges[l])
			for (uint64_t at = begin; at < end && samples.size() < MaxSamples * 2; at += stride)
				samples.push_back({at & ~uint64_t {0xfff}, static_cast<uint32_t>(std::min(stride, end - at)), static_cast<uint16_t>(l)});
	constexpr size_t                              Batch = 1024;
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries(Batch);
	uint64_t                                      in_ram[Labels] {};
	for (size_t begin = 0; begin < samples.size(); begin += Batch) {
		const size_t n = std::min(Batch, samples.size() - begin);
		for (size_t i = 0; i < n; i++) {
			entries[i]                = {};
			entries[i].VirtualAddress = reinterpret_cast<PVOID>(samples[begin + i].page);
		}
		if (begin != 0) SwitchToThread();
		if (K32QueryWorkingSetEx(GetCurrentProcess(), entries.data(), static_cast<DWORD>(n * sizeof(entries[0]))) == 0) return false;
		for (size_t i = 0; i < n; i++)
			if ((entries[i].VirtualAttributes.Flags & 1u) != 0) in_ram[samples[begin + i].label] += samples[begin + i].weight;
	}
	constexpr double GiBd = 1024.0 * 1024.0 * 1024.0;
	size_t           used = 0;
	uint64_t         sum  = 0;
	out[0]                = '\0';
	for (size_t l = 0; l < Labels; l++) {
		sum += in_ram[l];
		RamLineAdd(out, size, &used, "%s%s %.2f (%.2f)", l == 0 ? "" : ", ", labels[l], static_cast<double>(in_ram[l]) / GiBd,
		           static_cast<double>(range_bytes[l]) / GiBd);
	}
	*game_ram    = sum;
	*ms          = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	*samples_out = samples.size();
	*stride_out  = stride;
	return true;
}

// ---- (ram18) KYTY_RAM18=1 (texture-offload.h, Ram18On): the cold-range thread ----
// Every 5 s, on a lowest-priority thread: (1) the cold ranges the render thread took since (watchers on) written to the
// disk store and dropped (TextureOffload::DiscardRangeNative: a range that cannot be written stays in RAM); (2) the
// committed private pages of the labels' ranges (VirtualQuery; read-write or read-only only: no-access pages are watched
// by someone - a taken range, the GPU's kept bytes - and left alone), in 256 KiB blocks, a working-set sample every 64 KiB;
// a block with no sample in the working set for KYTY_RAM18_COLD_S is cold: Windows trimmed it (its bytes in the
// compression store, or never written); (3) cold blocks side by side merged into ranges of up to 1 MiB and proposed, up to
// KYTY_RAM18_MB a pass from where the last pass stopped (TextureOffload::Ram18Propose; the render thread takes them).
struct Ram18Pass {
	std::atomic<uint64_t> committed {0}, resident {0}, out {0}, ready {0}, proposed {0}, pass_us {0}, drop_us {0}, drops {0};
	std::atomic<uint64_t> passes {0};
};
Ram18Pass g_ram18;

// ---- (ram20) KYTY_RAM20=1: the game's memory that video memory holds a copy of, out of RAM ----
// One pass every 5 s on the RAM18 thread (lowest priority), before RAM18's own. Candidates: 256 KiB blocks of the game's
// allocators (the census' labels), committed private read-write or read-only pages only (no-access ones are someone's:
// a taken range, the GPU's kept bytes or its read watchers - their newer bytes are on the GPU, not ours to drop -), whose
// four 64 KiB granules are all under a GPU cache range of the last GPU residency pass (ram-guard.cpp's measurement,
// every 30 s). A block is taken when:
//  - cold: no working-set sample of it for KYTY_RAM20_COLD_S (default 10): in Windows' compression store, the page file,
//    or never written - the case of 1.07-1.45 GB "accessible out of RAM with a copy in the GPU" [M, 11-17 01:32] -; or
//  - warm (KYTY_RAM20_WARM, default 1): in RAM, all four granules images (sampled textures, render targets, GPU-written
//    images; buffers too with KYTY_RAM20_WARM_BUF=1) in this pass and the one before, and in an allocator the CPU does
//    not write every frame: VirtualMemo, gpupage, gpuheap, RTSharedAll, RTPersistAl (Ghidra 01.007.000: VirtualMemo
//    written only by the file read 0x8fed14 and the chunk memcpy 0x901459; RT arenas reserved whole at 0x91c680 through
//    0x806b30 for the GPU; DefaultTagg / cpuheap / cpubkt / cpupage written every frame - TaggedFrameAllocator 0x811f60 -:
//    cold only there).
// Taken ranges go the RAM18 way (texture-offload.h): read watchers on by the render thread (OffloadScan), then written to
// the disk store and reset by this thread; any touch - a game thread's fault, an emulator window or guarded read, a write
// back of GPU results (NativeGuardedWrite asks first with KYTY_RAM20), a kernel write of the game's buffer (fileSystem.cpp
// bounces it through a guarded read) - reads it back from the file on its own thread. Never from the GPU: with native
// memory the RAM copy is the one the CPU last saw and the GPU's newer bytes stay behind their own read watchers (pages
// with them are no-access and never taken); the file gives back exactly the bytes that were dropped.
struct Ram20Pass_ {
	std::atomic<uint64_t> gpu_committed {0}, warm {0}, cold_out {0}, ready_cold {0}, ready_warm {0}, proposed {0}, pass_us {0};
	std::atomic<uint64_t> snapshot {0}, passes {0};
};
Ram20Pass_ g_ram20;

void Ram20Pass() {
	using namespace GpuResidency;
	static const int      cold_s = [] {
        const char* v = Env("KYTY_RAM20_COLD_S");
        return v == nullptr ? 10 : std::clamp(std::atoi(v), 1, 3600);
	}();
	static const bool     warm_on  = Env("KYTY_RAM20_WARM") == nullptr || Env("KYTY_RAM20_WARM")[0] != '0';
	static const bool     warm_buf = Env("KYTY_RAM20_WARM_BUF") != nullptr && Env("KYTY_RAM20_WARM_BUF")[0] == '1';
	static const uint64_t budget   = EnvMiB("KYTY_RAM20_MB", 256 * MiB);
	static const char* const labels[] = {"VirtualMemo", "cpuheap", "gpuheap", "cpupage", "gpupage", "cpubkt",
	                                     "gpubkt",      "DefaultTagg", "RTSharedAll", "RTPersistAl"};
	static const bool        warm_label[] = {true, false, true, false, true, false, false, false, true, true};
	constexpr size_t         Labels = sizeof(labels) / sizeof(labels[0]);
	constexpr uint64_t       Granule = uint64_t {64} << 10u, Block = uint64_t {256} << 10u, Window = uint64_t {1} << 20u;
	constexpr size_t         PerBlock = Block / Granule, MaxPairs = 8192, Batch = 1024;
	// The granules under a GPU range: a bit per kind, this pass's and the one before (rebuilt when a new pass came).
	static std::unordered_map<uint64_t, uint8_t> now_map, before_map;
	static uint64_t                              have_pass = 0;
	static std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> since;
	static std::vector<uint64_t>                 pairs(MaxPairs * 2);
	static uint64_t                              cursor = 0;
	const auto t0 = std::chrono::steady_clock::now();
	{
		auto&              snap = Ram20Ranges();
		std::vector<Range> ranges;
		uint64_t           pass = 0;
		{
			std::lock_guard lock(snap.mutex);
			if (snap.pass != have_pass) {
				ranges = snap.ranges;
				pass   = snap.pass;
			}
		}
		if (pass != 0) {
			before_map.swap(now_map);
			now_map.clear();
			now_map.reserve(ranges.size() * 4 + 1024);
			for (const auto& range: ranges) {
				if (range.size == 0 || UINT64_MAX - range.address < range.size || range.kind >= Kinds) continue;
				for (uint64_t g = range.address / Granule; g <= (range.address + range.size - 1) / Granule; g++)
					now_map[g] |= static_cast<uint8_t>(1u << range.kind);
			}
			have_pass = pass;
			g_ram20.snapshot.store(pass, std::memory_order_relaxed);
		}
	}
	if (now_map.empty()) return; // (no answered residency pass yet)
	// The labels' committed read-write / read-only blocks wholly under GPU ranges.
	struct Cand {
		uint64_t block;
		uint8_t  label, mask;
		bool     run_end, stable;
	};
	std::vector<Cand> cands;
	uint64_t          gpu_committed = 0;
	for (size_t l = 0; l < Labels; l++) {
		const size_t n = Loader::DemonsSoulsMemMap::RangesOf(&labels[l], 1, pairs.data(), MaxPairs);
		if (n == SIZE_MAX) return;
		for (size_t i = 0; i < n; i++) {
			const uint64_t lo = pairs[2 * i], hi = pairs[2 * i + 1];
			for (uint64_t at = lo; at < hi;) {
				MEMORY_BASIC_INFORMATION info {};
				if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
				const uint64_t stop = std::min(hi, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
				if (stop <= at) break;
				const DWORD protect = info.Protect & 0xffu;
				if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && (info.Protect & PAGE_GUARD) == 0 &&
				    (protect == PAGE_READWRITE || protect == PAGE_READONLY)) {
					for (uint64_t block = (at + Block - 1) & ~(Block - 1); block + Block <= stop; block += Block) {
						uint8_t mask = 0;
						bool    all = true, stable = true;
						for (size_t k = 0; k < PerBlock && all; k++) {
							const auto found = now_map.find(block / Granule + k);
							all              = found != now_map.end() && found->second != 0;
							if (!all) break;
							mask |= found->second;
							const auto old = before_map.find(block / Granule + k);
							stable         = stable && old != before_map.end() && (old->second & found->second) != 0;
						}
						if (!all) continue;
						gpu_committed += Block;
						cands.push_back({block, static_cast<uint8_t>(l), mask, block + 2 * Block > stop, stable});
					}
				}
				at = stop;
			}
			if (cands.size() > (size_t {64} << 10u)) break;
		}
	}
	std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.block < b.block; });
	// Working-set samples: one page in each 64 KiB granule.
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries(cands.size() * PerBlock);
	for (size_t i = 0; i < cands.size(); i++)
		for (size_t k = 0; k < PerBlock; k++) {
			entries[i * PerBlock + k]                = {};
			entries[i * PerBlock + k].VirtualAddress = reinterpret_cast<PVOID>(cands[i].block + k * Granule);
		}
	for (size_t begin = 0; begin < entries.size(); begin += Batch) {
		const size_t count = std::min(Batch, entries.size() - begin);
		if (begin != 0) SwitchToThread();
		if (K32QueryWorkingSetEx(GetCurrentProcess(), entries.data() + begin, static_cast<DWORD>(count * sizeof(entries[0]))) == 0) return;
	}
	// The kind a range is counted under: the first of render target, GPU-written image, texture, GPU-written buffer, buffer.
	const auto kind_of = [](uint8_t mask) -> uint8_t {
		for (const Kind k: {Target, Storage, Texture, GpuWrittenBuffer, GpuBuffer})
			if ((mask & (1u << k)) != 0) return static_cast<uint8_t>(k);
		return static_cast<uint8_t>(GpuBuffer);
	};
	constexpr uint8_t ImageMask = (1u << Texture) | (1u << Target) | (1u << Storage);
	const auto        now       = std::chrono::steady_clock::now();
	std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> seen;
	std::vector<uint8_t> take(cands.size(), 0);
	uint64_t             warm = 0, cold_out = 0, ready_cold = 0, ready_warm = 0;
	for (size_t i = 0; i < cands.size(); i++) {
		size_t in = 0;
		for (size_t k = 0; k < PerBlock; k++) in += (entries[i * PerBlock + k].VirtualAttributes.Flags & 1u) != 0 ? 1 : 0;
		const auto& c = cands[i];
		if (in == 0) {
			cold_out += Block;
			const auto found = since.find(c.block);
			const auto first = found != since.end() ? found->second : now;
			seen.emplace(c.block, first);
			if (now - first >= std::chrono::seconds(cold_s)) {
				take[i] = 1;
				ready_cold += Block;
			}
			continue;
		}
		warm += in * Granule;
		const uint8_t allowed = warm_buf ? static_cast<uint8_t>(ImageMask | (1u << GpuBuffer) | (1u << GpuWrittenBuffer)) : ImageMask;
		if (warm_on && warm_label[c.label] && c.stable && (c.mask & ~allowed) == 0) {
			take[i] = 1;
			ready_warm += Block;
		}
	}
	since.swap(seen);
	// Merged by kind within 1 MiB windows (a range starts at the same place from pass to pass: its serial too), up to the
	// budget from where the last pass stopped.
	std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> ranges;
	const auto start    = static_cast<size_t>(std::lower_bound(cands.begin(), cands.end(), cursor, [](const Cand& c, uint64_t v) { return c.block < v; }) - cands.begin());
	uint64_t   proposed = 0, last = cursor;
	for (size_t step = 0; step < cands.size() && proposed < budget; step++) {
		const size_t i = (start + step) % cands.size();
		if (take[i] == 0) continue;
		const uint8_t r20 = static_cast<uint8_t>(kind_of(cands[i].mask) + 1);
		auto*         back = ranges.empty() ? nullptr : &ranges.back();
		if (back != nullptr && std::get<0>(*back) + std::get<1>(*back) == cands[i].block && std::get<1>(*back) + Block <= Window &&
		    (std::get<0>(*back) & ~(Window - 1)) == (cands[i].block & ~(Window - 1)) && std::get<2>(*back) == r20 && i != 0 &&
		    !cands[i - 1].run_end) {
			std::get<1>(*back) += Block;
		} else {
			ranges.emplace_back(cands[i].block, Block, r20);
		}
		proposed += Block;
		last = cands[i].block + Block;
	}
	cursor = proposed < budget ? 0 : last;
	TextureOffload::Ram20Propose(std::move(ranges));
	g_ram20.gpu_committed.store(gpu_committed, std::memory_order_relaxed);
	g_ram20.warm.store(warm, std::memory_order_relaxed);
	g_ram20.cold_out.store(cold_out, std::memory_order_relaxed);
	g_ram20.ready_cold.store(ready_cold, std::memory_order_relaxed);
	g_ram20.ready_warm.store(ready_warm, std::memory_order_relaxed);
	g_ram20.proposed.store(proposed, std::memory_order_relaxed);
	g_ram20.pass_us.store(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
	                      std::memory_order_relaxed);
	g_ram20.passes.fetch_add(1, std::memory_order_relaxed);
}

// (ram20) The "[ram20]" line into `out` ("" when nothing was read yet): what is out of RAM now by kind, what went and came
// back since the last line, the candidates the last pass saw, and the RAM the user counts (emulator + compression) now
// and against the first line.
void Ram20Line(char* out, size_t size, const RamLineProcesses& procs, int seconds) {
	static TextureOffload::Ram20View last, view;
	static bool                      have_last = false, have_view = false;
	static uint64_t                  first_real = 0;
	TextureOffload::Ram20View        now;
	const bool                       fresh = TextureOffload::Ram20Read(&now);
	if (fresh) {
		view      = now;
		have_view = true;
	}
	out[0] = '\0';
	if (!have_view) return;
	constexpr double GiBd = 1024.0 * 1024.0 * 1024.0, MiBd = 1024.0 * 1024.0;
	static const char* const names[] = {"tex", "RT", "imgGPU", "buf", "bufGPU"}; // (GpuResidency::Kind order)
	const auto d = [&](uint64_t a, uint64_t b) { return have_last ? a - b : a; };
	uint64_t   all_now = 0, taken = 0, reads = 0, read_bytes = 0;
	for (size_t k = 0; k < TextureOffload::Ram20View::Kinds; k++) {
		all_now += view.now[k];
		taken += d(view.taken[k], last.taken[k]);
		reads += d(view.reads[k], last.reads[k]);
		read_bytes += d(view.read_bytes[k], last.read_bytes[k]);
	}
	size_t used = 0;
	RamLineAdd(out, size, &used, "[ram20] %d s: soltado ahora %.2f GB%s (pico %.2f) =", seconds, static_cast<double>(all_now) / GiBd,
	           fresh ? "" : " (lectura anterior)", static_cast<double>(view.peak) / GiBd);
	for (size_t k = 0; k < TextureOffload::Ram20View::Kinds; k++)
		RamLineAdd(out, size, &used, " %s %.2f", names[k], static_cast<double>(view.now[k]) / GiBd);
	RamLineAdd(out, size, &used, "; tomados +%llu (MiB:", static_cast<unsigned long long>(taken));
	for (size_t k = 0; k < TextureOffload::Ram20View::Kinds; k++)
		RamLineAdd(out, size, &used, " %s %.0f", names[k], static_cast<double>(d(view.taken_bytes[k], last.taken_bytes[k])) / MiBd);
	RamLineAdd(out, size, &used,
	           "); al disco +%.0f MiB, en cero +%.0f MiB (solo soltados); vueltos por un toque +%llu (%.0f MiB, %.1f ms, el mayor %.1f "
	           "ms en total; MiB:",
	           static_cast<double>(d(view.written_bytes, last.written_bytes)) / MiBd,
	           static_cast<double>(d(view.zero_bytes, last.zero_bytes)) / MiBd, static_cast<unsigned long long>(reads),
	           static_cast<double>(read_bytes) / MiBd, static_cast<double>(d(view.read_us, last.read_us)) / 1000.0,
	           static_cast<double>(view.read_us_max) / 1000.0);
	for (size_t k = 0; k < TextureOffload::Ram20View::Kinds; k++)
		RamLineAdd(out, size, &used, " %s %.0f", names[k], static_cast<double>(d(view.read_bytes[k], last.read_bytes[k])) / MiBd);
	RamLineAdd(out, size, &used,
	           "); no escritos (se quedan en RAM) %llu; PERDIDAS %llu; ultima pasada: con copia en GPU comprometido %.2f GB, en RAM "
	           "%.2f, fuera de RAM %.2f, listos frios %.2f + tibios %.2f, propuestos %.0f MiB (%.1f ms, residencia #%llu)",
	           static_cast<unsigned long long>(view.failed), static_cast<unsigned long long>(view.lost),
	           static_cast<double>(g_ram20.gpu_committed.load()) / GiBd, static_cast<double>(g_ram20.warm.load()) / GiBd,
	           static_cast<double>(g_ram20.cold_out.load()) / GiBd, static_cast<double>(g_ram20.ready_cold.load()) / GiBd,
	           static_cast<double>(g_ram20.ready_warm.load()) / GiBd, static_cast<double>(g_ram20.proposed.load()) / MiBd,
	           static_cast<double>(g_ram20.pass_us.load()) / 1000.0, static_cast<unsigned long long>(g_ram20.snapshot.load()));
	if (procs.ok) {
		const uint64_t real = procs.own_private + procs.compression_ws;
		if (first_real == 0) first_real = real;
		RamLineAdd(out, size, &used, "; RAM real: emulador %.2f + compresion de Windows %.2f = %.2f GB (al empezar %.2f, %+.2f)",
		           static_cast<double>(procs.own_private) / GiBd, static_cast<double>(procs.compression_ws) / GiBd,
		           static_cast<double>(real) / GiBd, static_cast<double>(first_real) / GiBd,
		           (static_cast<double>(real) - static_cast<double>(first_real)) / GiBd);
	}
	if (fresh) {
		last      = view;
		have_last = true;
	}
}

void Ram18Thread() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.Ram18");
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
	const int      cold_s = [] {
        const char* v = Env("KYTY_RAM18_COLD_S");
        return v == nullptr ? 10 : std::clamp(std::atoi(v), 1, 3600);
	}();
	const uint64_t budget = EnvMiB("KYTY_RAM18_MB", 256 * MiB);
	// The labels (comma-separated; default the streamed textures' zone).
	std::vector<std::string> names;
	{
		const char* v    = Env("KYTY_RAM18_LABELS");
		std::string list = v != nullptr ? v : "VirtualMemo";
		for (size_t at = 0; at <= list.size();) {
			const size_t comma = std::min(list.find(',', at), list.size());
			if (comma > at) names.push_back(list.substr(at, comma - at));
			at = comma + 1;
		}
	}
	std::vector<const char*> labels;
	for (const auto& name: names) labels.push_back(name.c_str());
	constexpr uint64_t Block = uint64_t {256} << 10u, Sample = uint64_t {64} << 10u, Range = uint64_t {1} << 20u;
	constexpr size_t   MaxPairs = 16384;
	std::vector<uint64_t>                                          pairs(MaxPairs * 2);
	std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> since, seen;
	std::vector<uint64_t>                                          blocks; // (block starts, ascending; a run ends where a region does)
	std::vector<uint8_t>                                           run_end;
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION>                  entries;
	uint64_t                                                       cursor = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		if (!TextureOffload::Ram18On()) { // (read once the emulator set its switches; off: the thread ends)
			std::printf("[ram18] apagado: necesita el offload nativo (KYTY_RAIZ / KYTY_NATIVE_MEMORY) y KYTY_RAM17=1\n");
			std::fflush(stdout);
			return;
		}
		// (1) The ranges taken since: written and dropped.
		auto drop_start = std::chrono::steady_clock::now();
		const auto drops = TextureOffload::Ram18TakeDrops();
		for (const auto begin: drops) {
			(void)TextureOffload::DiscardRangeNative(begin);
			SwitchToThread();
		}
		g_ram18.drop_us.store(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - drop_start).count()),
		                      std::memory_order_relaxed);
		g_ram18.drops.fetch_add(drops.size(), std::memory_order_relaxed);
		// (ram20, KYTY_RAM20) Its own pass (the memory with a copy in video memory), whatever RAM18 proposes.
		if (TextureOffload::Ram20Active()) Ram20Pass();
		if (!TextureOffload::Ram18Own()) continue; // (ram20 alone: RAM18's own VirtualMemo pass off)
		// (2) The labels' committed blocks and their working-set samples.
		const auto   t0 = std::chrono::steady_clock::now();
		const size_t n  = labels.empty() ? SIZE_MAX : Loader::DemonsSoulsMemMap::RangesOf(labels.data(), labels.size(), pairs.data(), MaxPairs);
		if (n == SIZE_MAX) continue;
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		for (size_t i = 0; i < n; i++)
			if (pairs[2 * i + 1] > pairs[2 * i]) spans.emplace_back(pairs[2 * i], pairs[2 * i + 1]);
		std::sort(spans.begin(), spans.end());
		blocks.clear();
		run_end.clear();
		uint64_t committed = 0;
		for (const auto& [lo, hi]: spans) {
			for (uint64_t at = lo; at < hi;) {
				MEMORY_BASIC_INFORMATION info {};
				if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
				const uint64_t base = reinterpret_cast<uint64_t>(info.BaseAddress);
				const uint64_t stop = std::min(hi, base + static_cast<uint64_t>(info.RegionSize));
				if (stop <= at) break;
				const DWORD protect = info.Protect & 0xffu;
				if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && (info.Protect & PAGE_GUARD) == 0 &&
				    (protect == PAGE_READWRITE || protect == PAGE_READONLY)) {
					committed += stop - at;
					const uint64_t first = (at + Block - 1) & ~(Block - 1);
					for (uint64_t block = first; block + Block <= stop; block += Block) {
						blocks.push_back(block);
						run_end.push_back(block + 2 * Block > stop ? 1 : 0);
					}
				}
				at = stop;
			}
			if (blocks.size() > (uint64_t {64} << 10u)) break; // (16 GiB of blocks: more than any label holds)
		}
		// Samples: a page every 64 KiB of every block.
		constexpr size_t PerBlock = Block / Sample, Batch = 1024;
		entries.resize(blocks.size() * PerBlock);
		for (size_t i = 0; i < blocks.size(); i++)
			for (size_t k = 0; k < PerBlock; k++) {
				entries[i * PerBlock + k]                = {};
				entries[i * PerBlock + k].VirtualAddress = reinterpret_cast<PVOID>(blocks[i] + k * Sample);
			}
		bool ok = true;
		for (size_t begin = 0; begin < entries.size() && ok; begin += Batch) {
			const size_t count = std::min(Batch, entries.size() - begin);
			if (begin != 0) SwitchToThread();
			ok = K32QueryWorkingSetEx(GetCurrentProcess(), entries.data() + begin, static_cast<DWORD>(count * sizeof(entries[0]))) != 0;
		}
		if (!ok) continue;
		// (3) Cold blocks, merged and proposed.
		const auto now      = std::chrono::steady_clock::now();
		uint64_t   resident = 0, out = 0, ready = 0;
		seen.clear();
		std::vector<uint8_t> cold(blocks.size(), 0);
		for (size_t i = 0; i < blocks.size(); i++) {
			size_t in = 0;
			for (size_t k = 0; k < PerBlock; k++) in += (entries[i * PerBlock + k].VirtualAttributes.Flags & 1u) != 0 ? 1 : 0;
			resident += in * Sample;
			if (in != 0) continue;
			out += Block;
			const auto found = since.find(blocks[i]);
			const auto first = found != since.end() ? found->second : now;
			seen.emplace(blocks[i], first);
			if (now - first >= std::chrono::seconds(cold_s)) {
				cold[i] = 1;
				ready += Block;
			}
		}
		since.swap(seen);
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		const auto start = static_cast<size_t>(std::lower_bound(blocks.begin(), blocks.end(), cursor) - blocks.begin());
		uint64_t   proposed = 0, last = cursor;
		for (size_t step = 0; step < blocks.size() && proposed < budget; step++) {
			const size_t i = (start + step) % blocks.size();
			if (cold[i] == 0) continue;
			// (Within one 1 MiB-aligned window: a range starts at the same place from pass to pass, its serial too.)
			if (!ranges.empty() && ranges.back().first + ranges.back().second == blocks[i] && ranges.back().second + Block <= Range &&
			    (ranges.back().first & ~(Range - 1)) == (blocks[i] & ~(Range - 1)) && i != 0 && run_end[i - 1] == 0) {
				ranges.back().second += Block;
			} else {
				ranges.emplace_back(blocks[i], Block);
			}
			proposed += Block;
			last = blocks[i] + Block;
		}
		cursor = proposed < budget ? 0 : last;
		TextureOffload::Ram18Propose(std::move(ranges));
		g_ram18.committed.store(committed, std::memory_order_relaxed);
		g_ram18.resident.store(resident, std::memory_order_relaxed);
		g_ram18.out.store(out, std::memory_order_relaxed);
		g_ram18.ready.store(ready, std::memory_order_relaxed);
		g_ram18.proposed.store(proposed, std::memory_order_relaxed);
		g_ram18.pass_us.store(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
		                      std::memory_order_relaxed);
		g_ram18.passes.fetch_add(1, std::memory_order_relaxed);
	}
}

// (ram18) The "[ram18]" line into `out` ("" when nothing was read yet).
void Ram18Line(char* out, size_t size, const RamLineProcesses& procs) {
	static TextureOffload::Ram18View last, view;
	static bool                      have_last = false, have_view = false;
	TextureOffload::Ram18View        now;
	const bool                       fresh = TextureOffload::Ram18Read(&now);
	if (fresh) {
		view      = now;
		have_view = true;
	}
	out[0] = '\0';
	if (!have_view) return;
	constexpr double GiBd = 1024.0 * 1024.0 * 1024.0, MiBd = 1024.0 * 1024.0;
	const auto d = [&](uint64_t TextureOffload::Ram18View::*field) { return have_last ? view.*field - last.*field : view.*field; };
	size_t used = 0;
	RamLineAdd(out, size, &used,
	           "[ram18] 5 s: memoria fria en disco ahora %.2f GB%s (pico %.2f; %llu rangos); zona %s: comprometido %.2f GB, en RAM "
	           "%.2f, fuera de RAM %.2f, fria lista %.2f (pasada %.1f ms); tomadas +%llu (%.0f MiB); escritas +%llu (%.0f MiB, %.1f "
	           "ms); en cero +%llu (%.0f MiB, solo soltadas); vueltas por un toque +%llu (%.0f MiB, %.1f ms, la mayor %.1f ms en "
	           "total); desmapeadas +%.0f MiB; no escritas %llu; PERDIDAS %llu",
	           static_cast<double>(view.now) / GiBd, fresh ? "" : " (lectura anterior)", static_cast<double>(view.peak) / GiBd,
	           static_cast<unsigned long long>(view.ranges), Env("KYTY_RAM18_LABELS") != nullptr ? Env("KYTY_RAM18_LABELS") : "VirtualMemo",
	           static_cast<double>(g_ram18.committed.load()) / GiBd, static_cast<double>(g_ram18.resident.load()) / GiBd,
	           static_cast<double>(g_ram18.out.load()) / GiBd, static_cast<double>(g_ram18.ready.load()) / GiBd,
	           static_cast<double>(g_ram18.pass_us.load()) / 1000.0, static_cast<unsigned long long>(d(&TextureOffload::Ram18View::taken)),
	           static_cast<double>(d(&TextureOffload::Ram18View::taken_bytes)) / MiBd,
	           static_cast<unsigned long long>(d(&TextureOffload::Ram18View::written)),
	           static_cast<double>(d(&TextureOffload::Ram18View::written_bytes)) / MiBd,
	           static_cast<double>(d(&TextureOffload::Ram18View::write_us)) / 1000.0,
	           static_cast<unsigned long long>(d(&TextureOffload::Ram18View::zero)),
	           static_cast<double>(d(&TextureOffload::Ram18View::zero_bytes)) / MiBd,
	           static_cast<unsigned long long>(d(&TextureOffload::Ram18View::reads)),
	           static_cast<double>(d(&TextureOffload::Ram18View::read_bytes)) / MiBd,
	           static_cast<double>(d(&TextureOffload::Ram18View::read_us)) / 1000.0, static_cast<double>(view.read_us_max) / 1000.0,
	           static_cast<double>(d(&TextureOffload::Ram18View::unmapped)) / MiBd, static_cast<unsigned long long>(view.failed),
	           static_cast<unsigned long long>(view.lost));
	if (procs.ok)
		RamLineAdd(out, size, &used, "; RAM real: emulador %.2f + compresion de Windows %.2f = %.2f GB", static_cast<double>(procs.own_private) / GiBd,
		           static_cast<double>(procs.compression_ws) / GiBd, static_cast<double>(procs.own_private + procs.compression_ws) / GiBd);
	if (fresh) {
		last      = view;
		have_last = true;
	}
}

// (ram16) KYTY_RAM16=1: "[ram16]" every 5 s, what the RAM cuts save now and where the PC's RAM is, from system calls and
// the cuts' own counters (no lock of the emulator's waited for: the offload's state is only tried), and every
// KYTY_RAM16_CENSUS_S (default 30, 0 off) the census of the game's memory in RAM ("[ram16-censo]").
// ---- (mem1) KYTY_MEM1=1 (mem1.h): the C++ heap's free blocks given back, the VirtualMemo census, the "[mem1]" line ----
struct Mem1Heap {
	std::atomic<uint64_t> passes {0}, committed {0}, busy {0}, free_bytes {0}, given {0}, given_resident {0}, given_total {0};
	std::atomic<uint64_t> refused {0}, spans {0}, lock_us {0}, lock_us_max {0}, over_8ms {0};
	std::atomic<bool>     stopped {false};
};
Mem1Heap g_mem1_heap;

// One pass over the process heap (malloc/new): under its lock (this thread raised to the highest priority meanwhile, so
// no thread waits on a descheduled holder), its free blocks of 64 KiB or more past their first 64 bytes and page-aligned
// (the heap's header and free-list links stay) reset and taken out of the working set: their RAM and any page-file or
// compression copy given back, the heap's commit unchanged (a block given out again gets pages as malloc would: content
// undefined). Every buffer it uses is made before the lock (no heap call while holding it).
void Mem1HeapPass() {
	struct Span {
		uint64_t begin, end;
	};
	constexpr size_t   MaxSpans = 65536, MaxSamples = 131072, Batch = 4096;
	constexpr uint64_t Page = 4096, Min = uint64_t {64} << 10u, Step = uint64_t {64} << 10u;
	static auto* spans = static_cast<Span*>(VirtualAlloc(nullptr, MaxSpans * sizeof(Span), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	static auto* ws    = static_cast<PSAPI_WORKING_SET_EX_INFORMATION*>(
        VirtualAlloc(nullptr, MaxSamples * sizeof(PSAPI_WORKING_SET_EX_INFORMATION), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	if (spans == nullptr || ws == nullptr) return;
	HANDLE   heap      = GetProcessHeap();
	uint64_t committed = 0, busy = 0, free_bytes = 0, given = 0, resident = 0, refused = 0;
	size_t   count = 0, samples = 0;
	HANDLE   self     = GetCurrentThread();
	const int priority = GetThreadPriority(self);
	(void)SetThreadPriority(self, THREAD_PRIORITY_HIGHEST);
	const auto t0 = std::chrono::steady_clock::now();
	if (!HeapLock(heap)) {
		(void)SetThreadPriority(self, priority);
		return;
	}
	PROCESS_HEAP_ENTRY entry {};
	while (HeapWalk(heap, &entry)) {
		if ((entry.wFlags & PROCESS_HEAP_REGION) != 0) {
			committed += entry.Region.dwCommittedSize;
			continue;
		}
		if ((entry.wFlags & PROCESS_HEAP_UNCOMMITTED_RANGE) != 0) continue;
		if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) != 0) {
			busy += entry.cbData;
			continue;
		}
		free_bytes += entry.cbData;
		const uint64_t data  = reinterpret_cast<uint64_t>(entry.lpData);
		const uint64_t begin = (data + 64 + Page - 1) & ~(Page - 1);
		const uint64_t end   = (data + entry.cbData) & ~(Page - 1);
		if (end > begin && end - begin >= Min && count < MaxSpans) spans[count++] = {begin, end};
	}
	// What of them is in RAM now (a page each 64 KiB): the RAM given back now; the rest was in the compression store or
	// the page file (given back too: no copy kept).
	for (size_t i = 0; i < count; i++)
		for (uint64_t at = spans[i].begin; at < spans[i].end && samples < MaxSamples; at += Step) {
			ws[samples]                = {};
			ws[samples].VirtualAddress = reinterpret_cast<PVOID>(at);
			samples++;
		}
	for (size_t begin = 0; begin < samples; begin += Batch) {
		const size_t n = std::min(Batch, samples - begin);
		if (K32QueryWorkingSetEx(GetCurrentProcess(), ws + begin, static_cast<DWORD>(n * sizeof(ws[0]))) == 0) break;
		for (size_t k = 0; k < n; k++) resident += (ws[begin + k].VirtualAttributes.Flags & 1u) != 0 ? Step : 0;
	}
	// Reset: committed private read-write pages only (as memory.cpp's NativeOffloadDrop: MEM_RESET, then VirtualUnlock -
	// "not locked" - out of the working set).
	for (size_t i = 0; i < count; i++) {
		for (uint64_t at = spans[i].begin; at < spans[i].end;) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
			const uint64_t stop = std::min(spans[i].end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
			if (stop <= at) break;
			if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && (info.Protect & 0xffu) == PAGE_READWRITE &&
			    (info.Protect & PAGE_GUARD) == 0) {
				if (VirtualAlloc(reinterpret_cast<void*>(at), stop - at, MEM_RESET, PAGE_READWRITE) != nullptr) {
					given += stop - at;
					(void)VirtualUnlock(reinterpret_cast<void*>(at), stop - at);
				} else {
					refused += stop - at;
				}
			}
			at = stop;
		}
	}
	(void)HeapUnlock(heap);
	const auto us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
	(void)SetThreadPriority(self, priority);
	auto& h = g_mem1_heap;
	h.passes.fetch_add(1, std::memory_order_relaxed);
	h.committed.store(committed, std::memory_order_relaxed);
	h.busy.store(busy, std::memory_order_relaxed);
	h.free_bytes.store(free_bytes, std::memory_order_relaxed);
	h.given.store(given, std::memory_order_relaxed);
	h.given_resident.store(std::min(resident, given), std::memory_order_relaxed);
	h.given_total.fetch_add(given, std::memory_order_relaxed);
	h.refused.store(refused, std::memory_order_relaxed);
	h.spans.store(count, std::memory_order_relaxed);
	h.lock_us.store(us, std::memory_order_relaxed);
	if (us > h.lock_us_max.load(std::memory_order_relaxed)) h.lock_us_max.store(us, std::memory_order_relaxed);
	if (us > 8000) h.over_8ms.fetch_add(1, std::memory_order_relaxed);
	constexpr double MiBd = 1024.0 * 1024.0;
	std::printf("[mem1-heap] pasada %llu: monton C++ comprometido %.0f MiB, en uso %.0f, libre %.0f; soltado %.0f MiB en %zu bloques "
	            "(de ello en RAM %.0f; el resto estaba en la compresion o en el archivo de paginacion), rechazado %.0f; candado %.2f ms\n",
	            static_cast<unsigned long long>(h.passes.load()), static_cast<double>(committed) / MiBd,
	            static_cast<double>(busy) / MiBd, static_cast<double>(free_bytes) / MiBd, static_cast<double>(given) / MiBd, count,
	            static_cast<double>(std::min(resident, given)) / MiBd, static_cast<double>(refused) / MiBd,
	            static_cast<double>(us) / 1000.0);
	std::fflush(stdout);
}

// The VirtualMemo census (every 30 s): each 64 KiB granule of the zone's committed private pages, by where it is (in
// RAM: a working-set sample at its first page; taken: no access - RAM16/17/18's watchers -; out of RAM but accessible:
// compression store, page file or never written), whether an image of the GPU caches lies over it (the last GPU residency
// pass: sampled textures, render targets, GPU-written images), and when an emulator window last covered it (mem1.h's
// stamps) against when this census first saw it committed (its load, to 30 s). Read-only: no protection change, no read
// of guest bytes.
struct Mem1Vm {
	std::atomic<uint64_t> passes {0}, committed {0}, ram {0}, ram_image {0}, ram_lt10 {0}, ram_10_60 {0}, ram_gt60 {0};
	std::atomic<uint64_t> ram_never {0}, ram_young {0}, taken {0}, out {0}, out_image {0}, residency_pass {0}, pass_us {0};
	std::atomic<uint64_t> windows_10s {0};
};
Mem1Vm g_mem1_vm;

void Mem1VmCensus() {
	using namespace GpuResidency;
	const auto         t0 = std::chrono::steady_clock::now();
	constexpr size_t   MaxPairs = 16384, Batch = 1024;
	static std::vector<uint64_t> pairs(MaxPairs * 2);
	static std::vector<uint8_t>  covered(Mem1::Granules);
	static std::vector<uint32_t> first_seen(Mem1::Granules, 0);
	static std::vector<uint32_t> seen_pass(Mem1::Granules, 0);
	static uint32_t              pass = 0;
	pass++;
	const char*  labels[] = {"VirtualMemo"};
	const size_t n        = Loader::DemonsSoulsMemMap::RangesOf(labels, 1, pairs.data(), MaxPairs);
	if (n == SIZE_MAX) return;
	// The images over the zone (last residency pass).
	std::fill(covered.begin(), covered.end(), uint8_t {0});
	uint64_t residency = 0;
	{
		auto&           snap = Ram20Ranges();
		std::lock_guard lock(snap.mutex);
		residency = snap.pass;
		for (const auto& range: snap.ranges) {
			if (range.kind != Texture && range.kind != Target && range.kind != Storage) continue;
			const uint64_t lo = std::max(range.address, Mem1::VmBase), hi = std::min(range.address + range.size, Mem1::VmEnd);
			if (lo >= hi) continue;
			for (uint64_t g = (lo - Mem1::VmBase) >> Mem1::GranuleBits; g <= (hi - 1 - Mem1::VmBase) >> Mem1::GranuleBits; g++)
				covered[g] = 1;
		}
	}
	// The committed granules (bytes of each, a region may start or end inside one).
	struct Item {
		uint64_t at, bytes;
	};
	std::vector<Item> items;
	uint64_t          committed = 0, taken = 0;
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	for (size_t i = 0; i < n; i++)
		if (pairs[2 * i + 1] > pairs[2 * i]) spans.emplace_back(std::max(pairs[2 * i], Mem1::VmBase), std::min(pairs[2 * i + 1], Mem1::VmEnd));
	std::sort(spans.begin(), spans.end());
	const uint32_t now = Mem1::NowStamp();
	for (const auto& [lo, hi]: spans) {
		for (uint64_t at = lo; at < hi;) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
			const uint64_t stop = std::min(hi, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
			if (stop <= at) break;
			if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE) {
				committed += stop - at;
				const bool noaccess = (info.Protect & 0xffu) == PAGE_NOACCESS;
				for (uint64_t g_at = at; g_at < stop;) {
					const uint64_t g_end = std::min(stop, (g_at & ~(Mem1::Granule - 1)) + Mem1::Granule);
					const uint64_t g     = (g_at - Mem1::VmBase) >> Mem1::GranuleBits;
					if (first_seen[g] == 0 || seen_pass[g] + 1 < pass) first_seen[g] = now;
					seen_pass[g] = pass;
					if (noaccess) taken += g_end - g_at;
					else items.push_back({g_at, g_end - g_at});
					g_at = g_end;
				}
			}
			at = stop;
		}
		if (items.size() > (size_t {1} << 20u)) break;
	}
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> ws(items.size());
	for (size_t i = 0; i < items.size(); i++) ws[i].VirtualAddress = reinterpret_cast<PVOID>(items[i].at);
	for (size_t begin = 0; begin < ws.size(); begin += Batch) {
		const size_t count = std::min(Batch, ws.size() - begin);
		if (begin != 0) SwitchToThread();
		if (K32QueryWorkingSetEx(GetCurrentProcess(), ws.data() + begin, static_cast<DWORD>(count * sizeof(ws[0]))) == 0) return;
	}
	uint64_t ram = 0, ram_image = 0, lt10 = 0, mid = 0, gt60 = 0, never = 0, young = 0, out = 0, out_image = 0;
	auto*    stamps = Mem1::WindowStamps();
	for (size_t i = 0; i < items.size(); i++) {
		const uint64_t g     = (items[i].at - Mem1::VmBase) >> Mem1::GranuleBits;
		const uint64_t bytes = items[i].bytes;
		if ((ws[i].VirtualAttributes.Flags & 1u) == 0) {
			out += bytes;
			if (covered[g] != 0) out_image += bytes;
			continue;
		}
		ram += bytes;
		if (covered[g] != 0) {
			ram_image += bytes;
			continue;
		}
		if (now - first_seen[g] < 60) young += bytes;
		const uint32_t stamp = stamps[g].load(std::memory_order_relaxed);
		if (stamp == 0 || stamp < first_seen[g]) never += bytes;
		else if (now - stamp < 10) lt10 += bytes;
		else if (now - stamp <= 60) mid += bytes;
		else gt60 += bytes;
	}
	auto& v = g_mem1_vm;
	v.passes.fetch_add(1, std::memory_order_relaxed);
	v.committed.store(committed, std::memory_order_relaxed);
	v.ram.store(ram, std::memory_order_relaxed);
	v.ram_image.store(ram_image, std::memory_order_relaxed);
	v.ram_lt10.store(lt10, std::memory_order_relaxed);
	v.ram_10_60.store(mid, std::memory_order_relaxed);
	v.ram_gt60.store(gt60, std::memory_order_relaxed);
	v.ram_never.store(never, std::memory_order_relaxed);
	v.ram_young.store(young, std::memory_order_relaxed);
	v.taken.store(taken, std::memory_order_relaxed);
	v.out.store(out, std::memory_order_relaxed);
	v.out_image.store(out_image, std::memory_order_relaxed);
	v.residency_pass.store(residency, std::memory_order_relaxed);
	const auto us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
	v.pass_us.store(us, std::memory_order_relaxed);
	constexpr double GiBd = 1024.0 * 1024.0 * 1024.0;
	std::printf("[mem1-vm] 30 s, GB: VirtualMemo comprometido %.2f; en RAM %.2f = bajo una imagen de la GPU %.2f + sin imagen %.2f "
	            "(ventana del emulador hace <10 s %.2f, 10-60 s %.2f, >60 s %.2f, ninguna desde que se cargo %.2f; cargada hace "
	            "<60 s %.2f); sacada (sin acceso: RAM16-18) %.2f; fuera de RAM accesible %.2f (bajo imagen %.2f); etapa 2 "
	            "(VirtualMemo en la VRAM) sacaria de RAM %.2f mas lo comprimido de %.2f; residencia #%llu, %.1f ms\n",
	            static_cast<double>(committed) / GiBd, static_cast<double>(ram) / GiBd, static_cast<double>(ram_image) / GiBd,
	            static_cast<double>(ram - ram_image) / GiBd, static_cast<double>(lt10) / GiBd, static_cast<double>(mid) / GiBd,
	            static_cast<double>(gt60) / GiBd, static_cast<double>(never) / GiBd, static_cast<double>(young) / GiBd,
	            static_cast<double>(taken) / GiBd, static_cast<double>(out) / GiBd, static_cast<double>(out_image) / GiBd,
	            static_cast<double>(ram) / GiBd, static_cast<double>(out) / GiBd, static_cast<unsigned long long>(residency),
	            static_cast<double>(us) / 1000.0);
	std::fflush(stdout);
}

// The mem1 thread (lowest priority; the heap pass raises itself while it holds the heap's lock).
void Mem1Thread() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.Mem1");
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
	const char*  heap_env = Env("KYTY_MEM1_HEAP");
	const bool   heap_on  = heap_env != nullptr && heap_env[0] == '1'; // (11-17) off unless asked: a pass froze the game 1.8-2.5 s in the PC
	const double max_ms   = [] {
        const char* v = Env("KYTY_MEM1_HEAP_MAX_MS");
        return v == nullptr ? 20.0 : std::clamp(std::atof(v), 1.0, 1000.0);
	}();
	const auto start   = std::chrono::steady_clock::now();
	int64_t    next_heap = 60, next_vm = 30;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		const int64_t s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
		if (heap_on && !g_mem1_heap.stopped.load(std::memory_order_relaxed) && s >= next_heap) {
			Mem1HeapPass();
			next_heap = next_heap < 120 ? 120 : next_heap + 180;
			const double ms = static_cast<double>(g_mem1_heap.lock_us.load(std::memory_order_relaxed)) / 1000.0;
			if (ms > max_ms && next_heap > 120) { // (the passes at 60 s and 120 s - loading - always run)
				g_mem1_heap.stopped.store(true, std::memory_order_relaxed);
				std::printf("[mem1-heap] el candado duro %.1f ms (mas de %.0f, KYTY_MEM1_HEAP_MAX_MS): sin mas pasadas periodicas\n", ms, max_ms);
				std::fflush(stdout);
			}
		}
		if (s >= next_vm) {
			next_vm = s + 30;
			Mem1VmCensus();
		}
	}
}

// The "[mem1]" line (every 10 s, from the "[ram16]" thread): what each measure gives back now, the RAM the user counts,
// reads back and their time, losses, and the hitches it may cause (the heap's lock).
void Mem1Line(char* out, size_t size, const RamLineProcesses& procs, const TextureOffload::Ram16View* view, uint64_t freed,
              const MEMORYSTATUSEX& status) {
	static TextureOffload::Ram16View last;
	static bool                      have_last = false;
	constexpr double                 GiBd = 1024.0 * 1024.0 * 1024.0, MiBd = 1024.0 * 1024.0;
	const auto& h      = g_mem1_heap;
	const uint64_t heap = h.given.load(std::memory_order_relaxed);
	const uint64_t ring = g_mem1_ring.load(std::memory_order_relaxed);
	const uint64_t tex  = view != nullptr ? view->saved + view->cold_now : 0;
	uint64_t       reads = 0, read_bytes = 0, read_us = 0, lost = 0, read_max = 0;
	if (view != nullptr) {
		const auto d = [&](uint64_t TextureOffload::Ram16View::*field) { return have_last ? view->*field - last.*field : view->*field; };
		reads      = d(&TextureOffload::Ram16View::restores) + d(&TextureOffload::Ram16View::file_reads);
		read_bytes = d(&TextureOffload::Ram16View::restore_bytes) + d(&TextureOffload::Ram16View::file_read_bytes);
		read_us    = d(&TextureOffload::Ram16View::restore_us) + d(&TextureOffload::Ram16View::file_read_us);
		read_max   = view->file_read_us_max;
		lost       = view->missed + view->file_failed;
		last       = *view;
		have_last  = true;
	}
	const double emu  = procs.ok ? static_cast<double>(procs.own_private) / GiBd : 0.0;
	const double comp = procs.ok ? static_cast<double>(procs.compression_ws) / GiBd : 0.0;
	std::snprintf(out, size,
	              "[mem1] 10 s: ahorrado ahora %.2f GB = monton C++ %.2f (ultima pasada: libre %.2f de %.2f comprometido, soltado "
	              "%.2f, de ello estaba en RAM %.2f; %llu pasadas) + anillo de subida %.2f + texturas y memoria fria (RAM16-18) "
	              "%.2f + bloques libres del juego %.2f; RAM real: emulador %.2f + compresion de Windows %.2f = %.2f GB, "
	              "disponible %.2f GB; lecturas de vuelta +%llu (%.0f MiB, %.1f ms; la mayor del fichero %.1f ms); PERDIDAS "
	              "%llu; tirones atribuibles: candado del monton %.1f ms el ultimo, %.1f ms el mayor, %llu de mas de 8 ms%s",
	              static_cast<double>(heap + ring + tex + freed) / GiBd, static_cast<double>(heap) / GiBd,
	              static_cast<double>(h.free_bytes.load()) / GiBd, static_cast<double>(h.committed.load()) / GiBd,
	              static_cast<double>(heap) / GiBd, static_cast<double>(h.given_resident.load()) / GiBd,
	              static_cast<unsigned long long>(h.passes.load()), static_cast<double>(ring) / GiBd, static_cast<double>(tex) / GiBd,
	              static_cast<double>(freed) / GiBd, emu, comp, emu + comp, static_cast<double>(status.ullAvailPhys) / GiBd,
	              static_cast<unsigned long long>(reads), static_cast<double>(read_bytes) / MiBd, static_cast<double>(read_us) / 1000.0,
	              static_cast<double>(read_max) / 1000.0, static_cast<unsigned long long>(lost),
	              static_cast<double>(h.lock_us.load()) / 1000.0, static_cast<double>(h.lock_us_max.load()) / 1000.0,
	              static_cast<unsigned long long>(h.over_8ms.load()),
	              h.stopped.load(std::memory_order_relaxed) ? " (pasadas periodicas paradas)" : "");
}

void Ram16Thread() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.Ram16");
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	const int census_s = [] {
		const char* v = Env("KYTY_RAM16_CENSUS_S");
		return v == nullptr ? 30 : std::clamp(std::atoi(v), 0, 3600);
	}();
	constexpr double                       GiBd = 1024.0 * 1024.0 * 1024.0;
	constexpr double                       MiBd = 1024.0 * 1024.0;
	std::vector<uint8_t>                   buffer;
	TextureOffload::Ram16View              view, last;
	bool                                   have_view = false, have_last = false;
	auto                                   census_at = std::chrono::steady_clock::now();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		const auto     t0 = std::chrono::steady_clock::now();
		MEMORYSTATUSEX status {};
		status.dwLength = sizeof(status);
		(void)GlobalMemoryStatusEx(&status);
		PROCESS_MEMORY_COUNTERS_EX counters {};
		counters.cb = sizeof(counters);
		const bool have_commit =
		    K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)) != 0;
		RamLineProcesses procs;
		RamLineReadProcesses(&buffer, &procs);
		TextureOffload::Ram16View now;
		const bool fresh = TextureOffload::Ram16Read(&now);
		if (fresh) {
			view      = now;
			have_view = true;
		}
		const uint64_t freed = Libs::LibKernel::Memory::Ram16FreedReleasedNow();
		const uint64_t tex   = have_view ? view.saved : 0;
		const uint64_t cold  = have_view ? view.cold_now : 0; // (ram18)
		char           line17[640] {}; // (ram17)
		char           line18[900] {}; // (ram18)
		char           line20[1400] {}; // (ram20)
		char           line[1024];
		size_t         used = 0;
		RamLineAdd(line, sizeof(line), &used,
		           "[ram16] 5 s: ahorrado ahora %.2f GB = texturas fuera de RAM %.2f%s + bloques libres del juego devueltos %.2f",
		           static_cast<double>(tex + freed + cold) / GiBd, static_cast<double>(tex) / GiBd,
		           !have_view ? " (offload nativo apagado o sin leer)" : fresh ? "" : " (lectura anterior)",
		           static_cast<double>(freed) / GiBd);
		if (TextureOffload::Ram18On())
			RamLineAdd(line, sizeof(line), &used, " + memoria fria en disco (RAM18%s) %.2f", TextureOffload::Ram20Active() ? "+RAM20" : "",
			           static_cast<double>(cold) / GiBd);
		if (have_view) {
			const uint64_t restores = have_last ? view.restores - last.restores : view.restores;
			const uint64_t bytes    = have_last ? view.restore_bytes - last.restore_bytes : view.restore_bytes;
			const uint64_t us       = have_last ? view.restore_us - last.restore_us : view.restore_us;
			RamLineAdd(line, sizeof(line), &used,
			           "; texturas: re-descargas +%llu (%.0f MiB, %.1f ms), perdidas %llu, readmitidas %llu, dejado en total %.2f GB",
			           static_cast<unsigned long long>(restores), static_cast<double>(bytes) / MiBd, static_cast<double>(us) / 1000.0,
			           static_cast<unsigned long long>(view.missed), static_cast<unsigned long long>(view.retaken),
			           static_cast<double>(view.total) / GiBd);
			// (ram17) The disk store, every 5 s: what it holds, its writes and reads back (time on the thread doing them),
			// the images gone with no download and the takes held after a quick return.
			if (TextureOffload::Ram17On()) {
				const auto d = [&](uint64_t TextureOffload::Ram16View::*field) {
					return have_last ? view.*field - last.*field : view.*field;
				};
				std::snprintf(line17, sizeof(line17),
				              "[ram17] 5 s: en disco ahora %.2f GB (fichero %.2f GB); escritas +%llu (%.0f MiB, %.1f ms, hilo de "
				              "descarte); leidas de vuelta +%llu (%.0f MiB, %.1f ms, la mayor %.1f ms en total; de ellas antes de 30 s "
				              "+%llu); expulsiones sin descarga +%llu (%.0f MiB); retenidas en RAM por haber vuelto +%llu consultas; "
				              "fallos de fichero %llu",
				              static_cast<double>(view.file_used) / GiBd, static_cast<double>(view.file_size) / GiBd,
				              static_cast<unsigned long long>(d(&TextureOffload::Ram16View::file_writes)),
				              static_cast<double>(d(&TextureOffload::Ram16View::file_write_bytes)) / MiBd,
				              static_cast<double>(d(&TextureOffload::Ram16View::file_write_us)) / 1000.0,
				              static_cast<unsigned long long>(d(&TextureOffload::Ram16View::file_reads)),
				              static_cast<double>(d(&TextureOffload::Ram16View::file_read_bytes)) / MiBd,
				              static_cast<double>(d(&TextureOffload::Ram16View::file_read_us)) / 1000.0,
				              static_cast<double>(view.file_read_us_max) / 1000.0,
				              static_cast<unsigned long long>(d(&TextureOffload::Ram16View::file_quick)),
				              static_cast<unsigned long long>(d(&TextureOffload::Ram16View::orphaned)),
				              static_cast<double>(d(&TextureOffload::Ram16View::orphaned_bytes)) / MiBd,
				              static_cast<unsigned long long>(d(&TextureOffload::Ram16View::held)),
				              static_cast<unsigned long long>(view.file_failed));
			}
			last      = view;
			have_last = true;
		}
		if (procs.ok)
			RamLineAdd(line, sizeof(line), &used, "; emulador: privado en RAM %.2f GB", static_cast<double>(procs.own_private) / GiBd);
		if (have_commit) RamLineAdd(line, sizeof(line), &used, ", commit %.2f GB", static_cast<double>(counters.PrivateUsage) / GiBd);
		if (procs.ok)
			RamLineAdd(line, sizeof(line), &used, "; Windows: compresion %.2f GB", static_cast<double>(procs.compression_ws) / GiBd);
		RamLineAdd(line, sizeof(line), &used, ", disponible %.2f de %.2f GB, commit del sistema %.1f de %.1f GB; %.1f ms",
		           static_cast<double>(status.ullAvailPhys) / GiBd, static_cast<double>(status.ullTotalPhys) / GiBd,
		           static_cast<double>(status.ullTotalPageFile - status.ullAvailPageFile) / GiBd,
		           static_cast<double>(status.ullTotalPageFile) / GiBd,
		           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
		// (ram18) Every 5 s: the cold ranges out of RAM (on disk), the zone as the last pass saw it, what went and came
		// back, and the RAM the user counts: the emulator's private RAM plus Windows' compression store.
		if (TextureOffload::Ram18Own()) Ram18Line(line18, sizeof(line18), procs);
		// (ram20) Every KYTY_RAM20_LINE_S (default 10; 5-30).
		static const int line20_s = [] {
			const char* v = Env("KYTY_RAM20_LINE_S");
			return v == nullptr ? 10 : std::clamp(std::atoi(v) / 5 * 5, 5, 30);
		}();
		static int line20_tick = 0;
		if (TextureOffload::Ram20Active() && (line20_tick += 5) >= line20_s) {
			line20_tick = 0;
			Ram20Line(line20, sizeof(line20), procs, line20_s);
		}
		// (mem1, KYTY_MEM1=1) Every 10 s.
		char       line_m1[1200] {};
		static int mem1_tick = 0;
		if (Mem1::On() && (mem1_tick += 5) >= 10) {
			mem1_tick = 0;
			Mem1Line(line_m1, sizeof(line_m1), procs, have_view ? &view : nullptr, freed, status);
		}
		std::printf("%s\n", line);
		if (line17[0] != '\0') std::printf("%s\n", line17);
		if (line18[0] != '\0') std::printf("%s\n", line18);
		if (line20[0] != '\0') std::printf("%s\n", line20);
		if (line_m1[0] != '\0') std::printf("%s\n", line_m1);
		if (census_s > 0 && std::chrono::steady_clock::now() - census_at >= std::chrono::seconds(census_s)) {
			census_at = std::chrono::steady_clock::now();
			char     text[768];
			uint64_t game = 0, stride = 0;
			double   ms   = 0;
			size_t   samples = 0;
			if (Ram16Census(text, sizeof(text), &game, &ms, &samples, &stride)) {
				const uint64_t host = procs.ok && procs.own_private > game ? procs.own_private - game : 0;
				std::printf("[ram16-censo] GB en RAM (rango) por asignador del juego: %s; juego %.2f, emulador fuera del juego "
				            "%.2f (privado en RAM menos el juego: monton C++, Vulkan, JIT...; desglose en \"Private residency\"); "
				            "%zu muestras cada %llu KiB, %.1f ms\n",
				            text, static_cast<double>(game) / GiBd, static_cast<double>(host) / GiBd, samples,
				            static_cast<unsigned long long>(stride >> 10u), ms);
			} else {
				std::printf("[ram16-censo] sin mapa del juego todavia (KYTY_MEMMAP=0 o sin leer): sin censo\n");
			}
		}
		std::fflush(stdout);
	}
}

} // namespace
#endif

static void StartRam16Line() {
#if defined(_WIN32)
	static std::once_flag once;
	std::call_once(once, [] { std::thread(Ram16Thread).detach(); });
#endif
}

static void StartMem1() {
#if defined(_WIN32)
	static std::once_flag once;
	std::call_once(once, [] { std::thread(Mem1Thread).detach(); });
#endif
}

static void StartRam18() {
#if defined(_WIN32)
	static std::once_flag once;
	std::call_once(once, [] { std::thread(Ram18Thread).detach(); });
#endif
}

void Start() {
#if defined(_WIN32)
	static std::once_flag once;
	std::call_once(once, [] {
		if (RamLineEnabled()) std::thread(RamLineThread).detach(); // (ram2; its own switch, with the guard off too)
		if (const char* value = Env("KYTY_RAM_GUARD"); value != nullptr && value[0] == '0') {
			std::printf("RAM guard: off (KYTY_RAM_GUARD=0)\n");
			std::fflush(stdout);
			return;
		}
		std::thread(Watch).detach();
	});
#endif
}

} // namespace RamGuard

namespace GpuResidency {

#if defined(_WIN32)
namespace {

constexpr uint64_t SampleBits = 16; // one page sampled in each 64 KiB granule of guest address space
constexpr uint64_t SampleSpan = uint64_t {1} << SampleBits;
constexpr uint64_t PageSize   = 4096;
constexpr size_t   MaxSamples = size_t {1} << 21u; // (48 MiB of samples at most; ~13 GiB of ranges is ~220K)

struct Sample {
	uint64_t page   = 0; // guest address (the game's view)
	uint64_t alias  = 0; // the same page in the backing store's alias (0: no backing: unmapped)
	uint32_t weight = 0; // the range's bytes in the sample's 64 KiB granule
	Kind     kind   = Texture;
	uint8_t  state  = 0; // StateView | StateAlias | StateWritten | StateKnown
};
constexpr uint8_t StateView    = 1; // in the working set through the guest address
constexpr uint8_t StateAlias   = 2; // in the working set through the alias
constexpr uint8_t StateWritten = 4; // fresh memory: its 64 KiB granule was ever written (CPU, host or write-back)
constexpr uint8_t StateKnown   = 8; // fresh memory knows the granule

struct Totals {
	uint64_t bytes = 0, view_only = 0, alias_only = 0, both = 0, unmapped = 0, resident_unwritten = 0;
	[[nodiscard]] uint64_t Resident() const { return view_only + alias_only + both; }
	void Add(const Sample& sample, uint64_t weight) {
		bytes += weight;
		const bool view = (sample.state & StateView) != 0, alias = (sample.state & StateAlias) != 0;
		if (!view && !alias) {
			if (sample.alias == 0) unmapped += weight; // (no backing under it: nothing to be resident)
			return;
		}
		if (view && alias) both += weight;
		else if (view) view_only += weight;
		else alias_only += weight;
		if ((sample.state & StateKnown) != 0 && (sample.state & StateWritten) == 0) resident_unwritten += weight;
	}
};

// The order a granule several kinds share is counted under in the union.
int UnionRank(Kind kind) {
	switch (kind) {
		case Target: return 0;
		case Storage: return 1;
		case GpuWrittenBuffer: return 2;
		case Texture: return 3;
		case GpuBuffer:
		default: return 4;
	}
}

const char* KindName(Kind kind) {
	switch (kind) {
		case Target: return "render targets/depth/video out";
		case Storage: return "storage/GPU-written images";
		case GpuWrittenBuffer: return "GPU-written buffers";
		case GpuBuffer: return "other GPU buffers";
		case Texture:
		default: return "sampled textures";
	}
}

double ToMiB(uint64_t bytes) {
	return static_cast<double>(bytes) / 1048576.0;
}

// The samples of the ranges: one page in each 64 KiB granule a range covers (the page at the pass's offset in the
// granule, clamped into the range: over passes every 16th page), weighted by the range's bytes in the granule; each
// with its alias address (the backing pieces of the range: the guest address space's shared map lock, a lookup a
// range).
bool MakeSamples(const std::vector<Range>& ranges, uint64_t pass, std::vector<Sample>* samples) {
	std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
	const uint64_t                                   offset = (pass % (SampleSpan / PageSize)) * PageSize;
	for (const auto& range: ranges) {
		if (range.size == 0 || UINT64_MAX - range.address < range.size) continue;
		const uint64_t end    = range.address + range.size;
		const bool     backed = Libs::LibKernel::Memory::TryGetSparseBackingPieces(range.address, range.size, pieces);
		size_t         piece  = 0;
		uint64_t       piece_begin = 0; // (offset in the range of pieces[piece])
		for (uint64_t granule = range.address & ~(SampleSpan - 1); granule < end; granule += SampleSpan) {
			if (samples->size() >= MaxSamples) return false;
			const uint64_t lo    = std::max(granule, range.address);
			const uint64_t hi    = std::min(granule + SampleSpan, end);
			const uint64_t first = lo & ~(PageSize - 1), last = (hi - 1) & ~(PageSize - 1);
			Sample         sample;
			sample.page   = std::clamp(granule + offset, first, last);
			sample.weight = static_cast<uint32_t>(hi - lo);
			sample.kind   = range.kind;
			if (backed) {
				const uint64_t at = sample.page > range.address ? sample.page - range.address : 0;
				while (piece < pieces.size() && at >= piece_begin + pieces[piece].second) piece_begin += pieces[piece++].second;
				if (piece < pieces.size() && pieces[piece].first != nullptr)
					sample.alias = (reinterpret_cast<uint64_t>(pieces[piece].first) + (at - piece_begin)) & ~(PageSize - 1);
			}
			samples->push_back(sample);
			if (UINT64_MAX - granule < SampleSpan) break;
		}
	}
	return true;
}

// The working-set state of each sample's two pages (K32QueryWorkingSetEx, in batches); false when it failed.
bool QuerySamples(std::vector<Sample>* samples) {
	// (Samples a call, two entries each: a short call, then the CPU yielded, so the kernel's walk of the working set
	// never holds it long against the game's page faults.)
	constexpr size_t                               Batch = 1024;
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries(Batch * 2);
	HANDLE                                         process = GetCurrentProcess();
	const auto*                                    fresh   = FreshMemory::Get();
	for (size_t begin = 0; begin < samples->size(); begin += Batch) {
		const size_t count = std::min(Batch, samples->size() - begin);
		for (size_t i = 0; i < count; i++) {
			const auto& sample               = (*samples)[begin + i];
			entries[i * 2]                   = {};
			entries[i * 2 + 1]               = {};
			entries[i * 2].VirtualAddress     = reinterpret_cast<PVOID>(sample.page);
			entries[i * 2 + 1].VirtualAddress = reinterpret_cast<PVOID>(sample.alias != 0 ? sample.alias : sample.page);
		}
		if (begin != 0) SwitchToThread();
		if (K32QueryWorkingSetEx(process, entries.data(), static_cast<DWORD>(count * 2 * sizeof(entries[0]))) == 0)
			return false;
		for (size_t i = 0; i < count; i++) {
			auto& sample = (*samples)[begin + i];
			// (PSAPI_WORKING_SET_EX_BLOCK.Valid, bit 0: the page is in the process's working set at that address.)
			const bool view  = (entries[i * 2].VirtualAttributes.Flags & 1u) != 0;
			const bool alias = sample.alias != 0 && (entries[i * 2 + 1].VirtualAttributes.Flags & 1u) != 0;
			sample.state     = static_cast<uint8_t>((view ? StateView : 0) | (alias ? StateAlias : 0));
			if (fresh != nullptr && sample.alias != 0) {
				const uint64_t granule = FreshMemory::GranuleOf(*fresh, sample.alias);
				if (granule != UINT64_MAX) {
					sample.state = static_cast<uint8_t>(sample.state | StateKnown |
					                                    (FreshMemory::IsWritten(*fresh, granule) ? StateWritten : 0));
				}
			}
		}
	}
	return true;
}

// The lines (see the header gpu-residency.h, and the report in the commit): per kind as registered (a granule two
// kinds share is in both), then the union (each 64 KiB granule once, under the first kind by UnionRank).
void PrintGpuResidency(std::vector<Range>& ranges, uint64_t images, uint64_t buffers, double copy_ms, uint64_t pass) {
	const auto          t0 = std::chrono::steady_clock::now();
	std::vector<Sample> samples;
	samples.reserve(std::min<size_t>(MaxSamples, 1u << 18u));
	// (The backing lookups take the address space's shared lock, and the working-set queries the kernel's working-set
	// lock: at normal priority, so neither is held by a descheduled low-priority thread while a map, an unmap or a page
	// fault waits.)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
	const bool complete = MakeSamples(ranges, pass, &samples);
	ranges.clear();
	ranges.shrink_to_fit();
	const double lookup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	const bool   queried   = QuerySamples(&samples);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
	if (!queried) {
		std::printf("GPU residency (30 s): K32QueryWorkingSetEx failed (0x%08x)\n", static_cast<unsigned>(GetLastError()));
		std::fflush(stdout);
		return;
	}
	Totals by_kind[Kinds];
	for (const auto& sample: samples) by_kind[static_cast<size_t>(sample.kind)].Add(sample, sample.weight);
	// The union: by granule, the first kind by rank; the granule's largest weight; resident through a view where any of
	// its samples is; written where any of its samples' granules was.
	std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
		const uint64_t ga = a.page >> SampleBits, gb = b.page >> SampleBits;
		return ga != gb ? ga < gb : UnionRank(a.kind) < UnionRank(b.kind);
	});
	Totals union_kind[Kinds], union_all;
	for (size_t i = 0; i < samples.size();) {
		Sample   merged = samples[i];
		uint64_t weight = merged.weight;
		size_t   j      = i + 1;
		for (; j < samples.size() && (samples[j].page >> SampleBits) == (merged.page >> SampleBits); j++) {
			weight = std::max<uint64_t>(weight, samples[j].weight);
			if (merged.alias == 0) merged.alias = samples[j].alias;
			merged.state = static_cast<uint8_t>(merged.state | samples[j].state); // (any sample's view, any written)
		}
		union_kind[static_cast<size_t>(merged.kind)].Add(merged, weight);
		union_all.Add(merged, weight);
		i = j;
	}
	const auto   ws = LocalPlatform::ProcessMemory().working_set;
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::printf("GPU residency (30 s, MiB of guest memory resident in the working set, 1 page in 64 KiB sampled; "
	            "KYTY_GPU_RESIDENCY=0 off): %llu images and %llu buffers registered (copied on the GPU thread in %.2f ms), "
	            "%zu samples%s, %.1f ms on this thread (backing lookups %.1f ms); process working set %.0f\n",
	            static_cast<unsigned long long>(images), static_cast<unsigned long long>(buffers), copy_ms, samples.size(),
	            complete ? "" : " (capped: the rest not measured)", ms, lookup_ms, ToMiB(ws));
	const auto print = [](const char* what, const char* name, const Totals& t) {
		std::printf("GPU residency %s %s: range %.0f, resident %.0f = through the game's view only %.0f + the emulator's "
		            "alias only %.0f + both %.0f; not resident %.0f, unmapped %.0f; resident but never written %.0f\n",
		            what, name, ToMiB(t.bytes), ToMiB(t.Resident()), ToMiB(t.view_only), ToMiB(t.alias_only), ToMiB(t.both),
		            ToMiB(t.bytes - std::min(t.bytes, t.Resident() + t.unmapped)), ToMiB(t.unmapped),
		            ToMiB(t.resident_unwritten));
	};
	for (size_t kind = 0; kind < static_cast<size_t>(Kinds); kind++) print("[by kind]", KindName(static_cast<Kind>(kind)), by_kind[kind]);
	for (size_t kind = 0; kind < static_cast<size_t>(Kinds); kind++) print("[union]", KindName(static_cast<Kind>(kind)), union_kind[kind]);
	print("[union]", "all GPU-cached guest memory", union_all);
	std::fflush(stdout);
}

// ---- (ram19) KYTY_RAM19=1: where the game's and the emulator's memory is, against the GPU caches ----
// One pass every 30 s on this (lowest-priority) thread, after the GPU thread's copy of the caches' ranges. Nothing is
// written or protected: VirtualQuery (state, protection) and K32QueryWorkingSetEx (in RAM) only.
// Per allocator of the game (the census' labels): committed bytes (one sample per 64 KiB granule, weighted by the
// committed bytes of the region in it) split into
//   in RAM            - with the granule under a GPU cache range (by kind: a copy also in video memory) or under none;
//   out, no access    - PAGE_NOACCESS: taken by an offload (RAM16 textures, RAM18 cold ranges: reset, on disk, or
//                       watched GPU bytes): no RAM;
//   out, accessible   - committed, readable, not in the working set: in Windows' compression store, in the page file,
//                       or never written (demand-zero, costs nothing); also split by "under a GPU range" or not.
// Then the emulator's own private memory (everything committed outside the game's ranges), one sample per 256 KiB, the
// same three classes and the allocations with most out-of-RAM accessible memory (the compression store's likely
// owners) and most in RAM.
constexpr uint8_t R19None  = Kinds; // (no GPU cache range over the granule)
constexpr size_t  R19Kinds = static_cast<size_t>(Kinds) + 1;

struct R19Cell {
	uint64_t ram[R19Kinds] {}, out_access[R19Kinds] {}, noaccess = 0, committed = 0, range = 0;
};

const char* R19KindShort(size_t kind) {
	switch (kind) {
		case Texture: return "tex";
		case Target: return "RT";
		case Storage: return "imgGPU";
		case GpuWrittenBuffer: return "bufGPU";
		case GpuBuffer: return "buf";
		default: return "sinGPU";
	}
}

void PrintRam19(const std::vector<Range>& gpu_ranges, uint64_t pass) {
	static const char* const labels[] = {"VirtualMemo", "cpuheap", "gpuheap", "cpupage", "gpupage", "cpubkt",
	                                     "gpubkt",      "DefaultTagg", "RTSharedAll", "RTPersistAl"};
	constexpr size_t         Labels   = sizeof(labels) / sizeof(labels[0]);
	constexpr size_t         MaxPairs = 8192;
	constexpr uint64_t       Granule  = SampleSpan; // 64 KiB
	constexpr double         GiBd     = 1024.0 * 1024.0 * 1024.0;
	const auto               t0       = std::chrono::steady_clock::now();
	// The GPU ranges by granule: the kind first by UnionRank.
	std::unordered_map<uint64_t, uint8_t> gpu;
	gpu.reserve(gpu_ranges.size() * 4 + 1024);
	for (const auto& range: gpu_ranges) {
		if (range.size == 0 || UINT64_MAX - range.address < range.size) continue;
		for (uint64_t g = range.address / Granule; g <= (range.address + range.size - 1) / Granule; g++) {
			auto [it, fresh] = gpu.try_emplace(g, static_cast<uint8_t>(range.kind));
			if (!fresh && UnionRank(range.kind) < UnionRank(static_cast<Kind>(it->second))) it->second = static_cast<uint8_t>(range.kind);
		}
	}
	// The game's ranges, sorted (label of each), for the emulator's part.
	std::vector<uint64_t>                           pairs(MaxPairs * 2);
	std::vector<std::tuple<uint64_t, uint64_t, uint16_t>> game;
	for (size_t l = 0; l < Labels; l++) {
		const size_t n = Loader::DemonsSoulsMemMap::RangesOf(&labels[l], 1, pairs.data(), MaxPairs);
		if (n == SIZE_MAX) {
			std::printf("[ram19] sin mapa del juego todavia (KYTY_MEMMAP=0 o sin leer): sin recuento\n");
			std::fflush(stdout);
			return;
		}
		for (size_t i = 0; i < n; i++)
			if (pairs[2 * i + 1] > pairs[2 * i]) game.emplace_back(pairs[2 * i], pairs[2 * i + 1], static_cast<uint16_t>(l));
	}
	std::sort(game.begin(), game.end());
	struct S {
		uint64_t page = 0, weight = 0, base = 0;
		uint16_t label = 0; // (Labels: the emulator's)
		uint8_t  kind = R19None;
		bool     noaccess = false;
	};
	std::vector<S> samples;
	samples.reserve(size_t {1} << 18u);
	constexpr size_t MaxS   = size_t {1} << 20u;
	const uint64_t   offset = (pass % (Granule / PageSize)) * PageSize;
	R19Cell          cell[Labels + 1];
	// (1) The game's: VirtualQuery region by region inside each range.
	for (const auto& [begin, end, label]: game) {
		cell[label].range += end - begin;
		MEMORY_BASIC_INFORMATION info {};
		for (uint64_t at = begin; at < end && samples.size() < MaxS;) {
			if (VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) != sizeof(info)) break;
			const uint64_t r_end = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize);
			if (r_end <= at) break;
			if (info.State == MEM_COMMIT) {
				const bool noaccess = (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0 || info.Protect == 0;
				for (uint64_t g = at & ~(Granule - 1); g < r_end && samples.size() < MaxS; g += Granule) {
					const uint64_t lo = std::max(g, at), hi = std::min(g + Granule, r_end);
					const auto     found = gpu.find(g / Granule);
					S              s;
					s.page     = std::clamp(g + offset, lo & ~(PageSize - 1), (hi - 1) & ~(PageSize - 1));
					s.weight   = hi - lo;
					s.label    = label;
					s.kind     = found != gpu.end() ? found->second : R19None;
					s.noaccess = noaccess;
					samples.push_back(s);
				}
			}
			at = r_end;
		}
	}
	const size_t game_samples = samples.size();
	// (2) The emulator's: every committed private region outside the game's ranges, a sample per 256 KiB.
	{
		constexpr uint64_t Stride = uint64_t {256} << 10u;
		MEMORY_BASIC_INFORMATION info {};
		size_t                   cursor = 0;
		for (uint64_t at = 0x10000; samples.size() < MaxS && VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) == sizeof(info);) {
			const uint64_t r_begin = reinterpret_cast<uint64_t>(info.BaseAddress), r_end = r_begin + info.RegionSize;
			if (r_end <= at) break;
			if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE) {
				const bool noaccess = (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0 || info.Protect == 0;
				for (uint64_t p = r_begin; p < r_end && samples.size() < MaxS; p += Stride) {
					while (cursor < game.size() && std::get<1>(game[cursor]) <= p) cursor++;
					if (cursor < game.size() && std::get<0>(game[cursor]) <= p) continue; // (the game's: counted above)
					S s;
					s.page     = p;
					s.weight   = std::min(Stride, r_end - p);
					s.base     = reinterpret_cast<uint64_t>(info.AllocationBase);
					s.label    = static_cast<uint16_t>(Labels);
					s.noaccess = noaccess;
					samples.push_back(s);
				}
			}
			if (UINT64_MAX - info.RegionSize < r_begin) break;
			at = r_end;
		}
	}
	const double walk_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	// (3) The working set, in batches.
	constexpr size_t                               Batch = 1024;
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries(Batch);
	std::vector<uint8_t>                           in_ram(samples.size());
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
	bool ok = true;
	for (size_t b = 0; b < samples.size() && ok; b += Batch) {
		const size_t n = std::min(Batch, samples.size() - b);
		for (size_t i = 0; i < n; i++) {
			entries[i]                = {};
			entries[i].VirtualAddress = reinterpret_cast<PVOID>(samples[b + i].page);
		}
		if (b != 0) SwitchToThread();
		ok = K32QueryWorkingSetEx(GetCurrentProcess(), entries.data(), static_cast<DWORD>(n * sizeof(entries[0]))) != 0;
		for (size_t i = 0; i < n && ok; i++) in_ram[b + i] = static_cast<uint8_t>(entries[i].VirtualAttributes.Flags & 1u);
	}
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
	if (!ok) {
		std::printf("[ram19] K32QueryWorkingSetEx fallo (0x%08x)\n", static_cast<unsigned>(GetLastError()));
		std::fflush(stdout);
		return;
	}
	std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> emu; // base -> in RAM, out accessible
	for (size_t i = 0; i < samples.size(); i++) {
		const auto& s = samples[i];
		auto&       c = cell[s.label];
		c.committed += s.weight;
		if (in_ram[i] != 0) c.ram[s.kind] += s.weight;
		else if (s.noaccess) c.noaccess += s.weight;
		else c.out_access[s.kind] += s.weight;
		if (s.label == Labels) {
			auto& e = emu[s.base];
			if (in_ram[i] != 0) e.first += s.weight;
			else if (!s.noaccess) e.second += s.weight;
		}
	}
	const auto sum = [](const uint64_t (&a)[R19Kinds], bool gpu_only) {
		uint64_t t = 0;
		for (size_t k = 0; k < R19Kinds; k++)
			if (!gpu_only || k != R19None) t += a[k];
		return t;
	};
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	// The game, one line per allocator with anything committed.
	R19Cell all;
	for (size_t l = 0; l < Labels; l++) {
		const auto& c = cell[l];
		all.committed += c.committed;
		all.noaccess += c.noaccess;
		all.range += c.range;
		for (size_t k = 0; k < R19Kinds; k++) {
			all.ram[k] += c.ram[k];
			all.out_access[k] += c.out_access[k];
		}
		if (c.committed == 0) continue;
		char   text[700];
		size_t used = 0;
		text[0]     = '\0';
		RamGuard::RamLineAdd(text, sizeof(text), &used, "[ram19] %s: rango %.2f, commit %.2f; en RAM %.2f (copia en GPU %.2f:",
		                           labels[l], static_cast<double>(c.range) / GiBd, static_cast<double>(c.committed) / GiBd,
		                           static_cast<double>(sum(c.ram, false)) / GiBd, static_cast<double>(sum(c.ram, true)) / GiBd);
		for (size_t k = 0; k < R19None; k++)
			RamGuard::RamLineAdd(text, sizeof(text), &used, " %s %.2f", R19KindShort(k), static_cast<double>(c.ram[k]) / GiBd);
		RamGuard::RamLineAdd(text, sizeof(text), &used,
		                           "; sin copia en GPU %.2f); fuera de RAM: sin acceso (sacado) %.2f, accesible %.2f (con copia en GPU "
		                           "%.2f)",
		                           static_cast<double>(c.ram[R19None]) / GiBd, static_cast<double>(c.noaccess) / GiBd,
		                           static_cast<double>(sum(c.out_access, false)) / GiBd, static_cast<double>(sum(c.out_access, true)) / GiBd);
		std::printf("%s\n", text);
	}
	const auto& e = cell[Labels];
	std::vector<std::pair<uint64_t, uint64_t>> by_out, by_ram;
	for (const auto& [base, v]: emu) {
		if (v.second != 0) by_out.emplace_back(v.second, base);
		if (v.first != 0) by_ram.emplace_back(v.first, base);
	}
	std::sort(by_out.begin(), by_out.end(), std::greater<>());
	std::sort(by_ram.begin(), by_ram.end(), std::greater<>());
	char   text[1200];
	size_t used = 0;
	text[0]     = '\0';
	RamGuard::RamLineAdd(text, sizeof(text), &used,
	                           "[ram19] emulador fuera del juego: commit %.2f, en RAM %.2f, fuera de RAM sin acceso %.2f, accesible %.2f; "
	                           "mas fuera de RAM accesible (MiB):",
	                           static_cast<double>(e.committed) / GiBd, static_cast<double>(sum(e.ram, false)) / GiBd,
	                           static_cast<double>(e.noaccess) / GiBd, static_cast<double>(sum(e.out_access, false)) / GiBd);
	for (size_t i = 0; i < by_out.size() && i < 8; i++)
		RamGuard::RamLineAdd(text, sizeof(text), &used, " 0x%llx=%llu", static_cast<unsigned long long>(by_out[i].second),
		                           static_cast<unsigned long long>(by_out[i].first >> 20u));
	RamGuard::RamLineAdd(text, sizeof(text), &used, "; mas en RAM (MiB):");
	for (size_t i = 0; i < by_ram.size() && i < 8; i++)
		RamGuard::RamLineAdd(text, sizeof(text), &used, " 0x%llx=%llu", static_cast<unsigned long long>(by_ram[i].second),
		                           static_cast<unsigned long long>(by_ram[i].first >> 20u));
	std::printf("%s\n", text);
	// The summary: the RAM that is a duplicate of video memory, and what the compression store can hold.
	const uint64_t dup        = sum(all.ram, true);
	const uint64_t dup_gpuw   = all.ram[Target] + all.ram[Storage] + all.ram[GpuWrittenBuffer];
	const uint64_t out_game   = sum(all.out_access, false);
	const uint64_t out_emu    = sum(e.out_access, false);
	std::printf("[ram19] resumen (GB): juego commit %.2f, en RAM %.2f = con copia en GPU %.2f (de ello escrita por la GPU: RT, "
	            "imagenes y buffers %.2f; texturas %.2f; buffers %.2f) + sin copia en GPU %.2f; fuera de RAM sin acceso %.2f; "
	            "fuera de RAM accesible (compresion de Windows, fichero de paginacion o nunca escrito): juego %.2f (con copia en GPU "
	            "%.2f) + emulador %.2f = %.2f; emulador en RAM %.2f; %zu muestras (juego %zu), %.0f ms (recorrido %.0f ms), %llu "
	            "rangos de la GPU\n",
	            static_cast<double>(all.committed) / GiBd, static_cast<double>(sum(all.ram, false)) / GiBd,
	            static_cast<double>(dup) / GiBd, static_cast<double>(dup_gpuw) / GiBd, static_cast<double>(all.ram[Texture]) / GiBd,
	            static_cast<double>(all.ram[GpuBuffer]) / GiBd, static_cast<double>(all.ram[R19None]) / GiBd,
	            static_cast<double>(all.noaccess) / GiBd, static_cast<double>(out_game) / GiBd,
	            static_cast<double>(sum(all.out_access, true)) / GiBd, static_cast<double>(out_emu) / GiBd,
	            static_cast<double>(out_game + out_emu) / GiBd, static_cast<double>(sum(e.ram, false)) / GiBd, samples.size(),
	            game_samples, ms, walk_ms, static_cast<unsigned long long>(gpu_ranges.size()));
	std::fflush(stdout);
}

void Watch() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.GpuResidency");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
	std::printf("GPU residency: on (KYTY_GPU_RESIDENCY=0 turns it off): every 30 s, the working set over the guest ranges "
	            "of the GPU caches' images and buffers, read-only\n");
	std::fflush(stdout);
	auto& exchange = Get();
	for (uint64_t pass = 1;; pass++) {
		std::this_thread::sleep_for(std::chrono::seconds(30));
		std::vector<Range> ranges;
		ranges.reserve(size_t {1} << 16u); // (here, not on the GPU thread)
		std::vector<Range> stale;            // (a late answer to a pass not measured: freed outside the lock)
		uint64_t           images = 0, buffers = 0;
		double             copy_ms = 0.0;
		{
			// (Held only for these swaps and the wait's checks: the GPU thread only tries it, never waits.)
			std::unique_lock lock(exchange.mutex);
			stale.swap(exchange.ranges);
			exchange.ranges.swap(ranges);
			exchange.request = pass;
			exchange.requested.store(true, std::memory_order_release);
			// (No flip in 10 s - a loading screen held, the game paused in a debugger, or the answer dropped: this pass is
			// not measured.)
			if (!exchange.delivered.wait_for(lock, std::chrono::seconds(10), [&] { return exchange.answer == pass; })) {
				exchange.requested.store(false, std::memory_order_relaxed);
				lock.unlock();
				std::printf("GPU residency (30 s): no answer from the GPU thread in 10 s, not measured\n");
				std::fflush(stdout);
				continue;
			}
			ranges.swap(exchange.ranges);
			images  = exchange.images;
			buffers = exchange.buffers;
			copy_ms = exchange.copy_ms;
		}
		if (Ram20On() || Mem1::On()) { // (ram20: the RAM20 pass's map, a copy; mem1: its VirtualMemo census reads it too)
			auto&           snap = Ram20Ranges();
			std::lock_guard lock(snap.mutex);
			snap.ranges = ranges;
			snap.pass   = pass;
		}
		if (Ram19On()) PrintRam19(ranges, pass); // (ram19: before the line below frees the ranges)
		PrintGpuResidency(ranges, images, buffers, copy_ms, pass);
	}
}

} // namespace
#endif

void Start() {
#if defined(_WIN32)
	if (!Enabled()) {
		std::printf("GPU residency: off (KYTY_GPU_RESIDENCY=0)\n");
		std::fflush(stdout);
		return;
	}
	static std::once_flag once;
	std::call_once(once, [] { std::thread(Watch).detach(); });
#endif
}

} // namespace GpuResidency
