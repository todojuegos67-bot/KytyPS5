#include "loader/demonsSoulsPageFree.h"
#include "loader/demonsSoulsAllocNativo.h" // (m6) KYTY_ALLOC_NATIVO, instalado desde aqui

#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "local/local-platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string_view>
#include <thread>
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
#endif

namespace Loader::DemonsSoulsPageFree {
namespace {

enum class Mode { Off, Walk, GiveBack };

// La memoria que el emulador puede usar: la RAM fisica, o menos si un job la limita (pruebas con 16 GB).
uint64_t EmulatorMemory() {
#if defined(_WIN32)
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	uint64_t ram    = GlobalMemoryStatusEx(&status) != 0 ? static_cast<uint64_t>(status.ullTotalPhys) : 0;
	JOBOBJECT_EXTENDED_LIMIT_INFORMATION job {};
	if (QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &job, sizeof(job), nullptr) != 0) {
		const DWORD flags = job.BasicLimitInformation.LimitFlags;
		if ((flags & JOB_OBJECT_LIMIT_JOB_MEMORY) != 0 && job.JobMemoryLimit != 0)
			ram = ram == 0 ? job.JobMemoryLimit : std::min<uint64_t>(ram, job.JobMemoryLimit);
		if ((flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) != 0 && job.ProcessMemoryLimit != 0)
			ram = ram == 0 ? job.ProcessMemoryLimit : std::min<uint64_t>(ram, job.ProcessMemoryLimit);
	}
	return ram;
#else
	return 0;
#endif
}

Mode GetMode() {
	static const Mode mode = [] {
		if (const char* value = std::getenv("KYTY_ENGINE_FREE"); value != nullptr && *value != '\0') {
			const std::string_view text = value;
			if (text == "walk") return Mode::Walk;
#if defined(_WIN32)
			if (text == "giveback" || text == "1") return Mode::GiveBack;
#endif
			return Mode::Off;
		}
#if defined(_WIN32)
		// (+256 MiB: un PC de "16 GB" da algo menos de 16 GiB fisicos.)
		if (const auto ram = EmulatorMemory(); ram != 0 && ram <= (uint64_t {16} << 30u) + (uint64_t {256} << 20u))
			return Mode::Off; // (11-26 limpieza) antes GiveBack; KYTY_ENGINE_FREE=1 lo enciende
#endif
		return Mode::Off;
	}();
	return mode;
}

uint64_t          g_lock = 0, g_stats_global = 0, g_site = 0;
std::atomic<bool> g_stop {false}, g_running {false};

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// (Memoria del juego que el juego cambia a la vez: lecturas atomicas relajadas, alineadas; std::atomic_ref, C++20.)
uint64_t Load(uint64_t address) {
	return std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(address)).load(std::memory_order_relaxed);
}

uint32_t Load32(uint64_t address) {
	return std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t*>(address)).load(std::memory_order_relaxed);
}

// El cerrojo del asignador como lo toma el juego (0 libre, todo unos tomado; cmpxchg en 0x80f5ca).
bool LockAllocator(int patience_ms) {
	const int64_t until = NowNs() + int64_t {patience_ms} * 1'000'000;
	std::atomic_ref<uint64_t> word(*reinterpret_cast<uint64_t*>(g_lock));
	for (uint32_t spins = 0;; spins++) {
		uint64_t expected = 0;
		if (word.load(std::memory_order_relaxed) == 0 &&
		    word.compare_exchange_strong(expected, ~uint64_t {0}, std::memory_order_acquire, std::memory_order_relaxed))
			return true;
#if defined(_WIN32)
		YieldProcessor();
#endif
		if ((spins & 1023u) == 1023u) {
			if (NowNs() > until || g_stop.load(std::memory_order_relaxed)) return false;
			std::this_thread::yield();
		}
	}
}

void UnlockAllocator() {
	std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(g_lock)).store(0, std::memory_order_release);
}

bool GuestPointer(uint64_t pointer, bool checked) {
	if (pointer < 0x100000000ull || pointer >= (uint64_t {1} << 40u) || (pointer & 7u) != 0) return false;
	return !checked || Libs::Graphics::HostMemoryRangeIsReadable(pointer, 0x40);
}

std::atomic<uint64_t> g_bad_ranges {0}, g_bad_shown {0};

bool GoodRange(uint64_t address, uint64_t size) {
	constexpr uint64_t Top = uint64_t {1} << 40u;
	return address >= 0x100000000ull && size != 0 && address < Top && size <= Top - address && (address & 0xffffu) == 0 &&
	       (size & 0xffffu) == 0;
}

struct FreeRun {
	uint64_t descriptor = 0, address = 0, size = 0, head = 0;
};

struct WalkTotals {
	std::array<uint64_t, Pools> blocks {}, pages {}, counted_blocks {}, counted_pages {};
};

// (Cerrojo tomado) los bloques libres con todas sus paginas mapeadas. False: las listas no son como en el codigo.
bool WalkFreeLocked(std::vector<FreeRun>& runs, WalkTotals& totals, bool checked) {
	const uint64_t stats = Load(g_stats_global);
	if (!GuestPointer(stats, checked)) return false;
	for (uint64_t pool = 0; pool < Pools; pool++) {
		const uint64_t record = stats + pool * PoolStride, head = record + FreeHeadAt;
		totals.counted_blocks[pool] = Load32(record + FreeCountAt);
		totals.counted_pages[pool]  = Load32(record + FreePagesAt);
		uint64_t node               = Load(record + FreeFirstAt);
		for (uint64_t steps = 0; node != head; steps++) {
			if (steps > 1'000'000 || !GuestPointer(node, checked)) return false;
			const uint64_t descriptor = Load(node);
			if (!GuestPointer(descriptor, checked) || descriptor + NodeAt != node || Load(descriptor + ListAt) != head) return false;
			const uint64_t size = Load(descriptor + SizeAt), mapped = Load(descriptor + MappedAt);
			totals.blocks[pool]++;
			totals.pages[pool] += mapped >> 16u;
			if (mapped == size && size != 0) {
				const uint64_t address = Load(descriptor + AddressAt);
				// (Ghidra 0x80e7a0/0x80eb90/0x80f550: +0x60 es la direccion virtual del juego, alineada a 64 KiB; un rango
				// fuera de [4 GiB, 1 TiB) o sin alinear no es memoria del juego que siga el emulador: no se toca.)
				if (!GoodRange(address, size)) {
					g_bad_ranges.fetch_add(1, std::memory_order_relaxed);
					if (g_bad_shown.fetch_add(1, std::memory_order_relaxed) < 4) {
						std::printf("[page-free] diag: rango descartado pool %llu desc 0x%llx dir 0x%llx tam 0x%llx\n",
						            static_cast<unsigned long long>(pool), static_cast<unsigned long long>(descriptor),
						            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size));
						std::fflush(stdout);
					}
				} else {
					runs.push_back({descriptor, address, size, head});
				}
			}
			node = Load(descriptor + NextAt);
		}
	}
	return true;
}

bool StillFree(const FreeRun& run) {
	return Load(run.descriptor + ListAt) == run.head && Load(run.descriptor + AddressAt) == run.address &&
	       Load(run.descriptor + SizeAt) == run.size && Load(run.descriptor + MappedAt) == run.size &&
	       Load(run.descriptor + NodeAt) == run.descriptor;
}

#if defined(_WIN32)
// (Cerrojo tomado, bloque comprobado libre) las paginas de [address, address + size) soltadas.
// Memoria nativa (KYTY_NATIVE_MEMORY): MEM_RESET sobre la propia direccion del juego, solo en regiones comprometidas
// privadas y de lectura-escritura: una pagina vigilada (raiz, PageManager, KYTY_VRAM_UNICA/offload: solo lectura o sin
// acceso) se deja como esta y se cuenta en *skipped; una sin comprometer (commit-real) no tiene nada que soltar. El
// compromiso no cambia (MEM_RESET no descompromete): el juego la reutiliza sin fallo de compromiso.
// Sin memoria nativa: la vista de respaldo (DiscardVirtualMemory) y la del juego fuera del working set.
uint64_t DropLocked(uint64_t address, uint64_t size, uint64_t* skipped) {
	uint64_t done = 0;
	if (Libs::LibKernel::Memory::NativeMemoryEnabled()) {
		for (uint64_t at = address, end = address + size; at < end;) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) {
				*skipped += end - at;
				break;
			}
			const uint64_t stop =
			    std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
			if (stop <= at) break;
			const DWORD protect = info.Protect & 0xffu;
			const bool  rw      = (info.Protect & PAGE_GUARD) == 0 && (protect == PAGE_READWRITE || protect == PAGE_EXECUTE_READWRITE);
			if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && rw) {
				if (VirtualAlloc(reinterpret_cast<void*>(at), stop - at, MEM_RESET, PAGE_READWRITE) != nullptr) {
					(void)VirtualUnlock(reinterpret_cast<void*>(at), stop - at); // (falla "not locked": sale del working set)
					done += stop - at;
				} else {
					*skipped += stop - at;
				}
			} else if (info.State == MEM_COMMIT) {
				*skipped += stop - at;
			}
			at = stop;
		}
		return done;
	}
	thread_local std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
	pieces.clear();
	if (!Libs::LibKernel::Memory::TryGetBackingPieces(address, size, pieces) || pieces.empty()) {
		*skipped += size;
		return 0;
	}
	for (const auto& [pointer, length]: pieces) {
		if (DiscardVirtualMemory(const_cast<uint8_t*>(pointer), length) == ERROR_SUCCESS) {
			(void)VirtualUnlock(const_cast<uint8_t*>(pointer), length);
			done += length;
		} else {
			*skipped += length;
		}
	}
	(void)VirtualUnlock(reinterpret_cast<void*>(address), size);
	return done;
}

// Lo que tarda ahora el cerrojo del working set (fuera del cerrojo del asignador): con Windows recortando el proceso
// (PC justo de RAM) cada soltado lo espera; entonces se pausa en vez de bloquear al juego.
uint64_t ProbeWorkingSetNs() {
	static auto* const page = static_cast<volatile uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	if (page == nullptr) return 0;
	const int64_t start = NowNs();
	page[0]             = static_cast<uint8_t>(page[0] + 1u);
	(void)VirtualAlloc(const_cast<uint8_t*>(page), 4096, MEM_RESET, PAGE_READWRITE);
	return static_cast<uint64_t>(NowNs() - start);
}
#endif

double Mib(uint64_t bytes) {
	return static_cast<double>(bytes) / 1048576.0;
}

void WalkerThread() {
	LocalPlatform::SetThreadName("Kyty.PageFree");
	const bool give_back = GetMode() == Mode::GiveBack;
	// Cada 2 s; solo bloques libres FreeWalks+1 recorridos seguidos (~30 s): los libres menos tiempo el juego los volvia
	// a tomar enseguida. 64 KiB por vez bajo el cerrojo, 128 MiB por pasada, 50 ms de cerrojo por pasada como mucho.
	constexpr int64_t  PassNs = 2'000'000'000;
	[[maybe_unused]] constexpr uint64_t Chunk = uint64_t {64} << 10u, PassBudget = uint64_t {128} << 20u, FreeWalks = 14;
	[[maybe_unused]] constexpr uint64_t LockBudgetNs = 50'000'000, SlowChunkNs = 20'000'000, SlowAverageNs = 8'000'000, SlowProbeNs = 2'000'000;
	[[maybe_unused]] constexpr uint64_t PauseWalks = 15, MaxPauseWalks = 60;
	struct Seen {
		uint64_t first = 0;
		bool     given = false;
	};
	std::map<std::pair<uint64_t, uint64_t>, Seen> seen, now_seen;
	std::vector<FreeRun>                          runs;
	[[maybe_unused]] uint64_t walk = 0, pause = PauseWalks, resume_walk = 0, slow_passes = 0;
	// (desde la ultima linea) soltado, devuelto, omitido, cerrojo, preparacion de caches, pausas.
	uint64_t given_30 = 0, given_total = 0, back_30 = 0, back_total = 0, skipped_30 = 0, hold_ns = 0, hold_max_ns = 0;
	uint64_t prepare_ns = 0, paused = 0;
	int64_t  report_at = NowNs() + 30'000'000'000;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(PassNs));
		if (g_stop.load(std::memory_order_relaxed)) break;
		if (Load(g_stats_global) == 0) continue; // (el asignador aun no existe)
		walk++;
		runs.clear();
		WalkTotals totals;
		const bool checked = walk <= 3;
		if (!LockAllocator(50)) continue;
		const int64_t locked = NowNs();
		const bool    walked = WalkFreeLocked(runs, totals, checked);
		UnlockAllocator();
		hold_max_ns             = std::max(hold_max_ns, static_cast<uint64_t>(NowNs() - locked));
		const bool counts_match = walked && totals.blocks == totals.counted_blocks && totals.pages == totals.counted_pages;
		if (!walked || (checked && !counts_match)) {
			std::printf("[page-free] las listas de libres del asignador no son como en su codigo (recorridas %d: %llu+%llu bloques; "
			            "sus cuentas %llu+%llu): apagado\n",
			            walked ? 1 : 0, static_cast<unsigned long long>(totals.blocks[0]),
			            static_cast<unsigned long long>(totals.blocks[1]), static_cast<unsigned long long>(totals.counted_blocks[0]),
			            static_cast<unsigned long long>(totals.counted_blocks[1]));
			std::fflush(stdout);
			break;
		}
		if (static bool shown = false; !shown && !runs.empty()) {
			shown = true;
			std::printf("[page-free] diag: %llu bloques libres mapeados; primeros:", static_cast<unsigned long long>(runs.size()));
			for (size_t i = 0; i < runs.size() && i < 4; i++)
				std::printf(" [0x%llx +0x%llx]", static_cast<unsigned long long>(runs[i].address),
				            static_cast<unsigned long long>(runs[i].size));
			std::printf(" (descartados %llu)\n", static_cast<unsigned long long>(g_bad_ranges.load(std::memory_order_relaxed)));
			std::fflush(stdout);
		}
		now_seen.clear();
		for (const auto& run: runs) {
			const auto key   = std::make_pair(run.address, run.size);
			const auto found = seen.find(key);
			now_seen[key]    = found != seen.end() ? found->second : Seen {walk, false};
		}
		// Soltados que ya no estan libres: el juego los reutiliza; Windows le da paginas al tocarlas.
		for (const auto& [key, entry]: seen)
			if (entry.given && now_seen.find(key) == now_seen.end()) back_30 += key.second, back_total += key.second;
		std::vector<std::pair<FreeRun, bool*>> candidates;
		for (const auto& run: runs) {
			auto& entry = now_seen[std::make_pair(run.address, run.size)];
			if (give_back && entry.first + FreeWalks <= walk && !entry.given) candidates.emplace_back(run, &entry.given);
		}
		std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first.size > b.first.size; });
#if defined(_WIN32)
		uint64_t budget = PassBudget, pass_hold_ns = 0, chunks = 0;
		bool     slow = false, ended = false;
		if (!candidates.empty() && walk >= resume_walk) slow = ProbeWorkingSetNs() >= SlowProbeNs;
		for (auto& [run, given_flag]: candidates) {
			if (walk < resume_walk || slow || ended || budget == 0 || pass_hold_ns >= LockBudgetNs || g_stop.load(std::memory_order_relaxed))
				break;
			// Caches de la GPU sobre el bloque invalidadas antes (como antes de una escritura de todo el rango que no lee
			// nada): una imagen o buffer en cache sobre paginas que pasan a ceros se vuelve a subir si el juego reutiliza el
			// bloque. Sin el cerrojo del asignador (toma los de las caches). Deja las paginas vigiladas en lectura-escritura.
			// Rango fuera de lo que el emulador sigue (direccion 0, tamano 0 o por encima de 1 TiB): no se toca.
			if (run.address == 0 || run.size == 0 || run.address >= (1ull << 40u) || run.size > (1ull << 40u) - run.address) {
				continue;
			}
			// (El bloque puede haberse vuelto a usar desde el recorrido: sin invalidar caches de datos vivos.)
			if (!LockAllocator(20)) continue;
			const bool still = StillFree(run);
			UnlockAllocator();
			if (!still) continue;
			const int64_t prepared = NowNs();
			(void)Libs::LibKernel::Memory::TryPrepareHostOverwrite(run.address, run.size);
			prepare_ns += static_cast<uint64_t>(NowNs() - prepared);
			bool whole = true;
			for (uint64_t at = 0; at < run.size; at += Chunk) {
				if (budget == 0 || pass_hold_ns >= LockBudgetNs) {
					whole = false;
					break;
				}
				if (ProbeWorkingSetNs() >= SlowProbeNs) {
					whole = false;
					ended = slow = true;
					break;
				}
				if (!LockAllocator(20)) {
					whole = false;
					break;
				}
				const int64_t  held  = NowNs();
				const bool     free  = StillFree(run);
				const uint64_t bytes = std::min(Chunk, run.size - at);
				uint64_t       skipped = 0;
				const uint64_t done    = free ? DropLocked(run.address + at, bytes, &skipped) : 0;
				UnlockAllocator();
				const auto took = static_cast<uint64_t>(NowNs() - held);
				hold_ns += took;
				pass_hold_ns += took;
				hold_max_ns = std::max(hold_max_ns, took);
				chunks++;
				if (!free) {
					whole = false;
					break;
				}
				given_30 += done;
				given_total += done;
				skipped_30 += skipped;
				budget -= std::min(budget, bytes);
				if (took >= SlowChunkNs) {
					whole = at + bytes >= run.size;
					ended = true;
					break;
				}
				std::this_thread::yield();
			}
			if (whole) *given_flag = true;
		}
		slow = slow || (chunks != 0 && pass_hold_ns >= SlowAverageNs * chunks);
		if (slow && ++slow_passes >= 2) {
			resume_walk = walk + pause;
			pause       = std::min(pause * 2, MaxPauseWalks);
			slow_passes = 0;
			paused++;
		} else if (!slow && chunks != 0) {
			slow_passes = 0;
			pause       = PauseWalks;
		}
#endif
		seen.swap(now_seen);
		if (const int64_t now = NowNs(); now >= report_at) {
			uint64_t free_bytes = 0;
			for (const auto& run: runs)
				free_bytes += run.size;
			std::printf("[page-free] 30 s: %s; libres %llu bloques %.0f MiB; soltado %.0f MiB (total %.0f MiB); devuelto al juego "
			            "%.0f MiB (total %.0f MiB); retenido ahora %.0f MiB; omitido (vigilado/no RW) %.0f MiB; cerrojo %.1f ms "
			            "(max %.2f ms); caches %.1f ms; pausas %llu\n",
			            give_back ? "giveback" : "walk", static_cast<unsigned long long>(runs.size()), Mib(free_bytes),
			            Mib(given_30), Mib(given_total), Mib(back_30), Mib(back_total),
			            Mib(given_total > back_total ? given_total - back_total : 0), Mib(skipped_30),
			            static_cast<double>(hold_ns) / 1e6, static_cast<double>(hold_max_ns) / 1e6,
			            static_cast<double>(prepare_ns) / 1e6, static_cast<unsigned long long>(paused));
			std::fflush(stdout);
			given_30 = back_30 = skipped_30 = hold_ns = hold_max_ns = prepare_ns = paused = 0;
			report_at = now + 30'000'000'000;
		}
	}
	g_running.store(false, std::memory_order_release);
}
} // namespace

void Install(Program* program) {
	DemonsSoulsAllocNativo::Install(program); // (m6: liberaciones -> MEM_DECOMMIT en el acto; PageFree queda de red)
#if defined(__x86_64__) || defined(_M_X64)
	if (GetMode() == Mode::Off || g_site != 0 || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	const auto               found = GuestCode::FindUnique(*program, GuestCode::Pattern(FreeBlock));
	const GuestCode::Pattern body(FreeBlockBody);
	if (!found || !Libs::Graphics::HostMemoryRangeIsReadable(*found + FreeBlockBodyAt, body.Size()) ||
	    !body.Matches(reinterpret_cast<const uint8_t*>(*found + FreeBlockBodyAt))) {
		std::printf("[page-free] no se encontro la liberacion del asignador de una version conocida: apagado\n");
		std::fflush(stdout);
		return;
	}
	int32_t lock_displacement = 0, stats_displacement = 0;
	std::memcpy(&lock_displacement, reinterpret_cast<const void*>(*found + LockOperandAt), sizeof(lock_displacement));
	std::memcpy(&stats_displacement, reinterpret_cast<const void*>(*found + StatsOperandAt), sizeof(stats_displacement));
	g_lock         = *found + LockOperandEnd + static_cast<int64_t>(lock_displacement);
	g_stats_global = *found + StatsOperandEnd + static_cast<int64_t>(stats_displacement);
	g_site         = *found;
	g_stop.store(false, std::memory_order_relaxed);
	g_running.store(true, std::memory_order_release);
	std::thread(WalkerThread).detach();
	std::printf("[page-free] asignador: liberacion en eboot+0x%llx, cerrojo eboot+0x%llx, estadisticas eboot+0x%llx: %s "
	            "(KYTY_ENGINE_FREE; memoria del emulador %.1f GiB)\n",
	            static_cast<unsigned long long>(g_site - program->base_vaddr),
	            static_cast<unsigned long long>(g_lock - program->base_vaddr),
	            static_cast<unsigned long long>(g_stats_global - program->base_vaddr),
	            GetMode() == Mode::GiveBack ? "bloques libres ~30 s devueltos a Windows" : "solo recorrido",
	            static_cast<double>(EmulatorMemory()) / 1073741824.0);
	std::fflush(stdout);
#else
	(void)program;
#endif
}

void Clear() {
	DemonsSoulsAllocNativo::Clear();
	g_stop.store(true, std::memory_order_relaxed);
	// (El hilo duerme 2 s entre pasadas y no toca memoria del juego tras ver g_stop: se espera a que salga, 3 s como mucho,
	// antes de que se libere la memoria del juego.)
	for (int i = 0; i < 300 && g_running.load(std::memory_order_acquire); i++)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	g_site = 0;
}
} // namespace Loader::DemonsSoulsPageFree
