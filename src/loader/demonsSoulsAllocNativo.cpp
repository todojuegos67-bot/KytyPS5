#include "loader/demonsSoulsAllocNativo.h"

#include "common/abi.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsPageFree.h"
#include "loader/demonsSoulsPortAlloc.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "local/local-platform.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "guest-lazy-commit.h" // (NativeLazyCommit: commit-real; incluye windows.h y psapi.h en Windows)

namespace Loader::DemonsSoulsAllocNativo {
namespace {

namespace PF = Loader::DemonsSoulsPageFree;

bool Requested() {
	static const bool on = [] {
		// (p1) Con KYTY_PORT_ALLOC=1 la liberacion 0x80f550 es de DemonsSoulsPortAlloc (sustituye a este gancho).
		if (Loader::DemonsSoulsPortAlloc::Requested()) return false;
		const char* v = std::getenv("KYTY_ALLOC_NATIVO");
		return v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
	}();
	return on;
}

// Registro de un pool (0xa0 bytes desde [0x2ee4d60]). (R1, Ghidra 11-25 sobre 0x80d9a0 / 0x80e7a0 / 0x80f550):
//   +0x20 bloques en uso (int), +0x70 paginas de 64 KiB libres y mapeadas (int),
//   +0x80 bytes de DIRECCIONES en uso (tamano pedido de cada bloque, descriptor +0x58; 0x80d9a0 lo suma entero aunque
//         el bloque quede sin mapear: rama param_5==2 resta del mapeado, no de aqui) y +0x88 su pico;
//   +0x90 bytes MAPEADOS en uso (paginas de los 12 GiB fisicos, descriptor +0x50; 0x80d9a0 suma pagina a pagina al
//         mapear, 0x80f550 resta +0x50 al liberar) y +0x98 su pico.
// Antes se mostraba +0x80 como "en uso": son direcciones, no memoria (por eso 16,9 GiB con 12,8 reservados).
constexpr uint64_t PoolBlocksAt = 0x20, PoolFreePagesAt = 0x70, PoolSpanAt = 0x80, PoolSpanPeakAt = 0x88,
                   PoolMappedAt = 0x90, PoolMappedPeakAt = 0x98;
// gStats.mTotalMemory - overhead ("Total Available Allocator Memory", [0x2ee4d20]) = [0x2ee4d60] - 0x40.
constexpr uint64_t AvailableFromStats = 0x40;
// Prologo de 0x80f550 que se mueve al trampolin: push rbp; mov rbp,rsp; push r15; push r14; push rbx; sub rsp,0x18
// (13 bytes, sin direcciones relativas a RIP).
constexpr size_t PrologueSize = 13;

using FreeFn = void(KYTY_SYSV_ABI*)(uint64_t);

uint64_t           g_site = 0, g_stats_global = 0;
FreeFn             g_original = nullptr;
std::atomic<bool>  g_stop {false}, g_running {false};
std::atomic<uint64_t> g_frees {0}, g_freed_bytes {0}, g_decommitted {0}, g_kept {0}, g_bad {0};

uint64_t Load(uint64_t address) {
	return std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(address)).load(std::memory_order_relaxed);
}
uint32_t Load32(uint64_t address) {
	return std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t*>(address)).load(std::memory_order_relaxed);
}

bool GoodRange(uint64_t address, uint64_t size) {
	constexpr uint64_t Top = uint64_t {1} << 40u;
	return address >= 0x100000000ull && size != 0 && address < Top && size <= Top - address && (address & 0xffffu) == 0 &&
	       (size & 0xffffu) == 0;
}

// Entrada de 0x80f550 (hilo del juego, ABI SysV). El bloque aun no esta en ninguna lista: es solo del que lo libera.
void KYTY_SYSV_ABI FreeHook(uint64_t descriptor) {
#if defined(_WIN32)
	const uint64_t size    = *reinterpret_cast<const uint64_t*>(descriptor + PF::SizeAt);
	const uint64_t mapped  = *reinterpret_cast<const uint64_t*>(descriptor + PF::MappedAt);
	const uint64_t address = *reinterpret_cast<const uint64_t*>(descriptor + PF::AddressAt);
	g_frees.fetch_add(1, std::memory_order_relaxed);
	g_freed_bytes.fetch_add(mapped, std::memory_order_relaxed);
	// (mapped != size: 0x80f550 desmapea el resto el mismo, 0x80eb90; solo los bloques mapeados enteros van a la lista libre.)
	if (mapped == size && GoodRange(address, size)) {
		// Copias de la GPU sobre el bloque fuera antes (pasa a ceros): si se reutiliza se vuelven a subir de lo que escriba.
		(void)Libs::LibKernel::Memory::TryPrepareHostOverwrite(address, size);
		uint64_t       rest = 0;
		const uint64_t done = NativeLazyCommit::DecommitFree(address, size, &rest);
		g_decommitted.fetch_add(done, std::memory_order_relaxed);
		g_kept.fetch_add(rest, std::memory_order_relaxed);
	} else if (mapped == size) {
		g_bad.fetch_add(1, std::memory_order_relaxed);
	}
#endif
	g_original(descriptor);
}

double Mib(uint64_t b) {
	return static_cast<double>(b) / 1048576.0;
}

void ReportThread() {
	LocalPlatform::SetThreadName("Kyty.AllocNativo");
	uint64_t last_frees = 0, last_blocks = 0, last_decommitted = 0;
	bool     first = true;
	for (;;) {
		for (int i = 0; i < 50 && !g_stop.load(std::memory_order_relaxed); i++)
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		if (g_stop.load(std::memory_order_relaxed)) break;
		const uint64_t stats = Load(g_stats_global);
		if (stats < 0x100000000ull) continue; // (el asignador aun no existe)
		const uint64_t reserved = Load(g_stats_global - AvailableFromStats);
		std::array<uint64_t, PF::Pools> in_use {}, peak {}, span {}, span_peak {}, free_mapped {}, blocks {};
		uint64_t total_blocks = 0;
		for (uint64_t p = 0; p < PF::Pools; p++) {
			const uint64_t r = stats + p * PF::PoolStride;
			in_use[p]        = Load(r + PoolMappedAt);
			peak[p]          = Load(r + PoolMappedPeakAt);
			span[p]          = Load(r + PoolSpanAt);
			span_peak[p]     = Load(r + PoolSpanPeakAt);
			free_mapped[p]   = uint64_t {Load32(r + PoolFreePagesAt)} << 16u;
			blocks[p]        = Load32(r + PoolBlocksAt);
			total_blocks += blocks[p];
		}
		uint64_t commit = 0, ram = 0;
#if defined(_WIN32)
		PROCESS_MEMORY_COUNTERS_EX pmc {};
		pmc.cb = sizeof(pmc);
		if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)) != 0) {
			commit = pmc.PrivateUsage;
			ram    = pmc.WorkingSetSize;
		}
		const uint64_t lazy_committed = NativeLazyCommit::Get().committed.load(std::memory_order_relaxed);
#else
		const uint64_t lazy_committed = 0;
#endif
		const uint64_t frees = g_frees.load(std::memory_order_relaxed), dec = g_decommitted.load(std::memory_order_relaxed);
		// asignaciones = bloques en uso nuevos + liberados (bloques en uso: +0x20 de cada pool).
		const uint64_t freed_now = frees - last_frees;
		const uint64_t allocs    = first ? 0 : (total_blocks + freed_now >= last_blocks ? total_blocks + freed_now - last_blocks : 0);
		// (R1) Memoria real: mapeado del juego (+0x90) y, de ello, lo que el emulador tiene comprometido (al tocar).
		// Las direcciones (+0x80) se dan aparte para que la cifra antigua siga comparable.
		std::printf("[alloc-nativo] 5 s: direcciones en uso CPU %.0f GPU %.0f MiB (pico %.0f; tamano pedido, NO es memoria)\n",
		            Mib(span[0]), Mib(span[1]), Mib(span_peak[1]));
		std::printf("[alloc-nativo] 5 s: reservado %.0f MiB; mapeado en uso (memoria real) CPU %.0f MiB (pico %.0f, %llu bloques) GPU %.0f MiB (pico %.0f, "
		            "%llu bloques); libres aun mapeados CPU %.0f GPU %.0f MiB; juego comprometido al tocar %.0f MiB; proceso "
		            "commit %.0f MiB, en RAM %.0f MiB; %.0f asig/s, %.0f lib/s; descomprometido %.0f MiB (total %.0f MiB), "
		            "vigilado sin tocar %.0f MiB, rangos raros %llu%s\n",
		            Mib(reserved), Mib(in_use[0]), Mib(peak[0]), static_cast<unsigned long long>(blocks[0]), Mib(in_use[1]),
		            Mib(peak[1]), static_cast<unsigned long long>(blocks[1]), Mib(free_mapped[0]), Mib(free_mapped[1]),
		            Mib(lazy_committed), Mib(commit), Mib(ram), static_cast<double>(allocs) / 5.0,
		            static_cast<double>(freed_now) / 5.0, Mib(dec - last_decommitted), Mib(dec),
		            Mib(g_kept.load(std::memory_order_relaxed)), static_cast<unsigned long long>(g_bad.load(std::memory_order_relaxed)),
#if defined(_WIN32)
		            NativeLazyCommit::Enabled() ? "" : " (SIN commit-real: no se descompromete nada)"
#else
		            ""
#endif
		);
		std::fflush(stdout);
		last_frees       = frees;
		last_blocks      = total_blocks;
		last_decommitted = dec;
		first            = false;
	}
	g_running.store(false, std::memory_order_release);
}

} // namespace

void Install(Program* program) {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(_WIN32)
	if (!Requested() || g_site != 0 || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	const auto               found = GuestCode::FindUnique(*program, GuestCode::Pattern(PF::FreeBlock));
	const GuestCode::Pattern body(PF::FreeBlockBody);
	if (!found || !Libs::Graphics::HostMemoryRangeIsReadable(*found + PF::FreeBlockBodyAt, body.Size()) ||
	    !body.Matches(reinterpret_cast<const uint8_t*>(*found + PF::FreeBlockBodyAt))) {
		std::printf("[alloc-nativo] no se encontro la liberacion del asignador (01.007.000): apagado\n");
		std::fflush(stdout);
		return;
	}
	const uint64_t site = *found;
	int32_t        stats_displacement = 0;
	std::memcpy(&stats_displacement, reinterpret_cast<const void*>(site + PF::StatsOperandAt), sizeof(stats_displacement));
	g_stats_global = site + PF::StatsOperandEnd + static_cast<int64_t>(stats_displacement);

	// Trampolin: el prologo original (13 bytes) y jmp [rip+0] de vuelta a site+13.
	auto* tramp = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	if (tramp == nullptr) return;
	std::memcpy(tramp, reinterpret_cast<const void*>(site), PrologueSize);
	const uint8_t  jmp_back[6] = {0xff, 0x25, 0x00, 0x00, 0x00, 0x00};
	const uint64_t back        = site + PrologueSize;
	std::memcpy(tramp + PrologueSize, jmp_back, sizeof(jmp_back));
	std::memcpy(tramp + PrologueSize + sizeof(jmp_back), &back, sizeof(back));
	DWORD old = 0;
	if (VirtualProtect(tramp, 4096, PAGE_EXECUTE_READ, &old) == 0) {
		VirtualFree(tramp, 0, MEM_RELEASE);
		return;
	}
	g_original = reinterpret_cast<FreeFn>(tramp);

	// Entrada de 0x80f550: mov rax, FreeHook; jmp rax; nop (13 bytes; rax no lleva argumento en una funcion sin varargs).
	uint8_t        patch[PrologueSize] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0, 0x90};
	const uint64_t hook                = reinterpret_cast<uint64_t>(&FreeHook);
	std::memcpy(patch + 2, &hook, sizeof(hook));
	if (VirtualProtect(reinterpret_cast<void*>(site), PrologueSize, PAGE_EXECUTE_READWRITE, &old) == 0) {
		std::printf("[alloc-nativo] no se pudo escribir la entrada de la liberacion: apagado\n");
		std::fflush(stdout);
		g_original = nullptr;
		return;
	}
	std::memcpy(reinterpret_cast<void*>(site), patch, sizeof(patch));
	DWORD ignored = 0;
	(void)VirtualProtect(reinterpret_cast<void*>(site), PrologueSize, old, &ignored);
	(void)FlushInstructionCache(GetCurrentProcess(), reinterpret_cast<void*>(site), PrologueSize);
	g_site = site;

	g_stop.store(false, std::memory_order_relaxed);
	g_running.store(true, std::memory_order_release);
	std::thread(ReportThread).detach();
	std::printf("[alloc-nativo] liberacion del asignador (eboot+0x%llx) redirigida: bloque libre -> MEM_DECOMMIT en el acto "
	            "(KYTY_ALLOC_NATIVO; commit-real %s)\n",
	            static_cast<unsigned long long>(site - program->base_vaddr), NativeLazyCommit::CommitRealRequested() ? "si" : "NO");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

void Clear() {
	g_stop.store(true, std::memory_order_relaxed);
	for (int i = 0; i < 300 && g_running.load(std::memory_order_acquire); i++)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	// (El codigo del juego se descarga con el programa: el gancho se va con el.)
	g_site = 0;
}

} // namespace Loader::DemonsSoulsAllocNativo
