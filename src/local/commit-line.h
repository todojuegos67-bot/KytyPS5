#pragma once
// (COMMIT-LINE) KYTY_COMMIT_LINE=1 por defecto (=0 la apaga; solo mide, no cambia nada): una linea cada 10 s
//   [commit] 10 s, MiB: proceso X (pico en estos 10 s Y), sistema A de B (C %), RAM disponible D de E; top 5 ...
// Motivo (logs 20261125-202747/203403/204919): saber si lo que se agota es el limite de compromiso del sistema o la RAM
// fisica, y quien tiene el commit del proceso, con la misma linea antes y despues de cada cambio.
// - proceso: commit privado del proceso (PrivateUsage, lo que cuenta para el limite de compromiso); se muestrea cada
//   1 s y se da el maximo de los ultimos 10 s (asi sale el pico de la precompilacion de shaders aunque dure poco).
// - sistema: CommitTotal / CommitLimit de GetPerformanceInfo; RAM disponible: PhysicalAvailable.
// - top 5: bloques privados (AllocationBase) por lo comprometido ahora (VirtualQuery de todo el espacio, ~45 000 rangos,
//   unos ms cada 10 s, en un hilo aparte de prioridad baja), con el dueno si KYTY_ALLOC_GRANDE=1 vio la reserva
//   ("modulo+offset"), "monton C++" si es un segmento de un monton del proceso, y "?" si no se sabe.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include "local/alloc-grande.h"

namespace CommitLine {

[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_COMMIT_LINE");
		return v != nullptr && *v == '1'; // (11-26 limpieza) apagado por defecto
	}();
	return on;
}

inline uint64_t ProcessCommit() noexcept {
	PROCESS_MEMORY_COUNTERS_EX pmc {};
	pmc.cb = sizeof(pmc);
	if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)) == 0)
		return 0;
	return pmc.PrivateUsage;
}

// Los 5 bloques privados con mas commit: (base, bytes comprometidos).
inline std::vector<std::pair<uint64_t, uint64_t>> Top5() {
	std::vector<std::pair<uint64_t, uint64_t>> all;
	MEMORY_BASIC_INFORMATION                   info {};
	uint64_t                                   at = 0x10000, base = 0, committed = 0;
	auto flush = [&] {
		if (base != 0 && committed != 0) all.emplace_back(base, committed);
	};
	while (VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) == sizeof(info)) {
		const auto alloc = reinterpret_cast<uint64_t>(info.AllocationBase);
		if (alloc != base) {
			flush();
			base      = alloc;
			committed = 0;
		}
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE) committed += info.RegionSize;
		const uint64_t next = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		if (next <= at) break;
		at = next;
	}
	flush();
	const size_t n = std::min<size_t>(5, all.size());
	std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(n), all.end(),
	                  [](const auto& a, const auto& b) { return a.second > b.second; });
	all.resize(n);
	return all;
}

inline std::string Label(uint64_t base) {
	std::string owner = AllocGrande::OwnerOf(base);
	if (!owner.empty()) return owner;
	HANDLE heaps[256] {};
	const DWORD count = GetProcessHeaps(256, heaps);
	for (DWORD i = 0; i < count && i < 256; ++i) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(heaps[i], &info, sizeof(info)) == sizeof(info) && reinterpret_cast<uint64_t>(info.AllocationBase) == base)
			return "monton C++";
	}
	return AllocGrande::On() ? "?" : "? (dueno con KYTY_ALLOC_GRANDE=1)";
}

// Una vez, con el log ya iniciado (Emulator Init).
inline void Start() {
	if (!On()) return;
	std::printf("[commit] ON (KYTY_COMMIT_LINE=1 por defecto, =0 la apaga): cada 10 s commit del proceso (pico de los 10 s), "
	            "commit del sistema frente al limite, RAM disponible y los 5 bloques privados con mas commit\n");
	std::fflush(stdout);
	std::thread([] {
		SetThreadDescription(GetCurrentThread(), L"Kyty.CommitLine");
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
		const auto t0        = std::chrono::steady_clock::now();
		uint64_t   peak      = 0;
		uint64_t   peak_all  = 0;
		long long  last      = 0;
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			const uint64_t now = ProcessCommit();
			peak               = std::max(peak, now);
			peak_all           = std::max(peak_all, now);
			const auto s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count();
			if (s - last < 10) continue;
			last = s;
			PERFORMANCE_INFORMATION perf {};
			perf.cb = sizeof(perf);
			uint64_t sys = 0, limit = 0, avail = 0, phys = 0;
			if (K32GetPerformanceInfo(&perf, sizeof(perf)) != 0) {
				const uint64_t page = perf.PageSize;
				sys   = static_cast<uint64_t>(perf.CommitTotal) * page;
				limit = static_cast<uint64_t>(perf.CommitLimit) * page;
				avail = static_cast<uint64_t>(perf.PhysicalAvailable) * page;
				phys  = static_cast<uint64_t>(perf.PhysicalTotal) * page;
			}
			std::string top;
			for (const auto& [base, bytes]: Top5()) {
				char text[400];
				std::snprintf(text, sizeof(text), "%s0x%llx=%llu %s", top.empty() ? "" : ", ",
				              static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes >> 20u),
				              Label(base).c_str());
				top += text;
			}
			std::printf("[commit] %lld s, MiB (KYTY_COMMIT_LINE=1; =0 la apaga): proceso %llu (pico en estos 10 s %llu, desde "
			            "el arranque %llu); sistema %llu de %llu (%llu %%); RAM disponible %llu de %llu; top 5 bloques "
			            "privados por comprometido: %s\n",
			            static_cast<long long>(s), static_cast<unsigned long long>(now >> 20u), static_cast<unsigned long long>(peak >> 20u),
			            static_cast<unsigned long long>(peak_all >> 20u), static_cast<unsigned long long>(sys >> 20u),
			            static_cast<unsigned long long>(limit >> 20u),
			            static_cast<unsigned long long>(limit != 0 ? sys * 100u / limit : 0u),
			            static_cast<unsigned long long>(avail >> 20u), static_cast<unsigned long long>(phys >> 20u),
			            top.c_str());
			std::fflush(stdout);
			peak = 0;
		}
	}).detach();
}

} // namespace CommitLine
#else
namespace CommitLine {
inline void Start() {}
} // namespace CommitLine
#endif
