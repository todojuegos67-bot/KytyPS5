#pragma once
// (ALLOC-GRANDE) KYTY_ALLOC_GRANDE=1 (apagado por defecto; solo mide, no cambia nada): quien pide al sistema los bloques
// grandes del proceso. Motivo: en el log 20261124-221631 hay dos bloques privados de 1016 MiB (0xfc02000000 y
// 0xfda0ce0000; 516 + 183 MiB en RAM) fuera del juego y fuera de VMA (Vulkan en RAM del sistema 489 MiB), y en el codigo
// del emulador no hay ninguna reserva de ese tamano: hay que saber con datos de que modulo salen.
// Como: se cambian en las tablas de importacion (IAT) de todos los modulos cargados las ranuras de
// ntdll!NtAllocateVirtualMemory y ntdll!NtAllocateVirtualMemoryEx (kernelbase las importa: cubre VirtualAlloc,
// VirtualAllocEx y VirtualAlloc2 de cualquier DLL). Se repite cada 2 s los primeros 2 minutos y despues cada 30 s,
// para las DLL que se cargan tarde (el driver de Vulkan, Streamline, FSR, XeSS...). En el gancho, si la llamada sale
// bien, es de este proceso y es de 256 MiB o mas, se copian la pila (RtlCaptureStackBackTrace) y el tamano a un anillo
// sin locks; un hilo aparte resuelve el modulo de cada direccion (GetModuleHandleExW + GetModuleFileNameW) e imprime:
//   [alloc-grande] 0x<base> 1016 MiB commit|reserva prot 0x4 <- nvoglv64.dll+0x123456 (pila: a.dll+0x.. b.dll+0x..)
// El primer marco que no es de ntdll/kernelbase/kernel32 es el dueno. Ademas, en cada pasada se listan una vez los
// bloques privados de 256 MiB o mas que ya existian sin pasar por el gancho ("sin pila: anterior al gancho").
//
// (ALLOC-GRANDE 2, misma variable) Commit por quien llama: cada MEM_COMMIT que sale bien de KYTY_ALLOC_GRANDE_KB o mas
// (por defecto 64 KiB: los arenas de 1 GiB del driver crecen comprometiendo trozos dentro de su reserva, un minimo de
// 1 MiB podria no verlos; con KYTY_ALLOC_GRANDE_KB=1024 solo los de 1 MiB o mas) copia 16 marcos de pila a un segundo
// anillo sin locks (el gancho no resuelve ni reserva nada). El hilo Kyty.AllocGrande lo vacia cada 2 s, agrupa por
// dueno (primer marco que no es ntdll/kernelbase/kernel32) + primer marco de Demon's Souls.exe por encima (quien del
// emulador llego al driver), guarda los rangos (un rango vuelto a comprometer pasa al ultimo dueno) y cada 10 s imprime
// el top 10 por lo comprometido AHORA (VirtualQuery de sus rangos: liberados y descomprometidos no cuentan) con lo que
// esta en RAM (QueryWorkingSetEx, una muestra por 64 KiB, como mucho 4096 por rango):
//   [alloc-commit] 10 s: vistos N commits (X MiB), perdidos P; comprometido ahora por dueno (MiB, en RAM):
//   [alloc-commit]  1. nvoglv64.dll+0xa87a14 desde Demon's Souls.exe+0x123 : 1017 comprometidos, 466 en RAM, 812 commits
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

namespace AllocGrande {

[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_ALLOC_GRANDE");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

constexpr uint64_t kMin    = uint64_t {256} << 20u;
constexpr int      kFrames = 24;
constexpr int      kRing   = 256;

struct Entry {
	std::atomic<uint32_t> state {0}; // 0 libre, 1 escribiendo, 2 listo
	uint64_t              base = 0, size = 0;
	uint32_t              type = 0, prot = 0, frames = 0;
	void*                 pc[kFrames] {};
};
inline Entry                 g_ring[kRing];
inline std::atomic<uint32_t> g_head {0};
inline std::atomic<uint64_t> g_lost {0};

using NtAlloc   = LONG(NTAPI*)(HANDLE, PVOID*, ULONG_PTR, PSIZE_T, ULONG, ULONG);
using NtAllocEx = LONG(NTAPI*)(HANDLE, PVOID*, PSIZE_T, ULONG, ULONG, void*, ULONG);
inline NtAlloc   g_nt_alloc    = nullptr;
inline NtAllocEx g_nt_alloc_ex = nullptr;

// (COMMIT-LINE) base -> dueno de cada reserva grande vista (lo usa la linea [commit] de commit-line.h).
inline std::mutex                         g_owner_mutex;
inline std::map<uint64_t, std::string>    g_owner_of;
inline std::string OwnerOf(uint64_t base) {
	std::lock_guard lock(g_owner_mutex);
	const auto      it = g_owner_of.find(base);
	return it != g_owner_of.end() ? it->second : std::string {};
}
inline bool Self(HANDLE process) noexcept {
	return process == GetCurrentProcess() || GetProcessId(process) == GetCurrentProcessId();
}

// ---- (ALLOC-GRANDE 2) commits por quien llama ----
constexpr int kCFrames = 16;
constexpr int kCRing   = 8192;

struct CEntry {
	std::atomic<uint32_t> state {0}; // 0 libre, 1 escribiendo, 2 listo
	uint64_t              base = 0, size = 0;
	uint32_t              frames = 0;
	void*                 pc[kCFrames] {};
};
inline CEntry                g_cring[kCRing];
inline std::atomic<uint32_t> g_chead {0};
inline std::atomic<uint64_t> g_clost {0};
inline std::atomic<uint64_t> g_cseen {0};
inline std::atomic<uint64_t> g_cbytes {0};

inline uint64_t CommitMin() noexcept {
	static const uint64_t min = [] {
		const char* v  = std::getenv("KYTY_ALLOC_GRANDE_KB");
		uint64_t    kb = v != nullptr ? std::strtoull(v, nullptr, 10) : 64;
		if (kb < 4) kb = 4;
		return kb << 10u;
	}();
	return min;
}

inline void RecordCommit(uint64_t base, uint64_t size) noexcept {
	g_cseen.fetch_add(1, std::memory_order_relaxed);
	g_cbytes.fetch_add(size, std::memory_order_relaxed);
	const uint32_t i        = g_chead.fetch_add(1, std::memory_order_relaxed) % kCRing;
	auto&          e        = g_cring[i];
	uint32_t       expected = 0;
	if (!e.state.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
		g_clost.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	e.base   = base;
	e.size   = size;
	e.frames = RtlCaptureStackBackTrace(0, kCFrames, e.pc, nullptr); // los del gancho se saltan al agrupar
	e.state.store(2, std::memory_order_release);
}

inline void Seen(LONG status, HANDLE process, PVOID* base, PSIZE_T size, ULONG type) noexcept {
	if (status < 0 || size == nullptr || base == nullptr || *base == nullptr || (type & MEM_COMMIT) == 0) return;
	if (*size < CommitMin() || !Self(process)) return;
	RecordCommit(reinterpret_cast<uint64_t>(*base), *size);
}

inline void Record(uint64_t base, uint64_t size, ULONG type, ULONG prot) noexcept {
	const uint32_t i = g_head.fetch_add(1, std::memory_order_relaxed) % kRing;
	auto&          e = g_ring[i];
	uint32_t       expected = 0;
	if (!e.state.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
		g_lost.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	e.base   = base;
	e.size   = size;
	e.type   = type;
	e.prot   = prot;
	e.frames = RtlCaptureStackBackTrace(1, kFrames, e.pc, nullptr);
	e.state.store(2, std::memory_order_release);
}

inline LONG NTAPI HookAlloc(HANDLE process, PVOID* base, ULONG_PTR zero_bits, PSIZE_T size, ULONG type, ULONG prot) {
	const LONG status = g_nt_alloc(process, base, zero_bits, size, type, prot);
	Seen(status, process, base, size, type);
	if (status >= 0 && size != nullptr && *size >= kMin && base != nullptr && (type & (MEM_COMMIT | MEM_RESERVE)) != 0 &&
	    Self(process))
		Record(reinterpret_cast<uint64_t>(*base), *size, type, prot);
	return status;
}

inline LONG NTAPI HookAllocEx(HANDLE process, PVOID* base, PSIZE_T size, ULONG type, ULONG prot, void* params, ULONG count) {
	const LONG status = g_nt_alloc_ex(process, base, size, type, prot, params, count);
	Seen(status, process, base, size, type);
	if (status >= 0 && size != nullptr && *size >= kMin && base != nullptr && (type & (MEM_COMMIT | MEM_RESERVE)) != 0 &&
	    Self(process))
		Record(reinterpret_cast<uint64_t>(*base), *size, type, prot);
	return status;
}

// Ranuras de importacion de todos los modulos que apuntan a `from` -> `to`. Devuelve cuantas.
inline int Patch(void* from, void* to) noexcept {
	if (from == nullptr) return 0;
	HMODULE handles[2048];
	DWORD   needed = 0;
	if (EnumProcessModules(GetCurrentProcess(), handles, sizeof(handles), &needed) == 0) return 0;
	const size_t n       = std::min<size_t>(needed / sizeof(HMODULE), 2048);
	int          patched = 0;
	for (size_t m = 0; m < n; m++) {
		auto*       image = reinterpret_cast<uint8_t*>(handles[m]);
		const auto* dos   = reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
		if (dos->e_magic != IMAGE_DOS_SIGNATURE) continue;
		const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(image + dos->e_lfanew);
		if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) continue;
		const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IAT];
		if (dir.VirtualAddress == 0 || dir.Size == 0) continue;
		auto* slots = reinterpret_cast<void**>(image + dir.VirtualAddress);
		for (size_t s = 0; s < dir.Size / sizeof(void*); s++) {
			if (slots[s] != from) continue;
			DWORD old = 0;
			if (VirtualProtect(&slots[s], sizeof(void*), PAGE_READWRITE, &old) == 0) continue;
			slots[s] = to;
			VirtualProtect(&slots[s], sizeof(void*), old, &old);
			patched++;
		}
	}
	return patched;
}

// "modulo.dll+0x1234" de una direccion; false si no es de ningun modulo.
inline bool Where(const void* pc, char* out, size_t out_size, bool* system) noexcept {
	HMODULE module = nullptr;
	*system        = false;
	if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       static_cast<LPCWSTR>(pc), &module) == 0 ||
	    module == nullptr) {
		std::snprintf(out, out_size, "0x%llx(sin modulo)", static_cast<unsigned long long>(reinterpret_cast<uint64_t>(pc)));
		return false;
	}
	wchar_t path[MAX_PATH] {};
	GetModuleFileNameW(module, path, MAX_PATH);
	const wchar_t* name = std::wcsrchr(path, L'\\');
	name                = name != nullptr ? name + 1 : path;
	*system = _wcsicmp(name, L"ntdll.dll") == 0 || _wcsicmp(name, L"kernelbase.dll") == 0 || _wcsicmp(name, L"kernel32.dll") == 0;
	std::snprintf(out, out_size, "%ls+0x%llx", name,
	              static_cast<unsigned long long>(reinterpret_cast<uint64_t>(pc) - reinterpret_cast<uint64_t>(module)));
	return true;
}

inline void Drain(std::unordered_set<uint64_t>* seen) {
	for (auto& e: g_ring) {
		if (e.state.load(std::memory_order_acquire) != 2) continue;
		char owner[200] = "?", stack[700] = "";
		size_t used = 0;
		bool   have = false;
		for (uint32_t f = 0; f < e.frames; f++) {
			char text[200];
			bool system = false;
			Where(e.pc[f], text, sizeof(text), &system);
			if (!have && !system) {
				std::snprintf(owner, sizeof(owner), "%s", text);
				have = true;
			} else if (have && used + 2 < sizeof(stack) && f < 12) {
				used += static_cast<size_t>(std::snprintf(stack + used, sizeof(stack) - used, " %s", text));
				if (used >= sizeof(stack)) used = sizeof(stack) - 1;
			}
		}
		std::printf("[alloc-grande] 0x%llx %llu MiB %s prot 0x%x <- %s (pila:%s)\n", static_cast<unsigned long long>(e.base),
		            static_cast<unsigned long long>(e.size >> 20u), (e.type & MEM_COMMIT) != 0 ? "commit" : "reserva",
		            static_cast<unsigned>(e.prot), owner, stack);
		{
			std::lock_guard lock(g_owner_mutex);
			g_owner_of[e.base] = owner;
		}
		seen->insert(e.base);
		e.state.store(0, std::memory_order_release);
	}
	std::fflush(stdout);
}

// Bloques privados >= 256 MiB que el gancho no vio (anteriores a el o de un camino no enganchado), una vez cada uno.
inline void Census(std::unordered_set<uint64_t>* seen, std::unordered_set<uint64_t>* told) {
	MEMORY_BASIC_INFORMATION info {};
	uint64_t                 at = 0x10000;
	uint64_t                 base = 0, committed = 0;
	auto flush = [&] {
		if (base != 0 && committed >= kMin && seen->count(base) == 0 && told->insert(base).second)
			std::printf("[alloc-grande] 0x%llx %llu MiB comprometidos, sin pila (anterior al gancho o camino no enganchado)\n",
			            static_cast<unsigned long long>(base), static_cast<unsigned long long>(committed >> 20u));
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
	std::fflush(stdout);
}

// ---- (ALLOC-GRANDE 2) agrupado y top 10, solo en el hilo Kyty.AllocGrande ----
struct Owner {
	std::string name;
	uint64_t    commits = 0, bytes = 0;
};
struct CommitState {
	std::unordered_map<uint64_t, std::string> where;  // pc -> "modulo+0x.." (cache; bit bajo de la clave: es de sistema)
	std::unordered_map<std::string, uint32_t> index;  // nombre -> dueno
	std::vector<Owner>                        owners; // dueno
	std::map<uint64_t, std::pair<uint64_t, uint32_t>> ranges; // base -> {fin, dueno}
};

inline const std::string& Name(CommitState* st, void* pc, bool* system, bool* exe) {
	static const HMODULE self = GetModuleHandleW(nullptr);
	const auto           key  = reinterpret_cast<uint64_t>(pc);
	auto                 it   = st->where.find(key);
	if (it == st->where.end()) {
		char text[200];
		bool sys = false;
		Where(pc, text, sizeof(text), &sys);
		HMODULE module = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                   static_cast<LPCWSTR>(pc), &module);
		std::string s(1, sys ? 's' : (module == self ? 'e' : 'o'));
		s += text;
		it = st->where.emplace(key, std::move(s)).first;
	}
	*system = it->second[0] == 's';
	*exe    = it->second[0] == 'e';
	return it->second;
}

// Quita de `ranges` lo que se solapa con [b, e) (partiendo los bordes).
inline void Cut(CommitState* st, uint64_t b, uint64_t e) {
	auto it = st->ranges.lower_bound(b);
	if (it != st->ranges.begin()) {
		auto prev = std::prev(it);
		if (prev->second.first > b) it = prev;
	}
	while (it != st->ranges.end() && it->first < e) {
		const uint64_t rb = it->first, re = it->second.first;
		const uint32_t o = it->second.second;
		it               = st->ranges.erase(it);
		if (rb < b) st->ranges.emplace(rb, std::make_pair(b, o));
		if (re > e) {
			st->ranges.emplace(e, std::make_pair(re, o));
			break;
		}
	}
}

inline void DrainCommits(CommitState* st) {
	for (auto& e: g_cring) {
		if (e.state.load(std::memory_order_acquire) != 2) continue;
		std::string owner = "?", from;
		bool        have  = false;
		// Los primeros marcos son del gancho (en el exe): se empieza tras el primer marco de sistema (kernelbase, que es
		// quien importa NtAllocateVirtualMemory); si no hay ninguno, tras los marcos del exe del principio.
		uint32_t start = 0;
		{
			uint32_t first_sys = e.frames;
			for (uint32_t f = 0; f < e.frames && f < 8; f++) {
				bool sys = false, exe = false;
				Name(st, e.pc[f], &sys, &exe);
				if (sys) {
					first_sys = f;
					break;
				}
			}
			if (first_sys < e.frames) {
				start = first_sys;
			} else {
				for (; start < e.frames; start++) {
					bool sys = false, exe = false;
					Name(st, e.pc[start], &sys, &exe);
					if (!exe) break;
				}
			}
		}
		for (uint32_t f = start; f < e.frames; f++) {
			bool sys = false, exe = false;
			const std::string& n = Name(st, e.pc[f], &sys, &exe);
			if (!have && !sys) {
				owner = n.substr(1);
				have  = true;
				if (exe) break; // el propio emulador: no hace falta "desde"
			} else if (have && exe) {
				from = n.substr(1);
				break;
			}
		}
		const std::string key = from.empty() ? owner : owner + " desde " + from;
		auto              idx = st->index.find(key);
		if (idx == st->index.end()) {
			idx = st->index.emplace(key, static_cast<uint32_t>(st->owners.size())).first;
			st->owners.push_back({key, 0, 0});
		}
		auto& o = st->owners[idx->second];
		o.commits++;
		o.bytes += e.size;
		const uint64_t b = e.base & ~uint64_t {0xfff}, end = (e.base + e.size + 0xfff) & ~uint64_t {0xfff};
		Cut(st, b, end);
		st->ranges.emplace(b, std::make_pair(end, idx->second));
		e.state.store(0, std::memory_order_release);
	}
}

// Comprometido ahora y en RAM de [b, e) (privado). En RAM: una muestra por 64 KiB comprometidos, como mucho 4096.
inline void Measure(uint64_t b, uint64_t e, uint64_t* committed, uint64_t* resident) {
	MEMORY_BASIC_INFORMATION info {};
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> ws;
	uint64_t at = b, c = 0;
	std::vector<std::pair<uint64_t, uint64_t>> runs;
	while (at < e && VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) == sizeof(info)) {
		const uint64_t rb = std::max<uint64_t>(at, reinterpret_cast<uint64_t>(info.BaseAddress));
		const uint64_t re = std::min<uint64_t>(e, reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize);
		if (re <= rb) break;
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE) {
			c += re - rb;
			runs.emplace_back(rb, re);
		}
		at = re;
	}
	*committed += c;
	if (c == 0) return;
	const uint64_t step = std::max<uint64_t>(uint64_t {64} << 10u, ((c / 4096) + 0xffff) & ~uint64_t {0xffff});
	for (const auto& r: runs)
		for (uint64_t p = r.first; p < r.second; p += step) {
			PSAPI_WORKING_SET_EX_INFORMATION i {};
			i.VirtualAddress = reinterpret_cast<void*>(p);
			ws.push_back(i);
		}
	if (ws.empty()) return;
	if (K32QueryWorkingSetEx(GetCurrentProcess(), ws.data(), static_cast<DWORD>(ws.size() * sizeof(ws[0]))) == 0) return;
	uint64_t in = 0;
	for (const auto& i: ws)
		if (i.VirtualAttributes.Valid != 0) in++;
	*resident += c * in / ws.size();
}

inline void ReportCommits(CommitState* st, long long seconds) {
	std::vector<uint64_t> committed(st->owners.size(), 0), resident(st->owners.size(), 0);
	for (const auto& r: st->ranges) Measure(r.first, r.second.first, &committed[r.second.second], &resident[r.second.second]);
	std::vector<uint32_t> order(st->owners.size());
	for (uint32_t i = 0; i < order.size(); i++) order[i] = i;
	std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return committed[a] > committed[b]; });
	uint64_t all_c = 0, all_r = 0;
	for (size_t i = 0; i < committed.size(); i++) {
		all_c += committed[i];
		all_r += resident[i];
	}
	std::printf("[alloc-commit] %lld s: vistos %llu commits de %llu KiB o mas (%llu MiB en total), perdidos %llu; comprometido "
	            "ahora por lo visto %llu MiB, en RAM %llu MiB; top 10 por comprometido ahora (MiB):\n",
	            seconds, static_cast<unsigned long long>(g_cseen.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(CommitMin() >> 10u),
	            static_cast<unsigned long long>(g_cbytes.load(std::memory_order_relaxed) >> 20u),
	            static_cast<unsigned long long>(g_clost.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(all_c >> 20u), static_cast<unsigned long long>(all_r >> 20u));
	for (size_t k = 0; k < order.size() && k < 10; k++) {
		const uint32_t i = order[k];
		if (committed[i] == 0) break;
		std::printf("[alloc-commit] %2zu. %s : %llu comprometidos, %llu en RAM, %llu fuera; %llu commits (%llu MiB pedidos)\n", k + 1,
		            st->owners[i].name.c_str(), static_cast<unsigned long long>(committed[i] >> 20u),
		            static_cast<unsigned long long>(resident[i] >> 20u),
		            static_cast<unsigned long long>((committed[i] - resident[i]) >> 20u),
		            static_cast<unsigned long long>(st->owners[i].commits),
		            static_cast<unsigned long long>(st->owners[i].bytes >> 20u));
	}
	std::fflush(stdout);
}

// Una vez, con el log ya iniciado y antes de los graficos (Emulator Init).
inline void Start() {
	if (!On()) return;
	HMODULE ntdll  = GetModuleHandleW(L"ntdll.dll");
	g_nt_alloc     = reinterpret_cast<NtAlloc>(GetProcAddress(ntdll, "NtAllocateVirtualMemory"));
	g_nt_alloc_ex  = reinterpret_cast<NtAllocEx>(GetProcAddress(ntdll, "NtAllocateVirtualMemoryEx"));
	const int first = Patch(reinterpret_cast<void*>(g_nt_alloc), reinterpret_cast<void*>(&HookAlloc)) +
	                  Patch(reinterpret_cast<void*>(g_nt_alloc_ex), reinterpret_cast<void*>(&HookAllocEx));
	std::printf("[alloc-grande] ON (KYTY_ALLOC_GRANDE=1): %d ranuras de importacion enganchadas (NtAllocateVirtualMemory/Ex); "
	            "linea por cada bloque de 256 MiB o mas con el modulo que lo pide\n",
	            first);
	std::fflush(stdout);
	std::thread([] {
		SetThreadDescription(GetCurrentThread(), L"Kyty.AllocGrande");
		std::unordered_set<uint64_t> seen, told;
		CommitState                  commits;
		const auto                   t0 = std::chrono::steady_clock::now();
		long long                    last_patch = 0, last_report = 0;
		for (;;) {
			// (ALLOC-GRANDE 2) cada 2 s se vacia el anillo de commits; ganchos y censo como antes (2 s / 30 s).
			std::this_thread::sleep_for(std::chrono::seconds(2));
			const auto s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count();
			DrainCommits(&commits);
			if (s - last_report >= 10) {
				last_report = s;
				ReportCommits(&commits, s);
			}
			if (s >= 120 && s - last_patch < 30) continue;
			last_patch = s;
			const int more = Patch(reinterpret_cast<void*>(g_nt_alloc), reinterpret_cast<void*>(&HookAlloc)) +
			                 Patch(reinterpret_cast<void*>(g_nt_alloc_ex), reinterpret_cast<void*>(&HookAllocEx));
			if (more != 0) {
				std::printf("[alloc-grande] %d ranuras nuevas enganchadas (DLL cargadas despues)\n", more);
				std::fflush(stdout);
			}
			Drain(&seen);
			Census(&seen, &told);
			if (const auto lost = g_lost.exchange(0); lost != 0) std::printf("[alloc-grande] %llu registros perdidos (anillo lleno)\n", static_cast<unsigned long long>(lost));
		}
	}).detach();
}

} // namespace AllocGrande
#else
namespace AllocGrande {
inline void Start() {}
} // namespace AllocGrande
#endif
