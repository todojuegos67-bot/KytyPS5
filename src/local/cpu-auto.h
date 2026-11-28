// (cpu-auto) "Optimizacion de CPU: Automatico" del lanzador: clasifica la CPU (Vieja / Justa / Moderna / X3D) con la
// topologia de cpu5-hibrido.h, AVX2 (cpuid) y las L3 (asimetricas = V-Cache en un solo CCD) y fija con ello opciones
// que ya existen: ayuda con las subidas (KYTY_UPLOAD_HELP), sin espera activa (KYTY_SPIN_GATE), precompilacion
// todos/relajado, carga de shaders al maximo y todos los CCD. Sin valores ajustados a una CPU concreta ni flags de
// compilacion. Manual: cada opcion como la elija el usuario (settings.ini cpu_optimizacion_auto=0).
#ifndef CPUAUTO_DECL
#define CPUAUTO_DECL
#include <cstdio>
#include <string>

#include "local/cpu5-hibrido.h"

namespace CpuAuto {
enum class Cat { Vieja, Justa, Moderna, X3D };

struct Decision {
	Cat  cat          = Cat::Moderna;
	bool upload_help  = false;
	bool spin_gate    = false;
	bool precomp_all  = true;
	bool shader_turbo = true;
	bool all_ccds     = false;
	bool avx2 = true, l3_asim = false;
};

bool TieneAvx2();     // CPUAUTO_IMPL
bool L3Asimetrica();  // CPUAUTO_IMPL

inline Decision Calcular() {
	static const Decision d = [] {
		Decision r;
		const auto& t = Cpu5::Get();
		r.avx2        = TieneAvx2();
		r.l3_asim     = L3Asimetrica();
		const bool vieja = !r.avx2 || (t.nucleos > 0 && t.nucleos <= 4) || (t.hilos > 0 && t.hilos <= 8);
		const bool justa = (t.hilos > 0 && t.hilos <= 12) || (t.hibrida && t.p > 0 && t.p <= 6);
		r.cat = vieja ? Cat::Vieja : justa ? Cat::Justa : r.l3_asim ? Cat::X3D : Cat::Moderna;
		r.upload_help  = r.cat == Cat::Vieja || r.cat == Cat::Justa;
		r.spin_gate    = r.upload_help;
		r.precomp_all  = r.cat != Cat::Vieja;
		r.shader_turbo = r.cat != Cat::Vieja;
		r.all_ccds     = false; // X3D: el juego en el CCD con V-Cache; sin L3 asimetrica no cambia nada
		return r;
	}();
	return d;
}

inline const char* Nombre(Cat c) {
	return c == Cat::Vieja ? "Vieja" : c == Cat::Justa ? "Justa" : c == Cat::X3D ? "X3D" : "Moderna";
}

template <class S>
void Aplicar(S& s) {
	const Decision d   = Calcular();
	s.perf_upload_help = d.upload_help;
	s.perf_spin_gate   = d.spin_gate;
	s.precompile_threads = d.precomp_all ? "todos" : "relajado";
	s.shader_turbo     = d.shader_turbo;
	s.all_ccds         = d.all_ccds;
}

template <class S>
void Log(const S& s) {
	const Decision d = Calcular();
	const auto&    t = Cpu5::Get();
	std::printf("[cpu-auto] optimizacion de CPU: %s; categoria %s (%d hilos, %d nucleos, %d P, %d E, AVX2 %s, L3 %s); "
	            "ayuda subidas=%d, sin espera activa=%d, precompilar=%s, shaders al maximo=%d, todos los CCD=%d\n",
	            s.cpu_auto ? "automatico" : "manual", Nombre(d.cat), t.hilos, t.nucleos, t.p, t.e, d.avx2 ? "si" : "no",
	            d.l3_asim ? "asimetrica" : "simetrica", s.perf_upload_help ? 1 : 0, s.perf_spin_gate ? 1 : 0,
	            s.precompile_threads.c_str(), s.shader_turbo ? 1 : 0, s.all_ccds ? 1 : 0);
	std::fflush(stdout);
}

} // namespace CpuAuto
#endif // CPUAUTO_DECL

#if defined(CPUAUTO_IMPL) && !defined(CPUAUTO_HECHO)
#define CPUAUTO_HECHO
#ifdef _WIN32
#include <windows.h>
#endif
namespace CpuAuto {
bool TieneAvx2() {
#if defined(__x86_64__) || defined(_M_X64)
	unsigned a, b, c, d;
	__asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0u), "c"(0u));
	if (a < 7) return false;
	__asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(1u), "c"(0u));
	if ((c & (1u << 27)) == 0 || (c & (1u << 28)) == 0) return false; // OSXSAVE, AVX
	unsigned lo, hi;
	__asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
	if ((lo & 6u) != 6u) return false;
	__asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(7u), "c"(0u));
	return (b & (1u << 5)) != 0;
#else
	return true;
#endif
}

bool L3Asimetrica() {
#ifdef _WIN32
	DWORD len = 0;
	GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
	if (len == 0) return false;
	std::string buf(len, '\0');
	if (!GetLogicalProcessorInformationEx(RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len))
		return false;
	unsigned long long mx = 0, mn = ~0ull;
	int                n  = 0;
	for (DWORD at = 0; at < len;) {
		const auto* i = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + at);
		if (i->Size == 0) break;
		if (i->Relationship == RelationCache && i->Cache.Level == 3 && i->Cache.GroupMask.Group == 0) {
			n++;
			mx = i->Cache.CacheSize > mx ? i->Cache.CacheSize : mx;
			mn = i->Cache.CacheSize < mn ? i->Cache.CacheSize : mn;
		}
		at += i->Size;
	}
	return n >= 2 && mx > mn; // el mismo criterio que LargestL3Mask (ds-main.cpp)
#else
	return false;
#endif
}
} // namespace CpuAuto
#endif // CPUAUTO_IMPL
