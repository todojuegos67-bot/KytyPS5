// (cpu5-hibrido) Topologia de la CPU para el lanzador: nucleos P/E reales (EfficiencyClass de
// GetLogicalProcessorInformationEx), hilos y la decision "pocos hilos". Una CPU hibrida con 6 nucleos P o menos
// (i5-12400F 6P, i5-13400F/14400F 6P+4E: 16 hilos) entra en "pocos hilos" aunque tenga mas de 12 hilos: sus E-cores
// no sirven al render (ya va fijado a los P por KYTY_RECORDING_CPUS=auto). KYTY_HIBRIDO=0: solo cuenta hilos (<=12),
// como antes. Las dos opciones que activa (KYTY_UPLOAD_HELP=2, KYTY_SPIN_GATE=1) son adaptativas: solo actuan con la
// CPU cargada. Sin efecto en CPUs con 7+ nucleos P ni en no hibridas de 13+ hilos.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
// La deteccion se compila solo donde se define CPU5_HIBRIDO_IMPL: al final de ds-settings.cpp (windows.h incluido
// alli el ultimo, sin chocar con el resto del archivo; en Linux solo cuenta hilos).

namespace Cpu5 {
struct Topologia {
	int  hilos = 0, nucleos = 0, p = 0, e = 0, hilos_p = 0;
	bool hibrida = false, pocos = false, por_p = false, apagado = false;
};

Topologia Detectar();
const Topologia& Get();
void Log();

#ifdef CPU5_HIBRIDO_IMPL
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
Topologia Detectar() {
	Topologia t;
	t.hilos = static_cast<int>(std::thread::hardware_concurrency());
#ifdef _WIN32
	DWORD len = 0;
	(void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
	if (len > 0) {
		std::vector<char> buf(len);
		if (GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
			struct C { int cls, th; };
			std::vector<C> cores;
			int hi = 0, lo = 255;
			for (DWORD at = 0; at < len;) {
				const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + at);
				if (info->Size == 0) break;
				int th = 0;
				for (WORD g = 0; g < info->Processor.GroupCount; g++)
					for (KAFFINITY m = info->Processor.GroupMask[g].Mask; m != 0; m >>= 1) th += static_cast<int>(m & 1);
				const int cls = info->Processor.EfficiencyClass;
				cores.push_back({cls, th});
				hi = cls > hi ? cls : hi;
				lo = cls < lo ? cls : lo;
				at += info->Size;
			}
			t.nucleos = static_cast<int>(cores.size());
			t.hibrida = !cores.empty() && hi != lo;
			for (const auto& c: cores) {
				if (!t.hibrida || c.cls == hi) t.p++, t.hilos_p += c.th; else t.e++;
			}
		}
	}
#endif
	const char* sw = std::getenv("KYTY_HIBRIDO");
	t.apagado = sw != nullptr && sw[0] == '0';
	const bool por_hilos = t.hilos != 0 && t.hilos <= 12;
	t.por_p = !t.apagado && t.hibrida && t.p > 0 && t.p <= 6;
	t.pocos = por_hilos || t.por_p;
	return t;
}

const Topologia& Get() {
	static const Topologia t = Detectar();
	return t;
}

void Log() {
	const auto& t = Get();
	std::printf("[cpu5] topologia: %d hilos, %d nucleos (%s: %d P con %d hilos, %d E); pocos hilos: %s (%s)%s\n", t.hilos, t.nucleos,
	            t.hibrida ? "hibrida" : "no hibrida", t.p, t.hilos_p, t.e, t.pocos ? "si" : "no",
	            t.por_p ? "<=6 nucleos P" : (t.pocos ? "<=12 hilos" : "CPU grande"), t.apagado ? "; KYTY_HIBRIDO=0" : "");
	std::fflush(stdout);
}
#endif
} // namespace Cpu5
