#pragma once
// (w6, 11-25) KYTY_VRAM_UNICA=1 (apagado por defecto hasta validarlo; =0 o sin poner: como antes). Una sola copia de las
// texturas de streaming: la de la VRAM. Va dentro del offload nativo (KYTY_NATIVE_TEXTURE_OFFLOAD=1), que ya suelta la RAM
// de una textura con MEM_RESET (memory.cpp NativeOffloadDrop) y la re-descarga de la GPU si alguien la toca.
//
// (w6) Comparacion por trozo (cada mapeo): los iguales a la descarga salen de RAM, los distintos se quedaban en RAM. Un
// trozo distinto ya no veta ni la clase ni el formato.
//
// (w12) Los trozos distintos (logs 20261125-120123 y -121720: 391 trozos, 251 MiB). Medido: todos en texturas 2048x2048
// 12 mips 0x560000 (fmt 181/173/177), primer byte distinto en +0x20000; media 0,64 MiB por trozo distinto y 2,06 MiB por
// trozo igual, unos 2 de cada clase por imagen. Cuadra con los trozos de streaming: [0,0x20000) cola de mips,
// [0x20000,0x60000) mip 2, [0x60000,0x160000) mip 1, [0x160000,0x560000) mip 0: cola y mip 0 iguales, mips 1 y 2 distintos.
// Ghidra: el streamer (FUN_00ac5da0) sube de trozo en trozo (SetCurrentStreamingChunk 0x901330 con i+1); cada subida
// compromete el trozo y lo copia entero (memcpy), la bajada lo descompromete (op 2). La memoria del juego es la verdad (la
// escribe la CPU; el escaneo solo mira texturas sin escritura de la GPU): un trozo distinto es copia de la GPU vieja, o una
// descarga grabada antes de una subida pendiente.
// Arreglo: (a) antes de grabar la descarga de comprobacion se vacian las subidas pendientes de su rango
// (AsyncUpload::DrainRange); (b) la primera vez que una imagen sale con trozos distintos, esos trozos se suben otra vez
// desde la memoria del juego (InvalidateCpuAliases, como una escritura de la CPU) y la imagen se comprueba otra vez cuando
// vuelve a estar limpia; si sale igual, sale de RAM entera. Si vuelve a salir distinta se queda como antes (iguales fuera,
// distintos en RAM) y se cuenta en "siguen distintas": entonces no es copia vieja.
//
// Linea cada 30 s: "[vram-unica] 30 s: ..." y (u1) "[vram-unica] 30 s: MiB soltados ...; rechazadas ... por motivo".

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace VramUnica {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_UNICA");
		const bool  r = v != nullptr && v[0] == '1'; // (11-25 noche) APAGADO por defecto: texturas negras con 55/58/59
		std::printf("[vram-unica] KYTY_VRAM_UNICA=%d: texturas de streaming fuera de RAM por trozos probados (los distintos se "
		            "suben otra vez y se comprueban de nuevo; sin veto de formato)\n",
		            r ? 1 : 0);
		std::fflush(stdout);
		return r;
	}();
	return on;
}

struct Counters {
	std::atomic<uint64_t> pieces_equal {0}, bytes_equal {0}, pieces_differ {0}, bytes_differ {0};
	std::atomic<uint64_t> images_split {0}, runs_taken {0}, bytes_taken {0}, vetoes_avoided {0};
	// (w12)
	std::atomic<uint64_t> images_reup {0}, bytes_reup {0}, fixed_after_reup {0}, still_differ {0}, pieces_gpu_zero {0};
	// (u1) Ultima pasada de OffloadScan: bytes de texturas muestreadas que siguen en RAM, por motivo (foto, no suma).
	std::atomic<uint64_t> rej[16] {};
	std::atomic<uint64_t> scanned_tex {0}, scanned_vm {0};
};
// (u1) Motivos de rechazo (OffloadScan, texturas muestreadas no escritas por la GPU).
// (H2, 11-25) Los que antes salian sin contarse (log 20261125-215220: vistas 1638, contadas 368): grande (> tope
// KYTY_OFFLOAD_MAX_IMAGE_MB de TextureOffload::Taken, los arrays de VirtualMemo de 88-352 MiB), ya fuera/tocada/retenida,
// RT nativo, y las que esta pasada suelta o deja para soltar ("tomada").
enum Rej : int { RejSucia, RejSinVigilancia, RejPequena, RejSinRespaldo, RejPlan, RejClase, RejDistinta, RejEspera,
	             RejGrande, RejYaFuera, RejRtNativo, RejTomada, RejN };
static_assert(RejN <= 16, "rej[]");
inline const char* RejName(int r) {
	static const char* const n[] = {"sucia",      "sin vigilancia", "pequena (<minimo)", "sin respaldo nativo",
	                                "sin descarga", "clase/profundidad", "distinta", "en espera/comprobando",
	                                "grande (>tope, arrays)", "ya fuera/tocada/retenida", "RT nativo", "tomada esta pasada"};
	return r >= 0 && r < RejN ? n[r] : "?";
}
inline Counters& C() {
	static Counters c;
	return c;
}

// Render thread (OffloadScan): the 30 s line.
inline void Report() {
	using Clock      = std::chrono::steady_clock;
	static auto last = Clock::now();
	const auto  now  = Clock::now();
	if (!On() || now - last < std::chrono::seconds(30)) return;
	last     = now;
	auto& c  = C();
	const auto mib = [](uint64_t b) { return static_cast<double>(b) / 1048576.0; };
	std::printf("[vram-unica] 30 s: trozos iguales %llu (%.0f MiB), distintos %llu (%.0f MiB; GPU a ceros %llu); imagenes "
	            "partidas %llu; resubidas %llu (%.0f MiB), arregladas tras resubir %llu, siguen distintas %llu (en RAM); "
	            "soltado en total %llu rangos, %.0f MiB\n",
	            static_cast<unsigned long long>(c.pieces_equal.load()), mib(c.bytes_equal.load()),
	            static_cast<unsigned long long>(c.pieces_differ.load()), mib(c.bytes_differ.load()),
	            static_cast<unsigned long long>(c.pieces_gpu_zero.load()), static_cast<unsigned long long>(c.images_split.load()),
	            static_cast<unsigned long long>(c.images_reup.load()), mib(c.bytes_reup.load()),
	            static_cast<unsigned long long>(c.fixed_after_reup.load()), static_cast<unsigned long long>(c.still_differ.load()),
	            static_cast<unsigned long long>(c.runs_taken.load()), mib(c.bytes_taken.load()));
	// (u1) Soltado y rechazado por motivo (MiB, ultima pasada).
	std::printf("[vram-unica] 30 s: MiB soltados %.0f; texturas vistas en la ultima pasada %.0f MiB (en la zona VirtualMemo "
	            "%.0f); rechazadas (siguen en RAM) por motivo:",
	            mib(c.bytes_taken.load()), mib(c.scanned_tex.load()), mib(c.scanned_vm.load()));
	for (int r = 0; r < RejN; r++) std::printf("%s %s %.0f", r == 0 ? "" : ",", RejName(r), mib(c.rej[r].load()));
	std::printf("\n");
	// (w12) Por tipo de copia doble ([ram19]): lo que esta via saco de RAM. RT/imagenes de la GPU y buffers: sin tocar.
	std::printf("[ram-unica] liberado (MiB): texturas %.0f; RT e imagenes escritas por la GPU 0; buffers 0\n",
	            mib(c.bytes_taken.load()));
	std::fflush(stdout);
}

} // namespace VramUnica
