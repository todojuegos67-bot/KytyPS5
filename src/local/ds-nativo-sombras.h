#pragma once
// (F5 pase 2, pieza 1) KYTY_NATIVO_PASE_SOMBRAS=2 (apagado por defecto; Demon's Souls 01.007.000): EXTRACTOR de la lista de
// mallas de profundidad de las sombras, verificado contra la DCB. Solo mide: no cambia nada de lo que hace el juego.
//
// Ghidra (projelf_par1, direcciones del juego):
//  - CascadedShadows 0xa11220: por cascada i (0..3) lanza un trabajo de render (0xa04ed0 = RunFunctionAsRenderJob) con
//    0xa13650 -> 0xa13e60(ctx). ctx[0x10] = la lista visible de la cascada (render+0xe8b8 - i*0xae0), ctx[0x15] = i.
//  - 0xa13e60 dibuja: 0xab57d0(renderer, lista, ctx, cubo, i) con los cubos 0, 1 y 3 (+ 0xa57af0 VirtualGeo, 0xa9c9a0,
//    0xab6080 con i y i+4).
//  - 0xab57d0(renderer, lista, ctx, cubo, pase): vector de entradas de 0x20 bytes en lista+0x178+cubo*0x28 (inicio, fin).
//    entrada+8 = item; item[0] = material (material+0x20 = tecnica/shader: 0xa84ee0 la enlaza al cambiar), item[2] =
//    instancia de modelo (0xabc4b0 -> 0xabc100: submalla = *(inst+0x130) + entrada.0x14*0x48, CMesh = *(*(sub+0x20)+0x70),
//    skinning si sub+0x38 != 0; 0x914360(cmd, 4, item[4]+0x38, +0x3c) es el draw indexado), 0xaba900(item, ctx, entrada+0x18) = el draw.
//    Devuelve (fin - inicio) >> 5 = el numero de entradas. pase < 0: no es de sombras (telemetria "%s_info.csv").
//  - La malla (0xaa06a0): +0x298 bloque de vertices, +0x2a0 de indices (MALLA-NATIVA los registra).
//
// El gancho (loader/demonsSoulsNativoSombras.inc) en la entrada de 0xab57d0, solo con pase >= 0: cuenta entradas,
// materiales distintos y mallas con VkBuffer de MALLA-NATIVA; ejecuta la funcion entera del juego y cuenta los draws
// (IT_DRAW_*) que escribio en la DCB entre el cursor de antes y el de despues. Linea "[nativo-sombras]" cada 5 s.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "graphics/host_gpu/hostMemory.h" // (HostMemoryRangeIsReadable: el volcado de la GPU)
#include "local/ds-sombras-ab.h" // (sombras-ab: tiempos del camino nativo)
#include "local/ds-censo-escena.h" // (F5 pase 3: censo del escenario; mismo gancho 0xab57d0)

namespace DsNativoSombras {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_NATIVO_PASE_SOMBRAS");
		if (text == nullptr || text[0] == '\0' || text[1] != '\0') return 0u;
		// (=2 mide; =3 dibujo nativo en imagen propia + comparacion; =1 =3 y, en la cascada que dio >= 99,9 %, reemplaza)
		return text[0] == '1' ? 1u : text[0] == '2' ? 2u : text[0] == '3' ? 3u : 0u;
	}();
	return mode;
}

enum Counter : uint32_t {
	Calls,       // llamadas de sombras (pase >= 0)
	Entries,     // entradas de la lista (= draws que el juego pide)
	Materials,   // cambios de material (tecnica) dentro de la lista
	MeshNative,  // entradas cuya malla tiene VkBuffer (o memoria ya preparada) de MALLA-NATIVA
	MeshKnown,   // entradas cuya malla esta registrada en MALLA-NATIVA (con o sin VkBuffer)
	DcbDraws,    // draws IT_DRAW_* escritos en la DCB por la llamada
	DcbChecked,  // llamadas con la DCB contada (mismo buffer, cursor solo hacia arriba)
	DcbMatch,    // llamadas con draws de la DCB == entradas
	DcbOther,    // llamadas sin DCB contable (otro buffer / sin command buffer)
	Skinned,     // entradas con vertices animados (submalla+0x38 != 0)
	Requested,   // bloques de sombra pedidos al Preparer de MALLA-NATIVA desde aqui
	NativeStatic, // entradas con VkBuffer de MALLA-NATIVA y sin skinning (las que el pase nativo puede dibujar)
	Counters
};
inline std::atomic<uint64_t> g_counters[Counters] {};
inline std::atomic<uint64_t> g_cascade_entries[8] {};
inline void Add(Counter c, uint64_t n = 1) { g_counters[c].fetch_add(n, std::memory_order_relaxed); }

inline void Report() {
	using Clock = std::chrono::steady_clock;
	static std::atomic<int64_t> last {0};
	const int64_t now  = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
	int64_t       prev = last.load(std::memory_order_relaxed);
	if (now - prev < 5000 || !last.compare_exchange_strong(prev, now)) return;
	const auto v = [](Counter c) { return static_cast<unsigned long long>(g_counters[c].load(std::memory_order_relaxed)); };
	const auto e = [](int i) { return static_cast<unsigned long long>(g_cascade_entries[i].load(std::memory_order_relaxed)); };
	std::printf("[nativo-sombras] modo=%u (mide) llamadas=%llu entradas=%llu cambios_material=%llu malla_nativa=%llu "
	            "malla_registrada=%llu con_skinning=%llu sin_skinning=%llu nativa_sin_skinning=%llu pedidas_al_preparer=%llu | DCB: draws=%llu llamadas_contadas=%llu coinciden=%llu sin_dcb=%llu | por pase 0..7: "
	            "%llu %llu %llu %llu %llu %llu %llu %llu\n",
	            Mode(), v(Calls), v(Entries), v(Materials), v(MeshNative), v(MeshKnown), v(Skinned), v(Entries) - v(Skinned), v(NativeStatic), v(Requested), v(DcbDraws), v(DcbChecked),
	            v(DcbMatch), v(DcbOther), e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7));
	std::fflush(stdout);
}

// (volcado GPU, =2) La marca que el gancho de 0xab57d0 escribe en la DCB delante de los draws de una cascada (solo hasta
// que la GPU volco esa cascada): NOP R_ZERO de 4 dwords: cabecera, "SOMB", "MARC", pase. graphicsRun la salta y apunta el
// pase; el primer DrawIndex que viene despues vuelca los USER_DATA del VS (gs_user_sgpr), los punteros de 48 bits que
// forman (256 bytes, y 64 bytes de cada puntero de su tabla) y la direccion del codigo del VS.
constexpr uint32_t MarkDwords = 6; // (cabecera, "SOMB", "MARC", pase, indice del primer draw: lo, hi)
constexpr uint32_t MarkHeader = 0xC0000000u | ((MarkDwords - 2u) << 16u) | (0x10u << 8u);
constexpr uint32_t MarkMagic0 = 0x424d4f53u, MarkMagic1 = 0x4352414du; // "SOMB" "MARC"
inline std::atomic<uint32_t> g_gpu_done_mask {0}; // (bit por pase 0..3: ya volcado por la GPU)
inline std::atomic<uint32_t> g_gpu_pending {0xFFFFFFFFu}; // (pase marcado pendiente de volcar)

[[nodiscard]] inline bool IsMark(const uint32_t* p, uint32_t remaining) {
	return remaining >= MarkDwords && p[0] == MarkHeader && p[1] == MarkMagic0 && p[2] == MarkMagic1;
}
inline void GpuHex(const char* what, uint64_t addr, uint64_t bytes) {
	if (addr == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(addr, bytes)) {
		std::printf("[nativo-sombras-volcado]   GPU %s @%llx: no legible\n", what, static_cast<unsigned long long>(addr));
		return;
	}
	uint32_t w[64] {};
	std::memcpy(w, reinterpret_cast<const void*>(addr), bytes < sizeof(w) ? bytes : sizeof(w));
	float f[64] {};
	std::memcpy(f, w, sizeof(w));
	for (uint64_t i = 0; i < bytes / 4 && i + 3 < 64; i += 4)
		std::printf("[nativo-sombras-volcado]   GPU %s +%03llx: %08x %08x %08x %08x | %g %g %g %g\n", what,
		            static_cast<unsigned long long>(i * 4), w[i], w[i + 1], w[i + 2], w[i + 3], double(f[i]), double(f[i + 1]),
		            double(f[i + 2]), double(f[i + 3]));
}
inline std::atomic<uint64_t> g_gpu_pending_index {0}; // (indice del primer draw de la lista marcada, segun el juego)

// El codigo de un shader en hexadecimal, hasta s_endpgm (0xbf810000) o 4 KiB, para desensamblarlo fuera (llvm-mc amdgcn).
inline void GpuCode(const char* stage, uint64_t addr) {
	// (Log T32 11-27: con 8 huecos y 20 PS distintos, al llenarse se volcaba en CADA llamada: 522 MB de log y 2-3 ms por
	// llamada del gancho en los hilos del juego. Ahora cada shader una sola vez; con la tabla llena, nada.)
	static std::atomic<uint64_t> seen[64] {};
	if (addr == 0) return;
	bool claimed = false;
	for (auto& k: seen) {
		uint64_t expected = 0;
		if (k.load(std::memory_order_relaxed) == addr) return;
		if (k.compare_exchange_strong(expected, addr)) { claimed = true; break; }
		if (expected == addr) return;
	}
	if (!claimed) return;
	uint64_t dwords = 0;
	while (dwords < 1024 && Libs::Graphics::HostMemoryRangeIsReadable(addr + dwords * 4, 4)) {
		const uint32_t w = reinterpret_cast<const uint32_t*>(addr)[dwords++];
		if (w == 0xbf810000u) break;
	}
	std::printf("[nativo-sombras-volcado] GPU codigo %s @%llx: %llu dwords\n", stage, static_cast<unsigned long long>(addr),
	            static_cast<unsigned long long>(dwords));
	const auto* w = reinterpret_cast<const uint32_t*>(addr);
	for (uint64_t i = 0; i < dwords; i += 8) {
		std::printf("[nativo-sombras-volcado]   GPU codigo %s +%04llx:", stage, static_cast<unsigned long long>(i * 4));
		for (uint64_t j = i; j < i + 8 && j < dwords; ++j) std::printf(" %08x", w[j]);
		std::printf("\n");
	}
}
// (11-24) El hilo de la GPU NO lee memoria del juego: un fallo de pagina ahi espera a la propia GPU. Solo imprime los
// registros y deja las direcciones aqui; las lee un hilo del juego (el gancho, sin candados) en su siguiente llamada.
inline std::atomic<uint64_t> g_req_ptr[16] {};   // (punteros de 48 bits formados por pares de USER_DATA)
inline std::atomic<uint64_t> g_req_code[4] {};   // (codigo de es, ls, hs, gs)
inline std::atomic<uint32_t> g_req_ready {0};    // (pase + 1 cuando hay peticion)

inline void GpuUser(const char* stage, const uint32_t* user, uint32_t count, uint32_t& slot) {
	if (count > 32u) count = 32u;
	std::printf("[nativo-sombras-volcado]   GPU %s user_sgpr=%u:", stage, count);
	for (uint32_t i = 0; i < count; ++i) std::printf(" %08x", user[i]);
	std::printf("\n");
	for (uint32_t i = 0; i + 1 < count && slot < 16; ++i) {
		const uint64_t a = uint64_t {user[i]} | (uint64_t {user[i + 1] & 0xFFFFu} << 32u);
		if ((user[i + 1] >> 16u) != 0 || a < 0x100000ull || (a & 3u) != 0) continue;
		g_req_ptr[slot++].store(a, std::memory_order_relaxed);
	}
}
// (hilo de la GPU) Entre la marca y el primer draw: los paquetes que pueden escribir memoria (dispatch, DMA_DATA,
// WRITE_DATA, COPY_DATA) con su destino; en los dispatch, la direccion del CS. Solo palabras del paquete que el
// procesador ya esta leyendo; nada mas. Se imprimen en la linea "GPU pase=".
inline char     g_notes[768] = "";
inline uint32_t g_notes_len  = 0;
inline uint32_t g_notes_count = 0;
inline void GpuNotePacket(uint32_t opcode, const uint32_t* p, uint32_t remaining, uint64_t cs_addr) {
	const uint32_t len = ((p[0] >> 16u) & 0x3fffu) + 2u;
	if (len > remaining || g_notes_len >= sizeof(g_notes) - 96u) return;
	int n = 0;
	switch (opcode) {
		case 0x15: // IT_DISPATCH_DIRECT
		case 0x16: // IT_DISPATCH_INDIRECT
			n = std::snprintf(g_notes + g_notes_len, sizeof(g_notes) - g_notes_len, " dispatch(op=%02x cs=%llx)", opcode,
			                  static_cast<unsigned long long>(cs_addr));
			break;
		case 0x50: // IT_DMA_DATA: dst = p[4] | p[5] << 32
			if (len >= 7)
				n = std::snprintf(g_notes + g_notes_len, sizeof(g_notes) - g_notes_len, " dma(dst=%llx src=%llx bytes=%u)",
				                  static_cast<unsigned long long>(p[4] | (uint64_t {p[5]} << 32u)),
				                  static_cast<unsigned long long>(p[2] | (uint64_t {p[3]} << 32u)), p[6] & 0x3ffffffu);
			break;
		case 0x37: // IT_WRITE_DATA: dst = p[2] | p[3] << 32
			if (len >= 4)
				n = std::snprintf(g_notes + g_notes_len, sizeof(g_notes) - g_notes_len, " write(dst=%llx dw=%u)",
				                  static_cast<unsigned long long>(p[2] | (uint64_t {p[3]} << 32u)), len - 4u);
			break;
		case 0x40: // IT_COPY_DATA: dst = p[4] | p[5] << 32
			if (len >= 6)
				n = std::snprintf(g_notes + g_notes_len, sizeof(g_notes) - g_notes_len, " copy(dst=%llx src=%llx)",
				                  static_cast<unsigned long long>(p[4] | (uint64_t {p[5]} << 32u)),
				                  static_cast<unsigned long long>(p[2] | (uint64_t {p[3]} << 32u)));
			break;
		default: return;
	}
	if (n > 0) g_notes_len += static_cast<uint32_t>(n);
	++g_notes_count;
}

// (punto 1) La imagen de profundidad de las cascadas (z_write del primer draw marcado) y quien la lee como T#: el
// render executor, al resolver un T# nuevo (RenderExecutor::ResolveTextureUncached), con el T# ya descodificado (host).
inline std::atomic<uint64_t> g_shadow_depth {0};
inline std::atomic<uint64_t> g_shadow_reads {0};
inline void NoteTextureRead(uint64_t address, const uint32_t* t, bool storage) {
	const uint64_t target = g_shadow_depth.load(std::memory_order_relaxed);
	if (target == 0 || address != target) return;
	const uint64_t n = g_shadow_reads.fetch_add(1, std::memory_order_relaxed);
	if (n >= 16) return;
	std::printf("[nativo-sombras-volcado] LEE la profundidad %llx como %s (lectura %llu): T# %08x %08x %08x %08x %08x %08x %08x %08x "
	            "(ancho=%u alto=%u tipo=%u capas %u..%u) pase_marcado=%u\n",
	            static_cast<unsigned long long>(address), storage ? "imagen de escritura" : "textura",
	            static_cast<unsigned long long>(n), t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7],
	            (((t[1] >> 30u) & 0x3u) | ((t[2] & 0xFFFu) << 2u)) + 1u, ((t[2] >> 14u) & 0x3FFFu) + 1u, (t[3] >> 28u) & 0xFu,
	            (t[4] >> 16u) & 0x1FFFu, t[4] & 0x1FFFu, g_gpu_done_mask.load(std::memory_order_relaxed));
	std::fflush(stdout);
}

// (hilo de la GPU, en CommandProcessor::DrawIndex) Tras una marca: una vez por pase. Sin leer memoria del juego.
inline void GpuDrawDump(uint64_t es, uint64_t ls, uint64_t hs, uint64_t gs, const uint32_t* hs_user, uint32_t hs_count,
                        const uint32_t* gs_user, uint32_t gs_count, uint64_t draw_index, const char* state) {
	const uint32_t pass = g_gpu_pending.exchange(0xFFFFFFFFu, std::memory_order_relaxed);
	if (pass > 3u) return;
	const uint32_t bit = 1u << pass;
	if ((g_gpu_done_mask.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return;
	const uint64_t game_index = g_gpu_pending_index.load(std::memory_order_relaxed);
	std::printf("[nativo-sombras-volcado] GPU pase=%u es=%llx ls=%llx hs=%llx gs=%llx | indice del draw=%llx del juego=%llx (%s) | %s | escrituras antes del draw=%u:%s\n",
	            pass, static_cast<unsigned long long>(es), static_cast<unsigned long long>(ls), static_cast<unsigned long long>(hs),
	            static_cast<unsigned long long>(gs), static_cast<unsigned long long>(draw_index),
	            static_cast<unsigned long long>(game_index),
	            draw_index == game_index ? "coincide" : (draw_index >= game_index && draw_index < game_index + 0x100000 ? "mismo bloque" : "NO coincide"),
	            state != nullptr ? state : "", g_notes_count, g_notes);
	g_notes[0] = '\0', g_notes_len = 0, g_notes_count = 0;
	if (g_req_ready.load(std::memory_order_acquire) == 0) { // (una peticion a la vez; si hay otra pendiente, solo registros)
		for (auto& p: g_req_ptr) p.store(0, std::memory_order_relaxed);
		uint32_t slot = 0;
		GpuUser("hs", hs_user, hs_count, slot);
		GpuUser("gs", gs_user, gs_count, slot);
		g_req_code[0].store(es, std::memory_order_relaxed);
		g_req_code[1].store(ls, std::memory_order_relaxed);
		g_req_code[2].store(hs, std::memory_order_relaxed);
		g_req_code[3].store(gs, std::memory_order_relaxed);
		g_req_ready.store(pass + 1u, std::memory_order_release);
	}
	std::fflush(stdout);
}

// (hilo del juego, al entrar en el gancho, sin candados) Lee lo que pidio la GPU: 256 bytes de cada puntero, 256 de los
// 8 punteros de su tabla, y el codigo de cada etapa hasta s_endpgm o 4 KiB.
inline void GuestServeRequests() {
	const uint32_t ready = g_req_ready.load(std::memory_order_acquire);
	if (ready == 0) return;
	static std::atomic<bool> busy {false};
	if (busy.exchange(true, std::memory_order_acquire)) return;
	std::printf("[nativo-sombras-volcado] GPU datos del pase=%u (leidos por un hilo del juego)\n", ready - 1u);
	char name[64];
	for (uint32_t i = 0; i < 16; ++i) {
		const uint64_t a = g_req_ptr[i].load(std::memory_order_relaxed);
		if (a == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(a, 256)) continue;
		std::snprintf(name, sizeof(name), "ptr[%u]=%llx", i, static_cast<unsigned long long>(a));
		GpuHex(name, a, 256);
		const auto* table = reinterpret_cast<const uint64_t*>(a);
		for (uint32_t k = 0; k < 8; ++k) {
			const uint64_t t = table[k] & 0xFFFFFFFFFFFFull;
			if (t < 0x100000ull || !Libs::Graphics::HostMemoryRangeIsReadable(t, 256)) continue;
			std::snprintf(name, sizeof(name), "ptr[%u] tabla[%u]=%llx", i, k, static_cast<unsigned long long>(t));
			GpuHex(name, t, 256);
		}
	}
	static const char* const stages[4] = {"es", "ls", "hs", "gs"};
	for (uint32_t i = 0; i < 4; ++i) GpuCode(stages[i], g_req_code[i].load(std::memory_order_relaxed));
	std::fflush(stdout);
	g_req_ready.store(0, std::memory_order_release);
	busy.store(false, std::memory_order_release);
}

// ======================================================================================================================
// (F5 pase 2, =3 y =1) Sombras en cascada NATIVAS. Ver notas-sombras-nativas.md.
//  - El gancho de 0xab57d0 (hilo del juego, sin candados) escribe en la DCB una marca de INICIO antes de los draws del
//    cubo y una de FIN despues; entre medias deja en una ranura fija (g_nat_slots, por numero de secuencia) los datos
//    de cada draw: V# de posicion (tabla[0]+0x10), indice (CMesh+8; n/inicio de item[4]+0x38/+0x3c), las 12 float del
//    objeto (V# de tabla[2]+0x10) y las 4 de tabla[3]+0x14; y la vista de la cascada (*(ctx+0x100), 0x720 bytes, leida
//    DENTRO del gancho: el bloque del monton del fotograma se reutiliza despues).
//  - El hilo de la GPU: en el primer draw tras la marca de inicio copia la capa emulada (antes) a la imagen A; en la
//    marca de fin copia la capa (despues) a B, dibuja los draws en A con el VS de §3 y compara A y B (compute).
//  - Nunca lee memoria del juego: solo la ranura. Cualquier fallo: ese cubo se queda emulado y se cuenta el motivo.
// ======================================================================================================================
// (KYTY_NATIVO_SOMBRAS_SKIN=1, apagado por defecto) Tambien las entradas con skinning: sus vertices son la salida del
// compute de skinning (Ghidra 0xa92a94: dispatch 0x1d0a900 por objeto; 0x9f80a0 elige el bufer del fotograma por su
// sello), que la GPU escribe; el hilo de render los obtiene con BufferCache::ObtainBuffer, como el draw emulado.
[[nodiscard]] inline bool NatSkinOn() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_NATIVO_SOMBRAS_SKIN");
		return text != nullptr && text[0] == '1' && text[1] == '\0';
	}();
	return on;
}
constexpr uint32_t NatMagic1 = 0x5654414eu; // "NATV" (en lugar de "MARC")
constexpr uint32_t NatBegin = 0u, NatEnd = 1u;
[[nodiscard]] inline bool IsNatMark(const uint32_t* p, uint32_t remaining) {
	return remaining >= MarkDwords && p[0] == MarkHeader && p[1] == MarkMagic0 && p[2] == NatMagic1;
}
struct NatDraw {
	uint64_t vbase = 0, ibase = 0;
	uint32_t skin = 0; // (KYTY_NATIVO_SOMBRAS_SKIN=1: vertices del bufer de skinning por fotograma, escrito por la GPU)
	uint64_t es = 0; // (direccion del ES que el gancho vio en la DCB para este draw (estado SH arrastrado); 0: ninguno)
	uint32_t vs = 0; // (0: VS estatico gpu_ds_sombras.vert; 1: VS con skinning 0x3866090d, gpu_ds_sombras_skin.vert)
	// (s10 heredado, 11-25) Desplazamiento SH (0x8C + n) del USER_DATA s10 cuando NO esta en el bloque de la DCB (Ghidra:
	// Agc no lo emite para estos draws; 0x914360 -> 0x1d0b070 lo pasa en el modificador del draw a
	// sceAgcDcbDrawIndexOffset y el HLE del emulador lo ignora, asi que la GPU usa el valor que ya tenia el registro).
	// obj = fila 0; el hilo de la GPU lee el registro real (gs_user_sgpr) y solo la dibuja si (s10 & 0xffff) == 0.
	// 0: s10 conocido (o no hace falta).
	uint32_t s10_reg = 0;
	uint32_t vstride = 0, vrecords = 0, icount = 0, istart = 0, index32 = 0, skip = 0; // (skip: 1 omitida (la dibuja solo el juego, motivo en g_nat_entry_why); 2 vacia)
	float    obj[12] {}; // (tabla[2]: s24..s31, s48..s51)
	float    t3[4] {};   // (tabla[3]+0x14: s36..s39)
};
constexpr uint32_t NatMaxDraws = 4096, NatSlots = 32, NatViewBytes = 0x720;
constexpr uint32_t NatSkipFiller = 0x80000000u; // (PM4 tipo 2: un dword de relleno)
constexpr uint32_t NatSkipMaxDw  = 0x4000u;     // (NOP tipo 3: cuenta de 14 bits -> hasta 0x4000 dwords cubiertos)
// (PM4 tipo 3: la cuenta es total - 2, como MarkHeader.) `dw` = dwords que cubre la NOP contando su cabecera; el que
// sigue es la cabecera del primer paquete del estado final. dw = skip_dw + 1 <= 0x4001 -> cuenta <= 0x3FFF (14 bits).
[[nodiscard]] constexpr uint32_t NatSkipNop(uint32_t dw) { return 0xC0000000u | ((dw - 2u) << 16u) | (0x10u << 8u); }
static_assert(((NatSkipNop(NatSkipMaxDw + 1u) >> 16u) & 0x3fffu) + 2u == NatSkipMaxDw + 1u, "NOP: cuenta de 14 bits");
static_assert(((NatSkipNop(2u) >> 16u) & 0x3fffu) + 2u == 2u, "NOP: total - 2");
struct NatSlot {
	std::atomic<uint32_t> state {0}; // (0 libre, 1 escribiendo, 2 lista, 3 en uso por la GPU)
	uint32_t              seq = 0, pass = 0, bucket = 0, count = 0, skipped = 0; // (skipped: entradas omitidas)
	uint32_t              sh_full = 0; // (1: el gancho entendio todo el bloque hasta el cubo: s10 sin escribir de verdad)
	uint64_t              block = 0; // (commands->bottom del bloque de la DCB del cubo: el s10 heredado es el mismo en un bloque)
	uint64_t              bias = 0; // (ctx+0xa0, solo informativo: el sesgo se toma de los registros del emulador)
	// (SALTO, =1) El dword reservado por el gancho justo despues de la marca de inicio (direccion en la DCB) y los dwords
	// del juego que siguen (los paquetes del cubo). Si el dibujo nativo del cubo entero sale bien en la marca de inicio,
	// el hilo de la GPU escribe ahi una cabecera NOP que los cubre: ProcessPm4 no traduce ni un paquete del cubo. Detras
	// de los paquetes del juego el gancho dejo el estado final (SET_SH/CONTEXT/UCONFIG_REG, INDEX_*), asi lo que sigue
	// ve los mismos registros. 0: sin salto posible (el relleno de 1 dword se queda: no hace nada).
	uint64_t              skip_word = 0;
	uint32_t              skip_dw   = 0;
	alignas(16) uint8_t   view[NatViewBytes] {};
	NatDraw               draws[NatMaxDraws];
};
inline NatSlot               g_nat_slots[NatSlots];
inline std::atomic<uint32_t> g_nat_seq {0};
// (hilo del juego) Una ranura para `seq`; nullptr: ocupada (este cubo se queda emulado).
[[nodiscard]] inline NatSlot* NatClaim(uint32_t seq) {
	NatSlot& s     = g_nat_slots[seq % NatSlots];
	uint32_t state = s.state.load(std::memory_order_acquire);
	if (state == 2u && seq - s.seq >= NatSlots && s.state.compare_exchange_strong(state, 1u)) return &s; // (vieja: nadie la uso)
	uint32_t expected = 0;
	return s.state.compare_exchange_strong(expected, 1u, std::memory_order_acq_rel) ? &s : nullptr;
}
// (hilo de la GPU) La ranura de `seq` si esta lista (pasa a "en uso"); nullptr si no.
[[nodiscard]] inline NatSlot* NatTake(uint32_t seq) {
	NatSlot& s        = g_nat_slots[seq % NatSlots];
	uint32_t expected = 2u;
	if (s.state.load(std::memory_order_acquire) != 2u || s.seq != seq) return nullptr;
	return s.state.compare_exchange_strong(expected, 3u, std::memory_order_acq_rel) ? &s : nullptr;
}
inline void NatRelease(NatSlot* s) {
	if (s != nullptr) s->state.store(0u, std::memory_order_release);
}
// Estado dinamico de Vulkan que el emulador resolvio para el ultimo draw de la cascada (renderDraw.cpp,
// ResolveGraphicsDynamicParams): el dibujo nativo usa exactamente los mismos valores. Solo con g_nat_want_dyn.
struct NatDyn {
	float    vp[6] {};  // (x, y, ancho, alto, min, max)
	int32_t  sc[4] {};  // (x, y, ancho, alto)
	uint32_t depth_test = 0, depth_write = 0, depth_compare = 0, bias_enable = 0;
	float    bias_constant = 0, bias_clamp = 0, bias_slope = 0;
};
// (renderDraw.cpp escribe el del draw que resuelve mientras g_nat_dyn_target = pase; puede ser el hilo grabador, mas
// tarde que la marca: se guarda por pase y vale el ultimo escrito, de este cubo o de uno anterior de la misma cascada.)
inline std::atomic<uint32_t> g_nat_dyn_target {0xFFu};
inline std::atomic<uint32_t> g_nat_dyn_mask {0}; // (bit por pase: ya hay uno)
inline NatDyn                g_nat_dyn_pass[4] {};
// Resultados por cascada (ventana de 10 s) y motivos de fallo.
inline std::atomic<bool> g_nat_proven[4] {};
// (SKIN) Las entradas con skinning solo se reemplazan en una cascada cuyos cubos con skinning comparados dieron >= 99,9 %.
inline std::atomic<bool> g_nat_skin_proven[4] {};
// (diagnostico, hilo de la GPU) El ES del draw del juego que se va a emparejar (graphicsRun, antes de DsNatBeforeDraw).
inline uint64_t g_nat_cur_es = 0; // (=1: la cascada dio >= 99,9 % en una ventana de =3)

// Motivos contados por texto (literales: el puntero los identifica). Acumulados en la sesion.
struct NatWhy {
	std::atomic<const char*> why {nullptr};
	std::atomic<uint64_t>    count {0};
};
constexpr uint32_t NatWhySlots = 24;
inline NatWhy      g_nat_entry_why[NatWhySlots]; // (entradas que no se dibujan en nativo; el cubo sigue)
inline NatWhy      g_nat_cube_why[NatWhySlots];  // (cubos que no llegan a la GPU)
inline NatWhy      g_nat_info_why[NatWhySlots];  // (solo informativo: no omite nada)
inline void NatCount(NatWhy* table, const char* why) {
	for (uint32_t i = 0; i < NatWhySlots; ++i) {
		const char* at = table[i].why.load(std::memory_order_relaxed);
		if (at == nullptr) {
			const char* expected = nullptr;
			if (!table[i].why.compare_exchange_strong(expected, why) && expected != why) continue;
			at = why;
		}
		if (at == why) {
			table[i].count.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
}
inline void NatCountN(NatWhy* table, const char* why, uint64_t n) {
	if (n == 0) return;
	for (uint32_t i = 0; i < NatWhySlots; ++i) {
		const char* at = table[i].why.load(std::memory_order_relaxed);
		if (at == nullptr) {
			const char* expected = nullptr;
			if (!table[i].why.compare_exchange_strong(expected, why) && expected != why) continue;
			at = why;
		}
		if (at == why) {
			table[i].count.fetch_add(n, std::memory_order_relaxed);
			return;
		}
	}
}
// (SALTO) Motivo con el opcode del primer paquete no admitido: un literal estable por opcode (NatCount usa el puntero).
[[nodiscard]] inline const char* NatOpWhy(uint32_t op) {
	static char names[256][56];
	static const bool init = [] {
		for (uint32_t i = 0; i < 256; ++i) std::snprintf(names[i], sizeof(names[i]), "salto: paquete no admitido 0x%02X (va draw a draw)", i);
		return true;
	}();
	(void)init;
	return names[op & 0xffu];
}
inline size_t NatFormatWhy(const NatWhy* table, char* out, size_t size) {
	size_t at = 0;
	out[0]    = '\0';
	for (uint32_t i = 0; i < NatWhySlots && at + 1 < size; ++i) {
		const char* why = table[i].why.load(std::memory_order_relaxed);
		if (why == nullptr) break;
		const int n = std::snprintf(out + at, size - at, "%s%s=%llu", at == 0 ? "" : ", ", why,
		                            static_cast<unsigned long long>(table[i].count.load(std::memory_order_relaxed)));
		if (n <= 0) break;
		at += static_cast<size_t>(n);
	}
	return at;
}
inline std::atomic<uint64_t> g_nat_cubes {0}, g_nat_cubes_full {0}; // (cubos capturados; sin ninguna entrada omitida)
inline void NatNoteCapture(const char* why) { NatCount(g_nat_cube_why, why); }

// (diagnostico, hilo de la GPU) Lo que hay de verdad en la DCB entre la marca de inicio de un cubo y la de fin con su
// numero: draws directos indexados (0x35/0x27), otros draws, INDIRECT_BUFFER (0x3f/0x33) y si la marca de fin esta en
// el mismo bufer. Lo lee DsNatMark en la marca de inicio.
// (KYTY_NATIVO_SOMBRAS_DIAG=1, apagado) Los volcados por cubo, las lineas "sin pareja" y el recuento de la DCB entre marcas.
// (KYTY_NATIVO_SOMBRAS_INPASS=1, apagado) =1: el dibujo nativo dentro del render pass abierto del emulador si es esa capa.
[[nodiscard]] inline bool NatInPass() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_NATIVO_SOMBRAS_INPASS");
		return text != nullptr && text[0] == '1';
	}();
	return on;
}
// (KYTY_NATIVO_SOMBRAS_CASCADA, ENCENDIDO por defecto con =1; =0 lo apaga) Un solo dibujo nativo (un render pass, un
// bind) por cascada (todos sus cubos), en la marca de inicio de su primer cubo (o en su primer draw); si cualquier cubo
// no esta listo, va por cubo.
[[nodiscard]] inline bool NatCascada() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_NATIVO_SOMBRAS_CASCADA");
		return !(text != nullptr && text[0] == '0');
	}();
	return on;
}
// (hilo de la GPU, =1 + CASCADA) Las marcas de inicio de la misma cascada que siguen a esta en el mismo bufer (hasta una
// marca de otra cascada o el final): sus numeros, en orden; [0] es la de esta marca.
struct NatGroupScan {
	uint32_t pass = 0, count = 0;
	uint32_t seq[8] {};
};
inline NatGroupScan g_nat_group {};
inline void NatScanGroup(const uint32_t* p, uint32_t remaining, uint32_t pass, uint32_t seq) {
	NatGroupScan r {};
	r.pass   = pass;
	r.seq[0] = seq;
	r.count  = 1;
	uint32_t at = MarkDwords, packets = 0;
	while (at < remaining && packets < (1u << 18u) && r.count < 8) {
		const uint32_t h = p[at];
		if ((h >> 30u) == 2u || h == 0x80000000u) { ++at; continue; }
		if ((h >> 30u) != 3u) break;
		const uint32_t len = ((h >> 16u) & 0x3fffu) + 2u;
		if (at + len > remaining) break;
		if (IsNatMark(p + at, remaining - at) && (p[at + 3] & 0xFFu) == NatBegin) {
			if (((p[at + 3] >> 8u) & 0xFFu) != pass) break;
			r.seq[r.count++] = p[at + 4];
		}
		++packets;
		at += len;
	}
	g_nat_group = r;
}
[[nodiscard]] inline bool NatDiag() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_NATIVO_SOMBRAS_DIAG");
		return text != nullptr && text[0] == '1';
	}();
	return on;
}
struct NatDcbScan {
	uint32_t direct = 0, other_draws = 0, ib = 0, packets = 0;
	bool     end_found = false;
};
inline NatDcbScan g_nat_scan {};
// (diagnostico) Draws directos indexados que el recorrido vio en una pasada de estado, en una traduccion especulativa o
// en otro hilo (donde DsNatBeforeDraw no se llama), acumulado.
inline std::atomic<uint64_t> g_nat_draws_elsewhere[3] {};
inline void NatScanDcb(const uint32_t* p, uint32_t remaining, uint32_t seq) {
	NatDcbScan r {};
	uint32_t   at = MarkDwords;
	while (at < remaining && r.packets < (1u << 20u)) {
		const uint32_t h = p[at];
		if ((h >> 30u) == 2u || h == 0x80000000u) { ++at; continue; }
		if ((h >> 30u) != 3u) break;
		const uint32_t len = ((h >> 16u) & 0x3fffu) + 2u;
		if (at + len > remaining) break;
		const uint32_t op = (h >> 8u) & 0xffu;
		if (IsNatMark(p + at, remaining - at) && p[at + 4] == seq && (p[at + 3] & 0xFFu) == NatEnd) {
			r.end_found = true;
			break;
		}
		if (op == 0x35 || op == 0x27) ++r.direct;
		else if (op == 0x24 || op == 0x25 || op == 0x2d || op == 0x38 || op == 0x2c) ++r.other_draws;
		else if (op == 0x3f || op == 0x33) ++r.ib;
		++r.packets;
		at += len;
	}
	g_nat_scan = r;
}

} // namespace DsNativoSombras
