// (F5 pase 2, pieza 1) KYTY_NATIVO_PASE_SOMBRAS=2 (apagado por defecto; local/ds-nativo-sombras.h). Incluido por
// demonsSoulsCopy.cpp dentro de su espacio anonimo, despues de demonsSoulsNativoPase.inc (AgcCommandBuffer, Port2).
//
// Gancho en la entrada de 01.007.000 eboot+0xab57d0 (renderer, lista, ctx, cubo, pase): la funcion que recorre un cubo
// de la lista visible y emite sus draws. Con pase >= 0 (sombras: cascadas 0..3 y sus extra) cuenta la lista y, tras
// ejecutar la funcion entera del juego por un trampolin, los draws que escribio en la DCB. No cambia nada.
// Si KYTY_PORT_R / KYTY_PORTG ya enganchan esta funcion, los bytes no coinciden y este gancho no se instala.

using NativoSombrasFunction = KYTY_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t*, int32_t, int32_t);
NativoSombrasFunction g_nativo_sombras_original   = nullptr;
uint64_t              g_nativo_sombras_trampoline = 0;

constexpr uint64_t NativoSombrasOffset = 0xab57d0;
// 0xab57d0..+0x20 de 01.007.000 (projelf_par1): push rbp; mov rbp, rsp; push r15..r12; push rbx; sub rsp, 0x728; mov rax, [rip+..]
constexpr std::array<uint8_t, 32> NativoSombrasCode {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41,
                                                     0x54, 0x53, 0x48, 0x81, 0xec, 0x28, 0x07, 0x00, 0x00, 0x48, 0x8b,
                                                     0x05, 0xd5, 0xca, 0xc1, 0x01, 0x44, 0x89, 0x85, 0xdc, 0xf8};
constexpr size_t NativoSombrasMoved = 12; // (los 6 push / mov: sin dependencias de posicion)

// Los draws (IT_DRAW_*) de los paquetes tipo 3 en [begin, end).
uint64_t NativoSombrasCountDraws(const uint32_t* begin, const uint32_t* end) {
	uint64_t draws = 0;
	for (const uint32_t* p = begin; p < end;) {
		const uint32_t header = *p;
		const uint32_t type   = header >> 30u;
		if (type == 2u) { ++p; continue; }
		if (type != 3u) return draws;
		const uint32_t length = ((header >> 16u) & 0x3fffu) + 2u;
		switch ((header >> 8u) & 0xffu) {
			case 0x24: case 0x25: case 0x27: case 0x2d: case 0x35: case 0x38: ++draws; break;
			default: break;
		}
		p += length;
	}
	return draws;
}

// (volcado, =2) Unas pocas veces: el V# de vertices de un objeto estatico (CMesh+0x238+0x58, lo que 0x9f80a0 deja en
// ctx+0xb0), los primeros 256 bytes de sus constantes (*(sub+0x40) -> +8, 48 bits; 0x9f6c10 lo deja en ctx+0xe0) y el
// valor de la cascada (ctx[0x14] = *(param+0x250+i*8) en 0xa11220) con 64 bytes si es un puntero legible.
void NativoSombrasHex(const char* what, uint64_t addr, uint64_t bytes) {
	if (addr == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(addr, bytes)) {
		std::printf("[nativo-sombras-volcado]   %s @%llx: no legible\n", what, static_cast<unsigned long long>(addr));
		return;
	}
	const auto* f = reinterpret_cast<const float*>(addr);
	const auto* w = reinterpret_cast<const uint32_t*>(addr);
	for (uint64_t i = 0; i < bytes / 4; i += 4)
		std::printf("[nativo-sombras-volcado]   %s +%03llx: %08x %08x %08x %08x | %g %g %g %g\n", what,
		            static_cast<unsigned long long>(i * 4), w[i], w[i + 1], w[i + 2], w[i + 3], double(f[i]), double(f[i + 1]),
		            double(f[i + 2]), double(f[i + 3]));
}
void NativoSombrasVsharp(const char* what, uint64_t addr) {
	if (addr == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(addr, 16)) {
		std::printf("[nativo-sombras-volcado]   %s @%llx: no legible\n", what, static_cast<unsigned long long>(addr));
		return;
	}
	const auto* w = reinterpret_cast<const uint32_t*>(addr);
	std::printf("[nativo-sombras-volcado]   %s @%llx como V#: base=%llx stride=%u num_records=%u w3=%08x (formato=%u dst_sel=%03x) "
	            "crudo %08x %08x %08x %08x\n",
	            what, static_cast<unsigned long long>(addr),
	            static_cast<unsigned long long>(uint64_t {w[0]} | (uint64_t {w[1] & 0xFFFFu} << 32u)), (w[1] >> 16u) & 0x3FFFu, w[2],
	            w[3], (w[3] >> 12u) & 0x7Fu, w[3] & 0xFFFu, w[0], w[1], w[2], w[3]);
}
void NativoSombrasDump(const uint64_t* ctx, int32_t pass, uint64_t inst, uint64_t sub, uint64_t mesh, uint64_t block, const void* gpu) {
	static std::atomic<uint32_t> dumps {0};
	if (dumps.fetch_add(1, std::memory_order_relaxed) >= 2) return;
	std::printf("[nativo-sombras-volcado] pase=%d submalla=%llx CMesh=%llx bloque_v=%llx vkbuffer=%p\n", pass,
	            static_cast<unsigned long long>(sub), static_cast<unsigned long long>(mesh), static_cast<unsigned long long>(block), gpu);
	const uint64_t v = *reinterpret_cast<const uint64_t*>(mesh + 0x238 + 0x58);
	std::printf("[nativo-sombras-volcado]   CMesh+0x238+0x58 = %llx\n", static_cast<unsigned long long>(v));
	NativoSombrasVsharp("CMesh+0x290 (V# en linea)", mesh + 0x238 + 0x58);
	NativoSombrasVsharp("*(CMesh+0x290)", v);
	NativoSombrasHex("CMesh+0x238", mesh + 0x238, 0xb0);
	// (volcado 2) La tabla de V# a la que apunta CMesh+0x290 (el 0 es el indice: base = CMesh+0x2a0, stride 2): las 14.
	for (uint64_t k = 0; k < 14 && v != 0; ++k) {
		char name[32];
		std::snprintf(name, sizeof(name), "tabla V#[%llu]", static_cast<unsigned long long>(k));
		NativoSombrasVsharp(name, v + k * 16);
	}
	// sub+0x40 es 0 (constantes globales): la matriz del objeto esta en la instancia o en la submalla.
	NativoSombrasHex("instancia", inst, 0x180);
	NativoSombrasHex("submalla", sub, 0x48);
	NativoSombrasHex("ctx", reinterpret_cast<uint64_t>(ctx), 0x100);
	NativoSombrasVsharp("CBuffer0 V# (CMesh+8)", mesh + 8);
	const uint64_t cobj = *reinterpret_cast<const uint64_t*>(sub + 0x40);
	const uint64_t cadr = cobj != 0 && Libs::Graphics::HostMemoryRangeIsReadable(cobj + 8, 8)
	                          ? (*reinterpret_cast<const uint64_t*>(cobj + 8) & 0xFFFFFFFFFFFFull) : 0;
	std::printf("[nativo-sombras-volcado]   constantes: sub+0x40=%llx -> %llx\n", static_cast<unsigned long long>(cobj),
	            static_cast<unsigned long long>(cadr));
	NativoSombrasHex("constantes", cadr, 256);
	const uint64_t cas = ctx[0x14];
	std::printf("[nativo-sombras-volcado]   cascada: ctx[0x14]=%llx ctx[0xa]=%llx ctx[0x15]=%d\n", static_cast<unsigned long long>(cas),
	            static_cast<unsigned long long>(ctx[0xa]), static_cast<int>(static_cast<int32_t>(ctx[0x15])));
	NativoSombrasHex("cascada", cas, 64);
	std::fflush(stdout);
}

uint64_t g_nativo_sombras_base = 0; // (base del eboot, para las constantes globales *(eboot+0x2f0a5f8)+0x580)

// (volcado 3) Hasta 6 objetos estaticos distintos cuya instancia no tenga la identidad en +0x38..+0x74.
void NativoSombrasDumpObject(int32_t pass, uint64_t inst, uint64_t mesh) {
	static std::atomic<uint32_t> done {0};
	static std::atomic<uint64_t> seen[6] {};
	if (done.load(std::memory_order_relaxed) >= 6 || inst == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(inst, 0x180)) return;
	// (3x3 en +0x38..+0x58, 9 float seguidos; diagonal en 0, 4, 8) Solo con rotacion de verdad: |fuera de diagonal| > 0.01.
	float m[9];
	std::memcpy(m, reinterpret_cast<const void*>(inst + 0x38), sizeof(m));
	bool rotated = false;
	for (int i = 0; i < 9; ++i)
		if (i != 0 && i != 4 && i != 8 && (m[i] > 0.01f || m[i] < -0.01f)) rotated = true;
	if (!rotated) return;
	for (auto& k: seen)
		if (k.load(std::memory_order_relaxed) == inst) return;
	const uint32_t slot = done.fetch_add(1, std::memory_order_relaxed);
	if (slot >= 6) return;
	seen[slot].store(inst, std::memory_order_relaxed);
	std::printf("[nativo-sombras-volcado] objeto %u: pase=%d instancia=%llx CMesh=%llx\n", slot, pass,
	            static_cast<unsigned long long>(inst), static_cast<unsigned long long>(mesh));
	NativoSombrasHex("objeto", inst, 0x180);
	std::fflush(stdout);
}

// (volcado 3) Una vez por cascada (pase 0..3): 256 bytes en ctx+0xb8, +0xd0, +0xf0 y en las constantes globales.
void NativoSombrasDumpCascade(const uint64_t* ctx, int32_t pass) {
	static std::atomic<uint32_t> done_mask {0};
	if (pass < 0 || pass > 3) return;
	const uint32_t bit = 1u << static_cast<uint32_t>(pass);
	if ((done_mask.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return;
	std::printf("[nativo-sombras-volcado] cascada pase=%d ctx+0xb8=%llx ctx+0xd0=%llx ctx+0xf0=%llx ctx+0xa0=%llx\n", pass,
	            static_cast<unsigned long long>(ctx[0xb8 / 8]), static_cast<unsigned long long>(ctx[0xd0 / 8]),
	            static_cast<unsigned long long>(ctx[0xf0 / 8]), static_cast<unsigned long long>(ctx[0xa0 / 8]));
	// (VS de profundidad, desensamblado 11-24: lee de este bufer (tabla[1]) +0x0, +0x30, +0x1c0, +0x1d0..+0x20f, +0x2d0..+0x2db,
	// +0x6e0..+0x70f: 0x720 bytes por pase, para ver si cambian entre cascadas)
	NativoSombrasHex("ctx+0xb8", ctx[0xb8 / 8] & 0xFFFFFFFFFFFFull, 0x720);
	NativoSombrasHex("ctx+0xd0", ctx[0xd0 / 8] & 0xFFFFFFFFFFFFull, 256);
	NativoSombrasHex("ctx+0xf0", ctx[0xf0 / 8] & 0xFFFFFFFFFFFFull, 256);
	uint64_t gaddr = 0;
	const uint64_t gvar = g_nativo_sombras_base + 0x2f0a5f8;
	if (g_nativo_sombras_base != 0 && Libs::Graphics::HostMemoryRangeIsReadable(gvar, 8)) {
		const uint64_t g = *reinterpret_cast<const uint64_t*>(gvar);
		if (g != 0 && Libs::Graphics::HostMemoryRangeIsReadable(g + 0x580, 8)) {
			const uint64_t obj = *reinterpret_cast<const uint64_t*>(g + 0x580);
			if (obj != 0 && Libs::Graphics::HostMemoryRangeIsReadable(obj + 8, 8))
				gaddr = *reinterpret_cast<const uint64_t*>(obj + 8) & 0xFFFFFFFFFFFFull;
			std::printf("[nativo-sombras-volcado]   globales: *(eboot+0x2f0a5f8)=%llx +0x580 -> %llx -> %llx\n",
			            static_cast<unsigned long long>(g), static_cast<unsigned long long>(obj), static_cast<unsigned long long>(gaddr));
		}
	}
	NativoSombrasHex("globales", gaddr, 256);
	// (volcado 4) 0xa13e60 reserva por cascada tres bufers de 0x40 bytes (0x902350(.., 0x40, 0/1/2, 0)) y deja sus
	// descriptores (16 bytes) en (*ctx)+0x1d8 / +0x1e8 / +0x1f8: candidatos a las matrices de la cascada (4x4 = 0x40).
	const uint64_t c0 = ctx[0];
	for (uint64_t k = 0; k < 3 && c0 != 0; ++k) {
		const uint64_t at = c0 + 0x1d8 + k * 0x10;
		char           name[40];
		std::snprintf(name, sizeof(name), "(*ctx)+0x%llx", static_cast<unsigned long long>(0x1d8 + k * 0x10));
		NativoSombrasHex(name, at, 16);
		const uint64_t base = Libs::Graphics::HostMemoryRangeIsReadable(at, 8)
		                          ? (*reinterpret_cast<const uint64_t*>(at) & 0xFFFFFFFFFFFFull) : 0;
		std::snprintf(name, sizeof(name), "bufer%llu", static_cast<unsigned long long>(k));
		NativoSombrasHex(name, base, 0x40);
	}
	std::fflush(stdout);
}

// (11-24, estado GPU: las 4 cascadas = misma imagen 2826f0000 1024x1024, mismo viewport, capa 0..3; el VS lee la vista
// del bufer de la camara, tabla[1], siempre en la misma direccion) Despues de que el juego emita los draws de una
// cascada, en UNA linea: los float del bufer de la camara que lee el VS (+0x0..+0x3f, +0x1c0..+0x20f, +0x2d0..+0x2df,
// +0x6e0..+0x70f). Una vez por pase y cubo; lo lee el hilo del juego sin candados.
void NativoSombrasDumpCameraAfter(const uint64_t* ctx, int32_t pass, int32_t bucket) {
	static std::atomic<uint32_t> done_mask {0};
	if (pass < 0 || pass > 3 || bucket < 0 || bucket > 7) return;
	const uint32_t bit = 1u << static_cast<uint32_t>(pass * 8 + bucket);
	if ((done_mask.load(std::memory_order_relaxed) & bit) != 0) return;
	const uint64_t cam = ctx[0xb8 / 8] & 0xFFFFFFFFFFFFull;
	if (cam == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(cam, 0x710)) return;
	if ((done_mask.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return;
	static const uint32_t ranges[4][2] = {{0x0, 0x40}, {0x1c0, 0x210}, {0x2d0, 0x2e0}, {0x6e0, 0x710}};
	char   line[2048];
	size_t at = static_cast<size_t>(std::snprintf(line, sizeof(line), "[nativo-sombras-volcado] camara DESPUES pase=%d cubo=%d @%llx:",
	                                              pass, bucket, static_cast<unsigned long long>(cam)));
	for (const auto& r: ranges) {
		if (at < sizeof(line)) at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " +%03x:", r[0]));
		for (uint32_t o = r[0]; o < r[1] && at < sizeof(line); o += 4) {
			float f = 0;
			std::memcpy(&f, reinterpret_cast<const void*>(cam + o), sizeof(f));
			at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " %g", double(f)));
		}
	}
	std::printf("%s\n", line);
	std::fflush(stdout);
}

// (opcion 2, 11-24) Despues de que el juego emita los draws de una cascada, en el hilo del juego y sin candados: el
// SET_SH_REG (0x76) de la DCB de esta llamada que carga el puntero de USER_DATA del VS (s[8:9], la tabla de 0x9819d0),
// el ultimo antes del primer draw. Se reconoce porque su entrada [1] es el bufer de la camara (ctx+0xb8). De ahi, en
// UNA linea por pase: la tabla (8 punteros), tabla[1] en los offsets que lee el VS y si difiere de la camara leida antes
// de los draws, tabla[3]+0x14 (4 float) y las 12 float del primer objeto (V# de tabla[2] +0x10 -> base).
void NativoSombrasDumpUserTable(const uint64_t* ctx, int32_t pass, const uint32_t* begin, const uint32_t* end,
                                const uint8_t* cam_before) {
	static std::atomic<uint32_t> done_mask {0};
	if (pass < 0 || pass > 3 || begin == nullptr || end <= begin) return;
	const uint32_t bit = 1u << static_cast<uint32_t>(pass);
	if ((done_mask.load(std::memory_order_relaxed) & bit) != 0) return;
	const uint64_t cam = ctx[0xb8 / 8] & 0xFFFFFFFFFFFFull;
	uint64_t       table = 0;
	uint32_t       reg   = 0;
	for (const uint32_t* p = begin; p < end;) {
		const uint32_t h = *p;
		if ((h >> 30u) == 2u) { ++p; continue; }
		if ((h >> 30u) != 3u) break;
		const uint32_t len = ((h >> 16u) & 0x3fffu) + 2u;
		const uint32_t op  = (h >> 8u) & 0xffu;
		if (op == 0x24 || op == 0x25 || op == 0x27 || op == 0x2d || op == 0x35 || op == 0x38) break; // (primer draw)
		if (op == 0x76 && len >= 4 && p + len <= end) {
			for (uint32_t i = 2; i + 1 < len; ++i) {
				const uint64_t a = uint64_t {p[i]} | (uint64_t {p[i + 1] & 0xFFFFu} << 32u);
				if ((p[i + 1] >> 16u) != 0 || a < 0x100000ull || (a & 7u) != 0 ||
				    !Libs::Graphics::HostMemoryRangeIsReadable(a, 64))
					continue;
				const uint64_t e1 = reinterpret_cast<const uint64_t*>(a)[1] & 0xFFFFFFFFFFFFull;
				if (e1 == cam && cam != 0) {
					table = a;
					reg   = p[1] + (i - 2);
				}
			}
		}
		p += len;
	}
	if (table == 0) return;
	if ((done_mask.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return;
	const auto* t = reinterpret_cast<const uint64_t*>(table);
	char   line[3072];
	size_t at = static_cast<size_t>(std::snprintf(line, sizeof(line), "[nativo-sombras-volcado] TABLA VS pase=%d reg=0x%x @%llx:", pass, reg,
	                                              static_cast<unsigned long long>(table)));
	for (int k = 0; k < 8 && at < sizeof(line); ++k)
		at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " [%d]=%llx", k,
		                                        static_cast<unsigned long long>(t[k] & 0xFFFFFFFFFFFFull)));
	const uint64_t t1 = t[1] & 0xFFFFFFFFFFFFull;
	if (t1 != 0 && Libs::Graphics::HostMemoryRangeIsReadable(t1, 0x720)) {
		const bool same = cam_before != nullptr && std::memcmp(cam_before, reinterpret_cast<const void*>(t1), 0x720) == 0;
		if (at < sizeof(line))
			at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " | tabla[1] %s la camara de antes:",
			                                        same ? "IGUAL que" : "DISTINTA de"));
		static const uint32_t ranges[4][2] = {{0x0, 0x40}, {0x1c0, 0x210}, {0x2d0, 0x2e0}, {0x6e0, 0x710}};
		for (const auto& r: ranges) {
			if (at < sizeof(line)) at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " +%03x:", r[0]));
			for (uint32_t o = r[0]; o < r[1] && at < sizeof(line); o += 4) {
				float f = 0;
				std::memcpy(&f, reinterpret_cast<const void*>(t1 + o), sizeof(f));
				at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " %g", double(f)));
			}
		}
	}
	const uint64_t t3 = t[3] & 0xFFFFFFFFFFFFull;
	if (t3 != 0 && Libs::Graphics::HostMemoryRangeIsReadable(t3 + 0x14, 16) && at < sizeof(line)) {
		float f[4];
		std::memcpy(f, reinterpret_cast<const void*>(t3 + 0x14), sizeof(f));
		at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " | tabla[3]+0x14: %g %g %g %g", double(f[0]),
		                                        double(f[1]), double(f[2]), double(f[3])));
	}
	const uint64_t t2 = t[2] & 0xFFFFFFFFFFFFull;
	if (t2 != 0 && Libs::Graphics::HostMemoryRangeIsReadable(t2 + 0x10, 16)) {
		const auto*    v    = reinterpret_cast<const uint32_t*>(t2 + 0x10);
		const uint64_t base = uint64_t {v[0]} | (uint64_t {v[1] & 0xFFFFu} << 32u);
		if (base != 0 && Libs::Graphics::HostMemoryRangeIsReadable(base, 0x30) && at < sizeof(line)) {
			at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " | objeto0 @%llx:", static_cast<unsigned long long>(base)));
			for (uint32_t o = 0; o < 0x30 && at < sizeof(line); o += 4) {
				float f = 0;
				std::memcpy(&f, reinterpret_cast<const void*>(base + o), sizeof(f));
				at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " %g", double(f)));
			}
		}
	}
	std::printf("%s\n", line);
	std::fflush(stdout);
}

// (GHIDRA 11-24) 0xa13e60 empieza con 0xa00eb0(ctx): 0x9098d0(*(eboot+0x2f0a5f8)+0x650, ..) reserva un bloque NUEVO del
// monton del fotograma (0x7f9cf0), lo pone como base del CBuffer de vista global (+0x650: su V#, 48 bits) y lo devuelve;
// 0xa00eb0 lo rellena con la vista de la cascada desde ctx[0xa] (render+0xdf0+i*8): +0x1c0, +0x2dc..+0x314, +0x6c0..+0x6d0,
// +0x710 (la misma disposicion que el bufer de la camara que lee el VS). Una linea por pase, despues de los draws, en el
// hilo del juego y sin candados: los mismos offsets que lee el VS, leidos de ese bloque.
void NativoSombrasDumpCascadeView(int32_t pass) {
	static std::atomic<uint32_t> done_mask {0};
	if (pass < 0 || pass > 3 || g_nativo_sombras_base == 0) return;
	const uint32_t bit = 1u << static_cast<uint32_t>(pass);
	if ((done_mask.load(std::memory_order_relaxed) & bit) != 0) return;
	const uint64_t gvar = g_nativo_sombras_base + 0x2f0a5f8;
	if (!Libs::Graphics::HostMemoryRangeIsReadable(gvar, 8)) return;
	const uint64_t g = *reinterpret_cast<const uint64_t*>(gvar);
	if (g == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(g + 0x650, 8)) return;
	const uint64_t view = *reinterpret_cast<const uint64_t*>(g + 0x650) & 0xFFFFFFFFFFFFull;
	if (view == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(view, 0x720)) return;
	if ((done_mask.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return;
	static const uint32_t ranges[5][2] = {{0x0, 0x40}, {0x1c0, 0x210}, {0x2d0, 0x2e0}, {0x6e0, 0x710}, {0x710, 0x720}};
	char   line[2048];
	size_t at = static_cast<size_t>(std::snprintf(line, sizeof(line), "[nativo-sombras-volcado] VISTA CASCADA pase=%d @%llx:", pass,
	                                              static_cast<unsigned long long>(view)));
	for (const auto& r: ranges) {
		if (at < sizeof(line)) at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " +%03x:", r[0]));
		for (uint32_t o = r[0]; o < r[1] && at < sizeof(line); o += 4) {
			float f = 0;
			std::memcpy(&f, reinterpret_cast<const void*>(view + o), sizeof(f));
			at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, " %g", double(f)));
		}
	}
	std::printf("%s\n", line);
	std::fflush(stdout);
}

// (F5 pase 2, =3 y =1; local/ds-nativo-sombras.h) Una marca NATV (6 dwords) en la DCB: tipo (inicio/fin) | pase<<8, seq.
bool NativoSombrasNatMark(AgcCommandBuffer* commands, uint32_t kind, int32_t pass, uint32_t seq) {
	namespace H = DsNativoSombras;
	uint32_t* const up   = commands->cursor_up;
	uint32_t* const down = commands->cursor_down;
	if (up == nullptr || down == nullptr || reinterpret_cast<uintptr_t>(down) <= reinterpret_cast<uintptr_t>(up)) return false;
	const uint64_t available = (reinterpret_cast<uintptr_t>(down) - reinterpret_cast<uintptr_t>(up)) / sizeof(uint32_t);
	if (available <= commands->reserved_dw || available - commands->reserved_dw < H::MarkDwords) return false;
	up[0]               = H::MarkHeader;
	up[1]               = H::MarkMagic0;
	up[2]               = H::NatMagic1;
	up[3]               = kind | (static_cast<uint32_t>(pass) << 8u);
	up[4]               = seq;
	up[5]               = 0;
	commands->cursor_up = up + H::MarkDwords;
	return true;
}

// (SKIN) El VS de profundidad de las entradas con skinning (hash 0x3866090d8246e47e, 33 slots, 3 buferes): sus 95 dwords
// tal cual, del volcado del log 20261125-053825-022 (ES @386610000). gpu_ds_sombras_skin.vert lo reproduce op a op.
constexpr uint32_t kNativoSombrasSkinVs[95] = {
	0xbfa00003u, 0x93ebff03u, 0x00080008u, 0x8700ff03u, 0x000000ffu, 0x8f6a8c6bu, 0x887c6a00u, 0xbf800000u,
	0xbf900009u, 0x81ea6bc0u, 0x90fe6ac1u, 0xf8000941u, 0x00000000u, 0x81ea00c0u, 0xbf8cff0fu, 0x90fe6ac1u,
	0x4a0810f9u, 0x0684060au, 0xf4041a84u, 0xfa000010u, 0xf4080404u, 0xfa000000u, 0xbf8cc07fu, 0xf4080535u,
	0xfa000010u, 0xf4080608u, 0xfa000010u, 0xf4080708u, 0xfa000040u, 0xf4100009u, 0xfa0000c0u, 0xf4040409u,
	0xfa0001c0u, 0xf4040809u, 0xfa0002d0u, 0xf4001a89u, 0xfa0002d8u, 0xbf8cc07fu, 0xe0382020u, 0x80050a04u,
	0xe0382010u, 0x80050004u, 0xe0382000u, 0x80050604u, 0xe0082000u, 0x80060e05u, 0xe0042000u, 0x80070405u,
	0xbf8c3f71u, 0x3e1a1910u, 0x3e060510u, 0x3e121110u, 0xbf8c3f70u, 0xd52f0002u, 0x00010105u, 0x3e1a170fu,
	0xd541000bu, 0x040e1f01u, 0x3e120f0fu, 0x3e1a150eu, 0x3e16010eu, 0xd5410003u, 0x04261d06u, 0xd5410006u,
	0x000e1a02u, 0xd5410008u, 0x003e1a0eu, 0xd5410000u, 0x001e1a06u, 0xd5410009u, 0x002e1a0au, 0x0a021a6au,
	0xd5410007u, 0x041a1601u, 0xd5410006u, 0x0422160du, 0x3e001605u, 0x3e121609u, 0x0a0a0620u, 0x3e0e0600u,
	0x3e0c060cu, 0xd5410008u, 0x04020604u, 0x3e120608u, 0x0a001621u, 0x5e020901u, 0xd5410004u, 0x041c2106u,
	0x3e100c11u, 0x5e000105u, 0xf80008cfu, 0x06090804u, 0xf8000207u, 0x00020100u, 0xbf810000u,
};
// (SKIN2, 11-27) Los otros VS de skinning de profundidad (hash del emulador 0xcd218bf8feea0f5b y familia; "otro VS" =
// 1,52 M entradas en el log 20261127-041016-200), volcados con KYTY_NATIVO_SOMBRAS_DIAG=1 en 20261127-042910-357 y
// desensamblados (llvm-mc gfx1013): misma posicion que kNativoSombrasSkinVs, op a op y en el mismo orden de redondeo
// (filas del objeto de tabla[2]+0x10 idxen (s10 & 0xffff) + instance_id en +0/+16/+32; tabla[0]+0x10 format_xyz;
// 4x4 de tabla[1]+0xc0; desplazamiento por w de tabla[1]+0x1c0; z exportado). Solo cambian los param (UV) que no
// tocan la profundidad. Asi que gpu_ds_sombras_skin.vert (vs = 1) da el mismo pos0 bit a bit.
constexpr uint32_t kNativoSombrasSkinVs2[75] = { // (ES @26bc8c000, log 20261127-042910-357)
	0xbfa00003u, 0x93ebff03u, 0x00080008u, 0x8700ff03u, 0x000000ffu, 0x8f6a8c6bu, 0x887c6a00u, 0xbf800000u,
	0xbf900009u, 0x81ea6bc0u, 0x90fe6ac1u, 0xf8000941u, 0x00000000u, 0x81ea00c0u, 0xbf8cff0fu, 0x90fe6ac1u,
	0x4a0810f9u, 0x0684060au, 0xf4041a84u, 0xfa000010u, 0xf4080404u, 0xfa000000u, 0xbf8cc07fu, 0xf4080535u,
	0xfa000010u, 0xf4080608u, 0xfa000010u, 0xf4100009u, 0xfa0000c0u, 0xf4041a89u, 0xfa0001c0u, 0xbf8cc07fu,
	0xe0382020u, 0x80050a04u, 0xe0382010u, 0x80050004u, 0xe0382000u, 0x80050604u, 0xe0082000u, 0x80060e05u,
	0xbf8c3f70u, 0x3e1a1910u, 0xd5410004u, 0x040e2102u, 0x3e121110u, 0x3e1a170fu, 0x3e08030fu, 0x3e120f0fu,
	0x3e1a150eu, 0xd5410001u, 0x04121d00u, 0x3e120d0eu, 0xd5410004u, 0x000e1a02u, 0xd5410005u, 0x003e1a0eu,
	0xd5410003u, 0x001e1a06u, 0xd5410002u, 0x002e1a0au, 0x3e080201u, 0x3e0a020du, 0x3e060205u, 0xd5410000u,
	0x040a0209u, 0xd5410001u, 0x04121200u, 0x3e0a120cu, 0x3e061204u, 0x3e001208u, 0x3e020a6au, 0x3e060a6bu,
	0xf80008cfu, 0x05000301u, 0xbf810000u,
};
constexpr uint32_t kNativoSombrasSkinVs3[82] = { // (ES @26cab8900, log 20261127-042910-357)
	0xbfa00003u, 0x93ebff03u, 0x00080008u, 0x8700ff03u, 0x000000ffu, 0x8f6a8c6bu, 0x887c6a00u, 0xbf800000u,
	0xbf900009u, 0x81ea6bc0u, 0x90fe6ac1u, 0xf8000941u, 0x00000000u, 0x81ea00c0u, 0xbf8cff0fu, 0x90fe6ac1u,
	0x4a0810f9u, 0x0684060au, 0xf4041a84u, 0xfa000010u, 0xf4080404u, 0xfa000000u, 0xbf8cc07fu, 0xf4080535u,
	0xfa000010u, 0xf4080608u, 0xfa000010u, 0xf4080708u, 0xfa000040u, 0xf4100009u, 0xfa0000c0u, 0xf4041a89u,
	0xfa0001c0u, 0xbf8cc07fu, 0xe0382020u, 0x80050a04u, 0xe0382010u, 0x80050004u, 0xe0382000u, 0x80050604u,
	0xe0082000u, 0x80060e05u, 0xe0042000u, 0x80070405u, 0xbf8c3f71u, 0x3e1a1910u, 0x3e060510u, 0x3e121110u,
	0xbf8c3f70u, 0x5e040b04u, 0x3e1a170fu, 0xd5410001u, 0x040e1f01u, 0x3e120f0fu, 0x3e1a150eu, 0x3e02010eu,
	0x3e120d0eu, 0xd5410005u, 0x000e1a02u, 0xd5410006u, 0x003e1a0eu, 0xd5410004u, 0x001e1a06u, 0xd5410003u,
	0x002e1a0au, 0x3e0a0201u, 0x3e0c020du, 0x3e080205u, 0xd5410000u, 0x040e0209u, 0xd5410001u, 0x04161200u,
	0x3e0c120cu, 0x3e081204u, 0x3e001208u, 0x3e020c6au, 0x3e080c6bu, 0xf80008cfu, 0x06000401u, 0xf8000201u,
	0x00000002u, 0xbf810000u,
};
constexpr uint32_t kNativoSombrasSkinVs4[93] = { // (ES @388468c00, log 20261127-042910-357)
	0xbfa00003u, 0x93ebff03u, 0x00080008u, 0x8700ff03u, 0x000000ffu, 0x8f6a8c6bu, 0x887c6a00u, 0xbf800000u,
	0xbf900009u, 0x81ea6bc0u, 0x90fe6ac1u, 0xf8000941u, 0x00000000u, 0x81ea00c0u, 0xbf8cff0fu, 0x90fe6ac1u,
	0x4a2210f9u, 0x0684060au, 0xf4041a84u, 0xfa000010u, 0xf4080404u, 0xfa000000u, 0xbf8cc07fu, 0xf4080535u,
	0xfa000010u, 0xf4080608u, 0xfa000010u, 0xf4080708u, 0xfa000040u, 0xf4100009u, 0xfa0000c0u, 0xf4040409u,
	0xfa0001c0u, 0xf4040809u, 0xfa0002d0u, 0xf4001a89u, 0xfa0002d8u, 0xbf8cc07fu, 0xe0382020u, 0x80050a11u,
	0xe0382010u, 0x80050611u, 0xe0382000u, 0x80050011u, 0xe0082000u, 0x80060e05u, 0xe0042000u, 0x80070405u,
	0xbf8c3f71u, 0x3e1a1910u, 0x3e121110u, 0x3e060510u, 0xbf8c3f70u, 0xd52f0002u, 0x00010105u, 0x3e1a170fu,
	0x3e120f0fu, 0x3e06030fu, 0x3e1a150eu, 0xd5410001u, 0x04261d06u, 0x3e06010eu, 0xd5410007u, 0x000e1a02u,
	0xd5410008u, 0x003e1a0eu, 0xd5410009u, 0x001e1a06u, 0xd541000au, 0x002e1a0au, 0x0a001a6au, 0x3e0e0201u,
	0xd5410006u, 0x0422020du, 0xd5410008u, 0x04260205u, 0x3e140209u, 0x0a0a0620u, 0xd5410009u, 0x041e0600u,
	0x3e0c060cu, 0x3e100604u, 0x3e140608u, 0x0a020221u, 0x5e000900u, 0x3e120c10u, 0x3e100c11u, 0x5e020305u,
	0xf80008cfu, 0x060a0809u, 0xf800020fu, 0x02000111u, 0xbf810000u,
};
[[nodiscard]] inline bool NativoSombrasSkinCode(uint64_t es) {
	using Libs::Graphics::HostMemoryRangeIsReadable;
	const auto eq = [es](const uint32_t* code, size_t bytes) {
		return HostMemoryRangeIsReadable(es, bytes) && std::memcmp(reinterpret_cast<const void*>(es), code, bytes) == 0;
	};
	return es != 0 && (eq(kNativoSombrasSkinVs, sizeof(kNativoSombrasSkinVs)) || eq(kNativoSombrasSkinVs2, sizeof(kNativoSombrasSkinVs2)) ||
	                   eq(kNativoSombrasSkinVs3, sizeof(kNativoSombrasSkinVs3)) || eq(kNativoSombrasSkinVs4, sizeof(kNativoSombrasSkinVs4)));
}
// (hilo del juego) 1 si el codigo del ES en `es` es exactamente kNativoSombrasSkinVs (sin cache: la memoria de shaders
// se reutiliza; son 380 bytes por draw).
inline uint32_t NativoSombrasVsKind(uint64_t es) {
	using Libs::Graphics::HostMemoryRangeIsReadable;
	return NativoSombrasSkinCode(es) ? 1u : 0u;
}

// (COSTE) Log 20261125-065754-173: "recorrido DCB" 622.7 us por cubo (3.3 ms/fot). Cada HostMemoryRangeIsReadable es un
// VirtualQuery (llamada al sistema, mas lenta cuanto mas troceadas estan las protecciones): una por puntero candidato de
// cada SET_SH_REG y ~6 por draw. Aqui: la legibilidad por pagina de 4 KiB, recordada solo dentro de una llamada a
// NativoSombrasNatFill (generacion nueva en cada una): la respuesta es la misma (la proteccion es por pagina) con una
// VirtualQuery por pagina distinta en vez de una por lectura.
// (COSTE 2, log 20261125-092605-970: recorrido DCB 298 us por cubo, gancho 2,23 ms/fot.) La memoria se reiniciaba en
// cada llamada (CatchUp, Fill y Summary: 3 por cubo), asi que cada pagina de tablas indirectas, punteros de SET_SH_REG,
// V# y ES se volvia a consultar (VirtualQuery) en cada cubo. Ahora la generacion avanza cada 64 llamadas (~2-3
// fotogramas: ~12-18 por fotograma): los bloques del juego que se leen (monton del fotograma, tablas de registros,
// codigo de shaders) no se desmapean entre cubos de un mismo fotograma.
struct NativoSombrasPageMemo {
	uint64_t key[512] {};
	uint32_t gen[512] {};
	uint8_t  ok[512] {};
	uint32_t now   = 0;
	uint32_t calls = 0;
	// (VS de skinning: el resultado de la comparacion de 380 bytes por direccion de ES, misma generacion.)
	uint64_t es_key[16] {};
	uint32_t es_gen[16] {};
	uint8_t  es_kind[16] {};
	uint64_t ps_key[16] {};
	uint32_t ps_gen[16] {};
	uint8_t  ps_kind[16] {};
	uint16_t ps_off[16] {};
};
inline NativoSombrasPageMemo& NativoSombrasMemo() {
	static thread_local NativoSombrasPageMemo m;
	return m;
}
inline void NativoSombrasMemoNew() {
	auto& m = NativoSombrasMemo();
	if (m.now != 0 && (++m.calls & 63u) != 0) return;
	if (++m.now == 0) {
		std::memset(m.gen, 0, sizeof(m.gen));
		std::memset(m.es_gen, 0, sizeof(m.es_gen));
		std::memset(m.ps_gen, 0, sizeof(m.ps_gen));
		m.now = 1;
	}
}
inline bool NativoSombrasReadable(uint64_t a, uint64_t size) {
	if (a == 0 || size == 0 || UINT64_MAX - a < size) return false;
	auto& m = NativoSombrasMemo();
	for (uint64_t pg = a >> 12u, last = (a + size - 1) >> 12u; pg <= last; ++pg) {
		const uint32_t i = static_cast<uint32_t>((pg * 0x9E3779B97F4A7C15ull) >> 55u); // (0..511)
		if (m.gen[i] != m.now || m.key[i] != pg) {
			m.key[i] = pg;
			m.gen[i] = m.now;
			m.ok[i]  = Libs::Graphics::HostMemoryRangeIsReadable(pg << 12u, 4096) ? 1 : 0;
		}
		if (m.ok[i] == 0) return false;
	}
	return true;
}

// (DIAG, ALFA) Los datos que lee un PS de recorte, una vez por direccion de PS.
inline void NativoSombrasDiagPsData(uint64_t ps, uint64_t table) {
	static std::atomic<uint64_t> seen[64] {};
	if (ps == 0 || ps == ~uint64_t {0}) return;
	bool claimed = false;
	for (auto& k: seen) {
		uint64_t expected = 0;
		if (k.load(std::memory_order_relaxed) == ps) return;
		if (k.compare_exchange_strong(expected, ps)) { claimed = true; break; }
		if (expected == ps) return;
	}
	if (!claimed) return;
	const auto dump = [&](const char* what, uint64_t a, uint32_t bytes) {
		if (a < 0x100000ull || !NativoSombrasReadable(a, bytes)) {
			std::printf("[nativo-sombras-alfa] PS @%llx %s @%llx: no legible\n", static_cast<unsigned long long>(ps), what,
			            static_cast<unsigned long long>(a));
			return;
		}
		const auto* w = reinterpret_cast<const uint32_t*>(a);
		for (uint32_t i = 0; i < bytes / 4; i += 8) {
			std::printf("[nativo-sombras-alfa] PS @%llx %s +%03x:", static_cast<unsigned long long>(ps), what, i * 4);
			for (uint32_t j = i; j < i + 8 && j < bytes / 4; ++j) std::printf(" %08x", w[j]);
			std::printf("\n");
		}
	};
	dump("tabla", table, 0x40);
	if (table >= 0x100000ull && NativoSombrasReadable(table, 0x30)) {
		const auto* t = reinterpret_cast<const uint64_t*>(table);
		dump("cbuf", t[3] & 0xFFFFFFFFFFFFull, 0x2a0);
		dump("samplers", t[5] & 0xFFFFFFFFFFFFull, 0x180);
	}
	std::fflush(stdout);
}

// (hilo del juego) Que hace en profundidad el PS de un draw de skinning (T32/T34 11-27: 20 PS volcados, llvm-mc gfx1030):
//  1: solo "exp null done vm; s_endpgm" (f8001890 00000000 bf810000): igual que sin PS.
//  2: "fundido con bandera": s_load_dwordx2 s[12:13], s[0:1], 0x18; s_load_dword vcc_lo, s[12:13], OFF;
//     s_cmp_lg_u32 0, vcc_lo; s_cbranch_scc0 -> justo a la cola "s_mov exec, s[14:15]; exp null; s_endpgm". Con la
//     bandera (cbuf+OFF) a 0 el PS no descarta nada (8 de los 20: 26c978200, 26c994500, 26a7e2a00, ... con OFF 0x28 o
//     0x2c). Con bandera distinta de 0 descarta por una textura de ruido y la distancia (fundido): al juego.
//  0: otro (recorte alfa de verdad, varias texturas o desconocido): al juego.
// Recordado por direccion y generacion de la memo.
inline uint32_t NativoSombrasPsKind(uint64_t ps, uint32_t* flag_off) {
	auto&          m = NativoSombrasMemo();
	const uint32_t i = static_cast<uint32_t>((ps >> 8u) * 0x9E3779B97F4A7C15ull >> 60u); // (0..15)
	if (m.ps_gen[i] != m.now || m.ps_key[i] != ps) {
		uint32_t kind = 0, off = 0;
		if (NativoSombrasReadable(ps, 12)) {
			const auto* w = reinterpret_cast<const uint32_t*>(ps);
			if (w[0] == 0xf8001890u && w[1] == 0u && w[2] == 0xbf810000u) kind = 1;
		}
		if (kind == 0 && NativoSombrasReadable(ps, 11 * 4)) {
			const auto*    w      = reinterpret_cast<const uint32_t*>(ps);
			const uint32_t pre[]  = {0xbfa00003u, 0xf4040300u, 0xfa000018u, 0xbf8cc07fu, 0xf4001a86u};
			const uint32_t pre2[] = {0xbe8e047eu, 0xbefe0a7eu, 0xbf8cc07fu, 0xbf076a80u};
			bool           ok     = std::memcmp(w, pre, sizeof(pre)) == 0 && (w[5] & 0xFFFFF000u) == 0xfa000000u &&
			              std::memcmp(w + 6, pre2, sizeof(pre2)) == 0 && (w[10] & 0xFFFF0000u) == 0xbf840000u;
			if (ok) {
				const uint32_t target = 11u + (w[10] & 0xFFFFu); // (s_cbranch_scc0: PC + 4 + simm16 * 4)
				uint32_t       len    = 0;                       // (hasta el primer s_endpgm)
				while (len < 1024u && NativoSombrasReadable(ps + len * 4u, 4) && w[len] != 0xbf810000u) ++len;
				ok = len < 1024u && w[len] == 0xbf810000u && target + 3u == len && w[target] == 0xbefe040eu &&
				     w[target + 1] == 0xf8001890u && w[target + 2] == 0u;
				if (ok) { kind = 2; off = w[5] & 0xFFFu; }
			}
		}
		m.ps_key[i]  = ps;
		m.ps_gen[i]  = m.now;
		m.ps_kind[i] = static_cast<uint8_t>(kind);
		m.ps_off[i]  = static_cast<uint16_t>(off);
	}
	*flag_off = m.ps_off[i];
	return m.ps_kind[i];
}
// (ALFA, T38 11-27; 9 de 35 PS de skinning, llvm-mc gfx1013) F1: fundido con bandera (cbuf+0x28; con 0 se salta) y
// recorte alfa: s_load_dwordx2 s[16:17], s[0:1], 0x28 (tabla de samplers); s_load_dwordx4 s[8:11], s[16:17], 0x170
// (S#); s_load_dwordx8 s[0:7], s[12:13], T (T# en el cbuf); image_sample v0, v[2:3] dmask 0x1 (canal r, UV =
// param0.y.hi / param0.z.lo); s_load_dword vcc_lo, s[12:13], U (umbral); v_cmp_ge_f32 vcc_lo, vcc_lo, v0 -> descarta si
// umbral >= r. Devuelve false si el codigo no es exactamente ese esquema.
inline bool NativoSombrasPsF1(const uint32_t* w, uint32_t len, uint32_t* t_off, uint32_t* u_off) {
	static constexpr uint32_t pre[] = {0xbfa00003u, 0xbefc0302u, 0xf4040300u, 0xfa000018u, 0xbf8cc07fu, 0xf4001a86u,
	                                   0xfa000028u, 0xbe8e047eu, 0xbefe0a7eu, 0xbf8cc07fu, 0xbf076a80u};
	constexpr uint32_t        npre  = sizeof(pre) / sizeof(pre[0]);
	if (len < npre + 16u || std::memcmp(w, pre, sizeof(pre)) != 0) return false;
	if (w[len - 1] != 0xbf810000u || w[len - 2] != 0u || w[len - 3] != 0xf8001890u || w[len - 4] != 0xbefe040eu) return false;
	bool     samplers = false;
	uint32_t found    = 0;
	for (uint32_t i = npre; i + 1 < len; ++i) {
		if (w[i] == 0xf4040400u && w[i + 1] == 0xfa000028u) samplers = true; // (s_load_dwordx2 s[16:17], s[0:1], 0x28)
		if (w[i] != 0xf4080208u || w[i + 1] != 0xfa000170u) continue;       // (s_load_dwordx4 s[8:11], s[16:17], 0x170)
		if (i + 12 >= len || w[i + 2] != 0xf40c0006u || (w[i + 3] & 0xFFF00000u) != 0xfa000000u) return false;
		uint32_t j = i + 4;
		while (j < i + 7 && w[j] == 0xbf8cc07fu) ++j; // (s_waitcnt)
		if (w[j] != 0xf0800108u || w[j + 1] != 0x00400002u) return false; // (image_sample v0, v[2:3], s[0:7], s[8:11] dmask 1)
		if (w[j + 2] != 0xf4001a86u || (w[j + 3] & 0xFFF00000u) != 0xfa000000u) return false;
		uint32_t c = j + 4;
		while (c < j + 7 && (w[c] & 0xFFFF0000u) == 0xbf8c0000u) ++c; // (s_waitcnt)
		if (w[c] != 0x7c0c006au) return false;                       // (v_cmp_ge_f32 vcc_lo, vcc_lo, v0)
		*t_off = w[i + 3] & 0xFFFFFu;
		*u_off = w[j + 3] & 0xFFFFFu;
		++found;
	}
	return samplers && found == 1;
}
// (ALFA-a) F1 recordado por direccion de PS (memo aparte de NativoSombrasPsKind): 0 no, 1 si con t/u.
inline bool NativoSombrasPsF1At(uint64_t ps, uint32_t* t_off, uint32_t* u_off) {
	struct Memo { uint64_t key[16] {}; uint32_t gen[16] {}; uint8_t ok[16] {}; uint32_t t[16] {}, u[16] {}; };
	static thread_local Memo f;
	auto&          m = NativoSombrasMemo();
	const uint32_t i = static_cast<uint32_t>((ps >> 8u) * 0x9E3779B97F4A7C15ull >> 60u);
	if (f.gen[i] != m.now || f.key[i] != ps) {
		uint32_t len = 0;
		while (len < 1024u && NativoSombrasReadable(ps + len * 4u, 4) && reinterpret_cast<const uint32_t*>(ps)[len] != 0xbf810000u) ++len;
		uint32_t t = 0, u = 0;
		const bool ok = len < 1024u && NativoSombrasReadable(ps + len * 4u, 4) &&
		                NativoSombrasPsF1(reinterpret_cast<const uint32_t*>(ps), len + 1u, &t, &u);
		f.key[i] = ps; f.gen[i] = m.now; f.ok[i] = ok ? 1 : 0; f.t[i] = t; f.u[i] = u;
	}
	*t_off = f.t[i];
	*u_off = f.u[i];
	return f.ok[i] != 0;
}
// (ALFA-a) Los datos de una entrada F1 con el fundido apagado (cbuf+0x28 == 0): T# (cbuf+T), S# (*(tabla+0x28)+0x170) y
// umbral (cbuf+U). Falso: algo no legible, fundido encendido o T# nulo.
inline bool NativoSombrasAlfaRead(uint64_t ps_table, uint32_t t_off, uint32_t u_off, DsNativoSombras::NatAlfa& out) {
	if (ps_table < 0x100000ull || !NativoSombrasReadable(ps_table, 0x30)) return false;
	const auto*    t  = reinterpret_cast<const uint64_t*>(ps_table);
	const uint64_t cb = t[3] & 0xFFFFFFFFFFFFull, smp = t[5] & 0xFFFFFFFFFFFFull;
	if (cb < 0x100000ull || smp < 0x100000ull || !NativoSombrasReadable(cb, 0x2c + 4) || !NativoSombrasReadable(cb + t_off, 32) ||
	    !NativoSombrasReadable(cb + u_off, 4) || !NativoSombrasReadable(smp + 0x170, 16))
		return false;
	if (*reinterpret_cast<const uint32_t*>(cb + 0x28) != 0u) return false; // (fundido encendido)
	std::memcpy(out.tsharp, reinterpret_cast<const void*>(cb + t_off), sizeof(out.tsharp));
	std::memcpy(out.ssharp, reinterpret_cast<const void*>(smp + 0x170), sizeof(out.ssharp));
	std::memcpy(&out.threshold, reinterpret_cast<const void*>(cb + u_off), 4);
	return (out.tsharp[0] | (out.tsharp[1] & 0xFFu)) != 0;
}
// (tipo 2) La bandera que leera el PS: tabla = USER_DATA_PS s[0:1] (SH 0xC/0xD), cbuf = *(tabla+0x18), bandera =
// *(cbuf+OFF). true: se pudo leer y vale 0 (el PS no descarta). Leida al grabar el draw (los cbuf de Agc no se
// reescriben despues en el mismo fotograma); la comparacion con el juego lo vigila.
inline bool NativoSombrasPsFlagOff(uint64_t ps_table, uint32_t off) {
	if (ps_table < 0x100000ull || !NativoSombrasReadable(ps_table + 0x18, 8)) return false;
	const uint64_t cb = *reinterpret_cast<const uint64_t*>(ps_table + 0x18) & 0xFFFFFFFFFFFFull;
	if (cb < 0x100000ull || !NativoSombrasReadable(cb + off, 4)) return false;
	return *reinterpret_cast<const uint32_t*>(cb + off) == 0u;
}

// (0x63 / 0x64 / 0x9F) SET_SH/UCONFIG/CONTEXT_REG_INDIRECT, como CpOpIndirect*Regs (pm4Handlers.cpp): 5 dwords;
// [1] & ~3 | [2] << 32 = tabla de pares (desplazamiento, valor) en memoria del juego, [4] & 0x3fff = n de pares; el
// desplazamiento se normaliza quitando 0x70000000 y 0xffffffff es relleno. Agc manda asi el ES (0xC8/0xC9) y los
// registros de los shaders: sin leerlos, "ES no esta en la DCB". Falso: tabla no legible o paquete raro.
inline bool NativoSombrasIndirectPairs(const uint32_t* p, uint32_t len, const uint32_t** pairs, uint32_t* n) {
	*pairs = nullptr;
	*n     = 0;
	if (len != 5u) return false;
	const uint64_t a   = (uint64_t {p[1]} & 0xfffffffcu) | (uint64_t {p[2]} << 32u);
	const uint32_t num = p[4] & 0x3fffu;
	if (num == 0) return true;
	if (a == 0 || !NativoSombrasReadable(a, uint64_t {num} * 8u)) return false;
	*pairs = reinterpret_cast<const uint32_t*>(a);
	*n     = num;
	return true;
}
inline uint32_t NativoSombrasRegOffset(uint32_t raw) { return raw & ~0x70000000u; }
// (0x63) Los pares SH < 0x400 de un SET_SH_REG_INDIRECT a val/ok (SH_NOP 0x280 y 0xffffffff no cuentan). Falso: no legible.
inline bool NativoSombrasApplyShIndirect(const uint32_t* p, uint32_t len, uint32_t* val, uint8_t* ok) {
	const uint32_t* q = nullptr;
	uint32_t        n = 0;
	if (!NativoSombrasIndirectPairs(p, len, &q, &n)) return false;
	for (uint32_t i = 0; i < n; ++i, q += 2) {
		if (q[0] == 0xffffffffu) continue;
		const uint32_t r = NativoSombrasRegOffset(q[0]);
		if (r < 0x400u && r != 0x280u) { val[r] = q[1]; ok[r] = 1; }
	}
	return true;
}

// (hilo del juego) Estado de los registros SH (SET_SH_REG 0x76) del bufer de comandos de este hilo hasta parsed_to.
struct NativoSombrasSh {
	const void*     cmds      = nullptr;
	const uint32_t* bottom    = nullptr;
	const uint32_t* parsed_to = nullptr;
	bool            valid     = false;
	bool            full      = false; // (todos los paquetes desde el inicio del bloque entendidos: ok == 0 -> no escrito)
	uint32_t        val[0x400] {};
	uint8_t         ok[0x400] {};
};
inline NativoSombrasSh& NativoSombrasShState() {
	static thread_local NativoSombrasSh s;
	return s;
}
// Pone el estado SH al dia hasta `before` (inicio del cubo): sigue desde donde quedo el cubo anterior del mismo bufer y
// bloque; si es otro bufer, otro bloque o se reinicio, desde el principio del bloque y sin estado previo (lo que no se
// sabe queda sin poner: como antes, VS estatico). Solo lee la DCB que escribe este mismo hilo.
inline void NativoSombrasShCatchUp(const AgcCommandBuffer* commands, const uint32_t* before) {
	auto&           s    = NativoSombrasShState();
	NativoSombrasMemoNew();
	const uint32_t* from = s.parsed_to;
	if (!s.valid || s.cmds != commands || s.bottom != commands->bottom || s.parsed_to == nullptr || s.parsed_to > before ||
	    s.parsed_to < commands->bottom) {
		std::memset(s.ok, 0, sizeof(s.ok));
		from   = commands->bottom;
		s.full = true;
	}
	s.cmds      = commands;
	s.bottom    = commands->bottom;
	s.parsed_to = before;
	s.valid     = true;
	if (from == nullptr || before == nullptr || from > before) return;
	for (const uint32_t* p = from; p < before;) {
		const uint32_t h = *p;
		if ((h >> 30u) == 2u) { ++p; continue; }
		const uint32_t len = ((h >> 16u) & 0x3fffu) + 2u;
		if ((h >> 30u) != 3u || p + len > before) { std::memset(s.ok, 0, sizeof(s.ok)); s.full = false; return; } // (no se entiende: sin estado)
		if (((h >> 8u) & 0xffu) == 0x76u && len >= 3) {
			const uint32_t reg0 = p[1] & 0xFFFFu;
			for (uint32_t i = 2; i < len; ++i) {
				const uint32_t reg = reg0 + (i - 2);
				if (reg < 0x400u) { s.val[reg] = p[i]; s.ok[reg] = 1; }
			}
		}
		if (((h >> 8u) & 0xffu) == 0x63u && !NativoSombrasApplyShIndirect(p, len, s.val, s.ok)) {
			std::memset(s.ok, 0, sizeof(s.ok)); // (tabla no legible: sin estado, como un paquete no entendido)
			s.full      = false;
			s.parsed_to = nullptr;
			s.valid     = false;
			return;
		}
		p += len;
	}
}

// (DIAG, KYTY_NATIVO_SOMBRAS_DIAG=1, hilo del juego) El otro VS de skinning (no 0x3866090d; hash del emulador
// 0xcd218bf8feea0f5b, 37 slots, 4 buferes): su codigo (GpuCode, hasta s_endpgm, una vez por direccion), los USER_DATA
// GS que trae la DCB (s8.. = 0x8C..) y las 8 entradas de su tabla, con 64 bytes de cada V# en +0x00..+0x40 de las 4
// primeras. Con esto se replica operacion a operacion (gpu_ds_sombras_skin2.vert). 4 shaders distintos como mucho.
inline void NativoSombrasDiagOtherVs(uint64_t es, uint64_t table, const uint32_t* val, const uint8_t* ok, uint32_t treg) {
	static std::atomic<uint64_t> seen[4] {};
	for (auto& k: seen) {
		uint64_t expected = 0;
		if (k.load(std::memory_order_relaxed) == es) return;
		if (k.compare_exchange_strong(expected, es)) {
			char   line[1024];
			size_t at = static_cast<size_t>(std::snprintf(line, sizeof(line), "[nativo-sombras] DIAG otro VS skinning ES @%llx tabla @%llx (reg 0x%x) USER_DATA_GS:",
			                                              static_cast<unsigned long long>(es), static_cast<unsigned long long>(table), treg));
			for (uint32_t r = 0x8Cu; r < 0x8Cu + 32u && at < sizeof(line); ++r)
				at += static_cast<size_t>(ok[r] != 0 ? std::snprintf(line + at, sizeof(line) - at, " s%u=%08x", r - 0x8Cu + 8u, val[r])
				                                      : std::snprintf(line + at, sizeof(line) - at, " s%u=?", r - 0x8Cu + 8u));
			std::printf("%s\n", line);
			if (table != 0 && NativoSombrasReadable(table, 64)) {
				const auto* t = reinterpret_cast<const uint64_t*>(table);
				std::printf("[nativo-sombras] DIAG otro VS tabla: %llx %llx %llx %llx %llx %llx %llx %llx\n",
				            static_cast<unsigned long long>(t[0] & 0xFFFFFFFFFFFFull), static_cast<unsigned long long>(t[1] & 0xFFFFFFFFFFFFull),
				            static_cast<unsigned long long>(t[2] & 0xFFFFFFFFFFFFull), static_cast<unsigned long long>(t[3] & 0xFFFFFFFFFFFFull),
				            static_cast<unsigned long long>(t[4] & 0xFFFFFFFFFFFFull), static_cast<unsigned long long>(t[5] & 0xFFFFFFFFFFFFull),
				            static_cast<unsigned long long>(t[6] & 0xFFFFFFFFFFFFull), static_cast<unsigned long long>(t[7] & 0xFFFFFFFFFFFFull));
				for (int e = 0; e < 4; ++e) {
					const uint64_t a = t[e] & 0xFFFFFFFFFFFFull;
					if (a == 0 || !NativoSombrasReadable(a, 64)) continue;
					const auto* w = reinterpret_cast<const uint32_t*>(a);
					std::printf("[nativo-sombras] DIAG otro VS tabla[%d] @%llx: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
					            e, static_cast<unsigned long long>(a), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10],
					            w[11], w[12], w[13], w[14], w[15]);
				}
			}
			std::fflush(stdout);
			DsNativoSombras::GpuCode("es-skin-otro", es);
			return;
		}
	}
}

// (hilo del juego, sin candados) Los datos de cada draw de este cubo, en el orden de la lista (= orden de la DCB,
// medido al 100 %). La tabla de USER_DATA del VS de cada draw: el ultimo SET_SH_REG (0x76) antes de el con un puntero
// cuya entrada [1] es el bufer de camara (ctx+0xb8), como en NativoSombrasDumpUserTable. Falso: algo no cuadra.
const char* NativoSombrasNatFill(DsNativoSombras::NatSlot* slot, const uint64_t* ctx, uint64_t first, uint64_t last, const uint32_t* begin,
                                 const uint32_t* end) {
	namespace H = DsNativoSombras;
	const DsSombrasAb::Timer cost_fill(DsSombrasAb::CFill);
	NativoSombrasMemoNew();
	slot->alfa_n = 0;
	const auto HostMemoryRangeIsReadable = [](uint64_t a, uint64_t size) { return NativoSombrasReadable(a, size); };
	const uint64_t entries = (last - first) / 0x20;
	const uint64_t cam     = ctx[0xb8 / 8] & 0xFFFFFFFFFFFFull;
	if (cam == 0) return "sin camara";
	// 1) Las tablas de los draws, en orden.
	static thread_local uint64_t tables[H::NatMaxDraws];
	// (El draw tal como esta en la DCB: opcode, n de indices e inicio. DRAW_INDEX_OFFSET_2 (0x35): [2] inicio, [3] n;
	// DRAW_INDEX_2 (0x27): [4] n, inicio 0. Lo que la GPU dibuja de verdad: manda sobre item[4]+0x38/+0x3c.)
	static thread_local uint32_t dcb_op[H::NatMaxDraws], dcb_count[H::NatMaxDraws], dcb_start[H::NatMaxDraws];
	// (Y la base y el tamano de indices que ve ese draw en la DCB: INDEX_BASE (0x26) [1] lo, [2] hi; INDEX_TYPE (0x2A)
	// [1] & 3 (0: 16 bits, 1: 32); DRAW_INDEX_2 (0x27) trae su base en [2], [3]. 0: no aparece en esta llamada (el
	// juego no la reemite si no cambia, 0x9142c0): vale la del CMesh.)
	static thread_local uint64_t dcb_ibase[H::NatMaxDraws];
	static thread_local int32_t  dcb_itype[H::NatMaxDraws];
	// (SKIN) Por draw: el ES de la DCB (SPI_SHADER_PGM_LO/HI_ES 0xC8/0xC9: base = lo << 8 | (hi & 0xff) << 40; 0 si no
	// aparece) y USER_DATA s10 = el registro de la tabla (s8) + 2 (el VS de skinning indexa las filas del objeto con
	// s10 & 0xffff + instance_id). Registros SH vistos en esta DCB (desplazamiento < 0x400).
	static thread_local uint64_t dcb_es[H::NatMaxDraws];
	static thread_local uint64_t dcb_psu[H::NatMaxDraws]; // (USER_DATA_PS s[0:1], SH 0xC/0xD: tabla del PS; 0 si no)
	static thread_local uint64_t dcb_ps[H::NatMaxDraws]; // (SPI_SHADER_PGM_LO/HI_PS 0x8/0x9 de ese draw; ~0: no se sabe)
	static thread_local uint32_t dcb_s10[H::NatMaxDraws];
	static thread_local uint8_t  dcb_s10_ok[H::NatMaxDraws];
	static thread_local uint32_t dcb_treg[H::NatMaxDraws]; // (registro SH del puntero de la tabla (s8) de ese draw)
	static thread_local uint32_t sh_val[0x400];
	static thread_local uint8_t  sh_ok[0x400];
	// (Log 20261125-061906-403: "skinning: ES no esta en la DCB (se dibuja con el VS estatico)" 1,4 M de 1,9 M entradas;
	// cascadas 0-1 con skinning 17-38 %.) Agc solo reemite SPI_SHADER_PGM_ES (0xC8/0xC9) y los USER_DATA cuando cambian:
	// en la GPU siguen valiendo los de paquetes anteriores del mismo bufer. Se parte del estado SH que dejo el bufer hasta
	// `begin` (NativoSombrasShCatchUp), no de vacio.
	auto& persist = NativoSombrasShState();
	std::memcpy(sh_val, persist.val, sizeof(sh_val));
	std::memcpy(sh_ok, persist.ok, sizeof(sh_ok));
	persist.valid = false; // (si el recorrido falla, el estado deja de ser fiable)
	uint32_t                     table_reg = 0xFFFFFFFFu;
	uint64_t                     draws = 0, table = 0, index_base = 0;
	int32_t                      index_type = -1;
	for (const uint32_t* p = begin; p < end;) {
		const uint32_t h = *p;
		if ((h >> 30u) == 2u) { ++p; continue; }
		if ((h >> 30u) != 3u) return "paquete desconocido";
		const uint32_t len = ((h >> 16u) & 0x3fffu) + 2u;
		const uint32_t op  = (h >> 8u) & 0xffu;
		if (p + len > end) return "paquete cortado";
		if (op == 0x76 && len >= 3) {
			const uint32_t reg0 = p[1] & 0xFFFFu;
			for (uint32_t i = 2; i < len; ++i) {
				const uint32_t reg = reg0 + (i - 2);
				if (reg < 0x400u) { sh_val[reg] = p[i]; sh_ok[reg] = 1; }
			}
		}
		if (op == 0x63 && !NativoSombrasApplyShIndirect(p, len, sh_val, sh_ok)) return "SET_SH_REG_INDIRECT no legible";
		if (op == 0x76 && len >= 4) {
			for (uint32_t i = 2; i + 1 < len; ++i) {
				const uint64_t a = uint64_t {p[i]} | (uint64_t {p[i + 1] & 0xFFFFu} << 32u);
				if ((p[i + 1] >> 16u) != 0 || a < 0x100000ull || (a & 7u) != 0 || !HostMemoryRangeIsReadable(a, 32)) continue;
				// (11-27, log 20261127-042910-357: "reg 0xc" = USER_DATA_PS_0 trae el mismo puntero; tomarlo daba
				// "s10 fuera de USER_DATA_GS" 1,74 M. El VS de profundidad (ES/NGG) lee s8.. de USER_DATA_GS 0x8C..0xAB.)
				const uint32_t treg = (p[1] & 0xFFFFu) + (i - 2);
				if (treg < 0x8Cu || treg + 2u >= 0x8Cu + 32u) continue;
				if ((reinterpret_cast<const uint64_t*>(a)[1] & 0xFFFFFFFFFFFFull) == cam) {
					table     = a;
					table_reg = (p[1] & 0xFFFFu) + (i - 2);
				}
			}
		}
		if (op == 0x26 && len >= 3) index_base = (uint64_t {p[1]} | (uint64_t {p[2] & 0xFFFFu} << 32u)) & ~uint64_t {1};
		if (op == 0x2a && len >= 2) index_type = static_cast<int32_t>(p[1] & 3u);
		if (op == 0x24 || op == 0x25 || op == 0x27 || op == 0x2d || op == 0x35 || op == 0x38) {
			if (table == 0) return "draw sin tabla";
			if (draws >= H::NatMaxDraws) return "demasiados draws";
			dcb_ibase[draws] = op == 0x27 && len >= 5 ? (uint64_t {p[2]} | (uint64_t {p[3] & 0xFFFFu} << 32u)) : index_base;
			dcb_itype[draws] = index_type;
			dcb_op[draws]    = op;
			dcb_count[draws] = op == 0x35 && len >= 4 ? p[3] : op == 0x27 && len >= 5 ? p[4] : 0;
			dcb_start[draws] = op == 0x35 && len >= 4 ? p[2] : 0;
			dcb_ps[draws]    = sh_ok[0x8] != 0 && sh_ok[0x9] != 0 ? (uint64_t {sh_val[0x8]} << 8u) | (uint64_t {sh_val[0x9] & 0xFFu} << 40u) : ~uint64_t {0};
			dcb_psu[draws]   = sh_ok[0xC] != 0 && sh_ok[0xD] != 0 ? (uint64_t {sh_val[0xC]} | (uint64_t {sh_val[0xD] & 0xFFFFu} << 32u)) : 0;
			dcb_es[draws]    = sh_ok[0xC8] != 0 && sh_ok[0xC9] != 0
			                       ? (uint64_t {sh_val[0xC8]} << 8u) | (uint64_t {sh_val[0xC9] & 0xFFu} << 40u)
			                       : 0;
			dcb_s10_ok[draws] = table_reg < 0x3FEu && sh_ok[table_reg + 2u] != 0 ? 1 : 0;
			dcb_s10[draws]    = dcb_s10_ok[draws] != 0 ? sh_val[table_reg + 2u] : 0;
			dcb_treg[draws]   = table_reg;
			tables[draws++]  = table;
		}
		p += len;
	}
	std::memcpy(persist.val, sh_val, sizeof(sh_val));
	std::memcpy(persist.ok, sh_ok, sizeof(sh_ok));
	persist.valid     = true;
	persist.parsed_to = end;
	if (draws != entries) return "draws de la DCB != entradas";
	// 2) Por entrada: indice (CMesh+8, Ghidra 0x9142c0: 32 bits si (dword+4 & 0x3FFF0000) == 0x40000) y draw
	// (0xaba900 -> 0x914360: n = item[4]+0x38, inicio = item[4]+0x3c); por tabla: posicion, objeto y tabla[3].
	// (Ghidra 0xaba900: con (entrada+0x18 & 8) y sin el bit 4 de *(*ctx+8), el draw es 0xa2a150: indirecto, indice de
	// *(ctx[0xf]+0xa8)+0x168; no lo hace el nativo.) Una entrada que el nativo no puede dibujar se omite con su motivo
	// (g_nat_entry_why) y el cubo sigue: su draw del juego se hace siempre.
	const bool ctx_flag4 = ctx[0] != 0 && (*reinterpret_cast<const uint32_t*>(ctx[0] + 8) & 4u) != 0;
	uint32_t   k = 0, skipped = 0, empty = 0;
	// (Motivos contados en local y sumados una vez por cubo: un atomico compartido por entrada costaba en los hilos del juego.)
	struct Tally { const char* why; uint32_t n; };
	Tally      info_t[8] {}, entry_t[8] {};
	const auto tally = [](Tally* t, H::NatWhy* table, const char* why) {
		for (uint32_t i = 0; i < 8; ++i) {
			if (t[i].why == why) { ++t[i].n; return; }
			if (t[i].why == nullptr) { t[i].why = why; t[i].n = 1; return; }
		}
		H::NatCount(table, why);
	};
	const auto info  = [&](const char* why) { tally(info_t, H::g_nat_info_why, why); };
	const auto entry = [&](const char* why) { tally(entry_t, H::g_nat_entry_why, why); };
	struct Flush {
		Tally* a; Tally* b;
		~Flush() {
			for (uint32_t i = 0; i < 8 && a[i].why != nullptr; ++i) DsNativoSombras::NatCountN(DsNativoSombras::g_nat_info_why, a[i].why, a[i].n);
			for (uint32_t i = 0; i < 8 && b[i].why != nullptr; ++i) DsNativoSombras::NatCountN(DsNativoSombras::g_nat_entry_why, b[i].why, b[i].n);
		}
	} flush {info_t, entry_t};
	for (uint64_t e = first; e < last; e += 0x20, ++k) {
		auto& d = slot->draws[k];
		d.skip  = 1;
		d.skin  = 0;
		d.vs    = 0;
		d.es    = 0;
		d.s10_reg = 0;
		const char* why = [&]() -> const char* {
			const auto* item = reinterpret_cast<const uint64_t*>(*reinterpret_cast<const uint64_t*>(e + 8));
			if (item == nullptr || item[2] == 0 || item[4] == 0) return "sin item";
			if ((*reinterpret_cast<const uint8_t*>(e + 0x18) & 8u) != 0 && !ctx_flag4) return "draw indirecto (0xa2a150)";
			const uint64_t subs = *reinterpret_cast<const uint64_t*>(item[2] + 0x130);
			const int32_t  si   = *reinterpret_cast<const int32_t*>(e + 0x14);
			if (subs == 0 || si < 0) return "sin submalla";
			const uint64_t sub  = subs + uint64_t(si) * 0x48;
			const uint64_t anim = *reinterpret_cast<const uint64_t*>(sub + 0x38);
			if (anim != 0) {
				const auto* stamps = reinterpret_cast<const uint32_t*>(anim + 0x1c);
				if ((stamps[0] < stamps[1] ? stamps[1] : stamps[0]) != 0) {
					d.skin = 1; // (sin KYTY_NATIVO_SOMBRAS_SKIN se omite, pero despues de leer sus indices: el emparejado los usa) // (sus vertices: el V# de tabla[0]+0x10, bufer de skinning; los obtiene la GPU de su cache)
				}
			}
			const uint64_t part = *reinterpret_cast<const uint64_t*>(sub + 0x20);
			const uint64_t mesh = part != 0 ? *reinterpret_cast<const uint64_t*>(part + 0x70) : 0;
			if (mesh == 0) return "sin CMesh";
			const auto* iv = reinterpret_cast<const uint32_t*>(mesh + 8);
			d.ibase        = uint64_t {iv[0]} | (uint64_t {iv[1] & 0xFFFFu} << 32u);
			d.index32      = (iv[1] & 0x3FFF0000u) == 0x40000u ? 1u : 0u;
			d.icount       = *reinterpret_cast<const uint32_t*>(item[4] + 0x38);
			d.istart       = *reinterpret_cast<const uint32_t*>(item[4] + 0x3c);
			if (d.ibase == 0) return "indice sin base";
			if (dcb_op[k] != 0x35 && dcb_op[k] != 0x27) return "draw de la DCB no indexado directo";
			// (La base y el tamano de los indices, de la DCB cuando aparecen: lo que recibe CommandProcessor::DrawIndex.)
			if (dcb_ibase[k] != 0 && dcb_ibase[k] != d.ibase) {
				info("base de indices CMesh != DCB (se usa la DCB)");
				d.ibase = dcb_ibase[k];
			}
			if (dcb_itype[k] == 0 || dcb_itype[k] == 1) {
				if (static_cast<uint32_t>(dcb_itype[k]) != d.index32) info("tamano de indice CMesh != DCB (se usa la DCB)");
				d.index32 = static_cast<uint32_t>(dcb_itype[k]);
			} else if (dcb_itype[k] > 1) {
				return "indices de 8 bits";
			}
			if (dcb_count[k] != d.icount || dcb_start[k] != d.istart) {
				info("n/inicio del item != DCB (se usa la DCB)");
				d.icount = dcb_count[k];
				d.istart = dcb_start[k];
			}
			if (d.icount == 0) return "";  // (draw vacio en la DCB: nada que dibujar, ni nativo ni emulado)
			if (d.skin != 0 && !H::NatSkinOn()) return "con skinning";
			const auto*    t  = reinterpret_cast<const uint64_t*>(tables[k]);
			const uint64_t t0 = t[0] & 0xFFFFFFFFFFFFull, t2 = t[2] & 0xFFFFFFFFFFFFull, t3 = t[3] & 0xFFFFFFFFFFFFull;
			if (!HostMemoryRangeIsReadable(t0 + 0x10, 16) || !HostMemoryRangeIsReadable(t2 + 0x10, 16) ||
			    !HostMemoryRangeIsReadable(t3 + 0x14, 16))
				return "tabla no legible";
			const auto* pv = reinterpret_cast<const uint32_t*>(t0 + 0x10);
			d.vbase        = uint64_t {pv[0]} | (uint64_t {pv[1] & 0xFFFFu} << 32u);
			d.vstride      = (pv[1] >> 16u) & 0x3FFFu;
			d.vrecords     = pv[2];
			if (d.vstride != 12u || ((pv[3] >> 12u) & 0x7Fu) != 74u || d.vrecords == 0) return "posicion no es 32_32_32_FLOAT stride 12";
			const auto*    ov    = reinterpret_cast<const uint32_t*>(t2 + 0x10);
			const uint64_t obase = uint64_t {ov[0]} | (uint64_t {ov[1] & 0xFFFFu} << 32u);
			if (obase == 0 || !HostMemoryRangeIsReadable(obase, sizeof(d.obj))) return "objeto no legible";
			// (SKIN) Con el VS 0x3866090d8246e47e (codigo comparado dword a dword con el de la DCB del draw) las 3 filas
			// del objeto son buffer_load_dwordx4 idxen de este V# (+0, +16, +32), indice = (s10 & 0xffff) + instance_id
			// (v_add_nc_u32_sdwa v4, s10, v8 WORD_0): base + indice * stride. instance_id = 0 (el nativo dibuja 1 instancia).
			uint64_t       orow  = obase;
			d.es                   = dcb_es[k]; // (la GPU lo compara con el ES real del draw: g_nat_es_check)
			const uint32_t vs_kind = [&]() -> uint32_t { // (como NativoSombrasVsKind, recordado por ES y generacion)
				const uint64_t es = dcb_es[k];
				if (es == 0) return 0u;
				auto&          m = NativoSombrasMemo();
				const uint32_t i = static_cast<uint32_t>((es >> 8u) * 0x9E3779B97F4A7C15ull >> 60u); // (0..15)
				if (m.es_gen[i] == m.now && m.es_key[i] == es) return m.es_kind[i];
				const uint32_t r = NativoSombrasSkinCode(es) ? 1u : 0u; // (SKIN2: los 4 VS de skinning conocidos)
				m.es_key[i]  = es;
				m.es_gen[i]  = m.now;
				m.es_kind[i] = static_cast<uint8_t>(r);
				return r;
			}();
			// (Con skinning y sin el VS 0x3866090d en la DCB el VS estatico dibujaria la pose de reposo: se omite, la dibuja el juego.)
			if (d.skin != 0 && vs_kind != 1 && dcb_es[k] != 0 && H::NatDiag()) NativoSombrasDiagOtherVs(dcb_es[k], tables[k], sh_val, sh_ok, dcb_treg[k]);
			if (d.skin != 0 && vs_kind != 1)
				return dcb_es[k] == 0 ? "skinning: ES no esta en la DCB (la dibuja el juego)" : "skinning: otro VS (no 0x3866090d; la dibuja el juego)";
			// (T38 11-27, DIAG error: el "nativo mas cerca" con error real (|nativo-juego| hasta 0,66) esta en los cubos con
			// entradas "vs_skin_sin_skin" (VS de skinning sin sello de animacion: 8 px err 0,55 con una sola de ellas; miles en
			// las mezclas; los cubos solo con skinning, decenas). Esas tomaban los vertices como malla estatica, pero el VS lee
			// el bufer de skinning (tabla[0]+0x10) que se reescribe cada fotograma. Ahora van como skinning: vertices de la
			// cache en la marca de fin y la validacion propia del skinning.)
			if (d.skin == 0 && vs_kind == 1) {
				if (!H::NatSkinOn()) return "VS de skinning sin animacion (sin KYTY_NATIVO_SOMBRAS_SKIN: la dibuja el juego)";
				d.skin = 1;
				info("VS de skinning sin animacion: vertices del bufer de skinning");
			}
			// (11-27, log 20261127-060057-463, DIAG error: todo el error del nativo esta en el cubo 1 (kRS_Depth_Normal) y solo en
			// cubos con el VS de skinning; los cubos solo estaticos, 0 px.) Los VS de skinning exportan param0 (UV), el estatico
			// no: van con un PS que lee la UV (recorte alfa: pelo, tela, vegetacion) que el nativo (sin PS) no hace -> dibuja
			// de mas, "nativo mas cerca". Con el VS de skinning y un PS puesto (o sin saber cual), la dibuja el juego.
			// (T32 11-27, 20 PS distintos volcados y desensamblados (llvm-mc gfx1030): 16 hacen image_sample + v_cmp +
			// s_andn2 de la mascara viva = recorte alfa; 4 son solo "exp null done vm; s_endpgm" (3 dwords), iguales a no
			// tener PS: esas entradas van en nativo.)
			uint32_t       ps_off  = 0;
			const uint32_t ps_kind = vs_kind == 1 && dcb_ps[k] != 0 && dcb_ps[k] != ~uint64_t {0} ? NativoSombrasPsKind(dcb_ps[k], &ps_off) : 0u;
			if (ps_kind == 1)
				info("VS de skinning con PS solo exp null (dibujada en nativo)");
			else if (ps_kind == 2 && NativoSombrasPsFlagOff(dcb_psu[k], ps_off))
				info("VS de skinning con PS de fundido apagado (bandera 0: dibujada en nativo)");
			else if (ps_kind == 2)
				return "VS de skinning con PS de fundido encendido (la dibuja el juego)";
			else if (uint32_t t_off = 0, u_off = 0; H::NatAlfaOn() && ps_kind == 0 && vs_kind == 1 && dcb_ps[k] != 0 &&
			         dcb_ps[k] != ~uint64_t {0} && NativoSombrasPsF1At(dcb_ps[k], &t_off, &u_off)) {
				// (ALFA-a) Se captura para que el hilo de la GPU resuelva textura y sampler; la entrada la sigue dibujando el juego.
				if (slot->alfa_n < H::NatAlfaMax && NativoSombrasAlfaRead(dcb_psu[k], t_off, u_off, slot->alfa[slot->alfa_n])) {
					slot->alfa[slot->alfa_n++].entry = static_cast<uint32_t>(k);
					info("ALFA-a: F1 capturada (la dibuja el juego)");
				}
				return "VS de skinning con PS F1 de recorte alfa (ALFA-a: la dibuja el juego)";
			}
			else if (vs_kind == 1 && dcb_ps[k] != 0) {
				if (H::NatDiag()) {
					static std::atomic<uint32_t> shown {0};
					if (shown.fetch_add(1, std::memory_order_relaxed) < 8)
						std::printf("[nativo-sombras] DIAG VS de skinning con PS @%llx ES @%llx: al juego\n",
						            static_cast<unsigned long long>(dcb_ps[k]), static_cast<unsigned long long>(dcb_es[k]));
					// (Su codigo al log, para separar los PS que descartan (recorte alfa) de los que no.)
					if (dcb_ps[k] != ~uint64_t {0}) H::GpuCode("ps-skin", dcb_ps[k]);
					// (ALFA, T38) Para el recorte alfa nativo: el VS de skinning de este draw (de donde saca la UV de
					// param0.z) y, una vez por PS, sus descriptores: tabla USER_DATA_PS s[0:1], cbuf (*(tabla+0x18), los
					// T# en linea en cbuf+0x50/0x180/0x1c0/0x280 y umbrales 0x24/0x54/0x64) y la tabla de samplers
					// (*(tabla+0x28), S# en +0xb0 y +0x170). Una sola vez cada uno (64 huecos).
					H::GpuCode("es-skin", dcb_es[k]);
					NativoSombrasDiagPsData(dcb_ps[k], dcb_psu[k]);
				}
				return dcb_ps[k] == ~uint64_t {0} ? "VS de skinning con PS desconocido (la dibuja el juego)" : "VS de skinning con PS (recorte alfa: la dibuja el juego)";
			}
			if (vs_kind == 1 && dcb_ps[k] == 0) info("VS de skinning sin PS (dibujada en nativo)");
			if (vs_kind == 1) {
				if (dcb_s10_ok[k] == 0 && d.skin == 0) {
					// (Sin skinning y sin s10 en la DCB: como antes, VS estatico con la fila 0; validado en cascadas 2-3.)
				} else if (dcb_s10_ok[k] == 0) {
					// (Ghidra 11-25: Agc no emite s10 para estos draws; ver NatDraw::s10_reg. La GPU usa el valor que
					// tenia el registro: lo lee el hilo de la GPU (gs_user_sgpr) y la entrada solo se dibuja en nativo
					// si (s10 & 0xffff) == 0, con la fila 0 que se copia aqui.)
					const uint32_t reg = dcb_treg[k] + 2u;
					if (dcb_treg[k] >= 0x3FEu || reg < 0x8Cu || reg - 0x8Cu >= 32u) {
						// (DIAG 11-27, log 20261127-041016-200: 1,74 M entradas aqui) En que registro SH esta la tabla (s8) y que
						// valores SH trae el estado en ese rango: con esto se sabe de donde leer el s10 (otra etapa o sin tabla).
						static std::atomic<uint32_t> shown {0};
						if (H::NatDiag() && shown.fetch_add(1, std::memory_order_relaxed) < 12) {
							const uint32_t t = dcb_treg[k];
							std::printf("[nativo-sombras] DIAG s10 fuera: reg tabla=0x%x es=%llx pase=%u", t, static_cast<unsigned long long>(dcb_es[k]), slot->pass);
							if (t < 0x3F0u)
								for (uint32_t r = t; r < t + 8u; ++r)
									std::printf(sh_ok[r] != 0 ? " [0x%x]=%08x" : " [0x%x]=?", r, sh_val[r]);
							std::printf("\n");
							std::fflush(stdout);
						}
						return "skinning: s10 fuera de USER_DATA_GS";
					}
					if (!HostMemoryRangeIsReadable(obase, sizeof(d.obj))) return "objeto no legible";
					d.s10_reg = reg;
					d.vs      = 1;
					info("s10 no esta en la DCB: se usa el registro del emulador (fila 0 si vale 0)");
				} else {
					const uint32_t ostride  = (ov[1] >> 16u) & 0x3FFFu;
					const uint32_t orecords = ov[2];
					const uint32_t oindex   = dcb_s10[k] & 0xFFFFu;
					if (ostride != 0 && oindex >= orecords) return "skinning: indice de objeto fuera del V#";
					orow = obase + uint64_t {oindex} * ostride;
					if (!HostMemoryRangeIsReadable(orow, sizeof(d.obj))) return "objeto no legible";
					d.vs = 1;
				}
			}
			std::memcpy(d.obj, reinterpret_cast<const void*>(orow), sizeof(d.obj));
			std::memcpy(d.t3, reinterpret_cast<const void*>(t3 + 0x14), sizeof(d.t3));
			return nullptr;
		}();
		if (why != nullptr && why[0] == '\0') {
			d.skip = 2; // (vacio: no cuenta como omitida)
			++empty;
			info("draws vacios (0 indices en la DCB)");
		} else if (why != nullptr) {
			++skipped;
			entry(why);
		} else {
			d.skip = 0;
		}
	}
	if (skipped + empty == k) return skipped != 0 ? "ninguna entrada dibujable" : "solo draws vacios";
	slot->skipped = skipped;
	slot->count = k;
	return nullptr;
}

// (SALTO, =1; hilo del juego, sin candados) Detras de los paquetes del cubo [begin, end): su estado final, para que la
// GPU pueda saltar el cubo entero (cabecera NOP en el dword reservado) sin cambiar lo que ven los paquetes siguientes.
// Solo si el cubo tiene unicamente: SET_SH_REG / SET_CONTEXT_REG (registros < 0x400), SET_UCONFIG_REG, INDEX_BASE,
// INDEX_TYPE, NUM_INSTANCES, INDEX_BUFFER_SIZE, SET_*_REG_INDIRECT (0x63/0x64/0x9F), draws directos indexados
// (0x35/0x27) y NOP. Cualquier otro paquete: sin
// salto (motivo contado en "info"). Devuelve los dwords escritos (0 con *why = nullptr: cubo sin estado que dejar).
uint32_t NativoSombrasNatSummary(AgcCommandBuffer* commands, const uint32_t* begin, const uint32_t* end, const char** why) {
	namespace H = DsNativoSombras;
	static thread_local uint32_t sh[0x400], cx[0x400];
	static thread_local uint8_t  sh_ok[0x400], cx_ok[0x400];
	static thread_local uint32_t uc_reg[64], uc_val[64];
	uint32_t                     uc_n        = 0;
	const uint32_t*              last_pkt[4] = {nullptr, nullptr, nullptr, nullptr}; // (0x26, 0x2A, 0x2F, 0x13)
	// (0x63 / 0x64 / 0x9F, log 20261125-090852-271: 0x64 = 62961 y 0x9F = 18899 cubos sin salto.) Los indirectos se
	// repiten tal cual (5 dwords, en su orden; la GPU lee la misma tabla) ANTES del estado directo: exacto salvo si un
	// registro directo es anterior a un indirecto que lo vuelve a poner (entonces el directo ganaria): se comprueba con
	// los desplazamientos de la tabla y, si pasa, sin salto.
	constexpr uint32_t           MaxInd      = 32;
	const uint32_t*              ind[MaxInd];
	uint32_t                     ind_n       = 0;
	*why                                     = nullptr;
	NativoSombrasMemoNew();
	std::memset(sh_ok, 0, sizeof(sh_ok));
	std::memset(cx_ok, 0, sizeof(cx_ok));
	const uint64_t dw = static_cast<uint64_t>(end - begin);
	if (dw == 0 || dw > H::NatSkipMaxDw) return *why = "salto: cubo de mas de 0x4000 dwords (va draw a draw)", 0u;
	for (const uint32_t* p = begin; p < end;) {
		const uint32_t h = *p;
		if (h == H::NatSkipFiller) { ++p; continue; }
		if ((h >> 30u) != 3u) return *why = "salto: paquete no tipo 3", 0u;
		const uint32_t len = ((h >> 16u) & 0x3fffu) + 2u;
		const uint32_t op  = (h >> 8u) & 0xffu;
		if (p + len > end) return *why = "salto: paquete cortado", 0u;
		switch (op) {
			case 0x10: case 0x35: case 0x27: break;
			case 0x76: case 0x69: {
				if ((h & 0xffu) != 0 || len < 3 || (p[1] >> 16u) != 0) return *why = "salto: SET_SH/CONTEXT_REG con indice o bits", 0u;
				const uint32_t reg0 = p[1];
				if (reg0 + (len - 2) > 0x400u) return *why = "salto: registro >= 0x400", 0u;
				for (uint32_t i = 2; i < len; ++i) {
					const uint32_t r = reg0 + (i - 2);
					if (op == 0x76) sh[r] = p[i], sh_ok[r] = 1;
					else cx[r] = p[i], cx_ok[r] = 1;
				}
				break;
			}
			case 0x79: {
				if ((h & 0xffu) != 0 || len < 3 || (p[1] >> 16u) != 0) return *why = "salto: SET_UCONFIG_REG con indice o bits", 0u;
				for (uint32_t i = 2; i < len; ++i) {
					const uint32_t r = p[1] + (i - 2);
					uint32_t       k = 0;
					while (k < uc_n && uc_reg[k] != r) ++k;
					if (k == uc_n) {
						if (uc_n >= 64) return *why = "salto: demasiados UCONFIG", 0u;
						uc_reg[uc_n++] = r;
					}
					uc_val[k] = p[i];
				}
				break;
			}
			case 0x26: if (len > 4) return *why = "salto: INDEX_BASE raro", 0u; last_pkt[0] = p; break;
			case 0x2a: if (len > 4) return *why = "salto: INDEX_TYPE raro", 0u; last_pkt[1] = p; break;
			// (11-27, log 20261127-053203-329: "paquete no admitido 0x7A" = 67023 cubos sin salto: casi todos.) 0x7A =
			// SET_UCONFIG_REG_INDEX que escribe Agc (AgcDcbSetIndexSize, agc.cpp): solo VGT_INDEX_TYPE (0x243 con los bits de
			// indice 0x20000000). Es el tipo de indice: el ultimo de los dos (0x2A o este) gana, asi que comparte su hueco y
			// se repite tal cual.
			case 0x7a:
				if (len != 3 || (p[1] & 0x0fffffffu) != 0x243u) return *why = "salto: SET_UCONFIG_REG_INDEX no VGT_INDEX_TYPE", 0u;
				last_pkt[1] = p;
				break;
			case 0x2f: if (len > 4) return *why = "salto: NUM_INSTANCES raro", 0u; last_pkt[2] = p; break;
			case 0x13: if (len > 4) return *why = "salto: INDEX_BUFFER_SIZE raro", 0u; last_pkt[3] = p; break;
			case 0x63: case 0x64: case 0x9f: {
				const uint32_t* q = nullptr;
				uint32_t        n = 0;
				if (!NativoSombrasIndirectPairs(p, len, &q, &n)) return *why = "salto: tabla de registros indirecta no legible", 0u;
				if (n == 0) break; // (no pone nada: CpOpIndirect*Regs sale sin tocar el estado)
				for (uint32_t i = 0; i < n; ++i, q += 2) {
					if (q[0] == 0xffffffffu) continue;
					const uint32_t r = NativoSombrasRegOffset(q[0]);
					bool           clash = false;
					if (op == 0x63) clash = r < 0x400u && sh_ok[r] != 0;
					else if (op == 0x9f) clash = r < 0x400u && cx_ok[r] != 0;
					else for (uint32_t k = 0; k < uc_n && !clash; ++k) clash = uc_reg[k] == r;
					if (clash) return *why = "salto: indirecto pisa un registro directo anterior", 0u;
				}
				if (ind_n >= MaxInd) return *why = "salto: demasiados paquetes indirectos", 0u;
				ind[ind_n++] = p;
				break;
			}
			default: return *why = H::NatOpWhy(op), 0u;
		}
		p += len;
	}
	// Tamano y sitio (detras va la marca de fin).
	const auto runs = [](const uint8_t* ok, auto&& emit) {
		for (uint32_t r = 0; r < 0x400u;) {
			if (ok[r] == 0) { ++r; continue; }
			uint32_t n = 0;
			while (r + n < 0x400u && ok[r + n] != 0) ++n;
			emit(r, n);
			r += n;
		}
	};
	uint32_t need = 0;
	runs(sh_ok, [&](uint32_t, uint32_t n) { need += 2 + n; });
	runs(cx_ok, [&](uint32_t, uint32_t n) { need += 2 + n; });
	need += uc_n * 3;
	// (Un indirecto identico a otro posterior sobra: el posterior deja los mismos valores encima.)
	const auto ind_dup = [&](uint32_t i) {
		for (uint32_t j = i + 1; j < ind_n; ++j)
			if (std::memcmp(ind[i], ind[j], 5 * sizeof(uint32_t)) == 0) return true;
		return false;
	};
	for (uint32_t i = 0; i < ind_n; ++i)
		if (!ind_dup(i)) need += 5;
	for (const auto* q: last_pkt)
		if (q != nullptr) need += ((*q >> 16u) & 0x3fffu) + 2u;
	uint32_t* const up        = commands->cursor_up;
	uint32_t* const down      = commands->cursor_down;
	const uint64_t  available = (reinterpret_cast<uintptr_t>(down) - reinterpret_cast<uintptr_t>(up)) / sizeof(uint32_t);
	if (available <= commands->reserved_dw || available - commands->reserved_dw < uint64_t {need} + H::MarkDwords)
		return *why = "salto: sin sitio para el estado final", 0u;
	uint32_t* w = up;
	const auto put = [&](const uint32_t* val, uint32_t opc) {
		return [&, val, opc](uint32_t r, uint32_t n) {
			*w++ = 0xC0000000u | (n << 16u) | (opc << 8u);
			*w++ = r;
			for (uint32_t i = 0; i < n; ++i) *w++ = val[r + i];
		};
	};
	for (uint32_t i = 0; i < ind_n; ++i) {
		if (ind_dup(i)) continue;
		std::memcpy(w, ind[i], 5 * sizeof(uint32_t));
		w += 5;
	}
	runs(cx_ok, put(cx, 0x69u));
	runs(sh_ok, put(sh, 0x76u));
	for (uint32_t k = 0; k < uc_n; ++k) {
		*w++ = 0xC0000000u | (1u << 16u) | (0x79u << 8u);
		*w++ = uc_reg[k];
		*w++ = uc_val[k];
	}
	for (const auto* q: last_pkt) {
		if (q == nullptr) continue;
		const uint32_t len = ((*q >> 16u) & 0x3fffu) + 2u;
		std::memcpy(w, q, len * sizeof(uint32_t));
		w += len;
	}
	commands->cursor_up = w;
	return need;
}

// (=3 / =1) El gancho ligero: marca de inicio, vista de la cascada (antes de volver: el bloque se reutiliza), la funcion
// del juego, los datos de cada draw a la ranura y la marca de fin. Nada de volcados.
uint64_t NativoSombrasNatHook(uint64_t renderer, uint64_t list, uint64_t* ctx, int32_t bucket, int32_t pass) {
	namespace H = DsNativoSombras;
	const DsSombrasAb::Timer cost_hook(DsSombrasAb::CGancho);
	using Libs::Graphics::HostMemoryRangeIsReadable;
	// (DIAG) El codigo del VS de skinning que pidio el hilo de la GPU (dsSombrasNativas.inc): leido aqui, hilo del juego.
	if (H::NatDiag()) H::GuestServeRequests();
	auto* const commands = ctx[0] != 0 ? reinterpret_cast<AgcCommandBuffer*>(*reinterpret_cast<uint64_t*>(ctx[0])) : nullptr;
	const uint64_t first = *reinterpret_cast<const uint64_t*>(list + 0x178 + uint64_t(bucket) * 0x28);
	const uint64_t last  = *reinterpret_cast<const uint64_t*>(list + 0x180 + uint64_t(bucket) * 0x28);
	if (pass > 3 || commands == nullptr || last <= first || (last - first) % 0x20 != 0)
		return g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	if ((last - first) / 0x20 > H::NatMaxDraws) {
		H::NatNoteCapture("mas de NatMaxDraws entradas");
		return g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	}
	const uint32_t seq = H::g_nat_seq.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (!NativoSombrasNatMark(commands, H::NatBegin, pass, seq)) return g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	// (SALTO, =1) Un dword de relleno justo detras de la marca de inicio: la GPU lo cambia por una NOP que cubre el cubo
	// entero si lo dibuja en nativo (ProcessPm4 ya no traduce sus paquetes).
	uint32_t* reserve = nullptr;
	if (H::Mode() == 1) {
		uint32_t* const up    = commands->cursor_up;
		const uint64_t  avail = (reinterpret_cast<uintptr_t>(commands->cursor_down) - reinterpret_cast<uintptr_t>(up)) / sizeof(uint32_t);
		if (avail > commands->reserved_dw + 1u) {
			*up                 = H::NatSkipFiller;
			commands->cursor_up = up + 1;
			reserve             = up;
		}
	}
	static thread_local uint8_t view[H::NatViewBytes];
	const uint64_t              view_addr = ctx[0x100 / 8] & 0xFFFFFFFFFFFFull;
	const bool                  view_ok   = view_addr != 0 && HostMemoryRangeIsReadable(view_addr, sizeof(view));
	if (view_ok) std::memcpy(view, reinterpret_cast<const void*>(view_addr), sizeof(view));
	const uint64_t        bias   = ctx[0xa0 / 8];
	const uint32_t* const before = commands->cursor_up;
	uint32_t* const       bottom = commands->bottom;
	uint64_t result = 0;
	{
		const DsSombrasAb::Timer cost_original(DsSombrasAb::COriginal);
		result = g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	}
	const uint32_t* const after  = commands->cursor_up;
	H::NatSlot*           slot   = nullptr;
	if (view_ok && commands->bottom == bottom && after > before && static_cast<uint64_t>(after - before) < (uint64_t {1} << 24u))
		slot = H::NatClaim(seq);
	if (slot != nullptr) NativoSombrasShCatchUp(commands, before);
	const char* why = !view_ok                       ? "vista de la cascada no legible"
	                  : commands->bottom != bottom   ? "la DCB cambio de bloque"
	                  : !(after > before)            ? "sin paquetes"
	                  : slot == nullptr              ? "sin ranura libre"
	                                                 : NativoSombrasNatFill(slot, ctx, first, last, before, after);
	// (SALTO) El estado final del cubo detras de sus paquetes; sin el, la GPU no puede saltarlo (va draw a draw).
	uint64_t skip_word = 0;
	uint32_t skip_dw   = 0;
	if (why == nullptr && reserve != nullptr && reserve + 1 == before && commands->bottom == bottom) {
		const char* skip_why = nullptr;
		if (NativoSombrasNatSummary(commands, before, after, &skip_why) != 0 || skip_why == nullptr) {
			skip_word = reinterpret_cast<uint64_t>(reserve);
			skip_dw   = static_cast<uint32_t>(after - before);
		} else {
			H::NatCount(H::g_nat_info_why, skip_why);
		}
	}
	// (La marca de fin, siempre; la ranura solo se publica con ella escrita: =1 no puede quedarse saltando draws.)
	const bool end_ok = NativoSombrasNatMark(commands, H::NatEnd, pass, seq);
	if (why == nullptr && !end_ok) why = "sin sitio para la marca de fin";
	if (slot != nullptr) {
		if (why == nullptr && end_ok) {
			slot->seq    = seq;
			slot->pass   = static_cast<uint32_t>(pass);
			slot->bucket = static_cast<uint32_t>(bucket);
			slot->bias   = bias;
			slot->block  = reinterpret_cast<uint64_t>(bottom);
			slot->sh_full = NativoSombrasShState().full ? 1u : 0u;
			slot->skip_word = skip_word;
			slot->skip_dw   = skip_dw;
			std::memcpy(slot->view, view, sizeof(view));
			H::g_nat_cubes.fetch_add(1, std::memory_order_relaxed);
			if (slot->skipped == 0) H::g_nat_cubes_full.fetch_add(1, std::memory_order_relaxed);
			slot->state.store(2u, std::memory_order_release);

		} else {
			H::NatRelease(slot);
		}
	}
	if (why != nullptr) H::NatNoteCapture(why);
	return result;
}

// (F5 pase 3, pieza 0; local/ds-censo-escena.h) KYTY_NATIVO_PASE_ESCENA=2: pase < 0 (escenario y personajes). Marca CENS
// delante y detras de los draws de la llamada; el pase es la funcion que llamo (direccion de retorno). Solo mide.
bool NativoEscenaMark(AgcCommandBuffer* commands, uint32_t kind, uint32_t pass, uint32_t seq, uint32_t draws) {
	namespace C = DsCensoEscena;
	uint32_t* const up   = commands->cursor_up;
	uint32_t* const down = commands->cursor_down;
	if (up == nullptr || down == nullptr || reinterpret_cast<uintptr_t>(down) <= reinterpret_cast<uintptr_t>(up)) return false;
	const uint64_t available = (reinterpret_cast<uintptr_t>(down) - reinterpret_cast<uintptr_t>(up)) / sizeof(uint32_t);
	if (available <= commands->reserved_dw || available - commands->reserved_dw < C::MarkDwords) return false;
	up[0] = C::MarkHeader;
	up[1] = C::Magic0;
	up[2] = C::Magic1;
	up[3] = kind | (pass << 8u);
	up[4] = seq;
	up[5] = draws;
	commands->cursor_up = up + C::MarkDwords;
	return true;
}

uint64_t NativoEscenaCensus(uint64_t ret, uint64_t renderer, uint64_t list, uint64_t* ctx, int32_t bucket, int32_t pass) {
	namespace C = DsCensoEscena;
	const uint64_t off = ret - g_nativo_sombras_base;
	// (w8) pase >= 0 = sombras (solo con KYTY_NATIVO_PASE_SOMBRAS apagado): su propia fila por cascada.
	const uint32_t p   = pass >= 0 ? C::ShadowFirst + uint32_t(pass < 7 ? pass : 7) : C::PassOf(off);
	if (p == C::Other) C::g_last_other.store(off, std::memory_order_relaxed);
	auto& acc = C::g_acc[p];
	acc.calls.fetch_add(1, std::memory_order_relaxed);
	if (list != 0 && bucket >= 0 && bucket <= 7) {
		const uint64_t first = *reinterpret_cast<const uint64_t*>(list + 0x178 + uint64_t(bucket) * 0x28);
		const uint64_t last  = *reinterpret_cast<const uint64_t*>(list + 0x180 + uint64_t(bucket) * 0x28);
		if (last > first && (last - first) % 0x20 == 0 && (last - first) / 0x20 < 100000) {
			uint64_t technique = 0, techniques = 0;
			for (uint64_t e = first; e < last; e += 0x20) { // (entrada+8 = item; item[0] = material; material+0x20 = tecnica)
				const auto* item = reinterpret_cast<const uint64_t*>(*reinterpret_cast<const uint64_t*>(e + 8));
				const uint64_t t = (item != nullptr && item[0] != 0) ? *reinterpret_cast<const uint64_t*>(item[0] + 0x20) : 0;
				if (t != technique) { ++techniques; technique = t; }
			}
			acc.entries.fetch_add((last - first) / 0x20, std::memory_order_relaxed);
			acc.techniques.fetch_add(techniques, std::memory_order_relaxed);
		}
	}
	auto* const commands = (ctx != nullptr && ctx[0] != 0) ? reinterpret_cast<AgcCommandBuffer*>(*reinterpret_cast<uint64_t*>(ctx[0])) : nullptr;
	const uint32_t seq = C::g_seq.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (commands == nullptr || !NativoEscenaMark(commands, C::Begin, p, seq, 0))
		return g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	const uint32_t* const before = commands->cursor_up;
	const uint64_t result = g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	uint64_t draws = 0;
	if (commands->cursor_up >= before) draws = NativoSombrasCountDraws(before, commands->cursor_up); // (mismo bufer)
	acc.dcb_draws.fetch_add(draws, std::memory_order_relaxed);
	NativoEscenaMark(commands, C::End, p, seq, static_cast<uint32_t>(draws)); // (sin sitio: la GPU descarta el inicio)
	return result;
}

KYTY_SYSV_ABI uint64_t NativoSombrasHook(uint64_t renderer, uint64_t list, uint64_t* ctx, int32_t bucket, int32_t pass) {
	if (DsCensoEscena::Mode() != 0 && (pass < 0 || DsNativoSombras::Mode() == 0)) // (w8: tambien las sombras)
		return NativoEscenaCensus(reinterpret_cast<uint64_t>(__builtin_return_address(0)), renderer, list, ctx, bucket, pass);
	if (DsNativoSombras::Mode() == 0) return g_nativo_sombras_original(renderer, list, ctx, bucket, pass); // (solo censo)
	// (T32 11-27: en el tramo "juego" del A/B el gancho seguia recorriendo la DCB y rellenando ranuras: el juego pagaba el
	// coste del nativo y el A/B no valia. Ahora ese tramo es el juego tal cual. Seconds() y Juego(): lecturas de un static
	// y de un atomico.)
	if (DsSombrasAb::Seconds() != 0 && DsSombrasAb::Juego()) return g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	if ((DsNativoSombras::Mode() == 1 || DsNativoSombras::Mode() == 3) && pass >= 0 && list != 0 && ctx != nullptr &&
	    bucket >= 0 && bucket <= 7)
		return NativoSombrasNatHook(renderer, list, ctx, bucket, pass); // (=3 / =1: sin volcados)
	namespace H = DsNativoSombras;
	if (pass < 0 || list == 0 || ctx == nullptr || bucket < 0 || bucket > 7)
		return g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	H::Add(H::Calls);
	H::GuestServeRequests(); // (lo que pidio la GPU: leido aqui, en un hilo del juego y sin candados)
	NativoSombrasDumpCascade(ctx, pass);
	// La lista: entradas de 0x20 bytes; entrada+8 = item; item[0] = material (+0x20 tecnica), item[2] = malla.
	uint64_t       first_index = 0; // (V# del indice del CMesh de la primera entrada, CMesh+8: base de 48 bits)
	const uint64_t first = *reinterpret_cast<const uint64_t*>(list + 0x178 + uint64_t(bucket) * 0x28);
	const uint64_t last  = *reinterpret_cast<const uint64_t*>(list + 0x180 + uint64_t(bucket) * 0x28);
	uint64_t       entries = 0;
	if (last > first && (last - first) % 0x20 == 0 && (last - first) / 0x20 < 100000) {
		entries = (last - first) / 0x20;
		uint64_t technique = 0, materials = 0, known = 0, native = 0, skinned = 0, native_static = 0;
		std::array<std::pair<uint64_t, uint64_t>, 32> want {};
		size_t want_count = 0;
		auto&  r = MallaNativa::Get();
		// (11-24 congelacion al cruzar la niebla, pila del blackbox 20261124-211031-259: este gancho leia memoria del juego
		// CON el candado de MALLA-NATIVA tomado; un fallo de pagina (BufferCache::ReadMemory -> SendCommandSync) esperaba
		// a la GPU, y la GPU esperaba ese candado en ObtainBuffer -> Lookup. Ahora la memoria del juego se lee SIN candado,
		// por trozos, y el candado solo cubre las busquedas en el mapa (memoria del emulador).)
		struct Seen {
			uint64_t inst, sub, mesh, block;
			bool     skin;
		};
		constexpr size_t   Chunk = 128;
		std::array<Seen, Chunk> seen;
		std::array<uint8_t, Chunk> state {}; // (0 no registrada; 1 registrada; 2 con VkBuffer o preparada)
		for (uint64_t e0 = first; e0 < last; e0 += Chunk * 0x20) {
			size_t n = 0;
			for (uint64_t e = e0; e < last && e < e0 + Chunk * 0x20; e += 0x20) {
				const auto* item = reinterpret_cast<const uint64_t*>(*reinterpret_cast<const uint64_t*>(e + 8));
				if (item == nullptr) continue;
				const uint64_t t = item[0] != 0 ? *reinterpret_cast<const uint64_t*>(item[0] + 0x20) : 0;
				if (t != technique) { ++materials; technique = t; }
				// (Ghidra 0xabc100 / 0x9f80a0 / 0xaa1e20) item[2] = instancia de modelo; submalla = *(inst+0x130) +
				// entrada.0x14 * 0x48; el CMesh de 0xaa06a0 = *(*(sub+0x20)+0x70) (0xaa1e20 enlaza sus CBuffer, CMesh+8).
				const uint64_t inst = item[2];
				if (inst == 0) continue;
				const uint64_t subs = *reinterpret_cast<const uint64_t*>(inst + 0x130);
				const int32_t  si   = *reinterpret_cast<const int32_t*>(e + 0x14);
				if (subs == 0 || si < 0) continue;
				const uint64_t sub = subs + uint64_t(si) * 0x48;
				// (0x9f80a0) sub+0x38 = objeto de animacion; solo si max(+0x1c, +0x20) != 0 (sellos de fotograma) los
				// vertices salen de sub+0x30 (uno por fotograma); con 0 usa los estaticos de CMesh+0x238.
				const uint64_t anim = *reinterpret_cast<const uint64_t*>(sub + 0x38);
				bool           skin = false;
				if (anim != 0) {
					const auto* stamps = reinterpret_cast<const uint32_t*>(anim + 0x1c);
					skin = (stamps[0] < stamps[1] ? stamps[1] : stamps[0]) != 0;
				}
				if (skin) ++skinned;
				const uint64_t part = *reinterpret_cast<const uint64_t*>(sub + 0x20);
				const uint64_t mesh = part != 0 ? *reinterpret_cast<const uint64_t*>(part + 0x70) : 0;
				if (mesh == 0) continue;
				if (first_index == 0) {
					const auto* v = reinterpret_cast<const uint32_t*>(mesh + 8);
					first_index   = uint64_t {v[0]} | (uint64_t {v[1] & 0xFFFFu} << 32u);
				}
				const uint64_t block = *reinterpret_cast<const uint64_t*>(mesh + MallaNativa::VertexBlock);
				if (block == 0) continue;
				seen[n++] = {inst, sub, mesh, block, skin};
			}
			{
				std::shared_lock lock(r.mutex); // (solo el mapa: ninguna lectura de memoria del juego aqui dentro)
				for (size_t k = 0; k < n; ++k) {
					const auto it = r.blocks.find(seen[k].block);
					state[k]      = 0;
					if (it == r.blocks.end()) continue;
					const auto& b = it->second;
					state[k]      = (b.gpu != nullptr || b.pending != nullptr) ? 2 : 1;
					if (state[k] == 1 && !b.queued && !b.failed && b.budget_fails == 0 && want_count < want.size())
						want[want_count++] = {seen[k].block, b.gen}; // (las sombras no pasan por Lookup: se pide aqui)
				}
			}
			for (size_t k = 0; k < n; ++k) {
				if (state[k] == 0) continue;
				++known;
				if (state[k] == 2) {
					++native;
					if (!seen[k].skin) ++native_static;
				}
				if (!seen[k].skin) { // (volcados: sin candado)
					NativoSombrasDump(ctx, pass, seen[k].inst, seen[k].sub, seen[k].mesh, seen[k].block,
					                  state[k] == 2 ? reinterpret_cast<const void*>(1) : nullptr);
					NativoSombrasDumpObject(pass, seen[k].inst, seen[k].mesh);
				}
			}
		}
		// (MALLA-NATIVA) Pedir al hilo Preparer los bloques de sombra aun sin preparar (como hace Lookup en su primer uso).
		if (want_count != 0) {
			std::unique_lock w(r.mutex);
			uint64_t         queued = 0;
			for (size_t k = 0; k < want_count; ++k) {
				auto wi = r.blocks.find(want[k].first);
				if (wi == r.blocks.end() || wi->second.gen != want[k].second || wi->second.queued || wi->second.failed ||
				    wi->second.gpu != nullptr || wi->second.pending != nullptr)
					continue;
				wi->second.queued = true;
				r.requested.fetch_add(1, std::memory_order_relaxed);
				{
					std::lock_guard q(r.queue_mutex);
					r.queue.emplace_back(wi->first, wi->second.gen);
				}
				++queued;
			}
			w.unlock();
			if (queued != 0) r.queue_cv.notify_one();
			H::Add(H::Requested, queued);
		}
		H::Add(H::Entries, entries);
		H::Add(H::Materials, materials);
		H::Add(H::MeshKnown, known);
		H::Add(H::MeshNative, native);
		H::Add(H::Skinned, skinned);
		H::Add(H::NativeStatic, native_static);
		H::g_cascade_entries[pass & 7].fetch_add(entries, std::memory_order_relaxed);
	}
	// La DCB: ctx[0] = contexto, contexto[0] = command buffer (como en SSRClear).
	auto* const     commands = ctx[0] != 0 ? reinterpret_cast<AgcCommandBuffer*>(*reinterpret_cast<uint64_t*>(ctx[0])) : nullptr;
	// (volcado GPU) La marca delante de los draws de esta cascada, hasta que la GPU la vuelque (=2).
	// (solo con lista no vacia y su indice conocido: la marca de un cubo vacio no tiene draw propio detras)
	// (Diseno =3: el pase nativo usara la matriz del objeto del bufer de instancias, tabla[2] del VS, como el VS.)
	if (commands != nullptr && pass <= 3 && entries != 0 && first_index != 0 &&
	    (H::g_gpu_done_mask.load(std::memory_order_relaxed) & (1u << pass)) == 0) {
		uint32_t* const up   = commands->cursor_up;
		uint32_t* const down = commands->cursor_down;
		if (up != nullptr && down != nullptr && reinterpret_cast<uintptr_t>(down) > reinterpret_cast<uintptr_t>(up)) {
			const uint64_t available = (reinterpret_cast<uintptr_t>(down) - reinterpret_cast<uintptr_t>(up)) / sizeof(uint32_t);
			if (available > commands->reserved_dw && available - commands->reserved_dw >= H::MarkDwords) {
				up[0]               = H::MarkHeader;
				up[1]               = H::MarkMagic0;
				up[2]               = H::MarkMagic1;
				up[3]               = static_cast<uint32_t>(pass);
				up[4]               = static_cast<uint32_t>(first_index);
				up[5]               = static_cast<uint32_t>(first_index >> 32u);
				commands->cursor_up = up + H::MarkDwords;
			}
		}
	}
	uint32_t* const before   = commands != nullptr ? commands->cursor_up : nullptr;
	uint32_t* const bottom   = commands != nullptr ? commands->bottom : nullptr;
	// (opcion 2) La camara antes de los draws, para comparar con lo que apunta la tabla del VS despues.
	static thread_local uint8_t cam_before[0x720];
	bool                        have_before = false;
	if (entries != 0 && pass >= 0 && pass <= 3) {
		const uint64_t cam = ctx[0xb8 / 8] & 0xFFFFFFFFFFFFull;
		if (cam != 0 && Libs::Graphics::HostMemoryRangeIsReadable(cam, sizeof(cam_before))) {
			std::memcpy(cam_before, reinterpret_cast<const void*>(cam), sizeof(cam_before));
			have_before = true;
		}
	}
	const uint64_t  result   = g_nativo_sombras_original(renderer, list, ctx, bucket, pass);
	if (entries != 0) NativoSombrasDumpCameraAfter(ctx, pass, bucket); // (sin candados)
	if (entries != 0) NativoSombrasDumpCascadeView(pass); // (sin candados)
	if (entries != 0 && before != nullptr && commands != nullptr && commands->bottom == bottom && commands->cursor_up > before)
		NativoSombrasDumpUserTable(ctx, pass, before, commands->cursor_up, have_before ? cam_before : nullptr); // (sin candados)
	uint32_t* const after    = commands != nullptr ? commands->cursor_up : nullptr;
	if (before != nullptr && after != nullptr && commands->bottom == bottom && after >= before &&
	    static_cast<uint64_t>(after - before) < (uint64_t {1} << 24u)) {
		const uint64_t draws = NativoSombrasCountDraws(before, after);
		H::Add(H::DcbChecked);
		H::Add(H::DcbDraws, draws);
		if (draws == entries) H::Add(H::DcbMatch);
	} else {
		H::Add(H::DcbOther);
	}
	H::Report();
	return result;
}

// (w8, local/ds-censo-dcb.h) KYTY_CENSO_DCB=1: gancho en la entrada de RenderThread::KickoffSubmitJob 0xa08ae0
// (renderer, SRenderJobParams&, modo). Apunta que DCB y ACB va a enviar y quien la pidio (direccion de retorno), y
// ejecuta la funcion entera del juego por un trampolin. No cambia nada de lo que hace el juego.
using CensoKickFunction = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t, int32_t);
CensoKickFunction g_censo_kick_original   = nullptr;
uint64_t          g_censo_kick_trampoline = 0;
constexpr uint64_t CensoKickOffset = 0xa08ae0;
// 0xa08ae0 de 01.007.000 (projelf_par1): push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,..
constexpr std::array<uint8_t, 16> CensoKickCode {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec};
constexpr size_t CensoKickMoved = 12;

KYTY_SYSV_ABI void CensoKickHook(uint64_t renderer, uint64_t params, int32_t mode) {
	DsCensoDcb::OnKickoff(g_nativo_sombras_base, reinterpret_cast<uint64_t>(__builtin_return_address(0)) - g_nativo_sombras_base, mode);
	g_censo_kick_original(renderer, params, mode);
}

void InstallCensoKickHook(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (!DsCensoDcb::On() || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	const auto refuse = [](const char* why) {
		std::printf("[censo-dcb] gancho KickoffSubmitJob NO instalado: %s; el censo usa solo la pila\n", why);
		std::fflush(stdout);
	};
	if (program->mapped_size < 0x2f65f60) return refuse("the eboot is too short to be 01.007.000's");
	const auto address = program->base_vaddr + CensoKickOffset;
	g_nativo_sombras_base = program->base_vaddr;
	const auto* code = reinterpret_cast<const uint8_t*>(address);
	if (!Libs::Graphics::HostMemoryRangeIsReadable(address, CensoKickCode.size())) return refuse("its code is not readable");
	for (size_t i = 0; i < CensoKickCode.size(); ++i)
		if (code[i] != CensoKickCode[i]) return refuse("eboot+0xa08ae0 is not 01.007.000's (or another hook is there)");
	if (g_censo_kick_trampoline == 0)
		g_censo_kick_trampoline = Libs::LibKernel::Memory::AllocateRuntimeMemory(0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite,
		                                                                         "demons_souls_censo_kick");
	if (g_censo_kick_trampoline == 0) return refuse("no memory for its trampoline");
	std::array<uint8_t, CensoKickMoved + 12> trampoline {};
	std::memcpy(trampoline.data(), CensoKickCode.data(), CensoKickMoved);
	const uint64_t resume          = address + CensoKickMoved;
	trampoline[CensoKickMoved]     = 0x48;
	trampoline[CensoKickMoved + 1] = 0xb8;
	std::memcpy(trampoline.data() + CensoKickMoved + 2, &resume, sizeof(resume));
	trampoline[CensoKickMoved + 10] = 0xff;
	trampoline[CensoKickMoved + 11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(g_censo_kick_trampoline), trampoline.data(), trampoline.size());
	if (!Common::VirtualMemory::FlushInstructionCache(g_censo_kick_trampoline, trampoline.size()))
		return refuse("its trampoline could not be flushed");
	g_censo_kick_original = reinterpret_cast<CensoKickFunction>(g_censo_kick_trampoline);
	std::array<uint8_t, CensoKickMoved> entry {0x48, 0xb8};
	const auto hook = reinterpret_cast<uint64_t>(&CensoKickHook);
	std::memcpy(entry.data() + 2, &hook, sizeof(hook));
	entry[10] = 0xff;
	entry[11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(address), entry.data(), entry.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, entry.size())) {
		std::memcpy(reinterpret_cast<void*>(address), CensoKickCode.data(), CensoKickMoved);
		Common::VirtualMemory::FlushInstructionCache(address, CensoKickMoved);
		return refuse("its entry could not be flushed");
	}
	std::printf("[censo-dcb] gancho instalado en eboot+0xa08ae0 (KickoffSubmitJob): DCB y ACB por sitio que pidio el envio\n");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

void InstallNativoSombrasHook(Program* program) {
	InstallCensoKickHook(program); // (w8, KYTY_CENSO_DCB=1; independiente del resto)
#if defined(__x86_64__) || defined(_M_X64)
	if ((DsNativoSombras::Mode() == 0 && DsCensoEscena::Mode() == 0) || program == nullptr ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	const auto refuse = [](const char* why) {
		std::printf("[nativo-sombras] gancho NO instalado (KYTY_NATIVO_PASE_SOMBRAS): %s; nada cambia\n", why);
		std::fflush(stdout);
	};
	if (program->mapped_size < NativoSombrasOffset + NativoSombrasCode.size())
		return refuse("the eboot is too short to be 01.007.000's");
	const auto  address = program->base_vaddr + NativoSombrasOffset;
	g_nativo_sombras_base = program->base_vaddr;
	const auto* code    = reinterpret_cast<const uint8_t*>(address);
	if (!Libs::Graphics::HostMemoryRangeIsReadable(address, NativoSombrasCode.size())) return refuse("its code is not readable");
	for (size_t i = 0; i < NativoSombrasCode.size(); ++i)
		if (code[i] != NativoSombrasCode[i])
			return refuse("eboot+0xab57d0 is not 01.007.000's (another build, or KYTY_PORT_R / KYTY_PORTG already hooked it)");
	if (g_nativo_sombras_trampoline == 0)
		g_nativo_sombras_trampoline = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_nativo_sombras");
	if (g_nativo_sombras_trampoline == 0) return refuse("no memory for its trampoline");
	std::array<uint8_t, NativoSombrasMoved + 12> trampoline {};
	std::memcpy(trampoline.data(), NativoSombrasCode.data(), NativoSombrasMoved);
	const uint64_t resume              = address + NativoSombrasMoved;
	trampoline[NativoSombrasMoved]     = 0x48;
	trampoline[NativoSombrasMoved + 1] = 0xb8;
	std::memcpy(trampoline.data() + NativoSombrasMoved + 2, &resume, sizeof(resume));
	trampoline[NativoSombrasMoved + 10] = 0xff;
	trampoline[NativoSombrasMoved + 11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(g_nativo_sombras_trampoline), trampoline.data(), trampoline.size());
	if (!Common::VirtualMemory::FlushInstructionCache(g_nativo_sombras_trampoline, trampoline.size()))
		return refuse("its trampoline could not be flushed");
	g_nativo_sombras_original = reinterpret_cast<NativoSombrasFunction>(g_nativo_sombras_trampoline);
	std::array<uint8_t, NativoSombrasMoved> entry {0x48, 0xb8};
	const auto                              hook = reinterpret_cast<uint64_t>(&NativoSombrasHook);
	std::memcpy(entry.data() + 2, &hook, sizeof(hook));
	entry[10] = 0xff;
	entry[11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(address), entry.data(), entry.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, entry.size())) {
		std::memcpy(reinterpret_cast<void*>(address), NativoSombrasCode.data(), NativoSombrasMoved);
		Common::VirtualMemory::FlushInstructionCache(address, NativoSombrasMoved);
		return refuse("its entry could not be flushed");
	}
	std::printf("[nativo-sombras] gancho instalado (KYTY_NATIVO_PASE_SOMBRAS=%u: eboot+0xab57d0; %s) skinning=%u "
	            "(KYTY_NATIVO_SOMBRAS_SKIN) censo-escena=%u (KYTY_NATIVO_PASE_ESCENA)\n", DsNativoSombras::Mode(),
	            DsNativoSombras::Mode() == 2 ? "solo mide" : DsNativoSombras::Mode() == 3 ? "dibujo nativo aparte + comparacion" : "reemplaza las cascadas validadas",
	            DsNativoSombras::NatSkinOn() ? 1u : 0u, DsCensoEscena::Mode());
	std::fflush(stdout);
#else
	(void)program;
#endif
}
