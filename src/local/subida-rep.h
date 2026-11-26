#pragma once
// (SUBIDA-REP3, 11-24) KYTY_SUBIDA_REPARTO (por defecto 1; 0 lo apaga): refresco por capas CAMBIADAS de las texturas
// GRANDES por capas (1 nivel) que la CPU reescribe y que ya tenian contenido en la GPU (TextureCache::SubidaRepartida).
//
// Por que (log 20261124-054756-507): dos refrescos ENTEROS de arrays de 360 capas (135 y 180 MB, kind=cpu,
// tracked_before=1): ObtainBufferForImage path=large 95 y 62 ms en el hilo de render.
//
// La version anterior (SUBIDA-REP/REP2) repartia la subida en varios fotogramas y dejaba la imagen SUCIA entre medias:
// con la imagen sucia, los ~3500 draws por tabla de cada fotograma no pueden reutilizar su resolucion y movia la epoca
// de imagenes cada fotograma (log 20261124-074406-141: queue_run 65-78 ms con draw_us < 1 ms; sin ella 10 ms, log
// 20261124-075006-227). Ahora no se aplaza nada: el bind sale con la imagen limpia, igual que el camino normal.
//
// Que hace: al subirse entera por el camino normal, se guarda un hash por capa (XXH3) de sus bytes del juego. Cuando
// la CPU la vuelve a ensuciar, se vigilan sus paginas, se hashea cada capa sucia y se suben SOLO las capas cuyo hash
// cambio (UploadImagePartial por tramos contiguos); las demas ya estan iguales en la GPU. Ni budget, ni espera, ni
// MoveImageStateEpoch. En el camino normal de los draws: una comprobacion de "sucia" (0 ms).
// Cada 30 s (solo si hubo refrescos): [subida-rep] capas subidas / revisadas, MB evitados y ms.
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace SubidaRep {

inline uint64_t EnvU(const char* name, uint64_t def) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] != '\0' ? static_cast<uint64_t>(std::strtoull(v, nullptr, 10)) : def;
}
inline bool On() {
	static const bool on = [] {
		const bool set = EnvU("KYTY_SUBIDA_REPARTO", 0) /* (11-26 limpieza) apagado por defecto */ != 0;
		std::printf("[subida-rep] subida solo de capas cambiadas (texturas grandes por capas): %s (KYTY_SUBIDA_REPARTO=%d)\n",
		            set ? "on" : "off", set ? 1 : 0);
		std::fflush(stdout);
		return set;
	}();
	return on;
}
inline uint64_t MinBytes() { static const uint64_t v = EnvU("KYTY_SUBIDA_MIN_MB", 16) << 20u; return v; }

inline int64_t NowUs() {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

struct Capas {
	uint64_t              key = 0; // VkImage de cuando se hashearon (otro = recreada: no vale)
	std::vector<uint64_t> hash;    // hash de los bytes del juego de cada capa = lo que hay en la GPU
};

// Hilo de render (bajo el lock de la cache de texturas).
struct Estado {
	std::unordered_map<uint64_t, Capas> capas; // serial -> hashes por capa
	int64_t  win_start = 0;
	uint64_t refrescos = 0, revisadas = 0, subidas = 0, bytes_evitados = 0, us = 0, max_us = 0, base_us = 0, bases = 0;
};
inline Estado& S() { static Estado s; return s; }

// Tras cada refresco (solo en este camino, nunca en los draws normales).
inline void Resumen() {
	auto&         s   = S();
	const int64_t now = NowUs();
	if (s.win_start == 0) s.win_start = now;
	if (now - s.win_start < 30000000) return;
	std::printf("[subida-rep] 30 s: %llu refrescos, capas subidas %llu de %llu revisadas (%llu MB evitados), %.1f ms "
	            "(max %.1f ms); bases de hash %llu (%.1f ms)\n",
	            static_cast<unsigned long long>(s.refrescos), static_cast<unsigned long long>(s.subidas),
	            static_cast<unsigned long long>(s.revisadas), static_cast<unsigned long long>(s.bytes_evitados >> 20u),
	            static_cast<double>(s.us) / 1000.0, static_cast<double>(s.max_us) / 1000.0,
	            static_cast<unsigned long long>(s.bases), static_cast<double>(s.base_us) / 1000.0);
	std::fflush(stdout);
	s.win_start = now;
	s.refrescos = s.revisadas = s.subidas = s.bytes_evitados = s.us = s.max_us = s.base_us = s.bases = 0;
	if (s.capas.size() > 4096) s.capas.clear();
}

template <typename Handle>
inline uint64_t Key(const Handle& h) {
	return std::bit_cast<uint64_t>(h);
}

} // namespace SubidaRep
