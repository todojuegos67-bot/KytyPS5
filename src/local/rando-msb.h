#pragma once

// (randomizer, fase 1) Barajado de enemigos dentro del mismo mapa. El juego abre
// dvdroot/map/mapstudio/mXX_YY_00_00.msb (MSB de FromSoftware, 64 bits LE) y crea los enemigos desde sus partes; los
// visuales de Bluepoint se enlazan a esas partes (BPDsModel::TryInit con BP_MSB_PARTS_EXTRA). Se baraja en memoria,
// tras leerse el archivo; el archivo del disco no se toca.
//
// Qué se cambia: el SITIO, no el tipo. Entre los enemigos elegibles de un mismo MSB se permutan la posición + rotación
// (parte +0x20, 24 bytes), el índice de colisión (typedata +0x1C) y los puntos de patrulla (typedata +0x20, 6 int32).
// Modelo, NpcParam, think y entity id quedan juntos: el enemigo es el mismo, con sus recursos ya cargados por el mapa.
//
// Elegibles: parte de tipo 2 (enemigo), EventEntityId == -1 (ningún evento lo nombra), no citado por ningún evento
// (índice de parte), modelo c2xxx / c3xxx. Fuera: c0000 (fantasmas/NPC humanos), c5xxx/c6xxx (jefes, dragones c5120 /
// c5130), c7xxx (NPC). Modo seguro: ante cualquier dato fuera de rango no se cambia nada.
//
// Variables: KYTY_DS_RANDOMIZER=1 + KYTY_DS_RANDOMIZER_SEED (las pone el lanzador), KYTY_DS_RANDOMIZER_SHUFFLE=0 lo
// apaga. (rando23) Por defecto: "mezcla" (BuildMezcla: sitios fijos, cambia el enemigo de cada sitio); KYTY_DS_RANDOMIZER_MODE=fuertes
// vuelve a "fuertes al principio" (BuildFuertes, mueve sitios: lógica + aspecto);
// KYTY_DS_RANDOMIZER_MODE=barajar vuelve a esta fase 1; KYTY_DS_RANDOMIZER_CROSS=1 enciende la fase 2 (otros mundos).

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <map>
#include <chrono>
#include <vector>

namespace RandoMsb {

inline const char* Seed() {
	static const std::string seed = [] {
		const char* v = std::getenv("KYTY_DS_RANDOMIZER_SEED");
		return std::string(v != nullptr ? v : "");
	}();
	return seed.c_str();
}

inline bool Enabled() {
	static const bool on = [] {
		const char* r = std::getenv("KYTY_DS_RANDOMIZER");
		const char* s = std::getenv("KYTY_DS_RANDOMIZER_SHUFFLE");
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_CROSS");
		const bool  enabled = r != nullptr && std::strcmp(r, "1") == 0 && Seed()[0] != '\0' &&
		                     !(s != nullptr && std::strcmp(s, "0") == 0);
		if (enabled) {
			std::printf("[rando] barajado en el mismo mapa: activo (semilla %s)\n", Seed());
			const bool mundos = c != nullptr && std::strcmp(c, "1") == 0;
			std::printf("[rando] mundos 1-5 (m02..m06; Nexo y tutorial originales), mezcla en los mismos sitios; otros mundos: %s\n",
			            mundos ? "SI (tipos de otros mundos como copias en los huecos de cada mundo)" : "no");
			std::fflush(stdout);
		}
		return enabled;
	}();
	return on;
}

// (demonio) Experimento KYTY_DEMONIO_PATIO=1 (apagado por defecto): el Vanguard (c1030, NpcParam 103001, su entidad de
// m03_01_00_00 c1030_0000) en el primer patio de 1-1 (m02_00_00_00), en el sitio del soldado c2000_0000 (9.1, 21.2,
// -126.0; colision 766, sus grupos +0x44 se quedan). Ghidra: los modelos de cp11models.crmp se buscan en FUN_008b68a0
// (FNV-1a en mayusculas); "$/cp11demonssouls/dvdroot/chr/c1030/c1030.model?chrid103001" esta. Trazas: creacion
// (0x359990), modelo (0x1451700, la llama BPDsModel::TryInit 0x60a77b), FUN_01473fb0 (crear instancias del cgpr) y
// FUN_01472490 (quitar una entidad de las listas del mundo). Apagado: ni una linea de esto se ejecuta.
// (rando27) Dificultad del randomizer (lanzador, KYTY_DS_RANDOMIZER_DIFICULTAD): 1 facil (~30 % de los huecos con copia),
// 2 intermedio (todos los huecos elegibles, como hasta ahora), 3 dificil (todos + el Vanguard del primer patio).
inline int Dificultad() {
	static const int d = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_DIFICULTAD");
		const int   v = c != nullptr ? std::atoi(c) : 2;
		return v >= 1 && v <= 3 ? v : 2;
	}();
	return d;
}
// (jefes) Candidatos "de calibre Vanguard" para el hueco del patio (datos: NpcParam HP +0x2C, think NpcParam +4 ->
// NpcThinkParam logica/batalla; partes de los MSB sin entity id de jefe, sin charaInit, con entidad en el cgpr de su
// mapa). Jefes de arena (c5xxx, c4050 Flamelurker, c6020 Old Hero, c6050 Dragon God, c3040 Maneater, c7230 Allant,
// c5120/c5130 dragones) fuera: niebla, eventboss.lua, cinematicas. KYTY_DS_JEFE=<chr> fuerza uno (pruebas);
// KYTY_DEMONIO_PATIO=1 = Vanguard fijo; DIFICIL: uno al azar por carga (semilla + numero de carga).
struct JefeDef {
	const char* model;
	const char* map;   // mapa donante (MSB + legacy_instances.cgpr)
	const char* donor; // parte donante
	const char* name;
	int32_t     hp, battle;
};
inline const std::vector<JefeDef>& Jefes() {
	static const std::vector<JefeDef> j = {
	    {"c1030", "m03_01_00_00", "c1030_0000", "Vanguard (c1030_vanguard)", 1800, 103001},
	    {"c3130", "m06_01_00_00", "c3130_0007", "Gran bugbear (c3130_largebugbear)", 1290, 313000},
	    {"c3131", "m06_01_00_00", "c3131_0000", "Bugbear (c3131_bugbear)", 860, 313000},
	};
	return j;
}
inline const char* JefeForzado() { // "" = ninguno
	static const std::string f = [] {
		const char* c = std::getenv("KYTY_DS_JEFE");
		if (c != nullptr && c[0] != '\0') {
			std::string v(c);
			for (const auto& d: Jefes())
				if (v == d.model) return v;
			std::printf("[jefe] KYTY_DS_JEFE=%s no esta en la lista (c1030, c3130, c3131); se ignora\n", c);
			std::fflush(stdout);
		}
		const char* p = std::getenv("KYTY_DEMONIO_PATIO");
		if (p != nullptr && std::strcmp(p, "1") == 0) return std::string("c1030");
		return std::string();
	}();
	return f.c_str();
}
inline bool DemonioOn() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DEMONIO_PATIO");
		const char* r1 = std::getenv("KYTY_DS_RANDOMIZER");
		// KYTY_DEMONIO_PATIO=1 (Vanguard fijo) o KYTY_DS_JEFE=<chr> siguen valiendo; con el randomizer en DIFICIL va siempre.
		const bool  r = JefeForzado()[0] != '\0' ||
		               (r1 != nullptr && std::strcmp(r1, "1") == 0 && Dificultad() == 3 && !(c != nullptr && std::strcmp(c, "0") == 0));
		if (r) {
			std::printf("[demonio] jefe en el primer patio de Boletaria 1-1: %s\n",
			            JefeForzado()[0] != '\0' ? JefeForzado() : "al azar en cada carga (DIFICIL)");
			std::fflush(stdout);
		}
		return r;
	}();
	return on;
}
inline bool Traced() { return Enabled() || DemonioOn(); }

inline std::string MapName(const std::string& path) {
	const auto slash = path.find_last_of("/\\!");
	std::string name = path.substr(slash == std::string::npos ? 0 : slash + 1);
	const auto  dot  = name.rfind('.');
	if (dot != std::string::npos) name.resize(dot);
	return name;
}

inline bool IsMsbPath(const std::string& path) {
	if (path.size() < 5) return false;
	std::string low(path);
	for (auto& ch: low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	return low.compare(low.size() - 4, 4, ".msb") == 0 && low.find("mapstudio") != std::string::npos;
}

namespace Detail {

struct Reader {
	const std::vector<uint8_t>& d;
	bool                        ok = true;

	bool In(uint64_t off, uint64_t len) {
		if (off > d.size() || len > d.size() - off) ok = false;
		return ok;
	}
	int32_t I32(uint64_t off) {
		int32_t v = -1;
		if (In(off, 4)) std::memcpy(&v, d.data() + off, 4);
		return v;
	}
	int64_t I64(uint64_t off) {
		int64_t v = -1;
		if (In(off, 8)) std::memcpy(&v, d.data() + off, 8);
		return v;
	}
	float F32(uint64_t off) {
		float v = 0.0f;
		if (In(off, 4)) std::memcpy(&v, d.data() + off, 4);
		return v;
	}
	std::string CStr(uint64_t off) {
		std::string s;
		while (In(off, 1) && d[off] != 0 && s.size() < 64) s.push_back(static_cast<char>(d[off++]));
		return s;
	}
};

struct Param {
	std::string           name;
	std::vector<uint64_t> entries;
	uint64_t              end = 0; // offset of the next table
};

// Tables in order: MODEL_PARAM_ST, EVENT_PARAM_ST, POINT_PARAM_ST, PARTS_PARAM_ST, MAPSTUDIO_TREE_ST.
// Each: int32 version, int32 ?, int64 name offset, int32 count (entries + 1), int32 pad, then count int64 offsets; the
// last is the next table's offset (0 at the end).
inline bool ReadParams(Reader& r, std::vector<Param>* out) {
	uint64_t off = 0;
	for (int t = 0; t < 8; t++) {
		Param   p;
		const int64_t name_off = r.I64(off + 0x8);
		const int32_t count    = r.I32(off + 0x10);
		if (!r.ok || count < 1 || count > 100000 || name_off <= 0) return false;
		p.name = r.CStr(static_cast<uint64_t>(name_off));
		if (!r.In(off + 0x18, static_cast<uint64_t>(count) * 8)) return false;
		for (int32_t i = 0; i < count - 1; i++) {
			const int64_t e = r.I64(off + 0x18 + static_cast<uint64_t>(i) * 8);
			if (e <= 0 || !r.In(static_cast<uint64_t>(e), 0x10)) return false;
			p.entries.push_back(static_cast<uint64_t>(e));
		}
		const int64_t next = r.I64(off + 0x18 + static_cast<uint64_t>(count - 1) * 8);
		if (!r.ok || next < 0) return false;
		p.end = static_cast<uint64_t>(next);
		out->push_back(std::move(p));
		if (next == 0) break;
		if (static_cast<uint64_t>(next) <= off) return false;
		off = static_cast<uint64_t>(next);
	}
	return r.ok && out->size() >= 4 && (*out)[0].name == "MODEL_PARAM_ST" && (*out)[1].name == "EVENT_PARAM_ST" &&
	       (*out)[3].name == "PARTS_PARAM_ST";
}

inline uint64_t Fnv(const std::string& s, uint64_t h = 1469598103934665603ull) {
	for (unsigned char ch: s) {
		h ^= ch;
		h *= 1099511628211ull;
	}
	return h;
}

inline uint64_t SplitMix(uint64_t* state) {
	uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
	z          = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ull;
	z          = (z ^ (z >> 27u)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31u);
}

inline bool EligibleModel(const std::string& m) {
	if (m.size() != 5 || m[0] != 'c') return false;
	for (int i = 1; i < 5; i++)
		if (std::isdigit(static_cast<unsigned char>(m[i])) == 0) return false;
	return m[1] == '2' || m[1] == '3'; // c2xxx / c3xxx: enemigos normales
}

struct Enemy {
	uint64_t    part; // offset of the part entry
	uint64_t    td;   // offset of its type data
	std::string name, model;
};

} // namespace Detail

// (rando20) Índice de colisión (typedata +0x1C) válido: una parte de colisión (tipo 5) del mismo MSB.
// Ghidra FUN_006c0df0 (0x6c1d1c..0x6c1db3): al cargar el mapa, cada enemigo toma los DrawGroups (+0x44, 16 bytes) de la
// parte que cita su +0x1C; si el índice es -1 o se sale, se queda con los suyos.
inline bool CollOk(Detail::Reader& r, const std::vector<uint64_t>& parts, int32_t coll) {
	if (coll < 0 || static_cast<size_t>(coll) >= parts.size()) return false;
	return r.I32(parts[static_cast<size_t>(coll)] + 0x8) == 5 && r.ok;
}

// Baraja el MSB leído en |bytes| (el archivo entero, desde el offset 0). Devuelve true si cambió algo.
inline bool ShuffleBytes(const std::string& map, std::vector<uint8_t>& bytes) {
	using namespace Detail;
	Reader            r {bytes};
	std::vector<Param> params;
	if (!ReadParams(r, &params)) {
		std::printf("[rando] mapa %s: formato no reconocido, se deja el original\n", map.c_str());
		return false;
	}
	const auto& models = params[0].entries;
	const auto& events = params[1].entries;
	const auto& parts  = params[3].entries;

	// Índices de parte que cita algún evento (bloque común + datos del tipo): esos no se mueven.
	std::set<int32_t> cited;
	for (size_t k = 0; k < events.size(); k++) {
		const uint64_t e   = events[k];
		const uint64_t end = (k + 1 < events.size()) ? events[k + 1] : params[1].end;
		const int64_t  common = r.I64(e + 0x18);
		if (!r.ok || common <= 0 || end <= e + static_cast<uint64_t>(common) || end - e > 0x1000) {
			std::printf("[rando] mapa %s: eventos fuera de rango, se deja el original\n", map.c_str());
			return false;
		}
		for (uint64_t o = e + static_cast<uint64_t>(common); o + 4 <= end; o += 4) cited.insert(r.I32(o));
	}

	std::vector<Enemy> enemies;
	for (size_t i = 0; i < parts.size(); i++) {
		const uint64_t p = parts[i];
		const int64_t  name_off = r.I64(p);
		const int32_t  type     = r.I32(p + 0x8);
		const int32_t  model    = r.I32(p + 0x10);
		const int64_t  ent_off  = r.I64(p + 0x68);
		const int64_t  td_off   = r.I64(p + 0x70);
		if (!r.ok) return false;
		if (type != 2) continue;
		if (name_off <= 0 || ent_off <= 0 || td_off <= 0 || model < 0 || static_cast<size_t>(model) >= models.size()) {
			std::printf("[rando] mapa %s: parte %zu inconsistente, se deja el original\n", map.c_str(), i);
			return false;
		}
		const uint64_t me   = models[static_cast<size_t>(model)];
		const int64_t  mnam = r.I64(me);
		const std::string mname = mnam > 0 ? r.CStr(me + static_cast<uint64_t>(mnam)) : std::string();
		const int32_t  entity   = r.I32(p + static_cast<uint64_t>(ent_off));
		const uint64_t td       = p + static_cast<uint64_t>(td_off);
		if (!r.In(td, 0x38) || !r.In(p + 0x20, 24)) {
			std::printf("[rando] mapa %s: parte %zu fuera de rango, se deja el original\n", map.c_str(), i);
			return false;
		}
		if (entity != -1 || cited.count(static_cast<int32_t>(i)) != 0 || !EligibleModel(mname)) continue;
		bool finite = true;
		for (int f = 0; f < 6; f++) finite = finite && std::isfinite(r.F32(p + 0x20 + f * 4));
		if (!finite) continue;
		// (rando20) Solo sitios con colisión válida: el juego copia a la parte los grupos de dibujo de su colisión
		// (eboot 0x6c1d1c..0x6c1db3); con -1 o un índice malo el enemigo queda con grupos de otra zona: inactivo e
		// invisible, pero su cápsula sigue ahí = muro invisible.
		if (!CollOk(r, parts, r.I32(td + 0x1C))) continue;
		enemies.push_back({p, td, r.CStr(p + static_cast<uint64_t>(name_off)), mname});
	}
	if (!r.ok) return false;
	if (enemies.size() < 2) {
		std::printf("[rando] mapa %s: 0 enemigos cambiados (semilla %s)\n", map.c_str(), Seed());
		return false;
	}

	// Sitio = pos+rot (24 bytes), colisión (4), patrulla (24).
	struct Spot {
		uint8_t place[24];
		uint8_t groups[32]; // (rando20) DrawGroups + DispGroups del sitio (parte +0x44, 2 x 16 bytes)
		uint8_t coll[4];
		uint8_t patrol[24];
	};
	std::vector<Spot> spots(enemies.size());
	for (size_t i = 0; i < enemies.size(); i++) {
		std::memcpy(spots[i].place, bytes.data() + enemies[i].part + 0x20, 24);
		std::memcpy(spots[i].groups, bytes.data() + enemies[i].part + 0x44, 32);
		std::memcpy(spots[i].coll, bytes.data() + enemies[i].td + 0x1C, 4);
		std::memcpy(spots[i].patrol, bytes.data() + enemies[i].td + 0x20, 24);
	}
	std::vector<size_t> perm(enemies.size());
	for (size_t i = 0; i < perm.size(); i++) perm[i] = i;
	uint64_t state = Fnv(map, Fnv(Seed()));
	for (size_t i = perm.size() - 1; i > 0; i--) {
		const size_t j = static_cast<size_t>(SplitMix(&state) % (i + 1));
		std::swap(perm[i], perm[j]);
	}

	static std::mutex            log_mutex;
	static std::set<std::string> logged;
	bool                         first_time = false;
	{
		std::lock_guard<std::mutex> lock(log_mutex);
		first_time = logged.insert(map).second;
	}
	size_t changed = 0;
	for (size_t i = 0; i < enemies.size(); i++) {
		const size_t s = perm[i];
		if (s == i) continue;
		std::memcpy(bytes.data() + enemies[i].part + 0x20, spots[s].place, 24);
		std::memcpy(bytes.data() + enemies[i].part + 0x44, spots[s].groups, 32);
		std::memcpy(bytes.data() + enemies[i].td + 0x1C, spots[s].coll, 4);
		std::memcpy(bytes.data() + enemies[i].td + 0x20, spots[s].patrol, 24);
		changed++;
		if (first_time) {
			std::printf("[rando]   %s (%s) -> sitio de %s (%s)\n", enemies[i].name.c_str(), enemies[i].model.c_str(),
			            enemies[s].name.c_str(), enemies[s].model.c_str());
		}
	}
	std::printf("[rando] mapa %s: %zu enemigos cambiados (semilla %s)\n", map.c_str(), changed, Seed());
	std::fflush(stdout);
	return changed > 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// (randomizer, fase 2) Enemigos de otros mundos.
//
// Lo que se comprobó en los archivos del juego (PPSA01341, cp11demonssouls/dvdroot) y en el ejecutable:
// - Los personajes del mapa salen de la lista de modelos del MSB (MODEL_PARAM_ST, tipo 2): el número de instancias de
//   cada modelo coincide exacto con las partes que lo usan en los 15 MSB de mundo. arealoadlist/loadlist no nombran
//   personajes (solo MSB, mcg, mcp, nvmbnd), así que añadir un modelo a la lista es lo que lo hace cargar.
// - Lo de Bluepoint se busca por el nombre del modelo, no por el mapa: "$/cp11demonssouls/dvdroot/chr/%s/%s.model?chrid%d"
//   (lea en 0x1451c62; las dos %s son el mismo nombre de chr). Los modelos están en app0/characters/cXXXX_*, el sonido
//   en sound/banks/enemies/cXXXX_*, la IA en script/ai/out/*.lua y los parámetros en gameparam.parambnd: todo global.
// - m0X.chresdbnd y m99.chresdbnd solo llevan ESD de jefes y de tres especiales (c3040, c4050, c6050): los enemigos
//   normales no tienen ESD propia (usan enemycommon.esd, global). Por eso no hace falta fusionar ningún BND: basta con
//   no usar esos tipos.
// - Los offsets absolutos del MSB están solo en las cabeceras de tabla y en sus listas de offsets; dentro de cada entrada
//   todo es relativo. Reconstruir los 15 MSB sin cambios da el archivo idéntico byte a byte.
//
// Qué se hace: en cada mapa se eligen con la semilla de 4 a 8 tipos normales de OTROS mundos (lista blanca de tamaño
// humano o menor, que existan su chrbnd y su carpeta de Bluepoint). Sus entradas de modelo se copian de su MSB de origen
// y se insertan al final del grupo de enemigos; los índices de modelo de las partes que van detrás se corrigen y los
// contadores de instancias se recalculan. Una parte de los enemigos elegibles (los mismos de la fase 1) cambia de modelo
// y toma de un enemigo normal de origen los parámetros (typedata +0x00..+0x13 y +0x18: think/NPC, charaInit); conserva
// los del sitio (+0x14, colisión +0x1C, patrulla +0x20). Jefes, NPC, enemigos de evento y los citados por eventos no se
// tocan. Después se aplica la fase 1 (barajado de sitios) al resultado.
//
// Como el MSB crece, no se puede reescribir en el búfer del juego: se escribe una copia en
// %TEMP%/kyty-rando/<semilla>/<mapa>.msb y la lectura (KernelOpen / Apr) y el tamaño (KernelStat / Apr resolve) se
// redirigen a ella. Modo seguro: ante cualquier dato raro, falta de archivo o fallo al escribir, no hay copia y queda la
// fase 1 en memoria como antes.
//
// KYTY_DS_RANDOMIZER_CROSS=0 la apaga; KYTY_DS_RANDOMIZER_CROSS_TYPES (4..8, por defecto 5) y
// KYTY_DS_RANDOMIZER_CROSS_PCT (10..100, por defecto 50: % de enemigos elegibles que cambian de tipo).

inline bool CrossWorldsOn() {
	// (rando18) Apagado por defecto: el juego descarta en silencio a los enemigos de otros mundos (logs 111405 y 112432:
	// lógica y aspecto coinciden y Bluepoint carga sus recursos, pero no aparecen ni atacan). KYTY_DS_RANDOMIZER_CROSS=1
	// lo vuelve a encender para pruebas.
	// (rando24) Por ahora solo el primer mundo (m02) y sin mezclar mundos: apagado siempre, se pida o no.
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_CROSS");
		if (Enabled() && c != nullptr && std::strcmp(c, "1") == 0) {
			std::printf("[rando] enemigos de otros mundos: pedido pero apagado (por ahora solo Boletaria, mismo mapa)\n");
			std::fflush(stdout);
		}
		return false;
	}();
	return on;
}

// (rando18) "Fuertes al principio": dentro del MISMO mapa, los enemigos más fuertes pasan a los sitios más cercanos al
// inicio (la primera parte de jugador del MSB). Se mueven a la vez la lógica (MSB: pos+rot, colisión, patrulla) y el
// aspecto (matrices de la entidad de Bluepoint en legacy_instances.cgpr), para que vayan juntos. Por defecto con el
// randomizer activo y sin cruce de mundos; KYTY_DS_RANDOMIZER_MODE=barajar vuelve al barajado de la fase 1 (solo MSB).
inline bool FuertesOn() {
	static const bool on = [] {
		const char* m = std::getenv("KYTY_DS_RANDOMIZER_MODE");
		const bool  b = m != nullptr && std::strcmp(m, "barajar") == 0;
		const bool  r = Enabled() && !CrossWorldsOn() && !b;
		if (Enabled()) {
			const bool fu = m != nullptr && std::strcmp(m, "fuertes") == 0;
			std::printf("[rando] modo: %s\n", r ? (fu ? "fuertes al principio (mismo mapa, logica + aspecto)"
			                                           : "mezcla (mismo mapa: cada sitio igual, cambia el enemigo que lo ocupa)")
			                                    : (CrossWorldsOn() ? "otros mundos" : "barajar (solo MSB)"));
			std::fflush(stdout);
		}
		return r;
	}();
	return on;
}

// (rando23) "Mezcla" (modo por defecto): ningun sitio se mueve. Posicion, rotacion, colision +0x1C, grupos +0x44,
// patrulla y la matriz raiz de Bluepoint de cada sitio quedan como en el original; solo cambia QUE enemigo ocupa el
// sitio (modelo, NpcParam/think/charaInit y su entidad visual de Bluepoint, anclada a la raiz del sitio). Motivo: con
// sitios movidos salian muros invisibles solo con el randomizer; asi el mundo (colision, grupos, eventos) es identico al
// original. KYTY_DS_RANDOMIZER_MODE=fuertes vuelve al modo anterior.
inline bool MezclaOn() {
	static const bool on = [] {
		const char* m = std::getenv("KYTY_DS_RANDOMIZER_MODE");
		return FuertesOn() && !(m != nullptr && std::strcmp(m, "fuertes") == 0);
	}();
	return on;
}

// Copias del mapa redirigidas (MSB + legacy_instances): con otros mundos o con "fuertes al principio". Es lo que
// consultan fileSystem.cpp y libAmpr.cpp antes de RedirectRead / RedirectedSize.
inline bool CrossOn() { return DemonioOn() || CrossWorldsOn() || FuertesOn(); }

// (rando24) Alcance: el randomizer solo toca los mapas del primer mundo (Boletaria, m02_xx); el resto, original.
// (rando28) Todos los mundos: Boletaria (m02), Stonefang (m03), Latria (m04), Santuario (m05) y Valle (m06). El Nexo
// (m01) y el tutorial (m08) quedan originales: el MSB del Nexo solo tiene NPC (c0000 x14, c7080, c7130, c7150) y c4000_0015,
// todos con entity id de evento: ningun hueco de enemigo normal.
inline bool InScope(const std::string& map) {
	return map.size() >= 4 && (map[0] == 'm' || map[0] == 'M') && map[1] == '0' && map[2] >= '2' && map[2] <= '6' && map[3] == '_';
}

// (rando24) Hueco del demonio: el soldado c2000_0002 (20.5, 21.2, -123.0; colision 766, npc 200010) del primer patio.
// Logs 20261127-040451-674 / 040741-877: c2000_0002 lo crea el juego (0x359990 llamada 2) y c2000_0000 no aparece
// nunca en 0x359990 en ninguna prueba (tampoco con a2e3281). La mezcla deja ese sitio como esta.
inline const char* DemonioTarget() { return "c2000_0002"; }

// (rando24) KYTY_DS_RANDOMIZER_CGPR=0: la mezcla cambia SOLO el MSB (modelo + NpcParam/think); el legacy_instances.cgpr
// va original (sin entidades trasplantadas). Prueba de muros invisibles: si con esto no salen, la causa es el trasplante.
// (rando26) MUNDOS SI en el lanzador (KYTY_DS_RANDOMIZER_CROSS=1): la mezcla tambien usa como donantes enemigos normales
// de OTROS mundos (lista blanca CrossPool, entidad de su legacy_instances.cgpr). Mismo metodo que el Vanguard de 1-1:
// nada se mueve, el MSB del hueco no se toca, el tipo lo pone el gancho de creacion (0x359990 / 0x1451700) y el cgpr
// lleva la entidad del donante anclada al hueco con la colision del suelo y los campos de sitio del hueco (rando25).
inline bool MundosOn() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_CROSS");
		return Enabled() && c != nullptr && std::strcmp(c, "1") == 0;
	}();
	return on;
}

// (rando26) Turnos: cada carga del mapa (muerte, hoguera, volver a la zona) es una mezcla nueva (semilla + numero de
// carga). KYTY_DS_RANDOMIZER_ROTAR=0 vuelve a una sola mezcla por partida.
inline bool RotarOn() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_ROTAR");
		return !(c != nullptr && std::strcmp(c, "0") == 0);
	}();
	return on;
}

// (rando26) Huecos cambiados por gancho (otros mundos o turnos): hacen falta los ganchos 0x359990/0x1451700 y 0xef430.
inline bool MezclaHooksOn() { return MezclaOn() && (MundosOn() || RotarOn()); }

inline bool MezclaCgprOn() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_CGPR");
		const bool  r = !(c != nullptr && std::strcmp(c, "0") == 0);
		if (!r && FuertesOn()) {
			std::printf("[rando] mezcla: solo MSB (KYTY_DS_RANDOMIZER_CGPR=0), legacy_instances original\n");
			std::fflush(stdout);
		}
		return r;
	}();
	return on;
}

// (randomizer, fase 3) Bluepoint enlaza cada parte de enemigo del MSB con la entidad del MISMO NOMBRE de
// scripts/worlds/<mapa>/_cmn/<mapa>_legacy_instances.cgpr (lista en cp11demonssouls/_cmn/cp11instances.crmp, clave
// FNV-1a de "M02_00_00_00"): las 167 partes de enemigo de m02_00 (70 de m02_01, 252 de m05_01, 69 de m03_02) tienen
// su entidad con su nombre. Esa entidad lleva el aspecto del enemigo original (mascara, hombreras, telas, voces): por
// eso un c2000_0000 cambiado a c4020 seguia viendose como soldado. El aspecto por tipo sale de cp11models.crmp con
// la clave FNV-1a de "$/CP11DEMONSSOULS/DVDROOT/CHR/C4020/C4020.MODEL" (o "...?CHRID<NpcParam>"), la cadena de
// eboot 0x24e6c6b (uso en 0x1451c62); existe para los 19 tipos de otros mundos. Un enemigo de otro mundo se
// renombra a "cXXXX_9NNN" (ningun cgpr tiene ese nombre): sin entidad propia, Bluepoint usa el modelo por tipo.
// PROBADO (log 20261123-074644-111): NO sirve. El juego lee la copia del MSB, pero la lista de personajes que carga y
// su aspecto salen solo de legacy_instances.cgpr (entidades completas por nombre, con sus componentes horneados); un
// nombre sin entidad no carga nada nuevo (c4020/c4021/c3071/c2110 nunca se leen). Apagado por defecto;
// KYTY_DS_RANDOMIZER_RENAME=1 lo enciende para pruebas.
inline bool Rename() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_RENAME");
		return c != nullptr && std::strcmp(c, "1") == 0;
	}();
	return on;
}

// Traza (solo con el randomizer activo; KYTY_DS_RANDOMIZER_TRACE=0 la apaga): cada archivo de personaje (chr/ del
// juego original y prefabs de scripts/cp11/characters/), cada legacy_instances y cada MSB que el juego lee, una vez
// por archivo y como mucho 600 lineas. Dice en el log si se cargan los enemigos de otros mundos y si se lee la copia.
// Nivel de traza: KYTY_DS_RANDOMIZER_TRACE=0 apaga, 1 (por defecto) personajes/MSB/legacy, 2 TODOS los archivos que abre
// el juego (hasta 4000) y cada lectura de un MSB o legacy_instances con su desplazamiento y bytes. El 2 es para encontrar
// de donde saca el juego la lista de plantillas de personaje que precarga por mundo.
inline void InstallModelProbe(); // (rando8) mas abajo

// (rando10) Gancho en memoria. Log 20261123-093444-192: ni el MSB ni el cgpr servidos llegan a la tabla de partes del
// mapa en memoria ([map+0x928]/[map+0x860]): la sonda sigue pidiendo c2000/200013... desde eboot+0x366294. Ghidra:
// eboot+0x359990(obj) crea cada enemigo: [obj+8] = mapa, [obj+0x10] = indice de parte, [obj+0x14] = 1 jugador; con
// [map+0x22c]==1 la parte es [map+0x928]+idx*0x80 y su modelo [map+0x860]+[parte+0x10]*0x20 (si no, [map+0x48] y
// [map+0x30], listas con punteros en +0x18); [parte+0]: nombre, [parte+0x70]: typedata (+0xC = NpcParam = chrid).
// Llama a 0x367030 -> 0x366130 -> 0x1451700 (nombre del modelo en rsi, chrid en edx). Gancho A (entrada de 0x359990):
// si la parte es una elegida (nombre, modelo y NpcParam originales), copia en memoria el typedata del donante
// (+0x00..+0x13 think/NPC y +0x18 charaInit) y deja la sustitucion pendiente en el hilo. Gancho B (0x1451700, la sonda):
// si viene de eboot+0x366294 con ese modelo original, cambia rsi por el modelo del donante y edx por su chrid.
// KYTY_DS_RANDOMIZER_HOOK=0 lo apaga.
struct HookSwap {
	bool        has_pos = false; // (demonio) ademas, la parte debe estar en |pos| (+0x20), a menos de 1 m
	float       pos[3] {};
	std::string name;           // parte (c2000_0035)
	char        orig_model[16] {};
	int32_t     orig_npc  = 0;
	char        new_model[16] {};
	int32_t     new_npc   = 0;
	uint8_t     td_head[0x14] {};
	uint8_t     td_init[4] {};
	uint64_t    owner = 0;     // (rando26) mapa que la puso (Fnv del nombre); 0: sin dueño
	bool        dead  = false; // (rando26) de una carga anterior del mapa: ya no vale (no se borra: Pending la apunta)
};
inline std::mutex& HookMutex() {
	static std::mutex m;
	return m;
}
inline std::deque<HookSwap>& HookSwaps() {
	static std::deque<HookSwap> d;
	return d;
}
// (rando13) Tabla UNICA de cambios: la llena BuildCrossDyn con los enemigos que de verdad recibieron la entidad visual
// del donante en la copia del legacy_instances.cgpr (nombre de parte -> donante: modelo, NpcParam, think, typedata).
// El gancho NO decide: busca el nombre de la parte aqui; si no esta, no toca nada. Asi aspecto y logica son siempre el
// mismo enemigo (log 20261123-100826: con DynChoose por separado no coincidia ninguno).
// Clave: nombre de la parte + su indice dentro del tipo (+0xC de la parte, igual en el archivo y en la tabla del juego),
// porque el mismo nombre (c2000_0001) existe en varios mapas. Si el juego no diera ese indice, vale el nombre solo
// cuando es unico entre todos los mapas cambiados.
// (rando16) Clave MAPA + NOMBRE: el mismo nombre (c2000_0000) existe en m02_00, m02_01, m02_02 y m02_03, y en 1-1 el
// juego crea partes de los cuatro a la vez (log 20261123-104542: la tabla de nombre#indice mezclaba mapas). El mapa de
// cada parte creada se saca de la propia parte: (nombre, modelo, indice de modelo +0x10, indice en el tipo +0xC) se
// busca en el indice de TODOS los MSB que el juego ha leido (tal como se sirven: copia o original). Si sale un solo mapa,
// ese puntero de mapa ([obj+8]) queda asociado a ese mapa; si sale en varios, vale el mapa ya asociado al puntero. Si
// no se puede saber de forma inequivoca, la parte NO se cambia (nunca aspecto de un tipo y logica de otro).
inline std::string DynKey(const std::string& map, const std::string& name) { return map + "/" + name; }
inline std::map<std::string, HookSwap>& DynTable() {
	static std::map<std::string, HookSwap> t;
	return t;
}
inline std::string DynSig(const std::string& name, const std::string& model, int32_t mi, int32_t tix) {
	return name + "|" + model + "|" + std::to_string(mi) + "|" + std::to_string(tix);
}
inline std::map<std::string, std::set<std::string>>& DynSigIndex() { // firma de la parte -> mapas que la tienen
	static std::map<std::string, std::set<std::string>> t;
	return t;
}
inline std::map<uint64_t, std::string>& DynPtrMap() { // puntero de mapa del juego -> mapa ("" = conflicto)
	static std::map<uint64_t, std::string> t;
	return t;
}
struct DynHit {
	const HookSwap* sw  = nullptr;
	std::string     map;           // mapa resuelto (vacio: sin resolver)
	const char*     why = nullptr; // motivo si no cambia
};
// Con HookMutex() tomado.
inline DynHit DynLookup(uint64_t mapptr, const std::string& name, const std::string& model, int32_t mi, int32_t tix) {
	DynHit     h;
	const auto it  = DynSigIndex().find(DynSig(name, model, mi, tix));
	auto       pit = DynPtrMap().find(mapptr);
	if (pit != DynPtrMap().end() && pit->second.empty()) {
		h.why = "mapa en conflicto";
	} else if (it == DynSigIndex().end()) {
		h.why = "parte que no esta en ningun MSB leido";
	} else if (it->second.size() == 1) {
		const std::string& m = *it->second.begin();
		if (pit != DynPtrMap().end() && pit->second != m) {
			pit->second.clear();
			h.why = "mapa en conflicto";
		} else {
			DynPtrMap()[mapptr] = m;
			h.map               = m;
		}
	} else if (pit != DynPtrMap().end() && it->second.count(pit->second) != 0) {
		h.map = pit->second;
	} else {
		h.why = "nombre en varios mapas y mapa aun sin resolver";
	}
	if (!h.map.empty()) {
		const auto t = DynTable().find(DynKey(h.map, name));
		if (t == DynTable().end()) h.why = "no esta en la tabla";
		else h.sw = &t->second;
	}
	return h;
}
inline bool HookOn() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_HOOK");
		return CrossWorldsOn() && !(c != nullptr && std::strcmp(c, "0") == 0);
	}();
	return on;
}

// (rando11) Decision en el propio gancho, al crear cada enemigo (log 20261123-094340-960: los indices de parte en
// memoria no son los del MSB del archivo y los 4 elegidos de antemano no se crearon en la partida). Para cada enemigo
// normal que el juego crea (c2xxx/c3xxx, sin c3110, sin dialogo, sin entity id, con nombre que no sea no elegible en
// ningun MSB), un hash de (semilla, nombre) decide si cambia (KYTY_DS_RANDOMIZER_CROSS_PCT, por defecto 35 %) y a que
// tipo de otro mundo (los de Donors(): typedata real de ese tipo). Determinista: el mismo enemigo cambia igual al
// reaparecer. KYTY_DS_RANDOMIZER_DYN=0 lo apaga (quedan solo los 4 elegidos por mapa).
struct DynType {
	char    model[16] {};
	int32_t npc = 0;
	uint8_t td_head[0x14] {};
	uint8_t td_init[4] {};
};
inline std::vector<DynType>& DynPool() {
	static std::vector<DynType> v;
	return v;
}
inline std::set<std::string>& DynBadNames() {
	static std::set<std::string> s;
	return s;
}
inline std::atomic<bool>& DynReady() {
	static std::atomic<bool> b {false};
	return b;
}
inline bool DynOn() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_DYN");
		return HookOn() && !(c != nullptr && std::strcmp(c, "0") == 0);
	}();
	return on;
}
inline bool DynVisualOnFwd() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_DYN_VISUAL");
		return DynOn() && !(c != nullptr && std::strcmp(c, "0") == 0);
	}();
	return on;
}
inline int DynPct() {
	static const int p = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_CROSS_PCT");
		int         v = (c != nullptr && *c != '\0') ? std::atoi(c) : 100; // (rando14) todos los normales
		return v < 1 ? 1 : (v > 100 ? 100 : v);
	}();
	return p;
}

// (rando12) Decision unica del cambio de otro mundo, la usan el gancho (al crear el enemigo) y BuildCross3 (al montar la
// copia del legacy_instances.cgpr con la entidad visual del donante): mismo hash (semilla, nombre de la parte), mismo %
// y mismo tipo, sin depender de indices. nullptr: este enemigo no cambia.
inline const DynType* DynChoose(const std::string& pname, const std::string& model) {
	const auto&    pool = DynPool();
	const uint64_t h    = Detail::Fnv(pname, Detail::Fnv("dyn11", Detail::Fnv(Seed())));
	if (pool.empty() || static_cast<int>(h % 100) >= DynPct()) return nullptr;
	std::vector<const DynType*> cand;
	for (const auto& d: pool)
		if (model != d.model) cand.push_back(&d);
	if (cand.empty()) return nullptr;
	return cand[static_cast<size_t>((h >> 8) % cand.size())];
}

inline int TraceLevel() {
	static const int level = [] {
		if (!Enabled() && !DemonioOn()) return 0;
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_TRACE");
		if (c == nullptr || *c == '\0') return 1;
		return std::atoi(c) <= 0 ? 0 : (std::atoi(c) >= 2 ? 2 : 1);
	}();
	return level;
}

inline std::string TraceLow(const std::string& path) {
	std::string low(path);
	for (auto& ch: low) {
		ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
		if (ch == '\\') ch = '/';
	}
	return low;
}

inline const char* TraceTail(const std::string& low) {
	const auto cut  = low.find("dvdroot/");
	const auto cut2 = low.find("scripts/");
	const auto cut3 = low.find("app0/");
	size_t     from = 0;
	if (cut != std::string::npos) from = cut;
	else if (cut2 != std::string::npos) from = cut2;
	else if (cut3 != std::string::npos) from = cut3 + 5;
	return low.c_str() + from;
}

// Traza (solo con el randomizer activo): con el nivel 1, cada archivo de personaje (chr/ del juego original y prefabs de
// scripts/cp11/characters/), cada legacy_instances y cada MSB que el juego lee, una vez por archivo y como mucho 600
// lineas; con el nivel 2, todos los archivos (sin las animaciones del jugador) desde el primer MSB/legacy_instances, una
// vez cada uno y como mucho 20000 lineas.
inline void TraceRead(const std::string& path, bool from_copy) {
	const int level = TraceLevel();
	if (level >= 2 || HookOn() || DemonioOn()) InstallModelProbe(); // (rando8/10) sonda y gancho del modelo, antes de cargar ningun mapa
	if (level == 0) return;
	const std::string low = TraceLow(path);
	if (level < 2 && low.find("/chr/c") == std::string::npos && low.find("scripts/cp11/characters/") == std::string::npos &&
	    low.find("legacy_instances") == std::string::npos && low.find("mapstudio/") == std::string::npos)
		return;
	static std::mutex            m;
	static std::set<std::string> seen;
	static bool                  armed = false;
	std::lock_guard<std::mutex>  lock(m);
	if (level >= 2) {
		// (rando7) Nivel 2: fuera las animaciones del jugador (agotaban la traza antes de cargar el mapa) y solo desde
		// la primera lectura de un MSB de mapa o de un legacy_instances; hasta 20000 archivos.
		if (low.find("characters/c0000_main/") != std::string::npos ||
		    (low.size() >= 5 && low.compare(low.size() - 5, 5, ".cani") == 0))
			return;
		// (rando12) Solo personajes y scripts, sin texturas/sonidos/animaciones (la traza se agotaba antes de 1-1).
		if (low.find("characters/") == std::string::npos && low.find("/chr/") == std::string::npos &&
		    low.find("scripts/") == std::string::npos && low.find("mapstudio/") == std::string::npos)
			return;
		static const char* const kSkip[] = {".ctxr", ".ctxc", ".cesm", ".cani", ".txt", ".cmat", ".cprt"};
		for (const char* ext: kSkip) {
			const size_t n = std::strlen(ext);
			if (low.size() >= n && low.compare(low.size() - n, n, ext) == 0) return;
		}
		if (!armed) {
			if (low.find("legacy_instances") == std::string::npos && low.find("mapstudio/") == std::string::npos) return;
			armed = true;
		}
	}
	if (seen.size() >= (level >= 2 ? 20000u : 600u) || !seen.insert(low).second) return;
	std::printf("[rando-traza] lee %s%s\n", TraceTail(low), from_copy ? " (copia del randomizer)" : "");
	std::fflush(stdout);
}

// Nivel 2: cada lectura de un MSB de mapa o de un legacy_instances (desde, bytes pedidos, bytes leidos), hasta 3000 lineas.
// Sirve para ver si el juego lee la copia entera (las entidades trasplantadas van al final de la lista de objetos).
inline void TraceChunk(const std::string& path, int64_t offset, uint64_t asked, uint64_t got, bool from_copy) {
	if (TraceLevel() < 2) return;
	const std::string low = TraceLow(path);
	if (low.find("legacy_instances") == std::string::npos && low.find("mapstudio/") == std::string::npos) return;
	static std::atomic<int> lines {0};
	if (lines.fetch_add(1) >= 20000) return;
	std::printf("[rando-traza2] lectura %s desde %lld, pedidos %llu, leidos %llu%s\n", TraceTail(low),
	            static_cast<long long>(offset), static_cast<unsigned long long>(asked),
	            static_cast<unsigned long long>(got), from_copy ? " (copia)" : "");
	std::fflush(stdout);
}

// (rando8) Sonda del modelo de cada personaje. Solo con KYTY_DS_RANDOMIZER_TRACE=2 (diagnostico). Ghidra: eboot+0x1451700
// recibe el nombre del modelo (rsi, char*) y el chrid (edx) y monta "$/cp11demonssouls/dvdroot/chr/%s/%s.model?chrid%d"
// (eboot+0x1451c62) que se busca en cp11models.crmp: es lo que elige la plantilla (c2000_dregling_v01..v04). La llaman
// BPDsModel::TryInit (eboot+0x60a77b, con el nombre de [obj+0x10]+0x110 y el chrid de su BP_MSB_PARTS_EXTRA +8) y
// otros 4 sitios (0x363836, 0x36628f, 0x6193ad, 0x619deb). La sonda escribe cada llamada distinta:
//   [rando-traza3] modelo c2000 chrid 200010 desde eboot+0x60a780
// Si un enemigo cambiado llega aqui con su modelo nuevo (c2100...), el juego si crea el enemigo y la plantilla falla en
// el crmp; si nunca llega, el juego descarta la entidad antes. Se instala en la primera lectura de archivo con la traza 2
// (antes de cargar ningun mapa), solo si los 12 primeros bytes son los de 01.007.000; si no, no se toca nada.
#if defined(_WIN32) && (defined(__clang__) || defined(__GNUC__))
} // namespace RandoMsb
extern "C" __declspec(dllimport) int __stdcall VirtualProtect(void*, unsigned long long, unsigned long, unsigned long*);
extern "C" __declspec(dllimport) void* __stdcall VirtualAlloc(void*, unsigned long long, unsigned long, unsigned long);
extern "C" __declspec(dllimport) int __stdcall FlushInstructionCache(void*, const void*, unsigned long long);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();
namespace RandoMsb {

// Hueco de registros que guarda el stub (tras pushfq, rcx, rdx, rsi, rdi, r8..r11 y sub rsp,512):
// +0x220 rdi, +0x228 rsi, +0x230 rdx, +0x238 rcx, +0x248 retorno.
constexpr uint64_t kEbootBase = 0x900000000ull;

inline bool GuestPtrOk(uint64_t p) { return p >= 0x10000ull && p < 0x0000800000000000ull && (p & 7) == 0; }

// (rando27) Ghidra: proxy:IsCompleteEvent (FUN_002455c0 la registra en 0x248fc1 con el objeto 0x27b3288 -> FUN_00213ab0)
// lee, para ids < 20000, FUN_002a7550(*(eboot+0x27bf328), id): bit (id & 7) del byte [obj + id/8], ids hasta 0x467f.
// 4096 = Falange derrotada (m02_00_00_00.lua).
inline bool EventFlagOn(uint32_t id) {
	if (id > 0x467f) return false;
	const uint64_t obj = *reinterpret_cast<const uint64_t*>(kEbootBase + 0x27bf328);
	if (!GuestPtrOk(obj)) return false;
	return ((*reinterpret_cast<const uint8_t*>(obj + (id >> 3)) >> (id & 7)) & 1) != 0;
}

inline std::string GuestStr(const char* s) {
	std::string r;
	if (s == nullptr || !GuestPtrOk(reinterpret_cast<uint64_t>(s) & ~7ull)) return r;
	for (int i = 0; i < 15 && s[i] != '\0'; i++) r.push_back(s[i]);
	return r;
}

struct HookPending {
	const HookSwap* sw = nullptr;
	HookSwap        dyn; // (rando11) cambio decidido en el gancho
};
inline HookPending& Pending() {
	static thread_local HookPending p;
	return p;
}

// Gancho A: entrada de eboot+0x359990 (crear el enemigo de una parte del MSB del mapa).
__attribute__((sysv_abi, noinline)) inline void HookCreate(uint64_t obj, uint64_t, uint64_t, uint64_t, uint64_t /*regs*/) {
	if (const HookSwap* old = Pending().sw; old != nullptr) {
		static std::atomic<int> lines {0};
		if (lines.fetch_add(1) < 200) {
			std::printf("[rando] gancho: sin peticion de modelo para %s\n", old->name.c_str());
			std::fflush(stdout);
		}
	}
	Pending().sw = nullptr;
	// (demonio) dato: cuantas veces entra el juego aqui y por donde sale (Ghidra: 0x359990 es la entrada +0x28 de la
	// vtable 0x2581ee8 del cargador 0x359800 que crea 0x3233d0 para cada ChrIns de FUN_003effe0/0x34ae50).
	static std::atomic<int> calls {0};
	const int               call = calls.fetch_add(1);
	auto                    out  = [&](const char* why) {
                if (DemonioOn() && call < 40) {
                        std::printf("[demonio] 0x359990 llamada %d: sale sin mirar (%s)\n", call + 1, why);
                        std::fflush(stdout);
                }
	};
	if (!GuestPtrOk(obj)) return out("obj no valido");
	if (*reinterpret_cast<const uint8_t*>(obj + 0x14) == 1) return out("jugador / c0000"); // jugador
	const uint64_t map = *reinterpret_cast<const uint64_t*>(obj + 0x8);
	const int32_t  idx = *reinterpret_cast<const int32_t*>(obj + 0x10);
	if (!GuestPtrOk(map) || idx < 0 || idx > 100000) return out("puntero de mapa, parte o modelo no valido");
	uint64_t part = 0, mrec = 0;
	if (*reinterpret_cast<const uint8_t*>(map + 0x22c) == 1) {
		const uint64_t parts = *reinterpret_cast<const uint64_t*>(map + 0x928);
		const uint64_t mods  = *reinterpret_cast<const uint64_t*>(map + 0x860);
		if (!GuestPtrOk(parts) || !GuestPtrOk(mods)) return out("puntero de mapa, parte o modelo no valido");
		part = parts + static_cast<uint64_t>(idx) * 0x80;
		const int32_t mi = *reinterpret_cast<const int32_t*>(part + 0x10);
		if (mi < 0 || mi > 100000) return out("puntero de mapa, parte o modelo no valido");
		mrec = mods + static_cast<uint64_t>(mi) * 0x20;
	} else {
		const uint64_t plist = *reinterpret_cast<const uint64_t*>(map + 0x48);
		const uint64_t mlist = *reinterpret_cast<const uint64_t*>(map + 0x30);
		if (!GuestPtrOk(plist) || !GuestPtrOk(mlist)) return out("puntero de mapa, parte o modelo no valido");
		part = *reinterpret_cast<const uint64_t*>(plist + 0x18 + static_cast<uint64_t>(idx) * 8);
		if (!GuestPtrOk(part)) return out("puntero de mapa, parte o modelo no valido");
		const int32_t mi = *reinterpret_cast<const int32_t*>(part + 0x10);
		if (mi < 0 || mi > 100000) return out("puntero de mapa, parte o modelo no valido");
		mrec = *reinterpret_cast<const uint64_t*>(mlist + 0x18 + static_cast<uint64_t>(mi) * 8);
	}
	if (!GuestPtrOk(part) || !GuestPtrOk(mrec)) return out("puntero de mapa, parte o modelo no valido");
	const uint64_t name_p = *reinterpret_cast<const uint64_t*>(part);
	const uint64_t mname  = *reinterpret_cast<const uint64_t*>(mrec);
	const uint64_t td     = *reinterpret_cast<const uint64_t*>(part + 0x70);
	if (!GuestPtrOk(name_p & ~7ull) || !GuestPtrOk(mname & ~7ull) || !GuestPtrOk(td & ~3ull)) return out("puntero de mapa, parte o modelo no valido");
	const std::string pname = GuestStr(reinterpret_cast<const char*>(name_p));
	const std::string model = GuestStr(reinterpret_cast<const char*>(mname));
	const int32_t     npc   = *reinterpret_cast<const int32_t*>(td + 0xC);
	if (DemonioOn() && (pname == DemonioTarget() || call < 40)) { // (demonio) dato: partes que crea el juego (las primeras y el soldado)
		static std::atomic<int> dn {0};
		if (dn.fetch_add(1) < 80) {
			float pp[3];
			std::memcpy(pp, reinterpret_cast<const void*>(part + 0x20), 12);
			std::printf("[demonio] 0x359990 llamada %d: parte %d %s modelo %s npc %d en (%.1f, %.1f, %.1f)\n", call + 1, idx, pname.c_str(), model.c_str(), npc,
			            static_cast<double>(pp[0]), static_cast<double>(pp[1]), static_cast<double>(pp[2]));
			std::fflush(stdout);
		}
	}
	const HookSwap*   hit   = nullptr;
	bool              tabla = false;
	std::string       where = "?";
	if (DynVisualOnFwd()) {
		std::lock_guard<std::mutex> lock(HookMutex());
		const int32_t               tix = *reinterpret_cast<const int32_t*>(part + 0xC);
		const int32_t               mi  = *reinterpret_cast<const int32_t*>(part + 0x10);
		const DynHit                dh  = DynLookup(map, pname, model, mi, tix);
		hit                             = dh.sw;
		where                           = dh.map.empty() ? std::string("?") : dh.map;
		tabla                           = true;
		if (hit == nullptr && TraceLevel() >= 2 && npc > 0 && Detail::EligibleModel(model) && pname.compare(0, model.size(), model) == 0) {
			static std::atomic<int> nl {0};
			if (nl.fetch_add(1) < 2000) {
				std::printf("[rando] gancho: %s/%s no esta en la tabla (%s; modelo %s npc %d indice %d)\n", where.c_str(), pname.c_str(),
				            dh.why != nullptr ? dh.why : "-", model.c_str(), npc, tix);
				std::fflush(stdout);
			}
		}
	} else {
		std::lock_guard<std::mutex> lock(HookMutex());
		for (const auto& h: HookSwaps())
			if (!h.dead && h.name == pname && model == h.orig_model && (npc == h.orig_npc || npc == h.new_npc)) {
				if (h.has_pos) {
					float pp[3];
					std::memcpy(pp, reinterpret_cast<const void*>(part + 0x20), 12);
					const float dx = pp[0] - h.pos[0], dy = pp[1] - h.pos[1], dz = pp[2] - h.pos[2];
					const bool  ok = dx * dx + dy * dy + dz * dz < 1.0f;
					const bool vang = DemonioOn() && h.name == DemonioTarget() && h.owner == Detail::Fnv("demonio-jefe");
					std::printf("[%s] crear parte %d %s modelo %s npc %d en (%.1f, %.1f, %.1f): %s%s\n", vang ? "demonio" : "rando", idx,
					            pname.c_str(), model.c_str(), npc, static_cast<double>(pp[0]), static_cast<double>(pp[1]),
					            static_cast<double>(pp[2]), ok ? (vang ? "ES EL DEL PATIO, pasa al jefe " : "es el hueco, pasa a ") : "otro mapa, no se toca",
					            ok ? h.new_model : "");
					std::fflush(stdout);
					if (!ok) continue;
				}
				hit = &h;
				break;
			}
	}
	if (!tabla && hit == nullptr && DynOn() && DynReady().load() && npc > 0 && model != "c3110" && Detail::EligibleModel(model) &&
	    pname.compare(0, model.size(), model) == 0 && *reinterpret_cast<const int32_t*>(td + 0x10) == 0) {
		uint64_t ent = *reinterpret_cast<const uint64_t*>(part + 0x68);
		if (!GuestPtrOk(ent & ~3ull) && ent > 0 && ent < 0x1000) ent = part + ent;
		const bool no_entity = GuestPtrOk(ent & ~3ull) && *reinterpret_cast<const int32_t*>(ent) == -1;
		bool       bad       = true;
		{
			std::lock_guard<std::mutex> lock(HookMutex());
			bad = DynBadNames().count(pname) != 0;
		}
		const DynType* pick = (no_entity && !bad) ? DynChoose(pname, model) : nullptr;
		if (pick != nullptr) {
			{
				const DynType& d  = *pick;
				HookSwap&      hs = Pending().dyn;
				hs.name           = pname;
				std::snprintf(hs.orig_model, sizeof(hs.orig_model), "%s", model.c_str());
				hs.orig_npc = npc;
				std::memcpy(hs.new_model, d.model, sizeof(hs.new_model));
				hs.new_npc = d.npc;
				std::memcpy(hs.td_head, d.td_head, sizeof(hs.td_head));
				std::memcpy(hs.td_init, d.td_init, sizeof(hs.td_init));
				hit = &hs;
			}
		}
	}
	if (TraceLevel() >= 2) {
		static std::mutex            m;
		static std::set<std::string> seen;
		std::lock_guard<std::mutex>  lock(m);
		if (seen.size() < 5000 && seen.insert(where + "/" + pname).second) {
			std::printf("[rando-traza4] crear parte %d %s/%s modelo %s npc %d%s\n", idx, where.c_str(), pname.c_str(), model.c_str(), npc,
			            hit != nullptr ? " (elegida)" : "");
			std::fflush(stdout);
		}
	}
	if (hit == nullptr) return;
	if (npc != hit->new_npc) {
		std::memcpy(reinterpret_cast<void*>(td), hit->td_head, sizeof(hit->td_head));
		std::memcpy(reinterpret_cast<void*>(td + 0x18), hit->td_init, sizeof(hit->td_init));
	}
	Pending().sw = hit;
}

// Gancho B: entrada de eboot+0x1451700 (modelo y chrid de un personaje). Sonda (traza 2) y sustitucion (rando10).
__attribute__((sysv_abi, noinline)) inline void ProbeModel(uint64_t /*self*/, const char* name, uint64_t chrid, uint64_t ret,
                                                           uint64_t regs) {
	std::string model = "(nulo)";
	if (name != nullptr) {
		model.clear();
		for (int i = 0; i < 32 && name[i] != '\0'; i++) model.push_back(name[i]);
	}
	int32_t id = static_cast<int32_t>(static_cast<uint32_t>(chrid));
	const uint64_t from = ret >= kEbootBase ? ret - kEbootBase : ret;
	if (const HookSwap* sw = Pending().sw; sw != nullptr && from == 0x366294 &&
	                                       (DynVisualOnFwd() || (model == sw->orig_model && (id == sw->orig_npc || id == sw->new_npc)))) {
		Pending().sw = nullptr;
		*reinterpret_cast<uint64_t*>(regs + 0x228) = reinterpret_cast<uint64_t>(sw->new_model);
		*reinterpret_cast<uint64_t*>(regs + 0x230) = static_cast<uint64_t>(static_cast<uint32_t>(sw->new_npc));
		static std::atomic<int> changed {0};
		std::printf("[rando] gancho: %s -> %s (de la tabla; pedia %s npc %d, ahora npc %d) [%d cambiados]\n", sw->name.c_str(),
		            sw->new_model, model.c_str(), id, sw->new_npc, changed.fetch_add(1) + 1);
		std::fflush(stdout);
		model = sw->new_model;
		id    = sw->new_npc;
	}
	if (DemonioOn()) { // (demonio) dato: el juego pide modelos (las primeras peticiones, de cualquier enemigo)
		static std::atomic<int> any {0};
		if (any.fetch_add(1) < 30) {
			std::printf("[demonio] 0x1451700: modelo %s chrid %d desde eboot+0x%llx\n", model.c_str(), id, static_cast<unsigned long long>(from));
			std::fflush(stdout);
		}
	}
	if (DemonioOn() && (model == "c1030" || model == "c3130" || model == "c3131")) {
		static std::atomic<int> dl {0};
		if (dl.fetch_add(1) < 50) {
			std::printf("[demonio] modelo %s chrid %d pedido desde eboot+0x%llx (0x60a780 = BPDsModel::TryInit)\n", model.c_str(), id,
			            static_cast<unsigned long long>(from));
			std::fflush(stdout);
		}
	}
	if (TraceLevel() < 2) return;
	static std::mutex                                 m;
	static std::set<std::pair<std::string, int32_t>> seen;
	static int                                        lines = 0;
	std::lock_guard<std::mutex>                       lock(m);
	if (lines >= 5000 || !seen.insert({model, id}).second) return;
	lines++;
	std::printf("[rando-traza3] modelo %s chrid %d desde eboot+0x%llx\n", model.c_str(), id, static_cast<unsigned long long>(from));
	std::fflush(stdout);
}

// Stub: guarda registros y estado FP, llama a fn(rdi, rsi, rdx, retorno, hueco de registros) con ABI sysv, restaura
// (los cambios de fn en el hueco pasan a los registros), ejecuta el prologo movido (12 bytes) y salta a entrada+12.
inline bool InstallStub(uint64_t rva, uint64_t fn, const char* what) {
	static const uint8_t expect[12] {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54};
	const uint64_t       entry = kEbootBase + rva;
	auto*                at    = reinterpret_cast<uint8_t*>(entry);
	if (std::memcmp(at, expect, sizeof(expect)) != 0) {
		std::printf("[rando] %s: eboot+0x%llx no es la de 01.007.000; no se instala\n", what, static_cast<unsigned long long>(rva));
		return false;
	}
	auto* stub = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, 0x3000 /*COMMIT|RESERVE*/, 0x40 /*EXECUTE_READWRITE*/));
	if (stub == nullptr) return false;
	std::vector<uint8_t> c;
	auto bytes = [&](std::initializer_list<uint8_t> b) { c.insert(c.end(), b.begin(), b.end()); };
	auto imm64 = [&](uint64_t v) {
		for (int i = 0; i < 8; i++) c.push_back(static_cast<uint8_t>(v >> (8 * i)));
	};
	bytes({0x9c, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53}); // pushfq; rcx rdx rsi rdi r8-r11
	bytes({0x48, 0x81, 0xec, 0x00, 0x02, 0x00, 0x00});                                     // sub rsp, 512
	bytes({0x48, 0x0f, 0xae, 0x04, 0x24});                                                 // fxsave64 [rsp]
	bytes({0x48, 0x8b, 0x8c, 0x24, 0x48, 0x02, 0x00, 0x00});                               // mov rcx, [rsp+0x248] (retorno)
	bytes({0x49, 0x89, 0xe0});                                                             // mov r8, rsp (hueco)
	bytes({0x48, 0xb8});                                                                   // mov rax, fn
	imm64(fn);
	bytes({0xff, 0xd0});                                                                   // call rax
	bytes({0x48, 0x0f, 0xae, 0x0c, 0x24});                                                 // fxrstor64 [rsp]
	bytes({0x48, 0x81, 0xc4, 0x00, 0x02, 0x00, 0x00});                                     // add rsp, 512
	bytes({0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x9d}); // pops; popfq
	c.insert(c.end(), expect, expect + sizeof(expect));                                    // el prologo movido
	bytes({0x48, 0xb8});                                                                   // mov rax, entrada+12
	imm64(entry + sizeof(expect));
	bytes({0xff, 0xe0});                                                                   // jmp rax
	std::memcpy(stub, c.data(), c.size());
	FlushInstructionCache(GetCurrentProcess(), stub, c.size());
	uint8_t patch[12];
	patch[0] = 0x48;
	patch[1] = 0xb8;
	const uint64_t target = reinterpret_cast<uint64_t>(stub);
	std::memcpy(patch + 2, &target, 8);
	patch[10] = 0xff;
	patch[11] = 0xe0; // mov rax, stub; jmp rax (rax libre a la entrada: funcion sin varargs)
	unsigned long old = 0;
	if (VirtualProtect(at, sizeof(patch), 0x40, &old) == 0) {
		std::printf("[rando] %s: no se pudo escribir en el codigo; no se instala\n", what);
		return false;
	}
	std::memcpy(at, patch, sizeof(patch));
	unsigned long dummy = 0;
	VirtualProtect(at, sizeof(patch), old, &dummy);
	FlushInstructionCache(GetCurrentProcess(), at, sizeof(patch));
	std::printf("[rando] %s instalado en eboot+0x%llx\n", what, static_cast<unsigned long long>(rva));
	std::fflush(stdout);
	return true;
}

// (demonio) Trazas de entrada (sin cambiar nada): FUN_01473fb0(rdi, rsi, rdx...) crea las instancias de un recurso del
// cgpr; FUN_01472490(mundo, entidad, char) quita la entidad de las listas del mundo (+0x48 / +0x58). Cada llamada con su
// retorno, como mucho 400 lineas cada una; si el Vanguard se descarta, sale aqui justo despues de crearlo.
__attribute__((sysv_abi, noinline)) inline void DemonioTraceCreate(uint64_t a, uint64_t b, uint64_t c, uint64_t ret, uint64_t) {
	static std::atomic<int> n {0};
	if (n.fetch_add(1) >= 400) return;
	std::printf("[demonio] FUN_01473fb0(%llx, %llx, %llx) desde eboot+0x%llx\n", static_cast<unsigned long long>(a),
	            static_cast<unsigned long long>(b), static_cast<unsigned long long>(c),
	            static_cast<unsigned long long>(ret >= kEbootBase ? ret - kEbootBase : ret));
	std::fflush(stdout);
}
__attribute__((sysv_abi, noinline)) inline void DemonioTraceRemove(uint64_t a, uint64_t b, uint64_t c, uint64_t ret, uint64_t) {
	static std::atomic<int> n {0};
	if (n.fetch_add(1) >= 400) return;
	std::printf("[demonio] FUN_01472490 quitar entidad %llx del mundo %llx (%d) desde eboot+0x%llx\n",
	            static_cast<unsigned long long>(b), static_cast<unsigned long long>(a), static_cast<int>(c & 0xff),
	            static_cast<unsigned long long>(ret >= kEbootBase ? ret - kEbootBase : ret));
	std::fflush(stdout);
}

// (demonio) Ghidra: FUN_000ef430 (llamada desde FUN_000f13a0 al cargar el bloque) recorre las partes de enemigo de cada
// mapa (lista rdx: +0x10 cuantos, +0x18 punteros; mapa +0x252==1: tabla +0x928 de 0x80 desde +0x2e4, cuantas +0x2b4;
// si no, +0x190 / +0x188), lee el typedata (+0x70): si charaInit (+0x18) < 1, NpcParam (+0xC) -> aiThinkId
// (NpcParam +4) -> NpcThinkParam logicId/battleGoalID, y SOLO carga "ai:/out/%06d_logic.lua" / "_battle.lua" de esos
// ids (FUN_000ef990 / FUN_000efad0). El Vanguard cambiaba de tipo despues (0x359990): el mapa seguia diciendo soldado
// 200010 (think 200000) y 103001_battle.lua no se cargaba (en m02_00 solo 200000, 201000...): aparecia sin IA, quieto.
// Aqui, a la entrada de 0xef430, la parte del patio recibe ya el typedata del Vanguard (NpcParam 103001, think 103001:
// logica 10000, batalla 103001), asi el juego carga su script de combate como el de cualquier enemigo del mapa.
__attribute__((sysv_abi, noinline)) inline void DemonioPreAi(uint64_t, uint64_t, uint64_t list, uint64_t, uint64_t) {
	if (!GuestPtrOk(list)) return;
	const int32_t nmaps = *reinterpret_cast<const int32_t*>(list + 0x10);
	{
		static std::atomic<int> calls {0};
		if (calls.fetch_add(1) < 20) {
			std::printf("[rando] 0xef430 (scripts de IA): %d mapas\n", nmaps);
			std::fflush(stdout);
		}
	}
	if (nmaps <= 0 || nmaps > 64) return;
	std::lock_guard<std::mutex> lock(HookMutex());
	for (int32_t m = 0; m < nmaps; m++) {
		const uint64_t map = *reinterpret_cast<const uint64_t*>(list + 0x18 + static_cast<uint64_t>(m) * 8);
		if (!GuestPtrOk(map)) continue;
		const bool    flat = *reinterpret_cast<const uint8_t*>(map + 0x252) != 0;
		const int32_t n    = *reinterpret_cast<const int32_t*>(map + (flat ? 0x2b4 : 0x188));
		if (n <= 0 || n > 100000) continue;
		for (int32_t i = 0; i < n; i++) {
			uint64_t part = 0;
			if (flat) {
				const uint64_t tab = *reinterpret_cast<const uint64_t*>(map + 0x928);
				if (!GuestPtrOk(tab)) break;
				part = tab + static_cast<uint64_t>(*reinterpret_cast<const int32_t*>(map + 0x2e4) + i) * 0x80;
			} else {
				const uint64_t lst = *reinterpret_cast<const uint64_t*>(map + 0x190);
				if (!GuestPtrOk(lst)) break;
				part = *reinterpret_cast<const uint64_t*>(lst + static_cast<uint64_t>(i) * 8);
			}
			if (!GuestPtrOk(part)) continue;
			const uint64_t td     = *reinterpret_cast<const uint64_t*>(part + 0x70);
			const uint64_t name_p = *reinterpret_cast<const uint64_t*>(part);
			if (!GuestPtrOk(td & ~3ull) || !GuestPtrOk(name_p & ~7ull)) continue;
			const int32_t npc = *reinterpret_cast<const int32_t*>(td + 0xC);
			std::string   pname;
			for (const auto& h: HookSwaps()) {
				if (h.dead || !h.has_pos || (npc != h.orig_npc && npc != h.new_npc)) continue;
				if (pname.empty()) pname = GuestStr(reinterpret_cast<const char*>(name_p));
				if (pname != h.name) continue;
				float pp[3];
				std::memcpy(pp, reinterpret_cast<const void*>(part + 0x20), 12);
				const float dx = pp[0] - h.pos[0], dy = pp[1] - h.pos[1], dz = pp[2] - h.pos[2];
				if (dx * dx + dy * dy + dz * dz >= 1.0f) continue;
				std::memcpy(reinterpret_cast<void*>(td), h.td_head, sizeof(h.td_head));
				std::memcpy(reinterpret_cast<void*>(td + 0x18), h.td_init, sizeof(h.td_init));
				if (npc == h.new_npc) { // ya llevaba el tipo nuevo (log 060503: sin linea de 0xef430 y c1030 al crear)
					std::printf("[%s] 0xef430 (IA): %s ya lleva npc %d (%s) al elegir los scripts\n",
					            DemonioOn() && h.name == DemonioTarget() ? "demonio" : "rando", pname.c_str(), npc, h.new_model);
				} else if (DemonioOn() && h.name == DemonioTarget())
					std::printf("[demonio] 0xef430 (scripts de IA del mapa): %s npc %d -> %d (%s) antes de elegir los scripts; el juego "
					            "cargara su IA (ai:/out/<think>_battle.lua del luabnd)\n",
					            pname.c_str(), h.orig_npc, h.new_npc, h.new_model);
				else
					std::printf("[rando] 0xef430 (IA): %s npc %d -> %d (%s) antes de elegir los scripts\n", pname.c_str(), h.orig_npc,
					            h.new_npc, h.new_model);
				std::fflush(stdout);
				break;
			}
		}
	}
}

inline void InstallModelProbe() {
	static std::once_flag once;
	std::call_once(once, [] {
		const bool b = InstallStub(0x1451700, reinterpret_cast<uint64_t>(&ProbeModel), "gancho/sonda del modelo");
		if (TraceLevel() >= 2 && b) {
			std::printf("[rando-traza3] sonda del modelo instalada en eboot+0x1451700\n");
			std::fflush(stdout);
		}
		const bool mezcla_hooks = MezclaHooksOn(); // (rando26) otros mundos / turnos: tipo por gancho + IA en 0xef430
		if (b && (HookOn() || DemonioOn() || mezcla_hooks)) InstallStub(0x359990, reinterpret_cast<uint64_t>(&HookCreate), "gancho de creacion de enemigos");
		if (b && DemonioOn()) {
			InstallStub(0x1473fb0, reinterpret_cast<uint64_t>(&DemonioTraceCreate), "[demonio] traza FUN_01473fb0");
			InstallStub(0x1472490, reinterpret_cast<uint64_t>(&DemonioTraceRemove), "[demonio] traza FUN_01472490");
		}
		if (b && (DemonioOn() || mezcla_hooks))
			InstallStub(0xef430, reinterpret_cast<uint64_t>(&DemonioPreAi), "IA de los huecos cambiados en FUN_000ef430");
	});
}
#else
inline void InstallModelProbe() {}
inline bool EventFlagOn(uint32_t) { return false; }
#endif

namespace Detail {

inline int EnvInt(const char* name, int def, int lo, int hi) {
	const char* v = std::getenv(name);
	if (v == nullptr || v[0] == '\0') return def;
	const int x = std::atoi(v);
	return x < lo ? lo : (x > hi ? hi : x);
}

// (rando17) Variantes de diagnostico del cruce de mundos (claves en extra=). Ver BuildCrossDyn.
inline bool EnvIs(const char* name, const char* val) {
	const char* v = std::getenv(name);
	return v != nullptr && std::strcmp(v, val) == 0;
}

// Rutas en UTF-8 (en Windows un std::string hecho ruta se leería en la página de códigos ANSI).
inline std::filesystem::path Utf8Path(const std::string& s) {
#if defined(__cpp_char8_t)
	return std::filesystem::path(std::u8string(s.begin(), s.end()));
#else
	return std::filesystem::u8path(s);
#endif
}
inline std::string PathUtf8(const std::filesystem::path& p) {
	const auto u = p.u8string();
	return std::string(u.begin(), u.end());
}

inline std::string Lower(std::string s) {
	for (auto& ch: s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	return s;
}

// MSB entero como tablas: cabecera (0x18), relleno, nombre de la tabla y una entrada por offset (contiguas).
struct Table {
	std::vector<uint8_t>              hdr, pad, name;
	std::vector<std::vector<uint8_t>> blobs;
};

inline bool ParseTables(const std::vector<uint8_t>& d, std::vector<Table>* out) {
	Reader             r {d};
	std::vector<Param> params;
	if (!ReadParams(r, &params) || params.size() != 5) return false;
	uint64_t off = 0;
	for (size_t k = 0; k < params.size(); k++) {
		const auto&   p        = params[k];
		const int64_t name_off = r.I64(off + 0x8);
		const int32_t count    = r.I32(off + 0x10);
		const uint64_t end     = p.end != 0 ? p.end : d.size();
		const uint64_t arr_end = off + 0x18 + static_cast<uint64_t>(count) * 8;
		const uint64_t first   = p.entries.empty() ? end : p.entries[0];
		if (!r.ok || name_off < 0 || static_cast<uint64_t>(name_off) < arr_end || first < static_cast<uint64_t>(name_off) ||
		    end > d.size() || first > end)
			return false;
		Table t;
		t.hdr.assign(d.begin() + static_cast<ptrdiff_t>(off), d.begin() + static_cast<ptrdiff_t>(off + 0x18));
		t.pad.assign(d.begin() + static_cast<ptrdiff_t>(arr_end), d.begin() + name_off);
		t.name.assign(d.begin() + name_off, d.begin() + static_cast<ptrdiff_t>(first));
		for (size_t i = 0; i < p.entries.size(); i++) {
			const uint64_t a = p.entries[i];
			const uint64_t b = (i + 1 < p.entries.size()) ? p.entries[i + 1] : end;
			if (b < a || b > end) return false;
			t.blobs.emplace_back(d.begin() + static_cast<ptrdiff_t>(a), d.begin() + static_cast<ptrdiff_t>(b));
		}
		out->push_back(std::move(t));
		off = p.end;
	}
	return true;
}

inline std::vector<uint8_t> BuildTables(const std::vector<Table>& tabs) {
	std::vector<uint8_t> o;
	size_t               prev_next_slot = 0;
	for (size_t k = 0; k < tabs.size(); k++) {
		const auto&    t     = tabs[k];
		const uint64_t start = o.size();
		const uint64_t n     = t.blobs.size();
		std::vector<uint8_t> hdr = t.hdr;
		const int32_t  count    = static_cast<int32_t>(n + 1);
		const int64_t  name_off = static_cast<int64_t>(start + 0x18 + (n + 1) * 8 + t.pad.size());
		std::memcpy(hdr.data() + 0x10, &count, 4);
		std::memcpy(hdr.data() + 0x8, &name_off, 8);
		o.insert(o.end(), hdr.begin(), hdr.end());
		const size_t arr_at = o.size();
		o.resize(o.size() + (n + 1) * 8, 0);
		o.insert(o.end(), t.pad.begin(), t.pad.end());
		o.insert(o.end(), t.name.begin(), t.name.end());
		for (size_t i = 0; i < n; i++) {
			const int64_t at = static_cast<int64_t>(o.size());
			std::memcpy(o.data() + arr_at + i * 8, &at, 8);
			o.insert(o.end(), t.blobs[i].begin(), t.blobs[i].end());
		}
		if (k > 0) {
			const int64_t at = static_cast<int64_t>(start);
			std::memcpy(o.data() + prev_next_slot, &at, 8);
		}
		prev_next_slot = arr_at + n * 8;
	}
	return o;
}

inline int32_t BI32(const std::vector<uint8_t>& b, size_t off) {
	int32_t v = 0;
	if (off + 4 <= b.size()) std::memcpy(&v, b.data() + off, 4);
	return v;
}
inline int64_t BI64(const std::vector<uint8_t>& b, size_t off) {
	int64_t v = 0;
	if (off + 8 <= b.size()) std::memcpy(&v, b.data() + off, 8);
	return v;
}
inline std::string BStr(const std::vector<uint8_t>& b, int64_t off) {
	std::string s;
	if (off < 0) return s;
	for (size_t i = static_cast<size_t>(off); i < b.size() && b[i] != 0 && s.size() < 128; i++)
		s.push_back(static_cast<char>(b[i]));
	return s;
}
inline std::string ModelName(const std::vector<uint8_t>& blob) { return BStr(blob, BI64(blob, 0)); }

// Partes enemigas (tipo 2) de un MSB ya en tablas: índice de parte, offset del typedata en la entrada, si es elegible.
struct EnemyPart {
	size_t      index = 0;
	size_t      td    = 0;
	int32_t     model = -1;
	bool        eligible = false;
	std::string name;
};

inline bool EnemyParts(const std::vector<Table>& t, std::vector<EnemyPart>* out) {
	const auto& models = t[0].blobs;
	// Índices de parte citados por eventos (bloque común + datos del tipo), como en la fase 1.
	std::set<int32_t> cited;
	for (const auto& e: t[1].blobs) {
		const int64_t common = BI64(e, 0x18);
		if (common <= 0 || static_cast<uint64_t>(common) >= e.size() || e.size() > 0x1000) return false;
		for (size_t o = static_cast<size_t>(common); o + 4 <= e.size(); o += 4) cited.insert(BI32(e, o));
	}
	const auto& parts = t[3].blobs;
	for (size_t i = 0; i < parts.size(); i++) {
		const auto& p = parts[i];
		if (p.size() < 0x78) return false;
		if (BI32(p, 0x8) != 2) continue;
		const int32_t model  = BI32(p, 0x10);
		const int64_t ent    = BI64(p, 0x68);
		const int64_t td     = BI64(p, 0x70);
		const int64_t nameo  = BI64(p, 0x0);
		if (model < 0 || static_cast<size_t>(model) >= models.size() || ent <= 0 || td <= 0 || nameo <= 0 ||
		    static_cast<uint64_t>(ent) + 4 > p.size() || static_cast<uint64_t>(td) + 0x38 > p.size())
			return false;
		EnemyPart e;
		e.index = i;
		e.td    = static_cast<size_t>(td);
		e.model = model;
		e.name  = BStr(p, nameo);
		const std::string mname = ModelName(models[static_cast<size_t>(model)]);
		bool finite = true;
		for (int f = 0; f < 6; f++) {
			float v = 0;
			std::memcpy(&v, p.data() + 0x20 + f * 4, 4);
			finite = finite && std::isfinite(v);
		}
		e.eligible = BI32(p, static_cast<size_t>(ent)) == -1 && cited.count(static_cast<int32_t>(i)) == 0 &&
		             EligibleModel(mname) && finite && BI32(p, e.td + 0x10) == 0;
		out->push_back(std::move(e));
	}
	return true;
}

// Tipos que pueden ir a otros mundos: normales, de tamaño humano o menor, que andan (sin voladores, sin gigantes, sin
// los que tienen ESD propia ni los fijos al suelo). c3110 (lagarto de cristal) queda fuera: suelta objetos.
inline const std::vector<std::string>& CrossPool() {
	static const std::vector<std::string> pool = {"c2000", "c2010", "c2020", "c2030", "c2070", "c2080", "c2100",
	                                              "c2110", "c3010", "c3060", "c3070", "c3071", "c3120", "c3131",
	                                              "c3133", "c4020", "c4021", "c4022", "c4030"};
	return pool;
}

struct Donor {
	std::string          model, home; // home: "m05"
	std::vector<uint8_t> model_blob;
	uint8_t              td_head[0x14] {}; // typedata +0x00..+0x13
	uint8_t              td_init[4] {};    // typedata +0x18 (charaInit)
};

inline bool ReadWhole(const std::filesystem::path& p, std::vector<uint8_t>* out) {
	std::FILE* f = nullptr;
#ifdef _WIN32
	f = _wfopen(p.wstring().c_str(), L"rb");
#else
	f = std::fopen(p.string().c_str(), "rb");
#endif
	if (f == nullptr) return false;
	std::fseek(f, 0, SEEK_END);
	const long size = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	bool ok = size > 0 && size < (64l << 20);
	if (ok) {
		out->resize(static_cast<size_t>(size));
		ok = std::fread(out->data(), 1, out->size(), f) == out->size();
	}
	std::fclose(f);
	return ok;
}

inline bool HasBluepointModel(const std::filesystem::path& app0, const std::string& model) {
	std::error_code ec;
	if (!std::filesystem::is_directory(app0 / "characters", ec)) return false;
	for (std::filesystem::directory_iterator it(app0 / "characters", ec), end; !ec && it != end; it.increment(ec)) {
		const std::string n = Lower(PathUtf8(it->path().filename()));
		if (n.size() > model.size() && n.compare(0, model.size(), model) == 0 && n[model.size()] == '_') return true;
	}
	return false;
}

// Tabla de donantes, una vez: lee los MSB de los mundos m01..m07 junto al que se abre.
inline const std::vector<Donor>& Donors(const std::filesystem::path& mapstudio) {
	static std::once_flag     once;
	static std::vector<Donor> donors;
	std::call_once(once, [&] {
		std::error_code ec;
		const auto dvdroot = mapstudio.parent_path().parent_path();
		const auto app0    = dvdroot.parent_path().parent_path();
		std::vector<std::filesystem::path> files;
		for (std::filesystem::directory_iterator it(mapstudio, ec), end; !ec && it != end; it.increment(ec)) {
			const std::string n = Lower(PathUtf8(it->path().filename()));
			// mXX_YY_00_00.msb de los mundos 1..7 (no m08 tutorial, no m99 pruebas, no variantes _99).
			if (n.size() == 16 && n[0] == 'm' && n[1] == '0' && n[2] >= '1' && n[2] <= '7' && n.compare(6, 10, "_00_00.msb") == 0)
				files.push_back(it->path());
		}
		std::sort(files.begin(), files.end());
		for (const auto& model: CrossPool()) {
			const auto chrbnd = dvdroot / "chr" / model / (model + ".chrbnd");
			if (!std::filesystem::is_regular_file(chrbnd, ec) || !HasBluepointModel(app0, model)) {
				std::printf("[rando] otros mundos: %s sin chrbnd o sin modelo de Bluepoint, no se usa\n", model.c_str());
				continue;
			}
			Donor best;
			// Moda del typedata entre los elegibles; si no hay, entre los que no hablan y no tienen charaInit.
			std::vector<std::pair<std::vector<uint8_t>, std::string>> seen_elig, seen_any;
			for (const auto& f: files) {
				std::vector<uint8_t> d;
				std::vector<Table>   t;
				std::vector<EnemyPart> ps;
				if (!ReadWhole(f, &d) || !ParseTables(d, &t) || !EnemyParts(t, &ps)) continue;
				int32_t mi = -1;
				for (size_t i = 0; i < t[0].blobs.size(); i++)
					if (BI32(t[0].blobs[i], 0x8) == 2 && ModelName(t[0].blobs[i]) == model) mi = static_cast<int32_t>(i);
				if (mi < 0) continue;
				const std::string home = Lower(PathUtf8(f.filename())).substr(0, 3);
				for (const auto& e: ps) {
					if (e.model != mi) continue;
					const auto& p = t[3].blobs[e.index];
					std::vector<uint8_t> key(p.begin() + static_cast<ptrdiff_t>(e.td), p.begin() + static_cast<ptrdiff_t>(e.td + 0x14));
					key.insert(key.end(), p.begin() + static_cast<ptrdiff_t>(e.td + 0x18), p.begin() + static_cast<ptrdiff_t>(e.td + 0x1C));
					if (BI32(p, e.td + 0x10) != 0 || BI32(p, e.td + 0x18) != -1) continue;
					(e.eligible ? seen_elig : seen_any).emplace_back(key, home);
				}
				if (best.model_blob.empty()) {
					best.model_blob = t[0].blobs[static_cast<size_t>(mi)];
					best.home       = home;
				}
			}
			auto& seen = seen_elig.empty() ? seen_any : seen_elig;
			if (seen.empty() || best.model_blob.size() < 0x1C) {
				std::printf("[rando] otros mundos: %s sin enemigo normal de origen, no se usa\n", model.c_str());
				continue;
			}
			size_t best_i = 0, best_n = 0;
			for (size_t i = 0; i < seen.size(); i++) {
				size_t n = 0;
				for (const auto& s: seen) n += s.first == seen[i].first ? 1 : 0;
				if (n > best_n) best_n = n, best_i = i;
			}
			best.model = model;
			best.home  = seen[best_i].second;
			std::memcpy(best.td_head, seen[best_i].first.data(), 0x14);
			std::memcpy(best.td_init, seen[best_i].first.data() + 0x14, 4);
			donors.push_back(std::move(best));
		}
		std::printf("[rando] otros mundos: %zu tipos disponibles\n", donors.size());
		std::fflush(stdout);
	});
	return donors;
}

// Aplica el cruce a |bytes| (MSB entero). Devuelve true si cambió algo; |bytes| crece.
inline bool CrossBytes(const std::string& map, const std::filesystem::path& mapstudio, std::vector<uint8_t>& bytes) {
	std::vector<Table> t;
	if (!ParseTables(bytes, &t) || BuildTables(t) != bytes) {
		std::printf("[rando] mapa %s: otros mundos: formato no reconocido, solo barajado\n", map.c_str());
		return false;
	}
	std::vector<EnemyPart> ps;
	if (!EnemyParts(t, &ps)) {
		std::printf("[rando] mapa %s: otros mundos: partes fuera de rango, solo barajado\n", map.c_str());
		return false;
	}
	std::vector<size_t> elig;
	for (size_t i = 0; i < ps.size(); i++)
		if (ps[i].eligible && (!Rename() || ps[i].name.size() == 10)) elig.push_back(i);
	if (elig.empty()) return false;

	const auto&       donors = Donors(mapstudio);
	const std::string world  = Lower(map).substr(0, 3);
	auto&             models = t[0].blobs;
	std::set<std::string> present;
	size_t                last_enemy = models.size();
	for (size_t i = 0; i < models.size(); i++) {
		if (BI32(models[i], 0x8) != 2) continue;
		present.insert(ModelName(models[i]));
		last_enemy = i;
	}
	if (last_enemy == models.size()) return false;
	std::vector<const Donor*> cand;
	for (const auto& d: donors)
		if (d.home != world && present.count(d.model) == 0) cand.push_back(&d);
	if (cand.empty()) return false;

	uint64_t state = Fnv(map, Fnv("cross", Fnv(Seed())));
	for (size_t i = cand.size() - 1; i > 0; i--) std::swap(cand[i], cand[static_cast<size_t>(SplitMix(&state) % (i + 1))]);
	const size_t k   = std::min(cand.size(), static_cast<size_t>(EnvInt("KYTY_DS_RANDOMIZER_CROSS_TYPES", 5, 4, 8)));
	const int    pct = EnvInt("KYTY_DS_RANDOMIZER_CROSS_PCT", 50, 10, 100);
	cand.resize(k);

	// Qué enemigos cambian y a qué tipo.
	std::vector<int> pick(ps.size(), -1);
	std::vector<int> used(k, 0);
	for (size_t e: elig) {
		if (static_cast<int>(SplitMix(&state) % 100) >= pct) continue;
		const int which = static_cast<int>(SplitMix(&state) % k);
		pick[e]         = which;
		used[static_cast<size_t>(which)]++;
	}
	// Nuevas entradas de modelo, al final del grupo de enemigos.
	const size_t insert_at = last_enemy + 1;
	int32_t      type_idx  = 0;
	for (const auto& m: models)
		if (BI32(m, 0x8) == 2) type_idx++;
	std::vector<int32_t> new_index(k, -1);
	std::vector<std::vector<uint8_t>> added;
	for (size_t j = 0; j < k; j++) {
		if (used[j] == 0) continue;
		std::vector<uint8_t> blob = cand[j]->model_blob;
		std::memcpy(blob.data() + 0xC, &type_idx, 4);
		type_idx++;
		new_index[j] = static_cast<int32_t>(insert_at + added.size());
		added.push_back(std::move(blob));
	}
	if (added.empty()) return false;
	models.insert(models.begin() + static_cast<ptrdiff_t>(insert_at), added.begin(), added.end());
	const int32_t shift = static_cast<int32_t>(added.size());
	auto&         parts = t[3].blobs;
	for (auto& p: parts) {
		int32_t mi = BI32(p, 0x10);
		if (mi >= static_cast<int32_t>(insert_at)) {
			mi += shift;
			std::memcpy(p.data() + 0x10, &mi, 4);
		}
	}
	static std::mutex            log_mutex;
	static std::set<std::string> logged;
	bool                         first_time = false;
	{
		std::lock_guard<std::mutex> lock(log_mutex);
		first_time = logged.insert(map).second;
	}
	size_t changed = 0;
	for (size_t i = 0; i < ps.size(); i++) {
		if (pick[i] < 0) continue;
		const Donor& d  = *cand[static_cast<size_t>(pick[i])];
		auto&        p  = parts[ps[i].index];
		const std::string from = ModelName(models[static_cast<size_t>(BI32(p, 0x10))]);
		const int32_t mi = new_index[static_cast<size_t>(pick[i])];
		std::memcpy(p.data() + 0x10, &mi, 4);
		std::memcpy(p.data() + ps[i].td, d.td_head, 0x14);
		std::memcpy(p.data() + ps[i].td + 0x18, d.td_init, 4);
		// (fase 3) Nombre nuevo "cXXXX_9NNN": Bluepoint ya no le pega la instancia del enemigo original (su aspecto).
		std::string renamed;
		if (Rename()) {
			char nm[24];
			std::snprintf(nm, sizeof(nm), "%s_9%03d", d.model.c_str(), static_cast<int>(changed % 900));
			const int64_t nameo = BI64(p, 0x0);
			if (std::strlen(nm) == 10 && nameo > 0 && static_cast<uint64_t>(nameo) + 11 <= p.size() &&
			    BStr(p, nameo).size() == 10) {
				std::memcpy(p.data() + nameo, nm, 10);
				renamed = nm;
			}
		}
		changed++;
		if (first_time) {
			std::printf("[rando] mapa %s: %s %s -> %s (de %s)%s%s\n", map.c_str(), ps[i].name.c_str(), from.c_str(),
			            d.model.c_str(), d.home.c_str(), renamed.empty() ? "" : ", ahora ", renamed.c_str());
		}
	}
	// Contadores de instancias de todos los modelos (offset +0x18), recalculados.
	std::vector<int32_t> count(models.size(), 0);
	for (const auto& p: parts) {
		const int32_t mi = BI32(p, 0x10);
		if (mi >= 0 && static_cast<size_t>(mi) < count.size()) count[static_cast<size_t>(mi)]++;
	}
	for (size_t i = 0; i < models.size(); i++) std::memcpy(models[i].data() + 0x18, &count[i], 4);

	std::vector<uint8_t> out = BuildTables(t);
	// Comprobación: el resultado se vuelve a leer y cada parte apunta a un modelo válido con su contador.
	std::vector<Table>     check;
	std::vector<EnemyPart> check_ps;
	if (!ParseTables(out, &check) || !EnemyParts(check, &check_ps) || check[0].blobs.size() != models.size() ||
	    check[3].blobs.size() != parts.size() || check_ps.size() != ps.size()) {
		std::printf("[rando] mapa %s: otros mundos: el resultado no se relee bien, solo barajado\n", map.c_str());
		return false;
	}
	std::string types;
	for (size_t j = 0; j < k; j++) {
		if (used[j] == 0) continue;
		char buf[64];
		std::snprintf(buf, sizeof(buf), "%s%s (de %s) x%d", types.empty() ? "" : ", ", cand[j]->model.c_str(),
		              cand[j]->home.c_str(), used[j]);
		types += buf;
	}
	std::printf("[rando] mapa %s: %zu enemigos de otros mundos: %s\n", map.c_str(), changed, types.c_str());
	std::fflush(stdout);
	bytes.swap(out);
	return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// (randomizer, fase 3) Enemigos de otros mundos de verdad: trasplante de la entidad de Bluepoint.
//
// Probado con la traza (log 20261123-074644-111): el juego lee la copia del MSB, pero QUÉ personajes carga y cómo se
// ven sale de scripts/worlds/<mapa>/_cmn/<mapa>_legacy_instances.cgpr. Formato (comprobado de principio a fin en
// m02_00, m03_01, m03_02; se llega al pie exacto con el recuento de la cabecera):
//   u32 n_obj | n_obj x [hash u32 | longitud LEB128 | datos] | u32 n_raiz | registros | "PRPG" u32 0x86 u32 offset_pie
//   registro = guid[16] | u8 len | nombre | u32 flags | u32 n_hijos | n_hijos x (guid[16] | u32 índice de objeto)
//              | u32 n_sub | n_sub registros (recursivo)
// Cada enemigo es un registro raíz con el nombre de su parte del MSB. Sus objetos llevan el personaje entero (plantilla,
// mallas, sonidos). Los objetos 0xc1a69458 llevan la matriz de mundo (campo 0xad82abbb, 12 float, fila: p*A + t) y el
// objeto 0x9fe48589 del registro raíz lleva el enlace al MSB: 0x577fc591 = índice de parte, 0x16774b52 = id de evento,
// 0xf4033a22 = grupo. Ningún dato de objeto apunta a otro objeto por índice (comparado c4020_0000/0001 y c3071_0000/
// 0001 de m03_02: solo cambian GUID internos, nombres de animación y esos tres campos); las GUID que usan los objetos
// son las de su propio subárbol.
// Trasplante: el subárbol de un enemigo real de otro mundo (cada donante se usa una sola vez: sus GUID siguen únicas)
// se copia al final de los objetos del cgpr del mapa, con índices nuevos, la matriz llevada al sitio del enemigo que
// sustituye, el enlace al MSB de ese enemigo, y el registro raíz con el nombre y la GUID del original. En el MSB la
// parte toma el modelo y el typedata (think/NPC) de ese mismo donante. Si algo no cuadra, el mapa va sin cambios.
namespace Cg {

struct Obj {
	size_t   rec = 0, pay = 0, len = 0, end = 0; // registro completo [rec, end), datos [pay, pay+len)
	uint32_t hash = 0;
};
struct Rec {
	uint8_t                                         g[16] {};
	std::string                                     name;
	uint32_t                                        flags = 0;
	std::vector<std::pair<std::array<uint8_t, 16>, uint32_t>> ch;
	std::vector<Rec>                                sub;
	size_t                                          beg = 0, end = 0;
};
struct File {
	std::vector<uint8_t> d;
	std::vector<Obj>     objs;
	size_t               objend = 0;
	std::vector<Rec>     tops;
};

inline uint32_t U32(const std::vector<uint8_t>& d, size_t o) {
	uint32_t v = 0;
	std::memcpy(&v, d.data() + o, 4);
	return v;
}

inline bool Leb(const std::vector<uint8_t>& d, size_t* o, size_t lim, uint64_t* v) {
	*v = 0;
	for (int s = 0; s < 63; s += 7) {
		if (*o >= lim) return false;
		const uint8_t b = d[(*o)++];
		*v |= static_cast<uint64_t>(b & 0x7f) << s;
		if (b < 0x80) return true;
	}
	return false;
}

inline bool ParseRec(const std::vector<uint8_t>& d, size_t* o, size_t lim, uint32_t nobj, int depth, Rec* r) {
	if (depth > 64 || *o + 17 > lim) return false;
	r->beg = *o;
	std::memcpy(r->g, d.data() + *o, 16);
	*o += 16;
	const size_t ln = d[(*o)++];
	if (*o + ln + 8 > lim) return false;
	r->name.assign(reinterpret_cast<const char*>(d.data() + *o), ln);
	*o += ln;
	r->flags        = U32(d, *o);
	const uint32_t n = U32(d, *o + 4);
	*o += 8;
	if (n > 100000 || *o + static_cast<size_t>(n) * 20 + 4 > lim) return false;
	for (uint32_t k = 0; k < n; k++) {
		std::array<uint8_t, 16> g {};
		std::memcpy(g.data(), d.data() + *o, 16);
		const uint32_t idx = U32(d, *o + 16);
		if (idx >= nobj) return false;
		r->ch.emplace_back(g, idx);
		*o += 20;
	}
	const uint32_t ns = U32(d, *o);
	*o += 4;
	if (ns > 100000) return false;
	r->sub.resize(ns);
	for (uint32_t k = 0; k < ns; k++)
		if (!ParseRec(d, o, lim, nobj, depth + 1, &r->sub[k])) return false;
	r->end = *o;
	return true;
}

// Todo el archivo, hasta el pie exacto. false si algo no cuadra.
inline bool Parse(File* f) {
	const auto& d = f->d;
	if (d.size() < 32) return false;
	const size_t foot = d.size() - 12;
	if (std::memcmp(d.data() + foot, "PRPG", 4) != 0 || U32(d, foot + 4) != 0x86 || U32(d, foot + 8) != foot) return false;
	const uint32_t n = U32(d, 0);
	if (n == 0 || n > 4000000) return false;
	size_t o = 4;
	f->objs.clear();
	f->objs.reserve(n);
	for (uint32_t i = 0; i < n; i++) {
		Obj      ob;
		uint64_t len = 0;
		ob.rec = o;
		if (o + 5 > foot) return false;
		ob.hash = U32(d, o);
		o += 4;
		if (!Leb(d, &o, foot, &len) || len > foot - o) return false;
		ob.pay = o;
		ob.len = static_cast<size_t>(len);
		o += ob.len;
		ob.end = o;
		f->objs.push_back(ob);
	}
	f->objend = o;
	if (o + 4 > foot) return false;
	const uint32_t nt = U32(d, o);
	o += 4;
	if (nt > 1000000) return false;
	f->tops.clear();
	f->tops.resize(nt);
	for (uint32_t k = 0; k < nt; k++)
		if (!ParseRec(d, &o, foot, n, 0, &f->tops[k])) return false;
	return o == foot;
}

constexpr uint32_t kTransform = 0x5894a6c1u; // objeto con la matriz de mundo
constexpr uint32_t kLegacy    = 0x9fe48589u; // objeto con el enlace a la parte del MSB
constexpr uint8_t  kMatrix[5] = {0xbb, 0xab, 0x82, 0xad, 0x30};
constexpr uint8_t  kPartIdx[5] = {0x91, 0xc5, 0x7f, 0x57, 0x04};
constexpr uint8_t  kEventId[5] = {0x52, 0x4b, 0x77, 0x16, 0x04};
constexpr uint8_t  kGroup[5]   = {0x22, 0x3a, 0x03, 0xf4, 0x04};
constexpr uint8_t  kCollPart[5] = {0x64, 0x10, 0xa6, 0x9e, 0x04}; // (rando25) colision del suelo del sitio
constexpr uint8_t  kSiteA[5]    = {0x2e, 0xa8, 0x67, 0xe9, 0x04};
constexpr uint8_t  kSiteB[5]    = {0xc1, 0xa9, 0x67, 0xea, 0x04};

inline size_t Find(const std::vector<uint8_t>& d, size_t a, size_t b, const uint8_t* pat, size_t n) {
	for (size_t i = a; i + n <= b; i++)
		if (std::memcmp(d.data() + i, pat, n) == 0) return i;
	return std::string::npos;
}

struct Root {
	bool     ok = false;
	float    m[12] {};
	uint32_t part = 0, event = 0, group = 0;
	size_t   legacy_obj = 0;
};

// Matriz y enlace al MSB del registro raíz de un enemigo.
inline Root RootInfo(const File& f, const Rec& r) {
	Root out;
	bool have_m = false, have_l = false;
	for (const auto& [g, i]: r.ch) {
		const Obj& ob = f.objs[i];
		if (ob.hash == kTransform && !have_m) {
			const size_t k = Find(f.d, ob.pay, ob.pay + ob.len, kMatrix, 5);
			if (k != std::string::npos && k + 5 + 48 <= ob.pay + ob.len) {
				std::memcpy(out.m, f.d.data() + k + 5, 48);
				have_m = true;
			}
		} else if (ob.hash == kLegacy && !have_l) {
			const size_t a = Find(f.d, ob.pay, ob.pay + ob.len, kPartIdx, 5);
			const size_t b = Find(f.d, ob.pay, ob.pay + ob.len, kEventId, 5);
			const size_t c = Find(f.d, ob.pay, ob.pay + ob.len, kGroup, 5);
			if (a != std::string::npos && b != std::string::npos && c != std::string::npos) {
				out.part       = U32(f.d, a + 5);
				out.event      = U32(f.d, b + 5);
				out.group      = U32(f.d, c + 5);
				out.legacy_obj = i;
				have_l         = true;
			}
		}
	}
	bool finite = true;
	for (float v: out.m) finite = finite && std::isfinite(v);
	out.ok = have_m && have_l && finite;
	return out;
}

inline bool Inv3(const float a[9], float o[9]) {
	const double d = static_cast<double>(a[0]) * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
	                 a[2] * (a[3] * a[7] - a[4] * a[6]);
	if (!std::isfinite(d) || std::fabs(d) < 1e-6) return false;
	o[0] = static_cast<float>((a[4] * a[8] - a[5] * a[7]) / d);
	o[1] = static_cast<float>((a[2] * a[7] - a[1] * a[8]) / d);
	o[2] = static_cast<float>((a[1] * a[5] - a[2] * a[4]) / d);
	o[3] = static_cast<float>((a[5] * a[6] - a[3] * a[8]) / d);
	o[4] = static_cast<float>((a[0] * a[8] - a[2] * a[6]) / d);
	o[5] = static_cast<float>((a[2] * a[3] - a[0] * a[5]) / d);
	o[6] = static_cast<float>((a[3] * a[7] - a[4] * a[6]) / d);
	o[7] = static_cast<float>((a[1] * a[6] - a[0] * a[7]) / d);
	o[8] = static_cast<float>((a[0] * a[4] - a[1] * a[3]) / d);
	return true;
}
inline void Mul3(const float a[9], const float b[9], float o[9]) {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}
// Fila: p' = p*A + t. D lleva el mundo del donante al sitio del original: Md * D = Mt.
struct Delta {
	float a[9] {}, t[3] {};
};
inline bool MakeDelta(const float md[12], const float mt[12], Delta* out) {
	float inv[9];
	if (!Inv3(md, inv)) return false;
	Mul3(inv, mt, out->a);
	for (int j = 0; j < 3; j++)
		out->t[j] = mt[9 + j] - (md[9] * out->a[j] + md[10] * out->a[3 + j] + md[11] * out->a[6 + j]);
	return true;
}
inline void Apply(const Delta& dl, float m[12]) {
	float a[9];
	Mul3(m, dl.a, a);
	float t[3];
	for (int j = 0; j < 3; j++) t[j] = m[9] * dl.a[j] + m[10] * dl.a[3 + j] + m[11] * dl.a[6 + j] + dl.t[j];
	std::memcpy(m, a, 36);
	std::memcpy(m + 9, t, 12);
}

inline void Put32(std::vector<uint8_t>& o, uint32_t v) {
	const size_t at = o.size();
	o.resize(at + 4);
	std::memcpy(o.data() + at, &v, 4);
}

inline void WriteRec(std::vector<uint8_t>& o, const Rec& r, const std::unordered_map<uint32_t, uint32_t>* remap,
                     const uint8_t* root_guid, const std::string* root_name) {
	o.insert(o.end(), root_guid != nullptr ? root_guid : r.g, (root_guid != nullptr ? root_guid : r.g) + 16);
	const std::string& nm = root_name != nullptr ? *root_name : r.name;
	o.push_back(static_cast<uint8_t>(nm.size()));
	o.insert(o.end(), nm.begin(), nm.end());
	Put32(o, r.flags);
	Put32(o, static_cast<uint32_t>(r.ch.size()));
	for (const auto& [g, i]: r.ch) {
		o.insert(o.end(), g.begin(), g.end());
		Put32(o, remap != nullptr ? remap->at(i) : i);
	}
	Put32(o, static_cast<uint32_t>(r.sub.size()));
	for (const auto& s: r.sub) WriteRec(o, s, remap, nullptr, nullptr);
}

inline void CollectRefs(const Rec& r, std::vector<uint32_t>* out) {
	for (const auto& [g, i]: r.ch) out->push_back(i);
	for (const auto& s: r.sub) CollectRefs(s, out);
}
inline void CollectGuids(const Rec& r, std::vector<std::array<uint8_t, 16>>* out) {
	std::array<uint8_t, 16> g {};
	std::memcpy(g.data(), r.g, 16);
	out->push_back(g);
	for (const auto& [cg, i]: r.ch) out->push_back(cg);
	for (const auto& s: r.sub) CollectGuids(s, out);
}

struct Swap {
	size_t      target_top = 0; // registro raíz del mapa que se sustituye
	const File* donor      = nullptr;
	size_t      donor_top  = 0;
	uint64_t    guid_salt  = 0; // (rando12) != 0: GUID nuevos y deterministas para el subárbol (donante repetido)
	bool        has_dest   = false; // (demonio) matriz de destino propia en vez de la del registro sustituido
	float       dest[12] {};
};

// (rando12) GUID del subárbol del donante -> GUID nuevos (XOR con una clave del enemigo destino); el raíz pasa a ser el
// del enemigo original. Se aplica a registros, hijos y a cualquier aparición de esos 16 bytes dentro de los objetos.
using GuidMap = std::unordered_map<std::string, std::array<uint8_t, 16>>;
inline GuidMap MakeGuidMap(const Rec& dr, const uint8_t* root_target, uint64_t salt) {
	GuidMap                               gm;
	std::vector<std::array<uint8_t, 16>> gs;
	CollectGuids(dr, &gs);
	uint8_t key[16];
	uint64_t k1 = salt * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull, k2 = (salt ^ 0xa5a5a5a5a5a5a5a5ull) * 0xbf58476d1ce4e5b9ull;
	std::memcpy(key, &k1, 8);
	std::memcpy(key + 8, &k2, 8);
	for (const auto& g: gs) {
		bool zero = true;
		for (uint8_t b: g) zero = zero && b == 0;
		if (zero) continue;
		std::array<uint8_t, 16> n {};
		for (int i = 0; i < 16; i++) n[i] = static_cast<uint8_t>(g[i] ^ key[i]);
		gm[std::string(reinterpret_cast<const char*>(g.data()), 16)] = n;
	}
	std::array<uint8_t, 16> rt {};
	std::memcpy(rt.data(), root_target, 16);
	gm[std::string(reinterpret_cast<const char*>(dr.g), 16)] = rt;
	return gm;
}
inline void RemapRec(Rec& r, const GuidMap& gm) {
	if (auto it = gm.find(std::string(reinterpret_cast<const char*>(r.g), 16)); it != gm.end()) std::memcpy(r.g, it->second.data(), 16);
	for (auto& [g, i]: r.ch)
		if (auto it = gm.find(std::string(reinterpret_cast<const char*>(g.data()), 16)); it != gm.end()) g = it->second;
	for (auto& s: r.sub) RemapRec(s, gm);
}
inline size_t RemapBytes(std::vector<uint8_t>& o, size_t a, size_t b, const GuidMap& gm) {
	size_t n = 0;
	if (gm.empty()) return 0;
	for (size_t i = a; i + 16 <= b; i++) {
		auto it = gm.find(std::string(reinterpret_cast<const char*>(o.data() + i), 16));
		if (it == gm.end()) continue;
		std::memcpy(o.data() + i, it->second.data(), 16);
		i += 15;
		n++;
	}
	return n;
}

// Nuevo cgpr con los trasplantes. false si algo no cuadra.
inline bool Transplant(const File& t, const std::vector<Swap>& swaps, std::vector<uint8_t>* out) {
	std::vector<uint8_t> o;
	o.reserve(t.d.size() + swaps.size() * 400000);
	Put32(o, 0); // recuento, al final
	o.insert(o.end(), t.d.begin() + 4, t.d.begin() + static_cast<ptrdiff_t>(t.objend));
	uint32_t                                                 next = static_cast<uint32_t>(t.objs.size());
	std::vector<std::unordered_map<uint32_t, uint32_t>>      remaps(swaps.size());
	for (size_t s = 0; s < swaps.size(); s++) {
		const auto& sw = swaps[s];
		const File& dn = *sw.donor;
		const Rec&  dr = dn.tops[sw.donor_top];
		const Rec&  tr = t.tops[sw.target_top];
		const Root  rd = RootInfo(dn, dr);
		const Root  rt = RootInfo(t, tr);
		Delta       dl;
		if (!rd.ok || !rt.ok || !MakeDelta(rd.m, sw.has_dest ? sw.dest : rt.m, &dl)) return false;
		std::vector<uint32_t> refs;
		CollectRefs(dr, &refs);
		auto&         rm = remaps[s];
		const GuidMap gm = sw.guid_salt != 0 ? MakeGuidMap(dr, tr.g, sw.guid_salt) : GuidMap {};
		for (uint32_t i: refs) {
			if (rm.count(i) != 0) continue;
			rm[i]          = next++;
			const Obj& ob  = dn.objs[i];
			const size_t base = o.size();
			o.insert(o.end(), dn.d.begin() + static_cast<ptrdiff_t>(ob.rec), dn.d.begin() + static_cast<ptrdiff_t>(ob.end));
			const size_t pay = base + (ob.pay - ob.rec);
			RemapBytes(o, pay, pay + ob.len, gm);
			if (ob.hash == kTransform && !EnvIs("KYTY_DS_RANDOMIZER_KEEP_MATRIX", "donor")) {
				for (size_t k = Find(o, pay, pay + ob.len, kMatrix, 5); k != std::string::npos && k + 53 <= pay + ob.len;
				     k = Find(o, k + 53, pay + ob.len, kMatrix, 5)) {
					float m[12];
					std::memcpy(m, o.data() + k + 5, 48);
					Apply(dl, m);
					std::memcpy(o.data() + k + 5, m, 48);
				}
			}
			if (ob.hash == kLegacy) {
				const size_t a = Find(o, pay, pay + ob.len, kPartIdx, 5);
				if (a != std::string::npos && U32(o, a + 5) == rd.part) {
					std::memcpy(o.data() + a + 5, &rt.part, 4);
					const size_t b = Find(o, pay, pay + ob.len, kEventId, 5);
					const size_t c = Find(o, pay, pay + ob.len, kGroup, 5);
					if (b != std::string::npos) std::memcpy(o.data() + b + 5, &rt.event, 4);
					if (c != std::string::npos) std::memcpy(o.data() + c + 5, &rt.group, 4);
					// (rando25) Campos del SITIO que el juego lee de este objeto (no del MSB): colision del suelo
					// (0x9ea61064 = typedata +0x1C del MSB: 766 en c2000_0000, 788 en c2010_0029, 513 en c1030_0000 de
					// m03_01) y 0xe967a82e / 0xea67a9c1 (cambian de sitio a sitio dentro del mismo tipo). Se quedaban los
					// del donante: el enemigo colgaba de la colision de OTRO sitio (o de otro mapa, el Vanguard) y el juego
					// no lo creaba (0x359990) estando ahi su cuerpo: c2000_0000 nunca creado con trasplante, si sin el
					// (log 20261127-042910-357, llamada 1). Ahora van los del sitio original.
					const Obj& to = t.objs[rt.legacy_obj];
					for (const uint8_t* key: {kCollPart, kSiteA, kSiteB}) {
						const size_t dp = Find(o, pay, pay + ob.len, key, 5);
						const size_t sp = Find(t.d, to.pay, to.pay + to.len, key, 5);
						if (dp != std::string::npos && sp != std::string::npos) std::memcpy(o.data() + dp + 5, t.d.data() + sp + 5, 4);
					}
				}
			}
		}
	}
	std::memcpy(o.data(), &next, 4);
	Put32(o, static_cast<uint32_t>(t.tops.size()));
	for (size_t k = 0; k < t.tops.size(); k++) {
		const Swap* sw = nullptr;
		size_t      si = 0;
		for (size_t s = 0; s < swaps.size(); s++)
			if (swaps[s].target_top == k) sw = &swaps[s], si = s;
		if (sw == nullptr) {
			o.insert(o.end(), t.d.begin() + static_cast<ptrdiff_t>(t.tops[k].beg),
			         t.d.begin() + static_cast<ptrdiff_t>(t.tops[k].end));
		} else if (sw->guid_salt != 0) {
			Rec r = sw->donor->tops[sw->donor_top];
			RemapRec(r, MakeGuidMap(sw->donor->tops[sw->donor_top], t.tops[k].g, sw->guid_salt));
			WriteRec(o, r, &remaps[si], t.tops[k].g, &t.tops[k].name);
		} else {
			WriteRec(o, sw->donor->tops[sw->donor_top], &remaps[si], t.tops[k].g, &t.tops[k].name);
		}
	}
	const uint32_t foot = static_cast<uint32_t>(o.size());
	o.insert(o.end(), {'P', 'R', 'P', 'G'});
	Put32(o, 0x86);
	Put32(o, foot);
	out->swap(o);
	return true;
}

// (rando9) Edición in situ: en vez de trasplantar la entidad entera, el objeto 0x9fe48589 de la entidad ORIGINAL del
// enemigo (el que da al juego el nombre del modelo, campo 0xd01e2961, y el NpcParam/chrid, campo 0x6c7f55b9; Ghidra:
// eboot+0x359bfc -> 0x367030 -> 0x366130 -> 0x1451700 con modelo+chrid de la tabla de partes) se sustituye por el del
// donante (modelo, NpcParam, animación de inicio...) con el índice de parte, el evento y el grupo del original. Así no
// quedan dos entidades con el mismo índice de parte (la trasplantada perdía). Los índices de objeto no cambian.
inline void PutLeb(std::vector<uint8_t>& o, uint64_t v) {
	do {
		uint8_t b = static_cast<uint8_t>(v & 0x7f);
		v >>= 7;
		if (v != 0) b |= 0x80;
		o.push_back(b);
	} while (v != 0);
}

inline bool EditLegacy(const File& t, const std::vector<Swap>& swaps, std::vector<uint8_t>* out) {
	std::unordered_map<size_t, std::vector<uint8_t>> repl; // índice de objeto -> datos nuevos
	for (const auto& sw: swaps) {
		const File& dn = *sw.donor;
		const Root  rd = RootInfo(dn, dn.tops[sw.donor_top]);
		const Root  rt = RootInfo(t, t.tops[sw.target_top]);
		if (!rd.ok || !rt.ok || repl.count(rt.legacy_obj) != 0) return false;
		const Obj& ob = dn.objs[rd.legacy_obj];
		std::vector<uint8_t> pay(dn.d.begin() + static_cast<ptrdiff_t>(ob.pay),
		                         dn.d.begin() + static_cast<ptrdiff_t>(ob.pay + ob.len));
		const size_t a = Find(pay, 0, pay.size(), kPartIdx, 5);
		const size_t b = Find(pay, 0, pay.size(), kEventId, 5);
		const size_t c = Find(pay, 0, pay.size(), kGroup, 5);
		if (a == std::string::npos || b == std::string::npos || c == std::string::npos || U32(pay, a + 5) != rd.part)
			return false;
		std::memcpy(pay.data() + a + 5, &rt.part, 4);
		std::memcpy(pay.data() + b + 5, &rt.event, 4);
		std::memcpy(pay.data() + c + 5, &rt.group, 4);
		repl[rt.legacy_obj] = std::move(pay);
	}
	std::vector<uint8_t> o;
	o.reserve(t.d.size() + swaps.size() * 64);
	Put32(o, static_cast<uint32_t>(t.objs.size()));
	for (size_t i = 0; i < t.objs.size(); i++) {
		const Obj& ob = t.objs[i];
		auto       it = repl.find(i);
		if (it == repl.end()) {
			o.insert(o.end(), t.d.begin() + static_cast<ptrdiff_t>(ob.rec), t.d.begin() + static_cast<ptrdiff_t>(ob.end));
		} else {
			Put32(o, ob.hash);
			PutLeb(o, it->second.size());
			o.insert(o.end(), it->second.begin(), it->second.end());
		}
	}
	const size_t foot_old = t.d.size() - 12;
	o.insert(o.end(), t.d.begin() + static_cast<ptrdiff_t>(t.objend), t.d.begin() + static_cast<ptrdiff_t>(foot_old));
	const uint32_t foot = static_cast<uint32_t>(o.size());
	o.insert(o.end(), {'P', 'R', 'P', 'G'});
	Put32(o, 0x86);
	Put32(o, foot);
	out->swap(o);
	return true;
}

// Modelo (campo 0xd01e2961) y NpcParam (0x6c7f55b9) del objeto 0x9fe48589 de un registro raíz, para la validación/log.
inline bool LegacyModel(const File& f, const Rec& r, std::string* model, uint32_t* npc) {
	const Root ri = RootInfo(f, r);
	if (!ri.ok) return false;
	const Obj&     ob   = f.objs[ri.legacy_obj];
	const uint8_t  km[5] = {0x61, 0x29, 0x1e, 0xd0, 0x06};
	const uint8_t  kn[5] = {0xb9, 0x55, 0x7f, 0x6c, 0x04};
	const size_t   a    = Find(f.d, ob.pay, ob.pay + ob.len, km, 5);
	const size_t   b    = Find(f.d, ob.pay, ob.pay + ob.len, kn, 5);
	if (a == std::string::npos || b == std::string::npos || a + 6 > ob.pay + ob.len) return false;
	const size_t ln = f.d[a + 5];
	if (a + 6 + ln > ob.pay + ob.len) return false;
	model->assign(reinterpret_cast<const char*>(f.d.data() + a + 6), ln);
	*npc = U32(f.d, b + 5);
	return true;
}

// Trasplante entero (rando5) solo con KYTY_DS_RANDOMIZER_TRANSPLANT=1; por defecto, edición in situ (rando9).
inline bool UseTransplant() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_TRANSPLANT");
		return c != nullptr && std::strcmp(c, "1") == 0;
	}();
	return on;
}

// Comprobación de la copia: se lee entera, índices válidos, GUID de registros sin repetir.
inline bool Check(const std::vector<uint8_t>& bytes, size_t expect_objs) {
	File f;
	f.d = bytes;
	if (!Parse(&f) || f.objs.size() != expect_objs) return false;
	std::set<std::array<uint8_t, 16>> seen;
	std::vector<std::array<uint8_t, 16>> recg;
	std::function<void(const Rec&)> walk = [&](const Rec& r) {
		std::array<uint8_t, 16> g {};
		std::memcpy(g.data(), r.g, 16);
		recg.push_back(g);
		for (const auto& s: r.sub) walk(s);
	};
	for (const auto& r: f.tops) walk(r);
	for (const auto& g: recg)
		if (!seen.insert(g).second) return false;
	return true;
}

} // namespace Cg

struct Inst { // enemigo donante de otro mundo
	std::string          map, name, model;
	std::vector<uint8_t> model_blob;
	uint8_t              td_head[0x14] {};
	uint8_t              td_init[4] {};
};

inline std::filesystem::path MsbPathOf(const std::filesystem::path& app0, const std::string& map) {
	return app0 / "cp11demonssouls" / "dvdroot" / "map" / "mapstudio" / Utf8Path(map + ".msb");
}
inline std::filesystem::path CgprPathOf(const std::filesystem::path& app0, const std::string& map) {
	return app0 / "scripts" / "worlds" / Utf8Path(map) / "_cmn" / Utf8Path(map + "_legacy_instances.cgpr");
}

// Todos los enemigos normales elegibles de los mundos 1..7 de los tipos de la lista blanca (una vez).
inline const std::vector<Inst>& Insts(const std::filesystem::path& app0) {
	static std::once_flag    once;
	static std::vector<Inst> all;
	std::call_once(once, [&] {
		std::error_code ec;
		const auto      mapstudio = app0 / "cp11demonssouls" / "dvdroot" / "map" / "mapstudio";
		std::vector<std::filesystem::path> files;
		for (std::filesystem::directory_iterator it(mapstudio, ec), end; !ec && it != end; it.increment(ec)) {
			const std::string n = Lower(PathUtf8(it->path().filename()));
			if (n.size() == 16 && n[0] == 'm' && n[1] == '0' && n[2] >= '1' && n[2] <= '7' && n.compare(6, 10, "_00_00.msb") == 0)
				files.push_back(it->path());
		}
		std::sort(files.begin(), files.end());
		std::set<std::string> pool(CrossPool().begin(), CrossPool().end());
		for (const auto& f: files) {
			std::vector<uint8_t>   d;
			std::vector<Table>     t;
			std::vector<EnemyPart> ps;
			if (!ReadWhole(f, &d) || !ParseTables(d, &t) || !EnemyParts(t, &ps)) continue;
			const std::string map = Lower(PathUtf8(f.filename()));
			for (const auto& e: ps) {
				if (!e.eligible) continue;
				const auto&       mb    = t[0].blobs[static_cast<size_t>(e.model)];
				const std::string model = ModelName(mb);
				if (pool.count(model) == 0 || BI32(mb, 0x8) != 2) continue;
				const auto& p = t[3].blobs[e.index];
				if (BI32(p, e.td + 0x10) != 0 || BI32(p, e.td + 0x18) != -1) continue;
				Inst in;
				in.map        = map.substr(0, map.size() - 4);
				in.name       = e.name;
				in.model      = model;
				in.model_blob = mb;
				std::memcpy(in.td_head, p.data() + e.td, 0x14);
				std::memcpy(in.td_init, p.data() + e.td + 0x18, 4);
				all.push_back(std::move(in));
			}
		}
		std::printf("[rando] otros mundos (entidades): %zu enemigos donantes posibles\n", all.size());
		std::fflush(stdout);
	});
	return all;
}

inline std::mutex& CgCacheMutex() {
	static std::mutex m;
	return m;
}
inline const Cg::File* DonorFile(const std::filesystem::path& app0, const std::string& map) {
	static std::vector<std::pair<std::string, std::unique_ptr<Cg::File>>> cache;
	std::lock_guard<std::mutex> lock(CgCacheMutex());
	for (const auto& [m, f]: cache)
		if (m == map) return f.get();
	auto f = std::make_unique<Cg::File>();
	if (!ReadWhole(CgprPathOf(app0, map), &f->d) || !Cg::Parse(f.get())) f.reset();
	cache.emplace_back(map, std::move(f));
	return cache.back().second.get();
}
inline std::set<std::string>& UsedDonors() {
	static std::set<std::string> s;
	return s;
}

inline const Cg::Rec* TopByName(const Cg::File& f, const std::string& name, size_t* idx) {
	for (size_t i = 0; i < f.tops.size(); i++)
		if (f.tops[i].name == name) {
			*idx = i;
			return &f.tops[i];
		}
	return nullptr;
}

// (rando11) Tipos para el gancho dinamico y nombres de parte que no son elegibles en algun MSB (una vez).
inline void DynInit(const std::filesystem::path& app0) {
	static std::once_flag once;
	std::call_once(once, [&] {
		if (!DynOn()) return;
		const auto mapstudio = app0 / "cp11demonssouls" / "dvdroot" / "map" / "mapstudio";
		std::vector<DynType> pool;
		for (const auto& d: Donors(mapstudio)) {
			DynType t;
			std::snprintf(t.model, sizeof(t.model), "%s", d.model.c_str());
			std::memcpy(&t.npc, d.td_head + 0xC, 4);
			std::memcpy(t.td_head, d.td_head, sizeof(t.td_head));
			std::memcpy(t.td_init, d.td_init, sizeof(t.td_init));
			if (t.npc > 0) pool.push_back(t);
		}
		std::set<std::string> bad;
		std::error_code       ec;
		for (std::filesystem::directory_iterator it(mapstudio, ec), end; !ec && it != end; it.increment(ec)) {
			const std::string n = Lower(PathUtf8(it->path().filename()));
			if (n.size() < 5 || n.compare(n.size() - 4, 4, ".msb") != 0) continue;
			std::vector<uint8_t>   d;
			std::vector<Table>     t;
			std::vector<EnemyPart> ps;
			if (!ReadWhole(it->path(), &d) || !ParseTables(d, &t) || !EnemyParts(t, &ps)) continue;
			for (const auto& e: ps)
				if (!e.eligible) bad.insert(e.name);
		}
		{
			std::lock_guard<std::mutex> lock(HookMutex());
			DynPool()     = std::move(pool);
			DynBadNames() = std::move(bad);
		}
		DynReady().store(true);
		std::printf("[rando] gancho dinamico: %zu tipos de otros mundos, %zu nombres no elegibles, %d %% de los enemigos normales "
		            "(semilla %s)\n",
		            DynPool().size(), DynBadNames().size(), DynPct(), Seed());
		std::fflush(stdout);
	});
}

// (rando12) Con el gancho dinamico: los MISMOS enemigos que va a cambiar el gancho (DynChoose: hash de semilla y nombre)
// reciben en la copia del legacy_instances.cgpr la entidad visual completa de un enemigo real de ese tipo (trasplante:
// subárbol del donante con la matriz y el enlace al MSB del original, GUID nuevos deterministas y el registro raíz del
// original sustituido in situ, así la entidad del soldado ya no está en la lista). El gancho cambia la lógica (modelo
// legacy, NpcParam, think); el cgpr, lo que se ve. KYTY_DS_RANDOMIZER_DYN_VISUAL=0 lo apaga (solo gancho).
inline bool DynVisualOn() {
	static const bool on = [] {
		const char* c = std::getenv("KYTY_DS_RANDOMIZER_DYN_VISUAL");
		return DynOn() && !(c != nullptr && std::strcmp(c, "0") == 0);
	}();
	return on;
}

inline bool BuildCrossDyn(const std::string& map, const std::filesystem::path& app0, std::vector<uint8_t>* msb_out,
                          std::vector<uint8_t>* cgpr_out) {
	std::vector<uint8_t> msb;
	Cg::File             tf;
	if (!ReadWhole(MsbPathOf(app0, map), &msb) || !ReadWhole(CgprPathOf(app0, map), &tf.d)) return false;
	if (!Cg::Parse(&tf)) {
		std::printf("[rando] mapa %s: legacy_instances con formato no reconocido, sin otros mundos\n", map.c_str());
		return false;
	}
	std::vector<Table> t;
	if (!ParseTables(msb, &t) || BuildTables(t) != msb) return false;
	std::vector<EnemyPart> ps;
	if (!EnemyParts(t, &ps)) return false;
	auto& models = t[0].blobs;
	auto& parts  = t[3].blobs;
	// (rando16) Elegibilidad POR MAPA (EnemyParts de este MSB): antes se juntaban los no elegibles de todos los MSB y,
	// como los nombres se repiten entre mapas, 14 enemigos normales de 1-1 quedaban fuera.
	const auto& insts = Insts(app0);
	struct Pick {
		size_t          pi, top;
		const Inst*     in;
		const Cg::File* df;
		size_t          dtop;
	};
	std::vector<Pick> picks;
	size_t            no_donor = 0;
	// (rando14) Tipos de otros mundos del mapa: solo los que tienen de verdad una entidad donante legible (asi ningun
	// enemigo se queda sin cambiar por falta de donante), N por mapa (KYTY_DS_RANDOMIZER_CROSS_TYPES, por defecto 6,
	// 1..19) elegidos con la semilla y el mapa. El coste de RAM/VRAM va por tipo, no por enemigo.
	std::map<std::string, std::vector<const Inst*>> type_cands;
	std::vector<std::string>                       map_types;
	{
		std::set<std::string> pool_models;
		for (const auto& d: DynPool()) pool_models.insert(d.model);
		for (const auto& in: insts) {
			// (rando16) solo donantes de OTRO mundo (m02 nunca da a m02): antes c2000 de m02_01 entraba en m02_00.
			if (pool_models.count(in.model) == 0 || in.map.compare(0, 3, Lower(map), 0, 3) == 0) continue;
			const Cg::File* df = DonorFile(app0, in.map);
			if (df == nullptr) continue;
			size_t         dtop = 0;
			const Cg::Rec* dr   = TopByName(*df, in.name, &dtop);
			if (dr == nullptr || !Cg::RootInfo(*df, *dr).ok) continue;
			type_cands[in.model].push_back(&in);
		}
		for (const auto& kv: type_cands) map_types.push_back(kv.first);
		uint64_t st = Fnv(map, Fnv("dyn14types", Fnv(Seed())));
		for (size_t i = map_types.size(); i > 1; i--) std::swap(map_types[i - 1], map_types[static_cast<size_t>(SplitMix(&st) % i)]);
		const size_t n = static_cast<size_t>(EnvInt("KYTY_DS_RANDOMIZER_CROSS_TYPES", 6, 1, 19));
		if (map_types.size() > n) map_types.resize(n);
		std::string list;
		for (const auto& m: map_types) list += (list.empty() ? "" : ", ") + m;
		std::printf("[rando] mapa %s: tipos de otros mundos (%zu): %s\n", map.c_str(), map_types.size(), list.c_str());
		std::fflush(stdout);
	}
	if (map_types.empty()) return false;
	size_t why_inel = 0, why_name = 0, why_pct = 0, why_top = 0, why_root = 0; // (rando17) por que no entra cada enemigo
	for (size_t i = 0; i < ps.size(); i++) {
		const auto& e = ps[i];
		if (!e.eligible) { why_inel++; continue; }
		const std::string model = ModelName(models[static_cast<size_t>(e.model)]);
		const auto&       pb    = parts[e.index];
		if (model == "c3110" || e.name.compare(0, model.size(), model) != 0 || BI32(pb, e.td + 0xC) <= 0) { why_name++; continue; }
		{ // (rando14) % (por defecto 100: todos los normales)
			const uint64_t hp = Fnv(e.name, Fnv("dyn11", Fnv(Seed())));
			if (static_cast<int>(hp % 100) >= DynPct()) { why_pct++; continue; }
		}
		size_t         ti = 0;
		const Cg::Rec* r  = TopByName(tf, e.name, &ti);
		if (r == nullptr) { why_top++; continue; }
		const Cg::Root ri = Cg::RootInfo(tf, *r);
		if (!ri.ok || ri.part != e.index) { why_root++; continue; }
		// (rando14) tipo: uno de los N del mapa (hash del nombre); si ese no vale (mismo tipo o sin donante), el siguiente.
		const uint64_t h  = Fnv(e.name, Fnv("dyn12donor", Fnv(Seed())));
		bool           ok = false;
		for (size_t s = 0; s < map_types.size() && !ok; s++) {
			const std::string& tm = map_types[static_cast<size_t>((h + s) % map_types.size())];
			if (tm == model) continue;
			const auto& cand = type_cands[tm];
			for (size_t k = 0; k < cand.size() && !ok; k++) {
				const Inst*     in = cand[static_cast<size_t>(((h >> 8) + k) % cand.size())];
				const Cg::File* df = DonorFile(app0, in->map);
				if (df == nullptr) continue;
				size_t         dtop = 0;
				const Cg::Rec* dr   = TopByName(*df, in->name, &dtop);
				if (dr == nullptr || !Cg::RootInfo(*df, *dr).ok) continue;
				picks.push_back({i, ti, in, df, dtop});
				ok = true;
			}
		}
		if (!ok) no_donor++;
	}
	if (picks.empty()) {
		std::printf("[rando] mapa %s: otros mundos: ningun enemigo cambiado (de %zu: no elegibles %zu, nombre/modelo/npc %zu, "
		            "porcentaje %zu, sin entidad en el cgpr %zu, entidad con otro indice de parte %zu, sin donante %zu)\n",
		            map.c_str(), ps.size(), why_inel, why_name, why_pct, why_top, why_root, no_donor);
		std::fflush(stdout);
		return false;
	}

	// MSB coherente: modelos nuevos al final del grupo de enemigos, la parte toma modelo y typedata del donante.
	size_t last_enemy = models.size();
	for (size_t i = 0; i < models.size(); i++)
		if (BI32(models[i], 0x8) == 2) last_enemy = i;
	if (last_enemy == models.size()) return false;
	const size_t insert_at = last_enemy + 1;
	int32_t      type_idx  = 0;
	for (const auto& m: models)
		if (BI32(m, 0x8) == 2) type_idx++;
	std::vector<std::string>          added_names;
	std::vector<std::vector<uint8_t>> added;
	for (const auto& p: picks) {
		if (std::find(added_names.begin(), added_names.end(), p.in->model) != added_names.end()) continue;
		std::vector<uint8_t> blob = p.in->model_blob;
		std::memcpy(blob.data() + 0xC, &type_idx, 4);
		type_idx++;
		added_names.push_back(p.in->model);
		added.push_back(std::move(blob));
	}
	models.insert(models.begin() + static_cast<ptrdiff_t>(insert_at), added.begin(), added.end());
	const int32_t shift = static_cast<int32_t>(added.size());
	for (auto& p: parts) {
		int32_t mi = BI32(p, 0x10);
		if (mi >= static_cast<int32_t>(insert_at)) {
			mi += shift;
			std::memcpy(p.data() + 0x10, &mi, 4);
		}
	}
	std::vector<Cg::Swap> swaps;
	std::vector<HookSwap>    table;
	std::vector<std::string> table_names;
	for (const auto& p: picks) {
		const auto&   e  = ps[p.pi];
		auto&         pb = parts[e.index];
		{
			HookSwap hs;
			hs.name = map + "/" + e.name;
			std::snprintf(hs.orig_model, sizeof(hs.orig_model), "%s", ModelName(models[static_cast<size_t>(e.model)]).c_str());
			hs.orig_npc = BI32(pb, e.td + 0xC);
			std::snprintf(hs.new_model, sizeof(hs.new_model), "%s", p.in->model.c_str());
			std::memcpy(&hs.new_npc, p.in->td_head + 0xC, 4);
			std::memcpy(hs.td_head, p.in->td_head, sizeof(hs.td_head));
			std::memcpy(hs.td_init, p.in->td_init, sizeof(hs.td_init));
			table.push_back(std::move(hs));
			table_names.push_back(e.name);
		}
		const int32_t mi = static_cast<int32_t>(insert_at + static_cast<size_t>(
		                       std::find(added_names.begin(), added_names.end(), p.in->model) - added_names.begin()));
		std::memcpy(pb.data() + 0x10, &mi, 4);
		std::memcpy(pb.data() + e.td, p.in->td_head, 0x14);
		std::memcpy(pb.data() + e.td + 0x18, p.in->td_init, 4);
		swaps.push_back({p.top, p.df, p.dtop,
		                 EnvIs("KYTY_DS_RANDOMIZER_GUID_REMAP", "0") ? 0ull : (Fnv(e.name, Fnv("dyn12guid", Fnv(map, Fnv(Seed())))) | 1ull)});
	}
	std::vector<int32_t> count(models.size(), 0);
	for (const auto& p: parts) {
		const int32_t mi = BI32(p, 0x10);
		if (mi >= 0 && static_cast<size_t>(mi) < count.size()) count[static_cast<size_t>(mi)]++;
	}
	for (size_t i = 0; i < models.size(); i++) std::memcpy(models[i].data() + 0x18, &count[i], 4);
	std::vector<uint8_t> msb2 = BuildTables(t);
	// (rando17) KYTY_DS_RANDOMIZER_LOGIC=0: solo el aspecto (cgpr), el MSB y el gancho como el original.
	// KYTY_DS_RANDOMIZER_VISUAL=0: solo la logica (MSB + gancho), el cgpr original.
	const bool logic_on  = !EnvIs("KYTY_DS_RANDOMIZER_LOGIC", "0");
	const bool visual_on = !EnvIs("KYTY_DS_RANDOMIZER_VISUAL", "0");
	if (!logic_on) msb2 = msb;

	size_t new_objs = tf.objs.size();
	for (const auto& s: swaps) {
		std::vector<uint32_t> refs;
		Cg::CollectRefs(s.donor->tops[s.donor_top], &refs);
		new_objs += std::set<uint32_t>(refs.begin(), refs.end()).size();
	}
	std::vector<uint8_t>   cg2;
	std::vector<Table>     check;
	std::vector<EnemyPart> check_ps;
	if (!visual_on) new_objs = tf.objs.size();
	bool ok = (visual_on ? (Cg::Transplant(tf, swaps, &cg2) && Cg::Check(cg2, new_objs)) : (cg2 = tf.d, true)) &&
	          ParseTables(msb2, &check) && EnemyParts(check, &check_ps) && check_ps.size() == ps.size();
	if (ok && visual_on) { // la entidad de cada elegido es la del donante (modelo legacy del donante) y nombre/raíz del original
		Cg::File cf;
		cf.d = cg2;
		ok   = Cg::Parse(&cf) && cf.tops.size() == tf.tops.size();
		for (const auto& p: picks) {
			std::string m;
			uint32_t    npc = 0;
			ok = ok && cf.tops[p.top].name == ps[p.pi].name && std::memcmp(cf.tops[p.top].g, tf.tops[p.top].g, 16) == 0 &&
			     Cg::LegacyModel(cf, cf.tops[p.top], &m, &npc) && m == p.in->model && Cg::RootInfo(cf, cf.tops[p.top]).part == ps[p.pi].index;
		}
	}
	if (!ok) {
		std::printf("[rando] mapa %s: otros mundos (gancho + entidad): el resultado no se relee bien, sin cambios\n", map.c_str());
		std::fflush(stdout);
		return false;
	}
	for (const auto& p: picks)
		std::printf("[rando] mapa %s: %s %s -> %s (entidad visual de %s %s)\n", map.c_str(), ps[p.pi].name.c_str(),
		            ModelName(models[static_cast<size_t>(ps[p.pi].model)]).c_str(),
		            p.in->model.c_str(), p.in->map.c_str(), p.in->name.c_str());
	std::printf("[rando] mapa %s: via: gancho en memoria (logica) + entidad trasplantada en el cgpr (aspecto), misma decision; "
	            "%zu enemigos, %zu sin donante (cgpr %zu -> %zu objetos)\n",
	            map.c_str(), picks.size(), no_donor, tf.objs.size(), new_objs);
	std::printf("[rando] mapa %s: variante: logica %s, aspecto %s, GUID %s, matriz %s\n", map.c_str(),
	            logic_on ? "si" : "NO (LOGIC=0)", visual_on ? "si" : "NO (VISUAL=0)",
	            EnvIs("KYTY_DS_RANDOMIZER_GUID_REMAP", "0") ? "del donante (GUID_REMAP=0)" : "nuevos",
	            EnvIs("KYTY_DS_RANDOMIZER_KEEP_MATRIX", "donor") ? "del DONANTE (KEEP_MATRIX=donor)" : "del original");
	std::fflush(stdout);
	{
		std::lock_guard<std::mutex> lock(HookMutex());
		if (logic_on)
			for (size_t k = 0; k < table.size(); k++) DynTable()[DynKey(map, table_names[k])] = std::move(table[k]);
	}
	msb_out->swap(msb2);
	cgpr_out->swap(cg2);
	return true;
}

// (rando16) Indice de firmas de parte de un MSB tal como lo va a leer el juego (copia o original): lo usa el gancho
// para saber de que mapa es cada parte que crea.
inline void DynRegister(const std::string& map, const std::vector<uint8_t>& bytes) {
	std::vector<Table>     t;
	std::vector<EnemyPart> ps;
	if (!ParseTables(bytes, &t) || !EnemyParts(t, &ps)) return;
	std::lock_guard<std::mutex> lock(HookMutex());
	for (const auto& e: ps) {
		const auto& p = t[3].blobs[e.index];
		DynSigIndex()[DynSig(e.name, ModelName(t[0].blobs[static_cast<size_t>(e.model)]), BI32(p, 0x10), BI32(p, 0xC))].insert(map);
	}
}

// MSB y cgpr del mapa con enemigos de otros mundos. false: el mapa va sin cambios (solo fase 1).
inline bool BuildCross3(const std::string& map, const std::filesystem::path& app0, std::vector<uint8_t>* msb_out,
                        std::vector<uint8_t>* cgpr_out) {
	DynInit(app0);
	if (DynVisualOn() && DynReady().load()) return BuildCrossDyn(map, app0, msb_out, cgpr_out);
	std::vector<uint8_t> msb;
	Cg::File             tf;
	if (!ReadWhole(MsbPathOf(app0, map), &msb) || !ReadWhole(CgprPathOf(app0, map), &tf.d)) return false;
	if (!Cg::Parse(&tf)) {
		std::printf("[rando] mapa %s: legacy_instances con formato no reconocido, sin otros mundos\n", map.c_str());
		return false;
	}
	std::vector<Table> t;
	if (!ParseTables(msb, &t) || BuildTables(t) != msb) return false;
	std::vector<EnemyPart> ps;
	if (!EnemyParts(t, &ps)) return false;
	auto& models = t[0].blobs;
	auto& parts  = t[3].blobs;

	// Enemigos del mapa que se pueden cambiar: elegibles, con su entidad de Bluepoint y su enlace al MSB correcto.
	std::vector<std::pair<size_t, size_t>> targets; // (índice en ps, registro raíz)
	for (size_t i = 0; i < ps.size(); i++) {
		if (!ps[i].eligible) continue;
		size_t ti = 0;
		const Cg::Rec* r = TopByName(tf, ps[i].name, &ti);
		if (r == nullptr) continue;
		const Cg::Root ri = Cg::RootInfo(tf, *r);
		if (!ri.ok || ri.part != ps[i].index) continue;
		targets.emplace_back(i, ti);
	}
	if (targets.empty()) return false;

	const std::string world = Lower(map).substr(0, 3);
	std::set<std::string> present;
	size_t                last_enemy = models.size();
	for (size_t i = 0; i < models.size(); i++) {
		if (BI32(models[i], 0x8) != 2) continue;
		present.insert(ModelName(models[i]));
		last_enemy = i;
	}
	if (last_enemy == models.size()) return false;

	// Tipos de otros mundos (los que no hay en este mapa), por semilla.
	const auto&              insts = Insts(app0);
	std::vector<std::string> types;
	for (const auto& m: CrossPool()) {
		if (present.count(m) != 0) continue;
		for (const auto& in: insts)
			if (in.model == m && in.map.substr(0, 3) != world) {
				types.push_back(m);
				break;
			}
	}
	if (types.empty()) return false;
	uint64_t state = Fnv(map, Fnv("cross3", Fnv(Seed())));
	for (size_t i = types.size() - 1; i > 0; i--) std::swap(types[i], types[static_cast<size_t>(SplitMix(&state) % (i + 1))]);
	types.resize(std::min(types.size(), static_cast<size_t>(EnvInt("KYTY_DS_RANDOMIZER_CROSS_TYPES", 2, 1, 8))));
	for (size_t i = targets.size() - 1; i > 0; i--)
		std::swap(targets[i], targets[static_cast<size_t>(SplitMix(&state) % (i + 1))]);
	size_t want = std::min<size_t>(targets.size(), 4);
	if (const char* p = std::getenv("KYTY_DS_RANDOMIZER_CROSS_PCT"); p != nullptr && p[0] != '\0') {
		const int pct = EnvInt("KYTY_DS_RANDOMIZER_CROSS_PCT", 10, 1, 100);
		want = std::max<size_t>(1, targets.size() * static_cast<size_t>(pct) / 100);
	}

	// Donante para cada enemigo: un enemigo real de otro mundo, del tipo que toque, usado una sola vez.
	struct Pick {
		size_t       pi, top;
		const Inst*  in;
		const Cg::File* df;
		size_t       dtop;
	};
	std::vector<Pick> picks;
	for (size_t k = 0; k < targets.size() && picks.size() < want; k++) {
		const std::string& type = types[picks.size() % types.size()];
		std::vector<const Inst*> cand;
		for (const auto& in: insts)
			if (in.model == type && in.map.substr(0, 3) != world) cand.push_back(&in);
		for (size_t i = cand.size(); i > 1; i--) std::swap(cand[i - 1], cand[static_cast<size_t>(SplitMix(&state) % i)]);
		for (const Inst* in: cand) {
			const std::string key = in->map + "/" + in->name;
			if (UsedDonors().count(key) != 0) continue;
			const Cg::File* df = DonorFile(app0, in->map);
			if (df == nullptr) continue;
			size_t         dtop = 0;
			const Cg::Rec* dr   = TopByName(*df, in->name, &dtop);
			if (dr == nullptr || !Cg::RootInfo(*df, *dr).ok) continue;
			UsedDonors().insert(key);
			picks.push_back({targets[k].first, targets[k].second, in, df, dtop});
			break;
		}
	}
	if (picks.empty()) return false;

	// MSB: modelos nuevos al final del grupo de enemigos, la parte toma modelo y typedata del donante.
	const size_t insert_at = last_enemy + 1;
	int32_t      type_idx  = 0;
	for (const auto& m: models)
		if (BI32(m, 0x8) == 2) type_idx++;
	std::vector<std::string>          added_names;
	std::vector<std::vector<uint8_t>> added;
	for (const auto& p: picks) {
		if (std::find(added_names.begin(), added_names.end(), p.in->model) != added_names.end()) continue;
		std::vector<uint8_t> blob = p.in->model_blob;
		std::memcpy(blob.data() + 0xC, &type_idx, 4);
		type_idx++;
		added_names.push_back(p.in->model);
		added.push_back(std::move(blob));
	}
	models.insert(models.begin() + static_cast<ptrdiff_t>(insert_at), added.begin(), added.end());
	const int32_t shift = static_cast<int32_t>(added.size());
	for (auto& p: parts) {
		int32_t mi = BI32(p, 0x10);
		if (mi >= static_cast<int32_t>(insert_at)) {
			mi += shift;
			std::memcpy(p.data() + 0x10, &mi, 4);
		}
	}
	std::vector<Cg::Swap> swaps;
	std::vector<HookSwap> hooks; // (rando10) para el gancho en memoria
	for (const auto& p: picks) {
		const auto&   e  = ps[p.pi];
		auto&         pb = parts[e.index];
		{
			HookSwap h;
			h.name = e.name;
			std::snprintf(h.orig_model, sizeof(h.orig_model), "%s", ModelName(models[static_cast<size_t>(e.model)]).c_str());
			h.orig_npc = BI32(pb, e.td + 0xC);
			std::snprintf(h.new_model, sizeof(h.new_model), "%s", p.in->model.c_str());
			std::memcpy(&h.new_npc, p.in->td_head + 0xC, 4);
			std::memcpy(h.td_head, p.in->td_head, sizeof(h.td_head));
			std::memcpy(h.td_init, p.in->td_init, sizeof(h.td_init));
			hooks.push_back(std::move(h));
		}
		const int32_t mi = static_cast<int32_t>(insert_at + static_cast<size_t>(
		                       std::find(added_names.begin(), added_names.end(), p.in->model) - added_names.begin()));
		std::memcpy(pb.data() + 0x10, &mi, 4);
		std::memcpy(pb.data() + e.td, p.in->td_head, 0x14);
		std::memcpy(pb.data() + e.td + 0x18, p.in->td_init, 4);
		swaps.push_back({p.top, p.df, p.dtop});
	}
	std::vector<int32_t> count(models.size(), 0);
	for (const auto& p: parts) {
		const int32_t mi = BI32(p, 0x10);
		if (mi >= 0 && static_cast<size_t>(mi) < count.size()) count[static_cast<size_t>(mi)]++;
	}
	for (size_t i = 0; i < models.size(); i++) std::memcpy(models[i].data() + 0x18, &count[i], 4);
	std::vector<uint8_t> msb2 = BuildTables(t);

	std::vector<uint8_t> cg2;
	size_t               new_objs = tf.objs.size();
	const bool           transplant = Cg::UseTransplant();
	for (const auto& s: swaps) {
		if (!transplant) break;
		std::vector<uint32_t> refs;
		Cg::CollectRefs(s.donor->tops[s.donor_top], &refs);
		std::set<uint32_t> u(refs.begin(), refs.end());
		new_objs += u.size();
	}
	std::vector<Table>     check;
	std::vector<EnemyPart> check_ps;
	if (!(transplant ? Cg::Transplant(tf, swaps, &cg2) : Cg::EditLegacy(tf, swaps, &cg2)) || !Cg::Check(cg2, new_objs) || !ParseTables(msb2, &check) ||
	    !EnemyParts(check, &check_ps) || check_ps.size() != ps.size()) {
		std::printf("[rando] mapa %s: otros mundos: el resultado no se relee bien, sin cambios\n", map.c_str());
		std::fflush(stdout);
		for (const auto& p: picks) UsedDonors().erase(p.in->map + "/" + p.in->name);
		return false;
	}
	if (!transplant) {
		Cg::File cf;
		cf.d = cg2;
		bool ok = Cg::Parse(&cf);
		for (const auto& p: picks) {
			std::string m;
			uint32_t    npc = 0;
			ok = ok && Cg::LegacyModel(cf, cf.tops[p.top], &m, &npc) && m == p.in->model;
		}
		if (!ok) {
			std::printf("[rando] mapa %s: otros mundos (in situ): el modelo no quedó cambiado, sin cambios\n", map.c_str());
			std::fflush(stdout);
			for (const auto& p: picks) UsedDonors().erase(p.in->map + "/" + p.in->name);
			return false;
		}
	}
	for (const auto& p: picks)
		std::printf("[rando] mapa %s: %s %s -> %s (entidad de %s %s)\n", map.c_str(), ps[p.pi].name.c_str(),
		            ModelName(models[static_cast<size_t>(ps[p.pi].model)]).c_str(),
		            p.in->model.c_str(), p.in->map.c_str(), p.in->name.c_str());
	if (HookOn()) {
		std::lock_guard<std::mutex> lock(HookMutex());
		for (auto& h: hooks) HookSwaps().push_back(std::move(h));
	}
	std::printf("[rando] mapa %s: via: copia del MSB + cgpr%s\n", map.c_str(),
	            HookOn() ? " + gancho en memoria (eboot+0x359990 / 0x1451700)" : "");
	std::printf("[rando] mapa %s: %zu enemigos de otros mundos, %s (cgpr %zu -> %zu objetos)\n", map.c_str(),
	            picks.size(), transplant ? "entidad trasplantada" : "modelo y NpcParam cambiados in situ en su entidad",
	            tf.objs.size(), new_objs);
	std::fflush(stdout);
	msb_out->swap(msb2);
	cgpr_out->swap(cg2);
	return true;
}

// (rando19) Enemigos que salen en una cinemática (scnXXXXXX de scripts/worlds/mXX/_cmn/mXX_everywhere.cgpr y
// scripts/worlds/<mapa>/_cmn/<mapa>_everywhere.cgpr, p. ej. scn020000 de 1-1 cita c2000_0001..0004 y c2010_0001..0003).
// La cinemática anima su aspecto en el sitio original mientras la lógica (MSB) ya está en el nuevo: cápsula sin cuerpo
// visible = muro invisible. No se mueven (KYTY_DS_RANDOMIZER_MOVE_CINE=1 los vuelve a mover).
// (rando21) Solo las escenas del propio mapa: scnWWAAxx para mWW_AA (scn020000 = 1-1, scn020100 = 1-2). Los nombres de
// enemigo (c2010_0034...) se repiten en cada mapa del mundo; antes, las escenas de 1-2/1-3 del mXX_everywhere.cgpr
// dejaban quietos en 1-1 a c2010_0005/0006/0034..0050 (soldados de lanza y escudo) sin salir en ninguna escena de 1-1.
inline void CineScan(const Cg::File& f, const Cg::Rec& r, bool in_scn, const std::vector<std::string>& names,
                     std::set<std::string>* out, const std::string& pref) {
	const bool scn = in_scn || r.name.rfind(pref, 0) == 0;
	if (scn) {
		for (const auto& [g, i]: r.ch) {
			const Cg::Obj& ob = f.objs[i];
			const auto     b  = f.d.begin() + static_cast<std::ptrdiff_t>(ob.pay);
			const auto     e  = b + static_cast<std::ptrdiff_t>(ob.len);
			for (const auto& n: names) {
				if (n.empty() || out->count(n) != 0) continue;
				for (auto it = std::search(b, e, n.begin(), n.end()); it != e; it = std::search(it + 1, e, n.begin(), n.end())) {
					const auto after = it + static_cast<std::ptrdiff_t>(n.size());
					if (after == e || std::isdigit(*after) == 0) {
						out->insert(n);
						break;
					}
				}
			}
		}
	}
	for (const auto& s: r.sub) CineScan(f, s, scn, names, out, pref);
}
inline std::set<std::string> CineNames(const std::string& map, const std::filesystem::path& app0,
                                       const std::vector<std::string>& names) {
	std::set<std::string> out;
	if (EnvIs("KYTY_DS_RANDOMIZER_MOVE_CINE", "1") || map.size() < 3) return out;
	const std::string world = map.substr(0, 3);
	const std::filesystem::path files[] = {
	    app0 / "scripts" / "worlds" / Utf8Path(world) / "_cmn" / Utf8Path(world + "_everywhere.cgpr"),
	    app0 / "scripts" / "worlds" / Utf8Path(map) / "_cmn" / Utf8Path(map + "_everywhere.cgpr")};
	for (const auto& fp: files) {
		Cg::File f;
		if (!ReadWhole(fp, &f.d) || !Cg::Parse(&f)) continue;
		const std::string pref = "scn" + map.substr(1, 2) + (map.size() >= 6 ? map.substr(4, 2) : std::string());
		for (const auto& t: f.tops) CineScan(f, t, false, names, &out, pref);
	}
	return out;
}

// (rando18) Fuertes al principio. |msb_out| y |cgpr_out| son las copias del mapa con los sitios reasignados.
inline int FuerzaTipo(const std::string& m) {
	// Orden explícito de los tipos conocidos; el resto por rareza en el mapa (menos ejemplares = más fuerte).
	static const std::map<std::string, int> k = {
	    {"c2020", 100}, // caballero de Boletaria (ojos rojos / azules)
	    {"c2030", 90},  {"c2080", 90}, {"c4030", 90}, {"c3120", 85}, {"c2100", 80}, {"c2070", 70}, {"c4022", 70},
	    {"c4021", 65},  {"c4020", 60}, {"c2110", 60}, {"c3060", 55}, {"c3131", 55}, {"c2010", 40}, {"c3070", 35},
	    {"c3010", 30},  {"c3071", 30}, {"c3133", 25}, {"c2000", 20}};
	const auto it = k.find(m);
	return it != k.end() ? it->second : -1;
}

// (rando23) Mezcla: ver MezclaOn. Es "meter una copia": cada sitio recibe una COPIA del enemigo donante; nada se
// reubica. Sitios fijos; se permuta (con la semilla) el enemigo de cada sitio entre los enemigos
// normales del mapa (los mismos elegibles que antes: sin entidad de evento, no citados por eventos, c2xxx/c3xxx, sin
// cinematica, con colision valida). Un enemigo sin entidad no tiene estado de muerte guardado (los .lua solo usan
// entidades), asi que no hereda ningun "muerto" de otro sitio.
// (rando27) Mini-jefe de otro mundo en 1-1 tras derrotar a la Falange (flag 4096). Datos (NpcParam de gameparam.parambnd,
// HP en +0x2C; partes normales elegibles de los MSB: sin entity id, no citadas por eventos, sin charaInit): los de mas
// HP de otros mundos son c3130 (1290, gigante depravado de m06, demasiado grande para un hueco de soldado), c3131 (860,
// m06), c2101 (632, esqueleto de m05_01, 7 ejemplares con IA propia 210100: ataca al ver al jugador, tamaño humano) y
// c2080 (520). Se elige c2101 (m05_01 c2101_0003). Hueco: el arenero c2000_0006 (25.4, 30.2, -48.4; colision 765), el
// enemigo normal mas cercano a la arena (17.5, 30.5, -3.2) a la misma altura: los hoplitas (860-897) y el nucleo (800)
// no valen, el eventboss.lua los mata (ForceDead) con la Falange derrotada. No esta en la niebla ni en el camino del
// ascensor/hoguera de 1-2; es un sitio de soldado de siempre. Antes de 4096 el hueco es el soldado normal.
inline const char* MiniJefeMap() { return "m02_00_00_00"; }
inline const char* MiniJefeSlot() { return "c2000_0006"; }
inline bool FalangeMuerta() {
	const char* c = std::getenv("KYTY_DS_MINIJEFE"); // 1 / 0: forzar (pruebas)
	if (c != nullptr && c[0] == '1') return true;
	if (c != nullptr && c[0] == '0') return false;
	return EventFlagOn(4096);
}

// (rando26) Estado final de cada hueco de la mezcla (tipo que debe tener), para los ganchos de las cargas siguientes.
struct SlotFinal {
	std::string name, model;
	float       pos[3] {};
	uint8_t     td_head[0x14] {}, td_init[4] {};
};
// (rando26) Semilla de la carga en curso ("<semilla>" en la primera, "<semilla>/cN" en la carga N); vacia: Seed().
inline std::string& CurSeed() {
	static std::string s;
	return s;
}

inline bool BuildMezcla(const std::string& map, const std::filesystem::path& app0, std::vector<uint8_t>* msb_out,
                        std::vector<uint8_t>* cgpr_out, std::vector<SlotFinal>* finals = nullptr) {
	const std::string seed = CurSeed().empty() ? std::string(Seed()) : CurSeed();
	std::vector<uint8_t> msb;
	Cg::File             tf;
	if (!ReadWhole(MsbPathOf(app0, map), &msb) || !ReadWhole(CgprPathOf(app0, map), &tf.d)) return false;
	if (!Cg::Parse(&tf)) {
		std::printf("[rando] mapa %s: legacy_instances con formato no reconocido, se deja el original\n", map.c_str());
		return false;
	}
	std::vector<Table> t;
	if (!ParseTables(msb, &t) || BuildTables(t) != msb) return false;
	std::vector<EnemyPart> ps;
	if (!EnemyParts(t, &ps)) return false;
	auto& models = t[0].blobs;
	auto& parts  = t[3].blobs;
	std::vector<uint32_t> refc(tf.objs.size(), 0);
	for (const auto& r: tf.tops) {
		std::vector<uint32_t> refs;
		Cg::CollectRefs(r, &refs);
		std::sort(refs.begin(), refs.end());
		refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
		for (uint32_t o: refs) refc[o]++;
	}
	std::vector<std::string> enames;
	for (const auto& e: ps)
		if (e.eligible) enames.push_back(e.name);
	const std::set<std::string> cine = CineNames(map, app0, enames);
	struct Slot {
		size_t      pi = 0, top = 0;
		std::string model;
	};
	std::vector<Slot> sl;
	for (size_t i = 0; i < ps.size(); i++) {
		if (!ps[i].eligible || cine.count(ps[i].name) != 0) continue;
		if (DemonioOn() && map == "m02_00_00_00" && ps[i].name == DemonioTarget()) continue; // (rando24) hueco del Vanguard
		if (map == MiniJefeMap() && ps[i].name == MiniJefeSlot()) continue; // (rando27) hueco del mini-jefe tras la Falange
		const std::string mname = ModelName(models[static_cast<size_t>(ps[i].model)]);
		if (mname == "c3110") continue; // lagarto de cristal: suelta objetos
		const int32_t coll = BI32(parts[ps[i].index], ps[i].td + 0x1C);
		if (coll < 0 || static_cast<size_t>(coll) >= parts.size() || BI32(parts[static_cast<size_t>(coll)], 0x8) != 5) continue;
		size_t         ti = 0;
		const Cg::Rec* r  = TopByName(tf, ps[i].name, &ti);
		if (r == nullptr) continue;
		const Cg::Root ri = Cg::RootInfo(tf, *r);
		if (!ri.ok || ri.part != ps[i].index) continue;
		std::vector<uint32_t> refs;
		Cg::CollectRefs(*r, &refs);
		bool shared = false; // como en fuertes: una matriz compartida con otra entidad no es solo de este enemigo
		for (uint32_t o: refs) shared = shared || (tf.objs[o].hash == Cg::kTransform && refc[o] != 1);
		if (shared) continue;
		sl.push_back({i, ti, mname});
	}
	if (sl.size() < 2) {
		std::printf("[rando] mapa %s: mezcla: menos de 2 enemigos, se deja el original\n", map.c_str());
		return false;
	}
	// (rando25) Log 20261127-042910-357: con la permutacion casi todo era soldado <-> soldado (c2000 34, c2010 44 de 102)
	// y no se notaba. Ahora cada sitio elige primero un TIPO distinto del suyo, al azar entre todos los tipos normales del
	// mapa (los raros salen tanto como los comunes), y luego un ejemplar de ese tipo como donante (puede repetirse: cada
	// copia lleva GUID nuevos por sitio).
	std::map<std::string, std::vector<size_t>> by_type;
	for (size_t i = 0; i < sl.size(); i++) by_type[sl[i].model].push_back(i);
	uint64_t rs = Fnv(map, Fnv("mezcla2", Fnv(seed)));
	// (rando26) Otros mundos (MUNDOS SI): de 1 a 8 tipos de la lista blanca que no hay en este mapa (por defecto 4, otros
	// en cada carga), cada uno con hasta 6 enemigos reales de otro mundo como donantes (su entidad del cgpr de su mapa).
	struct XDonor {
		const Inst*     in = nullptr;
		const Cg::File* df = nullptr;
		size_t          dtop = 0;
	};
	std::vector<std::pair<std::string, std::vector<XDonor>>> xt;
	if (MundosOn()) {
		std::set<std::string> present;
		for (const auto& m: models)
			if (BI32(m, 0x8) == 2) present.insert(ModelName(m));
		const std::string        world = Lower(map).substr(0, 3);
		const auto&              insts = Insts(app0);
		std::vector<std::string> types;
		for (const auto& m: CrossPool()) {
			if (present.count(m) != 0) continue;
			for (const auto& in: insts)
				if (in.model == m && in.map.substr(0, 3) != world) {
					types.push_back(m);
					break;
				}
		}
		uint64_t xs = Fnv(map, Fnv("mundos", Fnv(seed)));
		for (size_t i = types.size(); i > 1; i--) std::swap(types[i - 1], types[static_cast<size_t>(SplitMix(&xs) % i)]);
		const size_t want = static_cast<size_t>(EnvInt("KYTY_DS_RANDOMIZER_CROSS_TYPES", 4, 1, 8));
		for (const auto& ty: types) {
			if (xt.size() >= want) break;
			std::vector<const Inst*> cand;
			for (const auto& in: insts)
				if (in.model == ty && in.map.substr(0, 3) != world) cand.push_back(&in);
			for (size_t i = cand.size(); i > 1; i--) std::swap(cand[i - 1], cand[static_cast<size_t>(SplitMix(&xs) % i)]);
			std::vector<XDonor> ds;
			for (const Inst* in: cand) {
				if (ds.size() >= 6) break;
				const Cg::File* df = DonorFile(app0, in->map);
				if (df == nullptr) continue;
				size_t         dtop = 0;
				const Cg::Rec* dr   = TopByName(*df, in->name, &dtop);
				if (dr == nullptr || !Cg::RootInfo(*df, *dr).ok) continue;
				ds.push_back({in, df, dtop});
			}
			if (!ds.empty()) xt.emplace_back(ty, std::move(ds));
		}
		std::string names;
		for (const auto& [ty, ds]: xt) names += (names.empty() ? "" : ", ") + ty + " (" + ds[0].in->map.substr(0, 3) + ")";
		std::printf("[rando] mapa %s: otros mundos: %zu tipos: %s\n", map.c_str(), xt.size(), names.c_str());
	}
	// Cada sitio: un tipo distinto del suyo al azar entre los del mapa y los de otros mundos; perm >= sl.size() = otro mundo
	// (perm - sl.size() = indice en xt).
	std::vector<size_t>  perm(sl.size());
	std::vector<XDonor>  xpick(sl.size());
	uint64_t             fs = Fnv(map, Fnv("facil", Fnv(seed))); // (rando27) FACIL: ~30 % de los huecos
	for (size_t k = 0; k < sl.size(); k++) {
		if (Dificultad() == 1 && SplitMix(&fs) % 100 >= 30) {
			perm[k] = k;
			continue;
		}
		std::vector<const std::vector<size_t>*> other;
		for (const auto& [m, v]: by_type)
			if (m != sl[k].model) other.push_back(&v);
		const size_t n = other.size() + xt.size();
		if (n == 0) {
			perm[k] = k;
			continue;
		}
		const size_t c = static_cast<size_t>(SplitMix(&rs) % n);
		if (c < other.size()) {
			const auto& v = *other[c];
			perm[k]       = v[static_cast<size_t>(SplitMix(&rs) % v.size())];
		} else {
			const auto& ds = xt[c - other.size()].second;
			perm[k]        = sl.size() + (c - other.size());
			xpick[k]       = ds[static_cast<size_t>(SplitMix(&rs) % ds.size())];
		}
	}
	// Lo que viaja con el enemigo (del original): indice de modelo (+0x10) y typedata +0x00..+0x13 y +0x18.
	struct Who {
		int32_t model;
		uint8_t head[0x14], init[4];
	};
	std::vector<Who> who(sl.size());
	for (size_t k = 0; k < sl.size(); k++) {
		const auto& e = ps[sl[k].pi];
		const auto& p = parts[e.index];
		who[k].model  = BI32(p, 0x10);
		std::memcpy(who[k].head, p.data() + e.td, 0x14);
		std::memcpy(who[k].init, p.data() + e.td + 0x18, 4);
	}
	std::vector<Cg::Swap> swaps;
	size_t                changed = 0, xchanged = 0;
	for (size_t k = 0; k < sl.size(); k++) { // sitio k recibe al enemigo perm[k]
		const size_t e = perm[k];
		const auto&  s = ps[sl[k].pi];
		auto&        p = parts[s.index];
		SlotFinal    f;
		f.name = s.name;
		std::memcpy(f.pos, p.data() + 0x20, 12);
		if (e >= sl.size()) { // (rando26) de otro mundo: el MSB del hueco no cambia de tipo; tipo e IA por los ganchos
			const XDonor& x = xpick[k];
			p[s.td + 0x14]  = 0;
			f.model         = x.in->model;
			std::memcpy(f.td_head, x.in->td_head, 0x14);
			std::memcpy(f.td_init, x.in->td_init, 4);
			swaps.push_back({sl[k].top, x.df, x.dtop, Fnv(s.name, Fnv("mezcla-guid", Fnv(seed))) | 1});
			changed++;
			xchanged++;
			std::printf("[rando] mezcla: %s sitio de %s (%s) -> %s (de otro mundo: %s %s)\n", map.c_str(), s.name.c_str(),
			            sl[k].model.c_str(), x.in->model.c_str(), x.in->map.c_str(), x.in->name.c_str());
			if (finals != nullptr) finals->push_back(f);
			continue;
		}
		if (sl[e].model == sl[k].model) { // mismo tipo: nada que cambiar
			f.model = sl[k].model;
			std::memcpy(f.td_head, p.data() + s.td, 0x14);
			std::memcpy(f.td_init, p.data() + s.td + 0x18, 4);
			if (finals != nullptr) finals->push_back(f);
			continue;
		}
		std::memcpy(p.data() + 0x10, &who[e].model, 4);
		std::memcpy(p.data() + s.td, who[e].head, 0x14);
		std::memcpy(p.data() + s.td + 0x18, who[e].init, 4);
		p[s.td + 0x14] = 0; // agresivo: sin el estado pasivo (rando22)
		f.model        = sl[e].model;
		std::memcpy(f.td_head, who[e].head, 0x14);
		std::memcpy(f.td_init, who[e].init, 4);
		if (finals != nullptr) finals->push_back(f);
		// Copia de la entidad del donante (del archivo original), anclada a la raiz del sitio, con GUID nuevos: el
		// donante sigue entero en su propio sitio (o recibe a su vez otra copia).
		swaps.push_back({sl[k].top, &tf, sl[e].top, Fnv(s.name, Fnv("mezcla-guid", Fnv(seed))) | 1});
		changed++;
		std::printf("[rando] mezcla: %s sitio de %s (%s) -> %s (de %s)\n", map.c_str(), s.name.c_str(), sl[k].model.c_str(),
		            sl[e].model.c_str(), ps[sl[e].pi].name.c_str());
	}
	if (changed == 0) return false;
	{ // instancias por modelo (+0x18), como en otros mundos
		std::vector<int32_t> count(models.size(), 0);
		for (const auto& p: parts) {
			const int32_t mi = BI32(p, 0x10);
			if (mi >= 0 && static_cast<size_t>(mi) < count.size()) count[static_cast<size_t>(mi)]++;
		}
		for (size_t i = 0; i < models.size(); i++) std::memcpy(models[i].data() + 0x18, &count[i], 4);
	}
	std::vector<uint8_t> cg2;
	size_t               new_objs = tf.objs.size();
	for (const auto& sw: swaps) {
		std::vector<uint32_t> refs;
		Cg::CollectRefs(sw.donor->tops[sw.donor_top], &refs);
		new_objs += std::set<uint32_t>(refs.begin(), refs.end()).size();
	}
	std::vector<uint8_t>   msb2 = BuildTables(t);
	std::vector<Table>     check;
	std::vector<EnemyPart> check_ps;
	if (!MezclaCgprOn()) { // (rando24) solo MSB: cgpr original
		swaps.clear();
		new_objs = tf.objs.size();
	}
	if (!Cg::Transplant(tf, swaps, &cg2) || !Cg::Check(cg2, new_objs) || !ParseTables(msb2, &check) ||
	    !EnemyParts(check, &check_ps) || check_ps.size() != ps.size()) {
		std::printf("[rando] mapa %s: mezcla: el resultado no se relee bien, se deja el original\n", map.c_str());
		std::fflush(stdout);
		return false;
	}
	std::printf("[rando] mapa %s: mezcla: %zu sitios con otro enemigo de %zu, %zu de otros mundos (sitios, colision y grupos "
	            "sin tocar; semilla %s)\n",
	            map.c_str(), changed, sl.size(), xchanged, seed.c_str());
	std::fflush(stdout);
	msb_out->swap(msb2);
	cgpr_out->swap(cg2);
	return true;
}

inline bool BuildFuertes(const std::string& map, const std::filesystem::path& app0, std::vector<uint8_t>* msb_out,
                         std::vector<uint8_t>* cgpr_out) {
	if (MezclaOn()) return BuildMezcla(map, app0, msb_out, cgpr_out); // (rando23)
	std::vector<uint8_t> msb;
	Cg::File             tf;
	if (!ReadWhole(MsbPathOf(app0, map), &msb) || !ReadWhole(CgprPathOf(app0, map), &tf.d)) return false;
	if (!Cg::Parse(&tf)) {
		std::printf("[rando] mapa %s: legacy_instances con formato no reconocido, se deja el original\n", map.c_str());
		return false;
	}
	std::vector<Table> t;
	if (!ParseTables(msb, &t) || BuildTables(t) != msb) return false;
	std::vector<EnemyPart> ps;
	if (!EnemyParts(t, &ps)) return false;
	auto& models = t[0].blobs;
	auto& parts  = t[3].blobs;

	// Inicio del mapa: primera parte de jugador (tipo 4).
	float start[3] {0, 0, 0};
	bool  have_start = false;
	for (const auto& p: parts) {
		if (p.size() < 0x38 || BI32(p, 0x8) != 4) continue;
		std::memcpy(start, p.data() + 0x20, 12);
		have_start = std::isfinite(start[0]) && std::isfinite(start[1]) && std::isfinite(start[2]);
		if (have_start) break;
	}
	if (!have_start) {
		std::printf("[rando] mapa %s: sin punto de inicio del jugador, se deja el original\n", map.c_str());
		return false;
	}

	// Cuántas veces cita cada objeto el árbol entero: un objeto compartido no se puede mover con un solo enemigo.
	// (Cuenta por registro raíz: dentro de una misma entidad un objeto puede salir varias veces.)
	std::vector<uint32_t> refc(tf.objs.size(), 0);
	for (const auto& r: tf.tops) {
		std::vector<uint32_t> refs;
		Cg::CollectRefs(r, &refs);
		std::sort(refs.begin(), refs.end());
		refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
		for (uint32_t o: refs) refc[o]++;
	}

	std::vector<std::string> enames;
	for (const auto& e: ps)
		if (e.eligible) enames.push_back(e.name);
	const std::set<std::string> cine = CineNames(map, app0, enames);
	for (const auto& n: cine)
		std::printf("[rando] mapa %s: fuertes al principio: %s sale en una cinematica, se queda en su sitio\n", map.c_str(),
		            n.c_str());
	const bool move_coll   = !EnvIs("KYTY_DS_RANDOMIZER_MOVE_COLLISION", "0");
	const bool move_patrol = !EnvIs("KYTY_DS_RANDOMIZER_MOVE_PATROL", "0");
	std::printf("[rando] mapa %s: fuertes al principio: colision del sitio %s, patrulla del sitio %s\n", map.c_str(),
	            move_coll ? "si" : "no (KYTY_DS_RANDOMIZER_MOVE_COLLISION=0)",
	            move_patrol ? "si" : "no, sin patrulla (KYTY_DS_RANDOMIZER_MOVE_PATROL=0)");

	struct Cand {
		size_t      pi = 0, top = 0;
		std::string name, model;
		int         fuerza = 0;
		float       dist   = 0;
		Cg::Root    root;
		std::vector<uint32_t> xf; // objetos de matriz de su entidad
	};
	std::vector<Cand> cs;
	std::map<std::string, int> cuenta;
	for (size_t i = 0; i < ps.size(); i++) {
		if (!ps[i].eligible) continue;
		const std::string mname = ModelName(models[static_cast<size_t>(ps[i].model)]);
		if (mname == "c3110") continue; // lagarto de cristal: suelta objetos
		if (cine.count(ps[i].name) != 0) continue; // (rando19)
		{ // (rando20) solo con colision valida (parte de tipo 5): el juego toma de ella los grupos del enemigo
			const int32_t coll = BI32(parts[ps[i].index], ps[i].td + 0x1C);
			if (coll < 0 || static_cast<size_t>(coll) >= parts.size() || parts[static_cast<size_t>(coll)].size() < 0x10 ||
			    BI32(parts[static_cast<size_t>(coll)], 0x8) != 5)
				continue;
		}
		size_t         ti = 0;
		{ // (rando20) solo con colisión válida (tipo 5): ver CollOk
			const int32_t coll = BI32(parts[ps[i].index], ps[i].td + 0x1C);
			if (coll < 0 || static_cast<size_t>(coll) >= parts.size() || parts[static_cast<size_t>(coll)].size() < 0x10 ||
			    BI32(parts[static_cast<size_t>(coll)], 0x8) != 5) {
				std::printf("[rando] mapa %s: %s sin colision valida, se queda en su sitio\n", map.c_str(), ps[i].name.c_str());
				continue;
			}
		}
		const Cg::Rec* r  = TopByName(tf, ps[i].name, &ti);
		if (r == nullptr) continue;
		const Cg::Root ri = Cg::RootInfo(tf, *r);
		if (!ri.ok || ri.part != ps[i].index) continue;
		std::vector<uint32_t> refs;
		Cg::CollectRefs(*r, &refs);
		std::vector<uint32_t> xf;
		bool                  shared = false;
		for (uint32_t o: refs)
			if (tf.objs[o].hash == Cg::kTransform) {
				if (refc[o] != 1) shared = true;
				xf.push_back(o);
			}
		std::sort(xf.begin(), xf.end());
		xf.erase(std::unique(xf.begin(), xf.end()), xf.end());
		if (shared || xf.empty()) continue;
		Cand c;
		c.pi    = i;
		c.top   = ti;
		c.name  = ps[i].name;
		c.model = mname;
		c.root  = ri;
		c.xf    = std::move(xf);
		float pos[3];
		std::memcpy(pos, parts[ps[i].index].data() + 0x20, 12);
		const float dx = pos[0] - start[0], dy = pos[1] - start[1], dz = pos[2] - start[2];
		c.dist = std::sqrt(dx * dx + dy * dy + dz * dz);
		cuenta[mname]++;
		cs.push_back(std::move(c));
	}
	if (cs.size() < 2) {
		std::printf("[rando] mapa %s: fuertes al principio: menos de 2 enemigos movibles, se deja el original\n", map.c_str());
		return false;
	}
	// (rando21) La semilla decide: fuerza con ruido (+-20) y distancia con factor 0,5..1,5. Antes era fijo (solo orden
	// por fuerza y distancia): con cualquier semilla salían los mismos enemigos en los mismos sitios.
	uint64_t rs = Fnv(map, Fnv(Seed()));
	std::vector<float> dkey(cs.size());
	for (size_t i = 0; i < cs.size(); i++) {
		auto&     c = cs[i];
		const int k = FuerzaTipo(c.model);
		c.fuerza    = (k >= 0 ? k : 50 - std::min(40, cuenta[c.model])) * 100 + static_cast<int>(SplitMix(&rs) % 4001) - 2000;
		dkey[i]     = c.dist * (0.5f + static_cast<float>(SplitMix(&rs) % 1001) / 1000.0f);
	}
	// Enemigos: más fuertes primero (empate: nombre). Sitios: más cerca del inicio primero.
	std::vector<size_t> by_f(cs.size()), by_d(cs.size());
	for (size_t i = 0; i < cs.size(); i++) by_f[i] = by_d[i] = i;
	std::stable_sort(by_f.begin(), by_f.end(), [&](size_t a, size_t b) {
		return cs[a].fuerza != cs[b].fuerza ? cs[a].fuerza > cs[b].fuerza : cs[a].name < cs[b].name;
	});
	std::stable_sort(by_d.begin(), by_d.end(), [&](size_t a, size_t b) { return dkey[a] < dkey[b]; });

	// Sitio (MSB) de cada candidato, del archivo original.
	struct Spot {
		uint8_t place[24], groups[32], coll[4], patrol[24];
	};
	std::vector<Spot> sp(cs.size());
	for (size_t i = 0; i < cs.size(); i++) {
		const auto& p  = parts[ps[cs[i].pi].index];
		const size_t td = ps[cs[i].pi].td;
		std::memcpy(sp[i].place, p.data() + 0x20, 24);
		std::memcpy(sp[i].groups, p.data() + 0x44, 32);
		std::memcpy(sp[i].coll, p.data() + td + 0x1C, 4);
		std::memcpy(sp[i].patrol, p.data() + td + 0x20, 24);
	}
	std::vector<uint8_t> cg = tf.d;
	size_t               moved = 0;
	for (size_t k = 0; k < cs.size(); k++) {
		const size_t e = by_f[k]; // enemigo
		const size_t s = by_d[k]; // sitio que recibe
		if (e == s) continue;
		Cg::Delta dl;
		if (!Cg::MakeDelta(cs[e].root.m, cs[s].root.m, &dl)) continue;
		// Aspecto: todas las matrices de su entidad, con el mismo cambio que lleva su raíz al sitio nuevo.
		for (uint32_t o: cs[e].xf) {
			const Cg::Obj& ob = tf.objs[o];
			for (size_t q = Cg::Find(tf.d, ob.pay, ob.pay + ob.len, Cg::kMatrix, 5);
			     q != std::string::npos && q + 53 <= ob.pay + ob.len; q = Cg::Find(tf.d, q + 53, ob.pay + ob.len, Cg::kMatrix, 5)) {
				float m[12];
				std::memcpy(m, tf.d.data() + q + 5, 48);
				Cg::Apply(dl, m);
				std::memcpy(cg.data() + q + 5, m, 48);
			}
		}
		// Lógica: sitio del MSB.
		auto&        p  = parts[ps[cs[e].pi].index];
		const size_t td = ps[cs[e].pi].td;
		std::memcpy(p.data() + 0x20, sp[s].place, 24);
		std::memcpy(p.data() + 0x44, sp[s].groups, 32); // (rando20) grupos del sitio, no los de la zona de origen
		if (move_coll) std::memcpy(p.data() + td + 0x1C, sp[s].coll, 4);
		// (rando22) typedata +0x14, byte bajo: 0x02 solo en los soldados quietos de la sala de 1-1 (c2000_0004/05/06/08/09/
		// 10/12/14, c2010_0017; 9 de 120 enemigos de m02_00); el resto 0x00. Movido, ese estado pasivo hacia que esperase a
		// que le pegasen. Al moverlo va con 0x00, como los demas. KYTY_DS_RANDOMIZER_PASIVOS=1 lo deja como estaba.
		if (!EnvIs("KYTY_DS_RANDOMIZER_PASIVOS", "1")) p[td + 0x14] = 0;
		if (move_patrol) std::memcpy(p.data() + td + 0x20, sp[s].patrol, 24);
		else std::memset(p.data() + td + 0x20, 0xFF, 24); // sin patrulla: no vuelve andando al sitio viejo
		moved++;
		std::printf("[rando] fuertes al principio: %s %s (%s) -> sitio de %s (distancia %.0f m)\n", map.c_str(),
		            cs[e].name.c_str(), cs[e].model.c_str(), cs[s].name.c_str(), static_cast<double>(cs[s].dist));
	}
	if (moved == 0) return false;
	if (!Cg::Check(cg, tf.objs.size())) {
		std::printf("[rando] mapa %s: fuertes al principio: la copia del legacy_instances no se lee, se deja el original\n",
		            map.c_str());
		return false;
	}
	*msb_out  = BuildTables(t);
	*cgpr_out = std::move(cg);
	std::printf("[rando] mapa %s: fuertes al principio: %zu enemigos movidos de %zu (mas fuerte en el sitio mas cercano: %s)\n",
	            map.c_str(), moved, cs.size(), cs[by_f[0]].model.c_str());
	std::fflush(stdout);
	return true;
}

// (demonio) Una sola parte: el soldado c2000_0000 de m02_00_00_00 pasa a ser el Vanguard de m03_01_00_00 (c1030_0000)
// como COPIA de tipo en el hueco del soldado (9.1, 21.2, -126.0, colision 766, primer patio): regla del usuario, nunca se
// mueve nada (moverlo rompe el juego). Antes: centro del patio / arena de la Falange movidos.
// Sin patrulla, sin estado pasivo. Va ENCIMA de lo que haya hecho el randomizer (fuertes /
// barajado): |msb| y |cg| entran ya construidos (vacios: los originales).
// Log 20261127-032608-861: con el modelo c1030 METIDO en la tabla de modelos del MSB (indices de modelo corridos) no
// hubo ni una peticion de c1030 (0x1451700) ni la linea del gancho 0x359990 (comparaba el modelo con "c2000" y el MSB
// ya decia c1030). Ghidra: FUN_003effe0 crea un ChrIns por cada parte de tipo 2 (0x34ae50 -> 0x367220 -> 0x3233d0 ->
// 0x359800, vtable 0x2581ee8); su entrada 5 (0x359990) lee modelo y NpcParam de la tabla de partes en memoria
// (map+0x928 / +0x860, hechas del MSB por FUN_006c2e90) y llama 0x367030 -> 0x366130 -> 0x1451700. Ahora el MSB NO
// cambia de modelo ni de tabla de modelos (soldado c2000, su NpcParam): el cambio a c1030 / 103001 (typedata del
// donante + modelo y chrid) lo hace el gancho en memoria solo para la parte en ese sitio, igual que el resto del mapa.
// Jefe de la carga |load| (0 = primera): el forzado, o uno de Jefes() por la semilla (cambia al morir / recargar).
inline const JefeDef& JefeDe(int load) {
	const auto& j = Jefes();
	const char* f = JefeForzado();
	for (const auto& d: j)
		if (f[0] != '\0' && std::strcmp(f, d.model) == 0) return d;
	const uint64_t h = Fnv(std::string(Seed()) + "/jefe/" + std::to_string(load));
	return j[static_cast<size_t>(h % j.size())];
}
inline bool BuildDemonio(const std::string& map, const std::filesystem::path& app0, std::vector<uint8_t>* msb_io,
                         std::vector<uint8_t>* cgpr_io, int load = 0) {
	static const char* kTarget   = DemonioTarget(); // (rando24) antes c2000_0000, que el juego no crea
	const JefeDef&     jefe      = JefeDe(load);
	const char*        kDonorMap = jefe.map;
	const char*        kDonor    = jefe.donor;
	const std::string  kModel    = jefe.model;
	if (map != "m02_00_00_00") return false;
	auto fail = [&](const char* why) {
		std::printf("[demonio] %s: %s; el jefe %s no se pone\n", map.c_str(), why, jefe.model);
		std::fflush(stdout);
		return false;
	};
	std::vector<uint8_t> msb = *msb_io, dmsb;
	Cg::File             tf;
	tf.d = *cgpr_io;
	if (msb.empty() && !ReadWhole(MsbPathOf(app0, map), &msb)) return fail("no se lee el MSB");
	if (tf.d.empty() && !ReadWhole(CgprPathOf(app0, map), &tf.d)) return fail("no se lee el cgpr");
	if (!Cg::Parse(&tf)) return fail("legacy_instances con formato no reconocido");
	std::vector<Table> t, dt;
	if (!ParseTables(msb, &t) || BuildTables(t) != msb) return fail("MSB no reconocido");
	std::vector<EnemyPart> ps, dps;
	if (!EnemyParts(t, &ps)) return fail("partes del MSB no reconocidas");
	if (!ReadWhole(MsbPathOf(app0, kDonorMap), &dmsb) || !ParseTables(dmsb, &dt) || !EnemyParts(dt, &dps))
		return fail("no se lee el MSB del mapa donante");
	auto& models = t[0].blobs;
	auto& parts  = t[3].blobs;

	// Hueco: el sitio del soldado c2000_0000 en el MSB ORIGINAL (9.1, 21.2, -126.0; primer patio). No se mueve nada:
	// la parte que el randomizer haya dejado en ese sitio (o el propio soldado) conserva sitio, colision, grupos, patrulla
	// y matriz del cgpr; solo cambia su TIPO (modelo c1030 / NpcParam 103001 / typedata en el gancho, recursos de
	// personaje del cgpr del Vanguard).
	std::vector<uint8_t>   omsb;
	std::vector<Table>     ot;
	std::vector<EnemyPart> ops;
	if (!ReadWhole(MsbPathOf(app0, map), &omsb) || !ParseTables(omsb, &ot) || !EnemyParts(ot, &ops)) return fail("no se lee el MSB original");
	size_t oi = ops.size();
	for (size_t i = 0; i < ops.size(); i++)
		if (ops[i].name == kTarget) oi = i;
	if (oi == ops.size()) return fail("no esta el soldado en el MSB original");
	float slot[3];
	std::memcpy(slot, ot[3].blobs[ops[oi].index].data() + 0x20, 12);
	size_t pi = ps.size();
	for (size_t i = 0; i < ps.size(); i++) {
		float q[3];
		std::memcpy(q, parts[ps[i].index].data() + 0x20, 12);
		const float dx = q[0] - slot[0], dy = q[1] - slot[1], dz = q[2] - slot[2];
		if (dx * dx + dy * dy + dz * dz < 0.25f) pi = i;
	}
	if (pi == ps.size()) return fail("no hay ninguna parte en el sitio del soldado");
	const std::string sname = ps[pi].name;
	const std::string smodel = ModelName(models[static_cast<size_t>(ps[pi].model)]);
	size_t         top = 0;
	const Cg::Rec* tr  = TopByName(tf, sname, &top);
	if (tr == nullptr || !Cg::RootInfo(tf, *tr).ok || Cg::RootInfo(tf, *tr).part != ps[pi].index)
		return fail("la parte del sitio no tiene su entidad en el cgpr");

	size_t dpi = dps.size();
	for (size_t i = 0; i < dps.size(); i++)
		if (dps[i].name == kDonor && ModelName(dt[0].blobs[static_cast<size_t>(dps[i].model)]) == kModel) dpi = i;
	if (dpi == dps.size()) return fail("no esta la parte donante en su mapa");
	const auto& dpart = dt[3].blobs[dps[dpi].index];
	uint8_t     td_head[0x14], td_init[4];
	std::memcpy(td_head, dpart.data() + dps[dpi].td, 0x14);
	std::memcpy(td_init, dpart.data() + dps[dpi].td + 0x18, 4);
	const Cg::File* df = DonorFile(app0, kDonorMap);
	if (df == nullptr) return fail("no se lee el legacy_instances.cgpr del mapa donante");
	size_t         dtop = 0;
	const Cg::Rec* dr   = TopByName(*df, kDonor, &dtop);
	if (dr == nullptr || !Cg::RootInfo(*df, *dr).ok) return fail("la parte donante no tiene entidad en el cgpr de su mapa");
	for (const auto& m: models)
		if (BI32(m, 0x8) == 2 && ModelName(m) == kModel) return fail("el modelo ya esta en el mapa");

	auto&        pb  = parts[ps[pi].index];
	const size_t td  = ps[pi].td;
	std::vector<uint8_t> msb2 = msb; // MSB sin tocar

	// Aspecto: entidad entera del Vanguard con la matriz y el enlace al MSB de la parte del sitio (sin mover).
	Cg::Swap sw {top, df, dtop};
	std::vector<uint32_t> refs;
	Cg::CollectRefs(df->tops[dtop], &refs);
	const size_t         new_objs = tf.objs.size() + std::set<uint32_t>(refs.begin(), refs.end()).size();
	std::vector<uint8_t> cg2;
	if (!Cg::Transplant(tf, {sw}, &cg2) || !Cg::Check(cg2, new_objs)) return fail("el resultado no se relee bien");

	HookSwap h;
	h.has_pos = true;
	std::memcpy(h.pos, slot, 12);
	h.name = sname;
	std::snprintf(h.orig_model, sizeof(h.orig_model), "%s", smodel.c_str());
	h.orig_npc = BI32(pb, td + 0xC);
	std::snprintf(h.new_model, sizeof(h.new_model), "%s", jefe.model);
	std::memcpy(&h.new_npc, td_head + 0xC, 4);
	std::memcpy(h.td_head, td_head, sizeof(h.td_head));
	std::memcpy(h.td_init, td_init, sizeof(h.td_init));
	h.owner = Fnv("demonio-jefe");
	{
		std::lock_guard<std::mutex> lock(HookMutex());
		for (auto& o: HookSwaps()) // el jefe de la carga anterior ya no vale (otro tipo en el mismo hueco)
			if (o.owner == h.owner) o.dead = true;
		HookSwaps().push_back(h);
	}
	std::printf("[demonio] %s: hueco del soldado %s (%.1f, %.1f, %.1f), colision %d: alli esta %s (%s, npc %d) -> tipo "
	            "%s npc %d, HP %d, batalla %06d_battle.lua (entidad de %s %s; carga %d); nada se mueve, MSB sin tocar (el tipo lo cambia el gancho); cgpr %zu -> %zu "
	            "objetos\n",
	            map.c_str(), kTarget, static_cast<double>(slot[0]), static_cast<double>(slot[1]), static_cast<double>(slot[2]), BI32(pb, td + 0x1C),
	            sname.c_str(), smodel.c_str(), h.orig_npc, jefe.name, h.new_npc, jefe.hp, jefe.battle, kDonorMap, kDonor, load, tf.objs.size(), new_objs);
	std::fflush(stdout);
	msb_io->swap(msb2);
	cgpr_io->swap(cg2);
	return true;
}

// (rando27) Mini-jefe (ver MiniJefeSlot): copia de c2101_0003 de m05_01 en el hueco c2000_0006 de 1-1. Nada se mueve:
// MSB sin tocar (tipo e IA por los ganchos 0xef430 / 0x359990 / 0x1451700 desde |finals|), cgpr con la entidad del
// esqueleto anclada al hueco (colision del suelo y campos de sitio del hueco). |cg| sin relleno.
inline bool BuildMiniJefe(const std::string& map, const std::filesystem::path& app0, std::vector<uint8_t>* cg,
                          std::vector<SlotFinal>* finals) {
	static const char* kDonorMap = "m05_01_00_00";
	static const char* kDonor    = "c2101_0003";
	auto fail = [&](const char* why) {
		std::printf("[rando] mini-jefe: %s; no se pone\n", why);
		std::fflush(stdout);
		return false;
	};
	if (map != MiniJefeMap()) return false;
	std::vector<uint8_t>   om, dm;
	std::vector<Table>     ot, dt;
	std::vector<EnemyPart> ops, dps;
	if (!ReadWhole(MsbPathOf(app0, map), &om) || !ParseTables(om, &ot) || !EnemyParts(ot, &ops)) return fail("no se lee el MSB de 1-1");
	if (!ReadWhole(MsbPathOf(app0, kDonorMap), &dm) || !ParseTables(dm, &dt) || !EnemyParts(dt, &dps)) return fail("no se lee el MSB de m05_01");
	const EnemyPart* slot = nullptr;
	for (const auto& e: ops)
		if (e.name == MiniJefeSlot()) slot = &e;
	const EnemyPart* don = nullptr;
	for (const auto& e: dps)
		if (e.name == kDonor && ModelName(dt[0].blobs[static_cast<size_t>(e.model)]) == "c2101") don = &e;
	if (slot == nullptr || !slot->eligible) return fail("no esta el hueco c2000_0006");
	if (don == nullptr || !don->eligible) return fail("no esta c2101_0003 en m05_01");
	Cg::File tf;
	tf.d = *cg;
	if (!Cg::Parse(&tf)) return fail("cgpr no reconocido");
	size_t         top = 0, dtop = 0;
	const Cg::Rec* tr  = TopByName(tf, slot->name, &top);
	if (tr == nullptr || !Cg::RootInfo(tf, *tr).ok || Cg::RootInfo(tf, *tr).part != slot->index) return fail("el hueco no tiene su entidad");
	const Cg::File* df = DonorFile(app0, kDonorMap);
	if (df == nullptr) return fail("no se lee m05_01_00_00_legacy_instances.cgpr");
	const Cg::Rec* dr = TopByName(*df, kDonor, &dtop);
	if (dr == nullptr || !Cg::RootInfo(*df, *dr).ok) return fail("c2101_0003 sin entidad");
	std::vector<uint32_t> refs;
	Cg::CollectRefs(df->tops[dtop], &refs);
	std::vector<uint8_t> cg2;
	if (!Cg::Transplant(tf, {Cg::Swap {top, df, dtop, Fnv("minijefe") | 1}}, &cg2) ||
	    !Cg::Check(cg2, tf.objs.size() + std::set<uint32_t>(refs.begin(), refs.end()).size()))
		return fail("el resultado no se relee bien");
	SlotFinal f;
	f.name = slot->name;
	f.model = "c2101";
	const auto& sp = ot[3].blobs[slot->index];
	const auto& dp = dt[3].blobs[don->index];
	std::memcpy(f.pos, sp.data() + 0x20, 12);
	std::memcpy(f.td_head, dp.data() + don->td, 0x14);
	std::memcpy(f.td_init, dp.data() + don->td + 0x18, 4);
	finals->push_back(f);
	int32_t npc = 0;
	std::memcpy(&npc, f.td_head + 0xC, 4);
	std::printf("[rando] mini-jefe: Falange derrotada (4096): %s (%.1f, %.1f, %.1f; colision %d) -> c2101 npc %d (de %s %s)\n",
	            slot->name.c_str(), static_cast<double>(f.pos[0]), static_cast<double>(f.pos[1]), static_cast<double>(f.pos[2]),
	            BI32(sp, slot->td + 0x1C), npc, kDonorMap, kDonor);
	std::fflush(stdout);
	cg->swap(cg2);
	return true;
}

struct RedirectEntry {
	bool                  done = false;
	std::filesystem::path copy; // MSB; vacío: sin copia
	uint64_t              size = 0;
	std::filesystem::path cgpr; // legacy_instances.cgpr; vacío: sin copia
	uint64_t              cgpr_size = 0;
	// (rando26) Turnos. El juego resuelve el tamaño de cada archivo una vez al arrancar (libAmpr: ~250K rutas), así que
	// cada carga sirve archivos del MISMO tamaño: el MSB es siempre el de la primera carga (la mezcla no cambia su
	// tamaño y no se rehace: el tipo de cada hueco en las cargas siguientes lo ponen los ganchos 0xef430 / 0x359990 /
	// 0x1451700 desde el tipo del MSB de la primera carga), y el cgpr va relleno con ceros hasta |cgpr_size| antes del
	// pie (Ghidra FUN_00ba29b0: objetos y registros se leen por recuento, lo que sobra hasta el pie no se lee).
	bool                                           rotate = false;
	int                                            load = 0;        // cargas servidas (0: ninguna)
	std::chrono::steady_clock::time_point          last {};         // última lectura del cgpr desde 0
	std::filesystem::path                          app0;
	std::map<std::string, std::pair<std::string, int32_t>> base; // nombre@pos -> (modelo, npc) en el MSB servido
	std::vector<uint8_t>                           msb0;            // MSB servido (para BuildDemonio en cada carga)
	bool                                           minijefe = false; // (rando27) la copia servida lleva el mini-jefe
};

inline std::mutex& RedirectMutex() {
	static std::mutex m;
	return m;
}
inline std::vector<std::pair<std::string, RedirectEntry>>& RedirectTable() {
	static std::vector<std::pair<std::string, RedirectEntry>> t;
	return t;
}

inline bool IsCgprPath(const std::string& path) {
	static const std::string tail = "_legacy_instances.cgpr";
	if (path.size() < tail.size() + 12) return false;
	std::string low = Lower(path.substr(path.size() - tail.size() - 12));
	for (auto& ch: low)
		if (ch == '\\') ch = '/';
	return low.compare(12, tail.size(), tail) == 0 && low[0] == 'm';
}

inline bool WriteFile(const std::filesystem::path& p, const std::vector<uint8_t>& bytes) {
	std::FILE* f = nullptr;
#ifdef _WIN32
	f = _wfopen(p.wstring().c_str(), L"wb");
#else
	f = std::fopen(p.string().c_str(), "wb");
#endif
	bool ok = f != nullptr && std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
	if (f != nullptr) ok = (std::fclose(f) == 0) && ok;
	return ok;
}

// (rando26) Relleno del cgpr hasta |target| bytes: ceros entre el último registro y el pie "PRPG" (el pie apunta a su sitio).
inline bool PadCgpr(std::vector<uint8_t>* cg, size_t target) {
	if (cg->size() < 12 || target < cg->size() || target > 0xffffffffull) return false;
	const size_t foot = cg->size() - 12;
	if (std::memcmp(cg->data() + foot, "PRPG", 4) != 0) return false;
	cg->insert(cg->begin() + static_cast<ptrdiff_t>(foot), target - cg->size(), 0);
	const uint32_t nf = static_cast<uint32_t>(target - 12);
	std::memcpy(cg->data() + nf + 8, &nf, 4);
	return true;
}

inline std::string BaseKey(const std::string& name, const float* pos) {
	char b[64];
	std::snprintf(b, sizeof(b), "@%.0f,%.0f,%.0f", static_cast<double>(pos[0]), static_cast<double>(pos[1]), static_cast<double>(pos[2]));
	return name + b;
}

// (rando26) Ganchos de una carga: cada hueco cuyo tipo final no es el del MSB servido. Las de la carga anterior del
// mapa quedan muertas (no se borran: Pending las puede apuntar).
inline void SetHooks(const std::string& map, int load, const std::vector<SlotFinal>& finals, const RedirectEntry& e) {
	const uint64_t owner = Fnv(map, Fnv("rando26"));
	std::lock_guard<std::mutex> lock(HookMutex());
	for (auto& h: HookSwaps())
		if (h.owner == owner) h.dead = true;
	size_t n = 0;
	for (const auto& f: finals) {
		const auto it = e.base.find(BaseKey(f.name, f.pos));
		if (it == e.base.end()) continue;
		int32_t npc = 0;
		std::memcpy(&npc, f.td_head + 0xC, 4);
		if (it->second.first == f.model && it->second.second == npc) continue;
		HookSwap h;
		h.has_pos = true;
		std::memcpy(h.pos, f.pos, 12);
		h.name = f.name;
		std::snprintf(h.orig_model, sizeof(h.orig_model), "%s", it->second.first.c_str());
		h.orig_npc = it->second.second;
		std::snprintf(h.new_model, sizeof(h.new_model), "%s", f.model.c_str());
		h.new_npc = npc;
		std::memcpy(h.td_head, f.td_head, sizeof(h.td_head));
		std::memcpy(h.td_init, f.td_init, sizeof(h.td_init));
		h.owner = owner;
		HookSwaps().push_back(h);
		n++;
	}
	std::printf("[rando] mapa %s: carga %d: %zu huecos con el tipo puesto por gancho (IA en 0xef430)\n", map.c_str(), load, n);
}

inline std::filesystem::path CopyDir() {
	std::error_code ec;
	const auto      dir = std::filesystem::temp_directory_path(ec) / "kyty-rando" /
	                 Utf8Path(DemonioOn() ? std::string("demonio-") + Seed() : std::string(Seed()));
	if (!ec) std::filesystem::create_directories(dir, ec);
	return ec ? std::filesystem::path() : dir;
}

// Copias del mapa (MSB + legacy_instances), una vez por mapa. |app0| sale de la ruta pedida.
inline RedirectEntry& EntryFor(const std::string& map, const std::filesystem::path& app0) {
	for (auto& [name, e]: RedirectTable())
		if (name == map) return e;
	RedirectEntry        e;
	e.done = true;
	e.app0 = app0;
	std::vector<uint8_t> msb, cg;
	std::vector<SlotFinal> finals;
	// (demonio) El Vanguard va ENCIMA de lo que haga el randomizer (antes lo sustituia: sin "[rando] mapa" en ningun mapa).
	bool built = false;
	const bool scope = InScope(map); // (rando28) mundos 1-5
	CurSeed().clear();
	if (scope && MezclaOn()) built = BuildMezcla(map, app0, &msb, &cg, &finals);
	else if (scope && FuertesOn()) built = BuildFuertes(map, app0, &msb, &cg);
	else if (scope && CrossWorldsOn()) built = BuildCross3(map, app0, &msb, &cg);
	if (built && !DynVisualOn() && !FuertesOn() && Enabled()) ShuffleBytes(map, msb); // barajado de la fase 1 en la copia
	const bool mezcla = built && MezclaOn();
	if (DemonioOn() && map == "m02_00_00_00") {
		if (!built) { // sin copia del randomizer: los originales (modo barajar: barajados antes, como en memoria)
			msb.clear();
			cg.clear();
			if (Enabled() && !FuertesOn() && !CrossWorldsOn() && ReadWhole(MsbPathOf(app0, map), &msb)) ShuffleBytes(map, msb);
		}
		built = BuildDemonio(map, app0, &msb, &cg) || built;
	}
	if (DynVisualOn()) { // (rando16) indice de partes de TODOS los mapas, tal como los lee el juego
		std::vector<uint8_t> orig;
		if (built) DynRegister(map, msb);
		else if (ReadWhole(MsbPathOf(app0, map), &orig)) DynRegister(map, orig);
	}
	if (mezcla) { // (rando26) tipo de cada hueco en el MSB servido; ganchos de los de otros mundos
		std::vector<Table>     t;
		std::vector<EnemyPart> ps;
		if (ParseTables(msb, &t) && EnemyParts(t, &ps))
			for (const auto& pe: ps) {
				const auto& p = t[3].blobs[pe.index];
				float       pos[3];
				std::memcpy(pos, p.data() + 0x20, 12);
				e.base[BaseKey(pe.name, pos)] = {ModelName(t[0].blobs[static_cast<size_t>(pe.model)]), BI32(p, pe.td + 0xC)};
			}
		SetHooks(map, 1, finals, e);
		e.rotate = RotarOn();
		if (e.rotate) {
			e.msb0 = msb;
			// Hueco para las cargas siguientes: otros donantes, otras entidades. 25% (minimo 2 MB) sobre la primera.
			const size_t cap = cg.size() + std::max<size_t>(cg.size() / 4, 2u << 20);
			if (!PadCgpr(&cg, cap)) e.rotate = false;
		}
	}
	if (built) {
		// (rando16) Con entidad visual trasplantada no se baraja: la matriz de cada entidad de Bluepoint es la del sitio
		// original de su parte; si la parte se mueve, aspecto y logica quedan en sitios distintos.
		const auto dir = CopyDir();
		const auto cm  = dir / Utf8Path(map + ".msb");
		const auto cc  = dir / Utf8Path(map + "_legacy_instances.cgpr");
		if (!dir.empty() && WriteFile(cm, msb) && WriteFile(cc, cg)) {
			e.copy      = cm;
			e.size      = msb.size();
			e.cgpr      = cc;
			e.cgpr_size = cg.size();
			std::printf("[rando] mapa %s: copias %s (%zu bytes) y %s (%zu bytes)%s\n", map.c_str(), PathUtf8(cm).c_str(),
			            msb.size(), PathUtf8(cc).c_str(), cg.size(), e.rotate ? "; turnos: otra mezcla en cada carga" : "");
		} else {
			e.rotate = false;
			std::printf("[rando] mapa %s: no se pudieron escribir las copias, solo barajado\n", map.c_str());
		}
		std::fflush(stdout);
	}
	RedirectTable().emplace_back(map, std::move(e));
	return RedirectTable().back().second;
}

// (rando26) Una lectura del cgpr del mapa desde el byte 0 = una carga del mapa (al arrancar, al morir, al descansar, al
// volver). Desde la segunda (separada al menos 3 s de la anterior) se monta otra mezcla con la semilla "<semilla>/cN":
// cgpr nuevo (mismo tamaño, relleno) en un archivo nuevo y ganchos nuevos; el MSB es el mismo.
inline void NoteLoad(const std::string& map, RedirectEntry& e) {
	const auto now = std::chrono::steady_clock::now();
	if (e.load > 0 && now - e.last < std::chrono::seconds(3)) {
		e.last = now;
		return;
	}
	e.last = now;
	const bool mj = map == MiniJefeMap() && FalangeMuerta(); // (rando27)
	if (e.load++ == 0 && mj == e.minijefe) return;           // primera carga: la copia ya hecha (sin mini-jefe)
	for (int attempt = 0; attempt < 4; attempt++) {
		CurSeed() = std::string(Seed()) + "/c" + std::to_string(e.load) + (attempt > 0 ? "t" + std::to_string(attempt) : "");
		std::vector<uint8_t>   msb, cg;
		std::vector<SlotFinal> finals;
		const bool             ok = BuildMezcla(map, e.app0, &msb, &cg, &finals);
		CurSeed().clear();
		if (!ok) continue;
		if (DemonioOn() && map == "m02_00_00_00") {
			std::vector<uint8_t> m0 = e.msb0; // el Vanguard: mismo hueco, sobre el MSB servido
			BuildDemonio(map, e.app0, &m0, &cg, e.load);
		}
		const bool mj_ok = mj && BuildMiniJefe(map, e.app0, &cg, &finals);
		if (!PadCgpr(&cg, e.cgpr_size)) {
			std::printf("[rando] mapa %s: carga %d: el cgpr no cabe (%zu > %llu), otro intento\n", map.c_str(), e.load, cg.size(),
			            static_cast<unsigned long long>(e.cgpr_size));
			continue;
		}
		const auto dir = CopyDir();
		const auto cc  = dir / Utf8Path(map + "_legacy_instances.c" + std::to_string(e.load) + ".cgpr");
		if (dir.empty() || !WriteFile(cc, cg)) break;
		std::error_code ec;
		if (e.load >= 3) std::filesystem::remove(dir / Utf8Path(map + "_legacy_instances.c" + std::to_string(e.load - 2) + ".cgpr"), ec);
		e.cgpr     = cc;
		e.minijefe = mj_ok;
		SetHooks(map, e.load, finals, e);
		std::printf("[rando] mapa %s: carga %d: mezcla nueva (semilla %s/c%d) en %s\n", map.c_str(), e.load, Seed(), e.load,
		            PathUtf8(cc).c_str());
		std::fflush(stdout);
		return;
	}
	std::printf("[rando] mapa %s: carga %d: no se pudo montar otra mezcla, se repite la anterior\n", map.c_str(), e.load);
	std::fflush(stdout);
}

// (demonio/mundos) Ghidra: FUN_000ef430 (desde FUN_000f13a0, "ai:/ai_script.loadlist") carga por cada parte del mapa
// "ai:/out/%06d_logic.lua" / "_battle.lua" (FUN_000ef990 / FUN_000efad0) del archivo del area ("script:/m%02d.luabnd").
// m02.luabnd (Boletaria, BND3 sin comprimir, 66 archivos) NO trae 103001_battle.lua (el Vanguard, solo en m03.luabnd):
// con el tipo bien puesto el juego no encuentra su script de combate y el Vanguard se queda quieto. Lo mismo para los
// enemigos de otros mundos (MUNDOS SI). Se sirve una copia de m02.luabnd con los ai\out\*.lua que le faltan, sacados
// de los luabnd de las otras areas (el primero que lo tenga); lo demas, byte a byte igual.
struct BndFile {
	uint32_t             flags = 0;
	int32_t              id    = 0;
	std::string          name;
	std::vector<uint8_t> data;
};

inline bool ParseBnd3(const std::vector<uint8_t>& b, std::vector<BndFile>* out) {
	if (b.size() < 0x20 || std::memcmp(b.data(), "BND3", 4) != 0 || b[0xC] != 0x74 || b[0xD] != 0 || b[0xE] != 0) return false;
	uint32_t n = 0;
	std::memcpy(&n, b.data() + 0x10, 4);
	if (n == 0 || n > 100000 || 0x20 + static_cast<uint64_t>(n) * 0x18 > b.size()) return false;
	for (uint32_t i = 0; i < n; i++) {
		uint32_t v[6];
		std::memcpy(v, b.data() + 0x20 + i * 0x18, sizeof(v)); // flags, tamano, offset, id, nombre, tamano sin comprimir
		if (v[1] != v[5] || static_cast<uint64_t>(v[2]) + v[1] > b.size() || v[4] >= b.size()) return false;
		BndFile f;
		f.flags = v[0];
		std::memcpy(&f.id, &v[3], 4);
		for (size_t k = v[4]; k < b.size() && b[k] != 0; k++) f.name.push_back(static_cast<char>(b[k]));
		f.data.assign(b.begin() + v[2], b.begin() + v[2] + v[1]);
		out->push_back(std::move(f));
	}
	return true;
}

inline std::vector<uint8_t> BuildBnd3(const std::vector<uint8_t>& orig, const std::vector<BndFile>& fs) {
	std::vector<uint8_t> b(orig.begin(), orig.begin() + 0x20); // cabecera original (version, formato)
	const uint32_t       n = static_cast<uint32_t>(fs.size());
	std::memcpy(b.data() + 0x10, &n, 4);
	b.resize(0x20 + static_cast<size_t>(n) * 0x18, 0);
	std::vector<uint32_t> noff(n);
	for (uint32_t i = 0; i < n; i++) {
		noff[i] = static_cast<uint32_t>(b.size());
		b.insert(b.end(), fs[i].name.begin(), fs[i].name.end());
		b.push_back(0);
	}
	const uint32_t hend = static_cast<uint32_t>(b.size());
	std::memcpy(b.data() + 0x14, &hend, 4);
	for (uint32_t i = 0; i < n; i++) {
		b.resize((b.size() + 15) & ~static_cast<size_t>(15), 0);
		const uint32_t v[6] {fs[i].flags, static_cast<uint32_t>(fs[i].data.size()), static_cast<uint32_t>(b.size()),
		                     static_cast<uint32_t>(fs[i].id), noff[i], static_cast<uint32_t>(fs[i].data.size())};
		std::memcpy(b.data() + 0x20 + i * 0x18, v, sizeof(v));
		b.insert(b.end(), fs[i].data.begin(), fs[i].data.end());
	}
	return b;
}

inline bool IsAiScript(const std::string& name) {
	std::string low = Lower(name);
	for (auto& ch: low)
		if (ch == '/') ch = '\\';
	return low.find("\\ai\\out\\") != std::string::npos && low.size() > 4 && low.compare(low.size() - 4, 4, ".lua") == 0;
}

// (jefe) Ghidra: la funcion Lua IsInsideTargetRegion es FUN_000f6490 (registrada en 0x104973/0x104a01 como
// "IsInsideTargetRegion") -> FUN_00201f90: busca la region por su id de entidad en el MSB cargado (FUN_0020d6d0) y si no
// existe devuelve 0 (falso). 103001_battle.lua (Vanguard de m03_01) solo ataca si el jugador esta en las regiones 2275 /
// 2276 del MSB de m03_01_00_00 ("debu103001Battle_IsEnableAttack") y solo persigue si esta en 2270 y el Vanguard fuera
// de 2271. Ninguna existe en m02_00_00_00 (POINT_PARAM_ST de ambos MSB): en el patio la copia siempre "espera 1 s
// mirando al objetivo" (GOAL_COMMON_Wait) y nunca pega. En la copia del script que va a otras areas esas pruebas de
// region pasan a pruebas de distancia (persigue si esta a mas de 6 m, ataca siempre que el reparto por distancia lo
// elija, vuelve a su sitio solo si se aleja mas de 15 m); los ataques (3000..3301) y sus repartos, sin tocar.
inline bool SinRegionesVanguard(std::vector<uint8_t>* d) {
	std::string t(d->begin(), d->end());
	const std::pair<const char*, const char*> rep[] = {
	    {"if ( ai:IsInsideTargetRegion(TARGET_ENE_0, 2270) ) then", "if ( true ) then"},
	    {"if ( not(ai:IsInsideTargetRegion(TARGET_SELF, 2271)) ) then", "if ( 6 < ai:GetDist(TARGET_ENE_0) ) then"},
	    {"if ( ai:IsInsideTargetRegion(TARGET_ENE_0, 2275) ) then return true; end", "if ( true ) then return true; end"},
	    {"if ( ai:IsInsideTargetRegion(TARGET_ENE_0, 2276) ) then return true; end", ""},
	    {"ai:AddObserveRegion( 0, TARGET_SELF, 2271 );", ""},
	    {"TARGET_ENE_0, APPROACH_DIST, AI_DIR_TYPE_CENTER", "TARGET_ENE_0, 6, AI_DIR_TYPE_CENTER"},
	    {"\tlocal BACKHOME_DIST = 3;", "\tlocal BACKHOME_DIST = 15;"},
	};
	for (const auto& r: rep) {
		const size_t at = t.find(r.first);
		if (at == std::string::npos || t.find(r.first, at + 1) != std::string::npos) return false;
		t.replace(at, std::strlen(r.first), r.second);
	}
	if (t.find("IsInsideTargetRegion") != std::string::npos || t.find("ObserveRegion( 0") != std::string::npos) return false;
	d->assign(t.begin(), t.end());
	return true;
}

struct LuabndCopy {
	bool                  done = false;
	std::filesystem::path copy;
	uint64_t              size = 0;
};
// (rando28) Una copia por area (m02..m06): "m02".."m06".
inline LuabndCopy& AreaLuabnd(const std::string& area) {
	static std::map<std::string, LuabndCopy> t;
	return t[area];
}

// |real| = .../dvdroot/script/m02.luabnd. Copia hecha una vez (bajo RedirectMutex).
inline const LuabndCopy& LuabndFor(const std::filesystem::path& real, const std::string& self) {
	auto& c = AreaLuabnd(self);
	if (c.done) return c;
	c.done = true;
	std::vector<uint8_t> orig;
	std::vector<BndFile> fs;
	if (!ReadWhole(real, &orig) || !ParseBnd3(orig, &fs)) {
		std::printf("[rando] %s.luabnd: no se lee o formato no reconocido; scripts de IA originales\n", self.c_str());
		std::fflush(stdout);
		return c;
	}
	std::set<std::string> have;
	int32_t               next_id = 0;
	for (const auto& f: fs) {
		have.insert(Lower(f.name));
		next_id = std::max(next_id, f.id + 1);
	}
	const size_t n0 = fs.size();
	bool         vang = false;
	for (const char* area: {"m03", "m02", "m01", "m04", "m05", "m06", "m08"}) {
		if (self == area) continue;
		std::vector<uint8_t> d;
		std::vector<BndFile> dfs;
		if (!ReadWhole(real.parent_path() / (std::string(area) + ".luabnd"), &d) || !ParseBnd3(d, &dfs)) continue;
		for (auto& f: dfs) {
			if (!IsAiScript(f.name) || !have.insert(Lower(f.name)).second) continue;
			for (const auto& j: Jefes())
				if (Lower(f.name).find(std::to_string(j.battle) + "_battle.lua") != std::string::npos) vang = true;
			f.id = next_id++;
			if (self != "m03" && Lower(f.name).find("103001_battle.lua") != std::string::npos) {
				const bool ok = SinRegionesVanguard(&f.data);
				std::printf("[jefe] %s.luabnd: 103001_battle.lua: %s\n", self.c_str(),
				            ok ? "pruebas de region de m03_01 (2270/2271/2275/2276) cambiadas por distancia: persigue y ataca"
				               : "texto no reconocido, se deja el original (sin ataques fuera de m03_01)");
			}
			fs.push_back(std::move(f));
		}
	}
	if (fs.size() == n0) return c;
	const auto bytes = BuildBnd3(orig, fs);
	std::vector<BndFile> check;
	const auto dir = CopyDir();
	const auto cp  = dir / Utf8Path(self + ".luabnd");
	if (!ParseBnd3(bytes, &check) || check.size() != fs.size() || dir.empty() || !WriteFile(cp, bytes)) {
		std::printf("[rando] %s.luabnd: la copia no se pudo hacer; scripts de IA originales\n", self.c_str());
		std::fflush(stdout);
		return c;
	}
	c.copy = cp;
	c.size = bytes.size();
	std::printf("[rando] %s.luabnd: copia con %zu scripts de IA de otras areas (%zu -> %zu archivos, %zu -> %zu bytes)%s\n",
	            self.c_str(), fs.size() - n0, n0, fs.size(), orig.size(), bytes.size(),
	            vang ? "; incluye scripts de combate de los jefes del patio" : "");
	if (DemonioOn() && self == "m02")
		for (const auto& j: Jefes()) {
			bool has = false;
			for (const auto& f: fs) has = has || Lower(f.name).find(std::to_string(j.battle) + "_battle.lua") != std::string::npos;
			std::printf("[jefe] m02.luabnd: %06d_battle.lua (%s): %s\n", j.battle, j.model, has ? "esta" : "NO ESTA (sin IA)");
		}
	std::fflush(stdout);
	return c;
}

// (rando28) ".../script/mXX.luabnd" de un mundo del randomizer (m02..m06) -> "mXX"; si no, "".
inline std::string AreaLuabndOf(const std::string& path) {
	std::string low = Lower(path);
	for (auto& ch: low)
		if (ch == '\\') ch = '/';
	if (low.size() < 18 || low.compare(low.size() - 18, 8, "/script/") != 0 || low.compare(low.size() - 7, 7, ".luabnd") != 0)
		return {};
	const std::string area = low.substr(low.size() - 10, 3);
	return InScope(area + "_") ? area : std::string();
}
// Vanguard (solo m02) o MUNDOS SI (cualquier mundo con donantes de otros mundos).
inline bool LuabndOn(const std::string& area) { return MundosOn() || (DemonioOn() && area == "m02"); }

} // namespace Detail

// Copia del MSB o del legacy_instances.cgpr con enemigos de otros mundos, o vacío si no hay. Se crea una vez por mapa.
// |offset| = 0: lectura desde el principio (una carga, rando26); -1: solo tamaño / sin lectura.
inline std::filesystem::path RedirectRead(const std::string& path, int64_t offset = -1) {
	using namespace Detail;
	if (!CrossOn()) return {};
	if (DemonioOn() || MezclaHooksOn()) InstallModelProbe(); // ganchos y trazas antes de cargar ningun mapa
	if (const std::string area = AreaLuabndOf(path); !area.empty()) { // scripts de IA que le faltan al mundo
		if (!LuabndOn(area)) return {};
		std::lock_guard<std::mutex> lock(RedirectMutex());
		return LuabndFor(Utf8Path(path), area).copy;
	}
	const bool msb = IsMsbPath(path);
	const bool cg  = !msb && IsCgprPath(path);
	if (!msb && !cg) return {};
	const std::filesystem::path real = Utf8Path(path);
	std::string map = Lower(MapName(path));
	if (cg) map = map.substr(0, map.size() - std::string("_legacy_instances").size());
	// app0/cp11demonssouls/dvdroot/map/mapstudio/X.msb y app0/scripts/worlds/X/_cmn/X_legacy_instances.cgpr
	const auto app0 = real.parent_path().parent_path().parent_path().parent_path().parent_path();
	std::lock_guard<std::mutex> lock(RedirectMutex());
	auto& e = EntryFor(map, app0);
	if (cg && offset == 0 && e.rotate) NoteLoad(map, e);
	return msb ? e.copy : e.cgpr;
}

inline std::optional<uint64_t> RedirectedSize(const std::string& path) {
	if (RedirectRead(path).empty()) return std::nullopt;
	if (const std::string area = Detail::AreaLuabndOf(path); !area.empty()) {
		std::lock_guard<std::mutex> lock(Detail::RedirectMutex());
		return Detail::AreaLuabnd(area).size;
	}
	const bool  msb = IsMsbPath(path);
	std::string map = Detail::Lower(MapName(path));
	if (!msb) map = map.substr(0, map.size() - std::string("_legacy_instances").size());
	std::lock_guard<std::mutex> lock(Detail::RedirectMutex());
	for (const auto& [name, e]: Detail::RedirectTable())
		if (name == map) return msb ? e.size : e.cgpr_size;
	return std::nullopt;
}

// Fase 1 sobre lo leído en memoria. Si el mapa ya va por la copia (fase 2), la copia ya está barajada: no se toca.
inline bool Shuffle(const std::string& path, std::vector<uint8_t>& bytes) {
	if (!InScope(Detail::Lower(MapName(path)))) return false; // (rando28) mundos 1-5
	if (FuertesOn()) return false; // (rando18) va por la copia (lógica + aspecto juntos) o se deja el original
	if ((CrossWorldsOn() || DemonioOn()) && !RedirectRead(path).empty()) return false; // la copia ya va barajada
	return ShuffleBytes(MapName(path), bytes);
}

} // namespace RandoMsb
