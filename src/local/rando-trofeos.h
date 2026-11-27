#ifndef EMULATOR_SRC_LOCAL_RANDO_TROFEOS_H_
#define EMULATOR_SRC_LOCAL_RANDO_TROFEOS_H_

// (rando-trofeos) "Trofeos Randomizer" por bandera de jefe muerto, ademas del DsActivityEnd "completed" (libNet.cpp).
// Para que: el trofeo salta al matar al jefe aunque el juego no cierre la actividad como "completed".
// Datos (Ghidra, eboot 01.007.000):
//  - FUN_002a7550(bits, flag): flag < 0x4680 -> (bits[flag >> 3] >> (flag & 7)) & 1.
//  - bits = *(eboot + 0x27bf328) (el getter FUN_002edd40 lo devuelve y se pasa tal cual como bits; 0 sin partida).
//  - La bandera de cada jefe: la tabla de actividades del juego (trophy.cpp kRandoBosses).
// Regla (sin falsos positivos al cargar una partida con el jefe ya muerto): el jefe se arma al empezar su combate
// (DsActivityStart de su actividad) solo si su bandera esta a 0; salta cuando esa bandera pasa a 1 con el mismo
// puntero de banderas. Si el puntero cambia (titulo / otra partida) se desarma todo. Nada de esto fuera del randomizer.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace RandoTrofeos {

constexpr uint32_t kFlagLimit = 0x4680;
constexpr uint64_t kFlagsPtr  = 0x900000000ull + 0x27bf328ull;

inline int FlagBit(const uint8_t* bits, uint32_t flag) {
	if (bits == nullptr || flag == 0 || flag >= kFlagLimit) return -1;
	return (bits[flag >> 3u] >> (flag & 7u)) & 1;
}

// read(ptr*): devuelve el puntero de banderas actual (0 si no hay) y lee la bandera; -1 si no se puede leer.
using ReadFn = int (*)(uint32_t flag, uint64_t* ptr);

class Watch {
public:
	struct Armed {
		std::string activity;
		uint32_t    flag;
		uint64_t    ptr;
	};
	// El combate de ese jefe empieza: armado solo si la bandera se lee a 0.
	bool Arm(const std::string& activity, uint32_t flag, ReadFn read) {
		if (flag == 0) return false;
		uint64_t  ptr = 0;
		const int v   = read(flag, &ptr);
		if (v != 0 || ptr == 0) return false;
		for (auto& a: armed)
			if (a.activity == activity) {
				a.ptr = ptr;
				return true;
			}
		armed.push_back({activity, flag, ptr});
		return true;
	}
	// Los jefes muertos desde que se armaron (se desarman); los de otro puntero se desarman sin saltar.
	std::vector<std::string> Poll(ReadFn read) {
		std::vector<std::string> out;
		for (size_t i = 0; i < armed.size();) {
			uint64_t  ptr = 0;
			const int v   = read(armed[i].flag, &ptr);
			if (ptr != armed[i].ptr) {
				armed.erase(armed.begin() + static_cast<std::ptrdiff_t>(i));
				continue;
			}
			if (v == 1) {
				out.push_back(armed[i].activity);
				armed.erase(armed.begin() + static_cast<std::ptrdiff_t>(i));
				continue;
			}
			i++;
		}
		return out;
	}
	bool Empty() const { return armed.empty(); }

private:
	std::vector<Armed> armed;
};

} // namespace RandoTrofeos

#endif
