#pragma once
// [volumen-general] Lanzador > Sonido: "Volumen general" (settings.ini volumen_general, 50-300 %, 100 = sin cambios).
// Por que: el juego mezcla bajo (puerto principal Float8ChStd con picos 0.03-0.11 = -19..-30 dBFS). En la PS5 el
// mastering del sistema (AudioOut2MasteringSetParam: compresor/limitador/sonoridad) da la sonoridad final; el emulador
// no lo hace. Ghidra: el juego solo tiene su propio "sound_masterVolume" (FUN_008c0840, cvar por defecto 1.0f) y su
// rack Scream "Master Standard (Fixed 7.1)"; no hay ganancia de sistema que el emulador olvide aplicar.
// Aplicado a la salida final de cada puerto que va al dispositivo de audio del PC (no a la vibracion ni al altavoz del
// mando), con limitador de pico: ataque instantaneo (sin sobrepasar nunca 0.89 = -1 dBFS), release ~100 ms, y un
// soft clip (tanh) de seguridad por encima del umbral. Con 100 % no se toca nada (ni copia).
// Fuente: PortLive::volumen (en vivo desde el menu) o KYTY_VOLUMEN_GENERAL (porcentaje) del lanzador.
#include "local/port-live.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace VolumenGeneral {

inline float Gain() {
	static const float env = [] {
		const char* v = std::getenv("KYTY_VOLUMEN_GENERAL");
		const float p = (v != nullptr && *v != '\0') ? static_cast<float>(std::atof(v)) : 100.0f;
		return std::clamp(p > 0.0f ? p : 100.0f, 50.0f, 300.0f) / 100.0f;
	}();
	const float live = PortLive::volumen.load(std::memory_order_relaxed);
	return live > 0.0f ? std::clamp(live, 0.5f, 3.0f) : env;
}

struct Limiter {
	float env = 1.0f; // current gain reduction (1 = none)
};

// Interleaved float frames, in place.
inline void Process(Limiter* st, float* s, uint32_t frames, uint32_t channels, uint32_t freq, float gain) {
	constexpr float CEIL = 0.89125094f; // -1 dBFS
	const float     rel  = 1.0f - std::exp(-1.0f / (0.1f * static_cast<float>(freq != 0 ? freq : 48000u)));
	float           g    = st->env;
	for (uint32_t f = 0; f < frames; f++) {
		float* x    = s + static_cast<size_t>(f) * channels;
		float  peak = 0.0f;
		for (uint32_t c = 0; c < channels; c++) peak = std::max(peak, std::fabs(x[c] * gain));
		const float target = peak > CEIL ? CEIL / peak : 1.0f;
		g = target < g ? target : g + (target - g) * rel; // instant attack, ~100 ms release
		const float k = gain * g;
		for (uint32_t c = 0; c < channels; c++) {
			float y = x[c] * k;
			const float a = std::fabs(y);
			if (a > CEIL) { // safety soft clip (only on what the limiter did not catch: NaN-free, never > 1)
				const float over = (a - CEIL) / (1.0f - CEIL);
				y = std::copysign(CEIL + (1.0f - CEIL) * std::tanh(over), y);
			}
			x[c] = y;
		}
	}
	st->env = g;
}

inline void ProcessS16(Limiter* st, int16_t* s, uint32_t frames, uint32_t channels, uint32_t freq, float gain) {
	thread_local float tmp[16 * 4096];
	const size_t       n = static_cast<size_t>(frames) * channels;
	if (channels == 0 || n > sizeof(tmp) / sizeof(tmp[0])) return;
	for (size_t i = 0; i < n; i++) tmp[i] = static_cast<float>(s[i]) / 32768.0f;
	Process(st, tmp, frames, channels, freq, gain);
	for (size_t i = 0; i < n; i++) s[i] = static_cast<int16_t>(std::clamp(tmp[i] * 32768.0f, -32768.0f, 32767.0f));
}

} // namespace VolumenGeneral
