#pragma once
// [audio-modos] Ajustes > Sonido: "Modo mezcla mono" / "Modo medianoche" (Demon's Souls, Scream/NCA).
// Ghidra: los dos modos los hace el propio DSP del juego (Scream NCA, enlazado estatico), no AudioOut2Mastering:
//   FUN_008c3610 (callback sound_screamEnableMonoMix) -> snd_SynthPremasterSubmixSetOutputDest(0/3, 0x2000000 Fixed 7.1 |
//   0x800000 3D PassThru) solo con mono apagado; con mono encendido no redirige y el juego mezcla en su puerto principal.
//   FUN_008c7a60 ApplyScreamMasterBussEffect -> snd_ActivateMixSnapshot("Midnight_Mode") + snd_ApplyEffectPreset.
// Al cambiar de modo, el juego deja de dar PCM a puertos que antes alimentaba cada grano (issue #40 ya lo cubria para
// los puertos objeto). Un puerto NO objeto sin PCM nuevo repetia su ultimo grano (zumbido/chasquidos).
// KYTY_AUDIO_MODOS=1: un puerto no objeto que el juego alimentaba grano a grano (>=100 granos seguidos) y deja de recibir
//   PCM suena en silencio ese grano (en la consola el PCM de un grano vale solo para ese grano). Apagado por defecto.
// Ademas: puerto (principal u objeto) con el mismo grano no silencioso 3 veces seguidas = buffer muerto -> silencio.
// Linea [audio-modos] (siempre, solo en eventos): puerto que deja/vuelve a ser alimentado, y llamadas a
//   AudioOut2MasteringSetParam / ContextSetAttributes / MasteringInit con sus parametros.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace AudioModos {

inline bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_AUDIO_MODOS");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

// A port fed bit-identical, non-silent PCM (first 1024 bytes) this many grains in a row is a dead buffer the game
// no longer writes (the steady beep after Mono Mix OFF): silenced / left out of the object mix.
inline constexpr uint32_t REPEAT_STALE = 3;

inline std::atomic<uint32_t> g_lines {0};
inline bool                  Allow() { return g_lines.fetch_add(1, std::memory_order_relaxed) < 200; }

// Buffer of zeros for a grain (max 16 ch * 4 bytes * frames).
inline const void* Silence(size_t bytes) {
	thread_local std::vector<uint8_t> zeros;
	if (zeros.size() < bytes) zeros.assign(bytes, 0);
	return zeros.data();
}

} // namespace AudioModos
