#pragma once
// (golpes) Enemigos que no reciben golpes / no mueren. Solo medicion, no cambia nada del juego. (golpes2) Siempre
// encendido (KYTY_GOLPES=0 lo apaga), en silencio: [golpes] ALERTA una vez por chr/id/motivo y resumen cada 60 s solo si hay algo.
//
// Ghidra: FUN_003286c0 = DemonsSoul::ChrIns::BP_CheckForExclude(dl_float32 dt [xmm0], dl_bool) const, rdi = ChrIns,
// esi = bool. Llamada desde FUN_00349410 (0x34951d). Es el detector propio de Bluepoint ("BUG TO TED! %d
// BP_CheckForExclude: c%d: %d: %s"); con bool == 0 comprueba, en este orden:
//   ZeroHp not dead          : [chr+0x710] < 1 y [[[chr+0x38]+0x40]+0x170] == 0 (sin vida y sin marca de muerto)
//   ForceDead                : [chr+0x581] & 2
//   SpEffectNeedUpdatePause  : [[chr+0x560]+0x10] == 1
//   m_bAfterWarpResetRequest : [chr+0x582] & 8 (tambien con bool != 0)
// y si alguno se cumple llama a FUN_003282f0 (excluir al personaje). [chr+0xa0] = chr (c%d), [chr+0xac] = id.
// Aqui, a la entrada, se repiten esas mismas lecturas (las mismas que hace el juego en las mismas condiciones) y se
// cuentan por motivo; cada 5 s una linea [golpes] con llamadas, motivos y los chr con "ZeroHp not dead".
// GIScene::GetOcclusionResult (FUN_009a4c20) es de luces (GI), no de personajes: no se mide.

#include "local/rando-msb.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <tuple>

#if defined(_WIN32) && (defined(__clang__) || defined(__GNUC__))
#include <chrono>

namespace Golpes {

inline bool On() {
	static const bool on = [] {
		const char* t = std::getenv("KYTY_GOLPES"); // (golpes2) siempre encendido; KYTY_GOLPES=0 lo apaga
		return !(t != nullptr && t[0] == '0');
	}();
	return on;
}

enum Motivo : int { MCalls, MCheck, MZeroHp, MForceDead, MSpPause, MWarp, MCount };
inline std::atomic<uint64_t> g_n[MCount] {};
inline std::atomic<uint64_t> g_last_ms {0};
constexpr int                kIds = 8;
inline std::atomic<int32_t>  g_zero_ids[kIds] {};
inline std::atomic<int>      g_zero_next {0};

inline uint64_t NowMs() {
	return static_cast<uint64_t>(
	           std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()) + 1;
}
inline std::atomic<uint64_t> g_t0 {0};
// (golpes2) Una linea por chr (+0xa0), id (+0xac) y motivo, la primera vez; como mucho 300 en total.
inline void Alert(const uint8_t* b, int motivo) {
	static const char* const names[MCount] {"", "", "ZeroHp-no-muerto (sin vida y no muere)", "ForceDead", "SpEffectPausa", "WarpReset"};
	static std::mutex                                  m;
	static std::set<std::tuple<int32_t, int32_t, int>> seen;
	const int32_t                                      chr = *reinterpret_cast<const int32_t*>(b + 0xa0);
	const int32_t                                      id  = *reinterpret_cast<const int32_t*>(b + 0xac);
	{
		std::lock_guard<std::mutex> lock(m);
		if (seen.size() >= 300 || !seen.insert({chr, id, motivo}).second) return;
	}
	const uint64_t t0 = g_t0.load();
	const uint64_t s  = t0 != 0 ? (NowMs() - t0) / 1000 : 0;
	std::printf("[golpes] ALERTA: c%d id %d motivo %s a los %llu min %02llu s de juego\n", chr, id, names[motivo],
	            (unsigned long long)(s / 60), (unsigned long long)(s % 60));
	std::fflush(stdout);
}

inline void Report() {
	const uint64_t now  = NowMs();
	uint64_t       last = g_last_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		g_last_ms.compare_exchange_strong(last, now);
		return;
	}
	if (now - last < 60000 || !g_last_ms.compare_exchange_strong(last, now)) return;
	uint64_t v[MCount];
	for (int i = 0; i < MCount; i++) v[i] = g_n[i].exchange(0);
	char ids[128];
	int  p = 0;
	ids[0] = '\0';
	for (int i = 0; i < kIds; i++) {
		const int32_t c = g_zero_ids[i].exchange(0);
		if (c != 0 && p < 110) p += std::snprintf(ids + p, sizeof(ids) - p, " c%d", c);
	}
	g_zero_next.store(0);
	if (v[MZeroHp] + v[MForceDead] + v[MSpPause] + v[MWarp] == 0) return; // (golpes2) en silencio si no pasa nada
	std::printf("[golpes] 60 s: BP_CheckForExclude llamadas=%llu comprobadas=%llu | ZeroHp-no-muerto=%llu ForceDead=%llu "
	            "SpEffectPausa=%llu WarpReset=%llu | chr sin vida y no muertos:%s\n",
	            (unsigned long long)v[MCalls], (unsigned long long)v[MCheck], (unsigned long long)v[MZeroHp],
	            (unsigned long long)v[MForceDead], (unsigned long long)v[MSpPause], (unsigned long long)v[MWarp],
	            p == 0 ? " ninguno" : ids);
	std::fflush(stdout);
}

__attribute__((sysv_abi, noinline)) inline void CheckForExclude(uint64_t chr, uint64_t flag, uint64_t, uint64_t, uint64_t) {
	g_n[MCalls].fetch_add(1, std::memory_order_relaxed);
	if (chr != 0) {
		const auto* b = reinterpret_cast<const uint8_t*>(chr);
		if ((flag & 0xff) == 0) {
			g_n[MCheck].fetch_add(1, std::memory_order_relaxed);
			if (*reinterpret_cast<const int32_t*>(b + 0x710) < 1) {
				const uint64_t p1 = *reinterpret_cast<const uint64_t*>(b + 0x38);
				const uint64_t p2 = p1 != 0 ? *reinterpret_cast<const uint64_t*>(p1 + 0x40) : 0;
				if (p2 != 0 && *reinterpret_cast<const uint8_t*>(p2 + 0x170) == 0) {
					{ g_n[MZeroHp].fetch_add(1, std::memory_order_relaxed); Alert(b, MZeroHp); }
					const int i = g_zero_next.fetch_add(1);
					if (i < kIds) g_zero_ids[i].store(*reinterpret_cast<const int32_t*>(b + 0xa0));
				}
			}
			if ((b[0x581] & 2) != 0) { g_n[MForceDead].fetch_add(1, std::memory_order_relaxed); Alert(b, MForceDead); }
			const uint64_t sp = *reinterpret_cast<const uint64_t*>(b + 0x560);
			if (sp != 0 && *reinterpret_cast<const uint8_t*>(sp + 0x10) == 1) { g_n[MSpPause].fetch_add(1, std::memory_order_relaxed); Alert(b, MSpPause); }
		}
		if ((b[0x582] & 8) != 0) { g_n[MWarp].fetch_add(1, std::memory_order_relaxed); Alert(b, MWarp); }
	}
	Report();
}

inline void InstallOnce() {
	if (!On()) return;
	static std::once_flag once;
	std::call_once(once, [] {
		g_t0.store(NowMs()); RandoMsb::InstallStub(0x3286c0, reinterpret_cast<uint64_t>(&CheckForExclude), "[golpes] contador BP_CheckForExclude"); });
}

} // namespace Golpes
#else
namespace Golpes {
inline void InstallOnce() {}
} // namespace Golpes
#endif
