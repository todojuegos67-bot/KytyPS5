// (rtx1) The game's own render settings read at run time and, on request, moved RELATIVE to the game's own values.
// Header-only: included once, by frame-gen.cpp (Upscale calls Rtx1::Tick every frame; the work is every 0.5 s).
//
// [G] eboot 01.007.000 (Ghidra, rtq1/ghidra/convars-render.tsv + rtx1/ghidra): every r_* / fog_* / AO convar is a slot
// of one render-config object whose address is held in the eboot's 8-byte variable at 0x26d22f0 (run time: +0x900000000,
// "*(_DAT_026d22f0) + off" in the registrations FUN_00944dc0 / 0095c140 / 00920080 / 0094c570). The passes read the
// slots every frame (Fog FUN_00994310 reads +0x8f4; RenderSSR FUN_009d5120 and SSRPreRender FUN_009d6030 read +0x788;
// FUN_00933d10 reads +0xa24), so a changed slot shows the next frame, with no restart and no shader change.
// Why relative: the defaults are not constants of the eboot (the slots are filled by the level's settings at load), so
// an absolute value would be a guess. Each scaled slot keeps the game's value as its base: when the game writes a new
// one (another area), that becomes the base; ours is base x factor, clamped to the range the game registered.
//
// KYTY_RTX1_FOG=0..4 (Luz volumetrica 4 = Ultra -> 4): fog_density x1.3, fog_scatterDirectionality (Henyey-Greenstein g,
// registered -1..1) 35 % of the way toward 0.85 (stronger shafts toward the light). (rtx1c) fog_dirScatterDirectionality is
// left alone: the game's own value is -1.0 ([L] 20261119-042737), the end of its range, more likely a "not used" mark than
// a phase to scale. Levels 1-3: nothing (the game's own fog, as now).
// KYTY_RTX1_AO=0..4 (Oclusion mejorada): Gtao_Intensity x1.15 / 1.3 / 1.45 / 1.6, Hbao_Radius x1.15 / 1.3 / 1.5 / 1.75,
// Hbao_MaxAOLinearDepth x1 / 1.5 / 2 / 3.
// KYTY_RTX1_SSR_OFF=1 (Reflejos estables 2-3): r_ssrMethod forced to 0 (ESSRMethod::Off) if the convar line did not.
// KYTY_DS_OPTCULL=1 (off by default): the engine's own r_EnableOptimizedFrustumCulling (bool, slot +0xce [G]: registered
// FUN_00947xxx, lea name 0x2482d72 at VA 0x9477ed, `mov $0xce,%edx; add [0x26d22f0]`) set to true. Its reader, the object
// culling at VA 0x9e2b18 (`cmpb $0x1,0xce(cfg)`, cfg = [obj+0x68]), takes the vectorized frustum test (planes negated
// with the sign mask 0x238e73c) only when it is 1, else the generic path at VA 0x9e319d. The game's own value is logged
// first (if it is already 1 nothing changes and the variable is useless); written back whenever the game resets it.
// KYTY_RTX1_LOG=1: the values every 30 s (always once when any of the above is on). Off (all unset): nothing runs.
// Safety: the object is accepted only if its enums / booleans / floats are in their registered ranges (else logged once
// and never touched); reads by NativeGuardedRead. (rtx1c) Writes: with native guest memory (KYTY_NATIVE_MEMORY / RAIZ:
// no backing view, so TryWriteCpuBacking always said no, [L] 20261119-042737 "no se pudo escribir ... (+0x8f4)") through
// NativeOffloadWrite, the emulator's guarded write window over guest pages whatever their protection; else
// TryWriteCpuBacking. Either false: not written, logged once.
#pragma once

#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::FrameGen::Rtx1 {

namespace Detail {

constexpr uint64_t ConfigPointer = 0x900000000ull + 0x26d22f0ull; // eboot 01.007.000 [G]
constexpr uint32_t BlockBytes    = 0xc00;

// Slots [G] (rtq1/ghidra/convars-render.tsv: name, offset, type, registered maximum).
constexpr uint32_t OffContactEnabled = 0x91, OffContactDistance = 0x98;
constexpr uint32_t OffCascadeShadows = 0xcd, OffSunMaxDistance = 0xd8, OffSpotShadows = 0xf0;
constexpr uint32_t OffIblResolution = 0x76c, OffIblTemporal = 0x770, OffSsrMethod = 0x788, OffSsrSinglePixel = 0x78c;
constexpr uint32_t OffSsrCone = 0x78d, OffSsrPlaneReject = 0x78e;
constexpr uint32_t OffFogEnable = 0x8e0, OffFogDitherXY = 0x8e2, OffFogDitherZ = 0x8e3, OffFogDensity = 0x8f4;
constexpr uint32_t OffFogScatterDir = 0x8f8, OffFogDirScatterDir = 0x8fc;
constexpr uint32_t OffAoMode = 0xa24, OffSsaoTemporalBlur = 0xa28, OffGtaoIntensity = 0xa2c, OffHbaoRadius = 0xa38;
constexpr uint32_t OffHbaoMaxDepth = 0xa48, OffVgShadowCascades = 0xa00;
constexpr uint32_t OffOptFrustumCull = 0xce; // r_EnableOptimizedFrustumCulling [G]

inline int EnvInt(const char* name, int lo, int hi) {
	const char* v = std::getenv(name);
	return v != nullptr && *v != '\0' ? std::clamp(std::atoi(v), lo, hi) : 0;
}

struct Scaled {
	uint32_t off   = 0;
	float    lo    = 0.0f, hi = 0.0f; // the registered range
	int      kind  = 0;               // 0: base x k, 1: toward `k` by `t` (directionality)
	float    k     = 1.0f, t = 0.0f;
	bool     have  = false;
	float    base  = 0.0f, written = 0.0f;
};

struct State {
	bool     init = false, enabled = false, refused = false, logged_once = false, write_failed_said = false;
	int      fog = 0, ao = 0;
	bool     ssr_off = false, log = false, optcull = false, optcull_said = false;
	uint64_t config = 0;
	std::chrono::steady_clock::time_point next {}, next_log {};
	std::array<Scaled, 8> slots {};
	int      n_slots = 0;
	uint64_t writes = 0, write_fails = 0, rebases = 0;
};

inline State& G() {
	static State s;
	return s;
}

inline float F(const uint8_t* b, uint32_t off) {
	float v = 0.0f;
	std::memcpy(&v, b + off, sizeof(v));
	return v;
}
inline int32_t I(const uint8_t* b, uint32_t off) {
	int32_t v = 0;
	std::memcpy(&v, b + off, sizeof(v));
	return v;
}

// The object is the render config only if its enums, booleans and floats look like the registered convars.
inline bool Plausible(const uint8_t* b) {
	const auto boolean = [&](uint32_t off) { return b[off] <= 1; };
	const auto finite  = [&](uint32_t off, float lo, float hi) {
        const float v = F(b, off);
        return std::isfinite(v) && v >= lo && v <= hi;
	};
	return I(b, OffSsrMethod) >= 0 && I(b, OffSsrMethod) <= 2 && I(b, OffIblResolution) >= 0 && I(b, OffIblResolution) <= 3 &&
	       I(b, OffAoMode) >= 0 && I(b, OffAoMode) <= 3 && boolean(OffFogEnable) && boolean(OffContactEnabled) &&
	       boolean(OffCascadeShadows) && boolean(OffOptFrustumCull) && boolean(OffSpotShadows) && boolean(OffIblTemporal) && boolean(OffSsrSinglePixel) &&
	       finite(OffFogDensity, 0.0f, 1000.0f) && finite(OffFogScatterDir, -1.0f, 1.0f) && finite(OffFogDirScatterDir, -1.0f, 1.0f) &&
	       finite(OffGtaoIntensity, 0.0f, 10.0f) && finite(OffHbaoRadius, 0.0f, 10.0f) && finite(OffHbaoMaxDepth, 0.0f, 5000.0f) &&
	       finite(OffSunMaxDistance, 0.0f, 1000.0f) && finite(OffContactDistance, 0.0f, 1.0f);
}

inline void AddSlot(State& s, uint32_t off, float lo, float hi, int kind, float k, float t) {
	if (s.n_slots >= static_cast<int>(s.slots.size())) return;
	Scaled& x = s.slots[static_cast<size_t>(s.n_slots++)];
	x.off = off, x.lo = lo, x.hi = hi, x.kind = kind, x.k = k, x.t = t;
}

inline void Init(State& s) {
	s.init    = true;
	s.fog     = EnvInt("KYTY_RTX1_FOG", 0, 4);
	s.ao      = EnvInt("KYTY_RTX1_AO", 0, 4);
	s.ssr_off = EnvInt("KYTY_RTX1_SSR_OFF", 0, 1) == 1;
	s.log     = EnvInt("KYTY_RTX1_LOG", 0, 1) == 1;
	s.optcull = EnvInt("KYTY_DS_OPTCULL", 0, 1) == 1;
	s.enabled = s.fog > 0 || s.ao > 0 || s.ssr_off || s.log || s.optcull;
	if (s.fog >= 4) {
		AddSlot(s, OffFogDensity, 0.0f, 1000.0f, 0, 1.3f, 0.0f);
		AddSlot(s, OffFogScatterDir, -1.0f, 1.0f, 1, 0.85f, 0.35f);
	}
	if (s.ao > 0) {
		static constexpr float intensity[] = {1.0f, 1.15f, 1.3f, 1.45f, 1.6f};
		static constexpr float radius[]    = {1.0f, 1.15f, 1.3f, 1.5f, 1.75f};
		static constexpr float depth[]     = {1.0f, 1.0f, 1.5f, 2.0f, 3.0f};
		AddSlot(s, OffGtaoIntensity, 0.0f, 10.0f, 0, intensity[s.ao], 0.0f);
		AddSlot(s, OffHbaoRadius, 0.0f, 10.0f, 0, radius[s.ao], 0.0f);
		if (depth[s.ao] != 1.0f) AddSlot(s, OffHbaoMaxDepth, 0.0f, 5000.0f, 0, depth[s.ao], 0.0f);
	}
	if (s.enabled) {
		std::printf("[rtx1] convars del juego en vivo: niebla nivel %d, oclusion nivel %d, SSR del juego %s, log %s "
		            "(objeto de ajustes en q[eboot+0x26d22f0]; cambios relativos al valor del propio juego)\n",
		            s.fog, s.ao, s.ssr_off ? "forzado a Off" : "como venga", s.log ? "cada 30 s" : "una vez");
		std::fflush(stdout);
	}
}

inline float Target(const Scaled& x) {
	const float v = x.kind == 0 ? x.base * x.k : (x.base < x.k ? x.base + (x.k - x.base) * x.t : x.base);
	return std::clamp(v, x.lo, x.hi);
}

inline bool Write(State& s, uint32_t off, const void* data, uint64_t size) {
	const bool native = Libs::LibKernel::Memory::NativeMemoryEnabled();
	if (native ? Libs::LibKernel::Memory::NativeOffloadWrite(s.config + off, data, size)
	           : Libs::LibKernel::Memory::TryWriteCpuBacking(s.config + off, data, size)) {
		++s.writes;
		return true;
	}
	++s.write_fails;
	if (!s.write_failed_said) {
		s.write_failed_said = true;
		std::printf("[rtx1] no se pudo escribir en el objeto de ajustes del juego (+0x%x, via %s): se deja como esta\n", off,
		            native ? "NativeOffloadWrite" : "TryWriteCpuBacking");
		std::fflush(stdout);
	}
	return false;
}

inline void LogValues(State& s, const uint8_t* b, const char* why) {
	std::printf("[rtx1] ajustes del juego (%s): fog_enable=%u density=%.4g scatterDir=%.3f dirScatterDir=%.3f ditherXY=%u ditherZ=%u | "
	            "AO_Mode=%d Ssao_DoTemporalBlur=%u Gtao_Intensity=%.3f Hbao_Radius=%.3f Hbao_MaxAOLinearDepth=%.1f | "
	            "r_ssrMethod=%d singlePixelStep=%u roughnessCone=%u planeReject=%u r_iblResolution=%d r_enableIBLTemporal=%u | contactShadows=%u dist=%.3f "
	            "cascadeShadows=%u optFrustumCulling=%u sunShadowMaxDistance=%.0f vgCascades=%d spotShadows=%u | escrituras %llu, fallidas %llu, "
	            "cambios del propio juego %llu\n",
	            why, b[OffFogEnable], F(b, OffFogDensity), F(b, OffFogScatterDir), F(b, OffFogDirScatterDir), b[OffFogDitherXY],
	            b[OffFogDitherZ], I(b, OffAoMode), b[OffSsaoTemporalBlur], F(b, OffGtaoIntensity), F(b, OffHbaoRadius),
	            F(b, OffHbaoMaxDepth), I(b, OffSsrMethod), b[OffSsrSinglePixel], b[OffSsrCone], b[OffSsrPlaneReject], I(b, OffIblResolution), b[OffIblTemporal],
	            b[OffContactEnabled], F(b, OffContactDistance), b[OffCascadeShadows], b[OffOptFrustumCull], F(b, OffSunMaxDistance),
	            I(b, OffVgShadowCascades), b[OffSpotShadows], static_cast<unsigned long long>(s.writes),
	            static_cast<unsigned long long>(s.write_fails), static_cast<unsigned long long>(s.rebases));
	std::fflush(stdout);
}

} // namespace Detail

// Every frame from FrameGen::Upscale; one guarded 3 KiB read every 0.5 s when on, nothing when off.
inline void Tick() {
	auto& s = Detail::G();
	if (!s.init) Detail::Init(s);
	if (!s.enabled || s.refused) return;
	const auto now = std::chrono::steady_clock::now();
	if (now < s.next) return;
	s.next = now + std::chrono::milliseconds(500);
	uint64_t config = 0;
	if (!Libs::LibKernel::Memory::NativeGuardedRead(&config, reinterpret_cast<const void*>(Detail::ConfigPointer), sizeof(config)) ||
	    config < 0x100000000ull || config > 0xff00000000ull)
		return; // (not made yet: the game is starting)
	static std::array<uint8_t, Detail::BlockBytes> block {};
	if (!Libs::LibKernel::Memory::NativeGuardedRead(block.data(), reinterpret_cast<const void*>(config), block.size())) return;
	if (!Detail::Plausible(block.data())) {
		static int tries = 0;
		if (++tries >= 20) { // (10 s of implausible values: not this build's object; never touched)
			s.refused = true;
			std::printf("[rtx1] el objeto en q[eboot+0x26d22f0] = 0x%llx no parece el de ajustes (valores fuera de rango): no se toca\n",
			            static_cast<unsigned long long>(config));
			std::fflush(stdout);
		}
		return;
	}
	s.config = config;
	if (!s.logged_once) { // (the game's own values first, before anything of ours)
		s.logged_once = true;
		s.next_log    = now + std::chrono::seconds(30);
		Detail::LogValues(s, block.data(), "primera lectura, valores del propio juego");
	}
	for (int i = 0; i < s.n_slots; ++i) {
		auto&       x   = s.slots[static_cast<size_t>(i)];
		const float cur = Detail::F(block.data(), x.off);
		if (!x.have || cur != x.written) {
			if (x.have) ++s.rebases; // (the game set its own value: that is the new base)
			x.base = cur, x.have = true;
		}
		const float want = Detail::Target(x);
		if (want != cur) {
			if (Detail::Write(s, x.off, &want, sizeof(want))) x.written = want;
			else x.written = cur;
		} else {
			x.written = cur;
		}
	}
	if (s.ssr_off && Detail::I(block.data(), Detail::OffSsrMethod) != 0) {
		const int32_t off = 0;
		Detail::Write(s, Detail::OffSsrMethod, &off, sizeof(off));
	}
	if (s.optcull && block[Detail::OffOptFrustumCull] != 1) {
		const uint8_t one = 1;
		const bool    ok  = Detail::Write(s, Detail::OffOptFrustumCull, &one, sizeof(one));
		if (!s.optcull_said) {
			s.optcull_said = true;
			std::printf("[optcull] r_EnableOptimizedFrustumCulling del juego = %u -> 1 (%s)\n", block[Detail::OffOptFrustumCull],
			            ok ? "puesto" : "no se pudo escribir");
			std::fflush(stdout);
		}
	} else if (s.optcull && !s.optcull_said) {
		s.optcull_said = true;
		std::printf("[optcull] r_EnableOptimizedFrustumCulling ya es 1 en el juego: KYTY_DS_OPTCULL no cambia nada\n");
		std::fflush(stdout);
	}
	if (s.log && now >= s.next_log) {
		s.next_log = now + std::chrono::seconds(30);
		Detail::LogValues(s, block.data(), "30 s");
	}
}

} // namespace Libs::Graphics::FrameGen::Rtx1
