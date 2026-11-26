#ifndef KYTY_SHADER_RECOMPILER_RTQ2_PATCH_H_
#define KYTY_SHADER_RECOMPILER_RTQ2_PATCH_H_
// (rtq2) "Suelo estable" (KYTY_RTQ2_IBL=1, ds-settings suelo_estable=1; off: nothing changes): one instruction of the
// game's IBL compute (CS 9fb5cc274b5e4f82, RenderIBL FUN_009d66d0, 01.007.000) replaced before it is decoded.
//
// What the data shows (rtq2/INFORME.md, ISA dumps of E:\dsdump, frame captures cap1-cap4):
//  - The G-buffer pixel shaders (e.g. PS d0da8798db01cf78, 67 draws of the Nexus floor frame) write TWO normals: MRT1 the
//    normal-mapped one (normal map sampled at pc 0xc8, perturbed TBN, 12+12-bit octahedral) and MRT3 the interpolated
//    vertex normal (V_INTERP only, pc 0x2a8..0x324, 8+8-bit octahedral), flagged by bit 0 of MRT2 (0x8001 | ...).
//  - The IBL builds the reflection direction (pc 0xb54..0xbf8) as
//        F = lerp(R2 + r*N1, R1, k),  k = clamp(dot(N2, R1) / (r^2 + 0.1))
//    with R1 = reflect(V, N1 normal-mapped) and R2 = reflect(V, N2 geometric). On a glossy floor k is ~1: the cubemap
//    and SSR are looked up along the normal-mapped R1, which swims with the view (the floor "ripples").
//  - Water is not in the G-buffer (its own forward pass): this never touches water's waves.
// The patch: pc 0xb8c `V_MUL_F32 v18.clamp, v3, v18` (k, SDWA: 0x102424f9 0x06062603) -> `V_MOV_B32 v18, 0` + `S_NOP 0`:
// k = 0, F = R2 + r*N1: the reflection follows the geometric normal (roughness still widens it by the detail normal).
// Pixels without a second normal (N2 = N1 in the shader, pc 0xa24..0xa5c) are unchanged (R2 = R1). Only when both words
// match (another build of the game: not patched, logged).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Rtq2Patch {

constexpr uint64_t IblHash   = 0x9fb5cc274b5e4f82ull;
constexpr size_t   IblWord   = 0xb8c / 4;
constexpr uint32_t IblOld[2] = {0x102424f9u, 0x06062603u};
constexpr uint32_t IblNew[2] = {0x7E240280u /* V_MOV_B32 v18, 0 */, 0xBF800000u /* S_NOP 0 */};

[[nodiscard]] inline bool IblOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RTQ2_IBL");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}


// (onda3) KYTY_ONDA_JUEGO=1: the game's TAA (CS a44e580f4f0cfb2b / 934a6e1197b99148, TemporalAA FUN_009e8800).
// Ghidra/ISA: the current-frame weight is  w = s20 + s21 * clamp(s22 * motion)  (pc 0xad4 `V_MAD_F32 v4, s21, v1, s20`,
// CB dwords 7..10 = offsets 28..40: r_aaCurrFrameAmount / r_aaCurrFrameMotionAmount / ...MotionFactor / deflicker
// exponent), then scaled by the deflicker term and applied at pc 0xce0 (out = hist + w*(cur - hist)). Standing still,
// motion = 0 and w = s20 (static amount): the history left by the walk (reflections reprojected with the floor's motion
// vectors, wrong for a reflection) fades over ~1/w frames: the ~1 s "onda". Walking, w gets + s21 and it looks right.
// Patch: src2 s20 -> s21:  w = s21 + s21*clamp(..): standing still the TAA uses the walking weight. Signature-checked;
// another build of the shader is not patched (logged).
constexpr uint64_t TaaHashes[2] = {0xa44e580f4f0cfb2bull, 0x934a6e1197b99148ull};
constexpr uint32_t TaaOld[2]    = {0xD5410004u /* V_MAD_F32 v4 */, 0x00520215u /* s21, v1, s20 */};
constexpr uint32_t TaaNew1      = 0x00540215u; /* s21, v1, s21 */

[[nodiscard]] inline bool TaaOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_ONDA_JUEGO");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

[[nodiscard]] inline bool HoldsTaaSignature(std::span<const uint32_t> words) {
	for (size_t i = 0; i + 1 < words.size(); ++i)
		if (words[i] == TaaOld[0] && words[i + 1] == TaaOld[1]) return true;
	return false;
}

// Whether `words` holds the IBL's two words anywhere (the shader prefetch's disk key: a group holding it gets another key
// while the patch is on, so the unpatched translation stored on disk is not taken).
[[nodiscard]] inline bool HoldsIblSignature(std::span<const uint32_t> words) {
	for (size_t i = 0; i + 1 < words.size(); ++i)
		if (words[i] == IblOld[0] && words[i + 1] == IblOld[1]) return true;
	return false;
}

// The code to decode: `code` itself, or a patched copy in `scratch`.
[[nodiscard]] inline std::span<const uint32_t> ApplyTaa(uint64_t hash, std::span<const uint32_t> code, std::vector<uint32_t>& scratch) {
	size_t at = 0, n = 0;
	for (size_t i = 0; i + 1 < code.size(); ++i)
		if (code[i] == TaaOld[0] && code[i + 1] == TaaOld[1]) { at = i; ++n; }
	if (n != 1) {
		std::printf("[onda3] TAA CS %016llx: firma del peso (V_MAD_F32 v4, s21, v1, s20) encontrada %zu veces: NO parcheado\n",
		            static_cast<unsigned long long>(hash), n);
		std::fflush(stdout);
		return code;
	}
	scratch.assign(code.begin(), code.end());
	scratch[at + 1] = TaaNew1;
	std::printf("[onda3] TAA CS %016llx parcheado en pc 0x%zx: quieto usa el peso de movimiento (s21)\n",
	            static_cast<unsigned long long>(hash), at * 4);
	std::fflush(stdout);
	return {scratch.data(), scratch.size()};
}

[[nodiscard]] inline std::span<const uint32_t> Apply(uint64_t hash, std::span<const uint32_t> code, std::vector<uint32_t>& scratch) {
	if ((hash == TaaHashes[0] || hash == TaaHashes[1]) && TaaOn()) return ApplyTaa(hash, code, scratch);
	if (hash != IblHash || !IblOn()) return code;
	if (code.size() <= IblWord + 1 || code[IblWord] != IblOld[0] || code[IblWord + 1] != IblOld[1]) {
		std::printf("[rtq2] suelo estable: CS %016llx sin la firma esperada en pc 0xb8c: NO parcheado\n",
		            static_cast<unsigned long long>(hash));
		std::fflush(stdout);
		return code;
	}
	scratch.assign(code.begin(), code.end());
	scratch[IblWord]     = IblNew[0];
	scratch[IblWord + 1] = IblNew[1];
	std::printf("[rtq2] suelo estable: CS %016llx parcheado (reflejo con la normal geometrica, pc 0xb8c)\n",
	            static_cast<unsigned long long>(hash));
	std::fflush(stdout);
	return {scratch.data(), scratch.size()};
}

} // namespace Libs::Graphics::ShaderRecompiler::Rtq2Patch

#endif
