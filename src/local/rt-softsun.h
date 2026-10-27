#ifndef EMULATOR_SRC_LOCAL_RT_SOFTSUN_H_
#define EMULATOR_SRC_LOCAL_RT_SOFTSUN_H_
// KYTY_HWRT_SOFTSUN=0..3 (No / Bajo / Medio / Alto; default 0: nothing changes): physically soft RT sun shadows on the
// KYTY_RT_SUN / KYTY_HWRT_SHADOW pass (rt-softsun.cpp).
//  - The sun's real apparent size (0.53 deg; x1 / x1.5 / x2 by level) sets both the cascades' PCSS cone and the TLAS
//    rays' jitter disc: the penumbra widens with the blocker distance (sharp at contact, soft far from the caster)
//    instead of KYTY_RT_SUN_ANGLE's fixed 0.8 deg and the hard edge it gave close to casters.
//  - Robust outdoors: the game re-renders its far sun cascade (the one the draws reveal, 1.5 km wide) only when it moves
//    and keeps it cached in between, so the copy the pass needs was taken a few times a minute and the pass ran only the
//    8 flips after each (9..40 frames in 10 s in the 10-27 logs). With this on, the last copy stays in use while the
//    game keeps binding that shadow array (or the arrays bound with it at the copy: its cache) in any dispatch of the
//    last 8 flips (ds-rt-shadow: or any sun-shadow consumer dispatch), up to KYTY_HWRT_SOFTSUN_HOLD flips (default: no cap). Gated indoors by that use: a flip with no such
//    binding ends it. While the copy is held, the composite may only darken (no brightening from a stale match).
//  - Turns KYTY_HWRT_SHADOW on at its own level when that is unset.
#include <cstdint>
#include <span>

namespace Libs::Graphics::FrameGen::SoftSun {

int  Level();           // 0..3
float AngleDeg();       // the sun's apparent diameter used (deg)
// RtSunAfterDispatch: a dispatch bound the cascades' array or one of its companions this flip.
void NoteUse(uint64_t flip);
// A copy was taken from `source`; `others` the other layered arrays the same dispatch bound (the game's cache of it).
void NoteCopy(uint64_t flip, uint64_t source, std::span<const uint64_t> others);
// The held copy (and its use tracking and 10 s line): on with KYTY_HWRT_SOFTSUN, KYTY_HWRT_SHADOW or KYTY_RT_SUN > 0
// (KYTY_HWRT_SOFTSUN_HOLD=0: off, the 8-flip rule as before).
bool HoldOn();
bool IsCompanion(uint64_t address);
// RtSunPrepare: whether a copy taken at copy_flip may still be used now (held: older than 8 flips).
// (ds-rt-shadow) `why`: the refusal's cause (a string literal) when false. The held copy lasts while the game uses it
// (its array, a companion, or any sun-shadow consumer dispatch: NoteUse), with no age cap unless KYTY_HWRT_SOFTSUN_HOLD is set.
bool CopyUsable(uint64_t now, uint64_t copy_flip, bool& held, const char** why = nullptr);
// Once a frame the pass ran (held or fresh) / was refused for want of a copy; logs a line every 10 s.
void NoteFrame(bool ran, bool held);

} // namespace Libs::Graphics::FrameGen::SoftSun

#endif // EMULATOR_SRC_LOCAL_RT_SOFTSUN_H_
