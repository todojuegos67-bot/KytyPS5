#ifndef EMULATOR_SRC_LOCAL_TITLE_STATE_H_
#define EMULATOR_SRC_LOCAL_TITLE_STATE_H_
// bruno-astro-1025: what frame generation (frame-gen.cpp, fsr3-fg.cpp) asks of the video-out side in the
// todojuegos67-bot fork, reduced to what this tree can tell:
// - MoviePlaying: a movie frame was decoded (AvPlayer, VideoDec2) within the last half second; frame generation is
//   off for movies (their frames come at their own rate, and the generic inputs would interpolate cuts).
// - PlayerInputSeen: the player pressed a button (controller.cpp) at least KYTY_WORLD_GATE_S seconds ago (default 0):
//   until then (logos, intro, "press a button") frame generation stays off.
// - CinematicPlaying, CutsceneHeld: Demon's Souls' in-engine cinematic detection; never on this tree.
// - SetDynamicFlipFps, ReachFps: the fork's flip floor for the dynamic multipliers; this tree's video-out has no flip
//   pacer to hold the game's flips, so both answer 0 (refused) and the dynamic modes pace by Reflex / FSR 3 only.
namespace Libs::VideoOut {
void NoteMovieFrame() noexcept;
[[nodiscard]] bool MoviePlaying() noexcept;
void NotePlayerInput() noexcept;
[[nodiscard]] bool PlayerInputSeen() noexcept;
[[nodiscard]] bool PlayerButtonSeen() noexcept;
[[nodiscard]] bool CinematicPlaying() noexcept;
[[nodiscard]] bool CutsceneHeld() noexcept;
int                SetDynamicFlipFps(int fps) noexcept;
[[nodiscard]] int  ReachFps(double target, int multiplier) noexcept;
} // namespace Libs::VideoOut
#endif // EMULATOR_SRC_LOCAL_TITLE_STATE_H_
