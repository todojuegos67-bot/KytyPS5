#ifndef EMULATOR_SRC_LOCAL_PORT_LIVE_H_
#define EMULATOR_SRC_LOCAL_PORT_LIVE_H_

// The "Demon's Souls" PC port build (-DKYTY_DS_PORT=ON, src/port): what its in-game options menu changes while the game
// runs, read by the engine where it already applies these settings. Every value's default means "as launched" (the
// environment switches): an emulator without the port never writes them, so nothing changes for it.

#include <atomic>

union SDL_Event;
struct SDL_Window;

namespace PortLive {

inline std::atomic<float> sharpness {-1.0f};  // KYTY_UPSCALE_SHARPNESS; < 0: the environment's
inline std::atomic<int>   fg_generate {0};    // generated frames per real frame (fixed DLSS-G, FSR 3 MFG); 0: as launched
inline std::atomic<float> fps_cap {0.0f};     // KYTY_FPS_CAP (with KYTY_FLIP_WHEN_READY); 0: as launched
inline std::atomic<int>   rumble {-1};        // KYTY_HAPTIC_RUMBLE; < 0: the environment's
inline std::atomic<float> rumble_gain {0.0f}; // KYTY_HAPTIC_RUMBLE_GAIN; 0: the environment's
inline std::atomic<bool>  menu_open {false};  // the options menu is shown: no frame generation, the game gets no input

// The port's hooks (null in the emulator): set once by the port's main before the game starts.
struct Hooks {
	void (*window_created)(SDL_Window* window); // after the game window and its icon
	bool (*event)(const SDL_Event& event);      // window thread: true when the event is the menu's (the game skips it)
	void (*presented)();                        // present thread, after each present
	bool (*wants_overlay)();                    // the overlay must be drawn (the menu is open)
	unsigned long long (*revision)();           // changes whenever the menu's picture does
	void (*draw)(float width, float height);    // present thread, between ImGui::NewFrame and Render (input fed first)
	const char* (*title)();                     // the window title (null: the emulator's)
};
inline const Hooks* g_hooks = nullptr;

} // namespace PortLive

#endif // EMULATOR_SRC_LOCAL_PORT_LIVE_H_
