// "Demon's Souls" PC port: the launcher before the game (RunStartMenu: Launcher-souls.ps1's window, its groups, rows,
// options and tooltips, every setting of settings.ini) and, in the game, F2 = save and quit (ds-resume.h).
//   (ds-pc-port8) F2, or R3 held + OPTIONS (OPTIONS then never reaches the game): the game's save is let finish, the saves
//   are copied, the place is kept for JUGAR and the game closes. There is no options menu in the game any more: every
//   option is in the launcher. The other keys (F7 mouse, F8 metrics, F11 full screen) stay the emulator's.
// Drawn by the system overlay (src/graphics/presentation/systemOverlay.cpp) into the swapchain image, after the upscaler
// and outside frame generation.

#include "port/ds-port.h"
#include <functional>
#include "port/ds-update.h"
#include "port/ds-precompile.h"
#include "port/ds-i18n.h"
#include "port/ds-steam.h"
#include "port/ds-resume.h"
#include "port/ds-style.h" // (trucos11) the Demon's Souls remake look
#include "port/ds-online.h" // (online5) Multijugador

#include "local/perf-overlay.h"
#include "local/local-platform.h"
#include "local/ram-guard.h"
#include "local/port-live.h"
#include "kernel/memory.h" // (trucos) the cheats' guest reads / writes
#include "loader/demonsSoulsInventory.h" // (trucos3) items and spells added on the game thread
#include "graphics/presentation/window/hostInput.h"
#include "libs/trophy.h"

#if defined(_WIN32)
// (11-10) Before SDL.h (its SDL_system.h only declares ID3D11Device): the launcher fog (ds-fog.inc) uses the full types.
#include <d3d11.h>
#include <psapi.h>
#include <d3dcompiler.h>
#include <mmsystem.h> // (chat2) PlaySoundW: the notice chime
#pragma comment(lib, "winmm.lib")
#undef interface
#undef small
#undef hyper
#endif
#include "SDL.h"
#include "SDL_syswm.h"
#include "imgui.h"
#include "libs/trophy-cup.h"
#include "imgui_internal.h" // ImGui::RegisterUserTexture (the background image)
#if defined(_WIN32)
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
#endif
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdarg>
#include <regex>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <iterator>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <shellapi.h>
#endif

namespace Port {

namespace {

struct Input {
	enum Kind { Key, Analog, MousePos, MouseButton, Wheel, Tab, Text } kind; // (online6: Text, the typed characters)
	int   key  = 0;
	bool  down = false;
	float x = 0, y = 0;
	std::string text = {};
};
// (online6) A text field of the F2 pages has the keyboard (ImGui's WantTextInput, present thread): Esc, Q / E and the
// pad's Circle go to the field (Esc leaves it) instead of the page's navigation.
std::atomic<bool> g_text_active {false};

Runtime                  g_rt;
Settings                 g_edit;       // the menu's copy (present thread)
Settings                 g_saved;      // what settings.ini holds now

// (onlineclean) Cheats are usable ONLY offline and with the manual "SIN TRUCOS" medallion off. Online (the game was
// launched with multijugador=1) forces every cheat off, hidden and locked, exactly like no_cheats, without the player
// touching the medallion: it avoids bans and desyncing other players. Graphics / performance are not cheats, untouched.
// (The server-reachable state is not a reliable "in a session" signal, so the launch mode decides.)
// (trucos-online) Cheats and online are mutually exclusive. "Cheats in use" = a cheat switch on now, or any cheat action
// applied in this process (Port::CheatsUsed: souls, Nivel personaje, Armas y objetos). While cheats
// are in use the online switch cannot be turned on (Multiplayer::Card) and a launch is offline (BuildLaunch, ds-main).
namespace Cheats { bool AnySwitchOn(); }
inline bool CheatsInUse() { return Port::CheatsUsed() || Cheats::AnySwitchOn(); }
// The live online switch blocks cheats only when online is really in use (saved on and not refused for cheats); the
// launch mode (g_rt.launched.multiplayer, false when the launch was refused) blocks them for the whole session.
inline bool CheatsBlocked() { return g_saved.no_cheats || g_rt.launched.randomizer || g_rt.launched.multiplayer || (g_saved.multiplayer && !CheatsInUse()); } // (rando1: the randomizer too)

// (ds-pc-port13) Preparar shaders: all CPU threads or relaxed (asked the first time, then settings.ini's
// precompilar_hilos; Opciones > Sistema changes it), then the precompile.
void StartShaders() {
	// (precomp-fix) No game folder yet: ask for it first (the precompile needs it; it is saved to settings.ini).
	if (g_saved.game.empty() && g_rt.pick_folder) {
		const std::string picked = g_rt.pick_folder();
		if (!picked.empty() && (!g_rt.is_game || g_rt.is_game(picked))) {
			g_edit.game = g_saved.game = picked;
			Settings on_disk;
			if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_edit;
			on_disk.game = picked;
			(void)SaveSettings(g_rt.ini_path, on_disk);
		}
	}
	Precompile::Choose(g_rt.ini_path, &g_edit);
	g_saved.precompile_threads = g_edit.precompile_threads;
	Precompile::Start();
}
// (ds-pc-port19) The precompile panel's mode button: all CPU threads <-> relaxed, saved to settings.ini (only that key);
// a running precompile restarts in the new mode (finished parts are kept).
void ToggleShaderThreads(bool running) {
	const bool all             = g_edit.precompile_threads != "todos";
	g_edit.precompile_threads  = all ? "todos" : "relajado";
	g_saved.precompile_threads = g_edit.precompile_threads;
	Precompile::SetAllThreads(all);
	Settings on_disk;
	if (LoadSettings(g_rt.ini_path, &on_disk)) {
		on_disk.precompile_threads = g_edit.precompile_threads;
		(void)SaveSettings(g_rt.ini_path, on_disk);
	}
	if (running) std::thread([] { Precompile::Stop(); Precompile::Start(); }).detach();
}
namespace Shop { // (trucos19) the shop's step navigation (defined below), for the window thread's Circle / Esc
extern std::atomic<int>  g_level;
extern std::atomic<bool> g_refocus;
bool                     Back();
} // namespace Shop
std::atomic<bool>        g_open {false};
std::atomic<bool>        g_trophy_page {false};      // (port34) F2 > Ver trofeos is shown (window and present threads)
// (ds-pc-port37) F2 > Mapear mando / Teclado y raton: the page shown (0 none, 1 pad mapper, 2 keyboard and mouse), and a
// key or mouse button being captured for a binding (the window thread writes its SDL name; Esc cancels).
std::atomic<int>         g_f1_page {0};
std::atomic<int>         g_page_scroll {0}; // (ds-pc-port37) L1 (up) / L2 (down) presses for the F2 page's list
bool                     g_l2_down = false;
std::atomic<bool>        g_kbm_capture {false};
std::atomic<bool>        g_pages_reset {false}; // (ds-pc-port42) SetOpen: the pages' own state (capture row) cleared at their next frame
std::mutex               g_capture_mutex;
std::string              g_captured; // "" while waiting; "\x1b" cancelled
// (remap-f2) "Mapear mando": a pad button being captured (5 s). The window thread (or the launcher's loop) writes the
// pressed button's SDL name (a, b, leftshoulder, lefttrigger, dpup, ...) to g_pad_captured under g_capture_mutex and
// clears g_pad_capture; "\x1b" = cancelled (Esc / right click), "" = no button (time out). While it waits, no pad,
// key or click reaches the game or moves the menu.
constexpr double         kPadCaptureSeconds = 5.0;
std::atomic<bool>        g_pad_capture {false};
std::atomic<double>      g_pad_capture_at {0.0};
std::atomic<bool>        g_pad_trig_armed[2] {}; // a trigger counts once it was seen released (one held at the start does not)
std::atomic<int>         g_pad_last {-1};        // the instance id of the pad that last pressed a button (its button names)
std::string              g_pad_captured;
std::atomic<bool>        g_focus_trophy_btn_request {false}; // (port34) window thread: back from it, focus its button
int                      g_trophy_tab_delta = 0;     // (port34) present thread: L1/R1, Q/E on the trophy page
bool                     g_focus_trophy_btn = false;
std::atomic<uint64_t>    g_revision {1};
std::atomic<bool>        g_reset_edit {false};
std::mutex               g_mutex;
std::deque<Input>        g_inputs;
int                      g_passthrough = 0; // window thread: zeroed axes sent to the game as the menu opened
int                      g_cursor_before = -1;
constexpr int            kTabCount = 8; // (online11) + Conexión online
// // (10-27c) Gráficos, Pantalla, Sonido, Controles, Sistema, Trofeos; (ds-pc-port37) Ajustes del juego
int                      g_tab = 0, g_tab_request = -1;
int                      g_fg_live = 0;
std::string              g_status;
ImTextureData*           g_background = nullptr;
bool                     g_background_tried = false;
int                      g_bg_w = 0, g_bg_h = 0;
bool                     g_first_present = true;
bool                     g_focus_once = true;
bool                     g_start_swallowed = false; // window thread: OPTIONS went to the menu's combo
std::atomic<bool>        g_popup {false}; // a combo is open (Circle closes it, not the menu)
bool                     g_start_mode = false; // (10-27b) the start menu (before the game), not F2
int                      g_start_result = 0;   // 1 Jugar, -1 Salir
bool                     g_config_mode = false; // (10-28) the Configurador before the game (its own window)
bool                     g_cfg_open = false;    // (10-28) the Configurador inside F2 (Sistema > Configurador)
std::vector<std::string> g_cfg_why;             // its "Hemos elegido: ..." lines (empty: not optimized yet)

void Bump() {
	g_revision.fetch_add(1, std::memory_order_relaxed);
}

namespace MetricsPanel { // (ds-pc-port41, below)
void Toggle();
bool Shown();
} // namespace MetricsPanel

// ---- (ds-pc-port38) The F2 menu's animations: open / close (fade and scale), page slide, buttons' hover and focus. ----
// Times on the steady clock (window and present threads); the overlay is redrawn (Bump) while one runs.
namespace Anim {
double Now() {
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
constexpr double         kOpen = 0.24, kClose = 0.16, kPage = 0.22; // (step5: a little longer, eased; real time)
std::atomic<double>      g_opened_at {0.0}, g_closed_at {0.0}, g_page_at {0.0};
int                      g_page_seen = 0;
float EaseOut(float t) {
	t = std::clamp(t, 0.0f, 1.0f);
	return 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
}
// (step5) Opening: ease-out quint; closing: ease-in cubic (slow start, then away).
float EaseOutQuint(float t) {
	t = 1.0f - std::clamp(t, 0.0f, 1.0f);
	return 1.0f - t * t * t * t * t;
}
float EaseIn(float t) {
	t = std::clamp(t, 0.0f, 1.0f);
	return t * t * t;
}
// 0..1 of an animation started at `at` lasting `length` (1: done).
float Progress(double at, double length) {
	return static_cast<float>(std::clamp((Now() - at) / length, 0.0, 1.0));
}
bool Closing() {
	return !g_open.load(std::memory_order_relaxed) && Now() - g_closed_at.load(std::memory_order_relaxed) < kClose;
}
// Per item, a value easing to 1 while hovered or focused and back to 0 (about 90 ms each way).
std::map<ImGuiID, float> g_hot;
float Hot(ImGuiID id, bool on) {
	float&      v  = g_hot[id];
	const float dt = std::clamp(ImGui::GetIO().DeltaTime, 0.0f, 0.1f);
	const float to = on ? 1.0f : 0.0f;
	v += (to - v) * std::min(1.0f, dt * 14.0f);
	if (std::fabs(to - v) < 0.002f) v = to;
	else Bump();
	return v;
}
// (11-12) Smooth scrolling of the current window (call right after its Begin / BeginChild): the mouse wheel moves a
// target and the scroll eases to it (~120 ms), instead of ImGui's instant jumps; keyboard / pad focus moves (SetScrollY
// elsewhere, ImGui's own scroll-to-focus) are followed as they are.
std::map<ImGuiID, float> g_scroll_target;
void SmoothScroll() {
	ImGuiWindow* w = ImGui::GetCurrentWindow();
	if (w == nullptr) return;
	const ImGuiID id     = w->ID;
	auto&         io     = ImGui::GetIO();
	const float   max_y  = ImGui::GetScrollMaxY();
	const float   now_y  = ImGui::GetScrollY();
	auto          it     = g_scroll_target.find(id);
	float         target = it != g_scroll_target.end() ? it->second : now_y;
	static std::map<ImGuiID, float> last_set;
	// Something else scrolled the window (focus, a page jump): follow it.
	if (auto ls = last_set.find(id); ls != last_set.end() && std::fabs(ls->second - now_y) > 0.5f) target = now_y;
	w->Flags |= ImGuiWindowFlags_NoScrollWithMouse;
	if (io.MouseWheel != 0.0f && max_y > 0.0f && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
		target -= io.MouseWheel * std::max(60.0f, ImGui::GetTextLineHeightWithSpacing() * 4.0f);
		io.MouseWheel = 0.0f;
	}
	target                  = std::clamp(target, 0.0f, max_y);
	g_scroll_target[id]     = target;
	const float dt          = std::clamp(io.DeltaTime, 0.0f, 0.1f);
	float       y           = now_y + (target - now_y) * std::min(1.0f, dt * 16.0f);
	if (std::fabs(target - y) < 0.5f) y = target;
	else Bump();
	if (y != now_y) ImGui::SetScrollY(y);
	last_set[id] = y;
}
// The last item's highlight: a gold wash, a border and an accent bar on the left that grow with the hover / focus.
void Highlight(float unit) {
	const ImGuiID id = ImGui::GetItemID();
	const bool    on = ImGui::IsItemHovered() || ImGui::IsItemFocused() || ImGui::IsItemActive();
	const float   v  = Hot(id, on);
	if (v <= 0.0f) return;
	const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
	auto*        dl = ImGui::GetWindowDrawList();
	const float  r  = ImGui::GetStyle().FrameRounding;
	const float  al = ImGui::GetStyle().Alpha;
	// (des-upstream-1106) A soft glow around it: three wider, fainter outlines.
	for (int k = 1; k <= 3; k++) {
		const float g = static_cast<float>(k) * 2.0f * unit;
		dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g}, IM_COL32(240, 200, 120, static_cast<int>(34.0f / k * v * al)), r + g, 0,
		            2.0f * unit);
	}
	dl->AddRectFilled(a, b, IM_COL32(222, 196, 140, static_cast<int>(46 * v * al)), r);
	dl->AddRect(a, b, IM_COL32(236, 206, 146, static_cast<int>(200 * v * al)), r, 0, std::max(1.0f, 1.5f * unit));
	dl->AddRectFilled({a.x, a.y + (b.y - a.y) * (0.5f - 0.36f * v)}, {a.x + 4.0f * unit, a.y + (b.y - a.y) * (0.5f + 0.36f * v)},
	                  IM_COL32(244, 214, 150, static_cast<int>(255 * v * al)), 2.0f * unit);
}
bool Button(const char* label, ImVec2 size, float unit) {
	const bool pressed = ImGui::Button(label, size);
	Highlight(unit);
	return pressed;
}
// Scales the vertices a window drew (from `first`) around `center`: the open / close zoom without re-rasterizing fonts
// at every size of the animation (ImGui 1.92 bakes glyphs per size).
void ScaleVertices(ImDrawList* dl, int first, ImVec2 center, float k, float dy = 0.0f) {
	if (dl == nullptr || (k == 1.0f && dy == 0.0f)) return;
	// (step5) Sub-pixel: no rounding, so the zoom and the rise are continuous (no stepping).
	for (int i = first; i < dl->VtxBuffer.Size; i++) {
		ImVec2& p = dl->VtxBuffer[i].pos;
		p.x       = center.x + (p.x - center.x) * k;
		p.y       = center.y + (p.y - center.y) * k + dy;
	}
}
} // namespace Anim

void Queue(const Input& input) {
	std::scoped_lock lock(g_mutex);
	g_inputs.push_back(input);
	Bump();
}

void SetOpen(bool open) {
	if (g_open.exchange(open) == open) return;
	// (ds-pc-port42) Open or closed, every page state starts clean: no F2 page, no trophy page, no key capture waiting, no
	// stale combo (g_popup was set only by the launcher's frames: a value left from there made Circle / B do nothing),
	// no queued scroll. A page or a capture left behind kept eating Esc / Circle and the menu looked stuck.
	g_f1_page     = 0;
	g_kbm_capture = false;
	{
		std::scoped_lock lock(g_capture_mutex); // (remap-f2) a pad capture waiting ends, cancelled
		if (g_pad_capture.exchange(false)) g_pad_captured = "\x1b";
	}
	g_popup       = false;
	g_page_scroll = 0;
	g_pages_reset = true;
	g_text_active = false; // (online6)
	std::printf("[port] F2 menu: %s\n", open ? "open" : "closed");
	std::fflush(stdout);
	(open ? Anim::g_opened_at : Anim::g_closed_at).store(Anim::Now(), std::memory_order_relaxed); // (ds-pc-port38)
	PortLive::menu_open.store(open, std::memory_order_relaxed);
	if (!open) g_trophy_page = false; // (port34) the game again: the next F2 opens the menu, not the page
	if (open) {
		g_reset_edit = true;
		// The game sees its sticks centered (it gets nothing while the menu is open; button releases still reach it).
		g_passthrough = 0;
		for (int i = 0; i < SDL_NumJoysticks(); i++) {
			if (SDL_IsGameController(i) == SDL_FALSE) continue;
			SDL_GameController* pad = SDL_GameControllerFromInstanceID(SDL_JoystickGetDeviceInstanceID(i));
			if (pad == nullptr) continue;
			for (int axis = 0; axis < SDL_CONTROLLER_AXIS_MAX; axis++) {
				SDL_Event e {};
				e.type        = SDL_CONTROLLERAXISMOTION;
				e.caxis.which = SDL_JoystickGetDeviceInstanceID(i);
				e.caxis.axis  = static_cast<Uint8>(axis);
				e.caxis.value = 0;
				if (SDL_PushEvent(&e) == 1) g_passthrough++;
			}
		}
		g_cursor_before = SDL_ShowCursor(SDL_QUERY);
		SDL_ShowCursor(SDL_ENABLE);
	} else if (g_cursor_before >= 0) {
		SDL_ShowCursor(g_cursor_before);
	}
	{
		std::scoped_lock lock(g_mutex);
		g_inputs.clear();
	}
	Bump();
}

bool PadHeld(SDL_JoystickID which, SDL_GameControllerButton button) {
	SDL_GameController* pad = SDL_GameControllerFromInstanceID(which);
	return pad != nullptr && SDL_GameControllerGetButton(pad, button) == 1;
}

ImGuiKey PadKey(int button) {
	switch (button) {
		case SDL_CONTROLLER_BUTTON_A: return ImGuiKey_GamepadFaceDown;
		case SDL_CONTROLLER_BUTTON_B: return ImGuiKey_GamepadFaceRight;
		case SDL_CONTROLLER_BUTTON_X: return ImGuiKey_GamepadFaceLeft;
		case SDL_CONTROLLER_BUTTON_Y: return ImGuiKey_GamepadFaceUp;
		case SDL_CONTROLLER_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
		case SDL_CONTROLLER_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
		case SDL_CONTROLLER_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
		case SDL_CONTROLLER_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
		default: return ImGuiKey_None;
	}
}

ImGuiKey KeyboardKey(SDL_Keycode key) {
	switch (key) {
		case SDLK_UP: return ImGuiKey_UpArrow;
		case SDLK_DOWN: return ImGuiKey_DownArrow;
		case SDLK_LEFT: return ImGuiKey_LeftArrow;
		case SDLK_RIGHT: return ImGuiKey_RightArrow;
		case SDLK_RETURN: return ImGuiKey_Enter;
		case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
		case SDLK_SPACE: return ImGuiKey_Space;
		case SDLK_TAB: return ImGuiKey_Tab;
		case SDLK_BACKSPACE: return ImGuiKey_Backspace;
		case SDLK_HOME: return ImGuiKey_Home;
		case SDLK_END: return ImGuiKey_End;
		// (online6) editing and the clipboard shortcuts of the text fields (Ctrl+A / C / V / X / Z / Y, Shift+Insert)
		case SDLK_DELETE: return ImGuiKey_Delete;
		case SDLK_INSERT: return ImGuiKey_Insert;
		case SDLK_ESCAPE: return ImGuiKey_Escape;
		case SDLK_a: return ImGuiKey_A;
		case SDLK_c: return ImGuiKey_C;
		case SDLK_v: return ImGuiKey_V;
		case SDLK_x: return ImGuiKey_X;
		case SDLK_y: return ImGuiKey_Y;
		case SDLK_z: return ImGuiKey_Z;
		case SDLK_LCTRL: return ImGuiKey_LeftCtrl;
		case SDLK_RCTRL: return ImGuiKey_RightCtrl;
		case SDLK_LSHIFT: return ImGuiKey_LeftShift;
		case SDLK_RSHIFT: return ImGuiKey_RightShift;
		case SDLK_LALT: return ImGuiKey_LeftAlt;
		case SDLK_RALT: return ImGuiKey_RightAlt;
		default: return ImGuiKey_None;
	}
}

// (remap-f2) The pad capture's end: `name` the button's SDL name, "\x1b" cancelled, "" no button in time.
void PadCaptureEnd(const char* name) {
	std::scoped_lock lock(g_capture_mutex);
	if (g_pad_capture.exchange(false)) g_pad_captured = name;
	Bump();
}
// (remap-f2) While a pad button is being captured: true when the event was taken (never passed to the game or the
// menu's navigation). Window thread (OnEvent) and the launcher's own event loop.
bool PadCaptureFeed(const SDL_Event& e) {
	if (!g_pad_capture.load()) return false;
	if (Anim::Now() - g_pad_capture_at.load() > kPadCaptureSeconds + 1.0) { // (the menu stopped drawing: never stuck)
		PadCaptureEnd("");
		return false;
	}
	switch (e.type) {
		case SDL_CONTROLLERBUTTONDOWN: {
			g_pad_last           = static_cast<int>(e.cbutton.which);
			const char* name     = SDL_GameControllerGetStringForButton(static_cast<SDL_GameControllerButton>(e.cbutton.button));
			PadCaptureEnd(name != nullptr && *name != '\0' ? name : "\x1b");
			return true;
		}
		case SDL_CONTROLLERAXISMOTION: {
			if (e.caxis.axis == SDL_CONTROLLER_AXIS_TRIGGERLEFT || e.caxis.axis == SDL_CONTROLLER_AXIS_TRIGGERRIGHT) {
				const int   side = e.caxis.axis == SDL_CONTROLLER_AXIS_TRIGGERLEFT ? 0 : 1;
				const float v    = e.caxis.value / 32767.0f;
				if (v < 0.3f) g_pad_trig_armed[side] = true;
				else if (v > 0.6f && g_pad_trig_armed[side].load()) {
					g_pad_last = static_cast<int>(e.caxis.which);
					PadCaptureEnd(side == 0 ? "lefttrigger" : "righttrigger");
				}
			}
			return true; // (no stick navigation while it waits)
		}
		case SDL_KEYDOWN:
			if (e.key.keysym.sym == SDLK_F2 || e.key.keysym.sym == SDLK_F1) return false; // (F1/F2 close the menu: the capture ends with it)
			if (e.key.keysym.sym == SDLK_ESCAPE && e.key.repeat == 0) PadCaptureEnd("\x1b");
			return true;
		case SDL_MOUSEBUTTONDOWN:
			if (e.button.button == SDL_BUTTON_RIGHT) PadCaptureEnd("\x1b");
			return true;
		case SDL_MOUSEWHEEL:
		case SDL_TEXTINPUT:
		case SDL_TEXTEDITING: return true;
		default: return false; // (releases, mouse motion, window events: as usual)
	}
}

// Window thread, before the game's handling: true when the event is the menu's.
bool OnEvent(const SDL_Event& e) {
	// (ds-pc-port8) While F2's save and quit runs, the game gets no input (it is closing).
	if (Resume::SaveAndQuitActive()) {
		switch (e.type) {
			case SDL_KEYDOWN:
			case SDL_TEXTINPUT:
			case SDL_TEXTEDITING:
			case SDL_CONTROLLERBUTTONDOWN:
			case SDL_CONTROLLERAXISMOTION:
			case SDL_MOUSEMOTION:
			case SDL_MOUSEBUTTONDOWN:
			case SDL_MOUSEWHEEL: return true;
			default: return false;
		}
	}
	const bool open = g_open.load();
	{
		// (online6) The emulator stops SDL's text input at start (window.cpp: no SDL_TEXTINPUT at all): while F2 is open it
		// is on, so the text fields get the typed characters; closed, off again as before. Window thread (SDL's).
		static bool text_on = false;
		if (open != text_on) {
			text_on = open;
			if (open) SDL_StartTextInput();
			else SDL_StopTextInput();
		}
	}
	if (open && PadCaptureFeed(e)) return true; // (remap-f2)
	if (e.type == SDL_CONTROLLERBUTTONDOWN) g_pad_last = static_cast<int>(e.cbutton.which);
	switch (e.type) {
		case SDL_KEYDOWN:
			// (ds-pc-port41) F8: the Metricas panel (metricas.ps1's, top-centre), not the emulator's overlay.
			if (e.key.keysym.sym == SDLK_F8) {
				if (e.key.repeat == 0) MetricsPanel::Toggle();
				return true;
			}
			// (11-22) The menu key is F1 or F2 (both open it); their repeats never reach the game, and neither does F1 (consumed).
			if ((e.key.keysym.sym == SDLK_F1 || e.key.keysym.sym == SDLK_F2) && e.key.repeat != 0) return true;
			if (e.key.keysym.sym == SDLK_F2 || e.key.keysym.sym == SDLK_F1) {
				// (port26) F2: the small in-game menu (show the launcher next time; close the game). Nothing is saved or skipped.
				if (open && g_trophy_page.exchange(false)) { // (port34) the trophy page: back to the menu
					g_focus_trophy_btn_request = true;
					Bump();
					return true;
				}
				SetOpen(!open);
				Bump();
				return true;
			}
			if (!open) return false;
			if (g_kbm_capture.load() && e.key.repeat == 0) {
				// (ds-pc-port37) A key for a binding: its SDL name (what KYTY_KBM_MAP reads), Esc cancels.
				std::scoped_lock lock(g_capture_mutex);
				const SDL_Keycode k = e.key.keysym.sym;
				g_captured = k == SDLK_ESCAPE ? std::string("\x1b") : std::string(SDL_GetKeyName(k));
				if (g_captured.empty()) g_captured = "\x1b";
				g_kbm_capture = false;
				Bump();
				return true;
			}
			if (g_text_active.load()) { // (online6) a text field has the keyboard: every key goes to it (Esc leaves it)
				if (const ImGuiKey k = KeyboardKey(e.key.keysym.sym); k != ImGuiKey_None) Queue({Input::Key, k, true});
				return true;
			}
			if (e.key.keysym.sym == SDLK_ESCAPE && e.key.repeat == 0) {
				if (g_f1_page.load() == 4 && !g_popup.load() && Shop::Back()) { // (trucos19) one step back inside the shop
					Bump();
					return true;
				}
				if (g_f1_page.load() == 4) Shop::g_level = 0, Shop::g_refocus = true;
				if (g_f1_page.exchange(0) != 0) {
					Bump();
					return true;
				}
				if (g_trophy_page.exchange(false)) {
					g_focus_trophy_btn_request = true;
					Bump();
					return true;
				}
				SetOpen(false);
				return true;
			}
			if (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_PAGEUP) Queue({Input::Tab, -1});
			if (e.key.keysym.sym == SDLK_e || e.key.keysym.sym == SDLK_PAGEDOWN) Queue({Input::Tab, +1});
			if (const ImGuiKey k = KeyboardKey(e.key.keysym.sym); k != ImGuiKey_None) Queue({Input::Key, k, true});
			return true;
		case SDL_KEYUP:
			if (e.key.keysym.sym == SDLK_F1 || e.key.keysym.sym == SDLK_F2) return true;
			if (open) {
				if (const ImGuiKey k = KeyboardKey(e.key.keysym.sym); k != ImGuiKey_None) Queue({Input::Key, k, false});
			}
			return false; // (releases reach the game: nothing stays held)
		case SDL_TEXTINPUT: // (online6) the typed characters to the menu's text fields
			if (open) Queue({Input::Text, 0, false, 0, 0, std::string(e.text.text)});
			return open;
		case SDL_TEXTEDITING: return open;
		case SDL_CONTROLLERBUTTONDOWN: {
			const auto b     = e.cbutton.button;
			// R3 held, then OPTIONS: OPTIONS never reaches the game while R3 is held (its menu would open too).
			const bool combo = b == SDL_CONTROLLER_BUTTON_START && PadHeld(e.cbutton.which, SDL_CONTROLLER_BUTTON_RIGHTSTICK);
			if (combo) {
				g_start_swallowed = true;
				SetOpen(!open); // (port26) as F2: the in-game menu
				Bump();
				return true;
			}
			if (!open) return false;
			// (ds-pc-port39) Waiting for a key / mouse button to rebind: the pad does not move the nav (that could press
			// another row); Circle cancels the capture only.
			if (g_kbm_capture.load()) {
				if (b == SDL_CONTROLLER_BUTTON_B) {
					std::scoped_lock lock(g_capture_mutex);
					g_captured    = "\x1b";
					g_kbm_capture = false;
					Bump();
				}
				return true;
			}
			if (g_text_active.load() && b == SDL_CONTROLLER_BUTTON_B) { // (online6) Circle leaves the text field first
				Queue({Input::Key, ImGuiKey_Escape, true});
				Queue({Input::Key, ImGuiKey_Escape, false});
				return true;
			}
			// (ds-pc-port37) The F2 pages: Circle back to the F2 menu, L1 scrolls up (L2, a trigger, below).
			if (g_f1_page.load() != 0 && !g_popup.load()) {
				if (b == SDL_CONTROLLER_BUTTON_B) {
					if (g_f1_page.load() == 4 && Shop::Back()) { // (trucos19) one step back inside the shop
						Bump();
						return true;
					}
					if (g_f1_page.load() == 4) Shop::g_level = 0, Shop::g_refocus = true;
					g_f1_page = 0;
					Bump();
					return true;
				}
				if (b == SDL_CONTROLLER_BUTTON_LEFTSHOULDER) {
					g_page_scroll--;
					Bump();
					return true;
				}
			}
			if (b == SDL_CONTROLLER_BUTTON_B && !g_popup.load()) {
				if (g_trophy_page.exchange(false)) {
					g_focus_trophy_btn_request = true;
					Bump();
					return true;
				}
				SetOpen(false);
				return true;
			}
			if (b == SDL_CONTROLLER_BUTTON_LEFTSHOULDER) Queue({Input::Tab, -1});
			if (b == SDL_CONTROLLER_BUTTON_RIGHTSHOULDER) Queue({Input::Tab, +1});
			if (const ImGuiKey k = PadKey(b); k != ImGuiKey_None) Queue({Input::Key, k, true});
			return true;
		}
		case SDL_CONTROLLERBUTTONUP:
			if (e.cbutton.button == SDL_CONTROLLER_BUTTON_START && g_start_swallowed) {
				g_start_swallowed = false;
				return true;
			}
			if (open) {
				if (const ImGuiKey k = PadKey(e.cbutton.button); k != ImGuiKey_None) Queue({Input::Key, k, false});
			}
			return false;
		case SDL_CONTROLLERAXISMOTION: {
			if (g_passthrough > 0 && e.caxis.value == 0) {
				g_passthrough--;
				return false;
			}
			if (!open) return false;
			const float v = e.caxis.value / (e.caxis.value < 0 ? 32768.0f : 32767.0f);
			if (e.caxis.axis == SDL_CONTROLLER_AXIS_TRIGGERLEFT) {
				// (ds-pc-port37) L2 on an F2 page: scrolls down, once a press.
				if (v > 0.5f && !g_l2_down && g_f1_page.load() != 0) {
					g_page_scroll++;
					Bump();
				}
				g_l2_down = v > 0.5f ? true : (v < 0.3f ? false : g_l2_down);
				return true;
			}
			// (ds-pc-port39) The stick's analog value with a dead zone: ImGui scrolls a window by the left stick's raw magnitude
			// (its manual nav scroll), so a resting stick's drift (a few %, up) fed as is pulled the F2 pages slowly back up
			// and away from the row being rebound. Below 0.35: exactly 0; above it, 0..1 from the edge of the dead zone.
			if (g_kbm_capture.load()) return true; // (ds-pc-port39: no nav while a binding waits)
			const auto dz = [](float x) { return x <= 0.35f ? 0.0f : std::min(1.0f, (x - 0.35f) / 0.65f); };
			if (e.caxis.axis == SDL_CONTROLLER_AXIS_LEFTY) {
				Queue({Input::Analog, ImGuiKey_GamepadLStickUp, v < -0.35f, dz(-v)});
				Queue({Input::Analog, ImGuiKey_GamepadLStickDown, v > 0.35f, dz(v)});
			} else if (e.caxis.axis == SDL_CONTROLLER_AXIS_LEFTX) {
				Queue({Input::Analog, ImGuiKey_GamepadLStickLeft, v < -0.35f, dz(-v)});
				Queue({Input::Analog, ImGuiKey_GamepadLStickRight, v > 0.35f, dz(v)});
			}
			return true;
		}
		case SDL_MOUSEMOTION:
			if (!open) return false;
			Queue({Input::MousePos, 0, false, static_cast<float>(e.motion.x), static_cast<float>(e.motion.y)});
			return true;
		case SDL_MOUSEBUTTONDOWN:
		case SDL_MOUSEBUTTONUP:
			if (!open) return false;
			if (g_kbm_capture.load() && e.type == SDL_MOUSEBUTTONDOWN) {
				const char* b = e.button.button == SDL_BUTTON_LEFT ? "Mouse:Left" : e.button.button == SDL_BUTTON_RIGHT ? "Mouse:Right"
				              : e.button.button == SDL_BUTTON_MIDDLE ? "Mouse:Middle" : e.button.button == SDL_BUTTON_X1 ? "Mouse:X1" : "Mouse:X2";
				std::scoped_lock lock(g_capture_mutex);
				g_captured    = std::string((SDL_GetModState() & KMOD_SHIFT) != 0 ? "Shift+" : "") + b;
				g_kbm_capture = false;
				Bump();
				return true;
			}
			Queue({Input::MousePos, 0, false, static_cast<float>(e.button.x), static_cast<float>(e.button.y)});
			Queue({Input::MouseButton, e.button.button == SDL_BUTTON_RIGHT ? 1 : (e.button.button == SDL_BUTTON_MIDDLE ? 2 : 0),
			       e.type == SDL_MOUSEBUTTONDOWN});
			return e.type == SDL_MOUSEBUTTONDOWN;
		case SDL_MOUSEWHEEL:
			if (!open) return false;
			if (g_kbm_capture.load() && e.wheel.y != 0) {
				std::scoped_lock lock(g_capture_mutex);
				g_captured    = e.wheel.y > 0 ? "Mouse:WheelUp" : "Mouse:WheelDown";
				g_kbm_capture = false;
				Bump();
				return true;
			}
			Queue({Input::Wheel, 0, false, static_cast<float>(e.wheel.x), static_cast<float>(e.wheel.y)});
			return true;
		default: return false;
	}
}

// (online6) The clipboard of the F2 text fields (Ctrl+V, Shift+Insert, Ctrl+C / X, the "Pegar" buttons): Windows' own
// (CF_UNICODETEXT), so it works whatever ImGui backend is (not) set; pasted text loses its trailing spaces / line breaks.
std::string ClipboardText() {
	std::string out;
#if defined(_WIN32)
	if (OpenClipboard(nullptr)) {
		if (HANDLE h = GetClipboardData(CF_UNICODETEXT); h != nullptr) {
			if (const auto* w = static_cast<const wchar_t*>(GlobalLock(h)); w != nullptr) {
				const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
				if (n > 1) {
					out.resize(static_cast<size_t>(n - 1));
					WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), n, nullptr, nullptr);
				}
				GlobalUnlock(h);
			}
		}
		CloseClipboard();
	}
#else
	if (char* s = SDL_GetClipboardText(); s != nullptr) {
		out = s;
		SDL_free(s);
	}
#endif
	while (!out.empty() && (out.back() == ' ' || out.back() == '\r' || out.back() == '\n' || out.back() == '\t')) out.pop_back();
	while (!out.empty() && (out.front() == ' ' || out.front() == '\r' || out.front() == '\n' || out.front() == '\t')) out.erase(out.begin());
	return out;
}
void SetClipboard(const char* text) {
#if defined(_WIN32)
	const int n = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
	if (n <= 0 || !OpenClipboard(nullptr)) return;
	EmptyClipboard();
	if (HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE, static_cast<size_t>(n) * sizeof(wchar_t)); g != nullptr) {
		MultiByteToWideChar(CP_UTF8, 0, text, -1, static_cast<wchar_t*>(GlobalLock(g)), n);
		GlobalUnlock(g);
		if (SetClipboardData(CF_UNICODETEXT, g) == nullptr) GlobalFree(g);
	}
	CloseClipboard();
#else
	SDL_SetClipboardText(text);
#endif
}
// (online6) After a text field: right click -> Pegar / Copiar / Borrar, and a small "Pegar" button beside it (pastes the
// clipboard into the field, replacing it; trailing spaces / line breaks dropped).
void PasteInto(char* buf, size_t size) {
	const std::string clip = ClipboardText();
	if (clip.empty()) return;
	ImGui::ClearActiveID(); // (a field being edited keeps its own copy: let go of it so the new text shows)
	std::snprintf(buf, size, "%s", clip.c_str());
}
void FieldTools(const char* id, char* buf, size_t size) {
	ImGui::PushID(id);
	if (ImGui::BeginPopupContextItem("##ctx")) {
		if (ImGui::MenuItem(I18n::T("Pegar"))) PasteInto(buf, size);
		if (ImGui::MenuItem(I18n::T("Copiar"))) SetClipboard(buf);
		if (ImGui::MenuItem(I18n::T("Borrar"))) {
			ImGui::ClearActiveID();
			buf[0] = '\0';
		}
		ImGui::EndPopup();
	}
	ImGui::SameLine();
	if (ImGui::Button(I18n::T("Pegar"))) PasteInto(buf, size);
	ImGui::PopID();
}
std::string g_clip_buffer; // (ImGui keeps the pointer until the next call)
const char* ImGuiGetClipboard(ImGuiContext*) {
	g_clip_buffer = ClipboardText();
	return g_clip_buffer.c_str();
}
void ImGuiSetClipboard(ImGuiContext*, const char* text) {
	SetClipboard(text != nullptr ? text : "");
}

void DrainInputs() {
	{
		// (online6) Once: the clipboard; each frame: whether a text field has the keyboard (OnEvent reads it).
		auto& pio = ImGui::GetPlatformIO();
		if (pio.Platform_GetClipboardTextFn != &ImGuiGetClipboard) {
			pio.Platform_GetClipboardTextFn = &ImGuiGetClipboard;
			pio.Platform_SetClipboardTextFn = &ImGuiSetClipboard;
		}
		g_text_active = ImGui::GetIO().WantTextInput;
	}
	std::deque<Input> inputs;
	{
		std::scoped_lock lock(g_mutex);
		inputs.swap(g_inputs);
	}
	auto& io = ImGui::GetIO();
	for (const auto& in: inputs) {
		switch (in.kind) {
			case Input::Key:
				io.AddKeyEvent(static_cast<ImGuiKey>(in.key), in.down);
				// (online6) ImGui reads Ctrl / Shift / Alt from their own mod keys (Ctrl+V, Shift+Insert)
				if (in.key == ImGuiKey_LeftCtrl || in.key == ImGuiKey_RightCtrl) io.AddKeyEvent(ImGuiMod_Ctrl, in.down);
				if (in.key == ImGuiKey_LeftShift || in.key == ImGuiKey_RightShift) io.AddKeyEvent(ImGuiMod_Shift, in.down);
				if (in.key == ImGuiKey_LeftAlt || in.key == ImGuiKey_RightAlt) io.AddKeyEvent(ImGuiMod_Alt, in.down);
				break;
			case Input::Analog: io.AddKeyAnalogEvent(static_cast<ImGuiKey>(in.key), in.down, in.x); break;
			case Input::MousePos: io.AddMousePosEvent(in.x, in.y); break;
			case Input::MouseButton: io.AddMouseButtonEvent(in.key, in.down); break;
			case Input::Wheel: io.AddMouseWheelEvent(in.x, in.y); break;
			case Input::Text: io.AddInputCharactersUTF8(in.text.c_str()); break; // (online6)
			case Input::Tab:
				if (g_trophy_page.load()) g_trophy_tab_delta += in.key; // (port34) the trophy page's filters
				else g_tab_request = (g_tab + in.key + kTabCount) % kTabCount;
				break;
		}
	}
}



void EnsureBackground() {
	if (g_background_tried) return;
	g_background_tried = true;
	int         channels = 0;
	std::string bytes;
	{
		std::ifstream in(std::filesystem::u8path(g_rt.image_path), std::ios::binary);
		bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	}
	auto* pixels = bytes.empty() ? nullptr
	                             : stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()),
	                                                     &g_bg_w, &g_bg_h, &channels, 4);
	if (pixels == nullptr) {
		std::printf("[port] menu background %s not loaded\n", g_rt.image_path.c_str());
		return;
	}
	g_background = IM_NEW(ImTextureData)();
	g_background->Create(ImTextureFormat_RGBA32, g_bg_w, g_bg_h);
	std::memcpy(g_background->GetPixels(), pixels, static_cast<size_t>(g_bg_w) * g_bg_h * 4);
	g_background->UseColors = true;
	stbi_image_free(pixels);
	ImGui::RegisterUserTexture(g_background);
}

ImFont* FindFont(const char* name) {
	for (ImFont* font: ImGui::GetIO().Fonts->Fonts)
		if (std::strstr(font->GetDebugName(), name) != nullptr) return font;
	return nullptr;
}

const ImVec4 kGold {0.87f, 0.77f, 0.55f, 1.0f};
const ImVec4 kWarn {1.0f, 0.78f, 0.30f, 1.0f};
const ImVec4 kDim {0.72f, 0.70f, 0.66f, 1.0f};

// ---- (remap2) "Mapear mando", graphic: your controller drawn in the middle (mando-ps5.png / mando-8bitdo.png, next to
// the exe), every button marked on it with a gold node and a fine line out to a stone plate with what it does now. Hover
// or focus a plate (mouse, D-pad / stick) and it lights up with its node; press a button on the real controller and its
// node flashes. Choose a plate (click, A / Cross or Enter) and a gold ring around that button empties in 5 s, with the
// seconds in Cinzel: press the new button on your controller and the two swap their actions (the press that was asked is
// confirmed with a flash on both), saved to settings.ini and applied live at once. The same button, Esc or a right click
// cancel; nothing pressed in 5 s changes nothing. "Restaurar por defecto" puts the default buttons back. Saved as before:
// mapa_mando (KYTY_PAD_MAP), "physical=PS5 control" (SDL names), empty = default; the old values load as they are. Shown
// full screen as F2 > Mapear mando (Page) and, smaller, in the Controles tab (Draw). (remap-f2 before: a list of actions.)
ImFont* DisplayFont(); // (Cinzel, below: the launcher's)
namespace PadMap {
// Your controller's buttons (SDL names) and what each does by default (window.cpp's table): written in this order.
const char* const kPhys[] = {"a", "b", "x", "y", "leftshoulder", "rightshoulder", "lefttrigger", "righttrigger",
                             "leftstick", "rightstick", "start", "back", "dpup", "dpdown", "dpleft", "dpright"};
const char* const kDefault[] = {"Cross", "Circle", "Square", "Triangle", "L1", "R1", "L2", "R2",
                                "L3", "R3", "Options", "None", "Up", "Down", "Left", "Right"};
// The game's actions (the PS5 controls) and their short names on the plates.
const std::pair<const char*, const char*> kActions[] = {
    {"R1", "Ataque"}, {"R2", "Ataque fuerte"}, {"L1", "Bloquear"}, {"L2", "Parry"}, {"Circle", "Rodar / atrás"},
    {"Cross", "Interactuar"}, {"Square", "Usar objeto"}, {"Triangle", "Dos manos"}, {"R3", "Fijar objetivo"}, {"L3", "Botón L3"},
    {"Up", "Cambiar magia"}, {"Down", "Cambiar objeto"}, {"Left", "Cambiar mano izq."}, {"Right", "Cambiar mano der."},
    {"Options", "Menú"}, {"TouchPad", "Panel táctil"}};

using Map = std::map<std::string, std::string>;

std::string DefaultFor(const std::string& phys) {
	for (size_t i = 0; i < std::size(kPhys); i++)
		if (phys == kPhys[i]) return kDefault[i];
	return phys == "touchpad" ? "TouchPad" : "None"; // (window.cpp: the touchpad click is TouchPad, the others nothing)
}

bool KnownPhys(const std::string& k) {
	if (k == "lefttrigger" || k == "righttrigger") return true;
	return SDL_GameControllerGetButtonFromString(k.c_str()) != SDL_CONTROLLER_BUTTON_INVALID;
}

Map Parse(const std::string& text) {
	Map m;
	for (size_t i = 0; i < std::size(kPhys); i++) m[kPhys[i]] = kDefault[i];
	size_t at = 0;
	while (at <= text.size()) {
		const size_t end  = std::min(text.find(';', at), text.size());
		const auto   item = text.substr(at, end - at);
		at                = end + 1;
		const size_t eq   = item.find('=');
		if (eq != std::string::npos && KnownPhys(item.substr(0, eq))) m[item.substr(0, eq)] = item.substr(eq + 1);
	}
	return m;
}

// The 16 buttons always (as before), then any other button (touchpad, paddles, misc1) not doing its default.
std::string Format(const Map& m) {
	bool same = true;
	for (const auto& [k, v]: m) same = same && v == DefaultFor(k);
	if (same) return {};
	std::string out;
	for (const char* k: kPhys) out += (out.empty() ? "" : ";") + std::string(k) + "=" + m.at(k);
	for (const auto& [k, v]: m) {
		bool base = false;
		for (const char* b: kPhys) base = base || k == b;
		if (!base && v != DefaultFor(k)) out += ";" + k + "=" + v;
	}
	return out;
}

// The controller whose button names are shown: the one that pressed a button last, else the first one open.
SDL_GameController* ActivePad() {
	if (const int last = g_pad_last.load(); last >= 0)
		if (SDL_GameController* pad = SDL_GameControllerFromInstanceID(last); pad != nullptr) return pad;
	for (int i = 0; i < SDL_NumJoysticks(); i++) {
		if (SDL_IsGameController(i) == SDL_FALSE) continue;
		if (SDL_GameController* pad = SDL_GameControllerFromInstanceID(SDL_JoystickGetDeviceInstanceID(i)); pad != nullptr) return pad;
	}
	return nullptr;
}
bool IsPlayStation(SDL_GameController* pad) {
	if (pad == nullptr) return false;
	const auto t = SDL_GameControllerGetType(pad);
	return t == SDL_CONTROLLER_TYPE_PS3 || t == SDL_CONTROLLER_TYPE_PS4 || t == SDL_CONTROLLER_TYPE_PS5;
}

// A button of your controller by its printed name: PlayStation's, else Xbox's (Xbox 360 / One, 8BitDo in X-input...).
std::string ButtonName(const std::string& phys, bool ps) {
	struct Name {
		const char* key;
		const char* xbox;
		const char* ps;
	};
	static const Name names[] = {
	    {"a", "A", "Cruz"}, {"b", "B", "Círculo"}, {"x", "X", "Cuadrado"}, {"y", "Y", "Triángulo"},
	    {"leftshoulder", "LB", "L1"}, {"rightshoulder", "RB", "R1"}, {"lefttrigger", "LT", "L2"}, {"righttrigger", "RT", "R2"},
	    {"leftstick", "Stick izq. (L3)", "L3"}, {"rightstick", "Stick der. (R3)", "R3"}, {"start", "Start / Menu", "Options"},
	    {"back", "Back / View", "Create / Share"}, {"guide", "Guide", "PS"}, {"dpup", "Cruceta arriba", "Cruceta arriba"},
	    {"dpdown", "Cruceta abajo", "Cruceta abajo"}, {"dpleft", "Cruceta izquierda", "Cruceta izquierda"},
	    {"dpright", "Cruceta derecha", "Cruceta derecha"}, {"touchpad", "Panel táctil", "Panel táctil"}, {"misc1", "Share", "Mic"},
	    {"paddle1", "P1", "P1"}, {"paddle2", "P2", "P2"}, {"paddle3", "P3", "P3"}, {"paddle4", "P4", "P4"}};
	for (const auto& n: names)
		if (phys == n.key) return I18n::T(ps ? n.ps : n.xbox);
	return phys;
}

std::string ActionName(const std::string& control) {
	if (control == "None") return I18n::T("Sin acción");
	for (const auto& [k, v]: kActions)
		if (control == k) return I18n::T(v);
	return control;
}

// ---- The drawings: where each button is on them (pixels of the PNG, measured on it), and on which side its plate is.
struct Spot {
	const char* phys;
	float       x, y;
	int         side; // 0 the left column, 1 the right one, 2 above the controller (the touchpad)
};
// mando-ps5.png (532 x 350). The triggers are not seen from the front: their nodes sit on a silhouette above L1 / R1.
// Each column in its plates' order, chosen (by trying every order) so that the lines cross and cover buttons the least.
const Spot kSpotsPs[] = {{"lefttrigger", 105, -15, 0},  {"leftshoulder", 105, 7, 0},   {"back", 142.5f, 34, 0},   {"dpup", 96.5f, 66, 0},
                         {"dpleft", 70.5f, 92.5f, 0},   {"dpdown", 96.5f, 118, 0},    {"dpright", 121.5f, 92.5f, 0}, {"leftstick", 182, 161, 0},
                         {"righttrigger", 425, -15, 1}, {"rightshoulder", 425, 7, 1}, {"y", 435.5f, 56, 1},      {"b", 472.5f, 92.5f, 1},
                         {"x", 399, 92.5f, 1},          {"start", 389, 34, 1},        {"a", 435.5f, 131, 1},     {"rightstick", 350.5f, 161, 1},
                         {"touchpad", 265, 50, 2}};
// mando-8bitdo.png (537 x 350): Xbox's layout (A below, B right, X left, Y up), the left stick above the D-pad.
const Spot kSpotsXb[] = {{"lefttrigger", 123, -13, 0},  {"leftshoulder", 123, 10, 0},  {"leftstick", 126, 90, 0},     {"back", 225.5f, 89, 0},
                         {"dpup", 193, 140, 0},         {"dpleft", 164, 166, 0},       {"dpright", 222, 166, 0},      {"dpdown", 193, 194, 0},
                         {"righttrigger", 418, -13, 1}, {"rightshoulder", 418, 10, 1}, {"y", 416.5f, 60, 1},          {"b", 454.3f, 96.7f, 1},
                         {"x", 378.5f, 95, 1},          {"a", 416.7f, 133.3f, 1},      {"start", 312, 89, 1},         {"rightstick", 338, 164, 1}};

// ---- The two drawings as textures, once per ImGui context (the launcher's, then the game's overlay).
struct Art {
	ImTextureData* tex = nullptr;
	int            w = 0, h = 0;
	bool           tried = false, linear = false;
};
Art           g_art[2];
ImGuiContext* g_art_ctx    = nullptr;
ImFont*       g_cinzel     = nullptr;
ImGuiContext* g_cinzel_ctx = nullptr;

Art& GetArt(int which) {
	if (g_art_ctx != ImGui::GetCurrentContext()) { // (another context: the old one's textures went with it)
		for (auto& a: g_art) a = Art {};
		g_art_ctx = ImGui::GetCurrentContext();
	}
	Art& a = g_art[which];
	if (!a.tried) {
		a.tried = true;
		const auto path = g_rt.exe_path.parent_path() / (which == 0 ? L"mando-ps5.png" : L"mando-8bitdo.png");
		std::ifstream     in(path, std::ios::binary);
		const std::string bytes {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
		int               channels = 0;
		auto*             px       = bytes.empty() ? nullptr
		                                           : stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()),
		                                                                   &a.w, &a.h, &channels, 4);
		if (px == nullptr) {
			std::printf("[port] mapear mando: %s not loaded (the buttons are drawn without it)\n", which == 0 ? "mando-ps5.png" : "mando-8bitdo.png");
		} else {
			a.tex = IM_NEW(ImTextureData)();
			a.tex->Create(ImTextureFormat_RGBA32, a.w, a.h);
			std::memcpy(a.tex->GetPixels(), px, static_cast<size_t>(a.w) * a.h * 4);
			a.tex->UseColors = true;
			stbi_image_free(px);
			ImGui::RegisterUserTexture(a.tex);
		}
	}
	// The launcher's SDL_Renderer samples nearest by default: bilinear once its texture exists (the game's overlay: linear).
	if (a.tex != nullptr && !a.linear && g_start_mode && a.tex->Status == ImTextureStatus_OK && a.tex->GetTexID() != ImTextureID_Invalid) {
		SDL_SetTextureScaleMode(reinterpret_cast<SDL_Texture*>(static_cast<intptr_t>(a.tex->GetTexID())), SDL_ScaleModeLinear);
		a.linear = true;
	}
	return a;
}

// Cinzel: the launcher's; in the game's overlay (which has none) the same font from the exe's resources, added once.
ImFont* Cinzel() {
	if (DisplayFont() != nullptr) return DisplayFont();
	if (g_cinzel_ctx != ImGui::GetCurrentContext()) {
		g_cinzel_ctx = ImGui::GetCurrentContext();
		g_cinzel     = FindFont("Cinzel");
#if defined(_WIN32)
		// (ImGui 1.92 with a renderer that has textures: a font can be added at any time, it is rasterized on use)
		if (g_cinzel == nullptr && (ImGui::GetIO().BackendFlags & ImGuiBackendFlags_RendererHasTextures) != 0)
			if (HRSRC res = FindResourceW(nullptr, MAKEINTRESOURCEW(3), MAKEINTRESOURCEW(10)); res != nullptr)
				if (HGLOBAL mem = LoadResource(nullptr, res); mem != nullptr) {
					void*       data = LockResource(mem);
					const DWORD size = SizeofResource(nullptr, res);
					if (data != nullptr && size > 0) {
						ImFontConfig cfg;
						cfg.FontDataOwnedByAtlas = false; // (the resource stays mapped for the process' life)
						cfg.OversampleH = cfg.OversampleV = 1;
						cfg.PixelSnapH                    = true;
						cfg.RasterizerMultiply            = 1.08f;
						cfg.GlyphExtraAdvanceX            = 2.0f;
						ImFormatString(cfg.Name, IM_ARRAYSIZE(cfg.Name), "Cinzel (mapear mando)");
						g_cinzel = ImGui::GetIO().Fonts->AddFontFromMemoryTTF(data, static_cast<int>(size), 0.0f, &cfg);
					}
				}
#endif
	}
	return g_cinzel != nullptr ? g_cinzel : ImGui::GetFont();
}
ImFont* Body() {
	ImFont* f = FindFont("segoeui");
	return f != nullptr ? f : ImGui::GetFont();
}
ImFont* Semi() {
	ImFont* f = FindFont("seguisb");
	return f != nullptr ? f : Body();
}

// The launcher closes (before its context goes): the drawings freed.
void Shutdown() {
	if (g_art_ctx == ImGui::GetCurrentContext())
		for (auto& a: g_art)
			if (a.tex != nullptr) {
				ImGui::UnregisterUserTexture(a.tex);
				IM_DELETE(a.tex);
			}
	for (auto& a: g_art) a = Art {};
	g_art_ctx    = nullptr;
	g_cinzel     = nullptr;
	g_cinzel_ctx = nullptr;
}

// ---- Drawing helpers.
float g_alpha = 1.0f; // the whole screen's fade
ImU32 Ca(int r, int g, int b, float a) {
	return IM_COL32(r, g, b, static_cast<int>(std::clamp(a * g_alpha, 0.0f, 1.0f) * 255.0f + 0.5f));
}
float Ease(float t) {
	t = std::clamp(t, 0.0f, 1.0f);
	return 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
}
float Clamp01(float t) {
	return std::clamp(t, 0.0f, 1.0f);
}
float Hash(int i, int salt) {
	uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ static_cast<uint32_t>(salt) * 40503u;
	h ^= h >> 15;
	h *= 2246822519u;
	h ^= h >> 13;
	return static_cast<float>(h & 0xFFFFu) / 65535.0f;
}
// A soft ellipse: `in` at the centre to `out` at the rim (vertex colours, no texture), `mid` at half the radius.
void Radial(ImDrawList* dl, ImVec2 c, float rx, float ry, ImU32 in, ImU32 mid, ImU32 out, int seg = 48) {
	if (rx <= 0.5f || ry <= 0.5f) return;
	const ImVec2 uv = dl->_Data->TexUvWhitePixel;
	dl->PrimReserve(seg * 9, seg * 2 + 1);
	const ImDrawIdx base = static_cast<ImDrawIdx>(dl->_VtxCurrentIdx);
	dl->PrimWriteVtx(c, uv, in);
	constexpr float kPi = 3.14159265f;
	for (int i = 0; i < seg; i++) {
		const float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(seg);
		dl->PrimWriteVtx({c.x + std::cos(a) * rx * 0.5f, c.y + std::sin(a) * ry * 0.5f}, uv, mid);
		dl->PrimWriteVtx({c.x + std::cos(a) * rx, c.y + std::sin(a) * ry}, uv, out);
	}
	for (int i = 0; i < seg; i++) {
		const auto m0 = static_cast<ImDrawIdx>(base + 1 + i * 2), o0 = static_cast<ImDrawIdx>(m0 + 1);
		const auto m1 = static_cast<ImDrawIdx>(base + 1 + ((i + 1) % seg) * 2), o1 = static_cast<ImDrawIdx>(m1 + 1);
		dl->PrimWriteIdx(base), dl->PrimWriteIdx(m0), dl->PrimWriteIdx(m1);
		dl->PrimWriteIdx(m0), dl->PrimWriteIdx(o0), dl->PrimWriteIdx(o1);
		dl->PrimWriteIdx(m0), dl->PrimWriteIdx(o1), dl->PrimWriteIdx(m1);
	}
}
// A slab with cut corners (`c`), shaded from `top` to `bottom`, and its outline.
void Slab(ImDrawList* dl, ImVec2 a, ImVec2 b, float c, ImU32 top, ImU32 bottom, ImU32 line, float thick) {
	const ImVec2 p[8] = {{a.x + c, a.y}, {b.x - c, a.y}, {b.x, a.y + c}, {b.x, b.y - c}, {b.x - c, b.y}, {a.x + c, b.y}, {a.x, b.y - c}, {a.x, a.y + c}};
	const auto   mix  = [&](float y) {
        const float t = Clamp01((y - a.y) / std::max(1.0f, b.y - a.y));
        ImVec4      x = ImGui::ColorConvertU32ToFloat4(top), z = ImGui::ColorConvertU32ToFloat4(bottom);
        return ImGui::ColorConvertFloat4ToU32({x.x + (z.x - x.x) * t, x.y + (z.y - x.y) * t, x.z + (z.z - x.z) * t, x.w + (z.w - x.w) * t});
	};
	const ImVec2 uv = dl->_Data->TexUvWhitePixel;
	dl->PrimReserve(8 * 3, 9);
	const ImDrawIdx base = static_cast<ImDrawIdx>(dl->_VtxCurrentIdx);
	dl->PrimWriteVtx({(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f}, uv, mix((a.y + b.y) * 0.5f));
	for (const auto& q: p) dl->PrimWriteVtx(q, uv, mix(q.y));
	for (int i = 0; i < 8; i++) dl->PrimWriteIdx(base), dl->PrimWriteIdx(static_cast<ImDrawIdx>(base + 1 + i)), dl->PrimWriteIdx(static_cast<ImDrawIdx>(base + 1 + (i + 1) % 8));
	if (thick > 0.0f) dl->AddPolyline(p, 8, line, ImDrawFlags_Closed, thick);
}
void Diamond(ImDrawList* dl, ImVec2 c, float r, ImU32 col) {
	dl->AddQuadFilled({c.x, c.y - r}, {c.x + r, c.y}, {c.x, c.y + r}, {c.x - r, c.y}, col);
}
// Text with extra letter spacing; returns its width (dl nullptr: measure only).
float Spaced(ImDrawList* dl, ImFont* f, float size, ImVec2 at, ImU32 col, const std::string& text, float spacing) {
	float       x = at.x;
	const char* s = text.c_str();
	const char* e = s + text.size();
	while (s < e) {
		unsigned int cp  = 0;
		const int    len = ImTextCharFromUtf8(&cp, s, e);
		if (len <= 0) break;
		const std::string ch(s, static_cast<size_t>(len));
		if (dl != nullptr) dl->AddText(f, size, {std::round(x), std::round(at.y)}, col, ch.c_str());
		x += f->CalcTextSizeA(size, FLT_MAX, 0.0f, ch.c_str()).x + spacing;
		s += len;
	}
	return x - at.x - (text.empty() ? 0.0f : spacing);
}
// Upper case (ASCII and the accented Latin letters of the six languages).
std::string Upper(const std::string& s) {
	std::string out = s;
	for (size_t i = 0; i < out.size(); i++) {
		const auto c = static_cast<unsigned char>(out[i]);
		if (c >= 'a' && c <= 'z') out[i] = static_cast<char>(c - 32);
		else if (c == 0xC3 && i + 1 < out.size()) {
			const auto d = static_cast<unsigned char>(out[i + 1]);
			if (d >= 0xA0 && d <= 0xBE && d != 0xB7) out[i + 1] = static_cast<char>(d - 0x20);
			i++;
		}
	}
	return out;
}
// The text cut to `max_w` with "...".
std::string Fit(ImFont* f, float size, const std::string& text, float max_w) {
	if (f->CalcTextSizeA(size, FLT_MAX, 0.0f, text.c_str()).x <= max_w) return text;
	std::string t = text;
	while (!t.empty()) {
		do t.pop_back();
		while (!t.empty() && (static_cast<unsigned char>(t.back()) & 0xC0) == 0x80);
		if (!t.empty()) t.pop_back();
		if (f->CalcTextSizeA(size, FLT_MAX, 0.0f, (t + "...").c_str()).x <= max_w) return t + "...";
	}
	return "...";
}
// The button's mark on a plate's medallion: PlayStation's symbols, Xbox's letters, an arrow for the D-pad.
void Glyph(ImDrawList* dl, const std::string& phys, ImVec2 c, float r, ImU32 col, bool ps, float e) {
	const float th = std::max(1.0f, 1.6f * e);
	const auto  text = [&](const char* t, float size) {
        ImFont*      f  = Semi();
        const ImVec2 ts = f->CalcTextSizeA(size, FLT_MAX, 0.0f, t);
        dl->AddText(f, size, {std::round(c.x - ts.x * 0.5f), std::round(c.y - ts.y * 0.5f)}, col, t);
	};
	const auto arrow = [&](float dx, float dy) {
		const ImVec2 tip {c.x + dx * r * 0.62f, c.y + dy * r * 0.62f};
		const ImVec2 l {c.x - dx * r * 0.18f - dy * r * 0.5f, c.y - dy * r * 0.18f + dx * r * 0.5f};
		const ImVec2 q {c.x - dx * r * 0.18f + dy * r * 0.5f, c.y - dy * r * 0.18f - dx * r * 0.5f};
		dl->AddTriangleFilled(tip, l, q, col);
	};
	if (ps && phys == "a") {
		const float k = r * 0.5f;
		dl->AddLine({c.x - k, c.y - k}, {c.x + k, c.y + k}, col, th);
		dl->AddLine({c.x - k, c.y + k}, {c.x + k, c.y - k}, col, th);
	} else if (ps && phys == "b") {
		dl->AddCircle(c, r * 0.55f, col, 32, th);
	} else if (ps && phys == "x") {
		const float k = r * 0.5f;
		dl->AddRect({c.x - k, c.y - k}, {c.x + k, c.y + k}, col, 0.0f, 0, th);
	} else if (ps && phys == "y") {
		const float k = r * 0.62f;
		dl->AddTriangle({c.x, c.y - k}, {c.x + k * 0.92f, c.y + k * 0.62f}, {c.x - k * 0.92f, c.y + k * 0.62f}, col, th);
	} else if (phys == "a" || phys == "b" || phys == "x" || phys == "y") {
		const char t[2] = {static_cast<char>(phys[0] - 32), 0};
		text(t, std::round(r * 1.15f));
	} else if (phys == "dpup") arrow(0, -1);
	else if (phys == "dpdown") arrow(0, 1);
	else if (phys == "dpleft") arrow(-1, 0);
	else if (phys == "dpright") arrow(1, 0);
	else if (phys == "start") {
		for (int i = -1; i <= 1; i++) dl->AddLine({c.x - r * 0.45f, c.y + i * r * 0.32f}, {c.x + r * 0.45f, c.y + i * r * 0.32f}, col, th);
	} else if (phys == "back") {
		dl->AddRect({c.x - r * 0.5f, c.y - r * 0.38f}, {c.x + r * 0.2f, c.y + r * 0.2f}, col, 1.5f * e, 0, th);
		dl->AddRect({c.x - r * 0.2f, c.y - r * 0.1f}, {c.x + r * 0.5f, c.y + r * 0.48f}, col, 1.5f * e, 0, th);
	} else if (phys == "touchpad") {
		dl->AddRect({c.x - r * 0.6f, c.y - r * 0.36f}, {c.x + r * 0.6f, c.y + r * 0.36f}, col, 3.0f * e, 0, th);
	} else {
		static const std::pair<const char*, const char*> kText[] = {{"leftshoulder", "L1|LB"}, {"rightshoulder", "R1|RB"}, {"lefttrigger", "L2|LT"},
		                                                            {"righttrigger", "R2|RT"}, {"leftstick", "L3|LS"},   {"rightstick", "R3|RS"}};
		for (const auto& [k, v]: kText)
			if (phys == k) {
				const std::string both = v;
				text((ps ? both.substr(0, 2) : both.substr(3, 2)).c_str(), std::round(r * 0.92f));
				return;
			}
		text("?", std::round(r));
	}
}

// ---- The state.
enum class Mode { Idle, Capture };
Mode        g_mode = Mode::Idle;
std::string g_src;                   // Capture: the button whose action moves to the one you press
double      g_started = 0.0, g_quiet_until = 0.0;
std::string g_msg;
ImU32       g_msg_col = IM_COL32(184, 176, 160, 255);
std::string g_flash_a, g_flash_b;    // the two buttons just swapped ("*": all, after Restaurar)
double      g_flash_at  = -10.0;
int         g_last_frame = -10;
double      g_shown_at  = 0.0;
bool        g_focus_req = false;
std::map<std::string, float> g_press; // per button, eased 0..1 while it is held on the real controller

void StartCapture(const std::string& phys) {
	std::scoped_lock lock(g_capture_mutex);
	g_pad_captured.clear();
	// A trigger held right now counts only after it is released (an analog trigger resting a little in is not a press).
	bool free_trig[2] {true, true};
	for (int i = 0; i < SDL_NumJoysticks(); i++) {
		if (SDL_IsGameController(i) == SDL_FALSE) continue;
		SDL_GameController* pad = SDL_GameControllerFromInstanceID(SDL_JoystickGetDeviceInstanceID(i));
		if (pad == nullptr) continue;
		free_trig[0] = free_trig[0] && SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_TRIGGERLEFT) < 9830;
		free_trig[1] = free_trig[1] && SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_TRIGGERRIGHT) < 9830;
	}
	g_pad_trig_armed[0] = free_trig[0];
	g_pad_trig_armed[1] = free_trig[1];
	g_started           = Anim::Now();
	g_pad_capture_at    = g_started;
	g_pad_capture       = true;
	g_mode              = Mode::Capture;
	g_src               = phys;
	g_msg.clear();
	Bump();
}

// `to` takes `from`'s action and `from` takes `to`'s (a swap: nothing is left without its button); true when it changed.
bool Swap(Map& m, const std::string& from, const std::string& to, bool ps) {
	if (!m.contains(from)) m[from] = DefaultFor(from);
	if (!m.contains(to)) m[to] = DefaultFor(to);
	const std::string act = m[from], other = m[to];
	if (act == other) {
		g_msg     = std::string(I18n::T("Sin cambios:")) + " " + ButtonName(to, ps) + " = " + ActionName(act);
		g_msg_col = IM_COL32(184, 176, 160, 255);
		return false;
	}
	m[to]      = act;
	m[from]    = other;
	g_msg      = std::string(I18n::T("Asignado:")) + " " + ActionName(act) + " = " + ButtonName(to, ps) + "     \xC2\xB7     " + I18n::T("Intercambiado:") + " " +
	        ActionName(other) + " = " + ButtonName(from, ps);
	g_msg_col  = IM_COL32(236, 206, 146, 255);
	g_flash_a  = from;
	g_flash_b  = to;
	g_flash_at = Anim::Now();
	return true;
}

// The game's map live (window.cpp), when it is not the one it has (not in the launcher: JUGAR starts with it).
void SetLive(const std::string& map) {
	static std::string live = g_rt.launched.pad_map; // (what the game started with: KYTY_PAD_MAP)
	if (g_start_mode || map == live) return;
	live = map;
	Libs::Graphics::HostPadMapSetLive(map.c_str());
}

// settings.ini's mapa_mando now (only that key), and the game's map live.
void Commit(const Settings& s) {
	Settings on_disk;
	if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_saved;
	on_disk.pad_map = s.pad_map;
	if (SaveSettings(g_rt.ini_path, on_disk)) g_saved.pad_map = s.pad_map;
	else g_msg += std::string("   ") + I18n::T("No se pudo guardar settings.ini"), g_msg_col = IM_COL32(255, 199, 77, 255);
	SetLive(s.pad_map);
}

// Is `phys` held on the controller now?
bool Held(SDL_GameController* pad, const std::string& phys) {
	if (pad == nullptr) return false;
	if (phys == "lefttrigger") return SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_TRIGGERLEFT) > 16000;
	if (phys == "righttrigger") return SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_TRIGGERRIGHT) > 16000;
	const auto b = SDL_GameControllerGetButtonFromString(phys.c_str());
	return b != SDL_CONTROLLER_BUTTON_INVALID && SDL_GameControllerGetButton(pad, b) != 0;
}

// A stone button for the footer; true when pressed (click, A / Cross, Enter).
bool StoneButton(const char* id, const std::string& label, ImVec2 a, ImVec2 b, float e) {
	ImGui::SetCursorScreenPos(a);
	const bool    pressed = ImGui::InvisibleButton(id, {b.x - a.x, b.y - a.y}, ImGuiButtonFlags_EnableNav);
	ImGuiContext& g       = *ImGui::GetCurrentContext();
	const float   h       = Anim::Hot(ImGui::GetItemID(), ImGui::IsItemHovered() || (ImGui::IsItemFocused() && g.NavCursorVisible));
	auto*         dl      = ImGui::GetWindowDrawList();
	for (int k = 1; k <= 3 && h > 0.0f; k++) {
		const float o = static_cast<float>(k) * 2.5f * e;
		Slab(dl, {a.x - o, a.y - o}, {b.x + o, b.y + o}, 7.0f * e + o, 0, 0, Ca(240, 200, 120, 0.13f / static_cast<float>(k) * h), 2.0f * e);
	}
	Slab(dl, a, b, 7.0f * e, Ca(44 + static_cast<int>(18 * h), 40 + static_cast<int>(15 * h), 34 + static_cast<int>(8 * h), 0.94f),
	     Ca(20, 19, 18, 0.94f), Ca(222, 190, 128, 0.42f + 0.5f * h), std::max(1.0f, 1.3f * e));
	ImFont*      f  = Semi();
	const float  fs = std::round(20.0f * e);
	const ImVec2 ts = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, label.c_str());
	dl->AddText(f, fs, {std::round((a.x + b.x - ts.x) * 0.5f), std::round((a.y + b.y - ts.y) * 0.5f)}, Ca(236 + static_cast<int>(12 * h), 226, 204, 1.0f),
	            label.c_str());
	return pressed;
}

// ---- The screen, in the rectangle (o, size); `page`: F2's full page (title, Volver). True when the map changed (saved
// and live already); `close`: Volver.
bool Screen(Settings& s, ImVec2 o, ImVec2 size, float unit, bool page, bool* close) {
	constexpr float kPi = 3.14159265f;
	Map             m       = Parse(s.pad_map);
	bool            changed = false;
	auto*           pad     = ActivePad();
	const bool      ps      = pad == nullptr || IsPlayStation(pad);
	const bool      xb_art  = pad != nullptr && !ps;
	const double    now     = Anim::Now();
	// Shown again after a pause (opened, or the page changed): its entrance plays again.
	if (ImGui::GetFrameCount() != g_last_frame + 1) {
		g_shown_at  = now;
		g_focus_req = page;
	}
	g_last_frame = ImGui::GetFrameCount();
	// A capture ended (a button, cancelled or out of time)?
	if (g_mode == Mode::Capture) {
		bool        done = false;
		std::string got;
		{
			std::scoped_lock lock(g_capture_mutex);
			if (!g_pad_capture.load()) {
				got  = g_pad_captured;
				done = true;
			} else if (now - g_started >= kPadCaptureSeconds) {
				g_pad_capture = false;
				done          = true;
			}
			g_pad_captured.clear();
		}
		if (done) {
			g_mode        = Mode::Idle;
			g_quiet_until = now + 0.35; // (the press that ended it does not start another one)
			if (got.empty() || got == "\x1b" || got == g_src) {
				g_msg     = I18n::T(got.empty() ? "Se acabó el tiempo: no se ha cambiado nada." : "Cancelado: no se ha cambiado nada.");
				g_msg_col = IM_COL32(184, 176, 160, 255);
			} else {
				changed = Swap(m, g_src, got, ps);
			}
		}
	}
	const bool  capturing = g_mode == Mode::Capture;
	const bool  can_start = !capturing && now >= g_quiet_until;
	const float left      = capturing ? static_cast<float>(kPadCaptureSeconds - (now - g_started)) : 0.0f;

	// Layout: designed for 1600 x (800 + the title's 120) at 1080p, scaled to fit.
	const float T0 = page ? 120.0f : 0.0f;
	const float e  = std::max(0.3f, std::min({unit, size.x / 1600.0f, size.y / (800.0f + T0)}));
	const float cx = std::round(o.x + size.x * 0.5f);
	const float y0 = std::round(o.y + std::max(0.0f, (size.y - (800.0f + T0) * e) * 0.5f));
	const auto  Y  = [&](float v) { return y0 + (v + T0) * e; };
	auto*       dl = ImGui::GetWindowDrawList();
	const float t_in  = Ease(static_cast<float>((now - g_shown_at) / 0.5));
	ImFont*     cinzel = Cinzel();
	ImFont*     body   = Body();

	// The drawing: 430 px high at 1080p, its top at 118.
	const Art&   art   = GetArt(xb_art ? 1 : 0);
	const float  aw    = art.tex != nullptr ? static_cast<float>(art.w) : (xb_art ? 537.0f : 532.0f);
	const float  ah    = art.tex != nullptr ? static_cast<float>(art.h) : 350.0f;
	const float  sc    = 430.0f * e / ah;
	const float  iw = aw * sc, ih = ah * sc;
	const float  rise  = (1.0f - t_in) * 18.0f * e;
	const ImVec2 i0 {std::round(cx - iw * 0.5f), std::round(Y(118) + rise)};
	const ImVec2 icenter {cx, i0.y + ih * 0.45f};
	const auto   At = [&](const Spot& sp) { return ImVec2 {i0.x + sp.x * sc, i0.y + sp.y * sc}; };
	const Spot*  spots  = xb_art ? kSpotsXb : kSpotsPs;
	const int    nspots = xb_art ? static_cast<int>(std::size(kSpotsXb)) : static_cast<int>(std::size(kSpotsPs));

	// Behind it: a warm glow and a shadow under it.
	Radial(dl, icenter, iw * 0.95f, ih * 1.05f, Ca(150, 112, 58, 0.30f * t_in), Ca(110, 80, 40, 0.12f * t_in), Ca(60, 40, 20, 0.0f));
	Radial(dl, {cx, i0.y + ih * 0.97f}, iw * 0.55f, ih * 0.12f, Ca(0, 0, 0, 0.55f * t_in), Ca(0, 0, 0, 0.25f * t_in), Ca(0, 0, 0, 0.0f));
	// The triggers' silhouettes behind L1 / R1 (not seen from the front): dark caps rising behind the bumpers.
	const float dim_all = capturing ? 0.55f : 1.0f;
	for (int side = 0; side < 2; side++) {
		const Spot&  sp = spots[side == 0 ? 0 : 8];
		const ImVec2 c  = At(sp);
		const float  w2 = 40.0f * sc, top = c.y - 11.0f * sc, bot = c.y + 24.0f * sc;
		dl->AddRectFilled({c.x - w2, top}, {c.x + w2, bot}, Ca(30, 28, 27, 0.97f * t_in * dim_all), 16.0f * sc, ImDrawFlags_RoundCornersTop);
		dl->AddRectFilledMultiColor({c.x - w2 + 3.0f * sc, top + 4.0f * sc}, {c.x + w2 - 3.0f * sc, top + 12.0f * sc}, Ca(255, 240, 210, 0.10f * t_in * dim_all),
		                            Ca(255, 240, 210, 0.10f * t_in * dim_all), Ca(255, 240, 210, 0.0f), Ca(255, 240, 210, 0.0f));
		dl->AddRect({c.x - w2, top}, {c.x + w2, bot}, Ca(150, 132, 100, 0.45f * t_in * dim_all), 16.0f * sc, ImDrawFlags_RoundCornersTop, std::max(1.0f, e));
	}
	if (art.tex != nullptr) {
		const int tint = capturing ? 108 : 205;
		dl->AddImage(art.tex->GetTexRef(), i0, {i0.x + iw, i0.y + ih}, {0, 0}, {1, 1},
		             Ca(tint, tint - 9, tint - 26, t_in));
	} else { // (no PNG: a plain outline, the nodes still show where each button is)
		dl->AddRect(i0, {i0.x + iw, i0.y + ih * 0.6f}, Ca(180, 160, 120, 0.4f * t_in), 60.0f * e, 0, 2.0f * e);
	}

	// The plates: two columns of 8 beside the drawing (and the touchpad's above it), 70 apart from 40.
	const float pw = 330.0f * e, ph = 58.0f * e;
	const float colx[2] = {std::round(cx - iw * 0.5f - 64.0f * e - pw), std::round(cx + iw * 0.5f + 64.0f * e)};
	int         row[2]  = {0, 0};
	struct Placed {
		const Spot* sp;
		ImVec2      a, b, node;
		int         index;
	};
	std::vector<Placed> placed;
	for (int i = 0; i < nspots; i++) {
		const Spot& sp = spots[i];
		Placed      p {&sp, {}, {}, At(sp), i};
		if (sp.side == 2) {
			const float w = 270.0f * e, h = 52.0f * e;
			p.a = {std::round(cx - w * 0.5f), std::round(Y(34))};
			p.b = {p.a.x + w, p.a.y + h};
		} else {
			const int k = row[sp.side]++;
			p.a       = {colx[sp.side], std::round(Y(40.0f + static_cast<float>(k) * 70.0f))};
			p.b       = {p.a.x + pw, p.a.y + ph};
		}
		placed.push_back(p);
	}

	// Their lines first (under the nodes and the plates), then everything else.
	ImGuiContext& g    = *ImGui::GetCurrentContext();
	const float   tsec = static_cast<float>(std::fmod(now, 1000.0));
	const float   fl   = 1.0f - Clamp01(static_cast<float>((now - g_flash_at) / 0.9));
	std::vector<float> hot(placed.size(), 0.0f), press(placed.size(), 0.0f), flash(placed.size(), 0.0f), appear(placed.size(), 0.0f);
	for (size_t i = 0; i < placed.size(); i++) {
		const std::string phys = placed[i].sp->phys;
		float&            pv   = g_press[phys];
		const float       dt   = std::clamp(ImGui::GetIO().DeltaTime, 0.0f, 0.1f);
		pv += ((Held(pad, phys) ? 1.0f : 0.0f) - pv) * std::min(1.0f, dt * (Held(pad, phys) ? 30.0f : 6.0f));
		press[i]  = pv;
		flash[i]  = (g_flash_a == "*" || phys == g_flash_a || phys == g_flash_b) ? fl : 0.0f;
		appear[i] = Ease(static_cast<float>((now - g_shown_at - 0.08 - 0.025 * placed[i].index) / 0.30));
		// The plate's hit area and the node's (mouse only): an item each, the plate's reachable with the D-pad / stick.
		ImGui::PushID(placed[i].sp->phys);
		ImGui::SetCursorScreenPos(placed[i].a);
		if (g_focus_req && phys == "rightshoulder") {
			ImGui::SetKeyboardFocusHere();
			g_focus_req = false;
		}
		bool clicked = ImGui::InvisibleButton("##plate", {placed[i].b.x - placed[i].a.x, placed[i].b.y - placed[i].a.y}, ImGuiButtonFlags_EnableNav);
		bool on      = ImGui::IsItemHovered() || (ImGui::IsItemFocused() && g.NavCursorVisible);
		const ImGuiID id = ImGui::GetItemID();
		const float   nr = 18.0f * e;
		ImGui::SetCursorScreenPos({placed[i].node.x - nr, placed[i].node.y - nr});
		ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
		clicked = ImGui::InvisibleButton("##node", {nr * 2.0f, nr * 2.0f}) || clicked;
		ImGui::PopItemFlag();
		on      = on || ImGui::IsItemHovered();
		hot[i]  = Anim::Hot(id, on && !capturing);
		if (clicked && can_start) StartCapture(phys);
		ImGui::PopID();
	}
	for (size_t i = 0; i < placed.size(); i++) {
		const Placed& p    = placed[i];
		const bool    src  = capturing && g_src == p.sp->phys;
		const float   dim  = capturing && !src ? 0.22f : 1.0f;
		const float   draw = Ease(static_cast<float>((now - g_shown_at - 0.16 - 0.025 * p.index) / 0.34));
		if (draw <= 0.0f) continue;
		const float  h = std::max({hot[i], src ? 1.0f : 0.0f, flash[i]});
		ImVec2       p0, p1;
		if (p.sp->side == 2) p0 = {(p.a.x + p.b.x) * 0.5f, p.b.y}, p1 = {p0.x, p0.y + 14.0f * e};
		else if (p.sp->side == 0) p0 = {p.b.x, (p.a.y + p.b.y) * 0.5f}, p1 = {p0.x + 26.0f * e, p0.y};
		else p0 = {p.a.x, (p.a.y + p.b.y) * 0.5f}, p1 = {p0.x - 26.0f * e, p0.y};
		// The line grows from the plate to the node (entrance).
		const float  l1 = std::hypot(p1.x - p0.x, p1.y - p0.y), l2 = std::hypot(p.node.x - p1.x, p.node.y - p1.y);
		float        len = (l1 + l2) * draw;
		ImVec2       pts[3] = {p0, p1, p.node};
		int          n      = 3;
		if (len < l1) pts[1] = {p0.x + (p1.x - p0.x) * len / l1, p0.y + (p1.y - p0.y) * len / l1}, n = 2;
		else if (len < l1 + l2) {
			const float k = (len - l1) / l2;
			pts[2]        = {p1.x + (p.node.x - p1.x) * k, p1.y + (p.node.y - p1.y) * k};
		}
		dl->AddPolyline(pts, n, Ca(0, 0, 0, 0.55f * dim * draw), 0, (3.4f + 1.0f * h) * e);
		dl->AddPolyline(pts, n, Ca(214 + static_cast<int>(30 * h), 182 + static_cast<int>(30 * h), 118 + static_cast<int>(30 * h), (0.55f + 0.45f * h) * dim),
		                0, (1.3f + 0.9f * h) * e);
		if (draw < 1.0f) continue;
		// The node: a gold ring with a dark eye; hot: a pulsing halo; held on the controller: a white-gold burst.
		const ImVec2 c  = p.node;
		const float  pr = press[i];
		if (h > 0.0f) {
			const float pulse = 0.5f + 0.5f * std::sin(tsec * 4.2f);
			Radial(dl, c, (34.0f + 8.0f * pulse) * e, (34.0f + 8.0f * pulse) * e, Ca(255, 220, 150, 0.75f * h * dim), Ca(240, 190, 110, 0.28f * h * dim),
			       Ca(240, 190, 110, 0.0f), 32);
			dl->AddCircle(c, (12.0f + 9.0f * pulse) * e, Ca(255, 228, 170, (0.85f - 0.6f * pulse) * h * dim), 40, 1.6f * e);
		}
		if (pr > 0.01f)
			Radial(dl, c, 40.0f * e, 40.0f * e, Ca(255, 246, 220, 0.85f * pr), Ca(255, 210, 140, 0.35f * pr), Ca(255, 200, 120, 0.0f), 32);
		dl->AddCircleFilled(c, 7.5f * e, Ca(0, 0, 0, 0.65f * dim), 24);
		dl->AddCircle(c, 5.6f * e, Ca(236 + static_cast<int>(19 * h), 206 + static_cast<int>(30 * h), 146 + static_cast<int>(60 * pr), dim), 24,
		              std::max(1.0f, (1.6f + 0.6f * h) * e));
		dl->AddCircleFilled(c, (2.2f + 1.2f * std::max(h, pr)) * e, Ca(255, 236, 190, dim), 16);
		if (flash[i] > 0.0f) { // (just swapped: a ring bursting out)
			const float k = 1.0f - flash[i];
			dl->AddCircle(c, (10.0f + 56.0f * Ease(k)) * e, Ca(255, 226, 160, flash[i]), 48, (3.0f - 2.0f * k) * e);
			Radial(dl, c, 46.0f * e, 46.0f * e, Ca(255, 240, 200, 0.7f * flash[i]), Ca(255, 210, 140, 0.25f * flash[i]), Ca(255, 200, 120, 0.0f), 32);
		}
	}
	// The plates.
	for (size_t i = 0; i < placed.size(); i++) {
		const Placed& p   = placed[i];
		const bool    src = capturing && g_src == p.sp->phys;
		const float   ap  = appear[i];
		if (ap <= 0.0f) continue;
		const float   dim   = capturing && !src ? 0.30f : 1.0f;
		const float   h     = std::max(hot[i], src ? 1.0f : 0.0f);
		const float   slide = (1.0f - ap) * 28.0f * e * (p.sp->side == 0 ? -1.0f : (p.sp->side == 1 ? 1.0f : 0.0f));
		const ImVec2  a {p.a.x + slide, p.a.y}, b {p.b.x + slide, p.b.y};
		const float   al = ap * dim;
		const float   ch = 9.0f * e;
		for (int k = 1; k <= 3 && h > 0.0f; k++) { // (the glow around it)
			const float gr = static_cast<float>(k) * 3.0f * e;
			Slab(dl, {a.x - gr, a.y - gr}, {b.x + gr, b.y + gr}, ch + gr, 0, 0, Ca(240, 196, 112, 0.24f / static_cast<float>(k) * h * al), 2.4f * e);
		}
		Slab(dl, a, b, ch, Ca(40 + static_cast<int>(34 * h), 37 + static_cast<int>(28 * h), 33 + static_cast<int>(14 * h), 0.96f * ap),
		     Ca(17 + static_cast<int>(8 * h), 16 + static_cast<int>(6 * h), 15, 0.96f * ap), Ca(214, 182, 118, (0.38f + 0.62f * h) * al), std::max(1.0f, (1.3f + 0.4f * h) * e));
		if (flash[i] > 0.0f) Slab(dl, a, b, ch, Ca(236, 200, 130, 0.38f * flash[i] * al), Ca(236, 200, 130, 0.10f * flash[i] * al), 0, 0.0f);
		dl->AddLine({a.x + ch + 2 * e, a.y + 1.5f * e}, {b.x - ch - 2 * e, a.y + 1.5f * e}, Ca(255, 240, 210, 0.10f * al), std::max(1.0f, e)); // (top bevel)
		// The medallion at its inner end (by the line), the button's mark in it.
		const bool   rightward = p.sp->side == 0; // (left column: the medallion on the right, the text right-aligned before it)
		const float  mr        = 19.0f * e;
		const ImVec2 mc        = p.sp->side == 2 ? ImVec2 {a.x + 34.0f * e, (a.y + b.y) * 0.5f}
		                                         : ImVec2 {rightward ? b.x - 34.0f * e : a.x + 34.0f * e, (a.y + b.y) * 0.5f};
		const float  pr        = press[i];
		dl->AddCircleFilled(mc, mr, Ca(14, 13, 12, al), 32);
		dl->AddCircle(mc, mr, Ca(222 + static_cast<int>(30 * std::max(h, pr)), 190 + static_cast<int>(30 * std::max(h, pr)), 128, (0.7f + 0.3f * h) * al), 32,
		              std::max(1.0f, 1.5f * e));
		if (pr > 0.01f) dl->AddCircleFilled(mc, mr - 2.0f * e, Ca(255, 226, 160, 0.30f * pr * al), 32);
		Glyph(dl, p.sp->phys, mc, mr * 0.78f, Ca(236 + static_cast<int>(19 * h), 222 + static_cast<int>(14 * h), 196, al), ps, e);
		// The action (Cinzel) and, under it, the button's name.
		const std::string control = m.contains(p.sp->phys) ? m.at(p.sp->phys) : DefaultFor(p.sp->phys);
		const bool        none    = control == "None";
		const float       ns = std::round(23.0f * e), cs = std::round(16.0f * e);
		const float       tw = pw - 34.0f * e - mr - 30.0f * e;
		const std::string act = Fit(cinzel, ns, ActionName(control), p.sp->side == 2 ? 270.0f * e - 34.0f * e - mr - 22.0f * e : tw);
		std::string       cap = Fit(body, cs, ButtonName(p.sp->phys, ps), tw);
		if (cap == ActionName(control)) cap.clear(); // (the touchpad: its name is its action)
		const float       aw2 = cinzel->CalcTextSizeA(ns, FLT_MAX, 0.0f, act.c_str()).x;
		const float       cw  = body->CalcTextSizeA(cs, FLT_MAX, 0.0f, cap.c_str()).x;
		const float       ty  = cap.empty() ? std::round((a.y + b.y - ns) * 0.5f) : std::round((a.y + b.y) * 0.5f - (ns + cs + 1.0f * e) * 0.5f) - 1.0f * e;
		const float       tx_a = rightward ? mc.x - mr - 14.0f * e - aw2 : mc.x + mr + 14.0f * e;
		const float       tx_c = rightward ? mc.x - mr - 14.0f * e - cw : mc.x + mr + 14.0f * e;
		const ImU32       acol = none ? Ca(150, 142, 130, al) : Ca(240 + static_cast<int>(15 * h), 230 + static_cast<int>(12 * h), 208 + static_cast<int>(10 * h), al);
		dl->AddText(cinzel, ns, {std::round(tx_a), ty}, acol, act.c_str());
		dl->AddText(body, cs, {std::round(tx_c), ty + ns + 1.0f * e}, Ca(176, 166, 148, 0.95f * al), cap.c_str());
		// Not its default: a small gold lozenge at the outer end.
		if (control != DefaultFor(p.sp->phys)) {
			const float dx = p.sp->side == 1 || p.sp->side == 2 ? b.x - 14.0f * e : a.x + 14.0f * e;
			Diamond(dl, {dx, (a.y + b.y) * 0.5f}, 4.5f * e, Ca(236, 200, 130, 0.9f * al));
		}
	}

	// The countdown on the button being changed: a gold ring emptying clockwise, a spark at its end, embers rising off it,
	// the seconds in Cinzel in its middle (each new second pops in).
	if (capturing) {
		const Spot* sp = nullptr;
		for (int i = 0; i < nspots; i++)
			if (g_src == spots[i].phys) sp = &spots[i];
		if (sp != nullptr) {
			const ImVec2 c    = At(*sp);
			const float  frac = Clamp01(left / static_cast<float>(kPadCaptureSeconds));
			const float  in   = Ease(static_cast<float>((now - g_started) / 0.25));
			const float  R    = (44.0f + 10.0f * (1.0f - in)) * e;
			const float  el   = static_cast<float>(now - g_started);
			Radial(dl, c, R * 2.1f, R * 2.1f, Ca(0, 0, 0, 0.55f * in), Ca(0, 0, 0, 0.35f * in), Ca(0, 0, 0, 0.0f), 40);
			Radial(dl, c, R * 1.7f, R * 1.7f, Ca(255, 190, 100, 0.0f), Ca(255, 190, 100, 0.20f * in), Ca(255, 170, 80, 0.0f), 48);
			dl->AddCircleFilled(c, R - 4.0f * e, Ca(12, 11, 10, 0.86f * in), 64);
			dl->AddCircle(c, R, Ca(255, 255, 255, 0.10f * in), 96, 6.0f * e);
			dl->AddCircle(c, R - 9.0f * e, Ca(214, 182, 118, 0.35f * in), 64, std::max(1.0f, 1.0f * e));
			const float a0 = -kPi * 0.5f, a1 = a0 + 2.0f * kPi * frac;
			if (frac > 0.0f) {
				dl->PathArcTo(c, R, a0, a1, 96);
				dl->PathStroke(Ca(150, 112, 60, 0.6f * in), 0, 10.0f * e); // (a darker under-stroke: depth)
				dl->PathArcTo(c, R, a0, a1, 96);
				dl->PathStroke(Ca(240, 210, 150, in), 0, 5.5f * e);
				const ImVec2 tip {c.x + std::cos(a1) * R, c.y + std::sin(a1) * R};
				Radial(dl, tip, 16.0f * e, 16.0f * e, Ca(255, 248, 225, 0.95f * in), Ca(255, 210, 130, 0.45f * in), Ca(255, 190, 100, 0.0f), 24);
				dl->AddCircleFilled(tip, 3.2f * e, Ca(255, 252, 240, in), 16);
			}
			// Embers: 26, each born on the ring, rising and fading (a pure function of the time: no state).
			for (int i = 0; i < 26; i++) {
				const float period = 1.0f + 0.7f * Hash(i, 1);
				const float ph     = std::fmod(el + Hash(i, 2) * period, period) / period;
				const float ang    = Hash(i, 3) * 2.0f * kPi + std::floor((el + Hash(i, 2) * period) / period) * 2.4f;
				if (ang - a0 > 2.0f * kPi * frac + 0.3f && ph < 0.05f) continue;
				const float  rr = R * (1.0f + 0.1f * ph);
				const ImVec2 q {c.x + std::cos(ang) * rr + std::sin(ph * 6.0f + static_cast<float>(i)) * 6.0f * e, c.y + std::sin(ang) * rr - ph * ph * 70.0f * e};
				const float  aa = std::sin(ph * kPi) * (0.55f + 0.45f * Hash(i, 4)) * in;
				const float  sz = (1.0f + 1.5f * Hash(i, 5)) * e * (1.0f - 0.45f * ph);
				Radial(dl, q, sz * 4.5f, sz * 4.5f, Ca(255, 170, 80, 0.30f * aa), Ca(255, 140, 60, 0.10f * aa), Ca(255, 120, 40, 0.0f), 12);
				dl->AddCircleFilled(q, sz, Ca(255, static_cast<int>(222 - 70 * ph), static_cast<int>(150 - 80 * ph), aa), 8);
			}
			// The seconds.
			const int         secs = static_cast<int>(std::ceil(std::max(0.0f, left)));
			const float       into = static_cast<float>(secs) - left; // 0 as a new second shows .. 1
			const float       k    = 1.0f + 0.35f * (1.0f - Ease(Clamp01(into / 0.25f)));
			const std::string num  = std::to_string(secs);
			const float       fs   = std::round(44.0f * e);
			const ImVec2      ts   = cinzel->CalcTextSizeA(fs, FLT_MAX, 0.0f, num.c_str());
			const int         first = dl->VtxBuffer.Size;
			dl->AddText(cinzel, fs, {std::round(c.x - ts.x * 0.5f), std::round(c.y - ts.y * 0.52f)}, Ca(250, 236, 200, in * Clamp01(0.4f + into * 3.0f)), num.c_str());
			Anim::ScaleVertices(dl, first, c, k);
		}
	}

	// Above: the title (page) and the controller's name.
	if (page) {
		const std::string title = Upper(I18n::T("Mapear mando"));
		const float       ts    = std::round(46.0f * e);
		const float       tw    = Spaced(nullptr, cinzel, ts, {0, 0}, 0, title, 6.0f * e);
		Spaced(dl, cinzel, ts, {std::round(cx - tw * 0.5f), std::round(o.y + 26.0f * e)}, Ca(232, 206, 150, t_in), title, 6.0f * e);
		const float ry = std::round(o.y + 86.0f * e);
		const float rw = 300.0f * e * t_in;
		dl->AddLine({cx - rw, ry}, {cx - 12.0f * e, ry}, Ca(214, 182, 118, 0.6f * t_in), std::max(1.0f, e));
		dl->AddLine({cx + 12.0f * e, ry}, {cx + rw, ry}, Ca(214, 182, 118, 0.6f * t_in), std::max(1.0f, e));
		Diamond(dl, {cx, ry}, 5.0f * e, Ca(236, 206, 146, t_in));
		Diamond(dl, {cx - rw, ry}, 2.5f * e, Ca(214, 182, 118, 0.7f * t_in));
		Diamond(dl, {cx + rw, ry}, 2.5f * e, Ca(214, 182, 118, 0.7f * t_in));
	}
	{
		std::string who;
		if (pad != nullptr) {
			const char* name = SDL_GameControllerName(pad);
			who              = std::string(I18n::T("Mando:")) + " " + (name != nullptr ? name : "?");
		} else {
			who = I18n::T("No hay ningún mando conectado.");
		}
		const float  fs = std::round(18.0f * e);
		const ImVec2 ts = body->CalcTextSizeA(fs, FLT_MAX, 0.0f, who.c_str());
		const float  wy = page ? o.y + 98.0f * e : Y(6);
		dl->AddText(body, fs, {std::round(cx - ts.x * 0.5f), std::round(wy)}, pad != nullptr ? Ca(170, 160, 142, t_in) : Ca(255, 199, 77, t_in), who.c_str());
	}

	// Below: the prompt while it waits, else the last result; then Restaurar (and Volver), then the help line.
	{
		const float sy = Y(606);
		if (capturing) {
			const std::string lead = std::string(I18n::T("Pulsa el botón del mando para:")) + " ";
			const std::string act  = ActionName(m.contains(g_src) ? m.at(g_src) : DefaultFor(g_src));
			const float       fs = std::round(24.0f * e), as = std::round(28.0f * e);
			const float       lw = body->CalcTextSizeA(fs, FLT_MAX, 0.0f, lead.c_str()).x;
			const float       aw3 = cinzel->CalcTextSizeA(as, FLT_MAX, 0.0f, act.c_str()).x;
			const float       x  = std::round(cx - (lw + aw3) * 0.5f);
			dl->AddText(body, fs, {x, std::round(sy + (as - fs) * 0.6f)}, Ca(240, 232, 214, 1.0f), lead.c_str());
			dl->AddText(cinzel, as, {x + lw, std::round(sy)}, Ca(240, 210, 150, 1.0f), act.c_str());
			const std::string sub = I18n::T("El mismo botón, Esc o clic derecho: cancelar");
			const float       ss  = std::round(18.0f * e);
			const ImVec2      st  = body->CalcTextSizeA(ss, FLT_MAX, 0.0f, sub.c_str());
			dl->AddText(body, ss, {std::round(cx - st.x * 0.5f), std::round(sy + 42.0f * e)}, Ca(176, 166, 148, 1.0f), sub.c_str());
			Bump();
		} else if (!g_msg.empty()) {
			const float  fs  = std::round(21.0f * e);
			const std::string msg = Fit(body, fs, g_msg, size.x - 40.0f * e);
			const ImVec2 ts  = body->CalcTextSizeA(fs, FLT_MAX, 0.0f, msg.c_str());
			ImVec4       col = ImGui::ColorConvertU32ToFloat4(g_msg_col);
			dl->AddText(body, fs, {std::round(cx - ts.x * 0.5f), std::round(sy + 8.0f * e)}, Ca(static_cast<int>(col.x * 255), static_cast<int>(col.y * 255),
			                                                                                  static_cast<int>(col.z * 255), 1.0f), msg.c_str());
		}
		ImGui::BeginDisabled(capturing);
		const float bw = 300.0f * e, bh = 52.0f * e, gap = 22.0f * e;
		const float by = Y(682);
		const float bx = page ? cx - bw - gap * 0.5f : cx - bw * 0.5f;
		if (StoneButton("##restore", I18n::T("Restaurar por defecto"), {std::round(bx), std::round(by)}, {std::round(bx + bw), std::round(by + bh)}, e) && can_start) {
			if (!s.pad_map.empty()) m = Parse(""), changed = true;
			g_msg      = I18n::T("Restaurados los botones por defecto.");
			g_msg_col  = IM_COL32(236, 206, 146, 255);
			g_flash_a  = "*";
			g_flash_b.clear();
			g_flash_at = now;
		}
		if (page && StoneButton("##back", I18n::T("Volver"), {std::round(cx + gap * 0.5f), std::round(by)}, {std::round(cx + gap * 0.5f + bw), std::round(by + bh)}, e) &&
		    close != nullptr)
			*close = true;
		ImGui::EndDisabled();
		const std::string help = page ? I18n::T("Elige un botón y pulsa en tu mando el nuevo en 5 s: se intercambian.   ·   B o Esc: volver")
		                              : I18n::T("Elige un botón y pulsa en tu mando el nuevo en 5 s: se intercambian.");
		const float  hs = std::round(17.0f * e);
		const ImVec2 ht = body->CalcTextSizeA(hs, FLT_MAX, 0.0f, help.c_str());
		dl->AddText(body, hs, {std::round(cx - ht.x * 0.5f), std::round(Y(762))}, Ca(150, 142, 128, t_in), help.c_str());
	}
	// The rectangle as one item (the window's size and scrolling).
	ImGui::SetCursorScreenPos(o);
	ImGui::Dummy(size);
	if (changed) {
		s.pad_map = Format(m);
		Commit(s);
	}
	Bump(); // (animated: pulses, the controller's live buttons)
	return changed;
}

// In the Controles tab: the screen as wide as the tab, without the title. True when the map changed.
bool Draw(Settings& s, float unit) {
	const float  w = ImGui::GetContentRegionAvail().x;
	const ImVec2 size {w, std::round(w * 800.0f / 1600.0f)};
	const float  a = g_alpha;
	g_alpha        = ImGui::GetStyle().Alpha;
	ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4 {0, 0, 0, 0}); // (the plates light up themselves)
	const bool changed = Screen(s, ImGui::GetCursorScreenPos(), size, unit, false, nullptr);
	ImGui::PopStyleColor();
	g_alpha        = a;
	return changed;
}

// F2 > Mapear mando: full screen over the game, darkened (a vignette, a warm light behind the controller, slow ash);
// false when it closes (Volver; B / Esc are the window thread's).
bool Page(float width, float height, float unit) {
	if (Anim::g_page_seen != g_f1_page.load()) {
		Anim::g_page_seen = g_f1_page.load();
		Anim::g_page_at   = Anim::Now();
	}
	const float t = Anim::EaseOut(Anim::Progress(Anim::g_page_at.load(), 0.30));
	g_alpha       = std::max(0.01f, t);
	ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
	ImGui::SetNextWindowSize({width, height}, ImGuiCond_Always);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4 {0, 0, 0, 0}); // (the plates light up themselves)
	ImGui::Begin("Mapear mando", nullptr,
	             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse);
	auto* dl = ImGui::GetWindowDrawList();
	// The backdrop: the game darkened, darker at the edges (a vignette), ash drifting up slowly.
	dl->AddRectFilled({0, 0}, {width, height}, Ca(6, 5, 5, 0.90f));
	Radial(dl, {width * 0.5f, height * 0.5f}, width * 0.75f, height * 0.75f, Ca(0, 0, 0, 0.0f), Ca(0, 0, 0, 0.10f), Ca(0, 0, 0, 0.75f), 64);
	dl->AddRectFilledMultiColor({0, 0}, {width, height * 0.18f}, Ca(0, 0, 0, 0.55f), Ca(0, 0, 0, 0.55f), Ca(0, 0, 0, 0.0f), Ca(0, 0, 0, 0.0f));
	dl->AddRectFilledMultiColor({0, height * 0.8f}, {width, height}, Ca(0, 0, 0, 0.0f), Ca(0, 0, 0, 0.0f), Ca(0, 0, 0, 0.6f), Ca(0, 0, 0, 0.6f));
	const float ts = static_cast<float>(std::fmod(Anim::Now(), 10000.0));
	for (int i = 0; i < 40; i++) {
		const float period = 9.0f + 8.0f * Hash(i, 7);
		const float ph     = std::fmod(ts + Hash(i, 8) * period, period) / period;
		const float x      = width * Hash(i, 9) + std::sin(ph * 6.0f + static_cast<float>(i)) * 24.0f * unit;
		const float y      = height * (1.05f - 1.15f * ph);
		const float aa     = std::sin(ph * 3.14159265f) * (0.10f + 0.22f * Hash(i, 10));
		const float sz     = (0.8f + 1.6f * Hash(i, 11)) * unit;
		dl->AddCircleFilled({x, y}, sz, Ca(255, 196, 130, aa), 8);
	}
	ImGui::PushFont(Body(), std::round(20.0f * unit));
	bool       close  = false;
	const float margin = 24.0f * unit;
	(void)Screen(g_edit, {margin, margin}, {width - margin * 2.0f, height - margin * 2.0f}, unit, true, &close);
	ImGui::PopFont();
	ImGui::End();
	ImGui::PopStyleColor(2);
	ImGui::PopStyleVar(2);
	g_alpha = 1.0f;
	return !close;
}
} // namespace PadMap

// (10-27c) A video game's options row: the label on the left, "<  value  >" on the right; Left/Right (arrows, D-pad,
// left stick) or a click change the value, A / Enter the next one. The focused (or hovered) row's text goes to the
// description panel on the right.
std::string g_desc_title, g_desc_text;
float       g_unit = 1.0f;

void Describe(const char* label, const char* tip) {
	if (ImGui::IsItemFocused() || ImGui::IsItemHovered()) {
		g_desc_title = I18n::T(label);
		g_desc_text  = tip != nullptr ? tip : "";
	}
}

// (10-28c / ds-pc-port8) The dropdown list of a row or a launcher field: every value in a vertical list under (or over)
// the field at `at` (`w` x `h`), `lw` wide; true when another value was picked. Opened with ImGui::OpenPopup("##list")
// in the same ID scope. `brand`: the values coloured as the launcher's lists (NVIDIA / DLSS green, Intel / XeSS blue,
// AMD / FSR red, the game's grey).
ImVec4 BrandColor(const char* text, const ImVec4& other) {
	const auto has = [&](const char* k) { return text != nullptr && std::strstr(text, k) != nullptr; };
	if (has("NVIDIA") || has("DLSS") || has("Reflex")) return {0.46f, 0.73f, 0.0f, 1.0f};
	if (has("Intel") || has("XeSS")) return {0.27f, 0.63f, 1.0f, 1.0f};
	if (has("AMD") || has("FSR")) return {0.93f, 0.27f, 0.27f, 1.0f};
	if (has("juego") || has("game") || has("jogo") || has("jeu") || has("gioco") || has("Spiel")) return {0.78f, 0.78f, 0.78f, 1.0f};
	return other;
}

bool DropList(int count, int& current, const std::function<const char*(int)>& list, ImVec2 at, float w, float h, float lw, float itemH,
              bool brand = false) {
	bool picked = false;
	{
		// (10-29) Keep the list inside the window and the viewport: cap its height to the room there is, open it upward
		// when there is more room above than below; when capped it scrolls (wheel, D-pad/arrows, scrollbar).
		const ImGuiStyle& st     = ImGui::GetStyle();
		const ImGuiViewport* vp  = ImGui::GetMainViewport();
		const ImRect root        = ImGui::GetCurrentWindow()->RootWindow->Rect();
		const float  top         = std::max(vp->WorkPos.y, root.Min.y) + 4.0f * g_unit;
		const float  bottom      = std::min(vp->WorkPos.y + vp->WorkSize.y, root.Max.y) - 4.0f * g_unit;
		const float  left        = std::max(vp->WorkPos.x, root.Min.x);
		const float  rightEdge   = std::min(vp->WorkPos.x + vp->WorkSize.x, root.Max.x);
		const float  full        = count * itemH + std::max(0, count - 1) * st.ItemSpacing.y + st.WindowPadding.y * 2.0f;
		const float  below       = std::max(0.0f, bottom - (at.y + h));
		const float  above       = std::max(0.0f, at.y - top);
		const bool   up          = full > below && above > below;
		const float  room        = std::max(itemH + st.WindowPadding.y * 2.0f, up ? above : below);
		const float  lh          = std::min(full, std::min(room, bottom - top));
		float        lx          = std::min(at.x + w - lw, rightEdge - lw);
		lx                       = std::max(lx, left);
		float ly                 = up ? at.y - lh : at.y + h;
		ly                       = std::clamp(ly, top, std::max(top, bottom - lh));
		ImGui::SetNextWindowPos({lx, ly});
		ImGui::SetNextWindowSize({std::min(lw, rightEdge - left), lh});
		const ImGuiWindowFlags pf = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove |
		                             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
		                             (full > lh ? ImGuiWindowFlags_AlwaysVerticalScrollbar : ImGuiWindowFlags_NoScrollbar);
		if (ImGui::BeginPopupEx(ImGui::GetID("##list"), pf)) { // g_popup is set from IsPopupOpen: Circle closes this list, not the menu
			for (int i = 0; i < count; i++) {
				const bool on = i == current;
				ImGui::PushID(i);
				const char* text  = I18n::T(list(i));
				const bool  color = on || brand;
				if (color) ImGui::PushStyleColor(ImGuiCol_Text, on ? kGold : BrandColor(text, ImGui::GetStyleColorVec4(ImGuiCol_Text)));
				if (ImGui::Selectable(text, on, ImGuiSelectableFlags_None, {0, itemH})) {
					if (i != current) {
						current = i;
						picked  = true;
					}
					ImGui::CloseCurrentPopup();
				}
				if (color) ImGui::PopStyleColor();
				if (on) {
					if (ImGui::IsWindowAppearing()) {
						ImGui::SetItemDefaultFocus();
						ImGui::SetScrollHereY(0.5f);
					}
				}
				// (10-29b) Keyboard / D-pad moving the selection past the visible edge scrolls the list with it, only on the
				// frame the nav focus moved here from a nav input: never from the wheel, the scrollbar or mouse hover.
				if (!ImGui::IsWindowAppearing() && ImGui::GetCurrentContext()->NavJustMovedToId == ImGui::GetItemID() &&
				    !ImGui::GetIO().MouseDown[0] && !ImGui::IsItemVisible())
					ImGui::SetScrollHereY(0.5f);
				ImGui::PopID();
			}
			ImGui::EndPopup();
		}
	}
	return picked;
}

// The row itself: `current` changed by `step` (-1 / +1) within count; true when it changed.
// (10-28c) `list`: the label of each value; when set, a click / Cross / Enter opens a vertical dropdown list of
// every value below the row (Left/Right still cycle as a shortcut). Without it (Sí/No rows) a click cycles.
bool SelectorRow(const char* label, bool restart, int count, int& current, const char* value, const char* tip,
                 const std::function<const char*(int)>* list = nullptr) {
	ImGui::PushID(label);
	const bool  disabled = (ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled) != 0;
	const float h        = 50.0f * g_unit;
	const ImVec2 at      = ImGui::GetCursorScreenPos();
	const float  w       = ImGui::GetContentRegionAvail().x;
	const bool   clicked = ImGui::Selectable("##row", false, ImGuiSelectableFlags_None, {w, h});
	const bool   focused = !disabled && (ImGui::IsItemFocused() || ImGui::IsItemHovered());
	Describe(label, tip);
	int step = 0;
	if (!disabled && ImGui::IsItemFocused()) {
		if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft) ||
		    ImGui::IsKeyPressed(ImGuiKey_GamepadLStickLeft))
			step = -1;
		if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight) ||
		    ImGui::IsKeyPressed(ImGuiKey_GamepadLStickRight))
			step = +1;
	}
	bool picked = false;
	if (clicked && !disabled && list != nullptr && count > 1) {
		ImGui::OpenPopup("##list");
	} else if (clicked && !disabled) {
		// (A click on the left arrow's half goes back.)
		const float x = ImGui::GetIO().MousePos.x;
		step          = ImGui::IsMouseReleased(0) && x < at.x + w * 0.72f && x > at.x + w * 0.45f ? -1 : +1;
	}
	auto*       dl    = ImGui::GetWindowDrawList();
	const ImU32 ink   = ImGui::GetColorU32(disabled ? ImVec4 {0.5f, 0.48f, 0.45f, 1.0f} : ImVec4 {0.93f, 0.91f, 0.88f, 1.0f});
	const ImU32 gold  = ImGui::GetColorU32(disabled ? ImVec4 {0.5f, 0.45f, 0.35f, 1.0f} : kGold);
	const float pad   = 16.0f * g_unit;
	const float ty    = at.y + (h - ImGui::GetFontSize()) * 0.5f;
	// (perfopts-1107) The focus underline grows from the centre (eased), a soul-lit point at its middle; a changed value
	// slides in from the side it came from and fades in (180 ms).
	const ImGuiID row_id = ImGui::GetID("##row");
	const float   fv     = Anim::Hot(row_id, focused);
	if (fv > 0.0f) {
		const float cx = at.x + w * 0.5f, hw = w * 0.5f * fv;
		const ImU32 g0 = ImGui::GetColorU32(ImVec4 {kGold.x, kGold.y, kGold.z, 0.0f});
		dl->AddRectFilledMultiColor({cx - hw, at.y + h - 2 * g_unit}, {cx, at.y + h}, g0, gold, gold, g0);
		dl->AddRectFilledMultiColor({cx, at.y + h - 2 * g_unit}, {cx + hw, at.y + h}, gold, g0, g0, gold);
		dl->AddCircleFilled({cx, at.y + h - 1 * g_unit}, 2.5f * g_unit * fv, IM_COL32(206, 228, 255, static_cast<int>(220 * fv)), 8);
	}
	static std::map<ImGuiID, std::pair<int, double>> changes; // the last value, when it changed (sign: the direction)
	auto& seen = changes.try_emplace(row_id, current, -1.0).first->second;
	static std::map<ImGuiID, int> dirs;
	if (seen.first != current) {
		dirs[row_id] = current > seen.first ? 1 : -1;
		seen = {current, Anim::Now()};
	}
	const float vchange = seen.second < 0 ? 1.0f : Anim::EaseOut(Anim::Progress(seen.second, 0.18));
	if (vchange < 1.0f) Bump();
	const float vshift = (1.0f - vchange) * 18.0f * g_unit * static_cast<float>(dirs[row_id]);
	std::string text = I18n::T(label);
	dl->AddText({at.x + pad, ty}, ink, text.c_str());
	if (restart && !g_start_mode) { // (10-28b) the launcher applies everything at start: no "(requiere reinicio)" there
		// (10-28) Only the rows that truly need a restart; saved now, applied at the next start.
		const ImVec2 ls = ImGui::CalcTextSize(text.c_str());
		ImGui::PushFont(nullptr, ImGui::GetFontSize() * 0.68f);
		dl->AddText({at.x + pad + ls.x + 10 * g_unit, ty + 6 * g_unit}, ImGui::GetColorU32(ImVec4 {0.62f, 0.58f, 0.50f, 1.0f}), "(requiere reinicio)");
		ImGui::PopFont();
	}
	const char*  shown = I18n::T(value);
	const ImVec2 vs    = ImGui::CalcTextSize(shown);
	const float  right = at.x + w - pad;
	const float  mid   = right - std::max(vs.x, 220.0f * g_unit) * 0.5f - 30.0f * g_unit;
	{
		ImVec4 gc = ImGui::ColorConvertU32ToFloat4(gold);
		gc.w *= 0.25f + 0.75f * vchange;
		dl->AddText({mid - vs.x * 0.5f + vshift, ty}, ImGui::GetColorU32(gc), shown);
	}
	if (count > 1) {
		const float half = std::max(vs.x, 220.0f * g_unit) * 0.5f + 18.0f * g_unit;
		const float cy   = at.y + h * 0.5f, s2 = 7.0f * g_unit;
		const ImU32 arrow = focused ? gold : ink;
		dl->AddTriangleFilled({mid - half, cy}, {mid - half + s2, cy - s2}, {mid - half + s2, cy + s2}, arrow);
		dl->AddTriangleFilled({mid + half, cy}, {mid + half - s2, cy - s2}, {mid + half - s2, cy + s2}, arrow);
	}
	if (list != nullptr) picked = DropList(count, current, *list, at, w, h, std::max(320.0f * g_unit, w * 0.45f), 40.0f * g_unit);
	ImGui::PopID();
	if (picked) return true;
	if (step == 0 || count <= 1) return false;
	current = (current + step + count) % count;
	return true;
}

template <typename Item, typename Id, typename Label>
bool ComboRow(const char* label, bool restart, int count, const Item* items, std::string& value, Id id, Label text,
              const char* tip = nullptr) {
	int current = 0;
	for (int i = 0; i < count; i++)
		if (value == id(items[i])) current = i;
	if (count == 0) return false;
	const std::function<const char*(int)> list = [&](int i) -> const char* { return text(items[i]); };
	if (!SelectorRow(label, restart, count, current, text(items[current]), tip, &list)) return false;
	value = id(items[current]);
	return true;
}

bool ChoiceRow(const char* label, bool restart, const Choice* items, int count, std::string& value, const char* tip = nullptr) {
	return ComboRow(label, restart, count, items, value, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; }, tip);
}

bool CheckRow(const char* label, bool restart, bool& value, const char* tip = nullptr) {
	int current = value ? 1 : 0;
	if (!SelectorRow(label, restart, 2, current, current ? "Sí" : "No", tip)) return false;
	value = current == 1;
	return true;
}

// A slider row in the same look (the label left, the slider right).
bool SliderRow(const char* label, float& v, float lo, float hi, const char* format, const char* tip = nullptr) {
	ImGui::AlignTextToFramePadding();
	ImGui::Dummy({16.0f * g_unit, 0});
	ImGui::SameLine();
	ImGui::TextUnformatted(I18n::T(label));
	ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.45f + ImGui::GetCursorStartPos().x);
	ImGui::SetNextItemWidth(-16.0f * g_unit);
	ImGui::PushID(label);
	const bool changed = ImGui::SliderFloat("##s", &v, lo, hi, format);
	ImGui::PopID();
	Describe(label, tip);
	return changed;
}

// ---- (des-upstream-1106) Procedural icons for the menus (no images, no game art): drawn with the draw list at any size,
// a line weight that follows the size, in one colour. `c`: the centre, `s`: the box's side. ----
namespace Icons {
enum Kind { Back, Trophy, Pad, Keyboard, Power, Monitor, Layers, Sliders, Sun, Gauge, Chip, Gear, Speaker, Flame, Eye,
            Skull, Sword, Shield, Soul, Ring, Potion, Helm, Chest }; // (trucos7: Helm, Chest) // (trucos) the cheats' own, drawn here (no game art)

void Draw(ImDrawList* dl, Kind kind, ImVec2 c, float s, ImU32 col) {
	const float h = s * 0.5f, w = std::max(1.0f, s * 0.085f);
	const auto  P = [&](float x, float y) { return ImVec2 {c.x + x * h, c.y + y * h}; };
	switch (kind) {
		case Back: // a curved return arrow
			dl->PathArcTo(P(0.05f, 0.1f), h * 0.55f, -1.5708f, 1.5708f, 16);
			dl->PathLineTo(P(-0.6f, 0.65f));
			dl->PathStroke(col, 0, w);
			dl->AddLine(P(-0.6f, -0.45f), P(0.05f, -0.45f), col, w);
			dl->AddTriangleFilled(P(-0.85f, -0.45f), P(-0.45f, -0.75f), P(-0.45f, -0.15f), col);
			break;
		case Trophy:
			dl->PathLineTo(P(-0.5f, -0.75f));
			dl->PathLineTo(P(0.5f, -0.75f));
			dl->PathArcTo(P(0, -0.25f), h * 0.5f, 0.0f, 3.14159f, 14);
			dl->PathFillConvex(col);
			dl->PathArcTo(P(-0.5f, -0.4f), h * 0.28f, 1.5708f, 4.712f, 10);
			dl->PathStroke(col, 0, w);
			dl->PathArcTo(P(0.5f, -0.4f), h * 0.28f, -1.5708f, 1.5708f, 10);
			dl->PathStroke(col, 0, w);
			dl->AddRectFilled(P(-0.08f, 0.2f), P(0.08f, 0.55f), col);
			dl->AddRectFilled(P(-0.42f, 0.55f), P(0.42f, 0.78f), col, w);
			break;
		case Pad:
			dl->AddRect(P(-0.85f, -0.45f), P(0.85f, 0.5f), col, h * 0.4f, 0, w);
			dl->AddLine(P(-0.55f, 0.0f), P(-0.15f, 0.0f), col, w);
			dl->AddLine(P(-0.35f, -0.2f), P(-0.35f, 0.2f), col, w);
			dl->AddCircleFilled(P(0.35f, -0.12f), w * 1.1f, col);
			dl->AddCircleFilled(P(0.55f, 0.1f), w * 1.1f, col);
			break;
		case Keyboard:
			dl->AddRect(P(-0.9f, -0.5f), P(0.9f, 0.5f), col, w * 1.5f, 0, w);
			for (int r = 0; r < 2; r++)
				for (int k = 0; k < 5; k++) dl->AddRectFilled(P(-0.68f + k * 0.3f, -0.28f + r * 0.27f), P(-0.52f + k * 0.3f, -0.14f + r * 0.27f), col);
			dl->AddRectFilled(P(-0.45f, 0.22f), P(0.45f, 0.32f), col);
			break;
		case Power:
			dl->PathArcTo(P(0, 0.08f), h * 0.68f, -1.5708f + 0.75f, -1.5708f + 6.2832f - 0.75f, 24);
			dl->PathStroke(col, 0, w);
			dl->AddLine(P(0, -0.85f), P(0, -0.05f), col, w);
			break;
		case Monitor:
			dl->AddRect(P(-0.85f, -0.65f), P(0.85f, 0.4f), col, w, 0, w);
			dl->AddLine(P(0, 0.4f), P(0, 0.7f), col, w);
			dl->AddLine(P(-0.4f, 0.75f), P(0.4f, 0.75f), col, w);
			break;
		case Layers:
			for (int i = 0; i < 3; i++) {
				const float y = -0.45f + i * 0.38f;
				dl->AddQuad(P(0, y - 0.35f), P(0.8f, y), P(0, y + 0.35f), P(-0.8f, y), col, w);
			}
			break;
		case Sliders:
			for (int i = 0; i < 3; i++) {
				const float y = -0.55f + i * 0.55f, k = i == 1 ? 0.35f : i == 0 ? -0.3f : 0.0f;
				dl->AddLine(P(-0.8f, y), P(0.8f, y), col, w);
				dl->AddCircleFilled(P(k, y), w * 2.0f, col);
			}
			break;
		case Sun:
			dl->AddCircleFilled(c, h * 0.32f, col);
			for (int i = 0; i < 8; i++) {
				const float a = i * 0.7854f;
				dl->AddLine(P(std::cos(a) * 0.5f, std::sin(a) * 0.5f), P(std::cos(a) * 0.85f, std::sin(a) * 0.85f), col, w);
			}
			break;
		case Gauge:
			dl->PathArcTo(P(0, 0.25f), h * 0.75f, 3.14159f, 6.2832f, 20);
			dl->PathStroke(col, 0, w);
			dl->AddLine(P(0, 0.25f), P(0.45f, -0.25f), col, w);
			dl->AddCircleFilled(P(0, 0.25f), w * 1.6f, col);
			break;
		case Chip:
			dl->AddRect(P(-0.5f, -0.5f), P(0.5f, 0.5f), col, w, 0, w);
			dl->AddRectFilled(P(-0.2f, -0.2f), P(0.2f, 0.2f), col);
			for (int i = -1; i <= 1; i++) {
				const float k = i * 0.3f;
				dl->AddLine(P(k, -0.5f), P(k, -0.8f), col, w);
				dl->AddLine(P(k, 0.5f), P(k, 0.8f), col, w);
				dl->AddLine(P(-0.5f, k), P(-0.8f, k), col, w);
				dl->AddLine(P(0.5f, k), P(0.8f, k), col, w);
			}
			break;
		case Gear:
			dl->AddCircle(c, h * 0.5f, col, 20, w * 1.6f);
			dl->AddCircle(c, h * 0.18f, col, 12, w);
			for (int i = 0; i < 8; i++) {
				const float a = i * 0.7854f;
				dl->AddLine(P(std::cos(a) * 0.55f, std::sin(a) * 0.55f), P(std::cos(a) * 0.82f, std::sin(a) * 0.82f), col, w * 2.0f);
			}
			break;
		case Speaker:
			dl->AddQuadFilled(P(-0.8f, -0.25f), P(-0.4f, -0.25f), P(-0.4f, 0.25f), P(-0.8f, 0.25f), col);
			dl->AddTriangleFilled(P(-0.45f, -0.25f), P(0.05f, -0.7f), P(0.05f, 0.7f), col);
			dl->AddTriangleFilled(P(-0.45f, -0.25f), P(0.05f, 0.7f), P(-0.45f, 0.25f), col);
			for (int i = 1; i <= 2; i++) {
				dl->PathArcTo(P(0.05f, 0), h * (0.28f * i), -0.8f, 0.8f, 10);
				dl->PathStroke(col, 0, w);
			}
			break;
		case Flame:
			dl->PathLineTo(P(0, -0.85f));
			dl->PathBezierCubicCurveTo(P(0.75f, -0.2f), P(0.65f, 0.75f), P(0, 0.8f), 14);
			dl->PathBezierCubicCurveTo(P(-0.65f, 0.75f), P(-0.75f, -0.05f), P(-0.2f, -0.35f), 14);
			dl->PathBezierCubicCurveTo(P(-0.1f, -0.1f), P(0.1f, -0.4f), P(0, -0.85f), 8);
			dl->PathStroke(col, ImDrawFlags_Closed, w);
			dl->AddCircleFilled(P(0, 0.35f), h * 0.2f, col);
			break;
		case Eye:
			dl->PathArcTo(P(0, 0.9f), h * 1.25f, -2.3f, -0.84f, 16);
			dl->PathArcTo(P(0, -0.9f), h * 1.25f, 0.84f, 2.3f, 16);
			dl->PathStroke(col, ImDrawFlags_Closed, w);
			dl->AddCircleFilled(c, h * 0.25f, col);
			break;
		// (trucos) Original line icons in the engraved-metal look (the caller draws a dark offset copy under them).
		case Skull:
			dl->PathArcTo(P(0, -0.15f), h * 0.62f, 3.3f, 6.12f, 20);
			dl->PathLineTo(P(0.38f, 0.45f));
			dl->PathLineTo(P(-0.38f, 0.45f));
			dl->PathStroke(col, ImDrawFlags_Closed, w);
			dl->AddCircleFilled(P(-0.25f, -0.1f), h * 0.15f, col);
			dl->AddCircleFilled(P(0.25f, -0.1f), h * 0.15f, col);
			dl->AddTriangleFilled(P(0, 0.08f), P(-0.08f, 0.24f), P(0.08f, 0.24f), col);
			for (int i = -1; i <= 1; i++) dl->AddLine(P(i * 0.16f, 0.45f), P(i * 0.16f, 0.72f), col, w);
			dl->AddLine(P(-0.3f, 0.72f), P(0.3f, 0.72f), col, w);
			break;
		case Sword:
			dl->AddQuadFilled(P(-0.07f, -0.9f), P(0.07f, -0.9f), P(0.07f, 0.3f), P(-0.07f, 0.3f), col);
			dl->AddTriangleFilled(P(-0.07f, -0.9f), P(0.07f, -0.9f), P(0, -1.0f), col);
			dl->AddLine(P(-0.42f, 0.32f), P(0.42f, 0.32f), col, w * 1.6f);
			dl->AddLine(P(0, 0.34f), P(0, 0.72f), col, w * 1.3f);
			dl->AddCircleFilled(P(0, 0.82f), h * 0.1f, col);
			break;
		case Shield:
			dl->PathLineTo(P(-0.65f, -0.7f));
			dl->PathLineTo(P(0.65f, -0.7f));
			dl->PathBezierCubicCurveTo(P(0.68f, 0.1f), P(0.4f, 0.55f), P(0, 0.88f), 14);
			dl->PathBezierCubicCurveTo(P(-0.4f, 0.55f), P(-0.68f, 0.1f), P(-0.65f, -0.7f), 14);
			dl->PathStroke(col, ImDrawFlags_Closed, w);
			dl->AddLine(P(0, -0.55f), P(0, 0.6f), col, w);
			dl->AddLine(P(-0.45f, -0.15f), P(0.45f, -0.15f), col, w);
			break;
		case Soul: // a wisp: a teardrop of light with a bright core
			dl->PathLineTo(P(0, -0.92f));
			dl->PathBezierCubicCurveTo(P(0.55f, -0.3f), P(0.62f, 0.7f), P(0, 0.72f), 14);
			dl->PathBezierCubicCurveTo(P(-0.62f, 0.7f), P(-0.55f, -0.3f), P(0, -0.92f), 14);
			dl->PathStroke(col, ImDrawFlags_Closed, w);
			dl->AddCircleFilled(P(0, 0.25f), h * 0.24f, col);
			break;
		case Ring:
			dl->AddCircle(P(0, 0.2f), h * 0.55f, col, 28, w * 1.6f);
			dl->AddQuadFilled(P(0, -0.75f), P(0.22f, -0.5f), P(0, -0.28f), P(-0.22f, -0.5f), col);
			break;
		case Helm: // (trucos7) a closed great helm with a visor slit
			dl->PathArcTo(P(0, -0.1f), h * 0.62f, 3.14159f, 6.2832f, 20);
			dl->PathLineTo(P(0.62f, 0.75f));
			dl->PathLineTo(P(-0.62f, 0.75f));
			dl->PathStroke(col, ImDrawFlags_Closed, w);
			dl->AddLine(P(-0.45f, 0.05f), P(0.45f, 0.05f), col, w * 1.4f);
			dl->AddLine(P(0, -0.7f), P(0, -0.05f), col, w);
			for (int i = -1; i <= 1; i += 2) dl->AddCircleFilled(P(0.3f * i, 0.45f), w * 0.9f, col);
			break;
		case Chest: // (trucos7) an iron-bound chest (the shop)
			dl->AddRect(P(-0.8f, -0.15f), P(0.8f, 0.7f), col, 0, 0, w);
			dl->PathArcTo(P(0, -0.15f), h * 0.8f, 3.14159f, 6.2832f, 20);
			dl->PathStroke(col, 0, w);
			dl->AddLine(P(-0.8f, -0.15f), P(0.8f, -0.15f), col, w);
			dl->AddRectFilled(P(-0.12f, -0.05f), P(0.12f, 0.25f), col);
			dl->AddLine(P(-0.45f, -0.75f), P(-0.45f, 0.7f), col, w * 0.8f);
			dl->AddLine(P(0.45f, -0.75f), P(0.45f, 0.7f), col, w * 0.8f);
			break;
		case Potion:
			dl->AddRect(P(-0.14f, -0.85f), P(0.14f, -0.55f), col, 0, 0, w);
			dl->PathLineTo(P(-0.14f, -0.55f));
			dl->PathLineTo(P(-0.14f, -0.3f));
			dl->PathArcTo(P(0, 0.25f), h * 0.58f, -1.95f, 1.95f + 3.14159f, 28);
			dl->PathLineTo(P(0.14f, -0.3f));
			dl->PathLineTo(P(0.14f, -0.55f));
			dl->PathStroke(col, 0, w);
			dl->AddCircleFilled(P(0, 0.35f), h * 0.38f, col);
			break;
	}
}

// The icon of a section by its (Spanish) title; -1 when it has none.
int ForSection(const char* title) {
	static const std::pair<const char*, Kind> map[] = {
	    {"Imagen", Monitor},        {"Frame generation", Layers}, {"Calidad gráfica", Sliders}, {"Ray tracing", Sun},
	    {"Modos Souls", Flame},     {"Pantalla", Monitor},        {"Rendimiento", Gauge},       {"Controles", Pad},
	    {"Tu equipo", Chip},        {"Configurador", Gear},       {"Ajustes del juego", Gear},  {"Sonido", Speaker},
	    {"Trofeos", Trophy},        {"Gráficos", Eye},            {"Sistema", Chip},  {"Conexión online", Soul}};
	for (const auto& [name, kind]: map)
		if (std::strcmp(name, title) == 0) return kind;
	return -1;
}
} // namespace Icons

// ---- (perfopts-1107) The F2 menu in the Demon's Souls look, all procedural (no game art, no logos): cold grey-blue
// carved stone (the Nexus), tarnished brass filigree (Boletaria's gothic ironwork and gilding), pale blue-white soul
// light (the archstones and the souls the player carries), a warm candle flicker (the Nexus' candles), drifting fog and
// floating soul motes. Drawn only while the menu is open or animating. ----
ImFont* DisplayFont(); // (Cinzel, below)
namespace Souls {
// Palette.
constexpr int kStoneT[3] = {38, 42, 50}, kStoneB[3] = {13, 15, 19};      // Nexus stone, lit from above
constexpr int kBrass[3] = {178, 152, 102}, kBrassDim[3] = {112, 96, 68}; // tarnished brass / gold
constexpr int kBrassHi[3] = {222, 200, 150};
constexpr int kSoul[3] = {206, 228, 255};                                 // soul light
constexpr int kCandle[3] = {255, 186, 104};                               // candle flame
inline ImU32 C(const int (&c)[3], float a) {
	return IM_COL32(c[0], c[1], c[2], static_cast<int>(std::clamp(a, 0.0f, 1.0f) * 255.0f));
}
inline float Time() {
	return static_cast<float>(std::fmod(Anim::Now(), 3600.0));
}
// A candle's flicker, 0.75..1.
inline float Flicker(float seed = 0.0f) {
	const float t = Time() + seed;
	return 0.87f + 0.06f * std::sin(t * 7.3f) + 0.04f * std::sin(t * 13.1f + 1.7f) + 0.03f * std::sin(t * 23.7f + 0.4f);
}
inline float Hash(int i) {
	const uint32_t x = static_cast<uint32_t>(i) * 2654435761u ^ 0x9E3779B9u;
	return static_cast<float>((x >> 8) & 0xFFFF) / 65535.0f;
}
// Fog: slow drifting soft banks behind the panel [a, b] (stacked translucent discs, grey-blue).
[[maybe_unused]] void Fog(ImDrawList* dl, ImVec2 a, ImVec2 b, float alpha) {
	const float t = Time(), w = b.x - a.x, h = b.y - a.y;
	for (int i = 0; i < 9; i++) {
		const float sx = Hash(i * 3 + 1), sy = Hash(i * 3 + 2), sr = Hash(i * 3 + 3);
		const float x  = a.x - w * 0.25f + std::fmod(sx * w * 1.5f + t * (6.0f + 8.0f * sr), w * 1.5f);
		const float y  = a.y + h * (0.15f + 0.8f * sy) + std::sin(t * 0.21f + i) * h * 0.03f;
		const float r  = h * (0.22f + 0.22f * sr);
		for (int k = 4; k >= 1; k--)
			dl->AddCircleFilled({x, y}, r * (0.45f + 0.18f * k), IM_COL32(150, 162, 178, static_cast<int>(alpha * 7.0f)), 32);
	}
}
// Soul motes: pale points rising slowly with a sway, fading in and out; `count` of them over [a, b].
[[maybe_unused]] void Motes(ImDrawList* dl, ImVec2 a, ImVec2 b, float alpha, int count, float scale) {
	const float t = Time(), w = b.x - a.x, h = b.y - a.y;
	for (int i = 0; i < count; i++) {
		const float speed = 0.05f + 0.07f * Hash(i * 7 + 11);
		const float life  = std::fmod(t * speed + Hash(i * 7 + 12), 1.0f);
		const float x     = a.x + w * Hash(i * 7 + 13) + std::sin(t * (0.6f + Hash(i) * 0.8f) + i) * 10.0f * scale;
		const float y     = b.y - h * life;
		const float fade  = std::sin(life * 3.14159f);
		const float r     = (1.1f + 1.6f * Hash(i * 7 + 14)) * scale;
		const bool  warm  = (i % 6) == 0; // a few candle embers among the souls
		const auto& col   = warm ? kCandle : kSoul;
		dl->AddCircleFilled({x, y}, r * 3.2f, C(col, alpha * fade * 0.10f), 12);
		dl->AddCircleFilled({x, y}, r, C(col, alpha * fade * 0.85f), 8);
	}
}
// A small filigree flourish: a diamond and two curls either side of `c`, `s` wide.
void Flourish(ImDrawList* dl, ImVec2 c, float s, ImU32 col, float wgt) {
	dl->AddQuadFilled({c.x, c.y - s * 0.16f}, {c.x + s * 0.12f, c.y}, {c.x, c.y + s * 0.16f}, {c.x - s * 0.12f, c.y}, col);
	for (int side = -1; side <= 1; side += 2) {
		const float d = static_cast<float>(side);
		dl->PathLineTo({c.x + d * s * 0.16f, c.y});
		dl->PathBezierCubicCurveTo({c.x + d * s * 0.32f, c.y - s * 0.22f}, {c.x + d * s * 0.52f, c.y - s * 0.12f}, {c.x + d * s * 0.44f, c.y}, 10);
		dl->PathBezierCubicCurveTo({c.x + d * s * 0.38f, c.y + s * 0.08f}, {c.x + d * s * 0.30f, c.y + s * 0.02f}, {c.x + d * s * 0.33f, c.y - s * 0.03f}, 6);
		dl->PathStroke(col, 0, wgt);
		dl->AddCircleFilled({c.x + d * s * 0.5f, c.y}, wgt * 0.9f, col, 6);
	}
}
// A divider: a hairline fading to both ends, a flourish in the middle.
// (trucos4) A vignette over the whole screen: the edges darkened towards the corners (four gradients), alpha 0..1.
void Vignette(ImDrawList* dl, float w, float h, float alpha) {
	const ImU32 dark = IM_COL32(0, 0, 0, static_cast<int>(200 * alpha)), clear = IM_COL32(0, 0, 0, 0);
	const float ex = std::round(w * 0.22f), ey = std::round(h * 0.26f);
	dl->AddRectFilledMultiColor({0, 0}, {ex, h}, dark, clear, clear, dark);
	dl->AddRectFilledMultiColor({w - ex, 0}, {w, h}, clear, dark, dark, clear);
	dl->AddRectFilledMultiColor({0, 0}, {w, ey}, dark, dark, clear, clear);
	dl->AddRectFilledMultiColor({0, h - ey}, {w, h}, clear, clear, dark, dark);
}
// (trucos4) A thin aged-gold double frame with corner flourishes, inset `m` px in [a, b].
void Frame(ImDrawList* dl, ImVec2 a, ImVec2 b, float unit, float alpha, float m) {
	const float t = std::max(1.0f, std::round(unit));
	const ImVec2 o {std::round(a.x + m), std::round(a.y + m)}, p {std::round(b.x - m), std::round(b.y - m)};
	dl->AddRect(o, p, C(kBrassDim, alpha * 0.8f), 0, 0, t);
	dl->AddRect({o.x + 4 * t, o.y + 4 * t}, {p.x - 4 * t, p.y - 4 * t}, C(kBrass, alpha * 0.45f), 0, 0, t);
	for (int i = 0; i < 4; i++) {
		const ImVec2 c {i & 1 ? p.x : o.x, i & 2 ? p.y : o.y};
		Flourish(dl, c, 18 * unit, C(kBrass, alpha * 0.9f), t);
	}
}
void Divider(ImDrawList* dl, float x0, float x1, float y, float unit, float alpha) {
	const float mid = (x0 + x1) * 0.5f, gap = 30.0f * unit;
	const ImU32 on = C(kBrass, 0.75f * alpha), off = C(kBrass, 0.0f);
	const float hw = std::max(1.0f, unit);
	dl->AddRectFilledMultiColor({x0, y}, {mid - gap, y + hw}, off, on, on, off);
	dl->AddRectFilledMultiColor({mid + gap, y}, {x1, y + hw}, on, off, off, on);
	Flourish(dl, {mid, y + hw * 0.5f}, 52.0f * unit, C(kBrass, 0.85f * alpha), 1.2f * unit);
}
// The panel: carved stone (a cold vertical gradient, faint etched courses), a double brass frame with corner ornaments.
void Panel(ImDrawList* dl, ImVec2 a, ImVec2 b, float unit, float alpha, float r) {
	dl->AddRectFilledMultiColor(a, b, C(kStoneT, 0.95f * alpha), C(kStoneT, 0.95f * alpha), C(kStoneB, 0.97f * alpha), C(kStoneB, 0.97f * alpha));
	// Etched courses of masonry: faint dark lines with a lighter lip under them.
	const float course = 46.0f * unit;
	for (float y = a.y + course; y < b.y - course * 0.5f; y += course) {
		dl->AddLine({a.x + 10 * unit, y}, {b.x - 10 * unit, y}, IM_COL32(0, 0, 0, static_cast<int>(36 * alpha)), 1.0f);
		dl->AddLine({a.x + 10 * unit, y + 1}, {b.x - 10 * unit, y + 1}, IM_COL32(170, 184, 204, static_cast<int>(8 * alpha)), 1.0f);
	}
	// The soft light from above (the Nexus' high windows) and a dark vignette at the foot.
	dl->AddRectFilledMultiColor(a, {b.x, a.y + (b.y - a.y) * 0.35f}, IM_COL32(180, 196, 220, static_cast<int>(16 * alpha)),
	                            IM_COL32(180, 196, 220, static_cast<int>(16 * alpha)), IM_COL32(180, 196, 220, 0), IM_COL32(180, 196, 220, 0));
	// The frame: an outer brass line, an inner hairline.
	const float hw = std::max(1.0f, unit);
	dl->AddRect(a, b, C(kBrassDim, 0.9f * alpha), r, 0, 1.6f * hw);
	const float in = 6.0f * unit;
	dl->AddRect({a.x + in, a.y + in}, {b.x - in, b.y - in}, C(kBrass, 0.45f * alpha), std::max(0.0f, r - in * 0.5f), 0, hw);
	// Corner ornaments: an L of brass with a small diamond and a curl.
	const float k = 22.0f * unit;
	const ImU32 bc = C(kBrass, 0.9f * alpha);
	const ImVec2 corners[4] = {a, {b.x, a.y}, b, {a.x, b.y}};
	for (int i = 0; i < 4; i++) {
		const float sx = (i == 0 || i == 3) ? 1.0f : -1.0f, sy = i < 2 ? 1.0f : -1.0f;
		const ImVec2 c {corners[i].x + sx * in, corners[i].y + sy * in};
		dl->AddLine(c, {c.x + sx * k, c.y}, bc, 2.0f * hw);
		dl->AddLine(c, {c.x, c.y + sy * k}, bc, 2.0f * hw);
		const ImVec2 d {c.x + sx * k * 0.42f, c.y + sy * k * 0.42f};
		dl->AddQuadFilled({d.x, d.y - 4 * unit}, {d.x + 4 * unit, d.y}, {d.x, d.y + 4 * unit}, {d.x - 4 * unit, d.y}, bc);
		dl->PathArcTo({c.x + sx * k, c.y + sy * 5 * unit}, 5 * unit, sx > 0 ? -1.5708f : -1.5708f, sx > 0 ? 1.5708f : -4.712f, 8);
		dl->PathStroke(bc, 0, hw);
	}
}
// The header's sigil: a carved monolith (an archstone-like slab, our own drawing) with an etched rune and a soul glow
// that breathes; `s`: its height.
void Monolith(ImDrawList* dl, ImVec2 c, float s, float unit, float alpha) {
	const float t = Time(), breathe = 0.65f + 0.35f * std::sin(t * 1.6f);
	const float hw = s * 0.26f, top = c.y - s * 0.5f, foot = c.y + s * 0.5f;
	// The glow behind it.
	for (int k = 5; k >= 1; k--) dl->AddCircleFilled({c.x, c.y - s * 0.05f}, s * (0.28f + 0.11f * k), C(kSoul, alpha * breathe * 0.035f), 32);
	// The slab: a pointed top, slightly tapering sides; darker on the right (lit from the left).
	const ImVec2 p[5] = {{c.x, top}, {c.x + hw, top + s * 0.22f}, {c.x + hw * 0.86f, foot}, {c.x - hw * 0.86f, foot}, {c.x - hw, top + s * 0.22f}};
	dl->AddConvexPolyFilled(p, 5, IM_COL32(58, 64, 74, static_cast<int>(240 * alpha)));
	dl->AddTriangleFilled(p[0], p[1], {c.x, foot}, IM_COL32(20, 22, 28, static_cast<int>(110 * alpha)));
	dl->AddQuadFilled(p[1], p[2], {c.x, foot}, p[0], IM_COL32(20, 22, 28, static_cast<int>(110 * alpha)));
	dl->AddPolyline(p, 5, C(kBrassDim, 0.9f * alpha), ImDrawFlags_Closed, std::max(1.0f, unit * 1.2f));
	// The plinth.
	dl->AddRectFilled({c.x - hw * 1.2f, foot}, {c.x + hw * 1.2f, foot + s * 0.07f}, IM_COL32(42, 46, 54, static_cast<int>(240 * alpha)));
	dl->AddRect({c.x - hw * 1.2f, foot}, {c.x + hw * 1.2f, foot + s * 0.07f}, C(kBrassDim, 0.8f * alpha), 0, 0, std::max(1.0f, unit));
	// The etched rune: a vertical stroke, a ring, two short bars, glowing soul-white.
	const ImU32 rune = C(kSoul, alpha * (0.55f + 0.45f * breathe));
	const float rw = std::max(1.0f, s * 0.035f);
	dl->AddLine({c.x, top + s * 0.2f}, {c.x, foot - s * 0.12f}, rune, rw);
	dl->AddCircle({c.x, c.y - s * 0.02f}, s * 0.11f, rune, 16, rw);
	dl->AddLine({c.x - s * 0.12f, top + s * 0.3f}, {c.x + s * 0.12f, top + s * 0.3f}, rune, rw);
	dl->AddLine({c.x - s * 0.08f, foot - s * 0.2f}, {c.x + s * 0.08f, foot - s * 0.2f}, rune, rw);
}
// Engraved text: a dark cut above-left, a faint light lip below-right, the face in tarnished gold.
void Etched(ImDrawList* dl, ImFont* f, float size, ImVec2 at, const char* text, float alpha) {
	const float o = std::max(1.0f, size * 0.04f);
	dl->AddText(f, size, {at.x - o, at.y - o}, IM_COL32(0, 0, 0, static_cast<int>(200 * alpha)), text);
	dl->AddText(f, size, {at.x + o, at.y + o}, IM_COL32(255, 236, 196, static_cast<int>(40 * alpha)), text);
	dl->AddText(f, size, at, C(kBrassHi, alpha), text);
}
// A sigil medallion for an entry's icon: an engraved ring with tick marks (turning slowly while focused), the icon
// inside in brass, a soul halo when focused.
void Medallion(ImDrawList* dl, ImVec2 c, float r, float v, float alpha, bool danger) {
	const float t = Time();
	if (v > 0.0f)
		for (int k = 4; k >= 1; k--) dl->AddCircleFilled(c, r * (1.0f + 0.16f * k), C(danger ? kCandle : kSoul, alpha * v * 0.05f), 32);
	dl->AddCircleFilled(c, r, IM_COL32(24, 27, 33, static_cast<int>(235 * alpha)), 32);
	// A bevel: dark rim above, light below (engraved).
	dl->PathArcTo(c, r * 0.94f, 3.4f, 6.0f, 16);
	dl->PathStroke(IM_COL32(0, 0, 0, static_cast<int>(160 * alpha)), 0, std::max(1.0f, r * 0.06f));
	dl->PathArcTo(c, r * 0.94f, 0.3f, 2.8f, 16);
	dl->PathStroke(IM_COL32(200, 210, 230, static_cast<int>(30 * alpha)), 0, std::max(1.0f, r * 0.05f));
	dl->AddCircle(c, r, C(kBrass, (0.55f + 0.4f * v) * alpha), 32, std::max(1.0f, r * 0.06f));
	const float spin = t * 0.5f * v;
	for (int i = 0; i < 12; i++) {
		const float a = spin + i * 0.5236f, k0 = i % 3 == 0 ? 0.70f : 0.78f;
		dl->AddLine({c.x + std::cos(a) * r * k0, c.y + std::sin(a) * r * k0}, {c.x + std::cos(a) * r * 0.86f, c.y + std::sin(a) * r * 0.86f},
		            C(kBrassDim, (0.5f + 0.5f * v) * alpha), std::max(1.0f, r * 0.04f));
	}
}
} // namespace Souls

// (perfopts-1107) The selection highlight that glides between the F2 menu's entries: each frame the focused (else
// hovered) entry gives its rectangle; the highlight eases towards it and is drawn under the entries (DrawInGameMenu).
namespace Glide {
ImVec2 g_a {0, 0}, g_b {0, 0}, g_ta {0, 0}, g_tb {0, 0};
bool   g_have = false, g_shown = false, g_focused = false, g_danger = false;
float  g_alpha = 0.0f;
void   Begin() {
    g_have    = false;
    g_focused = false;
}
// Whether the entry being drawn should be the target (a focused one wins over a hovered one).
bool Want(bool on, bool danger) {
	if (!on) return false;
	const bool focused = ImGui::IsItemFocused();
	if (g_have && g_focused && !focused) return false;
	g_focused = focused;
	g_danger  = danger;
	return true;
}
void Target(ImVec2 a, ImVec2 b) {
	g_ta   = a;
	g_tb   = b;
	g_have = true;
}
// Eases and draws (call with the entries' channel below the current one).
void Draw(ImDrawList* dl, float unit, float alpha) {
	const float dt = std::clamp(ImGui::GetIO().DeltaTime, 0.0f, 0.1f);
	const float k  = std::min(1.0f, dt * 16.0f);
	if (g_have && !g_shown) g_a = g_ta, g_b = g_tb, g_shown = true;
	if (g_have) {
		g_a.x += (g_ta.x - g_a.x) * k, g_a.y += (g_ta.y - g_a.y) * k;
		g_b.x += (g_tb.x - g_b.x) * k, g_b.y += (g_tb.y - g_b.y) * k;
	}
	g_alpha += ((g_have ? 1.0f : 0.0f) - g_alpha) * k;
	if (g_alpha < 0.01f) {
		g_shown = g_have;
		return;
	}
	if (std::fabs(g_ta.y - g_a.y) > 0.5f || std::fabs(g_alpha - (g_have ? 1.0f : 0.0f)) > 0.01f) Bump();
	const float al = alpha * g_alpha, g = 4.0f * unit, r = 8.0f * unit;
	const auto& c  = g_danger ? Souls::kCandle : Souls::kSoul;
	const float fl = Souls::Flicker(0.3f);
	for (int i = 3; i >= 1; i--) {
		const float e = g * static_cast<float>(i);
		dl->AddRectFilled({g_a.x - e, g_a.y - e}, {g_b.x + e, g_b.y + e}, Souls::C(c, al * 0.035f * fl), r + e);
	}
	dl->AddRect({g_a.x - 2 * unit, g_a.y - 2 * unit}, {g_b.x + 2 * unit, g_b.y + 2 * unit}, Souls::C(c, al * 0.35f * fl), r + 2 * unit, 0,
	            std::max(1.0f, unit));
	// The accent: a brass bar on the left with a soul-lit tip.
	const float my = (g_a.y + g_b.y) * 0.5f, hh = (g_b.y - g_a.y) * 0.36f;
	dl->AddRectFilled({g_a.x - 7 * unit, my - hh}, {g_a.x - 4 * unit, my + hh}, Souls::C(Souls::kBrass, al), 2 * unit);
	dl->AddCircleFilled({g_a.x - 5.5f * unit, my}, 3.0f * unit, Souls::C(c, al), 10);
}
} // namespace Glide

// (des-upstream-1106) The F2 menu's entries: a dark plate, its icon in a gold disc, the label, a chevron; on hover / focus
// a glow, the accent bar, the icon a little larger and brighter, the label and icon sliding right (Anim::Hot, eased);
// on open each entry slides in after the one above (`index` x 40 ms). An ImGui button underneath: mouse, keyboard and
// pad navigation as before. `danger`: a red tint (Cerrar el juego).
bool IconButton(const char* label, Icons::Kind icon, ImVec2 size, float unit, int index, bool danger = false) {
	const float st = Anim::EaseOut(Anim::Progress(Anim::g_opened_at.load() + 0.04 * index, 0.26));
	if (st < 1.0f) Bump();
	ImGui::PushID(label);
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0, 0, 0, 0});
	const bool pressed = ImGui::Button("##entry", size);
	ImGui::PopStyleColor(3);
	const bool  on = ImGui::IsItemHovered() || ImGui::IsItemFocused() || ImGui::IsItemActive();
	const float v  = Anim::Hot(ImGui::GetItemID(), on);
	const bool  down = ImGui::IsItemActive();
	ImVec2      a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
	const float slide = (1.0f - st) * 28.0f * unit;
	a.x += slide, b.x += slide;
	auto*       dl = ImGui::GetWindowDrawList();
	const float al = ImGui::GetStyle().Alpha * st;
	const float r  = 8.0f * unit;
	const int   tr = danger ? 230 : 236, tg = danger ? 120 : 206, tb = danger ? 100 : 146; // accent
	const auto  A  = [&](float x) { return static_cast<int>(std::clamp(x, 0.0f, 255.0f) * al); };
	// (perfopts-1107) An engraved iron plate (darker at the foot, a light lip on top, a dark cut below), a tarnished
	// brass border that warms with the focus, the icon in a sigil medallion, the label etched; the gliding highlight
	// (Glide, drawn under the entries) marks the focused one.
	const auto Al = [&](float x) { return std::clamp(x, 0.0f, 1.0f) * al; };
	dl->AddRectFilledMultiColor(a, b, IM_COL32(30, 33, 39, A(232.0f)), IM_COL32(30, 33, 39, A(232.0f)), IM_COL32(16, 18, 22, A(240.0f)),
	                            IM_COL32(16, 18, 22, A(240.0f)));
	dl->AddLine({a.x + r, a.y + 1}, {b.x - r, a.y + 1}, IM_COL32(190, 200, 220, A(22.0f + 18.0f * v)), 1.0f);
	dl->AddLine({a.x + r, b.y - 1}, {b.x - r, b.y - 1}, IM_COL32(0, 0, 0, A(170.0f)), 1.0f);
	if (danger && v > 0.0f) dl->AddRectFilled(a, b, IM_COL32(120, 40, 30, A(50.0f * v)), r);
	if (down) dl->AddRectFilled(a, b, IM_COL32(tr, tg, tb, A(40.0f)), r);
	const ImU32 border = danger ? IM_COL32(tr, tg, tb, A(60.0f + 140.0f * v))
	                            : IM_COL32(Souls::kBrass[0], Souls::kBrass[1], Souls::kBrass[2], A(70.0f + 150.0f * v));
	dl->AddRect(a, b, border, r, 0, std::max(1.0f, 1.2f * unit));
	if (Glide::Want(on, danger)) Glide::Target(a, b);
	// The icon in its medallion (breathing a little while focused).
	const float  hgt = b.y - a.y;
	const float  dx  = 5.0f * unit * v;
	const ImVec2 ic {a.x + hgt * 0.62f + dx, (a.y + b.y) * 0.5f};
	// (step5) Focus: the icon grows 14 % (eased, real time) with a soft glow behind the medallion.
	const float  ve      = v * v * (3.0f - 2.0f * v);
	const float  breathe = (1.0f + 0.14f * ve) * (1.0f + 0.03f * ve * std::sin(Souls::Time() * 3.0f));
	for (int k = 5; k >= 1; k--) {
		const float rr = hgt * 0.36f * breathe * (1.0f + 0.16f * static_cast<float>(k));
		dl->AddCircleFilled(ic, rr, danger ? IM_COL32(230, 110, 80, A(14.0f * ve)) : IM_COL32(236, 206, 146, A(14.0f * ve)), 48);
	}
	Souls::Medallion(dl, ic, hgt * 0.36f * breathe, v, Al(1.0f), danger);
	const int lift = static_cast<int>(50.0f * v);
	const ImU32 ink = danger ? IM_COL32(std::min(255, tr + lift / 3), std::min(255, tg + lift), std::min(255, tb + lift), A(255.0f))
	                         : IM_COL32(std::min(255, Souls::kBrass[0] + lift), std::min(255, Souls::kBrass[1] + lift),
	                                    std::min(255, Souls::kBrass[2] + lift), A(255.0f));
	Icons::Draw(dl, icon, {ic.x + 1, ic.y + 1}, hgt * 0.36f * breathe, IM_COL32(0, 0, 0, A(150.0f)));
	Icons::Draw(dl, icon, ic, hgt * 0.36f * breathe, ink);
	// The label (etched), then a small brass chevron that comes with the focus.
	ImFont*      f  = ImGui::GetFont();
	const float  fs = ImGui::GetFontSize();
	const ImVec2 at {ic.x + hgt * 0.36f + 16.0f * unit + dx, (a.y + b.y - fs) * 0.5f};
	dl->AddText(f, fs, {at.x + 1, at.y + 1}, IM_COL32(0, 0, 0, A(170.0f)), label);
	dl->AddText(f, fs, at, IM_COL32(232 + static_cast<int>(16 * v), 226 + static_cast<int>(18 * v), 214 + static_cast<int>(26 * v), A(235.0f + 20.0f * v)),
	            label);
	if (v > 0.01f) {
		const float cx = b.x - 22.0f * unit + 6.0f * unit * v, cy = (a.y + b.y) * 0.5f, k = 7.0f * unit;
		const ImU32 cc = IM_COL32(Souls::kBrassHi[0], Souls::kBrassHi[1], Souls::kBrassHi[2], A(230.0f * v));
		dl->AddLine({cx - k * 0.5f, cy - k}, {cx + k * 0.5f, cy}, cc, 2.0f * unit);
		dl->AddLine({cx + k * 0.5f, cy}, {cx - k * 0.5f, cy + k}, cc, 2.0f * unit);
	}
	ImGui::PopID();
	return pressed;
}

// (trucos11) Icons::Kind -> the aged-metal icon of ds-style.h.
DsStyle::Icon StyleIcon(Icons::Kind k) {
	switch (k) {
		case Icons::Back: return DsStyle::Icon::Back;
		case Icons::Trophy: return DsStyle::Icon::Trophy;
		case Icons::Pad: return DsStyle::Icon::Pad;
		case Icons::Keyboard: return DsStyle::Icon::Keyboard;
		case Icons::Power: return DsStyle::Icon::Power;
		case Icons::Sword: return DsStyle::Icon::Sword;
		case Icons::Shield: return DsStyle::Icon::Shield;
		case Icons::Helm: return DsStyle::Icon::Helm;
		case Icons::Ring: return DsStyle::Icon::Ring;
		case Icons::Potion: return DsStyle::Icon::Potion;
		case Icons::Soul: return DsStyle::Icon::Soul;
		case Icons::Sun: return DsStyle::Icon::Sun;
		case Icons::Skull: return DsStyle::Icon::Skull;
		case Icons::Chest: return DsStyle::Icon::Chest;
		case Icons::Flame: return DsStyle::Icon::Flame;
		case Icons::Gear: return DsStyle::Icon::Gear;
		case Icons::Eye: return DsStyle::Icon::Eye;
		default: return DsStyle::Icon::Gear;
	}
}
// (trucos13) The ash over F2 and its pages: one system, stepped once per ImGui frame; it shies away from the mouse or
// the focused card and scatters when the menu opens or a page changes.
DsStyle::Ash g_ash;
ImVec2       g_ash_repel {-1e9f, -1e9f};
void AshFrame(ImDrawList* bg, float width, float height, float alpha) {
	static int    last_frame = -1;
	static double seen_open = 0, seen_page = 0;
	const int     frame     = ImGui::GetFrameCount();
	if (frame != last_frame) {
		last_frame = frame;
		const double o = Anim::g_opened_at.load(), pg = Anim::g_page_at.load();
		if (o != seen_open || pg != seen_page) {
			seen_open = o, seen_page = pg;
			g_ash.Burst({width * 0.5f, height * 0.55f}, 260.0f * std::min(width / 1920.0f, height / 1080.0f));
		}
		const ImVec2 m = ImGui::GetIO().MousePos;
		const ImVec2 r = ImGui::IsMousePosValid(&m) && ImGui::GetIO().MouseDelta.x * ImGui::GetIO().MouseDelta.x + ImGui::GetIO().MouseDelta.y * ImGui::GetIO().MouseDelta.y > 0.0f ? m : g_ash_repel;
		g_ash.Update(std::clamp(ImGui::GetIO().DeltaTime, 0.0f, 0.05f), Souls::Time(), r, width, height);
		g_ash.Draw(bg, alpha, Souls::Time());
	}
}
// (trucos11) The F2 card, Demon's Souls remake look (ds-style.h): a dark plate with a thin border; focused, a candle light
// breathing from inside, gold corner brackets; an aged-metal medallion; the label in tracked Cinzel capitals. Entry:
// a plain fade with a short rise (no bounce), staggered card by card.
bool CardButton(const char* label, Icons::Kind icon, ImVec2 size, float unit, int index, bool danger = false) {
	const float k   = Anim::Progress(Anim::g_opened_at.load() + 0.06 * index, 0.45);
	if (k < 1.0f) Bump();
	const float ent = 1.0f - (1.0f - k) * (1.0f - k) * (1.0f - k); // ease-out cubic
	ImGui::PushID(label);
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4 {0, 0, 0, 0});
	const bool pressed = ImGui::Button("##card", size);
	ImGui::PopStyleColor(4);
	const ImGuiID id = ImGui::GetItemID();
	const bool    on = ImGui::IsItemHovered() || ImGui::IsItemFocused() || ImGui::IsItemActive();
	const float   v  = Anim::Hot(id, on);
	if (v > 0.0f && v < 1.0f) Bump();
	Bump(); // (the candle light breathes)
	// (trucos12) Parallax: the cards drift a little opposite to the focused one (eased); the focused card rises with a
	// warm god-ray over it and a pulsing soul aura around it.
	static float s_focus = 0.0f, s_focus_target = 0.0f;
	if (on) s_focus_target = static_cast<float>(index);
	if (index == 0) s_focus += (s_focus_target - s_focus) * 0.15f;
	if (std::fabs(s_focus_target - s_focus) > 0.01f) Bump();
	const float  par  = std::round((index - s_focus) * -6.0f * unit);
	const ImVec2 ra = ImGui::GetItemRectMin(), rb = ImGui::GetItemRectMax();
	const float  rise = std::round((1.0f - ent) * 24.0f * unit - 18.0f * unit * v);
	const ImVec2 a {ra.x + par, ra.y + rise}, b {rb.x + par, rb.y + rise};
	auto*        dl = ImGui::GetWindowDrawList();
	const float  al = ImGui::GetStyle().Alpha * std::clamp(k * 1.4f, 0.0f, 1.0f);
	const float  t  = Souls::Time();
	DsStyle::GodRay(dl, a, b, v, al, t);
	DsStyle::SoulAura(dl, a, b, v, al, t);
	DsStyle::CandlePlate(dl, a, b, v, 0.0f, al, t + index, unit);
	if (danger && v > 0.01f) dl->AddRectFilled(a, b, IM_COL32(120, 30, 24, static_cast<int>(40 * v * al)));
	const float cw = b.x - a.x, ch = b.y - a.y;
	const ImVec2 mc {(a.x + b.x) * 0.5f, a.y + ch * 0.38f};
	const float  mr = std::round(std::min(cw * 0.31f, ch * 0.21f));
	DsStyle::Medallion(dl, StyleIcon(icon), mc, mr, al, v, t, 0.0f);
	DsStyle::Sparks(dl, mc, mr, v, al, t); // (trucos13)
	if (on) g_ash_repel = mc;
	ImFont*           f   = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
	const std::string up  = DsStyle::Upper(label);
	float             px  = std::round(22.0f * unit);
	const float       tr  = std::round(3.0f * unit);
	const float       maxw = cw - 24.0f * unit;
	if (const float w = DsStyle::TrackedWidth(f, px, up.c_str(), tr); w > maxw) px = std::max(12.0f, std::floor(px * maxw / w));
	const float lw = DsStyle::TrackedWidth(f, px, up.c_str(), tr);
	const ImU32 tc = danger ? IM_COL32(236, 150, 128, static_cast<int>(255 * al)) : (v > 0.5f ? DsStyle::C(DsStyle::kBone, al) : DsStyle::C(DsStyle::kGold, al));
	DsStyle::TrackedText(dl, f, px, {std::round(a.x + (cw - lw) * 0.5f + 1), std::round(a.y + ch * 0.74f + 1)}, IM_COL32(0, 0, 0, static_cast<int>(200 * al)), up.c_str(), tr);
	if (v > 0.3f) // (trucos12) bloom on the gold label
		for (int d = 0; d < 4; d++)
			DsStyle::TrackedText(dl, f, px, {std::round(a.x + (cw - lw) * 0.5f) + (d % 2 ? 2 : -2), std::round(a.y + ch * 0.74f) + (d / 2 ? 2 : -2)},
			                     DsStyle::C(DsStyle::kGoldHi, al * v * 0.12f), up.c_str(), tr);
	DsStyle::TrackedText(dl, f, px, {std::round(a.x + (cw - lw) * 0.5f), std::round(a.y + ch * 0.74f)}, tc, up.c_str(), tr);
	ImGui::PopID();
	return pressed;
}

void Section(const char* title) {
	ImGui::Spacing();
	ImGui::PushFont(DisplayFont(), 30.0f * g_unit); // (perfopts-1107) Cinzel, as the header (nullptr: the current font)
	// (des-upstream-1106) The section's icon in a soft gold disc before its title; the rule under it fades out to the right.
	if (const int icon = Icons::ForSection(title); icon >= 0) {
		const float  side = ImGui::GetFontSize();
		const ImVec2 at   = ImGui::GetCursorScreenPos();
		const ImVec2 c {at.x + side * 0.5f, at.y + side * 0.55f};
		auto*        dl = ImGui::GetWindowDrawList();
		const float  al = ImGui::GetStyle().Alpha;
		dl->AddCircleFilled(c, side * 0.62f, IM_COL32(222, 190, 120, static_cast<int>(34 * al)), 24);
		Icons::Draw(dl, static_cast<Icons::Kind>(icon), c, side * 0.66f, ImGui::GetColorU32(kGold));
		ImGui::Dummy({side * 1.15f, side});
		ImGui::SameLine();
	}
	ImGui::TextColored(kGold, "%s", I18n::T(title));
	ImGui::PopFont();
	const ImVec2 p = ImGui::GetCursorScreenPos();
	const float  rw = ImGui::GetContentRegionAvail().x;
	const ImU32  r0 = ImGui::GetColorU32(ImVec4 {0.87f, 0.77f, 0.55f, 0.55f}), r1 = ImGui::GetColorU32(ImVec4 {0.87f, 0.77f, 0.55f, 0.0f});
	ImGui::GetWindowDrawList()->AddRectFilledMultiColor(p, {p.x + rw, p.y + std::max(1.0f, g_unit)}, r0, r1, r1, r0);
	// (perfopts-1107) A small brass diamond at the rule's start (the filigree of the menu's dividers).
	const float d = 4.0f * g_unit;
	ImGui::GetWindowDrawList()->AddQuadFilled({p.x, p.y - d}, {p.x + d, p.y}, {p.x, p.y + d}, {p.x - d, p.y}, r0);
	ImGui::Spacing();
}

int FixedDlssGenerated(const std::string& fg) {
	if (fg == "dlss-x2") return 1;
	if (fg == "dlss-x3" || fg == "dlss-120") return 2;
	if (fg == "dlss-x4") return 3;
	if (fg == "dlss-x5") return 4;
	if (fg == "dlss-x6") return 5;
	return 0;
}

void ApplyLive(const Settings& s) {
	const bool auto_sharp = s.sharpness == "auto" || s.sharpness.empty();
	PortLive::sharpness   = auto_sharp ? -1.0f : static_cast<float>(std::atof(s.sharpness.c_str()));
	PortLive::fps_cap     = s.fps_cap;
	PortLive::rumble      = s.rumble ? 1 : 0;
	PortLive::rumble_gain = s.rumble_gain > 0 ? s.rumble_gain : 4.0f;
	const int level       = 0; // (ds-pc-port37) Metricas is the launcher's window (metricas.ps1); the in-game panel stays on F8
	for (int i = 0; i < 3 && PerfOverlay::Level() != level; i++) PerfOverlay::CycleLevel();
	PortLive::rt_off = LiveRtOff(g_rt.launched, s); // (10-28) RT passes off live
	PadMap::SetLive(s.pad_map);                      // (remap-f2) e.g. Restablecer + Aplicar
}

// (10-28) What needs a restart: as NeedsRestart, but the ray tracing turned off live does not.
bool NeedsRestartLive(const Settings& launched, const Settings& now) {
	Settings n   = now;
	const int off = LiveRtOff(launched, now);
	if ((off & 1) != 0) n.rt_reflections = launched.rt_reflections;
	if ((off & 2) != 0) n.rt_lamps = LampLevel(launched), n.rt_particles = launched.rt_particles;
	if ((off & 4) != 0) n.godrays = launched.godrays;
	if ((off & 8) != 0) n.ao = launched.ao;
	if ((off & 16) != 0) n.torch = launched.torch;
	if ((off & 32) != 0) n.lens_flare = launched.lens_flare;
	if ((off & 64) != 0) n.rt_shadows = launched.rt_shadows;
	if ((off & 1) != 0) n.rt_ss_refl = launched.rt_ss_refl;
	// (11-10) The mouse's sensitivity and anti-dead-zone apply live with keyboard and mouse on.
	if (launched.kbm && n.kbm) n.kbm_sens = launched.kbm_sens, n.kbm_antidz = launched.kbm_antidz;
	n.pad_map = launched.pad_map; // (remap-f2) the pad map applies live (HostPadMapSetLive)
	return NeedsRestart(launched, n);
}

void Save() {
	if (SaveSettings(g_rt.ini_path, g_edit)) {
		g_saved  = g_edit;
		g_status = "Guardado en settings.ini";
	} else {
		g_status = "No se pudo guardar settings.ini";
	}
}

// ---- (10-27) Tu equipo: Launcher-souls.ps1's TU EQUIPO card, and its two buttons "Actualizar version" and "Volver a la
// version anterior" (ds-update.cpp; the network and the copy on a worker thread, the menu shows where it is). ----

struct UpdateUi {
	enum State { Idle, Checking, UpToDate, Available, Working, Done, Failed, Choosing, Confirming } state = Idle;
	std::mutex                        mutex;
	std::string                       text;     // the message shown
	Update::Release                   release;
	std::vector<Update::Backup>       backups;
	int                               chosen = 0;
	std::atomic<int64_t>              done {0}, total {0};
	bool                              cancel = false;
	std::atomic<bool>                 busy {false};
} g_upd;

void SetUpd(UpdateUi::State state, const std::string& text) {
	std::scoped_lock lock(g_upd.mutex);
	g_upd.state = state;
	g_upd.text  = text;
	Bump();
}

void StartWorker(void (*work)()) {
	if (g_upd.busy.exchange(true)) return;
	std::thread([work] {
		work();
		g_upd.busy = false;
		Bump();
	}).detach();
}

std::filesystem::path PortRoot() {
	return g_rt.exe_path.parent_path();
}

void CheckWork() {
#if defined(_WIN32)
	Update::Release r;
	std::string     error;
	if (!Update::CheckLatest(&r, &error)) {
		SetUpd(UpdateUi::Failed, error);
		return;
	}
	{
		std::scoped_lock lock(g_upd.mutex);
		g_upd.release = r;
	}
	// (10-28, the emulator launcher) Both versions shown; an installed version with the same or a newer date (the 8-digit
	// YYYYMMDD in the tag) is never offered the published one (no downgrade).
	const std::string cur  = Update::CurrentVersion();
	const std::string both = "Version instalada: " + cur + "\nUltima version del port en GitHub: " + r.tag;
	const std::string dc = Update::TagDate(cur), dr = Update::TagDate(r.tag);
	// (11-23) A folder that knows its zip (version-zip.txt): up to date only with that very zip; a zip replaced in the same
	// release (same tag, another zip name) is offered.
	const std::string iz = Update::InstalledZip();
	if (!iz.empty() && iz != r.asset) SetUpd(UpdateUi::Available, both + "\nNueva version " + r.asset + " \xE2\x80\x94 Cruz para actualizar");
	else if (!iz.empty() || r.tag == cur) SetUpd(UpdateUi::UpToDate, both + "\nYa tienes la ultima version.");
	else if ((!dc.empty() && !dr.empty() && dc > dr) || (r.published.size() >= 19 && r.published.substr(0, 19) <= Update::BuildIso())) SetUpd(UpdateUi::UpToDate, both + "\nTu version es mas nueva. No hace falta actualizar.");
	else SetUpd(UpdateUi::Available, both + "\nNueva version " + r.tag + " \xE2\x80\x94 Cruz para actualizar");
#else
	SetUpd(UpdateUi::Failed, "Solo en Windows.");
#endif
}

void InstallWork() {
#if defined(_WIN32)
	Update::Release r;
	{
		std::scoped_lock lock(g_upd.mutex);
		r = g_upd.release;
	}
	g_upd.cancel = false;
	g_upd.done   = 0;
	g_upd.total  = r.size;
	SetUpd(UpdateUi::Working, "Descargando " + r.asset + "...");
	wchar_t temp[MAX_PATH + 1] = {};
	GetTempPathW(MAX_PATH, temp);
	std::string safe = r.tag;
	for (char& c: safe)
		if (std::strchr("\\/:*?\"<>|", c) != nullptr) c = '_';
	const std::filesystem::path zip = std::filesystem::path(temp) / std::filesystem::u8path("ds-update-" + safe + ".zip");
	std::string                 error;
	if (!Update::Download(r, zip, [](int64_t d, int64_t t) { g_upd.done = d; g_upd.total = t; Bump(); }, &g_upd.cancel, &error)) {
		SetUpd(UpdateUi::Failed, error + ". No se ha cambiado nada.");
		return;
	}
	SetUpd(UpdateUi::Working, "Instalando (copia de seguridad y archivos nuevos)...");
	const auto     root = PortRoot();
	Update::Result res;
	if (Update::Writable(root)) res = Update::InstallZip(root, zip, r.tag, g_rt.game_rel);
	else res = Update::RunElevated(g_rt.exe_path, {L"--update-helper", root.wstring(), zip.wstring(), std::filesystem::u8path(r.tag).wstring(), g_rt.game_rel.wstring()});
	std::error_code ec;
	std::filesystem::remove(zip, ec);
	// Restarted by itself (before, a line asked to close and open it again: missed, the update looked like it did nothing).
	if (res.ok) {
		SetUpd(UpdateUi::Done, "Actualizacion completada a la version " + r.tag +
		                           ". Reiniciando con la nueva version... Tus shaders compilados y partidas guardadas se mantienen.");
		std::this_thread::sleep_for(std::chrono::seconds(3));
		RequestRestart(/*to_launcher=*/true); // (ds-pc-port38) the new version opens at the launcher
	} else SetUpd(UpdateUi::Failed, res.error);
#endif
}

void RestoreWork() {
#if defined(_WIN32)
	Update::Backup b;
	{
		std::scoped_lock lock(g_upd.mutex);
		if (g_upd.chosen < 0 || g_upd.chosen >= static_cast<int>(g_upd.backups.size())) return;
		b = g_upd.backups[g_upd.chosen];
	}
	SetUpd(UpdateUi::Working, "Restaurando la version " + b.version + "...");
	const auto     root = PortRoot();
	Update::Result res;
	if (Update::Writable(root)) {
		Update::Options o;
		o.game_rel = g_rt.game_rel;
		res        = Update::Restore(root, b.name, Update::CurrentVersion(), o);
	} else {
		res = Update::RunElevated(g_rt.exe_path, {L"--restore-helper", root.wstring(), std::filesystem::u8path(b.name).wstring(),
		                                          std::filesystem::u8path(Update::CurrentVersion()).wstring(), g_rt.game_rel.wstring()});
	}
	if (res.ok) {
		SetUpd(UpdateUi::Done, "Restaurada la version " + b.version + ". Reiniciando...");
		std::this_thread::sleep_for(std::chrono::seconds(3));
		RequestRestart(/*to_launcher=*/true); // (ds-pc-port38) the new version opens at the launcher
	} else SetUpd(UpdateUi::Failed, res.error);
#endif
}

void OpenRestore() {
	std::scoped_lock lock(g_upd.mutex);
	g_upd.backups = Update::ListBackups(PortRoot());
	g_upd.chosen  = 0;
	g_upd.state   = g_upd.backups.empty() ? UpdateUi::Failed : UpdateUi::Choosing;
	g_upd.text    = g_upd.backups.empty() ? "No hay versiones anteriores guardadas" : "Elige la version a restaurar:";
	Bump();
}

void OpenCheck() {
	SetUpd(UpdateUi::Checking, "Buscando la ultima version en GitHub...");
	StartWorker(CheckWork);
}

// (update1) No check at start any more (the notice came back every time): the red "Actualizar versión" button above the
// game folder checks GitHub on demand and this window gives the answer (up to date / new version: Sí, No / error).
enum NoticeKind { NoticeNone, NoticeChecking, NoticeUpToDate, NoticeAvailable, NoticeError };
std::atomic<int> g_notice_kind {NoticeNone};
std::string       g_notice_text;

void ManualCheck() {
#if defined(_WIN32)
	if (g_upd.busy.load() || g_notice_kind.load() == NoticeChecking) return;
	g_notice_kind = NoticeChecking;
	std::thread([] {
		Update::Release r;
		std::string     error;
		if (!Update::CheckLatest(&r, &error)) {
			{
				std::scoped_lock lock(g_upd.mutex);
				g_notice_text = std::string(I18n::T("No se pudo comprobar: ")) + error;
			}
			g_notice_kind = NoticeError;
			Bump();
			return;
		}
		const std::string cur = Update::CurrentVersion();
		const std::string dc = Update::TagDate(cur), dr = Update::TagDate(r.tag);
		const std::string iz = Update::InstalledZip(); // (11-23) the zip this folder came from: only that one is "latest"
		const bool latest = !iz.empty() ? iz == r.asset
		                                : (r.tag == cur || (!dc.empty() && !dr.empty() && dc > dr) || (r.published.size() >= 19 && r.published.substr(0, 19) <= Update::BuildIso()));
		{
			std::scoped_lock lock(g_upd.mutex);
			if (latest) g_notice_text = I18n::T("Tienes la última versión.");
			else {
				g_upd.release = r;
				char line[512];
				std::snprintf(line, sizeof(line), I18n::T("Hay una versión nueva (%s). ¿Quieres actualizar?"), r.tag.c_str());
				g_notice_text = line;
			}
		}
		g_notice_kind = latest ? NoticeUpToDate : NoticeAvailable;
		Bump();
	}).detach();
#else
	g_notice_text = "Solo en Windows.";
	g_notice_kind = NoticeError;
#endif
}

// Returns true when "Sí" started the update (the caller opens the options, where its progress is shown).
bool UpdateNotice(float width, float height) {
	const int kind = g_notice_kind.load();
	if (kind == NoticeNone) return false;
	std::string text;
	{
		std::scoped_lock lock(g_upd.mutex);
		text = kind == NoticeChecking ? std::string(I18n::T("Buscando la última versión...")) : g_notice_text;
	}
	ImGui::SetNextWindowPos({width * 0.5f, height * 0.5f}, ImGuiCond_Always, {0.5f, 0.5f});
	ImGui::SetNextWindowSize({std::min(width - 32.0f, 620.0f * g_unit), 0});
	ImGui::Begin("##update-notice", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
	                                          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::SetWindowFocus();
	ImGui::PushTextWrapPos(0.0f);
	ImGui::TextUnformatted(text.c_str());
	ImGui::PopTextWrapPos();
	ImGui::Spacing();
	bool started = false;
	if (kind == NoticeChecking) {
		ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), {380 * g_unit, 10 * g_unit}, "");
	} else if (kind == NoticeAvailable) {
		if (ImGui::Button(I18n::T("Sí"), {160 * g_unit, 40 * g_unit}) && !g_upd.busy.load()) {
			g_notice_kind = NoticeNone;
			g_tab_request = 4;
			SetUpd(UpdateUi::Working, "Descargando...");
			StartWorker(InstallWork);
			started = true;
		}
		ImGui::SameLine();
		if (ImGui::Button(I18n::T("No##upd"), {160 * g_unit, 40 * g_unit}) || ImGui::IsKeyPressed(ImGuiKey_Escape)) g_notice_kind = NoticeNone;
	} else {
		if (ImGui::Button(I18n::T("Aceptar"), {160 * g_unit, 40 * g_unit}) || ImGui::IsKeyPressed(ImGuiKey_Escape)) g_notice_kind = NoticeNone;
	}
	ImGui::End();
	return started;
}

// (10-27b) The start menu's shader block: the static precompile's state, its progress bar and its button.
void DrawShaderBlock(float unit) {
	const auto v = Precompile::State();
	if (v.running && v.total > 0) {
		char label[64];
		std::snprintf(label, sizeof(label), "%d / %d", v.done, v.total);
		ImGui::ProgressBar(static_cast<float>(v.done) / static_cast<float>(v.total), {-1, 30 * unit}, label);
	} else if (v.running) {
		ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), {-1, 30 * unit}, "Preparando...");
	}
	ImGui::TextColored(v.failed ? kWarn : (v.finished ? kGold : kDim), "%s",
	                   v.text.empty() ? I18n::T("Shaders: la cache de esta grafica ya esta preparada.") : I18n::T(v.text.c_str()));
	if (v.running) {
		ImGui::SameLine();
		if (ImGui::SmallButton(I18n::T("Detener"))) std::thread([] { Precompile::Stop(); }).detach();
	} else if (!v.finished) {
		ImGui::SameLine();
		if (ImGui::SmallButton(I18n::T("Preparar shaders"))) StartShaders();
	}
}

void InfoLine(const char* caption, const std::string& value, const ImVec4& color) {
	ImGui::TextColored(kDim, "%s", I18n::T(caption));
	ImGui::PushStyleColor(ImGuiCol_Text, color);
	ImGui::TextWrapped("%s", value.c_str());
	ImGui::PopStyleColor();
	ImGui::Spacing();
}

// (F2 Trofeos) The trophies: from the trophy module when it has loaded them, else _SaveData/<title>/trophies-list.json
// and trophies.json. Earned first (grade colour), then the rest greyed; hidden unearned ones as "Trofeo oculto".
struct TrophyRow {
	int         id = -1, grade = 0;
	bool        hidden = false, earned = false;
	std::string name, detail;
};

std::vector<TrophyRow> LoadTrophyRows() {
	std::vector<TrophyRow> rows;
	std::vector<Libs::Trophy::Info> all;
	std::vector<int>                unlocked;
	if (Libs::Trophy::Snapshot(&all, &unlocked)) {
		const std::set<int> got(unlocked.begin(), unlocked.end());
		for (const auto& i: all) rows.push_back({i.id, i.grade, i.hidden, got.count(i.id) != 0, i.name, i.detail});
		return rows;
	}
	std::error_code       ec;
	std::filesystem::path list;
	for (const auto& d: std::filesystem::directory_iterator("_SaveData", ec)) {
		const auto f = d.path() / "trophies-list.json";
		if (std::filesystem::exists(f, ec) && (list.empty() || f < list)) list = f;
	}
	if (list.empty()) return rows;
	std::set<int> got;
	if (std::ifstream f(list.parent_path() / "trophies.json", std::ios::binary); f) {
		const auto j = nlohmann::json::parse(f, nullptr, false);
		if (j.is_object() && j.contains("unlocked") && j["unlocked"].is_object())
			for (const auto& [k, v]: j["unlocked"].items()) got.insert(std::atoi(k.c_str()));
	}
	if (std::ifstream f(list, std::ios::binary); f) {
		const auto j = nlohmann::json::parse(f, nullptr, false);
		if (j.is_object() && j.contains("trophies") && j["trophies"].is_array()) {
			for (const auto& t: j["trophies"]) {
				if (!t.is_object()) continue;
				TrophyRow r;
				r.id = t.value("id", -1);
				const std::string g = t.contains("grade") && t["grade"].is_string() ? t["grade"].get<std::string>() : "";
				r.grade  = g == "P" ? 1 : g == "G" ? 2 : g == "S" ? 3 : g == "B" ? 4 : 0;
				r.hidden = t.contains("hidden") && t["hidden"].is_boolean() && t["hidden"].get<bool>();
				r.name   = t.contains("name") && t["name"].is_string() ? t["name"].get<std::string>() : "";
				r.detail = t.contains("detail") && t["detail"].is_string() ? t["detail"].get<std::string>() : "";
				r.earned = got.count(r.id) != 0;
				rows.push_back(std::move(r));
			}
		}
	}
	return rows;
}

void DrawTrophyTab(float unit) {
	static std::vector<TrophyRow> rows;
	static double                 loaded_at = -100.0;
	if (ImGui::GetTime() - loaded_at > 2.0) {
		rows      = LoadTrophyRows();
		loaded_at = ImGui::GetTime();
		std::stable_sort(rows.begin(), rows.end(), [](const TrophyRow& a, const TrophyRow& b) { return a.earned && !b.earned; });
	}
	Section("Trofeos");
	if (rows.empty()) {
		ImGui::TextWrapped("%s", "Aún no hay datos de trofeos: se cargan al iniciar el juego.");
		return;
	}
	auto points = [](int g) { return g == 1 ? 300 : g == 2 ? 90 : g == 3 ? 30 : g == 4 ? 15 : 0; };
	int  total = 0, have = 0, earned = 0;
	for (const auto& r: rows) {
		total += points(r.grade);
		if (r.earned) {
			have += points(r.grade);
			earned++;
		}
	}
	const float frac = total > 0 ? static_cast<float>(have) / static_cast<float>(total) : 0.0f;
	ImGui::PushFont(nullptr, 30.0f * unit);
	ImGui::TextColored(kGold, "%d%% completado", static_cast<int>(frac * 100.0f));
	ImGui::PopFont();
	ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4 {0.80f, 0.63f, 0.30f, 1.0f});
	ImGui::ProgressBar(frac, {-1.0f, 22.0f * unit}, "");
	ImGui::PopStyleColor();
	ImGui::Text("%d / %d trofeos", earned, static_cast<int>(rows.size()));
	ImGui::Separator();
	static const ImVec4 grade_col[5] = {{0.6f, 0.6f, 0.6f, 1.0f}, {0.70f, 0.85f, 1.00f, 1.0f}, {1.00f, 0.82f, 0.30f, 1.0f},
	                                    {0.78f, 0.80f, 0.84f, 1.0f}, {0.80f, 0.52f, 0.30f, 1.0f}};
	static const char*  grade_txt[5] = {"", "Platino", "Oro", "Plata", "Bronce"};
	const ImVec4        grey {0.45f, 0.45f, 0.45f, 1.0f};
	ImGui::BeginChild("##trophies", {0, 0}, ImGuiChildFlags_None); Anim::SmoothScroll();
	for (const auto& r: rows) {
		ImGui::PushID(r.id);
		const int    g      = r.grade >= 1 && r.grade <= 4 ? r.grade : 0;
		const bool   secret = r.hidden && !r.earned;
		const ImVec2 at     = ImGui::GetCursorScreenPos();
		const float  w      = ImGui::GetContentRegionAvail().x;
		const float  h      = 64.0f * unit;
		ImGui::Selectable("##t", false, ImGuiSelectableFlags_None, {w, h}); // focusable: D-pad / arrows scroll the list
		if (ImGui::IsItemFocused() || ImGui::IsItemHovered()) {
			g_desc_title = secret ? "Trofeo oculto" : r.name;
			g_desc_text  = std::string(grade_txt[g]) + (r.earned ? " - obtenido" : " - sin obtener") + "\n\n" +
			              (secret ? "Se revelar\xC3\xA1 al obtenerlo." : r.detail);
		}
		ImDrawList* dl = ImGui::GetWindowDrawList();
		const float r0 = 12.0f * unit;
		// (10-28) The grade's cup next to an earned trophy (the popup's icon); a grey circle for the rest.
		if (r.earned) Libs::Trophy::DrawCup(dl, at.x + 4 * unit, at.y + h * 0.5f - 18 * unit, 36 * unit, Libs::Trophy::CupColor(g));
		else dl->AddCircleFilled({at.x + 8 * unit + r0, at.y + h * 0.5f}, r0, ImGui::GetColorU32(grey));
		const float tx = at.x + 16 * unit + 2 * r0 + 6 * unit;
		dl->AddText({tx, at.y + 4 * unit}, ImGui::GetColorU32(r.earned ? ImVec4 {1, 1, 1, 1} : grey),
		            secret ? "Trofeo oculto" : r.name.c_str());
		const std::string sub = secret ? std::string(grade_txt[g]) : std::string(grade_txt[g]) + " - " + r.detail;
		dl->PushClipRect(at, {at.x + w, at.y + h}, true);
		dl->AddText({tx, at.y + h * 0.5f + 2 * unit}, ImGui::GetColorU32(r.earned ? grade_col[g] : ImVec4 {0.35f, 0.35f, 0.35f, 1.0f}), sub.c_str());
		dl->PopClipRect();
		ImGui::PopID();
	}
	ImGui::EndChild();
}

void DrawPcTab(float unit) {
	const ImVec4 ink {0.93f, 0.91f, 0.88f, 1.0f};
	const ImVec4 red {0.92f, 0.35f, 0.31f, 1.0f};
	ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0.12f, 0.10f, 0.08f, 0.85f});
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 14 * unit);
	ImGui::BeginChild("##card", {0, 330 * unit}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
	ImGui::PushFont(nullptr, 30.0f * unit);
	ImGui::TextColored(kGold, "%s", I18n::T("TU EQUIPO"));
	ImGui::PopFont();
	const auto& pc  = g_rt.pc;
	std::string gpu = pc.gpu.empty() ? std::string("no detectada") : pc.gpu;
	if (pc.vram > 0) gpu += "  (" + (g_rt.launched.simulate_vram8 ? std::string("8 GB simulados") : std::to_string((pc.vram + (512ull << 20)) >> 30) + " GB") + ")";
	if (!pc.nvidia && !pc.gpu.empty()) gpu += "  (sin DLSS: solo NVIDIA)";
	InfoLine("Grafica", gpu, ink);
	InfoLine("Procesador y RAM", (pc.cpu.empty() ? std::string("?") : pc.cpu) + "  |  " + (g_rt.launched.simulate_ram16 ? std::string("16 GB RAM (simulados)") : std::to_string(pc.ram_gb) + " GB RAM"), ink);
	if (!pc.recommended.empty()) {
		InfoLine("Modo recomendado", pc.recommended, pc.unsupported ? red : kGold);
		InfoLine("Esperado (estimacion)", pc.expected, pc.unsupported ? red : ink);
	} else {
		InfoLine("Modo recomendado", "Grafica sin datos: empieza por el Modo Souls equilibrado y mira las metricas.", ink);
	}
	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopStyleColor();
	ImGui::Spacing();
	ImGui::TextColored(kDim, "Version instalada: %s", Update::CurrentVersion());

	UpdateUi::State state;
	std::string     text;
	{
		std::scoped_lock lock(g_upd.mutex);
		state = g_upd.state;
		text  = g_upd.text;
	}
	const bool  busy = g_upd.busy.load();
	const float bw = 330 * unit, bh = 52 * unit;
	ImGui::BeginDisabled(busy);
	// (10-28c) "Actualizar version" / "Volver a la version anterior" live only in the bottom bar (next to JUGAR / Salir);
	// here only the confirmation once a newer port version was found.
	if (state == UpdateUi::Available) {
		if (ImGui::Button(("Actualizar a " + g_upd.release.tag).c_str(), {bw, bh})) StartWorker(InstallWork);
	}
	ImGui::EndDisabled();
	// (10-27c) "Anadir a Steam": a non-Steam shortcut with its icon and artwork (ds-steam.cpp).
	{
		static std::mutex  steam_mutex;
		static std::string steam_text;
		static std::atomic<bool> steam_busy {false};
		ImGui::BeginDisabled(steam_busy.load());
		if (ImGui::Button("Añadir a Steam", {bw, bh})) {
			steam_busy = true;
			{
				std::scoped_lock lock(steam_mutex);
				steam_text = "Añadiendo a Steam...";
			}
			std::thread([] {
				std::string report;
				const bool  ok = Steam::RunAddToSteamScript(g_rt.exe_path, &report) || Steam::AddToSteam(g_rt.exe_path, &report);
				std::scoped_lock lock(steam_mutex);
				steam_text = (ok ? "OK: " : "ERROR: ") + report;
				steam_busy = false;
			}).detach();
		}
		ImGui::EndDisabled();
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Anade \"Demon's Souls\" a la biblioteca de Steam como juego externo (con icono e imagenes). Mejor con Steam cerrado; si esta abierto, reinicialo para que aparezca.");
		std::scoped_lock lock(steam_mutex);
		if (!steam_text.empty()) ImGui::TextWrapped("%s", steam_text.c_str());
		else ImGui::TextColored(kDim, "%s", "Steam debe cerrarse antes o reiniciarse después para que aparezca el acceso.");
	}
	if (!text.empty()) ImGui::TextWrapped("%s", text.c_str());
	if (state == UpdateUi::Working && g_upd.total.load() > 0 && text.rfind("Descargando", 0) == 0) {
		const float f = static_cast<float>(static_cast<double>(g_upd.done.load()) / static_cast<double>(g_upd.total.load()));
		char        label[64];
		std::snprintf(label, sizeof(label), "%.0f / %.0f MB", g_upd.done.load() / 1048576.0, g_upd.total.load() / 1048576.0);
		ImGui::ProgressBar(f, {-1, 0}, label);
		if (ImGui::Button(I18n::T("Cancelar"))) g_upd.cancel = true;
	}
	if (state == UpdateUi::Choosing || state == UpdateUi::Confirming) {
		std::scoped_lock lock(g_upd.mutex);
		for (int i = 0; i < static_cast<int>(g_upd.backups.size()); i++) {
			if (ImGui::Selectable(g_upd.backups[i].version.c_str(), g_upd.chosen == i)) {
				g_upd.chosen = i;
				g_upd.state  = UpdateUi::Confirming;
				g_upd.text   = "Se restaura la version " + g_upd.backups[i].version +
				             ". Los archivos actuales del motor se guardan antes en _backup_update (se puede deshacer). La cache de shaders, las partidas, los seeds y los ajustes no se tocan.";
			}
		}
	}
	if (state == UpdateUi::Confirming) {
		if (ImGui::Button(I18n::T("Restaurar"), {bw * 0.6f, bh})) StartWorker(RestoreWork);
		ImGui::SameLine();
		if (ImGui::Button(I18n::T("Cancelar##r"), {bw * 0.6f, bh})) SetUpd(UpdateUi::Idle, "");
	}
}

// (rtx1b) The resolution choices labelled with their height and "panorámica" when not 16:9 (2560x1080 sat next to
// 2560x1440 and was picked by mistake: a 1080p output), and in full screen only the modes the monitor offers (SDL's list
// of the first display; when SDL has none, every size as before).
std::string ResLabel(const std::string& id) {
	int w = 0, h = 0;
	if (std::sscanf(id.c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) return id;
	return id + " (" + std::to_string(h) + "p" + (w * 9 != h * 16 ? std::string(", ") + I18n::T("panorámica") : std::string()) + ")";
}
bool MonitorHasMode(const char* id) {
	int w = 0, h = 0;
	if (std::sscanf(id, "%dx%d", &w, &h) != 2) return true;
	const int n = SDL_GetNumDisplayModes(0);
	if (n <= 0) return true;
	for (int i = 0; i < n; ++i) {
		SDL_DisplayMode mode {};
		if (SDL_GetDisplayMode(0, i, &mode) == 0 && mode.w == w && mode.h == h) return true;
	}
	return false;
}

// (10-28) Resolucion (F2 > Gráficos > Imagen): full screen, the desktop's or a monitor mode; window, its size.
void ResolutionRow(Settings& s) {
	static const char* sizes[] = {"1280x720", "1600x900", "1920x1080", "2560x1080", "2560x1440", "3440x1440", "3840x1600", "3840x2160"};
	static std::vector<std::string> ids, labels;
	ids.clear();
	labels.clear();
	if (s.fullscreen) {
		ids.emplace_back("escritorio");
		labels.emplace_back(std::string(I18n::T("Escritorio")) +
		                    (g_rt.pc.screen_h > 0 ? " (" + std::to_string(g_rt.pc.screen_h) + "p)" : std::string()));
	}
	for (const char* z: sizes) {
		if (s.fullscreen && !MonitorHasMode(z)) continue; // (rtx1b)
		ids.emplace_back(z);
		labels.emplace_back(ResLabel(z));
	}
	std::string value = s.fullscreen ? s.resolution : std::to_string(s.width) + "x" + std::to_string(s.height);
	if (std::find(ids.begin(), ids.end(), value) == ids.end()) {
		ids.push_back(value);
		labels.push_back(ResLabel(value));
	}
	std::vector<Choice> res;
	for (size_t i = 0; i < ids.size(); i++) res.push_back({ids[i].c_str(), labels[i].c_str()});
	if (ChoiceRow("Resolucion", true, res.data(), static_cast<int>(res.size()), value,
	              s.fullscreen ? "Resolucion de salida en pantalla completa. Escritorio: la de Windows. Otra: el monitor cambia a ella mientras juegas."
	                           : "Tamano de la ventana.")) {
		if (s.fullscreen) {
			s.resolution = value;
		} else if (int w = 0, h = 0; std::sscanf(value.c_str(), "%dx%d", &w, &h) == 2) {
			s.width  = w;
			s.height = h;
		}
		// (10-29) A 4K output: Iluminacion IBL Completa + Teselado Alto; any other: Mitad + Medio (changeable after).
		SetQualityForOutput(&s, OutputHeight(s, g_rt.pc));
	}
	// (10-29) The game always starts in its performance mode (no 30 cap); at 4K with the cinematic quality.
	ImGui::TextColored(kDim, "%s", I18n::T(OutputHeight(s, g_rt.pc) >= 2160 ? "Cinemático sin tope 30" : "Modo rendimiento"));
}

// (10-28) The Configurador: the detected hardware, "Optimizar según tu equipo" (Optimize, ds-settings.cpp) and "Continuar".
// (10-28b) `in_tab`: inside the launcher's Sistema tab (no title, no "Continuar").
namespace ChatUi { // (chat1) below
bool ToastActive();
}
namespace Multiplayer { // (online5) below
void SettingsBlock(float unit);
void Card(float unit); // (online11) the launcher's / F2's "Conexión online" section
}
void DrawConfigurator(float unit, bool in_tab = false) {
	const ImVec4 ink {0.93f, 0.91f, 0.88f, 1.0f};
	const auto&  pc = g_rt.pc;
	if (!in_tab) {
		ImGui::PushFont(nullptr, 34.0f * unit);
		ImGui::TextColored(kGold, "%s", "Configurador");
		ImGui::PopFont();
	}
	ImGui::PushTextWrapPos(0.0f);
	ImGui::TextColored(kDim, "%s", "Hemos detectado tu equipo. Puedes optimizar los gráficos para él o seguir con tus ajustes actuales.");
	ImGui::PopTextWrapPos();
	ImGui::Spacing();
	ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0.12f, 0.10f, 0.08f, 0.85f});
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 14 * unit);
	ImGui::BeginChild("##hw", {0, 330 * unit}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
	std::string gpu = pc.gpu.empty() ? std::string("no detectada") : pc.gpu;
	if (pc.vram > 0) gpu += "  -  " + (g_rt.launched.simulate_vram8 ? std::string("8 GB de VRAM (simulados)") : std::to_string((pc.vram + (512ull << 20)) >> 30) + " GB de VRAM");
	InfoLine("Tarjeta gráfica", gpu, ink);
	std::string cpu = pc.cpu.empty() ? std::string("?") : pc.cpu;
	if (pc.cpu_cores > 0) cpu += "  -  " + std::to_string(pc.cpu_cores) + " núcleos / " + std::to_string(pc.cpu_threads) + " hilos";
	InfoLine("Procesador", cpu, ink);
	const bool low = g_rt.launched.simulate_ram16 || (pc.ram_gb > 0 && pc.ram_gb <= 20);
	InfoLine("Memoria RAM", (g_rt.launched.simulate_ram16 ? std::string("16 GB (simulados)") : std::to_string(pc.ram_gb) + " GB") + (low ? "  (se activan las mejoras para equipos de 16 GB de RAM)" : ""), low ? kWarn : ink);
	InfoLine("Monitor", pc.screen_w > 0 ? std::to_string(pc.screen_w) + "x" + std::to_string(pc.screen_h) + " a " + std::to_string(pc.refresh_hz) + " Hz" : std::string("?"), ink);
	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopStyleColor();
	if (pc.unsupported) ImGui::TextColored(kWarn, "%s", pc.expected.c_str());
	ImGui::Spacing();
	const float bw = 420 * unit, bh = 60 * unit;
	if (ImGui::Button("Optimizar según tu equipo", {bw, bh})) {
		g_cfg_why         = Optimize(pc, &g_edit);
		g_edit.configured = true;
		Save();
		if (!g_config_mode && !g_start_mode) ApplyLive(g_edit);
	}
	if (ImGui::IsItemHovered() || ImGui::IsItemFocused()) {
		g_desc_title = "Optimizar según tu equipo";
		g_desc_text  = "Elige los ajustes con mejor rendimiento para tu gráfica, procesador, RAM y monitor (objetivo: 60 FPS o más) y los guarda en settings.ini. Cambia gráficos, ray tracing y la opción de 16 GB de RAM; revisa las otras pestañas para ver el resultado.";
		if (!in_tab) ImGui::SetTooltip("%s", g_desc_text.c_str());
	}
	if (!in_tab) {
	ImGui::SameLine();
	if (ImGui::Button("Continuar", {bw * 0.6f, bh})) {
		if (!g_edit.configured) {
			g_edit.configured = true;
			Save();
		}
		g_cfg_open = false;
		if (g_config_mode) g_start_result = 1;
	}
	if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", g_cfg_why.empty() ? "Sigue con tus ajustes actuales." : "Sigue con los ajustes elegidos.");
	}
	if (!g_cfg_why.empty()) {
		ImGui::Spacing();
		ImGui::PushFont(nullptr, 30.0f * unit);
		ImGui::TextColored(kGold, "%s", "Hemos elegido:");
		ImGui::PopFont();
		ImGui::PushTextWrapPos(0.0f);
		for (const auto& line: g_cfg_why) ImGui::BulletText("%s", line.c_str());
		ImGui::PopTextWrapPos();
		ImGui::TextColored(kDim, "%s", g_start_mode ? "Guardado. Se aplica al pulsar JUGAR; en el juego puedes cambiarlo con F2."
		                               : g_config_mode ? "Después puedes cambiar cualquier opción en el juego con F2."
		                                              : "Guardado. Lo marcado \"(requiere reinicio)\" se aplica la próxima vez que inicies el juego.");
	}
}

// (ds-pc-port10) The DLSS 5 rows' tooltips (launcher and F2).
const char* const kDlss5Tip =
    "DLSS 5 (Neural Rendering de NVIDIA): mejora la imagen con la IA de NVIDIA despu\xC3\xA9s del reescalado. Solo NVIDIA RTX. Necesita tu propio archivo nvngx_dlssnr.dll junto a \"Demon's Souls.exe\" (no viene incluido): sin \xC3\xA9l se ve como siempre. Compatibilidad: RTX 5000 s\xC3\xAD; RTX 4000 (Ada Lovelace) con el parche, sin estar confirmado al 100%; RTX 3000 y anteriores no (no tienen FP8). Suave, Medio y Fuerte bajan la resoluci\xC3\xB3n en los bordes de la pantalla para ahorrar GPU (Suave un poco, Fuerte m\xC3\xA1s); el centro siempre va a resoluci\xC3\xB3n completa. Experimental.";
const char* const kDlss5StrengthTip =
    "Intensidad del efecto de DLSS 5: Baja, Media o Alta (la de NVIDIA, por defecto). M\xC3\xA1s baja: se parece m\xC3\xA1s a la imagen original del juego.";
const char* const kDlss5FileTip =
    "Qu\xC3\xA9 nvngx_dlssnr.dll puede cargar DLSS 5. Oficial: solo si est\xC3\xA1 firmado por NVIDIA (recomendado). No oficial: lo carga aunque no est\xC3\xA9 firmado (un archivo modificado); bajo tu responsabilidad.";

void Draw(float width, float height) {
	if (g_reset_edit.exchange(false)) {
		g_edit       = g_saved;
		g_focus_once = true;
		g_cfg_open   = false;
		g_cfg_why.clear();
		g_status.clear();
		auto& io = ImGui::GetIO();
		io.ClearInputKeys();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
	}
	DrainInputs();
	EnsureBackground();
	const float unit = std::max(std::min(width / 1920.0f, height / 1080.0f), 0.5f);
	auto*       bg   = ImGui::GetBackgroundDrawList();
	if (g_background != nullptr && g_bg_w > 0 && g_bg_h > 0) {
		// Cover the screen, centered.
		const float scale = std::max(width / g_bg_w, height / g_bg_h);
		const float w = g_bg_w * scale, h = g_bg_h * scale;
		bg->AddImage(g_background->GetTexRef(), {(width - w) * 0.5f, (height - h) * 0.5f}, {(width + w) * 0.5f, (height + h) * 0.5f},
		             {0, 0}, {1, 1}, IM_COL32(255, 255, 255, 235));
	}
	bg->AddRectFilledMultiColor({0, 0}, {width, height}, IM_COL32(0, 0, 0, 90), IM_COL32(0, 0, 0, 90), IM_COL32(0, 0, 0, 200),
	                            IM_COL32(0, 0, 0, 200));

	// (10-27b) Balanced by construction: whatever this frame pushes is popped back to these sizes at its end; a
	// release never shows Dear ImGui's error box.
	auto&     io_errors = ImGui::GetIO();
#if defined(NDEBUG)
	io_errors.ConfigErrorRecoveryEnableTooltip = false;
	io_errors.ConfigErrorRecoveryEnableAssert  = false;
#endif
	ImGuiContext& ctx = *ImGui::GetCurrentContext();
	const int colors_before = ctx.ColorStack.Size, vars_before = ctx.StyleVarStack.Size;
	ImFont* font = FindFont("segoeui");
	ImGui::PushFont(font, 28.0f * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {32 * unit, 24 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {12 * unit, 12 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {12 * unit, 6 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 14 * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10 * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 12 * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 10 * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 8 * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 8 * unit);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0.02f, 0.02f, 0.025f, 0.80f});
	ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0.87f, 0.77f, 0.55f, 0.45f});
	ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4 {0.12f, 0.11f, 0.10f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_TabSelected, ImVec4 {0.55f, 0.43f, 0.22f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4 {0.70f, 0.55f, 0.28f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_CheckMark, kGold);
	ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4 {0.14f, 0.13f, 0.12f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.30f, 0.24f, 0.13f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.55f, 0.43f, 0.22f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4 {0.55f, 0.43f, 0.22f, 0.8f});
	ImGui::PushStyleColor(ImGuiCol_NavCursor, kGold);
	const ImVec2 size = g_start_mode ? ImVec2 {std::min(1500 * unit, width - 32), height - 48 * unit}
	                                 : ImVec2 {std::min(1500 * unit, width - 32), std::min(940 * unit, height - 32)};
	ImGui::SetNextWindowPos({width * 0.5f, height * 0.5f}, ImGuiCond_Always, {0.5f, 0.5f});
	ImGui::SetNextWindowSize(size, ImGuiCond_Always);
	if (g_focus_once) {
		g_focus_once = false;
		ImGui::SetNextWindowFocus();
	}
	ImGui::Begin("##PortMenu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
	ImGui::PushFont(font, 40.0f * unit);
	ImGui::TextColored(kGold, "%s", g_start_mode ? "DEMON'S SOULS" : "AJUSTES");
	ImGui::PopFont();
	ImGui::SameLine();
	ImGui::SetCursorPosX(size.x - ImGui::CalcTextSize(g_rt.game_info.c_str()).x - 32 * unit);
	ImGui::TextColored(kDim, "%s", g_rt.game_info.c_str());
	ImGui::Separator();
	if (g_config_mode || g_cfg_open) {
		DrawConfigurator(unit);
	} else {

	Settings&  s       = g_edit;
	bool live_changed = false;
	// (10-27) The tabs are Launcher-souls.ps1's groups, in its order, with its labels and tooltips.
	// (10-27c) A game's settings menu: five tabs (LB/RB or Q/E), each the launcher's groups as sections; the rows on the
	// left, the focused row's description on the right.
	// (10-28b) The launcher (before the game): Imagen, Calidad gráfica, Ray tracing, Controles, Sistema (the Configurador
	// folded in) and Trofeos; the same sections and rows as F2, so both write the same settings.ini the same way.
	const char* game_tabs[kTabCount]     = {"Gráficos", "Pantalla", "Sonido", "Controles", "Sistema", "Trofeos", "Conexión online", "Ajustes del juego"};
	const char* launcher_tabs[kTabCount] = {"Imagen", "Calidad gráfica", "Ray tracing", "Controles", "Sistema", "Trofeos", "Conexión online", "Ajustes del juego"};
	const char* const* tabs = g_start_mode ? launcher_tabs : game_tabs;
	static const std::vector<int> game_groups[kTabCount] = {{0, 1, 2, 3}, {5}, {}, {7}, {6, 8}, {}, {11}, {10}}; // (10-28) no Modos Souls
	static const std::vector<int> launcher_groups[kTabCount] = {{5, 0, 1}, {2}, {3}, {7}, {9, 6, 8}, {}, {11}, {10}};
	const std::vector<int>* groups = g_start_mode ? launcher_groups : game_groups;
	static const char* section_names[12] = {"Imagen", "Frame generation", "Calidad gráfica", "Ray tracing", "Modos Souls",
	                                        "Pantalla", "Rendimiento", "Controles", "Tu equipo", "Configurador", "Ajustes del juego", "Conexión online"};
	g_unit = unit;
	const float footer  = (g_start_mode ? 250 : 150) * unit;
	const bool  nvidia  = g_rt.pc.nvidia;
	// (10-28) The sections as a vertical list on the left (no tab strip to scroll): mouse, keyboard / D-pad navigation
	// (A / Intro selects) and LB / RB or Q / E as before.
	if (g_tab_request >= 0 && g_tab_request < kTabCount) g_tab = g_tab_request;
	if (g_tab < 0 || g_tab >= kTabCount) g_tab = 0;
	{
		// (10-28c) The sections as one "Sección" dropdown (no tabs or side list): mouse, D-pad + A / Intro opens it,
		// B / Circle closes it (g_popup), LB / RB, Q / E and RePág / AvPág as before.
		ImGui::PushFont(nullptr, 22.0f * unit);
		ImGui::AlignTextToFramePadding();
		ImGui::TextColored(kGold, "%s", I18n::T("Sección"));
		ImGui::SameLine();
		ImGui::SetNextItemWidth(360.0f * unit);
		if (ImGui::BeginCombo("##section", I18n::T(tabs[g_tab]), ImGuiComboFlags_HeightLarge)) {
			for (int t = 0; t < kTabCount; t++) {
				const bool on = g_tab == t;
				if (ImGui::Selectable((std::string(I18n::T(tabs[t])) + "###sec" + std::to_string(t)).c_str(), on)) g_tab = t;
				if (on) ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		ImGui::PopFont();
		ImGui::Separator();
	}
	ImGui::BeginGroup();
	{
		for (int once = 0, t = g_tab; once < 1; once++) {
			const float page_h = size.y - footer - ImGui::GetCursorPosY();
			const float list_w = ImGui::GetContentRegionAvail().x * 0.62f;
			ImGui::BeginChild("##page", {list_w, page_h}, ImGuiChildFlags_None); Anim::SmoothScroll();
			const bool fixed = FixedFg(s);
			g_desc_title.clear();
			g_desc_text.clear();
			if (t == 5) DrawTrophyTab(unit);
			else if (groups[t].empty()) {
				Section("Sonido");
				ImGui::TextWrapped("%s", "El sonido se ajusta en el menú del propio juego (Opciones > Sonido). El port no añade opciones de sonido.");
			}
			for (const int section: groups[t]) {
			if (groups[t].size() > 1) Section(section_names[section]);
			switch (section) {
				case 0: { // Imagen
					ResolutionRow(s);
					static std::vector<ResMode> shown;
					shown.clear();
					// (10-28) FSR 4: the 3D at 360 lines at least (its model outputs at the display's size).
					int out_h = g_rt.pc.screen_h > 0 ? g_rt.pc.screen_h : 1440;
					if (int w = 0, h = 0; s.fullscreen && std::sscanf(s.resolution.c_str(), "%dx%d", &w, &h) == 2 && h > 0) out_h = h;
					else if (!s.fullscreen && s.height > 0) out_h = s.height;
					for (int i = 0; i < kResModeCount; i++) {
						if (!nvidia && std::strcmp(kResModes[i].upscaler, "dlss") == 0) continue;
						if (std::strcmp(kResModes[i].upscaler, "fsr4") == 0 && out_h / kResModes[i].value < 359 && s.res != kResModes[i].id) continue;
						shown.push_back(kResModes[i]);
					}
					ComboRow("Reescalado", true, static_cast<int>(shown.size()), shown.data(), s.res, [](const ResMode& m) { return m.id; },
					         [](const ResMode& m) { return m.label; },
					         "Reescalado: dibuja el 3D a menos resolucion y lo sube a la de la pantalla (mas fps). DLSS: NVIDIA RTX. XeSS, FSR 3.1 y FSR 4: todas las graficas (FSR 4: modelo de AMD con INT8, RTX 20 / RX 6000 o mas nuevas; el 3D a 360 lineas como minimo).");
					ImGui::BeginDisabled(!nvidia);
					ChoiceRow("Preset DLSS", true, kDlssPresets, kDlssPresetCount, s.dlss_preset,
					          "Modelo de DLSS: auto (el de NVIDIA), K = DLSS 4, L y M = DLSS 4.5 (mas nitidos, algo mas lentos)");
					ImGui::EndDisabled();
					if (ChoiceRow("Nitidez", (s.sharpness == "auto") != (g_rt.launched.sharpness == "auto"), kSharpness, kSharpnessCount, s.sharpness,
					              "Nitidez del reescalado (DLSS, XeSS, FSR 3.1 y FSR 4), aplicada a la resolucion de la pantalla. Mas alto: mas nitido, pero puede marcar bordes y ruido."))
						live_changed = true;
					// (ds-pc-port10) DLSS 5 (restart).
					ImGui::BeginDisabled(!nvidia);
					ChoiceRow("DLSS 5", s.dlss5 != g_rt.launched.dlss5, kDlss5Levels, kDlss5LevelCount, s.dlss5, kDlss5Tip);
					{
						std::string file = s.dlss5_unsigned ? "1" : "0";
						if (ChoiceRow("Archivo DLSS 5", s.dlss5_unsigned != g_rt.launched.dlss5_unsigned, kDlss5Files, kDlss5FileCount, file, kDlss5FileTip))
							s.dlss5_unsigned = file == "1";
					}
					ChoiceRow("Intensidad DLSS 5", s.dlss5_strength != g_rt.launched.dlss5_strength, kDlss5Strengths, kDlss5StrengthCount, s.dlss5_strength,
					          kDlss5StrengthTip);
					ImGui::EndDisabled();
					if (!nvidia) ImGui::TextColored(kDim, "Sin DLSS: tu grafica no es NVIDIA.");
					break;
				}
				case 1: { // Frame generation
					static std::vector<Choice> modes;
					modes.clear();
					for (const char* id: {"sin-fg", "amd-x2", "amd-x3", "amd-x4", "amd-dinamico", "dlss-x2", "dlss-x3", "dlss-x4", "dlss-dinamico"})
						if (nvidia || std::strncmp(id, "dlss", 4) != 0) modes.push_back({id, FrameGenLabel(id)});
					if (fixed)
						ImGui::TextColored(kWarn, "%s", s.souls_mode == "equilibrado" ? "Modo Souls equilibrado: frame generation fija en AMD x2 (45 reales, 90 en pantalla)."
						                                                              : s.souls_mode == "low" ? "Modo Souls low profile: frame generation fija en AMD x2 (30 reales, 60 en pantalla)." : s.souls_mode == "ligero-plus" ? "Modo Souls ligero +: frame generation fija en AMD x2 (40 reales, 80 en pantalla)." : "Modo Souls ligero: frame generation fija en AMD x2 (30 reales, 60 en pantalla).");
					ImGui::BeginDisabled(fixed);
					std::string locked = "amd-x2";
					ChoiceRow("Frame generation", true, modes.data(), static_cast<int>(modes.size()), fixed ? locked : s.framegen,
					          "AMD: todas las graficas (tambien con reescalado DLSS). DLSS: NVIDIA RTX 40/50");
					ImGui::EndDisabled();
					const std::string fg = fixed ? locked : s.framegen;
					if (fg.rfind("dlss", 0) == 0) {
						ChoiceRow("Anti-Lag", true, kAntiLagDlssg, kAntiLagDlssgCount, s.antilag_dlssg,
						          "Con la frame generation de NVIDIA (DLSS-G). Reflex: el de NVIDIA (recomendado). Anti-Lag / Ultra: el nuestro con Reflex apagado, para comparar cual va mejor.");
					} else if (fg.rfind("amd", 0) == 0) {
						std::string al = s.antilag_mode == "auto" || s.antilag_mode.empty() ? (g_rt.pc.amd ? "amd2" : "normal") : s.antilag_mode;
						std::string ultra = "ultra";
						ImGui::BeginDisabled(fixed);
						if (ChoiceRow("Anti-Lag", true, g_rt.pc.amd ? kAntiLagAmd : kAntiLagOther, g_rt.pc.amd ? kAntiLagAmdCount : kAntiLagOtherCount,
						              fixed ? ultra : al,
						              "Con la frame generation de AMD. Anti-Lag 2 (AMD): el de serie de las graficas AMD. Anti-Lag: el nuestro, como mucho 2 frames en cola antes de leer el mando. Ultra: 1 (aun menos retraso). En los modos Souls ligero, equilibrado y low profile va siempre el nuestro en Ultra."))
							s.antilag_mode = al;
						ImGui::EndDisabled();
					}
					const int max_gen = FixedFg(g_rt.launched) ? 0 : FixedDlssGenerated(g_rt.launched.framegen);
					if (max_gen > 1 && s.framegen == g_rt.launched.framegen) {
						if (g_fg_live == 0) g_fg_live = max_gen;
						ImGui::AlignTextToFramePadding();
						ImGui::TextUnformatted(I18n::T("Multiplicador (en vivo)"));
						ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.42f + ImGui::GetCursorStartPos().x);
						ImGui::SetNextItemWidth(-1);
						int shown = g_fg_live + 1;
						if (ImGui::SliderInt("##mult", &shown, 2, max_gen + 1, "x%d")) {
							g_fg_live             = shown - 1;
							PortLive::fg_generate = g_fg_live;
						}
					}
					ImGui::TextColored(kDim, "FPS desbloqueados siempre, salvo en los modos Souls de FPS fijos.");
					ImGui::TextColored(kDim, "El menu se muestra sin frames generados (no se arrastra con la imagen).");
					break;
				}
				case 2: { // Calidad grafica (the launcher's group, in its order)
					ChoiceRow("Texturas", true, kTextureReduce, kTextureReduceCount, s.texture_reduce, "Tamaño de las texturas: Medio = la mitad, Bajo = un cuarto. Lo que más VRAM ahorra.");
					ChoiceRow("Distancia", true, kQualities, kQualityCount, s.quality, "Distancia de dibujado: a menos nivel, los objetos lejanos pasan antes a modelos más simples. Más FPS si va justo el procesador.");
					ChoiceRow("Sombras", true, kShadows, kShadowCount, s.shadows, "Sombras: Medio = las del sol hasta 125 m, Bajo = hasta 60 m y sin sombras de focos ni ambientales. Más FPS, pero pueden aparecer de golpe.");
					ChoiceRow("Detalle", true, kDetails, kDetailCount, s.detail, "Detalle del juego: sombras (cascadas y distancia), iluminación de entorno y relieve (teselado). Ultra > Muy alto (modo cine) > Alto (modo rendimiento de PS5) > Normal > Bajo.");
					ResolveQualityAuto(&s, g_rt.pc);
					ChoiceRow("Iluminación IBL", true, kIbl, kIblCount, s.ibl, "Resolución de la iluminación de entorno (IBL) del juego: Completa = la del modo cine de PS5, más fina; Mitad = la del modo rendimiento. Con 4K empieza en Completa, con otras salidas en Mitad.");
					ChoiceRow("Teselado", true, kTess, kTessCount, s.tess, "Relieve por teselado del juego (suelos, muros, piedras): Bajo, Medio, Alto (el del modo cine de PS5), Ultra y Extremo (más relieve y más lejos; cuesta GPU). Con 4K empieza en Alto, con otras salidas en Medio.");
					ChoiceRow("Reflejos", true, kReflections, kReflectionCount, s.reflections, "Reflejos del propio juego (agua, charcos, suelos): más pasos y más alcance en Ultra. Cuesta algo de GPU.");
					CheckRow("Filtrado anisotrópico x16", true, s.af16, "Filtrado anisotrópico x16: suelos y paredes nítidos a lo lejos y en ángulo. Cuesta muy poco.");
					ChoiceRow("Nitidez de texturas", true, kTextureSharpness, kTextureSharpnessCount, s.texture_sharpness,
					          "Texturas: nítidas = el detalle de la resolución final con reescalado. Extra nítidas: un escalón más (lo que recomienda NVIDIA para DLSS); puede brillar algo en rejas y hierba. Cuesta pocos FPS.");
					ChoiceRow("Desvanecimiento de objetos", true, kReflections, kReflectionCount, s.object_fade, "Distancia a la que los objetos lejanos se desvanecen. Ultra: aparecen de golpe mucho menos.");
					ChoiceRow("Nitidez del AA", true, kAaSharp, kAaSharpCount, s.aa_sharp, "Nitidez del antialiasing del juego. No cuesta FPS.");
					CheckRow("Desenfoque de movimiento", true, s.motion_blur, "Desenfoque de movimiento del juego (Sí = como en PS5).");
					CheckRow("Profundidad de campo", true, s.dof, "Desenfoque de lo lejano y de lo cercano en escenas (Sí = como en PS5).");
					CheckRow("Viñeta", true, s.vignette, "Oscurecimiento de los bordes de la imagen (Sí = como en PS5).");
					CheckRow("Bloom", true, s.bloom, "Resplandor alrededor de las luces fuertes (Sí = como en PS5).");
					CheckRow("Adaptación de la vista", true, s.eye_adapt, "La exposición se adapta al pasar de zonas oscuras a claras, como el ojo (Sí = como en PS5).");
					ImGui::TextColored(kDim, "%s", "Son ajustes internos del juego: los lee al arrancar.");
					break;
				}
				case 3: { // Ray tracing (ds-torch-1028's launcher rows, in its order; hardware RT only with the experimental switch)
					const Settings& l       = g_rt.launched;
					const bool      live_ok = !l.pure_game;
					// (10-28) Turning a launched pass off is live (PortLive::rt_off); on, or another level, needs a restart.
					const auto changed_not_live = [&](bool changed, int bit) { return changed && !(live_ok && (LiveRtOff(l, s) & bit) != 0); };
					ImGui::BeginDisabled(s.pure_game);
					if (ChoiceRow("Reflejos RT", changed_not_live(l.rt_ss_refl != s.rt_ss_refl, 1),
					              kRtReflLevels, kRtReflLevelCount, s.rt_ss_refl,
					              "Reflejos trazados por rayos guiados por los del propio juego: solo donde el juego refleja y con su intensidad, más nítidos (agua como espejo). Donde el rayo no encuentra nada queda el reflejo del juego. Solo con DLSS, XeSS o FSR. Bajo ~0,6 ms, Medio ~1 ms, Alto ~2 ms, Ultra ~3 ms (rayos a resolución completa y más fuertes). Híbrido: Alto + rayos por hardware (RT real) solo donde el rayo de pantalla no encuentra nada. Necesita GPU con RT. Beta. Apagarlo es inmediato."))
						live_changed = true;
					// (refl-sharp-1031)
					ChoiceRow("Nitidez de reflejos RT", s.rt_refl_sharp != l.rt_refl_sharp, kRtReflSharp, kRtReflSharpCount, s.rt_refl_sharp,
					          "Más nitidez en los reflejos RT (superficies y agua). Normal: como ahora. Alta: rayos a resolución completa en Bajo/Medio, menos desenfoque y menos mezcla con frames anteriores (~+0,3-0,8 ms). Máxima: además sin tablero de ajedrez, el desenfoque mínimo y el historial más corto; los espejos conservan el detalle, puede verse algo más de ruido (~+0,6-1,5 ms).");
					ChoiceRow("Ondas del agua en reflejos", s.rt_refl_waves != l.rt_refl_waves, kRtReflWaves, kRtReflWavesCount, s.rt_refl_waves,
					          "Cuánto mueven las ondas del agua el rayo del reflejo (el agua se ve igual). Como ahora: sin cambios. Suaves: 40 % de la ondulación, reflejos más limpios. Espejo: el agua refleja como un espejo plano. Sin coste.");
					// (rtq2) The old ray-tracing rows are retired (Reflejos RT, Nitidez, Ondas, Luz RT, Luz volumetrica, Particulas
					// volumetricas, Reflejos estables; forced off in ds-settings.cpp). Only Suelo estable and Reflejos RT nuevos.
					if (ChoiceRow("Luz volumétrica", changed_not_live(l.godrays != s.godrays, 4), kVolLevels, kVolLevelCount, s.godrays,
					              "Luz volumétrica por rayos: rayos de sol que entran por ventanas y puertas, y halo de luz en la niebla alrededor de antorchas. Solo con DLSS, XeSS o FSR. Bajo ~0,4 ms, Medio ~0,8 ms, Alto ~1,2 ms. Ultra: volumen a media resolución (rayos más definidos), más muestras y la niebla del propio juego más densa y más dirigida hacia la luz (~+1 ms). Apagarla es inmediato."))
						live_changed = true;
					ChoiceRow("Partículas volumétricas", s.motes != l.motes, kMotesLevels, kMotesLevelCount, s.motes, "Motas de polvo en suspensión iluminadas por los rayos de sol y el halo de antorchas y lámparas (la luz volumétrica); flotan despacio y quedan fijas en el mundo. Sutil / Media / Densa: cuántas. Solo con DLSS, XeSS o FSR. ~0,2-0,5 ms."); // (motes-1031)
					// (ds-pc-port9 / ds-pc-port10) No Luz de antorchas RT or Destellos de lente in the port (the old screen-space "Path tracing" is gone; pt1 adds the hardware one above).
					ChoiceRow("Humo y partículas", s.smoke != l.smoke, kSmokeLevels, kSmokeLevelCount, s.smoke,
					          "Humo y partículas más realistas sobre las del propio juego: Suave = el humo, la niebla de partículas y las chispas ya no cortan en línea dura donde tocan suelo y paredes (se funden en ~35 cm). Solo con DLSS, XeSS o FSR. ~0,3 ms.");
					// (ds-pc-port9) No Sombras RT in the port.
					ImGui::EndDisabled();
					// (rtq2) No "RT por hardware (experimental)" / Reflejos (RT real) / Sombras sol (RT real): retired.
					if (CheckRow("Juego original (sin efectos añadidos)", s.pure_game != l.pure_game, s.pure_game,
					             "Apaga todos los efectos visuales que añade el emulador (ray tracing, luz volumétrica, humo, SSGI, reflejos RT). Se ve el juego tal cual. Rendimiento, reescalador, DLSS 5, frame generation y latencia no cambian."))
						live_changed = true;
					break;
				}
				case 5: { // Pantalla y otros
					{
						// (10-27c) As a game's options: the display mode, then the output resolution (full screen: the
						// desktop's or a mode of the monitor's; window: its size). Saved to settings.ini ([Pantalla]).
						static const Choice modes[] = {{"1", "Pantalla completa"}, {"0", "Ventana"}};
						std::string mode = s.fullscreen ? "1" : "0";
						if (ChoiceRow("Modo de pantalla", true, modes, 2, mode, "Pantalla completa o ventana (F11 lo cambia en vivo).")) s.fullscreen = mode == "1";
					}
					CheckRow("HDR", true, s.hdr, "HDR10: activa antes \"Usar HDR\" en la configuracion de pantalla de Windows");
					CheckRow("VSync", true, s.vsync, "VSync: sin cortes de imagen (tearing); con FreeSync/G-Sync sigue funcionando");
					if (ChoiceRow("Metricas", false, kMetrics, kMetricsCount, s.metrics, "Panel dentro del juego (F8 lo cambia en vivo): FPS, el modo de frame generation, latencia y el detalle. Ya no abre una ventana aparte.")) live_changed = true;
					static const Choice aspects[] = {{"16:9", "Monitor 16:9"}, {"21:9", "Monitor 21:9"}, {"32:9", "Monitor 32:9"}, {"48:9", "Monitor 48:9"}};
					ChoiceRow("Monitor", true, aspects, 4, s.aspect, "Monitores panoramicos (ultrawide): el juego se dibuja a 21:9, 32:9 o 48:9 en vez de estirarse. Funciona con DLSS, XeSS y FSR. Los videos y algunos menus se ven estirados.");
					CheckRow("Mantener 16:9 (bandas negras)", true, s.aspect_fit);
					{
						std::vector<Choice> langs;
						for (int i = 0; i < I18n::kMenuLanguageCount; i++) langs.push_back({I18n::kMenuLanguages[i], I18n::kMenuLanguageNames[i]});
						if (s.menu_language.empty()) s.menu_language = "es";
						if (ChoiceRow("Idioma del menu", false, langs.data(), static_cast<int>(langs.size()), s.menu_language)) {
							I18n::SetLanguage(s.menu_language);
							// (The game's language follows the menu's until it is chosen on its own.)
							if (!s.game_language_set) s.language = I18n::GameLanguageFor(s.menu_language);
						}
					}
					std::string lang = std::to_string(s.language);
					if (ChoiceRow("Idioma", true, kLanguages, kLanguageCount, lang)) {
						s.language          = std::atoi(lang.c_str());
						s.game_language_set = true;
					}
					CheckRow("Copia de las partidas al arrancar", false, s.backup_saves);
					ImGui::Spacing();
					ImGui::TextColored(kDim, "%s", I18n::T("Carpeta del juego:"));
					ImGui::TextWrapped("%s", s.game.c_str());
					ImGui::TextColored(kDim, "%s", g_start_mode ? "Se cambia abajo (Cambiar...)." : "Se cambia en el menu de inicio, antes del juego.");
					break;
				}
				case 10: { // (ds-pc-port37) Ajustes del juego: the emulator launcher's window, as a section of its own
					Section("Ajustes del juego");
						static const Choice levels[] = {{"0", "Predeterminado"}, {"1", "Bajo"}, {"2", "Medio"}, {"3", "Alto"}, {"4", "Ultra"}, {"5", "Extremo"}};
						static const Choice onoff[]  = {{"0", "Predeterminado"}, {"1", "No"}, {"2", "Sí"}};
						ImGui::TextWrapped("%s", I18n::T("Ajustes del propio juego. Predeterminado = como viene en PS5, sin tocar nada. Pasa el ratón por encima de cada uno para ver qué hace y lo que cuesta. Mandan sobre las opciones de Calidad gráfica."));
						size_t      n   = 0;
						const auto* all = GameSettings(&n);
						const char* sec = "";
						for (size_t i = 0; i < n; i++) {
							const GameSetting& g = all[i];
							if (std::strcmp(sec, g.section) != 0) {
								sec = g.section;
								ImGui::Spacing();
								ImGui::TextColored(kGold, "%s", I18n::T(sec));
							}
							std::string v = std::to_string(GameSettingLevel(s, g.id));
							ImGui::PushID(g.id);
							if (ChoiceRow(g.name, true, g.onoff ? onoff : levels, g.onoff ? 3 : 6, v, g.tip)) SetGameSettingLevel(&s, g.id, std::atoi(v.c_str()));
							ImGui::PopID();
						}
						if (ImGui::Button(I18n::T("Todo predeterminado"))) ResetGameSettings(&s);
										break;
				}
				case 9: DrawConfigurator(unit, true); break; // (10-28b) the launcher's Sistema: hardware and Optimizar
				case 6: { // Sistema
					if (!g_start_mode) {
					// (10-28) The Configurador again: the detected hardware and "Optimizar según tu equipo".
					if (ImGui::Button("Configurador", {420 * unit, 56 * unit})) {
						g_cfg_open = true;
						g_cfg_why.clear();
					}
					if (ImGui::IsItemHovered() || ImGui::IsItemFocused()) {
						g_desc_title = "Configurador";
						g_desc_text  = "Muestra tu gráfica, procesador, RAM y monitor, y elige los ajustes con mejor rendimiento para tu equipo.";
					}
					ImGui::Spacing();
					}
					{
						// (10-28b) The launcher before the game (settings.ini menu_inicio; --sin-lanzador skips it once).
						bool show = s.start_menu;
						if (CheckRow("Mostrar el lanzador al iniciar", false, show,
						             "Muestra el lanzador (estas opciones, Optimizar según tu equipo y JUGAR) antes del juego. No: el juego arranca directo; las opciones siguen en F2. \"Demon's Souls.exe\" --lanzador lo muestra una vez."))
							s.start_menu = show;
					}
					// (10-28) Mejoras para equipos de 16 GB de RAM: the launcher's 16 GB profile, on PCs of 20 GB or less.
					const bool small = s.simulate_ram16 || (g_rt.pc.ram_gb > 0 && g_rt.pc.ram_gb <= 20);
					(void)small; // (port28) always shown and usable; off by default
					s.profile16 = LowRam(s, g_rt.pc);
					CheckRow("Usar todos los CCD (Ryzen de dos CCD)", true, s.all_ccds, "Ryzen de dos CCD: No = el juego en el CCD con más caché L3 (más rápido); Sí = todos los núcleos.");
					// (perfopts-1107) Opciones de rendimiento (upstream's opt-in switches; at the next start).
					ImGui::Spacing();
					ImGui::Dummy({16.0f * g_unit, 0});
					ImGui::SameLine();
					ImGui::TextColored(kGold, "%s", I18n::T("Opciones de rendimiento"));
					// (11-22) Registros PM4 directos, Subidas en dos hilos, Copias anticipadas y agrupadas: always on in code (ds-settings.cpp), no boxes.
					CheckRow("El render ayuda con las subidas", s.perf_upload_help != g_rt.launched.perf_upload_help, s.perf_upload_help,
					         "Si la CPU va cargada, el hilo de render hace él mismo las subidas pendientes en vez de esperar. Bueno en CPUs de 4-6 núcleos y portátiles (KYTY_UPLOAD_HELP).");
					CheckRow("Hilos sin espera activa con CPU cargada", s.perf_spin_gate != g_rt.launched.perf_spin_gate, s.perf_spin_gate,
					         "Los hilos de ayuda dejan de girar en vacío cuando quedan menos de 3 núcleos libres, dejando la CPU al juego. Para CPUs justas (KYTY_SPIN_GATE).");
					CheckRow("Shaders lane-local ampliados", s.perf_lane_wide != g_rt.launched.perf_lane_wide, s.perf_lane_wide,
					         "Más shaders de cálculo se traducen en su forma rápida. Cambia esos shaders: la primera vez se vuelven a compilar (algo de carga al empezar). Experimental (KYTY_LANE_LOCAL_WIDE).");
					CheckRow("Carga de shaders al máximo", true, s.shader_turbo,
					         "La preparación de shaders al empezar la partida con todos los hilos de la CPU (en prioridad baja: el juego va primero). Mucho más rápida; mientras dura pueden bajar algo los FPS. No = pocos hilos, como antes (para portátiles o CPUs justas).");
					{
						// (ds-pc-port13) The shader precompile's CPU use (asked before the first one).
						bool all = s.precompile_threads == "todos";
						if (CheckRow("Precompilar shaders con todos los hilos de la CPU", false, all,
						             "Sí = máxima velocidad, el PC irá cargado; No = modo relajado. Se aplica la próxima vez que se preparen los shaders."))
							s.precompile_threads = all ? "todos" : "relajado", Precompile::SetAllThreads(all);
					}
					break;
				}
				case 7: { // Controles (the launcher's "Controles" and "Mapear mando")
					const Settings& l = g_rt.launched;
					CheckRow("Teclado y ratón", s.kbm != l.kbm, s.kbm,
					         "Jugar con teclado y ratón: WASD para moverse, el ratón mueve la cámara, clic izquierdo R1, clic derecho L1, rueda o clic central fija objetivo. F7 suelta o captura el ratón. Con él activo la cámara gira sin la aceleración del juego, también con el stick derecho del mando: si juegas con mando, déjalo desactivado.");
					ImGui::BeginDisabled(!s.kbm);
					const auto restart_note = [&](bool changed) {
						if (changed && !g_start_mode) ImGui::TextColored(ImVec4 {0.62f, 0.58f, 0.50f, 1.0f}, "%s", "   (requiere reinicio)");
					};
					// (11-10) Sensitivity and anti-dead-zone apply live (with keyboard and mouse on since the start).
					bool mouse_live = false;
					mouse_live |= SliderRow("Sensibilidad del ratón", s.kbm_sens, 0.05f, 4.0f, "%.2f", "Cuánto gira la cámara por cada movimiento del ratón (1.00 = normal). Se aplica al momento.");
					mouse_live |= SliderRow("Antizona muerta", s.kbm_antidz, 0.0f, 0.5f, "%.2f",
					          "Salta la zona muerta del stick del juego: con más, un movimiento pequeño del ratón ya gira la cámara. Si la cámara se mueve sola, bájalo. Se aplica al momento.");
					if (mouse_live && !g_start_mode && l.kbm) {
						const float base_sens = l.kbm_sens > 0 ? l.kbm_sens : 1.0f;
						Libs::Graphics::HostInputKbmSetLive((s.kbm_sens > 0 ? s.kbm_sens : 1.0f) / base_sens, s.kbm_antidz >= 0 ? s.kbm_antidz : 0.15f);
					}
					if (!l.kbm) restart_note(s.kbm_sens != l.kbm_sens || s.kbm_antidz != l.kbm_antidz);
					{
						float cam = static_cast<float>(s.kbm_camera);
						if (SliderRow("Velocidad de cámara (ratón)", cam, 5.0f, 15.0f, "%.0f",
						              "La velocidad máxima de giro de la cámara del juego con teclado y ratón. 10 = el máximo del menú del juego. 11-15 (extra, experimental): más rápido que el menú; si la cámara hace cosas raras, vuelve a 10. Sin teclado y ratón se usa la del menú del juego. Con el ratón en modo directo (por defecto, KYTY_KBM_DIRECT) ya no limita al ratón: el ratón gira la cámara exactamente su ángulo, sin tope; solo afecta al stick."))
							s.kbm_camera = std::clamp(static_cast<int>(std::lround(cam)), 5, 15);
						restart_note(s.kbm_camera != l.kbm_camera);
					}
					CheckRow("Cámara instantánea (frame warp)", s.kbm_warp != l.kbm_warp, s.kbm_warp,
					         "La imagen gira con el ratón antes de que el juego dibuje el siguiente fotograma: la cámara responde al instante.");
					{
						std::string cal = s.kbm_calibrate ? "1" : "0";
						static const Choice cal_items[] = {{"0", "No"}, {"1", "En el próximo inicio"}};
						if (ChoiceRow("Calibrar cámara (kbm_curve)", s.kbm_calibrate, cal_items, 2, cal,
						              "Mide cómo gira la cámara del juego para que el ratón la mueva exactamente lo que se mueve (se guarda en kbm_curve.txt; una vez basta). Al iniciar: carga tu partida, quédate quieto en un sitio abierto sin tocar el ratón ni el stick; a los 3 s la cámara gira sola unos 13 s. Necesita un reescalador o frame generation."))
							s.kbm_calibrate = cal == "1";
					}
					ImGui::EndDisabled();
					ImGui::Spacing();
					if (CheckRow("Vibración", false, s.rumble, "Vibración del mando (en vivo).")) live_changed = true;
					if (SliderRow("Fuerza de la vibración", s.rumble_gain, 1.0f, 8.0f, "%.1f", "Fuerza de la vibración del mando (en vivo).")) live_changed = true;
					ImGui::Spacing();
					if (ImGui::CollapsingHeader(I18n::T("Mapear mando"))) (void)PadMap::Draw(s, unit); // (remap-f2: saved and live at once)
					ImGui::TextColored(kDim, "%s", s.pad_map.empty() ? I18n::T("Mando: botones de serie.") : I18n::T("Mando: remapeado."));
					// (10-28) The key bindings (the launcher's "Mapear mando" > Teclado y ratón), read only.
					if (ImGui::CollapsingHeader("Teclas (teclado y ratón)")) {
						static const char* kDefault =
						    "LeftStickUp=W;LeftStickDown=S;LeftStickLeft=A;LeftStickRight=D;Walk=Left Ctrl;R1=Mouse:Left;L1=Mouse:Right;R2=Shift+Mouse:Left;L2=Shift+Mouse:Right;R3=Mouse:Middle;R3=Mouse:WheelUp;R3=Mouse:WheelDown;Circle=Space;Square=E;Cross=F;Triangle=G;Left=Q;Right=R;Up=1;Down=2;Left=3;Right=4;Options=Tab;TouchPad=Backspace";
						static const std::pair<const char*, const char*> names[] = {
						    {"LeftStickUp", "Avanzar"}, {"LeftStickDown", "Retroceder"}, {"LeftStickLeft", "Izquierda"}, {"LeftStickRight", "Derecha"},
						    {"Walk", "Caminar"}, {"R1", "R1 (ataque)"}, {"L1", "L1 (bloquear)"}, {"R2", "R2 (ataque fuerte)"}, {"L2", "L2 (parada)"},
						    {"R3", "R3 (fijar objetivo)"}, {"Circle", "Círculo (rodar / correr)"}, {"Square", "Cuadrado (objeto)"}, {"Cross", "Cruz (acción)"},
						    {"Triangle", "Triángulo (cambiar agarre)"}, {"Left", "Izquierda cruceta"}, {"Right", "Derecha cruceta"}, {"Up", "Arriba cruceta"},
						    {"Down", "Abajo cruceta"}, {"Options", "Menú"}, {"TouchPad", "Panel táctil"}};
						const std::string map = s.kbm_map.empty() ? std::string(kDefault) : s.kbm_map;
						if (ImGui::BeginTable("##keys", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
							size_t at = 0;
							while (at <= map.size()) {
								const size_t end = std::min(map.find(';', at), map.size());
								const std::string item = map.substr(at, end - at);
								at = end + 1;
								const size_t eq = item.find('=');
								if (eq == std::string::npos) continue;
								const std::string c = item.substr(0, eq);
								std::string        shown = c;
								for (const auto& [k, v]: names)
									if (c == k) shown = v;
								ImGui::TableNextRow();
								ImGui::TableSetColumnIndex(0);
								ImGui::TextUnformatted(shown.c_str());
								ImGui::TableSetColumnIndex(1);
								ImGui::TextColored(kGold, "%s", item.substr(eq + 1).c_str());
							}
							ImGui::EndTable();
						}
						ImGui::TextColored(kDim, "%s", "Las teclas se cambian en mapa_teclado (settings.ini) o con \"Mapear mando\" del lanzador.");
					}
					ImGui::TextColored(kDim, "%s", "F7: soltar o capturar el ratón.  F8: métricas.  F11 / Alt+Intro: pantalla completa.");
					break;
				}
				case 8: DrawPcTab(unit); break; // Tu equipo (Añadir a Steam)
				case 11: Multiplayer::Card(unit); break; // (online11) its own section in the launcher and F2
				default: break;
			}
			}
			ImGui::EndChild();
			// The description panel.
			ImGui::SameLine();
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0.06f, 0.055f, 0.05f, 0.75f});
			ImGui::BeginChild("##desc", {0, page_h}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding); Anim::SmoothScroll();
			ImGui::PushFont(nullptr, 30.0f * unit);
			ImGui::TextColored(kGold, "%s", g_desc_title.empty() ? tabs[t] : g_desc_title.c_str());
			ImGui::PopFont();
			ImGui::Separator();
			ImGui::PushTextWrapPos(0.0f);
			ImGui::TextUnformatted(g_desc_text.empty() && g_start_mode ? "Flechas / cruceta: elegir y cambiar el valor.\nA / Intro: abrir la lista y elegir.\nLB / RB o Q / E (o el desplegable Sección): cambiar de sección.\n\nTodo se guarda en settings.ini y se aplica al pulsar JUGAR. Dentro del juego, F2 abre estas mismas opciones."
			                       : g_desc_text.empty() ? "Flechas / cruceta: elegir y cambiar el valor.\nA / Intro: abrir la lista y elegir.\nLB / RB o Q / E (o el desplegable Sección): cambiar de sección.\nB / Esc: volver al juego.\n\n(requiere reinicio): se guarda ya y se aplica en el próximo inicio. El resto se aplica al momento."
			                                           : g_desc_text.c_str());
			ImGui::PopTextWrapPos();
			ImGui::EndChild();
			ImGui::PopStyleColor();
		}
	}
	ImGui::EndGroup();
	g_tab_request = -1;
	g_popup       = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
	if (live_changed) ApplyLive(s);

	ImGui::SetCursorPosY(size.y - footer + 8 * unit);
	ImGui::Separator();
	if (g_start_mode) {
		{
			const bool valid = !g_rt.is_game || g_rt.is_game(s.game);
			ImGui::AlignTextToFramePadding();
			ImGui::TextColored(kDim, "%s", I18n::T("Carpeta del juego:"));
			ImGui::SameLine();
			ImGui::TextColored(valid ? kGold : kWarn, "%s", s.game.empty() ? I18n::T("(sin elegir)") : s.game.c_str());
			ImGui::SameLine();
			if (ImGui::SmallButton(I18n::T("Cambiar...")) && g_rt.pick_folder) {
				const std::string picked = g_rt.pick_folder();
				if (!picked.empty() && (!g_rt.is_game || g_rt.is_game(picked))) {
					s.game = picked;
					Save();
				} else if (!picked.empty()) {
					g_status = "Esa carpeta no tiene eboot.bin ni sce_sys: elige la carpeta del juego volcado.";
				}
			}
			if (!valid) {
				ImGui::SameLine();
				ImGui::TextColored(kWarn, "%s", I18n::T("(no valida: JUGAR te la pide)"));
			}
		}
		DrawShaderBlock(unit);
		ImGui::TextColored(kDim, "%s", g_status.empty() ? I18n::T("Los cambios se aplican al pulsar JUGAR (sin reiniciar). F2 dentro del juego: opciones en vivo.") : g_status.c_str());
		const float bw = 300 * unit, bh = 60 * unit;
		const auto  v  = Precompile::State();
		if (ImGui::Button(I18n::T(v.running && !v.playable ? "JUGAR (sin esperar)" : "JUGAR"), {bw, bh})) {
			// (No game folder: the picker at once; a valid one chosen, saved and played.)
			bool ok = !g_rt.is_game || g_rt.is_game(s.game);
			if (!ok && g_rt.pick_folder) {
				const std::string picked = g_rt.pick_folder();
				if (!picked.empty() && g_rt.is_game(picked)) {
					s.game = picked;
					ok     = true;
				} else if (!picked.empty()) {
					g_status = "Esa carpeta no tiene eboot.bin ni sce_sys: elige la carpeta del juego volcado.";
				}
			}
			if (ok) {
				Save();
				g_start_result = 1;
			}
		}
		if (v.running && ImGui::IsItemHovered())
			ImGui::SetTooltip("Detiene la preparacion de shaders (lo hecho se guarda y sigue la proxima vez). Sin ella, el juego compila shaders al jugar (tirones).");
		ImGui::SameLine();
		ImGui::BeginDisabled(g_upd.busy.load());
		if (ImGui::Button(I18n::T("Actualizar version"), {bw, bh})) {
			g_tab_request = 4;
			OpenCheck();
		}
		ImGui::SameLine();
		if (ImGui::Button(I18n::T("Volver a la version anterior"), {bw, bh})) {
			g_tab_request = 4;
			OpenRestore();
		}
		ImGui::EndDisabled();
		ImGui::SameLine();
		if (ImGui::Button(I18n::T("Salir"), {bw * 0.6f, bh})) g_start_result = -1;
		ImGui::SameLine();
		{
			// (10-28b) "No mostrar al iniciar": settings.ini menu_inicio=0 (F2 > Sistema or --lanzador bring it back).
			bool hide = !s.start_menu;
			ImGui::AlignTextToFramePadding();
			if (ImGui::Checkbox(I18n::T("No mostrar al iniciar"), &hide)) {
				s.start_menu = !hide;
				Save();
			}
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", "El juego arrancará directo. Para volver a verlo: F2 > Sistema > Mostrar el lanzador al iniciar, o \"Demon's Souls.exe\" --lanzador.");
		}
	} else {
	const bool needs = NeedsRestartLive(g_rt.launched, s);
	if (needs) ImGui::TextColored(kWarn, "Hay cambios \"(requiere reinicio)\": se aplican la próxima vez que inicies el juego (o pulsa Reiniciar ahora).");
	else ImGui::TextColored(kDim, "%s", g_status.empty() ? "B / Círculo / Esc: volver al juego (sin aplicar).   LB / RB, Q / E: pestañas." : g_status.c_str());
	const float bw = 220 * unit, bh = 52 * unit;
	if (ImGui::Button("Aplicar", {bw, bh})) {
		Save();
		ApplyLive(s);
	}
	ImGui::SameLine();
	if (ImGui::Button("Restablecer", {bw, bh})) {
		// (The defaults, but the game folder and the languages stay.)
		Settings d;
		d.game              = s.game;
		d.menu_language     = s.menu_language;
		d.language          = s.language;
		d.game_language_set = s.game_language_set;
		d.configured        = true;
		s                   = d;
		g_status            = "Valores por defecto (pulsa Aplicar para guardarlos).";
	}
	ImGui::SameLine();
	if (ImGui::Button("Volver", {bw, bh})) {
		Save();
		ApplyLive(s);
		SetOpen(false);
	}
	ImGui::SameLine();
	ImGui::BeginDisabled(!needs);
	if (ImGui::Button(I18n::T("Reiniciar ahora"), {bw, bh})) {
		Save();
		RequestRestart();
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button(I18n::T("Salir del juego"), {bw, bh})) RequestQuit();
	}
	} // (10-28) not the Configurador
	ImGui::End();
	// (11 colours and 4 variables pushed above; 13 popped before 10-27b: Dear ImGui's "PopStyleColor() too many times".)
	if (ctx.ColorStack.Size > colors_before) ImGui::PopStyleColor(ctx.ColorStack.Size - colors_before);
	if (ctx.StyleVarStack.Size > vars_before) ImGui::PopStyleVar(ctx.StyleVarStack.Size - vars_before);
	ImGui::PopFont();
	Bump(); // (redrawn while open: hover, the nav cursor)
}

// ---- (ds-pc-port8) The launcher: Launcher-souls.ps1's window, natively. The whole screen with fondo-launcher.jpg
// (the logo stays visible on the left), the dark column of settings on the right (880 wide at 1080 lines) with its two
// columns of groups (Imagen, Calidad grafica, Ray tracing | Frame generation, Pantalla y otros, Configurador,
// Rendimiento / PC, JUGAR, Actualizar / Volver, Ajustes / Salir, Controles, TU EQUIPO, PayPal), the X left of the
// column, Trofeos at the bottom left, Ajustes as a panel left of the column. Every row writes the same settings.ini key
// as before (Settings), each one has its tooltip. Titles in a serif face, the rest Segoe UI. ----

ImFont*        g_font_body  = nullptr; // Segoe UI
ImFont*        g_font_semi  = nullptr; // Segoe UI Semibold
ImFont*        g_font_serif = nullptr; // Palatino Linotype / Georgia / Times
ImFont*        g_font_display = nullptr; // (ds-pc-port12) Cinzel, bundled (JUGAR and the section titles)
ImFont* DisplayFont() {
	return g_font_display;
}
ImTextureData* g_knight     = nullptr; // caballero-espada.png (TU EQUIPO)
int            g_knight_w = 0, g_knight_h = 0;
bool           g_knight_tried   = false;
bool           g_show_settings  = false;
bool           g_show_controls  = false;
bool           g_show_trophies  = false;
bool           g_show_keys      = false;
std::string    g_steam_text;
std::mutex     g_steam_mutex;
std::atomic<bool> g_steam_busy {false};

const ImVec4 kInkL {0.93f, 0.90f, 0.83f, 1.0f}; // (launcher1) old parchment (was near white)
const ImVec4 kDimL {0.86f, 0.84f, 0.80f, 1.0f}; // (ds-pc-port9) secondary text, still high contrast
const ImVec4 kGoldL {0.80f, 0.69f, 0.48f, 1.0f}; // (launcher1) tarnished gold (was brighter)

struct LBox {
	ImVec2 at {};
	float  w = 0;
	bool   open = false;
	bool   card = false;
} g_lbox;

float U(float v) {
	return v * g_unit;
}

ImTextureData* LoadTexture(const std::string& path, int* w, int* h) {
	int         channels = 0;
	std::string bytes;
	{
		std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
		bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	}
	auto* pixels = bytes.empty() ? nullptr
	                             : stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()), w, h,
	                                                     &channels, 4);
	if (pixels == nullptr) {
		std::printf("[port] launcher image %s not loaded\n", path.c_str());
		return nullptr;
	}
	auto* tex = IM_NEW(ImTextureData)();
	tex->Create(ImTextureFormat_RGBA32, *w, *h);
	std::memcpy(tex->GetPixels(), pixels, static_cast<size_t>(*w) * *h * 4);
	tex->UseColors = true;
	stbi_image_free(pixels);
	ImGui::RegisterUserTexture(tex);
	return tex;
}

void FreeTexture(ImTextureData*& tex) {
	if (tex == nullptr) return;
	ImGui::UnregisterUserTexture(tex);
	IM_DELETE(tex);
	tex = nullptr;
}

// (des-fog-hq) The launcher's ambience: slow volumetric-looking fog and drifting ash, ONLY in the pre-game launcher (its own
// SDL window and loop, RunStartMenu, before the game starts). Never drawn by the Configurador nor the in-game F2 menu.
// Everything is a pure function of the launcher's elapsed time (no per-frame state), two small textures generated once
// (a seamless 512x512 fBm with periodic lattices: no seams; linearly filtered, dithered alpha: no banding) and a soft dot;
// Shutdown() frees both when the launcher closes (before the renderer backend goes).
namespace Ambience {
ImTextureData* g_fog  = nullptr;
ImTextureData* g_dot  = nullptr;
bool           g_tried = false;
Uint64         g_t0    = 0;
constexpr int  kFog    = 512;
constexpr int  kDot    = 64;
constexpr int  kDust   = 240;
bool g_fog_linear = false, g_dot_linear = false;

uint32_t Hash(uint32_t x) {
	x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
	return x;
}
float Rand01(uint32_t x) {
	return static_cast<float>(Hash(x) & 0xFFFFFFu) / 16777216.0f;
}
// Value noise on a lattice of `period` cells that wraps (tileable), quintic interpolation (C2: no creases).
float Noise(float x, float y, int period, uint32_t seed) {
	const int   xi = static_cast<int>(std::floor(x)), yi = static_cast<int>(std::floor(y));
	const float fx = x - xi, fy = y - yi;
	const auto  q  = [](float t) { return t * t * t * (t * (t * 6 - 15) + 10); };
	const auto  v  = [&](int a, int b) {
        a = ((a % period) + period) % period;
        b = ((b % period) + period) % period;
        return Rand01(seed ^ (static_cast<uint32_t>(a) * 73856093u) ^ (static_cast<uint32_t>(b) * 19349663u));
	};
	const float u = q(fx), w = q(fy);
	const float a = v(xi, yi) + (v(xi + 1, yi) - v(xi, yi)) * u;
	const float b = v(xi, yi + 1) + (v(xi + 1, yi + 1) - v(xi, yi + 1)) * u;
	return a + (b - a) * w;
}

void Create() {
	g_tried = true;
	g_t0    = SDL_GetPerformanceCounter();
	std::vector<float> f(static_cast<size_t>(kFog) * kFog);
	float lo = 1e9f, hi = -1e9f;
	for (int y = 0; y < kFog; y++)
		for (int x = 0; x < kFog; x++) {
			float sum = 0, amp = 1, norm = 0;
			int   period = 4;
			for (int o = 0; o < 6; o++) {
				const float s = static_cast<float>(period) / kFog;
				sum += amp * Noise(x * s, y * s, period, 0x51u + o * 977u);
				norm += amp;
				amp *= 0.5f;
				period *= 2;
			}
			const float n = sum / norm;
			f[static_cast<size_t>(y) * kFog + x] = n;
			lo = std::min(lo, n);
			hi = std::max(hi, n);
		}
	g_fog = IM_NEW(ImTextureData)();
	g_fog->Create(ImTextureFormat_RGBA32, kFog, kFog);
	auto* px = static_cast<unsigned char*>(g_fog->GetPixels());
	for (int i = 0; i < kFog * kFog; i++) {
		float n = (f[i] - lo) / std::max(1e-4f, hi - lo);
		n       = std::clamp((n - 0.25f) / 0.75f, 0.0f, 1.0f);
		n       = n * n * (3 - 2 * n); // soft wisps, empty gaps
		// (triangular dither of +-1 LSB: the low alpha never bands)
		const float d = Rand01(0xD17Bu + i) + Rand01(0x9E37u + i * 7u) - 1.0f;
		px[i * 4 + 0] = px[i * 4 + 1] = px[i * 4 + 2] = 255;
		px[i * 4 + 3] = static_cast<unsigned char>(std::clamp(n * 255.0f + d, 0.0f, 255.0f));
	}
	g_fog->UseColors = true;
	ImGui::RegisterUserTexture(g_fog);
	g_dot = IM_NEW(ImTextureData)();
	g_dot->Create(ImTextureFormat_RGBA32, kDot, kDot);
	auto* dp = static_cast<unsigned char*>(g_dot->GetPixels());
	for (int y = 0; y < kDot; y++)
		for (int x = 0; x < kDot; x++) {
			const float dx = (x + 0.5f) / kDot * 2 - 1, dy = (y + 0.5f) / kDot * 2 - 1;
			const float r2 = dx * dx + dy * dy;
			const float a  = r2 >= 1 ? 0.0f : std::exp(-r2 * 4.0f) * (1 - r2);
			unsigned char* p = dp + (y * kDot + x) * 4;
			p[0] = p[1] = p[2] = 255;
			p[3] = static_cast<unsigned char>(a * 255.0f + 0.5f);
		}
	g_dot->UseColors = true;
	ImGui::RegisterUserTexture(g_dot);
}

void Shutdown() {
	FreeTexture(g_fog);
	FreeTexture(g_dot);
	g_tried = false;
	g_fog_linear = g_dot_linear = false;
}

void Linear(ImTextureData* t, bool* done) {
	// (the SDL_Renderer backend's texture: bilinear, never nearest; set once, as soon as the backend has created it)
	if (*done || t == nullptr || t->Status != ImTextureStatus_OK || t->GetTexID() == ImTextureID_Invalid) return;
	SDL_SetTextureScaleMode(reinterpret_cast<SDL_Texture*>(static_cast<intptr_t>(t->GetTexID())), SDL_ScaleModeLinear);
	*done = true;
}

// One fog layer: the tile (side `tile` px) scrolled by (ox, oy), split at the tile borders (UVs stay in 0..1, the texture
// wraps seamlessly) and in rows whose vertex alpha follows a vertical profile (denser low, thin at the top).
void Layer(ImDrawList* dl, float W, float H, float tile, float ox, float oy, ImVec4 rgb, float alpha) {
	const auto prof = [&](float y) {
		const float t = std::clamp(y / H, 0.0f, 1.0f);
		return alpha * (0.35f + 0.65f * t * t);
	};
	ox = std::fmod(ox, tile); if (ox < 0) ox += tile;
	oy = std::fmod(oy, tile); if (oy < 0) oy += tile;
	dl->PushTexture(g_fog->GetTexRef());
	constexpr int kRows = 6;
	for (float ty = -oy; ty < H; ty += tile)
		for (float tx = -ox; tx < W; tx += tile) {
			const float x0 = std::max(0.0f, tx), x1 = std::min(W, tx + tile);
			const float u0 = (x0 - tx) / tile, u1 = (x1 - tx) / tile;
			const float y0 = std::max(0.0f, ty), y1 = std::min(H, ty + tile);
			if (x1 <= x0 || y1 <= y0) continue;
			dl->PrimReserve(kRows * 6, (kRows + 1) * 2);
			const ImDrawIdx base = static_cast<ImDrawIdx>(dl->_VtxCurrentIdx);
			for (int r = 0; r <= kRows; r++) {
				const float y = y0 + (y1 - y0) * r / kRows, v = (y - ty) / tile;
				const ImU32 c = ImGui::ColorConvertFloat4ToU32({rgb.x, rgb.y, rgb.z, prof(y)});
				dl->PrimWriteVtx({x0, y}, {u0, v}, c);
				dl->PrimWriteVtx({x1, y}, {u1, v}, c);
			}
			for (int r = 0; r < kRows; r++) {
				const ImDrawIdx a = static_cast<ImDrawIdx>(base + r * 2);
				dl->PrimWriteIdx(a); dl->PrimWriteIdx(static_cast<ImDrawIdx>(a + 1)); dl->PrimWriteIdx(static_cast<ImDrawIdx>(a + 3));
				dl->PrimWriteIdx(a); dl->PrimWriteIdx(static_cast<ImDrawIdx>(a + 3)); dl->PrimWriteIdx(static_cast<ImDrawIdx>(a + 2));
			}
		}
	dl->PopTexture();
}

void Draw(ImDrawList* dl, float W, float H, float unit) {
	return; // (des-launcher-clean) the launcher's fog and ash are off: nothing is drawn here
	if (!g_tried) Create();
	if (g_fog == nullptr || g_dot == nullptr || W <= 0 || H <= 0) return;
	Linear(g_fog, &g_fog_linear);
	Linear(g_dot, &g_dot_linear);
	const double t  = static_cast<double>(SDL_GetPerformanceCounter() - g_t0) / static_cast<double>(SDL_GetPerformanceFrequency());
	const float  tf = static_cast<float>(std::fmod(t, 36000.0)); // (wraps every 10 h; well before float precision goes)
	// Fog: three layers, far = larger and slower (parallax), cold grey-blue, low alpha (the text stays readable).
	const float S = std::max(W, H);
	Layer(dl, W, H, S * 1.55f, tf * 6.0f, tf * 1.2f + 37.0f, {0.62f, 0.67f, 0.74f, 1.0f}, 0.10f);
	Layer(dl, W, H, S * 1.10f, tf * 11.0f + 400.0f, -tf * 1.8f, {0.68f, 0.72f, 0.78f, 1.0f}, 0.085f);
	Layer(dl, W, H, S * 0.80f, -tf * 17.0f + 900.0f, tf * 2.6f + 250.0f, {0.74f, 0.77f, 0.82f, 1.0f}, 0.06f);
	// Ash: depth z (0 far .. 1 near) sets size, speed, brightness and softness (the near ones are larger, blurrier, fainter).
	dl->PushTexture(g_dot->GetTexRef());
	for (int i = 0; i < kDust; i++) {
		const uint32_t s  = 0xA5u + i * 131u;
		const float    z  = Rand01(s);
		const float    sp = 0.25f + 0.75f * z;
		const float    x0 = Rand01(s + 1) * W, y0 = Rand01(s + 2) * H;
		const float    ph = Rand01(s + 3) * 6.2831853f;
		float          x  = x0 + tf * (6.0f + 10.0f * Rand01(s + 4)) * sp * unit + std::sin(tf * 0.23f + ph) * 18.0f * sp * unit;
		float          y  = y0 - tf * (4.0f + 9.0f * Rand01(s + 5)) * sp * unit + std::sin(tf * 0.31f + ph * 1.7f) * 10.0f * sp * unit;
		const float    m  = 20.0f * unit;
		x = std::fmod(x + m, W + 2 * m); if (x < 0) x += W + 2 * m; x -= m;
		y = std::fmod(y + m, H + 2 * m); if (y < 0) y += H + 2 * m; y -= m;
		const float r       = (1.6f + 7.0f * z * z) * unit;
		const float flicker = 0.92f + 0.08f * std::sin(tf * (0.25f + 0.35f * Rand01(s + 6)) + ph); // (slow, slight: no twinkle)
		const float a       = (0.10f + 0.30f * (1 - std::abs(z - 0.45f) * 1.4f)) * flicker;
		if (a <= 0.01f) continue;
		const bool  ember = Rand01(s + 7) < 0.08f;
		const ImU32 c     = ember ? ImGui::ColorConvertFloat4ToU32({1.0f, 0.70f, 0.45f, a * 0.8f})
		                          : ImGui::ColorConvertFloat4ToU32({0.82f, 0.84f, 0.88f, a});
		dl->PrimReserve(6, 4);
		dl->PrimRectUV({x - r, y - r}, {x + r, y + r}, {0, 0}, {1, 1}, c);
	}
	dl->PopTexture();
}
} // namespace Ambience
#include "port/ds-fog.inc"

// The tooltip of the last item (the launcher's ToolTip: on hover, after a short pause; also on disabled rows).
void Tip(const char* tip) {
	if (tip == nullptr || *tip == '\0') return;
	if (!ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) return;
	if (ImGui::BeginTooltip()) {
		ImGui::PushTextWrapPos(U(440));
		ImGui::TextUnformatted(I18n::T(tip));
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

// A group box (the launcher's GroupBox): its title in gold serif; the rows inside are placed from its left edge.
void BoxBegin(const char* title, float w = 420, bool card = false) {
	auto* dl = ImGui::GetWindowDrawList();
	dl->ChannelsSplit(2);
	dl->ChannelsSetCurrent(1);
	g_lbox.at   = ImGui::GetCursorScreenPos();
	g_lbox.w    = U(w);
	g_lbox.open = true;
	g_lbox.card = card;
	ImGui::BeginGroup();
	ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), g_lbox.at.y + U(card ? 12 : 6)});
	ImGui::PushFont(g_font_display, std::round(U(card ? 25 : 20))); // (launcher1) whole-pixel sizes: crisp
	ImGui::TextColored(kGoldL, "%s", I18n::T(title));
	{
		// (launcher1) A fine rule under the title, fading out to the right, with a small lozenge at its start.
		const ImVec2 tmin = ImGui::GetItemRectMin(), tmax = ImGui::GetItemRectMax();
		const float  ly   = std::floor(tmax.y + U(3)) + 0.5f, lx = std::floor(tmin.x);
		const float  lw   = g_lbox.w - U(24);
		const float  r    = std::round(U(2.5f));
		dl->AddQuadFilled({lx, ly - r}, {lx + r, ly}, {lx, ly + r}, {lx - r, ly}, IM_COL32(204, 176, 122, 200));
		dl->AddRectFilledMultiColor({lx + r + U(3), ly - 0.5f}, {lx + lw, ly + 0.5f}, IM_COL32(204, 176, 122, 150), IM_COL32(204, 176, 122, 0),
		                            IM_COL32(204, 176, 122, 0), IM_COL32(204, 176, 122, 150));
	}
	ImGui::PopFont();
	ImGui::SetCursorScreenPos({g_lbox.at.x, g_lbox.at.y + U(card ? 46 : 32)});
	ImGui::Dummy({g_lbox.w, 0});
}

void BoxEnd() {
	ImGui::Dummy({g_lbox.w, U(8)});
	ImGui::EndGroup();
	auto*        dl  = ImGui::GetWindowDrawList();
	const ImVec2 end = {g_lbox.at.x + g_lbox.w, ImGui::GetItemRectMax().y};
	dl->ChannelsSetCurrent(0);
	if (g_lbox.card) {
		dl->AddRectFilledMultiColor(g_lbox.at, end, IM_COL32(30, 26, 20, 215), IM_COL32(20, 19, 19, 215), IM_COL32(12, 13, 18, 215),
		                            IM_COL32(22, 20, 19, 215));
		dl->AddRect(g_lbox.at, end, IM_COL32(222, 196, 140, 150), U(14), 0, 1.5f);
	} else {
		dl->AddRectFilled(g_lbox.at, end, IM_COL32(9, 11, 15, 235), U(5));
		dl->AddRect(g_lbox.at, end, IM_COL32(222, 196, 140, 95), U(5), 0, 1.0f);
	}
	dl->ChannelsMerge();
	g_lbox.open = false;
	ImGui::SetCursorScreenPos({g_lbox.at.x, end.y + U(8)});
	ImGui::Dummy({g_lbox.w, 0});
}

// Rows: their top (screen y) and their end (the cursor under them, the full width taken).
float RowY() {
	return ImGui::GetCursorScreenPos().y;
}

void RowEnd(float y, float h = 32) {
	ImGui::SetCursorScreenPos({g_lbox.at.x, y});
	ImGui::Dummy({g_lbox.w, U(h)});
}

void Label(float y, const char* text, const char* tip, float x = 10) {
	ImGui::SetCursorScreenPos({g_lbox.at.x + U(x), y + U(4)});
	ImGui::TextUnformatted(I18n::T(text));
	Tip(tip);
}

// A drop-down field (the launcher's ComboBox): the value and an arrow; a click, Cross or Enter opens the list of values.
bool Field(const char* id, float y, int count, int& current, const std::function<const char*(int)>& list, const char* tip, float x, float w,
           bool brand = false) {
	if (count <= 0) return false;
	current = std::clamp(current, 0, count - 1);
	ImGui::PushID(id);
	const ImVec2 at {g_lbox.at.x + U(x), y + U(2)};
	const ImVec2 size {U(w), U(27)};
	ImGui::SetCursorScreenPos(at);
	const bool clicked  = ImGui::InvisibleButton("##f", size);
	const bool disabled = (ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled) != 0;
	const bool hot      = !disabled && (ImGui::IsItemHovered() || ImGui::IsItemFocused());
	Tip(tip);
	if (clicked && !disabled && count > 1) ImGui::OpenPopup("##list");
	auto*       dl    = ImGui::GetWindowDrawList();
	const float alpha = disabled ? 0.45f : 1.0f;
	dl->AddRectFilled(at, {at.x + size.x, at.y + size.y}, hot ? IM_COL32(40, 44, 58, 255) : IM_COL32(24, 26, 34, 255), U(3));
	dl->AddRect(at, {at.x + size.x, at.y + size.y}, hot ? ImGui::GetColorU32(kGoldL) : IM_COL32(90, 92, 96, 200), U(3));
	const char* shown = I18n::T(list(current));
	ImVec4      ink   = brand ? BrandColor(shown, kInkL) : kInkL;
	ink.w *= alpha;
	const float fs = ImGui::GetFontSize();
	dl->PushClipRect({at.x + U(6), at.y}, {at.x + size.x - U(22), at.y + size.y}, true);
	dl->AddText(nullptr, 0.0f, {at.x + U(8), at.y + (size.y - fs) * 0.5f}, ImGui::GetColorU32(ink), shown);
	dl->PopClipRect();
	const float cx = at.x + size.x - U(12), cy = at.y + size.y * 0.5f, s = U(4.5f);
	dl->AddTriangleFilled({cx - s, cy - s * 0.6f}, {cx + s, cy - s * 0.6f}, {cx, cy + s * 0.7f},
	                      ImGui::GetColorU32(ImVec4 {kGoldL.x, kGoldL.y, kGoldL.z, alpha}));
	const bool picked = DropList(count, current, list, at, size.x, size.y, std::max(size.x, U(250)), U(30), brand);
	ImGui::PopID();
	return picked;
}

template <typename Item, typename Id, typename Text>
bool LRow(const char* label, int count, const Item* items, std::string& value, Id id, Text text, const char* tip, float fx = 146,
          float fw = 282, bool brand = false, float lx = 10) {
	const float y = RowY();
	if (label != nullptr && *label != '\0' && fx > lx + 40) Label(y, label, tip, lx);
	int current = 0;
	for (int i = 0; i < count; i++)
		if (value == id(items[i])) current = i;
	const std::function<const char*(int)> list = [&](int i) -> const char* { return text(items[i]); };
	const bool                            ch   = Field(label, y, count, current, list, tip, fx, fw, brand);
	if (ch) value = id(items[current]);
	RowEnd(y);
	return ch;
}

bool LChoice(const char* label, const Choice* items, int count, std::string& value, const char* tip, float fx = 146, float fw = 260) {
	return LRow(label, count, items, value, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; }, tip, fx, fw);
}

// A check box at `x` of the row at `y` (several may share a row; the caller ends it).
bool LCheck(float y, const char* label, bool& v, const char* tip, float x = 12) {
	ImGui::SetCursorScreenPos({g_lbox.at.x + U(x), y + U(3)});
	ImGui::PushID(label);
	const bool changed = ImGui::Checkbox(I18n::T(label), &v);
	ImGui::PopID();
	Tip(tip);
	return changed;
}

bool LCheckRow(const char* label, bool& v, const char* tip) {
	const float y = RowY();
	const bool  c = LCheck(y, label, v, tip);
	RowEnd(y);
	return c;
}

// A slider with its text above it (the launcher's TrackBar), `w` wide at `x`.
bool LSlider(float y, const char* label, float& v, float lo, float hi, const char* format, const char* tip, float x, float w) {
	ImGui::SetCursorScreenPos({g_lbox.at.x + U(x), y + U(2)});
	ImGui::TextUnformatted(I18n::T(label));
	Tip(tip);
	ImGui::SetCursorScreenPos({g_lbox.at.x + U(x), y + U(24)});
	ImGui::SetNextItemWidth(U(w));
	ImGui::PushID(label);
	const bool changed = ImGui::SliderFloat("##s", &v, lo, hi, format);
	ImGui::PopID();
	Tip(tip);
	return changed;
}

// The launcher's buttons: the primary one gold with dark text, the others dark with a grey border.
// (launcher1) JUGAR in Demon's Souls' manner: a dark stone slab in a fine double frame of tarnished gold with notched
// corners and small lozenges, thin flourishes running out from the inscription, the inscription itself in Cinzel, the
// colour of old parchment, on whole pixels. Hover / focus: a soft warm halo breathes around the slab and the letters
// brighten (eased, ~110 ms, real time); pressed: the slab darkens and the inscription sinks one pixel. Only the drawing
// changed: the click, the focus and the tooltip are the same ImGui::Button as before.
void DrawSoulsButton(ImDrawList* dl, ImVec2 a, ImVec2 b, ImFont* font, float font_size, const char* t, float hot, bool held) {
	const float  u     = g_unit;
	const float  al    = ImGui::GetStyle().Alpha;
	const auto   C     = [al](int r, int g, int bl, float alpha) {
        return IM_COL32(r, g, bl, static_cast<int>(std::clamp(alpha, 0.0f, 1.0f) * 255.0f * al));
	};
	const float  px    = std::max(1.0f, std::round(u)); // a hairline, whole pixels
	a                  = {std::floor(a.x) + 0.5f, std::floor(a.y) + 0.5f};
	b                  = {std::floor(b.x) - 0.5f, std::floor(b.y) - 0.5f};
	// (a, b: the item on whole pixels, the hairlines centred on them)
	const float  cut   = std::round(7.0f * u); // the notched corners
	const ImVec2 mid   = {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
	// The halo: wide, faint, warm; grows with the hover (eased) and a slow breath while it lasts.
	if (hot > 0.0f) {
		const float breath = 0.85f + 0.15f * std::sin(static_cast<float>(Anim::Now()) * 2.4f);
		for (int k = 6; k >= 1; k--) {
			const float g = k * 3.0f * u * hot;
			dl->AddRectFilled({a.x - g, a.y - g}, {b.x + g, b.y + g}, C(214, 170, 92, 0.035f * hot * breath), cut + g);
		}
	}
	// The slab: a cold stone gradient, darker at the bottom; pressed: darker still.
	const float k_press = held ? 0.72f : 1.0f;
	const auto  stone   = [&](float v, float alpha) {
        const int c = static_cast<int>(v * k_press);
        return C(c + 4, c + 2, c, alpha);
	};
	ImVec2 poly[8] = {{a.x + cut, a.y}, {b.x - cut, a.y}, {b.x, a.y + cut}, {b.x, b.y - cut},
	                  {b.x - cut, b.y}, {a.x + cut, b.y}, {a.x, b.y - cut}, {a.x, a.y + cut}};
	dl->AddConvexPolyFilled(poly, 8, stone(18, 0.93f));
	dl->AddRectFilledMultiColor({a.x + cut * 0.5f, a.y + px}, {b.x - cut * 0.5f, mid.y}, stone(46 + 18 * hot, 0.55f), stone(46 + 18 * hot, 0.55f),
	                            stone(22, 0.0f), stone(22, 0.0f));
	// A warm wash rising from the bottom edge while hot (the light of a bonfire on the stone).
	if (hot > 0.0f)
		dl->AddRectFilledMultiColor({a.x + cut * 0.5f, mid.y}, {b.x - cut * 0.5f, b.y - px}, C(200, 150, 70, 0.0f), C(200, 150, 70, 0.0f),
		                            C(200, 150, 70, 0.16f * hot), C(200, 150, 70, 0.16f * hot));
	// The double frame: outer tarnished gold, inner fainter, both hairlines.
	const ImU32 gold_o = C(176 + static_cast<int>(40 * hot), 148 + static_cast<int>(36 * hot), 96 + static_cast<int>(22 * hot), 0.92f);
	const ImU32 gold_i = C(150, 126, 84, 0.45f + 0.25f * hot);
	dl->AddPolyline(poly, 8, gold_o, ImDrawFlags_Closed, px);
	const float in = std::round(4.0f * u);
	ImVec2      ip[8] = {{a.x + cut + in * 0.4f, a.y + in}, {b.x - cut - in * 0.4f, a.y + in}, {b.x - in, a.y + cut + in * 0.4f},
	                     {b.x - in, b.y - cut - in * 0.4f}, {b.x - cut - in * 0.4f, b.y - in}, {a.x + cut + in * 0.4f, b.y - in},
	                     {a.x + in, b.y - cut - in * 0.4f}, {a.x + in, a.y + cut + in * 0.4f}};
	dl->AddPolyline(ip, 8, gold_i, ImDrawFlags_Closed, px);
	// Lozenges at the middle of the long sides and at both ends.
	const auto lozenge = [&](ImVec2 c, float r, ImU32 col) {
		dl->AddQuadFilled({c.x, c.y - r}, {c.x + r, c.y}, {c.x, c.y + r}, {c.x - r, c.y}, col);
	};
	const float lr = std::round(3.5f * u);
	lozenge({mid.x, a.y}, lr, gold_o);
	lozenge({mid.x, b.y}, lr, gold_o);
	lozenge({a.x, mid.y}, lr, gold_o);
	lozenge({b.x, mid.y}, lr, gold_o);
	// The inscription: measured on whole pixels, flourishes out from it to the inner frame.
	const ImVec2 ts  = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, t);
	const float  sink = held ? px : 0.0f;
	const ImVec2 p   = {std::floor(mid.x - ts.x * 0.5f), std::floor(mid.y - ts.y * 0.5f) + sink};
	const float  gap = std::round(14.0f * u);
	const float  fy  = std::floor(mid.y) + 0.5f + sink;
	const float  run = std::max(0.0f, (p.x - gap) - (a.x + in + std::round(14.0f * u)));
	if (run > 4.0f * u) {
		const float l0 = p.x - gap - run, r0 = p.x + ts.x + gap;
		// Fading out towards the frame (two segments, the outer fainter) and a small lozenge at the inner end.
		dl->AddLine({l0, fy}, {l0 + run * 0.5f, fy}, C(170, 142, 92, 0.30f + 0.2f * hot), px);
		dl->AddLine({l0 + run * 0.5f, fy}, {p.x - gap, fy}, C(190, 160, 104, 0.65f + 0.3f * hot), px);
		dl->AddLine({r0 + run * 0.5f, fy}, {r0 + run, fy}, C(170, 142, 92, 0.30f + 0.2f * hot), px);
		dl->AddLine({r0, fy}, {r0 + run * 0.5f, fy}, C(190, 160, 104, 0.65f + 0.3f * hot), px);
		lozenge({p.x - gap + lr * 0.5f, fy}, lr * 0.8f, gold_o);
		lozenge({r0 - lr * 0.5f, fy}, lr * 0.8f, gold_o);
	}
	// A dark bed under the letters (legible over the stone), then the parchment letters; hot: warmer and a faint glow.
	dl->AddText(font, font_size, {p.x, p.y + px}, C(0, 0, 0, 0.85f), t);
	if (hot > 0.0f) {
		const float o = std::round(1.0f * u);
		for (int dx = -1; dx <= 1; dx++)
			for (int dy = -1; dy <= 1; dy++)
				if (dx != 0 || dy != 0) dl->AddText(font, font_size, {p.x + dx * o, p.y + dy * o}, C(230, 180, 96, 0.10f * hot), t);
	}
	dl->AddText(font, font_size, p, C(218 + static_cast<int>(26 * hot), 206 + static_cast<int>(22 * hot), 178 + static_cast<int>(10 * hot), 1.0f), t);
}

bool LButton(const char* text, ImVec2 at, ImVec2 size, bool primary, ImFont* font, float font_size, const char* tip = nullptr) {
	ImGui::SetCursorScreenPos(at);
	// (launcher1) The primary button in the display face is drawn by DrawSoulsButton: ImGui's own frame transparent.
	const bool souls = primary && font == g_font_display && font != nullptr;
	if (souls) {
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0, 0, 0, 0});
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0, 0, 0, 0});
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0, 0, 0, 0});
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4 {0, 0, 0, 0});
		ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0, 0, 0, 0});
	} else if (primary) {
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.69f, 0.55f, 0.27f, 0.92f});
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.81f, 0.67f, 0.38f, 0.96f});
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0.60f, 0.47f, 0.22f, 1.0f});
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4 {0.08f, 0.06f, 0.04f, 1.0f});
		ImGui::PushStyleColor(ImGuiCol_Border, kGoldL);
	} else {
		// (launcher1) Stone grey and tarnished gold (was blue-grey).
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.075f, 0.072f, 0.068f, 0.84f});
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.15f, 0.135f, 0.115f, 0.90f});
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0.21f, 0.19f, 0.15f, 1.0f});
		ImGui::PushStyleColor(ImGuiCol_Text, kInkL);
		ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0.50f, 0.44f, 0.33f, 0.85f});
	}
	ImGui::PushFont(font, font_size);
	const bool clicked = ImGui::Button(I18n::T(text), size);
	if (souls) {
		const bool on = ImGui::IsItemHovered() || ImGui::IsItemFocused() || ImGui::IsItemActive();
		DrawSoulsButton(ImGui::GetWindowDrawList(), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), font, font_size, I18n::T(text),
		                Anim::Hot(ImGui::GetItemID(), on), ImGui::IsItemActive());
	} else {
		Anim::Highlight(g_unit); // (step5) eased hover / focus glow, border and accent bar (real time)
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(5);
	Tip(tip);
	return clicked;
}

// (11-13) Archivo de paginacion de hasta 100 GB en C: (PowerShell como administrador: Windows pide permiso). Evita que
// un PC con poca RAM llegue al limite de memoria comprometida (un PC de 16 GB se reinicio ahi). Hace falta reiniciar.
void SetPagefile(int gb) {
#if defined(_WIN32)
	// gb > 0: a fixed page file of that size on C:, written where Windows keeps it (the PagingFiles registry value)
	// with automatic management off. gb == 0: back to Windows' default (system managed). Applied at the next restart.
	std::string script = "@echo off\r\n";
	if (gb > 0) {
		const std::string mb = std::to_string(static_cast<long long>(gb) * 1024);
		script += "echo Archivo de paginacion fijo de " + std::to_string(gb) + " GB en C:\r\n"
		          "wmic computersystem where name=\"%computername%\" set AutomaticManagedPagefile=False >nul 2>&1\r\n"
		          "powershell -NoProfile -ExecutionPolicy Bypass -Command \"try { $c=Get-CimInstance Win32_ComputerSystem; "
		          "Set-CimInstance -InputObject $c -Property @{AutomaticManagedPagefile=$false} } catch {}\" >nul 2>&1\r\n"
		          "reg add \"HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management\" /v PagingFiles "
		          "/t REG_MULTI_SZ /d \"C:\\pagefile.sys " + mb + " " + mb + "\" /f\r\n"
		          "echo.\r\n"
		          "reg query \"HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management\" /v PagingFiles\r\n"
		          "echo.\r\n"
		          "echo Si arriba pone C:\\pagefile.sys " + mb + " " + mb + " esta hecho. REINICIA EL PC para aplicarlo.\r\n";
	} else {
		script += "echo Archivo de paginacion: el de Windows por defecto (lo gestiona Windows)\r\n"
		          "wmic computersystem where name=\"%computername%\" set AutomaticManagedPagefile=True >nul 2>&1\r\n"
		          "powershell -NoProfile -ExecutionPolicy Bypass -Command \"try { $c=Get-CimInstance Win32_ComputerSystem; "
		          "Set-CimInstance -InputObject $c -Property @{AutomaticManagedPagefile=$true} } catch {}\" >nul 2>&1\r\n"
		          "reg add \"HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management\" /v PagingFiles "
		          "/t REG_MULTI_SZ /d \"?:\\pagefile.sys\" /f\r\n"
		          "echo.\r\n"
		          "reg query \"HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management\" /v PagingFiles\r\n"
		          "echo.\r\n"
		          "echo Si arriba pone ?:\\pagefile.sys esta hecho. REINICIA EL PC para aplicarlo.\r\n";
	}
	script += "pause\r\n";
	char dir[MAX_PATH] = {};
	if (GetTempPathA(MAX_PATH, dir) == 0) return;
	const std::string path = std::string(dir) + "kyty-pagefile.cmd";
	if (FILE* f = std::fopen(path.c_str(), "wb"); f != nullptr) {
		std::fwrite(script.data(), 1, script.size(), f);
		std::fclose(f);
		// "runas": Windows asks for administrator permission (UAC) and runs the script elevated.
		const std::string args = "/c \"" + path + "\"";
		ShellExecuteA(nullptr, "runas", "cmd.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
	}
#else
	(void)gb;
#endif
}

void SetBigPagefile() { SetPagefile(60); }

// Whether the 60 GB page file is already written in the registry (then only a restart is missing).
static bool PagefileConfigured() {
#if defined(_WIN32)
	char  value[1024] = {};
	DWORD size        = sizeof(value) - 2;
	if (RegGetValueA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management",
	                 "PagingFiles", RRF_RT_REG_MULTI_SZ, nullptr, value, &size) == ERROR_SUCCESS) {
		return std::strstr(value, "C:\\pagefile.sys ") != nullptr && std::strstr(value, "C:\\pagefile.sys 0 0") == nullptr;
	}
#endif
	return false;
}

// The system commit limit (RAM + page file) in GB, 0 when unknown: shown under the page file button.
static uint64_t CommitLimitGb() {
#if defined(_WIN32)
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (GlobalMemoryStatusEx(&status) != 0) return status.ullTotalPageFile >> 30u;
#endif
	return 0;
}

void OpenUrl(const char* url) {
#if defined(_WIN32)
	ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#else
	(void)url;
#endif
}

// The resolution choices (F2's ResolutionRow before): full screen, the desktop's or a monitor mode; window, its size.
void ResolutionChoices(const Settings& s, std::vector<std::string>* ids, std::vector<std::string>* labels) {
	static const char* sizes[] = {"1280x720", "1600x900", "1920x1080", "2560x1080", "2560x1440", "3440x1440", "3840x1600", "3840x2160"};
	ids->clear();
	labels->clear();
	if (s.fullscreen) {
		ids->emplace_back("escritorio");
		labels->emplace_back(std::string(I18n::T("Escritorio")) +
		                     (g_rt.pc.screen_h > 0 ? " (" + std::to_string(g_rt.pc.screen_h) + "p)" : std::string()));
	}
	for (const char* z: sizes) {
		if (s.fullscreen && !MonitorHasMode(z)) continue; // (rtx1b)
		ids->emplace_back(z);
		labels->emplace_back(ResLabel(z));
	}
	const std::string value = s.fullscreen ? s.resolution : std::to_string(s.width) + "x" + std::to_string(s.height);
	if (std::find(ids->begin(), ids->end(), value) == ids->end()) {
		ids->push_back(value);
		labels->push_back(ResLabel(value));
	}
}

void LauncherImage(Settings& s) {
	const bool nvidia = g_rt.pc.nvidia;
	BoxBegin("Imagen");
	{
		static std::vector<std::string> ids, labels;
		ResolutionChoices(s, &ids, &labels);
		std::vector<Choice> res;
		for (size_t i = 0; i < ids.size(); i++) res.push_back({ids[i].c_str(), labels[i].c_str()});
		std::string value = s.fullscreen ? s.resolution : std::to_string(s.width) + "x" + std::to_string(s.height);
		const float y     = RowY();
		Label(y, "Resolución", "Resolución de salida: la de Windows (Escritorio) u otra (el monitor cambia a ella mientras juegas). Los reescaladores calculan su resolución sobre ella. En ventana: el tamaño de la ventana.");
		int current = 0;
		for (size_t i = 0; i < res.size(); i++)
			if (value == res[i].id) current = static_cast<int>(i);
		const std::function<const char*(int)> list = [&](int i) -> const char* { return res[static_cast<size_t>(i)].label; };
		if (Field("Resolución", y, static_cast<int>(res.size()), current, list,
		          "Resolución de salida: la de Windows (Escritorio) u otra (el monitor cambia a ella mientras juegas). Los reescaladores calculan su resolución sobre ella. En ventana: el tamaño de la ventana.",
		          146, 260)) {
			value = res[static_cast<size_t>(current)].id;
			if (s.fullscreen) {
				s.resolution = value;
			} else if (int w = 0, h = 0; std::sscanf(value.c_str(), "%dx%d", &w, &h) == 2) {
				s.width  = w;
				s.height = h;
			}
			// (10-29) A 4K output: Iluminacion IBL Completa + Teselado Alto; any other: Mitad + Medio (changeable after).
			SetQualityForOutput(&s, OutputHeight(s, g_rt.pc));
		}
		// (10-29) The game's mode that is applied, next to it.
		RowEnd(y);
		// (The game mode that is applied, under it.)
		const float my = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(148), my});
		ImGui::PushFont(g_font_semi, U(14));
		ImGui::TextColored(kGoldL, "%s", I18n::T(OutputHeight(s, g_rt.pc) >= 2160 ? "Cinemático sin tope 30" : "Modo rendimiento"));
		ImGui::PopFont();
		Tip("4K: el modo rendimiento del juego (cinemáticas sin el tope de 30) con la calidad cinemática: iluminación completa y teselado denso. Otras resoluciónes: modo rendimiento.");
		RowEnd(my, 22);
	}
	{
		static std::vector<ResMode> shown;
		shown.clear();
		const int out_h = OutputHeight(s, g_rt.pc);
		for (int i = 0; i < kResModeCount; i++) {
			if (!nvidia && std::strcmp(kResModes[i].upscaler, "dlss") == 0) continue;
			if (std::strcmp(kResModes[i].upscaler, "fsr4") == 0 && out_h / kResModes[i].value < 359 && s.res != kResModes[i].id) continue;
			shown.push_back(kResModes[i]);
		}
		LRow("Reescalado", static_cast<int>(shown.size()), shown.data(), s.res, [](const ResMode& m) { return m.id; },
		     [](const ResMode& m) { return m.label; },
		     "Reescalado: dibuja el 3D a menos resolución y lo sube a la de la pantalla (mas fps). DLSS: NVIDIA RTX. XeSS, FSR 3.1 y FSR 4: todas las graficas (FSR 4: modelo de AMD con INT8, RTX 20 / RX 6000 o mas nuevas; el 3D a 360 lineas como minimo).",
		     146, 260, true);
	}
	ImGui::BeginDisabled(!nvidia);
	LChoice("Preset DLSS", kDlssPresets, kDlssPresetCount, s.dlss_preset,
	        nvidia ? "Modelo de DLSS: auto (el de NVIDIA), K = DLSS 4, L y M = DLSS 4.5 (mas nitidos, algo mas lentos)" : "Sin DLSS: tu grafica no es NVIDIA.");
	ImGui::EndDisabled();
	LChoice("Nitidez", kSharpness, kSharpnessCount, s.sharpness,
	        "Nitidez del reescalado (DLSS, XeSS, FSR 3.1 y FSR 4), aplicada a la resolución de la pantalla. Más alto: mas nitido, pero puede marcar bordes y ruido.");
	// (ds-pc-port10) DLSS 5 (NVIDIA's Neural Rendering on the foveated grid) and which nvngx_dlssnr.dll it may load.
	ImGui::BeginDisabled(!nvidia);
	LChoice("DLSS 5", kDlss5Levels, kDlss5LevelCount, s.dlss5, nvidia ? kDlss5Tip : "Sin DLSS 5: tu grafica no es NVIDIA.");
	ImGui::BeginDisabled(s.dlss5 == "0");
	{
		std::string file = s.dlss5_unsigned ? "1" : "0";
		if (LChoice("Archivo DLSS 5", kDlss5Files, kDlss5FileCount, file, kDlss5FileTip)) s.dlss5_unsigned = file == "1";
	}
	LChoice("Intensidad DLSS 5", kDlss5Strengths, kDlss5StrengthCount, s.dlss5_strength, kDlss5StrengthTip);
	ImGui::EndDisabled();
	ImGui::EndDisabled();
	BoxEnd();
}

void LauncherQuality(Settings& s) {
	BoxBegin("Calidad gráfica");
	LChoice("Texturas", kTextureReduce, kTextureReduceCount, s.texture_reduce, "Tamaño de las texturas: Medio = la mitad, Bajo = un cuarto. Lo que más VRAM ahorra.");
	LChoice("Distancia", kQualities, kQualityCount, s.quality, "Distancia de dibujado: a menos nivel, los objetos lejanos pasan antes a modelos más simples. Más FPS si va justo el procesador.");
	LChoice("Sombras", kShadows, kShadowCount, s.shadows, "Sombras: Medio = las del sol hasta 125 m, Bajo = hasta 60 m y sin sombras de focos ni ambientales. Más FPS, pero pueden aparecer de golpe.");
	LChoice("Detalle", kDetails, kDetailCount, s.detail, "Detalle del juego: sombras (cascadas y distancia), iluminación de entorno y relieve (teselado). Ultra > Muy alto (modo cine) > Alto (modo rendimiento de PS5) > Normal > Bajo.");
	{
		const float y = RowY();
		LCheck(y, "Filtrado x16", s.af16, "Filtrado anisotrópico x16: suelos y paredes nítidos a lo lejos y en ángulo. Cuesta muy poco.", 10);
		int current = 0;
		for (int i = 0; i < kTextureSharpnessCount; i++)
			if (s.texture_sharpness == kTextureSharpness[i].id) current = i;
		const std::function<const char*(int)> list = [](int i) -> const char* { return kTextureSharpness[i].label; };
		if (Field("Nitidez de texturas", y, kTextureSharpnessCount, current, list,
		          "Texturas: nítidas = el detalle de la resolución final con reescalado. Extra nítidas: un escalón más (lo que recomienda NVIDIA para DLSS); puede brillar algo en rejas y hierba. Cuesta pocos FPS.",
		          216, 190))
			s.texture_sharpness = kTextureSharpness[current].id;
		RowEnd(y);
	}
	LChoice("Reflejos", kReflections, kReflectionCount, s.reflections, "Reflejos del propio juego (agua, charcos, suelos): más pasos y más alcance en Ultra. Cuesta algo de GPU.");
	LChoice("Desvanecimiento", kReflections, kReflectionCount, s.object_fade, "Distancia a la que los objetos lejanos se desvanecen. Ultra: aparecen de golpe mucho menos.");
	LChoice("Nitidez del AA", kAaSharp, kAaSharpCount, s.aa_sharp, "Nitidez del antialiasing del juego. No cuesta FPS.");
	{
		// The game's post-processing (marked: as the game ships it), two to a row.
		struct Post {
			const char* label;
			bool*       value;
			const char* tip;
		};
		const Post post[] = {{"Desenfoque de movimiento", &s.motion_blur, "Desenfoque de movimiento del juego (Sí = como en PS5)."},
		                     {"Bloom", &s.bloom, "Resplandor alrededor de las luces fuertes (Sí = como en PS5)."},
		                     {"Profundidad de campo", &s.dof, "Desenfoque de lo lejano y de lo cercano en escenas (Sí = como en PS5)."},
		                     {"Viñeta", &s.vignette, "Oscurecimiento de los bordes de la imagen (Sí = como en PS5)."},
		                     {"Adaptación de la vista", &s.eye_adapt, "La exposición se adapta al pasar de zonas oscuras a claras, como el ojo (Sí = como en PS5)."}};
		for (size_t i = 0; i < std::size(post); i += 2) {
			const float y = RowY();
			LCheck(y, post[i].label, *post[i].value, post[i].tip, 10);
			if (i + 1 < std::size(post)) LCheck(y, post[i + 1].label, *post[i + 1].value, post[i + 1].tip, 270);
			RowEnd(y, 30);
		}
	}
	ResolveQualityAuto(&s, g_rt.pc);
	LChoice("Iluminación IBL", kIbl, kIblCount, s.ibl, "Resolución de la iluminación de entorno (IBL) del juego: Completa = la del modo cine de PS5, más fina; Mitad = la del modo rendimiento. Con 4K empieza en Completa, con otras salidas en Mitad.");
	LChoice("Teselado", kTess, kTessCount, s.tess, "Relieve por teselado del juego (suelos, muros, piedras): Bajo, Medio, Alto (el del modo cine de PS5), Ultra y Extremo (más relieve y más lejos; cuesta GPU). Con 4K empieza en Alto, con otras salidas en Medio.");
	BoxEnd();
}

// (ds-pc-port37) The emulator launcher's "Ajustes del juego" window, as a drop-down box of the launcher's options.
bool g_show_game_settings = false;
void LauncherGameSettings(Settings& s) {
	BoxBegin("Ajustes del juego");
	static const Choice levels[] = {{"0", "Predeterminado"}, {"1", "Bajo"}, {"2", "Medio"}, {"3", "Alto"}, {"4", "Ultra"}, {"5", "Extremo"}};
	static const Choice onoff[]  = {{"0", "Predeterminado"}, {"1", "No"}, {"2", "Sí"}};
	{
		const float y = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), y});
		ImGui::PushTextWrapPos(g_lbox.at.x + g_lbox.w - U(12) - ImGui::GetWindowPos().x);
		ImGui::TextColored(kDimL, "%s", I18n::T("Ajustes del propio juego. Predeterminado = como viene en PS5, sin tocar nada. Pasa el ratón por encima de cada uno para ver qué hace y lo que cuesta. Mandan sobre las opciones de Calidad gráfica."));
		ImGui::PopTextWrapPos();
		RowEnd(y, (ImGui::GetCursorScreenPos().y - y) / g_unit + 6);
	}
	size_t      n   = 0;
	const auto* all = GameSettings(&n);
	const char* sec = "";
	for (size_t i = 0; i < n; i++) {
		const GameSetting& g = all[i];
		if (std::strcmp(sec, g.section) != 0) {
			sec           = g.section;
			const float y = RowY();
			ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), y + U(4)});
			ImGui::TextColored(kGoldL, "%s", I18n::T(sec));
			RowEnd(y, 30);
		}
		std::string v = std::to_string(GameSettingLevel(s, g.id));
		ImGui::PushID(g.id);
		if (LChoice(g.name, g.onoff ? onoff : levels, g.onoff ? 3 : 6, v, g.tip, 190, 216)) SetGameSettingLevel(&s, g.id, std::atoi(v.c_str()));
		ImGui::PopID();
	}
	{
		const float y = RowY() + U(2);
		if (LButton("Todo predeterminado", {g_lbox.at.x + U(12), y}, {U(396), U(32)}, false, g_font_semi, U(15),
		            "Pone todos en Predeterminado: el juego como viene, sin tocar nada."))
			ResetGameSettings(&s);
		RowEnd(y - U(2), 40);
	}
	BoxEnd();
}

void LauncherRayTracing(Settings& s) {
	BoxBegin("Ray tracing");
	ImGui::BeginDisabled(s.pure_game);
	const float fx = 154, fw = 252;
	LRow("Reflejos RT", kRtReflLevelCount, kRtReflLevels, s.rt_ss_refl, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Reflejos trazados por rayos guiados por los del propio juego: solo donde el juego refleja y con su intensidad, más nítidos (agua como espejo). Donde el rayo no encuentra nada queda el reflejo del juego. Solo con DLSS, XeSS o FSR. Bajo ~0,6 ms, Medio ~1 ms, Alto ~2 ms, Ultra ~3 ms (rayos a resolución completa y más fuertes). Híbrido: Alto + rayos por hardware (RT real) solo donde el rayo de pantalla no encuentra nada. Necesita GPU con RT. Beta.",
	     fx, fw);
	// (refl-sharp-1031)
	LRow("Nitidez reflejos RT", kRtReflSharpCount, kRtReflSharp, s.rt_refl_sharp, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Más nitidez en los reflejos RT (superficies y agua). Normal: como ahora. Alta: rayos a resolución completa en Bajo/Medio, menos desenfoque y menos mezcla con frames anteriores (~+0,3-0,8 ms). Máxima: además sin tablero de ajedrez, el desenfoque mínimo y el historial más corto; los espejos conservan el detalle, puede verse algo más de ruido (~+0,6-1,5 ms).", fx, fw);
	LRow("Ondas agua reflejos", kRtReflWavesCount, kRtReflWaves, s.rt_refl_waves, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Cuánto mueven las ondas del agua el rayo del reflejo (el agua se ve igual). Como ahora: sin cambios. Suaves: 40 % de la ondulación, reflejos más limpios. Espejo: el agua refleja como un espejo plano. Sin coste.", fx, fw);
	// (rtq2) The old ray-tracing rows are retired (forced off in ds-settings.cpp): only Suelo estable and Reflejos RT nuevos.
	LRow("Luz volumétrica", kVolLevelCount, kVolLevels, s.godrays, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Luz volumétrica por rayos: rayos de sol que entran por ventanas y puertas, y halo de luz en la niebla alrededor de antorchas. Solo con DLSS, XeSS o FSR. Bajo ~0,4 ms, Medio ~0,8 ms, Alto ~1,2 ms. Ultra: volumen a media resolución (rayos más definidos), más muestras y la niebla del propio juego más densa y más dirigida hacia la luz (~+1 ms).", fx, fw);
	LRow("Partículas volum.", kMotesLevelCount, kMotesLevels, s.motes, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Motas de polvo en suspensión iluminadas por los rayos de sol y el halo de antorchas y lámparas (la luz volumétrica); flotan despacio y quedan fijas en el mundo. Sutil / Media / Densa: cuántas. Solo con DLSS, XeSS o FSR. ~0,2-0,5 ms.", fx, fw); // (motes-1031)
	// (ds-pc-port10) Luz de antorchas RT and Destellos de lente are gone from the port (always off).
	LRow("Humo y partículas", kSmokeLevelCount, kSmokeLevels, s.smoke, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Humo y partículas más realistas sobre las del propio juego: Suave = el humo, la niebla de partículas y las chispas ya no cortan en línea dura donde tocan suelo y paredes (se funden en ~35 cm). Solo con DLSS, XeSS o FSR. ~0,3 ms.",
	     fx, fw);
	ImGui::EndDisabled();
	// (rtq2) No "RT por hardware (experimental)" rows: retired.
	LCheckRow("Juego original (sin efectos añadidos)", s.pure_game,
	          "Apaga todos los efectos visuales que añade el emulador (ray tracing, luz volumétrica, humo, SSGI, reflejos RT). Se ve el juego tal cual. Rendimiento, reescalador, DLSS 5, frame generation y latencia no cambian.");
	BoxEnd();
}

void LauncherFrameGen(Settings& s) {
	const bool nvidia = g_rt.pc.nvidia;
	BoxBegin("Frame generation");
	static std::vector<Choice> modes;
	modes.clear();
	for (const char* id: {"sin-fg", "amd-x2", "amd-x3", "amd-x4", "amd-dinamico", "dlss-x2", "dlss-x3", "dlss-x4", "dlss-dinamico"})
		if (nvidia || std::strncmp(id, "dlss", 4) != 0) modes.push_back({id, FrameGenLabel(id)});
	// (ds-pc-port9) Modo Souls ligero / equilibrado / low profile: AMD x2 fixed and its Anti-Lag Ultra (shown, not changeable).
	const bool  fixed  = FixedFg(s);
	std::string locked = "amd-x2", ultra = "ultra";
	ImGui::BeginDisabled(fixed);
	LRow("Frame generation", static_cast<int>(modes.size()), modes.data(), fixed ? locked : s.framegen, [](const Choice& c) { return c.id; },
	     [](const Choice& c) { return c.label; },
	     fixed ? "Con este Modo Souls la frame generation es fija: AMD x2." : "AMD: todas las graficas (tambien con reescalado DLSS). DLSS: NVIDIA RTX 40/50. FPS desbloqueados siempre.",
	     12, 396, true);
	if (fixed) {
		LChoice("Anti-Lag", g_rt.pc.amd ? kAntiLagAmd : kAntiLagOther, g_rt.pc.amd ? kAntiLagAmdCount : kAntiLagOtherCount, ultra,
		        "En los modos Souls ligero, equilibrado y low profile va siempre el nuestro en Ultra.", 146, 260);
	} else if (s.framegen.rfind("dlss", 0) == 0) {
		LChoice("Anti-Lag", kAntiLagDlssg, kAntiLagDlssgCount, s.antilag_dlssg,
		        "Con la frame generation de NVIDIA (DLSS-G). Reflex: el de NVIDIA (recomendado). Anti-Lag / Ultra: el nuestro con Reflex apagado, para comparar cual va mejor.",
		        146, 260);
	} else if (s.framegen.rfind("amd", 0) == 0) {
		std::string al = s.antilag_mode == "auto" || s.antilag_mode.empty() ? (g_rt.pc.amd ? "amd2" : "normal") : s.antilag_mode;
		if (LChoice("Anti-Lag", g_rt.pc.amd ? kAntiLagAmd : kAntiLagOther, g_rt.pc.amd ? kAntiLagAmdCount : kAntiLagOtherCount, al,
		            "Con la frame generation de AMD. Anti-Lag 2 (AMD): el de serie de las graficas AMD. Anti-Lag: el nuestro, como mucho 2 frames en cola antes de leer el mando. Ultra: 1 (aun menos retraso).",
		            146, 260))
			s.antilag_mode = al;
	}
	ImGui::EndDisabled();
	BoxEnd();
}

void LauncherDisplay(Settings& s) {
	BoxBegin("Pantalla y otros");
	{
		const float y = RowY();
		LCheck(y, "HDR", s.hdr, "HDR10: activa antes \"Usar HDR\" en la configuracion de pantalla de Windows", 12);
		LCheck(y, "VSync", s.vsync, "VSync: sin cortes de imagen (tearing); con FreeSync/G-Sync sigue funcionando", 92);
		static const Choice aspects[] = {{"16:9", "Monitor 16:9"}, {"21:9", "Monitor 21:9"}, {"32:9", "Monitor 32:9"}, {"48:9", "Monitor 48:9"}};
		int current = 0;
		for (int i = 0; i < 4; i++)
			if (s.aspect == aspects[i].id) current = i;
		const std::function<const char*(int)> list = [](int i) -> const char* { return aspects[i].label; };
		if (Field("Monitor", y, 4, current, list,
		          "Monitores panoramicos (ultrawide): el juego se dibuja a 21:9, 32:9 o 48:9 en vez de estirarse. Funciona con DLSS, XeSS y FSR. Los videos y algunos menus se ven estirados.",
		          280, 128))
			s.aspect = aspects[current].id;
		RowEnd(y);
	}
	{
		static const Choice modes[] = {{"1", "Pantalla completa"}, {"0", "Ventana"}};
		std::string mode = s.fullscreen ? "1" : "0";
		if (LChoice("Modo de pantalla", modes, 2, mode, "Pantalla completa o ventana (F11 lo cambia en vivo).", 150, 256)) s.fullscreen = mode == "1";
	}
	LCheckRow("Mantener 16:9 (bandas negras)", s.aspect_fit, "Con un monitor panorámico: el juego en 16:9 con bandas negras a los lados en vez de estirarse.");
	{
		// (ds-pc-port37) A check box, as the emulator launcher's "Metricas" (on: FPS and the detail, as before when chosen).
		bool on = s.metrics == "1" || s.metrics == "2";
		if (LCheckRow("Métricas", on, "Panel dentro del juego (F8 lo cambia en vivo): FPS, el modo de frame generation, latencia y el detalle."))
			s.metrics = on ? (s.metrics == "2" ? "2" : "1") : "0";
	}
	BoxEnd();
}

void LauncherConfigurator(Settings& s) {
	BoxBegin("Configurador");
	const float y = RowY() + U(2);
	if (LButton("Optimizar según tu equipo", {g_lbox.at.x + U(12), y}, {U(396), U(34)}, false, g_font_semi, U(16),
	            "Elige los ajustes con mejor rendimiento para tu gráfica, procesador, RAM y monitor (objetivo: 60 FPS o más) y los guarda en settings.ini. Cambia gráficos, ray tracing y la opción de 16 GB de RAM.")) {
		g_cfg_why    = Optimize(g_rt.pc, &s);
		s.configured = true;
		Save();
	}
	RowEnd(y - U(2), 40);
	if (!g_cfg_why.empty()) {
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), RowY()});
		ImGui::BeginGroup();
		ImGui::PushTextWrapPos(g_lbox.at.x + g_lbox.w - U(12) - ImGui::GetWindowPos().x);
		ImGui::TextColored(kGoldL, "%s", I18n::T("Hemos elegido:"));
		for (const auto& line: g_cfg_why) ImGui::BulletText("%s", line.c_str());
		ImGui::PopTextWrapPos();
		ImGui::EndGroup();
	}
	BoxEnd();
}

void LauncherPerformance(Settings& s) {
	BoxBegin("Rendimiento / PC");
	{
		const float y = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(14), y + U(6)});
		ImGui::TextColored(kGold, "Archivo de paginación de Windows");
		RowEnd(y, 30);
		const float y1 = RowY();
		if (LButton("Optimizado (60 GB)", {g_lbox.at.x + U(12), y1}, {U(196), U(32)}, false, g_font_semi, U(15),
		            "Pone en el disco C: un archivo de paginación fijo de 60 GB (Windows pedirá permiso de administrador). Evita que el PC se reinicie o el juego se cierre al quedarse sin memoria en equipos con poca RAM. Reinicia el PC después."))
			SetPagefile(60);
		if (LButton("Por defecto de Windows", {g_lbox.at.x + U(216), y1}, {U(196), U(32)}, false, g_font_semi, U(15),
		            "Deja el archivo de paginación como viene en Windows (lo gestiona Windows solo). Windows pedirá permiso de administrador. Reinicia el PC después."))
			SetPagefile(0);
		RowEnd(y1, 36);
		const float y2 = RowY();
		static int custom_gb = 32;
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(14), y2 + U(6)});
		ImGui::TextUnformatted("El que quieras (GB):");
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(176), y2 + U(2)});
		ImGui::SetNextItemWidth(U(110));
		if (ImGui::InputInt("##pagefile_gb", &custom_gb, 4, 16)) custom_gb = std::clamp(custom_gb, 4, 256);
		if (LButton("Aplicar", {g_lbox.at.x + U(296), y2}, {U(116), U(32)}, false, g_font_semi, U(15),
		            "Pone en el disco C: un archivo de paginación fijo del tamaño elegido (4 a 256 GB). Windows pedirá permiso de administrador. Reinicia el PC después."))
			SetPagefile(custom_gb);
		RowEnd(y2, 36);
		const uint64_t limit = CommitLimitGb();
		if (limit != 0) {
			const float ny = RowY();
			ImGui::SetCursorScreenPos({g_lbox.at.x + U(14), ny});
			const bool pending = limit < 64 && PagefileConfigured();
			ImGui::TextColored(limit >= 64 ? kGoldL : kGold, "Memoria total de Windows (RAM + paginación): %llu GB%s",
			                   static_cast<unsigned long long>(limit),
			                   limit >= 64 ? "" : pending ? "  -  hecho: REINICIA EL PC" : "  -  elige una opción y reinicia");
			RowEnd(ny, 24);
		}
	}
	s.profile16 = LowRam(s, g_rt.pc);
	{
		const float y = RowY();
		LCheck(y, "Todos los núcleos", s.all_ccds, "Solo para Ryzen con dos bloques de núcleos (CCD), como un 7950X3D o un 5900X; en otros procesadores no cambia nada. Apagado: el juego usa solo el bloque con más caché (memoria rápida dentro del procesador), que suele dar más FPS y menos tirones. Encendido: usa todos los núcleos. Déjalo apagado salvo que notes que encendido te va mejor.", 216);
		RowEnd(y);
	}
	// (11-14) Pruebas: la gráfica limitada a 8 GB de memoria de vídeo (KYTY_SIMULATE_VRAM_MB=8192).
	LCheckRow("Simular gráfica de 8 GB (pruebas)", s.simulate_vram8,
	          "Solo para pruebas: el juego usa como mucho 8 GB de memoria de vídeo, como en una gráfica de 8 GB, aunque la tuya tenga más. Se aplica al iniciar el juego.");
	{
		// (perfopts-1107) Opciones de rendimiento: upstream KytyPS5's opt-in switches, off = as before.
		const float y = RowY() + U(4);
		auto*       dl = ImGui::GetWindowDrawList();
		dl->AddLine({g_lbox.at.x + U(12), y}, {g_lbox.at.x + g_lbox.w - U(12), y}, IM_COL32(222, 196, 140, 60), 1.0f);
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), y + U(6)});
		ImGui::TextColored(kGoldL, "%s", I18n::T("Opciones de rendimiento"));
		Tip("Ajustes experimentales que buscan más FPS o menos tirones de distintas formas. Todos apagados = el juego va como siempre. La ganancia suele ser pequeña y depende de tu PC: prueba uno a uno. Se aplican al iniciar el juego. Si algo va mal, apágalos.");
		RowEnd(y, 30);
	}
	// (11-22) "Menos trabajo por cada dibujo", "Enviar texturas con dos hilos", "Devolver antes las copias de la gráfica" and
	// "Agrupar las copias de la gráfica": always on in code (ds-settings.cpp), no boxes.
	LCheckRow("Ayuda extra si el procesador va justo", s.perf_upload_help,
	          "Cuando el procesador va muy ocupado, el hilo que dibuja cada imagen hace él mismo los envíos de texturas pendientes en vez de quedarse esperando. En procesadores de 4 a 6 núcleos y portátiles puede dar unos FPS más (en pruebas, hasta un 4 %). Con un procesador sobrado casi nunca actúa: ahí da igual.");
	LCheckRow("Liberar el procesador cuando va saturado", s.perf_spin_gate,
	          "Unos hilos de ayuda del emulador esperan trabajo dando vueltas sin parar (gastan procesador sin hacer nada útil). Con esto dejan de hacerlo cuando quedan menos de 3 núcleos libres, y ese procesador vuelve al juego. En procesadores de 4 a 6 núcleos reduce bastante los bajones de FPS y los tirones. Con muchos núcleos casi nunca actúa. Recomendado en portátiles y PCs modestos.");
	LCheckRow("Traducción rápida para más efectos", s.perf_lane_wide,
	          "Más programas de efectos del juego (shaders: los pequeños programas que usa la gráfica para calcular y dibujar) se traducen a una versión más rápida. Puede dar algo más de FPS en la gráfica. La primera vez que lo actives esos shaders se vuelven a preparar: más carga y algún tirón al empezar. Experimental: si ves fallos gráficos, apágalo.");
	// (11-12) "Simular 16 GB RAM" removed too (the owner).
	// (11-12) "Simular grafica de 8 GB" removed (the owner: black textures with it).
	BoxEnd();
}

void LauncherControls(Settings& s) {
	BoxBegin("Controles");
	LCheckRow("Teclado y ratón", s.kbm,
	          "Jugar con teclado y ratón: WASD para moverse, el ratón mueve la cámara, clic izquierdo R1, clic derecho L1, rueda o clic central fija objetivo. F7 suelta o captura el ratón. Con él activo la cámara gira sin la aceleración del juego, también con el stick derecho del mando: si juegas con mando, déjalo desactivado.");
	ImGui::BeginDisabled(!s.kbm);
	{
		const float y = RowY();
		LSlider(y, "Sensibilidad del ratón", s.kbm_sens, 0.05f, 4.0f, "%.2f", "Cuánto gira la cámara por cada movimiento del ratón (1.00 = normal).", 12, 190);
		LSlider(y, "Antizona muerta", s.kbm_antidz, 0.0f, 0.5f, "%.2f",
		        "Salta la zona muerta del stick del juego: con más, un movimiento pequeño del ratón ya gira la cámara. Si la cámara se mueve sola, bájalo.", 216, 190);
		RowEnd(y, 58);
	}
	{
		const float y   = RowY();
		float       cam = static_cast<float>(s.kbm_camera);
		if (LSlider(y, "Velocidad de cámara (ratón)", cam, 5.0f, 15.0f, "%.0f",
		            "La velocidad máxima de giro de la cámara del juego con teclado y ratón. 10 = el máximo del menú del juego. 11-15 (extra, experimental): más rápido que el menú; si la cámara hace cosas raras, vuelve a 10. Con el ratón en modo directo (por defecto, KYTY_KBM_DIRECT) ya no limita al ratón: el ratón gira la cámara exactamente su ángulo, sin tope; solo afecta al stick.",
		            12, 190))
			s.kbm_camera = std::clamp(static_cast<int>(std::lround(cam)), 5, 15);
		LCheck(y + U(20), "Cámara instantánea", s.kbm_warp,
		       "Cámara instantánea (frame warp): la imagen gira con el ratón antes de que el juego dibuje el siguiente fotograma: la cámara responde al instante.", 216);
		RowEnd(y, 58);
	}
	{
		std::string cal = s.kbm_calibrate ? "1" : "0";
		static const Choice cal_items[] = {{"0", "No"}, {"1", "En el próximo inicio"}};
		if (LChoice("Calibrar cámara", cal_items, 2, cal,
		            "Mide cómo gira la cámara del juego para que el ratón la mueva exactamente lo que se mueve (se guarda en kbm_curve.txt; una vez basta). Al iniciar: carga tu partida, quédate quieto en un sitio abierto sin tocar el ratón ni el stick; a los 3 s la cámara gira sola unos 13 s.",
		            150, 256))
			s.kbm_calibrate = cal == "1";
	}
	ImGui::EndDisabled();
	{
		const float y = RowY();
		LCheck(y, "Vibración", s.rumble, "Vibración del mando.", 12);
		ImGui::BeginDisabled(!s.rumble);
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(150), y + U(3)});
		ImGui::SetNextItemWidth(U(256));
		ImGui::SliderFloat("##rumble", &s.rumble_gain, 1.0f, 8.0f, I18n::T("Fuerza %.1f"));
		Tip("Fuerza de la vibración del mando.");
		ImGui::EndDisabled();
		RowEnd(y);
	}
	{
		const float y = RowY();
		if (LButton(g_show_keys ? "Teclas  ^" : "Teclas  v", {g_lbox.at.x + U(12), y + U(2)}, {U(396), U(28)}, false, g_font_body, U(15),
		            "Las teclas de teclado y ratón (se cambian en mapa_teclado de settings.ini)."))
			g_show_keys = !g_show_keys;
		RowEnd(y, 34);
	}
	if (g_show_keys) {
		static const char* kDefault =
		    "LeftStickUp=W;LeftStickDown=S;LeftStickLeft=A;LeftStickRight=D;Walk=Left Ctrl;R1=Mouse:Left;L1=Mouse:Right;R2=Shift+Mouse:Left;L2=Shift+Mouse:Right;R3=Mouse:Middle;R3=Mouse:WheelUp;R3=Mouse:WheelDown;Circle=Space;Square=E;Cross=F;Triangle=G;Left=Q;Right=R;Up=1;Down=2;Left=3;Right=4;Options=Tab;TouchPad=Backspace";
		static const std::pair<const char*, const char*> names[] = {
		    {"LeftStickUp", "Avanzar"}, {"LeftStickDown", "Retroceder"}, {"LeftStickLeft", "Izquierda"}, {"LeftStickRight", "Derecha"},
		    {"Walk", "Caminar"}, {"R1", "R1 (ataque)"}, {"L1", "L1 (bloquear)"}, {"R2", "R2 (ataque fuerte)"}, {"L2", "L2 (parada)"},
		    {"R3", "R3 (fijar objetivo)"}, {"Circle", "Círculo (rodar / correr)"}, {"Square", "Cuadrado (objeto)"}, {"Cross", "Cruz (acción)"},
		    {"Triangle", "Triángulo (cambiar agarre)"}, {"Left", "Izquierda cruceta"}, {"Right", "Derecha cruceta"}, {"Up", "Arriba cruceta"},
		    {"Down", "Abajo cruceta"}, {"Options", "Menú"}, {"TouchPad", "Panel táctil"}};
		const std::string map = s.kbm_map.empty() ? std::string(kDefault) : s.kbm_map;
		size_t            at  = 0;
		while (at <= map.size()) {
			const size_t      end  = std::min(map.find(';', at), map.size());
			const std::string item = map.substr(at, end - at);
			at                     = end + 1;
			const size_t eq        = item.find('=');
			if (eq == std::string::npos) continue;
			const std::string c     = item.substr(0, eq);
			std::string       shown = c;
			for (const auto& [k, v]: names)
				if (c == k) shown = I18n::T(v);
			const float y = RowY();
			ImGui::SetCursorScreenPos({g_lbox.at.x + U(16), y});
			ImGui::TextUnformatted(shown.c_str());
			ImGui::SetCursorScreenPos({g_lbox.at.x + U(230), y});
			ImGui::TextColored(kGoldL, "%s", item.substr(eq + 1).c_str());
			RowEnd(y, 22);
		}
	}
	{
		const float y = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), y + U(4)});
		ImGui::PushTextWrapPos(g_lbox.at.x + g_lbox.w - U(12) - ImGui::GetWindowPos().x);
		ImGui::TextColored(kDimL, "%s", I18n::T("En el juego: F2 guardar y salir (o R3 + OPTIONS).  F7: soltar o capturar el ratón.  F8: métricas.  F11 / Alt+Intro: pantalla completa."));
		ImGui::PopTextWrapPos();
		ImGui::Dummy({0, U(4)});
	}
	BoxEnd();
}

// (ds-pc-port9) Modos Souls (Launcher-souls.ps1's group): one at most; choosing one applies its preset (ApplyPreset:
// the graphics, the upscaler and every ray tracing row to No); unticking it goes back to Maximo.
void LauncherSoulsModes(Settings& s) {
	BoxBegin("Graficas de 12 y 16 GB");
	struct Mode {
		const char* id;
		const char* label;
		const char* tip;
	};
	static const Mode modes[] = {
	    {"ligero", "Modo Souls ligero",
	     "Adaptado para graficas de 8 GB como RTX 3060, RTX 3060 Ti, RTX 3070, RTX 2070/2080, RX 6600, RX 6600 XT o RX 6650 XT: el juego a 30 FPS estables y frame generation de AMD x2 (60 en pantalla). A 1080p con FSR 4 Calidad. Ajustes preparados para graficas de 8 GB y 16 GB de RAM: no es conveniente cambiarlos."},
	    {"equilibrado", "Modo Souls equilibrado",
	     "Para graficas de 10 GB de VRAM o mas, de la potencia de una RTX 3080: RTX 2080 Ti, RTX 3070 Ti, RTX 3080, RTX 3080 Ti, RTX 4060 Ti, RTX 4070, RX 6800, RX 6800 XT, RX 6900 XT, RX 6950 XT, RX 7700 XT, RX 7800 XT o Intel Arc B580: el juego a 45 FPS estables y frame generation de AMD x2 (90 en pantalla). A 1440p, con el reescalado que elijas."},
	    {"ultra", "Modo Souls ultra",
	     "Para graficas de 12 y 16 GB de VRAM (RTX 5060 Ti 16 GB / 5070 / 5070 Ti, RX 9070 / 9070 XT): todo al maximo, Detail Very High, Luz RT y Oclusion en Alta. Los FPS dependen de la potencia de tu GPU y tu CPU."},
	    {"extremo", "Modo Souls extremo",
	     "Para graficas de 16 GB de VRAM. Para enfermos de Souls. RTX 5080, RTX 5090 o superior: modo cinematico del juego a 4K y absolutamente todo al maximo (Ultra en todo, filtrado x16 y texturas nitidas). Los FPS dependen de la potencia de tu GPU y tu CPU."},
	    {"ligero-plus", "Modo Souls ligero +",
	     "Para monitores de mas de 60 Hz. Como el Modo Souls ligero (mismos graficos), pero con el juego a 40 FPS estables y frame generation de AMD x2 (80 en pantalla): mas fluido y algo menos de latencia. Ajustes preparados para graficas de 8 GB y 16 GB de RAM: no es conveniente cambiarlos."},
	    {"low", "Modo Souls PC low profile (900p, todo en bajo)",
	     "Para PCs modestos: 30 FPS reales con frame generation de AMD x2 (60 en pantalla), el juego a 900p reescalado con FSR (nitidez 50 %) y todos los graficos en bajo (texturas, distancia, sombras, detalle, reflejos, objetos y nitidez). Al quitarlo vuelve todo a Maximo y a 1440p."},
	    {"sin-tirones", "Optimizado 16 GB RAM / 8 GB VRAM",
	     "Para PCs de 16 GB de RAM y graficas de 8 GB como RTX 3060 Ti, RTX 3070, RTX 4060, RTX 4060 Ti 8 GB, RTX 5060, RX 6600 XT, RX 6650 XT o RX 7600. 40 FPS estables con frame generation de AMD x2 (80 en pantalla), 1080p con FSR 4 Calidad. Menos carga y menos tirones: sin calcomanias proyectadas, reflejos y desvanecimiento en Normal, sin desenfoque de movimiento. Ajustes preparados para graficas de 8 GB y 16 GB de RAM: no es conveniente cambiarlos."}};
	const auto pick = [&](float y, const Mode& m, float x) {
		bool on = s.souls_mode == m.id;
		if (!LCheck(y, m.label, on, m.tip, x)) return;
		ApplyPreset(&s, on ? m.id : "ninguno");
		s.souls_mode = on ? m.id : "ninguno";
	};
	// (11-22) The 8 GB modes together: ligero | ligero +, then Optimizado (its own row: a long name); then
	// equilibrado | ultra, extremo | low profile. (modes[]: 0 ligero, 1 equilibrado, 2 ultra, 3 extremo, 4 ligero +,
	// 5 low, 6 optimizado)
	// (11-26, el usuario) Los de 8 GB en su propio apartado, con el aviso de no cambiar sus ajustes.
	const auto rows = [&](const int (*order)[2], int count) {
		for (int i = 0; i < count; i++) {
			const float y = RowY();
			pick(y, modes[order[i][0]], 12);
			if (order[i][1] >= 0) pick(y, modes[order[i][1]], 216);
			RowEnd(y);
		}
	};
	// (11-26, el usuario) Apartados por memoria de la grafica: 8 GB (+16 GB de RAM), 10 GB o mas, 12 y 16 GB; low profile aparte.
	static const int doce[1][2] = {{2, 3}};
	rows(doce, 1);
	BoxEnd();
	BoxBegin("Graficas de 10 GB o mas");
	static const int diez[1][2] = {{1, -1}};
	rows(diez, 1);
	BoxEnd();
	BoxBegin("PCs modestos");
	static const int bajo[1][2] = {{5, -1}};
	rows(bajo, 1);
	BoxEnd();
	BoxBegin("Graficas de 8 GB y 16 GB de RAM");
	static const int ocho[2][2] = {{0, -1}, {6, -1}}; // (11-26) sin "ligero +": solo Souls ligero y Optimizado
	rows(ocho, 2);
	{
		const float y = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(14), y + U(4)});
		ImGui::PushTextWrapPos(g_lbox.at.x + g_lbox.w - U(12) - ImGui::GetWindowPos().x);
		ImGui::TextColored(kWarn, "%s", I18n::T("Ajustes preparados especialmente para graficas de 8 GB y 16 GB de RAM (1080p con FSR 4 Calidad). No es conveniente cambiarlos: subirlos puede llenar la memoria de video y dar tirones o texturas que tardan en cargar."));
		ImGui::PopTextWrapPos();
		// (11-26) Alto real del texto (3 lineas con letra grande): la linea de la frame generation va debajo, sin pisarlo.
		const float alto = ImGui::GetItemRectMax().y - y + U(6);
		ImGui::SetCursorScreenPos({g_lbox.at.x, y});
		ImGui::Dummy({g_lbox.w, alto});
	}
	if (FixedFg(s)) {
		const float y = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(14), y + U(4)});
		ImGui::TextColored(kGoldL, "%s", I18n::T(s.souls_mode == "equilibrado" ? "Frame generation fija: AMD x2 (45 reales, 90 en pantalla)." : (s.souls_mode == "ligero-plus" || s.souls_mode == "sin-tirones") ? "Frame generation fija: AMD x2 (40 reales, 80 en pantalla)." : "Frame generation fija: AMD x2 (30 reales, 60 en pantalla)."));
		RowEnd(y, 28);
	}
	BoxEnd();
}

// Ajustes (Launcher-souls.ps1's Settings): the game's folder, Add to Steam, the languages, the FPS cap, the copies.
void LauncherAjustes(Settings& s) {
	BoxBegin("Ajustes");
	{
		const bool  valid = !g_rt.is_game || g_rt.is_game(s.game);
		const float y     = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), y + U(2)});
		ImGui::TextColored(kGoldL, "%s", I18n::T("Carpeta del juego volcado (PPSA01341)"));
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), ImGui::GetCursorScreenPos().y});
		ImGui::PushTextWrapPos(g_lbox.at.x + g_lbox.w - U(12) - ImGui::GetWindowPos().x);
		ImGui::TextColored(valid ? kInkL : kWarn, "%s", s.game.empty() ? I18n::T("(sin elegir)") : s.game.c_str());
		if (!valid) ImGui::TextColored(kWarn, "%s", I18n::T("(no valida: JUGAR te la pide)"));
		ImGui::PopTextWrapPos();
		const float by = ImGui::GetCursorScreenPos().y + U(4);
		if (LButton("Cambiar...", {g_lbox.at.x + U(12), by}, {U(190), U(32)}, false, g_font_semi, U(16),
		            "La carpeta con el juego volcado (la que tiene eboot.bin y sce_sys).") &&
		    g_rt.pick_folder) {
			const std::string picked = g_rt.pick_folder();
			if (!picked.empty() && (!g_rt.is_game || g_rt.is_game(picked))) {
				s.game = picked;
				Save();
				g_status = "Guardado en settings.ini";
			} else if (!picked.empty()) {
				g_status = "Esa carpeta no tiene eboot.bin ni sce_sys: elige la carpeta del juego volcado.";
			}
		}
		ImGui::BeginDisabled(g_steam_busy.load());
		if (LButton("Añadir a Steam", {g_lbox.at.x + U(214), by}, {U(194), U(32)}, false, g_font_semi, U(16),
		            "Anade \"Demon's Souls\" a la biblioteca de Steam como juego externo (con icono e imagenes). Mejor con Steam cerrado; si esta abierto, reinicialo para que aparezca.")) {
			g_steam_busy = true;
			{
				std::scoped_lock lock(g_steam_mutex);
				g_steam_text = I18n::T("Añadiendo a Steam...");
			}
			std::thread([] {
				std::string report;
				const bool  ok = Steam::RunAddToSteamScript(g_rt.exe_path, &report) || Steam::AddToSteam(g_rt.exe_path, &report);
				std::scoped_lock lock(g_steam_mutex);
				g_steam_text = (ok ? "OK: " : "ERROR: ") + report;
				g_steam_busy = false;
			}).detach();
		}
		ImGui::EndDisabled();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), by + U(38)});
		ImGui::PushTextWrapPos(g_lbox.at.x + g_lbox.w - U(12) - ImGui::GetWindowPos().x);
		{
			std::scoped_lock lock(g_steam_mutex);
			if (!g_steam_text.empty()) ImGui::TextColored(kInkL, "%s", g_steam_text.c_str());
		}
		if (!g_status.empty()) ImGui::TextColored(kGoldL, "%s", I18n::T(g_status.c_str()));
		ImGui::PopTextWrapPos();
		const float end = ImGui::GetCursorScreenPos().y;
		RowEnd(y, (end - y) / g_unit + 2);
	}
	{
		// (ds-pc-port9) "Automatico": Windows' UI language at every start (settings.ini idioma_interfaz empty).
		std::vector<Choice> langs;
		langs.push_back({"", "Automático (idioma de Windows)"});
		for (int i = 0; i < I18n::kMenuLanguageCount; i++) langs.push_back({I18n::kMenuLanguages[i], I18n::kMenuLanguageNames[i]});
		if (LChoice("Idioma del menu", langs.data(), static_cast<int>(langs.size()), s.menu_language,
		            "Idioma de este lanzador y de los textos del port en el juego. Automático: el de Windows.", 150, 256)) {
			const std::string ui = s.menu_language.empty() ? I18n::DetectWindowsLanguage() : s.menu_language;
			I18n::SetLanguage(ui);
			// (The game's language follows the menu's until it is chosen on its own.)
			if (!s.game_language_set) s.language = I18n::GameLanguageFor(ui);
		}
	}
	{
		std::string lang = std::to_string(s.language);
		if (LChoice("Idioma", kLanguages, kLanguageCount, lang, "Idioma del juego (textos y voces). Sigue al del menú hasta que elijas otro.", 150, 256)) {
			s.language          = std::atoi(lang.c_str());
			s.game_language_set = true;
		}
	}
	LCheckRow("Copia de las partidas al arrancar", s.backup_saves, "Copia _SaveData en _SaveData-copias cada vez que pulsas JUGAR.");
	// (ds-pc-port37) "Continuar donde lo dejé (F2)" removed (the owner asked).
	{
		const float y = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), y + U(2)});
		ImGui::TextColored(kDimL, I18n::T("Versión instalada: %s"), Update::CurrentVersion());
		RowEnd(y, 26);
	}
	BoxEnd();
}

// Actualizar version / Volver a la version anterior: their state, inside the options (the network and the copy on a
// worker thread).
void LauncherUpdateBox() {
	UpdateUi::State state;
	std::string     text;
	{
		std::scoped_lock lock(g_upd.mutex);
		state = g_upd.state;
		text  = g_upd.text;
	}
	if (state == UpdateUi::Idle) return;
	BoxBegin(state == UpdateUi::Choosing || state == UpdateUi::Confirming ? "Volver a la version anterior" : "Actualizar version");
	const float y = RowY();
	ImGui::SetCursorScreenPos({g_lbox.at.x + U(12), y});
	ImGui::BeginGroup();
	ImGui::PushItemWidth(U(396));
	ImGui::PushTextWrapPos(g_lbox.at.x + g_lbox.w - U(12) - ImGui::GetWindowPos().x);
	if (!text.empty()) ImGui::TextColored(kInkL, "%s", text.c_str());
	ImGui::PopTextWrapPos();
	const bool  busy = g_upd.busy.load();
	const float bh   = U(34);
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {U(8), U(6)});
	if (state == UpdateUi::Checking || (state == UpdateUi::Working && g_upd.total.load() <= 0))
		ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), {U(396), U(10)}, "");
	if (state == UpdateUi::Available) {
		ImGui::BeginDisabled(busy);
		if (ImGui::Button((std::string(I18n::T("Actualizar a ")) + g_upd.release.tag).c_str(), {U(260), bh})) StartWorker(InstallWork);
		ImGui::EndDisabled();
		ImGui::SameLine();
	}
	if (state == UpdateUi::Working && g_upd.total.load() > 0 && text.rfind("Descargando", 0) == 0) {
		const float f = static_cast<float>(static_cast<double>(g_upd.done.load()) / static_cast<double>(g_upd.total.load()));
		char        label[64];
		std::snprintf(label, sizeof(label), "%.0f / %.0f MB", g_upd.done.load() / 1048576.0, g_upd.total.load() / 1048576.0);
		ImGui::ProgressBar(f, {U(396), 0}, label);
		if (ImGui::Button(I18n::T("Cancelar"), {U(160), bh})) g_upd.cancel = true;
	}
	if (state == UpdateUi::Choosing || state == UpdateUi::Confirming) {
		std::scoped_lock lock(g_upd.mutex);
		for (int i = 0; i < static_cast<int>(g_upd.backups.size()); i++) {
			if (ImGui::Selectable(g_upd.backups[i].version.c_str(), g_upd.chosen == i, ImGuiSelectableFlags_None, {U(396), 0})) {
				g_upd.chosen = i;
				g_upd.state  = UpdateUi::Confirming;
				g_upd.text   = "Se restaura la version " + g_upd.backups[i].version +
				             ". Los archivos actuales del motor se guardan antes en _backup_update (se puede deshacer). La cache de shaders, las partidas, los seeds y los ajustes no se tocan.";
			}
		}
	}
	if (state == UpdateUi::Confirming) {
		ImGui::BeginDisabled(busy);
		if (ImGui::Button(I18n::T("Restaurar"), {U(180), bh})) StartWorker(RestoreWork);
		ImGui::EndDisabled();
		ImGui::SameLine();
	}
	if (state != UpdateUi::Working && state != UpdateUi::Checking) {
		if (ImGui::Button(I18n::T("Cerrar"), {U(140), bh})) SetUpd(UpdateUi::Idle, "");
	}
	ImGui::PopStyleVar();
	ImGui::PopItemWidth();
	ImGui::EndGroup();
	const float end = ImGui::GetItemRectMax().y;
	RowEnd(y, (end - y) / g_unit + 4);
	BoxEnd();
}

// ---- (trofeos-exe) TROFEOS at the launcher's bottom left: a tab (the cup, TROFEOS in Cinzel, earned / total, the
// percent by PS5 points and a fine progress line) that unfolds a panel upwards (a curtain revealed from the tab, the
// rows entering one after another, a gold glow on the row under the mouse or the pad's focus). Each trophy: its icon
// (trophy-icons/tropNNN.png; greyed when not earned), its name in Cinzel, its grade and the date and time it was earned
// (trophies.json keeps the unix time of each unlock); hidden ones not earned as "Trofeo oculto". Clicking or pressing A
// on a row unfolds its description. Data: LoadTrophyRows (trophies-list.json, the same as F2), read once when the
// launcher opens (two small JSON files) and again each time the panel opens; the icons are decoded and scaled on a
// worker thread the first time the panel opens, into one atlas texture (colour and grey versions), so the launcher's
// start costs nothing more. Pad: View / Share opens and closes it, the D-pad moves, A unfolds, B closes; Esc closes. ----
namespace TrophyPage {
std::string DateText(int64_t t); // ("Obtenido el 18/11/2026 20:43", below)
} // namespace TrophyPage

namespace LauncherTrophy {

struct Row {
	TrophyRow t;
	int64_t   when = 0; // unix time of the unlock (0: unknown)
};

struct IconJob {
	std::mutex            mutex;
	bool                  done = false;
	std::atomic<bool>     cancel {false};
	int                   cell = 0, pad = 2, stride = 0, cols = 0, width = 0, height = 0;
	std::vector<uint8_t>  pixels; // RGBA8
	std::map<int, int>    slots;  // trophy id -> slot (colour at 2 * slot, grey at 2 * slot + 1)
};

std::vector<Row>         g_rows;
int                      g_earned = 0, g_pct = 0;
int                      g_game_total = 0;    // (rando1) the game's trophies only (the "X / 38"): the randomizer's apart
int                      g_rando_first = -1;  // (rando1) index of the first "Trofeos Randomizer" row (they go last), -1 none
int                      g_grade_have[5] = {}, g_grade_all[5] = {};
bool                     g_loaded = false;
std::filesystem::path    g_dir;
bool                     g_open = false;
double                   g_changed_at = -100.0, g_opened_at = -100.0;
float                    g_k = 0.0f, g_k_from = 0.0f; // 0 closed .. 1 open (eased)
int                      g_sel = -1;                 // the row whose description is unfolded
std::map<int, float>     g_expand;
bool                     g_focus_panel = false, g_focus_first = false, g_focus_back = false, g_scroll_top = false;
std::shared_ptr<IconJob> g_job;
std::thread              g_worker;
ImTextureData*           g_atlas = nullptr;
std::shared_ptr<IconJob> g_art; // g_atlas' geometry and slots
double                   g_atlas_at = 0.0;
ImTextureID              g_linear_id = ImTextureID_Invalid;

ImU32 Ca(int r, int g, int b, float a) {
	return IM_COL32(r, g, b, static_cast<int>(std::clamp(a, 0.0f, 1.0f) * 255.0f + 0.5f));
}

float Px(float v) {
	return std::round(U(v));
}

std::filesystem::path FindDir() {
	std::error_code       ec;
	std::filesystem::path best;
	for (const auto& d: std::filesystem::directory_iterator("_SaveData", ec)) {
		const auto f = d.path() / "trophies-list.json";
		if (std::filesystem::exists(f, ec) && (best.empty() || f < best)) best = f;
	}
	return best.empty() ? best : best.parent_path();
}

void Reload() {
	g_loaded = true;
	g_dir    = FindDir();
	std::map<int, int64_t> times;
	if (!g_dir.empty()) {
		if (std::ifstream f(g_dir / "trophies.json", std::ios::binary); f) {
			const auto j = nlohmann::json::parse(f, nullptr, false);
			if (j.is_object() && j.contains("unlocked") && j["unlocked"].is_object())
				for (const auto& [k, v]: j["unlocked"].items()) times[std::atoi(k.c_str())] = v.is_number() ? v.get<int64_t>() : 0;
		}
	}
	auto rows = LoadTrophyRows();
	g_rows.clear();
	g_rows.reserve(rows.size());
	for (auto& r: rows) {
		if (Libs::Trophy::IsRando(r.id)) continue; // (rando-oculto 11-23) the randomizer's trophies hidden from the launcher
		Row x;
		if (r.earned)
			if (auto it = times.find(r.id); it != times.end()) x.when = it->second;
		x.t = std::move(r);
		g_rows.push_back(std::move(x));
	}
	// Earned first, the most recent on top; then the rest in the game's order. (rando1) The randomizer's trophies after
	// all the game's, in their own section.
	std::stable_sort(g_rows.begin(), g_rows.end(), [](const Row& a, const Row& b) {
		const bool ra = Libs::Trophy::IsRando(a.t.id), rb = Libs::Trophy::IsRando(b.t.id);
		if (ra != rb) return rb;
		if (a.t.earned != b.t.earned) return a.t.earned;
		if (a.t.earned) return a.when > b.when;
		return a.t.id < b.t.id;
	});
	auto points = [](int g) { return g == 1 ? 300 : g == 2 ? 90 : g == 3 ? 30 : g == 4 ? 15 : 0; };
	int  total = 0, have = 0;
	g_earned   = 0;
	for (int g = 0; g < 5; g++) g_grade_have[g] = g_grade_all[g] = 0;
	g_game_total  = 0;
	g_rando_first = -1;
	for (int i = 0; i < static_cast<int>(g_rows.size()); i++)
		if (Libs::Trophy::IsRando(g_rows[static_cast<size_t>(i)].t.id)) {
			g_rando_first = i;
			break;
		}
	for (const auto& r: g_rows) {
		if (Libs::Trophy::IsRando(r.t.id)) continue; // (rando1) not in the game's count nor percent
		g_game_total++;
		const int g = r.t.grade >= 1 && r.t.grade <= 4 ? r.t.grade : 0;
		total += points(g);
		g_grade_all[g]++;
		if (r.t.earned) {
			have += points(g);
			g_earned++;
			g_grade_have[g]++;
		}
	}
	g_pct = total > 0 ? have * 100 / total : (g_game_total == 0 ? 0 : g_earned * 100 / g_game_total);
}

// One icon into its cell: an area average (premultiplied, so the edges keep no dark fringe), then the gutter.
void PutIcon(IconJob& job, int slot, const unsigned char* src, int sw, int sh, bool grey) {
	const int c  = job.cell;
	const int x0 = (slot % job.cols) * job.stride + job.pad, y0 = (slot / job.cols) * job.stride + job.pad;
	for (int y = 0; y < c; y++) {
		const float fy0 = static_cast<float>(y) * sh / c, fy1 = static_cast<float>(y + 1) * sh / c;
		for (int x = 0; x < c; x++) {
			const float fx0 = static_cast<float>(x) * sw / c, fx1 = static_cast<float>(x + 1) * sw / c;
			float       acc[4] = {0, 0, 0, 0}, wsum = 0;
			for (int sy = static_cast<int>(fy0); sy < std::min(sh, static_cast<int>(std::ceil(fy1))); sy++) {
				const float wy = std::min(fy1, sy + 1.0f) - std::max(fy0, static_cast<float>(sy));
				for (int sx = static_cast<int>(fx0); sx < std::min(sw, static_cast<int>(std::ceil(fx1))); sx++) {
					const float          w = wy * (std::min(fx1, sx + 1.0f) - std::max(fx0, static_cast<float>(sx)));
					const unsigned char* p = src + (static_cast<size_t>(sy) * sw + sx) * 4;
					const float          a = p[3] / 255.0f;
					acc[0] += p[0] * a * w;
					acc[1] += p[1] * a * w;
					acc[2] += p[2] * a * w;
					acc[3] += a * w;
					wsum += w;
				}
			}
			float r = 0, g = 0, b = 0, a = 0;
			if (wsum > 0 && acc[3] > 0) {
				r = acc[0] / acc[3];
				g = acc[1] / acc[3];
				b = acc[2] / acc[3];
				a = acc[3] / wsum;
			}
			if (grey) {
				// Not earned: desaturated and darkened, a little warm (old stone).
				const float l = (0.30f * r + 0.59f * g + 0.11f * b) * 0.58f;
				r = l * 1.04f;
				g = l;
				b = l * 0.90f;
			}
			unsigned char* d = job.pixels.data() + (static_cast<size_t>(y0 + y) * job.width + x0 + x) * 4;
			d[0] = static_cast<unsigned char>(std::clamp(r, 0.0f, 255.0f));
			d[1] = static_cast<unsigned char>(std::clamp(g, 0.0f, 255.0f));
			d[2] = static_cast<unsigned char>(std::clamp(b, 0.0f, 255.0f));
			d[3] = static_cast<unsigned char>(std::clamp(a * 255.0f, 0.0f, 255.0f));
		}
	}
	// The gutter: the cell's edge pixels repeated (the linear filter never reads a neighbour).
	for (int y = -job.pad; y < c + job.pad; y++)
		for (int x = -job.pad; x < c + job.pad; x++) {
			if (x >= 0 && x < c && y >= 0 && y < c) continue;
			const int      cx = std::clamp(x, 0, c - 1), cy = std::clamp(y, 0, c - 1);
			const uint8_t* s  = job.pixels.data() + (static_cast<size_t>(y0 + cy) * job.width + x0 + cx) * 4;
			uint8_t*       d  = job.pixels.data() + (static_cast<size_t>(y0 + y) * job.width + x0 + x) * 4;
			std::memcpy(d, s, 4);
		}
}

// The worker: every trophy's PNG decoded and scaled to the size it is drawn at (colour and grey), in one atlas.
void BuildIcons(std::shared_ptr<IconJob> job, std::filesystem::path dir, std::vector<int> ids) {
	const int n     = static_cast<int>(ids.size());
	job->stride     = job->cell + 2 * job->pad;
	job->cols       = std::max(1, static_cast<int>(std::ceil(std::sqrt(2.0 * std::max(1, n)))));
	const int lines = (2 * std::max(1, n) + job->cols - 1) / job->cols;
	job->width      = job->cols * job->stride;
	job->height     = lines * job->stride;
	job->pixels.assign(static_cast<size_t>(job->width) * job->height * 4, 0);
	int loaded = 0;
	for (int i = 0; i < n && !job->cancel.load(); i++) {
		if (Libs::Trophy::IsRando(ids[i])) { // (rando1) the randomizer's medal, drawn in code
			std::vector<uint8_t> medal;
			Libs::Trophy::RandoMedal(256, ids[i], &medal); // (rando3) each trophy its own medal
			PutIcon(*job, 2 * i, medal.data(), 256, 256, false);
			PutIcon(*job, 2 * i + 1, medal.data(), 256, 256, true);
			job->slots[ids[i]] = i;
			loaded++;
			continue;
		}
		std::string bytes;
		for (int pick: {ids[i], ids[i] == Libs::Trophy::kNexusTrophyId ? 1 : -1}) {
			if (pick < 0) break;
			char name[32];
			std::snprintf(name, sizeof(name), "trop%03d.png", pick);
			std::ifstream in(dir / "trophy-icons" / name, std::ios::binary);
			if (!in) continue;
			bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
			if (!bytes.empty()) break;
		}
		if (bytes.empty()) continue;
		int  w = 0, h = 0, ch = 0;
		auto px = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()), &w, &h, &ch, 4);
		if (px == nullptr || w <= 0 || h <= 0) {
			if (px != nullptr) stbi_image_free(px);
			continue;
		}
		PutIcon(*job, 2 * i, px, w, h, false);
		PutIcon(*job, 2 * i + 1, px, w, h, true);
		stbi_image_free(px);
		job->slots[ids[i]] = i;
		loaded++;
	}
	std::printf("[port] launcher trophies: %d of %d icons loaded (%dx%d atlas)\n", loaded, n, job->width, job->height);
	std::fflush(stdout);
	std::scoped_lock lock(job->mutex);
	job->done = true;
}

void EnsureIcons() {
	if (g_atlas != nullptr) {
		// (the SDL renderer's texture: bilinear; set again if the texture was created again after a device reset)
		if (g_atlas->Status == ImTextureStatus_OK && g_atlas->GetTexID() != ImTextureID_Invalid && g_atlas->GetTexID() != g_linear_id) {
			SDL_SetTextureScaleMode(reinterpret_cast<SDL_Texture*>(static_cast<intptr_t>(g_atlas->GetTexID())), SDL_ScaleModeLinear);
			g_linear_id = g_atlas->GetTexID();
		}
		return;
	}
	if (g_job == nullptr) {
		if (g_art != nullptr || g_dir.empty() || g_rows.empty()) return; // (g_art without an atlas: no icon files, the cups)
		std::vector<int> ids;
		for (const auto& r: g_rows) ids.push_back(r.t.id);
		g_job       = std::make_shared<IconJob>();
		g_job->cell = std::clamp(static_cast<int>(Px(54)), 32, 192);
		g_worker    = std::thread(BuildIcons, g_job, g_dir, std::move(ids));
		return;
	}
	{
		std::scoped_lock lock(g_job->mutex);
		if (!g_job->done) return;
	}
	if (g_worker.joinable()) g_worker.join();
	auto job = std::move(g_job);
	g_job    = nullptr;
	if (job->slots.empty() || job->width <= 0 || job->height <= 0) {
		g_art = job; // (no icon: the cups are drawn; not tried again)
		g_art->pixels.clear();
		g_atlas = nullptr;
		return;
	}
	auto* tex = IM_NEW(ImTextureData)();
	tex->Create(ImTextureFormat_RGBA32, job->width, job->height);
	std::memcpy(tex->GetPixels(), job->pixels.data(), job->pixels.size());
	tex->UseColors = true;
	ImGui::RegisterUserTexture(tex);
	job->pixels.clear();
	job->pixels.shrink_to_fit();
	g_atlas    = tex;
	g_art      = std::move(job);
	g_atlas_at = ImGui::GetTime();
}

// The launcher closes (before the renderer goes): the worker stopped, the atlas freed, everything as at the start.
void Shutdown() {
	if (g_job != nullptr) g_job->cancel = true;
	if (g_worker.joinable()) g_worker.join();
	g_job = nullptr;
	FreeTexture(g_atlas);
	g_art.reset();
	g_linear_id = ImTextureID_Invalid;
	g_rows.clear();
	g_expand.clear();
	g_loaded = g_open = false;
	g_k = g_k_from = 0.0f;
	g_sel        = -1;
	g_changed_at = g_opened_at = -100.0;
	g_focus_panel = g_focus_first = g_focus_back = false;
}

bool IsOpen() {
	return g_open;
}

void Open(bool by_pad) {
	if (g_open) return;
	Reload();
	g_k_from      = g_k;
	g_open        = true;
	g_changed_at  = ImGui::GetTime();
	if (g_k < 0.05f) { // (the rows enter again, from the top, only from closed)
		g_opened_at  = g_changed_at;
		g_scroll_top = true;
	}
	g_focus_panel = true;
	g_focus_first = by_pad;
}

void Close(bool by_pad) {
	if (!g_open) return;
	g_k_from     = g_k;
	g_open       = false;
	g_changed_at = ImGui::GetTime();
	g_focus_back = by_pad;
}

void Toggle(bool by_pad) {
	if (g_open) Close(by_pad);
	else Open(by_pad);
}

// Text with extra letter spacing (the inscriptions): drawn glyph by glyph. Returns its width.
float Spaced(ImDrawList* dl, ImFont* font, float size, ImVec2 at, ImU32 col, const char* text, float spacing) {
	if (font == nullptr) font = ImGui::GetFont();
	float       x = at.x;
	const char* p = text;
	const char* e = text + std::strlen(text);
	while (p < e) {
		unsigned int c = 0;
		const int    n = ImTextCharFromUtf8(&c, p, e);
		if (n <= 0) break;
		if (dl != nullptr) dl->AddText(font, size, {std::floor(x), at.y}, col, p, p + n);
		x += font->CalcTextSizeA(size, FLT_MAX, 0.0f, p, p + n).x + spacing;
		p += n;
	}
	return x - at.x - (p > text ? spacing : 0.0f);
}

// The text cut with "..." to fit `max_w`.
std::string Fit(ImFont* font, float size, const std::string& text, float max_w) {
	if (font == nullptr) font = ImGui::GetFont();
	if (font->CalcTextSizeA(size, FLT_MAX, 0.0f, text.c_str()).x <= max_w) return text;
	const float dots = font->CalcTextSizeA(size, FLT_MAX, 0.0f, "...").x;
	std::string out  = text;
	while (!out.empty()) {
		size_t cut = out.size() - 1;
		while (cut > 0 && (static_cast<unsigned char>(out[cut]) & 0xC0) == 0x80) cut--; // (a whole UTF-8 character)
		out.resize(cut);
		while (!out.empty() && out.back() == ' ') out.pop_back();
		if (font->CalcTextSizeA(size, FLT_MAX, 0.0f, out.c_str()).x + dots <= max_w) break;
	}
	return out + "...";
}

void Lozenge(ImDrawList* dl, ImVec2 c, float r, ImU32 col) {
	dl->AddQuadFilled({c.x, c.y - r}, {c.x + r, c.y}, {c.x, c.y + r}, {c.x - r, c.y}, col);
}

// A chevron (^ at angle 0) centred at c, turned by `angle` radians.
void Chevron(ImDrawList* dl, ImVec2 c, float s, float angle, ImU32 col, float thick) {
	const float cs = std::cos(angle), sn = std::sin(angle);
	const auto  P  = [&](float x, float y) { return ImVec2 {c.x + x * cs - y * sn, c.y + x * sn + y * cs}; };
	const ImVec2 pts[3] = {P(-s, s * 0.5f), P(0, -s * 0.5f), P(s, s * 0.5f)};
	dl->AddPolyline(pts, 3, col, ImDrawFlags_None, thick);
}

// A small padlock (not earned) with its centre at c.
void Padlock(ImDrawList* dl, ImVec2 c, float s, float a) {
	dl->AddCircleFilled(c, s * 1.05f, Ca(8, 8, 10, 0.92f * a));
	dl->AddCircle(c, s * 1.05f, Ca(150, 136, 110, 0.65f * a), 0, std::max(1.0f, s * 0.10f));
	const ImU32 col = Ca(196, 184, 160, a);
	dl->PathArcTo({c.x, c.y - s * 0.12f}, s * 0.30f, 3.14159265f, 2.0f * 3.14159265f, 10);
	dl->PathStroke(col, ImDrawFlags_None, std::max(1.0f, s * 0.13f));
	dl->AddLine({c.x - s * 0.30f, c.y - s * 0.12f}, {c.x - s * 0.30f, c.y + s * 0.05f}, col, std::max(1.0f, s * 0.13f));
	dl->AddLine({c.x + s * 0.30f, c.y - s * 0.12f}, {c.x + s * 0.30f, c.y + s * 0.05f}, col, std::max(1.0f, s * 0.13f));
	dl->AddRectFilled({c.x - s * 0.46f, c.y + s * 0.02f}, {c.x + s * 0.46f, c.y + s * 0.56f}, col, s * 0.10f);
}

// The panel's reveal: every command clipped to below `top` (the curtain rising from the tab) and the vertices faded.
void Reveal(ImDrawList* dl, int first_vtx, float top, float alpha) {
	if (dl == nullptr) return;
	for (auto& cmd: dl->CmdBuffer) {
		cmd.ClipRect.y = std::max(cmd.ClipRect.y, top);
		if (cmd.ClipRect.w < cmd.ClipRect.y) cmd.ClipRect.w = cmd.ClipRect.y;
	}
	if (alpha >= 0.999f) return;
	for (int i = first_vtx; i < dl->VtxBuffer.Size; i++) {
		ImU32&      c = dl->VtxBuffer[i].col;
		const float a = static_cast<float>((c >> IM_COL32_A_SHIFT) & 0xFF) * alpha;
		c             = (c & ~IM_COL32_A_MASK) | (static_cast<ImU32>(a + 0.5f) << IM_COL32_A_SHIFT);
	}
}

const char* GradeName(int g) {
	static const char* names[5] = {"", "Platino", "Oro", "Plata", "Bronce"};
	return g >= 1 && g <= 4 ? I18n::T(names[g]) : "";
}

// The tab (always) and, open or closing, the panel above it. column_x: the left edge of the launcher's column.
// (tiles-fit) The four bottom tiles (TROFEOS, ONLINE, SIN TRUCOS, RANDOMIZER) share the room left of the right-hand
// column (JUGAR / Opciones): widths 400/300/300/300 with Px(14) gaps, shrunk evenly when they would reach the column.
float g_tiles_limit = 0.0f; // the right column's left x (set by Draw each frame); 0 = no limit
void TileRect(int i, float& tx, float& tw) {
	const float gap = Px(14), x0 = Px(16);
	const float base[4] = {Px(400), Px(300), Px(300), Px(300)};
	const float need  = base[0] + base[1] + base[2] + base[3];
	const float avail = (g_tiles_limit > 0.0f ? g_tiles_limit - gap : 1e9f) - x0 - gap * 3;
	const float k     = avail < need ? std::max(avail, 0.0f) / need : 1.0f;
	tx = x0;
	for (int j = 0; j < i; j++) tx += std::floor(base[j] * k) + gap;
	tx = std::round(tx);
	tw = std::floor(base[i] * k);
}

void Draw(float width, float height, float column_x) {
	if (!g_loaded) Reload();
	auto&        io  = ImGui::GetIO();
	const double now = ImGui::GetTime();
	if (!g_popup.load()) {
		if (ImGui::IsKeyPressed(ImGuiKey_GamepadBack, false)) Toggle(true);
		else if (g_open && ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false)) Close(true);
	}
	{
		const float t = static_cast<float>((now - g_changed_at) / (g_open ? 0.38 : 0.24));
		g_k = g_open ? g_k_from + (1.0f - g_k_from) * Anim::EaseOutQuint(t) : g_k_from * (1.0f - Anim::EaseOut(t));
	}
	if (g_open || g_k > 0.0f) EnsureIcons();

	ImFont*      cinzel = g_font_display != nullptr ? g_font_display : ImGui::GetFont();
	ImFont*      serif  = g_font_serif != nullptr ? g_font_serif : ImGui::GetFont();
	ImFont*      semi   = g_font_semi != nullptr ? g_font_semi : ImGui::GetFont();
	const ImU32  gold   = IM_COL32(222, 196, 140, 255);
	g_tiles_limit = column_x;
	float tx, tw;
	TileRect(0, tx, tw);
	const float th = Px(60);
	const float  ty = std::round(height - U(34) - th);
	const float  frac = static_cast<float>(g_pct) / 100.0f;

	// ---- The tab ----
	bool tab_hovered = false;
	ImGui::SetNextWindowPos({tx, ty});
	ImGui::SetNextWindowSize({tw, th});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::Begin("##trophy-tab", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
	                                          ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse |
	                                          ImGuiWindowFlags_NoFocusOnAppearing);
	ImGui::PopStyleVar();
	{
		ImGui::SetCursorScreenPos({tx, ty});
		const bool clicked = ImGui::InvisibleButton("##trophy-tab-button", {tw, th}, ImGuiButtonFlags_EnableNav);
		tab_hovered        = ImGui::IsItemHovered();
		if (tab_hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		const bool  on  = tab_hovered || ImGui::IsItemFocused() || ImGui::IsItemActive();
		const float hot = std::max(Anim::Hot(ImGui::GetItemID(), on), g_k * 0.6f);
		Tip("Tus trofeos: los obtenidos con su fecha y hora, y los que faltan. En el mando, el bot\xC3\xB3n View / Share los abre y los cierra.");
		if (clicked) Toggle(!ImGui::IsMouseReleased(ImGuiMouseButton_Left));
		auto*        dl = ImGui::GetWindowDrawList();
		const ImVec2 a {tx, ty}, b {tx + tw, ty + th};
		const float  rr = Px(6);
		dl->PushClipRect({0, 0}, {width, height}, false);
		for (int k = 1; k <= 3; k++) { // the glow around it
			const float g = static_cast<float>(k) * U(2.5f);
			dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g}, IM_COL32(240, 200, 120, static_cast<int>(30.0f / k * hot)), rr + g, 0, U(2));
		}
		dl->PopClipRect();
		dl->AddRectFilled(a, b, IM_COL32(9, 9, 11, 228), rr);
		dl->AddRectFilledMultiColor({a.x + rr * 0.5f, a.y + 1}, {b.x - rr * 0.5f, a.y + th * 0.55f}, Ca(70, 56, 32, 0.30f + 0.25f * hot),
		                            Ca(70, 56, 32, 0.10f), Ca(70, 56, 32, 0.0f), Ca(70, 56, 32, 0.0f));
		dl->AddRect(a, b, Ca(204, 176, 122, 0.45f + 0.45f * hot), rr, 0, std::max(1.0f, U(1.25f)));
		dl->AddRect({a.x + Px(3), a.y + Px(3)}, {b.x - Px(3), b.y - Px(3)}, Ca(204, 176, 122, 0.12f + 0.10f * hot), Px(4), 0, 1.0f);
		// The medallion with the cup.
		const float  mr = std::round(th * 0.34f);
		const ImVec2 mc {std::round(a.x + Px(12) + mr), std::round(a.y + th * 0.5f)};
		dl->AddCircleFilled(mc, mr, IM_COL32(22, 18, 12, 255), 32);
		dl->AddCircle(mc, mr, Ca(222, 196, 140, 0.55f + 0.45f * hot), 32, std::max(1.0f, U(1.4f)));
		dl->AddCircle(mc, mr - Px(3), Ca(222, 196, 140, 0.18f), 32, 1.0f);
		const float cs = std::round(mr * 1.25f);
		Libs::Trophy::DrawCup(dl, mc.x - cs * 0.5f, mc.y - cs * 0.48f, cs, Libs::Trophy::CupColor(g_earned > 0 ? 2 : 0));
		// TROFEOS, then the counter and the percent.
		const float lx = mc.x + mr + Px(16);
		Spaced(dl, cinzel, Px(19), {lx, a.y + Px(8)}, Ca(222, 196, 140, 0.85f + 0.15f * hot), I18n::T("TROFEOS"), U(2.5f));
		char count[64];
		if (!g_rows.empty()) std::snprintf(count, sizeof(count), "%d / %d", g_earned, g_game_total);
		else std::snprintf(count, sizeof(count), "-");
		const float cw = Spaced(dl, cinzel, Px(16), {lx, a.y + Px(32)}, IM_COL32(236, 226, 204, 240), count, U(1));
		if (!g_rows.empty()) {
			char pct[16];
			std::snprintf(pct, sizeof(pct), "%d%%", g_pct);
			Lozenge(dl, {std::round(lx + cw + Px(12)), std::round(a.y + Px(32) + Px(16) * 0.55f)}, Px(3), Ca(204, 176, 122, 0.8f));
			Spaced(dl, cinzel, Px(16), {lx + cw + Px(24), a.y + Px(32)}, gold, pct, U(1));
		}
		// The fine progress line along the bottom.
		const float px0 = lx, px1 = b.x - Px(46), py = std::round(b.y - Px(8));
		dl->AddRectFilled({px0, py}, {px1, py + Px(2)}, IM_COL32(255, 255, 255, 28));
		if (frac > 0.0f)
			dl->AddRectFilledMultiColor({px0, py}, {px0 + (px1 - px0) * frac, py + Px(2)}, IM_COL32(150, 112, 52, 255), IM_COL32(240, 206, 136, 255),
			                            IM_COL32(240, 206, 136, 255), IM_COL32(150, 112, 52, 255));
		// The chevron: up (the panel opens upwards), turned down when open.
		Chevron(dl, {std::round(b.x - Px(24)), std::round(a.y + th * 0.5f)}, Px(7), 3.14159265f * g_k, Ca(222, 196, 140, 0.7f + 0.3f * hot),
		        std::max(1.5f, U(1.8f)));
	}
	ImGui::End();

	// ---- The panel ----
	if (!g_open && g_k <= 0.0f) {
		if (g_focus_back) {
			g_focus_back = false;
			ImGui::SetWindowFocus("##launcher-bottom");
		}
		return;
	}
	const float pw  = std::clamp(column_x - tx - Px(24), Px(340), Px(580));
	const float pb  = std::round(ty - Px(10));
	const float ph  = std::round(std::min(U(720), pb - Px(18)));
	const float pt  = pb - ph;
	const bool  closing = !g_open;
	if (g_focus_panel) ImGui::SetNextWindowFocus();
	ImGui::SetNextWindowPos({tx, pt});
	ImGui::SetNextWindowSize({pw, ph});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::Begin("##trophy-panel", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
	                                            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse |
	                                            ImGuiWindowFlags_NoFocusOnAppearing | (closing ? ImGuiWindowFlags_NoInputs : 0));
	ImGui::PopStyleVar();
	g_focus_panel     = false;
	auto*  dl         = ImGui::GetWindowDrawList();
	const ImVec2 a {tx, pt}, b {tx + pw, pb};
	const float  rr   = Px(8);
	// Shadow, body, the warm light from above, the double gold edge and the corner brackets.
	dl->PushClipRect({0, 0}, {width, height}, false);
	for (int k = 1; k <= 4; k++) {
		const float g = static_cast<float>(k) * U(4);
		dl->AddRectFilled({a.x - g, a.y - g + U(4)}, {b.x + g, b.y + g + U(4)}, IM_COL32(0, 0, 0, 26), rr + g);
	}
	dl->PopClipRect();
	dl->AddRectFilled(a, b, IM_COL32(11, 10, 10, 238), rr);
	dl->AddRectFilledMultiColor({a.x + rr * 0.5f, a.y + 1}, {b.x - rr * 0.5f, a.y + Px(170)}, IM_COL32(64, 50, 28, 92), IM_COL32(64, 50, 28, 60),
	                            IM_COL32(64, 50, 28, 0), IM_COL32(64, 50, 28, 0));
	dl->AddRect(a, b, IM_COL32(204, 176, 122, 160), rr, 0, std::max(1.0f, U(1.25f)));
	dl->AddRect({a.x + Px(5), a.y + Px(5)}, {b.x - Px(5), b.y - Px(5)}, IM_COL32(204, 176, 122, 38), Px(5), 0, 1.0f);
	{
		const float in = Px(10), len = Px(20), th2 = std::max(1.0f, U(1.5f));
		const ImU32 c  = IM_COL32(222, 196, 140, 190);
		const ImVec2 corners[4] = {{a.x + in, a.y + in}, {b.x - in, a.y + in}, {a.x + in, b.y - in}, {b.x - in, b.y - in}};
		for (int k = 0; k < 4; k++) {
			const float sx = (k % 2 == 0) ? 1.0f : -1.0f, sy = (k < 2) ? 1.0f : -1.0f;
			const ImVec2 p = corners[k];
			dl->AddLine(p, {p.x + sx * len, p.y}, c, th2);
			dl->AddLine(p, {p.x, p.y + sy * len}, c, th2);
			Lozenge(dl, {p.x + sx * Px(5), p.y + sy * Px(5)}, Px(2.5f), c);
		}
	}
	// The header: TROFEOS and the percent, how many, the bar, each grade's count, a rule.
	const float hx = a.x + Px(26), hw = pw - Px(52);
	Spaced(dl, cinzel, Px(27), {hx, a.y + Px(20)}, gold, I18n::T("TROFEOS"), U(4));
	{
		char pct[16];
		std::snprintf(pct, sizeof(pct), "%d%%", g_pct);
		const float w = Spaced(nullptr, cinzel, Px(30), {0, 0}, 0, pct, U(1.5f));
		Spaced(dl, cinzel, Px(30), {hx + hw - w, a.y + Px(17)}, IM_COL32(240, 230, 208, 255), pct, U(1.5f));
	}
	{
		char line[128];
		std::snprintf(line, sizeof(line), I18n::T("%d de %d trofeos obtenidos"), g_earned, g_game_total);
		dl->AddText(serif, Px(15), {hx, a.y + Px(58)}, IM_COL32(206, 196, 176, 230), line);
	}
	{
		const float by = std::round(a.y + Px(78)), bh = Px(8);
		const float fill = frac * Anim::EaseOut(static_cast<float>((now - g_opened_at - 0.15) / 0.9));
		dl->AddRectFilled({hx, by}, {hx + hw, by + bh}, IM_COL32(255, 255, 255, 18), bh * 0.5f);
		dl->AddRect({hx, by}, {hx + hw, by + bh}, IM_COL32(204, 176, 122, 70), bh * 0.5f, 0, 1.0f);
		if (fill > 0.0f) {
			const float fx = hx + hw * fill;
			dl->AddRectFilledMultiColor({hx + 1, by + 1}, {fx, by + bh - 1}, IM_COL32(128, 92, 40, 255), IM_COL32(236, 200, 128, 255),
			                            IM_COL32(200, 160, 92, 255), IM_COL32(100, 72, 30, 255));
			// A slow shimmer running along the filled part.
			const float sw = Px(60), sx = hx + std::fmod(static_cast<float>(now) * U(110), hw + sw) - sw;
			dl->PushClipRect({hx, by}, {fx, by + bh}, true);
			dl->AddRectFilledMultiColor({sx, by}, {sx + sw * 0.5f, by + bh}, IM_COL32(255, 240, 200, 0), IM_COL32(255, 240, 200, 90),
			                            IM_COL32(255, 240, 200, 90), IM_COL32(255, 240, 200, 0));
			dl->AddRectFilledMultiColor({sx + sw * 0.5f, by}, {sx + sw, by + bh}, IM_COL32(255, 240, 200, 90), IM_COL32(255, 240, 200, 0),
			                            IM_COL32(255, 240, 200, 0), IM_COL32(255, 240, 200, 90));
			dl->PopClipRect();
			Lozenge(dl, {std::round(fx), by + bh * 0.5f}, Px(6), IM_COL32(250, 226, 170, 255));
			Lozenge(dl, {std::round(fx), by + bh * 0.5f}, Px(3), IM_COL32(120, 86, 36, 255));
		}
	}
	{
		const float gy = std::round(a.y + Px(108)), cell = hw / 4.0f;
		for (int g = 1; g <= 4; g++) {
			const float x = hx + cell * (g - 1);
			Libs::Trophy::DrawCup(dl, x, gy, Px(20), Libs::Trophy::CupColor(g, g_grade_have[g] > 0 ? 1.0f : 0.45f));
			char n[32];
			std::snprintf(n, sizeof(n), "%d/%d", g_grade_have[g], g_grade_all[g]);
			dl->AddText(semi, Px(14), {x + Px(26), gy + Px(2)}, Libs::Trophy::CupColor(g, g_grade_have[g] > 0 ? 1.0f : 0.6f), n);
		}
	}
	{
		const float ry = std::round(a.y + Px(140)) + 0.5f, mid = hx + hw * 0.5f;
		dl->AddRectFilledMultiColor({hx, ry - 0.5f}, {mid, ry + 0.5f}, IM_COL32(204, 176, 122, 0), IM_COL32(204, 176, 122, 170),
		                            IM_COL32(204, 176, 122, 170), IM_COL32(204, 176, 122, 0));
		dl->AddRectFilledMultiColor({mid, ry - 0.5f}, {hx + hw, ry + 0.5f}, IM_COL32(204, 176, 122, 170), IM_COL32(204, 176, 122, 0),
		                            IM_COL32(204, 176, 122, 0), IM_COL32(204, 176, 122, 170));
		Lozenge(dl, {std::round(mid), ry}, Px(4), IM_COL32(222, 196, 140, 220));
	}
	// The list (a child: ImGui scrolls it to the pad's focus; the wheel eases).
	const float  list_top = std::round(a.y + Px(152)), list_bottom = std::round(b.y - Px(40));
	ImDrawList*  list_dl  = nullptr;
	ImGui::SetCursorScreenPos({a.x + Px(12), list_top});
	ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4 {0, 0, 0, 0}); // (the rows draw their own focus glow)
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, Px(2)});
	ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, Px(8));
	if (g_scroll_top) ImGui::SetNextWindowScroll({0.0f, 0.0f}); // (opened from closed: the list from the top)
	g_scroll_top = false;
	ImGui::BeginChild("##trophy-list", {pw - Px(24), list_bottom - list_top}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
	Anim::SmoothScroll();
	list_dl = ImGui::GetWindowDrawList();
	if (g_rows.empty()) {
		const char*  msg = I18n::T("A\xC3\xBAn no hay trofeos: aparecen al jugar.");
		const float  w   = ImGui::GetContentRegionAvail().x;
		const ImVec2 ts  = serif->CalcTextSizeA(Px(17), FLT_MAX, w - Px(20), msg);
		list_dl->AddText(serif, Px(17), {ImGui::GetCursorScreenPos().x + (w - ts.x) * 0.5f, ImGui::GetCursorScreenPos().y + Px(40)},
		                 IM_COL32(206, 196, 176, 230), msg, nullptr, w - Px(20));
		ImGui::Dummy({w, Px(120)});
	}
	const float dt   = std::clamp(io.DeltaTime, 0.0f, 0.1f);
	const float isz  = std::round(U(54));
	const float h0   = Px(80);
	for (int i = 0; i < static_cast<int>(g_rows.size()); i++) {
		const Row&  r      = g_rows[static_cast<size_t>(i)];
		if (i == g_rando_first) { // (rando1) the "Trofeos Randomizer" section's title
			int got = 0, all = 0;
			for (int k = g_rando_first; k < static_cast<int>(g_rows.size()); k++, all++)
				if (g_rows[static_cast<size_t>(k)].t.earned) got++;
			char         title[96];
			std::snprintf(title, sizeof(title), "TROFEOS RANDOMIZER  %d / %d", got, all);
			const ImVec2 hp = ImGui::GetCursorScreenPos();
			const float  hw = ImGui::GetContentRegionAvail().x;
			ImGui::Dummy({hw, Px(48)});
			list_dl->AddLine({hp.x + Px(8), hp.y + Px(14)}, {hp.x + hw - Px(8), hp.y + Px(14)}, IM_COL32(204, 176, 122, 110), 1.0f);
			list_dl->AddText(serif, Px(18), {hp.x + Px(12), hp.y + Px(20)}, IM_COL32(222, 196, 140, 240), title);
		}
		const bool  secret = r.t.hidden && !r.t.earned;
		const int   g      = r.t.grade >= 1 && r.t.grade <= 4 ? r.t.grade : 0;
		const float lw     = ImGui::GetContentRegionAvail().x;
		const float tx0    = Px(12) + isz + Px(18); // the text's left, from the row's
		const float wrap   = std::max(Px(80), lw - tx0 - Px(40));
		const std::string desc = secret ? std::string(I18n::T("Se revelar\xC3\xA1 al obtenerlo.")) : r.t.detail;
		const float desc_size  = Px(16);
		const float desc_h     = desc.empty() ? 0.0f : serif->CalcTextSizeA(desc_size, FLT_MAX, wrap, desc.c_str()).y + Px(14);
		float&      e          = g_expand[r.t.id];
		{
			const float to = g_sel == r.t.id ? 1.0f : 0.0f;
			e += (to - e) * std::min(1.0f, dt * 11.0f);
			if (std::fabs(to - e) < 0.002f) e = to;
		}
		const float  ee = e * e * (3.0f - 2.0f * e); // (smoothstep)
		const float  h  = std::round(h0 + desc_h * ee);
		const ImVec2 at = ImGui::GetCursorScreenPos();
		ImGui::PushID(r.t.id);
		if (g_focus_first && i == 0) ImGui::SetKeyboardFocusHere();
		const bool pressed = ImGui::InvisibleButton("##row", {lw, h}, ImGuiButtonFlags_EnableNav); // (EnableNav: the pad reaches it)
		const bool hovered = ImGui::IsItemHovered();
		const bool on      = hovered || ImGui::IsItemFocused();
		const float hot    = Anim::Hot(ImGui::GetItemID(), on);
		if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		if (pressed && !closing) g_sel = g_sel == r.t.id ? -1 : r.t.id;
		ImGui::PopID();
		if (at.y > list_bottom + Px(10) || at.y + h < list_top - Px(10)) continue; // (out of view: only its space)
		// The entrance: each row a little after the one above (0.045 s), sliding in from the left and fading in.
		const float delay = 0.12f + 0.045f * static_cast<float>(std::min(i, 16));
		const float ein   = Anim::EaseOut(static_cast<float>((now - g_opened_at - delay) / 0.34));
		if (ein <= 0.0f) continue;
		const float  al = ein;
		const float  ox = std::round((1.0f - ein) * -U(28));
		const ImVec2 ra {at.x + ox, at.y}, rb {at.x + ox + lw, at.y + h};
		// The glow (hover / focus) and the selected row's steady light.
		const float lit = std::max(hot, g_sel == r.t.id ? 0.45f : 0.0f);
		if (lit > 0.0f) {
			for (int k = 1; k <= 2; k++) {
				const float gg = static_cast<float>(k) * U(1.5f);
				list_dl->AddRect({ra.x + gg, ra.y + gg}, {rb.x - gg, rb.y - gg}, Ca(240, 200, 120, 0.10f / k * hot * al), Px(6));
			}
			list_dl->AddRectFilledMultiColor(ra, rb, Ca(222, 196, 140, 0.16f * lit * al), Ca(222, 196, 140, 0.03f * lit * al),
			                                 Ca(222, 196, 140, 0.03f * lit * al), Ca(222, 196, 140, 0.16f * lit * al));
			list_dl->AddRect(ra, rb, Ca(236, 206, 146, 0.55f * hot * al), Px(5), 0, std::max(1.0f, U(1.25f)));
			const float bar = (rb.y - ra.y) * (0.5f - 0.40f * lit);
			list_dl->AddRectFilled({ra.x, ra.y + bar}, {ra.x + Px(3), rb.y - bar}, Ca(244, 214, 150, lit * al), Px(2));
		} else if (r.t.earned) {
			list_dl->AddRectFilledMultiColor(ra, {ra.x + lw * 0.5f, rb.y}, Ca(70, 56, 32, 0.22f * al), Ca(70, 56, 32, 0.0f), Ca(70, 56, 32, 0.0f),
			                                 Ca(70, 56, 32, 0.22f * al));
		}
		// The icon in its frame.
		const ImVec2 i0 {std::floor(ra.x + Px(12)), std::floor(ra.y + (h0 - isz) * 0.5f)}, i1 {i0.x + isz, i0.y + isz};
		list_dl->AddRectFilled({i0.x - Px(3), i0.y - Px(3)}, {i1.x + Px(3), i1.y + Px(3)}, Ca(6, 6, 8, al), Px(4));
		if (r.t.earned)
			list_dl->AddRect({i0.x - Px(3), i0.y - Px(3)}, {i1.x + Px(3), i1.y + Px(3)}, Libs::Trophy::CupColor(g, (0.75f + 0.25f * hot) * al), Px(4), 0,
			                 std::max(1.0f, U(1.5f)));
		else list_dl->AddRect({i0.x - Px(3), i0.y - Px(3)}, {i1.x + Px(3), i1.y + Px(3)}, Ca(110, 102, 88, 0.6f * al), Px(4), 0, 1.0f);
		bool drew_icon = false;
		if (!secret && g_atlas != nullptr && g_art != nullptr) {
			if (auto it = g_art->slots.find(r.t.id); it != g_art->slots.end()) {
				const int    slot = 2 * it->second + (r.t.earned ? 0 : 1);
				const float  u0   = static_cast<float>((slot % g_art->cols) * g_art->stride + g_art->pad) / g_art->width;
				const float  v0   = static_cast<float>((slot / g_art->cols) * g_art->stride + g_art->pad) / g_art->height;
				const float  u1   = u0 + static_cast<float>(g_art->cell) / g_art->width;
				const float  v1   = v0 + static_cast<float>(g_art->cell) / g_art->height;
				const float  fade = std::clamp(static_cast<float>((now - g_atlas_at) / 0.3), 0.0f, 1.0f);
				const float  lum  = r.t.earned ? 1.0f : 0.85f + 0.15f * hot;
				list_dl->AddImage(g_atlas->GetTexRef(), i0, i1, {u0, v0}, {u1, v1}, Ca(static_cast<int>(255 * lum), static_cast<int>(255 * lum),
				                                                                      static_cast<int>(255 * lum), fade * al));
				drew_icon = fade >= 1.0f;
			}
		}
		if (secret) {
			const ImVec2 qs = cinzel->CalcTextSizeA(Px(30), FLT_MAX, 0.0f, "?");
			list_dl->AddText(cinzel, Px(30), {std::round(i0.x + (isz - qs.x) * 0.5f), std::round(i0.y + (isz - qs.y) * 0.5f)}, Ca(150, 140, 120, al), "?");
		} else if (!drew_icon) {
			const float cs = std::round(isz * 0.62f);
			Libs::Trophy::DrawCup(list_dl, i0.x + (isz - cs) * 0.5f, i0.y + (isz - cs) * 0.5f, cs, Libs::Trophy::CupColor(r.t.earned ? g : 0, al * 0.9f));
		}
		if (!r.t.earned && !secret) Padlock(list_dl, {i1.x - Px(2), i1.y - Px(2)}, Px(9), al);
		// The name, the grade and the date.
		const float  nx    = ra.x + tx0;
		const float  ns    = Px(18);
		const std::string name = Fit(cinzel, ns, secret ? std::string(I18n::T("Trofeo oculto")) : r.t.name, lw - tx0 - Px(40));
		list_dl->AddText(cinzel, ns, {nx, ra.y + Px(15)}, r.t.earned ? Ca(240, 230, 208, al) : Ca(150, 144, 134, al), name.c_str());
		{
			const float ly = ra.y + Px(46);
			Lozenge(list_dl, {std::round(nx + Px(4)), std::round(ly + Px(8))}, Px(4), Libs::Trophy::CupColor(g, (r.t.earned ? 1.0f : 0.5f) * al));
			const char* gname = GradeName(g);
			list_dl->AddText(semi, Px(14), {nx + Px(14), ly}, Libs::Trophy::CupColor(g, (r.t.earned ? 1.0f : 0.6f) * al), gname);
			const float gw   = semi->CalcTextSizeA(Px(14), FLT_MAX, 0.0f, gname).x;
			const std::string when = r.t.earned ? TrophyPage::DateText(r.when) : std::string(I18n::T("Sin obtener"));
			list_dl->AddText(semi, Px(14), {nx + Px(14) + gw + Px(8), ly}, Ca(150, 140, 120, al), "\xC2\xB7");
			list_dl->AddText(serif, Px(14), {nx + Px(14) + gw + Px(20), ly}, r.t.earned ? Ca(206, 196, 176, al) : Ca(130, 124, 114, al), when.c_str());
		}
		Chevron(list_dl, {std::round(rb.x - Px(20)), std::round(ra.y + h0 * 0.5f)}, Px(5), 3.14159265f * (1.0f - ee),
		        Ca(222, 196, 140, (0.35f + 0.6f * std::max(hot, ee)) * al), std::max(1.2f, U(1.4f)));
		// The description, unfolding under the row.
		if (ee > 0.01f && !desc.empty()) {
			const float dy = ra.y + h0 - Px(4);
			list_dl->PushClipRect({ra.x, ra.y}, {rb.x, rb.y}, true);
			list_dl->AddRectFilled({nx - Px(10), dy}, {nx - Px(9), dy + desc_h - Px(12)}, Ca(204, 176, 122, 0.6f * ee * al));
			list_dl->AddText(serif, desc_size, {nx, dy + (1.0f - ee) * -Px(6)}, secret ? Ca(170, 160, 140, ee * al) : Ca(224, 214, 194, ee * al),
			                 desc.c_str(), nullptr, wrap);
			list_dl->PopClipRect();
		}
		// A fine rule under the row.
		list_dl->AddRectFilledMultiColor({ra.x + Px(10), rb.y + Px(1)}, {rb.x - Px(10), rb.y + Px(2)}, Ca(204, 176, 122, 0.22f * al),
		                                 Ca(204, 176, 122, 0.0f), Ca(204, 176, 122, 0.0f), Ca(204, 176, 122, 0.22f * al));
	}
	g_focus_first = false;
	ImGui::EndChild();
	ImGui::PopStyleVar(2);
	ImGui::PopStyleColor(2);
	{
		const char*  hint = I18n::T("Clic o A: ver la descripci\xC3\xB3n   \xC2\xB7   B o Esc: cerrar");
		const ImVec2 hs   = semi->CalcTextSizeA(Px(13), FLT_MAX, 0.0f, hint);
		dl->AddText(semi, Px(13), {std::round(a.x + (pw - hs.x) * 0.5f), std::round(b.y - Px(28))}, IM_COL32(200, 188, 164, 170), hint);
	}
	const bool panel_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
	ImGui::End();
	// A click anywhere else closes it.
	if (g_open && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !panel_hovered && !tab_hovered) Close(false);
	// The reveal: the curtain rises from the tab (and falls back when closing), with a gold edge on its way.
	const float top = pb - ph * g_k;
	const float alpha = std::clamp(g_k * 1.6f, 0.0f, 1.0f);
	Reveal(dl, 0, top, alpha);
	Reveal(list_dl, 0, top, alpha);
	if (g_k < 0.999f) {
		auto*       fg = ImGui::GetForegroundDrawList();
		const float e  = std::sin(3.14159265f * g_k); // (brightest mid-way)
		fg->AddRectFilledMultiColor({a.x, top - Px(10)}, {b.x, top}, Ca(240, 200, 120, 0.0f), Ca(240, 200, 120, 0.0f), Ca(240, 200, 120, 0.22f * e),
		                            Ca(240, 200, 120, 0.22f * e));
		fg->AddLine({a.x + Px(4), top}, {b.x - Px(4), top}, Ca(250, 224, 160, 0.9f * e), std::max(1.0f, U(1.5f)));
	}
	if (closing && g_k <= 0.0f && g_focus_back) {
		g_focus_back = false;
		ImGui::SetWindowFocus("##launcher-bottom");
	}
}

// (online12) ONLINE beside TROFEOS at the launcher's bottom left: same tab look (medallion with a pulsing soul, ONLINE in
// Cinzel); a click opens the "Conexión online" panel (account, password, server state) over the launcher.
bool g_online_open = false;
void DrawOnline(float width, float height) {
	ImFont*     cinzel = g_font_display != nullptr ? g_font_display : ImGui::GetFont();
	const float th = Px(60);
	float tx, tw;
	TileRect(1, tx, tw); // (tiles-fit)
	const float ty = std::round(height - U(34) - th);
	ImGui::SetNextWindowPos({tx, ty});
	ImGui::SetNextWindowSize({tw, th});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::Begin("##online-tab", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
	                                          ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing);
	ImGui::PopStyleVar();
	{
		ImGui::SetCursorScreenPos({tx, ty});
		const bool clicked = ImGui::InvisibleButton("##online-tab-button", {tw, th}, ImGuiButtonFlags_EnableNav);
		const bool hov     = ImGui::IsItemHovered();
		if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		const float hot = Anim::Hot(ImGui::GetItemID(), hov || ImGui::IsItemFocused() || g_online_open);
		Tip("Conexi\xC3\xB3n online: tu cuenta y contrase\xC3\xB1" "a para jugar con tus amigos (mensajes, manchas, fantasmas, invocaciones).");
		if (clicked) g_online_open = !g_online_open;
		auto*        dl = ImGui::GetWindowDrawList();
		const ImVec2 a {tx, ty}, b {tx + tw, ty + th};
		const float  rr = Px(6);
		dl->PushClipRect({0, 0}, {width, height}, false);
		for (int k = 1; k <= 3; k++) {
			const float g = static_cast<float>(k) * U(2.5f);
			dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g}, IM_COL32(150, 190, 255, static_cast<int>(30.0f / k * hot)), rr + g, 0, U(2));
		}
		dl->PopClipRect();
		dl->AddRectFilled(a, b, IM_COL32(9, 9, 11, 228), rr);
		dl->AddRectFilledMultiColor({a.x + rr * 0.5f, a.y + 1}, {b.x - rr * 0.5f, a.y + th * 0.55f}, Ca(40, 56, 90, 0.30f + 0.25f * hot),
		                            Ca(40, 56, 90, 0.10f), Ca(40, 56, 90, 0.0f), Ca(40, 56, 90, 0.0f));
		dl->AddRect(a, b, Ca(204, 176, 122, 0.45f + 0.45f * hot), rr, 0, std::max(1.0f, U(1.25f)));
		dl->AddRect({a.x + Px(3), a.y + Px(3)}, {b.x - Px(3), b.y - Px(3)}, Ca(204, 176, 122, 0.12f + 0.10f * hot), Px(4), 0, 1.0f);
		const float  mr = std::round(th * 0.34f);
		const ImVec2 mc {std::round(a.x + Px(12) + mr), std::round(a.y + th * 0.5f)};
		const float  pl = 0.5f + 0.5f * std::sin(static_cast<float>(ImGui::GetTime()) * 2.2f);
		dl->AddCircleFilled(mc, mr, IM_COL32(12, 16, 26, 255), 32);
		for (int k = 3; k >= 1; k--) dl->AddCircleFilled(mc, mr * (0.45f + 0.12f * k + 0.05f * pl), Ca(150, 190, 255, 0.05f * k), 32);
		dl->AddCircle(mc, mr, Ca(222, 196, 140, 0.55f + 0.45f * hot), 32, std::max(1.0f, U(1.4f)));
		Icons::Draw(dl, Icons::Soul, mc, mr * (1.25f + 0.06f * pl), Ca(190, 215, 255, 0.95f));
		const float lx = mc.x + mr + Px(16);
		Spaced(dl, cinzel, Px(19), {lx, a.y + Px(8)}, Ca(222, 196, 140, 0.85f + 0.15f * hot), "ONLINE", U(2.5f));
		Spaced(dl, cinzel, Px(14), {lx, a.y + Px(34)}, IM_COL32(200, 214, 236, 230), I18n::T("Cuenta y amigos"), U(1));
	}
	ImGui::End();
	if (!g_online_open) return;
	const ImVec2 ps {std::min(width * 0.9f, Px(980)), std::min(height * 0.8f, Px(860))};
	ImGui::SetNextWindowPos({std::round((width - ps.x) * 0.5f), std::round((height - ps.y) * 0.5f)});
	ImGui::SetNextWindowSize(ps);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0.05f, 0.05f, 0.06f, 0.96f});
	ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0.80f, 0.69f, 0.48f, 0.7f});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, Px(10));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {Px(28), Px(24)});
	ImGui::Begin("##online-panel", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
	ImGui::BeginChild("##online-body", {0, -Px(64)});
	Multiplayer::Card(U(1));
	ImGui::EndChild();
	if (Anim::Button(I18n::T("Cerrar"), {Px(220), Px(48)}, U(1))) g_online_open = false;
	if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false)) g_online_open = false;
	ImGui::End();
	ImGui::PopStyleVar(2);
	ImGui::PopStyleColor(2);
}

// (cleanbtn) "SIN TRUCOS" beside ONLINE at the launcher's bottom left: the same tab look as DrawOnline (medallion with a
// soul, caps in Cinzel). It is NOT a panel but a toggle: a click flips g_saved.no_cheats and saves it to settings.ini at
// once (the very save the old in-panel NoCheatsBlock box did). The medallion stays lit while no_cheats is ON, so the state
// is visible without opening anything. Launcher only (as DrawOnline). Behaviour unchanged: ON still hides every F2 cheat and
// forces them off at run time; this only moves the control out of the Conexión online panel onto the bottom bar.
void DrawNoCheats(float width, float height) {
	ImFont*     cinzel = g_font_display != nullptr ? g_font_display : ImGui::GetFont();
	const float th = Px(60);
	float tx, tw;
	TileRect(2, tx, tw); // (tiles-fit)
	const float ty = std::round(height - U(34) - th);
	ImGui::SetNextWindowPos({tx, ty});
	ImGui::SetNextWindowSize({tw, th});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::Begin("##nocheats-tab", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
	                                           ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing);
	ImGui::PopStyleVar();
	{
		const bool on = g_saved.no_cheats;
		ImGui::SetCursorScreenPos({tx, ty});
		const bool clicked = ImGui::InvisibleButton("##nocheats-tab-button", {tw, th}, ImGuiButtonFlags_EnableNav);
		const bool hov     = ImGui::IsItemHovered();
		if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		const float hot = Anim::Hot(ImGui::GetItemID(), hov || ImGui::IsItemFocused() || on);
		Tip("Oculta y desactiva todos los trucos del men\xC3\xBA F2 (armas, objetos, almas, nivel...). \xC3\x9Atil para jugar limpio online.");
		if (clicked) {
			// Same save as the old NoCheatsBlock, only toggling instead of reading a checkbox.
			const bool now = !g_saved.no_cheats;
			Settings on_disk;
			if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_saved;
			for (Settings* s: {&on_disk, &g_saved, &g_edit})
				s->no_cheats = now;
			(void)SaveSettings(g_rt.ini_path, on_disk);
			std::printf("[port] sin_trucos=%d\n", now ? 1 : 0);
		}
		auto*        dl = ImGui::GetWindowDrawList();
		const ImVec2 a {tx, ty}, b {tx + tw, ty + th};
		const float  rr = Px(6);
		dl->PushClipRect({0, 0}, {width, height}, false);
		for (int k = 1; k <= 3; k++) {
			const float g = static_cast<float>(k) * U(2.5f);
			dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g}, IM_COL32(150, 190, 255, static_cast<int>(30.0f / k * hot)), rr + g, 0, U(2));
		}
		dl->PopClipRect();
		dl->AddRectFilled(a, b, IM_COL32(9, 9, 11, 228), rr);
		dl->AddRectFilledMultiColor({a.x + rr * 0.5f, a.y + 1}, {b.x - rr * 0.5f, a.y + th * 0.55f}, Ca(40, 56, 90, 0.30f + 0.25f * hot),
		                            Ca(40, 56, 90, 0.10f), Ca(40, 56, 90, 0.0f), Ca(40, 56, 90, 0.0f));
		dl->AddRect(a, b, Ca(204, 176, 122, 0.45f + 0.45f * hot), rr, 0, std::max(1.0f, U(1.25f)));
		dl->AddRect({a.x + Px(3), a.y + Px(3)}, {b.x - Px(3), b.y - Px(3)}, Ca(204, 176, 122, 0.12f + 0.10f * hot), Px(4), 0, 1.0f);
		const float  mr = std::round(th * 0.34f);
		const ImVec2 mc {std::round(a.x + Px(12) + mr), std::round(a.y + th * 0.5f)};
		const float  pl = 0.5f + 0.5f * std::sin(static_cast<float>(ImGui::GetTime()) * 2.2f);
		dl->AddCircleFilled(mc, mr, IM_COL32(12, 16, 26, 255), 32);
		for (int k = 3; k >= 1; k--) dl->AddCircleFilled(mc, mr * (0.45f + 0.12f * k + 0.05f * pl), Ca(150, 190, 255, 0.05f * k), 32);
		dl->AddCircle(mc, mr, Ca(222, 196, 140, 0.55f + 0.45f * hot), 32, std::max(1.0f, U(1.4f)));
		Icons::Draw(dl, Icons::Soul, mc, mr * (1.25f + 0.06f * pl), Ca(190, 215, 255, 0.95f));
		const float lx = mc.x + mr + Px(16);
		Spaced(dl, cinzel, Px(19), {lx, a.y + Px(8)}, Ca(222, 196, 140, 0.85f + 0.15f * hot), "SIN TRUCOS", U(2.5f));
		Spaced(dl, cinzel, Px(14), {lx, a.y + Px(34)}, IM_COL32(200, 214, 236, 230), on ? "Activado" : "Desactivado", U(1));
	}
	ImGui::End();
}

// (rando1) RANDOMIZER beside SIN TRUCOS: a click turns it on / off (offline, cheats off, "Trofeos Randomizer"); a right
// click draws a new seed. The seed is kept in settings.ini (randomizer_semilla): the same seed, the same draw.
void DrawRandomizer(float width, float height) {
	ImFont*     cinzel = g_font_display != nullptr ? g_font_display : ImGui::GetFont();
	const float th = Px(60);
	float tx, tw;
	TileRect(3, tx, tw); // (tiles-fit)
	const float ty = std::round(height - U(34) - th);
	ImGui::SetNextWindowPos({tx, ty});
	ImGui::SetNextWindowSize({tw, th});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::Begin("##rando-tab", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
	                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing);
	ImGui::PopStyleVar();
	{
		const bool on = g_saved.randomizer;
		ImGui::SetCursorScreenPos({tx, ty});
		const bool clicked = ImGui::InvisibleButton("##rando-tab-button", {tw, th}, ImGuiButtonFlags_EnableNav | ImGuiButtonFlags_MouseButtonLeft |
		                                                                               ImGuiButtonFlags_MouseButtonRight);
		const bool right   = clicked && ImGui::IsMouseReleased(ImGuiMouseButton_Right);
		const bool hov     = ImGui::IsItemHovered();
		if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		const float hot = Anim::Hot(ImGui::GetItemID(), hov || ImGui::IsItemFocused() || on);
		Tip("Randomizer: partida sin conexi\xC3\xB3n y sin trucos, con los Trofeos Randomizer al derrotar a cada archidemonio. "
		    "Clic: activar / desactivar (semilla aleatoria en cada partida). Clic derecho: fijar una semilla nueva. "
		    "Clic en \"MUNDOS\" (a la derecha, con el randomizer activo): enemigos de otros mundos s\xC3\xAD / no (experimental). "
		    "Clic en \"NIVEL\": f\xC3\xA1" "cil (menos copias), intermedio (todas), dif\xC3\xAD" "cil (todas + el Vanguard en el primer patio).");
		// (rando20) The right part of the tile (only while on) toggles the enemies from other worlds.
		// (rando-ui2) Layout of the ON state, measured: [medal | title + seed | NIVEL | MUNDOS]. Each sub-column takes the
		// width of its widest label/value (fixed, so nothing jumps when cycling); the hit-test uses the same x's.
		const char* kDiff[3] = {"F\xC3\x81" "CIL", "INTERM.", "DIF\xC3\x8D" "CIL"};
		float       cfs      = Px(12); // the sub-columns' font size, shrunk if the left column gets too narrow
		float       col_pad  = Px(6), c1w = 0, c2w = 0;
		const float lx0      = tx + Px(12) + 2.0f * std::round(th * 0.34f) + Px(10); // the text after the medal
		const float rx       = tx + tw - Px(5);
		for (int it = 0; it < 4; it++) {
			c1w = Spaced(nullptr, cinzel, cfs, {0, 0}, 0, "NIVEL", 0.0f);
			for (const char* d: kDiff) c1w = std::max(c1w, Spaced(nullptr, cinzel, cfs, {0, 0}, 0, d, 0.0f));
			c2w = std::max(Spaced(nullptr, cinzel, cfs, {0, 0}, 0, "MUNDOS", 0.0f), Spaced(nullptr, cinzel, cfs, {0, 0}, 0, "S\xC3\x8D", 0.0f));
			c1w = std::ceil(c1w + 2 * col_pad);
			c2w = std::ceil(c2w + 2 * col_pad);
			if ((rx - c1w - c2w) - lx0 >= Px(110) || cfs <= Px(9)) break;
			cfs     = std::max(Px(9), cfs - Px(1));
			col_pad = std::max(Px(4), col_pad - Px(1));
		}
		const float c2x        = std::round(rx - c2w), c1x = std::round(c2x - c1w); // the dividers' x
		const float mx         = ImGui::GetIO().MousePos.x;
		const bool  cross_zone = clicked && !right && g_saved.randomizer && mx >= c2x;
		// (rando27) Zona "NIVEL" (a la izquierda de MUNDOS): FACIL -> INTERMEDIO -> DIFICIL -> FACIL.
		const bool  diff_zone = clicked && !right && !cross_zone && g_saved.randomizer && mx >= c1x;
		if (clicked) {
			Settings on_disk;
			if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_saved;
			const bool now   = (cross_zone || diff_zone) ? true : (right ? true : !g_saved.randomizer);
			const int  diff  = diff_zone ? (g_saved.randomizer_difficulty % 3) + 1 : g_saved.randomizer_difficulty;
			int        seed  = right ? 1 + static_cast<int>(static_cast<unsigned>(std::time(nullptr) * 2654435761u) % 999999u)
			                         : (now ? g_saved.randomizer_seed : 0); // 0: a new random seed each launch
			const bool cross = cross_zone ? !g_saved.randomizer_cross : (now ? g_saved.randomizer_cross : false);
			for (Settings* s: {&on_disk, &g_saved, &g_edit}) {
				s->randomizer       = now;
				s->randomizer_seed  = seed;
				s->randomizer_cross = cross;
				s->randomizer_difficulty = diff;
			}
			(void)SaveSettings(g_rt.ini_path, on_disk);
			std::printf("[port] randomizer=%d semilla=%d otros_mundos=%d dificultad=%d\n", now ? 1 : 0, seed, cross ? 1 : 0, diff);
		}
		auto*        dl = ImGui::GetWindowDrawList();
		const ImVec2 a {tx, ty}, b {tx + tw, ty + th};
		const float  rr = Px(6);
		dl->PushClipRect({0, 0}, {width, height}, false);
		for (int k = 1; k <= 3; k++) {
			const float g = static_cast<float>(k) * U(2.5f);
			dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g}, IM_COL32(230, 180, 90, static_cast<int>(30.0f / k * hot)), rr + g, 0, U(2));
		}
		dl->PopClipRect();
		dl->AddRectFilled(a, b, IM_COL32(9, 9, 11, 228), rr);
		dl->AddRectFilledMultiColor({a.x + rr * 0.5f, a.y + 1}, {b.x - rr * 0.5f, a.y + th * 0.55f}, Ca(90, 60, 30, 0.30f + 0.25f * hot),
		                            Ca(90, 60, 30, 0.10f), Ca(90, 60, 30, 0.0f), Ca(90, 60, 30, 0.0f));
		dl->AddRect(a, b, Ca(204, 176, 122, 0.45f + 0.45f * hot), rr, 0, std::max(1.0f, U(1.25f)));
		dl->AddRect({a.x + Px(3), a.y + Px(3)}, {b.x - Px(3), b.y - Px(3)}, Ca(204, 176, 122, 0.12f + 0.10f * hot), Px(4), 0, 1.0f);
		// The medal: iron disc, gold rim, crossed blades and the soul in the middle (as the trophies' icon).
		const float  mr = std::round(th * 0.34f);
		const ImVec2 mc {std::round(a.x + Px(12) + mr), std::round(a.y + th * 0.5f)};
		const float  pl = 0.5f + 0.5f * std::sin(static_cast<float>(ImGui::GetTime()) * 2.2f);
		dl->AddCircleFilled(mc, mr, IM_COL32(20, 19, 23, 255), 32);
		dl->AddCircle(mc, mr * 0.92f, Ca(200, 150, 70, 0.75f + 0.25f * hot), 32, std::max(1.5f, U(2.4f)));
		for (int s = -1; s <= 1; s += 2) {
			const ImVec2 p0 {mc.x - s * mr * 0.55f, mc.y + mr * 0.55f}, p1 {mc.x + s * mr * 0.48f, mc.y - mr * 0.48f};
			dl->AddLine(p0, p1, IM_COL32(170, 176, 190, 235), std::max(1.5f, U(2.2f)));
			const ImVec2 gc {p0.x + (p1.x - p0.x) * 0.18f, p0.y + (p1.y - p0.y) * 0.18f};
			dl->AddLine({gc.x - mr * 0.16f, gc.y - s * mr * 0.16f}, {gc.x + mr * 0.16f, gc.y + s * mr * 0.16f}, IM_COL32(205, 160, 80, 240),
			            std::max(1.5f, U(2.0f)));
		}
		for (int k = 3; k >= 1; k--) dl->AddCircleFilled(mc, mr * (0.10f + 0.07f * k + 0.03f * pl), Ca(200, 228, 255, 0.10f * k), 24);
		dl->AddCircleFilled(mc, mr * 0.12f, IM_COL32(245, 250, 255, 255), 16);
		const float lx = mc.x + mr + Px(10);
		// (rando-ui2) Title and seed fit their column (smaller, then the seed shortened): never under NIVEL / MUNDOS.
		const float lw  = (on ? c1x - Px(5) : b.x - Px(10)) - lx;
		const auto  fit = [&](const char* t, float size, float sp) {
			while (size > 6.0f && Spaced(nullptr, cinzel, size, {0, 0}, 0, t, sp) > lw) size -= 1.0f;
			return size;
		};
		const float tsp = on ? U(0.5f) : U(2.5f); // tighter letters while the sub-columns share the tile
		const float ts  = fit("RANDOMIZER", Px(19), tsp);
		Spaced(dl, cinzel, ts, {lx, std::round(a.y + Px(8) + (Px(19) - ts) * 0.5f)}, Ca(222, 196, 140, 0.85f + 0.15f * hot), "RANDOMIZER", tsp);
		char sub[64];
		if (on && g_saved.randomizer_seed != 0) std::snprintf(sub, sizeof(sub), "Semilla %d", g_saved.randomizer_seed);
		else if (on) std::snprintf(sub, sizeof(sub), "Semilla aleatoria");
		else std::snprintf(sub, sizeof(sub), "Desactivado");
		float ss = fit(sub, Px(14), 0.0f);
		if (on && ss < Px(11)) { // too small: the short form
			if (g_saved.randomizer_seed != 0) std::snprintf(sub, sizeof(sub), "#%d", g_saved.randomizer_seed);
			else std::snprintf(sub, sizeof(sub), "Aleatoria");
			ss = fit(sub, Px(14), 0.0f);
		}
		Spaced(dl, cinzel, ss, {lx, a.y + Px(34)}, IM_COL32(236, 214, 190, 230), sub, 0.0f);
		if (on) { // (rando27) "NIVEL" y (rando20) "MUNDOS": label on top, value below, centred in their column
			const int  d    = std::clamp(g_saved.randomizer_difficulty, 1, 3);
			const auto cell = [&](float x0, float w, const char* label, const char* value, ImU32 vcol) {
				dl->AddLine({x0, a.y + Px(10)}, {x0, b.y - Px(10)}, Ca(204, 176, 122, 0.35f), 1.0f);
				const float w1 = Spaced(nullptr, cinzel, cfs, {0, 0}, 0, label, 0.0f);
				const float w2 = Spaced(nullptr, cinzel, cfs, {0, 0}, 0, value, 0.0f);
				Spaced(dl, cinzel, cfs, {std::round(x0 + (w - w1) * 0.5f), std::round(a.y + Px(11))}, Ca(222, 196, 140, 0.75f), label, 0.0f);
				Spaced(dl, cinzel, cfs, {std::round(x0 + (w - w2) * 0.5f), std::round(b.y - Px(11) - cfs)}, vcol, value, 0.0f);
			};
			cell(c1x, c1w, "NIVEL", kDiff[d - 1],
			     d == 1 ? IM_COL32(170, 230, 160, 235) : (d == 3 ? IM_COL32(240, 120, 100, 240) : IM_COL32(235, 205, 140, 235)));
			cell(c2x, c2w, "MUNDOS", g_saved.randomizer_cross ? "S\xC3\x8D" : "NO",
			     g_saved.randomizer_cross ? IM_COL32(170, 230, 160, 235) : IM_COL32(200, 200, 200, 200));
		}
	}
	ImGui::End();
}
} // namespace LauncherTrophy

// (ds-pc-port11) Text with a dark outline (shadow) for the shaders' panel: legible over the panel and over the gold bar.
void ShadowText(ImDrawList* dl, ImFont* font, float size, ImVec2 at, ImU32 col, const char* text, float wrap, bool outline = false) {
	const float o = std::max(1.0f, size * 0.06f);
	const ImU32 dark = IM_COL32(0, 0, 0, outline ? 235 : 190);
	if (outline) {
		for (int dx = -1; dx <= 1; dx++)
			for (int dy = -1; dy <= 1; dy++)
				if (dx != 0 || dy != 0) dl->AddText(font, size, {at.x + dx * o, at.y + dy * o}, dark, text, nullptr, wrap);
	} else {
		dl->AddText(font, size, {at.x + o, at.y + o}, dark, text, nullptr, wrap);
	}
	dl->AddText(font, size, at, col, text, nullptr, wrap);
}

// (ds-pc-port11) The animated hourglass beside the shaders' text while they precompile: the sand runs from the top bulb
// into the bottom one, then the glass turns over (half a turn) and it starts again. Drawn every frame (the launcher
// redraws at the display's refresh), from ImGui's clock.
void DrawHourglass(ImDrawList* dl, ImVec2 c, float h, double t) {
	constexpr double kRun = 2.6, kFlip = 0.6, kPi = 3.14159265358979323846;
	const double     ph   = std::fmod(t, kRun + kFlip);
	float            fill = 1.0f, angle = 0.0f;
	if (ph < kRun) {
		fill = static_cast<float>(ph / kRun);
	} else {
		const double k = (ph - kRun) / kFlip;
		angle          = static_cast<float>(kPi * k * k * (3.0 - 2.0 * k)); // smoothstep
	}
	const float cs = std::cos(angle), sn = std::sin(angle);
	const auto  P  = [&](float x, float y) { return ImVec2 {c.x + x * cs - y * sn, c.y + x * sn + y * cs}; };
	const float hh = h * 0.5f, hw = h * 0.33f, gh = hh * 0.84f, nw = hw * 0.10f;
	const ImU32 frame = IM_COL32(222, 196, 140, 255), glass = IM_COL32(236, 226, 205, 200), tint = IM_COL32(255, 255, 255, 26);
	const ImU32 sand = IM_COL32(242, 196, 102, 255);
	// The glass.
	const ImVec2 top[3] = {P(-hw, -gh), P(hw, -gh), P(0, 0)};
	const ImVec2 bot[3] = {P(0, 0), P(hw, gh), P(-hw, gh)};
	dl->AddConvexPolyFilled(top, 3, tint);
	dl->AddConvexPolyFilled(bot, 3, tint);
	// The sand: what is left in the top bulb (a triangle down to the neck) and the heap in the bottom one.
	const float s = std::sqrt(std::max(0.0f, 1.0f - fill)), k = 0.78f;
	if (s > 0.02f) {
		const ImVec2 st[3] = {P(-hw * k * s, -gh * s), P(hw * k * s, -gh * s), P(0, -gh * 0.03f)};
		dl->AddConvexPolyFilled(st, 3, sand);
	}
	if (s < 0.98f) {
		const ImVec2 sb[4] = {P(-hw * k * s, gh * std::max(s, 0.06f)), P(hw * k * s, gh * std::max(s, 0.06f)), P(hw * k, gh * 0.97f), P(-hw * k, gh * 0.97f)};
		dl->AddConvexPolyFilled(sb, 4, sand);
	}
	// The falling stream, with grains running down it.
	if (angle == 0.0f && fill > 0.0f && fill < 0.995f) {
		const float y1 = gh * std::max(s, 0.06f);
		dl->AddLine(P(0, 0), P(0, y1), sand, std::max(1.0f, h * 0.035f));
		for (int i = 0; i < 3; i++) {
			const float f = static_cast<float>(std::fmod(t * 2.2 + i / 3.0, 1.0));
			dl->AddCircleFilled(P(0, y1 * f), std::max(1.0f, h * 0.035f), sand);
		}
	}
	// The outline of the glass, the posts and the two caps.
	const ImVec2 outline[6] = {P(-hw, -gh), P(hw, -gh), P(nw, 0), P(hw, gh), P(-hw, gh), P(-nw, 0)};
	dl->AddPolyline(outline, 6, glass, ImDrawFlags_Closed, std::max(1.0f, h * 0.045f));
	dl->AddLine(P(-hw * 1.18f, -hh), P(-hw * 1.18f, hh), frame, std::max(1.0f, h * 0.04f));
	dl->AddLine(P(hw * 1.18f, -hh), P(hw * 1.18f, hh), frame, std::max(1.0f, h * 0.04f));
	dl->AddLine(P(-hw * 1.4f, -hh), P(hw * 1.4f, -hh), frame, std::max(2.0f, h * 0.10f));
	dl->AddLine(P(-hw * 1.4f, hh), P(hw * 1.4f, hh), frame, std::max(2.0f, h * 0.10f));
}

// (step5) The launcher's fog and dust layer was removed: the static illustration only (as before the fog work).

void DrawLauncher(float width, float height) {
	DrainInputs();
	EnsureBackground();
	// (ds-pc-port9) Bigger text: 1.125 x the launcher's pixels at 1080 lines; the column (440 of those) on the right, its
	// width at most 34 % of the window: the art (the knight and the logo) is never covered.
	g_unit = std::clamp(height / 1080.0f, 0.55f, 3.0f) * 1.2f; // (step5: larger text, was 1.125)
	if (U(452) > width * 0.28f) g_unit = std::max(0.45f, width * 0.28f / 452.0f);
	const float col_w = U(440);
	const float px    = width - col_w - U(12);
	auto*       bg    = ImGui::GetBackgroundDrawList();
	bg->AddRectFilled({0, 0}, {width, height}, IM_COL32(14, 16, 22, 255));
	if (g_background != nullptr && g_bg_w > 0 && g_bg_h > 0) {
		// (ds-pc-port9) The whole width of the art, always (the logo at the left and the knights stay in view, the column
		// on the right only over the castle): a wider screen cuts only the ground at the bottom; a taller one gets dark
		// bands above and below.
		const float scale = width / g_bg_w;
		const float w = g_bg_w * scale, h = g_bg_h * scale;
		const float y0 = h >= height ? 0.0f : (height - h) * 0.5f;
		bg->AddImage(g_background->GetTexRef(), {0, y0}, {w, y0 + h});
		if (h < height) {
			bg->AddRectFilledMultiColor({0, y0}, {width, y0 + U(60)}, IM_COL32(14, 16, 22, 255), IM_COL32(14, 16, 22, 255), IM_COL32(14, 16, 22, 0), IM_COL32(14, 16, 22, 0));
			bg->AddRectFilledMultiColor({0, y0 + h - U(60)}, {width, y0 + h}, IM_COL32(14, 16, 22, 0), IM_COL32(14, 16, 22, 0), IM_COL32(14, 16, 22, 255), IM_COL32(14, 16, 22, 255));
		}
	}
	// (11-10) The fog and the embers, on the GPU (ds-fog.inc), over the art and under the menus (lanzador_ambiente).
	if (g_saved.launcher_ambience) {
		float ix = 0, iy = 0, iw = width, ih = height;
		if (g_background != nullptr && g_bg_w > 0 && g_bg_h > 0) {
			iw = width;
			ih = g_bg_h * (width / g_bg_w);
			iy = ih >= height ? 0.0f : (height - ih) * 0.5f;
		}
#if defined(_WIN32)
		FogGpu::Draw(bg, width, height, ix, iy, iw, ih, 1.0f);
#endif
	}
	if (ImFont* f = g_font_semi != nullptr ? g_font_semi : ImGui::GetFont(); f != nullptr) {
		// (ds-pc-port39) The fork's credit at the window's top-right corner, tight to it (margin 11 px x the unit, right-aligned
		// by its measured width), small and subtle with a soft shadow; on the foreground list: above the Opciones column.
		auto*        fg     = ImGui::GetForegroundDrawList();
		const float  size   = std::max(12.0f, std::round(U(14)));
		const float  margin = std::round(U(11));
		const char*  text   = "Fork: todojuegos67";
		const ImVec2 ts     = f->CalcTextSizeA(size, FLT_MAX, 0.0f, text);
		const ImVec2 at     = {std::floor(width - margin - ts.x), margin};
		fg->AddText(f, size, {at.x + 1, at.y + 1}, IM_COL32(0, 0, 0, 170), text);
		fg->AddText(f, size, {at.x + 1.5f, at.y + 1.5f}, IM_COL32(0, 0, 0, 90), text);
		fg->AddText(f, size, at, IM_COL32(236, 226, 204, 200), text);
		// (ds-pc-port52) The F2 hint just below it, same style, a little smaller: F2 switches the launcher between full
		// screen borderless and a normal window (RunStartMenu).
		const float  hsize = std::max(11.0f, std::round(U(12)));
		const char*  hint  = I18n::T("F2 modo ventana");
		const ImVec2 hs    = f->CalcTextSizeA(hsize, FLT_MAX, 0.0f, hint);
		const ImVec2 hat   = {std::floor(width - margin - hs.x), std::floor(at.y + ts.y + U(2))};
		fg->AddText(f, hsize, {hat.x + 1, hat.y + 1}, IM_COL32(0, 0, 0, 170), hint);
		fg->AddText(f, hsize, {hat.x + 1.5f, hat.y + 1.5f}, IM_COL32(0, 0, 0, 90), hint);
		fg->AddText(f, hsize, hat, IM_COL32(236, 226, 204, 200), hint);
	}

	auto& io = ImGui::GetIO();
#if defined(NDEBUG)
	io.ConfigErrorRecoveryEnableTooltip = false;
	io.ConfigErrorRecoveryEnableAssert  = false;
#endif
	ImGuiContext& ctx           = *ImGui::GetCurrentContext();
	const int     colors_before = ctx.ColorStack.Size, vars_before = ctx.StyleVarStack.Size;
	ImGui::PushFont(g_font_semi, U(16));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {U(14), U(12)});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {U(10), U(9)});
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {U(9), U(5)});
	ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, U(8));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, U(12));
	ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, U(10));
	ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, U(12));
	ImGui::PushStyleVar(ImGuiStyleVar_DisabledAlpha, 0.55f);
	ImGui::PushStyleColor(ImGuiCol_Text, kInkL);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0.025f, 0.03f, 0.04f, 0.90f});
	ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4 {0.07f, 0.075f, 0.10f, 0.99f});
	ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0.87f, 0.77f, 0.55f, 0.45f});
	ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4 {0.094f, 0.10f, 0.13f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4 {0.16f, 0.17f, 0.22f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4 {0.20f, 0.21f, 0.27f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_CheckMark, kGoldL);
	ImGui::PushStyleColor(ImGuiCol_SliderGrab, kGoldL);
	ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4 {1.0f, 0.88f, 0.62f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.094f, 0.11f, 0.15f, 0.90f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.17f, 0.20f, 0.25f, 0.95f});
	ImGui::PushStyleColor(ImGuiCol_Header, ImVec4 {0.24f, 0.25f, 0.31f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4 {0.55f, 0.43f, 0.22f, 0.80f});
	ImGui::PushStyleColor(ImGuiCol_NavCursor, kGoldL);
	ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kGoldL);
	ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4 {0.0f, 0.0f, 0.0f, 0.25f});
	ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, ImVec4 {0.55f, 0.48f, 0.36f, 0.85f});

	Settings& s = g_edit;
	// Esc: the open list, then the options, then the launcher (as Salir).
	if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) && !g_popup.load()) {
		if (LauncherTrophy::IsOpen()) LauncherTrophy::Close(true); // (trofeos-exe) the trophy panel first
		else if (g_upd.state != UpdateUi::Idle && !g_upd.busy.load()) SetUpd(UpdateUi::Idle, "");
		else if (g_show_settings) g_show_settings = false;
		else g_start_result = -1;
	}

	// ---- The bottom of the column: the shaders' state, where JUGAR continues, JUGAR, Opciones and Salir. ----
	const auto        pv   = Precompile::State();
	const std::string when; // (ds-pc-port37: no "continue where F2 left")
	// (ds-pc-port11) The shaders' panel: its text in big light letters wrapped inside the column (an animated hourglass
	// beside it while they precompile), then the progress bar with the percent and the parts, then Detener; all measured
	// first so that nothing overlaps JUGAR below.
	const bool  shader_row = pv.running || !pv.finished || pv.failed;
	const float sh_pad = U(12), sh_glass = pv.running ? U(40) : 0.0f, sh_gap = pv.running ? U(12) : 0.0f;
	const float sh_inner = U(420) - 2.0f * sh_pad;
	const float sh_wrap  = sh_inner - sh_glass - sh_gap;
	const float sh_head_size = U(20), sh_detail_size = U(17);
	const bool  sh_button = pv.running || !pv.finished;
	std::string sh_head, sh_detail;
	float       sh_head_h = 0.0f, sh_detail_h = 0.0f, sh_text_h = 0.0f, sh_h = 0.0f;
	if (shader_row) {
		const std::string raw = pv.text.empty() ? std::string("Shaders: la cache de esta grafica ya esta preparada.") : pv.text;
		const size_t      nl  = raw.find('\n');
		sh_head               = I18n::T(raw.substr(0, nl).c_str());
		if (nl != std::string::npos && pv.running) {
			char part[128] = "";
			if (pv.eta_s < 0) std::snprintf(part, sizeof(part), "%s", I18n::T("calculando el tiempo restante"));
			else if (pv.eta_s >= 3600)
				std::snprintf(part, sizeof(part), I18n::T("quedan unos %lld h %lld min"), pv.eta_s / 3600, pv.eta_s % 3600 / 60);
			else std::snprintf(part, sizeof(part), I18n::T("quedan unos %lld min"), std::max<long long>(1, pv.eta_s / 60));
			sh_detail = part;
			if (pv.resumed_pct > 0) {
				std::snprintf(part, sizeof(part), I18n::T("(retomada desde el %d %%)"), pv.resumed_pct);
				sh_detail += std::string("  ") + part;
			}
			if (pv.waiting_mem) sh_detail += std::string("  ") + I18n::T("[esperando memoria libre]");
		} else if (nl != std::string::npos) {
			sh_detail = raw.substr(nl + 1);
		}
		ImGui::PushFont(g_font_semi, sh_head_size);
		sh_head_h = ImGui::CalcTextSize(sh_head.c_str(), nullptr, false, sh_wrap).y;
		ImGui::PopFont();
		if (!sh_detail.empty()) {
			ImGui::PushFont(g_font_semi, sh_detail_size);
			sh_detail_h = U(6) + ImGui::CalcTextSize(sh_detail.c_str(), nullptr, false, sh_wrap).y;
			ImGui::PopFont();
		}
		sh_text_h = std::max(sh_head_h + sh_detail_h, sh_glass > 0.0f ? U(50) : 0.0f);
		sh_h      = sh_pad + sh_text_h + (pv.running ? U(12) + U(32) : 0.0f) + (sh_button ? U(12) + U(36) : 0.0f) + sh_pad;
	}
	float bottom_h = U(58 + 8 + 40 + 16);
	bottom_h += U(44); // (11-10) the "skip the intro" row
	bottom_h += U(44); // (update1) the red "Actualizar versión" row
	if (g_rt.pick_folder) bottom_h += U(44); // (port26) the game folder row
	if (shader_row) bottom_h += sh_h + U(14);
	if (!when.empty()) bottom_h += U(26);
	const float bottom_y = height - bottom_h - U(10);
	// A dark backing under the column's text (only the column: the art stays as it is).
	bg->AddRectFilledMultiColor({px + U(30), bottom_y - U(30)}, {width, height}, IM_COL32(0, 0, 0, 0), IM_COL32(0, 0, 0, 0), IM_COL32(4, 5, 8, 215),
	                            IM_COL32(4, 5, 8, 215));
	bg->AddRectFilledMultiColor({px - U(40), bottom_y - U(30)}, {px + U(30), height}, IM_COL32(0, 0, 0, 0), IM_COL32(0, 0, 0, 0), IM_COL32(4, 5, 8, 215),
	                            IM_COL32(4, 5, 8, 0));
	ImGui::SetNextWindowPos({px, bottom_y});
	ImGui::SetNextWindowSize({col_w, bottom_h + U(10)});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::Begin("##launcher-bottom", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
	                                               ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse);
	ImGui::PopStyleVar();
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {U(8), 0});
	g_lbox.at = {px + U(10), ImGui::GetCursorScreenPos().y};
	g_lbox.w  = U(420);
	{
		// (11-10) Skip the PlayStation logo video and the opening cinematic: above the game folder, saved at once.
		const float y     = RowY();
		std::string label = std::string(I18n::T("Saltar intro y videos de inicio: ")) + I18n::T(s.skip_intro ? "Si" : "No");
		if (LButton(label.c_str(), {g_lbox.at.x, y}, {U(420), U(36)}, s.skip_intro, g_font_semi, U(15),
		            "Salta el video de PlayStation y la cinematica del principio, hasta la pantalla de titulo.")) {
			s.skip_intro       = !s.skip_intro;
			g_saved.skip_intro = s.skip_intro;
			Settings on_disk;
			if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_edit;
			on_disk.skip_intro = s.skip_intro;
			g_status = SaveSettings(g_rt.ini_path, on_disk) ? "Guardado en settings.ini" : "No se pudo guardar settings.ini";
		}
		RowEnd(y, 44);
	}
	{
		// (update1) "Actualizar versión" in red, just above the game folder: checks GitHub only when pressed (ManualCheck).
		const float y = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x, y});
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.42f, 0.05f, 0.04f, 0.88f});
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.60f, 0.08f, 0.06f, 0.94f});
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0.72f, 0.10f, 0.07f, 1.0f});
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4 {1.0f, 0.86f, 0.82f, 1.0f});
		ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0.90f, 0.22f, 0.18f, 0.95f});
		ImGui::PushFont(g_font_semi, U(15));
		ImGui::BeginDisabled(g_upd.busy.load());
		const bool upd = ImGui::Button(I18n::T("Actualizar versión"), {U(420), U(36)});
		Anim::Highlight(g_unit);
		ImGui::EndDisabled();
		ImGui::PopFont();
		ImGui::PopStyleColor(5);
		Tip("Busca en GitHub si hay una versión nueva del port y te pregunta si quieres instalarla. Al arrancar ya no se comprueba nada.");
		if (upd) ManualCheck();
		RowEnd(y, 44);
	}
	if (g_rt.pick_folder) {
		// (port26) The game folder, above the shaders' panel: chosen, checked and saved to settings.ini at once (only that key).
		const float y     = RowY();
		std::string label = s.game.empty() ? std::string(I18n::T("Elegir carpeta del juego...")) : std::string(I18n::T("Carpeta del juego: ")) + s.game;
		if (label.size() > 60) label = label.substr(0, 28) + "..." + label.substr(label.size() - 29);
		if (LButton(label.c_str(), {g_lbox.at.x, y}, {U(420), U(36)}, false, g_font_semi, U(15),
		            "Elige la carpeta de Demon's Souls (la que tiene eboot.bin y sce_sys). Se guarda al momento en settings.ini.")) {
			const std::string picked = g_rt.pick_folder();
			if (!picked.empty()) {
				if (!g_rt.is_game || g_rt.is_game(picked)) {
					s.game       = picked;
					g_saved.game = picked;
					Settings on_disk;
					if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_edit;
					on_disk.game = picked;
					g_status = SaveSettings(g_rt.ini_path, on_disk) ? "Carpeta del juego guardada en settings.ini" : "No se pudo guardar settings.ini";
				} else {
					g_status = "Esa carpeta no es Demon's Souls (falta eboot.bin o sce_sys).";
				}
			}
		}
		RowEnd(y, 44);
	}
	if (shader_row) {
		auto*       dl = ImGui::GetWindowDrawList();
		const float sy = RowY();
		const float x0 = g_lbox.at.x, x1 = g_lbox.at.x + U(420);
		// The semi-transparent dark panel (the art shows faintly through it).
		dl->AddRectFilled({x0, sy}, {x1, sy + sh_h}, IM_COL32(5, 7, 11, 205), U(8));
		dl->AddRect({x0, sy}, {x1, sy + sh_h}, IM_COL32(222, 196, 140, 120), U(8), 0, 1.25f);
		const float tx = x0 + sh_pad + sh_glass + sh_gap;
		float       cy = sy + sh_pad;
		if (pv.running) DrawHourglass(dl, {x0 + sh_pad + sh_glass * 0.5f, cy + std::min(sh_text_h, U(64)) * 0.5f}, U(46), ImGui::GetTime());
		const ImU32 head_col = ImGui::GetColorU32(pv.failed ? kWarn : kInkL);
		ShadowText(dl, g_font_semi, sh_head_size, {tx, cy}, head_col, sh_head.c_str(), sh_wrap);
		if (!sh_detail.empty())
			ShadowText(dl, g_font_semi, sh_detail_size, {tx, cy + sh_head_h + U(6)}, ImGui::GetColorU32(kGoldL), sh_detail.c_str(), sh_wrap);
		cy += sh_text_h;
		if (pv.running) {
			cy += U(12);
			const float frac = pv.total > 0 ? static_cast<float>(pv.done) / static_cast<float>(pv.total) : -1.0f * static_cast<float>(ImGui::GetTime());
			ImGui::SetCursorScreenPos({x0 + sh_pad, cy});
			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, U(5));
			ImGui::ProgressBar(frac, {sh_inner, U(32)}, "");
			ImGui::PopStyleVar();
			char label[96] = "";
			if (pv.total > 0)
				std::snprintf(label, sizeof(label), I18n::T("%d %%   -   %d / %d partes"), pv.done * 100 / pv.total, pv.done, pv.total);
			else std::snprintf(label, sizeof(label), "%s", I18n::T("Preparando..."));
			ImGui::PushFont(g_font_semi, U(18));
			const ImVec2 ls = ImGui::CalcTextSize(label);
			ImGui::PopFont();
			ShadowText(dl, g_font_semi, U(18), {x0 + sh_pad + (sh_inner - ls.x) * 0.5f, cy + (U(32) - ls.y) * 0.5f}, ImGui::GetColorU32(kInkL), label, 0.0f,
			           true);
			cy += U(32);
		}
		if (sh_button) {
			cy += U(12);
			if (pv.running) {
				if (LButton("Detener", {x0 + sh_pad, cy}, {U(180), U(36)}, false, g_font_semi, U(18),
				            "Detiene la preparacion de shaders (lo hecho se guarda y sigue la proxima vez)."))
					std::thread([] { Precompile::Stop(); }).detach();
			} else if (LButton("Preparar shaders", {x0 + sh_pad, cy}, {U(240), U(36)}, false, g_font_semi, U(18),
			                   "Compila ya todos los pipelines del juego para esta grafica (evita tirones la primera vez)."))
				StartShaders();
			{
				const bool all_now = g_edit.precompile_threads == "todos";
				const float bx     = x0 + sh_pad + U(pv.running ? 196 : 256);
				const float bw     = std::max(U(120), std::min(U(220), x1 - sh_pad - bx));
				if (LButton(all_now ? "Hilos: m\xC3\xA1ximo" : "Hilos: relajado", {bx, cy}, {bw, U(36)}, false, g_font_semi, U(18),
				            "Cambia c\xC3\xB3mo precompila: m\xC3\xA1ximo = todos los hilos de la CPU (m\xC3\xA1s r\xC3\xA1pido, el PC va cargado); relajado = pocos hilos. Si est\xC3\xA1 precompilando, sigue en el modo nuevo sin perder lo hecho."))
					ToggleShaderThreads(pv.running);
			}
		}
		ImGui::SetCursorScreenPos({g_lbox.at.x, sy});
		ImGui::Dummy({g_lbox.w, sh_h + U(14)});
	}
	if (!when.empty()) {
		const float wy = RowY();
		char        line[256];
		std::snprintf(line, sizeof(line), I18n::T("Partida guardada con F2 el %s: JUGAR continúa allí."), when.c_str());
		ImGui::SetCursorScreenPos({g_lbox.at.x, wy});
		ImGui::PushFont(g_font_semi, U(14));
		ImGui::TextColored(kGoldL, "%s", line);
		ImGui::PopFont();
		Tip("En el juego, F2 (o R3 + OPTIONS) guarda y cierra. Con esto marcado, JUGAR pasa solo la pantalla de título y carga la partida en el sitio donde pulsaste F2.");
		RowEnd(wy, 26);
	}
	{
		const float y = RowY();
		g_focus_once  = false;
		if (LButton(pv.running && !pv.playable ? "JUGAR (sin esperar)" : "JUGAR", {g_lbox.at.x, y}, {U(420), U(58)}, true, g_font_display, std::round(U(32)),
		            pv.running ? "Detiene la preparacion de shaders (lo hecho se guarda y sigue la proxima vez). Sin ella, el juego compila shaders al jugar (tirones)."
		                       : "Guarda los ajustes en settings.ini y empieza el juego.")) {
			// (No game folder: the picker at once; a valid one chosen, saved and played.)
			bool ok = !g_rt.is_game || g_rt.is_game(s.game);
			if (!ok && g_rt.pick_folder) {
				const std::string picked = g_rt.pick_folder();
				if (!picked.empty() && g_rt.is_game(picked)) {
					s.game = picked;
					ok     = true;
				} else if (!picked.empty()) {
					g_status        = "Esa carpeta no tiene eboot.bin ni sce_sys: elige la carpeta del juego volcado.";
					g_show_settings = true;
				}
			}
			if (ok) {
				Save();
				g_start_result = 1;
			}
		}
		RowEnd(y, 66);
	}
	{
		const float y = RowY();
		if (LButton(g_show_settings ? "Opciones  ^" : "Opciones  v", {g_lbox.at.x, y}, {U(310), U(40)}, false, g_font_semi, U(17),
		            "Todas las opciones: Modos Souls, imagen, frame generation, calidad gráfica, ray tracing, pantalla, controles, ajustes, actualizar y trofeos."))
			g_show_settings = !g_show_settings;
		if (LButton("Salir", {g_lbox.at.x + U(320), y}, {U(100), U(40)}, false, g_font_semi, U(17), "Cierra el lanzador sin jugar (Esc)."))
			g_start_result = -1;
		RowEnd(y, 48);
	}
	ImGui::PopStyleVar();
	ImGui::End();

	// ---- Opciones: every group in one column above JUGAR, scrolled with the wheel. ----
	if (g_show_settings) {
		const float top = U(12);
		ImGui::SetNextWindowPos({px, top});
		// (ds-pc-port12) Down to just above the bottom block (it scrolls inside).
		ImGui::SetNextWindowSize({col_w, std::max(U(120), bottom_y - U(6) - top)});
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, U(8)});
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0.02f, 0.025f, 0.035f, 0.88f});
		ImGui::Begin("##launcher-options", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
		                                                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {U(8), 0});
		// (ds-pc-port12) The options' labels and values 2 pt larger, in the regular face and a lighter ink.
		ImGui::PushFont(g_font_body, U(18));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4 {1.0f, 0.99f, 0.97f, 1.0f});
		const ImVec2 origin = ImGui::GetCursorScreenPos();
		ImGui::SetCursorScreenPos({origin.x + U(8), origin.y});
		ImGui::BeginGroup();
		LauncherUpdateBox();
		{
			// (11-13) The launcher sets the page file itself: once per start, when Windows' memory limit (RAM + page
			// file) is under 48 GB and the 60 GB page file is not already set, it asks for administrator permission and
			// sets it (a 16 GB PC rebooted at that limit). Then the user restarts the PC.
			static bool asked = false;
			if (!asked) {
				asked = true;
				const uint64_t limit = CommitLimitGb();
				if (limit != 0 && limit < 48 && !PagefileConfigured()) SetBigPagefile();
			}
		}
		LauncherSoulsModes(s);
		LauncherImage(s);
		LauncherFrameGen(s);
		LauncherQuality(s);
		{
			// (ds-pc-port37) "Ajustes del juego": the emulator launcher's window, as a drop-down here.
			g_lbox.at     = {origin.x + U(8), RowY()};
			g_lbox.w      = U(420);
			const float y = RowY();
			if (LButton(g_show_game_settings ? "Ajustes del juego  ^" : "Ajustes del juego  v", {g_lbox.at.x, y}, {U(420), U(38)}, false,
			            g_font_semi, U(16), "Los ajustes del propio juego (sombras, niebla, reflejos, distancia...), como en el lanzador del emulador."))
				g_show_game_settings = !g_show_game_settings;
			RowEnd(y, 46);
		}
		if (g_show_game_settings) LauncherGameSettings(s);
		LauncherRayTracing(s);
		LauncherDisplay(s);
		LauncherConfigurator(s);
		LauncherPerformance(s);
		{
			g_lbox.at     = {origin.x + U(8), RowY()};
			g_lbox.w      = U(420);
			const float y = RowY();
			if (LButton(g_show_controls ? "Controles  ^" : "Controles  v", {g_lbox.at.x, y}, {U(420), U(38)}, false, g_font_semi, U(16),
			            "Teclado y ratón, sensibilidad, cámara y vibración."))
				g_show_controls = !g_show_controls;
			RowEnd(y, 46);
		}
		if (g_show_controls) LauncherControls(s);
		LauncherAjustes(s);
		{
			g_lbox.at = {origin.x + U(8), RowY()};
			g_lbox.w  = U(420);
			// (trofeos-exe) Trofeos left the options: the tab at the bottom left (LauncherTrophy::Draw, below).
			const float by = RowY();
			ImGui::BeginDisabled(g_upd.busy.load());
			if (LButton("Actualizar version", {g_lbox.at.x, by}, {U(205), U(40)}, false, g_font_semi, U(15),
			            "Busca en GitHub la ultima version publicada del port (DemonsSoulsPC) y la instala en esta misma carpeta. Nunca baja de version. Los archivos actuales del motor se guardan en _backup_update; la cache de shaders, las partidas, los seeds y los ajustes no se tocan.")) {
				OpenCheck();
				ImGui::SetScrollY(0.0f);
			}
			if (LButton("Volver a la version anterior", {g_lbox.at.x + U(215), by}, {U(205), U(40)}, false, g_font_semi, U(13),
			            "Vuelve a una version guardada en _backup_update. La actual se guarda antes (se puede deshacer). La cache de shaders, las partidas, los seeds y los ajustes no se tocan.")) {
				OpenRestore();
				ImGui::SetScrollY(0.0f);
			}
			ImGui::EndDisabled();
			RowEnd(by, 48);
			const float dy   = RowY();
			const char* text = I18n::T("\xE2\x99\xA5  Dona tu alma  -  PayPal: todojuegos67@gmail.com");
			const ImVec2 ts  = ImGui::CalcTextSize(text);
			ImGui::SetCursorScreenPos({g_lbox.at.x + (U(420) - ts.x) * 0.5f, dy});
			ImGui::TextColored(kGoldL, "%s", text);
			if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
			if (ImGui::IsItemClicked()) OpenUrl("https://www.paypal.com/donate?business=todojuegos67%40gmail.com&currency_code=EUR");
			Tip("Si te gusta el proyecto, puedes donar por PayPal. Gracias, portador de la maldición.");
			RowEnd(dy, 30);
		}
		ImGui::EndGroup();
		ImGui::PopStyleColor();
		ImGui::PopFont();
		ImGui::PopStyleVar();
		ImGui::End();
	}

	// (trofeos-exe) TROFEOS at the bottom left, above the version line; its panel unfolds upwards.
	LauncherTrophy::Draw(width, height, px);
	LauncherTrophy::DrawOnline(width, height); // (online12) ONLINE beside TROFEOS
	LauncherTrophy::DrawNoCheats(width, height); // (cleanbtn) SIN TRUCOS beside ONLINE: toggles g_saved.no_cheats
	LauncherTrophy::DrawRandomizer(width, height); // (rando20) back in the launcher, off by default

	// (ds-pc-port17) The installed version, always visible (bottom left), and the startup update notice.
	{
		char ver[256];
		std::snprintf(ver, sizeof(ver), I18n::T("Version instalada: %s"), Update::CurrentVersion());
		const std::string line = std::string(ver) + "  \xC2\xB7  " + Update::BuildStamp();
		bg->AddText({U(12), height - U(26)}, IM_COL32(220, 214, 200, 230), line.c_str());
	}
	if (UpdateNotice(width, height)) g_show_settings = true;

	g_popup = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
	if (ctx.ColorStack.Size > colors_before) ImGui::PopStyleColor(ctx.ColorStack.Size - colors_before);
	if (ctx.StyleVarStack.Size > vars_before) ImGui::PopStyleVar(ctx.StyleVarStack.Size - vars_before);
	ImGui::PopFont();
	Bump();
}

bool WantsOverlay() {
	return MetricsPanel::Shown() || g_open.load(std::memory_order_relaxed) || Resume::SaveAndQuitActive() || Anim::Closing() || // (ds-pc-port38: the fade-out)
	       ChatUi::ToastActive(); // (chat1) a notice on screen
}

// (port34) F2 > Ver trofeos: every trophy of the game, as the PS5's trophy page. Drawn with draw lists in one window of
// its own; the icons come from one atlas texture (Libs::Trophy::BuildArt, decoded on a worker thread) registered with
// ImGui as a managed texture (ImTextureData): the in-game overlay's Vulkan backend (ImGui 1.92, RendererHasTextures)
// uploads it once, and it costs a single descriptor (the overlay's pool holds 8).

namespace TrophyPage {

struct ArtJob {
	std::mutex                          mutex;
	std::shared_ptr<Libs::Trophy::Art>  ready;
	std::atomic<bool>                   running {false};
	std::atomic<bool>                   failed {false};
};
ArtJob& Job() {
	static ArtJob job; // (never destroyed before a detached worker ends)
	return job;
}

ImTextureData*                       g_tex     = nullptr;
ImGuiContext*                        g_tex_ctx = nullptr;
std::shared_ptr<Libs::Trophy::Art>   g_art; // the slots (pixels moved into g_tex)
bool                                 g_art_requested = false;

struct Row {
	Libs::Trophy::Info info;
	bool               earned = false;
	int64_t            time   = 0;
};
std::vector<Row>   g_rows;
int                g_status       = -1; // TrySnapshot's
double             g_refreshed_at = -100.0;
std::vector<int>   g_view; // indices into g_rows: the filter and the order
int                g_filter = 0, g_sort = 0; // 0 todos, 1 obtenidos, 2 sin obtener; 0 grado, 1 fecha
int                g_sel_id      = -1;
float              g_scroll = 0, g_scroll_target = 0;
double             g_opened_at = 0, g_list_at = 0;
std::map<int, float> g_hot;
ImVec2             g_last_mouse {-1, -1};
float              g_pill_x = -1, g_pill_w = 0;

void EnsureArt(float unit) {
	auto& job = Job();
	if (g_tex != nullptr && g_tex_ctx != ImGui::GetCurrentContext()) {
		g_tex = nullptr; // (another context: its backend owns the old one)
		g_art.reset();
		g_art_requested = false;
	}
	if (g_tex != nullptr) return;
	if (!g_art_requested && !job.running.load()) {
		g_art_requested = true;
		job.failed      = false;
		job.running     = true;
		// The icons drawn at about 112 units (and up to 6 % larger when hovered): that many pixels, 128 - 256.
		const int cell = std::clamp((static_cast<int>(std::ceil(118.0f * unit)) + 15) / 16 * 16, 128, 256);
		std::thread([cell] {
			auto art = std::make_shared<Libs::Trophy::Art>();
			const bool ok = Libs::Trophy::BuildArt(cell, art.get());
			auto&      j  = Job();
			{
				std::scoped_lock lock(j.mutex);
				if (ok && !art->pixels.empty()) j.ready = std::move(art);
				else j.failed = true;
			}
			j.running = false;
			Bump();
		}).detach();
	}
	std::shared_ptr<Libs::Trophy::Art> art;
	{
		std::scoped_lock lock(job.mutex);
		art = std::move(job.ready);
	}
	if (art == nullptr || art->width <= 0 || art->height <= 0) return;
	auto* tex = IM_NEW(ImTextureData)();
	tex->Create(ImTextureFormat_RGBA32, art->width, art->height);
	std::memcpy(tex->GetPixels(), art->pixels.data(), static_cast<size_t>(art->width) * art->height * 4);
	tex->UseColors = true;
	ImGui::RegisterUserTexture(tex);
	art->pixels.clear();
	art->pixels.shrink_to_fit();
	g_tex     = tex;
	g_tex_ctx = ImGui::GetCurrentContext();
	g_art     = std::move(art);
}

// The icon's UV in the atlas (false: no icon for it).
bool IconUv(int slot, ImVec2* uv0, ImVec2* uv1) {
	if (g_tex == nullptr || g_art == nullptr || slot < 0) return false;
	const auto& a = *g_art;
	const float x = static_cast<float>((slot % a.cols) * a.stride + a.pad), y = static_cast<float>((slot / a.cols) * a.stride + a.pad);
	*uv0          = {x / static_cast<float>(a.width), y / static_cast<float>(a.height)};
	*uv1          = {(x + static_cast<float>(a.cell)) / static_cast<float>(a.width), (y + static_cast<float>(a.cell)) / static_cast<float>(a.height)};
	return true;
}
int SlotOf(int id) {
	if (g_art == nullptr) return -1;
	const auto it = g_art->slots.find(id);
	return it == g_art->slots.end() ? -1 : it->second;
}

int GradeRank(int g) {
	return g >= 1 && g <= 4 ? g : 5;
}

void Rebuild() {
	g_view.clear();
	for (int i = 0; i < static_cast<int>(g_rows.size()); i++) {
		const auto& r = g_rows[static_cast<size_t>(i)];
		if ((g_filter == 1 && !r.earned) || (g_filter == 2 && r.earned)) continue;
		g_view.push_back(i);
	}
	std::stable_sort(g_view.begin(), g_view.end(), [](int ia, int ib) {
		const auto& a = g_rows[static_cast<size_t>(ia)];
		const auto& b = g_rows[static_cast<size_t>(ib)];
		if (g_sort == 1) {
			if (a.earned != b.earned) return a.earned;
			if (a.earned && a.time != b.time) return a.time > b.time; // newest first
		}
		if (GradeRank(a.info.grade) != GradeRank(b.info.grade)) return GradeRank(a.info.grade) < GradeRank(b.info.grade);
		return a.info.id < b.info.id;
	});
}

void Refresh(bool force) {
	const double now = ImGui::GetTime();
	if (!force && now - g_refreshed_at < 1.0) return;
	std::vector<Libs::Trophy::Info> all;
	std::map<int, int64_t>          times;
	const int                       status = Libs::Trophy::TrySnapshot(&all, &times);
	if (status == 0) return; // busy: the last copy, again next frame
	g_refreshed_at = now;
	g_status       = status;
	std::vector<Row> rows;
	if (status == 1) {
		rows.reserve(all.size());
		for (auto& i: all) {
			Row r;
			const auto it = times.find(i.id);
			r.earned      = it != times.end();
			r.time        = r.earned ? it->second : 0;
			r.info        = std::move(i);
			rows.push_back(std::move(r));
		}
	}
	bool same = rows.size() == g_rows.size();
	for (size_t i = 0; same && i < rows.size(); i++)
		same = rows[i].info.id == g_rows[i].info.id && rows[i].earned == g_rows[i].earned && rows[i].time == g_rows[i].time;
	if (same && !force) return;
	g_rows = std::move(rows);
	Rebuild();
}

void Open() {
	g_opened_at = g_list_at = ImGui::GetTime();
	g_scroll = g_scroll_target = 0;
	g_sel_id                   = -1;
	g_hot.clear();
	g_pill_x = -1;
	Refresh(true);
}

float Ease(float t) {
	t = std::clamp(t, 0.0f, 1.0f);
	return 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
}
ImU32 Col(int r, int g, int b, float a) {
	return IM_COL32(r, g, b, static_cast<int>(std::clamp(a, 0.0f, 1.0f) * 255.0f));
}
ImU32 WithAlpha(ImU32 c, float a) {
	const int base = static_cast<int>((c >> IM_COL32_A_SHIFT) & 0xFF);
	return (c & ~IM_COL32_A_MASK) | (static_cast<ImU32>(std::clamp(static_cast<float>(base) * a, 0.0f, 255.0f)) << IM_COL32_A_SHIFT);
}

ImFont* FontFor(const std::string& text, ImFont* latin) {
	static ImFont* cjk = nullptr;
	static bool    looked = false;
	static ImGuiContext* ctx = nullptr;
	if (!looked || ctx != ImGui::GetCurrentContext()) {
		looked = true;
		ctx    = ImGui::GetCurrentContext();
		cjk    = nullptr;
		for (const char* n: {"msyh", "simhei", "simsun"})
			if (cjk == nullptr) cjk = FindFont(n);
	}
	for (const unsigned char c: text)
		if (c >= 0xd0 && cjk != nullptr) return cjk; // (Cyrillic, CJK ...: the system's CJK face)
	return latin;
}

// The text cut to max_w with "..." (one line).
void TextFit(ImDrawList* dl, ImFont* font, float size, ImVec2 at, ImU32 col, const std::string& text, float max_w) {
	if (max_w <= 0) return;
	if (font->CalcTextSizeA(size, FLT_MAX, 0.0f, text.c_str()).x <= max_w) {
		dl->AddText(font, size, at, col, text.c_str());
		return;
	}
	const float dots = font->CalcTextSizeA(size, FLT_MAX, 0.0f, "...").x;
	size_t      lo = 0, hi = text.size();
	while (lo < hi) { // the longest prefix (on a UTF-8 boundary) that fits
		size_t mid = (lo + hi + 1) / 2;
		while (mid > 0 && mid < text.size() && (static_cast<unsigned char>(text[mid]) & 0xC0) == 0x80) mid--;
		if (mid <= lo) {
			hi = lo;
			break;
		}
		if (font->CalcTextSizeA(size, FLT_MAX, 0.0f, text.c_str(), text.c_str() + mid).x + dots <= max_w) lo = mid;
		else hi = mid - 1;
	}
	while (lo > 0 && lo < text.size() && (static_cast<unsigned char>(text[lo]) & 0xC0) == 0x80) lo--;
	const std::string cut = text.substr(0, lo) + "...";
	dl->AddText(font, size, at, col, cut.c_str());
}

// A padlock in the square (x, y, s).
void DrawLock(ImDrawList* dl, float x, float y, float s, ImU32 col) {
	dl->PathArcTo({x + s * 0.5f, y + s * 0.42f}, s * 0.22f, IM_PI, 2.0f * IM_PI, 12);
	dl->PathStroke(col, 0, s * 0.11f);
	dl->AddLine({x + s * 0.28f, y + s * 0.42f}, {x + s * 0.28f, y + s * 0.5f}, col, s * 0.11f);
	dl->AddLine({x + s * 0.72f, y + s * 0.42f}, {x + s * 0.72f, y + s * 0.5f}, col, s * 0.11f);
	dl->AddRectFilled({x + s * 0.16f, y + s * 0.48f}, {x + s * 0.84f, y + s * 0.94f}, col, s * 0.1f);
}

// The shimmer: a bright diagonal band sweeping over (p0, p1) once every few seconds.
void Shimmer(ImDrawList* dl, ImVec2 p0, ImVec2 p1, float t, float phase, float alpha) {
	const float period = 4.2f, sweep = 0.95f;
	const float u      = std::fmod(t + phase, period) / sweep;
	if (u >= 1.0f || alpha <= 0.0f) return;
	const float w = p1.x - p0.x, h = p1.y - p0.y;
	const float cx = p0.x - w * 0.6f + u * w * 2.2f, slant = h * 0.45f;
	dl->PushClipRect(p0, p1, true);
	for (int k = 0; k < 3; k++) {
		const float half = w * (k == 0 ? 0.16f : k == 1 ? 0.09f : 0.035f);
		const ImU32 c    = Col(255, 255, 255, alpha * (k == 0 ? 0.08f : k == 1 ? 0.12f : 0.20f));
		dl->AddQuadFilled({cx - half + slant, p0.y}, {cx + half + slant, p0.y}, {cx + half - slant, p1.y}, {cx - half - slant, p1.y}, c);
	}
	dl->PopClipRect();
}

std::string DateText(int64_t t) {
	if (t <= 0) return I18n::T("Obtenido");
	const std::time_t tt = static_cast<std::time_t>(t);
	std::tm           tm {};
#if defined(_WIN32)
	if (localtime_s(&tm, &tt) != 0) return I18n::T("Obtenido");
#else
	if (localtime_r(&tt, &tm) == nullptr) return I18n::T("Obtenido");
#endif
	char date[64], line[160];
	std::snprintf(date, sizeof(date), "%02d/%02d/%04d %02d:%02d", tm.tm_mday, tm.tm_mon + 1, tm.tm_year + 1900, tm.tm_hour, tm.tm_min);
	std::snprintf(line, sizeof(line), I18n::T("Obtenido el %s"), date);
	return line;
}

const char* GradeName(int g) {
	switch (g) {
		case 1: return I18n::T("Platino");
		case 2: return I18n::T("Oro");
		case 3: return I18n::T("Plata");
		case 4: return I18n::T("Bronce");
		default: return "";
	}
}
int Points(int g) {
	return g == 1 ? 300 : g == 2 ? 90 : g == 3 ? 30 : g == 4 ? 15 : 0;
}

bool Clicked(ImVec2 p0, ImVec2 p1) {
	const auto& io = ImGui::GetIO();
	return ImGui::IsMouseClicked(ImGuiMouseButton_Left) && io.MousePos.x >= p0.x && io.MousePos.x < p1.x && io.MousePos.y >= p0.y &&
	       io.MousePos.y < p1.y;
}
bool Hovered(ImVec2 p0, ImVec2 p1) {
	const auto& io = ImGui::GetIO();
	return io.MousePos.x >= p0.x && io.MousePos.x < p1.x && io.MousePos.y >= p0.y && io.MousePos.y < p1.y;
}
bool Pressed(std::initializer_list<ImGuiKey> keys) {
	for (const ImGuiKey k: keys)
		if (ImGui::IsKeyPressed(k, true)) return true;
	return false;
}

// Returns false when the page asks to go back (its Volver).
bool Draw(float width, float height, float unit) {
	const auto& io  = ImGui::GetIO();
	const float u   = unit;
	const float now = static_cast<float>(ImGui::GetTime());
	const float dt  = std::clamp(io.DeltaTime, 0.0f, 0.1f);
	EnsureArt(unit);
	Refresh(false);
	bool back = false;

	// Input: the filters (L1/R1, Q/E, PageUp/PageDown), the order (Y, Tab).
	if (g_trophy_tab_delta != 0) {
		g_filter           = ((g_filter + g_trophy_tab_delta) % 3 + 3) % 3;
		g_trophy_tab_delta = 0;
		Rebuild();
		g_list_at = ImGui::GetTime();
		g_scroll_target = 0;
	}
	if (Pressed({ImGuiKey_GamepadFaceUp, ImGuiKey_Tab})) {
		g_sort = 1 - g_sort;
		Rebuild();
		g_list_at = ImGui::GetTime();
	}

	// The page: fades and slides in.
	const float in = Ease((now - static_cast<float>(g_opened_at)) / 0.32f);
	ImGui::SetNextWindowPos({0, 0});
	ImGui::SetNextWindowSize({width, height});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::Begin("##trophy_page", nullptr,
	             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
	                 ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus);
	ImDrawList* dl   = ImGui::GetWindowDrawList();
	ImFont*     body = FindFont("segoeui");
	if (body == nullptr) body = ImGui::GetFont();
	dl->AddRectFilled({0, 0}, {width, height}, Col(0, 0, 0, 0.55f * in));
	DsStyle::Backdrop(dl, {0, 0}, {width, height}, in, Souls::Time()); // (trucos11) the same stone-and-fog backdrop as the other pages
	DsStyle::Motes(dl, {0, 0}, {width, height}, in * 0.7f, Souls::Time(), 30);
	const int trophy_first = dl->VtxBuffer.Size; (void)trophy_first;

	const float margin = std::max(24.0f * u, std::min(width, height) * 0.035f);
	const float pw     = std::min(width - margin * 2.0f, 1680.0f * u);
	const float ph     = height - margin * 2.0f;
	const float px0    = (width - pw) * 0.5f;
	const float py0    = margin + (1.0f - in) * 36.0f * u;
	const float pad    = 28.0f * u;
	const float round  = 18.0f * u;
	const ImU32 gold   = Col(222, 196, 140, in);
	// shadow, card, header band, border
	for (int k = 3; k >= 1; k--)
		dl->AddRectFilled({px0 - k * 4.0f * u, py0 - k * 3.0f * u}, {px0 + pw + k * 4.0f * u, py0 + ph + k * 6.0f * u}, Col(0, 0, 0, 0.10f * in), round + k * 4.0f * u);
	dl->AddRectFilled({px0, py0}, {px0 + pw, py0 + ph}, Col(17, 18, 22, 0.97f * in), round);
	const float header_h = 168.0f * u;
	dl->AddRectFilledMultiColor({px0 + round, py0}, {px0 + pw - round, py0 + header_h}, Col(46, 40, 30, 0.55f * in), Col(30, 32, 40, 0.55f * in),
	                            Col(17, 18, 22, 0.0f), Col(17, 18, 22, 0.0f));
	dl->AddRect({px0, py0}, {px0 + pw, py0 + ph}, Col(222, 196, 140, 0.28f * in), round, 0, 1.2f * u);

	// Totals.
	int total = 0, have = 0, earned = 0;
	int grade_total[5] = {}, grade_have[5] = {};
	for (const auto& r: g_rows) {
		const int g = r.info.grade >= 1 && r.info.grade <= 4 ? r.info.grade : 0;
		total += Points(g);
		grade_total[g]++;
		if (r.earned) {
			have += Points(g);
			earned++;
			grade_have[g]++;
		}
	}
	const float frac = total > 0 ? static_cast<float>(have) / static_cast<float>(total) : 0.0f;

	// Header: the game's icon, its title, the progress, the counts per grade.
	{
		const float isz = 116.0f * u;
		const float ix = px0 + pad, iy = py0 + (header_h - isz) * 0.5f;
		ImVec2      uv0, uv1;
		dl->AddRectFilled({ix - 2 * u, iy - 2 * u}, {ix + isz + 2 * u, iy + isz + 2 * u}, Col(222, 196, 140, 0.35f * in), 16.0f * u);
		if (g_art != nullptr && IconUv(g_art->game_slot, &uv0, &uv1))
			dl->AddImageRounded(g_tex->GetTexRef(), {ix, iy}, {ix + isz, iy + isz}, uv0, uv1, Col(255, 255, 255, in), 14.0f * u);
		else {
			dl->AddRectFilled({ix, iy}, {ix + isz, iy + isz}, Col(40, 42, 50, in), 14.0f * u);
			Libs::Trophy::DrawCup(dl, ix + isz * 0.15f, iy + isz * 0.12f, isz * 0.7f, Libs::Trophy::CupColor(2, in), in);
		}
		const float tx = ix + isz + 26.0f * u;
		const float grades_w = std::min(4.0f * 112.0f * u, pw * 0.34f);
		const float gx0      = px0 + pw - pad - grades_w;
		const float text_w   = gx0 - 30.0f * u - tx;
		dl->AddText(body, 16.0f * u, {tx, iy + 2 * u}, gold, I18n::T("TROFEOS"));
		std::string title = Libs::Trophy::GameTitle();
		if (title.empty()) title = "Demon's Souls";
		TextFit(dl, FontFor(title, body), 38.0f * u, {tx, iy + 20 * u}, Col(245, 242, 235, in), title, text_w);
		char line[160];
		std::snprintf(line, sizeof(line), I18n::T("%d de %d trofeos obtenidos"), earned, static_cast<int>(g_rows.size()));
		TextFit(dl, body, 18.0f * u, {tx, iy + 66 * u}, Col(200, 198, 192, in), line, text_w);
		// The bar: a track, the gold fill (animated in), a light sweeping along it and the percentage.
		char pct[16];
		std::snprintf(pct, sizeof(pct), "%d%%", static_cast<int>(std::floor(frac * 100.0f + 0.0001f)));
		const float pct_w = body->CalcTextSizeA(26.0f * u, FLT_MAX, 0.0f, "100%").x;
		const float bx0 = tx, bx1 = std::max(tx + 40 * u, gx0 - 40.0f * u - pct_w - 14.0f * u), by = iy + isz - 14.0f * u, bh = 12.0f * u;
		dl->AddRectFilled({bx0, by}, {bx1, by + bh}, Col(255, 255, 255, 0.10f * in), bh * 0.5f);
		const float shown = frac * Ease((now - static_cast<float>(g_opened_at) - 0.15f) / 0.9f);
		if (shown > 0.0f) {
			const float fx = bx0 + (bx1 - bx0) * shown;
			dl->AddRectFilledMultiColor({bx0 + bh * 0.5f, by}, {std::max(bx0 + bh * 0.5f, fx - bh * 0.5f), by + bh}, Col(176, 132, 64, in),
			                            Col(236, 204, 120, in), Col(236, 204, 120, in), Col(176, 132, 64, in));
			dl->AddCircleFilled({bx0 + bh * 0.5f, by + bh * 0.5f}, bh * 0.5f, Col(176, 132, 64, in), 16);
			dl->AddCircleFilled({std::max(bx0 + bh * 0.5f, fx - bh * 0.5f), by + bh * 0.5f}, bh * 0.5f, Col(236, 204, 120, in), 16);
			Shimmer(dl, {bx0, by}, {fx, by + bh}, now, 0.0f, in);
		}
		dl->AddText(body, 26.0f * u, {bx1 + 14.0f * u, by + bh * 0.5f - 15.0f * u}, gold, pct);
		// The grades.
		const float cell_w = grades_w / 4.0f;
		for (int g = 1; g <= 4; g++) {
			const float cx  = gx0 + cell_w * static_cast<float>(g - 1);
			const float cup = std::min(44.0f * u, cell_w * 0.5f);
			const float cy  = iy + 8.0f * u;
			if (g == 1 && grade_have[1] > 0) {
				const float p = 0.5f + 0.5f * std::sin(now * 2.4f);
				for (int k = 0; k < 4; k++)
					dl->AddCircleFilled({cx + cell_w * 0.5f, cy + cup * 0.5f}, cup * (0.55f + 0.14f * k + 0.08f * p), Col(170, 200, 235, (0.07f - 0.014f * k) * in), 32);
			}
			Libs::Trophy::DrawCup(dl, cx + (cell_w - cup) * 0.5f, cy, cup, Libs::Trophy::CupColor(g, in), in);
			char n[32];
			std::snprintf(n, sizeof(n), "%d/%d", grade_have[g], grade_total[g]);
			const float nw = body->CalcTextSizeA(22.0f * u, FLT_MAX, 0.0f, n).x;
			dl->AddText(body, 22.0f * u, {cx + (cell_w - nw) * 0.5f, cy + cup + 8.0f * u}, Col(240, 238, 232, in), n);
			const char* gn = GradeName(g);
			const float gw = body->CalcTextSizeA(15.0f * u, FLT_MAX, 0.0f, gn).x;
			dl->AddText(body, 15.0f * u, {cx + (cell_w - gw) * 0.5f, cy + cup + 36.0f * u}, Libs::Trophy::CupColor(g, in), gn);
		}
	}

	// The filters and the order.
	const float fy0 = py0 + header_h, fh = 46.0f * u;
	{
		const char* names[3] = {I18n::T("Todos"), I18n::T("Obtenidos"), I18n::T("Sin obtener")};
		const int   counts[3] = {static_cast<int>(g_rows.size()), earned, static_cast<int>(g_rows.size()) - earned};
		float       x = px0 + pad;
		float       sel_x = x, sel_w = 0;
		struct Pill {
			float x, w;
			std::string text;
		};
		Pill pills[3];
		for (int i = 0; i < 3; i++) {
			pills[i].text = std::string(names[i]) + "  " + std::to_string(counts[i]);
			pills[i].w    = body->CalcTextSizeA(19.0f * u, FLT_MAX, 0.0f, pills[i].text.c_str()).x + 36.0f * u;
			pills[i].x    = x;
			if (i == g_filter) sel_x = x, sel_w = pills[i].w;
			x += pills[i].w + 10.0f * u;
		}
		const float k = 1.0f - std::exp(-dt * 16.0f);
		if (g_pill_x < 0) g_pill_x = sel_x, g_pill_w = sel_w;
		g_pill_x += (sel_x - g_pill_x) * k;
		g_pill_w += (sel_w - g_pill_w) * k;
		dl->AddRectFilled({g_pill_x, fy0 + 4 * u}, {g_pill_x + g_pill_w, fy0 + fh - 4 * u}, Col(222, 196, 140, 0.95f * in), fh);
		for (int i = 0; i < 3; i++) {
			const ImVec2 p0 {pills[i].x, fy0 + 4 * u}, p1 {pills[i].x + pills[i].w, fy0 + fh - 4 * u};
			const bool   hot = Hovered(p0, p1);
			if (i != g_filter) dl->AddRect(p0, p1, Col(255, 255, 255, (hot ? 0.45f : 0.18f) * in), fh, 0, 1.0f * u);
			if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
			dl->AddText(body, 19.0f * u, {pills[i].x + 18.0f * u, fy0 + fh * 0.5f - 12.0f * u},
			            i == g_filter ? Col(24, 22, 18, in) : Col(225, 222, 215, in), pills[i].text.c_str());
			if (Clicked(p0, p1) && i != g_filter) {
				g_filter = i;
				Rebuild();
				g_list_at       = ImGui::GetTime();
				g_scroll_target = 0;
			}
		}
		const std::string sort = std::string(I18n::T("Ordenar:")) + " " + I18n::T(g_sort == 0 ? "Grado" : "Fecha");
		const float sw = body->CalcTextSizeA(19.0f * u, FLT_MAX, 0.0f, sort.c_str()).x + 36.0f * u;
		const ImVec2 s0 {px0 + pw - pad - sw, fy0 + 4 * u}, s1 {px0 + pw - pad, fy0 + fh - 4 * u};
		if (s0.x > x) {
			const bool hot = Hovered(s0, s1);
			dl->AddRect(s0, s1, Col(222, 196, 140, (hot ? 0.8f : 0.4f) * in), fh, 0, 1.0f * u);
			dl->AddText(body, 19.0f * u, {s0.x + 18.0f * u, fy0 + fh * 0.5f - 12.0f * u}, gold, sort.c_str());
			if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
			if (Clicked(s0, s1)) {
				g_sort = 1 - g_sort;
				Rebuild();
				g_list_at = ImGui::GetTime();
			}
		}
		dl->AddLine({px0 + pad, fy0 + fh + 10 * u}, {px0 + pw - pad, fy0 + fh + 10 * u}, Col(255, 255, 255, 0.08f * in), 1.0f * u);
	}

	// The footer: the keys, and Volver.
	const float foot_h = 46.0f * u;
	{
		const float y    = py0 + ph - foot_h;
		const char* hint = I18n::T("L1/R1 o Q/E: filtro   \xC2\xB7   Y o Tab: orden   \xC2\xB7   B, Esc o F2: volver");
		const char* vb   = I18n::T("Volver");
		const float vw   = body->CalcTextSizeA(18.0f * u, FLT_MAX, 0.0f, vb).x + 40.0f * u;
		const ImVec2 v0 {px0 + pad, y + 6 * u}, v1 {px0 + pad + vw, y + foot_h - 10 * u};
		const bool   hot = Hovered(v0, v1);
		dl->AddRect(v0, v1, Col(255, 255, 255, (hot ? 0.6f : 0.25f) * in), 8.0f * u, 0, 1.0f * u);
		dl->AddText(body, 18.0f * u, {v0.x + 20.0f * u, v0.y + (v1.y - v0.y) * 0.5f - 11.0f * u}, Col(230, 228, 222, in), vb);
		if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		if (Clicked(v0, v1)) back = true;
		const float hw = body->CalcTextSizeA(16.0f * u, FLT_MAX, 0.0f, hint).x;
		if (v1.x + 20 * u + hw < px0 + pw - pad)
			dl->AddText(body, 16.0f * u, {px0 + pw - pad - hw, y + foot_h * 0.5f - 12.0f * u}, Col(160, 158, 152, in), hint);
	}

	// The list.
	const ImVec2 l0 {px0 + pad, fy0 + fh + 22.0f * u}, l1 {px0 + pw - pad, py0 + ph - foot_h - 8.0f * u};
	const float  view_h = std::max(0.0f, l1.y - l0.y);
	if (g_status != 1) {
		const char* msg = g_status == -1 ? I18n::T("Cargando los trofeos...")
		                                 : I18n::T("No se encontraron los trofeos del juego (falta sce_sys/trophy2/trophy00.ucp).");
		const float mw  = std::min(body->CalcTextSizeA(22.0f * u, FLT_MAX, 0.0f, msg).x, l1.x - l0.x);
		dl->AddText(body, 22.0f * u, {(l0.x + l1.x - mw) * 0.5f, l0.y + view_h * 0.4f}, Col(210, 206, 198, in), msg, nullptr, l1.x - l0.x);
	} else if (g_view.empty()) {
		const char* msg = g_filter == 1 ? I18n::T("A\xC3\xBAn no has obtenido ning\xC3\xBAn trofeo.") : I18n::T("\xC2\xA1Has obtenido todos los trofeos!");
		const float mw  = body->CalcTextSizeA(22.0f * u, FLT_MAX, 0.0f, msg).x;
		dl->AddText(body, 22.0f * u, {(l0.x + l1.x - mw) * 0.5f, l0.y + view_h * 0.4f}, Col(210, 206, 198, in), msg);
	} else {
		const int   cols  = (l1.x - l0.x) >= 1300.0f * u ? 2 : 1;
		const float gap   = 16.0f * u;
		const float cw    = ((l1.x - l0.x) - gap * static_cast<float>(cols - 1)) / static_cast<float>(cols);
		const float ch    = 138.0f * u;
		const int   count = static_cast<int>(g_view.size());
		const int   rows  = (count + cols - 1) / cols;
		const float content = static_cast<float>(rows) * (ch + gap) - gap;
		const float max_scroll = std::max(0.0f, content - view_h + 4.0f * u);

		// The selection: kept by id; arrows, D-pad and the stick move it, the mouse picks what it moves over.
		int sel = -1;
		for (int i = 0; i < count; i++)
			if (g_rows[static_cast<size_t>(g_view[static_cast<size_t>(i)])].info.id == g_sel_id) sel = i;
		bool moved = false;
		const auto step = [&](int d) {
			sel   = sel < 0 ? 0 : std::clamp(sel + d, 0, count - 1);
			moved = true;
		};
		if (Pressed({ImGuiKey_DownArrow, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown})) step(cols);
		if (Pressed({ImGuiKey_UpArrow, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp})) step(-cols);
		if (Pressed({ImGuiKey_RightArrow, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight})) step(1);
		if (Pressed({ImGuiKey_LeftArrow, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft})) step(-1);
		if (Pressed({ImGuiKey_Home})) {
			sel   = 0;
			moved = true;
		}
		if (Pressed({ImGuiKey_End})) {
			sel   = count - 1;
			moved = true;
		}
		if (Hovered(l0, l1) && io.MouseWheel != 0.0f) g_scroll_target -= io.MouseWheel * (ch + gap) * 0.8f;
		if (moved && sel >= 0) {
			const float top = static_cast<float>(sel / cols) * (ch + gap);
			if (top < g_scroll_target) g_scroll_target = top - 8.0f * u;
			if (top + ch > g_scroll_target + view_h) g_scroll_target = top + ch - view_h + 8.0f * u;
		}
		g_scroll_target = std::clamp(g_scroll_target, 0.0f, max_scroll);
		g_scroll += (g_scroll_target - g_scroll) * (1.0f - std::exp(-dt * 14.0f));
		if (std::fabs(g_scroll_target - g_scroll) < 0.25f) g_scroll = g_scroll_target;
		const bool mouse_moved = io.MousePos.x != g_last_mouse.x || io.MousePos.y != g_last_mouse.y;
		g_last_mouse           = io.MousePos;
		if (mouse_moved && Hovered(l0, l1)) {
			const float lx = io.MousePos.x - l0.x, ly = io.MousePos.y - l0.y + g_scroll;
			const int   c = static_cast<int>(lx / (cw + gap)), r = static_cast<int>(ly / (ch + gap));
			const bool  inside = lx - static_cast<float>(c) * (cw + gap) < cw && ly - static_cast<float>(r) * (ch + gap) < ch;
			const int   i = r * cols + c;
			if (inside && c < cols && i >= 0 && i < count) sel = i;
		}
		g_sel_id = sel >= 0 ? g_rows[static_cast<size_t>(g_view[static_cast<size_t>(sel)])].info.id : -1;

		dl->PushClipRect(l0, l1, true);
		const float list_t = now - static_cast<float>(g_list_at);
		const int   first  = std::max(0, static_cast<int>(g_scroll / (ch + gap)) - 1) * cols;
		for (int i = first; i < count; i++) {
			const int   r   = i / cols, c = i % cols;
			const float y0  = l0.y + static_cast<float>(r) * (ch + gap) - g_scroll;
			if (y0 > l1.y) break;
			const auto& row = g_rows[static_cast<size_t>(g_view[static_cast<size_t>(i)])];
			const int   id  = row.info.id;
			const int   g   = row.info.grade >= 1 && row.info.grade <= 4 ? row.info.grade : 0;
			const bool  secret = row.info.hidden && !row.earned;
			// Stagger: each card a little after the one before (the first screen only).
			const int   order  = std::max(0, i - first);
			const float appear = Ease((list_t - 0.10f - 0.035f * static_cast<float>(std::min(order, 16))) / 0.35f);
			const float a      = in * appear;
			if (a <= 0.0f) continue;
			float& hot = g_hot[id];
			hot += ((i == sel ? 1.0f : 0.0f) - hot) * (1.0f - std::exp(-dt * 12.0f));
			const float x0 = l0.x + static_cast<float>(c) * (cw + gap);
			const float yy = y0 + (1.0f - appear) * 22.0f * u;
			const float sc = 1.0f + 0.018f * hot;
			const ImVec2 mid {x0 + cw * 0.5f, yy + ch * 0.5f};
			const ImVec2 p0 {mid.x - cw * 0.5f * sc, mid.y - ch * 0.5f * sc}, p1 {mid.x + cw * 0.5f * sc, mid.y + ch * 0.5f * sc};
			const ImU32  gc = Libs::Trophy::CupColor(g, a);
			const float  cr = 14.0f * u;
			// glow (selection), card, border
			for (int k = 3; k >= 1; k--)
				if (hot > 0.01f)
					dl->AddRect({p0.x - k * 2.5f * u, p0.y - k * 2.5f * u}, {p1.x + k * 2.5f * u, p1.y + k * 2.5f * u}, WithAlpha(gc, hot * (0.22f - 0.05f * k)),
					            cr + k * 2.5f * u, 0, 2.5f * u);
			dl->AddRectFilled(p0, p1, row.earned ? Col(31, 32, 38, a) : Col(24, 25, 29, a), cr);
			if (row.earned)
				dl->AddRectFilledMultiColor({p0.x + cr, p0.y}, {p1.x - cr, p0.y + ch * 0.5f}, WithAlpha(gc, 0.06f), Col(0, 0, 0, 0), Col(0, 0, 0, 0),
				                            WithAlpha(gc, 0.02f));
			dl->AddRect(p0, p1, hot > 0.01f ? WithAlpha(gc, 0.35f + 0.6f * hot) : Col(255, 255, 255, 0.07f * a), cr, 0, (1.0f + 1.0f * hot) * u);

			// The icon (larger when focused); a platinum's halo; the shimmer on earned ones; locked: dimmed, a padlock.
			const float isz = 112.0f * u * (1.0f + 0.06f * hot);
			const float icx = p0.x + 13.0f * u + 56.0f * u, icy = (p0.y + p1.y) * 0.5f;
			const ImVec2 i0 {icx - isz * 0.5f, icy - isz * 0.5f}, i1 {icx + isz * 0.5f, icy + isz * 0.5f};
			const float  ir = 10.0f * u;
			if (g == 1 && row.earned) {
				const float p = 0.5f + 0.5f * std::sin(now * 2.2f + static_cast<float>(id));
				for (int k = 1; k <= 5; k++)
					dl->AddRectFilled({i0.x - k * 2.6f * u * (1 + p * 0.4f), i0.y - k * 2.6f * u * (1 + p * 0.4f)},
					                  {i1.x + k * 2.6f * u * (1 + p * 0.4f), i1.y + k * 2.6f * u * (1 + p * 0.4f)}, Col(170, 205, 245, (0.10f - 0.018f * k) * a * (0.6f + 0.4f * p)),
					                  ir + k * 2.6f * u);
			}
			ImVec2 uv0, uv1;
			const bool has_icon = !secret && IconUv(SlotOf(id), &uv0, &uv1);
			dl->AddRectFilled(i0, i1, Col(40, 42, 50, a), ir);
			if (has_icon) {
				const ImU32 tint = row.earned ? Col(255, 255, 255, a) : Col(120, 120, 124, a);
				dl->AddImageRounded(g_tex->GetTexRef(), i0, i1, uv0, uv1, tint, ir);
				if (!row.earned) dl->AddRectFilled(i0, i1, Col(52, 52, 56, 0.55f * a), ir); // (desaturated)
				else Shimmer(dl, i0, i1, now, static_cast<float>(id) * 0.61f, a * (g == 1 ? 1.4f : 1.0f));
			} else if (secret) {
				const char* q  = "?";
				const float qs = isz * 0.55f;
				const ImVec2 qz = body->CalcTextSizeA(qs, FLT_MAX, 0.0f, q);
				dl->AddText(body, qs, {icx - qz.x * 0.5f, icy - qz.y * 0.5f}, Col(120, 120, 126, a), q);
			} else {
				Libs::Trophy::DrawCup(dl, i0.x + isz * 0.15f, i0.y + isz * 0.12f, isz * 0.7f,
				                      row.earned ? gc : Libs::Trophy::CupColor(0, a), a);
			}
			if (!row.earned) {
				const float ls = 30.0f * u;
				dl->AddCircleFilled({i1.x - ls * 0.5f - 2 * u, i1.y - ls * 0.5f - 2 * u}, ls * 0.62f, Col(14, 14, 17, 0.9f * a), 24);
				DrawLock(dl, i1.x - ls - 2 * u + ls * 0.1f, i1.y - ls - 2 * u + ls * 0.04f, ls * 0.8f, Col(200, 200, 205, a));
			}
			dl->AddRect(i0, i1, row.earned ? WithAlpha(gc, 0.55f) : Col(255, 255, 255, 0.06f * a), ir, 0, 1.0f * u);

			// The texts: the name, two lines of description, then the grade and the date (or Sin obtener).
			const float tx = p0.x + 13.0f * u + 112.0f * u + 22.0f * u, tw = p1.x - 18.0f * u - tx;
			const std::string name   = secret ? std::string(I18n::T("Trofeo oculto")) : row.info.name;
			const std::string detail = secret ? std::string(I18n::T("Se revelar\xC3\xA1 al obtenerlo.")) : row.info.detail;
			TextFit(dl, FontFor(name, body), 23.0f * u, {tx, p0.y + 14.0f * u}, row.earned ? Col(246, 244, 240, a) : Col(170, 170, 176, a), name, tw);
			{
				ImFont*      f  = FontFor(detail, body);
				const float  fs = 17.0f * u;
				const ImVec4 clip {tx, p0.y + 46.0f * u, tx + tw, p0.y + 46.0f * u + fs * 2.0f * 1.18f + 2.0f * u};
				dl->AddText(f, fs, {tx, p0.y + 46.0f * u}, row.earned ? Col(196, 196, 204, a) : Col(128, 128, 136, a), detail.c_str(), nullptr, tw, &clip);
			}
			{
				const float ly  = p1.y - 34.0f * u;
				const float cup = 22.0f * u;
				Libs::Trophy::DrawCup(dl, tx, ly, cup, row.earned ? gc : WithAlpha(gc, 0.55f), a);
				const char* gn = GradeName(g);
				dl->AddText(body, 16.0f * u, {tx + cup + 6.0f * u, ly + 1.0f * u}, row.earned ? gc : WithAlpha(gc, 0.6f), gn);
				const float gw   = body->CalcTextSizeA(16.0f * u, FLT_MAX, 0.0f, gn).x;
				const std::string when = row.earned ? DateText(row.time) : std::string(I18n::T("Sin obtener"));
				const float wx = tx + cup + 6.0f * u + gw + 10.0f * u;
				dl->AddText(body, 16.0f * u, {wx, ly + 1.0f * u}, Col(120, 120, 128, a), "\xC2\xB7");
				TextFit(dl, body, 16.0f * u, {wx + 14.0f * u, ly + 1.0f * u}, row.earned ? Col(222, 196, 140, a) : Col(140, 140, 148, a), when,
				        tx + tw - wx - 14.0f * u);
			}
		}
		dl->PopClipRect();
		// The scroll bar (where the view is), and fades at the list's edges.
		if (max_scroll > 0.0f) {
			const float bx = px0 + pw - pad * 0.5f, th = std::max(40.0f * u, view_h * view_h / (content + 4.0f * u));
			const float ty = l0.y + (view_h - th) * (g_scroll / max_scroll);
			dl->AddRectFilled({bx - 2 * u, l0.y}, {bx + 2 * u, l1.y}, Col(255, 255, 255, 0.05f * in), 2 * u);
			dl->AddRectFilled({bx - 2 * u, ty}, {bx + 2 * u, ty + th}, Col(222, 196, 140, 0.6f * in), 2 * u);
			if (g_scroll > 1.0f)
				dl->AddRectFilledMultiColor({l0.x, l0.y}, {l1.x, l0.y + 18 * u}, Col(17, 18, 22, in), Col(17, 18, 22, in), Col(17, 18, 22, 0), Col(17, 18, 22, 0));
			if (g_scroll < max_scroll - 1.0f)
				dl->AddRectFilledMultiColor({l0.x, l1.y - 18 * u}, {l1.x, l1.y}, Col(17, 18, 22, 0), Col(17, 18, 22, 0), Col(17, 18, 22, in), Col(17, 18, 22, in));
		}
	}
	// (trucos8) The entry as the other pages: zoom from 92 % about the centre with the fade (PageZoom's).
	if (in < 1.0f) Anim::ScaleVertices(dl, 0, {width * 0.5f, height * 0.5f}, 0.92f + 0.08f * in);
	ImGui::End();
	ImGui::PopStyleVar(2);
	return !back;
}

} // namespace TrophyPage


// ---- (ds-pc-port41) "Metricas" in the game: the emulator launcher's metricas.ps1 window drawn by the port's overlay, top-centre.
// The same rows in the same order, labels, colours (LimeGreen / Red / DeepSkyBlue / Orange / White / Khaki / Gold, names
// grey 150), Consolas Bold 15 / 16 / 34 / 10 px x S (S = screen height / 1080, 1..2.5), no background (the window was
// transparent), the 60 s frametime graph (green mean, orange peaks, 8.3 / 16.7 / 33.3 lines, 40 ms top). The same
// sources: NVML (what nvidia-smi reads) for GPU % / temperature / clock / VRAM, the processor's "% Processor
// Performance" counter x its maximum clock, CPU % from the system times, RAM and commit from Windows, and FPS / frame /
// latency / multiplier / upscaler from this run's log with metricas.ps1's own patterns. F8 shows / hides it.
namespace MetricsPanel {
struct Row {
	std::string name, value;
	ImU32       color;
};
constexpr ImU32 kGreen = IM_COL32(50, 205, 50, 255), kRed = IM_COL32(255, 0, 0, 255), kBlue = IM_COL32(0, 191, 255, 255),
                kOrange = IM_COL32(255, 165, 0, 255), kWhite = IM_COL32(255, 255, 255, 255), kKhaki = IM_COL32(240, 230, 140, 255),
                kGold = IM_COL32(255, 215, 0, 255), kName = IM_COL32(150, 150, 150, 255);
std::mutex          g_mutex;
std::vector<Row>    g_rows;
std::string         g_fps_text;
ImU32               g_fps_color = kWhite;
std::vector<double> g_ft, g_ft_max;
std::atomic<bool>   g_shown {false}, g_started {false};

ImU32 UseColor(double used, double total) {
	if (total <= 0) return kOrange;
	const double f = used / total;
	return f < 0.70 ? kGreen : f < 0.85 ? kGold : f < 0.95 ? kOrange : kRed;
}
std::string Fmt(const char* f, ...) {
	char    text[256];
	va_list args;
	va_start(args, f);
	std::vsnprintf(text, sizeof(text), f, args);
	va_end(args);
	return text;
}

#if defined(_WIN32)
// NVML (nvml.dll, the driver's: what nvidia-smi queries).
struct Nvml {
	bool  ok = false;
	void* dev = nullptr;
	int (*util)(void*, unsigned int*) = nullptr; // nvmlDeviceGetUtilizationRates (gpu, memory)
	int (*temp)(void*, int, unsigned int*) = nullptr;
	int (*clock)(void*, int, unsigned int*) = nullptr;
	int (*mem)(void*, unsigned long long*) = nullptr; // nvmlDeviceGetMemoryInfo (total, free, used)
	void Init() {
		HMODULE m = LoadLibraryW(L"nvml.dll");
		if (m == nullptr) {
			wchar_t sys[MAX_PATH] = {};
			GetSystemDirectoryW(sys, MAX_PATH);
			m = LoadLibraryW((std::wstring(sys) + L"\\nvml.dll").c_str());
		}
		if (m == nullptr) return;
		auto init  = reinterpret_cast<int (*)()>(reinterpret_cast<void*>(GetProcAddress(m, "nvmlInit_v2")));
		auto get   = reinterpret_cast<int (*)(unsigned int, void**)>(reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetHandleByIndex_v2")));
		util       = reinterpret_cast<int (*)(void*, unsigned int*)>(reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetUtilizationRates")));
		temp       = reinterpret_cast<int (*)(void*, int, unsigned int*)>(reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetTemperature")));
		clock      = reinterpret_cast<int (*)(void*, int, unsigned int*)>(reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetClockInfo")));
		mem        = reinterpret_cast<int (*)(void*, unsigned long long*)>(reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetMemoryInfo")));
		ok = init != nullptr && get != nullptr && util != nullptr && temp != nullptr && mem != nullptr && init() == 0 && get(0, &dev) == 0;
	}
};
// "\Processor Information(_Total)\% Processor Performance" (pdh.dll by name), as metricas.ps1's counter.
struct CpuPerf {
	void* query = nullptr;
	void* counter = nullptr;
	long (*collect)(void*) = nullptr;
	long (*value)(void*, unsigned long, unsigned long*, void*) = nullptr;
	int max_mhz = 0;
	void Init() {
		HKEY key = nullptr;
		if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key) == ERROR_SUCCESS) {
			DWORD mhz = 0, size = sizeof(mhz);
			if (RegQueryValueExW(key, L"~MHz", nullptr, nullptr, reinterpret_cast<BYTE*>(&mhz), &size) == ERROR_SUCCESS) max_mhz = static_cast<int>(mhz);
			RegCloseKey(key);
		}
		HMODULE m = LoadLibraryW(L"pdh.dll");
		if (m == nullptr) return;
		auto open = reinterpret_cast<long (*)(const wchar_t*, uintptr_t, void**)>(reinterpret_cast<void*>(GetProcAddress(m, "PdhOpenQueryW")));
		auto add  = reinterpret_cast<long (*)(void*, const wchar_t*, uintptr_t, void**)>(reinterpret_cast<void*>(GetProcAddress(m, "PdhAddEnglishCounterW")));
		collect   = reinterpret_cast<long (*)(void*)>(reinterpret_cast<void*>(GetProcAddress(m, "PdhCollectQueryData")));
		value     = reinterpret_cast<long (*)(void*, unsigned long, unsigned long*, void*)>(reinterpret_cast<void*>(GetProcAddress(m, "PdhGetFormattedCounterValue")));
		if (open == nullptr || add == nullptr || collect == nullptr || value == nullptr) return;
		if (open(nullptr, 0, &query) != 0) return;
		if (add(query, L"\\Processor Information(_Total)\\% Processor Performance", 0, &counter) != 0) counter = nullptr;
		if (counter != nullptr) collect(query);
	}
	int Mhz() {
		if (counter == nullptr || max_mhz <= 0) return 0;
		struct {
			unsigned long status;
			double        value;
		} v {};
		collect(query);
		if (value(counter, 0x200 /* PDH_FMT_DOUBLE */, nullptr, &v) != 0) return 0;
		return static_cast<int>(std::lround(v.value * max_mhz / 100.0));
	}
};
#endif

// The log's last 64 KiB (Get-Fps) and, read on from where it stopped, the upscaler's lines (Update-ScaleLog).
struct LogState {
	std::string path;
	uint64_t    pos = 0;
	std::string kind, ratio, xess_ver, preset, fsr_ver;
	int         in_h = 0, out_h = 0, native = 0;
	bool        dlss_ready = false, force4k = false;
	bool        dlss5_on = false; // (11-11) "DLSS 5: on, evaluate OK" seen in the log
	int64_t     last_key = -1;
	std::chrono::steady_clock::time_point last_key_at {};
};

double Num(const std::string& s) {
	return std::strtod(s.c_str(), nullptr);
}

void Sample(LogState& L,
#if defined(_WIN32)
            Nvml& nv, CpuPerf& cpu,
#endif
            uint64_t& idle0, uint64_t& total0) {
	std::vector<Row> rows;
	const Settings&  s = g_rt.launched;
	// The launcher.ini values (StartMetricsWindow's): scalelabel, fglabel, antilagon, soulsmodo, dlsspreset.
	const std::string souls = s.souls_mode;
	const std::string modo  = souls == "ligero" ? "Modo Souls ligero" : souls == "ligero-plus" ? "Modo Souls ligero +" : souls == "equilibrado" ? "Modo Souls equilibrado"
	                        : souls == "ultra" ? "Modo Souls ultra" : souls == "extremo" ? "Modo Souls extremo"
	                        : souls == "low" ? "Modo Souls PC low profile" : souls == "sin-tirones" ? "Optimizado 16 GB / 8 GB" : "Personalizado";
	const bool  fixed = FixedFg(s); // (11-22) the optimizado mode too
	std::string fg_label = fixed ? modo : FrameGenLabel(s.framegen);
	std::string al       = "No";
	if (fixed) al = "Ultra";
	else if (s.framegen.rfind("dlss", 0) == 0) al = s.antilag_dlssg == "ultra" ? "Ultra (Reflex apagado)" : s.antilag_dlssg == "normal" ? "Normal (Reflex apagado)" : "Reflex (DLSS-G)";
	else if (s.framegen.rfind("amd", 0) == 0) {
		const std::string m = s.antilag_mode.empty() || s.antilag_mode == "auto" ? (g_rt.pc.amd ? "amd2" : "normal") : s.antilag_mode;
		al = m == "amd2" ? "Anti-Lag 2 (AMD)" : m == "ultra" ? "Ultra" : m == "no" ? "No" : "Normal";
	}
	std::string scale_label = kResModes[0].label;
	for (int i = 0; i < kResModeCount; i++)
		if (s.res == kResModes[i].id) scale_label = kResModes[i].label;
	const int dlss_preset = s.dlss_preset == "K" ? 1 : s.dlss_preset == "L" ? 2 : s.dlss_preset == "M" ? 3 : 0;
#if defined(_WIN32)
	if (nv.ok) {
		unsigned int u[2] = {}, t = 0, clk = 0;
		unsigned long long m[3] = {};
		nv.util(nv.dev, u);
		nv.temp(nv.dev, 0, &t);
		if (nv.clock != nullptr) nv.clock(nv.dev, 0 /* graphics */, &clk);
		nv.mem(nv.dev, m);
		// (11-12) Simular grafica de 8 GB: the ballast held to simulate it is not the game's (14.7 / 8 GB before).
		const uint64_t ballast = g_rt.launched.simulate_vram8 ? LocalPlatform::SimulationBallast() : 0;
		const double used = static_cast<double>((m[2] > ballast ? m[2] - ballast : 0) >> 20);
		// (11-12) Simular grafica de 8 GB: the overlay shows the simulated card's 8 GB.
		const double total = g_rt.launched.simulate_vram8 ? 8192.0 : static_cast<double>(m[0] >> 20);
		rows.push_back({"GPU", Fmt("%3u %%   %u C", u[0], t), kOrange});
		if (clk > 0) rows.push_back({"GPU reloj", Fmt("%4u MHz", clk), kOrange});
		rows.push_back({"VRAM", Fmt("%.1f / %.0f GB", used / 1024.0, total / 1024.0), UseColor(used, total)});
	} else {
		rows.push_back({"GPU", "(sin nvidia-smi)", kOrange});
		if (g_rt.pc.amd) rows.push_back({"GPU reloj", "n/a", kOrange});
	}
	{
		FILETIME fi, fk, fu;
		if (GetSystemTimes(&fi, &fk, &fu)) {
			const auto q = [](FILETIME f) { return (static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
			const uint64_t idle = q(fi), total = q(fk) + q(fu);
			const double   busy = total > total0 ? 100.0 * (1.0 - static_cast<double>(idle - idle0) / static_cast<double>(total - total0)) : 0.0;
			idle0 = idle, total0 = total;
			rows.push_back({"CPU", Fmt("%3d %%", static_cast<int>(std::lround(std::clamp(busy, 0.0, 100.0)))), kOrange});
		}
		if (const int mhz = cpu.Mhz(); mhz > 0) rows.push_back({"CPU reloj", Fmt("%4d MHz", mhz), kOrange});
		MEMORYSTATUSEX ms {};
		ms.dwLength = sizeof(ms);
		if (GlobalMemoryStatusEx(&ms)) {
			const double gb = 1024.0 * 1024.0 * 1024.0;
			// (11-12) Simular 16 GB RAM: as the RAM guard counts it (the emulator's working set + 4 GB for Windows), not the
			// bigger PC's whole use.
			PROCESS_MEMORY_COUNTERS pmc {};
			const double ws = GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) != 0 ? static_cast<double>(pmc.WorkingSetSize) / gb : 0.0;
			// (11-12) A 16 GB PC never shows more than 16 GB in use (the rest goes to its page file): capped.
			const double used = g_rt.launched.simulate_ram16 ? std::min(ws + 4.0, 16.0) : static_cast<double>(ms.ullTotalPhys - ms.ullAvailPhys) / gb;
			// (11-12) Simular 16 GB RAM: the overlay shows the simulated PC's 16 GB.
			const double total = g_rt.launched.simulate_ram16 ? 16.0 : static_cast<double>(ms.ullTotalPhys) / gb;
			rows.push_back({"RAM", Fmt("%.1f / %.0f GB", used, total), UseColor(used, total)});
			const double climit = static_cast<double>(ms.ullTotalPageFile) / gb, cused = climit - static_cast<double>(ms.ullAvailPageFile) / gb;
			if (climit > 0) rows.push_back({"Commit", Fmt("%.1f / %.0f GB", cused, climit), cused / climit > 0.90 ? kRed : UseColor(cused, climit)});
		}
	}
#endif
	// Get-Fps: the log's last 64 KiB.
	const char* log_env = std::getenv("KYTY_RUN_LOG");
	std::string text;
	uint64_t    len = 0, take = 0;
	if (log_env != nullptr && *log_env != '\0') {
		std::ifstream in(std::filesystem::u8path(log_env), std::ios::binary);
		in.seekg(0, std::ios::end);
		len  = static_cast<uint64_t>(in.tellg());
		take = std::min<uint64_t>(len, 65536);
		text.resize(take);
		in.seekg(static_cast<std::streamoff>(len - take));
		in.read(text.data(), static_cast<std::streamsize>(take));
		// Update-ScaleLog: on from the last position, up to 8 MiB at a time.
		if (L.path != log_env) L = LogState {}, L.path = log_env;
		if (len < L.pos) L.pos = 0;
		const uint64_t more = std::min<uint64_t>(len - L.pos, 8u << 20);
		std::string    part(more, '\0');
		in.clear();
		in.seekg(static_cast<std::streamoff>(L.pos));
		in.read(part.data(), static_cast<std::streamsize>(more));
		L.pos += more;
		std::smatch mm;
		const auto last = [&](const std::string& t, const std::regex& re, std::smatch& out) {
			bool found = false;
			for (std::sregex_iterator it(t.begin(), t.end(), re), end; it != end; ++it) out = *it, found = true;
			return found;
		};
		static const std::regex re_ratio(R"(KYTY_UPSCALE_RATIO=([0-9.]+) on a (\d{3,4}) high display: (\d{3,4}))");
		static const std::regex re_disp(R"(on a (\d{3,4}) high display)");
		static const std::regex re_preset(R"(DLSS[^\r\n]*?(?:preset|model[o]?)[ =:]+([A-Z])\b)");
		static const std::regex re_sr(R"((DLSS|XeSS|FSR 3\.1|FSR 4\.\d)[^\r\n]*?(\d{3,4})x(\d{3,4}) -> (\d{3,4})x(\d{3,4}))");
		static const std::regex re_native(R"(performance mode renders (\d{3,4})x(\d{3,4}))");
		static const std::regex re_xess(R"(XeSS: libxess\.dll ([0-9.]+) loaded)");
		if (last(part, re_ratio, mm)) L.ratio = mm[1], L.out_h = std::atoi(mm[2].str().c_str()), L.in_h = std::atoi(mm[3].str().c_str());
		else if (last(part, re_disp, mm)) L.out_h = std::atoi(mm[1].str().c_str());
		if (part.find("starts in its cinematic mode (KYTY_DS_FORCE_MODE") != std::string::npos) {
			L.force4k = true;
			if (L.out_h == 0) L.out_h = 2160;
		}
		if (std::regex_search(part, std::regex(R"(DLSS super resolution[^\r\n]*ready)"))) L.dlss_ready = true, L.kind = "DLSS";
		if (part.find("DLSS 5: on, evaluate OK") != std::string::npos) L.dlss5_on = true; // (11-11)
		if (std::regex_search(part, mm, re_xess)) {
			L.xess_ver = mm[1];
			if (L.kind.empty()) L.kind = "XeSS";
		}
		if (last(part, re_preset, mm)) L.preset = mm[1];
		{
			std::vector<std::smatch> all;
			for (std::sregex_iterator it(part.begin(), part.end(), re_sr), end; it != end; ++it) all.push_back(*it);
			for (auto it = all.rbegin(); it != all.rend(); ++it) {
				const int h = std::atoi((*it)[3].str().c_str());
				if (h >= 600) {
					L.kind = (*it)[1].str() == "FSR 3.1" ? "FSR" : (*it)[1].str();
					// (des-upstream-1106) "FSR 4.1": AMD's SDK 2.4 runtime really created the context (Fsr3::UpscalerName).
					if (L.kind.rfind("FSR 4.", 0) == 0) L.fsr_ver = L.kind, L.kind = "FSR";
					else if (L.kind == "FSR") L.fsr_ver.clear();
					L.in_h = h;
					if (L.out_h == 0) L.out_h = std::atoi((*it)[5].str().c_str());
					break;
				}
			}
		}
		if (last(part, re_native, mm)) L.native = std::atoi(mm[2].str().c_str());
	}
	struct {
		double mean = 0, max = 0, fps = 0, fg_shown = 0, fg_real = 0, lat_mean = 0, lat_p95 = 0;
		int    fg_mult = 0;
		int64_t key = -1;
		bool   any = false;
	} f;
	if (!text.empty()) {
		static const std::regex re_fps(R"(\[fps\] \d+ frames: mean ([0-9.]+) ms, max ([0-9.]+))");
		static const std::regex re_f5(R"(Frames \(5 s\): ([0-9.]+) fps)");
		static const std::regex re_rs(R"(\[[a-z0-9]+\][^\r\n]*?real ([0-9.]+) fps, shown ([0-9.]+) fps)");
		static const std::regex re_x(R"(\[(dlssg|fsr3-mfg)\][^\r\n]*?x(\d))");
		static const std::regex re_lat(R"(\[lat\] input->present mean ([0-9.]+) ms, p95 ([0-9.]+), max ([0-9.]+); \+scanout ([0-9.]+); fg \+([0-9.]+) \(n (\d+)\)(?:; total ([0-9.]+), p95 ([0-9.]+))?)");
		std::vector<std::smatch> fm;
		for (std::sregex_iterator it(text.begin(), text.end(), re_fps), end; it != end; ++it) fm.push_back(*it);
		int64_t prev_idx = -1;
		if (!fm.empty()) {
			f.any = true;
			prev_idx = fm.size() > 1 ? fm[fm.size() - 2].position(0) : -1;
			f.mean = Num(fm.back()[1]), f.max = Num(fm.back()[2]);
			if (f.mean > 0) f.fps = std::round(1000.0 / f.mean);
			f.key = static_cast<int64_t>(len - take) + fm.back().position(0);
		} else {
			std::smatch m5;
			for (std::sregex_iterator it(text.begin(), text.end(), re_f5), end; it != end; ++it) m5 = *it;
			if (!m5.empty()) f.any = true, f.fps = std::round(Num(m5[1]));
		}
		std::smatch mx, mr, ml;
		bool hx = false, hr = false, hl = false;
		for (std::sregex_iterator it(text.begin(), text.end(), re_x), end; it != end; ++it) mx = *it, hx = true;
		for (std::sregex_iterator it(text.begin(), text.end(), re_rs), end; it != end; ++it) mr = *it, hr = true;
		for (std::sregex_iterator it(text.begin(), text.end(), re_lat), end; it != end; ++it) ml = *it, hl = true;
		if (hx) f.fg_mult = std::atoi(mx[2].str().c_str());
		if (hr && mr.position(0) > prev_idx && mr[2].str() != mr[1].str()) f.fg_shown = std::round(Num(mr[2])), f.fg_real = std::round(Num(mr[1]));
		if (hl) {
			const double add = Num(ml[4]) + Num(ml[5]);
			f.lat_mean = ml[7].matched ? Num(ml[7]) : Num(ml[1]) + add;
			f.lat_p95  = ml[8].matched ? Num(ml[8]) : Num(ml[2]) + add;
		}
	}
	const bool fg_on = !fg_label.empty() && fg_label != "Sin frame generation" && fg_label != "No frame generation";
	double     shown = 0;
	std::vector<double> ft, ftm;
	{
		std::scoped_lock lock(g_mutex);
		ft = g_ft, ftm = g_ft_max;
	}
	if (f.any) {
		shown = f.fg_shown > 0 && fg_on ? f.fg_shown : f.fps;
		if (f.mean > 0) {
			rows.push_back({"Frame", Fmt("%.1f ms", f.mean), kOrange});
			if (f.key != L.last_key) {
				L.last_key_at = std::chrono::steady_clock::now();
				L.last_key    = f.key;
				ft.push_back(f.mean), ftm.push_back(f.max);
				while (ft.size() > 60) ft.erase(ft.begin()), ftm.erase(ftm.begin());
			}
		}
	}
	if (f.lat_mean > 0) {
		const ImU32 lc = f.lat_mean < 40 ? kGreen : f.lat_mean < 60 ? kGold : f.lat_mean < 80 ? kOrange : kRed;
		rows.push_back({"Latencia", Fmt("%.0f ms (p95 %.0f)", f.lat_mean, f.lat_p95), lc});
	}
	// Get-Scale.
	{
		const char*       dot   = " \xC2\xB7 ";
		const char*       arrow = "\xE2\x86\x92";
		const std::string lk    = scale_label.substr(0, scale_label.find(' '));
		std::string       kind  = L.kind;
		if (kind.empty() && (lk == "DLSS" || lk == "XeSS" || lk == "FSR")) kind = lk;
		const int out_h = L.out_h != 0 ? L.out_h : (g_rt.pc.screen_h > 0 ? g_rt.pc.screen_h : 1440);
		std::string rk, mode, res, model;
		const auto mode_name = [](const std::string& label) {
			const size_t sp = label.find(' ');
			if (sp == std::string::npos) return std::string();
			std::string m = label.substr(sp + 1);
			const std::pair<const char*, const char*> reps[] = {{"U.cal.+", "Ultra calidad+"}, {"U.calidad", "Ultra calidad"}, {"Equilib.", "Equilibrado"},
			                                                    {"U.rendim.", "Ultra rendimiento"}, {"Rendim.", "Rendimiento"}};
			for (const auto& [a, b]: reps)
				if (const size_t at = m.find(a); at != std::string::npos) m.replace(at, std::strlen(a), b);
			return m;
		};
		if (!kind.empty()) {
			rk = kind;
			if (lk == kind) mode = mode_name(scale_label);
			int in_h = L.in_h;
			if (in_h == 0 && lk == kind && !L.ratio.empty()) in_h = static_cast<int>(std::lround(out_h / Num(L.ratio)));
			std::string ratio = !L.ratio.empty() ? " (x" + L.ratio + ")" : in_h != 0 ? Fmt(" (x%.3g)", out_h / static_cast<double>(in_h)) : std::string();
			res = in_h != 0 ? Fmt("%dp %s %dp%s", in_h, arrow, out_h, ratio.c_str()) : Fmt("%s %dp", arrow, out_h);
			if (kind == "DLSS") {
				std::string pk = L.preset;
				if (pk.empty()) pk = std::array<const char*, 4> {"", "K", "L", "M"}[dlss_preset];
				model = pk == "K" ? "K (Transformer, DLSS 4)" : pk == "L" ? "L (Transformer, DLSS 4.5)" : pk == "M" ? "M (Transformer, DLSS 4.5)" : pk.empty() ? "auto (NVIDIA)" : pk;
				if (!L.dlss_ready && L.in_h == 0) model += " ...";
			} else if (kind == "XeSS") model = L.xess_ver == "2.0.2" ? "XeSS 3.0.2" : !L.xess_ver.empty() ? "XeSS " + L.xess_ver : "XeSS";
			else if (kind == "FSR") model = L.fsr_ver.empty() ? "FSR 3.1" : L.fsr_ver + " (AMD SDK 2.4)";
		} else if (L.force4k) rk = "Nativo", res = "2160p (4K)";
		else if (scale_label.find("juego") != std::string::npos) rk = "Juego", res = Fmt("%sp %s %dp", scale_label.substr(0, scale_label.find('p')).c_str(), arrow, out_h), model = "reescalado del juego";
		else if (L.native != 0) rk = "Nativo", res = Fmt("%dp %s %dp", L.native, arrow, out_h);
		else rk = scale_label;
		if (scale_label.rfind("FSR 4", 0) == 0 && rk == "FSR") rk = "FSR 4", mode = scale_label.substr(6), model = "FSR 4 (IA, modelo v07)";
		// (11-11) DLSS 5 in the Modelo row: the level it was launched with; "activo" once the log says it evaluates.
		if (!g_rt.launched.dlss5.empty() && g_rt.launched.dlss5 != "0") {
			std::string lvl = g_rt.launched.dlss5;
			for (int i = 0; i < kDlss5LevelCount; i++)
				if (g_rt.launched.dlss5 == kDlss5Levels[i].id) lvl = kDlss5Levels[i].label;
			const std::string d5 = "DLSS 5 " + lvl + (L.dlss5_on ? " activo" : " ...");
			model = model.empty() ? d5 : model + " + " + d5;
			if (rk.empty()) rk = "Nativo";
		}
		if (!rk.empty()) {
			const ImU32 col = rk == "DLSS" ? kGreen : rk == "FSR" || rk == "FSR 4" ? kRed : rk == "XeSS" ? kBlue : kOrange;
			rows.push_back({"Escala", mode.empty() ? rk : rk + dot + mode, col});
			if (!res.empty()) rows.push_back({"Resol.", res, col});
			if (!model.empty()) rows.push_back({"Modelo", model, col});
		}
	}
	const bool fg_nv = fg_label.find("DLSS") != std::string::npos || fg_label.find("Nvidia") != std::string::npos;
	if (fg_on) {
		const ImU32 fg_col = fg_nv ? kGreen : kRed;
		if (fg_label.rfind("MODO ", 0) == 0 || fg_label.rfind("MODE ", 0) == 0) rows.push_back({"", fg_label, fg_col});
		else {
			rows.push_back({"FG", fg_label + (fg_nv ? " (modelo C)" : ""), fg_col});
			if (f.fg_mult > 0) rows.push_back({"FG ahora", Fmt("x%d", f.fg_mult), fg_col});
		}
	}
	rows.push_back({"Anti-Lag", al, al == "No" ? kKhaki : kGreen});
	rows.push_back({"", modo, kKhaki});
	if (f.key >= 0 && L.last_key_at != std::chrono::steady_clock::time_point {} &&
	    std::chrono::steady_clock::now() - L.last_key_at > std::chrono::seconds(3))
		shown = 0;
	std::scoped_lock lock(g_mutex);
	g_rows     = std::move(rows);
	// (ds-pc-port42) The shown rate big, and with frame generation the real one beside it (the same [fsr3] / [dlssg] line).
	g_fps_text = shown > 0 ? (fg_on && f.fg_real > 0 && f.fg_shown > 0 ? Fmt("FPS %.0f  (real %.0f)", shown, f.fg_real) : Fmt("FPS %.0f", shown)) : std::string();
	g_fps_color = fg_on ? (fg_nv ? kGreen : kRed) : kWhite;
	g_ft = std::move(ft), g_ft_max = std::move(ftm);
}

void Start() {
	if (g_started.exchange(true)) return;
	std::thread([] {
		LogState L;
#if defined(_WIN32)
		Nvml    nv;
		CpuPerf cpu;
		nv.Init();
		cpu.Init();
#endif
		uint64_t idle0 = 0, total0 = 0;
		for (;;) {
			if (g_shown.load(std::memory_order_relaxed)) {
				Sample(L,
#if defined(_WIN32)
				       nv, cpu,
#endif
				       idle0, total0);
				// (ds-pc-port45) The revision moves again, so the game's own frames redraw the panel with the new values (port43
				// dropped this Bump: the panel stopped updating). An idle vblank still does not re-present for it: the
				// presenter's refresh_revision counts the port's revision only while its menu is open (systemOverlay.cpp).
				Bump();
			}
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}).detach();
}

bool Shown() {
	return g_shown.load(std::memory_order_relaxed);
}

void Toggle() {
	g_shown = !g_shown.load();
	if (g_shown.load()) Start();
	Bump();
}

// Top-centre, 10 px x S from the top.
void Draw(float width, float height) {
	if (!g_shown.load(std::memory_order_relaxed)) return;
	const float S = std::clamp(height / 1080.0f, 1.0f, 2.5f);
	ImFont*     f = FindFont("consolab");
	if (f == nullptr) f = FindFont("segoeui");
	if (f == nullptr) f = ImGui::GetFont();
	std::vector<Row>    rows;
	std::string         fps;
	ImU32               fps_col;
	std::vector<double> ft, ftm;
	{
		std::scoped_lock lock(g_mutex);
		rows = g_rows, fps = g_fps_text, fps_col = g_fps_color, ft = g_ft, ftm = g_ft_max;
	}
	if (rows.empty()) rows.push_back({"", "Cargando...", kOrange});
	const float row_px = std::round(15 * S), title_px = std::round(16 * S), fps_px = std::round(34 * S), small_px = std::round(10 * S);
	const float pad = std::round(8 * S), row_h = std::ceil(row_px * 1.12f), title_h = std::ceil(title_px * 1.4f), fps_h = fps.empty() ? 0.0f : std::ceil(fps_px * 1.1f);
	float name_w = std::round(90 * S), val_w = std::round(230 * S), wide_w = 0;
	for (const auto& r: rows) {
		const float w = f->CalcTextSizeA(row_px, FLT_MAX, 0, r.value.c_str()).x;
		if (!r.name.empty()) {
			name_w = std::max(name_w, f->CalcTextSizeA(row_px, FLT_MAX, 0, r.name.c_str()).x + std::round(10 * S));
			val_w  = std::max(val_w, w);
		} else wide_w = std::max(wide_w, w);
	}
	const float W = std::min(std::max(name_w + val_w, wide_w) + 2 * pad, width * 0.4f);
	const float graph_h = std::round(90 * S);
	const float x0 = std::floor((width - W) * 0.5f), y0 = std::round(10 * S);
	auto*       dl = ImGui::GetForegroundDrawList();
	const char* title = "DEMON'S SOULS REMAKE";
	const float tw = f->CalcTextSizeA(title_px, FLT_MAX, 0, title).x;
	dl->AddText(f, title_px, {std::floor(x0 + (W - tw) * 0.5f), y0 + pad}, kWhite, title);
	float y = y0 + pad + title_h;
	const ImVec4 clip {x0, y0, x0 + W, y0 + 4000};
	for (const auto& r: rows) {
		if (!r.name.empty()) {
			dl->AddText(f, row_px, {x0 + pad, y}, kName, r.name.c_str(), nullptr, 0, &clip);
			dl->AddText(f, row_px, {x0 + pad + name_w, y}, r.color, r.value.c_str(), nullptr, 0, &clip);
		} else dl->AddText(f, row_px, {x0 + pad, y}, r.color, r.value.c_str(), nullptr, 0, &clip);
		y += row_h;
	}
	if (!fps.empty()) {
		dl->AddText(f, fps_px, {x0 + pad, y}, fps_col, fps.c_str());
		y += fps_h;
	}
	y += pad;
	// The frametime graph: 40 ms at the top, 8.3 / 16.7 / 33.3 lines, the peaks (orange) and the mean (green).
	const float gy = y, gw = W, top = 40.0f;
	for (float ms: {8.3f, 16.7f, 33.3f}) {
		const float ly = gy + graph_h - ms / top * graph_h;
		dl->AddLine({x0, ly}, {x0 + gw, ly}, IM_COL32(60, 60, 60, 255));
	}
	dl->AddText(f, small_px, {x0, gy}, IM_COL32(128, 128, 128, 255), "frametime (ms)  40   verde media, naranja picos");
	dl->AddText(f, small_px, {x0, gy + graph_h - 16.7f / top * graph_h - small_px}, IM_COL32(128, 128, 128, 255), "16.7");
	const auto line = [&](const std::vector<double>& list, ImU32 col) {
		if (list.size() < 2) return;
		const float step = gw / 59.0f, xs = x0 + gw - (list.size() - 1) * step;
		for (size_t i = 1; i < list.size(); i++) {
			const float y1 = gy + graph_h - std::min(static_cast<float>(list[i - 1]), top) / top * graph_h;
			const float y2 = gy + graph_h - std::min(static_cast<float>(list[i]), top) / top * graph_h;
			dl->AddLine({xs + (i - 1) * step, y1}, {xs + i * step, y2}, col, 2.0f * S);
		}
	};
	line(ftm, IM_COL32(255, 150, 40, 255));
	line(ft, IM_COL32(0, 255, 0, 255));
}
} // namespace MetricsPanel

// ---- (ds-pc-port37) F2 > Mapear mando and F2 > Teclado y raton: pages of the in-game menu (as Ver trofeos). ----
namespace F1Pages {
ImTextureData* g_mouse = nullptr;
ImGuiContext*  g_mouse_ctx = nullptr;
int            g_mouse_w = 0, g_mouse_h = 0;
bool           g_mouse_tried = false;
void LoadMouse() {
	if (g_mouse != nullptr && g_mouse_ctx != ImGui::GetCurrentContext()) g_mouse = nullptr, g_mouse_tried = false;
	if (g_mouse_tried) return;
	g_mouse_tried = true;
	std::ifstream in(g_rt.exe_path.parent_path() / L"raton.png", std::ios::binary);
	const std::string bytes {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
	int   channels = 0;
	auto* px = bytes.empty() ? nullptr
	                         : stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()), &g_mouse_w,
	                                                 &g_mouse_h, &channels, 4);
	if (px == nullptr) return;
	g_mouse = IM_NEW(ImTextureData)();
	g_mouse->Create(ImTextureFormat_RGBA32, g_mouse_w, g_mouse_h);
	std::memcpy(g_mouse->GetPixels(), px, static_cast<size_t>(g_mouse_w) * g_mouse_h * 4);
	g_mouse->UseColors = true;
	stbi_image_free(px);
	ImGui::RegisterUserTexture(g_mouse);
	g_mouse_ctx = ImGui::GetCurrentContext();
}

constexpr const char* kKbmDefault =
    "LeftStickUp=W;LeftStickDown=S;LeftStickLeft=A;LeftStickRight=D;Walk=Left Ctrl;R1=Mouse:Left;L1=Mouse:Right;R2=Shift+Mouse:Left;"
    "L2=Shift+Mouse:Right;R3=Mouse:Middle;R3=Mouse:WheelUp;R3=Mouse:WheelDown;Circle=Space;Square=E;Cross=F;Triangle=G;Left=Q;Right=R;"
    "Up=1;Down=2;Left=3;Right=4;Options=Tab;TouchPad=Backspace";
const char* ControlName(const std::string& c) {
	static const std::pair<const char*, const char*> names[] = {
	    {"LeftStickUp", "Mover adelante"}, {"LeftStickDown", "Mover atrás"}, {"LeftStickLeft", "Mover a la izquierda"},
	    {"LeftStickRight", "Mover a la derecha"}, {"Walk", "Andar (media velocidad)"}, {"R1", "R1 (ataque)"}, {"L1", "L1 (bloquear)"},
	    {"R2", "R2 (ataque fuerte)"}, {"L2", "L2 (parry / fuerte izq.)"}, {"R3", "R3 (fijar objetivo)"}, {"Circle", "Círculo (rodar / atrás)"},
	    {"Square", "Cuadrado (usar objeto)"}, {"Cross", "Cruz (interactuar)"}, {"Triangle", "Triángulo (dos manos)"},
	    {"Left", "Izquierda (cambiar mano izq.)"}, {"Right", "Derecha (cambiar mano der.)"}, {"Up", "Arriba (cambiar magia)"},
	    {"Down", "Abajo (cambiar objeto)"}, {"Options", "Options (menú)"}, {"TouchPad", "Panel táctil"}};
	for (const auto& [k, v]: names)
		if (c == k) return I18n::T(v);
	return c.c_str();
}
std::string KeyText(const std::string& k) {
	if (k.empty() || k == "None") return I18n::T("(sin asignar)");
	std::string pre, key = k;
	if (key.rfind("Shift+", 0) == 0) pre = std::string(I18n::T("Mayús + ")), key = key.substr(6);
	static const std::pair<const char*, const char*> names[] = {
	    {"Mouse:Left", "Clic izquierdo"}, {"Mouse:Right", "Clic derecho"}, {"Mouse:Middle", "Clic central"}, {"Mouse:X1", "Botón lateral 1"},
	    {"Mouse:X2", "Botón lateral 2"}, {"Mouse:WheelUp", "Rueda arriba"}, {"Mouse:WheelDown", "Rueda abajo"}, {"Space", "Espacio"},
	    {"Left Ctrl", "Ctrl izq."}, {"Left Shift", "Mayús izq."}, {"Left Alt", "Alt izq."}, {"Backspace", "Retroceso"}, {"Return", "Intro"}};
	for (const auto& [a, b]: names)
		if (key == a) return pre + I18n::T(b);
	return pre + key;
}

std::vector<std::pair<std::string, std::string>> g_rows; // (control, key) being edited
int  g_capture_row = -1;
bool g_loaded      = false;
std::string g_note;

void SaveNow() {
	Settings on_disk;
	if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_saved;
	on_disk.pad_map    = g_edit.pad_map;
	on_disk.kbm        = g_edit.kbm;
	on_disk.kbm_sens   = g_edit.kbm_sens;
	on_disk.kbm_antidz = g_edit.kbm_antidz;
	on_disk.kbm_camera = g_edit.kbm_camera;
	on_disk.kbm_warp   = g_edit.kbm_warp;
	on_disk.kbm_map    = g_edit.kbm_map;
	g_note = SaveSettings(g_rt.ini_path, on_disk) ? I18n::T("Guardado en settings.ini. Se aplica al reiniciar el juego.") : I18n::T("No se pudo guardar settings.ini");
	g_saved.pad_map = on_disk.pad_map, g_saved.kbm = on_disk.kbm, g_saved.kbm_sens = on_disk.kbm_sens, g_saved.kbm_antidz = on_disk.kbm_antidz;
	g_saved.kbm_camera = on_disk.kbm_camera, g_saved.kbm_warp = on_disk.kbm_warp, g_saved.kbm_map = on_disk.kbm_map;
}

// Both pages: full screen; false when the page closes.
// (ds-pc-port38) Their text was ImGui's default 13 px bitmap font (no font pushed: tiny, blurry once scaled): now the
// menu's Segoe UI at 26 px x the menu's unit (height / 1080) for the body, 46 px for the title, rasterized at exactly
// that size (ImGui 1.92 dynamic fonts: never an upscaled atlas), near-white text on a darker plate, larger controls,
// and the page slides in from the right with a fade.
int g_style_vars = 0, g_style_colors = 0;
float g_page_t = 1.0f; // (trucos7) the page's entry progress (Begin): PageZoom scales the page from 92 % with it
// (trucos7) The page's entry: everything the page window drew grows from 92 % about the screen centre while it fades in
// (vertex scale: glyphs stay baked at their size; at 1.0 the vertices are untouched: crisp).
void PageZoom() {
	if (g_page_t >= 1.0f) return;
	const ImVec2 wp = ImGui::GetWindowPos(), ws = ImGui::GetWindowSize();
	Anim::ScaleVertices(ImGui::GetWindowDrawList(), 0, {wp.x + ws.x * 0.5f, wp.y + ws.y * 0.5f}, 0.92f + 0.08f * g_page_t);
}
bool Begin(const char* title, float width, float height, float unit) {
	if (Anim::g_page_seen != g_f1_page.load()) {
		Anim::g_page_seen = g_f1_page.load();
		Anim::g_page_at   = Anim::Now();
	}
	const float t = Anim::EaseOut(Anim::Progress(Anim::g_page_at.load(), Anim::kPage));
	g_page_t      = t;
	if (t < 1.0f) Bump();
	ImGui::GetBackgroundDrawList()->AddRectFilled({0, 0}, {width, height}, IM_COL32(0, 0, 0, static_cast<int>(170 * t)));
	ImFont* font = FindFont("segoeui");
	ImGui::PushFont(font, std::round(26.0f * unit));
	const auto var = [&](ImGuiStyleVar v, ImVec2 x) { ImGui::PushStyleVar(v, x), g_style_vars++; };
	const auto var1 = [&](ImGuiStyleVar v, float x) { ImGui::PushStyleVar(v, x), g_style_vars++; };
	const auto col = [&](ImGuiCol c, ImVec4 x) { ImGui::PushStyleColor(c, x), g_style_colors++; };
	var1(ImGuiStyleVar_Alpha, std::max(0.01f, t));
	var(ImGuiStyleVar_WindowPadding, {std::round(56 * unit), std::round(34 * unit)});
	var(ImGuiStyleVar_ItemSpacing, {std::round(16 * unit), std::round(14 * unit)});
	var(ImGuiStyleVar_FramePadding, {std::round(16 * unit), std::round(9 * unit)});
	var(ImGuiStyleVar_CellPadding, {std::round(14 * unit), std::round(10 * unit)});
	var1(ImGuiStyleVar_FrameRounding, 10 * unit);
	var1(ImGuiStyleVar_GrabRounding, 8 * unit);
	var1(ImGuiStyleVar_GrabMinSize, 24 * unit);
	var1(ImGuiStyleVar_ScrollbarSize, 20 * unit);
	var1(ImGuiStyleVar_FrameBorderSize, std::max(1.0f, unit));
	col(ImGuiCol_Text, ImVec4 {0.97f, 0.96f, 0.93f, 1.0f});
	col(ImGuiCol_TextDisabled, ImVec4 {0.74f, 0.72f, 0.68f, 1.0f});
	col(ImGuiCol_WindowBg, ImVec4 {0, 0, 0, 0}); // (trucos11: the backdrop is drawn under the page)
	col(ImGuiCol_Border, ImVec4 {0.87f, 0.77f, 0.55f, 0.40f});
	col(ImGuiCol_FrameBg, ImVec4 {0.11f, 0.11f, 0.13f, 1.0f});
	col(ImGuiCol_FrameBgHovered, ImVec4 {0.17f, 0.16f, 0.15f, 1.0f});
	col(ImGuiCol_FrameBgActive, ImVec4 {0.22f, 0.20f, 0.17f, 1.0f});
	col(ImGuiCol_Button, ImVec4 {0.13f, 0.125f, 0.12f, 1.0f});
	col(ImGuiCol_ButtonHovered, ImVec4 {0.15f, 0.14f, 0.13f, 1.0f}); // (the transition is Anim::Highlight's)
	col(ImGuiCol_ButtonActive, ImVec4 {0.30f, 0.26f, 0.18f, 1.0f});
	col(ImGuiCol_CheckMark, kGold);
	col(ImGuiCol_SliderGrab, kGold);
	col(ImGuiCol_SliderGrabActive, ImVec4 {1.0f, 0.88f, 0.62f, 1.0f});
	col(ImGuiCol_Header, ImVec4 {0.30f, 0.26f, 0.18f, 0.85f});
	col(ImGuiCol_HeaderHovered, ImVec4 {0.42f, 0.35f, 0.22f, 0.85f});
	col(ImGuiCol_TableRowBg, ImVec4 {0.0f, 0.0f, 0.0f, 0.0f});
	col(ImGuiCol_TableRowBgAlt, ImVec4 {1.0f, 1.0f, 1.0f, 0.045f});
	col(ImGuiCol_Separator, ImVec4 {0.87f, 0.77f, 0.55f, 0.35f});
	col(ImGuiCol_NavCursor, kGold);
	// (Full screen, as the trophy page; it slides in 64 px from the right.)
	ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always); // (trucos7: zoom + fade, PageZoom; was a 64 px slide)
	ImGui::SetNextWindowSize({width, height}, ImGuiCond_Always);
	ImGui::Begin(title, nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
	{
		// (trucos4) The page in the Demon's Souls look: vignette, an aged-gold double frame, the title engraved in Cinzel
		// (whole-pixel size and position: crisp) over an ornamental divider.
		ImDrawList*  dl = ImGui::GetWindowDrawList();
		const ImVec2 wp = ImGui::GetWindowPos(), ws = ImGui::GetWindowSize();
		// (trucos11) The page: the stone-and-fog backdrop (ds-style.h) full screen, the title in large tracked bone-white
		// capitals, a rhombus rule; the page fades in with a slight zoom (PageZoom), cross-fading over the menu.
		DsStyle::Backdrop(ImGui::GetBackgroundDrawList(), {0, 0}, {width, height}, t, Souls::Time());
		DsStyle::FogLayers(ImGui::GetBackgroundDrawList(), {0, 0}, {width, height}, t, Souls::Time());
		DsStyle::Embers(ImGui::GetBackgroundDrawList(), {0, 0}, {width, height}, t * 0.6f, Souls::Time(), 40);
		AshFrame(ImGui::GetBackgroundDrawList(), width, height, t); // (trucos13)
		ImFont*           tf  = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
		const float       tsz = std::round(58.0f * unit);
		const std::string tt  = DsStyle::Upper(I18n::T(title));
		const ImVec2      at  = ImGui::GetCursorScreenPos();
		// (trucos12) The title letter by letter with a brief gold flash; a light sweeps across the page as it opens.
		const float since = static_cast<float>(Anim::Now() - Anim::g_page_at.load());
		const float tw    = DsStyle::IlluminatedTitle(dl, tf, tsz, {std::round(at.x + tsz * 0.1f), std::round(at.y)}, tt.c_str(), t, since, Souls::Time()); // (trucos13)
		if (since < 1.2f) Bump();
		DsStyle::LightSweep(ImGui::GetForegroundDrawList(), {0, 0}, {width, height}, since / 0.8f, 1.0f);
		DsStyle::Filigree(dl, {wp.x + std::round(60 * unit), wp.y + std::round(40 * unit)}, std::round(150 * unit), 1, 1, t, Souls::Time(), unit);
		DsStyle::Filigree(dl, {wp.x + ws.x - std::round(60 * unit), wp.y + std::round(40 * unit)}, std::round(150 * unit), -1, 1, t, Souls::Time(), unit);
		ImGui::Dummy({tw, tsz + 10 * unit});
		const float dy = std::round(ImGui::GetCursorScreenPos().y);
		DsStyle::DiamondRule(dl, at.x, at.x + std::min(ws.x - 2 * (at.x - wp.x), std::max(900 * unit, tw)), dy, t, unit);
		ImGui::Dummy({1, 18 * unit});
	}
	return true;
}
bool Footer(float unit) {
	ImGui::Separator();
	bool close = false;
	const ImVec2 bs {std::round(240 * unit), std::round(54 * unit)};
	if (Anim::Button(I18n::T("Guardar"), bs, unit)) SaveNow();
	ImGui::SameLine();
	if (g_saved.kbm != g_rt.launched.kbm || g_saved.kbm_map != g_rt.launched.kbm_map || // (remap-f2: the pad map is live)
	    g_saved.kbm_sens != g_rt.launched.kbm_sens || g_saved.kbm_antidz != g_rt.launched.kbm_antidz ||
	    g_saved.kbm_camera != g_rt.launched.kbm_camera || g_saved.kbm_warp != g_rt.launched.kbm_warp) {
		if (Anim::Button(I18n::T("Reiniciar ahora"), bs, unit)) RequestRestart();
		ImGui::SameLine();
	}
	if (Anim::Button(I18n::T("Volver"), bs, unit)) close = true;
	ImGui::SameLine();
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (bs.y - ImGui::GetTextLineHeight()) * 0.5f);
	ImGui::TextDisabled("%s", I18n::T("L1 / L2: desplazarse   Círculo / Esc: volver"));
	if (!g_note.empty()) {
		ImGui::SameLine();
		ImGui::TextColored(kDim, "%s", g_note.c_str());
	}
	PageZoom(); // (trucos7)
	ImGui::End();
	ImGui::PopStyleColor(g_style_colors);
	ImGui::PopStyleVar(g_style_vars);
	g_style_colors = g_style_vars = 0;
	ImGui::PopFont();
	return !close;
}

// (remap2) The controller drawn full screen, its buttons marked on it (PadMap::Page): no list, nothing to scroll.
bool DrawPad(float width, float height, float unit) {
	g_page_scroll = 0;
	g_popup       = false; // (no list of its own: Circle / B goes back)
	return PadMap::Page(width, height, unit);
}

bool DrawKbm(float width, float height, float unit) {
	if (g_pages_reset.exchange(false)) {
		g_capture_row = -1;
		g_loaded      = false;
		std::scoped_lock lock(g_capture_mutex);
		g_captured.clear();
	}
	if (!g_loaded) {
		g_rows.clear();
		const std::string map = g_edit.kbm_map.empty() ? std::string(kKbmDefault) : g_edit.kbm_map;
		size_t at = 0;
		while (at <= map.size()) {
			const size_t end  = std::min(map.find(';', at), map.size());
			const auto   item = map.substr(at, end - at);
			at                = end + 1;
			const size_t eq   = item.find('=');
			if (eq != std::string::npos) g_rows.emplace_back(item.substr(0, eq), item.substr(eq + 1));
		}
		g_loaded = true;
	}
	LoadMouse();
	Begin("Teclado y ratón", width, height, unit);
	ImGui::BeginChild("##kbmpage", {0, -std::round(84 * unit)}, ImGuiChildFlags_NavFlattened); Anim::SmoothScroll(); // (ds-pc-port41) the D-pad / stick move row by row inside the list and the page follows the focus
	if (const int d = g_page_scroll.exchange(0); d != 0) ImGui::SetScrollY(ImGui::GetScrollY() + d * ImGui::GetWindowHeight() * 0.6f);
	ImGui::Checkbox(I18n::T("Jugar con teclado y ratón"), &g_edit.kbm);
	Anim::Highlight(unit);
	// (Wrapped left of the mouse picture.)
	ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - (g_mouse != nullptr ? 280 * unit : 0.0f));
	ImGui::TextWrapped("%s", I18n::T("WASD para moverse, el ratón mueve la cámara, clic izquierdo R1, clic derecho L1, rueda o clic central fija objetivo. F7 suelta o captura el ratón. Si juegas con mando, déjalo desactivado."));
	ImGui::PopTextWrapPos();
	ImGui::BeginDisabled(!g_edit.kbm);
	ImGui::SetNextItemWidth(520 * unit);
	ImGui::SliderFloat(I18n::T("Sensibilidad del ratón"), &g_edit.kbm_sens, 0.05f, 4.0f, "%.2f");
	ImGui::SetNextItemWidth(520 * unit);
	ImGui::SliderFloat(I18n::T("Antizona muerta"), &g_edit.kbm_antidz, 0.0f, 0.5f, "%.2f");
	ImGui::SetNextItemWidth(520 * unit);
	ImGui::SliderInt(I18n::T("Velocidad de cámara (ratón)"), &g_edit.kbm_camera, 5, 15);
	ImGui::Checkbox(I18n::T("Cámara instantánea (frame warp)"), &g_edit.kbm_warp);
	if (g_mouse != nullptr) {
		const float h = 230 * unit, w = h * g_mouse_w / std::max(1, g_mouse_h);
		const ImVec2 at = {ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - w - 24 * unit, ImGui::GetWindowPos().y + 8 * unit};
		ImGui::GetWindowDrawList()->AddImage(g_mouse->GetTexRef(), at, {at.x + w, at.y + h});
	}
	ImGui::Spacing();
	ImGui::PushFont(nullptr, std::round(32.0f * unit));
	ImGui::TextColored(kGold, "%s", I18n::T("Teclas"));
	ImGui::PopFont();
	// A captured key or button for the row waiting.
	{
		std::scoped_lock lock(g_capture_mutex);
		if (g_capture_row >= 0 && !g_kbm_capture.load() && !g_captured.empty()) {
			if (g_captured != "\x1b" && g_capture_row < static_cast<int>(g_rows.size())) {
				// A key does one thing: another row with it is left without one (the launcher's rule).
				for (auto& r: g_rows)
					if (r.second == g_captured) r.second = "None";
				g_rows[g_capture_row].second = g_captured;
			}
			g_captured.clear();
			g_capture_row = -1;
		}
	}
	if (ImGui::BeginTable("##kbmkeys", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
		ImGui::TableSetupColumn("##c", ImGuiTableColumnFlags_WidthStretch, 1.1f);
		ImGui::TableSetupColumn("##k", ImGuiTableColumnFlags_WidthStretch, 1.0f);
		ImGui::TableSetupColumn("##b", ImGuiTableColumnFlags_WidthFixed, std::round(316 * unit));
		for (int i = 0; i < static_cast<int>(g_rows.size()); i++) {
			ImGui::PushID(i);
			ImGui::TableNextRow(0, std::round(52 * unit));
			ImGui::TableSetColumnIndex(0);
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(ControlName(g_rows[i].first));
			ImGui::TableSetColumnIndex(1);
			ImGui::AlignTextToFramePadding();
			if (g_capture_row == i) ImGui::TextColored(kWarn, "%s", I18n::T("Pulsa una tecla o un botón del ratón (Esc: cancelar)"));
			else ImGui::TextColored(kGold, "%s", KeyText(g_rows[i].second).c_str());
			ImGui::TableSetColumnIndex(2);
			const ImVec2 rb {std::round(150 * unit), std::round(44 * unit)};
			if (Anim::Button(I18n::T("Cambiar"), rb, unit) && g_capture_row < 0) {
				std::scoped_lock lock(g_capture_mutex);
				g_captured.clear();
				g_capture_row = i;
				g_kbm_capture = true;
			}
			ImGui::SameLine();
			if (Anim::Button(I18n::T("Quitar"), rb, unit)) g_rows[i].second = "None";
			ImGui::PopID();
		}
		ImGui::EndTable();
	}
	if (Anim::Button(I18n::T("Teclas de serie"), {0, std::round(50 * unit)}, unit)) {
		g_edit.kbm_map.clear();
		g_loaded = false;
	}
	ImGui::EndDisabled();
	ImGui::EndChild();
	// The rows as KYTY_KBM_MAP (the defaults: empty).
	if (g_loaded) {
		std::string map;
		for (const auto& [c, k]: g_rows) map += (map.empty() ? "" : ";") + c + "=" + k;
		g_edit.kbm_map = map == kKbmDefault ? std::string() : map;
	}
	const bool open = Footer(unit);
	if (!open) {
		g_kbm_capture = false;
		g_capture_row = -1;
		g_loaded      = false;
	}
	return open;
}
} // namespace F1Pages

// ---- (trucos6) The item lists from the game's OWN files in the user's game folder, read at run time (nothing copied):
//   <game>/cp11demonssouls/dvdroot/param/gameparam/gameparam.parambnd  (BND3: EquipParamWeapon / Protector / Accessory /
//     Goods / Magic .param; legacy PARAM: row count u16 at +0xa, rows {id, data offset, name offset} from +0x30)
//   <game>/cp11demonssouls/dvdroot/msg/<language>/item.msgbnd          (BND3 of FMG: names by id, UTF-16)
// Row layouts from dvdroot/paramdef/paramdef.paramdefbnd (field 0xb0 bytes, internal name at +0x90): EquipParamWeapon
// 184 bytes: iconId u16 +4, originEquipWep s32 +140 (the variant it is made from: Daga+1 10001 -> 10000; a path:
// "Rompemalla afilada+1" 10241 -> 10200 ... -> the base 10200's origin); Protector iconIdM u16 +4; Accessory iconId +4;
// Goods iconId u16 +0, goodsType u8 +2; Magic iconId s32 +4. FMG (the DeS layout): group count +0xc, string-offset
// table at [+0x18], groups {offset index, first id, last id} from +0x20; an offset below 0x20 is an empty string.
// Checked on PPSA01341 01.007.000's files: 1882 weapon rows, Daga 10000..10009, Rompemalla afilada+1 10241 (origin 10200).
namespace ItemDb {
struct Item {
	int         id = 0, icon = -1, origin = -1, root = 0, level = 0;
	bool        dup = false; // (trucos10) left out of the shop: a repeat / variant / test row (Load's last pass)
	int         model = 0;          // (trucos14) equipModelId (weapons / armour +2): the asset folder's number
	std::string bp, bp2;            // (trucos17) the remake's own name strings in the param row (bpPreviewKey / bpPreviewSuffix)
	int         price = 0, sub = 0; // (trucos7) basicPrice (souls); sub: weaponCategory (12 = shield) / magic 0 sorcery, 1 miracle
	std::string desc;               // (trucos7) the game's own lore text (item.msgbnd *うんちく.fmg, else *説明.fmg)
	std::string name, path; // path: the name without its "+n" (weapons: the upgrade path's own name)
};
struct Db {
	bool              tried = false, ok = false;
	std::string       why;
	std::vector<Item> cat[5]; // Armas, Armaduras, Anillos, Objetos, Magias y milagros
};
Db g_db;

using Bytes = std::string;
bool ReadFile(const std::filesystem::path& p, Bytes* out) {
	std::ifstream in(p, std::ios::binary);
	if (!in) return false;
	out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	return !out->empty();
}
template <class T> bool Get(const Bytes& b, size_t at, T* v) {
	if (at + sizeof(T) > b.size()) return false;
	std::memcpy(v, b.data() + at, sizeof(T));
	return true;
}
// BND3 (format 0x74: 24-byte entries from +0x20 {flags, size, offset, id, name offset, size}): name -> bytes.
std::vector<std::pair<std::string, Bytes>> Bnd3(const Bytes& b) {
	std::vector<std::pair<std::string, Bytes>> out;
	uint32_t n = 0;
	if (b.size() < 0x20 || b.compare(0, 4, "BND3") != 0 || !Get(b, 0x10, &n) || n > 4096) return out;
	for (uint32_t i = 0; i < n; i++) {
		uint32_t e[6] {};
		if (!Get(b, 0x20 + i * 24, &e)) break;
		if (e[2] + static_cast<uint64_t>(e[1]) > b.size() || e[4] >= b.size()) continue;
		const size_t end = b.find('\0', e[4]);
		out.emplace_back(b.substr(e[4], end == Bytes::npos ? 0 : end - e[4]), b.substr(e[2], e[1]));
	}
	return out;
}
std::string Utf16ToUtf8(const char* p, size_t units) {
	std::string s;
	for (size_t i = 0; i < units; i++) {
		uint32_t c = static_cast<uint8_t>(p[2 * i]) | (static_cast<uint8_t>(p[2 * i + 1]) << 8);
		if (c >= 0xd800 && c < 0xe000) c = '?';
		if (c < 0x80) s += static_cast<char>(c);
		else if (c < 0x800) s += static_cast<char>(0xc0 | (c >> 6)), s += static_cast<char>(0x80 | (c & 0x3f));
		else s += static_cast<char>(0xe0 | (c >> 12)), s += static_cast<char>(0x80 | ((c >> 6) & 0x3f)), s += static_cast<char>(0x80 | (c & 0x3f));
	}
	return s;
}
std::map<int, std::string> Fmg(const Bytes& b) {
	std::map<int, std::string> out;
	uint32_t groups = 0, table = 0;
	if (!Get(b, 0xc, &groups) || !Get(b, 0x18, &table) || groups > 100000) return out;
	for (uint32_t g = 0; g < groups; g++) {
		int32_t v[3] {};
		if (!Get(b, 0x20 + g * 12, &v) || v[2] < v[1] || v[2] - v[1] > 1000000) break;
		for (int32_t k = 0; k <= v[2] - v[1]; k++) {
			uint32_t so = 0;
			if (!Get(b, table + static_cast<size_t>(v[0] + k) * 4, &so) || so < 0x20 || so >= b.size()) continue;
			size_t e = so;
			while (e + 1 < b.size() && (b[e] != 0 || b[e + 1] != 0)) e += 2;
			if (e > so) out[v[1] + k] = Utf16ToUtf8(b.data() + so, (e - so) / 2);
		}
	}
	return out;
}
const Bytes* Find(const std::vector<std::pair<std::string, Bytes>>& files, const char* tail) {
	for (const auto& [n, b]: files)
		if (n.size() >= std::strlen(tail) && n.compare(n.size() - std::strlen(tail), Bytes::npos, tail) == 0) return &b;
	return nullptr;
}
void Load() {
	if (g_db.tried) return;
	g_db.tried = true;
	const auto load_t0 = std::chrono::steady_clock::now(); // (trucos8) measured: a disk cache only if this is over ~1 s
	struct Said {
		std::chrono::steady_clock::time_point t0;
		~Said() {
			std::printf("[trucos] lista: lectura y analisis en %lld ms\n",
			            static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()));
		}
	} said {load_t0};
	// (trucos9) The game folder the emulator runs (settings carpeta_juego: the same one the launch passes, its "Game args"
	// line reads <it>/packagecmdlineargs.txt), as UTF-8 -> a wide path on Windows (spaces / case kept as written).
	const std::filesystem::path root = std::filesystem::u8path(g_rt.launched.game) / "cp11demonssouls" / "dvdroot";
	const auto params_path = root / "param" / "gameparam" / "gameparam.parambnd";
	Bytes params, msg;
	if (!ReadFile(params_path, &params)) {
		g_db.why = "gameparam.parambnd no encontrado";
		std::error_code ec;
		std::printf("[trucos] lista: no se pudo leer %s (existe: %d, %s)\n", params_path.u8string().c_str(),
		            std::filesystem::exists(params_path, ec) ? 1 : 0, std::strerror(errno));
		return;
	}
	std::printf("[trucos] lista: leido %s (%zu bytes)\n", params_path.u8string().c_str(), params.size());
	// The names in the menu's language when the game has it, else Spanish, else English.
	const std::string& ml = g_rt.launched.menu_language; // (the menu's language id: es, en, pt-BR, fr, it, de)
	const char* first = ml.rfind("en", 0) == 0 ? "english" : ml.rfind("pt", 0) == 0 ? "portuguese" : ml.rfind("fr", 0) == 0 ? "french"
	                  : ml.rfind("it", 0) == 0 ? "italian" : ml.rfind("de", 0) == 0 ? "german" : "spanish";
	const char* order[] = {first, "spanish", "english", "na_english", "engus"};
	for (const char* l: order) {
		const auto mp = root / "msg" / l / "item.msgbnd";
		if (ReadFile(mp, &msg)) {
			std::printf("[trucos] lista: leido %s (%zu bytes)\n", mp.u8string().c_str(), msg.size());
			break;
		}
		std::printf("[trucos] lista: no esta %s\n", mp.u8string().c_str());
	}
	const auto pfiles = Bnd3(params);
	const auto mfiles = Bnd3(msg);
	std::printf("[trucos] lista: BND3 params %zu ficheros (%s), BND3 textos %zu ficheros (%s)\n", pfiles.size(), params.compare(0, 4, "BND3") == 0 ? "BND3" : "cabecera desconocida",
	            mfiles.size(), msg.size() >= 4 && msg.compare(0, 4, "BND3") == 0 ? "BND3" : "cabecera desconocida");
	// FMG files of item.msgbnd, by their (Japanese) names: goods, weapons, armor, rings, magic.
	const auto fmg = [&](const char* tail) {
		const Bytes* b = Find(mfiles, tail);
		return b != nullptr ? Fmg(*b) : std::map<int, std::string> {};
	};
	// (trucos9) The FMG names inside item.msgbnd are SHIFT-JIS bytes (the BND3 keeps the PS3 names as they are), not
	// UTF-8: with UTF-8 tails no file matched and every list came out empty (log 20261120-043912-622: "armas 0 ...").
	const auto n_goods = fmg("\x83\x41\x83\x43\x83\x65\x83\x80\x96\xBC" ".fmg");             // アイテム名
	const auto n_weap  = fmg("\x95\x90\x8A\xED\x96\xBC" ".fmg");                             // 武器名
	const auto n_arm   = fmg("\x96\x68\x8B\xEF\x96\xBC" ".fmg");                             // 防具名
	const auto n_ring  = fmg("\x83\x41\x83\x4E\x83\x5A\x83\x54\x83\x8A\x96\xBC" ".fmg"); // アクセサリ名
	const auto n_mag   = fmg("\x96\x82\x96\x40\x96\xBC" ".fmg");                             // 魔法名
	std::printf("[trucos] lista: item.msgbnd %zu bytes, %zu ficheros FMG; nombres: armas %zu, armaduras %zu, anillos %zu, objetos %zu, magias %zu\n",
	            msg.size(), mfiles.size(), n_weap.size(), n_arm.size(), n_ring.size(), n_goods.size(), n_mag.size());
	// (trucos7) Lore texts: アイテム / 武器 / 防具 / アクセサリ / 魔法 + うんちく.fmg (lore), else + 説明.fmg (short).
	const char* kUn = "\x82\xA4\x82\xF1\x82\xBF\x82\xAD" ".fmg", *kSe = "\x90\xE0\x96\xBE" ".fmg"; // (Shift-JIS, trucos9)
	const char* kPre[] = {"\x95\x90\x8A\xED", "\x96\x68\x8B\xEF", "\x83\x41\x83\x4E\x83\x5A\x83\x54\x83\x8A",
	                      "\x83\x41\x83\x43\x83\x65\x83\x80", "\x96\x82\x96\x40"};
	std::map<int, std::string> lore[5];
	for (int c = 0; c < 5; c++) {
		lore[c] = fmg((std::string(kPre[c]) + kSe).c_str());
		for (auto& [k, v]: fmg((std::string(kPre[c]) + kUn).c_str())) lore[c][k] = v;
	}
	struct Spec { const char* file; int cat, icon_off, icon_size; const std::map<int, std::string>* names; int price_off; };
	// (trucos7) basicPrice: EquipParamWeapon +172, Protector +48, Accessory +20, Goods +12 (paramdef); Magic has none.
	const Spec specs[] = {{"EquipParamWeapon.param", 0, 4, 2, &n_weap, 172}, {"EquipParamProtector.param", 1, 4, 2, &n_arm, 48},
	                      {"EquipParamAccessory.param", 2, 4, 2, &n_ring, 20}, {"EquipParamGoods.param", 3, 0, 2, &n_goods, 12},
	                      {"Magic.param", 4, 4, 4, &n_mag, -1}};
	for (const auto& sp: specs) {
		const Bytes* b = Find(pfiles, sp.file);
		uint16_t     n = 0;
		if (b == nullptr || !Get(*b, 0xa, &n)) {
			std::printf("[trucos] lista: PARAM %s no esta en gameparam.parambnd\n", sp.file);
			continue;
		}
		std::printf("[trucos] lista: PARAM %s %u filas, %zu nombres\n", sp.file, static_cast<unsigned>(n), sp.names->size());
		std::map<int, int> origin;
		for (uint32_t i = 0; i < n; i++) {
			uint32_t r[3] {};
			if (!Get(*b, 0x30 + i * 12, &r)) break;
			const auto nm = sp.names->find(static_cast<int>(r[0]));
			if (nm == sp.names->end() || nm->second.empty()) continue; // (rows without a name: tests, unused)
			Item it;
			it.id   = static_cast<int>(r[0]);
			it.name = nm->second;
			if (sp.icon_size == 2) {
				uint16_t v = 0;
				if (Get(*b, r[1] + sp.icon_off, &v)) it.icon = v;
			} else {
				int32_t v = 0;
				if (Get(*b, r[1] + sp.icon_off, &v)) it.icon = v;
			}
			if (sp.cat == 0) {
				int32_t o = -1;
				if (Get(*b, r[1] + 140, &o)) it.origin = o;
			}
			// The "+n" of the name: the level; the rest: the path (weapons) or the name.
			it.path        = it.name;
			const size_t p = it.name.rfind('+');
			if (p != std::string::npos && p + 1 < it.name.size() && std::all_of(it.name.begin() + p + 1, it.name.end(), ::isdigit)) {
				it.level = std::atoi(it.name.c_str() + p + 1);
				it.path  = it.name.substr(0, p);
			}
			if (sp.price_off >= 0) {
				int32_t v = 0;
				if (Get(*b, r[1] + sp.price_off, &v)) it.price = std::max(0, v);
			}
			if (sp.cat == 0) it.sub = static_cast<uint8_t>(b->size() > r[1] + 6 ? (*b)[r[1] + 6] : 0); // weaponCategory +6
			if (sp.cat == 1) it.sub = static_cast<uint8_t>(b->size() > r[1] + 8 ? (*b)[r[1] + 8] : 0); // (trucos14) protectorCategory +8
			if (sp.cat <= 1) {
				uint16_t m = 0;
				if (Get(*b, r[1] + 2, &m)) it.model = m; // equipModelId +2 (paramdef)
			}
			// (trucos17) The remake's string fields (paramdef type "string": a u32 offset from the param file's start):
			// Goods / Accessory bpPreviewKey +64 ("KingsSoul", "PrisonKey"...), Weapon bpPreviewSuffix +176 ("dagger"),
			// Protector bpPreviewSuffix +144 / _Female +152. The game builds Tex_InvIcon_ from them (0xbd530 goods:
			// row +0x40; 0xbd920 weapons: row +0xb0; 0xbe190 rings: row +0x40).
			{
				const auto str_at = [&](uint32_t field, std::string* out) {
					uint32_t so = 0;
					if (!Get(*b, r[1] + field, &so) || so == 0 || so >= b->size()) return;
					const size_t e = b->find('\0', so);
					if (e != Bytes::npos && e - so < 128) out->assign(b->data() + so, e - so);
				};
				if (sp.cat == 2 || sp.cat == 3) str_at(64, &it.bp);
				if (sp.cat == 0) str_at(176, &it.bp);
				if (sp.cat == 1) str_at(144, &it.bp), str_at(152, &it.bp2);
			}
			if (sp.cat == 4) it.sub = static_cast<uint8_t>(b->size() > r[1] ? (*b)[r[1]] : 0);         // ezStateBehaviorType +0
			{
				const int lc = sp.cat == 0 ? 0 : sp.cat == 1 ? 1 : sp.cat == 2 ? 2 : sp.cat == 3 ? 3 : 4;
				const auto d = lore[lc].find(it.id);
				if (d != lore[lc].end()) it.desc = d->second;
			}
			g_db.cat[sp.cat].push_back(it);
		}
	}
	// Weapons: each variant's root (follow originEquipWep to the row without one).
	{
		std::map<int, int> org;
		for (const auto& it: g_db.cat[0]) org[it.id] = it.origin;
		for (auto& it: g_db.cat[0]) {
			int r = it.id;
			for (int guard = 0; guard < 64; guard++) {
				const auto o = org.find(r);
				if (o == org.end() || o->second < 0 || org.count(o->second) == 0) break;
				r = o->second;
			}
			it.root = r;
		}
	}
	// (trucos10) One entry per item: weapons / shields only the base (no originEquipWep, no "+n"); every category drops
	// rows the game never translated (Japanese placeholders), test / debug rows ("Prueba", "test", "dummy", "Catalizador
	// universal") and any repeat of a name already listed (the first row, the lowest id, is kept).
	for (int c = 0; c < 5; c++) {
		std::set<std::string> seen;
		for (auto& it: g_db.cat[c]) {
			std::string low = it.name;
			for (auto& ch: low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
			bool bad = it.name.rfind("Catalizador universal", 0) == 0 || low.find("prueba") != std::string::npos ||
			           low.find("test") != std::string::npos || low.find("dummy") != std::string::npos;
			for (unsigned char ch: it.name)
				if (ch >= 0xe3) bad = true;
			if (c == 0 && (it.origin >= 0 || it.level != 0)) bad = true;
			it.dup = bad || !seen.insert(it.name).second;
		}
	}
	{
		int n[7] {};
		for (const auto& it: g_db.cat[0])
			if (!it.dup) n[it.sub == 12 ? 1 : 0]++;
		for (int c = 1; c <= 3; c++)
			for (const auto& it: g_db.cat[c]) n[c + 1] += it.dup ? 0 : 1;
		for (const auto& it: g_db.cat[4])
			if (!it.dup) n[it.sub == 1 ? 6 : 5]++;
		std::printf("[trucos] tienda (una entrada por objeto): armas %d, escudos %d, armaduras %d, anillos %d, objetos %d, magias %d, milagros %d\n",
		            n[0], n[1], n[2], n[3], n[4], n[5], n[6]);
	}
	g_db.ok = !g_db.cat[0].empty();
	std::printf("[trucos] lista desde los ficheros del juego: armas %zu, armaduras %zu, anillos %zu, objetos %zu, magias %zu\n",
	            g_db.cat[0].size(), g_db.cat[1].size(), g_db.cat[2].size(), g_db.cat[3].size(), g_db.cat[4].size());
	if (!g_db.ok) g_db.why = "sin datos";
}
} // namespace ItemDb

// ---- (trucos) F2 > Trucos: the game's OWN debug switches and the player's soul count, from the port's thread.
// Every address is eboot 01.007.000 (vaddr; at run time + 0x900000000, as rtx1-convars.h), found in the eboot (NOTAS.md):
//  - The bool convars cp11_characters_* (Bluepoint's bpe::ConVar, constructors in 0x355700-0x356100): each keeps its
//    value in a static byte (the constructor's 3rd argument, rdx) that the game code reads with cmpb $1:
//    playerNoDamage 0x27cbd50 (read in the damage routine at 0x32d689: when set and the hit one is the player, vtable
//    +0x120, the HP subtraction at 0x32d6fd is skipped), playerNoDead 0x27cbc60 (read at 0x325972, 0x32ae5a, 0x32d7af,
//    0x32fd13), playerNoStaminaConsume 0x27cbf30 (read at 0x331a68 in the stamina cost), playerExterminate 0x27cbb70
//    (read at 0x32d588: when the attacker is the player the damage becomes 999999.0, the float at 0x237ec28).
//  - Souls: the handler of the game's console command cp11_setSouls (0x4529c0) does
//    *(*(*(0x27e6948)) + 0x10) + 0x80 = min(souls, 999999999); cp11_addSouls (0x452a20) reads the same int.
//    Null while there is no player data (title screen): then nothing is read or written.
// Writes: TryWriteCpuBacking (false: not written, said once in the log). The switches are not saved (off at each start).
namespace Cheats {
constexpr uint64_t kBase = 0x900000000ull;
struct Flag {
	const char* label;
	const char* tip;
	uint64_t    vaddr;
	Icons::Kind icon;
	bool        on = false, written = false;
};
Flag g_flags[] = {
    {"Invencible", "Ningún golpe te quita vida.", 0x27cbd50, Icons::Shield},
    {"No morir", "Aunque la vida llegue a cero, sigues en pie.", 0x27cbc60, Icons::Potion},
    {"Aguante infinito", "Rodar, atacar y correr no gastan aguante.", 0x27cbf30, Icons::Ring},
    {"Matar de un golpe", "Cada golpe tuyo hace 999999 de daño.", 0x27cbb70, Icons::Sword},
    // (trucos21) vaddr 0: not a byte switch; Loader::DemonsSoulsInventory::SetNoWeight (equipment weight class forced
    // to Light via the game's own override convar at eboot+0x27c1268, item-burden limit lifted).
    {"Sin límite de peso", "Rueda y corre ligero con cualquier equipo; lleva todos los objetos que quieras.", 0, Icons::Helm},
};
bool AnySwitchOn() {
	for (const auto& f: g_flags)
		if (f.on) return true;
	return false;
}
std::chrono::steady_clock::time_point g_next {};
int         g_souls_edit = 0;
bool        g_souls_loaded = false;
std::string g_note;

// The address of the soul count, or 0 (no player data yet).
uint64_t SoulsAddress() {
	uint64_t a = 0, pgd = 0;
	if (!Libs::LibKernel::Memory::NativeGuardedRead(&a, reinterpret_cast<const void*>(kBase + 0x27e6948ull), 8) || a < 0x10000) return 0;
	if (!Libs::LibKernel::Memory::NativeGuardedRead(&pgd, reinterpret_cast<const void*>(a + 0x10), 8) || pgd < 0x10000) return 0;
	return pgd + 0x80;
}
bool ReadSouls(int* out) {
	const uint64_t at = SoulsAddress();
	int32_t        v  = -1;
	if (at == 0 || !Libs::LibKernel::Memory::NativeGuardedRead(&v, reinterpret_cast<const void*>(at), 4) || v < 0 || v > 999999999) return false;
	*out = v;
	return true;
}
// Every frame (DrawInGame); work every 0.5 s: the switches that are on are written again (cheap, one byte each); a
// switch turned off is written 0 once.
void Tick() {
	const auto now = std::chrono::steady_clock::now();
	if (now < g_next) return;
	g_next = now + std::chrono::milliseconds(500);
	// (stats1) "Deshabilitar trucos y mejoras" (launcher, settings.ini sin_trucos=1): every switch forced off (written 0
	// on the game thread below, once) and the game-thread cheat module locked (items, souls, attributes refused).
	// (onlineclean) no_cheats OR launched online: lock the game-thread module and force every switch off.
	const bool blocked = CheatsBlocked();
	Loader::DemonsSoulsInventory::SetLocked(blocked);
	if (blocked)
		for (auto& f: g_flags)
			f.on = false;
	// (trucos4) Written by the game thread (Loader::DemonsSoulsInventory, each frame): the port thread's TryWriteCpuBacking
	// was refused for the eboot's data ("[port] trucos: write refused at eboot+0x27cbd50", log 20261120-032627-014).
	for (auto& f: g_flags) {
		if (f.on != f.written) std::printf("[trucos] menu: %s -> %s (eboot+0x%llx)\n", f.label, f.on ? "on" : "off", static_cast<unsigned long long>(f.vaddr));
		if (f.vaddr == 0) {
			if (f.on || f.written) Loader::DemonsSoulsInventory::SetNoWeight(f.on); // (trucos21)
		} else if (f.on || f.written) Loader::DemonsSoulsInventory::SetFlag(f.vaddr, f.on);
		f.written = f.on;
	}
}

// ---- (stats1) Nivel personaje: the 8 attributes, the soul level and what follows from them, read from the player data
// (pgd = *(*(0x27e6948)+0x10); layout in loader/demonsSoulsInventory.cpp: hp +0x14, max +0x18; mp +0x20, max +0x24;
// stamina +0x30, max +0x34; attribute i base +0x40+8*i, current +0x44+8*i; souls +0x80; soul level +0x90) with
// guarded reads on this thread; written by the game thread the way the game's level-up does (SetStats).
// (stats4) The stored soul level (pgd+0x90) is kept incrementally by the game (0x4ab8b5: += delta per levelled point), not
// recomputed as sum-80. 0x3007a0 (clamp(sum,81,1079)-80) is a derived/UI helper over a stat block, NOT the stored soul
// level: a real character reads level == sum(attributes) (verified: level 131, attributes summing to 131). The apply moves
// the soul level by the same deltas the stats move, so no fixed per-class constant is needed and real saves are accepted.
const char* const kAttrNames[8] = {"Vitalidad", "Intelecto", "Aguante", "Fuerza", "Destreza", "Magia", "Fe", "Suerte"};
const char* const kAttrTips[8]  = {"Más vida máxima.", "Más maná máximo y más huecos de hechizos.", "Más aguante y más carga de equipo.",
                                   "Daño y requisitos de armas pesadas.", "Daño y requisitos de armas ágiles.", "Daño de las hechicerías.",
                                   "Daño de los milagros y más huecos de milagros.", "Más objetos encontrados y resistencia a la plaga."};
struct StatsView {
	bool    ok = false, consistent = false;
	int32_t base[8] {}, cur[8] {};
	int32_t level = 0, hp = 0, hp_max = 0, mp = 0, mp_max = 0, sp = 0, sp_max = 0;
};
bool ReadStats(StatsView* v) {
	*v = {};
	const uint64_t souls_at = SoulsAddress(); // (pgd + 0x80; 0 without player data)
	if (souls_at == 0) return false;
	const uint64_t pgd = souls_at - 0x80;
	int32_t        raw[0x84 / 4] {}; // pgd+0x10 .. pgd+0x94
	if (!Libs::LibKernel::Memory::NativeGuardedRead(raw, reinterpret_cast<const void*>(pgd + 0x10), sizeof(raw))) return false;
	const auto at = [&](uint32_t off) { return raw[(off - 0x10) / 4]; };
	v->hp = at(0x14), v->hp_max = at(0x18), v->mp = at(0x20), v->mp_max = at(0x24), v->sp = at(0x30), v->sp_max = at(0x34);
	v->level = at(0x90);
	int  sum = 0;
	bool range = true;
	for (int i = 0; i < 8; i++) {
		v->base[i] = at(0x40 + 8 * i);
		v->cur[i]  = at(0x44 + 8 * i);
		sum += v->cur[i];
		range = range && v->base[i] >= 1 && v->base[i] <= 99 && v->cur[i] >= 1 && v->cur[i] <= 99;
	}
	v->ok = true;
	// (stats4) "consistent" = the values are in range and the stored soul level is sane. It no longer requires
	// level == sum - 80: that was wrong (0x3007a0's sum-80 is a UI/derived helper, not the stored soul level, which the
	// game keeps incrementally at pgd+0x90). Real saves (e.g. level 131 with attributes summing to 131) failed it and were
	// shown as read-only with a false "no cuadran" warning. Range + a sane level is all the game actually guarantees.
	(void)sum;
	v->consistent = range && v->level >= 1 && v->level <= 999;
	return true;
}
int  g_attr_edit[8] {};
bool g_attr_loaded = false;
int  g_attr_seq    = -1; // the game thread's result counter seen
std::string g_attr_note;
bool g_attr_said_bad = false;
float g_attr_hot[8] {}; // (stats2) per-attribute plate hover glow
// (stats2) Display order asked by the owner (Vitalidad, Fuerza, Aguante, Destreza, Intelecto, Magia, Fe, Suerte) mapped
// to the game's attribute index (0 Vit, 1 Int, 2 End, 3 Str, 4 Dex, 5 Mag, 6 Fai, 7 Luck). g_attr_edit stays in game order.
const int kAttrOrder[8] = {0, 3, 2, 4, 1, 5, 6, 7};
// Clean menu-style icons (ds-style.h vector set, gold disc like the rest of the menu), one per game index. The real game
// stat icons live only inside a 335 MB hash-indexed UI atlas with no sprite names, so they cannot be pulled out reliably.
const DsStyle::Icon kAttrIcon[8] = {DsStyle::Icon::Potion, DsStyle::Icon::Soul, DsStyle::Icon::Shield, DsStyle::Icon::Sword,
                                    DsStyle::Icon::Ring, DsStyle::Icon::Flame, DsStyle::Icon::Sun, DsStyle::Icon::Stone};

// (stats2) One attribute as a carved plate (ds-style.h look): gold medallion with its icon, the name in Cinzel, the value
// big in gold with the base in parentheses when changed, and - / + buttons on the right (held: repeat). Edits g_attr_edit
// (1..99). Returns nothing; the whole block is wrapped in BeginDisabled when the read is not consistent.
void AttrRow(int gi, float unit, const StatsView& v) {
	int&         e  = g_attr_edit[gi];
	ImGui::PushID(gi);
	ImDrawList*  dl = ImGui::GetWindowDrawList();
	const float  al = ImGui::GetStyle().Alpha, t = Souls::Time();
	const ImVec2 at = ImGui::GetCursorScreenPos();
	const float  w  = ImGui::GetContentRegionAvail().x;
	const float  h  = std::round(78 * unit);
	// (stats3) THE navigable item, the same idiom the rest of the F2 menu uses (Cheats::Row, SelectorRow): one full-row
	// ImGui::Selectable. So the menu's own keyboard / gamepad navigation reaches it (D-pad up / down moves between the
	// attribute rows). The earlier - / + were absolutely-placed ImGui::Button widgets that the menu's navigation never
	// reached (and they sat inside a BeginDisabled), so neither the pad nor a click changed anything.
	const bool clicked = ImGui::Selectable("##arow", false, 0, {w, h});
	const bool focused = ImGui::IsItemFocused();
	const bool hoverrow = ImGui::IsItemHovered();
	float&     vh = g_attr_hot[gi];
	const float target = (focused || hoverrow) ? 1.0f : 0.0f;
	vh += (target - vh) * 0.2f;
	if (std::fabs(vh - target) > 0.01f) Bump();
	const ImVec2 a {std::round(at.x), std::round(at.y)}, b {std::round(at.x + w), std::round(at.y + h - 8 * unit)};
	const float  gh = b.y - a.y;
	// Right-side stepper: the - glyph, the value, the + glyph.
	const float plus_c  = b.x - std::round(40 * unit);
	const float minus_c = plus_c - std::round(150 * unit);
	const float val_cx  = (plus_c + minus_c) * 0.5f;
	// Change the value: Left / Right (D-pad, left stick, arrows; repeats while held) when the row is focused, as
	// SelectorRow does; or a click / Cross on the stepper's left (down) or right (up) half.
	int step = 0;
	if (focused) {
		if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft) || ImGui::IsKeyPressed(ImGuiKey_GamepadLStickLeft))
			step = -1;
		if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight) || ImGui::IsKeyPressed(ImGuiKey_GamepadLStickRight))
			step = +1;
	}
	if (clicked) {
		// A click on the stepper (right side): left half down, right half up. Cross with no mouse move counts as up.
		const float mx = ImGui::GetIO().MousePos.x;
		if (!ImGui::IsMouseDown(0) && !ImGui::IsMouseReleased(0)) step = +1; // activated by Cross / Enter, not the mouse
		else if (mx >= minus_c - std::round(34 * unit)) step = mx < val_cx ? -1 : +1;
	}
	if (step != 0) {
		e = std::clamp(e + step, 1, 99);
		Bump();
	}
	// Plate + medallion + name, drawn over the selectable (as Cheats::Row).
	DsStyle::CandlePlate(dl, a, b, vh, 0.0f, al, t + gi, unit);
	const ImVec2 c {std::round(a.x + 28 * unit + gh * 0.34f), std::round((a.y + b.y) * 0.5f)};
	DsStyle::Medallion(dl, kAttrIcon[gi], c, std::round(gh * 0.36f), al, vh, t, 0.0f);
	ImFont*     tf = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
	const float ts = std::round(30 * unit);
	const float tx = std::round(c.x + gh * 0.36f + 30 * unit);
	DsStyle::Title(dl, tf, ts, {tx, std::round((a.y + b.y) * 0.5f - ts * 0.62f)}, I18n::T(kAttrNames[gi]), al, 0.05f);
	// - / + glyphs, brighter when the row is focused so you see which one Left / Right move.
	const ImU32  arrow = DsStyle::C(focused ? DsStyle::kGoldHi : DsStyle::kBoneDim, al);
	const float  asz = std::round(28 * unit), cy = std::round((a.y + b.y) * 0.5f);
	const ImVec2 mm = tf->CalcTextSizeA(asz, FLT_MAX, 0, "-");
	dl->AddText(tf, asz, {std::round(minus_c - mm.x * 0.5f), std::round(cy - mm.y * 0.5f)}, arrow, "-");
	const ImVec2 pm = tf->CalcTextSizeA(asz, FLT_MAX, 0, "+");
	dl->AddText(tf, asz, {std::round(plus_c - pm.x * 0.5f), std::round(cy - pm.y * 0.5f)}, arrow, "+");
	// Value, big and gold, with the base in parentheses when changed.
	char num[16];
	std::snprintf(num, sizeof(num), "%d", e);
	const float  ns = std::round(34 * unit);
	const ImVec2 nsz = tf->CalcTextSizeA(ns, FLT_MAX, 0, num);
	dl->AddText(tf, ns, {std::round(val_cx - nsz.x * 0.5f), std::round(cy - ns * 0.5f)},
	            DsStyle::C(e != v.base[gi] ? DsStyle::kGoldHi : DsStyle::kBone, al), num);
	if (e != v.base[gi]) {
		char base[24];
		std::snprintf(base, sizeof(base), "(%d)", v.base[gi]);
		const float  bs = std::round(18 * unit);
		const ImVec2 bsz = tf->CalcTextSizeA(bs, FLT_MAX, 0, base);
		dl->AddText(tf, bs, {std::round(val_cx - bsz.x * 0.5f), std::round(b.y - bs - 2 * unit)}, DsStyle::C(DsStyle::kBoneDim, al), base);
	}
	Tip(kAttrTips[gi]);
	ImGui::PopID();
}

void LevelSection(float unit) {
	ImGui::Spacing();
	ImGui::Separator();
	ImGui::PushFont(DisplayFont(), std::round(34 * unit));
	ImGui::TextColored(kGold, "%s", I18n::T("Nivel personaje"));
	ImGui::PopFont();
	StatsView  v;
	const bool have   = ReadStats(&v);
	const bool online = g_rt.launched.multiplayer; // (the game was started with the online on: no stat edits there)
	if (online) {
		ImGui::TextColored(kDim, "%s", I18n::T("No disponible con la conexión online activada (evita expulsiones y desincronizar a otros jugadores)."));
		return;
	}
	if (!have) {
		g_attr_loaded = false;
		ImGui::TextColored(kDim, "%s", I18n::T("El nivel se puede cambiar cuando estás dentro de la partida."));
		return;
	}
	int seq = 0;
	const int result = Loader::DemonsSoulsInventory::StatsResult(&seq);
	if (g_attr_seq < 0) g_attr_seq = seq;
	if (seq != g_attr_seq) { // the game thread answered: show it and read the new values
		g_attr_seq    = seq;
		g_attr_loaded = false;
		g_attr_note   = I18n::T(result == 1 ? "Nivel cambiado." : result == -2 ? "No se cambió: los valores del juego no son los esperados." : "No se pudo: entra en la partida primero.");
	}
	if (!g_attr_loaded) {
		for (int i = 0; i < 8; i++)
			g_attr_edit[i] = v.base[i];
		g_attr_loaded = true;
	}
	if (!v.consistent && !g_attr_said_bad) {
		g_attr_said_bad = true;
		std::printf("[trucos] nivel: lectura no coherente (nivel %d; atributos %d %d %d %d %d %d %d %d): solo lectura\n", v.level, v.cur[0], v.cur[1], v.cur[2],
		            v.cur[3], v.cur[4], v.cur[5], v.cur[6], v.cur[7]);
	}
	int sum_new = 0, sum_cur = 0;
	for (int i = 0; i < 8; i++)
		sum_new += g_attr_edit[i], sum_cur += v.base[i];
	const int level_new = std::clamp(sum_new - sum_cur + v.level, 1, 999); // (stats4) current level + sum(deltas), as the game's level-up (0x4ab8b5); matches ApplyStats' write
	// (stats2) The soul level big at the top: a gold Soul medallion, the number in Cinzel, and the "-> N" preview while editing.
	ImDrawList*  dl = ImGui::GetWindowDrawList();
	const float  al = ImGui::GetStyle().Alpha, tm = Souls::Time();
	ImFont*      tf = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
	{
		const ImVec2 at  = ImGui::GetCursorScreenPos();
		const float  hh  = std::round(96 * unit);
		const float  mr  = std::round(hh * 0.40f);
		const ImVec2 mc {std::round(at.x + 20 * unit + mr), std::round(at.y + hh * 0.5f)};
		DsStyle::Medallion(dl, DsStyle::Icon::Soul, mc, mr, al, 0.6f, tm, 0.8f);
		const float lx = std::round(mc.x + mr + 26 * unit);
		DsStyle::SmallCaps(dl, tf, std::round(20 * unit), {lx, std::round(at.y + 14 * unit)}, I18n::T("Nivel de alma"), DsStyle::C(DsStyle::kBoneDim, al));
		char big[16];
		std::snprintf(big, sizeof(big), "%d", v.level);
		const float ls = std::round(54 * unit);
		DsStyle::GoldTitle(dl, tf, ls, {lx, std::round(at.y + 34 * unit)}, big, al, tm, 0.04f);
		const ImVec2 lsz = tf->CalcTextSizeA(ls, FLT_MAX, 0, big);
		if (level_new != v.level) {
			const float ax = std::round(lx + lsz.x + 20 * unit);
			dl->AddText(tf, std::round(30 * unit), {ax, std::round(at.y + 46 * unit)}, DsStyle::C(DsStyle::kBoneDim, al), "->");
			char nn[16];
			std::snprintf(nn, sizeof(nn), "%d", level_new);
			DsStyle::GoldTitle(dl, tf, ls, {std::round(ax + 44 * unit), std::round(at.y + 34 * unit)}, nn, al, tm, 0.04f);
		}
		ImGui::Dummy({0, hh});
	}
	// Vida / Maná / Aguante, max recomputed by the game.
	char line[256];
	std::snprintf(line, sizeof(line), "%s %d / %d          %s %d / %d          %s %d / %d", I18n::T("Vida"), v.hp, v.hp_max, I18n::T("Maná"), v.mp,
	              v.mp_max, I18n::T("Aguante"), v.sp, v.sp_max);
	ImGui::TextColored(kDim, "%s", line);
	ImGui::Spacing();
	// (stats3) The 8 attributes, in the owner's order. They are interactive even when the read looks off: the actual write
	// (Aplicar -> SetStats -> ApplyStats on the game thread) re-checks every value and the soul-level formula and refuses
	// safely, so nothing wrong is written; the row is no longer disabled (that is what stopped the pad from moving them).
	for (int slot = 0; slot < 8; slot++)
		AttrRow(kAttrOrder[slot], unit, v);
	ImGui::Spacing();
	const ImVec2 bs {std::round(200 * unit), std::round(44 * unit)};
	ImGui::PushID("nivel");
	if (Anim::Button(I18n::T("Aplicar"), bs, unit)) {
		const bool sent = Loader::DemonsSoulsInventory::SetStats(g_attr_edit);
		if (sent) Port::MarkCheatsUsed("Nivel personaje"); // (trucos-online)
		g_attr_note = I18n::T(sent ? "Aplicando..." : "No se pudo: entra en la partida primero.");
		std::printf("[trucos] nivel: pedido %d %d %d %d %d %d %d %d (nivel %d -> %d)\n", g_attr_edit[0], g_attr_edit[1], g_attr_edit[2], g_attr_edit[3],
		            g_attr_edit[4], g_attr_edit[5], g_attr_edit[6], g_attr_edit[7], v.level, level_new);
	}
	ImGui::SameLine();
	if (Anim::Button(I18n::T("Deshacer"), bs, unit)) g_attr_loaded = false, g_attr_note.clear();
	ImGui::PopID();
	if (!v.consistent) ImGui::TextColored(kDim, "%s", I18n::T("Aviso: los valores leídos no cuadran con el nivel del juego; al aplicar, el juego puede rechazarlo."));
	ImGui::TextColored(kDim, "%s", I18n::T("Vida, maná y aguante máximos los recalcula el propio juego, como al subir de nivel en el Nexo. Bajar Intelecto o Fe puede quitar huecos de hechizos."));
	if (!g_attr_note.empty()) ImGui::TextColored(kDim, "%s", g_attr_note.c_str());
}

// (trucos7) One cheat as a full-width carved plate: the engraved icon in a soul-lit medallion (it breathes when the cheat
// is on), the name in Cinzel, what it does under it, and a switch on the right (off: dark iron; on: gold track with a
// glowing soul knob). The whole plate toggles (click, A / Cross, Enter); the logic is the same `f.on` as before.
float g_row_hot[8] {}, g_row_on[8] {};
void Row(Flag& f, float unit, int index) {
	ImGui::PushID(index);
	const float  h  = std::round(96 * unit);
	const ImVec2 at = ImGui::GetCursorScreenPos();
	const float  w  = ImGui::GetContentRegionAvail().x;
	if (ImGui::Selectable("##row", false, 0, {w, h})) f.on = !f.on;
	const bool hot = ImGui::IsItemHovered() || ImGui::IsItemFocused();
	float& vh = g_row_hot[index & 7];
	float& vo = g_row_on[index & 7];
	vh += ((hot ? 1.0f : 0.0f) - vh) * 0.2f;
	vo += ((f.on ? 1.0f : 0.0f) - vo) * 0.2f;
	if (std::fabs(vh - (hot ? 1.0f : 0.0f)) > 0.01f || std::fabs(vo - (f.on ? 1.0f : 0.0f)) > 0.01f) Bump();
	// (trucos11) ds-style.h: the candle-lit plate, an aged-metal medallion (a soul glow while on), the name in tracked
	// bone-white Cinzel, what it does in muted bone, small-caps state and the iron / soul-light switch.
	ImDrawList*  dl = ImGui::GetWindowDrawList();
	const float  al = ImGui::GetStyle().Alpha, t = Souls::Time();
	const ImVec2 a {std::round(at.x), std::round(at.y)}, b {std::round(at.x + w), std::round(at.y + h - 10 * unit)};
	DsStyle::CandlePlate(dl, a, b, vh, 0.0f, al, t + index, unit);
	const float  rh = b.y - a.y;
	const ImVec2 c {std::round(a.x + 34 * unit + rh * 0.36f), std::round((a.y + b.y) * 0.5f)};
	DsStyle::Medallion(dl, StyleIcon(f.icon), c, std::round(rh * 0.38f), al, vh, t, vo);
	ImFont*     tf = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
	const float ts = std::round(34 * unit), ds = std::round(22 * unit);
	const float tx = std::round(c.x + rh * 0.38f + 34 * unit);
	DsStyle::Title(dl, tf, ts, {tx, std::round(a.y + 12 * unit)}, I18n::T(f.label), al, 0.05f);
	dl->AddText(ImGui::GetFont(), ds, {tx + 2, std::round(a.y + 22 * unit + ts)}, DsStyle::C(DsStyle::kBoneDim, al), I18n::T(f.tip));
	const float  sw = std::round(100 * unit), sh = std::round(40 * unit);
	const ImVec2 s0 {std::round(b.x - 50 * unit - sw), std::round((a.y + b.y - sh) * 0.5f)};
	DsStyle::Toggle(dl, s0, {s0.x + sw, s0.y + sh}, vo, al, t);
	DsStyle::SmallCaps(dl, tf, std::round(20 * unit), {std::round(s0.x - 150 * unit), std::round((a.y + b.y) * 0.5f - 10 * unit)}, I18n::T(f.on ? "Activo" : "Apagado"),
	                   f.on ? DsStyle::C(DsStyle::kGoldHi, al) : DsStyle::C(DsStyle::kBoneDim, al));
	ImGui::PopID();
}

bool Draw(float width, float height, float unit) {
	using namespace F1Pages;
	if (CheatsBlocked()) return false; // (stats1) "Deshabilitar trucos y mejoras"; (onlineclean) el online tambien lo fuerza
	Begin("Trucos", width, height, unit);
	Bump(); // (the icons' glow breathes)
	ImGui::BeginChild("##trucos", {0, -std::round(84 * unit)}, ImGuiChildFlags_NavFlattened);
	Anim::SmoothScroll();
	if (const int d = g_page_scroll.exchange(0); d != 0) ImGui::SetScrollY(ImGui::GetScrollY() + d * ImGui::GetWindowHeight() * 0.6f);
	ImGui::TextColored(kDim, "%s", I18n::T("Solo para jugar sin conexión. Se apagan al cerrar el juego."));
	ImGui::Spacing();
	int i = 0;
	for (auto& f: g_flags) Row(f, unit, i++);
	ImGui::Spacing();
	ImGui::Separator();
	// Souls: read when the page opens and on Leer; written on Aplicar.
	{
		const float  side = std::round(40 * unit);
		const ImVec2 at   = ImGui::GetCursorScreenPos();
		ImGui::Dummy({side, side});
		const ImVec2 c {std::round(at.x + side * 0.5f), std::round(at.y + side * 0.5f)};
		const float  pulse = 0.5f + 0.5f * std::sin(static_cast<float>(ImGui::GetTime()) * 1.7f);
		ImDrawList*  dl    = ImGui::GetWindowDrawList();
		for (int k = 3; k >= 1; k--)
			dl->AddCircleFilled(c, side * (0.3f + 0.12f * k), ImGui::GetColorU32(ImVec4 {0.72f, 0.82f, 1.0f, ImGui::GetStyle().Alpha * (0.03f + 0.04f * pulse)}), 32);
		Icons::Draw(dl, Icons::Soul, c, side * (0.78f + 0.04f * pulse), ImGui::GetColorU32(ImVec4 {0.86f, 0.90f, 0.98f, ImGui::GetStyle().Alpha}));
		ImGui::SameLine();
		ImGui::AlignTextToFramePadding();
		ImGui::TextUnformatted(I18n::T("Almas"));
		ImGui::SameLine();
		if (!g_souls_loaded) g_souls_loaded = ReadSouls(&g_souls_edit);
		const bool have = SoulsAddress() != 0;
		ImGui::BeginDisabled(!have);
		ImGui::SetNextItemWidth(std::round(300 * unit));
		ImGui::InputInt("##almas", &g_souls_edit, 1000, 100000);
		Anim::Highlight(unit);
		g_souls_edit = std::clamp(g_souls_edit, 0, 999999999);
		ImGui::SameLine();
		const ImVec2 bs {std::round(170 * unit), std::round(44 * unit)};
		if (Anim::Button(I18n::T("Aplicar"), bs, unit)) {
			const bool ok = Loader::DemonsSoulsInventory::SetSouls(g_souls_edit);
			if (ok) Port::MarkCheatsUsed("Almas"); // (trucos-online)
			g_note = I18n::T(ok ? "Almas cambiadas." : "No se pudo: entra en la partida primero.");
		}
		ImGui::SameLine();
		if (Anim::Button(I18n::T("Leer"), bs, unit)) g_souls_loaded = ReadSouls(&g_souls_edit);
		ImGui::EndDisabled();
		if (!have) ImGui::TextColored(kDim, "%s", I18n::T("Las almas se pueden editar cuando estás dentro de la partida."));
		if (!g_note.empty()) ImGui::TextColored(kDim, "%s", g_note.c_str());
	}
	LevelSection(unit); // (stats1) Nivel personaje
	// (trucos7) Items and spells: their own F2 section now, "Armas y objetos" (Shop::Draw).
	ImGui::EndChild();
	ImGui::Separator();
	bool close = Anim::Button(I18n::T("Volver"), {std::round(240 * unit), std::round(54 * unit)}, unit);
	ImGui::SameLine();
	ImGui::TextDisabled("%s", I18n::T("L1 / L2: desplazarse   Círculo / Esc: volver"));
	PageZoom(); // (trucos7)
	ImGui::End();
	ImGui::PopStyleColor(g_style_colors);
	ImGui::PopStyleVar(g_style_vars);
	g_style_colors = g_style_vars = 0;
	ImGui::PopFont();
	if (close) {
		g_souls_loaded = false;
		g_note.clear();
		g_attr_loaded = false; // (stats1)
		g_attr_note.clear();
	}
	return !close;
}
} // namespace Cheats

namespace ShopIcons { // (trucos14) below
void Tick();
bool Draw(ImDrawList* dl, int id, ImVec2 c, float r, float alpha);
bool DrawBig(ImDrawList* dl, int id, ImVec2 c, float r, float alpha); // (iconos-hd)
bool Has(int id);
} // namespace ShopIcons

// ---- (trucos7) F2 > "Armas y objetos": a shop over the game's own item tables (ItemDb: the user's game files, read once
// on a worker thread the first time F2 opens, a few MB parsed: then instant). Bought items are added by the game's own
// routines on the game thread (Loader::DemonsSoulsInventory: AddItem 0x432be0 / LearnMagic 0x44c590). Weapons and
// shields: only the base (+0, normal path: originEquipWep -1 and no "+n"); the game's blacksmith upgrades them with the
// forge stones, which are sold under Objetos. Price: the param's basicPrice (souls) or free.
namespace Shop {
namespace Inv = Loader::DemonsSoulsInventory;
std::atomic<int> g_state {0}; // 0 not loaded, 1 loading, 2 ready
void StartLoad() {
	int expected = 0;
	if (g_state.compare_exchange_strong(expected, 1))
		std::thread([] {
			ItemDb::Load();
			g_state = 2;
			Bump();
		}).detach();
}
struct Cat {
	const char* name;
	Icons::Kind icon;
	int         db;
	Inv::Kind   kind;
};
const Cat kCats[] = {{"Armas", Icons::Sword, 0, Inv::Kind::Weapon},     {"Escudos", Icons::Shield, 0, Inv::Kind::Weapon},
                     {"Armaduras", Icons::Helm, 1, Inv::Kind::Armor},    {"Anillos", Icons::Ring, 2, Inv::Kind::Ring},
                     {"Objetos", Icons::Potion, 3, Inv::Kind::Goods},    {"Magias", Icons::Soul, 4, Inv::Kind::Magic},
                     {"Milagros", Icons::Sun, 4, Inv::Kind::Magic}};
constexpr int kCatCount = 7;
bool Shown(int c, const ItemDb::Item& it) {
	if (it.dup) return false; // (trucos10)
	for (unsigned char ch: it.name)
		if (ch >= 0xe3) return false; // (rows the game never translated: Japanese placeholders, debug items)
	if (it.name.rfind("Catalizador universal", 0) == 0) return false;
	if (c <= 1) return it.origin < 0 && it.level == 0 && ((c == 1) == (it.sub == 12)); // weaponCategory 12: shields
	if (c == 2 && it.sub > 3) return false; // (trucos18) protectorCategory > 3: hairstyles / face rows, not armour
	if (c == 5) return it.sub == 0; // ezStateBehaviorType 0: sorcery
	if (c == 6) return it.sub == 1; // 1: miracle
	return true;
}
int         g_cat = 0, g_sel = -1, g_qty = 1;
// (trucos19) Pad / keyboard navigation in three steps: 0 the categories, 1 the item list, 2 the purchase (quantity,
// Comprar). X / Enter goes one step in (the focus jumps there), Circle / Esc one step back (Back(), from the window
// thread); the columns not at the current step take no navigation, so the focus never wanders to them. The mouse can
// still click anywhere (a click sets the step it belongs to).
std::atomic<int>  g_level {0};
std::atomic<bool> g_refocus {true}; // the step's own item gets the focus at the next frame
// Circle / Esc on the shop: true when it went back one step inside the shop (false: leave the page).
bool Back() {
	int l = g_level.load();
	if (l <= 0) return false;
	g_level   = l - 1;
	g_refocus = true;
	return true;
}
bool        g_free = false;
char        g_find[64] = "";
std::string g_note;
float       g_cat_anim[kCatCount] {};
struct Pending {
	bool        active = false;
	int         seq = 0, id = 0, cost = 0;
	std::string name;
};
Pending g_pending;

void BigIcon(ImDrawList* dl, Icons::Kind k, ImVec2 c, float size, float hot, float al) {
	DsStyle::Medallion(dl, StyleIcon(k), c, std::round(size * 0.46f), al, hot, Souls::Time(), hot); // (trucos11)
}

bool Draw(float width, float height, float unit) {
	using namespace F1Pages;
	StartLoad();
	ShopIcons::Tick(); // (trucos14) the real pictures (worker thread, cached)
	// (trucos8) A purchase waiting for the game's answer: charge on "added", nothing on "refused".
	if (g_pending.active && Inv::ResultSeq() != g_pending.seq) {
		int       rid = 0;
		const int r   = Inv::Last(&rid);
		g_pending.active = false;
		if (r > 0 && rid == g_pending.id) {
			int souls = 0;
			if (g_pending.cost > 0 && Cheats::ReadSouls(&souls)) Inv::SetSouls(std::max(0, souls - g_pending.cost));
			Port::MarkCheatsUsed("Armas y objetos"); // (trucos-online)
			g_note = std::string(I18n::T("Hecho: ")) + g_pending.name;
		} else {
			g_note = I18n::T("El juego no lo acept\xC3\xB3: no se han cobrado almas.");
		}
		std::printf("[trucos] tienda: %s -> %s\n", g_pending.name.c_str(), r > 0 && rid == g_pending.id ? "added, cobrado" : "refused, sin cobrar");
	}
	Begin("Armas y objetos", width, height, unit);
	Bump();
	const float al = ImGui::GetStyle().Alpha;
	const float list_h = ImGui::GetContentRegionAvail().y - std::round(84 * unit);
	// Left: the categories, big engraved icons; the chosen one glows.
	const int level = g_level.load();
	const bool refocus = g_refocus.exchange(false);
	ImGui::PushItemFlag(ImGuiItemFlags_NoNav, level != 0); // (trucos19) only the current step's column navigates
	ImGui::BeginChild("##cats", {std::round(320 * unit), list_h}, ImGuiChildFlags_NavFlattened);
	for (int c = 0; c < kCatCount; c++) {
		ImGui::PushID(c);
		const ImVec2 at  = ImGui::GetCursorScreenPos();
		const float  row = std::round(72 * unit);
		{
			const ImVec2 pa = ImGui::GetCursorScreenPos();
			DsStyle::CandlePlate(ImGui::GetWindowDrawList(), pa, {pa.x + ImGui::GetContentRegionAvail().x, pa.y + row - 6 * unit}, g_cat_anim[c] * (g_cat == c ? 0.0f : 1.0f),
			                     g_cat == c ? 1.0f : 0.0f, al, Souls::Time() + c, unit); // (trucos11)
		}
		ImGui::PushStyleColor(ImGuiCol_Header, ImVec4 {0, 0, 0, 0});
		ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4 {0, 0, 0, 0});
		ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4 {0, 0, 0, 0});
		if (refocus && level == 0 && c == g_cat) ImGui::SetKeyboardFocusHere();
		const bool picked = ImGui::Selectable("##c", g_cat == c, 0, {0, row});
		ImGui::PopStyleColor(3);
		if (picked) {
			if (g_cat != c) g_sel = -1, g_find[0] = 0;
			g_cat     = c;
			g_level   = 1; // (trucos19) into the list
			g_refocus = true;
		}
		const bool hot = ImGui::IsItemHovered() || ImGui::IsItemFocused() || g_cat == c;
		g_cat_anim[c] += ((hot ? 1.0f : 0.0f) - g_cat_anim[c]) * 0.18f;
		ImDrawList* dl = ImGui::GetWindowDrawList();
		BigIcon(dl, kCats[c].icon, {std::round(at.x + row * 0.55f), std::round(at.y + row * 0.5f)}, row * 0.86f, g_cat_anim[c], al);
		ImFont* f = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
		const float fs = std::round(30 * unit);
		DsStyle::SmallCaps(dl, f, fs, {std::round(at.x + row * 1.2f), std::round(at.y + (row - fs) * 0.5f)}, I18n::T(kCats[c].name),
		                   g_cat == c ? DsStyle::C(DsStyle::kBone, al) : DsStyle::C(DsStyle::kGold, al)); // (trucos11)
		ImGui::PopID();
	}
	ImGui::EndChild();
	ImGui::PopItemFlag();
	ImGui::SameLine();
	// Middle: the list.
	const float mid_w = std::round((ImGui::GetContentRegionAvail().x - 16 * unit) * 0.5f);
	ImGui::PushItemFlag(ImGuiItemFlags_NoNav, level != 1);
	ImGui::BeginChild("##shoplist", {mid_w, list_h}, ImGuiChildFlags_NavFlattened);
	ImGui::SetNextItemWidth(-std::round(130 * unit)); // (online6: room for "Pegar")
	ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true); // (the search box: mouse / click only, the pad never lands in it)
	ImGui::InputTextWithHint("##find", I18n::T("Buscar"), g_find, sizeof(g_find));
	bool list_focus_done = false;
	Anim::Highlight(unit);
	FieldTools("##findtools", g_find, sizeof(g_find)); // (online6) right click / Pegar
	ImGui::PopItemFlag();
	std::string f = g_find;
	for (auto& ch: f) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	const ItemDb::Item* chosen = nullptr;
	if (g_state.load() != 2) {
		ImGui::TextColored(kDim, "%s", I18n::T("Leyendo las tablas del juego..."));
	} else if (!ItemDb::g_db.ok) {
		ImGui::TextColored(kWarn, "%s", I18n::T("No se pudieron leer los ficheros del juego (gameparam.parambnd / item.msgbnd)."));
	} else {
		const auto& cat = kCats[g_cat];
		const float row = std::round(52 * unit);
		// (11-22) Armour by set: the pieces of one set (the same equipModelId: one parts folder per set) together, in the
		// order helmet, chest, arms, legs (protectorCategory 0-3); the sets in the game's own order (their first row).
		// The other categories keep the game's order.
		static std::vector<const ItemDb::Item*> armour_sorted;
		static size_t                           armour_n = static_cast<size_t>(-1);
		const auto&                             all      = ItemDb::g_db.cat[cat.db];
		std::vector<const ItemDb::Item*>        plain;
		const std::vector<const ItemDb::Item*>* list = &plain;
		if (cat.db == 1) {
			if (armour_n != all.size()) {
				armour_n = all.size();
				armour_sorted.clear();
				std::map<int, size_t> first; // set (model) -> its first row
				for (size_t i = 0; i < all.size(); i++) first.emplace(all[i].model != 0 ? all[i].model : -1 - static_cast<int>(i), i);
				for (const auto& it: all) armour_sorted.push_back(&it);
				std::stable_sort(armour_sorted.begin(), armour_sorted.end(), [&](const ItemDb::Item* x, const ItemDb::Item* y) {
					const size_t fx = first[x->model != 0 ? x->model : -1 - static_cast<int>(x - all.data())];
					const size_t fy = first[y->model != 0 ? y->model : -1 - static_cast<int>(y - all.data())];
					if (fx != fy) return fx < fy;
					return x->sub < y->sub;
				});
			}
			list = &armour_sorted;
		} else {
			plain.reserve(all.size());
			for (const auto& it: all) plain.push_back(&it);
		}
		for (const ItemDb::Item* itp: *list) {
			const auto& it = *itp;
			if (!Shown(g_cat, it)) continue;
			if (!f.empty()) {
				std::string low = it.name;
				for (auto& ch: low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
				if (low.find(f) == std::string::npos) continue;
			}
			if (g_sel == it.id) chosen = &it;
			ImGui::PushID(it.id);
			const ImVec2 at = ImGui::GetCursorScreenPos();
			if (refocus && level == 1 && !list_focus_done && (g_sel == it.id || g_sel < 0)) {
				ImGui::SetKeyboardFocusHere(); // (the chosen item, else the first)
				ImGui::SetScrollHereY(0.5f);
				list_focus_done = true;
			}
			if (ImGui::Selectable("##i", g_sel == it.id, 0, {0, row})) {
				g_sel     = it.id, g_qty = 1, g_note.clear();
				g_level   = 2; // (trucos19) to the purchase
				g_refocus = true;
			}
			if (ImGui::IsItemFocused() && g_sel != it.id && ImGui::GetIO().NavActive) g_sel = it.id, chosen = &it;
			Anim::Highlight(unit);
			ImDrawList*  dl = ImGui::GetWindowDrawList();
			const ImVec2 c {std::round(at.x + row * 0.5f), std::round(at.y + row * 0.5f)};
			if (g_sel == it.id || ImGui::IsItemHovered() || ImGui::IsItemFocused())
				DsStyle::CandlePlate(dl, at, {at.x + ImGui::GetContentRegionAvail().x, at.y + row}, 1.0f, 0.0f, al * 0.9f, Souls::Time(), unit); // (trucos11)
			if (!ShopIcons::Draw(dl, it.id, c, std::round(row * 0.5f), al)) // (trucos14) the game's own picture when there is one
				DsStyle::Medallion(dl, StyleIcon(cat.icon), c, std::round(row * 0.4f), al, g_sel == it.id ? 1.0f : 0.0f, Souls::Time());
			const float ty = std::round(at.y + (row - ImGui::GetFontSize()) * 0.5f);
			dl->AddText({std::round(at.x + row + 8 * unit), ty}, ImGui::GetColorU32(ImGuiCol_Text), it.name.c_str());
			char price[32];
			std::snprintf(price, sizeof(price), "%d", it.price);
			const char* pt = (g_free || it.price == 0) ? I18n::T("Gratis") : price;
			const float pw = ImGui::CalcTextSize(pt).x;
			dl->AddText({std::round(at.x + ImGui::GetContentRegionAvail().x - pw - 8 * unit), ty}, Souls::C(Souls::kSoul, al * 0.85f), pt);
			ImGui::PopID();
		}
	}
	ImGui::EndChild();
	ImGui::PopItemFlag();
	ImGui::SameLine();
	// Right: the chosen item, its lore text and the purchase.
	ImGui::PushItemFlag(ImGuiItemFlags_NoNav, level != 2);
	ImGui::BeginChild("##detail", {0, list_h}, ImGuiChildFlags_NavFlattened);
	if (chosen != nullptr) {
		ImDrawList*  dl = ImGui::GetWindowDrawList();
		const ImVec2 at = ImGui::GetCursorScreenPos();
		const float  big = std::round(150 * unit);
		{
			const ImVec2 sc {std::round(at.x + big * 0.6f), std::round(at.y + big * 0.55f)};
			DsStyle::Showcase(dl, StyleIcon(kCats[g_cat].icon), sc, std::round(big * 0.46f), al, Souls::Time()); // (trucos12)
			// (trucos14) The game's picture over the showcase's recess, when it has one.
			if (ShopIcons::Has(chosen->id)) {
				dl->AddCircleFilled(sc, std::round(big * 0.46f * 0.84f), DsStyle::C(DsStyle::kInk, al), 64); // (cover the generic icon)
				ShopIcons::DrawBig(dl, chosen->id, sc, std::round(big * 0.5f), al); // (iconos-hd) full resolution
			}
		}
		ImGui::Dummy({big * 1.2f, big * 1.1f});
		ImFont* tf = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
		const float ts = std::round(36 * unit);
		const ImVec2 tp = ImGui::GetCursorScreenPos();
		Souls::Etched(dl, tf, ts, {std::round(tp.x), std::round(tp.y)}, chosen->name.c_str(), al);
		ImGui::Dummy({1, ts + 6 * unit});
		Souls::Divider(dl, tp.x, tp.x + ImGui::GetContentRegionAvail().x, std::round(ImGui::GetCursorScreenPos().y), unit, al);
		ImGui::Dummy({1, 12 * unit});
		ImGui::PushTextWrapPos(0.0f);
		ImGui::TextColored(ImVec4 {0.82f, 0.80f, 0.74f, 1.0f}, "%s", chosen->desc.c_str());
		ImGui::PopTextWrapPos();
		ImGui::Spacing();
		const bool stack = kCats[g_cat].kind == Inv::Kind::Goods;
		if (stack) {
			ImGui::SetNextItemWidth(std::round(260 * unit));
			ImGui::SliderInt(I18n::T("Cantidad"), &g_qty, 1, 99); // (left / right on the pad change it)
			Anim::Highlight(unit);
		} else {
			g_qty = 1;
		}
		const int cost = g_free ? 0 : chosen->price * g_qty;
		int       souls = 0;
		const bool have = Cheats::ReadSouls(&souls);
		if (cost > 0) ImGui::TextColored(kGold, "%s %d   (%s %d)", I18n::T("Precio:"), cost, I18n::T("tienes"), have ? souls : 0);
		else ImGui::TextColored(kGold, "%s", I18n::T("Gratis"));
		const bool ready = Inv::Ready() && have;
		ImGui::BeginDisabled(!ready);
		if (refocus && level == 2) ImGui::SetKeyboardFocusHere(); // (trucos19) the focus lands on Comprar
		if (Anim::Button(I18n::T(kCats[g_cat].kind == Inv::Kind::Magic ? "Aprender" : "Comprar"), {std::round(300 * unit), std::round(56 * unit)}, unit)) {
			if (cost > souls) g_note = I18n::T("No tienes almas suficientes.");
			else if (g_pending.active) g_note = I18n::T("Espera: el juego est\xC3\xA1 a\xC3\xB1" "adiendo el anterior.");
			else if (Inv::Add(kCats[g_cat].kind, chosen->id, g_qty)) {
				// (trucos8) Charged only once the game's routine said "added" for this id (Inv::Result, below).
				g_pending = {true, Inv::ResultSeq(), chosen->id, cost, chosen->name};
				g_note    = I18n::T("Enviado al juego...");
				std::printf("[trucos] tienda: %s (id %d) x%d pedido, %d almas al confirmar\n", chosen->name.c_str(), chosen->id, g_qty, cost);
			} else g_note = I18n::T("No se pudo: entra en la partida primero.");
		}
		ImGui::EndDisabled();
		if (!ready) ImGui::TextColored(kDim, "%s", I18n::T("Disponible con la partida cargada."));
		if (!g_note.empty()) ImGui::TextColored(kDim, "%s", g_note.c_str());
	} else {
		ImGui::TextColored(kDim, "%s", I18n::T("Elige un objeto de la lista."));
	}
	ImGui::EndChild();
	ImGui::PopItemFlag();
	ImGui::Separator();
	ImGui::PushItemFlag(ImGuiItemFlags_NoNav, level != 0); // (the footer: only from the categories step)
	bool close = Anim::Button(I18n::T("Volver"), {std::round(240 * unit), std::round(54 * unit)}, unit);
	ImGui::SameLine();
	ImGui::Checkbox(I18n::T("Gratis (sin gastar almas)"), &g_free);
	Anim::Highlight(unit);
	ImGui::SameLine();
	ImGui::TextDisabled("%s", I18n::T("Círculo / Esc: volver"));
	ImGui::PopItemFlag();
	if (close) g_level = 0, g_refocus = true; // (trucos19) the next visit starts at the categories
	PageZoom();
	ImGui::End();
	ImGui::PopStyleColor(g_style_colors);
	ImGui::PopStyleVar(g_style_vars);
	g_style_colors = g_style_vars = 0;
	ImGui::PopFont();
	return !close;
}
} // namespace Shop
// ---- (trucos14) The real item pictures, from the user's own game files at run time (nothing copied into the port):
//   <game>/ui/textures/_ps5/global.ctal: 1 byte n, n bytes of path, u32 count, then count x {u32 key, 4 x (u, v) floats,
//     u32 page} (40 bytes); key = FNV-1a 32 of the UPPER-CASE texture name (checked: "TEX_INVICON_PLACEHOLDER",
//     "TEX_INVICON_PURESTONE", "TEX_INVICON_WP_A_99999_REPAIRALL" ... all found). Corners may be listed rotated: the
//     rectangle is their min / max.
//   <game>/ui/textures/_ps5/_ps5/global_pages.ctxr: 77-byte header, then 5 pages of 4096 x 4096 RGBA8 (premultiplied),
//     64 KiB "standard" tiles of 128 x 128 (the emulator's gpu_tiler_standard64 for 4-byte elements); only the tiles an
//     icon touches are read (by offset), never the 320 MiB.
//   Names: "Tex_InvIcon_<asset>" (the eboot's format at 0x245bf62, built by 0xbceb0). Weapons and shields: the asset is
//     the folder <game>/weapons/wp_<x>_<equipModelId %04d>_<name> (EquipParamWeapon equipModelId +2: Daga 100 ->
//     wp_a_0100_dagger), listed at run time. Armour: <game>/parts/<hd|bd|am|lg>_<m|f>_<equipModelId %04d>... (Protector
//     protectorCategory +8, equipModelId +2), when such a folder exists. Rings, goods and spells: no mapping known yet
//     (generic medallion).
// The shop's pictures go into ONE ImGui texture (2048 x 2048, 256 cells of 128 x 128; the overlay's descriptor pool is
// small), built on a worker thread the first time the shop opens and cached in <exe>/_TempData/tienda-iconos.bin.
// (iconos-hd) The game's pictures are 260 x 300 (483 of the atlas' 1229 entries, every Tex_InvIcon_ checked): the 128 px
// cells (120 px pictures) suit the list's 52 px rows, but the detail's picture is drawn at 150 * unit px (300 px at
// 4K) and was that 120 px picture upscaled (blurry). The detail now gets its own slot under the cells (kBig x kBig,
// rows kHeight..kTexH of the same texture: no new descriptor), filled on a worker thread from global_pages.ctxr for
// the chosen item at the size it is drawn (1:1 up to the game's 260 x 300), uploaded with a WantUpdates of that slot.
// The cells and their cache file are unchanged (the cache stays valid).
namespace ShopIcons {
constexpr int kCell = 128, kSide = 2048, kHeight = 3072, kPerRow = kSide / kCell, kMax = kPerRow * (kHeight / kCell), kPage = 4096; // (trucos15: 384 cells)
constexpr int kBig = 320, kTexH = kHeight + kBig; // (iconos-hd) the detail's slot: x 0..kBig, y kHeight..kTexH
struct Rect {
	int x0, y0, x1, y1, page;
};
struct State {
	std::atomic<int>                 phase {0}; // 0 not started, 1 building, 2 pixels ready (texture not made), 3 texture made, -1 failed
	std::vector<uint8_t>             pixels;    // kSide x kHeight RGBA (straight alpha)
	std::map<int, int>               cell;      // item id -> cell (weapons / shields / armour)
	ImTextureData*                   tex = nullptr;
	ImGuiContext*                    ctx = nullptr;
	// (iconos-hd) The detail's picture at full resolution.
	std::map<int, Rect>   src;       // item id -> its rectangle in global_pages.ctxr (written before phase 2, then read-only)
	std::filesystem::path ctxr_path; // (idem)
	std::atomic<bool>     big_busy {false};
	std::mutex            big_m;     // guards the big_done_* fields (the worker -> the UI thread)
	bool                  big_done = false, big_done_ok = false;
	int                   big_done_id = -1, big_done_s = 0;
	std::vector<uint8_t>  big_px;    // kBig x kBig RGBA, the worker's result
	int                   big_id = -1, big_s = 0; // in the texture's slot now (UI thread only)
	bool                  big_ok = false;
};
State g;

uint32_t Fnv1aUpper(const std::string& s) {
	uint32_t h = 0x811c9dc5u;
	for (unsigned char c: s) h = (h ^ static_cast<uint8_t>(std::toupper(c))) * 0x01000193u;
	return h;
}
uint32_t Swizzle64(uint32_t x, uint32_t y) { // standard64, 4-byte elements (gpu_tiler_standard64.inc)
	return ((x << 2) & 0x00cu) ^ ((x << 5) & 0x080u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u) ^ ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^
	       ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
}
std::map<uint32_t, Rect> ReadCtal(const std::filesystem::path& p) {
	std::map<uint32_t, Rect> out;
	ItemDb::Bytes            b;
	if (!ItemDb::ReadFile(p, &b) || b.empty()) return out;
	const size_t n = static_cast<uint8_t>(b[0]);
	uint32_t     count = 0;
	if (!ItemDb::Get(b, 1 + n, &count) || count > 100000) return out;
	for (uint32_t i = 0; i < count; i++) {
		const size_t o = 1 + n + 4 + static_cast<size_t>(i) * 40;
		uint32_t     key = 0, page = 0;
		float        uv[8] {};
		if (!ItemDb::Get(b, o, &key) || !ItemDb::Get(b, o + 4, &uv) || !ItemDb::Get(b, o + 36, &page)) break;
		float u0 = 1, v0 = 1, u1 = 0, v1 = 0;
		for (int k = 0; k < 4; k++) u0 = std::min(u0, uv[2 * k]), u1 = std::max(u1, uv[2 * k]), v0 = std::min(v0, uv[2 * k + 1]), v1 = std::max(v1, uv[2 * k + 1]);
		out[key] = {static_cast<int>(std::lround(u0 * kPage)), static_cast<int>(std::lround(v0 * kPage)), static_cast<int>(std::lround(u1 * kPage)),
		            static_cast<int>(std::lround(v1 * kPage)), static_cast<int>(page)};
	}
	return out;
}
// The pixels of a rectangle of global_pages.ctxr (straight alpha, w x h), the tiles it touches read by offset.
bool ReadSrc(std::ifstream& in, const Rect& r, std::map<uint64_t, std::string>& tiles, std::vector<uint8_t>* src) {
	const int w = r.x1 - r.x0, h = r.y1 - r.y0;
	if (w <= 0 || h <= 0 || w > 1024 || h > 1024 || r.page < 0 || r.page > 15) return false;
	src->assign(static_cast<size_t>(w) * h * 4, 0);
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++) {
			const uint32_t gx = r.x0 + x, gy = r.y0 + y;
			const uint64_t tile = static_cast<uint64_t>(r.page) * 1024u + (gy / 128) * (kPage / 128) + gx / 128;
			auto           it   = tiles.find(tile);
			if (it == tiles.end()) {
				std::string t(65536, '\0');
				in.seekg(static_cast<std::streamoff>(77 + static_cast<uint64_t>(r.page) * kPage * kPage * 4 + ((gy / 128) * (kPage / 128) + gx / 128) * 65536ull));
				in.read(t.data(), 65536);
				if (!in) return false;
				it = tiles.emplace(tile, std::move(t)).first;
			}
			const uint8_t* p = reinterpret_cast<const uint8_t*>(it->second.data()) + Swizzle64(gx % 128, gy % 128);
			uint8_t*       d = &(*src)[(static_cast<size_t>(y) * w + x) * 4];
			const int      a = p[3];
			for (int c = 0; c < 3; c++) d[c] = a > 0 ? static_cast<uint8_t>(std::min(255, p[c] * 255 / a)) : 0;
			d[3] = static_cast<uint8_t>(a);
		}
	return true;
}
// src (w x h) scaled by s (<= 1; box filter, alpha-weighted colour so the edges keep no dark fringe) into dst (a
// stride-pixel-wide RGBA buffer) with its top-left corner at (ox, oy). s = 1 is an exact copy.
void Scale(const std::vector<uint8_t>& src, int w, int h, float s, uint8_t* dst, int stride, int ox, int oy) {
	const int dw = std::max(1, static_cast<int>(w * s)), dh = std::max(1, static_cast<int>(h * s));
	for (int y = 0; y < dh; y++)
		for (int x = 0; x < dw; x++) {
			const int sx0 = static_cast<int>(x / s), sx1 = std::max(sx0 + 1, static_cast<int>((x + 1) / s)), sy0 = static_cast<int>(y / s), sy1 = std::max(sy0 + 1, static_cast<int>((y + 1) / s));
			float     acc[4] {};
			int       n = 0;
			for (int yy = sy0; yy < std::min(sy1, h); yy++)
				for (int xx = sx0; xx < std::min(sx1, w); xx++, n++) {
					const uint8_t* q = &src[(static_cast<size_t>(yy) * w + xx) * 4];
					for (int c = 0; c < 3; c++) acc[c] += q[c] * (q[3] / 255.0f);
					acc[3] += q[3];
				}
			if (n == 0) continue;
			uint8_t*    d = &dst[(static_cast<size_t>(oy + y) * stride + ox + x) * 4];
			const float a = acc[3] / n;
			for (int c = 0; c < 3; c++) d[c] = a > 0.5f ? static_cast<uint8_t>(std::min(255.0f, acc[c] / n * 255.0f / a)) : 0;
			d[3] = static_cast<uint8_t>(a);
		}
}
// One icon into its cell: read, scaled (box) to fit 120 px, centred (the list's pictures; unchanged by iconos-hd).
bool Blit(std::ifstream& in, const Rect& r, int cell, std::map<uint64_t, std::string>& tiles) {
	std::vector<uint8_t> src;
	if (!ReadSrc(in, r, tiles, &src)) return false;
	const int   w = r.x1 - r.x0, h = r.y1 - r.y0;
	const float s  = std::min(120.0f / w, 120.0f / h);
	const int   dw = std::max(1, static_cast<int>(w * s)), dh = std::max(1, static_cast<int>(h * s));
	Scale(src, w, h, s, g.pixels.data(), kSide, (cell % kPerRow) * kCell + (kCell - dw) / 2, (cell / kPerRow) * kCell + (kCell - dh) / 2);
	return true;
}
void Build() {
	const auto t0   = std::chrono::steady_clock::now();
	const auto game = std::filesystem::u8path(g_rt.launched.game);
	const auto ctal = game / "ui" / "textures" / "_ps5" / "global.ctal";
	const auto ctxr = game / "ui" / "textures" / "_ps5" / "_ps5" / "global_pages.ctxr";
	std::error_code ec;
	const auto      ctxr_size = std::filesystem::file_size(ctxr, ec);
	if (ec) {
		std::printf("[trucos] iconos: no se pudo abrir %s\n", ctxr.u8string().c_str());
		g.phase = -1;
		return;
	}
	const auto atlas = ReadCtal(ctal);
	// The asset folders the game has (names only).
	std::vector<std::string> weapons, parts;
	const auto list_dirs = [&](const std::filesystem::path& dir, std::vector<std::string>* out) {
		for (const auto& e: std::filesystem::directory_iterator(dir, ec))
			if (e.is_directory(ec)) {
				const auto u = e.path().filename().u8string(); // (char8_t in C++20)
				out->emplace_back(u.begin(), u.end());
			}
	};
	list_dirs(game / "weapons", &weapons);
	list_dirs(game / "parts", &parts);
	// id -> texture name.
	std::vector<std::pair<int, std::string>> want;
	char                                     pre[32];
	for (const auto& it: ItemDb::g_db.cat[0]) {
		if (it.dup) continue;
		// (trucos17) The row's own suffix: wp_<letter>_<model>_<bpPreviewSuffix> (98 of the shop's 99 in 01.007.000).
		if (!it.bp.empty()) {
			bool hit = false;
			for (char c = 'a'; c <= 'z' && !hit; c++) {
				char n[96];
				std::snprintf(n, sizeof(n), "Tex_InvIcon_wp_%c_%04d_%s", c, it.model, it.bp.c_str());
				if (atlas.count(Fnv1aUpper(n))) want.emplace_back(it.id, n), hit = true;
			}
			if (hit) continue;
		}
		std::snprintf(pre, sizeof(pre), "_%04d_", it.model);
		for (const auto& w: weapons)
			if (w.size() > 9 && w.compare(0, 3, "wp_") == 0 && w.compare(4, 6, pre) == 0 && atlas.count(Fnv1aUpper("Tex_InvIcon_" + w))) {
				want.emplace_back(it.id, "Tex_InvIcon_" + w);
				break;
			}
	}
	static const char* kPart[] = {"hd", "bd", "am", "lg"};
	for (const auto& it: ItemDb::g_db.cat[1]) {
		if (it.dup || it.sub < 0 || it.sub > 3) continue;
		{
			// (trucos17) The row's suffixes first: <part>_<m|f|a>_<model>_<bpPreviewSuffix(_Female)>.
			std::string hit;
			for (const std::string* sfx: {&it.bp, &it.bp2})
				for (const char* gch: {"m", "f", "a"})
					if (hit.empty() && !sfx->empty()) {
						char n[96];
						std::snprintf(n, sizeof(n), "Tex_InvIcon_%s_%s_%04d_%s", kPart[it.sub], gch, it.model, sfx->c_str());
						if (atlas.count(Fnv1aUpper(n))) hit = n;
					}
			if (!hit.empty()) {
				want.emplace_back(it.id, hit);
				continue;
			}
		}
		for (const char* gch: {"m", "f", "a"}) { // (trucos15: "a" = both, e.g. am_a_boletariaknight_blue; order hd/bd/am/lg confirmed: 71 of 112 named, chainmail's four parts extracted)
			std::snprintf(pre, sizeof(pre), "%s_%s_%04d", kPart[it.sub], gch, it.model);
			std::string found;
			if (atlas.count(Fnv1aUpper(std::string("Tex_InvIcon_") + pre))) found = pre;
			for (const auto& p: parts)
				if (found.empty() && p.compare(0, std::strlen(pre), pre) == 0 && atlas.count(Fnv1aUpper("Tex_InvIcon_" + p))) found = p;
			if (!found.empty()) {
				want.emplace_back(it.id, "Tex_InvIcon_" + found);
				break;
			}
		}
	}
	// (trucos15) Rings, goods and spells: the texture is the item's ENGLISH name in CamelCase ("Poison Cloud" ->
	// Tex_InvIcon_PoisonCloud, found in the atlas). The English names come from the user's msg/english/item.msgbnd (or
	// na_english / engus); a name is used only when its hash is in the atlas (no guess can pick a wrong picture).
	{
		ItemDb::Bytes en;
		for (const char* l: {"na_english", "english", "engus"}) // (trucos16: the game ships na_english)
			if (ItemDb::ReadFile(game / "cp11demonssouls" / "dvdroot" / "msg" / l / "item.msgbnd", &en)) break;
		const auto files = ItemDb::Bnd3(en);
		const auto fmg   = [&](const char* tail) {
            const ItemDb::Bytes* b = ItemDb::Find(files, tail);
            return b != nullptr ? ItemDb::Fmg(*b) : std::map<int, std::string> {};
		};
		const std::map<int, std::string> en_names[3] = {fmg("\x83\x41\x83\x4E\x83\x5A\x83\x54\x83\x8A\x96\xBC" ".fmg"), // rings
		                                                fmg("\x83\x41\x83\x43\x83\x65\x83\x80\x96\xBC" ".fmg"),         // goods
		                                                fmg("\x96\x82\x96\x40\x96\xBC" ".fmg")};                             // spells
		const auto camel = [](const std::string& n, bool keep_case) {
			std::string out;
			bool        up = true;
			for (unsigned char c: n) {
				if (std::isalnum(c)) {
					out += static_cast<char>(up ? std::toupper(c) : (keep_case ? c : std::tolower(c)));
					up = false;
				} else if (c != '\'') up = true;
			}
			return out;
		};
		int found[3] {}, by_key[2] {};
		for (int k = 0; k < 3; k++)
			for (const auto& it: ItemDb::g_db.cat[k + 2]) {
				if (it.dup) continue;
				// (trucos17) Rings and goods: the row's bpPreviewKey (126 of 126 goods, 26 of 26 rings in 01.007.000).
				if (!it.bp.empty() && atlas.count(Fnv1aUpper("Tex_InvIcon_" + it.bp))) {
					want.emplace_back(it.id, "Tex_InvIcon_" + it.bp);
					by_key[k < 2 ? k : 1]++;
					continue;
				}
				const auto e = en_names[k].find(it.id);
				if (e == en_names[k].end()) continue;
				// (trucos16) "Tex_InvIcon_" alone is in the atlas: never for an empty name.
				for (const std::string& cand: {camel(e->second, true), camel(e->second, false)})
					if (!cand.empty() && atlas.count(Fnv1aUpper("Tex_InvIcon_" + cand))) {
						want.emplace_back(it.id, "Tex_InvIcon_" + cand);
						found[k]++;
						break;
					}
			}
		std::printf("[trucos] iconos por clave del param: anillos %d, objetos %d; por nombre ingles (%s): anillos %d, objetos %d, magias/milagros %d\n",
		            by_key[0], by_key[1], en.empty() ? "sin item.msgbnd ingles" : "item.msgbnd ingles", found[0], found[1], found[2]);
	}
	if (want.size() > static_cast<size_t>(kMax)) want.resize(kMax);
	// (iconos-hd) Each item's rectangle, for the detail's full-resolution picture (with or without the cache).
	g.ctxr_path = ctxr;
	for (const auto& [id, n]: want)
		if (const auto r = atlas.find(Fnv1aUpper(n)); r != atlas.end()) g.src[id] = r->second;
	// The cache: valid for the same names and the same game files.
	const auto cache = std::filesystem::u8path(g_rt.exe_dir) / "_TempData" / "tienda-iconos.bin";
	// (trucos18) The cache key starts with the mapping's version: a new port build with another mapping rebuilds it by
	// itself (the id=name list below also changes it whenever a name changes).
	constexpr const char* kIconMapVersion = "iconmap-3";
	std::string key  = std::string(kIconMapVersion) + "|" + std::to_string(ctxr_size) + "|" + std::to_string(atlas.size());
	for (const auto& [id, n]: want) key += "|" + std::to_string(id) + "=" + n;
	g.pixels.assign(static_cast<size_t>(kSide) * kHeight * 4, 0);
	bool cached = false;
	{
		std::ifstream c(cache, std::ios::binary);
		uint32_t      kl = 0;
		if (c && c.read(reinterpret_cast<char*>(&kl), 4) && kl == key.size()) {
			std::string k(kl, '\0');
			if (c.read(k.data(), kl) && k == key && c.read(reinterpret_cast<char*>(g.pixels.data()), static_cast<std::streamsize>(g.pixels.size()))) cached = true;
		}
	}
	int ok = 0;
	if (!cached) {
		std::ifstream                   in(ctxr, std::ios::binary);
		std::map<uint64_t, std::string> tiles;
		for (size_t i = 0; i < want.size(); i++) {
			const auto r = atlas.find(Fnv1aUpper(want[i].second));
			if (r != atlas.end() && Blit(in, r->second, static_cast<int>(i), tiles)) ok++;
			if (tiles.size() > 256) tiles.clear(); // (at most 16 MiB of tiles held)
		}
		std::filesystem::create_directories(cache.parent_path(), ec);
		std::ofstream o(cache, std::ios::binary);
		const uint32_t kl = static_cast<uint32_t>(key.size());
		o.write(reinterpret_cast<const char*>(&kl), 4);
		o.write(key.data(), kl);
		o.write(reinterpret_cast<const char*>(g.pixels.data()), static_cast<std::streamsize>(g.pixels.size()));
	} else {
		ok = static_cast<int>(want.size());
	}
	for (size_t i = 0; i < want.size(); i++) g.cell[want[i].first] = static_cast<int>(i);
	std::printf("[trucos] iconos reales: %zu objetos con icono, %d dibujados, %s, %lld ms\n", want.size(), ok,
	            cached ? "desde la cache" : "leidos de global_pages.ctxr", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()));
	g.phase = 2;
	Bump();
}
// The UI thread: starts the build once (after the item tables), makes the texture when the pixels are ready.
void Tick() {
	int expected = 0;
	if (Shop::g_state.load() == 2 && ItemDb::g_db.ok && g.phase.compare_exchange_strong(expected, 1)) std::thread(Build).detach();
	if (g.phase.load() == 2) {
		g.tex = IM_NEW(ImTextureData)();
		g.tex->Create(ImTextureFormat_RGBA32, kSide, kTexH); // (iconos-hd) the cells + the detail's slot under them (zeroed by Create)
		std::memcpy(g.tex->GetPixels(), g.pixels.data(), g.pixels.size()); // (rows of kSide: the cells are the first kHeight rows)
		g.tex->UseColors = true;
		ImGui::RegisterUserTexture(g.tex);
		g.ctx = ImGui::GetCurrentContext();
		std::vector<uint8_t>().swap(g.pixels);
		g.phase = 3;
	}
}
bool Has(int id) {
	return g.phase.load() == 3 && g.tex != nullptr && g.ctx == ImGui::GetCurrentContext() && g.cell.count(id) != 0;
}
// Draws the item's real picture centred in [c - r, c + r]; false when it has none (the caller draws the medallion).
bool Draw(ImDrawList* dl, int id, ImVec2 c, float r, float alpha) {
	if (g.phase.load() != 3 || g.tex == nullptr || g.ctx != ImGui::GetCurrentContext()) return false;
	const auto it = g.cell.find(id);
	if (it == g.cell.end()) return false;
	const float  u0 = static_cast<float>((it->second % kPerRow) * kCell) / kSide, v0 = static_cast<float>((it->second / kPerRow) * kCell) / kTexH;
	const float  d  = static_cast<float>(kCell) / kSide, dv = static_cast<float>(kCell) / kTexH; // (iconos-hd: the texture is kTexH tall)
	dl->AddImage(g.tex->GetTexRef(), {std::round(c.x - r), std::round(c.y - r)}, {std::round(c.x + r), std::round(c.y + r)}, {u0, v0}, {u0 + d, v0 + dv},
	             IM_COL32(255, 255, 255, static_cast<int>(255 * alpha)));
	return true;
}
// (iconos-hd) The worker: item id's picture from global_pages.ctxr into an s x s square (top-left of a kBig x kBig
// buffer), framed as the cells (picture 120 / 128 of the square), scaled 1:1 at most (the game's own pixels at 4K).
void BuildBig(int id, int s) {
	std::vector<uint8_t> out(static_cast<size_t>(kBig) * kBig * 4, 0);
	bool                 ok = false;
	if (const auto r = g.src.find(id); r != g.src.end()) {
		std::ifstream                   in(g.ctxr_path, std::ios::binary);
		std::map<uint64_t, std::string> tiles;
		std::vector<uint8_t>            src;
		if (in && ReadSrc(in, r->second, tiles, &src)) {
			const int   w = r->second.x1 - r->second.x0, h = r->second.y1 - r->second.y0;
			const float fit = s * 120.0f / kCell;
			const float k   = std::min({1.0f, fit / w, fit / h});
			const int   dw = std::max(1, static_cast<int>(w * k)), dh = std::max(1, static_cast<int>(h * k));
			Scale(src, w, h, k, out.data(), kBig, std::max(0, (s - dw) / 2), std::max(0, (s - dh) / 2));
			ok = true;
		}
	}
	{
		std::lock_guard<std::mutex> lock(g.big_m);
		g.big_px      = std::move(out);
		g.big_done    = true;
		g.big_done_ok = ok;
		g.big_done_id = id;
		g.big_done_s  = s;
	}
	g.big_busy = false;
	Bump();
}
// (iconos-hd) The detail's picture: the full-resolution one once its worker is done, the cell's meanwhile (and when it
// fails). Same placement as Draw. False when the item has no picture.
bool DrawBig(ImDrawList* dl, int id, ImVec2 c, float r, float alpha) {
	if (!Has(id)) return false;
	// A finished picture into the texture's slot (the whole slot, so nothing of the previous one is left).
	{
		std::lock_guard<std::mutex> lock(g.big_m);
		if (g.big_done) {
			g.big_done = false;
			if (g.big_done_ok && g.tex->GetPixels() != nullptr && g.big_px.size() == static_cast<size_t>(kBig) * kBig * 4) {
				for (int y = 0; y < kBig; y++)
					std::memcpy(g.tex->GetPixelsAt(0, kHeight + y), &g.big_px[static_cast<size_t>(y) * kBig * 4], static_cast<size_t>(kBig) * 4);
				// (the backend uploads UpdateRect; while WantCreate is pending the whole texture goes up anyway)
				if (g.tex->Status == ImTextureStatus_OK || g.tex->Status == ImTextureStatus_WantUpdates) {
					g.tex->Updates.resize(0);
					g.tex->UpdateRect = {0, static_cast<unsigned short>(kHeight), static_cast<unsigned short>(kBig), static_cast<unsigned short>(kBig)};
					g.tex->Updates.push_back(g.tex->UpdateRect);
					g.tex->SetStatus(ImTextureStatus_WantUpdates);
				}
			}
			g.big_id = g.big_done_id, g.big_s = g.big_done_s, g.big_ok = g.big_done_ok;
		}
	}
	const int s = std::clamp(static_cast<int>(std::lround(2.0f * r)), 32, kBig);
	if ((g.big_id != id || g.big_s != s) && g.src.count(id) != 0) {
		bool expected = false;
		if (g.big_busy.compare_exchange_strong(expected, true)) std::thread(BuildBig, id, s).detach();
	}
	if (g.big_id != id || !g.big_ok || g.big_s <= 0) return Draw(dl, id, c, r, alpha);
	const float v0 = static_cast<float>(kHeight) / kTexH, du = static_cast<float>(g.big_s) / kSide, dv = static_cast<float>(g.big_s) / kTexH;
	dl->AddImage(g.tex->GetTexRef(), {std::round(c.x - r), std::round(c.y - r)}, {std::round(c.x + r), std::round(c.y + r)}, {0.0f, v0}, {du, v0 + dv},
	             IM_COL32(255, 255, 255, static_cast<int>(255 * alpha)));
	return true;
}
} // namespace ShopIcons

// (ds-pc-port8) In the game: F2's save and quit, a panel in the middle of the screen until the game closes.
// (ds-pc-port38) `closing`: the fade-out after F2 / Esc closed it (drawn disabled, no input taken).
// (bb-style-f1) The cursor is kept at every level of F2: each screen (0 the option bar, 1 Mando, 2 Teclado y ratón,
// 3 Trofeos) remembers the item that had the keyboard / pad focus; when a screen comes back (back from an option, from a
// sub-screen, or F2 opened again in this session) that item gets the focus again instead of ImGui's first item.
// ---- (online5) MULTIJUGADOR: F2 card + page, and the same settings in the launcher's Configurador. settings.ini
// [Multijugador] (multijugador / multijugador_servidor / _cuenta / _clave): any start (JUGAR.cmd too) plays online.
// The server (ds-online.cpp: "Demon's Souls.exe" --servidor): TCP 31413 accounts, TCP 31415 game + WebAPI, UDP 3758 relay.
// ---- (chat1) CHAT y AMIGOS: in "Conexión online" (launcher and F2). A worker asks the server (/kyty/chat) every 2.5 s
// for the messages waiting and who is connected; in the game a notice shows in a corner when a message arrives or a
// friend connects. Friends: settings.ini multijugador_amigos. Messages live on the server's memory only.
namespace ChatUi {
namespace On = Port::Online;
struct Line {
	std::string from, text;
	bool        mine = false;
};
std::mutex                                       g_cm;
std::map<std::string, std::vector<Line>>         g_convo;  // the other player -> lines
std::map<std::string, int>                       g_unread; // the other player -> messages not seen
std::vector<std::string>                         g_online;
std::set<std::string>                            g_was_online, g_friends;
bool                                             g_first_list = true, g_have_list = false;
struct ToastItem {
	std::string title, body;
	bool        friend_on = false; // a friend connected (a soul) or a message (a speech mark)
	double      at = 0.0;          // shown since (Anim::Now)
};
std::deque<ToastItem>                            g_toasts;
std::deque<std::pair<std::string, std::string>>  g_outbox; // to, text
std::string                                      g_error, g_me, g_open_with;
double                                           g_drawn_at = 0.0;
char                                             g_input[256] = "";
std::atomic<bool>                                g_started {false};

std::set<std::string> Split(const std::string& s) {
	std::set<std::string> out;
	std::string           cur;
	for (char c: s + ",") {
		if (c == ',' || c == ';' || c == ' ') {
			if (!cur.empty()) out.insert(cur);
			cur.clear();
		} else cur += c;
	}
	return out;
}
constexpr double kToastSeconds = 6.0;
// (chat2) A soft chime, made once in memory (two bell-like notes, a fifth apart), played without waiting.
void Chime(bool friend_on) {
#if defined(_WIN32)
	static std::vector<uint8_t> wav[2];
	auto& w = wav[friend_on ? 1 : 0];
	if (w.empty()) {
		const int   rate = 44100, n = rate * 7 / 10;
		const float f1 = friend_on ? 659.25f : 783.99f, f2 = friend_on ? 987.77f : 1046.5f; // E5-B5 / G5-C6
		std::vector<int16_t> pcm(n);
		for (int i = 0; i < n; i++) {
			const float t  = static_cast<float>(i) / rate;
			const float a1 = std::exp(-t * 6.0f), t2 = t - 0.12f, a2 = t2 > 0 ? std::exp(-t2 * 5.0f) : 0.0f;
			const float attack = std::min(1.0f, t * 200.0f);
			float v = attack * (a1 * (std::sin(6.2831853f * f1 * t) + 0.3f * std::sin(6.2831853f * f1 * 2 * t)) +
			                    a2 * (std::sin(6.2831853f * f2 * t2) + 0.25f * std::sin(6.2831853f * f2 * 2 * t2)));
			pcm[i] = static_cast<int16_t>(std::clamp(v * 0.18f, -1.0f, 1.0f) * 32767);
		}
		auto put32 = [&](uint32_t v) { for (int k = 0; k < 4; k++) w.push_back(static_cast<uint8_t>(v >> (8 * k))); };
		auto put16 = [&](uint16_t v) { w.push_back(static_cast<uint8_t>(v)); w.push_back(static_cast<uint8_t>(v >> 8)); };
		const uint32_t bytes = static_cast<uint32_t>(n * 2);
		for (char c: std::string("RIFF")) w.push_back(static_cast<uint8_t>(c));
		put32(36 + bytes);
		for (char c: std::string("WAVEfmt ")) w.push_back(static_cast<uint8_t>(c));
		put32(16); put16(1); put16(1); put32(rate); put32(rate * 2); put16(2); put16(16);
		for (char c: std::string("data")) w.push_back(static_cast<uint8_t>(c));
		put32(bytes);
		const auto* b = reinterpret_cast<const uint8_t*>(pcm.data());
		w.insert(w.end(), b, b + bytes);
	}
	PlaySoundW(reinterpret_cast<LPCWSTR>(w.data()), nullptr, SND_MEMORY | SND_ASYNC | SND_NODEFAULT);
#else
	(void)friend_on;
#endif
}
// (11-23) The chat history kept on disk, per account: chat-historial-<cuenta>.txt next to the exe (the last 200 lines
// of each conversation). Before: only in memory, gone when the game closed. One line: other TAB mine TAB from TAB text,
// with \\ \t \n escaped. (g_cm held)
std::filesystem::path HistoryPath() { return PortRoot() / std::filesystem::u8path("chat-historial-" + g_me + ".txt"); }
std::string HistEsc(const std::string& v) {
	std::string o;
	for (char c: v) o += c == '\\' ? std::string("\\\\") : c == '\t' ? std::string("\\t") : c == '\n' ? std::string("\\n") : c == '\r' ? std::string() : std::string(1, c);
	return o;
}
std::string HistUnesc(const std::string& v) {
	std::string o;
	for (size_t i = 0; i < v.size(); i++) {
		if (v[i] == '\\' && i + 1 < v.size()) {
			const char n = v[++i];
			o += n == 't' ? '\t' : n == 'n' ? '\n' : n;
		} else {
			o += v[i];
		}
	}
	return o;
}
void SaveHistory() {
	if (g_me.empty()) return;
	std::error_code ec;
	const auto      path = HistoryPath();
	auto            tmp  = path;
	tmp += ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return;
		for (const auto& [other, lines]: g_convo) {
			const size_t from = lines.size() > 200 ? lines.size() - 200 : 0;
			for (size_t i = from; i < lines.size(); i++)
				out << HistEsc(other) << '\t' << (lines[i].mine ? '1' : '0') << '\t' << HistEsc(lines[i].from) << '\t' << HistEsc(lines[i].text) << '\n';
		}
	}
	std::filesystem::rename(tmp, path, ec);
}
void LoadHistory() {
	if (g_me.empty()) return;
	std::ifstream in(HistoryPath(), std::ios::binary);
	std::string   line;
	while (std::getline(in, line)) {
		std::vector<std::string> f;
		size_t                   a = 0;
		for (size_t b; (b = line.find('\t', a)) != std::string::npos && f.size() < 3; a = b + 1) f.push_back(line.substr(a, b - a));
		f.push_back(line.substr(a));
		if (f.size() != 4) continue;
		g_convo[HistUnesc(f[0])].push_back({HistUnesc(f[2]), HistUnesc(f[3]), f[1] == "1"});
	}
}

void Toast(const std::string& title, const std::string& body, bool friend_on) { // (g_cm held)
	g_toasts.push_back({title, body.size() > 140 ? body.substr(0, 137) + "..." : body, friend_on, Anim::Now()});
	while (g_toasts.size() > 3) g_toasts.pop_front();
	Chime(friend_on);
}
// Starts the worker once (multijugador on and an account): from the launcher's / F2's card and every presented frame.
void Start() {
	// (chat2) KYTY_CHAT_DEMO=1: two sample notices in the game (20 s and 30 s after the first frame), to see them alone.
	static std::atomic<bool> demo_once {false};
	if (const char* d = std::getenv("KYTY_CHAT_DEMO"); d != nullptr && d[0] == '1' && !demo_once.exchange(true))
		std::thread([] {
			std::this_thread::sleep_for(std::chrono::seconds(20));
			{ std::lock_guard lock(g_cm); Toast("Amigo de prueba", I18n::T("Tu amigo se ha conectado"), true); }
			Bump();
			std::this_thread::sleep_for(std::chrono::seconds(10));
			{ std::lock_guard lock(g_cm); Toast(std::string(I18n::T("Mensaje de ")) + "Amigo de prueba", "Te espero en la Puerta de Boletaria, ¿te invoco?", false); }
			Bump();
		}).detach();
	if (g_started.load(std::memory_order_relaxed) || !g_saved.multiplayer || g_saved.online_name.empty() || g_saved.online_password.empty()) return;
	if (g_started.exchange(true)) return;
	const std::string name = g_saved.online_name, pass = g_saved.online_password;
	{
		std::lock_guard lock(g_cm);
		g_me      = name;
		g_friends = Split(g_saved.online_friends);
		if (g_convo.empty()) LoadHistory(); // (11-23) the conversations of the last times
	}
	std::thread([name, pass] {
		const std::string host = On::ServerHost() + ":" + std::to_string(On::kDefaultPort);
		for (;;) {
			std::string to, text;
			{
				std::lock_guard lock(g_cm);
				if (!g_outbox.empty()) {
					to   = g_outbox.front().first;
					text = g_outbox.front().second;
					g_outbox.pop_front();
				}
			}
			std::vector<On::ChatMessage> in;
			std::vector<std::string>     on;
			std::string                  err;
			const bool                   ok = On::Chat(host, name, pass, to, text, &in, &on, &err);
			bool                         more = false;
			{
				std::lock_guard lock(g_cm);
				if (!ok) {
					g_error = std::string(I18n::T("Chat: ")) + err;
					if (!to.empty() && err == "el servidor no responde") g_outbox.emplace_front(to, text); // (sent again)
					else if (!to.empty()) g_convo[to].push_back({"", std::string(I18n::T("(no enviado: ")) + err + ")", true}), SaveHistory();
				} else {
					g_error.clear();
					const bool shown = Anim::Now() - g_drawn_at < 0.5;
					for (const auto& m: in) {
						g_convo[m.from].push_back({m.from, m.text, false});
						if (!(shown && g_open_with == m.from)) g_unread[m.from]++;
						Toast(std::string(I18n::T("Mensaje de ")) + m.from, m.text, false);
					}
					if (!in.empty()) SaveHistory(); // (11-23)
					std::set<std::string> now(on.begin(), on.end());
					if (!g_first_list)
						for (const auto& n: now)
							if (n != name && !g_was_online.count(n) && g_friends.count(n)) Toast(n, I18n::T("Tu amigo se ha conectado"), true);
					g_first_list = false;
					g_was_online = std::move(now);
					g_online     = std::move(on);
					g_have_list  = true;
				}
				more = !g_outbox.empty();
			}
			Bump();
			std::this_thread::sleep_for(std::chrono::milliseconds(more ? 150 : 2500));
		}
	}).detach();
}
void SaveFriends() { // (UI thread; g_cm held by the caller)
	std::string list;
	for (const auto& f: g_friends) list += (list.empty() ? "" : ",") + f;
	Settings on_disk;
	if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_saved;
	for (Settings* s: {&on_disk, &g_saved, &g_edit}) s->online_friends = list;
	(void)SaveSettings(g_rt.ini_path, on_disk);
}
bool ToastActive() {
	std::lock_guard lock(g_cm);
	const double now = Anim::Now();
	while (!g_toasts.empty() && g_toasts.front().at + kToastSeconds < now) g_toasts.pop_front();
	return !g_toasts.empty();
}
// (chat2) The notices: a card that slides in from the right (top right), dark stone with a gold edge, a glowing soul
// (friend) or a speech mark (message), the name in gold and the text below; it eases out after 6 s.
void DrawToasts(float width, float height) {
	std::vector<ToastItem> list;
	{
		std::lock_guard lock(g_cm);
		list.assign(g_toasts.begin(), g_toasts.end());
	}
	if (list.empty()) return;
	const float  unit = std::max(std::min(width / 1920.0f, height / 1080.0f), 0.5f);
	ImFont*      f    = g_font_semi != nullptr ? g_font_semi : ImGui::GetFont();
	const float  ts   = std::round(25 * unit), bs = std::round(21 * unit), pad = std::round(18 * unit), w = std::round(460 * unit);
	const float  icon = std::round(64 * unit);
	auto*        dl   = ImGui::GetForegroundDrawList();
	float        y    = std::round(96 * unit);
	const double now  = Anim::Now();
	for (const auto& it: list) {
		const float age = static_cast<float>(now - it.at), left = static_cast<float>(kToastSeconds) - age;
		const float in  = Anim::EaseOutQuint(std::clamp(age / 0.45f, 0.0f, 1.0f));
		const float out = std::clamp(left / 0.4f, 0.0f, 1.0f);
		const float a   = std::min(in, out * out);
		const float dx  = (1.0f - in) * std::round(120 * unit) + (1.0f - out) * std::round(40 * unit);
		const float tw  = w - pad * 3 - icon;
		const ImVec2 t1 = f->CalcTextSizeA(ts, FLT_MAX, tw, it.title.c_str());
		const ImVec2 t2 = f->CalcTextSizeA(bs, FLT_MAX, tw, it.body.c_str());
		const float  h  = std::max(icon, t1.y + std::round(6 * unit) + t2.y) + pad * 2;
		const ImVec2 p0 {std::round(width - w - 28 * unit + dx), y}, p1 {p0.x + w, y + h};
		const float  r  = 12 * unit;
		for (int k = 3; k >= 1; k--) // soft shadow
			dl->AddRectFilled({p0.x - k * 3 * unit, p0.y - k * 2 * unit}, {p1.x + k * 3 * unit, p1.y + k * 5 * unit}, IM_COL32(0, 0, 0, static_cast<int>(40 * a)), r + k * 3 * unit);
		dl->AddRectFilledMultiColor(p0, p1, IM_COL32(30, 28, 34, static_cast<int>(240 * a)), IM_COL32(22, 21, 26, static_cast<int>(240 * a)),
		                            IM_COL32(14, 14, 18, static_cast<int>(240 * a)), IM_COL32(20, 19, 24, static_cast<int>(240 * a)));
		dl->AddRect(p0, p1, IM_COL32(222, 196, 140, static_cast<int>(210 * a)), r, 0, std::max(1.0f, 1.6f * unit));
		dl->AddRect({p0.x + 4 * unit, p0.y + 4 * unit}, {p1.x - 4 * unit, p1.y - 4 * unit}, IM_COL32(222, 196, 140, static_cast<int>(60 * a)), r * 0.7f, 0, 1.0f);
		// the icon: a breathing glow
		const ImVec2 c {p0.x + pad + icon * 0.5f, p0.y + h * 0.5f};
		const float  pl = 0.5f + 0.5f * std::sin(static_cast<float>(now) * 3.0f);
		const ImVec4 col = it.friend_on ? ImVec4 {0.62f, 0.95f, 0.70f, 1.0f} : ImVec4 {0.72f, 0.82f, 1.0f, 1.0f};
		for (int k = 3; k >= 1; k--)
			dl->AddCircleFilled(c, icon * (0.22f + 0.09f * k + 0.03f * pl), ImGui::GetColorU32(ImVec4 {col.x, col.y, col.z, a * 0.07f * k}), 32);
		Icons::Draw(dl, it.friend_on ? Icons::Soul : Icons::Ring, c, icon * 0.62f, ImGui::GetColorU32(ImVec4 {col.x, col.y, col.z, a}));
		// the text
		const float tx = p0.x + pad * 2 + icon, ty = p0.y + (h - (t1.y + std::round(6 * unit) + t2.y)) * 0.5f;
		dl->AddText(f, ts, {tx + 1, ty + 1}, IM_COL32(0, 0, 0, static_cast<int>(160 * a)), it.title.c_str(), nullptr, tw);
		dl->AddText(f, ts, {tx, ty}, IM_COL32(236, 206, 140, static_cast<int>(255 * a)), it.title.c_str(), nullptr, tw);
		dl->AddText(f, bs, {tx, ty + t1.y + std::round(6 * unit)}, IM_COL32(232, 226, 214, static_cast<int>(235 * a)), it.body.c_str(), nullptr, tw);
		y = p1.y + std::round(14 * unit);
	}
	Bump(); // (the animation)
}
// The players list with chat / friend buttons, and the open conversation. `fallback`: the server's /kyty/estado names.
void Draw(float unit, const std::vector<std::string>& fallback) {
	Start();
	std::lock_guard lock(g_cm);
	g_drawn_at = Anim::Now();
	if (!g_saved.multiplayer || g_saved.online_name.empty()) {
		ImGui::TextColored(kDim, "%s", I18n::T("Para chatear: activa el multijugador y crea tu cuenta (abajo)."));
		return;
	}
	const std::vector<std::string>& on = g_have_list ? g_online : fallback;
	std::set<std::string>           shown;
	ImGui::TextColored(kDimL, "%s", (std::string(I18n::T("Jugadores conectados: ")) + std::to_string(on.size())).c_str());
	auto row = [&](const std::string& name, bool online) {
		ImGui::PushID(name.c_str());
		const ImVec2 at = ImGui::GetCursorScreenPos();
		const float  r  = std::round(7 * unit), lh = ImGui::GetTextLineHeight();
		ImGui::GetWindowDrawList()->AddCircleFilled({at.x + r, at.y + lh * 0.5f + std::round(4 * unit)}, r,
		                                            online ? IM_COL32(80, 220, 110, 255) : IM_COL32(110, 110, 110, 255));
		ImGui::Dummy({r * 2 + std::round(8 * unit), lh});
		ImGui::SameLine();
		const bool me = name == g_me;
		std::string label = name + (me ? std::string(I18n::T(" (tú)")) : "") + (g_friends.count(name) ? std::string(I18n::T("  ★ amigo")) : "");
		if (int n = g_unread.count(name) ? g_unread[name] : 0; n > 0) label += "  (" + std::to_string(n) + ")";
		// (11-23) Online / offline written out; a friend who is online in green (name and state) to spot at a glance.
		label += online ? std::string(I18n::T("  - en línea")) : std::string(I18n::T("  - desconectado"));
		const bool friend_on = online && !me && g_friends.count(name) != 0;
		ImGui::TextColored(friend_on ? ImVec4 {0.35f, 0.90f, 0.45f, 1.0f} : online ? ImVec4 {0.93f, 0.91f, 0.88f, 1.0f} : kDim, "%s", label.c_str());
		if (!me) {
			const ImVec2 bs {std::round(150 * unit), 0};
			ImGui::SameLine(std::round(520 * unit));
			if (Anim::Button(g_open_with == name ? I18n::T("Cerrar chat") : I18n::T("Chatear"), bs, unit)) {
				g_open_with = g_open_with == name ? std::string() : name;
				g_unread[name] = 0;
			}
			ImGui::SameLine();
			const bool fr = g_friends.count(name) != 0;
			if (Anim::Button(fr ? I18n::T("Quitar amigo") : I18n::T("Añadir amigo"), {std::round(190 * unit), 0}, unit)) {
				if (fr) g_friends.erase(name);
				else g_friends.insert(name);
				SaveFriends();
			}
		}
		ImGui::PopID();
		shown.insert(name);
	};
	for (const auto& n: on) if (g_friends.count(n)) row(n, true); // (11-23) connected friends first
	for (const auto& n: on) if (!g_friends.count(n)) row(n, true);
	for (const auto& f: g_friends)
		if (!shown.count(f)) row(f, false); // friends not connected: gray
	if (!g_error.empty()) ImGui::TextColored(kDim, "%s", g_error.c_str());
	if (g_open_with.empty()) return;
	// The conversation.
	g_unread[g_open_with] = 0;
	ImGui::Spacing();
	ImGui::TextColored(kGold, "%s", (std::string(I18n::T("Chat con ")) + g_open_with).c_str());
	ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0.06f, 0.07f, 0.09f, 0.9f});
	ImGui::BeginChild("##chatlog", {0, std::round(260 * unit)}, ImGuiChildFlags_Borders);
	for (const auto& l: g_convo[g_open_with]) {
		const ImVec4 col = l.mine ? ImVec4 {0.72f, 0.82f, 1.0f, 1.0f} : ImVec4 {0.93f, 0.91f, 0.88f, 1.0f};
		ImGui::PushStyleColor(ImGuiCol_Text, col);
		ImGui::TextWrapped("%s: %s", l.mine ? I18n::T("Tú") : l.from.c_str(), l.text.c_str());
		ImGui::PopStyleColor();
	}
	if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4) ImGui::SetScrollHereY(1.0f);
	ImGui::EndChild();
	ImGui::PopStyleColor();
	ImGui::SetNextItemWidth(-std::round(170 * unit));
	bool send = ImGui::InputText("##chatin", g_input, sizeof(g_input), ImGuiInputTextFlags_EnterReturnsTrue);
	ImGui::SameLine();
	send |= Anim::Button(I18n::T("Enviar"), {std::round(150 * unit), 0}, unit);
	if (send && g_input[0] != 0) {
		g_outbox.emplace_back(g_open_with, g_input);
		g_convo[g_open_with].push_back({g_me, g_input, true});
		SaveHistory(); // (11-23)
		g_input[0] = 0;
		ImGui::SetKeyboardFocusHere(-1);
	}
}
// (11-22) Next to the account name: the friends connected now, each a button that opens the chat with them (the
// conversation shows in the chat box below, as from its list). Nothing when there is no account or none is connected.
void FriendsInline(float unit) {
	std::lock_guard lock(g_cm);
	if (!g_saved.multiplayer || g_saved.online_name.empty() || !g_have_list) return;
	std::vector<std::string> on;
	for (const auto& n: g_online)
		if (n != g_me && g_friends.count(n)) on.push_back(n);
	if (on.empty()) return;
	ImGui::SameLine(0, std::round(18 * unit));
	ImGui::TextColored(ImVec4 {0.45f, 0.86f, 0.52f, 1.0f}, "%s", I18n::T("Amigos conectados:"));
	for (const auto& n: on) {
		ImGui::SameLine();
		ImGui::PushID(n.c_str());
		std::string label = n;
		if (int k = g_unread.count(n) ? g_unread[n] : 0; k > 0) label += " (" + std::to_string(k) + ")";
		if (ImGui::SmallButton(label.c_str())) {
			g_open_with  = n;
			g_unread[n]  = 0;
		}
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", I18n::T("Abrir el chat con este amigo"));
		ImGui::PopID();
	}
}
} // namespace ChatUi

namespace Multiplayer {
namespace On = Port::Online;
std::mutex  g_m;
std::string g_state_json, g_public_ip, g_lan_ip, g_note, g_test;
bool        g_server_up = false, g_local_up = false, g_firewall = false, g_autostart = false;
std::atomic<bool> g_busy {false}, g_busy_ip {false};
std::chrono::steady_clock::time_point g_next {};
char        g_server[128] = "", g_name[32] = "", g_pass[64] = "";
bool        g_inited = false;
float       g_pulse_t = 0.0f;

void Note(const std::string& s) {
	std::lock_guard lock(g_m);
	g_note = s;
}
std::string Host() { // (online10) fixed server: this PC's own when it runs it, else the owner's via DuckDNS
	return g_local_up ? std::string("127.0.0.1") : std::string(On::kPublicServer);
}
bool IsLocal() {
	const auto h = Host();
	return h == "127.0.0.1" || h == "localhost";
}
void Refresh() { // (worker) the server's state, this PC's server, firewall, autostart
	if (g_busy.exchange(true)) return;
	const std::string host = Host();
	std::thread([host] {
		const std::string st    = On::ServerState(host + ":" + std::to_string(On::kDefaultPort));
		const bool        local = On::Probe("127.0.0.1:" + std::to_string(On::kDefaultPort), nullptr);
		const bool        fw    = On::FirewallReady();
		const bool        as    = On::Autostart();
		{
			std::lock_guard lock(g_m);
			g_state_json = st;
			g_server_up  = !st.empty();
			g_local_up   = local;
			g_firewall   = fw;
			g_autostart  = as;
		}
		g_busy = false;
		Bump();
	}).detach();
}
void FindIps() {
	if (g_busy_ip.exchange(true)) return;
	std::thread([] {
		const std::string lan = On::LanIp(), pub = On::PublicIp();
		{
			std::lock_guard lock(g_m);
			g_lan_ip = lan, g_public_ip = pub;
		}
		g_busy_ip = false;
		Bump();
	}).detach();
}
void Save(bool on) {
	Settings on_disk;
	if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_saved;
	for (Settings* s: {&on_disk, &g_saved, &g_edit}) {
		s->multiplayer     = on;
		s->online_server   = Host();
		s->online_name     = g_name;
		s->online_password = g_pass;
	}
	(void)SaveSettings(g_rt.ini_path, on_disk);
}
// The flat values of a key in the server's /kyty/estado JSON (a tiny reader for our own answer).
std::vector<std::string> Strings(const std::string& json, const std::string& array_key) {
	std::vector<std::string> out;
	const auto               a = json.find("\"" + array_key + "\":[");
	if (a == std::string::npos) return out;
	const auto e = json.find(']', a);
	for (size_t at = json.find('"', a + array_key.size() + 4); at != std::string::npos && at < e;) {
		const auto z = json.find('"', at + 1);
		out.push_back(json.substr(at + 1, z - at - 1));
		at = json.find('"', z + 1);
	}
	return out;
}
long long Number(const std::string& json, const std::string& key) {
	const auto a = json.find("\"" + key + "\":");
	return a == std::string::npos ? -1 : std::atoll(json.c_str() + a + key.size() + 3);
}

// The settings rows (the launcher's Configurador and the F2 page).
void SettingsBlock(float unit) {
	if (!g_inited) {
		g_inited = true;
		std::snprintf(g_server, sizeof(g_server), "%s", g_saved.online_server.c_str());
		std::snprintf(g_name, sizeof(g_name), "%s", g_saved.online_name.c_str());
		std::snprintf(g_pass, sizeof(g_pass), "%s", g_saved.online_password.c_str());
		FindIps();
	}
	if (std::chrono::steady_clock::now() >= g_next) {
		g_next = std::chrono::steady_clock::now() + std::chrono::seconds(3);
		Refresh();
	}
	std::string note;
	{
		std::lock_guard lock(g_m);
		note = g_note;
	}
	const ImVec2 bs {std::round(300 * unit), std::round(50 * unit)};
	// (trucos-online) Cheats in use: the box shows off and cannot be turned on (online and cheats never together).
	const bool   cheats = CheatsInUse() || g_saved.randomizer; // (rando1) the randomizer plays offline too
	bool         on     = g_saved.multiplayer && !cheats;
	if (ImGui::Checkbox(I18n::T("Conexión online"), &on)) {
		if (on && g_saved.randomizer) {
			Note(I18n::T("Desactiva el Randomizer para jugar online."));
		} else if (on && cheats) {
			Note(I18n::T("Desactiva los trucos (o reinicia sin usarlos) para jugar online."));
			std::printf("[port] online: no se puede activar, hay trucos en uso\n");
		} else {
			Save(on);
			Note(on ? I18n::T("Multijugador activado. Se aplica al reiniciar el juego.") : I18n::T("Multijugador apagado: el juego vuelve a ser sin conexión al reiniciar."));
		}
	}
	Anim::Highlight(unit);
	if (cheats) ImGui::TextColored(kGold, "%s", I18n::T("Desactiva los trucos (o reinicia sin usarlos) para jugar online."));
	ImGui::TextColored(kDim, "%s", (std::string(I18n::T("Servidor: ")) + On::kPublicServer + I18n::T(" (todos los jugadores al mismo)")).c_str()); // (online10)
	ImGui::SetNextItemWidth(std::round(420 * unit));
	ImGui::InputTextWithHint(I18n::T("Cuenta"), I18n::T("nombre (letras y números)"), g_name, sizeof(g_name));
	Anim::Highlight(unit);
	FieldTools("##g_name", g_name, sizeof(g_name)); // (online6) Pegar
	ChatUi::FriendsInline(unit); // (11-22) the friends connected, next to the name: a click opens their chat
	ImGui::SetNextItemWidth(std::round(420 * unit));
	ImGui::InputTextWithHint(I18n::T("Contraseña"), "", g_pass, sizeof(g_pass), ImGuiInputTextFlags_Password);
	Anim::Highlight(unit);
	FieldTools("##g_pass", g_pass, sizeof(g_pass)); // (online6) Pegar
	ImGui::BeginDisabled(g_busy.load());
	if (Anim::Button(I18n::T("Crear cuenta"), bs, unit)) {
		const std::string host = Host(), name = g_name, pass = g_pass;
		std::thread([host, name, pass] {
			std::string error;
			const bool  ok = On::CreateAccount(host, name, pass, &error);
			Note(ok ? std::string(I18n::T("Cuenta creada: ")) + name : std::string(I18n::T("No se pudo crear la cuenta: ")) + error);
			Bump();
		}).detach();
		Save(g_saved.multiplayer);
	}
	ImGui::SameLine();
	if (Anim::Button(I18n::T("Guardar"), bs, unit)) {
		Save(g_saved.multiplayer);
		Note(I18n::T("Guardado. Se aplica al reiniciar el juego."));
	}
	ImGui::EndDisabled();
	// (online7) what is written is kept without "Guardar": once no field is being edited, a server, account or password
	// different from the saved one goes to settings.ini (the game reads it at its next start).
	if (!ImGui::IsAnyItemActive() && (g_saved.online_server != Host() || g_saved.online_name != g_name || g_saved.online_password != g_pass)) {
		Save(g_saved.multiplayer);
		Note(I18n::T("Guardado. Se aplica al reiniciar el juego."));
	}
	if (!note.empty()) ImGui::TextColored(kGold, "%s", note.c_str());
}

// (online11) The server's state as a pulsing seal: soul blue connected, ember no answer.
void Seal(float unit, bool up) {
	// the server's state: a pulsing seal (soul blue: connected; ember: no answer), players and sessions.
	{
		ImDrawList*  dl   = ImGui::GetWindowDrawList();
		const float  side = std::round(54 * unit);
		const ImVec2 at   = ImGui::GetCursorScreenPos();
		ImGui::Dummy({side, side});
		g_pulse_t       = static_cast<float>(ImGui::GetTime());
		const float  pl = 0.5f + 0.5f * std::sin(g_pulse_t * (up ? 2.2f : 5.0f));
		const ImVec2 c {std::round(at.x + side * 0.5f), std::round(at.y + side * 0.5f)};
		const ImVec4 col = up ? ImVec4 {0.72f, 0.82f, 1.0f, 1.0f} : ImVec4 {0.95f, 0.45f, 0.25f, 1.0f};
		for (int k = 3; k >= 1; k--)
			dl->AddCircleFilled(c, side * (0.25f + 0.1f * k + 0.05f * pl), ImGui::GetColorU32(ImVec4 {col.x, col.y, col.z, ImGui::GetStyle().Alpha * 0.06f * k}), 32);
		Icons::Draw(dl, Icons::Soul, c, side * (0.7f + 0.05f * pl), ImGui::GetColorU32(ImVec4 {col.x, col.y, col.z, ImGui::GetStyle().Alpha}));
		ImGui::SameLine();
		ImGui::BeginGroup();
		const std::string head = std::string(I18n::T("Servidor ")) + Host() + ":31415  -  " + (up ? I18n::T("conectado") : I18n::T("no responde"));
		ImGui::TextColored(up ? kGold : kDim, "%s", head.c_str());
		const std::string mode = g_saved.multiplayer ? std::string(I18n::T("Multijugador activado (cuenta ")) + g_saved.online_name + ")"
		                                             : std::string(I18n::T("Multijugador apagado: sin conexión, como siempre"));
		ImGui::TextColored(kDim, "%s", mode.c_str());
		ImGui::EndGroup();
	}
}
// (stats1 -> cleanbtn) "Deshabilitar trucos y mejoras": el control se movió fuera de este panel al medallón "SIN TRUCOS"
// de la barra inferior del lanzador (LauncherTrophy::DrawNoCheats). El comportamiento de ocultar/forzar los trucos (F2
// oculta Trucos y Armas y objetos; Cheats::Tick los fuerza apagados con g_saved.no_cheats) no cambia; solo se movió la UI.

// (online11) "Conexión online" in the launcher and F2: title, seal, players, then account and password.
void Card(float unit) {
	Bump();
	bool up, local;
	std::string st;
	{
		std::lock_guard lock(g_m);
		up = g_server_up, local = g_local_up, st = g_state_json;
	}
	(void)local;
	ImGui::PushFont(DisplayFont(), 40.0f * unit);
	ImGui::TextColored(kGold, "%s", I18n::T("Conexión online"));
	ImGui::PopFont();
	ImGui::TextColored(kDim, "%s", I18n::T("Mensajes, manchas de sangre, fantasmas e invocaciones con tus amigos. Todos en el mismo servidor."));
	ImGui::Spacing();
	ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0.10f, 0.11f, 0.14f, 0.85f});
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 14 * unit);
	ImGui::BeginChild("##oncard", {0, 0}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding | ImGuiChildFlags_AutoResizeY);
	Seal(unit, up);
	if (up) ChatUi::Draw(unit, Strings(st, "jugadores")); // (chat1) the players (green: connected), chat and friends
	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopStyleColor();
	ImGui::Spacing();
	SettingsBlock(unit);
	// (cleanbtn) "Deshabilitar trucos y mejoras" ya no vive aquí: ahora es el medallón "SIN TRUCOS" de la barra inferior
	// del lanzador (LauncherTrophy::DrawNoCheats), visible sin abrir este panel.
}

bool Page(float width, float height, float unit) {
	using namespace F1Pages;
	Begin("Multijugador", width, height, unit);
	ImGui::BeginChild("##multi", {0, -std::round(84 * unit)}, ImGuiChildFlags_NavFlattened);
	Anim::SmoothScroll();
	if (const int d = g_page_scroll.exchange(0); d != 0) ImGui::SetScrollY(ImGui::GetScrollY() + d * ImGui::GetWindowHeight() * 0.6f);
	Bump();
	std::string st, pub, lan;
	bool        up, local, fw, as;
	{
		std::lock_guard lock(g_m);
		st = g_state_json, pub = g_public_ip, lan = g_lan_ip;
		up = g_server_up, local = g_local_up, fw = g_firewall, as = g_autostart;
	}
	Seal(unit, up);
	if (up) {
		// (11-23) In the game too: the players (friends online in green, first), the chat with the keyboard and friends,
		// as in the launcher (before: only a text line with the names, no chat in the game).
		ChatUi::Draw(unit, Strings(st, "jugadores"));
		ImGui::Spacing();
		const auto sessions = Number(st, "sesiones") >= 0 ? 0 : 0;
		(void)sessions;
		size_t     n_ses = 0;
		for (size_t a = st.find("\"sessionId\""); a != std::string::npos; a = st.find("\"sessionId\"", a + 1)) n_ses++;
		InfoLine(I18n::T("Partidas (sesiones) abiertas"), std::to_string(n_ses), kDimL);
		InfoLine(I18n::T("Mensajes / manchas guardados"), std::to_string(std::max(0LL, Number(st, "mensajes"))) + " / " + std::to_string(std::max(0LL, Number(st, "manchas"))), kDimL);
	}
	if (g_saved.multiplayer != g_rt.launched.multiplayer || g_saved.online_server != g_rt.launched.online_server || g_saved.online_name != g_rt.launched.online_name ||
	    g_saved.online_password != g_rt.launched.online_password) {
		ImGui::TextColored(kDim, "%s", I18n::T("El cambio se aplica al reiniciar el juego."));
		ImGui::SameLine();
		if (Anim::Button(I18n::T("Reiniciar ahora"), {std::round(260 * unit), std::round(44 * unit)}, unit)) RequestRestart();
	}
	ImGui::Spacing();
	ImGui::Separator();
	SettingsBlock(unit);
	ImGui::Spacing();
	ImGui::Separator();
	// This PC as the server (the host).
	ImGui::TextColored(kGold, "%s", I18n::T("Este PC como servidor (anfitrión)"));
	InfoLine(I18n::T("Servidor en este PC"), local ? I18n::T("encendido (TCP 31413, 31415; UDP 3758)") : I18n::T("apagado"), local ? kGold : kDim);
	const ImVec2 bs {std::round(330 * unit), std::round(50 * unit)};
	if (Anim::Button(local ? I18n::T("Apagar el servidor") : I18n::T("Encender el servidor"), bs, unit)) {
		if (local) {
			On::StopServer();
			Note(I18n::T("Servidor apagado."));
		} else {
			std::thread([] { Note(On::EnsureHost(On::kDefaultPort) ? I18n::T("Servidor encendido.") : I18n::T("El servidor no arranca: ¿algún puerto ocupado?")); Bump(); }).detach();
		}
		g_next = std::chrono::steady_clock::now() + std::chrono::milliseconds(800);
	}
	ImGui::SameLine();
	bool always = as;
	if (ImGui::Checkbox(I18n::T("Siempre encendido (arranca con Windows)"), &always)) {
		Note(On::SetAutostart(always, On::kDefaultPort) ? (always ? I18n::T("Arrancará con Windows.") : I18n::T("Ya no arranca con Windows.")) : I18n::T("No se pudo cambiar."));
		if (always && !local) std::thread([] { On::EnsureHost(On::kDefaultPort); Bump(); }).detach();
		g_next = {};
	}
	InfoLine(I18n::T("Firewall de Windows"), fw ? I18n::T("permitido") : I18n::T("sin permiso"), fw ? kGold : kDim);
	if (!fw) {
		ImGui::SameLine();
		if (Anim::Button(I18n::T("Permitir (pide permiso una vez)"), bs, unit))
			std::thread([] { Note(On::AddFirewallRule(On::kDefaultPort) ? I18n::T("Firewall: permitido.") : I18n::T("Firewall: no se aceptó el permiso.")); g_next = {}; Bump(); }).detach();
	}
	InfoLine(I18n::T("IP pública (para tus amigos)"), g_busy_ip.load() ? std::string(I18n::T("buscando...")) : (pub.empty() ? std::string("?") : pub), kDimL);
	InfoLine(I18n::T("IP en tu red"), lan.empty() ? std::string("?") : lan, kDimL);
	{
		ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
		ImGui::TextColored(kDim, "%s", I18n::T("Router (solo el anfitrión): TCP 31413, TCP 31415 y UDP 3758 hacia la IP de este PC. Los amigos no abren nada: el juego entre jugadores pasa por el servidor."));
		ImGui::PopTextWrapPos();
	}
	ImGui::EndChild();
	ImGui::Separator();
	bool close = Anim::Button(I18n::T("Volver"), {std::round(240 * unit), std::round(54 * unit)}, unit);
	ImGui::SameLine();
	ImGui::TextDisabled("%s", I18n::T("L1 / L2: desplazarse   Círculo / Esc: volver"));
	PageZoom();
	ImGui::End();
	ImGui::PopStyleColor(g_style_colors);
	ImGui::PopStyleVar(g_style_vars);
	g_style_colors = g_style_vars = 0;
	ImGui::PopFont();
	return !close;
}
} // namespace Multiplayer

namespace KeepFocus {
struct Mem {
	ImGuiID      id = 0;
	ImGuiWindow* window = nullptr;
};
Mem g_mem[7]; // (trucos: 4 the Trucos page, 5 Armas y objetos; online5: 6 Multijugador)
int g_screen = -1, g_frame = -10;
// Called once per drawn screen, after its items were submitted.
void After(int screen) {
	ImGuiContext& g     = *ImGui::GetCurrentContext();
	const int     frame = ImGui::GetFrameCount();
	const bool    back  = g_frame != frame - 1 || g_screen != screen;
	g_frame  = frame;
	g_screen = screen;
	Mem& m   = g_mem[screen];
	if (back) {
		if (m.id != 0 && m.window != nullptr) {
			ImGui::FocusWindow(m.window);
			ImGui::SetFocusID(m.id, m.window);
			g.NavInitRequest   = false; // (not ImGui's own pick of the first item)
			g.NavInitResult.ID = 0;
			g.NavCursorVisible = true;
			g.NavHighlightItemUnderNav = true;
		}
		return;
	}
	if (g.NavId != 0 && g.NavWindow != nullptr) m = {g.NavId, g.NavWindow};
}
} // namespace KeepFocus

void (*g_menu_rect_hook)(ImVec2, ImVec2) = nullptr; // (perfopts-1107) the F2 window's rectangle, for next frame's fog
void DrawInGameMenu(float width, float height, bool closing = false) {
	if (!closing) DrainInputs();
	const float unit = std::max(std::min(width / 1920.0f, height / 1080.0f), 0.5f);
	ImGui::GetIO().MouseDrawCursor = !closing;
	if (closing && (g_f1_page.load() != 0 || g_trophy_page.load())) return; // (a page closes at once)
	if (CheatsBlocked() && (g_f1_page.load() == 3 || g_f1_page.load() == 4)) g_f1_page = 0; // (stats1/onlineclean) Trucos / Armas y objetos hidden
	if (const int page = g_f1_page.load(); page != 0) {
		const bool open = page == 1 ? F1Pages::DrawPad(width, height, unit) : page == 3 ? Cheats::Draw(width, height, unit)
		                : page == 4 ? Shop::Draw(width, height, unit) : page == 5 ? Multiplayer::Page(width, height, unit)
		                            : F1Pages::DrawKbm(width, height, unit);
		KeepFocus::After(page == 1 ? 1 : page == 3 ? 4 : page == 4 ? 5 : page == 5 ? 6 : 2);
		if (!open) {
			if (page == 3) Cheats::g_souls_loaded = false;
			g_f1_page       = 0;
			Anim::g_page_at = Anim::Now(); // (ds-pc-port38) the menu fades in again
			F1Pages::g_note.clear();
		}
		return;
	}
	if (g_trophy_page.load()) {
		const bool trophy_open = TrophyPage::Draw(width, height, unit);
		KeepFocus::After(3);
		if (!trophy_open) {
			g_trophy_page      = false;
			g_focus_trophy_btn = true;
		}
		Bump(); // (animated: a new picture each frame)
		return;
	}
	// (ds-pc-port38) Open: fade in and grow from 94 %; close: fade out and shrink back (Anim, vertex scale: the fonts stay
	// rasterized at one size). Back from a page: the menu fades in again.
	// (step5) Real time (steady clock), not frames: ease-out quint in, ease-in cubic out.
	const float t = closing ? 1.0f - Anim::EaseIn(Anim::Progress(Anim::g_closed_at.load(), Anim::kClose))
	                        : Anim::EaseOutQuint(Anim::Progress(std::max(Anim::g_opened_at.load(), Anim::g_page_at.load()), Anim::kOpen));
	// (perfopts-1107) The idle animations (soul motes, fog, the breathing sigil, the candle flicker) redraw while open.
	Bump();
	Anim::g_page_seen = 0;
	// (trucos11) Entry: from black, then the stone-and-fog backdrop fades in over the game (ds-style.h Backdrop: blue-black
	// stone, slow fog banks, grain, strong vignette) with faint rising soul motes. No bounce, no slide.
	{
		ImDrawList* bg = ImGui::GetBackgroundDrawList();
		const float tm = Souls::Time();
		DsStyle::Backdrop(bg, {0, 0}, {width, height}, 0.88f * t, tm);
		DsStyle::FogLayers(bg, {0, 0}, {width, height}, t, tm); // (trucos12) layered fog with slow parallax
		DsStyle::Embers(bg, {0, 0}, {width, height}, t * 0.6f, tm); // (trucos12) embers and soul motes, flickering
		AshFrame(bg, width, height, t);                              // (trucos13) turbulent ash
		if (!closing && t < 1.0f) bg->AddRectFilled({0, 0}, {width, height}, IM_COL32(0, 0, 0, static_cast<int>(255 * (1.0f - t) * 0.6f)));
	}
	// (des-fog-hq) No fog, motes or vignette behind the F2 menu: the plain dim of the game image only (the ambience is
	// the pre-game launcher's alone).
	g_menu_rect_hook = nullptr;
	ImFont* font = FindFont("segoeui");
	ImGui::PushFont(font, std::round(32.0f * unit)); // (step5: larger text, whole-pixel size)
	ImGui::PushStyleVar(ImGuiStyleVar_Alpha, std::max(0.01f, t));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {32 * unit, 24 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {12 * unit, 14 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10 * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 14 * unit);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0, 0, 0, 0}); // (perfopts-1107: the carved stone panel, Souls::Panel)
	ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.13f, 0.125f, 0.12f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.15f, 0.14f, 0.13f, 1.0f}); // (the transition is Anim::Highlight's)
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0.30f, 0.26f, 0.18f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_NavCursor, kGold);
	ImGui::SetNextWindowPos({width * 0.5f, height * 0.5f}, ImGuiCond_Always, {0.5f, 0.5f});
	ImGui::Begin("##f1menu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove |
	                                     ImGuiWindowFlags_NoSavedSettings | (closing ? ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav : 0));
	ImDrawList* menu_dl    = ImGui::GetWindowDrawList();
	const int   menu_first = 0; // (the window's own list: its background and contents; closing: NoInputs, NoNav)
	{
		// (perfopts-1107) The header: a carved monolith sigil with its soul glow, the title engraved in Cinzel (tarnished
		// gold), a candle-lit filigree divider under it.
		menu_dl->ChannelsSplit(2);
		menu_dl->ChannelsSetCurrent(1);
		Glide::Begin();
		const float  side = ImGui::GetFontSize() * 1.7f;
		const ImVec2 at   = ImGui::GetCursorScreenPos();
		const float  al   = ImGui::GetStyle().Alpha;
		DsStyle::Medallion(menu_dl, DsStyle::Icon::Flame, {at.x + side * 0.45f, at.y + side * 0.5f}, side * 0.45f, al, 0.0f, Souls::Time(), 0.6f); // (trucos11)
		ImGui::Dummy({side * 0.95f, side});
		ImGui::SameLine();
		ImFont*      title_font = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
		const float  title_size = std::round(ImGui::GetFontSize() * 1.3f);
		const char*  title      = I18n::T("Demon's Souls PC");
		const ImVec2 tsz        = title_font->CalcTextSizeA(title_size, FLT_MAX, 0.0f, title);
		const ImVec2 tp {std::round(ImGui::GetCursorScreenPos().x), std::round(at.y + (side - tsz.y) * 0.5f)}; // (trucos2) whole pixels: crisp Cinzel
		// A warm candle glow behind the title, flickering.
		const float fl = Souls::Flicker();
		(void)fl;
		const std::string tup = DsStyle::Upper(title); // (trucos11) bone-white tracked capitals over a rhombus rule
		const float tw = DsStyle::IlluminatedTitle(menu_dl, title_font, title_size, tp, tup.c_str(), al,
		                                           static_cast<float>(Anim::Now() - Anim::g_opened_at.load()), Souls::Time()); // (trucos13)
		ImGui::Dummy({std::max(tsz.x, tw), side});
		const float dy = at.y + side + 10 * unit;
		DsStyle::DiamondRule(menu_dl, at.x, at.x + std::max(420 * unit, tw + side), dy, al, unit);
		ImGui::SetCursorScreenPos({at.x, dy + 4 * unit});
		ImGui::Dummy({420 * unit, 12 * unit});
	}
	bool show = g_saved.start_menu;
	if (ImGui::Checkbox(I18n::T("Mostrar el lanzador la pr\xC3\xB3xima vez que abras el juego"), &show)) {
		Settings on_disk;
		if (!LoadSettings(g_rt.ini_path, &on_disk)) on_disk = g_saved;
		on_disk.start_menu  = show;
		g_saved.start_menu  = show;
		g_edit.start_menu   = show;
		(void)SaveSettings(g_rt.ini_path, on_disk);
	}
	// (bb-style-f1) The options side by side as Bloodborne's hub cards (CardButton); same order, labels and actions.
	const ImVec2 bs {std::round(196 * unit), std::round(330 * unit)}; // (trucos7: 7 cards; online5: 8, narrower)
	ImGui::Dummy({1.0f, 24.0f * unit}); // (room for the focused card's lift)
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {28 * unit, 14 * unit});
	if (CardButton(I18n::T("Volver al juego"), Icons::Back, bs, unit, 0)) SetOpen(false);
	// (port34) The trophy page (Esc, B or F2 there: back here).
	ImGui::SameLine();
	if (g_focus_trophy_btn_request.exchange(false)) g_focus_trophy_btn = true;
	if (g_focus_trophy_btn) {
		ImGui::SetKeyboardFocusHere();
		g_focus_trophy_btn = false;
	}
	if (CardButton(I18n::T("Ver trofeos"), Icons::Trophy, bs, unit, 1)) {
		g_trophy_tab_delta = 0;
		TrophyPage::Open();
		g_trophy_page = true;
		Bump();
	}
	ImGui::SameLine();
	if (CardButton(I18n::T("Mapear mando"), Icons::Pad, bs, unit, 2)) g_f1_page = 1;
	ImGui::SameLine();
	if (CardButton(I18n::T("Teclado y ratón"), Icons::Keyboard, bs, unit, 3)) g_f1_page = 2;
	ImGui::SameLine();
	if (!CheatsBlocked()) { // (stats1/onlineclean) no_cheats o el online: sin Trucos ni Armas y objetos
		if (CardButton(I18n::T("Trucos"), Icons::Skull, bs, unit, 4)) g_f1_page = 3; // (trucos)
		ImGui::SameLine();
		if (CardButton(I18n::T("Armas y objetos"), Icons::Chest, bs, unit, 5)) g_f1_page = 4; // (trucos7) the shop
		ImGui::SameLine();
	}
	if (CardButton(I18n::T("Multijugador"), Icons::Soul, bs, unit, 6)) g_f1_page = 5; // (online5)
	ImGui::SameLine();
	if (CardButton(I18n::T("Cerrar el juego"), Icons::Power, bs, unit, 7, true)) {
		// Just closes: nothing saved, nothing skipped (the game's own saves are already on disk).
#if defined(_WIN32)
		// Not ExitProcess: it runs the DLLs' detach routines (the GPU driver's, Streamline's) on this thread while the
		// other threads may hold what those wait for (the same hang as assert.cpp's). TerminateProcess skips them.
		// The log's last lines: flushed on a helper thread, waited for 300 ms at most. A plain fflush here can wait for
		// good: standard output goes through the async pipe (AsyncStdout) and a stream lock held by a stuck thread.
		{
			static std::atomic<bool> flushed {false};
			std::thread([] {
				std::fflush(nullptr);
				flushed.store(true);
			}).detach();
			for (int i = 0; i < 30 && !flushed.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		ReturnToLauncher(); // (11-09) TerminateProcess runs no atexit: the launcher opens again from here
		TerminateProcess(GetCurrentProcess(), 0);
#else
		std::_Exit(0);
#endif
	}
	ImGui::PopStyleVar();
	ImGui::TextDisabled("%s", I18n::T("F2 o Esc: volver al juego"));
	if (!closing) KeepFocus::After(0);
	{
		// (perfopts-1107) Under the contents: the carved stone panel and the gliding highlight; while opening or closing,
		// a pale soul-light bloom around the panel that fades as it settles.
		const ImVec2 wp = ImGui::GetWindowPos(), ws = ImGui::GetWindowSize();
		const float  al = ImGui::GetStyle().Alpha;
		menu_dl->ChannelsSetCurrent(0);
		menu_dl->PushClipRectFullScreen();
		if (t < 1.0f)
			for (int k = 4; k >= 1; k--) {
				const float e = static_cast<float>(k) * 9.0f * unit;
				menu_dl->AddRectFilled({wp.x - e, wp.y - e}, {wp.x + ws.x + e, wp.y + ws.y + e}, Souls::C(Souls::kSoul, (1.0f - t) * 0.06f), 10 * unit + e);
			}
		DsStyle::Panel(menu_dl, wp, {wp.x + ws.x, wp.y + ws.y}, al, unit); // (trucos11)
		{
			// (trucos12) Gothic gold filigree in the four corners and a crest (scroll wings and a rosette) on the top edge.
			const ImVec2 pb {wp.x + ws.x, wp.y + ws.y};
			const float  fs = std::round(150 * unit), tm = Souls::Time();
			DsStyle::Filigree(menu_dl, wp, fs, 1, 1, al, tm, unit);
			DsStyle::Filigree(menu_dl, {pb.x, wp.y}, fs, -1, 1, al, tm, unit);
			DsStyle::Filigree(menu_dl, {wp.x, pb.y}, fs, 1, -1, al, tm, unit);
			DsStyle::Filigree(menu_dl, pb, fs, -1, -1, al, tm, unit);
			DsStyle::Crest(menu_dl, {std::round(wp.x + ws.x * 0.5f), wp.y}, std::round(260 * unit), al, tm, unit);
		}
		Glide::Draw(menu_dl, unit, al);
		menu_dl->PopClipRect();
		menu_dl->ChannelsMerge();
		if (g_menu_rect_hook != nullptr) g_menu_rect_hook(wp, ws);
		// The zoom: 96 % -> 100 % about the window's centre plus a short rise (14 px at 1080p), sub-pixel; once settled the
		// vertices are untouched (crisp text at its baked size).
		Anim::ScaleVertices(menu_dl, menu_first, {wp.x + ws.x * 0.5f, wp.y + ws.y * 0.5f}, 0.96f + 0.04f * t, (1.0f - t) * 14.0f * unit);
	}
	ImGui::End();
	ImGui::PopStyleColor(6);
	ImGui::PopStyleVar(5);
	ImGui::PopFont();
}

void DrawInGame(float width, float height) {
	if (g_open.load()) g_popup = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId); // (ds-pc-port42: the game's own frames, last frame's state)
	MetricsPanel::Draw(width, height); // (ds-pc-port41) under the menu too
	ChatUi::DrawToasts(width, height); // (chat1) a message or a friend connecting, top right
	if (g_open.load() && !Resume::SaveAndQuitActive()) {
		DrawInGameMenu(width, height);
		return;
	}
	if (Anim::Closing() && !Resume::SaveAndQuitActive()) { // (ds-pc-port38) the fade-out
		DrawInGameMenu(width, height, true);
		return;
	}
	ImGui::GetIO().MouseDrawCursor = false;
	if (!Resume::SaveAndQuitActive()) return;
	const float unit = std::max(std::min(width / 1920.0f, height / 1080.0f), 0.5f);
	ImGui::GetBackgroundDrawList()->AddRectFilled({0, 0}, {width, height}, IM_COL32(0, 0, 0, 150));
	ImFont* font = FindFont("segoeui");
	ImGui::PushFont(font, 28.0f * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {36 * unit, 28 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 14 * unit);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0.03f, 0.03f, 0.035f, 0.92f});
	ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0.87f, 0.77f, 0.55f, 0.55f});
	ImGui::SetNextWindowPos({width * 0.5f, height * 0.5f}, ImGuiCond_Always, {0.5f, 0.5f});
	ImGui::SetNextWindowSize({std::min(820 * unit, width - 32), 0}, ImGuiCond_Always);
	ImGui::Begin("##SaveQuit", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
	                                         ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
	ImGui::PushFont(nullptr, 40.0f * unit);
	ImGui::TextColored(kGold, "%s", I18n::T("Guardar y salir"));
	ImGui::PopFont();
	ImGui::Separator();
	ImGui::PushTextWrapPos(0.0f);
	const std::string text = Resume::SaveAndQuitText();
	ImGui::TextUnformatted(I18n::T(text.c_str()));
	ImGui::TextColored(kDimL, "%s", I18n::T("El juego se cierra solo en unos segundos. Al pulsar JUGAR en el lanzador continuarás donde estás."));
	ImGui::PopTextWrapPos();
	ImGui::End();
	ImGui::PopStyleColor(2);
	ImGui::PopStyleVar(2);
	ImGui::PopFont();
	Bump();
}

unsigned long long Revision() {
	return g_revision.load(std::memory_order_relaxed);
}

void Presented() {
	if (g_first_present) {
		g_first_present = false;
		CloseSplash();
	}
	Resume::Presented(); // (ds-pc-port8) JUGAR after F2: the automatic Continue
	Cheats::Tick();      // (trucos) each presented frame (DrawInGame runs only with the overlay up); work every 0.5 s
	ChatUi::Start();     // (chat1) once: the chat's worker (messages and friends' notices in the game)
}

void WindowCreated(SDL_Window* window) {
#if defined(_WIN32)
	SDL_SysWMinfo info {};
	SDL_VERSION(&info.version);
	if (SDL_GetWindowWMInfo(window, &info) == SDL_TRUE && info.subsystem == SDL_SYSWM_WINDOWS) {
		const HINSTANCE module = GetModuleHandleW(nullptr);
		auto*           big    = static_cast<HICON>(LoadImageW(module, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXICON),
		                                                      GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR));
		auto*           small  = static_cast<HICON>(LoadImageW(module, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON),
		                                                      GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR));
		if (big != nullptr) SendMessageW(info.info.win.window, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(big));
		if (small != nullptr) SendMessageW(info.info.win.window, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(small));
	}
#endif
	SDL_SetWindowTitle(window, "Demon's Souls");
}

const char* Title() {
	return "Demon's Souls";
}

// (ds-pc-port43) The menu (not the Metricas panel alone): what may re-present the last frame on an idle vblank.
bool MenuActive() {
	return g_open.load(std::memory_order_relaxed) || Resume::SaveAndQuitActive() || Anim::Closing();
}
const PortLive::Hooks kHooks {WindowCreated, OnEvent, Presented, WantsOverlay, Revision, DrawInGame, Title, MenuActive};

} // namespace

void InstallMenu(const Runtime& runtime) {
	g_rt    = runtime;
	// (ds-pc-port41) Metricas (settings.ini metricas=1/2): the panel shown from the start (F8 hides it).
	if (runtime.launched.metrics == "1" || runtime.launched.metrics == "2") MetricsPanel::Toggle();
	g_saved = runtime.launched;
	g_edit  = runtime.launched;
	ApplyLive(runtime.launched);
	const bool        fixed = FixedFg(runtime.launched);
	const std::string fg    = fixed ? std::string("amd-x2") : runtime.launched.framegen;
	if (fg != "sin-fg" && !fg.empty()) {
		PortLive::fg_label  = fixed ? (runtime.launched.souls_mode == "equilibrado" ? "MODO SOULS EQUILIBRADO"
		                               : runtime.launched.souls_mode == "low"       ? "MODO SOULS PC LOW PROFILE"
		                               : runtime.launched.souls_mode == "ligero-plus" ? "MODO SOULS LIGERO +"
		                                                                            : "MODO SOULS LIGERO")
		                            : FrameGenLabel(fg);
		PortLive::fg_plain  = fixed;
		PortLive::fg_vendor = fg.rfind("dlss", 0) == 0 ? 1 : 2;
		PortLive::fg_now    = fg == "amd-x2" ? 2 : (fg == "amd-x3" || fg == "amd-120") ? 3 : fg == "amd-x4" ? 4 : 0;
	}
	{
		static std::string gfx;
		const auto         level = [](const Choice* items, int count, const std::string& id) {
            for (int i = 0; i < count; i++)
                if (id == items[i].id) return items[i].label;
            return items[0].label;
		};
		gfx = std::string("Texturas ") + level(kTextureReduce, kTextureReduceCount, runtime.launched.texture_reduce) +
		      ", Distancia de dibujado " + level(kQualities, kQualityCount, runtime.launched.quality) + ", Sombras " +
		      level(kShadows, kShadowCount, runtime.launched.shadows) + ", Detalle " + level(kDetails, kDetailCount, runtime.launched.detail);
		PortLive::gfx_line = gfx.c_str();
	}
	PortLive::g_hooks   = &kHooks;
	PortLive::status_es = true; // (10-27b) "Preparando shaders X/Y" on screen
}

void RequestQuit() {
	SDL_Event quit {};
	quit.type = SDL_QUIT;
	SDL_PushEvent(&quit);
}

bool RunStartMenu(const Runtime& runtime, Settings* settings) {
#if defined(_WIN32)
	g_rt           = runtime;
	g_saved        = *settings;
	g_edit         = *settings;
	g_start_mode   = true;
	g_start_result = 0;
	// (launcher1) Pixel-exact launcher on scaled desktops (125 / 150 / 200 %): the same hints the game's window sets
	// (window.cpp), now before the launcher's SDL video init. Without them the process was DPI-unaware while the launcher
	// ran: SDL got the desktop in scaled-down units, ImGui drew into that smaller surface and Windows stretched the whole
	// window back to the screen (bitmap scaling: every letter blurred). The awareness is per process: the game's window,
	// created afterwards, asks for the same one.
	SDL_SetHint(SDL_HINT_WINDOWS_DPI_SCALING, "0");
	SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "permonitorv2");
	if (SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER | SDL_INIT_EVENTS) != 0) {
		std::printf("[port] start menu: SDL failed (%s): straight to the game\n", SDL_GetError());
		g_start_mode = false;
		return true;
	}
	// (ds-pc-port8) The launcher as Launcher-souls.ps1's: borderless over the whole main screen (the Configurador alone:
	// a window of 1600x900, at most 90% of the desktop).
	int    win_x = SDL_WINDOWPOS_CENTERED, win_y = SDL_WINDOWPOS_CENTERED, win_w = 1600, win_h = 900;
	Uint32 flags = SDL_WINDOW_ALLOW_HIGHDPI;
	if (SDL_Rect desk {}; !g_config_mode && SDL_GetDisplayBounds(0, &desk) == 0 && desk.w > 0 && desk.h > 0) {
		win_x = desk.x;
		win_y = desk.y;
		win_w = desk.w;
		win_h = desk.h;
		flags |= SDL_WINDOW_BORDERLESS;
	} else if (SDL_Rect usable {}; SDL_GetDisplayUsableBounds(0, &usable) == 0 && usable.w > 0 && usable.h > 0) {
		const float k = std::min({1.0f, usable.w * 0.9f / win_w, usable.h * 0.9f / win_h});
		win_w = static_cast<int>(win_w * k);
		win_h = static_cast<int>(win_h * k);
		flags |= SDL_WINDOW_RESIZABLE;
	}
	SDL_Window*  window = SDL_CreateWindow("Demon's Souls", win_x, win_y, win_w, win_h, flags);
	// (des-combo-1109) Direct3D 11 first (SDL2 picks Direct3D 9 by default on Windows: a D3D9 device in a borderless window
	// over the whole desktop, with the NVIDIA overlay/recording hooked in, showed frames without the illustration); the
	// default driver only if D3D11 cannot be created.
	SDL_Renderer* renderer = nullptr;
	if (window != nullptr) {
		SDL_SetHint(SDL_HINT_RENDER_DRIVER, "direct3d11");
		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
		SDL_SetHint(SDL_HINT_RENDER_DRIVER, "");
		if (renderer == nullptr) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
		if (SDL_RendererInfo info {}; renderer != nullptr && SDL_GetRendererInfo(renderer, &info) == 0)
			std::printf("[port] launcher renderer: %s\n", info.name != nullptr ? info.name : "?");
	}
	if (renderer == nullptr) {
		std::printf("[port] start menu: no window (%s): straight to the game\n", SDL_GetError());
		if (window != nullptr) SDL_DestroyWindow(window);
		SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER | SDL_INIT_EVENTS);
		g_start_mode = false;
		return true;
	}
	WindowCreated(window); // (the icon)
	ImGuiContext* previous = ImGui::GetCurrentContext();
	ImGuiContext* context  = ImGui::CreateContext();
	ImGui::SetCurrentContext(context);
	auto& io = ImGui::GetIO();
	io.IniFilename = nullptr;
	io.LogFilename = nullptr;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
	{
		// (ds-pc-port8) Segoe UI for the text (the launcher's), Segoe UI Semibold for buttons and captions, and a serif face
		// for the titles and JUGAR (Palatino Linotype, else Georgia, Constantia or Times New Roman: Windows' own).
		wchar_t windows[MAX_PATH + 1] = {};
		GetWindowsDirectoryW(windows, MAX_PATH);
		const auto fonts = std::filesystem::path(windows) / L"Fonts";
		const auto load  = [&](std::initializer_list<const wchar_t*> names) -> ImFont* {
			for (const wchar_t* name: names) {
				std::error_code ec;
				const auto      ttf = fonts / name;
				if (std::filesystem::exists(ttf, ec)) {
					// (launcher1) Crisp text: ImGui 1.92 rasterizes every glyph at the exact size it is drawn at, so the
					// glyphs go 1:1 on whole pixels (PixelSnapH, no oversampling). Before: OversampleH 3 / V 2 with
					// sub-pixel placement, drawn through the atlas' linear filter - every letter resampled (soft edges,
					// the vertical 2x halved by the filter). A touch heavier for the dark, busy art behind.
					ImFontConfig cfg;
					cfg.OversampleH        = 1;
					cfg.OversampleV        = 1;
					cfg.PixelSnapH         = true;
					cfg.RasterizerMultiply = 1.15f;
					return io.Fonts->AddFontFromFileTTF(ttf.string().c_str(), 0.0f, &cfg);
				}
			}
			return nullptr;
		};
		g_font_body  = load({L"segoeui.ttf"});
		g_font_semi  = load({L"seguisb.ttf", L"segoeuib.ttf"});
		g_font_serif = load({L"palab.ttf", L"pala.ttf", L"georgiab.ttf", L"constanb.ttf", L"timesbd.ttf"});
		if (g_font_semi == nullptr) g_font_semi = g_font_body;
		if (g_font_serif == nullptr) g_font_serif = g_font_semi;
		// (ds-pc-port12) Cinzel (OFL) from the exe's resources: rasterized at each exact size it is pushed at (ImGui 1.92's
		// dynamic fonts), oversampled, with a subtle letter spacing.
		if (HRSRC res = FindResourceW(nullptr, MAKEINTRESOURCEW(3), MAKEINTRESOURCEW(10)); res != nullptr) {
			if (HGLOBAL mem = LoadResource(nullptr, res); mem != nullptr) {
				void*       data = LockResource(mem);
				const DWORD size = SizeofResource(nullptr, res);
				if (data != nullptr && size > 0) {
					ImFontConfig cfg;
					cfg.FontDataOwnedByAtlas = false; // (the resource stays mapped for the process' life)
					cfg.OversampleH          = 1; // (launcher1) 1:1 on whole pixels, as the text faces
					cfg.OversampleV          = 1;
					cfg.PixelSnapH           = true;
					cfg.RasterizerMultiply   = 1.08f;
					cfg.GlyphExtraAdvanceX   = 2.0f; // (whole pixels: the inscription's letter spacing)
					g_font_display           = io.Fonts->AddFontFromMemoryTTF(data, static_cast<int>(size), 0.0f, &cfg);
				}
			}
		}
		if (g_font_display == nullptr) g_font_display = g_font_serif;
	}
	ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
	ImGui_ImplSDLRenderer2_Init(renderer);
	g_background_tried = false;
	g_background       = nullptr;
	g_focus_once       = true;
	// (des-combo-1109) The illustration and the ambience's textures are ready before the first frame (uploaded by its
	// RenderDrawData, before anything is drawn): no black launcher while they load.
	if (!g_config_mode) {
		EnsureBackground();
		FogGpu::g_renderer = renderer; // (11-10) the GPU fog draws through this renderer's D3D11 device
	}
	SDL_ShowCursor(SDL_ENABLE);
	CloseSplash();
	// (launcher1) Even frame pacing: VSync presents at the screen's rate; the launcher draws every n-th refresh, n the
	// whole divisor that keeps it at 60 fps or just above (60 Hz: 60, 120 Hz: 60, 144 Hz: 72, 165 Hz: 82.5), so every
	// frame lasts the same number of refreshes (no judder) and the CPU / GPU are not spent on 144-240 identical frames.
	// The wait is a high-resolution waitable timer (no busy loop); it also caps a launcher whose VSync is not honoured
	// (minimized, covered, a driver forcing it off) instead of letting it spin at thousands of frames a second.
	int refresh_hz = 60;
	if (SDL_DisplayMode mode {}; SDL_GetCurrentDisplayMode(0, &mode) == 0 && mode.refresh_rate >= 50) refresh_hz = mode.refresh_rate;
	const int    every_n   = std::max(1, static_cast<int>(std::floor(refresh_hz / 60.0 + 0.1)));
	const double frame_s   = static_cast<double>(every_n) / refresh_hz - 0.0015; // (a little early: VSync does the last bit)
	HANDLE       pace_wait = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */, TIMER_ALL_ACCESS);
	if (pace_wait == nullptr) pace_wait = CreateWaitableTimerW(nullptr, TRUE, nullptr);
	const Uint64 pace_freq = SDL_GetPerformanceFrequency();
	Uint64       pace_last = SDL_GetPerformanceCounter();
	const Uint64 fade_t0   = pace_last;
	std::printf("[port] launcher: %d Hz, drawn every %d refresh(es) (%.1f fps)\n", refresh_hz, every_n, static_cast<double>(refresh_hz) / every_n);
	while (g_start_result == 0) {
		SDL_Event e;
		while (SDL_PollEvent(&e) == 1) {
			if (PadCaptureFeed(e)) continue; // (remap-f2) a pad button being captured for "Mapear mando"
			if (e.type == SDL_CONTROLLERBUTTONDOWN) g_pad_last = static_cast<int>(e.cbutton.which);
			ImGui_ImplSDL2_ProcessEvent(&e);
			if (e.type == SDL_QUIT) g_start_result = -1;
			if (e.type == SDL_RENDER_DEVICE_RESET || e.type == SDL_RENDER_TARGETS_RESET) {
				// (des-combo-1109) The renderer lost its textures (device reset: a driver/overlay event): every ImGui texture
				// (the illustration, the ambience's, the fonts) is created again from the pixels it keeps.
				std::printf("[port] launcher: render %s reset: textures uploaded again\n", e.type == SDL_RENDER_DEVICE_RESET ? "device" : "targets");
				for (ImTextureData* tex: ImGui::GetPlatformIO().Textures) {
					if (tex->Status == ImTextureStatus_Destroyed || tex->GetPixels() == nullptr) continue;
					if (e.type == SDL_RENDER_DEVICE_RESET && tex->TexID != ImTextureID_Invalid)
						SDL_DestroyTexture(reinterpret_cast<SDL_Texture*>(static_cast<intptr_t>(tex->TexID)));
					if (e.type == SDL_RENDER_DEVICE_RESET) {
						tex->SetTexID(ImTextureID_Invalid);
						tex->SetStatus(ImTextureStatus_WantCreate);
					} else if (tex->Status == ImTextureStatus_OK) {
						tex->Updates.resize(0);
						tex->UpdateRect = {0, 0, static_cast<unsigned short>(tex->Width), static_cast<unsigned short>(tex->Height)};
						tex->Updates.push_back(tex->UpdateRect);
						tex->SetStatus(ImTextureStatus_WantUpdates);
					}
				}
				Ambience::g_fog_linear = Ambience::g_dot_linear = false;
			}
			// (ds-pc-port52) F2 in the launcher (not the Configurador, not while a key binding waits): full screen borderless
			// <-> a normal window with a title bar (moved, minimized and resized as any). The game's F2 menu is OnEvent's.
			if (!g_config_mode && e.type == SDL_KEYDOWN && e.key.repeat == 0 && e.key.keysym.sym == SDLK_F2 && !g_kbm_capture.load()) {
				static bool windowed = false;
				windowed             = !windowed;
				SDL_Rect desk {}, usable {};
				const bool has_desk = SDL_GetDisplayBounds(0, &desk) == 0 && desk.w > 0 && desk.h > 0;
				if (windowed) {
					int w = 1600, h = 900;
					if (SDL_GetDisplayUsableBounds(0, &usable) == 0 && usable.w > 0 && usable.h > 0) {
						const float k = std::min({1.0f, usable.w * 0.85f / w, usable.h * 0.85f / h});
						w             = static_cast<int>(w * k);
						h             = static_cast<int>(h * k);
					}
					SDL_SetWindowBordered(window, SDL_TRUE);
					SDL_SetWindowResizable(window, SDL_TRUE);
					SDL_SetWindowSize(window, w, h);
					SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
				} else {
					SDL_RestoreWindow(window); // (from maximized or minimized)
					SDL_SetWindowResizable(window, SDL_FALSE);
					SDL_SetWindowBordered(window, SDL_FALSE);
					if (has_desk) {
						SDL_SetWindowPosition(window, desk.x, desk.y);
						SDL_SetWindowSize(window, desk.w, desk.h);
					}
				}
				SDL_RaiseWindow(window);
				std::printf("[port] launcher: F2 -> %s\n", windowed ? "windowed" : "full screen borderless");
				std::fflush(stdout);
				continue;
			}
			if (g_config_mode && e.type == SDL_KEYDOWN && e.key.repeat == 0) {
				if (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_PAGEUP) Queue({Input::Tab, -1});
				if (e.key.keysym.sym == SDLK_e || e.key.keysym.sym == SDLK_PAGEDOWN) Queue({Input::Tab, +1});
			}
		}
		ImGui_ImplSDLRenderer2_NewFrame();
		ImGui_ImplSDL2_NewFrame();
		ImGui::NewFrame();
		if (g_config_mode) Draw(io.DisplaySize.x, io.DisplaySize.y);
		else DrawLauncher(io.DisplaySize.x, io.DisplaySize.y);
		if (!g_config_mode) {
			// (launcher1) The launcher rises out of black (0.6 s, ease-out cubic): the art, the fog and the column together.
			const float t = static_cast<float>(static_cast<double>(SDL_GetPerformanceCounter() - fade_t0) / pace_freq / 0.6);
			if (t < 1.0f)
				ImGui::GetForegroundDrawList()->AddRectFilled({0, 0}, io.DisplaySize,
				                                              IM_COL32(0, 0, 0, static_cast<int>(255.0f * (1.0f - Anim::EaseOut(t)))));
		}
		ImGui::Render();
		SDL_RenderSetScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
		ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
		SDL_RenderPresent(renderer);
		{
			// (launcher1) The pacing above: wait out the rest of this frame's slot, then let VSync align it.
			const double spent = static_cast<double>(SDL_GetPerformanceCounter() - pace_last) / pace_freq;
			if (const double rest = frame_s - spent; rest > 0.0005 && pace_wait != nullptr) {
				LARGE_INTEGER due {};
				due.QuadPart = -static_cast<LONGLONG>(rest * 1e7); // (relative, 100 ns units)
				if (SetWaitableTimer(pace_wait, &due, 0, nullptr, nullptr, FALSE) != 0) WaitForSingleObject(pace_wait, 100);
			}
			pace_last = SDL_GetPerformanceCounter();
		}
		// (des-fix-1110) The launcher's black-frame flicker, diagnosed: the illustration's texture not uploaded after the draw
		// (the frame showed only the clear colour). At most once a second.
		if (g_background != nullptr && g_background->Status != ImTextureStatus_OK) {
			static auto s_bg_logged = std::chrono::steady_clock::now() - std::chrono::seconds(2);
			if (const auto now = std::chrono::steady_clock::now(); now - s_bg_logged > std::chrono::seconds(1)) {
				s_bg_logged = now;
				std::printf("[port] launcher: the illustration's texture not ready after a frame (status %d)\n", static_cast<int>(g_background->Status));
				std::fflush(stdout);
			}
		}
	}
	if (pace_wait != nullptr) CloseHandle(pace_wait);
	if (g_start_result > 0) *settings = g_edit;
	// (ds-pc-port8) As Launcher-souls.ps1 (each change written at once): Salir / Esc keep the changes too.
	if (g_start_result < 0 && !g_config_mode) Save();
	FogGpu::Shutdown();
	FogGpu::g_renderer = nullptr;
	ImGui_ImplSDLRenderer2_Shutdown(); // (destroys the background's SDL texture too)
	ImGui_ImplSDL2_Shutdown();
	if (g_background != nullptr) {
		ImGui::UnregisterUserTexture(g_background);
		IM_DELETE(g_background);
	}
	Ambience::Shutdown(); // (des-fog-hq) fog and ash freed before the game starts
	FreeTexture(g_knight);
	g_knight_tried = false;
	LauncherTrophy::Shutdown(); // (trofeos-exe) its icon worker stopped, its atlas freed
	PadMap::Shutdown();         // (remap2) the controller drawings freed
	// (11-10) g_font_display too: it was left pointing into the destroyed launcher context, and the in-game menu (F2:
	// headers and cards through DisplayFont()) then used a freed ImFont - ImGui wrote through it into freed memory
	// (heap corruption: the game crashing in its job worker or the GPU lost right after the menu opened, or a crash
	// in ImFont::GetFontBaked, exe+0x1ff83c). The menu falls back to its current font.
	g_font_body = g_font_semi = g_font_serif = g_font_display = nullptr;
	ImGui::DestroyContext(context);
	ImGui::SetCurrentContext(previous);
	g_background       = nullptr;
	g_background_tried = false;
	g_start_mode       = false;
	g_status.clear();
	{
		std::scoped_lock lock(g_mutex);
		g_inputs.clear();
	}
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER | SDL_INIT_EVENTS);
	return g_start_result > 0;
#else
	(void)runtime;
	(void)settings;
	return true;
#endif
}

void RunConfigurator(const Runtime& runtime, Settings* settings) {
	// (10-28) The start menu's window and loop, showing the Configurador page only.
	g_config_mode = true;
	g_cfg_why.clear();
	RunStartMenu(runtime, settings);
	g_config_mode = false;
	g_cfg_why.clear();
}

} // namespace Port
