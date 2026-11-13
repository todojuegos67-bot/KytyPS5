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

#include "local/perf-overlay.h"
#include "local/local-platform.h"
#include "local/ram-guard.h"
#include "local/port-live.h"
#include "graphics/presentation/window/hostInput.h"
#include "libs/trophy.h"

#if defined(_WIN32)
// (11-10) Before SDL.h (its SDL_system.h only declares ID3D11Device): the launcher fog (ds-fog.inc) uses the full types.
#include <d3d11.h>
#include <psapi.h>
#include <d3dcompiler.h>
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
	enum Kind { Key, Analog, MousePos, MouseButton, Wheel, Tab } kind;
	int   key  = 0;
	bool  down = false;
	float x = 0, y = 0;
};

Runtime                  g_rt;
Settings                 g_edit;       // the menu's copy (present thread)
Settings                 g_saved;      // what settings.ini holds now

// (ds-pc-port13) Preparar shaders: all CPU threads or relaxed (asked the first time, then settings.ini's
// precompilar_hilos; Opciones > Sistema changes it), then the precompile.
void StartShaders() {
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
std::atomic<bool>        g_focus_trophy_btn_request {false}; // (port34) window thread: back from it, focus its button
int                      g_trophy_tab_delta = 0;     // (port34) present thread: L1/R1, Q/E on the trophy page
bool                     g_focus_trophy_btn = false;
std::atomic<uint64_t>    g_revision {1};
std::atomic<bool>        g_reset_edit {false};
std::mutex               g_mutex;
std::deque<Input>        g_inputs;
int                      g_passthrough = 0; // window thread: zeroed axes sent to the game as the menu opened
int                      g_cursor_before = -1;
constexpr int            kTabCount = 7; // (10-27c) Gráficos, Pantalla, Sonido, Controles, Sistema, Trofeos; (ds-pc-port37) Ajustes del juego
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
	g_popup       = false;
	g_page_scroll = 0;
	g_pages_reset = true;
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
		default: return ImGuiKey_None;
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
	switch (e.type) {
		case SDL_KEYDOWN:
			// (ds-pc-port41) F8: the Metricas panel (metricas.ps1's, top-centre), not the emulator's overlay.
			if (e.key.keysym.sym == SDLK_F8) {
				if (e.key.repeat == 0) MetricsPanel::Toggle();
				return true;
			}
			// (11-09) The menu key is F2 now (F1 made the game close at times): F1 and F2's repeats never reach the game.
			if (e.key.keysym.sym == SDLK_F1 || (e.key.keysym.sym == SDLK_F2 && e.key.repeat != 0)) return true;
			if (e.key.keysym.sym == SDLK_F2) {
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
			if (e.key.keysym.sym == SDLK_ESCAPE && e.key.repeat == 0) {
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
		case SDL_TEXTINPUT:
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
			// (ds-pc-port37) The F2 pages: Circle back to the F2 menu, L1 scrolls up (L2, a trigger, below).
			if (g_f1_page.load() != 0 && !g_popup.load()) {
				if (b == SDL_CONTROLLER_BUTTON_B) {
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

void DrainInputs() {
	std::deque<Input> inputs;
	{
		std::scoped_lock lock(g_mutex);
		inputs.swap(g_inputs);
	}
	auto& io = ImGui::GetIO();
	for (const auto& in: inputs) {
		switch (in.kind) {
			case Input::Key: io.AddKeyEvent(static_cast<ImGuiKey>(in.key), in.down); break;
			case Input::Analog: io.AddKeyAnalogEvent(static_cast<ImGuiKey>(in.key), in.down, in.x); break;
			case Input::MousePos: io.AddMousePosEvent(in.x, in.y); break;
			case Input::MouseButton: io.AddMouseButtonEvent(in.key, in.down); break;
			case Input::Wheel: io.AddMouseWheelEvent(in.x, in.y); break;
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

// ---- (ds-pc-port37) "Mapear mando": the emulator launcher's pad mapper (Launcher-souls.ps1 Show-PadMapper) in the menu. ----
// The same drawings (mando-ps5.png: the game's PS5 controls; mando-8bitdo.png: your controller's physical buttons, SDL names)
// and the same button centres measured on them (350 px high). Pick a physical button (on the right drawing or in the list),
// then the PS5 control it does (on the left drawing or in its list). Saved as mapa_mando (KYTY_PAD_MAP), empty = default.
namespace PadMap {
struct Spot {
	const char* name;
	float       x, y, d, h; // centre and size in the drawing's pixels (h = d but the touchpad's)
};
const char* const kPhys[] = {"a", "b", "x", "y", "leftshoulder", "rightshoulder", "lefttrigger", "righttrigger",
                             "leftstick", "rightstick", "start", "back", "dpup", "dpdown", "dpleft", "dpright"};
const char* const kDefault[] = {"Cross", "Circle", "Square", "Triangle", "L1", "R1", "L2", "R2",
                                "L3", "R3", "Options", "None", "Up", "Down", "Left", "Right"};
const char* const kPs5[] = {"Cross", "Circle", "Square", "Triangle", "L1", "R1", "L2", "R2", "L3", "R3",
                            "Options", "TouchPad", "Up", "Down", "Left", "Right", "None"};
const Spot kPsSpots[] = {{"Up", 96, 62, 29, 29},        {"Down", 96, 126, 29, 29},    {"Left", 65, 93, 29, 29},
                         {"Right", 128, 93, 29, 29},    {"Triangle", 435, 60, 38, 38}, {"Circle", 472, 94, 38, 38},
                         {"Cross", 435, 131, 38, 38},   {"Square", 398, 94, 38, 38},   {"L3", 178, 160, 67, 67},
                         {"R3", 346, 160, 67, 67},      {"Options", 389, 32, 20, 20},  {"TouchPad", 265, 52, 150, 80}};
const Spot kXbSpots[] = {{"y", 416, 58, 38, 38},        {"x", 377, 95, 38, 38},        {"b", 456, 95, 38, 38},
                         {"a", 416, 132, 38, 38},       {"leftstick", 125, 86, 72, 72}, {"rightstick", 339, 156, 72, 72},
                         {"dpup", 193, 140, 25, 25},    {"dpdown", 193, 192, 25, 25},  {"dpleft", 167, 166, 25, 25},
                         {"dpright", 219, 166, 25, 25}, {"back", 226, 89, 22, 22},     {"start", 312, 89, 22, 22}};

const char* Label(const std::string& c) {
	static const std::pair<const char*, const char*> names[] = {
	    {"Cross", "Cruz"}, {"Circle", "Círculo"}, {"Square", "Cuadrado"}, {"Triangle", "Triángulo"}, {"Options", "Options"},
	    {"TouchPad", "Panel táctil"}, {"Up", "Cruceta arriba"}, {"Down", "Cruceta abajo"}, {"Left", "Cruceta izquierda"},
	    {"Right", "Cruceta derecha"}, {"None", "(nada)"}, {"a", "A"}, {"b", "B"}, {"x", "X"}, {"y", "Y"},
	    {"leftshoulder", "LB / L1"}, {"rightshoulder", "RB / R1"}, {"lefttrigger", "LT / L2"}, {"righttrigger", "RT / R2"},
	    {"leftstick", "Stick izq. (L3)"}, {"rightstick", "Stick der. (R3)"}, {"start", "Start"}, {"back", "Select / Back"},
	    {"dpup", "Cruceta arriba"}, {"dpdown", "Cruceta abajo"}, {"dpleft", "Cruceta izquierda"}, {"dpright", "Cruceta derecha"}};
	for (const auto& [k, v]: names)
		if (c == k) return I18n::T(v);
	return c.c_str();
}

struct Tex {
	ImTextureData* tex = nullptr;
	int            w = 0, h = 0;
	ImGuiContext*  ctx = nullptr;
	bool           tried = false;
};
Tex g_ps, g_xb;

void Load(Tex& t, const wchar_t* file) {
	if (t.tex != nullptr && t.ctx != ImGui::GetCurrentContext()) t = {}; // (another context: its backend owns the old one)
	if (t.tried) return;
	t.tried = true;
	std::ifstream in(g_rt.exe_path.parent_path() / file, std::ios::binary);
	const std::string bytes {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
	int   channels = 0;
	auto* px = bytes.empty() ? nullptr
	                         : stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()), &t.w,
	                                                 &t.h, &channels, 4);
	if (px == nullptr) return;
	t.tex = IM_NEW(ImTextureData)();
	t.tex->Create(ImTextureFormat_RGBA32, t.w, t.h);
	std::memcpy(t.tex->GetPixels(), px, static_cast<size_t>(t.w) * t.h * 4);
	t.tex->UseColors = true;
	stbi_image_free(px);
	ImGui::RegisterUserTexture(t.tex);
	t.ctx = ImGui::GetCurrentContext();
}

std::map<std::string, std::string> Parse(const std::string& text) {
	std::map<std::string, std::string> m;
	for (size_t i = 0; i < std::size(kPhys); i++) m[kPhys[i]] = kDefault[i];
	size_t at = 0;
	while (at <= text.size()) {
		const size_t end  = std::min(text.find(';', at), text.size());
		const auto   item = text.substr(at, end - at);
		at                = end + 1;
		const size_t eq   = item.find('=');
		if (eq != std::string::npos && m.contains(item.substr(0, eq))) m[item.substr(0, eq)] = item.substr(eq + 1);
	}
	return m;
}

std::string Format(const std::map<std::string, std::string>& m) {
	bool same = true;
	for (size_t i = 0; i < std::size(kPhys); i++) same = same && m.at(kPhys[i]) == kDefault[i];
	if (same) return {};
	std::string out;
	for (const char* k: kPhys) out += (out.empty() ? "" : ";") + std::string(k) + "=" + m.at(k);
	return out;
}

std::string g_sel = "a"; // the physical button being mapped

// Draws one drawing with its spots at `at`, `scale`; returns the spot clicked ("" none). `hot(name)`: highlighted.
template <typename Hot>
std::string Drawing(const char* id, Tex& t, const Spot* spots, size_t n, ImVec2 at, float scale, Hot hot) {
	auto*       dl = ImGui::GetWindowDrawList();
	const float w = t.w > 0 ? t.w * scale : 540 * scale, h = 350 * scale;
	if (t.tex != nullptr) dl->AddImage(t.tex->GetTexRef(), at, {at.x + w, at.y + h});
	else dl->AddRectFilled(at, {at.x + w, at.y + h}, IM_COL32(40, 44, 54, 255), 12 * scale);
	std::string clicked;
	for (size_t i = 0; i < n; i++) {
		const Spot& sp = spots[i];
		const ImVec2 c {at.x + sp.x * scale, at.y + sp.y * scale};
		const ImVec2 half {sp.d * scale * 0.5f, sp.h * scale * 0.5f};
		ImGui::SetCursorScreenPos({c.x - half.x, c.y - half.y});
		ImGui::PushID(id);
		ImGui::PushID(static_cast<int>(i));
		const bool pressed = ImGui::InvisibleButton("##spot", {std::max(1.0f, half.x * 2), std::max(1.0f, half.y * 2)});
		const bool over    = ImGui::IsItemHovered();
		ImGui::PopID();
		ImGui::PopID();
		const bool  on   = hot(sp.name);
		const ImU32 ring = on ? IM_COL32(236, 200, 120, 255) : over ? IM_COL32(236, 200, 120, 200) : IM_COL32(236, 200, 120, 90);
		if (on || over) {
			if (sp.d != sp.h) dl->AddRectFilled({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y}, IM_COL32(236, 200, 120, 90), 8 * scale);
			else dl->AddCircleFilled(c, half.x, IM_COL32(236, 200, 120, 90));
		}
		if (sp.d != sp.h) dl->AddRect({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y}, ring, 8 * scale, 0, on ? 3.0f : 1.5f);
		else dl->AddCircle(c, half.x, ring, 0, on ? 3.0f : 1.5f);
		if (over) ImGui::SetTooltip("%s", Label(sp.name));
		if (pressed) clicked = sp.name;
	}
	ImGui::SetCursorScreenPos({at.x, at.y + h});
	return clicked;
}

// The section; true when the mapping changed.
bool Draw(Settings& s, float unit) {
	Load(g_ps, L"mando-ps5.png");
	Load(g_xb, L"mando-8bitdo.png");
	auto m       = Parse(s.pad_map);
	bool changed = false;
	ImGui::TextWrapped("%s", I18n::T("1. Elige un botón de TU mando (dibujo de la derecha o la lista). 2. Elige qué botón del juego hace (dibujo de la izquierda). Se aplica al reiniciar el juego."));
	const float avail = ImGui::GetContentRegionAvail().x;
	const float scale = std::min(unit, (avail - 16 * unit) / (2 * 540.0f));
	const ImVec2 at   = ImGui::GetCursorScreenPos();
	ImGui::TextColored(kGold, "%s", I18n::T("Botón del juego (PS5)"));
	ImGui::SetCursorScreenPos({at.x + 540 * scale + 16 * unit, at.y});
	ImGui::TextColored(kGold, "%s", I18n::T("Tu mando"));
	const float top = at.y + ImGui::GetTextLineHeightWithSpacing();
	const std::string target = m[g_sel];
	const std::string ps = Drawing("ps", g_ps, kPsSpots, std::size(kPsSpots), {at.x, top}, scale, [&](const char* n) { return target == n; });
	const std::string xb = Drawing("xb", g_xb, kXbSpots, std::size(kXbSpots), {at.x + 540 * scale + 16 * unit, top}, scale,
	                               [&](const char* n) { return g_sel == n; });
	if (!xb.empty()) g_sel = xb;
	if (!ps.empty() && m[g_sel] != ps) {
		m[g_sel] = ps;
		changed  = true;
	}
	ImGui::SetCursorScreenPos({at.x, top + 350 * scale + 8 * unit});
	// The shoulders and triggers (not on the drawings) and every button: the list.
	if (ImGui::BeginTable("##padmap", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame)) {
		for (const char* k: kPhys) {
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			if (ImGui::Selectable(Label(k), g_sel == k, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) g_sel = k;
			ImGui::TableSetColumnIndex(1);
			ImGui::PushID(k);
			ImGui::SetNextItemWidth(-1);
			if (ImGui::BeginCombo("##to", Label(m[k]))) {
				for (const char* c: kPs5)
					if (ImGui::Selectable(Label(c), m[k] == c)) {
						m[k]    = c;
						g_sel   = k;
						changed = true;
					}
				ImGui::EndCombo();
			}
			ImGui::PopID();
		}
		ImGui::EndTable();
	}
	if (ImGui::Button(I18n::T("Restablecer los botones de serie"))) {
		m       = Parse("");
		changed = true;
	}
	if (changed) s.pad_map = Format(m);
	return changed;
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
enum Kind { Back, Trophy, Pad, Keyboard, Power, Monitor, Layers, Sliders, Sun, Gauge, Chip, Gear, Speaker, Flame, Eye };

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
	}
}

// The icon of a section by its (Spanish) title; -1 when it has none.
int ForSection(const char* title) {
	static const std::pair<const char*, Kind> map[] = {
	    {"Imagen", Monitor},        {"Frame generation", Layers}, {"Calidad gráfica", Sliders}, {"Ray tracing", Sun},
	    {"Modos Souls", Flame},     {"Pantalla", Monitor},        {"Rendimiento", Gauge},       {"Controles", Pad},
	    {"Tu equipo", Chip},        {"Configurador", Gear},       {"Ajustes del juego", Gear},  {"Sonido", Speaker},
	    {"Trofeos", Trophy},        {"Gráficos", Eye},            {"Sistema", Chip}};
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

// (bb-style-f1) The F2 menu's top-level options as Bloodborne's hub cards, side by side: a dark plate with a bronze
// border, a warm light from above when focused, a medallion with the icon (halo, turning beads), the label in Cinzel,
// letter-spaced and centred (gold, parchment when focused), an ornamental rule under it; focus lifts the card 16 px and
// grows it 4 %, a shine sweeps once; on open the cards rise in one after another (ease-out-back, `index` x 70 ms).
// An ImGui button underneath: mouse click, and D-pad / stick / arrow keys left-right move between the cards (one row).
bool CardButton(const char* label, Icons::Kind icon, ImVec2 size, float unit, int index, bool danger = false) {
	static std::map<ImGuiID, double> focus_at;
	const float k   = Anim::Progress(Anim::g_opened_at.load() + 0.07 * index, 0.5);
	if (k < 1.0f) Bump();
	const float x1  = k - 1.0f, c1 = 1.70158f, c3 = c1 + 1.0f;
	const float ent = 1.0f + c3 * x1 * x1 * x1 + c1 * x1 * x1; // EaseOutBack
	ImGui::PushID(label);
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0, 0, 0, 0});
	ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4 {0, 0, 0, 0}); // (the card's own focus look)
	const bool pressed = ImGui::Button("##card", size);
	ImGui::PopStyleColor(4);
	const ImGuiID id = ImGui::GetItemID();
	const bool    on = ImGui::IsItemHovered() || ImGui::IsItemFocused() || ImGui::IsItemActive();
	const float   v  = Anim::Hot(id, on);
	if (on && v < 0.05f) focus_at[id] = Anim::Now();
	const ImVec2 ra = ImGui::GetItemRectMin(), rb = ImGui::GetItemRectMax();
	const float  cw = rb.x - ra.x, ch = rb.y - ra.y;
	const float  lift = -16.0f * unit * v + (1.0f - ent) * 80.0f * unit;
	const float  sc   = 1.0f + 0.04f * v;
	const ImVec2 c {(ra.x + rb.x) * 0.5f, (ra.y + rb.y) * 0.5f + lift};
	const ImVec2 a {c.x - cw * 0.5f * sc, c.y - ch * 0.5f * sc}, b {c.x + cw * 0.5f * sc, c.y + ch * 0.5f * sc};
	auto*        dl = ImGui::GetWindowDrawList();
	const float  al = ImGui::GetStyle().Alpha * std::clamp(k * 1.6f, 0.0f, 1.0f);
	const auto   G  = [&](float x) { return IM_COL32(228, 192, 122, static_cast<int>(255.0f * std::clamp(x * al, 0.0f, 1.0f))); };
	const auto   Br = [&](float x) { return IM_COL32(158, 112, 62, static_cast<int>(255.0f * std::clamp(x * al, 0.0f, 1.0f))); };
	const auto   Pa = [&](float x) { return IM_COL32(236, 228, 212, static_cast<int>(255.0f * std::clamp(x * al, 0.0f, 1.0f))); };
	const auto   Cl = [&](int r, int g, int bb, float x) { return IM_COL32(r, g, bb, static_cast<int>(255.0f * std::clamp(x * al, 0.0f, 1.0f))); };
	const float  rr = 6.0f * unit;
	// The plate (Bloodborne's Panel: dark, a soft shadow, a bronze frame warming with the focus, an inner hairline).
	for (int s = 3; s >= 1; s--) dl->AddRectFilled({a.x - s * 3 * unit, a.y - s * 2 * unit}, {b.x + s * 3 * unit, b.y + s * 5 * unit}, Cl(0, 0, 0, 0.12f), rr + s * 3 * unit);
	dl->AddRectFilledMultiColor(a, b, Cl(22, 19, 18, 0.94f), Cl(22, 19, 18, 0.94f), Cl(12, 10, 10, 0.96f), Cl(12, 10, 10, 0.96f));
	if (danger && v > 0.0f) dl->AddRectFilled(a, b, Cl(120, 40, 30, 0.22f * v), rr);
	if (ImGui::IsItemActive()) dl->AddRectFilled(a, b, G(0.10f), rr);
	dl->AddRect(a, b, danger ? Cl(230, 120, 100, 0.45f + 0.5f * v) : (v > 0.5f ? G(0.6f + 0.4f * v) : Br(0.75f)), rr, 0, std::max(1.0f, 1.4f * unit));
	dl->AddRect({a.x + 6 * unit, a.y + 6 * unit}, {b.x - 6 * unit, b.y - 6 * unit}, G(0.10f + 0.20f * v), rr * 0.6f, 0, std::max(1.0f, 0.8f * unit));
	if (v > 0.01f) dl->AddRectFilledMultiColor({a.x + 8 * unit, a.y + 8 * unit}, {b.x - 8 * unit, a.y + ch * 0.45f}, G(0.10f * v), G(0.10f * v), G(0.0f), G(0.0f));
	// The medallion and its icon.
	const ImVec2 m {c.x, a.y + (b.y - a.y) * 0.36f};
	const float  mr = std::min(cw * 0.28f, ch * 0.20f) * sc;
	const float  t  = Souls::Time();
	for (int g = 4; g >= 1; g--) dl->AddCircleFilled(m, mr + g * 6 * unit, G(0.025f * (0.3f + v)), 48);
	dl->AddCircleFilled(m, mr, Cl(14, 12, 12, 0.95f), 64);
	dl->AddCircle(m, mr, Br(0.9f), 64, std::max(1.0f, 1.6f * unit));
	dl->AddCircle(m, mr - 7 * unit, G(0.35f + 0.5f * v), 64, std::max(1.0f, unit));
	for (int d = 0; d < 24; d++) {
		const float ang = d * 0.2618f + t * 0.25f * (0.3f + v);
		dl->AddCircleFilled({m.x + std::cos(ang) * (mr + 10 * unit), m.y + std::sin(ang) * (mr + 10 * unit)}, 1.4f * unit, G(0.25f + 0.6f * v));
	}
	const ImU32 ic = danger ? Cl(236, 140, 110, 0.75f + 0.25f * v) : G(0.75f + 0.25f * v);
	Icons::Draw(dl, icon, {m.x + 1, m.y + 1}, mr * 0.85f, Cl(0, 0, 0, 0.6f));
	Icons::Draw(dl, icon, m, mr * 0.85f, ic);
	// The label: Cinzel (Demon's own bundled copy, the font Bloodborne's titles use), letter-spaced, fitted to the card.
	ImFont* f   = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
	float   px  = std::round(28.0f * unit);
	const float sp = 2.0f * unit;
	const auto  spaced_w = [&](float size) {
		float w = 0;
		for (const char* p = label; *p;) {
			const unsigned char u8 = static_cast<unsigned char>(*p);
			const int n = u8 < 0x80 ? 1 : (u8 >> 5) == 6 ? 2 : (u8 >> 4) == 14 ? 3 : 4;
			w += f->CalcTextSizeA(size, FLT_MAX, 0.0f, p, p + n).x + sp;
			p += n;
		}
		return std::max(0.0f, w - sp);
	};
	const float maxw = cw - 28.0f * unit;
	if (const float w = spaced_w(px); w > maxw) px = std::max(12.0f, std::floor(px * maxw / w));
	const float ty = a.y + (b.y - a.y) * 0.66f;
	float       x  = c.x - spaced_w(px) * 0.5f;
	const ImU32 tc = danger ? Cl(240, 150, 125, 1.0f) : (v > 0.5f ? Pa(1.0f) : G(1.0f));
	for (const char* p = label; *p;) {
		const unsigned char u8 = static_cast<unsigned char>(*p);
		const int n = u8 < 0x80 ? 1 : (u8 >> 5) == 6 ? 2 : (u8 >> 4) == 14 ? 3 : 4;
		dl->AddText(f, px, {std::floor(x + 1), std::floor(ty + 1)}, Cl(0, 0, 0, 0.7f), p, p + n);
		dl->AddText(f, px, {std::floor(x), std::floor(ty)}, tc, p, p + n);
		x += f->CalcTextSizeA(px, FLT_MAX, 0.0f, p, p + n).x + sp;
		p += n;
	}
	// The ornamental rule (Bloodborne's OrnLine: a line fading out each side, a diamond in the middle).
	{
		const float oy = ty + px + 18.0f * unit, hw = cw * 0.32f * 0.5f, oa = 0.5f + 0.5f * v, d = 4.0f * unit;
		dl->AddRectFilledMultiColor({c.x - hw, oy - 0.5f * unit}, {c.x - d * 1.6f, oy + 0.5f * unit}, Br(0.0f), Br(oa), Br(oa), Br(0.0f));
		dl->AddRectFilledMultiColor({c.x + d * 1.6f, oy - 0.5f * unit}, {c.x + hw, oy + 0.5f * unit}, Br(oa), Br(0.0f), Br(0.0f), Br(oa));
		dl->AddQuadFilled({c.x, oy - d}, {c.x + d, oy}, {c.x, oy + d}, {c.x - d, oy}, G(oa));
	}
	// A shine sweeping once across on focus.
	const float fs = static_cast<float>(Anim::Now() - focus_at[id]);
	if (on && fs < 0.9f) {
		Bump();
		const float sx = a.x - cw * 0.5f + (fs / 0.9f) * cw * 2.0f;
		dl->PushClipRect(a, b, true);
		dl->AddQuadFilled({sx, a.y}, {sx + cw * 0.18f, a.y}, {sx - cw * 0.22f, b.y}, {sx - cw * 0.40f, b.y}, Cl(255, 240, 200, 0.07f));
		dl->PopClipRect();
	}
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
	if (r.tag == cur) SetUpd(UpdateUi::UpToDate, both + "\nYa tienes la ultima version.");
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

// (ds-pc-port17) At start: a silent check; a newer published port version shows a notice right away (update or later).
std::atomic<bool> g_notice_ready {false};
bool              g_notice_done = false;
std::string       g_notice_text;

void StartupCheck() {
#if defined(_WIN32)
	static std::once_flag once;
	std::call_once(once, [] {
		std::thread([] {
			Update::Release r;
			std::string     error;
			if (!Update::CheckLatest(&r, &error)) return;
			const std::string cur = Update::CurrentVersion();
			const std::string dc = Update::TagDate(cur), dr = Update::TagDate(r.tag);
			if (r.tag == cur || (!dc.empty() && !dr.empty() && dc > dr) || (r.published.size() >= 19 && r.published.substr(0, 19) <= Update::BuildIso())) return;
			{
				std::scoped_lock lock(g_upd.mutex);
				if (g_upd.busy.load()) return;
				g_upd.release = r;
				g_notice_text = std::string(I18n::T("Hay una version nueva: ")) + (r.name.empty() ? r.tag : r.name + " (" + r.tag + ")") +
				                I18n::T(". Tienes: ") + cur + ".";
			}
			g_notice_ready = true;
			Bump();
		}).detach();
	});
#endif
}

void UpdateNotice(float width, float height) {
	if (g_notice_done || !g_notice_ready.load()) return;
	std::string text;
	{
		std::scoped_lock lock(g_upd.mutex);
		text = g_notice_text;
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
	if (ImGui::Button(I18n::T("Actualizar ahora"), {220 * g_unit, 40 * g_unit})) {
		g_notice_done = true;
		g_tab_request = 4;
		SetUpd(UpdateUi::Working, "Descargando...");
		StartWorker(InstallWork);
	}
	ImGui::SameLine();
	if (ImGui::Button(I18n::T("Mas tarde"), {160 * g_unit, 40 * g_unit})) g_notice_done = true;
	ImGui::End();
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
		ids.emplace_back(z);
		labels.emplace_back(z);
	}
	std::string value = s.fullscreen ? s.resolution : std::to_string(s.width) + "x" + std::to_string(s.height);
	if (std::find(ids.begin(), ids.end(), value) == ids.end()) {
		ids.push_back(value);
		labels.push_back(value);
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
	const char* game_tabs[kTabCount]     = {"Gráficos", "Pantalla", "Sonido", "Controles", "Sistema", "Trofeos", "Ajustes del juego"};
	const char* launcher_tabs[kTabCount] = {"Imagen", "Calidad gráfica", "Ray tracing", "Controles", "Sistema", "Trofeos", "Ajustes del juego"};
	const char* const* tabs = g_start_mode ? launcher_tabs : game_tabs;
	static const std::vector<int> game_groups[kTabCount] = {{0, 1, 2, 3}, {5}, {}, {7}, {6, 8}, {}, {10}}; // (10-28) no Modos Souls
	static const std::vector<int> launcher_groups[kTabCount] = {{5, 0, 1}, {2}, {3}, {7}, {9, 6, 8}, {}, {10}};
	const std::vector<int>* groups = g_start_mode ? launcher_groups : game_groups;
	static const char* section_names[11] = {"Imagen", "Frame generation", "Calidad gráfica", "Ray tracing", "Modos Souls",
	                                        "Pantalla", "Rendimiento", "Controles", "Tu equipo", "Configurador", "Ajustes del juego"};
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
						                                                              : s.souls_mode == "low" ? "Modo Souls low profile: frame generation fija en AMD x2 (30 reales, 60 en pantalla)." : s.souls_mode == "ligero-plus" ? "Modo Souls ligero +: frame generation fija en AMD x2 (35 reales, 70 en pantalla)." : "Modo Souls ligero: frame generation fija en AMD x2 (30 reales, 60 en pantalla).");
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
					ChoiceRow("Luz RT", s.rt_ssgi != l.rt_ssgi, kRtLightLevels, kRtLightLevelCount, s.rt_ssgi,
					          "Iluminación por rayos sobre la imagen (luz rebotada y sombreado de contacto). Solo funciona con DLSS, XeSS o FSR. Cuesta FPS. Experimental.");
					if (ChoiceRow("Luz volumétrica", changed_not_live(l.godrays != s.godrays, 4), kRtLevels, kRtLevelCount, s.godrays,
					              "Luz volumétrica por rayos: rayos de sol que entran por ventanas y puertas, y halo de luz en la niebla alrededor de antorchas. Solo con DLSS, XeSS o FSR. Bajo ~0,4 ms, Medio ~0,8 ms, Alto ~1,2 ms. Apagarla es inmediato."))
						live_changed = true;
					ChoiceRow("Partículas volumétricas", s.motes != l.motes, kMotesLevels, kMotesLevelCount, s.motes, "Motas de polvo en suspensión iluminadas por los rayos de sol y el halo de antorchas y lámparas (la luz volumétrica); flotan despacio y quedan fijas en el mundo. Sutil / Media / Densa: cuántas. Solo con DLSS, XeSS o FSR. ~0,2-0,5 ms."); // (motes-1031)
					// (ds-pc-port9 / ds-pc-port10) No Path tracing, Luz de antorchas RT or Destellos de lente in the port.
					ChoiceRow("Humo y partículas", s.smoke != l.smoke, kSmokeLevels, kSmokeLevelCount, s.smoke,
					          "Humo y partículas más realistas sobre las del propio juego: Suave = el humo, la niebla de partículas y las chispas ya no cortan en línea dura donde tocan suelo y paredes (se funden en ~35 cm). Solo con DLSS, XeSS o FSR. ~0,3 ms.");
					// (ds-pc-port9) No Sombras RT in the port.
					ImGui::EndDisabled();
					CheckRow("RT por hardware (experimental)", s.rt_hw_exp != l.rt_hw_exp, s.rt_hw_exp,
					         "Activa las opciones de trazado de rayos por hardware (Reflejos y Sombras sol por RT real). Experimental: puede tener fallos y cuesta mucha GPU y VRAM. Necesita GPU con RT.");
					if (s.rt_hw_exp) {
						ImGui::BeginDisabled(s.pure_game);
						{
							std::string x = s.rt_reflections ? "1" : "0";
							if (ChoiceRow("Reflejos (RT real)", changed_not_live(s.rt_reflections != l.rt_reflections || (s.rt_reflections && !l.rt_hw_exp), 1), kYesNo, kYesNoCount, x,
							              "Reflejos con trazado de rayos: refleja también lo que está fuera de pantalla o tapado, con sus texturas. Apagarlo es inmediato.")) {
								s.rt_reflections = x == "1";
								live_changed     = true;
							}
						}
						const bool sun_was = s.rt_sun;
						{
							std::string x = s.rt_sun ? "1" : "0";
							if (ChoiceRow("Sombras sol (RT real)", s.rt_sun != l.rt_sun || (s.rt_sun && !l.rt_hw_exp), kYesNo, kYesNoCount, x,
							              "Sombras del sol con trazado de rayos: sombras de contacto y las que faltaban. No se puede usar a la vez que la Luz volumétrica ni las Sombras RT."))
								s.rt_sun = x == "1";
						}
						// (vol25) Luz volumetrica / Sombras RT and Sombras sol share the engine's sun pass: choosing one turns the other off.
						if (s.rt_sun && !sun_was) s.godrays = "0", s.rt_shadows = "0";
						ImGui::EndDisabled();
					}
					if (s.rt_sun && s.rt_hw_exp && (s.godrays != "0" || s.rt_shadows != "0")) s.rt_sun = false;
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
						CheckRow("Ahorro de memoria RAM", s.ram16_boost != g_rt.launched.ram16_boost, s.ram16_boost,
						         "Perfil de memoria de 16 GB (6 GB fijos en RAM, la precarga espera si queda poca memoria). Pensado para PCs de 16 GB de RAM. No cambia los gráficos.");
					s.profile16 = LowRam(s, g_rt.pc);
					if (s.ram16_boost) ImGui::TextColored(kGold, "RAM %s: mejoras activadas. Cierra los navegadores mientras juegas.", s.simulate_ram16 ? "16 GB (simulado)" : (std::to_string(g_rt.pc.ram_gb) + " GB").c_str());
					CheckRow("Modo GPU seguro", true, s.safe_gpu, "Para gráficas que se cuelgan o dan pantallazo azul: más estable, ~37% menos FPS.");
					CheckRow("Usar todos los CCD (Ryzen de dos CCD)", true, s.all_ccds, "Ryzen de dos CCD: No = el juego en el CCD con más caché L3 (más rápido); Sí = todos los núcleos.");
					// (perfopts-1107) Opciones de rendimiento (upstream's opt-in switches; at the next start).
					ImGui::Spacing();
					ImGui::Dummy({16.0f * g_unit, 0});
					ImGui::SameLine();
					ImGui::TextColored(kGold, "%s", I18n::T("Opciones de rendimiento"));
					CheckRow("Registros PM4 directos", s.perf_pm4_regfile != g_rt.launched.perf_pm4_regfile, s.perf_pm4_regfile,
					         "Guarda directamente los registros de la GPU que no necesitan procesarse (comprobado al empezar). Menos trabajo de CPU por dibujo. Experimental (KYTY_PM4_REGFILE).");
					CheckRow("Subidas a la GPU en dos hilos", s.perf_upload_lanes != g_rt.launched.perf_upload_lanes, s.perf_upload_lanes,
					         "Reparte la protección de memoria de las subidas a la GPU entre dos hilos. Ayuda en zonas con mucha carga de texturas en CPUs de 8 núcleos o más. Experimental (KYTY_UPLOAD_LANES).");
					CheckRow("El render ayuda con las subidas", s.perf_upload_help != g_rt.launched.perf_upload_help, s.perf_upload_help,
					         "Si la CPU va cargada, el hilo de render hace él mismo las subidas pendientes en vez de esperar. Bueno en CPUs de 4-6 núcleos y portátiles (KYTY_UPLOAD_HELP).");
					CheckRow("Hilos sin espera activa con CPU cargada", s.perf_spin_gate != g_rt.launched.perf_spin_gate, s.perf_spin_gate,
					         "Los hilos de ayuda dejan de girar en vacío cuando quedan menos de 3 núcleos libres, dejando la CPU al juego. Para CPUs justas (KYTY_SPIN_GATE).");
					CheckRow("Copias de la GPU anticipadas", s.perf_eager_feedback != g_rt.launched.perf_eager_feedback, s.perf_eager_feedback,
					         "Devuelve a la memoria del juego las copias ya terminadas de la GPU antes de que las pida, evitando esperas. Puede quitar tirones. Experimental (KYTY_EAGER_FEEDBACK).");
					CheckRow("Copias de la GPU agrupadas", s.perf_feedback_batch != g_rt.launched.perf_feedback_batch, s.perf_feedback_batch,
					         "Graba juntas las copias de la GPU de cada bloque de comandos en vez de una a una. Experimental (KYTY_FEEDBACK_BATCH).");
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
					if (ImGui::CollapsingHeader(I18n::T("Mapear mando"))) {
						(void)PadMap::Draw(s, unit);
						if (s.pad_map != l.pad_map && !g_start_mode)
							ImGui::TextColored(ImVec4 {0.62f, 0.58f, 0.50f, 1.0f}, "%s", I18n::T("   (requiere reinicio)"));
					}
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

const ImVec4 kInkL {0.97f, 0.96f, 0.93f, 1.0f};
const ImVec4 kDimL {0.86f, 0.84f, 0.80f, 1.0f}; // (ds-pc-port9) secondary text, still high contrast
const ImVec4 kGoldL {0.87f, 0.77f, 0.55f, 1.0f};

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
	ImGui::PushFont(g_font_display, U(card ? 25 : 20));
	ImGui::TextColored(kGoldL, "%s", I18n::T(title));
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
bool LButton(const char* text, ImVec2 at, ImVec2 size, bool primary, ImFont* font, float font_size, const char* tip = nullptr) {
	ImGui::SetCursorScreenPos(at);
	if (primary) {
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.69f, 0.55f, 0.27f, 0.92f});
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.81f, 0.67f, 0.38f, 0.96f});
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0.60f, 0.47f, 0.22f, 1.0f});
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4 {0.08f, 0.06f, 0.04f, 1.0f});
		ImGui::PushStyleColor(ImGuiCol_Border, kGoldL);
	} else {
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.094f, 0.11f, 0.15f, 0.82f});
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.17f, 0.20f, 0.25f, 0.90f});
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4 {0.24f, 0.27f, 0.33f, 1.0f});
		ImGui::PushStyleColor(ImGuiCol_Text, kInkL);
		ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0.43f, 0.43f, 0.39f, 1.0f});
	}
	ImGui::PushFont(font, font_size);
	// (ds-pc-port12) The primary button in the display face: the label drawn engraved into the gold (dark letters, a
	// light edge under them and a soft dark one above), on whole pixels so it stays crisp.
	const bool engraved = primary && font == g_font_display;
	if (engraved) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4 {0, 0, 0, 0});
	const bool clicked = ImGui::Button(I18n::T(text), size);
	Anim::Highlight(g_unit); // (step5) eased hover / focus glow, border and accent bar (real time)
	if (engraved) {
		ImGui::PopStyleColor();
		auto*        dl  = ImGui::GetWindowDrawList();
		const char*  t   = I18n::T(text);
		const ImVec2 ts  = ImGui::CalcTextSize(t);
		const ImVec2 mn  = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
		const ImVec2 p   = {std::floor(mn.x + (mx.x - mn.x - ts.x) * 0.5f), std::floor(mn.y + (mx.y - mn.y - ts.y) * 0.5f)};
		const float  off = std::max(1.0f, std::round(font_size / 28.0f));
		dl->AddText(font, font_size, {p.x, p.y + off}, IM_COL32(255, 238, 196, 150), t);   // the light lower edge
		dl->AddText(font, font_size, {p.x, p.y - off}, IM_COL32(40, 26, 8, 90), t);        // the shade above
		dl->AddText(font, font_size, p, IM_COL32(22, 14, 5, 255), t);                      // the letters
		dl->AddText(font, font_size, {p.x + 0.5f * off, p.y}, IM_COL32(22, 14, 5, 160), t); // a touch more weight
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(5);
	Tip(tip);
	return clicked;
}

// (11-13) Archivo de paginacion de hasta 100 GB en C: (PowerShell como administrador: Windows pide permiso). Evita que
// un PC con poca RAM llegue al limite de memoria comprometida (un PC de 16 GB se reinicio ahi). Hace falta reiniciar.
void SetBigPagefile() {
#if defined(_WIN32)
	// A fixed 60 GB page file on C:, written where Windows keeps it (the PagingFiles registry value) with automatic
	// management turned off; the window shows the value read back. Applied at the next restart.
	static const char* script =
	    "@echo off\r\n"
	    "echo Archivo de paginacion fijo de 60 GB en C:\r\n"
	    "wmic computersystem where name=\"%computername%\" set AutomaticManagedPagefile=False >nul 2>&1\r\n"
	    "powershell -NoProfile -ExecutionPolicy Bypass -Command \"try { $c=Get-CimInstance Win32_ComputerSystem; "
	    "Set-CimInstance -InputObject $c -Property @{AutomaticManagedPagefile=$false} } catch {}\" >nul 2>&1\r\n"
	    "reg add \"HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management\" /v PagingFiles "
	    "/t REG_MULTI_SZ /d \"C:\\pagefile.sys 61440 61440\" /f\r\n"
	    "echo.\r\n"
	    "reg query \"HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management\" /v PagingFiles\r\n"
	    "echo.\r\n"
	    "echo Si arriba pone C:\\pagefile.sys 61440 61440 esta hecho. REINICIA EL PC para aplicarlo.\r\n"
	    "pause\r\n";
	char dir[MAX_PATH] = {};
	if (GetTempPathA(MAX_PATH, dir) == 0) return;
	const std::string path = std::string(dir) + "kyty-pagefile-60gb.cmd";
	if (FILE* f = std::fopen(path.c_str(), "wb"); f != nullptr) {
		std::fwrite(script, 1, std::strlen(script), f);
		std::fclose(f);
		// "runas": Windows asks for administrator permission (UAC) and runs the script elevated.
		const std::string args = "/c \"" + path + "\"";
		ShellExecuteA(nullptr, "runas", "cmd.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
	}
#endif
}

// Whether the 60 GB page file is already written in the registry (then only a restart is missing).
static bool PagefileConfigured() {
#if defined(_WIN32)
	char  value[1024] = {};
	DWORD size        = sizeof(value) - 2;
	if (RegGetValueA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management",
	                 "PagingFiles", RRF_RT_REG_MULTI_SZ, nullptr, value, &size) == ERROR_SUCCESS) {
		return std::strstr(value, "61440") != nullptr;
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
		ids->emplace_back(z);
		labels->emplace_back(z);
	}
	const std::string value = s.fullscreen ? s.resolution : std::to_string(s.width) + "x" + std::to_string(s.height);
	if (std::find(ids->begin(), ids->end(), value) == ids->end()) {
		ids->push_back(value);
		labels->push_back(value);
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
		Tip("4K: el modo rendimiento del juego (cinemáticas sin el tope de 30) con la calidad cinemática: iluminación completa y teselado denso. Otras resoluciones: modo rendimiento.");
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
		     "Reescalado: dibuja el 3D a menos resolucion y lo sube a la de la pantalla (mas fps). DLSS: NVIDIA RTX. XeSS, FSR 3.1 y FSR 4: todas las graficas (FSR 4: modelo de AMD con INT8, RTX 20 / RX 6000 o mas nuevas; el 3D a 360 lineas como minimo).",
		     146, 260, true);
	}
	ImGui::BeginDisabled(!nvidia);
	LChoice("Preset DLSS", kDlssPresets, kDlssPresetCount, s.dlss_preset,
	        nvidia ? "Modelo de DLSS: auto (el de NVIDIA), K = DLSS 4, L y M = DLSS 4.5 (mas nitidos, algo mas lentos)" : "Sin DLSS: tu grafica no es NVIDIA.");
	ImGui::EndDisabled();
	LChoice("Nitidez", kSharpness, kSharpnessCount, s.sharpness,
	        "Nitidez del reescalado (DLSS, XeSS, FSR 3.1 y FSR 4), aplicada a la resolucion de la pantalla. Mas alto: mas nitido, pero puede marcar bordes y ruido.");
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
	LRow("Luz RT", kRtLightLevelCount, kRtLightLevels, s.rt_ssgi, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Iluminación por rayos sobre la imagen (luz rebotada y sombreado de contacto). Solo funciona con DLSS, XeSS o FSR. Cuesta FPS. Experimental.", fx, fw);
	LRow("Luz volumétrica", kRtLevelCount, kRtLevels, s.godrays, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Luz volumétrica por rayos: rayos de sol que entran por ventanas y puertas, y halo de luz en la niebla alrededor de antorchas. Solo con DLSS, XeSS o FSR. Bajo ~0,4 ms, Medio ~0,8 ms, Alto ~1,2 ms.", fx, fw);
	LRow("Partículas volum.", kMotesLevelCount, kMotesLevels, s.motes, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Motas de polvo en suspensión iluminadas por los rayos de sol y el halo de antorchas y lámparas (la luz volumétrica); flotan despacio y quedan fijas en el mundo. Sutil / Media / Densa: cuántas. Solo con DLSS, XeSS o FSR. ~0,2-0,5 ms.", fx, fw); // (motes-1031)
	// (ds-pc-port10) Luz de antorchas RT and Destellos de lente are gone from the port (always off).
	LRow("Humo y partículas", kSmokeLevelCount, kSmokeLevels, s.smoke, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
	     "Humo y partículas más realistas sobre las del propio juego: Suave = el humo, la niebla de partículas y las chispas ya no cortan en línea dura donde tocan suelo y paredes (se funden en ~35 cm). Solo con DLSS, XeSS o FSR. ~0,3 ms.",
	     fx, fw);
	ImGui::EndDisabled();
	LCheckRow("RT por hardware (experimental)", s.rt_hw_exp,
	          "Activa las opciones de trazado de rayos por hardware (Reflejos y Sombras sol por RT real). Experimental: puede tener fallos y cuesta mucha GPU y VRAM. Necesita GPU con RT.");
	if (s.rt_hw_exp) {
		ImGui::BeginDisabled(s.pure_game);
		{
			std::string x = s.rt_reflections ? "1" : "0";
			if (LRow("Reflejos (RT real)", kYesNoCount, kYesNo, x, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
			         "Reflejos con trazado de rayos: refleja también lo que está fuera de pantalla o tapado, con sus texturas.", fx, fw))
				s.rt_reflections = x == "1";
		}
		const bool sun_was = s.rt_sun;
		{
			std::string x = s.rt_sun ? "1" : "0";
			if (LRow("Sombras sol (RT real)", kYesNoCount, kYesNo, x, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; },
			         "Sombras del sol con trazado de rayos: sombras de contacto y las que faltaban. No se puede usar a la vez que la Luz volumétrica ni las Sombras RT.",
			         fx, fw))
				s.rt_sun = x == "1";
		}
		// (vol25) Luz volumetrica / Sombras RT and Sombras sol share the engine's sun pass: choosing one turns the other off.
		if (s.rt_sun && !sun_was) s.godrays = "0", s.rt_shadows = "0";
		ImGui::EndDisabled();
	}
	if (s.rt_sun && s.rt_hw_exp && (s.godrays != "0" || s.rt_shadows != "0")) s.rt_sun = false;
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
	const bool small = s.simulate_ram16 || (g_rt.pc.ram_gb > 0 && g_rt.pc.ram_gb <= 20);
	{ // (port28) always shown and usable; off by default
		const float y = RowY();
		LCheck(y, "Ahorro de memoria RAM", s.ram16_boost,
		       "Usa la memoria con más cuidado: reserva 6 GB fijos para el juego y la carga anticipada de datos espera si queda poca RAM libre. Evita cierres y tirones fuertes por falta de memoria. No cambia los gráficos y normalmente no cuesta FPS. Actívalo si tu PC tiene 16 GB de RAM o menos; con 32 GB no hace falta.");
		RowEnd(y);
		if (small && s.ram16_boost) {
			const float ny = RowY();
			ImGui::SetCursorScreenPos({g_lbox.at.x + U(14), ny});
			ImGui::TextColored(kGoldL, "%s", I18n::T("RAM justa: perfil seguro. Cierra los navegadores mientras juegas."));
			RowEnd(ny, 24);
		}
	}
	{
		const float y = RowY();
		if (LButton("Archivo de paginación de 60 GB", {g_lbox.at.x + U(12), y}, {U(396), U(32)}, false, g_font_semi, U(15),
		            "Pone en el disco C: un archivo de paginación de 16 a 60 GB (Windows pedirá permiso de administrador). Evita que el PC se reinicie al quedarse sin memoria en equipos con poca RAM. Solo ocupa espacio cuando hace falta. Reinicia el PC después."))
			SetBigPagefile();
		RowEnd(y, 36);
		const uint64_t limit = CommitLimitGb();
		if (limit != 0) {
			const float ny = RowY();
			ImGui::SetCursorScreenPos({g_lbox.at.x + U(14), ny});
			const bool pending = limit < 64 && PagefileConfigured();
			ImGui::TextColored(limit >= 64 ? kGoldL : kGold, "Memoria total de Windows (RAM + paginación): %llu GB%s",
			                   static_cast<unsigned long long>(limit),
			                   limit >= 64 ? "" : pending ? "  -  hecho: REINICIA EL PC" : "  -  pulsa el botón y reinicia");
			RowEnd(ny, 24);
		}
	}
	s.profile16 = LowRam(s, g_rt.pc);
	{
		const float y = RowY();
		LCheck(y, "Modo seguro", s.safe_gpu, "Apaga una forma más rápida de mandar trabajo a la gráfica. Es más estable en gráficas o drivers que se cuelgan o dan pantallazo azul, pero cuesta mucho: unos 37 % menos de FPS. Déjalo apagado salvo que el juego se te cuelgue o te dé pantallazos.", 12);
		LCheck(y, "Todos los núcleos", s.all_ccds, "Solo para Ryzen con dos bloques de núcleos (CCD), como un 7950X3D o un 5900X; en otros procesadores no cambia nada. Apagado: el juego usa solo el bloque con más caché (memoria rápida dentro del procesador), que suele dar más FPS y menos tirones. Encendido: usa todos los núcleos. Déjalo apagado salvo que notes que encendido te va mejor.", 216);
		RowEnd(y);
	}
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
	LCheckRow("Menos trabajo por cada dibujo", s.perf_pm4_regfile,
	          "El juego manda a la gráfica miles de órdenes en cada imagen. Con esto, los ajustes sencillos que no hace falta traducir se apuntan directamente, sin pasar por el traductor. Ahorra algo de procesador: puede dar unos pocos FPS más si tu procesador es lo que te limita. No cambia la imagen. Experimental: si ves fallos gráficos, apágalo.");
	LCheckRow("Enviar texturas con dos hilos", s.perf_upload_lanes,
	          "Cuando llegan texturas y datos nuevos a la gráfica, parte de su preparación se reparte entre dos hilos (tareas que el procesador hace a la vez) en vez de uno. Puede quitar tirones en zonas con mucha carga de texturas. Útil con procesadores de 8 núcleos o más; con menos núcleos puede no notarse. Experimental.");
	LCheckRow("Ayuda extra si el procesador va justo", s.perf_upload_help,
	          "Cuando el procesador va muy ocupado, el hilo que dibuja cada imagen hace él mismo los envíos de texturas pendientes en vez de quedarse esperando. En procesadores de 4 a 6 núcleos y portátiles puede dar unos FPS más (en pruebas, hasta un 4 %). Con un procesador sobrado casi nunca actúa: ahí da igual.");
	LCheckRow("Liberar el procesador cuando va saturado", s.perf_spin_gate,
	          "Unos hilos de ayuda del emulador esperan trabajo dando vueltas sin parar (gastan procesador sin hacer nada útil). Con esto dejan de hacerlo cuando quedan menos de 3 núcleos libres, y ese procesador vuelve al juego. En procesadores de 4 a 6 núcleos reduce bastante los bajones de FPS y los tirones. Con muchos núcleos casi nunca actúa. Recomendado en portátiles y PCs modestos.");
	LCheckRow("Devolver antes las copias de la gráfica", s.perf_eager_feedback,
	          "Cuando la gráfica termina una copia de datos que el juego leerá después, se le devuelve en cuanto está lista en vez de esperar a que la pida. Así el juego no se queda esperando: puede quitar algunos tirones. No cambia la imagen. Experimental: si notas fallos, apágalo.");
	LCheckRow("Agrupar las copias de la gráfica", s.perf_feedback_batch,
	          "Esas mismas copias de datos de la gráfica se preparan juntas, por bloques, en vez de una a una. Un poco menos de trabajo para el procesador; la ganancia es pequeña. No cambia la imagen. Experimental: si notas fallos, apágalo.");
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
	BoxBegin("Modos Souls");
	struct Mode {
		const char* id;
		const char* label;
		const char* tip;
	};
	static const Mode modes[] = {
	    {"ligero", "Modo Souls ligero",
	     "Adaptado para graficas de 8 GB como RTX 3060, RTX 3060 Ti, RTX 3070, RTX 2070/2080, RX 6600, RX 6600 XT o RX 6650 XT: el juego a 30 FPS estables y frame generation de AMD x2 (60 en pantalla). A 1080p, con el antialiasing de XeSS o de FSR (sin reescalar)."},
	    {"equilibrado", "Modo Souls equilibrado",
	     "Para graficas de la potencia de una RTX 3080: RTX 2080 Ti, RTX 3070 Ti, RTX 3080, RTX 3080 Ti, RTX 4060 Ti, RTX 4070, RX 6800, RX 6800 XT, RX 6900 XT, RX 6950 XT, RX 7700 XT, RX 7800 XT o Intel Arc B580: el juego a 45 FPS estables y frame generation de AMD x2 (90 en pantalla). A 1440p, con el reescalado que elijas."},
	    {"ultra", "Modo Souls ultra",
	     "Para RTX 5060 / 5060 Ti / 5070 / 5070 Ti y RX 9070 / 9070 XT: todo al maximo, Detail Very High, Luz RT y Oclusion en Alta. Los FPS dependen de la potencia de tu GPU y tu CPU."},
	    {"extremo", "Modo Souls extremo",
	     "Para enfermos de Souls. RTX 5080, RTX 5090 o superior: modo cinematico del juego a 4K y absolutamente todo al maximo (Ultra en todo, filtrado x16 y texturas nitidas). Los FPS dependen de la potencia de tu GPU y tu CPU."},
	    {"ligero-plus", "Modo Souls ligero +",
	     "Para monitores de mas de 60 Hz. Como el Modo Souls ligero (mismos graficos), pero con el juego a 35 FPS estables y frame generation de AMD x2 (70 en pantalla): mas fluido y algo menos de latencia."},
	    {"low", "Modo Souls PC low profile (900p, todo en bajo)",
	     "Para PCs modestos: 30 FPS reales con frame generation de AMD x2 (60 en pantalla), el juego a 900p reescalado con FSR (nitidez 50 %) y todos los graficos en bajo (texturas, distancia, sombras, detalle, reflejos, objetos y nitidez). Al quitarlo vuelve todo a Maximo y a 1440p."}};
	const auto pick = [&](float y, const Mode& m, float x) {
		bool on = s.souls_mode == m.id;
		if (!LCheck(y, m.label, on, m.tip, x)) return;
		ApplyPreset(&s, on ? m.id : "ninguno");
		s.souls_mode = on ? m.id : "ninguno";
	};
	// Launcher-souls.ps1's places: ligero | equilibrado, ultra | extremo, PC low profile.
	for (size_t i = 0; i < 4; i += 2) {
		const float y = RowY();
		pick(y, modes[i], 12);
		pick(y, modes[i + 1], 216);
		RowEnd(y);
	}
	{
		const float y = RowY();
		pick(y, modes[4], 12);
		pick(y, modes[5], 216);
		RowEnd(y);
	}
	if (FixedFg(s)) {
		const float y = RowY();
		ImGui::SetCursorScreenPos({g_lbox.at.x + U(14), y + U(4)});
		ImGui::TextColored(kGoldL, "%s", I18n::T(s.souls_mode == "equilibrado" ? "Frame generation fija: AMD x2 (45 reales, 90 en pantalla)." : s.souls_mode == "ligero-plus" ? "Frame generation fija: AMD x2 (35 reales, 70 en pantalla)." : "Frame generation fija: AMD x2 (30 reales, 60 en pantalla)."));
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

// Trofeos: the count and % and, opened, the list (inside the options).
void LauncherTrophiesBox() {
	static std::vector<TrophyRow> rows;
	static double                 loaded_at = -100.0;
	if (ImGui::GetTime() - loaded_at > 3.0) {
		rows      = LoadTrophyRows();
		loaded_at = ImGui::GetTime();
		std::stable_sort(rows.begin(), rows.end(), [](const TrophyRow& a, const TrophyRow& b) { return a.earned && !b.earned; });
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
	const int pct = total > 0 ? have * 100 / total : 0;
	char      title[128];
	if (!rows.empty())
		std::snprintf(title, sizeof(title), "%s   %d / %d   (%d%%)  %s", I18n::T("Trofeos"), earned, static_cast<int>(rows.size()), pct, g_show_trophies ? "^" : "v");
	else std::snprintf(title, sizeof(title), "%s  %s", I18n::T("Trofeos"), g_show_trophies ? "^" : "v");
	const float y = RowY();
	ImGui::SetCursorScreenPos({g_lbox.at.x, y});
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4 {0.094f, 0.11f, 0.15f, 0.92f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {0.17f, 0.20f, 0.25f, 0.95f});
	ImGui::PushStyleColor(ImGuiCol_Border, ImVec4 {0.55f, 0.53f, 0.48f, 1.0f});
	ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, {0.0f, 0.5f});
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {U(48), U(4)});
	ImGui::PushFont(g_font_semi, U(16));
	if (ImGui::Button((std::string(title) + "###trophies").c_str(), {U(420), U(40)})) g_show_trophies = !g_show_trophies;
	ImGui::PopFont();
	ImGui::PopStyleVar(2);
	ImGui::PopStyleColor(3);
	Libs::Trophy::DrawCup(ImGui::GetWindowDrawList(), g_lbox.at.x + U(10), y + U(6), U(28), Libs::Trophy::CupColor(2));
	RowEnd(y, 46);
	if (!g_show_trophies) return;
	const float ly = RowY();
	ImGui::SetCursorScreenPos({g_lbox.at.x, ly});
	ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0.04f, 0.045f, 0.06f, 0.95f});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {U(8), U(4)});
	ImGui::BeginChild("##trophylist", {U(420), U(300)}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding); Anim::SmoothScroll();
	if (rows.empty()) {
		ImGui::TextWrapped("%s", I18n::T("Aún no hay trofeos: aparecen al jugar."));
	} else {
		ImGui::TextColored(kGoldL, I18n::T("%d%% completado   -   %d / %d"), pct, earned, static_cast<int>(rows.size()));
		ImGui::ProgressBar(total > 0 ? static_cast<float>(have) / static_cast<float>(total) : 0.0f, {-1.0f, U(8)}, "");
		static const ImVec4 grade_col[5] = {{0.7f, 0.7f, 0.7f, 1.0f}, {0.72f, 0.82f, 0.95f, 1.0f}, {0.96f, 0.80f, 0.32f, 1.0f},
		                                    {0.84f, 0.86f, 0.88f, 1.0f}, {0.86f, 0.56f, 0.32f, 1.0f}};
		static const char*  grade_txt[5] = {"", "Platino", "Oro", "Plata", "Bronce"};
		for (const auto& r: rows) {
			const int  g      = r.grade >= 1 && r.grade <= 4 ? r.grade : 0;
			const bool secret = r.hidden && !r.earned;
			ImGui::PushID(r.id);
			ImGui::TextColored(r.earned ? grade_col[g] : ImVec4 {0.62f, 0.62f, 0.62f, 1.0f}, "%s", I18n::T(grade_txt[g]));
			ImGui::SameLine(U(96));
			ImGui::TextColored(r.earned ? kInkL : ImVec4 {0.66f, 0.66f, 0.66f, 1.0f}, "%s", secret ? I18n::T("Trofeo oculto") : r.name.c_str());
			if (!secret) Tip(r.detail.c_str());
			ImGui::PopID();
		}
	}
	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopStyleColor();
	RowEnd(ly, 306);
}

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
		if (g_upd.state != UpdateUi::Idle && !g_upd.busy.load()) SetUpd(UpdateUi::Idle, "");
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
		if (LButton(pv.running && !pv.playable ? "JUGAR (sin esperar)" : "JUGAR", {g_lbox.at.x, y}, {U(420), U(58)}, true, g_font_display, U(35),
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
			LauncherTrophiesBox();
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

	// (ds-pc-port17) The installed version, always visible (bottom left), and the startup update notice.
	{
		char ver[256];
		std::snprintf(ver, sizeof(ver), I18n::T("Version instalada: %s"), Update::CurrentVersion());
		const std::string line = std::string(ver) + "  \xC2\xB7  " + Update::BuildStamp();
		bg->AddText({U(12), height - U(26)}, IM_COL32(220, 214, 200, 230), line.c_str());
	}
	StartupCheck();
	UpdateNotice(width, height);

	g_popup = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
	if (ctx.ColorStack.Size > colors_before) ImGui::PopStyleColor(ctx.ColorStack.Size - colors_before);
	if (ctx.StyleVarStack.Size > vars_before) ImGui::PopStyleVar(ctx.StyleVarStack.Size - vars_before);
	ImGui::PopFont();
	Bump();
}

bool WantsOverlay() {
	return MetricsPanel::Shown() || g_open.load(std::memory_order_relaxed) || Resume::SaveAndQuitActive() || Anim::Closing(); // (ds-pc-port38: the fade-out)
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
	                        : souls == "low" ? "Modo Souls PC low profile" : "Personalizado";
	const bool  fixed = souls == "ligero" || souls == "ligero-plus" || souls == "equilibrado" || souls == "low";
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
bool Begin(const char* title, float width, float height, float unit) {
	if (Anim::g_page_seen != g_f1_page.load()) {
		Anim::g_page_seen = g_f1_page.load();
		Anim::g_page_at   = Anim::Now();
	}
	const float t = Anim::EaseOut(Anim::Progress(Anim::g_page_at.load(), Anim::kPage));
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
	col(ImGuiCol_WindowBg, ImVec4 {0.025f, 0.025f, 0.03f, 0.97f});
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
	ImGui::SetNextWindowPos({std::round((1.0f - t) * 64.0f * unit), 0}, ImGuiCond_Always);
	ImGui::SetNextWindowSize({width, height}, ImGuiCond_Always);
	ImGui::Begin(title, nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
	ImGui::PushFont(nullptr, std::round(46.0f * unit));
	ImGui::TextColored(kGold, "%s", I18n::T(title));
	ImGui::PopFont();
	ImGui::Separator();
	ImGui::Spacing();
	return true;
}
bool Footer(float unit) {
	ImGui::Separator();
	bool close = false;
	const ImVec2 bs {std::round(240 * unit), std::round(54 * unit)};
	if (Anim::Button(I18n::T("Guardar"), bs, unit)) SaveNow();
	ImGui::SameLine();
	if (g_saved.pad_map != g_rt.launched.pad_map || g_saved.kbm != g_rt.launched.kbm || g_saved.kbm_map != g_rt.launched.kbm_map ||
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
	ImGui::End();
	ImGui::PopStyleColor(g_style_colors);
	ImGui::PopStyleVar(g_style_vars);
	g_style_colors = g_style_vars = 0;
	ImGui::PopFont();
	return !close;
}

bool DrawPad(float width, float height, float unit) {
	Begin("Mapear mando", width, height, unit);
	ImGui::BeginChild("##padpage", {0, -std::round(84 * unit)}, ImGuiChildFlags_NavFlattened); Anim::SmoothScroll(); // (ds-pc-port41) the D-pad / stick move row by row inside the list and the page follows the focus
	if (const int d = g_page_scroll.exchange(0); d != 0) ImGui::SetScrollY(ImGui::GetScrollY() + d * ImGui::GetWindowHeight() * 0.6f);
	(void)PadMap::Draw(g_edit, unit);
	ImGui::EndChild();
	return Footer(unit);
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

// (ds-pc-port8) In the game: F2's save and quit, a panel in the middle of the screen until the game closes.
// (ds-pc-port38) `closing`: the fade-out after F2 / Esc closed it (drawn disabled, no input taken).
// (bb-style-f1) The cursor is kept at every level of F2: each screen (0 the option bar, 1 Mando, 2 Teclado y ratón,
// 3 Trofeos) remembers the item that had the keyboard / pad focus; when a screen comes back (back from an option, from a
// sub-screen, or F2 opened again in this session) that item gets the focus again instead of ImGui's first item.
namespace KeepFocus {
struct Mem {
	ImGuiID      id = 0;
	ImGuiWindow* window = nullptr;
};
Mem g_mem[4];
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
	if (const int page = g_f1_page.load(); page != 0) {
		const bool open = page == 1 ? F1Pages::DrawPad(width, height, unit) : F1Pages::DrawKbm(width, height, unit);
		KeepFocus::After(page == 1 ? 1 : 2);
		if (!open) {
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
	ImGui::GetBackgroundDrawList()->AddRectFilled({0, 0}, {width, height}, IM_COL32(0, 0, 0, static_cast<int>(140 * t)));
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
		Souls::Monolith(menu_dl, {at.x + side * 0.4f, at.y + side * 0.5f}, side, unit, al);
		ImGui::Dummy({side * 0.95f, side});
		ImGui::SameLine();
		ImFont*      title_font = DisplayFont() != nullptr ? DisplayFont() : ImGui::GetFont();
		const float  title_size = std::round(ImGui::GetFontSize() * 1.3f);
		const char*  title      = I18n::T("Demon's Souls PC");
		const ImVec2 tsz        = title_font->CalcTextSizeA(title_size, FLT_MAX, 0.0f, title);
		const ImVec2 tp {ImGui::GetCursorScreenPos().x, at.y + (side - tsz.y) * 0.5f};
		// A warm candle glow behind the title, flickering.
		const float fl = Souls::Flicker();
		for (int k = 3; k >= 1; k--)
			menu_dl->AddRectFilled({tp.x - 10 * unit * k, tp.y + tsz.y * 0.2f - 4 * unit * k}, {tp.x + tsz.x + 10 * unit * k, tp.y + tsz.y * 0.8f + 4 * unit * k},
			                       Souls::C(Souls::kCandle, al * 0.022f * fl), 20 * unit);
		Souls::Etched(menu_dl, title_font, title_size, tp, title, al * (0.9f + 0.1f * fl));
		ImGui::Dummy({tsz.x, side});
		const float dy = at.y + side + 10 * unit;
		Souls::Divider(menu_dl, at.x, at.x + 420 * unit, dy, unit, al);
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
	const ImVec2 bs {std::round(270 * unit), std::round(340 * unit)};
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
	if (CardButton(I18n::T("Cerrar el juego"), Icons::Power, bs, unit, 4, true)) {
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
		Souls::Panel(menu_dl, wp, {wp.x + ws.x, wp.y + ws.y}, unit, al, 10 * unit);
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
					// (11-12) Sharper text: oversampled, sub-pixel placed, a touch heavier.
					ImFontConfig cfg;
					cfg.OversampleH       = 3;
					cfg.OversampleV       = 2;
					cfg.PixelSnapH        = false;
					cfg.RasterizerMultiply = 1.12f;
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
					cfg.OversampleH          = 3;
					cfg.OversampleV          = 2;
					cfg.GlyphExtraAdvanceX   = 1.5f;
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
	while (g_start_result == 0) {
		SDL_Event e;
		while (SDL_PollEvent(&e) == 1) {
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
		ImGui::Render();
		SDL_RenderSetScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
		ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
		SDL_RenderPresent(renderer);
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
