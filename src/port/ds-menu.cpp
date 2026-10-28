// "Demon's Souls" PC port: the in-game options menu. Drawn by the system overlay (src/graphics/presentation/
// systemOverlay.cpp) into the swapchain image, after the upscaler and outside frame generation: while it is open the
// frames go out without generated ones (swapchain.cpp, PortLive::menu_open), and the game gets no input.
//   F1, or R3 held + OPTIONS (OPTIONS then never reaches the game): open / close. Circle / Escape: close. L1 / R1, Q / E: the tabs.
// What the engine applies while playing (sharpness, the fps cap, the frame generation multiplier of a fixed DLSS mode,
// metrics, rumble, ray tracing turned off) is applied at once; the rest is saved to settings.ini and marked "(requiere reinicio)": applied at the next start, which
// the menu offers.

#include "port/ds-port.h"
#include "port/ds-update.h"
#include "port/ds-precompile.h"
#include "port/ds-i18n.h"
#include "port/ds-steam.h"

#include "local/perf-overlay.h"
#include "local/port-live.h"
#include "libs/trophy.h"

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
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <set>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
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
std::atomic<bool>        g_open {false};
std::atomic<uint64_t>    g_revision {1};
std::atomic<bool>        g_reset_edit {false};
std::mutex               g_mutex;
std::deque<Input>        g_inputs;
int                      g_passthrough = 0; // window thread: zeroed axes sent to the game as the menu opened
int                      g_cursor_before = -1;
constexpr int            kTabCount = 6; // (10-27c) Gráficos, Pantalla, Sonido, Controles, Sistema, Trofeos
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
bool                     g_start_mode = false; // (10-27b) the start menu (before the game), not F1
int                      g_start_result = 0;   // 1 Jugar, -1 Salir
bool                     g_config_mode = false; // (10-28) the Configurador before the game (its own window)
bool                     g_cfg_open = false;    // (10-28) the Configurador inside F1 (Sistema > Configurador)
std::vector<std::string> g_cfg_why;             // its "Hemos elegido: ..." lines (empty: not optimized yet)

void Bump() {
	g_revision.fetch_add(1, std::memory_order_relaxed);
}

void Queue(const Input& input) {
	std::scoped_lock lock(g_mutex);
	g_inputs.push_back(input);
	Bump();
}

void SetOpen(bool open) {
	if (g_open.exchange(open) == open) return;
	PortLive::menu_open.store(open, std::memory_order_relaxed);
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
	const bool open = g_open.load();
	switch (e.type) {
		case SDL_KEYDOWN:
			if (e.key.keysym.sym == SDLK_F1 && e.key.repeat == 0) {
				SetOpen(!open);
				return true;
			}
			if (!open) return false;
			if (e.key.keysym.sym == SDLK_ESCAPE && e.key.repeat == 0) {
				SetOpen(false);
				return true;
			}
			if (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_PAGEUP) Queue({Input::Tab, -1});
			if (e.key.keysym.sym == SDLK_e || e.key.keysym.sym == SDLK_PAGEDOWN) Queue({Input::Tab, +1});
			if (const ImGuiKey k = KeyboardKey(e.key.keysym.sym); k != ImGuiKey_None) Queue({Input::Key, k, true});
			return true;
		case SDL_KEYUP:
			if (e.key.keysym.sym == SDLK_F1) return true;
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
				SetOpen(!open);
				return true;
			}
			if (!open) return false;
			if (b == SDL_CONTROLLER_BUTTON_B && !g_popup.load()) {
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
			if (e.caxis.axis == SDL_CONTROLLER_AXIS_LEFTY) {
				Queue({Input::Analog, ImGuiKey_GamepadLStickUp, v < -0.35f, std::max(-v, 0.0f)});
				Queue({Input::Analog, ImGuiKey_GamepadLStickDown, v > 0.35f, std::max(v, 0.0f)});
			} else if (e.caxis.axis == SDL_CONTROLLER_AXIS_LEFTX) {
				Queue({Input::Analog, ImGuiKey_GamepadLStickLeft, v < -0.35f, std::max(-v, 0.0f)});
				Queue({Input::Analog, ImGuiKey_GamepadLStickRight, v > 0.35f, std::max(v, 0.0f)});
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
			Queue({Input::MousePos, 0, false, static_cast<float>(e.button.x), static_cast<float>(e.button.y)});
			Queue({Input::MouseButton, e.button.button == SDL_BUTTON_RIGHT ? 1 : (e.button.button == SDL_BUTTON_MIDDLE ? 2 : 0),
			       e.type == SDL_MOUSEBUTTONDOWN});
			return e.type == SDL_MOUSEBUTTONDOWN;
		case SDL_MOUSEWHEEL:
			if (!open) return false;
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
			case Input::Tab: g_tab_request = (g_tab + in.key + kTabCount) % kTabCount; break;
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

// The row itself: `current` changed by `step` (-1 / +1) within count; true when it changed.
bool SelectorRow(const char* label, bool restart, int count, int& current, const char* value, const char* tip) {
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
	if (clicked && !disabled) {
		// (A click on the left arrow's half goes back.)
		const float x = ImGui::GetIO().MousePos.x;
		step          = ImGui::IsMouseReleased(0) && x < at.x + w * 0.72f && x > at.x + w * 0.45f ? -1 : +1;
	}
	auto*       dl    = ImGui::GetWindowDrawList();
	const ImU32 ink   = ImGui::GetColorU32(disabled ? ImVec4 {0.5f, 0.48f, 0.45f, 1.0f} : ImVec4 {0.93f, 0.91f, 0.88f, 1.0f});
	const ImU32 gold  = ImGui::GetColorU32(disabled ? ImVec4 {0.5f, 0.45f, 0.35f, 1.0f} : kGold);
	const float pad   = 16.0f * g_unit;
	const float ty    = at.y + (h - ImGui::GetFontSize()) * 0.5f;
	if (focused) dl->AddRectFilled({at.x, at.y + h - 3 * g_unit}, {at.x + w, at.y + h}, gold);
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
	dl->AddText({mid - vs.x * 0.5f, ty}, gold, shown);
	if (count > 1) {
		const float half = std::max(vs.x, 220.0f * g_unit) * 0.5f + 18.0f * g_unit;
		const float cy   = at.y + h * 0.5f, s2 = 7.0f * g_unit;
		const ImU32 arrow = focused ? gold : ink;
		dl->AddTriangleFilled({mid - half, cy}, {mid - half + s2, cy - s2}, {mid - half + s2, cy + s2}, arrow);
		dl->AddTriangleFilled({mid + half, cy}, {mid + half - s2, cy - s2}, {mid + half - s2, cy + s2}, arrow);
	}
	ImGui::PopID();
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
	if (!SelectorRow(label, restart, count, current, text(items[current]), tip)) return false;
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

void Section(const char* title) {
	ImGui::Spacing();
	ImGui::PushFont(nullptr, 30.0f * g_unit);
	ImGui::TextColored(kGold, "%s", title);
	ImGui::PopFont();
	const ImVec2 p = ImGui::GetCursorScreenPos();
	ImGui::GetWindowDrawList()->AddLine(p, {p.x + ImGui::GetContentRegionAvail().x, p.y}, ImGui::GetColorU32(ImVec4 {0.87f, 0.77f, 0.55f, 0.35f}));
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
	const int level       = s.metrics == "2" ? 2 : (s.metrics == "1" ? 1 : 0);
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
	if (r.tag == Update::CurrentVersion()) SetUpd(UpdateUi::UpToDate, "Ya tienes la ultima version (" + r.tag + ").");
	else SetUpd(UpdateUi::Available, "Nueva version " + r.tag + " \xE2\x80\x94 Cruz para actualizar");
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
	if (res.ok)
		SetUpd(UpdateUi::Done, "Actualizacion completada a la version " + r.tag +
		                           ". Cierra el juego y vuelve a abrirlo para usar la nueva version. Tus shaders compilados y partidas guardadas se mantienen.");
	else SetUpd(UpdateUi::Failed, res.error);
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
	if (res.ok) SetUpd(UpdateUi::Done, "Restaurada la version " + b.version + ". Cierra el juego y vuelve a abrirlo.");
	else SetUpd(UpdateUi::Failed, res.error);
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
	                   v.text.empty() ? I18n::T("Shaders: la cache de esta grafica ya esta preparada.") : v.text.c_str());
	if (v.running) {
		ImGui::SameLine();
		if (ImGui::SmallButton(I18n::T("Detener"))) std::thread([] { Precompile::Stop(); }).detach();
	} else if (!v.finished) {
		ImGui::SameLine();
		if (ImGui::SmallButton(I18n::T("Preparar shaders"))) Precompile::Start();
	}
}

void InfoLine(const char* caption, const std::string& value, const ImVec4& color) {
	ImGui::TextColored(kDim, "%s", I18n::T(caption));
	ImGui::PushStyleColor(ImGuiCol_Text, color);
	ImGui::TextWrapped("%s", value.c_str());
	ImGui::PopStyleColor();
	ImGui::Spacing();
}

// (F1 Trofeos) The trophies: from the trophy module when it has loaded them, else _SaveData/<title>/trophies-list.json
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
	ImGui::BeginChild("##trophies", {0, 0}, ImGuiChildFlags_None);
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
			              (secret ? "Se revelará al obtenerlo." : r.detail);
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
	if (pc.vram > 0) gpu += "  (" + std::to_string((pc.vram + (512ull << 20)) >> 30) + " GB)";
	if (!pc.nvidia && !pc.gpu.empty()) gpu += "  (sin DLSS: solo NVIDIA)";
	InfoLine("Grafica", gpu, ink);
	InfoLine("Procesador y RAM", (pc.cpu.empty() ? std::string("?") : pc.cpu) + "  |  " + std::to_string(pc.ram_gb) + " GB RAM", ink);
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
	if (state == UpdateUi::Available) {
		if (ImGui::Button(("Actualizar a " + g_upd.release.tag).c_str(), {bw, bh})) StartWorker(InstallWork);
	} else if (ImGui::Button(I18n::T("Actualizar version"), {bw, bh})) {
		SetUpd(UpdateUi::Checking, "Buscando la ultima version en GitHub...");
		StartWorker(CheckWork);
	}
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("Busca la ultima version publicada en GitHub y la instala en esta misma carpeta. Guarda los archivos del motor actuales en _backup_update (las 2 ultimas) y no toca la cache de shaders, las partidas, los seeds ni los ajustes.");
	ImGui::SameLine();
	if (ImGui::Button(I18n::T("Volver a la version anterior"), {bw, bh})) {
		OpenRestore();
		std::scoped_lock lock(g_upd.mutex);
		state = g_upd.state;
		text  = g_upd.text;
	}
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("Vuelve a una version guardada en _backup_update. Antes guarda la actual (se puede deshacer). No toca la cache de shaders, las partidas, los seeds ni los ajustes.");
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
				const bool  ok = Steam::AddToSteam(g_rt.exe_path, &report);
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

// (10-28) Resolucion (F1 > Gráficos > Imagen): full screen, the desktop's or a monitor mode; window, its size.
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
	}
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
	if (pc.vram > 0) gpu += "  -  " + std::to_string((pc.vram + (512ull << 20)) >> 30) + " GB de VRAM";
	InfoLine("Tarjeta gráfica", gpu, ink);
	std::string cpu = pc.cpu.empty() ? std::string("?") : pc.cpu;
	if (pc.cpu_cores > 0) cpu += "  -  " + std::to_string(pc.cpu_cores) + " núcleos / " + std::to_string(pc.cpu_threads) + " hilos";
	InfoLine("Procesador", cpu, ink);
	const bool low = pc.ram_gb > 0 && pc.ram_gb <= 20;
	InfoLine("Memoria RAM", std::to_string(pc.ram_gb) + " GB" + (low ? "  (se activan las mejoras para equipos de 16 GB de RAM)" : ""), low ? kWarn : ink);
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
		ImGui::TextColored(kDim, "%s", g_start_mode ? "Guardado. Se aplica al pulsar JUGAR; en el juego puedes cambiarlo con F1."
		                               : g_config_mode ? "Después puedes cambiar cualquier opción en el juego con F1."
		                                              : "Guardado. Lo marcado \"(requiere reinicio)\" se aplica la próxima vez que inicies el juego.");
	}
}

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
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6 * unit);
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
	// folded in) and Trofeos; the same sections and rows as F1, so both write the same settings.ini the same way.
	const char* game_tabs[kTabCount]     = {"Gráficos", "Pantalla", "Sonido", "Controles", "Sistema", "Trofeos"};
	const char* launcher_tabs[kTabCount] = {"Imagen", "Calidad gráfica", "Ray tracing", "Controles", "Sistema", "Trofeos"};
	const char* const* tabs = g_start_mode ? launcher_tabs : game_tabs;
	static const std::vector<int> game_groups[kTabCount] = {{0, 1, 2, 3}, {5}, {}, {7}, {6, 8}, {}}; // (10-28) no Modos Souls
	static const std::vector<int> launcher_groups[kTabCount] = {{5, 0, 1}, {2}, {3}, {7}, {9, 6, 8}, {}};
	const std::vector<int>* groups = g_start_mode ? launcher_groups : game_groups;
	static const char* section_names[10] = {"Imagen", "Frame generation", "Calidad gráfica", "Ray tracing", "Modos Souls",
	                                        "Pantalla", "Rendimiento", "Controles", "Tu equipo", "Configurador"};
	g_unit = unit;
	const float footer  = (g_start_mode ? 250 : 150) * unit;
	const bool  nvidia  = g_rt.pc.nvidia;
	// (10-28) The sections as a vertical list on the left (no tab strip to scroll): mouse, keyboard / D-pad navigation
	// (A / Intro selects) and LB / RB or Q / E as before.
	if (g_tab_request >= 0 && g_tab_request < kTabCount) g_tab = g_tab_request;
	if (g_tab < 0 || g_tab >= kTabCount) g_tab = 0;
	{
		const float side_w = 210.0f * unit;
		ImGui::BeginChild("##sections", {side_w, size.y - footer - ImGui::GetCursorPosY()}, ImGuiChildFlags_Borders);
		ImGui::PushFont(nullptr, 22.0f * unit);
		for (int t = 0; t < kTabCount; t++) {
			const bool on = g_tab == t;
			if (on) ImGui::PushStyleColor(ImGuiCol_Text, kGold);
			if (ImGui::Selectable((std::string(I18n::T(tabs[t])) + "###sec" + std::to_string(t)).c_str(), on, ImGuiSelectableFlags_None,
			                      {0, 40.0f * unit}))
				g_tab = t;
			if (on) ImGui::PopStyleColor();
			if (on && g_tab_request == t) ImGui::SetScrollHereY();
		}
		ImGui::PopFont();
		ImGui::EndChild();
		ImGui::SameLine();
	}
	ImGui::BeginGroup();
	{
		for (int once = 0, t = g_tab; once < 1; once++) {
			const float page_h = size.y - footer - ImGui::GetCursorPosY();
			const float list_w = ImGui::GetContentRegionAvail().x * 0.62f;
			ImGui::BeginChild("##page", {list_w, page_h}, ImGuiChildFlags_None);
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
						                                                              : "Modo Souls ligero / low profile: frame generation fija en AMD x2 (30 reales, 60 en pantalla).");
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
					              "Reflejos trazados por rayos guiados por los del propio juego: solo donde el juego refleja y con su intensidad, más nítidos (agua como espejo). Donde el rayo no encuentra nada queda el reflejo del juego. Solo con DLSS, XeSS o FSR. Bajo ~0,6 ms, Medio ~1 ms, Alto ~2 ms, Ultra ~3 ms (rayos a resolución completa y más fuertes). Beta. Apagarlo es inmediato."))
						live_changed = true;
					ChoiceRow("Luz RT", s.rt_ssgi != l.rt_ssgi, kRtLightLevels, kRtLightLevelCount, s.rt_ssgi,
					          "Iluminación por rayos sobre la imagen (luz rebotada y sombreado de contacto). Solo funciona con DLSS, XeSS o FSR. Cuesta FPS. Experimental.");
					if (ChoiceRow("Luz volumétrica", changed_not_live(l.godrays != s.godrays, 4), kRtLevels, kRtLevelCount, s.godrays,
					              "Luz volumétrica por rayos: rayos de sol que entran por ventanas y puertas, y halo de luz en la niebla alrededor de antorchas. Solo con DLSS, XeSS o FSR. Bajo ~0,4 ms, Medio ~0,8 ms, Alto ~1,2 ms. Apagarla es inmediato."))
						live_changed = true;
					if (ChoiceRow("Oclusión ambiental", changed_not_live(l.ao != s.ao, 8), kRtLevels, kRtLevelCount, s.ao,
					              "Oclusión ambiental nueva (sin RT por hardware): oscurece solo la luz ambiental en rincones, juntas y bajo objetos, con radio en metros, sin halos ni ruido; nunca el cielo, el fuego ni lo emisivo. Sustituye a la oclusión del juego. Solo con DLSS, XeSS o FSR. Ahora mismo cuesta unos 7 ms en el juego (se está arreglando): mejor en No. Apagarla es inmediato."))
						live_changed = true;
					if (ChoiceRow("Luz de antorchas RT", changed_not_live(l.torch != s.torch, 16), kTorchLevels, kTorchLevelCount, s.torch,
					              "Luz de antorchas RT: el fuego de antorchas, braseros y hogueras ilumina paredes, suelos y estructuras cercanas con un rebote cálido que parpadea, y se refleja en superficies lisas o mojadas. No atraviesa paredes ni personajes. Solo en pantalla, sin RT por hardware. Solo con DLSS, XeSS o FSR. Bajo 2 fuegos ~0,3 ms, Medio 4 ~0,5 ms, Alto 8 ~0,9 ms. Híbrido: Alto + rayos por hardware (RT real) donde la pantalla no ve al que tapa la luz; solo oscurece, nunca añade luz. Necesita GPU con RT. ~+0,5 ms. Apagarla es inmediato."))
						live_changed = true;
					{
						std::string x = s.lens_flare ? "1" : "0";
						if (ChoiceRow("Destellos de lente", changed_not_live(l.lens_flare != s.lens_flare, 32), kYesNo, kYesNoCount, x,
						              "Destello de lente de antorchas y fuegos visibles: pequeños reflejos de la óptica, una estrella y una raya suaves. Desaparece si el fuego queda tapado y cerca de los bordes. Se dibuja antes del HUD. Solo con DLSS, XeSS o FSR. ~0,1 ms. Apagarlo es inmediato.")) {
							s.lens_flare = x == "1";
							live_changed = true;
						}
					}
					if (ChoiceRow("Sombras RT", changed_not_live(s.rt_shadows != l.rt_shadows, 64), kRtLevels, kRtLevelCount, s.rt_shadows,
					              "Sombras del sol por rayos sobre la imagen (de pantalla, sin RT real): sombras de contacto que las del juego pierden (bajo brazos, entre piernas, junto a muros), con borde suave y estable. Solo donde el juego ya muestra sol. Solo con DLSS, XeSS o FSR. Bajo ~0,3 ms, Medio ~0,6 ms, Alto ~1 ms."))
						live_changed = true;
					ImGui::EndDisabled();
					CheckRow("RT por hardware (experimental)", s.rt_hw_exp != l.rt_hw_exp, s.rt_hw_exp,
					         "Activa las opciones de trazado de rayos por hardware (Reflejos, Sombras sol y Antorchas por RT real). Experimental: puede tener fallos y cuesta mucha GPU y VRAM. Necesita GPU con RT.");
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
						{
							std::string lamps = LampLevel(s);
							if (ChoiceRow("Antorchas (RT real)", changed_not_live(LampLevel(l) != lamps || (lamps != "0" && !l.rt_hw_exp), 2), kRtLevels, kRtLevelCount, lamps,
							              "Antorchas, lámparas, fuego y magia iluminan lo que tienen cerca con trazado de rayos y sombras reales. Bajo / Medio / Alto: más luces y más rayos. Apagarlo es inmediato.")) {
								s.rt_lamps     = lamps;
								s.rt_particles = lamps != "0";
								live_changed   = true;
							}
						}
						// (vol25) Luz volumetrica / Sombras RT and Sombras sol share the engine's sun pass: choosing one turns the other off.
						if (s.rt_sun && !sun_was) s.godrays = "0", s.rt_shadows = "0";
						ImGui::EndDisabled();
					}
					if (s.rt_sun && s.rt_hw_exp && (s.godrays != "0" || s.rt_shadows != "0")) s.rt_sun = false;
					if (CheckRow("Juego original (sin efectos añadidos)", s.pure_game != l.pure_game, s.pure_game,
					             "Apaga todos los efectos visuales que añade el emulador (ray tracing, luz volumétrica, sombras RT, oclusión, luz de antorchas, destellos, SSGI, reflejos RT). Se ve el juego tal cual. Rendimiento, reescalador, frame generation y latencia no cambian."))
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
					CheckRow("G-Sync", true, s.gsync, "G-Sync / FreeSync: el emulador no ata su reloj a los Hz del monitor");
					if (ChoiceRow("Metricas", false, kMetrics, kMetricsCount, s.metrics, "Panel dentro del juego (F8 lo cambia en vivo): FPS, el modo de frame generation, latencia y el detalle. Ya no abre una ventana aparte.")) live_changed = true;
					static const Choice aspects[] = {{"16:9", "Monitor 16:9"}, {"21:9", "Monitor 21:9"}, {"32:9", "Monitor 32:9"}};
					ChoiceRow("Monitor", true, aspects, 3, s.aspect, "Monitores panoramicos (ultrawide): el juego se dibuja a 21:9 o 32:9 en vez de estirarse. Funciona con DLSS, XeSS y FSR. Los videos y algunos menus se ven estirados.");
					CheckRow("Mantener 16:9 (bandas negras)", true, s.aspect_fit);
					static const Choice caps[] = {{"0", "Auto (bajo el refresco)"}, {"60", "60"}, {"90", "90"}, {"120", "120"}, {"144", "144"}, {"165", "165"}, {"224", "224"}};
					std::string cap = s.fps_cap > 0 ? std::to_string(static_cast<int>(s.fps_cap)) : "0";
					if (ChoiceRow("Tope de FPS (G-Sync, sin FG)", false, caps, 7, cap, "En vivo con G-Sync sin frame generation")) {
						s.fps_cap    = static_cast<float>(std::atof(cap.c_str()));
						live_changed = true;
					}
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
						             "Muestra el lanzador (estas opciones, Optimizar según tu equipo y JUGAR) antes del juego. No: el juego arranca directo; las opciones siguen en F1. \"Demon's Souls.exe\" --lanzador lo muestra una vez."))
							s.start_menu = show;
					}
					// (10-28) Mejoras para equipos de 16 GB de RAM: the launcher's 16 GB profile, on PCs of 20 GB or less.
					const bool small = s.simulate_ram16 || (g_rt.pc.ram_gb > 0 && g_rt.pc.ram_gb <= 20);
					ImGui::BeginDisabled(!small);
					CheckRow("Mejoras para equipos de 16 GB de RAM", s.ram16_boost != g_rt.launched.ram16_boost, s.ram16_boost,
					         "Perfil de memoria de 16 GB (6 GB fijos en RAM, la precarga espera si queda poca memoria). Solo en PCs de 20 GB de RAM o menos. No cambia los gráficos.");
					ImGui::EndDisabled();
					s.profile16 = LowRam(s, g_rt.pc);
					if (!small) ImGui::TextColored(kDim, "RAM %d GB: no hacen falta (solo en PCs de 20 GB o menos).", g_rt.pc.ram_gb);
					else if (s.ram16_boost) ImGui::TextColored(kGold, "RAM %s: mejoras activadas. Cierra los navegadores mientras juegas.", s.simulate_ram16 ? "16 GB (simulado)" : (std::to_string(g_rt.pc.ram_gb) + " GB").c_str());
					CheckRow("Modo GPU seguro", true, s.safe_gpu, "Para gráficas que se cuelgan o dan pantallazo azul: más estable, ~37% menos FPS.");
					CheckRow("Usar todos los CCD (Ryzen de dos CCD)", true, s.all_ccds, "Ryzen de dos CCD: No = el juego en el CCD con más caché L3 (más rápido); Sí = todos los núcleos.");
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
					SliderRow("Sensibilidad del ratón", s.kbm_sens, 0.05f, 4.0f, "%.2f", "Cuánto gira la cámara por cada movimiento del ratón (1.00 = normal).");
					restart_note(s.kbm_sens != l.kbm_sens);
					SliderRow("Antizona muerta", s.kbm_antidz, 0.0f, 0.5f, "%.2f",
					          "Salta la zona muerta del stick del juego: con más, un movimiento pequeño del ratón ya gira la cámara. Si la cámara se mueve sola, bájalo.");
					restart_note(s.kbm_antidz != l.kbm_antidz);
					{
						float cam = static_cast<float>(s.kbm_camera);
						if (SliderRow("Velocidad de cámara (ratón)", cam, 5.0f, 15.0f, "%.0f",
						              "La velocidad máxima de giro de la cámara del juego con teclado y ratón. 10 = el máximo del menú del juego. 11-15 (extra, experimental): más rápido que el menú; si la cámara hace cosas raras, vuelve a 10. Sin teclado y ratón se usa la del menú del juego."))
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
					ImGui::TextColored(kDim, "%s", s.pad_map.empty() ? "Mando: botones de serie (mapa_mando en settings.ini para cambiarlos)." : "Mando: remapeado (mapa_mando en settings.ini).");
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
			ImGui::BeginChild("##desc", {0, page_h}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
			ImGui::PushFont(nullptr, 30.0f * unit);
			ImGui::TextColored(kGold, "%s", g_desc_title.empty() ? tabs[t] : g_desc_title.c_str());
			ImGui::PopFont();
			ImGui::Separator();
			ImGui::PushTextWrapPos(0.0f);
			ImGui::TextUnformatted(g_desc_text.empty() && g_start_mode ? "Flechas / cruceta: elegir y cambiar el valor.\nA / Intro: siguiente valor.\nLB / RB o Q / E (o la lista de la izquierda): cambiar de sección.\n\nTodo se guarda en settings.ini y se aplica al pulsar JUGAR. Dentro del juego, F1 abre estas mismas opciones."
			                       : g_desc_text.empty() ? "Flechas / cruceta: elegir y cambiar el valor.\nA / Intro: siguiente valor.\nLB / RB o Q / E (o la lista de la izquierda): cambiar de sección.\nB / Esc: volver al juego.\n\n(requiere reinicio): se guarda ya y se aplica en el próximo inicio. El resto se aplica al momento."
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
		ImGui::TextColored(kDim, "%s", g_status.empty() ? I18n::T("Los cambios se aplican al pulsar JUGAR (sin reiniciar). F1 dentro del juego: opciones en vivo.") : g_status.c_str());
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
			// (10-28b) "No mostrar al iniciar": settings.ini menu_inicio=0 (F1 > Sistema or --lanzador bring it back).
			bool hide = !s.start_menu;
			ImGui::AlignTextToFramePadding();
			if (ImGui::Checkbox(I18n::T("No mostrar al iniciar"), &hide)) {
				s.start_menu = !hide;
				Save();
			}
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", "El juego arrancará directo. Para volver a verlo: F1 > Sistema > Mostrar el lanzador al iniciar, o \"Demon's Souls.exe\" --lanzador.");
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

bool WantsOverlay() {
	return g_open.load(std::memory_order_relaxed);
}

unsigned long long Revision() {
	return g_revision.load(std::memory_order_relaxed);
}

void Presented() {
	if (g_first_present) {
		g_first_present = false;
		CloseSplash();
	}
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

const PortLive::Hooks kHooks {WindowCreated, OnEvent, Presented, WantsOverlay, Revision, Draw, Title};

} // namespace

void InstallMenu(const Runtime& runtime) {
	g_rt    = runtime;
	g_saved = runtime.launched;
	g_edit  = runtime.launched;
	ApplyLive(runtime.launched);
	const bool        fixed = FixedFg(runtime.launched);
	const std::string fg    = fixed ? std::string("amd-x2") : runtime.launched.framegen;
	if (fg != "sin-fg" && !fg.empty()) {
		PortLive::fg_label  = fixed ? (runtime.launched.souls_mode == "equilibrado" ? "MODO SOULS EQUILIBRADO"
		                               : runtime.launched.souls_mode == "low"       ? "MODO SOULS PC LOW PROFILE"
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
	// (10-28b) The launcher: its own window (1600x900, at most 90% of the desktop), whatever the game's display mode.
	int win_w = 1600, win_h = 900;
	if (SDL_Rect desk {}; SDL_GetDisplayUsableBounds(0, &desk) == 0 && desk.w > 0 && desk.h > 0) {
		const float k = std::min({1.0f, desk.w * 0.9f / win_w, desk.h * 0.9f / win_h});
		win_w = static_cast<int>(win_w * k);
		win_h = static_cast<int>(win_h * k);
	}
	const Uint32 flags  = SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_RESIZABLE;
	SDL_Window*  window = SDL_CreateWindow("Demon's Souls", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, win_w, win_h, flags);
	SDL_Renderer* renderer = window != nullptr ? SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC) : nullptr;
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
		wchar_t windows[MAX_PATH + 1] = {};
		GetWindowsDirectoryW(windows, MAX_PATH);
		const auto ttf = std::filesystem::path(windows) / L"Fonts" / L"segoeui.ttf";
		std::error_code ec;
		if (std::filesystem::exists(ttf, ec)) io.Fonts->AddFontFromFileTTF(ttf.string().c_str());
	}
	ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
	ImGui_ImplSDLRenderer2_Init(renderer);
	g_background_tried = false;
	g_background       = nullptr;
	g_focus_once       = true;
	SDL_ShowCursor(SDL_ENABLE);
	CloseSplash();
	while (g_start_result == 0) {
		SDL_Event e;
		while (SDL_PollEvent(&e) == 1) {
			ImGui_ImplSDL2_ProcessEvent(&e);
			if (e.type == SDL_QUIT) g_start_result = -1;
			if (e.type == SDL_KEYDOWN && e.key.repeat == 0) {
				if (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_PAGEUP) Queue({Input::Tab, -1});
				if (e.key.keysym.sym == SDLK_e || e.key.keysym.sym == SDLK_PAGEDOWN) Queue({Input::Tab, +1});
			}
			if (e.type == SDL_CONTROLLERBUTTONDOWN) {
				if (e.cbutton.button == SDL_CONTROLLER_BUTTON_LEFTSHOULDER) Queue({Input::Tab, -1});
				if (e.cbutton.button == SDL_CONTROLLER_BUTTON_RIGHTSHOULDER) Queue({Input::Tab, +1});
			}
		}
		ImGui_ImplSDLRenderer2_NewFrame();
		ImGui_ImplSDL2_NewFrame();
		ImGui::NewFrame();
		Draw(io.DisplaySize.x, io.DisplaySize.y);
		ImGui::Render();
		SDL_RenderSetScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
		ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
		SDL_RenderPresent(renderer);
	}
	if (g_start_result > 0) *settings = g_edit;
	ImGui_ImplSDLRenderer2_Shutdown(); // (destroys the background's SDL texture too)
	ImGui_ImplSDL2_Shutdown();
	if (g_background != nullptr) {
		ImGui::UnregisterUserTexture(g_background);
		IM_DELETE(g_background);
	}
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
