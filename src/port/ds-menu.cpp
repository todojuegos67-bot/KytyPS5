// "Demon's Souls" PC port: the in-game options menu. Drawn by the system overlay (src/graphics/presentation/
// systemOverlay.cpp) into the swapchain image, after the upscaler and outside frame generation: while it is open the
// frames go out without generated ones (swapchain.cpp, PortLive::menu_open), and the game gets no input.
//   F1, or R3 held + OPTIONS (OPTIONS then never reaches the game): open / close. Circle / Escape: close. L1 / R1, Q / E: the tabs.
// What the engine applies while playing (sharpness, the fps cap, the frame generation multiplier of a fixed DLSS mode,
// metrics, rumble) is applied at once; the rest is saved to settings.ini and marked "*": it needs a restart, which
// the menu offers.

#include "port/ds-port.h"

#include "local/perf-overlay.h"
#include "local/port-live.h"

#include "SDL.h"
#include "SDL_syswm.h"
#include "imgui.h"
#include "imgui_internal.h" // ImGui::RegisterUserTexture (the background image)
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
#include <mutex>
#include <string>

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
			case Input::Tab: g_tab_request = (g_tab + in.key + 5) % 5; break;
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

// A row: its label (with "*" when a change needs a restart) and a combo of `choices`.
template <typename Item, typename Id, typename Label>
bool ComboRow(const char* label, bool restart, int count, const Item* items, std::string& value, Id id, Label text,
              const char* tip = nullptr) {
	int current = 0;
	for (int i = 0; i < count; i++)
		if (value == id(items[i])) current = i;
	ImGui::AlignTextToFramePadding();
	ImGui::TextUnformatted(label);
	if (restart) {
		ImGui::SameLine(0, 4);
		ImGui::TextColored(kWarn, "*");
	}
	if (tip != nullptr && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
	ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.42f + ImGui::GetCursorStartPos().x);
	ImGui::SetNextItemWidth(-1);
	bool changed = false;
	ImGui::PushID(label);
	if (ImGui::BeginCombo("##c", text(items[current]))) {
		for (int i = 0; i < count; i++) {
			const bool selected = i == current;
			if (ImGui::Selectable(text(items[i]), selected)) {
				value   = id(items[i]);
				changed = true;
			}
			if (selected) ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	ImGui::PopID();
	return changed;
}

bool ChoiceRow(const char* label, bool restart, const Choice* items, int count, std::string& value, const char* tip = nullptr) {
	return ComboRow(label, restart, count, items, value, [](const Choice& c) { return c.id; }, [](const Choice& c) { return c.label; }, tip);
}

bool CheckRow(const char* label, bool restart, bool& value, const char* tip = nullptr) {
	ImGui::PushID(label);
	const bool changed = ImGui::Checkbox("##k", &value);
	ImGui::PopID();
	ImGui::SameLine();
	ImGui::TextUnformatted(label);
	if (restart) {
		ImGui::SameLine(0, 4);
		ImGui::TextColored(kWarn, "*");
	}
	if (tip != nullptr && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
	return changed;
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
}

void Save() {
	if (SaveSettings(g_rt.ini_path, g_edit)) {
		g_saved  = g_edit;
		g_status = "Guardado en settings.ini";
	} else {
		g_status = "No se pudo guardar settings.ini";
	}
}

void Draw(float width, float height) {
	if (g_reset_edit.exchange(false)) {
		g_edit       = g_saved;
		g_focus_once = true;
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

	ImFont* font = FindFont("segoeui");
	ImGui::PushFont(font, 24.0f * unit);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {32 * unit, 24 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {12 * unit, 12 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {12 * unit, 6 * unit});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6 * unit);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0.03f, 0.035f, 0.05f, 0.86f});
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
	const ImVec2 size {std::min(1100 * unit, width - 32), std::min(820 * unit, height - 32)};
	ImGui::SetNextWindowPos({width * 0.5f, height * 0.5f}, ImGuiCond_Always, {0.5f, 0.5f});
	ImGui::SetNextWindowSize(size, ImGuiCond_Always);
	if (g_focus_once) {
		g_focus_once = false;
		ImGui::SetNextWindowFocus();
	}
	ImGui::Begin("##PortMenu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
	ImGui::PushFont(font, 40.0f * unit);
	ImGui::TextColored(kGold, "OPCIONES");
	ImGui::PopFont();
	ImGui::SameLine();
	ImGui::SetCursorPosX(size.x - ImGui::CalcTextSize(g_rt.game_info.c_str()).x - 32 * unit);
	ImGui::TextColored(kDim, "%s", g_rt.game_info.c_str());
	ImGui::Separator();

	Settings&  s       = g_edit;
	bool live_changed = false;
	const char* tabs[5] = {"Pantalla", "Reescalado", "Frame generation", "Rendimiento", "Juego"};
	const float footer  = 150 * unit;
	if (ImGui::BeginTabBar("##tabs")) {
		for (int t = 0; t < 5; t++) {
			const ImGuiTabItemFlags flags = g_tab_request == t ? ImGuiTabItemFlags_SetSelected : 0;
			if (!ImGui::BeginTabItem(tabs[t], nullptr, flags)) continue;
			g_tab = t;
			ImGui::BeginChild("##page", {0, size.y - footer - ImGui::GetCursorPosY()}, ImGuiChildFlags_None);
			switch (t) {
				case 0: {
					CheckRow("Pantalla completa", true, s.fullscreen);
					CheckRow("HDR", true, s.hdr, "HDR10: solo con \"Usar HDR\" activado en Windows");
					CheckRow("VSync", true, s.vsync);
					if (CheckRow("Modo graficas de 8 GB", true, s.low_end, "Adaptado para graficas de 8 GB como RTX 3060, RTX 3060 Ti, RTX 3070, RTX 2070/2080, RX 6600, RX 6600 XT o RX 6650 XT: el juego a 30 FPS estables y frame generation de AMD x2 (60 en pantalla). Reescalado solo con XeSS o FSR.") &&
					    s.low_end && s.res.rfind("dlss", 0) == 0)
						s.res = "nativo";
					CheckRow("G-Sync / FreeSync", true, s.gsync, "El reloj del emulador no se ata a los Hz del monitor");
					CheckRow("FPS desbloqueados (experimental)", true, s.unlock_fps,
					         "El juego pasa de 60 FPS reales; videos, menus y cargas siguen a 60");
					CheckRow("Mantener 16:9 (bandas negras)", true, s.aspect_fit);
					static const Choice aspects[] = {{"16:9", "16:9"}, {"21:9", "21:9 (ultrapanoramico)"}, {"32:9", "32:9"}};
					ChoiceRow("Aspecto del 3D", true, aspects, 3, s.aspect);
					static const Choice caps[] = {{"0", "Auto (bajo el refresco)"}, {"60", "60"}, {"90", "90"},
					                              {"120", "120"}, {"144", "144"}, {"165", "165"}, {"224", "224"}};
					std::string cap = s.fps_cap > 0 ? std::to_string(static_cast<int>(s.fps_cap)) : "0";
					if (ChoiceRow("Tope de FPS (G-Sync, sin FG)", false, caps, 7, cap,
					              "En vivo con G-Sync + FPS desbloqueados sin frame generation")) {
						s.fps_cap    = static_cast<float>(std::atof(cap.c_str()));
						live_changed = true;
					}
					if (ChoiceRow("Metricas en pantalla", false, kMetrics, kMetricsCount, s.metrics)) live_changed = true;
					break;
				}
				case 1: {
					if (s.low_end && s.res.rfind("dlss", 0) == 0) s.res = "nativo";
					static std::vector<ResMode> shown;
					shown.clear();
					for (int i = 0; i < kResModeCount; i++)
						if (!s.low_end || std::string(kResModes[i].upscaler) != "dlss") shown.push_back(kResModes[i]);
					ComboRow("Reescalado", true, static_cast<int>(shown.size()), shown.data(), s.res, [](const ResMode& m) { return m.id; },
					         [](const ResMode& m) { return m.label; },
					         "DLSS: NVIDIA RTX. XeSS: todas las graficas. FSR: graficas de bajo rendimiento");
					if (ChoiceRow("Nitidez", (s.sharpness == "auto") != (g_rt.launched.sharpness == "auto"), kSharpness,
					              kSharpnessCount, s.sharpness))
						live_changed = true;
					ChoiceRow("Preset DLSS", true, kDlssPresets, kDlssPresetCount, s.dlss_preset);
					if (s.res.rfind("dlss", 0) == 0 && s.framegen.rfind("amd", 0) == 0)
						ImGui::TextColored(kWarn, "DLSS no va con la frame generation de AMD: se usara FSR.");
					break;
				}
				case 2: {
					static std::vector<Choice> modes;
					if (modes.empty()) {
						for (const char* id: {"sin-fg", "amd-x2", "amd-x3", "amd-x4", "amd-120", "amd-dinamico", "dlss-x2",
						                      "dlss-x3", "dlss-x4", "dlss-x5", "dlss-x6", "dlss-120", "dlss-dinamico"})
							modes.push_back({id, FrameGenLabel(id)});
					}
					if (s.low_end) ImGui::TextColored(kWarn, "Modo graficas de 8 GB: frame generation fija en AMD x2 (30 reales, 60 en pantalla).");
					ImGui::BeginDisabled(s.low_end);
					std::string locked = "amd-x2";
					ChoiceRow("Modo", true, modes.data(), static_cast<int>(modes.size()), s.low_end ? locked : s.framegen,
					          "AMD: todas las graficas. DLSS: NVIDIA RTX 40/50");
					ImGui::EndDisabled();
					const int max_gen = g_rt.launched.low_end ? 0 : FixedDlssGenerated(g_rt.launched.framegen);
					if (max_gen > 1 && s.framegen == g_rt.launched.framegen) {
						if (g_fg_live == 0) g_fg_live = max_gen;
						ImGui::AlignTextToFramePadding();
						ImGui::TextUnformatted("Multiplicador (en vivo)");
						ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.42f + ImGui::GetCursorStartPos().x);
						ImGui::SetNextItemWidth(-1);
						int shown = g_fg_live + 1;
						if (ImGui::SliderInt("##mult", &shown, 2, max_gen + 1, "x%d")) {
							g_fg_live            = shown - 1;
							PortLive::fg_generate = g_fg_live;
						}
					}
					ImGui::TextColored(kDim, "El menu se muestra sin frames generados (no se arrastra con la imagen).");
					break;
				}
				case 3: {
					CheckRow("Equipo de 16 GB RAM", true, s.profile16, "6 GB del juego fijos en RAM; la precarga espera con poca RAM libre");
					CheckRow("Modo GPU seguro", true, s.safe_gpu, "Para graficas que se cuelgan: mas estable, ~37% menos FPS");
					ChoiceRow("Texturas", true, kTextureReduce, kTextureReduceCount, s.texture_reduce);
					ChoiceRow("Distancia de dibujado", true, kQualities, kQualityCount, s.quality, "Solo el detalle lejano (LOD); Maximo = el juego tal cual");
					CheckRow("Simular grafica de 8 GB", true, s.simulate_vram8);
					CheckRow("Usar todos los CCD (Ryzen de dos CCD)", true, s.all_ccds);
					break;
				}
				case 4: {
					std::string lang = std::to_string(s.language);
					if (ChoiceRow("Idioma", true, kLanguages, kLanguageCount, lang)) s.language = std::atoi(lang.c_str());
					if (CheckRow("Vibracion", false, s.rumble)) live_changed = true;
					ImGui::AlignTextToFramePadding();
					ImGui::TextUnformatted("Fuerza de la vibracion");
					ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.42f + ImGui::GetCursorStartPos().x);
					ImGui::SetNextItemWidth(-1);
					if (ImGui::SliderFloat("##gain", &s.rumble_gain, 1.0f, 8.0f, "%.1f")) live_changed = true;
					CheckRow("Copia de las partidas al arrancar", false, s.backup_saves);
					ImGui::Spacing();
					ImGui::TextColored(kDim, "Carpeta del juego:");
					ImGui::TextWrapped("%s", s.game.c_str());
					ImGui::TextColored(kDim, "Para cambiarla, borra carpeta_juego en settings.ini.");
					ImGui::TextColored(kDim, "F8: metricas.  F11 / Alt+Enter: pantalla completa.");
					break;
				}
				default: break;
			}
			ImGui::EndChild();
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}
	g_tab_request = -1;
	g_popup       = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
	if (live_changed) ApplyLive(s);

	ImGui::SetCursorPosY(size.y - footer + 8 * unit);
	ImGui::Separator();
	const bool needs = NeedsRestart(g_rt.launched, s);
	if (needs) ImGui::TextColored(kWarn, "* Hay cambios que necesitan reiniciar el juego (guarda antes en una hoguera / al salir).");
	else ImGui::TextColored(kDim, "%s", g_status.empty() ? "* = necesita reiniciar.   Circulo / Esc: volver al juego." : g_status.c_str());
	const float bw = 220 * unit, bh = 52 * unit;
	if (ImGui::Button("Guardar y volver", {bw, bh})) {
		Save();
		ApplyLive(s);
		SetOpen(false);
	}
	ImGui::SameLine();
	ImGui::BeginDisabled(!needs);
	if (ImGui::Button("Reiniciar ahora", {bw, bh})) {
		Save();
		RequestRestart();
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button("Descartar", {bw, bh})) {
		g_edit = g_saved;
		ApplyLive(g_saved);
		SetOpen(false);
	}
	ImGui::SameLine();
	if (ImGui::Button("Salir del juego", {bw, bh})) RequestQuit();
	ImGui::End();
	ImGui::PopStyleColor(13);
	ImGui::PopStyleVar(4);
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
	const std::string fg = runtime.launched.low_end ? std::string("amd-x2") : runtime.launched.framegen;
	if (fg != "sin-fg" && !fg.empty()) {
		PortLive::fg_label  = FrameGenLabel(fg);
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
		      ", Distancia de dibujado " + level(kQualities, kQualityCount, runtime.launched.quality);
		PortLive::gfx_line = gfx.c_str();
	}
	PortLive::g_hooks = &kHooks;
}

void RequestQuit() {
	SDL_Event quit {};
	quit.type = SDL_QUIT;
	SDL_PushEvent(&quit);
}

} // namespace Port
