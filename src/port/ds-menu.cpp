// "Demon's Souls" PC port: the in-game options menu. Drawn by the system overlay (src/graphics/presentation/
// systemOverlay.cpp) into the swapchain image, after the upscaler and outside frame generation: while it is open the
// frames go out without generated ones (swapchain.cpp, PortLive::menu_open), and the game gets no input.
//   F1, or R3 held + OPTIONS (OPTIONS then never reaches the game): open / close. Circle / Escape: close. L1 / R1, Q / E: the tabs.
// What the engine applies while playing (sharpness, the fps cap, the frame generation multiplier of a fixed DLSS mode,
// metrics, rumble) is applied at once; the rest is saved to settings.ini and marked "*": it needs a restart, which
// the menu offers.

#include "port/ds-port.h"
#include "port/ds-update.h"

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
constexpr int            kTabCount = 9;
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

void InfoLine(const char* caption, const std::string& value, const ImVec4& color) {
	ImGui::TextColored(kDim, "%s", caption);
	ImGui::PushStyleColor(ImGuiCol_Text, color);
	ImGui::TextWrapped("%s", value.c_str());
	ImGui::PopStyleColor();
	ImGui::Spacing();
}

void DrawPcTab(float unit) {
	const ImVec4 ink {0.93f, 0.91f, 0.88f, 1.0f};
	const ImVec4 red {0.92f, 0.35f, 0.31f, 1.0f};
	ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0.12f, 0.10f, 0.08f, 0.85f});
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 14 * unit);
	ImGui::BeginChild("##card", {0, 330 * unit}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
	ImGui::PushFont(nullptr, 30.0f * unit);
	ImGui::TextColored(kGold, "TU EQUIPO");
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
	} else if (ImGui::Button("Actualizar version", {bw, bh})) {
		SetUpd(UpdateUi::Checking, "Buscando la ultima version en GitHub...");
		StartWorker(CheckWork);
	}
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("Busca la ultima version publicada en GitHub y la instala en esta misma carpeta. Guarda los archivos del motor actuales en _backup_update (las 2 ultimas) y no toca la cache de shaders, las partidas, los seeds ni los ajustes.");
	ImGui::SameLine();
	if (ImGui::Button("Volver a la version anterior", {bw, bh})) {
		std::scoped_lock lock(g_upd.mutex);
		g_upd.backups = Update::ListBackups(PortRoot());
		g_upd.chosen  = 0;
		g_upd.state   = g_upd.backups.empty() ? UpdateUi::Failed : UpdateUi::Choosing;
		g_upd.text    = g_upd.backups.empty() ? "No hay versiones anteriores guardadas" : "Elige la version a restaurar:";
		state         = g_upd.state;
		text          = g_upd.text;
	}
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("Vuelve a una version guardada en _backup_update. Antes guarda la actual (se puede deshacer). No toca la cache de shaders, las partidas, los seeds ni los ajustes.");
	ImGui::EndDisabled();
	if (!text.empty()) ImGui::TextWrapped("%s", text.c_str());
	if (state == UpdateUi::Working && g_upd.total.load() > 0 && text.rfind("Descargando", 0) == 0) {
		const float f = static_cast<float>(static_cast<double>(g_upd.done.load()) / static_cast<double>(g_upd.total.load()));
		char        label[64];
		std::snprintf(label, sizeof(label), "%.0f / %.0f MB", g_upd.done.load() / 1048576.0, g_upd.total.load() / 1048576.0);
		ImGui::ProgressBar(f, {-1, 0}, label);
		if (ImGui::Button("Cancelar")) g_upd.cancel = true;
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
		if (ImGui::Button("Restaurar", {bw * 0.6f, bh})) StartWorker(RestoreWork);
		ImGui::SameLine();
		if (ImGui::Button("Cancelar##r", {bw * 0.6f, bh})) SetUpd(UpdateUi::Idle, "");
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
	// (10-27) The tabs are Launcher-souls.ps1's groups, in its order, with its labels and tooltips.
	const char* tabs[kTabCount] = {"Imagen", "Frame generation", "Calidad grafica", "Ray tracing", "Modos Souls",
	                               "Pantalla y otros", "Rendimiento / PC", "Controles", "Tu equipo"};
	const float footer  = 150 * unit;
	const bool  nvidia  = g_rt.pc.nvidia;
	if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
		for (int t = 0; t < kTabCount; t++) {
			const ImGuiTabItemFlags flags = g_tab_request == t ? ImGuiTabItemFlags_SetSelected : 0;
			if (!ImGui::BeginTabItem(tabs[t], nullptr, flags)) continue;
			g_tab = t;
			ImGui::BeginChild("##page", {0, size.y - footer - ImGui::GetCursorPosY()}, ImGuiChildFlags_None);
			const bool fixed = FixedFg(s);
			switch (t) {
				case 0: { // Imagen
					static std::vector<ResMode> shown;
					shown.clear();
					for (int i = 0; i < kResModeCount; i++)
						if (nvidia || std::strcmp(kResModes[i].upscaler, "dlss") != 0) shown.push_back(kResModes[i]);
					ComboRow("Reescalado", true, static_cast<int>(shown.size()), shown.data(), s.res, [](const ResMode& m) { return m.id; },
					         [](const ResMode& m) { return m.label; },
					         "Reescalado: dibuja el 3D a menos resolucion y lo sube a 1440 (mas fps). DLSS: NVIDIA RTX. XeSS y FSR: todas las graficas.");
					ImGui::BeginDisabled(!nvidia);
					ChoiceRow("Preset DLSS", true, kDlssPresets, kDlssPresetCount, s.dlss_preset,
					          "Modelo de DLSS: auto (el de NVIDIA), K = DLSS 4, L y M = DLSS 4.5 (mas nitidos, algo mas lentos)");
					ImGui::EndDisabled();
					if (ChoiceRow("Nitidez", (s.sharpness == "auto") != (g_rt.launched.sharpness == "auto"), kSharpness, kSharpnessCount, s.sharpness,
					              "Nitidez del reescalado (DLSS, XeSS y FSR). Mas alto: mas nitido, pero puede marcar bordes y ruido."))
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
						ImGui::TextUnformatted("Multiplicador (en vivo)");
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
				case 2: { // Calidad grafica
					ChoiceRow("Texturas", true, kTextureReduce, kTextureReduceCount, s.texture_reduce, "Texturas: Medio = mitad de tamano, Bajo = un cuarto. Lo que mas VRAM ahorra.");
					ChoiceRow("Distancia", true, kQualities, kQualityCount, s.quality, "Distancia de dibujado: a menos nivel, los objetos lejanos pasan antes a modelos mas simples. Mas FPS si va justo el procesador.");
					ChoiceRow("Sombras", true, kShadows, kShadowCount, s.shadows, "Sombras: Medio = las del sol hasta 125 m, Bajo = hasta 60 m y sin sombras de focos ni ambientales. Mas FPS, pero pueden aparecer de golpe.");
					ChoiceRow("Detalle", true, kDetails, kDetailCount, s.detail, "Detalle del juego: sombras (cascadas y distancia), iluminacion de entorno y relieve (teselado). Ultra > Muy alto (modo cine) > Alto (modo rendimiento de PS5) > Normal > Bajo (graficas modestas).");
					ChoiceRow("Reflejos", true, kReflections, kReflectionCount, s.reflections, "Reflejos del propio juego (agua, charcos, suelos): mas pasos y mas alcance en Ultra. Cuesta algo de GPU.");
					ChoiceRow("Desvanecido objetos", true, kReflections, kReflectionCount, s.object_fade, "Distancia a la que los objetos lejanos se desvanecen. Ultra: aparecen de golpe mucho menos.");
					ChoiceRow("Nitidez AA", true, kAaSharp, kAaSharpCount, s.aa_sharp, "Nitidez del antialiasing del juego. No cuesta FPS.");
					CheckRow("Filtrado x16", true, s.af16, "Filtrado anisotropico x16 en las texturas: suelos y paredes nitidos a lo lejos y en angulo. Cuesta muy poco.");
					ChoiceRow("Texturas (nitidez)", true, kTextureSharpness, kTextureSharpnessCount, s.texture_sharpness,
					          "Nitidez de las texturas. Nitidas: el detalle de la resolucion final con reescalado. Extra nitidas: un escalon mas (lo que recomienda NVIDIA para DLSS); mas detalle en suelos y paredes, puede brillar algo en rejas y hierba. Cuesta pocos FPS.");
					ImGui::Spacing();
					ImGui::TextColored(kDim, "Postprocesado del juego (marcado = como viene):");
					CheckRow("Desenfoque de movimiento", true, s.motion_blur);
					CheckRow("Profundidad de campo", true, s.dof);
					CheckRow("Vineteado", true, s.vignette);
					CheckRow("Resplandor (bloom)", true, s.bloom);
					CheckRow("Adaptacion del ojo", true, s.eye_adapt);
					ImGui::Spacing();
					ChoiceRow("Bordes sombras", true, kShadowEdges, kShadowEdgeCount, s.shadow_edges,
					          "Filtrado de los bordes de las sombras del juego (antorchas, lamparas y sol): Alta quita los dientes de sierra (casi gratis), Muy alta bordes suaves, Ultra nitida al pie y suave al alejarse (~0,5-1,5 ms). Al cambiar de nivel hay un tiron la primera vez.");
					ChoiceRow("Sombra rincones", true, kCornerShadows, kCornerShadowCount, s.corner_shadow,
					          "Rincones, esquinas, debajo de mesas y carros y pies mas oscuros y asentados: mas fuerza en la oclusion de Ray tracing. Necesita 'Oclusion' (Ray tracing) encendido (si no, no hace nada). Sin coste.");
					break;
				}
				case 3: { // Ray tracing
					static const Choice yes_no[] = {{"0", "No"}, {"1", "Si"}};
					const auto row = [&](const char* label, bool& v, const char* tip) {
						std::string x = v ? "1" : "0";
						if (ChoiceRow(label, true, yes_no, 2, x, tip)) v = x == "1";
					};
					row("Reflejos", s.rt_reflections, "Reflejos con trazado de rayos: refleja tambien lo que esta fuera de pantalla o tapado, con sus texturas.");
					row("Sombras sol", s.rt_sun, "Sombras del sol con trazado de rayos: sombras de contacto y las que faltaban.");
					row("Niebla volum.", s.rt_fog, "Niebla y haces de luz por puertas y ventanas cortados por la geometria real, sin parpadeos. Solo suma luz.");
					row("Particulas", s.rt_particles, "El fuego, las chispas y la magia iluminan lo que tienen cerca, con sombras reales. Muy barato.");
					row("Metal/armadura", s.rt_metal, "Reflejos con trazado de rayos en armaduras, espadas, escudos y herrajes de metal.");
					row("Oclusion", s.rt_ao, "Oclusion ambiental con trazado de rayos: rincones, contactos y objetos asentados con la geometria real.");
					ImGui::Spacing();
					ImGui::TextColored(kDim, "Al elegir cualquier Modo Souls todo el ray tracing pasa a No (se enciende a mano despues).");
					break;
				}
				case 4: { // Modos Souls
					std::string mode = s.souls_mode;
					if (ChoiceRow("Modo Souls", true, kSoulsModes, kSoulsModeCount, mode,
					              "Ligero: graficas de 8 GB (RTX 3060-3070, RX 6600): 30 FPS estables y AMD x2 (60 en pantalla). Equilibrado: potencia de una RTX 3080: 45 FPS y AMD x2 (90). "
					              "Ultra: RTX 5060-5070 Ti, RX 9070. Extremo: para enfermos de Souls (RTX 5080 / 5090). PC low profile: PCs modestos, 900p y todo en bajo. Los FPS dependen de tu GPU y tu CPU.")) {
						s.souls_mode = mode;
						ApplyPreset(&s, mode == "ninguno" ? std::string("maximo") : mode);
					}
					ImGui::Spacing();
					ImGui::TextWrapped("Ligero: adaptado para graficas de 8 GB como RTX 3060, RTX 3060 Ti, RTX 3070, RTX 2070/2080, RX 6600: el juego a 30 FPS estables y frame generation de AMD x2 (60 en pantalla).");
					ImGui::TextWrapped("Equilibrado: RTX 2080 Ti, 3070 Ti, 3080, 4060 Ti, 4070, RX 6800-6950, RX 7700-7800, Arc B580: 45 FPS estables y AMD x2 (90 en pantalla).");
					ImGui::TextWrapped("Ultra: RTX 5060 / 5060 Ti / 5070 / 5070 Ti y RX 9070 / 9070 XT: todo al maximo y Detalle Muy alto.");
					ImGui::TextWrapped("Extremo: RTX 5080, RTX 5090 o superior: Ultra en todo, filtrado x16 y texturas nitidas.");
					ImGui::TextWrapped("PC low profile: 30 FPS reales con AMD x2, FSR Calidad (nitidez 100 %%) y todos los graficos en bajo.");
					if (!g_rt.pc.recommended.empty()) ImGui::TextColored(kGold, "Recomendado para tu grafica: %s", g_rt.pc.recommended.c_str());
					break;
				}
				case 5: { // Pantalla y otros
					CheckRow("Pantalla completa", true, s.fullscreen);
					CheckRow("HDR", true, s.hdr, "HDR10: activa antes \"Usar HDR\" en la configuracion de pantalla de Windows");
					CheckRow("VSync", true, s.vsync, "VSync: sin cortes de imagen (tearing); con FreeSync/G-Sync sigue funcionando");
					CheckRow("G-Sync", true, s.gsync, "G-Sync / FreeSync: el emulador no ata su reloj a los Hz del monitor");
					if (ChoiceRow("Metricas", false, kMetrics, kMetricsCount, s.metrics, "Muestra encima del juego los FPS, el modo de frame generation y el detalle")) live_changed = true;
					static const Choice aspects[] = {{"16:9", "Monitor 16:9"}, {"21:9", "Monitor 21:9"}, {"32:9", "Monitor 32:9"}};
					ChoiceRow("Monitor", true, aspects, 3, s.aspect, "Monitores panoramicos (ultrawide): el juego se dibuja a 21:9 o 32:9 en vez de estirarse. Funciona con DLSS, XeSS y FSR. Los videos y algunos menus se ven estirados.");
					CheckRow("Mantener 16:9 (bandas negras)", true, s.aspect_fit);
					static const Choice caps[] = {{"0", "Auto (bajo el refresco)"}, {"60", "60"}, {"90", "90"}, {"120", "120"}, {"144", "144"}, {"165", "165"}, {"224", "224"}};
					std::string cap = s.fps_cap > 0 ? std::to_string(static_cast<int>(s.fps_cap)) : "0";
					if (ChoiceRow("Tope de FPS (G-Sync, sin FG)", false, caps, 7, cap, "En vivo con G-Sync sin frame generation")) {
						s.fps_cap    = static_cast<float>(std::atof(cap.c_str()));
						live_changed = true;
					}
					std::string lang = std::to_string(s.language);
					if (ChoiceRow("Idioma", true, kLanguages, kLanguageCount, lang)) s.language = std::atoi(lang.c_str());
					CheckRow("Copia de las partidas al arrancar", false, s.backup_saves);
					ImGui::Spacing();
					ImGui::TextColored(kDim, "Carpeta del juego:");
					ImGui::TextWrapped("%s", s.game.c_str());
					ImGui::TextColored(kDim, "Para cambiarla, borra carpeta_juego en settings.ini.");
					break;
				}
				case 6: { // Rendimiento / PC
					const bool low = LowRam(s, g_rt.pc);
					bool       p16 = low ? true : s.profile16;
					ImGui::BeginDisabled(low);
					if (CheckRow("Optimizacion equipos 16 GB RAM", true, p16,
					             "Solo aplica el perfil de memoria de 16 GB en PCs reales de 20 GB de RAM o menos, o con \"Simular 16 GB RAM\". No cambia los graficos.") &&
					    !low)
						s.profile16 = p16;
					ImGui::EndDisabled();
					if (low) {
						s.profile16 = true;
						ImGui::TextColored(kGold, "RAM %s: perfil seguro. Cierra navegadores", s.simulate_ram16 ? "16 GB (simulado)" : (std::to_string(g_rt.pc.ram_gb) + " GB").c_str());
					}
					CheckRow("Simular 16 GB RAM", true, s.simulate_ram16, "Solo el juego: como mucho 12 GB en RAM (4 GB para Windows), el resto a disco, como un PC de 16 GB.");
					CheckRow("Simular grafica 8 GB", true, s.simulate_vram8, "El juego solo puede usar 8 GB de VRAM, como una grafica de 8 GB.");
					CheckRow("Modo GPU seguro", true, s.safe_gpu, "Para graficas que se cuelgan o dan pantallazo azul: mas estable, ~37% menos FPS");
					CheckRow("Usar todos los CCD (Ryzen de dos CCD)", true, s.all_ccds);
					break;
				}
				case 7: { // Controles
					CheckRow("Teclado y raton", true, s.kbm, "Jugar con teclado y raton: WASD para moverse, el raton mueve la camara, clic izquierdo R1, clic derecho L1, rueda o clic central fija objetivo. Las teclas: mapa_teclado en settings.ini. F7: solo el raton en vivo");
					ImGui::BeginDisabled(!s.kbm);
					ImGui::AlignTextToFramePadding();
					ImGui::TextUnformatted("Sensibilidad");
					if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cuanto gira la camara por cada movimiento del raton (1.00 = normal).");
					ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.42f + ImGui::GetCursorStartPos().x);
					ImGui::SetNextItemWidth(-1);
					ImGui::SliderFloat("##sens", &s.kbm_sens, 0.1f, 5.0f, "%.2f");
					ImGui::AlignTextToFramePadding();
					ImGui::TextUnformatted("Antizona muerta");
					if (ImGui::IsItemHovered()) ImGui::SetTooltip("Salta la zona muerta del stick del juego: con mas, un movimiento pequeno del raton ya gira la camara. Si la camara se mueve sola, bajalo.");
					ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.42f + ImGui::GetCursorStartPos().x);
					ImGui::SetNextItemWidth(-1);
					ImGui::SliderFloat("##adz", &s.kbm_antidz, 0.0f, 0.6f, "%.2f");
					ImGui::EndDisabled();
					if (CheckRow("Vibracion", false, s.rumble)) live_changed = true;
					ImGui::AlignTextToFramePadding();
					ImGui::TextUnformatted("Fuerza de la vibracion");
					ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.42f + ImGui::GetCursorStartPos().x);
					ImGui::SetNextItemWidth(-1);
					if (ImGui::SliderFloat("##gain", &s.rumble_gain, 1.0f, 8.0f, "%.1f")) live_changed = true;
					if (!s.pad_map.empty()) ImGui::TextColored(kDim, "Mando remapeado (mapa_mando en settings.ini).");
					ImGui::TextColored(kDim, "F8: metricas.  F11 / Alt+Enter: pantalla completa.");
					break;
				}
				case 8: DrawPcTab(unit); break; // Tu equipo
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
	PortLive::g_hooks = &kHooks;
}

void RequestQuit() {
	SDL_Event quit {};
	quit.type = SDL_QUIT;
	SDL_PushEvent(&quit);
}

} // namespace Port
