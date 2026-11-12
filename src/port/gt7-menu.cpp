// "Gran Turismo 7" PC port: the launcher's window and the metrics window while playing (gt7-port.h). Ported from the
// Demon's Souls port's launcher (ds-menu.cpp): the same sections on the left, the options in the middle, the
// description on the right, JUGAR; mouse, keyboard and controller (DualSense, 8BitDo, Xbox) navigation.

#include "port/gt7-i18n.h"
#include "port/gt7-port.h"
#include "port/gt7-settings.h"
#include "port/gt7-steam.h"

#include <windows.h>
#include <dxgi.h>
#include <pdh.h>
#include <psapi.h>
#include <shellapi.h>

#include <SDL3/SDL.h>
#include <cstdlib>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#include "misc/cpp/imgui_stdlib.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace Port {

namespace {

using I18n::T;

const ImVec4 kAccent {0.86f, 0.12f, 0.16f, 1.0f}; // GT red
const ImVec4 kText {0.93f, 0.95f, 0.98f, 1.0f};
const ImVec4 kDim {0.60f, 0.66f, 0.74f, 1.0f};
const ImVec4 kBlue {0.36f, 0.62f, 0.98f, 1.0f};

// ---- A window with Dear ImGui on SDL_Renderer. ----

struct Gui {
	SDL_Window*   window   = nullptr;
	SDL_Renderer* renderer = nullptr;
	ImGuiContext* context  = nullptr;
	ImFont*       body     = nullptr;
	ImFont*       title    = nullptr;
	SDL_Texture*  background = nullptr;
	int           bg_w = 0, bg_h = 0;

	bool Open(const char* name, int w, int h, SDL_WindowFlags flags) {
		window = SDL_CreateWindow(name, w, h, flags);
		if (window == nullptr) return false;
		renderer = SDL_CreateRenderer(window, nullptr);
		if (renderer == nullptr) return false;
		SDL_SetRenderVSync(renderer, 1);
		IMGUI_CHECKVERSION();
		context = ImGui::CreateContext();
		ImGui::SetCurrentContext(context);
		ImGuiIO& io = ImGui::GetIO();
		io.IniFilename = nullptr;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
		ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
		ImGui_ImplSDLRenderer3_Init(renderer);
		LoadFonts();
		Style();
		return true;
	}

	void LoadFonts() {
		ImGuiIO&     io = ImGui::GetIO();
		ImFontConfig cfg;
		wchar_t      windir[MAX_PATH] = {};
		GetWindowsDirectoryW(windir, MAX_PATH);
		const fs::path fonts = fs::path(windir) / L"Fonts";
		std::error_code ec;
		const fs::path segoe = fonts / L"segoeui.ttf";
		if (fs::exists(segoe, ec)) body = io.Fonts->AddFontFromFileTTF(Utf8(segoe.wstring()).c_str(), 20.0f, &cfg);
		if (body == nullptr) body = io.Fonts->AddFontDefault();
		// The display face (resource 3, as the Demon's Souls port's JUGAR).
		HMODULE module = GetModuleHandleW(nullptr);
		if (HRSRC res = FindResourceW(module, MAKEINTRESOURCEW(3), MAKEINTRESOURCEW(10)); res != nullptr) {
			HGLOBAL     data = LoadResource(module, res);
			const DWORD size = SizeofResource(module, res);
			void*       p    = data != nullptr ? LockResource(data) : nullptr;
			if (p != nullptr && size > 0) {
				ImFontConfig tc;
				tc.FontDataOwnedByAtlas = false;
				title = io.Fonts->AddFontFromMemoryTTF(p, static_cast<int>(size), 40.0f, &tc);
			}
		}
		if (title == nullptr) title = body;
	}

	static void Style() {
		ImGuiStyle& st = ImGui::GetStyle();
		ImGui::StyleColorsDark(&st);
		st.WindowRounding = 6.0f;
		st.FrameRounding  = 4.0f;
		st.GrabRounding   = 4.0f;
		st.FramePadding   = {10.0f, 6.0f};
		st.ItemSpacing    = {10.0f, 9.0f};
		ImVec4* c = st.Colors;
		c[ImGuiCol_WindowBg]        = {0.04f, 0.06f, 0.11f, 0.92f};
		c[ImGuiCol_ChildBg]         = {0.06f, 0.09f, 0.16f, 0.80f};
		c[ImGuiCol_FrameBg]         = {0.11f, 0.15f, 0.25f, 1.0f};
		c[ImGuiCol_FrameBgHovered]  = {0.17f, 0.23f, 0.37f, 1.0f};
		c[ImGuiCol_FrameBgActive]   = {0.20f, 0.28f, 0.45f, 1.0f};
		c[ImGuiCol_Button]          = {0.13f, 0.19f, 0.33f, 1.0f};
		c[ImGuiCol_ButtonHovered]   = {0.22f, 0.32f, 0.55f, 1.0f};
		c[ImGuiCol_ButtonActive]    = {0.28f, 0.40f, 0.66f, 1.0f};
		c[ImGuiCol_Header]          = {0.16f, 0.24f, 0.42f, 1.0f};
		c[ImGuiCol_HeaderHovered]   = {0.22f, 0.32f, 0.55f, 1.0f};
		c[ImGuiCol_HeaderActive]    = {0.86f, 0.12f, 0.16f, 0.85f};
		c[ImGuiCol_CheckMark]       = kAccent;
		c[ImGuiCol_SliderGrab]      = kAccent;
		c[ImGuiCol_SliderGrabActive] = {1.0f, 0.3f, 0.3f, 1.0f};
		c[ImGuiCol_NavCursor]       = kAccent;
		c[ImGuiCol_Separator]       = {0.25f, 0.30f, 0.42f, 1.0f};
	}

	void LoadBackground(const fs::path& image) {
		std::ifstream     in(image, std::ios::binary);
		const std::string bytes {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
		if (bytes.empty()) return;
		int   ch = 0;
		auto* px = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()), &bg_w, &bg_h, &ch, 4);
		if (px == nullptr) return;
		background = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, bg_w, bg_h);
		if (background != nullptr) SDL_UpdateTexture(background, nullptr, px, bg_w * 4);
		stbi_image_free(px);
	}

	void Begin() {
		ImGui::SetCurrentContext(context);
		ImGui_ImplSDLRenderer3_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();
	}

	void End(ImVec4 clear) {
		ImGui::Render();
		SDL_SetRenderDrawColorFloat(renderer, clear.x, clear.y, clear.z, clear.w);
		SDL_RenderClear(renderer);
		if (background != nullptr) {
			int w = 0, h = 0;
			SDL_GetRenderOutputSize(renderer, &w, &h);
			const float scale = std::max(static_cast<float>(w) / bg_w, static_cast<float>(h) / bg_h);
			SDL_FRect   dst {(w - bg_w * scale) / 2.0f, (h - bg_h * scale) / 2.0f, bg_w * scale, bg_h * scale};
			SDL_RenderTexture(renderer, background, nullptr, &dst);
		}
		ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
		SDL_RenderPresent(renderer);
	}

	void Close() {
		if (context != nullptr) {
			ImGui::SetCurrentContext(context);
			ImGui_ImplSDLRenderer3_Shutdown();
			ImGui_ImplSDL3_Shutdown();
			ImGui::DestroyContext(context);
			context = nullptr;
		}
		if (background != nullptr) SDL_DestroyTexture(background);
		if (renderer != nullptr) SDL_DestroyRenderer(renderer);
		if (window != nullptr) SDL_DestroyWindow(window);
		background = nullptr;
		renderer   = nullptr;
		window     = nullptr;
	}
};

// ---- The rows (a label on the left, the control on the right; hovering one shows its description). ----

std::string g_desc_title, g_desc_text;

void Describe(const char* label, const char* text) {
	if (ImGui::IsItemHovered() || ImGui::IsItemFocused()) {
		g_desc_title = T(label);
		g_desc_text  = text != nullptr ? T(text) : "";
	}
}

void Label(const char* label) {
	ImGui::AlignTextToFramePadding();
	ImGui::TextUnformatted(T(label));
	ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.45f);
	ImGui::SetNextItemWidth(-FLT_MIN);
}

bool ChoiceRow(const char* label, std::string& value, const Choice* choices, int count, const char* desc) {
	Label(label);
	int current = 0;
	for (int i = 0; i < count; i++)
		if (value == choices[i].id) current = i;
	bool changed = false;
	ImGui::PushID(label);
	if (ImGui::BeginCombo("##c", T(choices[current].label))) {
		for (int i = 0; i < count; i++) {
			if (ImGui::Selectable(T(choices[i].label), i == current)) {
				value   = choices[i].id;
				changed = true;
			}
			if (i == current) ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	Describe(label, desc);
	ImGui::PopID();
	return changed;
}

bool CheckRow(const char* label, bool& value, const char* desc) {
	Label(label);
	ImGui::PushID(label);
	const bool changed = ImGui::Checkbox("##b", &value);
	Describe(label, desc);
	ImGui::PopID();
	return changed;
}

bool IntRow(const char* label, int& value, int lo, int hi, const char* fmt, const char* desc) {
	Label(label);
	ImGui::PushID(label);
	const bool changed = ImGui::SliderInt("##i", &value, lo, hi, fmt);
	Describe(label, desc);
	ImGui::PopID();
	return changed;
}

bool FloatRow(const char* label, float& value, float lo, float hi, const char* fmt, const char* desc) {
	Label(label);
	ImGui::PushID(label);
	const bool changed = ImGui::SliderFloat("##f", &value, lo, hi, fmt);
	Describe(label, desc);
	ImGui::PopID();
	return changed;
}

bool TextRow(const char* label, std::string& value, const char* desc) {
	Label(label);
	ImGui::PushID(label);
	const bool changed = ImGui::InputText("##t", &value);
	Describe(label, desc);
	ImGui::PopID();
	return changed;
}

void Heading(const char* text) {
	ImGui::Spacing();
	ImGui::TextColored(kAccent, "%s", T(text));
	ImGui::Separator();
}

void Note(const char* text) {
	ImGui::PushTextWrapPos(0.0f);
	ImGui::TextColored(kDim, "%s", T(text));
	ImGui::PopTextWrapPos();
}

std::string Gb(uint64_t bytes) {
	char buf[32];
	std::snprintf(buf, sizeof(buf), "%.1f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
	return buf;
}

void OpenPath(const fs::path& p) {
	ShellExecuteW(nullptr, L"open", p.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

// ---- The sections. ----

enum Section { kHome, kGraphics, kUpscale, kFrameGen, kPerf, kPad, kWheel, kKbm, kMetrics, kPc, kDev, kSectionCount };
const char* const kSectionNames[kSectionCount] = {"Inicio",         "Gráficos", "Reescalado", "Generación de fotogramas", "Rendimiento", "Mando",
                                                  "Volante",        "Teclado y ratón", "Métricas", "Tu PC",                   "Desarrollador"};

struct State {
	const Runtime* rt = nullptr;
	Settings*      s  = nullptr;
	int            section = kHome;
	std::string    message;
	bool           dirty = false;
};

void DrawHome(State& st) {
	Settings& s = *st.s;
	Heading("Juego");
	Label("Carpeta del juego");
	ImGui::PushID("game");
	ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 130.0f);
	if (ImGui::InputText("##g", &s.game)) st.dirty = true;
	Describe("Carpeta del juego", "La carpeta del juego volcado: PPSA01317-app (con eboot.bin y sce_sys), Gran Turismo 7 versión 1.710.");
	ImGui::SameLine();
	if (ImGui::Button(T("Examinar..."), {-FLT_MIN, 0})) {
		const std::string picked = PickFolder();
		if (!picked.empty()) {
			s.game   = picked;
			st.dirty = true;
		}
	}
	ImGui::PopID();
	if (!IsGame(s.game)) ImGui::TextColored(kAccent, "%s", T("No se encuentra eboot.bin en la carpeta del juego."));
	Label("Parche (ETAHen .json)");
	ImGui::PushID("patch");
	ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 130.0f);
	if (ImGui::InputText("##p", &s.game_patch)) st.dirty = true;
	Describe("Parche (ETAHen .json)", "Opcional: un archivo de trucos / parches de ETAHen para el juego. Vacío: ninguno.");
	ImGui::SameLine();
	if (ImGui::Button(T("Examinar..."), {-FLT_MIN, 0})) {
		const std::string picked = PickFile(L"JSON", L"*.json");
		if (!picked.empty()) {
			s.game_patch = picked;
			st.dirty     = true;
		}
	}
	ImGui::PopID();

	Heading("Idioma");
	Label("Idioma del juego");
	{
		int cur = 0;
		for (int i = 0; i < kConsoleLanguageCount; i++)
			if (kConsoleLanguages[i].id == s.console_language) cur = i;
		if (ImGui::BeginCombo("##lang", kConsoleLanguages[cur].label)) {
			for (int i = 0; i < kConsoleLanguageCount; i++)
				if (ImGui::Selectable(kConsoleLanguages[i].label, i == cur)) {
					s.console_language = kConsoleLanguages[i].id;
					st.dirty           = true;
				}
			ImGui::EndCombo();
		}
		Describe("Idioma del juego", "El idioma de la consola que ve el juego (textos y voces de GT7).");
	}
	Label("Idioma del menú");
	{
		int cur = 0;
		for (int i = 0; i < I18n::kMenuLanguageCount; i++)
			if (s.menu_language == I18n::kMenuLanguages[i]) cur = i;
		if (ImGui::BeginCombo("##mlang", I18n::kMenuLanguageNames[cur])) {
			for (int i = 0; i < I18n::kMenuLanguageCount; i++)
				if (ImGui::Selectable(I18n::kMenuLanguageNames[i], i == cur)) {
					s.menu_language = I18n::kMenuLanguages[i];
					I18n::SetLanguage(s.menu_language);
					st.dirty = true;
				}
			ImGui::EndCombo();
		}
		Describe("Idioma del menú", "El idioma de este lanzador.");
	}
	if (TextRow("Nombre de usuario", s.user_name, "El nombre del usuario local de la consola (1 a 16 caracteres).")) st.dirty = true;
	if (CheckRow("Cerrar el lanzador al salir del juego", s.close_launcher, "Sí: al cerrar el juego no vuelve este menú.")) st.dirty = true;
}

void DrawGraphics(State& st) {
	Settings& s = *st.s;
	Heading("Pantalla");
	if (CheckRow("Pantalla completa", s.fullscreen, "Pantalla completa sin bordes (la del escritorio). F11 o Alt+Intro la cambia en el juego.")) st.dirty = true;
	if (ChoiceRow("Resolución", s.resolution, kResolutions, kResolutionCount, "El tamaño de la ventana del juego. En pantalla completa sin bordes se usa el del escritorio.")) st.dirty = true;
	if (ChoiceRow("Sincronización", s.present, kPresentModes, kPresentModeCount,
	              "Triple búfer: sin tirones ni tearing, latencia baja (recomendado). VSync: limita a la frecuencia del monitor. Sin sincronía: la menor latencia, con tearing."))
		st.dirty = true;
	if (IntRow("Frecuencia de vblank", s.vblank, 30, 240, "%d Hz", "La frecuencia de vblank virtual que ve el juego (60 en una PS5). Cambiarla puede acelerar o frenar el juego.")) st.dirty = true;
	if (CheckRow("Ocultar el cursor", s.hide_cursor, "Oculta el cursor tras 2 segundos sin moverlo.")) st.dirty = true;
	Heading("Calidad");
	if (CheckRow("Teselado", s.tessellation, "Dibuja los parches de teselado (desactivado: se omiten, más rápido).")) st.dirty = true;
	if (IntRow("Redibujar reflejos cada", s.reflections_every, 0, 8, s.reflections_every == 0 ? "(por defecto)" : "%d fotogramas",
	           "Cada cuántos fotogramas se redibujan los cubos de reflejos. Más: más rendimiento, reflejos menos actualizados. 0: lo del emulador."))
		st.dirty = true;
	if (CheckRow("Texturas bindless", s.bindless, "Lee las texturas por índice desde un montón global (bindless). Experimental.")) st.dirty = true;
}

void DrawComingSoon(State& st, bool framegen) {
	Settings& s = *st.s;
	Heading(framegen ? "Generación de fotogramas" : "Reescalado");
	ImGui::TextColored(kBlue, "%s", T("Próximamente"));
	Note("Esta versión del emulador (KytyPS5-GT7) todavía no trae DLSS, FSR ni XeSS. Las opciones están aquí para cuando lleguen: ahora no tienen efecto.");
	ImGui::BeginDisabled();
	if (framegen) {
		(void)ChoiceRow("Generación de fotogramas", s.framegen, kFrameGens, kFrameGenCount, "Próximamente.");
	} else {
		(void)ChoiceRow("Reescalado", s.upscaler, kUpscalers, kUpscalerCount, "Próximamente.");
		std::string q = "calidad";
		static const Choice kQ[] = {{"calidad", "Calidad"}, {"equilibrado", "Equilibrado"}, {"rendimiento", "Rendimiento"}};
		(void)ChoiceRow("Modo", q, kQ, 3, "Próximamente.");
		float sharp = 0.5f;
		(void)FloatRow("Nitidez", sharp, 0.0f, 1.0f, "%.2f", "Próximamente.");
	}
	ImGui::EndDisabled();
}

void DrawPerf(State& st) {
	Settings& s = *st.s;
	Heading("Procesador");
	if (ChoiceRow("Parches de CPU AMD", s.amd_cpu, kAmdCpu, kAmdCpuCount, "Parches de instrucciones para CPUs AMD (--amd-cpu). Automático: activados en una CPU AMD."))
		st.dirty = true;
	if (ChoiceRow("Prioridad del proceso", s.priority, kPriorities, kPriorityCount, "Alta: Windows da preferencia al juego frente a otros programas.")) st.dirty = true;
	if (CheckRow("Solo los núcleos con más caché (X3D)", s.x3d_ccd,
	             "En un Ryzen X3D de dos CCD (7950X3D, 9950X3D): el juego solo en el CCD con 3D V-Cache. En otras CPUs no hace nada."))
		st.dirty = true;
	Heading("Gráfica");
	Label("Tarjeta gráfica");
	{
		std::string shown = s.gpu < 0 ? std::string(T("Automática")) : "Vulkan #" + std::to_string(s.gpu);
		if (ImGui::BeginCombo("##gpu", shown.c_str())) {
			if (ImGui::Selectable(T("Automática"), s.gpu < 0)) {
				s.gpu    = -1;
				st.dirty = true;
			}
			for (int i = 0; i < 4; i++)
				if (ImGui::Selectable(("Vulkan #" + std::to_string(i)).c_str(), s.gpu == i)) {
					s.gpu    = i;
					st.dirty = true;
				}
			ImGui::EndCombo();
		}
		Describe("Tarjeta gráfica", "El índice del dispositivo Vulkan (--gpu). Automática: la dedicada. Útil en portátiles con dos gráficas.");
	}
	Heading("Shaders");
	if (CheckRow("Precompilar shaders", s.precompile_auto,
	             "Antes de jugar, cuando hace falta (primera vez, otra gráfica, otro driver u otra versión del emulador): crea todos los "
	             "pipelines del juego con todos los hilos de la CPU para que no se compilen en plena carrera. Se puede omitir."))
		st.dirty = true;
	if (ImGui::Button(T("Precompilar de nuevo"))) {
		ForgetPrecompile(*st.rt);
		st.message = T("Los shaders se precompilarán al pulsar JUGAR.");
	}
	Describe("Precompilar de nuevo", "Repite la precompilación al próximo JUGAR (por ejemplo tras varias partidas nuevas o si notas tirones).");
	if (CheckRow("Contadores en el log", s.log_stats,
	             "Escribe en logs\\ una línea de rendimiento con los fps cada 5 s (KYTY_PERF=5) y los contadores de sincronización (KYTY_SYNC_STATS=1)."))
		st.dirty = true;
	if (CheckRow("NVIDIA Smooth Motion", s.cuda_early,
	             "Inicializa CUDA antes que el juego (KYTY_CUDA_EARLY_INIT) para que Smooth Motion del driver de NVIDIA (RTX 40/50) pueda activarse."))
		st.dirty = true;
	Heading("Compatibilidad");
	if (CheckRow("Protección de la red zone", s.redzone, "Protege la red zone SysV del juego frente a las excepciones de Windows (recomendado).")) st.dirty = true;
	if (CheckRow("PlayGo de respaldo", s.playgo_hack, "Usa el sustituto de PlayGo (si el juego se queda esperando a instalar contenido).")) st.dirty = true;
}

void DrawPad(State& st) {
	Settings& s = *st.s;
	Heading("Mandos conectados");
	int  count = 0;
	auto* ids  = SDL_GetGamepads(&count);
	if (count == 0) Note("Ninguno. Conecta un DualSense (USB o Bluetooth) o un 8BitDo (modo XInput / Switch).");
	for (int i = 0; i < count; i++) {
		const char* name = SDL_GetGamepadNameForID(ids[i]);
		const auto  type = SDL_GetGamepadTypeForID(ids[i]);
		const char* kind = type == SDL_GAMEPAD_TYPE_PS5 ? "DualSense" : type == SDL_GAMEPAD_TYPE_PS4 ? "DualShock 4"
		                 : type == SDL_GAMEPAD_TYPE_XBOX360 || type == SDL_GAMEPAD_TYPE_XBOXONE ? "XInput (8BitDo / Xbox)"
		                 : type == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO ? "Switch Pro (8BitDo)" : "Genérico";
		ImGui::BulletText("%s  [%s]", name != nullptr ? name : "?", kind);
	}
	SDL_free(ids);
	Note("DualSense: gatillos adaptativos, vibración, altavoz y barra de luz. 8BitDo / Xbox: botones y vibración (los gatillos y el panel táctil se emulan).");
	Heading("DualSense");
	Label("Color de la barra de luz");
	{
		float col[3] = {0.0f, 0.3f, 1.0f};
		unsigned r = 0, g = 0, b = 0;
		const bool custom = s.pad_color.size() == 7 && std::sscanf(s.pad_color.c_str() + 1, "%02x%02x%02x", &r, &g, &b) == 3;
		if (custom) {
			col[0] = r / 255.0f;
			col[1] = g / 255.0f;
			col[2] = b / 255.0f;
		}
		bool use = custom;
		if (ImGui::Checkbox("##usecol", &use)) {
			s.pad_color = use ? "#0050FF" : "";
			st.dirty    = true;
		}
		Describe("Color de la barra de luz", "Sin marcar: el color que pone el juego.");
		if (use) {
			ImGui::SameLine();
			if (ImGui::ColorEdit3("##col", col, ImGuiColorEditFlags_NoInputs)) {
				char buf[8];
				std::snprintf(buf, sizeof(buf), "#%02X%02X%02X", static_cast<int>(col[0] * 255.0f + 0.5f), static_cast<int>(col[1] * 255.0f + 0.5f),
				              static_cast<int>(col[2] * 255.0f + 0.5f));
				s.pad_color = buf;
				st.dirty    = true;
			}
		}
	}
	if (IntRow("Volumen del altavoz", s.pad_volume, 0, 100, "%d %%", "El altavoz del DualSense.")) st.dirty = true;
	if (IntRow("Vibración", s.pad_vibration, 0, 100, "%d %%", "La intensidad de la vibración del mando.")) st.dirty = true;
}

void DrawWheel(State& st) {
	Settings& s = *st.s;
	Heading("Volante");
	if (CheckRow("Usar volante", s.wheel,
	             "El juego ve un volante (GT DD Pro) movido por tu volante o pedales (--wheel), con force feedback. El mapeo está en wheel.json."))
		st.dirty = true;
	ImGui::BeginDisabled(!s.wheel);
	if (IntRow("Fuerza (force feedback)", s.wheel_ffb_gain, 0, 100, "%d %%", "La fuerza del force feedback, en % de lo que pide el juego.")) st.dirty = true;
	if (IntRow("Grados de giro del volante", s.wheel_rotation, 180, 2520, "%d°", "Los grados configurados en el software de tu volante (900 por defecto).")) st.dirty = true;
	if (CheckRow("Tope suave", s.wheel_soft_lock, "Cuando el coche tiene menos giro que tu volante, se pone un tope suave en ese punto.")) st.dirty = true;
	if (CheckRow("Invertir force feedback", s.wheel_reverse_ffb, "Las fuerzas al revés (si el volante tira hacia fuera en las curvas).")) st.dirty = true;
	if (CheckRow("Amortiguación por software", s.wheel_soft_cond,
	             "Amortiguación y fricción calculadas aquí y enviadas como fuerza constante (para drivers que las manejan mal, p. ej. MOZA R3)."))
		st.dirty = true;
	if (CheckRow("Leer el volante en segundo plano", s.wheel_background, "Sigue leyendo el volante sin el foco en la ventana del juego (también los mandos).")) st.dirty = true;
	ImGui::EndDisabled();
	Heading("Dispositivos");
	int  count = 0;
	auto* ids  = SDL_GetJoysticks(&count);
	if (count == 0) Note("No hay volantes ni mandos conectados.");
	for (int i = 0; i < count; i++) {
		const char* name = SDL_GetJoystickNameForID(ids[i]);
		const auto  type = SDL_GetJoystickTypeForID(ids[i]);
		ImGui::BulletText("%s%s", name != nullptr ? name : "?", type == SDL_JOYSTICK_TYPE_WHEEL ? "  [volante]" : "");
	}
	SDL_free(ids);
	std::error_code ec;
	const fs::path  qt = st.rt->exe_dir / L"launcher.exe";
	ImGui::BeginDisabled(!fs::exists(qt, ec));
	if (ImGui::Button(T("Mapear volante..."))) OpenPath(qt);
	ImGui::EndDisabled();
	Describe("Mapear volante...", "Abre el lanzador de KytyPS5 (launcher.exe) que trae su ventana de mapeo del volante (ejes, botones, cambio en H, prueba de fuerza). Guarda en wheel.json.");
	Note("El mapeo de ejes y botones se hace una vez con \"Mapear volante...\"; aquí se ajusta la fuerza y el giro.");
}

void DrawKbm(State& st) {
	Settings& s = *st.s;
	Heading("Teclado y ratón");
	if (CheckRow("Activar teclado y ratón", s.kbm, "Pasa al emulador el mapa de teclas y la sensibilidad. Desactivado: solo las teclas por defecto.")) st.dirty = true;
	ImGui::BeginDisabled(!s.kbm);
	if (FloatRow("Sensibilidad del ratón", s.mouse_sens, 0.1f, 5.0f, "%.2f", "El ratón como stick derecho (F7 lo captura o lo suelta en el juego).")) st.dirty = true;
	Label("Mapa de teclas");
	if (ImGui::InputTextMultiline("##km", &s.keymap, {-FLT_MIN, 110.0f})) st.dirty = true;
	Describe("Mapa de teclas",
	         "Control=Tecla separados por ';'. Controles: Cross, Circle, Square, Triangle, L1, R1, L2, R2, L3, R3, Options, TouchPad, Up, Down, Left, Right, "
	         "LeftStickUp/Down/Left/Right, RightStickUp/Down/Left/Right. Teclas con su nombre SDL (W, Space, Left Shift) o Mouse:Left / Mouse:Right / Mouse:Middle.");
	if (ImGui::Button(T("Mapa para conducir"))) {
		s.keymap = "LeftStickLeft=A;LeftStickRight=D;R2=W;L2=S;Cross=Return;Circle=Backspace;Square=Space;Triangle=Tab;L1=Q;R1=E;Options=P;"
		           "Up=Up;Down=Down;Left=Left;Right=Right;R3=C;L3=V";
		st.dirty = true;
	}
	Describe("Mapa para conducir", "A / D: volante. W / S: acelerador y freno (R2 / L2). Intro: Cruz. Espacio: Cuadrado. Q / E: marchas (L1 / R1).");
	ImGui::SameLine();
	if (ImGui::Button(T("Teclas por defecto"))) {
		s.keymap.clear();
		st.dirty = true;
	}
	ImGui::EndDisabled();
	Heading("Por defecto");
	Note("Flechas: cruceta. J: Cruz, L: Círculo, I: Triángulo, K: Cuadrado. Q / E: L1 / R1. Z / C: L2 / R2. Mayús / Ctrl: L3 / R3. Intro: Options. "
	     "F7: ratón como stick derecho. F11: pantalla completa. Esc y F1 están reservadas.");
}

void DrawMetrics(State& st) {
	Settings& s = *st.s;
	Heading("Métricas");
	if (CheckRow("Mostrar métricas al jugar", s.metrics, "Una ventana pequeña siempre encima del juego con FPS, RAM y VRAM. F10 la muestra u oculta.")) st.dirty = true;
	ImGui::BeginDisabled(!s.metrics);
	if (CheckRow("FPS", s.metrics_fps, "Los fotogramas por segundo del juego (los que mide el emulador).")) st.dirty = true;
	if (CheckRow("RAM", s.metrics_ram, "La memoria del juego y la del sistema.")) st.dirty = true;
	if (CheckRow("VRAM", s.metrics_vram, "La memoria de vídeo que usa el juego y el total de la gráfica.")) st.dirty = true;
	if (CheckRow("Gráfica de FPS", s.metrics_graph, "Los FPS de los últimos 30 segundos.")) st.dirty = true;
	if (ChoiceRow("Posición", s.metrics_pos, kMetricsPos, kMetricsPosCount, "La esquina de la pantalla.")) st.dirty = true;
	ImGui::EndDisabled();
}

void DrawPc(State& st) {
	const PcInfo& pc = st.rt->pc;
	Heading("Tu equipo");
	ImGui::Text("GPU: %s (%s)", pc.gpu.c_str(), Gb(pc.vram).c_str());
	ImGui::Text("CPU: %s (%d núcleos, %d hilos)", pc.cpu.c_str(), pc.cpu_cores, pc.cpu_threads);
	ImGui::Text("RAM: %d GB", pc.ram_gb);
	ImGui::Text("Pantalla: %d x %d, %d Hz", pc.screen_w, pc.screen_h, pc.refresh_hz);
	if (pc.ram_gb > 0 && pc.ram_gb < 16) ImGui::TextColored(kAccent, "%s", T("Menos de 16 GB de RAM: el juego puede ir mal."));
	if (pc.vram > 0 && pc.vram < (6ull << 30)) ImGui::TextColored(kAccent, "%s", T("Menos de 6 GB de VRAM: baja la resolución."));
	Heading("Acciones");
	if (ImGui::Button(T("Añadir a Steam"))) {
		std::string report;
		if (!Steam::RunAddToSteamScript(st.rt->exe_path, &report)) (void)Steam::AddToSteam(st.rt->exe_path, &report);
		st.message = report;
	}
	Describe("Añadir a Steam", "Añade \"Gran Turismo 7\" como juego no de Steam (con su icono y, si está fondo-gt7.jpg, sus imágenes). Reinicia Steam después.");
	ImGui::SameLine();
	if (ImGui::Button(T("Abrir carpeta de logs"))) {
		std::error_code ec;
		fs::create_directories(st.rt->exe_dir / L"logs", ec);
		OpenPath(st.rt->exe_dir / L"logs");
	}
	ImGui::SameLine();
	if (ImGui::Button(T("Abrir settings.ini"))) OpenPath(fs::path(Wide(st.rt->ini_path)));
}

void DrawDev(State& st) {
	Settings& s = *st.s;
	Heading("Desarrollador");
	if (CheckRow("Validación de Vulkan", s.dev_vulkan_validation, "Las capas de validación de Vulkan (lento; necesitan el SDK).")) st.dirty = true;
	if (CheckRow("Log de shaders a archivo", s.dev_shader_log, "Los shaders a logs\\shaders.")) st.dirty = true;
	if (CheckRow("printf del juego a archivo", s.dev_printf_file, "La salida printf del juego a logs\\printf.txt.")) st.dirty = true;
	if (CheckRow("RenderDoc", s.dev_renderdoc, "Captura con RenderDoc (--rd).")) st.dirty = true;
	if (CheckRow("Tracy", s.dev_profiler, "El profiler Tracy (--profile).")) st.dirty = true;
	if (CheckRow("Consola del emulador", s.dev_console, "La consola de kyty_emulator a la vista (si no, su salida va a logs\\gt7-<fecha>.log).")) st.dirty = true;
	if (TextRow("Argumentos extra", s.dev_args, "Más opciones de kyty_emulator, separadas por espacios (kyty_emulator --help).")) st.dirty = true;
	if (TextRow("Entorno extra", s.dev_env, "Variables de entorno: CLAVE=valor;CLAVE=valor (p. ej. KYTY_...).")) st.dirty = true;
	Heading("Comando");
	const std::string cmd = BuildCommandLine(*st.rt, s);
	ImGui::PushTextWrapPos(0.0f);
	ImGui::TextColored(kDim, "%s", cmd.c_str());
	ImGui::PopTextWrapPos();
	if (ImGui::Button(T("Copiar comando"))) ImGui::SetClipboardText(cmd.c_str());
	ImGui::SameLine();
	if (ImGui::Button(T("Restaurar valores por defecto"))) {
		Settings d;
		d.game          = s.game;
		d.menu_language = s.menu_language;
		d.configured    = true;
		s               = d;
		st.dirty        = true;
	}
}

} // namespace

LauncherResult RunLauncher(const Runtime& rt, Settings* settings, const std::string& status) {
	Gui gui;
	if (!gui.Open("Gran Turismo 7", 1360, 820, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY)) {
		gui.Close();
		return LauncherResult::Quit;
	}
	gui.LoadBackground(rt.exe_dir / L"fondo-gt7.jpg");
	State st;
	st.rt      = &rt;
	st.s       = settings;
	st.message = status;
	LauncherResult result = LauncherResult::Quit;
	bool           done   = false;
	while (!done) {
		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			ImGui::SetCurrentContext(gui.context);
			ImGui_ImplSDL3_ProcessEvent(&e);
			if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) done = true;
		}
		if (SDL_GetWindowFlags(gui.window) & SDL_WINDOW_MINIMIZED) {
			SDL_Delay(30);
			continue;
		}
		gui.Begin();
		// LB / RB (Q / E with Ctrl): the previous / next section.
		if (ImGui::IsKeyPressed(ImGuiKey_GamepadL1)) st.section = (st.section + kSectionCount - 1) % kSectionCount;
		if (ImGui::IsKeyPressed(ImGuiKey_GamepadR1)) st.section = (st.section + 1) % kSectionCount;

		const ImGuiViewport* vp = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(vp->WorkPos);
		ImGui::SetNextWindowSize(vp->WorkSize);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0.02f, 0.03f, 0.07f, gui.background != nullptr ? 0.55f : 1.0f});
		ImGui::Begin("##launcher", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
		ImGui::PopStyleColor();

		// The left column: the title, the sections, JUGAR.
		const float left_w = 300.0f;
		ImGui::BeginChild("##left", {left_w, 0}, ImGuiChildFlags_None);
		ImGui::PushFont(gui.title, 34.0f);
		ImGui::TextColored(kText, "GRAN TURISMO");
		ImGui::SameLine(0, 8);
		ImGui::TextColored(kAccent, "7");
		ImGui::PopFont();
		ImGui::TextColored(kDim, "PC  ·  PPSA01317  ·  KytyPS5-GT7");
		ImGui::Spacing();
		for (int i = 0; i < kSectionCount; i++) {
			const bool soon = i == kUpscale || i == kFrameGen;
			std::string name = T(kSectionNames[i]);
			if (soon) name += std::string("  (") + T("Próximamente") + ")";
			if (ImGui::Selectable((name + "##sec" + std::to_string(i)).c_str(), st.section == i, 0, {0, 30.0f})) st.section = i;
		}
		ImGui::Dummy({0, 12.0f});
		const bool ok = IsGame(settings->game);
		ImGui::BeginDisabled(!ok);
		ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4 {1.0f, 0.25f, 0.28f, 1.0f});
		ImGui::PushFont(gui.title, 36.0f);
		if (ImGui::Button(T("JUGAR"), {-FLT_MIN, 64.0f})) {
			result = LauncherResult::Play;
			done   = true;
		}
		ImGui::PopFont();
		ImGui::PopStyleColor(2);
		ImGui::EndDisabled();
		if (ImGui::Button(T("Salir"), {-FLT_MIN, 0})) done = true;
		if (!st.message.empty()) {
			ImGui::PushTextWrapPos(0.0f);
			ImGui::TextColored(kBlue, "%s", st.message.c_str());
			ImGui::PopTextWrapPos();
		}
		ImGui::EndChild();
		ImGui::SameLine();

		// The options and the description.
		const float desc_w = 330.0f;
		ImGui::BeginChild("##page", {ImGui::GetContentRegionAvail().x - desc_w - 10.0f, 0}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
		ImGui::PushFont(gui.title, 28.0f);
		ImGui::TextColored(kText, "%s", T(kSectionNames[st.section]));
		ImGui::PopFont();
		switch (st.section) {
			case kHome: DrawHome(st); break;
			case kGraphics: DrawGraphics(st); break;
			case kUpscale: DrawComingSoon(st, false); break;
			case kFrameGen: DrawComingSoon(st, true); break;
			case kPerf: DrawPerf(st); break;
			case kPad: DrawPad(st); break;
			case kWheel: DrawWheel(st); break;
			case kKbm: DrawKbm(st); break;
			case kMetrics: DrawMetrics(st); break;
			case kPc: DrawPc(st); break;
			case kDev: DrawDev(st); break;
			default: break;
		}
		ImGui::EndChild();
		ImGui::SameLine();
		ImGui::BeginChild("##desc", {0, 0}, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
		ImGui::PushFont(gui.title, 24.0f);
		ImGui::TextColored(kAccent, "%s", g_desc_title.empty() ? T(kSectionNames[st.section]) : g_desc_title.c_str());
		ImGui::PopFont();
		ImGui::Separator();
		ImGui::PushTextWrapPos(0.0f);
		ImGui::TextUnformatted(g_desc_text.empty()
		                           ? T("Ratón, teclado o mando (DualSense, 8BitDo, Xbox).\nCruceta / flechas: moverse. A / Cruz / Intro: elegir.\nLB / RB: cambiar de sección.\n\nTodo se guarda en settings.ini y se aplica al pulsar JUGAR.")
		                           : g_desc_text.c_str());
		ImGui::PopTextWrapPos();
		ImGui::EndChild();
		ImGui::End();
		gui.End({0.02f, 0.03f, 0.07f, 1.0f});

		if (st.dirty) {
			(void)SaveSettings(rt.ini_path, *settings);
			st.dirty = false;
		}
	}
	gui.Close();
	return result;
}

// ---- While playing: the metrics window. ----

namespace {

struct GameWindowSearch {
	DWORD        pid;
	std::wstring title;
};

BOOL CALLBACK FindGameWindow(HWND hwnd, LPARAM lp) {
	auto* f   = reinterpret_cast<GameWindowSearch*>(lp);
	DWORD pid = 0;
	GetWindowThreadProcessId(hwnd, &pid);
	if (pid != f->pid || !IsWindowVisible(hwnd)) return TRUE;
	wchar_t text[512] = {};
	GetWindowTextW(hwnd, text, static_cast<int>(std::size(text)));
	if (std::wcsstr(text, L"fps") != nullptr) {
		f->title = text;
		return FALSE;
	}
	return TRUE;
}

bool g_toggle = false;

bool SDLCALL HotkeyHook(void*, MSG* msg) {
	if (msg->message == WM_HOTKEY && msg->wParam == 0x67) g_toggle = true;
	return true;
}

} // namespace

void WatchGame(const Runtime& rt, const Settings& s, HANDLE process, DWORD pid) {
	(void)rt;
	if (!s.metrics) {
		WaitForSingleObject(process, INFINITE);
		return;
	}
	const bool hotkey = RegisterHotKey(nullptr, 0x67, MOD_NOREPEAT, VK_F10) != 0;
	SDL_SetWindowsMessageHook(HotkeyHook, nullptr);

	Gui        gui;
	const int  w = 280, h = (s.metrics_graph ? 220 : 150);
	if (!gui.Open("Gran Turismo 7 - Métricas", w, h,
	              SDL_WINDOW_BORDERLESS | SDL_WINDOW_ALWAYS_ON_TOP | SDL_WINDOW_UTILITY | SDL_WINDOW_NOT_FOCUSABLE | SDL_WINDOW_TRANSPARENT)) {
		gui.Close();
		WaitForSingleObject(process, INFINITE);
		if (hotkey) UnregisterHotKey(nullptr, 0x67);
		SDL_SetWindowsMessageHook(nullptr, nullptr);
		return;
	}
	ImGui::GetIO().ConfigFlags &= ~(ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad);
	SDL_SetWindowOpacity(gui.window, 0.88f);
	{
		const bool right = s.metrics_pos.find("derecha") != std::string::npos, bottom = s.metrics_pos.find("abajo") != std::string::npos;
		const int  sw = rt.pc.screen_w > 0 ? rt.pc.screen_w : 1920, sh = rt.pc.screen_h > 0 ? rt.pc.screen_h : 1080;
		SDL_SetWindowPosition(gui.window, right ? sw - w - 16 : 16, bottom ? sh - h - 56 : 16);
	}

	// VRAM: the "GPU Process Memory" counters of the game's process.
	PDH_HQUERY   query   = nullptr;
	PDH_HCOUNTER counter = nullptr;
	if (s.metrics_vram && PdhOpenQueryW(nullptr, 0, &query) == ERROR_SUCCESS) {
		if (PdhAddEnglishCounterW(query, L"\\GPU Process Memory(*)\\Dedicated Usage", 0, &counter) != ERROR_SUCCESS) counter = nullptr;
		PdhCollectQueryData(query);
	}
	const std::wstring pid_tag = L"pid_" + std::to_wstring(pid) + L"_";
	const std::wregex  fps_re(L"fps: ([0-9]+)");

	std::deque<float> history;
	float             fps = -1.0f;
	uint64_t          ram = 0, vram = 0;
	Uint64            next_sample = 0;
	bool              shown = true;
	while (WaitForSingleObject(process, 0) == WAIT_TIMEOUT) {
		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			ImGui::SetCurrentContext(gui.context);
			ImGui_ImplSDL3_ProcessEvent(&e);
		}
		if (g_toggle) {
			g_toggle = false;
			shown    = !shown;
			if (shown) SDL_ShowWindow(gui.window);
			else SDL_HideWindow(gui.window);
		}
		const Uint64 now = SDL_GetTicks();
		if (now >= next_sample) {
			next_sample = now + 500;
			GameWindowSearch f {pid, {}};
			EnumWindows(FindGameWindow, reinterpret_cast<LPARAM>(&f));
			std::wsmatch m;
			if (!f.title.empty() && std::regex_search(f.title, m, fps_re)) fps = std::stof(m[1].str());
			if (fps >= 0.0f) {
				history.push_back(fps);
				while (history.size() > 60) history.pop_front();
			}
			PROCESS_MEMORY_COUNTERS_EX pmc {};
			if (GetProcessMemoryInfo(process, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) ram = pmc.PrivateUsage;
			if (counter != nullptr && PdhCollectQueryData(query) == ERROR_SUCCESS) {
				DWORD size = 0, items = 0;
				PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &size, &items, nullptr);
				std::vector<uint8_t> buf(size);
				auto* arr = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
				if (size > 0 && PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &size, &items, arr) == ERROR_SUCCESS) {
					uint64_t sum = 0;
					for (DWORD i = 0; i < items; i++)
						if (std::wcsstr(arr[i].szName, pid_tag.c_str()) != nullptr) sum += static_cast<uint64_t>(arr[i].FmtValue.largeValue);
					vram = sum;
				}
			}
		}
		if (!shown) {
			SDL_Delay(50);
			continue;
		}
		gui.Begin();
		ImGui::SetNextWindowPos({0, 0});
		ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
		ImGui::Begin("##metrics", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoSavedSettings);
		if (s.metrics_fps) {
			ImGui::PushFont(gui.title, 34.0f);
			const ImVec4 c = fps < 0.0f ? kDim : fps >= 55.0f ? ImVec4 {0.4f, 0.95f, 0.5f, 1.0f} : fps >= 30.0f ? ImVec4 {1.0f, 0.85f, 0.3f, 1.0f} : kAccent;
			if (fps < 0.0f) ImGui::TextColored(c, "-- FPS");
			else ImGui::TextColored(c, "%.0f FPS", fps);
			ImGui::PopFont();
		}
		if (s.metrics_ram) {
			MEMORYSTATUSEX m {};
			m.dwLength = sizeof(m);
			GlobalMemoryStatusEx(&m);
			ImGui::Text("RAM   %s  (%s / %s)", Gb(ram).c_str(), Gb(m.ullTotalPhys - m.ullAvailPhys).c_str(), Gb(m.ullTotalPhys).c_str());
		}
		if (s.metrics_vram) ImGui::Text("VRAM  %s / %s", counter != nullptr ? Gb(vram).c_str() : "?", Gb(rt.pc.vram).c_str());
		if (s.metrics_graph && !history.empty()) {
			std::vector<float> v(history.begin(), history.end());
			ImGui::PlotLines("##g", v.data(), static_cast<int>(v.size()), 0, nullptr, 0.0f, 120.0f, {-FLT_MIN, 60.0f});
		}
		ImGui::TextColored(kDim, "F10: %s", T("mostrar / ocultar"));
		ImGui::End();
		gui.End({0.02f, 0.03f, 0.07f, 0.85f});
	}
	if (query != nullptr) PdhCloseQuery(query);
	gui.Close();
	if (hotkey) UnregisterHotKey(nullptr, 0x67);
	SDL_SetWindowsMessageHook(nullptr, nullptr);
}

// ---- Shader precompile. ----

fs::path ShippedSeeds(const Runtime& rt);

namespace {

const wchar_t kTitle[] = L"PPSA01317";

fs::path CacheDir(const Runtime& rt) {
	return rt.exe_dir / L"_ShaderCache" / kTitle;
}

fs::path StampPath(const Runtime& rt) {
	return CacheDir(rt) / L"precompile-stamp.txt";
}

std::string ReadText(const fs::path& p) {
	std::ifstream in(p, std::ios::binary);
	return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void WriteText(const fs::path& p, const std::string& text) {
	std::ofstream out(p, std::ios::binary | std::ios::trunc);
	out << text;
}

// The graphics card's name and driver version (the one rt.pc.gpu names).
std::string DriverKey(const Runtime& rt) {
	std::string    key = rt.pc.gpu;
	IDXGIFactory1* factory = nullptr;
	if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) && factory != nullptr) {
		IDXGIAdapter1* adapter = nullptr;
		for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
			DXGI_ADAPTER_DESC1 d {};
			LARGE_INTEGER      version {};
			if (SUCCEEDED(adapter->GetDesc1(&d)) && Utf8(d.Description).rfind(rt.pc.gpu, 0) == 0 &&
			    SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &version))) {
				char text[64];
				std::snprintf(text, sizeof(text), " %u.%u.%u.%u", HIWORD(version.HighPart), LOWORD(version.HighPart),
				              HIWORD(version.LowPart), LOWORD(version.LowPart));
				key += text;
				adapter->Release();
				break;
			}
			adapter->Release();
		}
		factory->Release();
	}
	return key;
}

// What a precompile is valid for: the GPU, its driver, the emulator (its size and date: another build may translate
// shaders another way) and the options that change the shaders.
std::string StampKey(const Runtime& rt, const Settings& s) {
	std::error_code ec;
	const fs::path  emu  = rt.exe_dir / L"kyty_emulator.exe";
	const auto      size = fs::file_size(emu, ec);
	const auto      time = fs::last_write_time(emu, ec).time_since_epoch().count();
	const fs::path  shipped = ShippedSeeds(rt);
	const std::string seeds = shipped.empty() ? std::string("-")
	                                          : Utf8(shipped.filename().wstring()) + "/" + std::to_string(fs::file_size(shipped, ec));
	// The player's own guest.seeds too (each run appends the programs and pipelines new to it): before, only the
	// release's file counted, so a run's new seeds were never precompiled (the next run made them in game again).
	// Its size, not its time (a run that adds nothing leaves it as it was). KYTY_PRECOMPILE_NEW_SEEDS=0: as before.
	std::string own = "-";
	if (const char* v = std::getenv("KYTY_PRECOMPILE_NEW_SEEDS"); v == nullptr || v[0] != '0') {
		const auto own_size = fs::file_size(CacheDir(rt) / L"guest.seeds", ec);
		own                 = ec ? std::string("-") : std::to_string(own_size);
		ec.clear();
	}
	return "gpu=" + DriverKey(rt) + "\nemulador=" + std::to_string(size) + "/" + std::to_string(time) + "\nsemillas=" + seeds +
	       "\nsemillas_propias=" + own +
	       "\ngpu_indice=" +
	       std::to_string(s.gpu) + "\nbindless=" + (s.bindless ? "1" : "0") + "\nteselado=" + (s.tessellation ? "1" : "0") + "\n";
}

} // namespace

// The release's seed file: seeds-PPSA01317_<version>.seeds beside the exe or in seeds\ (the newest name wins).
fs::path ShippedSeeds(const Runtime& rt) {
	std::error_code ec;
	fs::path        best;
	for (const fs::path& dir: {rt.exe_dir, rt.exe_dir / L"seeds"}) {
		for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
			const std::wstring name = it->path().filename().wstring();
			if (it->is_regular_file() && name.rfind(std::wstring(L"seeds-") + kTitle, 0) == 0 &&
			    it->path().extension() == L".seeds" && (best.empty() || name > best.filename().wstring())) {
				best = it->path();
			}
		}
		ec.clear();
	}
	return best;
}

void MergeShippedSeeds(const Runtime& rt) {
	// The emulator reads the release's file with the player's own (KYTY_PRECOMPILE_SEEDS, set by RunPrecompile):
	// nothing to copy. Only the old SPIR-V seeds of an earlier build are left behind.
	(void)rt;
}

bool PrecompileDue(const Runtime& rt, const Settings& s, std::string* why) {
	std::error_code ec;
	if (!fs::exists(CacheDir(rt) / L"guest.seeds", ec) && ShippedSeeds(rt).empty()) {
		if (why != nullptr) *why = "sin semillas todavía";
		return false;
	}
	const std::string stamp = ReadText(StampPath(rt));
	if (stamp == StampKey(rt, s)) return false;
	if (why != nullptr) *why = stamp.empty() ? "primera vez" : "cambió la gráfica, el driver, el emulador, las semillas (o hay semillas nuevas de la última partida) o una opción";
	return true;
}

void ForgetPrecompile(const Runtime& rt) {
	std::error_code ec;
	fs::remove(StampPath(rt), ec);
}

std::string RunPrecompile(const Runtime& rt, const Settings& s, const std::wstring& command_line) {
	const fs::path  dir      = CacheDir(rt);
	const fs::path  progress = dir / L"precompile.progress";
	const fs::path  stop     = fs::path(progress.wstring() + L".stop");
	std::error_code ec;
	fs::remove(progress, ec);
	fs::remove(stop, ec);
	fs::create_directories(rt.exe_dir / L"logs", ec);

	SetEnvironmentVariableW(L"KYTY_PRECOMPILE", L"1");
	SetEnvironmentVariableW(L"KYTY_PRECOMPILE_PROGRESS", progress.wstring().c_str());
	const fs::path shipped = ShippedSeeds(rt);
	SetEnvironmentVariableW(L"KYTY_PRECOMPILE_SEEDS", shipped.empty() ? nullptr : shipped.wstring().c_str());
	wchar_t stamp[32];
	{
		const std::time_t t = std::time(nullptr);
		std::tm           tm {};
		localtime_s(&tm, &t);
		wcsftime(stamp, std::size(stamp), L"%Y%m%d-%H%M%S", &tm);
	}
	const fs::path      log_path = rt.exe_dir / L"logs" / (std::wstring(L"gt7-precompilar-") + stamp + L".log");
	SECURITY_ATTRIBUTES sa {sizeof(sa), nullptr, TRUE};
	HANDLE log = CreateFileW(log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	STARTUPINFOW si {};
	si.cb = sizeof(si);
	if (log != INVALID_HANDLE_VALUE) {
		si.dwFlags    = STARTF_USESTDHANDLES;
		si.hStdOutput = log;
		si.hStdError  = log;
		si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
	}
	std::wstring        cmd = command_line;
	PROCESS_INFORMATION pi {};
	const BOOL started = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, log != INVALID_HANDLE_VALUE,
	                                    CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS, nullptr, rt.exe_dir.c_str(), &si, &pi);
	SetEnvironmentVariableW(L"KYTY_PRECOMPILE", nullptr);
	SetEnvironmentVariableW(L"KYTY_PRECOMPILE_PROGRESS", nullptr);
	SetEnvironmentVariableW(L"KYTY_PRECOMPILE_SEEDS", nullptr);
	if (!started) {
		if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
		return "No se pudo iniciar la precompilación (error " + std::to_string(GetLastError()) + ").";
	}
	CloseHandle(pi.hThread);

	Gui gui;
	const bool window = gui.Open("Gran Turismo 7 - Shaders", 620, 230, 0);
	if (window) gui.LoadBackground(rt.exe_dir / L"fondo-gt7.jpg");
	unsigned long done = 0, total = 0, compiled = 0, failed = 0, final_flag = 0;
	bool          skipping = false, finished = false;
	Uint64        skip_at = 0, next_read = 0;
	while (WaitForSingleObject(pi.hProcess, 0) == WAIT_TIMEOUT) {
		const Uint64 now = SDL_GetTicks();
		if (now >= next_read) {
			next_read = now + 250;
			const std::string text = ReadText(progress);
			(void)std::sscanf(text.c_str(), "%lu %lu %lu %lu %lu", &done, &total, &compiled, &failed, &final_flag);
		}
		if (skipping && now - skip_at > 120000) {
			TerminateProcess(pi.hProcess, 1); // the save did not end in two minutes
			break;
		}
		if (!window) {
			SDL_Delay(100);
			continue;
		}
		SDL_Event e;
		bool      close = false;
		while (SDL_PollEvent(&e)) {
			ImGui::SetCurrentContext(gui.context);
			ImGui_ImplSDL3_ProcessEvent(&e);
			if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) close = true;
		}
		gui.Begin();
		ImGui::SetNextWindowPos({0, 0});
		ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4 {0.02f, 0.03f, 0.07f, 0.80f});
		ImGui::Begin("##precompile", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
		ImGui::PopStyleColor();
		ImGui::PushFont(gui.title, 26.0f);
		ImGui::TextColored(kText, "%s", T("Preparando los shaders"));
		ImGui::PopFont();
		ImGui::PushTextWrapPos(0.0f);
		ImGui::TextColored(kDim, "%s",
		                   T("Solo la primera vez (o tras cambiar de driver): así el juego no compila nada durante la carrera."));
		ImGui::PopTextWrapPos();
		const float fraction = total > 0 ? static_cast<float>(done) / static_cast<float>(total) : 0.0f;
		char        overlay[64];
		if (total > 0) std::snprintf(overlay, sizeof(overlay), "%lu / %lu", done, total);
		else std::snprintf(overlay, sizeof(overlay), "%s", T("Arrancando..."));
		ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kAccent);
		ImGui::ProgressBar(fraction, {-FLT_MIN, 28.0f}, overlay);
		ImGui::PopStyleColor();
		ImGui::BeginDisabled(skipping);
		if (ImGui::Button(skipping ? T("Guardando lo compilado...") : T("Omitir"), {-FLT_MIN, 0}) || (close && !skipping)) {
			WriteText(stop, "stop");
			skipping = true;
			skip_at  = now;
		}
		ImGui::EndDisabled();
		ImGui::End();
		gui.End({0.02f, 0.03f, 0.07f, 1.0f});
	}
	WaitForSingleObject(pi.hProcess, 5000);
	DWORD code = 1;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hProcess);
	if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
	gui.Close();
	{
		const std::string text = ReadText(progress);
		(void)std::sscanf(text.c_str(), "%lu %lu %lu %lu %lu", &done, &total, &compiled, &failed, &final_flag);
	}
	finished = code == 0 && final_flag == 1;
	fs::remove(stop, ec);
	// Done or skipped: not run again by itself until something changes (Rendimiento: "Precompilar de nuevo"). A
	// precompile that died (a fatal error: gt7-precompilar-20261112-054529 stopped at 31,367 programs and saved
	// nothing) is not stamped: before, its stamp left every later run compiling in game what it never made.
	if (finished || skipping) {
		WriteText(StampPath(rt), StampKey(rt, s));
	}
	if (!finished) {
		return "La precompilación de shaders no terminó (código " + std::to_string(code) + "). Log: logs\\" +
		       Utf8(log_path.filename().wstring());
	}
	return {};
}

} // namespace Port
