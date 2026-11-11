// "Gran Turismo 7" PC port: settings.ini and the emulator's command line (gt7-settings.h).
#include "port/gt7-settings.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <fstream>
#include <map>
#include <sstream>

namespace Port {

const Choice kResolutions[] = {{"escritorio", "Escritorio (la de Windows)"}, {"1280x720", "1280 x 720"},  {"1600x900", "1600 x 900"},
                               {"1920x1080", "1920 x 1080"},               {"2560x1440", "2560 x 1440"}, {"3440x1440", "3440 x 1440 (ultrapanorámica)"},
                               {"3840x2160", "3840 x 2160 (4K)"}};
const int    kResolutionCount = static_cast<int>(std::size(kResolutions));
const Choice kPresentModes[] = {{"mailbox", "Triple búfer (Mailbox)"}, {"fifo", "VSync (Fifo)"}, {"immediate", "Sin sincronía (Immediate)"}};
const int    kPresentModeCount = static_cast<int>(std::size(kPresentModes));
const Choice kUpscalers[] = {{"no", "Nativo"}, {"dlss", "NVIDIA DLSS"}, {"fsr", "AMD FSR"}, {"xess", "Intel XeSS"}};
const int    kUpscalerCount = static_cast<int>(std::size(kUpscalers));
const Choice kFrameGens[] = {{"no", "Sin generación de fotogramas"}, {"dlss-fg", "NVIDIA DLSS FG"}, {"fsr-fg", "AMD FSR FG"}, {"xess-fg", "Intel XeSS FG"}};
const int    kFrameGenCount = static_cast<int>(std::size(kFrameGens));
const Choice kAmdCpu[] = {{"auto", "Automático (según tu CPU)"}, {"si", "Sí"}, {"no", "No"}};
const int    kAmdCpuCount = static_cast<int>(std::size(kAmdCpu));
const Choice kPriorities[] = {{"normal", "Normal"}, {"alta", "Alta"}};
const int    kPriorityCount = static_cast<int>(std::size(kPriorities));
const Choice kMetricsPos[] = {{"arriba-izquierda", "Arriba a la izquierda"}, {"arriba-derecha", "Arriba a la derecha"},
                              {"abajo-izquierda", "Abajo a la izquierda"}, {"abajo-derecha", "Abajo a la derecha"}};
const int    kMetricsPosCount = static_cast<int>(std::size(kMetricsPos));
const Language kConsoleLanguages[] = {{3, "Español (España)"}, {20, "Español (Latinoamérica)"}, {1, "English (US)"}, {18, "English (UK)"},
                                      {2, "Français"},          {4, "Deutsch"},                {5, "Italiano"},     {7, "Português (Portugal)"},
                                      {17, "Português (Brasil)"}, {0, "日本語"},                 {6, "Nederlands"},   {8, "Русский"},
                                      {16, "Polski"},           {19, "Türkçe"}};
const int      kConsoleLanguageCount = static_cast<int>(std::size(kConsoleLanguages));

namespace {

std::string Trim(std::string s) {
	const auto b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos) return {};
	const auto e = s.find_last_not_of(" \t\r\n");
	return s.substr(b, e - b + 1);
}

bool Bool(const std::string& v, bool fallback) {
	std::string l = v;
	std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	if (l == "1" || l == "si" || l == "sí" || l == "true" || l == "yes" || l == "on") return true;
	if (l == "0" || l == "no" || l == "false" || l == "off") return false;
	return fallback;
}

int Int(const std::string& v, int fallback) {
	try {
		return std::stoi(v);
	} catch (...) {
		return fallback;
	}
}

float Float(const std::string& v, float fallback) {
	try {
		return std::stof(v);
	} catch (...) {
		return fallback;
	}
}

// The command line split on spaces, "quoted parts" kept whole.
std::vector<std::string> SplitArgs(const std::string& s) {
	std::vector<std::string> out;
	std::string              cur;
	bool                     quoted = false, any = false;
	for (char c: s) {
		if (c == '"') {
			quoted = !quoted;
			any    = true;
		} else if ((c == ' ' || c == '\t') && !quoted) {
			if (any) out.push_back(cur);
			cur.clear();
			any = false;
		} else {
			cur += c;
			any = true;
		}
	}
	if (any) out.push_back(cur);
	return out;
}

} // namespace

bool LoadSettings(const std::string& path, Settings* s) {
	std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
	if (!in) return false;
	std::map<std::string, std::string> kv;
	std::string                        line;
	while (std::getline(in, line)) {
		if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF) line = line.substr(3); // BOM
		line = Trim(line);
		if (line.empty() || line[0] == '#' || line[0] == ';' || line[0] == '[') continue;
		const auto eq = line.find('=');
		if (eq == std::string::npos) continue;
		kv[Trim(line.substr(0, eq))] = Trim(line.substr(eq + 1));
	}
	auto str = [&](const char* k, std::string& v) {
		if (auto it = kv.find(k); it != kv.end()) v = it->second;
	};
	auto bol = [&](const char* k, bool& v) {
		if (auto it = kv.find(k); it != kv.end()) v = Bool(it->second, v);
	};
	auto num = [&](const char* k, int& v, int lo, int hi) {
		if (auto it = kv.find(k); it != kv.end()) v = std::clamp(Int(it->second, v), lo, hi);
	};
	str("carpeta_juego", s->game);
	str("parche_juego", s->game_patch);
	num("idioma_juego", s->console_language, 0, 29);
	str("idioma_menu", s->menu_language);
	str("nombre_usuario", s->user_name);
	bol("cerrar_lanzador", s->close_launcher);
	bol("pantalla_completa", s->fullscreen);
	str("resolucion", s->resolution);
	str("sincronizacion", s->present);
	num("vblank", s->vblank, 30, 240);
	bol("teselado", s->tessellation);
	num("reflejos_cada", s->reflections_every, 0, 8);
	bol("ocultar_cursor", s->hide_cursor);
	bol("bindless", s->bindless);
	str("reescalado", s->upscaler);
	str("generacion_fotogramas", s->framegen);
	num("gpu", s->gpu, -1, 16);
	str("parches_cpu_amd", s->amd_cpu);
	str("prioridad", s->priority);
	bol("nucleos_x3d", s->x3d_ccd);
	bol("red_zone", s->redzone);
	bol("playgo_hack", s->playgo_hack);
	bol("smooth_motion", s->cuda_early);
	bol("precompilar_shaders", s->precompile_auto);
	bol("contadores_log", s->log_stats);
	str("color_mando", s->pad_color);
	num("volumen_mando", s->pad_volume, 0, 100);
	num("vibracion_mando", s->pad_vibration, 0, 100);
	bol("volante", s->wheel);
	num("volante_ffb", s->wheel_ffb_gain, 0, 100);
	num("volante_grados", s->wheel_rotation, 180, 2520);
	bol("volante_tope_suave", s->wheel_soft_lock);
	bol("volante_ffb_invertido", s->wheel_reverse_ffb);
	bol("volante_condiciones_software", s->wheel_soft_cond);
	bol("volante_segundo_plano", s->wheel_background);
	bol("teclado_raton", s->kbm);
	if (auto it = kv.find("sensibilidad_raton"); it != kv.end()) s->mouse_sens = std::clamp(Float(it->second, s->mouse_sens), 0.1f, 5.0f);
	str("mapa_teclado", s->keymap);
	bol("metricas", s->metrics);
	bol("metricas_fps", s->metrics_fps);
	bol("metricas_ram", s->metrics_ram);
	bol("metricas_vram", s->metrics_vram);
	bol("metricas_grafica", s->metrics_graph);
	str("metricas_posicion", s->metrics_pos);
	bol("dev_validacion_vulkan", s->dev_vulkan_validation);
	bol("dev_log_shaders", s->dev_shader_log);
	bol("dev_printf_archivo", s->dev_printf_file);
	bol("dev_renderdoc", s->dev_renderdoc);
	bol("dev_profiler", s->dev_profiler);
	bol("dev_consola", s->dev_console);
	str("argumentos_extra", s->dev_args);
	str("entorno_extra", s->dev_env);
	bol("configurado", s->configured);
	return true;
}

bool SaveSettings(const std::string& path, const Settings& s) {
	std::ostringstream o;
	auto               b = [](bool v) { return v ? "1" : "0"; };
	o << "# Gran Turismo 7 PC (KytyPS5-GT7): ajustes del lanzador. Se pueden editar a mano con el juego cerrado.\r\n"
	  << "[juego]\r\n"
	  << "carpeta_juego=" << s.game << "\r\n"
	  << "parche_juego=" << s.game_patch << "\r\n"
	  << "idioma_juego=" << s.console_language << "\r\n"
	  << "idioma_menu=" << s.menu_language << "\r\n"
	  << "nombre_usuario=" << s.user_name << "\r\n"
	  << "cerrar_lanzador=" << b(s.close_launcher) << "\r\n"
	  << "[graficos]\r\n"
	  << "pantalla_completa=" << b(s.fullscreen) << "\r\n"
	  << "resolucion=" << s.resolution << "\r\n"
	  << "sincronizacion=" << s.present << "\r\n"
	  << "vblank=" << s.vblank << "\r\n"
	  << "teselado=" << b(s.tessellation) << "\r\n"
	  << "reflejos_cada=" << s.reflections_every << "\r\n"
	  << "ocultar_cursor=" << b(s.hide_cursor) << "\r\n"
	  << "bindless=" << b(s.bindless) << "\r\n"
	  << "# (proximamente: esta version del emulador aun no trae DLSS / FSR / XeSS)\r\n"
	  << "reescalado=" << s.upscaler << "\r\n"
	  << "generacion_fotogramas=" << s.framegen << "\r\n"
	  << "[rendimiento]\r\n"
	  << "gpu=" << s.gpu << "\r\n"
	  << "parches_cpu_amd=" << s.amd_cpu << "\r\n"
	  << "prioridad=" << s.priority << "\r\n"
	  << "nucleos_x3d=" << b(s.x3d_ccd) << "\r\n"
	  << "red_zone=" << b(s.redzone) << "\r\n"
	  << "playgo_hack=" << b(s.playgo_hack) << "\r\n"
	  << "smooth_motion=" << b(s.cuda_early) << "\r\n"
	  << "precompilar_shaders=" << b(s.precompile_auto) << "\r\n"
	  << "contadores_log=" << b(s.log_stats) << "\r\n"
	  << "[mando]\r\n"
	  << "color_mando=" << s.pad_color << "\r\n"
	  << "volumen_mando=" << s.pad_volume << "\r\n"
	  << "vibracion_mando=" << s.pad_vibration << "\r\n"
	  << "[volante]\r\n"
	  << "volante=" << b(s.wheel) << "\r\n"
	  << "volante_ffb=" << s.wheel_ffb_gain << "\r\n"
	  << "volante_grados=" << s.wheel_rotation << "\r\n"
	  << "volante_tope_suave=" << b(s.wheel_soft_lock) << "\r\n"
	  << "volante_ffb_invertido=" << b(s.wheel_reverse_ffb) << "\r\n"
	  << "volante_condiciones_software=" << b(s.wheel_soft_cond) << "\r\n"
	  << "volante_segundo_plano=" << b(s.wheel_background) << "\r\n"
	  << "[teclado_raton]\r\n"
	  << "teclado_raton=" << b(s.kbm) << "\r\n"
	  << "sensibilidad_raton=" << s.mouse_sens << "\r\n"
	  << "mapa_teclado=" << s.keymap << "\r\n"
	  << "[metricas]\r\n"
	  << "metricas=" << b(s.metrics) << "\r\n"
	  << "metricas_fps=" << b(s.metrics_fps) << "\r\n"
	  << "metricas_ram=" << b(s.metrics_ram) << "\r\n"
	  << "metricas_vram=" << b(s.metrics_vram) << "\r\n"
	  << "metricas_grafica=" << b(s.metrics_graph) << "\r\n"
	  << "metricas_posicion=" << s.metrics_pos << "\r\n"
	  << "[desarrollador]\r\n"
	  << "dev_validacion_vulkan=" << b(s.dev_vulkan_validation) << "\r\n"
	  << "dev_log_shaders=" << b(s.dev_shader_log) << "\r\n"
	  << "dev_printf_archivo=" << b(s.dev_printf_file) << "\r\n"
	  << "dev_renderdoc=" << b(s.dev_renderdoc) << "\r\n"
	  << "dev_profiler=" << b(s.dev_profiler) << "\r\n"
	  << "dev_consola=" << b(s.dev_console) << "\r\n"
	  << "argumentos_extra=" << s.dev_args << "\r\n"
	  << "entorno_extra=" << s.dev_env << "\r\n"
	  << "configurado=" << b(s.configured) << "\r\n";
	const auto      p   = std::filesystem::u8path(path);
	const auto      tmp = std::filesystem::path(p.wstring() + L".tmp");
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return false;
		const std::string text = o.str();
		out.write(text.data(), static_cast<std::streamsize>(text.size()));
		if (!out) return false;
	}
	std::error_code ec;
	std::filesystem::rename(tmp, p, ec);
	return !ec;
}

LaunchPlan BuildLaunch(const Settings& s, bool amd_cpu_detected, int desktop_w, int desktop_h) {
	LaunchPlan p;
	auto       add = [&](const std::string& a, const std::string& v) {
		p.args.push_back(a);
		p.args.push_back(v);
	};
	add("--game", s.game);
	if (!s.game_patch.empty()) add("--game-patch", s.game_patch);
	int w = desktop_w > 0 ? desktop_w : 1920, h = desktop_h > 0 ? desktop_h : 1080;
	if (s.resolution != "escritorio") {
		int rw = 0, rh = 0;
		if (std::sscanf(s.resolution.c_str(), "%dx%d", &rw, &rh) == 2 && rw > 0 && rh > 0) {
			w = rw;
			h = rh;
		}
	}
	add("--screen-width", std::to_string(w));
	add("--screen-height", std::to_string(h));
	if (s.fullscreen) p.args.push_back("--fullscreen");
	if (s.hide_cursor) p.args.push_back("--hide-cursor");
	add("--present-mode", s.present == "fifo" ? "Fifo" : s.present == "immediate" ? "Immediate" : "Mailbox");
	if (s.vblank != 60) add("--vblank-frequency", std::to_string(s.vblank));
	if (s.tessellation) p.args.push_back("--tessellation");
	if (s.reflections_every > 0) add("--reflections-every", std::to_string(s.reflections_every));
	if (s.bindless) p.args.push_back("--bindless");
	if (s.gpu >= 0) add("--gpu", std::to_string(s.gpu));
	if (s.amd_cpu == "si" || (s.amd_cpu == "auto" && amd_cpu_detected)) p.args.push_back("--amd-cpu");
	p.args.push_back(s.redzone ? "--redzone" : "--no-redzone");
	if (s.playgo_hack) p.args.push_back("--playgo-hack");
	add("--console-language", std::to_string(s.console_language));
	if (!s.user_name.empty()) add("--user-name", s.user_name.substr(0, 16));
	if (!s.pad_color.empty()) add("--controller-color", s.pad_color);
	add("--controller-volume", std::to_string(s.pad_volume));
	add("--controller-vibration", std::to_string(s.pad_vibration));
	add("--wheel", s.wheel ? "true" : "false");
	if (s.wheel) add("--wheel-config", "wheel.json");
	if (s.kbm) {
		if (s.mouse_sens != 1.0f) {
			char buf[32];
			std::snprintf(buf, sizeof(buf), "MouseSensitivity=%.2f", s.mouse_sens);
			add("--keymap", buf);
		}
		size_t at = 0;
		while (at < s.keymap.size()) {
			const size_t      end  = std::min(s.keymap.find(';', at), s.keymap.size());
			const std::string item = Trim(s.keymap.substr(at, end - at));
			at                     = end + 1;
			if (item.find('=') != std::string::npos && item.front() != '=' && item.back() != '=') add("--keymap", item);
		}
	}
	if (s.dev_vulkan_validation) add("--vulkan-validation", "true");
	if (s.dev_shader_log) {
		add("--shader-log-direction", "File");
		add("--shader-log-folder", "logs/shaders");
	}
	if (s.dev_printf_file) {
		add("--printf-direction", "File");
		add("--printf-output-file", "logs/printf.txt");
	}
	if (s.dev_renderdoc) p.args.push_back("--rd");
	if (s.dev_profiler) p.args.push_back("--profile");
	for (auto& a: SplitArgs(s.dev_args)) p.args.push_back(a);

	if (s.cuda_early) p.env.emplace_back("KYTY_CUDA_EARLY_INIT", "1");
	if (s.log_stats) {
		// The emulator's counters in the log (entorno_extra, set after, can change them): a "Perf:" line with the fps
		// every 5 s (common/perfStats.cpp, KYTY_PERF=<seconds>) and the "SyncStats:" lines (bufferCache.cpp).
		p.env.emplace_back("KYTY_PERF", "5");
		p.env.emplace_back("KYTY_SYNC_STATS", "1");
	}
	size_t at = 0;
	while (at < s.dev_env.size()) {
		const size_t      end  = std::min(s.dev_env.find(';', at), s.dev_env.size());
		const std::string item = Trim(s.dev_env.substr(at, end - at));
		at                     = end + 1;
		const auto eq = item.find('=');
		if (eq != std::string::npos && eq > 0) p.env.emplace_back(Trim(item.substr(0, eq)), Trim(item.substr(eq + 1)));
	}
	return p;
}

bool WriteWheelJson(const std::string& path, const Settings& s, std::string* error) {
	using Json = nlohmann::json;
	const auto p = std::filesystem::u8path(path);
	Json       j = Json::object();
	{
		std::ifstream in(p, std::ios::binary);
		if (in) {
			try {
				j = Json::parse(in);
			} catch (...) {
				j = Json::object();
			}
		}
		if (!j.is_object()) j = Json::object();
	}
	if (!j.contains("version")) j["version"] = 1;
	if (!j.contains("model")) j["model"] = "gt_dd_pro";
	j["background_input"] = s.wheel_background;
	Json& ffb = j["ffb"];
	if (!ffb.is_object()) ffb = Json::object();
	ffb["gain"]                = s.wheel_ffb_gain;
	ffb["reverse"]             = s.wheel_reverse_ffb;
	ffb["software_conditions"] = s.wheel_soft_cond;
	Json& rot = j["rotation"];
	if (!rot.is_object()) rot = Json::object();
	rot["physical_degrees"] = s.wheel_rotation;
	rot["soft_lock"]        = s.wheel_soft_lock;
	std::ofstream out(p, std::ios::binary | std::ios::trunc);
	if (!out) {
		if (error) *error = "no se pudo escribir wheel.json";
		return false;
	}
	out << j.dump(2);
	return static_cast<bool>(out);
}

void ReadWheelJson(const std::string& path, Settings* s) {
	using Json = nlohmann::json;
	std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
	if (!in) return;
	try {
		const Json j = Json::parse(in);
		if (!j.is_object()) return;
		if (auto it = j.find("background_input"); it != j.end() && it->is_boolean()) s->wheel_background = it->get<bool>();
		if (auto f = j.find("ffb"); f != j.end() && f->is_object()) {
			if (auto it = f->find("gain"); it != f->end() && it->is_number()) s->wheel_ffb_gain = std::clamp(it->get<int>(), 0, 100);
			if (auto it = f->find("reverse"); it != f->end() && it->is_boolean()) s->wheel_reverse_ffb = it->get<bool>();
			if (auto it = f->find("software_conditions"); it != f->end() && it->is_boolean()) s->wheel_soft_cond = it->get<bool>();
		}
		if (auto r = j.find("rotation"); r != j.end() && r->is_object()) {
			if (auto it = r->find("physical_degrees"); it != r->end() && it->is_number()) s->wheel_rotation = std::clamp(it->get<int>(), 180, 2520);
			if (auto it = r->find("soft_lock"); it != r->end() && it->is_boolean()) s->wheel_soft_lock = it->get<bool>();
		}
	} catch (...) {
	}
}

} // namespace Port
