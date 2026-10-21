#include "port/ds-settings.h"

#include "port/ds-tables.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace Port {

// Launcher-souls.ps1 $resModes (the deployed launcher's list, labels as there).
const ResMode kResModes[] = {
    {"nativo", "1440p nativo", 1440, ""},
    {"dlss-dlaa", "DLSS DLAA", 1.0, "dlss"},
    {"dlss-ucalidad", "DLSS Ultra calidad", 1.3, "dlss"},
    {"dlss-calidad", "DLSS Calidad", 1.5, "dlss"},
    {"dlss-equilibrado", "DLSS Equilibrado", 1.72, "dlss"},
    {"dlss-rendimiento", "DLSS Rendimiento", 2.0, "dlss"},
    {"xess-aa", "XeSS AA", 1.0, "xess"},
    {"xess-ucalidad-plus", "XeSS Ultra calidad +", 1.3, "xess"},
    {"xess-ucalidad", "XeSS Ultra calidad", 1.5, "xess"},
    {"xess-calidad", "XeSS Calidad", 1.7, "xess"},
    {"xess-equilibrado", "XeSS Equilibrado", 2.0, "xess"},
    {"xess-rendimiento", "XeSS Rendimiento", 2.3, "xess"},
    {"fsr-aa", "FSR AA (FSR 3.1 nativo)", 1.0, "fsr"},
    {"fsr-ucalidad", "FSR Ultra calidad", 1.3, "fsr"},
    {"fsr-calidad", "FSR Calidad", 1.5, "fsr"},
    {"fsr-equilibrado", "FSR Equilibrado", 1.72, "fsr"},
    {"fsr-rendimiento", "FSR Rendimiento", 2.0, "fsr"},
};
const int kResModeCount = static_cast<int>(sizeof(kResModes) / sizeof(kResModes[0]));

const Choice kSharpness[] = {{"auto", "Auto"}, {"0", "0"}, {"0.25", "25%"}, {"0.5", "50%"}, {"0.75", "75%"}, {"1", "100%"}};
const int    kSharpnessCount = 6;
const Choice kDlssPresets[] = {{"auto", "Auto (NVIDIA)"}, {"K", "DLSS 4 (K)"}, {"L", "DLSS 4.5 (L)"}, {"M", "DLSS 4.5 (M)"}};
const int    kDlssPresetCount = 4;
const Choice kQualities[] = {{"standard", "Normal (PS5)"}, {"low", "Baja"}, {"lowest", "Minima"}};
const int    kQualityCount = 3;
const Choice kTextureReduce[] = {{"auto", "Auto (segun VRAM)"}, {"0", "Completas"}, {"1", "Mitad"}, {"2", "Un cuarto"}};
const int    kTextureReduceCount = 4;
const Choice kMetrics[] = {{"0", "No"}, {"1", "FPS"}, {"2", "FPS + detalle"}};
const int    kMetricsCount = 3;
const Choice kLanguages[] = {{"3", "Espanol (Espana)"}, {"20", "Espanol (Latinoamerica)"}, {"1", "English (US)"},
                             {"18", "English (UK)"},    {"2", "Francais"},                 {"4", "Deutsch"},
                             {"5", "Italiano"},         {"7", "Portugues (Portugal)"},     {"17", "Portugues (Brasil)"},
                             {"0", "Japanese"},         {"11", "Chinese (Simplified)"}};
const int    kLanguageCount = 11;

namespace {

std::string Trim(std::string s) {
	const auto not_space = [](unsigned char c) { return c != ' ' && c != '\t' && c != '\r' && c != '\n'; };
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
	s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
	return s;
}

bool Bool(const std::string& v, bool fallback) {
	if (v == "1" || v == "si" || v == "true" || v == "on") return true;
	if (v == "0" || v == "no" || v == "false" || v == "off") return false;
	return fallback;
}

std::string Num(double v) {
	char text[32];
	std::snprintf(text, sizeof(text), "%g", v);
	return text;
}

// The environment as run-windows.ps1's ordered hashtable.
struct Env {
	EnvList list;
	int     Find(const std::string& key) const {
        for (size_t i = 0; i < list.size(); i++)
            if (list[i].first == key) return static_cast<int>(i);
        return -1;
	}
	bool        Has(const std::string& key) const { return Find(key) >= 0; }
	std::string Get(const std::string& key) const { return Has(key) ? list[Find(key)].second : std::string(); }
	void        Set(const std::string& key, const std::string& value) {
        if (const int i = Find(key); i >= 0) list[i].second = value;
        else list.emplace_back(key, value);
	}
	void Remove(const std::string& key) {
		if (const int i = Find(key); i >= 0) list.erase(list.begin() + i);
	}
	void Default(const std::string& key, const std::string& value) {
		if (!Has(key)) Set(key, value);
	}
	void AppendConvars(const std::string& convars) {
		const std::string now = Get("KYTY_GAME_CONVARS");
		if (now.find(convars) != std::string::npos) return;
		Set("KYTY_GAME_CONVARS", now.empty() ? convars : now + ";" + convars);
	}
};

const ResMode* FindRes(const std::string& id) {
	for (int i = 0; i < kResModeCount; i++)
		if (id == kResModes[i].id) return &kResModes[i];
	return &kResModes[0];
}

const Tables::FgMode* FindFg(const std::string& id) {
	for (const auto& mode: Tables::kFgModes)
		if (id == mode.id) return &mode;
	return &Tables::kFgModes[0];
}

} // namespace

const char* FrameGenLabel(const std::string& id) {
	return FindFg(id)->label;
}

bool LoadSettings(const std::string& path, Settings* s) {
	std::ifstream in(path);
	if (!in) return false;
	std::map<std::string, std::string> v;
	std::string                        line;
	while (std::getline(in, line)) {
		line = Trim(line);
		if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF) line = line.substr(3); // UTF-8 BOM
		if (line.empty() || line[0] == ';' || line[0] == '#' || line[0] == '[') continue;
		const auto eq = line.find('=');
		if (eq == std::string::npos) continue;
		v[Trim(line.substr(0, eq))] = Trim(line.substr(eq + 1));
	}
	const auto str = [&](const char* k, std::string& out) {
		if (auto it = v.find(k); it != v.end()) out = it->second;
	};
	const auto boo = [&](const char* k, bool& out) {
		if (auto it = v.find(k); it != v.end()) out = Bool(it->second, out);
	};
	const auto num = [&](const char* k, auto& out) {
		if (auto it = v.find(k); it != v.end() && !it->second.empty()) out = static_cast<std::remove_reference_t<decltype(out)>>(std::atof(it->second.c_str()));
	};
	str("carpeta_juego", s->game);
	num("idioma", s->language);
	boo("copia_partidas", s->backup_saves);
	boo("pantalla_completa", s->fullscreen);
	num("ancho", s->width);
	num("alto", s->height);
	boo("hdr", s->hdr);
	boo("vsync", s->vsync);
	boo("gsync", s->gsync);
	boo("fps_desbloqueados", s->unlock_fps);
	boo("ajustar_aspecto", s->aspect_fit);
	str("aspecto", s->aspect);
	num("tope_fps", s->fps_cap);
	str("reescalado", s->res);
	str("nitidez", s->sharpness);
	str("preset_dlss", s->dlss_preset);
	str("frame_generation", s->framegen);
	boo("perfil_16gb", s->profile16);
	boo("gama_baja", s->low_end);
	boo("gpu_seguro", s->safe_gpu);
	str("calidad", s->quality);
	str("reducir_texturas", s->texture_reduce);
	boo("simular_vram_8gb", s->simulate_vram8);
	boo("todos_los_ccd", s->all_ccds);
	str("metricas", s->metrics);
	boo("vibracion", s->rumble);
	num("vibracion_fuerza", s->rumble_gain);
	str("extra", s->extra);
	return true;
}

bool SaveSettings(const std::string& path, const Settings& s) {
	std::ostringstream o;
	const auto         b = [](bool x) { return x ? "1" : "0"; };
	o << "; Demon's Souls PC - ajustes. Se cambian tambien dentro del juego (F1, o mantener R3 y pulsar OPTIONS).\n"
	  << "; 1 = si, 0 = no. Lo que no se reconoce se ignora; borrar este archivo vuelve a los valores por defecto.\n\n"
	  << "[Juego]\n"
	  << "; Carpeta del juego volcado (la que tiene eboot.bin y sce_sys), o un .zar. Vacia: se pregunta al arrancar.\n"
	  << "carpeta_juego=" << s.game << "\n"
	  << "; Idioma de la consola: 3 espanol (Espana), 20 espanol (Latinoamerica), 1 ingles (EE. UU.), 18 ingles (R. U.)\n"
	  << "idioma=" << s.language << "\n"
	  << "; Copia de seguridad de _SaveData en _SaveData-copias antes de jugar\n"
	  << "copia_partidas=" << b(s.backup_saves) << "\n\n"
	  << "[Pantalla]\n"
	  << "pantalla_completa=" << b(s.fullscreen) << "\n"
	  << "; Tamano de la ventana (sin pantalla completa)\n"
	  << "ancho=" << s.width << "\nalto=" << s.height << "\n"
	  << "; HDR10: solo si \"Usar HDR\" esta activado en la configuracion de pantalla de Windows\n"
	  << "hdr=" << b(s.hdr) << "\n"
	  << "vsync=" << b(s.vsync) << "\n"
	  << "; G-Sync / FreeSync: el reloj del emulador no se ata a los Hz del monitor\n"
	  << "gsync=" << b(s.gsync) << "\n"
	  << "; Experimental: el juego pasa de 60 FPS reales (videos y menus siguen a 60)\n"
	  << "fps_desbloqueados=" << b(s.unlock_fps) << "\n"
	  << "; Tope de FPS con G-Sync y FPS desbloqueados sin frame generation (0 = automatico, justo bajo el refresco)\n"
	  << "tope_fps=" << Num(s.fps_cap) << "\n"
	  << "; Mantener 16:9 con bandas negras en vez de estirar\n"
	  << "ajustar_aspecto=" << b(s.aspect_fit) << "\n"
	  << "; Aspecto del 3D: 16:9, 21:9 o 32:9 (ultrapanoramico)\n"
	  << "aspecto=" << s.aspect << "\n\n"
	  << "[Reescalado]\n"
	  << "; nativo, dlss-dlaa, dlss-ucalidad, dlss-calidad, dlss-equilibrado, dlss-rendimiento,\n"
	  << "; xess-aa, xess-ucalidad-plus, xess-ucalidad, xess-calidad, xess-equilibrado, xess-rendimiento,\n"
	  << "; fsr-aa, fsr-ucalidad, fsr-calidad, fsr-equilibrado, fsr-rendimiento\n"
	  << "reescalado=" << s.res << "\n"
	  << "; auto, 0, 0.25, 0.5, 0.75, 1\n"
	  << "nitidez=" << s.sharpness << "\n"
	  << "; auto, K (DLSS 4), L o M (DLSS 4.5)\n"
	  << "preset_dlss=" << s.dlss_preset << "\n\n"
	  << "[FrameGeneration]\n"
	  << "; sin-fg, amd-x2, amd-x3, amd-x4, amd-120, amd-dinamico,\n"
	  << "; dlss-x2, dlss-x3, dlss-x4, dlss-x5, dlss-x6, dlss-120, dlss-dinamico (NVIDIA RTX)\n"
	  << "frame_generation=" << s.framegen << "\n"
	  << "\n"
	  << "[Rendimiento]\n"
	  << "; PCs de 16 GB de RAM: 6 GB del juego fijos en RAM y la precarga de shaders espera con poca RAM libre\n"
	  << "perfil_16gb=" << b(s.profile16) << "\n"
	  << "; PC bajos requisitos: 30 FPS reales y frame generation AMD x2 (60 en pantalla); manda sobre frame_generation; sin DLSS de reescalado\n"
	  << "gama_baja=" << b(s.low_end) << "\n"
	  << "; Para graficas que se cuelgan o dan pantallazo azul (~37% menos FPS)\n"
	  << "gpu_seguro=" << b(s.safe_gpu) << "\n"
	  << "; standard, low, lowest (menos detalle lejano, sombras intactas: ayuda si limita la CPU)\n"
	  << "calidad=" << s.quality << "\n"
	  << "; auto, 0 (completas), 1 (mitad), 2 (un cuarto)\n"
	  << "reducir_texturas=" << s.texture_reduce << "\n"
	  << "simular_vram_8gb=" << b(s.simulate_vram8) << "\n"
	  << "; Ryzen con dos CCD: 0 = el juego en el CCD con mas cache L3 (mas rapido); 1 = todos los nucleos\n"
	  << "todos_los_ccd=" << b(s.all_ccds) << "\n\n"
	  << "[Interfaz]\n"
	  << "; Metricas en pantalla: 0 no, 1 FPS, 2 FPS + detalle\n"
	  << "metricas=" << s.metrics << "\n"
	  << "; Vibracion del mando (el audio haptico del DualSense como vibracion; mandos Xbox, DualSense por Bluetooth)\n"
	  << "vibracion=" << b(s.rumble) << "\n"
	  << "vibracion_fuerza=" << Num(s.rumble_gain) << "\n\n"
	  << "[Avanzado]\n"
	  << "; Interruptores del emulador: KYTY_X=1;KYTY_Y=2 (KYTY_X= lo quita). Solo para pruebas.\n"
	  << "extra=" << s.extra << "\n";
	const std::string tmp = path + ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return false;
		out << o.str();
		if (!out) return false;
	}
	std::remove(path.c_str());
	return std::rename(tmp.c_str(), path.c_str()) == 0;
}

LaunchPlan BuildLaunch(const Settings& s, const std::string& title_id) {
	LaunchPlan plan;
	Env        env;
	const bool demons = title_id.empty() || title_id.rfind("PPSA0134", 0) == 0;
	// The frame generation config (config/<mode>.json): the common switches, then the mode's.
	// PC bajos requisitos: AMD x2 whatever frame_generation says, the game flipped at 30 (12 vblanks of 360 Hz).
	const auto* fg = FindFg(s.low_end ? "amd-x2" : s.framegen);
	for (const auto& p: Tables::kBaseEnv) env.Set(p.key, p.value);
	for (int i = 0; i < fg->env_count; i++) env.Set(fg->env[i].key, fg->env[i].value);
	const bool with_fg = std::strcmp(fg->id, "sin-fg") != 0;
	const bool amd_fg  = std::strncmp(fg->id, "amd", 3) == 0;

	// Launcher-souls.ps1: super resolution. DLSS with AMD frame generation closed the emulator at device creation
	// (Streamline): the launcher refused it; here the same ratio with FSR.
	// PC bajos requisitos: no DLSS upscaling (a saved DLSS mode plays native).
	const bool     no_dlss  = s.low_end && std::string(FindRes(s.res)->upscaler) == "dlss";
	if (no_dlss) plan.notes.emplace_back("PC bajos requisitos: sin reescalado DLSS, se juega nativo");
	const ResMode* res      = no_dlss ? &kResModes[0] : FindRes(s.res);
	std::string    upscaler = res->upscaler;
	double         ratio    = upscaler.empty() ? 0.0 : res->value;
	if (upscaler == "dlss" && amd_fg) {
		upscaler = "fsr";
		plan.notes.emplace_back("DLSS (reescalado) no funciona con la frame generation de AMD: se usa FSR con la misma relacion");
	}

	// Launcher-souls.ps1's -Set switches.
	EnvList sets;
	if (s.safe_gpu) sets.emplace_back("KYTY_COMPUTE_CHAIN", "0");
	if (s.sharpness != "auto" && !s.sharpness.empty()) sets.emplace_back("KYTY_UPSCALE_SHARPNESS", s.sharpness);
	if (s.unlock_fps && with_fg) sets.emplace_back("KYTY_FRAMEGEN_BASE", "0");
	sets.emplace_back("KYTY_FAULT_STATS", "1");
	if (std::strcmp(fg->id, "amd-x4") == 0) sets.emplace_back("KYTY_VRR_CAP_FPS", "224");
	// Its vblank: 360 Hz with frame generation (the game's speed corrected), 960 unlocked without, else 60.
	const int vblank = with_fg ? 360 : (s.unlock_fps ? 960 : 60);
	plan.vblank      = vblank;
	const std::string& quality = s.quality;
	const std::string& texture = s.texture_reduce;

	// run-windows.ps1, in its order.
	if (env.Get("KYTY_RENDER_CPUS").empty() && env.Has("KYTY_RECORDING_CPUS")) env.Set("KYTY_RENDER_CPUS", env.Get("KYTY_RECORDING_CPUS"));
	if (env.Get("KYTY_ASYNC_UPLOAD") == "1") env.Set("KYTY_ASYNC_UPLOAD", "2");
	env.Default("KYTY_BUFFER_RECLAIM", "1");
	env.Default("KYTY_DRAW_PACKETS", "1");
	env.Default("KYTY_PARTIAL_IMAGE_DIRTY", "1");
	env.Default("KYTY_PARTIAL_ROW_BANDS", "1");
	env.Default("KYTY_ASYNC_XPR_PIPELINES", "1");
	env.Default("KYTY_NATIVE_XPR_RELOCATE", "1");
	env.Default("KYTY_NATIVE_XPR_STORE_BUDGET", "256");
	env.Default("KYTY_NATIVE_XPR_KEEP_FRAMES", "600");
	env.Default("KYTY_NATIVE_XPR_INSTANCES", "1");
	if (demons) env.Default("KYTY_GAME_CONVARS", "doIndexBufferCulling=false");
	if (ratio > 0) env.Set("KYTY_UPSCALE_RATIO", Num(ratio));
	env.Remove("KYTY_UPSCALE");
	if (upscaler == "dlss" && s.dlss_preset != "auto" && !s.dlss_preset.empty()) env.Set("KYTY_DLSS_PRESET", s.dlss_preset);
	if (!s.aspect.empty() && s.aspect != "16:9") env.Set("KYTY_RENDER_ASPECT", s.aspect);
	static const std::map<std::string, std::string> convars = {
	    // (Shadows untouched: their distance and kinds popped in visibly.)
	    {"low", "r_forceViewportHeightForPixelSizeCalc=720"},
	    {"lowest", "r_forceViewportHeightForPixelSizeCalc=480"}};
	if (auto it = convars.find(quality); it != convars.end() && demons) env.AppendConvars(it->second);
	if (texture == "0" || texture == "1" || texture == "2") env.Set("KYTY_TEXTURE_REDUCE", texture);
	env.Default("KYTY_UNMAP_PROTECT_SKIP", "1");
	env.Set("KYTY_DRIVER_CACHE_KEY", "1f3c29b70e536c8f5a5e812e793b33ac2eb6858b4810ed98ca2a51b9fa5eed97");
	env.Set("KYTY_SHADER_WARMUP_ONLY", "");
	env.Default("KYTY_SHADER_WARMUP_SECONDS", "60");
	env.Default("KYTY_TABLE_STORE_BUDGET", "8");
	env.Default("KYTY_HITCH_LOG_MS", "40");
	if (s.aspect_fit) env.Set("KYTY_PRESENT_ASPECT", "fit");
	env.Default("KYTY_ASYNC_DRAW_PIPELINES", "1");
	if (s.profile16) {
		env.Default("KYTY_RESIDENT_MB", "6144");
		env.Default("KYTY_PREFETCH_MIN_AVAIL_MB", "3072");
	}
	if (!upscaler.empty()) env.Set("KYTY_UPSCALER", upscaler);
	for (const auto& [k, v]: sets) env.Set(k, v);
	// settings.ini [Avanzado] extra: as -Set (KEY= removes it).
	{
		std::stringstream list(s.extra);
		std::string       pair;
		while (std::getline(list, pair, ';')) {
			pair          = Trim(pair);
			const auto eq = pair.find('=');
			if (eq == std::string::npos || eq == 0) continue;
			const auto key = Trim(pair.substr(0, eq)), value = Trim(pair.substr(eq + 1));
			if (value.empty()) env.Set(key, "");
			else env.Set(key, value);
		}
	}
	if (vblank > 60 && demons && env.Get("KYTY_GAME_CONVARS").find("gameMinDeltaTime") == std::string::npos)
		env.AppendConvars("gameMinDeltaTime=0.004");
	// The launcher's own environment.
	env.Set("KYTY_SIMULATE_VRAM_MB", s.simulate_vram8 ? "8192" : "");
	env.Set("KYTY_UNLOCK_30", "0");
	env.Set("KYTY_VSYNC_LOCK", "");
	env.Set("KYTY_FPS_CAP", "");
	env.Set("KYTY_FLIP_WHEN_READY", "");
	if (s.gsync) {
		env.Set("KYTY_VSYNC_LOCK", "0");
		if (s.unlock_fps && !with_fg) {
			env.Set("KYTY_FPS_CAP", s.fps_cap > 0 ? Num(s.fps_cap) : "auto");
			env.Set("KYTY_FLIP_WHEN_READY", "1");
		}
	}
	env.Set("KYTY_DLSSG_PRESET", "C"); // (always: the emulator applies it to nvngx_dlssg only)
	if (s.low_end) env.Set("KYTY_FLIP_FPS", "30");
	// The port's own: metrics (the launcher started metricas.ps1; here the in-game panel) and rumble.
	env.Set("KYTY_FPS_HUD", s.metrics == "1" || s.metrics == "2" ? s.metrics : "");
	env.Set("KYTY_HAPTIC_RUMBLE", s.rumble ? "1" : "0");
	env.Set("KYTY_HAPTIC_RUMBLE_GAIN", Num(s.rumble_gain > 0 ? s.rumble_gain : 4.0f));
	plan.env   = env.list;
	plan.ratio = ratio;

	// The command line (config/*.json's, then run-windows.ps1's options).
	auto& a = plan.args;
	a       = {"--screen-width", std::to_string(s.width > 0 ? s.width : 2560), "--screen-height",
	           std::to_string(s.height > 0 ? s.height : 1440), "--console-language", std::to_string(s.language),
	           "--shader-validation", "false", "--shader-log-direction", "Silent", "--printf-direction", "Silent",
	           "--vblank-frequency", std::to_string(vblank), "--present-mode", s.vsync ? "Fifo" : "Immediate"};
	if (s.fullscreen) a.emplace_back("--fullscreen");
	if (s.hdr) a.emplace_back("--hdr");
	a.emplace_back("--redzone");
	a.emplace_back("--game");
	a.push_back(s.game);
	return plan;
}

bool NeedsRestart(const Settings& l, const Settings& n) {
	// Live: sharpness, the fps cap, metrics and rumble.
	return l.game != n.game || l.language != n.language || l.fullscreen != n.fullscreen || l.width != n.width ||
	       l.height != n.height || l.hdr != n.hdr || l.vsync != n.vsync || l.gsync != n.gsync ||
	       l.unlock_fps != n.unlock_fps || l.aspect_fit != n.aspect_fit || l.aspect != n.aspect || l.res != n.res ||
	       l.dlss_preset != n.dlss_preset || l.framegen != n.framegen || l.low_end != n.low_end ||
	       l.profile16 != n.profile16 || l.safe_gpu != n.safe_gpu || l.quality != n.quality ||
	       l.texture_reduce != n.texture_reduce || l.simulate_vram8 != n.simulate_vram8 || l.all_ccds != n.all_ccds ||
	       l.extra != n.extra || (l.sharpness == "auto") != (n.sharpness == "auto");
}

} // namespace Port
