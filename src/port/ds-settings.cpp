#include "port/ds-settings.h"

#include "port/ds-tables.h"

#include <algorithm>
#include <thread>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>

namespace Port {

// Launcher-souls.ps1 $resModes (the deployed launcher's list, labels as there).
const ResMode kResModes[] = {
    // (10-27) Launcher-souls.ps1 $resModes: its labels; DLSS Calidad 1.333 (10-21: DLSS goes out at 4K inside, the game's
    // PS5 Pro path: it takes 1080 lines or more; 1440 / 1.5 = 960 left the picture black).
    {"nativo", "1440p nativo", 1440, ""},
    {"dlss-dlaa", "DLSS DLAA", 1.0, "dlss"},
    {"dlss-ucalidad", "DLSS U.calidad", 1.3, "dlss"},
    {"dlss-calidad", "DLSS Calidad", 1.333, "dlss"},
    {"dlss-equilibrado", "DLSS Equilib.", 1.72, "dlss"},
    {"dlss-rendimiento", "DLSS Rendim.", 2.0, "dlss"},
    {"dlss-urendimiento", "DLSS U.rendim.", 3.0, "dlss"}, // (10-28) 2160 / 3 = 720; below 4K BuildLaunch snaps it to 720
    {"xess-aa", "XeSS AA", 1.0, "xess"},
    {"xess-ucalidad-plus", "XeSS U.cal.+", 1.3, "xess"},
    {"xess-ucalidad", "XeSS U.calidad", 1.5, "xess"},
    {"xess-calidad", "XeSS Calidad", 1.7, "xess"},
    {"xess-equilibrado", "XeSS Equilib.", 2.0, "xess"},
    {"xess-rendimiento", "XeSS Rendim.", 2.3, "xess"},
    {"fsr-aa", "FSR AA nativo", 1.0, "fsr"},
    {"fsr-ucalidad", "FSR U.calidad", 1.3, "fsr"},
    {"fsr-calidad", "FSR Calidad", 1.5, "fsr"},
    {"fsr-equilibrado", "FSR Equilib.", 1.72, "fsr"},
    {"fsr-rendimiento", "FSR Rendim.", 2.0, "fsr"},
    {"fsr-900", "FSR 900p (AA)", 1.6, "fsr"},
    // (10-28) AMD FSR 4 (the v07 INT8 model, src/local/fsr4-sr.h; fsr4_shaders next to the exe): its own upscaler, one model
    // per preset (Launcher-souls.ps1 ds-torch-1028). Output at the display's size; the 3D down to 360 lines.
    {"fsr4-aa", "FSR 4 Nativo (AA)", 1.0, "fsr4", "native"},
    {"fsr4-calidad", "FSR 4 Calidad", 1.5, "fsr4", "quality"},
    {"fsr4-equilibrado", "FSR 4 Equilibrado", 1.7, "fsr4", "balanced"},
    {"fsr4-rendimiento", "FSR 4 Rendimiento", 2.0, "fsr4", "performance"},
    {"fsr4-urendimiento", "FSR 4 Ultra rendimiento", 3.0, "fsr4", "ultraperf"},
    {"fsr4-dinamico", "FSR 4 Dinámico", 1.5, "fsr4", "drs"},
    // (10-28) Launcher-souls.ps1's "juego" modes: no upscaler, the game's own upscale from this 3D height (KYTY_RENDER_HEIGHT).
    {"juego-1080", "1080p juego", 1080, ""},
    {"juego-900", "900p juego", 900, ""},
    {"juego-720", "720p juego", 720, ""},
};
const int kResModeCount = static_cast<int>(sizeof(kResModes) / sizeof(kResModes[0]));

const Choice kSharpness[] = {{"auto", "Nitidez: auto"}, {"0", "Nitidez: 0"}, {"0.25", "Nitidez: 25%"}, {"0.5", "Nitidez: 50%"}, {"0.75", "Nitidez: 75%"}, {"1", "Nitidez: 100%"}};
const int    kSharpnessCount = 6;
const Choice kDlssPresets[] = {{"auto", "DLSS auto"}, {"K", "DLSS 4 (K)"}, {"L", "DLSS 4.5 (L)"}, {"M", "DLSS 4.5 (M)"}};
const int    kDlssPresetCount = 4;
const Choice kQualities[] = {{"standard", "Maximo"}, {"high", "Alto"}, {"low", "Medio"}, {"optimizado", "Optimizado"}, {"lowest", "Bajo"}}; // draw distance (LOD)
const int    kQualityCount = 5;
const Choice kTextureReduce[] = {{"0", "Maximo"}, {"h", "Alto"}, {"1", "Medio"}, {"o", "Optimizado"}, {"2", "Bajo"}};
const int    kTextureReduceCount = 5;
const Choice kShadows[] = {{"maximo", "Maximo"}, {"alto", "Alto"}, {"medio", "Medio"}, {"optimizado", "Optimizado"}, {"bajo", "Bajo"}};
const int    kShadowCount = 5;
const Choice kIbl[] = {{"mitad", "Mitad"}, {"completa", "Completa"}};
const int    kIblCount = 2;
const Choice kTess[] = {{"bajo", "Bajo"}, {"optimizado", "Optimizado"}, {"medio", "Medio"}, {"alto", "Alto"}, {"ultra", "Ultra"}, {"extremo", "Extremo"}};
const int    kTessCount = 6;
const Choice kDetails[] = {{"ultra", "Ultra"}, {"muy-alto", "Muy alto (cine)"}, {"alto", "Alto (rendimiento de PS5)"}, {"normal", "Normal"}, {"optimizado", "Optimizado"}, {"bajo", "Bajo"}};
const int    kDetailCount = 6;
const Choice kReflections[] = {{"ultra", "Ultra"}, {"alto", "Alto (de serie)"}, {"normal", "Normal"}, {"optimizado", "Optimizado"}, {"bajo", "Bajo"}};
const int    kReflectionCount = 5;
const Choice kAaSharp[] = {{"ultra", "Ultra"}, {"alto", "Alto"}, {"normal", "Normal (de serie)"}, {"optimizado", "Optimizado"}, {"bajo", "Bajo"}, {"no", "No"}};
const int    kAaSharpCount = 6;
const Choice kShadowEdges[] = {{"0", "Normal"}, {"1", "Alta"}, {"2", "Muy alta"}, {"3", "Ultra"}};
const int    kShadowEdgeCount = 4;
const Choice kCornerShadows[] = {{"0", "Normal"}, {"1", "Alta"}, {"2", "Maxima"}};
const int    kCornerShadowCount = 3;
const Choice kTextureSharpness[] = {{"0", "Texturas: normales"}, {"1", "Texturas: nitidas"}, {"2", "Texturas: extra nitidas"}};
const int    kTextureSharpnessCount = 3;
const Choice kAntiLagAmd[] = {{"amd2", "Anti-Lag 2 (AMD)"}, {"normal", "Anti-Lag"}, {"ultra", "Anti-Lag Ultra"}, {"no", "Anti-Lag: no"}};
const int    kAntiLagAmdCount = 4;
const Choice kAntiLagOther[] = {{"no", "Anti-Lag: no"}, {"normal", "Anti-Lag"}, {"ultra", "Anti-Lag Ultra"}};
const int    kAntiLagOtherCount = 3;
const Choice kAntiLagDlssg[] = {{"reflex", "Reflex (NVIDIA)"}, {"normal", "Anti-Lag"}, {"ultra", "Anti-Lag Ultra"}};
const int    kAntiLagDlssgCount = 3;
const Choice kRtReflLevels[] = {{"0", "No"}, {"1", "Bajo"}, {"2", "Medio"}, {"3", "Alto"}, {"4", "Ultra"}, {"5", "Extremo"},
                              {"6", "H\xC3\xAD" "brido"}}; // (refl-hybrid-1028) Hibrido = Alto + the hardware fill
const int    kRtReflLevelCount = 7;
const Choice kRtReflSharp[] = {{"0", "Normal"}, {"1", "Alta"}, {"2", "M\xC3\xA1xima"}}; // (refl-sharp-1031)
const int    kRtReflSharpCount = 3;
const Choice kMotesLevels[] = {{"0", "No"}, {"1", "Sutil"}, {"2", "Media"}, {"3", "Densa"}}; // (motes-1031)
const int    kMotesLevelCount = 4;
const Choice kRtReflWaves[] = {{"0", "Como ahora"}, {"1", "Suaves"}, {"2", "Espejo"}}; // (refl-sharp-1031)
const int    kRtReflWavesCount = 3;
const Choice kRtLightLevels[] = {{"0", "No"}, {"1", "Media"}, {"2", "Alta"}, {"3", "Ultra"}, {"4", "Extremo"}};
const int    kRtLightLevelCount = 5;
const Choice kRtLevels[] = {{"0", "No"}, {"1", "Bajo"}, {"2", "Medio"}, {"3", "Alto"}};
// (ds-particles-1029) No, Suave. (ds-pc-port10) Its "Antorchas" level (the smoke lit by the torch light) is gone with
// the torch light: an old 2 is Suave.
const Choice kSmokeLevels[] = {{"0", "No"}, {"1", "Suave"}};
const int    kSmokeLevelCount = static_cast<int>(sizeof(kSmokeLevels) / sizeof(kSmokeLevels[0]));
const int    kRtLevelCount = 4;
const Choice kTorchLevels[] = {{"0", "No"}, {"1", "Bajo"}, {"2", "Medio"}, {"3", "Alto"}, {"4", "Híbrido"}};
const int    kTorchLevelCount = 5;
const Choice kDlss5Levels[] = {{"0", "No"}, {"1", "Suave"}, {"2", "Medio"}, {"3", "Fuerte"}}; // (ds-pc-port10)
const int    kDlss5LevelCount = 4;
const Choice kDlss5Files[] = {{"0", "Oficial (firmado por NVIDIA)"}, {"1", "No oficial (bajo tu responsabilidad)"}};
const int    kDlss5FileCount = 2;
const Choice kDlss5Strengths[] = {{"baja", "Baja"}, {"media", "Media"}, {"alta", "Alta"}};
const int    kDlss5StrengthCount = 3;
const Choice kYesNo[] = {{"0", "No"}, {"1", "Sí"}};
const int    kYesNoCount = 2;
const Choice kSoulsModes[] = {{"ninguno", "Personalizado (ningun modo)"},
                              {"ligero", "Modo Souls ligero"},
                              {"ligero-plus", "Modo Souls ligero +"},
                              {"equilibrado", "Modo Souls equilibrado"},
                              {"ultra", "Modo Souls ultra"},
                              {"extremo", "Modo Souls extremo"},
                              {"low", "Modo Souls PC low profile (900p, todo en bajo)"}};
const int    kSoulsModeCount = 7;
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
	num("idioma_juego", s->language);
	// (ds-pc-port9) The languages follow Windows' unless chosen in the launcher: idioma_interfaz (empty: Windows' UI
	// language, read at each start) and idioma_juego_elegido=1 (else the game's follows the interface's). The older
	// idioma_menu was written with the detected language at every start, so it is not read any more.
	boo("idioma_juego_elegido", s->game_language_set);
	str("idioma_interfaz", s->menu_language);
	if (s->menu_language == "auto") s->menu_language.clear();
	boo("copia_partidas", s->backup_saves);
	boo("continuar_f1", s->continue_f1);
	boo("pantalla_completa", s->fullscreen);
	num("ancho", s->width);
	num("alto", s->height);
	str("resolucion", s->resolution);
	str("iluminacion_ibl", s->ibl);
	if (s->ibl != "mitad" && s->ibl != "completa") s->ibl.clear();
	str("teselado", s->tess);
	if (s->tess != "bajo" && s->tess != "optimizado" && s->tess != "medio" && s->tess != "alto" && s->tess != "ultra" && s->tess != "extremo") s->tess.clear();
	if (s->resolution.empty()) s->resolution = "escritorio";
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
	str("dlss5", s->dlss5);
	boo("dlss5_no_oficial", s->dlss5_unsigned);
	str("dlss5_intensidad", s->dlss5_strength);
	if (s->dlss5_strength != "baja" && s->dlss5_strength != "media") s->dlss5_strength = "alta";
	str("frame_generation", s->framegen);
	boo("perfil_16gb", s->profile16);
	if (auto it = v.find("gama_baja"); it != v.end() && Bool(it->second, false)) s->souls_mode = "ligero"; // (before 10-23)
	boo("gpu_seguro", s->safe_gpu);
	str("calidad", s->quality);
	str("reducir_texturas", s->texture_reduce);
	boo("simular_vram_8gb", s->simulate_vram8);
	s->simulate_vram8 = false; // (11-12) option removed: always off, whatever the file says
	boo("todos_los_ccd", s->all_ccds);
	boo("carga_shaders_maxima", s->shader_turbo);
	str("precompilar_hilos", s->precompile_threads);
	str("metricas", s->metrics);
	boo("menu_inicio", s->start_menu);
	boo("lanzador_ambiente", s->launcher_ambience);
	boo("saltar_intro", s->skip_intro);
	boo("vibracion", s->rumble);
	num("vibracion_fuerza", s->rumble_gain);
	str("extra", s->extra);
	boo("antilag", s->antilag);
	// (before 10-27 "antilag" was a box, written 0 by default: the launcher's default, auto, either way)
	str("antilag_modo", s->antilag_mode);
	str("antilag_dlssg", s->antilag_dlssg);
	boo("simular_ram_16gb", s->simulate_ram16);
	s->simulate_ram16 = false; // (11-12) option removed: always off
	boo("rt_reflejos", s->rt_reflections);
	boo("rt_sombras_sol", s->rt_sun);
	boo("rt_niebla", s->rt_fog);
	boo("rt_particulas", s->rt_particles);
	boo("rt_metal", s->rt_metal);
	boo("rt_oclusion", s->rt_ao);
	str("luz_volumetrica", s->godrays);
	str("particulas_volumetricas", s->motes);
	boo("juego_original", s->pure_game);
	str("bordes_sombras", s->shadow_edges);
	str("sombra_rincones", s->corner_shadow);
	str("modo_souls", s->souls_mode);
	boo("optimizacion_8gb", s->gpu8);
	str("sombras", s->shadows);
	str("detalle", s->detail);
	str("reflejos", s->reflections);
	str("distancia_objetos", s->object_fade);
	str("nitidez_aa", s->aa_sharp);
	str("luz_rt", s->rt_light);
	str("oclusion_ambiental", s->ambient_occlusion);
	boo("desenfoque_movimiento", s->motion_blur);
	boo("profundidad_campo", s->dof);
	boo("vineteado", s->vignette);
	boo("resplandor", s->bloom);
	boo("adaptacion_ojo", s->eye_adapt);
	boo("filtrado_x16", s->af16);
	str("texturas_nitidas", s->texture_sharpness); // (0 / 1 before 10-27: the same values)
	boo("teclado_raton", s->kbm);
	num("raton_sensibilidad", s->kbm_sens);
	num("raton_antizona", s->kbm_antidz);
	str("mapa_teclado", s->kbm_map);
	str("mapa_mando", s->pad_map);
	str("ajustes_juego", s->game_settings);
	str("rt_antorchas", s->rt_lamps);
	num("raton_velocidad_camara_forzada", s->kbm_camera); // (step5: a new key; the old raton_velocidad_camara=10 forced the pad too)
	s->kbm_camera = std::clamp(s->kbm_camera, 5, 15);
	boo("raton_camara_instantanea", s->kbm_warp);
	boo("raton_calibrar", s->kbm_calibrate);
	boo("configurado", s->configured);
	boo("ahorro_memoria_v3", s->ram16_boost); // (port28) new key: the old one was on by default
	s->ram16_boost = false; // (11-13) options removed: Ahorro de memoria RAM and Modo seguro
	s->safe_gpu    = false;
	boo("rend_registros_pm4", s->perf_pm4_regfile);
	boo("rend_subidas_dos_hilos", s->perf_upload_lanes);
	boo("rend_ayuda_subidas", s->perf_upload_help);
	boo("rend_espera_hilos", s->perf_spin_gate);
	boo("rend_feedback_anticipado", s->perf_eager_feedback);
	boo("rend_feedback_agrupado", s->perf_feedback_batch);
	boo("rend_shaders_lane_local", s->perf_lane_wide);
	str("reflejos_rt", s->rt_ss_refl);
	str("nitidez_reflejos_rt", s->rt_refl_sharp);
	str("ondas_agua_reflejos", s->rt_refl_waves);
	str("luz_rt_pantalla", s->rt_ssgi);
	str("sombras_rt", s->rt_shadows);
	str("oclusion_ao", s->ao);
	str("humo_particulas", s->smoke);
	str("luz_antorchas", s->torch);
	boo("destellos_lente", s->lens_flare);
	boo("rt_experimental", s->rt_hw_exp);
	// (Out of range: No.)
	const auto clamp_level = [](std::string& v, int max) {
		const int n = std::atoi(v.c_str());
		v           = n >= 0 && n <= max && !v.empty() && std::isdigit(static_cast<unsigned char>(v[0])) != 0 ? std::to_string(n) : "0";
	};
	clamp_level(s->rt_ss_refl, 6);
	clamp_level(s->rt_refl_sharp, 2);
	clamp_level(s->motes, 3);
	clamp_level(s->rt_refl_waves, 2);
	clamp_level(s->rt_ssgi, 4);
	clamp_level(s->rt_shadows, 3);
	clamp_level(s->ao, 3);
	if (s->smoke != "0") s->smoke = "1"; // (ds-pc-port10) default Suave (an old 2, Antorchas, is Suave)
	clamp_level(s->torch, 4);
	clamp_level(s->dlss5, 3);
	// (ds-pc-port9) The Modos Souls are back (Launcher-souls.ps1's): an unknown modo_souls is none.
	{
		bool known = false;
		for (int i = 0; i < kSoulsModeCount; i++) known = known || s->souls_mode == kSoulsModes[i].id;
		if (!known) s->souls_mode = "ninguno";
	}
	// (ds-pc-port9) Path tracing, Sombras RT and the 16 GB RAM / 8 GB GPU simulations are gone: always off.
	s->ao             = "0";
	s->rt_shadows     = "0";
	// (ds-pc-port10) Luz de antorchas RT (and its Hibrido, and the torch lens flare) is gone from the port: always off.
	s->torch          = "0";
	s->lens_flare     = false;
	// (ds-pc-port10) and the hardware RT Antorchas (torches, lamps, fire and magic lights) too.
	s->rt_lamps       = "0";
	s->rt_particles   = false;
	// (11-11) the 16 GB RAM / 8 GB GPU simulations are back as Opciones de desarrollador in the launcher.
	return true;
}

bool SaveSettings(const std::string& path, const Settings& s) {
	std::ostringstream o;
	const auto         b = [](bool x) { return x ? "1" : "0"; };
	o << "; Demon's Souls PC - ajustes. Se cambian en el lanzador (se abre al ejecutar Demon's Souls.exe). En el juego F2 abre el menu.\n"
	  << "; 1 = si, 0 = no. Lo que no se reconoce se ignora; borrar este archivo vuelve a los valores por defecto.\n\n"
	  << "[Juego]\n"
	  << "; Carpeta del juego volcado (la que tiene eboot.bin y sce_sys), o un .zar. Vacia: se pregunta al arrancar.\n"
	  << "carpeta_juego=" << s.game << "\n"
	  << "; Idioma de la consola: 3 espanol (Espana), 20 espanol (Latinoamerica), 1 ingles (EE. UU.), 18 ingles (R. U.)\n"
	  << "; Idioma del juego (consola PS5: 3 espanol, 20 espanol latino, 1 ingles EE. UU., 18 ingles RU, 2 frances, 4 aleman, 5 italiano, 7 portugues, 17 portugues Brasil, 0 japones, 11 chino simplificado)\n"
	  << "idioma_juego=" << s.language << "\n"
	  << "; 1: idioma_juego elegido en el lanzador; 0: el del juego sigue al de la interfaz\n"
	  << "idioma_juego_elegido=" << b(s.game_language_set) << "\n"
	  << "; Idioma del lanzador y de los textos del port: es, en, pt, fr, it, de (vacio: el de Windows)\n"
	  << "idioma_interfaz=" << s.menu_language << "\n"
	  << "; Copia de seguridad de _SaveData en _SaveData-copias antes de jugar\n"
	  << "copia_partidas=" << b(s.backup_saves) << "\n"
	  << "\n"
	  << "[Pantalla]\n"
	  << "; Iluminacion IBL: mitad, completa; Teselado: bajo, medio, alto, ultra, extremo (vacio: segun la salida, 4K completa + alto, si no mitad + medio)\n"
	  << "iluminacion_ibl=" << s.ibl << "\n"
	  << "teselado=" << s.tess << "\n"
	  << "pantalla_completa=" << b(s.fullscreen) << "\n"
	  << "; Tamano de la ventana (sin pantalla completa)\n"
	  << "ancho=" << s.width << "\nalto=" << s.height << "\n"
	  << "; Resolucion en pantalla completa: escritorio (la de Windows) o AnchoxAlto, p. ej. 1920x1080 (el monitor cambia a ella)\n"
	  << "resolucion=" << s.resolution << "\n"
	  << "; HDR10: solo si \"Usar HDR\" esta activado en la configuracion de pantalla de Windows\n"
	  << "hdr=" << b(s.hdr) << "\n"
	  << "vsync=" << b(s.vsync) << "\n"
	  << "; G-Sync / FreeSync: el reloj del juego no se ata a los Hz del monitor\n"
	  << "gsync=" << b(s.gsync) << "\n"
	  << "; Experimental: el juego pasa de 60 FPS reales (videos y menus siguen a 60)\n"
	  << "fps_desbloqueados=" << b(s.unlock_fps) << "\n"
	  << "; Tope de FPS con G-Sync y FPS desbloqueados sin frame generation (0 = automatico, justo bajo el refresco)\n"
	  << "tope_fps=" << Num(s.fps_cap) << "\n"
	  << "; Mantener 16:9 con bandas negras en vez de estirar\n"
	  << "ajustar_aspecto=" << b(s.aspect_fit) << "\n"
	  << "; Aspecto del 3D: 16:9, 21:9, 32:9 o 48:9 (ultrapanoramico; 48:9 = triple monitor)\n"
	  << "aspecto=" << s.aspect << "\n\n"
	  << "[Reescalado]\n"
	  << "; nativo, dlss-dlaa, dlss-ucalidad, dlss-calidad, dlss-equilibrado, dlss-rendimiento,\n"
	  << "; dlss-urendimiento, juego-1080, juego-900, juego-720,\n"
	  << "; xess-aa, xess-ucalidad-plus, xess-ucalidad, xess-calidad, xess-equilibrado, xess-rendimiento,\n"
	  << "; fsr-aa, fsr-ucalidad, fsr-calidad, fsr-equilibrado, fsr-rendimiento (FSR 3.1),\n"
	  << "; fsr4-aa, fsr4-calidad, fsr4-equilibrado, fsr4-rendimiento, fsr4-urendimiento, fsr4-dinamico (AMD FSR 4)\n"
	  << "reescalado=" << s.res << "\n"
	  << "; auto, 0, 0.25, 0.5, 0.75, 1\n"
	  << "nitidez=" << s.sharpness << "\n"
	  << "; auto, K (DLSS 4), L o M (DLSS 4.5)\n"
	  << "preset_dlss=" << s.dlss_preset << "\n"
	  << "; DLSS 5 (Neural Rendering de NVIDIA, solo NVIDIA RTX): 0 no, 1 suave, 2 medio, 3 fuerte (suave / medio / fuerte bajan la\n"
	  << "; resolucion en los bordes de la pantalla para ahorrar GPU; el centro siempre a resolucion completa). Necesita tu propio\n"
	  << "; nvngx_dlssnr.dll junto a \"Demon's Souls.exe\" (no viene incluido).\n"
	  << "dlss5=" << s.dlss5 << "\n"
	  << "; Archivo DLSS 5: 0 oficial (solo si esta firmado por NVIDIA), 1 no oficial (lo carga aunque no lo este; bajo tu responsabilidad)\n"
	  << "dlss5_no_oficial=" << b(s.dlss5_unsigned) << "\n"
	  << "; Intensidad DLSS 5: baja, media, alta (cuanto se nota el efecto de NVIDIA)\n"
	  << "dlss5_intensidad=" << s.dlss5_strength << "\n\n"
	  << "[FrameGeneration]\n"
	  << "; sin-fg, amd-x2, amd-x3, amd-x4, amd-120, amd-dinamico,\n"
	  << "; dlss-x2, dlss-x3, dlss-x4, dlss-x5, dlss-x6, dlss-120, dlss-dinamico (NVIDIA RTX)\n"
	  << "frame_generation=" << s.framegen << "\n"
	  << "\n"
	  << "[Rendimiento]\n"
	  << "; Optimizacion equipos 16 GB RAM: el perfil de memoria de 16 GB (6 GB fijos en RAM, la precarga espera con\n"
	  << "; menos de 1,5 GB libres). Solo se aplica en PCs de 20 GB de RAM o menos, o con simular_ram_16gb. No cambia los graficos.\n"
	  << "perfil_16gb=" << b(s.profile16) << "\n"
	  << "; Para graficas que se cuelgan o dan pantallazo azul (~37% menos FPS)\n"
	  << "gpu_seguro=" << b(s.safe_gpu) << "\n"
	  << "; Pruebas: el juego como en un PC de 16 GB de RAM / con una grafica de 8 GB\n"
	  << "simular_ram_16gb=" << b(s.simulate_ram16) << "\n"
	  << "simular_vram_8gb=" << b(s.simulate_vram8) << "\n"
	  << "; Ryzen con dos CCD: 0 = el juego en el CCD con mas cache L3 (mas rapido); 1 = todos los nucleos\n"
	  << "todos_los_ccd=" << b(s.all_ccds) << "\n"
	  << "; Carga de shaders al maximo: la preparacion de shaders del juego con todos los hilos de la CPU (1) o con pocos (0)\n"
	  << "carga_shaders_maxima=" << b(s.shader_turbo) << "\n"
	  << "; Precompilar shaders: todos = todos los hilos de la CPU (mas rapido, el PC va cargado); relajado; preguntar\n"
	  << "precompilar_hilos=" << s.precompile_threads << "\n"
	  << "; Anti-Lag con la frame generation de AMD: auto (Anti-Lag 2 en graficas AMD, el nuestro en las demas), amd2, normal,\n"
	  << "; ultra, no. En los modos Souls ligero, equilibrado y low profile va siempre el nuestro en Ultra.\n"
	  << "antilag_modo=" << s.antilag_mode << "\n"
	  << "; Con la frame generation de NVIDIA: reflex (recomendado), normal o ultra (el nuestro, con Reflex apagado)\n"
	  << "antilag_dlssg=" << s.antilag_dlssg << "\n\n"
	  << "[Graficos]\n"
	  << "; Modos Souls: ninguno; low (PC low profile: 900p con FSR, todo en bajo, 30 reales x2 AMD); ligero (graficas de 8 GB: 30 reales x2 AMD, ligero + 40 reales x2 AMD, para monitores de mas de 60 Hz, FSR Calidad, texturas y sombras medio);\n"
	  << "; equilibrado (45 reales x2 AMD, texturas, distancia y sombras alto); ultra (RTX 5060-5070 Ti, RX 9070);\n"
	  << "; extremo (RTX 5080 / 5090: todo al maximo). Los FPS dependen de tu GPU y tu CPU.\n"
	  << "; Ligero, equilibrado y low mandan sobre frame_generation (AMD x2 fija; el reescalado DLSS si se puede usar).\n"
	  << "modo_souls=" << s.souls_mode << "\n"
	  << "; Optimizacion tarjetas 8 GB: en el menu pone texturas en bajo y distancia y sombras en medio\n"
	  << "optimizacion_8gb=" << b(s.gpu8) << "\n"
	  << "; Texturas: 0 (Maximo), h (Alto), 1 (Medio, mitad), 2 (Bajo, un cuarto)\n"
	  << "reducir_texturas=" << s.texture_reduce << "\n"
	  << "; Distancia de dibujado: standard (Maximo, el juego tal cual), high (Alto), low (Medio), lowest (Bajo)\n"
	  << "calidad=" << s.quality << "\n"
	  << "; Sombras: maximo (las del juego), alto (sol hasta 190 m), medio (125 m), bajo (60 m, sin focos ni ambientales)\n"
	  << "sombras=" << s.shadows << "\n"
	  << "; Detalle del juego: ultra, muy-alto (cine), alto (rendimiento de PS5), normal, bajo (teselado, cascadas, luz)\n"
	  << "detalle=" << s.detail << "\n"
	  << "; Reflejos del juego (SSR): ultra, alto (PS5), normal, bajo\n"
	  << "reflejos=" << s.reflections << "\n"
	  << "; Distancia de los objetos (fundido a lo lejos): ultra, alto (PS5), normal, bajo\n"
	  << "distancia_objetos=" << s.object_fade << "\n"
	  << "; Nitidez del antialiasing del juego: ultra, alto, normal (PS5), bajo, no\n"
	  << "nitidez_aa=" << s.aa_sharp << "\n"
	  << "; Bordes sombras (filtrado de las sombras del juego): 0 normal, 1 alta, 2 muy alta, 3 ultra\n"
	  << "bordes_sombras=" << s.shadow_edges << "\n"
	  << "; Sombra rincones (fuerza de la oclusion de ray tracing; necesita rt_oclusion=1): 0 normal, 1 alta, 2 maxima\n"
	  << "sombra_rincones=" << s.corner_shadow << "\n"
	  << "; Postprocesado del juego (1 = como en PS5, 0 = quitado)\n"
	  << "desenfoque_movimiento=" << b(s.motion_blur) << "\n"
	  << "profundidad_campo=" << b(s.dof) << "\n"
	  << "vineteado=" << b(s.vignette) << "\n"
	  << "resplandor=" << b(s.bloom) << "\n"
	  << "adaptacion_ojo=" << b(s.eye_adapt) << "\n"
	  << "; Filtrado anisotropico x16 (solo superficies de la escena)\n"
	  << "filtrado_x16=" << b(s.af16) << "\n"
	  << "; Texturas: 0 normales, 1 nitidas (con reescalado, el detalle de la resolucion final), 2 extra nitidas\n"
	  << "texturas_nitidas=" << s.texture_sharpness << "\n\n"
	  << "[RayTracing]\n"
	  << "; Efectos de pantalla (sin RT por hardware; solo con DLSS, XeSS o FSR):\n"
	  << "; Reflejos RT: 0 no, 1 bajo, 2 medio, 3 alto, 4 ultra, 5 extremo, 6 hibrido (alto + relleno por hardware; GPU con RT)\n"
	  << "reflejos_rt=" << s.rt_ss_refl << "\n"
	  << "; Nitidez de reflejos RT: 0 normal (de serie), 1 alta, 2 maxima (mas nitidos, cuesta mas GPU)\n"
	  << "nitidez_reflejos_rt=" << s.rt_refl_sharp << "\n"
	  << "; Ondas del agua en reflejos: 0 como ahora, 1 suaves, 2 espejo (solo el rayo del reflejo; el agua se ve igual)\n"
	  << "ondas_agua_reflejos=" << s.rt_refl_waves << "\n"
	  << "; Luz RT (luz rebotada en pantalla): 0 no, 1 media, 2 alta, 3 ultra, 4 extremo\n"
	  << "luz_rt_pantalla=" << s.rt_ssgi << "\n"
	  << "; Sombras RT (sombras del sol de pantalla): 0 no, 1 bajo, 2 medio, 3 alto\n"
	  << "sombras_rt=" << s.rt_shadows << "\n"
	  << "; Path tracing (oclusion y luz indirecta por RT por hardware; sin RT por hardware, oclusion de pantalla): 0 no, 1 bajo, 2 medio, 3 alto\n"
	  << "oclusion_ao=" << s.ao << "\n"
	  << "; Humo y particulas: 0 no, 1 suave (sin cortes duros)\n"
	  << "humo_particulas=" << s.smoke << "\n"
	  << "; RT por hardware (experimental): 1 activa las filas de trazado de rayos por hardware de abajo\n"
	  << "rt_experimental=" << b(s.rt_hw_exp) << "\n"
	  << "; Trazado de rayos por hardware (1 si, 0 no; solo con rt_experimental=1). Elegir un modo Souls los pone todos en 0.\n"
	  << "rt_reflejos=" << b(s.rt_reflections) << "\n"
	  << "rt_sombras_sol=" << b(s.rt_sun) << "\n"
	  << "rt_niebla=" << b(s.rt_fog) << "\n"
	  << "rt_particulas=" << b(s.rt_particles) << "\n"
	  << "rt_metal=" << b(s.rt_metal) << "\n"
	  << "rt_oclusion=" << b(s.rt_ao) << "\n"
	  << "; Luz volumetrica: 0 no, 1 bajo, 2 medio, 3 alto (excluye rt_sombras_sol)\n"
	  << "luz_volumetrica=" << s.godrays << "\n"
	  << "; Particulas volumetricas (motas de polvo iluminadas por la luz volumetrica): 0 no, 1 sutil, 2 media, 3 densa\n"
	  << "particulas_volumetricas=" << s.motes << "\n"
	  << "; Juego original: 1 apaga todos los efectos visuales que anade la version de PC\n"
	  << "juego_original=" << b(s.pure_game) << "\n\n"
	  << "[Mandos]\n"
	  << "; Teclado y raton (F7 tambien cambia el raton a mando en vivo)\n"
	  << "teclado_raton=" << b(s.kbm) << "\n"
	  << "raton_sensibilidad=" << Num(s.kbm_sens) << "\n"
	  << "raton_antizona=" << Num(s.kbm_antidz) << "\n"
	  << "; Teclas: Cross=F;R1=Mouse:Left;... (vacio = las de serie, estilo Souls)\n"
	  << "mapa_teclado=" << s.kbm_map << "\n"
	  << "; Mando remapeado: a=Cross;b=Circle;... (boton fisico de SDL = control de PS5; vacio = normal)\n"
	  << "mapa_mando=" << s.pad_map << "\n"
	  << "; Ajustes del juego (los del lanzador): gs_sun=3;gs_spot=1;... (gs_x=0 = como viene el juego; sin la clave = su predeterminado, gs_spot apagadas)\n"
	  << "ajustes_juego=" << s.game_settings << "\n"
	  << "; Forzar la velocidad de camara del juego (0 = la del menu del juego; 5-15 tambien cambia la del mando)\n"
	  << "raton_velocidad_camara_forzada=" << s.kbm_camera << "\n"
	  << "; Camara instantanea (frame warp): la imagen gira con el raton antes de que el juego dibuje\n"
	  << "raton_camara_instantanea=" << b(s.kbm_warp) << "\n"
	  << "; 1: el proximo arranque calibra la camara del raton (kbm_curve.txt) y vuelve a 0\n"
	  << "raton_calibrar=" << b(s.kbm_calibrate) << "\n\n"
	  << "[Configurador]\n"
	  << "; 0: el configurador de hardware se muestra al arrancar\n"
	  << "configurado=" << b(s.configured) << "\n"
	  << "; Mejoras para equipos de 16 GB de RAM: perfil de 16 GB (solo en PCs de 20 GB de RAM o menos)\n"
	  << "ahorro_memoria_v3=" << b(s.ram16_boost) << "\n"
	  << "; Opciones de rendimiento (0 = como siempre; 1 = la mejora experimental)\n"
	  << "; Registros PM4 directos (KYTY_PM4_REGFILE)\n"
	  << "rend_registros_pm4=" << b(s.perf_pm4_regfile) << "\n"
	  << "; Subidas a la GPU en dos hilos (KYTY_UPLOAD_LANES=2)\n"
	  << "rend_subidas_dos_hilos=" << b(s.perf_upload_lanes) << "\n"
	  << "; El hilo de render ayuda con las subidas si la CPU va cargada (KYTY_UPLOAD_HELP=2)\n"
	  << "rend_ayuda_subidas=" << b(s.perf_upload_help) << "\n"
	  << "; Los hilos de ayuda dejan de esperar activos si la CPU va cargada (KYTY_SPIN_GATE=1)\n"
	  << "rend_espera_hilos=" << b(s.perf_spin_gate) << "\n"
	  << "; Copias de la GPU devueltas antes de que el juego las pida (KYTY_EAGER_FEEDBACK)\n"
	  << "rend_feedback_anticipado=" << b(s.perf_eager_feedback) << "\n"
	  << "; Copias de la GPU agrupadas (KYTY_FEEDBACK_BATCH)\n"
	  << "rend_feedback_agrupado=" << b(s.perf_feedback_batch) << "\n"
	  << "; Shaders lane-local ampliados (KYTY_LANE_LOCAL_WIDE; recompila parte de los shaders)\n"
	  << "rend_shaders_lane_local=" << b(s.perf_lane_wide) << "\n\n"
	  << "[Interfaz]\n"
	  << "; Metricas en pantalla: 0 no, 1 FPS, 2 FPS + detalle\n"
	  << "metricas=" << s.metrics << "\n"
	  << "; Lanzador al iniciar: 1 si, 0 el juego arranca directo (F2 > Sistema lo cambia; \"Demon's Souls.exe\" --lanzador lo muestra una vez)\n"
	  << "menu_inicio=" << b(s.start_menu) << "\n"
	  << "; Niebla y ceniza animadas de fondo en el lanzador (0 = la ilustracion fija)\n"
	  << "lanzador_ambiente=" << b(s.launcher_ambience) << "\n"
	  << "; Saltar el video de PlayStation y la cinematica de inicio (1 si, 0 no)\n"
	  << "saltar_intro=" << b(s.skip_intro) << "\n"
	  << "; Vibracion del mando (el audio haptico del DualSense como vibracion; mandos Xbox, DualSense por Bluetooth)\n"
	  << "vibracion=" << b(s.rumble) << "\n"
	  << "vibracion_fuerza=" << Num(s.rumble_gain) << "\n\n"
	  << "[Avanzado]\n"
	  << "; Interruptores avanzados: KYTY_X=1;KYTY_Y=2 (KYTY_X= lo quita). Solo para pruebas.\n"
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


// (ds-pc-port37) Launcher-souls.ps1's $gameSettings, as it is (names, levels, convars and tips).
const GameSetting kGameSettings[] = {
	{"gs_sun", "Sombras", false, "Sombras del sol", {"r_virtualGeometryNumShadowCascades=1;sunShadowMaxDistance=60;sunShadowCascadeDistance0=60;sunShadowLightDistance=1000", "r_virtualGeometryNumShadowCascades=2;sunShadowMaxDistance=125;sunShadowCascadeDistance0=22;sunShadowCascadeDistance1=125;sunShadowLightDistance=1000", "r_virtualGeometryNumShadowCascades=2;sunShadowMaxDistance=250;sunShadowCascadeDistance0=40;sunShadowCascadeDistance1=250;sunShadowLightDistance=1000", "r_virtualGeometryNumShadowCascades=4;sunShadowMaxDistance=400;sunShadowCascadeDistance0=27;sunShadowCascadeDistance1=61;sunShadowCascadeDistance2=131;sunShadowCascadeDistance3=400;sunShadowLightDistance=1000", "r_virtualGeometryNumShadowCascades=4;sunShadowMaxDistance=1000;sunShadowCascadeDistance0=65;sunShadowCascadeDistance1=142;sunShadowCascadeDistance2=300;sunShadowCascadeDistance3=1000;sunShadowLightDistance=1000"},
	 "Sombras del sol del juego: hasta donde llegan y en cuantas cascadas se reparten. Bajo 60 m (1 cascada), Medio 125 m (2), Alto 250 m (2, el valor de fabrica), Ultra 400 m (4), Extremo 1000 m (4, el maximo del juego). Cuesta GPU y CPU (las sombras son unos 2600 dibujados por frame); Extremo puede costar un 10-15 % de FPS. Manda sobre el desplegable Sombras."},
	{"gs_spot", "Sombras", true, "Sombras de focos (antorchas)", {"r_enableSpotShadows=false", "r_enableSpotShadows=true", nullptr, nullptr, nullptr},
	 "Sombras que proyectan antorchas, hogueras y otras luces (unas 7 por escena en 1-1). No: mas FPS, pero sin esas sombras. El juego no tiene ajuste de distancia ni de resolucion para ellas.", 1},
	{"gs_contact", "Sombras", false, "Sombras de contacto", {"r_contactShadowsEnabled=false", "r_contactShadowsEnabled=true;r_contactShadowsDistance=0.03", "r_contactShadowsEnabled=true;r_contactShadowsDistance=0.06", "r_contactShadowsEnabled=true;r_contactShadowsDistance=0.1", "r_contactShadowsEnabled=true;r_contactShadowsDistance=0.3;r_contactShadowsThickness=0.12;r_contactShadowsSoftness=0.2"},
	 "Sombras de contacto del juego: sombras finas donde los objetos tocan el suelo y las paredes. Bajo: apagadas. Medio, Alto (de fabrica), Ultra y Extremo: alcance 0,03 / 0,06 / 0,10 / 0,30 (Extremo tambien mas gruesas y mas definidas). Cuesta poco; con mucho alcance pueden salir halos."},
	{"gs_ambient", "Iluminacion", false, "Sombreado ambiental del juego", {"r_enableAmbientShadowing=false", "r_enableAmbientShadowing=true;r_ambientShadowCutoffDistance=50;r_ambientShadowCutoffFadeDistance=10", "r_enableAmbientShadowing=true;r_ambientShadowCutoffDistance=100;r_ambientShadowCutoffFadeDistance=20", "r_enableAmbientShadowing=true;r_ambientShadowCutoffDistance=250;r_ambientShadowCutoffFadeDistance=40", "r_enableAmbientShadowing=true;r_ambientShadowCutoffDistance=1000;r_ambientShadowCutoffFadeDistance=100"},
	 "Sombreado ambiental del juego: oscurece bajo personajes y objetos con la luz del entorno. Bajo: apagado (mas FPS). Medio 50 m, Alto 100 m (de fabrica), Ultra 250 m, Extremo 1000 m (el maximo): hasta donde se dibuja. Mas distancia = mas GPU en zonas abiertas."},
	{"gs_gi", "Iluminacion", false, "Iluminacion global", {"GIEnableMultiBounce=false;GIUpdateNumIteraions=1;r_iblResolution=HalfResX", "GIEnableMultiBounce=true;GIUpdateNumIteraions=1;r_iblResolution=HalfResX", "GIEnableMultiBounce=true;GIUpdateNumIteraions=1;r_iblResolution=Full", "GIEnableMultiBounce=true;GIUpdateNumIteraions=2;GILightCutoffIntensity=0.0005;r_iblResolution=Full", "GIEnableMultiBounce=true;GIUpdateNumIteraions=4;GILightCutoffIntensity=0.0001;GILightMinSubtendedAngle=3;r_iblResolution=Full"},
	 "Iluminacion global del juego (luz que rebota). Bajo: sin rebote multiple y reflejos de entorno a media resolucion. Medio: como el modo rendimiento de PS5. Alto: reflejos de entorno completos (modo cine). Ultra y Extremo: 2 o 4 pasadas de actualizacion por frame y cuentan luces mas debiles y pequenas. Cuesta GPU (Extremo bastante). El modo del juego puede volver a fijar los reflejos de entorno al cargar."},
	{"gs_dist", "Distancia", false, "Distancia de dibujado", {"r_forceViewportHeightForPixelSizeCalc=480;r_fadeOutInstanceOverDistance=6", "r_forceViewportHeightForPixelSizeCalc=720;r_fadeOutInstanceOverDistance=10", "r_forceViewportHeightForPixelSizeCalc=1080;r_fadeOutInstanceOverDistance=15", "r_forceViewportHeightForPixelSizeCalc=1440;r_fadeOutInstanceOverDistance=35", "r_forceViewportHeightForPixelSizeCalc=2160;r_fadeOutInstanceOverDistance=60"},
	 "Distancia de dibujado: el detalle de los modelos (LOD) se elige como para una imagen de 480 / 720 / 1080 (de fabrica) / 1440 / 2160 lineas, y el fundido de los objetos lejanos pasa a 6 / 10 / 15 / 35 / 60. No cambia la resolucion. Bajo y Medio ayudan si va justo el procesador; Ultra y Extremo cuestan CPU y GPU. Manda sobre Distancia y Object Fade."},
	{"gs_ssr", "Efectos", false, "Reflejos del juego (SSR)", {"r_ssrEnableOverrides=true;r_ssrMaxIterations=32;r_ssrMaxRayLength=300;r_ssrCutoffDistance=250", "r_ssrEnableOverrides=true;r_ssrMaxIterations=64;r_ssrMaxRayLength=600;r_ssrCutoffDistance=500", "r_ssrEnableOverrides=true;r_ssrMaxIterations=128;r_ssrMaxRayLength=1000;r_ssrCutoffDistance=1000", "r_ssrEnableOverrides=true;r_ssrMaxIterations=192;r_ssrMaxRayLength=1400;r_ssrCutoffDistance=1500", "r_ssrEnableOverrides=true;r_ssrMaxIterations=256;r_ssrMaxRayLength=1600;r_ssrCutoffDistance=5000"},
	 "Reflejos en pantalla del juego (agua, charcos, suelos mojados). Pasos / largo del rayo / distancia: Bajo 32 / 300 / 250, Medio 64 / 600 / 500, Alto 128 / 1000 / 1000 (de fabrica), Ultra 192 / 1400 / 1500, Extremo 256 / 1600 / 5000 (el maximo). Cuesta GPU, sobre todo con lluvia. Manda sobre Reflections."},
	{"gs_tess", "Efectos", false, "Teselado / relieve", {"r_tessellationDensity=4.0;r_tessellationSmallestEdgeLength=0.08;r_tessellationFadeOutStart=6.0;r_tessellationFadeOutRange=2.0", "r_tessellationDensity=8.0;r_tessellationSmallestEdgeLength=0.04;r_tessellationFadeOutStart=12.0;r_tessellationFadeOutRange=3.0", "r_tessellationDensity=18.0;r_tessellationSmallestEdgeLength=0.02;r_tessellationFadeOutStart=15.0;r_tessellationFadeOutRange=3.0", "r_tessellationDensity=20.0;r_tessellationSmallestEdgeLength=0.015;r_tessellationFadeOutStart=25.0;r_tessellationFadeOutRange=5.0", "r_tessellationDensity=20.0;r_tessellationSmallestEdgeLength=0.01;r_tessellationFadeOutStart=40.0;r_tessellationFadeOutRange=8.0"},
	 "Teselado: relieve de verdad en suelos de piedra, barro y paredes. Bajo: menos relieve y solo de cerca. Medio: modo rendimiento de PS5. Alto: modo cine de PS5. Ultra y Extremo: mas denso y hasta 25 / 40 m. Cuesta GPU. Ojo: el juego vuelve a poner el de su modo al cargar su configuracion; si no ves cambio, es eso."},
	{"gs_fog", "Efectos", true, "Niebla volumetrica", {"fog_enable=false", "fog_enable=true", nullptr, nullptr, nullptr},
	 "Niebla volumetrica: la bruma iluminada por el sol y las antorchas. No: unos 4-5 % mas FPS, pero cambia mucho el aspecto del juego."},
	{"gs_fogden", "Efectos", false, "Densidad de niebla", {"fog_density=0.6", "fog_density=1.0", "fog_density=1.5", "fog_density=2.5", "fog_density=4.0"},
	 "Cuanta niebla hay en el aire (fog_density del juego, 1.0 = como viene). Bajo 0.6, Medio 1.0, Alto 1.5, Ultra 2.5, Extremo 4.0. Mas niebla = mas ambiente y rayos de luz mas visibles. Casi no cuesta."},
	{"gs_fogpart", "Efectos", true, "Niebla con particulas", {"fog_enableParticles=false", "fog_enableParticles=true", nullptr, nullptr, nullptr},
	 "Particulas dentro de la niebla volumetrica (motas y polvo en el aire, sobre todo con luz). Si: mas ambiente, cuesta poco. Necesita la niebla volumetrica activa."},
	{"gs_decal", "Efectos", true, "Calcomanias (sangre, suciedad)", {"r_projectedDecalEnable=false", "r_projectedDecalEnable=true", nullptr, nullptr, nullptr},
	 "Calcomanias proyectadas: sangre, manchas, suciedad y marcas en el suelo y las paredes. No: casi la mitad de dibujados menos en 1-1 y bastantes mas FPS, pero se pierde mucho detalle."},
	{"gs_mblur", "Postprocesado", true, "Desenfoque de movimiento", {"r_motionBlurEnable=false", "r_motionBlurEnable=true", nullptr, nullptr, nullptr},
	 "Desenfoque de movimiento del juego al girar la camara. No: imagen mas nitida en movimiento. Casi no cuesta. Manda sobre la casilla Motion Blur."},
	{"gs_dof", "Postprocesado", true, "Profundidad de campo", {"r_dofEnable=false", "r_dofEnable=true", nullptr, nullptr, nullptr},
	 "Profundidad de campo: desenfoque del fondo en cinematicas y menus. No: todo enfocado; ahorra un poco. Manda sobre la casilla Depth of Field."},
	{"gs_vignette", "Postprocesado", true, "Vineta (bordes oscuros)", {"r_vignetteEnable=false", "r_vignetteEnable=true", nullptr, nullptr, nullptr},
	 "Vineta: oscurecimiento de las esquinas de la imagen. No cuesta FPS. Manda sobre la casilla Vignette."},
	{"gs_bloom", "Postprocesado", true, "Bloom (resplandor)", {"r_bloomEnable=false", "r_bloomEnable=true", nullptr, nullptr, nullptr},
	 "Bloom: resplandor alrededor de las luces y las zonas brillantes. Cuesta muy poco. Manda sobre la casilla Bloom."},
	{"gs_eyeadapt", "Postprocesado", true, "Adaptacion de la vista", {"r_exposureEyeAdaptEnable=false", "r_exposureEyeAdaptEnable=true", nullptr, nullptr, nullptr},
	 "Adaptacion de la vista: la exposicion cambia poco a poco al pasar de zonas oscuras a claras. No: exposicion fija. No cuesta FPS. Manda sobre la casilla Eye Adaptation."},
	{"gs_aasharp", "Postprocesado", false, "Nitidez del antialiasing", {"r_aaSharpenStrengthStatic=1.2;r_aaSharpenStrengthMotion=1.2;r_aaHistorySharpeningStrength=0.5", "r_aaSharpenStrengthStatic=2.0;r_aaSharpenStrengthMotion=2.0;r_aaHistorySharpeningStrength=0.7", "r_aaSharpenStrengthStatic=2.5;r_aaSharpenStrengthMotion=2.5;r_aaHistorySharpeningStrength=0.8", "r_aaSharpenStrengthStatic=3.5;r_aaSharpenStrengthMotion=3.5;r_aaHistorySharpeningStrength=0.9", "r_aaSharpenStrengthStatic=5.0;r_aaSharpenStrengthMotion=5.0;r_aaHistorySharpeningStrength=1.0"},
	 "Nitidez del antialiasing temporal del juego: Bajo 1,2, Medio 2,0, Alto 2,5 (de fabrica), Ultra 3,5, Extremo 5,0 (la del historial de 0,5 a 1,0). No cuesta FPS; mucha puede marcar bordes y brillos. Manda sobre AA Sharpness."},
};

const GameSetting* GameSettings(size_t* count) {
	*count = std::size(kGameSettings);
	return kGameSettings;
}

int GameSettingLevel(const Settings& s, const char* id) {
	const std::string key = std::string(id) + "=";
	size_t at = 0;
	while (at <= s.game_settings.size()) {
		const size_t end  = std::min(s.game_settings.find(';', at), s.game_settings.size());
		const auto   item = s.game_settings.substr(at, end - at);
		at                = end + 1;
		if (item.rfind(key, 0) == 0) return std::clamp(std::atoi(item.c_str() + key.size()), 0, 5);
	}
	// Absent from ajustes_juego: the entry's own default (gs_spot = off). An explicit "=0" is Predeterminado (the game's own).
	for (const auto& g: kGameSettings)
		if (std::strcmp(g.id, id) == 0) return g.def;
	return 0;
}

void SetGameSettingLevel(Settings* s, const char* id, int level) {
	std::string out;
	for (const auto& g: kGameSettings) {
		const int v = std::strcmp(g.id, id) == 0 ? level : GameSettingLevel(*s, g.id);
		// Entries whose default is not 0 are always written, so that "Predeterminado" (0) survives a reload.
		if (v > 0 || g.def != 0) out += (out.empty() ? "" : ";") + std::string(g.id) + "=" + std::to_string(v);
	}
	s->game_settings = out;
}

void ResetGameSettings(Settings* s) {
	std::string out;
	for (const auto& g: kGameSettings)
		if (g.def != 0) out += (out.empty() ? "" : ";") + std::string(g.id) + "=0";
	s->game_settings = out;
}

std::string GameSettingsConvars(const Settings& s) {
	std::string out;
	for (const auto& g: kGameSettings) {
		const int v = GameSettingLevel(s, g.id);
		if (v > 0 && v <= 5 && g.levels[v - 1] != nullptr) out += (out.empty() ? "" : ";") + std::string(g.levels[v - 1]);
	}
	return out;
}

std::string ShadowFilterLevel(const Settings& s) {
	return s.shadow_edges == "1" || s.shadow_edges == "2" || s.shadow_edges == "3" ? s.shadow_edges : std::string();
}

std::string DriverCacheKey(const Settings& s) {
	std::string key = "1f3c29b70e536c8f5a5e812e793b33ac2eb6858b4810ed98ca2a51b9fa5eed97";
	if (const auto level = ShadowFilterLevel(s); !level.empty()) key.replace(key.size() - 2, 2, "f" + level);
	return key;
}

bool LowRam(const Settings& s, const PcInfo& pc) {
	// (10-28) "Mejoras para equipos de 16 GB de RAM" (on by default) on a PC of 20 GB or less, or simulated.
	return s.simulate_ram16 || s.ram16_boost; // (port28) usable on any PC (off by default; Optimizar turns it on at 20 GB or less)
}

void RecommendFor(PcInfo* pc) {
	// Launcher-souls.ps1 $gpuDb (10-23): the recommended mode and what to expect (measured on an RTX 5080 and a simulated
	// 8 GB card; the rest an ESTIMATE by relative power: the emulator depends a lot on the CPU too).
	struct Row {
		std::vector<const char*> any;
		const char*              mode;
		const char*              expected;
	};
	static const Row rows[] = {
	    {{"GTX 9", "GTX 10", "GTX TITAN", "RX 4", "RX 5", "VEGA", "RADEON VII"}, "NO COMPATIBLE",
	     "tu grafica no tiene fragmentShaderBarycentric (hace falta NVIDIA RTX 20 / GTX 16 o AMD RX 6000 o mas nuevas)"},
	    {{"RTX 509", "RTX 508", "RTX 409", "RTX 408", "RX 7900 XTX"}, "Modo Souls extremo",
	     "4K cine: ~40-50 FPS reales (150-224 en pantalla con FG). 1440p: 60-75 reales"},
	    {{"RTX 507", "RTX 4070 TI", "RTX 4070 SUPER", "RTX 309", "RTX 3080 TI", "RX 907", "RX 7900 XT", "RX 7900 GRE", "RX 6950"},
	     "Modo Souls ultra", "1440p: ~50-65 FPS reales (150-224 en pantalla con FG)"},
	    {{"RTX 4070", "RTX 4060 TI 16", "RTX 3080", "RTX 506", "RX 6800", "RX 6900", "RX 7800", "RX 7700", "B580", "B570"},
	     "Modo Souls equilibrado", "1440p: 45 FPS reales fijos (90 en pantalla con AMD x2)"},
	    {{"RTX 4060", "RTX 3070", "RTX 3060", "RTX 207", "RTX 208", "RTX 2060", "RX 66", "RX 7600", "RX 6700", "ARC A7"},
	     "Modo Souls ligero", "1080p: 30 FPS reales fijos (60 en pantalla con AMD x2); 8 GB de VRAM justos"},
	    {{"GTX", "RX 6500", "RX 6400", "ARC A3", "ARC A5", "RADEON(TM) GRAPHICS", "IRIS", "UHD"}, "Modo Souls PC low profile",
	     "900p: 30 FPS reales fijos (60 con AMD x2); puede no ir fluido"},
	};
	std::string up = pc->gpu;
	std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
	pc->recommended.clear();
	pc->expected.clear();
	pc->unsupported = false;
	for (const auto& row: rows) {
		bool hit = false;
		for (const char* k: row.any) {
			const auto at = up.find(k);
			if (at == std::string::npos) continue;
			// "RX 4" / "RX 5" / "GTX 9": the old generations only (RX 4xx / 5xx, RX 5x00), not RX 6000+ that also start so.
			const std::string key = k;
			if ((key == "RX 4" || key == "RX 5") && at + key.size() < up.size()) {
				size_t digits = 0;
				for (size_t i = at + key.size() - 1; i < up.size() && std::isdigit(static_cast<unsigned char>(up[i])) != 0; i++) digits++;
				if (digits != 3 && digits != 4) continue;
				if (digits == 4 && key == "RX 4") continue;
			}
			hit = true;
			break;
		}
		if (!hit) continue;
		pc->recommended = row.mode;
		pc->expected    = row.expected;
		pc->unsupported = std::strcmp(row.mode, "NO COMPATIBLE") == 0;
		return;
	}
}

LaunchPlan BuildLaunch(const Settings& s, const std::string& title_id, const PcInfo& pc) {
	LaunchPlan plan;
	Env        env;
	const bool demons = title_id.empty() || title_id.rfind("PPSA0134", 0) == 0;
	// The frame generation config (config/<mode>.json): the common switches, then the mode's.
	// Souls ligero / equilibrado / PC low profile: AMD x2 whatever frame_generation says, the game flipped at 30 / 45 / 30.
	// (10-23) No DLSS on a graphics card that is not NVIDIA (the launcher does not offer it).
	const bool  fixed_fg = FixedFg(s);
	std::string fg_id    = fixed_fg ? std::string("amd-x2") : s.framegen; // (11-14) ligero + is 40x2 now
	if (!pc.nvidia && fg_id.rfind("dlss", 0) == 0) {
		fg_id = "sin-fg";
		plan.notes.emplace_back("DLSS frame generation needs NVIDIA: without frame generation");
	}
	const auto* fg = FindFg(fg_id);
	for (const auto& p: Tables::kBaseEnv) env.Set(p.key, p.value);
	for (int i = 0; i < fg->env_count && fg->env[i].key != nullptr; i++) env.Set(fg->env[i].key, fg->env[i].value);
	const bool with_fg  = std::strcmp(fg->id, "sin-fg") != 0;
	const bool amd_fg   = std::strncmp(fg->id, "amd", 3) == 0;
	const bool dlss_fg  = std::strncmp(fg->id, "dlss", 4) == 0;
	// (10-22) Unlocked always (the launcher has no box any more), but in the fixed Souls modes.
	const bool unlocked = !fixed_fg;
	const bool low_ram  = LowRam(s, pc);

	// Super resolution. (10-23) DLSS upscaling with AMD frame generation works: allowed, in the Souls modes too.
	const ResMode* res      = FindRes(s.res);
	if (!pc.nvidia && std::strcmp(res->upscaler, "dlss") == 0) {
		// (DLSS needs NVIDIA: FSR at the same quality.)
		std::string other = std::string("fsr-") + (std::strcmp(res->id, "dlss-dlaa") == 0 ? "aa" : res->id + 5);
		if (other == "fsr-urendimiento") other = "fsr-rendimiento"; // (FSR 3.1 has no Ultra rendimiento)
		res = FindRes(other);
		if (std::strcmp(res->id, other.c_str()) != 0) res = FindRes("fsr-calidad");
		plan.notes.emplace_back(std::string("DLSS needs NVIDIA: ") + res->label);
	}
	std::string upscaler = res->upscaler;
	double      ratio    = upscaler.empty() ? 0.0 : res->value;
	// (10-27) DLSS in Souls ligero / PC low profile: DLSS outputs 4K inside and only takes 2160 / an exact ratio (1440, 1256,
	// 1080, 720): any DLSS quality there renders at 720 (DLAA at 1080 stays).
	if (upscaler == "dlss" && pc.nvidia && (s.souls_mode == "ligero" || s.souls_mode == "ligero-plus" || s.souls_mode == "low")) {
		int mode_w = 0, mode_h = 0;
		const bool mode = s.fullscreen && std::sscanf(s.resolution.c_str(), "%dx%d", &mode_w, &mode_h) == 2 && mode_h > 0;
		const int out_h = mode ? mode_h : s.fullscreen && pc.screen_h > 0 ? pc.screen_h : (s.height > 0 ? s.height : 1440);
		if (!(out_h == 1080 && ratio == 1.0)) ratio = std::round(out_h / 720.0 * 10000.0) / 10000.0;
	}
	// (ds-pc-port12, the emulator launcher's Resolve-Res of res-1029) The chosen resolution is the game's output, the upscaler's
	// and the swapchain's; the 3D = output / the upscaler's ratio. DLSS takes the 3D between half the output and the output, or
	// exactly a third (ultra performance), at least 360 lines (720 at 4K): else the nearest valid one below.
	const int res_out_h = OutputHeight(s, pc);
	if (upscaler == "dlss" && pc.nvidia && ratio > 0) {
		const int  out_h = res_out_h;
		const int  least = out_h < 2160 ? 360 : 720;
		const int  in_h  = static_cast<int>(std::lround(out_h / ratio));
		const int  half  = (out_h + 1) / 2;
		const int  third = static_cast<int>(std::lround(out_h / 3.0));
		const bool ok    = (in_h >= half && in_h <= out_h && in_h >= least) || (in_h == third && in_h >= least);
		if (!ok) {
			const int pick = (least <= third) ? third : (least >= half ? least : half);
			ratio          = std::round(out_h / static_cast<double>(pick) * 10000.0) / 10000.0;
			plan.notes.emplace_back("DLSS: the 3D at " + std::to_string(pick) + " lines (" + std::to_string(in_h) + " is not a DLSS height)");
		}
	}
	if (std::strcmp(res->id, "fsr-900") == 0 && pc.screen_h > 0) ratio = std::max(1.0, std::round(pc.screen_h / 900.0 * 10000.0) / 10000.0);

	// Launcher-souls.ps1 Start-Game's -Set switches, in its order.
	EnvList sets;
	if (low_ram) sets.emplace_back("KYTY_LOWRAM", "1"); // (10-24) the emulator's 16 GB limits
	if (s.simulate_ram16) sets.emplace_back("KYTY_SIMULATE_RAM_MB", "16384");
	// (11-11) Souls ligero +: the pad read 4 ms later than ligero's (39 ms before its 30 fps flip slot, the game's ~7 ms
	// of CPU plus margin): about 4 ms less input latency.
	if (fixed_fg)
		sets.emplace_back("KYTY_PAD_LATE_MS", s.souls_mode == "equilibrado" ? "29" : s.souls_mode == "ligero-plus" ? "33" : "43"); // (11-12) ligero 30x2 (43), (11-14) ligero + 40x2 (33: ~1.3 x its 25 ms slot, as 594353ceb) // (10-25) Anti-Lag+
	// (des-particles-fg) Souls ligero (30x2) and ligero + (30x3) only: the particle layer's motion vectors for AMD frame
	// generation (particle-mv.cpp): fire, sparks, rain, mist and spells interpolated instead of shown at the real 30.
	// (11-11) Off by default until verified in game (the first build lost the device): KYTY_FG_PARTICLE_MV=1 by hand.
	// if (fixed_fg && (s.souls_mode == "ligero" || s.souls_mode == "ligero-plus")) sets.emplace_back("KYTY_FG_PARTICLE_MV", "1");
	// (10-26) The RAM flush is gone (hitches): none of it, and out of the config too.
	for (const char* k: {"KYTY_MEMCUT", "KYTY_REBAR_STAGING", "KYTY_STAGING_RING_MB", "KYTY_RAM_TARGET_MB", "KYTY_RAM_FLUSH_SECONDS",
	                     "KYTY_RAM_FLUSH_MB", "KYTY_RAM_FLUSH_LEVEL"})
		sets.emplace_back(k, "");
	if (s.safe_gpu) sets.emplace_back("KYTY_COMPUTE_CHAIN", "0");
	// (perfopts-1107) Opciones de rendimiento: only set when chosen (off leaves the environment as it was).
	if (s.perf_pm4_regfile) sets.emplace_back("KYTY_PM4_REGFILE", "1");
	if (s.perf_upload_lanes) sets.emplace_back("KYTY_UPLOAD_LANES", "2");
	if (s.perf_upload_help) sets.emplace_back("KYTY_UPLOAD_HELP", "2");
	if (s.perf_spin_gate) sets.emplace_back("KYTY_SPIN_GATE", "1");
	if (s.perf_eager_feedback) sets.emplace_back("KYTY_EAGER_FEEDBACK", "1");
	if (s.perf_feedback_batch) sets.emplace_back("KYTY_FEEDBACK_BATCH", "1");
	if (s.perf_lane_wide) sets.emplace_back("KYTY_LANE_LOCAL_WIDE", "1");
	// Anti-Lag (10-24): with the AMD frame generation, AMD's Anti-Lag 2 on an AMD card (default) or ours (2 frames queued, Ultra 1);
	// the fixed Souls modes always ours in Ultra. With NVIDIA's frame generation, Reflex (default) or ours with Reflex off.
	std::string al = s.antilag_mode.empty() || s.antilag_mode == "auto" ? (pc.amd ? "amd2" : "normal") : s.antilag_mode;
	if (fixed_fg) al = "ultra";
	const bool amd_al = pc.amd && amd_fg && al == "amd2" && !fixed_fg;
	const bool own_al = (al == "normal" || al == "ultra") && amd_fg;
	const bool dg_own = dlss_fg && (s.antilag_dlssg == "normal" || s.antilag_dlssg == "ultra");
	if (dg_own) {
		sets.emplace_back("KYTY_DLSSG_REFLEX", "off");
		sets.emplace_back("KYTY_LOWLAT", "1");
		sets.emplace_back("KYTY_LOWLAT_QUEUE", s.antilag_dlssg == "ultra" ? "1" : "2");
	}
	if (own_al) {
		sets.emplace_back("KYTY_LOWLAT", "1");
		sets.emplace_back("KYTY_LOWLAT_QUEUE", al == "ultra" ? "1" : "2");
	}
	// (10-26) GTAO and the screen-space RT (Luz RT, volumetric, RT reflections, RT sun, RT particles) are gone: never passed.
	// Bordes sombras: the game's shadow maps filtered (KYTY_SHADOW_FILTER).
	if (const auto level = ShadowFilterLevel(s); !level.empty()) sets.emplace_back("KYTY_SHADOW_FILTER", level);
	// (10-27b) Ray tracing (hardware), the s25 launcher: only Reflejos, Sombras sol and Luces antorcha/lamparas (No / Si);
	// no hardware volumetric, metal or AO.
	{
		Env hw;
		if (s.rt_reflections) {
			hw.Set("KYTY_HWRT_REFL", "1");
			hw.Set("KYTY_HWRT_REFL_MATERIAL", "1");
		}
		if (s.rt_sun) hw.Set("KYTY_HWRT_SHADOW", "2");
		// (ds-pc-port10) No hardware RT Antorchas in the port (KYTY_HWRT_LIGHTS / _PARTICLES never passed).
		// (10-28, ds-rt1026-restore) Hardware RT only as experimental (rt_experimental): the engine needs KYTY_RT_HW_EXPERIMENTAL=1.
		if (!hw.list.empty() && s.rt_hw_exp) {
			sets.emplace_back("KYTY_RT_HW_EXPERIMENTAL", "1");
			sets.emplace_back("KYTY_HWRT", "1");
			sets.insert(sets.end(), hw.list.begin(), hw.list.end());
		}
	}
	// (10-28, the c3e58bd5 screen-space RT set) Luz RT (SSGI: run-windows -Ssgi 1-3; Extremo = Ultra with more bounce and
	// contact shading) and Reflejos RT (KYTY_RT_REFLECTIONS 1-5; Ultra and up with more presence).
	if (!s.pure_game && s.rt_ssgi != "0" && !s.rt_ssgi.empty()) {
		sets.emplace_back("KYTY_SSGI", std::to_string(std::min(std::atoi(s.rt_ssgi.c_str()), 3)));
		if (s.rt_ssgi == "4") {
			sets.emplace_back("KYTY_SSGI_INTENSITY", "1.6");
			sets.emplace_back("KYTY_SSGI_AO", "0.4");
		}
	}
	if (!s.pure_game && s.rt_ss_refl != "0" && !s.rt_ss_refl.empty()) {
		// (refl-hybrid-1028, Launcher-souls.ps1) Hibrido (6) = Alto (level 3) + the hardware fill (set after the torch below).
		sets.emplace_back("KYTY_RT_REFLECTIONS", s.rt_ss_refl == "6" ? "3" : s.rt_ss_refl);
		if (const int lv = std::atoi(s.rt_ss_refl.c_str()); lv >= 4 && lv <= 5) {
			sets.emplace_back("KYTY_RT_REFLECTIONS_MIN", "0.55");
			sets.emplace_back("KYTY_RT_REFLECTIONS_GUIDE_STRENGTH", "1.3");
		}
		// (refl-sharp-1031) empty (unset) at the defaults: exactly the previous behaviour
		sets.emplace_back("KYTY_RT_REFLECTIONS_SHARP", s.rt_refl_sharp == "0" ? "" : s.rt_refl_sharp);
		sets.emplace_back("KYTY_RT_REFLECTIONS_WATER_WAVES", s.rt_refl_waves == "0" ? "" : s.rt_refl_waves);
	} else {
		sets.emplace_back("KYTY_RT_REFLECTIONS", "");
		sets.emplace_back("KYTY_RT_REFLECTIONS_SHARP", "");
		sets.emplace_back("KYTY_RT_REFLECTIONS_WATER_WAVES", "");
	}
	sets.emplace_back("KYTY_DLSSG_PRESET", "C");
	sets.emplace_back("KYTY_FG_PACING", "0");        // (10-21) the emulator's frame pacing (stuck at 20 FPS) off
	sets.emplace_back("KYTY_DS_SIXTY_STEP", "0");    // (10-22) cinematic as on PS5
	sets.emplace_back("KYTY_DS_VARIABLE_STEP", "1"); // (10-21) physics at each frame's real time
	// Upscaler sharpness: (10-24) ligero 75 %, (10-24) PC low profile 100 %.
	std::string sharp = s.sharpness == "auto" ? std::string() : s.sharpness;
	if (s.souls_mode == "low") sharp = "1";
	else if (s.souls_mode == "ligero" || s.souls_mode == "ligero-plus") sharp = "0.75";
	if (!sharp.empty()) sets.emplace_back("KYTY_UPSCALE_SHARPNESS", sharp);
	if (fixed_fg) sets.emplace_back("KYTY_FLIP_FPS", s.souls_mode == "equilibrado" ? "45" : s.souls_mode == "ligero-plus" ? "40" : "30"); // (11-12) ligero 30x2, (11-14) ligero + 40x2 (360 Hz vblank / 40 = 9 vblanks exactly)
	if (unlocked && with_fg) sets.emplace_back("KYTY_FRAMEGEN_BASE", "0");
	if (s.af16) sets.emplace_back("KYTY_FORCE_AF", "16");
	if (s.texture_sharpness == "2") {
		// (10-25) Extra nitidas: log2(render / output) - 1 with an upscaler (NVIDIA's DLSS guide), -0.5 without; at least -2.
		const double lb = std::max(-2.0, ratio > 1.0 ? -std::log2(ratio) - 1.0 : -0.5);
		char         text[32];
		std::snprintf(text, sizeof(text), "%.3f", lb);
		sets.emplace_back("KYTY_TEXTURE_LOD_BIAS", text);
	} else if (s.texture_sharpness == "1") {
		sets.emplace_back("KYTY_TEXTURE_LOD_BIAS", "auto"); // (10-23) per texture: not the fog
	}
	sets.emplace_back("KYTY_DLSSG_PACE", "grid"); // (10-22 e) reflexfree waited once per shown frame: 14 real
	sets.emplace_back("KYTY_FAULT_STATS", "1");
	sets.emplace_back("KYTY_ALIAS_STATS", "0"); // (10-23) the alias walk of every mapping: hitches on area changes
	if (std::strcmp(fg->id, "amd-x4") == 0) sets.emplace_back("KYTY_VRR_CAP_FPS", "auto"); // (10-29, the launcher) the VRR cap from the monitor
	if (std::strstr(fg->config, "dyn") != nullptr) sets.emplace_back("KYTY_DYN_QUALITY", "0.95"); // (10-22 g) fill the monitor
	if (s.metrics == "1" || s.metrics == "2") sets.emplace_back("KYTY_LATENCY", "1");
	sets.emplace_back("KYTY_DEPTH_ARRAY_LOG", "1");
	sets.emplace_back("KYTY_LOWLAT_PACE_BEFORE_INPUT", "0");
	sets.emplace_back("KYTY_FLIP_QUEUE_LIMIT", "0");
	sets.emplace_back("KYTY_PHASE_PACE", "0");
	if (fixed_fg) sets.emplace_back("KYTY_DISPLAYED_WAIT", "1"); // (10-25) one frame queued as on the console: 30 / 45 only
	// Luz volumetrica (Bajo / Medio / Alto): KYTY_GODRAYS, its sun from the screen-space RT shadow at strength 0 (draws no
	// shadow). KYTY_RT_SUN=0 (vol25): without the cascade mode the volumetric gets the sun every frame (no fog popping).
	// (The engine has no sun-fix yet: Luz volumetrica and Sombras sol stay exclusive, the menu turns the other off.)
	// (10-28, ds-shadow-1028) Sombras RT (screen-space, cshadow.comp; never hardware RT): level 1 / 2 / 3 at strength 0.35 /
	// 0.5 / 0.7, without the cascade mode or the TLAS. With Luz volumetrica and no Sombras RT the pass runs at strength 0.
	// (ds-pc-port9) Path tracing (KYTY_AO + KYTY_AO_HW) and Sombras RT are retired (they looked bad): always off.
	const bool ao_on = false;
	sets.emplace_back("KYTY_AO", "0"); // (ao-1028)
	// (ds-ptao-1029, the launcher's "Path tracing") the hardware scene (TLAS) for the AO's rays; without hardware RT the
	// screen-space AO stays. The other hardware passes off without rt_experimental.
	if (ao_on) {
		sets.emplace_back("KYTY_AO_HW", "1");
		sets.emplace_back("KYTY_RT_HW_EXPERIMENTAL", "1");
		sets.emplace_back("KYTY_HWRT", "1");
		if (!s.rt_hw_exp)
			for (const char* k: {"KYTY_HWRT_REFL", "KYTY_HWRT_SHADOW", "KYTY_HWRT_LIGHTS", "KYTY_HWRT_PARTICLES", "KYTY_HWRT_GI", "KYTY_HWRT_AO",
			                     "KYTY_HWRT_CONTACT", "KYTY_HWRT_SOFTSUN", "KYTY_HWRT_WATER", "KYTY_HWRT_VOLSHADOW", "KYTY_RT_SUN"})
				sets.emplace_back(k, "0");
	}
	const bool godrays = !s.pure_game && (s.godrays == "1" || s.godrays == "2" || s.godrays == "3");
	const bool shadows = false; // (ds-pc-port9) Sombras RT retired
	if (godrays) sets.emplace_back("KYTY_GODRAYS", s.godrays);
	const bool motes = !s.pure_game && (s.motes == "1" || s.motes == "2" || s.motes == "3"); // (motes-1031)
	sets.emplace_back("KYTY_MOTES", motes ? s.motes : "");
	if (shadows) {
		static const char* strength[] = {"0", "0.35", "0.5", "0.7"};
		sets.emplace_back("KYTY_RT_SHADOWS", s.rt_shadows);
		sets.emplace_back("KYTY_RT_SHADOWS_STRENGTH", strength[std::atoi(s.rt_shadows.c_str())]);
		sets.emplace_back("KYTY_RT_SUN", "0");
		sets.emplace_back("KYTY_HWRT_SHADOW", "0");
	} else if (godrays || motes) { // (motes-1031: the motes' light needs the sun too)
		sets.emplace_back("KYTY_RT_SHADOWS", "1");
		sets.emplace_back("KYTY_RT_SHADOWS_STRENGTH", "0");
		sets.emplace_back("KYTY_RT_SUN", "0");
	} else {
		sets.emplace_back("KYTY_RT_SHADOWS", "");
		sets.emplace_back("KYTY_RT_SHADOWS_STRENGTH", "");
	}
	// (ds-pc-port10) Luz de antorchas RT (screen-space and its Hibrido) and the torch lens flare are gone from the port:
	// always off.
	sets.emplace_back("KYTY_TORCH", "0");
	sets.emplace_back("KYTY_TORCH_HW", "0");
	sets.emplace_back("KYTY_TORCH_FLARE", "0");
	// (refl-hybrid-1028) Reflejos RT Hibrido: the hardware scene (TLAS) only to fill where the screen-space ray misses (the
	// other hardware passes off without rt_experimental), as the launcher.
	if (!s.pure_game && s.rt_ss_refl == "6") {
		sets.emplace_back("KYTY_RT_REFLECTIONS_HYBRID", "1");
		sets.emplace_back("KYTY_RT_HW_EXPERIMENTAL", "1");
		sets.emplace_back("KYTY_HWRT", "1");
		if (!s.rt_hw_exp)
			for (const char* k: {"KYTY_HWRT_REFL", "KYTY_HWRT_SHADOW", "KYTY_HWRT_LIGHTS", "KYTY_HWRT_PARTICLES", "KYTY_HWRT_GI", "KYTY_HWRT_AO",
			                     "KYTY_HWRT_CONTACT", "KYTY_HWRT_SOFTSUN", "KYTY_HWRT_WATER", "KYTY_HWRT_VOLSHADOW", "KYTY_RT_SUN"})
				sets.emplace_back(k, "0");
	}
	// (ds-particles-1029) Humo y particulas: No / Suave / Antorchas (KYTY_SMOKE 0-2).
	sets.emplace_back("KYTY_SMOKE", !s.pure_game && s.smoke != "0" ? "1" : "0");
	// (ds-all-1029) The torch passes skip their work while no light is tracked (the detection keeps running).
	sets.emplace_back("KYTY_TORCH_IDLE", "1");
	// The emulator's distance depth of field (ds-dof-1029) is not in the port: always off.
	sets.emplace_back("KYTY_DOF", "0");
	// (ds-pc-port10, ds-foveated-1029's launcher) DLSS 5: NVIDIA's Neural Rendering after the upscaler on the foveated grid,
	// full resolution in the centre and less towards the edges (KYTY_FOV_EDGE 0.75 / 0.5 / 0.35); No: off. The file:
	// signed by NVIDIA only, or any (KYTY_DLSS_NR_UNSIGNED=1). NVIDIA only; without the user's nvngx_dlssnr.dll the
	// engine logs it and the picture stays the upscaler's.
	{
		static const char* edge[] = {"0.5", "0.75", "0.5", "0.35"};
		const int          level  = pc.nvidia ? std::clamp(std::atoi(s.dlss5.c_str()), 0, 3) : 0;
		if (level > 0) {
			sets.emplace_back("KYTY_FOVEATED", "1");
			sets.emplace_back("KYTY_DLSS_NR_PRESR", "0");
			sets.emplace_back("KYTY_FOV_EDGE", edge[level]);
			sets.emplace_back("KYTY_DLSS_NR_UNSIGNED", s.dlss5_unsigned ? "1" : "0");
			// (ds-pc-port40) Suave / Medio / Fuerte used to change only the foveated grid's edge density (GPU time): the
			// model's look stayed the same at every level and "alta" was the default 1. Now the level is the model's own
			// DLSSNR.Intensity and Intensidad the share of its edit applied (above 1: amplified), so each step is visible.
			static const char* intensity[] = {"1", "0.75", "1.1", "1.5"}; // (port40, as the user saw Intensidad media)
			sets.emplace_back("KYTY_DLSS_NR", "1");
			// (ds-pc-port43) Intensidad alta: the glitches (odd colours) only there. The model's own intensity at most 1 and the
			// edit share a little above media's (1.15): a bit stronger than media, no extrapolated colours. Baja / media as in
			// port42 (media looked right to the user).
			const bool alta = s.dlss5_strength != "baja" && s.dlss5_strength != "media";
			sets.emplace_back("KYTY_DLSS_NR_INTENSITY", alta && std::atof(intensity[level]) > 1.0 ? "1" : intensity[level]);
			// (ds-pc-port43) The share of the model's edit: baja 0.5, media 1 (as port42), alta 1.15 (was 1.5).
			sets.emplace_back("KYTY_DLSS_NR_STRENGTH", s.dlss5_strength == "baja" ? "0.5" : s.dlss5_strength == "media" ? "1" : "1.15");
		} else {
			sets.emplace_back("KYTY_FOVEATED", "0");
			sets.emplace_back("KYTY_DLSS_NR", "0");
			sets.emplace_back("KYTY_DLSS_NR_PRESR", "0");
		}
	}
	// Its vblank: 840 Hz with DLSS frame generation, 900 with AMD dynamic (10-29, the launcher: 60 and 75 fit, 56 does
	// not, so the game is not held at 56 with AMD x4), 360 with the other frame generation, 960 unlocked without, else 60.
	const bool vb900  = std::strcmp(fg->id, "amd-dinamico") == 0;
	const int  vblank = with_fg ? (dlss_fg ? 840 : vb900 ? 900 : 360) : (unlocked ? 960 : 60);
	plan.vblank       = vblank;
	const std::string& quality = s.quality;
	const std::string& texture = s.texture_reduce;

	// run-windows.ps1, in its order.
	if (env.Get("KYTY_RENDER_CPUS").empty() && env.Has("KYTY_RECORDING_CPUS")) env.Set("KYTY_RENDER_CPUS", env.Get("KYTY_RECORDING_CPUS"));
	if (env.Get("KYTY_ASYNC_UPLOAD") == "1") env.Set("KYTY_ASYNC_UPLOAD", "2");
	// (11-11) Always set: "" clears a KYTY_SKIP_INTRO=1 inherited from an earlier run (the restart after an update or the DLSS
	// fresh process kept it, so "Saltar intro" off still skipped).
	env.Set("KYTY_SKIP_INTRO", s.skip_intro ? "1" : "");
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
	// (11-13) Native-PC memory budget, set in the game's own engine at boot (no memory moved while playing): the
	// texture streaming pool (texturePoolSizeMiBs, "Streaming Pool Total Size MiBs") defaults to 8192 MiB of the
	// PS5's unified memory; on PC the textures live in video memory, so 4096 MiB of guest RAM is the budget.
	// (11-13) texturePoolSizeMiBs left at the game default (the user: textures for the graphics are not touched).
	// (11-13) With a 4 GB texture pool the game needs less than the PS5's 12 GiB allocator budget: the guest store is
	// 12288 MiB (was 13824), so the block the engine maps above 12 GiB and keeps for good (~1.1 GB) no longer exists.
	// Set once at boot; nothing is moved while playing.
	// (11-13) KYTY_GUEST_STORE_MB=12288 removed: later direct allocations (104 MiB, 2 MiB) no longer fit and textures
	// went black at times. The store stays at the PS5 size.
	// Guest memory committed only where the game maps it (at map time, not moved later): not all 13.8 GB at boot.
	// (11-13) KYTY_GUEST_LAZY_COMMIT=1 (off by default) replaces it with the emulator's lazy commit (each 64 KiB committed
	// at its first touch, never decommitted or moved).
	env.Default("KYTY_BACKING_ON_DEMAND", "1");
	// (native port) Native memory model: per-frame CPU->GPU arenas as an upload heap (no page-protection tracking),
	// streamed mips reported once per mip, render-target arenas GPU-only. Byte-checked; falls back if the eboot differs.
	env.Default("KYTY_NATIVE_MEM", "2"); // 3 (upload heap) only when the user turns it on: the first build cost 40 ms of CPU per frame
	// (native port) Guest memory written only where bytes change (no needless dirty pages); ReBAR upload rings.
	env.Default("KYTY_LEAN_HOSTMEM", "1");
	if (ratio > 0) env.Set("KYTY_UPSCALE_RATIO", Num(ratio));
	// (ds-pc-port12, res-1029) Without an upscaler the 3D is the output (native; the game's rows at their height, never above it).
	if (upscaler.empty()) {
		const bool juego = std::strncmp(res->id, "juego-", 6) == 0 && res->value < res_out_h;
		const int  rh    = juego ? static_cast<int>(res->value) : res_out_h;
		if (juego || rh != 1440) env.Set("KYTY_RENDER_HEIGHT", std::to_string(rh)); // (-RenderHeight)
	}
	// (ds-pc-port12, Launcher-souls.ps1 of ds-pacefix-1029) The output resolution and the height the upscalers work from.
	if (demons) {
		env.Set("KYTY_OUTPUT_HEIGHT", std::to_string(res_out_h));
		env.Set("KYTY_DISPLAY_HEIGHT", std::to_string(res_out_h));
		env.Set("KYTY_RES_UPSCALER", upscaler.empty() ? std::string("none") : std::string(res->id));
	}
	env.Remove("KYTY_UPSCALE");
	if (upscaler == "dlss" && s.dlss_preset != "auto" && !s.dlss_preset.empty()) env.Set("KYTY_DLSS_PRESET", s.dlss_preset);
	if (upscaler == "fsr4" && res->fsr4_preset != nullptr) env.Set("KYTY_FSR4_PRESET", res->fsr4_preset); // (run-windows -Fsr4Preset)
	if (!s.aspect.empty() && s.aspect != "16:9") env.Set("KYTY_RENDER_ASPECT", s.aspect);
	// -ForceMode performance (10-25: 4K no longer forces the cinematic mode: it held the game at 30 real).
	// (10-28, the emulator launcher) A 4K output starts the game in Cinematico, the other outputs in Rendimiento.
	bool cine4k = false;
	if (demons) {
		int out_h = s.fullscreen && pc.screen_h > 0 ? pc.screen_h : (s.height > 0 ? s.height : 1440);
		if (int w = 0, h = 0; s.fullscreen && std::sscanf(s.resolution.c_str(), "%dx%d", &w, &h) == 2 && h > 0) out_h = h;
		// (10-29, the owner) Always Rendimiento (Cinematico held the cutscenes at 30); the cinematic quality comes from
		// Iluminacion IBL / Teselado (4K defaults: Completa + Alto), first in KYTY_GAME_CONVARS_EXTRA below.
		env.Set("KYTY_DS_FORCE_MODE", "performance");
		cine4k = out_h >= 2160;
	}
	static const std::map<std::string, std::string> convars = {
	    {"high", "r_forceViewportHeightForPixelSizeCalc=1080"},
	    {"low", "r_forceViewportHeightForPixelSizeCalc=720"},
	    {"optimizado", "r_forceViewportHeightForPixelSizeCalc=600"},
    {"lowest", "r_forceViewportHeightForPixelSizeCalc=480"}};
	if (auto it = convars.find(quality); it != convars.end() && demons) env.AppendConvars(it->second);
	if (amd_al) env.Set("KYTY_ANTILAG", "1"); // (run-windows -AntiLag)
	if (demons) {
		// run-windows.ps1 $gfxExtra / -PostOff / $detailConvars / -Shadows (10-24: the cascade cuts spread, 75 % log +
		// 25 % linear from 0.5 m, the last one the maximum: else the far away was drawn black).
		const bool far = s.detail == "ultra"; // (10-24) the launcher's Ultra = run-windows' "Lejana"
		static const std::map<std::string, std::string> extra = {
		    {"refl_ultra", "r_ssrEnableOverrides=true;r_ssrMaxIterations=192;r_ssrMaxRayLength=1400;r_ssrCutoffDistance=1500"},
		    {"refl_normal", "r_ssrEnableOverrides=true;r_ssrMaxIterations=64;r_ssrMaxRayLength=600;r_ssrCutoffDistance=500"},
		    {"refl_optimizado", "r_ssrEnableOverrides=true;r_ssrMaxIterations=48;r_ssrMaxRayLength=450;r_ssrCutoffDistance=375"},
		    {"refl_bajo", "r_ssrEnableOverrides=true;r_ssrMaxIterations=32;r_ssrMaxRayLength=300;r_ssrCutoffDistance=250"},
		    {"fade_ultra", "r_fadeOutInstanceOverDistance=35"},
		    {"fade_normal", "r_fadeOutInstanceOverDistance=10"},
		    {"fade_optimizado", "r_fadeOutInstanceOverDistance=8"},
		    {"fade_bajo", "r_fadeOutInstanceOverDistance=6"},
		    {"aa_ultra", "r_aaSharpenStrengthStatic=5;r_aaSharpenStrengthMotion=5;r_aaHistorySharpeningStrength=1.0"},
		    {"aa_alto", "r_aaSharpenStrengthStatic=3.5;r_aaSharpenStrengthMotion=3.5;r_aaHistorySharpeningStrength=0.9"},
		    {"aa_optimizado", "r_aaSharpenStrengthStatic=1.6;r_aaSharpenStrengthMotion=1.6;r_aaHistorySharpeningStrength=0.6"},
		    {"aa_bajo", "r_aaSharpenStrengthStatic=1.2;r_aaSharpenStrengthMotion=1.2;r_aaHistorySharpeningStrength=0.5"},
		    {"aa_no", "r_aaSharpenStrengthStatic=0;r_aaSharpenStrengthMotion=0;r_aaHistorySharpeningStrength=0"}};
		for (const std::string& k: {"refl_" + s.reflections, "fade_" + s.object_fade, "aa_" + s.aa_sharp})
			if (auto it = extra.find(k); it != extra.end()) env.AppendConvars(it->second);
		std::string post;
		const auto  off = [&](bool on, const char* convar) {
            if (!on) post += (post.empty() ? "" : ";") + std::string(convar);
		};
		off(s.motion_blur, "r_motionBlurEnable=false");
		off(s.dof, "r_dofEnable=false");
		off(s.vignette, "r_vignetteEnable=false");
		off(s.bloom, "r_bloomEnable=false");
		off(s.eye_adapt, "r_exposureEyeAdaptEnable=false");
		if (!post.empty()) env.AppendConvars(post);
		if (far) env.Default("KYTY_RT_SHADOWS_FADE", "250");
		static const std::map<std::string, std::string> details = {
		    {"ultra", "r_virtualGeometryNumShadowCascades=4;sunShadowMaxDistance=1000;sunShadowCascadeDistance0=65;sunShadowCascadeDistance1=142;"
		              "sunShadowCascadeDistance2=300;sunShadowCascadeDistance3=1000;sunShadowLightDistance=1000;r_contactShadowsEnabled=true;"
		              "r_fadeOutInstanceOverDistance=60;r_ssrEnableOverrides=true;r_ssrMaxIterations=256;r_ssrMaxRayLength=1600;"
		              "r_ssrCutoffDistance=5000;r_iblResolution=Full;r_tessellationDensity=20.0;r_tessellationSmallestEdgeLength=0.015;"
		              "r_tessellationFadeOutStart=25.0;r_tessellationFadeOutRange=5.0"},
		    {"muy-alto", "r_virtualGeometryNumShadowCascades=2;sunShadowMaxDistance=250;sunShadowCascadeDistance0=40;sunShadowCascadeDistance1=250;"
		                 "r_iblResolution=Full;r_tessellationDensity=18.0;r_tessellationSmallestEdgeLength=0.02;r_tessellationFadeOutStart=15.0;"
		                 "r_tessellationFadeOutRange=3.0"},
		    {"normal", "r_virtualGeometryNumShadowCascades=2;sunShadowMaxDistance=180;sunShadowCascadeDistance0=30;sunShadowCascadeDistance1=180;"
		               "r_iblResolution=HalfResX;r_tessellationDensity=6.0;r_tessellationSmallestEdgeLength=0.06;r_tessellationFadeOutStart=9.0;"
		               "r_tessellationFadeOutRange=3.0"},
		    {"optimizado", "r_virtualGeometryNumShadowCascades=2;sunShadowMaxDistance=150;sunShadowCascadeDistance0=28;sunShadowCascadeDistance1=150;"
		                   "r_iblResolution=HalfResX;r_tessellationDensity=5.0;r_tessellationSmallestEdgeLength=0.07;r_tessellationFadeOutStart=7.5;"
		                   "r_tessellationFadeOutRange=2.5"},
		    {"bajo", "r_virtualGeometryNumShadowCascades=1;sunShadowMaxDistance=125;sunShadowCascadeDistance0=125;r_iblResolution=HalfResX;"
		             "r_tessellationDensity=4.0;r_tessellationSmallestEdgeLength=0.08;r_tessellationFadeOutStart=6.0;r_tessellationFadeOutRange=2.0"}};
		if (auto it = details.find(s.detail); it != details.end()) env.AppendConvars(it->second);
		static const std::map<std::string, std::string> shadows = {
		    {"alto", "sunShadowMaxDistance=190;sunShadowLightDistance=1000"},
		    {"medio", "sunShadowMaxDistance=125;sunShadowLightDistance=1000"},
		    {"optimizado", "sunShadowMaxDistance=90;sunShadowLightDistance=1000;r_enableAmbientShadowing=false"},
		    {"bajo", "sunShadowMaxDistance=60;sunShadowLightDistance=1000;r_enableAmbientShadowing=false;r_enableSpotShadows=false"}};
		if (auto it = shadows.find(s.shadows); it != shadows.end()) env.AppendConvars(it->second);
		{
			// (10-29) Iluminacion IBL and Teselado: first in KYTY_GAME_CONVARS_EXTRA (applied after KYTY_GAME_CONVARS, so they
			// win over Detalle's own IBL / tessellation values); a user's own extra list after them.
			const std::string ibl  = !s.ibl.empty() ? s.ibl : cine4k ? "completa" : "mitad";
			const std::string tess = !s.tess.empty() ? s.tess : cine4k ? "alto" : "medio";
			static const std::map<std::string, std::string> tess_cv = {
			    {"bajo", "r_tessellationDensity=4.0;r_tessellationSmallestEdgeLength=0.08;r_tessellationFadeOutStart=6.0;r_tessellationFadeOutRange=2.0"},
			    {"optimizado", "r_tessellationDensity=6.0;r_tessellationSmallestEdgeLength=0.06;r_tessellationFadeOutStart=9.0;r_tessellationFadeOutRange=2.5"},
			    {"medio", "r_tessellationDensity=8.0;r_tessellationSmallestEdgeLength=0.04;r_tessellationFadeOutStart=12.0;r_tessellationFadeOutRange=3.0"},
			    {"alto", "r_tessellationDensity=18.0;r_tessellationSmallestEdgeLength=0.02;r_tessellationFadeOutStart=15.0;r_tessellationFadeOutRange=3.0"},
			    {"ultra", "r_tessellationDensity=20.0;r_tessellationSmallestEdgeLength=0.015;r_tessellationFadeOutStart=25.0;r_tessellationFadeOutRange=5.0"},
			    {"extremo", "r_tessellationDensity=20.0;r_tessellationSmallestEdgeLength=0.01;r_tessellationFadeOutStart=40.0;r_tessellationFadeOutRange=8.0"}};
			std::string x = std::string("r_iblResolution=") + (ibl == "completa" ? "Full" : "HalfResX");
			if (auto it = tess_cv.find(tess); it != tess_cv.end()) x += ";" + it->second;
			// (ds-pc-port37) The "Ajustes del juego" next (the launcher's order: IBL / tessellation, its game settings, the user's).
			if (const std::string g = GameSettingsConvars(s); !g.empty()) x += ";" + g;
			if (const std::string old = env.Get("KYTY_GAME_CONVARS_EXTRA"); !old.empty()) x += ";" + old;
			env.Set("KYTY_GAME_CONVARS_EXTRA", x);
		}
	}
	// (11-11) "Maximo" runs as "Alto" (one level left out of the >= 2048 textures): whole streamed textures still showed
	// the Nexus floor / candle glitch on cards of 12 GB and more.
	env.Set("KYTY_TEXTURE_REDUCE", texture == "1" || texture == "2" || texture == "o" ? texture : "h");
	env.Default("KYTY_UNMAP_PROTECT_SKIP", "1");
	env.Set("KYTY_DRIVER_CACHE_KEY", DriverCacheKey(s));
	env.Set("KYTY_SHADER_WARMUP_ONLY", "");
	// (10-27b) The shader warmup and prefetch run beside the game's intro and title screen: their progress line stays on
	// screen until they finish ("Preparando shaders X/Y"; the launcher's 6 s hid it).
	env.Set("KYTY_SHADER_WARMUP", "1");
	env.Set("KYTY_SHADER_WARMUP_BACKGROUND", "");
	env.Set("KYTY_SHADER_PREFETCH", "");
	env.Set("KYTY_PREFETCH_RELEASE", "");
	env.Set("KYTY_STATUS_HUD_SECONDS", "6"); // (ds-pc-port37) as the emulator launcher's: shown 6 s, the work goes on behind
	env.Default("KYTY_SHADER_WARMUP_SECONDS", "60");
	env.Default("KYTY_TABLE_STORE_BUDGET", "8");
	// (ds-native-lean) The run log as a native game's: no SLOW lines (a line per frame over 25 ms, 3700 in a 3-minute log,
	// and a clock read around every instrumented call), the [fps] line once in 10 s (local/lean-log.h). The metrics
	// panel reads its numbers from memory. [Avanzado] extra KYTY_HITCH_LOG_MS=40;KYTY_LOG_VERBOSE=1: the old log.
	env.Set("KYTY_HITCH_LOG_MS", "0");
	env.Default("KYTY_LOG_PERIOD", "10"); // (the rate lines: [fps], [fsr3] / [dlssg] / [fsr3-mfg], FG particle native)
	if (s.aspect_fit) env.Set("KYTY_PRESENT_ASPECT", "fit");
	env.Default("KYTY_ASYNC_DRAW_PIPELINES", "1");
	// (ds-pc-port47) "Debe funcionar exactamente como en el launcher": every shader default as the emulator's (de604751
	// with its launch.json): KYTY_STATIC_WARMUP, KYTY_SCENE_WARMUP (by map) and the prefetch threads (16, then 8) left to
	// the emulator's own defaults, no shader gate, and no SPIR-V disk store (the emulator has none: off). The warmup on 16
	// threads, launch.json's KYTY_SHADER_WARMUP_THREADS.
	env.Set("KYTY_SPIRV_DISK", "0");
	// (ds-pc-port47) Root cause of the port-only hitches: ds-perf-1031's V-Cache pin (not in the emulator, de604751) set
	// KYTY_RENDER_CPUS to CPUs 2-15 of the 16 the port's process ran on, and every background thread keeps off those
	// (AvoidCpuList): the shader prefetch, warmup and every async pipeline worker squeezed onto CPUs 0-1 (port logs:
	// 8 pipelines made of the static store ending together after 2.28 s, "static store hit" 120-2300 ms; the emulator:
	// none). Off, as the emulator.
	env.Default("KYTY_VCACHE_PIN", "0");
	env.Set("KYTY_SHADER_GATE", "");
	env.Set("KYTY_SCENE_WARMUP", "");
	env.Set("KYTY_STATIC_WARMUP", "");
	env.Set("KYTY_SHADER_PREFETCH_THREADS", "");
	env.Set("KYTY_SHADER_WARMUP_THREADS", std::to_string(std::clamp(std::thread::hardware_concurrency(), 1u, 64u))); // (ds-pc-port47: every CPU)
	// (11-13) A hard RAM budget whatever the PC's free RAM: the RAM guard trims the working set (cold pages to the
	// standby list / page file, in quiet windows, a few hundred MiB a second) and runs the release passes when over it.
	// KYTY_RESIDENT_MB is a minimum, not a cap: with 64 GB and no pressure nothing held the 15.7 GB working set down.
	// (11-13) No fast memory moves: the user saw the PC reboot when memory was freed/moved quickly (working-set trims,
	// release passes). The RAM budget and the RAM guard's release passes are off; memory stays where the game put it.
	env.Default("KYTY_RAM_BUDGET_MB", "0");
	env.Default("KYTY_RAM_GUARD", "0");
	// (11-13) The low-RAM profile turns itself on with 20 GiB or less and would enable runtime releases (soft release,
	// RAM diet): off, nothing is freed or moved while playing.
	env.Default("KYTY_SOFT_RELEASE", "0");
	env.Default("KYTY_RAM_DIET", "0");
	// (11-13) Texture offload and GPU-only offload off on every PC: on 20 GiB or less they turned on by themselves,
	// dropped textures' guest bytes from RAM while playing and gave black textures.
	env.Default("KYTY_TEXTURE_OFFLOAD", "0");
	env.Default("KYTY_GPU_ONLY_OFFLOAD", "0");
	// (11-13) PCs of 20 GB or less: the image upload ring (host-visible memory the driver charges to commit) 128 MiB
	// instead of 256, set at creation; per-frame uploads are 6-30 MB. Nothing moved at runtime.
	if (pc.ram_gb > 0 && pc.ram_gb <= 20) env.Default("KYTY_STAGING_RING_MB", "128");
	// -Ram16 (10-24): "Optimizacion equipos 16 GB RAM" on a PC of 20 GB of RAM or less (or simulated) only.
	if (low_ram) { // (10-27b) on low-RAM PCs only, whatever the old box says
		env.Default("KYTY_RESIDENT_MB", "6144");
		env.Default("KYTY_PREFETCH_MIN_AVAIL_MB", "1536");
		env.Default("KYTY_BACKING_FILE", "0");
		// (11-13) The guest copy of GPU-only images is dropped once they are on the card: RAM not held twice.
		env.Default("KYTY_GPU_ONLY_OFFLOAD", "0");
		// (11-13) ...which needs the texture offload on: by default only with 20 GiB of RAM or less, so on a 64 GB PC with
		// the low-RAM profile nothing was dropped (no "Texture offload: on" line, no "Guest RAM by GPU cache" line).
		// (11-13) Texture offload off again: graphical glitches in game (its restores read zeros meanwhile).
		env.Default("KYTY_TEXTURE_OFFLOAD", "0");
	}
	if (!upscaler.empty()) env.Set("KYTY_UPSCALER", upscaler);
	for (const auto& [k, v]: sets) env.Set(k, v);
	// Juego original (the launcher's $pureGame): no visual pass the emulator adds; performance, upscaler, DLSS 5, frame
	// generation and latency the same (11-10: DLSS 5 kept, the user's choice).
	if (s.pure_game) {
		static const char* fx[] = {"KYTY_HWRT", "KYTY_RT_REFLECTIONS", "KYTY_RT_SHADOWS", "KYTY_RT_SUN", "KYTY_RT_PARTICLES", "KYTY_RT_LOCAL",
		                           "KYTY_GODRAYS", "KYTY_SHINE", "KYTY_SPEC_OCCLUSION", "KYTY_GTAO", "KYTY_SSGI", "KYTY_SSR", "KYTY_CONTACT_SHADOWS",
		                           "KYTY_AO", "KYTY_TORCH", "KYTY_DOF", "KYTY_SMOKE"};
		for (auto& [k, v]: env.list)
			for (const char* p: fx)
				if (k.rfind(p, 0) == 0) v.clear();
		for (const char* k: {"KYTY_HWRT", "KYTY_HWRT_REFL", "KYTY_HWRT_SHADOW", "KYTY_HWRT_LIGHTS", "KYTY_HWRT_PARTICLES", "KYTY_RT_REFLECTIONS",
		                     "KYTY_RT_SHADOWS", "KYTY_CONTACT_SHADOWS", "KYTY_RT_SUN", "KYTY_RT_PARTICLES", "KYTY_GODRAYS", "KYTY_SHINE",
		                     "KYTY_SPEC_OCCLUSION", "KYTY_GTAO", "KYTY_SSGI", "KYTY_SSR", "KYTY_HWRT_GI", "KYTY_HWRT_AO", "KYTY_HWRT_CONTACT",
		                     "KYTY_HWRT_SOFTSUN", "KYTY_HWRT_WATER", "KYTY_HWRT_VOLSHADOW", "KYTY_AO", "KYTY_TORCH", "KYTY_TORCH_FLARE",
		                     "KYTY_TORCH_HW", "KYTY_RT_HW_EXPERIMENTAL", "KYTY_SMOKE", "KYTY_DOF"})
			env.Set(k, "0");
	}
	// settings.ini [Avanzado] extra: as -Set (KEY= removes it).
	{
		std::stringstream list(s.extra);
		std::string       pair;
		while (std::getline(list, pair, ';')) {
			pair          = Trim(pair);
			const auto eq = pair.find('=');
			if (eq == std::string::npos || eq == 0) continue;
			env.Set(Trim(pair.substr(0, eq)), Trim(pair.substr(eq + 1)));
		}
	}
	// (11-13) The fire and particle blending as before: with KYTY_FG_PARTICLE_MV=1 the native particle vectors always on
	// (a run came up with KYTY_FG_PARTICLE_NATIVE=0 from outside the settings and ghosting around fire); only an explicit
	// KYTY_FG_PARTICLE_NATIVE in [Avanzado] extra changes it.
	if (env.Get("KYTY_FG_PARTICLE_MV") == "1" && s.extra.find("KYTY_FG_PARTICLE_NATIVE") == std::string::npos)
		env.Set("KYTY_FG_PARTICLE_NATIVE", "1");
	// (10-21) KYTY_DS_VARIABLE_STEP=1: the game's physics at each frame's real time (run-windows.ps1).
	if (demons && env.Get("KYTY_DS_VARIABLE_STEP") == "1" && env.Get("KYTY_GAME_CONVARS").find("physicsUseFixedTimestep") == std::string::npos)
		env.AppendConvars("physicsUseFixedTimestep=false;gameDeltaTimeCorrectionTolorance=0.01");
	if ((vblank > 60 || env.Get("KYTY_DS_VARIABLE_STEP") == "1") && demons &&
	    env.Get("KYTY_GAME_CONVARS").find("gameMinDeltaTime") == std::string::npos)
		env.AppendConvars("gameMinDeltaTime=0.004");
	// The launcher's own environment.
	env.Set("KYTY_SIMULATE_VRAM_MB", s.simulate_vram8 ? "8192" : "");
	env.Set("KYTY_UNLOCK_30", "0");
	env.Set("KYTY_VSYNC_LOCK", "");
	env.Set("KYTY_FPS_CAP", "");
	env.Set("KYTY_FLIP_WHEN_READY", "");
	if (s.gsync) {
		env.Set("KYTY_VSYNC_LOCK", "0");
		if (unlocked && !with_fg) {
			env.Set("KYTY_FPS_CAP", s.fps_cap > 0 ? Num(s.fps_cap) : "auto");
			env.Set("KYTY_FLIP_WHEN_READY", "1");
		}
	}
	// (10-23) Controls: keyboard and mouse, the pad's remap. (10-25) A little less sensitivity in 30x2 / 45x2.
	env.Set("KYTY_PAD_MAP", s.pad_map);
	env.Set("KYTY_KBM", s.kbm ? "1" : "");
	env.Set("KYTY_KBM_MAP", s.kbm ? s.kbm_map : "");
	const float sens = (s.kbm_sens > 0 ? s.kbm_sens : 1.0f) * (fixed_fg ? 0.92f : 1.0f);
	env.Set("KYTY_KBM_SENS", s.kbm ? Num(sens) : "");
	env.Set("KYTY_KBM_ANTIDZ", s.kbm ? Num(s.kbm_antidz >= 0 ? s.kbm_antidz : 0.15f) : "");
	// (10-28) The launcher's: Camara instantanea, Calibrar camara (once) and the mouse's camera speed (a game convar).
	env.Set("KYTY_KBM_WARP", s.kbm && s.kbm_warp ? "1" : "");
	env.Set("KYTY_KBM_CALIBRATE", s.kbm_calibrate ? "1" : "");
	if (s.kbm_calibrate) env.Set("KYTY_KBM", "1");
	// (step5) The mouse no longer changes the pad: cp11_Setting_CameraSpeed overrode the game's own camera speed option
	// for every input (the user's options save had 5; the convar forced 10: the pad's camera twice as fast with the
	// mouse on), and the eboot's turn-ramp patch (KYTY_KBM_NO_RAMP) also changed the pad's full-tilt turns. Both only
	// on request now: raton_velocidad_camara_forzada other than 0 sets the convar; KYTY_KBM_NO_RAMP=1 in the environment keeps
	// the ramp patch. The mouse's own speed is raton_sensibilidad.
	if ((s.kbm || s.kbm_calibrate) && demons && s.kbm_camera > 0)
		env.AppendConvars("cp11_Setting_CameraSpeed=" + std::to_string(std::clamp(s.kbm_camera, 5, 15)));
	if (s.kbm && std::getenv("KYTY_KBM_NO_RAMP") == nullptr) env.Set("KYTY_KBM_NO_RAMP", "0");
	// The port's own: metrics (the launcher started metricas.ps1; here the in-game panel) and rumble.
	env.Set("KYTY_FPS_HUD", ""); // (ds-pc-port37) Metricas: the emulator launcher's metricas.ps1 window (ds-main); F8 still shows the in-game panel
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
	// (10-27c) The full screen's output resolution (F1 > Pantalla > Resolucion): the monitor in that mode.
	if (int w = 0, h = 0; s.fullscreen && std::sscanf(s.resolution.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0)
		plan.env.emplace_back("KYTY_DISPLAY_MODE", std::to_string(w) + "x" + std::to_string(h));
	if (s.hdr) a.emplace_back("--hdr");
	a.emplace_back("--redzone");
	a.emplace_back("--game");
	a.push_back(s.game);
	return plan;
}

bool NeedsRestart(const Settings& l, const Settings& n) {
	// Live: sharpness, the fps cap, metrics and rumble.
	return l.game != n.game || l.language != n.language || l.fullscreen != n.fullscreen || l.resolution != n.resolution ||
	       l.width != n.width ||
	       l.height != n.height || l.hdr != n.hdr || l.vsync != n.vsync || l.gsync != n.gsync ||
	       l.unlock_fps != n.unlock_fps || l.aspect_fit != n.aspect_fit || l.aspect != n.aspect || l.res != n.res ||
	       l.dlss_preset != n.dlss_preset || l.framegen != n.framegen || l.souls_mode != n.souls_mode ||
	       l.antilag_mode != n.antilag_mode || l.antilag_dlssg != n.antilag_dlssg || l.shadows != n.shadows ||
	       l.detail != n.detail || l.reflections != n.reflections || l.object_fade != n.object_fade || l.aa_sharp != n.aa_sharp ||
	       l.shadow_edges != n.shadow_edges || l.corner_shadow != n.corner_shadow || l.rt_reflections != n.rt_reflections ||
	       l.rt_sun != n.rt_sun || l.rt_fog != n.rt_fog || LampLevel(l) != LampLevel(n) || l.rt_metal != n.rt_metal ||
	       l.rt_ao != n.rt_ao || l.godrays != n.godrays || l.motes != n.motes || l.pure_game != n.pure_game || l.motion_blur != n.motion_blur || l.dof != n.dof || l.vignette != n.vignette ||
	       l.bloom != n.bloom || l.eye_adapt != n.eye_adapt || l.af16 != n.af16 || l.texture_sharpness != n.texture_sharpness ||
	       l.kbm != n.kbm || l.kbm_sens != n.kbm_sens || l.kbm_antidz != n.kbm_antidz || l.kbm_map != n.kbm_map ||
	       l.pad_map != n.pad_map || l.game_settings != n.game_settings || l.kbm_camera != n.kbm_camera || l.kbm_warp != n.kbm_warp || l.ram16_boost != n.ram16_boost || l.perf_pm4_regfile != n.perf_pm4_regfile || l.perf_upload_lanes != n.perf_upload_lanes || l.perf_upload_help != n.perf_upload_help || l.perf_spin_gate != n.perf_spin_gate || l.perf_eager_feedback != n.perf_eager_feedback || l.perf_feedback_batch != n.perf_feedback_batch || l.perf_lane_wide != n.perf_lane_wide || l.profile16 != n.profile16 || l.simulate_ram16 != n.simulate_ram16 ||
	       l.safe_gpu != n.safe_gpu || l.quality != n.quality || l.texture_reduce != n.texture_reduce ||
	       l.simulate_vram8 != n.simulate_vram8 || l.all_ccds != n.all_ccds || l.shader_turbo != n.shader_turbo || l.extra != n.extra ||
	       l.rt_ss_refl != n.rt_ss_refl || l.rt_refl_sharp != n.rt_refl_sharp || l.rt_refl_waves != n.rt_refl_waves || l.rt_ssgi != n.rt_ssgi || l.rt_shadows != n.rt_shadows || l.ao != n.ao || l.smoke != n.smoke || l.ibl != n.ibl || l.tess != n.tess ||
	       l.torch != n.torch || l.lens_flare != n.lens_flare || l.rt_hw_exp != n.rt_hw_exp || l.dlss5 != n.dlss5 ||
	       l.dlss5_unsigned != n.dlss5_unsigned || l.dlss5_strength != n.dlss5_strength ||
	       (l.sharpness == "auto") != (n.sharpness == "auto");
}

bool FixedFg(const Settings& s) {
	return s.souls_mode == "ligero" || s.souls_mode == "ligero-plus" || s.souls_mode == "equilibrado" || s.souls_mode == "low";
}

void RtAllOff(Settings* s) {
	s->rt_reflections = s->rt_sun = s->rt_fog = s->rt_particles = s->rt_metal = s->rt_ao = false;
	// (10-28) and the screen-space set / new passes (rt_experimental itself stays as chosen)
	s->rt_ss_refl = s->rt_ssgi = s->rt_shadows = s->ao = s->torch = s->godrays = s->rt_lamps = s->motes = "0";
	s->lens_flare = false;
}

void ApplyPreset(Settings* s, const std::string& preset) {
	// Launcher-souls.ps1 (10-27): Set-PresetMax / Light / Ultra / Extreme / LowProfile and Souls equilibrado's handler.
	const auto max = [&]() {
		s->texture_reduce = "0";
		s->quality        = "standard";
		s->shadows        = "maximo";
		s->detail         = "alto";
		s->aa_sharp       = "normal";
		s->object_fade    = "alto";
		s->reflections    = "alto";
	};
	const auto post_on = [&]() { s->motion_blur = s->dof = s->vignette = s->bloom = s->eye_adapt = true; };
	s->rt_light          = "0"; // (10-26) gone
	s->ambient_occlusion = "0";
	if (preset == "ligero" || preset == "ligero-plus") {
		// Set-PresetLight; FSR Calidad recommended (10-25); sharpness 75 %.
		// (11-12) the user's screenshot: Texturas Medio, Distancia Alto, Sombras Alto, Detalle Alto (PS5 performance),
		// Filtrado x16, Texturas extra nitidas, Reflejos / Desvanecimiento Normal, Nitidez del AA Alto.
		// (11-12) the user's second screenshot: Texturas Medio, Distancia Maximo, Sombras Medio, Detalle Normal, x16,
		// extra sharp, Reflejos / Desvanecimiento Normal, AA Alto; motion blur off, DoF / bloom / vignette / eye adaptation on.
		s->texture_reduce    = "1";
		s->quality           = "high"; // (11-12) fourth screenshot: Distancia Alto, Detalle Normal, Desvanecimiento Normal,
		s->shadows           = "medio";  // Texturas nitidas, AA Normal (de serie), Teselado Bajo
		s->detail            = "normal";
		s->af16              = true;
		s->texture_sharpness = "1";
		s->aa_sharp          = "normal";
		s->object_fade       = "normal";
		s->reflections       = "normal";
		s->motion_blur = false;
		s->dof = s->bloom = s->vignette = s->eye_adapt = true;
		s->res            = "fsr4-calidad"; // (11-11) FSR 4 Calidad by default (was FSR 3.1 Calidad)
		s->pure_game      = true;           // (11-11) and Juego original (no added effects)
		s->sharpness      = "0.75";
		s->fullscreen     = true;           // (11-12) 1080p by default in Souls ligero / ligero +
		s->resolution     = "1920x1080";
	} else if (preset == "equilibrado") {
		// (10-25) The user's capture: DLSS Calidad (K), sharpness 100 %, textures Maximo, distance and shadows Alto, Detail
		// High, Reflections / Object Fade High, AA Sharpness High, x16, extra sharp textures, all the post-processing.
		max();
		s->quality           = "high";
		s->shadows           = "alto";
		s->aa_sharp          = "alto";
		s->af16              = true;
		s->texture_sharpness = "2";
		post_on();
		s->sharpness = "1";
		s->res       = "fsr4-calidad"; // (11-11) FSR 4 Calidad (was DLSS Calidad K) and Juego original
		s->dlss_preset = "K";
		s->pure_game = true;
	} else if (preset == "ultra") {
		max();
		s->res = "fsr4-calidad"; // (11-11) FSR 4 Calidad by default, and Juego original
		s->pure_game = true;
		s->detail   = "muy-alto";
		s->aa_sharp = "ultra"; // (10-23) costs nothing
	} else if (preset == "extremo") {
		max();
		s->detail      = "ultra";
		s->reflections = "ultra";
		s->object_fade = "ultra";
		s->aa_sharp    = "ultra";
		s->af16        = true;
		if (s->texture_sharpness == "0") s->texture_sharpness = "1";
	} else if (preset == "low") {
		// Set-PresetLowProfile: 900p-ish with FSR Calidad, everything low; what costs nearly nothing at its best.
		s->texture_reduce    = "2";
		s->quality           = "high";
		s->shadows           = "medio";
		s->detail            = "bajo";
		s->reflections       = "bajo";
		s->object_fade       = "bajo";
		s->aa_sharp          = "bajo";
		s->af16              = true;
		if (s->texture_sharpness == "0") s->texture_sharpness = "1";
		post_on();
		s->res       = "fsr-calidad";
		s->sharpness = "1";
	} else {
		max();
	}
	// (11-12) Equilibrado, Ultra and Extremo: 1440p by default (changeable by hand afterwards).
	if (preset == "equilibrado" || preset == "ultra" || preset == "extremo") {
		s->fullscreen = true;
		s->resolution = "2560x1440";
	}
	// (11-12) IBL and tessellation: Extremo at Alto + Completa; every other mode at the PS5 performance mode (mitad + medio).
	if (preset == "extremo") {
		s->ibl  = "completa";
		s->tess = "alto";
	} else {
		s->ibl  = "mitad";
		s->tess = (preset == "ligero" || preset == "ligero-plus") ? "bajo" : "medio"; // (11-12) ligero / + : Teselado Bajo
	}
	// (10-28, the emulator launcher's rule) Every mode / preset: every RT effect No (Reflejos RT, Luz RT, Luz volumetrica,
	// Oclusion ambiental, Luz de antorchas RT, Destellos de lente, Sombras RT and the hardware rows); enabled by hand after.
	RtAllOff(s);
}


int OutputHeight(const Settings& s, const PcInfo& pc) {
	int out_h = s.fullscreen && pc.screen_h > 0 ? pc.screen_h : (s.height > 0 ? s.height : 1440);
	if (int w = 0, h = 0; s.fullscreen && std::sscanf(s.resolution.c_str(), "%dx%d", &w, &h) == 2 && h > 0) out_h = h;
	return out_h;
}

void SetQualityForOutput(Settings* s, int out_h) {
	s->ibl  = out_h >= 2160 ? "completa" : "mitad";
	s->tess = out_h >= 2160 ? "alto" : "medio";
}

void ResolveQualityAuto(Settings* s, const PcInfo& pc) {
	const int out_h = OutputHeight(*s, pc);
	if (s->ibl.empty()) s->ibl = out_h >= 2160 ? "completa" : "mitad";
	if (s->tess.empty()) s->tess = out_h >= 2160 ? "alto" : "medio";
}

std::string LampLevel(const Settings& s) {
	if (s.rt_lamps == "0" || s.rt_lamps == "1" || s.rt_lamps == "2" || s.rt_lamps == "3") return s.rt_lamps;
	return s.rt_particles ? "2" : "0";
}

int LiveRtOff(const Settings& l, const Settings& n) {
	int        mask = 0;
	const auto on   = [](const std::string& v) { return !v.empty() && v != "0"; };
	// (10-28) The RT reflections pass is the screen-space Reflejos RT and the hardware one (rt_experimental) together.
	const auto refl = [&](const Settings& x) { return !x.pure_game && (on(x.rt_ss_refl) || (x.rt_reflections && x.rt_hw_exp)); };
	const auto lamp = [&](const Settings& x) { return !x.pure_game && x.rt_hw_exp && LampLevel(x) != "0"; };
	if (refl(l) && !refl(n)) mask |= 1;
	if (lamp(l) && !lamp(n)) mask |= 2;
	if (on(l.godrays) && (!on(n.godrays) || n.pure_game)) mask |= 4;
	if (on(l.ao) && (!on(n.ao) || n.pure_game)) mask |= 8;
	if (on(l.torch) && (!on(n.torch) || n.pure_game)) mask |= 16;
	if (l.lens_flare && (!n.lens_flare || n.pure_game)) mask |= 32;
	// Sombras RT: the sun pass also gives Luz volumetrica its sun, and holds the hardware sun: off live only when neither stays.
	const bool sun_hw = l.rt_sun && l.rt_hw_exp;
	if (on(l.rt_shadows) && (!on(n.rt_shadows) || n.pure_game) && !sun_hw && (!on(l.godrays) || (mask & 4) != 0)) mask |= 64;
	return l.pure_game ? 0 : mask;
}

// ---- (10-28) The Configurador's hardware-tier table ----
//
// Calibration (the owner's logs, RTX 5080 16 GB + Ryzen 9 9950X3D + 64 GB, 4K output, everything at its maximum, Detalle
// Ultra / Muy alto): the [fps] lines' mean frame time (median of each session):
//   DLSS 1/3 (0.92 Mpx rendered), no RT: 13.2 ms;   DLSS 1/2 (2.07 Mpx), no RT: 10.3-13.8 ms;
//   DLSS 1/1.333 (4.67 Mpx), no RT: 17.8 ms;          DLSS 1/3 + RT reflections: 13.5-17.2 ms;
//   DLSS 1/1.333 + RT reflections: 17.3-26 ms;         + Antorchas: +2-3 ms GPU (HWRTLIGHTS 1.3-5.1 ms);
//   Luz volumetrica 0.07-0.19 ms (GODRAYS), RT reflections pass 0.4 ms at 1/3 and 1.3-1.9 ms at 1/1.333 (+0.3 ms BLAS
//   build, +~2 ms on the render thread), VRAM 6-7 GB without RT, 10-12 GB with it.
// Fitted: frame ms = cpu_base + gpu_ms_per_mpx * Mpx (+ RT costs). For the 5080 + 9950X3D: cpu_base 9.5 ms (below ~1 Mpx
// the game is CPU bound at 11-13 ms), 1.9 ms per rendered megapixel; RT reflections 2.5 ms + 0.3 ms/Mpx; Antorchas 2.5 ms;
// Luz volumetrica 0.2 ms. The target is a mean of 14.5 ms or less (~69 FPS mean: 60 or more in the heavier places).
// Other graphics cards: gpu_ms_per_mpx scaled by 100 / their raster score (the 5080 = 100), the RT costs by
// 100 / (score * rt factor); other CPUs: cpu_base * 100 / their score.
namespace {

struct GpuTier {
	const char* key;   // upper-case substring of the name
	int         score; // raster vs the RTX 5080 = 100
	double      rt;    // ray tracing factor (RTX 40/50 = 1)
};

const GpuTier kGpuTiers[] = {
    // NVIDIA RTX 50
    {"RTX 5090", 140, 1.0}, {"RTX 5080", 100, 1.0}, {"RTX 5070 TI", 85, 1.0}, {"RTX 5070", 68, 1.0}, {"RTX 5060 TI", 48, 1.0}, {"RTX 5060", 38, 1.0},
    {"RTX 5050", 28, 1.0},
    // RTX 40
    {"RTX 4090", 125, 1.0}, {"RTX 4080 SUPER", 92, 1.0}, {"RTX 4080", 90, 1.0}, {"RTX 4070 TI SUPER", 78, 1.0}, {"RTX 4070 TI", 72, 1.0},
    {"RTX 4070 SUPER", 65, 1.0}, {"RTX 4070", 55, 1.0}, {"RTX 4060 TI", 40, 1.0}, {"RTX 4060", 32, 1.0}, {"RTX 4050", 24, 1.0},
    // RTX 30
    {"RTX 3090 TI", 70, 0.8}, {"RTX 3090", 62, 0.8}, {"RTX 3080 TI", 60, 0.8}, {"RTX 3080", 55, 0.8}, {"RTX 3070 TI", 45, 0.8},
    {"RTX 3070", 42, 0.8}, {"RTX 3060 TI", 36, 0.8}, {"RTX 3060", 28, 0.8}, {"RTX 3050", 15, 0.8},
    // RTX 20
    {"RTX 2080 TI", 40, 0.6}, {"RTX 2080 SUPER", 33, 0.6}, {"RTX 2080", 31, 0.6}, {"RTX 2070 SUPER", 29, 0.6}, {"RTX 2070", 26, 0.6},
    {"RTX 2060 SUPER", 24, 0.6}, {"RTX 2060", 20, 0.6}, {"GTX 16", 14, 0.0},
    // AMD RX 9000 / 7000 / 6000
    {"RX 9070 XT", 85, 0.8}, {"RX 9070", 75, 0.8}, {"RX 9060 XT", 42, 0.8}, {"RX 9060", 36, 0.8},
    {"RX 7900 XTX", 95, 0.55}, {"RX 7900 XT", 82, 0.55}, {"RX 7900 GRE", 68, 0.55}, {"RX 7800 XT", 60, 0.55}, {"RX 7700 XT", 50, 0.55},
    {"RX 7700", 46, 0.55}, {"RX 7600 XT", 32, 0.55}, {"RX 7600", 30, 0.55},
    {"RX 6950 XT", 62, 0.4}, {"RX 6900 XT", 58, 0.4}, {"RX 6800 XT", 55, 0.4}, {"RX 6800", 48, 0.4}, {"RX 6750 XT", 40, 0.4},
    {"RX 6700 XT", 38, 0.4}, {"RX 6700", 35, 0.4}, {"RX 6650 XT", 30, 0.4}, {"RX 6600 XT", 29, 0.4}, {"RX 6600", 25, 0.4},
    {"RX 6500", 12, 0.3}, {"RX 6400", 9, 0.3},
    // Intel Arc
    {"B580", 33, 0.6}, {"B570", 28, 0.6}, {"A770", 28, 0.6}, {"A750", 25, 0.6}, {"A580", 20, 0.6}, {"A380", 8, 0.5}, {"A310", 6, 0.5},
};

std::string Upper(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
	return s;
}

// Raster score (5080 = 100) and RT factor; unknown cards by their video memory.
void GpuScore(const PcInfo& pc, int* score, double* rt, std::string* how) {
	const std::string up = Upper(pc.gpu);
	for (const auto& g: kGpuTiers) {
		if (up.find(g.key) != std::string::npos) {
			*score = g.score;
			*rt    = g.rt;
			*how   = g.key;
			return;
		}
	}
	const int gb = static_cast<int>((pc.vram + (512ull << 20)) >> 30);
	*rt          = up.find("RTX") != std::string::npos ? 0.6 : 0.4;
	*score       = gb >= 16 ? 50 : gb >= 12 ? 35 : gb >= 8 ? 22 : 10;
	*how         = "VRAM " + std::to_string(gb) + " GB";
}

// CPU score (Ryzen 9 9950X3D = 100): the emulator's frame is mostly the CPU below ~1 Mpx.
int CpuScore(const PcInfo& pc, std::string* cls) {
	const std::string up = Upper(pc.cpu);
	const auto has = [&](const char* k) { return up.find(k) != std::string::npos; };
	int score = 0;
	if (has("X3D") && (has(" 9") || has(" 7"))) score = 100, *cls = "Ryzen X3D (7000 / 9000)";
	else if (has("X3D")) score = 85, *cls = "Ryzen X3D (5000)";
	else if (has("RYZEN") && (has(" 9 9") || has(" 7 9") || has(" 5 9") || has(" 9 7") || has(" 7 7") || has(" 5 7"))) score = 88, *cls = "Ryzen 7000 / 9000";
	else if (has("ULTRA") || has("I9-14") || has("I7-14") || has("I9-13") || has("I7-13")) score = 88, *cls = "Intel 13.a / 14.a gen. / Core Ultra";
	else if (has("RYZEN") && (has(" 5 5") || has(" 7 5") || has(" 9 5"))) score = 75, *cls = "Ryzen 5000";
	else if (has("I5-14") || has("I5-13") || has("-12")) score = 75, *cls = "Intel 12.a-14.a gen.";
	else if (has("RYZEN") && (has(" 3") || has(" 5 3") || has(" 7 3") || has(" 9 3"))) score = 60, *cls = "Ryzen 3000";
	else if (has("-11") || has("-10")) score = 60, *cls = "Intel 10.a / 11.a gen.";
	else score = 55, *cls = "otro";
	if (pc.cpu_cores > 0 && pc.cpu_cores < 6) score = score * 8 / 10;
	return score;
}

char g_fmt[64];
const char* F1(double v) {
	std::snprintf(g_fmt, sizeof(g_fmt), "%.0f", v);
	return g_fmt;
}

} // namespace

std::vector<std::string> Optimize(const PcInfo& pc, Settings* s) {
	std::vector<std::string> why;
	int         gpu = 0;
	double      rtf = 0;
	std::string gpu_how, cpu_how;
	GpuScore(pc, &gpu, &rtf, &gpu_how);
	const int    cpu   = CpuScore(pc, &cpu_how);
	const int    vram  = static_cast<int>((pc.vram + (512ull << 20)) >> 30);
	const double base  = 9.5 * 100.0 / std::max(cpu, 20);
	const double k     = 1.9 * 100.0 / std::max(gpu, 4);
	const double rtk   = rtf > 0 ? 100.0 / (std::max(gpu, 4) * rtf) : 1e9;
	const double target = 14.5;
	// The output: the chosen full-screen resolution, else the monitor's (else 1440p).
	int out_w = pc.screen_w > 0 ? pc.screen_w : 2560, out_h = pc.screen_h > 0 ? pc.screen_h : 1440;
	if (int w = 0, h = 0; s->fullscreen && std::sscanf(s->resolution.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0) out_w = w, out_h = h;
	const double out_mpx = out_w * static_cast<double>(out_h) / 1e6;
	const char*  up_kind = pc.nvidia ? "dlss" : (Upper(pc.gpu).find("ARC") != std::string::npos ? "xess" : "fsr");

	// The upscaler mode: the best image (lowest ratio, from native AA) whose frame fits without ray tracing.
	std::vector<const ResMode*> modes;
	for (int i = 0; i < kResModeCount; i++)
		if (std::strcmp(kResModes[i].upscaler, up_kind) == 0 && std::strcmp(kResModes[i].id, "fsr-900") != 0) modes.push_back(&kResModes[i]);
	std::sort(modes.begin(), modes.end(), [](const ResMode* a, const ResMode* b) { return a->value < b->value; });
	const ResMode* pick = modes.empty() ? &kResModes[0] : modes.back();
	double         ms   = 0;
	for (const ResMode* m: modes) {
		const double mpx = out_mpx / (m->value * m->value);
		ms               = base + k * mpx;
		if (ms <= target) {
			pick = m;
			break;
		}
	}
	const double mpx = out_mpx / (pick->value * pick->value);
	ms               = base + k * mpx;
	// Too slow even at the lowest: the game's own detail down (~15 % of the GPU each step).
	double detail_scale = 1.0;
	std::string detail  = gpu >= 80 ? "ultra" : gpu >= 55 ? "muy-alto" : gpu >= 30 ? "alto" : gpu >= 18 ? "normal" : "bajo";
	if (ms > target && detail != "bajo") {
		detail       = gpu >= 30 ? "normal" : "bajo";
		detail_scale = 0.75;
		ms           = base + k * mpx * detail_scale;
	}
	s->souls_mode = "ninguno";
	s->res        = pick->id;
	s->dlss_preset = pc.nvidia ? "K" : s->dlss_preset;
	s->sharpness   = pick->value >= 1.7 ? "0.5" : "0.25";
	s->detail      = detail;
	// Video memory: textures (6-7 GB used without RT at 4K on the 5080, 10-12 GB with it).
	s->texture_reduce = vram >= 12 ? "0" : vram >= 10 ? "h" : vram >= 8 ? "1" : "2";
	s->reflections    = gpu >= 80 ? "ultra" : gpu >= 30 ? "alto" : "normal";
	s->object_fade    = gpu >= 80 ? "ultra" : gpu >= 30 ? "alto" : "normal";
	s->aa_sharp       = "ultra"; // (free)
	s->shadows        = gpu >= 40 ? "maximo" : gpu >= 25 ? "alto" : "medio";
	s->shadow_edges   = gpu >= 30 ? "1" : "0"; // (Alta: almost free)
	s->quality        = cpu >= 85 ? "standard" : cpu >= 70 ? "high" : "low"; // draw distance: the CPU
	s->af16           = gpu >= 18;
	s->texture_sharpness = "1";
	s->motion_blur = s->dof = s->vignette = s->bloom = s->eye_adapt = true;
	s->pure_game = false;
	// The RAM (16 GB or less: the launcher's 16 GB profile; textures and distance at most Alto).
	s->ram16_boost = false; // (11-13) option removed
	if (s->ram16_boost) {
		if (s->texture_reduce == "0") s->texture_reduce = "h";
		if (s->quality == "standard") s->quality = "high";
	}
	// Ray tracing: what still fits, in this order: reflections, Luz volumetrica, Antorchas.
	RtAllOff(s);
	s->rt_lamps = "0";
	s->godrays  = "0";
	// (10-28, the emulator launcher's rule) The Configurador's tiers leave every RT effect at No: the user turns them on by hand.
	(void)rtk;
	const auto label = [](const Choice* items, int count, const std::string& id) {
		for (int i = 0; i < count; i++)
			if (id == items[i].id) return items[i].label;
		return items[0].label;
	};
	why.push_back(std::string("Gráfica: ") + gpu_how + " (potencia " + std::to_string(gpu) + " de 100; RTX 5080 = 100). Procesador: " + cpu_how +
	              " (" + std::to_string(cpu) + " de 100).");
	why.push_back(std::string("Reescalado: ") + pick->label + " a " + std::to_string(out_w) + "x" + std::to_string(out_h) +
	              " (el 3D a " + std::to_string(static_cast<int>(out_w / pick->value)) + "x" + std::to_string(static_cast<int>(out_h / pick->value)) + ").");
	why.push_back(std::string("Detalle ") + label(kDetails, kDetailCount, s->detail) + ", texturas " + label(kTextureReduce, kTextureReduceCount, s->texture_reduce) +
	              " (" + std::to_string(vram) + " GB de VRAM), sombras " + label(kShadows, kShadowCount, s->shadows) + ", distancia " +
	              label(kQualities, kQualityCount, s->quality) + " (según el procesador), filtrado x16 " + (s->af16 ? "sí" : "no") + ".");
	std::string rt = "Ray tracing: ";
	if (!s->rt_reflections && s->godrays == "0" && s->rt_lamps == "0" && s->rt_shadows == "0" && s->torch == "0" && !s->lens_flare &&
	    s->rt_ss_refl == "0")
		rt += "desactivado (no cabe a 60 FPS en este equipo).";
	else {
		if (s->rt_ss_refl != "0") rt += "reflejos RT bajos; ";
		if (s->rt_shadows != "0") rt += "sombras RT bajas; ";
		if (s->rt_reflections) rt += "reflejos RT por hardware; ";
		if (s->godrays != "0") rt += std::string("luz volumétrica ") + (s->godrays == "2" ? "media; " : "baja; ");
		rt.resize(rt.size() - 2);
		rt += ".";
	}
	why.push_back(rt);
	if (s->ram16_boost) why.push_back("RAM " + std::to_string(pc.ram_gb) + " GB: activadas las mejoras para equipos de 16 GB (perfil de memoria); texturas y distancia como mucho en Alto.");
	else why.push_back("RAM " + std::to_string(pc.ram_gb) + " GB: no hacen falta las mejoras para equipos de 16 GB.");
	const double fps = 1000.0 / ms;
	std::string  exp = std::string("Esperado: unos ") + F1(fps) + " FPS de media";
	if (pc.refresh_hz >= 100 && fps < pc.refresh_hz * 0.9) exp += std::string(" (tu monitor va a ") + std::to_string(pc.refresh_hz) + " Hz: la frame generation de Gráficos puede llenarlo)";
	if (base > target) exp += ". Tu procesador limita: por debajo de 60 FPS en las zonas pesadas aunque bajes gráficos";
	why.push_back(exp + ". Es una estimación: puedes cambiar cualquier opción en F2.");
	return why;
}

} // namespace Port
