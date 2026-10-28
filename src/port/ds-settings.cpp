#include "port/ds-settings.h"

#include "port/ds-tables.h"

#include <algorithm>
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
};
const int kResModeCount = static_cast<int>(sizeof(kResModes) / sizeof(kResModes[0]));

const Choice kSharpness[] = {{"auto", "Nitidez: auto"}, {"0", "Nitidez: 0"}, {"0.25", "Nitidez: 25%"}, {"0.5", "Nitidez: 50%"}, {"0.75", "Nitidez: 75%"}, {"1", "Nitidez: 100%"}};
const int    kSharpnessCount = 6;
const Choice kDlssPresets[] = {{"auto", "DLSS auto"}, {"K", "DLSS 4 (K)"}, {"L", "DLSS 4.5 (L)"}, {"M", "DLSS 4.5 (M)"}};
const int    kDlssPresetCount = 4;
const Choice kQualities[] = {{"standard", "Maximo"}, {"high", "Alto"}, {"low", "Medio"}, {"lowest", "Bajo"}}; // draw distance (LOD)
const int    kQualityCount = 4;
const Choice kTextureReduce[] = {{"0", "Maximo"}, {"h", "Alto"}, {"1", "Medio"}, {"2", "Bajo"}};
const int    kTextureReduceCount = 4;
const Choice kShadows[] = {{"maximo", "Maximo"}, {"alto", "Alto"}, {"medio", "Medio"}, {"bajo", "Bajo"}};
const int    kShadowCount = 4;
const Choice kDetails[] = {{"ultra", "Ultra"}, {"muy-alto", "Muy alto (cine)"}, {"alto", "Alto (rendimiento de PS5)"}, {"normal", "Normal"}, {"bajo", "Bajo"}};
const int    kDetailCount = 5;
const Choice kReflections[] = {{"ultra", "Ultra"}, {"alto", "Alto (de serie)"}, {"normal", "Normal"}, {"bajo", "Bajo"}};
const int    kReflectionCount = 4;
const Choice kAaSharp[] = {{"ultra", "Ultra"}, {"alto", "Alto"}, {"normal", "Normal (de serie)"}, {"bajo", "Bajo"}, {"no", "No"}};
const int    kAaSharpCount = 5;
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
const Choice kRtLightLevels[] = {{"0", "No"}, {"1", "Media"}, {"2", "Alta"}, {"3", "Ultra"}, {"4", "Extremo"}};
const int    kRtLightLevelCount = 5;
const Choice kRtLevels[] = {{"0", "No"}, {"1", "Bajo"}, {"2", "Medio"}, {"3", "Alto"}};
const int    kRtLevelCount = 4;
const Choice kTorchLevels[] = {{"0", "No"}, {"1", "Bajo"}, {"2", "Medio"}, {"3", "Alto"}, {"4", "Híbrido"}};
const int    kTorchLevelCount = 5;
const Choice kYesNo[] = {{"0", "No"}, {"1", "Sí"}};
const int    kYesNoCount = 2;
const Choice kSoulsModes[] = {{"ninguno", "Personalizado (ningun modo)"},
                              {"ligero", "Modo Souls ligero"},
                              {"equilibrado", "Modo Souls equilibrado"},
                              {"ultra", "Modo Souls ultra"},
                              {"extremo", "Modo Souls extremo"},
                              {"low", "Modo Souls PC low profile (900p, todo en bajo)"}};
const int    kSoulsModeCount = 6;
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
	for (const char* k: {"idioma", "idioma_juego"})
		if (auto it = v.find(k); it != v.end() && !it->second.empty()) s->game_language_set = true;
	str("idioma_menu", s->menu_language);
	boo("copia_partidas", s->backup_saves);
	boo("pantalla_completa", s->fullscreen);
	num("ancho", s->width);
	num("alto", s->height);
	str("resolucion", s->resolution);
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
	str("frame_generation", s->framegen);
	boo("perfil_16gb", s->profile16);
	if (auto it = v.find("gama_baja"); it != v.end() && Bool(it->second, false)) s->souls_mode = "ligero"; // (before 10-23)
	boo("gpu_seguro", s->safe_gpu);
	str("calidad", s->quality);
	str("reducir_texturas", s->texture_reduce);
	boo("simular_vram_8gb", s->simulate_vram8);
	boo("todos_los_ccd", s->all_ccds);
	str("metricas", s->metrics);
	boo("menu_inicio", s->start_menu);
	boo("vibracion", s->rumble);
	num("vibracion_fuerza", s->rumble_gain);
	str("extra", s->extra);
	boo("antilag", s->antilag);
	// (before 10-27 "antilag" was a box, written 0 by default: the launcher's default, auto, either way)
	str("antilag_modo", s->antilag_mode);
	str("antilag_dlssg", s->antilag_dlssg);
	boo("simular_ram_16gb", s->simulate_ram16);
	boo("rt_reflejos", s->rt_reflections);
	boo("rt_sombras_sol", s->rt_sun);
	boo("rt_niebla", s->rt_fog);
	boo("rt_particulas", s->rt_particles);
	boo("rt_metal", s->rt_metal);
	boo("rt_oclusion", s->rt_ao);
	str("luz_volumetrica", s->godrays);
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
	str("rt_antorchas", s->rt_lamps);
	num("raton_velocidad_camara", s->kbm_camera);
	s->kbm_camera = std::clamp(s->kbm_camera, 5, 15);
	boo("raton_camara_instantanea", s->kbm_warp);
	boo("raton_calibrar", s->kbm_calibrate);
	boo("configurado", s->configured);
	boo("mejoras_16gb", s->ram16_boost);
	str("reflejos_rt", s->rt_ss_refl);
	str("luz_rt_pantalla", s->rt_ssgi);
	str("sombras_rt", s->rt_shadows);
	str("oclusion_ao", s->ao);
	str("luz_antorchas", s->torch);
	boo("destellos_lente", s->lens_flare);
	boo("rt_experimental", s->rt_hw_exp);
	// (Out of range: No.)
	const auto clamp_level = [](std::string& v, int max) {
		const int n = std::atoi(v.c_str());
		v           = n >= 0 && n <= max && !v.empty() && std::isdigit(static_cast<unsigned char>(v[0])) != 0 ? std::to_string(n) : "0";
	};
	clamp_level(s->rt_ss_refl, 6);
	clamp_level(s->rt_ssgi, 4);
	clamp_level(s->rt_shadows, 3);
	clamp_level(s->ao, 3);
	clamp_level(s->torch, 4);
	// (10-28) No presets or modes any more: an old modo_souls is dropped (its graphics stay as saved).
	s->souls_mode = "ninguno";
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
	  << "; Idioma del juego (consola PS5: 3 espanol, 20 espanol latino, 1 ingles EE. UU., 18 ingles RU, 2 frances, 4 aleman, 5 italiano, 7 portugues, 17 portugues Brasil, 0 japones, 11 chino simplificado)\n"
	  << "idioma_juego=" << s.language << "\n"
	  << "; Idioma de los menus: es, en, pt, fr, it, de\n"
	  << "idioma_menu=" << s.menu_language << "\n"
	  << "; Copia de seguridad de _SaveData en _SaveData-copias antes de jugar\n"
	  << "copia_partidas=" << b(s.backup_saves) << "\n\n"
	  << "[Pantalla]\n"
	  << "pantalla_completa=" << b(s.fullscreen) << "\n"
	  << "; Tamano de la ventana (sin pantalla completa)\n"
	  << "ancho=" << s.width << "\nalto=" << s.height << "\n"
	  << "; Resolucion en pantalla completa: escritorio (la de Windows) o AnchoxAlto, p. ej. 1920x1080 (el monitor cambia a ella)\n"
	  << "resolucion=" << s.resolution << "\n"
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
	  << "; fsr-aa, fsr-ucalidad, fsr-calidad, fsr-equilibrado, fsr-rendimiento (FSR 3.1),\n"
	  << "; fsr4-aa, fsr4-calidad, fsr4-equilibrado, fsr4-rendimiento, fsr4-urendimiento, fsr4-dinamico (AMD FSR 4)\n"
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
	  << "; Anti-Lag con la frame generation de AMD: auto (Anti-Lag 2 en graficas AMD, el nuestro en las demas), amd2, normal,\n"
	  << "; ultra, no. En los modos Souls ligero, equilibrado y low profile va siempre el nuestro en Ultra.\n"
	  << "antilag_modo=" << s.antilag_mode << "\n"
	  << "; Con la frame generation de NVIDIA: reflex (recomendado), normal o ultra (el nuestro, con Reflex apagado)\n"
	  << "antilag_dlssg=" << s.antilag_dlssg << "\n\n"
	  << "[Graficos]\n"
	  << "; Modos Souls: ninguno; low (PC low profile: 900p con FSR, todo en bajo, 30 reales x2 AMD); ligero (graficas de 8 GB: 30 reales x2 AMD, FSR Calidad, texturas y sombras medio);\n"
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
	  << "; Luz RT (luz rebotada en pantalla): 0 no, 1 media, 2 alta, 3 ultra, 4 extremo\n"
	  << "luz_rt_pantalla=" << s.rt_ssgi << "\n"
	  << "; Sombras RT (sombras del sol de pantalla): 0 no, 1 bajo, 2 medio, 3 alto\n"
	  << "sombras_rt=" << s.rt_shadows << "\n"
	  << "; Oclusion ambiental: 0 no, 1 bajo, 2 medio, 3 alto (por ahora cuesta varios ms: mejor en 0)\n"
	  << "oclusion_ao=" << s.ao << "\n"
	  << "; Luz de antorchas RT: 0 no, 1 bajo, 2 medio, 3 alto, 4 hibrido (alto + rayos por hardware; GPU con RT)\n"
	  << "luz_antorchas=" << s.torch << "\n"
	  << "; Destellos de lente de antorchas y fuegos: 1 si, 0 no\n"
	  << "destellos_lente=" << b(s.lens_flare) << "\n"
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
	  << "; Juego original: 1 apaga todos los efectos visuales que anade el emulador\n"
	  << "; Antorchas (luces RT de antorchas, lamparas, fuego y magia): 0 no, 1 bajo, 2 medio, 3 alto\n"
	  << "rt_antorchas=" << LampLevel(s) << "\n"
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
	  << "; Velocidad de camara con raton (5-15; 10 = el maximo del menu del juego, 11-15 extra)\n"
	  << "raton_velocidad_camara=" << s.kbm_camera << "\n"
	  << "; Camara instantanea (frame warp): la imagen gira con el raton antes de que el juego dibuje\n"
	  << "raton_camara_instantanea=" << b(s.kbm_warp) << "\n"
	  << "; 1: el proximo arranque calibra la camara del raton (kbm_curve.txt) y vuelve a 0\n"
	  << "raton_calibrar=" << b(s.kbm_calibrate) << "\n\n"
	  << "[Configurador]\n"
	  << "; 0: el configurador de hardware se muestra al arrancar\n"
	  << "configurado=" << b(s.configured) << "\n"
	  << "; Mejoras para equipos de 16 GB de RAM: perfil de 16 GB (solo en PCs de 20 GB de RAM o menos)\n"
	  << "mejoras_16gb=" << b(s.ram16_boost) << "\n\n"
	  << "[Interfaz]\n"
	  << "; Metricas en pantalla: 0 no, 1 FPS, 2 FPS + detalle\n"
	  << "metricas=" << s.metrics << "\n"
	  << "; Menu de inicio (como el launcher) antes del juego: 1 si, 0 arranca directo\n"
	  << "menu_inicio=" << b(s.start_menu) << "\n"
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


bool LowRam(const Settings& s, const PcInfo& pc) {
	// (10-28) "Mejoras para equipos de 16 GB de RAM" (on by default) on a PC of 20 GB or less, or simulated.
	return s.simulate_ram16 || (s.ram16_boost && pc.ram_gb > 0 && pc.ram_gb <= 20);
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
	std::string fg_id    = fixed_fg ? "amd-x2" : s.framegen;
	if (!pc.nvidia && fg_id.rfind("dlss", 0) == 0) {
		fg_id = "sin-fg";
		plan.notes.emplace_back("DLSS frame generation needs NVIDIA: without frame generation");
	}
	const auto* fg = FindFg(fg_id);
	for (const auto& p: Tables::kBaseEnv) env.Set(p.key, p.value);
	for (int i = 0; i < fg->env_count; i++) env.Set(fg->env[i].key, fg->env[i].value);
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
		res = FindRes(other);
		if (std::strcmp(res->id, other.c_str()) != 0) res = FindRes("fsr-calidad");
		plan.notes.emplace_back(std::string("DLSS needs NVIDIA: ") + res->label);
	}
	std::string upscaler = res->upscaler;
	double      ratio    = upscaler.empty() ? 0.0 : res->value;
	// (10-27) DLSS in Souls ligero / PC low profile: DLSS outputs 4K inside and only takes 2160 / an exact ratio (1440, 1256,
	// 1080, 720): any DLSS quality there renders at 720 (DLAA at 1080 stays).
	if (upscaler == "dlss" && pc.nvidia && (s.souls_mode == "ligero" || s.souls_mode == "low")) {
		int mode_w = 0, mode_h = 0;
		const bool mode = s.fullscreen && std::sscanf(s.resolution.c_str(), "%dx%d", &mode_w, &mode_h) == 2 && mode_h > 0;
		const int out_h = mode ? mode_h : s.fullscreen && pc.screen_h > 0 ? pc.screen_h : (s.height > 0 ? s.height : 1440);
		if (!(out_h == 1080 && ratio == 1.0)) ratio = std::round(out_h / 720.0 * 10000.0) / 10000.0;
	}
	// (10-28, the launcher's "DLSS en cualquier resolucion") DLSS only takes the 3D at 2160 / 1440 / 1256 / 1080 / 720: when
	// the chosen quality does not land on one, the nearest valid height at or below it (at least 720) instead of the game
	// starting without DLSS.
	if (upscaler == "dlss" && pc.nvidia && ratio > 0) {
		int mode_w = 0, mode_h = 0;
		const bool mode = s.fullscreen && std::sscanf(s.resolution.c_str(), "%dx%d", &mode_w, &mode_h) == 2 && mode_h > 0;
		const int  out_h = mode ? mode_h : s.fullscreen && pc.screen_h > 0 ? pc.screen_h : (s.height > 0 ? s.height : 1440);
		const int  in_h  = static_cast<int>(std::lround(out_h / ratio));
		static const int ok[] = {2160, 1440, 1256, 1080, 720};
		if (std::find(std::begin(ok), std::end(ok), in_h) == std::end(ok)) {
			int pick = 720;
			for (int v: ok)
				if (v <= in_h && v <= out_h) {
					pick = v;
					break;
				}
			ratio = std::round(out_h / static_cast<double>(pick) * 10000.0) / 10000.0;
			plan.notes.emplace_back("DLSS: the 3D at " + std::to_string(pick) + " lines (" + std::to_string(in_h) + " is not a DLSS height)");
		}
	}
	if (std::strcmp(res->id, "fsr-900") == 0 && pc.screen_h > 0) ratio = std::max(1.0, std::round(pc.screen_h / 900.0 * 10000.0) / 10000.0);

	// Launcher-souls.ps1 Start-Game's -Set switches, in its order.
	EnvList sets;
	if (low_ram) sets.emplace_back("KYTY_LOWRAM", "1"); // (10-24) the emulator's 16 GB limits
	if (s.simulate_ram16) sets.emplace_back("KYTY_SIMULATE_RAM_MB", "16384");
	if (fixed_fg) sets.emplace_back("KYTY_PAD_LATE_MS", s.souls_mode == "equilibrado" ? "29" : "43"); // (10-25) Anti-Lag+
	// (10-26) The RAM flush is gone (hitches): none of it, and out of the config too.
	for (const char* k: {"KYTY_MEMCUT", "KYTY_REBAR_STAGING", "KYTY_STAGING_RING_MB", "KYTY_RAM_TARGET_MB", "KYTY_RAM_FLUSH_SECONDS",
	                     "KYTY_RAM_FLUSH_MB", "KYTY_RAM_FLUSH_LEVEL"})
		sets.emplace_back(k, "");
	if (s.safe_gpu) sets.emplace_back("KYTY_COMPUTE_CHAIN", "0");
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
	if (s.shadow_edges == "1" || s.shadow_edges == "2" || s.shadow_edges == "3") sets.emplace_back("KYTY_SHADOW_FILTER", s.shadow_edges);
	// (10-27b) Ray tracing (hardware), the s25 launcher: only Reflejos, Sombras sol and Luces antorcha/lamparas (No / Si);
	// no hardware volumetric, metal or AO.
	{
		Env hw;
		if (s.rt_reflections) {
			hw.Set("KYTY_HWRT_REFL", "1");
			hw.Set("KYTY_HWRT_REFL_MATERIAL", "1");
		}
		if (s.rt_sun) hw.Set("KYTY_HWRT_SHADOW", "2");
		// (10-28) Antorchas: the launcher's hwlamp4 levels (KYTY_HWRT_LIGHTS 1-3).
		if (const std::string lamps = LampLevel(s); lamps != "0") {
			hw.Set("KYTY_HWRT_PARTICLES", "2");
			hw.Set("KYTY_HWRT_LIGHTS", lamps);
		}
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
	} else {
		sets.emplace_back("KYTY_RT_REFLECTIONS", "");
	}
	sets.emplace_back("KYTY_DLSSG_PRESET", "C");
	sets.emplace_back("KYTY_FG_PACING", "0");        // (10-21) the emulator's frame pacing (stuck at 20 FPS) off
	sets.emplace_back("KYTY_DS_SIXTY_STEP", "0");    // (10-22) cinematic as on PS5
	sets.emplace_back("KYTY_DS_VARIABLE_STEP", "1"); // (10-21) physics at each frame's real time
	// Upscaler sharpness: (10-24) ligero 75 %, (10-24) PC low profile 100 %.
	std::string sharp = s.sharpness == "auto" ? std::string() : s.sharpness;
	if (s.souls_mode == "low") sharp = "1";
	else if (s.souls_mode == "ligero") sharp = "0.75";
	if (!sharp.empty()) sets.emplace_back("KYTY_UPSCALE_SHARPNESS", sharp);
	if (fixed_fg) sets.emplace_back("KYTY_FLIP_FPS", s.souls_mode == "equilibrado" ? "45" : "30");
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
	if (std::strcmp(fg->id, "amd-x4") == 0) sets.emplace_back("KYTY_VRR_CAP_FPS", "224");
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
	sets.emplace_back("KYTY_AO", !s.pure_game && (s.ao == "1" || s.ao == "2" || s.ao == "3") ? s.ao : "0"); // (ao-1028)
	const bool godrays = !s.pure_game && (s.godrays == "1" || s.godrays == "2" || s.godrays == "3");
	const bool shadows = !s.pure_game && (s.rt_shadows == "1" || s.rt_shadows == "2" || s.rt_shadows == "3");
	if (godrays) sets.emplace_back("KYTY_GODRAYS", s.godrays);
	if (shadows) {
		static const char* strength[] = {"0", "0.35", "0.5", "0.7"};
		sets.emplace_back("KYTY_RT_SHADOWS", s.rt_shadows);
		sets.emplace_back("KYTY_RT_SHADOWS_STRENGTH", strength[std::atoi(s.rt_shadows.c_str())]);
		sets.emplace_back("KYTY_RT_SUN", "0");
		sets.emplace_back("KYTY_HWRT_SHADOW", "0");
	} else if (godrays) {
		sets.emplace_back("KYTY_RT_SHADOWS", "1");
		sets.emplace_back("KYTY_RT_SHADOWS_STRENGTH", "0");
		sets.emplace_back("KYTY_RT_SUN", "0");
	} else {
		sets.emplace_back("KYTY_RT_SHADOWS", "");
		sets.emplace_back("KYTY_RT_SHADOWS_STRENGTH", "");
	}
	// (torch-1028) Luz de antorchas RT (screen-space) Bajo / Medio / Alto; Hibrido (torch-shadow-1028): Alto + the hardware
	// scene (TLAS) for the torch light only (the other hardware passes off without rt_experimental). Destellos de lente.
	if (!s.pure_game && s.torch != "0" && !s.torch.empty()) {
		sets.emplace_back("KYTY_TORCH", std::to_string(std::min(std::atoi(s.torch.c_str()), 3)));
		if (s.torch == "4") {
			sets.emplace_back("KYTY_TORCH_HW", "1");
			sets.emplace_back("KYTY_RT_HW_EXPERIMENTAL", "1");
			sets.emplace_back("KYTY_HWRT", "1");
			if (!s.rt_hw_exp)
				for (const char* k: {"KYTY_HWRT_REFL", "KYTY_HWRT_SHADOW", "KYTY_HWRT_LIGHTS", "KYTY_HWRT_PARTICLES", "KYTY_HWRT_GI", "KYTY_HWRT_AO",
				                     "KYTY_HWRT_CONTACT", "KYTY_HWRT_SOFTSUN", "KYTY_HWRT_WATER", "KYTY_HWRT_VOLSHADOW", "KYTY_RT_SUN"})
					sets.emplace_back(k, "0");
		}
	}
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
	if (!s.pure_game && s.lens_flare) sets.emplace_back("KYTY_TORCH_FLARE", "1");
	// Its vblank: 840 Hz with DLSS and AMD dynamic frame generation, 360 with the other frame generation, 960 unlocked
	// without, else 60.
	const bool vb840  = dlss_fg || std::strcmp(fg->id, "amd-dinamico") == 0;
	const int  vblank = with_fg ? (vb840 ? 840 : 360) : (unlocked ? 960 : 60);
	plan.vblank       = vblank;
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
	if (upscaler == "fsr4" && res->fsr4_preset != nullptr) env.Set("KYTY_FSR4_PRESET", res->fsr4_preset); // (run-windows -Fsr4Preset)
	if (!s.aspect.empty() && s.aspect != "16:9") env.Set("KYTY_RENDER_ASPECT", s.aspect);
	// -ForceMode performance (10-25: 4K no longer forces the cinematic mode: it held the game at 30 real).
	// (10-28, the emulator launcher) A 4K output starts the game in Cinematico, the other outputs in Rendimiento.
	if (demons) {
		int out_h = s.fullscreen && pc.screen_h > 0 ? pc.screen_h : (s.height > 0 ? s.height : 1440);
		if (int w = 0, h = 0; s.fullscreen && std::sscanf(s.resolution.c_str(), "%dx%d", &w, &h) == 2 && h > 0) out_h = h;
		env.Set("KYTY_DS_FORCE_MODE", out_h >= 2160 ? "cinematic" : "performance");
	}
	static const std::map<std::string, std::string> convars = {
	    {"high", "r_forceViewportHeightForPixelSizeCalc=1080"},
	    {"low", "r_forceViewportHeightForPixelSizeCalc=720"},
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
		    {"refl_bajo", "r_ssrEnableOverrides=true;r_ssrMaxIterations=32;r_ssrMaxRayLength=300;r_ssrCutoffDistance=250"},
		    {"fade_ultra", "r_fadeOutInstanceOverDistance=35"},
		    {"fade_normal", "r_fadeOutInstanceOverDistance=10"},
		    {"fade_bajo", "r_fadeOutInstanceOverDistance=6"},
		    {"aa_ultra", "r_aaSharpenStrengthStatic=5;r_aaSharpenStrengthMotion=5;r_aaHistorySharpeningStrength=1.0"},
		    {"aa_alto", "r_aaSharpenStrengthStatic=3.5;r_aaSharpenStrengthMotion=3.5;r_aaHistorySharpeningStrength=0.9"},
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
		    {"bajo", "r_virtualGeometryNumShadowCascades=1;sunShadowMaxDistance=125;sunShadowCascadeDistance0=125;r_iblResolution=HalfResX;"
		             "r_tessellationDensity=4.0;r_tessellationSmallestEdgeLength=0.08;r_tessellationFadeOutStart=6.0;r_tessellationFadeOutRange=2.0"}};
		if (auto it = details.find(s.detail); it != details.end()) env.AppendConvars(it->second);
		static const std::map<std::string, std::string> shadows = {
		    {"alto", "sunShadowMaxDistance=190;sunShadowLightDistance=1000"},
		    {"medio", "sunShadowMaxDistance=125;sunShadowLightDistance=1000"},
		    {"bajo", "sunShadowMaxDistance=60;sunShadowLightDistance=1000;r_enableAmbientShadowing=false;r_enableSpotShadows=false"}};
		if (auto it = shadows.find(s.shadows); it != shadows.end()) env.AppendConvars(it->second);
	}
	env.Set("KYTY_TEXTURE_REDUCE", texture == "h" || texture == "1" || texture == "2" ? texture : "0"); // (an old "auto": Maximo)
	env.Default("KYTY_UNMAP_PROTECT_SKIP", "1");
	env.Set("KYTY_DRIVER_CACHE_KEY", "1f3c29b70e536c8f5a5e812e793b33ac2eb6858b4810ed98ca2a51b9fa5eed97");
	env.Set("KYTY_SHADER_WARMUP_ONLY", "");
	// (10-27b) The shader warmup and prefetch run beside the game's intro and title screen: their progress line stays on
	// screen until they finish ("Preparando shaders X/Y"; the launcher's 6 s hid it).
	env.Set("KYTY_SHADER_WARMUP", "1");
	env.Set("KYTY_SHADER_WARMUP_BACKGROUND", "");
	env.Set("KYTY_SHADER_PREFETCH", "");
	env.Set("KYTY_PREFETCH_RELEASE", "");
	env.Set("KYTY_STATUS_HUD_SECONDS", "0");
	env.Default("KYTY_SHADER_WARMUP_SECONDS", "60");
	env.Default("KYTY_TABLE_STORE_BUDGET", "8");
	env.Default("KYTY_HITCH_LOG_MS", "40");
	if (s.aspect_fit) env.Set("KYTY_PRESENT_ASPECT", "fit");
	env.Default("KYTY_ASYNC_DRAW_PIPELINES", "1");
	// -Ram16 (10-24): "Optimizacion equipos 16 GB RAM" on a PC of 20 GB of RAM or less (or simulated) only.
	if (low_ram) { // (10-27b) on low-RAM PCs only, whatever the old box says
		env.Default("KYTY_RESIDENT_MB", "6144");
		env.Default("KYTY_PREFETCH_MIN_AVAIL_MB", "1536");
		env.Default("KYTY_BACKING_FILE", "0");
	}
	if (!upscaler.empty()) env.Set("KYTY_UPSCALER", upscaler);
	for (const auto& [k, v]: sets) env.Set(k, v);
	// Juego original (the launcher's $pureGame): no visual pass the emulator adds; performance, upscaler, frame generation
	// and latency the same.
	if (s.pure_game) {
		static const char* fx[] = {"KYTY_HWRT", "KYTY_RT_REFLECTIONS", "KYTY_RT_SHADOWS", "KYTY_RT_SUN", "KYTY_RT_PARTICLES", "KYTY_RT_LOCAL",
		                           "KYTY_GODRAYS", "KYTY_SHINE", "KYTY_SPEC_OCCLUSION", "KYTY_GTAO", "KYTY_SSGI", "KYTY_SSR", "KYTY_CONTACT_SHADOWS",
		                           "KYTY_AO", "KYTY_TORCH"};
		for (auto& [k, v]: env.list)
			for (const char* p: fx)
				if (k.rfind(p, 0) == 0) v.clear();
		for (const char* k: {"KYTY_HWRT", "KYTY_HWRT_REFL", "KYTY_HWRT_SHADOW", "KYTY_HWRT_LIGHTS", "KYTY_HWRT_PARTICLES", "KYTY_RT_REFLECTIONS",
		                     "KYTY_RT_SHADOWS", "KYTY_CONTACT_SHADOWS", "KYTY_RT_SUN", "KYTY_RT_PARTICLES", "KYTY_GODRAYS", "KYTY_SHINE",
		                     "KYTY_SPEC_OCCLUSION", "KYTY_GTAO", "KYTY_SSGI", "KYTY_SSR", "KYTY_HWRT_GI", "KYTY_HWRT_AO", "KYTY_HWRT_CONTACT",
		                     "KYTY_HWRT_SOFTSUN", "KYTY_HWRT_WATER", "KYTY_HWRT_VOLSHADOW", "KYTY_AO", "KYTY_TORCH", "KYTY_TORCH_FLARE",
		                     "KYTY_TORCH_HW", "KYTY_RT_HW_EXPERIMENTAL"})
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
	if ((s.kbm || s.kbm_calibrate) && demons) env.AppendConvars("cp11_Setting_CameraSpeed=" + std::to_string(std::clamp(s.kbm_camera, 5, 15)));
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
	       l.rt_ao != n.rt_ao || l.godrays != n.godrays || l.pure_game != n.pure_game || l.motion_blur != n.motion_blur || l.dof != n.dof || l.vignette != n.vignette ||
	       l.bloom != n.bloom || l.eye_adapt != n.eye_adapt || l.af16 != n.af16 || l.texture_sharpness != n.texture_sharpness ||
	       l.kbm != n.kbm || l.kbm_sens != n.kbm_sens || l.kbm_antidz != n.kbm_antidz || l.kbm_map != n.kbm_map ||
	       l.pad_map != n.pad_map || l.kbm_camera != n.kbm_camera || l.kbm_warp != n.kbm_warp || l.ram16_boost != n.ram16_boost || l.profile16 != n.profile16 || l.simulate_ram16 != n.simulate_ram16 ||
	       l.safe_gpu != n.safe_gpu || l.quality != n.quality || l.texture_reduce != n.texture_reduce ||
	       l.simulate_vram8 != n.simulate_vram8 || l.all_ccds != n.all_ccds || l.extra != n.extra ||
	       l.rt_ss_refl != n.rt_ss_refl || l.rt_ssgi != n.rt_ssgi || l.rt_shadows != n.rt_shadows || l.ao != n.ao ||
	       l.torch != n.torch || l.lens_flare != n.lens_flare || l.rt_hw_exp != n.rt_hw_exp ||
	       (l.sharpness == "auto") != (n.sharpness == "auto");
}

bool FixedFg(const Settings& s) {
	return s.souls_mode == "ligero" || s.souls_mode == "equilibrado" || s.souls_mode == "low";
}

void RtAllOff(Settings* s) {
	s->rt_reflections = s->rt_sun = s->rt_fog = s->rt_particles = s->rt_metal = s->rt_ao = false;
	// (10-28) and the screen-space set / new passes (rt_experimental itself stays as chosen)
	s->rt_ss_refl = s->rt_ssgi = s->rt_shadows = s->ao = s->torch = s->godrays = s->rt_lamps = "0";
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
	if (preset == "ligero") {
		// Set-PresetLight; FSR Calidad recommended (10-25); sharpness 75 %.
		s->texture_reduce = "1";
		s->quality        = "high";
		s->shadows        = "medio";
		s->detail         = "normal";
		s->aa_sharp       = "alto";
		s->object_fade    = "normal";
		s->reflections    = "normal";
		s->res            = "fsr-calidad";
		s->sharpness      = "0.75";
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
		s->res       = "dlss-calidad";
		s->dlss_preset = "K";
	} else if (preset == "ultra") {
		max();
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
	// (10-28, the emulator launcher's rule) Every mode / preset: every RT effect No (Reflejos RT, Luz RT, Luz volumetrica,
	// Oclusion ambiental, Luz de antorchas RT, Destellos de lente, Sombras RT and the hardware rows); enabled by hand after.
	RtAllOff(s);
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
	s->ram16_boost = pc.ram_gb > 0 && pc.ram_gb <= 20;
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
		if (s->torch != "0") rt += "luz de antorchas RT baja; ";
		if (s->lens_flare) rt += "destellos de lente; ";
		if (s->rt_reflections) rt += "reflejos RT por hardware; ";
		if (s->godrays != "0") rt += std::string("luz volumétrica ") + (s->godrays == "2" ? "media; " : "baja; ");
		if (s->rt_lamps != "0") rt += "antorchas RT por hardware; ";
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
	why.push_back(exp + ". Es una estimación: puedes cambiar cualquier opción en F1.");
	return why;
}

} // namespace Port
