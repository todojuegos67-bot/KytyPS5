// "Gran Turismo 7" PC port: the launcher's language (gt7-i18n.h).
#include "port/gt7-i18n.h"

#include <windows.h>

#include <cstring>
#include <map>
#include <string>

namespace Port::I18n {

const char* const kMenuLanguages[]     = {"es", "en"};
const char* const kMenuLanguageNames[] = {"Español", "English"};
const int         kMenuLanguageCount   = 2;

namespace {

int g_lang = 0;

const std::map<std::string, const char*>& English() {
	static const std::map<std::string, const char*> m = {
	    {"JUGAR", "PLAY"},
	    {"Salir", "Quit"},
	    {"Inicio", "Home"},
	    {"Gráficos", "Graphics"},
	    {"Rendimiento", "Performance"},
	    {"Reescalado", "Upscaling"},
	    {"Generación de fotogramas", "Frame generation"},
	    {"Mando", "Controller"},
	    {"Volante", "Wheel"},
	    {"Teclado y ratón", "Keyboard and mouse"},
	    {"Métricas", "Metrics"},
	    {"Tu PC", "Your PC"},
	    {"Desarrollador", "Developer"},
	    {"Carpeta del juego", "Game folder"},
	    {"Examinar...", "Browse..."},
	    {"Idioma del juego", "Game language"},
	    {"Idioma del menú", "Menu language"},
	    {"Nombre de usuario", "User name"},
	    {"Cerrar el lanzador al salir del juego", "Close the launcher when the game ends"},
	    {"Pantalla completa", "Fullscreen"},
	    {"Resolución", "Resolution"},
	    {"Sincronización", "Sync"},
	    {"Frecuencia de vblank", "Vblank frequency"},
	    {"Teselado", "Tessellation"},
	    {"Redibujar reflejos cada", "Redraw reflections every"},
	    {"Ocultar el cursor", "Hide the cursor"},
	    {"Texturas bindless", "Bindless textures"},
	    {"Próximamente", "Coming soon"},
	    {"Tarjeta gráfica", "Graphics card"},
	    {"Parches de CPU AMD", "AMD CPU patches"},
	    {"Prioridad del proceso", "Process priority"},
	    {"Solo los núcleos con más caché (X3D)", "Only the cores with the most cache (X3D)"},
	    {"Protección de la red zone", "Red zone protection"},
	    {"PlayGo de respaldo", "PlayGo fallback"},
	    {"NVIDIA Smooth Motion", "NVIDIA Smooth Motion"},
	    {"Color de la barra de luz", "Lightbar colour"},
	    {"Volumen del altavoz", "Speaker volume"},
	    {"Vibración", "Vibration"},
	    {"Usar volante", "Use a wheel"},
	    {"Fuerza (force feedback)", "Force feedback strength"},
	    {"Grados de giro del volante", "Wheel rotation (degrees)"},
	    {"Tope suave", "Soft lock"},
	    {"Invertir force feedback", "Reverse force feedback"},
	    {"Amortiguación por software", "Software damper and friction"},
	    {"Leer el volante en segundo plano", "Read the wheel in the background"},
	    {"Mapear volante...", "Map wheel..."},
	    {"Activar teclado y ratón", "Enable keyboard and mouse"},
	    {"Sensibilidad del ratón", "Mouse sensitivity"},
	    {"Mapa de teclas", "Key map"},
	    {"Mostrar métricas al jugar", "Show metrics while playing"},
	    {"Gráfica de FPS", "FPS graph"},
	    {"Posición", "Position"},
	    {"Añadir a Steam", "Add to Steam"},
	    {"Abrir carpeta de logs", "Open the logs folder"},
	    {"Validación de Vulkan", "Vulkan validation"},
	    {"Log de shaders a archivo", "Shader log to file"},
	    {"printf del juego a archivo", "Game printf to file"},
	    {"Consola del emulador", "Emulator console"},
	    {"Argumentos extra", "Extra arguments"},
	    {"Entorno extra", "Extra environment"},
	    {"Restaurar valores por defecto", "Restore defaults"},
	    {"El juego está en marcha.", "The game is running."},
	    {"No se encuentra eboot.bin en la carpeta del juego.", "eboot.bin is not in the game folder."},
	};
	return m;
}

} // namespace

void SetLanguage(const std::string& id) {
	g_lang = id == "en" ? 1 : 0;
}

const char* T(const char* es) {
	if (g_lang == 0 || es == nullptr) return es;
	const char* hash = std::strstr(es, "##");
	const std::string key = hash != nullptr ? std::string(es, hash) : std::string(es);
	const auto& m  = English();
	auto        it = m.find(key);
	if (it == m.end()) return es;
	if (hash == nullptr) return it->second;
	// (The id part kept: a stable string per key.)
	static std::map<std::string, std::string> joined;
	auto& j = joined[es];
	if (j.empty()) j = std::string(it->second) + hash;
	return j.c_str();
}

std::string DetectWindowsLanguage() {
	const LANGID id = GetUserDefaultUILanguage();
	return PRIMARYLANGID(id) == LANG_SPANISH ? "es" : "en";
}

int GameLanguageFor(const std::string& id) {
	return id == "en" ? 1 : 3;
}

} // namespace Port::I18n
