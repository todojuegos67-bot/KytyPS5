// The Demon's Souls PC port's English UI (src/port/ds-i18n.cpp). Plain C++: also on Linux:
//   g++ -std=c++20 -Isrc tests/PortI18nTests.cpp src/port/ds-i18n.cpp -o port_i18n_tests
#include "port/ds-i18n.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void ExpectEnglish(const char* spanish, const char* english) {
	const std::string actual = Port::I18n::T(spanish);
	if (actual != english) {
		std::printf("FAIL: English translation of \"%s\"\n  expected: %s\n  actual:   %s\n", spanish, english, actual.c_str());
		failures++;
	}
}

} // namespace

int main() {
	Port::I18n::SetLanguage("en");

	// The main launcher's folder and shader controls.
	ExpectEnglish("Elegir carpeta del juego...", "Choose game folder...");
	ExpectEnglish("Carpeta del juego: ", "Game folder: ");
	ExpectEnglish("Carpeta del juego guardada en settings.ini", "Game folder saved to settings.ini");
	ExpectEnglish("Esa carpeta no es Demon's Souls (falta eboot.bin o sce_sys).",
	              "That folder is not Demon's Souls (eboot.bin or sce_sys is missing).");
	ExpectEnglish("Elige la carpeta de Demon's Souls (la que tiene eboot.bin y sce_sys). Se guarda al momento en settings.ini.",
	              "Choose the Demon's Souls folder (the one containing eboot.bin and sce_sys). Saved to settings.ini immediately.");
	ExpectEnglish("Hilos: m\xC3\xA1ximo", "Threads: maximum");
	ExpectEnglish("Hilos: relajado", "Threads: relaxed");
	ExpectEnglish("Cambia c\xC3\xB3mo precompila: m\xC3\xA1ximo = todos los hilos de la CPU (m\xC3\xA1s r\xC3\xA1pido, el PC va cargado); relajado = pocos hilos. Si est\xC3\xA1 precompilando, sigue en el modo nuevo sin perder lo hecho.",
	              "Changes shader precompilation: maximum uses every CPU thread (faster, high PC load); relaxed uses fewer threads. If precompilation is running, it continues in the new mode without losing progress.");

	// Options added to the launcher without their English rows.
	ExpectEnglish("Modo Souls ligero +", "Souls Light+ mode");
	ExpectEnglish("Para monitores de mas de 60 Hz. Como el Modo Souls ligero (mismos graficos), pero con el juego a 40 FPS estables y frame generation de AMD x2 (80 en pantalla): mas fluido y algo menos de latencia.",
	              "For monitors above 60 Hz. Like Souls Light mode (same graphics), but with a stable 40 FPS in the game and AMD frame generation x2 (80 on screen): smoother and with slightly lower latency.");
	ExpectEnglish("Secci\xC3\xB3n", "Section");
	ExpectEnglish("No mostrar al iniciar", "Do not show at startup");
	ExpectEnglish("No se pudo guardar settings.ini", "Could not save settings.ini");
	ExpectEnglish("Archivo de paginaci\xC3\xB3n de 60 GB", "60 GB page file");
	ExpectEnglish("Pone en el disco C: un archivo de paginaci\xC3\xB3n de 16 a 60 GB (Windows pedir\xC3\xA1 permiso de administrador). Evita que el PC se reinicie al quedarse sin memoria en equipos con poca RAM. Solo ocupa espacio cuando hace falta. Reinicia el PC despu\xC3\xA9s.",
	              "Creates a page file of up to 60 GB on drive C: (Windows will ask for administrator permission). Prevents restarts caused by running out of memory on PCs with little RAM. It only uses disk space when needed. Restart the PC afterwards.");
	ExpectEnglish("Memoria total de Windows (RAM + paginaci\xC3\xB3n): %llu GB%s", "Total Windows memory (RAM + page file): %llu GB%s");
	ExpectEnglish("  -  pulsa el bot\xC3\xB3n y reinicia", "  -  click the button and restart");
	ExpectEnglish("Salta el video de PlayStation y la cinematica del principio, hasta la pantalla de titulo.",
	              "Skips the PlayStation video and the opening cinematic through to the title screen.");

	// The small F2 menu shown over the game.
	ExpectEnglish("Mostrar el lanzador la pr\xC3\xB3xima vez que abras el juego",
	              "Show the launcher the next time you open the game");
	ExpectEnglish("Volver al juego", "Back to game");
	ExpectEnglish("Cerrar el juego", "Close game");
	ExpectEnglish("F2 o Esc: volver al juego", "F2 or Esc: back to game");

	// The first-run Configurator and detected hardware.
	ExpectEnglish("Tarjeta gr\xC3\xA1" "fica", "Graphics card");
	ExpectEnglish("Procesador", "CPU");
	ExpectEnglish(" n\xC3\xBA" "cleos / ", " cores / ");
	ExpectEnglish(" hilos", " threads");
	ExpectEnglish("16 GB RAM (simulados)", "16 GB RAM (simulated)");
	ExpectEnglish("Continuar", "Continue");
	ExpectEnglish("Sigue con tus ajustes actuales.", "Keep your current settings.");
	ExpectEnglish("NO COMPATIBLE", "NOT SUPPORTED");
	ExpectEnglish("1440p: 45 FPS reales fijos (90 en pantalla con AMD x2)", "1440p: stable 45 real FPS (90 on screen with AMD x2)");

	// Direct ImGui text, trophy status and update messages.
	ExpectEnglish("Aplicar", "Apply");
	ExpectEnglish("Restablecer", "Reset");
	ExpectEnglish("(requiere reinicio)", "(needs a restart)");
	ExpectEnglish("A\xC3\xBAn no hay datos de trofeos: se cargan al iniciar el juego.", "No trophy data yet: it loads when the game starts.");
	ExpectEnglish("%d / %d trofeos", "%d / %d trophies");
	ExpectEnglish(" - sin obtener", " - not earned");
	ExpectEnglish("Version instalada: ", "Installed version: ");
	ExpectEnglish("\nUltima version del port en GitHub: ", "\nLatest port version on GitHub: ");
	ExpectEnglish("Descargando ", "Downloading ");
	ExpectEnglish("Restaurando la version ", "Restoring version ");
	ExpectEnglish(". No se ha cambiado nada.", ". Nothing was changed.");
	ExpectEnglish("  -  hecho: REINICIA EL PC", "  -  done: RESTART THE PC");

	// F8 metrics and the legacy performance HUD.
	ExpectEnglish("GPU reloj", "GPU clock");
	ExpectEnglish("CPU reloj", "CPU clock");
	ExpectEnglish("Latencia", "Latency");
	ExpectEnglish("Escala", "Scale");
	ExpectEnglish("Resol.", "Resolution");
	ExpectEnglish("Modelo", "Model");
	ExpectEnglish("FG ahora", "FG now");
	ExpectEnglish("Personalizado", "Custom");
	ExpectEnglish("reescalado del juego", "game upscaling");
	ExpectEnglish("Ultra (Reflex apagado)", "Ultra (Reflex off)");
	ExpectEnglish("frametime (ms)  40   verde media, naranja picos", "frametime (ms)  40   green average, orange peaks");
	ExpectEnglish("MODO SOULS EQUILIBRADO", "SOULS BALANCED MODE");

	// Hardware unit fragments composed with dynamic values.
	ExpectEnglish(" GB de VRAM", " GB of VRAM");
	ExpectEnglish(" a ", " at ");

	// Startup dialogs and splash status messages.
	ExpectEnglish("Solo quedan %.1f GB de RAM libres (de %.0f GB). El juego necesita unos 12 GB y puede ir a tirones o cerrarse.\n\n",
	              "Only %.1f GB of RAM free (of %.0f GB). The game needs about 12 GB and may stutter or close.\n\n");
	ExpectEnglish("Los programas que mas RAM usan ahora:\n", "Programs using the most RAM now:\n");
	ExpectEnglish("\nCierralos si puedes (navegador, Discord, etc.).\n\nAceptar: jugar igualmente    Cancelar: salir",
	              "\nClose them if you can (browser, Discord, etc.).\n\nOK: play anyway    Cancel: quit");
	ExpectEnglish("Memoria RAM", "RAM");
	ExpectEnglish("Demon's Souls: elige la carpeta del juego volcado (la que tiene eboot.bin y sce_sys)",
	              "Demon's Souls: choose the dumped game folder (the one containing eboot.bin and sce_sys)");
	ExpectEnglish("Demon's Souls ya se esta ejecutando.", "Demon's Souls is already running.");
	ExpectEnglish("No hay eboot.bin en:\n", "No eboot.bin found in:\n");
	ExpectEnglish("\n\nElige la carpeta del juego volcado (la que tiene eboot.bin y sce_sys).",
	              "\n\nChoose the dumped game folder (the one containing eboot.bin and sce_sys).");
	ExpectEnglish("Preparando shaders 1/3: la lista de shaders del juego (solo la primera vez)...",
	              "Preparing shaders 1/3: the game's shader list (first time only)...");
	ExpectEnglish("Preparando shaders 2/3: especializaciones de partidas grabadas (unos segundos)...",
	              "Preparing shaders 2/3: recorded-play specializations (a few seconds)...");
	ExpectEnglish("Preparando shaders 3/3: para esta grafica (primera vez o driver nuevo: un minuto). Despues se compilan todos los pipelines con su barra de progreso.",
	              "Preparing shaders 3/3: for this graphics card (first time or new driver: about a minute). All pipelines are then compiled with a progress bar.");
	ExpectEnglish("Comprobando la cache de shaders...", "Checking the shader cache...");
	ExpectEnglish("Guardando lo hecho...", "Saving progress...");
	ExpectEnglish("Iniciando...", "Starting...");

	if (failures == 0) std::printf("All English UI translation checks passed.\n");
	return failures == 0 ? 0 : 1;
}
