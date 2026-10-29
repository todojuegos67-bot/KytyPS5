// "Demon's Souls" PC port: the menus' strings (ds-i18n.h). Columns: es (the key, as in the code), en, pt-BR, fr, it, de.
#include "port/ds-i18n.h"

#include <cstring>
#include <map>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Port::I18n {

const char* const kMenuLanguages[]     = {"es", "en", "pt", "fr", "it", "de"};
const char* const kMenuLanguageNames[] = {"Espa\xC3\xB1ol", "English", "Portugu\xC3\xAAs (Brasil)", "Fran\xC3\xA7" "ais", "Italiano", "Deutsch"};
const int         kMenuLanguageCount   = 6;

namespace {

int g_lang = 0;

struct Row {
	const char* t[6];
};

// clang-format off
const Row kRows[] = {
	// Tabs and titles
	{{"Imagen", "Image", "Imagem", "Image", "Immagine", "Bild"}},
	{{"Frame generation", "Frame generation", "Gera\xC3\xA7\xC3\xA3o de quadros", "G\xC3\xA9n\xC3\xA9ration d'images", "Generazione di frame", "Frame-Generierung"}},
	{{"Calidad grafica", "Graphics quality", "Qualidade gr\xC3\xA1" "fica", "Qualit\xC3\xA9 graphique", "Qualit\xC3\xA0 grafica", "Grafikqualit\xC3\xA4t"}},
	{{"Ray tracing", "Ray tracing", "Ray tracing", "Ray tracing", "Ray tracing", "Raytracing"}},
	{{"Modos Souls", "Souls modes", "Modos Souls", "Modes Souls", "Modalit\xC3\xA0 Souls", "Souls-Modi"}},
	{{"Pantalla y otros", "Display and more", "Tela e outros", "\xC3\x89" "cran et autres", "Schermo e altro", "Anzeige und mehr"}},
	{{"Rendimiento / PC", "Performance / PC", "Desempenho / PC", "Performances / PC", "Prestazioni / PC", "Leistung / PC"}},
	{{"Controles", "Controls", "Controles", "Commandes", "Controlli", "Steuerung"}},
	{{"Tu equipo", "Your PC", "Seu PC", "Votre PC", "Il tuo PC", "Dein PC"}},
	{{"OPCIONES", "OPTIONS", "OP\xC3\x87\xC3\x95" "ES", "OPTIONS", "OPZIONI", "OPTIONEN"}},
	{{"TU EQUIPO", "YOUR PC", "SEU PC", "VOTRE PC", "IL TUO PC", "DEIN PC"}},
	// Rows
	{{"Reescalado", "Upscaling", "Upscaling", "Mise \xC3\xA0 l'\xC3\xA9" "chelle", "Upscaling", "Hochskalierung"}},
	{{"Preset DLSS", "DLSS preset", "Preset DLSS", "Pr\xC3\xA9r\xC3\xA9glage DLSS", "Preset DLSS", "DLSS-Preset"}},
	{{"Nitidez", "Sharpness", "Nitidez", "Nettet\xC3\xA9", "Nitidezza", "Sch\xC3\xA4rfe"}},
	{{"Nitidez AA", "AA sharpness", "Nitidez AA", "Nettet\xC3\xA9 AA", "Nitidezza AA", "AA-Sch\xC3\xA4rfe"}},
	{{"Anti-Lag", "Anti-Lag", "Anti-Lag", "Anti-Lag", "Anti-Lag", "Anti-Lag"}},
	{{"Texturas", "Textures", "Texturas", "Textures", "Texture", "Texturen"}},
	{{"Texturas (nitidez)", "Textures (sharpness)", "Texturas (nitidez)", "Textures (nettet\xC3\xA9)", "Texture (nitidezza)", "Texturen (Sch\xC3\xA4rfe)"}},
	{{"Distancia", "Draw distance", "Dist\xC3\xA2ncia", "Distance", "Distanza", "Sichtweite"}},
	{{"Sombras", "Shadows", "Sombras", "Ombres", "Ombre", "Schatten"}},
	{{"Detalle", "Detail", "Detalhe", "D\xC3\xA9tail", "Dettaglio", "Detail"}},
	{{"Reflejos", "Reflections", "Reflexos", "R\xC3\xA9" "flexions", "Riflessi", "Spiegelungen"}},
	{{"Desvanecido objetos", "Object fade", "Desvanecimento de objetos", "Fondu des objets", "Dissolvenza oggetti", "Objekt-Ausblendung"}},
	{{"Bordes sombras", "Shadow edges", "Bordas das sombras", "Bords des ombres", "Bordi delle ombre", "Schattenkanten"}},
	{{"Sombra rincones", "Corner shadows", "Sombra nos cantos", "Ombre des coins", "Ombra negli angoli", "Eckenschatten"}},
	{{"Desenfoque de movimiento", "Motion blur", "Desfoque de movimento", "Flou de mouvement", "Sfocatura di movimento", "Bewegungsunsch\xC3\xA4rfe"}},
	{{"Profundidad de campo", "Depth of field", "Profundidade de campo", "Profondeur de champ", "Profondit\xC3\xA0 di campo", "Tiefensch\xC3\xA4rfe"}},
	{{"Vineteado", "Vignette", "Vinheta", "Vignettage", "Vignettatura", "Vignette"}},
	{{"Resplandor (bloom)", "Bloom", "Bloom", "Bloom", "Bloom", "Bloom"}},
	{{"Adaptacion del ojo", "Eye adaptation", "Adapta\xC3\xA7\xC3\xA3o do olho", "Adaptation de l'\xC5\x93il", "Adattamento dell'occhio", "Augenanpassung"}},
	{{"Filtrado x16", "16x filtering", "Filtragem 16x", "Filtrage x16", "Filtro x16", "16x-Filterung"}},
	{{"Sombras sol", "Sun shadows", "Sombras do sol", "Ombres du soleil", "Ombre del sole", "Sonnenschatten"}},
	{{"Luces antorcha/lamparas", "Torch/lamp lights", "Luzes de tochas/l\xC3\xA2mpadas", "Lumi\xC3\xA8res torches/lampes", "Luci torce/lampade", "Fackel-/Lampenlicht"}},
	{{"Luz volumetrica", "Volumetric light", "Luz volum\xC3\xA9trica", "Lumi\xC3\xA8re volum\xC3\xA9trique", "Luce volumetrica", "Volumetrisches Licht"}},
	{{"Juego original (sin efectos anadidos)", "Original game (no added effects)", "Jogo original (sem efeitos adicionados)", "Jeu original (sans effets ajout\xC3\xA9s)", "Gioco originale (senza effetti aggiunti)", "Originalspiel (keine zus\xC3\xA4tzlichen Effekte)"}},
	{{"Modo Souls", "Souls mode", "Modo Souls", "Mode Souls", "Modalit\xC3\xA0 Souls", "Souls-Modus"}},
	{{"Pantalla completa", "Fullscreen", "Tela cheia", "Plein \xC3\xA9" "cran", "Schermo intero", "Vollbild"}},
	{{"Metricas", "Metrics", "M\xC3\xA9tricas", "Statistiques", "Statistiche", "Messwerte"}},
	{{"Monitor", "Monitor", "Monitor", "Moniteur", "Monitor", "Monitor"}},
	{{"Mantener 16:9 (bandas negras)", "Keep 16:9 (black bars)", "Manter 16:9 (barras pretas)", "Garder 16:9 (bandes noires)", "Mantieni 16:9 (bande nere)", "16:9 beibehalten (schwarze Balken)"}},
	{{"Tope de FPS (G-Sync, sin FG)", "FPS cap (G-Sync, no FG)", "Limite de FPS (G-Sync, sem FG)", "Limite FPS (G-Sync, sans FG)", "Limite FPS (G-Sync, senza FG)", "FPS-Limit (G-Sync, ohne FG)"}},
	{{"Idioma", "Game language", "Idioma do jogo", "Langue du jeu", "Lingua del gioco", "Spielsprache"}},
	{{"Idioma del menu", "Menu language", "Idioma do menu", "Langue du menu", "Lingua del menu", "Men\xC3\xBCsprache"}},
	{{"Copia de las partidas al arrancar", "Back up saves at start", "Copiar saves ao iniciar", "Sauvegarder les parties au d\xC3\xA9marrage", "Copia dei salvataggi all'avvio", "Spielst\xC3\xA4nde beim Start sichern"}},
	{{"Optimizacion equipos 16 GB RAM", "16 GB RAM PC optimization", "Otimiza\xC3\xA7\xC3\xA3o para PCs de 16 GB", "Optimisation PC 16 Go RAM", "Ottimizzazione PC 16 GB RAM", "Optimierung f\xC3\xBCr 16-GB-PCs"}},
	{{"Modo GPU seguro", "Safe GPU mode", "Modo GPU seguro", "Mode GPU s\xC3\xBBr", "Modalit\xC3\xA0 GPU sicura", "Sicherer GPU-Modus"}},
	{{"Usar todos los CCD (Ryzen de dos CCD)", "Use all CCDs (dual-CCD Ryzen)", "Usar todos os CCDs (Ryzen com dois CCDs)", "Utiliser tous les CCD (Ryzen bi-CCD)", "Usa tutti i CCD (Ryzen a due CCD)", "Alle CCDs nutzen (Ryzen mit zwei CCDs)"}},
	{{"Teclado y raton", "Keyboard and mouse", "Teclado e mouse", "Clavier et souris", "Tastiera e mouse", "Tastatur und Maus"}},
	{{"Vibracion", "Vibration", "Vibra\xC3\xA7\xC3\xA3o", "Vibration", "Vibrazione", "Vibration"}},
	{{"Fuerza de la vibracion", "Vibration strength", "For\xC3\xA7" "a da vibra\xC3\xA7\xC3\xA3o", "Force de vibration", "Intensit\xC3\xA0 vibrazione", "Vibrationsst\xC3\xA4rke"}},
	{{"Sensibilidad", "Sensitivity", "Sensibilidade", "Sensibilit\xC3\xA9", "Sensibilit\xC3\xA0", "Empfindlichkeit"}},
	{{"Antizona muerta", "Anti-deadzone", "Antizona morta", "Anti-zone morte", "Anti-zona morta", "Anti-Totzone"}},
	{{"Multiplicador (en vivo)", "Multiplier (live)", "Multiplicador (ao vivo)", "Multiplicateur (en direct)", "Moltiplicatore (dal vivo)", "Multiplikator (live)"}},
	// Choices
	{{"No", "Off", "N\xC3\xA3o", "Non", "No", "Aus"}},
	{{"Si", "On", "Sim", "Oui", "S\xC3\xAC", "An"}},
	{{"Bajo", "Low", "Baixo", "Bas", "Basso", "Niedrig"}},
	{{"Medio", "Medium", "M\xC3\xA9" "dio", "Moyen", "Medio", "Mittel"}},
	{{"Alto", "High", "Alto", "\xC3\x89lev\xC3\xA9", "Alto", "Hoch"}},
	{{"Alta", "High", "Alta", "\xC3\x89lev\xC3\xA9" "e", "Alta", "Hoch"}},
	{{"Muy alta", "Very high", "Muito alta", "Tr\xC3\xA8s \xC3\xA9lev\xC3\xA9" "e", "Molto alta", "Sehr hoch"}},
	{{"Maxima", "Maximum", "M\xC3\xA1xima", "Maximale", "Massima", "Maximal"}},
	{{"Maximo", "Maximum", "M\xC3\xA1ximo", "Maximum", "Massimo", "Maximum"}},
	{{"Normal", "Normal", "Normal", "Normal", "Normale", "Normal"}},
	{{"Normal (de serie)", "Normal (default)", "Normal (padr\xC3\xA3o)", "Normal (par d\xC3\xA9" "faut)", "Normale (predefinito)", "Normal (Standard)"}},
	{{"Alto (de serie)", "High (default)", "Alto (padr\xC3\xA3o)", "\xC3\x89lev\xC3\xA9 (par d\xC3\xA9" "faut)", "Alto (predefinito)", "Hoch (Standard)"}},
	{{"Muy alto (cine)", "Very high (cinematic)", "Muito alto (cinema)", "Tr\xC3\xA8s \xC3\xA9lev\xC3\xA9 (cin\xC3\xA9ma)", "Molto alto (cinema)", "Sehr hoch (Kino)"}},
	{{"Sin frame generation", "No frame generation", "Sem gera\xC3\xA7\xC3\xA3o de quadros", "Sans g\xC3\xA9n\xC3\xA9ration d'images", "Senza generazione di frame", "Ohne Frame-Generierung"}},
	{{"Personalizado (ningun modo)", "Custom (no mode)", "Personalizado (nenhum modo)", "Personnalis\xC3\xA9 (aucun mode)", "Personalizzato (nessuna modalit\xC3\xA0)", "Benutzerdefiniert (kein Modus)"}},
	{{"Texturas: normales", "Textures: normal", "Texturas: normais", "Textures : normales", "Texture: normali", "Texturen: normal"}},
	{{"Texturas: nitidas", "Textures: sharp", "Texturas: n\xC3\xADtidas", "Textures : nettes", "Texture: nitide", "Texturen: scharf"}},
	{{"Texturas: extra nitidas", "Textures: extra sharp", "Texturas: extra n\xC3\xADtidas", "Textures : tr\xC3\xA8s nettes", "Texture: extra nitide", "Texturen: extra scharf"}},
	{{"FPS + detalle", "FPS + detail", "FPS + detalhe", "FPS + d\xC3\xA9tail", "FPS + dettaglio", "FPS + Details"}},
	// Buttons and footer
	{{"JUGAR", "PLAY", "JOGAR", "JOUER", "GIOCA", "SPIELEN"}},
	{{"JUGAR (sin esperar)", "PLAY (don't wait)", "JOGAR (sem esperar)", "JOUER (sans attendre)", "GIOCA (senza attendere)", "SPIELEN (nicht warten)"}},
	{{"Actualizar version", "Update", "Atualizar vers\xC3\xA3o", "Mettre \xC3\xA0 jour", "Aggiorna versione", "Aktualisieren"}},
	{{"Volver a la version anterior", "Previous version", "Voltar \xC3\xA0 vers\xC3\xA3o anterior", "Version pr\xC3\xA9" "c\xC3\xA9" "dente", "Versione precedente", "Vorherige Version"}},
	{{"Salir", "Quit", "Sair", "Quitter", "Esci", "Beenden"}},
	{{"Salir del juego", "Quit game", "Sair do jogo", "Quitter le jeu", "Esci dal gioco", "Spiel beenden"}},
	{{"Guardar y volver", "Save and return", "Salvar e voltar", "Enregistrer et revenir", "Salva e torna", "Speichern und zur\xC3\xBC" "ck"}},
	{{"Reiniciar ahora", "Restart now", "Reiniciar agora", "Red\xC3\xA9marrer", "Riavvia ora", "Jetzt neu starten"}},
	{{"Descartar", "Discard", "Descartar", "Annuler", "Annulla", "Verwerfen"}},
	{{"Cancelar", "Cancel", "Cancelar", "Annuler", "Annulla", "Abbrechen"}},
	{{"Restaurar", "Restore", "Restaurar", "Restaurer", "Ripristina", "Wiederherstellen"}},
	{{"Cambiar...", "Change...", "Alterar...", "Changer...", "Cambia...", "\xC3\x84ndern..."}},
	{{"Detener", "Stop", "Parar", "Arr\xC3\xAAter", "Ferma", "Stoppen"}},
	{{"Preparar shaders", "Prepare shaders", "Preparar shaders", "Pr\xC3\xA9parer les shaders", "Prepara shader", "Shader vorbereiten"}},
	{{"Carpeta del juego:", "Game folder:", "Pasta do jogo:", "Dossier du jeu :", "Cartella del gioco:", "Spielordner:"}},
	{{"(sin elegir)", "(not chosen)", "(n\xC3\xA3o escolhida)", "(non choisi)", "(non scelta)", "(nicht gew\xC3\xA4hlt)"}},
	{{"(no valida: JUGAR te la pide)", "(not valid: PLAY asks for it)", "(inv\xC3\xA1lida: JOGAR vai pedir)", "(invalide : JOUER la demandera)", "(non valida: GIOCA la chieder\xC3\xA0)", "(ung\xC3\xBCltig: SPIELEN fragt danach)"}},
	{{"Los cambios se aplican al pulsar JUGAR (sin reiniciar). F1 dentro del juego: opciones en vivo.",
	  "Changes apply when you press PLAY (no restart). F1 in game: live options.",
	  "As mudan\xC3\xA7" "as valem ao apertar JOGAR (sem reiniciar). F1 no jogo: op\xC3\xA7\xC3\xB5" "es ao vivo.",
	  "Les changements s'appliquent en appuyant sur JOUER (sans red\xC3\xA9marrer). F1 en jeu : options en direct.",
	  "Le modifiche si applicano premendo GIOCA (senza riavviare). F1 in gioco: opzioni dal vivo.",
	  "\xC3\x84nderungen gelten beim Dr\xC3\xBC" "cken von SPIELEN (kein Neustart). F1 im Spiel: Live-Optionen."}},
	{{"Shaders: la cache de esta grafica ya esta preparada.", "Shaders: this graphics card's cache is ready.", "Shaders: o cache desta placa j\xC3\xA1 est\xC3\xA1 pronto.",
	  "Shaders : le cache de cette carte graphique est pr\xC3\xAAt.", "Shader: la cache di questa scheda \xC3\xA8 pronta.", "Shader: Der Cache dieser Grafikkarte ist bereit."}},
	{{"Grafica", "Graphics card", "Placa de v\xC3\xAD" "deo", "Carte graphique", "Scheda video", "Grafikkarte"}},
	{{"Procesador y RAM", "CPU and RAM", "Processador e RAM", "Processeur et RAM", "Processore e RAM", "Prozessor und RAM"}},
	{{"Modo recomendado", "Recommended mode", "Modo recomendado", "Mode recommand\xC3\xA9", "Modalit\xC3\xA0 consigliata", "Empfohlener Modus"}},
	{{"Esperado (estimacion)", "Expected (estimate)", "Esperado (estimativa)", "Attendu (estimation)", "Previsto (stima)", "Erwartet (Sch\xC3\xA4tzung)"}},
	// (10-28) ds-torch-1028's Ray tracing rows
	{{"Reflejos RT", "RT reflections", "Reflexos RT", "R\xC3\xA9" "flexions RT", "Riflessi RT", "RT-Reflexionen"}},
	{{"Luz RT", "RT lighting", "Luz RT", "\xC3\x89" "clairage RT", "Luce RT", "RT-Beleuchtung"}},
	{{"Luz volum\xC3\xA9trica", "Volumetric light", "Luz volum\xC3\xA9trica", "Lumi\xC3\xA8re volum\xC3\xA9trique", "Luce volumetrica", "Volumetrisches Licht"}},
	{{"Oclusi\xC3\xB3n ambiental", "Ambient occlusion", "Oclus\xC3\xA3o de ambiente", "Occlusion ambiante", "Occlusione ambientale", "Umgebungsverdeckung"}},
	{{"Iluminaci\xC3\xB3n IBL", "IBL lighting", "Ilumina\xC3\xA7\xC3\xA3o IBL", "\xC3\x89" "clairage IBL", "Illuminazione IBL", "IBL-Beleuchtung"}},
	{{"Teselado", "Tessellation", "Tessela\xC3\xA7\xC3\xA3o", "Tessellation", "Tassellatura", "Tessellierung"}},
	{{"Mitad", "Half", "Metade", "Moiti\xC3\xA9", "Met\xC3\xA0", "Halb"}},
	{{"Completa", "Full", "Completa", "Compl\xC3\xA8te", "Completa", "Voll"}},
	{{"Cinem\xC3\xA1tico sin tope 30", "Cinematic, no 30 cap", "Cinematogr\xC3\xA1" "fico sem limite de 30", "Cin\xC3\xA9matique sans limite 30", "Cinematico senza limite 30", "Kinomodus ohne 30-Limit"}},
	{{"Modo rendimiento", "Performance mode", "Modo desempenho", "Mode performance", "Modalit\xC3\xA0 prestazioni", "Leistungsmodus"}},
	{{"Path tracing", "Path tracing", "Path tracing", "Path tracing", "Path tracing", "Path Tracing"}},
	{{"Humo y part\xC3\xAD" "culas", "Smoke and particles", "Fuma\xC3\xA7" "a e part\xC3\xAD" "culas", "Fum\xC3\xA9" "e et particules", "Fumo e particelle", "Rauch und Partikel"}},
	{{"Suave", "Soft", "Suave", "Doux", "Morbido", "Weich"}},
	{{"Antorchas", "Torches", "Tochas", "Torches", "Torce", "Fackeln"}},
	{{"Luz de antorchas RT", "RT torch light", "Luz de tochas RT", "Lumi\xC3\xA8re des torches RT", "Luce delle torce RT", "RT-Fackellicht"}},
	{{"Destellos de lente", "Lens flare", "Reflexo de lente", "Reflet d'objectif", "Riflesso dell'obiettivo", "Linsenreflexe"}},
	{{"Sombras RT", "RT shadows", "Sombras RT", "Ombres RT", "Ombre RT", "RT-Schatten"}},
	{{"RT por hardware (experimental)", "Hardware RT (experimental)", "RT por hardware (experimental)", "RT mat\xC3\xA9riel (exp\xC3\xA9rimental)", "RT hardware (sperimentale)", "Hardware-RT (experimentell)"}},
	{{"Reflejos (RT real)", "Reflections (hardware RT)", "Reflexos (RT real)", "R\xC3\xA9" "flexions (RT mat\xC3\xA9riel)", "Riflessi (RT hardware)", "Reflexionen (Hardware-RT)"}},
	{{"Sombras sol (RT real)", "Sun shadows (hardware RT)", "Sombras do sol (RT real)", "Ombres du soleil (RT mat\xC3\xA9riel)", "Ombre del sole (RT hardware)", "Sonnenschatten (Hardware-RT)"}},
	{{"Antorchas (RT real)", "Torches (hardware RT)", "Tochas (RT real)", "Torches (RT mat\xC3\xA9riel)", "Torce (RT hardware)", "Fackeln (Hardware-RT)"}},
	{{"H\xC3\xAD" "brido", "Hybrid", "H\xC3\xAD" "brido", "Hybride", "Ibrido", "Hybrid"}},
	{{"Media", "Medium", "M\xC3\xA9" "dia", "Moyenne", "Media", "Mittel"}},
	{{"Extremo", "Extreme", "Extremo", "Extr\xC3\xAAme", "Estremo", "Extrem"}},
};
// clang-format on

const std::map<std::string, const Row*>& Table() {
	static const std::map<std::string, const Row*> table = [] {
		std::map<std::string, const Row*> t;
		for (const auto& row: kRows) t[row.t[0]] = &row;
		return t;
	}();
	return table;
}

} // namespace

void SetLanguage(const std::string& id) {
	g_lang = 0;
	for (int i = 0; i < kMenuLanguageCount; i++)
		if (id == kMenuLanguages[i]) g_lang = i;
}

const char* T(const char* es) {
	if (g_lang == 0 || es == nullptr) return es;
	// (An ImGui id after "##" stays: "Cancelar##r".)
	const char* hash = std::strstr(es, "##");
	if (hash != nullptr) {
		static thread_local std::map<std::string, std::string> joined;
		const std::string key(es, hash);
		const auto        it = Table().find(key);
		if (it == Table().end()) return es;
		auto& out = joined[std::string(es) + "|" + std::to_string(g_lang)];
		if (out.empty()) out = std::string(it->second->t[g_lang]) + hash;
		return out.c_str();
	}
	const auto it = Table().find(es);
	return it == Table().end() ? es : it->second->t[g_lang];
}

std::string DetectWindowsLanguage() {
#if defined(_WIN32)
	switch (PRIMARYLANGID(GetUserDefaultUILanguage())) {
		case LANG_SPANISH: return "es";
		case LANG_PORTUGUESE: return "pt";
		case LANG_FRENCH: return "fr";
		case LANG_ITALIAN: return "it";
		case LANG_GERMAN: return "de";
		default: return "en";
	}
#else
	return "es";
#endif
}

int GameLanguageFor(const std::string& id) {
	// PS5 console languages (the launcher's list): 3 es-ES, 1 en-US, 17 pt-BR, 2 fr, 5 it, 4 de.
	if (id == "en") return 1;
	if (id == "pt") return 17;
	if (id == "fr") return 2;
	if (id == "it") return 5;
	if (id == "de") return 4;
	return 3;
}

} // namespace Port::I18n
