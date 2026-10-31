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
	// (ds-pc-port13) the shader precompile's CPU use
	{{"\xC2\xBFUsar todos los hilos de la CPU para precompilar m\xC3\xA1s r\xC3\xA1pido? (S\xC3\xAD = m\xC3\xA1xima velocidad, el PC ir\xC3\xA1 cargado; No = modo relajado)", "Use all CPU threads to precompile faster? (Yes = maximum speed, the PC will be under heavy load; No = relaxed mode)", "Usar todos os threads da CPU para pr\xC3\xA9-compilar mais r\xC3\xA1pido? (Sim = velocidade m\xC3\xA1xima, o PC ficar\xC3\xA1 carregado; N\xC3\xA3o = modo relaxado)", "Utiliser tous les threads du processeur pour pr\xC3\xA9" "compiler plus vite ? (Oui = vitesse maximale, le PC sera tr\xC3\xA8s charg\xC3\xA9 ; Non = mode d\xC3\xA9tendu)", "Usare tutti i thread della CPU per precompilare pi\xC3\xB9 velocemente? (S\xC3\xAC = massima velocit\xC3\xA0, il PC sar\xC3\xA0 sotto carico; No = modalit\xC3\xA0 rilassata)", "Alle CPU-Threads zum schnelleren Vorkompilieren nutzen? (Ja = maximale Geschwindigkeit, der PC ist stark ausgelastet; Nein = entspannter Modus)"}},
	{{"Precompilar shaders con todos los hilos de la CPU", "Precompile shaders with all CPU threads", "Pr\xC3\xA9-compilar shaders com todos os threads da CPU", "Pr\xC3\xA9" "compiler les shaders avec tous les threads du processeur", "Precompila gli shader con tutti i thread della CPU", "Shader mit allen CPU-Threads vorkompilieren"}},
	{{"S\xC3\xAD = m\xC3\xA1xima velocidad, el PC ir\xC3\xA1 cargado; No = modo relajado. Se aplica la pr\xC3\xB3xima vez que se preparen los shaders.", "Yes = maximum speed, the PC will be under heavy load; No = relaxed mode. Applies the next time the shaders are prepared.", "Sim = velocidade m\xC3\xA1xima, o PC ficar\xC3\xA1 carregado; N\xC3\xA3o = modo relaxado. Vale na pr\xC3\xB3xima prepara\xC3\xA7\xC3\xA3o dos shaders.", "Oui = vitesse maximale, le PC sera tr\xC3\xA8s charg\xC3\xA9 ; Non = mode d\xC3\xA9tendu. S'applique \xC3\xA0 la prochaine pr\xC3\xA9paration des shaders.", "S\xC3\xAC = massima velocit\xC3\xA0, il PC sar\xC3\xA0 sotto carico; No = modalit\xC3\xA0 rilassata. Vale alla prossima preparazione degli shader.", "Ja = maximale Geschwindigkeit, der PC ist stark ausgelastet; Nein = entspannter Modus. Gilt beim n\xC3\xA4" "chsten Vorbereiten der Shader."}},
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
	// (ds-pc-port11) The shaders' panel of the launcher (ds-precompile.cpp's states, its counting line in parts).
	{{"Ten paciencia: se est\xC3\xA1n precompilando los shaders del juego (solo la primera vez).",
	  "Please be patient: the game's shaders are being precompiled (first time only).",
	  "Tenha paci\xC3\xAAncia: os shaders do jogo est\xC3\xA3o sendo pr\xC3\xA9-compilados (s\xC3\xB3 na primeira vez).",
	  "Patience : les shaders du jeu sont en cours de pr\xC3\xA9" "compilation (la premi\xC3\xA8re fois seulement).",
	  "Pazienza: gli shader del gioco sono in precompilazione (solo la prima volta).",
	  "Geduld: Die Shader des Spiels werden vorkompiliert (nur beim ersten Mal)."}},
	{{"%d %%   -   %d / %d partes", "%d %%   -   %d / %d parts", "%d %%   -   %d / %d partes", "%d %%   -   %d / %d parties", "%d %%   -   %d / %d parti",
	  "%d %%   -   %d / %d Teile"}},
	{{"calculando el tiempo restante", "estimating the time left", "calculando o tempo restante", "calcul du temps restant", "calcolo del tempo rimanente",
	  "verbleibende Zeit wird berechnet"}},
	{{"quedan unos %lld h %lld min", "about %lld h %lld min left", "faltam cerca de %lld h %lld min", "environ %lld h %lld min restantes",
	  "circa %lld h %lld min rimanenti", "noch etwa %lld h %lld min"}},
	{{"quedan unos %lld min", "about %lld min left", "faltam cerca de %lld min", "environ %lld min restantes", "circa %lld min rimanenti", "noch etwa %lld min"}},
	{{"(retomada desde el %d %%)", "(resumed from %d %%)", "(retomada a partir de %d %%)", "(reprise \xC3\xA0 %d %%)", "(ripresa dal %d %%)", "(fortgesetzt ab %d %%)"}},
	{{"[esperando memoria libre]", "[waiting for free memory]", "[aguardando mem\xC3\xB3ria livre]", "[en attente de m\xC3\xA9moire libre]", "[in attesa di memoria libera]",
	  "[warte auf freien Speicher]"}},
	{{"Preparando...", "Preparing...", "Preparando...", "Pr\xC3\xA9paration...", "Preparazione...", "Vorbereitung..."}},
	{{"Preparando shaders...", "Preparing shaders...", "Preparando shaders...", "Pr\xC3\xA9paration des shaders...", "Preparazione degli shader...", "Shader werden vorbereitet..."}},
	{{"Preparando shaders: lo que ya estaba hecho...", "Preparing shaders: what was already done...", nullptr, nullptr, nullptr, nullptr}},
	{{"Preparando shaders: guardando lo hecho...", "Preparing shaders: saving what is done...", nullptr, nullptr, nullptr, nullptr}},
	{{"Preparando shaders: guardando la cache...", "Preparing shaders: saving the cache...", nullptr, nullptr, nullptr, nullptr}},
	{{"Preparacion de shaders detenida: lo hecho se guarda y sigue la proxima vez.",
	  "Shader preparation stopped: what is done is kept and it continues next time.", nullptr, nullptr, nullptr, nullptr}},
	{{"Shaders listos: todos los pipelines del juego estan en la cache de esta grafica.",
	  "Shaders ready: every pipeline of the game is in this graphics card's cache.", nullptr, nullptr, nullptr, nullptr}},
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
	{{"Sombras RT", "RT shadows", "Sombras RT", "Ombres RT", "Ombre RT", "RT-Schatten"}},
	{{"RT por hardware (experimental)", "Hardware RT (experimental)", "RT por hardware (experimental)", "RT mat\xC3\xA9riel (exp\xC3\xA9rimental)", "RT hardware (sperimentale)", "Hardware-RT (experimentell)"}},
	{{"Reflejos (RT real)", "Reflections (hardware RT)", "Reflexos (RT real)", "R\xC3\xA9" "flexions (RT mat\xC3\xA9riel)", "Riflessi (RT hardware)", "Reflexionen (Hardware-RT)"}},
	{{"Sombras sol (RT real)", "Sun shadows (hardware RT)", "Sombras do sol (RT real)", "Ombres du soleil (RT mat\xC3\xA9riel)", "Ombre del sole (RT hardware)", "Sonnenschatten (Hardware-RT)"}},
	{{"H\xC3\xAD" "brido", "Hybrid", "H\xC3\xAD" "brido", "Hybride", "Ibrido", "Hybrid"}},
	{{"Media", "Medium", "M\xC3\xA9" "dia", "Moyenne", "Media", "Mittel"}},
	{{"Extremo", "Extreme", "Extremo", "Extr\xC3\xAAme", "Estremo", "Extrem"}},
	// (ds-pc-port8) The launcher and F1 (save and quit).
	{{"Resoluci\xC3\xB3n", "Resolution", "Resolu\xC3\xA7\xC3\xA3o", "R\xC3\xA9solution", "Risoluzione", "Aufl\xC3\xB6sung"}},
	{{"Escritorio", "Desktop", "\xC3\x81rea de trabalho", "Bureau", "Desktop", "Desktop"}},
	{{"Nitidez de texturas", "Texture sharpness", "Nitidez das texturas", "Nettet\xC3\xA9 des textures", "Nitidezza texture", "Textursch\xC3\xA4rfe"}},
	{{"Desvanecimiento", "Object fade", "Desvanecimento", "Fondu des objets", "Dissolvenza", "Ausblendung"}},
	{{"Nitidez del AA", "AA sharpness", "Nitidez do AA", "Nettet\xC3\xA9 de l'AA", "Nitidezza AA", "AA-Sch\xC3\xA4rfe"}},
	{{"Vi\xC3\xB1" "eta", "Vignette", "Vinheta", "Vignettage", "Vignettatura", "Vignette"}},
	{{"Adaptaci\xC3\xB3n de la vista", "Eye adaptation", "Adapta\xC3\xA7\xC3\xA3o do olho", "Adaptation de l'\xC5\x93il", "Adattamento dell'occhio", "Augenanpassung"}},
	{{"Humo y part\xC3\xAD" "culas", "Smoke and particles", "Fuma\xC3\xA7" "a e part\xC3\xAD" "culas", "Fum\xC3\xA9" "e et particules", "Fumo e particelle", "Rauch und Partikel"}},
	{{"Modo de pantalla", "Display mode", "Modo de tela", "Mode d'affichage", "Modalit\xC3\xA0 schermo", "Anzeigemodus"}},
	{{"Ventana", "Window", "Janela", "Fen\xC3\xAAtre", "Finestra", "Fenster"}},
	{{"Tope de FPS", "FPS cap", "Limite de FPS", "Limite FPS", "Limite FPS", "FPS-Limit"}},
	{{"Auto (bajo el refresco)", "Auto (below the refresh rate)", "Auto (abaixo da taxa de atualiza\xC3\xA7\xC3\xA3o)", "Auto (sous le rafra\xC3\xAE" "chissement)", "Auto (sotto la frequenza)", "Auto (unter der Bildwiederholrate)"}},
	{{"Configurador", "Configurator", "Configurador", "Configurateur", "Configuratore", "Konfigurator"}},
	{{"Hemos detectado tu equipo. Puedes optimizar los gr\xC3\xA1" "ficos para \xC3\xA9l o seguir con tus ajustes actuales.", "We have detected your PC. You can optimize the graphics for it or keep your current settings.", "Detectamos o seu PC. Voc\xC3\xAA pode otimizar os gr\xC3\xA1" "ficos para ele ou manter suas configura\xC3\xA7\xC3\xB5" "es atuais.", "Nous avons d\xC3\xA9tect\xC3\xA9 votre PC. Vous pouvez optimiser les graphismes pour lui ou garder vos r\xC3\xA9glages actuels.", "Abbiamo rilevato il tuo PC. Puoi ottimizzare la grafica per esso o mantenere le impostazioni attuali.", "Wir haben deinen PC erkannt. Du kannst die Grafik daf\xC3\xBCr optimieren oder deine aktuellen Einstellungen behalten."}},
	{{"Optimizar seg\xC3\xBAn tu equipo", "Optimize for your PC", "Otimizar para o seu PC", "Optimiser pour votre PC", "Ottimizza per il tuo PC", "F\xC3\xBCr deinen PC optimieren"}},
	{{"Hemos elegido:", "We chose:", "Escolhemos:", "Nous avons choisi :", "Abbiamo scelto:", "Wir haben gew\xC3\xA4hlt:"}},
	{{"Mejoras para equipos de 16 GB de RAM", "16 GB RAM PC improvements", "Melhorias para PCs com 16 GB de RAM", "Am\xC3\xA9liorations pour PC \xC3\xA0 16 Go de RAM", "Miglioramenti per PC con 16 GB di RAM", "Verbesserungen f\xC3\xBCr PCs mit 16 GB RAM"}},
	{{"RAM %d GB: no hacen falta (solo en PCs de 20 GB o menos).", "RAM %d GB: not needed (only on PCs with 20 GB or less).", "RAM %d GB: n\xC3\xA3o s\xC3\xA3o necess\xC3\xA1rias (s\xC3\xB3 em PCs com 20 GB ou menos).", "RAM %d Go : inutiles (seulement sur les PC de 20 Go ou moins).", "RAM %d GB: non servono (solo su PC con 20 GB o meno).", "RAM %d GB: nicht n\xC3\xB6tig (nur bei PCs mit 20 GB oder weniger)."}},
	{{"RAM justa: perfil seguro. Cierra los navegadores mientras juegas.", "Tight RAM: safe profile. Close your browsers while playing.", "RAM limitada: perfil seguro. Feche os navegadores enquanto joga.", "RAM juste : profil s\xC3\xBBr. Fermez les navigateurs pendant le jeu.", "RAM scarsa: profilo sicuro. Chiudi i browser mentre giochi.", "Knapper RAM: sicheres Profil. Schlie\xC3\x9F" "e die Browser beim Spielen."}},
	{{"Simular 16 GB RAM", "Simulate 16 GB RAM", "Simular 16 GB de RAM", "Simuler 16 Go de RAM", "Simula 16 GB di RAM", "16 GB RAM simulieren"}},
	{{"Simular gr\xC3\xA1" "fica de 8 GB", "Simulate an 8 GB GPU", "Simular GPU de 8 GB", "Simuler un GPU de 8 Go", "Simula GPU da 8 GB", "8-GB-GPU simulieren"}},
	{{"Sensibilidad del rat\xC3\xB3n", "Mouse sensitivity", "Sensibilidade do mouse", "Sensibilit\xC3\xA9 de la souris", "Sensibilit\xC3\xA0 del mouse", "Mausempfindlichkeit"}},
	{{"Velocidad de c\xC3\xA1mara (rat\xC3\xB3n)", "Camera speed (mouse)", "Velocidade da c\xC3\xA2mera (mouse)", "Vitesse de la cam\xC3\xA9ra (souris)", "Velocit\xC3\xA0 telecamera (mouse)", "Kamerageschwindigkeit (Maus)"}},
	{{"C\xC3\xA1mara instant\xC3\xA1nea", "Instant camera", "C\xC3\xA2mera instant\xC3\xA2nea", "Cam\xC3\xA9ra instantan\xC3\xA9" "e", "Telecamera istantanea", "Sofortkamera"}},
	{{"Calibrar c\xC3\xA1mara", "Calibrate camera", "Calibrar c\xC3\xA2mera", "Calibrer la cam\xC3\xA9ra", "Calibra telecamera", "Kamera kalibrieren"}},
	{{"En el pr\xC3\xB3ximo inicio", "On the next start", "No pr\xC3\xB3ximo in\xC3\xAD" "cio", "Au prochain d\xC3\xA9marrage", "Al prossimo avvio", "Beim n\xC3\xA4" "chsten Start"}},
	{{"Fuerza %.1f", "Strength %.1f", "For\xC3\xA7" "a %.1f", "Force %.1f", "Intensit\xC3\xA0 %.1f", "St\xC3\xA4rke %.1f"}},
	{{"Teclas  v", "Keys  v", "Teclas  v", "Touches  v", "Tasti  v", "Tasten  v"}},
	{{"Teclas  ^", "Keys  ^", "Teclas  ^", "Touches  ^", "Tasti  ^", "Tasten  ^"}},
	{{"Avanzar", "Forward", "Avan\xC3\xA7" "ar", "Avancer", "Avanti", "Vorw\xC3\xA4rts"}},
	{{"Retroceder", "Back", "Recuar", "Reculer", "Indietro", "R\xC3\xBC" "ckw\xC3\xA4rts"}},
	{{"Izquierda", "Left", "Esquerda", "Gauche", "Sinistra", "Links"}},
	{{"Derecha", "Right", "Direita", "Droite", "Destra", "Rechts"}},
	{{"Caminar", "Walk", "Andar", "Marcher", "Camminare", "Gehen"}},
	{{"R1 (ataque)", "R1 (attack)", "R1 (ataque)", "R1 (attaque)", "R1 (attacco)", "R1 (Angriff)"}},
	{{"L1 (bloquear)", "L1 (block)", "L1 (bloquear)", "L1 (bloquer)", "L1 (parare)", "L1 (blocken)"}},
	{{"R2 (ataque fuerte)", "R2 (strong attack)", "R2 (ataque forte)", "R2 (attaque puissante)", "R2 (attacco forte)", "R2 (starker Angriff)"}},
	{{"L2 (parada)", "L2 (parry)", "L2 (aparar)", "L2 (parade)", "L2 (parata)", "L2 (Parieren)"}},
	{{"R3 (fijar objetivo)", "R3 (lock on)", "R3 (travar alvo)", "R3 (verrouiller la cible)", "R3 (aggancia bersaglio)", "R3 (Ziel erfassen)"}},
	{{"C\xC3\xADrculo (rodar / correr)", "Circle (roll / run)", "C\xC3\xADrculo (rolar / correr)", "Rond (roulade / course)", "Cerchio (rotolare / correre)", "Kreis (Rolle / Rennen)"}},
	{{"Cuadrado (objeto)", "Square (item)", "Quadrado (item)", "Carr\xC3\xA9 (objet)", "Quadrato (oggetto)", "Quadrat (Gegenstand)"}},
	{{"Cruz (acci\xC3\xB3n)", "Cross (action)", "X (a\xC3\xA7\xC3\xA3o)", "Croix (action)", "Croce (azione)", "Kreuz (Aktion)"}},
	{{"Tri\xC3\xA1ngulo (cambiar agarre)", "Triangle (switch grip)", "Tri\xC3\xA2ngulo (trocar empunhadura)", "Triangle (changer de prise)", "Triangolo (cambia impugnatura)", "Dreieck (Griff wechseln)"}},
	{{"Izquierda cruceta", "D-pad left", "Direcional esquerda", "Croix gauche", "Croce sinistra", "Steuerkreuz links"}},
	{{"Derecha cruceta", "D-pad right", "Direcional direita", "Croix droite", "Croce destra", "Steuerkreuz rechts"}},
	{{"Arriba cruceta", "D-pad up", "Direcional cima", "Croix haut", "Croce su", "Steuerkreuz oben"}},
	{{"Abajo cruceta", "D-pad down", "Direcional baixo", "Croix bas", "Croce gi\xC3\xB9", "Steuerkreuz unten"}},
	{{"Men\xC3\xBA", "Menu", "Menu", "Menu", "Menu", "Men\xC3\xBC"}},
	{{"Panel t\xC3\xA1" "ctil", "Touchpad", "Touchpad", "Pav\xC3\xA9 tactile", "Touchpad", "Touchpad"}},
	{{"En el juego: F1 guardar y salir (o R3 + OPTIONS).  F7: soltar o capturar el rat\xC3\xB3n.  F8: m\xC3\xA9tricas.  F11 / Alt+Intro: pantalla completa.", "In game: F1 save and quit (or R3 + OPTIONS).  F7: release or capture the mouse.  F8: metrics.  F11 / Alt+Enter: fullscreen.", "No jogo: F1 salvar e sair (ou R3 + OPTIONS).  F7: soltar ou capturar o mouse.  F8: m\xC3\xA9tricas.  F11 / Alt+Enter: tela cheia.", "En jeu : F1 sauvegarder et quitter (ou R3 + OPTIONS).  F7 : lib\xC3\xA9rer ou capturer la souris.  F8 : statistiques.  F11 / Alt+Entr\xC3\xA9" "e : plein \xC3\xA9" "cran.", "In gioco: F1 salva ed esci (o R3 + OPTIONS).  F7: rilascia o cattura il mouse.  F8: statistiche.  F11 / Alt+Invio: schermo intero.", "Im Spiel: F1 speichern und beenden (oder R3 + OPTIONS).  F7: Maus freigeben oder fangen.  F8: Messwerte.  F11 / Alt+Eingabe: Vollbild."}},
	{{"no detectada", "not detected", "n\xC3\xA3o detectada", "non d\xC3\xA9tect\xC3\xA9" "e", "non rilevata", "nicht erkannt"}},
	{{"  (sin DLSS: solo NVIDIA)", "  (no DLSS: NVIDIA only)", "  (sem DLSS: s\xC3\xB3 NVIDIA)", "  (pas de DLSS : NVIDIA uniquement)", "  (niente DLSS: solo NVIDIA)", "  (kein DLSS: nur NVIDIA)"}},
	{{"Gr\xC3\xA1" "fica sin datos: pulsa Optimizar seg\xC3\xBAn tu equipo y mira las m\xC3\xA9tricas.", "GPU not in the list: press Optimize for your PC and watch the metrics.", "GPU sem dados: clique em Otimizar para o seu PC e veja as m\xC3\xA9tricas.", "GPU inconnu : cliquez sur Optimiser pour votre PC et regardez les statistiques.", "GPU senza dati: premi Ottimizza per il tuo PC e guarda le statistiche.", "GPU ohne Daten: klicke auf F\xC3\xBCr deinen PC optimieren und beobachte die Messwerte."}},
	{{"Versi\xC3\xB3n instalada: %s", "Installed version: %s", "Vers\xC3\xA3o instalada: %s", "Version install\xC3\xA9" "e : %s", "Versione installata: %s", "Installierte Version: %s"}},
	{{"Carpeta del juego volcado (PPSA01341)", "Dumped game folder (PPSA01341)", "Pasta do jogo extra\xC3\xAD" "do (PPSA01341)", "Dossier du jeu extrait (PPSA01341)", "Cartella del gioco estratto (PPSA01341)", "Ordner des gedumpten Spiels (PPSA01341)"}},
	{{"Guardado en settings.ini", "Saved to settings.ini", "Salvo em settings.ini", "Enregistr\xC3\xA9 dans settings.ini", "Salvato in settings.ini", "In settings.ini gespeichert"}},
	{{"Esa carpeta no tiene eboot.bin ni sce_sys: elige la carpeta del juego volcado.", "That folder has no eboot.bin or sce_sys: choose the dumped game's folder.", "Essa pasta n\xC3\xA3o tem eboot.bin nem sce_sys: escolha a pasta do jogo extra\xC3\xAD" "do.", "Ce dossier n'a ni eboot.bin ni sce_sys : choisissez le dossier du jeu extrait.", "Quella cartella non ha eboot.bin n\xC3\xA9 sce_sys: scegli la cartella del gioco estratto.", "Dieser Ordner hat weder eboot.bin noch sce_sys: w\xC3\xA4hle den Ordner des gedumpten Spiels."}},
	{{"A\xC3\xB1" "adir a Steam", "Add to Steam", "Adicionar ao Steam", "Ajouter \xC3\xA0 Steam", "Aggiungi a Steam", "Zu Steam hinzuf\xC3\xBCgen"}},
	{{"A\xC3\xB1" "adiendo a Steam...", "Adding to Steam...", "Adicionando ao Steam...", "Ajout \xC3\xA0 Steam...", "Aggiunta a Steam...", "Wird zu Steam hinzugef\xC3\xBCgt..."}},
	{{"Steam debe cerrarse antes o reiniciarse despu\xC3\xA9s para que aparezca el acceso.", "Close Steam first, or restart it afterwards, for the shortcut to appear.", "Feche o Steam antes ou reinicie-o depois para o atalho aparecer.", "Fermez Steam avant, ou red\xC3\xA9marrez-le apr\xC3\xA8s, pour voir le raccourci.", "Chiudi Steam prima o riavvialo dopo perch\xC3\xA9 compaia il collegamento.", "Schlie\xC3\x9F" "e Steam vorher oder starte es danach neu, damit die Verkn\xC3\xBCpfung erscheint."}},
	{{"Actualizar a ", "Update to ", "Atualizar para ", "Mettre \xC3\xA0 jour vers ", "Aggiorna a ", "Aktualisieren auf "}},
	{{"Hay una version nueva: ", "A new version is available: ", "H\xC3\xA1 uma vers\xC3\xA3o nova: ", "Une nouvelle version est disponible : ", "C'\xC3\xA8 una nuova versione: ", "Neue Version verf\xC3\xBCgbar: "}},
	{{". Tienes: ", ". You have: ", ". Voc\xC3\xAA tem: ", ". Vous avez : ", ". Hai: ", ". Installiert: "}},
	{{"Actualizar ahora", "Update now", "Atualizar agora", "Mettre \xC3\xA0 jour", "Aggiorna ora", "Jetzt aktualisieren"}},
	{{"Mas tarde", "Later", "Mais tarde", "Plus tard", "Pi\xC3\xB9 tardi", "Sp\xC3\xA4ter"}},
	{{"Cerrar", "Close", "Fechar", "Fermer", "Chiudi", "Schlie\xC3\x9F" "en"}},
	{{"Trofeos", "Trophies", "Trof\xC3\xA9us", "Troph\xC3\xA9" "es", "Trofei", "Troph\xC3\xA4" "en"}},
	{{"A\xC3\xBAn no hay trofeos: aparecen al jugar.", "No trophies yet: they appear as you play.", "Ainda n\xC3\xA3o h\xC3\xA1 trof\xC3\xA9us: aparecem ao jogar.", "Pas encore de troph\xC3\xA9" "es : ils apparaissent en jouant.", "Ancora nessun trofeo: compaiono giocando.", "Noch keine Troph\xC3\xA4" "en: sie erscheinen beim Spielen."}},
	{{"%d%% completado   -   %d / %d", "%d%% complete   -   %d / %d", "%d%% conclu\xC3\xAD" "do   -   %d / %d", "%d%% termin\xC3\xA9   -   %d / %d", "%d%% completato   -   %d / %d", "%d%% abgeschlossen   -   %d / %d"}},
	{{"Platino", "Platinum", "Platina", "Platine", "Platino", "Platin"}},
	{{"Oro", "Gold", "Ouro", "Or", "Oro", "Gold"}},
	{{"Plata", "Silver", "Prata", "Argent", "Argento", "Silber"}},
	{{"Bronce", "Bronze", "Bronze", "Bronze", "Bronzo", "Bronze"}},
	{{"Trofeo oculto", "Hidden trophy", "Trof\xC3\xA9u oculto", "Troph\xC3\xA9" "e cach\xC3\xA9", "Trofeo nascosto", "Versteckte Troph\xC3\xA4" "e"}},
	{{"Continuar donde lo dej\xC3\xA9 (F1)", "Continue where I left off (F1)", "Continuar de onde parei (F1)", "Reprendre l\xC3\xA0 o\xC3\xB9 je me suis arr\xC3\xAAt\xC3\xA9 (F1)", "Continua da dove ho lasciato (F1)", "Dort weiterspielen, wo ich aufgeh\xC3\xB6rt habe (F1)"}},
	{{"Partida guardada con F1 el %s: JUGAR contin\xC3\xBA" "a all\xC3\xAD.", "Game saved with F1 on %s: PLAY continues there.", "Jogo salvo com F1 em %s: JOGAR continua l\xC3\xA1.", "Partie sauvegard\xC3\xA9" "e avec F1 le %s : JOUER reprend l\xC3\xA0.", "Partita salvata con F1 il %s: GIOCA riprende da l\xC3\xAC.", "Mit F1 gespeichert am %s: SPIELEN macht dort weiter."}},
	{{"Ajustes", "Settings", "Configura\xC3\xA7\xC3\xB5" "es", "R\xC3\xA9glages", "Impostazioni", "Einstellungen"}},
	{{"Controles  v", "Controls  v", "Controles  v", "Commandes  v", "Controlli  v", "Steuerung  v"}},
	{{"Controles  ^", "Controls  ^", "Controles  ^", "Commandes  ^", "Controlli  ^", "Steuerung  ^"}},
	{{"\xE2\x99\xA5  Dona tu alma  -  PayPal: todojuegos67@gmail.com", "\xE2\x99\xA5  Donate your soul  -  PayPal: todojuegos67@gmail.com", "\xE2\x99\xA5  Doe sua alma  -  PayPal: todojuegos67@gmail.com", "\xE2\x99\xA5  Donnez votre \xC3\xA2me  -  PayPal : todojuegos67@gmail.com", "\xE2\x99\xA5  Dona la tua anima  -  PayPal: todojuegos67@gmail.com", "\xE2\x99\xA5  Spende deine Seele  -  PayPal: todojuegos67@gmail.com"}},
	{{"Guardar y salir", "Save and quit", "Salvar e sair", "Sauvegarder et quitter", "Salva ed esci", "Speichern und beenden"}},
	{{"Guardando la partida...", "Saving the game...", "Salvando o jogo...", "Sauvegarde de la partie...", "Salvataggio della partita...", "Spiel wird gespeichert..."}},
	{{"Esperando a que el juego termine de guardar...", "Waiting for the game to finish saving...", "Aguardando o jogo terminar de salvar...", "Attente de la fin de la sauvegarde du jeu...", "In attesa che il gioco finisca di salvare...", "Warte, bis das Spiel fertig gespeichert hat..."}},
	{{"Copia de seguridad de las partidas...", "Backing up the saves...", "Fazendo backup dos saves...", "Copie de s\xC3\xA9" "curit\xC3\xA9 des parties...", "Backup dei salvataggi...", "Spielst\xC3\xA4nde werden gesichert..."}},
	{{"Partida guardada. Cerrando el juego...", "Game saved. Closing the game...", "Jogo salvo. Fechando o jogo...", "Partie sauvegard\xC3\xA9" "e. Fermeture du jeu...", "Partita salvata. Chiusura del gioco...", "Spiel gespeichert. Das Spiel wird geschlossen..."}},
	{{"El juego se cierra solo en unos segundos. Al pulsar JUGAR en el lanzador continuar\xC3\xA1s donde est\xC3\xA1s.", "The game closes by itself in a few seconds. Press PLAY in the launcher to continue where you are.", "O jogo fecha sozinho em alguns segundos. Ao clicar em JOGAR no iniciador voc\xC3\xAA continua onde est\xC3\xA1.", "Le jeu se ferme tout seul dans quelques secondes. En appuyant sur JOUER dans le lanceur, vous reprendrez l\xC3\xA0 o\xC3\xB9 vous \xC3\xAAtes.", "Il gioco si chiude da solo tra pochi secondi. Premendo GIOCA nel launcher continuerai da dove sei.", "Das Spiel schlie\xC3\x9Ft sich in wenigen Sekunden von selbst. Mit SPIELEN im Launcher machst du dort weiter, wo du bist."}},
	{{"Nitidez: auto", "Sharpness: auto", "Nitidez: auto", "Nettet\xC3\xA9 : auto", "Nitidezza: auto", "Sch\xC3\xA4rfe: auto"}},
	{{"Nitidez: 0", "Sharpness: 0", "Nitidez: 0", "Nettet\xC3\xA9 : 0", "Nitidezza: 0", "Sch\xC3\xA4rfe: 0"}},
	{{"Nitidez: 25%", "Sharpness: 25%", "Nitidez: 25%", "Nettet\xC3\xA9 : 25%", "Nitidezza: 25%", "Sch\xC3\xA4rfe: 25%"}},
	{{"Nitidez: 50%", "Sharpness: 50%", "Nitidez: 50%", "Nettet\xC3\xA9 : 50%", "Nitidezza: 50%", "Sch\xC3\xA4rfe: 50%"}},
	{{"Nitidez: 75%", "Sharpness: 75%", "Nitidez: 75%", "Nettet\xC3\xA9 : 75%", "Nitidezza: 75%", "Sch\xC3\xA4rfe: 75%"}},
	{{"Nitidez: 100%", "Sharpness: 100%", "Nitidez: 100%", "Nettet\xC3\xA9 : 100%", "Nitidezza: 100%", "Sch\xC3\xA4rfe: 100%"}},
	{{"Alto (rendimiento de PS5)", "High (PS5 performance)", "Alto (desempenho do PS5)", "Haut (performances PS5)", "Alto (prestazioni PS5)", "Hoch (PS5-Leistung)"}},
	{{"Anti-Lag: no", "Anti-Lag: off", "Anti-Lag: n\xC3\xA3o", "Anti-Lag : non", "Anti-Lag: no", "Anti-Lag: aus"}},
	{{"1440p nativo", "1440p native", "1440p nativo", "1440p natif", "1440p nativo", "1440p nativ"}},
	{{"DLSS U.calidad", "DLSS Ultra quality", "DLSS Ultra qualidade", "DLSS Ultra qualit\xC3\xA9", "DLSS Ultra qualit\xC3\xA0", "DLSS Ultra-Qualit\xC3\xA4t"}},
	{{"DLSS Calidad", "DLSS Quality", "DLSS Qualidade", "DLSS Qualit\xC3\xA9", "DLSS Qualit\xC3\xA0", "DLSS Qualit\xC3\xA4t"}},
	{{"DLSS Equilib.", "DLSS Balanced", "DLSS Equilibrado", "DLSS \xC3\x89quilibr\xC3\xA9", "DLSS Bilanciato", "DLSS Ausgewogen"}},
	{{"DLSS Rendim.", "DLSS Performance", "DLSS Desempenho", "DLSS Performance", "DLSS Prestazioni", "DLSS Leistung"}},
	{{"DLSS U.rendim.", "DLSS Ultra performance", "DLSS Ultra desempenho", "DLSS Ultra performance", "DLSS Ultra prestazioni", "DLSS Ultra-Leistung"}},
	{{"XeSS U.cal.+", "XeSS Ultra quality+", "XeSS Ultra qualidade+", "XeSS Ultra qualit\xC3\xA9+", "XeSS Ultra qualit\xC3\xA0+", "XeSS Ultra-Qualit\xC3\xA4t+"}},
	{{"XeSS U.calidad", "XeSS Ultra quality", "XeSS Ultra qualidade", "XeSS Ultra qualit\xC3\xA9", "XeSS Ultra qualit\xC3\xA0", "XeSS Ultra-Qualit\xC3\xA4t"}},
	{{"XeSS Calidad", "XeSS Quality", "XeSS Qualidade", "XeSS Qualit\xC3\xA9", "XeSS Qualit\xC3\xA0", "XeSS Qualit\xC3\xA4t"}},
	{{"XeSS Equilib.", "XeSS Balanced", "XeSS Equilibrado", "XeSS \xC3\x89quilibr\xC3\xA9", "XeSS Bilanciato", "XeSS Ausgewogen"}},
	{{"XeSS Rendim.", "XeSS Performance", "XeSS Desempenho", "XeSS Performance", "XeSS Prestazioni", "XeSS Leistung"}},
	{{"FSR AA nativo", "FSR native AA", "FSR AA nativo", "FSR AA natif", "FSR AA nativo", "FSR natives AA"}},
	{{"FSR U.calidad", "FSR Ultra quality", "FSR Ultra qualidade", "FSR Ultra qualit\xC3\xA9", "FSR Ultra qualit\xC3\xA0", "FSR Ultra-Qualit\xC3\xA4t"}},
	{{"FSR Calidad", "FSR Quality", "FSR Qualidade", "FSR Qualit\xC3\xA9", "FSR Qualit\xC3\xA0", "FSR Qualit\xC3\xA4t"}},
	{{"FSR Equilib.", "FSR Balanced", "FSR Equilibrado", "FSR \xC3\x89quilibr\xC3\xA9", "FSR Bilanciato", "FSR Ausgewogen"}},
	{{"FSR Rendim.", "FSR Performance", "FSR Desempenho", "FSR Performance", "FSR Prestazioni", "FSR Leistung"}},
	{{"FSR 4 Nativo (AA)", "FSR 4 Native (AA)", "FSR 4 Nativo (AA)", "FSR 4 Natif (AA)", "FSR 4 Nativo (AA)", "FSR 4 Nativ (AA)"}},
	{{"FSR 4 Calidad", "FSR 4 Quality", "FSR 4 Qualidade", "FSR 4 Qualit\xC3\xA9", "FSR 4 Qualit\xC3\xA0", "FSR 4 Qualit\xC3\xA4t"}},
	{{"FSR 4 Equilibrado", "FSR 4 Balanced", "FSR 4 Equilibrado", "FSR 4 \xC3\x89quilibr\xC3\xA9", "FSR 4 Bilanciato", "FSR 4 Ausgewogen"}},
	{{"FSR 4 Rendimiento", "FSR 4 Performance", "FSR 4 Desempenho", "FSR 4 Performance", "FSR 4 Prestazioni", "FSR 4 Leistung"}},
	{{"FSR 4 Ultra rendimiento", "FSR 4 Ultra performance", "FSR 4 Ultra desempenho", "FSR 4 Ultra performance", "FSR 4 Ultra prestazioni", "FSR 4 Ultra-Leistung"}},
	{{"FSR 4 Din\xC3\xA1mico", "FSR 4 Dynamic", "FSR 4 Din\xC3\xA2mico", "FSR 4 Dynamique", "FSR 4 Dinamico", "FSR 4 Dynamisch"}},
	{{"1080p juego", "1080p game", "1080p jogo", "1080p jeu", "1080p gioco", "1080p Spiel"}},
	{{"900p juego", "900p game", "900p jogo", "900p jeu", "900p gioco", "900p Spiel"}},
	{{"720p juego", "720p game", "720p jogo", "720p jeu", "720p gioco", "720p Spiel"}},
	{{"AMD MFG dinamico (x2-x4)", "AMD MFG dynamic (x2-x4)", "AMD MFG din\xC3\xA2mico (x2-x4)", "AMD MFG dynamique (x2-x4)", "AMD MFG dinamico (x2-x4)", "AMD MFG dynamisch (x2-x4)"}},
	{{"DLSS MFG dinamico (x2-x4)", "DLSS MFG dynamic (x2-x4)", "DLSS MFG din\xC3\xA2mico (x2-x4)", "DLSS MFG dynamique (x2-x4)", "DLSS MFG dinamico (x2-x4)", "DLSS MFG dynamisch (x2-x4)"}},
	{{"RAM justa", "Tight RAM", "RAM limitada", "RAM juste", "RAM scarsa", "Knapper RAM"}},
	{{"Todos los CCD (Ryzen)", "All CCDs (Ryzen)", "Todos os CCDs (Ryzen)", "Tous les CCD (Ryzen)", "Tutti i CCD (Ryzen)", "Alle CCDs (Ryzen)"}},
	{{"Opciones  v", "Options  v", "Op\xC3\xA7\xC3\xB5" "es  v", "Options  v", "Opzioni  v", "Optionen  v"}},
	{{"Opciones  ^", "Options  ^", "Op\xC3\xA7\xC3\xB5" "es  ^", "Options  ^", "Opzioni  ^", "Optionen  ^"}},
	{{"Modo Souls ligero", "Souls Light mode", "Modo Souls leve", "Mode Souls l\xC3\xA9ger", "Modalit\xC3\xA0 Souls leggera", "Souls-Modus Leicht"}},
	{{"Modo Souls equilibrado", "Souls Balanced mode", "Modo Souls equilibrado", "Mode Souls \xC3\xA9quilibr\xC3\xA9", "Modalit\xC3\xA0 Souls bilanciata", "Souls-Modus Ausgewogen"}},
	{{"Modo Souls ultra", "Souls Ultra mode", "Modo Souls ultra", "Mode Souls ultra", "Modalit\xC3\xA0 Souls ultra", "Souls-Modus Ultra"}},
	{{"Modo Souls extremo", "Souls Extreme mode", "Modo Souls extremo", "Mode Souls extr\xC3\xAAme", "Modalit\xC3\xA0 Souls estrema", "Souls-Modus Extrem"}},
	{{"Modo Souls PC low profile (900p, todo en bajo)", "Souls PC low profile (900p, all low)", "Modo Souls PC low profile (900p, tudo no baixo)", "Mode Souls PC low profile (900p, tout en bas)", "Modalit\xC3\xA0 Souls PC low profile (900p, tutto basso)", "Souls-PC-Low-Profile (900p, alles niedrig)"}},
	{{"Frame generation fija: AMD x2 (45 reales, 90 en pantalla).", "Fixed frame generation: AMD x2 (45 real, 90 on screen).", "Gera\xC3\xA7\xC3\xA3o de quadros fixa: AMD x2 (45 reais, 90 na tela).", "G\xC3\xA9n\xC3\xA9ration d'images fixe : AMD x2 (45 r\xC3\xA9" "elles, 90 \xC3\xA0 l'\xC3\xA9" "cran).", "Generazione di frame fissa: AMD x2 (45 reali, 90 a schermo).", "Feste Frame-Generierung: AMD x2 (45 echte, 90 auf dem Bildschirm)."}},
	{{"Frame generation fija: AMD x2 (30 reales, 60 en pantalla).", "Fixed frame generation: AMD x2 (30 real, 60 on screen).", "Gera\xC3\xA7\xC3\xA3o de quadros fixa: AMD x2 (30 reais, 60 na tela).", "G\xC3\xA9n\xC3\xA9ration d'images fixe : AMD x2 (30 r\xC3\xA9" "elles, 60 \xC3\xA0 l'\xC3\xA9" "cran).", "Generazione di frame fissa: AMD x2 (30 reali, 60 a schermo).", "Feste Frame-Generierung: AMD x2 (30 echte, 60 auf dem Bildschirm)."}},
	{{"Autom\xC3\xA1tico (idioma de Windows)", "Automatic (Windows language)", "Autom\xC3\xA1tico (idioma do Windows)", "Automatique (langue de Windows)", "Automatico (lingua di Windows)", "Automatisch (Windows-Sprache)"}},
	{{"Cerrar el lanzador (Esc)", "Close the launcher (Esc)", "Fechar o iniciador (Esc)", "Fermer le lanceur (\xC3\x89" "chap)", "Chiudi il launcher (Esc)", "Launcher schlie\xC3\x9F" "en (Esc)"}},
	// (ds-pc-port8) The launcher's tooltips: Spanish and English (the other languages show the English one).
	{{"Resoluci\xC3\xB3n de salida: la de Windows (Escritorio) u otra (el monitor cambia a ella mientras juegas). Los reescaladores calculan su resoluci\xC3\xB3n sobre ella. En ventana: el tama\xC3\xB1o de la ventana.", "Output resolution: Windows' own (Desktop) or another one (the monitor switches to it while you play). The upscalers compute their resolution from it. In a window: the window's size.", nullptr, nullptr, nullptr, nullptr}},
	{{"4K: el modo rendimiento del juego (cinem\xC3\xA1ticas sin el tope de 30) con la calidad cinem\xC3\xA1tica: iluminaci\xC3\xB3n completa y teselado denso. Otras resoluciones: modo rendimiento.", "4K: the game's performance mode (cutscenes without the 30 cap) with cinematic quality: full lighting and dense tessellation. Other resolutions: performance mode.", nullptr, nullptr, nullptr, nullptr}},
	{{"Reescalado: dibuja el 3D a menos resolucion y lo sube a la de la pantalla (mas fps). DLSS: NVIDIA RTX. XeSS, FSR 3.1 y FSR 4: todas las graficas (FSR 4: modelo de AMD con INT8, RTX 20 / RX 6000 o mas nuevas; el 3D a 360 lineas como minimo).", "Upscaling: renders the 3D at a lower resolution and scales it up to the screen's (more FPS). DLSS: NVIDIA RTX. XeSS, FSR 3.1 and FSR 4: all GPUs (FSR 4: AMD's INT8 model, RTX 20 / RX 6000 or newer; the 3D at 360 lines at least).", nullptr, nullptr, nullptr, nullptr}},
	{{"Modelo de DLSS: auto (el de NVIDIA), K = DLSS 4, L y M = DLSS 4.5 (mas nitidos, algo mas lentos)", "DLSS model: auto (NVIDIA's default), K = DLSS 4, L and M = DLSS 4.5 (sharper, a bit slower)", nullptr, nullptr, nullptr, nullptr}},
	// (ds-pc-port10) DLSS 5
	{{"DLSS 5", "DLSS 5", "DLSS 5", "DLSS 5", "DLSS 5", "DLSS 5"}},
	{{"Archivo DLSS 5", "DLSS 5 file", "Arquivo do DLSS 5", "Fichier DLSS 5", "File DLSS 5", "DLSS-5-Datei"}},
	{{"Fuerte", "Strong", "Forte", "Fort", "Forte", "Stark"}},
	{{"Baja", "Low", "Baixa", "Faible", "Bassa", "Niedrig"}},
	{{"Intensidad DLSS 5", "DLSS 5 intensity", "Intensidade do DLSS 5", "Intensit\xC3\xA9 DLSS 5", "Intensit\xC3\xA0 DLSS 5", "DLSS-5-Intensit\xC3\xA4t"}},
	{{"Intensidad del efecto de DLSS 5: Baja, Media o Alta (la de NVIDIA, por defecto). M\xC3\xA1s baja: se parece m\xC3\xA1s a la imagen original del juego.",
	  "Strength of the DLSS 5 effect: Low, Medium or High (NVIDIA's, the default). Lower: closer to the game's original picture.",
	  "Intensidade do efeito do DLSS 5: Baixa, M\xC3\xA9" "dia ou Alta (a da NVIDIA, padr\xC3\xA3o). Mais baixa: mais parecida com a imagem original do jogo.",
	  "Intensit\xC3\xA9 de l'effet DLSS 5 : Faible, Moyenne ou \xC3\x89lev\xC3\xA9" "e (celle de NVIDIA, par d\xC3\xA9" "faut). Plus faible : plus proche de l'image d'origine du jeu.",
	  "Intensit\xC3\xA0 dell'effetto DLSS 5: Bassa, Media o Alta (quella di NVIDIA, predefinita). Pi\xC3\xB9 bassa: pi\xC3\xB9 simile all'immagine originale del gioco.",
	  "St\xC3\xA4rke des DLSS-5-Effekts: Niedrig, Mittel oder Hoch (die von NVIDIA, Standard). Niedriger: n\xC3\xA4her am Originalbild des Spiels."}},
	{{"Oficial (firmado por NVIDIA)", "Official (signed by NVIDIA)", "Oficial (assinado pela NVIDIA)", "Officiel (sign\xC3\xA9 par NVIDIA)", "Ufficiale (firmato da NVIDIA)", "Offiziell (von NVIDIA signiert)"}},
	{{"No oficial (bajo tu responsabilidad)", "Unofficial (at your own risk)", "N\xC3\xA3o oficial (por sua conta e risco)", "Non officiel (\xC3\xA0 vos risques)", "Non ufficiale (a tuo rischio)", "Inoffiziell (auf eigene Gefahr)"}},
	{{"Sin DLSS 5: tu grafica no es NVIDIA.", "No DLSS 5: your GPU is not NVIDIA.", "Sem DLSS 5: sua placa n\xC3\xA3o \xC3\xA9 NVIDIA.", "Pas de DLSS 5 : votre carte n'est pas NVIDIA.", "Niente DLSS 5: la tua scheda non \xC3\xA8 NVIDIA.", "Kein DLSS 5: Ihre Grafikkarte ist keine NVIDIA."}},
	{{"DLSS 5 (Neural Rendering de NVIDIA): mejora la imagen con la IA de NVIDIA despu\xC3\xA9s del reescalado. Solo NVIDIA RTX. Necesita tu propio archivo nvngx_dlssnr.dll junto a \"Demon's Souls.exe\" (no viene incluido): sin \xC3\xA9l se ve como siempre. Suave, Medio y Fuerte bajan la resoluci\xC3\xB3n en los bordes de la pantalla para ahorrar GPU (Suave un poco, Fuerte m\xC3\xA1s); el centro siempre va a resoluci\xC3\xB3n completa. Experimental.",
	  "DLSS 5 (NVIDIA Neural Rendering): improves the picture with NVIDIA's AI after the upscaler. NVIDIA RTX only. Needs your own nvngx_dlssnr.dll file next to \"Demon's Souls.exe\" (not included): without it the picture is as usual. Soft, Medium and Strong lower the resolution at the screen edges to save GPU (Soft a little, Strong more); the centre is always at full resolution. Experimental.",
	  "DLSS 5 (Neural Rendering da NVIDIA): melhora a imagem com a IA da NVIDIA depois do reescalonamento. S\xC3\xB3 NVIDIA RTX. Precisa do seu pr\xC3\xB3prio arquivo nvngx_dlssnr.dll ao lado de \"Demon's Souls.exe\" (n\xC3\xA3o inclu\xC3\xAD" "do): sem ele a imagem fica como sempre. Suave, M\xC3\xA9" "dio e Forte reduzem a resolu\xC3\xA7\xC3\xA3o nas bordas da tela para poupar GPU; o centro sempre fica em resolu\xC3\xA7\xC3\xA3o total. Experimental.",
	  "DLSS 5 (Neural Rendering de NVIDIA) : am\xC3\xA9liore l'image avec l'IA de NVIDIA apr\xC3\xA8s la mise \xC3\xA0 l'\xC3\xA9" "chelle. NVIDIA RTX uniquement. N\xC3\xA9" "cessite votre propre fichier nvngx_dlssnr.dll \xC3\xA0 c\xC3\xB4t\xC3\xA9 de \"Demon's Souls.exe\" (non inclus) : sans lui, l'image reste habituelle. Doux, Moyen et Fort baissent la r\xC3\xA9solution sur les bords de l'\xC3\xA9" "cran pour \xC3\xA9" "conomiser le GPU ; le centre est toujours en pleine r\xC3\xA9solution. Exp\xC3\xA9rimental.",
	  "DLSS 5 (Neural Rendering di NVIDIA): migliora l'immagine con l'IA di NVIDIA dopo l'upscaling. Solo NVIDIA RTX. Serve il tuo file nvngx_dlssnr.dll accanto a \"Demon's Souls.exe\" (non incluso): senza, l'immagine resta quella di sempre. Morbido, Medio e Forte abbassano la risoluzione ai bordi dello schermo per risparmiare GPU; il centro \xC3\xA8 sempre a risoluzione piena. Sperimentale.",
	  "DLSS 5 (NVIDIA Neural Rendering): verbessert das Bild mit NVIDIAs KI nach dem Hochskalieren. Nur NVIDIA RTX. Ben\xC3\xB6tigt Ihre eigene Datei nvngx_dlssnr.dll neben \"Demon's Souls.exe\" (nicht enthalten): ohne sie bleibt das Bild wie gewohnt. Weich, Mittel und Stark senken die Aufl\xC3\xB6sung an den Bildr\xC3\xA4ndern, um GPU zu sparen; die Mitte bleibt immer in voller Aufl\xC3\xB6sung. Experimentell."}},
	{{"Qu\xC3\xA9 nvngx_dlssnr.dll puede cargar DLSS 5. Oficial: solo si est\xC3\xA1 firmado por NVIDIA (recomendado). No oficial: lo carga aunque no est\xC3\xA9 firmado (un archivo modificado); bajo tu responsabilidad.",
	  "Which nvngx_dlssnr.dll DLSS 5 may load. Official: only if it is signed by NVIDIA (recommended). Unofficial: it loads it even when it is not signed (a modified file); at your own risk.",
	  "Qual nvngx_dlssnr.dll o DLSS 5 pode carregar. Oficial: s\xC3\xB3 se estiver assinado pela NVIDIA (recomendado). N\xC3\xA3o oficial: carrega mesmo sem assinatura (um arquivo modificado); por sua conta e risco.",
	  "Quel nvngx_dlssnr.dll DLSS 5 peut charger. Officiel : seulement s'il est sign\xC3\xA9 par NVIDIA (recommand\xC3\xA9). Non officiel : il le charge m\xC3\xAAme non sign\xC3\xA9 (un fichier modifi\xC3\xA9) ; \xC3\xA0 vos risques.",
	  "Quale nvngx_dlssnr.dll pu\xC3\xB2 caricare DLSS 5. Ufficiale: solo se firmato da NVIDIA (consigliato). Non ufficiale: lo carica anche se non firmato (un file modificato); a tuo rischio.",
	  "Welche nvngx_dlssnr.dll DLSS 5 laden darf. Offiziell: nur, wenn sie von NVIDIA signiert ist (empfohlen). Inoffiziell: l\xC3\xA4" "dt sie auch ohne Signatur (eine ver\xC3\xA4nderte Datei); auf eigene Gefahr."}},
	{{"Sin DLSS: tu grafica no es NVIDIA.", "No DLSS: your GPU is not NVIDIA.", nullptr, nullptr, nullptr, nullptr}},
	{{"Nitidez del reescalado (DLSS, XeSS, FSR 3.1 y FSR 4), aplicada a la resolucion de la pantalla. Mas alto: mas nitido, pero puede marcar bordes y ruido.", "Upscaling sharpness (DLSS, XeSS, FSR 3.1 and FSR 4), applied at the screen's resolution. Higher: sharper, but it can emphasize edges and noise.", nullptr, nullptr, nullptr, nullptr}},
	{{"Tama\xC3\xB1o de las texturas: Medio = la mitad, Bajo = un cuarto. Lo que m\xC3\xA1s VRAM ahorra.", "Texture size: Medium = half, Low = a quarter. Saves the most VRAM.", nullptr, nullptr, nullptr, nullptr}},
	{{"Distancia de dibujado: a menos nivel, los objetos lejanos pasan antes a modelos m\xC3\xA1s simples. M\xC3\xA1s FPS si va justo el procesador.", "Draw distance: at lower levels, distant objects switch to simpler models sooner. More FPS if the CPU is the bottleneck.", nullptr, nullptr, nullptr, nullptr}},
	{{"Sombras: Medio = las del sol hasta 125 m, Bajo = hasta 60 m y sin sombras de focos ni ambientales. M\xC3\xA1s FPS, pero pueden aparecer de golpe.", "Shadows: Medium = sun shadows up to 125 m, Low = up to 60 m and no spotlight or ambient shadows. More FPS, but they may pop in.", nullptr, nullptr, nullptr, nullptr}},
	{{"Detalle del juego: sombras (cascadas y distancia), iluminaci\xC3\xB3n de entorno y relieve (teselado). Ultra > Muy alto (modo cine) > Alto (modo rendimiento de PS5) > Normal > Bajo.", "Game detail: shadows (cascades and distance), environment lighting and relief (tessellation). Ultra > Very high (cinematic mode) > High (PS5 performance mode) > Normal > Low.", nullptr, nullptr, nullptr, nullptr}},
	{{"Filtrado anisotr\xC3\xB3pico x16: suelos y paredes n\xC3\xADtidos a lo lejos y en \xC3\xA1ngulo. Cuesta muy poco.", "16x anisotropic filtering: floors and walls stay sharp at a distance and at an angle. Very cheap.", nullptr, nullptr, nullptr, nullptr}},
	{{"Texturas: n\xC3\xADtidas = el detalle de la resoluci\xC3\xB3n final con reescalado. Extra n\xC3\xADtidas: un escal\xC3\xB3n m\xC3\xA1s (lo que recomienda NVIDIA para DLSS); puede brillar algo en rejas y hierba. Cuesta pocos FPS.", "Textures: sharp = the detail of the final resolution when upscaling. Extra sharp: one step more (what NVIDIA recommends for DLSS); may shimmer a little on grates and grass. Costs a few FPS.", nullptr, nullptr, nullptr, nullptr}},
	{{"Reflejos del propio juego (agua, charcos, suelos): m\xC3\xA1s pasos y m\xC3\xA1s alcance en Ultra. Cuesta algo de GPU.", "The game's own reflections (water, puddles, floors): more steps and longer range on Ultra. Costs some GPU.", nullptr, nullptr, nullptr, nullptr}},
	{{"Distancia a la que los objetos lejanos se desvanecen. Ultra: aparecen de golpe mucho menos.", "Distance at which far-away objects fade out. Ultra: much less pop-in.", nullptr, nullptr, nullptr, nullptr}},
	{{"Nitidez del antialiasing del juego. No cuesta FPS.", "Sharpness of the game's antialiasing. Costs no FPS.", nullptr, nullptr, nullptr, nullptr}},
	{{"Desenfoque de movimiento del juego (S\xC3\xAD = como en PS5).", "The game's motion blur (Yes = as on PS5).", nullptr, nullptr, nullptr, nullptr}},
	{{"Desenfoque de lo lejano y de lo cercano en escenas (S\xC3\xAD = como en PS5).", "Blur of far and near objects in scenes (Yes = as on PS5).", nullptr, nullptr, nullptr, nullptr}},
	{{"Oscurecimiento de los bordes de la imagen (S\xC3\xAD = como en PS5).", "Darkening of the picture's edges (Yes = as on PS5).", nullptr, nullptr, nullptr, nullptr}},
	{{"Resplandor alrededor de las luces fuertes (S\xC3\xAD = como en PS5).", "Glow around strong lights (Yes = as on PS5).", nullptr, nullptr, nullptr, nullptr}},
	{{"La exposici\xC3\xB3n se adapta al pasar de zonas oscuras a claras, como el ojo (S\xC3\xAD = como en PS5).", "The exposure adapts when going from dark to bright areas, like the eye (Yes = as on PS5).", nullptr, nullptr, nullptr, nullptr}},
	{{"Resoluci\xC3\xB3n de la iluminaci\xC3\xB3n de entorno (IBL) del juego: Completa = la del modo cine de PS5, m\xC3\xA1s fina; Mitad = la del modo rendimiento. Con 4K empieza en Completa, con otras salidas en Mitad.", "Resolution of the game's ambient lighting (IBL): Full = the PS5 cinematic mode's, finer; Half = the performance mode's. 4K starts at Full, other outputs at Half.", nullptr, nullptr, nullptr, nullptr}},
	{{"Relieve por teselado del juego (suelos, muros, piedras): Bajo, Medio, Alto (el del modo cine de PS5), Ultra y Extremo (m\xC3\xA1s relieve y m\xC3\xA1s lejos; cuesta GPU). Con 4K empieza en Alto, con otras salidas en Medio.", "The game's tessellation relief (floors, walls, stones): Low, Medium, High (the PS5 cinematic mode's), Ultra and Extreme (more relief, farther; costs GPU). 4K starts at High, other outputs at Medium.", nullptr, nullptr, nullptr, nullptr}},
	{{"Reflejos trazados por rayos guiados por los del propio juego: solo donde el juego refleja y con su intensidad, m\xC3\xA1s n\xC3\xADtidos (agua como espejo). Donde el rayo no encuentra nada queda el reflejo del juego. Solo con DLSS, XeSS o FSR. Bajo ~0,6 ms, Medio ~1 ms, Alto ~2 ms, Ultra ~3 ms (rayos a resoluci\xC3\xB3n completa y m\xC3\xA1s fuertes). H\xC3\xAD" "brido: Alto + rayos por hardware (RT real) solo donde el rayo de pantalla no encuentra nada. Necesita GPU con RT. Beta.", "Ray-traced reflections guided by the game's own: only where the game reflects and at its strength, sharper (mirror-like water). Where the ray finds nothing the game's reflection stays. DLSS, XeSS or FSR only. Low ~0.6 ms, Medium ~1 ms, High ~2 ms, Ultra ~3 ms (full-resolution, stronger rays). Hybrid: High + hardware rays (real RT) only where the screen ray finds nothing. Needs an RT GPU. Beta.", nullptr, nullptr, nullptr, nullptr}},
	{{"Iluminaci\xC3\xB3n por rayos sobre la imagen (luz rebotada y sombreado de contacto). Solo funciona con DLSS, XeSS o FSR. Cuesta FPS. Experimental.", "Screen-space ray-traced lighting (bounced light and contact shading). Only works with DLSS, XeSS or FSR. Costs FPS. Experimental.", nullptr, nullptr, nullptr, nullptr}},
	{{"Luz volum\xC3\xA9trica por rayos: rayos de sol que entran por ventanas y puertas, y halo de luz en la niebla alrededor de antorchas. Solo con DLSS, XeSS o FSR. Bajo ~0,4 ms, Medio ~0,8 ms, Alto ~1,2 ms.", "Ray-traced volumetric light: sun shafts through windows and doors, and a light halo in the fog around torches. DLSS, XeSS or FSR only. Low ~0.4 ms, Medium ~0.8 ms, High ~1.2 ms.", nullptr, nullptr, nullptr, nullptr}},
	{{"Path tracing (RT por hardware: la escena TLAS de la GPU): rayos por p\xC3\xADxel desde cada superficie que dan la oclusi\xC3\xB3n real (tambi\xC3\xA9n de lo que est\xC3\xA1 fuera de pantalla o detr\xC3\xA1s de la c\xC3\xA1mara), un rebote de luz indirecta con el color de lo que tocan y sombras suaves de contacto hacia el sol y las antorchas. Sustituye solo la luz ambiental y a la oclusi\xC3\xB3n del juego. Sin RT por hardware: oclusi\xC3\xB3n de pantalla suave. Solo con DLSS, XeSS o FSR. Bajo 1 rayo a media resoluci\xC3\xB3n, Medio 1 rayo, Alto 2 rayos a resoluci\xC3\xB3n completa (~1-3 ms).", "Path tracing (hardware RT: the GPU's TLAS scene): rays per pixel from every surface give real occlusion (also from off-screen and behind-camera geometry), one bounce of indirect light coloured by what they hit, and soft contact shadows towards the sun and torches. It replaces only the game's ambient light and occlusion. Without hardware RT: soft screen-space occlusion. DLSS, XeSS or FSR only. Low 1 ray at half resolution, Medium 1 ray, High 2 rays at full resolution (~1-3 ms).", nullptr, nullptr, nullptr, nullptr}},
	{{"Humo y part\xC3\xAD" "culas m\xC3\xA1s realistas sobre las del propio juego: Suave = el humo, la niebla de part\xC3\xAD" "culas y las chispas ya no cortan en l\xC3\xADnea dura donde tocan suelo y paredes (se funden en ~35 cm). Solo con DLSS, XeSS o FSR. ~0,3 ms.", "More realistic smoke and particles on top of the game's own: Soft = smoke, particle fog and sparks no longer cut in a hard line where they meet floors and walls (they fade in over ~35 cm). DLSS, XeSS or FSR only. ~0.3 ms.", nullptr, nullptr, nullptr, nullptr}},
	{{"Sombras del sol por rayos sobre la imagen (de pantalla, sin RT real): sombras de contacto que las del juego pierden (bajo brazos, entre piernas, junto a muros), con borde suave y estable. Solo donde el juego ya muestra sol. Solo con DLSS, XeSS o FSR. Bajo ~0,3 ms, Medio ~0,6 ms, Alto ~1 ms.", "Screen-space ray-traced sun shadows (no hardware RT): the contact shadows the game misses (under arms, between legs, by walls), soft and stable. Only where the game already shows sunlight. DLSS, XeSS or FSR only. Low ~0.3 ms, Medium ~0.6 ms, High ~1 ms.", nullptr, nullptr, nullptr, nullptr}},
	{{"Activa las opciones de trazado de rayos por hardware (Reflejos y Sombras sol por RT real). Experimental: puede tener fallos y cuesta mucha GPU y VRAM. Necesita GPU con RT.", "Turns on the hardware ray tracing options (Reflections and Sun shadows with real RT). Experimental: may have glitches and costs a lot of GPU and VRAM. Needs an RT GPU.", nullptr, nullptr, nullptr, nullptr}},
	{{"Reflejos con trazado de rayos: refleja tambi\xC3\xA9n lo que est\xC3\xA1 fuera de pantalla o tapado, con sus texturas.", "Ray-traced reflections: also reflects off-screen or hidden objects, with their textures.", nullptr, nullptr, nullptr, nullptr}},
	{{"Sombras del sol con trazado de rayos: sombras de contacto y las que faltaban. No se puede usar a la vez que la Luz volum\xC3\xA9trica ni las Sombras RT.", "Ray-traced sun shadows: contact shadows and the missing ones. Cannot be used together with Volumetric light or RT shadows.", nullptr, nullptr, nullptr, nullptr}},
	{{"Apaga todos los efectos visuales que a\xC3\xB1" "ade el emulador (ray tracing, luz volum\xC3\xA9trica, humo, SSGI, reflejos RT, DLSS 5). Se ve el juego tal cual. Rendimiento, reescalador, frame generation y latencia no cambian.", "Turns off every visual effect the emulator adds (ray tracing, volumetric light, smoke, SSGI, RT reflections, DLSS 5). You see the game as it is. Performance, upscaler, frame generation and latency are unchanged.", nullptr, nullptr, nullptr, nullptr}},
	{{"AMD: todas las graficas (tambien con reescalado DLSS). DLSS: NVIDIA RTX 40/50. FPS desbloqueados siempre.", "AMD: all GPUs (also with DLSS upscaling). DLSS: NVIDIA RTX 40/50. FPS always unlocked.", nullptr, nullptr, nullptr, nullptr}},
	{{"Con la frame generation de NVIDIA (DLSS-G). Reflex: el de NVIDIA (recomendado). Anti-Lag / Ultra: el nuestro con Reflex apagado, para comparar cual va mejor.", "With NVIDIA frame generation (DLSS-G). Reflex: NVIDIA's (recommended). Anti-Lag / Ultra: ours with Reflex off, to compare which feels better.", nullptr, nullptr, nullptr, nullptr}},
	{{"Con la frame generation de AMD. Anti-Lag 2 (AMD): el de serie de las graficas AMD. Anti-Lag: el nuestro, como mucho 2 frames en cola antes de leer el mando. Ultra: 1 (aun menos retraso).", "With AMD frame generation. Anti-Lag 2 (AMD): AMD GPUs' built-in one. Anti-Lag: ours, at most 2 frames queued before reading the controller. Ultra: 1 (even less lag).", nullptr, nullptr, nullptr, nullptr}},
	{{"HDR10: activa antes \"Usar HDR\" en la configuracion de pantalla de Windows", "HDR10: first turn on \"Use HDR\" in the Windows display settings", nullptr, nullptr, nullptr, nullptr}},
	{{"VSync: sin cortes de imagen (tearing); con FreeSync/G-Sync sigue funcionando", "VSync: no screen tearing; FreeSync/G-Sync keeps working", nullptr, nullptr, nullptr, nullptr}},
	{{"G-Sync / FreeSync: el emulador no ata su reloj a los Hz del monitor", "G-Sync / FreeSync: the emulator does not tie its clock to the monitor's refresh rate", nullptr, nullptr, nullptr, nullptr}},
	{{"Monitores panoramicos (ultrawide): el juego se dibuja a 21:9 o 32:9 en vez de estirarse. Funciona con DLSS, XeSS y FSR. Los videos y algunos menus se ven estirados.", "Ultrawide monitors: the game is rendered at 21:9 or 32:9 instead of stretched. Works with DLSS, XeSS and FSR. Movies and some menus look stretched.", nullptr, nullptr, nullptr, nullptr}},
	{{"Pantalla completa o ventana (F11 lo cambia en vivo).", "Fullscreen or window (F11 switches it live).", nullptr, nullptr, nullptr, nullptr}},
	{{"Con un monitor panor\xC3\xA1mico: el juego en 16:9 con bandas negras a los lados en vez de estirarse.", "With an ultrawide monitor: the game at 16:9 with black bars at the sides instead of stretched.", nullptr, nullptr, nullptr, nullptr}},
	{{"Panel dentro del juego (F8 lo cambia en vivo): FPS, el modo de frame generation, latencia y el detalle.", "In-game panel (F8 switches it live): FPS, the frame generation mode, latency and the details.", nullptr, nullptr, nullptr, nullptr}},
	{{"Tope de FPS con G-Sync sin frame generation.", "FPS cap with G-Sync and no frame generation.", nullptr, nullptr, nullptr, nullptr}},
	{{"Idioma de este lanzador y de los mensajes del port.", "Language of this launcher and of the port's messages.", nullptr, nullptr, nullptr, nullptr}},
	{{"Idioma del juego (textos y voces).", "The game's language (texts and voices).", nullptr, nullptr, nullptr, nullptr}},
	{{"Copia _SaveData en _SaveData-copias cada vez que pulsas JUGAR.", "Copies _SaveData to _SaveData-copias every time you press PLAY.", nullptr, nullptr, nullptr, nullptr}},
	{{"Elige los ajustes con mejor rendimiento para tu gr\xC3\xA1" "fica, procesador, RAM y monitor (objetivo: 60 FPS o m\xC3\xA1s) y los guarda en settings.ini. Cambia gr\xC3\xA1" "ficos, ray tracing y la opci\xC3\xB3n de 16 GB de RAM.", "Picks the best-performing settings for your GPU, CPU, RAM and monitor (target: 60 FPS or more) and saves them to settings.ini. It changes graphics, ray tracing and the 16 GB RAM option.", nullptr, nullptr, nullptr, nullptr}},
	{{"Perfil de memoria de 16 GB (6 GB fijos en RAM, la precarga espera si queda poca memoria). Solo en PCs de 20 GB de RAM o menos. No cambia los gr\xC3\xA1" "ficos.", "16 GB memory profile (6 GB fixed in RAM, the preload waits when memory is low). Only on PCs with 20 GB of RAM or less. Does not change the graphics.", nullptr, nullptr, nullptr, nullptr}},
	{{"Solo el juego: como mucho 12 GB en RAM (4 GB para Windows), el resto va al disco, como un PC de 16 GB.", "Game only: at most 12 GB in RAM (4 GB for Windows), the rest goes to disk, like a 16 GB PC.", nullptr, nullptr, nullptr, nullptr}},
	{{"El juego solo puede usar 8 GB de VRAM, como una gr\xC3\xA1" "fica de 8 GB.", "The game can only use 8 GB of VRAM, like an 8 GB GPU.", nullptr, nullptr, nullptr, nullptr}},
	{{"Para gr\xC3\xA1" "ficas que se cuelgan o dan pantallazo azul: m\xC3\xA1s estable, ~37% menos FPS.", "For GPUs that hang or blue-screen: more stable, ~37% fewer FPS.", nullptr, nullptr, nullptr, nullptr}},
	{{"Ryzen de dos CCD: No = el juego en el CCD con m\xC3\xA1s cach\xC3\xA9 L3 (m\xC3\xA1s r\xC3\xA1pido); S\xC3\xAD = todos los n\xC3\xBA" "cleos.", "Dual-CCD Ryzen: No = the game on the CCD with more L3 cache (faster); Yes = all cores.", nullptr, nullptr, nullptr, nullptr}},
	{{"Jugar con teclado y rat\xC3\xB3n: WASD para moverse, el rat\xC3\xB3n mueve la c\xC3\xA1mara, clic izquierdo R1, clic derecho L1, rueda o clic central fija objetivo. F7 suelta o captura el rat\xC3\xB3n. Con \xC3\xA9l activo la c\xC3\xA1mara gira sin la aceleraci\xC3\xB3n del juego, tambi\xC3\xA9n con el stick derecho del mando: si juegas con mando, d\xC3\xA9jalo desactivado.", "Play with keyboard and mouse: WASD to move, the mouse moves the camera, left click R1, right click L1, wheel or middle click locks on. F7 releases or captures the mouse. With it on, the camera turns without the game's acceleration, also with the controller's right stick: if you play with a controller, leave it off.", nullptr, nullptr, nullptr, nullptr}},
	{{"Cu\xC3\xA1nto gira la c\xC3\xA1mara por cada movimiento del rat\xC3\xB3n (1.00 = normal).", "How much the camera turns per mouse movement (1.00 = normal).", nullptr, nullptr, nullptr, nullptr}},
	{{"Salta la zona muerta del stick del juego: con m\xC3\xA1s, un movimiento peque\xC3\xB1o del rat\xC3\xB3n ya gira la c\xC3\xA1mara. Si la c\xC3\xA1mara se mueve sola, b\xC3\xA1jalo.", "Skips the game's stick deadzone: higher means a small mouse movement already turns the camera. If the camera moves by itself, lower it.", nullptr, nullptr, nullptr, nullptr}},
	{{"La velocidad m\xC3\xA1xima de giro de la c\xC3\xA1mara del juego con teclado y rat\xC3\xB3n. 10 = el m\xC3\xA1ximo del men\xC3\xBA del juego. 11-15 (extra, experimental): m\xC3\xA1s r\xC3\xA1pido que el men\xC3\xBA; si la c\xC3\xA1mara hace cosas raras, vuelve a 10.", "The game camera's maximum turn speed with keyboard and mouse. 10 = the game menu's maximum. 11-15 (extra, experimental): faster than the menu; if the camera acts strange, go back to 10.", nullptr, nullptr, nullptr, nullptr}},
	{{"C\xC3\xA1mara instant\xC3\xA1nea (frame warp): la imagen gira con el rat\xC3\xB3n antes de que el juego dibuje el siguiente fotograma: la c\xC3\xA1mara responde al instante.", "Instant camera (frame warp): the picture turns with the mouse before the game draws the next frame: the camera responds instantly.", nullptr, nullptr, nullptr, nullptr}},
	{{"Mide c\xC3\xB3mo gira la c\xC3\xA1mara del juego para que el rat\xC3\xB3n la mueva exactamente lo que se mueve (se guarda en kbm_curve.txt; una vez basta). Al iniciar: carga tu partida, qu\xC3\xA9" "date quieto en un sitio abierto sin tocar el rat\xC3\xB3n ni el stick; a los 3 s la c\xC3\xA1mara gira sola unos 13 s.", "Measures how the game camera turns so the mouse moves it exactly (saved in kbm_curve.txt; once is enough). At start: load your save, stand still in an open spot without touching the mouse or the stick; after 3 s the camera turns by itself for about 13 s.", nullptr, nullptr, nullptr, nullptr}},
	{{"Vibraci\xC3\xB3n del mando.", "Controller vibration.", nullptr, nullptr, nullptr, nullptr}},
	{{"Fuerza de la vibraci\xC3\xB3n del mando.", "Controller vibration strength.", nullptr, nullptr, nullptr, nullptr}},
	{{"Las teclas de teclado y rat\xC3\xB3n (se cambian en mapa_teclado de settings.ini).", "The keyboard and mouse keys (changed in mapa_teclado in settings.ini).", nullptr, nullptr, nullptr, nullptr}},
	{{"La carpeta con el juego volcado (la que tiene eboot.bin y sce_sys).", "The folder with the dumped game (the one with eboot.bin and sce_sys).", nullptr, nullptr, nullptr, nullptr}},
	{{"Anade \"Demon's Souls\" a la biblioteca de Steam como juego externo (con icono e imagenes). Mejor con Steam cerrado; si esta abierto, reinicialo para que aparezca.", "Adds \"Demon's Souls\" to the Steam library as a non-Steam game (with icon and artwork). Best with Steam closed; if it is open, restart it for the game to appear.", nullptr, nullptr, nullptr, nullptr}},
	{{"Detiene la preparacion de shaders (lo hecho se guarda y sigue la proxima vez). Sin ella, el juego compila shaders al jugar (tirones).", "Stops the shader preparation (what is done is kept and it continues next time). Without it, the game compiles shaders while playing (stutter).", nullptr, nullptr, nullptr, nullptr}},
	{{"Guarda los ajustes en settings.ini y empieza el juego.", "Saves the settings to settings.ini and starts the game.", nullptr, nullptr, nullptr, nullptr}},
	{{"En el juego, F1 (o R3 + OPTIONS) guarda y cierra. Con esto marcado, JUGAR pasa solo la pantalla de t\xC3\xADtulo y carga la partida en el sitio donde pulsaste F1.", "In the game, F1 (or R3 + OPTIONS) saves and closes. With this ticked, PLAY passes the title screen by itself and loads your game at the spot where you pressed F1.", nullptr, nullptr, nullptr, nullptr}},
	{{"Detiene la preparacion de shaders (lo hecho se guarda y sigue la proxima vez).", "Stops the shader preparation (what is done is kept and it continues next time).", nullptr, nullptr, nullptr, nullptr}},
	{{"Compila ya todos los pipelines del juego para esta grafica (evita tirones la primera vez).", "Compiles all of the game's pipelines for this GPU now (avoids stutter the first time).", nullptr, nullptr, nullptr, nullptr}},
	{{"Busca en GitHub la ultima version publicada del port (DemonsSoulsPC) y la instala en esta misma carpeta. Nunca baja de version. Los archivos actuales del motor se guardan en _backup_update; la cache de shaders, las partidas, los seeds y los ajustes no se tocan.", "Checks GitHub for the latest published port version (DemonsSoulsPC) and installs it in this same folder. It never downgrades. The current engine files are kept in _backup_update; the shader cache, saves, seeds and settings are not touched.", nullptr, nullptr, nullptr, nullptr}},
	{{"Vuelve a una version guardada en _backup_update. La actual se guarda antes (se puede deshacer). La cache de shaders, las partidas, los seeds y los ajustes no se tocan.", "Goes back to a version kept in _backup_update. The current one is saved first (it can be undone). The shader cache, saves, seeds and settings are not touched.", nullptr, nullptr, nullptr, nullptr}},
	{{"La carpeta del juego y A\xC3\xB1" "adir a Steam.", "The game folder and Add to Steam.", nullptr, nullptr, nullptr, nullptr}},
	{{"Cierra el lanzador sin jugar (Esc).", "Closes the launcher without playing (Esc).", nullptr, nullptr, nullptr, nullptr}},
	{{"Teclado y rat\xC3\xB3n, sensibilidad, c\xC3\xA1mara y vibraci\xC3\xB3n.", "Keyboard and mouse, sensitivity, camera and vibration.", nullptr, nullptr, nullptr, nullptr}},
	{{"Adaptado para graficas de 8 GB como RTX 3060, RTX 3060 Ti, RTX 3070, RTX 2070/2080, RX 6600, RX 6600 XT o RX 6650 XT: el juego a 30 FPS estables y frame generation de AMD x2 (60 en pantalla). A 1080p, con el antialiasing de XeSS o de FSR (sin reescalar).", "Tuned for 8 GB GPUs such as RTX 3060, RTX 3060 Ti, RTX 3070, RTX 2070/2080, RX 6600, RX 6600 XT or RX 6650 XT: the game at a stable 30 FPS with AMD frame generation x2 (60 on screen). At 1080p, with XeSS or FSR antialiasing (no upscaling).", nullptr, nullptr, nullptr, nullptr}},
	{{"Para graficas de la potencia de una RTX 3080: RTX 2080 Ti, RTX 3070 Ti, RTX 3080, RTX 3080 Ti, RTX 4060 Ti, RTX 4070, RX 6800, RX 6800 XT, RX 6900 XT, RX 6950 XT, RX 7700 XT, RX 7800 XT o Intel Arc B580: el juego a 45 FPS estables y frame generation de AMD x2 (90 en pantalla). A 1440p, con el reescalado que elijas.", "For GPUs around RTX 3080 level: RTX 2080 Ti, RTX 3070 Ti, RTX 3080, RTX 3080 Ti, RTX 4060 Ti, RTX 4070, RX 6800, RX 6800 XT, RX 6900 XT, RX 6950 XT, RX 7700 XT, RX 7800 XT or Intel Arc B580: the game at a stable 45 FPS with AMD frame generation x2 (90 on screen). At 1440p, with the upscaler of your choice.", nullptr, nullptr, nullptr, nullptr}},
	{{"Para RTX 5060 / 5060 Ti / 5070 / 5070 Ti y RX 9070 / 9070 XT: todo al maximo, Detail Very High, Luz RT y Oclusion en Alta. Los FPS dependen de la potencia de tu GPU y tu CPU.", "For RTX 5060 / 5060 Ti / 5070 / 5070 Ti and RX 9070 / 9070 XT: everything maxed, Detail Very High, RT Light and Occlusion on High. FPS depends on how powerful your GPU and CPU are.", nullptr, nullptr, nullptr, nullptr}},
	{{"Para enfermos de Souls. RTX 5080, RTX 5090 o superior: modo cinematico del juego a 4K y absolutamente todo al maximo (Ultra en todo, filtrado x16 y texturas nitidas). Los FPS dependen de la potencia de tu GPU y tu CPU.", "For Souls addicts. RTX 5080, RTX 5090 or better: the game's 4K cinematic mode and absolutely everything maxed (Ultra on everything, x16 filtering and sharp textures). FPS depends on how powerful your GPU and CPU are.", nullptr, nullptr, nullptr, nullptr}},
	{{"Para PCs modestos: 30 FPS reales con frame generation de AMD x2 (60 en pantalla), el juego a 900p reescalado con FSR (nitidez 50 %) y todos los graficos en bajo (texturas, distancia, sombras, detalle, reflejos, objetos y nitidez). Al quitarlo vuelve todo a Maximo y a 1440p.", "For modest PCs: 30 real FPS with AMD frame generation x2 (60 on screen), the game at 900p upscaled with FSR (50 % sharpness) with every graphics option on low (textures, distance, shadows, detail, reflections, objects and sharpness). Unticking it puts everything back to Max and 1440p.", nullptr, nullptr, nullptr, nullptr}},
	{{"Idioma de este lanzador y de los textos del port en el juego. Autom\xC3\xA1tico: el de Windows.", "Language of this launcher and of the port's in-game texts. Automatic: Windows' language.", nullptr, nullptr, nullptr, nullptr}},
	{{"Idioma del juego (textos y voces). Sigue al del men\xC3\xBA hasta que elijas otro.", "The game's language (texts and voices). It follows the menu's until you pick another one.", nullptr, nullptr, nullptr, nullptr}},
	{{"Con este Modo Souls la frame generation es fija: AMD x2.", "With this Souls mode the frame generation is fixed: AMD x2.", nullptr, nullptr, nullptr, nullptr}},
	{{"En los modos Souls ligero, equilibrado y low profile va siempre el nuestro en Ultra.", "The Souls Light, Balanced and low profile modes always use ours on Ultra.", nullptr, nullptr, nullptr, nullptr}},
	{{"Todas las opciones: Modos Souls, imagen, frame generation, calidad gr\xC3\xA1" "fica, ray tracing, pantalla, controles, ajustes, actualizar y trofeos.", "All the options: Souls modes, image, frame generation, graphics quality, ray tracing, display, controls, settings, update and trophies.", nullptr, nullptr, nullptr, nullptr}},
	{{"No hacen falta: solo en PCs de 20 GB de RAM o menos.", "Not needed: only on PCs with 20 GB of RAM or less.", nullptr, nullptr, nullptr, nullptr}},
	{{"Si te gusta el proyecto, puedes donar por PayPal. Gracias, portador de la maldici\xC3\xB3n.", "If you enjoy the project, you can donate via PayPal. Thank you, bearer of the curse.", nullptr, nullptr, nullptr, nullptr}},
	// (port34) F1 > Ver trofeos
	{{"Ver trofeos", "View trophies", "Ver trof\xC3\xA9us", "Voir les troph\xC3\xA9" "es", "Vedi trofei", "Troph\xC3\xA4" "en ansehen"}},
	{{"TROFEOS", "TROPHIES", "TROF\xC3\x89US", "TROPH\xC3\x89" "ES", "TROFEI", "TROPH\xC3\x84" "EN"}},
	{{"%d de %d trofeos obtenidos", "%d of %d trophies earned", "%d de %d trof\xC3\xA9us conquistados", "%d troph\xC3\xA9" "es obtenus sur %d", "%d trofei ottenuti su %d", "%d von %d Troph\xC3\xA4" "en erhalten"}},
	{{"Todos", "All", "Todos", "Tous", "Tutti", "Alle"}},
	{{"Obtenidos", "Earned", "Conquistados", "Obtenus", "Ottenuti", "Erhalten"}},
	{{"Sin obtener", "Not earned", "N\xC3\xA3o conquistados", "Non obtenus", "Non ottenuti", "Nicht erhalten"}},
	{{"Ordenar:", "Sort:", "Ordenar:", "Trier :", "Ordina:", "Sortieren:"}},
	{{"Grado", "Grade", "Grau", "Rang", "Grado", "Stufe"}},
	{{"Fecha", "Date", "Data", "Date", "Data", "Datum"}},
	{{"Volver", "Back", "Voltar", "Retour", "Indietro", "Zur\xC3\xBC" "ck"}},
	{{"L1/R1 o Q/E: filtro   \xC2\xB7   Y o Tab: orden   \xC2\xB7   B, Esc o F1: volver", "L1/R1 or Q/E: filter   \xC2\xB7   Y or Tab: sort   \xC2\xB7   B, Esc or F1: back", "L1/R1 ou Q/E: filtro   \xC2\xB7   Y ou Tab: ordem   \xC2\xB7   B, Esc ou F1: voltar", "L1/R1 ou Q/E : filtre   \xC2\xB7   Y ou Tab : tri   \xC2\xB7   B, \xC3\x89" "chap ou F1 : retour", "L1/R1 o Q/E: filtro   \xC2\xB7   Y o Tab: ordine   \xC2\xB7   B, Esc o F1: indietro", "L1/R1 oder Q/E: Filter   \xC2\xB7   Y oder Tab: Sortierung   \xC2\xB7   B, Esc oder F1: zur\xC3\xBC" "ck"}},
	{{"Cargando los trofeos...", "Loading the trophies...", "Carregando os trof\xC3\xA9us...", "Chargement des troph\xC3\xA9" "es...", "Caricamento dei trofei...", "Troph\xC3\xA4" "en werden geladen..."}},
	{{"No se encontraron los trofeos del juego (falta sce_sys/trophy2/trophy00.ucp).", "The game's trophies were not found (sce_sys/trophy2/trophy00.ucp is missing).", "Os trof\xC3\xA9us do jogo n\xC3\xA3o foram encontrados (falta sce_sys/trophy2/trophy00.ucp).", "Troph\xC3\xA9" "es du jeu introuvables (sce_sys/trophy2/trophy00.ucp manquant).", "Trofei del gioco non trovati (manca sce_sys/trophy2/trophy00.ucp).", "Die Troph\xC3\xA4" "en des Spiels wurden nicht gefunden (sce_sys/trophy2/trophy00.ucp fehlt)."}},
	{{"A\xC3\xBAn no has obtenido ning\xC3\xBAn trofeo.", "You haven't earned any trophy yet.", "Voc\xC3\xAA ainda n\xC3\xA3o conquistou nenhum trof\xC3\xA9u.", "Vous n'avez encore obtenu aucun troph\xC3\xA9" "e.", "Non hai ancora ottenuto nessun trofeo.", "Du hast noch keine Troph\xC3\xA4" "e erhalten."}},
	{{"\xC2\xA1Has obtenido todos los trofeos!", "You have earned every trophy!", "Voc\xC3\xAA conquistou todos os trof\xC3\xA9us!", "Vous avez obtenu tous les troph\xC3\xA9" "es !", "Hai ottenuto tutti i trofei!", "Du hast alle Troph\xC3\xA4" "en erhalten!"}},
	{{"Se revelar\xC3\xA1 al obtenerlo.", "Revealed once earned.", "Ser\xC3\xA1 revelado ao conquist\xC3\xA1-lo.", "R\xC3\xA9v\xC3\xA9l\xC3\xA9 une fois obtenu.", "Verr\xC3\xA0 rivelato quando lo otterrai.", "Wird beim Erhalten enth\xC3\xBCllt."}},
	{{"Obtenido", "Earned", "Conquistado", "Obtenu", "Ottenuto", "Erhalten"}},
	{{"Obtenido el %s", "Earned on %s", "Conquistado em %s", "Obtenu le %s", "Ottenuto il %s", "Erhalten am %s"}},
};
// clang-format on

// (ds-pc-port8) The Spanish key without its accents (a, e, i, o, u, n, u): the code writes "Calidad gráfica", the table
// may have "Calidad grafica".
std::string Fold(const std::string& text) {
	static const std::pair<const char*, char> kFold[] = {{"\xC3\xA1", 'a'}, {"\xC3\xA9", 'e'}, {"\xC3\xAD", 'i'}, {"\xC3\xB3", 'o'},
	                                                     {"\xC3\xBA", 'u'}, {"\xC3\xB1", 'n'}, {"\xC3\xBC", 'u'}, {"\xC3\x81", 'A'},
	                                                     {"\xC3\x89", 'E'}, {"\xC3\x8D", 'I'}, {"\xC3\x93", 'O'}, {"\xC3\x9A", 'U'},
	                                                     {"\xC3\x91", 'N'}};
	std::string out;
	for (size_t i = 0; i < text.size(); i++) {
		bool folded = false;
		if (i + 1 < text.size() && static_cast<unsigned char>(text[i]) == 0xC3) {
			for (const auto& [from, to]: kFold) {
				if (text[i + 1] == from[1]) {
					out += to;
					i++;
					folded = true;
					break;
				}
			}
		}
		if (!folded) out += text[i];
	}
	return out;
}

const std::map<std::string, const Row*>& Table() {
	static const std::map<std::string, const Row*> table = [] {
		std::map<std::string, const Row*> t;
		for (const auto& row: kRows) t.emplace(Fold(row.t[0]), &row); // (the first row of a key wins)
		for (const auto& row: kRows) t[row.t[0]] = &row;
		return t;
	}();
	return table;
}

const Row* Find(const std::string& es) {
	const auto& table = Table();
	if (const auto it = table.find(es); it != table.end()) return it->second;
	if (const auto it = table.find(Fold(es)); it != table.end()) return it->second;
	return nullptr;
}

// The current language's text of a row; a row without it (the tooltips: Spanish and English only) gives the English one.
const char* Text(const Row* row) {
	if (row->t[g_lang] != nullptr) return row->t[g_lang];
	return row->t[1] != nullptr ? row->t[1] : row->t[0];
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
		const Row*        row = Find(key);
		if (row == nullptr) return es;
		auto& out = joined[std::string(es) + "|" + std::to_string(g_lang)];
		if (out.empty()) out = std::string(Text(row)) + hash;
		return out.c_str();
	}
	const Row* row = Find(es);
	return row == nullptr ? es : Text(row);
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
