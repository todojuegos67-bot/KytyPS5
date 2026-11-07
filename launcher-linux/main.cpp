// kyty_launcher: the settings window for Linux (launcher-windows' KytyLauncher.exe), built with the
// emulator and linked statically against SDL2 and Dear ImGui: nothing to install. It writes the
// same launcher-settings.json and starts the game through run_ds.sh with the settings as the
// emulator's switches (environment variables), the precompile through precompile-linux.sh and
// DLSS's build through get-dlss-linux.sh, in a terminal window.
//
// Command line (a double-click passes none):
//   --play             the game with the saved settings, without this window (Steam's launch
//                      options, a shortcut); this process lasts until the game exits and returns
//                      its exit code. No game: the window.
//   --lang en|zh|pt    the window's language for this run (the saved choice otherwise)
// For tests:
//   --dry-run          the scripts the play, precompile, hints and DLSS buttons run, printed
//   --screenshot <dir> each page saved to <dir>/<n>-<page>.bmp, then exit

#include "commands.h"
#include "platform.h"
#include "strings.h"
#include "ui.h"

#include <SDL.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>
#include <string>

namespace {

void Usage() {
	std::printf(
	    "kyty_launcher: KytyPS5's settings window for Linux\n"
	    "  --play                 start the game with the saved settings, without the window\n"
	    "  --lang en|zh|pt        the window's language for this run\n"
	    "  --dry-run              print the scripts the buttons run\n"
	    "  --screenshot <folder>  save each page as a picture, then exit\n");
}

// The primary display's height in pixels (Game output: Automatic), or 0.
int DisplayHeight() {
	SDL_DisplayMode mode {};
	return SDL_GetDesktopDisplayMode(0, &mode) == 0 ? mode.h : 0;
}

bool SaveScreenshot(SDL_Renderer* renderer, const std::string& file) {
	int w = 0, h = 0;
	if (SDL_GetRendererOutputSize(renderer, &w, &h) != 0) return false;
	SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_ARGB8888);
	if (surface == nullptr) return false;
	const bool read  = SDL_RenderReadPixels(renderer, nullptr, SDL_PIXELFORMAT_ARGB8888,
	                                        surface->pixels, surface->pitch) == 0;
	const bool saved = read && SDL_SaveBMP(surface, file.c_str()) == 0;
	SDL_FreeSurface(surface);
	return saved;
}

} // namespace

int main(int argc, char* argv[]) {
	bool        play = false, dry_run = false;
	std::string screenshots, lang;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg == "--play")
			play = true;
		else if (arg == "--dry-run")
			dry_run = true;
		else if (arg == "--screenshot" && i + 1 < argc)
			screenshots = argv[++i];
		else if (arg == "--lang" && i + 1 < argc)
			lang = argv[++i];
		else if (arg == "--help" || arg == "-h") {
			Usage();
			return 0;
		}
	}

	LauncherContext context;
	Strings::Current =
	    Strings::FromCode(lang.empty() ? context.settings->Str("ui_language") : lang);
	context.settings->read_only = dry_run || !screenshots.empty();

	// (Before the window: SDL's video for the display's size, which Automatic output needs, also
	// for --play.)
	SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0");
	const bool video = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) == 0;
	if (video) context.display_height = DisplayHeight();

	Commands commands(context);
	if (dry_run) {
		const auto game =
		    GameInfo::Read(context.settings->Str("game"), context.precompile_tool, context.root);
		std::printf("root:       %s\nemulator:   %s\nsettings:   %s\ndisplay:    %d lines\n",
		            context.root.c_str(), context.emulator.c_str(),
		            context.settings->Path().c_str(), context.display_height);
		for (const auto& gpu: context.gpus)
			std::printf("gpu:        %s (%llu MiB)\n", gpu.name.c_str(),
			            static_cast<unsigned long long>(gpu.memory >> 20u));
		std::printf("dlss:       %s\n", context.DlssAvailable() ? context.DlssLibrary().c_str()
		                                : context.rtx ? "no libkyty_dlss.so (get-dlss-linux.sh)"
		                                              : "no NVIDIA RTX graphics card");
		std::printf("prompts:    %s\n", commands.KeyboardPromptsReport().c_str());
		std::printf("e-cores:    %s\nterminal:   %s\ngame:       %s %s %s\n\n",
		            context.efficiency_cpus.empty() ? "-" : context.efficiency_cpus.c_str(),
		            context.ChosenTerminal() != nullptr ? context.ChosenTerminal()->name.c_str()
		                                                : "none",
		            game.title_id.c_str(), game.version.c_str(), game.name.c_str());
		std::printf("==== play\n%s\n==== precompile\n%s\n==== hints\n%s\n==== dlss\n%s",
		            Commands::Script(commands.Play(game, false)).c_str(),
		            Commands::Script(commands.Precompile("")).c_str(),
		            Commands::Script(commands.Precompile("--export-hints")).c_str(),
		            Commands::Script(commands.GetDlss()).c_str());
		return 0;
	}
	if (play) {
		const auto game =
		    GameInfo::Read(context.settings->Str("game"), context.precompile_tool, context.root);
		if (game.Found() && !context.emulator.empty()) {
			// The keyboard prompts' icons first (a Play of the window does the same): a failure is
			// shown, as the Windows launcher's message box, and nothing starts.
			if (const auto problem = commands.ApplyKeyboardPrompts(); !problem.empty()) {
				std::fprintf(stderr, "kyty_launcher: %s\n", problem.c_str());
				if (video)
					SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "KytyPS5", problem.c_str(), nullptr);
				if (video) SDL_Quit();
				return 1;
			}
			if (video) SDL_Quit();
			// (Steam counts the game as running while this process lasts.)
			return commands.RunDirect(commands.Play(game, true), true);
		}
	}
	if (!video) {
		std::fprintf(stderr, "kyty_launcher: no display (%s)\n", SDL_GetError());
		return 1;
	}

#ifdef SDL_HINT_IME_SHOW_UI
	SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif
	const float display_scale = std::max(1.0f, ImGui_ImplSDL2_GetContentScaleForDisplay(0));
	// (The window at the interface size chosen on the Advanced page, else the display's.)
	const int       chosen_scale = context.settings->Int("ui_scale");
	const float     window_scale = chosen_scale >= 50 && chosen_scale <= 300
	                                   ? static_cast<float>(chosen_scale) / 100.0f
	                                   : display_scale;
	SDL_DisplayMode mode {};
	SDL_GetDesktopDisplayMode(0, &mode);
	int width  = static_cast<int>(1120 * window_scale);
	int height = static_cast<int>(760 * window_scale);
	// (A small screen: the window fits it.)
	if (mode.w > 0 && width > mode.w - 16) width = mode.w - 16;
	if (mode.h > 0 && height > mode.h - 64) height = mode.h - 64;
	SDL_Window* window =
	    SDL_CreateWindow("KytyPS5", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height,
	                     SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
	if (window == nullptr) {
		std::fprintf(stderr, "kyty_launcher: no window (%s)\n", SDL_GetError());
		return 1;
	}
	SDL_SetWindowMinimumSize(window, std::min(static_cast<int>(900 * window_scale), width),
	                         std::min(static_cast<int>(560 * window_scale), height));
	// (A graphics driver's renderer, else SDL's software one: the window opens on any PC.)
	SDL_Renderer* renderer =
	    SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (renderer == nullptr) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
	if (renderer == nullptr) {
		std::fprintf(stderr, "kyty_launcher: no renderer (%s)\n", SDL_GetError());
		return 1;
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io    = ImGui::GetIO();
	io.IniFilename = nullptr;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
	ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
	ImGui_ImplSDLRenderer2_Init(renderer);

	{
		LauncherWindow launcher(context, renderer);
		launcher.ApplyScale(display_scale);
		int frames =
		    3; // frames still to draw before waiting for an event (ImGui settles over a couple)
		bool done = false;
		int  shot = screenshots.empty() ? -1 : 0;
		if (shot == 0) launcher.ShowPage(0);
		int shot_frames = 0;
		while (!done) {
			SDL_Event event;
			bool      any = false;
			if (frames <= 0 && shot < 0) {
				// (Asleep until something happens; twice a second for the checks that come from
				// outside.)
				if (SDL_WaitEventTimeout(&event, 500) != 0) {
					any = true;
					ImGui_ImplSDL2_ProcessEvent(&event);
					if (event.type == SDL_QUIT) done = true;
					if (event.type == SDL_WINDOWEVENT &&
					    event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED)
						launcher.Activated();
				}
			}
			while (SDL_PollEvent(&event) != 0) {
				any = true;
				ImGui_ImplSDL2_ProcessEvent(&event);
				if (event.type == SDL_QUIT) done = true;
				if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
				    event.window.windowID == SDL_GetWindowID(window))
					done = true;
				if (event.type == SDL_WINDOWEVENT &&
				    event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED)
					launcher.Activated();
			}
			if (any) frames = 3;
			launcher.Poll();
			if (launcher.ScaleChanged()) {
				launcher.ApplyScale(display_scale);
				frames = 3;
			}
			if ((SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) != 0 && shot < 0) {
				frames = 0;
				continue;
			}
			ImGui_ImplSDLRenderer2_NewFrame();
			ImGui_ImplSDL2_NewFrame();
			ImGui::NewFrame();
			launcher.Frame();
			ImGui::Render();
			SDL_RenderSetScale(renderer, io.DisplayFramebufferScale.x,
			                   io.DisplayFramebufferScale.y);
			SDL_SetRenderDrawColor(renderer, 0x10, 0x10, 0x12, 0xff);
			SDL_RenderClear(renderer);
			ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
			if (shot >= 0 && ++shot_frames >= 6) {
				// (Read before the present: the back buffer holds this frame.)
				const auto file = screenshots + "/" + std::to_string(shot + 1) + "-" +
				                  LauncherWindow::PageName(shot) + ".bmp";
				if (!SaveScreenshot(renderer, file))
					std::fprintf(stderr, "kyty_launcher: %s not saved (%s)\n", file.c_str(),
					             SDL_GetError());
				shot_frames = 0;
				if (++shot >= LauncherWindow::ScreenshotCount())
					done = true;
				else
					launcher.ShowPage(shot);
			}
			SDL_RenderPresent(renderer);
			--frames;
			if (launcher.WantsClose()) done = true;
		}
	}

	ImGui_ImplSDLRenderer2_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
