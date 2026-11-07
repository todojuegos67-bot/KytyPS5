#ifndef KYTY_LAUNCHER_LINUX_UI_H
#define KYTY_LAUNCHER_LINUX_UI_H

#include "commands.h"
#include "game.h"
#include "keyboard_editor.h"
#include "platform.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct SDL_Renderer;
struct SDL_Texture;
struct ImFont;

// The settings window (launcher-windows' MainWindow): the same pages and settings, drawn with Dear
// ImGui on SDL2's renderer, both linked in statically (nothing to install).
class LauncherWindow {
public:
	LauncherWindow(LauncherContext& context, SDL_Renderer* renderer);
	~LauncherWindow();

	// Fonts and sizes for the display's scale (and the Advanced page's interface size): before the
	// first frame, and after a change between frames.
	void ApplyScale(float display_scale);
	bool ScaleChanged() const { return scale_changed_; }

	// One frame of the window's content (between ImGui::NewFrame and ImGui::Render).
	void Frame();
	// Things that finish off the window's thread (a game read, the shaders' state, a folder dialog)
	// are taken in.
	void Poll();
	// The window came to the front: the shaders' state is asked again (at most every 30 s), DLSS's
	// runtime looked for.
	void Activated();

	[[nodiscard]] bool WantsClose() const { return close_; }
	// (--screenshot: the pages, then the keyboard controls editor over the Game page.)
	void               ShowPage(int index);
	static int         PageCount() { return 7; }
	static int         ScreenshotCount() { return PageCount() + 1; }
	static const char* PageName(int index);

	// Called from other threads when their work is done: wakes the window up (an SDL user event).
	static void Wake();

private:
	template <typename T>
	struct Job {
		std::atomic<bool> done {false};
		T                 value {};
	};

	// Pages.
	void Sidebar(float width);
	void BottomBar();
	void PlayPage();
	void DisplayPage();
	void GraphicsPage();
	void GamePage();
	void ShadersPage();
	void AdvancedPage();
	void ExperimentalPage();
	void Modals();

	// Widgets in the Windows launcher's look.
	void PageTitle(const char* key);
	void Section(const char* key);
	void Desc(const std::string& text);
	void Row(const char* label_key, const std::function<void()>& control,
	         const std::string& desc = {});
	bool Switch(const char* id, bool* value, bool enabled = true);
	bool SettingSwitch(const char* id, const char* key, bool enabled = true);
	bool Combo(const char* id, int* index, const std::vector<std::string>& labels, float width,
	           bool enabled = true);
	bool FlatButton(const char* label, bool enabled = true);
	void Feedback(int issue);
	void Card(const char* id, const std::function<void()>& content);

	// State.
	void LoadGame(const std::string& path);
	void CheckStore(bool force);
	void LoadBanner();
	void Changed();
	void Play();
	void StartInTerminal(const Launch& launch);
	void Message(const std::string& text);
	void PickFolder(bool zar);
	void OpenCheats();

	// The Graphics page's rules (launcher-windows MainWindow: SelectRenderHeight,
	// UpdateRenderNote...).
	[[nodiscard]] bool             RenderForUpscaler() const;
	[[nodiscard]] bool             HdOutputChosen() const;
	[[nodiscard]] int              OutputLines() const;
	[[nodiscard]] double           RatioAspect() const;
	[[nodiscard]] int              MaxRenderHeight() const;
	[[nodiscard]] std::vector<int> AllowedRenderHeights() const;
	[[nodiscard]] int              SelectedRender() const;
	[[nodiscard]] std::string      RenderLabel(int height) const;
	[[nodiscard]] std::string      SuperResolutionMode(int height) const;
	[[nodiscard]] std::string      RenderNote() const;
	[[nodiscard]] std::string
	Summary(std::vector<std::pair<std::string, std::string>>* lines) const;

	Settings&                     S() { return *context_.settings; }
	[[nodiscard]] const Settings& S() const { return *context_.settings; }

	LauncherContext& context_;
	Commands         commands_;
	SDL_Renderer*    renderer_;
	KeyboardEditor   keyboard_editor_; // Game page: Keyboard controls

	ImFont* sans_          = nullptr;
	ImFont* serif_         = nullptr;
	float   scale_         = 1.0f;
	float   display_scale_ = 1.0f;
	bool    scale_changed_ = false;
	bool    fonts_loaded_  = false;

	int  page_  = 0;
	bool close_ = false;

	// The game.
	GameInfo                              game_;
	std::string                           game_text_;
	std::chrono::steady_clock::time_point game_edit_ {};
	bool                                  game_edit_pending_ = false;
	std::shared_ptr<Job<GameInfo>>        game_job_;
	SDL_Texture*                          banner_   = nullptr;
	int                                   banner_w_ = 0, banner_h_ = 0;

	// The precompiled shaders.
	StoreState                            store_ = StoreState::Unknown;
	std::shared_ptr<Job<StoreState>>      store_job_;
	std::chrono::steady_clock::time_point store_checked_ {};

	// The render heights last chosen for each kind of upscaling (a list that lacks one keeps it for
	// later).
	int wanted_taa_ = 1440;
	int wanted_sr_  = 1080;

	std::string extra_text_;
	bool        dlss_available_ = false;

	// Dialogs.
	std::string                       message_;
	bool                              message_open_ = false;
	std::shared_ptr<Job<std::string>> picker_job_;
	bool        browser_open_ = false; // the built-in folder browser (no zenity or kdialog)
	bool        browser_zar_  = false;
	std::string browser_path_;
	std::string browser_selected_;
	fs::path    cheat_path_;
	CheatFile   cheat_file_;
	bool        cheats_open_ = false;
	int         cheat_hover_ = -1;
	std::string notice_; // a line in the bottom bar after a start in a terminal
	std::chrono::steady_clock::time_point notice_at_ {};
};

#endif
