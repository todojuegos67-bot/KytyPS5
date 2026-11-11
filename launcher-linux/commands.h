#ifndef KYTY_LAUNCHER_LINUX_COMMANDS_H
#define KYTY_LAUNCHER_LINUX_COMMANDS_H

#include "game.h"
#include "platform.h"

#include <string>
#include <utility>
#include <vector>

// A command the buttons start: run_ds.sh, precompile-linux.sh or get-dlss-linux.sh, with the
// emulator's switches as environment variables (launcher-windows' Commands.cs builds
// run-windows.ps1's command line the same way).
struct Launch {
	enum class Pause { Never, OnError, Always };

	std::string                                      title;
	std::vector<std::pair<std::string, std::string>> set; // environment variables
	std::vector<std::string> unset; // variables the launcher decides that it leaves unset
	std::vector<std::string> argv;
	fs::path                 directory;
	fs::path                 log; // a copy of the output (tee), or empty
	Pause                    pause = Pause::Never;
};

class Commands {
public:
	explicit Commands(const LauncherContext& context): context_(context) {}

	// DLSS chosen and able to run (an NVIDIA RTX graphics card, libkyty_dlss.so).
	[[nodiscard]] bool Dlss() const;
	// FSR 3 upscaling (built into the emulator: any graphics card).
	[[nodiscard]] bool FsrUpscaler() const;
	// The game's 1080p output: chosen, or Automatic for a 16:9 window at most 1080 lines high.
	[[nodiscard]] bool HdOutput() const;
	// Game output "1440p", with DLSS or FSR 3 only: the PS5 Pro path's output at 2560x1440
	// (KYTY_OUTPUT_HEIGHT). The game's TAA has no such output: 4K.
	[[nodiscard]] bool Output1440() const;

	// The game window's height in pixels: in full screen the display's, else the chosen size's (a
	// window larger than the display is shrunk); 0 when unknown.
	static int  WindowHeight(const std::string& resolution, bool fullscreen, int display);
	static bool AutoHdOutput(const std::string& resolution, bool fullscreen,
	                         const std::string& ratio, int display);

	// The Game page's "Keyboard and mouse prompts": the 22 button icons of the game folder swapped
	// for key caps that match the saved layout (or the originals put back when the option is off),
	// as Play needs it done before the emulator starts (launcher-windows' StartPlay). Empty when done,
	// else why not (the text is shown as it is). Plays are the only caller: --dry-run changes nothing.
	[[nodiscard]] std::string ApplyKeyboardPrompts() const;
	// Whether the option is on, and the saved layout is usable ("" or why not): --dry-run's report.
	[[nodiscard]] std::string KeyboardPromptsReport() const;

	// direct (--play, a start from Steam or a shortcut): no window waits for a key afterwards.
	[[nodiscard]] Launch Play(const GameInfo& game, bool direct) const;
	// mode: "" (every shader), "--export-hints".
	[[nodiscard]] Launch Precompile(const std::string& mode) const;
	[[nodiscard]] Launch GetDlss() const;

	// The bash script that runs a launch (it deletes itself when it starts).
	[[nodiscard]] static std::string Script(const Launch& launch);
	// In a terminal window (the one chosen on the Advanced page, else the first found); false with
	// the reason.
	bool StartInTerminal(const Launch& launch, std::string* error) const;
	// Without a terminal: detached (its output only in its log), or waited for (its exit code;
	// --play).
	int RunDirect(const Launch& launch, bool wait) const;

	// The Advanced page's extra options: KEY=VALUE pairs and the emulator's --options.
	struct Extra {
		std::vector<std::pair<std::string, std::string>> set;
		std::vector<std::vector<std::string>> options; // each --name with its value, if it has one
	};
	[[nodiscard]] static Extra ParseExtra(const std::string& text);

private:
	[[nodiscard]] const Settings& S() const { return *context_.settings; }

	const LauncherContext& context_;
};

#endif
