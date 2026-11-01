#include "commands.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <regex>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// The variables a start of the game decides: each is set or left unset by the launcher, whatever
// the shell it was started from exported (as run-windows.ps1 builds the emulator's environment from
// its options).
const std::vector<std::string>& GameVariables() {
	static const std::vector<std::string> names {"KYTY_GAME",
	                                             "KYTY_BIN",
	                                             "KYTY_VBLANK",
	                                             "PRESENT_MODE",
	                                             "MANGO",
	                                             "KYTY_PRESENT_ASPECT",
	                                             "KYTY_LOW_LATENCY",
	                                             "KYTY_RENDER_HEIGHT",
	                                             "KYTY_RENDER_ASPECT",
	                                             "KYTY_GAME_CONVARS",
	                                             "KYTY_TEXTURE_REDUCE",
	                                             "KYTY_FRAMEGEN",
	                                             "KYTY_FRAMEGEN_BACKEND",
	                                             "KYTY_UPSCALER",
	                                             "KYTY_HD_OUTPUT",
	                                             "KYTY_OUTPUT_HEIGHT",
	                                             "KYTY_GPU",
	                                             "KYTY_FPS_HUD",
	                                             "KYTY_COMPUTE_CHAIN",
	                                             "KYTY_PRESENT_DIRECT",
	                                             "KYTY_VRR",
	                                             "KYTY_ANTI_LAG",
	                                             "KYTY_FLIP_RATE",
	                                             "KYTY_DLSS_PRESET",
	                                             "KYTY_DLSS_SHARPNESS",
	                                             "KYTY_FSR_SHARPNESS",
	                                             "KYTY_AUDIO_OUTPUT"};
	return names;
}

// The emulator's options that take no value (src/main.cpp).
bool IsFlag(const std::string& option) {
	return option == "--fullscreen" || option == "--hdr" || option == "--rd" ||
	       option == "--playgo-hack" || option == "--help";
}

bool IsVariableName(const std::string& text) {
	if (text.empty() || !(std::isalpha(static_cast<unsigned char>(text[0])) != 0 || text[0] == '_'))
		return false;
	return std::all_of(text.begin(), text.end(), [](char c) {
		return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
	});
}

void Put(std::vector<std::pair<std::string, std::string>>& set, const std::string& key,
         const std::string& value) {
	const auto it =
	    std::find_if(set.begin(), set.end(), [&](const auto& pair) { return pair.first == key; });
	if (it != set.end())
		it->second = value;
	else
		set.emplace_back(key, value);
}

bool Remove(std::vector<std::pair<std::string, std::string>>& set, const std::string& key) {
	const auto it =
	    std::find_if(set.begin(), set.end(), [&](const auto& pair) { return pair.first == key; });
	if (it == set.end()) return false;
	set.erase(it);
	return true;
}

std::string Stamp() {
	const std::time_t now = std::time(nullptr);
	std::tm           local {};
	localtime_r(&now, &local);
	char text[32];
	std::strftime(text, sizeof(text), "%Y%m%d-%H%M%S", &local);
	return text;
}

// The script in the user's runtime folder (private to them), executable: a terminal is given its
// path alone, which every terminal's way of taking a command runs the same.
std::string WriteScript(const std::string& text) {
	std::string dir = "/tmp";
	if (const char* runtime = std::getenv("XDG_RUNTIME_DIR");
	    runtime != nullptr && *runtime != 0 && access(runtime, W_OK) == 0)
		dir = runtime;
	std::string path = dir + "/kyty-launch-XXXXXX";
	const int   fd   = mkstemp(path.data());
	if (fd < 0) return "";
	const auto written = write(fd, text.data(), text.size());
	fchmod(fd, 0700);
	close(fd);
	if (written != static_cast<ssize_t>(text.size())) {
		unlink(path.c_str());
		return "";
	}
	return path;
}

} // namespace

bool Commands::Dlss() const {
	return context_.DlssAvailable() && S().Str("upscaler") == "dlss";
}

bool Commands::FsrUpscaler() const {
	return S().Str("upscaler") == "fsr";
}

int Commands::WindowHeight(const std::string& resolution, bool fullscreen, int display) {
	if (fullscreen) return display;
	const auto x = resolution.find('x');
	if (x == std::string::npos) return 0;
	const int height = std::atoi(resolution.c_str() + x + 1);
	if (height <= 0) return 0;
	return display > 0 && height > display ? display : height;
}

bool Commands::AutoHdOutput(const std::string& resolution, bool fullscreen,
                            const std::string& ratio, int display) {
	const int height = WindowHeight(resolution, fullscreen, display);
	return ratio == "16:9" && height > 0 && height <= 1080;
}

bool Commands::Output1440() const {
	return S().Str("output") == "1440p" && (Dlss() || FsrUpscaler());
}

bool Commands::HdOutput() const {
	const auto output = S().Str("output");
	return output == "1080p" ||
	       (output == "auto" && AutoHdOutput(S().Str("resolution"), S().Bool("fullscreen"),
	                                         S().Str("ratio"), context_.display_height));
}

Commands::Extra Commands::ParseExtra(const std::string& text) {
	// Words as a shell splits them, "double quotes" keeping spaces in one.
	std::vector<std::string> words;
	std::string              word;
	bool                     quoted = false, any = false;
	for (const char c: text) {
		if (c == '"') {
			quoted = !quoted;
			any    = true;
		} else if (std::isspace(static_cast<unsigned char>(c)) != 0 && !quoted) {
			if (any) words.push_back(word);
			word.clear();
			any = false;
		} else {
			word += c;
			any = true;
		}
	}
	if (any) words.push_back(word);
	Extra extra;
	auto  pair = [&](const std::string& text) {
        const auto equals = text.find('=');
        if (equals != std::string::npos && IsVariableName(text.substr(0, equals)))
            extra.set.emplace_back(text.substr(0, equals), text.substr(equals + 1));
	};
	for (size_t i = 0; i < words.size(); ++i) {
		const auto& w = words[i];
		if (w == "-Set" || w == "-set") {
			// (The Windows launcher's form: -Set A=1,B=2; a comma starts a new pair only before
			// KEY=.)
			if (i + 1 >= words.size()) break;
			static const std::regex split(",(?=[A-Za-z_][A-Za-z0-9_]*=)");
			const auto&             value = words[++i];
			for (std::sregex_token_iterator it(value.begin(), value.end(), split, -1), end;
			     it != end; ++it)
				pair(*it);
		} else if (w.size() > 2 && w.compare(0, 2, "--") == 0) {
			std::vector<std::string> option {w};
			if (!IsFlag(w) && i + 1 < words.size() && words[i + 1].compare(0, 2, "--") != 0)
				option.push_back(words[++i]);
			extra.options.push_back(option);
		} else {
			pair(w);
		}
	}
	return extra;
}

Launch Commands::Play(const GameInfo& game, bool direct) const {
	Launch launch;
	launch.title     = "KytyPS5";
	launch.directory = context_.root;
	launch.pause     = direct ? Launch::Pause::Never : Launch::Pause::OnError;
	launch.log       = context_.LogsFolder() / (Stamp() + ".log");
	auto& set        = launch.set;

	const auto resolution = S().Str("resolution");
	const auto x          = resolution.find('x');
	const auto width      = x == std::string::npos ? std::string("1920") : resolution.substr(0, x);
	const auto height     = x == std::string::npos ? std::string("1080") : resolution.substr(x + 1);
	std::vector<std::vector<std::string>> options {
	    {"--screen-width", width},
	    {"--screen-height", height},
	    {"--console-language", std::to_string(S().Int("language"))}};
	const auto ratio = S().Str("ratio");
	const bool wide  = ratio == "21:9" || ratio == "32:9";
	// (KYTY_PRESENT_ASPECT=fit: the 16:9 picture between black bars on a screen of another shape;
	// with 21:9 or 32:9 the frames without the 3D view, the movies, the title and the loading
	// screens, which fill the screen stretched otherwise: "wide_bars", in a window too.)
	if (S().Bool("fullscreen")) {
		options.push_back({"--fullscreen"});
		if (S().Bool("aspect") && !wide) Put(set, "KYTY_PRESENT_ASPECT", "fit");
	}
	if (wide && S().Bool("wide_bars")) Put(set, "KYTY_PRESENT_ASPECT", "fit");
	if (S().Bool("hdr")) options.push_back({"--hdr"});
	// The game's frames up to 120 fps: a 120 Hz vblank (run_ds.sh gives the game its least step
	// above 60). The 30 fps lock (KYTY_FLIP_RATE=1 below) keeps 60.
	Put(set, "KYTY_VBLANK", S().Bool("fps120") && !S().Bool("fps30") ? "120" : "60");
	// Presents wait for the display's refresh (Vulkan FIFO); run_ds.sh's mailbox otherwise.
	if (S().Bool("vsync")) Put(set, "PRESENT_MODE", "Fifo");
	// (AMD Anti-Lag, an experiment, works on top of low latency.)
	if (S().Bool("lowlatency") || S().Bool("antilag")) Put(set, "KYTY_LOW_LATENCY", "1");
	// The cheats file of this title, when it has one: the emulator applies the mods left enabled in
	// it.
	if (S().Bool("cheats")) {
		const auto patch = game.CheatFile(context_.root);
		if (!patch.empty()) options.push_back({"--game-patch", patch.string()});
	}
	const bool dlss = Dlss();
	// The upscaler's own render height (a chosen height is always passed: without one the game
	// follows the window); DLSS and FSR 3 share the PS5 Pro path's.
	const int render = dlss || FsrUpscaler() ? S().Int("render_dlss") : S().Int("render");
	if (render > 0) Put(set, "KYTY_RENDER_HEIGHT", std::to_string(render));
	if (wide) Put(set, "KYTY_RENDER_ASPECT", ratio);
	// Graphics presets (run-windows.ps1 -Quality, -TextureReduce): Demon's Souls' engine settings
	// that cut what a frame draws, and textures without their largest levels.
	auto quality = S().Str("quality");
	for (auto& c: quality)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	std::string convars;
	if (quality == "low" || quality == "lowest") {
		Put(set, "KYTY_TEXTURE_REDUCE", quality == "lowest" ? "2" : "1");
		if (game.DemonsSouls()) {
			convars = quality == "low"
			              ? "r_forceViewportHeightForPixelSizeCalc=720;sunShadowMaxDistance=125"
			              : "r_forceViewportHeightForPixelSizeCalc=480;sunShadowMaxDistance=60;"
			                "r_enableAmbientShadowing=false;r_enableSpotShadows=false;"
			                "r_projectedDecalEnable=false";
		}
	}
	// Upscaler "Off" (run-windows.ps1 -Antialiasing off, issue #220): Demon's Souls' TAA off and
	// the camera jitter TAA needs with it; the game's own upscaling stays.
	if (S().Str("upscaler") == "off" && game.DemonsSouls()) {
		if (!convars.empty()) convars += ";";
		convars += "r_aaMethod=0;r_aaEnableTemporalProjectMatrixJitter=false";
	}
	if (!convars.empty()) Put(set, "KYTY_GAME_CONVARS", convars);
	// Frame generation: FSR 3's (built in, any graphics card), else DLSS's (KYTY_FRAMEGEN =
	// generated frames per frame).
	const bool fsr_frames = S().Bool("fsr");
	const int  frame_gen  = S().Int("framegen");
	if (fsr_frames) {
		Put(set, "KYTY_FRAMEGEN", "1");
		Put(set, "KYTY_FRAMEGEN_BACKEND", "fsr");
	} else if (context_.DlssAvailable() && frame_gen >= 2) {
		Put(set, "KYTY_FRAMEGEN", std::to_string(frame_gen - 1));
	}
	if (dlss)
		Put(set, "KYTY_UPSCALER", "dlss");
	else if (FsrUpscaler())
		Put(set, "KYTY_UPSCALER", "fsr");
	// (With DLSS too: its PS5 Pro path then makes the 1080p output, the 3D at most 1080 high.)
	if (HdOutput())
		Put(set, "KYTY_HD_OUTPUT", "1");
	else if (Output1440())
		Put(set, "KYTY_OUTPUT_HEIGHT", "1440");
	// The graphics card chosen on the Advanced page (a part of its name, as the emulator matches
	// it).
	if (!S().Str("gpu").empty()) Put(set, "KYTY_GPU", S().Str("gpu"));
	// The emulator's panel at the top right of the game image (F8 in the game switches it).
	if (const int hud = S().Int("hud"); hud >= 1)
		Put(set, "KYTY_FPS_HUD", std::to_string(std::min(hud, 2)));
	// Safe GPU mode: Demon's Souls' compute work one dispatch after another.
	if (S().Bool("safegpu")) Put(set, "KYTY_COMPUTE_CHAIN", "0");
	if (S().Bool("present_direct")) Put(set, "KYTY_PRESENT_DIRECT", "1");
	// Frames flip as soon as they are done, not on the 60 Hz vblanks (VRR displays, issue #221).
	if (S().Bool("vrr_flips")) Put(set, "KYTY_VRR", "1");
	if (S().Bool("antilag")) Put(set, "KYTY_ANTI_LAG", "1");
	// The game's 3D view at most every 2nd vblank: a steady 30 fps at the same game speed.
	if (S().Bool("fps30")) Put(set, "KYTY_FLIP_RATE", "1");
	// The DLSS model (a preset letter; none: the DLSS version's own choice).
	if (const auto preset = S().Str("dlss_preset");
	    dlss && preset.size() == 1 && preset[0] >= 'A' && preset[0] <= 'O')
		Put(set, "KYTY_DLSS_PRESET", preset);
	// The sharpening chosen on the Graphics page: DLSS's (KYTY_DLSS_SHARPNESS, 0-1; none: not sharpened), FSR 3's
	// (KYTY_FSR_SHARPNESS; none: its 0.6).
	if (const auto sharpness = S().Str("dlss_sharpness");
	    dlss && (sharpness == "0.3" || sharpness == "0.6" || sharpness == "0.9"))
		Put(set, "KYTY_DLSS_SHARPNESS", sharpness);
	if (const auto sharpness = S().Str("fsr_sharpness");
	    FsrUpscaler() && (sharpness == "0" || sharpness == "0.3" || sharpness == "0.9"))
		Put(set, "KYTY_FSR_SHARPNESS", sharpness);
	// The audio output the game is told (it mixes for it; a TV when none is set, issue #207).
	if (const auto output = S().Str("audio_output"); output == "headphones" || output == "amplifier" || output == "soundbar")
		Put(set, "KYTY_AUDIO_OUTPUT", output);
	Put(set, "MANGO", S().Bool("mangohud") && context_.HasMangoHud() ? "1" : "0");
	Put(set, "KYTY_GAME", S().Str("game"));
	if (!context_.emulator.empty()) Put(set, "KYTY_BIN", context_.emulator.string());

	// The extra options: their pairs replace the launcher's (KEY= removes one the launcher set,
	// else turns the emulator's own default off, 0), their --options the launcher's of the same
	// name.
	const auto extra = ParseExtra(S().Str("extra_args"));
	for (const auto& [key, value]: extra.set) {
		if (!value.empty())
			Put(set, key, value);
		else if (!Remove(set, key))
			Put(set, key, "0");
	}
	// (Each of them stays: a repeated one, --keymap, is one more, not in place of the one before.)
	for (const auto& option: extra.options)
		options.erase(std::remove_if(options.begin(), options.end(),
		                             [&](const auto& o) { return o[0] == option[0]; }),
		              options.end());
	options.insert(options.end(), extra.options.begin(), extra.options.end());

	launch.argv = {context_.run_script.string()};
	for (const auto& option: options)
		launch.argv.insert(launch.argv.end(), option.begin(), option.end());
	for (const auto& name: GameVariables())
		if (std::none_of(set.begin(), set.end(),
		                 [&](const auto& pair) { return pair.first == name; }))
			launch.unset.push_back(name);
	return launch;
}

Launch Commands::Precompile(const std::string& mode) const {
	Launch launch;
	launch.title     = "KytyPS5 precompile";
	launch.directory = context_.root;
	launch.pause     = Launch::Pause::Always;
	launch.argv      = {context_.precompile_script.string()};
	if (!S().Str("game").empty())
		launch.argv.insert(launch.argv.end(), {"--game", S().Str("game")});
	if (!mode.empty())
		launch.argv.push_back(mode);
	else if (S().Bool("precompile_quick"))
		launch.argv.push_back("--quick");
	if (S().Bool("ecores") && !context_.efficiency_cpus.empty())
		launch.argv.insert(launch.argv.end(), {"--cpus", context_.efficiency_cpus});
	// (The pipelines are compiled for one card: the one the game uses.)
	if (!S().Str("gpu").empty())
		launch.set.emplace_back("KYTY_GPU", S().Str("gpu"));
	else
		launch.unset.emplace_back("KYTY_GPU");
	return launch;
}

Launch Commands::GetDlss() const {
	Launch launch;
	launch.title     = "KytyPS5 DLSS";
	launch.directory = context_.root;
	launch.pause     = Launch::Pause::Always;
	launch.argv      = {context_.dlss_script.string()};
	// (Next to the emulator, where it looks.)
	if (!context_.emulator.empty()) {
		std::error_code error;
		const auto      real = fs::canonical(context_.emulator, error);
		launch.argv.push_back((error ? context_.emulator : real).parent_path().string());
	}
	return launch;
}

// A word of the command as typed: quoted only when it has to be.
static std::string Word(const std::string& text) {
	const bool plain = !text.empty() && std::all_of(text.begin(), text.end(), [](char c) {
		return std::isalnum(static_cast<unsigned char>(c)) != 0 ||
		       std::strchr("_./:=,+-@%", c) != nullptr;
	});
	return plain ? text : Platform::Quote(text);
}

std::string Commands::Script(const Launch& launch) {
	using Platform::Quote;
	std::string text = "#!/usr/bin/env bash\n# KytyPS5 launcher: " + launch.title +
	                   " (this file deletes itself)\n";
	text += "rm -f -- \"$0\"\n";
	text += "printf '\\033]0;%s\\007' " + Quote(launch.title) + "\n";
	if (!launch.directory.empty())
		text += "cd -- " + Quote(launch.directory.string()) + " || exit 1\n";
	if (!launch.unset.empty()) {
		text += "unset";
		for (const auto& name: launch.unset)
			text += " " + name;
		text += "\n";
	}
	for (const auto& [key, value]: launch.set)
		text += "export " + key + "=" + Quote(value) + "\n";
	std::string command;
	// (A script through bash: an unpacker that dropped its execute permission does not stop it.)
	if (!launch.argv.empty() && launch.argv[0].size() > 3 &&
	    launch.argv[0].compare(launch.argv[0].size() - 3, 3, ".sh") == 0)
		command = "bash";
	for (const auto& arg: launch.argv)
		command += (command.empty() ? "" : " ") + Word(arg);
	// What runs, for a bug report.
	text += "echo " + Quote("command: " + command) + "\n";
	if (!launch.log.empty()) {
		text += "mkdir -p -- " + Quote(launch.log.parent_path().string()) + "\n";
		text += "echo " + Quote("log:     " + launch.log.string()) + "\n";
		// (Line by line into the log as the emulator prints, not in blocks: stdbuf where coreutils
		// has it.)
		text += "if command -v stdbuf >/dev/null 2>&1; then\n";
		text +=
		    "  stdbuf -oL -eL " + command + " 2>&1 | tee -- " + Quote(launch.log.string()) + "\n";
		text += "else\n";
		text += "  " + command + " 2>&1 | tee -- " + Quote(launch.log.string()) + "\n";
		text += "fi\n";
		text += "code=${PIPESTATUS[0]}\n";
	} else {
		text += command + "\n";
		text += "code=$?\n";
	}
	if (launch.pause != Launch::Pause::Never) {
		text += launch.pause == Launch::Pause::OnError ? "if [ \"$code\" -ne 0 ]; then\n"
		                                               : "if true; then\n";
		text += "  echo\n";
		text += "  echo \"Exited with code $code.\"\n";
		text += "  [ -t 0 ] && read -r -p 'Press Enter to close this window. ' _\n";
		text += "fi\n";
	}
	text += "exit \"$code\"\n";
	return text;
}

bool Commands::StartInTerminal(const Launch& launch, std::string* error) const {
	const Terminal* terminal = context_.ChosenTerminal();
	if (terminal == nullptr) {
		if (error != nullptr) *error = "no terminal program found";
		return false;
	}
	const auto script = WriteScript(Script(launch));
	if (script.empty()) {
		if (error != nullptr) *error = "cannot write the start script";
		return false;
	}
	std::vector<std::string> argv {terminal->name};
	argv.insert(argv.end(), terminal->before.begin(), terminal->before.end());
	argv.push_back(script);
	if (!Platform::Spawn(argv, launch.directory)) {
		unlink(script.c_str());
		if (error != nullptr) *error = terminal->name + " did not start";
		return false;
	}
	return true;
}

int Commands::RunDirect(const Launch& launch, bool wait) const {
	const auto script = WriteScript(Script(launch));
	if (script.empty()) return -1;
	if (!wait) {
		if (Platform::Spawn({"bash", script}, launch.directory)) return 0;
		unlink(script.c_str());
		return -1;
	}
	const pid_t pid = fork();
	if (pid < 0) {
		unlink(script.c_str());
		return -1;
	}
	if (pid == 0) {
		execlp("bash", "bash", script.c_str(), static_cast<char*>(nullptr));
		_exit(127);
	}
	int status = 0;
	while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
	}
	return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
