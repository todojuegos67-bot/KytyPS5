using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace KytyLauncher {

// The command lines the buttons run: run-windows.ps1 and precompile-windows.ps1 in a console window of their own, as
// launcher.ps1 started them (Get-PlayCommand, Get-PrecompileCommand), so run.cmd and the launcher stay one path.
sealed class Commands {
	readonly LauncherContext context;

	public Commands(LauncherContext context) {
		this.context = context;
	}

	Settings S { get { return context.Settings; } }

	// Windows PowerShell and cmd.exe by their paths: a PC whose PATH lacks their folder did not know "powershell"
	// (issue #55), and a file named "cmd" in the working folder made Windows ask which app should open it (issue #47).
	static string PowerShell() {
		var exe = Path.Combine(Environment.SystemDirectory, "WindowsPowerShell", "v1.0", "powershell.exe");
		return File.Exists(exe) ? "\"" + exe + "\"" : "powershell";
	}

	public static string Cmd() {
		var comspec = Environment.GetEnvironmentVariable("ComSpec");
		return !string.IsNullOrEmpty(comspec) && File.Exists(comspec) ? comspec : Path.Combine(Environment.SystemDirectory, "cmd.exe");
	}

	public bool Dlss { get { return context.FrameGenAvailable && S.Str("upscaler") == "dlss"; } }
	// FSR 3 upscaling, built into the emulator: any graphics card.
	public bool FsrUpscaler { get { return S.Str("upscaler") == "fsr"; } }

	// The game window's height in pixels: in full screen the display's, else the chosen size's (a window larger than the
	// display is shrunk by the emulator); 0 when unknown.
	public static int WindowHeight(string resolution, bool fullscreen, int display) {
		if (fullscreen) return display;
		var parts = (resolution ?? "").Split('x');
		int height;
		if (parts.Length < 2 || !int.TryParse(parts[1], out height)) return 0;
		return display > 0 && height > display ? display : height;
	}

	// Game output "Automatic": the game's 1080p output for a 16:9 game window (or screen, in full screen) at most 1080
	// lines high, which shows it pixel for pixel and saves the 4K post-processing; 4K otherwise.
	public static bool AutoHdOutput(string resolution, bool fullscreen, string ratio, int display) {
		var height = WindowHeight(resolution, fullscreen, display);
		return ratio == "16:9" && height > 0 && height <= 1080;
	}

	public bool HdOutput {
		get {
			var output = S.Str("output");
			return output == "1080p" ||
			       (output == "auto" && AutoHdOutput(S.Str("resolution"), S.Bool("fullscreen"), S.Str("ratio"), context.DisplayHeight));
		}
	}

	// Game output "1440p", with DLSS or FSR 3 only: the PS5 Pro path's output at 2560x1440 (run-windows.ps1 -OutputHeight,
	// KYTY_OUTPUT_HEIGHT: a 1440p screen gets the upscaler's own picture). The game's TAA has no such output: 4K.
	public bool Output1440 { get { return S.Str("output") == "1440p" && (Dlss || FsrUpscaler); } }

	// direct (--play, a start from Steam or a shortcut, maybe with only a gamepad at hand): no dialogs (-Prompt) and no
	// wait for a key after a crash.
	public string PlayCommand(GameInfo game = null, bool direct = false) {
		var size = S.Str("resolution").Split('x');
		var arguments = new List<string> { "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", Quote(Path.Combine(context.Root, "run-windows.ps1")) };
		if (!direct) arguments.Add("-Prompt");
		arguments.AddRange(new[] { "-Follow", "-Width", size[0], "-Height", size.Length > 1 ? size[1] : "1080", "-Language", Number(S.Int("language")) });
		var ratio = S.Str("ratio");
		var wide = ratio == "21:9" || ratio == "32:9";
		// (-AspectFit: the 16:9 picture between black bars on a screen of another shape; with 21:9 or 32:9 the frames without
		// the 3D view, the movies, the title and the loading screens, which fill the screen stretched otherwise: "wide_bars",
		// in a window too.)
		if (S.Bool("fullscreen")) {
			arguments.Add("-Fullscreen");
			if (S.Bool("aspect") && !wide) arguments.Add("-AspectFit");
		}
		if (wide && S.Bool("wide_bars")) arguments.Add("-AspectFit");
		if (!S.Bool("redzone")) arguments.Add("-NoRedZone");
		// (The 30 fps lock, Experimental: KYTY_FLIP_RATE=1 below, at 60 Hz: up to 120 fps it would be 60.)
		if (S.Bool("fps120") && !S.Bool("fps30")) arguments.Add("-Fps120");
		// Presents wait for the display's refresh (Vulkan FIFO): no tearing, and G-SYNC/FreeSync displays keep adapting.
		if (S.Bool("vsync")) arguments.AddRange(new[] { "-PresentMode", "Fifo" });
		// (AMD Anti-Lag, an experiment, works on top of low latency.)
		if (S.Bool("lowlatency") || S.Bool("antilag")) arguments.Add("-LowLatency");
		if (S.Bool("hdr")) arguments.Add("-Hdr");
		// The cheats file of this title, when it has one: the emulator applies the mods left enabled in it.
		if (S.Bool("cheats")) {
			var patch = (game ?? GameInfo.Read(S.Str("game"), context.PrecompileTool, context.Root)).CheatFile(context.Root);
			if (patch != null) arguments.AddRange(new[] { "-Patch", Quote(patch) });
		}
		var dlss = Dlss;
		// The upscaler's own render height (a chosen height is always passed: without one the game follows the window);
		// DLSS and FSR 3 share the PS5 Pro path's.
		var height = dlss || FsrUpscaler ? S.Int("render_dlss") : S.Int("render");
		if (height > 0) arguments.AddRange(new[] { "-RenderHeight", Number(height) });
		if (wide) arguments.AddRange(new[] { "-Aspect", ratio });
		var quality = S.Str("quality").ToLowerInvariant();
		if (quality == "low" || quality == "lowest") arguments.AddRange(new[] { "-Quality", quality, "-TextureReduce", quality == "lowest" ? "2" : "1" });
		var fsrOn = context.FsrAvailable && S.Bool("fsr");
		if (fsrOn) arguments.Add("-Fsr");
		var frameGen = S.Int("framegen");
		if (!fsrOn && context.FrameGenAvailable && frameGen >= 2) arguments.AddRange(new[] { "-FrameGen", Number(frameGen - 1) });
		if (dlss) arguments.AddRange(new[] { "-Upscaler", "dlss" });
		else if (FsrUpscaler) arguments.AddRange(new[] { "-Upscaler", "fsr" });
		// No anti-aliasing: Demon's Souls' TAA and its camera jitter off (the game's own upscaling stays).
		else if (S.Str("upscaler") == "off") arguments.AddRange(new[] { "-Antialiasing", "off" });
		// (With DLSS too: its PS5 Pro path then makes the 1080p output, the 3D at most 1080 high.)
		if (HdOutput) arguments.Add("-HdOutput");
		else if (Output1440) arguments.AddRange(new[] { "-OutputHeight", "1440" });
		if (context.FrameGenAvailable && ((!fsrOn && frameGen >= 2) || dlss)) arguments.AddRange(new[] { "-StreamlineDir", Quote(context.Streamline) });
		AddGpu(arguments);
		var sets = new List<string>();
		// The emulator's panel at the top right of the game image: the frame rate and frame times, or with the GPU,
		// processor and memory too (F8 in the game switches it).
		var hud = S.Int("hud");
		if (hud >= 1) sets.Add("KYTY_FPS_HUD=" + Number(Math.Min(hud, 2)));
		// Safe GPU mode: Demon's Souls' compute work one dispatch after another (no GPU hangs, ~37% slower).
		if (S.Bool("safegpu")) sets.Add("KYTY_COMPUTE_CHAIN=0");
		// Experiments: each frame presented as it is handed over (variable refresh rate displays, issue #37), and AMD Anti-Lag.
		if (S.Bool("present_direct")) sets.Add("KYTY_PRESENT_DIRECT=1");
		// Frames flip as soon as they are done, not on the 60 Hz vblanks (VRR displays, issue #221).
		if (S.Bool("vrr_flips")) sets.Add("KYTY_VRR=1");
		if (S.Bool("antilag")) sets.Add("KYTY_ANTI_LAG=1");
		// The game's 3D view at most every 2nd vblank: a steady 30 fps at the same game speed (movies and menus keep 60).
		if (S.Bool("fps30")) sets.Add("KYTY_FLIP_RATE=1");
		// Memory kept in RAM against the MEMORY_MANAGEMENT blue screen (issue #50): a 6 GB working-set floor, no trims of its own.
		if (S.Bool("keep_resident")) sets.AddRange(new[] { "KYTY_RESIDENT_MB=6144", "KYTY_WORKING_SET_TRIM=0" });
		// The audio output the game is told (it mixes for it; a TV when none is set, issue #207).
		var audioOutput = S.Str("audio_output");
		if (audioOutput == "headphones" || audioOutput == "amplifier" || audioOutput == "soundbar") sets.Add("KYTY_AUDIO_OUTPUT=" + audioOutput);
		// The DualSense speaker's own volume (0 when the switch above is off; the haptics stay), then what a hit feels
		// like: the motor drive as a percent of the mainline 2.0x, and the length of one drive report. Each is sent only
		// when moved off the emulator's own default, so an untouched launcher plays exactly as before.
		var padVolume = S.Bool("pad_speaker") ? Math.Max(0, Math.Min(S.Int("pad_speaker_volume"), 100)) : 0;
		if (padVolume != 100) sets.Add("KYTY_DUALSENSE_SPEAKER_VOLUME=" + Number(padVolume));
		var padStrength = Math.Max(0, Math.Min(S.Int("pad_vibration"), 400));
		if (padStrength != 200) sets.Add("KYTY_DUALSENSE_VIBRATION_MOTOR_GAIN=" + (padStrength / 100.0).ToString("0.##", CultureInfo.InvariantCulture));
		var padLength = Math.Max(5, Math.Min(S.Int("pad_vibration_ms"), 500));
		if (padLength != 60) {
			sets.Add("KYTY_DUALSENSE_VIBRATION_MOTOR_MS=" + Number(padLength));
			// The report's own length cannot lengthen the tail by itself: the hold falls to its floor first and
			// the fade stops there, so a longer report is never seen. The decay decides how long that takes, and
			// the slider scales both - 20 ms is a tap (0.35 per report), 60 ms the emulator's own 0.6, 500 ms a
			// long rumble (0.92).
			var padDecay = padLength <= 60
				? 0.30 + (0.60 - 0.30) * (padLength - 5) / 55.0
				: 0.60 + (0.92 - 0.60) * (padLength - 60) / 440.0;
			sets.Add("KYTY_DUALSENSE_VIBRATION_MOTOR_DECAY=" + padDecay.ToString("0.###", CultureInfo.InvariantCulture));
		}
		// The actuators (the sustained part of a hit - a spell's whole charge is mostly these). A percent of the
		// emulator's 4.0, sent only when moved.
		var padActuator = Math.Max(5, Math.Min(S.Int("pad_actuator"), 400));
		if (padActuator != 400) sets.Add("KYTY_DUALSENSE_HAPTICS_GAIN=" + (padActuator / 100.0).ToString("0.##", CultureInfo.InvariantCulture));
		// The DLSS model chosen on the Graphics page (a preset letter; none: the DLSS version's own choice).
		var preset = S.Str("dlss_preset");
		if (dlss && preset.Length == 1 && preset[0] >= 'A' && preset[0] <= 'O') sets.Add("KYTY_DLSS_PRESET=" + preset);
		// The sharpening chosen there: DLSS's (KYTY_DLSS_SHARPNESS, 0-1; none: not sharpened), FSR 3's (KYTY_FSR_SHARPNESS;
		// none: its 0.6).
		var sharpness = S.Str("dlss_sharpness");
		if (dlss && (sharpness == "0.3" || sharpness == "0.6" || sharpness == "0.9")) sets.Add("KYTY_DLSS_SHARPNESS=" + sharpness);
		var fsrSharpness = S.Str("fsr_sharpness");
		if (FsrUpscaler && (fsrSharpness == "0" || fsrSharpness == "0.3" || fsrSharpness == "0.9")) sets.Add("KYTY_FSR_SHARPNESS=" + fsrSharpness);
		// The Advanced page's extra options: their -Set pairs join (or replace) the launcher's own.
		var extra = ExtraOptions();
		foreach (var option in extra) {
			if (!IsSet(option.Key) || option.Value == null) continue;
			foreach (var pair in Regex.Split(option.Value, ",(?=[A-Za-z_][A-Za-z0-9_]*=)")) {
				var key = pair.Split('=')[0];
				sets.RemoveAll(set => string.Equals(set.Split('=')[0], key, StringComparison.OrdinalIgnoreCase));
				sets.Add(pair);
			}
		}
		if (sets.Count != 0) arguments.AddRange(new[] { "-Set", string.Join(",", sets) });
		var gamePath = S.Str("game");
		if (gamePath.Length != 0) arguments.AddRange(new[] { "-Game", Quote(gamePath) });
		// Their other options in place of the launcher's of the same name (run-windows.ps1 takes each once), else added.
		foreach (var option in extra) {
			if (IsSet(option.Key)) continue;
			var at = arguments.FindIndex(RunArgumentsStart, a => string.Equals(a, option.Key, StringComparison.OrdinalIgnoreCase));
			if (at >= 0) arguments.RemoveRange(at, at + 1 < arguments.Count && !arguments[at + 1].StartsWith("-") ? 2 : 1);
			arguments.Add(option.Key);
			if (option.Value != null) arguments.Add(option.Value.IndexOf(' ') >= 0 ? Quote(option.Value) : option.Value);
		}
		// The console stays for the live log; after a crash it waits for a key (not when direct).
		return PowerShell() + " " + string.Join(" ", arguments) + (direct ? "" : " & if !errorlevel! neq 0 pause");
	}

	// (Where run-windows.ps1's own arguments start: after -NoProfile -ExecutionPolicy Bypass -File <script>.)
	const int RunArgumentsStart = 5;
	// Characters cmd would take itself in the command line (/v:on: ! too), and quotes: an option holding one is left out.
	static readonly char[] CmdCharacters = { '&', '|', '<', '>', '^', '!', '%', '"' };

	// The graphics card chosen on the Advanced page (a PC with more than one), by its name: the game and the precompile
	// both use it (run-windows.ps1 / precompile-windows.ps1 -Gpu; issue #132).
	void AddGpu(List<string> arguments) {
		var gpu = S.Str("gpu");
		if (gpu.Length != 0 && gpu.IndexOfAny(CmdCharacters) < 0) arguments.AddRange(new[] { "-Gpu", Quote(gpu) });
	}

	static bool IsSet(string name) {
		return string.Equals(name, "-Set", StringComparison.OrdinalIgnoreCase);
	}

	// The Advanced page's extra options (settings extra_args), written as on a run.cmd line: each -Name with its value (the
	// next part when it does not start with -), "double quotes" keeping spaces in one. A value without a name is left out,
	// as is an option holding a character cmd would take itself.
	public List<KeyValuePair<string, string>> ExtraOptions() {
		var parts = new List<string>();
		var part = new StringBuilder();
		bool quoted = false, any = false;
		foreach (var c in S.Str("extra_args")) {
			if (c == '"') {
				quoted = !quoted;
				any = true;
			} else if (char.IsWhiteSpace(c) && !quoted) {
				if (any) parts.Add(part.ToString());
				part.Clear();
				any = false;
			} else {
				part.Append(c);
				any = true;
			}
		}
		if (any) parts.Add(part.ToString());
		var options = new List<KeyValuePair<string, string>>();
		for (int i = 0; i < parts.Count; i++) {
			var name = parts[i];
			if (name.Length < 2 || name[0] != '-') continue;
			var value = i + 1 < parts.Count && !parts[i + 1].StartsWith("-") ? parts[++i] : null;
			if (name.IndexOfAny(CmdCharacters) >= 0 || (value != null && value.IndexOfAny(CmdCharacters) >= 0)) continue;
			options.Add(new KeyValuePair<string, string>(name, value));
		}
		return options;
	}

	public string PlayArguments(GameInfo game = null, bool direct = false) {
		return "/v:on /s /c \"" + PlayCommand(game, direct) + "\"";
	}

	public string PrecompileCommand(string option) {
		var arguments = new List<string> { "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", Quote(Path.Combine(context.Root, "precompile-windows.ps1")) };
		if (!string.IsNullOrEmpty(option)) arguments.Add(option);
		if (S.Bool("ecores") && context.EfficiencyMask != 0) arguments.AddRange(new[] { "-Affinity", context.EfficiencyMask.ToString(CultureInfo.InvariantCulture) });
		AddGpu(arguments);
		var gamePath = S.Str("game");
		if (gamePath.Length != 0) arguments.AddRange(new[] { "-Game", Quote(gamePath) });
		return PowerShell() + " " + string.Join(" ", arguments) + " & pause";
	}

	public string PrecompileArguments(string option) {
		return "/s /c \"" + PrecompileCommand(option) + "\"";
	}

	// (--play: the console minimized, so that the game's window is the one in front.)
	public Process Start(string cmdArguments, bool minimized = false) {
		return Process.Start(new ProcessStartInfo(Cmd(), cmdArguments) { UseShellExecute = true, WorkingDirectory = context.Root,
			WindowStyle = minimized ? ProcessWindowStyle.Minimized : ProcessWindowStyle.Normal });
	}

	static string Quote(string text) {
		return "\"" + text + "\"";
	}

	static string Number(int value) {
		return value.ToString(CultureInfo.InvariantCulture);
	}
}

}
