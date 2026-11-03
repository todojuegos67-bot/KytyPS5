using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows;

namespace KytyLauncher {

// Command line (a double-click passes none):
//   --play                 the game started with the saved settings, without this window (Steam's launch options, a
//                          shortcut); this process lasts until the game's console closes. No game folder: the window.
//   --refocus <window> <steam> <pid>   (--play starts it, from Steam) the focus away from Steam's window and back to it
//                          once process <pid> has exited, as Alt+Tab twice would (issue #217)
// For tests:
//   --dry-run <file>       the play, precompile and hints commands the current settings make, written to <file>
//   --screenshot <folder>  each page rendered to <folder>\<page>.png, then exit
//   --lang en|zh           the window's language for this run (the saved choice otherwise)
//   --no-art               (with --screenshot) no game art in the banner
//   --demo-path <text>     (with --screenshot) the game path box shows <text> (pictures for the release notes)
public partial class App : Application {
	protected override void OnStartup(StartupEventArgs e) {
		base.OnStartup(e);
		string dryRun = null, screenshots = null, lang = null;
		for (int i = 0; i + 1 < e.Args.Length; i++) {
			if (e.Args[i] == "--dry-run") dryRun = e.Args[++i];
			else if (e.Args[i] == "--screenshot") screenshots = e.Args[++i];
			else if (e.Args[i] == "--lang") lang = e.Args[++i];
			else if (e.Args[i] == "--demo-path") KytyLauncher.MainWindow.ScreenshotPath = e.Args[++i];
		}
		KytyLauncher.MainWindow.ScreenshotNoArt = Array.IndexOf(e.Args, "--no-art") >= 0;
		var context = new LauncherContext();
		var refocus = Array.IndexOf(e.Args, "--refocus");
		context.Settings.ReadOnly = dryRun != null || screenshots != null || refocus >= 0;
		if (refocus >= 0 && refocus + 3 < e.Args.Length) {
			Refocus(e.Args[refocus + 1], e.Args[refocus + 2], e.Args[refocus + 3], context);
			return;
		}
		if (lang != null) Strings.Language = lang == "zh" ? UiLanguage.Chinese : UiLanguage.English;
		if (dryRun != null) {
			var text = new StringBuilder();
			var commands = new Commands(context);
			text.AppendLine("play:       cmd " + commands.PlayArguments());
			text.AppendLine("precompile: cmd " + commands.PrecompileArguments(""));
			text.AppendLine("hints:      cmd " + commands.PrecompileArguments("-ExportHints"));
			text.AppendLine(string.Format("efficiency cores: 0x{0:X}", context.EfficiencyMask));
			text.AppendLine(string.Format("display: {0} lines", context.DisplayHeight));
			text.AppendLine("frame generation: " + (context.FrameGenAvailable ? "NVIDIA's Streamline runtime in " + context.Streamline
				: !context.RtxGpu ? "no NVIDIA RTX graphics card (DLSS off)" : "no NVIDIA-signed Streamline runtime (settings: '" + context.Settings.Str("streamline") + "')"));
			File.WriteAllText(dryRun, text.ToString(), new UTF8Encoding(false));
			Shutdown(0);
			return;
		}
		if (Array.IndexOf(e.Args, "--play") >= 0) {
			var game = GameInfo.Read(context.Settings.Str("game"), context.PrecompileTool, context.Root);
			if (game.Found) {
				var commands = new Commands(context);
				// The window that had the focus as this started (Steam's Big Picture when Steam starts it): the emulator
				// hands the focus back to it as the game quits (issue #217: Windows gave it to another window, and Big
				// Picture took no pad input until clicked).
				var focus = Native.ForegroundWindow();
				var parent = Native.ParentProcess();
				Environment.SetEnvironmentVariable("KYTY_RETURN_FOCUS", focus);
				Environment.SetEnvironmentVariable("KYTY_RETURN_FOCUS_PID", parent);
				try {
					var process = commands.StartPlay(true);
					// Steam counts the game as running while this process lasts (its controller layout, play time).
					new Thread(() => {
						process.WaitForExit();
						// (Again, where the game ended without handing it back.)
						Native.Focus(focus, parent);
						// Big Picture leaves the pad alone while Steam counts the game as running, and the focus came back to
						// it before that ended: Play stayed grey until a click or Alt+Tab (issue #217). A helper out of
						// Steam's job takes the focus away and gives it back once this process is gone.
						if (Native.ProcessName(parent) == "steam") {
							var how = Native.StartOutsideJob(System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName,
								"--refocus " + focus + " " + parent + " " +
								System.Diagnostics.Process.GetCurrentProcess().Id.ToString(System.Globalization.CultureInfo.InvariantCulture));
							AppendToGameLog(context, "Focus (launcher): the helper for Big Picture " + (how ?? "could not be started"));
						}
						var code = process.ExitCode;
						Dispatcher.BeginInvoke(new Action(() => Shutdown(code)));
					}) { IsBackground = true }.Start();
				} catch (Exception error) {
					MessageBox.Show(Strings.F("error.start", error.Message), Strings.T("error.title"), MessageBoxButton.OK, MessageBoxImage.Error);
					Shutdown(1);
				}
				return;
			}
		}
		var window = new MainWindow(context);
		base.MainWindow = window;
		if (screenshots != null) {
			window.SaveScreenshots(screenshots);
			new KeyboardControlsWindow(context.Settings).SaveScreenshot(Path.Combine(screenshots, "8-keyboard-controls.png"));
			Shutdown(0);
			return;
		}
		window.Show();
	}

	// --refocus: once the --play process is gone (Steam counts the game as ended about then), 2.5 s for Big Picture to see
	// it, then the focus to a window of this one and back to Steam's: the fresh activation a click or Alt+Tab gives it.
	// The game's log gets a line on what happened.
	void Refocus(string focus, string steam, string pid, LauncherContext context) {
		ShutdownMode = ShutdownMode.OnExplicitShutdown;
		new Thread(() => {
			int id;
			if (int.TryParse(pid, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out id)) {
				try {
					using (var launcher = System.Diagnostics.Process.GetProcessById(id)) launcher.WaitForExit(120000);
				} catch (ArgumentException) {
					// (Gone already.)
				}
			}
			Thread.Sleep(2500);
			Window away = null;
			Dispatcher.Invoke(new Action(() => {
				away = new Window { Width = 1, Height = 1, Left = 0, Top = 0, WindowStyle = WindowStyle.None, ResizeMode = ResizeMode.NoResize,
				                    ShowInTaskbar = false, AllowsTransparency = true, Background = System.Windows.Media.Brushes.Transparent,
				                    Topmost = true };
				away.Show();
				Native.TakeForeground(new System.Windows.Interop.WindowInteropHelper(away).Handle);
			}));
			Thread.Sleep(200);
			var between = Native.ForegroundDescription();
			// (Closed first: closed after the hand-back, Windows gave the focus on to another window 2 ms later.)
			Dispatcher.Invoke(new Action(() => away.Close()));
			Thread.Sleep(100);
			Dispatcher.Invoke(new Action(() => Native.Focus(focus, steam)));
			Thread.Sleep(300);
			AppendToGameLog(context, "Focus (launcher's helper, after the game's end): away to " + between + ", then back; 0.3 s later in front: " +
			                         Native.ForegroundDescription());
			Dispatcher.BeginInvoke(new Action(() => Shutdown(0)));
		}) { IsBackground = true }.Start();
	}

	// A line at the end of the newest game log (run-windows.ps1's folder for them), for reports.
	static void AppendToGameLog(LauncherContext context, string line) {
		try {
			var folder = Directory.Exists(Path.Combine(context.Root, "_Build")) ? Path.Combine(context.Root, "_Build", "run-logs")
			                                                                     : Path.Combine(context.Root, "logs");
			var newest = new DirectoryInfo(folder).GetFiles("*.out.log").OrderByDescending(file => file.LastWriteTimeUtc).FirstOrDefault();
			if (newest != null) File.AppendAllText(newest.FullName, line + Environment.NewLine, new UTF8Encoding(false));
		} catch (Exception) {
			// (A log is a help, not a need.)
		}
	}
}

}
