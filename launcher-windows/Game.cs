using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace KytyLauncher {

// A game: its folder (eboot.bin and sce_sys) or that folder packed into a ZArchive (.zar), and what its param.json says.
sealed class GameInfo {
	public string Path = "";
	public string TitleId = "";
	public string Version = "";
	public string Name = "";
	public string Picture;   // sce_sys\pic0.png or pic1.png of the dump (the game's own art, read from the user's files)
	public string Icon;      // sce_sys\icon0.png
	public bool Found { get { return TitleId.Length != 0; } }
	public string GameId { get { return TitleId + "_" + Version; } }

	public static GameInfo Read(string path, string tool, string root) {
		var info = new GameInfo { Path = path ?? "" };
		if (info.Path.Length == 0) return info;
		Dictionary<string, object> param = null;
		try {
			var file = System.IO.Path.Combine(info.Path, "sce_sys", "param.json");
			if (Directory.Exists(info.Path) && File.Exists(file)) {
				param = Parse(File.ReadAllText(file, Encoding.UTF8));
				foreach (var name in new[] { "pic0.png", "pic1.png" }) {
					var picture = System.IO.Path.Combine(info.Path, "sce_sys", name);
					if (File.Exists(picture)) {
						info.Picture = picture;
						break;
					}
				}
				var icon = System.IO.Path.Combine(info.Path, "sce_sys", "icon0.png");
				if (File.Exists(icon)) info.Icon = icon;
			} else if (File.Exists(info.Path) && info.Path.EndsWith(".zar", StringComparison.OrdinalIgnoreCase) && tool != null) {
				// The precompile program copies param.json out of the archive (launcher.ps1's Read-GameParam).
				var copy = System.IO.Path.GetTempFileName();
				try {
					if (RunHidden(tool, "--game \"" + info.Path + "\" --param \"" + copy + "\"", root, 15000) == 0)
						param = Parse(File.ReadAllText(copy, Encoding.UTF8));
				} finally {
					try { File.Delete(copy); } catch (Exception) { }
				}
			}
		} catch (Exception) {
		}
		if (param == null) return info;
		info.TitleId = Text(param, "titleId");
		info.Version = Text(param, "contentVersion");
		var localized = param.ContainsKey("localizedParameters") ? param["localizedParameters"] as Dictionary<string, object> : null;
		if (localized != null) {
			var language = Text(localized, "defaultLanguage");
			var names = language.Length != 0 && localized.ContainsKey(language) ? localized[language] as Dictionary<string, object> : null;
			if (names != null) info.Name = Text(names, "titleName");
		}
		return info;
	}

	static Dictionary<string, object> Parse(string json) {
		return new JavaScriptSerializer { MaxJsonLength = 64 * 1024 * 1024 }.DeserializeObject(json) as Dictionary<string, object>;
	}

	static string Text(Dictionary<string, object> map, string key) {
		object value;
		return map.TryGetValue(key, out value) && value != null ? Convert.ToString(value) : "";
	}

	// (In the emulator's folder, as run-windows.ps1 runs it: the program finds _PipelineCache there.)
	public static int RunHidden(string file, string arguments, string workingDirectory, int timeoutMs, StringBuilder output = null) {
		var start = new ProcessStartInfo(file, arguments) {
			UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = output != null, RedirectStandardError = false,
			WorkingDirectory = workingDirectory
		};
		using (var process = Process.Start(start)) {
			if (process == null) return -1;
			if (output != null) {
				var read = process.StandardOutput.ReadToEndAsync();
				if (!process.WaitForExit(timeoutMs)) {
					try { process.Kill(); } catch (Exception) { }
					return -1;
				}
				output.Append(read.Result);
			} else if (!process.WaitForExit(timeoutMs)) {
				try { process.Kill(); } catch (Exception) { }
				return -1;
			}
			return process.ExitCode;
		}
	}

	// _Patches\<title id>.json, the GoldHEN-style cheat file of this title, or null.
	public string CheatFile(string root) {
		if (!Found) return null;
		var file = System.IO.Path.Combine(root, "_Patches", TitleId + ".json");
		return File.Exists(file) ? file : null;
	}
}

enum StoreState { Unknown, Checking, None, Current, Outdated, OutdatedAll }

// The precompiled shaders of a game version (_PipelineCache\static\<title>_<version>.*): what run-windows.ps1 asks the
// precompile program (--status) before it offers the precompile.
static class ShaderStore {
	public static Task<StoreState> CheckAsync(LauncherContext context, GameInfo game) {
		return Task.Run(() => Check(context, game));
	}

	static StoreState Check(LauncherContext context, GameInfo game) {
		if (!game.Found) return StoreState.Unknown;
		var store = Path.Combine(context.Root, "_PipelineCache", "static");
		// (.blobs: the store of a driver without pipeline binaries, each pipeline's own cache data; issue #290.)
		var made = new[] { ".binaries", ".bin", ".blobs" }.Any(ext => File.Exists(Path.Combine(store, game.GameId + ext)));
		if (!made) return StoreState.None;
		var seedName = "seeds-" + game.GameId + ".seeds";
		var seeds = new[] { Path.Combine(context.Root, seedName), Path.Combine(context.Root, "_Build", "static-precompile", seedName) }
			.FirstOrDefault(File.Exists);
		if (seeds == null || context.PrecompileTool == null) return StoreState.Current;
		try {
			var output = new StringBuilder();
			var code = GameInfo.RunHidden(context.PrecompileTool, "--game \"" + game.Path + "\" --seeds \"" + seeds + "\" --status", context.Root, 60000, output);
			if (code != 0) return StoreState.Unknown;
			var text = output.ToString();
			// (OutdatedAll: the store lacks a translation that changes nearly every shader, so the precompile redoes it all,
			// issue #246.)
			if (text.Contains("static cache outdated")) return text.Contains("next precompile all") ? StoreState.OutdatedAll : StoreState.Outdated;
			if (text.Contains("static cache current")) {
				// (A precompile that stopped before its last merge leaves large shard checkpoints. One older than its store is
				// an earlier run's - another graphics driver's, which no merge takes, issue #126 - not an unfinished one.)
				var unfinished = Directory.Exists(store) && new DirectoryInfo(store).GetFiles("*.shard*").Any(f => {
					var at = f.Name.IndexOf(".shard", StringComparison.Ordinal);
					var own = new FileInfo(Path.Combine(f.DirectoryName ?? store, f.Name.Substring(0, at)));
					return f.Length > (1L << 20) && !(own.Exists && own.LastWriteTimeUtc > f.LastWriteTimeUtc);
				});
				return unfinished ? StoreState.Outdated : StoreState.Current;
			}
			return StoreState.None;
		} catch (Exception) {
			return StoreState.Unknown;
		}
	}
}

}
