using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;

namespace KytyLauncher {

// launcher-settings.json next to the scripts: the same file and keys as launcher.ps1 wrote (run-windows.ps1 reads its
// "streamline" too), so an update keeps the choices. Keys this launcher does not know are kept as they were.
sealed class Settings {
	// The defaults launcher.ps1 had (its order is the file's order); ui_language is this launcher's own.
	static readonly object[][] Defaults = {
		new object[] {"game", ""}, new object[] {"resolution", "2560x1440"}, new object[] {"fullscreen", false},
		new object[] {"aspect", true}, new object[] {"language", 1}, new object[] {"redzone", true},
		new object[] {"cheats", false}, new object[] {"ecores", false}, new object[] {"fps120", false},
		new object[] {"render", 1440}, new object[] {"ratio", "16:9"}, new object[] {"quality", "standard"},
		new object[] {"vsync", false}, new object[] {"hdr", false}, new object[] {"framegen", 0}, new object[] {"hud", 0},
		new object[] {"streamline", ""}, new object[] {"safegpu", false}, new object[] {"upscaler", ""},
		new object[] {"render_dlss", 1080}, new object[] {"fsr", false}, new object[] {"output", "auto"},
		new object[] {"lowlatency", false}, new object[] {"ui_language", ""},
		// (Game page: the DualSense's own speaker.)
		new object[] {"pad_speaker", true}, new object[] {"pad_speaker_volume", 100},
		new object[] {"pad_vibration", 200}, new object[] {"pad_vibration_ms", 60},
		new object[] {"pad_actuator", 400},
		// (Experimental page.)
		new object[] {"present_direct", false}, new object[] {"vrr_flips", false}, new object[] {"antilag", false},
		new object[] {"keep_resident", false},
		new object[] {"fps30", false},
		// (Advanced page.)
		new object[] {"extra_args", ""}, new object[] {"gpu", ""}, new object[] {"reopen_launcher", false},
		// (Graphics page: the DLSS model, a preset letter or the DLSS version's own choice; the file has seen the game output's
		// Automatic.)
		new object[] {"dlss_preset", ""}, new object[] {"output_auto", true},
		// (Graphics page: the upscaler's sharpening, KYTY_DLSS_SHARPNESS's value ("" none) and KYTY_FSR_SHARPNESS's ("" its
		// 0.6).)
		new object[] {"dlss_sharpness", ""}, new object[] {"fsr_sharpness", ""},
		// (Game page: the audio output the game is told, "" a TV.)
		new object[] {"audio_output", ""},
		// (Display page: with 21:9 or 32:9 the movies, the title and the loading screens at 16:9 between black bars.)
		new object[] {"wide_bars", false},
	};

	readonly string path;
	readonly List<string> order = new List<string>();
	readonly Dictionary<string, object> values = new Dictionary<string, object>();

	public Settings(string path) {
		this.path = path;
		foreach (var pair in Defaults) Put((string)pair[0], pair[1]);
		try {
			if (File.Exists(path)) {
				// (UTF-8, with or without a byte order mark.)
				var parsed = new JavaScriptSerializer().DeserializeObject(File.ReadAllText(path, Encoding.UTF8)) as Dictionary<string, object>;
				if (parsed != null) {
					foreach (var pair in parsed) Put(pair.Key, pair.Value);
					// 4K was the game output's default, written by every save: a file from before Automatic has it as
					// Automatic once (which keeps 4K above 1080 lines); a 4K chosen since stays.
					if (!parsed.ContainsKey("output_auto") && Str("output") == "4k") Put("output", "auto");
				}
			}
		} catch (Exception) {
			// An unreadable file: the defaults (it is written again on the next save).
		}
		// (launcher.ps1's predecessor kept the game folder in game-path.txt.)
		var legacy = Path.Combine(Path.GetDirectoryName(path) ?? ".", "game-path.txt");
		if (Str("game").Length == 0 && File.Exists(legacy)) {
			try {
				using (var reader = new StreamReader(legacy)) Put("game", (reader.ReadLine() ?? "").Trim());
			} catch (Exception) {
			}
		}
	}

	void Put(string key, object value) {
		if (!values.ContainsKey(key)) order.Add(key);
		values[key] = value;
	}

	public string Str(string key) {
		object value;
		return values.TryGetValue(key, out value) && value != null ? Convert.ToString(value, CultureInfo.InvariantCulture) : "";
	}

	public bool Bool(string key) {
		object value;
		if (!values.TryGetValue(key, out value) || value == null) return false;
		if (value is bool) return (bool)value;
		if (value is string) return string.Equals((string)value, "true", StringComparison.OrdinalIgnoreCase);
		try {
			return Convert.ToDouble(value, CultureInfo.InvariantCulture) != 0;
		} catch (Exception) {
			return false;
		}
	}

	// (An older launcher saved true for the overlay's frame rate: true is 1.)
	public int Int(string key) {
		object value;
		if (!values.TryGetValue(key, out value) || value == null) return 0;
		if (value is bool) return (bool)value ? 1 : 0;
		int number;
		if (value is string) return int.TryParse((string)value, NumberStyles.Integer, CultureInfo.InvariantCulture, out number) ? number : 0;
		try {
			return Convert.ToInt32(value, CultureInfo.InvariantCulture);
		} catch (Exception) {
			return 0;
		}
	}

	public void Set(string key, object value) {
		Put(key, value);
	}

	// (Tests: --screenshot leaves the file as it is.)
	public bool ReadOnly;

	// With a byte order mark: Windows PowerShell 5.1 (run-windows.ps1) reads a file without one as the ANSI code page,
	// and a game folder or Streamline path with other letters than English ones would come out wrong.
	public void Save() {
		if (ReadOnly) return;
		var text = new StringBuilder();
		text.Append("{\r\n");
		for (int i = 0; i < order.Count; i++) {
			text.Append("    ").Append(Quote(order[i])).Append(": ").Append(Value(values[order[i]]));
			text.Append(i + 1 < order.Count ? ",\r\n" : "\r\n");
		}
		text.Append("}\r\n");
		try {
			File.WriteAllText(path, text.ToString(), new UTF8Encoding(true));
		} catch (Exception) {
			// A read-only folder: the choices last until the window closes.
		}
	}

	static string Value(object value) {
		if (value == null) return "null";
		if (value is bool) return (bool)value ? "true" : "false";
		if (value is string) return Quote((string)value);
		if (value is int || value is long || value is short || value is byte) return Convert.ToString(value, CultureInfo.InvariantCulture);
		if (value is double || value is float || value is decimal)
			return Convert.ToDouble(value, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture);
		// Lists and objects of a newer launcher: as the serializer writes them.
		return new JavaScriptSerializer().Serialize(value);
	}

	static string Quote(string text) {
		var quoted = new StringBuilder(text.Length + 2);
		quoted.Append('"');
		foreach (var c in text) {
			switch (c) {
				case '"': quoted.Append("\\\""); break;
				case '\\': quoted.Append("\\\\"); break;
				case '\n': quoted.Append("\\n"); break;
				case '\r': quoted.Append("\\r"); break;
				case '\t': quoted.Append("\\t"); break;
				default:
					if (c < 0x20) quoted.AppendFormat(CultureInfo.InvariantCulture, "\\u{0:x4}", (int)c);
					else quoted.Append(c);
					break;
			}
		}
		quoted.Append('"');
		return quoted.ToString();
	}
}

}
