using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Web.Script.Serialization;
using System.Windows;
using System.Windows.Controls;

namespace KytyLauncher {

// A GoldHEN-style mods file (_Patches\<title id>.json): every mod has a name, an "enabled" flag and the memory writes it
// makes. The emulator honours the flag (src/loader/gamePatch.cpp skips a mod whose flag is false), so choosing the mods
// only sets those flags (launcher.ps1's Show-CheatDialog and Save-CheatMods).
partial class CheatsWindow : Window {
	readonly List<CheckBox> boxes = new List<CheckBox>();

	CheatsWindow() {
		InitializeComponent();
	}

	static string ReadText(string file) {
		// (UTF-8: read as ANSI the quotes of its text can break.)
		return File.ReadAllText(file, Encoding.UTF8);
	}

	static Dictionary<string, object> Parse(string text) {
		return new JavaScriptSerializer { MaxJsonLength = 64 * 1024 * 1024 }.DeserializeObject(text) as Dictionary<string, object>;
	}

	static object[] Mods(Dictionary<string, object> data) {
		object mods;
		if (data == null || !data.TryGetValue("mods", out mods)) return new object[0];
		var list = mods as object[];
		if (list != null) return list;
		var array = mods as ArrayList;
		return array != null ? array.ToArray() : new object[0];
	}

	static bool Enabled(object mod) {
		var map = mod as Dictionary<string, object>;
		object value;
		return map == null || !map.TryGetValue("enabled", out value) || !(value is bool) || (bool)value;
	}

	static string Field(object mod, string key) {
		var map = mod as Dictionary<string, object>;
		object value;
		return map != null && map.TryGetValue(key, out value) && value != null ? Convert.ToString(value, CultureInfo.InvariantCulture) : "";
	}

	public static void Count(string file, out int on, out int total) {
		on = total = 0;
		try {
			var mods = Mods(Parse(ReadText(file)));
			total = mods.Length;
			on = mods.Count(Enabled);
		} catch (Exception) {
		}
	}

	public static void Edit(Window owner, string file) {
		Dictionary<string, object> data;
		string before;
		try {
			before = ReadText(file);
			data = Parse(before);
		} catch (Exception e) {
			MessageBox.Show(owner, Strings.F("cheats.unreadable", file, e.Message), Strings.T("cheats"), MessageBoxButton.OK, MessageBoxImage.Warning);
			return;
		}
		var mods = Mods(data);
		if (mods.Length == 0) {
			MessageBox.Show(owner, Strings.F("cheats.empty", file), Strings.T("cheats"), MessageBoxButton.OK, MessageBoxImage.Information);
			return;
		}
		var window = new CheatsWindow { Owner = owner };
		var name = Field(data, "name");
		window.Title = Strings.F("cheatsWindow", name.Length != 0 ? name : Field(data, "id"));
		window.All.Content = Strings.T("cheats.all");
		window.None.Content = Strings.T("cheats.none");
		window.Ok.Content = Strings.T("ok");
		window.Cancel.Content = Strings.T("cancel");
		foreach (var mod in mods) {
			var box = new CheckBox { Content = Field(mod, "name"), IsChecked = Enabled(mod), Margin = new Thickness(0, 5, 0, 5) };
			var description = Field(mod, "description");
			box.GotFocus += (s, e) => window.Description.Text = description;
			box.MouseEnter += (s, e) => window.Description.Text = description;
			window.boxes.Add(box);
			window.List.Children.Add(box);
		}
		window.All.Click += (s, e) => window.boxes.ForEach(b => b.IsChecked = true);
		window.None.Click += (s, e) => window.boxes.ForEach(b => b.IsChecked = false);
		window.Ok.Click += (s, e) => window.DialogResult = true;
		if (window.ShowDialog() != true) return;
		var wanted = window.boxes.Select(b => b.IsChecked == true).ToArray();
		var result = Save(file, before, wanted);
		if (result != null) MessageBox.Show(owner, Strings.T(result), Strings.T("cheats"), MessageBoxButton.OK, MessageBoxImage.Warning);
	}

	// The flags written; null, or what went wrong (the file left as it was, or put back from its backup).
	static string Save(string file, string before, bool[] wanted) {
		// Not over a file that changed while the window was open.
		if (ReadText(file) != before) return "cheats.changed";
		var data = Parse(before);
		var mods = Mods(data);
		if (mods.Length != wanted.Length) return "cheats.countChanged";
		for (int i = 0; i < mods.Length; i++) {
			var map = mods[i] as Dictionary<string, object>;
			if (map != null) map["enabled"] = wanted[i];
		}
		data["mods"] = mods;
		var text = new StringBuilder();
		Write(text, data, 0);
		text.Append("\r\n");
		File.Copy(file, file + ".bak", true);
		File.WriteAllText(file, text.ToString(), new UTF8Encoding(false));
		// Read back: the same names and flags, or the backup goes back.
		try {
			var check = Mods(Parse(ReadText(file)));
			if (check.Length != mods.Length) throw new InvalidDataException("count");
			for (int i = 0; i < check.Length; i++) {
				if (Field(check[i], "name") != Field(mods[i], "name") || Enabled(check[i]) != wanted[i]) throw new InvalidDataException("mod");
			}
		} catch (Exception) {
			File.Copy(file + ".bak", file, true);
			return "cheats.verifyFailed";
		}
		return null;
	}

	// JSON written back indented (the file stays readable for hand edits), its keys in their order.
	static void Write(StringBuilder text, object value, int depth) {
		var indent = new string(' ', 2 * (depth + 1));
		var closing = new string(' ', 2 * depth);
		var map = value as IDictionary<string, object>;
		if (map != null) {
			if (map.Count == 0) {
				text.Append("{}");
				return;
			}
			text.Append("{\r\n");
			var i = 0;
			foreach (var pair in map) {
				text.Append(indent).Append(Quote(pair.Key)).Append(": ");
				Write(text, pair.Value, depth + 1);
				text.Append(++i < map.Count ? ",\r\n" : "\r\n");
			}
			text.Append(closing).Append('}');
			return;
		}
		var list = value as IEnumerable;
		if (list != null && !(value is string)) {
			var items = list.Cast<object>().ToList();
			if (items.Count == 0) {
				text.Append("[]");
				return;
			}
			text.Append("[\r\n");
			for (int i = 0; i < items.Count; i++) {
				text.Append(indent);
				Write(text, items[i], depth + 1);
				text.Append(i + 1 < items.Count ? ",\r\n" : "\r\n");
			}
			text.Append(closing).Append(']');
			return;
		}
		if (value == null) text.Append("null");
		else if (value is bool) text.Append((bool)value ? "true" : "false");
		else if (value is string) text.Append(Quote((string)value));
		else if (value is decimal) text.Append(((decimal)value).ToString(CultureInfo.InvariantCulture));
		else if (value is double) text.Append(((double)value).ToString("R", CultureInfo.InvariantCulture));
		else text.Append(Convert.ToString(value, CultureInfo.InvariantCulture));
	}

	static string Quote(string text) {
		var quoted = new StringBuilder(text.Length + 2).Append('"');
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
		return quoted.Append('"').ToString();
	}
}

}
