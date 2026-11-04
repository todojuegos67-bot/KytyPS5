// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Web.Script.Serialization;
namespace KytyLauncher {
sealed class KeyboardLayout {
	public static readonly string[] Controls = {
	"Cross", "Circle", "Square", "Triangle", "L1", "L2", "L3", "R1", "R2", "R3", "Options",
	"Up", "Down", "Left", "Right", "TouchPad", "TouchPadRight", "LeftStickUp", "LeftStickDown", "LeftStickLeft", "LeftStickRight"
};
	public static readonly string[] Defaults = {
	"E", "Space", "R", "F", "Left Shift", "Left Ctrl", "X", "Mouse:Left", "Mouse:Right", "Mouse:Middle", "Escape",
	"Up", "Down", "Left", "Right", "Tab", "Backspace", "W", "S", "A", "D"
};
	public static readonly string[] Actions = {
	"Interact / confirm", "Dodge / back", "Use item", "Two-hand weapon", "Guard / left attack", "Parry / left strong attack", "Left stick press",
	"Attack", "Strong attack", "Lock on", "Menu", "Switch spell", "Switch item", "Switch left weapon", "Switch right weapon",
	"Touchpad / left", "Touchpad right", "Move forward", "Move back", "Move left", "Move right"
};
	public readonly Dictionary<string, string> Bindings = new Dictionary<string, string>();
	// Blank means preserve launch.json's sensitivity (or the emulator default).
	public string Sensitivity = "";
	public KeyboardLayout() { for (int i = 0; i < Controls.Length; i++) Bindings[Controls[i]] = Defaults[i]; }
	public static KeyboardLayout Read(Settings settings, bool validate = true) {
		var result = new KeyboardLayout();
		foreach (var control in Controls) {
			var input = settings.Str("keyboard_binding_" + control);
			if (input.Length != 0)
				result.Bindings[control] = input;
		}
		result.Sensitivity = settings.Str("keyboard_mouse_sensitivity");
		if (validate)
			result.Validate();
		return result;
	}
	public void Save(Settings settings) {
		Validate();
		foreach (var control in Controls)
			settings.Set("keyboard_binding_" + control, Bindings[control]);
		settings.Set("keyboard_mouse_sensitivity", Sensitivity);
	}
	public static bool Supported(string input) {
		if (input == null)
			return false;
		if (input.Length == 1 && "-=[]\\;'`,./".Contains(input))
			return true;
		if (input.Length == 1 && ((input[0] >= 'A' && input[0] <= 'Z') || (input[0] >= '0' && input[0] <= '9')))
			return true;
		int f;
		if (input.StartsWith("F") && int.TryParse(input.Substring(1), out f) && f >= 1 && f <= 24)
			return f != 7 && f != 8 && f != 11;
		if (input.StartsWith("Keypad ") && input.Length == 8 && char.IsDigit(input[7]))
			return true;
		return new[]{"Space","Return","Escape","Backspace","Tab","Left Shift","Left Ctrl","Left Alt",
		"Keypad +","Keypad -","Keypad *","Keypad /","Keypad .","NumLock","ScrollLock",
		"Up","Down","Left","Right","Home","End","PageUp","PageDown","Insert","Delete","CapsLock",
		"Mouse:Left","Mouse:Right","Mouse:Middle","Mouse:X1","Mouse:X2","Mouse:WheelUp","Mouse:WheelDown"}.Contains(input);
	}
	public void Validate() {
		var used = new HashSet<string>();
		foreach (var control in Controls) {
			string input;
			if (!Bindings.TryGetValue(control, out input) || !Supported(input))
				throw new IOException("Unsupported binding for " + control + ". Edit keyboard controls to choose a supported key.");
			if (!used.Add(input))
				throw new IOException("Duplicate binding: " + input + ". Edit keyboard controls to give each action its own input.");
		}
		double value;
		if (Sensitivity.Length != 0 && (!double.TryParse(Sensitivity, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out value) || value < 0.1 || value > 5))
			throw new IOException("Mouse sensitivity must be blank or a number from 0.1 to 5.");
	}
	public string Assign(string control, string input) {
		if (!Bindings.ContainsKey(control) || !Supported(input))
			throw new IOException("Unsupported key. F7, F8, F11 and Pause are reserved for the emulator.");
		var other = Bindings.FirstOrDefault(p => p.Key != control && p.Value == input).Key;
		if (other != null)
			Bindings[other] = Bindings[control];
		Bindings[control] = input;
		return other;
	}
	public string Encoded() {
		Validate();
		var pairs = Controls.Select(c => c + "=" + Bindings[c]).ToList();
		if (Sensitivity.Length != 0)
			pairs.Add("MouseSensitivity=" + Sensitivity);
		return Convert.ToBase64String(Encoding.UTF8.GetBytes(new JavaScriptSerializer().Serialize(pairs)));
	}
	public static string Label(string input) {
		switch (input) {
			case "Left Shift":
				return "SHIFT";
			case "Left Ctrl":
				return "CTRL";
			case "Left Alt":
				return "ALT";
			case "Escape":
				return "ESC";
			case "Return":
				return "ENTER";
			case "Backspace":
				return "BKSP";
			case "Mouse:Left":
				return "LMB";
			case "Mouse:Right":
				return "RMB";
			case "Mouse:Middle":
				return "MMB";
			case "Mouse:X1":
				return "M4";
			case "Mouse:X2":
				return "M5";
			case "Mouse:WheelUp":
				return "WHEEL\nUP";
			case "Mouse:WheelDown":
				return "WHEEL\nDOWN";
			case "PageUp":
				return "PGUP";
			case "PageDown":
				return "PGDN";
			case "CapsLock":
				return "CAPS";
			case "Up":
				return "\u2191";
			case "Down":
				return "\u2193";
			case "Left":
				return "\u2190";
			case "Right":
				return "\u2192";
		}
		return input.Replace("Keypad ", "NUM ").ToUpperInvariant();
	}
	public string IconLabel(string name) {
		var control = new Dictionary<string, string> {
		{"ps4_button_cross","Cross"},{"ps4_button_circle","Circle"},{"ps4_button_square","Square"},{"ps4_button_triangle","Triangle"},
		{"ps4_l1","L1"},{"ps4_l2","L2"},{"ps4_l3","L3"},{"ps4_r1","R1"},{"ps4_r2","R2"},{"ps4_r3","R3"},
		{"ps4_options","Options"},{"ps4_dpad_u","Up"},{"ps4_dpad_d","Down"},{"ps4_dpad_l","Left"},{"ps4_dpad_r","Right"},
		{"ps4_touchpad","TouchPad"},{"ps4_touchpad_left","TouchPad"},{"ps4_touchpad_right","TouchPadRight"}
	};
		name = Path.GetFileNameWithoutExtension(name);
		if (control.ContainsKey(name))
			return Label(Bindings[control[name]]);
		if (name == "ps4_stick_r")
			return "MOUSE";
		if (name == "ps4_stick_l")
			return Label(Bindings["LeftStickUp"]) + " " + Label(Bindings["LeftStickLeft"]) + "\n" + Label(Bindings["LeftStickDown"]) + " " + Label(Bindings["LeftStickRight"]);
		if (name == "ps4_dpad_lr")
			return Label(Bindings["Left"]) + " " + Label(Bindings["Right"]);
		if (name == "ps4_dpad_ud")
			return Label(Bindings["Up"]) + " " + Label(Bindings["Down"]);
		throw new IOException("Unknown keyboard prompt: " + name);
	}
}
}
