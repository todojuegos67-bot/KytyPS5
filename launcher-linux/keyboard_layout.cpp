#include "keyboard_layout.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <set>

namespace {

const std::vector<std::string> ControlNames = {
    "Cross", "Circle", "Square", "Triangle", "L1", "L2", "L3", "R1", "R2", "R3", "Options",
    "Up", "Down", "Left", "Right", "TouchPad", "TouchPadRight",
    "LeftStickUp", "LeftStickDown", "LeftStickLeft", "LeftStickRight"};

const std::vector<std::string> DefaultInputs = {
    "E", "Space", "R", "F", "Left Shift", "Left Ctrl", "X", "Mouse:Left", "Mouse:Right",
    "Mouse:Middle", "Escape", "Up", "Down", "Left", "Right", "Tab", "Backspace",
    "W", "S", "A", "D"};

const std::vector<std::string> ActionNames = {
    "Interact / confirm", "Dodge / back", "Use item", "Two-hand weapon", "Guard / left attack",
    "Parry / left strong attack", "Left stick press", "Attack", "Strong attack", "Lock on", "Menu",
    "Switch spell", "Switch item", "Switch left weapon", "Switch right weapon",
    "Touchpad / left", "Touchpad right", "Move forward", "Move back", "Move left", "Move right"};

size_t IndexOf(const std::string& control) {
	const auto it = std::find(ControlNames.begin(), ControlNames.end(), control);
	return it == ControlNames.end() ? ControlNames.size() : static_cast<size_t>(it - ControlNames.begin());
}

bool AllDigits(const std::string& text) {
	return !text.empty() &&
	       std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// A sensitivity as the Windows launcher reads it: digits with at most one decimal point.
bool ParseSensitivity(const std::string& text, double* value) {
	int digits = 0, points = 0;
	for (const char c: text) {
		if (c >= '0' && c <= '9')
			++digits;
		else if (c == '.')
			++points;
		else
			return false;
	}
	if (digits == 0 || points > 1) return false;
	std::string normal = text;
	if (normal.front() == '.') normal.insert(normal.begin(), '0');
	if (normal.back() == '.') normal.push_back('0');
	double parsed = 0;
	const auto result = std::from_chars(normal.data(), normal.data() + normal.size(), parsed);
	if (result.ec != std::errc() || result.ptr != normal.data() + normal.size()) return false;
	*value = parsed;
	return true;
}

} // namespace

const std::vector<std::string>& KeyboardLayout::Controls() {
	return ControlNames;
}

const std::vector<std::string>& KeyboardLayout::Defaults() {
	return DefaultInputs;
}

const std::vector<std::string>& KeyboardLayout::Actions() {
	return ActionNames;
}

KeyboardLayout::KeyboardLayout(): bindings_(DefaultInputs) {}

const std::string& KeyboardLayout::Binding(const std::string& control) const {
	static const std::string none;
	const auto               at = IndexOf(control);
	return at < bindings_.size() ? bindings_[at] : none;
}

void KeyboardLayout::SetBinding(const std::string& control, const std::string& input) {
	const auto at = IndexOf(control);
	if (at < bindings_.size()) bindings_[at] = input;
}

bool KeyboardLayout::Supported(const std::string& input) {
	if (input.empty()) return false;
	if (input.size() == 1) {
		const char c = input[0];
		if (std::strchr("-=[]\\;'`,./", c) != nullptr) return true;
		if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) return true;
	}
	if (input[0] == 'F' && input.size() >= 2 && input.size() <= 3 && AllDigits(input.substr(1)) &&
	    input[1] != '0') {
		const int f = std::stoi(input.substr(1));
		return f >= 1 && f <= 24 && f != 7 && f != 8 && f != 11;
	}
	if (input.compare(0, 7, "Keypad ") == 0 && input.size() == 8 && input[7] >= '0' && input[7] <= '9')
		return true;
	static const std::set<std::string> names = {
	    "Space", "Return", "Escape", "Backspace", "Tab", "Left Shift", "Left Ctrl", "Left Alt",
	    "Keypad +", "Keypad -", "Keypad *", "Keypad /", "Keypad .", "NumLock", "ScrollLock",
	    "Up", "Down", "Left", "Right", "Home", "End", "PageUp", "PageDown", "Insert", "Delete",
	    "CapsLock", "Mouse:Left", "Mouse:Right", "Mouse:Middle", "Mouse:X1", "Mouse:X2",
	    "Mouse:WheelUp", "Mouse:WheelDown"};
	return names.count(input) != 0;
}

void KeyboardLayout::Validate() const {
	std::set<std::string> used;
	for (size_t i = 0; i < ControlNames.size(); ++i) {
		const auto& input = bindings_[i];
		if (!Supported(input))
			throw PromptError("Unsupported binding for " + ControlNames[i] +
			                  ". Edit keyboard controls to choose a supported key.");
		if (!used.insert(input).second)
			throw PromptError("Duplicate binding: " + input +
			                  ". Edit keyboard controls to give each action its own input.");
	}
	double value = 0;
	if (!sensitivity.empty() && (!ParseSensitivity(sensitivity, &value) || value < 0.1 || value > 5))
		throw PromptError("Mouse sensitivity must be blank or a number from 0.1 to 5.");
}

std::string KeyboardLayout::Assign(const std::string& control, const std::string& input) {
	const auto at = IndexOf(control);
	if (at >= bindings_.size() || !Supported(input))
		throw PromptError("Unsupported key. F7, F8, F11 and Pause are reserved for the emulator.");
	std::string other;
	for (size_t i = 0; i < bindings_.size(); ++i) {
		if (i != at && bindings_[i] == input) {
			other = ControlNames[i];
			// (The control that had the input gets the one this control had.)
			bindings_[i] = bindings_[at];
			break;
		}
	}
	bindings_[at] = input;
	return other;
}

std::vector<std::string> KeyboardLayout::Pairs() const {
	Validate();
	std::vector<std::string> pairs;
	for (size_t i = 0; i < ControlNames.size(); ++i)
		pairs.push_back(ControlNames[i] + "=" + bindings_[i]);
	if (!sensitivity.empty()) pairs.push_back("MouseSensitivity=" + sensitivity);
	return pairs;
}

std::string KeyboardLayout::Label(const std::string& input) {
	static const std::vector<std::pair<std::string, std::string>> special = {
	    {"Left Shift", "SHIFT"},        {"Left Ctrl", "CTRL"},
	    {"Left Alt", "ALT"},            {"Escape", "ESC"},
	    {"Return", "ENTER"},            {"Backspace", "BKSP"},
	    {"Mouse:Left", "LMB"},          {"Mouse:Right", "RMB"},
	    {"Mouse:Middle", "MMB"},        {"Mouse:X1", "M4"},
	    {"Mouse:X2", "M5"},             {"Mouse:WheelUp", "WHEEL\nUP"},
	    {"Mouse:WheelDown", "WHEEL\nDOWN"}, {"PageUp", "PGUP"},
	    {"PageDown", "PGDN"},           {"CapsLock", "CAPS"},
	    {"Up", "\xE2\x86\x91"},         {"Down", "\xE2\x86\x93"},
	    {"Left", "\xE2\x86\x90"},       {"Right", "\xE2\x86\x92"}};
	for (const auto& [name, label]: special)
		if (input == name) return label;
	std::string text = input;
	if (text.compare(0, 7, "Keypad ") == 0) text.replace(0, 7, "NUM ");
	for (auto& c: text)
		if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
	return text;
}

std::string KeyboardLayout::IconLabel(const std::string& icon) const {
	std::string name = icon;
	if (const auto slash = name.find_last_of('/'); slash != std::string::npos) name.erase(0, slash + 1);
	if (const auto dot = name.find_last_of('.'); dot != std::string::npos) name.erase(dot);
	static const std::vector<std::pair<std::string, std::string>> controls = {
	    {"ps4_button_cross", "Cross"},   {"ps4_button_circle", "Circle"},
	    {"ps4_button_square", "Square"}, {"ps4_button_triangle", "Triangle"},
	    {"ps4_l1", "L1"},                {"ps4_l2", "L2"},
	    {"ps4_l3", "L3"},                {"ps4_r1", "R1"},
	    {"ps4_r2", "R2"},                {"ps4_r3", "R3"},
	    {"ps4_options", "Options"},      {"ps4_dpad_u", "Up"},
	    {"ps4_dpad_d", "Down"},          {"ps4_dpad_l", "Left"},
	    {"ps4_dpad_r", "Right"},         {"ps4_touchpad", "TouchPad"},
	    {"ps4_touchpad_left", "TouchPad"}, {"ps4_touchpad_right", "TouchPadRight"}};
	for (const auto& [icon_name, control]: controls)
		if (name == icon_name) return Label(Binding(control));
	if (name == "ps4_stick_r") return "MOUSE";
	if (name == "ps4_stick_l")
		return Label(Binding("LeftStickUp")) + " " + Label(Binding("LeftStickLeft")) + "\n" +
		       Label(Binding("LeftStickDown")) + " " + Label(Binding("LeftStickRight"));
	if (name == "ps4_dpad_lr") return Label(Binding("Left")) + " " + Label(Binding("Right"));
	if (name == "ps4_dpad_ud") return Label(Binding("Up")) + " " + Label(Binding("Down"));
	throw PromptError("Unknown keyboard prompt: " + name);
}
