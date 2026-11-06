#include "keyboard_layout.h"
#include "settings.h"

// The layout in launcher-settings.json (its own file: keyboard_layout.cpp needs nothing of the
// settings, so it can be tested alone).

KeyboardLayout KeyboardLayout::Read(const Settings& settings, bool validate) {
	KeyboardLayout result;
	for (const auto& control: Controls()) {
		const auto input = settings.Str("keyboard_binding_" + control);
		if (!input.empty()) result.SetBinding(control, input);
	}
	result.sensitivity = settings.Str("keyboard_mouse_sensitivity");
	if (validate) result.Validate();
	return result;
}

void KeyboardLayout::Save(Settings& settings) const {
	Validate();
	for (const auto& control: Controls())
		settings.Set("keyboard_binding_" + control, Binding(control));
	settings.Set("keyboard_mouse_sensitivity", sensitivity);
}
