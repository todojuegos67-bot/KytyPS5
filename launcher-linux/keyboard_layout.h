#ifndef KYTY_LAUNCHER_LINUX_KEYBOARD_LAYOUT_H
#define KYTY_LAUNCHER_LINUX_KEYBOARD_LAYOUT_H

#include <stdexcept>
#include <string>
#include <vector>

class Settings;

// Why the keyboard prompts or their layout cannot be used (launcher-windows throws IOException):
// the text is shown to the person as it is.
class PromptError: public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};

// The keyboard and mouse layout of the Game page (launcher-windows' KeyboardLayout.cs): the key
// or mouse input of each of the 21 pad controls that the prompt icons show, and the mouse
// sensitivity. It is saved in launcher-settings.json (keyboard_binding_<control>,
// keyboard_mouse_sensitivity: the Windows launcher's keys) and a Play passes it as --keymap pairs.
class KeyboardLayout {
public:
	static constexpr size_t ControlCount = 21;

	// "Cross", "Circle", ... the emulator's control names, in the order the editor lists them.
	static const std::vector<std::string>& Controls();
	static const std::vector<std::string>& Defaults();
	// What each control does in Demon's Souls with the default layout (the editor's labels).
	static const std::vector<std::string>& Actions();

	KeyboardLayout();

	// The saved layout (the defaults for what is not saved); checked unless validate is false.
	static KeyboardLayout Read(const Settings& settings, bool validate = true);
	// Checked, then written to the settings (the caller saves the file).
	void Save(Settings& settings) const;

	// The inputs the emulator accepts here: a key by its SDL name, or Mouse:<button or wheel>. F7,
	// F8 and F11 (and Pause) are the emulator's own.
	static bool Supported(const std::string& input);
	// Every control has a supported input, none twice; the sensitivity is blank or 0.1 to 5.
	void Validate() const;

	// Gives `control` the input; the control that had it gets the one `control` had (returned:
	// its name, or empty).
	std::string Assign(const std::string& control, const std::string& input);

	const std::string& Binding(const std::string& control) const;
	void               SetBinding(const std::string& control, const std::string& input);

	// "Cross=E", ... and "MouseSensitivity=2.4" when set: the --keymap values of a Play.
	[[nodiscard]] std::vector<std::string> Pairs() const;

	// The text on a key cap: "SHIFT", "ESC", "LMB", "\xE2\x86\x91" (an arrow)...; "\n" between lines.
	static std::string Label(const std::string& input);
	// The cap of a prompt icon ("ps4_button_cross" or "ps4_button_cross.ctxr"): what its control
	// is bound to (the left stick's four keys, the D-pad's pairs, the right stick: MOUSE).
	[[nodiscard]] std::string IconLabel(const std::string& icon) const;

	// Blank: the sensitivity of launch.json or the emulator's default.
	std::string sensitivity;

private:
	std::vector<std::string> bindings_;
};

#endif
