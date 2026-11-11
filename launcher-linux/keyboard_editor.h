#ifndef KYTY_LAUNCHER_LINUX_KEYBOARD_EDITOR_H
#define KYTY_LAUNCHER_LINUX_KEYBOARD_EDITOR_H

#include "keyboard_layout.h"

#include <map>
#include <string>

struct SDL_Renderer;
struct SDL_Texture;
struct ImFont;
class Settings;

// The window "Keyboard and mouse controls" (launcher-windows' KeyboardControlsWindow.cs): the 21
// controls of the layout, each with its key cap as the prompt icon will show it. Click one, press
// a key, a mouse button or the wheel; Save writes the layout to launcher-settings.json. A modal
// of the launcher's window, drawn with Dear ImGui.
class KeyboardEditor {
public:
	explicit KeyboardEditor(SDL_Renderer* renderer): renderer_(renderer) {}
	~KeyboardEditor();
	KeyboardEditor(const KeyboardEditor&)            = delete;
	KeyboardEditor& operator=(const KeyboardEditor&) = delete;

	// Opens it with the saved layout (an unusable one too, to be repaired here).
	void Open(Settings& settings);
	[[nodiscard]] bool IsOpen() const { return settings_ != nullptr; }

	// One frame (between ImGui::NewFrame and ImGui::Render); `scale` is the window's interface
	// scale, `title_font` the font of its page titles.
	void Frame(float scale, ImFont* title_font);

	// The name the layout and the emulator use for a key or mouse input of the editor's capture.
	// Public for the tests of the mapping.
	static std::string KeyName(int imgui_key);

private:
	void         Close();
	void         Save();
	void         Capture();
	void         Assign(const std::string& input);
	SDL_Texture* Preview(const std::string& label);
	void         ClearPreviews();

	SDL_Renderer* renderer_;
	Settings*     settings_ = nullptr;

	KeyboardLayout layout_;
	std::string    sensitivity_;
	std::string    capturing_; // the control waiting for an input, or empty
	std::string    status_;
	bool           open_request_    = false;
	bool           swallow_         = false; // the input that just set a binding is still down
	bool           command_hovered_ = false; // a command button is under the mouse
	bool           nav_restore_     = false;
	bool           clear_pending_   = false;

	// The key caps drawn so far, by label (nullptr: it could not be drawn).
	std::map<std::string, SDL_Texture*> previews_;
	bool                                preview_failed_ = false;
};

#endif
