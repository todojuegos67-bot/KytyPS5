#include "keyboard_editor.h"

#include "prompt_texture.h"
#include "settings.h"
#include "strings.h"

#include <SDL.h>
#include <algorithm>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

using Strings::F;
using Strings::T;

namespace {

std::string Trim(const std::string& text) {
	const auto first = text.find_first_not_of(" \t\r\n");
	if (first == std::string::npos) return {};
	return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

// The keyboard keys of ImGui's list (Tab .. KeypadEqual): the ones a capture looks at.
constexpr int FirstKey = ImGuiKey_Tab, LastKey = ImGuiKey_KeypadEqual;

bool AnyInputDown() {
	for (int key = FirstKey; key <= LastKey; ++key)
		if (ImGui::IsKeyDown(static_cast<ImGuiKey>(key))) return true;
	for (int button = 0; button < 5; ++button)
		if (ImGui::IsMouseDown(button)) return true;
	return false;
}

} // namespace

std::string KeyboardEditor::KeyName(int key) {
	if (key >= ImGuiKey_A && key <= ImGuiKey_Z) return std::string(1, static_cast<char>('A' + (key - ImGuiKey_A)));
	if (key >= ImGuiKey_0 && key <= ImGuiKey_9) return std::string(1, static_cast<char>('0' + (key - ImGuiKey_0)));
	if (key >= ImGuiKey_F1 && key <= ImGuiKey_F12) return "F" + std::to_string(key - ImGuiKey_F1 + 1);
	if (key >= ImGuiKey_F13 && key <= ImGuiKey_F24) return "F" + std::to_string(key - ImGuiKey_F13 + 13);
	if (key >= ImGuiKey_Keypad0 && key <= ImGuiKey_Keypad9) return "Keypad " + std::to_string(key - ImGuiKey_Keypad0);
	switch (key) {
		case ImGuiKey_Tab: return "Tab";
		case ImGuiKey_LeftArrow: return "Left";
		case ImGuiKey_RightArrow: return "Right";
		case ImGuiKey_UpArrow: return "Up";
		case ImGuiKey_DownArrow: return "Down";
		case ImGuiKey_PageUp: return "PageUp";
		case ImGuiKey_PageDown: return "PageDown";
		case ImGuiKey_Home: return "Home";
		case ImGuiKey_End: return "End";
		case ImGuiKey_Insert: return "Insert";
		case ImGuiKey_Delete: return "Delete";
		case ImGuiKey_Backspace: return "Backspace";
		case ImGuiKey_Space: return "Space";
		case ImGuiKey_Enter:
		case ImGuiKey_KeypadEnter: return "Return";
		case ImGuiKey_Escape: return "Escape";
		// (The emulator treats the right modifiers as the left ones.)
		case ImGuiKey_LeftCtrl:
		case ImGuiKey_RightCtrl: return "Left Ctrl";
		case ImGuiKey_LeftShift:
		case ImGuiKey_RightShift: return "Left Shift";
		case ImGuiKey_LeftAlt:
		case ImGuiKey_RightAlt: return "Left Alt";
		case ImGuiKey_Minus: return "-";
		case ImGuiKey_Equal: return "=";
		case ImGuiKey_LeftBracket: return "[";
		case ImGuiKey_RightBracket: return "]";
		case ImGuiKey_Backslash: return "\\";
		case ImGuiKey_Semicolon: return ";";
		case ImGuiKey_Apostrophe: return "'";
		case ImGuiKey_GraveAccent: return "`";
		case ImGuiKey_Comma: return ",";
		case ImGuiKey_Period: return ".";
		case ImGuiKey_Slash: return "/";
		case ImGuiKey_KeypadAdd: return "Keypad +";
		case ImGuiKey_KeypadSubtract: return "Keypad -";
		case ImGuiKey_KeypadMultiply: return "Keypad *";
		case ImGuiKey_KeypadDivide: return "Keypad /";
		case ImGuiKey_KeypadDecimal: return "Keypad .";
		case ImGuiKey_CapsLock: return "CapsLock";
		case ImGuiKey_NumLock: return "NumLock";
		case ImGuiKey_ScrollLock: return "ScrollLock";
		case ImGuiKey_Pause: return "Pause"; // (reserved: the layout says so)
		default: return "Unsupported";
	}
}

KeyboardEditor::~KeyboardEditor() {
	ClearPreviews();
}

void KeyboardEditor::ClearPreviews() {
	for (auto& [label, texture]: previews_)
		if (texture != nullptr) SDL_DestroyTexture(texture);
	previews_.clear();
	preview_failed_ = false;
}

void KeyboardEditor::Open(Settings& settings) {
	settings_    = &settings;
	layout_      = KeyboardLayout::Read(settings, false);
	sensitivity_ = layout_.sensitivity;
	capturing_.clear();
	status_ = T("kbd.status");
	try {
		layout_.Validate();
	} catch (const PromptError& error) {
		status_ = std::string(error.what()) + T("kbd.repair");
	}
	open_request_ = true;
	swallow_      = false;
}

SDL_Texture* KeyboardEditor::Preview(const std::string& label) {
	const auto found = previews_.find(label);
	if (found != previews_.end()) return found->second;
	SDL_Texture* texture = nullptr;
	try {
		const auto pixels = PromptTexture::Scaled(PromptTexture::Draw(label), 128);
		texture = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, 128, 128);
		if (texture != nullptr) {
			SDL_UpdateTexture(texture, nullptr, pixels.data(), 128 * 4);
			SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);
			SDL_SetTextureScaleMode(texture, SDL_ScaleModeBest);
		}
	} catch (const PromptError& error) {
		if (!preview_failed_) status_ = F("kbd.nopreview", {error.what()});
		preview_failed_ = true;
	}
	previews_[label] = texture;
	return texture;
}

void KeyboardEditor::Close() {
	ImGui::CloseCurrentPopup();
	settings_      = nullptr;
	capturing_.clear();
	clear_pending_ = true; // (the textures may be in this frame's draw list: freed in the next)
	if (nav_restore_) {
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		nav_restore_ = false;
	}
}

void KeyboardEditor::Assign(const std::string& input) {
	try {
		const auto other = layout_.Assign(capturing_, input);
		status_          = other.empty() ? std::string(T("kbd.updated")) : F("kbd.swapped", {other});
	} catch (const PromptError& error) {
		status_ = error.what();
	}
	capturing_.clear();
	swallow_ = true; // (the input that did it must not press a button when it is let go)
	if (nav_restore_) {
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		nav_restore_ = false;
	}
}

// The input of this frame while a control waits for one: a key, a mouse button (not on one of the
// window's own command buttons) or a step of the wheel.
void KeyboardEditor::Capture() {
	if (capturing_.empty()) return;
	ImGuiIO& io = ImGui::GetIO();
	for (int key = FirstKey; key <= LastKey; ++key) {
		if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(key), false)) {
			Assign(KeyName(key));
			return;
		}
	}
	static const char* const buttons[] = {"Mouse:Left", "Mouse:Right", "Mouse:Middle", "Mouse:X1", "Mouse:X2"};
	for (int button = 0; button < 5; ++button) {
		if (ImGui::IsMouseClicked(button) && !command_hovered_) {
			Assign(buttons[button]);
			return;
		}
	}
	if (io.MouseWheel != 0.0f) Assign(io.MouseWheel > 0.0f ? "Mouse:WheelUp" : "Mouse:WheelDown");
}

void KeyboardEditor::Save() {
	if (settings_ == nullptr) return;
	layout_.sensitivity = Trim(sensitivity_);
	// The saved values first: a layout that cannot be written is not left half in the settings.
	std::map<std::string, std::string> previous;
	for (const auto& control: KeyboardLayout::Controls())
		previous[control] = settings_->Str("keyboard_binding_" + control);
	const auto previous_sensitivity = settings_->Str("keyboard_mouse_sensitivity");
	auto       revert               = [&] {
        for (const auto& [control, input]: previous)
            settings_->Set("keyboard_binding_" + control, input);
        settings_->Set("keyboard_mouse_sensitivity", previous_sensitivity);
	};
	try {
		layout_.Validate();
		layout_.Save(*settings_);
	} catch (const PromptError& error) {
		revert();
		status_ = error.what();
		return;
	}
	if (!settings_->Save()) {
		revert();
		status_ = F("kbd.saveerror", {settings_->Path().string()});
		return;
	}
	Close();
}

void KeyboardEditor::Frame(float scale, ImFont* title_font) {
	if (clear_pending_ && settings_ == nullptr) {
		ClearPreviews();
		clear_pending_ = false;
	}
	if (open_request_) {
		ImGui::OpenPopup("##keyboard");
		open_request_  = false;
		clear_pending_ = false;
	}
	if (settings_ == nullptr && !ImGui::IsPopupOpen("##keyboard")) return;
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
	ImGui::SetNextWindowSize({std::min(780.0f * scale, viewport->Size.x - 40),
	                          std::min(700.0f * scale, viewport->Size.y - 40)});
	if (!ImGui::BeginPopupModal("##keyboard", nullptr,
	                            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove))
		return;
	if (swallow_ && !AnyInputDown()) swallow_ = false;
	command_hovered_ = false;
	// While a control waits for an input the keys must not move the window's focus or press its
	// buttons.
	ImGuiIO& io = ImGui::GetIO();
	if (!capturing_.empty() && (io.ConfigFlags & ImGuiConfigFlags_NavEnableKeyboard) != 0) {
		io.ConfigFlags &= ~ImGuiConfigFlags_NavEnableKeyboard;
		nav_restore_ = true;
	}

	ImGui::PushFont(title_font, 22.0f);
	ImGui::TextUnformatted(T("kbd.title"));
	ImGui::PopFont();
	ImGui::PushTextWrapPos(0.0f);
	ImGui::TextUnformatted(T("kbd.intro"));
	ImGui::PopTextWrapPos();
	ImGui::Dummy({0, 6.0f * scale});

	const float footer = ImGui::GetFrameHeightWithSpacing() * 3 + ImGui::GetTextLineHeight() * 3;
	ImGui::BeginChild("##controls", {0, -footer}, ImGuiChildFlags_Borders);
	const auto& controls = KeyboardLayout::Controls();
	const float image    = 44.0f * scale;
	for (size_t i = 0; i < controls.size(); ++i) {
		const auto& control = controls[i];
		ImGui::PushID(static_cast<int>(i));
		const std::string action = std::string(T(("kbd.action." + std::to_string(i)).c_str())) + " (" + control + ")";
		const float       row_y  = ImGui::GetCursorPosY();
		ImGui::SetCursorPosY(row_y + (image - ImGui::GetTextLineHeight()) * 0.5f);
		ImGui::TextUnformatted(action.c_str());
		ImGui::SameLine(ImGui::GetWindowWidth() - 300.0f * scale);
		ImGui::SetCursorPosY(row_y + (image - ImGui::GetFrameHeight()) * 0.5f);
		const bool        waiting = capturing_ == control;
		const std::string text    = waiting ? std::string(T("kbd.press")) : layout_.Binding(control);
		// (Pressed while another waits, it is that input; and what just set a binding, let go,
		// must not start the next.)
		if (ImGui::Button((text + "##binding").c_str(), {190.0f * scale, 0}) && capturing_.empty() && !swallow_) {
			capturing_ = control;
			status_    = T("kbd.capture");
		}
		ImGui::SameLine();
		ImGui::SetCursorPosY(row_y);
		const auto at = ImGui::GetCursorScreenPos();
		ImGui::Dummy({image, image});
		if (SDL_Texture* texture = Preview(KeyboardLayout::Label(layout_.Binding(control))))
			ImGui::GetWindowDrawList()->AddImage(static_cast<ImTextureID>(reinterpret_cast<intptr_t>(texture)), at,
			                                     {at.x + image, at.y + image});
		ImGui::PopID();
	}
	ImGui::EndChild();

	ImGui::PushTextWrapPos(0.0f);
	ImGui::TextUnformatted(status_.c_str());
	ImGui::PopTextWrapPos();
	ImGui::AlignTextToFramePadding();
	ImGui::TextUnformatted(T("kbd.sensitivity"));
	ImGui::SameLine();
	ImGui::SetNextItemWidth(80.0f * scale);
	ImGui::InputText("##sensitivity", &sensitivity_);
	if (ImGui::IsItemActive() || ImGui::IsItemHovered()) command_hovered_ = true; // (a click here is no binding)

	// The command buttons, on the right.
	const float button_width = 130.0f * scale;
	ImGui::SetCursorPosX(ImGui::GetWindowWidth() - (button_width * 4 + ImGui::GetStyle().ItemSpacing.x * 3 +
	                                                ImGui::GetStyle().WindowPadding.x));
	bool stop = false, reset = false, cancel = false, save = false;
	stop = ImGui::Button(T("kbd.stop"), {button_width, 0});
	command_hovered_ = command_hovered_ || ImGui::IsItemHovered();
	ImGui::SameLine();
	reset = ImGui::Button(T("kbd.reset"), {button_width, 0});
	command_hovered_ = command_hovered_ || ImGui::IsItemHovered();
	ImGui::SameLine();
	cancel = ImGui::Button(T("cancel"), {button_width, 0});
	command_hovered_ = command_hovered_ || ImGui::IsItemHovered();
	ImGui::SameLine();
	save = ImGui::Button(T("kbd.save"), {button_width, 0});
	command_hovered_ = command_hovered_ || ImGui::IsItemHovered();

	// The input of this frame is taken after the buttons are known (a click on a command button is
	// that command, not a binding).
	Capture();
	if (stop || reset || cancel || save) {
		capturing_.clear();
		if (nav_restore_) {
			io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
			nav_restore_ = false;
		}
	}
	if (stop) status_ = T("kbd.status");
	if (reset) {
		layout_      = KeyboardLayout();
		sensitivity_.clear();
		status_ = T("kbd.resetdone");
	}
	if (cancel || (capturing_.empty() && !swallow_ && ImGui::IsKeyPressed(ImGuiKey_Escape) && !ImGui::IsAnyItemActive()))
		Close();
	else if (save)
		Save();
	ImGui::EndPopup();
}
