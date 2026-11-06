// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// (bb-menu-104) The Bloodborne F1 menu. Look and layout from the Demon's Souls port's F1 menu
// (ds-menu.cpp: dark panel, gold titles, the launcher's pad photo mando-ps5.png / mando-8bitdo.png
// and raton.png with clickable spots; trophy page; trophy popup card and chime of trophy.cpp).

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <SDL3/SDL.h>
#include <fmt/format.h>
#include <imgui.h>
#include <pugixml.hpp>

#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "imgui/bb_menu.h"
#include "imgui/imgui_layer.h"
#include "imgui/imgui_std.h"
#include "imgui/imgui_texture.h"
#include "imgui/renderer/imgui_core.h"
#include "input/input_handler.h"
#include "input/input_mouse.h"

namespace BbMenu {
namespace {

// ---------------------------------------------------------------- colours (the DS menu's)
const ImVec4 kGold{0.93f, 0.78f, 0.47f, 1.0f};
const ImVec4 kDim{0.72f, 0.70f, 0.66f, 1.0f};
const ImVec4 kWarn{1.0f, 0.78f, 0.30f, 1.0f};
const ImU32 kRing = IM_COL32(236, 200, 120, 255);

// ---------------------------------------------------------------- motion helpers
float EaseOut(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return 1.0f - (1.0f - x) * (1.0f - x) * (1.0f - x);
}
float EaseInOut(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x < 0.5f ? 4 * x * x * x : 1.0f - std::pow(-2.0f * x + 2.0f, 3.0f) * 0.5f;
}
ImU32 Fade(ImU32 c, float a) {
    const u32 al = static_cast<u32>(((c >> IM_COL32_A_SHIFT) & 0xFF) * std::clamp(a, 0.0f, 1.0f));
    return (c & ~IM_COL32_A_MASK) | (al << IM_COL32_A_SHIFT);
}
// hover glow (pulsing gold rim) and press feedback over the last item
void ItemGlow(float rounding) {
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    auto* dl = ImGui::GetWindowDrawList();
    const float al = ImGui::GetStyle().Alpha;
    if (ImGui::IsItemActive()) {
        dl->AddRectFilled(a, b, IM_COL32(0, 0, 0, static_cast<int>(70 * al)), rounding);
        dl->AddRect(a, b, IM_COL32(255, 220, 150, static_cast<int>(230 * al)), rounding, 0, 2.0f);
    } else if (ImGui::IsItemHovered() || ImGui::IsItemFocused()) {
        const float pulse = 0.65f + 0.35f * std::sin(static_cast<float>(ImGui::GetTime()) * 5.0f);
        for (int i = 3; i >= 1; i--) {
            const float g = static_cast<float>(i) * 2.0f;
            dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g},
                        IM_COL32(236, 196, 110, static_cast<int>(40 * pulse * al / i)),
                        rounding + g, 0, 2.0f);
        }
        dl->AddRect(a, b, IM_COL32(236, 200, 120, static_cast<int>(200 * pulse * al)), rounding,
                    0, 1.5f);
    }
}
bool FancyButton(const char* label, const ImVec2& size = {0, 0}) {
    const bool r = ImGui::Button(label, size);
    ItemGlow(ImGui::GetStyle().FrameRounding);
    return r;
}
// eased wheel scrolling for the current child window
void SmoothScroll(float& target, float unit) {
    const ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f) {
        if (target < 0) {
            target = ImGui::GetScrollY();
        }
        target = std::clamp(target - io.MouseWheel * 160.0f * unit, 0.0f, ImGui::GetScrollMaxY());
    }
    if (target >= 0) {
        const float cur = ImGui::GetScrollY();
        float n = cur + (target - cur) * std::min(1.0f, io.DeltaTime * 14.0f);
        if (std::fabs(target - n) < 0.5f) {
            n = target;
            target = -1;
        }
        ImGui::SetScrollY(n);
    }
}

// ---------------------------------------------------------------- shared state
SDL_Window* g_window = nullptr;
std::atomic<bool> g_open{false};
Uint32 g_wake_event = 0;
std::atomic<bool> g_want_close{false};
std::atomic<bool> g_want_reload{false};
Input::MouseMode g_saved_mouse = Input::MouseMode::Off;
bool g_saved_relative = false;
bool g_restore_mouse_look = false; // the Teclado y ratón checkbox: mouse-look on when closing

// capture of a physical input (main thread writes, render thread reads)
enum class Capture { None, Pad, PadStick, Kbm };
std::mutex g_cap_mtx;
Capture g_capture = Capture::None;
Uint64 g_capture_since = 0;
std::string g_captured;  // name in the input config's words ("cross", "space", "leftbutton")
bool g_cap_cancel = false;
std::string g_capture_target; // output being rebound

// live physical pad state for the highlight
std::atomic<u32> g_pad_buttons{0}; // bit = SDL_GamepadButton
std::atomic<int> g_trig_l{0}, g_trig_r{0};

void Wake() {
    if (g_wake_event != 0) {
        SDL_Event e{};
        e.type = g_wake_event;
        SDL_PushEvent(&e);
    }
}

std::string Serial() {
    return std::string(Common::ElfInfo::Instance().GameSerial());
}

// ---------------------------------------------------------------- open / close (main thread)
void SetOpen(bool open) {
    if (open == g_open.load()) {
        return;
    }
    if (open) {
        ImGui::Core::AcquireGamepadInputCapture();
        // mouse-look off and the cursor free while the menu is open
        g_saved_mouse = Input::GetMouseMode();
        g_restore_mouse_look = g_saved_mouse == Input::MouseMode::Joystick;
        if (g_saved_mouse != Input::MouseMode::Off) {
            Input::ToggleMouseModeTo(g_saved_mouse); // same mode again = off
        }
        g_saved_relative = g_window != nullptr && SDL_GetWindowRelativeMouseMode(g_window);
        if (g_window != nullptr) {
            SDL_SetWindowRelativeMouseMode(g_window, false);
        }
        SDL_ShowCursor();
        g_open = true;
        LOG_INFO(Frontend, "[bb-menu] F1 menu open: game input paused");
    } else {
        g_open = false;
        {
            std::scoped_lock lk{g_cap_mtx};
            g_capture = Capture::None;
        }
        ImGui::Core::ReleaseGamepadInputCapture();
        if (g_restore_mouse_look) {
            if (Input::GetMouseMode() != Input::MouseMode::Joystick) {
                Input::ToggleMouseModeTo(Input::MouseMode::Joystick);
            }
            if (g_window != nullptr) {
                SDL_SetWindowRelativeMouseMode(g_window, true);
            }
        } else if (g_saved_mouse != Input::MouseMode::Off &&
                   g_saved_mouse != Input::MouseMode::Joystick) {
            Input::ToggleMouseModeTo(g_saved_mouse);
            if (g_window != nullptr) {
                SDL_SetWindowRelativeMouseMode(g_window, g_saved_relative);
            }
        }
        LOG_INFO(Frontend, "[bb-menu] F1 menu closed: game input back");
    }
}

// ---------------------------------------------------------------- names
const char* PadName(u8 b) {
    switch (b) {
    case SDL_GAMEPAD_BUTTON_SOUTH: return "cross";
    case SDL_GAMEPAD_BUTTON_EAST: return "circle";
    case SDL_GAMEPAD_BUTTON_WEST: return "square";
    case SDL_GAMEPAD_BUTTON_NORTH: return "triangle";
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return "l1";
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return "r1";
    case SDL_GAMEPAD_BUTTON_LEFT_STICK: return "l3";
    case SDL_GAMEPAD_BUTTON_RIGHT_STICK: return "r3";
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return "pad_up";
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return "pad_down";
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return "pad_left";
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return "pad_right";
    case SDL_GAMEPAD_BUTTON_START: return "options";
    case SDL_GAMEPAD_BUTTON_BACK: return "back";
    case SDL_GAMEPAD_BUTTON_LEFT_PADDLE1: return "lpaddle_high";
    case SDL_GAMEPAD_BUTTON_LEFT_PADDLE2: return "lpaddle_low";
    case SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1: return "rpaddle_high";
    case SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2: return "rpaddle_low";
    default: return nullptr;
    }
}

bool IsMouseName(const std::string& n) {
    return n.ends_with("button") || n.starts_with("mousewheel") || n.starts_with("sidebutton");
}

std::string KeyName(u32 code, bool mouse) {
    for (const auto& [name, v] : Input::string_to_keyboard_key_map) {
        if (v == code && IsMouseName(name) == mouse && name != "lwin" && name != "rwin") {
            return name;
        }
    }
    return {};
}

std::string Pretty(const std::string& token) {
    static const std::map<std::string, std::string> names = {
        {"cross", "✕ Cruz"}, {"circle", "○ Círculo"}, {"square", "□ Cuadrado"},
        {"triangle", "△ Triángulo"}, {"l1", "L1"}, {"r1", "R1"}, {"l2", "L2"}, {"r2", "R2"},
        {"l3", "L3"}, {"r3", "R3"}, {"options", "OPTIONS"}, {"back", "SHARE"},
        {"share", "SHARE"}, {"pad_up", "Cruceta arriba"}, {"pad_down", "Cruceta abajo"},
        {"pad_left", "Cruceta izquierda"}, {"pad_right", "Cruceta derecha"},
        {"axis_left_x", "Stick izquierdo"}, {"axis_left_y", "Stick izquierdo"},
        {"axis_right_x", "Stick derecho"}, {"axis_right_y", "Stick derecho"},
        {"lpaddle_high", "Pala izq. 1"}, {"lpaddle_low", "Pala izq. 2"},
        {"rpaddle_high", "Pala der. 1"}, {"rpaddle_low", "Pala der. 2"},
        {"space", "Espacio"}, {"enter", "Intro"}, {"escape", "Esc"},
        {"lshift", "Mayús izq."}, {"rshift", "Mayús der."}, {"lctrl", "Ctrl izq."},
        {"rctrl", "Ctrl der."}, {"lalt", "Alt izq."}, {"ralt", "Alt der."},
        {"tab", "Tabulador"}, {"backspace", "Retroceso"}, {"capslock", "Bloq Mayús"},
        {"up", "Flecha arriba"}, {"down", "Flecha abajo"}, {"left", "Flecha izq."},
        {"right", "Flecha der."}, {"leftbutton", "Clic izquierdo"},
        {"rightbutton", "Clic derecho"}, {"middlebutton", "Clic central"},
        {"sidebuttonback", "Botón lateral atrás"}, {"sidebuttonforward", "Botón lateral adelante"},
        {"mousewheelup", "Rueda arriba"}, {"mousewheeldown", "Rueda abajo"},
        {"mousewheelleft", "Rueda izq."}, {"mousewheelright", "Rueda der."},
        {"delete", "Supr"}, {"insert", "Insert"}, {"home", "Inicio"}, {"end", "Fin"},
        {"pgup", "RePág"}, {"pgdown", "AvPág"}};
    if (auto it = names.find(token); it != names.end()) {
        return it->second;
    }
    std::string up = token;
    for (auto& c : up) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return up;
}

std::string PrettyBinding(const std::string& b) {
    if (b.empty()) {
        return "(sin asignar)";
    }
    std::string out;
    std::stringstream ss(b);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        out += (out.empty() ? "" : " + ") + Pretty(tok);
    }
    return out;
}

// ---------------------------------------------------------------- the input config file
enum class Cat { Other, Pad, Kbm };

Cat Classify(const std::string& input) {
    if (input.empty()) {
        return Cat::Other;
    }
    std::stringstream ss(input);
    std::string tok;
    bool any_kbm = false, all_pad = true;
    while (std::getline(ss, tok, ',')) {
        if (const auto c = tok.find(':'); c != std::string::npos) {
            tok = tok.substr(0, c);
        }
        if (Input::string_to_keyboard_key_map.contains(tok)) {
            any_kbm = true;
        } else if (!Input::string_to_cbutton_map.contains(tok) &&
                   !Input::string_to_axis_map.contains(tok)) {
            all_pad = false;
        }
    }
    return any_kbm ? Cat::Kbm : all_pad ? Cat::Pad : Cat::Other;
}

struct Line {
    std::string raw, out, in; // out/in: stripped, "" for comments
};

struct Config {
    std::filesystem::path path;
    std::vector<Line> lines;
    bool loaded = false;

    void Load() {
        const std::string id = EmulatorSettings.IsUseUnifiedInputConfig() ? "default" : Serial();
        path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "input_config" /
               (id + ".ini");
        if (!std::filesystem::exists(path)) {
            // the emulator makes it at start (GetInputConfigFile); fall back to default.ini
            const auto def = path.parent_path() / "default.ini";
            std::error_code ec;
            std::filesystem::copy_file(def, path, ec);
        }
        lines.clear();
        std::ifstream in(path);
        std::string raw;
        while (std::getline(in, raw)) {
            Line l{raw, {}, {}};
            std::string s = raw;
            if (const auto h = s.find('#'); h != std::string::npos) {
                s = s.substr(0, h);
            }
            s.erase(std::remove_if(s.begin(), s.end(),
                                   [](unsigned char c) { return std::isspace(c); }),
                    s.end());
            if (const auto eq = s.find('='); eq != std::string::npos) {
                l.out = s.substr(0, eq);
                l.in = s.substr(eq + 1);
                if (const auto c = l.out.find(':'); c != std::string::npos) {
                    l.out = l.out.substr(0, c);
                }
            }
            lines.push_back(std::move(l));
        }
        loaded = true;
    }

    void Save() {
        std::ofstream out(path, std::ios::trunc);
        for (const auto& l : lines) {
            out << l.raw << "\n";
        }
        out.close();
        LOG_INFO(Frontend, "[bb-menu] input config saved to {}", path.string());
        g_want_reload = true;
        Wake();
    }

    std::string Get(const std::string& output, Cat cat) const {
        std::string all;
        for (const auto& l : lines) {
            if (l.out == output && Classify(l.in) == cat) {
                all += (all.empty() ? "" : " / ") + PrettyBinding(l.in);
            }
        }
        return all.empty() ? PrettyBinding("") : all;
    }

    std::string GetRaw(const std::string& output, Cat cat) const {
        for (const auto& l : lines) {
            if (l.out == output && Classify(l.in) == cat) {
                return l.in;
            }
        }
        return {};
    }

    bool Uses(const std::string& output, Cat cat, const std::string& input) const {
        for (const auto& l : lines) {
            if (l.out == output && Classify(l.in) == cat) {
                std::stringstream ss(l.in);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    if (tok == input) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // Replaces every `cat` binding of `output` with `input` ("" = unbind). A binding of the
    // same input on another output of the same group gets the old one (a swap).
    void Set(const std::string& output, Cat cat, const std::string& input,
             const std::vector<std::string>& group) {
        const std::string old = GetRaw(output, cat);
        std::vector<std::string> swapped;
        if (!input.empty()) {
            for (const auto& o : group) {
                if (o != output && GetRaw(o, cat) == input) {
                    swapped.push_back(o);
                }
            }
        }
        auto drop = [&](const std::string& o) {
            std::erase_if(lines,
                          [&](const Line& l) { return l.out == o && Classify(l.in) == cat; });
        };
        drop(output);
        for (const auto& o : swapped) {
            drop(o);
        }
        auto add = [&](const std::string& o, const std::string& in) {
            if (in.empty()) {
                return;
            }
            lines.push_back({o + " = " + in + "  # F1 menu", o, in});
        };
        add(output, input);
        for (const auto& o : swapped) {
            add(o, old);
        }
        Save();
    }

    // mouse_movement_params = deadzone, speed, speed offset
    void MouseParams(float& dz, float& sp, float& off) const {
        dz = 0.5f, sp = 1.0f, off = 0.125f;
        for (const auto& l : lines) {
            if (l.out == "mouse_movement_params") {
                std::stringstream ss(l.in);
                char c;
                ss >> dz >> c >> sp >> c >> off;
            }
        }
    }
    void SetMouseParams(float dz, float sp, float off) {
        std::erase_if(lines, [](const Line& l) { return l.out == "mouse_movement_params"; });
        const auto v = fmt::format("{:.3f}, {:.3f}, {:.3f}", dz, sp, off);
        lines.push_back({"mouse_movement_params = " + v + "  # F1 menu", "mouse_movement_params",
                         v});
        Save();
    }
    std::string MouseStick() const {
        for (const auto& l : lines) {
            if (l.out == "mouse_to_joystick") {
                return l.in;
            }
        }
        return "right";
    }
    void SetMouseStick(const std::string& s) {
        std::erase_if(lines, [](const Line& l) { return l.out == "mouse_to_joystick"; });
        lines.push_back({"mouse_to_joystick = " + s + "  # F1 menu", "mouse_to_joystick", s});
        Save();
    }
};
Config g_cfg;

// ---------------------------------------------------------------- textures
struct Tex {
    ImGui::RefCountedTexture tex;
    bool tried = false;
    float w = 0, h = 0;
};

std::filesystem::path AssetDir() {
    const char* base = SDL_GetBasePath();
    std::filesystem::path p = base != nullptr ? std::filesystem::path(base) : ".";
    return p / "menu_assets";
}

ImTextureID Get(Tex& t, const std::filesystem::path& file, float fw, float fh) {
    if (!t.tried) {
        t.tried = true;
        t.w = fw;
        t.h = fh;
        if (std::filesystem::exists(file)) {
            t.tex = ImGui::RefCountedTexture::DecodePngFile(file);
        } else {
            LOG_WARNING(Frontend, "[bb-menu] missing image {}", file.string());
        }
    }
    if (!t.tex) {
        return ImTextureID{};
    }
    const auto img = t.tex.GetTexture();
    if (img.width > 0) {
        t.w = static_cast<float>(img.width);
        t.h = static_cast<float>(img.height);
    }
    return img.im_id;
}

Tex g_pad_tex, g_pad_alt_tex, g_mouse_tex;
int g_pad_photo = 0; // 0 mando-ps5.png, 1 mando-8bitdo.png

// ---------------------------------------------------------------- PS4 symbols (drawn)
void DrawSymbol(ImDrawList* dl, const std::string& out, ImVec2 c, float r, ImU32 col) {
    const float t = std::max(1.5f, r * 0.22f);
    if (out == "cross") {
        dl->AddLine({c.x - r * 0.6f, c.y - r * 0.6f}, {c.x + r * 0.6f, c.y + r * 0.6f}, col, t);
        dl->AddLine({c.x - r * 0.6f, c.y + r * 0.6f}, {c.x + r * 0.6f, c.y - r * 0.6f}, col, t);
    } else if (out == "circle") {
        dl->AddCircle(c, r * 0.65f, col, 24, t);
    } else if (out == "square") {
        dl->AddRect({c.x - r * 0.6f, c.y - r * 0.6f}, {c.x + r * 0.6f, c.y + r * 0.6f}, col, 0, 0,
                    t);
    } else if (out == "triangle") {
        dl->AddTriangle({c.x, c.y - r * 0.7f}, {c.x - r * 0.7f, c.y + r * 0.5f},
                        {c.x + r * 0.7f, c.y + r * 0.5f}, col, t);
    }
}

// ---------------------------------------------------------------- the mapper pages
struct Spot {
    const char* output; // input-config output ("stick_left" = both axes)
    const char* label;
    float x, y, d, h; // centre and size on the 350 px high photo
};

// mando-ps5.png (532 x 350): the DS port's measured spots + the shoulders, SHARE and PS.
const Spot kPadSpots[] = {
    {"pad_up", "Cruceta arriba", 96, 62, 29, 29},
    {"pad_down", "Cruceta abajo", 96, 126, 29, 29},
    {"pad_left", "Cruceta izquierda", 65, 93, 29, 29},
    {"pad_right", "Cruceta derecha", 128, 93, 29, 29},
    {"triangle", "△ Triángulo", 435, 60, 38, 38},
    {"circle", "○ Círculo", 472, 94, 38, 38},
    {"cross", "✕ Cruz", 435, 131, 38, 38},
    {"square", "□ Cuadrado", 398, 94, 38, 38},
    {"l3", "L3 (pulsar stick izq.)", 178, 160, 67, 67},
    {"r3", "R3 (pulsar stick der.)", 346, 160, 67, 67},
    {"options", "OPTIONS", 389, 32, 20, 20},
    {"touchpad_center", "Panel táctil", 265, 52, 150, 80},
    {"l1", "L1", 95, 9, 56, 16},
    {"r1", "R1", 432, 9, 56, 16},
};
// mando-8bitdo.png (537 x 350), the DS port's spots, PS4 meaning.
const Spot kAltSpots[] = {
    {"triangle", "△ Triángulo", 416, 58, 38, 38}, {"square", "□ Cuadrado", 377, 95, 38, 38},
    {"circle", "○ Círculo", 456, 95, 38, 38},     {"cross", "✕ Cruz", 416, 132, 38, 38},
    {"l3", "L3", 125, 86, 72, 72},                {"r3", "R3", 339, 156, 72, 72},
    {"pad_up", "Cruceta arriba", 193, 140, 25, 25},
    {"pad_down", "Cruceta abajo", 193, 192, 25, 25},
    {"pad_left", "Cruceta izquierda", 167, 166, 25, 25},
    {"pad_right", "Cruceta derecha", 219, 166, 25, 25},
    {"touchpad_center", "Panel táctil (SHARE)", 226, 89, 22, 22},
    {"options", "OPTIONS", 312, 89, 22, 22},
};

struct Row {
    const char* output;
    const char* label;
};
const Row kPadRows[] = {
    {"cross", "✕ Cruz"},          {"circle", "○ Círculo"},
    {"square", "□ Cuadrado"},     {"triangle", "△ Triángulo"},
    {"l1", "L1"},                 {"r1", "R1"},
    {"l2", "L2"},                 {"r2", "R2"},
    {"l3", "L3"},                 {"r3", "R3"},
    {"options", "OPTIONS"},       {"touchpad_center", "Panel táctil"},
    {"pad_up", "Cruceta arriba"}, {"pad_down", "Cruceta abajo"},
    {"pad_left", "Cruceta izquierda"}, {"pad_right", "Cruceta derecha"},
    {"stick_left", "Stick izquierdo (mover)"}, {"stick_right", "Stick derecho (cámara)"},
};
const Row kKbmRows[] = {
    {"cross", "✕ Cruz"},          {"circle", "○ Círculo"},
    {"square", "□ Cuadrado"},     {"triangle", "△ Triángulo"},
    {"l1", "L1"},                 {"r1", "R1"},
    {"l2", "L2"},                 {"r2", "R2"},
    {"l3", "L3"},                 {"r3", "R3"},
    {"options", "OPTIONS"},       {"touchpad_center", "Panel táctil"},
    {"pad_up", "Cruceta arriba"}, {"pad_down", "Cruceta abajo"},
    {"pad_left", "Cruceta izquierda"}, {"pad_right", "Cruceta derecha"},
    {"axis_left_y_minus", "Stick izq. arriba"}, {"axis_left_y_plus", "Stick izq. abajo"},
    {"axis_left_x_minus", "Stick izq. izquierda"}, {"axis_left_x_plus", "Stick izq. derecha"},
    {"axis_right_y_minus", "Stick der. arriba"}, {"axis_right_y_plus", "Stick der. abajo"},
    {"axis_right_x_minus", "Stick der. izquierda"}, {"axis_right_x_plus", "Stick der. derecha"},
};

std::vector<std::string> Group(const Row* rows, size_t n) {
    std::vector<std::string> g;
    for (size_t i = 0; i < n; i++) {
        g.emplace_back(rows[i].output);
    }
    return g;
}

std::string BindingText(const std::string& output, Cat cat) {
    if (output == "stick_left" || output == "stick_right") {
        const auto x = g_cfg.GetRaw(output == "stick_left" ? "axis_left_x" : "axis_right_x", cat);
        return x.empty() ? "(sin asignar)" : Pretty(x);
    }
    return g_cfg.Get(output, cat);
}

void StartCapture(Capture c, const std::string& target) {
    std::scoped_lock lk{g_cap_mtx};
    g_capture = c;
    g_capture_target = target;
    g_capture_since = SDL_GetTicks();
    g_captured.clear();
    g_cap_cancel = false;
}

// The capture's result, applied on the render thread.
void ApplyCapture() {
    std::string got, target;
    bool cancel = false;
    Capture kind;
    {
        std::scoped_lock lk{g_cap_mtx};
        if (g_capture == Capture::None || (g_captured.empty() && !g_cap_cancel)) {
            return;
        }
        got = g_captured;
        target = g_capture_target;
        cancel = g_cap_cancel;
        kind = g_capture;
        g_capture = Capture::None;
        g_captured.clear();
    }
    if (cancel) {
        return;
    }
    if (kind == Capture::PadStick) {
        const std::string want = target == "stick_left" ? "axis_left" : "axis_right";
        const std::string other = target == "stick_left" ? "axis_right" : "axis_left";
        const std::string src = got == "left" ? "axis_left" : "axis_right";
        const std::string other_src = got == "left" ? "axis_right" : "axis_left";
        std::vector<std::string> g{"axis_left_x", "axis_left_y", "axis_right_x", "axis_right_y"};
        g_cfg.Set(want + "_x", Cat::Pad, src + "_x", g);
        g_cfg.Set(want + "_y", Cat::Pad, src + "_y", g);
        g_cfg.Set(other + "_x", Cat::Pad, other_src + "_x", g);
        g_cfg.Set(other + "_y", Cat::Pad, other_src + "_y", g);
    } else if (kind == Capture::Pad) {
        g_cfg.Set(target, Cat::Pad, got, Group(kPadRows, std::size(kPadRows)));
    } else {
        g_cfg.Set(target, Cat::Kbm, got, Group(kKbmRows, std::size(kKbmRows)));
    }
}

bool PhysicallyHeld(const std::string& output) {
    // the outputs whose pad binding is a button held now (the live highlight)
    const u32 bits = g_pad_buttons.load();
    for (u8 b = 0; b < 32; b++) {
        if ((bits & (1u << b)) != 0) {
            if (const char* n = PadName(b); n != nullptr && g_cfg.Uses(output, Cat::Pad, n)) {
                return true;
            }
        }
    }
    if (g_trig_l > 12000 && g_cfg.Uses(output, Cat::Pad, "l2")) {
        return true;
    }
    if (g_trig_r > 12000 && g_cfg.Uses(output, Cat::Pad, "r2")) {
        return true;
    }
    return false;
}

std::string g_selected;

// One photo with its spots; returns the output clicked ("" none).
std::string Photo(const char* id, ImTextureID tex, float tw, float th, const Spot* spots,
                  size_t n, ImVec2 at, float scale) {
    auto* dl = ImGui::GetWindowDrawList();
    const float w = tw * scale, h = th * scale;
    if (tex) {
        dl->AddImage(tex, at, {at.x + w, at.y + h});
    } else {
        dl->AddRectFilled(at, {at.x + w, at.y + h}, IM_COL32(40, 44, 54, 255), 12 * scale);
        dl->AddText({at.x + 12, at.y + 12}, IM_COL32(220, 200, 150, 255),
                    "(imagen no encontrada: menu_assets)");
    }
    std::string clicked;
    for (size_t i = 0; i < n; i++) {
        const Spot& sp = spots[i];
        const ImVec2 c{at.x + sp.x * scale, at.y + sp.y * scale};
        const ImVec2 half{sp.d * scale * 0.5f, sp.h * scale * 0.5f};
        ImGui::SetCursorScreenPos({c.x - half.x, c.y - half.y});
        ImGui::PushID(id);
        ImGui::PushID(static_cast<int>(i));
        const bool pressed = ImGui::InvisibleButton(
            "##spot", {std::max(1.0f, half.x * 2), std::max(1.0f, half.y * 2)});
        const bool over = ImGui::IsItemHovered() || ImGui::IsItemFocused();
        ImGui::PopID();
        ImGui::PopID();
        const bool on = g_selected == sp.output || PhysicallyHeld(sp.output);
        const ImU32 ring = on     ? kRing
                           : over ? IM_COL32(236, 200, 120, 200)
                                  : IM_COL32(236, 200, 120, 90);
        const bool rect = sp.d != sp.h;
        if (on || over) {
            if (rect) {
                dl->AddRectFilled({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y},
                                  IM_COL32(236, 200, 120, 90), 8 * scale);
            } else {
                dl->AddCircleFilled(c, half.x, IM_COL32(236, 200, 120, 90));
            }
        }
        if (rect) {
            dl->AddRect({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y}, ring,
                        8 * scale, 0, on ? 3.0f : 1.5f);
        } else {
            dl->AddCircle(c, half.x, ring, 0, on ? 3.0f : 1.5f);
        }
        if (over) {
            ImGui::SetTooltip("%s", sp.label);
        }
        if (pressed) {
            clicked = sp.output;
        }
    }
    ImGui::SetCursorScreenPos({at.x, at.y + h});
    return clicked;
}

// L2 / R2 (not on the photo): chips over its corners.
std::string TriggerChips(ImVec2 at, float w, float unit) {
    std::string clicked;
    const char* names[] = {"l2", "r2"};
    for (int i = 0; i < 2; i++) {
        const float x = i == 0 ? at.x + 40 * unit : at.x + w - 140 * unit;
        ImGui::SetCursorScreenPos({x, at.y});
        const bool held = g_selected == names[i] || PhysicallyHeld(names[i]);
        if (held) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.45f, 0.25f, 1.0f));
        }
        if (FancyButton(i == 0 ? "L2##chip" : "R2##chip", {100 * unit, 0})) {
            clicked = names[i];
        }
        if (held) {
            ImGui::PopStyleColor();
        }
    }
    return clicked;
}

void CaptureBanner(float unit) {
    std::scoped_lock lk{g_cap_mtx};
    if (g_capture == Capture::None) {
        return;
    }
    const char* what = g_capture == Capture::Kbm
                           ? "Pulsa la tecla o el botón del ratón (Esc: cancelar)"
                       : g_capture == Capture::PadStick
                           ? "Mueve el stick del mando que quieres usar (Esc: cancelar)"
                           : "Pulsa el botón del mando (Esc: cancelar)";
    ImGui::TextColored(kWarn, "Asignando %s:  %s", Pretty(g_capture_target).c_str(), what);
}

void ListTable(const char* id, const Row* rows, size_t n, Cat cat, float unit) {
    if (ImGui::BeginTable(id, 3,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                              ImGuiTableFlags_ScrollY,
                          {0, ImGui::GetContentRegionAvail().y - 4 * unit})) {
        ImGui::TableSetupColumn("Botón PS4", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Asignado a", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 120 * unit);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < n; i++) {
            const std::string out = rows[i].output;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::PushID(rows[i].output);
            const bool held = cat == Cat::Pad && PhysicallyHeld(out);
            if (ImGui::Selectable(rows[i].label, g_selected == out || held,
                                  ImGuiSelectableFlags_SpanAllColumns |
                                      ImGuiSelectableFlags_AllowOverlap)) {
                g_selected = out;
                StartCapture(cat == Cat::Kbm               ? Capture::Kbm
                             : out.starts_with("stick_") ? Capture::PadStick
                                                         : Capture::Pad,
                             out);
            }
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(BindingText(out, cat).c_str());
            ImGui::TableSetColumnIndex(2);
            if (!out.starts_with("stick_") && ImGui::SmallButton("Quitar")) {
                g_cfg.Set(out, cat, "", {});
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void PadPage(float unit) {
    ImGui::TextWrapped("Haz clic en un botón del mando (en la foto o en la lista) y pulsa el "
                       "botón físico que quieras usar. Se guarda y se aplica al momento. Los "
                       "botones que pulses se iluminan.");
    if (ImGui::RadioButton("Mando PS", g_pad_photo == 0)) {
        g_pad_photo = 0;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Mando genérico", g_pad_photo == 1)) {
        g_pad_photo = 1;
    }
    CaptureBanner(unit);
    const float avail = ImGui::GetContentRegionAvail().x;
    const float photo_w = avail * 0.52f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    Tex& t = g_pad_photo == 0 ? g_pad_tex : g_pad_alt_tex;
    const ImTextureID tex =
        Get(t, AssetDir() / (g_pad_photo == 0 ? "mando-ps5.png" : "mando-8bitdo.png"),
            g_pad_photo == 0 ? 532.0f : 537.0f, 350.0f);
    const float scale = photo_w / t.w;
    const std::string chip = TriggerChips(at, photo_w, unit);
    const float top = at.y + ImGui::GetFrameHeightWithSpacing() + 4 * unit;
    std::string hit = g_pad_photo == 0
                          ? Photo("pad", tex, t.w, t.h, kPadSpots, std::size(kPadSpots),
                                  {at.x, top}, scale)
                          : Photo("alt", tex, t.w, t.h, kAltSpots, std::size(kAltSpots),
                                  {at.x, top}, scale);
    if (hit.empty()) {
        hit = chip;
    }
    // the sticks' "move" entries under the photo
    ImGui::SetCursorScreenPos({at.x, top + t.h * scale + 8 * unit});
    if (FancyButton("Stick izquierdo (mover)")) {
        hit = "stick_left";
    }
    ImGui::SameLine();
    if (FancyButton("Stick derecho (cámara)")) {
        hit = "stick_right";
    }
    if (FancyButton("Restablecer el mando de serie")) {
        for (const auto& r : kPadRows) {
            const std::string o = r.output;
            if (o.starts_with("stick_")) {
                continue;
            }
            g_cfg.Set(o, Cat::Pad, o, {});
        }
        for (const char* a : {"axis_left_x", "axis_left_y", "axis_right_x", "axis_right_y"}) {
            g_cfg.Set(a, Cat::Pad, a, {});
        }
    }
    if (!hit.empty()) {
        g_selected = hit;
        StartCapture(hit.starts_with("stick_") ? Capture::PadStick : Capture::Pad, hit);
    }
    ImGui::SetCursorScreenPos({at.x + photo_w + 24 * unit, at.y});
    ImGui::BeginChild("##padlist", {0, ImGui::GetContentRegionAvail().y});
    ListTable("##padtab", kPadRows, std::size(kPadRows), Cat::Pad, unit);
    ImGui::EndChild();
}

const Spot kMouseSpots[] = {
    {"leftbutton", "Clic izquierdo", 72, 72, 40, 40},
    {"rightbutton", "Clic derecho", 140, 52, 40, 40},
    {"middlebutton", "Rueda / clic central", 103, 60, 18, 18},
    {"sidebuttonforward", "Botón lateral adelante", 40, 103, 16, 16},
    {"sidebuttonback", "Botón lateral atrás", 48, 128, 16, 16},
};

void KbmPage(float unit) {
    ImGui::TextWrapped("Haz clic en un botón del mando PS4 (en la foto o en la lista) y pulsa la "
                       "tecla o el botón del ratón que quieras. Se guarda al momento.");
    CaptureBanner(unit);
    const float avail = ImGui::GetContentRegionAvail().x;
    const float photo_w = avail * 0.46f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const ImTextureID tex = Get(g_pad_tex, AssetDir() / "mando-ps5.png", 532.0f, 350.0f);
    const float scale = photo_w / g_pad_tex.w;
    std::string hit = TriggerChips(at, photo_w, unit);
    const float top = at.y + ImGui::GetFrameHeightWithSpacing() + 4 * unit;
    // what the spot is bound to, under the label
    const std::string spot = Photo("kbmpad", tex, g_pad_tex.w, g_pad_tex.h, kPadSpots,
                                   std::size(kPadSpots), {at.x, top}, scale);
    if (!spot.empty()) {
        hit = spot;
    }
    float y = top + g_pad_tex.h * scale + 10 * unit;
    // the mouse: its photo, the buttons it binds and mouse-look
    const ImTextureID mtex = Get(g_mouse_tex, AssetDir() / "raton.png", 212.0f, 220.0f);
    const float ms = (180 * unit) / g_mouse_tex.h;
    std::string mclick = Photo("mouse", mtex, g_mouse_tex.w, g_mouse_tex.h, kMouseSpots,
                               std::size(kMouseSpots), {at.x, y}, ms);
    // a mouse button clicked on the photo: pick the PS4 button it does
    static std::string s_mouse;
    if (!mclick.empty()) {
        s_mouse = mclick;
        ImGui::OpenPopup("##mousebind");
    }
    if (ImGui::BeginPopup("##mousebind")) {
        ImGui::TextColored(kGold, "%s hace:", Pretty(s_mouse).c_str());
        for (const auto& r : kKbmRows) {
            if (ImGui::Selectable(r.label, g_cfg.Uses(r.output, Cat::Kbm, s_mouse))) {
                g_cfg.Set(r.output, Cat::Kbm, s_mouse, Group(kKbmRows, std::size(kKbmRows)));
            }
        }
        if (ImGui::Selectable("(nada)")) {
            for (const auto& r : kKbmRows) {
                if (g_cfg.GetRaw(r.output, Cat::Kbm) == s_mouse) {
                    g_cfg.Set(r.output, Cat::Kbm, "", {});
                }
            }
        }
        ImGui::EndPopup();
    }
    ImGui::SetCursorScreenPos({at.x + g_mouse_tex.w * ms + 20 * unit, y});
    ImGui::BeginGroup();
    ImGui::TextColored(kGold, "Ratón");
    float dz, sp, off;
    g_cfg.MouseParams(dz, sp, off);
    ImGui::SetNextItemWidth(260 * unit);
    static float s_sp = 1.0f, s_dz = 0.5f;
    static bool s_init = false;
    if (!s_init) {
        s_sp = sp, s_dz = dz, s_init = true;
    }
    ImGui::SliderFloat("Sensibilidad de la cámara", &s_sp, 0.1f, 5.0f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        g_cfg.SetMouseParams(s_dz, s_sp, off);
    }
    ImGui::SetNextItemWidth(260 * unit);
    ImGui::SliderFloat("Antizona muerta", &s_dz, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        g_cfg.SetMouseParams(s_dz, s_sp, off);
    }
    bool right = g_cfg.MouseStick() != "left";
    if (ImGui::Checkbox("El ratón mueve el stick derecho (cámara)", &right)) {
        g_cfg.SetMouseStick(right ? "right" : "left");
    }
    ImGui::Checkbox("Cámara con ratón activa al volver al juego (F7)", &g_restore_mouse_look);
    ImGui::EndGroup();
    if (!hit.empty()) {
        g_selected = hit;
        StartCapture(Capture::Kbm, hit);
    }
    ImGui::SetCursorScreenPos({at.x + photo_w + 24 * unit, at.y});
    ImGui::BeginChild("##kbmlist", {0, ImGui::GetContentRegionAvail().y});
    ListTable("##kbmtab", kKbmRows, std::size(kKbmRows), Cat::Kbm, unit);
    ImGui::EndChild();
}

// ---------------------------------------------------------------- trophies
struct Trophy {
    int id = -1;
    char grade = 'B';
    bool hidden = false, earned = false, custom = false;
    u64 stamp = 0;
    std::string name, detail;
    Tex icon;
};
std::vector<Trophy> g_trophies;
std::string g_trophy_error;
std::filesystem::path g_icons_dir;
int g_trophy_filter = 0;
double g_trophy_page_t = 0; // when the list was shown (row stagger, bar fill)
float g_trophy_bar = 0.0f;
float g_trophy_scroll = -1.0f;

struct CustomState;
CustomState ReadCustom();
void AddCustomTrophy();
void LoadTrophies() {
    g_trophies.clear();
    g_trophy_error.clear();
    AddCustomTrophy();
    const auto& map = Common::ElfInfo::Instance().GetTrophyIndexMap();
    if (map.empty()) {
        g_trophy_error = "No se encontraron los trofeos del juego (falta npbind.dat o la clave "
                         "de trofeos en keys.json). Mira el registro: [trophy].";
        return;
    }
    const std::string np = map.begin()->second;
    const auto base = Common::FS::GetUserPath(Common::FS::PathType::TrophyDir) / np;
    g_icons_dir = base / "Icons";
    std::filesystem::path xml = base / "Xml" / "TROP_03.XML"; // Spanish
    if (!std::filesystem::exists(xml)) {
        xml = base / "Xml" / "TROP.XML";
    }
    pugi::xml_document doc;
    if (!doc.load_file(xml.native().c_str())) {
        g_trophy_error = "No se pudo leer " + xml.string() +
                         " (¿falta la clave de trofeos en keys.json?)";
        return;
    }
    for (const auto& node : doc.child("trophyconf").children("trophy")) {
        Trophy t;
        t.id = node.attribute("id").as_int(-1);
        const std::string tt = node.attribute("ttype").value();
        t.grade = tt.empty() ? 'B' : tt[0];
        t.hidden = node.attribute("hidden").as_bool();
        t.name = node.child("name").text().as_string();
        t.detail = node.child("detail").text().as_string();
        g_trophies.push_back(std::move(t));
    }
    // earned: the user's save XML (home/<user>/trophy/<np>.xml)
    std::error_code ec;
    for (const auto& dir : std::filesystem::directory_iterator(EmulatorSettings.GetHomeDir(), ec)) {
        const auto save = dir.path() / "trophy" / (np + ".xml");
        if (!std::filesystem::exists(save)) {
            continue;
        }
        pugi::xml_document sd;
        if (!sd.load_file(save.native().c_str())) {
            continue;
        }
        for (const auto& node : sd.child("trophyconf").children("trophy")) {
            const int id = node.attribute("id").as_int(-1);
            for (auto& t : g_trophies) {
                if (t.id == id && node.attribute("unlockstate").as_bool()) {
                    t.earned = true;
                    t.stamp = std::max<u64>(t.stamp, node.attribute("timestamp").as_ullong());
                }
            }
        }
        break;
    }
}

const char* GradeName(char g) {
    return g == 'P' ? "Platino" : g == 'G' ? "Oro" : g == 'S' ? "Plata" : "Bronce";
}
ImU32 GradeColor(char g, float a = 1.0f) {
    const int al = static_cast<int>(255 * a);
    switch (g) {
    case 'P': return IM_COL32(170, 200, 235, al);
    case 'G': return IM_COL32(236, 196, 72, al);
    case 'S': return IM_COL32(200, 205, 212, al);
    default: return IM_COL32(205, 127, 70, al);
    }
}
// the DS port's cup (trophy-cup.h)
void DrawCup(ImDrawList* dl, float x, float y, float s, ImU32 col, float alpha = 1.0f) {
    const float th = s * 0.07f;
    dl->AddCircle({x + s * 0.22f, y + s * 0.30f}, s * 0.13f, col, 16, th);
    dl->AddCircle({x + s * 0.78f, y + s * 0.30f}, s * 0.13f, col, 16, th);
    dl->AddRectFilled({x + s * 0.24f, y + s * 0.12f}, {x + s * 0.76f, y + s * 0.52f}, col,
                      s * 0.24f, ImDrawFlags_RoundCornersBottom);
    dl->AddRectFilled({x + s * 0.22f, y + s * 0.10f}, {x + s * 0.78f, y + s * 0.16f}, col,
                      s * 0.02f);
    dl->AddRectFilled({x + s * 0.45f, y + s * 0.50f}, {x + s * 0.55f, y + s * 0.70f}, col);
    dl->AddRectFilled({x + s * 0.34f, y + s * 0.70f}, {x + s * 0.66f, y + s * 0.78f}, col,
                      s * 0.02f);
    dl->AddRectFilled({x + s * 0.26f, y + s * 0.78f}, {x + s * 0.74f, y + s * 0.88f}, col,
                      s * 0.03f);
    dl->AddRectFilled({x + s * 0.32f, y + s * 0.18f}, {x + s * 0.38f, y + s * 0.40f},
                      IM_COL32(255, 255, 255, static_cast<int>(90 * alpha)), s * 0.03f);
}

std::string DateText(u64 stamp) {
    if (stamp == 0) {
        return "fecha desconocida";
    }
    // np_trophy writes Unix seconds; RTC ticks (µs since year 1) are also accepted
    std::time_t t = stamp > 100000000000ull
                        ? static_cast<std::time_t>((stamp - 62135596800000000ull) / 1000000ull)
                        : static_cast<std::time_t>(stamp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%d/%m/%Y %H:%M", &tm);
    return buf;
}

// the custom trophy's own icon: a pale moon over Yharnam's spires, a bronze cup in front
void DrawCustomIcon(ImDrawList* dl, float x, float y, float s, float alpha) {
    const auto A = [&](int r, int g, int b, int a) {
        return IM_COL32(r, g, b, static_cast<int>(a * std::clamp(alpha, 0.0f, 1.0f)));
    };
    dl->AddRectFilledMultiColor({x, y}, {x + s, y + s}, A(40, 22, 26, 255), A(40, 22, 26, 255),
                                A(14, 12, 16, 255), A(14, 12, 16, 255));
    dl->AddCircleFilled({x + s * 0.68f, y + s * 0.30f}, s * 0.17f, A(236, 226, 196, 235), 28);
    dl->AddCircleFilled({x + s * 0.75f, y + s * 0.26f}, s * 0.15f, A(40, 22, 26, 255), 28);
    const float bx[] = {0.08f, 0.20f, 0.30f, 0.80f, 0.90f};
    const float bh[] = {0.42f, 0.30f, 0.50f, 0.38f, 0.46f};
    for (int i = 0; i < 5; i++) {
        const float cx = x + s * bx[i], top = y + s * bh[i];
        dl->AddTriangleFilled({cx - s * 0.05f, top + s * 0.12f}, {cx + s * 0.05f, top + s * 0.12f},
                              {cx, top}, A(8, 8, 10, 255));
        dl->AddRectFilled({cx - s * 0.05f, top + s * 0.12f}, {cx + s * 0.05f, y + s}, A(8, 8, 10, 255));
    }
    DrawCup(dl, x + s * 0.27f, y + s * 0.40f, s * 0.48f, A(205, 127, 70, 255), alpha);
    dl->AddRect({x, y}, {x + s, y + s}, A(236, 200, 120, 120), s * 0.1f, 0, 1.0f);
}

void TrophyPage(float unit) {
    if (!g_trophy_error.empty()) {
        ImGui::TextColored(kWarn, "%s", g_trophy_error.c_str());
        if (FancyButton("Reintentar")) {
            LoadTrophies();
        }
    }
    int earned = 0;
    for (const auto& t : g_trophies) {
        earned += t.earned;
    }
    const int total = static_cast<int>(g_trophies.size());
    const float pct = total > 0 ? 100.0f * earned / total : 0.0f;
    ImGui::TextColored(kGold, "%d de %d conseguidos", earned, total);
    ImGui::SameLine();
    ImGui::TextColored(kDim, "(%.0f%%)", pct);
    {
        // eased fill, gold gradient with a moving sheen
        const float goal = total > 0 ? static_cast<float>(earned) / total : 0.0f;
        g_trophy_bar += (goal - g_trophy_bar) * std::min(1.0f, ImGui::GetIO().DeltaTime * 4.0f);
        const ImVec2 a = ImGui::GetCursorScreenPos();
        const float bw = ImGui::GetContentRegionAvail().x, bh = 12 * unit;
        ImGui::Dummy({bw, bh});
        auto* dl = ImGui::GetWindowDrawList();
        const float al = ImGui::GetStyle().Alpha;
        dl->AddRectFilled(a, {a.x + bw, a.y + bh}, IM_COL32(20, 18, 16, static_cast<int>(230 * al)),
                          bh * 0.5f);
        const float fw = bw * g_trophy_bar;
        if (fw > 1.0f) {
            dl->AddRectFilledMultiColor(a, {a.x + fw, a.y + bh},
                                        IM_COL32(140, 86, 40, static_cast<int>(255 * al)),
                                        IM_COL32(236, 196, 110, static_cast<int>(255 * al)),
                                        IM_COL32(236, 196, 110, static_cast<int>(255 * al)),
                                        IM_COL32(140, 86, 40, static_cast<int>(255 * al)));
            const float sx = a.x + std::fmod(static_cast<float>(ImGui::GetTime()) * 0.35f, 1.4f) * bw - 0.2f * bw;
            dl->PushClipRect(a, {a.x + fw, a.y + bh}, true);
            dl->AddRectFilledMultiColor({sx, a.y}, {sx + 0.1f * bw, a.y + bh}, 0,
                                        IM_COL32(255, 255, 255, static_cast<int>(70 * al)),
                                        IM_COL32(255, 255, 255, static_cast<int>(70 * al)), 0);
            dl->PopClipRect();
        }
        dl->AddRect(a, {a.x + bw, a.y + bh}, IM_COL32(236, 200, 120, static_cast<int>(90 * al)),
                    bh * 0.5f);
    }
    const char* filters[] = {"Todos", "Conseguidos", "Por conseguir"};
    for (int i = 0; i < 3; i++) {
        if (i > 0) {
            ImGui::SameLine();
        }
        if (ImGui::RadioButton(filters[i], g_trophy_filter == i)) {
            g_trophy_filter = i;
        }
    }
    ImGui::SameLine();
    if (FancyButton("Actualizar")) {
        LoadTrophies();
    }
    ImGui::BeginChild("##trophies", {0, 0}, ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollWithMouse);
    SmoothScroll(g_trophy_scroll, unit);
    const float icon = 72 * unit;
    int row = 0;
    const float base_alpha = ImGui::GetStyle().Alpha;
    for (auto& t : g_trophies) {
        if ((g_trophy_filter == 1 && !t.earned) || (g_trophy_filter == 2 && t.earned)) {
            continue;
        }
        // staggered fade/slide in
        const float ra = EaseOut(static_cast<float>(ImGui::GetTime() - g_trophy_page_t -
                                                    std::min(row, 16) * 0.035) / 0.28f);
        row++;
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, base_alpha * ra);
        ImGui::PushID(t.id);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (1.0f - ra) * 40 * unit);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        const float h = icon + 16 * unit;
        // a focusable row (gamepad scroll)
        ImGui::Selectable("##row", false, 0, {w, h});
        const bool hov = ImGui::IsItemHovered() || ImGui::IsItemFocused();
        auto* dl0 = ImGui::GetWindowDrawList();
        // the rows' colours fade with ra (draw-list colours ignore the style alpha)
        struct FadeDl {
            ImDrawList* d;
            float a;
            void AddRectFilled(ImVec2 x, ImVec2 y, ImU32 c, float r = 0, ImDrawFlags f = 0) {
                d->AddRectFilled(x, y, Fade(c, a), r, f);
            }
            void AddRect(ImVec2 x, ImVec2 y, ImU32 c, float r = 0, ImDrawFlags f = 0, float th = 1) {
                d->AddRect(x, y, Fade(c, a), r, f, th);
            }
            void AddImage(ImTextureID t, ImVec2 x, ImVec2 y, ImVec2 u0, ImVec2 u1, ImU32 c) {
                d->AddImage(t, x, y, u0, u1, Fade(c, a));
            }
            void AddText(ImVec2 x, ImU32 c, const char* s) { d->AddText(x, Fade(c, a), s); }
            void AddText(ImFont* f, float fs, ImVec2 x, ImU32 c, const char* s, const char* e,
                         float ww) {
                d->AddText(f, fs, x, Fade(c, a), s, e, ww);
            }
        } dlf{dl0, ra};
        auto* dl = &dlf;
        dl->AddRectFilled({p.x + 2 * unit, p.y + 4 * unit}, {p.x + w + 2 * unit, p.y + h + 4 * unit},
                          IM_COL32(0, 0, 0, 90), 10 * unit);
        dl->AddRectFilled(p, {p.x + w, p.y + h},
                          t.earned ? IM_COL32(46, 40, 30, 225) : IM_COL32(26, 25, 28, 225),
                          10 * unit);
        if (t.earned) {
            dl->AddRectFilled(p, {p.x + 4 * unit, p.y + h}, GradeColor(t.grade), 10 * unit,
                              ImDrawFlags_RoundCornersLeft);
        }
        dl->AddRect(p, {p.x + w, p.y + h},
                    hov ? IM_COL32(236, 200, 120, 170) : IM_COL32(236, 200, 120, 28), 10 * unit, 0,
                    hov ? 2.0f : 1.0f);
        const ImVec2 ip{p.x + 8 * unit, p.y + 8 * unit};
        ImTextureID tex{};
        if (!t.custom && ImGui::IsRectVisible(p, {p.x + w, p.y + h})) {
            tex = Get(t.icon, g_icons_dir / fmt::format("TROP{:03d}.PNG", t.id), 240, 240);
        }
        if (tex) {
            dl->AddImage(tex, ip, {ip.x + icon, ip.y + icon}, {0, 0}, {1, 1},
                         t.earned ? IM_COL32_WHITE : IM_COL32(110, 110, 110, 255));
        } else if (t.custom) {
            DrawCustomIcon(dl0, ip.x, ip.y, icon, (t.earned ? 1.0f : 0.45f) * ra);
        } else {
            dl->AddRectFilled(ip, {ip.x + icon, ip.y + icon}, IM_COL32(44, 46, 54, 255),
                              8 * unit);
            DrawCup(dl0, ip.x + icon * 0.1f, ip.y + icon * 0.08f, icon * 0.8f,
                    GradeColor(t.grade, (t.earned ? 1.0f : 0.4f) * ra), ra);
        }
        const float tx = ip.x + icon + 14 * unit;
        const float fs = ImGui::GetFontSize();
        DrawCup(dl0, tx, p.y + 8 * unit, fs, GradeColor(t.grade, ra), ra);
        dl->AddText({tx + fs * 1.2f, p.y + 8 * unit}, GradeColor(t.grade), GradeName(t.grade));
        const std::string nm = t.custom ? t.name + "  (del port)"
                               : t.hidden && !t.earned ? t.name + "  (oculto)"
                                                       : t.name;
        dl->AddText({tx + fs * 6.0f, p.y + 8 * unit}, IM_COL32(245, 245, 247, 255), nm.c_str());
        dl->AddText(nullptr, fs, ImVec2{tx, p.y + 8 * unit + fs * 1.3f}, IM_COL32(190, 192, 200, 255),
                    t.detail.c_str(), nullptr, w - (tx - p.x) - 300 * unit);
        const std::string state =
            t.earned ? "Conseguido · " + DateText(t.stamp) : std::string("Por conseguir");
        const float sw = ImGui::CalcTextSize(state.c_str()).x;
        dl->AddText({p.x + w - sw - 14 * unit, p.y + 8 * unit},
                    t.earned ? IM_COL32(150, 220, 140, 255) : IM_COL32(160, 160, 165, 255),
                    state.c_str());
        ImGui::Dummy({0, 6 * unit});
        ImGui::PopID();
        ImGui::PopStyleVar();
    }
    ImGui::EndChild();
}

// ---------------------------------------------------------------- toast (DS popup)
struct Toast {
    std::filesystem::path icon_path;
    std::string name;
    char grade = 'B';
    double start = -1;
    Tex icon;
    std::string detail;
    bool custom = false;
};
std::mutex g_toast_mtx;
std::deque<Toast> g_toasts;
constexpr float kToastSeconds = 6.0f;

void PlayChime() {
    // the DS port's chime (our own: G5, D6, G6 bell partials), through SDL3 audio
    static std::mutex m;
    static SDL_AudioStream* stream = nullptr;
    std::scoped_lock lk{m};
    if (stream == nullptr) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            return;
        }
        SDL_AudioSpec spec{SDL_AUDIO_F32, 2, 48000};
        stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr,
                                           nullptr);
        if (stream == nullptr) {
            return;
        }
        SDL_ResumeAudioStreamDevice(stream);
    }
    constexpr int rate = 48000;
    constexpr double pi = 3.14159265358979323846;
    const int n = rate * 9 / 10;
    std::vector<float> buf(size_t(n) * 2);
    struct Note {
        double start, freq, amp;
    };
    const Note notes[] = {{0.00, 784.0, 0.22}, {0.11, 1174.7, 0.20}, {0.22, 1568.0, 0.13}};
    for (int i = 0; i < n; i++) {
        const double t = double(i) / rate;
        double v = 0;
        for (const auto& nt : notes) {
            const double u = t - nt.start;
            if (u < 0) {
                continue;
            }
            const double env = std::min(1.0, u / 0.006) * std::exp(-u * 5.5);
            v += nt.amp * env *
                 (std::sin(2 * pi * nt.freq * u) + 0.25 * std::sin(2 * pi * nt.freq * 2.01 * u) +
                  0.08 * std::sin(2 * pi * nt.freq * 3.0 * u));
        }
        const double fade = i > n - 2400 ? double(n - i) / 2400.0 : 1.0;
        buf[size_t(i) * 2] = buf[size_t(i) * 2 + 1] = static_cast<float>(v * fade * 0.6);
    }
    SDL_PutAudioStreamData(stream, buf.data(), static_cast<int>(buf.size() * sizeof(float)));
}

void DrawToast(float unit) {
    std::scoped_lock lk{g_toast_mtx};
    if (g_toasts.empty()) {
        return;
    }
    Toast& toast = g_toasts.front();
    const double now = ImGui::GetTime();
    if (toast.start < 0) {
        toast.start = now;
        std::thread(PlayChime).detach();
    }
    const float t = static_cast<float>(now - toast.start);
    if (t > kToastSeconds) {
        g_toasts.pop_front();
        return;
    }
    const float slide = 0.35f, fade = 0.6f;
    const float in = std::clamp(t / slide, 0.0f, 1.0f);
    const float ease = 1.0f - (1.0f - in) * (1.0f - in) * (1.0f - in);
    const float alpha = std::clamp((kToastSeconds - t) / fade, 0.0f, 1.0f) * std::min(1.0f, in * 2.0f);
    if (alpha <= 0.0f) {
        return;
    }
    auto* draw = ImGui::GetForegroundDrawList();
    ImFont* font = ImGui::GetFont();
    const float big = 26.0f * unit, small = 19.0f * unit;
    const float icon = 76.0f * unit, pad = 14.0f * unit;
    const std::string label = "Has obtenido un trofeo";
    const std::string grade_text = GradeName(toast.grade);
    const std::string top = grade_text + "  \xc2\xb7  " + label;
    const float text_w =
        std::max(font->CalcTextSizeA(big, FLT_MAX, 0.0f, toast.name.c_str()).x,
                 font->CalcTextSizeA(small, FLT_MAX, 0.0f, top.c_str()).x + 18.0f * unit);
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    const float w = std::min(pad * 3.0f + icon + text_w + 8.0f * unit, disp.x * 0.45f);
    const float h = icon + pad * 2.0f + (toast.detail.empty() ? 0.0f : small + 4.0f * unit);
    const float margin = 28.0f * unit;
    const float x1_end = disp.x - margin;
    const float x0 = x1_end - w + (1.0f - ease) * (w + margin + 4.0f);
    const float y0 = 20.0f * unit;
    const auto a = [&](int v) { return static_cast<int>(static_cast<float>(v) * alpha); };
    const float round = 12.0f * unit;
    draw->AddRectFilled({x0 + 3.0f * unit, y0 + 4.0f * unit},
                        {x0 + w + 3.0f * unit, y0 + h + 4.0f * unit}, IM_COL32(0, 0, 0, a(70)),
                        round);
    draw->AddRectFilled({x0, y0}, {x0 + w, y0 + h}, IM_COL32(28, 29, 33, a(235)), round);
    draw->AddRect({x0, y0}, {x0 + w, y0 + h}, IM_COL32(255, 255, 255, a(22)), round, 0, unit);
    const float ix = x0 + pad, iy = y0 + pad;
    const ImU32 grade = GradeColor(toast.grade, alpha);
    // the icon's animation: a scale pop with overshoot as the card lands, a glow pulse in
    // the grade's colour, then a shine sweep across it
    const float pt = std::clamp((t - 0.20f) / 0.45f, 0.0f, 1.0f);
    const float c1 = 1.70158f, c3 = c1 + 1.0f;
    const float pop = t < 0.20f ? 0.6f
                                : 0.6f + 0.4f * (1.0f + c3 * std::pow(pt - 1.0f, 3.0f) +
                                                 c1 * std::pow(pt - 1.0f, 2.0f));
    const float cx = ix + icon * 0.5f, cy = iy + icon * 0.5f;
    const float gl = t > 0.3f && t < 3.0f
                         ? (0.5f + 0.5f * std::sin((t - 0.3f) * 6.0f)) *
                               std::clamp((3.0f - t) / 0.8f, 0.0f, 1.0f)
                         : 0.0f;
    for (int i = 4; i >= 1; i--) {
        const float g = icon * 0.5f + static_cast<float>(i) * 4.0f * unit;
        draw->AddRectFilled({cx - g, cy - g}, {cx + g, cy + g},
                            GradeColor(toast.grade, alpha * gl * 0.10f), 12.0f * unit + i * 4.0f * unit);
    }
    const float hs = icon * 0.5f * pop;
    const ImVec2 q0{cx - hs, cy - hs}, q1{cx + hs, cy + hs};
    const ImTextureID tex = toast.custom ? ImTextureID{} : Get(toast.icon, toast.icon_path, 240, 240);
    if (toast.custom) {
        DrawCustomIcon(draw, q0.x, q0.y, hs * 2.0f, alpha);
    } else if (tex) {
        draw->AddImage(tex, q0, q1, {0, 0}, {1, 1}, IM_COL32(255, 255, 255, a(255)));
    } else {
        draw->AddRectFilled(q0, q1, IM_COL32(44, 46, 54, a(255)), 10.0f * unit);
        DrawCup(draw, q0.x + hs * 0.2f, q0.y + hs * 0.16f, hs * 1.6f, grade, alpha);
    }
    if (t > 0.65f && t < 1.45f) {
        const float st = (t - 0.65f) / 0.8f;
        const float sx = q0.x - hs + st * hs * 4.0f;
        draw->PushClipRect(q0, q1, true);
        const ImU32 cw = IM_COL32(255, 248, 225, a(150));
        draw->AddQuadFilled({sx, q0.y}, {sx + hs * 0.35f, q0.y}, {sx - hs * 0.25f, q1.y},
                            {sx - hs * 0.6f, q1.y}, cw);
        draw->PopClipRect();
    }
    const float tx = ix + icon + pad, ty = iy + 6.0f * unit;
    float lx = tx;
    draw->AddText(font, small, {lx, ty}, grade, grade_text.c_str());
    lx += font->CalcTextSizeA(small, FLT_MAX, 0.0f, grade_text.c_str()).x;
    draw->AddText(font, small, {lx, ty}, IM_COL32(170, 172, 180, a(255)), "  \xc2\xb7  ");
    lx += font->CalcTextSizeA(small, FLT_MAX, 0.0f, "  \xc2\xb7  ").x;
    draw->AddText(font, small, {lx, ty}, IM_COL32(170, 172, 180, a(255)), label.c_str());
    const ImVec4 clip{x0, y0, x0 + w - pad, y0 + h};
    draw->AddText(font, big, {tx, ty + small + 10.0f * unit}, IM_COL32(245, 245, 247, a(255)),
                  toast.name.c_str(), nullptr, 0.0f, &clip);
    if (!toast.detail.empty()) {
        draw->AddText(font, small, {tx, ty + small + big + 16.0f * unit},
                      IM_COL32(190, 192, 200, a(255)), toast.detail.c_str(), nullptr, 0.0f, &clip);
    }
}

// ---------------------------------------------------------------- custom trophy (not PSN)
// "Despertar en Yharnam": leaving Iosefka's Clinic for the first time. No verified guest
// memory map/event-flag offsets for CUSA03173 are available here, so the trigger uses the
// save data: a new save file/slot appearing (a new character is created in the clinic), then
// kCustomPlaySeconds of play time (persisted across runs) — about the time it takes to wake
// up, go through the clinic and walk out into Central Yharnam.
constexpr const char* kCustomName = "Despertar en Yharnam";
constexpr const char* kCustomDetail =
    "Sal de la Clínica de Iosefka por primera vez y comienza la cacería.";
constexpr int kCustomPlaySeconds = 420;
struct CustomState {
    bool armed = false, unlocked = false;
    int seconds = 0;
    u64 stamp = 0;
};
std::mutex g_custom_mtx;
std::filesystem::path CustomJson() {
    return Common::FS::GetUserPath(Common::FS::PathType::TrophyDir) / "custom_trophies.json";
}
CustomState ReadCustom() {
    CustomState s;
    std::ifstream f(CustomJson(), std::ios::binary);
    if (!f) {
        return s;
    }
    std::string j((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto num = [&](const char* k) -> u64 {
        const auto p = j.find(k);
        if (p == std::string::npos) {
            return 0;
        }
        const auto c = j.find(':', p);
        return c == std::string::npos ? 0 : std::strtoull(j.c_str() + c + 1, nullptr, 10);
    };
    const auto flag = [&](const char* k) {
        const auto p = j.find(k);
        if (p == std::string::npos) {
            return false;
        }
        const auto c = j.find(':', p);
        return c != std::string::npos && j.find("true", c) == j.find_first_not_of(" \t", c + 1);
    };
    s.unlocked = flag("\"unlocked\"");
    s.armed = flag("\"armed\"");
    s.seconds = static_cast<int>(num("\"play_seconds\""));
    s.stamp = num("\"timestamp\"");
    return s;
}
void WriteCustom(const CustomState& s) {
    std::scoped_lock lk{g_custom_mtx};
    std::error_code ec;
    const auto p = CustomJson();
    std::filesystem::create_directories(p.parent_path(), ec);
    std::string date;
    if (s.stamp != 0) {
        std::time_t t = static_cast<std::time_t>(s.stamp);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
        date = buf;
    }
    const std::string j = fmt::format(
        "{{\n  \"despertar_en_yharnam\": {{\n    \"name\": \"{}\",\n    \"grade\": \"B\",\n"
        "    \"unlocked\": {},\n    \"timestamp\": {},\n    \"date\": \"{}\",\n"
        "    \"armed\": {},\n    \"play_seconds\": {}\n  }}\n}}\n",
        kCustomName, s.unlocked ? "true" : "false", s.stamp, date, s.armed ? "true" : "false",
        s.seconds);
    const auto tmp = std::filesystem::path(p).concat(".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << j;
    }
    std::filesystem::rename(tmp, p, ec);
}
std::atomic<bool> g_custom_changed{false};
// every file under the save data folders (home/<user>/savedata/...), without sce_sys
std::vector<std::string> ScanSaves() {
    std::vector<std::string> out;
    std::error_code ec;
    const auto home = EmulatorSettings.GetHomeDir();
    for (auto it = std::filesystem::recursive_directory_iterator(
             home, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (it.depth() > 6) {
            it.disable_recursion_pending();
            continue;
        }
        const auto name = it->path().filename().string();
        if (it->is_directory(ec)) {
            if (name == "sce_sys" || name == "trophy") {
                it.disable_recursion_pending();
            }
            continue;
        }
        const std::string s = it->path().generic_string();
        if (s.find("/savedata/") != std::string::npos) {
            out.push_back(s);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}
void UnlockCustom(CustomState& s);
void CustomWatcher() {
    CustomState s = ReadCustom();
    if (s.unlocked) {
        return;
    }
    if (const char* v = std::getenv("BB_CUSTOM_TROPHY_TEST"); v != nullptr && v[0] == '1') {
        std::this_thread::sleep_for(std::chrono::seconds(20));
        UnlockCustom(s);
        return;
    }
    auto known = ScanSaves();
    LOG_INFO(Frontend, "[bb-trophy] custom '{}': watching {} save files (armed {}, {} s)",
             kCustomName, known.size(), s.armed, s.seconds);
    int since_write = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (!s.armed) {
            const auto now = ScanSaves();
            std::vector<std::string> added;
            std::set_difference(now.begin(), now.end(), known.begin(), known.end(),
                                std::back_inserter(added));
            if (!added.empty()) {
                s.armed = true;
                s.seconds = 0;
                WriteCustom(s);
                LOG_INFO(Frontend, "[bb-trophy] custom: new save data ({}): armed", added[0]);
            }
            known = now;
            continue;
        }
        s.seconds += 5;
        since_write += 5;
        if (s.seconds >= kCustomPlaySeconds) {
            UnlockCustom(s);
            return;
        }
        if (since_write >= 30) {
            since_write = 0;
            WriteCustom(s);
        }
    }
}

void UnlockCustom(CustomState& s) {
    s.unlocked = true;
    s.stamp = static_cast<u64>(std::time(nullptr));
    WriteCustom(s);
    LOG_INFO(Frontend, "[bb-trophy] custom trophy unlocked: {}", kCustomName);
    g_custom_changed = true;
    if (const char* v = std::getenv("BB_TROPHY_POPUP"); v != nullptr && v[0] == '0') {
        return;
    }
    std::scoped_lock lk{g_toast_mtx};
    Toast t;
    t.name = kCustomName;
    t.grade = 'B';
    t.detail = kCustomDetail;
    t.custom = true;
    g_toasts.push_back(std::move(t));
}
void AddCustomTrophy() {
    const CustomState s = ReadCustom();
    Trophy t;
    t.id = -100;
    t.grade = 'B';
    t.custom = true;
    t.earned = s.unlocked;
    t.stamp = s.stamp;
    t.name = kCustomName;
    t.detail = kCustomDetail;
    g_trophies.push_back(std::move(t));
}

// ---------------------------------------------------------------- Opciones: launcher.ini
// <exe dir>/launcher.ini, plain key=value lines; we only touch inicio_directo=0|1 and keep
// every other byte (encoding, BOM, line endings) as it was. Written atomically.
std::filesystem::path LauncherIni() {
    const char* base = SDL_GetBasePath();
    std::filesystem::path p = base != nullptr ? std::filesystem::path(base) : ".";
    return p / "launcher.ini";
}
struct IniText {
    int unit = 1;          // 1 = UTF-8/ANSI bytes, 2 = UTF-16LE, 3 = UTF-16BE
    std::string bom;       // raw BOM bytes, kept as they were
    std::vector<u32> cu;   // code units after the BOM
};
bool ReadIni(const std::filesystem::path& p, IniText& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        return false;
    }
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto at = [&](size_t i) { return static_cast<unsigned char>(b[i]); };
    size_t s = 0;
    if (b.size() >= 2 && at(0) == 0xFF && at(1) == 0xFE) {
        out.unit = 2, s = 2;
    } else if (b.size() >= 2 && at(0) == 0xFE && at(1) == 0xFF) {
        out.unit = 3, s = 2;
    } else if (b.size() >= 3 && at(0) == 0xEF && at(1) == 0xBB && at(2) == 0xBF) {
        s = 3;
    }
    out.bom = b.substr(0, s);
    out.cu.clear();
    if (out.unit == 1) {
        for (size_t i = s; i < b.size(); i++) {
            out.cu.push_back(at(i));
        }
    } else {
        for (size_t i = s; i + 1 < b.size(); i += 2) {
            out.cu.push_back(out.unit == 2 ? (at(i) | (at(i + 1) << 8)) : ((at(i) << 8) | at(i + 1)));
        }
        if ((b.size() - s) % 2 != 0) {
            out.cu.push_back(0x10000u | at(b.size() - 1)); // stray byte: kept raw
        }
    }
    return true;
}
std::string IniBytes(const IniText& t) {
    std::string b = t.bom;
    for (u32 c : t.cu) {
        if (c & 0x10000u) {
            b.push_back(static_cast<char>(c & 0xFF));
        } else if (t.unit == 1) {
            b.push_back(static_cast<char>(c));
        } else if (t.unit == 2) {
            b.push_back(static_cast<char>(c & 0xFF)), b.push_back(static_cast<char>(c >> 8));
        } else {
            b.push_back(static_cast<char>(c >> 8)), b.push_back(static_cast<char>(c & 0xFF));
        }
    }
    return b;
}
// the line [begin, end) (end before its terminator) of the key, or false
bool FindIniKey(const IniText& t, const char* key, size_t& lb, size_t& le) {
    const size_t n = t.cu.size();
    size_t i = 0;
    while (i <= n) {
        size_t e = i;
        while (e < n && t.cu[e] != '\n' && t.cu[e] != '\r') {
            e++;
        }
        size_t k = i;
        while (k < e && (t.cu[k] == ' ' || t.cu[k] == '\t')) {
            k++;
        }
        size_t q = k;
        bool ok = true;
        for (const char* c = key; *c; c++, q++) {
            if (q >= e || std::tolower(static_cast<int>(t.cu[q] & 0x7F)) != *c || t.cu[q] > 0x7F) {
                ok = false;
                break;
            }
        }
        if (ok) {
            while (q < e && (t.cu[q] == ' ' || t.cu[q] == '\t')) {
                q++;
            }
            if (q < e && t.cu[q] == '=') {
                lb = i, le = e;
                return true;
            }
        }
        if (e >= n) {
            break;
        }
        i = e + ((t.cu[e] == '\r' && e + 1 < n && t.cu[e + 1] == '\n') ? 2 : 1);
    }
    return false;
}
struct DirectState {
    bool exists = false;
    bool on = false;
    std::string error;
} g_direct;
void ReadDirect() {
    g_direct = {};
    IniText t;
    const auto p = LauncherIni();
    if (!std::filesystem::exists(p) || !ReadIni(p, t)) {
        return;
    }
    g_direct.exists = true;
    size_t lb, le;
    if (FindIniKey(t, "inicio_directo", lb, le)) {
        size_t q = lb;
        while (q < le && t.cu[q] != '=') {
            q++;
        }
        q++;
        while (q < le && (t.cu[q] == ' ' || t.cu[q] == '\t')) {
            q++;
        }
        g_direct.on = q < le && t.cu[q] == '1';
    }
}
bool WriteDirect(bool on) {
    const auto p = LauncherIni();
    IniText t;
    if (!ReadIni(p, t)) {
        g_direct.error = "No se pudo leer launcher.ini";
        return false;
    }
    const std::string line = std::string("inicio_directo=") + (on ? "1" : "0");
    std::vector<u32> nl(line.begin(), line.end());
    size_t lb, le;
    if (FindIniKey(t, "inicio_directo", lb, le)) {
        t.cu.erase(t.cu.begin() + lb, t.cu.begin() + le);
        t.cu.insert(t.cu.begin() + lb, nl.begin(), nl.end());
    } else {
        // the file's own line ending (CRLF if it has any)
        bool crlf = false, cr_only = false;
        for (size_t i = 0; i < t.cu.size(); i++) {
            if (t.cu[i] == '\r') {
                if (i + 1 < t.cu.size() && t.cu[i + 1] == '\n') {
                    crlf = true;
                } else {
                    cr_only = true;
                }
                break;
            }
            if (t.cu[i] == '\n') {
                break;
            }
        }
        std::vector<u32> eol = crlf ? std::vector<u32>{'\r', '\n'}
                               : cr_only ? std::vector<u32>{'\r'}
                                         : std::vector<u32>{'\n'};
#ifdef _WIN32
        if (t.cu.empty()) {
            eol = {'\r', '\n'};
        }
#endif
        if (!t.cu.empty() && t.cu.back() != '\n' && t.cu.back() != '\r') {
            t.cu.insert(t.cu.end(), eol.begin(), eol.end());
        }
        t.cu.insert(t.cu.end(), nl.begin(), nl.end());
        t.cu.insert(t.cu.end(), eol.begin(), eol.end());
    }
    const auto tmp = std::filesystem::path(p).concat(".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        const std::string b = IniBytes(t);
        f.write(b.data(), static_cast<std::streamsize>(b.size()));
        f.flush();
        if (!f) {
            g_direct.error = "No se pudo escribir launcher.ini";
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, p, ec); // MoveFileEx(REPLACE_EXISTING) on Windows
    if (ec) {
        g_direct.error = "No se pudo reemplazar launcher.ini: " + ec.message();
        std::filesystem::remove(tmp, ec);
        return false;
    }
    g_direct.error.clear();
    LOG_INFO(Frontend, "[bb-menu] launcher.ini: inicio_directo={}", on ? 1 : 0);
    return true;
}


// ---------------------------------------------------------------- the layer
constexpr int kTabs = 5;
class MenuLayer final : public ImGui::Layer {
public:
    int tab = 0;
    int want_tab = -1;
    bool was_open = false;
    bool confirm_exit = false;
    float anim = 0.0f;       // open/close 0..1
    double tab_t = 0;        // tab switch time (content cross-fade)
    float ul_x0 = 0, ul_x1 = 0; // sliding tab underline
    bool ul_init = false;

    void Draw() override {
        const ImVec2 disp = ImGui::GetIO().DisplaySize;
        const float unit = std::max(disp.y / 1440.0f, 0.5f);
        DrawToast(std::max(disp.y / 1080.0f, 0.5f));
        const float dt = ImGui::GetIO().DeltaTime;
        anim = std::clamp(anim + (g_open ? dt : -dt) / 0.20f, 0.0f, 1.0f);
        if (!g_open && anim <= 0.0f) {
            was_open = false;
            return;
        }
        const bool closing = !g_open;
        if (g_open && !was_open) {
            was_open = true;
            g_cfg.Load();
            LoadTrophies();
            ReadDirect();
            confirm_exit = false;
            tab_t = ImGui::GetTime();
            g_trophy_page_t = tab_t;
            g_trophy_bar = 0.0f;
            ul_init = false;
            ImGui::SetNextWindowFocus();
        }
        if (g_custom_changed.exchange(false)) {
            LoadTrophies();
        }
        const float e = g_open ? EaseOut(anim) : EaseInOut(anim);
        if (!closing) {
            ApplyCapture();
        }
        bool capturing;
        {
            std::scoped_lock lk{g_cap_mtx};
            capturing = g_capture != Capture::None;
        }
        // dim the game (fades with the menu) and a vignette in place of a blur
        auto* bg = ImGui::GetBackgroundDrawList();
        bg->AddRectFilled({0, 0}, disp, IM_COL32(4, 3, 6, static_cast<int>(165 * e)));
        const ImU32 vin = IM_COL32(0, 0, 0, static_cast<int>(120 * e)), clr = IM_COL32(0, 0, 0, 0);
        bg->AddRectFilledMultiColor({0, 0}, {disp.x, disp.y * 0.25f}, vin, vin, clr, clr);
        bg->AddRectFilledMultiColor({0, disp.y * 0.75f}, disp, clr, clr, vin, vin);
        const float sc = 0.94f + 0.06f * e;
        const ImVec2 size{disp.x * 0.86f * sc, disp.y * 0.86f * sc};
        const ImVec2 center{disp.x * 0.5f, disp.y * 0.5f + (1.0f - e) * 40.0f * unit};
        // soft shadow under the panel
        for (int i = 1; i <= 4; i++) {
            const float g = static_cast<float>(i) * 10.0f * unit;
            bg->AddRectFilled({center.x - size.x * 0.5f - g, center.y - size.y * 0.5f - g + 12 * unit},
                              {center.x + size.x * 0.5f + g, center.y + size.y * 0.5f + g + 12 * unit},
                              IM_COL32(0, 0, 0, static_cast<int>(28 * e)), 14 * unit + g);
        }
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, {0.5f, 0.5f});
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, e);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 14 * unit);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {28 * unit, 22 * unit});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {14 * unit, 8 * unit});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {14 * unit, 10 * unit});
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8 * unit);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.055f, 0.05f, 0.06f, 0.97f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.93f, 0.78f, 0.47f, 0.35f));
        ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(0.16f, 0.15f, 0.14f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TabSelected, ImVec4(0.45f, 0.36f, 0.20f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(0.60f, 0.48f, 0.27f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.20f, 0.17f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.50f, 0.40f, 0.22f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.45f, 0.36f, 0.20f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.55f, 0.44f, 0.24f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_NavCursor, kGold);
        if (ImGui::Begin("Menú (F1)##bbmenu", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
                             (closing ? ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav : 0))) {
            {
                // gothic trim: a gold rule under the title and corner filigree
                auto* wd = ImGui::GetWindowDrawList();
                const ImVec2 wp = ImGui::GetWindowPos(), ws = ImGui::GetWindowSize();
                const ImU32 gold = IM_COL32(236, 200, 120, static_cast<int>(150 * e));
                const float c = 26 * unit, o = 8 * unit;
                const ImVec2 cs[4] = {{wp.x + o, wp.y + o}, {wp.x + ws.x - o, wp.y + o},
                                      {wp.x + o, wp.y + ws.y - o}, {wp.x + ws.x - o, wp.y + ws.y - o}};
                for (int k = 0; k < 4; k++) {
                    const float sx = (k & 1) ? -1.0f : 1.0f, sy = (k & 2) ? -1.0f : 1.0f;
                    wd->AddLine(cs[k], {cs[k].x + sx * c, cs[k].y}, gold, 1.5f);
                    wd->AddLine(cs[k], {cs[k].x, cs[k].y + sy * c}, gold, 1.5f);
                    wd->AddCircleFilled({cs[k].x + sx * 5 * unit, cs[k].y + sy * 5 * unit}, 2.0f * unit, gold);
                }
            }
            ImGui::SetWindowFontScale(std::max(1.0f, disp.y / 1440.0f * 1.6f));
            ImGui::TextColored(kGold, "BLOODBORNE  ·  MENÚ");
            ImGui::SameLine();
            ImGui::TextColored(kDim,
                               "   F1 / ○: volver al juego    L1 / R1: cambiar de pestaña");
            // L1 / R1 change tabs (not while rebinding)
            if (!capturing && !confirm_exit && !closing) {
                if (ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false)) {
                    want_tab = (tab + kTabs - 1) % kTabs;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false)) {
                    want_tab = (tab + 1) % kTabs;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false) ||
                    ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                    g_want_close = true;
                    Wake();
                }
            }
            if (ImGui::BeginTabBar("##tabs")) {
                const char* names[kTabs] = {"Mando", "Teclado y ratón", "Trofeos", "Opciones",
                                            "Salir del juego"};
                float sel_x0 = ul_x0, sel_x1 = ul_x1, sel_y = 0;
                for (int i = 0; i < kTabs; i++) {
                    ImGuiTabItemFlags f = want_tab == i ? ImGuiTabItemFlags_SetSelected : 0;
                    const bool open_tab = ImGui::BeginTabItem(names[i], nullptr, f);
                    if (open_tab) {
                        sel_x0 = ImGui::GetItemRectMin().x;
                        sel_x1 = ImGui::GetItemRectMax().x;
                        sel_y = ImGui::GetItemRectMax().y;
                        if (tab != i) {
                            g_selected.clear();
                            std::scoped_lock lk{g_cap_mtx};
                            g_capture = Capture::None;
                            tab_t = ImGui::GetTime();
                            if (i == 2) {
                                g_trophy_page_t = tab_t;
                                g_trophy_bar = 0.0f;
                            }
                        }
                        tab = i;
                        // sliding gold underline
                        if (!ul_init) {
                            ul_x0 = sel_x0, ul_x1 = sel_x1, ul_init = true;
                        }
                        const float k = std::min(1.0f, dt * 14.0f);
                        ul_x0 += (sel_x0 - ul_x0) * k;
                        ul_x1 += (sel_x1 - ul_x1) * k;
                        auto* wd = ImGui::GetWindowDrawList();
                        wd->AddRectFilled({ul_x0 - 4 * unit, sel_y - 3 * unit}, {ul_x1 + 4 * unit, sel_y + 5 * unit},
                                          IM_COL32(236, 196, 110, static_cast<int>(40 * e)), 4 * unit);
                        wd->AddRectFilled({ul_x0, sel_y - 1 * unit}, {ul_x1, sel_y + 3 * unit},
                                          IM_COL32(236, 200, 120, static_cast<int>(255 * e)), 2 * unit);
                        // content cross-fade + slight rise
                        const float ce = EaseOut(static_cast<float>(ImGui::GetTime() - tab_t) / 0.22f);
                        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, e * (0.0f + ce));
                        ImGui::Dummy({0, (1.0f - ce) * 14 * unit});
                        ImGui::Spacing();
                        switch (i) {
                        case 0: PadPage(unit); break;
                        case 1: KbmPage(unit); break;
                        case 2: TrophyPage(unit); break;
                        case 3: OptionsPage(unit); break;
                        default: ExitPage(unit); break;
                        }
                        ImGui::PopStyleVar();
                        ImGui::EndTabItem();
                    }
                    ItemGlow(4 * unit);
                }
                want_tab = -1;
                ImGui::EndTabBar();
            }
        }
        ImGui::End();
        ImGui::PopStyleColor(10);
        ImGui::PopStyleVar(7);
    }

    void OptionsPage(float unit) {
        ImGui::TextColored(kGold, "Inicio");
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::BeginDisabled(!g_direct.exists);
        bool on = g_direct.on;
        if (ImGui::Checkbox("Iniciar directamente (sin pasar por el launcher)", &on)) {
            if (WriteDirect(on)) {
                g_direct.on = on;
            }
        }
        ItemGlow(4 * unit);
        ImGui::EndDisabled();
        ImGui::Indent(40 * unit);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 1500 * unit);
        ImGui::TextColored(kDim, "Si está activado, la próxima vez que abras el launcher el juego "
                                 "se inicia directamente con los ajustes guardados actuales. "
                                 "Mantén pulsada la tecla Mayús (Shift) al abrir el launcher para "
                                 "volver a verlo.");
        if (!g_direct.exists) {
            ImGui::TextColored(kWarn, "No se encontró launcher.ini junto a shadPS4.exe (%s). Abre "
                                      "el juego una vez desde el launcher para crearlo.",
                               LauncherIni().string().c_str());
        }
        if (!g_direct.error.empty()) {
            ImGui::TextColored(kWarn, "%s", g_direct.error.c_str());
        }
        ImGui::PopTextWrapPos();
        ImGui::Unindent(40 * unit);
    }

    void ExitPage(float unit) {
        ImGui::TextWrapped("Cierra el emulador de forma ordenada. Guarda antes en el juego: "
                           "Bloodborne guarda solo, espera a que no se vea el icono de guardado.");
        ImGui::Spacing();
        if (!confirm_exit) {
            if (FancyButton("Salir del juego", {360 * unit, 70 * unit})) {
                confirm_exit = true;
            }
            return;
        }
        ImGui::TextColored(kWarn, "¿Seguro que quieres salir del juego?");
        if (FancyButton("Sí, salir", {300 * unit, 70 * unit})) {
            LOG_INFO(Frontend, "[bb-menu] Salir del juego: closing the emulator");
            SDL_Event e{};
            e.type = SDL_EVENT_QUIT;
            SDL_PushEvent(&e);
        }
        ImGui::SameLine();
        if (FancyButton("No, volver", {300 * unit, 70 * unit})) {
            confirm_exit = false;
        }
    }
};
MenuLayer g_layer;
bool g_inited = false;

} // namespace

void Init(SDL_Window* window) {
    if (g_inited) {
        return;
    }
    g_inited = true;
    g_window = window;
    g_wake_event = SDL_RegisterEvents(1);
    ImGui::Layer::AddLayer(&g_layer);
    std::thread(CustomWatcher).detach();
}

bool IsOpen() {
    return g_open;
}

bool PushTrophyToast(const std::filesystem::path& icon, const std::string& name,
                     std::string_view grade) {
    if (const char* v = std::getenv("BB_TROPHY_POPUP"); v != nullptr && v[0] == '0') {
        return false;
    }
    std::scoped_lock lk{g_toast_mtx};
    Toast t;
    t.icon_path = icon;
    t.name = name;
    t.grade = grade.empty() ? 'B' : grade[0];
    g_toasts.push_back(std::move(t));
    // the F1 list: refresh on the next open
    return true;
}

bool OnSdlEvent(SDL_Event& event) {
    if (!g_inited) {
        return false;
    }
    if (g_wake_event != 0 && event.type == g_wake_event) {
        if (g_want_reload.exchange(false)) {
            Input::ParseInputConfig(Serial());
            LOG_INFO(Frontend, "[bb-menu] input config reloaded");
        }
        if (g_want_close.exchange(false)) {
            SetOpen(false);
        }
        return true;
    }
    const bool f1 = (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) &&
                    event.key.key == SDLK_F1;
    if (f1) {
        if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
            SetOpen(!g_open);
        }
        return true;
    }
    // live pad state (also when closed, cheap)
    if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN || event.type == SDL_EVENT_GAMEPAD_BUTTON_UP) {
        const u32 bit = 1u << (event.gbutton.button & 31);
        if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
            g_pad_buttons |= bit;
        } else {
            g_pad_buttons &= ~bit;
        }
    } else if (event.type == SDL_EVENT_GAMEPAD_AXIS_MOTION) {
        if (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) {
            g_trig_l = event.gaxis.value;
        } else if (event.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
            g_trig_r = event.gaxis.value;
        }
    }
    if (!g_open) {
        return false;
    }
    // a rebind waiting for its input
    {
        std::scoped_lock lk{g_cap_mtx};
        if (g_capture != Capture::None && g_captured.empty() && !g_cap_cancel &&
            SDL_GetTicks() - g_capture_since > 250) {
            std::string got;
            bool consumed = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) {
                g_cap_cancel = true;
                consumed = true;
            } else if (g_capture == Capture::Kbm) {
                if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
                    got = KeyName(event.key.key, false);
                    consumed = true;
                } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                    got = KeyName(event.button.button, true);
                    consumed = true;
                } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                    got = event.wheel.y > 0   ? "mousewheelup"
                          : event.wheel.y < 0 ? "mousewheeldown"
                          : event.wheel.x > 0 ? "mousewheelright"
                                              : "mousewheelleft";
                    consumed = true;
                }
            } else if (g_capture == Capture::Pad) {
                if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
                    if (const char* n = PadName(event.gbutton.button)) {
                        got = n;
                    }
                    consumed = true;
                } else if (event.type == SDL_EVENT_GAMEPAD_AXIS_MOTION &&
                           event.gaxis.value > 16000) {
                    if (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) {
                        got = "l2";
                    } else if (event.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
                        got = "r2";
                    }
                    consumed = !got.empty();
                }
            } else if (g_capture == Capture::PadStick) {
                if (event.type == SDL_EVENT_GAMEPAD_AXIS_MOTION &&
                    std::abs(int(event.gaxis.value)) > 20000) {
                    const auto ax = event.gaxis.axis;
                    if (ax == SDL_GAMEPAD_AXIS_LEFTX || ax == SDL_GAMEPAD_AXIS_LEFTY) {
                        got = "left";
                    } else if (ax == SDL_GAMEPAD_AXIS_RIGHTX || ax == SDL_GAMEPAD_AXIS_RIGHTY) {
                        got = "right";
                    }
                    consumed = !got.empty();
                }
            }
            if (!got.empty()) {
                g_captured = got;
            }
            if (consumed) {
                if (got.empty() && !g_cap_cancel) {
                    g_cap_cancel = true; // an input that cannot be bound: cancel
                }
                return true; // not to ImGui (no click / Cross on the menu)
            }
        }
    }
    ImGui::Core::ProcessEvent(&event);
    switch (event.type) {
    // releases go on to the game's input state so nothing stays held after the menu
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_MOUSE_BUTTON_UP:
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_UP:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION:
    case SDL_EVENT_GAMEPAD_SENSOR_UPDATE:
        // (pad events update the controller state; the game reads it neutral while the
        // capture is held: libScePad "Intercepted")
        return false;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_TEXT_INPUT:
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_WHEEL:
    case SDL_EVENT_MOUSE_MOTION:
        return true;
    default:
        return false;
    }
}

} // namespace BbMenu
