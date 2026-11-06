// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// (bb-menu-121) Reworked: full-screen hub + one screen per section, see the 1.21 notes below.
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
#include "input/controller.h"
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
// (bb-menu-133) "Jugar con teclado y ratón": mouse-look (mouse -> stick, cursor captured and
// hidden) from the start and whenever the menu closes; persisted in <user>/bb_menu.ini.
bool g_kbm_play = false;
bool g_vib_on = true;   // (bb-input-135) rumble
float g_vib_pct = 100; // (bb-input-135) rumble intensity, %
float g_mouse_sens = 0; // (bb-input-137) mouse camera sensitivity, 0 = the config's
bool g_invert_y = false;
// (bb-menu-133) Mando > "Probar y mapear": the pad never moves the menu, presses are shown live
std::atomic<bool> g_test_mode{false};
// (bb-mapwiz-136) Mando > "Mapear controles": the wizard listens for one physical pad input
// (event thread writes g_wiz_got under g_cap_mtx, render thread takes it)
std::atomic<bool> g_wiz_listen{false};
std::string g_wiz_got;
Uint64 g_wiz_start_down = 0; // OPTIONS: a short press is an input, a long hold leaves
std::atomic<int> g_axis[4]{}; // SDL LEFTX, LEFTY, RIGHTX, RIGHTY
std::atomic<bool> g_touch{false};

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

// ---------------------------------------------------------------- bb_menu.ini (bb-menu-133)
std::filesystem::path MenuIni() {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "bb_menu.ini";
}
void LoadMenuSettings() {
    std::ifstream in(MenuIni());
    std::string l;
    while (std::getline(in, l)) {
        if (l.rfind("kbm_play=", 0) == 0) g_kbm_play = l.size() > 9 && l[9] == '1';
        if (l.rfind("invert_y=", 0) == 0) g_invert_y = l.size() > 9 && l[9] == '1';
        if (l.rfind("vibration=", 0) == 0) g_vib_on = !(l.size() > 10 && l[10] == '0');
        if (l.rfind("vibration_pct=", 0) == 0) g_vib_pct = std::clamp(std::atof(l.c_str() + 14), 0.0, 200.0);
        if (l.rfind("mouse_sens=", 0) == 0) g_mouse_sens = std::clamp(std::atof(l.c_str() + 11), 0.0, 10.0);
    }
    Input::SetMouseSensitivity(g_mouse_sens);
    Input::SetMouseInvertY(g_invert_y);
    Input::SetVibrationSettings(g_vib_on, static_cast<int>(g_vib_pct));
}
void SaveMenuSettings() {
    std::ofstream out(MenuIni(), std::ios::trunc);
    out << "# Bloodborne F1 menu (bb-menu-133)\n"
        << "kbm_play=" << (g_kbm_play ? 1 : 0) << "\n"
        << "invert_y=" << (g_invert_y ? 1 : 0) << "\n"
        << "vibration=" << (g_vib_on ? 1 : 0) << "\n"
        << "vibration_pct=" << static_cast<int>(g_vib_pct) << "\n"
        << "mouse_sens=" << fmt::format("{:.2f}", g_mouse_sens) << "\n";
    Input::SetMouseSensitivity(g_mouse_sens);
    Input::SetMouseInvertY(g_invert_y);
    Input::SetVibrationSettings(g_vib_on, static_cast<int>(g_vib_pct));
    LOG_INFO(Frontend, "[bb-menu] bb_menu.ini: kbm_play={} invert_y={}", g_kbm_play, g_invert_y);
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
        g_test_mode = false;
        g_wiz_listen = false;
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
        {"cross", "Cruz"}, {"circle", "Círculo"}, {"square", "Cuadrado"},
        {"triangle", "Triángulo"}, {"l1", "L1"}, {"r1", "R1"}, {"l2", "L2"}, {"r2", "R2"},
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

// ---------------------------------------------------------------- 1.21 look: fonts, palette, helpers
// (bb-menu-121) Fonts (SIL OFL 1.1, menu_assets\fonts, licences next to them): Cinzel for titles,
// Cormorant Garamond for text, Inter for small UI. Dynamic ImGui fonts: baked at the pixel size
// each call asks for (sharp at 1440p), Spanish accents included. Missing files: default font.
ImFont* g_ft_title = nullptr;
ImFont* g_ft_text = nullptr;
ImFont* g_ft_italic = nullptr;
ImFont* g_ft_ui = nullptr;

ImFont* FT(ImFont* f) {
    return f != nullptr ? f : ImGui::GetFont();
}
float FontK() {
    const ImGuiStyle& s = ImGui::GetStyle();
    float k = s.FontScaleMain * s.FontScaleDpi;
#ifndef IMGUI_DISABLE_OBSOLETE_FUNCTIONS
    k *= ImGui::GetIO().FontGlobalScale; // shadPS4 sets the legacy 0.5
#endif
    return k > 0.01f ? k : 1.0f;
}
void PushF(ImFont* f, float px) {
    ImGui::PushFont(f, std::max(1.0f, px / FontK()));
}

// global alpha of what is drawn now (menu open/close x screen transition)
float g_a = 1.0f;
ImU32 Col(int r, int g, int b, float a = 1.0f) {
    return IM_COL32(r, g, b, static_cast<int>(255.0f * std::clamp(a * g_a, 0.0f, 1.0f)));
}
ImU32 Gold(float a = 1.0f) {
    return Col(228, 192, 122, a);
}
ImU32 Bronze(float a = 1.0f) {
    return Col(158, 112, 62, a);
}
ImU32 Parch(float a = 1.0f) {
    return Col(236, 228, 212, a);
}
ImU32 Muted(float a = 1.0f) {
    return Col(158, 148, 134, a);
}

float EaseOutBack(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    const float c1 = 1.70158f, c3 = c1 + 1.0f;
    return 1.0f + c3 * std::pow(x - 1.0f, 3.0f) + c1 * std::pow(x - 1.0f, 2.0f);
}
float Approach(float cur, float target, float dt, float speed) {
    return cur + (target - cur) * std::min(1.0f, dt * speed);
}
float Now() {
    return static_cast<float>(ImGui::GetTime());
}

int Utf8Len(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
}
ImVec2 TextSize(ImFont* f, float px, const char* s, const char* e = nullptr) {
    return FT(f)->CalcTextSizeA(px, FLT_MAX, 0.0f, s, e);
}
void Txt(ImDrawList* dl, ImFont* f, float px, ImVec2 p, ImU32 c, const char* s,
         const char* e = nullptr) {
    dl->AddText(FT(f), px, {std::floor(p.x), std::floor(p.y)}, c, s, e);
}
void TxtC(ImDrawList* dl, ImFont* f, float px, float cx, float y, ImU32 c, const char* s) {
    Txt(dl, f, px, {cx - TextSize(f, px, s).x * 0.5f, y}, c, s);
}
void TxtR(ImDrawList* dl, ImFont* f, float px, float rx, float y, ImU32 c, const char* s) {
    Txt(dl, f, px, {rx - TextSize(f, px, s).x, y}, c, s);
}
// letter-spaced caps (titles)
float SpacedW(ImFont* f, float px, const char* s, float sp) {
    float w = 0;
    for (const char* p = s; *p;) {
        const int n = Utf8Len(static_cast<unsigned char>(*p));
        w += TextSize(f, px, p, p + n).x + sp;
        p += n;
    }
    return std::max(0.0f, w - sp);
}
void TxtSpaced(ImDrawList* dl, ImFont* f, float px, ImVec2 p, ImU32 c, const char* s, float sp,
               bool center) {
    float x = center ? p.x - SpacedW(f, px, s, sp) * 0.5f : p.x;
    for (const char* q = s; *q;) {
        const int n = Utf8Len(static_cast<unsigned char>(*q));
        Txt(dl, f, px, {x, p.y}, c, q, q + n);
        x += TextSize(f, px, q, q + n).x + sp;
        q += n;
    }
}
std::vector<std::string> Wrap(ImFont* f, float px, const std::string& s, float w) {
    std::vector<std::string> lines;
    std::string cur, word;
    std::stringstream ss(s);
    while (ss >> word) {
        const std::string t = cur.empty() ? word : cur + " " + word;
        if (!cur.empty() && TextSize(f, px, t.c_str()).x > w) {
            lines.push_back(cur);
            cur = word;
        } else {
            cur = t;
        }
    }
    if (!cur.empty()) {
        lines.push_back(cur);
    }
    return lines;
}

// thin gold rule with a diamond in the middle, fading at both ends
void OrnLine(ImDrawList* dl, float cx, float y, float half, float a, float u) {
    const ImU32 c = Gold(a), z = Gold(0.0f);
    const float gap = 16 * u, th = std::max(1.0f, 1.3f * u), d = 5.5f * u;
    dl->AddRectFilledMultiColor({cx - half, y - th * 0.5f}, {cx - gap, y + th * 0.5f}, z, c, c, z);
    dl->AddRectFilledMultiColor({cx + gap, y - th * 0.5f}, {cx + half, y + th * 0.5f}, c, z, z, c);
    dl->AddQuadFilled({cx, y - d}, {cx + d, y}, {cx, y + d}, {cx - d, y}, c);
    dl->AddQuad({cx, y - d * 2.0f}, {cx + d * 2.0f, y}, {cx, y + d * 2.0f}, {cx - d * 2.0f, y},
                Gold(a * 0.55f), th);
    dl->AddCircleFilled({cx - gap - 6 * u, y}, 1.8f * u, c);
    dl->AddCircleFilled({cx + gap + 6 * u, y}, 1.8f * u, c);
}

// a dark gothic panel: soft shadow, vertical gradient, bronze frame, inner hairline,
// corner filigree. glow 0..1 lights the frame.
void Panel(ImDrawList* dl, ImVec2 a, ImVec2 b, float u, float glow = 0.0f, float fill = 1.0f) {
    const float r = 6 * u;
    for (int i = 3; i >= 1; i--) {
        const float g = i * 7.0f * u;
        dl->AddRectFilled({a.x - g * 0.5f, a.y - g * 0.2f + 8 * u}, {b.x + g * 0.5f, b.y + g + 8 * u},
                          Col(0, 0, 0, 0.10f * fill), r + g);
    }
    if (glow > 0.01f) {
        for (int i = 4; i >= 1; i--) {
            const float g = i * 3.0f * u;
            dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g}, Gold(0.10f * glow / i * 2.0f),
                        r + g, 0, 2.0f * u);
        }
    }
    dl->AddRectFilledMultiColor(a, b, Col(30, 25, 22, 0.94f * fill), Col(30, 25, 22, 0.94f * fill),
                                Col(11, 10, 11, 0.96f * fill), Col(11, 10, 11, 0.96f * fill));
    dl->AddRect(a, b, Bronze(0.55f + 0.45f * glow), r, 0, std::max(1.0f, 1.2f * u));
    const float in = 7 * u;
    dl->AddRect({a.x + in, a.y + in}, {b.x - in, b.y - in}, Gold(0.08f + 0.12f * glow), r * 0.5f, 0,
                1.0f);
    const float c = 22 * u;
    const ImVec2 cs[4] = {{a.x + in, a.y + in}, {b.x - in, a.y + in}, {a.x + in, b.y - in},
                          {b.x - in, b.y - in}};
    for (int k = 0; k < 4; k++) {
        const float sx = (k & 1) ? -1.0f : 1.0f, sy = (k & 2) ? -1.0f : 1.0f;
        const ImU32 gc = Gold(0.55f + 0.4f * glow);
        dl->AddLine(cs[k], {cs[k].x + sx * c, cs[k].y}, gc, std::max(1.0f, 1.4f * u));
        dl->AddLine(cs[k], {cs[k].x, cs[k].y + sy * c}, gc, std::max(1.0f, 1.4f * u));
        const ImVec2 q{cs[k].x + sx * 6 * u, cs[k].y + sy * 6 * u};
        const float d = 2.6f * u;
        dl->AddQuadFilled({q.x, q.y - d}, {q.x + d, q.y}, {q.x, q.y + d}, {q.x - d, q.y}, gc);
    }
}

// an invisible hit area; the cursor is always followed by an item (no boundary asserts)
bool Hit(const char* id, ImVec2 a, ImVec2 b, bool* hovered = nullptr) {
    ImGui::SetCursorScreenPos(a);
    const bool r =
        ImGui::InvisibleButton(id, {std::max(1.0f, b.x - a.x), std::max(1.0f, b.y - a.y)});
    if (hovered != nullptr) {
        *hovered = ImGui::IsItemHovered();
    }
    return r;
}
bool MouseMoved() {
    const ImVec2 d = ImGui::GetIO().MouseDelta;
    return d.x != 0.0f || d.y != 0.0f;
}

// ---------------------------------------------------------------- menu navigation
// Pad / keyboard navigation goes through this queue (main thread pushes, render thread takes):
// only the d-pad, the face buttons and L1 / R1 move the menu; the sticks never do.
enum class Nav { Up, Down, Left, Right, Accept, Back, Clear, Prev, Next };
std::mutex g_nav_mtx;
std::vector<Nav> g_nav;
void PushNav(Nav n) {
    std::scoped_lock lk{g_nav_mtx};
    if (g_nav.size() < 16) {
        g_nav.push_back(n);
    }
}
std::vector<Nav> TakeNav() {
    std::scoped_lock lk{g_nav_mtx};
    std::vector<Nav> v;
    v.swap(g_nav);
    return v;
}

// ---------------------------------------------------------------- PS4 glyphs
ImU32 FaceColor(const std::string& out, float a) {
    if (out == "triangle") return Col(112, 196, 182, a);
    if (out == "circle") return Col(222, 112, 108, a);
    if (out == "cross") return Col(132, 158, 226, a);
    if (out == "square") return Col(214, 142, 196, a);
    return Gold(a);
}
std::string FocusKey(const std::string& out) {
    if (out.starts_with("axis_left")) return "stick_left";
    if (out.starts_with("axis_right")) return "stick_right";
    return out;
}
std::string Short(const std::string& out) {
    static const std::map<std::string, std::string> m = {
        {"l1", "L1"}, {"r1", "R1"}, {"l2", "L2"}, {"r2", "R2"}, {"l3", "L3"}, {"r3", "R3"},
        {"options", "OPT"}, {"touchpad_center", "TP"}, {"back", "SH"},
        {"pad_up", "\xe2\x86\x91"}, {"pad_down", "\xe2\x86\x93"}, {"pad_left", "\xe2\x86\x90"},
        {"pad_right", "\xe2\x86\x92"}, {"stick_left", "LS"}, {"stick_right", "RS"},
        {"axis_left_y_minus", "LS\xe2\x86\x91"}, {"axis_left_y_plus", "LS\xe2\x86\x93"},
        {"axis_left_x_minus", "LS\xe2\x86\x90"}, {"axis_left_x_plus", "LS\xe2\x86\x92"},
        {"axis_right_y_minus", "RS\xe2\x86\x91"}, {"axis_right_y_plus", "RS\xe2\x86\x93"},
        {"axis_right_x_minus", "RS\xe2\x86\x90"}, {"axis_right_x_plus", "RS\xe2\x86\x92"}};
    if (auto it = m.find(out); it != m.end()) return it->second;
    return "?";
}
// a round PS4 button badge: face symbols drawn, the rest as short text
void Glyph(ImDrawList* dl, const std::string& out, ImVec2 c, float r, float a = 1.0f,
           bool lit = false) {
    if (lit) {
        for (int i = 3; i >= 1; i--) {
            dl->AddCircleFilled(c, r + i * r * 0.18f, Gold(0.10f * a));
        }
    }
    dl->AddCircleFilled(c, r, Col(18, 15, 14, 0.96f * a));
    dl->AddCircle(c, r, lit ? Gold(a) : Bronze(0.9f * a), 0, std::max(1.0f, r * 0.08f));
    if (out == "cross" || out == "circle" || out == "square" || out == "triangle") {
        DrawSymbol(dl, out, c, r * 0.62f, FaceColor(out, a));
        return;
    }
    const std::string s = Short(out);
    const float px = s.size() <= 2 || s.starts_with("\xe2") ? r * 0.95f : r * 0.72f;
    const ImVec2 ts = TextSize(g_ft_ui, px, s.c_str());
    Txt(dl, g_ft_ui, px, {c.x - ts.x * 0.5f, c.y - ts.y * 0.5f}, Parch(a), s.c_str());
}
// a keyboard key cap
float KeyCapW(const std::string& label, float u) {
    return TextSize(g_ft_ui, 19 * u, label.c_str()).x + 22 * u;
}
void KeyCap(ImDrawList* dl, ImVec2 p, const std::string& label, float u, float a, bool lit) {
    const float w = KeyCapW(label, u), h = 34 * u;
    dl->AddRectFilled({p.x, p.y + 3 * u}, {p.x + w, p.y + h + 3 * u}, Col(0, 0, 0, 0.5f * a),
                      6 * u);
    dl->AddRectFilledMultiColor(p, {p.x + w, p.y + h}, Col(48, 42, 36, a), Col(48, 42, 36, a),
                                Col(26, 23, 21, a), Col(26, 23, 21, a));
    dl->AddRect(p, {p.x + w, p.y + h}, lit ? Gold(a) : Bronze(0.8f * a), 6 * u, 0,
                std::max(1.0f, 1.2f * u));
    Txt(dl, g_ft_ui, 19 * u, {p.x + 11 * u, p.y + (h - 19 * u) * 0.5f - 1 * u},
        lit ? Gold(a) : Parch(a), label.c_str());
}

// ---------------------------------------------------------------- the pads: spots and rows
struct Spot {
    const char* output; // PS4 output (game photo) or physical input (your pad)
    float x, y, d, h;   // centre and size in the photo's pixels
};

// mando-ps4.png (1536 x 1024, bb-menu-133 photo): the game's pad. L2 / R2 drawn over the
// bumpers' top.
const Spot kPs4Spots[] = {
    {"l2", 353, 52, 150, 36},         {"r2", 1173, 52, 150, 36},
    {"l1", 353, 100, 170, 40},        {"r1", 1173, 100, 170, 40},
    {"pad_up", 353, 240, 72, 72},     {"pad_down", 353, 387, 72, 72},
    {"pad_left", 287, 316, 72, 72},   {"pad_right", 420, 316, 72, 72},
    {"triangle", 1187, 220, 92, 92},  {"circle", 1291, 316, 92, 92},
    {"cross", 1191, 413, 92, 92},     {"square", 1093, 316, 92, 92},
    {"l3", 549, 500, 150, 150},       {"r3", 981, 500, 150, 150},
    {"stick_left", 549, 500, 220, 220}, {"stick_right", 981, 500, 220, 220},
    {"options", 1040, 183, 40, 74},   {"back", 497, 183, 40, 74},
    {"touchpad_center", 769, 240, 433, 227},
};
// your pad, physical inputs (the input config's words); triggers above the photo
const Spot kPs5Phys[] = {
    {"pad_up", 96, 62, 29, 29},    {"pad_down", 96, 126, 29, 29}, {"pad_left", 65, 93, 29, 29},
    {"pad_right", 128, 93, 29, 29}, {"triangle", 435, 60, 38, 38}, {"circle", 472, 94, 38, 38},
    {"cross", 435, 131, 38, 38},   {"square", 398, 94, 38, 38},   {"l3", 178, 160, 67, 67},
    {"r3", 346, 160, 67, 67},      {"options", 389, 32, 20, 20},  {"back", 142, 32, 20, 20},
    {"l1", 95, 9, 56, 16},         {"r1", 432, 9, 56, 16},        {"l2", 95, -16, 56, 16},
    {"r2", 432, -16, 56, 16},
};
const Spot k8bdPhys[] = {
    {"triangle", 416, 58, 38, 38}, {"square", 377, 95, 38, 38},   {"circle", 456, 95, 38, 38},
    {"cross", 416, 132, 38, 38},   {"l3", 125, 86, 72, 72},       {"r3", 339, 156, 72, 72},
    {"pad_up", 193, 140, 25, 25},  {"pad_down", 193, 192, 25, 25}, {"pad_left", 167, 166, 25, 25},
    {"pad_right", 219, 166, 25, 25}, {"back", 226, 89, 22, 22},   {"options", 312, 89, 22, 22},
    {"l1", 100, 6, 60, 16},        {"r1", 437, 6, 60, 16},        {"l2", 100, -18, 60, 16},
    {"r2", 437, -18, 60, 16},
};
const Spot kPs4Phys[] = {
    {"l2", 353, 52, 150, 36},        {"r2", 1173, 52, 150, 36},       {"l1", 353, 100, 170, 40},
    {"r1", 1173, 100, 170, 40},      {"pad_up", 353, 240, 72, 72},    {"pad_down", 353, 387, 72, 72},
    {"pad_left", 287, 316, 72, 72},  {"pad_right", 420, 316, 72, 72}, {"triangle", 1187, 220, 92, 92},
    {"circle", 1291, 316, 92, 92},   {"cross", 1191, 413, 92, 92},    {"square", 1093, 316, 92, 92},
    {"l3", 549, 500, 150, 150},      {"r3", 981, 500, 150, 150},      {"options", 1040, 183, 40, 74},
    {"back", 497, 183, 40, 74},
};

struct Row {
    const char* output;
    const char* label;
};
const Row kPadRows[] = {
    {"cross", "Cruz"},          {"circle", "Círculo"},
    {"square", "Cuadrado"},     {"triangle", "Triángulo"},
    {"l1", "L1"},               {"r1", "R1"},
    {"l2", "L2"},               {"r2", "R2"},
    {"l3", "L3"},               {"r3", "R3"},
    {"options", "OPTIONS"},     {"touchpad_center", "Panel táctil"},
    {"pad_up", "Cruceta arriba"}, {"pad_down", "Cruceta abajo"},
    {"pad_left", "Cruceta izquierda"}, {"pad_right", "Cruceta derecha"},
    {"stick_left", "Stick izquierdo"}, {"stick_right", "Stick derecho"},
};
const Row kKbmRows[] = {
    {"cross", "Cruz"},          {"circle", "Círculo"},
    {"square", "Cuadrado"},     {"triangle", "Triángulo"},
    {"l1", "L1"},               {"r1", "R1"},
    {"l2", "L2"},               {"r2", "R2"},
    {"l3", "L3"},               {"r3", "R3"},
    {"options", "OPTIONS"},     {"touchpad_center", "Panel táctil"},
    {"pad_up", "Cruceta arriba"}, {"pad_down", "Cruceta abajo"},
    {"pad_left", "Cruceta izquierda"}, {"pad_right", "Cruceta derecha"},
    {"axis_left_y_minus", "Stick izq. arriba"}, {"axis_left_y_plus", "Stick izq. abajo"},
    {"axis_left_x_minus", "Stick izq. izquierda"}, {"axis_left_x_plus", "Stick izq. derecha"},
    {"axis_right_y_minus", "Stick der. arriba"}, {"axis_right_y_plus", "Stick der. abajo"},
    {"axis_right_x_minus", "Stick der. izquierda"}, {"axis_right_x_plus", "Stick der. derecha"},
};

// (bb-mapwiz-136) the wizard's order: one PS4 button at a time. Sticks are left as they are
// (noisy, rarely remapped); the physical touchpad always gives the touchpad by itself, the
// "Panel táctil" step adds a button for it (Share / Create / Back by default).
const Row kWizSteps[] = {
    {"cross", "Cruz"},           {"circle", "Círculo"},        {"square", "Cuadrado"},
    {"triangle", "Triángulo"},   {"l1", "L1"},                 {"r1", "R1"},
    {"l2", "L2"},                {"r2", "R2"},                 {"l3", "L3"},
    {"r3", "R3"},                {"options", "Options"},       {"touchpad_center", "Panel táctil (Share / Create)"},
    {"pad_up", "D-pad arriba"},  {"pad_down", "D-pad abajo"},  {"pad_left", "D-pad izquierda"},
    {"pad_right", "D-pad derecha"},
};
constexpr int kWizN = static_cast<int>(std::size(kWizSteps));
std::vector<std::string> Tokens(const std::string& s);
std::string PhysLabel(const std::string& tok) {
    static const std::map<std::string, const char*> m = {
        {"cross", "Cruz"}, {"circle", "Círculo"}, {"square", "Cuadrado"},
        {"triangle", "Triángulo"}, {"l1", "L1"}, {"r1", "R1"}, {"l2", "L2"}, {"r2", "R2"},
        {"l3", "L3"}, {"r3", "R3"}, {"options", "Options / Start"}, {"back", "Share / Back"},
        {"pad_up", "D-pad arriba"}, {"pad_down", "D-pad abajo"}, {"pad_left", "D-pad izq."},
        {"pad_right", "D-pad der."}, {"lpaddle_high", "Pala izq. 1"},
        {"lpaddle_low", "Pala izq. 2"}, {"rpaddle_high", "Pala der. 1"},
        {"rpaddle_low", "Pala der. 2"},
    };
    std::string out;
    for (const auto& t : Tokens(tok)) {
        auto it = m.find(t);
        out += (out.empty() ? "" : " + ") + (it != m.end() ? std::string(it->second) : t);
    }
    return out.empty() ? "(sin asignar)" : out;
}

// Bloodborne's own (PS4) controls
const char* DefaultFn(const std::string& out) {
    static const std::map<std::string, const char*> m = {
        {"r1", "Ataque ligero"},
        {"r2", "Ataque fuerte / cargado"},
        {"l1", "Transformar arma"},
        {"l2", "Arma de fuego / mano izq."},
        {"triangle", "Vial de sangre"},
        {"square", "Usar objeto rápido"},
        {"circle", "Esquivar / correr"},
        {"cross", "Acción / interactuar"},
        {"l3", "Saltar (corriendo)"},
        {"r3", "Fijar objetivo"},
        {"options", "Menú"},
        {"touchpad_center", "Gestos"},
        {"pad_up", "Cambiar objeto"},
        {"pad_down", "Cambiar objeto rápido"},
        {"pad_left", "Arma de mano izquierda"},
        {"pad_right", "Arma de mano derecha"},
        {"stick_left", "Moverse"},
        {"stick_right", "Cámara"},
        {"back", "Compartir"},
    };
    if (auto it = m.find(FocusKey(out)); it != m.end()) return it->second;
    return "";
}
const char* RowLabel(const std::string& out) {
    for (const auto& r : kKbmRows) {
        if (out == r.output) return r.label;
    }
    for (const auto& r : kPadRows) {
        if (out == r.output) return r.label;
    }
    return "";
}

std::vector<std::string> Group(const Row* rows, size_t n) {
    std::vector<std::string> g;
    for (size_t i = 0; i < n; i++) {
        g.emplace_back(rows[i].output);
    }
    return g;
}
std::vector<std::string> Tokens(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) {
        if (!t.empty()) v.push_back(t);
    }
    return v;
}

std::string BindingText(const std::string& output, Cat cat) {
    if (output == "stick_left" || output == "stick_right") {
        const auto x = g_cfg.GetRaw(output == "stick_left" ? "axis_left_x" : "axis_right_x", cat);
        return x.empty() ? "(sin asignar)" : Pretty(x);
    }
    return g_cfg.Get(output, cat);
}
// the physical inputs (your pad) bound to a PS4 output
std::vector<std::string> PhysOf(const std::string& output) {
    if (output == "stick_left" || output == "stick_right") {
        const auto x = g_cfg.GetRaw(output == "stick_left" ? "axis_left_x" : "axis_right_x", Cat::Pad);
        if (x.starts_with("axis_left")) return {"l3"};
        if (x.starts_with("axis_right")) return {"r3"};
        return {};
    }
    return Tokens(g_cfg.GetRaw(output, Cat::Pad));
}

void StartCapture(Capture c, const std::string& target) {
    std::scoped_lock lk{g_cap_mtx};
    g_capture = c;
    g_capture_target = target;
    g_capture_since = SDL_GetTicks();
    g_captured.clear();
    g_cap_cancel = false;
}
void StartCaptureFor(const std::string& out, Cat cat) {
    StartCapture(cat == Cat::Kbm ? Capture::Kbm
                 : out.starts_with("stick_") ? Capture::PadStick
                                             : Capture::Pad,
                 out);
}

// The capture's result, applied on the render thread.
void ApplyCapture() {
    std::string got, target;
    bool cancel = false;
    Capture kind;
    {
        std::scoped_lock lk{g_cap_mtx};
        if (g_capture == Capture::None) {
            return;
        }
        // 5 s without an input: cancelled
        if (g_captured.empty() && !g_cap_cancel && SDL_GetTicks() - g_capture_since > 5250) {
            g_cap_cancel = true;
        }
        if (g_captured.empty() && !g_cap_cancel) {
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

// a physical input held now
bool PhysHeld(const std::string& token) {
    const u32 bits = g_pad_buttons.load();
    for (u8 b = 0; b < 32; b++) {
        if ((bits & (1u << b)) != 0) {
            if (const char* n = PadName(b); n != nullptr && token == n) {
                return true;
            }
        }
    }
    return (token == "l2" && g_trig_l > 12000) || (token == "r2" && g_trig_r > 12000) ||
           (token.starts_with("touchpad") && g_touch);
}
// a PS4 output whose pad binding is held now (the live highlight)
bool PhysicallyHeld(const std::string& output) {
    for (const auto& t : PhysOf(output)) {
        if (PhysHeld(t)) return true;
    }
    return false;
}

Tex g_ps4_tex;
const Spot* FindSpot(const Spot* s, size_t n, const std::string& out) {
    for (size_t i = 0; i < n; i++) {
        if (out == s[i].output) return &s[i];
    }
    return nullptr;
}

// ---------------------------------------------------------------- the game's pad (left panel)
// The DS4 photo with a callout per PS4 button: its glyph and Bloodborne's default function,
// joined to the button by a thin gold leader. Returns the output clicked; `hover` gets the
// output under the mouse; `anchors` the spots' screen centres (for the links).
std::string Ps4Panel(ImVec2 a, ImVec2 b, float u, const std::string& focus, std::string& hover,
                     std::map<std::string, ImVec2>& anchors, const char* subtitle) {
    auto* dl = ImGui::GetWindowDrawList();
    Panel(dl, a, b, u);
    Txt(dl, g_ft_title, 30 * u, {a.x + 30 * u, a.y + 22 * u}, Gold(), "MANDO DE PS4");
    Txt(dl, g_ft_italic, 24 * u, {a.x + 30 * u, a.y + 58 * u}, Muted(), subtitle);
    const ImTextureID tex = Get(g_ps4_tex, AssetDir() / "mando-ps4.png", 1536.0f, 1024.0f);
    const float pw = b.x - a.x, top = a.y + 100 * u, bottom = b.y - 18 * u;
    const float photo_w = std::min(pw * 0.50f, (bottom - top) * 1.5f);
    const float s = photo_w / 1536.0f, photo_h = 1024.0f * s;
    const ImVec2 p0{a.x + (pw - photo_w) * 0.5f, top + (bottom - top - photo_h) * 0.5f};
    // a soft pool of light under the pad
    for (int i = 5; i >= 1; i--) {
        const float rr = photo_w * (0.30f + i * 0.05f);
        dl->AddCircleFilled({p0.x + photo_w * 0.5f, p0.y + photo_h * 0.52f}, rr,
                            Col(120, 90, 50, 0.025f), 48);
    }
    if (tex) {
        dl->AddImage(tex, p0, {p0.x + photo_w, p0.y + photo_h}, {0, 0}, {1, 1}, Col(255, 255, 255));
    }
    const float t = Now();
    const auto at = [&](const Spot& sp) { return ImVec2{p0.x + sp.x * s, p0.y + sp.y * s}; };
    // spots
    for (const auto& sp : kPs4Spots) {
        const std::string o = sp.output;
        const ImVec2 c = at(sp);
        anchors[o] = c;
        if (o == "back") continue;
        const bool on = focus == o || hover == o;
        bool held = PhysicallyHeld(o);
        if (g_test_mode && o.starts_with("stick_")) {
            // (bb-menu-133) live stick: a gold bead where the stick bound to it points
            const auto src = PhysOf(o);
            const bool lsrc = !src.empty() && src[0] == "l3";
            const bool rsrc = !src.empty() && src[0] == "r3";
            if (lsrc || rsrc) {
                const float ax = g_axis[lsrc ? 0 : 2] / 32767.0f, ay = g_axis[lsrc ? 1 : 3] / 32767.0f;
                const float rr = sp.d * s * 0.32f;
                const ImVec2 q{c.x + std::clamp(ax, -1.0f, 1.0f) * rr,
                               c.y + std::clamp(ay, -1.0f, 1.0f) * rr};
                const float mag = std::sqrt(ax * ax + ay * ay);
                dl->AddCircle(c, rr, Gold(0.35f), 0, std::max(1.0f, u));
                dl->AddLine(c, q, Gold(0.6f), 2.0f * u);
                dl->AddCircleFilled(q, 9 * u, Gold(0.5f + 0.5f * std::min(mag, 1.0f)));
                held = held || mag > 0.3f;
            }
        }
        const float pulse = 0.6f + 0.4f * std::sin(t * 5.0f);
        const ImVec2 half{sp.d * s * 0.5f, sp.h * s * 0.5f};
        const bool rect = sp.d != sp.h;
        if (o == "l2" || o == "r2") {
            // trigger ghost over the bumper
            dl->AddRectFilled({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y},
                              Col(24, 21, 20, 0.85f), half.y);
            dl->AddRect({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y},
                        on || held ? Gold() : Bronze(0.7f), half.y, 0, std::max(1.0f, 1.2f * u));
            const char* l = o == "l2" ? "L2" : "R2";
            const ImVec2 ts = TextSize(g_ft_ui, 17 * u, l);
            Txt(dl, g_ft_ui, 17 * u, {c.x - ts.x * 0.5f, c.y - ts.y * 0.5f}, Parch(0.9f), l);
        }
        if (!on && !held) {
            continue;
        }
        const float al = held ? 1.0f : pulse;
        if (rect) {
            dl->AddRectFilled({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y},
                              Gold(0.22f * al), 10 * s);
            dl->AddRect({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y}, Gold(al),
                        10 * s, 0, 2.0f * u);
        } else {
            for (int i = 3; i >= 1; i--) {
                dl->AddCircle(c, half.x + i * 4 * u, Gold(0.12f * al), 0, 3.0f * u);
            }
            dl->AddCircleFilled(c, half.x, Gold(0.20f * al));
            dl->AddCircle(c, half.x, Gold(al), 0, 2.0f * u);
        }
    }
    // callout columns
    static const char* kLeft[] = {"l2", "l1", "pad_up", "pad_left", "pad_right",
                                  "pad_down", "l3", "stick_left"};
    static const char* kRight[] = {"r2", "r1", "touchpad_center", "options", "triangle",
                                   "square", "circle", "cross", "r3", "stick_right"};
    std::string clicked;
    const float row_h = std::min(52 * u, (bottom - top) / 10.5f);
    for (int side = 0; side < 2; side++) {
        const bool left = side == 0;
        const char** list = left ? kLeft : kRight;
        const int n = left ? 8 : 10;
        std::vector<std::pair<float, const char*>> v;
        for (int i = 0; i < n; i++) {
            v.push_back({anchors[list[i]].y, list[i]});
        }
        std::stable_sort(v.begin(), v.end(),
                         [](const auto& x, const auto& y) { return x.first < y.first; });
        // keep near the spot's height, at least a row apart, inside the panel
        std::vector<float> ys(v.size());
        float prev = top - row_h;
        for (size_t i = 0; i < v.size(); i++) {
            ys[i] = std::max(v[i].first - row_h * 0.5f, prev + row_h);
            prev = ys[i];
        }
        const float over = prev + row_h - bottom;
        for (size_t i = 0; i < v.size() && over > 0; i++) {
            ys[i] -= over;
        }
        for (size_t i = 1; i < v.size(); i++) {
            ys[i] = std::max(ys[i], ys[i - 1] + row_h);
        }
        const float x0 = left ? a.x + 22 * u : p0.x + photo_w - 6 * u;
        const float x1 = left ? p0.x + 6 * u : b.x - 22 * u;
        for (size_t i = 0; i < v.size(); i++) {
            const std::string o = v[i].second;
            const float y = ys[i], cy = y + row_h * 0.5f;
            const float r = row_h * 0.34f;
            const ImVec2 gc{left ? x1 - r - 4 * u : x0 + r + 4 * u, cy};
            bool hov = false;
            const std::string id = "##co_" + o;
            if (Hit(id.c_str(), {x0, y + 2 * u}, {x1, y + row_h - 2 * u}, &hov)) {
                clicked = o;
            }
            if (hov) hover = o;
            const bool on = focus == o || hover == o;
            if (on) {
                const ImU32 g0 = Gold(0.16f), g1 = Gold(0.0f);
                if (left) {
                    dl->AddRectFilledMultiColor({x0, y + 3 * u}, {x1, y + row_h - 3 * u}, g1, g0,
                                                g0, g1);
                } else {
                    dl->AddRectFilledMultiColor({x0, y + 3 * u}, {x1, y + row_h - 3 * u}, g0, g1,
                                                g1, g0);
                }
            }
            // leader: glyph -> photo edge -> spot
            const ImVec2 sc = anchors[o];
            const ImVec2 e0{left ? gc.x + r : gc.x - r, cy};
            const ImVec2 e1{left ? std::min(sc.x, std::max(e0.x, p0.x + photo_w * 0.05f))
                                 : std::max(sc.x, std::min(e0.x, p0.x + photo_w * 0.95f)),
                            cy};
            const ImU32 lc = on ? Gold(0.95f) : Gold(0.28f);
            const float lt = std::max(1.0f, (on ? 1.8f : 1.0f) * u);
            dl->AddLine(e0, e1, lc, lt);
            dl->AddLine(e1, sc, lc, lt);
            dl->AddCircleFilled(sc, (on ? 4.0f : 2.6f) * u, lc);
            Glyph(dl, o, gc, r, 1.0f, on || PhysicallyHeld(o));
            const char* fn = DefaultFn(o);
            const float px = 24 * u;
            const float ty = cy - px * 0.62f;
            const ImU32 tc = on ? Parch() : Col(206, 196, 180, 0.92f);
            if (left) {
                TxtR(dl, g_ft_text, px, gc.x - r - 10 * u, ty, tc, fn);
            } else {
                Txt(dl, g_ft_text, px, {gc.x + r + 10 * u, ty}, tc, fn);
            }
        }
    }
    // the spots are clickable too
    for (const auto& sp : kPs4Spots) {
        const std::string o = sp.output;
        if (o == "back" || o.starts_with("stick_")) continue;
        const ImVec2 c = at(sp);
        const ImVec2 half{std::max(8 * u, sp.d * s * 0.5f), std::max(8 * u, sp.h * s * 0.5f)};
        bool hov = false;
        const std::string id = "##sp_" + o;
        if (Hit(id.c_str(), {c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y}, &hov)) {
            clicked = o;
        }
        if (hov) hover = o;
    }
    return clicked;
}

// ---------------------------------------------------------------- your pad (right panel)
// g_pad_photo: 0 DualSense (mando-ps5.png), 1 8BitDo, 2 DualShock 4

struct PhysPhoto {
    Tex* tex;
    const char* file;
    float w, h;
    const Spot* spots;
    size_t n;
};
PhysPhoto CurrentPhys() {
    switch (g_pad_photo) {
    case 1: return {&g_pad_alt_tex, "mando-8bitdo.png", 537, 350, k8bdPhys, std::size(k8bdPhys)};
    case 2: return {&g_ps4_tex, "mando-ps4.png", 1536, 1024, kPs4Phys, std::size(kPs4Phys)};
    default: return {&g_pad_tex, "mando-ps5.png", 532, 350, kPs5Phys, std::size(kPs5Phys)};
    }
}

// Draws your pad with, next to each physical button, the PS4 button(s) it gives now
// (remapped ones lit). `phys` gets the physical spots' screen centres.
[[maybe_unused]] void PhysPanel(ImVec2 a, ImVec2 b, float u, const std::string& focus,
               std::map<std::string, ImVec2>& phys, std::string& hover) {
    auto* dl = ImGui::GetWindowDrawList();
    Panel(dl, a, b, u);
    Txt(dl, g_ft_title, 30 * u, {a.x + 30 * u, a.y + 22 * u}, Gold(), "TU MANDO");
    Txt(dl, g_ft_italic, 24 * u, {a.x + 30 * u, a.y + 58 * u}, Muted(),
        "qué hace ahora cada botón");
    // photo chooser chips
    const char* names[] = {"DualSense", "8BitDo", "DualShock 4"};
    float cx = b.x - 26 * u;
    for (int i = 2; i >= 0; i--) {
        const float w = TextSize(g_ft_ui, 18 * u, names[i]).x + 28 * u;
        const ImVec2 c0{cx - w, a.y + 26 * u}, c1{cx, a.y + 60 * u};
        bool hov = false;
        const std::string id = std::string("##ph") + names[i];
        if (Hit(id.c_str(), c0, c1, &hov)) {
            g_pad_photo = i;
        }
        const bool sel = g_pad_photo == i;
        dl->AddRectFilled(c0, c1, sel ? Col(70, 54, 32, 0.95f) : Col(22, 19, 18, 0.9f), 17 * u);
        dl->AddRect(c0, c1, sel || hov ? Gold() : Bronze(0.6f), 17 * u, 0, std::max(1.0f, u));
        Txt(dl, g_ft_ui, 18 * u, {c0.x + 14 * u, c0.y + 7 * u}, sel ? Gold() : Parch(0.8f),
            names[i]);
        cx = c0.x - 10 * u;
    }
    TxtR(dl, g_ft_ui, 16 * u, b.x - 26 * u, a.y + 66 * u, Muted(0.8f), "L1 / R1: cambiar foto");
    const PhysPhoto ph = CurrentPhys();
    const ImTextureID tex = Get(*ph.tex, AssetDir() / ph.file, ph.w, ph.h);
    const float pw = b.x - a.x, top = a.y + 130 * u, bottom = b.y - 40 * u;
    const float photo_w = std::min(pw * 0.78f, (bottom - top) * ph.w / ph.h);
    const float s = photo_w / ph.w, photo_h = ph.h * s;
    const ImVec2 p0{a.x + (pw - photo_w) * 0.5f, top + (bottom - top - photo_h) * 0.5f};
    for (int i = 5; i >= 1; i--) {
        dl->AddCircleFilled({p0.x + photo_w * 0.5f, p0.y + photo_h * 0.52f},
                            photo_w * (0.30f + i * 0.05f), Col(120, 90, 50, 0.025f), 48);
    }
    if (tex) {
        dl->AddImage(tex, p0, {p0.x + photo_w, p0.y + photo_h}, {0, 0}, {1, 1}, Col(255, 255, 255));
    } else {
        TxtC(dl, g_ft_ui, 18 * u, p0.x + photo_w * 0.5f, p0.y + photo_h * 0.5f, Muted(),
             "(imagen no encontrada en menu_assets)");
    }
    const std::vector<std::string> focus_phys = PhysOf(FocusKey(focus));
    const float t = Now();
    for (size_t i = 0; i < ph.n; i++) {
        const Spot& sp = ph.spots[i];
        const std::string p = sp.output;
        const ImVec2 c{p0.x + sp.x * s, p0.y + sp.y * s};
        phys[p] = c;
        const ImVec2 half{std::max(7 * u, sp.d * s * 0.5f), std::max(7 * u, sp.h * s * 0.5f)};
        const bool rect = sp.d != sp.h;
        const bool linked = std::find(focus_phys.begin(), focus_phys.end(), p) != focus_phys.end();
        const bool held = PhysHeld(p);
        bool hov = false;
        const std::string id = "##pp_" + p;
        Hit(id.c_str(), {c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y}, &hov);
        if ((p == "l2" || p == "r2") && sp.y < 0) {
            dl->AddRectFilled({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y},
                              Col(24, 21, 20, 0.9f), half.y);
            dl->AddRect({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y}, Bronze(0.8f),
                        half.y, 0, std::max(1.0f, u));
            const char* l = p == "l2" ? "L2" : "R2";
            const ImVec2 ts = TextSize(g_ft_ui, 15 * u, l);
            Txt(dl, g_ft_ui, 15 * u, {c.x - ts.x * 0.5f, c.y - ts.y * 0.5f}, Parch(0.9f), l);
        }
        const float al = held ? 1.0f : linked ? 0.65f + 0.35f * std::sin(t * 5.0f) : hov ? 0.7f : 0.0f;
        if (al > 0.0f) {
            if (rect) {
                dl->AddRectFilled({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y},
                                  Gold(0.25f * al), 6 * u);
                dl->AddRect({c.x - half.x, c.y - half.y}, {c.x + half.x, c.y + half.y}, Gold(al),
                            6 * u, 0, 2.0f * u);
            } else {
                for (int k = 3; k >= 1; k--) {
                    dl->AddCircle(c, half.x + k * 4 * u, Gold(0.12f * al), 0, 3.0f * u);
                }
                dl->AddCircleFilled(c, half.x, Gold(0.22f * al));
                dl->AddCircle(c, half.x, Gold(al), 0, 2.0f * u);
            }
        }
        // which PS4 buttons this physical one gives
        std::vector<std::string> gives;
        for (const auto& r : kPadRows) {
            const std::string o = r.output;
            const auto ph_of = PhysOf(o);
            if (std::find(ph_of.begin(), ph_of.end(), p) != ph_of.end()) {
                if (o.starts_with("stick_") && p != "l3" && p != "r3") continue;
                gives.push_back(o);
            }
        }
        if (hov && !gives.empty()) {
            hover = gives[0];
        }
        float gx = c.x + half.x * 0.75f + 4 * u, gy = c.y - half.y * 0.75f - 4 * u;
        for (const auto& o : gives) {
            const bool same = o == p || (o == "stick_left" && p == "l3") ||
                              (o == "stick_right" && p == "r3");
            Glyph(dl, o, {gx, gy}, 14 * u, same ? 0.75f : 1.0f, !same || FocusKey(focus) == o);
            gx += 30 * u;
        }
    }
}

// a glowing link from the game's button to the physical one(s), with a travelling bead
void Link(ImDrawList* dl, ImVec2 p, ImVec2 q, float u) {
    const float dx = std::max(80 * u, std::fabs(q.x - p.x) * 0.45f);
    const ImVec2 c1{p.x + dx, p.y}, c2{q.x - dx, q.y};
    dl->AddBezierCubic(p, c1, c2, q, Gold(0.10f), 9 * u);
    dl->AddBezierCubic(p, c1, c2, q, Gold(0.22f), 4 * u);
    dl->AddBezierCubic(p, c1, c2, q, Gold(0.95f), std::max(1.0f, 1.6f * u));
    for (int k = 0; k < 2; k++) {
        const float tt = std::fmod(Now() * 0.6f + k * 0.5f, 1.0f);
        const float m = 1.0f - tt;
        const ImVec2 bp{m * m * m * p.x + 3 * m * m * tt * c1.x + 3 * m * tt * tt * c2.x + tt * tt * tt * q.x,
                        m * m * m * p.y + 3 * m * m * tt * c1.y + 3 * m * tt * tt * c2.y + tt * tt * tt * q.y};
        dl->AddCircleFilled(bp, 6 * u, Gold(0.18f));
        dl->AddCircleFilled(bp, 3 * u, Col(255, 236, 190, 0.95f));
    }
    dl->AddCircleFilled(p, 5 * u, Gold());
    dl->AddCircleFilled(q, 5 * u, Gold());
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

// a small padlock (locked trophies)
void DrawLock(ImDrawList* dl, ImVec2 c, float s, ImU32 col) {
    const float w = s * 0.62f, h = s * 0.48f;
    dl->PathArcTo({c.x, c.y - h * 0.15f}, w * 0.32f, 3.14159f, 2.0f * 3.14159f, 16);
    dl->PathStroke(col, 0, std::max(1.5f, s * 0.09f));
    dl->AddLine({c.x - w * 0.32f, c.y - h * 0.15f}, {c.x - w * 0.32f, c.y + h * 0.05f}, col,
                std::max(1.5f, s * 0.09f));
    dl->AddLine({c.x + w * 0.32f, c.y - h * 0.15f}, {c.x + w * 0.32f, c.y + h * 0.05f}, col,
                std::max(1.5f, s * 0.09f));
    dl->AddRectFilled({c.x - w * 0.5f, c.y}, {c.x + w * 0.5f, c.y + h}, col, s * 0.06f);
    dl->AddCircleFilled({c.x, c.y + h * 0.42f}, s * 0.06f, Col(10, 9, 10, 0.9f));
}

// Trofeos: progress header (animated bar, per-grade counts), filter chips, a grid of cards
// (large icon, grade badge, name, description, date; locked ones dimmed with a padlock).
int g_trophy_focus = 0;
bool g_trophy_scroll_to = false;
void TrophyScreen(ImVec2 a, ImVec2 b, float u) {
    auto* dl = ImGui::GetWindowDrawList();
    const float t = Now();
    int earned = 0;
    int by_grade[4][2] = {}; // P G S B: earned, total
    for (const auto& tr : g_trophies) {
        earned += tr.earned;
        const int gi = tr.grade == 'P' ? 0 : tr.grade == 'G' ? 1 : tr.grade == 'S' ? 2 : 3;
        by_grade[gi][1]++;
        by_grade[gi][0] += tr.earned;
    }
    const int total = static_cast<int>(g_trophies.size());
    // header card
    const ImVec2 h0 = a, h1{b.x, a.y + 150 * u};
    Panel(dl, h0, h1, u);
    const std::string big = fmt::format("{} / {}", earned, total);
    Txt(dl, g_ft_title, 54 * u, {h0.x + 36 * u, h0.y + 26 * u}, Gold(), big.c_str());
    const float bw0 = TextSize(g_ft_title, 54 * u, big.c_str()).x;
    Txt(dl, g_ft_italic, 26 * u, {h0.x + 36 * u + bw0 + 18 * u, h0.y + 50 * u}, Muted(),
        "trofeos conseguidos");
    const float pct = total > 0 ? 100.0f * earned / total : 0.0f;
    const std::string ps = fmt::format("{:.0f}%", pct * std::min(1.0f, static_cast<float>(t - g_trophy_page_t) / 0.9f));
    TxtR(dl, g_ft_title, 40 * u, h1.x - 36 * u, h0.y + 34 * u, Parch(), ps.c_str());
    {
        const float goal = total > 0 ? static_cast<float>(earned) / total : 0.0f;
        g_trophy_bar = Approach(g_trophy_bar, goal, ImGui::GetIO().DeltaTime, 3.0f);
        const ImVec2 ba{h0.x + 36 * u, h0.y + 100 * u};
        const float bw = (h1.x - h0.x) * 0.55f, bh = 10 * u;
        dl->AddRectFilled(ba, {ba.x + bw, ba.y + bh}, Col(8, 7, 8, 0.95f), bh * 0.5f);
        dl->AddRect(ba, {ba.x + bw, ba.y + bh}, Bronze(0.6f), bh * 0.5f);
        const float fw = bw * g_trophy_bar;
        if (fw > 2.0f) {
            dl->AddRectFilledMultiColor(ba, {ba.x + fw, ba.y + bh}, Col(120, 70, 34),
                                        Col(236, 198, 112), Col(236, 198, 112), Col(120, 70, 34));
            const float sx = ba.x + std::fmod(t * 0.4f, 1.6f) * bw - 0.3f * bw;
            dl->PushClipRect(ba, {ba.x + fw, ba.y + bh}, true);
            dl->AddRectFilledMultiColor({sx, ba.y}, {sx + 0.08f * bw, ba.y + bh}, Col(255, 255, 255, 0),
                                        Col(255, 250, 230, 0.55f), Col(255, 250, 230, 0.55f),
                                        Col(255, 255, 255, 0));
            dl->PopClipRect();
            dl->AddCircleFilled({ba.x + fw, ba.y + bh * 0.5f}, bh * 0.9f, Gold(0.35f));
        }
        // per grade
        float gx = ba.x + bw + 40 * u;
        const char grades[4] = {'P', 'G', 'S', 'B'};
        for (int i = 0; i < 4; i++) {
            if (by_grade[i][1] == 0) continue;
            DrawCup(dl, gx, h0.y + 86 * u, 34 * u, GradeColor(grades[i], g_a), g_a);
            const std::string c = fmt::format("{}/{}", by_grade[i][0], by_grade[i][1]);
            Txt(dl, g_ft_ui, 20 * u, {gx + 40 * u, h0.y + 92 * u}, Parch(0.9f), c.c_str());
            gx += 40 * u + TextSize(g_ft_ui, 20 * u, c.c_str()).x + 30 * u;
        }
    }
    // filter chips
    const char* fl[] = {"Todos", "Conseguidos", "Por conseguir"};
    float fx = a.x;
    const float fy = h1.y + 18 * u;
    for (int i = 0; i < 3; i++) {
        const float w = TextSize(g_ft_ui, 19 * u, fl[i]).x + 34 * u;
        const ImVec2 c0{fx, fy}, c1{fx + w, fy + 38 * u};
        bool hov = false;
        const std::string id = std::string("##tf") + fl[i];
        if (Hit(id.c_str(), c0, c1, &hov)) {
            g_trophy_filter = i;
            g_trophy_focus = 0;
            g_trophy_page_t = t;
        }
        const bool sel = g_trophy_filter == i;
        dl->AddRectFilled(c0, c1, sel ? Col(70, 54, 32, 0.95f) : Col(22, 19, 18, 0.9f), 19 * u);
        dl->AddRect(c0, c1, sel || hov ? Gold() : Bronze(0.6f), 19 * u, 0, std::max(1.0f, u));
        Txt(dl, g_ft_ui, 19 * u, {c0.x + 17 * u, c0.y + 8 * u}, sel ? Gold() : Parch(0.8f), fl[i]);
        fx = c1.x + 12 * u;
    }
    Txt(dl, g_ft_ui, 17 * u, {fx + 14 * u, fy + 10 * u}, Muted(0.8f), "L1 / R1: filtro");
    if (!g_trophy_error.empty()) {
        Txt(dl, g_ft_text, 24 * u, {a.x, fy + 56 * u}, Col(232, 170, 90), g_trophy_error.c_str());
        return;
    }
    // the grid (scrolls; d-pad moves the focus card by card)
    std::vector<Trophy*> list;
    for (auto& tr : g_trophies) {
        if ((g_trophy_filter == 1 && !tr.earned) || (g_trophy_filter == 2 && tr.earned)) continue;
        list.push_back(&tr);
    }
    g_trophy_focus = std::clamp(g_trophy_focus, 0, std::max(0, static_cast<int>(list.size()) - 1));
    const ImVec2 g0{a.x, fy + 56 * u};
    const float gw = b.x - a.x, gh = b.y - g0.y;
    const int cols = gw > 1700 * u ? 3 : 2;
    const float gap = 18 * u;
    const float cw = (gw - gap * (cols - 1) - 14 * u) / cols, ch = 150 * u;
    ImGui::SetCursorScreenPos(g0);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, ImVec4(0.55f, 0.42f, 0.24f, 0.6f * g_a));
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8 * u);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
    ImGui::BeginChild("##tgrid", {gw, gh}, 0,
                      ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollWithMouse);
    SmoothScroll(g_trophy_scroll, u);
    auto* cd = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const int rows = (static_cast<int>(list.size()) + cols - 1) / cols;
    const float icon = ch - 28 * u;
    for (int i = 0; i < static_cast<int>(list.size()); i++) {
        Trophy& tr = *list[i];
        const int r = i / cols, c = i % cols;
        const float ra = EaseOut((t - g_trophy_page_t - std::min(i, 18) * 0.03f) / 0.35f);
        const ImVec2 p{origin.x + c * (cw + gap), origin.y + r * (ch + gap) + (1.0f - ra) * 30 * u};
        const ImVec2 q{p.x + cw, p.y + ch};
        bool hov = false;
        ImGui::PushID(i);
        Hit("##tc", p, q, &hov);
        ImGui::PopID();
        if (hov && MouseMoved()) g_trophy_focus = i;
        const bool on = g_trophy_focus == i;
        if (!ImGui::IsRectVisible(p, q)) continue;
        const float sg = g_a;
        g_a = sg * ra;
        // card
        dl = cd;
        dl->AddRectFilled({p.x + 4 * u, p.y + 6 * u}, {q.x + 4 * u, q.y + 6 * u}, Col(0, 0, 0, 0.35f),
                          8 * u);
        dl->AddRectFilledMultiColor(p, q, tr.earned ? Col(40, 33, 26, 0.95f) : Col(22, 20, 21, 0.92f),
                                    tr.earned ? Col(40, 33, 26, 0.95f) : Col(22, 20, 21, 0.92f),
                                    Col(14, 13, 13, 0.95f), Col(14, 13, 13, 0.95f));
        if (on) {
            for (int k = 3; k >= 1; k--) {
                dl->AddRect({p.x - k * 3 * u, p.y - k * 3 * u}, {q.x + k * 3 * u, q.y + k * 3 * u},
                            Gold(0.10f / k * 2), 8 * u + k * 3 * u, 0, 2 * u);
            }
        }
        dl->AddRect(p, q, on ? Gold(0.95f) : tr.earned ? Bronze(0.7f) : Bronze(0.3f), 8 * u, 0,
                    std::max(1.0f, (on ? 1.8f : 1.0f) * u));
        if (tr.earned) {
            dl->AddRectFilled(p, {p.x + 4 * u, q.y}, GradeColor(tr.grade, g_a), 8 * u,
                              ImDrawFlags_RoundCornersLeft);
        }
        // icon in a grade-coloured frame
        const ImVec2 ip{p.x + 16 * u, p.y + 14 * u};
        ImTextureID tex{};
        if (!tr.custom) {
            tex = Get(tr.icon, g_icons_dir / fmt::format("TROP{:03d}.PNG", tr.id), 240, 240);
        }
        dl->AddRectFilled({ip.x - 3 * u, ip.y - 3 * u}, {ip.x + icon + 3 * u, ip.y + icon + 3 * u},
                          tr.earned ? GradeColor(tr.grade, 0.8f * g_a) : Col(60, 56, 54, 0.8f), 6 * u);
        if (tex) {
            dl->AddImage(tex, ip, {ip.x + icon, ip.y + icon}, {0, 0}, {1, 1},
                         tr.earned ? Col(255, 255, 255) : Col(90, 88, 92, 0.9f));
        } else if (tr.custom) {
            DrawCustomIcon(dl, ip.x, ip.y, icon, (tr.earned ? 1.0f : 0.35f) * g_a);
        } else {
            dl->AddRectFilled(ip, {ip.x + icon, ip.y + icon}, Col(30, 28, 30), 4 * u);
            DrawCup(dl, ip.x + icon * 0.15f, ip.y + icon * 0.12f, icon * 0.7f,
                    GradeColor(tr.grade, (tr.earned ? 1.0f : 0.35f) * g_a), g_a);
        }
        if (!tr.earned) {
            dl->AddRectFilled(ip, {ip.x + icon, ip.y + icon}, Col(0, 0, 0, 0.35f), 4 * u);
            dl->AddCircleFilled({ip.x + icon * 0.5f, ip.y + icon * 0.5f}, icon * 0.22f,
                                Col(10, 9, 10, 0.75f));
            DrawLock(dl, {ip.x + icon * 0.5f, ip.y + icon * 0.46f}, icon * 0.3f, Parch(0.85f));
        } else if (on) {
            // shine sweep over the focused, earned icon
            const float st = std::fmod(t * 0.5f, 1.6f);
            if (st < 1.0f) {
                const float sx = ip.x - icon * 0.4f + st * icon * 1.8f;
                dl->PushClipRect(ip, {ip.x + icon, ip.y + icon}, true);
                dl->AddQuadFilled({sx, ip.y}, {sx + icon * 0.22f, ip.y},
                                  {sx - icon * 0.18f, ip.y + icon}, {sx - icon * 0.4f, ip.y + icon},
                                  Col(255, 248, 225, 0.35f));
                dl->PopClipRect();
            }
        }
        // text
        const float tx = ip.x + icon + 20 * u, tw = q.x - tx - 16 * u;
        DrawCup(dl, tx, p.y + 14 * u, 22 * u, GradeColor(tr.grade, (tr.earned ? 1.0f : 0.6f) * g_a), g_a);
        Txt(dl, g_ft_ui, 17 * u, {tx + 28 * u, p.y + 16 * u},
            GradeColor(tr.grade, (tr.earned ? 1.0f : 0.6f) * g_a), GradeName(tr.grade));
        const std::string st = tr.earned ? DateText(tr.stamp) : std::string("Bloqueado");
        TxtR(dl, g_ft_ui, 16 * u, q.x - 16 * u, p.y + 17 * u,
             tr.earned ? Col(170, 206, 150) : Muted(0.8f), st.c_str());
        const bool secret = tr.hidden && !tr.earned;
        const std::string nm = secret ? std::string("Trofeo oculto") : tr.name;
        const ImVec4 clip{tx, p.y, q.x - 12 * u, q.y};
        dl->AddText(FT(g_ft_title), 25 * u, {tx, p.y + 44 * u}, tr.earned ? Parch() : Col(200, 192, 180, 0.8f),
                    nm.c_str(), nullptr, 0.0f, &clip);
        const std::string det = secret ? std::string("Sigue jugando para descubrirlo.")
                                       : tr.custom ? tr.detail + " (del port)" : tr.detail;
        const auto lines = Wrap(g_ft_text, 22 * u, det, tw);
        for (size_t k = 0; k < lines.size() && k < 2; k++) {
            std::string l = lines[k];
            if (k == 1 && lines.size() > 2) l += "\xe2\x80\xa6";
            Txt(dl, g_ft_text, 22 * u, {tx, p.y + 80 * u + k * 25 * u}, Muted(tr.earned ? 1.0f : 0.75f),
                l.c_str());
        }
        g_a = sg;
    }
    // scroll extents
    ImGui::SetCursorScreenPos({origin.x, origin.y + rows * (ch + gap)});
    ImGui::Dummy({1, 1});
    if (g_trophy_scroll_to && !list.empty()) {
        g_trophy_scroll_to = false;
        const float y0 = (g_trophy_focus / cols) * (ch + gap);
        const float cur = ImGui::GetScrollY();
        if (y0 < cur) {
            g_trophy_scroll = y0;
        } else if (y0 + ch > cur + gh) {
            g_trophy_scroll = std::min(ImGui::GetScrollMaxY(), y0 + ch - gh + gap);
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(3);
}
int TrophyCols(float gw, float u) {
    return gw > 1700 * u ? 3 : 2;
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

// (bb-menu-121) The trophy card: slides in from the top right with a soft overshoot, a frame
// glowing in the grade's colour, the icon pops in with a burst of sparks and a shine sweep,
// "TROFEO OBTENIDO" in Cinzel, the name in Cormorant Garamond. Slides back out after 6 s.
void DrawToast(float u) {
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
    const float saved = g_a;
    const float in = EaseOutBack(t / 0.55f);
    const float out = EaseInOut(std::clamp((t - (kToastSeconds - 0.5f)) / 0.5f, 0.0f, 1.0f));
    g_a = std::clamp(t / 0.25f, 0.0f, 1.0f) * (1.0f - out);
    if (g_a <= 0.0f) {
        g_a = saved;
        return;
    }
    auto* dl = ImGui::GetForegroundDrawList();
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    const ImU32 gc = GradeColor(toast.grade, g_a);
    const auto GC = [&](float a) { return GradeColor(toast.grade, std::clamp(a * g_a, 0.0f, 1.0f)); };
    const float icon = 104 * u, pad = 20 * u;
    const float name_px = 34 * u, head_px = 21 * u, small_px = 19 * u;
    const float text_w = std::max({TextSize(g_ft_text, name_px, toast.name.c_str()).x,
                                   SpacedW(g_ft_title, head_px, "TROFEO OBTENIDO", 3 * u) + 0,
                                   toast.detail.empty() ? 0.0f
                                                        : TextSize(g_ft_text, small_px + 3 * u,
                                                                   toast.detail.c_str()).x});
    const float w = std::clamp(pad * 3 + icon + text_w + 10 * u, 520 * u, disp.x * 0.42f);
    const float h = icon + pad * 2;
    const float margin = 36 * u;
    const float x0 = disp.x - margin - w + (1.0f - in) * (w + margin) + out * (w + margin) * 0.6f;
    const float y0 = 36 * u - (1.0f - std::min(1.0f, t / 0.4f)) * 14 * u;
    const ImVec2 a{x0, y0}, b{x0 + w, y0 + h};
    const float r = 8 * u;
    // shadow + grade glow (pulses a few times, then rests)
    for (int i = 4; i >= 1; i--) {
        const float g = i * 8.0f * u;
        dl->AddRectFilled({a.x - g * 0.4f, a.y + g * 0.3f}, {b.x + g * 0.4f, b.y + g}, Col(0, 0, 0, 0.09f),
                          r + g);
    }
    const float pulse = t < 3.2f ? 0.55f + 0.45f * std::sin(t * 4.0f) : 0.45f;
    for (int i = 5; i >= 1; i--) {
        const float g = i * 3.5f * u;
        dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g}, GC(0.09f * pulse * 6.0f / (i + 1)),
                    r + g, 0, 3.0f * u);
    }
    dl->AddRectFilledMultiColor(a, b, Col(34, 28, 24, 0.97f), Col(24, 20, 19, 0.97f),
                                Col(12, 11, 12, 0.97f), Col(16, 14, 14, 0.97f));
    dl->AddRect(a, b, gc, r, 0, std::max(1.0f, 1.6f * u));
    dl->AddRect({a.x + 6 * u, a.y + 6 * u}, {b.x - 6 * u, b.y - 6 * u}, Gold(0.16f), r * 0.6f, 0, 1.0f);
    // a light sweeping over the whole card once it lands
    if (t > 0.45f && t < 1.35f) {
        const float st = (t - 0.45f) / 0.9f;
        const float sx = a.x - w * 0.3f + st * w * 1.6f;
        dl->PushClipRect(a, b, true);
        dl->AddRectFilledMultiColor({sx - w * 0.12f, a.y}, {sx, b.y}, Col(255, 240, 200, 0.0f),
                                    Col(255, 240, 200, 0.10f), Col(255, 240, 200, 0.10f),
                                    Col(255, 240, 200, 0.0f));
        dl->AddRectFilledMultiColor({sx, a.y}, {sx + w * 0.12f, b.y}, Col(255, 240, 200, 0.10f),
                                    Col(255, 240, 200, 0.0f), Col(255, 240, 200, 0.0f),
                                    Col(255, 240, 200, 0.10f));
        dl->PopClipRect();
    }
    // icon: pop with overshoot, frame in the grade's colour, sparks, shine
    const float cx = a.x + pad + icon * 0.5f, cy = a.y + pad + icon * 0.5f;
    const float pt = std::clamp((t - 0.18f) / 0.5f, 0.0f, 1.0f);
    const float pop = t < 0.18f ? 0.0f : EaseOutBack(pt);
    if (pop > 0.01f) {
        const float hs = icon * 0.5f * pop;
        const ImVec2 q0{cx - hs, cy - hs}, q1{cx + hs, cy + hs};
        for (int i = 4; i >= 1; i--) {
            const float g = hs + i * 4.0f * u;
            dl->AddRectFilled({cx - g, cy - g}, {cx + g, cy + g}, GC(0.07f * pulse), 6 * u + i * 3 * u);
        }
        dl->AddRectFilled({q0.x - 3 * u, q0.y - 3 * u}, {q1.x + 3 * u, q1.y + 3 * u}, GC(0.9f), 5 * u);
        const ImTextureID tex =
            toast.custom ? ImTextureID{} : Get(toast.icon, toast.icon_path, 240, 240);
        if (toast.custom) {
            DrawCustomIcon(dl, q0.x, q0.y, hs * 2.0f, g_a);
        } else if (tex) {
            dl->AddImage(tex, q0, q1, {0, 0}, {1, 1}, Col(255, 255, 255));
        } else {
            dl->AddRectFilled(q0, q1, Col(30, 28, 30), 4 * u);
            DrawCup(dl, q0.x + hs * 0.25f, q0.y + hs * 0.2f, hs * 1.5f, gc, g_a);
        }
        if (t > 0.7f && t < 1.5f) {
            const float st = (t - 0.7f) / 0.8f;
            const float sx = q0.x - hs + st * hs * 4.0f;
            dl->PushClipRect(q0, q1, true);
            dl->AddQuadFilled({sx, q0.y}, {sx + hs * 0.35f, q0.y}, {sx - hs * 0.25f, q1.y},
                              {sx - hs * 0.6f, q1.y}, Col(255, 248, 225, 0.55f));
            dl->PopClipRect();
        }
        // sparks
        if (t > 0.2f && t < 1.4f) {
            const float st = (t - 0.2f) / 1.2f;
            for (int k = 0; k < 14; k++) {
                const float ang = k * 0.4488f + 0.3f;
                const float dist = icon * (0.45f + 0.55f * EaseOut(st)) * (0.8f + 0.25f * ((k * 7) % 5) / 4.0f);
                const ImVec2 sp{cx + std::cos(ang) * dist, cy + std::sin(ang) * dist};
                dl->AddCircleFilled(sp, (2.6f - 1.6f * st) * u, GC(1.0f - st));
            }
        }
    }
    // text
    const float tx = a.x + pad * 2 + icon, ty = a.y + pad + 4 * u;
    const float ha = std::clamp((t - 0.25f) / 0.35f, 0.0f, 1.0f);
    const float slide = (1.0f - EaseOut(ha)) * 16 * u;
    TxtSpaced(dl, g_ft_title, head_px, {tx + slide, ty}, GC(ha), "TROFEO OBTENIDO", 3 * u, false);
    {
        const float lw = std::min(w - (tx - a.x) - pad, 260 * u) * EaseOut(ha);
        dl->AddRectFilledMultiColor({tx, ty + head_px + 8 * u}, {tx + lw, ty + head_px + 9.5f * u},
                                    GC(0.8f), GC(0.0f), GC(0.0f), GC(0.8f));
    }
    const float na = std::clamp((t - 0.35f) / 0.35f, 0.0f, 1.0f);
    const ImVec4 clip{a.x, a.y, b.x - pad, b.y};
    dl->AddText(FT(g_ft_text), name_px, {tx + (1.0f - EaseOut(na)) * 20 * u, ty + head_px + 16 * u},
                Parch(na), toast.name.c_str(), nullptr, 0.0f, &clip);
    const std::string sub = toast.detail.empty() ? std::string(GradeName(toast.grade))
                                                 : std::string(GradeName(toast.grade)) + "  \xc2\xb7  " + toast.detail;
    dl->AddText(FT(g_ft_ui), small_px, {tx, ty + head_px + 20 * u + name_px}, Muted(na),
                sub.c_str(), nullptr, 0.0f, &clip);
    // progress line along the bottom
    const float life = std::clamp(t / kToastSeconds, 0.0f, 1.0f);
    dl->AddRectFilled({a.x + r, b.y - 3 * u}, {a.x + r + (w - 2 * r) * (1.0f - life), b.y - 1.5f * u},
                      GC(0.55f));
    g_a = saved;
}

// ---------------------------------------------------------------- custom trophies (not PSN)
// (bb-trophy-134) Three port trophies driven by guest map loads. Kernel open() calls
// BbMenu::OnGuestFileOpen; a map load is the open of /app0/dvdroot_ps4/event/mAA_BB_CC_DD.emevd.dcx
// (one per load in shad_log.txt, seen for m27_00_00_00 / m32_00_00_00). Map ids (public
// FromSoft modding lists): m24_01_00_00 = Central Yharnam (Iosefka's Clinic is inside this
// map, not a separate one), m21_00_00_00 = Hunter's Dream. "New game" = a new save file appears
// (as 1.17). No verified event-flag offsets for CUSA03173 1.09 here, so:
//  - Despertar en Yharnam: kYharnamSeconds of time spent in m24_01 after the new game (map load
//    alone cannot tell the clinic from the street); old 7-minute rule if no map load is ever seen.
//  - El primer sueño: first m21_00 load after the new game (exact).
// (bb-trophy-134b) New game = a character slot /savedata0/userdata000N written before any map
// load and before that slot was read (since boot or the last system-data userdata0010 read).
// Every new game re-arms the trophies not yet unlocked; progress counts only while that slot
// is the active character.
//  - Ni la bestia, ni el miedo: HEURISTIC: the first m21_00 load comes after at least
//    BB_BEAST_HEUR_SECONDS (default 0 = off and hidden from the grid; e.g. 240 to enable) in m24_01; dying to the beast is quick,
//    killing it unarmed is not.
struct CustomDef {
    const char* key;
    const char* name;
    const char* detail;
    char grade;
};
constexpr CustomDef kCustom[3] = {
    {"despertar_en_yharnam", "Despertar en Yharnam",
     "Sal de la Clínica de Iosefka por primera vez y comienza la cacería.", 'B'},
    {"el_primer_sueno", "El primer sueño",
     "Muere ante la bestia de la clínica y despierta en el Sueño del Cazador.", 'B'},
    {"ni_la_bestia_ni_el_miedo", "Ni la bestia, ni el miedo",
     "Derrota a la bestia de la clínica antes de despertar en el Sueño del Cazador.", 'S'},
};
constexpr int kCustomPlaySeconds = 420; // fallback, no map load seen
constexpr int kYharnamSeconds = 420;
constexpr const char* kMapYharnam = "m24_01_00_00";
constexpr const char* kMapDream = "m21_00_00_00";
struct CustomState {
    bool armed = false;
    bool unlocked[3]{};
    u64 stamp[3]{};
    int seconds = 0;          // since armed
    int yharnam_seconds = 0;  // in m24_01 since armed
    bool yharnam_seen = false, dream_seen = false;
    int slot = -1; // character slot of this run (userdataNNNN), -1 = unknown
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
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto section = [&](const char* sec) -> std::string {
        const auto p = all.find(fmt::format("\"{}\"", sec));
        if (p == std::string::npos) {
            return {};
        }
        const auto e = all.find('}', p);
        return all.substr(p, e == std::string::npos ? std::string::npos : e - p);
    };
    const auto num = [](const std::string& j, const char* k) -> u64 {
        const auto p = j.find(k);
        if (p == std::string::npos) {
            return 0;
        }
        const auto c = j.find(':', p);
        return c == std::string::npos ? 0 : std::strtoull(j.c_str() + c + 1, nullptr, 10);
    };
    const auto flag = [](const std::string& j, const char* k) {
        const auto p = j.find(k);
        if (p == std::string::npos) {
            return false;
        }
        const auto c = j.find(':', p);
        return c != std::string::npos && j.find("true", c) == j.find_first_not_of(" \t", c + 1);
    };
    for (int i = 0; i < 3; i++) {
        const std::string j = section(kCustom[i].key);
        s.unlocked[i] = flag(j, "\"unlocked\"");
        s.stamp[i] = num(j, "\"timestamp\"");
    }
    std::string pr = section("progress");
    if (pr.empty()) {
        pr = section(kCustom[0].key); // 1.17-1.33 file: armed / play_seconds lived there
    }
    s.armed = flag(pr, "\"armed\"");
    s.seconds = static_cast<int>(num(pr, "\"play_seconds\""));
    s.yharnam_seconds = static_cast<int>(num(pr, "\"yharnam_seconds\""));
    s.yharnam_seen = flag(pr, "\"yharnam_seen\"");
    s.dream_seen = flag(pr, "\"dream_seen\"");
    s.slot = pr.find("\"slot\"") == std::string::npos ? -1
                                                     : static_cast<int>(num(pr, "\"slot\"")) - 1;
    return s;
}
void WriteCustom(const CustomState& s) {
    std::scoped_lock lk{g_custom_mtx};
    std::error_code ec;
    const auto p = CustomJson();
    std::filesystem::create_directories(p.parent_path(), ec);
    const auto date = [](u64 st) -> std::string {
        if (st == 0) {
            return {};
        }
        std::time_t t = static_cast<std::time_t>(st);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
        return buf;
    };
    std::string j = "{\n";
    for (int i = 0; i < 3; i++) {
        j += fmt::format("  \"{}\": {{\n    \"name\": \"{}\",\n    \"grade\": \"{}\",\n"
                         "    \"unlocked\": {},\n    \"timestamp\": {},\n    \"date\": \"{}\"\n  }},\n",
                         kCustom[i].key, kCustom[i].name, kCustom[i].grade,
                         s.unlocked[i] ? "true" : "false", s.stamp[i], date(s.stamp[i]));
    }
    j += fmt::format("  \"progress\": {{\n    \"armed\": {},\n    \"play_seconds\": {},\n"
                     "    \"yharnam_seconds\": {},\n    \"yharnam_seen\": {},\n"
                     "    \"dream_seen\": {},\n    \"slot\": {}\n  }}\n}}\n",
                     s.armed ? "true" : "false", s.seconds, s.yharnam_seconds,
                     s.yharnam_seen ? "true" : "false", s.dream_seen ? "true" : "false",
                     s.slot + 1); // stored +1 (0 = unknown)
    const auto tmp = std::filesystem::path(p).concat(".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << j;
    }
    std::filesystem::rename(tmp, p, ec);
}
std::atomic<bool> g_custom_changed{false};

// map loads from the guest open() hook, consumed by CustomWatcher
struct MapLoad {
    std::string id;
    std::chrono::steady_clock::time_point at;
};
std::mutex g_map_mtx;
std::deque<MapLoad> g_map_loads;
std::string g_last_map; // last map loaded this session (any time)
std::chrono::steady_clock::time_point g_last_map_at{};
std::atomic<int> g_map_parts_logged{0};
// (bb-trophy-134b) new game from the save container: Bloodborne keeps one container
// (/savedata0 = .../savedata/<title>/SPRJ0005) with character slots userdata0000..0009
// (0010 = system data). A new game writes a slot before any map load and without reading
// that slot first; loading an existing character reads its slot (flags 0) first.
std::atomic<int> g_newgame_slot{-1};
std::atomic<bool> g_any_map_session{false};
std::atomic<unsigned> g_slot_read_mask{0};
std::atomic<int> g_slot_logged{0};
std::atomic<int> g_active_slot{-1}; // last character slot read or written

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
void UnlockCustom(CustomState& s, int i, const char* reason);
int BeastHeurSeconds();
bool AllUnlocked(const CustomState& s) {
    return s.unlocked[0] && s.unlocked[1] && (s.unlocked[2] || BeastHeurSeconds() == 0);
}
int BeastHeurSeconds() {
    if (const char* v = std::getenv("BB_BEAST_HEUR_SECONDS"); v != nullptr && v[0] != 0) {
        return std::max(0, std::atoi(v));
    }
    return 0; // off by default (heuristic); hidden from the grid while off
}
void CustomWatcher() {
    CustomState s = ReadCustom();
    if (AllUnlocked(s)) {
        LOG_INFO(Frontend, "[bbtrophy] all custom trophies already unlocked");
        return;
    }
    if (const char* v = std::getenv("BB_CUSTOM_TROPHY_TEST"); v != nullptr && v[0] == '1') {
        for (int i = 0; i < 3; i++) {
            if (i == 2 && BeastHeurSeconds() == 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(i == 0 ? 20 : 8));
            if (!s.unlocked[i]) {
                UnlockCustom(s, i, "BB_CUSTOM_TROPHY_TEST=1");
            }
        }
        return;
    }
    if (s.armed && s.slot < 0) {
        // a 1.17-1.34 run armed by the old file rule: no slot, cannot be trusted
        LOG_INFO(Frontend, "[bbtrophy] old run without a character slot dropped; next new game re-arms");
        s.armed = false;
        WriteCustom(s);
    }
    const int beast_heur = BeastHeurSeconds();
    auto known = ScanSaves();
    LOG_INFO(Frontend,
             "[bbtrophy] watching {} save files (armed {}, {} s, {} s in {}, yharnam_seen {}, "
             "dream_seen {}, beast heuristic {} s)",
             known.size(), s.armed, s.seconds, s.yharnam_seconds, kMapYharnam, s.yharnam_seen,
             s.dream_seen, beast_heur);
    std::string cur_map;
    bool any_map = s.yharnam_seen || s.dream_seen;
    int since_write = 0;
    const auto on_map = [&](const std::string& id) {
        cur_map = id;
        any_map = true;
        if (!s.armed) {
            return;
        }
        if (id == kMapYharnam && !s.yharnam_seen) {
            s.yharnam_seen = true;
            LOG_INFO(Frontend, "[bbtrophy] new game: first {} (Central Yharnam) load", id);
            WriteCustom(s);
        }
        if (id == kMapDream && !s.dream_seen) {
            s.dream_seen = true;
            LOG_INFO(Frontend,
                     "[bbtrophy] new game: first {} (Hunter's Dream) load after {} s in {}", id,
                     s.yharnam_seconds, kMapYharnam);
            if (!s.unlocked[1]) {
                UnlockCustom(s, 1, "first Hunter's Dream load after a new game");
            }
            if (!s.unlocked[2] && beast_heur > 0) {
                if (beast_heur > 0 && s.yharnam_seconds >= beast_heur) {
                    UnlockCustom(s, 2,
                                 "HEURISTIC: first Hunter's Dream reached after >= "
                                 "BB_BEAST_HEUR_SECONDS in Central Yharnam");
                } else {
                    LOG_INFO(Frontend,
                             "[bbtrophy] 'Ni la bestia, ni el miedo' not given: {} s < {} s "
                             "(heuristic, beast assumed not killed)",
                             s.yharnam_seconds, beast_heur);
                }
            }
            WriteCustom(s);
        }
    };
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::deque<MapLoad> loads;
        {
            std::scoped_lock lk{g_map_mtx};
            loads.swap(g_map_loads);
        }
        // a new game re-arms every not-yet-unlocked trophy, whatever earlier runs left
        // (the 1.17 "new save file" rule is gone: Bloodborne rewrites slots/backups in one
        // container, so new files are not new games; the slot-write rule replaces it)
        std::vector<std::string> added;
        const int ng_slot = g_newgame_slot.exchange(-1);
        if (ng_slot >= 0) {
            added.insert(added.begin(),
                         fmt::format("/savedata0/userdata{:04d} (new character slot write)", ng_slot));
        }
        if (!added.empty()) {
            CustomState fresh;
            for (int i = 0; i < 3; i++) {
                fresh.unlocked[i] = s.unlocked[i];
                fresh.stamp[i] = s.stamp[i];
            }
            s = fresh;
            s.armed = true;
            s.slot = ng_slot;
            LOG_INFO(Frontend,
                     "[bbtrophy] new game ({}): armed, per-run progress reset (slot {}, current "
                     "map '{}', unlocked: {} {} {})",
                     added[0], ng_slot, cur_map, s.unlocked[0], s.unlocked[1], s.unlocked[2]);
            cur_map.clear();
            WriteCustom(s);
        }
        // progress counts only while the run's own character is the active one
        const int active = g_active_slot.load();
        const bool own = s.slot >= 0 && active == s.slot;
        for (const auto& m : loads) {
            if (s.armed && own) {
                on_map(m.id);
            } else {
                cur_map = m.id;
                any_map = true;
            }
        }
        if (!s.armed || !own) {
            continue;
        }
        s.seconds += 5;
        if (cur_map == kMapYharnam) {
            s.yharnam_seconds += 5;
        }
        since_write += 5;
        if (!s.unlocked[0]) {
            if (s.yharnam_seen && s.yharnam_seconds >= kYharnamSeconds) {
                UnlockCustom(s, 0, "time in Central Yharnam (m24_01) after a new game");
            } else if (!any_map && s.seconds >= kCustomPlaySeconds) {
                UnlockCustom(s, 0, "fallback: no map load seen, 7 min after new save data");
            }
        }
        if (AllUnlocked(s)) {
            WriteCustom(s);
            LOG_INFO(Frontend, "[bbtrophy] all custom trophies unlocked, watcher done");
            return;
        }
        if (since_write >= 30) {
            since_write = 0;
            WriteCustom(s);
        }
    }
}

void UnlockCustom(CustomState& s, int i, const char* reason) {
    s.unlocked[i] = true;
    s.stamp[i] = static_cast<u64>(std::time(nullptr));
    WriteCustom(s);
    LOG_INFO(Frontend, "[bbtrophy] unlocked '{}' ({}): {}", kCustom[i].name,
             GradeName(kCustom[i].grade), reason);
    g_custom_changed = true;
    if (const char* v = std::getenv("BB_TROPHY_POPUP"); v != nullptr && v[0] == '0') {
        return;
    }
    std::scoped_lock lk{g_toast_mtx};
    Toast t;
    t.name = kCustom[i].name;
    t.grade = kCustom[i].grade;
    t.detail = kCustom[i].detail;
    t.custom = true;
    g_toasts.push_back(std::move(t));
}
void AddCustomTrophy() {
    const CustomState s = ReadCustom();
    const int n = BeastHeurSeconds() > 0 ? 3 : 2; // heuristic trophy only when enabled
    for (int i = 0; i < n; i++) {
        Trophy t;
        t.id = -100 - i;
        t.grade = kCustom[i].grade;
        t.custom = true;
        t.earned = s.unlocked[i];
        t.stamp = s.stamp[i];
        t.name = kCustom[i].name;
        t.detail = kCustom[i].detail;
        g_trophies.push_back(std::move(t));
    }
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
// (bb-menu-121) F1 opens a full-screen hub (Mando, Teclado y ratón, Trofeos, Opciones, Salir
// del juego); a card opens only its own screen, Volver (Circle / Esc) goes back to the hub.
enum class Screen { Hub = -1, Pad = 0, Kbm = 1, Trophies = 2, Options = 3, Exit = 4 };
constexpr int kCards = 5;

// hub icons, gold line art
void IconPad(ImDrawList* dl, ImVec2 c, float s, ImU32 col) {
    const float th = std::max(1.5f, s * 0.035f);
    dl->AddRect({c.x - 0.40f * s, c.y - 0.20f * s}, {c.x + 0.40f * s, c.y + 0.10f * s}, col,
                0.14f * s, 0, th);
    dl->AddCircle({c.x - 0.30f * s, c.y + 0.12f * s}, 0.15f * s, col, 24, th);
    dl->AddCircle({c.x + 0.30f * s, c.y + 0.12f * s}, 0.15f * s, col, 24, th);
    dl->AddRectFilled({c.x - 0.36f * s, c.y - 0.08f * s}, {c.x - 0.20f * s, c.y - 0.04f * s}, col);
    dl->AddRectFilled({c.x - 0.30f * s, c.y - 0.14f * s}, {c.x - 0.26f * s, c.y + 0.02f * s}, col);
    for (int i = 0; i < 4; i++) {
        const float ang = i * 1.5708f;
        dl->AddCircleFilled({c.x + 0.28f * s + std::cos(ang) * 0.065f * s,
                             c.y - 0.06f * s + std::sin(ang) * 0.065f * s},
                            0.025f * s, col);
    }
    dl->AddCircle({c.x - 0.11f * s, c.y + 0.03f * s}, 0.06f * s, col, 16, th);
    dl->AddCircle({c.x + 0.11f * s, c.y + 0.03f * s}, 0.06f * s, col, 16, th);
}
void IconKbm(ImDrawList* dl, ImVec2 c, float s, ImU32 col) {
    const float th = std::max(1.5f, s * 0.035f);
    const ImVec2 k0{c.x - 0.46f * s, c.y - 0.12f * s}, k1{c.x + 0.16f * s, c.y + 0.20f * s};
    dl->AddRect(k0, k1, col, 0.04f * s, 0, th);
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 6; i++) {
            const float x = k0.x + 0.05f * s + i * 0.095f * s, y = k0.y + 0.05f * s + r * 0.075f * s;
            dl->AddRectFilled({x, y}, {x + 0.06f * s, y + 0.045f * s}, col, 0.01f * s);
        }
    }
    dl->AddRect({c.x + 0.24f * s, c.y - 0.14f * s}, {c.x + 0.46f * s, c.y + 0.22f * s}, col,
                0.11f * s, 0, th);
    dl->AddLine({c.x + 0.35f * s, c.y - 0.14f * s}, {c.x + 0.35f * s, c.y - 0.02f * s}, col, th);
}
void IconGear(ImDrawList* dl, ImVec2 c, float s, ImU32 col, float rot) {
    const float th = std::max(1.5f, s * 0.035f);
    dl->AddCircle(c, 0.24f * s, col, 32, th);
    dl->AddCircle(c, 0.10f * s, col, 24, th);
    for (int i = 0; i < 8; i++) {
        const float a = rot + i * 0.7854f;
        const float ca = std::cos(a), sa = std::sin(a), w = 0.05f * s;
        const ImVec2 p0{c.x + ca * 0.24f * s, c.y + sa * 0.24f * s};
        const ImVec2 p1{c.x + ca * 0.34f * s, c.y + sa * 0.34f * s};
        dl->AddQuadFilled({p0.x - sa * w, p0.y + ca * w}, {p1.x - sa * w * 0.8f, p1.y + ca * w * 0.8f},
                          {p1.x + sa * w * 0.8f, p1.y - ca * w * 0.8f}, {p0.x + sa * w, p0.y - ca * w},
                          col);
    }
}
void IconDoor(ImDrawList* dl, ImVec2 c, float s, ImU32 col) {
    const float th = std::max(1.5f, s * 0.035f);
    const float w = 0.20f * s, top = c.y - 0.10f * s, bot = c.y + 0.30f * s, x = c.x - 0.08f * s;
    dl->PathArcTo({x, top}, w, 3.14159f, 2 * 3.14159f, 24);
    dl->PathLineTo({x + w, bot});
    dl->PathLineTo({x - w, bot});
    dl->PathStroke(col, ImDrawFlags_Closed, th);
    dl->AddLine({x, top - w}, {x, bot}, col, th * 0.7f);
    dl->AddCircleFilled({x + w * 0.5f, c.y + 0.10f * s}, 0.025f * s, col);
    // the way out
    dl->AddLine({c.x + 0.10f * s, c.y + 0.08f * s}, {c.x + 0.40f * s, c.y + 0.08f * s}, col, th);
    dl->AddTriangleFilled({c.x + 0.44f * s, c.y + 0.08f * s}, {c.x + 0.34f * s, c.y + 0.01f * s},
                          {c.x + 0.34f * s, c.y + 0.15f * s}, col);
}

// custom drawn controls
bool Toggle(const char* id, ImVec2 p, bool on, float& anim, float u, bool focus) {
    const ImVec2 q{p.x + 68 * u, p.y + 34 * u};
    bool hov = false;
    const bool r = Hit(id, p, q, &hov);
    anim = Approach(anim, on ? 1.0f : 0.0f, ImGui::GetIO().DeltaTime, 12.0f);
    auto* dl = ImGui::GetWindowDrawList();
    if (focus || hov) {
        dl->AddRect({p.x - 4 * u, p.y - 4 * u}, {q.x + 4 * u, q.y + 4 * u}, Gold(0.5f), 21 * u, 0,
                    2 * u);
    }
    dl->AddRectFilled(p, q, anim > 0.5f ? Col(96, 72, 38, 0.95f) : Col(30, 27, 26, 0.95f), 17 * u);
    dl->AddRect(p, q, Bronze(0.9f), 17 * u, 0, std::max(1.0f, u));
    const float kx = p.x + 17 * u + anim * 34 * u;
    dl->AddCircleFilled({kx, p.y + 17 * u}, 12 * u, anim > 0.5f ? Gold() : Col(150, 140, 128));
    return r;
}
// returns true when the value was committed (mouse released or a step)
bool Slider(const char* id, ImVec2 p, float w, float& v, float mn, float mx, float u, bool focus) {
    const float h = 34 * u;
    ImGui::SetCursorScreenPos(p);
    ImGui::InvisibleButton(id, {w, h});
    const bool active = ImGui::IsItemActive(), hov = ImGui::IsItemHovered();
    if (active) {
        const float k = std::clamp((ImGui::GetIO().MousePos.x - p.x) / w, 0.0f, 1.0f);
        v = mn + (mx - mn) * k;
    }
    const bool commit = ImGui::IsItemDeactivated();
    auto* dl = ImGui::GetWindowDrawList();
    const float k = std::clamp((v - mn) / (mx - mn), 0.0f, 1.0f);
    const float cy = p.y + h * 0.5f;
    dl->AddRectFilled({p.x, cy - 3 * u}, {p.x + w, cy + 3 * u}, Col(8, 7, 8, 0.95f), 3 * u);
    dl->AddRectFilledMultiColor({p.x, cy - 3 * u}, {p.x + w * k, cy + 3 * u}, Col(120, 70, 34),
                                Col(236, 198, 112), Col(236, 198, 112), Col(120, 70, 34));
    if (focus || hov || active) {
        dl->AddCircleFilled({p.x + w * k, cy}, 18 * u, Gold(0.18f));
    }
    dl->AddCircleFilled({p.x + w * k, cy}, 11 * u, focus || active ? Gold() : Col(200, 186, 160));
    dl->AddCircle({p.x + w * k, cy}, 11 * u, Col(40, 30, 20), 0, std::max(1.0f, u));
    return commit;
}
// a framed text button; returns clicked
bool CardButton(const char* id, ImVec2 a, ImVec2 b, const char* label, float u, bool focus,
                bool danger = false) {
    bool hov = false;
    const bool r = Hit(id, a, b, &hov);
    auto* dl = ImGui::GetWindowDrawList();
    const bool on = focus || hov;
    const float t = Now();
    if (on) {
        const float pulse = 0.7f + 0.3f * std::sin(t * 5.0f);
        for (int i = 3; i >= 1; i--) {
            const float g = i * 3.0f * u;
            dl->AddRect({a.x - g, a.y - g}, {b.x + g, b.y + g},
                        danger ? Col(200, 60, 50, 0.12f * pulse) : Gold(0.12f * pulse), 6 * u + g, 0,
                        2 * u);
        }
    }
    const ImU32 top = danger ? Col(92, 24, 22, 0.95f) : Col(54, 44, 32, 0.95f);
    const ImU32 bot = danger ? Col(40, 12, 12, 0.95f) : Col(22, 19, 17, 0.95f);
    dl->AddRectFilledMultiColor(a, b, top, top, bot, bot);
    dl->AddRect(a, b, on ? (danger ? Col(236, 120, 100) : Gold()) : Bronze(0.8f), 6 * u, 0,
                std::max(1.0f, 1.4f * u));
    const float px = 26 * u;
    const ImVec2 ts = TextSize(g_ft_title, px, label);
    Txt(dl, g_ft_title, px, {(a.x + b.x - ts.x) * 0.5f, (a.y + b.y - ts.y) * 0.5f},
        on ? Parch() : Col(220, 210, 194), label);
    return r;
}

class MenuLayer final : public ImGui::Layer {
public:
    Screen screen = Screen::Hub;
    bool was_open = false;
    float anim = 0.0f;   // open/close 0..1
    double scr_t = 0;    // screen switch time
    float scr_dir = 1.0f;
    double hub_t = 0;    // hub entrance (staggered cards)
    int hub_focus = 0;
    float card_hov[kCards] = {};
    double card_focus_t[kCards] = {};
    int pad_focus = 0;   // kPadRows index, 18 = reset
    int kbm_focus = 0;   // kKbmRows index, 24.. settings
    float kbm_scroll = -1.0f;
    bool kbm_scroll_to = false;
    int exit_focus = 0;
    float tog_direct = 0, tog_stick = 0, tog_look = 0, tog_inv = 0, tog_vib = 0;
    int opt_focus = 0; // (bb-input-135) 0 direct start, 1 vibration, 2 intensity
    double test_hold_since = -1; // (bb-mapwiz-136) OPTIONS held in the wizard
    // (bb-mapwiz-136) Mando > "Mapear controles": countdown -> one PS4 button at a time ->
    // summary. Nothing is written until GUARDAR, so leaving keeps the mapping as it was.
    enum class Wiz { Off, Count, Seq, Summary };
    Wiz wiz = Wiz::Off;
    double wiz_t = 0, wiz_step_t = 0, wiz_warn_t = -1;
    int wiz_step = 0, wiz_focus = 0;
    std::string wiz_warn;
    std::vector<std::string> wiz_new; // per step: the physical input pressed ("" = kept)
    float s_sp = 1.0f, s_dz = 0.5f;
    double dpad_since[4] = {-1, -1, -1, -1}, dpad_rep[4] = {};

    // 1.11 (bb-fgfix-111): the default (true) made ImGui::Core::MustKeepDrawing() always true,
    // so the video-out thread re-presented the last frame (DrawLastFrame) on every vblank tick
    // with no guest flip: ~170 extra presents a second that DLSS-G / FSR 3 FG took as real
    // frames with no inputs (real 225 fps -> dynamic x1, no guest-release samples, FSR 3 FG
    // stalls). Redraw on our own only while the menu or a trophy toast is on screen.
    bool ShouldKeepDrawing() override {
        if (g_open || anim > 0.0f) return true;
        std::scoped_lock lk{g_toast_mtx};
        return !g_toasts.empty();
    }

    void Go(Screen s) {
        scr_dir = s == Screen::Hub ? -1.0f : 1.0f;
        if (s == Screen::Hub) {
            hub_t = ImGui::GetTime() - 0.25; // a shorter stagger when coming back
        }
        if (s == Screen::Trophies) {
            g_trophy_page_t = ImGui::GetTime();
            g_trophy_bar = 0.0f;
            g_trophy_focus = 0;
            g_trophy_scroll = -1.0f;
        }
        if (s == Screen::Exit) {
            exit_focus = 0;
        }
        screen = s;
        scr_t = ImGui::GetTime();
        WizEnd();
        std::scoped_lock lk{g_cap_mtx};
        g_capture = Capture::None;
    }
    void Close() {
        g_want_close = true;
        Wake();
    }

    // d-pad auto-repeat (held > 0.4 s)
    void DpadRepeat(std::vector<Nav>& nav) {
        const u32 bits = g_pad_buttons.load();
        const int btn[4] = {SDL_GAMEPAD_BUTTON_DPAD_UP, SDL_GAMEPAD_BUTTON_DPAD_DOWN,
                           SDL_GAMEPAD_BUTTON_DPAD_LEFT, SDL_GAMEPAD_BUTTON_DPAD_RIGHT};
        const Nav dir[4] = {Nav::Up, Nav::Down, Nav::Left, Nav::Right};
        const double now = ImGui::GetTime();
        for (int i = 0; i < 4; i++) {
            if ((bits & (1u << btn[i])) == 0) {
                dpad_since[i] = -1;
                continue;
            }
            if (dpad_since[i] < 0) {
                dpad_since[i] = now;
                dpad_rep[i] = now;
            } else if (now - dpad_since[i] > 0.4 && now - dpad_rep[i] > 0.075) {
                dpad_rep[i] = now;
                nav.push_back(dir[i]);
            }
        }
    }

    void HandleNav(const std::vector<Nav>& nav) {
        for (const Nav n : nav) {
            if (n == Nav::Back) {
                if (screen == Screen::Pad && wiz != Wiz::Off) {
                    WizEnd(); // Esc / Círculo: leave the wizard without saving
                    continue;
                }
                if (screen == Screen::Hub) {
                    Close();
                } else {
                    Go(Screen::Hub);
                }
                continue;
            }
            switch (screen) {
            case Screen::Hub:
                if (n == Nav::Left || n == Nav::Up) hub_focus = (hub_focus + kCards - 1) % kCards;
                if (n == Nav::Right || n == Nav::Down) hub_focus = (hub_focus + 1) % kCards;
                if (n == Nav::Accept) Go(static_cast<Screen>(hub_focus));
                break;
            case Screen::Pad: {
                if (wiz == Wiz::Count || wiz == Wiz::Seq) break; // only Esc / OPTIONS hold
                if (wiz == Wiz::Summary) {
                    if (n == Nav::Left || n == Nav::Right || n == Nav::Up || n == Nav::Down)
                        wiz_focus ^= 1;
                    if (n == Nav::Accept) {
                        if (wiz_focus == 0) WizSave();
                        WizEnd();
                    }
                    break;
                }
                if (n == Nav::Left || n == Nav::Right || n == Nav::Up || n == Nav::Down)
                    pad_focus = pad_focus == 0 ? 1 : 0;
                if (n == Nav::Accept) {
                    if (pad_focus == 0) WizStart();
                    else ResetPad();
                }
                break;
            }
            case Screen::Kbm: {
                const int rows = static_cast<int>(std::size(kKbmRows));
                const int cnt = rows + 5;
                if (n == Nav::Up) kbm_focus = (kbm_focus + cnt - 1) % cnt, kbm_scroll_to = true;
                if (n == Nav::Down) kbm_focus = (kbm_focus + 1) % cnt, kbm_scroll_to = true;
                if (n == Nav::Left || n == Nav::Right) {
                    const float d = n == Nav::Left ? -1.0f : 1.0f;
                    float dz, sp, off;
                    g_cfg.MouseParams(dz, sp, off);
                    if (kbm_focus == rows) {
                        s_sp = std::clamp(s_sp + d * 0.1f, 0.1f, 10.0f);
                        g_cfg.SetMouseParams(s_dz, s_sp, off);
                        g_mouse_sens = s_sp, SaveMenuSettings();
                    } else if (kbm_focus == rows + 1) {
                        s_dz = std::clamp(s_dz + d * 0.02f, 0.0f, 1.0f);
                        g_cfg.SetMouseParams(s_dz, s_sp, off);
                    }
                }
                if (n == Nav::Accept) {
                    if (kbm_focus < rows) {
                        StartCaptureFor(kKbmRows[kbm_focus].output, Cat::Kbm);
                    } else if (kbm_focus == rows + 2) {
                        g_cfg.SetMouseStick(g_cfg.MouseStick() != "left" ? "left" : "right");
                    } else if (kbm_focus == rows + 3) {
                        g_kbm_play = !g_kbm_play;
                        g_restore_mouse_look = g_kbm_play;
                        SaveMenuSettings();
                    } else if (kbm_focus == rows + 4) {
                        g_invert_y = !g_invert_y;
                        SaveMenuSettings();
                    }
                }
                if (n == Nav::Clear && kbm_focus < rows) {
                    g_cfg.Set(kKbmRows[kbm_focus].output, Cat::Kbm, "", {});
                }
                break;
            }
            case Screen::Trophies: {
                const float u = std::max(ImGui::GetIO().DisplaySize.y / 1440.0f, 0.5f);
                const int cols = TrophyCols(ImGui::GetIO().DisplaySize.x - 2 * 110 * u, u);
                if (n == Nav::Left) g_trophy_focus--;
                if (n == Nav::Right) g_trophy_focus++;
                if (n == Nav::Up) g_trophy_focus -= cols;
                if (n == Nav::Down) g_trophy_focus += cols;
                g_trophy_focus = std::max(0, g_trophy_focus); // upper clamp when drawn
                if (n == Nav::Prev || n == Nav::Next) {
                    g_trophy_filter = (g_trophy_filter + (n == Nav::Next ? 1 : 2)) % 3;
                    g_trophy_focus = 0;
                    g_trophy_page_t = ImGui::GetTime();
                    g_trophy_scroll = 0.0f;
                }
                g_trophy_scroll_to = true;
                break;
            }
            case Screen::Options:
                if (n == Nav::Up) opt_focus = (opt_focus + 2) % 3;
                if (n == Nav::Down) opt_focus = (opt_focus + 1) % 3;
                if (n == Nav::Accept && opt_focus == 0 && g_direct.exists) {
                    if (WriteDirect(!g_direct.on)) g_direct.on = !g_direct.on;
                }
                if (n == Nav::Accept && opt_focus == 1) g_vib_on = !g_vib_on, SaveMenuSettings(), Input::TestRumble();
                if (opt_focus == 2 && (n == Nav::Left || n == Nav::Right)) {
                    g_vib_pct = std::clamp(g_vib_pct + (n == Nav::Left ? -10.0f : 10.0f), 0.0f, 200.0f);
                    SaveMenuSettings(), Input::TestRumble();
                }
                break;
            case Screen::Exit:
                if (n == Nav::Left || n == Nav::Right || n == Nav::Up || n == Nav::Down) {
                    exit_focus ^= 1;
                }
                if (n == Nav::Accept) {
                    if (exit_focus == 1) Quit();
                    else Go(Screen::Hub);
                }
                break;
            }
        }
    }

    static void Quit() {
        LOG_INFO(Frontend, "[bb-menu] Salir del juego: closing the emulator");
        SDL_Event e{};
        e.type = SDL_EVENT_QUIT;
        SDL_PushEvent(&e);
    }
    static void ResetPad() {
        for (const auto& r : kPadRows) {
            const std::string o = r.output;
            if (o.starts_with("stick_")) continue;
            g_cfg.Set(o, Cat::Pad, o, {});
        }
        for (const char* a : {"axis_left_x", "axis_left_y", "axis_right_x", "axis_right_y"}) {
            g_cfg.Set(a, Cat::Pad, a, {});
        }
    }

    // dimmed game, vignette, warm haze and slow embers (only while open)
    void Backdrop(ImVec2 disp, float e, float u) {
        auto* bg = ImGui::GetBackgroundDrawList();
        const float t = Now();
        const float sa = g_a;
        g_a = e;
        bg->AddRectFilled({0, 0}, disp, Col(5, 4, 6, 0.80f));
        const ImU32 v = Col(0, 0, 0, 0.75f), z = Col(0, 0, 0, 0.0f);
        bg->AddRectFilledMultiColor({0, 0}, {disp.x, disp.y * 0.30f}, v, v, z, z);
        bg->AddRectFilledMultiColor({0, disp.y * 0.70f}, disp, z, z, v, v);
        bg->AddRectFilledMultiColor({0, 0}, {disp.x * 0.22f, disp.y}, v, z, z, v);
        bg->AddRectFilledMultiColor({disp.x * 0.78f, 0}, disp, z, v, v, z);
        // warm haze breathing in the middle
        const float br = 0.75f + 0.25f * std::sin(t * 0.6f);
        for (int i = 6; i >= 1; i--) {
            bg->AddCircleFilled({disp.x * 0.5f, disp.y * 0.45f}, disp.x * 0.07f * i,
                                Col(120, 74, 36, 0.018f * br), 64);
        }
        // embers rising
        for (int i = 0; i < 46; i++) {
            const float h1 = std::fmod(std::sin(i * 12.9898f) * 43758.5453f, 1.0f);
            const float h2 = std::fmod(std::sin(i * 78.233f) * 12345.678f, 1.0f);
            const float rx = std::fabs(h1), rs = std::fabs(h2);
            const float speed = 0.025f + 0.04f * rs;
            const float life = std::fmod(t * speed + rx * 7.0f, 1.0f);
            const float y = disp.y * (1.05f - life * 1.1f);
            const float x = disp.x * rx + std::sin(t * (0.3f + rs) + i) * 30 * u;
            const float al = std::sin(life * 3.14159f) * (0.25f + 0.45f * rs);
            const float r = (1.2f + 2.2f * rs) * u;
            bg->AddCircleFilled({x, y}, r * 2.6f, Col(230, 140, 60, al * 0.12f));
            bg->AddCircleFilled({x, y}, r, Col(250, 190, 110, al));
        }
        g_a = sa;
    }

    void Header(ImVec2 disp, float u, float ha) {
        auto* dl = ImGui::GetWindowDrawList();
        const float sa = g_a;
        g_a = sa * ha;
        if (screen == Screen::Hub) {
            TxtSpaced(dl, g_ft_title, 70 * u, {disp.x * 0.5f, 92 * u}, Gold(), "BLOODBORNE", 14 * u,
                      true);
            TxtC(dl, g_ft_italic, 28 * u, disp.x * 0.5f, 180 * u, Muted(), "El Sueño del Cazador");
            OrnLine(dl, disp.x * 0.5f, 228 * u, 420 * u, 0.9f, u);
        } else {
            static const char* titles[] = {"MANDO", "TECLADO Y RATÓN", "TROFEOS", "OPCIONES",
                                           "SALIR DEL JUEGO"};
            TxtSpaced(dl, g_ft_title, 22 * u, {disp.x * 0.5f, 38 * u}, Bronze(), "BLOODBORNE", 6 * u,
                      true);
            TxtSpaced(dl, g_ft_title, 50 * u, {disp.x * 0.5f, 66 * u}, Gold(),
                      titles[static_cast<int>(screen)], 6 * u, true);
            OrnLine(dl, disp.x * 0.5f, 140 * u, 520 * u, 0.9f, u);
            // a back button top-left
            bool hov = false;
            const ImVec2 b0{60 * u, 56 * u}, b1{250 * u, 104 * u};
            if (Hit("##back", b0, b1, &hov)) {
                Go(Screen::Hub);
            }
            dl->AddRectFilled(b0, b1, hov ? Col(60, 48, 32, 0.9f) : Col(20, 18, 17, 0.8f), 24 * u);
            dl->AddRect(b0, b1, hov ? Gold() : Bronze(0.7f), 24 * u, 0, std::max(1.0f, 1.2f * u));
            Glyph(dl, "circle", {b0.x + 24 * u, (b0.y + b1.y) * 0.5f}, 14 * u);
            Txt(dl, g_ft_title, 22 * u, {b0.x + 48 * u, b0.y + 11 * u}, hov ? Parch() : Parch(0.85f),
                "VOLVER");
        }
        g_a = sa;
    }

    void Footer(ImVec2 disp, float u) {
        auto* dl = ImGui::GetWindowDrawList();
        std::vector<std::pair<const char*, const char*>> hints;
        switch (screen) {
        case Screen::Hub:
            hints = {{"cross", "Entrar"}, {"circle", "Volver al juego"}, {"F1", "Cerrar"}};
            break;
        case Screen::Pad:
            if (wiz == Wiz::Count || wiz == Wiz::Seq) {
                hints = {{"options", "Mantén OPTIONS 3 s: salir sin guardar"},
                         {"Esc", "Salir sin guardar"}};
                break;
            }
            if (wiz == Wiz::Summary) {
                hints = {{"pad", "Elegir"}, {"cross", "Aceptar"}, {"circle", "Descartar"}};
                break;
            }
            hints = {{"pad", "Elegir"}, {"cross", "Aceptar"}, {"circle", "Volver"}};
            break;
        case Screen::Kbm:
            hints = {{"pad", "Elegir"}, {"cross", "Asignar tecla"}, {"triangle", "Quitar"},
                     {"circle", "Volver"}};
            break;
        case Screen::Trophies:
            hints = {{"pad", "Moverse"}, {"l1", "Filtro"}, {"circle", "Volver"}};
            break;
        default:
            hints = {{"cross", "Aceptar"}, {"circle", "Volver"}};
            break;
        }
        const float px = 21 * u, r = 15 * u, gap = 36 * u;
        float total = 0;
        for (const auto& h : hints) {
            total += 2 * r + 10 * u + TextSize(g_ft_ui, px, h.second).x + gap;
        }
        float x = (disp.x - total + gap) * 0.5f;
        const float y = disp.y - 52 * u;
        OrnLine(dl, disp.x * 0.5f, y - 26 * u, std::min(disp.x * 0.35f, 700 * u), 0.45f, u);
        for (const auto& h : hints) {
            const std::string g = h.first;
            if (g == "F1" || g == "Esc") {
                KeyCap(dl, {x - 4 * u, y - 17 * u}, g, u, 1.0f, false);
                x += KeyCapW(g, u) - 2 * r - 4 * u;
            } else if (g == "pad") {
                Glyph(dl, "pad_up", {x + r, y}, r);
            } else {
                Glyph(dl, g, {x + r, y}, r);
            }
            Txt(dl, g_ft_ui, px, {x + 2 * r + 10 * u, y - px * 0.6f}, Parch(0.85f), h.second);
            x += 2 * r + 10 * u + TextSize(g_ft_ui, px, h.second).x + gap;
        }
    }

    void Draw() override {
        const ImVec2 disp = ImGui::GetIO().DisplaySize;
        const float u = std::max(disp.y / 1440.0f, 0.5f);
        g_a = 1.0f;
        DrawToast(u);
        const float dt = ImGui::GetIO().DeltaTime;
        anim = std::clamp(anim + (g_open ? dt : -dt) / 0.25f, 0.0f, 1.0f);
        if (!g_open && anim <= 0.0f) {
            was_open = false;
            TakeNav();
            return;
        }
        const bool closing = !g_open;
        if (g_open && !was_open) {
            was_open = true;
            g_cfg.Load();
            LoadTrophies();
            ReadDirect();
            float off;
            g_cfg.MouseParams(s_dz, s_sp, off);
            if (g_mouse_sens > 0) s_sp = g_mouse_sens;
            screen = Screen::Hub;
            WizEnd();
            hub_t = ImGui::GetTime();
            scr_t = hub_t;
            for (auto& h : card_hov) h = 0.0f;
            TakeNav();
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
        std::vector<Nav> nav = TakeNav();
        if (!capturing && !closing) {
            if (!g_test_mode) DpadRepeat(nav);
            HandleNav(nav);
        }
        Backdrop(disp, e, u);

        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize(disp, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, e);
        const ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNav |
            ImGuiWindowFlags_NoScrollWithMouse |
            (closing || capturing ? ImGuiWindowFlags_NoInputs : 0);
        if (ImGui::Begin("##bbmenu121", nullptr, flags)) {
            PushF(g_ft_ui, 20 * u);
            const float st = static_cast<float>(ImGui::GetTime() - scr_t);
            const float ca = EaseOut(st / 0.32f);
            g_a = e;
            Header(disp, u, std::clamp(st / 0.25f, 0.0f, 1.0f));
            g_a = e * ca;
            const float slide = (1.0f - ca) * 70 * u * scr_dir;
            const ImVec2 a{110 * u + slide, 180 * u}, b{disp.x - 110 * u + slide, disp.y - 100 * u};
            switch (screen) {
            case Screen::Hub: HubScreen(disp, u); break;
            case Screen::Pad: PadScreen(a, b, u); break;
            case Screen::Kbm: KbmScreen(a, b, u); break;
            case Screen::Trophies: TrophyScreen(a, b, u); break;
            case Screen::Options: OptionsScreen(a, b, u); break;
            case Screen::Exit: ExitScreen(disp, u); break;
            }
            g_a = e;
            Footer(disp, u);
            if (capturing) {
                CaptureOverlay(disp, u);
            }
            ImGui::PopFont();
        }
        ImGui::End();
        ImGui::PopStyleVar(3);
        g_a = 1.0f;
    }

    // ------------------------------------------------------------ hub
    void HubScreen(ImVec2 disp, float u) {
        static const char* titles[kCards] = {"MANDO", "TECLADO Y RATÓN", "TROFEOS", "OPCIONES",
                                             "SALIR"};
        int earned = 0;
        for (const auto& tr : g_trophies) earned += tr.earned;
        const std::string tsub = fmt::format("{} de {} conseguidos", earned, g_trophies.size());
        const char* subs[kCards] = {"Reasigna los botones de tu mando",
                                    "Teclas, ratón y cámara", tsub.c_str(),
                                    "Inicio directo y ajustes", "Cierra el juego con seguridad"};
        const float gap = 30 * u;
        const float cw = std::min(330 * u, (disp.x - 240 * u - gap * (kCards - 1)) / kCards);
        const float ch = std::min(cw * 1.6f, disp.y * 0.50f);
        const float x0 = (disp.x - (cw * kCards + gap * (kCards - 1))) * 0.5f;
        const float cy = disp.y * 0.56f;
        const float t = Now();
        const float dt = ImGui::GetIO().DeltaTime;
        auto* dl = ImGui::GetWindowDrawList();
        const float base = g_a;
        for (int i = 0; i < kCards; i++) {
            const float k = std::clamp(static_cast<float>(ImGui::GetTime() - hub_t - i * 0.07) / 0.5f,
                                       0.0f, 1.0f);
            const float ent = EaseOutBack(k);
            const ImVec2 ra{x0 + i * (cw + gap), cy - ch * 0.5f};
            const ImVec2 rb{ra.x + cw, ra.y + ch};
            bool hov = false;
            const std::string id = "##card" + std::to_string(i);
            if (Hit(id.c_str(), ra, rb, &hov)) {
                hub_focus = i;
                g_a = base;
                Go(static_cast<Screen>(i));
                return;
            }
            if (hov && MouseMoved()) hub_focus = i;
            const bool on = hub_focus == i;
            if (on && card_hov[i] < 0.05f) card_focus_t[i] = ImGui::GetTime();
            card_hov[i] = Approach(card_hov[i], on ? 1.0f : 0.0f, dt, 10.0f);
            const float h = card_hov[i];
            const float lift = -16 * u * h + (1.0f - ent) * 80 * u;
            const float sc = 1.0f + 0.04f * h;
            const ImVec2 c{(ra.x + rb.x) * 0.5f, (ra.y + rb.y) * 0.5f + lift};
            const ImVec2 a{c.x - cw * 0.5f * sc, c.y - ch * 0.5f * sc};
            const ImVec2 b{c.x + cw * 0.5f * sc, c.y + ch * 0.5f * sc};
            g_a = base * std::clamp(k * 1.6f, 0.0f, 1.0f);
            Panel(dl, a, b, u, h);
            // a warm light from above when focused
            if (h > 0.01f) {
                dl->AddRectFilledMultiColor({a.x + 8 * u, a.y + 8 * u}, {b.x - 8 * u, a.y + ch * 0.45f},
                                            Gold(0.10f * h), Gold(0.10f * h), Gold(0.0f), Gold(0.0f));
            }
            // medallion
            const ImVec2 m{c.x, a.y + (b.y - a.y) * 0.34f};
            const float mr = cw * 0.30f * sc;
            for (int g = 4; g >= 1; g--) {
                dl->AddCircleFilled(m, mr + g * 6 * u, Gold(0.025f * (0.3f + h)), 48);
            }
            dl->AddCircleFilled(m, mr, Col(14, 12, 12, 0.95f), 64);
            dl->AddCircle(m, mr, Bronze(0.9f), 64, std::max(1.0f, 1.6f * u));
            dl->AddCircle(m, mr - 7 * u, Gold(0.35f + 0.5f * h), 64, std::max(1.0f, u));
            for (int d = 0; d < 24; d++) {
                const float ang = d * 0.2618f + t * 0.25f * (0.3f + h);
                dl->AddCircleFilled({m.x + std::cos(ang) * (mr + 10 * u), m.y + std::sin(ang) * (mr + 10 * u)},
                                    1.4f * u, Gold(0.25f + 0.6f * h));
            }
            const ImU32 ic = Col(228, 192, 122, 0.75f + 0.25f * h);
            const float is = mr * 1.7f;
            switch (i) {
            case 0: IconPad(dl, m, is, ic); break;
            case 1: IconKbm(dl, m, is, ic); break;
            case 2: DrawCup(dl, m.x - is * 0.32f, m.y - is * 0.34f, is * 0.64f, ic, g_a); break;
            case 3: IconGear(dl, m, is, ic, t * 0.4f * h); break;
            default: IconDoor(dl, m, is, ic); break;
            }
            // title, rule, subtitle
            const float ty = a.y + (b.y - a.y) * 0.64f;
            const float tpx = std::min(30 * u, cw * 0.095f);
            TxtSpaced(dl, g_ft_title, tpx, {c.x, ty}, h > 0.5f ? Parch() : Gold(), titles[i], 2 * u,
                      true);
            OrnLine(dl, c.x, ty + tpx + 18 * u, cw * 0.32f, 0.5f + 0.5f * h, u);
            const auto lines = Wrap(g_ft_italic, 24 * u, subs[i], cw - 50 * u);
            for (size_t l = 0; l < lines.size() && l < 3; l++) {
                TxtC(dl, g_ft_italic, 24 * u, c.x, ty + tpx + 38 * u + l * 27 * u, Muted(0.9f + 0.1f * h),
                     lines[l].c_str());
            }
            // shine sweep once on focus
            const float fs = static_cast<float>(ImGui::GetTime() - card_focus_t[i]);
            if (on && fs < 0.9f) {
                const float sx = a.x - cw * 0.5f + (fs / 0.9f) * cw * 2.0f;
                dl->PushClipRect(a, b, true);
                dl->AddQuadFilled({sx, a.y}, {sx + cw * 0.18f, a.y}, {sx - cw * 0.22f, b.y},
                                  {sx - cw * 0.40f, b.y}, Col(255, 240, 200, 0.07f));
                dl->PopClipRect();
            }
        }
        g_a = base;
    }

    // ------------------------------------------------------------ Mando
    // (bb-mapwiz-136) the mapping wizard
    void WizStart() {
        wiz = Wiz::Count;
        wiz_t = ImGui::GetTime();
        wiz_step = 0;
        wiz_focus = 0;
        wiz_warn.clear();
        wiz_warn_t = -1;
        wiz_new.assign(kWizN, "");
        test_hold_since = -1;
        g_test_mode = true; // the pad never navigates while it runs
        g_wiz_listen = false;
        std::scoped_lock lk{g_cap_mtx};
        g_wiz_got.clear();
    }
    void WizEnd() {
        wiz = Wiz::Off;
        g_test_mode = false;
        g_wiz_listen = false;
        test_hold_since = -1;
    }
    // the final pad binding of every step: the button pressed, or the current one; a kept
    // binding that a pressed one took gets that step's old binding (a swap) or none
    std::vector<std::string> WizResolve() const {
        std::vector<std::string> cur(kWizN), fin(kWizN);
        for (int i = 0; i < kWizN; i++) cur[i] = g_cfg.GetRaw(kWizSteps[i].output, Cat::Pad);
        auto claimed = [&](const std::string& t) {
            for (int k = 0; k < kWizN; k++) {
                if (!wiz_new[k].empty() && wiz_new[k] == t) return k;
            }
            return -1;
        };
        for (int i = 0; i < kWizN; i++) {
            if (!wiz_new[i].empty()) {
                fin[i] = wiz_new[i];
                continue;
            }
            std::string out;
            for (const auto& t : Tokens(cur[i])) {
                std::string use = t;
                if (const int j = claimed(t); j >= 0 && j != i) {
                    use.clear();
                    for (const auto& c : Tokens(cur[j])) {
                        if (claimed(c) < 0) use = c;
                    }
                }
                if (!use.empty() && out.find(use) == std::string::npos) {
                    out += (out.empty() ? "" : ",") + use;
                }
            }
            fin[i] = out;
        }
        return fin;
    }
    void WizSave() {
        const auto fin = WizResolve();
        for (int i = 0; i < kWizN; i++) {
            const std::string o = kWizSteps[i].output;
            std::erase_if(g_cfg.lines,
                          [&](const Line& l) { return l.out == o && Classify(l.in) == Cat::Pad; });
            if (!fin[i].empty()) {
                g_cfg.lines.push_back({o + " = " + fin[i] + "  # F1 menu (mapear)", o, fin[i]});
            }
        }
        g_cfg.Save(); // one write, one reload
        LOG_INFO(Frontend, "[bb-menu] mapping wizard: saved");
    }
    void WizNext() {
        wiz_step++;
        wiz_step_t = ImGui::GetTime();
        wiz_warn_t = -1;
        g_wiz_listen = false; // wait for every button to be released first
        if (wiz_step >= kWizN) {
            wiz = Wiz::Summary;
            wiz_focus = 0;
            g_test_mode = false; // the summary is a normal menu
        }
    }
    // runs every frame while the wizard is on; true when it was left
    bool WizTick() {
        const double now = ImGui::GetTime();
        // OPTIONS held 3 s: leave without saving
        const bool hold = (g_pad_buttons.load() & (1u << SDL_GAMEPAD_BUTTON_START)) != 0;
        if (!hold) {
            test_hold_since = -1;
        } else if (test_hold_since < 0) {
            test_hold_since = now;
        } else if (now - test_hold_since >= 3.0) {
            LOG_INFO(Frontend, "[bb-menu] mapping wizard: left (OPTIONS held), nothing saved");
            WizEnd();
            return true;
        }
        if (wiz == Wiz::Count && now - wiz_t >= 3.0) {
            wiz = Wiz::Seq;
            wiz_step = -1;
            WizNext();
        }
        if (wiz != Wiz::Seq) return false;
        if (!g_wiz_listen) {
            const bool idle = g_pad_buttons.load() == 0 && g_trig_l < 8000 && g_trig_r < 8000;
            if (idle) {
                std::scoped_lock lk{g_cap_mtx};
                g_wiz_got.clear();
                g_wiz_start_down = 0;
                g_wiz_listen = true;
            }
            if (now - wiz_step_t > 5.0) WizNext(); // timer runs anyway
            return false;
        }
        std::string got;
        {
            std::scoped_lock lk{g_cap_mtx};
            got.swap(g_wiz_got);
        }
        if (!got.empty()) {
            for (int j = 0; j < wiz_step; j++) {
                if (wiz_new[j] == got) {
                    wiz_warn = PhysLabel(got) + " ya está asignado a " + kWizSteps[j].label +
                               ". Pulsa otro botón.";
                    wiz_warn_t = now;
                    wiz_step_t = now; // a fresh 5 s
                    g_wiz_listen = false;
                    return false;
                }
            }
            wiz_new[wiz_step] = got;
            WizNext();
        } else if (now - wiz_step_t > 5.0) {
            WizNext(); // nothing pressed: keeps its mapping
        }
        return false;
    }

    // ------------------------------------------------------------ Mando
    // (bb-mapwiz-136) only the DS4 photo and the mapping wizard
    void PadScreen(ImVec2 a, ImVec2 b, float u) {
        if (wiz != Wiz::Off && WizTick()) {
            // left: fall through to the idle view
        }
        pad_focus = std::clamp(pad_focus, 0, 1);
        const double now = ImGui::GetTime();
        const float pb = b.y - 210 * u;
        const std::string focus =
            wiz == Wiz::Seq && wiz_step < kWizN ? kWizSteps[wiz_step].output : "";
        std::string hover;
        std::map<std::string, ImVec2> anchors;
        const char* sub = wiz == Wiz::Off ? "los controles del juego"
                          : wiz == Wiz::Summary ? "resumen del mapeo"
                                                : "mapeando: pulsa en tu mando el botón indicado";
        Ps4Panel(a, {b.x, pb}, u, focus, hover, anchors, sub);
        auto* dl = ImGui::GetWindowDrawList();
        // the step's button: an extra glow on the photo
        if (!focus.empty() && anchors.contains(focus)) {
            const ImVec2 c = anchors[focus];
            const float ph = static_cast<float>(std::fmod(now, 1.0));
            for (int i = 0; i < 2; i++) {
                const float k = std::fmod(ph + i * 0.5f, 1.0f);
                dl->AddCircle(c, (30 + 70 * k) * u, Gold(0.8f * (1.0f - k)), 0, 3.0f * u);
            }
        }
        const ImVec2 d0{a.x, pb + 18 * u}, d1{b.x, b.y};
        const float cy = (d0.y + d1.y) * 0.5f;
        // OPTIONS hold ring (any wizard phase)
        auto hold_ring = [&](ImVec2 c) {
            const float k = test_hold_since < 0
                                ? 0.0f
                                : std::clamp(static_cast<float>(now - test_hold_since) / 3.0f,
                                             0.0f, 1.0f);
            const float r = 26 * u;
            dl->AddCircle(c, r, Bronze(0.5f), 48, 4 * u);
            if (k > 0.0f) {
                dl->PathArcTo(c, r, -IM_PI * 0.5f, -IM_PI * 0.5f + k * 2.0f * IM_PI, 48);
                dl->PathStroke(Gold(), 0, 5 * u);
            }
            Glyph(dl, "options", c, 14 * u, 1.0f, k > 0.0f);
            TxtR(dl, g_ft_ui, 17 * u, c.x - r - 12 * u, c.y - 10 * u, Muted(),
                 "Mantén OPTIONS 3 s o Esc: salir sin guardar");
        };
        if (wiz == Wiz::Off) {
            Panel(dl, d0, d1, u, 0.3f);
            Txt(dl, g_ft_title, 28 * u, {d0.x + 40 * u, cy - 40 * u}, Gold(), "MAPEAR TU MANDO");
            Txt(dl, g_ft_italic, 22 * u, {d0.x + 40 * u, cy + 2 * u}, Parch(0.9f),
                "Te iremos pidiendo cada botón de PS4: pulsa el de tu mando que quieras usar.");
            const float bw = 460 * u, bh = 88 * u, rw = 240 * u;
            float x = d1.x - 30 * u - rw;
            if (CardButton("##rst", {x, cy - 29 * u}, {x + rw, cy + 29 * u}, "RESTABLECER", u,
                           pad_focus == 1)) {
                ResetPad();
            }
            x -= bw + 20 * u;
            if (CardButton("##wiz", {x, cy - bh * 0.5f}, {x + bw, cy + bh * 0.5f},
                           "MAPEAR CONTROLES", u, pad_focus == 0)) {
                WizStart();
            }
            return;
        }
        if (wiz == Wiz::Count) {
            Panel(dl, d0, d1, u, 0.6f + 0.3f * std::sin(Now() * 3.0f));
            const int n = 3 - static_cast<int>(now - wiz_t);
            const std::string big = fmt::format("Preparado...  {}", std::clamp(n, 1, 3));
            Txt(dl, g_ft_title, 54 * u, {d0.x + 40 * u, cy - 48 * u}, Gold(), big.c_str());
            Txt(dl, g_ft_italic, 22 * u, {d0.x + 40 * u, cy + 20 * u}, Parch(0.9f),
                "Pulsa en tu mando el botón que se ilumine. Si no pulsas nada en 5 s, se queda "
                "como está.");
            const float bw = 240 * u;
            const ImVec2 q0{d1.x - 30 * u - bw, cy - 29 * u}, q1{d1.x - 30 * u, cy + 29 * u};
            if (CardButton("##wzc", q0, q1, "CANCELAR", u, false)) {
                WizEnd();
                return;
            }
            hold_ring({q0.x - 50 * u, cy + 56 * u});
            return;
        }
        if (wiz == Wiz::Seq && wiz_step < kWizN) {
            Panel(dl, d0, d1, u, 0.6f + 0.3f * std::sin(Now() * 3.0f));
            const std::string& o = kWizSteps[wiz_step].output;
            Glyph(dl, o, {d0.x + 70 * u, cy - 14 * u}, 38 * u, 1.0f, true);
            const std::string p = std::string("Pulsa el botón para: ") + kWizSteps[wiz_step].label;
            Txt(dl, g_ft_title, 40 * u, {d0.x + 130 * u, cy - 62 * u}, Gold(), p.c_str());
            const std::string fn = std::string("En Bloodborne: ") + DefaultFn(o) +
                                   "   ·   ahora: " + PhysLabel(g_cfg.GetRaw(o, Cat::Pad));
            Txt(dl, g_ft_italic, 22 * u, {d0.x + 130 * u, cy - 12 * u}, Muted(), fn.c_str());
            // timeout bar
            const float k = std::clamp(1.0f - static_cast<float>(now - wiz_step_t) / 5.0f, 0.0f, 1.0f);
            const float bx0 = d0.x + 130 * u, bx1 = d0.x + (d1.x - d0.x) * 0.55f, by = cy + 30 * u;
            dl->AddRectFilled({bx0, by}, {bx1, by + 8 * u}, Col(30, 26, 22), 4 * u);
            dl->AddRectFilled({bx0, by}, {bx0 + (bx1 - bx0) * k, by + 8 * u}, Gold(), 4 * u);
            const std::string prog = fmt::format("{} / {}", wiz_step + 1, kWizN);
            Txt(dl, g_ft_title, 30 * u, {bx1 + 20 * u, by - 14 * u}, Parch(), prog.c_str());
            if (wiz_warn_t >= 0 && now - wiz_warn_t < 3.0) {
                Txt(dl, g_ft_ui, 20 * u, {bx0, by + 20 * u}, Col(230, 120, 90), wiz_warn.c_str());
            }
            hold_ring({d1.x - 60 * u, cy});
            return;
        }
        if (wiz == Wiz::Summary) {
            // the list over the photo
            const auto fin = WizResolve();
            const ImVec2 s0{a.x + 60 * u, a.y + 96 * u}, s1{b.x - 60 * u, pb - 10 * u};
            dl->AddRectFilled(s0, s1, Col(14, 12, 11, 0.94f), 10 * u);
            Panel(dl, s0, s1, u, 0.4f);
            const int per = (kWizN + 1) / 2;
            const float colw = (s1.x - s0.x - 80 * u) * 0.5f;
            const float rh = std::min(46 * u, (s1.y - s0.y - 40 * u) / per);
            for (int i = 0; i < kWizN; i++) {
                const float x = s0.x + 40 * u + (i / per) * (colw + 20 * u);
                const float y = s0.y + 20 * u + (i % per) * rh;
                const std::string o = kWizSteps[i].output;
                Glyph(dl, o, {x + 18 * u, y + rh * 0.5f}, 15 * u, 1.0f, !wiz_new[i].empty());
                Txt(dl, g_ft_text, 24 * u, {x + 46 * u, y + rh * 0.5f - 14 * u}, Parch(),
                    kWizSteps[i].label);
                const std::string v =
                    "->  " + PhysLabel(fin[i]) + (wiz_new[i].empty() ? "  (sin cambios)" : "");
                Txt(dl, g_ft_ui, 20 * u, {x + colw * 0.48f, y + rh * 0.5f - 11 * u},
                    wiz_new[i].empty() ? Muted() : Gold(), v.c_str());
            }
            Panel(dl, d0, d1, u, 0.3f);
            Txt(dl, g_ft_title, 30 * u, {d0.x + 40 * u, cy - 36 * u}, Gold(), "¿GUARDAR EL MAPEO?");
            Txt(dl, g_ft_italic, 22 * u, {d0.x + 40 * u, cy + 4 * u}, Parch(0.9f),
                "Descartar deja el mando exactamente como estaba.");
            const float bw = 260 * u, bh = 66 * u;
            float x = d1.x - 30 * u - bw;
            if (CardButton("##wzd", {x, cy - bh * 0.5f}, {x + bw, cy + bh * 0.5f}, "DESCARTAR", u,
                           wiz_focus == 1)) {
                WizEnd();
                return;
            }
            x -= bw + 20 * u;
            if (CardButton("##wzs", {x, cy - bh * 0.5f}, {x + bw, cy + bh * 0.5f}, "GUARDAR", u,
                           wiz_focus == 0)) {
                WizSave();
                WizEnd();
            }
        }
    }

    // ------------------------------------------------------------ Teclado y ratón
    void KbmScreen(ImVec2 a, ImVec2 b, float u) {
        const int rows = static_cast<int>(std::size(kKbmRows));
        kbm_focus = std::clamp(kbm_focus, 0, rows + 4);
        const std::string focus = kbm_focus < rows ? kKbmRows[kbm_focus].output : "";
        const float mid = a.x + (b.x - a.x) * 0.48f;
        std::string hover;
        std::map<std::string, ImVec2> anchors;
        const std::string clicked = Ps4Panel(a, {mid - 14 * u, b.y}, u, FocusKey(focus), hover,
                                             anchors, "elige un botón y pulsa una tecla");
        if (!clicked.empty()) {
            const std::string want = clicked == "stick_left"    ? "axis_left_y_minus"
                                     : clicked == "stick_right" ? "axis_right_y_minus"
                                                                : clicked;
            for (int i = 0; i < rows; i++) {
                if (want == kKbmRows[i].output) kbm_focus = i, kbm_scroll_to = true;
            }
            if (!clicked.starts_with("stick_")) StartCaptureFor(want, Cat::Kbm);
        }
        auto* dl = ImGui::GetWindowDrawList();
        const ImVec2 pa{mid + 14 * u, a.y}, pbr{b.x, b.y};
        Panel(dl, pa, pbr, u);
        Txt(dl, g_ft_title, 30 * u, {pa.x + 30 * u, pa.y + 22 * u}, Gold(), "TECLADO Y RATÓN");
        Txt(dl, g_ft_italic, 24 * u, {pa.x + 30 * u, pa.y + 58 * u}, Muted(),
            "la tecla o el botón que da cada botón de PS4");
        const float set_h = 300 * u;
        const ImVec2 l0{pa.x + 20 * u, pa.y + 104 * u};
        const float lw = pbr.x - pa.x - 40 * u, lh = pbr.y - l0.y - set_h - 20 * u;
        // the list
        ImGui::SetCursorScreenPos(l0);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, ImVec4(0.55f, 0.42f, 0.24f, 0.6f * g_a));
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8 * u);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
        ImGui::BeginChild("##kbml", {lw, lh}, 0,
                          ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollWithMouse);
        SmoothScroll(kbm_scroll, u);
        auto* cd = ImGui::GetWindowDrawList();
        const float rh = 54 * u;
        const float rw = lw - 14 * u;
        for (int i = 0; i < rows; i++) {
            const std::string o = kKbmRows[i].output;
            const ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::PushID(i);
            const bool click = ImGui::InvisibleButton("##r", {rw, rh});
            const bool hov = ImGui::IsItemHovered();
            ImGui::PopID();
            if (hov && MouseMoved()) kbm_focus = i;
            if (click) {
                kbm_focus = i;
                StartCaptureFor(o, Cat::Kbm);
            }
            const bool on = kbm_focus == i;
            if (on || hov) {
                cd->AddRectFilledMultiColor(p, {p.x + rw, p.y + rh}, Gold(on ? 0.16f : 0.08f),
                                            Gold(0.0f), Gold(0.0f), Gold(on ? 0.16f : 0.08f));
                if (on) cd->AddRectFilled(p, {p.x + 3 * u, p.y + rh}, Gold());
            }
            cd->AddLine({p.x, p.y + rh - 0.5f}, {p.x + rw, p.y + rh - 0.5f}, Bronze(0.18f));
            Glyph(cd, o, {p.x + 30 * u, p.y + rh * 0.5f}, 17 * u, 1.0f, on);
            Txt(cd, g_ft_text, 25 * u, {p.x + 60 * u, p.y + rh * 0.5f - 15 * u},
                on ? Parch() : Col(214, 204, 188), RowLabel(o));
            const auto toks = Tokens(g_cfg.GetRaw(o, Cat::Kbm));
            float kx = p.x + rw - 10 * u;
            if (toks.empty()) {
                TxtR(cd, g_ft_italic, 22 * u, kx, p.y + rh * 0.5f - 13 * u, Muted(0.7f), "sin asignar");
            }
            for (auto it = toks.rbegin(); it != toks.rend(); ++it) {
                const std::string l = Pretty(*it);
                kx -= KeyCapW(l, u);
                KeyCap(cd, {kx, p.y + (rh - 34 * u) * 0.5f}, l, u, 1.0f, on);
                kx -= 8 * u;
            }
        }
        if (kbm_scroll_to && kbm_focus < rows) {
            kbm_scroll_to = false;
            const float y0 = kbm_focus * rh, cur = ImGui::GetScrollY();
            if (y0 < cur) kbm_scroll = y0;
            else if (y0 + rh > cur + lh) kbm_scroll = std::min(ImGui::GetScrollMaxY(), y0 + rh - lh);
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(3);
        // mouse settings
        const ImVec2 s0{pa.x + 30 * u, pbr.y - set_h};
        OrnLine(dl, (pa.x + pbr.x) * 0.5f, s0.y - 6 * u, (pbr.x - pa.x) * 0.4f, 0.5f, u);
        const ImTextureID mtex = Get(g_mouse_tex, AssetDir() / "raton.png", 212.0f, 220.0f);
        const float mh = set_h - 40 * u, ms = mh / g_mouse_tex.h;
        const ImVec2 m0{s0.x, s0.y + 20 * u};
        if (mtex) {
            dl->AddImage(mtex, m0, {m0.x + g_mouse_tex.w * ms, m0.y + mh}, {0, 0}, {1, 1},
                         Col(255, 255, 255));
        }
        static const struct {
            const char* in;
            float x, y, d;
        } kMouse[] = {{"leftbutton", 72, 72, 40},     {"rightbutton", 140, 52, 40},
                      {"middlebutton", 103, 60, 18},  {"sidebuttonforward", 40, 103, 16},
                      {"sidebuttonback", 48, 128, 16}};
        static std::string s_mouse;
        for (const auto& mb : kMouse) {
            const ImVec2 c{m0.x + mb.x * ms, m0.y + mb.y * ms};
            const float r = std::max(8 * u, mb.d * ms * 0.5f);
            bool hov = false;
            const std::string id = std::string("##m") + mb.in;
            if (Hit(id.c_str(), {c.x - r, c.y - r}, {c.x + r, c.y + r}, &hov)) {
                s_mouse = mb.in;
                ImGui::OpenPopup("##mousebind");
            }
            if (hov) {
                dl->AddCircleFilled(c, r, Gold(0.25f));
                dl->AddCircle(c, r, Gold(), 0, 2 * u);
            }
        }
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.07f, 0.06f, 0.06f, 0.97f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.62f, 0.44f, 0.24f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.45f, 0.34f, 0.18f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.35f, 0.27f, 0.15f, 0.8f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16 * u, 12 * u});
        if (ImGui::BeginPopup("##mousebind", ImGuiWindowFlags_NoNav)) {
            ImGui::TextColored(ImVec4(0.89f, 0.75f, 0.48f, 1.0f), "%s hace:", Pretty(s_mouse).c_str());
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
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(4);
        // sliders and toggles
        const float sx = m0.x + g_mouse_tex.w * ms + 40 * u, sw = pbr.x - sx - 60 * u;
        float sy = s0.y + 14 * u;
        float dz, sp, off;
        g_cfg.MouseParams(dz, sp, off);
        const auto label = [&](int idx, const std::string& l, const std::string& val) {
            const bool on = kbm_focus == idx;
            if (on) dl->AddRectFilled({sx - 14 * u, sy + 2 * u}, {sx - 10 * u, sy + 24 * u}, Gold());
            Txt(dl, g_ft_text, 23 * u, {sx, sy}, on ? Parch() : Col(214, 204, 188), l.c_str());
            if (!val.empty()) TxtR(dl, g_ft_ui, 19 * u, sx + sw, sy + 3 * u, Gold(), val.c_str());
        };
        label(rows, "Sensibilidad de la cámara", fmt::format("{:.2f}", s_sp));
        if (Slider("##sens", {sx, sy + 26 * u}, sw, s_sp, 0.1f, 10.0f, u, kbm_focus == rows)) {
            g_cfg.SetMouseParams(s_dz, s_sp, off);
            g_mouse_sens = s_sp, SaveMenuSettings();
        }
        sy += 72 * u;
        label(rows + 1, "Zona muerta", fmt::format("{:.2f}", s_dz));
        if (Slider("##dz", {sx, sy + 26 * u}, sw, s_dz, 0.0f, 1.0f, u, kbm_focus == rows + 1)) {
            g_cfg.SetMouseParams(s_dz, s_sp, off);
        }
        sy += 76 * u;
        const float half = sw * 0.5f;
        {
            const bool right = g_cfg.MouseStick() != "left";
            if (Toggle("##mstick", {sx, sy}, right, tog_stick, u, kbm_focus == rows + 2)) {
                g_cfg.SetMouseStick(right ? "left" : "right");
            }
            Txt(dl, g_ft_text, 22 * u, {sx + 80 * u, sy + 4 * u}, Parch(0.9f), "Ratón = cámara");
            // (bb-menu-133) persisted: mouse-look + hidden, captured cursor while playing
            if (Toggle("##mlook", {sx + half, sy}, g_kbm_play, tog_look, u,
                       kbm_focus == rows + 3)) {
                g_kbm_play = !g_kbm_play;
                g_restore_mouse_look = g_kbm_play;
                SaveMenuSettings();
            }
            Txt(dl, g_ft_text, 22 * u, {sx + half + 80 * u, sy + 4 * u}, Parch(0.9f),
                "Jugar con teclado y ratón");
            sy += 50 * u;
            if (Toggle("##minv", {sx, sy}, g_invert_y, tog_inv, u, kbm_focus == rows + 4)) {
                g_invert_y = !g_invert_y;
                SaveMenuSettings();
            }
            Txt(dl, g_ft_text, 22 * u, {sx + 80 * u, sy + 4 * u}, Parch(0.9f), "Invertir eje Y");
            Txt(dl, g_ft_italic, 20 * u, {sx + half, sy + 6 * u}, Muted(),
                "Cursor oculto al jugar; F1 lo libera (F7 alterna)");
        }
    }

    // ------------------------------------------------------------ Opciones
    void OptionsScreen(ImVec2 a, ImVec2 b, float u) {
        auto* dl = ImGui::GetWindowDrawList();
        const float w = std::min(1300 * u, b.x - a.x);
        const ImVec2 p0{(a.x + b.x - w) * 0.5f, a.y + 30 * u}, p1{p0.x + w, p0.y + 560 * u};
        Panel(dl, p0, p1, u, 0.2f);
        Txt(dl, g_ft_title, 30 * u, {p0.x + 40 * u, p0.y + 30 * u}, Gold(), "INICIO");
        OrnLine(dl, p0.x + 40 * u + 160 * u, p0.y + 88 * u, 160 * u, 0.5f, u);
        const ImVec2 tg{p0.x + 40 * u, p0.y + 116 * u};
        ImGui::BeginDisabled(!g_direct.exists);
        if (Toggle("##direct", tg, g_direct.on, tog_direct, u, opt_focus == 0) && g_direct.exists) {
            if (WriteDirect(!g_direct.on)) g_direct.on = !g_direct.on;
        }
        ImGui::EndDisabled();
        Txt(dl, g_ft_text, 28 * u, {tg.x + 90 * u, tg.y}, Parch(),
            "Iniciar directamente (sin pasar por el launcher)");
        const auto lines = Wrap(g_ft_italic, 24 * u,
                                "Si está activado, la próxima vez que abras el launcher el juego se "
                                "inicia directamente con los ajustes guardados. Mantén pulsada la "
                                "tecla Mayús (Shift) al abrir el launcher para volver a verlo.",
                                w - 170 * u);
        float y = tg.y + 50 * u;
        for (const auto& l : lines) {
            Txt(dl, g_ft_italic, 24 * u, {tg.x + 90 * u, y}, Muted(), l.c_str());
            y += 28 * u;
        }
        if (!g_direct.exists) {
            const std::string m = "No se encontró launcher.ini junto a shadPS4.exe. Abre el juego "
                                  "una vez desde el launcher para crearlo.";
            Txt(dl, g_ft_ui, 19 * u, {tg.x + 90 * u, y + 8 * u}, Col(232, 170, 90), m.c_str());
        }
        if (!g_direct.error.empty()) {
            Txt(dl, g_ft_ui, 19 * u, {tg.x + 90 * u, y + 34 * u}, Col(232, 120, 100),
                g_direct.error.c_str());
        }
        // (bb-input-135) rumble (scePadSetVibration -> SDL), saved in bb_menu.ini
        const ImVec2 vt{tg.x, p0.y + 370 * u};
        if (Toggle("##vib", vt, g_vib_on, tog_vib, u, opt_focus == 1)) {
            g_vib_on = !g_vib_on;
            SaveMenuSettings(), Input::TestRumble();
        }
        Txt(dl, g_ft_text, 28 * u, {vt.x + 90 * u, vt.y}, Parch(), "Vibración");
        Txt(dl, g_ft_text, 24 * u, {vt.x + 90 * u, vt.y + 56 * u}, Parch(0.9f),
            fmt::format("Intensidad {:.0f} %", g_vib_pct).c_str());
        if (Slider("##vibpct", {vt.x + 90 * u, vt.y + 92 * u}, 500 * u, g_vib_pct, 0.0f, 200.0f, u,
                   opt_focus == 2)) {
            SaveMenuSettings(), Input::TestRumble();
        }
    }

    // ------------------------------------------------------------ Salir
    void ExitScreen(ImVec2 disp, float u) {
        auto* dl = ImGui::GetWindowDrawList();
        const float w = 1000 * u, h = 420 * u;
        const float slide = (1.0f - std::clamp(g_a, 0.0f, 1.0f)) * 30 * u;
        const ImVec2 p0{(disp.x - w) * 0.5f, disp.y * 0.5f - h * 0.5f + slide}, p1{p0.x + w, p0.y + h};
        Panel(dl, p0, p1, u, 0.35f);
        const float cx = disp.x * 0.5f;
        TxtSpaced(dl, g_ft_title, 42 * u, {cx, p0.y + 50 * u}, Gold(), "¿SALIR DEL JUEGO?", 4 * u, true);
        OrnLine(dl, cx, p0.y + 120 * u, 300 * u, 0.7f, u);
        const auto lines = Wrap(g_ft_text, 27 * u,
                                "Bloodborne guarda solo: espera a que no se vea el icono de "
                                "guardado antes de salir.",
                                w - 160 * u);
        float y = p0.y + 150 * u;
        for (const auto& l : lines) {
            TxtC(dl, g_ft_text, 27 * u, cx, y, Parch(0.9f), l.c_str());
            y += 33 * u;
        }
        const float bw = 300 * u, bh = 70 * u, by = p1.y - 50 * u - bh;
        if (CardButton("##no", {cx - bw - 20 * u, by}, {cx - 20 * u, by + bh}, "VOLVER", u,
                       exit_focus == 0)) {
            Go(Screen::Hub);
        }
        if (CardButton("##yes", {cx + 20 * u, by}, {cx + 20 * u + bw, by + bh}, "SALIR", u,
                       exit_focus == 1, true)) {
            Quit();
        }
    }

    // ------------------------------------------------------------ capture
    void CaptureOverlay(ImVec2 disp, float u) {
        Capture kind;
        std::string target;
        Uint64 since;
        {
            std::scoped_lock lk{g_cap_mtx};
            kind = g_capture;
            target = g_capture_target;
            since = g_capture_since;
        }
        if (kind == Capture::None) return;
        auto* dl = ImGui::GetForegroundDrawList();
        const float el = static_cast<float>(SDL_GetTicks() - since) / 1000.0f;
        const float fa = EaseOut(el / 0.2f);
        const float sa = g_a;
        g_a = sa * fa;
        dl->AddRectFilled({0, 0}, disp, Col(0, 0, 0, 0.62f));
        const float w = 860 * u, h = 360 * u;
        const float sc = 0.94f + 0.06f * EaseOutBack(el / 0.3f);
        const ImVec2 c{disp.x * 0.5f, disp.y * 0.5f};
        const ImVec2 p0{c.x - w * 0.5f * sc, c.y - h * 0.5f * sc}, p1{c.x + w * 0.5f * sc, c.y + h * 0.5f * sc};
        Panel(dl, p0, p1, u, 0.6f + 0.4f * std::sin(el * 4.0f));
        const char* head = kind == Capture::Kbm        ? "PULSA UNA TECLA"
                           : kind == Capture::PadStick ? "MUEVE UN STICK"
                                                       : "PULSA UN BOTÓN";
        const char* sub = kind == Capture::Kbm ? "o un botón del ratón para"
                          : kind == Capture::PadStick ? "del mando (el que quieras usar) para"
                                                      : "de tu mando para";
        TxtSpaced(dl, g_ft_title, 38 * u, {c.x, p0.y + 46 * u}, Gold(), head, 4 * u, true);
        OrnLine(dl, c.x, p0.y + 108 * u, 240 * u, 0.7f, u);
        TxtC(dl, g_ft_italic, 25 * u, c.x, p0.y + 128 * u, Muted(), sub);
        // the target: glyph + name + default function
        const std::string nm = std::string(RowLabel(target));
        const float gw = 34 * u * 2 + 16 * u + TextSize(g_ft_title, 30 * u, nm.c_str()).x;
        Glyph(dl, target, {c.x - gw * 0.5f + 34 * u, p0.y + 200 * u}, 30 * u, 1.0f, true);
        Txt(dl, g_ft_title, 30 * u, {c.x - gw * 0.5f + 84 * u, p0.y + 182 * u}, Parch(), nm.c_str());
        TxtC(dl, g_ft_text, 24 * u, c.x, p0.y + 240 * u, Muted(0.9f), DefaultFn(target));
        // countdown ring (5 s)
        const float left = std::clamp(1.0f - (el - 0.25f) / 5.0f, 0.0f, 1.0f);
        const ImVec2 rc{p1.x - 60 * u, p1.y - 56 * u};
        dl->AddCircle(rc, 22 * u, Bronze(0.4f), 48, 3 * u);
        if (left > 0.0f) {
            dl->PathArcTo(rc, 22 * u, -1.5708f, -1.5708f + left * 6.2832f, 48);
            dl->PathStroke(Gold(), 0, 3 * u);
        }
        const std::string secs = fmt::format("{}", static_cast<int>(std::ceil(left * 5.0f)));
        const ImVec2 ts = TextSize(g_ft_ui, 18 * u, secs.c_str());
        Txt(dl, g_ft_ui, 18 * u, {rc.x - ts.x * 0.5f, rc.y - ts.y * 0.5f}, Parch(), secs.c_str());
        Txt(dl, g_ft_ui, 19 * u, {p0.x + 40 * u, p1.y - 68 * u}, Muted(0.85f),
            "Esc: cancelar  ·  se cancela sola en 5 s");
        g_a = sa;
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
    LoadMenuSettings();
    if (g_kbm_play) {
        // (bb-menu-133) "Jugar con teclado y ratón": mouse-look and a captured cursor from start
        if (Input::GetMouseMode() != Input::MouseMode::Joystick) {
            Input::ToggleMouseModeTo(Input::MouseMode::Joystick);
        }
        SDL_SetWindowRelativeMouseMode(window, true);
        LOG_INFO(Frontend, "[bb-menu] keyboard + mouse play on: mouse-look, cursor captured");
    }
    ImGui::Layer::AddLayer(&g_layer);
    std::thread(CustomWatcher).detach();
}

void LoadFonts() {
    // (bb-menu-121) the menu's fonts (SIL OFL 1.1), from menu_assets\fonts next to the exe
    ImGuiIO& io = ImGui::GetIO();
    const auto dir = AssetDir() / "fonts";
    const auto load = [&](const char* file) -> ImFont* {
        const auto p = dir / file;
        std::error_code ec;
        if (!std::filesystem::is_regular_file(p, ec)) {
            LOG_WARNING(Frontend, "[bb-menu] missing font {}", p.string());
            return nullptr;
        }
        ImFontConfig cfg{};
        cfg.OversampleH = 2;
        cfg.OversampleV = 2;
        cfg.PixelSnapH = false;
        const auto u8 = p.u8string();
        return io.Fonts->AddFontFromFileTTF(reinterpret_cast<const char*>(u8.c_str()), 32.0f, &cfg);
    };
    g_ft_title = load("Cinzel-SemiBold.ttf");
    g_ft_text = load("CormorantGaramond-Medium.ttf");
    g_ft_italic = load("CormorantGaramond-MediumItalic.ttf");
    g_ft_ui = load("Inter-Medium.ttf");
    if (g_ft_italic == nullptr) g_ft_italic = g_ft_text;
    LOG_INFO(Frontend, "[bb-menu] fonts: title {} text {} ui {}", g_ft_title != nullptr,
             g_ft_text != nullptr, g_ft_ui != nullptr);
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

bool IsPadInput(Uint32 t) {
    return t == SDL_EVENT_GAMEPAD_BUTTON_DOWN || t == SDL_EVENT_GAMEPAD_BUTTON_UP ||
           t == SDL_EVENT_GAMEPAD_AXIS_MOTION || t == SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN ||
           t == SDL_EVENT_GAMEPAD_TOUCHPAD_UP || t == SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION ||
           t == SDL_EVENT_GAMEPAD_SENSOR_UPDATE;
}
bool IsKbmInput(Uint32 t) {
    return t == SDL_EVENT_KEY_DOWN || t == SDL_EVENT_KEY_UP || t == SDL_EVENT_TEXT_INPUT ||
           t == SDL_EVENT_MOUSE_BUTTON_DOWN || t == SDL_EVENT_MOUSE_BUTTON_UP ||
           t == SDL_EVENT_MOUSE_WHEEL || t == SDL_EVENT_MOUSE_MOTION;
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
        } else if (static_cast<int>(event.gaxis.axis) <= static_cast<int>(SDL_GAMEPAD_AXIS_RIGHTY)) {
            g_axis[event.gaxis.axis] = event.gaxis.value;
        }
    } else if (event.type == SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN) {
        g_touch = true;
    } else if (event.type == SDL_EVENT_GAMEPAD_TOUCHPAD_UP) {
        g_touch = false;
    } else if (event.type == SDL_EVENT_WINDOW_FOCUS_GAINED && !g_open && g_kbm_play &&
               g_window != nullptr && Input::GetMouseMode() == Input::MouseMode::Joystick) {
        SDL_SetWindowRelativeMouseMode(g_window, true); // re-capture after Alt+Tab
    }
    if (!g_open) {
        return false;
    }
    const Uint32 type = event.type;
    // (bb-menu-121) a rebind owns every input until it gets one, is cancelled (Esc) or times
    // out (5 s): nothing reaches the menu (no navigation, no clicks) and stick noise is ignored.
    {
        std::scoped_lock lk{g_cap_mtx};
        if (g_capture != Capture::None) {
            if (g_captured.empty() && !g_cap_cancel && SDL_GetTicks() - g_capture_since > 250) {
                std::string got;
                if (type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) {
                    g_cap_cancel = true;
                } else if (g_capture == Capture::Kbm) {
                    if (type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
                        got = KeyName(event.key.key, false);
                        if (got.empty()) g_cap_cancel = true; // a key that cannot be bound
                    } else if (type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                        got = KeyName(event.button.button, true);
                        if (got.empty()) g_cap_cancel = true;
                    } else if (type == SDL_EVENT_MOUSE_WHEEL &&
                               (std::fabs(event.wheel.y) >= 0.5f || std::fabs(event.wheel.x) >= 0.5f)) {
                        got = event.wheel.y > 0   ? "mousewheelup"
                              : event.wheel.y < 0 ? "mousewheeldown"
                              : event.wheel.x > 0 ? "mousewheelright"
                                                  : "mousewheelleft";
                    }
                } else if (g_capture == Capture::Pad) {
                    if (type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
                        if (const char* n = PadName(event.gbutton.button)) {
                            got = n;
                        }
                    } else if (type == SDL_EVENT_GAMEPAD_AXIS_MOTION && event.gaxis.value > 16000) {
                        if (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) {
                            got = "l2";
                        } else if (event.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
                            got = "r2";
                        }
                    }
                } else if (g_capture == Capture::PadStick) {
                    if (type == SDL_EVENT_GAMEPAD_AXIS_MOTION &&
                        std::abs(int(event.gaxis.value)) > 22000) {
                        const auto ax = event.gaxis.axis;
                        if (ax == SDL_GAMEPAD_AXIS_LEFTX || ax == SDL_GAMEPAD_AXIS_LEFTY) {
                            got = "left";
                        } else if (ax == SDL_GAMEPAD_AXIS_RIGHTX || ax == SDL_GAMEPAD_AXIS_RIGHTY) {
                            got = "right";
                        }
                    }
                }
                if (!got.empty()) {
                    g_captured = got;
                }
            }
            if (IsPadInput(type)) {
                // the pad state still updates (the game reads it neutral: input captured)
                return false;
            }
            if (IsKbmInput(type)) {
                // releases go on so nothing stays held in the game; nothing reaches ImGui
                return !(type == SDL_EVENT_KEY_UP || type == SDL_EVENT_MOUSE_BUTTON_UP);
            }
        }
    }
    // the menu's own navigation: d-pad, face buttons, L1 / R1, arrows, Enter, Esc. The sticks
    // never move it; nothing used here reaches the game.
    // (bb-mapwiz-136) the wizard: a pad press is an answer, never navigation
    if (g_wiz_listen && IsPadInput(type)) {
        std::string got;
        if (type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
            if (event.gbutton.button == SDL_GAMEPAD_BUTTON_START) {
                g_wiz_start_down = SDL_GetTicks(); // answered on release (a hold leaves)
            } else if (const char* n = PadName(event.gbutton.button)) {
                got = n;
            }
        } else if (type == SDL_EVENT_GAMEPAD_BUTTON_UP &&
                   event.gbutton.button == SDL_GAMEPAD_BUTTON_START && g_wiz_start_down != 0) {
            if (SDL_GetTicks() - g_wiz_start_down < 700) got = "options";
            g_wiz_start_down = 0;
        } else if (type == SDL_EVENT_GAMEPAD_AXIS_MOTION && event.gaxis.value > 16000) {
            if (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) got = "l2";
            if (event.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) got = "r2";
        }
        if (!got.empty()) {
            std::scoped_lock lk{g_cap_mtx};
            if (g_wiz_got.empty()) g_wiz_got = got;
        }
        return false;
    }
    if (type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && g_test_mode) {
        return false; // (bb-mapwiz-136) wizard on: the pad never navigates
    }
    if (type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
        switch (event.gbutton.button) {
        case SDL_GAMEPAD_BUTTON_DPAD_UP: PushNav(Nav::Up); break;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN: PushNav(Nav::Down); break;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT: PushNav(Nav::Left); break;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: PushNav(Nav::Right); break;
        case SDL_GAMEPAD_BUTTON_SOUTH: PushNav(Nav::Accept); break;
        case SDL_GAMEPAD_BUTTON_EAST: PushNav(Nav::Back); break;
        case SDL_GAMEPAD_BUTTON_NORTH: PushNav(Nav::Clear); break;
        case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: PushNav(Nav::Prev); break;
        case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: PushNav(Nav::Next); break;
        default: break;
        }
        return false; // pad state only (game input captured while open)
    }
    if (IsPadInput(type)) {
        return false;
    }
    if (type == SDL_EVENT_KEY_DOWN) {
        const bool rep = event.key.repeat;
        switch (event.key.key) {
        case SDLK_UP: PushNav(Nav::Up); break;
        case SDLK_DOWN: PushNav(Nav::Down); break;
        case SDLK_LEFT: PushNav(Nav::Left); break;
        case SDLK_RIGHT: PushNav(Nav::Right); break;
        case SDLK_RETURN:
        case SDLK_KP_ENTER:
        case SDLK_SPACE:
            if (!rep) PushNav(Nav::Accept);
            break;
        case SDLK_ESCAPE:
        case SDLK_BACKSPACE:
            if (!rep) PushNav(Nav::Back);
            break;
        case SDLK_DELETE:
            if (!rep) PushNav(Nav::Clear);
            break;
        case SDLK_Q:
        case SDLK_PAGEUP:
            if (!rep) PushNav(Nav::Prev);
            break;
        case SDLK_E:
        case SDLK_PAGEDOWN:
        case SDLK_TAB:
            if (!rep) PushNav(Nav::Next);
            break;
        default: break;
        }
        return true;
    }
    if (type == SDL_EVENT_KEY_UP) {
        return false;
    }
    if (type == SDL_EVENT_TEXT_INPUT) {
        return true;
    }
    if (type == SDL_EVENT_MOUSE_BUTTON_DOWN || type == SDL_EVENT_MOUSE_WHEEL ||
        type == SDL_EVENT_MOUSE_MOTION) {
        ImGui::Core::ProcessEvent(&event);
        return true;
    }
    if (type == SDL_EVENT_MOUSE_BUTTON_UP) {
        ImGui::Core::ProcessEvent(&event);
        return false; // also to the game's input state, so nothing stays held
    }
    return false;
}


// (bb-trophy-134) guest open() hook: map loads for the custom trophies. Any thread, cheap.
void OnGuestFileOpen(const char* path, int flags) {
    if (path == nullptr) {
        return;
    }
    const std::string_view p{path};
    if (p.starts_with("/savedata0/userdata") && p.size() == 23) {
        const int slot = (p[19] - '0') * 1000 + (p[20] - '0') * 100 + (p[21] - '0') * 10 + (p[22] - '0');
        const bool write = (flags & 3) != 0;
        if (g_slot_logged.fetch_add(1) < 60) {
            LOG_INFO(Frontend, "[bbtrophy] save slot userdata{:04d} {} (map loaded this session: {})",
                     slot, write ? "write" : "read", g_any_map_session.load());
        }
        if (slot == 10 && !write) {
            // system data read: boot / title menu. A new game from here is a fresh session.
            g_any_map_session = false;
            g_slot_read_mask = 0;
        }
        if (slot >= 0 && slot <= 9) {
            g_active_slot = slot;
            const unsigned bit = 1u << slot;
            if (!write) {
                g_slot_read_mask.fetch_or(bit);
            } else if (!g_any_map_session.load() && (g_slot_read_mask.load() & bit) == 0) {
                g_slot_read_mask.fetch_or(bit); // once per slot and session
                LOG_INFO(Frontend,
                         "[bbtrophy] new game: character slot userdata{:04d} written before any "
                         "map load, without being read first",
                         slot);
                g_newgame_slot = slot;
            }
        }
        return;
    }
    const auto ev = p.find("/event/m");
    if (ev != std::string_view::npos && p.ends_with(".emevd.dcx") && p.size() >= ev + 7 + 12) {
        const std::string id{p.substr(ev + 7, 12)}; // mAA_BB_CC_DD
        if (id[3] != '_' || id[6] != '_' || id[9] != '_') {
            return;
        }
        const char* what = id == kMapYharnam ? " (Central Yharnam)"
                           : id == kMapDream ? " (Hunter's Dream)"
                                             : "";
        LOG_INFO(Frontend, "[bbtrophy] map load {}{}", id, what);
        g_any_map_session = true;
        std::scoped_lock lk{g_map_mtx};
        g_map_loads.push_back({id, std::chrono::steady_clock::now()});
        g_last_map = id;
        g_last_map_at = std::chrono::steady_clock::now();
        return;
    }
    // evidence for a later exact "left the clinic" rule: first opens of Central Yharnam parts
    const auto mp = p.find("/map/m24_01_00_00/");
    if (mp != std::string_view::npos && p.ends_with(".flver.dcx") &&
        g_map_parts_logged.fetch_add(1) < 400) {
        LOG_INFO(Frontend, "[bbtrophy] m24_01 part {}", p.substr(mp + 18));
    }
}

} // namespace BbMenu
