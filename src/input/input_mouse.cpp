// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/types.h"
#include "input/controller.h"
#include "input/input_handler.h"
#include "input_mouse.h"

#include <common/singleton.h>
#include <emulator.h>
#include "SDL3/SDL.h"

extern Frontend::WindowSDL* g_window;

namespace Input {

extern std::list<std::pair<InputEvent, bool>> pressed_keys;

int mouse_joystick_binding = 2; // (bb-input-135) right stick unless the config says otherwise
float mouse_deadzone_offset = 0.5, mouse_speed = 1, mouse_speed_offset = 0.1250;
bool mouse_gyro_roll_mode = false;
bool mouse_invert_y = false;
Uint32 mouse_polling_id = 0;
MouseMode mouse_mode = MouseMode::Off;

// Switches mouse to a set mode or turns mouse emulation off if it was already in that mode.
// Returns whether the mode is turned on.
MouseMode GetMouseMode() {
    return mouse_mode;
}

bool ToggleMouseModeTo(MouseMode m) {
    if (mouse_mode == m) {
        mouse_mode = MouseMode::Off;
    } else {
        mouse_mode = m;
    }
    return mouse_mode == m;
}

void SetMouseToJoystick(int joystick) {
    mouse_joystick_binding = joystick;
}

void SetMouseParams(float mdo, float ms, float mso) {
    mouse_deadzone_offset = mdo;
    mouse_speed = ms;
    mouse_speed_offset = mso;
}

// (bb-input-137) F1 menu sensitivity (bb_menu.ini mouse_sens), wins over the config's speed
static float g_sens_override = 0.0f;
void SetMouseSensitivity(float s) {
    g_sens_override = std::clamp(s, 0.0f, 10.0f);
}
float GetMouseSensitivity() {
    return g_sens_override > 0 ? g_sens_override : mouse_speed;
}

static float EnvF(const char* n, float d) {
    const char* v = std::getenv(n);
    return v && *v ? static_cast<float>(std::atof(v)) : d;
}

void SetMouseInvertY(bool invert) {
    mouse_invert_y = invert;
}

void SetMouseGyroRollMode(bool mode) {
    mouse_gyro_roll_mode = mode;
}

void EmulateJoystick(GameController* controller, u32 interval) {

    Axis axis_x, axis_y;
    switch (mouse_joystick_binding) {
    case 1:
        axis_x = Axis::LeftX;
        axis_y = Axis::LeftY;
        break;
    case 2:
        axis_x = Axis::RightX;
        axis_y = Axis::RightY;
        break;
    default:
        return; // no update needed
    }

    float d_x = 0, d_y = 0;
    SDL_GetRelativeMouseState(&d_x, &d_y);
    if (mouse_invert_y) {
        d_y = -d_y;
    }

    // (bb-input-137) linear mouse -> stick: no jump to half tilt on the first count (the old
    // deadzone offset 0.5 of 128), counts/s * sensitivity map to stick tilt, the tilt is
    // pre-curved (^1/gamma) against the game's own stick -> rate curve (FromSoftware cameras turn
    // much slower than linear below full tilt; the DeS port measured ~stick^3), and what a
    // saturated stick cannot turn is carried to the next polls (capped, BB_KBM_CARRY_MS) so the
    // total rotation follows the mouse. BB_KBM_LEGACY=1: the 1.36 mapping.
    static const bool legacy = EnvF("BB_KBM_LEGACY", 0) != 0;
    static const float gamma = std::clamp(EnvF("BB_KBM_GAMMA", 2.0f), 1.0f, 4.0f);
    static const float game_dz = std::clamp(EnvF("BB_KBM_DZ", 0.10f), 0.0f, 0.5f);
    static const float carry_ms = std::clamp(EnvF("BB_KBM_CARRY_MS", 120.0f), 0.0f, 1000.0f);
    static const float full_cps = std::max(EnvF("BB_KBM_FULL_CPS", 1200.0f), 50.0f);
    static float carry_x = 0, carry_y = 0;
    static float carried_sum = 0;
    float a_x = 0, a_y = 0;
    bool active = d_x != 0 || d_y != 0;
    if (legacy) {
        float output_speed =
            SDL_clamp(sqrt(d_x * d_x + d_y * d_y) * mouse_speed + mouse_speed_offset * 128,
                      mouse_deadzone_offset * 128, 128.0);
        float angle = atan2(d_y, d_x);
        a_x = cos(angle) * output_speed, a_y = sin(angle) * output_speed;
    } else {
        const float sens = std::max(GetMouseSensitivity(), 0.05f);
        const float ms = interval > 0 ? static_cast<float>(interval) : 33.0f;
        const float full = full_cps * ms / 1000.0f / sens; // counts this poll for full tilt
        const float cx = d_x + carry_x, cy = d_y + carry_y;
        const float c = std::sqrt(cx * cx + cy * cy);
        carry_x = carry_y = 0;
        if (c > 0.01f) {
            active = true;
            float mag = c / full;
            if (mag > 1.0f) {
                float ex = cx - cx / mag, ey = cy - cy / mag;
                const float e = std::sqrt(ex * ex + ey * ey);
                const float cap = full * carry_ms / ms;
                if (e > cap) ex *= cap / e, ey *= cap / e;
                carry_x = ex, carry_y = ey;
                carried_sum += std::min(e, cap);
                mag = 1.0f;
            }
            const float tilt = game_dz + (1.0f - game_dz) * std::pow(mag, 1.0f / gamma);
            a_x = cx / c * tilt * 128.0f, a_y = cy / c * tilt * 128.0f;
        }
    }

    // (bb-input-135) mouse wins while it moves; at rest the stick is centred only once, so a
    // physical pad's right stick is not overwritten with "centre" every 33 ms.
    static bool was_moving = false;
    const bool moving = active;
    int out_x = 0, out_y = 0;
    if (moving) {
        out_x = GetAxis(-0x80, 0x7f, a_x);
        out_y = GetAxis(-0x80, 0x7f, a_y);
        controller->Axis(axis_x, out_x, false);
        controller->Axis(axis_y, out_y, false);
    } else if (was_moving) {
        controller->Axis(axis_x, GetAxis(-0x80, 0x7f, 0), false);
        controller->Axis(axis_y, GetAxis(-0x80, 0x7f, 0), false);
    }
    was_moving = moving;

    // once a second: what the mouse did and what went to the stick
    static float sum_x = 0, sum_y = 0;
    static int last_x = 0, last_y = 0, frames = 0;
    static Uint64 last_log = 0;
    sum_x += d_x, sum_y += d_y;
    if (moving) last_x = out_x, last_y = out_y;
    frames++;
    const Uint64 now = SDL_GetTicks();
    if (now - last_log >= 1000) {
        const bool captured = g_window != nullptr &&
                              SDL_GetWindowRelativeMouseMode(g_window->GetSDLWindow());
        LOG_INFO(Input,
                 "[kbm] mouse dx/dy {:.0f}/{:.0f} ({} polls), stick out {}/{} ({} stick), "
                 "captured {}, sens {:.2f}, carried {:.0f} counts (pending {:.0f}), {}",
                 sum_x, sum_y, frames, last_x, last_y, axis_x == Axis::RightX ? "right" : "left",
                 captured ? "yes" : "no", GetMouseSensitivity(), carried_sum,
                 std::sqrt(carry_x * carry_x + carry_y * carry_y),
                 legacy ? "legacy" : "linear+carry");
        carried_sum = 0;
        sum_x = sum_y = 0, frames = 0, last_x = last_y = 0, last_log = now;
    }
}

constexpr float constant_down_accel[3] = {0.0f, 9.81f, 0.0f};
void EmulateGyro(GameController* controller, u32 interval) {
    float d_x = 0, d_y = 0;
    SDL_GetRelativeMouseState(&d_x, &d_y);
    controller->UpdateAcceleration(constant_down_accel);
    float gyro_from_mouse[3] = {-d_y / 100, -d_x / 100, 0.0f};
    if (mouse_gyro_roll_mode) {
        gyro_from_mouse[1] = 0.0f;
        gyro_from_mouse[2] = -d_x / 100;
    }
    controller->UpdateGyro(gyro_from_mouse);
}

void EmulateTouchpad(GameController* controller, u32 interval) {
    float x, y;
    SDL_MouseButtonFlags mouse_buttons = SDL_GetMouseState(&x, &y);
    controller->SetTouchpadState(0, (mouse_buttons & SDL_BUTTON_LMASK) != 0,
                                 std::clamp(x / g_window->GetWidth(), 0.0f, 1.0f),
                                 std::clamp(y / g_window->GetHeight(), 0.0f, 1.0f));
    controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad,
                       (mouse_buttons & SDL_BUTTON_RMASK) != 0);
}

void ApplyMouseInputBlockers() {
    switch (mouse_mode) {
    case MouseMode::Touchpad:
        for (auto& k : pressed_keys) {
            if (k.first.input.sdl_id == SDL_BUTTON_LEFT ||
                k.first.input.sdl_id == SDL_BUTTON_RIGHT) {
                k.second = true;
            }
        }
        break;
    default:
        break;
    }
}

Uint32 MousePolling(void* param, Uint32 id, Uint32 interval) {
    auto* controller = (GameController*)param;
    switch (mouse_mode) {
    case MouseMode::Joystick:
        EmulateJoystick(controller, interval);
        break;
    case MouseMode::Gyro:
        EmulateGyro(controller, interval);
        break;
    case MouseMode::Touchpad:
        EmulateTouchpad(controller, interval);
        break;

    default:
        break;
    }
    return interval;
}

} // namespace Input
