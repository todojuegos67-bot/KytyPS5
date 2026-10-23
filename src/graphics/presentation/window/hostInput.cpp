#include "graphics/presentation/window/hostInput.h"

#include "SDL_error.h"
#include "SDL_events.h"
#include "SDL_keyboard.h"
#include "SDL_keycode.h"
#include "SDL_mouse.h"
#include "SDL_timer.h"
#include "SDL_video.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "libs/controller.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <string_view>

namespace Libs::Graphics {

namespace {

struct ControlInfo {
	std::string_view name;
	uint32_t         button   = 0;
	Controller::Axis axis     = Controller::Axis::AxisMax;
	bool             positive = false;
	float            touch_x  = 0.0f;
};

static constexpr std::array CONTROL_INFO = {
    ControlInfo {"L3", Controller::PAD_BUTTON_L3},
    ControlInfo {"R3", Controller::PAD_BUTTON_R3},
    ControlInfo {"Options", Controller::PAD_BUTTON_OPTIONS},
    ControlInfo {"Up", Controller::PAD_BUTTON_UP},
    ControlInfo {"Right", Controller::PAD_BUTTON_RIGHT},
    ControlInfo {"Down", Controller::PAD_BUTTON_DOWN},
    ControlInfo {"Left", Controller::PAD_BUTTON_LEFT},
    ControlInfo {"L2", Controller::PAD_BUTTON_L2},
    ControlInfo {"R2", Controller::PAD_BUTTON_R2},
    ControlInfo {"L1", Controller::PAD_BUTTON_L1},
    ControlInfo {"R1", Controller::PAD_BUTTON_R1},
    ControlInfo {"Triangle", Controller::PAD_BUTTON_TRIANGLE},
    ControlInfo {"Circle", Controller::PAD_BUTTON_CIRCLE},
    ControlInfo {"Cross", Controller::PAD_BUTTON_CROSS},
    ControlInfo {"Square", Controller::PAD_BUTTON_SQUARE},
    ControlInfo {"TouchPad", Controller::PAD_BUTTON_TOUCH_PAD, Controller::Axis::AxisMax, false,
                 0.25f},
    ControlInfo {"TouchPadRight", Controller::PAD_BUTTON_TOUCH_PAD, Controller::Axis::AxisMax,
                 false, 0.75f},
    ControlInfo {"LeftStickLeft", 0, Controller::Axis::LeftX, false},
    ControlInfo {"LeftStickRight", 0, Controller::Axis::LeftX, true},
    ControlInfo {"LeftStickUp", 0, Controller::Axis::LeftY, false},
    ControlInfo {"LeftStickDown", 0, Controller::Axis::LeftY, true},
    ControlInfo {"RightStickLeft", 0, Controller::Axis::RightX, false},
    ControlInfo {"RightStickRight", 0, Controller::Axis::RightX, true},
    ControlInfo {"RightStickUp", 0, Controller::Axis::RightY, false},
    ControlInfo {"RightStickDown", 0, Controller::Axis::RightY, true},
};

constexpr std::size_t INVALID_CONTROL = CONTROL_INFO.size();

struct Binding {
	SDL_Keycode key          = SDLK_UNKNOWN;
	uint8_t     mouse_button = 0;
	std::size_t control      = INVALID_CONTROL;
};

constexpr int              MOUSE_POLL_INTERVAL_MS = 33;
constexpr std::string_view MOUSE_SENSITIVITY      = "MouseSensitivity=";

struct MouseJoystickState {
	bool     enabled   = false;
	bool     output    = false;
	uint64_t next_poll = 0;
};

MouseJoystickState g_mouse;

std::size_t ControlFromName(std::string_view name) {
	const auto info = std::find_if(CONTROL_INFO.begin(), CONTROL_INFO.end(),
	                               [name](const auto& item) { return item.name == name; });
	return static_cast<std::size_t>(std::distance(CONTROL_INFO.begin(), info));
}

SDL_Keycode NormalizeKey(SDL_Keycode key) {
	switch (key) {
		case SDLK_RSHIFT: return SDLK_LSHIFT;
		case SDLK_RCTRL: return SDLK_LCTRL;
		case SDLK_RALT: return SDLK_LALT;
		case SDLK_RGUI: return SDLK_LGUI;
		default: return key;
	}
}

uint8_t MouseButtonFromName(std::string_view name) {
	static constexpr std::array buttons = {
	    std::pair {std::string_view("Mouse:Left"), uint8_t {SDL_BUTTON_LEFT}},
	    std::pair {std::string_view("Mouse:Middle"), uint8_t {SDL_BUTTON_MIDDLE}},
	    std::pair {std::string_view("Mouse:Right"), uint8_t {SDL_BUTTON_RIGHT}},
	    std::pair {std::string_view("Mouse:X1"), uint8_t {SDL_BUTTON_X1}},
	    std::pair {std::string_view("Mouse:X2"), uint8_t {SDL_BUTTON_X2}},
	};

	const auto button = std::find_if(buttons.begin(), buttons.end(),
	                                 [name](const auto& item) { return item.first == name; });
	return button != buttons.end() ? button->second : 0;
}

bool Conflicts(const Binding& first, const Binding& second) {
	return first.control == second.control ||
	       (first.key != SDLK_UNKNOWN && first.key == second.key) ||
	       (first.mouse_button != 0 && first.mouse_button == second.mouse_button);
}

class InputMap {
public:
	InputMap() {
		for (const auto& value: Config::GetKeymap()) {
			const std::string_view entry = value;
			if (entry.starts_with(MOUSE_SENSITIVITY)) {
				const float sensitivity =
				    std::strtof(value.c_str() + MOUSE_SENSITIVITY.size(), nullptr);
				m_mouse_sensitivity =
				    std::clamp(std::isfinite(sensitivity) ? sensitivity : 1.0f, 0.1f, 5.0f);
				continue;
			}
			const auto split = entry.find('=');

			Binding binding;
			if (split != std::string_view::npos) {
				binding.control       = ControlFromName(entry.substr(0, split));
				const auto host_input = entry.substr(split + 1);
				binding.mouse_button  = MouseButtonFromName(host_input);
				if (binding.mouse_button == 0) {
					binding.key = NormalizeKey(SDL_GetKeyFromName(std::string(host_input).c_str()));
				}
			}

			const bool reserved = binding.key == SDLK_ESCAPE || binding.key == SDLK_SPACE ||
			                      binding.key == SDLK_F7 || binding.key == SDLK_F8 ||
			                      binding.key == SDLK_F11;
			if (binding.control == INVALID_CONTROL || reserved ||
			    (binding.key == SDLK_UNKNOWN && binding.mouse_button == 0)) {
				EXIT("Invalid input mapping: %s\n", value.c_str());
			}
			Add(binding);
		}
	}
	[[nodiscard]] bool  Custom() const { return m_size != 0; }
	[[nodiscard]] float MouseSensitivity() const { return m_mouse_sensitivity; }

	[[nodiscard]] std::size_t FindKey(int key_code) const {
		key_code = NormalizeKey(static_cast<SDL_Keycode>(key_code));
		const auto binding =
		    std::find_if(m_bindings.begin(), m_bindings.begin() + m_size,
		                 [key_code](const auto& item) { return item.key == key_code; });
		return binding != m_bindings.begin() + m_size ? binding->control : INVALID_CONTROL;
	}

	[[nodiscard]] std::size_t FindMouseButton(uint8_t mouse_button) const {
		const auto binding = std::find_if(
		    m_bindings.begin(), m_bindings.begin() + m_size,
		    [mouse_button](const auto& item) { return item.mouse_button == mouse_button; });
		return binding != m_bindings.begin() + m_size ? binding->control : INVALID_CONTROL;
	}

private:
	void Add(const Binding& binding) {
		for (std::size_t index = 0; index < m_size;) {
			if (Conflicts(m_bindings[index], binding)) {
				m_bindings[index] = m_bindings[--m_size];
			} else {
				index++;
			}
		}
		EXIT_IF(m_size >= m_bindings.size());
		m_bindings[m_size++] = binding;
	}

	std::array<Binding, CONTROL_INFO.size()> m_bindings {};
	std::size_t                              m_size              = 0;
	float                                    m_mouse_sensitivity = 1.0f;
};

const InputMap& GetInputMap() {
	static const InputMap map;
	return map;
}

void SetButton(uint32_t button, bool down) {
	if (button == Controller::PAD_BUTTON_L2) {
		Controller::SetAxis(Controller::HOST_INPUT_CONTROLLER_ID, Controller::Axis::TriggerLeft,
		                    down ? 255 : 0);
	} else if (button == Controller::PAD_BUTTON_R2) {
		Controller::SetAxis(Controller::HOST_INPUT_CONTROLLER_ID, Controller::Axis::TriggerRight,
		                    down ? 255 : 0);
	} else if (button != 0) {
		Controller::SetButton(Controller::HOST_INPUT_CONTROLLER_ID, button, down);
	}
}

void SetTouchPad(float x, bool down) {
	Controller::SetTouchPad(Controller::HOST_INPUT_CONTROLLER_ID, 0, down, x, 0.5f);
}

uint32_t DefaultKeyboardButton(int key_code) {
	switch (NormalizeKey(static_cast<SDL_Keycode>(key_code))) {
		case SDLK_UP: return Controller::PAD_BUTTON_UP;
		case SDLK_LEFT: return Controller::PAD_BUTTON_LEFT;
		case SDLK_DOWN: return Controller::PAD_BUTTON_DOWN;
		case SDLK_RIGHT: return Controller::PAD_BUTTON_RIGHT;
		case SDLK_j: return Controller::PAD_BUTTON_CROSS;
		case SDLK_i: return Controller::PAD_BUTTON_TRIANGLE;
		case SDLK_k: return Controller::PAD_BUTTON_SQUARE;
		case SDLK_l: return Controller::PAD_BUTTON_CIRCLE;
		case SDLK_q: return Controller::PAD_BUTTON_L1;
		case SDLK_e: return Controller::PAD_BUTTON_R1;
		case SDLK_LSHIFT: return Controller::PAD_BUTTON_L3;
		case SDLK_LCTRL: return Controller::PAD_BUTTON_R3;
		case SDLK_RETURN:
		case SDLK_RETURN2: return Controller::PAD_BUTTON_OPTIONS;
		default: return 0;
	}
}

struct StickKeys {
	bool left  = false;
	bool right = false;
	bool up    = false;
	bool down  = false;
};

void SetStickAxis(Controller::Axis axis, bool negative, bool positive) {
	int value = 128;
	if (negative && !positive) {
		value = 0;
	} else if (positive && !negative) {
		value = 255;
	}
	Controller::SetAxis(Controller::HOST_INPUT_CONTROLLER_ID, axis, value);
}

void SetControl(std::size_t control, bool down) {
	if (control == INVALID_CONTROL) {
		return;
	}

	const auto& info = CONTROL_INFO[control];
	if (info.button == Controller::PAD_BUTTON_TOUCH_PAD) {
		SetTouchPad(info.touch_x, down);
		return;
	}
	if (info.button != 0) {
		SetButton(info.button, down);
		return;
	}

	struct AxisKeys {
		bool negative = false;
		bool positive = false;
	};
	static std::array<AxisKeys, 4> axes;

	const auto axis = static_cast<std::size_t>(info.axis);
	EXIT_IF(axis >= axes.size());
	if (info.positive) {
		axes[axis].positive = down;
	} else {
		axes[axis].negative = down;
	}
	SetStickAxis(info.axis, axes[axis].negative, axes[axis].positive);
}

void DefaultKeyboardInput(int key_code, bool down) {
	static StickKeys left;
	static StickKeys right;

	switch (NormalizeKey(static_cast<SDL_Keycode>(key_code))) {
		case SDLK_BACKSPACE: SetTouchPad(0.25f, down); return;
		case SDLK_TAB: SetTouchPad(0.75f, down); return;
		case SDLK_a:
			left.left = down;
			SetStickAxis(Controller::Axis::LeftX, left.left, left.right);
			return;
		case SDLK_d:
			left.right = down;
			SetStickAxis(Controller::Axis::LeftX, left.left, left.right);
			return;
		case SDLK_w:
			left.up = down;
			SetStickAxis(Controller::Axis::LeftY, left.up, left.down);
			return;
		case SDLK_s:
			left.down = down;
			SetStickAxis(Controller::Axis::LeftY, left.up, left.down);
			return;
		case SDLK_f:
			right.left = down;
			SetStickAxis(Controller::Axis::RightX, right.left, right.right);
			return;
		case SDLK_h:
			right.right = down;
			SetStickAxis(Controller::Axis::RightX, right.left, right.right);
			return;
		case SDLK_t:
			right.up = down;
			SetStickAxis(Controller::Axis::RightY, right.up, right.down);
			return;
		case SDLK_g:
			right.down = down;
			SetStickAxis(Controller::Axis::RightY, right.up, right.down);
			return;
		default: SetButton(DefaultKeyboardButton(key_code), down); return;
	}
}

void MouseToJoystick(int delta_x, int delta_y) {
	const double distance = std::hypot(delta_x, delta_y);
	const double scale =
	    std::clamp(distance * GetInputMap().MouseSensitivity() + 16.0, 64.0, 128.0) / distance;
	const auto map_axis = [scale](int delta) {
		return std::clamp(128 + static_cast<int>(std::lround(delta * scale)), 0, 255);
	};
	Controller::SetRightStick(Controller::HOST_INPUT_CONTROLLER_ID, map_axis(delta_x),
	                          map_axis(delta_y));
}

void CenterMouseStick() {
	if (!g_mouse.output) {
		return;
	}
	Controller::SetRightStick(Controller::HOST_INPUT_CONTROLLER_ID, 128, 128);
	g_mouse.output = false;
}

bool SetRelativeMouseMode(bool enabled) {
	if (SDL_SetRelativeMouseMode(enabled ? SDL_TRUE : SDL_FALSE) == 0) {
		return true;
	}
	LOGF("Mouse-to-joystick relative mode failed: %s\n", SDL_GetError());
	return false;
}

// ---- KYTY_KBM=1: keyboard and mouse play (opt-in; without it nothing below runs) ----
// The mouse: relative mode (SDL reads Windows Raw Input, WM_INPUT, unaccelerated counts), the cursor hidden and
// confined (window grab, ClipCursor); Esc or losing focus releases it, a click takes it again. Counts accumulate
// (1/65536 count fixed point) and become the right stick once per guest pad read: deflection = sensitivity *
// rate / KBM_FULL_RATE, curve ^exponent, an anti-deadzone floor, the counts a saturated stick cannot show yet carried
// to the next read. Without counts the stick decays to centre (KYTY_KBM_DECAY_MS). Once a second the log has the
// counts received, applied and pending.
constexpr double KBM_FULL_RATE = 2000.0; // counts/s for a full deflection at sensitivity 1
constexpr int64_t KBM_ONE      = 65536;

enum KbmPseudo : std::size_t { KBM_WALK = CONTROL_INFO.size() + 1 };

struct KbmBinding {
	SDL_Keycode key          = SDLK_UNKNOWN;
	uint8_t     mouse_button = 0;  // SDL_BUTTON_*; 100/101: wheel up/down
	bool        shift        = false;
	std::size_t control      = INVALID_CONTROL;
};

struct KbmState {
	bool                    enabled  = false;
	float                   sens     = 1.0f;
	float                   expo     = 1.0f;
	float                   adz      = 0.15f;
	float                   decay_ms = 25.0f;
	float                   walk     = 0.5f;
	std::vector<KbmBinding> bindings;
	bool                    captured = false;
	bool                    walking  = false;
	std::array<std::size_t, 8> held_mouse {}; // the control each mouse button pressed (released with it)
	std::array<bool, 4>     stick_keys {};    // left, right, up, down
	std::atomic<int64_t>    acc_x {0};
	std::atomic<int64_t>    acc_y {0};
	std::atomic<int64_t>    received {0};
	std::atomic<int64_t>    applied {0};
	std::atomic<int64_t>    wheel_release_ns {0};
	std::atomic<uint32_t>   wheel_control {0};
	// Guest pad thread only:
	int64_t last_ns = 0;
	int64_t log_ns  = 0;
	int     polls   = 0;
	double  out_x   = 0.0;
	double  out_y   = 0.0;
};

KbmState g_kbm;

int64_t KbmNow() {
	return static_cast<int64_t>(SDL_GetPerformanceCounter() * 1e9 / static_cast<double>(SDL_GetPerformanceFrequency()));
}

float EnvFloat(const char* name, float def, float lo, float hi) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return def;
	const float f = std::strtof(v, nullptr);
	return std::isfinite(f) ? std::clamp(f, lo, hi) : def;
}

bool KbmParse(std::string_view entry, KbmBinding* out) {
	const auto split = entry.find('=');
	if (split == std::string_view::npos) return false;
	const auto control = entry.substr(0, split);
	auto       host    = entry.substr(split + 1);
	KbmBinding b;
	b.control = control == "Walk" ? static_cast<std::size_t>(KBM_WALK) : ControlFromName(control);
	if (b.control == INVALID_CONTROL) return false;
	if (host == "None" || host.empty()) {
		b.key = SDLK_UNKNOWN;
		*out  = b;
		return true;
	}
	if (host.starts_with("Shift+")) {
		b.shift = true;
		host    = host.substr(6);
	}
	if (host == "Mouse:WheelUp") {
		b.mouse_button = 100;
	} else if (host == "Mouse:WheelDown") {
		b.mouse_button = 101;
	} else {
		b.mouse_button = MouseButtonFromName(host);
	}
	if (b.mouse_button == 0) {
		b.key = NormalizeKey(SDL_GetKeyFromName(std::string(host).c_str()));
		if (b.key == SDLK_UNKNOWN) return false;
	}
	*out = b;
	return true;
}

constexpr std::string_view KBM_DEFAULTS =
    "LeftStickUp=W;LeftStickDown=S;LeftStickLeft=A;LeftStickRight=D;Walk=Left Ctrl;"
    "R1=Mouse:Left;L1=Mouse:Right;R2=Shift+Mouse:Left;L2=Shift+Mouse:Right;"
    "R3=Mouse:Middle;R3=Mouse:WheelUp;R3=Mouse:WheelDown;"
    "Circle=Space;Square=E;Cross=F;Triangle=G;Left=Q;Right=R;"
    "Up=1;Down=2;Left=3;Right=4;Options=Tab;TouchPad=Backspace";

void KbmAddList(std::string_view list, bool user) {
	std::vector<std::size_t> replaced;
	while (!list.empty()) {
		const auto end   = list.find(';');
		const auto entry = list.substr(0, end);
		list             = end == std::string_view::npos ? std::string_view {} : list.substr(end + 1);
		if (entry.empty()) continue;
		KbmBinding b;
		if (!KbmParse(entry, &b)) {
			LOGF("KYTY_KBM: ignored mapping '%s'\n", std::string(entry).c_str());
			continue;
		}
		if (user && std::find(replaced.begin(), replaced.end(), b.control) == replaced.end()) {
			// A control the user maps loses its default inputs; an input the user takes leaves its default control.
			replaced.push_back(b.control);
			std::erase_if(g_kbm.bindings, [&](const auto& o) { return o.control == b.control; });
		}
		if (user && (b.key != SDLK_UNKNOWN || b.mouse_button != 0)) {
			std::erase_if(g_kbm.bindings, [&](const auto& o) {
				return o.control != b.control && o.key == b.key && o.mouse_button == b.mouse_button && o.shift == b.shift;
			});
		}
		if (b.key != SDLK_UNKNOWN || b.mouse_button != 0) g_kbm.bindings.push_back(b);
	}
}

void KbmInit() {
	const char* on = std::getenv("KYTY_KBM");
	if (on == nullptr || std::string_view(on) != "1") return;
	g_kbm.enabled  = true;
	g_kbm.sens     = EnvFloat("KYTY_KBM_SENS", 1.0f, 0.05f, 20.0f);
	g_kbm.expo     = EnvFloat("KYTY_KBM_EXP", 1.0f, 0.3f, 3.0f);
	g_kbm.adz      = EnvFloat("KYTY_KBM_ANTIDZ", 0.15f, 0.0f, 0.6f);
	g_kbm.decay_ms = EnvFloat("KYTY_KBM_DECAY_MS", 25.0f, 0.0f, 500.0f);
	g_kbm.walk     = EnvFloat("KYTY_KBM_WALK", 0.5f, 0.1f, 1.0f);
	KbmAddList(KBM_DEFAULTS, false);
	if (const char* map = std::getenv("KYTY_KBM_MAP"); map != nullptr) KbmAddList(map, true);
	LOGF("KYTY_KBM: keyboard and mouse on (sens %.2f, exp %.2f, antideadzone %.2f, decay %.0f ms, %zu bindings)\n",
	     g_kbm.sens, g_kbm.expo, g_kbm.adz, g_kbm.decay_ms, g_kbm.bindings.size());
	Controller::SetRightStickProvider(+[](int* x, int* y) -> bool {
		auto&         k   = g_kbm;
		const int64_t now = KbmNow();
		double        dt  = k.last_ns == 0 ? 1.0 / 60.0 : static_cast<double>(now - k.last_ns) * 1e-9;
		dt                = std::clamp(dt, 0.001, 0.1);
		k.last_ns         = now;
		k.polls++;

		if (const auto rel = k.wheel_release_ns.load(std::memory_order_acquire); rel != 0 && now >= rel) {
			k.wheel_release_ns.store(0, std::memory_order_relaxed);
			SetButton(k.wheel_control.load(std::memory_order_relaxed), false);
		}

		const int64_t ax  = k.acc_x.load(std::memory_order_acquire);
		const int64_t ay  = k.acc_y.load(std::memory_order_acquire);
		const double  cx  = static_cast<double>(ax) / KBM_ONE;
		const double  cy  = static_cast<double>(ay) / KBM_ONE;
		const double  mag = std::hypot(cx, cy) / dt * k.sens / KBM_FULL_RATE;
		if (mag > 0.0) {
			// A saturated stick shows part of the counts now; the rest stays for the next read (none is dropped).
			const double  f    = mag > 1.0 ? 1.0 / mag : 1.0;
			const int64_t used_x = f >= 1.0 ? ax : static_cast<int64_t>(std::llround(static_cast<double>(ax) * f));
			const int64_t used_y = f >= 1.0 ? ay : static_cast<int64_t>(std::llround(static_cast<double>(ay) * f));
			k.acc_x.fetch_sub(used_x, std::memory_order_acq_rel);
			k.acc_y.fetch_sub(used_y, std::memory_order_acq_rel);
			k.applied.fetch_add(std::llabs(used_x) + std::llabs(used_y), std::memory_order_relaxed);
			const double m   = std::min(mag, 1.0);
			const double out = k.adz + (1.0 - k.adz) * std::pow(m, static_cast<double>(k.expo));
			const double len = std::hypot(cx, cy);
			k.out_x          = cx / len * out;
			k.out_y          = cy / len * out;
		} else if (k.decay_ms <= 0.0f) {
			k.out_x = k.out_y = 0.0;
		} else {
			const double d = std::exp(-dt * 1000.0 / k.decay_ms);
			k.out_x *= d;
			k.out_y *= d;
			if (std::hypot(k.out_x, k.out_y) < k.adz * 0.5 + 0.01) k.out_x = k.out_y = 0.0;
		}

		if (k.log_ns == 0) k.log_ns = now;
		if (now - k.log_ns >= 1000000000) {
			const auto rec = k.received.exchange(0, std::memory_order_relaxed);
			const auto app = k.applied.exchange(0, std::memory_order_relaxed);
			const auto pend = std::llabs(k.acc_x.load(std::memory_order_relaxed)) +
			                  std::llabs(k.acc_y.load(std::memory_order_relaxed));
			if (rec != 0 || app != 0) {
				LOGF("KYTY_KBM mouse: received %.0f applied %.2f pending %.2f counts in %.2f s, %d pad reads\n",
				     static_cast<double>(rec) / KBM_ONE, static_cast<double>(app) / KBM_ONE,
				     static_cast<double>(pend) / KBM_ONE, (now - k.log_ns) * 1e-9, k.polls);
			}
			k.log_ns = now;
			k.polls  = 0;
		}

		if (!k.captured && k.out_x == 0.0 && k.out_y == 0.0) return false;
		if (k.out_x == 0.0 && k.out_y == 0.0) return false; // centred: the host pad's stick stays usable
		*x = std::clamp(static_cast<int>(std::lround(128.0 + k.out_x * 127.5)), 0, 255);
		*y = std::clamp(static_cast<int>(std::lround(128.0 + k.out_y * 127.5)), 0, 255);
		return true;
	});
}

void KbmCapture(bool on) {
	if (!g_kbm.enabled || g_kbm.captured == on) return;
	SDL_Window* window = SDL_GetKeyboardFocus();
	if (on && window == nullptr) return;
	if (SDL_SetRelativeMouseMode(on ? SDL_TRUE : SDL_FALSE) != 0) {
		LOGF("KYTY_KBM: relative mouse mode failed: %s\n", SDL_GetError());
		return;
	}
	if (window != nullptr) SDL_SetWindowGrab(window, on ? SDL_TRUE : SDL_FALSE);
	g_kbm.captured = on;
	if (!on) {
		g_kbm.acc_x.store(0);
		g_kbm.acc_y.store(0);
	}
	LOGF("KYTY_KBM: mouse %s\n", on ? "captured (Esc releases)" : "released (click to capture)");
}

void KbmLeftStick() {
	const auto& s     = g_kbm.stick_keys;
	const int   reach = g_kbm.walking ? static_cast<int>(std::lround(127.0f * g_kbm.walk)) : 127;
	const auto  value = [reach](bool neg, bool pos) { return neg == pos ? 128 : (neg ? 128 - reach - 1 : 128 + reach); };
	Controller::SetAxis(Controller::HOST_INPUT_CONTROLLER_ID, Controller::Axis::LeftX,
	                    std::clamp(value(s[0], s[1]), 0, 255));
	Controller::SetAxis(Controller::HOST_INPUT_CONTROLLER_ID, Controller::Axis::LeftY,
	                    std::clamp(value(s[2], s[3]), 0, 255));
}

void KbmControl(std::size_t control, bool down) {
	if (control == KBM_WALK) {
		g_kbm.walking = down;
		KbmLeftStick();
		return;
	}
	if (control >= CONTROL_INFO.size()) return;
	const auto& info = CONTROL_INFO[control];
	if (info.axis == Controller::Axis::LeftX || info.axis == Controller::Axis::LeftY) {
		const std::size_t i = (info.axis == Controller::Axis::LeftX ? 0 : 2) + (info.positive ? 1 : 0);
		g_kbm.stick_keys[i] = down;
		KbmLeftStick();
		return;
	}
	SetControl(control, down);
}

const KbmBinding* KbmFind(SDL_Keycode key, uint8_t mouse_button, bool shift) {
	const KbmBinding* plain = nullptr;
	for (const auto& b: g_kbm.bindings) {
		if (b.key == key && b.mouse_button == mouse_button) {
			if (b.shift == shift) return &b;
			if (!b.shift) plain = &b;
		}
	}
	return plain;
}
} // namespace

void HostInputInit() {
	KbmInit();
	if (!g_kbm.enabled) GetInputMap();
}

bool HostInputKbmEnabled() {
	return g_kbm.enabled;
}

void HostInputFocus(bool focused) {
	if (!g_kbm.enabled) return;
	if (focused) {
		KbmCapture(true);
	} else {
		KbmCapture(false);
		Controller::ResetInputState();
	}
}

bool HostInputMouseMotion(int dx, int dy) {
	if (!g_kbm.enabled) return false;
	if (g_kbm.captured) {
		g_kbm.acc_x.fetch_add(static_cast<int64_t>(dx) * KBM_ONE, std::memory_order_acq_rel);
		g_kbm.acc_y.fetch_add(static_cast<int64_t>(dy) * KBM_ONE, std::memory_order_acq_rel);
		g_kbm.received.fetch_add((std::llabs(dx) + std::llabs(dy)) * KBM_ONE, std::memory_order_relaxed);
	}
	return true;
}

bool HostInputMouseWheel(int y) {
	if (!g_kbm.enabled) return false;
	if (!g_kbm.captured || y == 0) return true;
	const auto* b = KbmFind(SDLK_UNKNOWN, y > 0 ? 100 : 101, (SDL_GetModState() & KMOD_SHIFT) != 0);
	if (b != nullptr && b->control < CONTROL_INFO.size() && CONTROL_INFO[b->control].button != 0) {
		// A notch is a short press: down now, up about 80 ms later (at a guest pad read).
		g_kbm.wheel_control.store(CONTROL_INFO[b->control].button, std::memory_order_relaxed);
		SetButton(CONTROL_INFO[b->control].button, true);
		g_kbm.wheel_release_ns.store(KbmNow() + 80000000, std::memory_order_release);
	}
	return true;
}

void HostInputKey(int key_code, bool down) {
	if (g_kbm.enabled) {
		const auto key = NormalizeKey(static_cast<SDL_Keycode>(key_code));
		if (key == SDLK_ESCAPE && g_kbm.captured) {
			if (down) KbmCapture(false);
			return;
		}
		const auto* b = KbmFind(key == SDLK_ESCAPE ? SDLK_TAB : key, 0, false); // Esc, released: Options' key
		if (b != nullptr) KbmControl(b->control, down);
		return;
	}
	const auto& map = GetInputMap();
	if (map.Custom()) {
		SetControl(map.FindKey(key_code), down);
	} else {
		DefaultKeyboardInput(key_code, down);
	}
}

void HostInputMouseButton(uint8_t mouse_button, bool down) {
	if (g_kbm.enabled) {
		if (mouse_button == 0 || mouse_button >= g_kbm.held_mouse.size()) return;
		if (!g_kbm.captured) {
			if (down) KbmCapture(true); // the capturing click goes no further
			return;
		}
		auto& held = g_kbm.held_mouse[mouse_button];
		if (down) {
			const auto* b = KbmFind(SDLK_UNKNOWN, mouse_button, (SDL_GetModState() & KMOD_SHIFT) != 0);
			held          = b != nullptr ? b->control + 1 : 0;
			if (held != 0) KbmControl(held - 1, true);
		} else if (held != 0) {
			KbmControl(held - 1, false);
			held = 0;
		}
		return;
	}
	const auto& map = GetInputMap();
	if (map.Custom() && mouse_button != 0) {
		SetControl(map.FindMouseButton(mouse_button), down);
	}
}

void HostInputToggleMouseToJoystick() {
	if (g_kbm.enabled) {
		KbmCapture(!g_kbm.captured);
		return;
	}
	if (g_mouse.enabled) {
		SetRelativeMouseMode(false);
		CenterMouseStick();
		g_mouse = {};
		LOGF("Mouse to right stick: disabled\n");
		return;
	}

	if (!SetRelativeMouseMode(true)) {
		return;
	}
	int ignored_x = 0;
	int ignored_y = 0;
	SDL_GetRelativeMouseState(&ignored_x, &ignored_y);
	g_mouse.enabled   = true;
	g_mouse.next_poll = SDL_GetTicks64() + MOUSE_POLL_INTERVAL_MS;
	LOGF("Mouse to right stick: enabled (F7 to release)\n");
}

int PollMouse(uint64_t now_ms) {
	if (now_ms < g_mouse.next_poll) {
		return static_cast<int>(g_mouse.next_poll - now_ms);
	}
	g_mouse.next_poll = now_ms + MOUSE_POLL_INTERVAL_MS;

	int delta_x = 0;
	int delta_y = 0;
	SDL_GetRelativeMouseState(&delta_x, &delta_y);
	if (delta_x == 0 && delta_y == 0) {
		CenterMouseStick();
		return MOUSE_POLL_INTERVAL_MS;
	}

	MouseToJoystick(delta_x, delta_y);
	g_mouse.output = true;
	return MOUSE_POLL_INTERVAL_MS;
}

bool HostInputWaitEvent(SDL_Event* event) {
	if (!g_mouse.enabled || SDL_GetKeyboardFocus() == nullptr) {
		CenterMouseStick();
		if (SDL_WaitEvent(event) == 0) {
			EXIT("%s\n", SDL_GetError());
		}
		return true;
	}

	const int timeout_ms = PollMouse(SDL_GetTicks64());
	SDL_ClearError();
	if (SDL_WaitEventTimeout(event, timeout_ms) != 0) {
		return true;
	}
	if (SDL_GetError()[0] != '\0') {
		EXIT("%s\n", SDL_GetError());
	}
	return false;
}

} // namespace Libs::Graphics
