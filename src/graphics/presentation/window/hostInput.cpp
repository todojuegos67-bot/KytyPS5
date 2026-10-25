#include "graphics/presentation/window/hostInput.h"

#include "SDL_error.h"
#include "SDL_events.h"
#include "SDL_filesystem.h"
#include "SDL_keyboard.h"
#include "SDL_keycode.h"
#include "SDL_mouse.h"
#include "SDL_timer.h"
#include "SDL_video.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "libs/controller.h"
#include "loader/demonsSoulsCamera.h"
#include "loader/gameArgs.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
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
// (1/65536 count fixed point) and become the right stick once per game frame: (10-25) deflection = counts/s * degrees
// per count (KYTY_KBM_DEG_PER_COUNT * sensitivity) / the camera's top rate (learned from the camera), ^exponent, an
// anti-deadzone floor; what a saturated stick cannot show is carried one frame at most, the rest dropped. Without counts the stick decays to centre (KYTY_KBM_DECAY_MS). Once a second the log has the
// counts received, applied and pending.
constexpr int64_t KBM_ONE      = 65536;

enum KbmPseudo : std::size_t { KBM_WALK = CONTROL_INFO.size() + 1 };

struct KbmBinding {
	SDL_Keycode key          = SDLK_UNKNOWN;
	uint8_t     mouse_button = 0;  // SDL_BUTTON_*; 100/101: wheel up/down
	bool        shift        = false;
	std::size_t control      = INVALID_CONTROL;
};

// ---- The camera the emulator sees (FrameGen::OnDraw, the G-buffer's view_proj, render thread) -> the pad thread ----
// Yaw about world +Y (KYTY_SSR_UP's default) of the forward row, unwrapped, with the host time it was seen.
struct CamSample {
	int64_t t     = 0;
	double  yaw   = 0.0;
	double  pitch = 0.0; // (10-25) degrees above the horizon (the KYTY_KBM_WARP pitch gain)
};
std::mutex                 g_cam_mutex;
std::array<CamSample, 256> g_cam_ring {};
uint64_t                   g_cam_count = 0;
double                     g_cam_raw   = 0.0;
double                     g_cam_turns = 0.0;

// The samples seen in [t0, t1] (oldest first).
std::vector<CamSample> CamSamples(int64_t t0, int64_t t1) {
	std::vector<CamSample> out;
	std::lock_guard        lock(g_cam_mutex);
	const uint64_t         n = std::min<uint64_t>(g_cam_count, g_cam_ring.size());
	for (uint64_t i = g_cam_count - n; i < g_cam_count; ++i) {
		const auto& s = g_cam_ring[i % g_cam_ring.size()];
		if (s.t >= t0 && s.t <= t1) out.push_back(s);
	}
	return out;
}

bool CamLatest(CamSample* s) {
	std::lock_guard lock(g_cam_mutex);
	if (g_cam_count == 0) return false;
	*s = g_cam_ring[(g_cam_count - 1) % g_cam_ring.size()];
	return true;
}

// Least squares yaw = a + b * t (t in s from t0) over samples; false with fewer than 3.
bool CamFit(const std::vector<CamSample>& v, int64_t t0, double* a, double* b) {
	if (v.size() < 3) return false;
	double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
	for (const auto& s: v) {
		const double x = (s.t - t0) * 1e-9;
		n += 1, sx += x, sy += s.yaw, sxx += x * x, sxy += x * s.yaw;
	}
	const double d = n * sxx - sx * sx;
	if (std::abs(d) < 1e-12) return false;
	*b = (n * sxy - sx * sy) / d;
	*a = (sy - *b * sx) / n;
	return true;
}

// ---- The game's stick -> camera response (KYTY_KBM_CALIBRATE=1 measures it into kbm_curve.txt next to the exe) ----
constexpr std::array<double, 5> KBM_LEVELS {0.2, 0.4, 0.6, 0.8, 1.0};

struct KbmCurve {
	bool                  valid = false;
	std::array<double, 5> rate {};     // deg/s at each KBM_LEVELS deflection (steady)
	double                onset_ms = 0; // stick -> first camera motion (pad reads, render, the game's own frames)
	double                tau_ms   = 0; // the game's acceleration ramp as a first-order lag
	double                sign     = 1; // yaw direction of +X stick
	int                   camspeed = 0; // (10-25) the game camera speed it was measured at (0: not recorded)
	int                   noramp   = 0; // (10-25) KYTY_KBM_NO_RAMP in effect when measured (not recorded: 0, the ramp)
	std::vector<std::pair<double, double>> pts; // (stick, rate), monotonic, dead zone included

	void Build() {
		pts.clear();
		pts.emplace_back(0.0, 0.0);
		size_t k = 0;
		while (k < rate.size() && rate[k] < 1.0) ++k;
		if (k == rate.size()) return;
		const double prev_s = k == 0 ? 0.0 : KBM_LEVELS[k - 1];
		double       dz     = prev_s;
		if (k + 1 < rate.size() && rate[k + 1] > rate[k]) {
			dz = KBM_LEVELS[k] - rate[k] * (KBM_LEVELS[k + 1] - KBM_LEVELS[k]) / (rate[k + 1] - rate[k]);
		}
		dz = std::clamp(dz, prev_s, KBM_LEVELS[k] - 0.01);
		pts.emplace_back(dz, 0.0);
		double last = 0.0;
		for (size_t i = k; i < rate.size(); ++i) {
			last = std::max(rate[i], last + 0.01);
			pts.emplace_back(KBM_LEVELS[i], last);
		}
		valid = true;
	}
	[[nodiscard]] double MaxRate() const { return pts.empty() ? 0.0 : pts.back().second; }
	[[nodiscard]] double Rate(double s) const {
		for (size_t i = 1; i < pts.size(); ++i)
			if (s <= pts[i].first) {
				const auto& p = pts[i - 1];
				const auto& q = pts[i];
				return p.second + (q.second - p.second) * (s - p.first) / std::max(q.first - p.first, 1e-9);
			}
		return MaxRate();
	}
	// The smallest stick that gives the rate (0 for none).
	[[nodiscard]] double Stick(double r) const {
		if (r <= 0.0 || pts.size() < 3) return 0.0;
		for (size_t i = 2; i < pts.size(); ++i)
			if (r <= pts[i].second) {
				const auto& p = pts[i - 1];
				const auto& q = pts[i];
				return p.first + (q.first - p.first) * (r - p.second) / std::max(q.second - p.second, 1e-9);
			}
		return 1.0;
	}
};

std::string KbmCurvePath() {
	std::string dir;
	if (char* base = SDL_GetBasePath(); base != nullptr) {
		dir = base;
		SDL_free(base);
	}
	return dir + "kbm_curve.txt";
}

bool KbmLoadCurve(KbmCurve* c) {
	FILE* f = std::fopen(KbmCurvePath().c_str(), "r");
	if (f == nullptr) return false;
	char line[256];
	int  got = 0;
	while (std::fgets(line, sizeof(line), f) != nullptr) {
		double s = 0, r = 0, v = 0;
		if (std::sscanf(line, "stick %lf rate %lf", &s, &r) == 2) {
			for (size_t i = 0; i < KBM_LEVELS.size(); ++i)
				if (std::abs(KBM_LEVELS[i] - s) < 0.01) c->rate[i] = std::abs(r), got |= 1 << i;
		} else if (std::sscanf(line, "onset_ms %lf", &v) == 1) {
			c->onset_ms = v;
		} else if (std::sscanf(line, "tau_ms %lf", &v) == 1) {
			c->tau_ms = std::clamp(v, 0.0, 500.0);
		} else if (int cs = 0; std::sscanf(line, "camspeed %d", &cs) == 1) {
			c->camspeed = cs;
		} else if (int nr = 0; std::sscanf(line, "noramp %d", &nr) == 1) {
			c->noramp = nr;
		} else if (std::sscanf(line, "sign %lf", &v) == 1) {
			c->sign = v < 0 ? -1 : 1;
		}
	}
	std::fclose(f);
	if (got != 0x1f) return false;
	c->Build();
	return c->valid;
}

void KbmSaveCurve(const KbmCurve& c) {
	const auto path = KbmCurvePath();
	FILE*      f    = std::fopen(path.c_str(), "w");
	if (f == nullptr) {
		LOGF("KYTY_KBM calibrate: cannot write %s\n", path.c_str());
		return;
	}
	std::fprintf(f, "# Kyty KBM: Demon's Souls stick -> camera yaw (KYTY_KBM_CALIBRATE=1 rewrites it; delete to go back to the linear mapping)\n");
	for (size_t i = 0; i < KBM_LEVELS.size(); ++i) std::fprintf(f, "stick %.2f rate %.2f\n", KBM_LEVELS[i], c.rate[i]);
	std::fprintf(f, "onset_ms %.1f\ntau_ms %.1f\nsign %.0f\n", c.onset_ms, c.tau_ms, c.sign);
	// (10-25) The camera speed it holds for (a curve measured at another one is not used).
	if (c.camspeed > 0) std::fprintf(f, "camspeed %d\n", c.camspeed);
	// (10-25) The camera's turn ramp removed or not (KYTY_KBM_NO_RAMP): a curve measured the other way is not used.
	std::fprintf(f, "noramp %d\n", c.noramp);
	std::fclose(f);
	LOGF("KYTY_KBM calibrate: curve written to %s\n", path.c_str());
}

// The calibration: once the camera has been seen steady for 3 s (standing still), each level for 0.8 s right then
// 0.8 s left, 0.5 s of centred stick before each; the yaw samples give the onset (first 0.25 degrees), the steady rate
// (slope after the ramp) and the ramp (the fitted line's lag behind the onset).
struct KbmCalib {
	enum Phase { Off, Wait, Rest, Drive, Done } phase = Off;
	int64_t phase_ns   = 0;
	int64_t steady_ns  = 0;
	int64_t remind_ns  = 0;
	int     step       = 0; // level * 2 + direction
	int64_t drive_ns   = 0;
	double  drive_yaw  = 0;
	int     drive_reads = 0;
	int     reads      = 0;
	std::array<double, 10> rate {};
	std::array<double, 10> onset {};
	std::array<double, 10> tau {};
	std::array<double, 10> sgn {};
	std::array<bool, 10>   ok {};
	double  read_ms_sum = 0;
	int     read_ms_n   = 0;
};
constexpr double CAL_REST_S  = 0.5;
constexpr double CAL_DRIVE_S = 0.8;

struct KbmState {
	bool                    enabled  = false;
	float                   sens     = 1.0f;
	float                   expo     = 1.0f;
	float                   adz      = 0.15f;
	float                   decay_ms = 25.0f;
	float                   walk     = 0.5f;
	// (10-25) The most a saturated stick still owes the mouse after this read (KYTY_KBM_CARRY_MS; 0, the default: one
	// game frame of full tilt, the rest dropped). Was 200 ms: a flick beyond the camera's top speed kept turning for
	// up to 200 ms after the hand stopped (the "holgura", worse with a higher sensitivity, which saturates sooner).
	float                   carry_ms = 0.0f;
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
	std::atomic<double>     warp_debt_x {0.0}; // KYTY_KBM_WARP: degrees owed after the last read (curve mapping)
	std::atomic<double>     warp_debt_y {0.0};
	// (10-25) One stick per game frame: a read with no guest flip since the previous one (the game reads the pad more
	// than once a frame, scePadRead and scePadReadState, a few us apart) repeats that read's stick and consumes nothing
	// (before, the first read took every count and the second saw none: a centred stick, so the camera and whatever the
	// game drives from the other read, the character's turn, got different sticks in the same frame).
	std::atomic<uint64_t>   flips {0};    // guest flips reserved (HostInputKbmOnFlip)
	std::atomic<uint64_t>   read_seq {0}; // game-frame reads made
	// (10-25) KYTY_KBM_WARP: what each game-frame read sent (linear: counts; curve: degrees), by read_seq; the warp
	// shows the reads after the presented frame's flip (sent, not shown yet) plus what is still pending.
	struct Sent {
		uint64_t seq = 0;
		double   x = 0.0, y = 0.0;
	};
	std::mutex             sent_mutex;
	std::array<Sent, 64>   sent {};
	// The game's camera degrees per unit sent (linear: deg per count; curve: a correction, 1 = the curve), learned from
	// the camera itself (0: not learned yet, no warp in the linear mapping).
	std::atomic<double>    gain_x {0.0};
	std::atomic<double>    gain_y {0.0};
	std::atomic<uint32_t>   wheel_control {0};
	// Guest pad thread only:
	int64_t last_ns = 0;
	int64_t log_ns  = 0;
	int     polls   = 0;
	double  out_x   = 0.0;
	double  out_y   = 0.0;
	uint64_t last_flips = 0;     // flips seen at the last game-frame read
	bool     last_ret   = false; // that read's answer (repeated by a read in the same frame)
	int      last_x = 128, last_y = 128;
	float    smooth_ms = 2.0f;   // (10-25) KYTY_KBM_SMOOTH_MS: the counts drained over this (sub-frame), 0 = all at once
	// (10-25) Linear mapping, linearized: the camera's degrees/s at full stick (rmax), learned from the camera; the
	// stick then asks for counts/s * deg per count (KYTY_KBM_DEG_PER_COUNT * sens), whatever the game camera speed.
	double   rmax         = 100.0;
	bool     rmax_learned = false;
	double   win_stick_x  = 0.0; // stick-seconds of x deflection (0..1, before the anti-deadzone) in the gain window
	// The gain's window (game-frame reads): degrees the camera turned, units sent.
	double   win_cam_x = 0.0, win_cam_y = 0.0, win_sent_x = 0.0, win_sent_y = 0.0;
	double   cam_prev_pitch = 0.0;
	// F7 without KYTY_KBM: the mouse alone on the right stick through this same per-read path.
	bool     mouse_only = false;
	bool     provider_set = false;
	// The camera-matched mapping (kbm_curve.txt): mouse counts -> degrees owed -> the stick the curve gives that rate.
	KbmCurve curve;
	double   dpc    = 0.05; // degrees per count at sensitivity 1 (KYTY_KBM_DEG_PER_COUNT)
	double   debt_x = 0.0, debt_y = 0.0; // degrees the camera still owes the mouse
	double   model  = 0.0;               // the rate the game's camera is giving now (its ramp modelled)
	KbmCalib cal;
	// The once-a-second line: degrees asked, degrees seen, latency.
	double  desired_deg  = 0.0;
	double  measured_deg = 0.0;
	int64_t cam_prev_t   = 0;
	double  cam_prev_yaw = 0.0;
	int64_t last_motion_ns = 0;
	bool    lat_pending  = false;
	int64_t lat_t0       = 0;
	int     lat_reads    = 0;
	double  lat_yaw0     = 0.0;
	int     lat_last_reads = -1;
	double  lat_last_ms  = 0.0;
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

// One calibration drive analysed (after its rest, so late camera samples are in).
void KbmCalibAnalyze(KbmCalib& c) {
	const int    i   = c.step;
	const auto   v   = CamSamples(c.drive_ns - 50000000, c.drive_ns + static_cast<int64_t>((CAL_DRIVE_S + CAL_REST_S) * 1e9));
	const double dir = (i % 2 == 0) ? 1.0 : -1.0;
	c.ok[i]          = false;
	if (v.size() < 10) return;
	double onset = -1.0;
	for (const auto& s: v)
		if (s.t >= c.drive_ns && std::abs(s.yaw - c.drive_yaw) > 0.25) {
			onset = (s.t - c.drive_ns) * 1e-9;
			break;
		}
	c.ok[i] = true;
	if (onset < 0.0) { // no motion: inside the game's dead zone
		c.rate[i] = 0.0, c.onset[i] = -1.0, c.tau[i] = 0.0;
		return;
	}
	double w0 = onset + 0.3, w1 = onset + CAL_DRIVE_S - 0.03;
	if (w1 - w0 < 0.15) w0 = onset + CAL_DRIVE_S * 0.5;
	std::vector<CamSample> win;
	for (const auto& s: v) {
		const double t = (s.t - c.drive_ns) * 1e-9;
		if (t >= w0 && t <= w1) win.push_back(s);
	}
	double a = 0, b = 0;
	if (!CamFit(win, c.drive_ns, &a, &b)) {
		c.ok[i] = false;
		return;
	}
	c.rate[i]  = std::abs(b);
	c.sgn[i]   = (b * dir) < 0 ? -1.0 : 1.0;
	c.onset[i] = onset;
	c.tau[i]   = std::abs(b) > 1.0 ? std::max(0.0, (c.drive_yaw - a) / b - onset) : 0.0;
	LOGF("KYTY_KBM calibrate: stick %+.1f: %.1f deg/s, camera moves %.0f ms after the stick, ramp %.0f ms (%zu samples)\n",
	     dir * KBM_LEVELS[i / 2], c.rate[i], onset * 1000.0, c.tau[i] * 2300.0, win.size());
}

void KbmCalibFinish(KbmState& k) {
	auto&    c = k.cal;
	KbmCurve curve;
	double   on = 0, ta = 0, sg = 0;
	int      n_on = 0, n_ta = 0;
	for (size_t l = 0; l < KBM_LEVELS.size(); ++l) {
		double sum = 0;
		int    n   = 0;
		for (int d = 0; d < 2; ++d) {
			const int i = static_cast<int>(l) * 2 + d;
			if (!c.ok[i]) continue;
			sum += c.rate[i], ++n;
			if (c.onset[i] >= 0 && l >= 1) on += c.onset[i], ++n_on;
			if (c.rate[i] > 1.0 && l >= 1) ta += c.tau[i], ++n_ta, sg += c.sgn[i];
		}
		if (n == 0) {
			LOGF("KYTY_KBM calibrate: failed (no camera samples at stick %.1f); kbm_curve.txt unchanged. Run it again standing still in an open area\n",
			     KBM_LEVELS[l]);
			c.phase = KbmCalib::Done;
			return;
		}
		curve.rate[l] = sum / n;
	}
	curve.onset_ms = n_on > 0 ? on / n_on * 1000.0 : 0.0;
	curve.tau_ms   = n_ta > 0 ? ta / n_ta * 1000.0 : 0.0;
	curve.sign     = sg < 0 ? -1.0 : 1.0;
	curve.camspeed = Loader::GameArgs::KbmCameraSpeed(); // (10-25) the speed it holds for
	curve.noramp   = Loader::DemonsSoulsCamera::NoRampMode();
	curve.Build();
	if (!curve.valid) {
		LOGF("KYTY_KBM calibrate: failed (the camera did not turn at any stick value); kbm_curve.txt unchanged\n");
		c.phase = KbmCalib::Done;
		return;
	}
	const double read_ms = c.read_ms_n > 0 ? c.read_ms_sum / c.read_ms_n : 16.7;
	LOGF("KYTY_KBM calibrate: done: 0.2 %.1f, 0.4 %.1f, 0.6 %.1f, 0.8 %.1f, 1.0 %.1f deg/s; onset %.0f ms (%.1f pad reads of %.1f ms), "
	     "ramp tau %.0f ms\n",
	     curve.rate[0], curve.rate[1], curve.rate[2], curve.rate[3], curve.rate[4], curve.onset_ms, curve.onset_ms / read_ms, read_ms,
	     curve.tau_ms);
	KbmSaveCurve(curve);
	k.curve = curve;
	k.debt_x = k.debt_y = k.model = 0.0;
	c.phase = KbmCalib::Done;
}

// The calibration's stick for this read (true while it drives the stick).
bool KbmCalibStep(KbmState& k, int64_t now, double dt) {
	auto& c = k.cal;
	if (c.phase == KbmCalib::Off || c.phase == KbmCalib::Done) return false;
	k.acc_x.store(0), k.acc_y.store(0); // the mouse is ignored meanwhile
	k.out_x = k.out_y = 0.0;
	CamSample last;
	const bool cam = CamLatest(&last) && now - last.t < 250000000;
	if (c.phase == KbmCalib::Wait) {
		bool still = false;
		if (cam) {
			double a = 0, b = 0;
			still = CamFit(CamSamples(now - 1000000000, now), now, &a, &b) && std::abs(b) < 3.0;
		}
		if (!still) c.steady_ns = now;
		if (now - c.remind_ns >= 10000000000LL || c.remind_ns == 0) {
			c.remind_ns = now;
			LOGF("KYTY_KBM calibrate: waiting: stand still in an open area, hands off the mouse and the right stick (camera %s)\n",
			     cam ? (still ? "seen, steady" : "seen, moving") : "not seen yet: in game, with an upscaler (DLSS/XeSS/FSR) or frame generation on");
		}
		if (now - c.steady_ns >= 3000000000LL) {
			LOGF("KYTY_KBM calibrate: start: the camera turns by itself for about 13 s, do not touch anything\n");
			c.phase = KbmCalib::Rest, c.phase_ns = now, c.step = 0;
		}
		return true;
	}
	c.read_ms_sum += dt * 1000.0, ++c.read_ms_n;
	const double t = (now - c.phase_ns) * 1e-9;
	if (c.phase == KbmCalib::Rest) {
		if (t < CAL_REST_S) return true;
		if (c.drive_ns != 0) { // the previous drive, its tail seen
			KbmCalibAnalyze(c);
			if (++c.step >= 10) {
				KbmCalibFinish(k);
				return false;
			}
		}
		c.phase = KbmCalib::Drive, c.phase_ns = now, c.drive_ns = now;
		c.drive_yaw = cam ? last.yaw : 0.0;
	}
	if (c.phase == KbmCalib::Drive) {
		if (t >= CAL_DRIVE_S && now != c.phase_ns) {
			c.phase = KbmCalib::Rest, c.phase_ns = now;
			return true;
		}
		k.out_x = ((c.step % 2 == 0) ? 1.0 : -1.0) * KBM_LEVELS[c.step / 2];
	}
	return true;
}

// (10-25) What a game-frame read sent, for the warp (linear: counts; curve: degrees).
void KbmNoteSent(KbmState& k, double x, double y) {
	const uint64_t  seq = k.read_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
	std::lock_guard lock(k.sent_mutex);
	k.sent[seq % k.sent.size()] = KbmState::Sent {seq, x, y};
}

// (10-25) The camera's degrees per unit sent, from a window of game-frame reads (abs sums: the latency between the
// stick and the camera only blurs the window's edges). Once a second; a window with too little turning is skipped.
void KbmLearnGain(KbmState& k) {
	const bool   curve = k.curve.valid;
	const double min_sent = curve ? 4.0 : 80.0; // degrees / counts
	auto learn = [&](std::atomic<double>& gain, double cam, double sent, const char* axis) {
		if (sent < min_sent || cam < 0.5) return;
		const double lo = curve ? 0.2 : 0.002, hi = curve ? 5.0 : 2.0;
		const double r  = std::clamp(cam / sent, lo, hi);
		const double g  = gain.load(std::memory_order_relaxed);
		const double n  = g <= 0.0 ? r : g + (std::clamp(r, g * 0.5, g * 2.0) - g) * 0.3;
		gain.store(n, std::memory_order_relaxed);
		static int said = 0;
		if (g <= 0.0 || said < 4) {
			++said;
			LOGF("KYTY_KBM warp: camera %s gain %.4f %s (window %.0f %s sent, %.1f deg turned)\n", axis, n,
			     curve ? "x the curve" : "deg per count", sent, curve ? "deg" : "counts", cam);
		}
	};
	// (10-25) The linear mapping's top rate: the camera's turn over the stick-seconds asked of it.
	if (!curve && k.win_stick_x >= 0.15 && k.win_cam_x >= 2.0) {
		const double r = std::clamp(k.win_cam_x / k.win_stick_x, 20.0, 1500.0);
		const double n = k.rmax_learned ? k.rmax + (std::clamp(r, k.rmax * 0.5, k.rmax * 2.0) - k.rmax) * 0.3
		                                : 0.5 * (k.rmax + r);
		static int said_rmax = 0;
		if (!k.rmax_learned || said_rmax < 6) {
			++said_rmax;
			LOGF("KYTY_KBM: camera top rate %.0f deg/s at full stick (measured %.0f over %.2f stick-s): %.3f deg per count\n", n, r,
			     k.win_stick_x, k.dpc * k.sens);
		}
		k.rmax = n, k.rmax_learned = true;
	}
	k.win_stick_x = 0.0;
	learn(k.gain_x, k.win_cam_x, k.win_sent_x, "yaw");
	learn(k.gain_y, k.win_cam_y, k.win_sent_y, "pitch");
	k.win_cam_x = k.win_cam_y = k.win_sent_x = k.win_sent_y = 0.0;
}

// The right stick at a guest pad read (Controller's provider). (10-25) One value per game frame: a read with no guest
// flip since the last one (and within 50 ms) repeats it. The counts are drained over KYTY_KBM_SMOOTH_MS (a sub-frame
// exponential: the stick follows counts/s, not how many counts happened to arrive before this read).
bool KbmProvide(int* x, int* y) {
	auto&         k   = g_kbm;
	const int64_t now = KbmNow();

	if (const auto rel = k.wheel_release_ns.load(std::memory_order_acquire); rel != 0 && now >= rel) {
		k.wheel_release_ns.store(0, std::memory_order_relaxed);
		SetButton(k.wheel_control.load(std::memory_order_relaxed), false);
	}

	const uint64_t flips = k.flips.load(std::memory_order_acquire);
	if (k.last_ns != 0 && flips == k.last_flips && now - k.last_ns < 50000000) {
		*x = k.last_x, *y = k.last_y;
		return k.last_ret;
	}
	k.last_flips = flips;

	double dt = k.last_ns == 0 ? 1.0 / 60.0 : static_cast<double>(now - k.last_ns) * 1e-9;
	dt        = std::clamp(dt, 0.004, 0.1);
	k.last_ns = now;
	k.polls++;

	// The camera's yaw path since the last read (the log's measured rate, the latency probe, the warp's gain).
	CamSample cam;
	const bool have_cam = CamLatest(&cam);
	if (have_cam && cam.t != k.cam_prev_t) {
		if (k.cam_prev_t != 0 && cam.t - k.cam_prev_t < 200000000) {
			k.measured_deg += std::abs(cam.yaw - k.cam_prev_yaw);
			k.win_cam_x += std::abs(cam.yaw - k.cam_prev_yaw);
			k.win_cam_y += std::abs(cam.pitch - k.cam_prev_pitch);
		}
		k.cam_prev_t = cam.t, k.cam_prev_yaw = cam.yaw, k.cam_prev_pitch = cam.pitch;
	}

	double     sent_x = 0.0, sent_y = 0.0;
	const bool calibrating = KbmCalibStep(k, now, dt);
	if (!calibrating) {
		int64_t       ax = k.acc_x.load(std::memory_order_acquire);
		int64_t       ay = k.acc_y.load(std::memory_order_acquire);
		double        cx = static_cast<double>(ax) / KBM_ONE;
		double        cy = static_cast<double>(ay) / KBM_ONE;
		if (ax != 0 || ay != 0) {
			if (now - k.last_motion_ns > 150000000 && std::abs(cx) >= std::abs(cy) && have_cam) {
				k.lat_pending = true, k.lat_t0 = now, k.lat_reads = 0, k.lat_yaw0 = cam.yaw;
			}
			k.last_motion_ns = now;
		}
		if (k.lat_pending) {
			if (have_cam && cam.t > k.lat_t0 && std::abs(cam.yaw - k.lat_yaw0) > 0.2) {
				k.lat_last_reads = k.lat_reads, k.lat_last_ms = (cam.t - k.lat_t0) * 1e-6, k.lat_pending = false;
			} else if (now - k.lat_t0 > 1000000000) {
				k.lat_pending = false;
			}
			++k.lat_reads;
		}
		const double dpc   = k.dpc * k.sens;
		const double drain = k.smooth_ms > 0.0f ? 1.0 - std::exp(-dt * 1000.0 / k.smooth_ms) : 1.0;
		if (k.curve.valid) {
			// Counts drained (KYTY_KBM_SMOOTH_MS) into degrees owed, paid at the curve's rate; the stick the inverse curve
			// gives for the rate that pays them this read, raised while the game's ramp lags (feed-forward), clamped at
			// full with the rest carried.
			const int64_t ux_c = drain >= 1.0 ? ax : static_cast<int64_t>(std::llround(static_cast<double>(ax) * drain));
			const int64_t uy_c = drain >= 1.0 ? ay : static_cast<int64_t>(std::llround(static_cast<double>(ay) * drain));
			k.acc_x.fetch_sub(ux_c, std::memory_order_acq_rel);
			k.acc_y.fetch_sub(uy_c, std::memory_order_acq_rel);
			cx = static_cast<double>(ux_c) / KBM_ONE, cy = static_cast<double>(uy_c) / KBM_ONE;
			k.applied.fetch_add(std::llabs(ux_c) + std::llabs(uy_c), std::memory_order_relaxed);
			k.debt_x += cx * dpc, k.debt_y += cy * dpc;
			k.desired_deg += std::hypot(cx, cy) * dpc;
			double L = std::hypot(k.debt_x, k.debt_y);
			// (10-24) Owed degrees capped at carry_ms of the camera's top rate (was 180 deg): a flick beyond the game's
			// speed turns at full tilt for at most carry_ms after the hand stops, not seconds.
			// (10-25) This read's and one more frame's worth by default (KYTY_KBM_CARRY_MS 0), the rest dropped.
			const double cap = std::max(k.curve.MaxRate() * (k.carry_ms > 0.0f ? k.carry_ms * 1e-3 + dt : 2.0 * dt), 0.5);
			if (L > cap) k.debt_x *= cap / L, k.debt_y *= cap / L, L = cap;
			const double tau = k.curve.tau_ms * 1e-3;
			const double a   = tau > 1e-4 ? 1.0 - std::exp(-dt / tau) : 1.0;
			if (L < 0.01) {
				k.debt_x = k.debt_y = 0.0;
				k.model *= 1.0 - a;
				k.out_x = k.out_y = 0.0;
			} else {
				const double want = L / dt;
				const double cmd  = std::clamp(k.model + (want - k.model) / a, 0.0, k.curve.MaxRate());
				k.model += (cmd - k.model) * a;
				const double paid = std::min(k.model * dt, L);
				const double ux = k.debt_x / L, uy = k.debt_y / L;
				const double st   = k.curve.Stick(cmd);
				k.debt_x -= ux * paid, k.debt_y -= uy * paid;
				k.out_x = ux * st, k.out_y = uy * st;
				sent_x = ux * paid, sent_y = uy * paid;
			}
		} else {
			// No kbm_curve.txt: the linear mapping (counts/s -> deflection). (10-25) The counts drained this read (a
			// fraction, KYTY_KBM_SMOOTH_MS) over dt: the rate; a saturated stick's excess stays, capped at carry_ms of
			// full tilt (was: the first read of a frame took every count, the counts/s noisy with the read's timing).
			// (10-25) Linearized: full stick at rmax / (deg per count) counts/s, rmax learned from the camera.
			const double cpf = std::max(k.rmax / std::max(dpc, 1e-6), 1.0); // counts/s at full stick
			double ux = cx * drain, uy = cy * drain;
			double mag = std::hypot(ux, uy) / dt / cpf;
			if (mag > 1.0) ux /= mag, uy /= mag, mag = 1.0;
			int64_t used_x = static_cast<int64_t>(std::llround(ux * KBM_ONE));
			int64_t used_y = static_cast<int64_t>(std::llround(uy * KBM_ONE));
			if (mag < 0.005) used_x = used_y = 0; // under a count or so a second: nothing sent
			{
				// The rest, capped: one game frame of full tilt (KYTY_KBM_CARRY_MS 0) or that many ms; beyond, dropped.
				const double  lim = (k.carry_ms > 0.0f ? k.carry_ms * 1e-3 : dt) * cpf * KBM_ONE;
				const double  rx = static_cast<double>(ax - used_x), ry = static_cast<double>(ay - used_y);
				const double  rl = std::hypot(rx, ry);
				int64_t       drop_x = 0, drop_y = 0;
				if (rl > lim) {
					drop_x = static_cast<int64_t>(std::llround(rx * (1.0 - lim / rl)));
					drop_y = static_cast<int64_t>(std::llround(ry * (1.0 - lim / rl)));
				} else if (used_x == 0 && used_y == 0 && rl < 2.0 * KBM_ONE) {
					drop_x = ax, drop_y = ay; // a hand at rest: the last fraction of a count or two dropped
				}
				k.acc_x.fetch_sub(used_x + drop_x, std::memory_order_acq_rel);
				k.acc_y.fetch_sub(used_y + drop_y, std::memory_order_acq_rel);
			}
			if (used_x != 0 || used_y != 0) {
				k.applied.fetch_add(std::llabs(used_x) + std::llabs(used_y), std::memory_order_relaxed);
				const double fx = static_cast<double>(used_x) / KBM_ONE, fy = static_cast<double>(used_y) / KBM_ONE;
				k.desired_deg += std::hypot(fx, fy) * dpc;
				const double out = k.adz + (1.0 - k.adz) * std::pow(mag, static_cast<double>(k.expo));
				const double len = std::hypot(fx, fy);
				k.out_x          = fx / len * out;
				k.out_y          = fy / len * out;
				sent_x = fx, sent_y = fy;
				k.win_stick_x += mag * std::abs(fx) / len * dt;
			} else if (k.decay_ms <= 0.0f) {
				k.out_x = k.out_y = 0.0;
			} else {
				const double d = std::exp(-dt * 1000.0 / k.decay_ms);
				k.out_x *= d;
				k.out_y *= d;
				if (std::hypot(k.out_x, k.out_y) < k.adz * 0.5 + 0.01) k.out_x = k.out_y = 0.0;
			}
		}
	}
	k.win_sent_x += std::abs(sent_x), k.win_sent_y += std::abs(sent_y);
	KbmNoteSent(k, sent_x, sent_y);

	k.warp_debt_x.store(k.curve.valid ? k.debt_x : 0.0, std::memory_order_relaxed);
	k.warp_debt_y.store(k.curve.valid ? k.debt_y : 0.0, std::memory_order_relaxed);

	if (k.log_ns == 0) k.log_ns = now;
	if (now - k.log_ns >= 1000000000) {
		const double secs = (now - k.log_ns) * 1e-9;
		const auto   rec  = k.received.exchange(0, std::memory_order_relaxed);
		const auto   app  = k.applied.exchange(0, std::memory_order_relaxed);
		const auto   pend = std::llabs(k.acc_x.load(std::memory_order_relaxed)) + std::llabs(k.acc_y.load(std::memory_order_relaxed));
		if (!calibrating) KbmLearnGain(k);
		if (rec != 0 || app != 0) {
			char lat[64];
			if (k.lat_last_reads >= 0)
				std::snprintf(lat, sizeof(lat), "%d pad reads (%.0f ms)", k.lat_last_reads, k.lat_last_ms);
			else
				std::snprintf(lat, sizeof(lat), "not measured yet");
			LOGF("KYTY_KBM mouse: received %.0f applied %.2f pending %.2f counts in %.2f s, %d game-frame reads; %s: desired %.1f deg/s, "
			     "camera measured %.1f deg/s, owed %.1f deg; latency mouse -> camera %s; warp gain %.4f / %.4f\n",
			     static_cast<double>(rec) / KBM_ONE, static_cast<double>(app) / KBM_ONE, static_cast<double>(pend) / KBM_ONE, secs,
			     k.polls, k.curve.valid ? "curve" : "linear", k.desired_deg / secs, k.measured_deg / secs,
			     std::hypot(k.debt_x, k.debt_y), lat, k.gain_x.load(std::memory_order_relaxed),
			     k.gain_y.load(std::memory_order_relaxed));
		}
		k.desired_deg = k.measured_deg = 0.0;
		k.log_ns = now;
		k.polls  = 0;
	}

	bool ret = true;
	if (calibrating) {
		*x = std::clamp(static_cast<int>(std::lround(128.0 + k.out_x * 127.5)), 0, 255);
		*y = 128;
	} else if (k.out_x == 0.0 && k.out_y == 0.0) {
		*x = *y = 128;
		ret = false; // centred: the host pad's stick stays usable
	} else {
		*x = std::clamp(static_cast<int>(std::lround(128.0 + k.out_x * 127.5)), 0, 255);
		*y = std::clamp(static_cast<int>(std::lround(128.0 + k.out_y * 127.5)), 0, 255);
	}
	k.last_x = *x, k.last_y = *y, k.last_ret = ret;
	return ret;
}

void KbmSetupMouse() {
	auto& k = g_kbm;
	if (k.provider_set) return;
	k.provider_set = true;
	k.dpc          = EnvFloat("KYTY_KBM_DEG_PER_COUNT", 0.05f, 0.001f, 2.0f);
	// (10-25) The camera's top rate until measured: ~18 deg/s per camera speed step (5: ~90 deg/s).
	const int camspeed = Loader::GameArgs::KbmCameraSpeed();
	k.rmax             = camspeed > 0 ? 18.0 * camspeed : 100.0;
	const bool calibrate = [] {
		const char* cal = std::getenv("KYTY_KBM_CALIBRATE");
		return cal != nullptr && std::string_view(cal) == "1";
	}();
	const int noramp = Loader::DemonsSoulsCamera::NoRampMode();
	if (!calibrate && KbmLoadCurve(&k.curve) && k.curve.camspeed > 0 && camspeed > 0 && k.curve.camspeed != camspeed) {
		LOGF("KYTY_KBM: %s was measured at camera speed %d, the game runs at %d: not used (linear mapping); calibrate "
		     "again at %d\n",
		     KbmCurvePath().c_str(), k.curve.camspeed, camspeed, camspeed);
		k.curve = KbmCurve {};
	} else if (k.curve.valid && k.curve.noramp != noramp) {
		// (10-25) Measured with the game's turn ramp and now without it (or the other way): other rates, another ramp.
		LOGF("KYTY_KBM: %s was measured with KYTY_KBM_NO_RAMP=%d, now %d: not used (linear mapping); calibrate again\n",
		     KbmCurvePath().c_str(), k.curve.noramp, noramp);
		k.curve = KbmCurve {};
	} else if (k.curve.valid) {
		if (k.curve.camspeed == 0)
			LOGF("KYTY_KBM: %s does not say its camera speed (older calibration): used as is; calibrate again if the camera "
			     "speed changed\n",
			     KbmCurvePath().c_str());
	}
	if (k.curve.valid) {
		LOGF("KYTY_KBM: camera curve %s: 0.2 %.1f, 0.4 %.1f, 0.6 %.1f, 0.8 %.1f, 1.0 %.1f deg/s, ramp tau %.0f ms; %.3f deg per count\n",
		     KbmCurvePath().c_str(), k.curve.rate[0], k.curve.rate[1], k.curve.rate[2], k.curve.rate[3], k.curve.rate[4],
		     k.curve.tau_ms, k.dpc * k.sens);
	} else {
		LOGF("KYTY_KBM: no camera curve (%s): linear mapping, %.3f deg per count, camera speed %d (top rate %.0f deg/s until "
		     "measured); KYTY_KBM_CALIBRATE=1 measures the curve\n",
		     KbmCurvePath().c_str(), k.dpc * k.sens, camspeed, k.rmax);
	}
	// (10-25) The game's own full-tilt turn ramp (the camera speeding up while the stick is at full, the "delay"):
	// removed at load by Loader::DemonsSoulsCamera unless KYTY_KBM_NO_RAMP=0 (its own log line says if it was).
	LOGF("KYTY_KBM: camera turn ramp %s (KYTY_KBM_NO_RAMP=%d)\n",
	     noramp == 0 ? "the game's (kept)" : noramp == 1 ? "removed, yaw linear at the game's top rate" : "removed, yaw linear at the game's base rate",
	     noramp);
	if (const char* cal = std::getenv("KYTY_KBM_CALIBRATE"); cal != nullptr && std::string_view(cal) == "1") {
		k.cal.phase = KbmCalib::Wait;
		LOGF("KYTY_KBM calibrate: on: load a save, stand still in an open area and let go of the mouse and the right stick\n");
	}
	Controller::SetRightStickProvider(&KbmProvide);
}

void KbmInit() {
	const char* on  = std::getenv("KYTY_KBM");
	const char* cal = std::getenv("KYTY_KBM_CALIBRATE");
	if ((on == nullptr || std::string_view(on) != "1") && (cal == nullptr || std::string_view(cal) != "1")) {
		LOGF("KYTY_KBM: off (F7: the mouse alone on the right stick)\n");
		return;
	}
	g_kbm.enabled  = true;
	g_kbm.sens     = EnvFloat("KYTY_KBM_SENS", 1.0f, 0.05f, 20.0f);
	g_kbm.expo     = EnvFloat("KYTY_KBM_EXP", 1.0f, 0.3f, 3.0f);
	g_kbm.adz      = EnvFloat("KYTY_KBM_ANTIDZ", 0.15f, 0.0f, 0.6f);
	g_kbm.decay_ms = EnvFloat("KYTY_KBM_DECAY_MS", 0.0f, 0.0f, 500.0f);
	g_kbm.walk     = EnvFloat("KYTY_KBM_WALK", 0.5f, 0.1f, 1.0f);
	g_kbm.carry_ms = EnvFloat("KYTY_KBM_CARRY_MS", 0.0f, 0.0f, 2000.0f);
	g_kbm.smooth_ms = EnvFloat("KYTY_KBM_SMOOTH_MS", 2.0f, 0.0f, 100.0f);
	KbmAddList(KBM_DEFAULTS, false);
	if (const char* map = std::getenv("KYTY_KBM_MAP"); map != nullptr) KbmAddList(map, true);
	LOGF("KYTY_KBM: keyboard and mouse on (sens %.2f, exp %.2f, antideadzone %.2f, decay %.0f ms, smoothing %.0f ms, carry %.0f ms, "
	     "one stick per game frame, %zu bindings)\n",
	     g_kbm.sens, g_kbm.expo, g_kbm.adz, g_kbm.decay_ms, g_kbm.smooth_ms, g_kbm.carry_ms, g_kbm.bindings.size());
	KbmSetupMouse();
}

void KbmCapture(bool on) {
	if ((!g_kbm.enabled && !g_kbm.mouse_only) || g_kbm.captured == on) return;
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

void HostInputCameraForward(float fx, float fy, float fz) {
	if (!g_kbm.provider_set) return;
	if (std::hypot(fx, fz) < 0.2f) return; // looking straight up or down: no yaw
	const double raw = std::atan2(static_cast<double>(fx), static_cast<double>(fz)) * 57.29577951308232;
	const int64_t now = KbmNow();
	std::lock_guard lock(g_cam_mutex);
	if (g_cam_count != 0) {
		const double d = raw - g_cam_raw;
		if (d > 180.0) g_cam_turns -= 360.0;
		if (d < -180.0) g_cam_turns += 360.0;
	}
	g_cam_raw = raw;
	const double pitch = std::atan2(static_cast<double>(fy), std::hypot(static_cast<double>(fx), static_cast<double>(fz))) *
	                     57.29577951308232;
	g_cam_ring[g_cam_count % g_cam_ring.size()] = CamSample {now, raw + g_cam_turns, pitch};
	++g_cam_count;
}

void HostInputInit() {
	KbmInit();
	if (!g_kbm.enabled) GetInputMap();
}

uint64_t HostInputKbmOnFlip() {
	auto& k = g_kbm;
	// (The read sequence first: the flip shows the reads made before it.)
	const uint64_t seq = k.read_seq.load(std::memory_order_acquire);
	k.flips.fetch_add(1, std::memory_order_acq_rel);
	return seq;
}

bool HostInputKbmWarpDegrees(uint64_t shown_seq, double* yaw, double* pitch) {
	auto& k = g_kbm;
	if (!k.enabled || !k.captured) return false;
	static const uint64_t behind = [] {
		const char* v = std::getenv("KYTY_KBM_WARP_BEHIND");
		return v != nullptr ? static_cast<uint64_t>(std::clamp(std::atoi(v), 0, 8)) : uint64_t {0};
	}();
	const bool   curve = k.curve.valid;
	const double gx    = k.gain_x.load(std::memory_order_relaxed);
	double       gy    = k.gain_y.load(std::memory_order_relaxed);
	if (gy <= 0.0) gy = gx;
	// Linear mapping: no warp until the camera's degrees per count are known (the counts alone say nothing of the
	// game's turn: camera speed, dead zone). Curve: the curve's degrees, corrected once learned.
	const double kx = curve ? (gx > 0.0 ? gx : 1.0) : gx;
	const double ky = curve ? (gy > 0.0 ? gy : 1.0) : gy;
	if (kx <= 0.0) {
		*yaw = *pitch = 0.0;
		return true;
	}
	// Pending: the counts not read yet (the smoothing's reservoir too) and, with the curve, the degrees still owed.
	const double unit = curve ? k.dpc * k.sens : 1.0;
	double       px   = static_cast<double>(k.acc_x.load(std::memory_order_relaxed)) / KBM_ONE * unit;
	double       py   = static_cast<double>(k.acc_y.load(std::memory_order_relaxed)) / KBM_ONE * unit;
	if (curve) px += k.warp_debt_x.load(std::memory_order_relaxed), py += k.warp_debt_y.load(std::memory_order_relaxed);
	// Sent after the presented frame's flip: the game has them, its frame does not show them yet. They leave this sum
	// exactly when a frame that has them is presented (its flip's sequence), so the shift hands over to the real turn.
	{
		const uint64_t  from = shown_seq > behind ? shown_seq - behind : 0;
		std::lock_guard lock(k.sent_mutex);
		for (const auto& s: k.sent)
			if (s.seq > from) px += s.x, py += s.y;
	}
	*yaw   = px * kx;
	*pitch = py * ky;
	return std::isfinite(*yaw) && std::isfinite(*pitch);
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
	if (!g_kbm.enabled && !g_kbm.mouse_only) return false;
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
	// The mouse alone (no KYTY_KBM): the same per-pad-read path as KYTY_KBM (no 33 ms timer).
	if (std::getenv("KYTY_MOUSE_TIMER") == nullptr) {
		KbmSetupMouse();
		if (g_kbm.mouse_only) {
			KbmCapture(false);
			g_kbm.mouse_only = false;
			LOGF("Mouse to right stick: disabled\n");
		} else {
			g_kbm.mouse_only = true;
			KbmCapture(true);
			if (!g_kbm.captured) g_kbm.mouse_only = false;
			LOGF("Mouse to right stick: %s (F7 to release; read at each guest pad read)\n", g_kbm.mouse_only ? "enabled" : "failed");
		}
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
