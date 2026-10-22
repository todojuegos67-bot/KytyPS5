#include "libs/controller.h"

#include "SDL.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/padData.h"
#include "graphics/presentation/videoOut.h"
#include "loader/demonsSoulsWarp.h"
#include "local/anti-lag.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace Libs::Controller {

LIB_NAME("Pad", "Pad");

constexpr int PAD_ERROR_INVALID_ARG    = -2137915391; /* 0x80920001 */
constexpr int PAD_ERROR_INVALID_HANDLE = -2137915389; /* 0x80920003 */

constexpr uint32_t RUMBLE_DURATION_MS = 0xffff;
constexpr uint32_t RELEASE_FLUSH_MS   = 50;

struct PadControllerInformation {
	float    touch_pixel_density;
	uint16_t touch_resolution_x;
	uint16_t touch_resolution_y;
	uint8_t  stick_dead_zone_left;
	uint8_t  stick_dead_zone_right;
	uint8_t  connection_type;
	uint8_t  connected_count;
	bool     connected;
	int      device_class;
	uint8_t  reserve[8];
};

struct PadLightBarParam {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t reserve;
};

struct PadVibrationParam {
	uint8_t large_motor;
	uint8_t small_motor;
};

struct PadTriggerEffectCommand {
	uint32_t mode;
	uint8_t  reserve[4];
	uint8_t  data[48];
};

struct PadTriggerEffectParam {
	uint8_t                 trigger_mask;
	uint8_t                 reserve[7];
	PadTriggerEffectCommand command[2];
};

static_assert(sizeof(PadTriggerEffectCommand) == 56);
static_assert(sizeof(PadTriggerEffectParam) == 120);

struct DualSenseEffects {
	uint8_t enable_bits;
	uint8_t reserve[9];
	uint8_t right_trigger[11];
	uint8_t left_trigger[11];
};

static_assert(sizeof(DualSenseEffects) == 32);

struct ControllerState {
	struct Touch {
		uint8_t  id   = 0;
		bool     down = false;
		uint16_t x    = 0;
		uint16_t y    = 0;
	};

	uint64_t time                                  = 0;
	uint32_t buttons                               = 0;
	int      axes[static_cast<int>(Axis::AxisMax)] = {128, 128, 128, 128, 0, 0};
	Touch    touch[2];
};

class GameController {
public:
	GameController() = default;

	KYTY_CLASS_NO_COPY(GameController);

	void Connect(int id);
	void Disconnect(int id);
	void Button(int id, uint32_t button, bool down);
	void Axis(int id, Axis axis, int value);
	void RightStick(int id, int x, int y);
	void TouchPad(int id, int finger, bool down, float x, float y);
	void ResetInputState();
	void ReleaseHostPads();
	void GetConnectionInfo(bool* flag, int* count);
	void SetVibration(uint8_t large_motor, uint8_t small_motor);
	void SetLightBar(uint8_t r, uint8_t g, uint8_t b);
	bool SetTriggerEffect(const PadTriggerEffectParam& param);
	int32_t GetTriggerEffectState(bool left);
	void ReadState(ControllerState* state, bool* flag, int* count);
	int  ReadStates(ControllerState* states, int states_num, bool* flag, int* count);

private:
	static constexpr uint32_t STATES_MAX = 64;

	void CheckActive();
	void AddState();

	Common::Mutex    m_mutex;
	std::vector<int> m_connected_ids;
	int              m_active_id       = -1;
	bool             m_connected       = false;
	int              m_connected_count = 0;
	ControllerState  m_state;
	ControllerState  m_states[STATES_MAX];
	bool             m_obtained[STATES_MAX] {};
	uint32_t         m_states_num    = 0;
	uint32_t         m_first_state   = 0;
	uint8_t          m_next_touch_id = 1;
	// Every effect the game set, whether or not the host pad could reproduce it:
	// scePadGetTriggerEffectState answers from it (m_trigger_sent keeps the sent one only).
	PadTriggerEffectParam m_state_effect {};
};

static GameController* g_controller = nullptr;

// SetScriptedInput's values; `until` (steady clock ns, 0: none) is written last.
struct ScriptedInput {
	std::atomic<int64_t>                until {0};
	std::array<std::atomic<int32_t>, 4> axes {};
	std::atomic<uint32_t>               buttons {0};
};
static ScriptedInput g_scripted;

static int64_t SteadyNanoseconds() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void pad_fill_data(PadData* data, const ControllerState& state, bool connected,
                          int connected_count) {
	EXIT_IF(data == nullptr);

	std::memset(data, 0, sizeof(*data));

	data->buttons           = state.buttons;
	data->left_stick_x      = state.axes[static_cast<int>(Axis::LeftX)];
	data->left_stick_y      = state.axes[static_cast<int>(Axis::LeftY)];
	data->right_stick_x     = state.axes[static_cast<int>(Axis::RightX)];
	data->right_stick_y     = state.axes[static_cast<int>(Axis::RightY)];
	data->analog_buttons_l2 = state.axes[static_cast<int>(Axis::TriggerLeft)];
	data->analog_buttons_r2 = state.axes[static_cast<int>(Axis::TriggerRight)];
	data->orientation_w     = 1.0f;
	for (const auto& touch: state.touch) {
		if (touch.down) {
			auto& output = data->touch_data.touch[data->touch_data.touch_num++];
			output.x     = touch.x;
			output.y     = touch.y;
			output.id    = touch.id;
		}
	}
	if (const auto until = g_scripted.until.load(std::memory_order_acquire); until != 0 && SteadyNanoseconds() < until) {
		uint8_t* axes[] = {&data->left_stick_x, &data->left_stick_y, &data->right_stick_x, &data->right_stick_y};
		for (size_t i = 0; i < 4; i++) {
			if (const auto value = g_scripted.axes[i].load(std::memory_order_relaxed); value >= 0) {
				*axes[i] = static_cast<uint8_t>(std::min(value, 255));
			}
		}
		data->buttons |= g_scripted.buttons.load(std::memory_order_relaxed);
	}
	// The player's first button (the world gate in videoOut.cpp starts from it).
	if (connected && data->buttons != 0) Libs::VideoOut::NotePlayerInput();
	// (10-22) Only a stick ends a cutscene's hold (a button, X skipping included, keeps it until the player moves).
	if (connected && (std::abs(data->left_stick_x - 128) > 48 || std::abs(data->left_stick_y - 128) > 48 ||
	                  std::abs(data->right_stick_x - 128) > 48 || std::abs(data->right_stick_y - 128) > 48))
		Libs::VideoOut::NotePlayerActivity();
	data->connected              = connected;
	data->timestamp              = state.time;
	data->connected_count        = static_cast<uint8_t>(std::min(connected_count, 255));
	data->device_unique_data_len = 0;
}

static bool trigger_effect_zones(uint8_t* effect, const uint8_t* strengths, uint8_t type,
                                 uint8_t frequency = 0) {
	uint16_t active = 0;
	uint32_t packed = 0;
	for (int i = 0; i < 10; i++) {
		if (strengths[i] > 8) {
			return false;
		}
		if (strengths[i] != 0) {
			active |= static_cast<uint16_t>(1u << i);
			packed |= static_cast<uint32_t>(strengths[i] - 1u) << (3 * i);
		}
	}

	effect[0] = active != 0 && (type != 0x26 || frequency != 0) ? type : 0x05;
	effect[1] = static_cast<uint8_t>(active);
	effect[2] = static_cast<uint8_t>(active >> 8u);
	effect[3] = static_cast<uint8_t>(packed);
	effect[4] = static_cast<uint8_t>(packed >> 8u);
	effect[5] = static_cast<uint8_t>(packed >> 16u);
	effect[6] = static_cast<uint8_t>(packed >> 24u);
	effect[9] = frequency;
	return true;
}

static bool trigger_effect_to_dualsense(const PadTriggerEffectCommand& command, uint8_t* effect) {
	std::memset(effect, 0, 11);
	effect[0] = 0x05;

	uint8_t strengths[10] = {};
	switch (command.mode) {
		case 0: return true;
		case 1:
			if (command.data[0] > 9 || command.data[1] > 8) {
				return false;
			}
			std::fill(strengths + command.data[0], strengths + 10, command.data[1]);
			return trigger_effect_zones(effect, strengths, 0x21);
		case 2: {
			const auto start    = command.data[0];
			const auto end      = command.data[1];
			const auto strength = command.data[2];
			if (start < 2 || start > 7 || end <= start || end > 8 || strength > 8) {
				return false;
			}
			if (strength == 0) {
				return true;
			}
			const uint16_t zones = static_cast<uint16_t>((1u << start) | (1u << end));
			effect[0]            = 0x25;
			effect[1]            = static_cast<uint8_t>(zones);
			effect[2]            = static_cast<uint8_t>(zones >> 8u);
			effect[3]            = strength - 1u;
			return true;
		}
		case 3:
			if (command.data[0] > 9 || command.data[1] > 8) {
				return false;
			}
			std::fill(strengths + command.data[0], strengths + 10, command.data[1]);
			return trigger_effect_zones(effect, strengths, 0x26, command.data[2]);
		case 4: return trigger_effect_zones(effect, command.data, 0x21);
		case 5: {
			const int start          = command.data[0];
			const int end            = command.data[1];
			const int start_strength = command.data[2];
			const int end_strength   = command.data[3];
			if (start > 8 || end <= start || end > 9 || start_strength < 1 || start_strength > 8 ||
			    end_strength < 1 || end_strength > 8) {
				return false;
			}
			for (int i = start; i < 10; i++) {
				const int delta  = (end_strength - start_strength) * (i - start);
				const int length = end - start;
				strengths[i]     = static_cast<uint8_t>(
				    i >= end ? end_strength
				             : start_strength +
				                   (delta + (delta < 0 ? -length / 2 : length / 2)) / length);
			}
			return trigger_effect_zones(effect, strengths, 0x21);
		}
		case 6: return trigger_effect_zones(effect, command.data + 1, 0x26, command.data[0]);
		default: return false;
	}
}

void Initialize() {
	EXIT_IF(g_controller != nullptr);

	g_controller = new GameController;
	g_controller->Connect(HOST_INPUT_CONTROLLER_ID);
}

void Shutdown() {
	EmergencyShutdown();
	delete g_controller;
	g_controller = nullptr;
}

void EmergencyShutdown() {
	if (g_controller != nullptr) {
		g_controller->ReleaseHostPads();
	}
}

void GameController::Connect(int id) {
	Common::LockGuard lock(m_mutex);

	if (std::find(m_connected_ids.begin(), m_connected_ids.end(), id) != m_connected_ids.end()) {
		return;
	}

	m_connected_ids.push_back(id);

	CheckActive();
}

void GameController::Disconnect(int id) {
	Common::LockGuard lock(m_mutex);

	// SDL2 reports every removed joystick as a removed controller, also one that never was one (a wheel, a virtual or
	// unmapped device: "we don't know if this was a game controller"), and a controller that could not be opened was
	// never connected here (issue #49: the game ended when such a device went away).
	const auto it = std::find(m_connected_ids.begin(), m_connected_ids.end(), id);
	// A disconnect for a pad never seen connected (a host pad that dropped while it was being opened)
	// closed the game: it changes nothing here.
	if (it == m_connected_ids.end()) {
		return;
	}

	m_connected_ids.erase(it);

	CheckActive();
}

void GameController::CheckActive() {
	int  new_active_id = -1;
	bool new_connected = false;

	if (!m_connected_ids.empty()) {
		new_active_id = m_connected_ids[0];
		for (const auto id: m_connected_ids) {
			if (id != HOST_INPUT_CONTROLLER_ID) {
				new_active_id = id;
				break;
			}
		}
		new_connected = true;
	}

	if (m_connected == new_connected && m_active_id == new_active_id) {
		return;
	}
	if (!m_connected && new_connected) {
		m_connected_count++;
	}
	m_active_id     = new_active_id;
	m_connected     = new_connected;
	m_state         = {};
	m_states_num    = 0;
	m_first_state   = 0;
	m_next_touch_id = 1;
}

void GameController::AddState() {
	if (m_states_num >= STATES_MAX) {
		m_states_num  = STATES_MAX - 1;
		m_first_state = (m_first_state + 1) % STATES_MAX;
	}

	const auto index  = (m_first_state + m_states_num) % STATES_MAX;
	m_states[index]   = m_state;
	m_obtained[index] = false;
	m_states_num++;
}

void GameController::Button(int id, uint32_t button, bool down) {
	Common::LockGuard lock(m_mutex);

	// The keyboard shares the player-1 pad with the active gamepad.
	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		m_state.time = LibKernel::KernelGetProcessTime();

		m_state.buttons = down ? m_state.buttons | button : m_state.buttons & ~button;

		AddState();
	}
}

void GameController::Axis(int id, Controller::Axis axis, int value) {
	Common::LockGuard lock(m_mutex);

	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		m_state.time = LibKernel::KernelGetProcessTime();

		int axis_id = static_cast<int>(axis);

		EXIT_IF(axis_id < 0 || axis_id >= static_cast<int>(Controller::Axis::AxisMax));

		m_state.axes[axis_id] = value;

		uint32_t trigger = 0;
		if (axis == Controller::Axis::TriggerLeft) {
			trigger = PAD_BUTTON_L2;
		} else if (axis == Controller::Axis::TriggerRight) {
			trigger = PAD_BUTTON_R2;
		}
		if (trigger != 0) {
			m_state.buttons = value > 0 ? m_state.buttons | trigger : m_state.buttons & ~trigger;
		}

		AddState();
	}
}

void GameController::RightStick(int id, int x, int y) {
	Common::LockGuard lock(m_mutex);

	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		m_state.time                                 = LibKernel::KernelGetProcessTime();
		m_state.axes[static_cast<int>(Axis::RightX)] = x;
		m_state.axes[static_cast<int>(Axis::RightY)] = y;
		AddState();
	}
}

void GameController::TouchPad(int id, int finger, bool down, float x, float y) {
	if (finger < 0 || finger >= 2) {
		return;
	}

	Common::LockGuard lock(m_mutex);
	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		auto& touch  = m_state.touch[finger];
		m_state.time = LibKernel::KernelGetProcessTime();
		if (down && !touch.down) {
			touch.id        = m_next_touch_id;
			m_next_touch_id = m_next_touch_id == 127 ? 1 : m_next_touch_id + 1;
		}
		touch.down = down;
		touch.x    = static_cast<uint16_t>(std::clamp(x, 0.0f, 1.0f) * 1920.0f);
		touch.y    = static_cast<uint16_t>(std::clamp(y, 0.0f, 1.0f) * 943.0f);
		if (id == HOST_INPUT_CONTROLLER_ID) {
			m_state.buttons = down ? m_state.buttons | PAD_BUTTON_TOUCH_PAD
			                       : m_state.buttons & ~PAD_BUTTON_TOUCH_PAD;
		}
		AddState();
	}
}

void GameController::ResetInputState() {
	Common::LockGuard lock(m_mutex);
	m_state         = {};
	m_state.time    = LibKernel::KernelGetProcessTime();
	m_states_num    = 0;
	m_first_state   = 0;
	m_next_touch_id = 1;
	AddState();
}

void GameController::ReleaseHostPads() {
	Common::LockGuard lock(m_mutex);

	std::vector<SDL_GameController*> pads;
	for (const auto id: m_connected_ids) {
		if (id == HOST_INPUT_CONTROLLER_ID) {
			continue;
		}
		if (auto* pad = SDL_GameControllerFromInstanceID(static_cast<SDL_JoystickID>(id));
		    pad != nullptr) {
			if (SDL_GameControllerGetType(pad) == SDL_CONTROLLER_TYPE_PS5) {
				DualSenseEffects effect {};
				effect.enable_bits     = 0x0c;
				effect.right_trigger[0] = 0x05;
				effect.left_trigger[0]  = 0x05;
				(void)SDL_GameControllerSendEffect(pad, &effect, sizeof(effect));
			}
			(void)SDL_GameControllerRumble(pad, 0, 0, 0);
			(void)SDL_GameControllerSetLED(pad, 0, 0, 0);
			pads.push_back(pad);
		}
	}

	if (!pads.empty()) {
		SDL_Delay(RELEASE_FLUSH_MS);
		for (auto* pad: pads) {
			SDL_GameControllerClose(pad);
		}
	}

	m_connected_ids.clear();
}

void GameController::SetVibration(uint8_t large_motor, uint8_t small_motor) {
	Common::LockGuard lock(m_mutex);

	if (m_active_id == HOST_INPUT_CONTROLLER_ID) {
		return;
	}

	auto* pad = SDL_GameControllerFromInstanceID(static_cast<SDL_JoystickID>(m_active_id));
	if (pad == nullptr) {
		return;
	}

	const auto large = static_cast<uint16_t>(large_motor * 0x101U);
	const auto small = static_cast<uint16_t>(small_motor * 0x101U);
	if (SDL_GameControllerRumble(pad, large, small, RUMBLE_DURATION_MS) != 0) {
		LOGF("\t rumble failed: %s\n", SDL_GetError());
	}
}

void GameController::SetLightBar(uint8_t r, uint8_t g, uint8_t b) {
	Common::LockGuard lock(m_mutex);
	if (auto* pad = SDL_GameControllerFromInstanceID(static_cast<SDL_JoystickID>(m_active_id));
	    pad != nullptr) {
		(void)SDL_GameControllerSetLED(pad, r, g, b);
	}
}

bool GameController::SetTriggerEffect(const PadTriggerEffectParam& param) {
	if ((param.trigger_mask & ~0x03u) != 0) {
		return false;
	}

	DualSenseEffects effect {};
	if ((param.trigger_mask & 0x01u) != 0) {
		effect.enable_bits |= 0x08;
		if (!trigger_effect_to_dualsense(param.command[0], effect.left_trigger)) {
			return false;
		}
	}
	if ((param.trigger_mask & 0x02u) != 0) {
		effect.enable_bits |= 0x04;
		if (!trigger_effect_to_dualsense(param.command[1], effect.right_trigger)) {
			return false;
		}
	}
	if (effect.enable_bits == 0) {
		return true;
	}

	Common::LockGuard lock(m_mutex);
	// Remember it for scePadGetTriggerEffectState even when no DualSense takes it: a game that polls
	// the state has to see the effect it asked for, not a zero.
	m_state_effect = param;
	auto* pad = SDL_GameControllerFromInstanceID(static_cast<SDL_JoystickID>(m_active_id));
	if (pad != nullptr && SDL_GameControllerGetType(pad) == SDL_CONTROLLER_TYPE_PS5) {
		(void)SDL_GameControllerSendEffect(pad, &effect, sizeof(effect));
	}
	return true;
}

// A trigger's travel, 0..255. An L2/R2 that arrives as a button only (input remapping, pads or
// tools that report the trigger digitally) counts as full travel.
static int TriggerTravel(const ControllerState& state, Axis axis) {
	const auto button = axis == Axis::TriggerLeft ? PAD_BUTTON_L2 : PAD_BUTTON_R2;
	const int  value  = state.axes[static_cast<int>(axis)];
	// An analog press also sets the button, so only a button without any travel is digital.
	return (state.buttons & button) != 0 && value == 0 ? 255 : value;
}

// A real DualSense reports which effect runs and where the trigger is against it. Without that
// report (SDL does not expose it, and other pads have no trigger motors) the state follows from the
// game's effect and the trigger's travel: 0..255 maps to the effect's ten positions 0..9. Games wait
// for the weapon or feedback state before acting on L2/R2 - the state stayed 0 before, so those
// actions only ever half happened (a bow's draw, for one).
static int32_t TriggerEffectState(const PadTriggerEffectCommand& command, int value) {
	const int  pos     = std::clamp(value, 0, 255) * 10 / 256;
	const bool pressed = value >= 8; // below this a trigger counts as released (analog rest noise)

	switch (command.mode) {
		case 1: // feedback: resistance from position data[0], strength data[1]
			return pressed && command.data[1] != 0 && pos >= command.data[0]
			           ? PAD_TRIGGER_STATE_IS_PUSHING
			           : PAD_TRIGGER_STATE_NO_FORCE;
		case 2: // weapon: resistance from data[0], released at data[1]
			if (pos >= command.data[1]) {
				return PAD_TRIGGER_STATE_WEAPON_FULLY_PRESSED;
			}
			return pos >= command.data[0] ? PAD_TRIGGER_STATE_WEAPON_ALMOST_PRESSED
			                              : PAD_TRIGGER_STATE_WEAPON_NOT_PRESSED;
		case 3: // vibration from position data[0], amplitude data[1], frequency data[2]
			return pressed && command.data[1] != 0 && command.data[2] != 0 && pos >= command.data[0]
			           ? PAD_TRIGGER_STATE_VIBRATION_IS_FIRING
			           : PAD_TRIGGER_STATE_VIBRATION_NOT_FIRING;
		case 4: // multiple-position feedback: strength per position
			return pressed && command.data[pos] != 0 ? PAD_TRIGGER_STATE_IS_PUSHING
			                                         : PAD_TRIGGER_STATE_NO_FORCE;
		case 5: // slope feedback from position data[0]
			return pressed && pos >= command.data[0] ? PAD_TRIGGER_STATE_IS_PUSHING
			                                         : PAD_TRIGGER_STATE_NO_FORCE;
		case 6: // multiple-position vibration: frequency data[0], amplitude per position
			return pressed && command.data[0] != 0 && command.data[1 + pos] != 0
			           ? PAD_TRIGGER_STATE_VIBRATION_IS_FIRING
			           : PAD_TRIGGER_STATE_VIBRATION_NOT_FIRING;
		default: return 0;
	}
}

int32_t GameController::GetTriggerEffectState(bool left) {
	Common::LockGuard lock(m_mutex);
	const auto        index = left ? 0 : 1;
	if ((m_state_effect.trigger_mask & (1u << index)) == 0) {
		return 0;
	}
	const auto axis = left ? Axis::TriggerLeft : Axis::TriggerRight;
	return TriggerEffectState(m_state_effect.command[index], TriggerTravel(m_state, axis));
}

void GameController::GetConnectionInfo(bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;
}

void GameController::ReadState(ControllerState* state, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(state == nullptr);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;
	*state = m_state;
}

int GameController::ReadStates(ControllerState* states, int states_num, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(states == nullptr);
	EXIT_IF(states_num < 1 || states_num > STATES_MAX);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;

	int ret_num = 0;

	if (m_connected) {
		if (m_states_num != 0) {
			for (uint32_t i = 0; i < m_states_num; i++) {
				if (ret_num >= states_num) {
					break;
				}
				auto index = (m_first_state + i) % STATES_MAX;
				if (!m_obtained[index]) {
					m_obtained[index] = true;

					states[ret_num++] = m_states[index];
				}
			}
		}
	}

	return ret_num;
}

void Connect(int id) {
	g_controller->Connect(id);
}

void Disconnect(int id) {
	g_controller->Disconnect(id);
}

void SetButton(int id, uint32_t button, bool down) {
	g_controller->Button(id, button, down);
}

void SetAxis(int id, Axis axis, int value) {
	g_controller->Axis(id, axis, value);
}

void SetRightStick(int id, int x, int y) {
	g_controller->RightStick(id, x, y);
}

void SetTouchPad(int id, int finger, bool down, float x, float y) {
	g_controller->TouchPad(id, finger, down, x, y);
}

void SetScriptedInput(int left_x, int left_y, int right_x, int right_y, uint32_t buttons, double seconds) {
	g_scripted.until.store(0, std::memory_order_release);
	const int values[] = {left_x, left_y, right_x, right_y};
	for (size_t i = 0; i < 4; i++) g_scripted.axes[i].store(values[i], std::memory_order_relaxed);
	g_scripted.buttons.store(buttons, std::memory_order_relaxed);
	if (seconds > 0) {
		g_scripted.until.store(SteadyNanoseconds() + static_cast<int64_t>(seconds * 1e9), std::memory_order_release);
	}
}

void ResetInputState() {
	g_controller->ResetInputState();
}

int KYTY_SYSV_ABI PadInit() {
	PRINT_NAME();

	return OK;
}

static bool PadOpenArgsAreValid(int user_id, int type, int index) {
	constexpr int user_id_system     = 0xff;
	constexpr int port_type_standard = 0;
	constexpr int port_type_special  = 2;
	constexpr int port_type_remote   = 16;
	const bool    personal_port =
	    user_id == Config::GetUserId() && (type == port_type_standard || type == port_type_special);
	const bool system_remote_control = user_id == user_id_system && type == port_type_remote;
	return index == 0 && (personal_port || system_remote_control);
}

int KYTY_SYSV_ABI PadOpen(int user_id, int type, int index, const void* param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n"
	     "\t param   = 0x%016" PRIx64 "\n",
	     user_id, type, index, reinterpret_cast<uint64_t>(param));

	constexpr int pad_error_invalid_arg = -2137915391; /* 0x80920001 */

	if (!PadOpenArgsAreValid(user_id, type, index)) {
		return pad_error_invalid_arg;
	}

	int handle = 1;

	return handle;
}

int KYTY_SYSV_ABI PadGetHandle(int user_id, int type, int index) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n",
	     user_id, type, index);

	constexpr int pad_error_device_no_handle = -2137915384; /* 0x80920008 */

	if (!PadOpenArgsAreValid(user_id, type, index)) {
		return pad_error_device_no_handle;
	}

	return 1;
}

int KYTY_SYSV_ABI PadSetMotionSensorState(int handle, bool enable) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));

	return OK;
}

int KYTY_SYSV_ABI PadSetAngularVelocityDeadbandState(int handle, bool enable) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));

	return OK;
}

int KYTY_SYSV_ABI PadResetOrientation(int handle) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

int KYTY_SYSV_ABI PadGetControllerInformation(int handle, PadControllerInformation* info) {
	PRINT_NAME();

	int  connected_count = 0;
	bool connected       = false;

	g_controller->GetConnectionInfo(&connected, &connected_count);

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (info == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	std::memset(info, 0, sizeof(*info));

	info->touch_pixel_density   = 44.86f;
	info->touch_resolution_x    = 1920;
	info->touch_resolution_y    = 943;
	info->stick_dead_zone_left  = controller_get_axis(-32768, 32767, 8000) - 128;
	info->stick_dead_zone_right = controller_get_axis(-32768, 32767, 8000) - 128;
	info->connection_type       = 0;
	info->connected_count       = static_cast<uint8_t>(std::min(connected_count, 255));
	info->connected             = connected;
	info->device_class          = 0;

	return OK;
}

int KYTY_SYSV_ABI PadReadState(int handle, PadData* data) {
	PRINT_NAME();
	Loader::DemonsSoulsWarp::RunQueuedCommand(); // (live `gamecmd`: on the game's own thread)
	Libs::Graphics::AntiLag::OnInput();          // KYTY_ANTILAG: the frame's INPUT mark

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState state;

	g_controller->ReadState(&state, &connected, &connected_count);

	pad_fill_data(data, state, connected, connected_count);

	return OK;
}

int KYTY_SYSV_ABI PadRead(int handle, PadData* data, int num) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(num < 1 || num > 64);
	Libs::Graphics::AntiLag::OnInput(); // KYTY_ANTILAG: the frame's INPUT mark
	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	std::memset(data, 0, sizeof(PadData) * static_cast<size_t>(num));

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState states[64]      = {};

	int ret_num = g_controller->ReadStates(states, num, &connected, &connected_count);

	if (!connected || ret_num == 0) {
		if (connected) {
			g_controller->ReadState(&states[0], &connected, &connected_count);
		}
		ret_num = 1;
	}

	for (int i = 0; i < ret_num; i++) {
		pad_fill_data(&data[i], states[i], connected, connected_count);
	}

	return ret_num;
}

// 10-21: Demon's Souls vibrates through the DualSense's haptics (an audio port), never PadSetVibration: without the
// pad's USB audio endpoint the pad stayed still. The audio's loudness drives the rumble motors instead.
// KYTY_HAPTIC_RUMBLE=0 off; KYTY_HAPTIC_RUMBLE_GAIN (default 4) scales it.
void RumbleFromHaptics(float left, float right) {
	static const float gain = [] {
		const char* v = std::getenv("KYTY_HAPTIC_RUMBLE_GAIN");
		const float g = v != nullptr ? static_cast<float>(std::atof(v)) : 4.0f;
		return g > 0.0f ? g : 4.0f;
	}();
	if (g_controller == nullptr) return;
	const auto to_motor = [](float x) {
		x = x * gain;
		if (x < 0.02f) return 0;                      // noise floor: still
		return static_cast<int>(std::min(x, 1.0f) * 255.0f + 0.5f);
	};
	const int large = to_motor(left);
	const int small = to_motor(right);
	static int  last_large = -1, last_small = -1;
	static auto last_sent  = std::chrono::steady_clock::time_point {};
	const auto  now        = std::chrono::steady_clock::now();
	const bool  changed    = std::abs(large - last_large) > 6 || std::abs(small - last_small) > 6 ||
	                     ((large == 0) != (last_large == 0)) || ((small == 0) != (last_small == 0));
	if (!changed && now - last_sent < std::chrono::milliseconds(100)) return;
	if (!changed && large == 0 && small == 0) return;
	last_large = large;
	last_small = small;
	last_sent  = now;
	static bool logged = false;
	if (!logged && (large != 0 || small != 0)) {
		logged = true;
		std::printf("Pad: haptics played as rumble (no DualSense USB audio endpoint), first levels %d/%d\n", large, small);
		std::fflush(stdout);
	}
	g_controller->SetVibration(static_cast<uint8_t>(large), static_cast<uint8_t>(small));
}

int KYTY_SYSV_ABI PadSetVibration(int handle, const PadVibrationParam* param) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	LOGF("\t large_motor = %d\n"
	     "\t small_motor = %d\n",
	     static_cast<int>(param->large_motor), static_cast<int>(param->small_motor));

	g_controller->SetVibration(param->large_motor, param->small_motor);

	return OK;
}

int KYTY_SYSV_ABI PadResetLightBar(int handle) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

int KYTY_SYSV_ABI PadSetLightBar(int handle, const PadLightBarParam* param) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	g_controller->SetLightBar(param->r, param->g, param->b);

	return OK;
}

int KYTY_SYSV_ABI PadSetTriggerEffect(int handle, const PadTriggerEffectParam* param) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	return g_controller->SetTriggerEffect(*param) ? OK : PAD_ERROR_INVALID_ARG;
}

int KYTY_SYSV_ABI PadGetTriggerEffectState(int handle, PadTriggerEffectStateInformation* info) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (info == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	info->state[0] = g_controller->GetTriggerEffectState(true);
	info->state[1] = g_controller->GetTriggerEffectState(false);

	// The first changes, so a log shows whether the game polls this and what it sees (printf: the
	// console's LOGF is silent in the release configuration).
	static int                    reports = 0;
	static std::array<int32_t, 2> last {-1, -1};
	if (reports < 32 && (info->state[0] != last[0] || info->state[1] != last[1])) {
		std::printf("Pad: trigger state %d,%d\n", info->state[0], info->state[1]);
		last[0] = info->state[0];
		last[1] = info->state[1];
		reports++;
	}
	return OK;
}

} // namespace Libs::Controller
