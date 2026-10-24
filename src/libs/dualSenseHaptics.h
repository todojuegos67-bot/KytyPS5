#ifndef KYTY_LIBS_DUALSENSE_HAPTICS_H_
#define KYTY_LIBS_DUALSENSE_HAPTICS_H_

#include <cstdint>
#include <string>

// The DualSense's own output endpoint and its haptic actuators.
//
// Windows lists the pad's USB audio interface as an output device of its own, with FOUR channels:
// the first pair plays on the pad's speaker, the second pair drives its haptic actuators. The
// emulator never opened it, so the guest's vibration port (AUDIO_OUT_PORT_TYPE_VIBRATION) went
// nowhere and the pad never vibrated.
//
// This is the Windows implementation (WASAPI, through the endpoint the pad's USB ID names). The
// endpoint cannot be picked by name: Windows calls the pad's endpoint and the machine's speakers
// both "Speakers"/"扬声器" (see the notes). Linux keeps using SDL's device list.
//
// Environment and dualsense-haptics.txt (the file wins; both optional):
//   KYTY_DUALSENSE_HAPTICS=0|1     off when 0 (default: on)
//   KYTY_DUALSENSE_HAPTICS_CHANNELS=23|01|all   which of the endpoint's channels take the vibration
//                                  (default: 23, the actuator pair)
//   KYTY_DUALSENSE_HAPTICS_GAIN=<float>         applied to the vibration samples (default: 1.0)
//   KYTY_DUALSENSE_SPEAKER_CHANNELS=01|23|all   the same for the pad speaker (default: 01)
//   KYTY_DUALSENSE_SPEAKER_GAIN=<float>         0..64 (default: 8.0, +18 dB). What the guest sends
//                                  to a pad speaker port is quiet - Demon's Souls peaks at about
//                                  -40 dB - and the controller's own amplifier then buries it in
//                                  its noise floor, so the digital gain is what makes it clear.
//                                  8 is clear, 16 borderline, 32 clips (4.0 is what
//                                  github.com/Jetsku/KytyPS5's launcher gives at 200%).
//   KYTY_DUALSENSE_HAPTICS_LOG=1   prints what it finds and what it writes (default: off)
namespace Libs::Controller::DualSenseHaptics {

// True when the host has a usable DualSense output endpoint (found once, then cached).
bool Available();

// The endpoint's own sample rate and channel count, 0 when there is no endpoint.
uint32_t SampleRate();
uint32_t EndpointChannels();

// Opens the endpoint for one of its routes: the pad speaker port plays on the endpoint's FIRST
// channel pair, the vibration port on its second (the actuators). Several routes share the one
// stream, and it stays open while any route is. False when there is no endpoint, when it cannot
// take the format, or when that route is already open: the caller then leaves the port as it was.
bool Open(bool speaker);
// Closes the route, and tears the stream down once no route is left.
void Close(bool speaker);

// True while the speaker route is open: a pad speaker port whose samples the pad takes must not be
// played on the main output as well, or the two paths' different latencies are heard as an echo.
bool SpeakerActive();

// Writes `frames` frames of `source_channels`-channel audio (signed 16-bit, or float when
// `is_float`) at `source_rate` into the route's channels of the endpoint, resampling when the rates
// differ and applying the route's gain. `gains`, when the guest set them, scales each source channel
// (its AudioOut2 gain attribute is where a pad's speaker and vibration levels live). Never waits:
// when the endpoint's queue is behind, the audio is dropped (haptics are best effort, and the caller
// is the guest's audio thread).
bool Write(const void* data, uint32_t frames, uint32_t source_rate, uint32_t source_channels,
           bool is_float, bool speaker, const float* gains);

} // namespace Libs::Controller::DualSenseHaptics

#endif /* KYTY_LIBS_DUALSENSE_HAPTICS_H_ */
