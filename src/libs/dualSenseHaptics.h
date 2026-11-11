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
// This is the Windows implementation (WASAPI, through the endpoint the pad's USB ID names, or, where no
// endpoint's path holds it, the one whose whole name names the pad: "扬声器 (DualSense Wireless
// Controller)"; the short name is "Speakers"/"扬声器" for the machine's speakers too). Linux opens the
// same USB audio device through SDL2's audio list (PipeWire/PulseAudio/ALSA name it "DualSense Wireless
// Controller ..." or "Wireless Controller"), the vibration route only, and OFF by default there:
// KYTY_DUALSENSE_HAPTICS=1 turns it on; KYTY_DUALSENSE_HAPTICS_DEVICE=<text> names the device when the
// list's names differ. The pad speaker port keeps the main output on Linux.
//
// The environment, all optional (the launcher passes these as -Set options):
//   KYTY_DUALSENSE_HAPTICS=0|1     off when 0 (default: on; Linux: off until set to anything but 0)
//   KYTY_DUALSENSE_HAPTICS_DEVICE=<text>  Linux: the SDL output device whose name holds this text is the pad
//                                  (default: audio.cpp's finder: a DualSense-named device, none when two)
//   KYTY_DUALSENSE_HAPTICS_CHANNELS=23|01|all   which of the endpoint's channels take the vibration
//                                  (default: 23, the actuator pair)
//   KYTY_DUALSENSE_HAPTICS_GAIN=<float>         applied to the vibration samples (default: 4.0 - the
//                                  actuators are felt rather than heard, and at 1.0 they are barely
//                                  there in play)
//   KYTY_DUALSENSE_VIBRATION_MOTOR_GAIN=<float> 0..16 (default: 2.0). How hard the rumble motors are
//                                  driven from a block's peak: 0 leaves the work to the actuators
//                                  (faint), 2.0 is the mainline's reference, 4 doubles it. Strength is
//                                  taste - lower it when a spell's whole charge buzzes, raise it when a
//                                  hit is barely there.
//   KYTY_DUALSENSE_VIBRATION_MOTOR_MS=<1..500>   how long each drive report lasts (default: 60; SDL2 reads
//   KYTY_DUALSENSE_VIBRATION_MOTOR_DECAY=<0..1)  0 as "no expiry", so 1 is the smallest; the fade also sends
//                                  an explicit stop when it reaches its floor) and how fast the hold falls
//                                  between reports (default: 0.6; must be below 1 - at exactly 1 it never
//                                  reaches the floor). Together, the tail of a hit: a hit is one block,
//                                  so the default ends it in about 250 ms; longer or slower reads as a
//                                  hit that keeps rumbling.
//   KYTY_DUALSENSE_SPEAKER_CHANNELS=01|23|all   the same for the pad speaker (default: 01)
//   KYTY_DUALSENSE_SPEAKER_GAIN=<float>         0..64, default 1.0 (unity). The pad's own amplifier and
//                                  the +6 dB preamp below make the speaker audible; digital gain on
//                                  top of them clips inside the pad, heard as tearing rather than as
//                                  loudness - which is what the released 8.0 (+18 dB) here did. The Bluetooth
//                                  route uses the same unity gain, so both routes sound alike.
//   KYTY_DUALSENSE_SPEAKER_PREAMP=0|1  the controller's own +6 dB speaker preamp (default: on).
//                                  github.com/Jetsku/KytyPS5 sends it, Linux does too, and so does
//                                  that add-on - so it is the preamp plus too much gain, not the
//                                  preamp alone, that clips. Turn it off to buy 6 dB of headroom.
//   KYTY_DUALSENSE_SPEAKER_VOLUME=0..100  the controller's own volume (default: 100). Lowering it
//                                  leaves its amplifier more headroom, and the gain above takes over
//                                  the loudness.
//   KYTY_DUALSENSE_SPEAKER_LIMITER=0|1  a soft limiter above 0.8 on the speaker route (default: off):
//                                  loud moments bend towards full scale instead of clamping, which is
//                                  what the controller's amplifier then finds hard to take.
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
