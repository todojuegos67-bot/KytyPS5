#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SUPER_RESOLUTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SUPER_RESOLUTION_H_

#include <cstdint>

namespace Libs::Graphics::SuperResolution {

// Sony's PS5 Pro super resolution (PSSR: libScePsml's "Mfsr") as Demon's Souls 01.007.000 dispatches it
// (SuperResolution::Apply in place of its TAA, display mode 5): the dispatch call (libs/libPsml.cpp) writes this
// packet into the game's command buffer (an IT_NOP of type R_SUPER_RESOLUTION) with what the call's parameter
// block held then; the command processor runs it in its order (CommandProcessor::SuperResolution).
// The parameter block: its first seven fields point to texture descriptors (T#s), copied here; the rest (camera,
// jitter, motion vector scale, sizes, reset flag) are copied as they are.
inline constexpr uint32_t TextureSlots   = 7;
inline constexpr uint32_t ParameterBytes = 0x160;

struct Packet {
	uint32_t context = 0;                   // the context's number (one per CreateContext)
	uint32_t output_width = 0, output_height = 0; // the context's output size
	uint32_t textures_given = 0;            // bit i: slot i's pointer was set
	uint32_t textures[TextureSlots][8] {};  // the T#s
	uint32_t parameters[ParameterBytes / 4] {};
};

inline constexpr uint32_t PacketDwords = 1 + static_cast<uint32_t>(sizeof(Packet) / 4); // with the header

} // namespace Libs::Graphics::SuperResolution

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SUPER_RESOLUTION_H_
