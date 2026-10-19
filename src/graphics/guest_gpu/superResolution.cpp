#include "graphics/guest_gpu/superResolution.h"

#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/shader/shaderBindings.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

// KYTY_SR_LOG=1: what the game handed Sony's upscaler (its first packets, then one a minute at 60 fps).
static void LogPacket(const SuperResolution::Packet& sr) {
	static std::atomic<uint32_t> seen {0};
	const auto                   count = seen.fetch_add(1, std::memory_order_relaxed);
	if (count >= 4 && count % 3600 != 0) return;
	std::printf("Super resolution %u: context %u, output %ux%u\n", count, sr.context, sr.output_width, sr.output_height);
	for (uint32_t slot = 0; slot < SuperResolution::TextureSlots; slot++) {
		if ((sr.textures_given & (1u << slot)) == 0) {
			std::printf("  slot %u: none\n", slot);
			continue;
		}
		ShaderTextureResource t {};
		std::memcpy(t.fields, sr.textures[slot], sizeof(t.fields));
		std::printf("  slot %u: 0x%010" PRIx64 " %ux%u format %u tile %u type %u levels %u-%u dst %03x [%08x %08x %08x %08x "
		            "%08x %08x %08x %08x]\n",
		            slot, t.Base40(), t.Width5() + 1u, t.Height5() + 1u, static_cast<uint32_t>(t.Format()),
		            static_cast<uint32_t>(t.TileMode()), static_cast<uint32_t>(t.Type()), t.BaseLevel(), t.LastLevel(),
		            t.DstSelXYZW(), t.fields[0], t.fields[1], t.fields[2], t.fields[3], t.fields[4], t.fields[5],
		            t.fields[6], t.fields[7]);
	}
	for (uint32_t at = 0; at < SuperResolution::ParameterBytes / 4; at += 4) {
		float f[4] {};
		std::memcpy(f, &sr.parameters[at], sizeof(f));
		std::printf("  +0x%03x: %08x %08x %08x %08x | %g %g %g %g\n", at * 4, sr.parameters[at], sr.parameters[at + 1],
		            sr.parameters[at + 2], sr.parameters[at + 3], f[0], f[1], f[2], f[3]);
	}
	std::fflush(stdout);
}

void CommandProcessor::SuperResolution(const uint32_t* packet, uint32_t dwords) {
	// (A state pass: no work.)
	if (m_state_only) return;
	SuperResolution::Packet sr {};
	if (dwords * 4u < sizeof(sr)) return;
	std::memcpy(&sr, packet, sizeof(sr));
	static const bool log = std::getenv("KYTY_SR_LOG") != nullptr;
	if (log) LogPacket(sr);
	CheckBuffer();
	m_renderer.GetComputeRenderExecutor().RunSuperResolution(CurrentBuffer(), sr);
}

} // namespace Libs::Graphics
