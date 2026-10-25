#ifndef EMULATOR_SRC_LOCAL_RT_LOCAL_LIGHTS_H_
#define EMULATOR_SRC_LOCAL_RT_LOCAL_LIGHTS_H_
// KYTY_RTLS_DEBUG=1|2 / KYTY_RT_LOCAL_SHADOWS (rtlocal-1025, step 1 of RT local-light shadows): finds the game's local
// light list (torches, lamps, braziers) in guest memory and proves its layout, without guessing a struct.
//
// Ground truth: every shadowed local light has a shadow-map projection somewhere the game's shading reads it (the spot
// pool 1024x1024x80, the cube pool 384x384x360). A perspective projection's apex (the point where its x, y and w rows
// are all 0: the light's position, whatever atlas scale/bias is folded into x and y) and its forward / near / far / cone
// are exact, independent facts about a real light. A buffer bound to the deferred lighting (CS 9fb5cc274b5e4f82 by
// default), to its tile culling (the CS writing the lighting's tile image) or to a shadow-pool consumer is the light
// list when float3s equal to >= 2 distinct apexes sit at one residue of one stride, the entries between them are
// plausible positions too, and it holds over >= 3 scans of different flips. The other fields of the proven struct are
// then reported against the same facts (radius = the shadow's far plane, the direction = its forward, the cone = its
// field of view), with value ranges for the rest (colour, type, shadow index).
//
// Only the normal dispatch path is seen (renderCompute.cpp's ProgTrace): a culling CS on the table path can be routed
// there with KYTY_TABLE_DENY=<hash>. The camera is FrameGen's G-buffer view-projection (an upscaler or frame
// generation on: FrameGen::Capturing()).
//
// Options: KYTY_RTLS_DEBUG=1 (the overlay: yellow crosses at shadow apexes, green circles at proven lights, with their
// radius when a field is proven to be it; a log line every KYTY_RTLS_PERIOD flips) | 2 (also the candidates and every
// scan); KYTY_RTLS_CS=<hex,...> (programs to scan, default the lighting CS); KYTY_RTLS_SCAN_ALL=1 (every normal-path
// CS); KYTY_RTLS_PERIOD (flips between scans of a program, 30); KYTY_RTLS_TOL (match tolerance in metres, 0.05);
// KYTY_RTLS_DUMP=<n> (n scans' raw buffers + apexes to KYTY_RTLS_DUMP_DIR, default _TempData/rtls, for offline RE).
// KYTY_RT_LOCAL_SHADOWS=1..3 turns the discovery on as well; the shadow pass itself (step 2) is not in this build.
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace RtLocal {

// Any of the options above (read once).
[[nodiscard]] bool Wanted();
// KYTY_RTLS_DEBUG > 0: the overlay's markers (DebugMarkers).
[[nodiscard]] bool OverlayWanted();

struct DispatchBuffer {
	uint64_t address = 0, size = 0;
	bool     written = false;
};
struct DispatchImage {
	uint64_t address = 0;
	uint32_t width = 0, height = 0, layers = 0;
	bool     depth_like = false; // depth, R32F or R16 (a shadow map)
	bool     written    = false;
};
// A normal-path dispatch's bindings, before it is recorded (render thread, only while Wanted()).
void OnDispatch(uint64_t cs_hash, std::span<const DispatchBuffer> buffers, std::span<const DispatchImage> images,
                std::span<const uint32_t> flattened_srt);

struct Marker {
	float       x = 0, y = 0, radius = 0; // pixels of the overlay
	uint32_t    rgba   = 0xffffffffu;     // IM_COL32 order (a << 24 | b << 16 | g << 8 | r)
	int         shape  = 0;               // 0 circle, 1 cross, 2 text only
	std::string label;
};
// KYTY_RTLS_DEBUG's overlay (presenter thread): the markers for a game image at (x0, y0) of w x h pixels; false: none.
bool DebugMarkers(float x0, float y0, float w, float h, std::vector<Marker>& out);

} // namespace RtLocal

#endif // EMULATOR_SRC_LOCAL_RT_LOCAL_LIGHTS_H_
