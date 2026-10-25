#ifndef EMULATOR_SRC_LOCAL_HWRT_SCENE_H_
#define EMULATOR_SRC_LOCAL_HWRT_SCENE_H_
// KYTY_HWRT=1: hardware ray tracing, phase 1 - the game's opaque 3D scene captured into Vulkan acceleration structures
// (VK_KHR_acceleration_structure + VK_KHR_ray_query) every frame, with no visual change (KYTY_HWRT_DEBUG=1|2: a ray
// query view over the TAA's output, to check the geometry lines up with the screen). Off by default: nothing here runs,
// no extension or feature is asked for, and no pipeline changes without it.
//
//  Capture point: the G-buffer draws (colour targets >= 3, a depth target with depth writes, a triangle list), on the
//  table path, where almost all of them are drawn (a render-thread cost of one record per draw, nothing on the game's
//  threads). Their pipeline is a variant whose vertex shader (the translated SPIR-V, patched) also stores gl_Position at
//  capture[rec.base + instance * rec.window + (gl_VertexIndex - gl_BaseVertex)], rec = records[pc[31] + gl_DrawID] (push
//  constant dword 31, outside every program's push data; no render-pass break, indirect and multi-draws included, every
//  instance of an instanced draw). The positions are the game's own clip-space output (skinning, wind, instancing all
//  done by its shader); the camera's view-projection (flattened SRT words 29.., FrameGen) inverted gives world space.
//  Mesh key: index buffer address + first index, index count, instances, vertex offset, VS (what draws the same mesh).
//  A new key copies its indices (guest memory, read on the CPU once) and gets its own world-space vertex buffer + BLAS.
//  A static key is captured again only every ~32-64 frames to compare (a GPU max-delta); one that moved becomes dynamic:
//  captured and refit every frame. The TLAS (every key seen in the last KYTY_HWRT_KEEP frames) is rebuilt every frame.
//  All the GPU work is recorded right after the game's TAA dispatch (graphics queue).
#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

// (<windows.h>, through vk_mem_alloc.h, defines MemoryBarrier; vk::MemoryBarrier is meant everywhere here.)
#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Libs::Graphics {
class TextureCache;
struct TextureBinding;
namespace HwScene {

// The push constant dword holding the draw's first record (no program's push data reaches it: checked per draw).
constexpr uint32_t CursorDword = 31;

// KYTY_HWRT=1 (read once).
[[nodiscard]] bool Requested();
// At device creation (only when Requested): the extensions, and the features chained in front of `next` when the
// device has them all (Active() false otherwise, logged).
void  AddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& device_extensions);
void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& device_extensions, void* next);
// Requested and the device has the features.
[[nodiscard]] bool Active();

// The capturing variant of a vertex module (made once per module; null when it cannot be made: logged).
vk::ShaderModule CaptureModule(GraphicContext& graphics, vk::ShaderModule original);
// The pipeline-key id of a vertex program's capturing variant (the buffers' addresses baked into its SPIR-V mixed in).
[[nodiscard]] uint64_t VariantId(uint64_t program_id);

// Why a draw was not captured (the 10 s log).
enum class Skip : uint32_t {
	NotScene,     // (not counted: not a G-buffer draw)
	NoVariant,    // its capture pipeline is not made yet (compiling, or the shader could not be patched)
	Speculation,  // drawn by a speculation thread (KYTY_SPECULATE)
	PushData,     // a program's push data reaches dword 31
	Records,      // the frame's record ring is full
	Topology,     // indices unreadable / not a triangle list count / too many vertices
	Budget,       // a new mesh left for a later frame (build budget)
	Entries,      // the mesh table is full
	GpuWritten,   // its indirect arguments or indices were written by the GPU (guest memory stale)
	Count
};
void NoteSkip(Skip why, uint32_t draws = 1);
// A G-buffer draw state seen at the table store (counted once per state, for the log).
void NoteSceneState(bool capturable);

// One G-buffer draw (one gl_DrawID of a vk draw call).
struct Draw {
	uint64_t vs = 0, ps = 0;
	uint64_t index_address  = 0; // guest address of the draw's first index (index base + first index * size)
	uint32_t index_size     = 2; // 2 or 4
	uint32_t index_count    = 0;
	uint32_t instance_count = 1;
	int32_t  vertex_offset  = 0;
};
// Render thread, before a capture-pipeline draw call of `count` draws: the cursor to push (always valid: a record
// that captures nothing when the ring is full), then Note for each draw (record = cursor + i).
[[nodiscard]] uint32_t Reserve(uint32_t count);
void                   Note(uint32_t record, const Draw& draw);
// A known G-buffer vertex shader's flattened SRT (words 29..44: the camera's view-projection, clip = M * world).
void NoteCamera(uint64_t vs_hash, std::span<const uint32_t> flattened_srt);

// After the game's TAA dispatch (renderCompute.cpp, outside rendering): the frame's world-space gather, BLAS builds and
// refits, the TLAS; KYTY_HWRT_DEBUG's view into the TAA's output (images[6]); the 10 s log.
void AfterTaa(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer command, std::span<const TextureBinding> images);

} // namespace HwScene
} // namespace Libs::Graphics

#endif
