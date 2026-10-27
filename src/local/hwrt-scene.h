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

#include <array>
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

// (hwrt3, KYTY_HWRT_REFL_MATERIAL=1, only with KYTY_HWRT=1 and KYTY_HWRT_REFL=1) Real materials for the reflections'
// off-screen hits: a captured mesh also stores its pixel shader's base-colour texcoord (a vertex output, packed as half2
// into its world vertices' w) and that shader's base-colour texture, copied once (downsampled, 128x128) into a layer of
// an albedo array this module owns - the hit shader never reads a game image, so no game view can be stale for it.
[[nodiscard]] bool MaterialRequested();
[[nodiscard]] bool MaterialActive();
// One sampled image of a G-buffer draw's pixel stage, as its descriptor set binds it (the table set, made once): the
// SPIR-V binding and array element, the cache's image and its serial then, the view.
struct PsImage {
	uint32_t          binding = 0, element = 0;
	uint32_t          image_index = UINT32_MAX, image_generation = 0;
	uint64_t          serial = 0;
	vk::ImageView     view   = nullptr;
	vk::Format        format = vk::Format::eUndefined;
	vk::ImageViewType type   = vk::ImageViewType::e2D;
};
// The albedo array for the reflections (binding 24 of their material pipeline): false while it does not exist.
[[nodiscard]] bool MaterialAtlas(vk::ImageView& view, vk::Sampler& sampler);

// One G-buffer draw (one gl_DrawID of a vk draw call).
struct Draw {
	// (hwrt3, MaterialActive only) the draw's pixel module and its sampled images, and the texture cache
	vk::ShaderModule             ps_module = nullptr;
	const std::vector<PsImage>*  ps_images = nullptr;
	TextureCache*                cache     = nullptr;
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
// True: this draw is captured this frame (else its record captures nothing).
bool                   Note(uint32_t record, const Draw& draw);
// A known G-buffer vertex shader's flattened SRT (words 29..44: the camera's view-projection, clip = M * world).
void NoteCamera(uint64_t vs_hash, std::span<const uint32_t> flattened_srt);

// After the game's TAA dispatch (renderCompute.cpp, outside rendering): the frame's world-space gather, BLAS builds and
// refits, the TLAS; KYTY_HWRT_DEBUG's view into the TAA's output (images[6]); the 10 s log.
void AfterTaa(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer command, std::span<const TextureBinding> images);
// The same build where the game has no TAA pass: the PS5 Pro path's super-resolution packet (renderSuperResolution.cpp,
// before the upscale; `where` names the hook in the log once). Commands go in order with the renderer's (a recording
// packet with KYTY_VULKAN_RECORDING).
void EndFrame(GraphicContext& graphics, vk::CommandBuffer command, const char* where);
// (hwrt2) The scene just built, for the ray-traced effects recorded after it in the same frame (KYTY_HWRT_REFL): the
// TLAS (a VkAccelerationStructureKHR as an integer) and a storage buffer holding RayInfo (hwrt-scene.cpp: clip <-> world
// rows, the camera, the materials' address). False: no scene this frame.
struct RayScene {
	uint64_t       tlas = 0;
	vk::Buffer     info = nullptr;
	vk::DeviceSize size = 0;
};
[[nodiscard]] bool CurrentRayScene(RayScene& out);
// KYTY_HWRT_DEBUG's view into an RGBA16F storage image in the GENERAL layout (the PS5 Pro path: the upscaled output).
void DebugView(GraphicContext& graphics, vk::CommandBuffer command, vk::ImageView view, vk::Format format, vk::Extent2D extent);
// (KYTY_HWRT_SHADOW, frame-gen.cpp) The scene TLAS for a ray query pass recorded after this frame's build, in order with it
// (null until one exists); built_now: it was built this frame (else it holds an older frame's scene).
[[nodiscard]] vk::AccelerationStructureKHR SceneTlas(bool* built_now);
// (KYTY_HWRT_LIGHTS, hw-lights.cpp) The G-buffer camera's view-projection of the last captured frame (the TLAS's world;
// row-major, clip = M * world); false until one was seen.
bool SceneCamera(std::array<float, 16>& vp);
// (KYTY_HWRT_WATER, rt-water.cpp) The water surfaces' own TLAS, built this frame next to the scene's from the draws of
// the water pixel shaders (FrameGen::RtWaterIsWaterPs; never in the scene TLAS, so no other pass sees them), and its
// materials words (8 per instance, as RayInfo's: PS, VS, id, world vertices, indices). False: none this frame.
struct WaterScene {
	vk::AccelerationStructureKHR tlas = nullptr;
	vk::DeviceAddress            materials = 0;
	uint32_t                     instances = 0;
};
[[nodiscard]] bool CurrentWaterScene(WaterScene& out);
// (the 10 s log) water draws noted, water meshes known / built, frames with a water TLAS (counters since the start)
void WaterCounters(uint64_t& draws, uint32_t& known, uint32_t& built, uint64_t& frames);

// (hwrt2g, KYTY_HWRT_AO / KYTY_HWRT_GI: hwrt-gi.cpp) The TLAS built this frame and its camera (clip -> world rows, world ->
// clip rows, the position), for a ray query pass recorded after the build in the same frame. False: no scene this frame.
struct GiSceneInfo {
	vk::AccelerationStructureKHR tlas = nullptr;
	std::array<float, 16>        inv {}, vp {};
	std::array<float, 4>         cam {};
};
[[nodiscard]] bool GiScene(GiSceneInfo& out);

} // namespace HwScene
} // namespace Libs::Graphics

#endif
