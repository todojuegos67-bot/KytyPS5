#ifndef EMULATOR_SRC_LOCAL_HW_RT_H_
#define EMULATOR_SRC_LOCAL_HW_RT_H_
// KYTY_RT_SHADOWS_HW=1 (with KYTY_RT_SHADOWS=1..3): hardware ray-traced sun shadows (VK_KHR_acceleration_structure +
// VK_KHR_ray_query), a prototype. Off by default; nothing here runs, and no extension or feature is asked for, without it.
//
//  1. Capture: the game's own sun shadow-map draws (a depth-only draw into a square, layered >= 512 px depth target)
//     get a variant of their vertex shader: the translated SPIR-V with a store of gl_Position into a capture buffer
//     (buffer device address, baked in) at [cursor + gl_VertexIndex], the cursor set by the render thread before the
//     draw. The draw's indices are copied (rebased) on the CPU into a host-visible index ring; the orthographic light
//     matrix is taken from the draw's flattened SRT and inverted (light clip -> the matrix's input space, world).
//  2. Build: one BLAS per frame (a geometry per captured draw; its transform = world -> camera view * light clip ->
//     world), refit when the draws' primitive counts are the same as last frame's; one TLAS of one instance.
//  3. Trace: cshadow.comp built with KYTY_HW_RT: a ray query per pixel toward the sun (the existing penumbra jitter),
//     in the camera's view space; the existing temporal + a-trous denoise and composite. Screen-space while the AS is
//     not ready.
#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::HwRt {

// KYTY_RT_SHADOWS_HW=1 (read once).
[[nodiscard]] bool Requested();
// At device creation (only when Requested): the extensions, and the features chained in front of `next` when the
// device has them all (false in Supported() otherwise, logged).
void  AddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& device_extensions);
void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& device_extensions, void* next);
[[nodiscard]] bool Supported();

// The capture (render thread). Whether this draw is one to capture: a depth-only draw into a sun shadow-map array.
[[nodiscard]] bool CaptureWanted();
// The vertex shaders seen drawing into a sun shadow-map array (Demon's Souls' daa84716d49bf71d known from the start):
// the table and native-record paths leave their draws to the normal path, where the capture is.
void NoteShadowVs(uint64_t vs_hash);
[[nodiscard]] bool IsShadowVs(uint64_t vs_hash);
// The capturing variant of a vertex module (made once per module; null when it cannot be made: logged once).
vk::ShaderModule CaptureModule(GraphicContext& graphics, vk::ShaderModule original);
// The pipeline-key id of a vertex program's capturing variant: the program's id mixed with the capture buffers'
// addresses baked into its SPIR-V (a pipeline stored by an earlier run, with other addresses, never matches it).
[[nodiscard]] uint64_t VariantId(uint64_t program_id);
// One draw: its indices (null: a non-indexed draw of `count` vertices from first_vertex), index element size (2 or 4),
// the vertex offset, the flattened SRT (for the light matrix). True and the cursor to set when it is captured.
bool PrepareDraw(GraphicContext& graphics, std::span<const uint32_t> srt, const void* indices, uint32_t element_size,
                 uint32_t count, int32_t vertex_offset, uint32_t first_vertex, uint32_t* cursor);
// Records the cursor's update (outside rendering) and the barrier to the vertex shader.
void RecordCursor(vk::CommandBuffer command, uint32_t cursor);

// The build (CShadowApply, outside rendering): the frame's captured draws into the BLAS / TLAS. world_to_view: 3x4 rows
// (view = M * [world, 1]); identity when the light matrices' input space is the view. Null: no AS this frame.
vk::AccelerationStructureKHR Build(GraphicContext& graphics, vk::CommandBuffer command, const std::array<double, 12>& world_to_view);
// Timestamps around the trace pass (the 10 s log's trace ms).
void TraceBegin(vk::CommandBuffer command);
void TraceEnd(vk::CommandBuffer command);
// The existing statistics' share of pixels shadowed (for the 10 s log).
void NoteShadowed(double percent);

} // namespace Libs::Graphics::HwRt

#endif
