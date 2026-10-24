#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_

#include <cstdint>
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/render.h"

namespace Libs::Graphics {

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count = 0;
	bool                  ps_active = true;
	ShaderVertexInputInfo vs_input_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
};

struct DrawCallInfo {
	const char*          name = nullptr;
	CommandBufferDebugOp debug_op = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;
};

struct DrawEmitInfo {
	std::span<const vk::DrawIndexedIndirectCommand> direct_run;
	uint64_t run_mapping_epoch = 0, run_alias_epoch = 0;
	bool     indexed = false;
	// The offsets below come from the draw state (a direct DrawIndex), not from draw arguments:
	// a native record stores them for its direct draws.
	bool     state_offsets = false;
	int32_t  vertex_offset = 0;
	uint32_t first_vertex = 0;
	uint32_t first_instance = 0;
	// DrawIndexArgs::gpu_args: one indirect draw of these arguments.
	uint64_t gpu_args        = 0;
	uint32_t gpu_args_count  = 0;
	uint32_t gpu_args_stride = 0;
	uint64_t gpu_count_addr  = 0; // DrawIndexArgs::gpu_count_addr
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct DrawIndexRun {
	static constexpr size_t MaxDraws = 64;
	std::array<vk::DrawIndexedIndirectCommand, MaxDraws> commands {};
	DrawIndexBufferSource indices;
};
[[nodiscard]] bool BuildDrawIndexRun(std::span<const DrawIndexArgs> draws, DrawIndexRun& run);

struct ShaderVertexInputInfo;

[[nodiscard]] int32_t  ResolveVertexOffset(uint32_t                     index_offset,
                                           const ShaderVertexInputInfo& vs_input_info);
[[nodiscard]] uint32_t ResolveInstanceOffset(const ShaderVertexInputInfo& vs_input_info);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
