#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/upstreamPaths.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/fsr3FrameGen.h"
#include "graphics/host_gpu/renderer/wolverineHudless.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include "gpu_tiler_shaders/mesh_indirect_args_batch_spv.h"
#include "gpu_tiler_shaders/mesh_indirect_args_spv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.has_dynamic_vertex_fetch) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

// A draw the host doesn't make, once per path and shaders and then at each doubling of its count. The project's rule
// is that no draw is skipped; these paths came from upstream, and this says whether the game reaches them.
static void NoteSkippedDraw(const char* path, const CommandBuffer& buffer, int vertex_stores = -1) {
	const auto& shaders = buffer.GetShaders();
	const auto  ps = shaders.GetPs().ps_regs.data_addr, es = shaders.GetVs().es_regs.data_addr,
	           gs = shaders.GetVs().gs_regs.data_addr;
	static std::mutex                                                        mutex;
	static std::map<std::tuple<const char*, uint64_t, uint64_t, uint64_t>, uint64_t> seen;
	std::lock_guard                                                          lock(mutex);
	const auto count = ++seen[{path, ps, es, gs}];
	if (std::has_single_bit(count)) {
		LOGF("Draw skipped (%s): count=%" PRIu64 " ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 " vertex_stores=%d\n", path, count,
		     ps, es, gs, vertex_stores);
	}
}

static void LogFramebufferSkip(const char* draw_name, const RenderDepthInfo& depth,
                               const CommandBuffer& buffer, uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF("DrawFramebufferSkip[%u]: %s depth_format=%s depth_image=%s depth_vaddr_num=%d "
	     "target_mask=0x%08" PRIx32 " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	     log_id, draw_name, vk::to_string(depth.desc.view_info.format).c_str(),
	     depth.image_id ? "yes" : "no",
	     static_cast<int>(!depth.desc.info.data.Empty()) +
	         static_cast<int>(depth.desc.info.HasStencil()),
	     ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	// Rect lists are drawn without culling.
	vk::CullModeFlags cull_mode {};
	if (!Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		if (mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
	}
	dynamic_state.SetCullMode(cull_mode);
	dynamic_state.SetFrontFace(mode.face ? vk::FrontFace::eClockwise
	                                     : vk::FrontFace::eCounterClockwise);

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps({front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		                            {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop || sh_regs.AlphaToMask();
}

// A draw's topology and shader stages; its targets are the executor's m_targets (upstream 7ae328b31).
struct DrawRenderState {
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
	// Input: the user-data slots a GPU-read indirect draw patches, and whether the GPU reads its records.
	ShaderIndirectOffsets indirect_offsets;
	bool                  indirect_draw = false;
};

// A draw's state holds three stages' arrays of 32 entries and was built whole for every draw, thousands of times a frame,
// though a draw fills few entries and nothing reads past a stage's counts. So the GPU thread keeps one, and puts back what
// the last draw can have written: everything but those arrays, and of them the entries up to the counts and the next one,
// which a lookup can leave half written. Only DrawIndex and DrawAuto take it, and DrawAuto calling itself is done with it.
static void ResetDrawRenderState(DrawRenderState& state, const DrawRenderState& fresh) {
	static_assert(std::is_trivially_copyable_v<DrawRenderState>);
	const auto bytes = [](auto& first, const auto& second, size_t size) { std::memcpy(&first, &second, size); };
	bytes(state, fresh, reinterpret_cast<const char*>(&fresh.vertex_info) - reinterpret_cast<const char*>(&fresh));
	bytes(state.ps_input_info, fresh.ps_input_info,
	      sizeof(DrawRenderState) - (reinterpret_cast<const char*>(&fresh.ps_input_info) - reinterpret_cast<const char*>(&fresh)));
	for (size_t i = 0; i < state.vertex_info.size(); i++) {
		auto&       info  = state.vertex_info[i];
		const auto& clean = fresh.vertex_info[i];
		const auto  resources =
		    static_cast<size_t>(std::min(info.resources_num + 1, static_cast<int>(ShaderVertexInputInfo::RES_MAX)));
		const auto buffers =
		    static_cast<size_t>(std::min(info.buffers_num + 1, static_cast<int>(ShaderVertexInputInfo::RES_MAX)));
		bytes(info.resources, clean.resources, resources * sizeof(info.resources[0]));
		bytes(info.resources_dst, clean.resources_dst, resources * sizeof(info.resources_dst[0]));
		bytes(info.buffers, clean.buffers, buffers * sizeof(info.buffers[0]));
		bytes(info.stage, clean.stage,
		      sizeof(ShaderVertexInputInfo) - (reinterpret_cast<const char*>(&clean.stage) - reinterpret_cast<const char*>(&clean)));
	}
}

static DrawRenderState& TakeDrawRenderState(ShaderIndirectOffsets indirect_offsets, bool indirect_draw) {
	static const auto            fresh = std::make_unique<const DrawRenderState>();
	thread_local auto            state = std::make_unique<DrawRenderState>();
	ResetDrawRenderState(*state, *fresh);
	state->indirect_offsets = indirect_offsets;
	state->indirect_draw    = indirect_draw;
	return *state;
}

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto& owner = cache.m_slot_images[target.image_id];
		if (((!owner.registered && !owner.info.data.Empty()) || owner.binding.needs_rebind) && !KYTY_UPSTREAM_PATH("86139e36f")) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto& owner = cache.m_slot_images[depth.image_id];
		if ((!owner.registered || owner.binding.needs_rebind) && !KYTY_UPSTREAM_PATH("86139e36f")) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		// Apply pending clears to compressed stencil as well as depth.
		if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("depth attachment feedback loop is not supported by the host\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = ctx.HasObjectPrimitiveIdOutput() ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh_regs.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh_regs.target_output_mode[index] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

bool DrawPixelExports(const HW::Context& ctx, const HW::Shader& sh_ctx,
                      std::array<Prospero::ColorComponentMapping, 8>& mapping) {
	const auto color_output_mask = DrawColorOutputMask(ctx);
	mapping                      = {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order)
			        .export_mapping;
		}
	}
	return sh_ctx.GetPs().ps_regs.data_addr != 0 &&
	       (color_output_mask != 0 || PixelShaderHasDepthOrCoverageSideEffects(ctx.GetShaderRegisters()));
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer,
                                                   uint32_t render_target_slice_offset) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, render_target_slice_offset, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// vertex_offset is the first vertex of non-indexed draws.
struct DrawEmitInfo {
	int32_t          vertex_offset  = 0;
	uint32_t         first_instance = 0;
	DrawIndirectArgs indirect;
	// A mesh draw whose arguments a guest shader wrote: their guest address, and whether the shader
	// takes no first vertex or first instance from them.
	uint64_t gpu_arguments = 0;
	bool     native        = true;
	// A draw whose instance count a guest shader wrote at this guest address (DrawIndexArgs).
	uint64_t instance_count_address = 0;
};

// The push constants of mesh_indirect_args.comp.
struct MeshArgumentsParameters {
	uint64_t arguments            = 0;
	uint64_t output               = 0;
	uint32_t primitive_size       = 0;
	uint32_t primitive_step       = 0;
	uint32_t primitives_per_group = 0;
	uint32_t native               = 0;
	uint32_t max_groups_x         = 0;
	uint32_t max_groups_y         = 0;
	uint32_t max_groups_total     = 0;
	uint32_t unused               = 0;
};
static_assert(sizeof(MeshArgumentsParameters) == 48);

// One draw in mesh_indirect_args_batch.comp's list.
struct MeshArgumentsRecord {
	uint64_t arguments            = 0;
	uint64_t output               = 0;
	uint32_t primitive_size       = 0;
	uint32_t primitive_step       = 0;
	uint32_t primitives_per_group = 0;
	uint32_t native               = 0;
};
static_assert(sizeof(MeshArgumentsRecord) == 32);

// The push constants of mesh_indirect_args_batch.comp.
struct MeshArgumentsBatchParameters {
	uint64_t records          = 0;
	uint32_t max_groups_x     = 0;
	uint32_t max_groups_y     = 0;
	uint32_t max_groups_total = 0;
	uint32_t unused           = 0;
};
static_assert(sizeof(MeshArgumentsBatchParameters) == 24);

vk::Pipeline RenderExecutor::MeshArgumentsPipeline() {
	if (m_mesh_arguments_pipeline) {
		return m_mesh_arguments_pipeline;
	}
	const auto                  device = m_context.GetGraphics().device;
	const vk::PushConstantRange range {vk::ShaderStageFlagBits::eCompute, 0,
	                                   sizeof(MeshArgumentsParameters)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &range;
	RequireVulkanSuccess(device.createPipelineLayout(&layout_info, nullptr, &m_mesh_arguments_layout),
	                     "create mesh-arguments pipeline layout");
	const auto module = CompileSPV(MESH_INDIRECT_ARGS_SPV, device);
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = m_mesh_arguments_layout;
	const auto result =
	    device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_mesh_arguments_pipeline);
	device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh-arguments pipeline");
	return m_mesh_arguments_pipeline;
}

vk::Pipeline RenderExecutor::MeshArgumentsBatchPipeline() {
	if (m_mesh_batch_pipeline) {
		return m_mesh_batch_pipeline;
	}
	const auto                  device = m_context.GetGraphics().device;
	const vk::PushConstantRange range {vk::ShaderStageFlagBits::eCompute, 0,
	                                   sizeof(MeshArgumentsBatchParameters)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &range;
	RequireVulkanSuccess(device.createPipelineLayout(&layout_info, nullptr, &m_mesh_batch_layout),
	                     "create mesh-arguments batch pipeline layout");
	const auto module = CompileSPV(MESH_INDIRECT_ARGS_BATCH_SPV, device);
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = m_mesh_batch_layout;
	const auto result =
	    device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_mesh_batch_pipeline);
	device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh-arguments batch pipeline");
	return m_mesh_batch_pipeline;
}

// A pass break and two barriers a GPU-made mesh draw cost Marvel's Wolverine's Trask about 6 ms of the GPU's 30 (inference
// from a GPU Trace). So one dispatch goes before the pass, reading a list in mapped memory that the pass's draws fill in
// until the command buffer is submitted. Its arguments are read earlier than the draw's place in the guest's order,
// which is the same as long as nothing wrote them in between: anything recorded that could (a dispatch, a transfer, a
// draw with buffer stores, an upload of CPU-written arguments in ObtainBuffer) calls EndRendering first, which closes
// the batch, and ExecutePreparedDraw closes it after a draw with any other store.
void RenderExecutor::BatchMeshArguments(CommandBuffer& buffer, const MeshArgumentsParameters& parameters) {
	static constexpr uint32_t Records       = 128;
	static constexpr uint32_t RecordsOffset = 16;
	static_assert(sizeof(vk::DispatchIndirectCommand) <= RecordsOffset);
	auto& ring  = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	auto& batch = m_mesh_arguments_batch;
	auto& scheduler = m_context.GetCommandScheduler();
	if (batch.mapped == nullptr || batch.count == Records || batch.tick != scheduler.CurrentTick() ||
	    batch.end_rendering_calls != buffer.EndRenderingCalls()) {
		const auto [mapped, offset] = ring.Map(RecordsOffset + Records * sizeof(MeshArgumentsRecord), 16);
		EXIT_IF(mapped == nullptr);
		const vk::DispatchIndirectCommand command {0u, 1u, 1u};
		std::memcpy(mapped, &command, sizeof(command));
		ring.Commit();
		buffer.EndRendering();
		const auto        vk_buffer = buffer.Recorder();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0,
		                          nullptr, 0, nullptr);
		const MeshArgumentsBatchParameters batch_parameters {
		    .records          = ring.BufferDeviceAddress() + offset + RecordsOffset,
		    .max_groups_x     = parameters.max_groups_x,
		    .max_groups_y     = parameters.max_groups_y,
		    .max_groups_total = parameters.max_groups_total,
		};
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, MeshArgumentsBatchPipeline());
		vk_buffer.pushConstants(m_mesh_batch_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                        sizeof(batch_parameters), &batch_parameters);
		vk_buffer.dispatchIndirect(ring.Handle(), offset); // host writes before the submit need no barrier
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
		after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
		                          vk::PipelineStageFlagBits::eDrawIndirect |
		                              vk::PipelineStageFlagBits::eMeshShaderEXT,
		                          {}, 1, &after, 0, nullptr, 0, nullptr);
		batch = {mapped, 0u, scheduler.CurrentTick(), buffer.EndRenderingCalls()};
	}
	const MeshArgumentsRecord record {
	    .arguments            = parameters.arguments,
	    .output               = parameters.output,
	    .primitive_size       = parameters.primitive_size,
	    .primitive_step       = parameters.primitive_step,
	    .primitives_per_group = parameters.primitives_per_group,
	    .native               = parameters.native,
	};
	std::memcpy(batch.mapped + RecordsOffset + batch.count * sizeof(record), &record, sizeof(record));
	++batch.count;
	std::memcpy(batch.mapped, &batch.count, sizeof(batch.count)); // the dispatch's x
}

RenderExecutor::~RenderExecutor() {
	m_upscaler_contexts.clear(); // each feature releases its upscaler's resources
	for (const auto& context: m_fsr_contexts) {
		Fsr3::Destroy(context.fsr);
	}
	const auto device = m_context.GetGraphics().device;
	if (m_mesh_batch_pipeline) {
		device.destroyPipeline(m_mesh_batch_pipeline, nullptr);
	}
	if (m_mesh_batch_layout) {
		device.destroyPipelineLayout(m_mesh_batch_layout, nullptr);
	}
	if (m_mesh_arguments_pipeline) {
		device.destroyPipeline(m_mesh_arguments_pipeline, nullptr);
	}
	if (m_mesh_arguments_layout) {
		device.destroyPipelineLayout(m_mesh_arguments_layout, nullptr);
	}
	device.destroyPipeline(m_pssr_pipeline, nullptr);
	device.destroyPipelineLayout(m_pssr_pipeline_layout, nullptr);
	device.destroyDescriptorSetLayout(m_pssr_descriptor_layout, nullptr);
}

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298: stop where the address space is free. Reserved pages inside count (they read as zero): Marvel's
		// Wolverine's geometry pool with the game's lazy commit maps its 2 MiB pages as blocks land.
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize(), true);
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	state.ps_input_info = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> target_export_mapping;
	(void)color_output_mask; // DrawPixelExports derives the same mask
	(void)DrawPixelExports(ctx, sh_ctx, target_export_mapping);
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	buffer.GetContext().GetImageHeap().BeginDraw();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info, state.indirect_offsets,
	    state.indirect_draw);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	RefreshShaders(buffer, draw, color_output_mask, state);
	// Vertex programs beyond the host mesh limits are not compiled.
	if (!state.programs.vertex[0]) {
		NoteSkippedDraw("vertex program beyond the mesh limits", buffer);
		return false;
	}
	// An empty screen scissor rasterizes nothing in any viewport, so only vertex-stage buffer
	// writes would remain (PPSA02110 ends mip reductions with a 0x0 pass past the last level).
	if (const auto& vp = buffer.GetRegisters().GetScreenViewport();
	    vp.screen_scissor_right <= vp.screen_scissor_left || vp.screen_scissor_bottom <= vp.screen_scissor_top) {
		bool vertex_writes = false;
		for (uint32_t i = 0; i < state.programs.VertexStageCount(); i++) {
			vertex_writes = vertex_writes || HasShaderBufferWrites(state.vertex_info[i].stage);
		}
		if (!vertex_writes) {
			NoteSkippedDraw("empty screen scissor", buffer);
			return false;
		}
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				mrt_mask |= 1u << output.location;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	m_targets.color_count = 0;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, m_targets.colors[m_targets.color_count],
			                         render_target_slice_offset, slot);
			if (m_targets.colors[m_targets.color_count].image_id) {
				m_targets.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, m_targets.depth);
	// TODO(PPSA09243): Vulkan passes take one sample count. Where PS5 attachments mix counts,
	// keep those matching the raster count instead of resolving multisampled shadows.
	const auto colors       = std::span {m_targets.colors, m_targets.color_count};
	const auto pass_samples = m_targets.depth.image_id ? m_targets.depth.desc.info.samples
	                                                    : m_targets.colors[0].desc.info.samples;
	if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
		    return color.desc.info.samples != pass_samples;
	    })) {
		const auto raster =
		    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
		const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
			return color.desc.info.samples != raster;
		});
		m_targets.color_count -= static_cast<uint32_t>(dropped.size());
		if (m_targets.depth.image_id && m_targets.depth.desc.info.samples != raster) {
			m_targets.depth = {};
		}
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(
			    "Warning: dropping attachments whose sample count differs from the raster.\n");
		}
	}

	if (m_targets.color_count == 0 && !m_targets.depth.image_id && !state.ps_active) {
		// Without targets its vertex stages' stores would be all it does: are there any?
		int vertex_stores = 0;
		for (const auto& info: state.vertex_info) {
			if (info.stage) {
				const auto& program = *info.stage.program;
				vertex_stores += static_cast<int>(program.has_address_writes);
				for (const auto& resource: program.info.buffers) vertex_stores += static_cast<int>(resource.written);
				for (const auto& resource: program.info.images) vertex_stores += static_cast<int>(resource.written);
			}
		}
		NoteSkippedDraw("no targets, no pixel shader", buffer, vertex_stores);
		LogFramebufferSkip(draw.Name(), m_targets.depth, buffer, draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(Cmd                          vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF((prepared.buffers[i] == nullptr) && !KYTY_UPSTREAM_PATH("e827c92a2"));
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(Cmd vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
                                 const DrawRenderState& state, const DrawTargets& targets,
                                 uint32_t index_type_and_size, const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	static const RenderColorInfo none {};

	const auto& color = targets.color_count != 0 ? targets.colors[0] : none;
	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), color, targets.depth, buffer, state.ps_input_info,
		                   draw.index_count, 0);
	}
	LogDrawInputState(buffer, color, state.vertex_info[0], index_type_and_size, draw.index_count,
	                  index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, Cmd vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count,
				               static_cast<uint32_t>(emit.vertex_offset), emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count,
					               i + static_cast<uint32_t>(emit.vertex_offset),
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto& depth = m_targets.depth;
	auto& color = m_targets.colors[0];
	EXIT_NOT_IMPLEMENTED(!m_context.GetGraphics().maintenance8_enabled ||
	                     !Prospero::IsRectList(ucfg.GetPrimType()) ||
	                     draw.IsIndexed() || draw.index_count != 3 || draw.instance_count != 1 ||
	                     emit.vertex_offset != 0 || emit.first_instance != 0 ||
	                     m_targets.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	                     color.desc.info.samples != 1 ||
	                     depth.desc.info.extent.width != color.Extent().width ||
	                     depth.desc.info.extent.height != color.Extent().height ||
	                     depth.desc.view_info.layer_count != 1 ||
	                     color.desc.view_info.layer_count != 1 ||
	                     hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	                     state.ps_input_info.ps_depth_export_enable ||
	                     state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		m_context.GetCommandScheduler().EndRendering();
		const auto command = buffer.Recorder();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

std::atomic<uint64_t> g_caches_touched {0};

// Whether a stage's stores could change what a later record of its packet reads.
static bool StageWrites(const ShaderStageRuntime& stage) {
	const auto& program = *stage.program;
	return program.has_address_writes ||
	       std::ranges::any_of(program.info.buffers, [](const auto& resource) { return resource.written; }) ||
	       std::ranges::any_of(program.info.images, [](const auto& resource) { return resource.written; });
}

static bool HasClear(const RenderState& state) {
	const auto clears = [](const RenderAttachment& attachment) {
		return attachment.depth_clear || attachment.stencil_clear;
	};
	return clears(state.depth_stencil_attachment) || std::ranges::any_of(state.color_attachments, clears);
}

// Upstream abc509d33: the task shader's inputs for an indirect mesh draw, Params in gpu_mesh_indirect.task.
struct MeshTaskParams {
	vk::DeviceAddress records        = 0;
	vk::DeviceAddress index_address  = 0;
	uint32_t          stride         = 0;
	uint32_t          index_size     = 0;
	uint32_t          index_limit    = 0;
	uint32_t          primitive_size = 0;
	uint32_t          group_step     = 0;
	uint32_t          max_groups     = 0;
	uint32_t          max_instances  = 0;
	uint32_t          max_total      = 0;
};

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	auto& ucfg = buffer.GetUserConfig();
	if (Fsr3::FrameGen::Active() && state.ps_active && state.ps_input_info.stage && m_targets.color_count > 0 &&
	    m_targets.colors[0].image_id) {
		WolverineHudless::BeforeDraw(buffer, state.ps_input_info.stage, m_targets.colors[0], [&] {
			ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, primitive_restart_enable);
		});
	}
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	// The mesh shader reads its five draw dwords by device address (EmitMeshDrawParameter).
	vk::DeviceAddress mesh_block = 0;
	uint64_t          mesh_tasks = UINT64_MAX; // a GPU-made task command's offset in the stream ring
	bool              mesh_cpu_block = false;  // the draw block is written after the index buffer is taken
	// Upstream abc509d33: a pipeline made for an indirect mesh draw reads its records in a task shader.
	const bool mesh_task = mesh_active && state.vertex_info[0].indirect_draw;
	if (mesh_task) {
		const auto& mesh   = state.vertex_info[0].mesh;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		EXIT_NOT_IMPLEMENTED(emit.indirect.address == 0 || mesh.primitives_per_group == 0 ||
		                     (mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable)) ||
		                     emit.indirect.count > std::min(limits.maxTaskWorkGroupCount[0],
		                                                    limits.maxTaskWorkGroupTotalCount));
	} else if (mesh_active && emit.gpu_arguments != 0) {
		const auto& mesh = state.vertex_info[0].mesh;
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		// The task command and the draw dwords are made on the GPU from the arguments a guest
		// shader wrote there: nothing waits for them here.
		auto& cache                 = m_context.GetBufferCache();
		auto& ring                  = cache.GetUtilityBuffer(MemoryUsage::Stream);
		const auto [mapped, offset] = ring.Map(48, 16);
		EXIT_IF(mapped == nullptr);
		ring.Commit();
		const auto [arguments, arguments_offset] = cache.ObtainBuffer(emit.gpu_arguments, 16, false);
		EXIT_IF(arguments == nullptr);
		const auto&                   limits = m_context.GetGraphics().mesh_shader_properties;
		const MeshArgumentsParameters parameters {
		    .arguments            = arguments->BufferDeviceAddress() + arguments_offset,
		    .output               = ring.BufferDeviceAddress() + offset,
		    .primitive_size       = mesh.InputPrimitiveSize(),
		    .primitive_step       = mesh.InputPrimitiveStep(),
		    .primitives_per_group = mesh.primitives_per_group,
		    .native               = emit.native ? 1u : 0u,
		    .max_groups_x         = limits.maxMeshWorkGroupCount[0],
		    .max_groups_y         = limits.maxMeshWorkGroupCount[1],
		    .max_groups_total     = limits.maxMeshWorkGroupTotalCount,
		};
		// Mapped memory that isn't coherent would need flushing at the submit: those hosts convert draw by draw.
		static const bool batch_on = [] {
			const char* value = std::getenv("KYTY_MESH_BATCH"); // 0: convert draw by draw
			return value == nullptr || value[0] != '0';
		}();
		if (batch_on && ring.IsCoherent()) {
			BatchMeshArguments(buffer, parameters);
		} else {
			m_mesh_arguments_batch.mapped = nullptr;
			buffer.EndRendering();
			const auto        vk_buffer = buffer.Recorder();
			vk::MemoryBarrier before {};
			before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
			before.dstAccessMask = vk::AccessFlagBits::eShaderRead;
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                          vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0,
			                          nullptr, 0, nullptr);
			vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, MeshArgumentsPipeline());
			vk_buffer.pushConstants(m_mesh_arguments_layout, vk::ShaderStageFlagBits::eCompute, 0,
			                        sizeof(parameters), &parameters);
			vk_buffer.dispatch(1, 1, 1);
			vk::MemoryBarrier after {};
			after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
			after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
			                          vk::PipelineStageFlagBits::eDrawIndirect |
			                              vk::PipelineStageFlagBits::eMeshShaderEXT,
			                          {}, 1, &after, 0, nullptr, 0, nullptr);
		}
		mesh_tasks = offset;
		mesh_block = ring.BufferDeviceAddress() + offset + 16u;
	} else if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
		mesh_cpu_block = true;
	}

	uint64_t counted = UINT64_MAX; // a GPU-made draw record's offset in the stream ring
	if (emit.instance_count_address != 0) {
		// The instance count is what a guest shader wrote (NUM_INSTANCES after an indirect multi-draw): the
		// draw's record is made here and the GPU copies the count into it, so nothing waits for that shader.
		auto&       cache = m_context.GetBufferCache();
		auto&       ring  = cache.GetUtilityBuffer(MemoryUsage::Stream);
		const auto  first = static_cast<uint32_t>(emit.vertex_offset);
		const auto  record =
		    draw.IsIndexed() ? std::array<uint32_t, 5> {draw.index_count, 0u, 0u, first, emit.first_instance}
		                      : std::array<uint32_t, 5> {draw.index_count, 0u, first, emit.first_instance, 0u};
		const auto [mapped, offset] = ring.Map(sizeof(record), 16);
		EXIT_IF(mapped == nullptr);
		std::memcpy(mapped, record.data(), sizeof(record));
		ring.Commit();
		const auto [count, count_offset] = cache.ObtainBuffer(emit.instance_count_address, sizeof(uint32_t), false);
		EXIT_IF(count == nullptr);
		buffer.EndRendering();
		const auto        vk_buffer = buffer.Recorder();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {},
		                          1, &before, 0, nullptr, 0, nullptr);
		const vk::BufferCopy copy {count_offset, offset + sizeof(uint32_t), sizeof(uint32_t)};
		vk_buffer.copyBuffer(count->Handle(), ring.Handle(), 1, &copy);
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eDrawIndirect,
		                          {}, 1, &after, 0, nullptr, 0, nullptr);
		counted = offset;
	}

	// Mesh shaders read the original guest indices in whole DWORDs by device address. Their buffer
	// exists before the descriptors bind, so creating it merges no buffer this draw binds.
	uint64_t index_begin = 0;
	uint64_t index_end   = 0;
	if (mesh_active && draw.IsIndexed()) {
		const auto size = static_cast<uint64_t>(draw.index_count) * index_source.guest_element_size;
		index_begin     = index_source.address & ~uint64_t {3};
		index_end       = (index_source.address + size + 3u) & ~uint64_t {3};
		(void)m_context.GetBufferCache().FindBuffer(index_begin, index_end - index_begin);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
		descriptor_stages[stage_count++] = &*bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	PrepareGraphicsBindings(stages, std::span {m_targets.colors, m_targets.color_count});
	PreparedVertexBuffers        vertex_bindings;
	PreparedIndexBuffer          index_binding;
	std::pair<Buffer*, uint64_t> mesh_index_buffer {};
	std::pair<Buffer*, uint64_t> args_buffer {};
	// The draw's buffer acquisitions, the stages' buffers last, before the pipeline and the render targets. A buffer-cache
	// join leaves the bindings taken before it on the old buffer, so they repeat until a pass has no join; at the cap the
	// draw goes on with what it has, as the descriptors of either variant would.
	auto&    buffer_cache = m_context.GetBufferCache();
	uint32_t passes       = 0;
	bool     joined       = false;
	do {
		const auto joins = buffer_cache.Joins();
		if (!mesh_active) {
			LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
			index_binding   = PrepareIndexBuffer(buffer, index_source);
		} else if (draw.IsIndexed()) {
			mesh_index_buffer = buffer_cache.ObtainBuffer(index_begin, index_end - index_begin, false);
		}
		if (emit.indirect.address != 0) {
			const auto size = static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
			                  (draw.IsIndexed() ? sizeof(vk::DrawIndexedIndirectCommand)
			                                    : sizeof(vk::DrawIndirectCommand));
			args_buffer = buffer_cache.ObtainBuffer(emit.indirect.address, size, false);
			EXIT_IF((args_buffer.second & 3u) != 0);
		}
		for (auto* stage: stages) {
			RebindBuffers(*stage);
		}
		joined = buffer_cache.Joins() != joins;
	} while (++passes < BufferCache::kJoinPasses && joined);
	vk::DeviceAddress mesh_indices = 0;
	if (mesh_index_buffer.first != nullptr) {
		mesh_indices = mesh_index_buffer.first->BufferDeviceAddress() + mesh_index_buffer.second +
		               (index_source.address & 3u);
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	auto& pipeline = m_context.GetPipelineCache().GetGraphicsPipeline(
	    std::span {m_targets.colors, m_targets.color_count}, m_targets.depth, vertex_stages, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.programs);
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering =
	    AcquireRenderTargets(buffer, m_targets.colors, m_targets.color_count, m_targets.depth,
	                         feedback_aspects, stages);
	if (mesh_task) {
		// The task shader's parameters: the push dwords that hold the draw block's address hold theirs.
		const auto&          mesh   = state.vertex_info[0].mesh;
		const auto&          limits = m_context.GetGraphics().mesh_shader_properties;
		const MeshTaskParams params {
		    .records        = args_buffer.first->BufferDeviceAddress() + args_buffer.second,
		    .index_address  = mesh_indices,
		    .stride         = emit.indirect.stride,
		    .index_size     = index_source.guest_element_size,
		    .index_limit    = draw.index_count,
		    .primitive_size = mesh.InputPrimitiveSize(),
		    .group_step     = mesh.primitives_per_group * mesh.InputPrimitiveStep(),
		    .max_groups     = limits.maxMeshWorkGroupCount[0],
		    .max_instances  = limits.maxMeshWorkGroupCount[1],
		    .max_total      = limits.maxMeshWorkGroupTotalCount,
		};
		auto& ring                  = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		const auto [mapped, offset] = ring.Map(sizeof(params), 16);
		EXIT_IF(mapped == nullptr);
		std::memcpy(mapped, &params, sizeof(params));
		ring.Commit();
		mesh_block = ring.BufferDeviceAddress() + offset;
		static std::atomic_flag first = ATOMIC_FLAG_INIT;
		if (!first.test_and_set(std::memory_order_relaxed)) {
			LOGF("upstream abc509d33: first indirect mesh draw through the task shader (%u records)\n",
			     emit.indirect.count);
		}
	}
	if (mesh_cpu_block) {
		const uint32_t draw_data[] {draw.index_count, static_cast<uint32_t>(emit.vertex_offset),
		                            index_source.guest_element_size, static_cast<uint32_t>(mesh_indices),
		                            static_cast<uint32_t>(mesh_indices >> 32u)};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		auto& ring                  = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		const auto [mapped, offset] = ring.Map(sizeof(draw_data), 16);
		EXIT_IF(mapped == nullptr);
		std::memcpy(mapped, draw_data, sizeof(draw_data));
		ring.Commit();
		mesh_block = ring.BufferDeviceAddress() + offset;
	}

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Recorder();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (!mesh_active) {
		CommitIndexBuffer(vk_buffer, index_binding);
	}
	using ShaderRecompiler::IR::PushData;
	const auto draw_stages =
	    vk::ShaderStageFlagBits::eFragment |
	    (mesh_active ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex) |
	    (mesh_task ? vk::ShaderStageFlags {vk::ShaderStageFlagBits::eTaskEXT} : vk::ShaderStageFlags {});
	const auto draw_dwords =
	    PushData::GraphicsDrawDwords(mesh_active, state.ps_input_info.object_primitive_id);
	if (draw_dwords != 0) {
		std::array<uint32_t, PushData::MeshDrawDwordCount> draw_data {};
		if (mesh_active) {
			draw_data[0] = static_cast<uint32_t>(mesh_block);
			draw_data[1] = static_cast<uint32_t>(mesh_block >> 32u);
		} else {
			draw_data[0] = buffer.GetUserConfig().GetObjectId();
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, draw_stages, 0,
		                        draw_dwords * sizeof(uint32_t), draw_data.data());
	}
	if (const auto planes = vertex_stages.back().user_clip_planes; planes != 0) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, draw_stages,
		                        PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(planes) * 4u * sizeof(uint32_t),
		                        buffer.GetRegisters().GetUserClipPlanes().data());
	}

	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), m_targets.depth,
	                           rendering);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);

	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		// As for dispatches, buffer writes wait for earlier accesses to the memory.
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteHazardBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	m_context.GetCommandScheduler().GetDynamicState().BindGraphicsPipeline(vk_buffer, pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	auto&         occlusion_queries = m_context.GetOcclusionQueries();
	const int32_t occlusion_query   = occlusion_queries.Begin(vk_buffer);
	if (mesh_task) {
		vk_buffer.drawMeshTasksEXT(emit.indirect.count, 1, 1);
	} else if (mesh_active && mesh_tasks != UINT64_MAX) {
		vk_buffer.drawMeshTasksIndirectEXT(
		    m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream).Handle(), mesh_tasks, 1,
		    sizeof(vk::DrawMeshTasksIndirectCommandEXT));
	} else if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else if (counted != UINT64_MAX) {
		const auto& ring = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		if (draw.IsIndexed()) {
			vk_buffer.drawIndexedIndirect(ring.Handle(), counted, 1, sizeof(vk::DrawIndexedIndirectCommand));
		} else {
			vk_buffer.drawIndirect(ring.Handle(), counted, 1, sizeof(vk::DrawIndirectCommand));
		}
	} else if (args_buffer.first == nullptr) {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	} else if (draw.IsIndexed()) {
		vk_buffer.drawIndexedIndirect(args_buffer.first->Handle(), args_buffer.second,
		                              emit.indirect.count, emit.indirect.stride);
	} else {
		vk_buffer.drawIndirect(args_buffer.first->Handle(), args_buffer.second,
		                       emit.indirect.count, emit.indirect.stride);
	}
	occlusion_queries.End(vk_buffer, occlusion_query);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	// Stores that aren't buffer writes make no pass break, and could still reach a later draw's arguments.
	if (m_mesh_arguments_batch.mapped != nullptr &&
	    std::ranges::any_of(stages, [](const PreparedBindings* stage) { return StageWrites(*stage->runtime); })) {
		m_mesh_arguments_batch.mapped = nullptr;
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
	// A clear changes where the pass ends: AcquireRenderTargets consumes a pending one at the first draw.
	if (mesh_active && !mesh_task && mesh_tasks == UINT64_MAX && !shader_write_stages && !feedback_aspects &&
	    !primitive_restart_enable && index_source.guest_element_size != 1 && !HasClear(rendering) &&
	    !graphics_debug_dump_enabled() &&
	    std::ranges::none_of(stages, [](const PreparedBindings* stage) { return StageWrites(*stage->runtime); })) {
		m_later_records = {.valid       = true,
		                   .tick        = m_context.GetCommandScheduler().CurrentTick(),
		                   .rendering   = rendering,
		                   .layout      = pipeline.pipeline_layout,
		                   .stages      = draw_stages,
		                   .draw_dwords = draw_dwords,
		                   .index_size  = index_source.guest_element_size,
		                   .uses_dma    = std::ranges::any_of(stages,
		                                                      [](const PreparedBindings* stage) {
			                                                      return stage->runtime->program->info.uses_dma;
		                                                      }),
		                   .state       = &state,
		                   .caches_touched = m_draw_caches_touched,
		                   .stage       = stages[0],
		                   .set_layout  = pipeline.descriptor_set_layout,
		                   .push_descriptors = pipeline.uses_push_descriptors};
	}
}

// Vulkan offsets for a prepared draw, or nullopt when its indirect records need the CPU: mesh
// and quad-list draws for counts, tessellation draws for the index/SGPR prolog. Indirect records
// carry their own offsets.
static std::optional<DrawEmitInfo> ResolveDrawEmit(const HW::UserConfig&        ucfg,
                                                   const ShaderVertexInputInfo& vs,
                                                   DrawOffsetSource             source,
                                                   int32_t                      first_vertex,
                                                   uint32_t                     first_instance,
                                                   const DrawIndirectArgs&      indirect,
                                                   uint64_t                     instance_count_address = 0) {
	if ((indirect.address != 0 || source == DrawOffsetSource::IndirectArgs) &&
	    ucfg.GetIndexOffset() != 0) {
		static std::once_flag warning_once;
		std::call_once(warning_once, [&] {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU: indirect draw ignores GE_INDX_OFFSET=0x{:08x}; geometry may be incorrect.\n",
			    ucfg.GetIndexOffset()));
		});
	}
	if (instance_count_address != 0) {
		if (vs.stage.program->stage == ShaderType::Mesh ||
		    ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy) {
			return std::nullopt;
		}
		auto emit = ResolveDrawEmit(ucfg, vs, source, first_vertex, first_instance, indirect);
		if (emit) emit->instance_count_address = instance_count_address;
		return emit;
	}
	if (indirect.address != 0) {
		if (!vs.indirect_draw ||
		    ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy ||
		    vs.indirect_offsets != indirect.offsets) {
			return std::nullopt;
		}
		return DrawEmitInfo {.indirect = indirect};
	}
	if (source == DrawOffsetSource::IndirectArgs) {
		// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
		return vs.has_dynamic_vertex_fetch ? DrawEmitInfo {first_vertex, first_instance} : DrawEmitInfo {};
	}
	const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), vs);
	return DrawEmitInfo {vertex_offset + first_vertex, instance_offset};
}

bool RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid() && !KYTY_UPSTREAM_PATH("45b35fafb"));
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	m_later_records.valid = false;
	m_draw_caches_touched = g_caches_touched.load(std::memory_order_relaxed);
	if (args.index_count == 0 ||
	    (args.instance_count == 0 && args.indirect.address == 0 && args.instance_count_address == 0)) {
		return true;
	}
	// A depth-to-color copy needs its instance count here.
	if (args.instance_count_address != 0 && buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	auto& state = TakeDrawRenderState(args.indirect.offsets, args.indirect.address != 0);
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, m_targets, args.index_type_and_size, args.index_addr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source, args.base_vertex,
	                                  args.first_instance, args.indirect, args.instance_count_address);
	if (emit) {
		ExecutePreparedDraw(submit_id, buffer, draw, state, topology, *emit, index_source,
		                    primitive_restart);
	}
	ResetBindings();
	return emit.has_value();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid() && !KYTY_UPSTREAM_PATH("45b35fafb"));
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	m_later_records.valid = false;
	m_draw_caches_touched = g_caches_touched.load(std::memory_order_relaxed);
	if (args.indirect.address == 0 && args.gpu_arguments == 0 &&
	    (args.vertex_count == 0 || (args.instance_count == 0 && args.instance_count_address == 0))) {
		return true;
	}
	if (args.instance_count_address != 0 && buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return true;
	}
	auto& state = TakeDrawRenderState(args.indirect.offsets, args.indirect.address != 0);
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return true;
	}
	if (args.gpu_arguments != 0 && state.vertex_info[0].stage.program->stage != ShaderType::Mesh) {
		// Only a mesh draw makes its counts on the GPU here; others the GPU path declined (a legacy
		// quad list) need them on the CPU: they are read.
		ResetBindings();
		uint32_t words[4] {};
		m_context.GetBufferCache().ReadGuestBytes(args.gpu_arguments, words, sizeof(words));
		auto read           = args;
		read.vertex_count   = words[0];
		read.instance_count = words[1];
		read.first_vertex   = words[2];
		read.first_instance = words[3];
		read.gpu_arguments  = 0;
		return DrawAuto(submit_id, buffer, read);
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		NoteSkippedDraw("rect list, no vertex exports, pixel inputs", buffer);
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, m_targets, 0, nullptr);
	// Arguments a shader wrote reach here only for a mesh draw (the GPU path takes the others); its
	// shader takes no first vertex or instance from them unless its fetch is rewritten.
	auto emit =
	    args.gpu_arguments != 0
	        ? std::optional<DrawEmitInfo> {DrawEmitInfo {.gpu_arguments = args.gpu_arguments,
	                                                     .native = !state.vertex_info[0].has_dynamic_vertex_fetch}}
	        : ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                          static_cast<int32_t>(args.first_vertex), args.first_instance, args.indirect,
	                          args.instance_count_address);
	if (!emit && args.mesh_gpu_arguments != 0 && state.vertex_info[0].stage.program->stage == ShaderType::Mesh) {
		emit = DrawEmitInfo {.gpu_arguments = args.mesh_gpu_arguments,
		                     .native        = !state.vertex_info[0].has_dynamic_vertex_fetch};
	}
	if (emit) {
		ExecutePreparedDraw(submit_id, buffer, draw, state, topology, *emit, {}, false);
	}
	ResetBindings();
	return emit.has_value();
}

// No packet runs between two records, and the registers are the same: a full draw would prepare and bind what's bound.
// Only what's the record's own is left: its index range, its block and the draw. Anything those end or restart (an
// upload ends the pass, a stream wrap submits) sends it through the full draw, and its block goes unused.
bool RenderExecutor::DrawLaterMeshRecord(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args,
                                         bool indexed, uint32_t draw_index_slot, uint32_t draw_index) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.PopPendingOperations();
	if (indexed) {
		buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id, args.index_count, 0, 1,
		                    args.instance_count, reinterpret_cast<uint64_t>(args.index_addr));
	} else {
		buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id, args.index_count, 0,
		                    static_cast<uint32_t>(args.base_vertex), args.instance_count, args.first_instance);
	}

	Common::LockGuard lock(m_context.GetMutex());
	const auto&       later = m_later_records;
	if (!later.valid || later.caches_touched != g_caches_touched.load(std::memory_order_relaxed)) {
		return false;
	}
	if (args.index_count == 0 || args.instance_count == 0) {
		return true;
	}
	const auto& state = *later.state;
	const auto& mesh  = state.vertex_info[0].mesh;
	const auto  primitives = mesh.InputPrimitiveCount(args.index_count);
	if (primitives == 0) {
		return true;
	}
	const uint32_t mesh_groups = (primitives - 1u) / mesh.primitives_per_group + 1u;
	const auto&    limits      = m_context.GetGraphics().mesh_shader_properties;
	if (mesh_groups > limits.maxMeshWorkGroupCount[0] || args.instance_count > limits.maxMeshWorkGroupCount[1] ||
	    static_cast<uint64_t>(mesh_groups) * args.instance_count > limits.maxMeshWorkGroupTotalCount) {
		return false;
	}
	const auto emit = ResolveDrawEmit(buffer.GetUserConfig(), state.vertex_info[0], DrawOffsetSource::IndirectArgs,
	                                  args.base_vertex, args.first_instance, {});
	const auto        index_address = reinterpret_cast<uint64_t>(args.index_addr);
	auto&             cache         = m_context.GetBufferCache();
	vk::DeviceAddress mesh_indices  = 0;
	if (indexed) {
		const auto size  = static_cast<uint64_t>(args.index_count) * later.index_size;
		const auto begin = index_address & ~uint64_t {3};
		const auto end   = (index_address + size + 3u) & ~uint64_t {3};
		const auto joins = cache.Joins();
		const auto [indices, offset] = cache.ObtainBuffer(begin, end - begin, false);
		if (indices == nullptr || cache.Joins() != joins) {
			return false; // a join moved buffers the recorded descriptors bind: the full draw path takes it
		}
		mesh_indices = indices->BufferDeviceAddress() + offset + (index_address & 3u);
	}
	const uint32_t draw_data[] {args.index_count, static_cast<uint32_t>(emit->vertex_offset),
	                            later.index_size, static_cast<uint32_t>(mesh_indices),
	                            static_cast<uint32_t>(mesh_indices >> 32u)};
	static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
	auto& ring                  = cache.GetUtilityBuffer(MemoryUsage::Stream);
	const auto [mapped, offset] = ring.Map(sizeof(draw_data), 16);
	EXIT_IF(mapped == nullptr);
	std::memcpy(mapped, draw_data, sizeof(draw_data));
	ring.Commit();
	const vk::DeviceAddress mesh_block = ring.BufferDeviceAddress() + offset;
	if (later.uses_dma) {
		m_context.PrepareBda();
	}
	uint32_t   index_dword = 0;
	const bool new_index   = draw_index_slot != UINT32_MAX &&
	                         UploadDrawIndex(m_context, *later.stage, draw_index_slot, draw_index, index_dword);
	if (scheduler.CurrentTick() != later.tick || !buffer.RenderingWith(later.rendering)) {
		return false;
	}

	if (new_index) {
		CommitDrawIndex(buffer, index_dword, draw_index);
	}
	const auto vk_buffer = buffer.Recorder();
	if (later.draw_dwords != 0) {
		std::array<uint32_t, ShaderRecompiler::IR::PushData::MeshDrawDwordCount> block {};
		block[0] = static_cast<uint32_t>(mesh_block);
		block[1] = static_cast<uint32_t>(mesh_block >> 32u);
		vk_buffer.pushConstants(later.layout, later.stages, 0, later.draw_dwords * sizeof(uint32_t), block.data());
	}
	auto&         occlusion_queries = m_context.GetOcclusionQueries();
	const int32_t occlusion_query   = occlusion_queries.Begin(vk_buffer);
	vk_buffer.drawMeshTasksEXT(mesh_groups, args.instance_count, 1);
	occlusion_queries.End(vk_buffer, occlusion_query);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, args.instance_count);
	return true;
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if ((!src.image_id || !dst.image_id) && !KYTY_UPSTREAM_PATH("e827c92a2")) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

} // namespace Libs::Graphics
