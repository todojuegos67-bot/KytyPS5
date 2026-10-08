#include "graphics/host_gpu/renderer/renderDraw.h"
#include "live-trace-gpu.h"

#include "common/assert.h"
#include "common/common.h"
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
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderReadObserver.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "live-census.h"
#include "live-counters.h"
#include "frame-capture.h"
#include "frame-gen.h"
#include "native-preparation-state.h"
#include "slow-log.h"
#include "xpr-capture.h"
#include "speculation-state.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-draw-packet.h"
#include "vulkan-recording.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

extern "C" {
// 0: a draw run reads no buffer range that any image overlaps; 1: sampled
// images whose contents still come from the CPU may overlap its read ranges.
volatile std::atomic_uint32_t kyty_local_draw_run_ranges_mode {0};
// KYTY_DRAW_PACKETS: the ordinary DrawIndex/DrawAuto draws are recorded as packets too (their
// debug phases are CPU-side notes only), instead of ~20 separately recorded Vulkan calls each.
volatile std::atomic_uint32_t kyty_local_draw_packets_mode {0};
}

// Draw state in reusable preparation storage when that switch is on.
class NativeDrawState {
	const bool                                enabled = NativePreparationScratchEnabled();
	NativePreparationScratch<DrawRenderState> storage {enabled};

public:
	NativeDrawState() {
		if (!enabled) {
			return;
		}
		auto& state      = storage.Get();
		state.depth_info = {};
		// ResolveRenderColorTarget resets each active slot before every path,
		// including slot zero for depth-only draws. Remaining array slots are
		// never read beyond color_count.
		if (!kyty_local_preparation_trim_mode.load(std::memory_order_relaxed)) {
			std::fill(std::begin(state.color_info), std::end(state.color_info), RenderColorInfo {});
		}
		state.color_count = 0;
		state.ps_active   = true;
		state.programs    = {};
	}
	DrawRenderState& Get() { return storage.Get(); }
};

int32_t ResolveVertexOffset(uint32_t index_offset, const ShaderVertexInputInfo& vs_input_info) {
	if (index_offset != 0 || !vs_input_info.fetch_embedded) {
		return static_cast<int32_t>(index_offset);
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = vs_input_info.stage.resources;
	if (program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			return static_cast<int32_t>(resources.user_data[index]);
		}
	}

	return 0;
}

uint32_t ResolveInstanceOffset(const ShaderVertexInputInfo& vs_input_info) {
	if (!vs_input_info.fetch_embedded) {
		return 0;
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = vs_input_info.stage.resources;
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			return resources.user_data[index];
		}
	}

	return 0;
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

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
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
	    log_id, buffer.GetContext().FrameNumber(), draw_name, RenderColorTypeName(color),
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
	     log_id, buffer.GetContext().FrameNumber(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u attr_num=%d\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index, b.attr_num);

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

				for (int ai = 0; ai < b.attr_num; ai++) {
					const auto  res_index = b.attr_indices[ai];
					const auto& r         = vs_input_info.resources[res_index];
					const auto& rd        = vs_input_info.resources_dst[res_index];
					const auto  offset    = b.attr_offsets[ai];
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

		for (int ai = 0; ai < b.attr_num; ai++) {
			const auto  res_index = b.attr_indices[ai];
			const auto& r         = vs_input_info.resources[res_index];
			const auto& rd        = vs_input_info.resources_dst[res_index];
			LOGF("DrawInputState[%u]: attr[%d] res=%d offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, res_index, b.attr_offsets[ai], rd.register_start, rd.registers_num,
			     rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

template<typename Buffer, typename Color, typename Depth, typename Dispatch>
static void SetGraphicsDynamicParamsImpl(const Buffer& buffer, vk::CommandBuffer vk_buffer,
                                         bool indexed_viewports, const Color* colors,
                                         uint32_t color_count, const Depth& depth,
                                         const Dispatch& dispatch) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(colors == nullptr);
	const auto& ctx = buffer.GetRegisters();

	const auto&  vp = ctx.GetScreenViewport();
	vk::Extent2D framebuffer_extent {};
	if (color_count > 0 && colors[0].image_id) {
		framebuffer_extent = colors[0].Extent();
	} else if (depth.image_id) {
		framebuffer_extent = {depth.desc.info.extent.width, depth.desc.info.extent.height};
	} else {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		framebuffer_extent = {limits.maxFramebufferWidth, limits.maxFramebufferHeight};
	}

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
	vk_buffer.setViewportWithCount(viewport_count, viewports.data(), dispatch);
	vk_buffer.setScissorWithCount(viewport_count, scissors.data(), dispatch);

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
	vk_buffer.setLineWidth(line_width, dispatch);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	vk_buffer.setBlendConstants(blend_constants.data(), dispatch);
	vk_buffer.setDepthTestEnable(depth.depth_test_enable ? VK_TRUE : VK_FALSE, dispatch);
	vk_buffer.setDepthWriteEnable(depth.depth_write_enable ? VK_TRUE : VK_FALSE, dispatch);
	vk_buffer.setDepthCompareOp(depth.depth_compare_op, dispatch);

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	vk_buffer.setDepthBiasEnable(depth_bias_enable ? VK_TRUE : VK_FALSE, dispatch);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor, dispatch);
	}

	if (depth.stencil_test_enable) {
		vk_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
		                                depth.stencil_dynamic_front.compareMask, dispatch);
		vk_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
		                                depth.stencil_dynamic_back.compareMask, dispatch);
		vk_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
		                              depth.stencil_dynamic_front.writeMask, dispatch);
		vk_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eBack,
		                              depth.stencil_dynamic_back.writeMask, dispatch);
		vk_buffer.setStencilReference(vk::StencilFaceFlagBits::eFront,
		                              depth.stencil_dynamic_front.reference, dispatch);
		vk_buffer.setStencilReference(vk::StencilFaceFlagBits::eBack,
		                              depth.stencil_dynamic_back.reference, dispatch);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	// Color-control operation selects special color-buffer paths, not the normal component write
	// mask. Attachment availability therefore follows the target write mask.
	for (uint32_t i = 0; i < color_count; i++) {
		enable[i] = render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot) != 0
		                ? VK_TRUE
		                : VK_FALSE;
	}
	if (color_count != 0) {
		vk_buffer.setColorWriteEnableEXT(color_count, enable, dispatch);
	}
#endif
}

static bool UsesIndexedViewports(const ShaderVertexInputInfo& input) {
	const auto& outputs = input.stage.program->info.outputs;
	return std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	});
}

void RenderExecutor::CommitGraphicsState(CommandBuffer& buffer, const ShaderVertexInputInfo& input,
                                         const RenderColorInfo* colors, uint32_t color_count,
                                         const RenderDepthInfo& depth, vk::Pipeline pipeline,
                                         vk::ImageAspectFlags feedback_aspects) {
	const auto vk_buffer = buffer.Handle();
	// The direct path binds the pipeline, the dynamic state and (just before, in
	// CommitIndexBuffer) the index buffer: a native XPR draw after it must bind its own again.
	buffer.InvalidateGraphicsState();
	SetGraphicsDynamicParamsImpl(buffer, vk_buffer, UsesIndexedViewports(input), colors,
	                             color_count, depth, VULKAN_HPP_DEFAULT_DISPATCHER);
	if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
		vk_buffer.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
namespace LocalDrawRecording {
static void Replay(std::span<const LocalVulkanRecording::Segment> segments,
                   const vk::detail::DispatchLoaderDynamic& dispatch) {
    const auto& draw = *static_cast<const Draw*>(segments[0].data);
    const auto command = draw.command;
    const auto vertex_count = static_cast<uint32_t>(segments[1].size / sizeof(vk::Buffer));
    if (vertex_count)
        command.bindVertexBuffers(0, vertex_count,
            static_cast<const vk::Buffer*>(segments[1].data),
            static_cast<const vk::DeviceSize*>(segments[2].data), dispatch);
    if (draw.index_buffer)
        command.bindIndexBuffer(draw.index_buffer, draw.index_offset, draw.index_type, dispatch);

    // Reuse is valid only within the same recording generation and pipeline.
    // The queue invalidates the generation on command-buffer lifetime APIs and
    // raw dynamic-state/pipeline commands. Padding differences only cause a miss.
    struct Cache {
        uint64_t epoch = UINT64_MAX;
        vk::CommandBuffer command;
        vk::Pipeline pipeline;
        DynamicState state;
    };
    static thread_local Cache cache;
    const auto epoch = LocalVulkanRecording::StateEpoch();
    const bool reuse = epoch != UINT64_MAX && cache.epoch == epoch &&
        cache.command == command && cache.pipeline == draw.pipeline &&
        std::memcmp(&cache.state, &draw.state, sizeof(DynamicState)) == 0;
    if (!reuse) {
        const auto& state = draw.state;
        SetGraphicsDynamicParamsImpl(state.buffer, command, state.indexed_viewports,
                                      state.colors, state.color_count, state.depth, dispatch);
        if (state.feedback_enabled)
            command.setAttachmentFeedbackLoopEnableEXT(state.feedback_aspects, dispatch);
        command.bindPipeline(vk::PipelineBindPoint::eGraphics, draw.pipeline, dispatch);
        cache = {epoch, command, draw.pipeline, state};
    }

    const auto* commands = static_cast<const vk::DrawIndexedIndirectCommand*>(segments[3].data);
    for (size_t i = 0; i < segments[3].size / sizeof(*commands); ++i) {
        const auto& item = commands[i];
        if (draw.indexed)
            command.drawIndexed(item.indexCount, item.instanceCount, item.firstIndex,
                                item.vertexOffset, item.firstInstance, dispatch);
        else command.draw(item.indexCount, item.instanceCount, item.firstIndex,
                          item.firstInstance, dispatch);
    }
}
void Record(const Draw& draw, std::span<const vk::Buffer> vertex_buffers,
            std::span<const vk::DeviceSize> vertex_offsets,
            std::span<const vk::DrawIndexedIndirectCommand> commands) {
    static_assert(std::is_trivially_copyable_v<Draw>);
    EXIT_IF(vertex_buffers.size() != vertex_offsets.size());
    for (auto buffer : vertex_buffers) EXIT_IF(!buffer);
    const LocalVulkanRecording::Segment segments[] {
        {&draw, sizeof(draw)}, {vertex_buffers.data(), vertex_buffers.size_bytes()},
        {vertex_offsets.data(), vertex_offsets.size_bytes()},
        {commands.data(), commands.size_bytes()}
    };
    if (!LocalVulkanRecording::EnqueuePacket(Replay, segments))
        LocalVulkanRecording::ReplayInline(Replay, segments);
}
static DynamicState Capture(const CommandBuffer& buffer, const ShaderVertexInputInfo& input,
                            const RenderColorInfo* colors, uint32_t color_count,
                            const RenderDepthInfo& depth, bool feedback,
                            vk::ImageAspectFlags aspects) {
    DynamicState state {};
    const auto& ctx = buffer.GetRegisters();
    state.buffer.registers = {ctx.GetScreenViewport(), ctx.GetClipControl(),
        ctx.GetScanModeControl(), ctx.GetModeControl(), ctx.GetPolyOffset(),
        ctx.GetBlendColor(), ctx.GetLineWidth(), ctx.GetRenderTargetMask()};
    const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
    state.buffer.graphics.properties.limits = {limits.maxFramebufferWidth, limits.maxFramebufferHeight,
        {limits.maxViewportDimensions[0], limits.maxViewportDimensions[1]}};
    state.color_count = color_count;
    for (uint32_t i = 0; i < color_count; ++i)
        state.colors[i] = {colors[i].Extent(), colors[i].target_slot, bool(colors[i].image_id)};
    state.depth.desc.info.extent = {depth.desc.info.extent.width, depth.desc.info.extent.height};
    state.depth.desc.view_info.format = depth.desc.view_info.format;
    state.depth.stencil_dynamic_front = depth.stencil_dynamic_front;
    state.depth.stencil_dynamic_back = depth.stencil_dynamic_back;
    state.depth.depth_compare_op = depth.depth_compare_op;
    state.depth.image_id = bool(depth.image_id);
    state.depth.depth_test_enable = depth.depth_test_enable;
    state.depth.depth_write_enable = depth.depth_write_enable;
    state.depth.stencil_test_enable = depth.stencil_test_enable;
    state.indexed_viewports = UsesIndexedViewports(input);
    state.feedback_enabled = feedback;
    state.feedback_aspects = aspects;
    return state;
}
} // namespace LocalDrawRecording
#endif

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return ShaderAddressValid(vs.es_regs.data_addr);
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 const std::optional<PreparedBindings>& pixel) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = color_count;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		SetVulkanObjectNameF(m_context.GetGraphics().device, image.backing.image,
		                     "Kyty.MRT{}.Image[guest=0x{:016x} size=0x{:x} format={}]",
		                     target.target_slot, image.info.data.address, image.info.data.size,
		                     static_cast<uint32_t>(image.info.pixel_format));
		SetVulkanObjectNameF(m_context.GetGraphics().device, image_view,
		                     "Kyty.MRT{}.View[guest=0x{:016x} mip={} layer={}+{}]",
		                     target.target_slot, image.info.data.address,
		                     target.desc.view_info.base_level, target.desc.view_info.base_layer,
		                     target.desc.view_info.layer_count);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		if (attachment_samples == 0) {
			attachment_samples = target.desc.info.samples;
		} else if (attachment_samples != target.desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, target.desc.info.samples);
		}
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		auto& attachment        = state.color_attachments[i];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		uint32_t   htile_fill       = 0;
		bool       htile_fill_known = false;
		const bool meta_cleared =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer,
		                        &htile_fill, &htile_fill_known);
		const bool stencil_compressed = depth.desc.info.metadata.stencil_compressed;
		const bool depth_uniform =
		    meta_cleared && htile_fill_known && !htile_fill_clears_depth(htile_fill) &&
		    htile_fill_depth_uniform(htile_fill, stencil_compressed);
		const bool depth_meta_clear =
		    meta_cleared &&
		    (!htile_fill_known || htile_fill_clears_depth(htile_fill) || depth_uniform);
		depth.stencil_meta_clear_enable = meta_cleared && htile_fill_known && stencil_compressed &&
		                                  htile_fill_clears_stencil(htile_fill);
		if (depth_uniform) {
			depth.depth_clear_value = htile_fill_depth_value(htile_fill, stencil_compressed);
		}
		depth.depth_load_clear_enable = depth.depth_clear_enable || depth_meta_clear;
		if (meta_cleared &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		SetVulkanObjectNameF(m_context.GetGraphics().device, image.backing.image,
		                     "Kyty.DepthTarget.Image[guest=0x{:016x} size=0x{:x} format={}]",
		                     image.info.data.address, image.info.data.size,
		                     static_cast<uint32_t>(image.info.pixel_format));
		SetVulkanObjectNameF(m_context.GetGraphics().device, image_view,
		                     "Kyty.DepthTarget.View[guest=0x{:016x} layer={}+{}]",
		                     image.info.data.address, depth.desc.view_info.base_layer,
		                     depth.desc.view_info.layer_count);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
		const bool feedback = depth.depth_write_enable && pixel &&
		    std::ranges::any_of(pixel->images, [&](const TextureBinding& binding) {
			    if (binding.image_id != depth.image_id ||
			        binding.desc.type != TextureCache::BindingType::Texture) {
				    return false;
			    }
			    const auto native =
			        std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
			    EXIT_IF(native == image.views.end());
			    const auto& sampled = native->info;
			    const auto& target = depth.desc.view_info;
			    return (sampled.aspect & vk::ImageAspectFlagBits::eDepth) &&
			           ImageRangeOverlaps(sampled.base_level, sampled.level_count,
			                              target.base_level, target.level_count) &&
			           ImageRangeOverlaps(sampled.base_layer, sampled.layer_count,
			                              target.base_layer, target.layer_count);
		    });
		if (feedback && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			EXIT("depth attachment feedback loop is not supported by the host\n");
		}
		auto layout = feedback ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
		                       : depth_attachment_layout(depth);
		auto writable = depth.AttachmentWriteAspects() & vk::ImageAspectFlagBits::eStencil;
		if (depth.depth_write_enable) {
			writable |= vk::ImageAspectFlagBits::eDepth;
		}
		const auto pixel_writable = feedback ? writable & ~vk::ImageAspectFlagBits::eDepth : writable;
		if (static_cast<bool>(image.binding.pixel_sampled_aspects & pixel_writable) ||
		    static_cast<bool>(image.binding.other_sampled_aspects & writable)) {
			layout = vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		              vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		// A target shaders sampled in this layout keeps the shader reads in its access while no aspect they sample
		// is written as an attachment (no hazard between the two uses): draws alternating between them (the decal
		// pass: a stencil mark, then a decal sampling depth) need no barrier, nor a render pass break, each.
		if (const auto& current = image.backing.state;
		    image.backing.subresource_states.empty() && current.layout == layout &&
		    static_cast<bool>(current.access_mask & vk::AccessFlagBits2::eShaderRead) &&
		    !static_cast<bool>((image.binding.pixel_sampled_aspects | image.binding.other_sampled_aspects) & writable)) {
			access |= vk::AccessFlagBits2::eShaderRead;
		}
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
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
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable || depth.stencil_meta_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	} else if (attachment_samples == 0 ||
	           vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {}) {
		EXIT("render state has no valid attachments\n");
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static bool DrawHasActivePixelShader(const CommandBuffer& buffer) {
	const auto& ctx              = buffer.GetRegisters();
	const auto& sh_regs          = ctx.GetShaderRegisters();
	const bool  has_color_output = (ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask) != 0;
	return ShaderAddressValid(buffer.GetShaders().GetPs().ps_regs.data_addr) &&
	       (has_color_output || PixelShaderHasDepthOrCoverageSideEffects(sh_regs));
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(const ShaderVertexInputBuffer& buffer,
                                           const ShaderVertexInputInfo& info) {
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < buffer.attr_num; i++) {
		const auto& resource = info.resources[buffer.attr_indices[i]];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? static_cast<uint64_t>(buffer.attr_offsets[i]) +
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
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(vertex, vs_input_info);
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
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		if (size == 0) { // unmapped: its slots bind the null buffer (below)
			range.acquired_end = range.requested_end;
			range.binding      = {nullptr, 0};
			continue;
		}
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, range.binding.first->Handle(),
		    "Kyty.VertexBufferRange[guest=0x{:016x} size=0x{:x}]", range.base_address, size);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(vertex, vs_input_info);
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
		if (range->binding.first == nullptr) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, prepared.buffers[i],
		    "Kyty.VertexBuffer[slot={} guest=0x{:016x} size=0x{:x} stride={} records={}]", i,
		    vertex.addr, size, vertex.stride, vertex.num_records);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, bool auto_draw,
                            vk::PrimitiveTopology& topology) {

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
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kRectList:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kRectListLegacy:
			if (!auto_draw) {
				EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			topology = vk::PrimitiveTopology::eTriangleStrip;
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

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
                                            bool log_setup_phases, DrawRenderState& state) {
	auto& ctx = buffer.GetRegisters();

	if (ResolveColorTargets(buffer, render_target_slice_offset)) {
		return false;
	}
	if (log_setup_phases) {
		LogDrawPhase(draw.name, "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if (slot == 0 || (render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0 &&
		                  ctx.GetRenderTarget(slot).base.addr != 0)) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (log_setup_phases) {
		LogDrawPhase(draw.name, "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);

	state.ps_active       = DrawHasActivePixelShader(buffer);
	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.name, state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

// False when a program's resource tables did not evaluate: the draw is skipped.
static bool RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw, bool log_phases,
                           DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	ResetNativeStageInput(state.ps_input_info);
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t i = 0; i < state.color_count; i++) {
		target_export_mapping[state.color_info[i].target_slot] = state.color_info[i].export_mapping;
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (log_phases) {
		LogDrawPhase(draw.name, "GetGraphicsPrograms");
	}
	(void)PipelineCache::TakeMaterializationFailure();
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vs_input_info, state.ps_input_info);
	return !PipelineCache::TakeMaterializationFailure();
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
	if (source.host_data != nullptr) {
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=transient size=0x{:x} type={}]", source.size,
		                     static_cast<uint32_t>(source.type));
	} else {
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=0x{:016x} size=0x{:x} type={}]",
		                     source.address, source.size, static_cast<uint32_t>(source.type));
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		vk_buffer.bindVertexBuffers(0, prepared.count, prepared.buffers.data(),
		                            prepared.offsets.data());
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
                                 const DrawRenderState& state, bool always_log,
                                 bool force_legacy_rect_log, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!always_log && !force_legacy_rect_log) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.name, state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vs_input_info, index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const ShaderVertexInputInfo& vs_input_info, const DrawCallInfo& draw,
                               const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
			if (emit.indexed) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kRectListLegacy:
			if (emit.indexed) {
				EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			// Sarah
			EXIT_NOT_IMPLEMENTED(!(draw.index_count == 3 && vs_input_info.buffers_num == 0));
			vk_buffer.draw(4, draw.instance_count, emit.first_vertex, emit.first_instance);
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (emit.indexed) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

// Live `capture`: the draw's shaders, targets, textures and fixed-function state.
static void CaptureFrameDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
                             const DrawRenderState& state, const PreparedBindings& vertex,
                             const std::optional<PreparedBindings>& pixel) {
	auto& call     = FrameCapture::g_call;
	call.prepared  = true;
	call.count     = draw.index_count;
	call.instances = draw.instance_count;
	if (state.vs_input_info.stage.program != nullptr) {
		call.vs = state.vs_input_info.stage.program->shader_hash;
	}
	if (state.ps_active && state.ps_input_info.stage.program != nullptr) {
		call.ps = state.ps_input_info.stage.program->shader_hash;
	}
	auto& shaders   = buffer.GetShaders();
	call.vs_address = shaders.GetVs().gs_regs.data_addr;
	call.ps_address = shaders.GetPs().ps_regs.data_addr;

	const auto& hw       = buffer.GetRegisters();
	const auto& screen   = hw.GetScreenViewport();
	const auto& viewport = screen.viewports[0];
	call.viewport[0]     = viewport.xscale;
	call.viewport[1]     = viewport.xoffset;
	call.viewport[2]     = viewport.yscale;
	call.viewport[3]     = viewport.yoffset;
	call.viewport[4]     = viewport.zscale;
	call.viewport[5]     = viewport.zoffset;
	call.scissor[0]      = screen.screen_scissor_left;
	call.scissor[1]      = screen.screen_scissor_top;
	call.scissor[2]      = screen.screen_scissor_right;
	call.scissor[3]      = screen.screen_scissor_bottom;
	call.target_mask     = hw.GetRenderTargetMask();
	call.blend_mask      = 0;
	for (uint32_t slot = 0; slot < 8; ++slot) {
		if (hw.GetBlendControl(slot).enable) call.blend_mask |= 1u << slot;
	}
	const auto& blend0 = hw.GetBlendControl(0);
	call.blend0        = blend0.color_srcblend | (uint32_t {blend0.color_destblend} << 8u) |
	              (uint32_t {blend0.color_comb_fcn} << 16u);
	const auto& depth = hw.GetDepthControl();
	call.z_enable     = depth.z_enable ? 1 : 0;
	call.z_write      = depth.z_write_enable ? 1 : 0;
	call.z_func       = depth.zfunc;

	for (uint32_t i = 0; i < state.color_count; ++i) {
		call.images.push_back(FrameCapture::FromInfo(state.color_info[i].desc.info,
		                                             FrameCapture::ColorTarget,
		                                             FrameCapture::StagePs, i));
	}
	if (state.depth_info.image_id) {
		call.images.push_back(FrameCapture::FromInfo(state.depth_info.desc.info,
		                                             FrameCapture::DepthTarget,
		                                             FrameCapture::StagePs, 0));
	}
	const auto textures = [&](const PreparedBindings& prepared, uint8_t stage) {
		const auto* program = prepared.runtime != nullptr ? prepared.runtime->program : nullptr;
		for (uint32_t i = 0; i < prepared.images.size(); ++i) {
			auto image = FrameCapture::FromInfo(prepared.images[i].desc.info, FrameCapture::Texture,
			                                    stage, i);
			if (program != nullptr && i < program->info.images.size()) {
				const auto& resource = program->info.images[i];
				image.storage =
				    resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
				image.written = resource.written ? 1 : 0;
			}
			call.images.push_back(image);
		}
	};
	textures(vertex, FrameCapture::StageVs);
	if (pixel) {
		textures(*pixel, FrameCapture::StagePs);
	}
	if (FrameCapture::WantsData(call.vs) || FrameCapture::WantsData(call.ps)) {
		const auto constants = [](const PreparedBindings& prepared, const std::string& stage) {
			if (prepared.runtime != nullptr) {
				FrameCapture::AddWords(stage + ".user", 0, prepared.runtime->resources.user_data);
				FrameCapture::AddWords(stage + ".srt", 0, prepared.runtime->resources.flattened_srt);
			}
			for (uint32_t i = 0; i < prepared.buffer_sources.size(); ++i) {
				FrameCapture::AddGuest(stage + ".buf" + std::to_string(i),
				                       prepared.buffer_sources[i].address,
				                       prepared.buffer_sources[i].size);
			}
		};
		constants(vertex, "vs");
		if (pixel) {
			constants(*pixel, "ps");
		}
	}
}

// KYTY_ASYNC_DRAW_PIPELINES=1: a draw whose pipeline no cache holds has it compiled on a worker
// (TryCreateGraphicsPipeline) instead of stopping the frame for the driver compile (100 ms to
// seconds on NVIDIA when a new area or effect appears). The draw waits at most
// KYTY_ASYNC_DRAW_WAIT_MS (default 3) and is skipped otherwise: the object shows up a few frames
// later instead of a stutter. Off by default.
static bool AsyncDrawPipelines() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_DRAW_PIPELINES");
		return value != nullptr && *value != '\0' && std::string_view(value) != "0";
	}();
	return enabled;
}
static std::chrono::microseconds AsyncDrawWait() {
	static const std::chrono::microseconds wait = [] {
		const char* value = std::getenv("KYTY_ASYNC_DRAW_WAIT_MS");
		const double ms   = value != nullptr && *value != '\0' ? std::atof(value) : 3.0;
		return std::chrono::microseconds(static_cast<int64_t>(std::clamp(ms, 0.0, 1000.0) * 1000.0));
	}();
	return wait;
}
static std::atomic<uint64_t> g_async_draws_skipped {0};

bool RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
                                         bool primitive_restart_enable, bool log_pipeline_phase,
                                         bool set_bind_debug, bool set_auto_debug) {
	auto& ucfg = buffer.GetUserConfig();
	const bool mesh_active = state.vs_input_info.stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vs_input_info.mesh;
		if (primitive_restart_enable || mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), emit.indexed, primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return true;
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
	}

	if (mesh_active && emit.indexed) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.name, "PrepareBindings");
	NativePreparationScratch<GraphicsBindings> binding_storage(
	    kyty_local_binding_scratch_mode.load(std::memory_order_relaxed) != 0);
	auto& bindings = binding_storage.Get();
	PrepareGraphicsBindingsInto(state.vs_input_info.stage, state.ps_input_info.stage,
	                            state.ps_active, bindings);
	if (FrameCapture::Active()) {
		CaptureFrameDraw(buffer, draw, state, bindings.vertex, bindings.pixel);
	}
	if (FrameGen::Enabled() && state.vs_input_info.stage.program != nullptr) {
		FrameGen::OnDraw(state.vs_input_info.stage.program->shader_hash,
		                 state.vs_input_info.stage.resources.flattened_srt);
	}
	if (XprCapture::g_state.pending) {
		CaptureXprTargets(state, bindings);
	}
	if (!emit.direct_run.empty()) {
		const auto safe_images = [&](const PreparedBindings& prepared) {
			for (size_t index = 0; index < prepared.images.size(); ++index) {
				const auto& texture = prepared.images[index];
				const auto& source = texture.desc.info.data;
				// Null descriptors resolve to a host-owned dummy image, with no
				// guest backing that an earlier draw could overwrite.
				if (source.Empty() && DecodeNativeDescriptor<ShaderTextureResource>(
				        prepared.runtime->resources.images[index]).IsNull()) continue;
				if (!source.Valid()) { return false; }
				// Attachment backing was proved unique before preparing shaders.
				// Read-only textures may alias each other, but cannot alias any
				// attachment through a different guest virtual address.
				const auto overlaps = [&](ImageId id, const ImageInfo& target) {
					if (!id) return false;
					if (texture.image_id == id) return true;
					for (const auto range : {target.data, target.stencil, target.metadata.range})
						if (!range.Empty() && ImageRangeOverlaps(source, range)) return true;
					return false;
				};
				if (overlaps(state.depth_info.image_id, state.depth_info.desc.info)) return false;
				for (uint32_t i = 0; i < state.color_count; ++i)
					if (overlaps(state.color_info[i].image_id, state.color_info[i].desc.info)) return false;
			}
			return true;
		};
		if (!safe_images(bindings.vertex) || (bindings.pixel && !safe_images(*bindings.pixel))) {
			return false;
		}
	}
	// KYTY_ASYNC_DRAW_PIPELINES: the pipeline before anything is recorded, so a draw whose pipeline
	// is still compiling can be dropped (nothing of it reached the command buffer yet).
	PipelineCache::Pipeline* async_pipeline = nullptr;
	if (AsyncDrawPipelines()) {
		auto& cache    = m_context.GetPipelineCache();
		const auto try_create = [&] {
			return cache.TryCreateGraphicsPipeline(std::span {state.color_info, state.color_count}, state.depth_info,
			                                       state.vs_input_info, buffer,
			                                       state.ps_active ? &state.ps_input_info : nullptr, topology,
			                                       primitive_restart_enable, state.programs.vertex,
			                                       state.programs.pixel, false);
		};
		async_pipeline = try_create();
		if (async_pipeline == nullptr) {
			const auto waiting  = cache.PipelineWaiting();
			const auto deadline = std::chrono::steady_clock::now() + AsyncDrawWait();
			while (waiting != nullptr && !waiting->load(std::memory_order_acquire) &&
			       std::chrono::steady_clock::now() < deadline) {
				std::this_thread::yield();
			}
			async_pipeline = try_create();
		}
		if (async_pipeline == nullptr) {
			const auto skipped = g_async_draws_skipped.fetch_add(1, std::memory_order_relaxed) + 1;
			if ((skipped & (skipped - 1)) == 0) {
				std::printf("Async draw pipelines: %llu draws skipped while their pipelines compile\n",
				            static_cast<unsigned long long>(skipped));
			}
			return true;
		}
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.name, "PrepareVertexBuffers");
		vertex_bindings = AcquireVertexBuffers(buffer, state.vs_input_info);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	std::pair<Buffer*, uint64_t> gpu_args {};
	if (emit.gpu_args != 0) {
		const uint64_t size = emit.indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
		gpu_args = m_context.GetBufferCache().ObtainBuffer(
		    emit.gpu_args, uint64_t {emit.gpu_args_count - 1u} * emit.gpu_args_stride + size, false);
		EXIT_IF(gpu_args.first == nullptr);
	}
	if (!emit.direct_run.empty() &&
	    (m_context.GetGpuResources().MappingEpoch() != emit.run_mapping_epoch ||
	     m_context.GetGpuResources().PreparationAliasEpoch() != emit.run_alias_epoch)) {
		return false;
	}
	const auto rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
	                         bindings.pixel);

	if (log_pipeline_phase) {
		LogDrawPhase(draw.name, "CreatePipeline");
	}
	auto& pipeline = async_pipeline != nullptr
	                     ? *async_pipeline
	                     : m_context.GetPipelineCache().CreateGraphicsPipeline(
	                           std::span {state.color_info, state.color_count}, state.depth_info,
	                           state.vs_input_info, buffer, state.ps_active ? &state.ps_input_info : nullptr,
	                           topology, primitive_restart_enable, state.programs.vertex, state.programs.pixel);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	bool record_draw = false;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	const auto primitive = ucfg.GetPrimType();
	record_draw = LocalVulkanRecording::PacketsEnabled() && !mesh_active && emit.gpu_args == 0 &&
	    ((!set_bind_debug && !set_auto_debug) ||
	     kyty_local_draw_packets_mode.load(std::memory_order_relaxed) != 0) &&
	    (primitive == Prospero::PrimitiveType::kPointList ||
	     primitive == Prospero::PrimitiveType::kLineList ||
	     primitive == Prospero::PrimitiveType::kLineStrip ||
	     primitive == Prospero::PrimitiveType::kTriList ||
	     primitive == Prospero::PrimitiveType::kTriFan ||
	     primitive == Prospero::PrimitiveType::kTriStrip ||
	     primitive == Prospero::PrimitiveType::kRectList);
#endif
	auto vk_buffer = buffer.Handle();
	if (set_bind_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x100u);
	}
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x200u);
	}
	if (!mesh_active && !record_draw) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (bindings.pixel.has_value()) {
		if (set_auto_debug) {
			SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
		}
	}
	std::array<PreparedBindings*, 2> descriptor_stages {&bindings.vertex, nullptr};
	const size_t                     descriptor_stage_count = bindings.pixel.has_value() ? 2u : 1u;
	if (bindings.pixel) {
		descriptor_stages[1] = &*bindings.pixel;
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline,
	               std::span {descriptor_stages.data(), descriptor_stage_count});
	if (mesh_active) {
		const uint32_t draw_data[] {
		    draw.index_count,
		    emit.indexed ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
		    emit.first_instance, index_source.guest_element_size,
		    static_cast<uint32_t>(index_source.address),
		    static_cast<uint32_t>(index_source.address >> 32u)};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(draw_data), draw_data);
	} else if (!record_draw) {
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	const auto feedback_aspects = rendering.depth_stencil_attachment.image_layout ==
	                                      vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
	                                  ? vk::ImageAspectFlags {vk::ImageAspectFlagBits::eDepth}
	                                  : vk::ImageAspectFlags {};
	if (!record_draw) {
		LiveCounters::Add(LiveCounters::DirectDraws);
		if (mesh_active) LiveCounters::Add(LiveCounters::MeshDraws);
		CommitGraphicsState(buffer, state.vs_input_info, state.color_info, state.color_count,
		                    state.depth_info, pipeline.pipeline, feedback_aspects);
	} else {
		buffer.InvalidateGraphicsState();
	}

	LogDrawPhase(draw.name, "BeginRendering");
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	if (!record_draw) {
		if (set_auto_debug) {
			SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
		}
		if (mesh_active) {
			vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
		} else if (emit.gpu_args != 0 && emit.indexed) {
			vk_buffer.drawIndexedIndirect(gpu_args.first->Handle(), gpu_args.second, emit.gpu_args_count,
			                              emit.gpu_args_stride);
		} else if (emit.gpu_args != 0) {
			vk_buffer.drawIndirect(gpu_args.first->Handle(), gpu_args.second, emit.gpu_args_count,
			                       emit.gpu_args_stride);
		} else if (!emit.direct_run.empty()) {
			for (const auto& item: emit.direct_run) {
				vk_buffer.drawIndexed(item.indexCount, item.instanceCount, item.firstIndex,
				                      item.vertexOffset, item.firstInstance);
			}
		} else {
			EmitDrawPrimitives(ucfg, vk_buffer, state.vs_input_info, draw, emit);
		}
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	else {
		LocalDrawRecording::Draw packet {};
		packet.state = LocalDrawRecording::Capture(
		    buffer, state.vs_input_info, state.color_info, state.color_count, state.depth_info,
		    m_context.GetGraphics().attachment_feedback_loop_enabled, feedback_aspects);
		packet.command = vk_buffer;
		packet.pipeline = pipeline.pipeline;
		packet.index_buffer = index_binding.buffer;
		packet.index_offset = index_binding.offset;
		packet.index_type = index_binding.type;
		packet.indexed = emit.indexed;
		const vk::DrawIndexedIndirectCommand command {draw.index_count, draw.instance_count,
		    emit.indexed ? 0u : emit.first_vertex, emit.vertex_offset, emit.first_instance};
		LocalDrawRecording::Record(packet, {vertex_bindings.buffers.data(), vertex_bindings.count},
		    {vertex_bindings.offsets.data(), vertex_bindings.count},
		    emit.direct_run.empty() ? std::span {&command, 1} : emit.direct_run);
	}
#endif

	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	if (HasShaderBufferWrites(state.vs_input_info.stage)) {
		shader_write_stages |= mesh_active ? vk::PipelineStageFlagBits::eMeshShaderEXT
		                                   : vk::PipelineStageFlagBits::eVertexShader;
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.name, "DrawComplete");
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// After the draw is recorded: storing may clean buffer owners (copies) and so
	// may end rendering or restart the scheduler.
	if (m_native_xpr_verify.record != nullptr)
		NativeXprVerify(buffer, state, rendering, std::span {descriptor_stages.data(), descriptor_stage_count}, emit);
	// (A table draw of vertices too: TableStore stores what a table try asked for.)
	if (m_table_xpr)
		TableStore(buffer, state, topology, primitive_restart_enable, rendering, !mesh_active && vertex_bindings.count == 0);
	if (m_native_xpr_store) {
		m_native_xpr_store = false;
		if (!mesh_active && emit.indexed && vertex_bindings.count == 0)
			NativeXprStore(buffer, state, topology, primitive_restart_enable, rendering,
			               std::span {descriptor_stages.data(), descriptor_stage_count}, emit);
		else
			LiveCounters::Add(LiveCounters::XprRefuseDraw);
	}
#endif
	return true;
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "native-xpr.inc"
#include "table-xpr.inc"
#endif

bool BuildDrawIndexRun(std::span<const DrawIndexArgs> draws, DrawIndexRun& run) {
	if (draws.size() < 2 || draws.size() > DrawIndexRun::MaxDraws) return false;
	const auto index_type = draws[0].index_type_and_size;
	if (index_type > 1) return false;
	const uint32_t element_size = index_type == 0 ? 2 : 4;
	uint64_t       begin = UINT64_MAX, end = 0;
	for (const auto& item: draws) {
		const auto address = reinterpret_cast<uint64_t>(item.index_addr);
		if (item.index_type_and_size != index_type ||
		    item.offset_source != DrawOffsetSource::IndirectArgs || !item.index_count ||
		    !item.instance_count || item.render_target_slice_offset || !address ||
		    address % element_size ||
		    address > UINT64_MAX - uint64_t(item.index_count) * element_size)
			return false;
		begin = std::min(begin, address);
		end   = std::max(end, address + uint64_t(item.index_count) * element_size);
	}
	// Avoid widening scattered tiny index ranges into a large upload.
	if (end - begin > 16u * 1024u * 1024u) return false;
	DrawIndexRun next;
	for (size_t i = 0; i < draws.size(); ++i) {
		const auto& item = draws[i];
		next.commands[i] = {
		    item.index_count, item.instance_count,
		    static_cast<uint32_t>((reinterpret_cast<uint64_t>(item.index_addr) - begin) /
		                          element_size),
		    item.base_vertex, item.first_instance};
	}
	next.indices.address = begin;
	next.indices.size    = end - begin;
	next.indices.type    = index_type == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
	next.indices.guest_element_size = element_size;
	run                             = next;
	return true;
}
// A draw run reads `range` as a buffer only when its guest backing is unique,
// no render target of the run aliases it and no image overlaps it, except, with
// `sampled_overlaps`, sampled images whose contents still come from the CPU:
// nothing in the run writes those.
bool RenderExecutor::ReadOnlyDrawBufferRangeSafe(GuestRange range, const DrawRenderState& state,
                                                 bool sampled_overlaps) {
	if (!range.address || !range.size) {
		return true;
	}
	if (!range.Valid() || !LibKernel::Memory::IsUniqueGuestBackingRange(range.address, range.size)) {
		return false;
	}
	auto& textures = m_context.GetTextureCache();
	if (sampled_overlaps ? textures.ClassifyReadOnlyBufferOverlap(range.address, range.size) ==
	                           TextureCache::ReadOnlyBufferOverlap::Unsafe
	                     : textures.HasTrackedDataOverlap(range.address, range.size)) {
		return false;
	}
	const auto overlaps = [&](ImageId id, const ImageInfo& image) {
		if (!id) {
			return false;
		}
		for (const auto target: {image.data, image.stencil, image.metadata.range}) {
			if (!target.Empty() && ImageRangeOverlaps(range, target)) {
				return true;
			}
		}
		return false;
	};
	if (overlaps(state.depth_info.image_id, state.depth_info.desc.info)) {
		return false;
	}
	for (uint32_t i = 0; i < state.color_count; ++i) {
		if (overlaps(state.color_info[i].image_id, state.color_info[i].desc.info)) {
			return false;
		}
	}
	return true;
}

bool RenderExecutor::TryDrawIndexRun(uint64_t submit_id, CommandBuffer& buffer,
                                     std::span<const DrawIndexArgs> draws,
                                     std::span<const uint64_t>      argument_addresses) {
	if (draws.size() < 2 || draws.size() > 64 || argument_addresses.size() < draws.size() ||
	    argument_addresses.size() > 64 || buffer.IsInvalid() ||
	    buffer.GetUserConfig().GetPrimType() != Prospero::PrimitiveType::kTriList ||
	    buffer.GetRegisters().GetColorControl().mode > 1 ||
	    buffer.GetRegisters().GetClipControl().clip_disable ||
	    !DrawHasValidVertexShader(buffer.GetShaders()))
		return false;
	DrawIndexRun run;
	if (!BuildDrawIndexRun(draws, run) || ResolvePrimitiveRestart(buffer, run.indices))
		return false;
	if (!m_context.GetGpuResources().IsMapped(run.indices.address, run.indices.size)) return false;
	m_context.GetCommandScheduler().PopPendingOperations();
	LiveTrace::MarkAfter gpu_mark {[&] { return m_context.GetCommandScheduler().Current().RawHandle(); },
	                               LiveTrace::MarkDraw, buffer.GetShaders().GetPs().ps_regs.data_addr};
	Common::LockGuard lock(m_context.GetMutex());
	const auto        mapping_epoch = m_context.GetGpuResources().MappingEpoch();
	const auto        alias_epoch   = m_context.GetGpuResources().PreparationAliasEpoch();
	if (!alias_epoch) return false;
	const DrawCallInfo draw {"DrawIndexRun", CommandBufferDebugOp::DrawIndex, draws[0].index_count,
	                         draws[0].instance_count, draws[0].first_instance};
	NativeDrawState state_storage;
	auto& state = state_storage.Get();
	if (!PrepareDrawRenderState(buffer, draw, 0, false, state)) {
		ResetBindings();
		return false;
	}
	// Stencil tests/writes remain ordered by the original individual draws.
	// Only explicit per-draw clears require a boundary here.
	if (state.depth_info.depth_clear_enable || state.depth_info.stencil_clear_enable) {
		ResetBindings();
		return false;
	}
	const auto unique_target = [](ImageId id, const ImageInfo& info) {
		if (!id) return true;
		for (const auto range: {info.data, info.stencil, info.metadata.range})
			if (!range.Empty() && (!range.Valid() || !LibKernel::Memory::IsUniqueGuestBackingRange(
			                                             range.address, range.size)))
				return false;
		return true;
	};
	bool targets_unique = unique_target(state.depth_info.image_id, state.depth_info.desc.info);
	for (uint32_t i = 0; i < state.color_count; ++i)
		targets_unique &=
		    unique_target(state.color_info[i].image_id, state.color_info[i].desc.info);
	if (!targets_unique) {
		ResetBindings();
		return false;
	}
	const auto overlaps_target = [&](GuestRange source) {
		const auto overlaps = [&](ImageId id, const ImageInfo& target) {
			if (!id) return false;
			for (const auto range: {target.data, target.stencil, target.metadata.range})
				if (!range.Empty() && ImageRangeOverlaps(source, range)) return true;
			return false;
		};
		if (overlaps(state.depth_info.image_id, state.depth_info.desc.info)) return true;
		for (uint32_t i = 0; i < state.color_count; ++i)
			if (overlaps(state.color_info[i].image_id, state.color_info[i].desc.info)) return true;
		return false;
	};
	for (const auto address: argument_addresses) {
		if (!GuestRange {address, 20}.Valid() || overlaps_target({address, 20})) {
			ResetBindings();
			return false;
		}
	}
	bool reads_safe   = true;
	auto observe_read = [&](uint64_t address, uint64_t size) {
		if (!GuestRange {address, size}.Valid() || overlaps_target({address, size}))
			reads_safe = false;
	};
	{
		ShaderReadObserver observer(
		    [](void* userdata, uint64_t address, uint64_t size) {
			    (*static_cast<decltype(observe_read)*>(userdata))(address, size);
		    },
		    &observe_read);
		reads_safe = RefreshShaders(buffer, draw, false, state) && reads_safe;
	}
	if (!reads_safe) {
		ResetBindings();
		return false;
	}
	const auto read_only = [](const ShaderStageRuntime& stage) {
		const auto& p = *stage.program;
		return !p.info.uses_dma &&
		       std::ranges::none_of(
		           p.info.images,
		           [](const auto& image) { return image.written || image.atomic; }) &&
		       !HasShaderBufferWrites(stage) &&
		       std::ranges::none_of(p.bindings.descriptors, [](const auto& binding) {
			       return binding.kind == ShaderRecompiler::IR::DescriptorBindingKind::Gds;
		       });
	};
	if (state.vs_input_info.stage.program->stage != ShaderType::Vertex ||
	    state.vs_input_info.buffers_num || state.vs_input_info.resources_num ||
	    !read_only(state.vs_input_info.stage) ||
	    (state.ps_active && !read_only(state.ps_input_info.stage))) {
		ResetBindings();
		return false;
	}
	// Exclude buffer/attachment feedback through either virtual or physical
	// aliases. Ordinary attachment depth/stencil ordering stays on Vulkan.
	const bool sampled_overlaps =
	    kyty_local_draw_run_ranges_mode.load(std::memory_order_relaxed) != 0;
	const auto safe_range = [&](uint64_t address, uint64_t size) {
		return ReadOnlyDrawBufferRangeSafe({address, size}, state, sampled_overlaps);
	};
	const auto safe_buffers = [&](const ShaderStageRuntime& stage) {
		for (const auto& value: stage.resources.buffers) {
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(value);
			const auto requested =
			    uint64_t(descriptor.NumRecords()) * std::max<uint16_t>(1, descriptor.Stride());
			const auto address = descriptor.Base48();
			if (address && requested &&
			    !safe_range(address, LibKernel::Memory::ClampRangeSize(address, requested)))
				return false;
		}
		return true;
	};
	if (!safe_range(run.indices.address, run.indices.size) ||
	    !safe_buffers(state.vs_input_info.stage) ||
	    (state.ps_active && !safe_buffers(state.ps_input_info.stage))) {
		ResetBindings();
		return false;
	}
	DrawEmitInfo emit {};
	emit.indexed           = true;
	emit.direct_run        = {run.commands.data(), draws.size()};
	emit.run_mapping_epoch = mapping_epoch;
	emit.run_alias_epoch   = alias_epoch;
	const bool issued =
	    ExecutePreparedDraw(submit_id, buffer, draw, state, vk::PrimitiveTopology::eTriangleList,
	                        emit, run.indices, false, false, false, false);
	ResetBindings();
	return issued;
}

// XPR capture for the offline replay tests (xpr-capture.h). Starts a pending
// capture once RefreshShaders has set up the draw's shaders; CaptureXprTargets
// completes and writes it when the bindings and render targets are resolved.
void RenderExecutor::CaptureXprDraw(CommandBuffer& buffer, const DrawRenderState& state,
                                    const DrawIndexArgs& args) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();
	auto  pending = std::make_unique<XprCapture::PendingDraw>();
	auto& d       = *pending;
	d.draw = {args.index_count, args.instance_count, static_cast<uint32_t>(args.base_vertex),
	          args.first_instance, args.index_type_and_size};
	const auto stage_buffers = [](XprCapture::StageCapture& capture) {
		for (const auto& value: capture.resources.buffers) {
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(value);
			const auto address    = descriptor.Base48();
			const auto stride     = descriptor.Stride();
			const auto requested  = stride != 0 ? uint64_t(stride) * descriptor.NumRecords()
			                                    : uint64_t(descriptor.NumRecords());
			const auto size = address == 0 || requested == 0
			                      ? 0
			                      : Libs::LibKernel::Memory::ClampRangeSize(address, requested);
			capture.buffers.emplace_back(address, size <= (64ull << 20u) ? size : 0);
		}
	};
	{
		ShaderVertexInputInfo info;
		ResetNativeVertexInput(info);
		const auto params = PrepareProgram(sh_ctx.GetVs(), ctx, buffer.GetUserConfig(), info);
		auto& r = d.vertex.record;
		r.stage = state.vs_input_info.stage.program->stage;
		r.hash  = params.hash;
		r.user_data_count = static_cast<uint32_t>(params.user_data.size());
		r.code.assign(params.code.begin(), params.code.end());
		r.back_code.assign(params.back_code.begin(), params.back_code.end());
		r.vertex = info;
		BuildStageStaticKey(info, r.static_key);
		d.vertex.user_data    = params.user_data;
		d.vertex.code_address = params.Base();
		d.vertex.push_start   = state.vs_input_info.stage.program->bindings.push_data_start_dword;
		d.vertex.resources    = state.vs_input_info.stage.resources;
		stage_buffers(d.vertex);
	}
	d.has_pixel = state.ps_active && state.ps_input_info.stage.program != nullptr;
	if (d.has_pixel) {
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
		for (uint32_t i = 0; i < state.color_count; i++)
			mapping[state.color_info[i].target_slot] = state.color_info[i].export_mapping;
		ShaderPixelInputInfo info;
		ResetNativeStageInput(info);
		const auto params = PrepareProgram(sh_ctx.GetPs(), ctx.GetShaderRegisters(), mapping, info);
		auto& r = d.pixel.record;
		r.stage = ShaderType::Pixel;
		r.hash  = params.hash;
		r.user_data_count = static_cast<uint32_t>(params.user_data.size());
		r.code.assign(params.code.begin(), params.code.end());
		r.pixel = info;
		BuildStageStaticKey(info, r.static_key);
		d.pixel.user_data    = params.user_data;
		d.pixel.code_address = params.Base();
		d.pixel.push_start   = state.ps_input_info.stage.program->bindings.push_data_start_dword;
		d.pixel.resources    = state.ps_input_info.stage.resources;
		stage_buffers(d.pixel);
	}
	const auto block = [](std::vector<uint32_t>& out, const auto& value) {
		out.resize((sizeof(value) + 3) / 4);
		std::memcpy(out.data(), &value, sizeof(value));
	};
	block(d.context, ctx);
	block(d.shaders, sh_ctx);
	block(d.user_config, buffer.GetUserConfig());
	// Shader map entries and the guest memory they point to.
	const auto& vs = sh_ctx.GetVs();
	for (const auto address: {vs.es_regs.data_addr, vs.gs_regs.data_addr, sh_ctx.GetPs().ps_regs.data_addr}) {
		ShaderMappedData data {};
		if (address == 0 || !ShaderLookupMappedData(address, &data) ||
		    std::ranges::any_of(d.entries, [&](const auto& e) { return e.address == address; }))
			continue;
		XprCapture::ShaderEntry entry;
		entry.address         = address;
		entry.type            = static_cast<uint32_t>(data.type);
		entry.code_size       = data.code_size_bytes;
		entry.scratch         = data.scratch_size_dwords;
		entry.semantics_count = data.num_input_semantics;
		entry.semantics       = reinterpret_cast<uint64_t>(data.input_semantics);
		entry.user_data       = reinterpret_cast<uint64_t>(data.user_data);
		d.entries.push_back(entry);
		uint64_t window = data.code_size_bytes;
		const auto* code = reinterpret_cast<const uint32_t*>(address);
		if (code[0] == 0xBEEB03FFu) window = std::max<uint64_t>(window, (uint64_t(code[1]) + 1) * 8 + 32);
		d.exact.emplace_back(address, window);
		if (data.input_semantics != nullptr && data.num_input_semantics != 0)
			d.exact.emplace_back(entry.semantics, uint64_t(data.num_input_semantics) * sizeof(ShaderSemantic));
		if (data.user_data != nullptr) {
			d.exact.emplace_back(entry.user_data, sizeof(ShaderUserData));
			const auto& ud = *data.user_data;
			if (ud.direct_resource_offset != nullptr && ud.direct_resource_count != 0)
				d.exact.emplace_back(reinterpret_cast<uint64_t>(ud.direct_resource_offset),
				                     uint64_t(ud.direct_resource_count) * sizeof(uint16_t));
			for (uint32_t i = 0; i < 4; ++i)
				if (ud.sharp_resource_offset[i] != nullptr && ud.sharp_resource_count[i] != 0)
					d.exact.emplace_back(reinterpret_cast<uint64_t>(ud.sharp_resource_offset[i]),
					                     uint64_t(ud.sharp_resource_count[i]) * sizeof(ShaderSharp));
		}
	}
	d.reads = std::move(XprCapture::g_state.reads);
	XprCapture::g_state.pending = std::move(pending);
}

void RenderExecutor::CaptureXprTargets(const DrawRenderState& state,
                                       const GraphicsBindings& bindings) {
	auto pending = std::move(XprCapture::g_state.pending);
	if (!pending) return;
	auto&      d   = *pending;
	using Kind     = XprCapture::FillKind;
	const auto add = [&](std::vector<XprCapture::Fill>& out, const GuestRange& range, Kind kind,
	                     uint32_t value = 0) {
		if (!range.Empty()) out.push_back({range.address, range.size, kind, value});
	};
	for (const auto* prepared: {&bindings.vertex, bindings.pixel ? &*bindings.pixel : nullptr}) {
		if (prepared == nullptr) continue;
		for (const auto& image: prepared->images) {
			add(d.fills, image.desc.info.data, Kind::Texture);
			add(d.fills, image.desc.info.stencil, Kind::Texture);
			add(d.fills, image.desc.info.metadata.range, Kind::Metadata);
		}
	}
	for (uint32_t i = 0; i < state.color_count; ++i) {
		const auto& info = state.color_info[i].desc.info;
		add(d.fills, info.data, Kind::ColorTarget);
		add(d.fills, info.metadata.range, Kind::Metadata);
		add(d.targets, info.data, Kind::ColorTarget);
	}
	if (state.depth_info.image_id) {
		const auto& info = state.depth_info.desc.info;
		const auto  op   = state.depth_info.depth_compare_op;
		const bool  far_is_one = op == vk::CompareOp::eLess || op == vk::CompareOp::eLessOrEqual;
		const bool  d16 = info.pixel_format == vk::Format::eD16Unorm || info.pixel_format == vk::Format::eD16UnormS8Uint;
		const uint32_t far_word = !far_is_one ? 0u : d16 ? 0xffffffffu : 0x3f800000u;
		add(d.fills, info.data, Kind::DepthTarget, far_word);
		add(d.fills, info.stencil, Kind::Stencil);
		add(d.fills, info.metadata.range, Kind::Metadata);
		add(d.targets, info.data, Kind::DepthTarget, far_word);
	}
	XprCapture::WriteDraw(d);
	XprCapture::Written();
}

bool RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	XprCapture::g_state.pending.reset(); // a capture never spans two draws
	KYTY_PROFILER_FUNCTION();
	FrameCapture::Scope frame_capture("DrawIndex", false);
	if (FrameCapture::Active()) {
		FrameCapture::g_call.index       = reinterpret_cast<uint64_t>(args.index_addr);
		FrameCapture::g_call.index_size  = args.index_type_and_size;
		FrameCapture::g_call.base_vertex = args.base_vertex;
		FrameCapture::g_call.indirect    = args.gpu_args;
	}

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();
	LiveCensus::Scope census(LiveCensus::Draw, sh_ctx.GetVs().gs_regs.data_addr, sh_ctx.GetPs().ps_regs.data_addr);
	LiveCensus::DrawPhases census_phases(sh_ctx.GetPs().ps_regs.data_addr);
	LiveTrace::MarkAfter gpu_mark {[&] { return m_context.GetCommandScheduler().Current().RawHandle(); },
	                               LiveTrace::MarkDraw, sh_ctx.GetPs().ps_regs.data_addr};
	LiveCounters::g_last_dispatch_shader = 0;

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || args.instance_count == 0) {
		return true;
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (LiveCounters::g_dispatch_keys_on.load(std::memory_order_relaxed)) {
		// Local diagnostic: was this object (the native XPR key: shader addresses and user data)
		// drawn last frame too? What a native record for direct draws could reuse.
		static std::unordered_set<uint64_t> previous, current;
		static uint32_t                     keys_frame = 0;
		const auto                          frame      = static_cast<uint32_t>(m_context.FrameNumber());
		if (frame != keys_frame) {
			previous.swap(current);
			current.clear();
			keys_frame = frame;
		}
		thread_local std::vector<uint32_t> key;
		NativeXprKey(buffer, key);
		const auto hash = XXH3_64bits(key.data(), key.size() * sizeof(uint32_t));
		current.insert(hash);
		LiveCounters::Add(previous.contains(hash) ? LiveCounters::DrawKeySame : LiveCounters::DrawKeyNew);
	}
#endif

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer)) {
		if (FrameCapture::Active()) FrameCapture::g_call.consumed = "metadata_or_depth_copy";
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		sh_print("GraphicsRenderDrawIndex():Shader:", sh_ctx);
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
	if (!GetDrawTopology(ucfg, false, topology)) {
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

	const DrawCallInfo draw {"DrawIndex", CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	NativeDrawState state_storage;
	auto& state = state_storage.Get();
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, true,
	                            state)) {
		ResetBindings();
		return true;
	}
	if (args.gpu_args != 0 && state.vs_input_info.stage.program->stage == ShaderType::Mesh) {
		// A mesh draw sizes its task grid from the counts.
		ResetBindings();
		return false;
	}

	bool programs_ok = false;
	if (XprCapture::Enabled() && XprCapture::g_state.current_xpr &&
	    XprCapture::g_state.capture_this) {
		auto& reads = XprCapture::g_state.reads;
		reads.clear();
		auto observe = [&](uint64_t address, uint64_t size) { reads.emplace_back(address, size); };
		{
			ShaderReadObserver observer(
			    [](void* userdata, uint64_t address, uint64_t size) {
				    (*static_cast<decltype(observe)*>(userdata))(address, size);
			    },
			    &observe);
			programs_ok = RefreshShaders(buffer, draw, true, state);
		}
		if (programs_ok) CaptureXprDraw(buffer, state, args);
	} else {
		programs_ok = RefreshShaders(buffer, draw, true, state);
	}
	if (!programs_ok) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, true, false, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto vertex_offset =
	    indirect
	        ? args.base_vertex
	        : ResolveVertexOffset(ucfg.GetIndexOffset(), state.vs_input_info) + args.base_vertex;

	DrawEmitInfo emit {};
	emit.indexed       = true;
	emit.state_offsets = !indirect && args.base_vertex == 0;
	emit.vertex_offset = vertex_offset;
	emit.first_instance =
	    indirect ? args.first_instance : ResolveInstanceOffset(state.vs_input_info);

	emit.gpu_args        = args.gpu_args;
	emit.gpu_args_count  = args.gpu_args_count;
	emit.gpu_args_stride = args.gpu_args_stride;

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart, true, true, false);
	ResetBindings();
	return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();
	FrameCapture::Scope frame_capture("DrawAuto", false);

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();
	LiveTrace::MarkAfter gpu_mark {[&] { return m_context.GetCommandScheduler().Current().RawHandle(); },
	                               LiveTrace::MarkDraw, sh_ctx.GetPs().ps_regs.data_addr};

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	if (args.vertex_count == 0 || args.instance_count == 0) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer)) {
		if (FrameCapture::Active()) FrameCapture::g_call.consumed = "metadata_or_depth_copy";
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		sh_print("GraphicsRenderDrawIndexAuto():Shader:", sh_ctx);
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

	const DrawCallInfo draw {"DrawIndexAuto", CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	NativeDrawState state_storage;
	auto& state = state_storage.Get();
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, false,
	                            state)) {
		ResetBindings();
		return true;
	}

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, true, topology)) {
		ResetBindings();
		return true;
	}
	if (!RefreshShaders(buffer, draw, false, state)) {
		ResetBindings();
		return true;
	}
	if (args.gpu_args != 0 && state.vs_input_info.stage.program->stage == ShaderType::Mesh) {
		ResetBindings();
		return false;
	}

	const bool rect_list = topology == vk::PrimitiveTopology::ePatchList;
	if (rect_list && state.vs_input_info.buffers_num == 0 &&
	    state.vs_input_info.stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, false,
	                     ucfg.GetPrimType() == Prospero::PrimitiveType::kRectListLegacy, 0,
	                     nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto vertex_offset =
	    indirect ? static_cast<int32_t>(args.first_vertex)
	             : ResolveVertexOffset(ucfg.GetIndexOffset(), state.vs_input_info) +
	                   static_cast<int32_t>(args.first_vertex);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset);
	emit.first_instance =
	    indirect ? args.first_instance : ResolveInstanceOffset(state.vs_input_info);

	emit.gpu_args        = args.gpu_args;
	emit.gpu_args_count  = args.gpu_args_count;
	emit.gpu_args_stride = args.gpu_args_stride;

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false, false,
	                    false, true);
	ResetBindings();
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
	if (!src.image_id || !dst.image_id) {
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
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
