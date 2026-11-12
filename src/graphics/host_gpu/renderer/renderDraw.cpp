#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/gpuPhases.h"
#include "graphics/host_gpu/renderer/tableMode.h"
#include "graphics/host_gpu/renderer/tableTargets.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawRepeats.h"
#include "graphics/host_gpu/renderer/frameGraph.h"
#include "graphics/host_gpu/renderer/frameGraphQueries.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/rendererLock.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
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

// Diagnostics: the raster state of the draws into depth targets 8192 or more texels wide (the
// top-down shadow maps of a whole track), a line for each state the first time (at most 48). How
// they are biased decides how much the ground shadows itself under a low sun.
static void LogBigDepthDraw(const CommandBuffer& buffer, const ShaderVertexInputInfo& vs_input_info,
                            const RenderDepthInfo& depth, bool bias_enable, float constant,
                            float slope) {
	const auto& info = depth.desc.info;
	if (!depth.image_id || info.extent.width < 8192 || vs_input_info.stage.program == nullptr) {
		return;
	}
	const auto& ctx  = buffer.GetRegisters();
	const auto& mode = ctx.GetModeControl();
	const auto& po   = ctx.GetPolyOffset();
	const auto& vp   = ctx.GetScreenViewport().viewports[0];
	const auto& clip = ctx.GetClipControl();
	const auto  vs   = vs_input_info.stage.program->shader_hash;
	uint64_t    key  = info.data.address ^ (vs * 0x9e3779b97f4a7c15ull);
	for (const float value: {po.clamp, po.front_scale, po.front_offset, po.back_scale,
	                         po.back_offset, vp.zscale, vp.zoffset}) {
		key = (key ^ std::bit_cast<uint32_t>(value)) * 0x100000001b3ull;
	}
	key ^= (uint64_t {mode.cull_front} << 1u) | (uint64_t {mode.cull_back} << 2u) |
	       (uint64_t {mode.face} << 3u) | (uint64_t {mode.poly_offset_front_enable} << 4u) |
	       (uint64_t {mode.poly_offset_back_enable} << 5u) |
	       (uint64_t {static_cast<uint32_t>(depth.depth_compare_op)} << 6u) |
	       (uint64_t {depth.depth_write_enable} << 9u) | (uint64_t {clip.clip_disable} << 10u);
	static std::mutex                   logged_mutex;
	static std::unordered_set<uint64_t> logged;
	std::lock_guard                     lock(logged_mutex);
	if (logged.size() >= 48 || !logged.insert(key).second) {
		return;
	}
	LOGF("Big depth draw: 0x%012" PRIx64 " %ux%u %s, vs %016" PRIx64
	     ": cull front %d back %d, front face %s; offset front %d back %d, scale %g/%g, offset "
	     "%g/%g, clamp %g, db bits %d%s -> bias %s, constant %g, slope %g; depth test %d write "
	     "%d op %d; viewport scale %g,%g offset %g,%g z scale %g offset %g; clip disable %d, z "
	     "clip %d\n",
	     info.data.address, info.extent.width, info.extent.height,
	     vk::to_string(depth.desc.view_info.format).c_str(), vs, mode.cull_front ? 1 : 0,
	     mode.cull_back ? 1 : 0, mode.face ? "cw" : "ccw", mode.poly_offset_front_enable ? 1 : 0,
	     mode.poly_offset_back_enable ? 1 : 0, static_cast<double>(po.front_scale),
	     static_cast<double>(po.back_scale), static_cast<double>(po.front_offset),
	     static_cast<double>(po.back_offset), static_cast<double>(po.clamp),
	     static_cast<int>(po.neg_num_db_bits), po.db_is_float_fmt ? " float" : "",
	     bias_enable ? "on" : "off", static_cast<double>(constant), static_cast<double>(slope),
	     depth.depth_test_enable ? 1 : 0, depth.depth_write_enable ? 1 : 0,
	     static_cast<int>(depth.depth_compare_op), static_cast<double>(vp.xscale),
	     static_cast<double>(vp.yscale), static_cast<double>(vp.xoffset),
	     static_cast<double>(vp.yoffset), static_cast<double>(vp.zscale),
	     static_cast<double>(vp.zoffset), clip.clip_disable ? 1 : 0, clip.IsZClipEnabled() ? 1 : 0);
}

// depth:equal-relaxed: whether a larger depth is closer on this depth target (reversed Z), from
// the last draw that tested it for GREATER or LESS (or equal) while the step was on; before one,
// from the depth clear value (0: reversed Z).
static bool ReversedDepth(const RenderDepthInfo& depth) {
	static std::mutex                         mutex;
	static std::unordered_map<uint64_t, bool> reversed;
	const uint64_t                            address = depth.desc.info.data.address;
	const std::lock_guard                     lock(mutex);
	switch (depth.depth_compare_op) {
		case vk::CompareOp::eGreater:
		case vk::CompareOp::eGreaterOrEqual: reversed[address] = true; return true;
		case vk::CompareOp::eLess:
		case vk::CompareOp::eLessOrEqual: reversed[address] = false; return false;
		default: break;
	}
	const auto found = reversed.find(address);
	return found != reversed.end() ? found->second : depth.depth_clear_value == 0.0f;
}

// Round 96, diagnostics: KYTY_EQUAL_ALWAYS=<vertex shader hash>[,<hash>...] (hex): a draw of one
// of those vertex shaders that tests depth for EQUAL tests nothing instead, without writing
// depth, so its pixels show on screen wherever its geometry lands. The crowd's colour passes
// pass none of the samples their depth prepass drew (round 93's probes), so the spectators keep
// no colour (dark shapes in the stands); this shows whether their geometry lands elsewhere, or
// lands right with a depth a little apart.
static bool EqualTestedNothing(uint64_t vs_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> parsed;
		const char*           text = std::getenv("KYTY_EQUAL_ALWAYS");
		while (text != nullptr && *text != '\0') {
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) {
				break;
			}
			parsed.push_back(value);
			text = *end == ',' ? end + 1 : end;
		}
		if (!parsed.empty()) {
			LOGF("Render: KYTY_EQUAL_ALWAYS: draws of %zu vertex shaders that test depth for "
			     "EQUAL test nothing\n",
			     parsed.size());
		}
		return parsed;
	}();
	return !hashes.empty() && std::ranges::find(hashes, vs_hash) != hashes.end();
}

static void SetGraphicsDynamicParams(const CommandBuffer& buffer, const CommandRecorder& vk_buffer,
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
	vk_buffer.setViewportWithCount(viewport_count, viewports.data());
	vk_buffer.setScissorWithCount(viewport_count, scissors.data());

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
	vk_buffer.setLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	vk_buffer.setBlendConstants(blend_constants.data());
	// KYTY_BISECT's depth:equal-relaxed: a test for EQUAL passes "as close or closer" with a small
	// bias towards the camera, without writing depth (ReversedDepth says which way is closer; GT7
	// uses reversed Z). A pixel the depth prepass and this pass compute a little apart is drawn
	// then; one truly behind something is not.
	const float relax_units = depth.depth_test_enable ? Bisect::DepthEqualRelaxed() : 0.0f;
	const bool  relax_step  = relax_units > 0.0f;
	const bool  reversed_z  = relax_step && ReversedDepth(depth);
	const bool  relax_equal = relax_step && depth.depth_compare_op == vk::CompareOp::eEqual;
	const bool  equal_always = depth.depth_test_enable &&
	                          depth.depth_compare_op == vk::CompareOp::eEqual &&
	                          EqualTestedNothing(vs_input_info.stage.program->shader_hash);
	vk_buffer.setDepthTestEnable(depth.depth_test_enable ? VK_TRUE : VK_FALSE);
	vk_buffer.setDepthWriteEnable(
	    depth.depth_write_enable && !relax_equal && !equal_always ? VK_TRUE : VK_FALSE);
	vk_buffer.setDepthCompareOp(equal_always   ? vk::CompareOp::eAlways
	                            : !relax_equal ? depth.depth_compare_op
	                            : reversed_z   ? vk::CompareOp::eGreaterOrEqual
	                                           : vk::CompareOp::eLessOrEqual);

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	float       constant_factor   = 0.0f;
	float       slope_factor      = 0.0f;
	if (relax_equal) {
		// <units> of the primitive's depth precision (a float depth's is 2^-23 of its value, so
		// 256 units are 0.003%, 16384 are 0.2%), and its depth slope over a pixel (four pixels
		// from 4096 units): a position the two passes compute that much apart still matches.
		const float slope = relax_units >= 4096.0f ? 4.0f : 1.0f;
		constant_factor   = reversed_z ? relax_units : -relax_units;
		slope_factor      = reversed_z ? slope : -slope;
		vk_buffer.setDepthBiasEnable(VK_TRUE);
		vk_buffer.setDepthBias(constant_factor, 0.0f, slope_factor);
	} else {
		vk_buffer.setDepthBiasEnable(depth_bias_enable ? VK_TRUE : VK_FALSE);
	}
	if (depth_bias_enable && !relax_equal) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		constant_factor = ConvertPolygonOffsetConstantFactor(guest_constant_factor, poly_offset,
		                                                     depth.desc.view_info.format);
		slope_factor    = (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}
	// The bias set (a relaxed EQUAL's too); the compare op and write logged are the game's.
	LogBigDepthDraw(buffer, vs_input_info, depth, depth_bias_enable || relax_equal, constant_factor,
	                slope_factor);

	vk_buffer.setStencilTestEnable(depth.stencil_test_enable ? VK_TRUE : VK_FALSE);
	if (depth.stencil_test_enable) {
		const auto set_stencil = [&](vk::StencilFaceFlagBits face, const vk::StencilOpState& state) {
			vk_buffer.setStencilOp(face, state.failOp, state.passOp, state.depthFailOp, state.compareOp);
			vk_buffer.setStencilCompareMask(face, state.compareMask);
			vk_buffer.setStencilWriteMask(face, state.writeMask);
			vk_buffer.setStencilReference(face, state.reference);
		};
		set_stencil(vk::StencilFaceFlagBits::eFront, depth.stencil_front);
		set_stencil(vk::StencilFaceFlagBits::eBack, depth.stencil_back);
	}

	if (PipelineDynamicRasterStateEnabled()) {
		// Formerly pipeline-key fields (PipelineCache::TryGetGraphicsPipeline), from the same
		// registers and with the same rule: rect lists are drawn without culling.
		const bool        rect_list = Prospero::IsRectList(buffer.GetUserConfig().GetPrimType());
		vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
		if (!rect_list && mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
		if (!rect_list && mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		vk_buffer.setCullMode(cull_mode);
		vk_buffer.setFrontFace(mode.face ? vk::FrontFace::eClockwise
		                                 : vk::FrontFace::eCounterClockwise);
#if !defined(__APPLE__)
		// Without a depth attachment the static pipeline had no depth/stencil state at all.
		const bool bounds_test =
		    depth.depth_bounds_test_enable && rendering.depth_stencil_attachment.has_depth;
		vk_buffer.setDepthBoundsTestEnable(bounds_test ? VK_TRUE : VK_FALSE);
		if (bounds_test) {
			vk_buffer.setDepthBounds(depth.depth_min_bounds, depth.depth_max_bounds);
		}
#endif
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	if (rendering.num_color_attachments != 0) {
		vk_buffer.setColorWriteEnableEXT(rendering.num_color_attachments, enable);
	}
#endif
}

// KYTY_TRACE_IMAGES: a draw into a followed render or depth target, with its first viewport and
// final scissor as the guest set them (TextureCache::TraceTargetDraw).
static void TraceTargetDraws(TextureCache& textures, const CommandBuffer& buffer,
                             const RenderDepthInfo& depth, const RenderColorInfo* colors,
                             uint32_t color_count, const RenderState& rendering) {
	const auto& ctx     = buffer.GetRegisters();
	const auto& vp      = ctx.GetScreenViewport();
	const auto& guest   = vp.viewports[0];
	const auto  scissor = calc_final_scissor(vp, ctx.GetScanModeControl(),
	                                         vk::Extent2D {rendering.width, rendering.height}, 0);
	const auto  state =
	    fmt::format("viewport ({:.1f},{:.1f} {:.1f}x{:.1f}) scissor ({},{})-({},{}) area {}x{}{}",
	                guest.xoffset - guest.xscale, guest.yoffset - guest.yscale, guest.xscale * 2.0f,
	                guest.yscale * 2.0f, scissor.left, scissor.top, scissor.right, scissor.bottom,
	                rendering.width, rendering.height,
	                ctx.GetClipControl().clip_disable ? " (no clipping)" : "");
	if (depth.image_id) {
		const auto& meta = depth.desc.info.metadata;
		textures.TraceTargetDraw(
		    depth.image_id, "depth",
		    fmt::format(
		        "{}; depth test {} write {} op {}, clear by register {} at the pass's start "
		        "{} to {}, {}",
		        state, depth.depth_test_enable ? 1 : 0, depth.depth_write_enable ? 1 : 0,
		        static_cast<int>(depth.depth_compare_op), depth.depth_clear_enable ? 1 : 0,
		        depth.depth_load_clear_enable ? 1 : 0, depth.depth_clear_value,
		        meta.kind == ImageMetadataKind::Htile
		            ? fmt::format("HTile 0x{:012x}+0x{:x}", meta.range.address, meta.range.size)
		            : std::string("no HTile")));
	}
	for (uint32_t i = 0; i < color_count; i++) {
		if (colors[i].image_id) {
			textures.TraceTargetDraw(colors[i].image_id, "render", state);
		}
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

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
};

// Round 82 (from the GTA V fork, 12bb591a's AcquireDrawRenderState): a draw state is ~36 KB, most
// of it the input infos of the three vertex stages, and value-initializing one on the stack for
// each draw cost ~0.6 us there. Each thread's draws reuse one state, value-initialized member by
// member as `DrawRenderState state {}` would be, except the input infos of the second and third
// vertex stages, which only tessellated draws write (DescribeGraphicsStages): those are cleared
// only after a tessellated draw. A draw made while another is being made on the same thread gets
// a state of its own. KYTY_BISECT's drawstate:old value-initializes a whole state per draw.
template <typename T>
static void ValueInitialize(T& object) {
	if constexpr (!std::is_trivially_destructible_v<T>) {
		object.~T();
	}
	::new (static_cast<void*>(std::addressof(object))) T {};
}

class DrawStateLease {
public:
	explicit DrawStateLease(bool tessellated) {
		thread_local Slot slot;
		if (slot.in_use || Bisect::DrawStateOld()) {
			m_own.emplace();
			m_state = &*m_own;
			return;
		}
		slot.in_use = true;
		m_slot      = &slot;
		auto& state = slot.state;
		ValueInitialize(state.depth_info);
		for (auto& color: state.color_info) {
			ValueInitialize(color);
		}
		state.color_count = 0;
		state.ps_active   = true;
		ValueInitialize(state.vertex_info[0]);
		if (slot.tessellated) {
			ValueInitialize(state.vertex_info[1]);
			ValueInitialize(state.vertex_info[2]);
		}
		ValueInitialize(state.ps_input_info);
		ValueInitialize(state.programs);
		slot.tessellated = tessellated;
		m_state          = &state;
	}
	~DrawStateLease() {
		if (m_slot != nullptr) {
			m_slot->in_use = false;
		}
	}
	KYTY_CLASS_NO_COPY(DrawStateLease);

	[[nodiscard]] DrawRenderState& Get() { return *m_state; }

private:
	struct Slot {
		DrawRenderState state {};
		// The last draw was tessellated: it may have written the second and third stages.
		bool tessellated = false;
		bool in_use      = false;
	};
	Slot*                          m_slot  = nullptr;
	DrawRenderState*               m_state = nullptr;
	std::optional<DrawRenderState> m_own;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

// The render area must lie inside every attachment. The guest extent comes from discovery, the
// attachment is a view of a cache image: should the image be smaller (an alias of a transient
// pool recreated at another size), rendering the guest extent would write past it. Bounded to the
// image, and logged.
static vk::Extent2D ClampToBacking(const Image& image, uint32_t base_level, vk::Extent2D extent,
                                   uint64_t address, const char* kind) {
	const vk::Extent2D backing {std::max(image.info.extent.width >> base_level, 1u),
	                            std::max(image.info.extent.height >> base_level, 1u)};
	if (backing.width >= extent.width && backing.height >= extent.height) {
		return extent;
	}
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 32u) {
		LOGF("Render target: %s 0x%016" PRIx64 " is %ux%u but its image is %ux%u (level %u): "
		     "render area bounded to the image\n",
		     kind, address, extent.width, extent.height, backing.width, backing.height,
		     base_level);
	}
	return {std::min(extent.width, backing.width), std::min(extent.height, backing.height)};
}

// A depth target's HTile fill and its stencil (round 65). With the stencil's compression in the
// HTile (DB_STENCIL_INFO.TILE_STENCIL_DISABLE=0) a word holds ZMask [3:0], the stencil results
// SR0/SR1 [7:4], SMem [9:8] and the depth range [31:12]; SMem 0 is the stencil's cleared state, as
// ZMask 0 is the depth's (Mesa, radv_get_htile_fast_clear_value). The host cleared only the depth
// of a target whose HTile a fill cleared, so its stencil kept what earlier frames left in it. GT7
// clears its course maps' depth targets with one compute fill a frame and draws their track with
// a stencil test. The track went from the maps once round 63b cleared their colors every frame,
// and the cursors stayed, as a stencil left by earlier frames would have it (before that, the
// track the first frames drew stayed in the image, under the cursors' trails; round 64's "cmask
// off" brought the trails back but not the track). KYTY_HTILE_STENCIL_CLEARS=0 (or KYTY_BISECT's
// htile:depth) clears the depth alone, as before.
[[nodiscard]] static bool HtileStencilClears() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HTILE_STENCIL_CLEARS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && !Bisect::HtileDepthOnly();
}

[[nodiscard]] static constexpr uint32_t HtileZMask(uint32_t value) {
	return value & 0xfu;
}
[[nodiscard]] static constexpr uint32_t HtileSMem(uint32_t value) {
	return (value >> 8u) & 0x3u;
}

// The first HTile clears of each depth target (up to 32), with what the fill held.
static void LogHtileClear(const RenderDepthInfo& depth, std::optional<uint32_t> fill,
                          bool stencil_cleared) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> logged;
	const auto&                         info = depth.desc.info;
	{
		std::lock_guard lock(mutex);
		if (logged.size() >= 32 || !logged.insert(info.data.address).second) {
			return;
		}
	}
	const auto  fill_text    = fill.has_value()
	                               ? fmt::format("its HTile fill 0x{:08x} (ZMask {}, SMem {})", *fill,
	                                             HtileZMask(*fill), HtileSMem(*fill))
	                               : std::string("its HTile clear (the value unknown)");
	std::string stencil_text = "kept";
	if (!info.metadata.stencil_compressed) {
		stencil_text = "is not in the HTile (kept)";
	} else if (stencil_cleared) {
		stencil_text = fmt::format("too, to {}", depth.stencil_clear_value);
	} else if (fill.has_value() && HtileSMem(*fill) == 0) {
		stencil_text = "kept (KYTY_HTILE_STENCIL_CLEARS=0 or htile:depth)";
	}
	LOGF("DepthTarget: 0x%012" PRIx64 " %ux%u, %u samples, %s: the depth is cleared to %g, the "
	     "stencil %s\n",
	     info.data.address, info.extent.width, info.extent.height, info.samples, fill_text.c_str(),
	     depth.depth_clear_value, stencil_text.c_str());
}

// Vulkan resolves float and normalised formats by averaging; integer formats must resolve with
// SAMPLE_ZERO. SAMPLE_ZERO is valid for every format, so it is also the answer when the emulator
// does not classify the guest format.
static vk::ResolveModeFlagBits ColorResolveMode(Prospero::BufferFormat format) {
	switch (Prospero::SampledTextureNumericClass(format)) {
		case Prospero::TextureNumericClass::Float: return vk::ResolveModeFlagBits::eAverage;
		default: return vk::ResolveModeFlagBits::eSampleZero;
	}
}

RenderExecutor::~RenderExecutor() = default;

RenderExecutor::MsColorTarget&
RenderExecutor::GetMsColorTarget(CommandBuffer& buffer, ImageId image_id, const Image& source,
                                 uint32_t width, uint32_t height, uint32_t layers, uint32_t samples,
                                 uint32_t base_level, uint32_t base_layer, bool& created) {
	created = false;
	for (auto& target: m_ms_color_targets) {
		if (target.image_id == image_id && target.format == source.info.pixel_format &&
		    target.width == width && target.height == height && target.layers == layers &&
		    target.samples == samples && target.base_level == base_level &&
		    target.base_layer == base_layer) {
			return target;
		}
	}

	ImageInfo info {};
	info.pixel_format    = source.info.pixel_format;
	info.guest_format    = source.info.guest_format;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {width, height, 1};
	info.resources       = {1, layers};
	info.bytes_per_block = source.info.bytes_per_block;
	info.pitch           = width * source.info.bytes_per_block;
	info.samples         = samples;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.bgra16          = source.info.bgra16;

	MsColorTarget target {};
	target.image_id   = image_id;
	target.format     = info.pixel_format;
	target.width      = width;
	target.height     = height;
	target.layers     = layers;
	target.samples    = samples;
	target.base_level = base_level;
	target.base_layer = base_layer;
	target.image      = std::make_unique<Image>(buffer.GetGraphics(),
	                                            m_context.GetTextureCache().m_scheduler, info);
	ImageViewInfo view_info {};
	view_info.format      = info.pixel_format;
	view_info.type        = vk::ImageViewType::e2D;
	view_info.aspect      = vk::ImageAspectFlagBits::eColor;
	view_info.layer_count = layers;
	view_info.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	target.view           = target.image->FindView(view_info);
	EXIT_IF(target.view == nullptr);
	// ponytail: the temporaries live as long as the executor, one per guest render target shape.
	// Ceiling: a game with thousands of distinct targets keeps a few hundred MB; pool by size when
	// that shows up.
	m_ms_color_targets.push_back(std::move(target));
	created = true;
	return m_ms_color_targets.back();
}

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
	// EQAA fallback, decided exactly as pipelineCache.cpp does: 1-sample colour over multisampled
	// depth with no host mixed-sample support (RenderDoc). The colour renders into a temporary
	// image at the depth sample count and resolves onto the guest target, so depth stays attached.
	const uint32_t depth_samples = depth.image_id ? depth.desc.info.samples : 0;
	uint32_t       color_samples = 0;
	const bool     emulate_color =
	    MsColorTargetsResolvable({colors, color_count}, color_samples) &&
	    EmulatesColorAtDepthSamples(
	        m_context.GetGraphics().AcceptsMixedSamples(color_samples, depth_samples),
	        color_samples, depth_samples);
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		vk::ImageView           attachment_view   = image_view;
		vk::ImageLayout         attachment_layout = layout;
		vk::ImageView           resolve_view      = nullptr;
		vk::ImageLayout         resolve_layout    = vk::ImageLayout::eUndefined;
		vk::ResolveModeFlagBits resolve_mode      = vk::ResolveModeFlagBits::eNone;
		if (emulate_color) {
			bool  created = false;
			auto& ms = GetMsColorTarget(buffer, target.image_id, image, image.info.extent.width,
			                            image.info.extent.height, view.layer_count, depth_samples,
			                            view.base_level, view.base_layer, created);
			// The temporary carries this pass's colour so far: seed it when the draw opens a pass,
			// and when the target joins a pass that was already running without it.
			if (created || !buffer.IsRendering()) {
				cache.m_blit_helper.ReplicateColorToMs(image, *ms.image);
			}
			{
				const GuestImageUses guest_use;
				ms.image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
				                  vk::AccessFlagBits2::eColorAttachmentWrite, {},
				                  buffer.Recorder());
			}
			attachment_view   = ms.view;
			attachment_layout = vk::ImageLayout::eColorAttachmentOptimal;
			resolve_view      = image_view;
			resolve_layout    = layout;
			resolve_mode      = ColorResolveMode(image.info.guest_format);
		}
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		{
			const GuestImageUses guest_use;
			image.Transit(layout, image.binding.attachment_access,
			              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
			                                     view.layer_count},
			              buffer.Recorder());
		}
		const auto extent       = ClampToBacking(image, view.base_level, target.Extent(),
		                                         target.desc.info.data.address, "color");
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment                = state.color_attachments[target.target_slot];
		attachment.image_view           = attachment_view;
		attachment.image_layout         = attachment_layout;
		attachment.resolve_image_view   = resolve_view;
		attachment.resolve_image_layout = resolve_layout;
		attachment.resolve_mode         = resolve_mode;
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
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.guest_array_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || meta_clear;
		if (meta_clear && depth.desc.info.HasStencil()) {
			// The fill's value says whether it cleared the stencil too (HtileSMem).
			const auto fill           = cache.MetaFillValue(metadata.range.address);
			const bool clears_stencil = metadata.stencil_compressed && fill.has_value() &&
			                            HtileSMem(*fill) == 0 && HtileStencilClears();
			if (clears_stencil) {
				depth.stencil_clear_enable = true;
			}
			LogHtileClear(depth, fill, clears_stencil);
		}
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.guest_array_layer, false)) {
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
			// Without VK_EXT_attachment_feedback_loop_* (the AMD Windows driver), the draw samples
			// the depth target it writes in the GENERAL layout: not defined by Vulkan, but what
			// the hardware does anyway, and better than ending the emulator.
			static std::atomic_bool warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				LOGF("Warning: depth attachment feedback loop without host support; "
				     "using the GENERAL layout\n");
			}
		}
		auto       layout          = depth_attachment_layout(depth);
		const bool readable_layout = !(sampled_aspects & ~DepthReadableAspects(layout));
		if (!readable_layout) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		              vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		// Round 91: a draw that samples the depth target without writing what it samples reads it
		// as a texture too: that access now, as its descriptors will ask (PrepareBindings).
		// Otherwise the state went from the attachment's to the attachment's and a shader read's
		// at every such draw, a barrier each way and the rendering ended twice per draw. After
		// BryanKAdams' BryKytyPS5 (408d4e57). Only in a layout that reads the sampled aspect: in
		// the feedback-loop or GENERAL layout, a draw writing one mip while sampling another
		// would then find no barrier at all after the draw before wrote what it samples. The F8
		// step depthread:old as before.
		if (sampled_aspects && !feedback_aspects && readable_layout && !Bisect::DepthReadOld()) {
			access |= vk::AccessFlagBits2::eShaderRead;
		}
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		{
			const GuestImageUses guest_use;
			image.Transit(layout, access,
			              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
			                                     view.layer_count},
			              buffer.Recorder());
		}
		const auto depth_extent =
		    ClampToBacking(image, view.base_level,
		                   {depth.desc.info.extent.width, depth.desc.info.extent.height},
		                   depth.desc.info.data.address, "depth");
		state.width               = std::min(state.width, depth_extent.width);
		state.height              = std::min(state.height, depth_extent.height);
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
		attachment.stencil_clear  = depth.stencil_clear_enable;
		// ponytail: no mixed samples and no colour emulation: the draw renders without depth, as
		// the pipeline was built (pipelineCache.cpp). Depth tests are lost for it; emulating more
		// layouts (arrays, mips) would keep them.
		if (!m_context.GetGraphics().AcceptsMixedSamples(color_samples, depth.desc.info.samples) &&
		    !emulate_color) {
			attachment = {};
		}
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

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = 0;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if (sh_regs.target_output_mode[slot] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
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

// DB_RENDER_CONTROL.COPY_DEPTH_TO_COLOR / COPY_STENCIL_TO_COLOR: instead of a normal draw, the DB
// copies the bound depth (or stencil) samples into the bound color target; COPY_SAMPLE or
// COPY_CENTROID picks the sample. Drivers use it to resolve MSAA depth into a color surface.
// Not emulated yet: the draw is skipped, so the color target keeps its previous contents. The
// first draws are logged so the copy can be built from the states games actually use.
static bool ConsumeDepthColorCopy(const CommandBuffer& buffer) {
	const auto& hw = buffer.GetRegisters();
	const auto& rc = hw.GetRenderControl();
	if (!rc.copy_depth_to_color && !rc.copy_stencil_to_color) {
		return false;
	}
	static std::atomic_uint log_count {0};
	if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
		const auto& z    = hw.GetDepthRenderTarget();
		const auto  slot = render_target_first_bound_slot(buffer);
		const auto& rt   = hw.GetRenderTarget(slot);
		LOGF("DepthColorCopy: temporary: skipped (%s%s) sample=%u centroid=%u | depth "
		     "fmt=%u stencil_fmt=%u samples_log2=%u %ux%u read=0x%010" PRIx64 " | color slot=%u "
		     "base=0x%010" PRIx64 " fmt=%u type=%u samples_log2=%u fragments_log2=%u %ux%u "
		     "mask=0x%08" PRIx32 " cb_mode=%u\n",
		     rc.copy_depth_to_color ? "depth" : "", rc.copy_stencil_to_color ? "+stencil" : "",
		     static_cast<uint32_t>(rc.copy_sample), rc.copy_centroid ? 1u : 0u,
		     static_cast<uint32_t>(z.z_info.format), static_cast<uint32_t>(z.stencil_info.format),
		     static_cast<uint32_t>(z.z_info.num_samples), static_cast<uint32_t>(z.size.x_max) + 1u,
		     static_cast<uint32_t>(z.size.y_max) + 1u, z.z_read_base_addr, slot, rt.base.addr,
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     rt.attrib.num_samples, rt.attrib.num_fragments, rt.attrib2.width + 1u,
		     rt.attrib2.height + 1u, hw.GetRenderTargetMask(),
		     static_cast<uint32_t>(hw.GetColorControl().mode));
	}
	return true;
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
	// Nonzero: a draw whose counts the GPU reads from these guest DrawIndirectArgs.
	uint64_t gpu_args_address = 0;
	// Nonzero: a vertex-pipeline draw whose instance count the GPU reads from this guest dword.
	uint64_t gpu_instance_count_address = 0;
};

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
		// A buffer only constant-only embedded fetches refer to is never read: bind the null
		// buffer instead of acquiring whatever its unconsumed descriptor names.
		bool fetched = false;
		for (int a = 0; a < vs_input_info.resources_num && !fetched; a++) {
			fetched = vs_input_info.resources_dst[a].buffer_index == i &&
			          VertexAttributeFetched(vs_input_info, a);
		}
		const auto size = fetched ? VertexBufferDescriptorSize(i, vs_input_info) : 0;
		sizes[i]        = size;
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

// KYTY_DRAW_REPEATS=1 (round 83, DrawRepeats): how much of each frame's draws the frame before
// had too: the share a memo of a draw's prepared state could serve.
static void NoteDrawRepeat(const DrawCallInfo& draw, const DrawRenderState& state,
                           const DrawIndexBufferSource& index, size_t vertex_stage_count,
                           uint64_t image_generation) {
	if (!DrawRepeats::Enabled()) {
		return;
	}
	static DrawRepeats::Stats stats("Draw");
	const auto mix = [](uint64_t seed, const void* data, size_t size) {
		return size == 0 ? seed : XXH3_64bits_withSeed(data, size, seed);
	};
	uint64_t   inputs   = 0x5eed;
	uint64_t   resolved = 0;
	uint64_t   textures = 0x7e47;
	const auto stage    = [&](const ShaderStageRuntime& runtime) {
        if (!runtime) {
            return;
        }
        const auto  hash = runtime.program->shader_hash;
        const auto& snap = *runtime.resources;
        inputs           = mix(inputs, &hash, sizeof(hash));
        inputs   = mix(inputs, snap.user_data.data(), snap.user_data.size() * sizeof(uint32_t));
		textures         = mix(textures, &hash, sizeof(hash));
		resolved = mix(resolved, snap.buffers.data(),
		                  snap.buffers.size() * sizeof(ShaderRecompiler::IR::DescriptorValue));
        resolved = mix(resolved, snap.images.data(),
		                  snap.images.size() * sizeof(ShaderRecompiler::IR::DescriptorValue));
        resolved = mix(resolved, snap.samplers.data(),
		                  snap.samplers.size() * sizeof(ShaderRecompiler::IR::DescriptorValue));
        resolved =
            mix(resolved, snap.flattened_srt.data(), snap.flattened_srt.size() * sizeof(uint32_t));
		textures = mix(textures, snap.images.data(),
		               snap.images.size() * sizeof(ShaderRecompiler::IR::DescriptorValue));
		textures = mix(textures, snap.samplers.data(),
		               snap.samplers.size() * sizeof(ShaderRecompiler::IR::DescriptorValue));
	};
	for (size_t i = 0; i < vertex_stage_count; i++) {
		stage(state.vertex_info[i].stage);
	}
	if (state.ps_active) {
		stage(state.ps_input_info.stage);
	}
	std::array<uint64_t, 8> targets {};
	for (uint32_t i = 0; i < state.color_count && i < 6u; i++) {
		targets[i] = state.color_info[i].desc.info.data.address;
	}
	targets[6]     = state.depth_info.image_id ? state.depth_info.desc.info.data.address : 0u;
	textures       = mix(textures, targets.data(), sizeof(uint64_t) * 7u);
	targets[7]     = (static_cast<uint64_t>(draw.index_count) << 32u) | draw.instance_count;
	resolved       = mix(resolved, targets.data(), sizeof(targets));
	resolved       = mix(resolved, &index.address, sizeof(index.address));
	const auto& vs = state.vertex_info[0];
	for (int b = 0; b < vs.buffers_num; b++) {
		const std::array<uint64_t, 2> vb {vs.buffers[b].addr,
		                                  (static_cast<uint64_t>(vs.buffers[b].stride) << 32u) |
		                                      vs.buffers[b].num_records};
		resolved = mix(resolved, vb.data(), sizeof(vb));
	}
	resolved ^= inputs * 0x9e3779b97f4a7c15ull;
	stats.Note(FrameGraph::GpuThreadFrames(), inputs, resolved, textures, image_generation);
}

// Round 158: the draws whose sample coverage the guest narrows in ways the host left out or
// only partly reproduces (ghost cars that drew solid, issue 30): an AA sample mask
// (PA_SC_AA_MASK) clearing samples of a pixel of the 2x2 quad (applied since round 159, by a
// copy of the pixel shader: spirvCoverageMask.h), alpha to coverage
// (applied, without DB_ALPHA_TO_MASK's dither offsets) and coverage the pixel shader exports. A
// "Coverage (10 s):" line counts them; the first draws of each kind with a pixel shader not seen
// in that kind yet are logged ("Coverage:" lines) with their samples and blending.
static void NoteCoverage(const CommandBuffer& buffer, const DrawRenderState& state) {
	static std::atomic<uint64_t> draws {0}, partial_masks {0}, alpha_draws {0}, export_draws {0};
	static std::atomic<int64_t>  last_report {0};
	const auto&                  registers = buffer.GetRegisters();
	const auto&                  shader    = registers.GetShaderRegisters();
	const auto&                  config    = registers.GetAaConfig();
	const uint32_t               samples   = 1u << std::min<uint32_t>(config.msaa_num_samples, 4u);
	const uint32_t               full      = samples >= 16u ? 0xffffu : (1u << samples) - 1u;
	const std::array masks = {registers.GetAaMask(0) & 0xffffu, registers.GetAaMask(0) >> 16u,
	                          registers.GetAaMask(1) & 0xffffu, registers.GetAaMask(1) >> 16u};
	const bool       masked =
	    std::ranges::any_of(masks, [&](uint32_t mask) { return (mask & full) != full; });
	const bool alpha    = state.ps_active && shader.AlphaToMask();
	const bool exported = state.ps_active && shader.db_shader_control.shader_mask_export_enable;
	const auto count    = draws.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (masked) partial_masks.fetch_add(1, std::memory_order_relaxed);
	if (alpha) alpha_draws.fetch_add(1, std::memory_order_relaxed);
	if (exported) export_draws.fetch_add(1, std::memory_order_relaxed);
	if ((count & 1023u) == 0u) {
		const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
		                     std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
		auto last = last_report.load(std::memory_order_relaxed);
		if (last == 0) {
			last_report.compare_exchange_strong(last, now, std::memory_order_relaxed);
		} else if (now - last >= 10000 &&
		           last_report.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
			const auto total    = draws.exchange(0, std::memory_order_relaxed);
			const auto masked_n = partial_masks.exchange(0, std::memory_order_relaxed);
			const auto alpha_n  = alpha_draws.exchange(0, std::memory_order_relaxed);
			const auto export_n = export_draws.exchange(0, std::memory_order_relaxed);
			if (masked_n + alpha_n + export_n != 0) {
				LOGF("Coverage (10 s): %" PRIu64 " draws: %" PRIu64
				     " with an AA mask clearing samples, %" PRIu64
				     " with alpha to coverage, %" PRIu64 " exporting coverage (round 158)\n",
				     total, masked_n, alpha_n, export_n);
			}
		}
	}
	if (!masked && !alpha && !exported) {
		return;
	}
	const uint64_t                      ps = state.ps_active && state.ps_input_info.stage
	                                             ? state.ps_input_info.stage.program->shader_hash
	                                             : 0;
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	static uint32_t                     logged = 0;
	{
		std::lock_guard lock(mutex);
		// A shader's AA masks as well: a dither pattern that moves from frame to frame shows.
		const uint64_t kinds = (masked ? 1u : 0u) | (alpha ? 2u : 0u) | (exported ? 4u : 0u);
		const uint64_t key =
		    (ps * 8u + kinds) ^
		    (masked ? (uint64_t {registers.GetAaMask(0)} << 32u | registers.GetAaMask(1)) *
		                  0x9e3779b97f4a7c15ull
		            : 0u);
		if (logged >= 96u || !seen.insert(key).second) {
			return;
		}
		logged++;
	}
	const auto& eqaa        = registers.GetEqaaControl();
	const auto& color       = registers.GetBlendColor();
	const auto  slot        = state.color_count > 0 ? state.color_info[0].target_slot : 0u;
	const auto& blend       = registers.GetBlendControl(slot);
	const auto  color_count = state.color_count;
	LOGF(
	    "Coverage:%s%s%s ps %016" PRIx64 " vs %016" PRIx64 " frame %" PRIu64
	    ": AA mask %04x %04x %04x %04x of %u samples (exposed %u), host samples "
	    "color %u depth %u, alpha to mask %d (DB_ALPHA_TO_MASK 0x%08x), mask export %d, EQAA "
	    "samples mask export %u alpha %u iter %u, %u targets, slot %u blend %d (%u %u %u, alpha %u "
	    "%u %u) write mask 0x%x, constant %g %g %g %g\n",
	    masked ? " aa-mask" : "", alpha ? " alpha" : "", exported ? " export" : "", ps,
	    state.vertex_info[0].stage.program->shader_hash, FrameGraph::GpuThreadFrames(), masks[0],
	    masks[1], masks[2], masks[3], samples,
	    1u << std::min<uint32_t>(config.msaa_exposed_samples, 4u),
	    color_count > 0 ? state.color_info[0].desc.info.samples : 0u,
	    state.depth_info.image_id ? state.depth_info.desc.info.samples : 0u,
	    shader.AlphaToMask() ? 1 : 0, registers.GetAlphaToMaskValue(),
	    shader.db_shader_control.shader_mask_export_enable ? 1 : 0,
	    static_cast<uint32_t>(eqaa.mask_export_num_samples),
	    static_cast<uint32_t>(eqaa.alpha_to_mask_num_samples),
	    static_cast<uint32_t>(eqaa.ps_iter_samples), color_count, slot, blend.enable ? 1 : 0,
	    static_cast<uint32_t>(blend.color_srcblend), static_cast<uint32_t>(blend.color_comb_fcn),
	    static_cast<uint32_t>(blend.color_destblend), static_cast<uint32_t>(blend.alpha_srcblend),
	    static_cast<uint32_t>(blend.alpha_comb_fcn), static_cast<uint32_t>(blend.alpha_destblend),
	    (registers.GetRenderTargetMask() >> (slot * 4u)) & 0xfu, color.red, color.green, color.blue,
	    color.alpha);
}

// KYTY_OP_STATS=1 (round 84): the draw's group in the "Draw targets" line, its first color target
// (its depth target when it has none): the passes the GPU thread's draw time goes to.
static void NoteDrawTarget(const DrawRenderState& state) {
	if (!GpuPhases::OpStatsOn()) {
		return;
	}
	const ImageInfo* info  = nullptr;
	bool             depth = false;
	if (state.color_count > 0 && state.color_info[0].desc.info.data.address != 0) {
		info = &state.color_info[0].desc.info;
	} else if (state.depth_info.image_id) {
		info  = &state.depth_info.desc.info;
		depth = true;
	}
	if (info == nullptr || info->data.address == 0) {
		return;
	}
	const uint64_t extent = (info->extent.width & 0xffffu) |
	                        (uint64_t {info->extent.height & 0xffffu} << 16u) |
	                        (uint64_t {info->resources.layers} << 32u);
	GpuPhases::SetOpKey(info->data.address | (depth ? uint64_t {1} << 63u : 0u), extent);
}

// KYTY_CUBE_EVERY=<n> (round 85; KYTY_BISECT's cube:<n>): the draws into a reflection cube run
// every nth frame only (n from 1, every frame, the default, to 8). The game draws the scene into
// a 512x512 cube of six layers every frame for its cars' reflections: 550 to 750 draws a frame,
// 30 to 40 % of the GPU thread's draw time (round 84's KYTY_OP_STATS). On the other frames the
// cube keeps the last frame's pictures. A cube is a color target of six square layers of at most
// 2048 pixels; its address is learned at its first draw, and a draw is left out while that
// address is its first color target and the draw writes it. GPU thread only.
namespace {
std::array<uint64_t, 4> g_cube_targets {};
uint32_t                g_cube_target_count = 0;
uint64_t                g_cube_draws_left   = 0;
} // namespace

static uint32_t CubeEvery() {
	if (const auto step = Bisect::CubeEvery(); step != 0) {
		return step;
	}
	static const uint32_t every = [] {
		const char* value = std::getenv("KYTY_CUBE_EVERY");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 1ul;
		return static_cast<uint32_t>(std::clamp(n, 1ul, 8ul));
	}();
	return every;
}

static void LearnCubeTarget(CommandBuffer& buffer, const DrawRenderState& state) {
	if (state.color_count == 0 || g_cube_target_count >= g_cube_targets.size()) {
		return;
	}
	const auto& color = state.color_info[0];
	const auto& info  = color.desc.info;
	if (color.target_slot != 0 || info.resources.layers != 6 ||
	    info.extent.width != info.extent.height || info.extent.width > 2048u) {
		return;
	}
	const auto address = buffer.GetRegisters().GetRenderTarget(0).base.addr;
	const auto known   = std::span(g_cube_targets.data(), g_cube_target_count);
	if (address == 0 || std::ranges::find(known, address) != known.end()) {
		return;
	}
	g_cube_targets[g_cube_target_count++] = address;
	// Round 87: its draws may be skipped while their shaders translate or pipelines compile.
	NoteReflectionCubeTarget(address);
	LOGF("Reflection cube: color target 0x%" PRIx64 ", %ux%u, 6 layers: its draws run every "
	     "frame, or every nth with KYTY_CUBE_EVERY=<n> (now %u)\n",
	     address, info.extent.width, info.extent.height, CubeEvery());
}

static bool SkipCubeDraw(CommandBuffer& buffer) {
	if (g_cube_target_count == 0) {
		return false;
	}
	const auto every = CubeEvery();
	if (every <= 1 || FrameGraph::GpuThreadFrames() % every == 0) {
		return false;
	}
	const auto& regs    = buffer.GetRegisters();
	const auto  address = regs.GetRenderTarget(0).base.addr;
	const auto  known   = std::span(g_cube_targets.data(), g_cube_target_count);
	if (address == 0 || std::ranges::find(known, address) == known.end() ||
	    (DrawColorOutputMask(regs) & 1u) == 0) {
		return false;
	}
	const auto left = ++g_cube_draws_left;
	if ((left & (left - 1u)) == 0 && left >= 1024u) {
		LOGF("Reflection cube: %" PRIu64 " draws left out so far (KYTY_CUBE_EVERY=%u)\n", left,
		     every);
	}
	return true;
}

// KYTY_DUMP_TESS_DRAWS="<hash>[,<hash>...]" (round 82, the roof spike): the inputs of the first
// tessellated draws (up to 4 distinct index buffers per hash) whose stages include one of these
// shaders, as guest memory holds them: the index buffer, each vertex buffer the first stage
// fetches from (with its V#s), and each buffer every vertex stage binds, into dump_tess_*.bin
// next to the log, with "TessDump:" lines naming them (and whether the GPU wrote the bytes, in
// which case guest memory may be older than what the GPU reads).
static void DumpTessellatedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
                                const DrawRenderState& state, const DrawIndexBufferSource& index,
                                std::span<PreparedBindings> stage_bindings) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		if (const char* value = std::getenv("KYTY_DUMP_TESS_DRAWS"); value != nullptr) {
			for (const char* cursor = value; *cursor != '\0';) {
				char*      end  = nullptr;
				const auto hash = std::strtoull(cursor, &end, 16);
				if (end == cursor) {
					break;
				}
				list.push_back(hash);
				cursor = *end == ',' ? end + 1 : end;
			}
		}
		return list;
	}();
	if (hashes.empty()) {
		return;
	}
	const auto stage_count = state.programs.VertexStageCount();
	uint64_t   matched     = 0;
	for (uint32_t i = 0; i < stage_count && matched == 0; i++) {
		const auto hash = state.vertex_info[i].stage.program->shader_hash;
		if (std::find(hashes.begin(), hashes.end(), hash) != hashes.end()) {
			matched = hash;
		}
	}
	if (matched == 0) {
		return;
	}
	// GPU thread only: the (hash, index buffer, count) dumped so far.
	static std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> dumped;
	const auto key = std::make_tuple(matched, index.address, draw.index_count);
	if (std::find(dumped.begin(), dumped.end(), key) != dumped.end() ||
	    std::count_if(dumped.begin(), dumped.end(),
	                  [&](const auto& entry) { return std::get<0>(entry) == matched; }) >= 4) {
		return;
	}
	dumped.push_back(key);
	const auto           number = dumped.size();
	auto&                cache  = buffer.GetContext().GetBufferCache();
	const auto           frame  = FrameGraph::GpuThreadFrames();
	std::vector<uint8_t> bytes;
	const auto           write = [&](const std::string& name, uint64_t address, uint64_t size,
                           const void* host) -> std::string {
        constexpr uint64_t MaxBytes = 32ull << 20u;
        size                        = std::min(size, MaxBytes);
        if (size == 0) {
            return "(empty)";
        }
        bytes.resize(size);
        if (host != nullptr) {
            std::memcpy(bytes.data(), host, size);
        } else if (!Libs::LibKernel::Memory::TryReadBacking(address, bytes.data(), size)) {
            return "(not readable)";
        }
        const auto path = fmt::format("dump_tess_{:016x}_{}_{}.bin", matched, number, name);
        if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
            std::fwrite(bytes.data(), 1, bytes.size(), file);
            std::fclose(file);
        }
        const bool gpu_written = host == nullptr && cache.IsRegionGpuModified(address, size);
        return fmt::format("{} ({} bytes{})", path, size, gpu_written ? ", GPU-written" : "");
	};
	const auto& ucfg = buffer.GetUserConfig();
	LOGF("TessDump: draw %zu of shader %016" PRIx64 " (frame %" PRIu64 "): %u indices x %u, index "
	     "offset %u, %u control points, primitive type %u\n",
	     number, matched, frame, draw.index_count, draw.instance_count, ucfg.GetIndexOffset(),
	     (buffer.GetRegisters().GetShaderRegisters().m_vgtLsHsConfig >> 8u) & 0x3fu,
	     static_cast<uint32_t>(ucfg.GetPrimType()));
	LOGF("TessDump:   vgt_ls_hs_config 0x%08x, vgt_tf_param 0x%08x\n",
	     buffer.GetRegisters().GetShaderRegisters().m_vgtLsHsConfig,
	     buffer.GetRegisters().GetShaderRegisters().m_vgtTfParam);
	const auto index_size =
	    static_cast<uint64_t>(draw.index_count) * std::max(index.guest_element_size, 1u);
	LOGF("TessDump:   indices at 0x%016" PRIx64 ", %u bytes each%s: %s\n", index.address,
	     index.guest_element_size, index.host_data != nullptr ? " (expanded on the host)" : "",
	     write("indices", index.address, index_size, index.host_data).c_str());
	for (uint32_t i = 0; i < stage_count; i++) {
		const auto& info    = state.vertex_info[i];
		const auto* program = info.stage.program;
		// (Each stage's user data is in its shader-dumps/<stage>_<hash>.txt.)
		LOGF("TessDump:   stage %u: type %d, shader %016" PRIx64 "\n", i,
		     static_cast<int>(program->stage), program->shader_hash);
		for (int b = 0; b < info.buffers_num; b++) {
			const auto& vb = info.buffers[b];
			LOGF("TessDump:     vertex buffer %d: 0x%016" PRIx64 ", stride %u, %u records: %s\n", b,
			     vb.addr, vb.stride, vb.num_records,
			     write(fmt::format("s{}_vb{}", i, b), vb.addr,
			           static_cast<uint64_t>(vb.stride) * vb.num_records, nullptr)
			         .c_str());
		}
		for (int r = 0; r < info.resources_num; r++) {
			const auto& v = info.resources[r];
			LOGF("TessDump:     fetch %d (buffer %d, registers %d+%d, attribute %d): V# "
			     "%08x %08x %08x %08x\n",
			     r, info.resources_dst[r].buffer_index, info.resources_dst[r].register_start,
			     info.resources_dst[r].registers_num, info.resources_dst[r].attr_id, v.fields[0],
			     v.fields[1], v.fields[2], v.fields[3]);
		}
		if (i < stage_bindings.size()) {
			const auto& sources = stage_bindings[i].buffer_sources;
			for (size_t b = 0; b < sources.size(); b++) {
				LOGF("TessDump:     bound buffer %zu: 0x%016" PRIx64 ", 0x%" PRIx64 " bytes: %s\n",
				     b, sources[b].address, sources[b].size,
				     write(fmt::format("s{}_buf{}", i, b), sources[b].address, sources[b].size,
				           nullptr)
				         .c_str());
			}
		}
	}
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
	// Later phases name the draw's shaders too (the first is annotated once they are chosen).
	buffer.CarryDiagnosticAnnotation();
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

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
			// KYTY_BISECT's tess:off leaves the tessellated draws out.
			if (!Config::TessellationEnabled() || Bisect::TessOff()) return false;
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

// Whether `count` indices of `element_size` bytes hold `value`: blocks of 256 without an early exit
// inside, which the compiler vectorizes.
static bool HoldsIndex(const uint8_t* bytes, uint64_t count, uint32_t element_size,
                       uint32_t value) {
	constexpr uint64_t Block = 256;
	for (uint64_t at = 0; at < count; at += Block) {
		const uint64_t n   = std::min(Block, count - at);
		bool           hit = false;
		if (element_size == 2) {
			const auto wanted = static_cast<uint16_t>(value);
			for (uint64_t i = 0; i < n; i++) {
				uint16_t index = 0;
				std::memcpy(&index, bytes + (at + i) * 2u, 2);
				hit |= index == wanted;
			}
		} else {
			for (uint64_t i = 0; i < n; i++) {
				uint32_t index = 0;
				std::memcpy(&index, bytes + (at + i) * 4u, 4);
				hit |= index == value;
			}
		}
		if (hit) {
			return true;
		}
	}
	return false;
}

// RewriteListReset: a list draw whose indices hold the reset index, left as it is because its
// vertex stage runs as a mesh shader (which reads the guest's indices itself). Logged: how often
// that leaves a spike in place shows in the logs.
static uint32_t LeaveMeshListReset(uint64_t address, uint32_t restart) {
	static std::atomic<uint64_t> left {0};
	const auto                   n = left.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 4 || std::has_single_bit(n)) {
		LOGF("Draw: an indexed list at 0x%016" PRIx64 " holds the primitive reset index 0x%x, but "
		     "its vertex stage runs as a mesh shader, which reads the guest's indices: left as it "
		     "is (%" PRIu64 " such draws so far)\n",
		     address, restart, n);
	}
	return UINT32_MAX;
}

// RewriteListReset: one more draw drawn with rewritten indices; logged at powers of two from 1024
// (the first ones have lines of their own). Returns the count.
static uint64_t CountListResetDraw() {
	static std::atomic<uint64_t> draws {0};
	const auto                   n = draws.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n >= 1024 && std::has_single_bit(n)) {
		LOGF("Draw: %" PRIu64 " list draws drawn without the primitives their primitive reset "
		     "indices drop\n",
		     n);
	}
	return n;
}

// Round 80: an indexed list draw (points, lines, triangles, tessellation patches) with the guest's
// primitive reset on, whose indices hold the reset index. The console's primitive assembly treats
// it in lists as in strips (GE_MULTI_PRIM_IB_RESET_EN, as OpenGL's primitive restart): the
// primitive it falls in is dropped and the next one starts after it. Vulkan restarts strips and
// fans only, so such an index was drawn as a vertex: 65535 (or 0xffffffff) plus the draw's base, a
// vertex past the mesh, whose position the vertex shader fetches from whatever lies there. A
// suspect for the spike on some cars' glass roofs (2026-10-17). The indices are drawn rewritten
// without the dropped primitives, from the stream buffer, with the count of those kept. A draw
// whose vertex stage runs as a mesh shader (a merged geometry stage) reads its indices from guest
// memory itself and is left as it is (logged).
//
// Returns the count of indices kept (written to `rewritten`, in the guest's element size), or
// UINT32_MAX when the draw is left as it is. KYTY_LIST_RESET=0 or KYTY_BISECT's listreset:off
// leaves every draw as it is.
static uint32_t RewriteListReset(const CommandBuffer& buffer, BufferCache& cache,
                                 const DrawIndexBufferSource& source, uint32_t index_count,
                                 bool mesh, std::vector<uint8_t>& rewritten) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LIST_RESET");
		return value == nullptr || value[0] != '0';
	}();
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	if (!enabled || (control & 0x1u) == 0 || Bisect::ListResetOff()) {
		return UINT32_MAX;
	}
	const bool patches = buffer.GetUserConfig().GetPrimType() == Prospero::PrimitiveType::kPatch;
	uint32_t   per_primitive = 0;
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kPointList: per_primitive = 1; break;
		case Prospero::PrimitiveType::kLineList: per_primitive = 2; break;
		case Prospero::PrimitiveType::kTriList: per_primitive = 3; break;
		case Prospero::PrimitiveType::kPatch:
			// A tessellated draw's patches (the car bodies are drawn so): their input control
			// points, as the tessellation setup reads them (VGT_LS_HS_CONFIG.HS_NUM_INPUT_CP).
			per_primitive =
			    (buffer.GetRegisters().GetShaderRegisters().m_vgtLsHsConfig >> 8u) & 0x3fu;
			if (per_primitive == 0 || per_primitive > 32) {
				return UINT32_MAX;
			}
			break;
		default: return UINT32_MAX;
	}
	const auto element_size = source.guest_element_size;
	// Larger lists than this are left as they are: the stream buffer the rewritten indices go
	// through takes at most 64 MB at a time.
	constexpr uint64_t MaxBytes = 16ull << 20u;
	if ((element_size != 2 && element_size != 4) || source.address == 0 || source.size == 0 ||
	    source.size > MaxBytes || source.host_data != nullptr || index_count == 0) {
		return UINT32_MAX;
	}
	const uint32_t index_mask = element_size == 2 ? 0xffffu : 0xffffffffu;
	const auto     reset      = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset & ~index_mask) != 0) {
		return UINT32_MAX; // matches no index of this size
	}
	const uint32_t          restart = reset & index_mask;
	static std::atomic_bool noted {false};
	if (!noted.exchange(true, std::memory_order_relaxed)) {
		LOGF("Draw: the guest's primitive reset is on for indexed list draws (reset index 0x%x, "
		     "control 0x%x): a list whose indices hold it drops the primitive it falls in "
		     "(round 80; KYTY_LIST_RESET=0 draws it as a vertex, as before)\n",
		     reset, control);
	}

	// What each index buffer held when it was last looked at. One without the reset index is
	// looked at again once its pages are CPU-dirty (written since their last upload), and a
	// quarter of a second after it was last looked at anyway: an upload of a neighbour on its
	// pages takes their CPU-dirty mark away. One with it keeps the bytes it held (and the indices
	// it was drawn with), used again while it holds the same bytes.
	struct Key {
		uint64_t address;
		uint64_t size;
		uint32_t element_size;
		uint32_t restart;
		uint32_t per_primitive;
		bool     operator==(const Key&) const = default;
	};
	struct KeyHash {
		size_t operator()(const Key& key) const noexcept {
			return static_cast<size_t>((key.address * 0x9e3779b97f4a7c15ull) ^ (key.size << 7u) ^
			                           key.restart ^ (uint64_t {key.per_primitive} << 48u) ^
			                           (static_cast<uint64_t>(key.element_size) << 60u));
		}
	};
	struct Seen {
		bool                                  holds_reset     = false;
		bool                                  rewritten_valid = false; // `rewritten` made
		std::chrono::steady_clock::time_point scanned {};
		std::vector<uint8_t>                  original;  // holds_reset: the guest's bytes
		std::vector<uint8_t>                  rewritten; // holds_reset: the indices kept
	};
	constexpr auto RescanAfter = std::chrono::milliseconds(250);
	// The bytes those entries keep, bounded: past the bound they all start again.
	constexpr uint64_t                                  MaxCachedBytes = 64ull << 20u;
	thread_local std::unordered_map<Key, Seen, KeyHash> seen;
	thread_local uint64_t                               cached_bytes = 0;
	const Key  key {source.address, source.size, element_size, restart, per_primitive};
	auto       found = seen.find(key);
	const auto now   = std::chrono::steady_clock::now();
	if (found != seen.end() && !found->second.holds_reset &&
	    now - found->second.scanned < RescanAfter &&
	    !cache.IsRegionCpuModified(source.address, source.size)) {
		return UINT32_MAX;
	}

	// The guest's bytes, as the console reads them: not when the GPU wrote them (they would wait
	// for a download), nor across mappings.
	const auto* bytes = LibKernel::Memory::FindGpuCleanBacking(source.address, source.size);
	if (bytes == nullptr) {
		return UINT32_MAX;
	}
	const uint64_t count      = std::min<uint64_t>(index_count, source.size / element_size);
	const uint64_t byte_count = count * element_size;
	if (found != seen.end() && found->second.holds_reset &&
	    found->second.original.size() == byte_count &&
	    std::memcmp(found->second.original.data(), bytes, byte_count) == 0) {
		if (mesh) {
			return LeaveMeshListReset(source.address, restart);
		}
		if (found->second.rewritten_valid) {
			rewritten = found->second.rewritten;
			(void)CountListResetDraw();
			return static_cast<uint32_t>(rewritten.size() / element_size);
		}
	}
	if (seen.size() >= 65536 || cached_bytes > MaxCachedBytes) {
		seen.clear();
		cached_bytes = 0;
	}
	auto& entry = seen[key];
	cached_bytes -=
	    std::min<uint64_t>(cached_bytes, entry.original.size() + entry.rewritten.size());
	// Their memory goes too (a cleared vector keeps it).
	std::vector<uint8_t>().swap(entry.original);
	std::vector<uint8_t>().swap(entry.rewritten);
	entry.rewritten_valid = false;
	entry.scanned         = now;
	entry.holds_reset     = HoldsIndex(bytes, count, element_size, restart);
	if (!entry.holds_reset) {
		return UINT32_MAX;
	}
	entry.original.assign(bytes, bytes + byte_count);
	cached_bytes += entry.original.size();
	if (mesh) {
		return LeaveMeshListReset(source.address, restart);
	}

	// The console's assembly: a reset index drops the primitive it falls in and the next starts
	// after it. A primitive's indices are consecutive in the guest's (no reset index among them):
	// each one complete is copied whole.
	const auto read = [&](uint64_t i) -> uint32_t {
		if (element_size == 2) {
			uint16_t value = 0;
			std::memcpy(&value, bytes + i * 2u, 2);
			return value;
		}
		uint32_t value = 0;
		std::memcpy(&value, bytes + i * 4u, 4);
		return value;
	};
	rewritten.resize(byte_count);
	uint64_t kept      = 0;
	uint64_t resets    = 0;
	uint64_t filled    = 0;
	uint64_t first_hit = UINT64_MAX;
	for (uint64_t i = 0; i < count; i++) {
		if (read(i) == restart) {
			first_hit = std::min(first_hit, i);
			resets++;
			filled = 0;
			continue;
		}
		if (++filled < per_primitive) {
			continue;
		}
		std::memcpy(rewritten.data() + kept * element_size,
		            bytes + (i + 1 - per_primitive) * element_size, per_primitive * element_size);
		kept += per_primitive;
		filled = 0;
	}
	rewritten.resize(kept * element_size);
	entry.rewritten       = rewritten;
	entry.rewritten_valid = true;
	cached_bytes += entry.rewritten.size();

	const auto                   draw_count = CountListResetDraw();
	static std::atomic<uint64_t> lists {0};
	const auto                   n = lists.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 16 || std::has_single_bit(n)) {
		const auto& shaders = buffer.GetShaders();
		const auto  kind    = patches ? fmt::format("patch ({} control points)", per_primitive)
		                      : per_primitive == 3 ? std::string("triangle")
		                      : per_primitive == 2 ? std::string("line")
		                                           : std::string("point");
		LOGF("Draw: an indexed %s list at 0x%016" PRIx64 " holds the primitive reset index 0x%x "
		     "%" PRIu64 " times in %" PRIu64 " indices (the first at %" PRIu64
		     "): drawn with the %" PRIu64 " indices of the primitives it keeps, as the console "
		     "assembles them (vertex shader at 0x%016" PRIx64 ", pixel shader at 0x%016" PRIx64
		     "; %" PRIu64 " such index lists so far, %" PRIu64 " draws)\n",
		     kind.c_str(), source.address, restart, resets, count, first_hit, kept,
		     static_cast<uint64_t>(patches ? shaders.GetVs().ls_regs.data_addr
		                                   : shaders.GetVs().es_regs.data_addr),
		     static_cast<uint64_t>(shaders.GetPs().ps_regs.data_addr), n, draw_count);
	}
	return static_cast<uint32_t>(kept);
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
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	const GpuPhases::Scope phase(GpuPhases::Phase::DrawPrograms);
	std::optional<GpuPhases::Scope> tess_phase;
	if (buffer.GetUserConfig().GetPrimType() == Prospero::PrimitiveType::kPatch) {
		tess_phase.emplace(GpuPhases::Phase::TessPrograms);
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info);
}

bool DrawPrepHasVertexShader(const HW::Shader& sh_ctx) {
	return DrawHasValidVertexShader(sh_ctx);
}

bool DrawPrepPixelState(const HW::Context& ctx, const HW::Shader& sh_ctx,
                        std::array<Prospero::ColorComponentMapping, 8>& target_export_mapping) {
	static_assert(RENDER_COLOR_ATTACHMENTS_MAX == 8);
	const auto& shader_regs       = ctx.GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(ctx);
	const bool  ps_active         = sh_ctx.GetPs().ps_regs.data_addr != 0 &&
	                       (color_output_mask != 0 ||
	                        PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	target_export_mapping = {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	return ps_active;
}

// KYTY_LOG_DRAWS=<first_frame>:<count>: from that frame on, log count draws with their shader
// hashes, counts and whether their programs exist (-1: not resolved, the draw is empty). A census
// of what a scene submits.
static void LogDrawCensusLine(CommandBuffer& buffer, const char* kind, uint32_t count,
                              uint32_t instances, int vs_program, int ps_program, int ps_active) {
	static const auto setting = [] {
		struct Setting {
			uint64_t first_frame = 0;
			uint32_t count       = 0;
		} result;
		if (const char* value = std::getenv("KYTY_LOG_DRAWS"); value != nullptr) {
			char* end          = nullptr;
			result.first_frame = std::strtoull(value, &end, 10);
			if (end != nullptr && *end == ':') {
				result.count = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
			}
		}
		return result;
	}();
	static std::atomic<uint32_t> used {0};
	if (setting.count == 0) {
		return;
	}
	const auto frame = static_cast<uint64_t>(buffer.GetContext().GetGpu().GetFrameNum());
	// KYTY_LOG_DRAWS_TRIGGER=<file>: also wait until that file exists (checked once per frame), so
	// the census can start at a screen reached by hand.
	static const char*           trigger    = std::getenv("KYTY_LOG_DRAWS_TRIGGER");
	static std::atomic<uint64_t> checked    = UINT64_MAX;
	static std::atomic_bool      triggered  = trigger == nullptr;
	if (!triggered && checked.exchange(frame) != frame) {
		if (FILE* file = std::fopen(trigger, "rb"); file != nullptr) {
			std::fclose(file);
			triggered = true;
			LOGF("Draw census: triggered at frame %" PRIu64 "\n", frame);
		}
	}
	if (!triggered || frame < setting.first_frame ||
	    used.fetch_add(1, std::memory_order_relaxed) >= setting.count) {
		return;
	}
	const auto& sh_ctx = buffer.GetShaders();
	const auto  hash   = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
	LOGF("Draw census: frame=%" PRIu64 " %s es=%016" PRIx64 " gs=%016" PRIx64 " ps=%016" PRIx64
	     " count=%u instances=%u vs_program=%d ps_program=%d ps_active=%d\n",
	     frame, kind, hash(sh_ctx.GetVs().es_regs.data_addr),
	     hash(sh_ctx.GetVs().gs_regs.data_addr), hash(sh_ctx.GetPs().ps_regs.data_addr), count,
	     instances, vs_program, ps_program, ps_active);
}

static void LogDrawCensus(CommandBuffer& buffer, const DrawCallInfo& draw,
                          const DrawRenderState& state) {
	LogDrawCensusLine(buffer, draw.Name(), draw.index_count, draw.instance_count,
	                  state.programs.vertex[0] ? 1 : 0, state.programs.pixel ? 1 : 0,
	                  state.ps_active ? 1 : 0);
}

// Diagnostics: the draw parameters of the watched shader's draws in the watched frame (see
// InWatchedShaderFrame); each line precedes that draw's WatchShader bindings.
void RenderExecutor::LogWatchedDraw(const DrawCallInfo& draw, const DrawRenderState& state,
                                    uint32_t index_offset, int32_t base_vertex,
                                    uint32_t first_vertex, int32_t host_vertex_offset,
                                    uint32_t host_first_instance, bool indirect) {
	static std::atomic<uint32_t> logged {0};
	const auto*                  program = state.vertex_info[0].stage.program;
	if (program == nullptr ||
	    !InWatchedShaderFrame(program->shader_hash,
	                          m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
	                          logged)) {
		return;
	}
	LOGF("WatchShader draw: %s count=%u instances=%u first_instance=%u index_offset=%u "
	     "base_vertex=%d first_vertex=%u host_vertex_offset=%d host_first_instance=%u "
	     "indirect=%d\n",
	     draw.Name(), draw.index_count, draw.instance_count, draw.first_instance, index_offset,
	     base_vertex, first_vertex, host_vertex_offset, host_first_instance, indirect ? 1 : 0);
}

// Diagnostics: draws into 2_10_10_10 color targets. GT7 renders to one when a race is left, and
// the device faults soon after (2026-10-13): the first 16 are logged; KYTY_SKIP_10BIT_TARGETS=1
// skips them (true), to see whether the fault follows them.
static bool TenBitTargetDraw(const DrawCallInfo& draw, const DrawRenderState& state) {
	static const bool skip = [] {
		const char* value = std::getenv("KYTY_SKIP_10BIT_TARGETS");
		return value != nullptr && value[0] == '1';
	}();
	for (uint32_t i = 0; i < state.color_count; i++) {
		const auto& color  = state.color_info[i];
		const auto  format = color.desc.info.guest_format;
		if (format < Prospero::BufferFormat::k2_10_10_10UNorm ||
		    format > Prospero::BufferFormat::k2_10_10_10SInt) {
			continue;
		}
		static std::atomic<uint32_t> seen {0};
		const auto                   count = seen.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (count <= 16u || (count & (count - 1u)) == 0) {
			const auto extent = color.Extent();
			LOGF("Render target: 10-bit target draw %u%s: %s count=%u slot %u 0x%016" PRIx64
			     " %ux%u samples=%u guest format %u host %d (of %u color targets, depth "
			     "0x%016" PRIx64 " samples %u) VS 0x%016" PRIx64 " PS 0x%016" PRIx64 "\n",
			     count, skip ? " (skipped)" : "", draw.Name(), draw.index_count, color.target_slot,
			     color.desc.info.data.address, extent.width, extent.height, color.desc.info.samples,
			     static_cast<uint32_t>(format), static_cast<int>(color.desc.info.pixel_format),
			     state.color_count,
			     state.depth_info.image_id ? state.depth_info.desc.info.data.address : 0,
			     state.depth_info.image_id ? state.depth_info.desc.info.samples : 0,
			     state.vertex_info[0].stage ? state.vertex_info[0].stage.program->shader_hash : 0,
			     state.ps_active && state.ps_input_info.stage
			         ? state.ps_input_info.stage.program->shader_hash
			         : 0);
		}
		return skip;
	}
	return false;
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	// KYTY_TABLE: the time of the steps its tables replace (programs and targets), per draw.
	const uint64_t table_start = TableMode::Enabled() ? GpuPhases::Now() : 0;
	RefreshShaders(buffer, draw, color_output_mask, state);
	LogDrawCensus(buffer, draw, state);
	if (!state.programs.vertex[0] || (state.ps_active && !state.programs.pixel)) {
		OcclusionQueries::SkippedDraw(m_context, buffer.GetRegisters().GetDbCountControl());
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not drawn: no %s program (vs 0x%012" PRIx64 " ps 0x%012" PRIx64 ")",
			                !state.programs.vertex[0] ? "vertex" : "pixel",
			                buffer.GetShaders().GetVs().gs_regs.data_addr != 0
			                    ? buffer.GetShaders().GetVs().gs_regs.data_addr
			                    : buffer.GetShaders().GetVs().es_regs.data_addr,
			                buffer.GetShaders().GetPs().ps_regs.data_addr);
		}
		return false;
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.index;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawTargets);
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((mrt_mask & (1u << slot)) != 0) {
				ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
				                         render_target_slice_offset, slot);
				if (state.color_info[state.color_count].image_id) {
					state.color_count++;
				}
			}
		}
		if (draw.IsIndexed()) {
			LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
		}
		ResolveRenderDepthTarget(buffer, state.depth_info);
	}
	if (table_start != 0) {
		TableMode::AddDraw(TableMode::ProgramHitFlag(), GpuPhases::Now() - table_start);
	}
	if (state.depth_info.unsupported) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not drawn: a depth or stencil state the host cannot draw with");
		}
		return false;
	}

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not drawn: no render targets and no pixel shader");
		}
		return false;
	}
	if (TenBitTargetDraw(draw, state)) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not drawn: a 10-bit target (KYTY_SKIP_10BIT_TARGETS)");
		}
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

static void CommitVertexBuffers(const CommandRecorder&       vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(const CommandRecorder&     vk_buffer,
                              const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, const CommandRecorder& vk_buffer,
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
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
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

// Round 93, diagnostics (an F9 capture): a draw that tests depth for EQUAL (a colour pass over its
// depth prepass) is drawn again after itself, its colour, stencil and depth writes off, each time
// in an occlusion query (FrameGraphQueries::Kind): "as close or closer" with a bias towards the
// camera, and without a depth test. Only a draw whose stages write no memory (no mesh shader).
// The next draw sets its dynamic state again. Round 105: also one that writes depth too (the
// crowd's colour passes do; writing the depth it equals changes nothing). Round 107: also one
// with its counts on the GPU (the crowd's draws), drawn again from the same arguments, and a
// third probe "as close or closer" without the bias (Kind::Front). The probes replace the draw's
// own polygon offset (none in the crowd's passes).
static void DrawDepthEqualProbes(const HW::UserConfig& ucfg, const CommandRecorder& vk_buffer,
                                 const DrawCallInfo& draw, const DrawEmitInfo& emit,
                                 const DrawRenderState&                 state,
                                 std::span<const ShaderVertexInputInfo> vertex_stages,
                                 const RenderState& rendering, vk::Buffer indirect_buffer,
                                 vk::DeviceSize indirect_offset) {
	const auto& depth = state.depth_info;
	if (!depth.image_id || !depth.depth_test_enable ||
	    depth.depth_compare_op != vk::CompareOp::eEqual) {
		return;
	}
	// Stores through addresses and GDS too: a probe would repeat them (and the one without a depth
	// test, for hidden fragments as well).
	const auto writes_memory = [](const ShaderStageRuntime& runtime) {
		const auto& program = *runtime.program;
		return HasShaderBufferWrites(runtime) || program.has_address_writes ||
		       std::ranges::any_of(program.bindings.descriptors,
		                           [](const auto& binding) {
			                           return binding.kind ==
			                                  ShaderRecompiler::IR::DescriptorBindingKind::Gds;
		                           }) ||
		       std::ranges::any_of(program.info.images, [](const auto& image) {
			       return image.written &&
			              image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
		       });
	};
	if (state.ps_active && writes_memory(state.ps_input_info.stage)) {
		return;
	}
	for (const auto& stage: vertex_stages) {
		if (writes_memory(stage.stage)) {
			return;
		}
	}
#if defined(__APPLE__)
	// No color write enables to turn off (MoltenVK): the probes would draw.
	if (rendering.num_color_attachments != 0) {
		return;
	}
#else
	if (rendering.num_color_attachments != 0) {
		const std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> off {};
		vk_buffer.setColorWriteEnableEXT(rendering.num_color_attachments, off.data());
	}
#endif
	if (depth.stencil_test_enable) {
		vk_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, 0u);
	}
	vk_buffer.setDepthWriteEnable(VK_FALSE);
	const bool reversed_z = ReversedDepth(depth);
	for (const auto kind: {FrameGraphQueries::Kind::Close, FrameGraphQueries::Kind::Front,
	                       FrameGraphQueries::Kind::Anywhere}) {
		const bool tested = kind != FrameGraphQueries::Kind::Anywhere;
		const bool biased = kind == FrameGraphQueries::Kind::Close;
		vk_buffer.setDepthCompareOp(!tested      ? vk::CompareOp::eAlways
		                            : reversed_z ? vk::CompareOp::eGreaterOrEqual
		                                         : vk::CompareOp::eLessOrEqual);
		vk_buffer.setDepthBiasEnable(biased ? VK_TRUE : VK_FALSE);
		if (biased) {
			vk_buffer.setDepthBias(reversed_z ? 16384.0f : -16384.0f, 0.0f,
			                       reversed_z ? 4.0f : -4.0f);
		}
		const auto query = FrameGraphQueries::BeginExtra(vk_buffer, kind);
		if (query < 0) {
			return;
		}
		if (indirect_buffer && draw.IsIndexed()) {
			vk_buffer.drawIndexedIndirect(indirect_buffer, indirect_offset, 1, 20u);
		} else if (indirect_buffer) {
			vk_buffer.drawIndirect(indirect_buffer, indirect_offset, 1, 16u);
		} else {
			EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
		}
		FrameGraphQueries::EndDraw(vk_buffer, query);
	}
}

// Diagnostics (the frame graph): the draw's shaders, render targets and depth state, before its
// stages' bindings (PrepareBindings); the shaders' code is dumped. Dispatch-like draws (a few
// vertices, or no depth target) show their buffers' contents after the frame.
static void LogFrameGraphDraw(RenderContext& context, const CommandBuffer& buffer,
                              const DrawCallInfo& draw, const DrawRenderState& state,
                              std::span<const ShaderVertexInputInfo> vertex_stages,
                              vk::PrimitiveTopology                  topology,
                              const DrawIndexBufferSource&           index_source) {
	std::string stages;
	for (const auto& stage: vertex_stages) {
		stages += fmt::format(" {:016x}", stage.stage.program->shader_hash);
	}
	const auto ps = state.ps_active && state.ps_input_info.stage
	                    ? state.ps_input_info.stage.program->shader_hash
	                    : 0;
	FrameGraph::Log("  shaders: vertex%s ps %016" PRIx64 " topology %d", stages.c_str(), ps,
	                static_cast<int>(topology));
	// Round 158: how the guest narrows the draw's sample coverage (see NoteCoverage).
	{
		const auto& registers = buffer.GetRegisters();
		const auto& shader    = registers.GetShaderRegisters();
		const auto& config    = registers.GetAaConfig();
		const auto& color     = registers.GetBlendColor();
		FrameGraph::Log("  coverage: AA mask 0x%08x 0x%08x, samples log2 %u (exposed %u), alpha "
		                "to mask %d (0x%08x), mask export %d, blend constant %g %g %g %g",
		                registers.GetAaMask(0), registers.GetAaMask(1),
		                static_cast<uint32_t>(config.msaa_num_samples),
		                static_cast<uint32_t>(config.msaa_exposed_samples),
		                state.ps_active && shader.AlphaToMask() ? 1 : 0,
		                registers.GetAlphaToMaskValue(),
		                shader.db_shader_control.shader_mask_export_enable ? 1 : 0, color.red,
		                color.green, color.blue, color.alpha);
	}
	auto& textures = context.GetTextureCache();
	for (uint32_t i = 0; i < state.color_count; i++) {
		const auto& color   = state.color_info[i];
		const auto& info    = color.desc.info;
		const auto  extent  = color.Extent();
		const auto  cached  = color.image_id ? textures.DiagnosticImageState(color.image_id)
		                                     : TextureCache::DiagnosticState {};
		const char  flags[] = {cached.gpu_written ? 'G' : '-', cached.buffer_write ? 'B' : '-',
                              cached.cpu_dirty ? 'C' : '-', '\0'};
		FrameGraph::Log(
		    "  rt slot %u: 0x%012" PRIx64 "+0x%" PRIx64
		    " %ux%u mip %u layer %u fmt %d guest %u samples %u tile %u [%s] overlaps %u (%u "
		    "GPU-written)",
		    color.target_slot, info.data.address, info.data.size, extent.width, extent.height,
		    color.guest_mip_level, color.guest_array_layer, static_cast<int>(info.pixel_format),
		    static_cast<uint32_t>(info.guest_format), info.samples,
		    static_cast<uint32_t>(info.tile_mode), flags, cached.overlaps, cached.gpu_overlaps);
		// How the draw writes it (round 65): the guest's blend control and the slot's write mask.
		const auto& registers = buffer.GetRegisters();
		const auto& blend     = registers.GetBlendControl(color.target_slot);
		FrameGraph::Log(
		    "  rt slot %u blend: enable %d color %u %u %u alpha %u %u %u separate %d, "
		    "write mask 0x%x, bypass %d, clear word 0x%08x",
		    color.target_slot, blend.enable ? 1 : 0, static_cast<uint32_t>(blend.color_srcblend),
		    static_cast<uint32_t>(blend.color_comb_fcn),
		    static_cast<uint32_t>(blend.color_destblend),
		    static_cast<uint32_t>(blend.alpha_srcblend),
		    static_cast<uint32_t>(blend.alpha_comb_fcn),
		    static_cast<uint32_t>(blend.alpha_destblend), blend.separate_alpha_blend ? 1 : 0,
		    (registers.GetRenderTargetMask() >> (color.target_slot * 4u)) & 0xfu,
		    registers.GetRenderTarget(color.target_slot).info.blend_bypass ? 1 : 0,
		    info.metadata.clear_word);
		// A small linear target, the kind the CPU reads back: shown after the frame as guest
		// memory holds it.
		if (info.tile_mode == Prospero::TileMode::kLinear && info.data.size <= 64u * 1024u) {
			FrameGraph::NoteReadback(info.data.address, info.data.size,
			                         textures.ReadsBackForCpu(info));
		}
	}
	if (state.depth_info.image_id) {
		const auto& depth = state.depth_info;
		const auto& info  = depth.desc.info;
		// Its HTile, and whether a fill there left a clear for this draw to apply (round 64).
		const bool htile = info.metadata.kind == ImageMetadataKind::Htile;
		FrameGraph::Log(
		    "  depth: 0x%012" PRIx64 "+0x%" PRIx64
		    " %ux%u fmt %d samples %u test %d write %d op %d bounds %d stencil %d clear %d"
		    " htile 0x%012" PRIx64 " htile-clear %d",
		    info.data.address, info.data.size, info.extent.width, info.extent.height,
		    static_cast<int>(info.pixel_format), info.samples, depth.depth_test_enable ? 1 : 0,
		    depth.depth_write_enable ? 1 : 0, static_cast<int>(depth.depth_compare_op),
		    depth.depth_bounds_test_enable ? 1 : 0, depth.stencil_test_enable ? 1 : 0,
		    depth.depth_clear_enable ? 1 : 0, htile ? info.metadata.range.address : 0,
		    htile && textures.IsMetaCleared(info.metadata.range.address, depth.guest_array_layer)
		        ? 1
		        : 0);
		// The values a clear of it takes, and the stencil test (round 65: the course maps' track,
		// drawn with depth and stencil tests, went missing while the cursors drew).
		FrameGraph::Log("  depth clear value %g, stencil clear %d value %u",
		                depth.depth_clear_value, depth.stencil_clear_enable ? 1 : 0,
		                static_cast<uint32_t>(depth.stencil_clear_value));
		if (depth.stencil_test_enable) {
			const auto face = [](const vk::StencilOpState& op) {
				return fmt::format(
				    "compare {} ref 0x{:02x} read 0x{:02x} write 0x{:02x} fail {} pass "
				    "{} depth-fail {}",
				    vk::to_string(op.compareOp), op.reference, op.compareMask, op.writeMask,
				    vk::to_string(op.failOp), vk::to_string(op.passOp),
				    vk::to_string(op.depthFailOp));
			};
			FrameGraph::Log("  stencil front: %s", face(depth.stencil_front).c_str());
			FrameGraph::Log("  stencil back: %s", face(depth.stencil_back).c_str());
		}
	}
	// Round 90: what turns its positions into depth (clipping, the viewport, depth bias) and where
	// its indices and vertices come from, so that two passes over the same geometry (a depth
	// prepass and a colour pass testing for EQUAL) can be compared.
	{
		const auto& registers = buffer.GetRegisters();
		const auto& clip      = registers.GetClipControl();
		const auto& screen    = registers.GetScreenViewport();
		const auto& viewport  = screen.viewports[0];
		const auto& mode      = registers.GetModeControl();
		const auto& poly      = registers.GetPolyOffset();
		FrameGraph::Log(
		    "  raster: clip disable %d dx %d z-clip off %d %d, vte 0x%x, viewport x %g "
		    "%+g y %g %+g z %g %+g (%g to %g), cull %d%d front %d, poly offset %d%d "
		    "(front %g %+g back %g %+g clamp %g, %d bits%s)",
		    clip.clip_disable ? 1 : 0, clip.dx_clip_space ? 1 : 0, clip.min_z_clip_disable ? 1 : 0,
		    clip.max_z_clip_disable ? 1 : 0, screen.transform_control, viewport.xscale,
		    viewport.xoffset, viewport.yscale, viewport.yoffset, viewport.zscale, viewport.zoffset,
		    viewport.zmin, viewport.zmax, mode.cull_front ? 1 : 0, mode.cull_back ? 1 : 0,
		    mode.face ? 1 : 0, mode.poly_offset_front_enable ? 1 : 0,
		    mode.poly_offset_back_enable ? 1 : 0, poly.front_scale, poly.front_offset,
		    poly.back_scale, poly.back_offset, poly.clamp, -static_cast<int>(poly.neg_num_db_bits),
		    poly.db_is_float_fmt ? " float" : "");
		auto& cache = context.GetBufferCache();
		if (draw.IsIndexed()) {
			FrameGraph::Log(
			    "  index: 0x%012" PRIx64 "+0x%" PRIx64 " %u bytes each, hash %s",
			    index_source.address, index_source.size, index_source.guest_element_size,
			    index_source.host_data != nullptr
			        ? "(rewritten on the host)"
			        : FrameGraph::BytesHash(cache, index_source.address, index_source.size)
			              .c_str());
		}
		for (const auto& stage: vertex_stages) {
			FrameGraph::Log("  %016" PRIx64 " inputs: %s, %d buffers, %d attributes, clip-space "
			                "transform %d",
			                stage.stage.program->shader_hash,
			                stage.fetch_embedded   ? "fetched by the vertex input"
			                : stage.fetch_external ? "fetched by an external fetch shader"
			                                       : "read by the shader",
			                stage.buffers_num, stage.resources_num,
			                stage.clip_space.enabled ? 1 : 0);
			for (int i = 0; i < stage.buffers_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
				const auto&    vb    = stage.buffers[i];
				const uint64_t bytes = static_cast<uint64_t>(vb.stride) * vb.num_records;
				FrameGraph::Log("    vb[%d] 0x%012" PRIx64 " stride %u records %u hash %s", i,
				                vb.addr, vb.stride, vb.num_records,
				                FrameGraph::BytesHash(cache, vb.addr, bytes).c_str());
				if (FrameGraph::DumpsShaderMemory(stage.stage.program->shader_hash)) {
					FrameGraph::LogBytes(fmt::format("vb memory 0x{:012x}", vb.addr).c_str(),
					                     vb.addr, bytes);
				}
			}
			if (draw.IsIndexed() && index_source.host_data == nullptr &&
			    FrameGraph::DumpsShaderMemory(stage.stage.program->shader_hash)) {
				FrameGraph::LogBytes(
				    fmt::format("index memory 0x{:012x}", index_source.address).c_str(),
				    index_source.address, index_source.size);
			}
			for (int i = 0; i < stage.resources_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
				const auto& resource = stage.resources[i];
				const auto& dst      = stage.resources_dst[i];
				FrameGraph::Log("    attr[%d] v%d+%d from vb[%d] base 0x%012" PRIx64
				                " stride %u format %u dst %03x%s",
				                i, dst.register_start, dst.registers_num, dst.buffer_index,
				                resource.Base48(), resource.Stride(),
				                static_cast<uint32_t>(resource.RawFormat()), resource.DstSelXYZW(),
				                resource.AddTid() ? " add-tid" : "");
			}
		}
	}
	const auto& shaders = buffer.GetShaders();
	const auto  note    = [](const char* stage, uint64_t address) {
        if (address != 0) {
            FrameGraph::NoteShader(stage, ShaderDeclaredHash(address), address, nullptr, 0);
        }
	};
	note("ES", shaders.GetVs().es_regs.data_addr);
	note("LS", shaders.GetVs().ls_regs.data_addr);
	note("HS", shaders.GetVs().hs_regs.data_addr);
	note("GS", shaders.GetVs().gs_regs.data_addr);
	if (state.ps_active) {
		note("PS", shaders.GetPs().ps_regs.data_addr);
	}
	FrameGraph::SetShowBuffers(draw.index_count <= 6 || !state.depth_info.image_id);
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	auto& ucfg = buffer.GetUserConfig();
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	NoteDrawRepeat(draw, state, index_source, vertex_stages.size(),
	               m_context.GetTextureCache().ImageSetGeneration());
	NoteDrawTarget(state);
	NoteCoverage(buffer, state);
	LearnCubeTarget(buffer, state);
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
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
		// Counts read on the GPU (DRAW_INDIRECT): one group per vertex, as fast launch runs.
		EXIT_NOT_IMPLEMENTED(emit.gpu_args_address != 0 && !mesh.fast_launch);
	}
	if (mesh_active && emit.gpu_args_address == 0) {
		const auto& mesh       = state.vertex_info[0].mesh;
		const auto  primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  not drawn: no primitives");
			}
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (draw.instance_count > limits.maxMeshWorkGroupCount[1]) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	// DrawIndirectArgs, or DrawIndexedIndirectArgs (VkDrawIndexedIndirectCommand).
	const uint32_t gpu_args_size = draw.IsIndexed() ? 20u : 16u;
	if (emit.gpu_args_address != 0) {
		// The mesh shader reads the start vertex there, and the counts are copied from there:
		// synchronize and register the arguments like the guest indices below.
		(void)m_context.GetBufferCache().ObtainBuffer(emit.gpu_args_address, gpu_args_size, false,
		                                              false, {}, true);
	}
	if (emit.gpu_instance_count_address != 0) {
		(void)m_context.GetBufferCache().ObtainBuffer(emit.gpu_instance_count_address, 4u, false,
		                                              false, {}, true);
	}
	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	if (FrameGraph::Logging()) {
		LogFrameGraphDraw(m_context, buffer, draw, state, vertex_stages, topology, index_source);
	}
	// KYTY_BISECT's tess:bit:<k> (round 82): the tessellated draws are numbered in their frame
	// (the GPU thread's: between the suspend points it processes), and a step leaves out those
	// whose number has bit k set. The first whole frame of each step lists them (once a step),
	// so that the pictures' answer (a draw's number) names its shaders.
	if (ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch && Bisect::TessBitsConfigured()) {
		// GPU thread only.
		static uint64_t tess_frame       = UINT64_MAX;
		static uint32_t tess_number      = 0;
		static uint64_t listed_steps     = 0;
		static uint64_t listing_frame    = UINT64_MAX;
		static uint32_t step_first_frame = UINT32_MAX;
		const auto      frame            = FrameGraph::GpuThreadFrames();
		const auto      step             = Bisect::StepNumber();
		const auto      step_bit         = step < 64u ? uint64_t {1} << step : 0u;
		if (frame != tess_frame) {
			tess_frame  = frame;
			tess_number = 0;
			// A frame that starts after the step turned on is a whole frame of it.
			if ((listed_steps & step_bit) == 0 && step_bit != 0) {
				if (step_first_frame != step) {
					step_first_frame = step;
				} else {
					listed_steps |= step_bit;
					listing_frame = frame;
				}
			}
		}
		const auto number = tess_number++;
		if (frame == listing_frame) {
			std::string stages;
			for (const auto& stage: vertex_stages) {
				stages += fmt::format(" {}:{:016x}", static_cast<int>(stage.stage.program->stage),
				                      stage.stage.program->shader_hash);
			}
			LOGF("Bisect: tess draw %u (frame %" PRIu64 ", step %u):%s ps %016" PRIx64
			     ", %u indices x %u, %u control points, indices at 0x%" PRIx64 ", color 0x%" PRIx64
			     ", depth 0x%" PRIx64 "\n",
			     number, frame, step, stages.c_str(),
			     state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull,
			     draw.index_count, draw.instance_count,
			     (buffer.GetRegisters().GetShaderRegisters().m_vgtLsHsConfig >> 8u) & 0x3fu,
			     index_source.address,
			     state.color_count != 0 ? state.color_info[0].desc.info.data.address : 0ull,
			     state.depth_info.image_id ? state.depth_info.desc.info.data.address : 0ull);
		}
		if (const auto bit = Bisect::TessSkipBit(); bit >= 0 && ((number >> bit) & 1u) != 0) {
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  not drawn: KYTY_BISECT's tess:bit:%d (tessellated draw %u)", bit,
				                number);
			}
			return;
		}
	}
	// KYTY_BISECT (bisect.h): a step that leaves out the draws of one of these shaders.
	if (Bisect::Active()) {
		uint64_t skipped = 0;
		for (const auto& stage: vertex_stages) {
			if (skipped == 0 && Bisect::SkipsShader(stage.stage.program->shader_hash)) {
				skipped = stage.stage.program->shader_hash;
			}
		}
		if (skipped == 0 && state.ps_active &&
		    Bisect::SkipsShader(state.ps_input_info.stage.program->shader_hash)) {
			skipped = state.ps_input_info.stage.program->shader_hash;
		}
		if (skipped != 0) {
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  not drawn: KYTY_BISECT leaves out %016" PRIx64, skipped);
			}
			return;
		}
	}
	// EQAA (fewer color than depth samples). KYTY_FRAME_GRAPH_EQAA=1: the first such draw (GT7's
	// 1920x2160 rendering after a race in Frame Rate mode) asks for a frame capture (F9's).
	// Through NV mixed samples it may not shade per sample: a pixel shader reading its sample
	// index runs its per-pixel copy (shaders.cpp), and one without that copy (the rewrite gave
	// up) is skipped instead of losing the device, before anything is bound for it.
	if (state.color_count != 0 && state.depth_info.image_id &&
	    state.depth_info.desc.view_info.format != vk::Format::eUndefined &&
	    state.color_info[0].desc.info.samples < state.depth_info.desc.info.samples) {
		static const bool capture = [] {
			const char* value = std::getenv("KYTY_FRAME_GRAPH_EQAA");
			return value != nullptr && value[0] == '1';
		}();
		static std::atomic_bool requested {false};
		if (capture && !requested.exchange(true, std::memory_order_relaxed)) {
			LOGF("FG 0 first EQAA draw (KYTY_FRAME_GRAPH_EQAA): capturing the next frame\n");
			FrameGraph::RequestCapture();
		}
		if (state.ps_active &&
		    buffer.GetGraphics().mixed_attachment_samples ==
		        GraphicContext::MixedAttachmentSamples::Nv &&
		    state.programs.pixel.pixel_rate_module == nullptr && !EqaaPerSampleShading() &&
		    std::ranges::any_of(
		        state.ps_input_info.stage.program->info.inputs, [](const auto& input) {
			        return input.kind == ShaderRecompiler::IR::StageInputKind::SampleId;
		        })) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("Pipeline: EQAA draw skipped: pixel shader 0x%016" PRIx64
				     " reads its sample index and has no per-pixel copy\n",
				     state.ps_input_info.stage.program->shader_hash);
			}
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  skipped: an EQAA draw whose pixel shader shades per sample");
			}
			return;
		}
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawTextures);
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			bindings.vertex[i].normalize_unnormalized =
			    state.programs.vertex[i].normalized_samplers;
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
			descriptor_stages[stage_count++] = &bindings.vertex[i];
		}
		if (state.ps_active) {
			if (!bindings.pixel) bindings.pixel.emplace();
			PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
			descriptor_stages[stage_count++] = &*bindings.pixel;
		}
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	if (ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch) {
		DumpTessellatedDraw(buffer, draw, state, index_source, bindings.vertex);
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawVertex);
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	// A draw that writes no memory may be skipped while its new pipeline compiles in the
	// background; its render targets are not touched. One that writes buffers or storage images
	// waits for the pipeline, as others may read what it writes.
	const auto writes_memory = [](const ShaderStageRuntime& runtime) {
		return HasShaderBufferWrites(runtime) ||
		       std::ranges::any_of(runtime.program->info.images, [](const auto& image) {
			       return image.written &&
			              image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
		       });
	};
	bool writes = state.ps_active && writes_memory(state.ps_input_info.stage);
	for (const auto& stage: vertex_stages) {
		writes = writes || writes_memory(stage.stage);
	}
	// Glyphs and cached layers: drawn once into small targets (SmallColorTargetDraw), unless the
	// game draws the target every frame (round 87: its reflection cubes, RedrawnTargetDraw).
	auto defer = PipelineDefer::Allowed;
	if (writes) {
		defer = PipelineDefer::WritesMemory;
	} else if (SmallColorTargetDraw(buffer.GetRegisters()) && !state.programs.redrawn_target) {
		defer = PipelineDefer::SmallTarget;
	}

	PipelineCache::Pipeline* deferred_pipeline = nullptr;
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawPipeline);
		deferred_pipeline = m_context.GetPipelineCache().TryGetGraphicsPipeline(
		    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages,
		    buffer, state.ps_active ? &state.ps_input_info : nullptr, topology,
		    primitive_restart_enable, state.programs, defer);
	}
	if (deferred_pipeline == nullptr) {
		OcclusionQueries::SkippedDraw(m_context, buffer.GetRegisters().GetDbCountControl());
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  skipped: its pipeline is still compiling");
		}
		return;
	}
	auto& pipeline = *deferred_pipeline;
	vk::ImageAspectFlags feedback_aspects;
	GpuPhases::Scope     rendering_phase(GpuPhases::Phase::DrawRendering);
	const auto rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
	                         feedback_aspects, stages);
	rendering_phase.End();
	if (FrameGraph::Logging() && state.depth_info.image_id) {
		// What the depth attachment's rendering starts with (an HTile clear among them).
		const auto& ds = rendering.depth_stencil_attachment;
		FrameGraph::Log("  depth attachment: depth %s, stencil %s",
		                !ds.has_depth    ? "none"
		                : ds.depth_clear ? "cleared"
		                                 : "loaded",
		                !ds.has_stencil    ? "none"
		                : ds.stencil_clear ? "cleared"
		                                   : "loaded");
	}

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory. Recorded through the recording thread when it runs.
	GpuPhases::Scope emit_phase(GpuPhases::Phase::DrawEmit);
	const auto       vk_buffer = buffer.Recorder();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	buffer.AnnotateDiagnosticCheckpoint(
	    vertex_stages.front().stage.program->shader_hash,
	    state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0,
	    state.color_count != 0 ? state.color_info[0].desc.info.samples : 0,
	    state.depth_info.image_id ? state.depth_info.desc.info.samples : 0,
	    (state.ps_active && writes_memory(state.ps_input_info.stage) ? 1u : 0u) |
	        (std::ranges::any_of(vertex_stages,
	                             [&](const auto& stage) { return writes_memory(stage.stage); })
	             ? 2u
	             : 0u) |
	        (std::min(rendering.num_layers, 255u) << 8u) |
	        (std::min(rendering.num_color_attachments, 255u) << 16u) |
	        (rendering.depth_stencil_attachment.image_view ? 1u << 24u : 0u));
	if (buffer.RecordsCheckpoints()) {
		DiagnosticTargets targets;
		const auto        pack = [](uint32_t width, uint32_t height) {
			return std::min(width, 0xffffu) | (std::min(height, 0xffffu) << 16u);
		};
		if (state.color_count != 0) {
			const auto& color     = state.color_info[0];
			const auto  extent    = color.Extent();
			targets.color_address = color.desc.info.data.address;
			targets.color_extent  = pack(extent.width, extent.height);
			targets.color_format  = static_cast<uint32_t>(color.desc.info.pixel_format);
		}
		for (uint32_t i = 1; i < state.color_count && i <= DiagnosticTargets::MoreColors; i++) {
			const auto& color              = state.color_info[i];
			const auto  extent             = color.Extent();
			targets.more_addresses[i - 1u] = color.desc.info.data.address;
			targets.more_extents[i - 1u]   = pack(extent.width, extent.height);
			targets.more_formats[i - 1u]   = static_cast<uint32_t>(color.desc.info.pixel_format);
			targets.more_slots[i - 1u]     = color.target_slot | (color.desc.info.samples << 8u);
		}
		if (state.depth_info.image_id) {
			const auto& depth     = state.depth_info.desc.info;
			targets.depth_address = depth.data.address;
			targets.depth_extent  = pack(depth.extent.width, depth.extent.height);
			targets.depth_format  = static_cast<uint32_t>(depth.pixel_format);
		}
		const auto& ds = rendering.depth_stencil_attachment;
		targets.area   = pack(rendering.width, rendering.height);
		targets.flags  = std::min(rendering.num_color_attachments, 255u) |
		                (rendering.num_color_attachments != 0 &&
		                         rendering.color_attachments[0].is_clear
		                     ? 1u << 8u
		                     : 0u) |
		                (ds.depth_clear ? 1u << 9u : 0u) | (ds.stencil_clear ? 1u << 10u : 0u) |
		                (ds.has_stencil ? 1u << 11u : 0u);
		buffer.AnnotateDiagnosticTargets(targets);
	}
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	// Every phase marker, for indexed draws too: a device loss then names the step it was in.
	if (state.ps_active) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	emit_phase.End();
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawCommit);
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	}
	emit_phase.Restart();
	if (mesh_active) {
		// GPU arguments: dword 3 = 0xFFFFFFFF tells the fast-launch entry to read the start vertex
		// at dwords 4-5's address + 8 (the shader's other draw dwords are unused then).
		const bool     gpu_args = emit.gpu_args_address != 0;
		const uint32_t draw_data[] {
		    gpu_args ? 0u : draw.index_count,
		    gpu_args ? 0u
		             : draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset)
		                                : emit.first_vertex,
		    gpu_args ? 0u : emit.first_instance,
		    gpu_args ? 0xFFFFFFFFu : index_source.guest_element_size,
		    static_cast<uint32_t>(gpu_args ? emit.gpu_args_address : index_source.address),
		    static_cast<uint32_t>((gpu_args ? emit.gpu_args_address : index_source.address) >> 32u),
		    0u};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(draw_data), draw_data);
	} else {
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	SetGraphicsDynamicParams(buffer, vk_buffer, vertex_stages.back(), state.depth_info, rendering);
	if (m_context.GetTextureCache().TraceEnabled()) {
		TraceTargetDraws(m_context.GetTextureCache(), buffer, state.depth_info, state.color_info,
		                 state.color_count, rendering);
	}
	if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
		vk_buffer.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
	}

	// GPU arguments: the group counts (vertex count, instance count, 1) go into a ring entry for
	// drawMeshTasksIndirectEXT, outside the render pass; the arguments are never read on the CPU.
	vk::Buffer     mesh_indirect_buffer = nullptr;
	vk::DeviceSize mesh_indirect_offset = 0;
	constexpr uint32_t MeshIndirectEntries = 4096;
	if (mesh_active && emit.gpu_args_address != 0) {
		auto&              cache = m_context.GetBufferCache();
		auto&              args  = cache.GetBuffer(cache.FindBuffer(emit.gpu_args_address, 16u));
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<Buffer>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal,
			    0, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
			    MeshIndirectEntries * 16u);
		}
		mesh_indirect_buffer = m_mesh_indirect->Handle();
		mesh_indirect_offset = (m_mesh_indirect_next++ % MeshIndirectEntries) * 16u;
		m_context.GetCommandScheduler().EndRendering();
		// The arguments' writer (a dispatch or a copy) and the entry's last indirect read come
		// before the copy.
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                       vk::AccessFlagBits::eTransferWrite |
		                       vk::AccessFlagBits::eIndirectCommandRead;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
		                          0, nullptr);
		const vk::BufferCopy copy {args.Offset(emit.gpu_args_address), mesh_indirect_offset, 8u};
		vk_buffer.copyBuffer(args.Handle(), mesh_indirect_buffer, 1, &copy);
		vk_buffer.fillBuffer(mesh_indirect_buffer, mesh_indirect_offset + 8u, 4u, 1u);
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &after, 0,
		                          nullptr, 0, nullptr);
	}

	// Vertex pipeline, counts on the GPU: the arguments' writer (a dispatch or a copy) comes
	// before the indirect read; the arguments are read where the cache holds them.
	vk::Buffer     vertex_indirect_buffer = nullptr;
	vk::DeviceSize vertex_indirect_offset = 0;
	if (!mesh_active && emit.gpu_instance_count_address != 0) {
		// Only the instance count is on the GPU: a ring entry gets the known counts and offsets,
		// and the GPU copies the instance count into it.
		auto& cache  = m_context.GetBufferCache();
		auto& source = cache.GetBuffer(cache.FindBuffer(emit.gpu_instance_count_address, 4u));
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<Buffer>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal,
			    0, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
			    MeshIndirectEntries * 16u);
		}
		vertex_indirect_buffer = m_mesh_indirect->Handle();
		vertex_indirect_offset = (m_mesh_indirect_next++ % MeshIndirectEntries) * 16u;
		m_context.GetCommandScheduler().EndRendering();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                       vk::AccessFlagBits::eTransferWrite |
		                       vk::AccessFlagBits::eIndirectCommandRead;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
		                          0, nullptr);
		const uint32_t command[4] {draw.index_count, 0u, emit.first_vertex, emit.first_instance};
		vk_buffer.updateBuffer(vertex_indirect_buffer, vertex_indirect_offset, sizeof(command),
		                       command);
		const vk::BufferCopy copy {source.Offset(emit.gpu_instance_count_address),
		                           vertex_indirect_offset + sizeof(uint32_t), sizeof(uint32_t)};
		vk_buffer.copyBuffer(source.Handle(), vertex_indirect_buffer, 1, &copy);
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &after, 0,
		                          nullptr, 0, nullptr);
	} else if (!mesh_active && emit.gpu_args_address != 0) {
		auto& cache = m_context.GetBufferCache();
		auto& args  = cache.GetBuffer(cache.FindBuffer(emit.gpu_args_address, gpu_args_size));
		vertex_indirect_buffer = args.Handle();
		vertex_indirect_offset = args.Offset(emit.gpu_args_address);
		m_context.GetCommandScheduler().EndRendering();
		vk::MemoryBarrier ready {};
		ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		ready.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &ready, 0,
		                          nullptr, 0, nullptr);
	}

	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages && !GuestGpu::Pm4SyncBarriers()) {
		// As for dispatches, buffer writes wait for earlier accesses to the memory (mainline
		// 70668fbd: this replaces the global barriers of PM4 flush events, which
		// KYTY_PM4_BARRIERS=global records again, without this).
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteHazardBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "BeginRendering");
	SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	FrameGraphQueries::PrepareDraw(m_context.GetGraphics().device, m_context.GetCommandScheduler(),
	                               vk_buffer);
	// Round 119: the guest's occlusion counters (OcclusionQueries); one occlusion query at a time,
	// so a capture's own query leaves the draw to it.
	const int32_t occlusion_query = OcclusionQueries::PrepareDraw(
	    m_context, vk_buffer, buffer.GetRegisters().GetDbCountControl());
	m_context.GetCommandScheduler().BeginRendering(rendering);
	// Round 172: a guest pipeline keeps the dynamic state the draws set (the color write enables
	// only with color targets), which the draw set before it.
	vk_buffer.bindGuestPipeline(pipeline.pipeline, rendering.num_color_attachments != 0);
	SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	GpuTiming::Before(m_context, buffer);
	OcclusionQueries::BeginDraw(vk_buffer, occlusion_query);
	const int32_t samples_query =
	    occlusion_query < 0 ? FrameGraphQueries::BeginDraw(vk_buffer) : -1;
	int64_t gpu_zone = -1;
	if (Profiler::g_counters) {
		// The draw's program, as its GPU zone's name and its command zone's text.
		char      name[48];
		const int size = std::snprintf(
		    name, sizeof(name), "VS %016" PRIx64 " PS %016" PRIx64,
		    vertex_stages.back().stage.program->shader_hash,
		    state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0);
		gpu_zone = GpuProfiler::Begin(m_context, buffer, name, static_cast<size_t>(size), 0x4e79a7);
		Profiler::CommandZone::Annotate("%s", name);
	}
	if (mesh_active) {
		if (mesh_indirect_buffer) {
			vk_buffer.drawMeshTasksIndirectEXT(mesh_indirect_buffer, mesh_indirect_offset, 1, 16u);
		} else {
			// A draw with more groups than the host allows in one dispatch (RADV: 65,535 in x) is
			// split; draw dword 6 gives each part its first group.
			constexpr auto base_offset = (ShaderRecompiler::IR::PushData::MeshDrawDwordCount - 1u) *
			                             sizeof(uint32_t);
			const auto& limits   = m_context.GetGraphics().mesh_shader_properties;
			const auto  per_draw = std::min<uint64_t>(
			    limits.maxMeshWorkGroupCount[0],
			    std::max<uint64_t>(1u, limits.maxMeshWorkGroupTotalCount /
			                               std::max(draw.instance_count, 1u)));
			uint32_t groups_done = 0;
			do {
				const auto groups =
				    static_cast<uint32_t>(std::min<uint64_t>(mesh_groups - groups_done, per_draw));
				if (groups_done != 0) {
					// The same stages as the draw dwords' push above (the layout's range).
					vk_buffer.pushConstants(pipeline.pipeline_layout,
					                        vk::ShaderStageFlagBits::eMeshEXT |
					                            vk::ShaderStageFlagBits::eFragment,
					                        base_offset, sizeof(groups_done), &groups_done);
				}
				vk_buffer.drawMeshTasksEXT(groups, draw.instance_count, 1);
				groups_done += groups;
			} while (groups_done < mesh_groups);
		}
	} else if (vertex_indirect_buffer && draw.IsIndexed()) {
		vk_buffer.drawIndexedIndirect(vertex_indirect_buffer, vertex_indirect_offset, 1, 20u);
	} else if (vertex_indirect_buffer) {
		vk_buffer.drawIndirect(vertex_indirect_buffer, vertex_indirect_offset, 1, 16u);
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	Profiler::Add(Profiler::Counter::Draws);
	Common::PerfStats::AddDraw();
	m_context.GetCommandScheduler().NoteDraw(
	    state.ps_active ? state.ps_input_info.stage.program->shader_hash
	                    : vertex_stages.back().stage.program->shader_hash,
	    static_cast<bool>(vertex_indirect_buffer) || static_cast<bool>(mesh_indirect_buffer));
	FrameGraphQueries::EndDraw(vk_buffer, samples_query);
	OcclusionQueries::EndDraw(vk_buffer, occlusion_query);
	GpuProfiler::End(buffer, gpu_zone);
	GpuTiming::AfterDraw(
	    m_context, buffer,
	    state.ps_active ? state.ps_input_info.stage.program->shader_hash
	                    : vertex_stages.back().stage.program->shader_hash,
	    [&] {
		    GpuTiming::DrawDetail detail;
		    detail.vs        = vertex_stages.front().stage.program->shader_hash;
		    detail.count     = draw.index_count;
		    detail.instances = draw.instance_count;
		    detail.width     = rendering.width;
		    detail.height    = rendering.height;
		    detail.layers    = rendering.num_layers;
		    detail.color_samples =
		        state.color_count != 0 ? state.color_info[0].desc.info.samples : 0u;
		    detail.depth_samples =
		        state.depth_info.image_id ? state.depth_info.desc.info.samples : 0u;
		    detail.flags = (draw.IsIndexed() ? GpuTiming::DrawDetail::Indexed : 0u) |
		                   (static_cast<bool>(vertex_indirect_buffer) ||
		                            static_cast<bool>(mesh_indirect_buffer)
		                        ? GpuTiming::DrawDetail::GpuArguments
		                        : 0u) |
		                   (mesh_active ? GpuTiming::DrawDetail::Mesh : 0u);
		    if (state.ps_active && bindings.pixel) {
			    DescribeLightGrid(*bindings.pixel, detail);
		    }
		    if (vertex_stages.size() == 3) {
			    const auto& tess    = state.vertex_info[0].tess;
			    detail.flags       |= GpuTiming::DrawDetail::Tessellated;
			    detail.hs           = vertex_stages[1].stage.program->shader_hash;
			    detail.patch_points = tess.input_control_points;
			    detail.tess_min     = std::bit_cast<float>(tess.min_level_bits);
			    detail.tess_max     = std::bit_cast<float>(tess.max_level_bits);
		    }
		    return detail;
	    });
	if (FrameGraph::Logging() && state.depth_info.image_id && state.depth_info.depth_test_enable) {
		// Which way is closer on this depth target, from the draws that test GREATER or LESS.
		(void)ReversedDepth(state.depth_info);
	}
	if (samples_query >= 0 && !mesh_active) {
		DrawDepthEqualProbes(ucfg, vk_buffer, draw, emit, state, vertex_stages, rendering,
		                     vertex_indirect_buffer, vertex_indirect_offset);
	}

	SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	// Round 166: with barriers:guest the guest's own sync points order the draw's writes.
	if (shader_write_stages && !GuestGpu::GuestPm4Barriers()) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	// The draw's writes are in this command buffer now, before a periodic flush submits it.
	bool     address_writes = false;
	uint64_t address_writer = 0;
	for (const auto& stage: vertex_stages) {
		if (!address_writes && stage.stage.program->has_address_writes) {
			address_writes = true;
			address_writer = stage.stage.program->shader_hash;
		}
	}
	if (!address_writes && state.ps_active &&
	    state.ps_input_info.stage.program->has_address_writes) {
		address_writes = true;
		address_writer = state.ps_input_info.stage.program->shader_hash;
	}
	m_context.GetBufferCache().WritesRecorded(address_writes, address_writer);
	LogDrawPhase(draw.Name(), "DrawComplete");
	SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
}

// A DRAW_INDEX_INDIRECT whose arguments the GPU reads, drawn with the arguments read here instead
// (waiting for the GPU when it wrote them): the index buffer's start and the counts.
static DrawIndexArgs ReadIndexedIndirectArgs(const DrawIndexArgs& args) {
	uint32_t words[5] {};
	std::memcpy(words, reinterpret_cast<const void*>(args.gpu_args_address), sizeof(words));
	const uint32_t element = args.index_type_and_size == 1 ? 4u : 2u;
	DrawIndexArgs  drawn   = args;
	drawn.index_count      = std::min(words[0], args.index_count);
	drawn.index_addr       = static_cast<const uint8_t*>(args.index_addr) +
	                   static_cast<uint64_t>(words[2]) * element;
	drawn.instance_count   = words[1];
	drawn.base_vertex      = static_cast<int32_t>(words[3]);
	drawn.first_instance   = words[4];
	drawn.gpu_args_address = 0;
	return drawn;
}

// Whether the GPU can read a DRAW_INDEX_INDIRECT's arguments, known before the draw's state:
// not a legacy quad list (drawn as a call per quad) nor a custom primitive-restart index (found by
// scanning the indices).
static bool GpuIndexedArgsSupported(const CommandBuffer& buffer, uint32_t index_type_and_size) {
	const auto& ucfg = buffer.GetUserConfig();
	if (ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy) {
		return false;
	}
	const auto control = ucfg.GetPrimitiveResetControl();
	if ((control & 0x1u) == 0) {
		return true;
	}
	// As ResolvePrimitiveRestart: only the native restart index needs no scan.
	const uint32_t index_mask = index_type_and_size == 1 ? 0xffffffffu : 0xffffu;
	const auto     reset      = buffer.GetRegisters().GetPrimitiveResetIndex();
	return (reset & index_mask) == index_mask;
}

const char* RenderExecutor::ConsumeDrawAsOperation(CommandBuffer& buffer,
                                                   uint32_t       render_target_slice_offset) {
	if (ConsumeMetadataColorOperation(buffer)) {
		return "a metadata color operation";
	}
	if (ConsumeDepthColorCopy(buffer)) {
		return "a depth-to-color copy";
	}
	if (DepthStencilCopy(buffer)) {
		return "a depth/stencil copy";
	}
	if (ResolveColorTargets(buffer, render_target_slice_offset)) {
		return "a resolve";
	}
	return nullptr;
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& guest_args) {
	KYTY_PROFILER_FUNCTION();
	const GpuPhases::Scope draw_phase(GpuPhases::Phase::Draw);
	const GpuPhases::OpStats op_stats(false);
	// KYTY_BISECT's slow:gpu (round 114).
	Bisect::SpinMicroseconds(Bisect::SlowGpuMicroseconds());
	// A tessellated draw's time, whole (round 78).
	std::optional<GpuPhases::Scope> tess_phase;
	if (buffer.GetUserConfig().GetPrimType() == Prospero::PrimitiveType::kPatch) {
		tess_phase.emplace(GpuPhases::Phase::TessDraw);
	}
	DrawIndexArgs args = guest_args;
	if (args.gpu_args_address != 0 &&
	    !GpuIndexedArgsSupported(buffer, args.index_type_and_size)) {
		args = ReadIndexedIndirectArgs(args);
	}

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawPending);
		m_context.GetCommandScheduler().PopPendingOperations(false);
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	const RendererLock lock(m_context.GetMutex());
	if (args.index_count == 0 || args.instance_count == 0) {
		LogDrawCensusLine(buffer, "DrawIndex-empty", args.index_count, args.instance_count, -1, -1,
		                  -1);
		return;
	}
	if (SkipCubeDraw(buffer)) {
		return;
	}

	if (FrameGraph::Logging()) {
		FrameGraph::BeginOp();
		FrameGraph::Log("draw %s count=%u instances=%u", "DrawIndex", args.index_count,
		                args.instance_count);
	}
	GpuPhases::Scope consume_phase(GpuPhases::Phase::DrawConsume);
	if (const char* consumed = ConsumeDrawAsOperation(buffer, args.render_target_slice_offset)) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  consumed as %s", consumed);
		}
		ResetBindings();
		return;
	}
	consume_phase.End();

	if (!DrawHasValidVertexShader(sh_ctx)) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not drawn: no vertex shader");
		}
		return;
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
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not drawn: primitive type %u",
			                static_cast<uint32_t>(ucfg.GetPrimType()));
		}
		return;
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

	DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count, args.instance_count,
	                   args.first_instance};
	DrawStateLease   state_lease(ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch);
	DrawRenderState& state = state_lease.Get();
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}
	if (args.gpu_args_address != 0 &&
	    state.vertex_info[0].stage.program->stage == ShaderType::Mesh) {
		// Mesh draws need the counts for their group count: read here.
		args = ReadIndexedIndirectArgs(args);
		if (args.index_count == 0 || args.instance_count == 0) {
			ResetBindings();
			return;
		}
		index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
		index_source.size =
		    static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
		draw = {CommandBufferDebugOp::DrawIndex, args.index_count, args.instance_count,
		        args.first_instance};
	}

	// The guest's primitive reset in a list draw (round 80, RewriteListReset): after the draw's
	// programs are known, as a mesh-shader vertex stage reads the guest's indices itself.
	std::vector<uint8_t> list_reset_indices;
	if (args.gpu_args_address == 0) {
		const auto* vertex_program = state.vertex_info[0].stage.program;
		const bool  mesh = vertex_program != nullptr && vertex_program->stage == ShaderType::Mesh;
		if (const auto kept = RewriteListReset(buffer, m_context.GetBufferCache(), index_source,
		                                       args.index_count, mesh, list_reset_indices);
		    kept != UINT32_MAX) {
			if (kept == 0) {
				if (FrameGraph::Logging()) {
					FrameGraph::Log("  not drawn: the primitive reset index drops every primitive");
				}
				ResetBindings();
				return;
			}
			args.index_count       = kept;
			index_source.host_data = list_reset_indices.data();
			index_source.size      = list_reset_indices.size();
			draw = {CommandBufferDebugOp::DrawIndex, args.index_count, args.instance_count,
			        args.first_instance};
		}
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	emit.vertex_offset  = vertex_offset + args.base_vertex;
	emit.first_instance = instance_offset;
	emit.gpu_args_address = args.gpu_args_address;
	if (FrameGraph::Logging()) {
		// Round 105: which instances and vertices the draw takes (a crowd pass's people).
		FrameGraph::Log("  offsets: first instance %u, vertex offset %d, index offset %u%s",
		                emit.first_instance, emit.vertex_offset, ucfg.GetIndexOffset(),
		                indirect ? " (the GPU's arguments)" : "");
	}
	LogWatchedDraw(draw, state, ucfg.GetIndexOffset(), args.base_vertex, 0, emit.vertex_offset,
	               emit.first_instance, indirect);

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();
	const GpuPhases::Scope draw_phase(GpuPhases::Phase::Draw);
	const GpuPhases::OpStats op_stats(false);
	// KYTY_BISECT's slow:gpu (round 114).
	Bisect::SpinMicroseconds(Bisect::SlowGpuMicroseconds());
	// A tessellated draw's time, whole (round 78).
	std::optional<GpuPhases::Scope> tess_phase;
	if (buffer.GetUserConfig().GetPrimType() == Prospero::PrimitiveType::kPatch) {
		tess_phase.emplace(GpuPhases::Phase::TessDraw);
	}

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawPending);
		m_context.GetCommandScheduler().PopPendingOperations(false);
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	const RendererLock lock(m_context.GetMutex());
	if (args.gpu_args_address == 0 &&
	    (args.vertex_count == 0 ||
	     (args.instance_count == 0 && args.gpu_instance_count_address == 0))) {
		LogDrawCensusLine(buffer, "DrawIndexAuto-empty", args.vertex_count, args.instance_count, -1,
		                  -1, -1);
		return;
	}
	if (SkipCubeDraw(buffer)) {
		return;
	}

	if (FrameGraph::Logging()) {
		FrameGraph::BeginOp();
		FrameGraph::Log("draw %s count=%u instances=%u", "DrawIndexAuto", args.vertex_count,
		                args.instance_count);
	}
	GpuPhases::Scope consume_phase(GpuPhases::Phase::DrawConsume);
	if (const char* consumed = ConsumeDrawAsOperation(buffer, args.render_target_slice_offset)) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  consumed as %s", consumed);
		}
		ResetBindings();
		return;
	}
	consume_phase.End();

	if (!DrawHasValidVertexShader(sh_ctx)) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not drawn: no vertex shader");
		}
		return;
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
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not drawn: primitive type %u",
			                static_cast<uint32_t>(ucfg.GetPrimType()));
		}
		ResetBindings();
		return;
	}
	DrawStateLease   state_lease(ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch);
	DrawRenderState& state = state_lease.Get();
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	// Counts read on the GPU work for fast-launch mesh draws and vertex-pipeline draws with one
	// draw call; others read the guest arguments here (waiting for the GPU when it wrote them).
	DrawAutoArgs drawn = args;
	if (args.gpu_args_address != 0) {
		const bool mesh   = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
		const bool gpu_ok = mesh ? state.vertex_info[0].mesh.fast_launch
		                         : ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy;
		if (!gpu_ok) {
			uint32_t words[4] {};
			std::memcpy(words, reinterpret_cast<const void*>(args.gpu_args_address), sizeof(words));
			drawn.vertex_count     = words[0];
			drawn.instance_count   = words[1];
			drawn.first_vertex     = words[2];
			drawn.first_instance   = words[3];
			drawn.gpu_args_address = 0;
			if (drawn.vertex_count == 0 || drawn.instance_count == 0) {
				ResetBindings();
				return;
			}
		}
	}
	if (args.gpu_instance_count_address != 0) {
		const bool mesh   = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
		const bool gpu_ok = !mesh && ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy;
		if (!gpu_ok) {
			std::memcpy(&drawn.instance_count,
			            reinterpret_cast<const void*>(args.gpu_instance_count_address),
			            sizeof(uint32_t));
			drawn.gpu_instance_count_address = 0;
			if (drawn.instance_count == 0) {
				ResetBindings();
				return;
			}
		}
	}
	const DrawCallInfo drawn_info {CommandBufferDebugOp::DrawIndexAuto, drawn.vertex_count,
	                               drawn.instance_count, drawn.first_instance};

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = drawn.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, drawn.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(drawn.first_vertex));
	emit.first_instance = instance_offset;
	emit.gpu_args_address = drawn.gpu_args_address;
	emit.gpu_instance_count_address = drawn.gpu_instance_count_address;
	LogWatchedDraw(drawn_info, state, ucfg.GetIndexOffset(), 0, drawn.first_vertex,
	               static_cast<int32_t>(emit.first_vertex), emit.first_instance, indirect);

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, drawn_info, state, topology, emit, index_source, false);
	ResetBindings();
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
	// A target can lie inside a larger image (a face of a cube or a slice of an array that the
	// guest points a color target at): FindImage then moved the view to that level and layer,
	// which is where the resolve has to read and write. It resolved into the register view's
	// layer (always 0), so every face landed in the first (from the GTA V fork, 6bb0811e).
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

} // namespace Libs::Graphics
