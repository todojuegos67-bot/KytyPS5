#include "graphics/shader/shader.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/frameGraph.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderVertexMetadata.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace Libs::Graphics {

struct ShaderMapEntry {
	ShaderMappedData data;
	uint64_t hash;
};

static std::unique_ptr<std::unordered_map<uint64_t, ShaderMapEntry>> g_shader_map;
static std::mutex                                                      g_shader_map_mutex;
// Bumped by every ShaderMapUserData: each thread's copies of entries (FindShaderMapEntry) are
// current while it stays the same.
static std::atomic<uint64_t> g_shader_map_generation {1};

// KYTY_SHADER_MAP_CACHE=0: every lookup takes the map's lock, as before round 78.
static bool ShaderMapCacheEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHADER_MAP_CACHE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// The entry ShaderMapUserData registered for addr. Each draw looks up its stages' entries (a
// tessellated one five times, a dispatch twice), and so do the draw-prep scanner's threads: all of
// them took this one lock. Entries are only ever replaced, never removed, so a thread keeps copies
// of the ones it asked for, until the next registration.
static bool FindShaderMapEntry(uint64_t addr, ShaderMapEntry& out) {
	struct Slot {
		uint64_t       addr       = 0;
		uint64_t       generation = 0;
		ShaderMapEntry entry {};
	};
	thread_local std::array<Slot, 32> slots {};
	const auto generation = g_shader_map_generation.load(std::memory_order_acquire);
	auto&      slot       = slots[((addr >> 4u) * 0x9E3779B97F4A7C15ull) >> 59u];
	const bool cached     = ShaderMapCacheEnabled();
	if (cached && slot.generation == generation && slot.addr == addr) {
		out = slot.entry;
		return true;
	}
	std::scoped_lock lock(g_shader_map_mutex);
	if (g_shader_map == nullptr) {
		return false;
	}
	const auto iter = g_shader_map->find(addr);
	if (iter == g_shader_map->end()) {
		return false;
	}
	out = iter->second;
	if (cached) {
		// The generation read before the lock: a registration since then makes this copy stale.
		slot = {addr, generation, iter->second};
	}
	return true;
}

void ShaderInit() {
	EXIT_IF(g_shader_map != nullptr);

	g_shader_map = std::make_unique<std::unordered_map<uint64_t, ShaderMapEntry>>();
}

void ShaderMapUserData(uint64_t addr, const ShaderMappedData& data) {
	EXIT_IF(g_shader_map == nullptr);

	if (addr == 0 || data.code_size_bytes == 0 || data.code_size_bytes % sizeof(uint32_t) != 0) {
		EXIT("ShaderMapUserData shader=0x%016" PRIx64 " has invalid AGC shader_size=0x%08" PRIx32 "\n",
		     addr, data.code_size_bytes);
	}
	const auto hash = XXH3_64bits(reinterpret_cast<const void*>(addr), data.code_size_bytes);
	// Draws read the input semantics from this copy, not from the header: the header can share a
	// page with memory the GPU writes each frame, and reading it per draw faults and drains the
	// pipeline. Copied before taking the lock, as the read itself can fault.
	auto mapped = data;
	if (data.input_semantics != nullptr) {
		const auto count =
		    std::min<uint32_t>(data.num_input_semantics, ShaderMappedData::MaxInputSemantics);
		std::copy_n(data.input_semantics, count, mapped.input_semantics_snapshot.begin());
	}
	std::scoped_lock lock(g_shader_map_mutex);
	(*g_shader_map)[addr] = {mapped, hash};
	g_shader_map_generation.fetch_add(1, std::memory_order_acq_rel);
}

uint64_t ShaderMapGeneration() {
	return g_shader_map_generation.load(std::memory_order_acquire);
}

static ShaderMapEntry ShaderGetMappedData(uint64_t addr, const char* label) {
	EXIT_IF(g_shader_map == nullptr);

	if (ShaderMapEntry entry; FindShaderMapEntry(addr, entry)) {
		return entry;
	}

	EXIT("%s shader=0x%016" PRIx64 " is missing from ShaderMap\n", label, addr);
}

static ShaderParams GetShaderParams(uint64_t shader_addr, uint64_t hash,
	                                std::span<const uint32_t> user_data,
	                                const ShaderMappedData& data, uint32_t user_data_base = 0) {
	const auto code_words = data.code_size_bytes / sizeof(uint32_t);
	const auto code = std::span {reinterpret_cast<const uint32_t*>(shader_addr), code_words};
	ShaderParams params {
	    .code            = code,
	    .user_data_count = static_cast<uint32_t>(user_data.size()) + user_data_base,
	    .hash            = hash,
	};
	EXIT_IF(user_data.size() > HW::UserSgprInfo::SGPRS_MAX ||
	        params.user_data_count > params.user_data.size());
	std::copy(user_data.begin(), user_data.end(), params.user_data.begin() + user_data_base);
	return params;
}

#if 0
// Kept as disabled debugging guards for investigating unusual stage register state.
static void vs_check(const HW::VertexShaderInfo& vs, const HW::ShaderRegisters& sh) {
	const auto is_zero_or_wave64_subgroup = [](uint32_t value) {
		return value == 0 || value <= 0x40;
	};
	const auto is_known_gs_out_prim_type = [](uint32_t value) {
		switch (static_cast<Prospero::GsOutputPrimitiveType>(value)) {
			case Prospero::GsOutputPrimitiveType::kPoints:
			case Prospero::GsOutputPrimitiveType::kLines:
			case Prospero::GsOutputPrimitiveType::kTriangles:
			case Prospero::GsOutputPrimitiveType::k2dRectangle:
			case Prospero::GsOutputPrimitiveType::kRectList: return true;
		}

		return false;
	};
	const bool ps5_ngg_passthrough_triangle_path =
	    vs.es_regs.data_addr != 0 && vs.gs_regs.data_addr == vs.es_regs.data_addr &&
	    sh.m_geNggSubgrpCntl == 0x00000001 && sh.m_vgtGsMaxVertOut == 0x00000003 &&
	    sh.m_vgtGsOutPrimType == 0x00000002 && sh.m_geMaxOutputPerSubgroup <= 0x000000c0;

	if (vs.es_regs.data_addr != 0 || vs.gs_regs.data_addr != 0) {
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.priority != 0);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.float_mode != 192);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.dx10_clamp != true);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.debug_mode != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.ieee_mode != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.cu_group_enable != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.require_forward_progress != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.threadgroup_configuration != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.gs_vgpr_component_count != 3);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.fp16_overflow != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.scratch_en != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.offchip_lds != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.es_vgpr_component_count != 3);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.lds_size != 0);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.shared_vgprs != 0);
	}

	for (uint32_t value = sh.m_spiShaderPosFormat; value != 0; value >>= 4u) {
		EXIT_NOT_IMPLEMENTED((value & 0xfu) != 0 && (value & 0xfu) != 0x4u);
	}
	if (sh.m_paClVsOutCntl != 0x00000000) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: accepting PA_CL_VS_OUT_CNTL = 0x%08" PRIx32 "\n",
			     sh.m_paClVsOutCntl);
			logged = true;
		}
	}

	EXIT_NOT_IMPLEMENTED(sh.m_spiShaderIdxFormat != 0x00000000 &&
	                     sh.m_spiShaderIdxFormat != 0x00000001);
	EXIT_NOT_IMPLEMENTED(sh.m_geNggSubgrpCntl != 0x00000000 && sh.m_geNggSubgrpCntl != 0x00000001);
	EXIT_NOT_IMPLEMENTED(sh.m_vgtGsInstanceCnt != 0x00000000);
	EXIT_NOT_IMPLEMENTED(!is_zero_or_wave64_subgroup(sh.GetEsVertsPerSubgrp()));
	EXIT_NOT_IMPLEMENTED(!is_zero_or_wave64_subgroup(sh.GetGsPrimsPerSubgrp()));
	EXIT_NOT_IMPLEMENTED(!is_zero_or_wave64_subgroup(sh.GetGsInstPrimsInSubgrp()));
	EXIT_NOT_IMPLEMENTED(!is_zero_or_wave64_subgroup(sh.m_geMaxOutputPerSubgroup) &&
	                     !ps5_ngg_passthrough_triangle_path);
	EXIT_NOT_IMPLEMENTED(sh.m_vgtEsgsRingItemsize != 0x00000000 &&
	                     sh.m_vgtEsgsRingItemsize != 0x00000004);
	EXIT_NOT_IMPLEMENTED(sh.m_vgtGsMaxVertOut != 0x00000000 && !ps5_ngg_passthrough_triangle_path);
	EXIT_NOT_IMPLEMENTED(!is_known_gs_out_prim_type(sh.m_vgtGsOutPrimType));
}

static void ps_check(const HW::PsStageRegisters& ps, const HW::ShaderRegisters& sh) {
	if (sh.target_output_mode[0] != 0 && sh.target_output_mode[0] != 2 &&
	    sh.target_output_mode[0] != 4 && sh.target_output_mode[0] != 5 &&
	    sh.target_output_mode[0] != 6 && sh.target_output_mode[0] != 7 &&
	    sh.target_output_mode[0] != 9) {
		EXIT("Not implemented (sh.target_output_mode[0] != 0 && sh.target_output_mode[0] != 2 && "
		     "sh.target_output_mode[0] != 4 && sh.target_output_mode[0] != 5 && "
		     "sh.target_output_mode[0] != 6 && sh.target_output_mode[0] != 7 && "
		     "sh.target_output_mode[0] != 9)\n");
	}
	EXIT_NOT_IMPLEMENTED(sh.db_shader_control.conservative_z_export_value != 0x00000000);
	EXIT_NOT_IMPLEMENTED(sh.db_shader_control.shader_z_behavior != 0x00000001 &&
	                     sh.db_shader_control.shader_z_behavior != 0x00000000);
	// EXIT_NOT_IMPLEMENTED(ps.shader_kill_enable != false);
	// EXIT_NOT_IMPLEMENTED(ps.shader_execute_on_noop != false);
	// EXIT_NOT_IMPLEMENTED(ps.m_spiShaderPgmRsrc1Ps != 0x002c0000);
	// EXIT_NOT_IMPLEMENTED(ps.m_spiShaderPgmRsrc2Ps != 0x00000000);
	// EXIT_NOT_IMPLEMENTED(ps.vgprs != 0x00 && ps.vgprs != 0x01);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.priority != 0);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.float_mode != 192);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.dx10_clamp != true);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.debug_mode != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.ieee_mode != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.cu_group_disable != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.require_forward_progress != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.fp16_overflow != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc2.scratch_en != false);
	// EXIT_NOT_IMPLEMENTED(ps.user_sgpr != 0 && ps.user_sgpr != 4 && ps.user_sgpr != 12);
	EXIT_NOT_IMPLEMENTED(ps.rsrc2.wave_cnt_en != false);
	if (ps.rsrc2.extra_lds_size != 0) {
		static std::atomic_uint log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("\t PS extra LDS reservation = 0x%02" PRIx8 ", continuing\n",
			     ps.rsrc2.extra_lds_size);
		}
	}
	EXIT_NOT_IMPLEMENTED(ps.rsrc2.raster_ordered_shading != 0);
	EXIT_NOT_IMPLEMENTED(ps.rsrc2.shared_vgprs != 0);

	if (sh.shader_z_format != 0x00000000 && sh.shader_z_format != 0x00000001 &&
	    !sh.db_shader_control.shader_z_export_enable) {
		static std::atomic_uint log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("\t shader_z_format = 0x%08" PRIx32
			     " with z export disabled, ignoring depth export format\n",
			     sh.shader_z_format);
		}
	}
	EXIT_NOT_IMPLEMENTED(sh.db_shader_control.shader_z_export_enable &&
	                     sh.shader_z_format != 0x00000000 && sh.shader_z_format != 0x00000001);
	constexpr uint32_t ps_input_linear_center = 0x00000020u;
	constexpr uint32_t ps_input_pos_w         = 0x00000800u;
	constexpr uint32_t ps_input_front_face    = 0x00001000u;
	constexpr uint32_t supported_ps_input_bits =
	    0x00000702u | ps_input_linear_center | ps_input_pos_w | ps_input_front_face;
	EXIT_NOT_IMPLEMENTED((sh.ps_input_ena & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED((sh.ps_input_addr & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED(sh.ps_input_ena != sh.ps_input_addr);
	// EXIT_NOT_IMPLEMENTED(ps.m_spiPsInControl != 0x00000000);
	constexpr uint32_t baryc_persp_mask =
	    0x00000003u | 0x00000030u | 0x00000300u | 0x00003000u;
	constexpr uint32_t baryc_linear_mask = 0x00030000u | 0x00300000u | 0x03000000u;
	constexpr uint32_t baryc_known_mask  = baryc_persp_mask | baryc_linear_mask;
	EXIT_NOT_IMPLEMENTED((sh.baryc_cntl & ~baryc_known_mask) != 0);
	EXIT_NOT_IMPLEMENTED((sh.baryc_cntl & baryc_persp_mask) != 0);
	if ((sh.ps_input_ena & ps_input_linear_center) == 0 && (sh.baryc_cntl & baryc_linear_mask) != 0) {
		static std::atomic_uint log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("\t ignoring inactive linear SPI_BARYC_CNTL bits: 0x%08" PRIx32 "\n",
			     sh.baryc_cntl & baryc_linear_mask);
		}
	} else {
		EXIT_NOT_IMPLEMENTED((sh.baryc_cntl & baryc_linear_mask) != 0x00000000 &&
		                     (sh.baryc_cntl & baryc_linear_mask) != 0x01000000);
	}
	if ((sh.m_cbShaderMask & 0x0000000f) != 0x0000000f) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: accepting partial CB_SHADER_MASK = 0x%08" PRIx32 "\n",
			     sh.m_cbShaderMask);
			logged = true;
		}
	}
	if ((sh.m_cbShaderMask & ~0x0000000fu) != 0) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: ignoring extra CB_SHADER_MASK MRT bits: 0x%08" PRIx32 "\n",
			     sh.m_cbShaderMask);
			logged = true;
		}
	}

	if (sh.db_shader_control.other_bits != 0x00000000) {
		static std::atomic_uint log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("\t temporary: ignoring unsupported DB_SHADER_CONTROL bits 0x%08" PRIx32 "\n",
			     sh.db_shader_control.other_bits);
		}
	}
	EXIT_NOT_IMPLEMENTED(sh.m_paScShaderControl != 0x00000000);
}

static void cs_check(const HW::CsStageRegisters& cs, const HW::ShaderRegisters& /*sh*/) {
	// EXIT_NOT_IMPLEMENTED(cs.num_thread_x != 0x00000040);
	// EXIT_NOT_IMPLEMENTED(cs.num_thread_y != 0x00000001);
	// EXIT_NOT_IMPLEMENTED(cs.num_thread_z != 0x00000001);
	// EXIT_NOT_IMPLEMENTED(cs.vgprs != 0x00 && cs.vgprs != 0x01);
	EXIT_NOT_IMPLEMENTED(cs.priority != 0x00);
	EXIT_NOT_IMPLEMENTED(cs.debug_mode != false);
	EXIT_NOT_IMPLEMENTED(cs.require_forward_progress != false);
	EXIT_NOT_IMPLEMENTED(cs.shared_vgprs != 0x00);
	EXIT_NOT_IMPLEMENTED(cs.scratch_en != 0x00);
	// EXIT_NOT_IMPLEMENTED(cs.user_sgpr != 0x0c);
	if (cs.tgid_x_en == 0x00) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: compute shader has TGID X disabled\n");
			logged = true;
		}
	} else {
		EXIT_NOT_IMPLEMENTED(cs.tgid_x_en != 0x01);
	}
	// EXIT_NOT_IMPLEMENTED(cs.tgid_y_en != 0x00);
	// EXIT_NOT_IMPLEMENTED(cs.tgid_z_en != 0x00);
	EXIT_NOT_IMPLEMENTED(cs.tg_size_en != 0x00);
	EXIT_NOT_IMPLEMENTED(cs.tidig_comp_cnt > 2);

	//	EXIT_NOT_IMPLEMENTED(cs.m_computePgmRsrc1 != 0x002c0040);
	//	EXIT_NOT_IMPLEMENTED(cs.m_computePgmRsrc2 != 0x00000098);
	//	EXIT_NOT_IMPLEMENTED(cs.m_computeNumThreadX != 0x00000040);
	//	EXIT_NOT_IMPLEMENTED(cs.m_computeNumThreadY != 0x00000001);
	//	EXIT_NOT_IMPLEMENTED(cs.m_computeNumThreadZ != 0x00000001);
}
#endif

// `mismatch`, when given, is set (and the description left unfinished) where a buffer would end
// the emulator: the draw-prep scanner's descriptions, from vertex tables that may be stale.
static void ShaderDetectBuffers(ShaderVertexInputInfo& info, bool* mismatch = nullptr) {
	KYTY_PROFILER_FUNCTION();

	info.buffers_num = 0;

	for (int ri = 0; ri < info.resources_num; ri++) {
		const auto& r = info.resources[ri];

		auto& destination = info.resources_dst[ri];
		int bi = 0;
		for (; bi < info.buffers_num; bi++) {
			auto& b = info.buffers[bi];

			uint64_t stride = b.stride;

			if (stride == r.Stride() &&
			    b.fetch_index == destination.fetch_index) {
				uint64_t rbase   = r.Base48();
				uint64_t base    = std::min(rbase, b.addr);
				uint64_t offset1 = rbase - base;
				uint64_t offset2 = b.addr - base;

				if (offset1 < stride && offset2 < stride) {
					if (mismatch != nullptr && b.num_records != r.NumRecords()) {
						*mismatch = true;
						return;
					}
					EXIT_NOT_IMPLEMENTED(b.num_records != r.NumRecords());
					b.addr = base;
					break;
				}
			}
		}

		if (bi == info.buffers_num) {
			EXIT_NOT_IMPLEMENTED(info.buffers_num >= ShaderVertexInputInfo::RES_MAX);
			info.buffers[bi] = {.addr = r.Base48(),
			                    .stride = r.Stride(),
			                    .num_records = r.NumRecords(),
			                    .fetch_index = destination.fetch_index};
			info.buffers_num++;
		}
		destination.buffer_index = bi;
	}
}

static void ShaderApplyAttribSemantics(ShaderVertexInputInfo& info,
                                       std::span<const ShaderSemantic> input_semantics,
                                       const uint32_t* attrib, const uint32_t* buffer) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(attrib == nullptr || buffer == nullptr);

	const bool debug_dump = Config::GraphicsDebugDumpEnabled();

	for (uint32_t i = 0; i < input_semantics.size(); i++) {
		const auto& in = input_semantics[i];

		EXIT_NOT_IMPLEMENTED(in.static_vb_index == 1 || in.static_attribute == 1);

		uint32_t reg  = in.hardware_mapping;
		uint32_t size = in.size_in_elements;

		if (debug_dump) {
			LOGF("reg = %u, size = %u, va[%u] = 0x%08" PRIx32 "\n", reg, size, i,
			     attrib[in.semantic]);
		}

		size_t index = attrib[in.semantic] & 0x1fu;
		auto   format =
		    static_cast<Prospero::VertexAttribFormat>((attrib[in.semantic] >> 5u) & 0x1ffu);
		uint32_t offset      = (attrib[in.semantic] >> 14u) & 0xfffu;
		uint32_t fetch_index = (attrib[in.semantic] >> 26u) & 0x1u;

		EXIT_NOT_IMPLEMENTED(index >= ShaderVertexInputInfo::RES_MAX);

		const auto* sharp = &buffer[index * 4];

		EXIT_NOT_IMPLEMENTED(info.resources_num >= ShaderVertexInputInfo::RES_MAX);

		auto& r           = info.resources[info.resources_num];
		auto& rd          = info.resources_dst[info.resources_num];
		rd.register_start = static_cast<int>(reg);
		rd.registers_num  = static_cast<int>(size);
		rd.attr_id        = static_cast<int>(in.semantic);
		rd.fetch_index    = fetch_index;
		r.fields[0]       = sharp[0];
		r.fields[1]       = sharp[1];
		r.fields[2]       = sharp[2];
		r.fields[3]       = sharp[3];
		if (r.SwizzleEnabled() || r.AddTid()) {
			// A swizzled vertex buffer (groups of 8 << index stride vertices, each dword of a
			// vertex 4 * that many bytes from the next), or one that adds the lane to the index
			// (ADD_TID), has no Vulkan vertex input. Until the vertex fetch does it in the
			// shader, it is read as if linear: that attribute comes out wrong, the emulator keeps
			// running (each descriptor once, the first 32). GT7 binds an ADD_TID one somewhere
			// past its menus (2026-10-15).
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			const auto key = r.Base48() ^ (static_cast<uint64_t>(r.fields[3]) << 32u);
			if (logged.size() < 32 && logged.insert(key).second) {
				LOGF("Vertex input: %s vertex buffer read as if linear: attribute %u "
				     "(semantic %u) V# [0x%08x 0x%08x 0x%08x 0x%08x] base 0x%012" PRIx64
				     " stride %u records %u index stride %u attribute format %u offset %u\n",
				     r.SwizzleEnabled() ? (r.AddTid() ? "swizzled ADD_TID" : "swizzled")
				                        : "ADD_TID",
				     i, static_cast<uint32_t>(in.semantic), r.fields[0], r.fields[1], r.fields[2],
				     r.fields[3], r.Base48(), r.Stride(), r.NumRecords(), 8u << r.IndexStride(),
				     static_cast<uint32_t>(format), offset);
			}
		}
		if (format != Prospero::VertexAttribFormat::kInvalid) {
			const auto                   format_raw    = static_cast<uint32_t>(format);
			const auto                   buffer_format = format_raw >> 2u;
			const auto                   channels      = (format_raw & 3u) + 1u;
			// AGC vertex formats encode the buffer format above the two channel-count bits.
			// The fetch prolog selects X001, XY01, XYZ1, or XYZW from that count.
			r.fields[3] = (r.fields[3] & ~((0x7fu << 12u) | 0xfffu)) |
			              (buffer_format << 12u) |
			              DstSel(4, channels > 1u ? 5u : 0u, channels > 2u ? 6u : 0u,
			                     channels > 3u ? 7u : 1u);
		}
		if (offset != 0) {
			r.UpdateAddress48(r.Base48() + offset);
		}

		info.resources_num++;
	}
}

static uint32_t ShaderCalcPsSystemInputBase(const HW::ShaderRegisters& regs) {
	constexpr uint32_t ps_input_persp_sample    = 0x00000001u;
	constexpr uint32_t ps_input_persp_center    = 0x00000002u;
	constexpr uint32_t ps_input_persp_centroid  = 0x00000004u;
	constexpr uint32_t ps_input_persp_pull      = 0x00000008u;
	constexpr uint32_t ps_input_linear_sample   = 0x00000010u;
	constexpr uint32_t ps_input_linear_center   = 0x00000020u;
	constexpr uint32_t ps_input_linear_centroid = 0x00000040u;
	constexpr uint32_t ps_input_line_stipple    = 0x00000080u;
	constexpr uint32_t ps_input_pos_xy          = 0x00000300u;
	constexpr uint32_t ps_input_pos_z           = 0x00000400u;
	constexpr uint32_t ps_input_pos_w           = 0x00000800u;
	constexpr uint32_t ps_input_front_face      = 0x00001000u;
	constexpr uint32_t ps_input_ancillary       = 0x00002000u;
	constexpr uint32_t ps_input_sample_coverage = 0x00004000u;
	constexpr uint32_t ps_input_pos_fixed_pt    = 0x00008000u;
	constexpr uint32_t supported_ps_input_bits =
	    ps_input_persp_sample | ps_input_persp_center | ps_input_persp_centroid |
	    ps_input_persp_pull | ps_input_linear_sample | ps_input_linear_center |
	    ps_input_linear_centroid | ps_input_line_stipple | ps_input_pos_xy | ps_input_pos_z |
	    ps_input_pos_w | ps_input_front_face | ps_input_ancillary | ps_input_sample_coverage |
	    ps_input_pos_fixed_pt;

	EXIT_NOT_IMPLEMENTED((regs.ps_input_ena & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED((regs.ps_input_addr & ~supported_ps_input_bits) != 0);
	// SPI_PS_INPUT_ADDR lays the VGPRs out, SPI_PS_INPUT_ENA is the part of them the hardware
	// fills: an input in ADDR alone keeps its VGPRs, with undefined values
	// (ShaderGetStaticInputInfoPS). Each pair of them once in the log.
	if (regs.ps_input_ena != regs.ps_input_addr) {
		static std::mutex                   mutex;
		static std::unordered_set<uint64_t> reported;
		bool                                first = false;
		{
			std::lock_guard lock(mutex);
			first =
			    reported.size() < 64 &&
			    reported
			        .insert((static_cast<uint64_t>(regs.ps_input_ena) << 32u) | regs.ps_input_addr)
			        .second;
		}
		if (first) {
			::printf("PS inputs: SPI_PS_INPUT_ENA = 0x%08" PRIx32
			         ", SPI_PS_INPUT_ADDR = 0x%08" PRIx32 ", the VGPRs laid out as ADDR\n",
			         regs.ps_input_ena, regs.ps_input_addr);
			LOGF("PS inputs: SPI_PS_INPUT_ENA = 0x%08" PRIx32 ", SPI_PS_INPUT_ADDR = 0x%08" PRIx32
			     ", the VGPRs laid out as ADDR\n",
			     regs.ps_input_ena, regs.ps_input_addr);
		}
	}

	const uint32_t inputs = regs.ps_input_addr;
	uint32_t       reg    = 0;
	if ((inputs & ps_input_persp_sample) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_center) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_centroid) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_pull) != 0) {
		reg += 3;
	}
	if ((inputs & ps_input_linear_sample) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_linear_center) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_linear_centroid) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_line_stipple) != 0) {
		reg += 1;
	}
	return reg;
}

// Whether this is the first report of `kind` for the shader: a draw that fails to describe its
// vertex inputs is skipped (PrepareProgram), and a scene can issue it every frame.
static bool FirstVertexInputReport(uint64_t shader_addr, uint32_t kind) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> reported;
	std::lock_guard                     lock(mutex);
	return reported.insert(shader_addr ^ (static_cast<uint64_t>(kind) << 56u)).second;
}

// The user SGPRs that hold a vertex shader's vertex table pointers, as the draw path read them from
// its AGC header (round 78): the draw-prep scanner describes a tessellated draw's LS stage from
// these, without reading the header in guest memory itself. Current while the shader map's
// generation stays (a shader registered again at the address may have other ones).
struct VertexTableRegs {
	int      buffer_reg = -1;
	int      attrib_reg = -1;
	uint64_t generation = 0;
};
static std::mutex                                    g_vertex_table_regs_mutex;
static std::unordered_map<uint64_t, VertexTableRegs> g_vertex_table_regs;

static void NoteVertexTableRegs(uint64_t shader_addr, const ShaderVertexMetadata& metadata) {
	const auto generation = g_shader_map_generation.load(std::memory_order_acquire);
	// Each thread notes a shader once a generation.
	thread_local std::unordered_map<uint64_t, uint64_t> noted;
	if (noted.size() > 65536) {
		noted.clear();
	}
	auto& noted_generation = noted[shader_addr];
	if (noted_generation == generation) {
		return;
	}
	noted_generation = generation;
	std::scoped_lock lock(g_vertex_table_regs_mutex);
	if (g_vertex_table_regs.size() > 65536) {
		g_vertex_table_regs.clear();
	}
	g_vertex_table_regs[shader_addr] = {metadata.vertex_buffer_reg, metadata.vertex_attrib_reg,
	                                    generation};
}

static bool FindVertexTableRegs(uint64_t shader_addr, VertexTableRegs& regs) {
	const auto       generation = g_shader_map_generation.load(std::memory_order_acquire);
	std::scoped_lock lock(g_vertex_table_regs_mutex);
	const auto       found = g_vertex_table_regs.find(shader_addr);
	if (found == g_vertex_table_regs.end() || found->second.generation != generation) {
		return false;
	}
	regs = found->second;
	return true;
}

// ShaderGetStaticVertexInputInfo for the draw-prep scanner (round 78), from its copy of the
// registers, which can be stale: the table registers the draw path found in the header, the
// tables' words copied from memory no GPU work has written (never a fault, never a pointer
// followed), and false wherever the draw path would end the emulator or log.
static bool ShaderGetStaticVertexInputInfoForScan(uint64_t                   shader_addr,
                                                  const HW::UserSgprInfo&    user_sgpr,
                                                  const HW::ShaderRegisters& sh,
                                                  const ShaderMappedData&    data,
                                                  ShaderVertexInputInfo&     info) {
	info                     = {};
	info.pa_cl_vs_out_cntl   = sh.m_paClVsOutCntl;
	info.scratch_size_dwords = data.scratch_size_dwords;
	VertexTableRegs regs;
	if (shader_addr == 0 || !FindVertexTableRegs(shader_addr, regs)) {
		return false;
	}
	if (regs.attrib_reg < 0) {
		return true;
	}
	constexpr int Sgprs = HW::UserSgprInfo::SGPRS_MAX;
	if (regs.buffer_reg < 0 || regs.attrib_reg + 1 >= Sgprs || regs.buffer_reg + 1 >= Sgprs ||
	    data.input_semantics == nullptr || data.num_input_semantics == 0 ||
	    data.num_input_semantics > ShaderVertexInputInfo::RES_MAX) {
		return false;
	}
	info.fetch_external   = false;
	info.fetch_embedded   = true;
	info.fetch_attrib_reg = regs.attrib_reg;
	info.fetch_buffer_reg = regs.buffer_reg;
	const auto attrib_address =
	    static_cast<uint64_t>(user_sgpr.value[regs.attrib_reg]) |
	    (static_cast<uint64_t>(user_sgpr.value[regs.attrib_reg + 1]) << 32u);
	const auto buffer_address =
	    static_cast<uint64_t>(user_sgpr.value[regs.buffer_reg]) |
	    (static_cast<uint64_t>(user_sgpr.value[regs.buffer_reg + 1]) << 32u);
	if (attrib_address == 0 || buffer_address == 0) {
		return false;
	}
	const auto semantics    = std::span<const ShaderSemantic>(data.input_semantics_snapshot.data(),
	                                                          data.num_input_semantics);
	uint32_t   max_semantic = 0;
	for (const auto& in: semantics) {
		if (in.static_vb_index == 1 || in.static_attribute == 1) {
			return false;
		}
		max_semantic = std::max<uint32_t>(max_semantic, in.semantic);
	}
	std::array<uint32_t, 256> attrib {};
	if (!LibKernel::Memory::TryReadGpuCleanBacking(attrib_address, attrib.data(),
	                                               (max_semantic + 1u) * sizeof(uint32_t))) {
		return false;
	}
	uint32_t max_index = 0;
	for (const auto& in: semantics) {
		max_index = std::max<uint32_t>(max_index, attrib[in.semantic] & 0x1fu);
	}
	std::array<uint32_t, ShaderVertexInputInfo::RES_MAX * 4> buffer {};
	if (!LibKernel::Memory::TryReadGpuCleanBacking(buffer_address, buffer.data(),
	                                               (max_index + 1u) * 4u * sizeof(uint32_t))) {
		return false;
	}
	ShaderApplyAttribSemantics(info, semantics, attrib.data(), buffer.data());
	bool mismatch = false;
	ShaderDetectBuffers(info, &mismatch);
	return !mismatch;
}

static bool ShaderGetStaticVertexInputInfo(uint64_t shader_addr, const HW::UserSgprInfo& user_sgpr,
                                           uint32_t user_sgpr_num, const HW::ShaderRegisters& sh,
                                           const ShaderMappedData& data,
                                           ShaderVertexInputInfo&  info) {
	KYTY_PROFILER_FUNCTION();

	info = {};

	info.pa_cl_vs_out_cntl = sh.m_paClVsOutCntl;

	EXIT_NOT_IMPLEMENTED(shader_addr == 0);
	info.scratch_size_dwords = data.scratch_size_dwords;

	if (data.user_data == nullptr && FirstVertexInputReport(shader_addr, 0)) {
		LOGF("ShaderGetVertexInputInfo(): no AGC user data for shader=0x%016" PRIx64
		     " user_sgpr_num=%u\n",
		     shader_addr, user_sgpr_num);
	}
	ShaderVertexMetadata metadata;
	std::string          metadata_error;
	if (!ShaderReadVertexMetadata(data, HW::UserSgprInfo::SGPRS_MAX, metadata, &metadata_error)) {
		if (FirstVertexInputReport(shader_addr, 1)) {
			LOGF("ShaderGetVertexInputInfo(): invalid AGC metadata shader=0x%016" PRIx64 ": %s\n",
			     shader_addr, metadata_error.c_str());
		}
		return false;
	}
	NoteVertexTableRegs(shader_addr, metadata);

	if (metadata.vertex_attrib_reg >= 0) {
		info.fetch_external   = false;
		info.fetch_embedded   = true;
		info.fetch_attrib_reg = metadata.vertex_attrib_reg;
		info.fetch_buffer_reg = metadata.vertex_buffer_reg;

		const auto* attrib = reinterpret_cast<const uint32_t*>(
		    static_cast<uint64_t>(user_sgpr.value[metadata.vertex_attrib_reg]) |
		    (static_cast<uint64_t>(user_sgpr.value[metadata.vertex_attrib_reg + 1]) << 32u));
		const auto* buffer = reinterpret_cast<const uint32_t*>(
		    static_cast<uint64_t>(user_sgpr.value[metadata.vertex_buffer_reg]) |
		    (static_cast<uint64_t>(user_sgpr.value[metadata.vertex_buffer_reg + 1]) << 32u));

		if (attrib == nullptr || buffer == nullptr) {
			if (FirstVertexInputReport(shader_addr, 2)) {
				LOGF("ShaderGetVertexInputInfo(): null vertex table pointer shader=0x%016" PRIx64
				     " attrib_reg=%d buffer_reg=%d user_sgpr_num=%u\n",
				     shader_addr, metadata.vertex_attrib_reg, metadata.vertex_buffer_reg,
				     user_sgpr_num);
			}
			return false;
		}
		// Round 114: the tables are guest memory the user data points at, checked before they are
		// read. A draw whose user data held no table address there (two time trials, issue 8:
		// 0xb081083800000012) ended the emulator with an access violation in the reads below; it
		// is skipped, as one with null tables is.
		if (!metadata.input_semantics.empty()) {
			uint32_t max_semantic = 0;
			for (const auto& in: metadata.input_semantics) {
				max_semantic = std::max<uint32_t>(max_semantic, in.semantic);
			}
			bool readable = LibKernel::Memory::IsCommittedRange(
			    reinterpret_cast<uint64_t>(attrib), (uint64_t {max_semantic} + 1u) * 4u);
			if (readable) {
				uint32_t max_index = 0;
				for (const auto& in: metadata.input_semantics) {
					max_index = std::max<uint32_t>(max_index, attrib[in.semantic] & 0x1fu);
				}
				readable = LibKernel::Memory::IsCommittedRange(reinterpret_cast<uint64_t>(buffer),
				                                               (uint64_t {max_index} + 1u) * 16u);
			}
			if (!readable) {
				if (FirstVertexInputReport(shader_addr, 3)) {
					LOGF("ShaderGetVertexInputInfo(): vertex tables outside guest memory "
					     "shader=0x%016" PRIx64 " attrib=0x%016" PRIx64
					     " (s%d) buffer=0x%016" PRIx64
					     " (s%d) user_sgpr_num=%u: the draw is skipped\n",
					     shader_addr, reinterpret_cast<uint64_t>(attrib),
					     metadata.vertex_attrib_reg, reinterpret_cast<uint64_t>(buffer),
					     metadata.vertex_buffer_reg, user_sgpr_num);
				}
				return false;
			}
		}
		ShaderApplyAttribSemantics(info, metadata.input_semantics, attrib, buffer);
		ShaderDetectBuffers(info);
	}
	return true;
}

static void ShaderGetStaticInputInfoPS(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	const ShaderMappedData& data, ShaderPixelInputInfo& ps_info) {
	KYTY_PROFILER_FUNCTION();

	ps_info = {};
	ps_info.scratch_size_dwords = data.scratch_size_dwords;

	// SPI_PS_IN_CONTROL: NUM_INTERP occupies bits 5:0 and PS_W32_EN is bit 15.
	ps_info.input_num            = sh.ps_in_control & 0x3fu;
	if ((sh.ps_in_control & 0x8000u) != 0) {
		ps_info.wave_size = 32;
	}
	EXIT_NOT_IMPLEMENTED(ps_info.input_num > std::size(ps_info.interpolator_settings));
	ps_info.ps_system_input_base = ShaderCalcPsSystemInputBase(sh);
	// The VGPRs follow SPI_PS_INPUT_ADDR (layout_inputs); the inputs in both registers are the
	// ones the hardware fills (active_inputs). They differ only when the game sets them apart.
	const uint32_t layout_inputs = sh.ps_input_addr;
	const uint32_t active_inputs = sh.ps_input_ena & sh.ps_input_addr;
	if ((active_inputs & 0x00000002u) != 0) {
		ps_info.ps_perspective_center_vgpr = (layout_inputs & 0x00000001u) != 0 ? 2u : 0u;
	}
	if ((active_inputs & 0x00000004u) != 0) {
		ps_info.ps_perspective_centroid_vgpr = 2u * std::popcount(layout_inputs & 0x3u);
	}
	for (uint32_t i = 0; i < data.num_input_semantics && i < ps_info.input_num &&
	                     i < ShaderMappedData::MaxInputSemantics;
	     i++) {
		const auto& semantic = data.input_semantics_snapshot[i];
		if (semantic.is_custom != 0 && semantic.is_f16 == 0) {
			ps_info.custom_interpolation_mask |= 1u << i;
		}
	}
	// A position or front-face input in ADDR alone keeps its VGPR: it gets the value anyway (the
	// hardware leaves it undefined), so the VGPRs after it stay where the shader reads them. The
	// ancillary VGPR is the last of them; reading the sample index would turn on sample shading.
	ps_info.ps_pos_x                     = (layout_inputs & 0x00000100u) != 0;
	ps_info.ps_pos_y                     = (layout_inputs & 0x00000200u) != 0;
	ps_info.ps_pos_z                     = (layout_inputs & 0x00000400u) != 0;
	ps_info.ps_pos_w                     = (layout_inputs & 0x00000800u) != 0;
	ps_info.ps_front_face                = (layout_inputs & 0x00001000u) != 0;
	ps_info.ps_ancillary                 = (active_inputs & 0x00002000u) != 0;
	ps_info.ps_sample_shading            = (active_inputs & 0x00000011u) != 0;
	ps_info.ps_no_perspective            = (sh.ps_input_ena & sh.ps_input_addr & 0x00000020u) != 0;
	ps_info.ps_pixel_kill_enable         = sh.db_shader_control.shader_kill_enable;
	ps_info.ps_depth_export_enable       = sh.db_shader_control.shader_z_export_enable;
	ps_info.ps_sample_mask_export_enable = sh.db_shader_control.shader_mask_export_enable;
	// Alpha-to-mask coverage must gate depth writes, so it rules out early fragment tests.
	ps_info.ps_early_z =
	    (sh.db_shader_control.shader_z_behavior == 1 && !sh.db_shader_control.shader_kill_enable &&
	     !sh.db_shader_control.shader_z_export_enable &&
	     !sh.db_shader_control.shader_mask_export_enable && !sh.AlphaToMask());
	ps_info.ps_execute_on_noop = sh.db_shader_control.shader_execute_on_noop;

	for (uint32_t i = 0; i < ps_info.input_num; i++) {
		ps_info.interpolator_settings[i] = sh.ps_interpolator_settings[i];
	}

	for (int i = 0; i < 8; i++) {
		ps_info.target_output_mode[i]    = sh.target_output_mode[i];
		ps_info.target_export_mapping[i] = sh.target_output_mode[i] != 0
		                                       ? target_export_mapping[i]
		                                       : Prospero::ColorComponentMapping {};
	}
}

static void ShaderGetStaticInputInfoCS(const HW::ComputeShaderInfo& regs,
                                       const HW::ShaderRegisters& /*sh*/,
                                       const ShaderMappedData& data, ShaderComputeInputInfo& info) {
	const bool dispatch_thread_dimensions   = info.dispatch_thread_dimensions;
	const bool dispatch_dimensions_indirect = info.dispatch_dimensions_indirect;
	const auto host_subgroup_size           = info.host_subgroup_size;
	const auto workgroup_counts             = std::to_array(info.workgroup_counts);
	info                                    = {};
	info.dispatch_thread_dimensions         = dispatch_thread_dimensions;
	info.dispatch_dimensions_indirect       = dispatch_dimensions_indirect;
	info.host_subgroup_size                 = host_subgroup_size;
	std::ranges::copy(workgroup_counts, info.workgroup_counts);
	info.threads_num[0]                   = regs.cs_regs.num_thread_x;
	info.threads_num[1]                   = regs.cs_regs.num_thread_y;
	info.threads_num[2]                   = regs.cs_regs.num_thread_z;
	info.lds_size_dwords                  = static_cast<uint32_t>(regs.cs_regs.lds_size) * 128u;
	info.scratch_size_dwords              = data.scratch_size_dwords;
	info.float_mode                       = regs.cs_regs.float_mode;
	info.group_id[0]                      = regs.cs_regs.tgid_x_en != 0;
	info.group_id[1]                      = regs.cs_regs.tgid_y_en != 0;
	info.group_id[2]                      = regs.cs_regs.tgid_z_en != 0;
	info.wave_size                        = regs.cs_regs.wave_size;
	info.thread_ids_num      = regs.cs_regs.tidig_comp_cnt + 1;
	info.tg_size_en          = regs.cs_regs.tg_size_en != 0;

	info.workgroup_register = regs.cs_regs.user_sgpr;
}

void BuildStageStaticKey(const ShaderVertexInputInfo& info, std::vector<uint32_t>& key) {
	EXIT_IF(info.resources_num < 0 || info.resources_num > ShaderVertexInputInfo::RES_MAX);
	key.clear();
	key.push_back(static_cast<uint32_t>(info.fetch_embedded));
	key.push_back(static_cast<uint32_t>(info.fetch_attrib_reg));
	key.push_back(static_cast<uint32_t>(info.fetch_buffer_reg));
	key.push_back(info.resources_num);
	key.push_back(info.wave_size);
	key.push_back(info.scratch_size_dwords);
	key.push_back(info.pa_cl_vs_out_cntl);
	key.push_back(static_cast<uint32_t>(info.clip_space.enabled));
	if (info.clip_space.enabled) {
		for (const float value: info.clip_space.scale) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
		for (const float value: info.clip_space.offset) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
		for (const float value: info.clip_space.half_extent) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
	}

	key.push_back(info.mesh.threads_num[0]);
	if (info.mesh.threads_num[0] != 0) {
		const auto& mesh = info.mesh;
		key.insert(key.end(), {mesh.wave_size, mesh.host_subgroup_size, mesh.lds_size_dwords,
		                       mesh.scratch_size_dwords, mesh.input_primitive,
		                       mesh.primitives_per_group, mesh.vertices_per_group,
		                       mesh.max_vertices, mesh.max_primitives, mesh.provoking_vertex,
		                       static_cast<uint32_t>(mesh.fast_launch), mesh.passes});
	}
	key.push_back(info.tess.input_control_points);
	if (info.tess.input_control_points != 0) {
		const auto& tess = info.tess;
		key.insert(key.end(),
		           {tess.output_control_points, tess.ls_stride, tess.hs_stride, tess.domain,
		            tess.partitioning, tess.output_topology, tess.patch_base, tess.patch_size,
		            tess.min_level_bits, tess.max_level_bits});
	}

	for (int i = 0; i < info.resources_num; i++) {
		const auto& resource    = info.resources[i];
		const auto& destination = info.resources_dst[i];
		key.push_back(destination.registers_num);
		key.push_back(static_cast<uint32_t>(destination.attr_id));
		key.push_back(resource.fields[3] & 0x7ffffu); // Channels and format used by embedded fetch.
	}
}

void BuildStageStaticKey(const ShaderPixelInputInfo& info, std::vector<uint32_t>& key) {
	EXIT_IF(info.input_num > std::size(info.interpolator_settings));
	key.clear();
	key.push_back(info.scratch_size_dwords);
	key.push_back(info.input_num);
	key.push_back(info.wave_size);
	key.push_back(info.ps_system_input_base);
	key.push_back(info.custom_interpolation_mask);
	key.push_back(info.ps_perspective_center_vgpr);
	key.push_back(info.ps_perspective_centroid_vgpr);
	key.push_back(static_cast<uint32_t>(info.ps_pos_x));
	key.push_back(static_cast<uint32_t>(info.ps_pos_y));
	key.push_back(static_cast<uint32_t>(info.ps_pos_z));
	key.push_back(static_cast<uint32_t>(info.ps_pos_w));
	key.push_back(static_cast<uint32_t>(info.ps_front_face));
	key.push_back(static_cast<uint32_t>(info.ps_ancillary));
	key.push_back(static_cast<uint32_t>(info.ps_no_perspective));
	key.push_back(static_cast<uint32_t>(info.ps_pixel_kill_enable));
	key.push_back(static_cast<uint32_t>(info.ps_depth_export_enable));
	key.push_back(static_cast<uint32_t>(info.ps_sample_mask_export_enable));
	key.push_back(static_cast<uint32_t>(info.ps_early_z));
	key.push_back(static_cast<uint32_t>(info.dual_source_blending));
	key.push_back(static_cast<uint32_t>(info.alpha_blend_source));
	key.insert(key.end(), std::begin(info.target_output_mode), std::end(info.target_output_mode));
	key.push_back(info.target_uint_mask);
	key.push_back(info.target_sint_mask);
	key.push_back(info.target_half_float_mask);
	for (uint32_t base = 0; base < info.target_export_mapping.size(); base += 4u) {
		uint32_t packed = 0;
		for (uint32_t i = 0; i < 4u; i++) {
			packed |= static_cast<uint32_t>(info.target_export_mapping[base + i].packed)
			          << (i * 8u);
		}
		key.push_back(packed);
	}
	key.insert(key.end(), std::begin(info.interpolator_settings),
	           std::begin(info.interpolator_settings) + info.input_num);
}

void BuildStageStaticKey(const ShaderComputeInputInfo& info, std::vector<uint32_t>& key) {
	key.clear();
	key.push_back(info.workgroup_register);
	key.push_back(info.wave_size | (static_cast<uint32_t>(info.float_mode) << 8u));
	key.push_back(info.host_subgroup_size);
	key.push_back(info.thread_ids_num);
	key.push_back(info.lds_size_dwords);
	key.push_back(info.scratch_size_dwords);
	key.push_back(static_cast<uint32_t>(info.dispatch_thread_dimensions));
	key.push_back(static_cast<uint32_t>(info.dispatch_dimensions_indirect));
	for (int i = 0; i < 3; i++) {
		key.push_back(info.threads_num[i]);
		key.push_back(static_cast<uint32_t>(info.group_id[i]));
	}
	key.push_back(static_cast<uint32_t>(info.tg_size_en));
}

ShaderParams PrepareProgram(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            const HW::UserConfig& user_config, ShaderVertexInputInfo& info) {
	const auto& sh     = context.GetShaderRegisters();
	const auto [data, hash] = ShaderGetMappedData(regs.es_regs.data_addr, "ShaderGetInputInfoVS():");
	const bool merged = (context.GetShaderStages() & 0x20u) != 0;
	auto        params = GetShaderParams(
	    regs.es_regs.data_addr, hash,
	    std::span<const uint32_t>(regs.gs_user_sgpr.value, regs.gs_regs.rsrc2.user_sgpr), data,
	    merged ? 8u : 0u);
	if (!merged) {
		if (!ShaderGetStaticVertexInputInfo(regs.es_regs.data_addr, regs.gs_user_sgpr,
		                                    regs.gs_regs.rsrc2.user_sgpr, sh, data, info)) {
			// Vertex tables the guest left null, or metadata that does not parse (logged above):
			// the draw is skipped, as a shader that gives up (DescribeGraphicsStages). GT7 issued
			// such a draw on entering its garage, which ended the emulator here.
			info.prepare_failed = true;
			return params;
		}
		info.wave_size = (context.GetShaderStages() & 0x00400000u) != 0 ? 32u : 64u;
		return params;
	}
	// NGG user SGPRs start at s8; a separately compiled GS back half also receives
	// its user-data pointer in s0:s1.
	info                     = {};
	info.logical_stage       = ShaderType::Mesh;
	info.pa_cl_vs_out_cntl   = sh.m_paClVsOutCntl;
	auto& mesh               = info.mesh;
	mesh.input_primitive     = static_cast<uint32_t>(user_config.GetPrimType());
	mesh.wave_size           = (context.GetShaderStages() & 0x00400000u) != 0 ? 32u : 64u;
	mesh.max_vertices        = sh.m_geMaxOutputPerSubgroup;
	mesh.provoking_vertex    = context.GetModeControl().provoking_vtx_last ? 2u : 0u;
	mesh.lds_size_dwords     = static_cast<uint32_t>(regs.gs_regs.rsrc2.lds_size) * 128u;
	mesh.scratch_size_dwords = data.scratch_size_dwords;
	const auto fast_launch = (context.GetShaderStages() >> 19u) & 3u;
	EXIT_NOT_IMPLEMENTED(fast_launch > 1u);
	mesh.fast_launch = fast_launch != 0;
	if (data.type == Prospero::ShaderBinaryType::kGsFront) {
		EXIT_IF(regs.gs_regs.data_addr == 0);
		const auto [back, back_hash] = ShaderGetMappedData(regs.gs_regs.data_addr, "ShaderGetInputInfoGS():");
		const auto back_params =
		    GetShaderParams(regs.gs_regs.data_addr, back_hash, {}, back);
		params.back_code = back_params.code;
		params.user_data[0] = static_cast<uint32_t>(regs.gs_regs.user_data_addr);
		params.user_data[1] = static_cast<uint32_t>(regs.gs_regs.user_data_addr >> 32u);
		const uint64_t hashes[] = {params.hash, back_params.hash};
		params.hash = XXH3_64bits(hashes, sizeof(hashes));
		mesh.scratch_size_dwords = std::max(mesh.scratch_size_dwords, back.scratch_size_dwords);
	}
	const auto& group = user_config.GetGeControl();
	// Round 126: a GS whose VGT_GS_MAX_VERT_OUT is 0 emits no vertex, so the draw puts nothing
	// on screen: it is skipped (as the draws of a shader that gives up) instead of ending the
	// emulator ("unsupported GS assembly ... vertices=0", a tester's time trial, issue 8). What
	// it would write to memory besides is lost with it. Logged once per shader.
	if (sh.m_vgtGsMaxVertOut == 0u) {
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint64_t> logged;
		{
			std::lock_guard lock(logged_mutex);
			if (logged.size() < 64 && logged.insert(params.hash).second) {
				LOGF("GS 0x%016" PRIx64 " at 0x%016" PRIx64 " emits no vertex (VGT_GS_MAX_VERT_OUT "
				     "0; input primitive %u, output %u, GE %u/%u, max output %u): its draws are "
				     "skipped (round 126)\n",
				     params.hash, regs.es_regs.data_addr, mesh.input_primitive,
				     sh.m_vgtGsOutPrimType, group.primitive_group_size, group.vertex_group_size,
				     mesh.max_vertices);
			}
		}
		info.prepare_failed = true;
		return params;
	}
	if (mesh.fast_launch) {
		const auto& user_vgpr = user_config.GetGeUserVgprEn();
		EXIT_NOT_IMPLEMENTED(data.type != Prospero::ShaderBinaryType::kGs ||
		                     regs.gs_regs.rsrc1.gs_vgpr_component_count != 0u ||
		                     regs.gs_regs.rsrc2.es_vgpr_component_count != 1u ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kPointList ||
		                     group.primitive_group_size != 1u || group.vertex_group_size != 1u ||
		                     user_vgpr.vgpr1 || user_vgpr.vgpr2 || user_vgpr.vgpr3 ||
		                     mesh.max_vertices != sh.m_vgtGsMaxVertOut);
	} else {
		EXIT_NOT_IMPLEMENTED(regs.gs_regs.rsrc1.gs_vgpr_component_count != 3u ||
		                     regs.gs_regs.rsrc2.es_vgpr_component_count != 3u);
	}
	if ((user_config.GetPrimType() != Prospero::PrimitiveType::kPointList &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kLineList &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriFan &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriStrip &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList) ||
	    sh.m_vgtGsOutPrimType != 2u || sh.m_vgtGsMaxVertOut < 3u ||
	    group.vertex_group_size < mesh.InputPrimitiveSize() ||
	    mesh.max_vertices == 0u) {
		EXIT("unsupported GS assembly: input=%u output=%u vertices=%u GE=%u/%u max_output=%u\n",
		     mesh.input_primitive, sh.m_vgtGsOutPrimType, sh.m_vgtGsMaxVertOut,
		     group.primitive_group_size, group.vertex_group_size, mesh.max_vertices);
	}
	// NGG can export indexed primitives independently of the output vertex count. In particular,
	// meshlet shaders can emit more triangles than vertices - 2 (the title's forest).
	const auto primitive_amplification = sh.m_geNggSubgrpCntl & 0x1ffu;
	// Fast launch (mesh shader mode) takes its primitive capacity from PRIM_AMP_FACTOR as well:
	// a meshlet usually emits more triangles than vertices.
	mesh.max_primitives = mesh.fast_launch
	                          ? std::max(mesh.max_vertices, primitive_amplification)
	                          : group.primitive_group_size *
	                                std::max(sh.m_vgtGsMaxVertOut - 2u, primitive_amplification);
	mesh.primitives_per_group = std::min({static_cast<uint32_t>(group.primitive_group_size),
	                                      mesh.InputPrimitiveCount(group.vertex_group_size),
	                                      mesh.max_vertices / sh.m_vgtGsMaxVertOut});
	EXIT_IF(mesh.primitives_per_group == 0u);
	mesh.vertices_per_group = mesh.InputVertexCount(mesh.primitives_per_group);
	const auto explicit_threads = (sh.m_geNggSubgrpCntl >> 9u) & 0x1ffu;
	const auto output_threads = std::max({mesh.max_vertices, mesh.max_primitives, explicit_threads});
	mesh.threads_num[0] =
	    ((output_threads + mesh.wave_size - 1u) / mesh.wave_size) * mesh.wave_size;
	mesh.threads_num[1] = mesh.threads_num[2] = 1u;
	return params;
}

// The content hash ShaderMapUserData registered for the code at addr, or 0.
static uint64_t MappedShaderHash(uint64_t addr) {
	ShaderMapEntry entry;
	return FindShaderMapEntry(addr, entry) ? entry.hash : 0;
}

uint64_t ShaderDeclaredHash(uint64_t shader_addr) {
	return MappedShaderHash(shader_addr);
}

// Declared in frameGraph.h (diagnostics): not part of the recompiler's interface.
uint32_t ShaderCodeSizeBytes(uint64_t shader_addr) {
	ShaderMapEntry entry;
	return FindShaderMapEntry(shader_addr, entry) ? entry.data.code_size_bytes : 0u;
}

// Tessellation program pairs the analysis has already rejected, keyed by the registered LS/HS
// hashes and the tessellation configuration. The analysis is a full decode of both programs;
// without this a rejected pair is re-analysed on every draw that uses it.
static std::mutex                   g_rejected_tessellation_mutex;
static std::unordered_set<uint64_t> g_rejected_tessellation;
// The pairs the analysis accepted, with what it found: analysed once, not per draw.
static std::unordered_map<uint64_t, ShaderTessellationInputInfo> g_accepted_tessellation;

// Diagnostics: the guest code of each tessellation program set, once per set, into
// shader-dumps/ next to the log (as frame-graph captures write theirs), with the registers that
// configure it, for offline work on the stages.
static void DumpTessellationPrograms(const std::array<ShaderParams, 3>& params,
                                     const std::array<uint64_t, 3>&     hashes,
                                     const HW::ShaderRegisters&         sh) {
	std::error_code created;
	std::filesystem::create_directories("shader-dumps", created);
	static constexpr std::array<const char*, 3> Stages {"LS", "HS", "ES"};
	for (size_t stage = 0; stage < params.size(); stage++) {
		char path[96];
		std::snprintf(path, sizeof(path), "shader-dumps/%s_%016" PRIx64 ".bin", Stages[stage],
		              hashes[stage]);
		std::error_code checked;
		if (std::filesystem::exists(path, checked)) continue;
		if (FILE* file = std::fopen(path, "wb"); file != nullptr) {
			std::fwrite(params[stage].code.data(), sizeof(uint32_t), params[stage].code.size(),
			            file);
			std::fclose(file);
		}
		std::snprintf(path, sizeof(path), "shader-dumps/%s_%016" PRIx64 ".txt", Stages[stage],
		              hashes[stage]);
		if (FILE* file = std::fopen(path, "w"); file != nullptr) {
			std::fprintf(file,
			             "stage=%s hash=%016" PRIx64 " bytes=%zu vgt_ls_hs_config=0x%08x "
			             "vgt_tf_param=0x%08x\nuser_data=",
			             Stages[stage], hashes[stage], params[stage].code.size() * 4u,
			             sh.m_vgtLsHsConfig, sh.m_vgtTfParam);
			for (uint32_t index = 0;
			     index < params[stage].user_data_count && index < params[stage].user_data.size();
			     index++) {
				std::fprintf(file, "%08x ", params[stage].user_data[index]);
			}
			std::fprintf(file, "\n");
			std::fclose(file);
		}
	}
}

static uint64_t TessellationProgramKey(const HW::VertexShaderInfo& regs,
                                       const HW::ShaderRegisters&  sh) {
	const uint64_t words[4] = {MappedShaderHash(regs.ls_regs.data_addr),
	                           MappedShaderHash(regs.hs_regs.data_addr), sh.m_vgtLsHsConfig,
	                           sh.m_vgtTfParam};
	return XXH3_64bits(words, sizeof(words));
}

// `scan`: the draw-prep scanner's description (PrepareTessellationProgramsForScan), from a copy of
// the registers that can be stale: it takes only a program set the analysis has accepted already,
// returns false where the draw path would end the emulator, and logs and dumps nothing.
static bool PrepareTessellationProgramsImpl(const HW::VertexShaderInfo&           regs,
                                            const HW::Context&                    context,
                                            std::array<ShaderVertexInputInfo, 3>& input_info,
                                            std::array<ShaderParams, 3>& params, bool scan) {
	const auto& sh        = context.GetShaderRegisters();
	const auto  tess_key  = TessellationProgramKey(regs, sh);
	// What the analysis found for this set, when it accepted it (one lock for both questions).
	std::optional<ShaderTessellationInputInfo> accepted_tess;
	{
		std::scoped_lock lock(g_rejected_tessellation_mutex);
		if (g_rejected_tessellation.contains(tess_key)) {
			return false;
		}
		if (const auto found = g_accepted_tessellation.find(tess_key);
		    found != g_accepted_tessellation.end()) {
			accepted_tess = found->second;
		} else if (scan) {
			return false;
		}
	}
	ShaderMapEntry local_entry;
	ShaderMapEntry control_entry;
	ShaderMapEntry evaluation_entry;
	if (scan) {
		if (!FindShaderMapEntry(regs.ls_regs.data_addr, local_entry) ||
		    !FindShaderMapEntry(regs.hs_regs.data_addr, control_entry) ||
		    !FindShaderMapEntry(regs.es_regs.data_addr, evaluation_entry)) {
			return false;
		}
	} else {
		local_entry      = ShaderGetMappedData(regs.ls_regs.data_addr, "ShaderGetInputInfoLS():");
		control_entry    = ShaderGetMappedData(regs.hs_regs.data_addr, "ShaderGetInputInfoHS():");
		evaluation_entry = ShaderGetMappedData(regs.es_regs.data_addr, "ShaderGetInputInfoTES():");
	}
	const auto& [local, local_hash]           = local_entry;
	const auto& [control, control_hash]       = control_entry;
	const auto& [evaluation, evaluation_hash] = evaluation_entry;
	const bool unsupported_types = local.type != Prospero::ShaderBinaryType::kHsFront ||
	                               control.type != Prospero::ShaderBinaryType::kHsBack ||
	                               evaluation.type != Prospero::ShaderBinaryType::kGs;
	const bool unsupported_stages =
	    (context.GetShaderStages() & 0x4u) == 0 || (context.GetShaderStages() & 0x00600020u) != 0;
	// User SGPR counts are 6-bit fields: past 32 the draw path ends the emulator (GetShaderParams).
	if (scan && (unsupported_types || unsupported_stages || regs.hs_regs.user_data_addr == 0 ||
	             regs.hs_regs.rsrc2.user_sgpr > HW::UserSgprInfo::SGPRS_MAX ||
	             regs.gs_regs.rsrc2.user_sgpr > HW::UserSgprInfo::SGPRS_MAX)) {
		return false;
	}
	// Round 138: a set this path cannot run (a wave32 hull or domain stage, a geometry shader
	// after the tessellator, or other binary types) has its patch draws skipped, as a set the
	// analysis rejects, instead of ending the emulator (a replay after a long race did). Once per
	// set and stage mask: the mask says which it was.
	if (unsupported_types || unsupported_stages) {
		const uint32_t                      stages = context.GetShaderStages();
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint64_t> logged;
		std::scoped_lock                    lock(logged_mutex);
		if (logged.size() < 64u && logged.insert(tess_key ^ (uint64_t {stages} << 1u)).second) {
			LOGF("Tessellation: LS 0x%016" PRIx64 " / HS 0x%016" PRIx64 " / ES 0x%016" PRIx64
			     " not supported, its patch draws are skipped: VGT_SHADER_STAGES_EN 0x%08x "
			     "(%s%s%s%s), "
			     "binary types %u/%u/%u\n",
			     local_hash, control_hash, evaluation_hash, stages,
			     (stages & 0x4u) == 0 ? "no hull stage; " : "",
			     (stages & 0x20u) != 0 ? "a geometry shader after the tessellator; " : "",
			     (stages & 0x00200000u) != 0 ? "wave32 hull stage; " : "",
			     (stages & 0x00400000u) != 0 ? "wave32 domain stage" : "",
			     static_cast<unsigned>(local.type), static_cast<unsigned>(control.type),
			     static_cast<unsigned>(evaluation.type));
		}
		return false;
	}
	EXIT_IF(regs.hs_regs.user_data_addr == 0);

	const auto local_users      = std::span(regs.hs_user_sgpr.value, regs.hs_regs.rsrc2.user_sgpr);
	const auto evaluation_users = std::span(regs.gs_user_sgpr.value, regs.gs_regs.rsrc2.user_sgpr);
	params = {
	    GetShaderParams(regs.ls_regs.data_addr, local_hash, local_users, local),
	    GetShaderParams(regs.hs_regs.data_addr, control_hash, local_users, control, 8u),
	    GetShaderParams(regs.es_regs.data_addr, evaluation_hash, evaluation_users,
	                    evaluation),
	};
	// The fused HS back half receives its separate user-data address in s0:s1.
	// RDNA2 reserves s0:s7 before the native HS user SGPRs.
	params[1].user_data[0] = static_cast<uint32_t>(regs.hs_regs.user_data_addr);
	params[1].user_data[1] = static_cast<uint32_t>(regs.hs_regs.user_data_addr >> 32u);

	input_info = {};
	if (scan) {
		if (!ShaderGetStaticVertexInputInfoForScan(regs.ls_regs.data_addr, regs.hs_user_sgpr, sh,
		                                           local, input_info[0])) {
			return false;
		}
	} else if (!ShaderGetStaticVertexInputInfo(regs.ls_regs.data_addr, regs.hs_user_sgpr,
	                                           regs.hs_regs.rsrc2.user_sgpr, sh, local,
	                                           input_info[0])) {
		EXIT("failed to prepare local shader program\n");
	}
	input_info[0].logical_stage       = ShaderType::Local;
	input_info[1].logical_stage       = ShaderType::TessellationControl;
	input_info[1].scratch_size_dwords = control.scratch_size_dwords;
	input_info[2].logical_stage       = ShaderType::TessellationEvaluation;
	input_info[2].scratch_size_dwords = evaluation.scratch_size_dwords;
	input_info[2].pa_cl_vs_out_cntl   = sh.m_paClVsOutCntl;

	ShaderTessellationInputInfo tess {
	    .input_control_points  = (sh.m_vgtLsHsConfig >> 8u) & 0x3fu,
	    .output_control_points = (sh.m_vgtLsHsConfig >> 14u) & 0x3fu,
	    .domain                = sh.m_vgtTfParam & 0x3u,
	    .partitioning          = (sh.m_vgtTfParam >> 2u) & 0x3u,
	    .output_topology       = (sh.m_vgtTfParam >> 5u) & 0x3u,
	};
	const bool bad_points = tess.input_control_points == 0 || tess.input_control_points > 32 ||
	                        tess.output_control_points == 0 || tess.output_control_points > 32;
	if (scan && bad_points) {
		return false;
	}
	EXIT_IF(bad_points);
	const bool analysed = accepted_tess.has_value();
	if (analysed) {
		tess = *accepted_tess;
	} else {
		DumpTessellationPrograms(params, {local_hash, control_hash, evaluation_hash}, sh);
		const bool accepted =
		    ShaderRecompiler::AnalyzeTessellationPrograms(params[0].code, params[1].code, tess);
		std::scoped_lock lock(g_rejected_tessellation_mutex);
		if (!accepted) {
			// Once per pair: the analysis names its reason on the line before ("LS/HS
			// tessellation ...").
			if (g_rejected_tessellation.insert(tess_key).second) {
				LOGF("Tessellation: LS 0x%016" PRIx64 " / HS 0x%016" PRIx64 " / ES 0x%016" PRIx64
				     " (input cp %u, output cp %u, VGT_TF_PARAM 0x%08x) not supported: its patch "
				     "draws are skipped\n",
				     local_hash, control_hash, evaluation_hash, tess.input_control_points,
				     tess.output_control_points, sh.m_vgtTfParam);
			}
			return false;
		}
		if (g_accepted_tessellation.emplace(tess_key, tess).second) {
			LOGF("Tessellation: LS 0x%016" PRIx64 " / HS 0x%016" PRIx64 " / ES 0x%016" PRIx64
			     " (input cp %u, output cp %u, VGT_TF_PARAM 0x%08x) accepted\n",
			     local_hash, control_hash, evaluation_hash, tess.input_control_points,
			     tess.output_control_points, sh.m_vgtTfParam);
		}
	}
	// The tessellator's factor clamp (not part of the analysis or its cache key).
	tess.min_level_bits = sh.m_vgtHosMinTessLevel;
	tess.max_level_bits = sh.m_vgtHosMaxTessLevel;
	// KYTY_BISECT's tess:max:<f>: factors up to f at most (a larger clamp, or none, is lowered),
	// which keys new shaders. A test of whether the giant draws are tessellated (2026-10-16).
	if (const float cap = Libs::Graphics::Bisect::TessMaxLevel(); cap > 0.0f) {
		const auto max = std::bit_cast<float>(tess.max_level_bits);
		if (tess.max_level_bits == 0u || !(max <= cap)) {
			tess.max_level_bits = std::bit_cast<uint32_t>(cap);
		}
		if (std::bit_cast<float>(tess.min_level_bits) > cap) {
			tess.min_level_bits = std::bit_cast<uint32_t>(cap);
		}
	}
	static std::atomic_bool factor_ranges_logged {false};
	if (!scan && !factor_ranges_logged.load(std::memory_order_relaxed)) {
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint64_t> logged;
		std::scoped_lock                    lock(logged_mutex);
		if (logged.size() >= 8u) {
			factor_ranges_logged.store(true, std::memory_order_relaxed);
		} else if (logged.insert((uint64_t {tess.min_level_bits} << 32u) | tess.max_level_bits)
		               .second) {
			LOGF(
			    "Tessellation factor range: VGT_HOS_MIN_TESS_LEVEL %g, VGT_HOS_MAX_TESS_LEVEL %g\n",
			    static_cast<double>(std::bit_cast<float>(tess.min_level_bits)),
			    static_cast<double>(std::bit_cast<float>(tess.max_level_bits)));
		}
	}
	for (auto& stage: input_info) {
		stage.tess = tess;
	}
	return true;
}

bool PrepareTessellationPrograms(const HW::VertexShaderInfo& regs, const HW::Context& context,
                                 std::array<ShaderVertexInputInfo, 3>& input_info,
                                 std::array<ShaderParams, 3>&          params) {
	return PrepareTessellationProgramsImpl(regs, context, input_info, params, false);
}

bool PrepareTessellationProgramsForScan(const HW::VertexShaderInfo&           regs,
                                        const HW::Context&                    context,
                                        std::array<ShaderVertexInputInfo, 3>& input_info,
                                        std::array<ShaderParams, 3>&          params) {
	return PrepareTessellationProgramsImpl(regs, context, input_info, params, true);
}

ShaderParams PrepareProgram(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    ShaderPixelInputInfo&                               ps_info) {
	const auto [data, hash] = ShaderGetMappedData(regs.ps_regs.data_addr, "ShaderGetInputInfoPS():");
	ShaderGetStaticInputInfoPS(regs, sh, target_export_mapping, data, ps_info);
	return GetShaderParams(
	    regs.ps_regs.data_addr, hash,
	    std::span<const uint32_t>(regs.ps_user_sgpr.value, regs.ps_regs.rsrc2.user_sgpr), data);
}

ShaderParams PrepareProgram(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                            ShaderComputeInputInfo& info) {
	const auto [data, hash] = ShaderGetMappedData(regs.cs_regs.data_addr, "ShaderGetInputInfoCS():");
	ShaderGetStaticInputInfoCS(regs, sh, data, info);
	return GetShaderParams(
	    regs.cs_regs.data_addr, hash,
	    std::span<const uint32_t>(regs.cs_user_sgpr.value, regs.cs_regs.user_sgpr), data);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ShaderDbgDumpInputInfo(const ShaderVertexInputInfo& info) {
	KYTY_PROFILER_BLOCK("ShaderDbgDumpInputInfo(Vs)");

	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t fetch_external = %s\n"
	     "\t fetch_embedded = %s\n",
	     info.fetch_external ? "true" : "false", info.fetch_embedded ? "true" : "false");

	for (int i = 0; i < info.resources_num; i++) {
		LOGF("\t input %d\n", i);

		const auto& r  = info.resources[i];
		const auto& rd = info.resources_dst[i];

		LOGF("\t\t register_start   = %d\n"
		     "\t\t registers_num    = %d\n"
		     "\t\t fetch_index      = %" PRIu32 "\n"
		     "\t\t buffer_index     = %d\n",
		     rd.register_start, rd.registers_num, rd.fetch_index, rd.buffer_index);
		LOGF("\t\t fields           = %08" PRIx32 "%08" PRIx32 "%08" PRIx32 "%08" PRIx32 "\n",
		     r.fields[3], r.fields[2], r.fields[1], r.fields[0]);
		LOGF("\t\t Base()           = %" PRIx64 "\n"
		     "\t\t Stride()         = %" PRIu16 "\n"
		     "\t\t SwizzleEnabled() = %s\n"
		     "\t\t NumRecords()     = %" PRIu32 "\n"
		     "\t\t DstSelX()        = %" PRIu8 "\n"
		     "\t\t DstSelY()        = %" PRIu8 "\n"
		     "\t\t DstSelZ()        = %" PRIu8 "\n"
		     "\t\t DstSelW()        = %" PRIu8 "\n",
		     r.Base48(), r.Stride(), r.SwizzleEnabled() ? "true" : "false", r.NumRecords(),
		     r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW());
		LOGF("\t\t Format()         = %" PRIu8 "\n"
		     "\t\t OutOfBounds()    = %" PRIu8 "\n",
		     r.RawFormat(), r.OutOfBounds());
		LOGF("\t\t AddTid()         = %s\n", r.AddTid() ? "true" : "false");
	}

	for (int i = 0; i < info.buffers_num; i++) {
		LOGF("\t buffer %d\n", i);

		const auto& r = info.buffers[i];
		LOGF("\t\t addr        = %" PRIx64 "\n"
		     "\t\t stride      = %" PRIu32 "\n"
		     "\t\t num_records = %" PRIu32 "\n"
		     "\t\t fetch_index = %" PRIu32 "\n",
		     r.addr, r.stride, r.num_records, r.fetch_index);
	}
}

void ShaderDbgDumpInputInfo(const ShaderPixelInputInfo& info) {
	KYTY_PROFILER_BLOCK("ShaderDbgDumpInputInfo(Ps)");

	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t input_num            = %u\n"
	     "\t ps_system_input_base = %u\n"
	     "\t custom_interpolation_mask = 0x%08" PRIx32 "\n"
	     "\t ps_perspective_center_vgpr = %" PRIu32 "\n"
	     "\t ps_perspective_centroid_vgpr = %" PRIu32 "\n"
	     "\t ps_pos_x             = %s\n"
	     "\t ps_pos_y             = %s\n"
	     "\t ps_pos_z             = %s\n"
	     "\t ps_pos_w             = %s\n"
	     "\t ps_front_face        = %s\n"
	     "\t ps_ancillary         = %s\n"
	     "\t ps_sample_shading    = %s\n"
	     "\t ps_no_perspective    = %s\n"
	     "\t ps_pixel_kill_enable = %s\n"
	     "\t ps_early_z           = %s\n"
	     "\t ps_execute_on_noop   = %s\n",
	     info.input_num, info.ps_system_input_base, info.custom_interpolation_mask,
	     info.ps_perspective_center_vgpr, info.ps_perspective_centroid_vgpr,
	     info.ps_pos_x ? "true" : "false",
	     info.ps_pos_y ? "true" : "false", info.ps_pos_z ? "true" : "false",
	     info.ps_pos_w ? "true" : "false", info.ps_front_face ? "true" : "false",
	     info.ps_ancillary ? "true" : "false",
	     info.ps_sample_shading ? "true" : "false", info.ps_no_perspective ? "true" : "false",
	     info.ps_pixel_kill_enable ? "true" : "false", info.ps_early_z ? "true" : "false",
	     info.ps_execute_on_noop ? "true" : "false");

	for (uint32_t i = 0; i < info.input_num; i++) {
		LOGF("\t interpolator_settings[%u] = %u\n", i, info.interpolator_settings[i]);
	}
}

void ShaderDbgDumpInputInfo(const ShaderComputeInputInfo& info) {
	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t workgroup_register = %d\n"
	     "\t thread_ids_num     = %d\n"
	     "\t wave_size          = %u\n"
	     "\t lds_size_dwords    = %u\n"
	     "\t threads_num        = {%u, %u, %u}\n"
	     "\t tg_size_en         = %s\n",
	     info.workgroup_register, info.thread_ids_num, info.wave_size, info.lds_size_dwords,
	     info.threads_num[0], info.threads_num[1], info.threads_num[2],
	     info.tg_size_en ? "true" : "false");
	LOGF("\t threadgroup_id     = {%s, %s, %s}\n", info.group_id[0] ? "true" : "false",
	     info.group_id[1] ? "true" : "false", info.group_id[2] ? "true" : "false");
}

} // namespace Libs::Graphics
