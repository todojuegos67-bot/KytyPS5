#include "graphics/shader/shader.h"
#include "native-preparation-state.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/magicEnum.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderVertexMetadata.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
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

struct ShaderBinaryInfo {
	uint8_t  signature[7];
	uint8_t  version;
	uint32_t pssl_or_cg  : 1;
	uint32_t cached      : 1;
	uint32_t type        : 4;
	uint32_t source_type : 2;
	uint32_t length      : 24;
	uint8_t  chunk_usage_base_offset_dw;
	uint8_t  num_input_usage_slots;
	uint8_t  is_srt                 : 1;
	uint8_t  is_srt_used_info_valid : 1;
	uint8_t  is_extended_usage_info : 1;
	uint8_t  reserved2              : 5;
	uint8_t  reserved3;
	uint32_t hash0;
	uint32_t hash1;
	uint32_t crc32;
};

static std::unique_ptr<std::unordered_map<uint64_t, ShaderMappedData>> g_shader_map;
static std::mutex                                                      g_shader_map_mutex;
// Bumped on every registration: invalidates the per-thread last lookup below.
static std::atomic<uint64_t> g_shader_map_generation {0};

void ShaderInit() {
	EXIT_IF(g_shader_map != nullptr);

	g_shader_map = std::make_unique<std::unordered_map<uint64_t, ShaderMappedData>>();
}

void ShaderMapUserData(uint64_t addr, const ShaderMappedData& data) {
	EXIT_IF(g_shader_map == nullptr);

	std::scoped_lock lock(g_shader_map_mutex);

	(*g_shader_map)[addr] = data;
	g_shader_map_generation.fetch_add(1, std::memory_order_release);
}

uint64_t ShaderMapGeneration() {
	return g_shader_map_generation.load(std::memory_order_acquire);
}

bool ShaderLookupMappedData(uint64_t addr, ShaderMappedData* out) {
	EXIT_IF(g_shader_map == nullptr || out == nullptr);
	std::scoped_lock lock(g_shader_map_mutex);
	const auto iter = g_shader_map->find(addr);
	if (iter == g_shader_map->end()) return false;
	*out = iter->second;
	return true;
}

static ShaderMappedData ShaderGetMappedData(uint64_t addr, const char* label) {
	EXIT_IF(g_shader_map == nullptr);

	// The thread's earlier lookups by address, two per set, the latest first: a frame's draws and
	// dispatches use hundreds of shaders (the last four lookups held a fraction of them).
	struct Recent {
		uint64_t         addr = 0, generation = UINT64_MAX;
		ShaderMappedData data {};
	};
	thread_local std::array<std::array<Recent, 2>, 512> recent {};
	const auto generation = g_shader_map_generation.load(std::memory_order_acquire);
	auto&      set        = recent[((addr >> 8u) * 0x9e3779b97f4a7c15ull) >> 55u];
	if (set[0].addr == addr && set[0].generation == generation) return set[0].data;
	if (set[1].addr == addr && set[1].generation == generation) {
		std::swap(set[0], set[1]);
		return set[0].data;
	}

	std::scoped_lock lock(g_shader_map_mutex);

	if (auto iter = g_shader_map->find(addr); iter != g_shader_map->end()) {
		set[1] = set[0];
		set[0] = {addr, generation, iter->second};
		return iter->second;
	}

	EXIT("%s shader=0x%016" PRIx64 " is missing from ShaderMap\n", label, addr);
}

// Shader memory read as HashShaderCode reads the code: GPU ownership is byte-exact, CPU protection
// covers whole pages, so bytes that only share a page with GPU-written data come from the backing (a
// load would fault on the page and wait for its download). GPU-written bytes are loaded.
static void ReadShaderMemory(uint64_t address, void* out, size_t bytes) {
	if (!LibKernel::Memory::TryReadGpuCleanBackingOnWatchedPage(address, out, bytes))
		std::memcpy(out, reinterpret_cast<const void*>(address), bytes);
}

static uint64_t GetDeclaredShaderHash(uint64_t shader_addr) {
	EXIT_IF(shader_addr == 0);
	std::array<uint32_t, 2> code {};
	ReadShaderMemory(shader_addr, code.data(), sizeof(code));
	if (code[0] != 0xBEEB03FF) return 0;
	// The header's hash0 and hash1 (ShaderBinaryInfo).
	std::array<uint32_t, 2> hash {};
	ReadShaderMemory(shader_addr + static_cast<uint64_t>(code[1] + 1u) * 2u * sizeof(uint32_t) +
	                     offsetof(ShaderBinaryInfo, hash0),
	                 hash.data(), sizeof(hash));
	return (static_cast<uint64_t>(hash[1]) << 32u) | hash[0];
}

static uint64_t HashShaderCode(std::span<const uint32_t> code) {
	// GPU ownership is byte-exact, whereas CPU protection covers whole pages.
	// Hash current backing bytes when only unrelated data on the page is GPU dirty.
	// Hash all code bytes every time: a partial fingerprint can miss code patches.
	if (!code.empty() && code.size_bytes() <= 1024 * 1024) {
		thread_local std::vector<uint32_t> scratch;
		if (scratch.size() < code.size()) scratch.resize(code.size());
		if (LibKernel::Memory::TryReadGpuCleanBackingOnWatchedPage(
		        reinterpret_cast<uint64_t>(code.data()), scratch.data(), code.size_bytes())) {
			return XXH3_64bits(scratch.data(), code.size_bytes());
		}
	}
	return XXH3_64bits(code.data(), code.size_bytes());
}

// The hash of a registered shader (its declared hash, else HashShaderCode), computed once per registration
// (any registration moves the generation) and again at every 64th use, which also catches code rewritten in
// place: hashing ~2 KiB of code at every draw and dispatch was ~1.4% of the render thread.
static uint64_t ShaderCodeHash(uint64_t shader_addr, std::span<const uint32_t> code) {
	struct Entry {
		uint64_t addr = 0, generation = UINT64_MAX, hash = 0;
		uint32_t uses = 0;
	};
	// Direct-mapped: ~800 shaders a frame at Boletaria (460 compute) collided often in 1024 entries. On the
	// heap: only the threads that hash need one.
	thread_local auto entries = std::make_unique<std::array<Entry, 8192>>();
	const auto generation = g_shader_map_generation.load(std::memory_order_acquire);
	auto&      entry      = (*entries)[((shader_addr >> 8u) * 0x9e3779b97f4a7c15ull) >> 51u];
	const bool known      = entry.addr == shader_addr && entry.generation == generation;
	if (known && (++entry.uses & 63u) != 0) return entry.hash;
	const auto declared = GetDeclaredShaderHash(shader_addr);
	const auto hash     = declared != 0 ? declared : HashShaderCode(code);
	if (known && hash != entry.hash) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16)
			std::printf("Shader code changed in place: shader=0x%016" PRIx64 " hash 0x%016" PRIx64 " -> 0x%016" PRIx64
			            "\n",
			            shader_addr, entry.hash, hash);
	}
	entry = {shader_addr, generation, hash, known ? entry.uses : 0u};
	return hash;
}

// Fills `params` in place: its user-data vector keeps its capacity across calls.
static void FillShaderParams(uint64_t shader_addr, const char* label, std::span<const uint32_t> user_data,
                             const ShaderMappedData& data, ShaderParams& params) {
	if (data.code_size_bytes == 0 || data.code_size_bytes % sizeof(uint32_t) != 0) {
		EXIT("%s hash=0x%016" PRIx64 " shader=0x%016" PRIx64
		     " has invalid AGC shader_size=0x%08" PRIx32 "\n",
		     label, GetDeclaredShaderHash(shader_addr), shader_addr, data.code_size_bytes);
	}
	const auto code_words = data.code_size_bytes / sizeof(uint32_t);
	const auto code = std::span {reinterpret_cast<const uint32_t*>(shader_addr), code_words};
	params.code = code;
	params.user_data.assign(user_data.begin(), user_data.end());
	params.hash      = ShaderCodeHash(shader_addr, code);
	params.back_code = {};
}

uint64_t ShaderHashAt(uint64_t addr) {
	const auto data = ShaderGetMappedData(addr, "ShaderHashAt");
	if (data.code_size_bytes == 0 || data.code_size_bytes % sizeof(uint32_t) != 0) return 0;
	return ShaderCodeHash(addr, std::span {reinterpret_cast<const uint32_t*>(addr), data.code_size_bytes / sizeof(uint32_t)});
}

static ShaderParams GetShaderParams(uint64_t shader_addr, const char* label, std::span<const uint32_t> user_data,
                                    const ShaderMappedData& data) {
	ShaderParams params;
	FillShaderParams(shader_addr, label, user_data, data, params);
	return params;
}

static void ShaderDetectBuffers(ShaderVertexInputInfo& info) {
	KYTY_PROFILER_FUNCTION();

	info.buffers_num = 0;

	for (int ri = 0; ri < info.resources_num; ri++) {
		const auto& r = info.resources[ri];

		bool merged = false;
		for (int bi = 0; bi < info.buffers_num; bi++) {
			auto& b = info.buffers[bi];

			uint64_t stride = b.stride;

			if (stride == r.Stride() &&
			    b.fetch_index == static_cast<uint32_t>(info.resources_dst[ri].fetch_index)) {
				uint64_t rbase   = r.Base48();
				uint64_t base    = std::min(rbase, b.addr);
				uint64_t offset1 = rbase - base;
				uint64_t offset2 = b.addr - base;

				if (offset1 < stride && offset2 < stride) {
					EXIT_NOT_IMPLEMENTED(b.num_records != r.NumRecords());
					b.addr = base;
					EXIT_NOT_IMPLEMENTED(b.attr_num >= ShaderVertexInputBuffer::ATTR_MAX);
					b.attr_indices[b.attr_num++] = ri;
					merged                       = true;
					break;
				}
			}
		}

		if (!merged) {
			EXIT_NOT_IMPLEMENTED(info.buffers_num >= ShaderVertexInputInfo::RES_MAX);
			int bi                           = info.buffers_num++;
			info.buffers[bi].addr            = r.Base48();
			info.buffers[bi].stride          = r.Stride();
			info.buffers[bi].num_records     = r.NumRecords();
			info.buffers[bi].fetch_index     = info.resources_dst[ri].fetch_index;
			info.buffers[bi].attr_num        = 1;
			info.buffers[bi].attr_indices[0] = ri;
		}
	}

	for (int bi = 0; bi < info.buffers_num; bi++) {
		auto& b = info.buffers[bi];
		for (int ri = 0; ri < b.attr_num; ri++) {
			b.attr_offsets[ri] = info.resources[b.attr_indices[ri]].Base48() - b.addr;
		}
	}
}

static void ShaderApplyAttribSemantics(ShaderVertexInputInfo& info,
                                       const ShaderSemantic*  input_semantics,
                                       uint32_t num_input_semantics, const uint32_t* attrib,
                                       const uint32_t* buffer) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(attrib == nullptr || buffer == nullptr);

	for (uint32_t i = 0; i < num_input_semantics; i++) {
		const auto& in = input_semantics[i];

		EXIT_NOT_IMPLEMENTED(in.static_vb_index == 1 || in.static_attribute == 1);

		uint32_t reg  = in.hardware_mapping;
		uint32_t size = in.size_in_elements;

		if (Config::GraphicsDebugDumpEnabled()) {
			LOGF("reg = %u, size = %u, va[%u] = 0x%08" PRIx32 "\n", reg, size, i,
			     attrib[in.semantic]);
		}

		size_t index = attrib[in.semantic] & 0x1fu;
		auto   format =
		    static_cast<Prospero::VertexAttribFormat>((attrib[in.semantic] >> 5u) & 0x1ffu);
		uint32_t offset      = (attrib[in.semantic] >> 14u) & 0xfffu;
		uint32_t fetch_index = (attrib[in.semantic] >> 26u) & 0x1u;

		if (fetch_index != 0) {
			static std::atomic<uint64_t> log_count = 0;
			const bool log_this = log_count.load(std::memory_order_relaxed) < 64 && log_count.fetch_add(1) < 64;
			if (log_this) {
				LOGF("\t temporary: PS5 vertex attrib semantic %u uses fetch index %u, buffer "
				     "index %zu\n",
				     static_cast<uint32_t>(in.semantic), fetch_index, index);
			}
		}

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
		if (format != Prospero::VertexAttribFormat::kInvalid) {
			const auto                   format_raw    = static_cast<uint32_t>(format);
			const auto                   buffer_format = format_raw >> 2u;
			const auto                   channels      = (format_raw & 3u) + 1u;
			static std::atomic<uint64_t> log_count      = 0;
			const bool log_this = log_count.load(std::memory_order_relaxed) < 64 && log_count.fetch_add(1) < 64;
			if (log_this) {
				LOGF("\t PS5 vertex attrib semantic %u uses attrib format %u -> buffer "
				     "format %u, offset %u, buffer index %zu\n",
				     static_cast<uint32_t>(in.semantic), static_cast<uint32_t>(format),
				     static_cast<uint32_t>(buffer_format), offset, index);
			}
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
	EXIT_NOT_IMPLEMENTED(regs.ps_input_ena != regs.ps_input_addr);

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

static bool ShaderGetStaticInputInfoVS(const HW::VertexShaderInfo& regs,
	                                   const HW::ShaderRegisters& sh,
	                                   const ShaderMappedData& data,
	                                   ShaderVertexInputInfo& info) {
	KYTY_PROFILER_FUNCTION();

	ResetNativeVertexInput(info);

	info.pa_cl_vs_out_cntl = sh.m_paClVsOutCntl;

	EXIT_NOT_IMPLEMENTED(regs.es_regs.data_addr == 0);

	uint64_t                shader_addr   = regs.es_regs.data_addr;
	const HW::UserSgprInfo& user_sgpr     = regs.gs_user_sgpr;
	auto                    user_sgpr_num = regs.gs_regs.rsrc2.user_sgpr;
	info.scratch_size_dwords = data.scratch_size_dwords;

	if (data.user_data == nullptr) {
		LOGF("ShaderGetInputInfoVS(): no AGC user data for shader=0x%016" PRIx64 " es=0x%016" PRIx64
		     " gs=0x%016" PRIx64 " user_sgpr_num=%u\n",
		     shader_addr, regs.es_regs.data_addr, regs.gs_regs.data_addr,
		     static_cast<uint32_t>(user_sgpr_num));
	}
	ShaderVertexMetadata metadata;
	std::string          metadata_error;
	if (!ShaderReadVertexMetadata(data, HW::UserSgprInfo::SGPRS_MAX, metadata, &metadata_error)) {
		LOGF("ShaderGetInputInfoVS(): invalid AGC metadata shader=0x%016" PRIx64 ": %s\n",
		     shader_addr, metadata_error.c_str());
		return false;
	}

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
			LOGF("ShaderGetInputInfoVS(): null vertex table pointer shader=0x%016" PRIx64 "\n",
			     shader_addr);
			return false;
		}
		ShaderApplyAttribSemantics(info, metadata.input_semantics.data(),
		                           metadata.input_semantics_count, attrib, buffer);
		ShaderDetectBuffers(info);
	}
	return true;
}

static void ShaderGetStaticInputInfoPS(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	const ShaderMappedData& data, ShaderPixelInputInfo& ps_info) {
	KYTY_PROFILER_FUNCTION();

	ResetNativeStageInput(ps_info);
	ps_info.scratch_size_dwords = data.scratch_size_dwords;

	// SPI_PS_IN_CONTROL.NUM_INTERP occupies bits 5:0. Keep the remaining control
	// flags in the hardware state and extract only the input count here.
	ps_info.input_num            = sh.ps_in_control & 0x3fu;
	EXIT_NOT_IMPLEMENTED(ps_info.input_num > std::size(ps_info.interpolator_settings));
	ps_info.ps_system_input_base = ShaderCalcPsSystemInputBase(sh);
	const uint32_t active_inputs = sh.ps_input_ena & sh.ps_input_addr;
	if ((active_inputs & 0x00000002u) != 0) {
		ps_info.ps_perspective_center_vgpr = (active_inputs & 0x00000001u) != 0 ? 2u : 0u;
	}
	for (uint32_t i = 0; i < data.num_input_semantics && i < ps_info.input_num && i < 32u; i++) {
		const auto& semantic = data.input_semantics[i];
		if (semantic.is_custom != 0 && semantic.is_f16 == 0) {
			ps_info.custom_interpolation_mask |= 1u << i;
		}
	}
	ps_info.ps_pos_x                     = (active_inputs & 0x00000100u) != 0;
	ps_info.ps_pos_y                     = (active_inputs & 0x00000200u) != 0;
	ps_info.ps_pos_z                     = (active_inputs & 0x00000400u) != 0;
	ps_info.ps_pos_w                     = (active_inputs & 0x00000800u) != 0;
	ps_info.ps_front_face                = (active_inputs & 0x00001000u) != 0;
	ps_info.ps_ancillary                 = (active_inputs & 0x00002000u) != 0;
	ps_info.ps_sample_shading            = (active_inputs & 0x00000011u) != 0;
	ps_info.ps_no_perspective            = (sh.ps_input_ena & sh.ps_input_addr & 0x00000020u) != 0;
	ps_info.ps_pixel_kill_enable         = sh.db_shader_control.shader_kill_enable;
	ps_info.ps_depth_export_enable       = sh.db_shader_control.shader_z_export_enable;
	ps_info.ps_sample_mask_export_enable = sh.db_shader_control.shader_mask_export_enable;
	ps_info.ps_early_z =
	    (sh.db_shader_control.shader_z_behavior == 1 && !sh.db_shader_control.shader_kill_enable &&
	     !sh.db_shader_control.shader_z_export_enable &&
	     !sh.db_shader_control.shader_mask_export_enable);
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
	const bool dispatch_thread_dimensions = info.dispatch_thread_dimensions;
	const auto host_subgroup_size         = info.host_subgroup_size;
	ResetNativeStageInput(info);
	info.dispatch_thread_dimensions       = dispatch_thread_dimensions;
	info.host_subgroup_size               = host_subgroup_size;
	info.threads_num[0]                   = regs.cs_regs.num_thread_x;
	info.threads_num[1]                   = regs.cs_regs.num_thread_y;
	info.threads_num[2]                   = regs.cs_regs.num_thread_z;
	info.lds_size_dwords                  = static_cast<uint32_t>(regs.cs_regs.lds_size) * 128u;
	info.scratch_size_dwords              = data.scratch_size_dwords;
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
	key.push_back(info.scratch_size_dwords);
	key.push_back(info.pa_cl_vs_out_cntl);
	key.push_back(static_cast<uint32_t>(info.start_instance_sgpr));
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
		                       mesh.max_vertices, mesh.max_primitives, mesh.provoking_vertex});
	}

	for (int i = 0; i < info.resources_num; i++) {
		const auto& resource    = info.resources[i];
		const auto& destination = info.resources_dst[i];
		key.push_back(destination.register_start);
		key.push_back(destination.registers_num);
		key.push_back(destination.fetch_index);
		key.push_back(static_cast<uint32_t>(destination.attr_id));
		key.push_back(resource.Stride());
		key.push_back(static_cast<uint32_t>(resource.SwizzleEnabled()));
		key.push_back(resource.DstSelX());
		key.push_back(resource.DstSelY());
		key.push_back(resource.DstSelZ());
		key.push_back(resource.DstSelW());
		key.push_back(resource.RawFormat());
		key.push_back(resource.OutOfBounds());
		key.push_back(static_cast<uint32_t>(resource.AddTid()));
	}
}

void BuildStageStaticKey(const ShaderPixelInputInfo& info, std::vector<uint32_t>& key) {
	EXIT_IF(info.input_num > std::size(info.interpolator_settings));
	key.clear();
	key.push_back(info.scratch_size_dwords);
	key.push_back(info.input_num);
	key.push_back(info.ps_system_input_base);
	key.push_back(info.custom_interpolation_mask);
	key.push_back(info.ps_perspective_center_vgpr);
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
	key.insert(key.end(), std::begin(info.target_output_mode), std::end(info.target_output_mode));
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
	key.push_back(info.wave_size);
	key.push_back(info.host_subgroup_size);
	key.push_back(info.thread_ids_num);
	key.push_back(info.lds_size_dwords);
	key.push_back(info.scratch_size_dwords);
	key.push_back(static_cast<uint32_t>(info.dispatch_thread_dimensions));
	for (int i = 0; i < 3; i++) {
		key.push_back(info.threads_num[i]);
		key.push_back(static_cast<uint32_t>(info.group_id[i]));
	}
	key.push_back(static_cast<uint32_t>(info.tg_size_en));
}

ShaderParams PrepareProgram(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            const HW::UserConfig& user_config, ShaderVertexInputInfo& info) {
	ShaderParams params;
	PrepareProgramInto(regs, context, user_config, info, params);
	return params;
}

void PrepareProgramInto(const HW::VertexShaderInfo& regs, const HW::Context& context,
                        const HW::UserConfig& user_config, ShaderVertexInputInfo& info,
                        ShaderParams& params) {
	const auto& sh   = context.GetShaderRegisters();
	const auto  data = ShaderGetMappedData(regs.es_regs.data_addr, "ShaderGetInputInfoVS():");
	FillShaderParams(regs.es_regs.data_addr, "ShaderRecompiler VS",
	                 std::span<const uint32_t>(regs.gs_user_sgpr.value, regs.gs_regs.rsrc2.user_sgpr), data,
	                 params);
	// The GS user SGPRs start at s8: one the shader receives may hold the draw's start instance.
	const auto start_instance      = regs.start_instance_user_sgpr;
	const auto start_instance_sgpr =
	    start_instance >= 0 && static_cast<uint32_t>(start_instance) < regs.gs_regs.rsrc2.user_sgpr
	        ? 8 + start_instance : -1;
	if ((context.GetShaderStages() & 0x20u) == 0) {
		if (!ShaderGetStaticInputInfoVS(regs, sh, data, info)) {
			EXIT("failed to prepare vertex shader program\n");
		}
		info.start_instance_sgpr = start_instance_sgpr;
		return;
	}
	// NGG user SGPRs start at s8; a separately compiled GS back half also receives
	// its user-data pointer in s0:s1.
	params.user_data.insert(params.user_data.begin(), 8u, 0u);
	ResetNativeVertexInput(info);
	info.start_instance_sgpr = start_instance_sgpr;
	info.pa_cl_vs_out_cntl   = sh.m_paClVsOutCntl;
	auto& mesh               = info.mesh;
	mesh.input_primitive     = static_cast<uint32_t>(user_config.GetPrimType());
	mesh.wave_size           = (context.GetShaderStages() & 0x00400000u) != 0 ? 32u : 64u;
	mesh.max_vertices        = sh.m_geMaxOutputPerSubgroup;
	mesh.provoking_vertex    = context.GetModeControl().provoking_vtx_last ? 2u : 0u;
	mesh.lds_size_dwords     = static_cast<uint32_t>(regs.gs_regs.rsrc2.lds_size) * 128u;
	mesh.scratch_size_dwords = data.scratch_size_dwords;
	if (data.type == Prospero::ShaderBinaryType::kGsFront) {
		EXIT_IF(regs.gs_regs.data_addr == 0);
		const auto back = ShaderGetMappedData(regs.gs_regs.data_addr, "ShaderGetInputInfoGS():");
		const auto back_params =
		    GetShaderParams(regs.gs_regs.data_addr, "ShaderRecompiler GS", {}, back);
		params.back_code = back_params.code;
		params.user_data[0] = static_cast<uint32_t>(regs.gs_regs.user_data_addr);
		params.user_data[1] = static_cast<uint32_t>(regs.gs_regs.user_data_addr >> 32u);
		const uint64_t hashes[] = {params.hash, back_params.hash};
		params.hash = XXH3_64bits(hashes, sizeof(hashes));
		mesh.scratch_size_dwords = std::max(mesh.scratch_size_dwords, back.scratch_size_dwords);
	}
	EXIT_NOT_IMPLEMENTED(regs.gs_regs.rsrc1.gs_vgpr_component_count != 3u ||
	                     regs.gs_regs.rsrc2.es_vgpr_component_count != 3u);
	const auto& group = user_config.GetGeControl();
	if ((user_config.GetPrimType() != Prospero::PrimitiveType::kPointList &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kLineList &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriStrip &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList) ||
	    sh.m_vgtGsOutPrimType != 2u || sh.m_vgtGsMaxVertOut < 3u ||
	    group.vertex_group_size < mesh.InputPrimitiveSize() ||
	    mesh.max_vertices == 0u) {
		EXIT("unsupported GS assembly: input=%u output=%u vertices=%u GE=%u/%u max_output=%u\n",
		     mesh.input_primitive, sh.m_vgtGsOutPrimType, sh.m_vgtGsMaxVertOut,
		     group.primitive_group_size, group.vertex_group_size, mesh.max_vertices);
	}
	mesh.max_primitives       = group.primitive_group_size * (sh.m_vgtGsMaxVertOut - 2u);
	mesh.primitives_per_group = std::min({static_cast<uint32_t>(group.primitive_group_size),
	                                      mesh.InputPrimitiveCount(group.vertex_group_size),
	                                      mesh.max_vertices / sh.m_vgtGsMaxVertOut});
	EXIT_IF(mesh.primitives_per_group == 0u);
	mesh.vertices_per_group = mesh.InputVertexCount(mesh.primitives_per_group);
	mesh.threads_num[0] =
	    ((mesh.max_vertices + mesh.wave_size - 1u) / mesh.wave_size) * mesh.wave_size;
	mesh.threads_num[1] = mesh.threads_num[2] = 1u;
}

ShaderParams PrepareProgram(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    ShaderPixelInputInfo&                               ps_info) {
	ShaderParams params;
	PrepareProgramInto(regs, sh, target_export_mapping, ps_info, params);
	return params;
}

void PrepareProgramInto(const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
                        std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
                        ShaderPixelInputInfo& ps_info, ShaderParams& params) {
	const auto data = ShaderGetMappedData(regs.ps_regs.data_addr, "ShaderGetInputInfoPS():");
	ShaderGetStaticInputInfoPS(regs, sh, target_export_mapping, data, ps_info);
	FillShaderParams(regs.ps_regs.data_addr, "ShaderRecompiler PS",
	                 std::span<const uint32_t>(regs.ps_user_sgpr.value, regs.ps_regs.rsrc2.user_sgpr), data,
	                 params);
}

ShaderParams PrepareProgram(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                            ShaderComputeInputInfo& info) {
	ShaderParams params;
	PrepareProgramInto(regs, sh, info, params);
	return params;
}

void PrepareProgramInto(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                        ShaderComputeInputInfo& info, ShaderParams& params) {
	const auto data = ShaderGetMappedData(regs.cs_regs.data_addr, "ShaderGetInputInfoCS():");
	ShaderGetStaticInputInfoCS(regs, sh, data, info);
	FillShaderParams(regs.cs_regs.data_addr, "ShaderRecompiler CS",
	                 std::span<const uint32_t>(regs.cs_user_sgpr.value, regs.cs_regs.user_sgpr), data,
	                 params);
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
		     "\t\t fetch_index      = %" PRIu32 "\n",
		     rd.register_start, rd.registers_num, rd.fetch_index);
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
		     "\t\t fetch_index = %" PRIu32 "\n"
		     "\t\t attr_num    = %" PRId32 "\n",
		     r.addr, r.stride, r.num_records, r.fetch_index, r.attr_num);
		for (int j = 0; j < r.attr_num; j++) {
			LOGF("\t\t attr_indices[%d]  = %d\n"
			     "\t\t attr_offsets[%d]  = %u\n",
			     j, r.attr_indices[j], j, r.attr_offsets[j]);
		}
	}
}

void ShaderDbgDumpInputInfo(const ShaderPixelInputInfo& info) {
	KYTY_PROFILER_BLOCK("ShaderDbgDumpInputInfo(Ps)");

	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t input_num            = %u\n"
	     "\t ps_system_input_base = %u\n"
	     "\t custom_interpolation_mask = 0x%08" PRIx32 "\n"
	     "\t ps_perspective_center_vgpr = %" PRIu32 "\n"
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
	     info.ps_perspective_center_vgpr, info.ps_pos_x ? "true" : "false",
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

bool ShaderAddressValid(uint64_t addr) {
	return reinterpret_cast<const uint32_t*>(addr) != nullptr;
}

} // namespace Libs::Graphics
