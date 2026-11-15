#include "graphics/host_gpu/renderer/depthRenderTarget.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <limits>

namespace Libs::Graphics {

[[noreturn]] static void DepthFatal(const char* format, ...) {
	std::fputs("Depth target fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	EXIT("unsupported render state; details were printed above\n");
}

static vk::StencilOp ConvertStencilOp(uint8_t value, uint8_t write_mask, uint8_t op_value) {
	if (write_mask == 0) {
		return vk::StencilOp::eKeep;
	}
	switch (static_cast<Prospero::StencilOp>(value)) {
		case Prospero::StencilOp::kKeep: return vk::StencilOp::eKeep;
		case Prospero::StencilOp::kZero: return vk::StencilOp::eZero;
		case Prospero::StencilOp::kReplaceTest: return vk::StencilOp::eReplace;
		case Prospero::StencilOp::kReplaceOp:
			if ((op_value & write_mask) == 0) {
				return vk::StencilOp::eZero;
			}
			return vk::StencilOp::eReplace;
		case Prospero::StencilOp::kAddClamp: return vk::StencilOp::eIncrementAndClamp;
		case Prospero::StencilOp::kSubClamp: return vk::StencilOp::eDecrementAndClamp;
		case Prospero::StencilOp::kInvert: return vk::StencilOp::eInvert;
		case Prospero::StencilOp::kAddWrap: return vk::StencilOp::eIncrementAndWrap;
		case Prospero::StencilOp::kSubWrap: return vk::StencilOp::eDecrementAndWrap;
		case Prospero::StencilOp::kXor:
			if ((write_mask & op_value) == 0) {
				return vk::StencilOp::eKeep;
			}
			if ((write_mask & ~op_value) != 0) {
				DepthFatal("unsupported stencil XOR operands: write mask=0x%02" PRIx8
				           ", operation value=0x%02" PRIx8,
				           write_mask, op_value);
			}
			return vk::StencilOp::eInvert;
		default: DepthFatal("unsupported stencil operation: 0x%02" PRIx8, value);
	}
}

static vk::StencilOpState ConvertStencilState(
    uint8_t compare, const std::array<uint8_t, 3>& operations, uint8_t op_value,
    const vk::StencilOpState& state) {
	const auto test_value = state.reference;
	auto reference       = test_value;
	auto required_bits   = state.compareMask;
	if (compare == static_cast<uint8_t>(vk::CompareOp::eAlways) ||
	    compare == static_cast<uint8_t>(vk::CompareOp::eNever)) {
		required_bits = 0;
	}
	std::array<vk::StencilOp, 3> converted {};
	for (size_t i = 0; i < operations.size(); i++) {
		converted[i] = ConvertStencilOp(operations[i], state.writeMask, op_value);
		if (converted[i] != vk::StencilOp::eReplace) {
			continue;
		}
		auto replacement = test_value;
		if (static_cast<Prospero::StencilOp>(operations[i]) == Prospero::StencilOp::kReplaceOp) {
			replacement = op_value;
		}
		if (((reference ^ replacement) & required_bits & state.writeMask) != 0) {
			DepthFatal("unsupported stencil replacement: compare=%u, compare mask=0x%02" PRIx32
			           ", write mask=0x%02" PRIx32 ", operation value=0x%02" PRIx8
			           ", test value=0x%02" PRIx32,
			           compare, state.compareMask, state.writeMask, op_value, test_value);
		}
		// Vulkan shares one reference between comparison and every replacement on this face.
		reference = (reference & ~state.writeMask) | (replacement & state.writeMask);
		required_bits |= state.writeMask;
	}
	return {converted[0], converted[1], converted[2], static_cast<vk::CompareOp>(compare),
	        state.compareMask, state.writeMask, reference};
}

[[nodiscard]] static vk::Format ResolveHostDepthAttachmentFormat(GraphicContext&          graphics,
                                                                 const DepthFormatPolicy& policy,
                                                                 bool     has_stencil,
                                                                 uint32_t samples) {
	const auto required_samples = vulkan_sample_count(samples);
	const auto supports         = [&](vk::Format format) {
		vk::ImageFormatProperties properties {};
		return format != vk::Format::eUndefined &&
		       graphics.GetImageFormatProperties(
		           format, vk::ImageType::e2D, vk::ImageTiling::eOptimal, DepthTargetImageUsage(),
		           vk::ImageCreateFlags {}, &properties) == vk::Result::eSuccess &&
		       static_cast<bool>(properties.sampleCounts & required_samples);
	};
	if (!has_stencil) {
		return supports(policy.depth_attachment_format) ? policy.depth_attachment_format
		                                                : vk::Format::eUndefined;
	}
	for (const auto format: policy.stencil_attachment_formats) {
		if (supports(format)) {
			return format;
		}
	}
	return vk::Format::eUndefined;
}

static TextureCache::ImageDesc MakeDepthTargetDesc(const CommandBuffer& buffer,
                                                  const HW::DepthRenderTarget& z,
                                                  bool write_buffer = false) {
	const bool has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	const auto depth_address = write_buffer ? z.z_write_base_addr : z.z_read_base_addr;
	const auto stencil_address =
	    write_buffer ? z.stencil_write_base_addr : z.stencil_read_base_addr;
	if (!z.z_info.HasValidTextureCompatibility() ||
	    !z.stencil_info.HasValidTextureCompatibility()) {
		DepthFatal("invalid PS5 depth texture-compatibility encoding");
	}
	const bool has_htile = z.z_info.htile_acceleration;
	const bool unsupported_shading_rate_encoding =
	    z.shading_rate_encoding > 1 || (z.shading_rate_encoding != 0 && !has_htile);
	const auto samples   = render_sample_count(z.z_info.num_samples);
	if (samples == 0) {
		DepthFatal("unsupported depth fragment count: %u", z.z_info.num_samples);
	}
	const auto view = ResolveTargetViewInfo(z.depth_view.slice_start, z.depth_view.slice_max);
	switch (view.type) {
		case TargetViewType::Image2D:
		case TargetViewType::Image2DArray: break;
		case TargetViewType::Unsupported:
			DepthFatal("invalid depth view: base=%u last=%u", z.depth_view.slice_start,
			           z.depth_view.slice_max);
	}
	// EXPCLEAR permits an HTile acceleration state; the host attachment is already expanded.
	if (z.z_info.max_mip_level != 0 || unsupported_shading_rate_encoding || depth_address == 0 ||
	    (depth_address & 0xffffu) != 0) {
		DepthFatal("unsupported depth register state");
	}
	if (has_stencil) {
		if (z.stencil_info.format != Prospero::StencilFormat::k8UInt || stencil_address == 0 ||
		    (stencil_address & 0xffffu) != 0) {
			DepthFatal("unsupported stencil attachment state");
		}
	} else if (z.stencil_read_base_addr != 0 || z.stencil_write_base_addr != 0) {
		DepthFatal("stencil state without an active stencil attachment");
	}
	if (has_htile) {
		if (z.htile_data_base_addr == 0 || (z.htile_data_base_addr & 0x7fffu) != 0) {
			DepthFatal("invalid HTile metadata address");
		}
	}
	if (!z.size.valid) {
		DepthFatal("missing depth extent");
	}
	// With a single level, the selected mip is a surface of that mip's extent at the base address.
	const auto     mip    = z.depth_view.current_mip_level;
	const uint32_t width  = std::max((static_cast<uint32_t>(z.size.x_max) + 1u) >> mip, 1u);
	const uint32_t height = std::max((static_cast<uint32_t>(z.size.y_max) + 1u) >> mip, 1u);
	if (width > 16384 || height > 16384) {
		DepthFatal("invalid depth extent");
	}
	const auto* policy = FindDepthFormatPolicy(z.z_info.format);
	if (policy == nullptr) {
		DepthFatal("unsupported depth/stencil format pair");
	}
	const auto ideal_format = DepthAttachmentFormat(*policy, has_stencil);
	const auto format = ResolveHostDepthAttachmentFormat(buffer.GetGraphics(), *policy, has_stencil, samples);
	if (format == vk::Format::eUndefined) {
		DepthFatal("no host depth/stencil format supports required usage for %s",
		           vk::to_string(ideal_format).c_str());
	}
	const auto     guest_format = policy->guest_format;
	const uint32_t bytes        = policy->bytes_per_element;
	const auto     pitch        = TileGetRenderTargetPitch(width, bytes, z.z_info.num_samples);
	TileSizeAlign depth_size {};
	TileSizeAlign stencil_size {};
	TileSizeAlign htile_size {};
	if (!TileGetDepthSize(width, height, z.z_info.format, z.stencil_info.format, has_htile,
	                      stencil_size, htile_size, depth_size, z.z_info.num_samples) ||
	    depth_size.align != 65536 || depth_size.size == 0 ||
	    (has_stencil != (stencil_size.align == 65536 && stencil_size.size != 0)) ||
	    (has_htile != (htile_size.align == 32768 && htile_size.size != 0))) {
		DepthFatal("unsupported depth/stencil/HTile footprint");
	}
	if (depth_size.size > UINT64_MAX / view.image_layers ||
	    stencil_size.size > UINT64_MAX / view.image_layers ||
	    htile_size.size > UINT64_MAX / view.image_layers) {
		DepthFatal("layered depth footprint overflow");
	}
	const auto depth_backing_size   = depth_size.size * view.image_layers;
	const auto stencil_backing_size = stencil_size.size * view.image_layers;
	const auto htile_backing_size   = htile_size.size * view.image_layers;
	if (!GuestRange {depth_address, depth_backing_size}.Valid() ||
	    (has_stencil && !GuestRange {stencil_address, stencil_backing_size}.Valid()) ||
	    (has_htile && !GuestRange {z.htile_data_base_addr, htile_backing_size}.Valid())) {
		DepthFatal("layered depth backing range is invalid");
	}
	TextureCache::ImageDesc desc {};
	desc.type                 = TextureCache::BindingType::DepthTarget;
	desc.info.data            = {depth_address, depth_backing_size};
	desc.info.stencil =
	    has_stencil ? GuestRange {stencil_address, stencil_backing_size} : GuestRange {};
	desc.info.pixel_format    = format;
	desc.info.guest_format    = guest_format;
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {width, height, 1};
	desc.info.resources       = {1, view.image_layers};
	desc.info.pitch           = pitch;
	desc.info.bytes_per_block = bytes;
	desc.info.samples         = samples;
	desc.info.tile_mode       = Prospero::TileMode::kDepth;
	desc.info.UpdateSize();
	desc.info.metadata.range =
	    has_htile ? GuestRange {z.htile_data_base_addr, htile_backing_size} : GuestRange {};
	desc.info.metadata.kind   = has_htile ? ImageMetadataKind::Htile : ImageMetadataKind::None;
	desc.info.metadata.stencil_compressed =
	    has_stencil && has_htile && !z.stencil_info.htile_stencil_disabled;
	desc.view_info.format = format;
	desc.view_info.type =
	    view.layer_count == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	desc.view_info.aspect      = ImageViewOps::DepthAspectMask(format);
	desc.view_info.base_level  = 0;
	desc.view_info.level_count = 1;
	desc.view_info.base_layer  = view.base_layer;
	desc.view_info.layer_count = view.layer_count;
	desc.view_info.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
	return desc;
}

// Whether a draw with these registers has a depth or stencil attachment (ResolveRenderDepthTarget).
static bool DepthTargetBound(const HW::Context& hw) {
	const auto& z           = hw.GetDepthRenderTarget();
	const auto& rc          = hw.GetRenderControl();
	const auto& dc          = hw.GetDepthControl();
	const bool  has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	// With HTile, DEPTH_COMPRESS_DISABLE expands touched tiles even with depth/stencil tests off.
	// PPSA32236 fast-clears HTile each frame, runs this pass, then samples the depth as raw memory.
	// Unbound, the clear never lands: NaN depth makes one compute pass ~95 ms (60 -> 10 fps).
	const bool depth_active = dc.z_enable || dc.depth_bounds_enable || rc.depth_clear_enable ||
	                          rc.copy_depth_to_color ||
	                          (rc.depth_compress_disable && z.z_info.htile_acceleration);
	const bool stencil_active =
	    has_stencil && (dc.stencil_enable || rc.stencil_clear_enable || rc.copy_stencil_to_color);
	if (!depth_active && !stencil_active) {
		return false;
	}
	const bool attachment_unbound =
	    z.z_info.format == Prospero::DepthFormat::kInvalid &&
	    z.stencil_info.format == Prospero::StencilFormat::kInvalid && z.z_info.num_samples == 0 &&
	    z.z_info.texture_compatibility == Prospero::TextureCompatiblePlaneCompression::kDisable &&
	    !z.z_info.expclear_enabled && !z.z_info.partially_resident && z.z_info.max_mip_level == 0 &&
	    z.stencil_info.texture_compatibility == Prospero::TextureCompatibleStencil::kDisable &&
	    !z.stencil_info.expclear_enabled && !z.stencil_info.partially_resident &&
	    z.depth_view.slice_start == 0 && z.depth_view.slice_max == 0 &&
	    z.depth_view.current_mip_level == 0 && !z.depth_view.depth_write_disable &&
	    !z.depth_view.stencil_write_disable && z.z_read_base_addr == 0 && z.z_write_base_addr == 0 &&
	    z.stencil_read_base_addr == 0 && z.stencil_write_base_addr == 0 &&
	    z.htile_data_base_addr == 0 &&
	    // DB_DEPTH_SIZE_XY is independent state and may remain programmed after the attachment
	    // formats and addresses are unbound. A zero encoding is the valid 1x1 value, so its
	    // presence alone must not manufacture a depth attachment.
	    !z.z_info.htile_acceleration && z.shading_rate_encoding == 0 && z.size.x_max == 0 &&
	    z.size.y_max == 0;
	if (attachment_unbound) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("DepthTarget: ignoring enabled depth/stencil state without a bound attachment\n");
		}
		return false;
	}
	return true;
}

bool DrawDepthTargetFormat(GraphicContext& graphics, const HW::Context& hw, vk::Format& format, uint32_t& samples) {
	if (!DepthTargetBound(hw)) return false;
	const auto& z       = hw.GetDepthRenderTarget();
	const auto* policy  = FindDepthFormatPolicy(z.z_info.format);
	samples             = render_sample_count(z.z_info.num_samples);
	if (policy == nullptr || samples == 0) return false;
	format = ResolveHostDepthAttachmentFormat(
	    graphics, *policy, z.stencil_info.format != Prospero::StencilFormat::kInvalid, samples);
	return format != vk::Format::eUndefined;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
// The fields ResolveRenderDepthTarget sets (it leaves the others, and the stencil states while the stencil test is off).
static void AssignDepthTarget(RenderDepthInfo& r, const RenderDepthInfo& from) {
	r.desc                     = from.desc;
	r.depth_clear_enable       = from.depth_clear_enable;
	r.depth_clear_value        = from.depth_clear_value;
	r.depth_test_enable        = from.depth_test_enable;
	r.depth_write_enable       = from.depth_write_enable;
	r.depth_compare_op         = from.depth_compare_op;
	r.depth_bounds_test_enable = from.depth_bounds_test_enable;
	r.stencil_clear_enable     = from.stencil_clear_enable;
	r.stencil_clear_value      = from.stencil_clear_value;
	r.stencil_test_enable      = from.stencil_test_enable;
	if (from.stencil_test_enable) {
		r.stencil_front = from.stencil_front;
		r.stencil_back  = from.stencil_back;
	}
	r.image_id = from.image_id;
}

// Mode 2's check, field by field; the clear value by its bits.
static bool SameDepthTarget(const RenderDepthInfo& a, const RenderDepthInfo& b) {
	const auto& x = a.desc.info;
	const auto& y = b.desc.info;
	return a.image_id == b.image_id && a.desc.view_info == b.desc.view_info && a.desc.type == b.desc.type &&
	       x.data.address == y.data.address && x.data.size == y.data.size && x.stencil.address == y.stencil.address &&
	       x.stencil.size == y.stencil.size && x.pixel_format == y.pixel_format && x.extent.width == y.extent.width &&
	       x.extent.height == y.extent.height && x.resources.layers == y.resources.layers && x.pitch == y.pitch &&
	       x.samples == y.samples && x.metadata.kind == y.metadata.kind &&
	       x.metadata.range.address == y.metadata.range.address && x.metadata.range.size == y.metadata.range.size &&
	       x.metadata.stencil_compressed == y.metadata.stencil_compressed &&
	       a.depth_clear_enable == b.depth_clear_enable &&
	       std::bit_cast<uint32_t>(a.depth_clear_value) == std::bit_cast<uint32_t>(b.depth_clear_value) &&
	       a.depth_test_enable == b.depth_test_enable && a.depth_write_enable == b.depth_write_enable &&
	       a.depth_compare_op == b.depth_compare_op && a.depth_bounds_test_enable == b.depth_bounds_test_enable &&
	       a.stencil_clear_enable == b.stencil_clear_enable && a.stencil_clear_value == b.stencil_clear_value &&
	       a.stencil_test_enable == b.stencil_test_enable &&
	       (!a.stencil_test_enable || (a.stencil_front == b.stencil_front && a.stencil_back == b.stencil_back));
}

void RenderExecutor::ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& r) {
	KYTY_PROFILER_FUNCTION();
	r = {};

	const auto& hw          = buffer.GetRegisters();
	const auto& z           = hw.GetDepthRenderTarget();
	const auto& rc          = hw.GetRenderControl();
	const auto& dc          = hw.GetDepthControl();
	const auto& sc          = hw.GetStencilControl();
	const auto& sm          = hw.GetStencilMask();
	const bool  has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	const int   target_memo = TargetMemoMode();
	if (target_memo != 0 && (++m_target_calls & 4095u) == 0u) {
		const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		                                           std::chrono::steady_clock::now().time_since_epoch())
		                                           .count());
		if (m_target_stat_start == 0) {
			m_target_stat_start = now;
		} else if (now - m_target_stat_start >= 5000u) {
			LOGF("Target memo (5 s): %" PRIu64 " hits, %" PRIu64 " misses, %" PRIu64 " wrong\n", m_target_hits, m_target_misses,
			     m_target_wrong);
			m_target_hits = m_target_misses = m_target_wrong = 0;
			m_target_stat_start                              = now;
		}
	}
	if (!DepthTargetBound(hw)) {
		return;
	}
	// The memo, as for a color slot: copies of the registers read here, the clear value by its bits.
	static_assert(std::is_trivially_copyable_v<HW::DepthRenderTarget> && std::is_trivially_copyable_v<HW::RenderControl> &&
	              std::is_trivially_copyable_v<HW::DepthControl> && std::is_trivially_copyable_v<HW::StencilControl> &&
	              std::is_trivially_copyable_v<HW::StencilMask>);
	auto&          memo          = m_depth_memo;
	const uint32_t clear_bits    = std::bit_cast<uint32_t>(hw.GetDepthClearValue());
	const uint32_t stencil_clear = hw.GetStencilClearValue();
	bool           hit           = false;
	if (target_memo != 0 && memo.stamp != 0 && memo.clear_bits == clear_bits && memo.stencil_clear == stencil_clear &&
	    std::memcmp(&memo.z, &z, sizeof(z)) == 0 && std::memcmp(&memo.rc, &rc, sizeof(rc)) == 0 &&
	    std::memcmp(&memo.dc, &dc, sizeof(dc)) == 0 && std::memcmp(&memo.sc, &sc, sizeof(sc)) == 0 &&
	    std::memcmp(&memo.sm, &sm, sizeof(sm)) == 0 &&
	    m_context.GetTextureCache().RepeatLookup(memo.info.image_id, memo.stamp, memo.info.desc,
	                                             memo.info.desc.view_info.base_layer)) {
		hit = true;
		++m_target_hits;
		if (target_memo != 2) {
			AssignDepthTarget(r, memo.info);
			BindRenderTarget(r.image_id);
			return;
		}
	} else if (target_memo != 0) {
		++m_target_misses;
	}
	if (rc.copy_stencil_to_color ||
	    dc.zfunc > static_cast<uint8_t>(vk::CompareOp::eAlways) ||
	    (!z.depth_view.depth_write_disable && z.z_write_base_addr != z.z_read_base_addr) ||
	    (has_stencil && !z.depth_view.stencil_write_disable &&
	     z.stencil_write_base_addr != z.stencil_read_base_addr)) {
		DepthFatal("unsupported depth register state");
	}
	if (rc.copy_depth_to_color &&
	    (z.z_info.num_samples != 0 || rc.copy_sample != 0 ||
	     z.z_info.format != Prospero::DepthFormat::kZ32F ||
	     dc.z_enable || dc.z_write_enable || dc.depth_bounds_enable || dc.stencil_enable ||
	     rc.depth_clear_enable || rc.stencil_clear_enable)) {
		DepthFatal("unsupported depth-to-color copy state");
	}
	r.desc = MakeDepthTargetDesc(buffer, z);
	r.depth_clear_enable      = rc.depth_clear_enable;
	r.depth_clear_value       = hw.GetDepthClearValue();
	r.depth_test_enable       = dc.z_enable;
	r.depth_write_enable      = r.depth_test_enable && dc.z_write_enable &&
	                            !z.depth_view.depth_write_disable && !r.depth_clear_enable;
	r.depth_compare_op        = static_cast<vk::CompareOp>(dc.zfunc);

	r.depth_bounds_test_enable = dc.depth_bounds_enable;

	r.stencil_clear_enable =
	    has_stencil && rc.stencil_clear_enable && !z.depth_view.stencil_write_disable;
	r.stencil_clear_value = hw.GetStencilClearValue();
	r.stencil_test_enable = has_stencil && dc.stencil_enable;
	if (r.stencil_test_enable) {
		const bool stencil_ops_disabled =
		    rc.stencil_clear_enable || z.depth_view.stencil_write_disable;
		const uint8_t front_write_mask = stencil_ops_disabled ? 0 : sm.stencil_writemask;
		const uint8_t back_write_mask  = stencil_ops_disabled ? 0 : sm.stencil_writemask_bf;
		if (dc.stencilfunc > static_cast<uint8_t>(vk::CompareOp::eAlways) ||
		    (dc.backface_enable &&
		     dc.stencilfunc_bf > static_cast<uint8_t>(vk::CompareOp::eAlways))) {
			DepthFatal("unsupported stencil compare state");
		}
		r.stencil_front = ConvertStencilState(
		    dc.stencilfunc, {sc.stencil_fail, sc.stencil_zpass, sc.stencil_zfail},
		    sm.stencil_opval, {.compareMask = sm.stencil_mask, .writeMask = front_write_mask,
		                       .reference = sm.stencil_testval});
		if (dc.backface_enable) {
			r.stencil_back = ConvertStencilState(
			    dc.stencilfunc_bf, {sc.stencil_fail_bf, sc.stencil_zpass_bf, sc.stencil_zfail_bf},
			    sm.stencil_opval_bf, {.compareMask = sm.stencil_mask_bf, .writeMask = back_write_mask,
			                          .reference = sm.stencil_testval_bf});
		} else {
			r.stencil_back = r.stencil_front;
		}
	}
	auto&    cache = m_context.GetTextureCache();
	uint64_t plain = 0;
	r.image_id     = cache.FindImage(r.desc, false, true, target_memo != 0 ? &plain : nullptr);
	BindRenderTarget(r.image_id);
	if (target_memo != 0) {
		if (hit && !SameDepthTarget(memo.info, r)) {
			++m_target_wrong;
			LOGF("Target memo: a depth hit was wrong at 0x%010" PRIx64 "\n", z.z_read_base_addr);
		}
		memo.z             = z;
		memo.rc            = rc;
		memo.dc            = dc;
		memo.sc            = sc;
		memo.sm            = sm;
		memo.clear_bits    = clear_bits;
		memo.stencil_clear = stencil_clear;
		AssignDepthTarget(memo.info, r);
		memo.stamp = plain;
	}
}

bool RenderExecutor::DepthStencilCopy(CommandBuffer& buffer) {
	const auto& hw       = buffer.GetRegisters();
	const auto& z        = hw.GetDepthRenderTarget();
	const auto& override = hw.GetDepthRenderOverride();
	if (hw.GetColorControl().mode != 0) {
		return false;
	}
	const bool depth_copy = override.force_z_dirty && override.force_z_valid &&
	                        z.z_info.format != Prospero::DepthFormat::kInvalid &&
	                        z.z_read_base_addr != 0 && z.z_write_base_addr != 0 &&
	                        z.z_read_base_addr != z.z_write_base_addr;
	const bool stencil_copy = override.force_stencil_dirty && override.force_stencil_valid &&
	                          z.stencil_info.format != Prospero::StencilFormat::kInvalid &&
	                          z.stencil_read_base_addr != 0 && z.stencil_write_base_addr != 0 &&
	                          z.stencil_read_base_addr != z.stencil_write_base_addr;
	if (!depth_copy && !stencil_copy) {
		return false;
	}

	auto  read_desc  = MakeDepthTargetDesc(buffer, z);
	auto  write_desc = MakeDepthTargetDesc(buffer, z, true);
	auto& cache      = m_context.GetTextureCache();
	const auto read_id = cache.FindImage(read_desc);
	BindRenderTarget(read_id);
	const auto write_id = cache.FindImage(write_desc);
	BindRenderTarget(write_id);
	cache.UpdateImage(read_id);
	cache.UpdateImage(write_id);
	cache.MarkGpuWritten(write_id);
	auto& source      = cache.GetImage(read_id);
	auto& destination = cache.GetImage(write_id);
	EXIT_IF(read_id == write_id || source.backing.format != destination.backing.format);

	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Recorder();
	const ImageSubresourceRange range {read_desc.view_info.base_level, 1,
	                                  read_desc.view_info.base_layer,
	                                  read_desc.view_info.layer_count};
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	               range, command);
	destination.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	                    range, command);
	std::array<vk::ImageCopy, 2> regions {};
	uint32_t                   count = 0;
	for (const auto aspect: {vk::ImageAspectFlagBits::eDepth, vk::ImageAspectFlagBits::eStencil}) {
		if ((aspect == vk::ImageAspectFlagBits::eDepth && !depth_copy) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil && !stencil_copy)) {
			continue;
		}
		auto& region = regions[count++];
		region.srcSubresource = {aspect, range.base_level, range.base_layer,
		                         range.layer_count};
		region.dstSubresource = region.srcSubresource;
		region.extent = write_desc.info.extent;
	}
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
	                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
	                  count, regions.data());
	return true;
}

vk::ImageAspectFlags RenderDepthInfo::AttachmentWriteAspects() const {
	const auto format = desc.view_info.format;
	if (format == vk::Format::eUndefined) {
		return {};
	}

	const auto           available = ImageViewOps::DepthAspectMask(format);
	vk::ImageAspectFlags writes = load_clear_aspects & available;
	if ((available & vk::ImageAspectFlagBits::eDepth) &&
	    depth_write_enable) {
		writes |= vk::ImageAspectFlagBits::eDepth;
	}
	if (!(available & vk::ImageAspectFlagBits::eStencil)) {
		return writes;
	}

	const auto face_writes = [&](const vk::StencilOpState& state) {
		if (state.writeMask == 0) {
			return false;
		}
		bool can_pass = state.compareOp != vk::CompareOp::eNever;
		bool can_fail = state.compareOp != vk::CompareOp::eAlways;
		if (state.compareMask == 0) {
			switch (state.compareOp) {
				case vk::CompareOp::eEqual:
				case vk::CompareOp::eLessOrEqual:
				case vk::CompareOp::eGreaterOrEqual:
				case vk::CompareOp::eAlways:
					can_pass = true;
					can_fail = false;
					break;
				case vk::CompareOp::eNever:
				case vk::CompareOp::eLess:
				case vk::CompareOp::eGreater:
				case vk::CompareOp::eNotEqual:
					can_pass = false;
					can_fail = true;
					break;
				default: break;
			}
		}
		const bool depth_pass = !depth_test_enable || depth_compare_op != vk::CompareOp::eNever;
		const bool depth_fail = depth_test_enable && depth_compare_op != vk::CompareOp::eAlways;
		return (can_fail && state.failOp != vk::StencilOp::eKeep) ||
		       (can_pass && depth_pass && state.passOp != vk::StencilOp::eKeep) ||
		       (can_pass && depth_fail && state.depthFailOp != vk::StencilOp::eKeep);
	};
	if (stencil_test_enable && (face_writes(stencil_front) || face_writes(stencil_back))) {
		writes |= vk::ImageAspectFlagBits::eStencil;
	}
	return writes;
}

} // namespace Libs::Graphics
