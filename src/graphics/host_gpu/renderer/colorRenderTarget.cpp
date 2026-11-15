#include "graphics/host_gpu/renderer/colorRenderTarget.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace Libs::Graphics {

static bool DccAlphaOnMsb(const HW::ColorInfo& info) {
	switch (info.format) {
		case Prospero::ChannelLayout::k10_10_10_2:
		case Prospero::ChannelLayout::k10_10_10_2Float:
		case Prospero::ChannelLayout::k5_5_5_1: return true;
		case Prospero::ChannelLayout::k2_10_10_10:
		case Prospero::ChannelLayout::k1_5_5_5: return false;
		default: break;
	}
	const auto components =
	    Prospero::ResolveRenderTargetFormat(info.format, info.channel_type).components;
	if (components == 1) {
		return info.channel_order != Prospero::ChannelOrder::kStandard;
	}
	return components == 3 || info.channel_order == Prospero::ChannelOrder::kStandard ||
	       info.channel_order == Prospero::ChannelOrder::kAlt;
}

int RenderExecutor::TargetMemoMode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_TARGET_MEMO");
		return v == nullptr ? 1 : std::atoi(v);
	}();
	return mode;
}

// What a full resolve gives, field by field (mode 2's check).
static bool SameColorTarget(const RenderColorInfo& a, const RenderColorInfo& b) {
	const auto& x = a.desc.info;
	const auto& y = b.desc.info;
	return a.image_id == b.image_id && a.target_slot == b.target_slot && a.guest_mip_level == b.guest_mip_level &&
	       a.guest_array_layer == b.guest_array_layer &&
	       std::memcmp(&a.export_mapping, &b.export_mapping, sizeof(a.export_mapping)) == 0 &&
	       a.desc.view_info == b.desc.view_info && a.desc.type == b.desc.type && x.data.address == y.data.address &&
	       x.data.size == y.data.size && x.pixel_format == y.pixel_format && x.guest_format == y.guest_format &&
	       x.type == y.type && x.extent.width == y.extent.width && x.extent.height == y.extent.height &&
	       x.extent.depth == y.extent.depth && x.resources.levels == y.resources.levels &&
	       x.resources.layers == y.resources.layers && x.bytes_per_block == y.bytes_per_block && x.samples == y.samples &&
	       x.tile_mode == y.tile_mode && x.metadata.kind == y.metadata.kind &&
	       x.metadata.range.address == y.metadata.range.address && x.metadata.range.size == y.metadata.range.size &&
	       x.metadata.clear_word == y.metadata.clear_word;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& r,
                                              uint32_t         render_target_slice_offset,
                                              uint32_t rt_slot, bool ignore_target_mask,
                                              bool exact_format) {
	KYTY_PROFILER_FUNCTION();
	const auto& hw = buffer.GetRegisters();

	const auto& rt      = hw.GetRenderTarget(rt_slot);
	auto        mask    = render_target_mask_slot(hw.GetRenderTargetMask(), rt_slot);
	if (ignore_target_mask && rt.base.addr != 0 && mask == 0) {
		mask = 0x0f;
	}

	r             = {};
	r.target_slot = rt_slot;

	if (rt.base.addr == 0 || mask == 0) {
		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("RenderColorTarget: no color output slot=%" PRIu32 " base=0x%010" PRIx64
				     " slot_mask=0x%01" PRIx32 " target_mask=0x%08" PRIx32
				     " rt_slice_offset=%" PRIu32 "\n",
				     rt_slot, rt.base.addr, mask, hw.GetRenderTargetMask(),
				     render_target_slice_offset);
			}
		}

		return;
	}
	// The memo: a copy of the slot's registers, compared bytewise (unequal padding can only cost a hit). On a hit the
	// lookup's touch and any pending color clear repeat (RepeatLookup) and the target is bound as a resolve binds it.
	static_assert(std::is_trivially_copyable_v<HW::RenderTarget>);
	const int        target_memo = TargetMemoMode();
	ColorTargetMemo* memo        = target_memo != 0 && !exact_format && rt_slot < m_color_memo.size() &&
	                                     !graphics_debug_dump_enabled()
	                                   ? &m_color_memo[rt_slot]
	                                   : nullptr;
	bool             hit         = false;
	if (memo != nullptr && memo->stamp != 0 && memo->mask == mask && memo->slice_offset == render_target_slice_offset &&
	    std::memcmp(&memo->rt, &rt, sizeof(rt)) == 0 &&
	    m_context.GetTextureCache().RepeatLookup(memo->info.image_id, memo->stamp, memo->info.desc,
	                                             memo->info.desc.view_info.base_layer)) {
		hit = true;
		++m_target_hits;
		if (target_memo != 2) {
			r = memo->info;
			BindRenderTarget(r.image_id);
			return;
		}
	} else if (memo != nullptr) {
		++m_target_misses;
	}
	const auto samples = render_sample_count(rt.attrib.num_fragments);
	// TODO(PPSA24156): Replace this 2x MSAA approximation with proper 4-coverage/2-fragment EQAA.
	const bool approximate_eqaa = rt.attrib.num_samples == 2 && rt.attrib.num_fragments == 1;
	if (approximate_eqaa) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(
			    "Warning: approximating 4-coverage/2-fragment EQAA with 2x MSAA.\n");
		}
	}
	if (samples == 0 || (rt.attrib.num_samples != rt.attrib.num_fragments && !approximate_eqaa)) {
		EXIT("unsupported render-target sample configuration: samples=%u fragments=%u\n",
		     rt.attrib.num_samples, rt.attrib.num_fragments);
	}
	const uint32_t levels = rt.attrib2.num_mip_levels + 1u;
	if (rt.attrib3.dimension > 2u) {
		EXIT("unsupported render-target dimension: %u\n", rt.attrib3.dimension);
	}
	// 1D targets are 2D images one texel high.
	const bool is_1d  = rt.attrib3.dimension == 0u;
	const bool volume = rt.attrib3.dimension == 2u;
	if (is_1d && rt.attrib2.height != 0) {
		EXIT("1D render target has nonzero height: %u\n", rt.attrib2.height);
	}
	if (!volume && rt.attrib3.depth != 0) {
		EXIT("non-3D render target has nonzero depth: %u\n", rt.attrib3.depth);
	}
	if (is_1d && samples != 1) {
		EXIT("multisampled 1D render targets are unsupported\n");
	}
	if (volume && samples != 1) {
		EXIT("multisampled 3D render targets are unsupported\n");
	}
	const uint32_t depth = volume ? rt.attrib3.depth + 1u : 1u;
	// For volumes, CB_COLOR_VIEW bounds exported slices; ATTRIB3 defines storage depth.
	// The host attachment contains only the selected slices that exist in this mip.
	const uint32_t last_layer = volume
	                                ? std::min(rt.view.last_array_slice_index,
	                                           std::max(depth >> rt.view.current_mip_level, 1u) - 1u)
	                                : rt.view.last_array_slice_index;
	const auto view = ResolveTargetViewInfo(
	    rt.view.base_array_slice_index, last_layer, render_target_slice_offset);
	switch (view.type) {
		case TargetViewType::Image2D:
		case TargetViewType::Image2DArray: break;
		case TargetViewType::Unsupported:
			EXIT("invalid render-target view: base=%u last=%u draw_offset=%u\n",
			     rt.view.base_array_slice_index, rt.view.last_array_slice_index,
			     render_target_slice_offset);
	}
	if (graphics_debug_dump_enabled()) {
		static std::atomic_uint log_count = 0;
		const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
		if (log_id < 128) {
			LOGF("RenderColorTarget: inspect slot=%" PRIu32 " base=0x%010" PRIx64
			     " mask=0x%01" PRIx32 " attrib2_width=%" PRIu32 " attrib2_height=%" PRIu32
			     " attrib3_tile=0x%08" PRIx32 " attrib3_dim=0x%08" PRIx32 " fmt=0x%08" PRIx32
			     " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32 "\n",
			     rt_slot, rt.base.addr, mask, rt.attrib2.width, rt.attrib2.height,
			     static_cast<uint32_t>(rt.attrib3.tile_mode), rt.attrib3.dimension,
			     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
			     static_cast<uint32_t>(rt.info.channel_order));
		}
	}

	// Color-control state selects the color-buffer operation and logical blend operation.
	// The normal copy operation is a regular color write, not an attachment clear.
	// Nonlinear clear values are still stored as normalized components.
	// Metadata clears are materialized during image discovery; render-pass loads preserve contents.
	uint32_t   width  = 0;
	uint32_t       height       = 0;
	bool       tile   = false;
	const bool     standard4    = rt.attrib3.tile_mode == Prospero::TileMode::kStandard4KB;
	const bool     standard64   = rt.attrib3.tile_mode == Prospero::TileMode::kStandard64KB;
	const bool     depth_tile   = rt.attrib3.tile_mode == Prospero::TileMode::kDepth;
	const bool     standard256  = rt.attrib3.tile_mode == Prospero::TileMode::kStandard256B;
	const bool     texture_tile = standard256 || standard4 || standard64 || depth_tile;

	switch (rt.attrib3.tile_mode) {
		case Prospero::TileMode::kLinear:
		case Prospero::TileMode::kStandard256B:
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
		case Prospero::TileMode::kDepth:
		case Prospero::TileMode::kRenderTarget:
			tile = !RenderIsColorTileModeLinear(rt.attrib3.tile_mode);
			break;
		default: EXIT("unknown tile mode: %u\n", static_cast<uint32_t>(rt.attrib3.tile_mode));
	}
	if (samples > 1 && (!tile || levels != 1)) {
		EXIT("multisampled render targets require a single-mip tiled surface\n");
	}
	if (texture_tile && samples != 1) {
		EXIT("texture-tiled color render targets do not support multisampling\n");
	}
	if (volume && !tile) {
		EXIT("linear 3D render targets are unsupported\n");
	}

	width  = rt.attrib2.width + 1;
	height = rt.attrib2.height + 1;
	const auto target_format =
	    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order);
	// A surface with one stored level starts every MIP_LEVEL at the base address, so a level with
	// mip 0's padded pitch aliases mip 0's top-left texels.
	uint32_t mip = rt.view.current_mip_level;
	if (mip != 0 && levels == 1 && samples == 1 && !volume &&
	    TileGetTexturePitch(target_format.guest_format, std::max(width >> mip, 1u),
	                        rt.attrib3.tile_mode) ==
	        TileGetTexturePitch(target_format.guest_format, width, rt.attrib3.tile_mode)) {
		mip = 0;
	}
	if (mip >= levels) {
		EXIT("unsupported render-target mip range: current=%u levels=%u\n",
		     rt.view.current_mip_level, levels);
	}
	const auto bytes_per_element = target_format.bytes_per_element;
	if (bytes_per_element == 0) {
		EXIT("render-target format has no valid element size\n");
	}
	const auto transfer_format = ImageOps::RenderTargetTransferFormat(bytes_per_element);
	TileTextureBlockLayout texture_tile_layout {};
	if (texture_tile &&
	    (!TileGetTextureBlockLayout(transfer_format, rt.attrib3.tile_mode, volume,
	                                texture_tile_layout) ||
	     (rt.base.addr & (texture_tile_layout.block.block_size - 1u)) != 0 ||
	     rt.info.fmask_compression_enable || rt.info.fmask_data_compression_disable ||
	     rt.info.fmask_one_frag_mode || rt.info.cmask_fast_clear_enable ||
	     rt.info.dcc_compression_enable || rt.cmask.addr != 0 || rt.fmask.addr != 0 ||
	     rt.dcc_addr.addr != 0 || rt.dcc.data_write_on_dcc_clear_to_reg)) {
		EXIT("unsupported texture-tiled render target: addr=0x%016" PRIx64 " tile=%u"
		     " dimension=%u depth=%u levels=%u layer=%u/%u samples=%u fragments=%u bpe=%u"
		     " cmask=0x%016" PRIx64 " fmask=0x%016" PRIx64 " dcc=0x%016" PRIx64 "\n",
		     rt.base.addr, static_cast<uint32_t>(rt.attrib3.tile_mode), rt.attrib3.dimension,
		     rt.attrib3.depth, levels, view.base_layer, view.image_layers, rt.attrib.num_samples,
		     rt.attrib.num_fragments, bytes_per_element, rt.cmask.addr, rt.fmask.addr,
		     rt.dcc_addr.addr);
	}
	if (depth_tile && (is_1d || volume || view.base_layer != 0 || view.image_layers != 1)) {
		EXIT("unsupported depth-tiled color render-target view: dimension=%u depth=%u"
		     " layer=%u/%u\n",
		     rt.attrib3.dimension, rt.attrib3.depth, view.base_layer, view.image_layers);
	}
	if (standard256) {
		static std::atomic_uint log_count = 0;
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32u) {
			LOGF("RenderColorTarget: Standard256B slot=%" PRIu32 " base=0x%010" PRIx64 " %ux%u bpe=%u levels=%u layers=%u\n",
			     rt_slot, rt.base.addr, width, height, bytes_per_element, levels, view.image_layers);
		}
	}
	auto& desc = r.desc;
	desc.type              = TextureCache::BindingType::RenderTarget;
	desc.info.data            = {rt.base.addr, 0};
	desc.info.pixel_format = target_format.format;
	desc.info.guest_format = target_format.guest_format;
	desc.info.type         = volume ? Prospero::ImageType::kColor3D : Prospero::ImageType::kColor2D;
	desc.info.extent       = {width, height, depth};
	desc.info.resources       = {levels, volume ? 1u : view.image_layers};
	desc.info.bytes_per_block = bytes_per_element;
	desc.info.samples         = samples;
	desc.info.tile_mode       = rt.attrib3.tile_mode;
	if (!volume && rt.attrib3.tile_mode == Prospero::TileMode::kRenderTarget &&
	    levels > std::bit_width(std::max(width, height))) {
		EXIT("unsupported render-target mip count\n");
	}
	desc.info.UpdateSize();
	if (!desc.info.data.Valid()) {
		EXIT("render-target backing range is invalid\n");
	}
	const bool has_dcc        = rt.info.dcc_compression_enable && rt.dcc_addr.addr != 0;
	const bool has_cmask      = !rt.info.dcc_compression_enable && rt.info.cmask_fast_clear_enable &&
	                            rt.cmask.addr != 0 &&
	                            !rt.attrib3.write_vrs_rate_hint_to_cmask;
	if (has_dcc || has_cmask) {
		TileSizeAlign metadata_size {};
		const auto layers = volume ? depth : view.image_layers;
		if (has_dcc) {
			(void)TileGetDccSize(width, height, layers, bytes_per_element, levels,
			                     rt.attrib3.tile_mode, metadata_size, rt.attrib.num_fragments);
			desc.info.metadata.dcc_alpha_msb = DccAlphaOnMsb(rt.info);
		} else {
			(void)TileGetCmaskSize(width, height, layers, levels, metadata_size);
		}
		// DCC owns the clear when both planes are enabled.
		desc.info.metadata.kind = has_dcc ? ImageMetadataKind::Dcc : ImageMetadataKind::Cmask;
		desc.info.metadata.range = {has_dcc ? rt.dcc_addr.addr : rt.cmask.addr, metadata_size.size};
		desc.info.metadata.clear_word =
		    rt.clear_word0.word0 | (uint64_t {rt.clear_word1.word1} << 32u);
		desc.info.metadata.clear_register_valid = true;
	}
	desc.view_info.format = target_format.format;
	desc.view_info.type =
	    view.layer_count == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
	desc.view_info.base_level  = mip;
	desc.view_info.level_count = 1;
	desc.view_info.base_layer  = view.base_layer;
	desc.view_info.layer_count = view.layer_count;
	desc.view_info.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	auto& texture_cache        = m_context.GetTextureCache();
	r.guest_mip_level          = mip;
	r.guest_array_layer        = view.base_layer;
	uint64_t plain             = 0;
	r.image_id                 = texture_cache.FindImage(r.desc, exact_format, true, memo != nullptr ? &plain : nullptr);
	r.export_mapping           = target_format.export_mapping;
	BindRenderTarget(r.image_id);
	if (memo != nullptr) {
		if (hit && !SameColorTarget(memo->info, r)) {
			++m_target_wrong;
			LOGF("Target memo: a color hit was wrong at slot %" PRIu32 " 0x%010" PRIx64 "\n", rt_slot, rt.base.addr);
		}
		memo->rt           = rt;
		memo->mask         = mask;
		memo->slice_offset = render_target_slice_offset;
		memo->info         = r;
		memo->stamp        = plain;
	}
}

} // namespace Libs::Graphics
