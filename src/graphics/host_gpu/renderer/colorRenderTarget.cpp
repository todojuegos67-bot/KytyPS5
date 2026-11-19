#include "graphics/host_gpu/renderer/colorRenderTarget.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/replay.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics {

// KYTY_MSAA_CMASK_CLEARS=0, or KYTY_BISECT's cmask:off: a multisampled target's CMASK fast clears
// are left out, as before round 63 (its image then keeps what earlier frames drew into it).
static bool MsaaCmaskClears() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MSAA_CMASK_CLEARS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && !Bisect::MsaaCmaskOff();
}

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

// gt7-replay-1119 (KYTY_REPLAY, replay.h): a draw's color target as resolved at its place, from
// the target's registers (CB_COLORn_*, the slot's mask nibble, the draw's slice offset). The
// description is a function of those alone (and of KYTY_MSAA_CMASK_CLEARS, fixed for a run, and
// KYTY_BISECT, under which nothing is kept); the image is still found (FindImage) and bound.
namespace {
struct ReplayTarget {
	bool             kept = false;
	HW::RenderTarget rt;
	uint32_t         mask         = 0;
	uint32_t         slice_offset = 0;
	RenderColorInfo  info; // as made, before FindImage (image_id empty)
};
constexpr size_t ReplayTargetLimit = 1u << 14u;
std::unordered_map<uint64_t, ReplayTarget>& ReplayTargets() {
	thread_local std::unordered_map<uint64_t, ReplayTarget> targets;
	return targets;
}
// KYTY_REPLAY=2: the first field in which two descriptions differ, or null.
const char* DiffTargetInfo(const RenderColorInfo& a, const RenderColorInfo& b) {
	const auto& x = a.desc.info;
	const auto& y = b.desc.info;
	if (x.data.address != y.data.address || x.data.size != y.data.size) return "memoria";
	if (x.pixel_format != y.pixel_format || x.guest_format != y.guest_format) return "formato";
	if (x.type != y.type || x.extent != y.extent || x.resources != y.resources) return "extension";
	if (x.pitch != y.pitch || x.bytes_per_block != y.bytes_per_block || x.samples != y.samples ||
	    x.tile_mode != y.tile_mode)
		return "disposicion";
	if (x.mip_layout != y.mip_layout) return "mips";
	if (x.metadata.kind != y.metadata.kind || x.metadata.range.address != y.metadata.range.address ||
	    x.metadata.range.size != y.metadata.range.size ||
	    x.metadata.clear_word != y.metadata.clear_word ||
	    x.metadata.clear_word1 != y.metadata.clear_word1 ||
	    x.metadata.dcc_alpha_msb != y.metadata.dcc_alpha_msb)
		return "metadatos";
	if (!(a.desc.view_info == b.desc.view_info) || a.desc.type != b.desc.type) return "vista";
	if (a.guest_mip_level != b.guest_mip_level || a.guest_array_layer != b.guest_array_layer ||
	    a.target_slot != b.target_slot)
		return "nivel/capa";
	if (std::memcmp(&a.export_mapping, &b.export_mapping, sizeof(a.export_mapping)) != 0)
		return "export";
	return nullptr;
}
} // namespace

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
	// gt7-replay-1119 (KYTY_REPLAY): the description kept at this draw's place, for the same
	// registers.
	ReplayTarget* replay      = nullptr;
	bool          replay_same = false;
	if (const auto draw = !ignore_target_mask && !exact_format ? Replay::DrawKey() : 0;
	    draw != 0) {
		if (Bisect::Active() || graphics_debug_dump_enabled()) {
			Replay::NoteMiss(Replay::Part::Target, Replay::Miss::Ineligible);
		} else {
			auto& targets = ReplayTargets();
			if (targets.size() >= ReplayTargetLimit) {
				targets.clear();
				Replay::NoteDropped(Replay::Part::Target);
			}
			replay      = &targets[Replay::PartKey(draw, 2u, rt_slot)];
			replay_same = replay->kept && replay->mask == mask &&
			              replay->slice_offset == render_target_slice_offset &&
			              std::memcmp(&replay->rt, &rt, sizeof(rt)) == 0;
			if (replay_same) {
				Replay::NoteHit(Replay::Part::Target);
			} else {
				Replay::NoteMiss(Replay::Part::Target,
				                 replay->kept ? Replay::Miss::Words : Replay::Miss::NoEntry);
			}
			if (replay_same && Replay::Mode() == 1) {
				r          = replay->info;
				r.image_id = m_context.GetTextureCache().FindImage(r.desc, exact_format);
				BindRenderTarget(r.image_id);
				return;
			}
		}
	}
	// NUM_SAMPLES (coverage) and NUM_FRAGMENTS (stored colors) are both log2. EQAA stores fewer
	// fragments than it has coverage samples (e.g. 4s/2f) and keeps the sample->fragment map in
	// FMASK. The guest surface layout follows the fragment count, so the host image does too;
	// the extra coverage samples only reach the depth attachment (see the pipeline's mixed-sample
	// path) and the FMASK emulation maps each sample onto an existing fragment.
	const auto samples = render_sample_count(rt.attrib.num_fragments);
	if (samples == 0 || render_sample_count(rt.attrib.num_samples) == 0 ||
	    rt.attrib.num_samples < rt.attrib.num_fragments) {
		EXIT("unsupported render-target sample configuration: samples=%u fragments=%u\n",
		     rt.attrib.num_samples, rt.attrib.num_fragments);
	}
	if (rt.attrib.num_samples != rt.attrib.num_fragments) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("RenderColorTarget: EQAA target %ux coverage / %ux fragments, host image uses %u "
			     "samples\n",
			     render_sample_count(rt.attrib.num_samples), samples, samples);
		}
	}
	const uint32_t levels = rt.attrib2.num_mip_levels + 1u;
	if (levels == 0 || levels > 16 || rt.view.current_mip_level >= levels) {
		EXIT("unsupported render-target mip range: current=%u levels=%u\n",
		     rt.view.current_mip_level, levels);
	}
	static constexpr std::array image_types {Prospero::ImageType::kColor1D,
	                                         Prospero::ImageType::kColor2D,
	                                         Prospero::ImageType::kColor3D};
	if (rt.attrib3.dimension >= image_types.size()) {
		EXIT("unsupported render-target dimension: %u\n", rt.attrib3.dimension);
	}
	const auto image_type = image_types[rt.attrib3.dimension];
	const bool is_1d      = image_type == Prospero::ImageType::kColor1D;
	const bool volume     = image_type == Prospero::ImageType::kColor3D;
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
	uint32_t   height = 0;
	uint32_t   pitch  = 0;
	uint64_t   size   = 0;
	bool       tile   = false;
	const bool     standard4    = rt.attrib3.tile_mode == Prospero::TileMode::kStandard4KB;
	const bool     standard64   = rt.attrib3.tile_mode == Prospero::TileMode::kStandard64KB;
	const bool     depth_tile   = rt.attrib3.tile_mode == Prospero::TileMode::kDepth;
	const bool     texture_tile = standard4 || standard64 || depth_tile;

	switch (rt.attrib3.tile_mode) {
		case Prospero::TileMode::kLinear:
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

	width  = rt.attrib2.width + 1;
	height = rt.attrib2.height + 1;
	const auto target_format =
	    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order);
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
	if ((standard64 || depth_tile) &&
	    (rt.attrib3.dimension != 1 || rt.attrib3.depth != 0 ||
	     (depth_tile && (view.base_layer != 0 || view.image_layers != 1)))) {
		EXIT("unsupported 64KB texture-tiled render-target view: dimension=%u depth=%u"
		     " layer=%u/%u\n",
		     rt.attrib3.dimension, rt.attrib3.depth, view.base_layer, view.image_layers);
	}
	if (samples == 1) {
		pitch = TileGetTexturePitch(transfer_format, width, rt.attrib3.tile_mode);
	} else {
		pitch = TileGetRenderTargetPitch(width, bytes_per_element, rt.attrib.num_fragments);
	}
	if (pitch == 0) {
		EXIT("unsupported render-target pitch: width=%u bytes=%u\n", width, bytes_per_element);
	}

	TileSizeOffset    mip_sizes[16] {};
	TilePaddedSize    mip_padded[16] {};
	TileSurfaceLayout volume_layout {};
	uint64_t          backing_size = 0;
	if (volume) {
		const TileSurfaceDescription description {transfer_format,
		                                          rt.attrib3.tile_mode,
		                                          TileSurfaceDimension::Dim3D,
		                                          width,
		                                          height,
		                                          depth,
		                                          levels,
		                                          1};
		if (!tile || !TileGetTiledTextureLayout(description, volume_layout)) {
			EXIT("unsupported 3D render-target layout: %ux%ux%u levels=%u tile=%u\n", width, height,
			     depth, levels, static_cast<uint32_t>(rt.attrib3.tile_mode));
		}
		size         = volume_layout.block_slice_size;
		backing_size = volume_layout.total_size;
	} else {
		TileSizeAlign layout {};
		bool          valid_layout = false;
		if (samples == 1) {
			TileGetTextureSize(transfer_format, width, height, levels, rt.attrib3.tile_mode,
			                   &layout, mip_sizes, mip_padded);
			valid_layout = layout.size != 0 && layout.align != 0 &&
			               (rt.attrib3.tile_mode != Prospero::TileMode::kRenderTarget ||
			                levels <= std::bit_width(std::max(width, height)));
		} else {
			valid_layout = TileGetRenderTargetSize(width, height, pitch, bytes_per_element,
			                                       layout, rt.attrib.num_fragments);
			mip_sizes[0]  = {layout.size, 0, 0, 0, 0, 0};
			mip_padded[0] = {pitch, height};
		}
		if (!valid_layout) {
			EXIT("unsupported render-target layout: %ux%u pitch=%u bytes=%u levels=%u\n", width,
			     height, pitch, bytes_per_element, levels);
		}
		size = layout.size;
	}
	if (size == 0 || (!volume && size > UINT64_MAX / view.image_layers)) {
		EXIT("render-target memory footprint is invalid\n");
	}
	if (!volume) {
		backing_size = size * view.image_layers;
	}
	if (backing_size == 0) {
		EXIT("render-target backing is empty\n");
	}
	if (!GuestRange {rt.base.addr, backing_size}.Valid()) {
		EXIT("render-target backing range is invalid\n");
	}

	auto& desc = r.desc;
	desc.type              = TextureCache::BindingType::RenderTarget;
	desc.info.data         = {rt.base.addr, backing_size};
	desc.info.pixel_format = target_format.format;
	desc.info.guest_format = target_format.guest_format;
	desc.info.type         = image_type;
	desc.info.extent       = {width, height, depth};
	desc.info.resources    = {levels, volume ? 1u : view.image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block = bytes_per_element;
	desc.info.samples         = samples;
	desc.info.tile_mode       = rt.attrib3.tile_mode;
	const bool has_dcc        = rt.info.dcc_compression_enable && rt.dcc_addr.addr != 0;
	// A multisampled target's CMASK too (it covers pixels, as a single-sample one's, and a fast
	// clear zeroes it when no DCC owns the clear; its FMASK only maps samples to fragments, which
	// the host image does not need). GT7 clears its course maps (513x326 and 833x466, 2 samples)
	// every frame by zeroing their CMASK and FMASK with a compute fill: without this the maps
	// kept every frame's car icons (2026-10-16). Its metadata must start on a 4 KiB block.
	const bool msaa_cmask = samples > 1 && MsaaCmaskClears() && rt.cmask.addr % 0x1000u == 0;
	const bool has_cmask  = !rt.info.dcc_compression_enable && rt.info.cmask_fast_clear_enable &&
	                       rt.cmask.addr != 0 &&
	                       (samples == 1 ? !rt.info.fmask_compression_enable : msaa_cmask) &&
	                       !rt.attrib3.write_vrs_rate_hint_to_cmask;
	if (has_cmask && samples > 1) {
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint64_t> logged;
		std::lock_guard                     lock(logged_mutex);
		if (logged.size() < 16 && logged.insert(rt.base.addr).second) {
			LOGF("Render target: 0x%012" PRIx64 " %ux%u, %u samples: its CMASK fast clears "
			     "(CMASK 0x%012" PRIx64 ", FMASK 0x%012" PRIx64 ") clear it "
			     "(KYTY_MSAA_CMASK_CLEARS=0 leaves them out)\n",
			     rt.base.addr, width, height, samples, rt.cmask.addr, rt.fmask.addr);
		}
	}
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
		// DCC owns the clear when both planes are enabled; single-sample CMASK stays expanded.
		desc.info.metadata.kind = has_dcc ? ImageMetadataKind::Dcc : ImageMetadataKind::Cmask;
		desc.info.metadata.range = {has_dcc ? rt.dcc_addr.addr : rt.cmask.addr, metadata_size.size};
		desc.info.metadata.clear_word           = rt.clear_word0.word0;
		desc.info.metadata.clear_word1          = rt.clear_word1.word1;
		desc.info.metadata.clear_register_valid = true;
	}
	for (uint32_t level = 0; level < levels; level++) {
		if (volume) {
			const auto& mip             = volume_layout.mips[level];
			desc.info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
			continue;
		}
		const auto level_offset =
		    mip_sizes[level].src_size != 0 ? mip_sizes[level].src_offset : mip_sizes[level].offset;
		const auto level_size =
		    static_cast<uint64_t>(mip_sizes[level].src_size != 0 ? mip_sizes[level].src_size
		                                                         : mip_sizes[level].size) *
		    view.image_layers;
		desc.info.mip_layout[level] = {
		    level_offset,
		    level_size,
		    mip_padded[level].width,
		    mip_padded[level].height,
		};
	}
	desc.view_info.format = target_format.format;
	if (is_1d) {
		desc.view_info.type =
		    view.layer_count == 1 ? vk::ImageViewType::e1D : vk::ImageViewType::e1DArray;
	} else {
		desc.view_info.type =
		    view.layer_count == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	}
	desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
	desc.view_info.base_level  = rt.view.current_mip_level;
	desc.view_info.level_count = 1;
	desc.view_info.base_layer  = view.base_layer;
	desc.view_info.layer_count = view.layer_count;
	desc.view_info.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	auto& texture_cache        = m_context.GetTextureCache();
	r.guest_mip_level          = rt.view.current_mip_level;
	r.guest_array_layer        = view.base_layer;
	r.export_mapping           = target_format.export_mapping;
	if (replay != nullptr) {
		// The description before FindImage (which may move the view into a larger image).
		if (replay_same) {
			Replay::NoteVerified(Replay::Part::Target, DiffTargetInfo(replay->info, r),
			                     rt.base.addr, rt_slot);
		} else {
			std::memcpy(static_cast<void*>(&replay->rt), &rt, sizeof(rt));
			replay->mask          = mask;
			replay->slice_offset  = render_target_slice_offset;
			replay->info          = r;
			replay->info.image_id = {};
			replay->kept          = true;
			Replay::NoteStored(Replay::Part::Target);
		}
	}
	r.image_id                 = texture_cache.FindImage(r.desc, exact_format);
	BindRenderTarget(r.image_id);
}

} // namespace Libs::Graphics
