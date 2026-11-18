// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/bb_present_sr.h"
#include "video_core/renderer_vulkan/bb_ui_cut.h"

#include <memory>
#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace BB {
class RtReflect; // bb-rt-reflect
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    /// The copy itself (CopyBuffer adds bb-dma-dedup's statistics and skip).
    void CopyBufferImpl(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// bb-native fault-around: after a guest write fault at addr, unprotects the rest of its
    /// aligned 64 KiB block in one batch (BB_NATIVE_FAULTAROUND=0 off, =N pages, default 16).
    void FaultAround(VAddr addr);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    void OnFence();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    // bb-rt-reflect: G-buffer snapshot at the composite draw, pass before the post-process.
    void BbRtReflectSnapshot(VideoCore::Image* d32, VideoCore::Image* rt0, VideoCore::Image* rt1);
    void BbRtReflectPost(const GraphicsPipeline& pipeline);
    void BbRtLightRun(const GraphicsPipeline& pipeline); // bb-rt-light: before the composite
    bool BindResources(const Pipeline* pipeline);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0, bool is_indirect = false);

    void ResetBindings(bool is_compute);
    // bb-clean-021-bbport: Supermedo's HUD cut (bb_ui_cut.h): his OnFrameStart / OnColorTarget /
    // OnDraw before the draw's resources are bound.
    void BbUiDraw(const GraphicsPipeline* pipeline);

public:
    // bb-hud-native: a Scaleform HAL BeginScene / EndScene marker in the command stream.
    void BbHudNativeMarker(bool begin, u32 seq);

private:
    bool BbUiSingleTarget(const GraphicsPipeline* pipeline) const;
    void BbLodBias(const GraphicsPipeline* pipeline);
    // Supermedo's UpdateUiTransform: the scale of a draw into the output-size UI images.
    void BbUpdateUiTransform();
    // bb-hud-fullres: the scissor of viewport `index` for a draw laid out for the output.
    vk::Rect2D BbFullResScissor(u32 index) const;
    void BbObjMvDraw(const GraphicsPipeline* pipeline, u64 identity, u32 count, u32 instances);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    ImageBinding db_desc;
    mutable bool bb_jitter_draw = false;
    u32 bb_prof_label = 0; // bb-gpu-prof: the label of the pass BeginRendering set up
    u32 BbProfPassLabel(const RenderState& state, const GraphicsPipeline* pipeline);
    // bb-gpu6 (BB_LOD_BIAS): extra texture LOD bias of this draw's samplers (scene draws while
    // upscaling), and the one BindResources applies (0 for compute).
    float bb_lod_bias = 0.0f, bb_bind_lod_bias = 0.0f;
    u32 bb_tex_filter = 0, bb_bind_tex_filter = 0; // bb-tex: BB::Texture::FilterBits()

public:
    BB::SceneDepth bb_scene_depth; // bb-clean-021 (bb_present_sr.h)
    BB::UiCut bb_cut;

private:
    // A pass drawn into the upscaler's output-size images (Supermedo's ui_pass / target_scale).
    struct {
        bool active = false, native_ui = false;
        std::pair<u32, u32> guest{1, 1}, output{1, 1};
    } bb_ui_pass;
    mutable std::array<float, 2> bb_target_scale{1.0f, 1.0f};
    // bb-hud-fullres: this draw is in the movie viewport's (output) pixels; the guest target's
    // size and the size of the images it is drawn into.
    bool bb_fullres_draw = false;
    std::pair<u32, u32> bb_fullres_guest{1, 1}, bb_fullres_extent{1, 1};
    bool bb_display_pass = false; // the previous draw went into a display buffer

private:

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        vk::AccessFlags2 src_access;
    };
    std::vector<BoundBuffer> bound_buffers;
    std::vector<VideoCore::ImageId> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    bool attachment_feedback_loop{};
    bool needs_barrier{};

    std::unique_ptr<BB::RtReflect> bb_rt_reflect; // bb-rt: reflect / volumetric / shadows
    // bb-rt-reflect: the composite's HDR target (guest address) and whether this frame's pass
    // is still to run (snapshot taken, no draw has read the HDR target since); forward draws
    // into it in between (for the log).
    VAddr bb_rt_hdr_addr = 0;
    bool bb_rt_refl_pending = false;
    u32 bb_rt_refl_fwd = 0;
    u32 bb_rt_refl_chain_logged = 0;
    bool bb_rt_reflect_hook_first = false;
    u32 bb_rt_refl_reads = 0;
    // bb-rt-light: the largest single-sample depth target written (depth write on) since the
    // last composite draw, in command-stream order (no frame counter: the flip counter is bumped
    // on the presenter thread and drifts against the GPU command stream). The composite samples
    // only an R32F copy of the scene depth, so the lighting takes the D32 from here.
    // Every depth target bound since the last composite (for the selection and its log); the
    // composite picks the largest written single-sample one with the scene's aspect ratio.
    struct BbRtDepthSeen {
        VideoCore::ImageId id{};
        u64 uid = 0;
        VAddr addr = 0;
        u32 w = 0, h = 0, samples = 1;
        vk::Format format{};
        bool depth_test = false, depth_write = false;
        // Bound with the G-buffer MRT (Ghidra: renderer FUN_026c39f0 creates GBuffer RT0
        // (WS normal) / RT1 with format 10 = RGBA8 at the scene size and attaches the shared scene
        // depth +0x200 to them, to the HDR target and to Light Acc): >= 2 colour attachments of
        // the depth's size, >= 2 of them RGBA8. Counted per render-pass begin; rt0 = that MRT's first RGBA8 target (RT0).
        u32 gbuf_draws = 0;
        VideoCore::ImageId rt0_id{};
        u64 rt0_uid = 0;
        VideoCore::ImageId rt1_id{}; // the MRT's second RGBA8 target (GBuffer RT1: shininess)
        u64 rt1_uid = 0;
    };
    std::vector<BbRtDepthSeen> bb_rt_depth_seen;
    bool bb_rt_depth_fresh = false;  // a depth target was written since the last composite
    // The last selected scene D32 (kept across frames: fallback when no candidate matches).
    VideoCore::ImageId bb_rt_depth_last_id{};
    u64 bb_rt_depth_last_uid = 0;
    VAddr bb_rt_depth_last_addr = 0;
    bool bb_rt_light_gap = true;     // the last composite (not a duplicate) did not run the pass
};

} // namespace Vulkan
