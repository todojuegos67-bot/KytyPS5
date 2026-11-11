// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <string>
#include <unordered_set>

#include "video_core/bb/bb_fg.h"

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

    // Bloodborne upscaling to the window size (vk_rasterizer_bb.cpp): when this frame's flip image
    // was upscaled before its HUD, adds the HUD onto the upscaled image (recorded into cmd, the
    // flip image in shader-read layout) and returns that image's view and size to present.
    // The presenter's view format of the flip (GetFrameViewFormat): A8R8G8B8 display buffers are
    // viewed as B8G8R8A8 over the R8G8B8A8 flip image, which swaps red and blue natively; the
    // window-size path reads the flip through the same format (0.24).
    void BBSetPresentViewFormat(vk::Format view_format);
    bool BBPresentComposite(vk::CommandBuffer cmd, VideoCore::Image& flip, vk::Format view_format,
                            vk::ImageView& out_view, vk::Extent2D& out_size);
    // Copies the presented image (before its HUD) and records the window-size upscale of it.
    bool BBPresentUpscale(VideoCore::Image& flip, bool late = false);
    // 1.07 (BB_HUD_SHARP, bb_respatch.h): after the HUD-point upscale, the output-size canvas is
    // filled with the upscaled scene; the frame's later draws into the flip are drawn into it.
    void BBHudSharpBegin(VideoCore::Image& flip);
    bool BBHudSharpTarget(const VideoCore::Image& image, vk::Format view_format,
                          vk::ImageView& view, vk::ImageLayout& layout, u32& width, u32& height);
    bool BBHudSharpPresent(vk::CommandBuffer cmd, VideoCore::Image& flip, vk::Format view_format,
                           vk::ImageView& out_view, vk::Extent2D& out_size);
    // 1.16 (bb-hud-116): a draw that samples the HUD target (the game's final UI-to-flip pass)
    // after the HUD point reads the canvas instead (twin_view: the HUD-less canvas, for the
    // frame generation twin draw); a depth-tested HUD draw uses an output-size depth twin.
    bool BBHudSharpSampled(const VideoCore::Image& image, const VideoCore::ImageViewInfo& view_info,
                           bool draws_into_it, vk::ImageView& view, vk::ImageLayout& layout,
                           vk::ImageView& twin_view);
    bool BBHudSharpDepth(VideoCore::Image& depth, vk::ImageView& view, vk::ImageLayout& layout);
    void BBHudSharpTwinDone();
    bool BBHudSharpDepthOnly(const VideoCore::Image& depth, u32& width, u32& height);
    // The last present went through the HUD canvas (the presenter's [frame] line).
    static bool BBHudSharpPresented();
    // 1.40 flash guard: the last BBHudSharpPresent replaced the frame by the last fully written
    // canvas (or black) instead of an image not written with this frame's scene.
    static bool BBFlashGuarded();
    // 1.40 flash guard: the presenter skipped frame generation for one real frame (counted).
    static void BBFlashGuardFgSkip();
    // 0.39 (BB_FG_NATIVE): frame generation's inputs with no upscaler; the image is not changed.
    bool BBFgNativeAnalyze(VideoCore::Image& flip, bool late = false);
    static const char* BBPresentWhy();
    // Bloodborne: the end of the presenter's pass, the last per-stage GPU timestamp.
    static void BBStampPresentEnd(vk::CommandBuffer cmd);
    // Bloodborne frame generation (bb_fg.h): this frame's depth and motion vectors copied into
    // the slot's own images (the game reuses its own before the frame is presented), the camera
    // and jitter of its upscale; and the upscaled image before the HUD at the output size (the
    // composite without the HUD) when the window-size path presented it. False: not a 3D frame.
    // BB_HUD_DIAG=1 (diagnostic only, nothing drawn differently): the blend state of the draws
    // into the presented image after the HUD-less copy, and the HUD's alpha in that image.
    void BBHudBlendLog(VAddr target, u32 enable, u32 csrc, u32 cfunc, u32 cdst, u32 separate,
                       u32 asrc, u32 afunc, u32 adst, u32 write_mask);
    void BBHudAlphaCheck(VideoCore::Image& flip);
    // Frame generation's HUD-less image on the in-place path: the image the presented image is
    // copied from (A) is copied before its first HUD draw, and the game's own per-channel curve
    // from A to the presented image (learned from the frame) is applied to that copy.
    void BBNoteSample(VAddr address, u32 width, u32 height);
    void BBHudCapturePoint(VideoCore::ImageId target);
    void BBHudlessLut(VideoCore::Image& flip);
    bool BBFrameGenCapture(u32 slot, BB::FG::Inputs& out, vk::ImageView& hudless_view,
                           vk::Format view_format);

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
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
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
    // BB_SCENE_UPSCALE: host / guest size of the current draw's scaled targets (1 when none).
    float bb_rt_scale_x = 1.0f, bb_rt_scale_y = 1.0f;
    u32 bb_rt_guest_w = 0, bb_rt_guest_h = 0;
    // 1.07 sharp HUD: this draw goes into the output-size HUD canvas instead of the flip.
    bool bb_hud_redirect = false;
    // 1.16: this draw samples the HUD target (bound to the canvas), its twin bindings
    // (image_infos index, HUD-less view) and the twin draw's target (the HUD-less canvas).
    bool bb_hs_sampled = false;
    std::vector<std::pair<size_t, vk::ImageView>> bb_hs_twin;
    vk::ImageView bb_hs_twin_dst{};
    vk::ImageLayout bb_hs_twin_layout{};
    u32 bb_num_cbs = 0;
    bool BBPatchConstants(const void* data, u64 size, std::vector<u32>& out) const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0, bool is_indirect = false);

    void ResetBindings(bool is_compute);
    // bb-149-objmv: object motion vectors v2 (bb_objmv2.h) of a draw.
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

    VideoCore::TextureCache::ImageDesc bb_gb_depth_desc{};
    bool bb_gb_depth_valid{};

    // Bloodborne upscaling (video_core/bb/bb_upscale.h, vk_rasterizer_bb.cpp).
    bool BBInject(VideoCore::ImageId color_id, const VideoCore::ImageViewInfo& view_info,
                  vk::ImageView& out_view, vk::ImageLayout& out_layout);
    bool BBUpscaledView(const VideoCore::ImageViewInfo& view_info, vk::ImageView& out_view,
                        vk::ImageLayout& out_layout);
    // BB 0.80 native chain: the upscaled image into the source's output-size image (+ content
    // check); BB_DUMP_FRAME frame dump.
    void BBNativeFillSource(VideoCore::ImageId scaled_id, VideoCore::TextureCache::ImageDesc desc);
    void BBDumpLog(const std::string& line);
    bool BBDumpOn();
    void BBDumpImage(VideoCore::Image& image, const std::string& what);
    void BBDumpPass(std::span<const VideoCore::ImageId> targets, const std::string& desc);
    std::unordered_set<const void*> bb_scene_cs; // compute pipelines of the scene passes
    void BBHudlessCopy(VideoCore::Image& flip);
    bool BBRecordUpscale(vk::Image image, vk::ImageView view, vk::Format format,
                         vk::ImageUsageFlags usage, bool late = false, bool mv_only = false);
    void BBOnDraw(u32 num_cb, u64 vs_hash, u64 ps_hash);
    bool attachment_feedback_loop{};
    bool needs_barrier{};
};

} // namespace Vulkan
