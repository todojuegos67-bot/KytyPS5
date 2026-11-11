// SPDX-FileCopyrightText: Copyright 2026 IFreemz (shadps4_dlss bridge), bbport contributors, KytyPS5 contributors
// SPDX-License-Identifier: MIT
//
// C interface between kyty_emulator and libkyty_dlss.so (tools/local/dlss-bridge), the Linux way to NVIDIA DLSS:
// Streamline exists for Windows only, so on Linux the emulator loads this bridge at run time, which alone links
// NVIDIA's DLSS (NGX) SDK and loads libnvidia-ngx-dlss.so (and libnvidia-ngx-dlssg.so for frame generation) from
// its folder. The emulator contains no NVIDIA code and runs as before without the bridge.
// Adapted from bbport's bridge (spacepepperoni/uncharted_Linux, gpu/dlss_bridge), itself from
// IFreemz/shadPS4-Bloodborne-DLSS-FSR.
//
// ABI 3 (this copy): UTF-8 paths, the motion vectors' scale for DLSS, and frame generation's camera as the emulator
// knows it (its matrices, vectors and motion vector scale) instead of a guessed projection.

#ifndef EMULATOR_SRC_LOCAL_KYTY_DLSS_BRIDGE_H_
#define EMULATOR_SRC_LOCAL_KYTY_DLSS_BRIDGE_H_

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KYTY_DLSS_BRIDGE_ABI 3
#define KYTY_DLSS_BRIDGE_LIBRARY "libkyty_dlss.so"
#define KYTY_DLSS_BRIDGE_ENTRY "KytyDlssGetApi"

typedef void (*KytyDlssLogFn)(int warning, const char* message);

typedef struct KytyDlssImage {
	VkImage                 image;
	VkImageView             view;
	VkImageSubresourceRange range;
	VkFormat                format;
	uint32_t                width;
	uint32_t                height;
} KytyDlssImage;

// DLSS super resolution (DLAA when input and output are the same size).
typedef struct KytyDlssFeature {
	uint32_t input_width;
	uint32_t input_height;
	uint32_t output_width;
	uint32_t output_height;
	int32_t  quality;        // 0 DLAA, 1 Quality, 2 Balanced, 3 Performance, 4 Ultra Performance
	int32_t  depth_inverted; // 1: near is 1
	uint32_t preset;         // NVSDK_NGX_DLSS_Hint_Render_Preset (1 = A ...), 0 = the driver's default
	int32_t  hdr;            // 1: linear HDR color with automatic exposure, 0: tone-mapped color
} KytyDlssFeature;

typedef struct KytyDlssEvaluate {
	KytyDlssImage color;  // sampled (shader read-only layout)
	KytyDlssImage depth;  // sampled, hardware depth
	KytyDlssImage motion; // sampled, at the render size
	KytyDlssImage output; // storage, general layout
	float         jitter_x;   // render pixels
	float         jitter_y;
	float         mv_scale_x; // motion vector * scale = render pixels, current to previous
	float         mv_scale_y;
	int32_t       reset;
	float         frame_ms;
	float         sharpness; // 0..1; 0 leaves the output unsharpened
} KytyDlssEvaluate;

// DLSS frame generation: frames between two finished frames.
typedef struct KytyDlssgFeature {
	uint32_t width; // backbuffer: input and output color
	uint32_t height;
	VkFormat format;
	uint32_t render_width; // depth and motion vectors
	uint32_t render_height;
} KytyDlssgFeature;

typedef struct KytyDlssgEvaluate {
	KytyDlssImage backbuffer; // sampled: the finished frame (UI included)
	KytyDlssImage depth;      // sampled, hardware depth values
	KytyDlssImage motion;     // sampled, at the render size
	KytyDlssImage output;     // storage, general layout: the generated frame
	KytyDlssImage hudless;    // sampled: the backbuffer before the UI; image null when none
	// Row-major, row vectors (p' = p * M), without the TAA jitter: Streamline's and NGX's convention.
	float    camera_view_to_clip[16];
	float    clip_to_camera_view[16];
	float    clip_to_prev_clip[16];
	float    prev_clip_to_clip[16];
	float    camera_pos[3];
	float    camera_up[3];
	float    camera_right[3];
	float    camera_fwd[3];
	float    jitter_x; // render pixels
	float    jitter_y;
	float    mv_scale_x; // motion vector * scale = UV units ([-1, 1]), current to previous
	float    mv_scale_y;
	float    camera_near;
	float    camera_far;
	float    camera_fov; // vertical, radians
	float    camera_aspect;
	int32_t  depth_inverted;
	int32_t  reset;
	uint32_t multi_frame_count; // frames generated for this pair (1: 2x .. 3: 4x)
	uint32_t multi_frame_index; // 1 .. multi_frame_count, in order
	// The part of the backbuffer the game image covers (width 0: all of it).
	uint32_t backbuffer_x, backbuffer_y, backbuffer_width, backbuffer_height;
} KytyDlssgEvaluate;

typedef struct KytyDlssApi {
	uint32_t abi;
	// UTF-8 paths: the folder with libnvidia-ngx-dlss.so.*, and a writable data folder.
	int32_t (*Configure)(const char* library_directory, const char* data_directory, KytyDlssLogFn log);
	// The Vulkan extensions NGX needs (DLSS and, when the GPU has it, frame generation); the arrays stay valid until
	// Shutdown. Before the instance and before the device are created.
	int32_t (*InstanceExtensions)(uint32_t* count, const VkExtensionProperties** extensions);
	int32_t (*DeviceExtensions)(VkInstance instance, VkPhysicalDevice physical, uint32_t* count,
	                            const VkExtensionProperties** extensions);
	int32_t (*Initialize)(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
	                      PFN_vkGetInstanceProcAddr get_instance_proc, PFN_vkGetDeviceProcAddr get_device_proc);
	// A new feature retires the previous one, released some evaluations later (when the GPU is done with it).
	int32_t (*CreateFeature)(VkCommandBuffer command, const KytyDlssFeature* feature);
	int32_t (*Evaluate)(VkCommandBuffer command, const KytyDlssEvaluate* evaluate);
	// The caller must have waited for the GPU work that used the features.
	void (*Shutdown)(void);
	// Frames frame generation can make per pair of rendered frames, after Initialize: 0 unavailable, 1 (2x), 3 (up to
	// 4x, GeForce RTX 50).
	int32_t (*FrameGenMaxCount)(void);
	int32_t (*FrameGenCreate)(VkCommandBuffer command, const KytyDlssgFeature* feature);
	// Called multi_frame_count times per rendered frame, multi_frame_index 1, 2, ...
	int32_t (*FrameGenEvaluate)(VkCommandBuffer command, const KytyDlssgEvaluate* evaluate);
} KytyDlssApi;

typedef const KytyDlssApi* (*KytyDlssGetApiFn)(void);

#ifdef __cplusplus
}
#endif

#endif // EMULATOR_SRC_LOCAL_KYTY_DLSS_BRIDGE_H_
