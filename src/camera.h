// Camera finder (phase 2, for the upscalers): the game's projection matrix, found generically in the constant buffers
// its scene draws read, and from it the camera jitter, the vertical FOV and the near / far planes.
//
// How (DXVK 2.x, D3D11): DXVK binds each constant buffer as a VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER through a descriptor
// update template (one VkDescriptorBufferInfo per binding). We record the templates' uniform-buffer entries, every
// buffer's memory binding and every mapped allocation, and while a pass that draws into the main depth buffer (the
// analyzer's pick) is being recorded, the first few hundred uniform ranges its draws bind are read from host memory and
// scanned, 16-byte aligned, for a perspective matrix in either memory order:
//   form T (HLSL column_major packing of a row-vector matrix): m0=a  m5=b  m10=c  m11=d  m14=+-1  m15=0, jitter at m2, m6
//   form R (row-major / the transposed convention):            m0=a  m5=b  m10=c  m14=d  m11=+-1  m15=0, jitter at m8, m9
// with the off-diagonal terms ~0 and b/a matching the render aspect. The jitter is the (m2, m6) / (m8, m9) term, in NDC:
// pixels = ndc * size / 2 (y flipped), the FidelityFX / DLSS / XeSS convention. It counts as "detected" once it takes
// several distinct sub-pixel values over the last frames (a TAA sequence), else the game does not jitter (or uploads only
// a combined view-projection, which this cannot see: then no jitter is known and 0 is used).
//
// Halton injection (menu): when a projection is found but does not jitter (the game's TAA off), a Halton(2,3) offset is
// added to it in the mapped constant memory right before the GPU reads it, so the upscaler gets jittered frames.
// Generic: only matrix shapes are used, never addresses, offsets or shader hashes.
#pragma once
#include "common.h"

namespace inj::camera {

// Resource tracking (device hooks).
void OnCreateTemplate(VkDescriptorUpdateTemplate tmpl, const VkDescriptorUpdateTemplateCreateInfo& info);
void OnDestroyTemplate(VkDescriptorUpdateTemplate tmpl);
void OnCreateBuffer(VkBuffer buffer, VkDeviceSize size);
void OnBindBufferMemory(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset);
void OnDestroyBuffer(VkBuffer buffer);
void OnMapMemory(VkDeviceMemory memory, VkDeviceSize offset, void* data);
void OnUnmapMemory(VkDeviceMemory memory);

// The scene-pass window (record order, this thread): set at vkCmdBeginRendering, cleared at vkCmdEndRendering.
void SetScenePass(bool scene);
// Descriptor updates (inside a scene pass the uniform ranges they bind are scanned).
void OnUpdateWithTemplate(VkDescriptorUpdateTemplate tmpl, const void* data);
void OnUpdateDescriptorSets(uint32_t count, const VkWriteDescriptorSet* writes);
void OnCmdUpdateBuffer(const void* data, VkDeviceSize size);

// Once per presented frame: settle what this frame's scan found. render: the scene's size (the main depth's).
void EndFrame(VkExtent2D render);

struct State {
	bool     found = false;           // a projection matrix seen in the last frames
	bool     jitter_detected = false; // it takes several sub-pixel offsets (the game's own TAA sequence)
	bool     injecting = false;       // our Halton offsets are being written into it
	float    jitter_px[2] {0, 0};     // this frame's (detected or injected), pixels, FidelityFX convention
	float    proj[16] {};             // row-vector convention (D3D): row r at proj[4r..4r+3], jitter removed
	float    fov = 0;                 // vertical, radians
	float    near_plane = 0, far_plane = 0;
	bool     reversed = false;        // from the matrix (near maps to 1)
	char     form = '-';              // 'T' or 'R' (memory order), '-' none
	uint32_t distinct = 0;            // distinct jitter values over the last 32 frames
	uint32_t scanned = 0, candidates = 0; // last frame's scanned ranges / matching matrices
};
State Current();
// The Halton offset the injector would use this frame (pixels), for a phase count from the upscale ratio.
void  SetInjection(bool on, float ratio);

void DrawMenu();
void Shutdown();

} // namespace inj::camera
