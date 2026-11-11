#pragma once

// Shader modules kept as SPIR-V instead of driver objects (KYTY_MODULE_STORE, default on where VK_KHR_maintenance5 is
// enabled): a pipeline then gets its shaders' SPIR-V inline (CreatePipelineHandle) and the driver keeps none of it
// once the pipeline is made. NVIDIA kept a copy of every module's SPIR-V while the module lived: ~1.8 GB of heap with
// the 1.07 warmup's ~16k modules, and more with every area played. The pipeline keys (the static store's) are the
// same either way. A stored module's handle is the store's own (an address in an array of its
// own, never a driver object): only pipeline creation takes it. Its SPIR-V waits zstd-compressed in a scratch file
// (a fifth of the bytes), the last few loaded in memory.

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics::ModuleStore {

// At device creation: whether the device takes SPIR-V inline.
void Enable(bool maintenance5);
[[nodiscard]] bool Enabled();
// A stored module of `spirv`, or null (no scratch file, or the store is off): the caller makes a driver module.
vk::ShaderModule Add(const std::vector<uint32_t>& spirv);
// A stored module of SPIR-V in another scratch file (the shader prefetch's): `stored` bytes at `offset`, zstd-compressed
// or as they are, `words` words of SPIR-V. Read from there: nothing copied. The caller keeps the file open as long as the
// process runs. (Adopting a prefetched module decompressed, compressed again and wrote its SPIR-V: half of an adoption,
// ~0.1 s of the prologue walk's slow frames at 16 GB.)
vk::ShaderModule AddFrom(uint64_t file, uint64_t offset, uint32_t stored, uint32_t words, bool compressed);
[[nodiscard]] bool Holds(vk::ShaderModule module);
// Its SPIR-V (null when it cannot be read). A released module's still reads (a worker may make a pipeline of it).
std::shared_ptr<const std::vector<uint32_t>> Load(vk::ShaderModule module);
void                                         Release(vk::ShaderModule module);
struct Statistics {
	uint64_t modules = 0, spirv_bytes = 0, stored_bytes = 0, loads = 0, load_misses = 0;
};
Statistics Stats();

} // namespace Libs::Graphics::ModuleStore

namespace Libs::Graphics {
// A module CreateModule made: the store's or the driver's.
void DestroyShaderModule(vk::Device device, vk::ShaderModule module);
} // namespace Libs::Graphics
