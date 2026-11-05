#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPILE_GUARD_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPILE_GUARD_H_

#include <vulkan/vulkan_core.h>

#include <cstdint>

// Per-thread "what is the driver compiling right now" markers, read by the crash handler.
// A Scope is opened right before vkCreate*Pipelines and closed after it returns. When a host
// exception hits while a marker is open, the crash report names the pipeline key, the guest
// shader and the SPIR-V size, and the guest shader is appended to the DontInline blacklist
// (HelperInlinePolicy) so the next run translates it with inlined helpers.
namespace Libs::Graphics::CompileGuard {

// Remembers which guest shader a runtime shader module was translated from.
void NoteModule(VkShaderModule module, uint64_t guest_hash, uint64_t words);

class Scope {
public:
	// module may be null; guest_hash 0 = look it up from the module (NoteModule).
	Scope(const char* what, uint64_t key, VkShaderModule module, uint64_t guest_hash, uint64_t words);
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	int m_slot = -1;
};

// Crash handler only: prints every open marker (no locks, no allocation) and appends the guest
// shaders of the markers to the blacklist. Returns the number of open markers.
int ReportAndBlacklist();

// Crash handler only: "name.dll+0x1234" for a host code address, or an empty string.
// Writes into out (size bytes).
void DescribeAddress(uint64_t address, char* out, unsigned size);

} // namespace Libs::Graphics::CompileGuard

#endif
