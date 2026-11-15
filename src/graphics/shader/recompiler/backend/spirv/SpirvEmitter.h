#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// Host device capabilities and user options that change the emitted SPIR-V. Built once per device.
struct Profile {
	bool maintenance8         = false; // Dynamic texel offsets on image samples.
	bool native_float_min_max = false; // NMin/NMax give the guest's NaN and signed-zero results.
	bool robust_buffers       = false; // robustBufferAccess2: a storage-buffer read past its range is 0.
	// robustStorageBufferAccessSizeAlignment (1 or 4): the bound range is rounded up to it for the
	// check, so an 8- or 16-bit read is 0 past the range only when it is no larger than the element.
	uint32_t robust_buffer_alignment = 4;
	// OpReadClockKHR Device tick in picoseconds; 0 without shaderDeviceClock.
	uint32_t device_clock_period_ps = 0;
	// Compute pipelines can require full subgroups of exactly 32 invocations (computeFullSubgroups,
	// subgroup size 32 fixed). A wave64 program can then run one guest lane per invocation, its
	// wave over two subgroups (ShaderRecompiler.cpp, WaveSplitProblem).
	bool wave64_split = false;
	// 8 and 8_8 sRGB textures sample through sRGB views; without one the shader decodes them.
	bool r8_srgb_views   = false;
	bool r8g8_srgb_views = false;
	// the buffer table variant (ResourceSpecialization::buffer_table), set per program.
	bool buffer_table = false;
	// Native float ops: NaN, signed-zero and denormal results may differ from the guest.
	bool relaxed_float = false;
};

inline constexpr Profile DefaultProfile {};

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t       push_data_start_dword = 0,
                                  const Profile& profile               = DefaultProfile,
                                  bool           emit_debug_info       = false,
                                  uint32_t       descriptor_set        = 0);

// A coherent (GLC) buffer load is followed by a device-scope acquire barrier, so that a load polling for another
// workgroup's write sees it. Without one, AMD's drivers keep serving the cached value; NVIDIA reloads anyway. The
// device layer enables it on every vendor but NVIDIA.
void SetCoherentLoadAcquire(bool enabled);
bool GetCoherentLoadAcquire();

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
