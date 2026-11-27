#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ShaderInfoOptions {
	const ShaderVertexInputInfo*  vertex  = nullptr;
	const ShaderPixelInputInfo*   pixel   = nullptr;
	const ShaderComputeInputInfo* compute = nullptr;
};

// Completes the immutable shader interface after resource tracking. On failure Program::info and
// all completion state remain unchanged.
void CollectShaderInfo(Program& program, const ShaderInfoOptions& options);

// (pascal-compat) The device has fragmentShaderBarycentric (default: yes). Without it (NVIDIA Pascal / Maxwell, older
// AMD) no pixel input is per-vertex and no gl_BaryCoord is declared: v_interp_mov P0 reads the interpolated value and
// P10 / P20 read 0, so the game's own P0 + I*P10 + J*P20 gives the interpolated value; raw I / J read 0. Set with the
// device (vulkanWindow.cpp), before any program compiles; with the feature nothing changes.
void               SetDeviceFragmentBarycentric(bool supported);
[[nodiscard]] bool DeviceFragmentBarycentric();

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_ */
