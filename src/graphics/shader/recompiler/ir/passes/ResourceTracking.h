#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Collects immutable resource topology from typed SSA handles, interns their resolved dwords in
// descriptor_sources, then writes dense indices to handle flags and MemoryInfo.
// `storage_reads` (a pixel shader with Frontend::TranslatorFeaturePixelInterlock): the loads of an image the shader also
// stores through the same descriptor read the storage image (ImageResource::read of a storage image).
void TrackResources(Program& program, bool storage_reads = false);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_ */
