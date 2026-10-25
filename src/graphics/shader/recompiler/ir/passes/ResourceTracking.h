#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <stdexcept>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Thrown when a shader uses a resource pattern the tracker cannot express (for example a store
// or atomic through a descriptor the GPU selects at runtime). The pipeline cache catches it and
// skips that shader's draws/dispatches. KYTY_FATAL_UNSUPPORTED_SHADERS=1 restores the old exit.
struct UnsupportedResources: std::runtime_error {
	explicit UnsupportedResources(const std::string& what): std::runtime_error(what) {}
};

// Resolves native descriptor sources, plans their scalar reads, and assigns dense resource bindings.
void TrackResources(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg,
                    ShaderStageInputInfo input = {});

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_ */
