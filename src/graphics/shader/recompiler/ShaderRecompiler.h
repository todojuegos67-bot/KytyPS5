#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shader.h"

#include <functional>
#include <array>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	bool                        non_fatal = false;
	bool                        bindless_images = false;
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        early_dump                 = false;
	bool                        emit_debug_info            = false;
	bool                        maintenance8_enabled       = false;
	const char*                 dump_label                 = nullptr;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	ShaderStageInputInfo        input_info;
	// Research: reads a subroutine the shader calls (S_SWAPPC_B64 through a pointer in user
	// data) from guest memory, as many words as are readable up to a bound. Unset, such calls
	// stay unsupported.
	std::function<std::vector<uint32_t>(uint64_t address)> read_code;
};

struct TranslateResult {
	bool        unsupported = false;
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
	// Research: user-data dwords (indices into CompileOptions::user_data) that held the targets
	// of inlined calls. The program is valid only while they hold the same values.
	std::vector<uint32_t> call_target_user_data;
	// A pixel shader with POPS ordering (IR::ShaderInfo::uses_pops).
	bool uses_pops = false;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

// Decoded source and typed scalar call queries are retained once per cache source.
struct ShaderSource {
	struct Call {
		uint32_t instruction;
		std::array<IR::Value, 2> target;
	};
	struct Linked {
		std::vector<uint32_t> code;
		Decoder::Program decoded;
		std::vector<uint32_t> observed_function;
	};
	std::vector<uint32_t> code;
	Decoder::Program decoded;
	IR::ResourcePlan call_targets;
	std::optional<Call> call;
	std::optional<Linked> linked;
	uint64_t revision = 0;
	std::vector<std::pair<uint64_t, uint64_t>> reads;
};

[[nodiscard]] ShaderSource PrepareShaderSource(std::span<const uint32_t> code,
	                                           const CompileOptions& options);
const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime);
[[nodiscard]] TranslateResult TranslateProgram(const Decoder::Program& decoded,
	                                           const CompileOptions& options);

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
