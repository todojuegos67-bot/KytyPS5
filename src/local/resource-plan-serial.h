#pragma once

// A resource plan (IR::ResourcePlan: the descriptor/SRT value graph a program's draws materialize) as bytes and
// back, for plans kept on disk until they are needed: the shader prefetch translates every program of the game, and
// its 20K plans took ~2.5 GB of RAM (Inst nodes with their own vectors). Its linear SRT plan is rebuilt on loading.
// The bytes are this build's: never written for another emulator build to read.

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ResourcePlanSerial {

// False (out unchanged) for a plan with a value the format does not hold.
bool Save(const Libs::Graphics::ShaderRecompiler::IR::ResourcePlan& plan, std::vector<uint8_t>& out);
// False for malformed bytes; `plan` is then empty. Its linear SRT plan is rebuilt (IR::BuildLinearSrtPlan, as
// ExtractResourcePlan ends), or with `defer_linear` left to the caller: *defer_linear tells whether it had one.
bool Load(std::span<const uint8_t> bytes, Libs::Graphics::ShaderRecompiler::IR::ResourcePlan& plan,
          bool* defer_linear = nullptr);

// (ds-pc-port37) A translated module's specialization and compiled info (IR::CompiledShaderInfo: BDA read plan, table
// plan, ShaderInfo, binding layout), for the shader prefetch's disk store. The same rules: this build's bytes only.
bool SaveModule(const Libs::Graphics::ShaderRecompiler::IR::ResourceSpecialization& specialization,
                const Libs::Graphics::ShaderRecompiler::IR::CompiledShaderInfo& info, std::vector<uint8_t>& out);
bool LoadModule(std::span<const uint8_t> bytes, Libs::Graphics::ShaderRecompiler::IR::ResourceSpecialization& specialization,
                Libs::Graphics::ShaderRecompiler::IR::CompiledShaderInfo& info);
// A translated module's infos (what the shader prefetch keeps of it besides its SPIR-V) and the specialization it was
// compiled with, appended to `out`; false (out unchanged) when the bytes would not load back the same. Read back at
// `at` (advanced past them); false for malformed bytes. This build's bytes too.
bool SaveModule(const Libs::Graphics::ShaderRecompiler::IR::CompiledShaderInfo&     info,
                const Libs::Graphics::ShaderRecompiler::IR::ResourceSpecialization& specialization, std::vector<uint8_t>& out);
bool LoadModule(std::span<const uint8_t> bytes, size_t& at, Libs::Graphics::ShaderRecompiler::IR::CompiledShaderInfo& info,
                Libs::Graphics::ShaderRecompiler::IR::ResourceSpecialization& specialization);

} // namespace ResourcePlanSerial
