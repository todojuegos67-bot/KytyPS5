#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_HELPER_INLINE_POLICY_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_HELPER_INLINE_POLICY_H_

#include <cstdint>

// When the SPIR-V memory helpers (get_bda_pointer, load_indirect_buffer_*,
// indirect_buffer_format_geometry, convert_indirect_buffer_format) are marked DontInline.
//
// Inlined at every call site, the helpers grow some Ghost of Yotei compute shaders 4-5x
// (435K -> 2.0M words) and NVIDIA needed up to 141 s per pipeline. With DontInline everywhere
// (ytest14) compiles were fast but the game crashed inside a driver DLL ~30 s in, so DontInline is
// now used only where the estimated inlined expansion is at least KYTY_SPIRV_DONTINLINE_WORDS
// (default 200000; 0 = never), and never for a guest shader listed in
// _PipelineCache/dontinline-blacklist.txt (written by the crash handler when a driver compile of
// such a module faults).
namespace Libs::Graphics::ShaderRecompiler::HelperInlinePolicy {

inline constexpr const char* kBlacklistPath = "_PipelineCache/dontinline-blacklist.txt";

uint64_t ThresholdWords();
bool     GuestBlacklisted(uint64_t guest_hash);
// Mixed into the translation-cache key so a policy change re-translates the affected shaders.
uint64_t CacheKey(uint64_t guest_hash);

} // namespace Libs::Graphics::ShaderRecompiler::HelperInlinePolicy

#endif
