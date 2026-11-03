#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TRANSLATIONCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TRANSLATIONCACHE_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::TranslationCache {

// Persistent guest->SPIR-V translation cache (_PipelineCache/<title>.xlat).
// Key: 128-bit hash of everything a permutation's SPIR-V depends on (guest code bytes, stage static key, resource specialization, push-data cursor, called function code,
// device features and the recompiler's environment switches), salted with the build's git
// revision, so any change to the translator invalidates the whole file. Value: SPIR-V plus the
// CompiledShaderInfo (reflection/bindings) that CompileProgram would have produced.
// KYTY_TRANSLATION_CACHE=0 disables it.
struct Key {
	uint64_t a = 0, b = 0;
	bool operator==(const Key&) const = default;
};

struct Entry {
	std::vector<uint32_t>                    spirv;
	ShaderRecompiler::IR::CompiledShaderInfo info;
};

class KeyBuilder {
public:
	KeyBuilder();
	~KeyBuilder();
	void Add(const void* data, size_t bytes);
	template <typename T>
	void Value(const T& v) {
		Add(&v, sizeof(v));
	}
	void Words(std::span<const uint32_t> w) {
		Value(static_cast<uint64_t>(w.size()));
		Add(w.data(), w.size_bytes());
	}
	[[nodiscard]] Key Finish();

private:
	void* m_state = nullptr;
};

bool Enabled();
// Starts loading the file in the background (called once, at pipeline cache init).
void Open(const std::filesystem::path& path);
// Waits for the load if it is still running. Thread-safe.
[[nodiscard]] std::optional<Entry> Find(const Key& key);
// Queues the record for the background writer. Thread-safe.
void Store(const Key& key, std::span<const uint32_t> spirv,
           const ShaderRecompiler::IR::CompiledShaderInfo& info);
// Prints hit/miss/store counters at most every 5 s.
void Report();

} // namespace Libs::Graphics::TranslationCache

#endif
