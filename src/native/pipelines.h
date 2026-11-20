#pragma once
// (f5) Pipeline / material system: the interface (design; no implementation in this build).
//
// Today the pipeline of a draw is found per draw from the registers the PM4 left (memos of pair / variant / state,
// td14+15 0.37 ms p50) and the shaders are the translated ones (AOT cache). Natively a pipeline is created once per
// (material, pass, target formats) when the material is bound for the first time and found by a key the game already
// hands its draw loop: the material bind 0xa84de0 (f5p1: per material change inside 0xad8bc0).
//
// Shaders: the translated SPIR-V of the game's shaders is reused as is (render1 §2: the XPR depth shaders read
// matrices / vertices by BDA from R, no rewrite needed). The cache key of users' shader caches is not changed (rule:
// never make users recompile). New shaders are only the passes' glue (copies, comparisons).
//
// Verify: a native pass's pipeline must produce the emulated one's output for the same inputs (verify.h); the state
// (blend, depth test / write, cull, formats) is read from the emulated path the first time and compared afterwards.
#include <cstdint>

namespace NativeRender {

struct MaterialKey {
	uint64_t material  = 0; // the game's material pointer as passed to 0xa84de0 (stable while loaded)
	uint32_t pass      = 0; // pass-registry id
	uint32_t targets   = 0; // hash of the target formats / sample count
	[[nodiscard]] bool operator==(const MaterialKey& o) const {
		return material == o.material && pass == o.pass && targets == o.targets;
	}
};

struct PipelineRef {
	uint64_t pipeline = 0; // VkPipeline
	uint64_t layout   = 0; // VkPipelineLayout
	bool     ready    = false; // compiled (never block the GPU thread on a compile: the pass falls back that frame)
};

class PipelineSystem {
public:
	virtual ~PipelineSystem() = default;
	// The material was loaded / bound for the first time: create (asynchronously) its pipelines for the passes that
	// draw it, from the emulated path's state for that key (the first emulated draw is the template).
	virtual void OnMaterial(const MaterialKey& key, const void* emulated_state, uint32_t state_size) = 0;
	[[nodiscard]] virtual PipelineRef Find(const MaterialKey& key) const = 0;
	// The material was unloaded (its memory freed by the game): drop its pipelines.
	virtual void Forget(uint64_t material) = 0;
};

} // namespace NativeRender
