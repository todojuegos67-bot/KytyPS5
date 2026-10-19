#pragma once

#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

struct EmbeddedFetchLoad {
	uint32_t              pc         = 0;
	int                   attrib_id  = -1;
	uint32_t              components = 0;
	std::vector<uint32_t> prolog_loads;
};

struct EmbeddedFetchPlan {
	std::vector<EmbeddedFetchLoad> loads;
	int32_t                        vertex_offset_sgpr   = -1;
	int32_t                        instance_offset_sgpr = -1;
};

struct TranslateOptions {
	ShaderType                    stage               = ShaderType::Unknown;
	uint32_t                      wave_size           = 64;
	uint64_t                      shader_hash         = 0;
	uint32_t                      user_data_base      = 0;
	uint32_t                      user_data_count     = 64;
	uint32_t                      scratch_dwords      = 0;
	bool                          dispatcher_fallback = false;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string_view              fallback_reason;
	const ShaderVertexInputInfo*  vertex         = nullptr;
	const ShaderPixelInputInfo*   pixel          = nullptr;
	const ShaderComputeInputInfo* compute        = nullptr;
	const EmbeddedFetchPlan*      embedded_fetch = nullptr;
};

IR::Program TranslateProgram(const Decoder::Program& decoded, const CFG::Graph& cfg,
                             const TranslateOptions& options);

// A wave64 program whose results in each lane depend on no lane outside its 32-lane half (LaneLocal.cpp): on a host
// with 32-wide subgroups it runs one GCN lane per invocation. `reason`: why not; `uniform_loads`: whether it took a buffer
// load at one address for a value the same in every lane (KYTY_LANE_LOCAL_UNIFORM_LOADS).
bool IsWaveLaneLocal(const Decoder::Program& program, std::string* reason = nullptr, bool* uniform_loads = nullptr);

// What makes a wave64 program lane-local (LaneLocal.cpp): taken buffer loads at one address (uniform_loads), or, for a
// 64-thread group whose two halves cooperate (`cooperative`), the lane reads the wave runs through workgroup memory and
// the lane-mask branches the wave decides (by pc).
struct LaneLocalProof {
	bool                  uniform_loads = false;
	std::vector<uint32_t> exchange_reads;
	std::vector<uint32_t> wave_branches;
	[[nodiscard]] bool    Cooperative() const { return !exchange_reads.empty() || !wave_branches.empty(); }
};
bool ProveWaveLaneLocal(const Decoder::Program& program, bool cooperative, LaneLocalProof& proof,
                        std::string* reason = nullptr);

// Translation features that change the SPIR-V of shaders a static pipeline store holds pipelines for (bits). A store
// made before one keeps its pipelines in use only while the translation leaves that feature out: no update makes a
// player compile pipelines again (the static precompile stamps its store with them: PipelineCache).
constexpr uint32_t TranslatorFeatureCooperativeHalves = 1u; // LaneLocal.cpp's cooperative halves (9935d547)
constexpr uint32_t TranslatorFeatureTableSlotsAtUse   = 2u; // a table mode slot loaded where the shader reads it
constexpr uint32_t TranslatorFeatures = TranslatorFeatureCooperativeHalves | TranslatorFeatureTableSlotsAtUse;
// Those the translation uses (default: all; the emulator sets those its static store was made with before it translates
// anything). KYTY_LANE_LOCAL_COOP=0/1 overrides the cooperative halves either way, KYTY_TABLE_SLOTS_AT_USE=0/1 the
// table slots' loads.
void                   SetTranslatorFeatures(uint32_t features);
[[nodiscard]] uint32_t TranslatorFeatureSet();
// Those this GPU's translation may use (default: all) and those it uses whatever the store was made with (`required`,
// default: none), set with the device before anything is translated: TranslatorFeatureSet() holds no others (but those
// KYTY_LANE_LOCAL_COOP=1 / KYTY_TABLE_SLOTS_AT_USE=1 force), and a static store made without the others is up to date.
void                   SetTranslatorDeviceFeatures(uint32_t features, uint32_t required = 0);
[[nodiscard]] uint32_t TranslatorDeviceFeatures();

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
