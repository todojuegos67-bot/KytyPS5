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
// with 32-wide subgroups it runs one GCN lane per invocation. `reason`: why not.
bool IsWaveLaneLocal(const Decoder::Program& program, std::string* reason = nullptr);

// What makes a wave64 program lane-local (LaneLocal.cpp), for a 64-thread group whose two halves cooperate
// (`cooperative`): the lane reads the wave runs through workgroup memory and the lane-mask branches the wave decides
// (by pc).
struct LaneLocalProof {
	std::vector<uint32_t> exchange_reads;
	std::vector<uint32_t> wave_branches;
	std::vector<uint32_t> lds_barriers; // (TranslatorFeatureLaneLocalWide) PCs a workgroup barrier goes before
	[[nodiscard]] bool    Cooperative() const { return !exchange_reads.empty() || !wave_branches.empty(); }
};
bool ProveWaveLaneLocal(const Decoder::Program& program, bool cooperative, LaneLocalProof& proof,
                        std::string* reason = nullptr);

// Translation features that change the SPIR-V of shaders a static pipeline store holds pipelines for (bits). A store
// made before one keeps its pipelines in use only while the translation leaves that feature out: no update makes a
// player compile pipelines again (the static precompile stamps its store with them: PipelineCache).
constexpr uint32_t TranslatorFeatureCooperativeHalves = 1u; // LaneLocal.cpp's cooperative halves (9935d547)
constexpr uint32_t TranslatorFeatureTableSlotsAtUse   = 2u; // a table mode slot loaded where the shader reads it
// V_MIN/MAX/MIN3/MAX3_F32 as GLSL NMin/NMax, which AMD GPUs execute with the guest GPU's own NaN and signed-zero rules
// (v_min_f32/v_max_f32), in place of an exact emulation (about a dozen instructions each; upstream PR #1761).
constexpr uint32_t TranslatorFeatureNativeFloatMinMax = 4u;
// Vector register writes under EXEC as their values where no use observes the lanes outside it (IR::EliminateExecSelects:
// RTX 5090 at 1-1, 4K: GPU -3%; the heaviest compute shaders 4-14% fewer instructions).
constexpr uint32_t TranslatorFeatureExecSelects = 8u;
// Address loads from one base in a block share the first one's page walk where they stay in its page (LoadBda). Measured
// on the Steam Deck only: in the Steam Deck build's default set; elsewhere KYTY_BDA_GROUP_WALKS=1.
constexpr uint32_t TranslatorFeatureBdaGroupWalks = 16u;
// Pixel shaders' DPP quad_perm moves read by quad operations, defined on helper pixels where a shuffle is not (the
// derivatives at triangle edges: the debug menu font's lines, PR #243); with the GPU's quad operations only.
constexpr uint32_t TranslatorFeaturePixelQuadReads = 32u;
// V_MAD/V_MAC/V_MADMK/V_MADAK_F32 as the hardware runs them, the product rounded before the add (FPMad32; only
// V_FMA_F32 is fused). Fused, the last bits differ where vectors nearly cancel: Demon's Souls' fog particles on the
// loading screen picked a wrong colour and flashed brown patches for one frame.
constexpr uint32_t TranslatorFeatureUnfusedMad = 64u;
// The lane-local proof's wider rules (LaneLocal.cpp), one bit for all of them. So far: S_GETPC_B64 admitted, the program
// counter being the same in every lane and both halves (the wave64 programs that read it use it for a table of
// constants after their code: s_getpc, s_add_u32 of the table's offset, per-lane loads through it); S_SETPC and the
// subvector loops stay refused; LDS plain reads and writes, a workgroup barrier placed between those another lane may
// meet in one barrier-free stretch (LdsBarriers: the two halves of a lane-local wave are two subgroups, not in lockstep);
// DS_*_ADDTID through the wave's lane (IR WaveLaneId), a lane's own slot of a 64-dword window whose base (M0 a constant
// on every path) the proof knows, so two of one window or of disjoint windows need no barrier; DPP whose row mask treats
// the halves apart refused. Off unless KYTY_LANE_LOCAL_WIDE=1 (not in TranslatorFeatures until measured: it
// changes those programs' SPIR-V).
constexpr uint32_t TranslatorFeatureLaneLocalWide = 128u;
// Pixel shaders that read the barycentrics (the I/J the GCN pixel shader starts with in v0/v1) without any per-vertex
// input take them from VK_AMD_shader_explicit_vertex_parameter (BaryCoordSmoothAMD/BaryCoordNoPerspAMD, I and J), not
// VK_KHR_fragment_shader_barycentric. Only on AMD's Windows driver with its older compiler ("AMD proprietary shader
// compiler": GCN cards such as Vega and Polaris), which refused every pipeline with such a pixel shader although it
// reports the KHR feature (issue #305: 7 of Demon's Souls' 173, the picture's post-processing among them: black screen).
// No other GPU may use it (TranslatorDeviceFeatures): their SPIR-V is unchanged.
constexpr uint32_t TranslatorFeatureAmdBarycentrics = 256u;
constexpr uint32_t TranslatorFeatures = TranslatorFeatureCooperativeHalves | TranslatorFeatureTableSlotsAtUse |
                                        TranslatorFeatureNativeFloatMinMax | TranslatorFeatureExecSelects |
                                        TranslatorFeaturePixelQuadReads | TranslatorFeatureUnfusedMad |
                                        TranslatorFeatureAmdBarycentrics
#if defined(KYTY_STEAM_DECK)
                                        | TranslatorFeatureBdaGroupWalks
#endif
    ;
// Those a store made without stays up to date for (no precompile offered for them alone): the next precompile, made
// for another reason (a new graphics driver, new seeds), brings them; worth less than a precompile's time.
constexpr uint32_t TranslatorFeaturesWithNextPrecompile = TranslatorFeatureExecSelects | TranslatorFeatureBdaGroupWalks |
                                                          TranslatorFeaturePixelQuadReads | TranslatorFeatureUnfusedMad;
// Those that change nearly every shader's SPIR-V: the next precompile of a store made without one compiles all of it
// again, not only what is new (PipelineCache::StaticCacheRedoesAll; the launchers say so). The unfused multiply-adds
// change 19,787 of the 1.07 seeds' 20,043 programs.
constexpr uint32_t TranslatorFeaturesBroad = TranslatorFeatureExecSelects | TranslatorFeatureUnfusedMad;
// Those the translation uses (default: all; the emulator sets those its static store was made with before it translates
// anything). KYTY_LANE_LOCAL_COOP=0/1 overrides the cooperative halves either way, KYTY_TABLE_SLOTS_AT_USE=0/1 the
// table slots' loads, KYTY_NATIVE_FLOAT_MIN_MAX=0/1 the float min/max, KYTY_EXEC_SELECTS=0/1 the EXEC selects,
// KYTY_BDA_GROUP_WALKS=0/1 the shared page walks, KYTY_PIXEL_QUAD_READS=0/1 the pixel shaders' quad reads,
// KYTY_UNFUSED_MAD=0/1 the multiply-adds, KYTY_AMD_BARYCENTRICS=0/1 the AMD barycentrics (1 only to look at that
// translation elsewhere: a GPU without VK_AMD_shader_explicit_vertex_parameter refuses its pipelines).
void                   SetTranslatorFeatures(uint32_t features);
[[nodiscard]] uint32_t TranslatorFeatureSet();
// Those this GPU's translation may use (default: all) and those it uses whatever the store was made with (`required`,
// default: none), set with the device before anything is translated: TranslatorFeatureSet() holds no others (but those
// KYTY_LANE_LOCAL_COOP=1 / KYTY_TABLE_SLOTS_AT_USE=1 / KYTY_NATIVE_FLOAT_MIN_MAX=1 force), and a static store made
// without the others is up to date.
void                   SetTranslatorDeviceFeatures(uint32_t features, uint32_t required = 0);
[[nodiscard]] uint32_t TranslatorDeviceFeatures();

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
