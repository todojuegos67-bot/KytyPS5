#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERFUNCTIONS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERFUNCTIONS_H_

#include <compare>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

using ShaderCodeReader = std::function<bool(uint64_t, std::span<uint32_t>)>;

// The s_trap code the expansion emits for a call target outside the captured set. Guest
// s_trap instructions are no-ops, as on retail hardware; only this code reports a trap.
constexpr uint32_t ShaderCallMissTrapCode = 0x7fu;

// The expansion's miss trap names the scalar pair holding the missed target in bits 8-14 of its
// immediate: simm16 = ShaderCallMissTrapCode | ((target_reg + 1) << 8). Zero there means the trap
// does not name one. The trap reports s[target_reg .. target_reg + 3]: the target and the pair
// after it (a specialized variant's data). With ShaderCallMissDropFlag the trap also drops the
// lanes that reach it: they leave EXEC for the rest of the wave, and no later write of EXEC
// brings them back (a partitioned expansion, ShaderCallPartition).
constexpr uint32_t ShaderCallMissDropFlag = 0x8000u;
constexpr uint32_t ShaderCallMissTrapImmediate(uint32_t target_reg, bool drop = false) {
	return ShaderCallMissTrapCode | (((target_reg + 1u) & 0x7fu) << 8u) |
	       (drop ? ShaderCallMissDropFlag : 0u);
}

// Call-miss feedback. A table can hold more targets than one expansion can inline (GT7's material
// tables hold hundreds), so a table call over the bound inlines the targets the GPU reported
// missing (ReportShaderCallMiss, from the fault buffer) and traps on the rest. Reports gather for
// a short while before the next expansion takes them, to batch the recompiles.
// Returns true the first time the target is reported for the shader.
// A table call target, with the two table words the call passes along when the expansion
// specializes on them (zero otherwise).
struct ShaderCallVariant {
	uint64_t target = 0;
	uint32_t data0  = 0;
	uint32_t data1  = 0;
	auto     operator<=>(const ShaderCallVariant&) const = default;
};
// A specialized target whose data keeps changing (GT7 allocates its race material records anew
// each frame) never matches a learned variant again: past a bound it is dropped from the
// expansion, left to the miss path, and no longer reported. churned is set when this report
// dropped it.
bool ReportShaderCallMiss(uint64_t shader_hash, const ShaderCallVariant& variant,
                          bool* churned = nullptr);
// The variants reported for the shader so far (ascending), or null.
std::shared_ptr<const std::vector<ShaderCallVariant>> ShaderCallMissTargets(uint64_t shader_hash);
// Whether a table call whose records carry a data pointer inlines each target once and leaves
// the pointer to the shader (KYTY_CALL_RUNTIME_RECORDS, on by default) rather than one variant
// specialized on each record. Tests switch it.
bool ShaderCallRuntimeRecords();
void SetShaderCallRuntimeRecords(bool value);

// A material function with a loop is inlined (round 95's default; KYTY_CALL_LOOPS=0 leaves it to
// the miss path; see InlineLoopingVariants in ShaderFunctions.cpp for why it was not before), with
// its loops bounded: the expansion brackets its body with these two S_NOPs (the immediates real
// code never uses), and the translation leaves every loop with all its code between them at its
// header once the invocation has run ShaderCallLoopCap() rounds of such loops (KYTY_CALL_LOOP_CAP,
// 1024 by default; 0 leaves them unbounded; BudgetLoopHeaders in Translate.cpp). A count read from
// data the emulator does not reproduce (a placeholder texture, say) then costs a bounded number
// of rounds instead of hanging the GPU. Tests switch the inlining.
bool               ShaderCallLoopsInlined();
void               SetShaderCallLoopsInlined(bool value);
constexpr uint32_t ShaderCallLoopBodyBegin = 0xbf80c0deu;
constexpr uint32_t ShaderCallLoopBodyEnd   = 0xbf80c0dfu;
uint32_t           ShaderCallLoopCap();
// A ray tracing shader (one with IMAGE_BVH_INTERSECT_RAY) runs this many rounds of its loops per
// invocation, all its loops together, and then leaves each loop at its header
// (KYTY_RAY_LOOP_CAP, 16384 by default; 0 leaves them unbounded; BudgetLoopHeaders in
// Translate.cpp): its traversal follows node data, and a node list the emulator reads as zero
// (memory it does not have) never ends. Its calls are its hit functions, which return at once
// unless KYTY_RAY_HIT_CALLS=1 (RayHitCallsReturn in ShaderFunctions.cpp). Round 101: 16384 (4096
// before). The garage's reflection traversal walks the BVH once for a whole wave of 64 rays, and
// with 4096 its waves stopped short in many tiles of 16x8 rays (the blocks in the reflections).
// Its waves count their rounds (RayLoopSlot), and the host logs them ("Ray loops:").
uint32_t ShaderRayLoopCap();

// Partitioned expansions (round 99; KYTY_CALL_PARTITIONS=0 turns them off): a shader whose one
// call is a material table of runtime records is expanded once per target, each expansion
// inlining that target alone, and dispatched once per expansion. The lanes that call another
// target leave the wave at the call (ShaderCallMissDropFlag), so each lane runs its material in
// exactly one of the dispatches, and the shader's stores are those of one dispatch with every
// target inlined. The driver's compile time grows faster than linearly with a pipeline's size:
// an expansion with every target of a garage material pass (9 million SPIR-V words) did not
// finish compiling in ten minutes, one with one target takes seconds. Only shaders whose code
// stays correct with lanes dropped there are partitioned (no LDS sharing, barriers, lane reads
// or atomics outside the inlined bodies: PartitionableShader in ShaderFunctions.cpp).
struct ShaderCallPartition {
	// In: the target whose expansion to build; 0 for the first one.
	uint64_t target = 0;
	// Out: the targets of the shader's partitions (ascending), each one dispatch; one zero entry
	// when the shader is partitioned but has no target yet (one dispatch whose lanes all leave at
	// the call, reporting their targets); empty when the shader is not partitioned.
	std::vector<uint64_t> targets;
	// Out (round 118): the target the expansion runs (the one asked for, or the first), which its
	// dispatch passes to the shader (GetPartitionTarget); 0 when it runs none.
	uint64_t built = 0;
	// Out (round 118): how many targets the shader has partitions for, before
	// ShaderCallPartitionIdle left out the ones not reported lately (targets.size() otherwise).
	size_t known = 0;
};
bool ShaderCallPartitions();
void SetShaderCallPartitions(bool value);
// Round 118: a partition's expansion compares the call's target with the partition's target as
// the dispatch passes it in, not as a literal: S_CMP_EQ_U32 with these scalar source codes (249
// and 250, reserved in a scalar instruction's source fields; the decoder reads them as the
// partition target's low and high words, the translation as GetPartitionTarget). The same
// material function at another address then expands to the same code, one program and one
// pipeline. KYTY_PARTITION_RUNTIME_TARGET=0 compares with the address, as before. Tests switch it.
constexpr uint32_t ShaderCallPartitionTargetLow  = 249u;
constexpr uint32_t ShaderCallPartitionTargetHigh = 250u;
bool               ShaderCallPartitionRuntimeTarget();
void               SetShaderCallPartitionRuntimeTarget(bool value);
// Round 118: a window in milliseconds (0, the default: none). With one, a partitioned shader runs
// only the partitions of the targets its dispatches reported in it (ReportShaderCallMiss: the
// lanes of each target in view leave every other partition's dispatch at the call and report it,
// every frame), and always one, the one reported last: its known targets grow with every material
// the game loaded, and each partition's dispatch runs the shader's code before the call for every
// lane. The pipeline cache sets it (KYTY_PARTITION_IDLE_MS).
void     SetShaderCallPartitionIdle(uint32_t milliseconds);
uint32_t ShaderCallPartitionIdle();
// Diagnostics (shader_batch_tool --partition-check): why the code outside a shader's call keeps
// it from being partitioned, or empty when it would be (its call permitting).
std::string ShaderCallPartitionCheck(std::span<const uint32_t> code);

// Round rtexp: canonical expansions (KYTY_CALL_CANONICAL, on by default): a table call's inlined
// variants in ascending order, and a material table call of runtime records inlining the union of
// the reported targets the shader's tables offered so far in the run (see CanonicalSwitch in
// ShaderFunctions.cpp). Tests switch it.
bool ShaderCallCanonical();
void SetShaderCallCanonical(bool value);
// For the log: the table call targets known (reported in this run or read from earlier runs'
// call_targets.txt), the shaders they belong to, and how many times a call site's union grew.
void ShaderCallKnownTargets(uint64_t& targets, uint64_t& shaders, uint64_t& union_growths);

// Expands scalar calls through immutable pointers or bounded scalar-buffer tables.
// Table calls retain runtime selection and trap on targets outside the captured set.
// Empty output means no calls; unresolved calls never discard guest instructions.
// consulted_user_data, when given, receives the user-data registers the result depends on;
// together with the code and the memory read through `read`, they determine the result.
bool InlineShaderFunctions(std::span<const uint32_t> code, uint64_t base,
                           std::span<const uint32_t> user_data, const ShaderCodeReader& read,
                           std::vector<uint32_t>& expanded, std::string& reason,
                           uint32_t wave_size = 64,
                           std::vector<uint32_t>* consulted_user_data = nullptr);

// InlineShaderFunctions for a caller that expands the same shaders over and over (the pipeline
// cache, once per dispatch). The decoded shader is kept while its code is unchanged, and an
// expansion while the call targets resolve to the same addresses and the callee code reads the
// same; only the target resolution runs every time. The results equal InlineShaderFunctions'.
class ShaderFunctionExpander {
public:
	ShaderFunctionExpander();
	~ShaderFunctionExpander();
	ShaderFunctionExpander(const ShaderFunctionExpander&)            = delete;
	ShaderFunctionExpander& operator=(const ShaderFunctionExpander&) = delete;

	bool Expand(std::span<const uint32_t> code, uint64_t base, std::span<const uint32_t> user_data,
	            const ShaderCodeReader& read, std::vector<uint32_t>& expanded, std::string& reason,
	            uint32_t wave_size);
	// With the shader's hash: table calls over the bound inline the targets reported missing for
	// it (ShaderCallMissTargets), and so do the material table calls of runtime records, whatever
	// their size: the expansion then depends on the targets the shader runs, not on the table,
	// and a scene with new materials rebuilds only the shaders that call them. summary, when
	// given, receives a line about a new expansion that leaves table targets out (empty
	// otherwise).
	// partition, when given: the shader's partitioned expansion for partition->target, if it is
	// partitioned (partition->targets says so); the expansion with every target otherwise.
	bool Expand(std::span<const uint32_t> code, uint64_t base, std::span<const uint32_t> user_data,
	            const ShaderCodeReader& read, std::vector<uint32_t>& expanded, std::string& reason,
	            uint32_t wave_size, uint64_t shader_hash, std::string* summary,
	            ShaderCallPartition* partition = nullptr);
	// The translation of an expansion this expander produced failed at `pc` (bytes into the
	// expanded code): the table target whose inlined body holds pc is left out of later
	// expansions, of every shader, and takes the miss path instead. Returns that target, or 0
	// when pc is in no inlined table body.
	uint64_t ExcludeFailedCallTarget(std::span<const uint32_t> expanded, uint32_t pc);
	// A table target an earlier run excluded (ExcludeFailedCallTarget, kept between runs by the
	// pipeline cache): left out of later expansions from now on. False when it already was.
	bool ExcludeCallTarget(uint64_t target);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics::ShaderRecompiler::Decoder

#endif
