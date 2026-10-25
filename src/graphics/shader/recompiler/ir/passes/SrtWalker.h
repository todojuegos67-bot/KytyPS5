#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <cstring>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	// read_specialization_memory succeeds for an aligned, nonzero 64-byte block only when every
	// dword in it would succeed with the same value, so one refresh may read whole blocks.
	bool                      specialization_block_reads = false;
	// Userdata for read_specialization_memory; null means userdata.
	void*                     specialization_userdata    = nullptr;
	// Optional: reads an aligned, nonzero 64-byte block for read_memory (with userdata) only when
	// reading each of its dwords with read_memory would give the same values; false otherwise.
	SrtMemoryReader           read_memory_block          = nullptr;
	std::span<const uint32_t> workgroup_counts;
};

class SrtReadCapture {
public:
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges):
	    m_source(source), m_ranges(ranges) {}
	SrtRuntime ObservedRuntime();

private:
	static bool ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
};

// Retained reads no longer need the guest instruction PC. A clean read evaluates
// its address and bounds through the strict reader as well as the final DWORD.
struct SrtReadFlags {
	uint32_t index = 0;
	uint32_t clean = 0;
};

// The direct copy used when a runtime has no ordinary reader. Nothing is mapped below 64 KiB, so a
// zero or tiny SRT base (an unbound pair, KytyPS5#1268) reads zero instead of faulting.
inline void SrtCopyHostMemory(uint64_t address, std::span<uint32_t> values) {
	constexpr uint64_t NullRegionEnd = 0x10000;
	if (address < NullRegionEnd || address + values.size_bytes() < address) [[unlikely]] {
		std::memset(values.data(), 0, values.size_bytes());
		return;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
}

[[nodiscard]] inline void* SpecializationUserdata(const SrtRuntime& runtime) {
	return runtime.specialization_userdata != nullptr ? runtime.specialization_userdata
	                                                  : runtime.userdata;
}

// True while an activity walk evaluates a control-flow condition. A specialization reader may
// then fail instead of waiting for the GPU: the walk follows every successor, and the shader,
// which makes the same loads, takes the branch itself.
[[nodiscard]] bool EvaluatingCondition() noexcept;

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          SrtWalker* clean_evaluator = nullptr, Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Walks the blocks the plan's conditions reach: the descriptor sources they use (an empty
	// span means that all sources are active) and the scalar reads they make.
	std::span<const uint8_t> FindActiveSources();
	// Refreshes the reachable scalar reads (after the activity walk, which it makes if the
	// caller has not); the others read as zero.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);

private:
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	ResourcePlan::EvaluationContext& m_context;
	// FindActiveSources has run for this refresh.
	bool                             m_activity_found = false;
};

// Whether plans compiled from now on read consecutive flat SRT slots as runs (the default).
void SetFlatRunReads(bool enabled);

// Resolves a plan's descriptor, SRT, condition and fill values into index-based nodes. The plan
// must not be modified afterwards.
const CompiledResourcePlan& CompileResourcePlan(const ResourcePlan& program);

// SrtWalker over a CompiledResourcePlan. It keeps SrtWalker's evaluation contexts, clean
// predicates, EXEC-mask selects and failure semantics, so both produce identical results.
class SrtEvaluator {
public:
	SrtEvaluator(const ResourcePlan& program, const CompiledResourcePlan& compiled,
	             const SrtRuntime& runtime, bool clean_flat_slots = false,
	             SrtEvaluator* clean_evaluator = nullptr,
	             uint32_t      active_mask     = ResourceNode::NoNode);
	~SrtEvaluator();
	SrtEvaluator(const SrtEvaluator&)            = delete;
	SrtEvaluator& operator=(const SrtEvaluator&) = delete;

	bool Evaluate(uint32_t node, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// An empty span means that all sources are active.
	std::span<const uint8_t> FindActiveSources();
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	bool EvaluateWide(uint32_t node, uint64_t& result);
	bool EvaluateInst(const ResourceNode& node, uint64_t& result);
	bool EvaluateRawRead(const ResourceNode& node, uint64_t& result);
	// Reads a run's dwords into their flat slots; false leaves them unchanged.
	bool ReadFlatRun(const CompiledResourcePlan::FlatRun& run, std::vector<uint32_t>& flat);

	const ResourcePlan&              m_program;
	const CompiledResourcePlan&      m_compiled;
	const ResourceNode*              m_nodes;
	SrtRuntime                       m_runtime;
	bool                             m_clean_flat_slots = false;
	SrtEvaluator*                    m_clean_evaluator  = nullptr;
	uint32_t                         m_active_mask      = ResourceNode::NoNode;
	ResourcePlan::EvaluationContext& m_context;
	ResourcePlan::EvaluationContext::Entry* m_memo;
	uint64_t                         m_generation;
	// Set by a successful RefreshFlatBuffer: this walker's slot values.
	const std::vector<uint32_t>*     m_flat = nullptr;
	// FindActiveSources has run for this refresh.
	bool                             m_activity_found = false;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
