#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <span>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;
class SrtTraceRecorder;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);
// The words of [address, address + words * 4) where they lie, all on one 4 KiB page, or nullptr when the reader can't
// vouch for them (SrtReadLog::StillReads).
using SrtMemoryPeek = const uint32_t* (*)(void* userdata, uint64_t address, size_t words);

// Image descriptors at base + first + k * step, k < count, within size bytes that a sampled
// root selects by GPU-computed byte offset. The host keeps them resident instead of
// materializing them.
struct ImageHeapRequest {
	uint64_t             base  = 0;
	uint64_t             size  = 0;
	uint64_t             step  = 0;
	uint64_t             first = 0;
	uint64_t             count = 0;
	const ImageResource* root  = nullptr;
};

struct ImageHeapView {
	uint64_t                         slots      = 0; // Device address of one slot per heap dword.
	uint32_t                         slot_count = 0; // Keys past it select a trailing null slot.
	std::span<const DescriptorValue> kinds; // One representative descriptor per kind.
};

class ImageHeapSource {
public:
	// False when the host does not keep the table resident.
	virtual bool Acquire(const ImageHeapRequest& request, ImageHeapView& view) = 0;

protected:
	~ImageHeapSource() = default;
};

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	uint64_t                  indirect_args_address = 0;
	ImageHeapSource*          image_heap            = nullptr;
	// The host has a guest image heap (--bindless): a large descriptor table's images are read by slot.
	bool                      guest_image_heap           = false;
	// Compute: the workgroup size, which with workgroup_counts bounds the invocation IDs.
	std::span<const uint32_t> workgroup_size;
	// A plain guest load gives what both readers give (the GPU thread: a GPU-written page faults and is
	// downloaded, as the strict reader downloads it), so MaterializeResources may run the x86 walker.
	bool                      direct_reads = false;
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

// Every guest read of one materialization, with its outcome and words, so the next call can tell
// without walking again whether its result would be the same.
class SrtReadLog {
public:
	// Past this many words a log gives up (64 KB): a program that reads a whole descriptor heap (1.3 MB a draw in the bar
	// title) never repeated, and copying its reads only cost time.
	static constexpr size_t MaxWords = 16384u;
	// The ahead scanner's limit (4 MB): an effect program's image table (618 KB at Marvel's Wolverine's Leap Attack
	// fight) is read whole on every draw, and a take, which compares in place, costs a quarter of materializing it again.
	static constexpr size_t MaxAheadWords = size_t {1} << 20u;
	// The runtime to materialize with; it reads through this log, which keeps up to max_words words. The x86 walker
	// loads guest memory itself: RecordWalk keeps what its walk gave.
	SrtRuntime Recording(const SrtRuntime& source, size_t max_words = MaxWords);
	// After the materialization: when it took the plan's x86 walk, that walk's results (its outputs,
	// reached blocks, flat buffer and loaded addresses).
	void RecordWalk(const ResourcePlan& plan, std::span<const uint32_t> flat);
	// True when the log is complete (it gives up past MaxWords), the runtime equals the logged one,
	// every read gives the same outcome and words again, and a recorded walk walks the same again.
	// When it is false after walking, walked (if non-null) gets that walk's flat buffer, valid until the
	// next call: the plan's scratch holds the rest of it, for MaterializeResources to take.
	bool Repeats(const SrtRuntime& runtime, const ResourcePlan& plan,
	             const std::vector<uint32_t>** walked = nullptr) const;
	void Clear() { m_valid = false; }
	[[nodiscard]] size_t Words() const { return m_words.size(); }
	// Gives the log up and frees what it holds.
	void Release() {
		m_valid = false;
		std::vector<uint32_t>().swap(m_words);
		std::vector<Entry>().swap(m_reads);
	}
	// A log recorded on another thread (the ahead scanner's): true when it is complete, every logged read succeeded
	// and read, a reader that refuses what it cannot vouch for, gives the same words again. With peek, each read is
	// proved page by page where its words lie, without copying them (read isn't called).
	bool StillReads(SrtMemoryReader read, void* userdata, SrtMemoryPeek peek = nullptr) const;
	// After StillReads: the log as if recorded through runtime, for the repeat checks of the thread that took it.
	void Retarget(const SrtRuntime& runtime);

private:
	struct Entry {
		uint64_t address;
		uint32_t first; // in m_words
		uint32_t count;
		bool     strict;
		bool     ok;
	};
	static bool ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool Read(const SrtRuntime& via, bool strict, uint64_t address, std::span<uint32_t> values);
	bool        Record(bool strict, uint64_t address, std::span<uint32_t> values);

	SrtRuntime            m_source; // its spans are the caller's: compared through the copies below
	std::vector<uint32_t> m_user_data;
	std::vector<uint32_t> m_workgroup_counts;
	std::vector<uint32_t> m_workgroup_size;
	std::vector<Entry>    m_reads;
	std::vector<uint32_t> m_words;
	size_t                m_max_words = MaxWords;
	bool                  m_valid = false;
	// The recorded walk: scratch outputs and reach bytes, then the loaded addresses; its flat buffer.
	std::vector<uint64_t>                              m_walk;
	std::vector<uint32_t>                              m_walk_flat;
	bool                                               m_walked = false;
	mutable std::vector<uint32_t>                      m_check_flat;
};

// Retained reads no longer need the guest instruction PC. A clean read evaluates
// its address and bounds through the strict reader as well as the final DWORD.
struct SrtReadFlags {
	uint32_t index = 0;
	uint32_t clean = 0;
};

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uniform across a compute wave, including values derived from its workgroup ID.
// This does not imply that the host can evaluate the value.
bool ValidateWaveUniformValue(const ResourcePlan& program, Value value);
// Compiles the plan's x86 walker (upstream d7856753e). Without values it walks the plan's resources;
// otherwise it only evaluates the values, with every block reachable.
void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values = {});
// Runs the walker. flat receives the scalar reads of reachable blocks and reads, when non-null,
// one DWORD range per guest load. False when a reachable read is unknown or a load faulted outside
// the null page region.
bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads);
// Appends the last walk's loads to reads, one DWORD range each, as WalkSrt does.
void WalkedLoads(const ResourcePlan& plan, std::vector<std::pair<uint64_t, uint64_t>>& reads);
// Results of the last walk; false when unknown (or not a walker output).
bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result);
bool WalkedDescriptor(const ResourcePlan& plan, uint32_t source, DescriptorValue& result);
bool WalkedActive(const ResourcePlan& plan, uint32_t source);
bool WalkedImageActive(const ResourcePlan& plan, uint32_t image);
bool WalkedReached(const ResourcePlan& plan, uint32_t block);
// The value's output index in the plan's walker, UINT32_MAX when it is none.
uint32_t WalkOutput(const ResourcePlan& plan, const Inst& inst);
void     SetWalkOutput(const ResourcePlan& plan, const Inst& inst, uint32_t index);
// Length of the guest load at rip if it belongs to the walker running on this thread, else 0.
// The exception handler completes such a load as a zero read (bdf48dcf3) and reports its address.
uint32_t SrtWalkerLoadLength(uint64_t rip);
void     SrtWalkerFaultedLoad(uint64_t address);
// A walk without a reader reads guest words by pointer. On a page a shader wrote, that read faults, and the fault reads
// the page back: a submit and a wait for all GPU work. Most such reads are of words no shader wrote (Marvel's
// Wolverine's GPU-driven tables: 68-100 % of the GPU thread's faults), which are current in the backing. The renderer's
// reader serves those (true) and declines everything else (false: the pointer read and its fault, as before).
using SrtCleanReader = bool (*)(uint64_t address, std::span<uint32_t> values);
void SetSrtCleanReader(SrtCleanReader reader);
// The exception handler, for a faulting walker load (SrtWalkerLoadLength, SrtPointerLoadLength): the word from the
// clean reader, if it serves it.
bool SrtWalkerCleanLoad(uint64_t address, uint32_t& value);
// Length of the interpreter's guest load (mov eax, [r10]) if rip is it, else 0. Unlike the walker's loads, a fault
// there that the clean reader does not serve stays the fault it is (a read back, or a crash).
uint32_t SrtPointerLoadLength(uint64_t rip);

// The x86 walk against the interpreter's refresh of the same plan and runtime (tests). The interpreter refreshes
// first, with a clean evaluator, then the walker walks.
struct WalkComparison {
	bool        walked    = false; // WalkSrt succeeded
	bool        refreshed = false; // the interpreter's refresh succeeded
	uint32_t    mismatches = 0;    // both know it and it differs: flat buffer, reach, activity, a descriptor or value
	uint32_t    walker_only = 0;   // only the walker knows a descriptor or value, or only it walked
	uint32_t    interpreter_only = 0; // only the interpreter knows it: MaterializeResources asks the interpreter
	uint32_t    reach_smaller = 0; // blocks the walker leaves unreached that the interpreter reaches (counted apart)
	std::string first;             // the first difference
};
WalkComparison CompareWalk(const ResourcePlan& plan, const SrtRuntime& runtime);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

// Nonnegative affine coefficients: element zero is constant, remaining elements are
// caller-selected leaves. Runtime range proofs must additionally exclude U32 wrap.
template <size_t Size>
std::optional<std::array<uint64_t, Size>> AffineIndex(
    Value value, const auto& resolve, const auto& leaf, uint32_t depth = 0) {
	value = resolve(value);
	if (depth > 32u || value.GetType() != Type::U32) return {};
	std::array<uint64_t, Size> result {};
	if (value.IsImmediate()) {
		result[0] = value.U32();
		return result;
	}
	if (const auto index = leaf(value)) {
		result[*index] = 1u;
		return result;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) return {};
	auto left = AffineIndex<Size>(inst->Arg(0), resolve, leaf, depth + 1u);
	auto right = AffineIndex<Size>(inst->Arg(1), resolve, leaf, depth + 1u);
	if (!left || !right) return {};
	const auto constant = [](const auto& coefficients) {
		return std::ranges::all_of(std::span(coefficients).subspan(1u),
		                          [](uint64_t coefficient) { return coefficient == 0u; });
	};
	if (op == ValueOpcode::IMul32 && !constant(*right)) std::swap(left, right);
	if (op != ValueOpcode::IAdd32 && !constant(*right)) return {};
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32u) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (size_t index = 0; index < Size; ++index) {
		(*left)[index] = op == ValueOpcode::IAdd32
		    ? (*left)[index] + (*right)[index] : (*left)[index] * (*right)[0];
		if ((*left)[index] > UINT32_MAX) return {};
	}
	return left;
}

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          SrtWalker* clean_evaluator = nullptr);
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);
	// Takes the plan's last x86 walk (WalkSrt) as this session's refresh: its flat buffer, reached
	// blocks and active sources. Its walked values and descriptors then answer Evaluate and
	// EvaluateDescriptor; the rest is interpreted with the walk's reached blocks.
	void AdoptWalk(const std::vector<uint32_t>& flat);

private:
	friend class SrtTraceRecorder;
	SrtWalker(SrtWalker& parent, Value active_mask, bool clean = false,
	          SrtWalker* clean_evaluator = nullptr);
	ResourcePlan::EvaluationContext& AcquireContext();
	void ResetUnknowns();
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);
	bool Interpret(std::vector<uint32_t>& flat);
	bool Refresh(std::vector<uint32_t>& flat);
	bool Fetch(const ResourcePlan::DescriptorFetch& fetch, uint32_t& result) const;
	bool Replay(std::vector<uint32_t>& flat, const SrtRuntime& runtime);
	// Recording hooks (no-ops unless this refresh is recorded): how the current instruction's
	// result follows from its operands, and a choice made on the operand just evaluated.
	void Form(uint8_t form, uint32_t imm = 0);
	void Check(bool zero, bool low32 = false);
	void CheckEqual(bool equal);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	const std::vector<uint8_t>*      m_reachable_blocks = nullptr;
	const SrtWalker*                m_session;
	uint8_t                         m_read_policy;
	mutable uint32_t                m_session_contexts = 0; // a root walker: the contexts its session holds
	ResourcePlan::EvaluationContext& m_context;
	SrtTraceRecorder*               m_record = nullptr;
	bool                            m_used   = false; // it evaluated something: its memo is not fresh
	// After a refresh: the flattened SRT it made, which EvaluateDescriptor fetches from. The clean evaluator of the
	// refreshing walker gets it too, for the clean reads only (its memo holds those, not the others).
	const std::vector<uint32_t>*     m_flat            = nullptr;
	bool                            m_flat_clean_only = false;
	bool                            m_walked          = false; // AdoptWalk: the plan's scratch has the walk
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
