#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, uint32_t* value);
using SrtMemorySync   = bool (*)(void* userdata, uint64_t address, uint64_t size);
// A rejected probe performs no read/sync; the original scalar sequence follows.
using SrtMemorySpan = bool (*)(void*, uint64_t, uint32_t*, uint32_t count, bool clean);
// The end of the guest mapping an address lies in (0: none).
using SrtMappingEnd = uint64_t (*)(void*, uint64_t address);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	SrtMemorySync             sync_memory                = nullptr;
	SrtMemorySpan             try_read_memory_span       = nullptr;
	SrtMappingEnd             mapping_end                = nullptr; // (address probes: MaterializeResources)
};

enum class RuntimeValueType { Any, Integer };

// Collects reachable ReadConst values. Immediate offsets receive compact flat-buffer slots;
// dynamic offsets remain explicit and are never assigned a fake slot.
void BuildSrtPlan(Program& program);
void BuildLinearSrtPlan(ResourcePlan& program);
bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any,
                          std::string* reason = nullptr);
bool EvaluateUniformValues(const ResourcePlan& program, std::span<const Value> values,
                            const SrtRuntime& runtime, std::span<uint32_t> results);

bool EvaluateDescriptorSource(const ResourcePlan& program, uint32_t source,
                              const SrtRuntime& runtime, DescriptorValue& result);

// Evaluates one runtime snapshot transactionally. Scalar values and ReadConst results shared by
// several descriptors are memoized once across the batch.
bool EvaluateDescriptorSources(const ResourcePlan& program, std::span<const uint32_t> sources,
                               const SrtRuntime& runtime, std::vector<DescriptorValue>& results);

// Evaluates potentially reachable descriptor sources and the flattened immediate SRT with one
// memoized scalar walk. Inactive descriptors are zero; on failure no destination is changed.
bool EvaluateRuntimeSources(const ResourcePlan& program, std::span<const uint32_t> sources,
                            const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                            std::vector<uint32_t>& flat, std::span<const uint8_t> clean_flat_slots,
                            std::vector<uint8_t>& active_sources);

bool WalkSrt(const ResourcePlan& program, const SrtRuntime& runtime,
             std::vector<uint32_t>& flat);

// Every guest dword one linear SRT evaluation reads, split by use (native XPR
// records). A `data` read reaches only the flattened SRT, unchanged: its flat
// positions are plain copies of its address. Every other read - one feeding an
// address, a descriptor dword or a computed value, or a clean read - is
// `structural`, with the value it had. False when the program is not evaluated
// by a single linear plan (no plan, control-flow variants, a failed read).
struct SrtReadTrace {
	// Why a read is structural (bits): it feeds another read's address (`Pointer`: as the low or high
	// half of a pointer read's base, `Address`: otherwise), a descriptor dword, a computed value, or it
	// is a control-flow predicate input.
	enum Use : uint8_t { Address = 1, Descriptor = 2, Computed = 4, Predicate = 8, Pointer = 16 };
	// What a pointer read's address is based on: user data pair `root` (user[k] | user[k + 1] << 32)
	// or the values of the structural reads `low` and `high`, plus an offset that does not depend on it.
	static constexpr uint8_t  NoRoot = 0xff;
	static constexpr uint32_t NoBase = UINT32_MAX;
	struct Base {
		uint8_t  root = NoRoot;
		uint32_t low = NoBase, high = NoBase;
	};
	struct Read {
		uint64_t address = 0;
		uint32_t value   = 0;
		bool     clean   = false;
		uint8_t  use     = 0;
		Base     base;
	};
	// Structural reads copied unchanged into a descriptor dword of the snapshot.
	enum class Kind : uint8_t { Buffer, Image, Sampler };
	struct Feed {
		uint32_t read  = 0; // index into structural
		Kind     kind  = Kind::Buffer;
		uint32_t index = 0; // resource index in the snapshot
		uint32_t dword = 0;
	};
	std::vector<std::pair<uint32_t, uint64_t>> data;       // (flat index, address)
	std::vector<Base>                          data_bases; // per data read
	bool                                       unrooted = false; // reads without their roots (the interpreter's)
	std::vector<Read>                          structural;
	std::vector<Feed>                          feeds;
	// Structural reads that are also copied unchanged into the flattened SRT:
	// (index into structural, flat index).
	std::vector<std::pair<uint32_t, uint32_t>> flat_feeds;
	size_t                                     flat_words = 0;
};
bool TraceLinearSrtReads(const ResourcePlan& program, const SrtRuntime& runtime, SrtReadTrace& trace);
// How the SRT evaluation uses user data (bit k: user[k]): the words it may read, and the pairs
// (user[k], user[k + 1]) its linear plan uses for nothing but the bases of its pointer reads (an
// evaluation with another value there reads the same offsets from the new base). Every word and
// no pair for a plan with control-flow variants, a uniform fill or indirect images.
struct SrtUserDataUse {
	uint32_t read     = UINT32_MAX;
	uint32_t pointers = 0;
};
SrtUserDataUse UserDataUse(const ResourcePlan& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
