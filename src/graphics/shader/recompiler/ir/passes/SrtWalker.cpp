#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abi.h"
#include "common/assert.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <fmt/format.h>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Lean's first walker read a pointer below 64 KB (and its levels below) as zero tables (c527c28ba); a load
// there through such a pointer stays a zero read. Any other fault fails the walk: the readers fail there.
constexpr uint64_t NullPointerLoads = 0x20000;

namespace {

std::atomic<SrtCleanReader> g_clean_reader = nullptr;
// An interpreter load the exception handler completed as 0 (SrtPointerFaultedLoad): its read fails.
thread_local bool t_pointer_faulted = false;

// The interpreter's guest loads, as the x86 walker's: mov eax, [r10] (41 8b 02), which the exception handler completes
// from the clean reader when it faults (SrtPointerLoadLength). No check costs anything on a page no shader wrote.
class PointerLoad: public Xbyak::CodeGenerator {
public:
	using Function = uint32_t(KYTY_SYSV_ABI*)(uint64_t address);
	PointerLoad(): Xbyak::CodeGenerator(64) {
		mov(r10, rdi);
		load = getSize();
		mov(eax, dword[r10]);
		ret();
		ready();
		function = getCode<Function>();
	}
	size_t   load     = 0;
	Function function = nullptr;
};

const PointerLoad& PointerLoader() {
	static const PointerLoad loader;
	return loader;
}

// A walk's read by pointer (no reader). Up to 16 words, or any read below NullPointerLoads, load one by one through
// PointerLoad, whose faults on unmapped memory read zero (false past NullPointerLoads, as the x86 walker's); a longer
// read (a whole descriptor table) is copied, and a fault there reads its page back, as before.
[[nodiscard]] bool ReadByPointer(uint64_t address, std::span<uint32_t> values) {
	if (values.size() > 16u && address >= NullPointerLoads) {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
		return true;
	}
	const auto load   = PointerLoader().function;
	t_pointer_faulted = false;
	for (size_t i = 0; i < values.size(); i++) values[i] = load(address + 4u * i);
	return !t_pointer_faulted;
}

} // namespace

void SetSrtCleanReader(SrtCleanReader reader) {
	(void)PointerLoader(); // built here, not in the exception handler
	g_clean_reader.store(reader, std::memory_order_relaxed);
}

bool SrtWalkerCleanLoad(uint64_t address, uint32_t& value) {
	const auto reader = g_clean_reader.load(std::memory_order_relaxed);
	return reader != nullptr && reader(address, {&value, 1});
}

void SrtPointerFaultedLoad(uint64_t address) {
	if (address >= NullPointerLoads) t_pointer_faulted = true;
}

uint32_t SrtPointerLoadLength(uint64_t rip) {
	const auto& loader = PointerLoader();
	return rip == reinterpret_cast<uint64_t>(loader.getCode()) + loader.load ? 3u : 0u;
}

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? ReadStrict : nullptr;
	runtime.read_memory = ReadOrdinary;
	return runtime;
}

bool SrtReadCapture::ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool SrtReadCapture::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (capture.m_source.read_memory != nullptr) {
		if (!capture.m_source.read_memory(capture.m_source.userdata, address, values)) return false;
	} else if (!ReadByPointer(address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

SrtRuntime SrtReadLog::Recording(const SrtRuntime& source, size_t max_words) {
	m_max_words = max_words;
	m_source = source;
	m_user_data.assign(source.user_data.begin(), source.user_data.end());
	m_workgroup_counts.assign(source.workgroup_counts.begin(), source.workgroup_counts.end());
	m_workgroup_size.assign(source.workgroup_size.begin(), source.workgroup_size.end());
	m_reads.clear();
	m_words.clear();
	m_valid = true;
	auto runtime = source;
	runtime.userdata = this;
	runtime.read_specialization_memory = source.read_specialization_memory != nullptr ? ReadStrict : nullptr;
	runtime.read_memory = ReadOrdinary;
	return runtime;
}

bool SrtReadLog::Read(const SrtRuntime& via, bool strict, uint64_t address, std::span<uint32_t> values) {
	if (strict) return via.read_specialization_memory(via.userdata, address, values);
	if (via.read_memory != nullptr) return via.read_memory(via.userdata, address, values);
	return ReadByPointer(address, values);
}

bool SrtReadLog::Record(bool strict, uint64_t address, std::span<uint32_t> values) {
	const bool ok = Read(m_source, strict, address, values);
	if (m_valid && m_words.size() + values.size() > m_max_words) m_valid = false;
	if (!m_valid) return ok;
	// A successful read that goes on where the last one ended, of the same kind and on the same 4 KiB page (the page
	// cache's, GuestPageReadCache), extends it: read again whole it gives the same outcome and words as in parts. At
	// Marvel's Wolverine's Trask outpost a take's log was about 76 reads of one word each, three in four of them going
	// on from the last, and the GPU thread proved them one by one (3.8 ms a frame).
	if (ok && !values.empty() && !m_reads.empty()) {
		auto&          last = m_reads.back();
		constexpr auto page = [](uint64_t at) { return at >> 12u; };
		if (last.ok && last.strict == strict && last.first + last.count == m_words.size() &&
		    address == last.address + uint64_t {last.count} * 4u &&
		    page(last.address) == page(address + values.size_bytes() - 1u)) {
			last.count += static_cast<uint32_t>(values.size());
			m_words.insert(m_words.end(), values.begin(), values.end());
			return ok;
		}
	}
	// A failed read decides the result as much as a successful one.
	m_reads.push_back({address, static_cast<uint32_t>(m_words.size()), static_cast<uint32_t>(values.size()),
	                   strict, ok});
	if (ok) m_words.insert(m_words.end(), values.begin(), values.end());
	return ok;
}

bool SrtReadLog::ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return static_cast<SrtReadLog*>(userdata)->Record(true, address, values);
}

bool SrtReadLog::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return static_cast<SrtReadLog*>(userdata)->Record(false, address, values);
}

void SrtReadLog::RecordWalk(const ResourcePlan& plan, std::span<const uint32_t> flat) {
	m_walked = plan.srt_walked;
	if (!m_walked) return;
	const auto* scratch = plan.srt_scratch.data();
	const auto  reach   = (plan.control_flow.size() + 7u) / 8u;
	m_walk.assign(scratch, scratch + plan.srt_outputs / 8u);
	m_walk.resize(m_walk.size() + reach, 0u);
	std::memcpy(m_walk.data() + plan.srt_outputs / 8u, reinterpret_cast<const uint8_t*>(scratch) + plan.srt_reach,
	            plan.control_flow.size());
	m_walk.insert(m_walk.end(), scratch + plan.srt_captures / 8u, scratch + plan.srt_captures / 8u + plan.srt_loads);
	// MaterializeResources appends indirect mappings after the walk's words.
	m_walk_flat.assign(flat.begin(), flat.begin() + std::min(flat.size(), plan.srt_reads.size()));
	m_walk_fills = plan.srt_fills;
}

namespace {

// A log's reads are a few words each, thousands of times a frame: no allocation per call. Not cleared: a read that
// succeeds fills what it's given, and one that fails isn't compared.
class ReadWords {
public:
	[[nodiscard]] std::span<uint32_t> Get(uint32_t count) {
		if (count <= m_small.size()) return {m_small.data(), count};
		m_large.resize(count);
		return m_large;
	}

private:
	std::array<uint32_t, 64> m_small;
	std::vector<uint32_t>    m_large;
};

} // namespace

bool SrtReadLog::Repeats(const SrtRuntime& runtime, const ResourcePlan& plan,
                         const std::vector<uint32_t>** walked) const {
	if (walked != nullptr) *walked = nullptr;
	if (!m_valid || runtime.shader_base != m_source.shader_base ||
	    runtime.indirect_args_address != m_source.indirect_args_address ||
	    runtime.image_heap != m_source.image_heap || runtime.guest_image_heap != m_source.guest_image_heap ||
	    runtime.read_memory != m_source.read_memory ||
	    runtime.read_specialization_memory != m_source.read_specialization_memory ||
	    runtime.userdata != m_source.userdata || !std::ranges::equal(runtime.user_data, m_user_data) ||
	    !std::ranges::equal(runtime.workgroup_counts, m_workgroup_counts) ||
	    !std::ranges::equal(runtime.workgroup_size, m_workgroup_size))
		return false;
	if (m_walked) {
		// The walk reads first, as in MaterializeResources; what it gave decides the reads after it.
		if (!runtime.direct_reads || !WalkSrt(plan, runtime, m_check_flat, nullptr)) return false;
		if (walked != nullptr) *walked = &m_check_flat;
		// A word left to the GPU reads 0: the same flat buffer repeats only with the same words left to the GPU.
		if (m_check_flat != m_walk_flat || plan.srt_fills != m_walk_fills) return false;
		const auto* scratch  = plan.srt_scratch.data();
		const auto  outputs  = plan.srt_outputs / 8u;
		const auto  reach    = (plan.control_flow.size() + 7u) / 8u;
		const auto* captures = scratch + plan.srt_captures / 8u;
		if (m_walk.size() != outputs + reach + plan.srt_loads || !std::equal(scratch, scratch + outputs, m_walk.data()) ||
		    std::memcmp(reinterpret_cast<const uint8_t*>(scratch) + plan.srt_reach, m_walk.data() + outputs,
		                plan.control_flow.size()) != 0 ||
		    !std::equal(captures, captures + plan.srt_loads, m_walk.data() + outputs + reach))
			return false;
	}
	ReadWords buffer;
	for (const auto& read: m_reads) {
		const auto words = buffer.Get(read.count);
		if (Read(runtime, read.strict, read.address, words) != read.ok ||
		    (read.ok && !std::equal(words.begin(), words.end(), m_words.begin() + read.first)))
			return false;
	}
	return true;
}

bool SrtReadLog::StillReads(SrtMemoryReader read, void* userdata, SrtMemoryPeek peek) const {
	if (!m_valid || m_walked) return false;
	if (peek != nullptr) {
		for (const auto& entry: m_reads) {
			if (!entry.ok) return false;
			uint64_t address = entry.address;
			for (size_t done = 0; done < entry.count;) {
				const size_t n = std::min<size_t>(entry.count - done, std::max<uint64_t>((4096u - (address & 4095u)) / 4u, 1u));
				const uint32_t* words = peek(userdata, address, n);
				if (words == nullptr || std::memcmp(words, m_words.data() + entry.first + done, n * 4u) != 0) return false;
				done += n;
				address += uint64_t {n} * 4u;
			}
		}
		return true;
	}
	ReadWords buffer;
	for (const auto& entry: m_reads) {
		const auto words = buffer.Get(entry.count);
		if (!entry.ok || !read(userdata, entry.address, words) ||
		    !std::equal(words.begin(), words.end(), m_words.begin() + entry.first))
			return false;
	}
	return true;
}

void SrtReadLog::Retarget(const SrtRuntime& runtime) {
	m_source = runtime;
	m_user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	m_workgroup_counts.assign(runtime.workgroup_counts.begin(), runtime.workgroup_counts.end());
	m_workgroup_size.assign(runtime.workgroup_size.begin(), runtime.workgroup_size.end());
}

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

// Replay traces: what an op of a trace computes (ResourcePlan::TraceOp::kind), and how the
// interpreter told the recorder an instruction's result follows from its operands.
enum TraceKind : uint8_t {
	TracePure,       // ApplyPureOp over its args
	TracePass,       // its arg
	TraceFixed,      // the recorded result: the checks before it fix it
	TraceUserData,   // user data word imm
	TraceShaderBase,
	TraceRawRead,    // ReadRawSlot; flags bit 0: through the clean runtime
	TraceCheckZero,  // abandon unless (arg == 0) == flags
	TraceCheckEqual, // abandon unless (arg 0 == arg 1) == flags
	TraceCondition,  // a block condition: abandon unless it succeeds (ok) and is nonzero (flags) as recorded
	TraceFlat,       // flat[imm] = arg
};
enum TraceForm : uint8_t {
	FormNone, // a pure opcode with its operands in order, or a failure with no operand
	FormPure, // ApplyPureOp over the operands in order
	FormPureSwapped,
	FormPassFirst,
	FormPassLast,
	FormFixed,
	FormUserData,
	FormShaderBase,
	FormRawRead, // imm: memory index, bit 31 when the first operand is the guard
	FormExtract64,
};
constexpr int32_t  TraceFailed    = INT32_MIN;
constexpr uint32_t TraceMaxOps    = 1u << 16u;
constexpr uint32_t RawReadGuarded = 1u << 31u;

// The operand count of an opcode whose result is a function of all its operands.
uint32_t PureArity(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::LogicalNot: return 1;
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract: return 3;
		case ValueOpcode::BitFieldInsert: return 4;
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMax32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::CompositeConstructU64: return 2;
		default: return 0;
	}
}

float F32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

// The interpreter's arithmetic once an instruction's operands are known; false when the result is
// not a value the host can trust. flush: FPCompareFlags::flush_input_denorms.
bool ApplyPureOp(ValueOpcode op, bool flush, const uint64_t* args, uint64_t& result) {
	const auto a = args[0];
	const auto b = args[1];
	const auto c = args[2];
	const auto d = args[3];
	switch (op) {
		case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); return true;
		case ValueOpcode::IAdd64: result = a + b; return true;
		case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); return true;
		case ValueOpcode::ISub64: result = a - b; return true;
		case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); return true;
		case ValueOpcode::UMulHi:
			result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
			return true;
		case ValueOpcode::UMin32: result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b)); return true;
		case ValueOpcode::UMax32: result = std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b)); return true;
		case ValueOpcode::SMax32:
			result = static_cast<uint32_t>(std::max(static_cast<int32_t>(a), static_cast<int32_t>(b)));
			return true;
		case ValueOpcode::ConvertF32U32:
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::ConvertU32F32: {
			// The conversion saturates and maps NaN to zero.
			const auto value = F32(a);
			result           = std::isnan(value) || value <= 0.0f ? 0u
			                   : value >= 4294967296.0f ? UINT32_MAX
			                                            : static_cast<uint32_t>(value);
			return true;
		}
		case ValueOpcode::ConvertS32F32: {
			// The conversion saturates and maps NaN to zero.
			const auto value = F32(a);
			result           = std::isnan(value) ? 0u
			                   : value >= 2147483648.0f ? static_cast<uint32_t>(INT32_MAX)
			                   : value <= -2147483648.0f
			                       ? static_cast<uint32_t>(INT32_MIN)
			                       : static_cast<uint32_t>(static_cast<int32_t>(value));
			return true;
		}
		case ValueOpcode::FPMul32: result = std::bit_cast<uint32_t>(F32(a) * F32(b)); return true;
		case ValueOpcode::FPTrunc32: result = std::bit_cast<uint32_t>(std::trunc(F32(a))); return true;
		case ValueOpcode::FPRecipIFlag32: {
			const auto exponent = (a >> 23u) & 0xffu;
			// Normal positive powers of two have exact normal reciprocals in every FP mode.
			if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
			result = (254u - exponent) << 23u;
			return true;
		}
		case ValueOpcode::FPIsNan32: result = std::isnan(F32(a)); return true;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32: {
			const auto operand = [&](uint64_t bits) {
				if (flush && (bits & 0x7fffffffu) < 0x00800000u) {
					bits &= 0x80000000u;
				}
				return F32(bits);
			};
			result = op == ValueOpcode::FPOrdLessThanEqual32 ? operand(a) <= operand(b) : operand(a) >= operand(b);
			return true;
		}
		case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); return true;
		case ValueOpcode::BitwiseAnd64: result = a & b; return true;
		case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); return true;
		case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); return true;
		case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); return true;
		case ValueOpcode::ShiftLeftLogical32: result = static_cast<uint32_t>(a) << (b & 31u); return true;
		case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); return true;
		case ValueOpcode::ShiftRightLogical32: result = static_cast<uint32_t>(a) >> (b & 31u); return true;
		case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); return true;
		case ValueOpcode::ShiftRightArithmetic32:
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
			return true;
		case ValueOpcode::ShiftRightArithmetic64:
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case ValueOpcode::BitFieldUExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case ValueOpcode::BitFieldSExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
				bits |= ~mask;
			}
			result = bits;
			return true;
		}
		case ValueOpcode::BitFieldInsert: {
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::IEqual32: result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b); return true;
		case ValueOpcode::INotEqual32: result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b); return true;
		case ValueOpcode::IEqual64: result = a == b; return true;
		case ValueOpcode::INotEqual64: result = a != b; return true;
		case ValueOpcode::ULessThan32: result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b); return true;
		case ValueOpcode::ULessThanEqual32: result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b); return true;
		case ValueOpcode::UGreaterThan32: result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b); return true;
		case ValueOpcode::SGreaterThanEqual32:
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >= std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); return true;
		case ValueOpcode::LogicalNot: result = a == 0u; return true;
		case ValueOpcode::CompositeConstructU64:
			result = static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::CompositeExtractU64: result = static_cast<uint32_t>(a >> (b * 32u)); return true;
		default: return false;
	}
}

// A raw read once its handle's words (low, high, and for buffers records and word3) and offset are
// known: the address, the bounds a buffer read keeps to, and the read itself.
bool ReadRawSlot(const MemoryInfo& mem, ValueOpcode op, const uint64_t* args, const SrtRuntime& runtime,
                 uint64_t& result) {
	const bool vector    = op == ValueOpcode::LoadBufferU32;
	const auto low       = args[0];
	const auto high      = args[1];
	const auto offset    = args[2];
	const auto records   = args[3];
	const auto word3     = args[4];
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (op == ValueOpcode::ReadConstBuffer || vector) {
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	const auto reader = vector ? runtime.read_specialization_memory : runtime.read_memory;
	if (reader != nullptr) {
		if (!reader(runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		if (vector || !ReadByPointer(address, {&word, 1})) return false;
	}
	result = word;
	return true;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMax32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type,
	                          bool wave_uniform = false)
	    : m_program(program), m_type(type), m_wave_uniform(wave_uniform) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type != RuntimeValueType::Any &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (m_type == RuntimeValueType::ImmutableInteger &&
		    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
		     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
		     SharedAccessOf(op) != SharedAccess::None)) {
			return finish(false);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU32 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (m_wave_uniform && op == ValueOpcode::GetBuiltin) {
			const auto axis = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
			return finish(m_program.stage == ShaderType::Compute && inst->NumArgs() == 2u &&
			              inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) &&
			              axis.IsImmediate() && axis.GetType() == Type::U32 && axis.U32() < 3u);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type != RuntimeValueType::Any && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u || source->GetOpcode() != ValueOpcode::CompositeConstructU32x2) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	bool                            m_wave_uniform;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};

// The reads a replay made, served in order to the interpreter that takes over when it gives up, so
// that nothing is read twice: the readers' side effects (captured ranges, the read log, strict
// checks) happen once, as without a replay. Past the first read that differs it reads through.
class ReadShim {
public:
	explicit ReadShim(const SrtRuntime& source): m_source(source), m_runtime(source) {
		m_runtime.userdata                   = this;
		m_runtime.read_memory                = source.read_memory != nullptr ? Ordinary : nullptr;
		m_runtime.read_specialization_memory = source.read_specialization_memory != nullptr ? Strict : nullptr;
		Reads().clear();
		Words().clear();
	}
	[[nodiscard]] const SrtRuntime& Runtime() const { return m_runtime; }
	void Serve() { m_serving = true; }

private:
	struct Read {
		uint64_t address = 0;
		uint32_t first   = 0;
		uint32_t count   = 0;
		bool     strict  = false;
		bool     ok      = false;
	};
	static std::vector<Read>& Reads() {
		thread_local std::vector<Read> reads;
		return reads;
	}
	static std::vector<uint32_t>& Words() {
		thread_local std::vector<uint32_t> words;
		return words;
	}
	bool Forward(bool strict, uint64_t address, std::span<uint32_t> values) {
		auto& reads = Reads();
		auto& words = Words();
		if (m_serving && m_next < reads.size()) {
			const auto& read = reads[m_next];
			if (read.strict == strict && read.address == address && read.count == values.size()) {
				++m_next;
				if (read.ok) std::copy_n(words.begin() + read.first, read.count, values.begin());
				return read.ok;
			}
			m_next = reads.size();
		}
		const bool ok = (strict ? m_source.read_specialization_memory : m_source.read_memory)(
		    m_source.userdata, address, values);
		if (!m_serving) {
			reads.push_back({address, static_cast<uint32_t>(words.size()), static_cast<uint32_t>(values.size()),
			                 strict, ok});
			if (ok) words.insert(words.end(), values.begin(), values.end());
		}
		return ok;
	}
	static bool Ordinary(void* self, uint64_t address, std::span<uint32_t> values) {
		return static_cast<ReadShim*>(self)->Forward(false, address, values);
	}
	static bool Strict(void* self, uint64_t address, std::span<uint32_t> values) {
		return static_cast<ReadShim*>(self)->Forward(true, address, values);
	}

	SrtRuntime m_source;
	SrtRuntime m_runtime;
	size_t     m_next    = 0;
	bool       m_serving = false;
};

} // namespace

// Replay traces. RefreshFlatBuffer, interpreted, evaluates a few hundred instructions through
// memoized recursive walkers, with nested EXEC walkers and a fixed-point pass over the blocks; that
// was a sixth of the GPU thread on Marvel's Wolverine's title scenes. The recorder writes one
// interpreted refresh down as the evaluations it made, in order: each instruction a walker evaluated
// is an op over the refs of the operands it asked for. The next refresh of the plan replays the
// list in a loop and checks, at the point the interpreter made it, every choice that depended on a
// value: the arm a Select took, where a LogicalAnd or LogicalOr stopped, the early exits of
// ULessThan(Equal)32, a Phi's agreeing operands, a buffer read's lane guard, each block condition's
// outcome. Every op must also succeed or fail as recorded. Then the replay evaluated what the
// interpreter would have, in the same order, and reads only what it would have read; at the first
// check that fails it gives up and the interpreter takes over (ReadShim). After a replay the walkers'
// memos hold what the interpreted refresh left in them (Trace::seeds), for the evaluations after it.
class SrtTraceRecorder {
public:
	SrtTraceRecorder(ResourcePlan::Trace& trace, const SrtRuntime& ordinary): m_trace(trace), m_ordinary(ordinary) {
		m_trace.ops.clear();
		m_trace.immediates.clear();
		m_trace.seeds.clear();
		m_trace.valid = false;
		m_frames.push_back({});
	}
	int32_t Immediate(uint64_t value) {
		m_trace.immediates.push_back(value);
		return -static_cast<int32_t>(m_trace.immediates.size());
	}
	void Leaf(int32_t ref) { m_children.push_back(ref); }
	// The op whose result a walker found in its memo.
	int32_t Memoized(const ResourcePlan::EvaluationContext& context, uint32_t index) {
		const auto* refs = Find(context);
		if (refs == nullptr || index >= refs->size() || (*refs)[index] == TraceFailed) {
			m_aborted = true;
			return TraceFailed;
		}
		return (*refs)[index];
	}
	void Begin() { m_frames.push_back({.begin = static_cast<uint32_t>(m_children.size())}); }
	void End(const SrtWalker& walker, const Inst& inst, uint32_t index, bool ok, uint64_t value);
	void Form(uint8_t form, uint32_t imm) {
		m_frames.back().form = form;
		m_frames.back().imm  = imm;
	}
	// flags bit 0: the operand was zero, bit 1: its low 32 bits were compared.
	void Check(bool zero, bool low32) {
		Add({.kind = TraceCheckZero, .flags = static_cast<uint8_t>((zero ? 1u : 0u) | (low32 ? 2u : 0u)),
		     .args = {m_children.back()}});
	}
	void CheckEqual(bool equal) {
		Add({.kind = TraceCheckEqual, .flags = equal, .args = {m_children[m_frames.back().begin], m_children.back()}});
	}
	void Condition(bool ok, bool nonzero) {
		Add({.kind = TraceCondition, .flags = nonzero, .ok = ok, .args = {TakeRequest()}});
	}
	void Flat(uint32_t offset) { Add({.kind = TraceFlat, .imm = offset, .args = {TakeRequest()}}); }
	void Finish(const SrtWalker& walker, uint8_t key);

private:
	struct Frame {
		uint32_t begin = 0;
		uint8_t  form  = FormNone;
		uint32_t imm   = 0;
	};
	int32_t Add(const ResourcePlan::TraceOp& op) {
		if (m_trace.ops.size() >= TraceMaxOps) m_aborted = true;
		m_trace.ops.push_back(op);
		return static_cast<int32_t>(m_trace.ops.size() - 1u);
	}
	int32_t TakeRequest() {
		const auto ref = m_children.back();
		m_children.pop_back();
		return ref;
	}
	std::vector<int32_t>* Find(const ResourcePlan::EvaluationContext& context) {
		for (auto& [owner, refs]: m_memo)
			if (owner == &context) return &refs;
		return nullptr;
	}

	ResourcePlan::Trace& m_trace;
	SrtRuntime           m_ordinary; // the refreshing walker's: a read through another is a clean read
	std::vector<int32_t> m_children;
	std::vector<Frame>   m_frames;
	// Per walker memo: the op that last gave each evaluation index its value.
	std::vector<std::pair<const ResourcePlan::EvaluationContext*, std::vector<int32_t>>> m_memo;
	bool m_aborted = false;
};

void SrtTraceRecorder::End(const SrtWalker& walker, const Inst& inst, uint32_t index, bool ok, uint64_t value) {
	const auto frame = m_frames.back();
	m_frames.pop_back();
	const std::span<const int32_t> operands(m_children.data() + frame.begin, m_children.size() - frame.begin);
	ResourcePlan::TraceOp op {.ok = ok, .opcode = inst.GetOpcode(), .imm = frame.imm, .value = value};
	const auto take = [&](size_t first, size_t count, size_t limit) {
		if (operands.size() < first || count > limit) {
			m_aborted = true;
			return;
		}
		std::copy_n(operands.begin() + static_cast<ptrdiff_t>(first), count, op.args.begin());
		op.count = static_cast<uint8_t>(count);
	};
	const auto pure = [&](uint32_t arity) {
		op.kind = TracePure;
		op.imm  = arity;
		op.flags = (op.opcode == ValueOpcode::FPOrdLessThanEqual32 || op.opcode == ValueOpcode::FPOrdGreaterThanEqual32) &&
		           inst.Flags<FPCompareFlags>().flush_input_denorms;
		take(0, operands.size(), arity);
	};
	switch (frame.form) {
		case FormNone:
			if (const auto arity = PureArity(op.opcode); arity != 0) {
				pure(arity);
			} else if (!ok && operands.empty()) {
				op.kind = TraceFixed;
			} else {
				m_aborted = true;
			}
			break;
		case FormPure: pure(2); break;
		case FormPureSwapped: // ULessThan32 asked for its second operand first
			pure(2);
			if (op.count == 2) std::swap(op.args[0], op.args[1]);
			break;
		case FormPassFirst:
		case FormPassLast:
			op.kind = TracePass;
			if (operands.empty()) {
				m_aborted = true;
			} else {
				op.args[0] = frame.form == FormPassFirst ? operands.front() : operands.back();
				op.count   = 1;
			}
			break;
		case FormFixed: op.kind = TraceFixed; break;
		case FormUserData: op.kind = TraceUserData; break;
		case FormShaderBase: op.kind = TraceShaderBase; break;
		case FormRawRead: {
			const auto guarded = (frame.imm & RawReadGuarded) != 0u ? 1u : 0u;
			op.kind  = TraceRawRead;
			op.imm   = frame.imm & ~RawReadGuarded;
			op.flags = walker.m_runtime.read_memory != m_ordinary.read_memory ||
			           walker.m_runtime.read_specialization_memory != m_ordinary.read_specialization_memory;
			take(guarded, operands.size() - std::min<size_t>(operands.size(), guarded), 5);
			break;
		}
		case FormExtract64:
			op.kind = TracePure;
			op.imm  = 2;
			if (operands.size() != 1u) {
				m_aborted = true;
			} else {
				op.args  = {operands[0], Immediate(frame.imm)};
				op.count = 2;
			}
			break;
		default: m_aborted = true; break;
	}
	m_children.resize(frame.begin);
	const auto ref = Add(op);
	auto*      refs = Find(walker.m_context);
	if (refs == nullptr) refs = &m_memo.emplace_back(&walker.m_context, std::vector<int32_t> {}).second;
	if (refs->size() <= index) refs->resize(index + 1u, TraceFailed);
	(*refs)[index] = ref;
	m_children.push_back(ref);
}

void SrtTraceRecorder::Finish(const SrtWalker& walker, uint8_t key) {
	if (m_aborted) return;
	const auto& program = walker.m_program;
	m_trace.key     = key;
	m_trace.visited = program.visited_blocks;
	m_trace.active  = program.active_sources;
	m_trace.active_images = program.active_images;
	for (uint8_t root = 0; root < 2u; ++root) {
		const auto* owner = root == 0u ? &walker : walker.m_clean_evaluator;
		if (owner == nullptr) continue;
		const auto& context = owner->m_context;
		const auto* refs    = Find(context);
		for (uint32_t index = 0; refs != nullptr && index < refs->size() && index < context.values.size(); ++index) {
			const auto ref        = (*refs)[index];
			const auto generation = context.values[index].generation;
			if (ref == TraceFailed) continue;
			if (generation == context.generation || generation == (context.generation | 2u))
				m_trace.seeds.push_back({index, ref, root, generation != context.generation});
		}
	}
	m_trace.valid = true;
}

// The x86 SRT walker of upstream KytyPS5 (d7856753e, faulting loads bdf48dcf3) by nmzik, for the GPU thread:
// MaterializeResources runs it where the runtime has direct reads and takes its outputs; values it does not
// know are evaluated by the interpreter below (SrtWalker), as are all walks of the ahead workers.
namespace {

bool ImmediateBits(Value value, uint64_t& bits) {
	switch (value.GetType()) {
		case Type::U1: bits = value.U1(); return true;
		case Type::U8: bits = value.U8(); return true;
		case Type::U16: bits = value.U16(); return true;
		case Type::U32: bits = value.U32(); return true;
		case Type::U64: bits = value.U64(); return true;
		case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); return true;
		default: return false;
	}
}

// Scratch layout in bytes: per descriptor source its eight DWORDs and a known DWORD, one
// activity byte per source, then per image, then one qword per walked value (all ones when
// unknown). The walker's value slots, captured addresses, reachability and states follow.
constexpr uint32_t DescriptorStride = 9u * sizeof(uint32_t);

uint32_t ActiveOffset(const ResourcePlan& plan) {
	return static_cast<uint32_t>(plan.descriptor_sources.size()) * DescriptorStride;
}

uint32_t OutputOffset(const ResourcePlan& plan) {
	const auto active = plan.descriptor_sources.size() + plan.info.images.size();
	return (ActiveOffset(plan) + static_cast<uint32_t>(active) + 7u) & ~7u;
}

bool HasSse41() {
	static const bool sse41 = Xbyak::util::Cpu().has(Xbyak::util::Cpu::tSSE41);
	return sse41;
}

const uint8_t* Scratch(const ResourcePlan& plan) {
	return reinterpret_cast<const uint8_t*>(plan.srt_scratch.data());
}

thread_local const Xbyak::CodeGenerator* t_running_walker = nullptr;
// The running walk's shader-only loads the exception handler completed as 0 (null: it completes none).
thread_local std::vector<uint64_t>*        t_fill_loads = nullptr;
std::atomic<SrtFillGate>                   g_fill_gate  = nullptr;
thread_local bool                         t_walker_faulted = false;

// Every guest load is mov eax, [r10] (41 8b 02), so a fault completes it as a zero read.
void LoadGuest(Xbyak::CodeGenerator& code, const Xbyak::RegExp& address, bool shader_only = false) {
	code.lea(code.r10, code.ptr[address]);
	if (shader_only) {
		// mov eax, [r10+0] (41 8b 42 00): the same load, which the exception handler tells apart (SrtWalkerShaderOnlyLoad).
		code.db(0x41);
		code.db(0x8b);
		code.db(0x42);
		code.db(0x00);
		return;
	}
	code.mov(code.eax, code.dword[code.r10]);
}


// Emits one x86 function per plan. Each value, per EXEC-mask context, is a memoized thunk with a
// state byte (0 unevaluated, 1 unknown or in progress, 2 known, 3 a control Phi holding its first
// value) and a qword slot; a reference is `call thunk; cmp state, 2`. Operands that are unknown,
// constant or plain user data are folded at emission. Registers: rdi user data, rsi flat buffer,
// rdx scratch, r8 capture cursor.
class WalkerCompiler {
public:
	WalkerCompiler(ResourcePlan& plan, Xbyak::CodeGenerator& code): m_plan(plan), m_code(code) {}

	void Compile(std::span<const Value> values);

private:
	enum class Kind : uint8_t { Unknown, Constant, UserData, ShaderBase, Node };
	struct Operand {
		Kind     kind  = Kind::Unknown;
		uint64_t value = 0;

		bool operator==(const Operand& other) const = default;
	};
	// masked: the operand depends on its context's EXEC mask, so it is not the root context's.
	struct Ref {
		Operand operand;
		bool    masked = false;
	};
	struct Node {
		const Inst*          inst    = nullptr;
		uint32_t             context = 0;
		Operand              result;
		std::vector<Operand> args;
		uint32_t             slot = 0, state = 0;
		uint32_t             address = UINT32_MAX; // a shader-only read's: scratch offset of its loaded address
		Xbyak::Label         label;
		bool                 pending = true, referenced = false, masked = false;
	};

	static constexpr auto Near = Xbyak::CodeGenerator::T_NEAR;

	bool Thunk(uint32_t index) const {
		const auto& node = m_nodes[index];
		return node.referenced || node.result == Operand {Kind::Node, index};
	}
	static bool MayBeUnknown(const Operand& operand) {
		return operand.kind == Kind::Unknown || operand.kind == Kind::Node;
	}

	uint32_t Context(Value mask) {
		mask = mask.Resolve();
		const auto found = std::ranges::find(m_masks, mask);
		if (found != m_masks.end()) return static_cast<uint32_t>(found - m_masks.begin());
		m_masks.push_back(mask);
		return static_cast<uint32_t>(m_masks.size() - 1u);
	}

	void AddOutput(Value value, uint32_t gate);
	void Consume(std::vector<std::vector<uint32_t>>& gates);
	Ref Evaluate(Value value, uint32_t context);
	Operand Analyze(uint32_t index);
	Operand AnalyzeRead(Node& node, const auto& arg, Operand self);
	void Layout();
	void Gate(std::vector<uint32_t>& gates, const Xbyak::Label& skip);
	void Zero(uint32_t offset, uint32_t size);
	void Branch(const Operand& operand, const Xbyak::Label& unknown);
	void Load(const Xbyak::Reg64& reg, const Operand& operand);
	void Fetch(const Operand& operand, const Xbyak::Label& unknown);
	void Set(void (Xbyak::CodeGenerator::*condition)(const Xbyak::Operand&));
	void Store(const Node& node);
	void EmitNode(uint32_t index);
	void EmitRead(const Node& node);
	void EmitStrict(const Node& node);

	ResourcePlan&                                     m_plan;
	Xbyak::CodeGenerator&                             m_code;
	std::vector<Value>                                m_masks {Value {}};
	std::deque<Node>                                  m_nodes;
	std::map<std::pair<const Inst*, uint32_t>, uint32_t> m_memo;
	std::vector<Value>                                m_outputs;
	std::vector<std::vector<uint32_t>>                m_output_gates;
	std::unordered_set<const Inst*>                   m_shader_only; // a compute plan's reads only the shader uses
	Xbyak::Label                                      m_ret;
	uint32_t m_shader_base = 0, m_reach = 0, m_states = 0, m_states_end = 0;
	uint32_t m_user_data = 0, m_captures = 0;
};

void WalkerCompiler::AddOutput(Value value, uint32_t gate) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr) return;
	auto index = IR::WalkOutput(m_plan, *inst);
	if (index >= m_outputs.size() || m_outputs[index].TryInstruction() != inst) {
		index = static_cast<uint32_t>(m_outputs.size());
		IR::SetWalkOutput(m_plan, *inst, index);
		m_outputs.push_back(value.Resolve());
		m_output_gates.emplace_back();
	}
	m_output_gates[index].push_back(gate);
}

// Mirrors MaterializeResources: each descriptor it reads back is gated by the activity of the
// resource root that needs it, and so are an indirect root's selector values.
void WalkerCompiler::Consume(std::vector<std::vector<uint32_t>>& gates) {
	const auto& sources = m_plan.descriptor_sources;
	const auto  consume = [&](uint32_t root) {
		if (root >= sources.size()) return;
		const auto& indirect = sources[root].indirect_descriptor;
		if (!indirect) {
			gates[root].push_back(root);
			return;
		}
		const auto add = [&](uint32_t source) {
			if (source < sources.size()) gates[source].push_back(root);
		};
		if (indirect->sources.empty()) {
			add(indirect->table_source);
			if (indirect->selector) {
				add(indirect->selector->source);
				if (indirect->selector->rows) add(indirect->selector->rows->source);
			}
		}
		for (const auto source: indirect->sources) add(source);
		AddOutput(indirect->key_count, root);
		AddOutput(indirect->selector_first, root);
		AddOutput(indirect->selector_mask, root);
	};
	for (const auto& buffer: m_plan.info.buffers) consume(buffer.source);
	for (const auto& image: m_plan.info.images) consume(image.source);
	for (const auto& sampler: m_plan.info.samplers) {
		const auto* indirect = sampler.source < sources.size() && sources[sampler.source].indirect_descriptor
		    ? &*sources[sampler.source].indirect_descriptor : nullptr;
		if (indirect == nullptr || indirect->sources.empty()) {
			consume(sampler.source);
		} else if (indirect->sources[0] < sources.size()) {
			gates[indirect->sources[0]].push_back(indirect->sources[0]);
		}
	}
	const auto& fill = m_plan.uniform_fill;
	if (fill.fill.words != 0) {
		AddOutput(fill.pattern_size, UINT32_MAX);
		AddOutput(fill.element_count, UINT32_MAX);
		for (const auto value: fill.values) AddOutput(value, UINT32_MAX);
	}
}

WalkerCompiler::Ref WalkerCompiler::Evaluate(Value value, uint32_t context) {
	value = value.Resolve();
	if (value.IsEmpty()) return {};
	if (value.IsImmediate()) {
		uint64_t bits = 0;
		return ImmediateBits(value, bits) ? Ref {{Kind::Constant, bits}} : Ref {};
	}
	const auto mask = m_masks[context];
	if (context != 0 && value == mask) return {{Kind::Constant, 1u}, true};
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (context != 0 && IsRuntimeSelect(op) && inst->NumArgs() == 3 && inst->Arg(0).Resolve() == mask) {
		return {Evaluate(inst->Arg(1), context).operand, true};
	}
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConditionRef: return Evaluate(inst->Arg(0), context);
		case ValueOpcode::CompositeExtractU32x2: {
			const auto* source = inst->Arg(0).ResolveInstruction();
			const auto  index  = inst->Arg(1).Resolve();
			if (source == nullptr || source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 ||
			    !index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) return {};
			return Evaluate(source->Arg(index.U32()), context);
		}
		case ValueOpcode::ReadFirstLane:
			// The lane read does not depend on the EXEC mask it is read under.
			return {Evaluate(inst->Arg(0), Context(inst->Arg(1))).operand};
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_plan.user_data_base || reg - m_plan.user_data_base >= m_plan.user_data_count) return {};
			return {{Kind::UserData, reg - m_plan.user_data_base}};
		}
		case ValueOpcode::GetShaderBase: return {{Kind::ShaderBase}};
		case ValueOpcode::Phi:
			if (!m_plan.control_phi_predecessors.contains(inst)) {
				const auto invariant = ResolveInvariantPhi(m_plan, value);
				return invariant.IsEmpty() ? Ref {} : Evaluate(invariant, context);
			}
			break;
		default: break;
	}
	const auto [found, inserted] = m_memo.try_emplace({inst, context}, static_cast<uint32_t>(m_nodes.size()));
	const auto index = found->second;
	if (!inserted) {
		auto& node = m_nodes[index];
		if (!node.pending) return {node.result, node.masked};
		// A cycle: the thunk reads as unknown while it runs.
		node.referenced = true;
		return {{Kind::Node, index}, context != 0};
	}
	auto& node   = m_nodes.emplace_back();
	node.inst    = inst;
	node.context = context;
	node.result  = Analyze(index);
	node.pending = false;
	if (context != 0 && !node.masked && !node.referenced) {
		node.result = Evaluate(value, 0).operand;
		return {node.result};
	}
	return {node.result, node.masked};
}

WalkerCompiler::Operand WalkerCompiler::Analyze(uint32_t index) {
	auto&       node = m_nodes[index];
	const auto& inst = *node.inst;
	const auto  self = Operand {Kind::Node, index};
	const auto  arg  = [&](Value value) {
		const auto ref = Evaluate(value, node.context);
		node.masked |= ref.masked;
		return ref.operand;
	};
	const auto constant = [](uint64_t value) { return Operand {Kind::Constant, value}; };
	switch (inst.GetOpcode()) {
		case ValueOpcode::Phi: {
			bool known = false;
			for (size_t i = 0; i < inst.NumArgs(); ++i) {
				node.args.push_back(arg(inst.Arg(i)));
				known |= node.args.back().kind != Kind::Unknown;
			}
			return known ? self : Operand {};
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32: return AnalyzeRead(node, arg, self);
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: {
			const auto predicate = arg(inst.Arg(0));
			if (predicate.kind == Kind::Unknown) return {};
			if (predicate.kind == Kind::Constant) return arg(inst.Arg(predicate.value != 0u ? 1 : 2));
			node.args = {predicate, arg(inst.Arg(1)), arg(inst.Arg(2))};
			return node.args[1].kind == Kind::Unknown && node.args[2].kind == Kind::Unknown ? Operand {} : self;
		}
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32: {
			// No U32 is below zero and every U32 is at least zero, including an unknown loop-carried
			// or varying lane value. args[0] is that zero-tested bound.
			const bool less  = inst.GetOpcode() == ValueOpcode::ULessThan32;
			const auto bound = arg(inst.Arg(less ? 1 : 0));
			if (bound.kind == Kind::Unknown) return {};
			if (bound.kind == Kind::Constant && static_cast<uint32_t>(bound.value) == 0u) return constant(!less);
			node.args = {bound, arg(inst.Arg(less ? 0 : 1))};
			return node.args[1].kind == Kind::Unknown && bound.kind == Kind::Constant ? Operand {} : self;
		}
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd: {
			// A deciding operand, nonzero for Or and zero for And, fixes the result.
			const bool conjunction = inst.GetOpcode() == ValueOpcode::LogicalAnd;
			const auto decides     = [&](const Operand& x) {
				return x.kind == Kind::Constant && (x.value == 0u) == conjunction;
			};
			const auto a = arg(inst.Arg(0));
			if (decides(a)) return constant(!conjunction);
			const auto b = arg(inst.Arg(1));
			if (decides(b)) return constant(!conjunction);
			if (a.kind == Kind::Constant && b.kind == Kind::Constant) return constant(conjunction);
			if (a.kind == Kind::Constant && b.kind == Kind::Unknown) return {};
			// Only lanes satisfying the left EXEC factor can make a conjunction true, so an
			// unknown left side retries the right one under that factor as its mask.
			Operand refined;
			if (conjunction && MayBeUnknown(a)) {
				refined = Evaluate(inst.Arg(1), Context(inst.Arg(0))).operand;
				if (refined == b || (refined.kind == Kind::Constant && refined.value != 0u)) refined = {};
				if (a.kind == Kind::Unknown && refined.kind == Kind::Constant) return constant(0u);
			}
			if (a.kind == Kind::Unknown && refined.kind == Kind::Unknown &&
			    (b.kind == Kind::Unknown || b.kind == Kind::Constant)) return {};
			node.args = {a, b, refined};
			return self;
		}
		case ValueOpcode::CompositeExtractU64: {
			const auto component = inst.Arg(1).Resolve();
			if (!component.IsImmediate() || component.GetType() != Type::U32 || component.U32() >= 2u) return {};
			[[fallthrough]];
		}
		default:
			// The other runtime-uniform ops are strict; a U32x2 pair is only read through its extracts.
			if (!IsRuntimeUniformOp(inst.GetOpcode()) || inst.GetOpcode() == ValueOpcode::CompositeConstructU32x2 ||
			    (inst.GetOpcode() == ValueOpcode::FPTrunc32 && !HasSse41()))
				return {};
			for (size_t i = 0; i < inst.NumArgs(); ++i) {
				node.args.push_back(arg(inst.Arg(i)));
				if (node.args.back().kind == Kind::Unknown) return {};
			}
			return self;
	}
}

WalkerCompiler::Operand WalkerCompiler::AnalyzeRead(Node& node, const auto& arg, Operand self) {
	const auto& inst = *node.inst;
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (!IsRawRead(m_plan, inst) || handle == nullptr) return {};
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	if (vector) {
		const auto guard = arg(inst.Arg(4));
		if (guard.kind == Kind::Unknown) return {};
		if (guard.kind == Kind::Constant && guard.value == 0u) return {Kind::Constant, 0u};
		node.args.push_back(guard);
	}
	const bool buffer = inst.GetOpcode() != ValueOpcode::LoadAddressU32;
	const auto offset = static_cast<int32_t>(m_plan.memory_info[inst.Flags<MemoryFlags>().index].offset);
	bool failed = buffer && (handle->NumArgs() != 4u || offset < 0);
	for (const auto& word: {handle->Arg(0), handle->Arg(1), inst.Arg(1)}) {
		node.args.push_back(arg(word));
		failed |= node.args.back().kind == Kind::Unknown;
	}
	for (uint32_t word = 2; buffer && !failed && word < 4u; ++word) {
		node.args.push_back(arg(handle->Arg(word)));
		failed |= node.args.back().kind == Kind::Unknown;
	}
	if (!failed) return self;
	// A disabled vector lane still reads zero.
	if (!vector || node.args[0].kind == Kind::Constant) return {};
	node.args.resize(1);
	return self;
}

void WalkerCompiler::Layout() {
	uint32_t offset = OutputOffset(m_plan) + static_cast<uint32_t>(m_outputs.size()) * 8u;
	m_shader_base = offset;
	offset += 8u;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (!Thunk(index)) continue;
		m_nodes[index].slot = offset;
		offset += 8u;
		// Only the root context's node: a read in another EXEC context feeds a value the host evaluates.
		if (m_nodes[index].context == 0 && m_shader_only.contains(m_nodes[index].inst)) {
			m_nodes[index].address = offset;
			offset += 8u;
		}
	}
	m_reach = offset;
	m_plan.srt_outputs = m_shader_base;
	m_plan.srt_reach   = m_reach;
	offset += static_cast<uint32_t>(m_plan.control_flow.size());
	m_states = offset;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (Thunk(index)) m_nodes[index].state = offset++;
	}
	m_states_end = offset;
	// Captured addresses come last; EmitRead counts them.
	m_plan.srt_captures = (offset + 7u) & ~7u;
}

// Falls through when any gating source is active. The list is not empty.
void WalkerCompiler::Gate(std::vector<uint32_t>& gates, const Xbyak::Label& skip) {
	auto& c = m_code;
	std::ranges::sort(gates);
	if (gates.back() == UINT32_MAX) return;
	gates.erase(std::unique(gates.begin(), gates.end()), gates.end());
	Xbyak::Label active;
	for (size_t i = 0; i < gates.size(); ++i) {
		c.cmp(c.byte[c.rdx + ActiveOffset(m_plan) + gates[i]], 0);
		if (i + 1u < gates.size()) {
			c.jne(active, Near);
		} else {
			c.je(skip, Near);
		}
	}
	c.L(active);
}

void WalkerCompiler::Zero(uint32_t offset, uint32_t size) {
	auto& c = m_code;
	if (size == 0) return;
	c.mov(c.r11, c.rdi);
	c.lea(c.rdi, c.ptr[c.rdx + offset]);
	c.mov(c.ecx, size);
	c.xor_(c.eax, c.eax);
	c.rep();
	c.stosb();
	c.mov(c.rdi, c.r11);
}

// Jumps to unknown unless the operand has a value at run time.
void WalkerCompiler::Branch(const Operand& operand, const Xbyak::Label& unknown) {
	auto& c = m_code;
	if (operand.kind == Kind::Unknown) {
		c.jmp(unknown, Near);
	} else if (operand.kind == Kind::Node) {
		auto& node = m_nodes[operand.value];
		c.call(node.label);
		c.cmp(c.byte[c.rdx + node.state], 2);
		c.jne(unknown, Near);
	}
}

void WalkerCompiler::Load(const Xbyak::Reg64& reg, const Operand& operand) {
	auto& c = m_code;
	switch (operand.kind) {
		case Kind::Constant: c.mov(reg, operand.value); break;
		case Kind::UserData:
			m_user_data = std::max(m_user_data, static_cast<uint32_t>(operand.value) + 1u);
			c.mov(reg.cvt32(), c.dword[c.rdi + operand.value * sizeof(uint32_t)]);
			break;
		case Kind::ShaderBase: c.mov(reg, c.qword[c.rdx + m_shader_base]); break;
		case Kind::Node: c.mov(reg, c.qword[c.rdx + m_nodes[operand.value].slot]); break;
		case Kind::Unknown: break;
	}
}

// Loads the operand into rax, or jumps to unknown when it has no value at run time.
void WalkerCompiler::Fetch(const Operand& operand, const Xbyak::Label& unknown) {
	Branch(operand, unknown);
	Load(m_code.rax, operand);
}

// Sets eax to 1 when the condition holds, else 0.
void WalkerCompiler::Set(void (Xbyak::CodeGenerator::*condition)(const Xbyak::Operand&)) {
	(m_code.*condition)(m_code.al);
	m_code.movzx(m_code.eax, m_code.al);
}

void WalkerCompiler::Store(const Node& node) {
	auto& c = m_code;
	c.mov(c.qword[c.rdx + node.slot], c.rax);
	c.mov(c.byte[c.rdx + node.state], 2);
	c.ret();
}

void WalkerCompiler::EmitNode(uint32_t index) {
	auto&       c    = m_code;
	auto&       node = m_nodes[index];
	const auto& args = node.args;
	c.L(node.label);
	c.cmp(c.byte[c.rdx + node.state], 0);
	c.jne(m_ret, Near);
	c.mov(c.byte[c.rdx + node.state], 1);
	if (node.result != Operand {Kind::Node, index}) {
		// A cycle referenced this value before it folded to another operand.
		Fetch(node.result, m_ret);
		Store(node);
		return;
	}
	switch (node.inst->GetOpcode()) {
		case ValueOpcode::Phi: {
			// Only predecessors reached by this walk contribute, and they must agree. The state is 3
			// once one of them has stored the value; callers read it like 1, as unknown.
			const auto& blocks = m_plan.control_phi_predecessors.at(node.inst);
			const auto  state  = c.byte[c.rdx + node.state];
			for (size_t i = 0; i < args.size(); ++i) {
				Xbyak::Label next, first;
				c.cmp(c.byte[c.rdx + m_reach + blocks[i]], 0);
				c.je(next, Near);
				Fetch(args[i], m_ret);
				c.cmp(state, 1);
				c.je(first, Near);
				c.cmp(c.rax, c.qword[c.rdx + node.slot]);
				c.jne(m_ret, Near);
				c.jmp(next, Near);
				c.L(first);
				c.mov(c.qword[c.rdx + node.slot], c.rax);
				c.mov(state, 3);
				c.L(next);
			}
			c.cmp(state, 3);
			c.jne(m_ret, Near);
			c.mov(state, 2);
			c.ret();
			return;
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32: EmitRead(node); return;
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: {
			Xbyak::Label other;
			Fetch(args[0], m_ret);
			c.test(c.rax, c.rax);
			c.jz(other, Near);
			Fetch(args[1], m_ret);
			Store(node);
			c.L(other);
			Fetch(args[2], m_ret);
			Store(node);
			return;
		}
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32: {
			const bool less = node.inst->GetOpcode() == ValueOpcode::ULessThan32;
			Xbyak::Label bound_zero;
			Fetch(args[0], m_ret);
			c.test(c.eax, c.eax);
			c.jz(bound_zero, Near);
			Branch(args[1], m_ret);
			Load(c.rax, args[0]);
			Load(c.rcx, args[1]);
			// ULT tests other < bound, ULE bound <= other.
			c.cmp(c.ecx, c.eax);
			Set(less ? &Xbyak::CodeGenerator::setb : &Xbyak::CodeGenerator::setae);
			Store(node);
			c.L(bound_zero);
			c.mov(c.eax, uint32_t {!less});
			Store(node);
			return;
		}
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd: {
			const bool conjunction = node.inst->GetOpcode() == ValueOpcode::LogicalAnd;
			Xbyak::Label unknown_left, refine, decided;
			const auto decide = [&] {
				if (conjunction) {
					c.jz(decided, Near);
				} else {
					c.jnz(decided, Near);
				}
			};
			Branch(args[0], unknown_left);
			if (args[0].kind != Kind::Unknown) {
				Load(c.rax, args[0]);
				c.test(c.rax, c.rax);
				decide();
				Fetch(args[1], m_ret);
				c.test(c.rax, c.rax);
				Set(&Xbyak::CodeGenerator::setne);
				Store(node);
			}
			c.L(unknown_left);
			if (MayBeUnknown(args[0])) {
				// Without the left side only a deciding right side, or a conjunction's zero
				// refined right side, is known.
				if (args[1].kind != Kind::Unknown) {
					Fetch(args[1], refine);
					c.test(c.rax, c.rax);
					decide();
				}
				c.L(refine);
				Fetch(args[2], m_ret);
				c.test(c.rax, c.rax);
				c.jnz(m_ret, Near);
			}
			c.L(decided);
			c.mov(c.eax, uint32_t {!conjunction});
			Store(node);
			return;
		}
		default: EmitStrict(node); return;
	}
}

// Raw scalar and uniform DWORD buffer loads; out-of-bounds reads are zero.
void WalkerCompiler::EmitRead(const Node& node) {
	auto&      c    = m_code;
	const auto& inst = *node.inst;
	const auto op = inst.GetOpcode();
	const bool vector = op == ValueOpcode::LoadBufferU32;
	auto args = std::span(node.args);
	Xbyak::Label zero;
	if (vector) {
		Fetch(args[0], m_ret);
		c.test(c.rax, c.rax);
		c.jz(zero, Near);
		args = args.subspan(1);
		if (args.empty()) c.jmp(m_ret, Near);
	}
	if (!args.empty()) {
		for (const auto& arg: args) Branch(arg, m_ret);
		const auto immediate =
		    static_cast<int64_t>(static_cast<int32_t>(m_plan.memory_info[inst.Flags<MemoryFlags>().index].offset)) &
		    ~int64_t {3};
		// rax = the 48-bit base (high << 32 | low) with its low two bits clear.
		Load(c.rax, args[1]);
		c.shl(c.rax, 48);
		c.shr(c.rax, 16);
		Load(c.rcx, args[0]);
		c.and_(c.ecx, ~3u);
		c.or_(c.rax, c.rcx);
		Load(c.r9, args[2]);
		c.and_(c.r9d, ~3u);
		if (op == ValueOpcode::LoadAddressU32) {
			c.lea(c.rax, c.ptr[c.rax + c.r9 + static_cast<size_t>(immediate)]);
			c.mov(c.rcx, c.rax);
			c.shr(c.rcx, 48);
			c.jnz(m_ret, Near);
			LoadGuest(c, c.rax, node.address != UINT32_MAX);
		} else {
			Load(c.r11, args[1]);
			c.shr(c.r11d, 16);
			if (vector) {
				// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
				c.test(c.r11d, 1u << 15u);
				c.jnz(m_ret, Near);
				Load(c.rcx, args[4]);
				c.test(c.ecx, (1u << 23u) | 0xf0000000u);
				c.jnz(m_ret, Near);
				c.test(c.r11d, 0x3fffu);
				c.jz(zero, Near);
				c.test(c.ecx, 0x7fu << 12u);
				c.jz(zero, Near);
			}
			c.and_(c.r11d, 0x3fffu);
			Load(c.rcx, args[3]);
			if (vector) {
				c.test(c.ecx, c.ecx);
				c.jz(zero, Near);
			}
			Xbyak::Label sized;
			c.test(c.r11d, c.r11d);
			c.jz(sized, Near);
			c.imul(c.rcx, c.r11);
			c.L(sized);
			c.lea(c.r11, c.ptr[c.r9 + static_cast<size_t>(immediate) + 4u]);
			c.cmp(c.r11, c.rcx);
			c.ja(zero, Near);
			LoadGuest(c, c.rax + c.r9 + static_cast<size_t>(immediate), node.address != UINT32_MAX);
		}
		if (node.address != UINT32_MAX) c.mov(c.qword[c.rdx + node.address], c.r10);
		++m_captures;
		c.mov(c.qword[c.r8], c.r10);
		c.add(c.r8, 8);
		Store(node);
	}
	c.L(zero);
	c.xor_(c.eax, c.eax);
	Store(node);
}

void WalkerCompiler::EmitStrict(const Node& node) {
	auto&       c    = m_code;
	const auto& inst = *node.inst;
	for (const auto& arg: node.args) Branch(arg, m_ret);
	const std::array regs {c.rax, c.rcx, c.r9, c.r11};
	for (size_t i = 0; i < node.args.size() && i < regs.size(); ++i) Load(regs[i], node.args[i]);
	// Field ops require offset + width <= 32. Leaves (1 << width) - 1 in r11, the offset in
	// cl and the former rcx (the offset or the inserted value) in r10.
	const auto field = [&](const Xbyak::Reg32& offset, const Xbyak::Reg32& width) {
		c.lea(c.r10, c.ptr[offset.cvt64() + width.cvt64()]);
		c.cmp(c.r10, 32);
		c.ja(m_ret, Near);
		c.mov(c.r10, c.rcx);
		c.mov(c.ecx, width);
		c.mov(c.r11d, 1);
		c.shl(c.r11, c.cl);
		c.dec(c.r11);
		c.mov(c.ecx, offset.getIdx() == Xbyak::Operand::RCX ? c.r10d : offset);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::CompositeConstructU64:
			c.shl(c.rcx, 32);
			c.or_(c.rax, c.rcx);
			break;
		case ValueOpcode::CompositeExtractU64:
			if (inst.Arg(1).Resolve().U32() != 0u) c.shr(c.rax, 32);
			c.mov(c.eax, c.eax);
			break;
		case ValueOpcode::IAdd32: c.add(c.eax, c.ecx); break;
		case ValueOpcode::IAdd64: c.add(c.rax, c.rcx); break;
		case ValueOpcode::ISub32: c.sub(c.eax, c.ecx); break;
		case ValueOpcode::ISub64: c.sub(c.rax, c.rcx); break;
		case ValueOpcode::IMul32: c.imul(c.eax, c.ecx); break;
		case ValueOpcode::UMulHi:
			c.imul(c.rax, c.rcx);
			c.shr(c.rax, 32);
			break;
		case ValueOpcode::UMin32:
			c.cmp(c.eax, c.ecx);
			c.cmova(c.eax, c.ecx);
			break;
		case ValueOpcode::UMax32:
			c.cmp(c.eax, c.ecx);
			c.cmovb(c.eax, c.ecx);
			break;
		case ValueOpcode::SMax32:
			c.cmp(c.eax, c.ecx);
			c.cmovl(c.eax, c.ecx);
			break;
		case ValueOpcode::ConvertF32U32:
			c.cvtsi2ss(c.xmm0, c.rax);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::ConvertU32F32: {
			// The conversion saturates and maps NaN to zero.
			Xbyak::Label done;
			c.movd(c.xmm0, c.eax);
			c.xorps(c.xmm1, c.xmm1);
			c.xor_(c.eax, c.eax);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jbe(done, Near);
			c.mov(c.ecx, 0x4f800000u);
			c.movd(c.xmm1, c.ecx);
			c.mov(c.eax, UINT32_MAX);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jae(done, Near);
			c.cvttss2si(c.rax, c.xmm0);
			c.L(done);
			break;
		}
		case ValueOpcode::ConvertS32F32: {
			// The conversion saturates and maps NaN to zero. cvttss2si's out-of-range result,
			// INT32_MIN, is already right for negative overflow.
			Xbyak::Label done;
			c.movd(c.xmm0, c.eax);
			c.xor_(c.eax, c.eax);
			c.ucomiss(c.xmm0, c.xmm0);
			c.jp(done, Near);
			c.mov(c.ecx, 0x4f000000u);
			c.movd(c.xmm1, c.ecx);
			c.mov(c.eax, INT32_MAX);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jae(done, Near);
			c.cvttss2si(c.eax, c.xmm0);
			c.L(done);
			break;
		}
		case ValueOpcode::FPMul32:
			c.movd(c.xmm0, c.eax);
			c.movd(c.xmm1, c.ecx);
			c.mulss(c.xmm0, c.xmm1);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::FPTrunc32:
			c.movd(c.xmm0, c.eax);
			c.roundss(c.xmm0, c.xmm0, 3);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::FPRecipIFlag32:
			// Normal positive powers of two have exact normal reciprocals in every FP mode.
			c.test(c.eax, 0x807fffffu);
			c.jnz(m_ret, Near);
			c.shr(c.eax, 23);
			c.jz(m_ret, Near);
			c.cmp(c.eax, 254);
			c.jae(m_ret, Near);
			c.neg(c.eax);
			c.add(c.eax, 254);
			c.shl(c.eax, 23);
			break;
		case ValueOpcode::FPIsNan32:
			c.and_(c.eax, 0x7fffffffu);
			c.cmp(c.eax, 0x7f800000u);
			Set(&Xbyak::CodeGenerator::seta);
			break;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (inst.Flags<FPCompareFlags>().flush_input_denorms) {
				for (const auto& reg: {c.eax, c.ecx}) {
					Xbyak::Label normal;
					c.mov(c.r11d, reg);
					c.and_(c.r11d, 0x7fffffffu);
					c.cmp(c.r11d, 0x00800000u);
					c.jae(normal, Near);
					c.and_(reg, 0x80000000u);
					c.L(normal);
				}
			}
			c.movd(c.xmm0, c.eax);
			c.movd(c.xmm1, c.ecx);
			// Unordered operands set CF, so AE is false for NaN.
			if (inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32) {
				c.ucomiss(c.xmm1, c.xmm0);
			} else {
				c.ucomiss(c.xmm0, c.xmm1);
			}
			Set(&Xbyak::CodeGenerator::setae);
			break;
		case ValueOpcode::BitwiseAnd32: c.and_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseAnd64: c.and_(c.rax, c.rcx); break;
		case ValueOpcode::BitwiseOr32: c.or_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseXor32: c.xor_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseNot32: c.not_(c.eax); break;
		case ValueOpcode::ShiftLeftLogical32: c.shl(c.eax, c.cl); break;
		case ValueOpcode::ShiftLeftLogical64: c.shl(c.rax, c.cl); break;
		case ValueOpcode::ShiftRightLogical32: c.shr(c.eax, c.cl); break;
		case ValueOpcode::ShiftRightLogical64: c.shr(c.rax, c.cl); break;
		case ValueOpcode::ShiftRightArithmetic32: c.sar(c.eax, c.cl); break;
		case ValueOpcode::ShiftRightArithmetic64: c.sar(c.rax, c.cl); break;
		case ValueOpcode::BitFieldUExtract:
			field(c.ecx, c.r9d);
			c.shr(c.rax, c.cl);
			c.and_(c.rax, c.r11);
			break;
		case ValueOpcode::BitFieldSExtract: {
			Xbyak::Label done;
			field(c.ecx, c.r9d);
			c.shr(c.rax, c.cl);
			c.and_(c.rax, c.r11);
			// Width zero tests bit 63, which is clear.
			c.lea(c.ecx, c.ptr[c.r9 - 1]);
			c.bt(c.rax, c.rcx);
			c.jnc(done, Near);
			c.not_(c.r11);
			c.or_(c.rax, c.r11);
			c.mov(c.eax, c.eax);
			c.L(done);
			break;
		}
		case ValueOpcode::BitFieldInsert:
			field(c.r9d, c.r11d);
			c.shl(c.r11, c.cl);
			c.shl(c.r10, c.cl);
			c.and_(c.r10, c.r11);
			c.not_(c.r11);
			c.and_(c.rax, c.r11);
			c.or_(c.rax, c.r10);
			c.mov(c.eax, c.eax);
			break;
		case ValueOpcode::IEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		case ValueOpcode::INotEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::setne);
			break;
		case ValueOpcode::IEqual64:
			c.cmp(c.rax, c.rcx);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		case ValueOpcode::INotEqual64:
			c.cmp(c.rax, c.rcx);
			Set(&Xbyak::CodeGenerator::setne);
			break;
		case ValueOpcode::UGreaterThan32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::seta);
			break;
		case ValueOpcode::SGreaterThanEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::setge);
			break;
		case ValueOpcode::LogicalXor:
			c.test(c.rax, c.rax);
			c.setne(c.al);
			c.test(c.rcx, c.rcx);
			c.setne(c.cl);
			c.xor_(c.al, c.cl);
			c.movzx(c.eax, c.al);
			break;
		case ValueOpcode::LogicalNot:
			c.test(c.rax, c.rax);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		default: EXIT("unexpected SRT walker opcode\n");
	}
	Store(node);
}

void WalkerCompiler::Compile(std::span<const Value> values) {
	auto&       c         = m_code;
	const auto& sources   = m_plan.descriptor_sources;
	const auto& blocks    = m_plan.control_flow;
	const bool  resources = values.empty();
	std::vector<std::vector<uint32_t>> gates(sources.size());
	if (resources) Consume(gates);
	for (const auto value: values) AddOutput(value, UINT32_MAX);
	// Analyze every root first, so thunk bodies only reuse finished operands.
	std::vector<Operand>              conditions, reads, outputs;
	std::vector<std::vector<Operand>> descriptors(sources.size());
	if (resources && m_plan.stage == ShaderType::Compute) {
		for (size_t slot = 0; slot < m_plan.srt_reads.size() && slot < m_plan.srt_read_host.size(); ++slot) {
			if (m_plan.srt_read_host[slot] == 0u) m_shader_only.insert(m_plan.srt_reads[slot].value.ResolveInstruction());
		}
	}
	if (resources) {
		for (const auto& block: blocks) conditions.push_back(Evaluate(block.condition, 0).operand);
		for (const auto& read: m_plan.srt_reads) reads.push_back(Evaluate(read.value, 0).operand);
		for (uint32_t source = 0; source < sources.size(); ++source) {
			if (gates[source].empty()) continue;
			for (uint32_t word = 0; word < sources[source].dword_count; ++word)
				descriptors[source].push_back(Evaluate(sources[source].dwords[word], 0).operand);
		}
	}
	for (const auto value: m_outputs) outputs.push_back(Evaluate(value, 0).operand);
	Layout();
	m_plan.srt_fill_address.assign(reads.size(), UINT32_MAX);
	for (size_t slot = 0; slot < reads.size(); ++slot) {
		if (reads[slot].kind != Kind::Node) continue;
		const auto& node = m_nodes[reads[slot].value];
		if (m_shader_only.contains(node.inst)) m_plan.srt_fill_address[slot] = node.address;
	}

	Xbyak::Label pass, fail, reset;
	c.mov(c.qword[c.rdx + m_shader_base], c.rcx);
	Zero(m_reach, m_states_end - m_reach);
	c.lea(c.r8, c.ptr[c.rdx + m_plan.srt_captures]);
	if (!resources) {
		for (uint32_t block = 0; block < blocks.size(); ++block) c.mov(c.byte[c.rdx + m_reach + block], 1);
	} else if (!blocks.empty()) {
		// Reachability grows monotonically. A newly reached predecessor of an evaluated
		// control Phi restarts the walk with fresh values; a retreating edge resumes at its target.
		std::vector<std::vector<uint32_t>> rechecks(blocks.size());
		for (uint32_t index = 0; index < m_nodes.size(); ++index) {
			const auto& node = m_nodes[index];
			if (node.inst->GetOpcode() != ValueOpcode::Phi || node.result != Operand {Kind::Node, index}) continue;
			for (const auto block: m_plan.control_phi_predecessors.at(node.inst)) rechecks[block].push_back(node.state);
		}
		std::vector<Xbyak::Label> starts(blocks.size());
		c.mov(c.byte[c.rdx + m_reach], 1);
		c.L(pass);
		for (uint32_t block = 0; block < blocks.size(); ++block) {
			Xbyak::Label next;
			const auto mark = [&](uint32_t target) {
				Xbyak::Label reached;
				c.cmp(c.byte[c.rdx + m_reach + target], 0);
				c.jne(reached, Near);
				c.mov(c.byte[c.rdx + m_reach + target], 1);
				for (const auto state: rechecks[target]) {
					c.cmp(c.byte[c.rdx + state], 0);
					c.jne(reset, Near);
				}
				if (target <= block) c.jmp(starts[target], Near);
				c.L(reached);
			};
			const auto& successors = blocks[block].successors;
			const auto  condition  = conditions[block];
			c.L(starts[block]);
			c.cmp(c.byte[c.rdx + m_reach + block], 0);
			c.je(next, Near);
			if (condition.kind == Kind::Constant) {
				mark(successors[condition.value != 0u ? 0 : 1]);
			} else {
				if (condition.kind != Kind::Unknown) {
					Xbyak::Label both, other;
					Fetch(condition, both);
					c.test(c.rax, c.rax);
					c.jz(other, Near);
					mark(successors[0]);
					c.jmp(next, Near);
					c.L(other);
					mark(successors[1]);
					c.jmp(next, Near);
					c.L(both);
				}
				for (const auto successor: successors) mark(successor);
			}
			c.L(next);
		}
	}
	if (resources) {
		const auto flat = [&](uint32_t slot) {
			const auto offset = slot < reads.size() ? m_plan.srt_reads[slot].flat_offset : UINT32_MAX;
			if (offset >= reads.size()) {
				c.jmp(fail, Near);
				return;
			}
			Fetch(reads[slot], fail);
			c.mov(c.dword[c.rsi + offset * sizeof(uint32_t)], c.eax);
		};
		if (blocks.empty()) {
			for (uint32_t slot = 0; slot < reads.size(); ++slot) flat(slot);
		}
		for (uint32_t block = 0; block < blocks.size(); ++block) {
			if (blocks[block].srt_reads.empty()) continue;
			Xbyak::Label skip;
			c.cmp(c.byte[c.rdx + m_reach + block], 0);
			c.je(skip, Near);
			for (const auto slot: blocks[block].srt_reads) flat(slot);
			c.L(skip);
		}
		// A source or image no block lists is always active.
		for (uint32_t index = 0; index < sources.size() + m_plan.info.images.size(); ++index) {
			const bool     image = index >= sources.size();
			const uint32_t id    = image ? index - static_cast<uint32_t>(sources.size()) : index;
			bool listed = false;
			for (uint32_t block = 0; block < blocks.size(); ++block) {
				const auto& list = image ? blocks[block].images : blocks[block].sources;
				if (std::ranges::find(list, id) == list.end()) continue;
				if (listed) {
					c.or_(c.al, c.byte[c.rdx + m_reach + block]);
				} else {
					c.mov(c.al, c.byte[c.rdx + m_reach + block]);
				}
				listed = true;
			}
			const auto active = c.byte[c.rdx + ActiveOffset(m_plan) + index];
			if (listed) {
				c.mov(active, c.al);
			} else {
				c.mov(active, 1);
			}
		}
		for (uint32_t source = 0; source < sources.size(); ++source) {
			if (gates[source].empty()) continue;
			Xbyak::Label skip;
			const auto known = c.dword[c.rdx + source * DescriptorStride + 8u * sizeof(uint32_t)];
			c.mov(known, 0);
			Gate(gates[source], skip);
			for (uint32_t word = 0; word < descriptors[source].size(); ++word) {
				Fetch(descriptors[source][word], skip);
				c.mov(c.dword[c.rdx + source * DescriptorStride + word * sizeof(uint32_t)], c.eax);
			}
			c.mov(known, 1);
			c.L(skip);
		}
	}
	for (uint32_t index = 0; index < outputs.size(); ++index) {
		Xbyak::Label skip;
		const auto output = c.qword[c.rdx + OutputOffset(m_plan) + index * 8u];
		c.mov(output, -1);
		Gate(m_output_gates[index], skip);
		Fetch(outputs[index], skip);
		c.mov(c.eax, c.eax);
		c.mov(output, c.rax);
		c.L(skip);
	}
	c.mov(c.rax, c.r8);
	c.L(m_ret);
	c.ret();
	c.L(fail);
	c.xor_(c.eax, c.eax);
	c.ret();
	if (resources && !blocks.empty()) {
		c.L(reset);
		Zero(m_states, m_states_end - m_states);
		c.lea(c.r8, c.ptr[c.rdx + m_plan.srt_captures]);
		c.jmp(pass, Near);
	}
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (Thunk(index)) EmitNode(index);
	}
	m_plan.srt_scratch.assign(m_plan.srt_captures / 8u + m_captures, 0u);
	m_plan.srt_user_data = m_user_data;
	c.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}

} // namespace

void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values) {
	plan.srt_code.reset();
	plan.srt_output_index.clear();
	if (values.empty() && (!plan.srt_plan_complete || !plan.resource_tracking_complete)) return;
	plan.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	WalkerCompiler(plan, *plan.srt_code).Compile(values);
}

bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads) {
	if (plan.srt_code == nullptr || runtime.user_data.size() < plan.srt_user_data) return false;
	flat.assign(plan.srt_reads.size(), 0u);
	using Walker = const uint64_t* KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*, uint64_t*, uint64_t);
	thread_local std::vector<uint64_t> fill_loads;
	fill_loads.clear();
	plan.srt_fills.clear();
	const auto gate = g_fill_gate.load(std::memory_order_relaxed);
	t_fill_loads     = !plan.srt_fill_address.empty() && gate != nullptr && gate() ? &fill_loads : nullptr;
	// A read the walk skips (out of bounds, or not reached) keeps no address from an earlier walk: 0 is never filled.
	for (const auto offset: plan.srt_fill_address) {
		if (offset != UINT32_MAX) plan.srt_scratch[offset / 8u] = 0;
	}
	t_running_walker = plan.srt_code.get();
	t_walker_faulted = false;
	const auto* end = plan.srt_code->getCode<Walker>()(runtime.user_data.data(), flat.data(),
	                                                     plan.srt_scratch.data(), runtime.shader_base);
	t_running_walker = nullptr;
	t_fill_loads     = nullptr;
	if (end == nullptr || t_walker_faulted) return false;
	// A load completed as 0 is the word of each read whose address the walk kept there.
	for (size_t slot = 0; !fill_loads.empty() && slot < plan.srt_fill_address.size(); ++slot) {
		const auto offset = plan.srt_fill_address[slot];
		if (offset == UINT32_MAX) continue;
		const auto address = plan.srt_scratch[offset / 8u];
		if (std::ranges::find(fill_loads, address) != fill_loads.end())
			plan.srt_fills.emplace_back(address, plan.srt_reads[slot].flat_offset);
	}
	plan.srt_loads = static_cast<uint32_t>(end - (plan.srt_scratch.data() + plan.srt_captures / 8u));
	if (reads != nullptr) WalkedLoads(plan, *reads);
	return true;
}

void WalkedLoads(const ResourcePlan& plan, std::vector<std::pair<uint64_t, uint64_t>>& reads) {
	const auto* loads = plan.srt_scratch.data() + plan.srt_captures / 8u;
	for (const auto* read = loads; read != loads + plan.srt_loads; ++read) reads.emplace_back(*read, sizeof(uint32_t));
}

bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result) {
	value = value.Resolve();
	uint64_t bits = 0;
	if (value.IsImmediate()) {
		if (!ImmediateBits(value, bits)) return false;
		result = static_cast<uint32_t>(bits);
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return false;
	const auto output = WalkOutput(plan, *inst);
	if (output == UINT32_MAX || OutputOffset(plan) + output * 8u >= plan.srt_outputs) return false;
	std::memcpy(&bits, Scratch(plan) + OutputOffset(plan) + output * 8u, sizeof(bits));
	result = static_cast<uint32_t>(bits);
	return bits <= UINT32_MAX;
}

bool WalkedDescriptor(const ResourcePlan& plan, uint32_t source, DescriptorValue& result) {
	if (source >= plan.descriptor_sources.size()) return false;
	std::array<uint32_t, 9> words {};
	std::memcpy(words.data(), Scratch(plan) + source * DescriptorStride, DescriptorStride);
	if (words[8] == 0u) return false;
	result = {};
	result.dword_count = plan.descriptor_sources[source].dword_count;
	std::copy_n(words.begin(), result.dword_count, result.dwords.begin());
	return true;
}

bool WalkedActive(const ResourcePlan& plan, uint32_t source) {
	return Scratch(plan)[ActiveOffset(plan) + source] != 0u;
}

bool WalkedImageActive(const ResourcePlan& plan, uint32_t image) {
	return Scratch(plan)[ActiveOffset(plan) + plan.descriptor_sources.size() + image] != 0u;
}

bool WalkedReached(const ResourcePlan& plan, uint32_t block) {
	return Scratch(plan)[plan.srt_reach + block] != 0u;
}

uint32_t WalkOutput(const ResourcePlan& plan, const Inst& inst) {
	const auto index = inst.EvaluationIndex(plan.evaluation_value_count);
	return index < plan.srt_output_index.size() ? plan.srt_output_index[index] : UINT32_MAX;
}

void SetWalkOutput(const ResourcePlan& plan, const Inst& inst, uint32_t output) {
	const auto index = inst.EvaluationIndex(plan.evaluation_value_count);
	if (index >= plan.srt_output_index.size()) plan.srt_output_index.resize(index + 1u, UINT32_MAX);
	plan.srt_output_index[index] = output;
}

uint32_t SrtWalkerLoadLength(uint64_t rip) {
	if (t_running_walker == nullptr ||
	    rip - reinterpret_cast<uint64_t>(t_running_walker->getCode()) >= t_running_walker->getSize())
		return 0;
	const auto* code = reinterpret_cast<const uint8_t*>(rip);
	if (code[0] != 0x41u || code[1] != 0x8bu) return 0;
	if (code[2] == 0x02u) return 3;
	return code[2] == 0x42u && code[3] == 0x00u ? 4 : 0;
}

void SetSrtGpuFills(SrtFillGate gate) {
	g_fill_gate.store(gate, std::memory_order_relaxed);
}

bool SrtWalkerShaderOnlyLoad(uint64_t rip) {
	return SrtWalkerLoadLength(rip) == 4u;
}

bool SrtWalkerFillLoad(uint64_t address) {
	// A null pointer's load reads zero (SrtWalkerFaultedLoad), never a GPU copy from that address.
	if (t_fill_loads == nullptr || address < NullPointerLoads) return false;
	t_fill_loads->push_back(address);
	return true;
}

void SrtWalkerFaultedLoad(uint64_t address) {
	if (address >= NullPointerLoads) t_walker_faulted = true;
}

WalkComparison CompareWalk(const ResourcePlan& plan, const SrtRuntime& runtime) {
	WalkComparison result;
	const auto differ = [&](uint32_t& count, std::string what) {
		if (result.first.empty()) result.first = std::move(what);
		++count;
	};
	auto reference = runtime;
	reference.direct_reads = false;
	SrtWalker clean(plan, CleanRuntime(reference));
	SrtWalker walker(plan, reference, &clean);
	std::vector<uint32_t> flat, walked_flat;
	result.refreshed   = walker.RefreshFlatBuffer(flat);
	const auto visited = plan.visited_blocks;
	const auto active  = plan.active_sources;
	const auto images  = plan.active_images;
	std::vector<std::pair<uint64_t, uint64_t>> loads;
	result.walked = plan.srt_code != nullptr && WalkSrt(plan, runtime, walked_flat, &loads);
	if (!result.walked || !result.refreshed) {
		if (result.walked) differ(result.walker_only, "only the walker walked");
		if (result.refreshed) ++result.interpreter_only;
		return result;
	}
	if (walked_flat != flat) {
		for (size_t slot = 0; slot < flat.size(); ++slot) {
			if (slot < walked_flat.size() && walked_flat[slot] == flat[slot]) continue;
			const auto* read = slot < plan.srt_reads.size() ? plan.srt_reads[slot].value.Resolve().TryInstruction() : nullptr;
			std::string addresses;
			for (size_t load = 0; load < loads.size() && load < 6u; ++load)
				addresses += fmt::format(" {:#x}", loads[load].first);
			differ(result.mismatches,
			       fmt::format("flat[{}] walker {:#x} interpreter {:#x} (read opcode {}, loads{})", slot,
			                   slot < walked_flat.size() ? walked_flat[slot] : 0u, flat[slot],
			                   read != nullptr ? static_cast<uint32_t>(read->GetOpcode()) : 0u, addresses));
			break;
		}
	}
	if (!plan.control_flow.empty()) {
		for (uint32_t block = 0; block < plan.control_flow.size(); ++block) {
			const bool reached = WalkedReached(plan, block);
			const bool interpreted = block < visited.size() && visited[block] != 0u;
			if (reached == interpreted) continue;
			if (!reached) {
				differ(result.reach_smaller, fmt::format("block {} unreached by the walker only", block));
			} else {
				differ(result.mismatches, fmt::format("block {} reached by the walker only", block));
			}
		}
		for (uint32_t source = 0; source < plan.descriptor_sources.size(); ++source) {
			const bool interpreted = source >= active.size() || active[source] != 0u;
			if (WalkedActive(plan, source) != interpreted && result.reach_smaller == 0u)
				differ(result.mismatches, fmt::format("source {} activity", source));
		}
		for (uint32_t image = 0; image < plan.info.images.size(); ++image) {
			const bool interpreted = image >= images.size() || images[image] != 0u;
			if (WalkedImageActive(plan, image) != interpreted && result.reach_smaller == 0u)
				differ(result.mismatches, fmt::format("image {} activity", image));
		}
	}
	// Values both evaluate under the same reached blocks; with fewer blocks the walker's Phis may know more.
	for (uint32_t source = 0; source < plan.descriptor_sources.size(); ++source) {
		DescriptorValue walked_value, value;
		const bool walked = WalkedDescriptor(plan, source, walked_value);
		const bool used   = walked || plan.control_flow.empty() || source >= active.size() || active[source] != 0u;
		if (!used) continue;
		const bool known = walker.EvaluateDescriptor(source, value) || clean.EvaluateDescriptor(source, value);
		if (walked && !known) {
			differ(result.walker_only, fmt::format("descriptor {} known to the walker only", source));
		} else if (!walked && known) {
			++result.interpreter_only;
		} else if (walked && walked_value.dwords != value.dwords) {
			differ(result.mismatches, fmt::format("descriptor {} dword0 walker {:#x} interpreter {:#x}", source,
			                                      walked_value.dwords[0], value.dwords[0]));
		}
	}
	const auto compare_value = [&](Value value, std::string_view name) {
		const auto* inst = value.Resolve().TryInstruction();
		if (inst == nullptr || WalkOutput(plan, *inst) == UINT32_MAX) return;
		uint32_t walked_value = 0, interpreted = 0;
		const bool walked = WalkedValue(plan, value, walked_value);
		const bool known  = clean.Evaluate(value, interpreted);
		if (walked && !known) {
			differ(result.walker_only, fmt::format("{} known to the walker only", name));
		} else if (!walked && known) {
			++result.interpreter_only;
		} else if (walked && walked_value != interpreted) {
			differ(result.mismatches, fmt::format("{} walker {:#x} interpreter {:#x}", name, walked_value, interpreted));
		}
	};
	for (const auto& source: plan.descriptor_sources) {
		if (!source.indirect_descriptor) continue;
		compare_value(source.indirect_descriptor->key_count, "key_count");
		compare_value(source.indirect_descriptor->selector_first, "selector_first");
		compare_value(source.indirect_descriptor->selector_mask, "selector_mask");
	}
	compare_value(plan.uniform_fill.pattern_size, "fill pattern_size");
	compare_value(plan.uniform_fill.element_count, "fill element_count");
	for (const auto value: plan.uniform_fill.values) compare_value(value, "fill value");
	return result;
}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     SrtWalker* clean_evaluator)
    : m_program(program), m_runtime(runtime), m_clean_evaluator(clean_evaluator),
      m_session(this),
      m_read_policy(clean_evaluator != nullptr ? 2u : 0u), m_context(AcquireContext()) {}

SrtWalker::SrtWalker(SrtWalker& parent, Value active_mask, bool clean,
                     SrtWalker* clean_evaluator)
    : m_program(parent.m_program), m_runtime(clean ? CleanRuntime(parent.m_runtime) : parent.m_runtime),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_reachable_blocks(parent.m_reachable_blocks), m_session(parent.m_session),
      m_read_policy((parent.m_read_policy & 1u) | (clean ? 1u : 0u) |
                    (clean_evaluator != nullptr ? 2u : 0u)), m_context(AcquireContext()), m_record(parent.m_record) {}

SrtWalker::~SrtWalker() {
	if (m_session != this) return;
	if (m_session_contexts == 1u) {
		m_context.session = nullptr;
		return;
	}
	for (auto& context: m_program.evaluation_contexts)
		if (context.session == this) context.session = nullptr;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	if (m_walked && WalkedValue(m_program, value, result)) return true;
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext() {
	// A root walker starts its session: none of the contexts is the session's yet (a session releases its own when
	// it ends), so it takes the first free one without looking for a match. A plan's scratch can hold well over a
	// hundred nested contexts, and MaterializeResources starts two sessions a call.
	const bool root = m_session == this;
	ResourcePlan::EvaluationContext* unused = nullptr;
	for (auto& context: m_program.evaluation_contexts) {
		if (!root && context.session == m_session && context.active_mask == m_active_mask &&
		    context.read_policy == m_read_policy) return context;
		if (context.session == nullptr && unused == nullptr) {
			unused = &context;
			if (root) break;
		}
	}
	++m_session->m_session_contexts;
	auto& context = unused != nullptr ? *unused : m_program.evaluation_contexts.emplace_back();
	context.session = m_session;
	context.active_mask = m_active_mask;
	context.read_policy = m_read_policy;
	context.generation += 4;
	return context;
}

void SrtWalker::ResetUnknowns() {
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session != m_session) continue;
		for (auto& entry: context.values)
			if (entry.generation == (context.generation | 2u)) entry.generation = 0;
	}
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	m_used = true;
	value  = value.Resolve();
	if (value.IsImmediate()) {
		bool ok = true;
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); break;
			case Type::U8: result = value.U8(); break;
			case Type::U16: result = value.U16(); break;
			case Type::U32: result = value.U32(); break;
			case Type::U64: result = value.U64(); break;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); break;
			default: ok = false; break;
		}
		if (m_record != nullptr) m_record->Leaf(ok ? m_record->Immediate(result) : TraceFailed);
		return ok;
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		if (m_record != nullptr) m_record->Leaf(m_record->Immediate(1u));
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		if (m_record != nullptr) m_record->Leaf(TraceFailed);
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		if (m_record != nullptr) m_record->Leaf(m_record->Memoized(m_context, index));
		return true;
	}
	// Visiting and completed-unknown entries both stop repeated evaluation in this
	// context. Only completed unknowns are retried when reachability becomes tighter.
	if (m_context.values[index].generation == (m_context.generation | 1u) ||
	    m_context.values[index].generation == (m_context.generation | 2u)) {
		if (m_record != nullptr) m_record->Leaf(TraceFailed);
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	if (m_record != nullptr) m_record->Begin();
	const bool evaluated = EvaluateInst(*inst, out);
	if (m_record != nullptr) m_record->End(*this, *inst, index, evaluated, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = m_context.generation | 2u;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

void SrtWalker::Form(uint8_t form, uint32_t imm) {
	if (m_record != nullptr) m_record->Form(form, imm);
}

void SrtWalker::Check(bool zero, bool low32) {
	if (m_record != nullptr) m_record->Check(zero, low32);
}

void SrtWalker::CheckEqual(bool equal) {
	if (m_record != nullptr) m_record->CheckEqual(equal);
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	if (const auto control = m_program.control_phi_predecessors.find(&inst);
	    control != m_program.control_phi_predecessors.end()) {
		bool found = false;
		Form(FormFixed);
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			if (m_reachable_blocks != nullptr &&
			    !m_reachable_blocks->at(control->second[index])) continue;
			uint64_t incoming = 0;
			if (!Arg(inst, index, incoming)) return false;
			if (found) CheckEqual(incoming == result);
			if (found && incoming != result) return false;
			result = incoming;
			found = true;
		}
		if (found) Form(FormPassFirst);
		return found;
	}
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	Form(value.IsEmpty() ? FormFixed : FormPassLast);
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		Form(FormExtract64, component);
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		Form(FormPassLast);
		return EvaluateWide(source->Arg(component), result);
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<SrtReadFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto guard = vector ? inst.Arg(4).Resolve() : Value {};
	uint32_t   guarded = 0;
	Form(FormFixed);
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		Check(enabled == 0u);
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
		guarded = RawReadGuarded;
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	Form(FormRawRead, flags.index | guarded);
	std::array<uint64_t, 5> words {}; // low, high, offset, records, word3
	if (!Arg(*handle, 0, words[0]) || !Arg(*handle, 1, words[1]) || !Arg(inst, 1, words[2])) {
		return false;
	}
	if ((inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector) &&
	    (handle->NumArgs() != 4u || !Arg(*handle, 2, words[3]) || !Arg(*handle, 3, words[4]))) {
		return false;
	}
	return ReadRawSlot(m_program.memory_info[flags.index], inst.GetOpcode(), words.data(), m_runtime, result);
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	uint64_t   a  = 0;
	uint64_t   b  = 0;
	const auto op = inst.GetOpcode();
	switch (op) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base) {
				return false;
			}
			Form(FormUserData, reg - m_program.user_data_base);
			if (reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase:
			Form(FormShaderBase);
			result = m_runtime.shader_base;
			return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			Form(FormPassLast);
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			SrtWalker clean_active(*this, inst.Arg(1), true);
			SrtWalker active(*this, inst.Arg(1), false, &clean_active);
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConditionRef: Form(FormPassLast); return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (m_clean_evaluator != nullptr &&
				    (inst.Flags<SrtReadFlags>().clean != 0u ||
				     (op == ValueOpcode::LoadBufferU32 && m_clean_evaluator->m_active_mask == m_active_mask))) {
					Form(FormPassLast);
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			Form(FormFixed);
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!predicate.EvaluateWide(inst.Arg(0), a)) {
				return false;
			}
			Check(a == 0u);
			Form(FormPassLast);
			return Arg(inst, a != 0u ? 1u : 2u, result);
		}
		case ValueOpcode::ULessThan32:
			// No U32 is below zero, including an unknown loop-carried value.
			Form(FormFixed);
			if (!Arg(inst, 1, b)) return false;
			Check(static_cast<uint32_t>(b) == 0u, true);
			if (static_cast<uint32_t>(b) == 0u) {
				result = 0u;
				return true;
			}
			Form(FormPureSwapped);
			if (Arg(inst, 0, a)) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThanEqual32:
			// A zero lower bound accepts every U32, including a varying lane value.
			Form(FormFixed);
			if (!Arg(inst, 0, a)) return false;
			Check(a == 0u);
			if (a == 0u) {
				result = 1u;
				return true;
			}
			Form(FormPure);
			if (Arg(inst, 1, b)) {
				result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd: {
			Form(FormFixed);
			const bool left = Arg(inst, 0, a);
			if (left) Check(a == 0u);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			const bool right = Arg(inst, 1, b);
			if (right) Check(b == 0u);
			if (right && (b == 0u || left)) {
				result = b != 0u;
				return true;
			}
			if (!left) {
				// Only lanes satisfying the left EXEC factor can make the conjunction
				// true. Use a separate memo so this assumption cannot escape its RHS.
				SrtWalker active(m_clean_evaluator != nullptr ? *m_clean_evaluator : *this,
				                 inst.Arg(0));
				const bool masked = active.EvaluateWide(inst.Arg(1), b);
				if (masked) Check(b == 0u);
				if (masked && b == 0u) {
					result = 0u;
					return true;
				}
			}
			return false;
		}
		case ValueOpcode::LogicalOr: {
			Form(FormFixed);
			const bool left = Arg(inst, 0, a);
			if (left) Check(a == 0u);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			const bool right = Arg(inst, 1, b);
			if (right) Check(b == 0u);
			if (!right || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU32: return false;
		default: {
			const auto arity = PureArity(op);
			if (arity == 0u) break;
			std::array<uint64_t, 4> args {};
			for (uint32_t index = 0; index < arity; ++index) {
				if (!Arg(inst, index, args[index])) return false;
			}
			const bool flush = (op == ValueOpcode::FPOrdLessThanEqual32 || op == ValueOpcode::FPOrdGreaterThanEqual32) &&
			                   inst.Flags<FPCompareFlags>().flush_input_denorms;
			return ApplyPureOp(op, flush, args.data(), result);
		}
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	if (m_walked && WalkedDescriptor(m_program, source, result)) return true;
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	const auto* fetch = m_flat != nullptr && source < m_program.descriptor_fetch.size()
	                        ? m_program.descriptor_fetch[source].data() : nullptr;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (fetch != nullptr && Fetch(fetch[index], result.dwords[index])) continue;
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

// What Evaluate gives for the dword after this session's refresh, without evaluating it: a read the refresh made
// stands in the flattened SRT as the walker's memo holds it (the interpreter, the replay's seeds and the JIT walker all
// leave it there), a user-data word and an immediate are what EvaluateInst returns. False: evaluate it.
bool SrtWalker::Fetch(const ResourcePlan::DescriptorFetch& fetch, uint32_t& result) const {
	switch (fetch.kind) {
		case ResourcePlan::DescriptorFetch::Flat: {
			if (m_flat_clean_only && fetch.read->Flags<SrtReadFlags>().clean == 0u) return false;
			// With control flow only the visited blocks' reads were refreshed.
			if (fetch.block != UINT32_MAX &&
			    (fetch.block >= m_program.visited_blocks.size() || !m_program.visited_blocks[fetch.block]))
				return false;
			if (fetch.index >= m_flat->size()) return false;
			result = (*m_flat)[fetch.index];
			return true;
		}
		case ResourcePlan::DescriptorFetch::UserData:
			if (fetch.index >= m_runtime.user_data.size()) return false;
			result = m_runtime.user_data[fetch.index];
			return true;
		case ResourcePlan::DescriptorFetch::Immediate: result = fetch.index; return true;
		default: return false;
	}
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	m_flat = nullptr;
	if (m_clean_evaluator != nullptr) m_clean_evaluator->m_flat = nullptr;
	if (!Refresh(flat)) return false;
	m_flat = &flat;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->m_flat            = &flat;
		m_clean_evaluator->m_flat_clean_only = true;
	}
	return true;
}

void SrtWalker::AdoptWalk(const std::vector<uint32_t>& flat) {
	auto& visited = m_program.visited_blocks;
	auto& active  = m_program.active_sources;
	auto& images  = m_program.active_images;
	if (m_program.control_flow.empty()) {
		visited.clear();
		active.clear();
		images.clear();
	} else {
		// As Interpret leaves them: every source and image a reached block lists, or no block lists, is active.
		visited.resize(m_program.control_flow.size());
		for (uint32_t block = 0; block < visited.size(); ++block) visited[block] = WalkedReached(m_program, block);
		active.resize(m_program.descriptor_sources.size());
		for (uint32_t source = 0; source < active.size(); ++source) active[source] = WalkedActive(m_program, source);
		images.resize(m_program.info.images.size());
		for (uint32_t image = 0; image < images.size(); ++image) images[image] = WalkedImageActive(m_program, image);
	}
	for (auto* walker: {this, m_clean_evaluator}) {
		if (walker == nullptr) continue;
		walker->m_walked           = true;
		walker->m_flat             = &flat;
		walker->m_flat_clean_only  = false; // direct reads: the flat words are what the strict reader gives
		walker->m_reachable_blocks = m_program.control_flow.empty() ? nullptr : &visited;
	}
}

bool SrtWalker::Refresh(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	// A replay gives what the interpreter gives from fresh memos, as MaterializeResources' walkers start.
	if (!m_program.srt_trace || m_record != nullptr || m_used ||
	    (m_clean_evaluator != nullptr && m_clean_evaluator->m_used)) return Interpret(flat);
	if (m_program.trace_skip != 0u) {
		--m_program.trace_skip;
		return Interpret(flat);
	}
	// A replay reads clean values through CleanRuntime of this walker's runtime, as
	// MaterializeResources sets its clean evaluator up; another setup is only interpreted.
	const auto same_readers = [](const SrtRuntime& a, const SrtRuntime& b) {
		return a.read_memory == b.read_memory && a.read_specialization_memory == b.read_specialization_memory &&
		       a.userdata == b.userdata && a.user_data.data() == b.user_data.data() &&
		       a.user_data.size() == b.user_data.size() && a.shader_base == b.shader_base;
	};
	if (m_clean_evaluator != nullptr && !same_readers(m_clean_evaluator->m_runtime, CleanRuntime(m_runtime)))
		return Interpret(flat);
	auto&         trace = m_program.trace;
	const uint8_t key   = (m_clean_evaluator != nullptr ? 1u : 0u) |
	                      (m_runtime.read_specialization_memory != nullptr ? 2u : 0u);
	const auto source       = m_runtime;
	const auto clean_source = m_clean_evaluator != nullptr ? m_clean_evaluator->m_runtime : SrtRuntime {};
	ReadShim   shim(source);
	const bool replayed = trace.valid && trace.key == key;
	if (replayed) {
		if (Replay(flat, shim.Runtime())) {
			m_program.trace_misses = 0;
			++m_program.trace_replays;
			return true;
		}
		// The interpreter takes over and is given the replay's reads first.
		shim.Serve();
		m_runtime = shim.Runtime();
		if (m_clean_evaluator != nullptr) m_clean_evaluator->m_runtime = CleanRuntime(m_runtime);
	}
	SrtTraceRecorder recorder(trace, m_runtime);
	m_record = &recorder;
	if (m_clean_evaluator != nullptr) m_clean_evaluator->m_record = &recorder;
	const bool ok = Interpret(flat);
	m_record = nullptr;
	if (m_clean_evaluator != nullptr) m_clean_evaluator->m_record = nullptr;
	if (replayed) {
		m_runtime = source;
		if (m_clean_evaluator != nullptr) m_clean_evaluator->m_runtime = clean_source;
	}
	if (ok) recorder.Finish(*this, key);
	// A plan whose refreshes keep taking other paths is interpreted for a while without recording.
	if (replayed && ++m_program.trace_misses >= 4u) {
		m_program.trace_misses = 0;
		m_program.trace_skip   = 256;
		trace.valid            = false;
	}
	return ok;
}

bool SrtWalker::Interpret(std::vector<uint32_t>& flat) {
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const auto* inst = read.value.ResolveInstruction();
		const bool clean = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (read.flat_offset >= flat.size() || !evaluator.Evaluate(read.value, flat[read.flat_offset])) return false;
		if (m_record != nullptr) m_record->Flat(read.flat_offset);
		return true;
	};
	auto& active = m_program.active_sources;
	auto& active_images = m_program.active_images;
	if (m_program.control_flow.empty()) {
		active.clear();
		active_images.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	auto& visited = m_program.visited_blocks;
	auto& reachable = m_program.reachable_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 1u);
	ResetUnknowns();
	m_reachable_blocks = &visited;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->ResetUnknowns();
		m_clean_evaluator->m_reachable_blocks = &visited;
	}
	for (;;) {
		reachable.assign(m_program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reachable.at(index)) continue;
			reachable[index] = 1u;
			const auto& block = m_program.control_flow[index];
			uint32_t condition = 0;
			bool     known     = false;
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!block.condition.IsEmpty() &&
			    (m_runtime.read_specialization_memory != nullptr || !m_program.capture_specialization_reads)) {
				known = predicate.Evaluate(block.condition, condition);
				if (m_record != nullptr) m_record->Condition(known, condition != 0u);
			}
			if (known) {
				pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		if (reachable == visited) break;
		visited.swap(reachable);
		if (m_program.control_phi_predecessors.empty()) break;
		// Only unknown values need another evaluation: successful Phi facts remain
		// true as their possible predecessors shrink. Keep uniform-read memos stable.
		ResetUnknowns();
		if (m_clean_evaluator != nullptr) m_clean_evaluator->ResetUnknowns();
	}
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	active_images.assign(m_program.info.images.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) active.at(source) = 0u;
		for (const auto image: block.images) active_images.at(image) = 0u;
	}
	for (uint32_t index = 0; index < visited.size(); ++index) {
		if (!visited[index]) continue;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (const auto image: block.images) active_images[image] = 1u;
		for (const auto slot: block.srt_reads)
			if (!refresh(slot)) return false;
	}
	return true;
}

// The plan's trace over this refresh's inputs, reading through runtime (and its clean runtime);
// false as soon as a check fails, before reading anything the interpreter would not.
bool SrtWalker::Replay(std::vector<uint32_t>& flat, const SrtRuntime& runtime) {
	const auto& trace  = m_program.trace;
	auto&       values = m_program.trace_values;
	auto&       oks    = m_program.trace_ok;
	values.resize(trace.ops.size());
	oks.resize(trace.ops.size());
	const auto clean = CleanRuntime(runtime);
	const auto get   = [&](int32_t ref, uint64_t& value) {
		if (ref >= 0) {
			value = values[static_cast<size_t>(ref)];
			return oks[static_cast<size_t>(ref)] != 0u;
		}
		if (ref == TraceFailed) return false;
		value = trace.immediates[static_cast<size_t>(-1 - ref)];
		return true;
	};
	std::array<uint64_t, 5> args {};
	const auto operands = [&](const ResourcePlan::TraceOp& op) {
		for (uint32_t index = 0; index < op.count; ++index) {
			if (!get(op.args[index], args[index])) return false;
		}
		return true;
	};
	if (m_program.control_flow.empty()) {
		flat.resize(m_program.srt_reads.size());
	} else {
		flat.assign(m_program.srt_reads.size(), 0u);
	}
	for (size_t index = 0; index < trace.ops.size(); ++index) {
		const auto& op    = trace.ops[index];
		uint64_t    value = 0;
		bool        ok    = false;
		switch (op.kind) {
			case TracePure:
				ok = operands(op);
				// With every operand there the interpreter would have asked for the next one.
				if (ok && op.count < op.imm) return false;
				ok = ok && ApplyPureOp(op.opcode, op.flags != 0u, args.data(), value);
				break;
			case TracePass: ok = get(op.args[0], value); break;
			case TraceFixed:
				ok    = op.ok;
				value = op.value;
				break;
			case TraceUserData:
				ok = op.imm < runtime.user_data.size();
				if (ok) value = runtime.user_data[op.imm];
				break;
			case TraceShaderBase:
				ok    = true;
				value = runtime.shader_base;
				break;
			case TraceRawRead: {
				const uint32_t needed = op.opcode == ValueOpcode::LoadAddressU32 ? 3u : 5u;
				ok = operands(op);
				if (ok && op.count < needed && op.count != 3u) return false;
				// Three words of a buffer: its handle has no record count (a failure the interpreter knew).
				ok = ok && op.count >= needed &&
				     ReadRawSlot(m_program.memory_info[op.imm], op.opcode, args.data(),
				                 (op.flags & 1u) != 0u ? clean : runtime, value);
				break;
			}
			case TraceCheckZero: {
				uint64_t operand = 0;
				if (!get(op.args[0], operand)) return false;
				const bool zero = (op.flags & 2u) != 0u ? static_cast<uint32_t>(operand) == 0u : operand == 0u;
				if (zero != ((op.flags & 1u) != 0u)) return false;
				continue;
			}
			case TraceCheckEqual: {
				uint64_t first = 0;
				uint64_t last  = 0;
				if (!get(op.args[0], first) || !get(op.args[1], last) || (first == last) != (op.flags != 0u))
					return false;
				continue;
			}
			case TraceCondition: {
				uint64_t   condition = 0;
				const bool known     = get(op.args[0], condition);
				if (known != op.ok || (known && (static_cast<uint32_t>(condition) != 0u) != (op.flags != 0u)))
					return false;
				continue;
			}
			case TraceFlat: {
				uint64_t word = 0;
				if (!get(op.args[0], word)) return false;
				flat[op.imm] = static_cast<uint32_t>(word);
				continue;
			}
			default: return false;
		}
		if (ok != op.ok) return false;
		values[index] = value;
		oks[index]    = ok ? 1u : 0u;
	}
	if (!m_program.control_flow.empty()) {
		m_program.visited_blocks = trace.visited;
		m_reachable_blocks       = &m_program.visited_blocks;
		if (m_clean_evaluator != nullptr) m_clean_evaluator->m_reachable_blocks = &m_program.visited_blocks;
	}
	m_program.active_sources = trace.active;
	m_program.active_images  = trace.active_images;
	// The memos as the interpreted refresh left them, for the evaluations after the refresh.
	for (const auto& seed: trace.seeds) {
		auto& context = seed.context == 0u ? m_context : m_clean_evaluator->m_context;
		if (context.values.size() < m_program.evaluation_value_count)
			context.values.resize(m_program.evaluation_value_count);
		auto& entry      = context.values[seed.index];
		entry.generation = seed.unknown ? (context.generation | 2u) : context.generation;
		if (!seed.unknown) entry.value = values[static_cast<size_t>(seed.ref)];
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool ValidateWaveUniformValue(const ResourcePlan& program, Value value) {
	return RuntimeValidator(program, RuntimeValueType::Any, true).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
