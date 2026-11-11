#ifndef KYTY_LINEAR_SRT_H
#define KYTY_LINEAR_SRT_H

#include "common/abi.h"
#if defined(__x86_64__) || defined(_M_X64)
#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>
#endif
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "jit-arena.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <map>
#include <set>

namespace Libs::Graphics::ShaderRecompiler::IR {

// One immutable program for a complete descriptor/flat transaction. Integer
// expressions become native code, shared reads execute once in their original
// order, and no evaluator callback or IR lookup occurs during execution.
struct LinearSrtPlan {
	enum class Kind { Immediate, User, Base, Unary, Binary, Select, Read };
	struct Node {
		Kind                    kind      = Kind::Immediate;
		ValueOpcode             op        = ValueOpcode::Void;
		uint64_t                immediate = 0;
		std::array<uint32_t, 5> args {};
		bool                    clean = false;
	};
	std::vector<Node>                                 nodes;
	struct ReadGroup {
		std::array<uint32_t, 16> indices {};
		uint32_t                 count = 0;
	};
	std::vector<ReadGroup>                            read_groups;
	std::vector<uint32_t>                             descriptor_words, flat_words, sources;
	std::vector<uint8_t>                              clean_slots, descriptor_sizes;
	std::vector<uint8_t>                              active_sources;
	std::vector<std::shared_ptr<const LinearSrtPlan>> control_variants;
	uint32_t                                          active_count = 0;
	// The generated register convention is explicit on every host, including
	// Windows. C++ helpers bridge to the host ABI of reader callbacks.
	using Function    = KYTY_SYSV_ABI bool (*)(const SrtRuntime*, uint64_t*);
	Function function = nullptr;
    std::shared_ptr<const LinearSrtPlan> single_condition;
    std::array<uint32_t, 3> single_condition_variants{};
#if defined(__x86_64__) || defined(_M_X64)
	std::unique_ptr<Xbyak::CodeGenerator> code;
#endif

	// Keep exact scalar bounds and retain ordered scalar fallback on a rejected span.
	static bool ReadAddress(const Node* node, const uint64_t* values, uint64_t& address) {
		const auto     low = values[node->args[0]], high = values[node->args[1]];
		const auto     offset = values[node->args[2]];
		const uint64_t base   = ((high << 32) | static_cast<uint32_t>(low)) & 0x0000ffffffffffffull;
		const auto     immediate = static_cast<int64_t>(static_cast<int32_t>(node->immediate));
		if (node->op == ValueOpcode::ReadConstBuffer) {
			const auto records = static_cast<uint32_t>(values[node->args[3]]);
			if (immediate < 0) return false;
			const auto     byte_offset = uint64_t(immediate) + static_cast<uint32_t>(offset);
			const auto     aligned     = byte_offset & ~uint64_t {3};
			const auto     stride      = (static_cast<uint32_t>(high) >> 16) & 0x3fff;
			const uint64_t size        = stride == 0 ? records : uint64_t(stride) * records;
			if (aligned > size || size - aligned < 4) return false;
			address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
		} else {
			const auto relative =
			    (immediate & ~int64_t {3}) + int64_t(static_cast<uint32_t>(offset) & ~3u);
			const auto aligned_base = base & ~uint64_t {3};
			if (relative < 0) {
				if (uint64_t(-relative) > aligned_base) return false;
				address = aligned_base - uint64_t(-relative);
			} else {
				if (uint64_t(relative) > 0x0000ffffffffffffull - aligned_base) return false;
				address = aligned_base + uint64_t(relative);
			}
		}
		return true;
	}
	static bool KYTY_SYSV_ABI Read(const SrtRuntime* runtime, const Node* node,
	                               const uint64_t* values, uint64_t* output) {
		uint64_t address = 0;
		if (!ReadAddress(node, values, address)) return false;
		uint32_t   value = 0;
		const auto reader =
		    node->clean ? runtime->read_specialization_memory : runtime->read_memory;
		if (reader) {
			if (!reader(runtime->userdata, address, &value)) return false;
		} else {
			if (node->clean) return false;
			std::memcpy(&value, reinterpret_cast<const void*>(address), 4);
		}
		*output = value;
		return true;
	}
	static bool KYTY_SYSV_ABI ReadSpan(const SrtRuntime* runtime, uint64_t address, uint32_t* words,
	                                   uint32_t count, bool clean) {
		return runtime->try_read_memory_span(runtime->userdata, address, words, count, clean);
	}
};

#if defined(__x86_64__) || defined(_M_X64)
class LinearSrtCompiler {
	const ResourcePlan&                              source;
	LinearSrtPlan&                                   result;
	std::span<const uint8_t>                         active_sources;
	std::map<std::pair<const Inst*, bool>, uint32_t> known;
	std::set<std::pair<const Inst*, bool>>           visiting;
	static constexpr uint32_t                        Invalid = UINT32_MAX;
	using Kind                                               = LinearSrtPlan::Kind;
	using Node                                               = LinearSrtPlan::Node;

	bool SameArgument(uint32_t a, uint32_t b) const {
		return a == b || (result.nodes[a].kind == Kind::Immediate &&
		                  result.nodes[b].kind == Kind::Immediate &&
		                  result.nodes[a].immediate == result.nodes[b].immediate);
	}
	bool ConsecutiveRead(const Node& a, const Node& b) const {
		if (a.op != b.op || !SameArgument(a.args[0], b.args[0]) ||
		    !SameArgument(a.args[1], b.args[1]) ||
		    (a.op == ValueOpcode::ReadConstBuffer && !SameArgument(a.args[3], b.args[3])))
			return false;
		if (result.nodes[a.args[2]].kind != Kind::Immediate ||
		    result.nodes[b.args[2]].kind != Kind::Immediate)
			return false;
		const auto offset = [&](const Node& n) {
			const int64_t immediate = static_cast<int32_t>(n.immediate);
			const auto    value     = static_cast<uint32_t>(result.nodes[n.args[2]].immediate);
			return n.op == ValueOpcode::ReadConstBuffer
			           ? (immediate + value) & ~int64_t {3}
			           : (immediate & ~int64_t {3}) + (value & ~3u);
		};
		if (a.op == ValueOpcode::ReadConstBuffer &&
		    (static_cast<int32_t>(a.immediate) < 0 || static_cast<int32_t>(b.immediate) < 0))
			return false;
		return offset(b) == offset(a) + 4;
	}
	void GroupReads() {
		// Move only infallible immediates across reads. Never cross user loads,
		// computations, another reader policy, or a parent-pointer dependency.
		for (uint32_t first = 0; first < result.nodes.size(); ++first) {
			if (result.nodes[first].kind != Kind::Read) continue;
			LinearSrtPlan::ReadGroup group {};
			group.indices[group.count++] = first;
			for (uint32_t next = first + 1; next < result.nodes.size() && group.count < 16;
			     ++next) {
				const auto& n = result.nodes[next];
				if (n.kind == Kind::Immediate) continue;
				if (n.kind != Kind::Read || n.clean != result.nodes[first].clean) break;
				if (!ConsecutiveRead(result.nodes[group.indices[group.count - 1]], n)) break;
				bool dependent = false;
				for (const auto arg: n.args)
					dependent |= arg >= first && result.nodes[arg].kind != Kind::Immediate;
				if (dependent) break;
				group.indices[group.count++] = next;
			}
			if (group.count < 2) continue;
			first = group.indices[group.count - 1];
			result.read_groups.push_back(group);
		}
	}
	void EmitCompiledGroup(Xbyak::CodeGenerator& c, const LinearSrtPlan::ReadGroup& group,
	                       Xbyak::Label& scalar, Xbyak::Label& after) {
		using namespace Xbyak::util;
		const auto& first    = result.nodes[group.indices[0]];
		const auto& last     = result.nodes[group.indices[group.count - 1]];
		const auto  relative = [&](const Node& n) -> int64_t {
            const auto immediate = int64_t {static_cast<int32_t>(n.immediate)};
            const auto offset = int64_t {static_cast<uint32_t>(result.nodes[n.args[2]].immediate)};
            return n.op == ValueOpcode::ReadConstBuffer
			            ? (immediate + offset) & ~int64_t {3}
			            : (immediate & ~int64_t {3}) + (offset & ~int64_t {3});
		};
		const auto start = relative(first), end = relative(last);
		// These are compiler invariants, not a relaxation of runtime bounds.
		EXIT_IF(group.count < 2 || group.count > 16 || end - start != (group.count - 1) * 4u);
		c.mov(rax, c.qword[r13 + first.args[1] * 8]);
		c.shl(rax, 32);
		c.mov(esi, c.dword[r13 + first.args[0] * 8]);
		c.or_(rsi, rax);
		c.mov(rax, 0x0000fffffffffffcull);
		c.and_(rsi, rax);
		if (first.op == ValueOpcode::ReadConstBuffer) {
			EXIT_IF(start < 0 || static_cast<int32_t>(first.immediate) < 0);
			c.mov(eax, c.dword[r13 + first.args[1] * 8]);
			c.shr(eax, 16);
			c.and_(eax, 0x3fff);
			c.mov(edx, c.dword[r13 + first.args[3] * 8]);
			Xbyak::Label no_stride;
			c.test(eax, eax);
			c.jz(no_stride);
			c.imul(rdx, rax);
			c.L(no_stride);
			c.mov(rax, static_cast<uint64_t>(end) + 4u);
			c.cmp(rdx, rax);
			c.jb(scalar, Xbyak::CodeGenerator::T_NEAR);
		} else {
			if (start < 0) {
				c.mov(rax, static_cast<uint64_t>(-start));
				c.cmp(rsi, rax);
				c.jb(scalar, Xbyak::CodeGenerator::T_NEAR);
			}
			if (end >= 0) {
				c.mov(rax, 0x0000ffffffffffffull - static_cast<uint64_t>(end));
				c.cmp(rsi, rax);
				c.ja(scalar, Xbyak::CodeGenerator::T_NEAR);
			}
		}
		if (start != 0) {
			c.mov(rax, static_cast<uint64_t>(start));
			c.add(rsi, rax);
		}
		c.mov(rdi, r12);
		c.mov(rdx, rsp);
		c.mov(ecx, group.count);
		c.mov(r8d, first.clean ? 1 : 0);
		// An explicit SysV helper bridges to the host callback ABI on Windows too.
		c.mov(rax, reinterpret_cast<uint64_t>(&LinearSrtPlan::ReadSpan));
		c.call(rax);
		c.test(al, al);
		c.jz(scalar, Xbyak::CodeGenerator::T_NEAR);
		for (uint32_t j = 0; j < group.count; ++j) {
			c.mov(eax, c.dword[rsp + j * 4]);
			c.mov(c.qword[r13 + group.indices[j] * 8], rax);
		}
		c.jmp(after, Xbyak::CodeGenerator::T_NEAR);
	}

	static bool Binary(ValueOpcode op) {
		switch (op) {
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::IMul64:
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
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::UGreaterThanEqual32:
			case ValueOpcode::UMin32:
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor: return true;
			default: return false;
		}
	}
public:
	// Why Build failed (the first rejected construct), for diagnostics.
	const char* failure   = nullptr;
	ValueOpcode failed_op = ValueOpcode::Void;

private:
	uint32_t Fail(const char* why, ValueOpcode op = ValueOpcode::Void) {
		if (failure == nullptr) {
			failure   = why;
			failed_op = op;
		}
		return Invalid;
	}

	uint32_t Add(Value value, bool clean, unsigned depth = 0) {
		if (depth > 64 || result.nodes.size() >= 8192) return Fail("size");
		Node node;
		node.clean       = clean;
		const Inst* inst = nullptr;
		if (value.IsImmediate()) {
			switch (value.GetType()) {
				case Type::U1: node.immediate = value.U1(); break;
				case Type::U8: node.immediate = value.U8(); break;
				case Type::U16: node.immediate = value.U16(); break;
				case Type::U32: node.immediate = value.U32(); break;
				case Type::U64: node.immediate = value.U64(); break;
				case Type::F32: node.immediate = std::bit_cast<uint32_t>(value.F32Value()); break;
				default: return Fail("immediate type");
			}
		} else {
			inst = value.TryInstruction();
			if (!inst) return Fail("not an instruction");
			const auto key = std::pair {inst, clean};
			if (auto found = known.find(key); found != known.end()) return found->second;
			if (!visiting.insert(key).second) return Fail("cycle");
			struct Pop {
				decltype(visiting)& set;
				decltype(key)       key_value;
				~Pop() { set.erase(key_value); }
			} pop {visiting, key};
			node.op          = inst->GetOpcode();
			const auto arg   = [&](size_t i) { return Add(inst->Arg(i), clean, depth + 1); };
			const auto alias = [&](uint32_t index) {
				if (index != Invalid) known.emplace(key, index);
				return index;
			};
			switch (node.op) {
				case ValueOpcode::Identity:
				case ValueOpcode::BitCastU32F32:
				case ValueOpcode::BitCastF32U32: return alias(arg(0));
				case ValueOpcode::GetUserData: {
					const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
					if (reg < source.user_data_base) return Fail("user data below base");
					node.kind      = Kind::User;
					node.immediate = reg - source.user_data_base;
					break;
				}
				case ValueOpcode::GetShaderBase: node.kind = Kind::Base; break;
				case ValueOpcode::ReadConst: {
					const auto slot = inst->Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
					    slot.U32() >= source.srt_reads.size())
						return Fail("dynamic const slot");
					const bool use_clean = clean || (slot.U32() < result.clean_slots.size() &&
					                                 result.clean_slots[slot.U32()]);
					return alias(Add(source.srt_reads[slot.U32()].value, use_clean, depth + 1));
				}
				case ValueOpcode::CompositeExtractU32x2: {
					const auto  component = inst->Arg(1).Resolve();
					const auto* packed    = inst->Arg(0).ResolveInstruction();
					if (!packed || !component.IsImmediate() || component.GetType() != Type::U32 ||
					    component.U32() >= 2)
						return Fail("composite extract u32x2", packed ? packed->GetOpcode() : ValueOpcode::Void);
					if (packed->GetOpcode() == ValueOpcode::CompositeConstructU32x2)
						return alias(Add(packed->Arg(component.U32()), clean, depth + 1));
					if (packed->GetOpcode() == ValueOpcode::IAddCarry32) {
						// 64-bit address arithmetic (s_add_u32 + s_addc_u32). As the interpreter
						// evaluates it: word 0 is the 32-bit sum of both operands, word 1 its carry,
						// which is the sum being below an operand. Operands in their original order.
						const auto lhs = Add(packed->Arg(0), clean, depth + 1);
						const auto rhs = Add(packed->Arg(1), clean, depth + 1);
						if (lhs == Invalid || rhs == Invalid) return Invalid;
						Node sum;
						sum.kind    = Kind::Binary;
						sum.op      = ValueOpcode::IAdd32;
						sum.clean   = clean;
						sum.args[0] = lhs;
						sum.args[1] = rhs;
						const uint32_t sum_index = result.nodes.size();
						result.nodes.push_back(sum);
						if (component.U32() == 0) return alias(sum_index);
						Node carry;
						carry.kind    = Kind::Binary;
						carry.op      = ValueOpcode::ULessThan32;
						carry.clean   = clean;
						carry.args[0] = sum_index;
						carry.args[1] = lhs;
						const uint32_t carry_index = result.nodes.size();
						result.nodes.push_back(carry);
						return alias(carry_index);
					}
					return Fail("composite extract u32x2", packed->GetOpcode());
				}
				case ValueOpcode::CompositeExtractU64: {
					const auto component = inst->Arg(1).Resolve();
					if (!component.IsImmediate() || component.GetType() != Type::U32 ||
					    component.U32() >= 2)
						return Fail("composite extract u64");
					node.kind      = Kind::Unary;
					node.immediate = component.U32();
					node.args[0]   = arg(0);
					break;
				}
                case ValueOpcode::BitFieldUExtract: {
                    const auto offset = inst->Arg(1).Resolve();
                    const auto width = inst->Arg(2).Resolve();
                    if (!offset.IsImmediate() || offset.GetType() != Type::U32 ||
                        !width.IsImmediate() || width.GetType() != Type::U32 ||
                        offset.U32() > 32 || width.U32() > 32 - offset.U32()) return Fail("bitfield");
                    const auto input = arg(0);
                    if (input == Invalid) return Invalid;
                    const auto mask = width.U32() == 32 ? UINT32_MAX :
                                      width.U32() == 0 ? 0u : (1u << width.U32()) - 1;
                    const auto shift_amount = Add(Value(offset.U32()), clean, depth + 1);
                    const auto mask_value = Add(Value(mask), clean, depth + 1);
                    if (shift_amount == Invalid || mask_value == Invalid) return Invalid;
                    // Lower to existing integer nodes so the Xbyak code needs no
                    // bitfield operation. Even width zero still evaluates input.
                    Node shift;
                    shift.kind = Kind::Binary; shift.op = ValueOpcode::ShiftRightLogical32;
                    shift.clean = clean; shift.args[0] = input; shift.args[1] = shift_amount;
                    const uint32_t shifted = result.nodes.size(); result.nodes.push_back(shift);
                    Node bits;
                    bits.kind = Kind::Binary; bits.op = ValueOpcode::BitwiseAnd32;
                    bits.clean = clean; bits.args[0] = shifted; bits.args[1] = mask_value;
                    const uint32_t extracted = result.nodes.size(); result.nodes.push_back(bits);
                    return alias(extracted);
                }
				case ValueOpcode::BitwiseNot32:
				case ValueOpcode::LogicalNot:
					node.kind    = Kind::Unary;
					node.args[0] = arg(0);
					break;
				case ValueOpcode::SelectU32:
				case ValueOpcode::SelectU1:
				case ValueOpcode::SelectF32:
					// The original evaluator eagerly evaluates all three operands.
					node.kind = Kind::Select;
					for (int i = 0; i < 3; ++i)
						node.args[i] = arg(i);
					break;
				case ValueOpcode::LoadAddressU32:
				case ValueOpcode::ReadConstBuffer: {
					const auto  memory = inst->Flags<MemoryFlags>().index;
					const auto* handle = inst->Arg(0).ResolveInstruction();
					const bool  buffer = node.op == ValueOpcode::ReadConstBuffer;
					if (memory >= source.memory_info.size() || !handle ||
					    handle->NumArgs() != (buffer ? 4u : 2u) ||
					    source.memory_info[memory].kind !=
					        (buffer ? ResourceKind::ScalarBuffer : ResourceKind::ScalarAddress))
						return Fail("memory handle", node.op);
					node.kind      = Kind::Read;
					node.immediate = source.memory_info[memory].offset;
					node.args[0]   = Add(handle->Arg(0), clean, depth + 1);
					node.args[1]   = Add(handle->Arg(1), clean, depth + 1);
					node.args[2]   = arg(1);
					if (buffer)
						for (int i = 2; i < 4; ++i)
							node.args[i + 1] = Add(handle->Arg(i), clean, depth + 1);
					break;
				}
				default:
					if (!Binary(node.op)) return Fail("opcode", node.op);
					node.kind    = Kind::Binary;
					node.args[0] = arg(0);
					node.args[1] = arg(1);
					break;
			}
			if (std::ranges::find(node.args, Invalid) != node.args.end()) return Invalid;
		}
		const uint32_t index = result.nodes.size();
		result.nodes.push_back(node);
		if (inst) known.emplace(std::pair {inst, clean}, index);
		return index;
	}

public:
	LinearSrtCompiler(const ResourcePlan& input, LinearSrtPlan& output,
	                  std::span<const uint8_t> active = {})
	    : source(input), result(output), active_sources(active) {}
	bool Build() {
		if (!source.srt_plan_complete) return Fail("incomplete plan"), false;
		if (!source.control_flow.empty() && active_sources.empty()) return Fail("control flow"), false;
		if (!active_sources.empty() && active_sources.size() != source.descriptor_sources.size())
			return Fail("active sources"), false;
		result.active_sources.assign(active_sources.begin(), active_sources.end());
		result.sources      = source.materialization_sources;
		result.clean_slots  = source.clean_flat_slots;
		result.active_count = source.descriptor_sources.size();
		for (const auto source_index: result.sources) {
			if (source_index >= source.descriptor_sources.size()) return Fail("source index"), false;
			const auto& descriptor = source.descriptor_sources[source_index];
			if (descriptor.dword_count > descriptor.dwords.size()) return Fail("descriptor size"), false;
			result.descriptor_sizes.push_back(descriptor.dword_count);
			for (uint32_t i = 0; i < descriptor.dword_count; ++i) {
				if (!active_sources.empty() && !active_sources[source_index]) {
					result.descriptor_words.push_back(Invalid);
					continue;
				}
				const auto index = Add(descriptor.dwords[i], false);
				if (index == Invalid) return false;
				result.descriptor_words.push_back(index);
			}
		}
		result.flat_words.assign(source.srt_reads.size(), Invalid);
		for (const auto& read: source.srt_reads) {
			if (read.flat_offset >= result.flat_words.size()) return Fail("flat offset"), false;
			const auto index = Add(read.value, read.flat_offset < result.clean_slots.size() &&
			                                       result.clean_slots[read.flat_offset]);
			if (index == Invalid) return false;
			result.flat_words[read.flat_offset] = index;
		}
		if (result.nodes.empty() && active_sources.empty()) return Fail("empty"), false;
		if (!Generate(true)) return Fail("code generation"), false;
		return true;
	}
    bool BuildPredicate(Value condition) {
        const auto value = Add(condition, true);
        if (value == Invalid || result.nodes.size() > 128) return false;
        result.flat_words = {value};
        // The predicate interpreter issues scalar clean reads, never spans.
        return Generate(false);
    }
	bool Generate(bool group_reads) {
		if (group_reads) GroupReads();
		// The plan keeps these for its life (the shader prefetch's for every program of the game: ~300 MB with 16 GB of
		// RAM), without their growth's spare capacity; before the code takes the nodes' addresses.
		result.nodes.shrink_to_fit();
		result.read_groups.shrink_to_fit();
		result.descriptor_words.shrink_to_fit();
		result.descriptor_sizes.shrink_to_fit();
		Xbyak::ClearError();
		result.code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
		if (Xbyak::GetError() != 0 || !result.code->getCode()) {
			Xbyak::ClearError();
			return false;
		}
		auto& c = *result.code;
		using namespace Xbyak::util;
		Xbyak::Label fail, done;
		std::vector<Xbyak::Label> after_group(result.read_groups.size());
		std::vector<uint32_t>     group_starts(result.nodes.size(), Invalid);
		std::vector<uint32_t>     group_ends(result.nodes.size() + 1, Invalid);
		for (uint32_t g = 0; g < result.read_groups.size(); ++g) {
			const auto& group                              = result.read_groups[g];
			group_starts[group.indices[0]]                 = g;
			group_ends[group.indices[group.count - 1] + 1] = g;
		}
		c.push(r12);
		c.push(r13);
		const uint32_t stack_size = result.read_groups.empty() ? 8 : 72;
		c.sub(rsp, stack_size);
		c.mov(r12, rdi);
		c.mov(r13, rsi);
		// SrtRuntime::user_data is a standard-library span. Its physical layout
		// stays in a C++ accessor instead of embedding an ABI assumption here.
		for (uint32_t i = 0; i < result.nodes.size(); ++i) {
			if (group_ends[i] != Invalid) c.L(after_group[group_ends[i]]);
			if (group_starts[i] != Invalid) {
				const auto   g     = group_starts[i];
				const auto&  group = result.read_groups[g];
				Xbyak::Label scalar;
				c.cmp(c.qword[r12 + offsetof(SrtRuntime, try_read_memory_span)], 0);
				c.jz(scalar, Xbyak::CodeGenerator::T_NEAR);
				// Only infallible immediates have moved across the ordered reads.
				for (uint32_t j = i + 1; j < group.indices[group.count - 1]; ++j) {
					if (result.nodes[j].kind != Kind::Immediate) continue;
					c.mov(rax, result.nodes[j].immediate);
					c.mov(c.qword[r13 + j * 8], rax);
				}
				EmitCompiledGroup(c, group, scalar, after_group[g]);
				c.L(scalar);
			}
			const auto& n = result.nodes[i];
			switch (n.kind) {
				case Kind::Immediate: c.mov(rax, n.immediate); break;
				case Kind::User:
					c.mov(rdi, r12);
					c.mov(rsi, n.immediate);
					c.lea(rdx, c.ptr[r13 + i * 8]);
					c.mov(rax, reinterpret_cast<uint64_t>(&ReadUser));
					c.call(rax);
					c.test(al, al);
					c.jz(fail, Xbyak::CodeGenerator::T_NEAR);
					continue;
				case Kind::Base:
					c.mov(rax, c.qword[r12 + offsetof(SrtRuntime, shader_base)]);
					break;
				case Kind::Read:
					c.mov(rdi, r12);
					c.mov(rsi, reinterpret_cast<uint64_t>(&n));
					c.mov(rdx, r13);
					c.lea(rcx, c.ptr[r13 + i * 8]);
					c.mov(rax, reinterpret_cast<uint64_t>(&LinearSrtPlan::Read));
					c.call(rax);
					c.test(al, al);
					c.jz(fail, Xbyak::CodeGenerator::T_NEAR);
					continue;
				case Kind::Unary:
					c.mov(rax, c.qword[r13 + n.args[0] * 8]);
					if (n.op == ValueOpcode::CompositeExtractU64) {
						if (n.immediate) c.shr(rax, 32);
						c.mov(eax, eax);
					} else if (n.op == ValueOpcode::BitwiseNot32)
						c.not_(eax);
					else {
						c.test(rax, rax);
						c.setz(al);
						c.movzx(eax, al);
					}
					break;
				case Kind::Select:
					c.mov(rax, c.qword[r13 + n.args[1] * 8]);
					c.mov(rcx, c.qword[r13 + n.args[2] * 8]);
					c.cmp(c.qword[r13 + n.args[0] * 8], 0);
					c.cmovz(rax, rcx);
					break;
				case Kind::Binary:
					c.mov(rax, c.qword[r13 + n.args[0] * 8]);
					c.mov(rcx, c.qword[r13 + n.args[1] * 8]);
					switch (n.op) {
						case ValueOpcode::IAdd32: c.add(eax, ecx); break;
						case ValueOpcode::IAdd64: c.add(rax, rcx); break;
						case ValueOpcode::ISub32: c.sub(eax, ecx); break;
						case ValueOpcode::ISub64: c.sub(rax, rcx); break;
						case ValueOpcode::IMul32: c.imul(eax, ecx); break;
						case ValueOpcode::IMul64: c.imul(rax, rcx); break;
						case ValueOpcode::BitwiseAnd32: c.and_(eax, ecx); break;
						case ValueOpcode::BitwiseAnd64: c.and_(rax, rcx); break;
						case ValueOpcode::BitwiseOr32: c.or_(eax, ecx); break;
						case ValueOpcode::BitwiseXor32: c.xor_(eax, ecx); break;
						case ValueOpcode::ShiftLeftLogical32: c.shl(eax, cl); break;
						case ValueOpcode::ShiftLeftLogical64: c.shl(rax, cl); break;
						case ValueOpcode::ShiftRightLogical32: c.shr(eax, cl); break;
						case ValueOpcode::ShiftRightLogical64: c.shr(rax, cl); break;
						case ValueOpcode::ShiftRightArithmetic32: c.sar(eax, cl); break;
						case ValueOpcode::ShiftRightArithmetic64: c.sar(rax, cl); break;
						case ValueOpcode::CompositeConstructU64:
							c.mov(eax, eax);
							c.shl(rcx, 32);
							c.or_(rax, rcx);
							break;
						case ValueOpcode::IEqual32:
							c.cmp(eax, ecx);
							c.sete(al);
							c.movzx(eax, al);
							break;
						case ValueOpcode::INotEqual32:
							c.cmp(eax, ecx);
							c.setne(al);
							c.movzx(eax, al);
							break;
						case ValueOpcode::ULessThan32:
							c.cmp(eax, ecx);
							c.setb(al);
							c.movzx(eax, al);
							break;
						case ValueOpcode::UGreaterThan32:
							c.cmp(eax, ecx);
							c.seta(al);
							c.movzx(eax, al);
							break;
						case ValueOpcode::UGreaterThanEqual32:
							c.cmp(eax, ecx);
							c.setae(al);
							c.movzx(eax, al);
							break;
						case ValueOpcode::UMin32:
							c.cmp(eax, ecx);
							c.cmova(eax, ecx);
							c.mov(eax, eax);
							break;
						case ValueOpcode::LogicalAnd:
						case ValueOpcode::LogicalOr:
						case ValueOpcode::LogicalXor:
							c.test(rax, rax);
							c.setne(al);
							c.test(rcx, rcx);
							c.setne(cl);
							if (n.op == ValueOpcode::LogicalAnd)
								c.and_(al, cl);
							else if (n.op == ValueOpcode::LogicalOr)
								c.or_(al, cl);
							else
								c.xor_(al, cl);
							c.movzx(eax, al);
							break;
						default: std::abort();
					}
					break;
			}
			c.mov(c.qword[r13 + i * 8], rax);
		}
		if (group_ends.back() != Invalid) c.L(after_group[group_ends.back()]);
		c.mov(eax, 1);
		c.jmp(done);
		c.L(fail);
		c.xor_(eax, eax);
		c.L(done);
		c.add(rsp, stack_size);
		c.pop(r13);
		c.pop(r12);
		c.ret();
		if (Xbyak::GetError() != 0 || c.getSize() > 256 * 1024) {
			Xbyak::ClearError();
			return false;
		}
		c.readyRE();
		if (Xbyak::GetError() != 0) {
			Xbyak::ClearError();
			return false;
		}
		result.function = c.getCode<LinearSrtPlan::Function>();
		// Its code alone, in the shared executable memory (jit-arena.h: calls go through a register, jumps to its own
		// labels); the generator and its page-aligned buffer go.
		if (const void* copy = JitArena::Copy(c.getCode(), c.getSize())) {
			result.function = reinterpret_cast<LinearSrtPlan::Function>(const_cast<void*>(copy));
			result.code.reset();
		}
		return true;
	}
	static bool KYTY_SYSV_ABI ReadUser(const SrtRuntime* runtime, uint64_t index,
	                                   uint64_t* output) {
		if (index >= runtime->user_data.size()) return false;
		*output = runtime->user_data[index];
		return true;
	}
};
#endif
} // namespace Libs::Graphics::ShaderRecompiler::IR
#endif
