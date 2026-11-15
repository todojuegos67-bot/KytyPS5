#include "graphics/shader/recompiler/ShaderCalls.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

void PrepareCallTargets(ShaderSource& source, const CompileOptions& options) {
	using namespace IR;
	EXIT_NOT_IMPLEMENTED(options.stage != ShaderType::Compute);
	source.wave_size = options.wave_size;
	for (uint32_t i = 0; i < source.decoded.instructions.size(); ++i) {
		if (source.decoded.instructions[i].opcode != Decoder::Opcode::S_SWAPPC_B64) continue;
		source.calls.push_back({.instruction = i});
	}
	const auto graph = CFG::BuildGraph(source.decoded);
	Program query;
	query.stage = options.stage;
	query.user_data_base = options.user_data_base;
	query.user_data_count = static_cast<uint32_t>(options.user_data.size());
	query.wave_size = options.wave_size;
	auto& block = *query.block_storage.emplace_back(std::make_unique<Block>());
	query.blocks.push_back(&block);
	struct Producer { bool ready = false; std::map<uint32_t, Value> values; };
	std::map<uint32_t, Producer> producers;
	std::function<Value(uint32_t, uint32_t)> Register;
	std::function<Value(uint32_t, Value)> Resolve = [&](uint32_t before, Value value) -> Value {
		value = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr) return value;
		uint32_t code = UINT32_MAX;
		switch (inst->GetOpcode()) {
			case ValueOpcode::GetScalarRegister: code = static_cast<uint32_t>(inst->Arg(0).ScalarRegister()); break;
			case ValueOpcode::GetVccLo: code = 106u; break;
			case ValueOpcode::GetVccHi: code = 107u; break;
			default: break;
		}
		if (code != UINT32_MAX) {
			const auto result = Register(before, code);
			inst->ReplaceUsesWith(result);
			return result;
		}
		for (size_t i = 0; i < inst->NumArgs(); ++i) inst->SetArg(i, Resolve(before, inst->Arg(i)));
		return value;
	};
	// The scalar registers an instruction writes, as its translation says: the decoder does not give
	// every destination its width (Marvel's Wolverine CS 0x65730dab35e283f7: s_bitset0_b32 s7 taken
	// for a writer of s8, a call target, which made two definitions reach the call).
	std::map<uint32_t, std::set<uint32_t>> written;
	const auto Writes = [&](uint32_t index, uint32_t code) {
		if (!CFG::MayWriteScalarRegister(source.decoded.instructions[index], code, source.wave_size))
			return false;
		auto [entry, inserted] = written.try_emplace(index);
		if (inserted) {
			Block translated;
			Frontend::Translator translator(query, &translated, 1u, Frontend::NoLaneSpills,
			    (options.input_info.compute->float_mode & 0x10u) == 0u,
			    !options.input_info.compute->async_compute);
			translator.TranslateInstruction(source.decoded.instructions[index]);
			for (const auto& inst: translated) {
				switch (inst.GetOpcode()) {
					case ValueOpcode::SetScalarRegister:
						entry->second.insert(static_cast<uint32_t>(inst.Arg(0).ScalarRegister()));
						break;
					case ValueOpcode::SetVccLo: entry->second.insert(106u); break;
					case ValueOpcode::SetVccHi: entry->second.insert(107u); break;
					default: break;
				}
			}
		}
		return entry->second.contains(code);
	};
	Register = [&](uint32_t before, uint32_t code) -> Value {
		std::vector<uint32_t> crossed_calls;
		const auto definition = CFG::FindScalarDefinition(source.decoded, graph, before, code,
		                                                  source.wave_size, crossed_calls,
		                                                  [&](uint32_t i) { return Writes(i, code); });
		// Several reaching definitions select the value on the GPU, so no SRT walk evaluates it.
		if (!definition) return Value(&block.AppendNewInst(ValueOpcode::UndefU32));
		const auto index = *definition;
		for (const auto crossed: crossed_calls) {
			std::ranges::find(source.calls, crossed, &ShaderSource::Call::instruction)->preserved_registers.set(code);
		}
		if (index == UINT32_MAX) {
			EXIT_IF(code < query.user_data_base || code - query.user_data_base >= query.user_data_count);
			return IREmitter(&block).GetUserData(static_cast<ScalarReg>(code));
		}
		auto [entry, inserted] = producers.try_emplace(index);
		auto& producer = entry->second;
		if (inserted) {
			Block translated;
			Frontend::Translator translator(query, &translated, 1u, Frontend::NoLaneSpills,
			    (options.input_info.compute->float_mode & 0x10u) == 0u,
			    !options.input_info.compute->async_compute);
			translator.TranslateInstruction(source.decoded.instructions[index]);
			for (const auto& inst: translated) {
				uint32_t destination = UINT32_MAX;
				Value value;
				switch (inst.GetOpcode()) {
					case ValueOpcode::SetScalarRegister:
						destination = static_cast<uint32_t>(inst.Arg(0).ScalarRegister()); value = inst.Arg(1); break;
					case ValueOpcode::SetVccLo: destination = 106u; value = inst.Arg(0); break;
					case ValueOpcode::SetVccHi: destination = 107u; value = inst.Arg(0); break;
					default: break;
				}
				if (destination != UINT32_MAX) producer.values.insert_or_assign(destination, value);
			}
			for (auto& [destination, value]: producer.values) value = Resolve(index, value);
			for (auto& inst: translated) inst.SetParent(&block);
			block.Instructions().splice(block.end(), translated.Instructions());
			producer.ready = true;
		}
		EXIT_IF(!producer.ready || !producer.values.contains(code));
		return producer.values.at(code).Resolve();
	};
	for (auto& site: source.calls) {
		const auto& call = source.decoded.instructions[site.instruction];
		EXIT_NOT_IMPLEMENTED(call.src0.kind != Decoder::OperandKind::Sgpr ||
		                     call.dst.kind != Decoder::OperandKind::Sgpr ||
		                     call.src0.reg != call.dst.reg || call.dst.reg >= 105u);
		for (uint32_t i = 0; i < 2u; ++i) {
			const auto value = Register(site.instruction, call.src0.reg + i);
			if (!ValidateRuntimeValue(query, value, RuntimeValueType::Integer)) {
				EXIT_IF(value.Instruction()->GetOpcode() != ValueOpcode::ReadConstBuffer);
				source.unsupported_call_pc = call.pc;
				source.calls.clear();
				return;
			}
			auto& root = query.value_storage.emplace_back(ValueOpcode::ReferenceU32);
			root.SetArg(0, value);
			site.target[i] = Value(&root);
		}
	}
	RewriteToSsa(query.blocks);
	ConstantPropagationPass(query.blocks, query.wave_size);
	RemoveIdentities(query);
	EliminateDeadCode(query.blocks);
	for (auto& site: source.calls)
		for (auto& value: site.target) value = value.Instruction()->Arg(0).Resolve();
	query.value_storage.clear();
	for (auto& inst: block) {
		inst.SetParent(nullptr);
		if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 || inst.GetOpcode() == ValueOpcode::ReadConstBuffer)
			inst.SetFlags(SrtReadFlags {.index = inst.Flags<MemoryFlags>().index});
	}
	query.value_storage.splice(query.value_storage.end(), block.Instructions());
	source.call_targets = std::move(static_cast<ResourcePlan&>(query));
}

namespace {

// The calls of source linked: each site's leaf function read through read(address, words) at its address.
const Decoder::Program& LinkCalls(ShaderSource& source,
                                  const std::function<bool(uint64_t, std::span<uint32_t>)>& read) {
	auto& reads = source.reads;
	const auto query_reads = reads.size();
	if (source.linked) {
		auto& linked = *source.linked;
		bool unchanged = true;
		for (size_t i = 0; i < source.calls.size(); ++i) {
			const auto& function = linked.functions[i];
			const auto expected = std::span(linked.code).subspan(function.offset, function.size);
			const auto preceding = std::span(source.calls).first(i);
			const auto previous = std::ranges::find(preceding, source.calls[i].address, &ShaderSource::Call::address);
			if (previous != preceding.end()) {
				const auto& observed = linked.functions[previous - preceding.begin()];
				if (std::ranges::equal(expected, std::span(linked.code).subspan(observed.offset, observed.size))) continue;
				unchanged = false;
				break;
			}
			linked.observed_function.resize(function.size);
			if (!read(source.calls[i].address, linked.observed_function) ||
			    !std::ranges::equal(expected, linked.observed_function)) {
				unchanged = false;
				break;
			}
		}
		if (unchanged) return linked.decoded;
		reads.resize(query_reads);
	}
	ShaderSource::Linked linked {.code = source.code, .decoded = source.decoded};
	const auto& last = source.decoded.instructions.back();
	EXIT_NOT_IMPLEMENTED(last.opcode != Decoder::Opcode::S_ENDPGM);
	// Keep the caller's data/footer at native offsets. Expand leaf CFGs per call site
	// before SSA, so return edges neither require a dispatcher nor merge resource roots.
	uint32_t function_pc = last.pc + last.word_count * sizeof(uint32_t);
	std::map<uint64_t, std::vector<Decoder::Instruction>> functions;
	for (size_t i = 0; i < source.calls.size(); ++i) {
		const auto& site = source.calls[i];
		const auto& call = source.decoded.instructions[site.instruction];
		const auto address = source.calls[i].address;
		auto [entry, inserted] = functions.try_emplace(address);
		auto& function = entry->second;
		if (inserted) {
			for (uint32_t index = 0;;) {
				std::array<uint32_t, Decoder::MaxInstructionRawWords> words {};
				EXIT_IF(index > UINT32_MAX / sizeof(uint32_t) - words.size() ||
				        address > UINT64_MAX - (uint64_t {index} + words.size()) * sizeof(uint32_t));
				uint32_t count = 1;
				Decoder::Instruction inst;
				for (;;) {
					if (!read(address + uint64_t {index} * sizeof(uint32_t), std::span(words).first(count)))
						EXIT("shader call at pc 0x%08x cannot read leaf code", call.pc);
					inst = {};
					Decoder::DecodeInstruction(words, 0u, inst);
					if (inst.word_count <= count) break;
					count = inst.word_count;
				}
				inst.pc = index * sizeof(uint32_t);
				if (Decoder::IsDirectBranch(inst.opcode)) inst.branch_target += inst.pc;
				index += inst.word_count;
				EXIT_NOT_IMPLEMENTED(inst.opcode == Decoder::Opcode::S_SWAPPC_B64 ||
				                     inst.opcode == Decoder::Opcode::S_GETPC_B64 ||
				                     inst.opcode == Decoder::Opcode::S_ENDPGM);
				function.push_back(inst);
				if (inst.opcode == Decoder::Opcode::S_SETPC_B64) break;
			}
		}
		const uint32_t function_size = function.back().pc + function.back().word_count * sizeof(uint32_t);
		EXIT_IF(function_pc > UINT32_MAX - function_size);
		const auto function_offset = static_cast<uint32_t>(linked.code.size());
		for (auto inst: function) {
			EXIT_IF(CFG::MayWriteScalarRegister(inst, call.dst.reg, source.wave_size) ||
			        CFG::MayWriteScalarRegister(inst, call.dst.reg + 1u, source.wave_size));
			for (uint32_t code = 0; code < site.preserved_registers.size(); ++code) {
				if (site.preserved_registers[code] &&
				    CFG::MayWriteScalarRegister(inst, code, source.wave_size))
					EXIT("shader call at pc 0x%08x clobbers scalar target dependency s%u", call.pc, code);
			}
			linked.code.insert(linked.code.end(), inst.raw, inst.raw + inst.word_count);
			if (inst.opcode == Decoder::Opcode::S_SETPC_B64) {
				EXIT_NOT_IMPLEMENTED(inst.src0.kind != Decoder::OperandKind::Sgpr || inst.src0.reg != call.dst.reg);
				inst.opcode = Decoder::Opcode::S_BRANCH;
				inst.branch_target = call.pc + call.word_count * sizeof(uint32_t);
			} else if (Decoder::IsDirectBranch(inst.opcode)) {
				EXIT_IF(inst.branch_target >= function_size);
				inst.branch_target += function_pc;
			}
			inst.pc += function_pc;
			linked.decoded.instructions.push_back(std::move(inst));
		}
		linked.functions.push_back({function_offset, function_size / 4u});
		linked.decoded.instructions[site.instruction].branch_target = function_pc;
		function_pc += function_size;
	}
	reads.resize(query_reads);
	for (size_t i = 0; i < source.calls.size(); ++i) {
		const auto preceding = std::span(source.calls).first(i);
		if (std::ranges::find(preceding, source.calls[i].address, &ShaderSource::Call::address) == preceding.end())
			reads.emplace_back(source.calls[i].address, linked.functions[i].size * sizeof(uint32_t));
	}
	linked.decoded.code = linked.code;
	linked.observed_function.resize(std::ranges::max_element(linked.functions, {},
	    &ShaderSource::Linked::Function::size)->size);

	source.linked = std::move(linked);
	++source.revision;
	return source.linked->decoded;
}


} // namespace

const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime) {
	EXIT_IF(source.unsupported_call_pc != UINT32_MAX);
	auto& reads = source.reads;
	reads.clear();
	IR::SrtReadCapture capture(runtime, reads);
	const auto clean = IR::CleanRuntime(capture.ObservedRuntime());
	IR::SrtWalker walker(source.call_targets, clean);
	for (auto& site: source.calls) {
		uint32_t low = 0, high = 0;
		if (!walker.Evaluate(site.target[0], low) || !walker.Evaluate(site.target[1], high))
			EXIT("shader call at pc 0x%08x has an unavailable scalar target",
			     source.decoded.instructions[site.instruction].pc);
		// SWAPPC reads the old aliased pair and ignores the target's low two bits.
		const auto address = ((uint64_t {high} << 32u) | low) & ~uint64_t {3};
		EXIT_IF(address == 0u);
		site.address = address;
	}
	return LinkCalls(source, [&](uint64_t address, std::span<uint32_t> words) {
		return clean.read_memory(clean.userdata, address, words);
	});
}

const Decoder::Program& LinkShaderSource(ShaderSource& source, std::span<const std::vector<uint32_t>> callees) {
	EXIT_IF(source.unsupported_call_pc != UINT32_MAX || callees.size() != source.calls.size());
	// One stand-in address per distinct callee: the linker shares a leaf between the sites that call it.
	for (size_t i = 0; i < source.calls.size(); ++i) {
		size_t first = i;
		for (size_t k = 0; k < i; ++k) {
			if (std::ranges::equal(callees[k], callees[i])) {
				first = k;
				break;
			}
		}
		source.calls[i].address = uint64_t {first + 1u} << 32u;
	}
	source.reads.clear();
	source.linked.reset();
	return LinkCalls(source, [&](uint64_t address, std::span<uint32_t> words) {
		const auto  index  = static_cast<size_t>(address >> 32u) - 1u;
		const auto  offset = static_cast<size_t>(address & 0xffffffffu) / sizeof(uint32_t);
		if (index >= callees.size() || offset > callees[index].size() ||
		    words.size() > callees[index].size() - offset) {
			return false;
		}
		std::copy_n(callees[index].begin() + static_cast<ptrdiff_t>(offset), words.size(), words.begin());
		return true;
	});
}

} // namespace Libs::Graphics::ShaderRecompiler
