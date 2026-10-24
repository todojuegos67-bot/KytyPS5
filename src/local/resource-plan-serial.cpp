#include "resource-plan-serial.h"

#include "graphics/shader/recompiler/ir/passes/LinearSrt.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace ResourcePlanSerial {

namespace IR = Libs::Graphics::ShaderRecompiler::IR;

namespace {

constexpr uint32_t Magic = 0x4e4c5052u; // "RPLN"

// A field added to one of these types must be added here too (Save, Load and Same). The sizes are the 64-bit Windows
// build's (Microsoft's STL; libstdc++'s containers have other sizes: the Linux build checks nothing here).
#if defined(_MSC_VER) || defined(_WIN32)
static_assert(sizeof(void*) != 8 || sizeof(IR::ResourcePlan) == 496, "ResourcePlan changed: update resource-plan-serial");
static_assert(sizeof(void*) != 8 || sizeof(IR::ShaderInfo) == 192, "ShaderInfo changed: update resource-plan-serial");
static_assert(sizeof(void*) != 8 || sizeof(IR::ImageResource) == 88, "ImageResource changed: update resource-plan-serial");
static_assert(sizeof(void*) != 8 || sizeof(IR::StageInput) == 56, "StageInput changed: update resource-plan-serial");
static_assert(sizeof(void*) != 8 || sizeof(IR::StageOutput) == 48, "StageOutput changed: update resource-plan-serial");
static_assert(sizeof(void*) != 8 || sizeof(IR::DescriptorSource) == 184, "DescriptorSource changed: update resource-plan-serial");
static_assert(sizeof(void*) != 8 || sizeof(IR::ResourceBlock) == 64, "ResourceBlock changed: update resource-plan-serial");
static_assert(sizeof(void*) != 8 || sizeof(IR::UniformFillPlan) == 96, "UniformFillPlan changed: update resource-plan-serial");
#endif
static_assert(std::is_trivially_copyable_v<IR::MemoryInfo>);
static_assert(std::is_trivially_copyable_v<IR::BufferResource>);
static_assert(std::is_trivially_copyable_v<IR::SamplerResource>);
static_assert(std::is_trivially_copyable_v<IR::SampledResourcePair>);
static_assert(std::is_trivially_copyable_v<IR::DescriptorSource::IndirectImage>);
static_assert(std::is_trivially_copyable_v<IR::UniformFill>);

enum class Tag : uint8_t { Empty, Inst, Scalar, Vector, U1, U8, U16, U32, U64, F16, F32 };

class Writer {
public:
	explicit Writer(std::vector<uint8_t>& out): m_out(out) {}
	template <typename T>
	void Put(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
		m_out.insert(m_out.end(), bytes, bytes + sizeof(T));
	}
	template <typename T>
	void PutVector(const std::vector<T>& values) {
		static_assert(std::is_trivially_copyable_v<T>);
		Put(static_cast<uint32_t>(values.size()));
		const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
		m_out.insert(m_out.end(), bytes, bytes + values.size() * sizeof(T));
	}
	void PutString(const std::string& value) {
		Put(static_cast<uint32_t>(value.size()));
		m_out.insert(m_out.end(), value.begin(), value.end());
	}

private:
	std::vector<uint8_t>& m_out;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> bytes): m_bytes(bytes) {}
	template <typename T>
	bool Get(T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (m_bytes.size() - m_at < sizeof(T)) return false;
		std::memcpy(&value, m_bytes.data() + m_at, sizeof(T));
		m_at += sizeof(T);
		return true;
	}
	template <typename T>
	bool GetVector(std::vector<T>& values) {
		uint32_t count = 0;
		if (!Get(count) || (m_bytes.size() - m_at) / sizeof(T) < count) return false;
		values.resize(count);
		std::memcpy(values.data(), m_bytes.data() + m_at, count * sizeof(T));
		m_at += count * sizeof(T);
		return true;
	}
	bool GetString(std::string& value) {
		uint32_t count = 0;
		if (!Get(count) || m_bytes.size() - m_at < count) return false;
		value.assign(reinterpret_cast<const char*>(m_bytes.data() + m_at), count);
		m_at += count;
		return true;
	}
	[[nodiscard]] bool AtEnd() const { return m_at == m_bytes.size(); }

private:
	std::span<const uint8_t> m_bytes;
	size_t                   m_at = 0;
};

bool PutValue(Writer& w, const std::unordered_map<const IR::Inst*, uint32_t>& index, IR::Value value) {
	if (value.IsEmpty()) {
		w.Put(Tag::Empty);
		return true;
	}
	if (const auto* inst = value.TryInstruction()) {
		const auto found = index.find(inst);
		if (found == index.end()) return false; // a value outside the plan's storage
		w.Put(Tag::Inst);
		w.Put(found->second);
		return true;
	}
	switch (value.GetType()) {
		case IR::Type::ScalarReg: w.Put(Tag::Scalar); w.Put(value.ScalarRegister()); return true;
		case IR::Type::VectorReg: w.Put(Tag::Vector); w.Put(value.VectorRegister()); return true;
		case IR::Type::U1: w.Put(Tag::U1); w.Put(static_cast<uint8_t>(value.U1() ? 1 : 0)); return true;
		case IR::Type::U8: w.Put(Tag::U8); w.Put(value.U8()); return true;
		case IR::Type::U16: w.Put(Tag::U16); w.Put(value.U16()); return true;
		case IR::Type::U32: w.Put(Tag::U32); w.Put(value.U32()); return true;
		case IR::Type::U64: w.Put(Tag::U64); w.Put(value.U64()); return true;
		case IR::Type::F16: w.Put(Tag::F16); w.Put(value.F16Bits()); return true;
		case IR::Type::F32: w.Put(Tag::F32); w.Put(std::bit_cast<uint32_t>(value.F32Value())); return true;
		default: return false;
	}
}

bool GetValue(Reader& r, const std::vector<IR::Inst*>& insts, IR::Value& value) {
	Tag tag {};
	if (!r.Get(tag)) return false;
	switch (tag) {
		case Tag::Empty: value = {}; return true;
		case Tag::Inst: {
			uint32_t at = 0;
			if (!r.Get(at) || at >= insts.size()) return false;
			value = IR::Value(insts[at]);
			return true;
		}
		case Tag::Scalar: { IR::ScalarReg v {}; if (!r.Get(v)) return false; value = IR::Value(v); return true; }
		case Tag::Vector: { IR::VectorReg v {}; if (!r.Get(v)) return false; value = IR::Value(v); return true; }
		case Tag::U1: { uint8_t v = 0; if (!r.Get(v)) return false; value = IR::Value(v != 0); return true; }
		case Tag::U8: { uint8_t v = 0; if (!r.Get(v)) return false; value = IR::Value(v); return true; }
		case Tag::U16: { uint16_t v = 0; if (!r.Get(v)) return false; value = IR::Value(v); return true; }
		case Tag::U32: { uint32_t v = 0; if (!r.Get(v)) return false; value = IR::Value(v); return true; }
		case Tag::U64: { uint64_t v = 0; if (!r.Get(v)) return false; value = IR::Value(v); return true; }
		case Tag::F16: { uint16_t v = 0; if (!r.Get(v)) return false; value = IR::Value::F16(v); return true; }
		case Tag::F32: { uint32_t v = 0; if (!r.Get(v)) return false; value = IR::Value::F32(std::bit_cast<float>(v)); return true; }
		default: return false;
	}
}

void PutInfo(Writer& w, const IR::ShaderInfo& info) {
	w.PutVector(info.buffers);
	w.Put(static_cast<uint32_t>(info.images.size()));
	for (const auto& image: info.images) {
		w.Put(image.source);
		w.Put(image.first_use_pc);
		w.Put(image.resource_class);
		w.Put(image.numeric_class);
		w.Put(image.dimension);
		w.Put(image.mip_mode);
		w.Put(image.mip_count);
		w.Put(image.conversion_format);
		w.Put(image.shader_swizzle);
		w.Put(static_cast<uint8_t>(image.read | image.written << 1 | image.atomic << 2 | image.depth_compare << 3 |
		                           image.cube << 4 | image.r128 << 5));
		w.Put(image.indirect_root);
		w.Put(image.indirect_mapping_offset);
		w.Put(image.indirect_search_iterations);
		w.Put(image.max_array_index);
		w.PutVector(image.indirect_resources);
	}
	w.PutVector(info.samplers);
	w.PutVector(info.sampled_pairs);
	w.Put(static_cast<uint32_t>(info.inputs.size()));
	for (const auto& input: info.inputs) {
		w.Put(input.kind);
		w.Put(input.location);
		w.Put(input.component_count);
		w.PutString(input.debug_name);
		w.Put(input.per_vertex);
	}
	w.Put(static_cast<uint32_t>(info.outputs.size()));
	for (const auto& output: info.outputs) {
		w.Put(output.kind);
		w.Put(output.index);
		w.Put(output.location);
		w.PutString(output.debug_name);
	}
	w.Put(info.vertex_fetch_components);
	w.Put(info.vertex_offset_sgpr);
	w.Put(info.instance_offset_sgpr);
	w.Put(info.has_bitwise_xor);
	w.Put(info.uses_dma);
}

bool GetInfo(Reader& r, IR::ShaderInfo& info) {
	uint32_t count = 0;
	if (!r.GetVector(info.buffers) || !r.Get(count)) return false;
	info.images.resize(count);
	for (auto& image: info.images) {
		uint8_t bits = 0;
		if (!r.Get(image.source) || !r.Get(image.first_use_pc) || !r.Get(image.resource_class) ||
		    !r.Get(image.numeric_class) || !r.Get(image.dimension) || !r.Get(image.mip_mode) || !r.Get(image.mip_count) ||
		    !r.Get(image.conversion_format) || !r.Get(image.shader_swizzle) || !r.Get(bits) || !r.Get(image.indirect_root) ||
		    !r.Get(image.indirect_mapping_offset) || !r.Get(image.indirect_search_iterations) ||
		    !r.Get(image.max_array_index) || !r.GetVector(image.indirect_resources))
			return false;
		image.read          = (bits & 1u) != 0;
		image.written       = (bits & 2u) != 0;
		image.atomic        = (bits & 4u) != 0;
		image.depth_compare = (bits & 8u) != 0;
		image.cube          = (bits & 16u) != 0;
		image.r128          = (bits & 32u) != 0;
	}
	if (!r.GetVector(info.samplers) || !r.GetVector(info.sampled_pairs) || !r.Get(count)) return false;
	info.inputs.resize(count);
	for (auto& input: info.inputs) {
		if (!r.Get(input.kind) || !r.Get(input.location) || !r.Get(input.component_count) || !r.GetString(input.debug_name) ||
		    !r.Get(input.per_vertex))
			return false;
	}
	if (!r.Get(count)) return false;
	info.outputs.resize(count);
	for (auto& output: info.outputs) {
		if (!r.Get(output.kind) || !r.Get(output.index) || !r.Get(output.location) || !r.GetString(output.debug_name))
			return false;
	}
	return r.Get(info.vertex_fetch_components) && r.Get(info.vertex_offset_sgpr) && r.Get(info.instance_offset_sgpr) &&
	       r.Get(info.has_bitwise_xor) && r.Get(info.uses_dma);
}

} // namespace

bool Save(const IR::ResourcePlan& plan, std::vector<uint8_t>& out) {
	std::vector<uint8_t> bytes;
	Writer               w(bytes);
	std::unordered_map<const IR::Inst*, uint32_t> index;
	index.reserve(plan.value_storage.size());
	for (const auto& inst: plan.value_storage) index.emplace(&inst, static_cast<uint32_t>(index.size()));

	w.Put(Magic);
	w.Put(plan.stage);
	w.Put(plan.shader_hash);
	w.Put(plan.user_data_base);
	w.Put(plan.user_data_count);
	// The values: every node first (opcode, flags), then their operands (a node's may come after it in the storage).
	w.Put(static_cast<uint32_t>(plan.value_storage.size()));
	for (const auto& inst: plan.value_storage) {
		w.Put(inst.GetOpcode());
		w.Put(inst.Flags<uint64_t>());
	}
	for (const auto& inst: plan.value_storage) {
		w.Put(static_cast<uint32_t>(inst.NumArgs()));
		for (size_t i = 0; i < inst.NumArgs(); ++i) {
			// (Phi operands' blocks are left out: a plan's are all null.)
			if (inst.GetOpcode() == IR::ValueOpcode::Phi && inst.PhiBlock(i) != nullptr) return false;
			if (!PutValue(w, index, inst.Arg(i))) return false;
		}
	}
	w.PutVector(plan.memory_info);
	w.Put(static_cast<uint32_t>(plan.descriptor_sources.size()));
	for (const auto& source: plan.descriptor_sources) {
		w.Put(source.dword_count);
		for (uint32_t i = 0; i < source.dword_count; ++i)
			if (!PutValue(w, index, source.dwords[i])) return false;
		for (uint32_t i = source.dword_count; i < source.dwords.size(); ++i)
			if (!source.dwords[i].IsEmpty()) return false;
		w.Put(static_cast<uint8_t>(source.indirect_image.has_value()));
		if (source.indirect_image) w.Put(*source.indirect_image);
	}
	w.Put(static_cast<uint32_t>(plan.control_flow.size()));
	for (const auto& block: plan.control_flow) {
		if (!PutValue(w, index, block.condition)) return false;
		w.PutVector(block.successors);
		w.PutVector(block.sources);
	}
	w.PutVector(plan.materialization_sources);
	w.Put(static_cast<uint32_t>(plan.srt_reads.size()));
	for (const auto& read: plan.srt_reads) {
		if (!PutValue(w, index, read.value)) return false;
		w.Put(read.flat_offset);
	}
	w.PutVector(plan.clean_flat_slots);
	w.Put(static_cast<uint8_t>(plan.requires_specialization_memory | plan.srt_plan_complete << 1 |
	                           plan.resource_tracking_complete << 2 | (plan.linear_srt != nullptr) << 3));
	PutInfo(w, plan.info);
	w.Put(plan.uniform_fill.fill);
	for (const auto& value: plan.uniform_fill.values)
		if (!PutValue(w, index, value)) return false;
	out = std::move(bytes);
	return true;
}

bool Load(std::span<const uint8_t> bytes, IR::ResourcePlan& plan, bool* defer_linear) {
	plan = IR::ResourcePlan {};
	Reader   r(bytes);
	uint32_t magic = 0, count = 0;
	if (!r.Get(magic) || magic != Magic || !r.Get(plan.stage) || !r.Get(plan.shader_hash) || !r.Get(plan.user_data_base) ||
	    !r.Get(plan.user_data_count) || !r.Get(count))
		return false;
	std::vector<IR::Inst*> insts;
	insts.reserve(count);
	for (uint32_t i = 0; i < count; ++i) {
		IR::ValueOpcode opcode {};
		uint64_t        flags = 0;
		if (!r.Get(opcode) || !r.Get(flags)) return plan = {}, false;
		insts.push_back(&plan.value_storage.emplace_back(opcode, flags));
	}
	for (auto* inst: insts) {
		uint32_t args = 0;
		if (!r.Get(args)) return plan = {}, false;
		const bool phi = inst->GetOpcode() == IR::ValueOpcode::Phi;
		if (!phi && args != inst->NumArgs() && inst->NumArgs() != 0) return plan = {}, false;
		for (uint32_t i = 0; i < args; ++i) {
			IR::Value value;
			if (!GetValue(r, insts, value)) return plan = {}, false;
			if (phi) inst->AddPhiOperand(nullptr, value);
			else inst->SetArg(i, value);
		}
	}
	if (!r.GetVector(plan.memory_info) || !r.Get(count)) return plan = {}, false;
	plan.descriptor_sources.resize(count);
	for (auto& source: plan.descriptor_sources) {
		uint8_t indirect = 0;
		if (!r.Get(source.dword_count) || source.dword_count > source.dwords.size()) return plan = {}, false;
		for (uint32_t i = 0; i < source.dword_count; ++i)
			if (!GetValue(r, insts, source.dwords[i])) return plan = {}, false;
		if (!r.Get(indirect)) return plan = {}, false;
		if (indirect != 0) {
			IR::DescriptorSource::IndirectImage image {};
			if (!r.Get(image)) return plan = {}, false;
			source.indirect_image = image;
		}
	}
	if (!r.Get(count)) return plan = {}, false;
	plan.control_flow.resize(count);
	for (auto& block: plan.control_flow)
		if (!GetValue(r, insts, block.condition) || !r.GetVector(block.successors) || !r.GetVector(block.sources))
			return plan = {}, false;
	if (!r.GetVector(plan.materialization_sources) || !r.Get(count)) return plan = {}, false;
	plan.srt_reads.resize(count);
	for (auto& read: plan.srt_reads)
		if (!GetValue(r, insts, read.value) || !r.Get(read.flat_offset)) return plan = {}, false;
	uint8_t bits = 0;
	if (!r.GetVector(plan.clean_flat_slots) || !r.Get(bits) || !GetInfo(r, plan.info) || !r.Get(plan.uniform_fill.fill))
		return plan = {}, false;
	for (auto& value: plan.uniform_fill.values)
		if (!GetValue(r, insts, value)) return plan = {}, false;
	if (!r.AtEnd()) return plan = {}, false;
	plan.requires_specialization_memory = (bits & 1u) != 0;
	plan.srt_plan_complete              = (bits & 2u) != 0;
	plan.resource_tracking_complete     = (bits & 4u) != 0;
	const bool linear = (bits & 8u) != 0;
	if (defer_linear != nullptr) {
		*defer_linear = linear;
		return true;
	}
	// (As ExtractResourcePlan ends: from the plan alone.)
	if (linear) IR::BuildLinearSrtPlan(plan);
	return !linear || plan.linear_srt != nullptr || (plan = {}, false);
}

} // namespace ResourcePlanSerial
