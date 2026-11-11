#include "graphics/host_gpu/renderer/pipeline/pipelineSeeds.h"

#include "common/logging/log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <xxhash.h>

namespace Libs::Graphics::PipelineSeeds {

namespace fs = std::filesystem;

namespace {

constexpr char     FileMagic[8]  = {'K', 'Y', 'S', 'E', 'E', 'D', '0', '1'};
constexpr uint8_t  KindGraphics  = 1;
constexpr uint8_t  KindCompute   = 2;
constexpr uint32_t MaxRecordSize = 16u << 20u;

// ---- writer ---------------------------------------------------------------------------------

class Writer {
public:
	std::vector<uint8_t> bytes;
	bool                 ok = true;

	void Put(const void* data, size_t size) {
		const auto* p = static_cast<const uint8_t*>(data);
		bytes.insert(bytes.end(), p, p + size);
	}
	template <typename T>
	void Put(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		Put(&value, sizeof(T));
	}
	template <typename T>
	void PutArray(const T* data, uint32_t count) {
		Put(count);
		if (count != 0) {
			Put(data, sizeof(T) * count);
		}
	}
	// A Vulkan struct with its pNext (and any pointer the caller cleared) zeroed.
	template <typename T>
	void PutStruct(T copy) {
		copy.pNext = nullptr;
		Put(copy);
	}
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> bytes): m_bytes(bytes) {}
	bool Get(void* data, size_t size) {
		if (m_at + size > m_bytes.size()) {
			m_ok = false;
			std::memset(data, 0, size);
			return false;
		}
		std::memcpy(data, m_bytes.data() + m_at, size);
		m_at += size;
		return true;
	}
	template <typename T>
	T Get() {
		T value {};
		Get(&value, sizeof(T));
		return value;
	}
	[[nodiscard]] bool Ok() const { return m_ok; }

private:
	std::span<const uint8_t> m_bytes;
	size_t                   m_at = 0;
	bool                     m_ok = true;
};

// Everything a replayed create info points to.
struct Arena {
	std::vector<std::unique_ptr<uint8_t[]>> blocks;
	template <typename T>
	T* Alloc(size_t count = 1) {
		auto block = std::make_unique<uint8_t[]>(sizeof(T) * std::max<size_t>(count, 1));
		std::memset(block.get(), 0, sizeof(T) * std::max<size_t>(count, 1));
		T* p = reinterpret_cast<T*>(block.get());
		blocks.push_back(std::move(block));
		return p;
	}
	template <typename T>
	T* ReadArray(Reader& r, uint32_t& count) {
		count = r.Get<uint32_t>();
		if (count > 4096) {
			count = 0;
			return nullptr;
		}
		if (count == 0) {
			return nullptr;
		}
		T* p = Alloc<T>(count);
		r.Get(p, sizeof(T) * count);
		return p;
	}
};

// pNext chains: only the structures shaders.cpp chains. false: one this does not know.
bool PutChain(Writer& w, const void* next) {
	while (next != nullptr) {
		const auto* base = static_cast<const VkBaseInStructure*>(next);
		w.Put(static_cast<uint32_t>(base->sType));
		switch (base->sType) {
			case VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO:
				w.PutStruct(*static_cast<const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*>(next));
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO: {
				auto s = *static_cast<const VkPipelineRenderingCreateInfo*>(next);
				w.PutArray(s.pColorAttachmentFormats, s.colorAttachmentCount);
				s.pColorAttachmentFormats = nullptr;
				w.PutStruct(s);
				break;
			}
			case VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_AMD: {
				auto s = *static_cast<const VkAttachmentSampleCountInfoAMD*>(next);
				w.PutArray(s.pColorAttachmentSamples, s.pColorAttachmentSamples != nullptr ? s.colorAttachmentCount : 0);
				s.pColorAttachmentSamples = nullptr;
				w.PutStruct(s);
				break;
			}
			case VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT:
				w.PutStruct(*static_cast<const VkPipelineViewportDepthClipControlCreateInfoEXT*>(next));
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT:
				w.PutStruct(*static_cast<const VkPipelineRasterizationDepthClipStateCreateInfoEXT*>(next));
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT:
				w.PutStruct(*static_cast<const VkPipelineRasterizationProvokingVertexStateCreateInfoEXT*>(next));
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_REDUCTION_STATE_CREATE_INFO_NV:
				w.PutStruct(*static_cast<const VkPipelineCoverageReductionStateCreateInfoNV*>(next));
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT: {
				auto s = *static_cast<const VkPipelineColorWriteCreateInfoEXT*>(next);
				w.PutArray(s.pColorWriteEnables, s.pColorWriteEnables != nullptr ? s.attachmentCount : 0);
				s.pColorWriteEnables = nullptr;
				w.PutStruct(s);
				break;
			}
			default: w.ok = false; return false;
		}
		next = base->pNext;
	}
	w.Put(uint32_t {0});
	return true;
}

template <typename T>
void* ReadStruct(Arena& a, Reader& r) {
	T* s = a.Alloc<T>();
	r.Get(s, sizeof(T));
	s->pNext = nullptr;
	return s;
}

const void* ReadChain(Arena& a, Reader& r) {
	const void*          head = nullptr;
	VkBaseOutStructure*  last = nullptr;
	for (int guard = 0; guard < 16 && r.Ok(); guard++) {
		const auto type = static_cast<VkStructureType>(r.Get<uint32_t>());
		if (type == 0) {
			break;
		}
		void* s = nullptr;
		switch (type) {
			case VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO:
				s = ReadStruct<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo>(a, r);
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO: {
				uint32_t n       = 0;
				auto*    formats = a.ReadArray<VkFormat>(r, n);
				auto*    p       = static_cast<VkPipelineRenderingCreateInfo*>(ReadStruct<VkPipelineRenderingCreateInfo>(a, r));
				p->pColorAttachmentFormats = formats;
				s                          = p;
				break;
			}
			case VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_AMD: {
				uint32_t n       = 0;
				auto*    samples = a.ReadArray<VkSampleCountFlagBits>(r, n);
				auto*    p       = static_cast<VkAttachmentSampleCountInfoAMD*>(ReadStruct<VkAttachmentSampleCountInfoAMD>(a, r));
				p->pColorAttachmentSamples = samples;
				s                          = p;
				break;
			}
			case VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT:
				s = ReadStruct<VkPipelineViewportDepthClipControlCreateInfoEXT>(a, r);
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT:
				s = ReadStruct<VkPipelineRasterizationDepthClipStateCreateInfoEXT>(a, r);
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT:
				s = ReadStruct<VkPipelineRasterizationProvokingVertexStateCreateInfoEXT>(a, r);
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_REDUCTION_STATE_CREATE_INFO_NV:
				s = ReadStruct<VkPipelineCoverageReductionStateCreateInfoNV>(a, r);
				break;
			case VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT: {
				uint32_t n       = 0;
				auto*    enables = a.ReadArray<VkBool32>(r, n);
				auto*    p       = static_cast<VkPipelineColorWriteCreateInfoEXT*>(ReadStruct<VkPipelineColorWriteCreateInfoEXT>(a, r));
				p->pColorWriteEnables = enables;
				s                     = p;
				break;
			}
			default: return nullptr;
		}
		auto* node  = static_cast<VkBaseOutStructure*>(s);
		node->pNext = nullptr;
		if (last == nullptr) {
			head = node;
		} else {
			last->pNext = node;
		}
		last = node;
	}
	return head;
}

// ---- state ----------------------------------------------------------------------------------

struct State {
	std::mutex                                 mutex;
	fs::path                                   directory;
	uint64_t                                   session   = 0;
	bool                                       recording = false;
	std::unordered_map<VkShaderModule, uint64_t> modules;   // handle -> SPIR-V hash
	std::unordered_set<uint64_t>               spirv_known; // files in seed-spirv
	std::unordered_set<uint64_t>               records;     // record hashes in pipelines.seeds
	uint32_t                                   new_records = 0;
	bool                                       logged_unknown = false;

	// writer thread
	struct Job {
		bool                 spirv = false;
		uint64_t             hash  = 0;
		std::vector<uint8_t> bytes;
	};
	std::deque<Job>         queue;
	std::condition_variable wake, idle;
	bool                    writing = false;
	std::thread             writer;
};

State& S() {
	static State s;
	return s;
}

fs::path SpirvPath(const fs::path& dir, uint64_t hash) {
	char name[32];
	std::snprintf(name, sizeof(name), "%016llx.spv", static_cast<unsigned long long>(hash));
	return dir / "seed-spirv" / name;
}

fs::path SeedsPath(const fs::path& dir) {
	return dir / "pipelines.seeds";
}

void WriterLoop() {
	auto& s = S();
	std::error_code ec;
	fs::create_directories(s.directory / "seed-spirv", ec);
	for (;;) {
		State::Job job;
		{
			std::unique_lock lock(s.mutex);
			s.wake.wait(lock, [&] { return !s.queue.empty(); });
			job       = std::move(s.queue.front());
			s.queue.pop_front();
			s.writing = true;
		}
		if (job.spirv) {
			const auto path = SpirvPath(s.directory, job.hash);
			const auto tmp  = fs::path(path.native() + fs::path::string_type(1, '~'));
			{
				std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
				out.write(reinterpret_cast<const char*>(job.bytes.data()), static_cast<std::streamsize>(job.bytes.size()));
			}
			fs::rename(tmp, path, ec);
		} else {
			const auto path  = SeedsPath(s.directory);
			const bool fresh = !fs::exists(path, ec) || fs::file_size(path, ec) == 0;
			std::ofstream out(path, std::ios::binary | std::ios::app);
			if (fresh) {
				out.write(FileMagic, sizeof(FileMagic));
			}
			const auto size = static_cast<uint32_t>(job.bytes.size());
			out.write(reinterpret_cast<const char*>(&size), sizeof(size));
			out.write(reinterpret_cast<const char*>(&job.hash), sizeof(job.hash));
			out.write(reinterpret_cast<const char*>(job.bytes.data()), size);
		}
		{
			std::lock_guard lock(s.mutex);
			s.writing = false;
			if (s.queue.empty()) {
				s.idle.notify_all();
			}
		}
	}
}

void Enqueue(State::Job job) {
	auto& s = S();
	s.queue.push_back(std::move(job));
	if (!s.writer.joinable()) {
		s.writer = std::thread(WriterLoop);
		s.writer.detach();
	}
	s.wake.notify_one();
}

// Calls `fn(hash, payload)` for each intact record of the file.
template <typename F>
void ReadSeeds(const fs::path& path, F&& fn) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		return;
	}
	std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (file.size() < sizeof(FileMagic) || std::memcmp(file.data(), FileMagic, sizeof(FileMagic)) != 0) {
		return;
	}
	size_t at = sizeof(FileMagic);
	while (at + 12 <= file.size()) {
		uint32_t size = 0;
		uint64_t hash = 0;
		std::memcpy(&size, file.data() + at, 4);
		std::memcpy(&hash, file.data() + at + 4, 8);
		at += 12;
		if (size > MaxRecordSize || at + size > file.size()) {
			break; // a cut tail (a run that ended while writing)
		}
		std::span<const uint8_t> payload(file.data() + at, size);
		at += size;
		if (XXH3_64bits(payload.data(), payload.size()) != hash) {
			continue;
		}
		fn(hash, payload);
	}
}

bool PutModule(Writer& w, VkShaderModule module) {
	auto& s  = S();
	auto  it = s.modules.find(module);
	if (it == s.modules.end()) {
		w.ok = false;
		return false;
	}
	w.Put(it->second);
	return true;
}

void PutLayout(Writer& w, const Layout& layout) {
	for (const auto& b: layout.bindings) {
		if (b.pImmutableSamplers != nullptr) {
			w.ok = false;
		}
	}
	w.Put(static_cast<uint32_t>(layout.bindings.size()));
	for (auto b: layout.bindings) {
		b.pImmutableSamplers = nullptr;
		w.Put(static_cast<VkDescriptorSetLayoutBinding>(b));
	}
	w.Put(static_cast<uint8_t>(layout.push_descriptors));
	w.Put(static_cast<uint8_t>(layout.bindless));
	w.Put(static_cast<VkPushConstantRange>(layout.push_constants));
}

void PutStage(Writer& w, const VkPipelineShaderStageCreateInfo& stage) {
	w.Put(static_cast<uint32_t>(stage.flags));
	w.Put(static_cast<uint32_t>(stage.stage));
	PutModule(w, stage.module);
	const std::string name = stage.pName != nullptr ? stage.pName : "main";
	w.PutArray(name.data(), static_cast<uint32_t>(name.size()));
	const auto* spec = stage.pSpecializationInfo;
	w.Put(static_cast<uint8_t>(spec != nullptr));
	if (spec != nullptr) {
		w.PutArray(spec->pMapEntries, spec->mapEntryCount);
		w.PutArray(static_cast<const uint8_t*>(spec->pData), static_cast<uint32_t>(spec->dataSize));
	}
	PutChain(w, stage.pNext);
}

void Commit(Writer& w) {
	auto& s = S();
	if (!w.ok) {
		if (!s.logged_unknown) {
			s.logged_unknown = true;
			LOGF("Pipeline seeds: a pipeline could not be recorded (an unknown structure or module)\n");
		}
		return;
	}
	const auto hash = XXH3_64bits(w.bytes.data(), w.bytes.size());
	if (!s.records.insert(hash).second) {
		return;
	}
	s.new_records++;
	Enqueue({.spirv = false, .hash = hash, .bytes = std::move(w.bytes)});
}

} // namespace

// ---- recording ------------------------------------------------------------------------------

bool PrecompileRequested() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PRECOMPILE");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

void Open(const fs::path& directory, uint64_t session) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	s.directory = directory;
	s.session   = session;
	const char* env = std::getenv("KYTY_PIPELINE_SEEDS");
	s.recording = !directory.empty() && !PrecompileRequested() && !(env != nullptr && env[0] == '0');
	if (directory.empty()) {
		return;
	}
	std::error_code ec;
	for (fs::directory_iterator it(directory / "seed-spirv", ec), end; !ec && it != end; it.increment(ec)) {
		if (it->path().extension() == ".spv") {
			s.spirv_known.insert(std::strtoull(it->path().stem().string().c_str(), nullptr, 16));
		}
	}
	uint32_t total = 0, mine = 0;
	ReadSeeds(SeedsPath(directory), [&](uint64_t hash, std::span<const uint8_t> payload) {
		s.records.insert(hash);
		total++;
		Reader r(payload);
		(void)r.Get<uint8_t>();
		mine += r.Get<uint64_t>() == session ? 1u : 0u;
	});
	LOGF("Pipeline seeds: %u in %s (%u of this GPU and recompiler), %zu SPIR-V modules; recording %s\n", total,
	     SeedsPath(directory).string().c_str(), mine, s.spirv_known.size(),
	     s.recording ? "on" : (PrecompileRequested() ? "off (precompile)" : "off (KYTY_PIPELINE_SEEDS=0)"));
}

bool Recording() {
	return S().recording;
}

void NoteModule(vk::ShaderModule module, std::span<const uint32_t> code) {
	auto& s = S();
	if (!s.recording || module == nullptr) {
		return;
	}
	const auto      hash = XXH3_64bits(code.data(), code.size_bytes());
	std::lock_guard lock(s.mutex);
	s.modules[static_cast<VkShaderModule>(module)] = hash;
	if (s.spirv_known.insert(hash).second) {
		State::Job job {.spirv = true, .hash = hash};
		job.bytes.assign(reinterpret_cast<const uint8_t*>(code.data()),
		                 reinterpret_cast<const uint8_t*>(code.data()) + code.size_bytes());
		Enqueue(std::move(job));
	}
}

void RecordGraphics(const vk::GraphicsPipelineCreateInfo& vk_info, const Layout& layout) {
	auto& s = S();
	if (!s.recording) {
		return;
	}
	const auto&     info = static_cast<const VkGraphicsPipelineCreateInfo&>(vk_info);
	std::lock_guard lock(s.mutex);
	Writer          w;
	w.Put(KindGraphics);
	w.Put(s.session);
	w.Put(static_cast<uint32_t>(info.flags & ~VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT));
	PutLayout(w, layout);
	w.Put(info.stageCount);
	for (uint32_t i = 0; i < info.stageCount; i++) {
		PutStage(w, info.pStages[i]);
	}
	// Each state: present flag, its arrays, the struct (pointers cleared), its chain.
	const auto present = [&](const void* p) {
		w.Put(static_cast<uint8_t>(p != nullptr));
		return p != nullptr;
	};
	if (present(info.pVertexInputState)) {
		auto v = *info.pVertexInputState;
		w.PutArray(v.pVertexBindingDescriptions, v.vertexBindingDescriptionCount);
		w.PutArray(v.pVertexAttributeDescriptions, v.vertexAttributeDescriptionCount);
		v.pVertexBindingDescriptions   = nullptr;
		v.pVertexAttributeDescriptions = nullptr;
		w.PutStruct(v);
		PutChain(w, info.pVertexInputState->pNext);
	}
	if (present(info.pInputAssemblyState)) {
		w.PutStruct(*info.pInputAssemblyState);
		PutChain(w, info.pInputAssemblyState->pNext);
	}
	if (present(info.pTessellationState)) {
		w.PutStruct(*info.pTessellationState);
		PutChain(w, info.pTessellationState->pNext);
	}
	if (present(info.pViewportState)) {
		auto v = *info.pViewportState;
		w.PutArray(v.pViewports, v.pViewports != nullptr ? v.viewportCount : 0);
		w.PutArray(v.pScissors, v.pScissors != nullptr ? v.scissorCount : 0);
		v.pViewports = nullptr;
		v.pScissors  = nullptr;
		w.PutStruct(v);
		PutChain(w, info.pViewportState->pNext);
	}
	if (present(info.pRasterizationState)) {
		w.PutStruct(*info.pRasterizationState);
		PutChain(w, info.pRasterizationState->pNext);
	}
	if (present(info.pMultisampleState)) {
		auto       m     = *info.pMultisampleState;
		const auto words = (static_cast<uint32_t>(m.rasterizationSamples) + 31u) / 32u;
		w.PutArray(m.pSampleMask, m.pSampleMask != nullptr ? words : 0);
		m.pSampleMask = nullptr;
		w.PutStruct(m);
		PutChain(w, info.pMultisampleState->pNext);
	}
	if (present(info.pDepthStencilState)) {
		w.PutStruct(*info.pDepthStencilState);
		PutChain(w, info.pDepthStencilState->pNext);
	}
	if (present(info.pColorBlendState)) {
		auto c = *info.pColorBlendState;
		w.PutArray(c.pAttachments, c.pAttachments != nullptr ? c.attachmentCount : 0);
		c.pAttachments = nullptr;
		w.PutStruct(c);
		PutChain(w, info.pColorBlendState->pNext);
	}
	if (present(info.pDynamicState)) {
		auto d = *info.pDynamicState;
		w.PutArray(d.pDynamicStates, d.dynamicStateCount);
		d.pDynamicStates = nullptr;
		w.PutStruct(d);
		PutChain(w, info.pDynamicState->pNext);
	}
	PutChain(w, info.pNext);
	Commit(w);
}

void RecordCompute(const vk::ComputePipelineCreateInfo& vk_info, const Layout& layout) {
	auto& s = S();
	if (!s.recording) {
		return;
	}
	const auto&     info = static_cast<const VkComputePipelineCreateInfo&>(vk_info);
	std::lock_guard lock(s.mutex);
	Writer          w;
	w.Put(KindCompute);
	w.Put(s.session);
	w.Put(static_cast<uint32_t>(info.flags & ~VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT));
	PutLayout(w, layout);
	PutStage(w, info.stage);
	PutChain(w, info.pNext);
	Commit(w);
}

void Flush() {
	auto&            s = S();
	std::unique_lock lock(s.mutex);
	s.idle.wait_for(lock, std::chrono::seconds(30), [&] { return s.queue.empty() && !s.writing; });
	if (s.new_records != 0) {
		LOGF("Pipeline seeds: %u new pipelines recorded (%zu in all)\n", s.new_records, s.records.size());
	}
}

// ---- replay ---------------------------------------------------------------------------------

namespace {

struct Replayed {
	Arena                         arena;
	std::vector<vk::ShaderModule> modules;
	vk::DescriptorSetLayout       set_layout = nullptr;
	vk::PipelineLayout            layout     = nullptr;
	vk::Device                    device;
	~Replayed() {
		for (auto m: modules) device.destroyShaderModule(m, nullptr);
		if (layout != nullptr) device.destroyPipelineLayout(layout, nullptr);
		if (set_layout != nullptr) device.destroyDescriptorSetLayout(set_layout, nullptr);
	}
};

bool ReadFile(const fs::path& path, std::vector<uint32_t>& words) {
	std::ifstream in(path, std::ios::binary | std::ios::ate);
	if (!in) return false;
	const auto size = static_cast<size_t>(in.tellg());
	if (size == 0 || size % 4 != 0) return false;
	words.resize(size / 4);
	in.seekg(0);
	in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
	return static_cast<bool>(in);
}

bool ReadLayout(Replayed& out, Reader& r, vk::DescriptorSetLayout bindless) {
	uint32_t n        = 0;
	auto*    bindings = out.arena.ReadArray<VkDescriptorSetLayoutBinding>(r, n);
	const bool push      = r.Get<uint8_t>() != 0;
	const bool uses_bl   = r.Get<uint8_t>() != 0;
	const auto push_range = r.Get<VkPushConstantRange>();
	if (!r.Ok() || (uses_bl && bindless == nullptr)) return false;
	VkDescriptorSetLayoutCreateInfo set {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	set.flags        = push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0;
	set.bindingCount = n;
	set.pBindings    = bindings;
	if (out.device.createDescriptorSetLayout(reinterpret_cast<const vk::DescriptorSetLayoutCreateInfo*>(&set), nullptr,
	                                         &out.set_layout) != vk::Result::eSuccess) {
		out.set_layout = nullptr;
		return false;
	}
	std::array<vk::DescriptorSetLayout, 2> sets {out.set_layout, bindless};
	vk::PipelineLayoutCreateInfo           info {};
	info.setLayoutCount         = uses_bl ? 2 : 1;
	info.pSetLayouts            = sets.data();
	info.pushConstantRangeCount = 1;
	vk::PushConstantRange range {};
	std::memcpy(static_cast<void*>(&range), &push_range, sizeof(range));
	info.pPushConstantRanges = &range;
	if (out.device.createPipelineLayout(&info, nullptr, &out.layout) != vk::Result::eSuccess) {
		out.layout = nullptr;
		return false;
	}
	return true;
}

bool ReadStage(Replayed& out, Reader& r, VkPipelineShaderStageCreateInfo& stage, const fs::path& dir) {
	stage       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	stage.flags = r.Get<uint32_t>();
	stage.stage = static_cast<VkShaderStageFlagBits>(r.Get<uint32_t>());
	const auto hash = r.Get<uint64_t>();
	uint32_t   n    = 0;
	auto*      name = out.arena.ReadArray<char>(r, n);
	char*      z    = out.arena.Alloc<char>(n + 1);
	if (n != 0) std::memcpy(z, name, n);
	stage.pName = z;
	if (r.Get<uint8_t>() != 0) {
		auto* spec          = out.arena.Alloc<VkSpecializationInfo>();
		spec->pMapEntries   = out.arena.ReadArray<VkSpecializationMapEntry>(r, spec->mapEntryCount);
		uint32_t size       = 0;
		spec->pData         = out.arena.ReadArray<uint8_t>(r, size);
		spec->dataSize      = size;
		stage.pSpecializationInfo = spec;
	}
	stage.pNext = ReadChain(out.arena, r);
	if (!r.Ok()) return false;
	std::vector<uint32_t> words;
	if (!ReadFile(SpirvPath(dir, hash), words)) return false;
	vk::ShaderModuleCreateInfo info {};
	info.codeSize = words.size() * 4;
	info.pCode    = words.data();
	vk::ShaderModule module = nullptr;
	if (out.device.createShaderModule(&info, nullptr, &module) != vk::Result::eSuccess) return false;
	out.modules.push_back(module);
	stage.module = static_cast<VkShaderModule>(module);
	return true;
}

template <typename T>
const T* ReadState(Replayed& out, Reader& r) {
	if (r.Get<uint8_t>() == 0) return nullptr;
	auto* s = static_cast<T*>(ReadStruct<T>(out.arena, r));
	s->pNext = ReadChain(out.arena, r);
	return s;
}

enum class Outcome { Compiled, Cached, Failed };

Outcome ReplayOne(std::span<const uint8_t> payload, vk::Device device, vk::PipelineCache cache,
                  std::span<const vk::PipelineCache> older, vk::DescriptorSetLayout bindless, bool control,
                  const fs::path& dir) {
	Replayed out;
	out.device = device;
	Reader r(payload);
	const auto kind = r.Get<uint8_t>();
	(void)r.Get<uint64_t>();
	const auto flags = r.Get<uint32_t>();
	if (!ReadLayout(out, r, bindless)) return Outcome::Failed;

	VkPipeline pipeline = VK_NULL_HANDLE;
	const auto create   = [&](auto& info, auto fn) {
		// The caches that already hold it first, without compiling.
		if (control) {
			std::vector<vk::PipelineCache> lookups(older.begin(), older.end());
			lookups.push_back(cache);
			for (auto c: lookups) {
				auto lookup = info;
				lookup.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
				if (fn(c, lookup) == VK_SUCCESS && pipeline != VK_NULL_HANDLE) {
					device.destroyPipeline(vk::Pipeline(pipeline), nullptr);
					return Outcome::Cached;
				}
				if (pipeline != VK_NULL_HANDLE) device.destroyPipeline(vk::Pipeline(pipeline), nullptr);
				pipeline = VK_NULL_HANDLE;
			}
		}
		const auto result = fn(cache, info);
		if (pipeline != VK_NULL_HANDLE) device.destroyPipeline(vk::Pipeline(pipeline), nullptr);
		return result == VK_SUCCESS ? Outcome::Compiled : Outcome::Failed;
	};

	if (kind == KindCompute) {
		VkComputePipelineCreateInfo info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		info.flags = flags;
		if (!ReadStage(out, r, info.stage, dir)) return Outcome::Failed;
		info.pNext              = ReadChain(out.arena, r);
		info.layout             = static_cast<VkPipelineLayout>(out.layout);
		info.basePipelineIndex  = -1;
		if (!r.Ok()) return Outcome::Failed;
		return create(info, [&](vk::PipelineCache c, const VkComputePipelineCreateInfo& i) {
			return static_cast<VkResult>(device.createComputePipelines(
			    c, 1, reinterpret_cast<const vk::ComputePipelineCreateInfo*>(&i), nullptr,
			    reinterpret_cast<vk::Pipeline*>(&pipeline)));
		});
	}
	if (kind != KindGraphics) return Outcome::Failed;
	VkGraphicsPipelineCreateInfo info {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	info.flags      = flags;
	info.stageCount = r.Get<uint32_t>();
	if (info.stageCount == 0 || info.stageCount > 5) return Outcome::Failed;
	auto* stages = out.arena.Alloc<VkPipelineShaderStageCreateInfo>(info.stageCount);
	for (uint32_t i = 0; i < info.stageCount; i++) {
		if (!ReadStage(out, r, stages[i], dir)) return Outcome::Failed;
	}
	info.pStages = stages;
	if (r.Get<uint8_t>() != 0) {
		uint32_t nb = 0, na = 0;
		auto*    b = out.arena.ReadArray<VkVertexInputBindingDescription>(r, nb);
		auto*    a = out.arena.ReadArray<VkVertexInputAttributeDescription>(r, na);
		auto*    v = static_cast<VkPipelineVertexInputStateCreateInfo*>(ReadStruct<VkPipelineVertexInputStateCreateInfo>(out.arena, r));
		v->pVertexBindingDescriptions   = b;
		v->pVertexAttributeDescriptions = a;
		v->pNext                        = ReadChain(out.arena, r);
		info.pVertexInputState          = v;
	}
	info.pInputAssemblyState = ReadState<VkPipelineInputAssemblyStateCreateInfo>(out, r);
	info.pTessellationState  = ReadState<VkPipelineTessellationStateCreateInfo>(out, r);
	if (r.Get<uint8_t>() != 0) {
		uint32_t nv = 0, ns = 0;
		auto*    vp = out.arena.ReadArray<VkViewport>(r, nv);
		auto*    sc = out.arena.ReadArray<VkRect2D>(r, ns);
		auto*    v  = static_cast<VkPipelineViewportStateCreateInfo*>(ReadStruct<VkPipelineViewportStateCreateInfo>(out.arena, r));
		v->pViewports = vp;
		v->pScissors  = sc;
		v->pNext      = ReadChain(out.arena, r);
		info.pViewportState = v;
	}
	info.pRasterizationState = ReadState<VkPipelineRasterizationStateCreateInfo>(out, r);
	if (r.Get<uint8_t>() != 0) {
		uint32_t n    = 0;
		auto*    mask = out.arena.ReadArray<VkSampleMask>(r, n);
		auto*    m    = static_cast<VkPipelineMultisampleStateCreateInfo*>(ReadStruct<VkPipelineMultisampleStateCreateInfo>(out.arena, r));
		m->pSampleMask = mask;
		m->pNext       = ReadChain(out.arena, r);
		info.pMultisampleState = m;
	}
	info.pDepthStencilState = ReadState<VkPipelineDepthStencilStateCreateInfo>(out, r);
	if (r.Get<uint8_t>() != 0) {
		uint32_t n   = 0;
		auto*    att = out.arena.ReadArray<VkPipelineColorBlendAttachmentState>(r, n);
		auto*    c   = static_cast<VkPipelineColorBlendStateCreateInfo*>(ReadStruct<VkPipelineColorBlendStateCreateInfo>(out.arena, r));
		c->pAttachments = att;
		c->pNext        = ReadChain(out.arena, r);
		info.pColorBlendState = c;
	}
	if (r.Get<uint8_t>() != 0) {
		uint32_t n  = 0;
		auto*    ds = out.arena.ReadArray<VkDynamicState>(r, n);
		auto*    d  = static_cast<VkPipelineDynamicStateCreateInfo*>(ReadStruct<VkPipelineDynamicStateCreateInfo>(out.arena, r));
		d->pDynamicStates = ds;
		d->pNext          = ReadChain(out.arena, r);
		info.pDynamicState = d;
	}
	info.pNext             = ReadChain(out.arena, r);
	info.layout            = static_cast<VkPipelineLayout>(out.layout);
	info.basePipelineIndex = -1;
	if (!r.Ok()) return Outcome::Failed;
	return create(info, [&](vk::PipelineCache c, const VkGraphicsPipelineCreateInfo& i) {
		return static_cast<VkResult>(device.createGraphicsPipelines(
		    c, 1, reinterpret_cast<const vk::GraphicsPipelineCreateInfo*>(&i), nullptr,
		    reinterpret_cast<vk::Pipeline*>(&pipeline)));
	});
}

} // namespace

ReplayResult Replay(vk::Device device, vk::PipelineCache driver_cache, std::span<const vk::PipelineCache> older,
                    vk::DescriptorSetLayout bindless, bool cache_control) {
	ReplayResult result;
	auto&        s = S();
	if (s.directory.empty()) {
		LOGF("Pipeline seeds: no shader cache directory (no title id); nothing to precompile\n");
		return result;
	}
	std::vector<std::vector<uint8_t>> seeds;
	std::unordered_set<uint64_t>      seen; // a merged package may repeat records
	ReadSeeds(SeedsPath(s.directory), [&](uint64_t hash, std::span<const uint8_t> payload) {
		if (!seen.insert(hash).second) {
			return;
		}
		Reader r(payload);
		(void)r.Get<uint8_t>();
		if (r.Get<uint64_t>() != s.session) {
			result.other++;
			return;
		}
		seeds.emplace_back(payload.begin(), payload.end());
	});
	result.total = static_cast<uint32_t>(seeds.size());

	const char* progress_env  = std::getenv("KYTY_PRECOMPILE_PROGRESS");
	const fs::path progress   = progress_env != nullptr && *progress_env != '\0' ? fs::u8path(progress_env) : fs::path();
	const fs::path stop_file  = progress.empty() ? fs::path() : fs::path(progress.native() + fs::path(".stop").native());
	const uint32_t threads    = std::max(1u, std::thread::hardware_concurrency());
	LOGF("Pipeline seeds: precompiling %u pipelines on %u threads (%u of another GPU or recompiler skipped)\n",
	     result.total, threads, result.other);

	std::atomic<uint32_t> next {0}, done {0}, compiled {0}, cached {0}, failed {0};
	std::atomic<bool>     stop {false};
	const auto            begin = std::chrono::steady_clock::now();
	std::vector<std::thread> workers;
	for (uint32_t t = 0; t < threads; t++) {
		workers.emplace_back([&] {
			for (;;) {
				if (stop.load(std::memory_order_relaxed)) return;
				const auto i = next.fetch_add(1);
				if (i >= seeds.size()) return;
				switch (ReplayOne(seeds[i], device, driver_cache, older, bindless, cache_control, s.directory)) {
					case Outcome::Compiled: compiled++; break;
					case Outcome::Cached: cached++; break;
					case Outcome::Failed: failed++; break;
				}
				done++;
			}
		});
	}
	const auto write_progress = [&](bool final) {
		if (progress.empty()) return;
		const auto tmp = fs::path(progress.native() + fs::path(".tmp").native());
		{
			std::ofstream out(tmp, std::ios::trunc);
			out << done.load() << ' ' << result.total << ' ' << compiled.load() << ' ' << failed.load() << ' '
			    << (final ? 1 : 0) << '\n';
		}
		std::error_code ec;
		fs::rename(tmp, progress, ec);
	};
	while (done.load() < result.total) {
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		write_progress(false);
		std::error_code ec;
		if (!stop_file.empty() && fs::exists(stop_file, ec)) {
			stop = true;
			break;
		}
	}
	for (auto& w: workers) w.join();
	result.compiled = compiled;
	result.cached   = cached;
	result.failed   = failed;
	result.stopped  = stop.load();
	const auto secs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count();
	LOGF("Pipeline seeds: precompile %s in %lld s: %u compiled, %u already cached, %u failed, %u of %u done\n",
	     result.stopped ? "stopped" : "finished", static_cast<long long>(secs), result.compiled, result.cached,
	     result.failed, done.load(), result.total);
	write_progress(true);
	if (!stop_file.empty()) {
		std::error_code ec;
		fs::remove(stop_file, ec);
	}
	return result;
}

} // namespace Libs::Graphics::PipelineSeeds
