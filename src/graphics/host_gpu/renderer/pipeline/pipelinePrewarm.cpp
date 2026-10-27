#include "graphics/host_gpu/renderer/pipeline/pipelinePrewarm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <tuple>
#include <unordered_set>
#include <xxhash.h>
#include <set>
#include <string>
#include <spirv-tools/libspirv.hpp>

namespace Libs::Graphics::Prewarm {

namespace {

// File: magic, then records { u32 kind, u32 words, u64 key, u32 payload[words] }.
// kind 'S': SPIR-V blob (key = content hash), 'C': compute pipeline, 'G': graphics pipeline.
constexpr char     Magic[8] = {'K', 'P', 'R', 'E', 'W', 'R', 'M', '2'};
constexpr uint32_t KindSpirv    = 'S';
constexpr uint32_t KindCompute  = 'C';
// 'G' (first version) copied the trailing padding of three structs: stack garbage, so the key of
// the same pipeline differed on every creation (graphics: 11 served / 713 missed). Ignored now.
constexpr uint32_t KindGraphicsV1 = 'G';
constexpr uint32_t KindGraphics   = 'g';
constexpr uint32_t KindMeta       = 'M'; // key: SPIR-V hash, payload: guest shader hash (2 words)
constexpr uint32_t MaxWords     = 64u * 1024u * 1024u;

using Clock = std::chrono::steady_clock;

uint64_t Hash(const void* data, size_t bytes, uint64_t seed) {
	const uint64_t h = XXH3_64bits_withSeed(data, bytes, seed);
	return h == 0 ? 1 : h;
}

// ---- flat word stream for pipeline descriptions ----------------------------------------------

struct Writer {
	std::vector<uint32_t> w;
	void U(uint32_t v) { w.push_back(v); }
	void U64(uint64_t v) {
		U(static_cast<uint32_t>(v));
		U(static_cast<uint32_t>(v >> 32u));
	}
	void Raw(const void* p, size_t bytes) {
		const size_t n = (bytes + 3) / 4;
		const size_t o = w.size();
		w.resize(o + n, 0);
		std::memcpy(w.data() + o, p, bytes);
	}
	// the POD fields of a Vulkan struct after its sType/pNext header, without trailing padding
	template <typename T>
	void Used(const T& s, size_t end) {
		constexpr size_t off = offsetof(VkBaseInStructure, pNext) + sizeof(void*);
		Raw(reinterpret_cast<const uint8_t*>(&s) + off, end - off);
	}
	template <typename T>
	void Tail(const T& s) {
		constexpr size_t off = offsetof(VkBaseInStructure, pNext) + sizeof(void*);
		Raw(reinterpret_cast<const uint8_t*>(&s) + off, sizeof(T) - off);
	}
};

struct Reader {
	std::span<const uint32_t> w;
	size_t                    pos = 0;
	bool                      ok  = true;
	uint32_t                  U() {
		if (pos >= w.size()) {
			ok = false;
			return 0;
		}
		return w[pos++];
	}
	uint64_t U64() {
		const uint64_t lo = U();
		return lo | (static_cast<uint64_t>(U()) << 32u);
	}
	void Raw(void* p, size_t bytes) {
		const size_t n = (bytes + 3) / 4;
		if (pos + n > w.size()) {
			ok = false;
			std::memset(p, 0, bytes);
			return;
		}
		std::memcpy(p, w.data() + pos, bytes);
		pos += n;
	}
	template <typename T>
	void Used(T& s, size_t end) {
		constexpr size_t off = offsetof(VkBaseInStructure, pNext) + sizeof(void*);
		Raw(reinterpret_cast<uint8_t*>(&s) + off, end - off);
	}
	template <typename T>
	void Tail(T& s) {
		constexpr size_t off = offsetof(VkBaseInStructure, pNext) + sizeof(void*);
		Raw(reinterpret_cast<uint8_t*>(&s) + off, sizeof(T) - off);
	}
};

// ---- state -----------------------------------------------------------------------------------

enum class State : uint8_t { Queued, Building, Ready, Done };

struct Entry {
	State        state    = State::Queued;
	vk::Pipeline pipeline = nullptr;
};

struct Pending {
	uint32_t              kind = 0;
	uint64_t              key  = 0;
	std::vector<uint32_t> desc;
};

struct Context {
	std::mutex                                   mutex;
	std::condition_variable                      cv;
	std::unordered_map<uint64_t, uint64_t>       hash_of_module;  // module -> SPIR-V hash
	std::unordered_map<uint64_t, uint64_t>       spirv_offset;    // SPIR-V hash -> payload offset
	std::unordered_map<uint64_t, uint32_t>       spirv_words;
	std::unordered_set<uint64_t>                 known;           // pipeline keys in the file
	std::unordered_map<uint64_t, Entry>          entries;
	std::vector<std::shared_ptr<Pending>>        queue;
	size_t                                       next = 0;
	std::vector<std::thread>                     threads;
	std::FILE*                                   out = nullptr;
	uint64_t                                     out_pos = 0;
	std::filesystem::path                        path;
	bool                                         started = false;
	bool                                         stop    = false;
	size_t                                       built   = 0;
	size_t                                       failed  = 0;
	size_t                                       queued_c = 0;
	size_t                                       queued_g = 0;
	uint64_t                                     served[2] = {};
	uint64_t                                     missed[2] = {};
	Clock::time_point                            start_time;
	Clock::time_point                            next_report;
	std::vector<std::tuple<uint64_t, uint64_t, std::vector<uint32_t>>> early_spirv; // hash, guest, code
	std::unordered_map<uint64_t, uint64_t>       guest_of_spirv;
	std::unordered_map<uint64_t, std::vector<uint64_t>> spirv_of_guest; // in the seed at boot
	std::unordered_set<uint64_t>                 seed_spirv;          // SPIR-V in the seed at boot
	std::vector<std::shared_ptr<Pending>>        seed_descs;          // pipeline records at boot
	bool                                         seed_loaded = false;
	int                                          diag_left   = 40;
	uint64_t                                     taken_twice[2] = {};
	uint64_t                                     not_in_seed[2] = {};
	GraphicContext*                              graphics     = nullptr;
	vk::PipelineCache                            driver_cache = nullptr;
};

Context& Ctx() {
	static auto* ctx = new Context; // never destroyed: worker threads may outlive static destruction
	return *ctx;
}

// runtime counters (lock-free; read by Report)
std::atomic<uint64_t> g_runtime_compiles[2];
std::atomic<uint64_t> g_runtime_ns[2];
std::atomic<uint64_t> g_runtime_slow[2];
std::atomic<uint64_t> g_runtime_sync[2];
std::atomic<uint64_t> g_runtime_max_ns {0};
std::atomic<uint64_t> g_translations {0};
std::atomic<uint64_t> g_translation_ns {0};
std::atomic<uint64_t> g_translation_max_ns {0};

uint64_t ModuleId(vk::ShaderModule module) {
	return reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(module));
}

bool Seek(std::FILE* f, uint64_t pos) {
#if defined(_WIN32)
	return _fseeki64(f, static_cast<__int64>(pos), SEEK_SET) == 0;
#else
	return fseeko(f, static_cast<off_t>(pos), SEEK_SET) == 0;
#endif
}

// caller holds the mutex
void WriteRecordLocked(Context& ctx, uint32_t kind, uint64_t key, std::span<const uint32_t> payload) {
	if (ctx.out == nullptr) {
		return;
	}
	const auto n = static_cast<uint32_t>(payload.size());
	std::fwrite(&kind, 4, 1, ctx.out);
	std::fwrite(&n, 4, 1, ctx.out);
	std::fwrite(&key, 8, 1, ctx.out);
	const uint64_t payload_pos = ctx.out_pos + 16;
	std::fwrite(payload.data(), 4, n, ctx.out);
	std::fflush(ctx.out);
	ctx.out_pos = payload_pos + uint64_t {n} * 4u;
	if (kind == KindSpirv) {
		ctx.spirv_offset[key] = payload_pos;
		ctx.spirv_words[key]  = n;
	}
}

void ReportLocked(Context& ctx, const char* when) {
	const auto ms = [](uint64_t ns) { return static_cast<unsigned long long>(ns / 1'000'000u); };
	std::printf(
	    "PipelineStats (%s): prewarm %zu/%zu built (%zu compute + %zu graphics recorded, %zu failed) | "
	    "gameplay compute %llu served / %llu missed (%llu new, %llu again), graphics %llu served / %llu missed (%llu new, %llu again) | "
	    "runtime compiles: compute %llu (%llu ms, %llu sync, %llu >50ms), graphics %llu (%llu ms, %llu sync, "
	    "%llu >50ms), worst %llu ms | translations %llu (%llu ms, worst %llu ms)\n",
	    when, ctx.built, ctx.queue.size(), ctx.queued_c, ctx.queued_g, ctx.failed,
	    static_cast<unsigned long long>(ctx.served[0]), static_cast<unsigned long long>(ctx.missed[0]),
	    static_cast<unsigned long long>(ctx.not_in_seed[0]), static_cast<unsigned long long>(ctx.taken_twice[0]),
	    static_cast<unsigned long long>(ctx.served[1]), static_cast<unsigned long long>(ctx.missed[1]),
	    static_cast<unsigned long long>(ctx.not_in_seed[1]), static_cast<unsigned long long>(ctx.taken_twice[1]),
	    static_cast<unsigned long long>(g_runtime_compiles[0].load()), ms(g_runtime_ns[0].load()),
	    static_cast<unsigned long long>(g_runtime_sync[0].load()),
	    static_cast<unsigned long long>(g_runtime_slow[0].load()),
	    static_cast<unsigned long long>(g_runtime_compiles[1].load()), ms(g_runtime_ns[1].load()),
	    static_cast<unsigned long long>(g_runtime_sync[1].load()),
	    static_cast<unsigned long long>(g_runtime_slow[1].load()), ms(g_runtime_max_ns.load()),
	    static_cast<unsigned long long>(g_translations.load()), ms(g_translation_ns.load()),
	    ms(g_translation_max_ns.load()));
	std::fflush(stdout);
}

void MaybeReport() {
	auto& ctx = Ctx();
	const auto now = Clock::now();
	std::unique_lock lock(ctx.mutex, std::try_to_lock);
	if (!lock.owns_lock() || !ctx.started || now < ctx.next_report) {
		return;
	}
	ctx.next_report = now + std::chrono::seconds(60);
	ReportLocked(ctx, "60s");
}

void AtomicMax(std::atomic<uint64_t>& a, uint64_t v) {
	uint64_t cur = a.load(std::memory_order_relaxed);
	while (cur < v && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
	}
}

// ---- builders --------------------------------------------------------------------------------

struct Layout {
	vk::DescriptorSetLayout set    = nullptr;
	vk::PipelineLayout      layout = nullptr;
};

void WriteLayout(Writer& wr, std::span<const vk::DescriptorSetLayoutBinding> bindings, uint32_t layout_flags,
                 uint32_t push_stages) {
	wr.U(layout_flags);
	wr.U(push_stages);
	wr.U(static_cast<uint32_t>(bindings.size()));
	for (const auto& b: bindings) {
		wr.U(b.binding);
		wr.U(static_cast<uint32_t>(b.descriptorType));
		wr.U(b.descriptorCount);
		wr.U(static_cast<uint32_t>(b.stageFlags));
	}
}

bool BuildLayout(Reader& rd, vk::Device device, uint32_t push_size, Layout& out) {
	const uint32_t layout_flags = rd.U();
	const uint32_t push_stages  = rd.U();
	const uint32_t nb           = rd.U();
	if (!rd.ok || nb > 4096) {
		return false;
	}
	std::vector<vk::DescriptorSetLayoutBinding> bindings(nb);
	for (auto& b: bindings) {
		b.binding         = rd.U();
		b.descriptorType  = static_cast<vk::DescriptorType>(rd.U());
		b.descriptorCount = rd.U();
		b.stageFlags      = static_cast<vk::ShaderStageFlags>(rd.U());
	}
	if (!rd.ok) {
		return false;
	}
	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = static_cast<vk::DescriptorSetLayoutCreateFlags>(layout_flags);
	create.bindingCount = nb;
	create.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&create, nullptr, &out.set) != vk::Result::eSuccess) {
		return false;
	}
	const vk::PushConstantRange push {static_cast<vk::ShaderStageFlags>(push_stages), 0, push_size};
	vk::PipelineLayoutCreateInfo info {};
	info.setLayoutCount         = 1;
	info.pSetLayouts            = &out.set;
	info.pushConstantRangeCount = push_stages != 0 ? 1 : 0;
	info.pPushConstantRanges    = &push;
	return device.createPipelineLayout(&info, nullptr, &out.layout) == vk::Result::eSuccess;
	// layouts stay alive: the pipeline is used with gameplay's identically defined ones
}

vk::ShaderModule LoadModule(Context& ctx, std::FILE* f, uint64_t hash, vk::Device device) {
	uint64_t offset = 0;
	uint32_t words  = 0;
	{
		std::lock_guard lock(ctx.mutex);
		auto            it = ctx.spirv_offset.find(hash);
		if (it == ctx.spirv_offset.end()) {
			return nullptr;
		}
		offset = it->second;
		words  = ctx.spirv_words[hash];
	}
	std::vector<uint32_t> code(words);
	if (!Seek(f, offset) || std::fread(code.data(), 4, words, f) != words || Hash(code.data(), words * 4u, 0) != hash) {
		return nullptr;
	}
	// Seed SPIR-V is validated before it reaches the driver: a module recorded by a build that
	// emitted invalid code (b0a4787) crashed the driver in every later replay.
	// KYTY_PREWARM_VALIDATE=0 skips it.
	static const bool validate = [] {
		const char* v = std::getenv("KYTY_PREWARM_VALIDATE");
		return v == nullptr || std::strcmp(v, "0") != 0;
	}();
	if (validate) {
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
		std::string          first;
		tools.SetMessageConsumer([&first](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
			if (first.empty() && m != nullptr) first = m;
		});
		spvtools::ValidatorOptions options;
		options.SetAllowOffsetTextureOperand(true);
		if (!tools.Validate(code.data(), code.size(), options)) {
			std::printf("Prewarm: seed SPIR-V %016llx (%u words) is invalid, skipped: %s\n",
			            static_cast<unsigned long long>(hash), words, first.c_str());
			std::fflush(stdout);
			return nullptr;
		}
	}
	vk::ShaderModuleCreateInfo info {};
	info.codeSize           = code.size() * 4u;
	info.pCode              = code.data();
	vk::ShaderModule module = nullptr;
	return device.createShaderModule(&info, nullptr, &module) == vk::Result::eSuccess ? module : nullptr;
}

constexpr uint32_t PushSize = 128; // ShaderRecompiler::IR::NativePushConstantSize, carried in the record

uint32_t FastCompileFlags(uint64_t words);

vk::Pipeline BuildCompute(Context& ctx, std::FILE* f, const Pending& p) {
	Reader         rd {p.desc};
	const uint64_t spirv          = rd.U64();
	const uint32_t subgroup       = rd.U();
	const uint32_t create_flags   = rd.U();
	const uint32_t push_size      = rd.U();
	Layout         layout;
	if (!rd.ok || !BuildLayout(rd, ctx.graphics->device, push_size, layout)) {
		return nullptr;
	}
	auto module = LoadModule(ctx, f, spirv, ctx.graphics->device);
	if (module == nullptr) {
		return nullptr;
	}
	vk::PipelineShaderStageCreateInfo                     stage {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo sg {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	if (subgroup != 0) {
		sg.requiredSubgroupSize = subgroup;
		stage.pNext             = &sg;
	}
	vk::ComputePipelineCreateInfo info {};
	// A record made before its SPIR-V size was known has no DISABLE_OPTIMIZATION; apply the same
	// size rule here so a huge shader (Yotei: 448K words) is not optimized for minutes in the warm-up.
	uint32_t module_words = 0;
	{
		std::lock_guard lock(ctx.mutex);
		if (const auto it = ctx.spirv_words.find(spirv); it != ctx.spirv_words.end()) module_words = it->second;
	}
	info.flags             = static_cast<vk::PipelineCreateFlags>(create_flags | FastCompileFlags(module_words));
	info.stage             = stage;
	info.layout            = layout.layout;
	info.basePipelineIndex = -1;
	vk::Pipeline pipeline  = nullptr;
	const auto   r = ctx.graphics->device.createComputePipelines(ctx.driver_cache, 1, &info, nullptr, &pipeline);
	ctx.graphics->device.destroyShaderModule(module, nullptr);
	return r == vk::Result::eSuccess ? pipeline : nullptr;
}

// Graphics description: the create info walked struct by struct. Any pNext the walker does not
// know makes the pipeline unrecordable (RegisterGraphics returns 0).
bool KnownChain(const void* next, std::initializer_list<VkStructureType> allowed) {
	for (auto* s = static_cast<const VkBaseInStructure*>(next); s != nullptr; s = s->pNext) {
		if (std::find(allowed.begin(), allowed.end(), s->sType) == allowed.end()) {
			return false;
		}
	}
	return true;
}

template <typename T>
const T* FindChain(const void* next, VkStructureType type) {
	for (auto* s = static_cast<const VkBaseInStructure*>(next); s != nullptr; s = s->pNext) {
		if (s->sType == type) {
			return reinterpret_cast<const T*>(s);
		}
	}
	return nullptr;
}

vk::Pipeline BuildGraphics(Context& ctx, std::FILE* f, const Pending& p) {
	auto           device = ctx.graphics->device;
	Reader         rd {p.desc};
	VkGraphicsPipelineCreateInfo info {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	info.flags               = rd.U();
	const uint32_t push_size = rd.U();
	// stages
	const uint32_t nstages = rd.U();
	if (!rd.ok || nstages > 6) {
		return nullptr;
	}
	VkPipelineShaderStageCreateInfo stages[6] {};
	vk::ShaderModule                modules[6] {};
	bool                            ok = true;
	for (uint32_t i = 0; i < nstages; i++) {
		stages[i].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[i].stage  = static_cast<VkShaderStageFlagBits>(rd.U());
		stages[i].pName  = "main";
		const uint64_t h = rd.U64();
		modules[i]       = rd.ok ? LoadModule(ctx, f, h, device) : nullptr;
		ok               = ok && modules[i] != nullptr;
		stages[i].module = modules[i];
	}
	info.stageCount = nstages;
	info.pStages    = stages;

	VkPipelineVertexInputStateCreateInfo vi {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	std::vector<VkVertexInputBindingDescription>   vb;
	std::vector<VkVertexInputAttributeDescription> va;
	if (rd.U() != 0) {
		vb.resize(std::min<uint32_t>(rd.U(), 64));
		rd.Raw(vb.data(), vb.size() * sizeof(VkVertexInputBindingDescription));
		va.resize(std::min<uint32_t>(rd.U(), 64));
		rd.Raw(va.data(), va.size() * sizeof(VkVertexInputAttributeDescription));
		vi.vertexBindingDescriptionCount   = static_cast<uint32_t>(vb.size());
		vi.pVertexBindingDescriptions      = vb.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(va.size());
		vi.pVertexAttributeDescriptions    = va.data();
		info.pVertexInputState             = &vi;
	}
	VkPipelineInputAssemblyStateCreateInfo ia {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	if (rd.U() != 0) {
		rd.Used(ia, offsetof(VkPipelineInputAssemblyStateCreateInfo, primitiveRestartEnable) + 4);
		info.pInputAssemblyState = &ia;
	}
	VkPipelineTessellationStateCreateInfo ts {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
	if (rd.U() != 0) {
		rd.Used(ts, offsetof(VkPipelineTessellationStateCreateInfo, patchControlPoints) + 4);
		info.pTessellationState = &ts;
	}
	VkPipelineViewportStateCreateInfo vp {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	VkPipelineViewportDepthClipControlCreateInfoEXT dcc {
	    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
	if (rd.U() != 0) {
		vp.flags         = rd.U();
		vp.viewportCount = rd.U();
		vp.scissorCount  = rd.U();
		if (rd.U() != 0) {
			rd.Used(dcc, offsetof(VkPipelineViewportDepthClipControlCreateInfoEXT, negativeOneToOne) + 4);
			vp.pNext = &dcc;
		}
		info.pViewportState = &vp;
	}
	VkPipelineRasterizationStateCreateInfo rs {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	VkPipelineRasterizationDepthClipStateCreateInfoEXT clip {
	    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT};
	VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pv {
	    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT};
	if (rd.U() != 0) {
		rd.Used(rs, offsetof(VkPipelineRasterizationStateCreateInfo, lineWidth) + 4);
		const void* chain = nullptr;
		if (rd.U() != 0) {
			rd.Used(clip, offsetof(VkPipelineRasterizationDepthClipStateCreateInfoEXT, depthClipEnable) + 4);
			clip.pNext = chain;
			chain      = &clip;
		}
		if (rd.U() != 0) {
			rd.Used(pv, offsetof(VkPipelineRasterizationProvokingVertexStateCreateInfoEXT, provokingVertexMode) + 4);
			pv.pNext = chain;
			chain    = &pv;
		}
		rs.pNext                 = chain;
		info.pRasterizationState = &rs;
	}
	VkPipelineMultisampleStateCreateInfo ms {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	if (rd.U() != 0) {
		ms.flags                 = rd.U();
		ms.rasterizationSamples  = static_cast<VkSampleCountFlagBits>(rd.U());
		ms.sampleShadingEnable   = rd.U();
		const uint32_t mss       = rd.U();
		std::memcpy(&ms.minSampleShading, &mss, 4);
		ms.alphaToCoverageEnable = rd.U();
		ms.alphaToOneEnable      = rd.U();
		info.pMultisampleState   = &ms;
	}
	VkPipelineDepthStencilStateCreateInfo ds {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
	if (rd.U() != 0) {
		rd.Used(ds, offsetof(VkPipelineDepthStencilStateCreateInfo, maxDepthBounds) + 4);
		info.pDepthStencilState = &ds;
	}
	VkPipelineColorBlendStateCreateInfo cb {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	VkPipelineColorWriteCreateInfoEXT   cw {VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT};
	std::vector<VkPipelineColorBlendAttachmentState> atts;
	std::vector<VkBool32>                            cwe;
	if (rd.U() != 0) {
		cb.flags         = rd.U();
		cb.logicOpEnable = rd.U();
		cb.logicOp       = static_cast<VkLogicOp>(rd.U());
		atts.resize(std::min<uint32_t>(rd.U(), 8));
		rd.Raw(atts.data(), atts.size() * sizeof(VkPipelineColorBlendAttachmentState));
		rd.Raw(cb.blendConstants, sizeof(cb.blendConstants));
		cb.attachmentCount = static_cast<uint32_t>(atts.size());
		cb.pAttachments    = atts.data();
		if (rd.U() != 0) {
			cwe.resize(std::min<uint32_t>(rd.U(), 8));
			rd.Raw(cwe.data(), cwe.size() * sizeof(VkBool32));
			cw.attachmentCount    = static_cast<uint32_t>(cwe.size());
			cw.pColorWriteEnables = cwe.data();
			cb.pNext              = &cw;
		}
		info.pColorBlendState = &cb;
	}
	VkPipelineDynamicStateCreateInfo dyn {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
	std::vector<VkDynamicState>      dyns;
	if (rd.U() != 0) {
		dyns.resize(std::min<uint32_t>(rd.U(), 128));
		rd.Raw(dyns.data(), dyns.size() * sizeof(VkDynamicState));
		dyn.dynamicStateCount = static_cast<uint32_t>(dyns.size());
		dyn.pDynamicStates    = dyns.data();
		info.pDynamicState    = &dyn;
	}
	VkPipelineRenderingCreateInfo ren {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
	std::vector<VkFormat>         formats;
	if (rd.U() != 0) {
		ren.viewMask = rd.U();
		formats.resize(std::min<uint32_t>(rd.U(), 8));
		rd.Raw(formats.data(), formats.size() * sizeof(VkFormat));
		ren.colorAttachmentCount    = static_cast<uint32_t>(formats.size());
		ren.pColorAttachmentFormats = formats.data();
		ren.depthAttachmentFormat   = static_cast<VkFormat>(rd.U());
		ren.stencilAttachmentFormat = static_cast<VkFormat>(rd.U());
		info.pNext                  = &ren;
	}
	Layout layout;
	ok = ok && rd.ok && BuildLayout(rd, device, push_size, layout) && rd.ok;
	vk::Pipeline pipeline = nullptr;
	if (ok) {
		info.layout             = layout.layout;
		info.basePipelineIndex  = -1;
		const auto r = device.createGraphicsPipelines(
		    ctx.driver_cache, 1, reinterpret_cast<const vk::GraphicsPipelineCreateInfo*>(&info), nullptr, &pipeline);
		if (r != vk::Result::eSuccess) {
			pipeline = nullptr;
		}
	}
	for (uint32_t i = 0; i < nstages; i++) {
		if (modules[i] != nullptr) {
			device.destroyShaderModule(modules[i], nullptr);
		}
	}
	return pipeline;
}

// Crash guard: every replay build is logged as begun/ended in <seed>.inflight. A build begun but
// never ended (the driver crashed the process inside it) is skipped by the next replay.
std::mutex                  g_inflight_mutex;
std::FILE*                  g_inflight = nullptr;
std::set<uint64_t>          g_crashed;

void InflightMark(char what, uint64_t key) {
	std::lock_guard lock(g_inflight_mutex);
	if (g_inflight == nullptr) return;
	std::fprintf(g_inflight, "%c %016llx\n", what, static_cast<unsigned long long>(key));
	std::fflush(g_inflight);
}

void OpenInflight(const std::filesystem::path& seed) {
	const auto path = std::filesystem::path(seed.string() + ".inflight");
	std::set<uint64_t> begun;
	if (std::FILE* in = std::fopen(path.string().c_str(), "r"); in != nullptr) {
		char               what = 0;
		unsigned long long key  = 0;
		while (std::fscanf(in, " %c %llx", &what, &key) == 2) {
			if (what == 'B') begun.insert(key);
			else if (what == 'E') begun.erase(key);
			else if (what == 'X') g_crashed.insert(key);
		}
		std::fclose(in);
	}
	for (const auto key: begun) g_crashed.insert(key);
	g_inflight = std::fopen(path.string().c_str(), "w");
	if (g_inflight != nullptr) {
		for (const auto key: g_crashed) {
			std::fprintf(g_inflight, "X %016llx\n", static_cast<unsigned long long>(key));
			std::printf("Prewarm: pipeline %016llx crashed a previous replay; skipped\n",
			            static_cast<unsigned long long>(key));
		}
		std::fflush(g_inflight);
		std::fflush(stdout);
	}
}

bool CrashedBefore(uint64_t key) {
	std::lock_guard lock(g_inflight_mutex);
	return g_crashed.contains(key);
}

void Worker() {
	auto&      ctx = Ctx();
	std::FILE* f   = std::fopen(ctx.path.string().c_str(), "rb");
	for (;;) {
		std::shared_ptr<Pending> rec;
		{
			std::lock_guard lock(ctx.mutex);
			if (ctx.stop) {
				break;
			}
			while (ctx.next < ctx.queue.size()) {
				auto& cand = ctx.queue[ctx.next++];
				auto  it   = ctx.entries.find(cand->key);
				if (it != ctx.entries.end() && it->second.state == State::Queued) {
					it->second.state = State::Building;
					rec              = cand;
					break;
				}
			}
			if (rec == nullptr) {
				break;
			}
		}
		vk::Pipeline pipeline = nullptr;
		const bool   crashed_before = CrashedBefore(rec->key);
		InflightMark('B', rec->key);
		const auto   build_start = Clock::now();
		const size_t desc_words  = rec->desc.size();
		if (f != nullptr && !crashed_before) {
			pipeline = rec->kind == KindCompute ? BuildCompute(ctx, f, *rec) : BuildGraphics(ctx, f, *rec);
		}
		if (const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - build_start).count(); ms >= 2000.0) {
			std::printf("Prewarm: slow %s pipeline key=%016llx desc_words=%zu built in %.0f ms\n",
			            rec->kind == KindCompute ? "compute" : "graphics", static_cast<unsigned long long>(rec->key),
			            desc_words, ms);
			std::fflush(stdout);
		}
		InflightMark('E', rec->key);
		rec->desc.clear();
		rec->desc.shrink_to_fit();
		{
			std::lock_guard lock(ctx.mutex);
			auto&           e = ctx.entries[rec->key];
			if (pipeline != nullptr) {
				e.state    = State::Ready;
				e.pipeline = pipeline;
				ctx.built++;
			} else {
				e.state = State::Done;
				ctx.failed++;
			}
			const size_t total = ctx.queue.size();
			const size_t n     = ctx.built + ctx.failed;
			const size_t step  = std::max<size_t>(1, total / 10);
			if (n % step == 0 || n == total) {
				const auto s =
				    std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - ctx.start_time).count();
				std::printf("Prewarm: %zu/%zu pipelines (%lld s)\n", n, total, static_cast<long long>(s));
				std::fflush(stdout);
				if (n == total) {
					ReportLocked(ctx, "prewarm finished");
				}
			}
		}
		ctx.cv.notify_all();
	}
	if (f != nullptr) {
		std::fclose(f);
	}
}

uint32_t FastCompileFlags(uint64_t words) {
	static const uint64_t fast_compile_words = [] {
		const char* v = std::getenv("KYTY_FAST_COMPILE_WORDS");
		return v != nullptr ? std::strtoull(v, nullptr, 10) : 100000ull;
	}();
	return fast_compile_words != 0 && words > fast_compile_words
	           ? static_cast<uint32_t>(VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT)
	           : 0u;
}

// ---- miss diagnostics (KYTY_PREWARM_DIAG=<n> first misses, default 40; 0 = off) ---------------

bool ReadSpirv(Context& ctx, std::FILE* f, uint64_t h, std::vector<uint32_t>& out) {
	auto it = ctx.spirv_offset.find(h);
	if (it == ctx.spirv_offset.end() || f == nullptr) return false;
	out.resize(ctx.spirv_words[h]);
	return Seek(f, it->second) && std::fread(out.data(), 4, out.size(), f) == out.size();
}

void DiffWords(const char* what, std::span<const uint32_t> a, std::span<const uint32_t> b) {
	size_t diff = 0;
	char   buf[512];
	int    len = 0;
	const size_t n = std::min(a.size(), b.size());
	for (size_t i = 0; i < n; i++) {
		if (a[i] != b[i]) {
			if (diff < 10 && len < 440) {
				len += std::snprintf(buf + len, sizeof(buf) - len, " [%zu] %08x->%08x", i, a[i], b[i]);
			}
			diff++;
		}
	}
	buf[len] = '\0';
	std::printf("PrewarmMiss:   %s: seed %zu words, now %zu words, %zu of %zu differ:%s\n", what, a.size(),
	            b.size(), diff, n, buf);
}

void DiffSpirvVsSeed(Context& ctx, std::FILE* f, const char* what, uint64_t now_hash) {
	const auto gi    = ctx.guest_of_spirv.find(now_hash);
	const uint64_t g = gi != ctx.guest_of_spirv.end() ? gi->second : 0;
	if (ctx.seed_spirv.contains(now_hash)) {
		std::printf("PrewarmMiss:   %s SPIR-V %016llx (guest %016llx) is in the seed: same code\n", what,
		            static_cast<unsigned long long>(now_hash), static_cast<unsigned long long>(g));
		return;
	}
	const auto cands = ctx.spirv_of_guest.find(g);
	std::printf("PrewarmMiss:   %s SPIR-V %016llx (guest %016llx, %u words) NOT in the seed; seed has %zu SPIR-V of this guest shader\n",
	            what, static_cast<unsigned long long>(now_hash), static_cast<unsigned long long>(g),
	            ctx.spirv_words[now_hash], g != 0 && cands != ctx.spirv_of_guest.end() ? cands->second.size() : 0);
	if (g == 0 || cands == ctx.spirv_of_guest.end() || cands->second.empty()) return;
	uint64_t best = cands->second.front();
	for (const auto c: cands->second) {
		if (ctx.spirv_words[c] == ctx.spirv_words[now_hash]) { best = c; break; }
	}
	std::vector<uint32_t> a, b;
	if (ReadSpirv(ctx, f, best, a) && ReadSpirv(ctx, f, now_hash, b)) DiffWords("SPIR-V words", a, b);
}

void DiagMissLocked(Context& ctx, uint32_t kind, uint64_t key, std::span<const uint32_t> desc) {
	static const int limit = [] {
		const char* v = std::getenv("KYTY_PREWARM_DIAG");
		return v != nullptr ? std::atoi(v) : 40;
	}();
	if (!ctx.seed_loaded || ctx.diag_left <= 0 || limit <= 0) return;
	if (40 - ctx.diag_left >= limit) return;
	ctx.diag_left--;
	std::FILE* f = std::fopen(ctx.path.string().c_str(), "rb");
	std::printf("PrewarmMiss: %s pipeline %016llx not in the seed (%zu desc words)\n",
	            kind == KindCompute ? "compute" : "graphics", static_cast<unsigned long long>(key), desc.size());
	// stage SPIR-V hashes
	std::vector<uint64_t> stages;
	if (kind == KindCompute && desc.size() >= 2) {
		stages.push_back(desc[0] | (uint64_t {desc[1]} << 32u));
	} else if (kind == KindGraphics && desc.size() >= 3) {
		for (uint32_t i = 0; i < desc[2] && 3 + i * 3 + 2 < desc.size(); i++) {
			stages.push_back(desc[3 + i * 3 + 1] | (uint64_t {desc[3 + i * 3 + 2]} << 32u));
		}
	}
	bool all_in_seed = true;
	for (size_t i = 0; i < stages.size(); i++) {
		char what[32];
		std::snprintf(what, sizeof(what), "stage %zu", i);
		DiffSpirvVsSeed(ctx, f, what, stages[i]);
		all_in_seed = all_in_seed && ctx.seed_spirv.contains(stages[i]);
	}
	if (all_in_seed) {
		// same shaders: find the seed record of the same kind with the most equal words
		const Pending* best = nullptr;
		size_t         best_eq = 0;
		for (const auto& r: ctx.seed_descs) {
			if (r->kind != kind || r->desc.size() != desc.size()) continue;
			size_t eq = 0;
			for (size_t i = 0; i < desc.size(); i++) eq += r->desc[i] == desc[i] ? 1 : 0;
			if (best == nullptr || eq > best_eq) { best = r.get(); best_eq = eq; }
		}
		if (best != nullptr) {
			DiffWords("state words (nearest seed record)", best->desc, desc);
		} else {
			std::printf("PrewarmMiss:   no seed record of the same size (state differs in shape)\n");
		}
	}
	if (f != nullptr) std::fclose(f);
	std::fflush(stdout);
}

uint64_t RecordPipeline(uint32_t kind, Writer& wr) {
	auto&          ctx = Ctx();
	const uint64_t key = Hash(wr.w.data(), wr.w.size() * 4u, kind);
	std::lock_guard lock(ctx.mutex);
	if (ctx.known.insert(key).second) {
		ctx.not_in_seed[kind == KindGraphics ? 1 : 0]++;
		DiagMissLocked(ctx, kind, key, wr.w);
		WriteRecordLocked(ctx, kind, key, wr.w);
	}
	return key;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PREWARM");
		return v == nullptr || std::strcmp(v, "0") != 0;
	}();
	return enabled;
}

void NoteSpirv(vk::ShaderModule module, std::span<const uint32_t> spirv, uint64_t guest_hash) {
	if (!Enabled() || module == nullptr || spirv.empty()) {
		return;
	}
	const uint64_t  h   = Hash(spirv.data(), spirv.size() * 4u, 0);
	auto&           ctx = Ctx();
	std::lock_guard lock(ctx.mutex);
	ctx.hash_of_module[ModuleId(module)] = h;
	if (!ctx.started) {
		// before the seed is open (shaders translate before the first pipeline): kept and written
		// at EnsureStarted, else these pipelines would never be recorded
		ctx.early_spirv.emplace_back(h, guest_hash, std::vector<uint32_t>(spirv.begin(), spirv.end()));
		return;
	}
	if (!ctx.spirv_offset.contains(h)) {
		WriteRecordLocked(ctx, KindSpirv, h, spirv);
	}
	if (guest_hash != 0 && ctx.guest_of_spirv.emplace(h, guest_hash).second) {
		const uint32_t g[2] = {static_cast<uint32_t>(guest_hash), static_cast<uint32_t>(guest_hash >> 32u)};
		WriteRecordLocked(ctx, KindMeta, h, g);
	}
}

static bool SpirvOf(vk::ShaderModule module, uint64_t& hash) {
	auto&           ctx = Ctx();
	std::lock_guard lock(ctx.mutex);
	auto            it = ctx.hash_of_module.find(ModuleId(module));
	if (it == ctx.hash_of_module.end() || !ctx.spirv_offset.contains(it->second)) {
		return false;
	}
	hash = it->second;
	return true;
}

uint64_t Register(vk::ShaderModule module, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                  uint32_t required_subgroup_size, uint32_t layout_flags) {
	if (!Enabled() || module == nullptr) {
		return 0;
	}
	uint64_t spirv = 0;
	if (!SpirvOf(module, spirv)) {
		return 0;
	}
	for (const auto& b: bindings) {
		if (b.pImmutableSamplers != nullptr) {
			return 0;
		}
	}
	uint32_t words = 0;
	{
		auto&           ctx = Ctx();
		std::lock_guard lock(ctx.mutex);
		words = ctx.spirv_words[spirv];
	}
	Writer wr;
	wr.U64(spirv);
	wr.U(required_subgroup_size);
	wr.U(FastCompileFlags(words));
	wr.U(PushSize);
	WriteLayout(wr, bindings, layout_flags, static_cast<uint32_t>(VK_SHADER_STAGE_COMPUTE_BIT));
	return RecordPipeline(KindCompute, wr);
}

uint64_t RegisterGraphics(const vk::GraphicsPipelineCreateInfo&            cinfo,
                          std::span<const vk::DescriptorSetLayoutBinding> bindings, uint32_t push_stages,
                          uint32_t layout_flags) {
	if (!Enabled()) {
		return 0;
	}
	const auto& info = static_cast<const VkGraphicsPipelineCreateInfo&>(cinfo);
	if (info.stageCount > 6 || !KnownChain(info.pNext, {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO})) {
		return 0;
	}
	for (const auto& b: bindings) {
		if (b.pImmutableSamplers != nullptr) {
			return 0;
		}
	}
	Writer wr;
	wr.U(info.flags);
	wr.U(PushSize);
	wr.U(info.stageCount);
	for (uint32_t i = 0; i < info.stageCount; i++) {
		const auto& s = info.pStages[i];
		uint64_t    h = 0;
		if (s.pNext != nullptr || s.flags != 0 || s.pSpecializationInfo != nullptr ||
		    std::strcmp(s.pName, "main") != 0 || !SpirvOf(vk::ShaderModule(s.module), h)) {
			return 0;
		}
		wr.U(static_cast<uint32_t>(s.stage));
		wr.U64(h);
	}
	if (const auto* vi = info.pVertexInputState; vi != nullptr) {
		if (vi->pNext != nullptr || vi->flags != 0) return 0;
		wr.U(1);
		wr.U(vi->vertexBindingDescriptionCount);
		wr.Raw(vi->pVertexBindingDescriptions, vi->vertexBindingDescriptionCount * sizeof(VkVertexInputBindingDescription));
		wr.U(vi->vertexAttributeDescriptionCount);
		wr.Raw(vi->pVertexAttributeDescriptions,
		       vi->vertexAttributeDescriptionCount * sizeof(VkVertexInputAttributeDescription));
	} else {
		wr.U(0);
	}
	if (const auto* ia = info.pInputAssemblyState; ia != nullptr) {
		if (ia->pNext != nullptr) return 0;
		wr.U(1);
		wr.Used(*ia, offsetof(VkPipelineInputAssemblyStateCreateInfo, primitiveRestartEnable) + 4);
	} else {
		wr.U(0);
	}
	if (const auto* ts = info.pTessellationState; ts != nullptr) {
		if (ts->pNext != nullptr) return 0;
		wr.U(1);
		wr.Used(*ts, offsetof(VkPipelineTessellationStateCreateInfo, patchControlPoints) + 4);
	} else {
		wr.U(0);
	}
	if (const auto* vp = info.pViewportState; vp != nullptr) {
		if (vp->pViewports != nullptr || vp->pScissors != nullptr ||
		    !KnownChain(vp->pNext, {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT})) {
			return 0;
		}
		wr.U(1);
		wr.U(vp->flags);
		wr.U(vp->viewportCount);
		wr.U(vp->scissorCount);
		const auto* dcc = FindChain<VkPipelineViewportDepthClipControlCreateInfoEXT>(
		    vp->pNext, VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT);
		wr.U(dcc != nullptr ? 1 : 0);
		if (dcc != nullptr) wr.Used(*dcc, offsetof(VkPipelineViewportDepthClipControlCreateInfoEXT, negativeOneToOne) + 4);
	} else {
		wr.U(0);
	}
	if (const auto* rs = info.pRasterizationState; rs != nullptr) {
		if (!KnownChain(rs->pNext, {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT,
		                            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT})) {
			return 0;
		}
		wr.U(1);
		wr.Used(*rs, offsetof(VkPipelineRasterizationStateCreateInfo, lineWidth) + 4);
		const auto* clip = FindChain<VkPipelineRasterizationDepthClipStateCreateInfoEXT>(
		    rs->pNext, VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT);
		wr.U(clip != nullptr ? 1 : 0);
		if (clip != nullptr) wr.Used(*clip, offsetof(VkPipelineRasterizationDepthClipStateCreateInfoEXT, depthClipEnable) + 4);
		const auto* pv = FindChain<VkPipelineRasterizationProvokingVertexStateCreateInfoEXT>(
		    rs->pNext, VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT);
		wr.U(pv != nullptr ? 1 : 0);
		if (pv != nullptr) wr.Used(*pv, offsetof(VkPipelineRasterizationProvokingVertexStateCreateInfoEXT, provokingVertexMode) + 4);
	} else {
		wr.U(0);
	}
	if (const auto* ms = info.pMultisampleState; ms != nullptr) {
		if (ms->pNext != nullptr || ms->pSampleMask != nullptr) return 0;
		wr.U(1);
		wr.U(ms->flags);
		wr.U(static_cast<uint32_t>(ms->rasterizationSamples));
		wr.U(ms->sampleShadingEnable);
		uint32_t mss = 0;
		std::memcpy(&mss, &ms->minSampleShading, 4);
		wr.U(mss);
		wr.U(ms->alphaToCoverageEnable);
		wr.U(ms->alphaToOneEnable);
	} else {
		wr.U(0);
	}
	if (const auto* ds = info.pDepthStencilState; ds != nullptr) {
		if (ds->pNext != nullptr) return 0;
		wr.U(1);
		wr.Used(*ds, offsetof(VkPipelineDepthStencilStateCreateInfo, maxDepthBounds) + 4);
	} else {
		wr.U(0);
	}
	if (const auto* cb = info.pColorBlendState; cb != nullptr) {
		if (cb->attachmentCount > 8 ||
		    !KnownChain(cb->pNext, {VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT})) {
			return 0;
		}
		wr.U(1);
		wr.U(cb->flags);
		wr.U(cb->logicOpEnable);
		wr.U(static_cast<uint32_t>(cb->logicOp));
		wr.U(cb->attachmentCount);
		wr.Raw(cb->pAttachments, cb->attachmentCount * sizeof(VkPipelineColorBlendAttachmentState));
		wr.Raw(cb->blendConstants, sizeof(cb->blendConstants));
		const auto* cw = FindChain<VkPipelineColorWriteCreateInfoEXT>(
		    cb->pNext, VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT);
		wr.U(cw != nullptr ? 1 : 0);
		if (cw != nullptr) {
			if (cw->attachmentCount > 8) return 0;
			wr.U(cw->attachmentCount);
			wr.Raw(cw->pColorWriteEnables, cw->attachmentCount * sizeof(VkBool32));
		}
	} else {
		wr.U(0);
	}
	if (const auto* dyn = info.pDynamicState; dyn != nullptr) {
		if (dyn->pNext != nullptr || dyn->dynamicStateCount > 128) return 0;
		wr.U(1);
		wr.U(dyn->dynamicStateCount);
		wr.Raw(dyn->pDynamicStates, dyn->dynamicStateCount * sizeof(VkDynamicState));
	} else {
		wr.U(0);
	}
	if (const auto* ren = FindChain<VkPipelineRenderingCreateInfo>(info.pNext,
	                                                               VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
	    ren != nullptr) {
		if (ren->colorAttachmentCount > 8) return 0;
		wr.U(1);
		wr.U(ren->viewMask);
		wr.U(ren->colorAttachmentCount);
		wr.Raw(ren->pColorAttachmentFormats, ren->colorAttachmentCount * sizeof(VkFormat));
		wr.U(static_cast<uint32_t>(ren->depthAttachmentFormat));
		wr.U(static_cast<uint32_t>(ren->stencilAttachmentFormat));
	} else {
		wr.U(0);
	}
	WriteLayout(wr, bindings, layout_flags, push_stages);
	return RecordPipeline(KindGraphics, wr);
}

bool EnsureStarted(GraphicContext& graphics, const std::string& title_id, vk::PipelineCache driver_cache) {
	auto& ctx = Ctx();
	static const bool only = [] {
		const char* v = std::getenv("KYTY_PREWARM_ONLY");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	{
		std::lock_guard lock(ctx.mutex);
		if (ctx.started) {
			return false;
		}
		ctx.started     = true;
		ctx.start_time  = Clock::now();
		ctx.next_report = ctx.start_time + std::chrono::seconds(60);
		if (!Enabled() || title_id.empty()) {
			return false;
		}
		ctx.graphics     = &graphics;
		ctx.driver_cache = driver_cache;
		std::error_code ec;
		std::filesystem::create_directories("_PipelineCache", ec);
		ctx.path = std::filesystem::path("_PipelineCache") / (title_id + ".prewarm2");
		// First-ever run: start from a shipped seed (PrewarmSeeds/<title>.prewarm2 or <title>.prewarm2
		// beside the emulator) so the start-up warm-up builds the pipelines before gameplay.
		if (!std::filesystem::exists(ctx.path, ec) || std::filesystem::file_size(ctx.path, ec) <= 8u) {
			for (const auto& shipped: {std::filesystem::path("PrewarmSeeds") / (title_id + ".prewarm2"),
			                           std::filesystem::path(title_id + ".prewarm2")}) {
				if (std::filesystem::exists(shipped, ec) &&
				    std::filesystem::copy_file(shipped, ctx.path,
				                               std::filesystem::copy_options::overwrite_existing, ec)) {
					std::printf("Prewarm: first run, using the shipped seed %s\n", shipped.string().c_str());
					break;
				}
			}
		}

		uint64_t valid_end = 0;
		if (std::FILE* f = std::fopen(ctx.path.string().c_str(), "rb"); f != nullptr) {
			char magic[8] = {};
			if (std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, Magic, 8) == 0) {
				valid_end = 8;
				for (;;) {
					uint32_t kind = 0;
					uint32_t n    = 0;
					uint64_t key  = 0;
					if (std::fread(&kind, 4, 1, f) != 1 || std::fread(&n, 4, 1, f) != 1 ||
					    std::fread(&key, 8, 1, f) != 1 || n > MaxWords) {
						break;
					}
					const uint64_t payload_pos = valid_end + 16;
					if (kind == KindSpirv) {
						// verified when loaded; skip the payload here
						if (!Seek(f, payload_pos + uint64_t {n} * 4u)) {
							break;
						}
						// a torn tail is detected by the next header read failing / a short file
						ctx.spirv_offset[key] = payload_pos;
						ctx.spirv_words[key]  = n;
					} else if (kind == KindMeta || kind == KindGraphicsV1) {
						std::vector<uint32_t> payload(n);
						if (n != 0 && std::fread(payload.data(), 4, n, f) != n) {
							break;
						}
						if (kind == KindMeta && n == 2) {
							const uint64_t g = payload[0] | (uint64_t {payload[1]} << 32u);
							if (ctx.guest_of_spirv.emplace(key, g).second) ctx.spirv_of_guest[g].push_back(key);
						}
					} else if (kind == KindCompute || kind == KindGraphics) {
						auto rec  = std::make_shared<Pending>();
						rec->kind = kind;
						rec->key  = key;
						rec->desc.resize(n);
						if (std::fread(rec->desc.data(), 4, n, f) != n) {
							break;
						}
						if (Hash(rec->desc.data(), n * 4u, kind) == key && ctx.known.insert(key).second) {
							ctx.entries[key] = {};
							(kind == KindCompute ? ctx.queued_c : ctx.queued_g)++;
							ctx.seed_descs.push_back(std::make_shared<Pending>(*rec));
							ctx.queue.push_back(std::move(rec));
						}
					} else {
						break;
					}
					valid_end = payload_pos + uint64_t {n} * 4u;
				}
			}
			std::fclose(f);
		}
		for (const auto& [h, off]: ctx.spirv_offset) ctx.seed_spirv.insert(h);
		ctx.seed_loaded = valid_end != 0;
		const uint64_t file_size = std::filesystem::exists(ctx.path, ec) ? std::filesystem::file_size(ctx.path, ec) : 0;
		if (valid_end > file_size) {
			valid_end = 0; // torn SPIR-V blob at the end: start over is safest
		}
		if (valid_end == 0) {
			ctx.seed_spirv.clear();
			ctx.seed_descs.clear();
			ctx.guest_of_spirv.clear();
			ctx.spirv_of_guest.clear();
			ctx.spirv_offset.clear();
			ctx.spirv_words.clear();
			ctx.known.clear();
			ctx.entries.clear();
			ctx.queue.clear();
			ctx.queued_c = ctx.queued_g = 0;
			ctx.out = std::fopen(ctx.path.string().c_str(), "wb");
			if (ctx.out != nullptr) {
				std::fwrite(Magic, 1, 8, ctx.out);
				std::fflush(ctx.out);
				ctx.out_pos = 8;
			}
		} else {
			std::filesystem::resize_file(ctx.path, valid_end, ec); // drop a torn last record
			ctx.out     = std::fopen(ctx.path.string().c_str(), "ab");
			ctx.out_pos = valid_end;
		}
		if (ctx.out == nullptr) {
			std::printf("Prewarm: cannot write %s; recording off\n", ctx.path.string().c_str());
		}
		for (const auto& [h, guest, code]: ctx.early_spirv) {
			if (!ctx.spirv_offset.contains(h)) {
				WriteRecordLocked(ctx, KindSpirv, h, code);
			}
			if (guest != 0 && ctx.guest_of_spirv.emplace(h, guest).second) {
				const uint32_t g[2] = {static_cast<uint32_t>(guest), static_cast<uint32_t>(guest >> 32u)};
				WriteRecordLocked(ctx, KindMeta, h, g);
			}
		}
		ctx.early_spirv.clear();
		ctx.early_spirv.shrink_to_fit();
		const unsigned hw      = std::thread::hardware_concurrency();
		unsigned       threads = std::clamp(hw / 3u, 1u, 12u);
		if (const char* v = std::getenv("KYTY_PREWARM_THREADS"); v != nullptr && *v != '\0') {
			threads = std::clamp(static_cast<unsigned>(std::strtoul(v, nullptr, 10)), 1u, 64u);
		}
		threads = std::min<unsigned>(threads, static_cast<unsigned>(std::max<size_t>(1, ctx.queue.size())));
		std::printf("Prewarm: %s: %zu recorded pipelines (%zu compute, %zu graphics, %zu SPIR-V modules), "
		            "building them on %u threads\n",
		            ctx.path.string().c_str(), ctx.queue.size(), ctx.queued_c, ctx.queued_g,
		            ctx.spirv_offset.size(), ctx.queue.empty() ? 0u : threads);
		std::fflush(stdout);
		if (!ctx.queue.empty()) {
			OpenInflight(ctx.path);
			for (unsigned i = 0; i < threads; i++) {
				ctx.threads.emplace_back(Worker);
			}
		}
	}
	// KYTY_PREWARM_WAIT: default on when the seed has pipelines (a compile pause at the first
	// pipeline with progress in the log, instead of minutes of skipped dispatches); 0 = off.
	const bool wait = [&] {
		const char* v = std::getenv("KYTY_PREWARM_WAIT");
		if (v != nullptr && *v != '\0') return std::strcmp(v, "0") != 0;
		std::lock_guard lock(ctx.mutex);
		return !ctx.queue.empty();
	}();
	if (wait || only) {
		std::printf("Prewarm: waiting for the replay (%s)\n", only ? "KYTY_PREWARM_ONLY=1" : "KYTY_PREWARM_WAIT, default with a seed; 0 = play while it builds");
		std::fflush(stdout);
		std::unique_lock lock(ctx.mutex);
		const auto total      = ctx.queue.size();
		const auto wait_start = Clock::now();
		// The warm-up never waits on a few stragglers: once at most 1% (min 8) of the pipelines are
		// left and nothing finished for 3 s, the game starts and they finish in the background
		// (Take() waits only if gameplay needs one of them before it is done).
		// Yotei: one 448K-word compute pipeline kept the warm-up at 2344/2345 for minutes.
		size_t last_done     = 0;
		auto   last_progress = Clock::now();
		const size_t straggler_limit = std::max<size_t>(8, total / 100);
		while (!ctx.cv.wait_for(lock, std::chrono::seconds(1),
		                        [&] { return ctx.built + ctx.failed >= ctx.queue.size(); })) {
			const auto done = ctx.built + ctx.failed;
			if (done != last_done) {
				last_done     = done;
				last_progress = Clock::now();
			} else if (!only && total - done <= straggler_limit &&
			           Clock::now() - last_progress >= std::chrono::seconds(3)) {
				std::printf("Prewarm: warm-up leaves %zu slow pipelines to the background and starts the game\n",
				            total - done);
				for (const auto& [key, entry]: ctx.entries) {
					if (entry.state == State::Building) {
						std::printf("Prewarm:   still building key=%016llx\n", static_cast<unsigned long long>(key));
					}
				}
				std::fflush(stdout);
				break;
			}
			std::printf("Prewarm: warm-up %zu/%zu (%.0f%%), %.0f s\n", static_cast<size_t>(done), total,
			            total != 0 ? 100.0 * static_cast<double>(done) / static_cast<double>(total) : 100.0,
			            std::chrono::duration<double>(Clock::now() - wait_start).count());
			std::fflush(stdout);
		}
		std::printf("Prewarm: warm-up done, %zu pipelines in %.1f s\n", total,
		            std::chrono::duration<double>(Clock::now() - wait_start).count());
		std::fflush(stdout);
	}
	return only;
}

vk::Pipeline Take(uint64_t key, bool graphics) {
	if (key == 0) {
		return nullptr;
	}
	auto&        ctx    = Ctx();
	vk::Pipeline result = nullptr;
	{
		std::unique_lock lock(ctx.mutex);
		auto             it = ctx.entries.find(key);
		if (it != ctx.entries.end()) {
			if (it->second.state == State::Building) {
				ctx.cv.wait(lock, [&] { return it->second.state != State::Building; });
			}
			if (it->second.state == State::Ready) {
				result = it->second.pipeline;
			} else if (it->second.state == State::Done) {
				ctx.taken_twice[graphics ? 1 : 0]++;
			}
			it->second.state    = State::Done;
			it->second.pipeline = nullptr;
		}
		(result != nullptr ? ctx.served : ctx.missed)[graphics ? 1 : 0]++;
	}
	MaybeReport();
	return result;
}

thread_local bool t_async_thread = false;

void MarkAsyncThread() {
	t_async_thread = true;
}

void NoteRuntimeCompile(bool graphics, uint64_t ns) {
	const bool async = t_async_thread;
	const int  i     = graphics ? 1 : 0;
	g_runtime_compiles[i].fetch_add(1, std::memory_order_relaxed);
	g_runtime_ns[i].fetch_add(ns, std::memory_order_relaxed);
	if (!async) g_runtime_sync[i].fetch_add(1, std::memory_order_relaxed);
	if (ns > 50'000'000u) g_runtime_slow[i].fetch_add(1, std::memory_order_relaxed);
	AtomicMax(g_runtime_max_ns, ns);
	MaybeReport();
}

void NoteTranslation(uint64_t ns) {
	g_translations.fetch_add(1, std::memory_order_relaxed);
	g_translation_ns.fetch_add(ns, std::memory_order_relaxed);
	AtomicMax(g_translation_max_ns, ns);
}

void Shutdown() {
	auto& ctx = Ctx();
	{
		std::lock_guard lock(ctx.mutex);
		if (!ctx.started) {
			return;
		}
		ctx.stop = true;
		ReportLocked(ctx, "exit");
	}
	for (auto& t: ctx.threads) {
		t.join();
	}
	std::lock_guard lock(ctx.mutex);
	ctx.threads.clear();
	if (ctx.out != nullptr) {
		std::fclose(ctx.out);
		ctx.out = nullptr;
	}
}

} // namespace Libs::Graphics::Prewarm
