#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/moduleStore.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineBinaries.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCacheBlobs.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"
#include "device-fault.h"
#include "local-platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <type_traits>
#include <zstd.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

#include "shader-warmup-cache.h"

namespace Libs::Graphics {

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		static bool logged_113 = false;
		if (!logged_113) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
			logged_113 = true;
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static bool logged_121 = false;
		if (!logged_121) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
			logged_121 = true;
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u\n", raw_format);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

// native_bindings: the native XPR variant (src/local/native-xpr.inc). The same
// SPIR-V, but one ordinary descriptor set per object in which the flattened SRT
// and shader data are dynamic storage buffers, so each draw only supplies offsets.
static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits stage, bool native_bindings) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	for (const auto& binding: program.bindings.descriptors) {
		auto type = NativeDescriptorType(binding.kind);
		if (native_bindings && (binding.kind == Kind::FlattenedSrt || binding.kind == Kind::ShaderData)) {
			type = vk::DescriptorType::eStorageBufferDynamic;
		}
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind), type,
		     NativeDescriptorCount(binding), stage, nullptr});
	}
}

// The precompile's capture (PipelineBinaryWriter): a pipeline the static pipeline cache holds (the first
// store is made from it), its binaries then from the driver's own cache (a pipeline created with the
// capture flag is never a cache hit: compiled); else compiled with its binaries kept.
template <typename Info, typename Create>
static vk::Result CapturePipelineBinaries(GraphicContext& graphics, PipelineCache::Pipeline& pipeline, Info info,
                                          vk::PipelineCache driver_cache, const PipelineBinaries::Key& key,
                                          Create create) {
	auto* writer = graphics.pipeline_binary_writer;
	if (graphics.static_pipeline_cache != nullptr && graphics.pipeline_cache_control_enabled) {
		auto lookup = info;
		lookup.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
		if (create(graphics.static_pipeline_cache, lookup) == vk::Result::eSuccess) {
			if (writer->Capture(key, nullptr, &info)) return vk::Result::eSuccess;
			graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
		}
		pipeline.pipeline = nullptr;
	}
	vk::PipelineCreateFlags2CreateInfoKHR flags {};
	flags.flags = vk::PipelineCreateFlags2KHR(static_cast<VkPipelineCreateFlags>(info.flags)) |
	              vk::PipelineCreateFlagBits2KHR::eCaptureDataKHR;
	flags.pNext       = info.pNext;
	info.pNext        = &flags;
	const auto result = create(driver_cache, info);
	if (result == vk::Result::eSuccess && !writer->Capture(key, pipeline.pipeline)) {
		std::printf("Pipeline binaries: the driver gave none for a pipeline\n");
	}
	return result;
}

namespace {
struct ZstdThread {
	ZSTD_CCtx*        compress   = nullptr;
	ZSTD_DCtx*        decompress = nullptr;
	std::vector<char> packed;
	~ZstdThread() {
		ZSTD_freeCCtx(compress);
		ZSTD_freeDCtx(decompress);
	}
};
thread_local ZstdThread t_zstd;
} // namespace

std::span<const char> ZstdPack(const void* data, size_t bytes, int level) {
	auto& zstd = t_zstd;
	if (zstd.compress == nullptr) zstd.compress = ZSTD_createCCtx();
	const auto bound = ZSTD_compressBound(bytes);
	if (zstd.packed.size() < bound) zstd.packed.resize(bound);
	const auto size = zstd.compress != nullptr
	                      ? ZSTD_compressCCtx(zstd.compress, zstd.packed.data(), bound, data, bytes, level)
	                      : ZSTD_compress(zstd.packed.data(), bound, data, bytes, level);
	if (ZSTD_isError(size)) return {};
	return {zstd.packed.data(), size};
}

bool ZstdUnpack(std::span<const char> packed, void* out, size_t bytes) {
	auto& zstd = t_zstd;
	if (zstd.decompress == nullptr) zstd.decompress = ZSTD_createDCtx();
	const auto size = zstd.decompress != nullptr
	                      ? ZSTD_decompressDCtx(zstd.decompress, out, bytes, packed.data(), packed.size())
	                      : ZSTD_decompress(out, bytes, packed.data(), packed.size());
	return size == bytes;
}

namespace ModuleStore {
namespace {
// A stored module's handle: the address of its index's byte here (untouched: no memory).
constexpr size_t HandleSpace = size_t {1} << 22u;
alignas(64) uint8_t g_handles[HandleSpace];

struct Entry {
	uint64_t file       = 0; // the store's (g_file) or another's (AddFrom)
	uint64_t offset     = 0;
	uint32_t stored     = 0; // bytes in the file
	uint32_t words      = 0;
	bool     compressed = false;
};
std::atomic<bool>     g_enabled {false};
std::once_flag        g_file_once;
uint64_t              g_file = 0;
std::atomic<uint64_t> g_end {0};
std::shared_mutex     g_mutex;
std::vector<Entry>    g_entries;
std::atomic<uint64_t> g_live {0}, g_spirv_bytes {0}, g_stored_bytes {0}, g_loads {0}, g_load_misses {0};
// The last modules loaded: a module's pipelines are mostly made one after another.
struct Recent {
	size_t                                       index = SIZE_MAX;
	std::shared_ptr<const std::vector<uint32_t>> spirv;
};
std::mutex               g_recent_mutex;
std::array<Recent, 16>   g_recent;
size_t                   g_recent_next = 0;

size_t IndexOf(vk::ShaderModule module) {
	return reinterpret_cast<const uint8_t*>(static_cast<VkShaderModule>(module)) - g_handles;
}

vk::ShaderModule Push(const Entry& entry) {
	size_t index = 0;
	{
		std::unique_lock lock(g_mutex);
		index = g_entries.size();
		if (index >= HandleSpace) return nullptr;
		g_entries.push_back(entry);
	}
	g_live.fetch_add(1, std::memory_order_relaxed);
	g_spirv_bytes.fetch_add(uint64_t {entry.words} * sizeof(uint32_t), std::memory_order_relaxed);
	return vk::ShaderModule(reinterpret_cast<VkShaderModule>(&g_handles[index]));
}
} // namespace

void Enable(bool maintenance5) {
	const char* setting = std::getenv("KYTY_MODULE_STORE");
	g_enabled           = maintenance5 && (setting == nullptr || std::string_view(setting) != "0");
}

bool Enabled() {
	return g_enabled.load(std::memory_order_relaxed);
}

vk::ShaderModule Add(const std::vector<uint32_t>& spirv) {
	if (!Enabled() || spirv.empty()) return nullptr;
	std::call_once(g_file_once, [] { g_file = LocalPlatform::OpenScratchFile(); });
	if (g_file == 0) return nullptr;
	const auto  bytes      = spirv.size() * sizeof(uint32_t);
	const auto  packed     = ZstdPack(spirv.data(), bytes);
	const bool  compressed = !packed.empty() && packed.size() < bytes;
	const void* data       = compressed ? static_cast<const void*>(packed.data()) : spirv.data();
	const auto  stored     = compressed ? packed.size() : bytes;
	const auto        offset      = g_end.fetch_add(stored, std::memory_order_relaxed);
	if (stored > UINT32_MAX || !LocalPlatform::WriteScratchFile(g_file, offset, data, stored)) return nullptr;
	const auto module = Push({g_file, offset, static_cast<uint32_t>(stored), static_cast<uint32_t>(spirv.size()), compressed});
	if (module) g_stored_bytes.fetch_add(stored, std::memory_order_relaxed);
	return module;
}

vk::ShaderModule AddFrom(uint64_t file, uint64_t offset, uint32_t stored, uint32_t words, bool compressed) {
	if (!Enabled() || file == 0 || words == 0) return nullptr;
	return Push({file, offset, stored, words, compressed});
}

bool Holds(vk::ShaderModule module) {
	const auto* address = reinterpret_cast<const uint8_t*>(static_cast<VkShaderModule>(module));
	return address >= g_handles && address < g_handles + HandleSpace;
}

std::shared_ptr<const std::vector<uint32_t>> Load(vk::ShaderModule module) {
	if (!Holds(module)) return nullptr;
	const auto index = IndexOf(module);
	g_loads.fetch_add(1, std::memory_order_relaxed);
	{
		std::lock_guard lock(g_recent_mutex);
		for (const auto& recent: g_recent)
			if (recent.index == index) return recent.spirv;
	}
	g_load_misses.fetch_add(1, std::memory_order_relaxed);
	Entry entry;
	{
		std::shared_lock lock(g_mutex);
		if (index >= g_entries.size()) return nullptr;
		entry = g_entries[index];
	}
	auto       spirv = std::make_shared<std::vector<uint32_t>>(entry.words);
	const auto bytes = spirv->size() * sizeof(uint32_t);
	if (entry.compressed) {
		std::vector<char> packed(entry.stored);
		if (!LocalPlatform::ReadScratchFile(entry.file, entry.offset, packed.data(), packed.size()) ||
		    !ZstdUnpack(packed, spirv->data(), bytes))
			return nullptr;
	} else if (!LocalPlatform::ReadScratchFile(entry.file, entry.offset, spirv->data(), bytes)) {
		return nullptr;
	}
	std::lock_guard lock(g_recent_mutex);
	g_recent[g_recent_next++ % g_recent.size()] = {index, spirv};
	return spirv;
}

void Release(vk::ShaderModule module) {
	if (!Holds(module)) return;
	g_live.fetch_sub(1, std::memory_order_relaxed);
}

Statistics Stats() {
	return {g_live.load(), g_spirv_bytes.load(), g_stored_bytes.load(), g_loads.load(), g_load_misses.load()};
}
} // namespace ModuleStore

void DestroyShaderModule(vk::Device device, vk::ShaderModule module) {
	if (ModuleStore::Holds(module)) {
		ModuleStore::Release(module);
	} else if (module) {
		device.destroyShaderModule(module, nullptr);
	}
}

// Stored modules (ModuleStore) of a pipeline's stages as their SPIR-V inline: the stages and what they point to.
struct InlineStages {
	std::array<vk::PipelineShaderStageCreateInfo, 8>                   stages {};
	std::array<vk::ShaderModuleCreateInfo, 8>                          modules {};
	std::array<std::shared_ptr<const std::vector<uint32_t>>, 8>         spirv {};
	// The stage made inline, or false when it holds no stored module.
	bool Take(vk::PipelineShaderStageCreateInfo& stage, size_t i) {
		if (!ModuleStore::Holds(stage.module)) return false;
		spirv[i] = ModuleStore::Load(stage.module);
		EXIT_IF(spirv[i] == nullptr);
		modules[i].codeSize = spirv[i]->size() * sizeof(uint32_t);
		modules[i].pCode    = spirv[i]->data();
		modules[i].pNext    = stage.pNext;
		stage.pNext         = &modules[i];
		stage.module        = nullptr;
		return true;
	}
	void Apply(vk::GraphicsPipelineCreateInfo& info) {
		if (!ModuleStore::Enabled() || info.stageCount > stages.size()) return;
		bool any = false;
		for (uint32_t i = 0; i < info.stageCount; i++) {
			stages[i] = info.pStages[i];
			any |= Take(stages[i], i);
		}
		if (any) info.pStages = stages.data();
	}
	void Apply(vk::ComputePipelineCreateInfo& info) {
		if (ModuleStore::Enabled()) (void)Take(info.stage, 0);
	}
};

bool PipelineKeyLog() {
	static const bool on = std::getenv("KYTY_PIPELINE_KEY_LOG") != nullptr;
	return on;
}

namespace {
std::mutex                                                          g_key_log_mutex;
std::unordered_map<VkShaderModule, std::pair<uint64_t, uint64_t>> g_key_log_modules;
std::unordered_map<VkPipelineLayout, uint64_t>                      g_key_log_layouts;

// (KYTY_PIPELINE_KEY_LOG) What a pipeline layout was made of.
void NotePipelineKeyLayout(vk::PipelineLayout layout, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                           bool push_descriptors) {
	if (!PipelineKeyLog()) return;
	std::vector<uint32_t> words {push_descriptors ? 1u : 0u, ShaderRecompiler::IR::NativePushConstantSize};
	for (const auto& binding: bindings) {
		words.insert(words.end(), {binding.binding, static_cast<uint32_t>(binding.descriptorType), binding.descriptorCount,
		                           static_cast<uint32_t>(binding.stageFlags)});
	}
	std::lock_guard lock(g_key_log_mutex);
	g_key_log_layouts[static_cast<VkPipelineLayout>(layout)] = XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
}

std::vector<vk::ShaderModule> ModulesOf(const vk::GraphicsPipelineCreateInfo& info) {
	std::vector<vk::ShaderModule> modules;
	for (uint32_t i = 0; i < info.stageCount; i++) modules.push_back(info.pStages[i].module);
	return modules;
}
std::vector<vk::ShaderModule> ModulesOf(const vk::ComputePipelineCreateInfo& info) { return {info.stage.module}; }

template <typename Info>
void LogPipelineKey(const char* what, const PipelineBinaries::Key& key, const Info& info) {
	std::lock_guard lock(g_key_log_mutex);
	const auto      layout = g_key_log_layouts.find(static_cast<VkPipelineLayout>(info.layout));
	std::string     line   = fmt::format("PIPEKEY {} key={:016x} layout={:016x}", what,
	                                     XXH3_64bits(key.bytes.data(), key.size),
	                                     layout != g_key_log_layouts.end() ? layout->second : 0);
	for (const auto module: ModulesOf(info)) {
		const auto found = g_key_log_modules.find(static_cast<VkShaderModule>(module));
		const auto [shader, spirv] =
		    found != g_key_log_modules.end() ? found->second : std::pair<uint64_t, uint64_t> {0, 0};
		line += fmt::format(" {:016x}/{:016x}", shader, spirv);
	}
	// What else the key is made of: the create flags, the pNext chain, each stage's flags, pNext and specialization.
	const auto chain = [](const void* next) {
		std::string text;
		for (const auto* s = static_cast<const VkBaseInStructure*>(next); s != nullptr; s = s->pNext)
			text += fmt::format("{}{}", text.empty() ? "" : ",", static_cast<int>(s->sType));
		return text;
	};
	line += fmt::format(" | flags={:x} next=[{}]", static_cast<uint32_t>(info.flags), chain(info.pNext));
	const auto stage_text = [&](const vk::PipelineShaderStageCreateInfo& stage) {
		std::string text = fmt::format(" stage={:x}/{:x}/{} next=[{}]", static_cast<uint32_t>(stage.stage),
		                               static_cast<uint32_t>(stage.flags), stage.pName != nullptr ? stage.pName : "",
		                               chain(stage.pNext));
		if (const auto* spec = stage.pSpecializationInfo; spec != nullptr) {
			text += fmt::format(" spec={}:{}:{:016x}", spec->mapEntryCount, spec->dataSize,
			                    spec->pData != nullptr ? XXH3_64bits(spec->pData, spec->dataSize) : 0);
		}
		if (const auto* sub = static_cast<const VkBaseInStructure*>(stage.pNext); sub != nullptr &&
		    sub->sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO) {
			text += fmt::format(" subgroup={}",
			                    reinterpret_cast<const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*>(sub)
			                        ->requiredSubgroupSize);
		}
		return text;
	};
	if constexpr (std::is_same_v<Info, vk::ComputePipelineCreateInfo>) {
		line += stage_text(info.stage);
	} else {
		for (uint32_t i = 0; i < info.stageCount; i++) line += stage_text(info.pStages[i]);
	}
	std::printf("%s\n", line.c_str());
}
} // namespace

// (KYTY_PIPELINE_SOURCE_LOG) Each module's SPIR-V size.
static std::unordered_map<VkShaderModule, size_t> g_module_bytes;

bool InlineKeyCheck() {
	static const bool on = std::getenv("KYTY_INLINE_KEY_CHECK") != nullptr;
	return on;
}
static std::mutex                                                 g_inline_mutex;
static std::unordered_map<VkShaderModule, std::vector<uint32_t>> g_inline_spirv;
void NoteInlineSpirv(vk::ShaderModule module, const std::vector<uint32_t>& spirv) {
	std::lock_guard lock(g_inline_mutex);
	g_inline_spirv[static_cast<VkShaderModule>(module)] = spirv;
}

// (KYTY_INLINE_KEY_CHECK) The key of the pipeline with its stages' SPIR-V inline against `key` (its modules').
template <typename Info>
static void CheckInlineKey(vk::Device device, const Info& info, const PipelineBinaries::Key& key) {
	static std::atomic<uint64_t> same {0}, differ {0}, missing {0};
	std::vector<vk::PipelineShaderStageCreateInfo> stages;
	if constexpr (std::is_same_v<Info, vk::GraphicsPipelineCreateInfo>) {
		stages.assign(info.pStages, info.pStages + info.stageCount);
	} else {
		stages.push_back(info.stage);
	}
	std::vector<vk::ShaderModuleCreateInfo> modules(stages.size());
	{
		std::lock_guard lock(g_inline_mutex);
		for (size_t i = 0; i < stages.size(); i++) {
			const auto found = g_inline_spirv.find(static_cast<VkShaderModule>(stages[i].module));
			if (found == g_inline_spirv.end()) {
				missing++;
				return;
			}
			modules[i].codeSize = found->second.size() * sizeof(uint32_t);
			modules[i].pCode    = found->second.data();
			modules[i].pNext    = stages[i].pNext;
			stages[i].pNext     = &modules[i];
			stages[i].module    = nullptr;
		}
	}
	auto inline_info = info;
	if constexpr (std::is_same_v<Info, vk::GraphicsPipelineCreateInfo>) {
		inline_info.pStages = stages.data();
	} else {
		inline_info.stage = stages[0];
	}
	PipelineBinaries::Key inline_key;
	const bool            ok    = PipelineBinaries::PipelineKey(device, &inline_info, inline_key);
	const bool            equal = ok && inline_key.size == key.size && std::memcmp(inline_key.bytes.data(), key.bytes.data(), key.size) == 0;
	const auto            count = (equal ? same : differ).fetch_add(1) + 1;
	if (!equal && count <= 5)
		std::printf("Inline key check: DIFFERENT key (%s, %u stages)\n", ok ? "made" : "not made",
		            static_cast<unsigned>(stages.size()));
	if ((same + differ) % 100 == 0)
		std::printf("Inline key check: %llu same, %llu different, %llu without SPIR-V\n",
		            static_cast<unsigned long long>(same.load()), static_cast<unsigned long long>(differ.load()),
		            static_cast<unsigned long long>(missing.load()));
}

void NotePipelineKeyModule(vk::ShaderModule module, uint64_t shader_hash, uint64_t spirv_hash, size_t spirv_bytes) {
	std::lock_guard lock(g_key_log_mutex);
	g_key_log_modules[static_cast<VkShaderModule>(module)] = {shader_hash, spirv_hash};
	if (spirv_bytes != 0) g_module_bytes[static_cast<VkShaderModule>(module)] = spirv_bytes;
}

static size_t ModuleSpirvBytes(vk::ShaderModule module) {
	if (module == nullptr) return 0;
	std::lock_guard lock(g_key_log_mutex);
	const auto      found = g_module_bytes.find(static_cast<VkShaderModule>(module));
	return found != g_module_bytes.end() ? found->second : 0;
}

// One descriptor set layout and pipeline layout per binding set: every pipeline with the same bindings (one shader's
// specializations and states have the same) shares them. Made when first needed - each pair cost the driver 0.2-0.5 ms
// (a laptop's walk: median 0.19 ms, p90 0.5 ms, up to 3.8 ms, for every pipeline it made) - and destroyed with the
// pipeline cache (DestroySharedLayouts).
namespace {
struct SharedLayout {
	std::vector<uint32_t>   words; // what the layouts are made of (SharedLayouts)
	vk::DescriptorSetLayout set    = nullptr;
	vk::PipelineLayout      layout = nullptr;
};
std::mutex                                                        g_shared_layouts_mutex;
std::unordered_multimap<uint64_t, std::unique_ptr<SharedLayout>> g_shared_layouts;
} // namespace

static void SharedLayouts(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                          std::span<const vk::DescriptorSetLayoutBinding> bindings, bool native_bindings,
                          vk::ShaderStageFlags push_stages) {
	uint32_t descriptor_count = 0;
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
	}
	pipeline.native_bindings       = native_bindings;
	pipeline.uses_push_descriptors = !native_bindings && descriptor_count <= graphics.max_push_descriptors;
	std::vector<uint32_t> words {pipeline.uses_push_descriptors ? 1u : 0u, static_cast<uint32_t>(push_stages)};
	words.reserve(words.size() + bindings.size() * 4u);
	for (const auto& binding: bindings) {
		words.insert(words.end(), {binding.binding, static_cast<uint32_t>(binding.descriptorType), binding.descriptorCount,
		                           static_cast<uint32_t>(binding.stageFlags)});
	}
	const auto      key = XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
	std::lock_guard lock(g_shared_layouts_mutex);
	for (auto [found, end] = g_shared_layouts.equal_range(key); found != end; ++found) {
		if (found->second->words != words) continue;
		pipeline.descriptor_set_layout = found->second->set;
		pipeline.pipeline_layout       = found->second->layout;
		return;
	}
	auto shared   = std::make_unique<SharedLayout>();
	shared->words = std::move(words);
	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = pipeline.uses_push_descriptors ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                                                     : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &shared->set) != vk::Result::eSuccess);
	const vk::PushConstantRange  push_constants {push_stages, 0, ShaderRecompiler::IR::NativePushConstantSize};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &shared->set;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_constants;
	EXIT_NOT_IMPLEMENTED(graphics.device.createPipelineLayout(&layout_info, nullptr, &shared->layout) != vk::Result::eSuccess);
	NotePipelineKeyLayout(shared->layout, bindings, pipeline.uses_push_descriptors);
	pipeline.descriptor_set_layout = shared->set;
	pipeline.pipeline_layout       = shared->layout;
	g_shared_layouts.emplace(key, std::move(shared));
}

void DestroySharedLayouts(vk::Device device) {
	std::lock_guard lock(g_shared_layouts_mutex);
	for (const auto& [key, shared]: g_shared_layouts) {
		(void)key;
		device.destroyPipelineLayout(shared->layout, nullptr);
		device.destroyDescriptorSetLayout(shared->set, nullptr);
	}
	g_shared_layouts.clear();
}

bool PipelineSourceLog() {
	static const bool on = std::getenv("KYTY_PIPELINE_SOURCE_LOG") != nullptr;
	return on;
}

static std::atomic<bool> g_module_hashes {false};

void RecordModuleSpirvHashes() {
	g_module_hashes.store(true, std::memory_order_relaxed);
}

bool ModuleSpirvHashesRecorded() {
	return g_module_hashes.load(std::memory_order_relaxed) || PipelineKeyLog() || PipelineSourceLog();
}

// (Noted for the pipeline key and source logs only: 0 otherwise.)
static uint64_t ModuleSpirvHash(vk::ShaderModule module) {
	if (module == nullptr) return 0;
	std::lock_guard lock(g_key_log_mutex);
	const auto      found = g_key_log_modules.find(static_cast<VkShaderModule>(module));
	return found != g_key_log_modules.end() ? found->second.second : 0;
}

uint64_t GraphicsPipelineIdentity(uint64_t vertex_spirv, uint64_t pixel_spirv, const PipelineRenderingState& rendering,
                                  const PipelineVertexInputState& vertex_input, const PipelineStaticParameters& state,
                                  const ShaderVertexInputInfo& vs_input_info, const ShaderPixelInputInfo* ps_input_info,
                                  bool native_bindings) {
	// The fixed-function state as a recipe of a seed file or warmup recording holds it (its words after the shaders).
	LocalShaderWarmup::PipelineRecord record;
	record.vertex       = 0;
	record.pixel        = 0;
	record.rendering    = rendering;
	record.vertex_input = vertex_input;
	record.state        = state;
	LocalShaderWarmup::Writer writer;
	(void)LocalShaderWarmup::VisitPipeline(writer, record);
	std::vector<uint32_t> vs_key, ps_key;
	BuildStageStaticKey(vs_input_info, vs_key);
	if (ps_input_info != nullptr) BuildStageStaticKey(*ps_input_info, ps_key);
	const auto     hash    = [](std::span<const uint32_t> words) { return XXH3_64bits(words.data(), words.size_bytes()); };
	const uint64_t parts[] = {vertex_spirv,
	                          pixel_spirv,
	                          native_bindings ? 1u : 0u,
	                          writer.words.size() > 3 ? hash(std::span(writer.words).subspan(3)) : 0,
	                          hash(vs_key),
	                          ps_input_info != nullptr ? hash(ps_key) : 0};
	return XXH3_64bits(parts, sizeof(parts));
}

uint64_t ComputePipelineIdentity(uint64_t spirv, const ShaderComputeInputInfo& input_info, bool native_bindings) {
	std::vector<uint32_t> key;
	BuildStageStaticKey(input_info, key);
	const uint64_t parts[] = {spirv, native_bindings ? 1u : 0u, XXH3_64bits(key.data(), key.size() * sizeof(uint32_t))};
	return XXH3_64bits(parts, sizeof(parts));
}

// What the game's pipelines got from a static store (ReportStaticStore): made of what it holds, not among them, held
// but their binaries or cache not made again from the data, refused by the driver when made of them (the last result),
// and pipelines it could not be asked for (no driver key; no identity).
struct StoreOutcomes {
	std::atomic<uint64_t> hits {0}, absent {0}, damaged {0}, refused {0}, unasked {0};
	std::atomic<int32_t>  refusal {0};

	void Count(std::atomic<uint64_t>& counter) { counter.fetch_add(1, std::memory_order_relaxed); }
	void Refused(vk::Result result) {
		Count(refused);
		refusal.store(static_cast<int32_t>(result), std::memory_order_relaxed);
	}
	void Report(const char* store, const char* data, const char* unasked_what) const {
		const auto counts = std::array {hits.load(), absent.load(), damaged.load(), refused.load(), unasked.load()};
		if (counts[0] + counts[1] + counts[2] + counts[3] + counts[4] == 0) return;
		const auto last = counts[3] != 0 ? vk::to_string(static_cast<vk::Result>(refusal.load())) : std::string("none");
		std::printf("Static pipeline %s: %llu pipelines made of them, %llu not among them, %llu whose %s the driver did "
		            "not make again, %llu it refused to make of them (last: %s), %llu %s\n",
		            store, static_cast<unsigned long long>(counts[0]), static_cast<unsigned long long>(counts[1]),
		            static_cast<unsigned long long>(counts[2]), data, static_cast<unsigned long long>(counts[3]),
		            last.c_str(), static_cast<unsigned long long>(counts[4]), unasked_what);
	}
};
static StoreOutcomes g_binary_outcomes, g_blob_outcomes;

void ReportStaticStore() {
	g_binary_outcomes.Report("binaries", "binaries", "without a driver key");
	g_blob_outcomes.Report("blobs", "pipeline cache", "without an identity");
	std::fflush(stdout);
}

// (KYTY_PIPELINE_SOURCE_LOG) Whether the static cache or store gave this thread's last pipeline, and for a pipeline
// cache blob: its size and the microseconds of its read, of the cache made of it and of the pipeline's creation.
static thread_local bool t_static_hit = false;

bool LastPipelineFromStore() {
	return t_static_hit;
}
struct BlobTimes {
	size_t bytes     = 0;
	double read_us   = 0.0;
	double cache_us  = 0.0;
	double create_us = 0.0;
};
static thread_local BlobTimes t_blob_times;
// (KYTY_PIPELINE_SOURCE_LOG) The microseconds of the last pipeline's descriptor set and pipeline layouts.
static thread_local double t_layout_us = 0.0;

static double MicrosecondsSince(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

// The precompile's pipeline another of its shards holds: none is made (the caller destroys a null one).
static bool ClaimedElsewhere(const GraphicContext& graphics, vk::Result result) {
	return result == vk::Result::ePipelineCompileRequired && graphics.pipeline_binary_writer != nullptr;
}

// A pipeline cache made of one pipeline's cache data (pipelineCacheBlobs.h), or none.
static vk::PipelineCache BlobCache(GraphicContext& graphics, const std::vector<uint8_t>& data) {
	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize  = data.size();
	create.pInitialData     = data.data();
	vk::PipelineCache cache = nullptr;
	return graphics.device.createPipelineCache(&create, nullptr, &cache) == vk::Result::eSuccess ? cache : nullptr;
}

// KYTY_PIPELINE_STATS=1 (a survey): each pipeline made once more with the driver's statistics
// (VK_KHR_pipeline_executable_properties), a "PIPESTATS kind hash hash executable: name=value; ..." line per executable,
// and destroyed. A pipeline's local memory (its spills) is per thread: the driver keeps one pool for the largest of
// them times the threads the GPU runs, video memory it never gives back (8 GB cards: ~1 GB more with 1-1 dragon fire's
// table forms before 966c891d; destroying 13,000 idle pipelines freed none).
template <typename Info, typename Create>
static void LogPipelineStatistics(GraphicContext& graphics, Info info, Create create, const char* kind, uint64_t a,
                                  uint64_t b) {
	if (!graphics.pipeline_statistics_enabled) return;
	InlineStages inlined;
	inlined.Apply(info);
	info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
	vk::Pipeline made = nullptr;
	if (create(info, made) != vk::Result::eSuccess || made == nullptr) return;
	vk::PipelineInfoKHR pipeline_info {};
	pipeline_info.pipeline = made;
	uint32_t executables   = 0;
	(void)graphics.device.getPipelineExecutablePropertiesKHR(&pipeline_info, &executables, nullptr);
	std::vector<vk::PipelineExecutablePropertiesKHR> properties(executables);
	(void)graphics.device.getPipelineExecutablePropertiesKHR(&pipeline_info, &executables, properties.data());
	for (uint32_t e = 0; e < executables; e++) {
		vk::PipelineExecutableInfoKHR executable {};
		executable.pipeline        = made;
		executable.executableIndex = e;
		uint32_t count             = 0;
		(void)graphics.device.getPipelineExecutableStatisticsKHR(&executable, &count, nullptr);
		std::vector<vk::PipelineExecutableStatisticKHR> statistics(count);
		(void)graphics.device.getPipelineExecutableStatisticsKHR(&executable, &count, statistics.data());
		std::string line = fmt::format("PIPESTATS {} {:016x} {:016x} {}:", kind, a, b, properties[e].name.data());
		for (const auto& statistic: statistics) {
			switch (statistic.format) {
				case vk::PipelineExecutableStatisticFormatKHR::eBool32:
					line += fmt::format(" {}={};", statistic.name.data(), statistic.value.b32);
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eInt64:
					line += fmt::format(" {}={};", statistic.name.data(), statistic.value.i64);
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eUint64:
					line += fmt::format(" {}={};", statistic.name.data(), statistic.value.u64);
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
					line += fmt::format(" {}={:.1f};", statistic.name.data(), statistic.value.f64);
					break;
			}
		}
		std::printf("%s\n", line.c_str());
	}
	std::fflush(stdout);
	graphics.device.destroyPipeline(made, nullptr);
}

// The static precompile's pipeline when it holds this one (no compilation), else built as `build` says. `identity`
// (GraphicsPipelineIdentity, ComputePipelineIdentity; 0: none) finds it among pipeline cache blobs.
template <typename Info, typename Create>
static vk::Result CreatePipelineHandle(GraphicContext& graphics, PipelineCache::Pipeline& pipeline, Info info,
                                       vk::PipelineCache driver_cache, PipelineBuild build, uint64_t identity,
                                       Create create) {
	t_static_hit      = false;
	t_blob_times      = {};
	// Stored modules go in as their SPIR-V (the key log names the modules).
	const Info   original = info;
	InlineStages inlined;
	inlined.Apply(info);
	const auto cached = [&](vk::PipelineCache cache) {
		if (cache == nullptr || !graphics.pipeline_cache_control_enabled) return false;
		auto lookup = info;
		lookup.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
		if (create(cache, lookup) == vk::Result::eSuccess) return true;
		pipeline.pipeline = nullptr;
		return false;
	};
	// Its binaries (pipelineBinaries.h) when the store holds them; the precompile keeps every pipeline's.
	const auto* store  = graphics.pipeline_binaries;
	auto*       writer = graphics.pipeline_binary_writer;
	if (build != PipelineBuild::Optimize && (store != nullptr || writer != nullptr)) {
		PipelineBinaries::Key key;
		if (PipelineBinaries::PipelineKey(graphics.device, &info, key)) {
			if (writer != nullptr && PipelineKeyLog()) LogPipelineKey("tool", key, original);
			if (InlineKeyCheck()) CheckInlineKey(graphics.device, original, key);
			// Another shard of this precompile holds it: no pipeline (see CreatePipelineInternal).
			if (writer != nullptr && writer->Claimed(key)) return vk::Result::ePipelineCompileRequired;
			if (const auto binaries = store != nullptr ? store->Load(key) : PipelineBinaries::Handles {};
			    !binaries.binaries.empty()) {
				vk::PipelineBinaryInfoKHR from {};
				from.binaryCount       = static_cast<uint32_t>(binaries.binaries.size());
				from.pPipelineBinaries = reinterpret_cast<const vk::PipelineBinaryKHR*>(binaries.binaries.data());
				auto with              = info;
				from.pNext             = with.pNext;
				with.pNext             = &from;
				const auto result      = create(nullptr, with);
				if (result == vk::Result::eSuccess) {
					if (writer != nullptr) writer->Copy(key, *store);
					if (writer == nullptr) g_binary_outcomes.Count(g_binary_outcomes.hits);
					t_static_hit = true;
					return vk::Result::eSuccess;
				}
				pipeline.pipeline = nullptr;
				if (writer == nullptr) g_binary_outcomes.Refused(result);
			} else if (writer == nullptr) {
				g_binary_outcomes.Count(store->Holds(key) ? g_binary_outcomes.damaged : g_binary_outcomes.absent);
			}
			if (writer == nullptr && PipelineKeyLog()) LogPipelineKey("miss", key, original);
			if (writer != nullptr) return CapturePipelineBinaries(graphics, pipeline, info, driver_cache, key, create);
		} else if (writer == nullptr) {
			g_binary_outcomes.Count(g_binary_outcomes.unasked);
		}
	}
	// Without pipeline binaries, its own cache data by its identity: a cache made of it gives the pipeline without
	// compiling (and holds nothing else: the driver copies a whole static cache into memory). The precompile compiles
	// one it lacks into a cache of its own, whose data it keeps.
	const auto* blobs       = graphics.pipeline_blobs;
	auto*       blob_writer = graphics.pipeline_blob_writer;
	if (build != PipelineBuild::Optimize && identity == 0 && blobs != nullptr && blob_writer == nullptr &&
	    graphics.pipeline_cache_control_enabled) {
		g_blob_outcomes.Count(g_blob_outcomes.unasked);
	}
	if (build != PipelineBuild::Optimize && identity != 0 && (blobs != nullptr || blob_writer != nullptr) &&
	    graphics.pipeline_cache_control_enabled) {
		std::vector<uint8_t> data;
		const bool           timed = PipelineSourceLog();
		auto                 start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		if (blobs != nullptr && blobs->Load(identity, data)) {
			if (timed) {
				t_blob_times.bytes   = data.size();
				t_blob_times.read_us = MicrosecondsSince(start);
				start                = std::chrono::steady_clock::now();
			}
			if (const auto cache = BlobCache(graphics, data)) {
				if (timed) {
					t_blob_times.cache_us = MicrosecondsSince(start);
					start                 = std::chrono::steady_clock::now();
				}
				auto lookup = info;
				lookup.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
				const auto result = create(cache, lookup);
				if (timed) t_blob_times.create_us = MicrosecondsSince(start);
				graphics.device.destroyPipelineCache(cache, nullptr);
				if (result == vk::Result::eSuccess) {
					if (blob_writer != nullptr) blob_writer->Copy(identity, *blobs);
					if (blob_writer == nullptr) g_blob_outcomes.Count(g_blob_outcomes.hits);
					t_static_hit = true;
					return result;
				}
				pipeline.pipeline = nullptr;
				if (blob_writer == nullptr) g_blob_outcomes.Refused(result);
			} else if (blob_writer == nullptr) {
				g_blob_outcomes.Count(g_blob_outcomes.damaged);
			}
		} else if (blobs != nullptr && blob_writer == nullptr) {
			g_blob_outcomes.Count(blobs->Holds(identity) ? g_blob_outcomes.damaged : g_blob_outcomes.absent);
		}
		if (blob_writer != nullptr) {
			// A static pipeline cache from before the blobs (a conversion, KYTY_STATIC_BLOBS_FROM_CACHE): its hit gives the
			// driver the pipeline, which a cache of the pipeline's own may then take without compiling.
			if (cached(graphics.static_pipeline_cache)) {
				graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
				pipeline.pipeline = nullptr;
			}
			vk::PipelineCacheCreateInfo empty {};
			vk::PipelineCache           own = nullptr;
			if (graphics.device.createPipelineCache(&empty, nullptr, &own) == vk::Result::eSuccess) {
				const auto result = create(own, info);
				size_t     size   = 0;
				if (result == vk::Result::eSuccess &&
				    graphics.device.getPipelineCacheData(own, &size, nullptr) == vk::Result::eSuccess && size != 0) {
					data.resize(size);
					if (graphics.device.getPipelineCacheData(own, &size, data.data()) == vk::Result::eSuccess) {
						data.resize(size);
						(void)blob_writer->Add(identity, std::move(data));
					}
				}
				graphics.device.destroyPipelineCache(own, nullptr);
				return result;
			}
		}
	}
	if (build != PipelineBuild::Optimize && store == nullptr && cached(graphics.static_pipeline_cache)) {
		t_static_hit = true;
		return vk::Result::eSuccess;
	}
	if (build != PipelineBuild::Fast) return create(driver_cache, info);
	if (cached(driver_cache)) return vk::Result::eSuccess;
	// In no cache: NVIDIA compiles this ~100x faster, so the draw or dispatch does not stall.
	info.flags |= vk::PipelineCreateFlagBits::eDisableOptimization;
	pipeline.unoptimized = true;
	return create(nullptr, info);
}

// KYTY_PS_WAVE64=1 (a test for issue #24: AMD's 26.9.2 driver runs pixel shaders on 32-lane subgroups, 26.9.1 on 64, and
// the mosaics on see-through effects came back with it): pixel shaders on 64-lane subgroups, where the GPU lets a
// pipeline require that; the log says which.
static bool PixelWave64Required(const GraphicContext& graphics) {
	static const bool required = [&] {
		const char* value = std::getenv("KYTY_PS_WAVE64");
		if (value == nullptr || std::string_view(value) != "1") return false;
		if (graphics.fragment_wave64_requirable) {
			std::printf("Pixel shaders: 64-lane subgroups required (KYTY_PS_WAVE64=1)\n");
		} else {
			std::printf("KYTY_PS_WAVE64=1: this GPU lets no pipeline require a pixel shader subgroup size (sizes %u-%u, "
			            "required sizes for stages 0x%x)\n",
			            graphics.min_subgroup_size, graphics.max_subgroup_size,
			            static_cast<vk::ShaderStageFlags::MaskType>(graphics.required_subgroup_size_stages));
		}
		std::fflush(stdout);
		return graphics.fragment_wave64_requirable;
	}();
	return required;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(
    GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
    const PipelineRenderingState& rendering, const PipelineVertexInputState& vertex_input,
    const ShaderVertexInputInfo& vs_input_info, const ShaderProgram& vertex_program,
    const ShaderPixelInputInfo* ps_input_info, const ShaderProgram& pixel_program,
    const PipelineStaticParameters& static_params, vk::PipelineCache driver_cache,
    bool native_bindings, PipelineBuild build) {
	const auto source_start = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	t_layout_us             = 0.0;
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	const bool with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                        rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	const bool mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const auto vertex_stage =
	    mesh ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex;

	const bool rect_list = !mesh && static_params.topology == vk::PrimitiveTopology::ePatchList;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;

	vk::ShaderModuleCreateInfo create_info {};
	vk::Result result {};
	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		create_info.codeSize = shaders.control.size() * 4;
		create_info.pCode    = shaders.control.data();
		result =
		    graphics.device.createShaderModule(&create_info, nullptr, &tess_control_shader_module);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done result=%s module=%p\n",
			     vk::to_string(result).c_str(), static_cast<void*>(tess_control_shader_module));
		}
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

		create_info.codeSize = shaders.evaluation.size() * 4;
		create_info.pCode    = shaders.evaluation.data();
		result =
		    graphics.device.createShaderModule(&create_info, nullptr, &tess_eval_shader_module);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done result=%s module=%p\n",
			     vk::to_string(result).c_str(), static_cast<void*>(tess_eval_shader_module));
		}
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	}

	EXIT_NOT_IMPLEMENTED(
	    rect_list && (tess_control_shader_module == nullptr || tess_eval_shader_module == nullptr));

	vk::PipelineShaderStageCreateInfo vert_shader_stage_info {};
	vert_shader_stage_info.stage  = vertex_stage;
	vert_shader_stage_info.module = vertex_program.module;
	vert_shader_stage_info.pName  = "main";

	vk::PipelineShaderStageCreateInfo frag_shader_stage_info {};
	frag_shader_stage_info.stage  = vk::ShaderStageFlagBits::eFragment;
	frag_shader_stage_info.module = pixel_program.module;
	frag_shader_stage_info.pName  = "main";
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo frag_subgroup_size {};
	if (PixelWave64Required(graphics)) {
		frag_subgroup_size.requiredSubgroupSize = 64;
		frag_shader_stage_info.pNext            = &frag_subgroup_size;
	}

	vk::PipelineShaderStageCreateInfo tess_control_shader_stage_info {};
	tess_control_shader_stage_info.stage  = vk::ShaderStageFlagBits::eTessellationControl;
	tess_control_shader_stage_info.module = tess_control_shader_module;
	tess_control_shader_stage_info.pName  = "main";

	vk::PipelineShaderStageCreateInfo tess_eval_shader_stage_info {};
	tess_eval_shader_stage_info.stage  = vk::ShaderStageFlagBits::eTessellationEvaluation;
	tess_eval_shader_stage_info.module = tess_eval_shader_module;
	tess_eval_shader_stage_info.pName  = "main";

	vk::PipelineShaderStageCreateInfo shader_stages[4]   = {};
	uint32_t                          shader_stage_count = 0;
	shader_stages[shader_stage_count++]                  = vert_shader_stage_info;
	if (rect_list) {
		shader_stages[shader_stage_count++] = tess_control_shader_stage_info;
		shader_stages[shader_stage_count++] = tess_eval_shader_stage_info;
	}
	if (ps_active) {
		shader_stages[shader_stage_count++] = frag_shader_stage_info;
	}

	vk::VertexInputAttributeDescription input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription   input_desc[ShaderVertexInputInfo::RES_MAX] {};

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		input_attr[index].binding  = vertex_input.attributes[index].binding;
		input_attr[index].location = index;
		input_attr[index].offset   = vertex_input.attributes[index].offset;

		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], input_attr[index].format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, input_attr[index].binding, input_attr[index].offset,
				     input_desc[input_attr[index].binding].stride,
				     static_cast<int>(input_attr[index].format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static bool logged = false;
			if (!logged) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
				logged = true;
			}
		}

		EXIT_NOT_IMPLEMENTED(vs_input_info.resources[index].AddTid());
		EXIT_NOT_IMPLEMENTED(vs_input_info.resources[index].SwizzleEnabled());

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	vk::PipelineVertexInputStateCreateInfo vertex_input_info {};
	vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	vertex_input_info.pVertexBindingDescriptions      = input_desc;
	vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	vertex_input_info.pVertexAttributeDescriptions    = input_attr;

	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = static_params.topology;
	input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);

	vk::PipelineViewportStateCreateInfo viewport_state {};
	viewport_state.pNext = &depth_clip_control;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (static_params.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (static_params.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}

	vk::FrontFace front_face =
	    (static_params.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);

	vk::PipelineRasterizationDepthClipStateCreateInfoEXT clip_ext {};
	clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineRasterizationStateCreateInfo rasterizer {};
	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	rasterizer.pNext = &clip_ext;
#endif
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &provoking_vertex;
	}
	rasterizer.cullMode  = cull_mode;
	rasterizer.frontFace = front_face;
	rasterizer.polygonMode = static_params.polygon_mode;
	rasterizer.lineWidth = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisampling {};
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;

	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable =
		    (static_params.blend_enable[i] && !static_params.blend_bypass[i]) ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i] ? GetBlendFactor(static_params.alpha_srcblend[i])
		                                           : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	vk::Bool32 color_write_enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		color_write_enable[i] = VK_TRUE;
	}

	vk::PipelineColorWriteCreateInfoEXT color_write {};
	color_write.attachmentCount    = rendering.color_count;
	color_write.pColorWriteEnables = color_write_enable;

	vk::PipelineColorBlendStateCreateInfo color_blending {};
	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	color_blending.pNext = &color_write;
#endif
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment;

	if (build != PipelineBuild::Optimize) {
		std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
		AddLayoutBindings(descriptor_bindings, *vs_input_info.stage.program, vertex_stage,
		                  native_bindings);
		if (ps_active) {
			EXIT_IF(!ps_input_info->stage);
			AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
			                  vk::ShaderStageFlagBits::eFragment, native_bindings);
		}
		const auto layout_start = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		EXIT_IF(pipeline.pipeline_layout != nullptr);
		SharedLayouts(graphics, pipeline, descriptor_bindings, native_bindings,
		              vertex_stage | vk::ShaderStageFlagBits::eFragment);
		if (PipelineSourceLog()) t_layout_us = MicrosecondsSince(layout_start);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: pipeline layout VS=%" PRIu64 " PS=%" PRIu64 " layout=%p\n", vertex_program.id,
			     ps_active ? pixel_program.id : 0, static_cast<void*>(pipeline.pipeline_layout));
		}
	}
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_info {};
	depth_stencil_info.depthBoundsTestEnable =
#if defined(__APPLE__)
	    VK_FALSE; // MoltenVK lacks the depthBounds feature; depth-bounds testing is disabled
#else
	    (static_params.depth_bounds_test_enable ? VK_TRUE : VK_FALSE);
#endif
	depth_stencil_info.stencilTestEnable = (static_params.stencil_test_enable ? VK_TRUE : VK_FALSE);
	depth_stencil_info.front.failOp      = static_params.stencil_front.failOp;
	depth_stencil_info.front.passOp      = static_params.stencil_front.passOp;
	depth_stencil_info.front.depthFailOp = static_params.stencil_front.depthFailOp;
	depth_stencil_info.front.compareOp   = static_params.stencil_front.compareOp;
	depth_stencil_info.back.failOp       = static_params.stencil_back.failOp;
	depth_stencil_info.back.passOp       = static_params.stencil_back.passOp;
	depth_stencil_info.back.depthFailOp  = static_params.stencil_back.depthFailOp;
	depth_stencil_info.back.compareOp    = static_params.stencil_back.compareOp;
	depth_stencil_info.minDepthBounds    = static_params.depth_min_bounds;
	depth_stencil_info.maxDepthBounds    = static_params.depth_max_bounds;

	std::vector<vk::DynamicState> dynamic_states {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
#if !defined(__APPLE__)
	if (rendering.color_count != 0) {
		dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}

	vk::PipelineDynamicStateCreateInfo dynamic_state {};
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates    = dynamic_states.data();

	vk::GraphicsPipelineCreateInfo  pipeline_info {};
	vk::PipelineRenderingCreateInfo rendering_info {};
	rendering_info.colorAttachmentCount    = rendering.color_count;
	rendering_info.pColorAttachmentFormats = rendering.color_formats.data();
	rendering_info.depthAttachmentFormat   = rendering.depth_format;
	rendering_info.stencilAttachmentFormat = rendering.stencil_format;
	pipeline_info.pNext                    = &rendering_info;
	pipeline_info.stageCount               = shader_stage_count;
	pipeline_info.pStages                  = shader_stages;
	pipeline_info.pVertexInputState        = mesh ? nullptr : &vertex_input_info;
	pipeline_info.pInputAssemblyState      = mesh ? nullptr : &input_assembly;
	vk::PipelineTessellationStateCreateInfo tessellation_state {};
	tessellation_state.patchControlPoints = 3;
	pipeline_info.pTessellationState      = (rect_list ? &tessellation_state : nullptr);
	pipeline_info.pViewportState          = &viewport_state;
	pipeline_info.pRasterizationState     = &rasterizer;
	pipeline_info.pMultisampleState       = &multisampling;
	pipeline_info.pDepthStencilState      = (with_depth ? &depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState        = &color_blending;
	pipeline_info.pDynamicState           = &dynamic_state;
	pipeline_info.layout                  = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex       = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines begin VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (with_depth ? "true" : "false"), (static_params.blend_enable[0] ? "true" : "false"),
		     dynamic_state.dynamicStateCount);
	}
	// Its identity: for pipeline cache blobs and the source log.
	const auto     identity_start = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const uint64_t identity =
	    PipelineSourceLog() || graphics.pipeline_blobs != nullptr || graphics.pipeline_blob_writer != nullptr
	        ? GraphicsPipelineIdentity(ModuleSpirvHash(vertex_program.module),
	                                   ps_active ? ModuleSpirvHash(pixel_program.module) : 0, rendering, vertex_input,
	                                   static_params, vs_input_info, ps_input_info, native_bindings)
	        : 0;
	const auto handle_start = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	result = CreatePipelineHandle(graphics, pipeline, pipeline_info, driver_cache, build, identity,
	                              [&](vk::PipelineCache cache, const vk::GraphicsPipelineCreateInfo& info) {
		                              return graphics.device.createGraphicsPipelines(cache, 1, &info, nullptr,
		                                                                            &pipeline.pipeline);
	                              });
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done result=%s pipeline=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	}
	EXIT_NOT_IMPLEMENTED(!ClaimedElsewhere(graphics, result) &&
	                     (result != vk::Result::eSuccess || pipeline.pipeline == nullptr));
	LogPipelineStatistics(
	    graphics, pipeline_info,
	    [&](const vk::GraphicsPipelineCreateInfo& create, vk::Pipeline& made) {
		    return graphics.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &made);
	    },
	    vs_input_info.stage.program->table_mode ? "gfx-table" : native_bindings ? "gfx-native" : "gfx-normal",
	    vs_input_info.stage.program->shader_hash, ps_active ? ps_input_info->stage.program->shader_hash : 0);
	if (DeviceFault::BreadcrumbsOn()) {
		DeviceFault::NotePipeline(pipeline.pipeline,
		                          fmt::format("{} vs {:016x} ps {:016x}",
		                                      vs_input_info.stage.program->table_mode ? "gfx-table"
		                                      : native_bindings                       ? "gfx-native"
		                                                                              : "gfx-normal",
		                                      vs_input_info.stage.program->shader_hash,
		                                      ps_active ? ps_input_info->stage.program->shader_hash : 0));
	}
	// KYTY_PIPELINE_SOURCE_LOG (diagnostic): each pipeline's kind, identity (the precompile's), whether the static cache
	// gave it, how it was built (0 full, 1 fast, 2 optimize) and the time it took.
	if (PipelineSourceLog()) {
		const bool  table = vs_input_info.stage.program->table_mode;
		const char* kind  = table ? "gfx-table" : native_bindings ? "gfx-native" : "gfx-normal";
		std::printf("PIPESRC %s key=%016llx static=%d build=%d unoptimized=%d ms=%.1f blob=%zu read_us=%.0f cache_us=%.0f "
		            "create_us=%.0f\n",
		            kind, static_cast<unsigned long long>(identity), t_static_hit ? 1 : 0, static_cast<int>(build),
		            pipeline.unoptimized ? 1 : 0,
		            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - source_start).count(),
		            t_blob_times.bytes, t_blob_times.read_us, t_blob_times.cache_us, t_blob_times.create_us);
		std::printf("PIPESPV key=%016llx spirv=%zu before_us=%.0f identity_us=%.0f handle_us=%.0f layout_us=%.0f\n",
		            static_cast<unsigned long long>(identity),
		            ModuleSpirvBytes(vertex_program.module) + (ps_active ? ModuleSpirvBytes(pixel_program.module) : 0),
		            std::chrono::duration<double, std::micro>(identity_start - source_start).count(),
		            std::chrono::duration<double, std::micro>(handle_start - identity_start).count(),
		            MicrosecondsSince(handle_start), t_layout_us);
	}

	if (tess_control_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
	}
	if (tess_eval_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache,
                            PipelineBuild build, bool native_bindings) {
	const auto source_start = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	t_layout_us             = 0.0;
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	// A wave64 program translated for 32-lane subgroups (emulated: SupportsComputeWave64) requires those.
	const auto wave_size = input_info.stage.program->wave_size;
	const auto subgroup  = wave_size == 64u && input_info.host_subgroup_size == 32u ? 32u : wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    subgroup >= graphics.min_subgroup_size && subgroup <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = subgroup;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	if (build != PipelineBuild::Optimize) {
		std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
		AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
		                  vk::ShaderStageFlagBits::eCompute, native_bindings);
		const auto layout_start = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		EXIT_IF(pipeline.pipeline_layout != nullptr);
		SharedLayouts(graphics, pipeline, descriptor_bindings, native_bindings, vk::ShaderStageFlagBits::eCompute);
		if (PipelineSourceLog()) t_layout_us = MicrosecondsSince(layout_start);
		LOGF("PipelineTrace: pipeline layout CS layout=%p\n", static_cast<void*>(pipeline.pipeline_layout));
	}
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::ComputePipelineCreateInfo info {};
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	const auto     identity_start = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const uint64_t identity =
	    PipelineSourceLog() || graphics.pipeline_blobs != nullptr || graphics.pipeline_blob_writer != nullptr
	        ? ComputePipelineIdentity(ModuleSpirvHash(compute_module), input_info, native_bindings)
	        : 0;
	const auto handle_start = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const auto result =
	    CreatePipelineHandle(graphics, pipeline, info, driver_cache, build, identity,
	                         [&](vk::PipelineCache cache, const vk::ComputePipelineCreateInfo& create) {
		                         return graphics.device.createComputePipelines(cache, 1, &create, nullptr,
		                                                                      &pipeline.pipeline);
	                         });
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	EXIT_NOT_IMPLEMENTED(!ClaimedElsewhere(graphics, result) &&
	                     (result != vk::Result::eSuccess || pipeline.pipeline == nullptr));
	LogPipelineStatistics(
	    graphics, info,
	    [&](const vk::ComputePipelineCreateInfo& create, vk::Pipeline& made) {
		    return graphics.device.createComputePipelines(nullptr, 1, &create, nullptr, &made);
	    },
	    native_bindings ? "cs-table" : "cs-normal", input_info.stage.program->shader_hash, 0);
	if (DeviceFault::BreadcrumbsOn()) {
		DeviceFault::NotePipeline(pipeline.pipeline, fmt::format("{} cs {:016x}", native_bindings ? "cs-table" : "cs-normal",
		                                                         input_info.stage.program->shader_hash));
	}
	if (PipelineSourceLog()) {
		std::printf("PIPESRC %s key=%016llx static=%d build=%d unoptimized=%d ms=%.1f blob=%zu read_us=%.0f cache_us=%.0f "
		            "create_us=%.0f\n",
		            native_bindings ? "cs-table" : "cs-normal", static_cast<unsigned long long>(identity),
		            t_static_hit ? 1 : 0, static_cast<int>(build), pipeline.unoptimized ? 1 : 0,
		            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - source_start).count(),
		            t_blob_times.bytes, t_blob_times.read_us, t_blob_times.cache_us, t_blob_times.create_us);
		std::printf("PIPESPV key=%016llx spirv=%zu before_us=%.0f identity_us=%.0f handle_us=%.0f layout_us=%.0f\n",
		            static_cast<unsigned long long>(identity), ModuleSpirvBytes(compute_module),
		            std::chrono::duration<double, std::micro>(identity_start - source_start).count(),
		            std::chrono::duration<double, std::micro>(handle_start - identity_start).count(),
		            MicrosecondsSince(handle_start), t_layout_us);
	}
}

} // namespace Libs::Graphics
