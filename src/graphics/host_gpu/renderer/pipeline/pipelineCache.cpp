#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/computeAhead.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineAhead.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/host_gpu/renderer/pipeline/stageStore.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>
#include <zstd.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};
// The precompile's pipelines kept for the GPU thread, at most. Each holds ~1.5 MB of the driver's host memory (NVIDIA),
// and the precompile makes them for the whole game: at Marvel's Wolverine's Trask stand ~3,960 were handed over, the GPU
// thread took 504 (boot, title and the load), and the rest held ~5 GB all session (session 62). An evicted one is still
// in the driver cache: the GPU thread's own create of it was a hit of ~1 ms (p90 7 ms), never a compile.
// KYTY_PRECOMPILE_READY_MAX sets it; 0 keeps all, as before.
constexpr size_t ReadyPipelineMax = 1024;

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// PA_SU_SC_MODE_CNTL.POLY_MODE disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

// What a stage store record (stageStore.h) depends on besides its stage record: the layout, the code and switches
// (Origin), this build of them, the GPU and driver, and the device's SPIR-V profile. Empty: no store.
std::vector<uint8_t> StageStoreHeader(const vk::PhysicalDeviceProperties& properties,
                                      const ShaderRecompiler::Spirv::Profile& profile, bool splits_fragment_set) {
	const auto stamp = Precompile::StageStore::BuildStamp();
	if (stamp == 0) return {};
	const auto text = fmt::format("stages1|{:016x}|{:016x}|{}|{}{}{}{}{}{}{}|{}|{}|{}", Precompile::Origin(), stamp,
	                              DriverCacheSignature(properties), profile.maintenance8 ? 1 : 0,
	                              profile.native_float_min_max ? 1 : 0, profile.robust_buffers ? 1 : 0,
	                              profile.wave64_split ? 1 : 0, profile.r8_srgb_views ? 1 : 0, profile.r8g8_srgb_views ? 1 : 0,
	                              profile.relaxed_float ? 1 : 0,
	                              profile.robust_buffer_alignment,
	                              profile.device_clock_period_ps, splits_fragment_set ? 1 : 0);
	return {text.begin(), text.end()};
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

// Marvel's Wolverine 1.05: the vertex and mesh shaders whose recorded pipelines the precompile also pairs with their
// other recorded vertex records (Precompile::Options::pair_vertex_shaders). Its common material passes: a first session
// at Trask Outpost met three pipelines of recorded parts no route had drawn, in frames of 0.6 and 1.7 s (session 14).
// KYTY_PRECOMPILE_PAIR=0 leaves it off.
std::unordered_set<uint64_t> PairedVertexShaders(const std::string& title_id) {
	const char* pair = std::getenv("KYTY_PRECOMPILE_PAIR");
	if (title_id != "PPSA03671" || (pair != nullptr && pair[0] == '0')) return {};
	return {0xb1b8b8b4ce99b497ull, 0xd4526a350524b689ull};
}

// Marvel's Wolverine 1.05: the precompile also makes the pixel records PredictVolumeForm guesses
// (Precompile::Options::predict_volume_pixels). Trask Outpost's first session met 27 new records of known shaders; the
// rule, from the journal before it, gives 22 of them exactly (session 14). KYTY_PRECOMPILE_PREDICT=0 leaves it off.
bool PredictsVolumePixels(const std::string& title_id) {
	const char* predict = std::getenv("KYTY_PRECOMPILE_PREDICT");
	return title_id == "PPSA03671" && (predict == nullptr || predict[0] != '0');
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

} // namespace

bool WriteDriverCacheFile(const std::filesystem::path& path, std::string_view signature,
                          std::span<const uint8_t> payload) {
	if (payload.empty() || payload.size() > std::numeric_limits<uint32_t>::max() ||
	    !Common::File::CreateDirectories(path.parent_path())) {
		return false;
	}
	std::string prefix(signature);
	const auto  payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	auto temp_path = path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	std::error_code error;
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed) return false;
	std::filesystem::rename(temp_path, path, error);
	return !error;
}

bool ReadDriverCacheFile(const std::filesystem::path& path, std::string_view signature, std::vector<uint8_t>& payload) {
	payload.clear();
	if (!Common::File::IsFileExisting(path)) return false;
	Common::File file(path, Common::File::Mode::Read);
	const auto   file_size = file.IsInvalid() ? 0 : file.Size();
	if (file_size < signature.size() + sizeof(uint64_t) || file_size > std::numeric_limits<uint32_t>::max()) {
		file.Close();
		return false;
	}
	std::string cached_signature(signature.size(), '\0');
	uint64_t    payload_hash = 0;
	payload.resize(file_size - signature.size() - sizeof(payload_hash));
	uint32_t signature_read = 0;
	uint32_t hash_read      = 0;
	uint32_t payload_read   = 0;
	file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()), &signature_read);
	file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
	file.Read(payload.data(), static_cast<uint32_t>(payload.size()), &payload_read);
	file.Close();
	if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
	    payload_read != payload.size() || cached_signature != signature ||
	    XXH3_64bits(payload.data(), payload.size()) != payload_hash) {
		payload.clear();
		return false;
	}
	return true;
}

namespace {

// The backing bytes of the guest pages one program lookup reads strictly (MaterializeResources' clean reads and tables),
// found once per page: each read otherwise took the backing store's lock, a map lookup and the buffer cache's GPU-dirty
// query (our old fork's GuestPageReadCache, 2026-09-30, jungle 6.2 -> 7.8 fps then). Only pages with no GPU-written
// byte are kept, and only within one lookup on the GPU thread, which marks nothing GPU-written while it materializes.
// Any other page takes TryReadBufferBacking as before. Page 0 is never kept (GuestRange treats address 0 as no range).
struct GuestPageReadCache {
	static constexpr size_t Capacity = 16;
	struct Entry {
		uint64_t       page    = 0;
		const uint8_t* backing = nullptr;
	};
	std::array<Entry, Capacity> entries {};
	size_t                      count = 0;
	size_t                      next  = 0;

	void Clear() { count = next = 0; }
	// True when the page is GPU-clean and the words were copied from its backing.
	bool Read(uint64_t address, std::span<uint32_t> values) {
		const uint8_t* at = Locate(address, values.size_bytes());
		if (at == nullptr) return false;
		std::memcpy(values.data(), at, values.size_bytes());
		return true;
	}
	// The backing bytes of [address, address + bytes) when they lie on one GPU-clean page, else nullptr.
	const uint8_t* Locate(uint64_t address, size_t bytes) {
		const auto page = address & ~(TRACKER_PAGE_SIZE - 1u);
		if (page == 0 || bytes == 0 || ((address + bytes - 1u) & ~(TRACKER_PAGE_SIZE - 1u)) != page) {
			return nullptr;
		}
		const Entry* entry = nullptr;
		for (size_t i = 0; i < count; i++) {
			if (entries[i].page == page) {
				entry = &entries[i];
				break;
			}
		}
		if (entry == nullptr) {
			auto& slot = count < Capacity ? entries[count++] : entries[next++ % Capacity];
			slot = {.page = page, .backing = Libs::LibKernel::Memory::FindGpuCleanBacking(page, TRACKER_PAGE_SIZE)};
			entry = &slot;
		}
		if (entry->backing == nullptr) {
			return nullptr;
		}
		return entry->backing + (address - page);
	}
};

// One per thread: its address is the runtime's userdata, which the read log compares from call to call.
thread_local GuestPageReadCache t_page_read_cache;

// PipelineCache::AheadSubmission: 0 on threads that are neither the GPU thread nor the scanner (the workers).
thread_local uint64_t t_ahead_submission = 0;

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata); cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

// The ahead scanner materializes a draw's or dispatch's resources before the GPU thread reaches it, to build its
// pipeline ahead, and logs every guest word it read. The GPU thread, about to materialize the same program with the
// same runtime, takes that result instead when each logged read gives the same words through the page cache's
// clean-only reader: a page the GPU has written refuses, and the draw materializes as before. The same program is the
// stage, shader hash and static key, as Ahead::ready keys a plan both threads use.
struct AheadResult {
	ShaderType                                   stage = ShaderType::Unknown;
	uint64_t                                     hash  = 0;
	std::vector<uint32_t>                        static_key, user_data, workgroup_counts, workgroup_size;
	uint64_t                                     shader_base = 0, indirect_args_address = 0;
	bool                                         guest_image_heap = false;
	ShaderRecompiler::IR::SrtReadLog             log;
	ShaderRecompiler::IR::ResourceSnapshot       resources;
	ShaderRecompiler::IR::ResourceSpecialization specialization;

	static uint64_t Key(ShaderType stage, uint64_t hash, std::span<const uint32_t> static_key,
	                    std::span<const uint32_t> user_data, uint64_t shader_base,
	                    std::span<const uint32_t> workgroup_counts, uint64_t indirect_args_address) {
		auto key = XXH3_64bits_withSeed(static_key.data(), static_key.size_bytes(),
		                                hash ^ (static_cast<uint64_t>(stage) << 56u));
		key      = XXH3_64bits_withSeed(user_data.data(), user_data.size_bytes(), key ^ shader_base);
		return XXH3_64bits_withSeed(workgroup_counts.data(), workgroup_counts.size_bytes(), key ^ indirect_args_address);
	}
	bool Matches(ShaderType other_stage, uint64_t other_hash, std::span<const uint32_t> other_static_key,
	             const ShaderRecompiler::IR::SrtRuntime& runtime) const {
		return stage == other_stage && hash == other_hash && std::ranges::equal(static_key, other_static_key) &&
		       shader_base == runtime.shader_base && indirect_args_address == runtime.indirect_args_address &&
		       guest_image_heap == runtime.guest_image_heap && std::ranges::equal(user_data, runtime.user_data) &&
		       std::ranges::equal(workgroup_counts, runtime.workgroup_counts) &&
		       std::ranges::equal(workgroup_size, runtime.workgroup_size);
	}
};

class AheadResults {
public:
	void Publish(std::unique_ptr<AheadResult> result) {
		const auto key = AheadResult::Key(result->stage, result->hash, result->static_key, result->user_data,
		                                  result->shader_base, result->workgroup_counts, result->indirect_args_address);
		// Several workers publish at once. Each frees what it holds here, outside the lock, and keeps the storage.
		static thread_local std::vector<std::unique_ptr<AheadResult>> doomed;
		{
			std::lock_guard lock(m_mutex);
			doomed.swap(m_spent);
			auto& slot = m_results[key];
			if (slot) {
				Hold(*slot, -1);
				doomed.push_back(std::move(slot));
			}
			Hold(*result, 1);
			slot = std::move(result);
			m_order.push_back(key);
			// The oldest go first: the GPU thread takes a result within a frame or two, or never (its draw repeated).
			while (m_order.size() > Capacity) {
				if (const auto found = m_results.find(m_order.front()); found != m_results.end()) {
					Hold(*found->second, -1);
					doomed.push_back(std::move(found->second));
					m_results.erase(found);
				}
				m_order.pop_front();
			}
			m_count.store(m_results.size(), std::memory_order_relaxed);
		}
		doomed.clear();
	}
	// The GPU thread, about to materialize: true when it took the scanner's result for this program and runtime into
	// resources, specialization and log.
	bool Take(ShaderType stage, uint64_t hash, std::span<const uint32_t> static_key,
	          const ShaderRecompiler::IR::SrtRuntime& runtime, ShaderRecompiler::IR::ResourceSnapshot& resources,
	          ShaderRecompiler::IR::ResourceSpecialization& specialization, ShaderRecompiler::IR::SrtReadLog& log) {
		if (m_count.load(std::memory_order_relaxed) == 0) return false;
		const auto key = AheadResult::Key(stage, hash, static_key, runtime.user_data, runtime.shader_base,
		                                  runtime.workgroup_counts, runtime.indirect_args_address);
		std::unique_ptr<AheadResult> result;
		{
			std::lock_guard lock(m_mutex);
			const auto      found = m_results.find(key);
			if (found == m_results.end()) {
				// The scanner hasn't published it (yet): it need not, this draw is past.
				if (t_ahead_submission != 0) {
					// shortcut: the marks of draws the scanner never reaches pile up; all go at a capacity's worth.
					if (m_passed.size() >= Capacity) m_passed.clear();
					m_passed[key] = t_ahead_submission;
				}
				return false;
			}
			result = std::move(found->second);
			Hold(*result, -1);
			m_results.erase(found);
			m_count.store(m_results.size(), std::memory_order_relaxed);
		}
		const auto read_clean = [](void* cache, uint64_t address, std::span<uint32_t> values) {
			return static_cast<GuestPageReadCache*>(cache)->Read(address, values);
		};
		const auto peek_clean = [](void* cache, uint64_t address, size_t words) {
			return reinterpret_cast<const uint32_t*>(static_cast<GuestPageReadCache*>(cache)->Locate(address, words * 4u));
		};
		const bool taken = result->Matches(stage, hash, static_key, runtime) &&
		                   result->log.StillReads(read_clean, &t_page_read_cache, peek_clean);
		if (taken) {
			result->log.Retarget(runtime);
			std::swap(resources, result->resources);
			std::swap(specialization, result->specialization);
			std::swap(log, result->log);
		}
		// What is left of the result, and what it swapped out, is the scanner's to free (Publish): a result's dozens of
		// small blocks cost the GPU thread more than a millisecond a frame at Marvel's Wolverine's Trask outpost.
		std::lock_guard lock(m_mutex);
		m_spent.push_back(std::move(result));
		return taken;
	}
	// The scanner, before it materializes: true when the GPU thread already passed this draw in the submission the
	// scanner reads (it found no result for the key there), so a result would only be evicted.
	bool Passed(uint64_t key) {
		if (t_ahead_submission == 0) return false;
		std::lock_guard lock(m_mutex);
		const auto      found = m_passed.find(key);
		if (found == m_passed.end()) return false;
		const bool passed = found->second == t_ahead_submission;
		m_passed.erase(found);
		return passed;
	}

private:
	// Logs over 64 KB (the scanner's, up to 4 MB) count against a cap of the queued results' total: a result published
	// past it gives its log up, so it can't be taken and the GPU thread materializes it as before. Under the mutex.
	// shortcut: a fixed 128 MB (Marvel's Wolverine's fight held 410 MB without one, 34-lb2-big2); measure a larger cap
	// against the commit peak before raising it.
	static constexpr size_t BigLogCap = size_t {128} << 20u;
	void Hold(AheadResult& result, int sign) {
		const size_t bytes = result.log.Words() * 4u;
		if (bytes <= ShaderRecompiler::IR::SrtReadLog::MaxWords * 4u) return;
		if (sign < 0) {
			m_big_bytes -= bytes;
		} else if (m_big_bytes + bytes > BigLogCap) {
			result.log.Release();
		} else {
			m_big_bytes += bytes;
		}
	}
	// shortcut: 8192 results, about a frame of the fight's materializations; raise it if taken results fall short of
	// the scanner's.
	static constexpr size_t Capacity = 8192u;
	size_t                                                        m_big_bytes = 0; // logs over 64 KB held, under the mutex
	std::mutex                                                    m_mutex;
	std::unordered_map<uint64_t, std::unique_ptr<AheadResult>>    m_results;
	std::deque<uint64_t>                                          m_order;
	std::atomic<size_t>                                           m_count {0};
	std::vector<std::unique_ptr<AheadResult>>                     m_spent;  // taken results (at most a capacity's), for Publish to free
	std::unordered_map<uint64_t, uint64_t>                        m_passed; // key the GPU thread found absent -> its submission
};

// The reuse store's key of a shader with calls: its static key and its linked code (the callees decide the program).
void AppendLinkedKey(std::vector<uint32_t>& key, std::span<const uint32_t> linked_code) {
	const auto hash = XXH3_64bits(linked_code.data(), linked_code.size_bytes());
	key.insert(key.end(), {0x4c4c4143u /* "CALL" */, static_cast<uint32_t>(hash), static_cast<uint32_t>(hash >> 32u)});
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

// A stage's log label and dump name.
std::pair<const char*, const char*> StageNames(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return {"ShaderRecompiler VS", "vs"};
		case ShaderType::Mesh: return {"ShaderRecompiler MS", "ms"};
		case ShaderType::Local: return {"ShaderRecompiler LS", "ls"};
		case ShaderType::TessellationControl: return {"ShaderRecompiler HS", "hs"};
		case ShaderType::TessellationEvaluation: return {"ShaderRecompiler DS", "ds"};
		case ShaderType::Pixel: return {"ShaderRecompiler PS", "ps"};
		case ShaderType::Compute: return {"ShaderRecompiler CS", "cs"};
		default: EXIT("invalid pipeline shader stage\n");
	}
}

// The options ProgramCache::Get translates and compiles a stage with; a precompile's replay of a record takes the same.
ShaderRecompiler::CompileOptions StageCompileOptions(ShaderType stage, uint64_t shader_hash,
                                                     std::span<const uint32_t> user_data,
                                                     std::span<const uint32_t> back_code, ShaderStageInputInfo input,
                                                     const ShaderRecompiler::Spirv::Profile& profile,
                                                     bool split_fragment_set) {
	ShaderRecompiler::CompileOptions options;
	options.stage           = stage;
	options.profile         = &profile;
	options.descriptor_set  = stage == ShaderType::Pixel && split_fragment_set ? ShaderRecompiler::IR::FragmentSet : 0u;
	options.shader_hash     = shader_hash;
	options.user_data       = user_data;
	options.back_code       = back_code;
	options.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.early_dump      = options.dump_ir;
	options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
	options.dump_label      = StageNames(stage).first;
	options.input_info      = input;
	if (stage == ShaderType::Pixel) {
		options.wave_size = input.pixel->wave_size;
	} else if (stage == ShaderType::Compute) {
		options.wave_size = input.compute->wave_size;
	} else {
		options.user_data_base = 8;
		options.wave_size      = input.vertex->wave_size;
		if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
			options.user_data_base = 0;
		}
		// A tessellation stage's wave comes from its input (VGT_SHADER_STAGES_EN).
		if (stage == ShaderType::Mesh) options.wave_size = input.vertex->mesh.wave_size;
	}
	return options;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		// take(*this): true when it gave the entry the ahead scanner's result for this draw (AheadResults::Take).
		template <bool HasCalls, typename TakeAhead>
		Permutation* FindPermutation(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                             uint32_t push_data_cursor, TakeAhead&& take) {
			// A draw whose runtime, guest words and walk are those of the entry's previous draw has its
			// resources already. A call source can change the plan: those entries materialize every time.
			// So does every draw while the resident image heap is attached, which acquires its views as it
			// materializes.
			// A check that walked and found the draw changed hands its walk to the materialization.
			const std::vector<uint32_t>* walked = nullptr;
			if (HasCalls || runtime.image_heap != nullptr || !read_log.Repeats(runtime, resource_plan, &walked)) {
				const bool log = !HasCalls;
				if (!log) read_log.Clear();
				if (!log || runtime.image_heap != nullptr || !take(*this)) {
					EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
					    resource_plan, log ? read_log.Recording(runtime) : runtime, resources, specialization, walked));
					if (log) read_log.RecordWalk(resource_plan, resources.flattened_srt);
				}
			}
			// A repeated draw kept its specialization: the buffer table variant is chosen per call.
			specialization.buffer_table = ShaderRecompiler::IR::BufferTableWanted(resource_plan);
			for (auto& candidate: permutations) {
				const auto& layout = candidate.program.bindings;
				if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
				        push_data_cursor, layout.ShaderDataDwords()) ||
				    candidate.specialization != specialization) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
				}
				return &candidate;
			}
			return nullptr;
		}

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		bool                                       skip_dispatch = false;
		ShaderRecompiler::IR::SrtReadLog            read_log;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.profile->maintenance8)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		const auto id = ++next_shader_id;
		specialization_of.emplace(id, specialization);
		return {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id         = id,
		                       .module     = module,
		                       .spirv_hash = XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t))},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			auto keyed          = input_info;
			keyed.async_compute = input_info.async_compute && QueueKeyed(params);
			BuildStageStaticKey(keyed, lookup_key.static_state);
		} else {
			BuildStageStaticKey(input_info, lookup_key.static_state);
		}
		auto                                         entry = programs.find(lookup_key);
		t_page_read_cache.Clear(); // valid within this lookup only
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .userdata                   = &t_page_read_cache,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .image_heap                 = image_heap,
		    .guest_image_heap           = guest_image_heap,
		    .direct_reads               = Libs::Graphics::GuestGpu::IsGpuThread(),
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			runtime.workgroup_size   = input_info.threads_num;
			runtime.indirect_args_address = input_info.indirect_args_address;
		}
		bool source_changed = false;
		if (entry != programs.end()) {
			auto& cached = entry->second;
			if (cached.skip_dispatch) return {};
			Permutation* permutation = nullptr;
			if (cached.call_source) {
				source_changed = cached.RefreshCallSource(runtime);
				if (!source_changed) {
					permutation = cached.FindPermutation<true>(runtime, push_data_cursor, [](SourceEntry&) { return false; });
				}
			} else {
				permutation = cached.FindPermutation<false>(runtime, push_data_cursor, [&](SourceEntry& source) {
					if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
						std::vector<uint32_t> static_key;
						BuildStageStaticKey(input_info, static_key);
						return ahead_results.Take(stage, params.hash, static_key, runtime, source.resources,
						                          source.specialization, source.read_log);
					} else {
						return ahead_results.Take(stage, params.hash, lookup_key.static_state, runtime, source.resources,
						                          source.specialization, source.read_log);
					}
				});
			}
			if (permutation != nullptr) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* stage_name = StageNames(stage).second;
		auto        options    = StageCompileOptions(stage, params.hash, user_data, params.back_code, stage_input, profile,
		                                             graphics.SplitsFragmentSet());
		// A worker translated this program already (Ahead::ready): its module, not a second translation.
		{
			if (owner != nullptr && (entry == programs.end() || entry->second.call_source == nullptr)) {
				std::vector<uint32_t> ready_key;
				BuildStageStaticKey(input_info, ready_key);
				if (entry == programs.end()) {
					if (auto plan = owner->TakeReadyPlan(stage, params.hash, ready_key)) {
						entry = programs.try_emplace(lookup_key, std::move(*plan), nullptr).first;
						EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(entry->second.resource_plan, runtime,
						                                                    entry->second.resources,
						                                                    entry->second.specialization));
					}
				}
				std::vector<uint32_t>                    spirv;
				ShaderRecompiler::IR::CompiledShaderInfo info;
				if (entry != programs.end() &&
				    owner->TakeReadyModule(stage, params.hash, ready_key, entry->second.specialization, push_data_cursor,
				                           spirv, info)) {
					LOGF("Shader build: %s hash=0x%016" PRIx64 " from a worker\n", stage_name, params.hash);
					if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, spirv, options.profile->maintenance8)) {
						EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
						     options.shader_hash);
					}
					DumpShaderSpirv(stage_name, options.shader_hash, spirv);
					const auto module = CompileSPV(spirv, graphics.device);
					EXIT_IF(module == nullptr);
					const auto id = ++next_shader_id;
					specialization_of.emplace(id, entry->second.specialization);
					entry->second.permutations.push_back({.specialization = entry->second.specialization,
					                                      .program        = std::move(info),
					                                      .handle         = {.id = id, .module = module,
					                                                         .spirv_hash = XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t))}});
					const auto& permutation = entry->second.permutations.back();
					input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
					Record(stage, params, input_info, push_data_cursor, entry->second.specialization, permutation.handle);
					permutation.program.bindings.AdvancePushData(push_data_cursor);
					return permutation.handle;
				}
			}
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		ShaderRecompiler::ShaderSource source;
		if (entry == programs.end()) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (source.unsupported_call_pc != UINT32_MAX) {
			PipelineCacheLog("!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x{:016x} pc=0x{:08x} "
			                 "reason=unsupported GPU-selected callable target; rendering is incomplete",
			                 params.hash, source.unsupported_call_pc);
			programs.try_emplace(lookup_key, ShaderRecompiler::IR::ResourcePlan{}, nullptr)
			    .first->second.skip_dispatch = true;
			return {};
		}
		// A shader with calls the precompile translated (journaled with its callees): linked as below, its plan and
		// module are taken from the reuse store under its linked code instead of translating it here.
		if (owner != nullptr && entry == programs.end() && !source.calls.empty()) {
			ShaderRecompiler::RefreshShaderSource(source, runtime);
			std::vector<uint32_t> ready_key;
			BuildStageStaticKey(input_info, ready_key);
			AppendLinkedKey(ready_key, source.linked->code);
			if (auto plan = owner->TakeReadyPlan(stage, params.hash, ready_key)) {
				plan->source_reads = source.reads; // the draw's reads, not the precompile's stand-in addresses
				entry = programs.try_emplace(lookup_key, std::move(*plan),
				                             std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)))
				            .first;
				EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(entry->second.resource_plan, runtime,
				                                                    entry->second.resources,
				                                                    entry->second.specialization));
				std::vector<uint32_t>                    spirv;
				ShaderRecompiler::IR::CompiledShaderInfo info;
				if (owner->TakeReadyModule(stage, params.hash, ready_key, entry->second.specialization, push_data_cursor,
				                           spirv, info)) {
					LOGF("Shader build: %s hash=0x%016" PRIx64 " with calls from a worker\n", stage_name, params.hash);
					if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, spirv, options.profile->maintenance8)) {
						EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
						     options.shader_hash);
					}
					DumpShaderSpirv(stage_name, options.shader_hash, spirv);
					const auto module = CompileSPV(spirv, graphics.device);
					EXIT_IF(module == nullptr);
					const auto id = ++next_shader_id;
					specialization_of.emplace(id, entry->second.specialization);
					const auto function_code = entry->second.FunctionCode();
					entry->second.permutations.push_back(
					    {.function_code  = {function_code.begin(), function_code.end()},
					     .specialization = entry->second.specialization,
					     .program        = std::move(info),
					     .handle         = {.id = id, .module = module,
					                        .spirv_hash = XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t))}});
					const auto& permutation = entry->second.permutations.back();
					input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
					if (auto callees = CalleeShaders(*entry->second.call_source); !callees.empty()) {
						Record(stage, params, input_info, push_data_cursor, entry->second.specialization, permutation.handle,
						       std::move(callees));
					}
					permutation.program.bindings.AdvancePushData(push_data_cursor);
					return permutation.handle;
				}
			}
		}
		auto* current = entry == programs.end() ? &source : entry->second.call_source.get();
		ShaderRecompiler::TranslateResult translated;
		if (current != nullptr) {
			if (!current->calls.empty()) {
				if (entry == programs.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
				translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
				translated.program.source_reads = current->reads;
			} else {
				translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
			}
		} else {
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		if (entry == programs.end()) {
			auto retained = !source.calls.empty() ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program), std::move(retained)).first;
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
		} else if (source_changed) {
			entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			if (const auto* permutation = entry->second.FindPermutation<true>(runtime, push_data_cursor,
			                                                                         [](SourceEntry&) { return false; })) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}
		// The GPU thread compiles this itself: a precompile still running is needed now, not in the background.
		if (owner != nullptr && owner->m_precompiler != nullptr) owner->m_precompiler->Hurry();
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor,
		    entry->second.FunctionCode()));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		if (entry->second.call_source == nullptr) {
			Record(stage, params, input_info, push_data_cursor, entry->second.specialization, permutation.handle);
		} else if (auto callees = CalleeShaders(*entry->second.call_source); !callees.empty()) {
			Record(stage, params, input_info, push_data_cursor, entry->second.specialization, permutation.handle,
			       std::move(callees));
		}
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// A shader with calls: each call site's callee as the whole shader the game created at its address (its leaf is
	// read from there), for the journal. None when a call goes elsewhere: such a shader is not recorded.
	static std::vector<std::vector<uint32_t>> CalleeShaders(const ShaderRecompiler::ShaderSource& source) {
		std::vector<std::vector<uint32_t>> callees;
		for (const auto& site: source.calls) {
			const auto bytes = ShaderMappedCodeSize(site.address);
			if (bytes == 0 || bytes % sizeof(uint32_t) != 0) return {};
			const auto* code = reinterpret_cast<const uint32_t*>(site.address);
			callees.emplace_back(code, code + bytes / sizeof(uint32_t));
		}
		return callees;
	}

	// A new permutation into the journal (shaderPrecompile.h), with every input of its translation; push_data_cursor as
	// Get received it, or PushData::NoStart for a program without push data: that program is the same at every cursor
	// (FindPermutation takes it for any), and a vertex stage's cursor is where its draw's pixel shader's push data ends.
	// Keyed by the cursor it was first drawn at, one program had a key per session, and its precompiled pipelines missed
	// in the next. A program with calls reads its callees from guest memory and is not recorded.
	template <typename InputInfo>
	void Record(ShaderType stage, const ShaderParams& params, const InputInfo& input_info, uint32_t push_data_cursor,
	            const ShaderRecompiler::IR::ResourceSpecialization& specialization, const ShaderProgram& handle,
	            std::vector<std::vector<uint32_t>> callees = {}) {
		if (journal == nullptr) return;
		const auto* program = input_info.stage.program;
		Precompile::StageRecord record {.stage            = stage,
		                                .shader_hash      = params.hash,
		                                .user_data_count  = params.user_data_count,
		                                .input            = Precompile::RecordedInput(input_info),
		                                .push_data_cursor = program != nullptr && !program->bindings.UsesPushData()
		                                                        ? ShaderRecompiler::IR::PushData::NoStart
		                                                        : push_data_cursor,
		                                .specialization   = specialization,
		                                .spirv_hash       = handle.spirv_hash,
		                                .origin           = Precompile::Origin()};
		record.code.resize(params.code.size()); // the key reads the sizes only
		record.back_code.resize(params.back_code.size());
		record.callees = std::move(callees); // the key reads their hashes
		const auto key = record.Key();
		record_of.emplace(handle.id, key);
		if (recorded.contains(key)) return;
		// The guest's code under the shader changed since the lookup: the record would be of other code.
		if (Precompile::CodeHash(params.code, params.back_code) != params.hash) return;
		record.code.assign(params.code.begin(), params.code.end());
		record.back_code.assign(params.back_code.begin(), params.back_code.end());
		recorded.insert(key);
		journal->Append(record);
	}

	ProgramCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap)
	    : graphics(graphics), image_heap(image_heap), guest_image_heap(graphics.guest_image_heap != nullptr),
	      // Native NMin/NMax on every GPU. GLSL.std.450 gives the guest's result for one NaN source and
	      // leaves -0 against +0 and two NaNs open; AMD's v_min/v_max match the guest there too, other
	      // GPUs may not. The integer fixes for those two cases cost 12 % of a large pixel shader's
	      // compile on NVIDIA, for results that only a later sign-sensitive use of a zero could show.
	      profile {.maintenance8 = graphics.maintenance8_enabled,
		           .native_float_min_max   = true,
		           .robust_buffers         = graphics.robust_buffer_access2,
		           .robust_buffer_alignment = graphics.robust_storage_buffer_alignment,
		           .device_clock_period_ps = graphics.device_clock_period_ps,
		           .wave64_split = graphics.compute_full_subgroups_enabled && graphics.subgroup_size == 32u,
		           .r8_srgb_views   = graphics.r8_srgb_views,
		           .r8g8_srgb_views = graphics.r8g8_srgb_views,
		           .relaxed_float   = Config::RelaxedShaderFloatEnabled()} {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	std::unordered_map<uint64_t, bool>                           reads_queue_id; // by shader hash, see QueueKeyed

	// The queue a compute shader runs on changes its translation only through S_GETREG_B32 (HW_ID2,
	// Translator::graphics_compute): a shader without it is one entry for the graphics and the async
	// compute queues, not two (the GI prepass shaders were each translated and specialized twice).
	// A shader with calls may read it in the callee: keyed by queue as before.
	bool QueueKeyed(const ShaderParams& params) {
		const auto [found, inserted] = reads_queue_id.try_emplace(params.hash, false);
		if (inserted) {
			ShaderRecompiler::Decoder::Program decoded;
			ShaderRecompiler::Decoder::DecodeProgram(params.code, decoded);
			found->second = decoded.has_swap_pc || std::ranges::any_of(decoded.instructions, [](const auto& inst) {
				using ShaderRecompiler::Decoder::Opcode;
				return inst.opcode == Opcode::S_GETREG_B32 || inst.opcode == Opcode::S_SETPC_B64;
			});
		}
		return found->second;
	}
	ProgramKey                                                  lookup_key;
	GraphicContext&                                             graphics;
	ShaderRecompiler::IR::ImageHeapSource*                      image_heap;
	bool                                                        guest_image_heap;
	const ShaderRecompiler::Spirv::Profile                      profile;
	uint64_t                                                    next_shader_id = 0;
	// Each module's resource specialization by its id (graphics ahead's templates; the GPU thread's).
	std::unordered_map<uint64_t, ShaderRecompiler::IR::ResourceSpecialization> specialization_of;
	PipelineCache*                                              owner          = nullptr; // its workers' modules, see Ahead::ready
	Precompile::JournalWriter*                                  journal        = nullptr; // where Record writes
	std::unordered_set<uint64_t>                                recorded;  // stage record keys in the journal
	std::unordered_map<uint64_t, uint64_t>                      record_of; // a module's stage record key, by its id
	AheadResults                                                ahead_results; // the scanner's materializations
};

PipelineCache::PipelineCache(GraphicContext&                         graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics, image_heap)),
      m_libraries(std::make_unique<PipelineLibraries>()),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	if (Config::ShaderCacheEnabled()) {
		// Upstream's shader store (ec8b75b24) belongs to its pipeline cache, which this build replaces;
		// the driver's pipeline cache below is kept as before.
		Log::WriteToConsoleAndLog("Warning: --shader-cache is not supported by this build; shaders are not kept.\n");
	}
	InitializeDriverCache();
	StartComputeAhead();
	m_program_cache->owner = this;
	InitializePrecompile();
}

PipelineCache::~PipelineCache() {
	// Its workers hand translations to the ahead store (the reuse store): they stop first.
	m_precompiler.reset();
	for (auto& [key, ready]: m_ready_pipelines) {
		(void)key;
		m_graphics.device.destroyPipeline(ready.pipeline.pipeline, nullptr);
		m_graphics.device.destroyPipelineLayout(ready.pipeline.pipeline_layout, nullptr);
		m_graphics.device.destroyDescriptorSetLayout(ready.pipeline.descriptor_set_layout, nullptr);
		m_graphics.device.destroyDescriptorSetLayout(ready.pipeline.fragment_set_layout, nullptr);
		for (const auto layout: ready.pipeline.group_set_layouts) m_graphics.device.destroyDescriptorSetLayout(layout, nullptr);
	}
	m_ready_pipelines.clear();
	m_ready_order.clear();
	StopComputeAhead();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->fragment_set_layout, nullptr);
			for (const auto layout: pipeline->group_set_layouts) m_graphics.device.destroyDescriptorSetLayout(layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	m_libraries->Destroy(m_graphics.device);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists &&
	    !ReadDriverCacheFile(m_driver_cache_path, DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties()),
	                         initial_data)) {
		PipelineCacheLog("Vulkan pipeline cache: invalidating {} (driver, data or size mismatch)", path);
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	m_driver_cache_loaded_bytes = initial_data.size();
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_driver_cache_dirty = true;
	if (std::chrono::steady_clock::now() < m_next_driver_cache_save) {
		return;
	}
	// The hash and the write of 150-200 MB took 0.16-0.8 s on the GPU thread (Marvel's Wolverine): a worker does
	// them. A checkpoint while the last one still writes waits for a later pipeline.
	if (m_driver_cache_write.valid()) {
		if (m_driver_cache_write.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
		m_driver_cache_write.get();
	}
	if (auto write = SnapshotDriverCache()) {
		m_driver_cache_write = std::async(std::launch::async, std::move(write));
	}
}

void PipelineCache::Save() {
	if (m_driver_cache_write.valid()) {
		m_driver_cache_write.get();
	}
	if (auto write = SnapshotDriverCache()) {
		write();
	}
}

std::function<void()> PipelineCache::SnapshotDriverCache() {
	if (m_driver_cache == nullptr || !m_driver_cache_dirty) {
		return {};
	}
	m_next_driver_cache_save = std::chrono::steady_clock::now() + DriverCacheSaveInterval;
	// The pipelines marked warm until now are all in the data taken below (shaderPrecompile.h, WarmList).
	auto marked = m_warm != nullptr ? m_warm->TakeMarked() : std::vector<uint64_t> {};

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		if (m_warm != nullptr) m_warm->Commit(std::move(marked), false, 0);
		return {};
	}
	payload.resize(size);
	// Pipelines made from here on mark the cache dirty again; a failed write keeps it dirty.
	m_driver_cache_dirty = false;
	return [this, payload = std::move(payload), signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties()),
	        path = m_driver_cache_path, marked = std::move(marked)]() mutable {
		if (!WriteDriverCacheFile(path, signature, payload)) {
			m_driver_cache_dirty       = true;
			m_driver_cache_save_failed = true;
			if (m_warm != nullptr) m_warm->Commit(std::move(marked), false, 0);
			PipelineCacheLog("Vulkan pipeline cache: failed to write {}", Common::PathToString(path));
			return;
		}
		if (m_warm != nullptr) m_warm->Commit(std::move(marked), true, payload.size());
		PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
		                 static_cast<double>(payload.size()) / 1'000'000.0, Common::PathToString(path));
	};
}

// A precompile's replay of a record (shaderPrecompile.h): ProgramCache::Get's translation and compile, with the same
// options, then the pipeline as GetGraphicsPipeline or GetComputePipeline creates it, into the driver cache.
class PipelineCache::PrecompileCompiler final : public Precompile::Compiler {
public:
	explicit PrecompileCompiler(PipelineCache& cache): m_cache(cache) {}

	bool CompileStage(const Precompile::StageRecord& record, bool keep, Precompile::StageModule& out) override {
		static constexpr std::array<uint32_t, 40> NoUserData {}; // the translation reads the count only
		const bool pixel   = record.stage == ShaderType::Pixel;
		const bool compute = record.stage == ShaderType::Compute;
		if (record.user_data_count > NoUserData.size() || pixel != std::holds_alternative<ShaderPixelInputInfo>(record.input) ||
		    compute != std::holds_alternative<ShaderComputeInputInfo>(record.input)) {
			return false;
		}
		out.input = record.input;
		ShaderStageInputInfo input {};
		std::visit(
		    [&](auto& info) {
			    using Info = std::decay_t<decltype(info)>;
			    if constexpr (std::is_same_v<Info, ShaderVertexInputInfo>) {
				    input.vertex = &info;
			    } else if constexpr (std::is_same_v<Info, ShaderPixelInputInfo>) {
				    input.pixel = &info;
			    } else {
				    input.compute = &info;
			    }
		    },
		    out.input);
		auto&      graphics = m_cache.m_graphics;
		keep                = keep && m_cache.m_ahead != nullptr;
		auto*      store    = m_cache.m_stage_store.get();
		if (store != nullptr) {
			if (auto bytes = store->Find(record.Key()); bytes.has_value()) {
				if (LoadStage(*bytes, record, keep, out)) return true;
				LOGF("SHADER_PRECOMPILE stage store: the record of %016" PRIx64 " did not load, translating it\n",
				     record.Key());
			}
		}
		const auto options  = StageCompileOptions(record.stage, record.shader_hash,
		                                          std::span(NoUserData).first(record.user_data_count), record.back_code,
		                                          input, m_cache.m_program_cache->profile, graphics.SplitsFragmentSet());
		auto source = ShaderRecompiler::PrepareShaderSource(record.code, options);
		if (source.unsupported_call_pc != UINT32_MAX || source.calls.size() != record.callees.size()) return false;
		ShaderRecompiler::TranslateResult translated;
		if (source.calls.empty()) {
			translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		} else {
			// As the GPU thread translates it (ProgramCache::Get): linked, then its reads for materialization. The
			// GPU thread keeps no ready module of a shader with calls: only the pipeline goes to the driver cache.
			translated = ShaderRecompiler::TranslateProgram(ShaderRecompiler::LinkShaderSource(source, record.callees),
			                                                options);
			translated.program.source_reads = source.reads;
		}
		std::vector<uint32_t> static_key;
		std::visit([&](const auto& info) { BuildStageStaticKey(info, static_key); }, out.input);
		// The GPU thread takes a shader with calls' plan and module by its linked code too (ProgramCache::Get); the
		// plan's reads are the stand-in addresses here and are replaced by the draw's on the take.
		if (!record.callees.empty()) AppendLinkedKey(static_key, source.linked->code);
		// The reuse store: the GPU thread's lookup of this permutation takes the plan and module (Ahead::ready).
		if (keep) m_cache.NoteReadyPlan(record.stage, record.shader_hash, static_key, translated.program);
		// The stage store (the launcher's precompile): the plan as the reuse store takes it. A plan that cannot be
		// extracted or stored leaves the stage to be translated again on start.
		std::vector<uint8_t> plan;
		if (store != nullptr && store->Writing()) {
			try {
				Common::ScopedExitThrows throws;
				plan = ShaderRecompiler::IR::SaveResourcePlan(ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
			} catch (const Common::ExitException& error) {
				LOGF("SHADER_PRECOMPILE stage store: no plan of %016" PRIx64 ": %s\n", record.Key(), error.message.c_str());
			}
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, record.specialization,
		                                               record.push_data_cursor);
		out.spirv_hash = XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t));
		out.module     = CompileSPV(result.spirv, graphics.device);
		if (out.module == nullptr) return false;
		out.info = std::move(result.program).TakeCompiledInfo();
		if (!plan.empty()) {
			ShaderRecompiler::IR::Writer writer;
			ShaderRecompiler::IR::IoAll(writer, record.stage, record.shader_hash, static_key, plan, result.spirv, out.info);
			if (writer.ok) store->Append(record.Key(), writer.bytes);
		}
		if (keep) {
			m_cache.NoteReady(record.stage, record.shader_hash, std::move(static_key), record.specialization, result.spirv,
			                  out.info);
		}
		std::visit([&](auto& info) { info.stage = {.program = &out.info, .resources = &out.resources}; }, out.input);
		return true;
	}

	// A stage the store holds: its module, and its plan and module for the reuse store when keep, as the translation
	// above makes them. False: a record of another stage, or damaged.
	bool LoadStage(std::span<const uint8_t> bytes, const Precompile::StageRecord& record, bool keep,
	               Precompile::StageModule& out) {
		namespace IR = ShaderRecompiler::IR;
		IR::Reader               reader {bytes};
		ShaderType               stage {};
		uint64_t                 hash = 0;
		std::vector<uint32_t>    static_key, spirv;
		std::vector<uint8_t>     plan_bytes;
		IR::CompiledShaderInfo   info;
		IR::IoAll(reader, stage, hash, static_key, plan_bytes, spirv, info);
		if (!reader.ok || !reader.bytes.empty() || stage != record.stage || hash != record.shader_hash || spirv.empty()) {
			return false;
		}
		std::unique_ptr<IR::ResourcePlan> plan;
		if (keep) {
			plan = std::make_unique<IR::ResourcePlan>();
			try {
				Common::ScopedExitThrows throws;
				if (!IR::LoadResourcePlan(plan_bytes, *plan)) return false;
			} catch (const Common::ExitException&) {
				return false;
			}
		}
		out.spirv_hash = XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t));
		out.module     = CompileSPV(spirv, m_cache.m_graphics.device);
		if (out.module == nullptr) return false;
		out.info = info;
		if (keep) {
			m_cache.NoteReadyPlan(stage, hash, static_key, std::move(plan));
			m_cache.NoteReady(stage, hash, std::move(static_key), record.specialization, spirv, info);
		}
		std::visit([&](auto& input) { input.stage = {.program = &out.info, .resources = &out.resources}; }, out.input);
		return true;
	}

	void DestroyStage(Precompile::StageModule& stage) override {
		if (stage.module != nullptr) m_cache.m_graphics.device.destroyShaderModule(stage.module, nullptr);
		stage.module = nullptr;
	}

	bool CreateGraphics(const Precompile::PipelineRecord& record, std::span<Precompile::StageModule* const> vertex,
	                    Precompile::StageModule* pixel, bool keep) override {
		std::array<ShaderVertexInputInfo, 3> vertex_info {};
		GraphicsPrograms                     programs;
		if (vertex.empty() || vertex.size() > vertex_info.size()) return false;
		for (size_t i = 0; i < vertex.size(); i++) {
			const auto* info = std::get_if<ShaderVertexInputInfo>(&vertex[i]->input);
			if (info == nullptr) return false;
			vertex_info[i]     = *info;
			programs.vertex[i] = {.id = i + 1u, .module = vertex[i]->module, .spirv_hash = vertex[i]->spirv_hash};
		}
		ShaderPixelInputInfo pixel_info {};
		if (pixel != nullptr) {
			const auto* info = std::get_if<ShaderPixelInputInfo>(&pixel->input);
			if (info == nullptr) return false;
			pixel_info     = *info;
			programs.pixel = {.id = 4u, .module = pixel->module, .spirv_hash = pixel->spirv_hash};
		}
		auto&    graphics = m_cache.m_graphics;
		Pipeline pipeline;
		CreatePipelineInternal(graphics, pipeline, record.rendering, record.vertex_input,
		                       std::span(vertex_info.data(), vertex.size()), pixel != nullptr ? &pixel_info : nullptr,
		                       programs, record.static_params, m_cache.m_driver_cache,
		                       graphics.SplitsFragmentSet() ? m_cache.m_libraries.get() : nullptr);
		if (!hand_over || !keep || pipeline.pipeline == nullptr) return Destroy(pipeline);
		ReadyPipeline ready {.pipeline = pipeline};
		for (size_t i = 0; i < vertex.size(); i++) ready.spirv[i] = vertex[i]->spirv_hash;
		if (pixel != nullptr) ready.spirv[3] = pixel->spirv_hash;
		return Keep(record.Key(), ready);
	}

	bool   hand_over = false; // keep created pipelines for the GPU thread (PipelineCache::m_ready_pipelines)
	size_t ready_max = ReadyPipelineMax; // of them at most, oldest out first

	// A compute pipeline is handed over as a graphics one is (spirv[0] its stage's): the GPU thread's own create of a
	// warm one was a driver-cache hit of ~2.7 ms, one after another (Marvel's Wolverine's boot: 108 in one frame).
	bool CreateCompute(uint64_t key, Precompile::StageModule& stage) override {
		const auto* info = std::get_if<ShaderComputeInputInfo>(&stage.input);
		if (info == nullptr) return false;
		Pipeline pipeline;
		CreatePipelineInternal(m_cache.m_graphics, pipeline, *info, stage.module, m_cache.m_driver_cache);
		if (!hand_over || pipeline.pipeline == nullptr) return Destroy(pipeline);
		ReadyPipeline ready {.pipeline = pipeline};
		ready.spirv[0] = stage.spirv_hash;
		return Keep(key, ready);
	}

private:
	// Into the GPU thread's ready map; past ready_max the oldest there are destroyed.
	bool Keep(uint64_t key, ReadyPipeline ready) {
		std::lock_guard lock(m_cache.m_ready_mutex);
		if (!m_cache.m_ready_pipelines.try_emplace(key, ready).second) return Destroy(ready.pipeline);
		m_cache.m_ready_order.push_back(key);
		Precompile::TrimOldest(m_cache.m_ready_pipelines, m_cache.m_ready_order, ready_max,
		                       [this](ReadyPipeline& old) { Destroy(old.pipeline); });
		return true;
	}

	bool Destroy(Pipeline& pipeline) {
		auto&      device  = m_cache.m_graphics.device;
		const bool created = pipeline.pipeline != nullptr;
		device.destroyPipeline(pipeline.pipeline, nullptr);
		device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
		device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
		device.destroyDescriptorSetLayout(pipeline.fragment_set_layout, nullptr);
		for (const auto layout: pipeline.group_set_layouts) device.destroyDescriptorSetLayout(layout, nullptr);
		return created;
	}

	PipelineCache& m_cache;
};

void PipelineCache::InitializePrecompile() {
	m_precompile_compiler = std::make_unique<PrecompileCompiler>(*this);
	const auto title_id   = PipelineCacheTitleId();
	if (title_id.empty()) return;
	const auto base = std::filesystem::path("_PipelineCache") / title_id;
	const auto with = [&](const char* suffix) {
		auto path = base;
		path += suffix;
		return path;
	};
	m_journal_path  = with(".journal");
	m_warm_path     = with(".warm");
	m_inflight_path = with(".inflight");
	m_suspects_path = with(".suspects");
	m_hints_path    = with(".hints");
	const auto& properties = m_graphics.GetPhysicalDeviceProperties();
	if (const auto header = StageStoreHeader(properties, m_program_cache->profile, m_graphics.SplitsFragmentSet());
	    !header.empty()) {
		m_stage_store = std::make_unique<Precompile::StageStore>();
		if (m_stage_store->Open(with(".stages"), header, Config::PrecompileOnly())) {
			PipelineCacheLog("SHADER_PRECOMPILE stage store: {} stages, {:.1f} MB{}", m_stage_store->Size(),
			                 static_cast<double>(m_stage_store->Bytes()) / 1'000'000.0,
			                 m_stage_store->Writing() ? ", written by this precompile" : "");
		} else {
			m_stage_store.reset();
		}
	}
	if (m_driver_cache != nullptr) {
		m_warm = std::make_unique<Precompile::WarmList>();
		m_warm->Load(m_warm_path, DriverCacheSignature(properties), m_driver_cache_loaded_bytes);
	}
	// RunPrecompileOnly reads the journal itself, and never appends to it. KYTY_PRECOMPILE=0 (the launcher's "Precompile
	// on start" off) leaves both the precompile on start and the recording off.
	const char* enabled = std::getenv("KYTY_PRECOMPILE");
	if (Config::PrecompileOnly() || (enabled != nullptr && enabled[0] == '0')) return;
	const auto device  = Precompile::DeviceFingerprint(properties.vendorID, properties.deviceID);
	auto       journal = Precompile::ReadJournal(m_journal_path, device);
	if (journal.foreign) {
		// Another GPU's or layout's journal: kept aside, and this one starts over.
		auto old = m_journal_path;
		old += ".old";
		std::error_code error;
		std::filesystem::rename(m_journal_path, old, error);
		journal = {};
	}
	const auto loaded = journal.stages.size() + journal.pipelines.size();
	// The shipped hints, on the first start after they appear or change: without them a first session's precompile on
	// start has nothing to compile unless the launcher's Precompile ran (one import reads ~1.5 GB of game files, ~1.5 s
	// warm). The records this PC has come first, as in RunPrecompileOnly.
	if (const auto hints = Precompile::ShippedHints(title_id); !hints.empty()) {
		std::string done;
		if (std::ifstream stamp(m_hints_path); stamp) std::getline(stamp, done);
		if (done != Precompile::HintsStamp(hints)) {
			auto result = ImportHintsFile(hints);
			PipelineCacheLog("SHADER_PRECOMPILE hints {}: {}", Common::PathToString(hints),
			                 result.ok ? fmt::format("{} stages, {} pipelines, {} missing, {:.1f} s", result.stages, result.pipelines,
			                                         result.missing, result.seconds)
			                           : "not imported (" + result.reason + ")");
			std::ranges::move(result.journal.stages, std::back_inserter(journal.stages));
			std::ranges::move(result.journal.pipelines, std::back_inserter(journal.pipelines));
		}
	}
	const bool migrated = Precompile::MigrateBufferTable(journal); // after the hints, recorded in the normal variant
	journal.Dedupe();
	const bool rewrite = migrated || !journal.found || journal.torn || journal.rejected != 0 ||
	                     journal.stages.size() + journal.pipelines.size() != loaded;
	m_journal = std::make_unique<Precompile::JournalWriter>();
	if (!m_journal->Open(m_journal_path, device, journal, rewrite)) {
		PipelineCacheLog("SHADER_PRECOMPILE journal: cannot write {}", Common::PathToString(m_journal_path));
		m_journal.reset();
	}
	m_program_cache->journal = m_journal.get();
	for (const auto& record: journal.stages) m_program_cache->recorded.insert(record.Key());
	for (const auto& record: journal.pipelines) m_recorded_pipelines.insert(record.Key());
	PipelineCacheLog("SHADER_PRECOMPILE journal: {} stages, {} pipelines{}{}{}", journal.stages.size(), journal.pipelines.size(),
	                 journal.torn ? ", torn tail dropped" : "",
	                 journal.rejected != 0 ? fmt::format(", {} damaged records dropped", journal.rejected) : "",
	                 m_warm != nullptr ? fmt::format(", {} warm", m_warm->Size()) : "");
	if (journal.stages.empty()) return;
	Precompile::Options options;
	// shortcut: a quarter of the threads, 1-6, below normal priority: the game keeps its cores.
	options.threads = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 6u);
	if (const char* threads = std::getenv("KYTY_PRECOMPILE_THREADS"); threads != nullptr) {
		options.threads = std::clamp(static_cast<uint32_t>(std::strtoul(threads, nullptr, 10)), 1u, 64u);
	}
	const char* reuse = std::getenv("KYTY_PRECOMPILE_REUSE");
	options.reuse     = reuse == nullptr || reuse[0] != '0';
	// The GPU thread takes the pipelines the workers created (driver-cache hits for warm ones) instead of creating each
	// again at its first draw: a scene cut's burst of new pipelines cost it 150-200 ms a frame (session 14).
	const char* hand_over = std::getenv("KYTY_PRECOMPILE_HAND_OVER");
	options.hand_over     = options.reuse && (hand_over == nullptr || hand_over[0] != '0');
	m_precompile_compiler->hand_over = options.hand_over;
	if (const char* ready_max = std::getenv("KYTY_PRECOMPILE_READY_MAX"); ready_max != nullptr) {
		const auto value = std::strtoull(ready_max, nullptr, 10);
		m_precompile_compiler->ready_max = value == 0 ? std::numeric_limits<size_t>::max() : static_cast<size_t>(value);
	}
	options.inflight  = m_inflight_path;
	options.suspects  = Precompile::LoadSuspects(m_inflight_path, m_suspects_path);
	options.pair_vertex_shaders = PairedVertexShaders(title_id);
	options.predict_volume_pixels = PredictsVolumePixels(title_id);
	Precompile::RegisterInflightCleanup(m_inflight_path);
	m_precompiler = std::make_unique<Precompile::Precompiler>(*m_precompile_compiler, std::move(options), std::move(journal),
	                                                          m_warm.get());
	m_precompiler->on_created = [this](uint64_t key, bool graphics) {
		if (m_warm != nullptr) m_warm->Mark(key, graphics);
		m_driver_cache_dirty = true;
	};
	const auto& progress = m_precompiler->GetProgress();
	PipelineCacheLog("SHADER_PRECOMPILE on start: {} jobs ({} compute, {} graphics), {} warm, {} suspects", progress.total.load(),
	                 progress.compute.load(), progress.graphics.load(), progress.warm.load(), progress.suspects.load());
}

// A recorded draw without its pixel shader, for Phase B's seeds (shaderPrecompile.h, TemplateRecord).
Precompile::TemplateRecord PipelineCache::MakeTemplate(CommandBuffer& command, std::span<const RenderColorInfo> colors,
                                                       const RenderDepthInfo& depth, vk::PrimitiveTopology topology,
                                                       bool primitive_restart, const Precompile::PipelineRecord& record) {
	const auto bytes = [](const auto& block) {
		static_assert(std::is_trivially_copyable_v<std::decay_t<decltype(block)>>);
		const auto* begin = reinterpret_cast<const uint8_t*>(&block);
		return std::vector<uint8_t>(begin, begin + sizeof(block));
	};
	const auto&                ctx = command.GetRegisters();
	Precompile::TemplateRecord out {.pixel = record.pixel, .pipeline = record.Key()};
	out.context     = bytes(ctx);
	out.user_config = bytes(command.GetUserConfig());
	out.pixel_regs  = bytes(command.GetShaders().GetPs());
	std::array<Prospero::ColorComponentMapping, 8> mapping {};
	(void)DrawPixelExports(ctx, command.GetShaders(), mapping);
	for (size_t i = 0; i < mapping.size(); i++) out.export_mapping[i] = mapping[i].packed;
	for (const auto& color: colors) {
		out.colors.push_back({.slot         = color.target_slot,
		                      .format       = static_cast<uint32_t>(color.desc.view_info.format),
		                      .guest_format = static_cast<uint32_t>(color.desc.info.guest_format),
		                      .samples      = color.desc.info.samples,
		                      .mapping      = color.export_mapping.packed});
	}
	out.depth_bound       = static_cast<bool>(depth.image_id);
	out.depth_format      = static_cast<uint32_t>(depth.desc.view_info.format);
	out.depth_samples     = depth.desc.info.samples;
	out.depth_test        = depth.depth_test_enable;
	out.topology          = static_cast<uint32_t>(topology);
	out.primitive_restart = primitive_restart;
	return out;
}

// The renderer is complete once the GPU thread asks for programs: the workers start then.
void PipelineCache::StartPrecompile() {
	m_precompile_started = true;
	if (m_precompiler != nullptr) m_precompiler->Start();
}

// A new graphics pipeline's record: its stages' record keys and its state; none when a stage has no record.
std::optional<Precompile::PipelineRecord> PipelineCache::MakePipelineRecord(const GraphicsPipelineKey& key,
                                                                            const GraphicsPrograms& programs,
                                                                            bool ps_active) const {
	if (m_journal == nullptr && m_precompiler == nullptr && m_warm == nullptr) return std::nullopt;
	const auto&                of = m_program_cache->record_of;
	Precompile::PipelineRecord record {.rendering = key.rendering, .vertex_input = key.vertex_input,
	                                   .static_params = key.static_params};
	for (uint32_t i = 0; i < programs.VertexStageCount(); i++) {
		const auto found = of.find(programs.vertex[i].id);
		if (found == of.end()) return std::nullopt;
		record.vertex[i] = found->second;
	}
	if (ps_active) {
		const auto found = of.find(programs.pixel.id);
		if (found == of.end()) return std::nullopt;
		record.pixel = found->second;
	}
	return record;
}

Precompile::ImportResult PipelineCache::ImportHintsFile(const std::filesystem::path&                         path,
                                                       const std::function<void(uint64_t done, uint64_t total)>& progress) {
	Precompile::HintsInfo info {.title_id = PipelineCacheTitleId()};
	Loader::SystemContentParamSfoGetString("APP_VER", &info.app_version);
	// An imported record's host fields as this GPU sets them (GetComputeProgram, PrepareVertexStages).
	const auto& limits = m_graphics.GetPhysicalDeviceProperties().limits;
	const auto  adjust = [&](Precompile::StageRecord& record) {
		if (auto* compute = std::get_if<ShaderComputeInputInfo>(&record.input)) {
			compute->host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
			compute->lds_storage        = compute->lds_size_dwords * 4u > limits.maxComputeSharedMemorySize;
		} else if (auto* vertex = std::get_if<ShaderVertexInputInfo>(&record.input)) {
			if (vertex->logical_stage == ShaderType::Mesh) vertex->mesh.host_subgroup_size = m_graphics.subgroup_size;
			if (vertex->clip_space.enabled) {
				vertex->clip_space.half_extent[0] = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
				vertex->clip_space.half_extent[1] = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
			}
		}
	};
	auto result = Precompile::ImportHints(path, Config::GetGameDir(), info, adjust, progress);
	if (result.ok && !m_hints_path.empty()) {
		std::ofstream(m_hints_path, std::ios::trunc) << Precompile::HintsStamp(path) << '\n';
	}
	return result;
}

void PipelineCache::AheadSubmission(uint64_t seq) {
	t_ahead_submission = seq;
}

int PipelineCache::RunPrecompileOnly() {
	const auto line = [](const std::string& text) {
		std::fputs(text.c_str(), stdout);
		std::fputc('\n', stdout);
		std::fflush(stdout);
		LOGF("%s\n", text.c_str());
	};
	if (m_journal_path.empty() || m_driver_cache == nullptr) {
		line("PRECOMPILE_DONE status=unavailable");
		return 2;
	}
	const auto& properties = m_graphics.GetPhysicalDeviceProperties();
	const auto  device     = Precompile::DeviceFingerprint(properties.vendorID, properties.deviceID);
	Precompile::HintsInfo info {.title_id = PipelineCacheTitleId()};
	Loader::SystemContentParamSfoGetString("APP_VER", &info.app_version);

	if (const auto& path = Config::GetPrecompileExportHints(); !path.empty()) {
		auto journal = Precompile::ReadJournal(m_journal_path, 0);
		journal.Dedupe();
		const auto result = Precompile::ExportHints(journal, info, path);
		line(fmt::format("PRECOMPILE_HINTS kind=export status={} stages={} pipelines={} no_footer={}", result.ok ? "ok" : "failed",
		                 result.stages, result.pipelines, result.no_footer));
		line(fmt::format("PRECOMPILE_DONE status={}", result.ok ? "exported" : "export_failed"));
		return result.ok ? 0 : 5;
	}

	if (const auto& path = Config::GetPrecompileImportHints(); !path.empty()) {
		auto last   = std::chrono::steady_clock::now();
		auto result = ImportHintsFile(path, [&](uint64_t done, uint64_t total) {
			const auto now = std::chrono::steady_clock::now();
			if (now - last < std::chrono::seconds(1)) return;
			last = now;
			line(fmt::format("PRECOMPILE_HINTS_PROGRESS read_mb={} total_mb={}", done >> 20u, total >> 20u));
		});
		if (!result.ok) {
			line(fmt::format("PRECOMPILE_HINTS kind=import status=failed reason={}", result.reason));
			line("PRECOMPILE_DONE status=import_failed");
			return 6;
		}
		auto journal = Precompile::ReadJournal(m_journal_path, device);
		if (journal.foreign) journal = {};
		// The records this PC has first, then the imported ones it lacks.
		std::ranges::move(result.journal.stages, std::back_inserter(journal.stages));
		std::ranges::move(result.journal.pipelines, std::back_inserter(journal.pipelines));
		journal.Dedupe();
		if (!Precompile::WriteJournal(m_journal_path, device, journal)) {
			line("PRECOMPILE_HINTS kind=import status=failed reason=unwritable");
			line("PRECOMPILE_DONE status=import_failed");
			return 6;
		}
		line(fmt::format("PRECOMPILE_HINTS kind=import status=ok stages={} pipelines={} missing={} read_mb={} seconds={:.1f}",
		                 result.stages, result.pipelines, result.missing, result.bytes_read >> 20u, result.seconds));
	}

	auto journal = Precompile::ReadJournal(m_journal_path, device);
	if (journal.foreign) journal = {};
	Precompile::MigrateBufferTable(journal);
	journal.Dedupe();
	Precompile::Options options;
	options.threads  = Config::GetPrecompileThreads() != 0 ? Config::GetPrecompileThreads()
	                                                       : std::max(1u, std::thread::hardware_concurrency());
	options.only     = true;
	options.inflight = m_inflight_path;
	options.suspects = Precompile::LoadSuspects(m_inflight_path, m_suspects_path);
	options.pair_vertex_shaders = PairedVertexShaders(info.title_id);
	options.predict_volume_pixels = PredictsVolumePixels(info.title_id);
	Precompile::RegisterInflightCleanup(m_inflight_path);
	const auto threads = options.threads;
	Precompile::Precompiler precompiler(*m_precompile_compiler, std::move(options), std::move(journal), m_warm.get());
	precompiler.on_created = [this](uint64_t key, bool graphics) {
		if (m_warm != nullptr) m_warm->Mark(key, graphics);
		m_driver_cache_dirty = true;
	};
	const auto& progress = precompiler.GetProgress();
	line(fmt::format("PRECOMPILE_START total={} compute={} graphics={} cached={} suspects={} threads={}", progress.total.load(),
	                 progress.compute.load(), progress.graphics.load(), progress.warm.load(), progress.suspects.load(),
	                 std::min<uint64_t>(threads, std::max<uint64_t>(progress.total.load(), 1u))));
	// The launcher stops a precompile with "quit" on stdin, never by killing it: a killed job becomes a suspect.
	static std::atomic<bool> quit = false;
	std::thread([] {
		for (std::string command; std::getline(std::cin, command);) {
			if (command == "quit" || command == "stop") {
				quit = true;
				return;
			}
		}
	}).detach();
	const auto start     = std::chrono::steady_clock::now();
	auto       next_save = start + std::chrono::minutes(6);
	const auto elapsed   = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
	const auto report    = [&](const char* phase) {
		const auto done  = progress.finished.load();
		const auto total = progress.total.load();
		const auto eta   = done == 0 ? 0.0 : elapsed() * static_cast<double>(total - done) / static_cast<double>(done);
		line(fmt::format("PRECOMPILE_PROGRESS phase={} done={} total={} compute={}/{} graphics={}/{} compiled={} threads={} "
		                 "elapsed_s={:.0f} eta_s={:.0f}",
		                 phase, done, total, progress.compute_finished.load(), progress.compute.load(),
		                 progress.graphics_finished.load(), progress.graphics.load(), progress.compiled.load(),
		                 progress.threads.load(), elapsed(), eta));
	};
	precompiler.Start();
	auto next_report = start;
	while (!precompiler.Finished()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		const auto now = std::chrono::steady_clock::now();
		if (quit) {
			report("stopping");
			precompiler.Stop();
			break;
		}
		if (now >= next_report) {
			report("compile");
			next_report = now + std::chrono::seconds(1);
		}
		// A killed run loses little: the work done is saved every 6 minutes.
		if (now >= next_save) {
			Save();
			line(fmt::format("PRECOMPILE_SAVE kind=periodic elapsed_s={:.0f}", elapsed()));
			next_save = now + std::chrono::minutes(6);
		}
	}
	precompiler.Stop();
	report("save");
	Save();
	const bool complete = progress.done.load();
	if (m_stage_store != nullptr) {
		line(fmt::format("PRECOMPILE_STAGE_STORE stages={} mb={:.1f}", m_stage_store->Size(),
		                 static_cast<double>(m_stage_store->Bytes()) / 1'000'000.0));
	}
	line(fmt::format("PRECOMPILE_DONE status={} done={} total={} compiled={} failed={} mismatches={} seconds={:.1f}",
	                 complete ? "complete" : "stopped", progress.finished.load(), progress.total.load(),
	                 progress.compiled.load(), progress.failed.load(), progress.mismatches.load(), elapsed()));
	if (m_driver_cache_save_failed) return 4;
	return complete ? 0 : 3;
}

// The pixel stage as a draw prepares it, from registers only: GetGraphicsPrograms and graphics
// ahead share it, so the two build the same shader.
static ShaderParams PreparePixelStage(const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
                                      const HW::Context& context, const HW::UserConfig& user_config,
                                      std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
                                      bool mesh_active, bool tess_active, bool object_primitive_id,
                                      ShaderPixelInputInfo& pixel_info) {
	auto pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
	pixel_info.object_primitive_id = object_primitive_id;
	for (uint32_t export_index = 0; export_index < 8u; export_index++) {
		const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
		if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
		const auto& rt = context.GetRenderTarget(slot);
		const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
		if (rt.base.addr == 0 || target_mask == 0u) continue;
		if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
			pixel_info.target_uint_mask |= 1u << export_index;
		} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
			pixel_info.target_sint_mask |= 1u << export_index;
		}
		const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
		if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
		const auto mapping = pixel_info.target_export_mapping[export_index];
		const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
		const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
		if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
		    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
		    pixel_info.target_output_mode[export_index] != 9u) {
			EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
			     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
			     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
			     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
		}
		pixel_info.target_packed_unorm_mask |= 1u << export_index;
		pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
	}
	if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
		pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
	} else if (context.GetModeControl().provoking_vtx_last) {
		pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
	} else {
		pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
	}
	const auto& blend = context.GetBlendControl(0);
	pixel_info.dual_source_blending =
	    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
	    (BlendFactorIsDualSource(blend.color_srcblend) ||
	     BlendFactorIsDualSource(blend.color_destblend) ||
	     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
	                                     BlendFactorIsDualSource(blend.alpha_destblend))));
	if (pixel_info.dual_source_blending) {
		// MRT1 supplies the second blend source for target 0; blended targets are not integer.
		pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
		pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		pixel_info.target_uint_mask &= ~2u;
		pixel_info.target_sint_mask &= ~2u;
	} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
	           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
	           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
	                       std::end(pixel_info.target_output_mode),
	                       [](uint8_t mode) { return mode == 0; })) {
		switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
			case BlendMappingSupport::SourceAlpha:
				pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
				break;
			case BlendMappingSupport::SourceAlphaOne:
				pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
				break;
			case BlendMappingSupport::SourceAlphaZero:
				pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
				break;
			default: break;
		}
		if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
			pixel_info.dual_source_blending     = true;
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = {};
		}
	}
	return pixel_params;
}

// The vertex stages as a draw prepares them, from registers only: GetGraphicsPrograms and graphics
// ahead's exact builds share it, so the two key the same vertex programs. Object primitive ID and the
// push data cursor after the draw's own block come back through the last two.
std::array<ShaderParams, 3> PipelineCache::PrepareVertexStages(
    const HW::VertexShaderInfo& vertex_regs, const HW::ShaderRegisters& sh, const HW::Context& context,
    const HW::UserConfig& user_config, bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
    bool& object_primitive_id, uint32_t& push_data_cursor, ShaderIndirectOffsets indirect_offsets,
    bool indirect_draw, bool* exceeds_host) const {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (!tess_active && !mesh_active) {
		vertex_info[0].indirect_offsets = indirect_offsets;
		vertex_info[0].indirect_draw    = indirect_draw;
	}
	object_primitive_id = context.HasObjectPrimitiveIdOutput();
	if (object_primitive_id) {
		// PA_CL_OBJPRIM_ID_CNTL = 2: VGT_OBJECT_ID plus the pipelined 32-bit primitive ID.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto lanes        = std::min(mesh.host_subgroup_size, mesh.wave_size);
		auto       host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) * lanes;
		// shortcut: a subgroup with more threads than a host mesh workgroup has runs only the
		// waves that fit, and the primitives of the other waves are missing. Run it in passes.
		if (const auto most = std::min(limits.maxMeshWorkGroupInvocations,
		                               limits.maxMeshWorkGroupSize[0]);
		    host_threads > most && most >= lanes && mesh.threads_num[1] == 1u &&
		    mesh.threads_num[2] == 1u) {
			static std::atomic_flag logged = ATOMIC_FLAG_INIT;
			if (!logged.test_and_set()) {
				LOGF("Mesh shader: a subgroup of %u threads runs %u of them (host limit)\n",
				     host_threads, most / lanes * lanes);
			}
			mesh.threads_num[0] = most / lanes * mesh.wave_size;
			host_threads        = most / lanes * lanes;
			// Thread i exports vertex i and primitive i: declare no more than the running threads
			// write, or the rest are drawn from undefined outputs (planes across the view).
			mesh.max_vertices   = std::min(mesh.max_vertices, mesh.threads_num[0]);
			mesh.max_primitives = std::min(mesh.max_primitives, mesh.threads_num[0]);
		}
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT_IF(exceeds_host == nullptr);
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: dropping mesh draws the host does not support (shader 0x{:016x}).\n",
				    vertex_params[0].hash));
			}
			LOGF("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n", host_threads,
			     mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
			*exceeds_host = true;
			return {};
		}
		// Upstream abc509d33: a task shader launches each indirect record's groups and passes its draw parameters in a
		// payload beside the LDS, the allocation pair, the Layer array and a wave exchange pair; without room for it,
		// or with the switch off, the CPU reads the records.
		if (indirect_draw && m_graphics.task_shader_enabled &&
		    (mesh.lds_size_dwords + 4u + mesh.max_vertices + ShaderRecompiler::IR::PushData::MeshTaskPayloadDwordCount) *
		            sizeof(uint32_t) <=
		        limits.maxMeshPayloadAndSharedMemorySize) {
			vertex_info[0].indirect_offsets = indirect_offsets;
			vertex_info[0].indirect_draw    = true;
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		// CLIP_DISABLE also disables user planes. How they combine with shader clip and cull
		// distances is unverified; draw push constants only reach VS/mesh and PS, and the
		// generated RECTLIST stages do not forward clip distances.
		EXIT_NOT_IMPLEMENTED((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		                     Prospero::IsRectList(user_config.GetPrimType()));
		last_stage.user_clip_planes = clip_control.user_clip_planes;
	}
	push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id, last_stage.user_clip_planes);
	return vertex_params;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    ShaderIndirectOffsets indirect_offsets, bool indirect_draw) {
	if (!m_precompile_started) StartPrecompile();
	bool       object_primitive_id = false;
	uint32_t   push_data_cursor    = 0;
	bool       exceeds_host        = false;
	const auto vertex_params       = PrepareVertexStages(vertex_regs, sh, context, user_config, pixel_active,
	                                                     vertex_info, object_primitive_id, push_data_cursor,
	                                                     indirect_offsets, indirect_draw, &exceeds_host);
	// A mesh shader past the host's limits: no programs, and the draw is dropped (PrepareDrawRenderState).
	if (exceeds_host) return {};
	const bool tess_active         = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	const bool mesh_active         = vertex_info[0].logical_stage == ShaderType::Mesh;
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PreparePixelStage(pixel_regs, sh, context, user_config, target_export_mapping, mesh_active,
		                                 tess_active, object_primitive_id, pixel_info);
	}
	GraphicsPrograms  result;
	if (pixel_active) {
		// Past the largest draw block whatever the draw: a pixel shader under a mesh and under a
		// vertex shader is one build, not two.
		uint32_t pixel_cursor = ShaderRecompiler::IR::PushData::MeshDrawDwordCount;
		result.pixel          = m_program_cache->Get(pixel_params, pixel_info, pixel_cursor);
		if (pixel_cursor != ShaderRecompiler::IR::PushData::MeshDrawDwordCount) push_data_cursor = pixel_cursor;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

// shortcut: ray tracing is left out, its dispatches do nothing (Senaxx, 2026-10-13: a first
// session spends minutes compiling these shaders, and on wolverine-lean the device was lost right
// after them). Run them again when that cost and the loss are dealt with.
// KYTY_RT=<list> (Senaxx, 2026-11-11: ray tracing staged, this switch off by default) runs the listed ray-tracing
// passes instead of skipping them. Names: shadows, ao, refit, weather, ssgi, probes, restir, or "all"; or 16-digit
// program hashes. Empty or unset: every ray-tracing shader is skipped as before. The traversal is the emulated one
// (image_bvh_intersect_ray in SPIR-V). A pass that calls hit shaders through a GPU-read table still stops at that call.
// KYTY_RT=ssgi,weather,refit runs PS5 Pro mode's five that translate (Senaxx, 2026-11-13: "Merge, off by default"): it
// lights rooms the skipped SSGI left near-black, at about half the frame rate there (place 96, mostly the refits).
static bool RayTracingAllowed(uint64_t hash) {
	// Program hashes of Marvel's Wolverine 1.05 (side-session-notes/56-raytracing.md); trinity (PS5 Pro) ones last.
	static constexpr std::pair<std::string_view, std::array<uint64_t, 4>> Named[] = {
	    // CS_RtShadowTraceRaysBase, CS_RayTracingShadows, CS_RtShadowTraceRaysTrinity
	    {"shadows", {0xcd358e2ccaa62046ull, 0x1786217adb98a31bull, 0x6b02ce8531cbf723ull}},
	    // CS_SsaoGenerateWithRayTracing, its trinity form
	    {"ao", {0xc00ed51713451ceaull, 0x90dc7d97cd755004ull}},
	    // sce::Psr psr_qbvh_refit_bottom_level, _fan_refit_copy_dispatch_args, _fan_refit_recover_aabbs,
	    // _fan_refit_async_shader_mesh_0
	    {"refit", {0xfb3ca6c727da6121ull, 0x0f28200e45df66dbull, 0xd5009f68789143e6ull, 0xa19405f3da6f63beull}},
	    // psr_qbvh_refit_bottom_level_trinity, psr_qbvh_tri_pair_refit_async_shader_mesh_0 (PS5 Pro mode)
	    {"refit", {0x9e63fd0fcf1cf261ull, 0xc0e04a22ed3c93abull}},
	    // CS_WeatherParticles, CS_WeatherGSplashProcess
	    {"weather", {0xb790bfb6f776e13aull, 0x166bae05919556b1ull}},
	    // CS_WeatherParticlesTrinity, CS_WeatherGSplashProcessTrinity (PS5 Pro mode)
	    {"weather", {0x33cf9c10fe1a4032ull, 0x5e1d069f2929d767ull}},
	    // CS_SsgiGenerateWithRayTracingTrinity (PS5 Pro mode)
	    {"ssgi", {0x1506742f9bc5e781ull}},
	    // CS_DynamicLgDispatchRaysBase
	    {"probes", {0x3b54f880ca406b01ull}},
	    // CS_GiReferenceReStirSpatialBase
	    {"restir", {0x32a0830adfe4cab2ull}},
	};
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_RT");
		return std::string(value != nullptr ? value : "");
	}();
	if (list.empty()) return false;
	const auto listed = [&](std::string_view name) {
		for (size_t begin = 0; begin <= list.size();) {
			const size_t end = std::min(list.find(',', begin), list.size());
			if (std::string_view(list).substr(begin, end - begin) == name) return true;
			begin = end + 1;
		}
		return false;
	};
	if (listed("all") || listed(fmt::format("{:016x}", hash))) return true;
	for (const auto& [name, hashes]: Named) {
		if (std::ranges::find(hashes, hash) != hashes.end() && hash != 0 && listed(name)) return true;
	}
	return false;
}

// A ray-tracing shader is one that intersects rays (image_bvh_intersect_ray and its eight-wide
// form), or one of Marvel's Wolverine's four passes that only prepare or resolve for those.
static bool IsRayTracingShader(const ShaderParams& params) {
	// Program hashes (code and literal buffer, ShaderMapUserData); before upstream ae67e2db, the
	// hashes of the whole binaries were bad108e74fb72e9f, 4e7f2c6bb9b158a1, 8bfd230b9cd875a2 and
	// c4df2a00067e0666.
	static constexpr std::array<uint64_t, 4> Passes {0xfb3ca6c727da6121ull, 0x0f28200e45df66dbull,
	                                                 0xd5009f68789143e6ull, 0xa19405f3da6f63beull};
	// PS5 Pro mode's two, its programs have other hashes: 9e63fd0f (min and max of floats, about 650
	// dispatches a frame, 33 of 51 GPU ms) and c0e04a22, as many, which nothing but the intersecting
	// shaders left out above could use.
	static constexpr std::array<uint64_t, 2> ProPasses {0x9e63fd0fcf1cf261ull, 0xc0e04a22ed3c93abull};
	// The GPU thread and the ahead workers both ask.
	static std::mutex                         mutex;
	static std::unordered_map<uint64_t, bool> known;
	if (std::scoped_lock lock {mutex}; known.contains(params.hash)) {
		return known[params.hash];
	}
	// Decoded without the lock: reading the code can fault, and the fault's read back waits for the
	// GPU thread, which then must not be waiting for this lock (a freeze after the logos).
	using namespace ShaderRecompiler::Decoder;
	bool ray_tracing = std::ranges::find(Passes, params.hash) != Passes.end() ||
	                   (Loader::SystemContentIsTrinityMode() &&
	                    std::ranges::find(ProPasses, params.hash) != ProPasses.end());
	for (uint32_t word = 0; !ray_tracing && word < params.code.size();) {
		Instruction inst;
		DecodeInstruction(params.code, word, inst);
		ray_tracing = inst.family == Family::MIMG && inst.opcode_id >= 0xe5u && inst.opcode_id <= 0xe7u;
		if (inst.opcode == Opcode::S_ENDPGM || inst.word_count == 0u) break;
		word += inst.word_count;
	}
	const bool allowed = ray_tracing && RayTracingAllowed(params.hash);
	ray_tracing &= !allowed;
	std::scoped_lock lock {mutex};
	if (known.try_emplace(params.hash, ray_tracing).second && (ray_tracing || allowed)) {
		LOGF("Ray tracing: compute shader 0x%016" PRIx64 " %s\n", params.hash,
		     allowed ? "runs (KYTY_RT)" : "is skipped");
	}
	return ray_tracing;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	if (!m_precompile_started) StartPrecompile();
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	if (IsRayTracingShader(params)) return {};
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineCache::SkipsComputeShader(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh) {
	ShaderComputeInputInfo input_info {};
	return regs.cs_regs.data_addr != 0 && IsRayTracingShader(PrepareProgram(regs, sh, input_info));
}

// Compute ahead (ComputeAheadScanner; KYTY_COMPUTE_AHEAD=0 turns it off). A shader is translated once
// per program key, and every dispatch scanned is walked for its specialization; a new one goes to a
// worker, which emits the SPIR-V and builds the pipeline into the driver cache. The GPU thread's own
// build of the same SPIR-V is then a driver cache hit, and it waits for a worker already building the
// same pipeline instead of compiling it a second time; one still queued it drops (NoteGpuBuild). The
// walk reads plain guest memory only: a guess from data the GPU has yet to write builds a pipeline
// nobody uses, nothing else.
//
// The scanner's thread does not translate: a draw or dispatch whose programs are translated is prepared there (its
// specializations and the dedupe, next to nothing), and one with a program still to translate is handed over to a
// worker with a copy of its registers (the prepare stage, PrepareGraphicsAhead and PrepareComputeAhead). A burst of
// new shaders is then translated on the workers side by side, and none of them holds up the draws after it. The
// worker reads guest memory later than the scanner would have, but still plain memory only, as above.

// A draw's or dispatch's registers as the scanner had them, for a worker's prepare stage.
struct PipelineCache::AheadRegisters {
	HW::VertexShaderInfo                           vertex_regs;
	HW::PixelShaderInfo                            pixel_regs;
	HW::ComputeShaderInfo                          compute_regs;
	HW::ShaderRegisters                            sh;
	HW::UserConfig                                 user_config;
	std::array<Prospero::ColorComponentMapping, 8> mapping {};
	ShaderIndirectOffsets                          indirect_offsets; // a draw's: PrepareVertexStages
	bool                                           indirect_draw = false;
};

enum class PipelineCache::AheadStep : uint8_t {
	Build,    // a worker's prepare stage: build the job it made, on this worker
	Done,     // nothing more here: queued from the scanner's thread, a duplicate, or not built ahead
	Parked,   // the handed job waits for another worker's translation of its program
	HandOver, // the scanner's thread: a program is still to be translated, the draw goes to a worker
};

struct PipelineCache::AheadJob {
	ShaderParams                                       params;
	ShaderComputeInputInfo                             input_info;
	ShaderRecompiler::IR::ResourceSpecialization       specialization;
	std::shared_ptr<ShaderRecompiler::TranslateResult> translated;
	// A pixel shader: built with each template of the vertex shader it was drawn with.
	bool                                               pixel = false;
	ShaderPixelInputInfo                               pixel_info;
	uint64_t                                           vertex_hash = 0;
	uint64_t                                           state_signature = 0;
	// Its worker's pipelines, pixel stage left out, once chosen, and how many are built (WaitForGraphicsAhead).
	bool                                               chosen = false;
	std::vector<GraphicsPipelineKey>                   builds;
	size_t                                             built = 0;
	// An exact build (PrepareExactVertex): the draw's own vertex program and the registers the rest of its key
	// comes from; exact_reason says why a job has none and is built with the guesses.
	uint32_t                                           exact_reason = 0;
	ShaderParams                                       vertex_params;
	ShaderVertexInputInfo                              vertex_info; // as the scanner prepared it, no stage
	std::shared_ptr<ShaderRecompiler::TranslateResult> vertex_translated; // until the worker compiles it
	std::shared_ptr<ShaderRecompiler::TranslateResult>* vertex_program_translated = nullptr; // its Ahead::Program's
	std::vector<uint32_t>                              vertex_static_key;
	ShaderRecompiler::IR::ResourceSpecialization       vertex_specialization;
	PipelineVertexInputState                           vertex_input;
	uint32_t                                           vertex_push_cursor = 0; // after the draw's own block
	bool                                               mesh               = false;
	bool                                               rect_list          = false;
	vk::PrimitiveTopology                              topology           = vk::PrimitiveTopology::ePointList;
	std::shared_ptr<const HW::Context>                 context;
	// The exact draw or dispatch it builds (GraphicsAheadIdentity, ComputeAheadIdentity); 0 while it is unknown, and for
	// a job built with the guesses.
	uint64_t                                           identity = 0;
	// A draw or dispatch the scanner handed over: prepared by the worker that takes it, and built there. params.hash
	// (the pixel or compute shader) and pixel are set; context too for a draw.
	bool                                               prepare = false;
	std::unique_ptr<AheadRegisters>                    registers;
	uint64_t                                           handed_key = 0; // Ahead::handing
	uint64_t                                           submission = 0; // an offloaded draw's (AheadSubmission)
	uint32_t*                                          handed_count = nullptr; // its Ahead::Program's handed
};

struct PipelineCache::Ahead {
	using Job = AheadJob;
	struct Program {
		AheadTranslation<std::shared_ptr<Job>>                    translation; // under mutex
		std::mutex                                                walk; // MaterializeResources: the plan's scratch is its own
		ShaderRecompiler::IR::ResourcePlan                        plan; // set once, before translation is Ready
		std::shared_ptr<ShaderRecompiler::TranslateResult>        translated; // until a job takes it
		uint32_t                                                  handed = 0; // draws handed over and not prepared yet
		std::vector<ShaderRecompiler::IR::ResourceSpecialization> built;
		// A pixel shader: each specialization with the vertex shader and pass it was built for; a
		// pixel shader drawn again under another vertex shader or pass needs another pipeline.
		std::vector<std::pair<ShaderRecompiler::IR::ResourceSpecialization, uint64_t>> drawn;
	};
	// Graphics ahead: a graphics pipeline the GPU thread built, everything but its pixel stage. A
	// new pixel shader drawn with the same vertex shader usually comes with the same state, so a
	// worker builds it with these and the GPU thread's own build is a driver-cache hit.
	struct GraphicsTemplate {
		PipelineRenderingState                                  rendering;
		PipelineVertexInputState                                vertex_input;
		PipelineStaticParameters                                static_params;
		std::array<ShaderVertexInputInfo, 3>                    vertex_info;
		std::array<ShaderRecompiler::IR::CompiledShaderInfo, 3> vertex_programs;
		ShaderRecompiler::IR::ResourceSnapshot                  resources; // the stages need one; layouts do not read it
		GraphicsPrograms                                        programs; // the vertex stages' modules
		uint32_t                                                stages = 1;
		bool                                                    mesh   = false;
		// The vertex module's static key and specialization, for an exact job's match (one stage only).
		bool                                                    exact_vertex = false;
		std::vector<uint32_t>                                   vertex_static_key;
		ShaderRecompiler::IR::ResourceSpecialization            vertex_specialization;
	};
	std::unordered_map<uint64_t, std::vector<std::shared_ptr<const GraphicsTemplate>>> templates; // by vertex shader hash
	// shortcut: never trimmed; a pass and blend mode a key, a few hundred in the jungle.
	std::unordered_map<uint64_t, std::shared_ptr<const GraphicsTemplate>> by_state; // the latest by DrawStateSignature
	// Pixel jobs drawn before their vertex shader had a template: queued when it gets its first. A
	// pass of new shaders is one submission, scanned while the GPU thread has built none of them.
	std::unordered_map<uint64_t, std::vector<std::shared_ptr<Job>>> parked; // by vertex shader hash
	uint32_t                                                        parked_count = 0;
	std::map<std::pair<uint64_t, std::vector<uint32_t>>, Program> programs; // under mutex; a Program stays where it is
	// What the workers translated, for the GPU thread's program cache: a miss there takes the module
	// instead of translating the shader again (300-500 ms each for the GI prepass shaders).
	struct ReadyModule {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<uint8_t>                         packed; // the SPIR-V, zstd: ~600 MB raw at Trask, all session
		uint32_t                                     words = 0;
		ShaderRecompiler::IR::CompiledShaderInfo     info;
	};
	struct ReadyProgram {
		std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan; // until the GPU thread takes it
		bool                                                planned = false;
		std::vector<ReadyModule>                            modules;
	};
	std::map<std::tuple<ShaderType, uint64_t, std::vector<uint32_t>>, ReadyProgram> ready; // stage, hash, static key
	uint64_t                                                                        ready_words = 0;
	std::mutex                                 mutex;
	std::condition_variable                    changed;
	std::deque<std::shared_ptr<Job>>           jobs;
	// Draws the scanner handed to the workers to prepare (Offload): taken before jobs, oldest first, as the GPU thread
	// reaches them; offloaded counts them until prepared.
	std::deque<std::shared_ptr<Job>>           prepares;
	uint32_t                                   offloaded = 0;
	std::vector<std::shared_ptr<Job>>          running; // taken by a worker
	// The registers' keys of the draws and dispatches handed over and not prepared yet: the same registers again are
	// the same draw, and are not handed over twice.
	std::unordered_set<uint64_t>               handing;
	// What the GPU thread built itself (NoteGpuBuild): a job for it would only repeat that build.
	std::unordered_set<uint64_t>               gpu_built;
	bool                                       stop = false;
	std::vector<std::thread>                   workers;
	uint64_t                                   built = 0, waits = 0, waited_ms = 0, handed = 0, dropped = 0;
	// Graphics jobs built exactly, and why the others were built with the guesses.
	enum ExactCount : uint32_t { Exact, Tessellation, Restart, Prepare, Unsupported, Materialize, NoTemplate, OwnVertex, ExactCounts };
	std::array<uint64_t, ExactCounts> exact {};

	enum class Take : uint8_t { Go, HandOver, Parked, Drop };
	// A program's translation, once (AheadTranslation): Go once it is translated. On the scanner's thread (handed null)
	// one still to translate is HandOver; on a worker the handed job translates it, or is Parked until another worker
	// has. translate(plan, translated) returns false for a program not built ahead.
	template <typename Translate>
	Take TakeProgram(Program& program, const std::shared_ptr<Job>& handed, uint64_t hash, Translate&& translate) {
		{
			std::lock_guard lock(mutex);
			using Step      = AheadTranslation<std::shared_ptr<Job>>::Step;
			const auto step = handed != nullptr ? program.translation.Take(handed) : program.translation.Peek();
			if (step == Step::Ready) return Take::Go;
			if (step == Step::Unsupported) return Take::Drop;
			if (step == Step::Waiting && handed == nullptr) return Take::HandOver;
			if (step == Step::Waiting) {
				// It waits on the program, not on a worker: the GPU thread does not wait for it (WaitForGraphicsAhead).
				std::erase(running, handed);
				return Take::Parked;
			}
		}
		ShaderRecompiler::IR::ResourcePlan                 plan;
		std::shared_ptr<ShaderRecompiler::TranslateResult> translated;
		bool                                               supported = false;
		try {
			Common::ScopedExitThrows throws;
			supported = translate(plan, translated);
		} catch (const Common::ExitException& error) {
			LOGF("Ahead: gave up translating 0x%016" PRIx64 ": %s\n", hash, error.message.c_str());
		}
		{
			std::lock_guard lock(mutex);
			if (supported) {
				program.plan       = std::move(plan);
				program.translated = std::move(translated);
			}
			// The jobs that waited go first: they were queued before the ones queued now.
			auto waiting = program.translation.Finish(supported);
			jobs.insert(jobs.begin(), std::make_move_iterator(waiting.begin()), std::make_move_iterator(waiting.end()));
		}
		changed.notify_all();
		return supported ? Take::Go : Take::Drop;
	}

	// A translated program's specialization for these registers. Two threads may ask at once. The materialization
	// goes to results, for the GPU thread (AheadResults), under stage, hash and static_key.
	static bool Specialize(Program& program, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                        ShaderRecompiler::IR::ResourceSpecialization& specialization, AheadResults& results,
	                        ShaderType stage, uint64_t hash, std::span<const uint32_t> static_key) {
		if (results.Passed(AheadResult::Key(stage, hash, static_key, runtime.user_data, runtime.shader_base,
		                                    runtime.workgroup_counts, runtime.indirect_args_address))) {
			return false;
		}
		auto result = std::make_unique<AheadResult>();
		{
			std::lock_guard walk(program.walk);
			if (!ShaderRecompiler::IR::MaterializeResources(program.plan,
			                                                result->log.Recording(runtime, ShaderRecompiler::IR::SrtReadLog::MaxAheadWords),
			                                                result->resources,
			                                                specialization)) {
				return false;
			}
		}
		result->stage                 = stage;
		result->hash                  = hash;
		result->static_key.assign(static_key.begin(), static_key.end());
		result->user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
		result->workgroup_counts.assign(runtime.workgroup_counts.begin(), runtime.workgroup_counts.end());
		result->workgroup_size.assign(runtime.workgroup_size.begin(), runtime.workgroup_size.end());
		result->shader_base           = runtime.shader_base;
		result->indirect_args_address = runtime.indirect_args_address;
		result->guest_image_heap      = runtime.guest_image_heap;
		result->specialization        = specialization;
		results.Publish(std::move(result));
		return true;
	}

	// The scanner's thread, under the lock: waits until fewer than 64 jobs are queued; false once the workers stop.
	// Only the scanner waits for room: a worker that did would wait for itself.
	bool RoomForScanner(std::unique_lock<std::mutex>& lock) {
		changed.wait(lock, [&] { return stop || jobs.size() < 64; });
		return !stop;
	}

	// The scanner's thread: a draw or dispatch with a program still to translate (on program) goes to a worker.
	// shortcut: 32 draws of a program handed over and not prepared yet, and 64 jobs queued, as for the builds. The scanner
	// waits for room (RoomForScanner) rather than lose the draw: dropped, it was never made again, and the GPU thread
	// compiled 71 such graphics pipelines itself in a cold run (side-session-notes/ahead-prepare.md).
	void HandOver(Program& program, std::shared_ptr<Job> job, uint64_t key) {
		{
			std::unique_lock lock(mutex);
			changed.wait(lock, [&] { return stop || (program.handed < 32 && jobs.size() < 64); });
			if (stop || !handing.insert(key).second) return;
			program.handed++;
			handed++;
			job->prepare      = true;
			job->handed_key   = key;
			job->handed_count = &program.handed;
			jobs.push_back(std::move(job));
		}
		changed.notify_all();
	}
};

// The worker's walk: the guest's bytes as they are. TryReadBufferBacking (ReadShaderGuestMemory)
// refuses GPU memory off the GPU thread; bytes the GPU has yet to write give a wrong guess only.
static bool ReadAheadGuestMemory(void* /*data*/, uint64_t address, std::span<uint32_t> values) {
	if (address < 0x10000u) { // as ReadCurrentGuestMemory
		std::ranges::fill(values, 0u);
		return true;
	}
	// A take compares every word read here on the GPU thread (SrtReadLog::StillReads), so a half-done write is caught.
	return !values.empty() && Libs::LibKernel::Memory::TryReadBackingCached(address, values.data(), values.size_bytes());
}

// As ProgramCache::Get sets them for a compute shader: the worker's SPIR-V must be the GPU thread's.
static ShaderRecompiler::CompileOptions ComputeOptions(const ShaderParams& params,
                                                      ShaderComputeInputInfo& input_info,
                                                      const ShaderRecompiler::Spirv::Profile& profile) {
	ShaderRecompiler::CompileOptions options;
	options.stage                = ShaderType::Compute;
	options.profile              = &profile;
	options.shader_hash          = params.hash;
	options.user_data            = std::span(params.user_data).first(params.user_data_count);
	options.back_code            = params.back_code;
	options.dump_ir              = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.early_dump           = options.dump_ir;
	options.emit_debug_info      = Config::GraphicsDebugDumpEnabled();
	options.dump_label           = "ShaderRecompiler CS";
	options.input_info.compute   = &input_info;
	options.wave_size            = input_info.wave_size;
	return options;
}

// As ProgramCache::Get sets them for a vertex or mesh shader.
static ShaderRecompiler::CompileOptions VertexOptions(const ShaderParams& params, ShaderVertexInputInfo& input_info,
                                                     const ShaderRecompiler::Spirv::Profile& profile) {
	const bool                       mesh = input_info.logical_stage == ShaderType::Mesh;
	ShaderRecompiler::CompileOptions options;
	options.stage              = input_info.logical_stage;
	options.profile            = &profile;
	options.shader_hash        = params.hash;
	options.user_data          = std::span(params.user_data).first(params.user_data_count);
	options.back_code          = params.back_code;
	options.dump_ir            = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.early_dump         = options.dump_ir;
	options.emit_debug_info    = Config::GraphicsDebugDumpEnabled();
	options.dump_label         = mesh ? "ShaderRecompiler MS" : "ShaderRecompiler VS";
	options.input_info.vertex  = &input_info;
	options.user_data_base     = mesh ? 0u : 8u;
	options.wave_size          = mesh ? input_info.mesh.wave_size : input_info.wave_size;
	return options;
}

// A specialization as a hash, for the scanner's set of what a pixel shader was drawn with.
static uint64_t SpecializationHash(const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint64_t seed) {
	std::vector<uint32_t> words;
	for (const auto& buffer: specialization.buffers) {
		words.insert(words.end(), {buffer.packed_stride, static_cast<uint32_t>(buffer.descriptor_format),
		                           buffer.descriptor_swizzle,
		                           static_cast<uint32_t>(buffer.zero_stride_oob) | (static_cast<uint32_t>(buffer.scalar_empty) << 1u),
		                           buffer.indirect_root, buffer.indirect_mapping_offset, buffer.indirect_search_iterations});
	}
	words.push_back(UINT32_MAX);
	for (const auto& image: specialization.images) {
		words.insert(words.end(), {static_cast<uint32_t>(image.numeric_class), static_cast<uint32_t>(image.dimension),
		                           image.mip_count, static_cast<uint32_t>(image.conversion_format), image.shader_swizzle,
		                           image.indirect_root, image.indirect_mapping_offset, image.indirect_search_iterations,
		                           static_cast<uint32_t>(image.cube) | (static_cast<uint32_t>(image.fmask) << 1u) |
		                               (static_cast<uint32_t>(image.heap) << 2u) | (static_cast<uint32_t>(image.guest_heap) << 3u)});
	}
	words.push_back(UINT32_MAX);
	for (const auto& sampler: specialization.samplers) {
		words.insert(words.end(), {sampler.indirect_root, sampler.indirect_mapping_offset, sampler.indirect_search_iterations,
		                           static_cast<uint32_t>(sampler.guest_heap)});
	}
	if (specialization.buffer_table) words.push_back(0x7ab1e000u);
	return XXH3_64bits_withSeed(words.data(), words.size() * sizeof(uint32_t), seed);
}

// The exact draw a graphics job builds a pipeline for, as the scanner prepared it and as the GPU thread draws it
// (PipelineCache::GraphicsIdentity): its two programs, their static keys and specializations, its vertex input and its
// pass and blend state. 0 is "unknown".
static uint64_t GraphicsAheadIdentity(uint64_t pixel_hash, const std::vector<uint32_t>& pixel_key,
                                      const ShaderRecompiler::IR::ResourceSpecialization& pixel_specialization,
                                      uint64_t vertex_hash, const std::vector<uint32_t>& vertex_key,
                                      const ShaderRecompiler::IR::ResourceSpecialization& vertex_specialization,
                                      const PipelineVertexInputState& vertex_input, uint64_t state_signature) {
	auto hash = XXH3_64bits_withSeed(pixel_key.data(), pixel_key.size() * sizeof(uint32_t), pixel_hash);
	hash      = SpecializationHash(pixel_specialization, hash);
	hash      = XXH3_64bits_withSeed(vertex_key.data(), vertex_key.size() * sizeof(uint32_t), hash ^ vertex_hash);
	hash      = SpecializationHash(vertex_specialization, hash);
	std::vector<uint32_t> input {vertex_input.binding_count, vertex_input.attribute_count};
	for (uint32_t i = 0; i < vertex_input.binding_count; i++) {
		input.insert(input.end(), {vertex_input.bindings[i].stride, static_cast<uint32_t>(vertex_input.bindings[i].instance)});
	}
	for (uint32_t i = 0; i < vertex_input.attribute_count; i++) {
		input.insert(input.end(), {vertex_input.attributes[i].offset, vertex_input.attributes[i].binding});
	}
	hash = XXH3_64bits_withSeed(input.data(), input.size() * sizeof(uint32_t), hash ^ state_signature);
	return hash != 0 ? hash : 1;
}

// The exact dispatch a compute job builds a pipeline for: its program, static key and specialization.
static uint64_t ComputeAheadIdentity(uint64_t shader_hash, const std::vector<uint32_t>& key,
                                     const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	const auto hash = SpecializationHash(specialization, XXH3_64bits_withSeed(key.data(), key.size() * sizeof(uint32_t), shader_hash));
	return hash != 0 ? hash : 1;
}

// As ProgramCache::Get sets them for a pixel shader.
static ShaderRecompiler::CompileOptions PixelOptions(const ShaderParams& params, ShaderPixelInputInfo& input_info,
                                                    const ShaderRecompiler::Spirv::Profile& profile, bool split_fragment_set) {
	ShaderRecompiler::CompileOptions options;
	options.stage                 = ShaderType::Pixel;
	options.profile               = &profile;
	options.descriptor_set        = split_fragment_set ? ShaderRecompiler::IR::FragmentSet : 0u;
	options.shader_hash           = params.hash;
	options.user_data             = std::span(params.user_data).first(params.user_data_count);
	options.back_code             = params.back_code;
	options.dump_ir               = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.early_dump            = options.dump_ir;
	options.emit_debug_info       = Config::GraphicsDebugDumpEnabled();
	options.dump_label            = "ShaderRecompiler PS";
	options.input_info.pixel      = &input_info;
	options.wave_size             = input_info.wave_size;
	return options;
}

// The registers a graphics pipeline's render targets and static state come from (GetGraphicsPipeline):
// the scanner's draw and the GPU thread's build of one pass and blend mode give the same value.
static uint64_t DrawStateSignature(const HW::Context& ctx, Prospero::PrimitiveType primitive) {
	uint64_t   hash = 0;
	const auto mix  = [&](const auto& value) { hash = XXH3_64bits_withSeed(&value, sizeof(value), hash); };
	const auto mask = ctx.GetRenderTargetMask();
	mix(mask);
	mix(primitive);
	for (uint32_t slot = 0; slot < 8; slot++) {
		if (((mask >> (slot * 4u)) & 0xfu) == 0) continue;
		mix(ctx.GetRenderTarget(slot).info);
		mix(ctx.GetRenderTarget(slot).attrib);
		mix(ctx.GetBlendControl(slot));
	}
	mix(ctx.GetDepthRenderTarget().z_info);
	mix(ctx.GetDepthRenderTarget().stencil_info);
	mix(ctx.GetModeControl());
	mix(ctx.GetClipControl());
	mix(ctx.GetColorControl());
	return hash;
}

void PipelineCache::RecordGraphicsTemplate(const GraphicsPipelineKey& key, std::span<const ShaderVertexInputInfo> vertex_info,
                                           const GraphicsPrograms& programs, uint64_t state_signature) {
	auto tmpl = std::make_shared<Ahead::GraphicsTemplate>();
	tmpl->rendering     = key.rendering;
	tmpl->vertex_input  = key.vertex_input;
	tmpl->static_params = key.static_params;
	tmpl->stages        = static_cast<uint32_t>(std::min<size_t>(vertex_info.size(), 3u));
	for (uint32_t i = 0; i < tmpl->stages; i++) {
		tmpl->vertex_info[i]       = vertex_info[i];
		tmpl->vertex_programs[i]   = *vertex_info[i].stage.program;
		tmpl->vertex_info[i].stage = {.program = &tmpl->vertex_programs[i], .resources = &tmpl->resources};
	}
	tmpl->programs       = programs;
	tmpl->programs.pixel = {};
	tmpl->mesh           = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (const auto found = m_program_cache->specialization_of.find(programs.vertex[0].id);
	    tmpl->stages == 1 && found != m_program_cache->specialization_of.end()) {
		BuildStageStaticKey(vertex_info[0], tmpl->vertex_static_key);
		tmpl->vertex_specialization = found->second;
		tmpl->exact_vertex          = true;
	}
	const auto hash      = vertex_info[0].stage.program->shader_hash;
	std::lock_guard lock(m_ahead->mutex);
	auto&           list = m_ahead->templates[hash];
	// Latest last: a state seen again moves to the back.
	std::erase_if(list, [&](const auto& other) {
		return other->programs.vertex[0].id == programs.vertex[0].id && other->rendering == key.rendering &&
		       other->vertex_input == key.vertex_input && other->static_params == key.static_params;
	});
	m_ahead->by_state[state_signature] = tmpl;
	list.push_back(std::move(tmpl));
	if (list.size() > 16) list.erase(list.begin()); // shortcut: a vertex shader's 16 latest states
	if (const auto parked = m_ahead->parked.find(hash); parked != m_ahead->parked.end()) {
		for (auto& job: parked->second) m_ahead->jobs.push_back(std::move(job));
		m_ahead->parked_count -= static_cast<uint32_t>(parked->second.size());
		m_ahead->parked.erase(parked);
		m_ahead->changed.notify_all();
	}
}

// The vertex module id in the key of an exact build whose vertex module the worker compiled itself: the GPU
// thread's module of the same SPIR-V has an id of its own (WaitForGraphicsAhead matches any).
static constexpr uint64_t OwnVertexModuleId = UINT64_MAX - 1;

// Graphics ahead's exact builds (the prepare stage): the draw's own vertex program, its static key and specialization
// from the scanner's registers and plain guest memory as the pixel shader's, and the draw's topology and vertex input.
// Ahead::Exact when it has them, or why not; step is HandOver or Parked when its vertex program is still to translate.
uint32_t PipelineCache::PrepareExactVertex(const AheadRegisters& draw, const HW::Context& context,
                                           const std::shared_ptr<AheadJob>& handed, AheadJob& job, AheadStep& step) {
	step                   = AheadStep::Build;
	const auto& user_config = draw.user_config;
	const auto  primitive   = user_config.GetPrimType();
	if (primitive == Prospero::PrimitiveType::kPatch) return Ahead::Tessellation;
	// A strip's custom restart index is looked for in the index data on the GPU thread, which the scanner cannot read.
	const bool strip = primitive == Prospero::PrimitiveType::kLineStrip || primitive == Prospero::PrimitiveType::kTriFan ||
	                   primitive == Prospero::PrimitiveType::kTriStrip;
	if ((strip && (user_config.GetPrimitiveResetControl() & 1u) != 0) || !GetDrawTopology(user_config, job.topology)) {
		return Ahead::Restart;
	}
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderParams                         params;
	try {
		Common::ScopedExitThrows throws;
		bool object_primitive_id = false;
		params   = PrepareVertexStages(draw.vertex_regs, draw.sh, context, user_config, true, vertex_info, object_primitive_id,
		                               job.vertex_push_cursor, draw.indirect_offsets, draw.indirect_draw)[0];
		job.mesh = vertex_info[0].logical_stage == ShaderType::Mesh;
		job.vertex_input = MakeVertexInput(vertex_info[0], job.mesh);
		BuildStageStaticKey(vertex_info[0], job.vertex_static_key);
	} catch (const Common::ExitException&) {
		return Ahead::Prepare;
	}
	auto program_key = job.vertex_static_key;
	program_key.push_back(0x56455254u); // a vertex key, apart from the pixel and compute keys
	Ahead::Program* program = nullptr;
	{
		std::lock_guard lock(m_ahead->mutex);
		program = &m_ahead->programs[{params.hash, std::move(program_key)}];
	}
	auto options = VertexOptions(params, vertex_info[0], m_program_cache->profile);
	switch (m_ahead->TakeProgram(*program, handed, params.hash, [&](auto& plan, auto& translated) {
		auto source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (!source.calls.empty()) return false;
		translated = std::make_shared<ShaderRecompiler::TranslateResult>(ShaderRecompiler::TranslateProgram(source.decoded, options));
		plan       = ShaderRecompiler::IR::ExtractResourcePlan(translated->program);
		return true;
	})) {
		case Ahead::Take::Go: break;
		case Ahead::Take::Drop: return Ahead::Unsupported;
		case Ahead::Take::HandOver: step = AheadStep::HandOver; return Ahead::Unsupported;
		case Ahead::Take::Parked: step = AheadStep::Parked; return Ahead::Unsupported;
	}
	job.vertex_program_translated = &program->translated;
	try {
		Common::ScopedExitThrows throws;
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = options.user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadAheadGuestMemory,
		    .read_specialization_memory = ReadAheadGuestMemory,
		    .guest_image_heap           = m_graphics.guest_image_heap != nullptr,
		};
		if (!Ahead::Specialize(*program, runtime, job.vertex_specialization, m_program_cache->ahead_results,
		                       vertex_info[0].logical_stage, params.hash, job.vertex_static_key)) {
			return Ahead::Materialize;
		}
	} catch (const Common::ExitException&) {
		std::lock_guard lock(m_ahead->mutex);
		program->translation.SetUnsupported();
		return Ahead::Unsupported;
	}
	job.rect_list     = Prospero::IsRectList(primitive);
	job.vertex_params = params;
	job.vertex_info   = vertex_info[0];
	return Ahead::Exact;
}

// Graphics ahead's prepare stage: the draw's pixel program and specialization, its exact vertex program, the dedupe
// and the job. On the scanner's thread (handed null) the job is queued, or the draw handed over when one of its programs
// is still to translate; on a worker (handed: the scanner's copy of the draw) the stage translates what it has to, and
// returns the job to build in build (Build).
PipelineCache::AheadStep PipelineCache::PrepareGraphicsAhead(const AheadRegisters& draw, const HW::Context& context,
                                                             const std::shared_ptr<AheadJob>& handed,
                                                             std::shared_ptr<AheadJob>& build) {
	const auto vertex_hash = ShaderMappedHash(draw.vertex_regs.es_regs.data_addr);
	if (vertex_hash == 0 || !ShaderIsMapped(draw.pixel_regs.ps_regs.data_addr)) return AheadStep::Done;
	const auto& user_config = draw.user_config;
	bool        mesh        = false;
	{
		std::lock_guard lock(m_ahead->mutex);
		const auto      found = m_ahead->templates.find(vertex_hash);
		if (found == m_ahead->templates.end() || found->second.empty()) {
			mesh = ShaderVertexIsMesh(context, draw.vertex_regs);
		} else {
			mesh = found->second.back()->mesh;
		}
	}
	Ahead::Program* current = nullptr;
	try {
		Common::ScopedExitThrows throws;
		const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
		ShaderPixelInputInfo pixel_info {};
		const auto params = PreparePixelStage(draw.pixel_regs, draw.sh, context, user_config, draw.mapping, mesh,
		                                      tess_active, context.HasObjectPrimitiveIdOutput(), pixel_info);
		std::vector<uint32_t> pixel_key;
		BuildStageStaticKey(pixel_info, pixel_key);
		auto program_key = pixel_key;
		program_key.push_back(0x50495845u); // a pixel key, apart from the compute keys
		{
			std::lock_guard lock(m_ahead->mutex);
			current = &m_ahead->programs[{params.hash, std::move(program_key)}];
		}
		auto&      program   = *current;
		const auto signature = DrawStateSignature(context, user_config.GetPrimType());
		// The scanner's thread: the draw to a worker, keyed by the registers it is prepared from.
		const auto hand_over = [&](Ahead::Program& untranslated) {
			auto key = XXH3_64bits_withSeed(params.user_data.data(), params.user_data_count * sizeof(uint32_t), params.hash);
			key      = XXH3_64bits_withSeed(pixel_key.data(), pixel_key.size() * sizeof(uint32_t), key);
			key      = XXH3_64bits_withSeed(&draw.vertex_regs, sizeof(draw.vertex_regs), key ^ signature);
			key      = XXH3_64bits_withSeed(&draw.indirect_offsets, sizeof(draw.indirect_offsets), key + draw.indirect_draw);
			auto job = std::make_shared<Ahead::Job>();
			job->pixel       = true;
			job->params.hash = params.hash;
			job->registers   = std::make_unique<AheadRegisters>(draw);
			job->context     = std::make_shared<const HW::Context>(context);
			m_ahead->HandOver(untranslated, std::move(job), key);
			return AheadStep::Done;
		};
		auto options = PixelOptions(params, pixel_info, m_program_cache->profile, m_graphics.SplitsFragmentSet());
		switch (m_ahead->TakeProgram(program, handed, params.hash, [&](auto& plan, auto& translated) {
			auto source = ShaderRecompiler::PrepareShaderSource(params.code, options);
			if (!source.calls.empty()) return false;
			translated = std::make_shared<ShaderRecompiler::TranslateResult>(ShaderRecompiler::TranslateProgram(source.decoded, options));
			plan       = ShaderRecompiler::IR::ExtractResourcePlan(translated->program);
			return true;
		})) {
			case Ahead::Take::Go: break;
			case Ahead::Take::HandOver: return hand_over(program);
			case Ahead::Take::Parked: return AheadStep::Parked;
			case Ahead::Take::Drop: return AheadStep::Done;
		}
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = options.user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadAheadGuestMemory,
		    .read_specialization_memory = ReadAheadGuestMemory,
		    .guest_image_heap           = m_graphics.guest_image_heap != nullptr,
		};
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// Past the cap too: the GPU thread takes the materialization (AheadResults).
		if (!Ahead::Specialize(program, runtime, specialization, m_program_cache->ahead_results, ShaderType::Pixel,
		                       params.hash, pixel_key)) {
			return AheadStep::Done;
		}
		// shortcut: 32 pipelines a pixel shader; raise it if the misses show a shader past it.
		if (std::lock_guard lock(m_ahead->mutex); program.drawn.size() >= 32) return AheadStep::Done;
		auto job          = std::make_shared<Ahead::Job>();
		auto vertex_step  = AheadStep::Build;
		job->exact_reason = PrepareExactVertex(draw, context, handed, *job, vertex_step);
		if (vertex_step == AheadStep::Parked) return AheadStep::Parked;
		if (vertex_step == AheadStep::HandOver) return hand_over(program); // its vertex program is still to translate
		const bool exact    = job->exact_reason == Ahead::Exact;
		auto       draw_key = XXH3_64bits_withSeed(&vertex_hash, sizeof(vertex_hash), signature);
		if (exact) {
			// Another vertex specialization in the same pass is another pipeline.
			draw_key = XXH3_64bits_withSeed(job->vertex_static_key.data(),
			                                job->vertex_static_key.size() * sizeof(uint32_t), draw_key);
			draw_key = SpecializationHash(job->vertex_specialization, draw_key);
			job->identity = GraphicsAheadIdentity(params.hash, pixel_key, specialization, job->vertex_params.hash,
			                                      job->vertex_static_key, job->vertex_specialization, job->vertex_input,
			                                      signature);
		}
		const auto drawn = std::pair {specialization, draw_key};
		{
			std::unique_lock lock(m_ahead->mutex);
			if (handed == nullptr && !m_ahead->RoomForScanner(lock)) return AheadStep::Done;
			// Built already: the IR kept for a first job goes (a later job translates on its worker).
			if ((exact && m_ahead->gpu_built.contains(job->identity)) ||
			    std::ranges::find(program.drawn, drawn) != program.drawn.end()) {
				program.translated = nullptr;
				return AheadStep::Done;
			}
			program.drawn.push_back(drawn);
			job->translated = std::exchange(program.translated, nullptr);
			if (exact) job->vertex_translated = std::exchange(*job->vertex_program_translated, nullptr);
		}
		if (exact) job->context = handed != nullptr ? handed->context : std::make_shared<const HW::Context>(context);
		job->params          = params;
		job->specialization  = std::move(specialization);
		job->pixel           = true;
		job->pixel_info      = pixel_info;
		job->vertex_hash     = vertex_hash;
		job->state_signature = signature;
		{
			std::lock_guard lock(m_ahead->mutex);
			// An exact job needs no template: without one it builds the draw's own vertex module.
			if (const auto found = m_ahead->templates.find(vertex_hash);
			    !exact && (found == m_ahead->templates.end() || found->second.empty())) {
				// shortcut: 512 parked jobs; one draw pass of new shaders stayed far below it.
				if (m_ahead->parked_count >= 512) return AheadStep::Done;
				m_ahead->parked_count++;
				m_ahead->parked[vertex_hash].push_back(std::move(job));
				return AheadStep::Done;
			}
			if (handed == nullptr) {
				m_ahead->jobs.push_back(std::move(job));
			} else {
				// The GPU thread sees the job it may wait for (WaitForGraphicsAhead) with its identity from here on.
				std::ranges::replace(m_ahead->running, handed, job);
				build = std::move(job);
			}
		}
		m_ahead->changed.notify_all();
		return handed != nullptr ? AheadStep::Build : AheadStep::Done;
	} catch (const Common::ExitException& error) {
		if (current != nullptr) {
			std::lock_guard lock(m_ahead->mutex);
			current->translation.SetUnsupported();
		}
		LOGF("Graphics ahead: gave up: %s\n", error.message.c_str());
		return AheadStep::Done;
	}
}

void PipelineCache::BuildGraphicsAhead(const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
                                       const HW::ShaderRegisters& sh, const HW::Context& context,
                                       const HW::UserConfig& user_config,
                                       std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
                                       ShaderIndirectOffsets indirect_offsets, bool indirect_draw) {
	if (m_ahead == nullptr) return;
	AheadRegisters draw {.vertex_regs      = vertex_regs,
	                     .pixel_regs       = pixel_regs,
	                     .sh               = sh,
	                     .user_config      = user_config,
	                     .indirect_offsets = indirect_offsets,
	                     .indirect_draw    = indirect_draw};
	std::ranges::copy(target_export_mapping, draw.mapping.begin());
	// The scanner alone fell behind the GPU thread inside a frame (materializing was 70 % of its thread at Marvel's
	// Wolverine's Trask outpost): the workers prepare the draw, as a handed-over one. Past 4 a worker queued, the
	// scanner prepares it itself, as before.
	{
		std::unique_lock lock(m_ahead->mutex);
		if (!m_ahead->stop && m_ahead->offloaded < 4u * m_ahead->workers.size()) {
			auto job          = std::make_shared<AheadJob>();
			job->pixel        = true;
			job->prepare      = true;
			job->registers    = std::make_unique<AheadRegisters>(draw);
			job->context      = std::make_shared<const HW::Context>(context);
			job->handed_count = &m_ahead->offloaded;
			job->submission   = t_ahead_submission;
			m_ahead->offloaded++;
			m_ahead->prepares.push_back(std::move(job));
			lock.unlock();
			m_ahead->changed.notify_one();
			return;
		}
	}
	std::shared_ptr<AheadJob> build;
	(void)PrepareGraphicsAhead(draw, context, nullptr, build);
}

void PipelineCache::StartComputeAhead() {
	if (!ComputeAheadScanner::Enabled()) return;
	m_ahead = std::make_unique<Ahead>();
	const auto threads = AheadWorkerCount(std::thread::hardware_concurrency());
	for (uint32_t i = 0; i < threads; i++) m_ahead->workers.emplace_back([this] { RunComputeAhead(); });
	PipelineCacheLog("Compute ahead: {} worker threads", threads);
}

void PipelineCache::StopComputeAhead() {
	if (m_ahead == nullptr) return;
	{
		std::lock_guard lock(m_ahead->mutex);
		m_ahead->stop = true;
		m_ahead->jobs.clear();
		m_ahead->prepares.clear();
	}
	m_ahead->changed.notify_all();
	for (auto& worker: m_ahead->workers) worker.join();
	LOGF("Compute ahead: %" PRIu64 " built, %" PRIu64 " handed over to prepare, %" PRIu64
	     " dropped as built by the GPU thread; the GPU thread waited %" PRIu64 " times (%" PRIu64 " ms)\n",
	     m_ahead->built, m_ahead->handed, m_ahead->dropped, m_ahead->waits, m_ahead->waited_ms);
	m_ahead.reset();
}

// Compute ahead's prepare stage, as PrepareGraphicsAhead's for a dispatch.
PipelineCache::AheadStep PipelineCache::PrepareComputeAhead(const AheadRegisters& dispatch, ShaderComputeInputInfo input_info,
                                                            const std::shared_ptr<AheadJob>& handed,
                                                            std::shared_ptr<AheadJob>& build) {
	Ahead::Program* current = nullptr;
	// The scanner's registers can be another dispatch's: a translation they break is a lost guess.
	try {
		Common::ScopedExitThrows throws;
		const auto scanned            = input_info; // as the scanner had it, for a worker
		input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
		const auto params             = PrepareProgram(dispatch.compute_regs, dispatch.sh, input_info);
		if (IsRayTracingShader(params)) return AheadStep::Done;
		input_info.lds_storage = input_info.lds_size_dwords * 4u >
		    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
		std::vector<uint32_t> static_key;
		BuildStageStaticKey(input_info, static_key);
		{
			std::lock_guard lock(m_ahead->mutex);
			current = &m_ahead->programs[{params.hash, static_key}];
		}
		auto& program = *current;
		auto  options = ComputeOptions(params, input_info, m_program_cache->profile);
		switch (m_ahead->TakeProgram(program, handed, params.hash, [&](auto& plan, auto& translated) {
			auto source = ShaderRecompiler::PrepareShaderSource(params.code, options);
			// shortcut: shaders with calls link their leaves at run time; not built ahead, add them
			// when the Pipelines tab shows them stalling.
			if (!source.calls.empty()) {
				LOGF("Compute ahead: CS=0x%016" PRIx64 " has calls, not built ahead\n", params.hash);
				return false;
			}
			translated = std::make_shared<ShaderRecompiler::TranslateResult>(ShaderRecompiler::TranslateProgram(source.decoded, options));
			plan       = ShaderRecompiler::IR::ExtractResourcePlan(translated->program);
			return true;
		})) {
			case Ahead::Take::Go: break;
			case Ahead::Take::HandOver: {
				auto key = XXH3_64bits_withSeed(&dispatch.compute_regs, sizeof(dispatch.compute_regs), params.hash);
				key      = XXH3_64bits_withSeed(static_key.data(), static_key.size() * sizeof(uint32_t), key);
				key      = XXH3_64bits_withSeed(input_info.workgroup_counts, sizeof(input_info.workgroup_counts),
				                                key ^ input_info.indirect_args_address);
				auto job = std::make_shared<Ahead::Job>();
				job->params.hash = params.hash;
				job->input_info  = scanned;
				job->registers   = std::make_unique<AheadRegisters>(dispatch);
				m_ahead->HandOver(program, std::move(job), key);
				return AheadStep::Done;
			}
			case Ahead::Take::Parked: return AheadStep::Parked;
			case Ahead::Take::Drop: return AheadStep::Done;
		}
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = options.user_data,
		    .shader_base                = params.Base(),
		    // Without it the walk reads guest addresses as host pointers.
		    .read_memory                = ReadAheadGuestMemory,
		    .read_specialization_memory = ReadAheadGuestMemory,
		    .guest_image_heap           = m_graphics.guest_image_heap != nullptr,
		};
		runtime.workgroup_counts      = input_info.workgroup_counts;
		runtime.workgroup_size        = input_info.threads_num;
		runtime.indirect_args_address = input_info.indirect_args_address;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// shortcut: at most 16 guesses a program key and 64 waiting jobs (the scanner waits for room), so registers
		// misread for every dispatch cannot flood the workers; raise if the Pipelines tab shows misses beyond them.
		if (std::lock_guard lock(m_ahead->mutex); program.built.size() >= 16) return AheadStep::Done;
		if (!Ahead::Specialize(program, runtime, specialization, m_program_cache->ahead_results, ShaderType::Compute,
		                       params.hash, static_key)) {
			return AheadStep::Done;
		}
		const auto identity = ComputeAheadIdentity(params.hash, static_key, specialization);
		{
			std::unique_lock lock(m_ahead->mutex);
			if (handed == nullptr && !m_ahead->RoomForScanner(lock)) return AheadStep::Done;
			if (m_ahead->gpu_built.contains(identity) ||
			    std::ranges::find(program.built, specialization) != program.built.end()) {
				program.translated = nullptr; // built already: a later job translates on its worker
				return AheadStep::Done;
			}
			program.built.push_back(specialization);
			auto job = std::make_shared<Ahead::Job>(Ahead::Job {
			    .params = params, .input_info = input_info, .specialization = std::move(specialization),
			    .translated = std::exchange(program.translated, nullptr)});
			job->identity = identity;
			if (handed == nullptr) {
				m_ahead->jobs.push_back(std::move(job));
			} else {
				std::ranges::replace(m_ahead->running, handed, job);
				build = std::move(job);
			}
		}
		m_ahead->changed.notify_all();
		return handed != nullptr ? AheadStep::Build : AheadStep::Done;
	} catch (const Common::ExitException& error) {
		if (current != nullptr) {
			std::lock_guard lock(m_ahead->mutex);
			current->translation.SetUnsupported();
		}
		LOGF("Compute ahead: gave up: %s\n", error.message.c_str());
		return AheadStep::Done;
	}
}

void PipelineCache::BuildComputeAhead(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                                      ShaderComputeInputInfo input_info) {
	if (m_ahead == nullptr) return;
	AheadRegisters dispatch {.compute_regs = regs, .sh = sh};
	std::shared_ptr<AheadJob> build;
	(void)PrepareComputeAhead(dispatch, input_info, nullptr, build);
}

// A worker's module, kept for the GPU thread's program cache (Ahead::ready).
void PipelineCache::NoteReady(ShaderType stage, uint64_t hash, std::vector<uint32_t> static_key,
                              const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                              const std::vector<uint32_t>& spirv, const ShaderRecompiler::IR::CompiledShaderInfo& info) {
	std::vector<uint8_t> packed(ZSTD_compressBound(spirv.size() * sizeof(uint32_t)));
	const auto packed_size = ZSTD_compress(packed.data(), packed.size(), spirv.data(), spirv.size() * sizeof(uint32_t), 1);
	if (ZSTD_isError(packed_size)) return;
	packed.resize(packed_size);
	packed.shrink_to_fit();
	std::lock_guard lock(m_ahead->mutex);
	// shortcut: 192 M words (768 MB of host memory) kept at most. 256 MB held the ahead modules of a first session, but
	// not Marvel's Wolverine's precompile (hints v6: past 250 MB 34 s into the boot), and the modules it dropped were
	// translated again on the GPU thread (session 14). The store shrinks as the GPU thread takes its modules.
	if (m_ahead->ready_words + spirv.size() > (192u << 20u)) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true)) LOGF("Reuse store full at %" PRIu64 " MB: modules dropped\n", m_ahead->ready_words * 4u >> 20u);
		return;
	}
	auto& program = m_ahead->ready[{stage, hash, std::move(static_key)}];
	// shortcut: 32 modules a program. The precompile's 453 seeds of one compute shader's image variants filled the
	// store, and the pixel shaders after them were translated again on the GPU thread in the fight (session 14).
	// A variant past them is still in the driver cache; only its translation is done again.
	if (program.modules.size() >= 32u) return;
	// A module is the one TakeReadyModule wants by its specialization and its push-data place: another place of the
	// same specialization is another module, not a duplicate.
	if (std::ranges::any_of(program.modules, [&](const auto& module) {
		    return module.specialization == specialization &&
		           module.info.bindings.push_data_start_dword == info.bindings.push_data_start_dword;
	    })) {
		return;
	}
	program.modules.push_back({specialization, std::move(packed), static_cast<uint32_t>(spirv.size()), info});
	m_ahead->ready_words += spirv.size();
}

// The program's resource plan, once, from the worker's translation: a plan owns compiled code, so it is
// moved to the GPU thread, not copied.
void PipelineCache::NoteReadyPlan(ShaderType stage, uint64_t hash, const std::vector<uint32_t>& static_key,
                                  const ShaderRecompiler::IR::Program& translated) {
	{
		std::lock_guard lock(m_ahead->mutex);
		auto& program = m_ahead->ready[{stage, hash, static_key}];
		if (program.planned) return;
		program.planned = true;
	}
	auto plan = std::make_unique<ShaderRecompiler::IR::ResourcePlan>(ShaderRecompiler::IR::ExtractResourcePlan(translated));
	std::lock_guard lock(m_ahead->mutex);
	m_ahead->ready[{stage, hash, static_key}].plan = std::move(plan);
}

// A plan the stage store held (PrecompileCompiler::LoadStage), unless the program has one.
void PipelineCache::NoteReadyPlan(ShaderType stage, uint64_t hash, const std::vector<uint32_t>& static_key,
                                  std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan) {
	std::lock_guard lock(m_ahead->mutex);
	auto& program = m_ahead->ready[{stage, hash, static_key}];
	if (program.planned) return;
	program.planned = true;
	program.plan    = std::move(plan);
}

// The GPU thread: the plan a worker extracted for this program, and a worker's module for this
// specialization at this push-data place (the module leaves the store).
std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> PipelineCache::TakeReadyPlan(ShaderType stage, uint64_t hash,
                                                                                   const std::vector<uint32_t>& static_key) {
	if (m_ahead == nullptr) return nullptr;
	std::lock_guard lock(m_ahead->mutex);
	const auto found = m_ahead->ready.find({stage, hash, static_key});
	return found != m_ahead->ready.end() ? std::move(found->second.plan) : nullptr;
}

bool PipelineCache::TakeReadyModule(ShaderType stage, uint64_t hash, const std::vector<uint32_t>& static_key,
                                    const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                    uint32_t push_data_cursor, std::vector<uint32_t>& spirv,
                                    ShaderRecompiler::IR::CompiledShaderInfo& info) {
	if (m_ahead == nullptr) return false;
	std::vector<uint8_t> packed;
	uint32_t             words = 0;
	{
		std::lock_guard lock(m_ahead->mutex);
		const auto found = m_ahead->ready.find({stage, hash, static_key});
		if (found == m_ahead->ready.end()) return false;
		auto& modules = found->second.modules;
		const auto module = std::ranges::find_if(modules, [&](const auto& candidate) {
			const auto& bindings = candidate.info.bindings;
			return candidate.specialization == specialization &&
			       bindings.push_data_start_dword ==
			           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor, bindings.ShaderDataDwords());
		});
		if (module == modules.end()) return false;
		packed = std::move(module->packed);
		words  = module->words;
		info   = std::move(module->info);
		m_ahead->ready_words -= words;
		modules.erase(module);
	}
	spirv.resize(words);
	const auto size = ZSTD_decompress(spirv.data(), spirv.size() * sizeof(uint32_t), packed.data(), packed.size());
	return !ZSTD_isError(size) && size == spirv.size() * sizeof(uint32_t);
}

void PipelineCache::RunComputeAhead() {
	for (;;) {
		std::shared_ptr<Ahead::Job> job;
		{
			std::unique_lock lock(m_ahead->mutex);
			m_ahead->changed.wait(lock, [this] { return m_ahead->stop || !m_ahead->prepares.empty() || !m_ahead->jobs.empty(); });
			if (m_ahead->stop) return;
			auto& queue = !m_ahead->prepares.empty() ? m_ahead->prepares : m_ahead->jobs;
			job         = std::move(queue.front());
			queue.pop_front();
			m_ahead->running.push_back(job);
		}
		m_ahead->changed.notify_all(); // room in the queue for a scanner that waits for it (RoomForScanner)
		const auto handed = job->prepare ? job : nullptr;
		auto       step   = AheadStep::Build;
		try {
		Common::ScopedExitThrows throws;
		// A draw or dispatch the scanner handed over: prepared here, then built here.
		if (handed != nullptr) {
			PipelineCache::AheadSubmission(handed->submission); // 0 for a handed-over one: no skip
			std::shared_ptr<Ahead::Job> build;
			step = handed->pixel ? PrepareGraphicsAhead(*handed->registers, *handed->context, handed, build)
			                     : PrepareComputeAhead(*handed->registers, handed->input_info, handed, build);
			if (step == AheadStep::Build) job = std::move(build);
		}
		if (step == AheadStep::Build && job->pixel) RunGraphicsAhead(*job);
		if (step == AheadStep::Build && !job->pixel) {
		auto options = ComputeOptions(job->params, job->input_info, m_program_cache->profile);
		if (job->translated == nullptr) {
			job->translated = std::make_shared<ShaderRecompiler::TranslateResult>(ShaderRecompiler::TranslateProgram(
			    ShaderRecompiler::PrepareShaderSource(job->params.code, options).decoded, options));
		}
		std::vector<uint32_t> key;
		BuildStageStaticKey(job->input_info, key);
		NoteReadyPlan(ShaderType::Compute, job->params.hash, key, job->translated->program);
		auto result = ShaderRecompiler::CompileProgram(std::move(*job->translated), options, job->specialization, 0);
		const auto module = CompileSPV(result.spirv, m_graphics.device);
		if (module != nullptr) {
			auto info = std::move(result.program).TakeCompiledInfo();
			NoteReady(ShaderType::Compute, job->params.hash, std::move(key), job->specialization, result.spirv, info);
			ShaderRecompiler::IR::ResourceSnapshot resources; // the stage must have one; layouts read the program only
			job->input_info.stage = {.program = &info, .resources = &resources};
			Pipeline   pipeline;
			CreatePipelineInternal(m_graphics, pipeline, job->input_info, module, m_driver_cache);
			m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline.fragment_set_layout, nullptr);
			for (const auto layout: pipeline.group_set_layouts) m_graphics.device.destroyDescriptorSetLayout(layout, nullptr);
			m_graphics.device.destroyShaderModule(module, nullptr);
		}
		}
		} catch (const Common::ExitException& error) {
			LOGF("Compute ahead: gave up CS=0x%016" PRIx64 ": %s\n", job->params.hash, error.message.c_str());
		}
		{
			std::lock_guard lock(m_ahead->mutex);
			if (step == AheadStep::Build) m_ahead->built++;
			// Parked, it left running in TakeProgram and may be on another worker already.
			if (step != AheadStep::Parked) std::erase(m_ahead->running, job);
			if (handed != nullptr && step != AheadStep::Parked) {
				std::erase(m_ahead->running, handed);
				m_ahead->handing.erase(handed->handed_key);
				--*handed->handed_count;
			}
		}
		m_ahead->changed.notify_all();
	}
}

// A worker: the pixel shader's module, then a pipeline with each template of its vertex shader.
void PipelineCache::RunGraphicsAhead(AheadJob& job) {
	auto options = PixelOptions(job.params, job.pixel_info, m_program_cache->profile, m_graphics.SplitsFragmentSet());
	if (job.translated == nullptr) {
		job.translated = std::make_shared<ShaderRecompiler::TranslateResult>(ShaderRecompiler::TranslateProgram(
		    ShaderRecompiler::PrepareShaderSource(job.params.code, options).decoded, options));
	}
	std::vector<uint32_t> key;
	BuildStageStaticKey(job.pixel_info, key);
	NoteReadyPlan(ShaderType::Pixel, job.params.hash, key, job.translated->program);
	// The pixel stage's push data starts where GetGraphicsPrograms puts it.
	auto result = ShaderRecompiler::CompileProgram(std::move(*job.translated), options, job.specialization,
	                                               ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
	const auto module = CompileSPV(result.spirv, m_graphics.device);
	if (module == nullptr) return;
	auto                                   info = std::move(result.program).TakeCompiledInfo();
	NoteReady(ShaderType::Pixel, job.params.hash, std::move(key), job.specialization, result.spirv, info);
	ShaderRecompiler::IR::ResourceSnapshot resources;
	job.pixel_info.stage = {.program = &info, .resources = &resources};
	std::vector<std::shared_ptr<const Ahead::GraphicsTemplate>> list;
	std::shared_ptr<const Ahead::GraphicsTemplate>              state; // the draw's pass and blend mode, from any vertex shader
	{
		std::lock_guard lock(m_ahead->mutex);
		if (const auto found = m_ahead->templates.find(job.vertex_hash); found != m_ahead->templates.end()) list = found->second;
		if (const auto found = m_ahead->by_state.find(job.state_signature); found != m_ahead->by_state.end()) state = found->second;
	}
	// The GPU thread puts the vertex stage's push data after the pixel shader's (GetGraphicsPrograms),
	// so only a vertex module at that place can be the one it builds with: those first.
	uint32_t pixel_cursor = ShaderRecompiler::IR::PushData::MeshDrawDwordCount;
	info.bindings.AdvancePushData(pixel_cursor);
	const auto in_place = [&](const Ahead::GraphicsTemplate& tmpl) {
		const auto& bindings = tmpl.vertex_programs[0].bindings;
		const auto  at       = [&](uint32_t cursor) {
			return bindings.push_data_start_dword == ShaderRecompiler::IR::PushData::StartFor(cursor, bindings.ShaderDataDwords());
		};
		if (pixel_cursor != ShaderRecompiler::IR::PushData::MeshDrawDwordCount) return at(pixel_cursor);
		return tmpl.mesh ? at(ShaderRecompiler::IR::PushData::MeshDrawDwordCount) : at(0) || at(1);
	};
	// An exact job builds once: the template whose vertex module is the draw's own, with the draw's render
	// targets, static state and vertex input made as the GPU thread makes them (MakeGraphicsKey).
	auto                                   reason = job.exact_reason;
	std::optional<Ahead::GraphicsTemplate> exact;
	vk::ShaderModule                       vertex_module = nullptr; // a vertex module compiled here
	if (reason == Ahead::Exact) {
		const auto vertex_cursor =
		    pixel_cursor != ShaderRecompiler::IR::PushData::MeshDrawDwordCount ? pixel_cursor : job.vertex_push_cursor;
		const auto match = std::ranges::find_if(list, [&](const auto& tmpl) {
			const auto& bindings = tmpl->vertex_programs[0].bindings;
			return tmpl->exact_vertex && tmpl->vertex_static_key == job.vertex_static_key &&
			       tmpl->vertex_specialization == job.vertex_specialization &&
			       bindings.push_data_start_dword ==
			           ShaderRecompiler::IR::PushData::StartFor(vertex_cursor, bindings.ShaderDataDwords());
		});
		// No template has the draw's vertex module yet: compile it here, as the GPU thread's program cache will
		// (the same translation, specialization and push data place, so the same SPIR-V).
		std::optional<Ahead::GraphicsTemplate> own;
		if (match == list.end()) {
			auto voptions = VertexOptions(job.vertex_params, job.vertex_info, m_program_cache->profile);
			if (job.vertex_translated == nullptr) {
				job.vertex_translated = std::make_shared<ShaderRecompiler::TranslateResult>(ShaderRecompiler::TranslateProgram(
				    ShaderRecompiler::PrepareShaderSource(job.vertex_params.code, voptions).decoded, voptions));
			}
			const auto stage = job.vertex_info.logical_stage;
			NoteReadyPlan(stage, job.vertex_params.hash, job.vertex_static_key, job.vertex_translated->program);
			auto vertex = ShaderRecompiler::CompileProgram(std::move(*job.vertex_translated), voptions,
			                                               job.vertex_specialization, vertex_cursor);
			vertex_module = CompileSPV(vertex.spirv, m_graphics.device);
			if (vertex_module != nullptr) {
				own.emplace();
				own->vertex_programs[0]   = std::move(vertex.program).TakeCompiledInfo();
				NoteReady(stage, job.vertex_params.hash, job.vertex_static_key, job.vertex_specialization, vertex.spirv,
				          own->vertex_programs[0]);
				own->vertex_info[0]       = job.vertex_info;
				own->vertex_info[0].stage = {.program = &own->vertex_programs[0], .resources = &own->resources};
				own->programs.vertex[0]   = {.id         = OwnVertexModuleId,
				                             .module     = vertex_module,
				                             .spirv_hash = XXH3_64bits(vertex.spirv.data(), vertex.spirv.size() * sizeof(uint32_t))};
				own->stages               = 1;
				own->mesh                 = job.mesh;
			}
		}
		const Ahead::GraphicsTemplate* base = match != list.end() ? match->get() : own ? &*own : nullptr;
		if (base == nullptr) {
			reason = Ahead::NoTemplate;
		} else {
			if (own) reason = Ahead::OwnVertex;
			const auto& ctx      = *job.context;
			uint32_t    mrt_mask = 0;
			for (const auto& output: info.info.outputs) {
				if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
				    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
					mrt_mask |= 1u << output.location;
				}
			}
			mrt_mask &= DrawColorOutputMask(ctx);
			// As ResolveRenderColorTarget and ResolveRenderDepthTarget find them; the ids only say "bound".
			std::array<RenderColorInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
			uint32_t                                                   color_count = 0;
			for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
				const auto& rt = ctx.GetRenderTarget(slot);
				if ((mrt_mask & (1u << slot)) == 0 || rt.base.addr == 0 ||
				    render_target_mask_slot(ctx.GetRenderTargetMask(), slot) == 0) {
					continue;
				}
				const auto format = TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order);
				auto&      color  = colors[color_count++];
				color.target_slot           = slot;
				color.desc.view_info.format = format.format;
				color.desc.info.guest_format = format.guest_format;
				color.desc.info.samples     = render_sample_count(rt.attrib.num_fragments);
				color.export_mapping        = format.export_mapping;
				color.image_id              = ImageId {0};
			}
			RenderDepthInfo depth;
			if (DrawDepthTargetFormat(m_graphics, ctx, depth.desc.view_info.format, depth.desc.info.samples)) {
				depth.image_id = ImageId {0};
			} else {
				depth.desc.view_info.format = vk::Format::eUndefined;
			}
			// As PrepareDrawRenderState: with mixed sample counts only the raster's attachments stay.
			const auto pass_samples = depth.image_id ? depth.desc.info.samples : colors[0].desc.info.samples;
			if (std::any_of(colors.begin(), colors.begin() + color_count,
			                [&](const RenderColorInfo& color) { return color.desc.info.samples != pass_samples; })) {
				const auto raster = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
				const auto kept   = std::remove_if(colors.begin(), colors.begin() + color_count,
				                                   [&](const RenderColorInfo& color) { return color.desc.info.samples != raster; });
				color_count       = static_cast<uint32_t>(kept - colors.begin());
				if (depth.image_id && depth.desc.info.samples != raster) depth = {};
			}
			exact.emplace(*base);
			// The stages point into their template, which may be `own` or may go with list.clear() below.
			for (uint32_t i = 0; i < exact->stages; i++) {
				exact->vertex_info[i].stage = {.program = &exact->vertex_programs[i], .resources = &exact->resources};
			}
			auto programs  = exact->programs;
			programs.pixel = {.id = UINT64_MAX, .module = module};
			const auto key = MakeGraphicsKey(std::span(colors.data(), color_count), depth, ctx, job.rect_list,
			                                 &job.pixel_info, job.topology, false, programs);
			exact->rendering     = key.rendering;
			exact->static_params = key.static_params;
			exact->vertex_input  = job.vertex_input;
		}
	}
	{
		std::lock_guard lock(m_ahead->mutex);
		m_ahead->exact[reason]++;
		if (const auto& counts = m_ahead->exact; std::accumulate(counts.begin(), counts.end(), uint64_t {0}) % 256 == 0) {
			LOGF("Graphics ahead: %" PRIu64 " jobs built exactly (%" PRIu64 " with their own vertex module); with guesses: "
			     "tessellation %" PRIu64 ", restart %" PRIu64 ", vertex stages %" PRIu64 ", unsupported %" PRIu64
			     ", vertex resources %" PRIu64 ", no vertex module %" PRIu64 "\n",
			     counts[Ahead::Exact] + counts[Ahead::OwnVertex], counts[Ahead::OwnVertex], counts[Ahead::Tessellation],
			     counts[Ahead::Restart], counts[Ahead::Prepare], counts[Ahead::Unsupported], counts[Ahead::Materialize],
			     counts[Ahead::NoTemplate]);
		}
	}
	// shortcut: 3 states, the latest first and one per vertex module first: a vertex shader's
	// modules alternate (one hash, other specializations), and with the oldest 3 the GPU thread's
	// build missed in 110 of 284 creates; a job of 8 kept the GPU thread waiting.
	if (exact) list.clear(); // no guesses
	std::ranges::reverse(list);
	std::ranges::stable_partition(list, [&](const auto& tmpl) { return in_place(*tmpl); });
	std::vector<std::shared_ptr<const Ahead::GraphicsTemplate>> chosen;
	for (const auto& tmpl: list) {
		if (chosen.size() < 3 && std::ranges::none_of(chosen, [&](const auto& other) {
			    return other->programs.vertex[0].id == tmpl->programs.vertex[0].id;
		    }))
			chosen.push_back(tmpl);
	}
	for (const auto& tmpl: list) {
		if (chosen.size() < 3 && std::ranges::find(chosen, tmpl) == chosen.end()) chosen.push_back(tmpl);
	}
	// First the best vertex module with the draw's own pass and blend mode, then the states seen.
	std::vector<std::pair<const Ahead::GraphicsTemplate*, const Ahead::GraphicsTemplate*>> builds;
	if (state != nullptr && !chosen.empty() && state.get() != chosen.front().get() &&
	    (state->rendering != chosen.front()->rendering || !(state->static_params == chosen.front()->static_params)))
		builds.emplace_back(chosen.front().get(), state.get());
	for (const auto& tmpl: chosen) {
		if (builds.size() < 3) builds.emplace_back(tmpl.get(), tmpl.get());
	}
	if (exact) builds = {{&*exact, &*exact}};
	{
		std::lock_guard lock(m_ahead->mutex);
		for (const auto& [tmpl, with]: builds) {
			auto& key           = job.builds.emplace_back();
			key.rendering       = with->rendering;
			key.vertex_input    = tmpl->vertex_input;
			key.static_params   = with->static_params;
			for (uint32_t i = 0; i < key.vertex_shader_ids.size(); i++) key.vertex_shader_ids[i] = tmpl->programs.vertex[i].id;
		}
		job.chosen = true;
	}
	m_ahead->changed.notify_all();
	for (const auto& [tmpl, with]: builds) {
		auto programs  = tmpl->programs;
		programs.pixel = {.id         = UINT64_MAX,
		                  .module     = module,
		                  .spirv_hash = XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t))};
		Pipeline   pipeline;
		CreatePipelineInternal(m_graphics, pipeline, with->rendering, tmpl->vertex_input,
		                       std::span(tmpl->vertex_info.data(), tmpl->stages), &job.pixel_info, programs,
		                       with->static_params, m_driver_cache,
		                       m_graphics.SplitsFragmentSet() ? m_libraries.get() : nullptr);
		m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
		m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
		m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
		m_graphics.device.destroyDescriptorSetLayout(pipeline.fragment_set_layout, nullptr);
		for (const auto layout: pipeline.group_set_layouts) m_graphics.device.destroyDescriptorSetLayout(layout, nullptr);
		{
			std::lock_guard lock(m_ahead->mutex);
			job.built++;
		}
		m_ahead->changed.notify_all();
	}
	m_graphics.device.destroyShaderModule(module, nullptr);
	if (vertex_module != nullptr) m_graphics.device.destroyShaderModule(vertex_module, nullptr);
}

// The GPU thread builds this draw's or dispatch's pipeline itself (identity as its job has it): a job still queued
// for it is dropped, and none is made for it later, since a worker's build would only repeat this one and hold a worker
// a whole job long. A job a worker has begun is waited for instead (WaitForGraphicsAhead, WaitForComputeAhead).
void PipelineCache::NoteGpuBuild(uint64_t identity) {
	if (identity == 0) return;
	std::lock_guard lock(m_ahead->mutex);
	m_ahead->gpu_built.insert(identity);
	m_ahead->dropped += std::erase_if(m_ahead->jobs, [&](const auto& job) { return !job->prepare && job->identity == identity; });
}

// The GPU thread: the identity an exact graphics job has for this draw (GraphicsAheadIdentity); 0 for tessellation and
// for a module without a specialization.
uint64_t PipelineCache::GraphicsIdentity(std::span<const ShaderVertexInputInfo> vertex_info, const ShaderPixelInputInfo& pixel_info,
                                         const GraphicsPrograms& programs, const PipelineVertexInputState& vertex_input,
                                         uint64_t state_signature) const {
	const auto& of = m_program_cache->specialization_of;
	if (programs.VertexStageCount() != 1 || vertex_info.empty()) return 0;
	const auto pixel = of.find(programs.pixel.id), vertex = of.find(programs.vertex[0].id);
	if (pixel == of.end() || vertex == of.end()) return 0;
	std::vector<uint32_t> pixel_key, vertex_key;
	BuildStageStaticKey(pixel_info, pixel_key);
	BuildStageStaticKey(vertex_info[0], vertex_key);
	return GraphicsAheadIdentity(pixel_info.stage.program->shader_hash, pixel_key, pixel->second,
	                             vertex_info[0].stage.program->shader_hash, vertex_key, vertex->second, vertex_input,
	                             state_signature);
}

// A worker reads a shader's code at its guest address. Where that shares a page with bytes the GPU wrote the read faults,
// and the worker waits for a read back that only the GPU thread can do, which must not be waiting for that worker.
template <typename Done>
static void WaitForAheadWorker(std::unique_lock<std::mutex>& lock, std::condition_variable& changed, Done&& done) {
	while (!changed.wait_for(lock, std::chrono::milliseconds(1), done)) {
		lock.unlock();
		GuestGpu::RunReadBacks();
		lock.lock();
	}
}

// The GPU thread: a worker building this dispatch's pipeline is waited for; its pipeline is then in the driver cache,
// and the GPU thread's own build of the same SPIR-V a hit. One still preparing (identity 0) may be it too. A job still
// queued is not waited for (NoteGpuBuild drops it): the GPU thread's own build is no slower than the worker's whole job.
void PipelineCache::WaitForComputeAhead(uint64_t shader_hash, uint64_t identity) {
	if (m_ahead == nullptr) return;
	NoteGpuBuild(identity);
	std::unique_lock lock(m_ahead->mutex);
	const auto building = [&] {
		return std::ranges::any_of(m_ahead->running, [&](const auto& job) {
			return !job->pixel && job->params.hash == shader_hash &&
			       (job->identity == 0 || identity == 0 || job->identity == identity);
		});
	};
	if (!building()) return;
	const auto start = std::chrono::steady_clock::now();
	WaitForAheadWorker(lock, m_ahead->changed, [&] { return m_ahead->stop || !building(); });
	m_ahead->waits++;
	m_ahead->waited_ms += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
}

// The GPU thread: a worker already building this draw's pipeline is waited for until it is in the driver cache.
// A job still queued, one whose guesses leave this pipeline out, or one with it still to come after other builds
// is not: the GPU thread's own build is no slower.
void PipelineCache::WaitForGraphicsAhead(uint64_t pixel_hash, uint64_t identity, const GraphicsPipelineKey& key) {
	if (m_ahead == nullptr) return;
	auto stages         = key;
	stages.ps_shader_id = 0;
	std::unique_lock lock(m_ahead->mutex);
	const auto building = [&] {
		return std::ranges::any_of(m_ahead->running, [&](const auto& job) {
			if (!job->pixel || job->params.hash != pixel_hash) return false;
			// Another exact draw of the pixel shader: another pipeline.
			if (identity != 0 && job->identity != 0 && job->identity != identity) return false;
			// Still preparing, or on the pixel shader's module: its first build may be this one.
			if (!job->chosen) return true;
			const auto found = std::ranges::find_if(job->builds, [&](const GraphicsPipelineKey& build) {
				if (build.vertex_shader_ids[0] != OwnVertexModuleId) return build == stages;
				auto own                 = build;
				own.vertex_shader_ids[0] = stages.vertex_shader_ids[0];
				return own == stages;
			});
			return found != job->builds.end() && job->built == static_cast<size_t>(found - job->builds.begin());
		});
	};
	if (!building()) return;
	const auto start = std::chrono::steady_clock::now();
	WaitForAheadWorker(lock, m_ahead->changed, [&] { return m_ahead->stop || !building(); });
	m_ahead->waits++;
	m_ahead->waited_ms += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

// A draw's pipeline key but its vertex input, from its registers, render targets and stages. The GPU
// thread's draws (GetGraphicsPipeline) and graphics ahead's exact builds both make it here, so the two agree.
PipelineCache::GraphicsPipelineKey PipelineCache::MakeGraphicsKey(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth, const HW::Context& ctx, bool rect_list,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology, bool primitive_restart_enable,
    const GraphicsPrograms& programs) const {
	const bool             ps_active   = ps_input_info != nullptr;
	const auto             color_count = static_cast<uint32_t>(colors.size());
	const HW::ModeControl& mc          = ctx.GetModeControl();

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_active ? programs.pixel.id : 0;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);
	return key;
}

// The vertex input state of a draw's pipeline: its vertex buffers' strides and attribute offsets.
PipelineVertexInputState PipelineCache::MakeVertexInput(const ShaderVertexInputInfo& vs_input_info, bool mesh) {
	PipelineVertexInputState vertex_input;
	if (!mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}
	return vertex_input;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	auto& ctx = command.GetRegisters();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	auto key = MakeGraphicsKey(colors, depth, ctx, Prospero::IsRectList(command.GetUserConfig().GetPrimType()),
	                           ps_input_info, topology, primitive_restart_enable, programs);
	key.vertex_input    = MakeVertexInput(vs_input_info, vs_input_info.stage.program->stage == ShaderType::Mesh);
	auto& static_params = key.static_params;
	auto& rendering     = key.rendering;

	if (m_last_graphics_key != nullptr && key == *m_last_graphics_key) return *m_last_graphics_pipeline;
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		m_last_graphics_key      = &iter->first;
		m_last_graphics_pipeline = iter->second.get();
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	// A precompile worker creating this pipeline now is waited for; one still queued is dropped (shaderPrecompile.h).
	// One the driver cache holds (precompiled, saved or this session's) is a hit: no ahead worker is waited for, as
	// one may still translate its pixel shader for seconds (a first cold run with a precompile: 46 s of such waits).
	const auto record     = MakePipelineRecord(key, programs, ps_active);
	const auto record_key = record ? record->Key() : 0;
	const bool precompiled =
	    record_key != 0 && ((m_precompiler != nullptr && m_precompiler->BeforeCreate(record_key)) ||
	                        (m_warm != nullptr && m_warm->Contains(record_key, true)));
	// One a precompile worker created in this process is taken when its stages are this draw's own (same SPIR-V).
	bool taken = false;
	if (record_key != 0) {
		std::lock_guard lock(m_ready_mutex);
		if (const auto found = m_ready_pipelines.find(record_key); found != m_ready_pipelines.end()) {
			bool same = found->second.spirv[3] == (ps_active ? programs.pixel.spirv_hash : 0);
			for (uint32_t i = 0; i < 3; i++) {
				same = same && found->second.spirv[i] == (i < programs.VertexStageCount() ? programs.vertex[i].spirv_hash : 0);
			}
			if (same) {
				*cached = found->second.pipeline;
				m_ready_pipelines.erase(found);
				taken   = true;
			}
		}
	}
	if (m_ahead != nullptr) {
		const auto signature = DrawStateSignature(ctx, command.GetUserConfig().GetPrimType());
		RecordGraphicsTemplate(key, vertex_info, programs, signature);
		if (ps_active) {
			const auto identity = GraphicsIdentity(vertex_info, *ps_input_info, programs, key.vertex_input, signature);
			NoteGpuBuild(identity);
			if (!precompiled && !taken) WaitForGraphicsAhead(ps_input_info->stage.program->shader_hash, identity, key);
		}
	}
	if (!taken) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, static_params, m_driver_cache,
		                       m_graphics.SplitsFragmentSet() ? m_libraries.get() : nullptr);
	}
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	if (record_key != 0) {
		if (m_warm != nullptr) m_warm->Mark(record_key, true);
		if (m_journal != nullptr && m_recorded_pipelines.insert(record_key).second) {
			m_journal->Append(*record);
			if (ps_active) m_journal->Append(MakeTemplate(command, colors, depth, topology, primitive_restart_enable, *record));
		}
	}
	CheckpointDriverCache();

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	// A compute record is its own pipeline (shaderPrecompile.h); as for graphics, one the driver cache holds waits for
	// no ahead worker.
	const auto record      = m_program_cache->record_of.find(compute_program.id);
	const auto record_key  = record != m_program_cache->record_of.end() ? record->second : 0;
	const bool precompiled = record_key != 0 && ((m_precompiler != nullptr && m_precompiler->BeforeCreate(record_key)) ||
	                                             (m_warm != nullptr && m_warm->Contains(record_key, false)));
	if (m_ahead != nullptr) {
		std::vector<uint32_t> static_key;
		BuildStageStaticKey(input_info, static_key);
		const auto& of    = m_program_cache->specialization_of;
		const auto  found = of.find(compute_program.id);
		const auto  hash  = input_info.stage.program->shader_hash;
		const auto  identity = found != of.end() ? ComputeAheadIdentity(hash, static_key, found->second) : 0;
		if (precompiled) {
			NoteGpuBuild(identity);
		} else {
			WaitForComputeAhead(hash, identity);
		}
	}

	auto cached = std::make_unique<Pipeline>();
	// One a precompile worker created in this process is taken when its SPIR-V is this dispatch's own.
	bool taken = false;
	if (record_key != 0) {
		std::lock_guard lock(m_ready_mutex);
		const std::array<uint64_t, 4> spirv {compute_program.spirv_hash};
		if (const auto found = m_ready_pipelines.find(record_key);
		    found != m_ready_pipelines.end() && found->second.spirv == spirv) {
			*cached = found->second.pipeline;
			m_ready_pipelines.erase(found);
			taken = true;
		}
	}
	if (!taken) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                       m_driver_cache);
	}
	if (record_key != 0 && m_warm != nullptr) m_warm->Mark(record_key, false);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	CheckpointDriverCache();

	return *iter->second;
}
} // namespace Libs::Graphics
