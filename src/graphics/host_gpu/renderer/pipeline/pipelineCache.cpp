#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/speculation.h"
#include "graphics/host_gpu/renderer/tableMode.h"
#include "graphics/host_gpu/gpuPhases.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/frameGraph.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/host_gpu/renderer/pipeline/spirvCoverageMask.h"
#include "graphics/host_gpu/renderer/pipeline/spirvMeshTest.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtNative.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <thread>
#include <unordered_map>
#include <deque>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

bool ShaderFailureNonFatal() {
	static const bool value = true;
	return value;
}

namespace {

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
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
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
	// No emulator revision: the driver keys every pipeline by its whole create info, SPIR-V
	// included, so a new build reuses the pipelines whose shaders did not change. Keying on the
	// revision threw the whole cache away at every rebuild and release, and the first start then
	// compiled everything behind a black screen.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

// The largest driver cache file kept (and loaded): pipelines of older builds stay in the file. The
// limit was 1 GiB until round 64, and GT7's grew past it after two weeks of races: the next run
// refused it, compiled every pipeline again (60,000 draws were skipped while they compiled in one
// race, the track among them) and kept four cores busy for minutes. 3.75 GiB: under 4 GiB, which
// the file reads and writes count in.
constexpr uint64_t DriverCacheMaxBytes = uint64_t {3840} << 20u;

// Reads and writes of the cache file, 256 MiB at a time (one call for a gigabyte-sized buffer is
// the kind of request a file system may refuse).
constexpr uint64_t DriverCacheChunk = uint64_t {256} << 20u;

uint64_t ReadInChunks(Common::File& file, uint8_t* data, uint64_t size) {
	uint64_t done = 0;
	while (done < size) {
		const auto part = static_cast<uint32_t>(std::min(size - done, DriverCacheChunk));
		uint32_t   read = 0;
		file.Read(data + done, part, &read);
		done += read;
		if (read != part) {
			break;
		}
	}
	return done;
}

uint64_t WriteInChunks(Common::File& file, const uint8_t* data, uint64_t size) {
	uint64_t done = 0;
	while (done < size) {
		const auto part    = static_cast<uint32_t>(std::min(size - done, DriverCacheChunk));
		uint32_t   written = 0;
		file.Write(data + done, part, &written);
		done += written;
		if (written != part) {
			break;
		}
	}
	return done;
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

// _ShaderCache/<title id>, next to _PipelineCache; empty without a title id.
std::filesystem::path ShaderDiskCacheDirectory() {
	const auto title_id = PipelineCacheTitleId();
	return title_id.empty() ? std::filesystem::path() : std::filesystem::path("_ShaderCache") / title_id;
}

// Round rtexp: expansions of shaders with calls whose source missed the shader disk cache (each
// translated in game), for the "Call expansions" line at the end of the run.
static std::atomic<uint64_t> g_expansions_in_game {0};

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// What the shader disk cache's folder holds at the start (where it is, its files and their size):
// round 68's runs found it empty, every shader was translated again, and why could not be told.
void LogShaderCacheFolder(const std::filesystem::path& directory) {
	if (directory.empty()) {
		return;
	}
	std::error_code error;
	const auto      absolute = std::filesystem::absolute(directory, error);
	const auto      shown    = (error ? directory : absolute).string();
	if (!std::filesystem::is_directory(directory, error)) {
		PipelineCacheLog("Shader disk cache folder: {} does not exist yet (every shader is "
		                 "translated, and saved there)",
		                 shown);
		return;
	}
	uint64_t files = 0;
	uint64_t bytes = 0;
	for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end;
	     it.increment(error)) {
		std::error_code entry_error;
		if (it->is_regular_file(entry_error)) {
			files++;
			const auto size = it->file_size(entry_error);
			bytes += entry_error ? 0 : size;
		}
	}
	PipelineCacheLog("Shader disk cache folder: {}: {} files, {} MB{}", shown, files, bytes >> 20u,
	                 error ? " (the listing stopped early)" : "");
}

// The shader disk cache keeps every entry it ever wrote, and each build whose recompiler differs
// writes a new set: the folder held 342,000 files and 102 GB in round 69, nearly all of them for
// recompilers long gone. An entry file starts with a header (magic, format, sizes, hash: 32 bytes)
// and then its key, whose first 16 bytes are the magic, the format and the recompiler's version
// (ShaderDiskCache::SessionKey). A file whose first bytes differ from this build's can never be a
// hit for it (Store::Load rejects it), and is removed, in the background, 30 s after the start,
// once per recompiler version (a "pruned_<version>" file marks it done). Files with this
// recompiler's version are kept whatever else their key holds (another GPU, other switches).
// KYTY_SHADER_CACHE_PRUNE=0 keeps everything.
using ShaderCachePrefix = std::array<uint8_t, 16>;

bool ShaderCachePruneEnabled() {
	const char* value = std::getenv("KYTY_SHADER_CACHE_PRUNE");
	return value == nullptr || value[0] != '0';
}

void PruneShaderCache(std::stop_token stop, std::filesystem::path directory,
                      ShaderCachePrefix prefix) {
	constexpr size_t HeaderBytes = 32;
	std::error_code  error;
	const auto       version = ShaderDiskCache::SourceVersion();
	const auto       marker  = directory / fmt::format("pruned_{:016x}", version);
	if (std::filesystem::exists(marker, error)) {
		return;
	}
	for (int i = 0; i < 300; i++) {
		if (stop.stop_requested()) {
			return;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	if (!std::filesystem::is_directory(directory, error)) {
		return;
	}
	// Round 71: a run that stops before the end (the game closed) leaves the name of the last file
	// it kept in pruning_<version>, and the next run goes on after it instead of opening every
	// file again from the first (round 70 got through a fifth of 344,000 files in a run).
	const auto  progress = directory / fmt::format("pruning_{:016x}", version);
	std::string resume;
	{
		std::ifstream file(progress);
		std::getline(file, resume);
	}
	uint64_t                           removed       = 0;
	uint64_t                           removed_bytes = 0;
	uint64_t                           kept          = 0;
	uint64_t                           examined      = 0;
	uint64_t                           skipped       = 0;
	std::string                        last_kept;
	std::vector<std::filesystem::path> old_markers;
	const auto                         save_progress = [&] {
        if (!last_kept.empty()) {
            std::ofstream(progress, std::ios::trunc) << last_kept << "\n";
        }
	};
	// Two passes at most: the first skips up to the resume point; when the resume file is not
	// found (removed by hand), the second goes through everything.
	for (int pass = 0; pass < 2; pass++) {
		bool skipping = !resume.empty() && pass == 0;
		bool found    = false;
		for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end;
		     it.increment(error)) {
			if (stop.stop_requested()) {
				save_progress();
				PipelineCacheLog("Shader disk cache: removing old entries paused at the end of the "
				                 "run: {} files ({} MB) removed, {} kept; the next run goes on",
				                 removed, removed_bytes >> 20u, kept);
				return;
			}
			const auto& path = it->path();
			const auto  name = path.filename().string();
			if (name.starts_with("pruned_") || name.starts_with("pruning_")) {
				if (path.filename() != marker.filename() &&
				    path.filename() != progress.filename()) {
					old_markers.push_back(path);
				}
				continue;
			}
			if (skipping) {
				skipped++;
				if (name == resume) {
					skipping = false;
					found    = true;
				}
				continue;
			}
			std::error_code entry_error;
			if (path.extension() != ".bin" || !it->is_regular_file(entry_error)) {
				continue;
			}
			const auto size = it->file_size(entry_error);
			if (entry_error) {
				continue;
			}
			std::array<uint8_t, HeaderBytes + 16> head {};
			{
				std::ifstream file(path, std::ios::binary);
				if (!file) {
					continue;
				}
				file.read(reinterpret_cast<char*>(head.data()),
				          static_cast<std::streamsize>(head.size()));
			}
			const bool current =
			    size >= head.size() && std::memcmp(head.data(), prefix.data(), 8) == 0 &&
			    std::memcmp(head.data() + HeaderBytes, prefix.data(), prefix.size()) == 0;
			if (current) {
				kept++;
				last_kept = name;
			} else if (std::filesystem::remove(path, entry_error)) {
				removed++;
				removed_bytes += size;
			}
			// Gently: a pause every 64 files; the resume point every 4096; a line every 32768.
			if ((++examined & 63u) == 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			if ((examined & 4095u) == 0) {
				save_progress();
			}
			if ((examined & 32767u) == 0) {
				PipelineCacheLog("Shader disk cache: removing old entries: {} files examined, {} "
				                 "({} MB) removed so far",
				                 examined, removed, removed_bytes >> 20u);
			}
		}
		if (error || pass == 1 || resume.empty() || found) {
			break;
		}
		// The resume point was not there: everything again, from the first file.
		PipelineCacheLog("Shader disk cache: the resume point of removing old entries is gone; "
		                 "starting over");
		old_markers.clear();
		skipped = 0;
	}
	if (error) {
		save_progress();
		PipelineCacheLog("Shader disk cache: removing old entries stopped early ({}): {} files "
		                 "({} MB) removed, {} kept",
		                 error.message(), removed, removed_bytes >> 20u, kept);
		return;
	}
	for (const auto& old_marker: old_markers) {
		std::filesystem::remove(old_marker, error);
	}
	std::filesystem::remove(progress, error);
	std::ofstream(marker) << "entries of other recompilers removed\n";
	PipelineCacheLog("Shader disk cache: {} files ({} MB) of other recompilers removed, {} kept, "
	                 "{} gone through in earlier runs (KYTY_SHADER_CACHE_PRUNE=0 keeps them)",
	                 removed, removed_bytes >> 20u, kept, skipped);
}

// The backing bytes of the guest pages one resource evaluation (MaterializeResources, a function
// expansion) reads, found once per page: every SRT dword read otherwise took the backing store's
// lock, two map lookups and the GPU-dirty range queries (a third of the GPU thread in the
// jungle). Only pages with no GPU-written byte are cached; within one synchronous evaluation on
// the GPU thread nothing marks them GPU-written. A page that is not clean stays on the exact path.
struct GuestPageReadCache {
	static constexpr size_t Capacity = 16;
	struct Entry {
		uint64_t       page    = 0;
		const uint8_t* backing = nullptr;
	};
	std::array<Entry, Capacity> entries {};
	size_t                      count = 0;
	size_t                      next  = 0;

	// The page's backing bytes when it is GPU-clean, else null (SrtRuntime::map_clean_page).
	const uint8_t* Map(uint64_t page) {
		for (size_t i = 0; i < count; i++) {
			if (entries[i].page == page) {
				return entries[i].backing;
			}
		}
		auto& slot = count < Capacity ? entries[count++] : entries[next++ % Capacity];
		slot       = {.page    = page,
		              .backing = Libs::LibKernel::Memory::FindGpuCleanBacking(page, TRACKER_PAGE_SIZE)};
		return slot.backing;
	}

	// True when the page is GPU-clean and the bytes were copied from its backing.
	bool Read(uint64_t address, std::span<uint32_t> values) {
		const auto page = Common::AlignDown(address, TRACKER_PAGE_SIZE);
		if (Common::AlignDown(address + values.size_bytes() - 1, TRACKER_PAGE_SIZE) != page) {
			return false;
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
			slot       = {.page    = page,
			              .backing = Libs::LibKernel::Memory::FindGpuCleanBacking(page,
			                                                                     TRACKER_PAGE_SIZE)};
			entry      = &slot;
		}
		if (entry->backing == nullptr) {
			return false;
		}
		std::memcpy(values.data(), entry->backing + (address - page), values.size_bytes());
		return true;
	}
};

bool UsesShaderClock(const ShaderRecompiler::IR::Program& program) {
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == ShaderRecompiler::IR::ValueOpcode::ReadClockRealtime64) {
				return true;
			}
		}
	}
	return false;
}

// Diagnostics: the shader whose resources are being evaluated, and how often its SRT reads
// hit memory the GPU owns (each such read drains the GPU).
thread_local uint64_t t_srt_shader_hash = 0;
void CountGpuOwnedSrtRead(uint64_t address, uint32_t bytes) {
	struct Entry {
		uint64_t count   = 0;
		uint64_t address = 0;
		uint32_t bytes   = 0;
	};
	static std::unordered_map<uint64_t, Entry> counts;
	static uint64_t                            total = 0;
	auto& entry   = counts[t_srt_shader_hash];
	entry.count++;
	entry.address = address;
	entry.bytes   = bytes;
	if (++total % 4096 == 0) {
		LOGF("GpuOwnedSrtReads: total=%" PRIu64 " shaders=%zu\n", total, counts.size());
		for (const auto& [hash, e]: counts) {
			LOGF("\t hash=0x%016" PRIx64 " count=%" PRIu64 " last=0x%016" PRIx64 " bytes=%u\n",
			     hash, e.count, e.address, e.bytes);
		}
	}
}

// Ordinary (raw) SRT reads: only memory the guest has committed, read through the guest
// mapping so a GPU-owned page is refreshed first. Without a reader the walker dereferenced
// whatever address a descriptor chain produced, including 0 on a path the shader never takes.
bool ReadShaderGuestMemoryRaw(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	// Bytes the GPU has not written are current in the backing store. Reading them there skips
	// the tracked-page fault, which drains the GPU to refresh whatever else on the page the GPU
	// wrote: constants that share a page with GPU-written arguments cost a drain per dispatch.
	if (Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(),
	                                                    values.size_bytes())) {
		return true;
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	if (t_srt_shader_hash != 0 && Libs::Graphics::GuestGpu::IsGpuThread()) {
		CountGpuOwnedSrtRead(address, static_cast<uint32_t>(values.size_bytes()));
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

// KYTY_SRT_GPU_FILL=0: read every flat SRT slot on the host, waiting for the GPU when it wrote
// the bytes, instead of leaving shader-only slots to the GPU (SrtGpuFill).
bool SrtGpuFillEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_GPU_FILL");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool SrtBytesGpuWritten(uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::IsGpuBufferWritten(address, size);
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	if (Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes())) {
		return true;
	}
	// The bytes are mapped but GPU-owned. Reading them through the guest mapping takes the
	// tracked-page fault, which drains the GPU and refreshes the page, so the value read is
	// the one the shader would see. Before, this only ever succeeded because the aggressive
	// garbage collector happened to have downloaded the range first.
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

// The reader of a stage walked in the parallel window (StagePrepWorker): bytes no GPU work has
// written, from the backing store, and nothing else: no download, no fault, no wait for the GPU.
// A read it cannot serve fails the walk, which the GPU thread then repeats on the exact path.
bool ReadShaderGuestMemoryProbe(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	return Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(),
	                                                       values.size_bytes());
}

// "VS", "PS", ... for zone texts.
const char* StageLabel(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "VS";
		case ShaderType::Mesh: return "MS";
		case ShaderType::Local: return "LS";
		case ShaderType::TessellationControl: return "HS";
		case ShaderType::TessellationEvaluation: return "DS";
		case ShaderType::Pixel: return "PS";
		case ShaderType::Compute: return "CS";
		default: return "??";
	}
}

// --skip-shaders and KYTY_SKIP_SHADER_HASHES="hash,hash,...": skip the draws and dispatches of
// these guest shaders, the same way a shader that fails to compile is skipped. To see what one
// shader contributes, to step past one that loses the device, or to leave out work nothing can
// use (the ray-tracing BVH updates whose only consumer gives up).
bool SkipShaderRequested(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const auto parse = [&result](std::string_view list) {
			size_t start = 0;
			while (start < list.size()) {
				const auto end = std::min(list.find(',', start), list.size());
				if (end > start) {
					result.push_back(
					    std::strtoull(std::string(list.substr(start, end - start)).c_str(), nullptr, 16));
				}
				start = end + 1;
			}
		};
		parse(Config::GetSkipShaderHashes());
		if (const char* value = std::getenv("KYTY_SKIP_SHADER_HASHES"); value != nullptr) {
			parse(value);
		}
		for (const auto hash: result) {
			LOGF("ProgramCache: skipping the draws and dispatches of shader 0x%016" PRIx64 "\n",
			     hash);
		}
		return result;
	}();
	return std::ranges::find(hashes, shader_hash) != hashes.end();
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	// KYTY_DUMP_SPIRV_HASHES="hash,hash,...": dump these shaders' SPIR-V without the full
	// graphics debug dump.
	static const std::string requested = [] {
		const char* value = std::getenv("KYTY_DUMP_SPIRV_HASHES");
		return value != nullptr ? std::string(value) : std::string();
	}();
	// "all" dumps every shader as it compiles.
	if (!Config::GraphicsDebugDumpEnabled() && requested != "all" &&
	    (requested.empty() ||
	     requested.find(fmt::format("{:016x}", shader_hash)) == std::string::npos)) {
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

// Round 93: KYTY_CALL_EVERY_TARGET (default on): a compute shader whose expansion gave up is
// expanded again with every target of its call tables (ProgramCache::ExpandShaderFunctions).
static bool ExpandEveryTarget() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_CALL_EVERY_TARGET");
		return text == nullptr || text[0] != '0';
	}();
	return value;
}

// Round 94: the most words an expansion with every table target may have (ExpandShaderFunctions).
static uint32_t EveryTargetWords() {
	static const uint32_t value = [] {
		const char* text = std::getenv("KYTY_CALL_EVERY_TARGET_WORDS");
		return text != nullptr && *text != '\0'
		           ? static_cast<uint32_t>(std::strtoul(text, nullptr, 10))
		           : 16384u;
	}();
	return value;
}

// Round 97: KYTY_CALL_KEEP_LAST (default on): GetComputeProgram runs a shader's last ready
// expansion while a newer one compiles in the background.
static bool KeepLastExpansion() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_CALL_KEEP_LAST");
		return text == nullptr || text[0] != '0';
	}();
	return value;
}

// Round 94: the SPIR-V words from which a compute pipeline compiles in the background
// (PipelineCache::TryGetComputePipeline; KYTY_COMPUTE_BACKGROUND_WORDS, 0: none).
static uint32_t ComputeBackgroundWords() {
	static const uint32_t value = [] {
		const char* text = std::getenv("KYTY_COMPUTE_BACKGROUND_WORDS");
		return text != nullptr && *text != '\0'
		           ? static_cast<uint32_t>(std::strtoul(text, nullptr, 10))
		           : 1000000u;
	}();
	return value;
}

// Round 98: how many compute pipelines compile in the background at once
// (KYTY_COMPUTE_BACKGROUND_THREADS, 1 to 16), one per shader. Without a limit, a run in the ray
// tracing mode started 71 compiles of material passes of up to 14 million SPIR-V words, each new
// expansion of a shader while the older ones still compiled: up to 27 compiled at once, ordinary
// pipelines took up to 36 s meanwhile, memory ran out and the driver ended the emulator.
// Round 117: the default was 3 (six partitions of 0.4 to 0.6 million words, 20 to 35 s each on a
// Ryzen 9 7950X3D); the ray tracing mode's garage needed about 90 of them after each change of
// the translator, minutes of blotched cars. Now a quarter of the logical processors, from 3 to 8,
// and at most 5 with less than 48 GB of memory, 3 with less than 24.
static uint32_t ComputeBackgroundThreads() {
	static const uint32_t value = [] {
		const char* text = std::getenv("KYTY_COMPUTE_BACKGROUND_THREADS");
		if (text != nullptr && *text != '\0') {
			const auto slots =
			    static_cast<uint32_t>(std::clamp(std::strtoul(text, nullptr, 10), 1ul, 16ul));
			LOGF("Compute pipelines: up to %u compile in the background at once (a partition takes "
			     "half of one; KYTY_COMPUTE_BACKGROUND_THREADS)\n",
			     slots);
			return slots;
		}
		const uint32_t processors = std::max(std::thread::hardware_concurrency(), 1u);
		uint32_t       slots      = std::clamp(processors / 4u, 3u, 8u);
		uint64_t       total      = 0;
		uint64_t       available  = 0;
		const bool     memory     = Common::VirtualMemory::QueryPhysical(total, available);
		const uint64_t gb         = total >> 30u;
		if (memory && gb < 47u) {
			slots = std::min(slots, gb < 23u ? 3u : 5u);
		}
		LOGF("Compute pipelines: up to %u compile in the background at once (a partition takes "
		     "half of one), from %u logical processors and %s of memory "
		     "(KYTY_COMPUTE_BACKGROUND_THREADS sets it)\n",
		     slots, processors,
		     memory ? fmt::format("{} GB", (total + (uint64_t {1} << 29u)) >> 30u).c_str()
		            : "an unknown amount");
		return slots;
	}();
	return value;
}

// Round 93: a compute program that gives up leaves what an offline replay needs (the shader
// batch tool's --replay mode) in shader-dumps/: the code it translated (expanded when it calls
// functions), its user data and its stage's registers, as CS_<hash>_<words>.gaveup.{bin,txt}.
// The first 64 of a run, once for each shader and code size (a second expansion of the same
// shader gets files of its own). Round 94: and one whose SPIR-V has ComputeBackgroundWords or
// more, as .big.{bin,txt}, to see offline what makes it so big.
static void DumpComputeForReplay(const char* kind, uint64_t hash, std::span<const uint32_t> code,
                                 std::span<const uint32_t>     user_data,
                                 const ShaderComputeInputInfo& info, uint32_t failure_pc) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> dumped;
	{
		const std::lock_guard lock(mutex);
		const uint64_t        key = hash ^ (uint64_t {code.size()} << 40u) ^
		                     (static_cast<uint64_t>(static_cast<uint8_t>(kind[0])) << 56u);
		if (dumped.size() >= 64 || !dumped.insert(key).second) {
			return;
		}
	}
	std::error_code created;
	std::filesystem::create_directories("shader-dumps", created);
	char path[112];
	std::snprintf(path, sizeof(path), "shader-dumps/CS_%016" PRIx64 "_%zu.%s.bin", hash,
	              code.size(), kind);
	if (FILE* file = std::fopen(path, "wb"); file != nullptr) {
		std::fwrite(code.data(), sizeof(uint32_t), code.size(), file);
		std::fclose(file);
	}
	std::snprintf(path, sizeof(path), "shader-dumps/CS_%016" PRIx64 "_%zu.%s.txt", hash,
	              code.size(), kind);
	FILE* file = std::fopen(path, "w");
	if (file == nullptr) {
		return;
	}
	std::fprintf(file, "hash=%016" PRIx64 " words=%zu failure_pc=0x%x\n", hash, code.size(),
	             failure_pc);
	std::fprintf(
	    file, "threads=%u %u %u lds_dwords=%u scratch_dwords=%u wave=%u host_subgroup=%u\n",
	    info.threads_num[0], info.threads_num[1], info.threads_num[2], info.lds_size_dwords,
	    info.scratch_size_dwords, info.wave_size, info.host_subgroup_size);
	std::fprintf(file,
	             "float_mode=%u group_id=%d %d %d thread_ids=%d tg_size=%d workgroup_register=%d "
	             "thread_dimensions=%d indirect=%d counts=%u %u %u\n",
	             info.float_mode, info.group_id[0] ? 1 : 0, info.group_id[1] ? 1 : 0,
	             info.group_id[2] ? 1 : 0, info.thread_ids_num, info.tg_size_en ? 1 : 0,
	             info.workgroup_register, info.dispatch_thread_dimensions ? 1 : 0,
	             info.dispatch_dimensions_indirect ? 1 : 0, info.workgroup_counts[0],
	             info.workgroup_counts[1], info.workgroup_counts[2]);
	std::fprintf(file, "lds_storage=%d dispatch_threads=%u %u %u\n", info.lds_storage ? 1 : 0,
	             info.dispatch_threads_num[0], info.dispatch_threads_num[1],
	             info.dispatch_threads_num[2]);
	std::fprintf(file, "user_data=");
	for (const auto word: user_data) {
		std::fprintf(file, "%08x ", word);
	}
	std::fprintf(file, "\n");
	std::fclose(file);
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

// The SPIR-V validator's verdict, whatever --shader-validation says (failures are logged).
bool ValidateShaderSpirvNow(const char* label, uint64_t shader_hash,
                            const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
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

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	return !Config::ShaderValidationEnabled() || ValidateShaderSpirvNow(label, shader_hash, spirv);
}

// Round 148: KYTY_MESH_TEST's copy of a mesh shader (spirvMeshTest.h), checked by the SPIR-V
// validator; empty when the shader runs as it is. Round 151: also the host's (MeshHostFlags: no
// Layer output on an AMD GPU).
static std::vector<uint32_t> MeshTestVariant(const ShaderRecompiler::CompileOptions& options,
                                             const std::vector<uint32_t>& spirv, uint32_t flags) {
	auto                         variant = SpirvMeshTest(spirv, flags);
	static std::atomic<uint32_t> logged {0};
	const bool                   log = logged.fetch_add(1, std::memory_order_relaxed) < 64;
	if (variant.empty()) {
		if (log) {
			LOGF("Mesh shader 0x%016" PRIx64 " has nothing to take out (%s)\n", options.shader_hash,
			     MeshTestNames(flags).c_str());
		}
		return {};
	}
	if (!ValidateShaderSpirvNow(options.dump_label, options.shader_hash, variant)) {
		LOGF("Mesh shader 0x%016" PRIx64
		     ": its copy (%s) is not valid SPIR-V; the shader runs as it is\n",
		     options.shader_hash, MeshTestNames(flags).c_str());
		return {};
	}
	if (log) {
		LOGF("Mesh shader 0x%016" PRIx64 " runs as a copy (%s): %zu words, was %zu\n",
		     options.shader_hash, MeshTestNames(flags).c_str(), variant.size(), spirv.size());
	}
	DumpShaderSpirv("ms_test", options.shader_hash, variant);
	return variant;
}

// What SpirvWithoutSampleId learns from parsing a module (SPIRV-Tools tells literal operands from
// <id> operands, which a word-by-word scan cannot: the variable's id is small, and so are storage
// classes, extended instruction numbers and memory operands).
struct SampleIdScan {
	uint32_t sample_id   = 0;
	bool     unsupported = false; // another per-sample input, or SampleId where it is not rewritten
	bool     other_use   = false; // the variable named other than by a load, decoration, name or
	                              // the entry point's interface
};

bool IsIdOperand(spv_operand_type_t type) {
	switch (type) {
		case SPV_OPERAND_TYPE_ID:
		case SPV_OPERAND_TYPE_TYPE_ID:
		case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
		case SPV_OPERAND_TYPE_SCOPE_ID:
		case SPV_OPERAND_TYPE_OPTIONAL_ID:
		case SPV_OPERAND_TYPE_VARIABLE_ID: return true;
		default: return false;
	}
}

// First pass: the variable decorated BuiltIn SampleId, and anything else that shades per sample.
spv_result_t ScanSampleIdDecorations(void* user_data, const spv_parsed_instruction_t* inst) {
	auto&       scan  = *static_cast<SampleIdScan*>(user_data);
	const auto* words = inst->words;
	switch (static_cast<spv::Op>(inst->opcode)) {
		case spv::OpDecorate:
			if (inst->num_words >= 3 && words[2] == spv::DecorationSample) {
				scan.unsupported = true;
			}
			if (inst->num_words >= 4 && words[2] == spv::DecorationBuiltIn) {
				if (words[3] == spv::BuiltInSampleId) {
					if (scan.sample_id != 0 && scan.sample_id != words[1]) {
						scan.unsupported = true;
					}
					scan.sample_id = words[1];
				} else if (words[3] == spv::BuiltInSamplePosition) {
					scan.unsupported = true;
				}
			}
			break;
		case spv::OpMemberDecorate:
			if ((inst->num_words >= 4 && words[3] == spv::DecorationSample) ||
			    (inst->num_words >= 5 && words[3] == spv::DecorationBuiltIn &&
			     (words[4] == spv::BuiltInSampleId || words[4] == spv::BuiltInSamplePosition))) {
				scan.unsupported = true;
			}
			break;
		// A decoration group could decorate the variable (or another input) without naming it.
		case spv::OpGroupDecorate:
		case spv::OpGroupMemberDecorate: scan.unsupported = true; break;
		default: break;
	}
	return SPV_SUCCESS;
}

// Second pass: every <id> operand naming the variable, other than the forms that are rewritten.
spv_result_t ScanSampleIdUses(void* user_data, const spv_parsed_instruction_t* inst) {
	auto& scan = *static_cast<SampleIdScan*>(user_data);
	switch (static_cast<spv::Op>(inst->opcode)) {
		case spv::OpDecorate:
		case spv::OpName:
		case spv::OpEntryPoint: return SPV_SUCCESS;
		default: break;
	}
	for (uint16_t i = 0; i < inst->num_operands; i++) {
		const auto& operand = inst->operands[i];
		if (!IsIdOperand(operand.type) || inst->words[operand.offset] != scan.sample_id) {
			continue;
		}
		// A load from the variable (its pointer operand) is what the rewrite keeps.
		if (static_cast<spv::Op>(inst->opcode) == spv::OpLoad && operand.offset == 3) {
			continue;
		}
		scan.other_use = true;
	}
	return SPV_SUCCESS;
}

// One parse of a module by SPIRV-Tools (every SPIR-V version the emitter writes), with operand
// types; false when it does not parse.
bool ParseSpirv(std::span<const uint32_t> spirv, SampleIdScan& scan,
                spv_parsed_instruction_fn_t callback) {
	static const spv_context context    = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
	spv_diagnostic           diagnostic = nullptr;
	const auto               result =
	    spvBinaryParse(context, &scan, spirv.data(), spirv.size(), nullptr, callback, &diagnostic);
	spvDiagnosticDestroy(diagnostic);
	return result == SPV_SUCCESS;
}

} // namespace

bool EqaaPerSampleShading() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_EQAA_PER_SAMPLE");
		return text != nullptr && text[0] == '1';
	}();
	return value;
}

std::vector<uint32_t> SpirvWithoutSampleId(std::span<const uint32_t> spirv) {
	constexpr uint32_t HeaderWords = 5;
	if (spirv.size() < HeaderWords || spirv[0] != spv::MagicNumber) {
		return {};
	}
	SampleIdScan scan;
	if (!ParseSpirv(spirv, scan, ScanSampleIdDecorations) || scan.sample_id == 0 ||
	    scan.unsupported || !ParseSpirv(spirv, scan, ScanSampleIdUses) || scan.other_use) {
		return {};
	}
	const uint32_t sample_id = scan.sample_id;

	struct Instruction {
		size_t   offset;
		uint32_t count;
		spv::Op  opcode;
	};
	std::vector<Instruction> instructions;
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (count == 0 || offset + count > spirv.size()) {
			return {};
		}
		instructions.push_back(
		    {offset, count, static_cast<spv::Op>(spirv[offset] & spv::OpCodeMask)});
		offset += count;
	}
	const auto words = [&](const Instruction& inst) {
		return spirv.subspan(inst.offset, inst.count);
	};
	// Its declaration: an Input variable of a pointer to a 32-bit integer.
	const Instruction* variable = nullptr;
	uint32_t           pointer  = 0;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpVariable && w.size() == 4 && w[2] == sample_id) {
			if (w[3] != spv::StorageClassInput) {
				return {};
			}
			variable = &inst;
			pointer  = w[1];
		}
	}
	if (variable == nullptr) {
		return {};
	}
	uint32_t integer = 0;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypePointer && w.size() == 4 && w[1] == pointer &&
		    w[2] == spv::StorageClassInput) {
			integer = w[3];
		}
	}
	bool integer_ok = false;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypeInt && w.size() == 4 && w[1] == integer && w[2] == 32) {
			integer_ok = true;
		}
	}
	if (!integer_ok) {
		return {};
	}
	// A Private pointer to the integer and a 0 of it: existing ones declared before the variable
	// where the module has them, new ones (declared just before it) otherwise.
	uint32_t bound        = spirv[3];
	uint32_t private_type = 0;
	uint32_t zero         = 0;
	for (const auto& inst: instructions) {
		if (&inst == variable) {
			break;
		}
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypePointer && w.size() == 4 &&
		    w[2] == spv::StorageClassPrivate && w[3] == integer && private_type == 0) {
			private_type = w[1];
		} else if (inst.opcode == spv::OpConstant && w.size() == 4 && w[1] == integer &&
		           w[3] == 0 && zero == 0) {
			zero = w[2];
		}
	}
	const bool new_private = private_type == 0;
	const bool new_zero    = zero == 0;
	if (new_private) {
		private_type = bound++;
	}
	if (new_zero) {
		zero = bound++;
	}
	const bool interface_lists_private = spirv[1] >= 0x00010400u;

	std::vector<uint32_t> out(spirv.begin(), spirv.begin() + HeaderWords);
	out[3] = bound;
	out.reserve(spirv.size() + 16);
	const auto append = [&](std::initializer_list<uint32_t> list) {
		out.insert(out.end(), list.begin(), list.end());
	};
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpDecorate && w[1] == sample_id) {
			continue; // BuiltIn, Flat
		}
		if (inst.opcode == spv::OpCapability && w.size() == 2 &&
		    w[1] == spv::CapabilitySampleRateShading) {
			continue; // nothing else needs it (no other per-sample input, see the scan)
		}
		if (inst.opcode == spv::OpEntryPoint && !interface_lists_private) {
			// Before SPIR-V 1.4 the interface lists Input and Output variables only.
			size_t name_end = 3;
			while (name_end < w.size()) {
				const auto word = w[name_end++];
				if ((word & 0xff000000u) == 0) {
					break; // the word holding the name's terminating zero
				}
			}
			std::vector<uint32_t> rewritten(w.begin(),
			                                w.begin() + static_cast<ptrdiff_t>(name_end));
			for (size_t i = name_end; i < w.size(); i++) {
				if (w[i] != sample_id) {
					rewritten.push_back(w[i]);
				}
			}
			rewritten[0] = (static_cast<uint32_t>(rewritten.size()) << spv::WordCountShift) |
			               spv::OpEntryPoint;
			out.insert(out.end(), rewritten.begin(), rewritten.end());
			continue;
		}
		if (&inst == variable) {
			if (new_private) {
				append({(4u << spv::WordCountShift) | spv::OpTypePointer, private_type,
				        spv::StorageClassPrivate, integer});
			}
			if (new_zero) {
				append({(4u << spv::WordCountShift) | spv::OpConstant, integer, zero, 0u});
			}
			append({(5u << spv::WordCountShift) | spv::OpVariable, private_type, sample_id,
			        spv::StorageClassPrivate, zero});
			continue;
		}
		out.insert(out.end(), w.begin(), w.end());
	}
	return out;
}

namespace {

// KYTY_EXACT_FLOAT: "1" for every vertex-stage shader, or a comma-separated list of shader hashes
// (hex) for those alone (round 102: the depth and colour passes of one mesh, without recompiling
// every pipeline of a race).
struct ExactFloatSetting {
	bool                  all = false;
	std::vector<uint64_t> hashes;
};

const ExactFloatSetting& ExactFloatSettingValue() {
	static const ExactFloatSetting setting = [] {
		ExactFloatSetting result;
		const char*       text = std::getenv("KYTY_EXACT_FLOAT");
		if (text == nullptr || text[0] == '\0') return result;
		if (std::strcmp(text, "1") == 0) {
			result.all = true;
			return result;
		}
		while (text != nullptr && *text != '\0') {
			char*      end  = nullptr;
			const auto hash = std::strtoull(text, &end, 16);
			if (end == text) break;
			result.hashes.push_back(hash);
			text = *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	return setting;
}

} // namespace

bool ExactFloatPositions() {
	const auto& setting = ExactFloatSettingValue();
	return setting.all || !setting.hashes.empty();
}

bool ExactFloatShader(uint64_t shader_hash) {
	const auto& setting = ExactFloatSettingValue();
	return setting.all || std::ranges::find(setting.hashes, shader_hash) != setting.hashes.end();
}

std::vector<uint32_t> SpirvExactFloat(std::span<const uint32_t> spirv) {
	constexpr uint32_t HeaderWords = 5;
	if (spirv.size() < HeaderWords || spirv[0] != spv::MagicNumber) {
		return {};
	}
	// One walk: what is decorated already, the Position built-ins, the Output variables and
	// types, the float arithmetic results and where the annotations end.
	std::unordered_set<uint32_t>               no_contraction;
	std::unordered_set<uint32_t>               invariant;
	std::unordered_set<uint64_t>               invariant_members;
	std::vector<uint32_t>                      position_variables;
	std::vector<std::pair<uint32_t, uint32_t>> position_members;
	std::unordered_set<uint32_t>               output_variables;
	std::vector<uint32_t>                      output_pointees;
	std::unordered_map<uint32_t, uint32_t>     array_elements;
	std::vector<uint32_t>                      arithmetic;
	size_t                                     annotations_end = 0;
	// Without annotations, the new ones go after the debug instructions (the end of the
	// instructions that precede the annotation section).
	size_t     preamble_end = 0;
	bool       in_preamble  = true;
	const auto member_key = [](uint32_t structure, uint32_t member) {
		return (static_cast<uint64_t>(structure) << 32u) | member;
	};
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (count == 0 || offset + count > spirv.size()) {
			return {};
		}
		const auto w      = spirv.subspan(offset, count);
		const auto opcode = static_cast<spv::Op>(spirv[offset] & spv::OpCodeMask);
		switch (opcode) {
			case spv::OpCapability:
			case spv::OpExtension:
			case spv::OpExtInstImport:
			case spv::OpMemoryModel:
			case spv::OpEntryPoint:
			case spv::OpExecutionMode:
			case spv::OpExecutionModeId:
			case spv::OpString:
			case spv::OpSourceExtension:
			case spv::OpSource:
			case spv::OpSourceContinued:
			case spv::OpName:
			case spv::OpMemberName:
			case spv::OpModuleProcessed:
				if (in_preamble) {
					preamble_end = offset + count;
				}
				break;
			default: in_preamble = false; break;
		}
		switch (opcode) {
			case spv::OpDecorate:
				if (count >= 3 && w[2] == spv::DecorationNoContraction) {
					no_contraction.insert(w[1]);
				} else if (count >= 3 && w[2] == spv::DecorationInvariant) {
					invariant.insert(w[1]);
				} else if (count >= 4 && w[2] == spv::DecorationBuiltIn &&
				           w[3] == spv::BuiltInPosition) {
					position_variables.push_back(w[1]);
				}
				annotations_end = offset + count;
				break;
			case spv::OpMemberDecorate:
				if (count >= 4 && w[3] == spv::DecorationInvariant) {
					invariant_members.insert(member_key(w[1], w[2]));
				} else if (count >= 5 && w[3] == spv::DecorationBuiltIn &&
				           w[4] == spv::BuiltInPosition) {
					position_members.emplace_back(w[1], w[2]);
				}
				annotations_end = offset + count;
				break;
			case spv::OpDecorationGroup:
			case spv::OpGroupDecorate:
			case spv::OpGroupMemberDecorate:
			case spv::OpDecorateId:
			case spv::OpDecorateString:
			case spv::OpMemberDecorateString: annotations_end = offset + count; break;
			// Only outputs are made Invariant: a tessellation or geometry stage's Position input
			// is the stage before's output.
			case spv::OpVariable:
				if (count >= 4 && w[3] == spv::StorageClassOutput) {
					output_variables.insert(w[2]);
				}
				break;
			case spv::OpTypePointer:
				if (count >= 4 && w[2] == spv::StorageClassOutput) {
					output_pointees.push_back(w[3]);
				}
				break;
			case spv::OpTypeArray:
			case spv::OpTypeRuntimeArray:
				if (count >= 3) {
					array_elements[w[1]] = w[2];
				}
				break;
			// The arithmetic a driver may fuse into one operation (a multiply and an add into an
			// FMA) where the guest GPU ran them one after the other, rounding in between.
			case spv::OpFAdd:
			case spv::OpFSub:
			case spv::OpFMul:
			case spv::OpFDiv:
			case spv::OpVectorTimesScalar:
			case spv::OpMatrixTimesScalar:
			case spv::OpVectorTimesMatrix:
			case spv::OpMatrixTimesVector:
			case spv::OpMatrixTimesMatrix:
			case spv::OpDot:
				if (count >= 3) {
					arithmetic.push_back(w[2]);
				}
				break;
			default: break;
		}
		offset += count;
	}
	if (annotations_end == 0) {
		annotations_end = preamble_end;
	}
	if (annotations_end == 0) {
		return {};
	}
	// The types an Output variable holds: a pointee, and the elements of a pointee that is an
	// array (a mesh or tessellation control stage's per-vertex outputs).
	std::unordered_set<uint32_t> output_types;
	for (auto type: output_pointees) {
		for (size_t depth = 0; depth < 4 && output_types.insert(type).second; depth++) {
			const auto element = array_elements.find(type);
			if (element == array_elements.end()) {
				break;
			}
			type = element->second;
		}
	}

	std::vector<uint32_t> added;
	for (const auto variable: position_variables) {
		if (output_variables.contains(variable) && invariant.insert(variable).second) {
			added.insert(added.end(), {(3u << spv::WordCountShift) | spv::OpDecorate, variable,
			                           spv::DecorationInvariant});
		}
	}
	for (const auto& [structure, member]: position_members) {
		if (output_types.contains(structure) &&
		    invariant_members.insert(member_key(structure, member)).second) {
			added.insert(added.end(), {(4u << spv::WordCountShift) | spv::OpMemberDecorate,
			                           structure, member, spv::DecorationInvariant});
		}
	}
	for (const auto result: arithmetic) {
		if (no_contraction.insert(result).second) {
			added.insert(added.end(), {(3u << spv::WordCountShift) | spv::OpDecorate, result,
			                           spv::DecorationNoContraction});
		}
	}
	if (added.empty()) {
		return {};
	}
	std::vector<uint32_t> out;
	out.reserve(spirv.size() + added.size());
	out.insert(out.end(), spirv.begin(), spirv.begin() + static_cast<ptrdiff_t>(annotations_end));
	out.insert(out.end(), added.begin(), added.end());
	out.insert(out.end(), spirv.begin() + static_cast<ptrdiff_t>(annotations_end), spirv.end());
	return out;
}

// KYTY_UNNORMALIZED_FIX=0: hull shaders get no SpirvNormalizedExplicitLod copy.
static bool UnnormalizedFixBuilt() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_UNNORMALIZED_FIX");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool StageSamplersAllUnnormalized(const ShaderStageRuntime& stage) {
	if (!stage || stage.program->info.sampled_pairs.empty()) {
		return false;
	}
	const auto& info = stage.program->info;
	for (const auto& pair: info.sampled_pairs) {
		if (pair.sampler >= info.samplers.size()) {
			return false;
		}
		const auto index = info.samplers[pair.sampler].snapshot_index;
		if (index >= stage.resources->samplers.size()) {
			return false;
		}
		const auto& value = stage.resources->samplers[index];
		// FORCE_UNNORMALIZED, S# word 0 bit 15 (ShaderSamplerResource::ForceUnormCoords).
		if (value.dword_count == 0 || ((value.dwords[0] >> 15u) & 1u) == 0) {
			return false;
		}
	}
	return true;
}

bool UnnormalizedSamplingFix() {
	return UnnormalizedFixBuilt() && !Bisect::UnnormalizedOld();
}

namespace {

// Round 79: a hash of a stage's specialization, field by field (the structs hold padding), for
// ProgramCache::Get's permutation lookup. A field added to them must be hashed too: the sizes
// below change with one.
uint64_t SpecializationHash(const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	using Specialization = ShaderRecompiler::IR::ResourceSpecialization;
	static_assert(sizeof(Specialization::Buffer) == 28 && sizeof(Specialization::Image) == 40 &&
	                  sizeof(Specialization::Sampler) == 8 &&
	                  sizeof(Specialization) == 3 * sizeof(std::vector<uint32_t>),
	              "ResourceSpecialization changed: hash its new fields in SpecializationHash");
	thread_local std::vector<uint32_t> words;
	words.clear();
	words.push_back(static_cast<uint32_t>(specialization.buffers.size()));
	words.push_back(static_cast<uint32_t>(specialization.images.size()));
	words.push_back(static_cast<uint32_t>(specialization.samplers.size()));
	for (const auto& buffer: specialization.buffers) {
		words.insert(words.end(),
		             {buffer.packed_stride, static_cast<uint32_t>(buffer.descriptor_format),
		              buffer.descriptor_swizzle,
		              static_cast<uint32_t>(buffer.zero_stride_oob) |
		                  (static_cast<uint32_t>(buffer.write_tracked) << 1u),
		              buffer.indirect_root, buffer.indirect_mapping_offset,
		              buffer.indirect_search_iterations});
	}
	for (const auto& image: specialization.images) {
		words.insert(
		    words.end(),
		    {static_cast<uint32_t>(image.numeric_class), static_cast<uint32_t>(image.dimension),
		     image.mip_count, static_cast<uint32_t>(image.conversion_format), image.shader_swizzle,
		     image.indirect_root, image.indirect_mapping_offset, image.indirect_search_iterations,
		     static_cast<uint32_t>(image.cube) | (static_cast<uint32_t>(image.fmask) << 1u) |
		         (static_cast<uint32_t>(image.bindless) << 2u) |
		         (static_cast<uint32_t>(image.runtime) << 3u) |
		         (static_cast<uint32_t>(image.clamp_half_float) << 4u),
		     static_cast<uint32_t>(image.fmask_fragments)});
	}
	for (const auto& sampler: specialization.samplers) {
		words.insert(words.end(),
		             {static_cast<uint32_t>(sampler.bindless), sampler.bindless_mapping_offset});
	}
	const auto hash = XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
	return hash != 0 ? hash : 1u; // 0 means none (DrawPrep::StageResult)
}

} // namespace

namespace {

// The result type of every result id of a module, from one SPIRV-Tools parse.
spv_result_t ScanResultTypes(void* user_data, const spv_parsed_instruction_t* inst) {
	auto& types = *static_cast<std::unordered_map<uint32_t, uint32_t>*>(user_data);
	if (inst->result_id != 0 && inst->type_id != 0) {
		types[inst->result_id] = inst->type_id;
	}
	return SPV_SUCCESS;
}

} // namespace

std::vector<uint32_t> SpirvNormalizedExplicitLod(std::span<const uint32_t> spirv) {
	constexpr uint32_t HeaderWords = 5;
	if (spirv.size() < HeaderWords || spirv[0] != spv::MagicNumber) {
		return {};
	}
	std::unordered_map<uint32_t, uint32_t> result_types;
	{
		static const spv_context context    = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
		spv_diagnostic           diagnostic = nullptr;
		const auto result = spvBinaryParse(context, &result_types, spirv.data(), spirv.size(),
		                                   nullptr, ScanResultTypes, &diagnostic);
		spvDiagnosticDestroy(diagnostic);
		if (result != SPV_SUCCESS) {
			return {};
		}
	}
	struct Image {
		uint32_t dim     = 0;
		bool     arrayed = false;
		bool     ms      = false;
	};
	std::unordered_map<uint32_t, Image>    images;         // OpTypeImage
	std::unordered_map<uint32_t, uint32_t> sampled_images; // OpSampledImage result -> image
	std::unordered_map<uint64_t, uint32_t> vectors;        // component type << 32 | count
	uint32_t                               glsl = 0, int_type = 0, float_type = 0;
	bool                                   image_query      = false;
	size_t                                 capabilities_end = 0, globals_end = 0;
	std::vector<size_t> samples; // offsets of OpImageSampleExplicitLod to rewrite
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (count == 0 || offset + count > spirv.size()) {
			return {};
		}
		const auto w = spirv.subspan(offset, count);
		switch (static_cast<spv::Op>(spirv[offset] & spv::OpCodeMask)) {
			case spv::OpCapability:
				image_query      = image_query || (count >= 2 && w[1] == spv::CapabilityImageQuery);
				capabilities_end = offset + count;
				break;
			case spv::OpExtInstImport:
				if (count >= 3 && std::string_view(reinterpret_cast<const char*>(&w[2])) ==
				                      std::string_view("GLSL.std.450")) {
					glsl = w[1];
				}
				break;
			case spv::OpTypeInt:
				if (count == 4 && w[2] == 32 && w[3] == 1) int_type = w[1];
				break;
			case spv::OpTypeFloat:
				if (count >= 3 && w[2] == 32) float_type = w[1];
				break;
			case spv::OpTypeVector:
				if (count == 4) vectors[(uint64_t {w[2]} << 32u) | w[3]] = w[1];
				break;
			case spv::OpTypeImage:
				if (count >= 9) images[w[1]] = {w[3], w[5] != 0, w[6] != 0};
				break;
			case spv::OpSampledImage:
				if (count == 5) sampled_images[w[2]] = w[3];
				break;
			case spv::OpFunction:
				if (globals_end == 0) globals_end = offset;
				break;
			case spv::OpImageSampleExplicitLod:
				// Lod alone or with offsets (no bias, gradients or other operands).
				if (count >= 7 && (w[5] & spv::ImageOperandsLodMask) != 0 &&
				    (w[5] & ~(spv::ImageOperandsLodMask | spv::ImageOperandsConstOffsetMask |
				              spv::ImageOperandsOffsetMask)) == 0) {
					samples.push_back(offset);
				} else {
					return {};
				}
				break;
			// Any other use of a sampler: the copy's samplers are made normalized, which only
			// the rewritten samples are ready for.
			case spv::OpImageSampleImplicitLod:
			case spv::OpImageSampleDrefImplicitLod:
			case spv::OpImageSampleDrefExplicitLod:
			case spv::OpImageSampleProjImplicitLod:
			case spv::OpImageSampleProjExplicitLod:
			case spv::OpImageSampleProjDrefImplicitLod:
			case spv::OpImageSampleProjDrefExplicitLod:
			case spv::OpImageGather:
			case spv::OpImageDrefGather:
			case spv::OpImageSparseSampleImplicitLod:
			case spv::OpImageSparseSampleExplicitLod:
			case spv::OpImageSparseSampleDrefImplicitLod:
			case spv::OpImageSparseSampleDrefExplicitLod:
			case spv::OpImageSparseGather:
			case spv::OpImageSparseDrefGather:
			case spv::OpImageQueryLod: return {};
			default: break;
		}
		offset += count;
	}
	if (samples.empty() || glsl == 0 || float_type == 0 || globals_end == 0 ||
	    capabilities_end == 0) {
		return {};
	}
	uint32_t              bound = spirv[3];
	std::vector<uint32_t> globals;
	const auto            op = [](spv::Op opcode, uint32_t words) {
        return (words << spv::WordCountShift) | static_cast<uint32_t>(opcode);
	};
	if (int_type == 0) {
		int_type = bound++;
		globals.insert(globals.end(), {op(spv::OpTypeInt, 4), int_type, 32u, 1u});
	}
	const auto vector = [&](uint32_t component, uint32_t components) {
		if (components == 1) return component;
		auto& id = vectors[(uint64_t {component} << 32u) | components];
		if (id == 0) {
			id = bound++;
			globals.insert(globals.end(), {op(spv::OpTypeVector, 4), id, component, components});
		}
		return id;
	};
	const uint32_t half = bound++, zero = bound++, one = bound++;
	globals.insert(globals.end(),
	               {op(spv::OpConstant, 4), float_type, half, 0x3f000000u, op(spv::OpConstant, 4),
	                int_type, zero, 0u, op(spv::OpConstant, 4), int_type, one, 1u});

	// Before each sample: the coordinates divided by the size of the mip the LOD names (the
	// nearest one, within the image's), the array layer as it is.
	std::unordered_map<size_t, std::vector<uint32_t>> inserted;
	std::unordered_map<size_t, uint32_t>              coordinates;
	for (const auto offset: samples) {
		const auto w = spirv.subspan(offset, spirv[offset] >> spv::WordCountShift);
		// Every sample is rewritten, or none (the copy's samplers are all normalized).
		const auto found = sampled_images.find(w[3]);
		if (found == sampled_images.end()) return {};
		const uint32_t image      = found->second;
		const auto     image_type = result_types.find(image);
		if (image_type == result_types.end()) return {};
		const auto info = images.find(image_type->second);
		if (info == images.end() || info->second.ms || info->second.dim > spv::Dim3D) return {};
		const uint32_t spatial         = info->second.dim == spv::Dim1D   ? 1u
		                                 : info->second.dim == spv::Dim2D ? 2u
		                                                                  : 3u;
		const uint32_t components      = spatial + (info->second.arrayed ? 1u : 0u);
		const auto     coordinate_type = result_types.find(w[4]);
		if (coordinate_type == result_types.end() ||
		    coordinate_type->second != vector(float_type, components)) {
			return {};
		}
		const uint32_t lod      = w[6];
		auto&          code     = inserted[offset];
		const uint32_t lod_half = bound++, lod_floor = bound++, lod_int = bound++, levels = bound++,
		               last = bound++, level = bound++, size = bound++, size_float = bound++;
		const auto size_type = vector(int_type, components);
		const auto float_vec = vector(float_type, components);
		code.insert(code.end(), {op(spv::OpFAdd, 5),
		                         float_type,
		                         lod_half,
		                         lod,
		                         half,
		                         op(spv::OpExtInst, 6),
		                         float_type,
		                         lod_floor,
		                         glsl,
		                         8u /* Floor */,
		                         lod_half,
		                         op(spv::OpConvertFToS, 4),
		                         int_type,
		                         lod_int,
		                         lod_floor,
		                         op(spv::OpImageQueryLevels, 4),
		                         int_type,
		                         levels,
		                         image,
		                         op(spv::OpISub, 5),
		                         int_type,
		                         last,
		                         levels,
		                         one,
		                         op(spv::OpExtInst, 8),
		                         int_type,
		                         level,
		                         glsl,
		                         45u /* SClamp */,
		                         lod_int,
		                         zero,
		                         last,
		                         op(spv::OpImageQuerySizeLod, 5),
		                         size_type,
		                         size,
		                         image,
		                         level,
		                         op(spv::OpConvertSToF, 4),
		                         float_vec,
		                         size_float,
		                         size});
		uint32_t normalized = 0;
		if (components == 1) {
			normalized = bound++;
			code.insert(code.end(), {op(spv::OpFDiv, 5), float_type, normalized, w[4], size_float});
		} else {
			std::vector<uint32_t> parts;
			for (uint32_t k = 0; k < components; k++) {
				const uint32_t part = bound++;
				code.insert(code.end(),
				            {op(spv::OpCompositeExtract, 5), float_type, part, w[4], k});
				if (k < spatial) {
					const uint32_t extent = bound++, divided = bound++;
					code.insert(code.end(),
					            {op(spv::OpCompositeExtract, 5), float_type, extent, size_float, k,
					             op(spv::OpFDiv, 5), float_type, divided, part, extent});
					parts.push_back(divided);
				} else {
					parts.push_back(part);
				}
			}
			normalized = bound++;
			code.push_back(op(spv::OpCompositeConstruct, 3 + components));
			code.push_back(float_vec);
			code.push_back(normalized);
			code.insert(code.end(), parts.begin(), parts.end());
		}
		coordinates[offset] = normalized;
	}
	if (coordinates.empty()) {
		return {};
	}
	std::vector<uint32_t> out(spirv.begin(), spirv.begin() + HeaderWords);
	out[3] = bound;
	out.reserve(spirv.size() + globals.size() + 64u * coordinates.size());
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (offset == globals_end) {
			out.insert(out.end(), globals.begin(), globals.end());
		}
		if (const auto code = inserted.find(offset);
		    code != inserted.end() && coordinates.contains(offset)) {
			out.insert(out.end(), code->second.begin(), code->second.end());
		}
		const size_t first = out.size();
		out.insert(out.end(), spirv.begin() + static_cast<ptrdiff_t>(offset),
		           spirv.begin() + static_cast<ptrdiff_t>(offset + count));
		if (const auto coordinate = coordinates.find(offset); coordinate != coordinates.end()) {
			out[first + 4] = coordinate->second;
		}
		if (!image_query && offset + count == capabilities_end) {
			out.insert(out.end(), {op(spv::OpCapability, 2), spv::CapabilityImageQuery});
		}
		offset += count;
	}
	return out;
}

// A/B switch for background pipeline compiles (KYTY_ASYNC_PIPELINES=0 compiles every graphics
// pipeline on the GPU thread, as before) and how long a draw waits for a new one
// (KYTY_PIPELINE_WAIT_MS, default 20) before it is skipped.
static const bool g_async_pipelines = [] {
	const char* value = std::getenv("KYTY_ASYNC_PIPELINES");
	return value == nullptr || value[0] != '0';
}();
// Pipelines whose stages are this small (SPIR-V words, all stages; KYTY_PIPELINE_SYNC_WORDS)
// compile on the GPU thread: a few milliseconds, and their draws are never skipped. A skipped
// draw is lost for good when the game draws it once into a cached layer: the title's UI drew its
// WOLVERINE logo once, during the cold-start compile backlog, and the logo stayed missing.
static const uint64_t g_pipeline_sync_words = [] {
	const char* value = std::getenv("KYTY_PIPELINE_SYNC_WORDS");
	return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {12000};
}();
static const bool g_dynamic_raster_state = [] {
	const char* value = std::getenv("KYTY_DYNAMIC_RASTER_STATE");
	return value == nullptr || value[0] != '0';
}();

bool PipelineDynamicRasterStateEnabled() {
	return g_dynamic_raster_state;
}

namespace {

// Zeroes key fields that cannot change the pipeline PrepareGraphicsPipeline creates, so draws
// that differ only in them share one pipeline instead of compiling identical copies (idea from
// Jetsku/KytyPS5 ef5e8fd9).
// - Cull mode and front face are dynamic state, recorded per draw from the same registers with
//   the same rect-list rule (SetGraphicsDynamicParams). The polygon mode stays static; it is
//   resolved from the real cull bits before they leave the key.
// - The depth-bounds test enable and bounds are dynamic state as well, so each DB_DEPTH_BOUNDS
//   value no longer creates its own pipeline (before, even with the test off). On macOS the
//   pipeline always disables the test (MoltenVK has no depthBounds), so they never mattered.
// Blend factors and ops of an attachment with blending off, and the alpha ones without separate
// alpha blending, are never written into the key (TryGetGraphicsPipeline), so they need no rule.
// KYTY_DYNAMIC_RASTER_STATE=0 keeps all of these in the key and bakes them into the pipeline.
void NormalizeGraphicsPipelineKey(PipelineStaticParameters& params) {
	if (!g_dynamic_raster_state) {
		return;
	}
	params.cull_front               = false;
	params.cull_back                = false;
	params.face                     = false;
	params.depth_bounds_test_enable = false;
	params.depth_min_bounds         = 0.0f;
	params.depth_max_bounds         = 0.0f;
}

} // namespace

// Draws without depth or stencil whose vertex shaders total at most KYTY_PIPELINE_SYNC_UI_VS_WORDS
// SPIR-V words (default 8000; 0 = off) also compile on the GPU thread, whatever the pixel
// shader's size: those are UI and 2D layer draws, which a game may also draw only once. On cold
// starts the title's frosted panels lost their blur (PS 0x09161b28ae4039bb, 80k words) and
// showed as dark boxes.
static const uint64_t g_pipeline_sync_ui_vs_words = [] {
	const char* value = std::getenv("KYTY_PIPELINE_SYNC_UI_VS_WORDS");
	return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {8000};
}();
static const std::chrono::milliseconds g_pipeline_wait = [] {
	const char* value = std::getenv("KYTY_PIPELINE_WAIT_MS");
	return std::chrono::milliseconds(value != nullptr ? std::strtoul(value, nullptr, 10) : 20u);
}();
// Round 87: the time a frame spends in all waiting for background compiles
// (KYTY_PIPELINE_FRAME_WAIT_MS; by default 6 ms, 2 ms when the driver's caches are asked first, as
// a pipeline they hold is then created at once). Before, each new pipeline had its 20 ms.
static const std::optional<uint32_t> g_pipeline_frame_wait_ms = []() -> std::optional<uint32_t> {
	const char* value = std::getenv("KYTY_PIPELINE_FRAME_WAIT_MS");
	if (value == nullptr || *value == '\0') {
		return std::nullopt;
	}
	return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}();
// KYTY_PIPELINE_PROBE=0: never ask the driver's caches first (round 87).
static const bool g_pipeline_probe = [] {
	const char* value = std::getenv("KYTY_PIPELINE_PROBE");
	return value == nullptr || value[0] != '0';
}();

struct PendingGraphicsPipeline {
	std::unique_ptr<GraphicsPipelineBuild> build;
	// For the profiler: the guest shaders (pixel 0 without one).
	uint64_t                               vs_hash      = 0;
	uint64_t                               ps_hash      = 0;
	vk::PipelineCache                      driver_cache = nullptr;
	// The full cache files to look it up in first (empty: the GPU thread did).
	std::span<const vk::PipelineCache> older;
	vk::Pipeline                       pipeline = nullptr;
	vk::Result                         result   = vk::Result::eSuccess;
	std::atomic<bool>                  done {false};
	// Round 130: true while a worker looks it up in the driver's caches alone (the GPU thread did
	// that before); one they do not hold is then compiled like any other.
	std::atomic<bool> looking_up {false};
	// Round 134: compiled (not found by a lookup alone): new in the driver cache, for a save
	// during play (set before done).
	bool compiled = false;
	// Built first without driver optimization (set before quick_ready): its draws use it while the
	// optimized one compiles, instead of being skipped for seconds.
	vk::Pipeline      quick = nullptr;
	std::atomic<bool> quick_ready {false};
	// The last time a draw asked for it (AskClockMs): the workers compile the most recently
	// asked first. LIFO by submit starved a pipeline drawn every frame since the start (the
	// player's car stayed transparent in a race) behind the new ones each frame brought.
	mutable std::atomic<uint64_t> asked {0};
};

static uint64_t AskClockMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// KYTY_PIPELINE_BACKGROUND_THREADS (2): compile workers that may take jobs no draw asked for in
// the last 2 s, and the optimized compiles of pipelines drawn unoptimized meanwhile.
static const uint32_t g_pipeline_background = [] {
	const char* value = std::getenv("KYTY_PIPELINE_BACKGROUND_THREADS");
	return value != nullptr && *value != '\0'
	           ? std::max<uint32_t>(1u, static_cast<uint32_t>(std::strtoul(value, nullptr, 10)))
	           : 2u;
}();

// KYTY_PIPELINE_QUICK=1 (round 141, from Franseven; off by default): a missing pipeline is first
// built without driver optimization and drawn at once, the optimized one swapped in later. A new
// car otherwise shows transparent until its compile is done (1280 compiles of 1.8 s median, up to
// 10.7 s, on 8 workers), but the unoptimized pipelines can cost frame rate until they are
// replaced, and only the background workers make the optimized ones. Off: draws are skipped
// until the optimized pipeline is compiled, as before.
static bool PipelineQuick() {
	static const bool on = [] {
		const char* value   = std::getenv("KYTY_PIPELINE_QUICK");
		const bool  enabled = value != nullptr && value[0] == '1';
		LOGF("Pipelines: unoptimized pipelines first %s (KYTY_PIPELINE_QUICK)\n",
		     enabled ? "on" : "off");
		return enabled;
	}();
	return on;
}

// Round 130: a lookup's answer other than found or not found (the driver cannot answer), for the
// GPU thread to stop asking (0: none yet).
static std::atomic<int32_t> g_lookup_unanswered {0};

// Round 92: pipelines the full, read-only cache files gave (lookups alone), for the save's line.
static std::atomic<uint64_t> g_older_file_hits {0};

// Any thread: the pipeline from one of the full cache files that holds it (true), looking it up
// alone (pipelineCreationCacheControl).
static bool CreateFromOlderFiles(const GraphicsPipelineBuild&       build,
                                 std::span<const vk::PipelineCache> older, vk::Pipeline* pipeline) {
	for (const auto cache: older) {
		const auto result = CreateGraphicsPipeline(build, cache, pipeline, true);
		if (result == vk::Result::eSuccess && *pipeline != nullptr) {
			g_older_file_hits.fetch_add(1, std::memory_order_relaxed);
			return true;
		}
		if (*pipeline != nullptr) {
			build.device.destroyPipeline(*pipeline, nullptr);
			*pipeline = nullptr;
		}
		if (result != vk::Result::ePipelineCompileRequired) {
			return false;
		}
	}
	return false;
}

// Any thread: from the full cache files when one holds it, else through the cache that is saved.
static vk::Result CreateThroughCacheFiles(const GraphicsPipelineBuild&       build,
                                          std::span<const vk::PipelineCache> older,
                                          vk::PipelineCache driver_cache, vk::Pipeline* pipeline) {
	if (CreateFromOlderFiles(build, older, pipeline)) {
		return vk::Result::eSuccess;
	}
	return CreateGraphicsPipeline(build, driver_cache, pipeline);
}

// Counts a finished pipeline build and puts it on the timeline (slow ones as a warning).
static void ReportPipelineBuilt(Profiler::Counter counter, const char* kind, uint64_t first_hash,
                                uint64_t second_hash, std::chrono::steady_clock::time_point begin) {
	const auto us = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                         std::chrono::steady_clock::now() - begin)
	                                         .count());
	Profiler::Add(counter);
	Profiler::Add(Profiler::Counter::PipelineBuildTime, us);
	Profiler::Message(us >= 100000 ? Profiler::MessageWarning : Profiler::MessageInfo,
	                  "%s pipeline built in %.1f ms: 0x%016" PRIx64 " 0x%016" PRIx64, kind,
	                  static_cast<double>(us) / 1000.0, first_hash, second_hash);
}

static void ReportGraphicsPipelineBuilt(uint64_t vs_hash, uint64_t ps_hash,
                                        std::chrono::steady_clock::time_point begin) {
	ReportPipelineBuilt(Profiler::Counter::GraphicsPipelinesBuilt, "Graphics", vs_hash, ps_hash,
	                    begin);
}

// Round 130, a worker: the pipeline from the driver's caches, looking it up alone (true), as the
// GPU thread did before; it is compiled next when they do not hold it (job.older is then empty,
// as they were asked).
static bool LookUpPipeline(PendingGraphicsPipeline& job) {
	const auto running =
	    GpuPhases::g_worker_lookups.running.fetch_add(1, std::memory_order_relaxed) + 1u;
	const auto begin  = std::chrono::steady_clock::now();
	auto       result = vk::Result::eSuccess;
	if (!CreateFromOlderFiles(*job.build, job.older, &job.pipeline)) {
		result = CreateGraphicsPipeline(*job.build, job.driver_cache, &job.pipeline, true);
	}
	const bool found = result == vk::Result::eSuccess && job.pipeline != nullptr;
	if (!found && job.pipeline != nullptr) {
		job.build->device.destroyPipeline(job.pipeline, nullptr);
		job.pipeline = nullptr;
	}
	if (!found && result != vk::Result::eSuccess &&
	    result != vk::Result::ePipelineCompileRequired) {
		int32_t none = 0;
		(void)g_lookup_unanswered.compare_exchange_strong(none, static_cast<int32_t>(result),
		                                                  std::memory_order_relaxed);
	}
	GpuPhases::g_worker_lookups.running.fetch_sub(1, std::memory_order_relaxed);
	GpuPhases::AddWorkerLookup(
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                              std::chrono::steady_clock::now() - begin)
	                              .count()),
	    found, running);
	job.older = {};
	if (found) {
		job.result = vk::Result::eSuccess;
		ReportGraphicsPipelineBuilt(job.vs_hash, job.ps_hash, begin);
	}
	job.looking_up.store(false, std::memory_order_release);
	return found;
}

// Compiles graphics pipelines on worker threads. The driver compiled on the GPU thread, which
// stopped the whole emulated GPU for seconds per new pipeline (103 slow pipelines, 170 s of a
// 600 s first pass through the prologue).
class PipelineCompiler {
public:
	// Round 130: `lookup_threads` more threads only look pipelines up in the driver's caches
	// (the compile threads do too, before their compiles): a lookup is not kept waiting behind
	// the compiles of a new track, which take about a second each.
	PipelineCompiler(uint32_t threads, uint32_t lookup_threads) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this, i] { Run(false, i); });
		}
		for (uint32_t i = 0; i < lookup_threads; i++) {
			m_threads.emplace_back([this] { Run(true, 0); });
		}
		m_lookup_threads = lookup_threads;
	}
	~PipelineCompiler() { Stop(); }
	KYTY_CLASS_NO_COPY(PipelineCompiler);

	// Whether jobs may be looked up on the workers (lookup threads were made).
	[[nodiscard]] bool LooksUp() const { return m_lookup_threads != 0; }

	void Submit(std::shared_ptr<PendingGraphicsPipeline> job) {
		const bool lookup = job->looking_up.load(std::memory_order_relaxed);
		job->asked.store(AskClockMs(), std::memory_order_relaxed);
		{
			std::lock_guard lock(m_mutex);
			(lookup ? m_lookups : m_queue).push_back(std::move(job));
		}
		if (lookup) {
			m_lookup_work.notify_one();
		}
		m_work.notify_one();
	}

	// Round 144: a draw that may not be skipped waits for this job, so it counts as asked for now
	// and the next free worker takes it. Nothing else asks while the GPU thread waits: after 2 s
	// every queued job was background work, which only the background workers took, the oldest
	// and the newest first, and a race start on a cold cache waited 190 s for one pipeline.
	// Round 145: `stalled` (the GPU thread waited 100 ms already) lets every worker take the
	// background work for the next 500 ms. The game is stopped meanwhile, so the cores are free,
	// and the next draws that may not be skipped find their pipelines done: one race start waited
	// for 51 in a row, 49 s, with 13 of the 16 workers idle.
	void Ask(const PendingGraphicsPipeline& job, bool stalled) {
		const auto now = AskClockMs();
		job.asked.store(now, std::memory_order_relaxed);
		if (stalled) {
			m_stalled_until.store(now + 500, std::memory_order_relaxed);
		}
		m_work.notify_all();
	}

	// Waits up to the budget; true when the job is done.
	bool Wait(const PendingGraphicsPipeline& job, std::chrono::microseconds budget) {
		KYTY_PROFILER_BLOCK("PipelineCompiler::Wait");
		std::unique_lock lock(m_mutex);
		return m_done.wait_for(lock, budget,
		                       [&job] { return job.done.load(std::memory_order_acquire); });
	}

	// Finishes the compiles in progress; queued ones are left to their owner.
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			if (m_stopped) {
				return;
			}
			m_stopped = true;
			m_queue.clear();
			m_lookups.clear();
			m_optimize.clear();
		}
		m_work.notify_all();
		m_lookup_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
		m_threads.clear();
	}

	[[nodiscard]] bool Stopped() {
		std::lock_guard lock(m_mutex);
		return m_stopped;
	}

private:
	// Under m_mutex: the next job for compile worker `index`, if it may take one now.
	// Lookups first: each takes milliseconds, a compile about a second. Then the compile a draw
	// asked for last: the scene on screen (a scene stayed in showed its objects only after the
	// backlog of the last one). Worker 0 takes the oldest instead: a car seen for a moment left
	// its compiles queued behind each new frame's, and they never ran, not even when it came back.
	// Jobs no draw asked for in the last 2 s and the optimized compiles of pipelines drawn
	// unoptimized run on the first g_pipeline_background workers only: 8 compiles behind the
	// scene took the cores the game shares, and it ran slower until a restart. While the GPU
	// thread is stalled on a compile, every worker takes them (round 145, Ask).
	bool Pick(uint32_t index, std::shared_ptr<PendingGraphicsPipeline>& job, bool& lookup) {
		lookup = !m_lookups.empty();
		if (lookup) {
			job = std::move(m_lookups.front());
			m_lookups.pop_front();
			return true;
		}
		const bool background = index < g_pipeline_background ||
		                        AskClockMs() < m_stalled_until.load(std::memory_order_relaxed);
		if (!m_queue.empty()) {
			// ponytail: a linear scan; a heap if the queue grows to many thousands.
			auto pick = index == 0
			                ? m_queue.begin()
			                : std::max_element(m_queue.begin(), m_queue.end(),
			                                   [](const auto& a, const auto& b) {
				                                   return a->asked.load(std::memory_order_relaxed) <
				                                          b->asked.load(std::memory_order_relaxed);
			                                   });
			if (background ||
			    AskClockMs() - (*pick)->asked.load(std::memory_order_relaxed) < 2000) {
				job = std::move(*pick);
				m_queue.erase(pick);
				return true;
			}
		}
		if (background && !m_optimize.empty()) {
			job = std::move(m_optimize.back());
			m_optimize.pop_back();
			return true;
		}
		return false;
	}

	void Run(bool lookups_only, uint32_t index) {
		KYTY_PROFILER_THREAD(lookups_only ? "PipelineLookup" : "PipelineCompiler");
		// Below the guest's threads: more workers then cost the game nothing it needs.
		Common::LowerThreadPriority();
		for (;;) {
			std::shared_ptr<PendingGraphicsPipeline> job;
			bool                                     lookup = false;
			{
				std::unique_lock lock(m_mutex);
				if (lookups_only) {
					m_lookup_work.wait(lock, [this] { return m_stopped || !m_lookups.empty(); });
					if (m_stopped) {
						return;
					}
					lookup = true;
					job    = std::move(m_lookups.front());
					m_lookups.pop_front();
				} else {
					// A job turns background with time alone: look again every 200 ms.
					while (!m_stopped && !Pick(index, job, lookup)) {
						m_work.wait_for(lock, std::chrono::milliseconds(200));
					}
					if (m_stopped) {
						return;
					}
				}
			}
			if (lookup) {
				bool found = false;
				{
					Profiler::Phases zone;
					zone.SetText("VS 0x%016" PRIx64 " PS 0x%016" PRIx64, job->vs_hash,
					             job->ps_hash);
					KYTY_PROFILER_PHASE(zone, "PipelineCompiler::LookUp",
					                    profiler::colors::DeepOrangeA200);
					found = LookUpPipeline(*job);
				}
				if (found) {
					Profiler::Add(Profiler::Counter::PipelinesPending, -1);
					{
						std::lock_guard lock(m_mutex);
						job->done.store(true, std::memory_order_release);
					}
					m_done.notify_all();
					continue;
				}
				// Not in the caches: compiled like any other. Once the workers stop, the owner
				// compiles it (FinishPending), as one still queued.
				{
					std::lock_guard lock(m_mutex);
					if (m_stopped) {
						continue;
					}
					m_queue.push_back(std::move(job));
				}
				m_work.notify_one();
				continue;
			}
			// A job with full cache files to look in first may find it there in milliseconds.
			if (PipelineQuick() && job->older.empty() &&
			    !job->quick_ready.load(std::memory_order_relaxed)) {
				// Not into the saved cache: only the optimized pipeline is kept.
				if (CreateGraphicsPipeline(*job->build, nullptr, &job->quick, false, true) !=
				        vk::Result::eSuccess &&
				    job->quick != nullptr) {
					job->build->device.destroyPipeline(job->quick, nullptr);
					job->quick = nullptr;
				}
				if (job->quick != nullptr) {
					job->quick_ready.store(true, std::memory_order_release);
					std::lock_guard lock(m_mutex);
					if (!m_stopped) {
						m_optimize.push_back(std::move(job));
						continue;
					}
				}
			}
			{
				Profiler::Phases zone;
				zone.SetText("VS 0x%016" PRIx64 " PS 0x%016" PRIx64, job->vs_hash, job->ps_hash);
				KYTY_PROFILER_PHASE(zone, "PipelineCompiler::Compile", profiler::colors::DeepOrangeA200);
				const auto begin = std::chrono::steady_clock::now();
				job->result = CreateThroughCacheFiles(*job->build, job->older, job->driver_cache,
				                                      &job->pipeline);
				job->compiled    = true;
				ReportGraphicsPipelineBuilt(job->vs_hash, job->ps_hash, begin);
				Profiler::Add(Profiler::Counter::PipelinesPending, -1);
			}
			{
				std::lock_guard lock(m_mutex);
				job->done.store(true, std::memory_order_release);
			}
			m_done.notify_all();
		}
	}

	std::mutex                                           m_mutex;
	std::condition_variable                              m_work;
	std::condition_variable                              m_lookup_work;
	std::condition_variable                              m_done;
	std::deque<std::shared_ptr<PendingGraphicsPipeline>> m_queue;
	std::deque<std::shared_ptr<PendingGraphicsPipeline>> m_lookups;
	std::deque<std::shared_ptr<PendingGraphicsPipeline>> m_optimize;
	std::vector<std::thread>                             m_threads;
	uint32_t                                             m_lookup_threads = 0;
	// Round 145: until this AskClockMs, every worker may take background work (Ask).
	std::atomic<uint64_t>                                m_stalled_until {0};
	bool                                                 m_stopped        = false;
};

// Graphics shader translation on worker threads (plan.md P1; KYTY_ASYNC_TRANSLATION=0 translates
// every shader on the GPU thread, as before). A new shader's translation (decode, CFG, IR,
// resource tracking) took ~85 ms on the GPU thread, and an effect brings dozens: the first
// Square slash in the jungle stopped the emulated GPU for 2-3 s, a new area for minutes. The
// translation depends only on the code and the stage inputs, never on guest memory or the draw's
// user data values (the disk cache's source key holds the same), so a worker can do it while the
// draw is skipped; the materialization and the SPIR-V that need guest memory stay on the GPU
// thread. Only draws with the depth test on are deferred (GetGraphicsPrograms): UI and layer
// draws, which a game may draw once, translate at once, as their pipelines compile at once, but
// for targets the game draws every frame (RedrawnTargetDraw, round 87).
// Compute always translates at once: a skipped dispatch can leave data unwritten.
static const bool g_async_translation = [] {
	const char* value = std::getenv("KYTY_ASYNC_TRANSLATION");
	return value == nullptr || value[0] != '0';
}();

struct TranslationJob {
	std::vector<uint32_t>                code;
	std::vector<uint32_t>                back_code;
	std::vector<uint32_t>                user_data;
	ShaderVertexInputInfo                vertex {};
	ShaderPixelInputInfo                 pixel {};
	// Round 99: a compute shader's partition (ShaderCallPartition), which may be skipped while it
	// translates; and where its resource tracking failed (UINT32_MAX: it did not), which the
	// worker's thread knows.
	ShaderComputeInputInfo                compute {};
	uint32_t                              failure_pc = UINT32_MAX;
	ShaderRecompiler::CompileOptions     options;
	ShaderRecompiler::TranslateResult    result;
	std::chrono::steady_clock::time_point queued;
	std::atomic<bool>                    done {false};
};

class TranslationWorkers {
public:
	explicit TranslationWorkers(uint32_t threads) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~TranslationWorkers() {
		{
			std::lock_guard lock(m_mutex);
			m_stopped = true;
			m_queue.clear();
		}
		m_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(TranslationWorkers);

	void Submit(std::shared_ptr<TranslationJob> job) {
		{
			std::lock_guard lock(m_mutex);
			m_queue.push_back(std::move(job));
		}
		m_work.notify_one();
	}

private:
	void Run() {
		KYTY_PROFILER_THREAD("ShaderTranslator");
		Common::LowerThreadPriority();
		for (;;) {
			std::shared_ptr<TranslationJob> job;
			{
				std::unique_lock lock(m_mutex);
				m_work.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				// Newest first, as the pipeline compiles: the scene on screen before the backlog.
				job = std::move(m_queue.back());
				m_queue.pop_back();
			}
			job->result = ShaderRecompiler::TranslateProgram(job->code, job->options);
			if (job->result.unsupported) {
				job->failure_pc = ShaderRecompiler::IR::LastResourceTrackingFailurePc();
			}
			job->done.store(true, std::memory_order_release);
		}
	}

	std::mutex                                  m_mutex;
	std::condition_variable                     m_work;
	std::deque<std::shared_ptr<TranslationJob>> m_queue;
	std::vector<std::thread>                    m_threads;
	bool                                        m_stopped = false;
};

// Round 134: a shader disk cache file read on a worker thread (DiskReaders): the GPU thread read
// each on its own, two a shader new to the session, while the draw waited. A race start read
// about 3,000 at 7 ms each on a first run after a build (22 s of the GPU thread, the freezes of
// lap 1), at 1 ms on the next run. A deferrable 3D draw is skipped until its read is done, as it
// is while its shader translates.
struct DiskRead {
	std::vector<uint8_t> key;
	std::vector<uint8_t> payload;
	bool                 found = false;
	std::atomic<bool>    done {false};
};

class DiskReaders {
public:
	DiskReaders(ShaderDiskCache::Store& store, uint32_t threads): m_store(store) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~DiskReaders() {
		{
			std::lock_guard lock(m_mutex);
			m_stopped = true;
			m_queue.clear();
		}
		m_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(DiskReaders);

	void Submit(std::shared_ptr<DiskRead> job) {
		{
			std::lock_guard lock(m_mutex);
			m_queue.push_back(std::move(job));
		}
		m_work.notify_one();
	}

private:
	void Run() {
		KYTY_PROFILER_THREAD("ShaderDiskRead");
		for (;;) {
			std::shared_ptr<DiskRead> job;
			{
				std::unique_lock lock(m_mutex);
				m_work.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				job = std::move(m_queue.front());
				m_queue.pop_front();
			}
			const auto begin = std::chrono::steady_clock::now();
			job->found       = m_store.Load(job->key, job->payload);
			GpuPhases::AddWorkerDiskRead(
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                              std::chrono::steady_clock::now() - begin)
			                              .count()),
			    job->found);
			job->done.store(true, std::memory_order_release);
		}
	}

	ShaderDiskCache::Store&               m_store;
	std::mutex                            m_mutex;
	std::condition_variable               m_work;
	std::deque<std::shared_ptr<DiskRead>> m_queue;
	std::vector<std::thread>              m_threads;
	bool                                  m_stopped = false;
};

// Vertex and pixel SRT refresh in parallel (plan.md P10, after Jetsku's draw-prep S3, c7c4da45).
// The SRT refresh is a third of the GPU thread in the jungle, and a draw's two stages walk
// independent plans into independent snapshots. In a short window per draw, this helper walks the
// vertex stage while the GPU thread walks the pixel stage; both only read bytes no GPU work has
// written (ReadShaderGuestMemoryProbe), so neither downloads, faults or waits, and nothing changes
// the GPU-written state the other reads. A walk that fails there (a byte the GPU wrote, memory
// not mapped) is repeated by the GPU thread on the exact path, as before. The helper spins
// KYTY_STAGE_PREP_SPIN_US (default 250) after its last job, then sleeps; a draw never waits for a
// sleeping helper, it wakes it for the next. Off by default: in the jungle a window took ~7 us,
// because the helper's walk runs on cold caches and the GPU thread waits for it, which cost more
// than the walk it saved (Tracy, 2026-10-07). KYTY_STAGE_PREP_PARALLEL=1 turns it on;
// KYTY_STAGE_PREP_VERIFY=1 repeats each parallel walk on the exact path and logs differences.
static const bool g_stage_prep_parallel = [] {
	const char* value = std::getenv("KYTY_STAGE_PREP_PARALLEL");
	return value != nullptr && value[0] == '1';
}();

inline void StagePrepPause() {
#if defined(__x86_64__) || defined(_M_X64)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

class StagePrepWorker {
public:
	using Job = void (*)(void* context);

	StagePrepWorker() {
		const char* spin = std::getenv("KYTY_STAGE_PREP_SPIN_US");
		m_spin           = std::chrono::microseconds(spin != nullptr ? std::atoi(spin) : 250);
		m_thread         = std::thread([this] { Run(); });
	}
	~StagePrepWorker() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_stop_flag.store(true, std::memory_order_release);
		m_wake.notify_one();
		m_thread.join();
	}
	KYTY_CLASS_NO_COPY(StagePrepWorker);

	// Hands the job to the helper; false when it sleeps (it is woken for the next draw) or is
	// still busy, and the caller does the work itself.
	bool Post(Job job, void* context) {
		if (m_sleeping.load(std::memory_order_acquire)) {
			{
				std::lock_guard lock(m_mutex);
				m_wake_requested = true;
			}
			m_wake.notify_one();
			return false;
		}
		if (m_state.load(std::memory_order_acquire) != Idle) {
			return false;
		}
		m_job     = job;
		m_context = context;
		m_state.store(Posted, std::memory_order_release);
		return true;
	}

	// Waits for the posted job; false when the helper never started it (taken back).
	bool Join() {
		uint32_t expected = Posted;
		if (m_state.compare_exchange_strong(expected, Idle, std::memory_order_acq_rel)) {
			return false;
		}
		while (m_state.load(std::memory_order_acquire) != Done) {
			StagePrepPause();
		}
		m_state.store(Idle, std::memory_order_release);
		return true;
	}

private:
	enum : uint32_t { Idle, Posted, Running, Done };

	void Run() {
		KYTY_PROFILER_THREAD("StagePrep");
		Common::PreferPerformanceCores("stage preparation helper");
		Common::RegisterGpuSideThread("StagePrep");
		Libs::LibKernel::Memory::SetGpuReadDelegate(true);
		auto last_job = std::chrono::steady_clock::now();
		while (!m_stop_flag.load(std::memory_order_acquire)) {
			uint32_t expected = Posted;
			if (m_state.load(std::memory_order_acquire) == Posted &&
			    m_state.compare_exchange_strong(expected, Running, std::memory_order_acq_rel)) {
				m_job(m_context);
				m_state.store(Done, std::memory_order_release);
				last_job = std::chrono::steady_clock::now();
				continue;
			}
			if (std::chrono::steady_clock::now() - last_job < m_spin) {
				StagePrepPause();
				continue;
			}
			std::unique_lock lock(m_mutex);
			m_sleeping.store(true, std::memory_order_release);
			m_wake.wait(lock, [this] { return m_wake_requested || m_stop; });
			m_wake_requested = false;
			m_sleeping.store(false, std::memory_order_release);
			last_job = std::chrono::steady_clock::now();
		}
	}

	std::atomic<uint32_t>     m_state {Idle};
	Job                       m_job     = nullptr;
	void*                     m_context = nullptr;
	std::atomic<bool>         m_sleeping {false};
	std::atomic<bool>         m_stop_flag {false};
	std::mutex                m_mutex;
	std::condition_variable   m_wake;
	bool                      m_wake_requested = false;
	bool                      m_stop           = false;
	std::chrono::microseconds m_spin {250};
	std::thread               m_thread;
};

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	PipelineKeyHash::Mix(hash, key.rendering.color_samples);
	PipelineKeyHash::Mix(hash, key.rendering.depth_samples);
	PipelineKeyHash::Mix(hash, key.rendering.layer_limit);
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
		// Exact expanded code includes callees and their original return-PC constants.
		// A different target/body must not reuse a program compiled for an earlier call.
		std::vector<uint32_t> function_code;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint64_t                                     specialization_hash = 0;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		// Held around every walk of resource_plan, whose memo and replay state are mutable: the
		// draw-prep scanner (KYTY_DRAW_PREP) walks programs while the GPU thread does.
		std::mutex walk_mutex;
		// Draw prep: the serial of the prepared result whose refresh `resources` holds now (0 after a
		// walk here), so a later draw of the same packet can use it again.
		uint64_t prepared_serial = 0;
		// The permutation the last lookup found (at most one matches), compared first.
		size_t last_permutation = 0;
		// SpecializationHash(specialization), when the draw-prep scanner made it with the
		// specialization the last take swapped in (round 79); the permutation is then found by it.
		uint64_t specialization_hash       = 0;
		bool     specialization_hash_valid = false;
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
			PipelineKeyHash::Mix(hash, key.function_code.size());
			// The static state and expanded code too (round 68): with only the source's shape,
			// every static variant of a shader (its vertex layouts, render target formats, ...)
			// went into one bucket, and each draw's lookup compared its key with them in turn.
			PipelineKeyHash::Mix(
			    hash, static_cast<std::size_t>(XXH3_64bits(
			              key.static_state.data(), key.static_state.size() * sizeof(uint32_t))));
			if (!key.function_code.empty()) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(XXH3_64bits(
				                               key.function_code.data(),
				                               key.function_code.size() * sizeof(uint32_t))));
			}
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	struct Compiled {
		const char*                              unsupported = nullptr;
		std::vector<uint32_t>                    spirv;
		ShaderRecompiler::IR::CompiledShaderInfo program;
	};

	Compiled CompileTranslated(const ShaderRecompiler::CompileOptions&             options,
	                           ShaderRecompiler::TranslateResult                   translated,
	                           const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                           uint32_t push_data_start_dword) {
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (result.unsupported != nullptr) {
			return {.unsupported = result.unsupported};
		}
		last_spirv_words = result.spirv.size();
		return {.spirv   = std::move(result.spirv),
		        .program = std::move(result.program).TakeCompiledInfo()};
	}

	// Round 95: whether the code being translated is an expansion that inlined called functions
	// (ShaderProgram::inlined_calls): it is longer than the shader by more than the trap stubs of
	// an expansion that inlines none.
	[[nodiscard]] bool InlinedCalls(const ShaderParams& params) const {
		return lookup_key.function_code.size() > params.code.size() + 64u;
	}

	// Round 159: a pixel shader permutation's disk cache entry, by the hash its file is named by,
	// for the copy that applies AA masks (CoverageVariants reads its SPIR-V back from there).
	template <typename InputInfo>
	void NotePixelEntry(const ShaderProgram& handle, std::span<const uint8_t> permutation_key) {
		if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			if (coverage != nullptr && handle && !permutation_key.empty()) {
				const auto hash = XXH3_128bits(permutation_key.data(), permutation_key.size());
				coverage->NotePixelEntry(handle.id, {hash.high64, hash.low64});
			}
		}
	}

	// The modules of a permutation: the stage's, and its per-pixel and normalized-LOD copies when
	// it has them (AddPermutation, and the precompile of guest seeds).
	struct Modules {
		vk::ShaderModule module     = nullptr;
		vk::ShaderModule pixel_rate = nullptr;
		vk::ShaderModule normalized = nullptr;
	};
	Modules MakeModules(const ShaderRecompiler::CompileOptions&         options,
	                    const std::vector<uint32_t>&                    spirv,
	                    const ShaderRecompiler::IR::CompiledShaderInfo& program) {
		std::vector<uint32_t> exact;
		if (options.stage != ShaderType::Pixel && options.stage != ShaderType::Compute &&
		    ExactFloatShader(options.shader_hash)) {
			exact = ExactFloatVariant(options, spirv);
		}
		std::vector<uint32_t> mesh_test;
		if (options.stage == ShaderType::Mesh) {
			if (const auto flags = MeshTestSetting() | MeshHostFlags(); flags != 0) {
				mesh_test = MeshTestVariant(options, exact.empty() ? spirv : exact, flags);
			}
		}
		const auto module = CompileSPV(!mesh_test.empty() ? mesh_test
		                               : exact.empty()    ? spirv
		                                                  : exact,
		                               device);
		EXIT_IF(module == nullptr);
		vk::ShaderModule pixel_rate_module = nullptr;
		if (options.stage == ShaderType::Pixel && pixel_rate_variants &&
		    std::ranges::any_of(program.info.inputs, [](const auto& input) {
			    return input.kind == ShaderRecompiler::IR::StageInputKind::SampleId;
		    })) {
			pixel_rate_module = CompilePixelRateVariant(options, spirv);
		}
		vk::ShaderModule normalized_module = nullptr;
		if (options.stage == ShaderType::TessellationControl && UnnormalizedFixBuilt() &&
		    !program.info.sampled_pairs.empty()) {
			normalized_module = CompileNormalizedVariant(options, exact.empty() ? spirv : exact);
		}
		return {.module = module, .pixel_rate = pixel_rate_module, .normalized = normalized_module};
	}

	// ---- Guest pipeline seeds (pipelineSeeds.h) ----------------------------------------------
	// The code Get translates (set by Get before AddPermutation).
	std::span<const uint32_t> seed_code;

	// The stage inputs as a seed keeps them: no runtime pointers, no host-dependent fields (the
	// precompile sets those for the player's GPU, as GetGraphicsPrograms/GetComputeProgram do).
	template <typename InputInfo>
	static InputInfo SeedInfo(InputInfo info) {
		info.stage = {};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			info.host_subgroup_size = 0;
			info.lds_storage        = false;
		} else if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			info.mesh.host_subgroup_size = 0;
			info.mesh.passes             = 0;
			info.prepare_failed          = false;
		}
		return info;
	}

	template <typename InputInfo>
	static constexpr uint8_t SeedInfoKind() {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			return 0;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			return 1;
		} else {
			return 2;
		}
	}

	// The canonical identity of a program seed: what the disk cache keys (without the session).
	template <typename InputInfo>
	static uint64_t SeedProgramHash(ShaderType stage, uint64_t shader_hash, uint32_t wave_size,
	                                uint32_t user_data_base, std::span<const uint32_t> code,
	                                std::span<const uint32_t> back_code,
	                                std::span<const uint32_t> user_data, const InputInfo& info,
	                                const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                uint32_t push_data_cursor) {
		ShaderDiskCache::ByteWriter key;
		key.bytes.reserve(code.size_bytes() + back_code.size_bytes() + 1024u);
		key.Put(static_cast<uint32_t>(stage));
		key.Put(shader_hash);
		key.Put(wave_size);
		key.Put(user_data_base);
		key.PutWords(code);
		key.PutWords(back_code);
		key.PutWords(user_data);
		ShaderDiskCache::AppendInputKey(key, info);
		ShaderDiskCache::AppendSpecialization(key, specialization);
		key.Put(push_data_cursor);
		return XXH3_64bits(key.bytes.data(), key.bytes.size());
	}

	// A new permutation: its seed, once per run and program (nothing per draw).
	template <typename InputInfo>
	void RecordSeed(const ShaderRecompiler::CompileOptions& options, const InputInfo& input_info,
	                const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                uint32_t push_data_cursor, const ShaderProgram& handle) {
		if (!PipelineSeeds::Recording() || seed_code.empty()) {
			return;
		}
		const auto info = SeedInfo(input_info);
		const auto hash = SeedProgramHash(options.stage, options.shader_hash, options.wave_size,
		                                  options.user_data_base, seed_code, options.back_code,
		                                  options.user_data, info, specialization, push_data_cursor);
		ShaderDiskCache::ByteWriter payload;
		payload.Put(PipelineSeeds::KindProgram);
		payload.Put(static_cast<uint32_t>(options.stage));
		payload.Put(options.shader_hash);
		payload.Put(options.wave_size);
		payload.Put(options.user_data_base);
		payload.Put(push_data_cursor);
		payload.Put(SeedInfoKind<InputInfo>());
		payload.PutWords(seed_code);
		payload.PutWords(options.back_code);
		payload.PutWords(options.user_data);
		payload.Put<uint64_t>(sizeof(InputInfo));
		payload.PutBytes({reinterpret_cast<const uint8_t*>(&info), sizeof(InputInfo)});
		ShaderDiskCache::AppendSpecialization(payload, specialization);
		const std::array<uint64_t, 2> ids {handle.id, handle.normalized_id};
		PipelineSeeds::RecordProgram(hash, std::move(payload.bytes), ids);
		seed_code = {};
	}

	// A program seed, parsed.
	struct SeedProgram {
		uint64_t                                     record = 0;
		ShaderType                                   stage  = ShaderType::Unknown;
		uint64_t                                     shader_hash = 0;
		uint32_t                                     wave_size = 0, user_data_base = 0, push_data_cursor = 0;
		uint8_t                                      kind = 0;
		std::vector<uint32_t>                        code, back_code, user_data;
		ShaderVertexInputInfo                        vertex {};
		ShaderPixelInputInfo                         pixel {};
		ShaderComputeInputInfo                       compute {};
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// Made by the precompile.
		bool                                         ok = false;
		std::vector<uint32_t>                        spirv;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		Modules                                      modules;
	};

	static bool ParseSeedProgram(std::span<const uint8_t> bytes, SeedProgram& seed) {
		ShaderDiskCache::ByteReader in(bytes);
		uint8_t                     tag = 0;
		uint32_t                    stage = 0;
		uint64_t                    size = 0;
		if (!in.Get(tag) || tag != PipelineSeeds::KindProgram || !in.Get(stage) ||
		    !in.Get(seed.shader_hash) || !in.Get(seed.wave_size) || !in.Get(seed.user_data_base) ||
		    !in.Get(seed.push_data_cursor) || !in.Get(seed.kind) || !in.GetWords(seed.code) ||
		    !in.GetWords(seed.back_code) || !in.GetWords(seed.user_data) || !in.Get(size)) {
			return false;
		}
		seed.stage = static_cast<ShaderType>(stage);
		void* target = nullptr;
		switch (seed.kind) {
			case 0: target = &seed.vertex; if (size != sizeof(ShaderVertexInputInfo)) return false; break;
			case 1: target = &seed.pixel; if (size != sizeof(ShaderPixelInputInfo)) return false; break;
			case 2: target = &seed.compute; if (size != sizeof(ShaderComputeInputInfo)) return false; break;
			default: return false;
		}
		const auto offset = bytes.size() - in.Remaining();
		if (in.Remaining() < size) {
			return false;
		}
		std::memcpy(target, bytes.data() + offset, size);
		ShaderDiskCache::ByteReader rest(bytes.subspan(offset + size));
		return ShaderDiskCache::ReadSpecialization(rest, seed.specialization) && rest.AtEnd() &&
		       !seed.code.empty() && seed.user_data.size() <= 64;
	}

	// Called once before ReplaySeed runs on several threads.
	void PrepareReplay() {
		if (disk != nullptr && disk_session_key.empty()) {
			disk_session_key = ShaderDiskCache::SessionKey(disk_session);
		}
	}

	enum class SeedOutcome : uint8_t { Cached, Compiled, Failed };

	// Translates and compiles a program seed into the disk cache (what Get would on its first
	// draw), keeping its SPIR-V and program for the pipelines. Any thread; no modules made.
	template <typename InputInfo>
	SeedOutcome ReplaySeed(SeedProgram& seed, InputInfo& info) {
		ShaderRecompiler::CompileOptions options;
		ShaderStageInputInfo             stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &info;
		} else {
			stage_input.compute = &info;
		}
		options.stage           = seed.stage;
		options.shader_hash     = seed.shader_hash;
		options.user_data       = seed.user_data;
		options.back_code       = seed.back_code;
		options.dump_label      = "ShaderRecompiler (precompile)";
		options.input_info      = stage_input;
		options.wave_size       = seed.wave_size;
		options.user_data_base  = seed.user_data_base;
		options.non_fatal       = ShaderFailureNonFatal();
		options.bindless_images = bindless_images;
		ShaderParams params;
		params.code            = seed.code;
		params.back_code       = seed.back_code;
		params.hash            = seed.shader_hash;
		std::vector<uint8_t> source_key, permutation_key, payload;
		bool                 source_cached = false;
		if (disk != nullptr) {
			BuildSourceKey(source_key, seed.code, params, options, info);
			BuildPermutationKey(permutation_key, source_key, seed.specialization, seed.push_data_cursor);
			ShaderDiskCache::PermutationRecord record;
			if (disk->Load(permutation_key, payload) &&
			    ShaderDiskCache::DecodePermutationRecord(payload, record)) {
				seed.spirv   = std::move(record.spirv);
				seed.program = std::move(record.program);
				return SeedOutcome::Cached;
			}
			ShaderDiskCache::SourceRecord source;
			if (disk->Load(source_key, payload) && ShaderDiskCache::DecodeSourceRecord(payload, source)) {
				if (source.unsupported || (source.uses_clock && !shader_clock)) {
					return SeedOutcome::Failed;
				}
				source_cached = true;
			}
		}
		auto translated = ShaderRecompiler::TranslateProgram(seed.code, options);
		if (translated.unsupported) {
			return SeedOutcome::Failed;
		}
		const bool uses_clock = UsesShaderClock(translated.program);
		if (disk != nullptr && !source_cached) {
			const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			if (ShaderDiskCache::EncodeSourceRecord(false, uses_clock, &plan, payload) &&
			    ShaderDiskCache::SourceRecordRoundTrips(payload)) {
				disk->Save(source_key, payload);
			}
		}
		if (uses_clock && !shader_clock) {
			return SeedOutcome::Failed;
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               seed.specialization, seed.push_data_cursor);
		if (result.unsupported != nullptr || result.spirv.empty()) {
			return SeedOutcome::Failed;
		}
		seed.spirv   = std::move(result.spirv);
		seed.program = std::move(result.program).TakeCompiledInfo();
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, seed.spirv)) {
			return SeedOutcome::Failed;
		}
		if (disk != nullptr) {
			ShaderDiskCache::EncodePermutationRecord(seed.spirv, seed.program, payload);
			if (ShaderDiskCache::PermutationRecordRoundTrips(payload)) {
				disk->Save(permutation_key, payload);
			}
		}
		return SeedOutcome::Compiled;
	}

	// The modules of a replayed seed (serialized: the variant counters are not atomic).
	void MakeSeedModules(SeedProgram& seed) {
		std::lock_guard                  lock(seed_modules_mutex);
		ShaderRecompiler::CompileOptions options;
		options.stage       = seed.stage;
		options.shader_hash = seed.shader_hash;
		options.dump_label  = "ShaderRecompiler (precompile)";
		seed.modules        = MakeModules(options, seed.spirv, seed.program);
	}
	void FlushDiskForReplay() {
		if (disk != nullptr) {
			disk->Flush();
		}
	}
	std::mutex seed_modules_mutex;

	// A compiled (or cached) permutation of the source: its module, bound for this stage.
	template <typename InputInfo>
	ShaderProgram AddPermutation(SourceEntry& source, const char* stage_name,
	                             const ShaderRecompiler::CompileOptions&  options,
	                             const std::vector<uint32_t>&             spirv,
	                             ShaderRecompiler::IR::CompiledShaderInfo program,
	                             InputInfo& input_info, uint32_t& push_data_cursor,
	                             bool inlined_calls) {
		Profiler::Phases phases;
		phases.SetText("%s 0x%016" PRIx64, stage_name, options.shader_hash);
		KYTY_PROFILER_PHASE(phases, "Shader: SPIR-V validate", profiler::colors::Blue300);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, spirv);

		KYTY_PROFILER_PHASE(phases, "Shader: module create", profiler::colors::Blue300);
		const auto modules           = MakeModules(options, spirv, program);
		const auto module            = modules.module;
		const auto pixel_rate_module = modules.pixel_rate;
		const auto normalized_module = modules.normalized;
		phases.End();
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(spirv.size()), options.wave_size);
		}
		const auto id = ++next_shader_id;
		source.permutations.push_back({
		    .specialization      = source.specialization,
		    .specialization_hash = SpecializationHash(source.specialization),
		    .program             = std::move(program),
		    .handle              = {.id                = id,
		                            .module            = module,
		                            .spirv_words       = static_cast<uint32_t>(spirv.size()),
		                            .hash              = options.shader_hash,
		                            .inlined_calls     = inlined_calls,
		                            .pixel_rate_module = pixel_rate_module,
		                            .normalized_module = normalized_module,
		                            .normalized_id     = normalized_module != nullptr ? ++next_shader_id : 0u},
		});
		const auto& permutation = source.permutations.back();
		RecordSeed(options, input_info, source.specialization, push_data_cursor, permutation.handle);
		input_info.stage = {.program = &permutation.program, .resources = &source.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, entry]: programs) {
			counts[static_cast<size_t>(key.stage)] += entry.permutations.size();
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

	// The first materialization of a new source; a failure skips this draw (see Get).
	static bool MaterializeFirstUse(SourceEntry&                           source,
	                                const ShaderRecompiler::IR::SrtRuntime& runtime,
	                                ShaderType stage, const char* stage_name, uint64_t hash) {
		bool ok = false;
		{
			std::lock_guard walk_lock(source.walk_mutex);
			source.prepared_serial = 0;
			ok = ShaderRecompiler::IR::MaterializeResources(source.resource_plan, runtime,
			                                                source.resources, source.specialization);
		}
		if (ok) {
			return true;
		}
		if (!ShaderFailureNonFatal()) {
			EXIT("shader resource materialization failed\n");
		}
		Profiler::Add(Profiler::Counter::SkippedResources);
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
			     ": resource materialization failed on first use (materialization line %d)\n",
			     static_cast<uint32_t>(stage), hash,
			     ShaderRecompiler::IR::LastIndirectImageFailureLine());
			Profiler::Message(Profiler::MessageWarning,
			                  "Resources unreadable on first use, draw skipped: %s 0x%016" PRIx64
			                  " (materialization line %d; first 16 reported)",
			                  stage_name, hash, ShaderRecompiler::IR::LastIndirectImageFailureLine());
		}
		return false;
	}

	static void ReportShaderClock(ShaderType stage, const char* stage_name, uint64_t hash) {
		if (!ShaderFailureNonFatal()) {
			EXIT("S_MEMREALTIME needs shaderDeviceClock\n");
		}
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
			     ": S_MEMREALTIME needs shaderDeviceClock\n",
			     static_cast<uint32_t>(stage), hash);
		}
		Profiler::Add(Profiler::Counter::ShadersGaveUp);
		Profiler::Message(Profiler::MessageFailure,
		                  "Shader gave up: %s 0x%016" PRIx64 ": S_MEMREALTIME needs shaderDeviceClock",
		                  stage_name, hash);
	}

	// The disk cache key of a source: the session key and everything TranslateProgram reads.
	template <typename InputInfo>
	void BuildSourceKey(std::vector<uint8_t>& key, std::span<const uint32_t> code,
	                    const ShaderParams& params, const ShaderRecompiler::CompileOptions& options,
	                    const InputInfo& input_info) {
		if (disk_session_key.empty()) {
			// Built at first use: the device layer has set the emitter's switches by then.
			disk_session_key = ShaderDiskCache::SessionKey(disk_session);
		}
		ShaderDiskCache::ByteWriter writer;
		writer.bytes.reserve(disk_session_key.size() + code.size_bytes() +
		                     params.back_code.size_bytes() + 512u);
		writer.PutBytes(disk_session_key);
		writer.Put<uint8_t>('S');
		writer.Put(static_cast<uint32_t>(options.stage));
		writer.Put(options.shader_hash);
		writer.Put(options.wave_size);
		writer.Put(options.user_data_base);
		writer.Put(static_cast<uint32_t>(options.user_data.size()));
		writer.Put(options.bindless_images);
		writer.Put(options.non_fatal);
		writer.PutWords(code);
		writer.PutWords(params.back_code);
		ShaderDiskCache::AppendInputKey(writer, input_info);
		// A probed shader (KYTY_PROBE) keys its points, and a dumped one (KYTY_DUMP_SHADER_HASHES)
		// its request, so it translates: diagnosing one shader keeps the others.
		if (const auto probe = ShaderRecompiler::ShaderProbeKey(options.shader_hash);
		    !probe.empty()) {
			writer.Put<uint8_t>('D');
			writer.PutBytes({reinterpret_cast<const uint8_t*>(probe.data()), probe.size()});
		}
		if (ShaderRecompiler::ShaderDumpRequested(options.shader_hash)) {
			writer.Put<uint8_t>('X');
		}
		key = std::move(writer.bytes);
	}

	static void BuildPermutationKey(std::vector<uint8_t>& key, std::span<const uint8_t> source_key,
	                                const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                uint32_t push_data_cursor) {
		ShaderDiskCache::ByteWriter writer;
		writer.bytes.reserve(source_key.size() + 256u);
		writer.PutBytes(source_key);
		writer.Put<uint8_t>('P');
		ShaderDiskCache::AppendSpecialization(writer, specialization);
		writer.Put(push_data_cursor);
		key = std::move(writer.bytes);
	}

	// Stores a fresh entry; in verify mode compares it with the cached one first (`cached`, empty
	// when there was none) and stores it only when they differ.
	void StoreEntry(bool source, uint64_t hash, std::span<const uint8_t> key,
	                std::span<const uint8_t> cached, std::span<const uint8_t> payload) {
		const char* kind = source ? "source" : "permutation";
		if (!(source ? ShaderDiskCache::SourceRecordRoundTrips(payload)
		             : ShaderDiskCache::PermutationRecordRoundTrips(payload))) {
			if (disk->counters.round_trip_failures++ < 16) {
				LOGF("Shader disk cache: the %s entry of hash=0x%016" PRIx64
				     " does not decode to itself; not cached\n",
				     kind, hash);
			}
			return;
		}
		if (!cached.empty()) {
			disk->counters.verified++;
			if (std::ranges::equal(cached, payload)) {
				return;
			}
			if (disk->counters.verify_mismatches++ < 32) {
				LOGF("Shader disk cache: verify mismatch: %s entry of hash=0x%016" PRIx64
				     " (%zu bytes cached, %zu fresh)\n",
				     kind, hash, cached.size(), payload.size());
				Profiler::Message(Profiler::MessageFailure,
				                  "Shader disk cache: verify mismatch: %s entry of 0x%016" PRIx64,
				                  kind, hash);
			}
		}
		disk->Save(key, payload);
	}

	void CountDiskLookup() {
		if (++disk_lookups % 4096u == 0) {
			disk->LogTotals("so far");
		}
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		const GpuPhases::Scope get_phase(GpuPhases::Phase::ProgramGet);
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		if (SkipShaderRequested(params.hash)) {
			Profiler::Add(Profiler::Counter::SkippedSkipList);
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": skip list", StageLabel(stage),
			                                params.hash);
			return ShaderProgram {};
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		GpuPhases::Scope key_phase(GpuPhases::Phase::ProgramKey);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		lookup_key.function_code.clear();
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (forced_function_code != nullptr) {
				// Round 97: an earlier expansion of the shader, run while a newer one compiles
				// (PipelineCache::GetComputeProgram).
				lookup_key.function_code = *forced_function_code;
			} else {
				ExpandShaderFunctions(params, user_data, input_info.wave_size,
				                      lookup_key.function_code);
			}
		}
		// gt7-prog-1112, KYTY_PROG_MEMO: the entry this stage's shader found last, when the key is
		// the same (ProgramMemo), without hashing the key for `unsupported` and `programs`.
		ProgramMemo* memo_slot = nullptr;
		bool         memo_hit  = false;
		if (ProgMemoEnabled()) {
			memo_slot = &MemoSlotFor(stage, params.code.data());
			memo_hit  = memo_slot->Matches(lookup_key, params.code.data(), unsupported.size(),
			                               programs.size(), programs.bucket_count());
			GpuPhases::Count(memo_hit ? GpuPhases::Phase::ProgramMemoHit
			                          : GpuPhases::Phase::ProgramMemoMiss,
			                 1);
		}
		if (!memo_hit && unsupported.contains(lookup_key)) {
			Profiler::Add(Profiler::Counter::SkippedGaveUp);
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": shader gave up",
			                                StageLabel(stage), params.hash);
			return ShaderProgram {};
		}
		KYTY_PROFILER_BLOCK("ProgramCache::Get");
		auto entry = memo_hit ? memo_slot->entry : programs.find(lookup_key);
		if (memo_slot != nullptr && !memo_hit && entry != programs.end()) {
			memo_slot->Store(lookup_key, params.code.data(), unsupported.size(), programs.size(),
			                 programs.bucket_count(), entry);
		}
		key_phase.End();
		// Scoped to this call only: see GuestPageReadCache.
		GuestPageReadCache                           clean_read_cache;
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &clean_read_cache,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		static_assert(TRACKER_PAGE_SIZE == 4096u);
		runtime.map_clean_page = +[](void* userdata, uint64_t page) {
			return static_cast<GuestPageReadCache*>(userdata)->Map(page);
		};
		runtime.page_userdata = &clean_read_cache;
		runtime.page_shift    = 12;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			for (uint32_t axis = 0; axis < 3u; axis++) {
				runtime.workgroup_size[axis] = input_info.threads_num[axis];
			}
		}
		if (entry != programs.end()) {
			// gt7-spec-1112, KYTY_SPECULATE (speculation.h): the refresh the draw-prep workers
			// speculated, validated as TakeScanned always does, on a short path without the walk's
			// set-up. Every 64th one goes the classic way, to measure what this saves.
			bool     fast_done     = false;
			bool     scanned_tried = false;
			bool     calibrating   = false;
			uint64_t spec_start    = 0;
			if (Speculation::Enabled() && DrawPrep::Enabled() && !Bisect::DrawPrepOff()) {
				spec_start = GpuPhases::Now();
				if (Speculation::CalibrateNext()) {
					calibrating = true;
					Speculation::Add(Speculation::Counter::Calibration);
				} else {
					entry->second.resources.gpu_fills.clear();
					if (stage == ShaderType::Compute && SrtGpuFillEnabled()) {
						runtime.gpu_written = SrtBytesGpuWritten;
						runtime.gpu_fills   = &entry->second.resources.gpu_fills;
					}
					scanned_tried = true;
					{
						const GpuPhases::Scope phase(GpuPhases::Phase::ProgramScanned);
						fast_done = TakeScanned(entry->second, stage, runtime, params.hash);
					}
					runtime.gpu_written = nullptr;
					runtime.gpu_fills   = nullptr;
					if (fast_done) {
						Speculation::Add(Speculation::Counter::FastHit);
						Speculation::AddFastCycles(GpuPhases::Now() - spec_start);
					} else {
						Speculation::Add(Speculation::Counter::FastMiss);
					}
				}
			}
			if (!fast_done) {
				// One zone per refresh, named by the program (kept with KYTY_PROFILE_ZONES=0).
				Profiler::Phases refresh_zone;
				refresh_zone.SetText("%s %016" PRIx64, StageLabel(stage), params.hash);
				KYTY_PROFILER_PHASE(refresh_zone, "SRT refresh", profiler::colors::Amber300);
				// Compute: flat slots only the shader uses are left to the GPU when it wrote their
				// bytes; the renderer copies them on the GPU before the dispatch (SrtGpuFill).
				entry->second.resources.gpu_fills.clear();
				if (stage == ShaderType::Compute && SrtGpuFillEnabled()) {
					runtime.gpu_written = SrtBytesGpuWritten;
					runtime.gpu_fills   = &entry->second.resources.gpu_fills;
				}
				t_srt_shader_hash = params.hash;
				bool materialized = false;
				bool scanned      = false;
				if (TakePrepared(&entry->second)) {
					entry->second.prepared_serial = 0;
					// Refreshed in this draw's parallel window (PrepareStagesInParallel).
					static const bool verify = std::getenv("KYTY_STAGE_PREP_VERIFY") != nullptr;
					materialized             = true;
					if (verify) {
						VerifyPrepared(entry->second, runtime, stage, params.hash);
					}
				} else if (!scanned_tried && DrawPrep::Enabled() && !Bisect::DrawPrepOff() && [&] {
					           const GpuPhases::Scope phase(GpuPhases::Phase::ProgramScanned);
					           return TakeScanned(entry->second, stage, runtime, params.hash);
				           }()) {
					// Prepared by the draw-prep scanner from the same inputs and bytes.
					materialized = true;
					scanned      = true;
				} else {
					const GpuPhases::Scope phase(GpuPhases::Phase::ProgramWalk);
					std::lock_guard        walk_lock(entry->second.walk_mutex);
					entry->second.prepared_serial = 0;
					materialized = ShaderRecompiler::IR::MaterializeResources(
					    entry->second.resource_plan, runtime, entry->second.resources,
					    entry->second.specialization);
				}
				if (!scanned) {
					// Refreshed here: its specialization has no hash made with it.
					entry->second.specialization_hash_valid = false;
				}
				if (calibrating && scanned) {
					// The classic way of a speculated refresh: what the short path compares with.
					Speculation::AddClassicCycles(GpuPhases::Now() - spec_start);
				}
				t_srt_shader_hash = 0;
				runtime.gpu_written = nullptr;
				runtime.gpu_fills   = nullptr;
				if (ShaderRecompiler::IR::SrtNativeStats stats;
				    ShaderRecompiler::IR::TakeSrtNativeReport(stats)) {
					LOGF("SRT native: %u plans compiled, %u failed, %" PRIu64 " KB of code, %" PRIu64
					     " instructions, %" PRIu64 " through the interpreter\n",
					     stats.plans, stats.failed, stats.bytes / 1024u, stats.instructions,
					     stats.interpreted);
				}
				if (!materialized) {
					// A descriptor source that cannot be read right now (memory the guest has
					// not mapped or filled yet) skips this draw rather than the session; the
					// next one re-evaluates from scratch.
					if (!ShaderFailureNonFatal()) {
						EXIT("shader resource materialization failed\n");
					}
					Profiler::Add(Profiler::Counter::SkippedResources);
					Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": resources unreadable",
					                                StageLabel(stage), params.hash);
					static std::atomic<uint32_t> reported = 0;
					if (reported.fetch_add(1) < 16) {
						LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
						     ": resource materialization failed (materialization line %d)\n",
						     static_cast<uint32_t>(stage), params.hash,
						     ShaderRecompiler::IR::LastIndirectImageFailureLine());
						Profiler::Message(Profiler::MessageWarning,
						                  "Resources unreadable, draw skipped: stage %u 0x%016" PRIx64
						                  " (materialization line %d; first 16 reported)",
						                  static_cast<uint32_t>(stage), params.hash,
						                  ShaderRecompiler::IR::LastIndirectImageFailureLine());
					}
					return ShaderProgram {};
				}
			}
			GpuPhases::Scope match_phase(GpuPhases::Phase::ProgramMatch);
			auto&      source  = entry->second;
			// Round 79: by the hash the scanner made, when there is one (comparing the
			// specializations themselves read the scanner's freshly written vectors: ~2 % of the
			// GPU thread in a race). KYTY_BISECT's spechash:off compares them again.
			const bool by_hash = source.specialization_hash_valid && !Bisect::SpecHashOff();
			const auto matches = [&](const Permutation& candidate) {
				const auto& layout = candidate.program.bindings;
				return layout.push_data_start_dword ==
				           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
				                                                    layout.ShaderDataDwords()) &&
				       (by_hash ? candidate.specialization_hash == source.specialization_hash
				                : candidate.specialization == source.specialization);
			};
			Permutation* permutation = nullptr;
			if (source.last_permutation < source.permutations.size() &&
			    matches(source.permutations[source.last_permutation])) {
				permutation = &source.permutations[source.last_permutation];
			} else if (const auto found = std::ranges::find_if(source.permutations, matches);
			           found != source.permutations.end()) {
				permutation             = &*found;
				source.last_permutation = static_cast<size_t>(found - source.permutations.begin());
			}
			if (permutation != nullptr && by_hash && DrawPrep::VerifyMode() != 0 &&
			    !(permutation->specialization == source.specialization)) {
				// KYTY_DRAW_PREP_VERIFY: the hash found a permutation of another specialization.
				DrawPrep::Add(DrawPrep::Counter::VerifyMismatches);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					LOGF("DrawPrep verify: %s 0x%016" PRIx64 " specialization hash %016" PRIx64
					     " found a permutation of another specialization\n",
					     StageLabel(stage), params.hash, source.specialization_hash);
				}
				if (DrawPrep::VerifyMode() == 2) {
					EXIT("DrawPrep verify: specialization hash collision\n");
				}
			}
			match_phase.End();
			if (permutation != nullptr) {
				if (table_capture && table_capture_source_count < table_capture_sources.size()) {
					// KYTY_TABLE: the source and specialization a table hit must find again.
					table_capture_sources[table_capture_source_count++] = {
					    &source, source.specialization_hash_valid ? source.specialization_hash
					                                              : 0u};
				}
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &source.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		// A translation on the workers (TranslationWorkers): the draw waits for it by being skipped.
		if (!translating.empty()) {
			if (const auto job = translating.find(lookup_key);
			    job != translating.end() && !job->second->done.load(std::memory_order_acquire)) {
				translation_skips++;
				GpuPhases::Count(GpuPhases::Phase::TranslationSkip, 1);
				Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": translating",
				                                StageLabel(stage), params.hash);
				return ShaderProgram {};
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
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		options.non_fatal = ShaderFailureNonFatal();
		options.bindless_images = bindless_images;
		const auto compile_code = lookup_key.function_code.empty()
		                              ? params.code
		                              : std::span<const uint32_t>(lookup_key.function_code);
		seed_code = compile_code;
		// The shader disk cache (shaderDiskCache.h): a source entry stands in for the translation
		// of a source new to this session, a permutation entry for the compile. In verify mode
		// both are only compared with a fresh translation. A hit is not a compile: it is not
		// counted in ShadersCompiled or ShaderCompileTime.
		std::vector<uint8_t> source_key;
		std::vector<uint8_t> cached_source;
		std::vector<uint8_t> permutation_key;
		std::vector<uint8_t> cached_permutation;
		// A translation finished on the workers was looked up (and missed) when it was queued: a
		// second lookup only reads (or fails to open) the same file again and counts a second miss.
		// KYTY_SHADER_DISK_ONCE=0 looks again. The keys are still built (the result is saved under
		// them below).
		static const bool disk_once = [] {
			const char* v = std::getenv("KYTY_SHADER_DISK_ONCE");
			return v == nullptr || v[0] != '0';
		}();
		const bool translated_on_worker =
		    disk_once && entry == programs.end() && !translating.empty() &&
		    translating.contains(lookup_key);
		if (disk != nullptr && translated_on_worker) {
			BuildSourceKey(source_key, compile_code, params, options, input_info);
		} else if (disk != nullptr) {
			GpuPhases::Scope disk_phase(GpuPhases::Phase::ShaderDisk);
			Profiler::Phases disk_zone;
			disk_zone.SetText("%s 0x%016" PRIx64, stage_name, params.hash);
			KYTY_PROFILER_PHASE(disk_zone, "Shader disk cache", profiler::colors::CyanA700);
			BuildSourceKey(source_key, compile_code, params, options, input_info);
			// Round 134: a deferrable 3D draw's records are read on the workers (DiskReaders).
			const bool read_async =
			    disk_readers != nullptr && defer_translation && stage != ShaderType::Compute;
			const auto skip_for_read = [&] {
				disk_phase.Cancel();
				GpuPhases::Count(GpuPhases::Phase::DiskReadSkip, 1);
				Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": reading its shader",
				                                StageLabel(stage), params.hash);
			};
			if (entry == programs.end()) {
				std::vector<uint8_t>          payload;
				ShaderDiskCache::SourceRecord record;
				bool                          loaded = false;
				if (read_async) {
					if (TakeDiskRead(source_key, payload, loaded) != DiskReadState::Ready) {
						skip_for_read();
						return ShaderProgram {};
					}
				} else {
					loaded = disk->Load(source_key, payload);
				}
				CountDiskLookup();
				if (!loaded) {
					disk->counters.source_misses++;
					if (!lookup_key.function_code.empty()) {
						// Round rtexp: an expansion translated (and its pipeline made) in game.
						g_expansions_in_game.fetch_add(1, std::memory_order_relaxed);
					}
					// Round seedcs: whether the seeds hold this source (then its key differs from
					// the precompile's) or not (new to them). Only on a miss.
					int seeded = -1;
					if (PipelineSeeds::MissCheck()) {
						seeded = PipelineSeeds::HasSource(PipelineSeeds::SourceId(
						             static_cast<uint32_t>(stage), params.hash, compile_code))
						             ? 1
						             : 0;
						PipelineSeeds::NoteSourceMiss(seeded == 1);
					}
					// KYTY_SHADER_MISS_LOG=<n> (64 by default, 0: none): the first misses named, to
					// tell what a precompile did not cover (only on a miss: nothing per draw).
					static const uint32_t miss_log = [] {
						const char* v = std::getenv("KYTY_SHADER_MISS_LOG");
						return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10))
						                    : 64u;
					}();
					static std::atomic<uint32_t> misses_logged {0};
					if (misses_logged.load(std::memory_order_relaxed) < miss_log &&
					    misses_logged.fetch_add(1, std::memory_order_relaxed) < miss_log) {
						PipelineCacheLog("Shader disk cache miss: {} 0x{:016x}, {} words{}{}, key "
						                 "{:016x}{}",
						                 stage_name, params.hash, compile_code.size(),
						                 lookup_key.function_code.empty() ? "" : " (expanded)",
						                 read_async ? "" : " (read here)",
						                 XXH3_64bits(source_key.data(), source_key.size()),
						                 seeded == 1   ? ", in the seeds (another key)"
						                 : seeded == 0 ? ", not in the seeds"
						                               : "");
					}
				} else if (disk_verify) {
					disk->counters.source_hits++;
					cached_source = std::move(payload);
				} else if (!ShaderDiskCache::DecodeSourceRecord(payload, record)) {
					disk->counters.rejected++;
					disk->counters.source_misses++;
				} else {
					disk->counters.source_hits++;
					if (record.unsupported) {
						Profiler::Add(Profiler::Counter::ShadersGaveUp);
						Profiler::Message(Profiler::MessageFailure,
						                  "Shader gave up: %s 0x%016" PRIx64 ": in an earlier run (disk cache)",
						                  stage_name, params.hash);
						unsupported.insert(lookup_key);
						return ShaderProgram {};
					}
					if (record.uses_clock && !shader_clock) {
						ReportShaderClock(stage, stage_name, params.hash);
						unsupported.insert(lookup_key);
						return ShaderProgram {};
					}
					{
						std::unique_lock programs_lock(programs_mutex);
						entry = programs.try_emplace(lookup_key, std::move(record.plan)).first;
					}
					if (!MaterializeFirstUse(entry->second, runtime, stage, stage_name, params.hash)) {
						return ShaderProgram {};
					}
				}
			}
			if (entry != programs.end()) {
				BuildPermutationKey(permutation_key, source_key, entry->second.specialization,
				                    push_data_cursor);
				std::vector<uint8_t>               payload;
				ShaderDiskCache::PermutationRecord record;
				bool                               loaded = false;
				if (read_async) {
					if (TakeDiskRead(permutation_key, payload, loaded) != DiskReadState::Ready) {
						skip_for_read();
						return ShaderProgram {};
					}
				} else {
					loaded = disk->Load(permutation_key, payload);
				}
				if (!loaded) {
					disk->counters.permutation_misses++;
				} else if (disk_verify) {
					disk->counters.permutation_hits++;
					cached_permutation = std::move(payload);
				} else if (!ShaderDiskCache::DecodePermutationRecord(payload, record)) {
					disk->counters.rejected++;
					disk->counters.permutation_misses++;
				} else {
					disk->counters.permutation_hits++;
					Profiler::Message(Profiler::MessageInfo,
					                  "Shader from the disk cache: %s 0x%016" PRIx64 ", %zu SPIR-V words",
					                  stage_name, params.hash, record.spirv.size());
					if (!lookup_key.function_code.empty()) {
						gave_up_call_shaders.erase(params.hash);
					}
					auto handle = AddPermutation(entry->second, stage_name, options, record.spirv,
					                             std::move(record.program), input_info,
					                             push_data_cursor, InlinedCalls(params));
					NotePixelEntry<InputInfo>(handle, permutation_key);
					return handle;
				}
			}
		}

		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (call_partition != nullptr && !call_partition->targets.empty()) {
				// Round 99: a partition's program (ShaderCallPartition) translates on the workers
				// while its dispatch is skipped, as a 3D draw's shaders do: the other partitions
				// run meanwhile. Its SPIR-V is made here, at most as often as
				// GetComputeProgram's budget lets (partition_budget): a scene with new materials
				// brings dozens of partitions at once.
				if (translators != nullptr && !translating.contains(lookup_key)) {
					auto job = std::make_shared<TranslationJob>();
					job->code.assign(compile_code.begin(), compile_code.end());
					job->user_data.assign(user_data.begin(), user_data.end());
					job->options                    = options;
					job->options.user_data          = job->user_data;
					job->options.back_code          = {};
					job->compute                    = input_info;
					job->options.input_info.compute = &job->compute;
					job->queued                     = std::chrono::steady_clock::now();
					translating.emplace(lookup_key, job);
					translators->Submit(std::move(job));
					translation_skips++;
					GpuPhases::Count(GpuPhases::Phase::TranslationSkip, 1);
					Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": translating",
					                                StageLabel(stage), params.hash);
					return ShaderProgram {};
				}
				if (!partition_budget) {
					Profiler::CommandZone::Annotate("skipped %s %016" PRIx64
					                                ": its partition waits for its turn",
					                                StageLabel(stage), params.hash);
					return ShaderProgram {};
				}
				partition_built = true;
			}
		}
		// The whole compile as one zone on the timeline, its phases below it.
		GpuPhases::Scope compile_phase(GpuPhases::Phase::ShaderCompile);
		Profiler::Phases compile_zone;
		compile_zone.SetText("%s 0x%016" PRIx64, stage_name, params.hash);
		KYTY_PROFILER_PHASE(compile_zone, "Shader compile", profiler::colors::DeepOrangeA200);
		const auto compile_begin = std::chrono::steady_clock::now();
		ShaderRecompiler::TranslateResult translated;
		// Where the translation's resource tracking failed: on this thread, or on a worker's.
		uint32_t failure_pc      = UINT32_MAX;
		bool     translated_here = true;
		if (const auto job = translating.find(lookup_key); job != translating.end()) {
			// Done (the check above returned while it was not): take the worker's translation.
			const auto waited = std::chrono::duration<double, std::milli>(
			                        std::chrono::steady_clock::now() - job->second->queued)
			                        .count();
			if (translations_done++ < 32) {
				LOGF("ShaderTranslator: %s 0x%016" PRIx64 " ready after %.0f ms, %" PRIu64
				     " draws skipped so far while translating\n",
				     stage_name, params.hash, waited, translation_skips);
			}
			translated      = std::move(job->second->result);
			failure_pc      = job->second->failure_pc;
			translated_here = false;
			translating.erase(job);
		} else if (defer_translation && stage != ShaderType::Compute && translators != nullptr) {
			auto job = std::make_shared<TranslationJob>();
			job->code.assign(compile_code.begin(), compile_code.end());
			job->back_code.assign(params.back_code.begin(), params.back_code.end());
			job->user_data.assign(user_data.begin(), user_data.end());
			job->options           = options;
			job->options.user_data = job->user_data;
			job->options.back_code = job->back_code;
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				job->vertex                   = input_info;
				job->options.input_info.vertex = &job->vertex;
			} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
				job->pixel                   = input_info;
				job->options.input_info.pixel = &job->pixel;
			}
			job->queued = std::chrono::steady_clock::now();
			translating.emplace(lookup_key, job);
			translators->Submit(std::move(job));
			compile_phase.Cancel();
			translation_skips++;
			GpuPhases::Count(GpuPhases::Phase::TranslationSkip, 1);
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": translating",
			                                StageLabel(stage), params.hash);
			return ShaderProgram {};
		} else {
			translated = ShaderRecompiler::TranslateProgram(compile_code, options);
		}
		if (translated.unsupported) {
			if (translated_here) {
				failure_pc = ShaderRecompiler::IR::LastResourceTrackingFailurePc();
			}
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				DumpComputeForReplay("gaveup", params.hash, compile_code, user_data, input_info,
				                     failure_pc);
			}
			if (!lookup_key.function_code.empty()) {
				// A table target's inlined body the resource tracking could not follow: leave
				// it out of the next expansion (it takes the miss path) instead of giving up
				// the whole shader for every material.
				const auto pc = failure_pc;
				const auto target =
				    pc != UINT32_MAX
				        ? function_expander.ExcludeFailedCallTarget(lookup_key.function_code, pc)
				        : 0;
				// Not kept for the next run either (SaveCallTargets): its expansion would give up
				// again before the exclusion is learned. The exclusion itself is kept
				// (excluded_call_targets.txt, KYTY_CALL_EXCLUDED_KEEP), with a hash of the words
				// at the target, so the next run leaves the target out from its first expansion.
				if (target != 0) {
					excluded_call_targets.insert(target);
					NoteExcludedTarget(target);
				} else {
					gave_up_call_shaders.insert(params.hash);
				}
				static std::atomic<uint32_t> exclusion_reports {0};
				if (target != 0 && exclusion_reports.fetch_add(1) < 64) {
					LOGF("Shader function expansion hash=%016" PRIx64
					     ": excluding table target 0x%016" PRIx64
					     " (its body failed resource tracking at pc 0x%08x)\n",
					     params.hash, target, pc);
				}
				// The material passes of the ray tracing mode call their materials through tables
				// of runtime records: an expansion inlines only the targets the GPU reported
				// missing (ReportShaderCallMiss), and the first one, which inlines none, gave up
				// in resource tracking outside any inlined body, so the shader never ran to report
				// any (round 92's capture: 31 dispatches, the cars' reflections never shaded). A
				// compute shader in that state is expanded next with every target of its tables
				// (none when a table holds more than the expander inlines without reports).
				// KYTY_CALL_EVERY_TARGET=0: no such expansion.
				if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
					const auto reported =
					    ShaderRecompiler::Decoder::ShaderCallMissTargets(params.hash);
					if (target == 0 && ExpandEveryTarget() &&
					    ShaderRecompiler::Decoder::ShaderCallRuntimeRecords() &&
					    (reported == nullptr || reported->empty()) &&
					    !every_target_too_big.contains(params.hash) &&
					    expand_every_target.insert(params.hash).second) {
						LOGF("Shader function expansion hash=%016" PRIx64
						     ": its translation gave up before the GPU reported any table target; "
						     "it is expanded again with every table target\n",
						     params.hash);
					}
				}
			}
			// Remember the refusal: a skipped shader is dispatched again every frame, and
			// re-deriving the same answer costs as much as a compile each time. (Not for an
			// expansion: a later session learns its exclusions again.)
			if (disk != nullptr && entry == programs.end() && lookup_key.function_code.empty()) {
				std::vector<uint8_t> payload;
				if (ShaderDiskCache::EncodeSourceRecord(true, false, nullptr, payload)) {
					StoreEntry(true, params.hash, source_key, cached_source, payload);
				}
			}
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		const bool uses_clock = UsesShaderClock(translated.program);
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		if (entry == programs.end()) {
			plan.emplace(ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
			if (disk != nullptr) {
				std::vector<uint8_t> payload;
				if (ShaderDiskCache::EncodeSourceRecord(false, uses_clock, &*plan, payload)) {
					StoreEntry(true, params.hash, source_key, cached_source, payload);
				} else if (disk->counters.round_trip_failures++ < 16) {
					LOGF("Shader disk cache: the resource plan of hash=0x%016" PRIx64
					     " refers outside itself; not cached\n",
					     params.hash);
				}
			}
		}
		if (!shader_clock && uses_clock) {
			ReportShaderClock(stage, stage_name, params.hash);
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		if (entry == programs.end()) {
			{
				std::unique_lock programs_lock(programs_mutex);
				entry = programs.try_emplace(lookup_key, std::move(*plan)).first;
			}
			if (!MaterializeFirstUse(entry->second, runtime, stage, stage_name, params.hash)) {
				return ShaderProgram {};
			}
			if (disk != nullptr) {
				BuildPermutationKey(permutation_key, source_key, entry->second.specialization,
				                    push_data_cursor);
				if (disk_verify) {
					std::vector<uint8_t> payload;
					if (disk->Load(permutation_key, payload)) {
						cached_permutation = std::move(payload);
					}
				}
			}
		}
		auto compiled = CompileTranslated(options, std::move(translated),
		                                  entry->second.specialization, push_data_cursor);
		if (compiled.unsupported != nullptr) {
			// Skipped, as a shader that gives up is, instead of ending the emulator.
			const char* label = options.dump_label != nullptr ? options.dump_label : "Shader";
			::printf("%s gave up hash=0x%016" PRIx64 ": %s\n", label, params.hash,
			         compiled.unsupported);
			LOGF("%s gave up hash=0x%016" PRIx64 ": %s\n", label, params.hash,
			     compiled.unsupported);
			Profiler::Add(Profiler::Counter::SkippedGaveUp);
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (ComputeBackgroundWords() != 0 &&
			    compiled.spirv.size() >= ComputeBackgroundWords()) {
				DumpComputeForReplay("big", params.hash, compile_code, user_data, input_info, 0);
			}
		}
		std::vector<uint8_t> permutation_payload;
		if (disk != nullptr) {
			ShaderDiskCache::EncodePermutationRecord(compiled.spirv, compiled.program,
			                                         permutation_payload);
		}
		if (!lookup_key.function_code.empty()) {
			gave_up_call_shaders.erase(params.hash); // a later expansion of it translated
		}
		const auto handle = AddPermutation(entry->second, stage_name, options, compiled.spirv,
		                                   std::move(compiled.program), input_info,
		                                   push_data_cursor, InlinedCalls(params));
		{
			const auto us = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                         std::chrono::steady_clock::now() - compile_begin)
			                                         .count());
			Profiler::Add(Profiler::Counter::ShadersCompiled);
			Profiler::Add(Profiler::Counter::ShaderCompileTime, us);
			Profiler::Message(Profiler::MessageInfo,
			                  "Shader compiled: %s 0x%016" PRIx64 " in %.1f ms, %zu SPIR-V words",
			                  stage_name, params.hash, static_cast<double>(us) / 1000.0,
			                  last_spirv_words);
		}
		compile_zone.End();
		if (disk != nullptr) {
			// After AddPermutation: SPIR-V that fails validation is never stored.
			StoreEntry(false, params.hash, permutation_key, cached_permutation,
			           permutation_payload);
			NotePixelEntry<InputInfo>(handle, permutation_key);
		}
		return handle;
	}

	ProgramCache(vk::Device device, bool shader_clock, bool bindless_images,
	             ShaderDiskCache::SessionInfo session, std::filesystem::path disk_directory)
	    : device(device), shader_clock(shader_clock), bindless_images(bindless_images),
	      disk_session(session), disk_verify(ShaderDiskCache::VerifyEnabled()) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		if (Config::GraphicsDebugDumpEnabled()) {
			// The dumps are written while translating: every shader has to be translated.
			PipelineCacheLog("Shader disk cache: off (graphics debug dump)");
		} else {
			LogShaderCacheFolder(disk_directory);
			disk = ShaderDiskCache::Store::Open(std::move(disk_directory));
			if (disk != nullptr && ShaderCachePruneEnabled()) {
				// Only the key's first 16 bytes are compared (its format and the recompiler's
				// version), which nothing at run time changes.
				const auto key = ShaderDiskCache::SessionKey(disk_session);
				if (key.size() >= ShaderCachePrefix {}.size()) {
					ShaderCachePrefix prefix {};
					std::copy_n(key.begin(), prefix.size(), prefix.begin());
					disk_pruner = std::jthread(PruneShaderCache, disk->Directory(), prefix);
				}
			}
		}
		if (disk != nullptr) {
			call_targets_path = disk->Directory() / "call_targets.txt";
			LoadCallTargets();
			excluded_targets_path = disk->Directory() / "excluded_call_targets.txt";
			LoadExcludedTargets();
		}
		if (g_async_translation) {
			// KYTY_TRANSLATION_THREADS=<n> (default a third of the processors, 2 to 12, at below
			// normal priority; a quarter, 2 to 8, since round 86; an eighth, 2 to 4, before; 2
			// before round 70). They work only while new
			// shaders are translated, a first run's long stretch of skipped draws.
			const char* value   = std::getenv("KYTY_TRANSLATION_THREADS");
			const auto  threads = std::max<uint32_t>(
                value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
                                 : std::clamp(std::thread::hardware_concurrency() / 3u, 2u, 12u),
			    1u);
			translators = std::make_unique<TranslationWorkers>(threads);
			PipelineCacheLog("Shader translation: graphics shaders of 3D draws on {} worker threads",
			                 threads);
			// Round 134: KYTY_SHADER_READ_THREADS=<n> (default 4; 0 reads on the GPU thread, as
			// before) read the disk cache's files of 3D draws' shaders.
			const char* reads_value = std::getenv("KYTY_SHADER_READ_THREADS");
			const auto  read_threads =
                reads_value != nullptr && *reads_value != '\0'
			         ? static_cast<uint32_t>(std::strtoul(reads_value, nullptr, 10))
			         : 4u;
			if (disk != nullptr && read_threads != 0 && !disk_verify) {
				disk_readers = std::make_unique<DiskReaders>(*disk, std::min(read_threads, 16u));
				PipelineCacheLog("Shader disk cache: files of 3D draws' shaders read on {} worker "
				                 "threads (KYTY_SHADER_READ_THREADS)",
				                 std::min(read_threads, 16u));
			}
		}
		if (g_stage_prep_parallel) {
			stage_prep = std::make_unique<StagePrepWorker>();
			PipelineCacheLog("Stage preparation: vertex and pixel SRT refresh in parallel");
		}
	}

	void FlushDisk() {
		if (disk != nullptr) {
			disk->Flush();
			disk->LogTotals("saved");
		}
		SaveCallTargets();
		SaveExcludedTargets();
		// Round rtfix: translations (partitions of table calls among them) whose program was never
		// made before the end of the run: neither their source nor their permutation is on disk,
		// so the next run translates them again under the same key.
		if (!translating.empty()) {
			size_t expansions = 0;
			for (const auto& [key, job]: translating) {
				(void)job;
				expansions += key.function_code.empty() ? 0u : 1u;
			}
			PipelineCacheLog("Shader translations left unbuilt at the end: {} ({} expansions)",
			                 translating.size(), expansions);
		}
	}

	// Round rtfix: table targets whose inlined bodies failed resource tracking
	// (ExcludeFailedCallTarget), kept between runs: each run learned them again, one failed
	// translation of the expansion (and a miss of the shader disk cache under the same key as the
	// run before) per target. A kept target is excluded only once the guest words at its address
	// hash as they did when it failed (ExcludedWordsHash), so other code at that address in a
	// later run is not left out. KYTY_CALL_EXCLUDED_KEEP=0: neither read nor written.
	static bool KeepExcludedTargets() {
		static const bool value = [] {
			const char* text = std::getenv("KYTY_CALL_EXCLUDED_KEEP");
			return text == nullptr || text[0] != '0';
		}();
		return value;
	}
	static constexpr const char* ExcludedTargetsHeader = "kyty excluded call targets 1";
	static constexpr size_t      ExcludedHashWords     = 64;

	// 0 when the words cannot be read.
	static uint64_t ExcludedWordsHash(uint64_t target) {
		std::array<uint32_t, ExcludedHashWords> words {};
		if (!ReadShaderGuestMemoryRaw(nullptr, target, words)) {
			return 0;
		}
		const auto hash = XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
		return hash != 0 ? hash : 1;
	}

	void NoteExcludedTarget(uint64_t target) {
		if (!KeepExcludedTargets() || excluded_hashes.size() >= 65536) {
			return;
		}
		if (const auto hash = ExcludedWordsHash(target); hash != 0) {
			excluded_hashes[target] = hash;
		}
	}

	void LoadExcludedTargets() {
		if (!KeepExcludedTargets() || excluded_targets_path.empty()) {
			return;
		}
		std::ifstream file(excluded_targets_path);
		if (!file) {
			return;
		}
		std::string line;
		if (!std::getline(file, line)) {
			return;
		}
		if (!line.empty() && line.back() == '\r') {
			line.pop_back();
		}
		if (line != ExcludedTargetsHeader) {
			return;
		}
		while (std::getline(file, line) && pending_exclusions.size() < 65536) {
			uint64_t target = 0;
			uint64_t hash   = 0;
			if (std::sscanf(line.c_str(), "%" SCNx64 " %" SCNx64, &target, &hash) == 2 &&
			    target != 0 && hash != 0) {
				pending_exclusions.emplace_back(target, hash);
			}
		}
		PipelineCacheLog("Table call targets: {} excluded by earlier runs (applied once their code "
		                 "matches)",
		                 pending_exclusions.size());
	}

	// Called before expansions: the kept exclusions whose code is readable now are applied (when
	// it is the code that failed) or dropped (other code).
	void ApplyPendingExclusions() {
		if (pending_exclusions.empty() || (pending_tries++ & 63u) != 0) {
			return;
		}
		std::erase_if(pending_exclusions, [&](const std::pair<uint64_t, uint64_t>& kept) {
			const auto hash = ExcludedWordsHash(kept.first);
			if (hash == 0) {
				return false;
			}
			if (hash == kept.second) {
				(void)function_expander.ExcludeCallTarget(kept.first);
				excluded_call_targets.insert(kept.first);
				excluded_hashes[kept.first] = hash;
				exclusions_applied++;
			}
			return true;
		});
	}

	void SaveExcludedTargets() {
		if (!KeepExcludedTargets() || excluded_targets_path.empty() ||
		    (excluded_hashes.empty() && pending_exclusions.empty())) {
			return;
		}
		std::string text = std::string(ExcludedTargetsHeader) + "\n";
		for (const auto& [target, hash]: excluded_hashes) {
			text += fmt::format("{:016x} {:016x}\n", target, hash);
		}
		// Not seen this run: kept for a later one.
		for (const auto& [target, hash]: pending_exclusions) {
			if (!excluded_hashes.contains(target)) {
				text += fmt::format("{:016x} {:016x}\n", target, hash);
			}
		}
		auto temp_path = excluded_targets_path;
		temp_path += ".tmp";
		{
			std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
			file << text;
			if (!file.flush()) {
				return;
			}
		}
		std::error_code error;
		std::filesystem::rename(temp_path, excluded_targets_path, error);
		PipelineCacheLog("Table call targets: {} excluded kept for the next run ({} applied from "
		                 "earlier runs)",
		                 excluded_hashes.size(), exclusions_applied);
	}

	// Round 95: the table call targets the GPU reported (ReportShaderCallMiss), kept between runs
	// in the shader cache folder (call_targets.txt; KYTY_CALL_TARGETS_KEEP=0: neither read nor
	// written). A shader's expansion inlines the targets reported so far, in their order, so
	// targets that arrive in another order or batches make other expansions and other SPIR-V: a
	// second visit to the garage compiled its ray tracing material passes again (a minute or more
	// each) for other sets of the same materials. Reported again at the start, the targets of the
	// earlier runs make the first expansion the last one of those runs, whose pipeline the
	// driver's cache holds. Only targets without record data (runtime records) are kept: those
	// specialized on a record follow data the game allocates anew.
	static bool KeepCallTargets() {
		static const bool value = [] {
			const char* text = std::getenv("KYTY_CALL_TARGETS_KEEP");
			return text == nullptr || text[0] != '0';
		}();
		return value;
	}

	void LoadCallTargets() {
		if (!KeepCallTargets() || call_targets_path.empty()) {
			return;
		}
		std::ifstream file(call_targets_path);
		if (!file) {
			return;
		}
		std::string line;
		const auto  next_line = [&] {
            if (!std::getline(file, line)) {
                return false;
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return true;
		};
		if (!next_line() || line != CallTargetsHeader) {
			// Written anew at the end of the run (only this file's writer reads it).
			PipelineCacheLog("Table call targets: {} is not one this build reads; it is written "
			                 "anew",
			                 Common::PathToString(call_targets_path));
			return;
		}
		uint64_t targets = 0;
		while (next_line() && targets < MaxKeptCallTargets) {
			uint64_t hash   = 0;
			uint64_t target = 0;
			if (std::sscanf(line.c_str(), "%" SCNx64 " %" SCNx64, &hash, &target) != 2 ||
			    hash == 0 || target == 0) {
				continue;
			}
			(void)ShaderRecompiler::Decoder::ReportShaderCallMiss(hash, {target, 0u, 0u});
			call_shaders.insert(hash);
			targets++;
		}
		PipelineCacheLog("Table call targets: {} of {} shaders from earlier runs", targets,
		                 call_shaders.size());
	}

	void SaveCallTargets() {
		if (!KeepCallTargets() || call_targets_path.empty() || call_shaders.empty()) {
			return;
		}
		std::vector<uint64_t> hashes(call_shaders.begin(), call_shaders.end());
		std::sort(hashes.begin(), hashes.end());
		std::string text    = std::string(CallTargetsHeader) + "\n";
		uint64_t    targets = 0;
		for (const auto hash: hashes) {
			// A shader whose last expansion gave up outside any inlined body starts from no
			// targets again, as before round 95 (from its last set it would give up at once).
			const auto reported = gave_up_call_shaders.contains(hash)
			                          ? nullptr
			                          : ShaderRecompiler::Decoder::ShaderCallMissTargets(hash);
			if (reported == nullptr) {
				continue;
			}
			for (const auto& variant: *reported) {
				if (variant.data0 != 0u || variant.data1 != 0u || targets >= MaxKeptCallTargets ||
				    excluded_call_targets.contains(variant.target)) {
					continue;
				}
				text += fmt::format("{:016x} {:016x}\n", hash, variant.target);
				targets++;
			}
		}
		auto temp_path = call_targets_path;
		temp_path += ".tmp";
		{
			std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
			file << text;
			if (!file.flush()) {
				PipelineCacheLog("Table call targets: failed to write {}",
				                 Common::PathToString(temp_path));
				return;
			}
		}
		std::error_code error;
		std::filesystem::rename(temp_path, call_targets_path, error);
		if (error) {
			PipelineCacheLog("Table call targets: failed to write {}",
			                 Common::PathToString(call_targets_path));
			return;
		}
		PipelineCacheLog("Table call targets: {} kept for the next run", targets);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
				if (permutation.handle.pixel_rate_module != nullptr) {
					device.destroyShaderModule(permutation.handle.pixel_rate_module, nullptr);
				}
				if (permutation.handle.normalized_module != nullptr) {
					device.destroyShaderModule(permutation.handle.normalized_module, nullptr);
				}
			}
		}
	}

	// The module NV mixed-sample draws use for a pixel shader that reads its sample index
	// (SpirvWithoutSampleId), or null. Reported for the first shaders that get one.
	vk::ShaderModule CompilePixelRateVariant(const ShaderRecompiler::CompileOptions& options,
	                                         const std::vector<uint32_t>&            spirv) {
		auto variant = SpirvWithoutSampleId(spirv);
		if (variant.empty()) {
			LOGF("Pipeline: pixel shader 0x%016" PRIx64
			     " reads its sample index in a form the per-pixel copy does not rewrite; its EQAA "
			     "draws are skipped\n",
			     options.shader_hash);
			return nullptr;
		}
		// The rewrite is mechanical, but a broken copy is a broken pipeline: the validator checks
		// the first copies whatever --shader-validation says, and every copy after one fails.
		// Four, not 32 (round 87): no copy has ever failed, and checking 32 took the GPU thread
		// 1.8 s at every start (the game's are large).
		constexpr uint32_t CheckedCopies = 4;
		if (pixel_rate_checked < CheckedCopies || pixel_rate_failed ||
		    Config::ShaderValidationEnabled()) {
			if (!ValidateShaderSpirvNow(options.dump_label, options.shader_hash, variant)) {
				pixel_rate_failed = true;
				LOGF("Pipeline: the per-pixel copy of pixel shader 0x%016" PRIx64
				     " is not valid SPIR-V; its EQAA draws are skipped\n",
				     options.shader_hash);
				return nullptr;
			}
			pixel_rate_checked++;
		}
		const auto                   variant_module = CompileSPV(variant, device);
		static std::atomic<uint32_t> logged {0};
		if (variant_module != nullptr && logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Pipeline: pixel shader 0x%016" PRIx64
			     " reads its sample index: EQAA draws run a per-pixel copy (index 0)\n",
			     options.shader_hash);
		}
		return variant_module;
	}

	// KYTY_EXACT_FLOAT=1: the copy of a stage before the rasterizer whose float arithmetic the
	// driver may not fuse and whose position is invariant (SpirvExactFloat), or empty for the
	// shader as it is. The first copies are validated, and every copy after one that fails; a copy
	// that fails is not used (that shader alone runs as it is), and the log says whether the
	// shader itself was valid. After eight failures (each logs both modules) no more copies.
	std::vector<uint32_t> ExactFloatVariant(const ShaderRecompiler::CompileOptions& options,
	                                        const std::vector<uint32_t>&            spirv) {
		constexpr uint32_t MaxFailures = 8;
		if (exact_float_failures >= MaxFailures) {
			return {};
		}
		if (!exact_float_logged) {
			exact_float_logged = true;
			LOGF("Pipeline: KYTY_EXACT_FLOAT: %s keep every float multiply and add apart, and "
			     "their positions are invariant\n",
			     ExactFloatSettingValue().all ? "vertex-stage shaders"
			                                  : fmt::format("the {} vertex-stage shaders listed",
			                                                ExactFloatSettingValue().hashes.size())
			                                        .c_str());
		}
		auto variant = SpirvExactFloat(spirv);
		if (variant.empty()) {
			return {};
		}
		constexpr uint32_t CheckedCopies = 32;
		if (exact_float_rewritten < CheckedCopies || exact_float_failures > 0 ||
		    Config::ShaderValidationEnabled()) {
			if (!ValidateShaderSpirvNow(options.dump_label, options.shader_hash, variant)) {
				exact_float_failures++;
				const bool original_valid =
				    ValidateShaderSpirvNow(options.dump_label, options.shader_hash, spirv);
				LOGF("Pipeline: KYTY_EXACT_FLOAT: the copy of shader 0x%016" PRIx64
				     " is not valid SPIR-V (%s); the shader runs as it is (%u such so far)\n",
				     options.shader_hash,
				     original_valid ? "the shader itself is: the copy is wrong"
				                    : "nor is the shader itself",
				     exact_float_failures);
				if (exact_float_failures == MaxFailures) {
					LOGF("Pipeline: KYTY_EXACT_FLOAT: off from now on (%u copies failed); %u "
					     "shaders "
					     "keep their copies\n",
					     exact_float_failures, exact_float_rewritten);
				}
				return {};
			}
		}
		exact_float_rewritten++;
		return variant;
	}

	// The copy of a hull shader whose explicit-LOD samples take normalized coordinates
	// (SpirvNormalizedExplicitLod), or null. Validated like the exact-float copies: the first ones
	// and every one after a failure; one that fails is not used.
	vk::ShaderModule CompileNormalizedVariant(const ShaderRecompiler::CompileOptions& options,
	                                          const std::vector<uint32_t>&            spirv) {
		constexpr uint32_t MaxFailures = 8;
		if (normalized_failures >= MaxFailures) {
			return nullptr;
		}
		const auto variant = SpirvNormalizedExplicitLod(spirv);
		if (variant.empty()) {
			return nullptr;
		}
		constexpr uint32_t CheckedCopies = 32;
		if (normalized_copies < CheckedCopies || normalized_failures > 0 ||
		    Config::ShaderValidationEnabled()) {
			if (!ValidateShaderSpirvNow(options.dump_label, options.shader_hash, variant)) {
				normalized_failures++;
				LOGF("Pipeline: the normalized-LOD copy of hull shader 0x%016" PRIx64
				     " is not valid SPIR-V; its draws sample as before (%u such so far)\n",
				     options.shader_hash, normalized_failures);
				return nullptr;
			}
		}
		const auto module = CompileSPV(variant, device);
		if (module != nullptr && normalized_copies++ < 8) {
			LOGF("Pipeline: hull shader 0x%016" PRIx64
			     " samples at an explicit LOD: a copy with normalized coordinates is used where "
			     "its samplers are unnormalized\n",
			     options.shader_hash);
		}
		return module;
	}

	// Shader function expansion decodes the shader and every callee, and a shader with table
	// calls can be dispatched hundreds of times a frame; the expander keeps what it can reuse.
	void ExpandShaderFunctions(const ShaderParams& params, std::span<const uint32_t> user_data,
	                           uint32_t wave_size, std::vector<uint32_t>& expanded) {
		// KYTY_CALL_EXPANSION=0: no shader function expansion (shaders with calls give up, as
		// before it existed) — an escape hatch while table-call expansion is new.
		static const bool expansion_enabled = [] {
			const char* value = std::getenv("KYTY_CALL_EXPANSION");
			return value == nullptr || value[0] != '0';
		}();
		if (!expansion_enabled) {
			expanded.clear();
			return;
		}
		ApplyPendingExclusions();
		// Round 82 (from the Frank fork, 1165e585 / 76e4833d): a shader the expander found to make
		// no calls is not expanded again while the shader map's hash of its code is the same (the
		// program lookup trusts that hash too): the expander compared the whole code at every
		// dispatch to see that it had not changed. KYTY_BISECT's expand:old expands it each time.
		const auto code_key = reinterpret_cast<uint64_t>(params.code.data());
		if (params.hash != 0 && !Bisect::ExpandOld()) {
			if (const auto it = call_free.find(code_key); it != call_free.end() &&
			                                              it->second.hash == params.hash &&
			                                              it->second.words == params.code.size()) {
				expanded.clear();
				return;
			}
		}
		std::string reason;
		std::string summary;
		// Round 93: a shader whose expansion with the reported table targets alone could not be
		// translated is expanded with every target of its tables (ExpandEveryTarget).
		if (const auto every = expand_every_target.find(params.hash);
		    every != expand_every_target.end()) {
			// Reports arrived after all (a target the expansion left out ran): from now on the
			// expansion follows them, as for any shader.
			const auto reported = ShaderRecompiler::Decoder::ShaderCallMissTargets(params.hash);
			if (reported != nullptr && !reported->empty()) {
				expand_every_target.erase(every);
			}
		}
		const auto expand = [&](uint64_t expand_hash) {
			return function_expander.Expand(
			    params.code, params.Base(), user_data,
			    [](uint64_t address, std::span<uint32_t> words) {
				    return ReadShaderGuestMemoryRaw(nullptr, address, words);
			    },
			    expanded, reason, wave_size, expand_hash, &summary,
			    expand_hash != 0 ? call_partition : nullptr);
		};
		bool ok = false;
		if (expand_every_target.contains(params.hash)) {
			ok = expand(0);
			// Round 94: an expansion with every target of more than KYTY_CALL_EVERY_TARGET_WORDS
			// words (default 16384; 0: no limit) follows the reports again: 108 material functions
			// of a table of 198 made one of 120,888 words, whose translation kept the GPU thread
			// busy for good (the garage stopped).
			if (ok && EveryTargetWords() != 0 && expanded.size() > EveryTargetWords()) {
				LOGF("Shader function expansion hash=%016" PRIx64 ": with every table target it "
				     "has %zu words, more than %u; it follows the reported targets again\n",
				     params.hash, expanded.size(), EveryTargetWords());
				expand_every_target.erase(params.hash);
				every_target_too_big.insert(params.hash);
				expanded.clear();
				reason.clear();
				summary.clear();
				ok = expand(params.hash);
			}
		} else {
			ok = expand(params.hash);
		}
		if (ok && !expanded.empty() && params.hash != 0 && call_shaders.size() < 65536) {
			call_shaders.insert(params.hash);
		}
		if (ok && expanded.empty() && summary.empty() && params.hash != 0) {
			if (call_free.size() >= 65536) {
				call_free.clear();
			}
			call_free[code_key] = {params.hash, params.code.size()};
		}
		if (!summary.empty()) {
			// A table call left targets out: they take the miss trap, and the GPU reports the
			// ones that run (ReportShaderCallMiss) for the next expansion. Logged once per
			// distinct summary of a shader (the table sizes are left out of the comparison).
			const auto inlined = summary.find("inlined=");
			const auto key     = fmt::format("{:016x} {}", params.hash,
			                                 inlined != std::string::npos ? summary.substr(inlined)
			                                                              : summary);
			if (partial_logged.size() < 4096 && partial_logged.insert(key).second) {
				LOGF("Shader function expansion hash=%016" PRIx64 ": %s%s\n", params.hash,
				     summary.starts_with("partition") ? "" : "partial: ", summary.c_str());
			}
		}
		if (!ok) {
			// Once per shader and reason: a failing shader is expanded again every dispatch.
			if (failure_logged.size() < 4096 &&
			    failure_logged.insert(fmt::format("{:016x} {}", params.hash, reason)).second) {
				::printf("Shader function expansion hash=%016" PRIx64 ": %s\n", params.hash,
				         reason.c_str());
				LOGF("Shader function expansion hash=%016" PRIx64 ": %s\n", params.hash,
				     reason.c_str());
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash>    programs;
	// Round 97: set by GetComputeProgram for one Get: the expansion to use instead of expanding.
	const std::vector<uint32_t>* forced_function_code = nullptr;
	// Round 99: set by GetComputeProgram for one Get: the partition to expand
	// (ShaderCallPartition), whether a new partition's SPIR-V may be made now, and whether it was.
	ShaderRecompiler::Decoder::ShaderCallPartition* call_partition   = nullptr;
	bool                                            partition_budget = false;
	bool                                            partition_built  = false;
	// Taken exclusively by the GPU thread when it inserts into programs (entries are never
	// erased, so pointers to them stay valid), shared by the draw-prep scanner's lookups.
	std::shared_mutex                                              programs_mutex;
	std::unordered_set<ProgramKey, ProgramKeyHash>                 unsupported;
	ShaderRecompiler::Decoder::ShaderFunctionExpander             function_expander;
	// Round 93: compute shaders expanded with every table target (ExpandEveryTarget); round 94:
	// and those whose expansion with every target was too big for it (EveryTargetWords).
	std::unordered_set<uint64_t> expand_every_target;
	std::unordered_set<uint64_t> every_target_too_big;
	// Shaders the expander found to make no calls, by code address: the hash and size of the code
	// it saw (round 82).
	struct CallFreeCode {
		uint64_t hash  = 0;
		size_t   words = 0;
	};
	std::unordered_map<uint64_t, CallFreeCode>                     call_free;
	std::unordered_set<std::string>                               partial_logged;
	std::unordered_set<std::string>                               failure_logged;
	ProgramKey                                                     lookup_key;
	// gt7-prog-1112, KYTY_PROG_MEMO (off by default): per stage and shader code address, the last
	// key Get found in `programs` and the iterator it found. A hit needs the whole key equal
	// (stage, the shader map's hash, user data count, code size, the static state and the
	// expanded code, compared word for word), so it finds what programs.find would: the memo only
	// skips hashing the key twice (unsupported, programs) and the bucket search. The iterator is
	// kept only while `programs` has the same size and bucket count (entries are never erased, and
	// only an insertion, which changes the size, can rehash), and `unsupported` the same size (it
	// only grows: a key added to it after the store misses, and is checked as before).
	using ProgramMap = std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash>;
	struct ProgramMemo {
		bool                  valid           = false;
		ShaderType            stage           = ShaderType::Unknown;
		const uint32_t*       code            = nullptr;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		size_t                unsupported_size = 0;
		size_t                programs_size    = 0;
		size_t                bucket_count     = 0;
		std::vector<uint32_t> static_state;
		std::vector<uint32_t> function_code;
		ProgramMap::iterator  entry;

		[[nodiscard]] bool Matches(const ProgramKey& key, const uint32_t* code_address,
		                           size_t unsupported_now, size_t programs_now,
		                           size_t buckets_now) const {
			return valid && code == code_address && stage == key.stage && hash == key.hash &&
			       user_data_count == key.user_data_count && code_size == key.code_size &&
			       unsupported_size == unsupported_now && programs_size == programs_now &&
			       bucket_count == buckets_now && static_state == key.static_state &&
			       function_code == key.function_code;
		}
		void Store(const ProgramKey& key, const uint32_t* code_address, size_t unsupported_now,
		           size_t programs_now, size_t buckets_now, ProgramMap::iterator found) {
			valid            = true;
			stage            = key.stage;
			code             = code_address;
			hash             = key.hash;
			user_data_count  = key.user_data_count;
			code_size        = key.code_size;
			unsupported_size = unsupported_now;
			programs_size    = programs_now;
			bucket_count     = buckets_now;
			static_state.assign(key.static_state.begin(), key.static_state.end());
			function_code.assign(key.function_code.begin(), key.function_code.end());
			entry = found;
		}
	};
	static constexpr size_t           ProgramMemoSlots = 512;
	std::array<ProgramMemo, ProgramMemoSlots> program_memo {};
	[[nodiscard]] static bool ProgMemoEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_PROG_MEMO");
			return value != nullptr && value[0] != '\0' && value[0] != '0';
		}();
		return enabled;
	}
	// Direct-mapped by code address and stage: a slot another shader took is a miss, then
	// replaced.
	ProgramMemo& MemoSlotFor(ShaderType stage, const uint32_t* code_address) {
		auto index = reinterpret_cast<uintptr_t>(code_address) >> 8u;
		index ^= index >> 9u;
		index ^= static_cast<uintptr_t>(stage) * 0x9e37u;
		return program_memo[index & (ProgramMemoSlots - 1u)];
	}
	vk::Device                                                  device;
	bool                                                        shader_clock = false;
	bool                                                        bindless_images = false;
	// The device draws EQAA through NV mixed samples: pixel shaders that read their sample index
	// get a per-pixel copy too (CompilePixelRateVariant), the first ones validated.
	bool                                                        pixel_rate_variants = false;
	uint32_t                                                    pixel_rate_checked  = 0;
	bool                                                        pixel_rate_failed   = false;
	// KYTY_EXACT_FLOAT's copies (ExactFloatVariant): how many, and how many failed validation.
	// CompileNormalizedVariant's copies: how many were made, and how many failed validation.
	uint32_t                                                    normalized_copies     = 0;
	uint32_t                                                    normalized_failures   = 0;
	uint32_t                                                    exact_float_rewritten = 0;
	uint32_t                                                    exact_float_failures  = 0;
	bool                                                        exact_float_logged    = false;
	uint64_t                                                    next_shader_id = 0;
	// KYTY_TABLE (tableMode.h): while table_capture is set (GetGraphicsPrograms' table path, GPU
	// thread), each TakeScanned that serves a stage notes the read set it validated, in call order;
	// a stage served otherwise (a walk here) leaves table_capture_count short.
	bool                                      table_capture       = false;
	uint32_t                                  table_capture_count = 0;
	std::array<const DrawPrep::ReadSet*, 4>   table_capture_reads {};
	// ... and each Get that found its permutation: its source entry and specialization hash (0:
	// none made with the refresh), in call order (pixel first, then the vertex stages).
	uint32_t                                                 table_capture_source_count = 0;
	std::array<std::pair<SourceEntry*, uint64_t>, 4>         table_capture_sources {};

	// KYTY_TABLE: a programs table hit's refresh of one stage: the draw-prep scanner's result for
	// this packet (TakeScanned, from this draw's user data and the bytes it read, validated), and
	// true when it has the specialization the entry's permutation was made for. Otherwise the
	// draw takes the normal path (which finds the scanner's result again, or walks).
	bool TableRefresh(SourceEntry& entry, ShaderType stage, const ShaderParams& params,
	                  uint64_t specialization_hash) {
		if (specialization_hash == 0 || !DrawPrep::Enabled() || Bisect::DrawPrepOff() ||
		    SkipShaderRequested(params.hash)) {
			return false;
		}
		GuestPageReadCache               clean_read_cache;
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data   = std::span(params.user_data).first(params.user_data_count),
		    .shader_base = params.Base(),
		    .read_memory = ReadShaderGuestMemoryRaw,
		    .userdata    = &clean_read_cache,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		runtime.map_clean_page = +[](void* userdata, uint64_t page) {
			return static_cast<GuestPageReadCache*>(userdata)->Map(page);
		};
		runtime.page_userdata = &clean_read_cache;
		runtime.page_shift    = 12;
		entry.resources.gpu_fills.clear();
		bool taken = false;
		{
			const GpuPhases::Scope phase(GpuPhases::Phase::ProgramScanned);
			taken = TakeScanned(entry, stage, runtime, params.hash);
		}
		return taken && entry.specialization_hash_valid &&
		       entry.specialization_hash == specialization_hash;
	}
	// SPIR-V size of the last permutation compiled (for the profiler).
	size_t                                                      last_spirv_words = 0;
	// The shader disk cache (null when off) and its session key, built at first use.
	ShaderDiskCache::SessionInfo                                disk_session;
	std::vector<uint8_t>                                        disk_session_key;
	bool                                                        disk_verify  = false;
	uint64_t                                                    disk_lookups = 0;
	std::unique_ptr<ShaderDiskCache::Store>                     disk;
	// Round 159: told each pixel shader permutation's disk entry (PipelineCache::m_coverage).
	CoverageVariants* coverage = nullptr;
	// Round 95: call_targets.txt in the shader cache folder, and the shaders whose table call
	// targets go into it (LoadCallTargets, SaveCallTargets).
	static constexpr const char* CallTargetsHeader  = "kyty table call targets 1";
	static constexpr uint64_t    MaxKeptCallTargets = 65536;
	std::filesystem::path        call_targets_path;
	std::unordered_set<uint64_t> call_shaders;
	// Round rtfix: excluded_call_targets.txt (LoadExcludedTargets, SaveExcludedTargets).
	std::filesystem::path                          excluded_targets_path;
	std::unordered_map<uint64_t, uint64_t>         excluded_hashes;
	std::vector<std::pair<uint64_t, uint64_t>>     pending_exclusions;
	uint64_t                                       pending_tries      = 0;
	uint64_t                                       exclusions_applied = 0;
	// Targets an expansion left out after their inlined bodies failed resource tracking, and
	// shaders whose expansion gave up outside any inlined body: not kept.
	std::unordered_set<uint64_t> excluded_call_targets;
	std::unordered_set<uint64_t> gave_up_call_shaders;
	// Removes the disk cache's entries no build of this recompiler can read (PruneShaderCache).
	std::jthread disk_pruner;
	// Graphics translations in progress on the workers (KYTY_ASYNC_TRANSLATION), by program.
	std::unordered_map<ProgramKey, std::shared_ptr<TranslationJob>, ProgramKeyHash> translating;
	uint64_t                                                    translation_skips = 0;
	uint64_t                                                    translations_done = 0;
	// Set by GetGraphicsPrograms for the draw at hand: may its shaders translate in the background?
	bool                                                        defer_translation = false;
	// The entries whose snapshot the parallel window refreshed for the draw at hand
	// (PrepareStagesInParallel); Get takes them instead of walking again.
	std::array<SourceEntry*, 2>                                 prepared {};
	ProgramKey                                                  prep_key;
	uint64_t                                                    prep_windows    = 0;
	uint64_t                                                    prep_vs_ok      = 0;
	uint64_t                                                    prep_ps_ok      = 0;
	uint64_t                                                    prep_declined   = 0;
	uint64_t                                                    prep_mismatches = 0;
	std::unique_ptr<StagePrepWorker>                            stage_prep;
	// Round 134: shader disk cache reads in progress on the workers (DiskReaders), by key.
	std::unordered_map<std::string, std::shared_ptr<DiskRead>> disk_reads;
	std::unique_ptr<DiskReaders>                               disk_readers;
	// Last: destroyed first, so its threads have stopped before anything else goes.
	std::unique_ptr<TranslationWorkers>                         translators;

	// Round 134, GPU thread: the record under `key` read on a worker. Ready: the read is done
	// (`found` says whether the file held it, `payload` has it); otherwise the read is on its way
	// (asked for now, or before), and the draw waits by being skipped.
	enum class DiskReadState : uint8_t { Ready, Reading };
	DiskReadState TakeDiskRead(std::span<const uint8_t> key, std::vector<uint8_t>& payload,
	                           bool& found) {
		std::string name(reinterpret_cast<const char*>(key.data()), key.size());
		const auto  it = disk_reads.find(name);
		if (it == disk_reads.end()) {
			if (disk_reads.size() >= 8192) {
				// Reads nobody took (a permutation the next draw did not ask for again).
				std::erase_if(disk_reads, [](const auto& read) {
					return read.second->done.load(std::memory_order_acquire);
				});
			}
			auto job = std::make_shared<DiskRead>();
			job->key.assign(key.begin(), key.end());
			disk_reads.emplace(std::move(name), job);
			disk_readers->Submit(std::move(job));
			return DiskReadState::Reading;
		}
		if (!it->second->done.load(std::memory_order_acquire)) {
			return DiskReadState::Reading;
		}
		found   = it->second->found;
		payload = std::move(it->second->payload);
		disk_reads.erase(it);
		return DiskReadState::Ready;
	}

	// A stage's program entry as Get would find it, or null (not translated yet, skipped).
	template <typename InputInfo>
	SourceEntry* FindForPrep(const ShaderParams& params, const InputInfo& input_info,
	                         ShaderType stage) {
		if (SkipShaderRequested(params.hash)) {
			return nullptr;
		}
		prep_key.stage           = stage;
		prep_key.hash            = params.hash;
		prep_key.user_data_count = params.user_data_count;
		prep_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, prep_key.static_state);
		prep_key.function_code.clear();
		const auto entry = programs.find(prep_key);
		return entry != programs.end() ? &entry->second : nullptr;
	}

	// The SRT refresh of Get with the probe reader (ReadShaderGuestMemoryProbe).
	static bool WalkProbe(SourceEntry& entry, const ShaderParams& params) {
		GuestPageReadCache               cache;
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = std::span(params.user_data).first(params.user_data_count),
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryProbe,
		    .userdata                   = &cache,
		    .read_specialization_memory = ReadShaderGuestMemoryProbe,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		runtime.map_clean_page = +[](void* userdata, uint64_t page) {
			return static_cast<GuestPageReadCache*>(userdata)->Map(page);
		};
		runtime.page_userdata = &cache;
		runtime.page_shift    = 12;
		std::lock_guard walk_lock(entry.walk_mutex);
		return ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime,
		                                                  entry.resources, entry.specialization);
	}

	// The parallel window of a draw (StagePrepWorker): the helper refreshes the vertex stage while
	// this thread refreshes the pixel stage. Both entries must exist already.
	void PrepareStagesInParallel(const ShaderParams& ps_params, const ShaderPixelInputInfo& ps_info,
	                             const ShaderParams&          vs_params,
	                             const ShaderVertexInputInfo& vs_info) {
		prepared = {};
		if (stage_prep == nullptr) {
			return;
		}
		auto* ps = FindForPrep(ps_params, ps_info, ShaderType::Pixel);
		if (ps == nullptr) {
			return;
		}
		auto* vs = FindForPrep(vs_params, vs_info, vs_info.logical_stage);
		if (vs == nullptr || vs == ps) {
			return;
		}
		// The walks below write both entries' refreshes, even when they fail: neither holds a
		// prepared result or the specialization a hash was made of any more.
		for (auto* entry: {ps, vs}) {
			entry->prepared_serial           = 0;
			entry->specialization_hash_valid = false;
		}
		struct Context {
			SourceEntry*        entry;
			const ShaderParams* params;
			bool                ok;
		} job {vs, &vs_params, false};
		KYTY_PROFILER_BLOCK("StagePrep::Window");
		if (!stage_prep->Post(
		        [](void* context) {
			        auto& job = *static_cast<Context*>(context);
			        job.ok    = WalkProbe(*job.entry, *job.params);
		        },
		        &job)) {
			prep_declined++;
			return;
		}
		const bool ps_ok = WalkProbe(*ps, ps_params);
		const bool vs_ok = stage_prep->Join() && job.ok;
		prep_windows++;
		prep_ps_ok += ps_ok ? 1u : 0u;
		prep_vs_ok += vs_ok ? 1u : 0u;
		prepared = {ps_ok ? ps : nullptr, vs_ok ? vs : nullptr};
		if ((prep_windows & (prep_windows - 1u)) == 0u && prep_windows >= 1024u) {
			LOGF("Stage preparation: %" PRIu64 " parallel windows, pixel walks served %" PRIu64
			     ", vertex walks served %" PRIu64 ", %" PRIu64 " declined (helper asleep or "
			     "busy), %" PRIu64 " verify mismatches\n",
			     prep_windows, prep_ps_ok, prep_vs_ok, prep_declined, prep_mismatches);
		}
	}

	bool TakePrepared(const SourceEntry* entry) {
		for (auto& slot: prepared) {
			if (slot == entry && entry != nullptr) {
				slot = nullptr;
				return true;
			}
		}
		return false;
	}

	// KYTY_STAGE_PREP_VERIFY=1: the exact walk again after a parallel one, compared.
	void VerifyPrepared(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                    ShaderType stage, uint64_t hash) {
		const auto resources      = entry.resources;
		const auto specialization = entry.specialization;
		bool       ok             = false;
		{
			std::lock_guard walk_lock(entry.walk_mutex);
			ok = ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime,
			                                                entry.resources, entry.specialization);
		}
		if (!ok) {
			return;
		}
		const auto& fresh = entry.resources;
		if (resources.flattened_srt == fresh.flattened_srt && resources.buffers == fresh.buffers &&
		    resources.images == fresh.images && resources.samplers == fresh.samplers &&
		    specialization == entry.specialization) {
			return;
		}
		if (prep_mismatches++ < 32) {
			LOGF("Stage preparation: verify mismatch %s 0x%016" PRIx64 " (flat %d, buffers %d, "
			     "images %d, samplers %d, specialization %d)\n",
			     StageLabel(stage), hash,
			     static_cast<int>(resources.flattened_srt != fresh.flattened_srt),
			     static_cast<int>(resources.buffers != fresh.buffers),
			     static_cast<int>(resources.images != fresh.images),
			     static_cast<int>(resources.samplers != fresh.samplers),
			     static_cast<int>(!(specialization == entry.specialization)));
		}
	}

	// Draw prep (KYTY_DRAW_PREP, drawPrep.h). Scanner thread: the entry Get would find for these
	// inputs, or null (not translated yet, skipped).
	template <typename InputInfo>
	SourceEntry* FindForScan(const ShaderParams& params, const InputInfo& input_info,
	                         ShaderType stage, ProgramKey& key) {
		if (SkipShaderRequested(params.hash)) {
			return nullptr;
		}
		key.stage           = stage;
		key.hash            = params.hash;
		key.user_data_count = params.user_data_count;
		key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, key.static_state);
		key.function_code.clear();
		std::shared_lock programs_lock(programs_mutex);
		const auto       entry = programs.find(key);
		return entry != programs.end() ? &entry->second : nullptr;
	}

	// The probe reader (bytes no GPU work has written, nothing else), recording what it read.
	struct RecordingReader {
		GuestPageReadCache  cache;
		DrawPrep::ReadSet*  reads  = nullptr;
		bool                failed = false;
	};
	// A read the probe refuses fails the preparation even where the walk tolerates it (a guarded
	// read through a null base reads zero): the GPU thread's walk would read those bytes, and the
	// certificate cannot cover a read it did not record.
	static bool ReadRecorded(void* userdata, uint64_t address, std::span<uint32_t> values) {
		auto& reader = *static_cast<RecordingReader*>(userdata);
		if (!ReadShaderGuestMemoryProbe(&reader.cache, address, values)) {
			reader.failed = true;
			return false;
		}
		reader.reads->Add(address, values.data(), static_cast<uint32_t>(values.size_bytes()));
		return true;
	}

	// The draw-prep scanner's own copy of an entry's resource plan (round 71), or null to walk the
	// entry's plan under its lock as before (KYTY_BISECT's scanplan:shared, or a plan that cannot
	// be encoded). A plan's memo and replay state are mutable, so the scanner and the GPU thread
	// took turns on its walk lock: once the GPU thread was as fast as the scanner (round 69), it
	// waited for the scanner's walk of the very program it needed, ~17 % of its time at Tokyo in
	// round 70 (Mtx_lock in ProgramCache::Get), and then walked it again itself. The copy is the
	// plan decoded from its disk cache encoding (which round-trips exactly before anything is
	// stored): the same values, with a memo of its own. One per entry and scanner thread.
	ShaderRecompiler::IR::ResourcePlan* ScannerPlan(SourceEntry& entry) {
		if (Bisect::ScanPlanShared()) {
			return nullptr;
		}
		thread_local std::unordered_map<const SourceEntry*,
		                                std::unique_ptr<ShaderRecompiler::IR::ResourcePlan>>
		    copies;
		if (const auto found = copies.find(&entry); found != copies.end()) {
			return found->second.get();
		}
		// A bound on the memory the copies take (a plan is tens of KB), over all the scanner's
		// threads (round 73: its workers have copies of their own): past it, new programs are
		// walked under the lock as before.
		constexpr size_t           MaxCopies = 12288;
		static std::atomic<size_t> total {0};
		if (total.load(std::memory_order_relaxed) >= MaxCopies) {
			return nullptr;
		}
		total.fetch_add(1, std::memory_order_relaxed);
		std::vector<uint8_t> bytes;
		bool                 encoded = false;
		{
			std::lock_guard walk_lock(entry.walk_mutex);
			encoded =
			    ShaderDiskCache::EncodeSourceRecord(false, false, &entry.resource_plan, bytes);
		}
		std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan;
		// As the disk cache stores nothing that does not decode to the same encoding again.
		if (encoded && ShaderDiskCache::SourceRecordRoundTrips(bytes)) {
			ShaderDiskCache::SourceRecord record;
			if (ShaderDiskCache::DecodeSourceRecord(bytes, record) && !record.unsupported) {
				plan = std::make_unique<ShaderRecompiler::IR::ResourcePlan>(std::move(record.plan));
			}
		}
		static std::atomic<uint64_t> made {0};
		static std::atomic<uint64_t> shared {0};
		static std::atomic<uint64_t> encoded_bytes {0};
		if (plan) {
			encoded_bytes += bytes.size();
		}
		const auto count = plan ? ++made : ++shared;
		if ((count & (count - 1u)) == 0u && count >= 64u) {
			LOGF("DrawPrep: the scanner walks %" PRIu64 " plans of its own (%" PRIu64
			     " KB encoded), %" PRIu64 " shared with the GPU thread (not encodable)\n",
			     made.load(), encoded_bytes.load() >> 10u, shared.load());
		}
		auto* result = plan.get();
		copies.emplace(&entry, std::move(plan));
		return result;
	}

	// Scanner thread: the SRT refresh Get would run for these inputs, into `result`.
	template <typename InputInfo>
	bool ScanStage(const ShaderParams& params, const InputInfo& input_info, ShaderType stage,
	               DrawPrep::StageResult& result) {
		result.ok = false;
		thread_local ProgramKey key;
		auto*                   entry = FindForScan(params, input_info, stage, key);
		if (entry == nullptr) {
			DrawPrep::Add(DrawPrep::Counter::SkipNoEntry);
			return false;
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		result.stage       = static_cast<uint32_t>(stage);
		result.entry       = entry;
		result.shader_base = params.Base();
		result.user_data.assign(user_data.begin(), user_data.end());
		result.reads.Clear();
		RecordingReader                  reader {.reads = &result.reads};
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadRecorded,
		    .userdata                   = &reader,
		    .read_specialization_memory = ReadRecorded,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		// No map_clean_page: every read goes through the recorder.
		result.workgroup_counts = {};
		result.workgroup_size   = {};
		result.gpu_fills        = false;
		result.resources.gpu_fills.clear();
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// As Get: the dispatch size, and GPU-written slots left to the GPU.
			for (uint32_t axis = 0; axis < 3u; axis++) {
				result.workgroup_counts[axis] = input_info.workgroup_counts[axis];
				result.workgroup_size[axis]   = input_info.threads_num[axis];
			}
			runtime.workgroup_counts = result.workgroup_counts;
			runtime.workgroup_size   = result.workgroup_size;
			if (SrtGpuFillEnabled()) {
				result.gpu_fills    = true;
				runtime.gpu_written = SrtBytesGpuWritten;
				runtime.gpu_fills   = &result.resources.gpu_fills;
			}
		}
		bool ok = false;
		if (auto* own = ScannerPlan(*entry); own != nullptr) {
			ok = ShaderRecompiler::IR::MaterializeResources(*own, runtime, result.resources,
			                                                result.specialization);
		} else {
			std::lock_guard walk_lock(entry->walk_mutex);
			ok = ShaderRecompiler::IR::MaterializeResources(entry->resource_plan, runtime,
			                                                result.resources, result.specialization);
		}
		if (!ok || reader.failed) {
			DrawPrep::Add(DrawPrep::Counter::FailWalk);
			return false;
		}
		result.specialization_hash = SpecializationHash(result.specialization);
		static std::atomic<uint64_t> serials {0};
		result.ok       = true;
		result.consumed = false;
		result.serial   = ++serials;
		DrawPrep::Add(DrawPrep::Counter::Prepared);
		return true;
	}

	// The first field in which two refreshes differ, or null.
	static const char* DiffRefresh(const ShaderRecompiler::IR::ResourceSnapshot&       a,
	                               const ShaderRecompiler::IR::ResourceSpecialization& a_spec,
	                               const ShaderRecompiler::IR::ResourceSnapshot&       b,
	                               const ShaderRecompiler::IR::ResourceSpecialization& b_spec) {
		auto sorted = [](std::vector<std::pair<uint64_t, uint64_t>> reads) {
			std::ranges::sort(reads);
			reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
			return reads;
		};
		if (a.flattened_srt != b.flattened_srt) return "flattened_srt";
		if (a.buffers != b.buffers) return "buffers";
		if (a.images != b.images) return "images";
		if (a.samplers != b.samplers) return "samplers";
		if (a.user_data != b.user_data) return "user_data";
		if (a.buffer_write_extents != b.buffer_write_extents) return "buffer_write_extents";
		if (a.bindless_heaps != b.bindless_heaps) return "bindless_heaps";
		if (a.bindless_sampler_heaps != b.bindless_sampler_heaps) return "bindless_sampler_heaps";
		if (!(a.uniform_fill == b.uniform_fill)) return "uniform_fill";
		if (a.gpu_fills != b.gpu_fills) return "gpu_fills";
		if (sorted(a.specialization_reads) != sorted(b.specialization_reads)) {
			return "specialization_reads";
		}
		if (!(a_spec == b_spec)) return "specialization";
		return nullptr;
	}

	// GPU thread: the scanner's refresh of this stage for the current packet, when it was made
	// from the same program, user data and shader base and its bytes are unchanged and clean.
	bool TakeScanned(SourceEntry& entry, ShaderType stage,
	                 const ShaderRecompiler::IR::SrtRuntime& runtime, uint64_t hash) {
		auto* result = DrawPrep::Take(static_cast<uint32_t>(stage));
		if (result == nullptr) {
			return false;
		}
		if (result->entry != &entry) {
			DrawPrep::Add(DrawPrep::Counter::MissEntry);
			return false;
		}
		// Used already by an earlier draw of this packet: the entry must still hold its refresh.
		const bool again = result->consumed;
		if (again && entry.prepared_serial != result->serial) {
			DrawPrep::Add(DrawPrep::Counter::MissNone);
			DrawPrep::Add(DrawPrep::Counter::MissAgain);
			return false;
		}
		if (result->shader_base != runtime.shader_base ||
		    !std::ranges::equal(result->user_data, runtime.user_data) ||
		    result->gpu_fills != (runtime.gpu_written != nullptr)) {
			DrawPrep::Add(DrawPrep::Counter::MissInputs);
			return false;
		}
		if (stage == ShaderType::Compute &&
		    (!std::ranges::equal(result->workgroup_counts, runtime.workgroup_counts) ||
		     result->workgroup_size != runtime.workgroup_size)) {
			DrawPrep::Add(DrawPrep::Counter::MissInputs);
			return false;
		}
		if (!result->reads.ValidateFast()) {
			DrawPrep::Add(DrawPrep::Counter::MissReads);
			return false;
		}
		// A slot left to the GPU must still be GPU-written, or the walk here would read it.
		for (const auto& fill: (again ? entry.resources : result->resources).gpu_fills) {
			if (!SrtBytesGpuWritten(fill.address, sizeof(uint32_t))) {
				DrawPrep::Add(DrawPrep::Counter::MissReads);
				return false;
			}
		}
		result->consumed = true;
		if (table_capture && table_capture_count < table_capture_reads.size()) {
			table_capture_reads[table_capture_count++] = &result->reads;
		}
		// The refresh as it will be used: the result's before the swap, the entry's when again.
		const auto& used_resources      = again ? entry.resources : result->resources;
		const auto& used_specialization = again ? entry.specialization : result->specialization;
		if (DrawPrep::VerifyMode() != 0) {
			ShaderRecompiler::IR::ResourceSnapshot       fresh;
			ShaderRecompiler::IR::ResourceSpecialization fresh_specialization;
			bool                                         ok = false;
			auto                                         fresh_runtime = runtime;
			if (fresh_runtime.gpu_fills != nullptr) {
				fresh_runtime.gpu_fills = &fresh.gpu_fills;
			}
			{
				std::lock_guard walk_lock(entry.walk_mutex);
				ok = ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, fresh_runtime,
				                                                fresh, fresh_specialization);
			}
			DrawPrep::Add(DrawPrep::Counter::VerifyChecks);
			const char* field = ok ? DiffRefresh(used_resources, used_specialization, fresh,
			                                     fresh_specialization)
			                       : "walk failed";
			if (field != nullptr) {
				DrawPrep::Add(DrawPrep::Counter::VerifyMismatches);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					std::string detail;
					if (std::strcmp(field, "specialization_reads") == 0) {
						const auto& a = used_resources.specialization_reads;
						const auto& b = fresh.specialization_reads;
						detail = fmt::format(" (prepared {} reads, walk {}:", a.size(), b.size());
						for (const auto& read: a) {
							if (std::ranges::find(b, read) == b.end()) {
								detail += fmt::format(" +{:x}/{}", read.first, read.second);
							}
						}
						for (const auto& read: b) {
							if (std::ranges::find(a, read) == a.end()) {
								detail += fmt::format(" -{:x}/{}", read.first, read.second);
							}
						}
						detail += ")";
					}
					LOGF("DrawPrep verify: %s 0x%016" PRIx64 " differs from the walk: %s%s\n",
					     StageLabel(stage), hash, field, detail.c_str());
				}
				if (DrawPrep::VerifyMode() == 2) {
					EXIT("DrawPrep verify: %s 0x%016" PRIx64 " differs from the walk: %s\n",
					     StageLabel(stage), hash, field);
				}
			}
		}
		if (again) {
			DrawPrep::Add(DrawPrep::Counter::Reused);
			return true;
		}
		// Swapped, not moved: the entry's old buffers go back to the scanner with the result.
		std::swap(entry.resources, result->resources);
		std::swap(entry.specialization, result->specialization);
		entry.specialization_hash       = result->specialization_hash;
		entry.specialization_hash_valid = result->specialization_hash != 0;
		entry.prepared_serial = result->serial;
		DrawPrep::Add(DrawPrep::Counter::Used);
		return true;
	}

	void SetDeferTranslation(bool value) { defer_translation = value; }
};

// KYTY_TABLE (tableMode.h): GetGraphicsPrograms' table. An entry holds a draw's registers (its key),
// what GetGraphicsPrograms made of them, the stages' resource snapshots (owned here: the program
// cache's source entry is refreshed by every later draw of the shader) and the read sets their
// walks recorded (with the KYTY_MT_PROG seal). GPU thread only.
struct PipelineCache::ProgramTable {
	static constexpr uint32_t Slots     = 16384;
	static constexpr uint32_t SeenSlots = 65536;

	struct Entry {
		uint64_t                                       hash           = 0;
		uint64_t                                       program_gen    = 0;
		uint64_t                                       shader_map_gen = 0;
		HW::VertexShaderInfo                           vertex_regs;
		HW::PixelShaderInfo                            pixel_regs;
		HW::Context                                    context;
		HW::UserConfig                                 user_config;
		std::array<Prospero::ColorComponentMapping, 8> export_mapping {};
		bool                                           pixel_active = false;
		GraphicsPrograms                               programs;
		uint32_t                                       vertex_count = 0;
		std::array<ShaderVertexInputInfo, 3>           vertex_info {};
		ShaderPixelInputInfo                           pixel_info {};
		// Per stage, in Get's call order (pixel first when active, then the vertex stages): the
		// source entry, the specialization hash of the permutation found, and what the program
		// key took from describe (code, user data count, static input state).
		uint32_t                                         stage_count = 0;
		std::array<ProgramCache::SourceEntry*, 4>        sources {};
		std::array<uint64_t, 4>                          specialization {};
		std::array<const uint32_t*, 4>                   code {};
		std::array<uint64_t, 4>                          code_hash {};
		std::array<uint32_t, 4>                          user_data_count {};
		std::array<std::vector<uint32_t>, 4>             static_key;
	};
	// Stored the second time its key comes (SeenSlots).
	std::vector<uint32_t> static_key_scratch;

	std::vector<std::unique_ptr<Entry>> slots {Slots};
	std::vector<uint64_t>               seen = std::vector<uint64_t>(SeenSlots, 0);

	static uint64_t Hash(const HW::VertexShaderInfo& vertex_regs,
	                     const HW::PixelShaderInfo& pixel_regs, const HW::Context& context,
	                     const HW::UserConfig&                               user_config,
	                     std::span<const Prospero::ColorComponentMapping, 8> export_mapping,
	                     bool                                                pixel_active) {
		uint64_t h = XXH3_64bits(&context, sizeof(context));
		h          = XXH3_64bits_withSeed(&vertex_regs, sizeof(vertex_regs), h);
		h          = XXH3_64bits_withSeed(&pixel_regs, sizeof(pixel_regs), h);
		h          = XXH3_64bits_withSeed(&user_config, sizeof(user_config), h);
		h          = XXH3_64bits_withSeed(export_mapping.data(), export_mapping.size_bytes(), h);
		return h ^ (pixel_active ? 0x9e3779b97f4a7c15ull : 0u) ^ 1u; // never 0
	}

	// The key's registers: the user SGPR values left out (their pointers into per-frame memory
	// differ every frame, so no key came twice: "stored 0"); a hit takes this draw's user data
	// through describe and the scanner's refresh.
	static void KeyRegs(const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
	                    HW::VertexShaderInfo& vertex_key, HW::PixelShaderInfo& pixel_key) {
		std::memcpy(&vertex_key, &vertex_regs, sizeof(vertex_key));
		std::memcpy(&pixel_key, &pixel_regs, sizeof(pixel_key));
		std::memset(vertex_key.hs_user_sgpr.value, 0, sizeof(vertex_key.hs_user_sgpr.value));
		std::memset(vertex_key.gs_user_sgpr.value, 0, sizeof(vertex_key.gs_user_sgpr.value));
		std::memset(pixel_key.ps_user_sgpr.value, 0, sizeof(pixel_key.ps_user_sgpr.value));
	}

	[[nodiscard]] static bool SameKey(const Entry& e, const HW::VertexShaderInfo& vertex_regs,
	                                  const HW::PixelShaderInfo& pixel_regs,
	                                  const HW::Context& context, const HW::UserConfig& user_config,
	                                  std::span<const Prospero::ColorComponentMapping, 8> mapping,
	                                  bool pixel_active) {
		return e.pixel_active == pixel_active &&
		       std::memcmp(&e.context, &context, sizeof(context)) == 0 &&
		       std::memcmp(&e.vertex_regs, &vertex_regs, sizeof(vertex_regs)) == 0 &&
		       std::memcmp(&e.pixel_regs, &pixel_regs, sizeof(pixel_regs)) == 0 &&
		       std::memcmp(&e.user_config, &user_config, sizeof(user_config)) == 0 &&
		       std::memcmp(e.export_mapping.data(), mapping.data(), mapping.size_bytes()) == 0;
	}
};

static_assert(std::is_trivially_copyable_v<HW::Context> &&
              std::is_trivially_copyable_v<HW::VertexShaderInfo> &&
              std::is_trivially_copyable_v<HW::PixelShaderInfo> &&
              std::is_trivially_copyable_v<HW::UserConfig> &&
              std::is_trivially_copyable_v<Prospero::ColorComponentMapping>);

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics),
      m_program_cache(std::make_unique<ProgramCache>(
          graphics.device, graphics.shader_device_clock_enabled,
          graphics.bindless_enabled && Config::BindlessImagesEnabled(),
          ShaderDiskCache::SessionInfo {
              .vendor_id           = graphics.GetPhysicalDeviceProperties().vendorID,
              .device_id           = graphics.GetPhysicalDeviceProperties().deviceID,
              .bindless_images     = graphics.bindless_enabled && Config::BindlessImagesEnabled(),
              .float_image_atomics = Config::FloatImageAtomicsEnabled()},
          ShaderDiskCacheDirectory())) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// Round 118: KYTY_PARTITION_IDLE_MS=<ms> (0, the default: off): a partitioned shader runs only
	// the partitions of the materials its dispatches reported within that time
	// (ShaderCallPartitionIdle).
	if (const char* idle = std::getenv("KYTY_PARTITION_IDLE_MS");
	    idle != nullptr && *idle != '\0') {
		const auto milliseconds = static_cast<uint32_t>(
		    std::min<unsigned long>(std::strtoul(idle, nullptr, 10), 600000ul));
		ShaderRecompiler::Decoder::SetShaderCallPartitionIdle(milliseconds);
		LOGF("Partitioned dispatches: only the materials reported in the last %u ms run "
		     "(KYTY_PARTITION_IDLE_MS)\n",
		     milliseconds);
	}
	m_program_cache->pixel_rate_variants =
	    graphics.mixed_attachment_samples == GraphicContext::MixedAttachmentSamples::Nv &&
	    !EqaaPerSampleShading();
	// Round 159: the copies of fragment shaders that apply PA_SC_AA_MASK (spirvCoverageMask.h).
	m_coverage = std::make_unique<CoverageVariants>(
	    graphics.device, m_program_cache->disk != nullptr ? m_program_cache->disk->Directory()
	                                                      : std::filesystem::path {});
	m_program_cache->coverage = m_coverage.get();
	LOGF("AA masks: draws whose PA_SC_AA_MASK differs by pixel run a copy of their pixel shader "
	     "that applies it%s (round 159; KYTY_AA_MASK=0 or KYTY_BISECT's aamask:off leave it out)\n",
	     !AaMaskOn() ? " - off (KYTY_AA_MASK=0)"
	     : m_program_cache->disk == nullptr
	         ? " - only for draws without one: the shader disk cache, which the copies are made "
	           "from, is off"
	         : "");
	InitializeDriverCache();
	{
		// Guest pipeline seeds (pipelineSeeds.h): beside the shader disk cache, for any GPU.
		PipelineSeeds::Open(ShaderDiskCacheDirectory(), SeedFormat());
	}
	if (g_async_pipelines) {
		// KYTY_PIPELINE_THREADS=<n>; by default half the processors, 2 to 16, at below normal
		// priority (a third, 2 to 10, before; 4 at most before round 70). A race brings hundreds of
		// new pipelines at once (other cars, lap 2), each pixel shader of 150,000-230,000 SPIR-V
		// words taking the driver about a second, and their draws are skipped until they are done:
		// with 4 threads the track and the cars vanished for ten seconds while the process used 4
		// of 32 processors.
		const char* value = std::getenv("KYTY_PIPELINE_THREADS");
		const auto  threads =
            value != nullptr && *value != '\0'
		         ? std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u,
		                                32u)
		         : std::clamp(std::thread::hardware_concurrency() / 2u, 2u, 16u);
		m_probe = g_pipeline_probe && graphics.pipeline_cache_control_enabled;
		// Round 130: KYTY_PIPELINE_LOOKUP_THREADS=<n> threads more only look new pipelines up in
		// the driver's caches (by default an eighth of the processors, 2 to 4; 0: the GPU
		// thread looks them up, as before). The GPU thread took 8 ms a lookup in a race, the
		// first lap's new pipelines 2-4 s of each 10 s: the first lap of each session was the
		// slowest and froze. Only a draw into a target the game draws every frame
		// (RedrawnTargetDraw) is left out until its lookup is done; another is looked up on the
		// GPU thread still, as a picture drawn once would miss it.
		const char*    lookup_value = std::getenv("KYTY_PIPELINE_LOOKUP_THREADS");
		const uint32_t lookup_threads =
		    !m_probe ? 0u
		    : lookup_value != nullptr && *lookup_value != '\0'
		        ? std::min<uint32_t>(static_cast<uint32_t>(std::strtoul(lookup_value, nullptr, 10)),
		                             16u)
		        : std::clamp(std::thread::hardware_concurrency() / 8u, 2u, 4u);
		m_compiler                   = std::make_unique<PipelineCompiler>(threads, lookup_threads);
		const uint32_t frame_wait_ms = g_pipeline_frame_wait_ms.value_or(m_probe ? 2u : 6u);
		PipelineCacheLog("Vulkan pipelines: compiled on {} worker threads (wait {} ms for one, {} "
		                 "ms a frame); the driver's caches asked first: {}{}",
		                 threads, g_pipeline_wait.count(), frame_wait_ms,
		                 m_probe                                    ? "yes"
		                 : !graphics.pipeline_cache_control_enabled ? "no (not supported)"
		                                                            : "no (KYTY_PIPELINE_PROBE=0)",
		                 !m_probe ? ""
		                 : lookup_threads != 0
		                     ? fmt::format(", on {} lookup threads for draws into targets drawn "
		                                   "every frame (round 130)",
		                                   lookup_threads)
		                     : std::string(", on the GPU thread (KYTY_PIPELINE_LOOKUP_THREADS=0)"));
	}
}

void PipelineCache::WaitForPipeline(const PendingGraphicsPipeline& job) {
	// The budget is renewed each frame, and every 50 ms for a game whose frames the GPU thread
	// does not see end (no suspend points).
	const auto frame = FrameGraph::GpuThreadFrames();
	const auto now   = std::chrono::steady_clock::now();
	if (frame != m_wait_frame || now - m_wait_renewed >= std::chrono::milliseconds(50)) {
		m_wait_frame   = frame;
		m_wait_renewed = now;
		m_wait_left =
		    std::chrono::milliseconds(g_pipeline_frame_wait_ms.value_or(m_probe ? 2u : 6u));
	}
	const auto budget = std::min<std::chrono::microseconds>(g_pipeline_wait, m_wait_left);
	if (budget.count() <= 0) {
		return;
	}
	{
		const GpuPhases::Scope wait_phase(GpuPhases::Phase::PipelineWait);
		(void)m_compiler->Wait(job, budget);
	}
	const auto spent = std::chrono::duration_cast<std::chrono::microseconds>(
	    std::chrono::steady_clock::now() - now);
	m_wait_left -= std::min(spent, m_wait_left);
}

PipelineCache::~PipelineCache() {
	Save();
	(void)FinishBackgroundCompute(std::chrono::hours(24));
	for (const auto cache: m_driver_cache_parked) {
		m_graphics.device.destroyPipelineCache(cache, nullptr);
	}
	m_driver_cache_parked.clear();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			if (pipeline->pending && pipeline->pending->done.load(std::memory_order_acquire)) {
				m_graphics.device.destroyPipeline(pipeline->pending->pipeline, nullptr);
			}
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	for (const auto quick: m_quick_retired) {
		m_graphics.device.destroyPipeline(quick, nullptr);
	}
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	destroy(m_compute_waiting);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
	for (const auto cache: m_driver_older) {
		m_graphics.device.destroyPipelineCache(cache, nullptr);
	}
	m_driver_older.clear();
}

// Round 92: the driver cache is kept in up to KYTY_PIPELINE_CACHE_FILES files (default 3). A
// file of DriverCacheFullBytes or more takes no new pipelines: it is loaded read only, and a
// pipeline it holds is created from it (the driver is asked to look it up alone,
// pipelineCreationCacheControl). New pipelines go to the first file that is not full, the only one
// saved, and only when it gained some. Before, one file took everything; once it reached the
// 3.75 GiB a file may hold (after weeks of races) no new pipeline was kept, and every run compiled
// them all again, behind black screens and skipped draws.
constexpr uint64_t DriverCacheFullBytes = uint64_t {2048} << 20u;

static uint32_t DriverCacheFiles(const GraphicContext& graphics) {
	if (!graphics.pipeline_cache_control_enabled) {
		return 1; // no lookups alone: one file, as before
	}
	const char* value = std::getenv("KYTY_PIPELINE_CACHE_FILES");
	return value != nullptr && *value != '\0'
	           ? std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u,
	                                  8u)
	           : 3u;
}

static std::filesystem::path DriverCacheFilePath(const std::string& title_id, uint32_t index) {
	return std::filesystem::path("_PipelineCache") /
	       (index == 0 ? title_id + ".bin" : fmt::format("{}.{}.bin", title_id, index));
}

// The marker of a file whose save was refused (it would have passed DriverCacheMaxBytes): the
// next run takes it as full, and new pipelines go to the next file.
static std::filesystem::path DriverCacheFullMarker(const std::filesystem::path& file_path) {
	auto marker = file_path;
	marker += ".full";
	return marker;
}

enum class DriverCacheFile : uint8_t {
	Missing,
	Unreadable,   // it exists but could not be read: left alone, and not written
	Incompatible, // another driver, device or a damaged file: written anew
	Valid,
};

// A cache file's driver data (the reason it is not Valid is logged).
static DriverCacheFile ReadDriverCacheFile(const std::filesystem::path& file_path,
                                           const std::string&           signature,
                                           std::vector<uint8_t>&        data) {
	data.clear();
	if (!Common::File::IsFileExisting(file_path)) {
		return DriverCacheFile::Missing;
	}
	const auto path = Common::PathToString(file_path);
	PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	Common::File file(file_path, Common::File::Mode::Read);
	if (file.IsInvalid()) {
		PipelineCacheLog("Vulkan pipeline cache: {} could not be opened; not used, not written",
		                 path);
		return DriverCacheFile::Unreadable;
	}
	const auto file_size = file.Size();
	if (file_size < signature.size() + sizeof(uint64_t) || file_size > DriverCacheMaxBytes) {
		file.Close();
		PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size: {} MB, the "
		                 "limit is {} MB)",
		                 path, file_size >> 20u, DriverCacheMaxBytes >> 20u);
		return DriverCacheFile::Incompatible;
	}
	std::string cached_signature(signature.size(), '\0');
	uint64_t    payload_hash   = 0;
	uint32_t    signature_read = 0;
	uint32_t    hash_read      = 0;
	file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
	          &signature_read);
	file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
	if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash)) {
		file.Close();
		PipelineCacheLog("Vulkan pipeline cache: {} could not be read; not used, not written",
		                 path);
		return DriverCacheFile::Unreadable;
	}
	if (cached_signature != signature) {
		file.Close();
		PipelineCacheLog(
		    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)", path);
		return DriverCacheFile::Incompatible;
	}
	data.resize(file_size - signature.size() - sizeof(payload_hash));
	const auto payload_read = ReadInChunks(file, data.data(), data.size());
	file.Close();
	if (payload_read != data.size()) {
		data.clear();
		PipelineCacheLog("Vulkan pipeline cache: {} could not be read; not used, not written",
		                 path);
		return DriverCacheFile::Unreadable;
	}
	if (XXH3_64bits(data.data(), data.size()) != payload_hash) {
		data.clear();
		PipelineCacheLog(
		    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)", path);
		return DriverCacheFile::Incompatible;
	}
	return DriverCacheFile::Valid;
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

	const auto signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto files     = DriverCacheFiles(m_graphics);
	const auto create    = [this](const std::vector<uint8_t>& data, vk::PipelineCache* cache) {
        vk::PipelineCacheCreateInfo info {};
        info.initialDataSize = data.size();
        info.pInitialData    = data.empty() ? nullptr : data.data();
        return m_graphics.device.createPipelineCache(&info, nullptr, cache);
	};
	std::vector<uint8_t> initial_data;
	uint32_t             write_index = files;
	uint64_t             older_bytes = 0;
	for (uint32_t index = 0; index < files; index++) {
		const auto file_path = DriverCacheFilePath(title_id, index);
		const auto state     = ReadDriverCacheFile(file_path, signature, initial_data);
		if (state == DriverCacheFile::Unreadable) {
			continue;
		}
		const bool valid = state == DriverCacheFile::Valid;
		const bool full  = valid && (initial_data.size() >= DriverCacheFullBytes ||
                                    Common::File::IsFileExisting(DriverCacheFullMarker(file_path)));
		if (!full || index + 1 == files) {
			// New pipelines go here (a missing or unusable file is written anew).
			write_index         = index;
			m_driver_cache_path = file_path;
			if (!valid) {
				PipelineCacheLog("Vulkan pipeline cache: initializing {}",
				                 Common::PathToString(file_path));
			}
			break;
		}
		vk::PipelineCache older  = nullptr;
		const auto        result = create(initial_data, &older);
		if (result != vk::Result::eSuccess) {
			PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); not used",
			                 Common::PathToString(file_path), vk::to_string(result));
			continue;
		}
		m_driver_older.push_back(older);
		older_bytes += initial_data.size();
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {} ({} MB: full, read only)",
		                 initial_data.size(), Common::PathToString(file_path),
		                 initial_data.size() >> 20u);
	}
	if (write_index == files) {
		// Every file is full or unreadable: new pipelines are kept for this run only.
		initial_data.clear();
	}
	const auto path = write_index == files ? std::string("(no file: not saved)")
	                                       : Common::PathToString(m_driver_cache_path);

	auto result = create(initial_data, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		result = create(initial_data, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	m_driver_cache_loaded = initial_data.size();
	if (!initial_data.empty()) {
		PipelineCacheLog(
		    "Vulkan pipeline cache: loaded {} bytes from {} ({} MB of the {} MB a file "
		    "may hold)",
		    initial_data.size(), path, initial_data.size() >> 20u, DriverCacheMaxBytes >> 20u);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
	PipelineCacheLog("Vulkan pipeline cache: new pipelines are kept in {} ({} files at most; {} "
	                 "full files looked up first, {} MB; {} MB loaded in all)",
	                 path, files, m_driver_older.size(), older_bytes >> 20u,
	                 (older_bytes + initial_data.size()) >> 20u);
}

uint8_t PipelineCache::RecordPipelineSeed(const GraphicsPipelineKey& key, const GraphicsPrograms& programs,
                                          uint32_t vertex_stages, bool ps_active) {
	ShaderDiskCache::ByteWriter payload;
	payload.Put(PipelineSeeds::KindPipeline);
	payload.Put(static_cast<uint8_t>(vertex_stages));
	for (uint32_t i = 0; i < vertex_stages; i++) {
		const auto hash = PipelineSeeds::ProgramHash(programs.vertex[i].id);
		if (hash == 0) {
			return SeedUnrecorded; // a program without a seed (recording was off when it was made)
		}
		payload.Put(hash);
		payload.Put(programs.vertex[i].normalized_samplers);
	}
	payload.Put(ps_active);
	if (ps_active) {
		const auto hash = PipelineSeeds::ProgramHash(programs.pixel.id);
		if (hash == 0) {
			return SeedUnrecorded;
		}
		payload.Put(hash);
	}
	payload.PutBytes({reinterpret_cast<const uint8_t*>(&key.rendering), sizeof(key.rendering)});
	payload.PutBytes({reinterpret_cast<const uint8_t*>(&key.static_params), sizeof(key.static_params)});
	payload.Put(key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		payload.Put(key.vertex_input.bindings[i].stride);
		payload.Put(key.vertex_input.bindings[i].instance);
	}
	payload.Put(key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		payload.Put(key.vertex_input.attributes[i].offset);
		payload.Put(key.vertex_input.attributes[i].binding);
	}
	return PipelineSeeds::RecordPipeline(std::move(payload.bytes)) ? SeedKnown : SeedNew;
}

uint64_t PipelineCache::SeedFormat() {
	// The structures a seed copies whole: another layout is another format (the seeds of an older
	// emulator build are then set aside, and a release's seed file has to be made again).
	const uint64_t sizes[] = {2,
	                          sizeof(ShaderVertexInputInfo),
	                          sizeof(ShaderPixelInputInfo),
	                          sizeof(ShaderComputeInputInfo),
	                          sizeof(PipelineRenderingState),
	                          sizeof(PipelineStaticParameters),
	                          static_cast<uint64_t>(ShaderVertexInputInfo::RES_MAX)};
	return XXH3_64bits(sizes, sizeof(sizes));
}

// KYTY_SEED_VERTEX_DEFAULTS=0: the precompile keeps the cleared mesh fields of non-mesh vertex
// seeds (the keys before this fix).
static bool SeedVertexDefaults() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_SEED_VERTEX_DEFAULTS");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

// KYTY_PRECOMPILE_COMPUTE=0: the precompile makes no compute pipelines (their programs still go to
// the shader disk cache).
static bool PrecompileCompute() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PRECOMPILE_COMPUTE");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

void PipelineCache::PrecompileSeeds() {
	using Seed    = ProgramCache::SeedProgram;
	using Outcome = ProgramCache::SeedOutcome;
	const auto begin   = std::chrono::steady_clock::now();
	uint32_t   other   = 0;
	auto       records = PipelineSeeds::LoadAll(&other);

	std::unordered_map<uint64_t, std::unique_ptr<Seed>> programs;
	std::vector<std::span<const uint8_t>>              pipelines;
	uint32_t                                           damaged = 0;
	for (const auto& record: records) {
		if (record.payload.empty()) continue;
		if (record.payload[0] == PipelineSeeds::KindProgram) {
			auto seed = std::make_unique<Seed>();
			if (ProgramCache::ParseSeedProgram(record.payload, *seed)) {
				seed->record = record.hash;
				programs.emplace(record.hash, std::move(seed));
			} else {
				damaged++;
			}
		} else if (record.payload[0] == PipelineSeeds::KindPipeline) {
			pipelines.emplace_back(record.payload);
		}
	}
	std::vector<Seed*> program_list;
	program_list.reserve(programs.size());
	for (auto& [hash, seed]: programs) program_list.push_back(seed.get());
	const auto total = static_cast<uint32_t>(program_list.size() + pipelines.size());
	const uint32_t threads = std::max(1u, std::thread::hardware_concurrency());
	PipelineCacheLog("Pipeline seeds: precompiling {} programs and {} graphics pipelines on {} threads "
	                 "({} records unreadable, {} files of another build)",
	                 program_list.size(), pipelines.size(), threads, damaged, other);
	m_program_cache->PrepareReplay();

	const ShaderRecompiler::IR::ResourceSnapshot no_resources {};
	std::atomic<uint32_t> done {0}, compiled {0}, failed {0}, cached {0};
	std::atomic<uint32_t> compute_cached {0}, compute_compiled {0}, compute_failed {0};
	uint32_t              compute_seeds = 0;
	for (const auto* seed: program_list) {
		compute_seeds += seed->kind == 2 ? 1u : 0u;
	}
	std::atomic<bool>     stop {false};
	const bool            wave64 = m_graphics.SupportsComputeWave64();
	const auto            lds_limit =
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;

	const auto run_parallel = [&](size_t count, const auto& fn) {
		std::atomic<size_t>      next {0};
		std::atomic<uint32_t>    exited {0};
		std::vector<std::thread> workers;
		for (uint32_t t = 0; t < threads; t++) {
			workers.emplace_back([&] {
				for (;;) {
					const auto i = next.fetch_add(1);
					if (i >= count || stop.load(std::memory_order_relaxed)) break;
					fn(i);
					done++;
				}
				exited++;
			});
		}
		while (exited.load() < threads) {
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
			PipelineSeeds::WriteProgress(done, total, compiled, failed, false);
			if (PipelineSeeds::StopRequested()) {
				stop = true;
			}
		}
		for (auto& w: workers) w.join();
	};

	// 1. Programs: translated and compiled into the shader disk cache; compute pipelines made.
	run_parallel(program_list.size(), [&](size_t i) {
		auto&   seed    = *program_list[i];
		Outcome outcome = Outcome::Failed;
		switch (seed.kind) {
			case 0: {
				auto& info = seed.vertex;
				if (info.logical_stage != ShaderType::Mesh && SeedVertexDefaults()) {
					// SeedInfo cleared these for every vertex-like stage, but the game leaves a
					// non-mesh stage's at their defaults (only GetGraphicsPrograms' mesh branch sets
					// them), and the disk cache keys them (AppendInputKey): without this, every
					// VS/LS/HS/DS/ES the precompile translated was saved under a key the game never
					// asks for (gt7-20261112-040749: 12,275 source misses after a precompile).
					const ShaderMeshInputInfo defaults {};
					info.mesh.host_subgroup_size = defaults.host_subgroup_size;
					info.mesh.passes             = defaults.passes;
				}
				if (info.logical_stage == ShaderType::Mesh) {
					if (!m_graphics.mesh_shader_enabled) break;
					auto&      mesh     = info.mesh;
					const auto required = m_graphics.GraphicsSubgroupSize(vk::ShaderStageFlagBits::eMeshEXT,
					                                                      mesh.wave_size);
					mesh.host_subgroup_size = required != 0 ? required : m_graphics.subgroup_size;
					const auto& limits      = m_graphics.mesh_shader_properties;
					mesh.passes             = mesh.PassesFor(
                        std::min(limits.maxMeshWorkGroupInvocations, limits.maxMeshWorkGroupSize[0]));
					if (mesh.passes == 0) break;
				}
				outcome = m_program_cache->ReplaySeed(seed, info);
				break;
			}
			case 1: outcome = m_program_cache->ReplaySeed(seed, seed.pixel); break;
			case 2:
				seed.compute.host_subgroup_size = wave64 ? 64u : 32u;
				seed.compute.lds_storage        = seed.compute.lds_size_dwords * 4u > lds_limit;
				outcome = m_program_cache->ReplaySeed(seed, seed.compute);
				break;
			default: break;
		}
		if (outcome == Outcome::Failed) {
			failed++;
			return;
		}
		(outcome == Outcome::Compiled ? compiled : cached)++;
		m_program_cache->MakeSeedModules(seed);
		seed.ok = seed.modules.module != nullptr;
		if (seed.ok && seed.kind == 2 && PrecompileCompute()) {
			// Round seedcs: as the game makes it (GetComputePipeline): looked up in the driver's
			// caches first, compiled into the one that is saved when none holds it; counted apart.
			Pipeline pipeline;
			auto     info = seed.compute;
			info.stage    = {.program = &seed.program, .resources = &no_resources};
			bool made     = CreatePipelineInternal(m_graphics, pipeline, info, seed.modules.module,
			                                       m_driver_cache, m_driver_older, ComputeBuild::LookUp);
			if (made) {
				compute_cached++;
			} else if (pipeline.pipeline_layout != nullptr && !pipeline.build_failed) {
				const auto begin = std::chrono::steady_clock::now();
				made = CreatePipelineInternal(m_graphics, pipeline, info, seed.modules.module,
				                              m_driver_cache, {}, ComputeBuild::Compile);
				const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
				                    std::chrono::steady_clock::now() - begin)
				                    .count();
				(!made ? compute_failed
				 : ms < PipelineMadeCounts::CachedMs ? compute_cached
				                                     : compute_compiled)++;
			} else {
				compute_failed++;
			}
			if (made && pipeline.pipeline != nullptr) {
				m_graphics.device.destroyPipeline(pipeline.pipeline);
			}
			if (pipeline.pipeline_layout != nullptr) m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout);
			if (pipeline.descriptor_set_layout != nullptr)
				m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout);
		}
	});

	// 2. Graphics pipelines, from the programs' modules, into the driver cache.
	if (!stop.load()) {
		run_parallel(pipelines.size(), [&](size_t i) {
			ShaderDiskCache::ByteReader in(pipelines[i]);
			uint8_t                     tag = 0, stages = 0;
			bool                        ps_active = false;
			std::array<Seed*, 3>        vertex {};
			std::array<bool, 3>         normalized {};
			Seed*                       pixel = nullptr;
			const auto find = [&](uint64_t hash) -> Seed* {
				const auto it = programs.find(hash);
				return it != programs.end() && it->second->ok ? it->second.get() : nullptr;
			};
			bool ok = in.Get(tag) && in.Get(stages) && (stages == 1 || stages == 3);
			for (uint32_t s = 0; ok && s < stages; s++) {
				uint64_t hash = 0;
				ok = in.Get(hash) && in.Get(normalized[s]) && (vertex[s] = find(hash)) != nullptr &&
				     vertex[s]->kind == 0;
			}
			if (ok && in.Get(ps_active) && ps_active) {
				uint64_t hash = 0;
				ok = in.Get(hash) && (pixel = find(hash)) != nullptr && pixel->kind == 1;
			}
			GraphicsPipelineKey key {};
			const auto          read_raw = [&](void* target, size_t size) {
                const auto offset = pipelines[i].size() - in.Remaining();
                if (!ok || in.Remaining() < size) return false;
                std::memcpy(target, pipelines[i].data() + offset, size);
                ShaderDiskCache::ByteReader skip(pipelines[i].subspan(offset + size));
                in = skip;
                return true;
			};
			ok = ok && read_raw(&key.rendering, sizeof(key.rendering)) &&
			     read_raw(&key.static_params, sizeof(key.static_params)) &&
			     in.Get(key.vertex_input.binding_count) &&
			     key.vertex_input.binding_count <= ShaderVertexInputInfo::RES_MAX;
			for (uint32_t b = 0; ok && b < key.vertex_input.binding_count; b++) {
				ok = in.Get(key.vertex_input.bindings[b].stride) && in.Get(key.vertex_input.bindings[b].instance);
			}
			ok = ok && in.Get(key.vertex_input.attribute_count) &&
			     key.vertex_input.attribute_count <= ShaderVertexInputInfo::RES_MAX;
			for (uint32_t a = 0; ok && a < key.vertex_input.attribute_count; a++) {
				ok = in.Get(key.vertex_input.attributes[a].offset) &&
				     in.Get(key.vertex_input.attributes[a].binding);
			}
			if (!ok || !in.AtEnd()) {
				failed++;
				return;
			}
			std::array<ShaderVertexInputInfo, 3> vertex_info {};
			GraphicsPrograms                     progs;
			for (uint32_t s = 0; s < stages; s++) {
				vertex_info[s]       = vertex[s]->vertex;
				vertex_info[s].stage = {.program = &vertex[s]->program, .resources = &no_resources};
				auto& p              = progs.vertex[s];
				p.id                 = s + 1;
				p.module             = vertex[s]->modules.module;
				p.spirv_words        = static_cast<uint32_t>(vertex[s]->spirv.size());
				p.hash               = vertex[s]->shader_hash;
				if (normalized[s]) {
					if (vertex[s]->modules.normalized == nullptr) {
						failed++;
						return;
					}
					p.module              = vertex[s]->modules.normalized;
					p.normalized_samplers = true;
				}
			}
			ShaderPixelInputInfo pixel_info {};
			if (ps_active) {
				pixel_info                   = pixel->pixel;
				pixel_info.stage             = {.program = &pixel->program, .resources = &no_resources};
				progs.pixel.id                = 4;
				progs.pixel.module            = pixel->modules.module;
				progs.pixel.spirv_words       = static_cast<uint32_t>(pixel->spirv.size());
				progs.pixel.hash              = pixel->shader_hash;
				progs.pixel.pixel_rate_module = pixel->modules.pixel_rate;
			}
			Pipeline   pipeline;
			const auto build = PrepareGraphicsPipeline(
			    m_graphics, pipeline, key.rendering, key.vertex_input,
			    std::span<const ShaderVertexInputInfo>(vertex_info.data(), stages),
			    ps_active ? &pixel_info : nullptr, progs, key.static_params);
			if (CreateFromOlderFiles(*build, m_driver_older, &pipeline.pipeline)) {
				cached++;
			} else if (CreateGraphicsPipeline(*build, m_driver_cache, &pipeline.pipeline) ==
			               vk::Result::eSuccess &&
			           pipeline.pipeline != nullptr) {
				compiled++;
			} else {
				failed++;
			}
			if (pipeline.pipeline != nullptr) m_graphics.device.destroyPipeline(pipeline.pipeline);
			if (pipeline.pipeline_layout != nullptr) m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout);
			if (pipeline.descriptor_set_layout != nullptr)
				m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout);
		});
	}

	for (auto& [hash, seed]: programs) {
		for (const auto module: {seed->modules.module, seed->modules.pixel_rate, seed->modules.normalized}) {
			if (module != nullptr) m_graphics.device.destroyShaderModule(module);
		}
	}
	m_program_cache->FlushDiskForReplay();
	const auto secs =
	    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count();
	PipelineCacheLog("Pipeline seeds: precompile {} in {} s: {} made, {} already cached, {} failed, {} of {} done",
	                 stop.load() ? "stopped" : "finished", secs, compiled.load(), cached.load(), failed.load(),
	                 done.load(), total);
	PipelineCacheLog("Pipeline seeds: compute pipelines of {} compute programs: {} already in the driver's "
	                 "caches, {} compiled into it, {} failed{}",
	                 compute_seeds, compute_cached.load(), compute_compiled.load(), compute_failed.load(),
	                 PrecompileCompute() ? "" : " (KYTY_PRECOMPILE_COMPUTE=0: none made)");
	PipelineSeeds::WriteProgress(done, total, compiled, failed, true);
	PipelineSeeds::RemoveStopFile();
}

void PipelineCache::LogPipelinesMade() {
	// KYTY_PIPELINE_SUMMARY=0: no line. Not in precompile mode (it has its own lines).
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PIPELINE_SUMMARY");
		return v == nullptr || v[0] != '0';
	}();
	static std::atomic<bool> logged {false};
	if (!on || PipelineSeeds::PrecompileRequested() || logged.exchange(true)) {
		return;
	}
	auto&    counts = MadeCounts();
	uint64_t seeded = 0, unseeded = 0;
	PipelineSeeds::MissCounts(seeded, unseeded);
	PipelineCacheLog("Pipelines this run: graphics {} from the driver's caches, {} compiled in game; "
	                 "compute {} from the caches, {} compiled in game ({} in the background); shader "
	                 "source misses: {} of sources in the seeds (another key), {} not in the seeds "
	                 "(a create under {} ms counts as from the caches)",
	                 counts.graphics_cached.load(), counts.graphics_compiled.load(),
	                 counts.compute_cached.load(), counts.compute_compiled.load(), m_background_started,
	                 seeded, unseeded, PipelineMadeCounts::CachedMs);
	// Why the graphics pipelines compiled in game were not made by the precompile.
	PipelineCacheLog("Graphics pipelines compiled in game: {} in guest.seeds before this run (the "
	                 "precompile did not make them: it failed, did not run since, or the driver "
	                 "dropped them), {} new to the seeds (made by the next precompile), {} without "
	                 "a seed (a program without one, a coverage copy or KYTY_PIPELINE_SEEDS=0); "
	                 "the first KYTY_SHADER_MISS_LOG are named by \"Pipeline compiled in game\"",
	                 counts.graphics_compiled_seed[SeedKnown].load(),
	                 counts.graphics_compiled_seed[SeedNew].load(),
	                 counts.graphics_compiled_seed[SeedUnrecorded].load());
}
void PipelineCache::Save() {
	PipelineSeeds::Flush();
	LogPipelinesMade();
	if (!PipelineSeeds::PrecompileRequested()) {
		// Round rtexp: canonical expansions (ShaderCallCanonical), once per run.
		static std::atomic<bool> expansions_logged {false};
		if (!expansions_logged.exchange(true)) {
			uint64_t targets = 0, shaders = 0, growths = 0;
			ShaderRecompiler::Decoder::ShaderCallKnownTargets(targets, shaders, growths);
			PipelineCacheLog("Call expansions: {} table call targets known ({} shaders), {} "
			                 "expansions translated in game, union grew {} times ({})",
			                 targets, shaders, g_expansions_in_game.load(), growths,
			                 ShaderRecompiler::Decoder::ShaderCallCanonical()
			                     ? "canonical: union of the targets, ascending; "
			                       "KYTY_CALL_CANONICAL=0 turns it off"
			                     : "KYTY_CALL_CANONICAL=0: as before");
		}
	}
	{
		// Round 134: no save during play starts from now, and one still writing finishes first.
		std::lock_guard lock(m_save_mutex);
		m_saved = true;
		if (m_save_thread.joinable()) {
			m_save_thread.join();
		}
	}
	m_program_cache->FlushDisk();
	// No worker may use the driver caches once they are destroyed below; later pipelines are
	// compiled on the GPU thread (FinishPending).
	if (m_compiler != nullptr) {
		m_compiler->Stop();
	}
	// Round 96: a bounded wait. The emulator ends with quick_exit after this, without tearing the
	// device down, so a compile still running may go on into a driver cache that is kept (not
	// destroyed) until then; its pipeline is compiled again next run. Before, closing waited for
	// every compile, minutes for the ray tracing mode's material passes, while the window showed
	// "not responding".
	const auto still_compiling = FinishBackgroundCompute(std::chrono::seconds(20));
	if (!m_driver_older.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: {} pipelines this run came from the full files",
		                 g_older_file_hits.load(std::memory_order_relaxed));
		for (const auto cache: m_driver_older) {
			m_graphics.device.destroyPipelineCache(cache, nullptr);
		}
		m_driver_older.clear();
	}
	if (m_driver_cache == nullptr) {
		return;
	}
	if (m_driver_cache_path.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: every file is full or unreadable; the new "
		                 "pipelines of this run are not saved");
		ReleaseDriverCache(still_compiling);
		return;
	}

	size_t bytes = 0;
	switch (WriteDriverCacheFile(false, bytes)) {
		case CacheWrite::Saved:
		case CacheWrite::Unchanged:
		case CacheWrite::TooLarge: ReleaseDriverCache(still_compiling); break;
		case CacheWrite::Failed: break;
	}
}

// Save's write of the driver cache to its file, and round 134's saves during play
// (MaybeSaveDuringPlay, on a thread of their own): those leave a cache past the size a file may
// hold to the save at the exit, which marks the file full.
PipelineCache::CacheWrite PipelineCache::WriteDriverCacheFile(bool during_play, size_t& bytes) {
	bytes                     = 0;
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
		if (size == m_driver_cache_loaded) {
			// No pipeline was added: the file stays as it is (it was written at the exits before).
			if (!during_play) {
				PipelineCacheLog(
				    "Vulkan pipeline cache: {} has no new pipelines; not written again",
				    Common::PathToString(m_driver_cache_path));
			}
			return CacheWrite::Unchanged;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return CacheWrite::Failed;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (prefix.size() + payload.size() > DriverCacheMaxBytes) {
		if (during_play) {
			// The save at the exit says so and marks the file full.
			return CacheWrite::TooLarge;
		}
		// The next run would refuse a file this large and compile everything again (round 64's
		// runs, after the file passed the old 1 GiB limit). The file already there stays.
		PipelineCacheLog("Vulkan pipeline cache: {} MB is past the {} MB a file may hold; not "
		                 "saved (the file from before stays, and the next run takes it as full)",
		                 payload.size() >> 20u, DriverCacheMaxBytes >> 20u);
		Common::File marker;
		if (marker.Create(DriverCacheFullMarker(m_driver_cache_path))) {
			marker.Close();
		}
		return CacheWrite::TooLarge;
	}
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return CacheWrite::Failed;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint64_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		payload_written = WriteInChunks(file, payload.data(), payload.size());
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return CacheWrite::Failed;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}{}", payload.size(),
	                 Common::PathToString(m_driver_cache_path),
	                 during_play ? " during play (KYTY_PIPELINE_SAVE_MINUTES)" : "");
	if (const auto marker = DriverCacheFullMarker(m_driver_cache_path);
	    Common::File::IsFileExisting(marker)) {
		std::error_code error;
		std::filesystem::remove(marker, error);
	}
	// The file holds this much now: a save with nothing added since leaves it.
	m_driver_cache_loaded = payload.size();
	bytes                 = payload.size();
	return CacheWrite::Saved;
}

// Round 134: every KYTY_PIPELINE_SAVE_MINUTES (5 by default; 0: only at the exit), once 100 new
// pipelines or more were compiled since the last save, the driver cache is written to its file on
// a thread of its own. It was saved only when the emulator closed normally: a player whose
// sessions ended in crashes started each one from the same 793 MB cache and compiled 1,700 to
// 4,750 pipelines again every time (2026-10-23).
void PipelineCache::MaybeSaveDuringPlay() {
	static const uint32_t minutes = [] {
		const char* value = std::getenv("KYTY_PIPELINE_SAVE_MINUTES");
		return value != nullptr && *value != '\0'
		           ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
		           : 5u;
	}();
	constexpr uint64_t MinNewPipelines = 100;
	m_new_since_save++;
	const auto now = std::chrono::steady_clock::now();
	if (m_last_save == std::chrono::steady_clock::time_point {}) {
		m_last_save = now;
	}
	if (minutes == 0 || m_new_since_save < MinNewPipelines ||
	    now - m_last_save < std::chrono::minutes(minutes) ||
	    m_save_running.load(std::memory_order_acquire)) {
		return;
	}
	std::lock_guard lock(m_save_mutex);
	if (m_saved || m_driver_cache == nullptr || m_driver_cache_path.empty()) {
		return;
	}
	if (m_save_thread.joinable()) {
		m_save_thread.join();
	}
	const auto pipelines = m_new_since_save;
	m_new_since_save     = 0;
	m_last_save          = now;
	m_save_running.store(true, std::memory_order_release);
	m_save_thread = std::thread([this, pipelines] {
		Common::LowerThreadPriority();
		size_t     bytes  = 0;
		const auto result = WriteDriverCacheFile(true, bytes);
		if (result == CacheWrite::Saved) {
			PipelineCacheLog("Vulkan pipeline cache: {} new pipelines in that save", pipelines);
		}
		m_save_running.store(false, std::memory_order_release);
	});
}

void PipelineCache::ReleaseDriverCache(size_t still_compiling) {
	if (still_compiling == 0) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	} else {
		// A background compile still uses it (FinishBackgroundCompute, the destructor's wait).
		m_driver_cache_parked.push_back(m_driver_cache);
	}
	m_driver_cache = nullptr;
}

// Glyphs, icons and other cached layers are drawn once into small offscreen targets, and a draw
// skipped while its shaders translate or its pipeline compiles is lost for good: the Leap Attack
// prompt lost letters ("L p Att k") in every new game whose session translated shaders
// (2026-10-08, ISSUES #27), whatever the depth state of those draws. They translate and compile
// at once instead; a one-time cost per program, none once the disk caches hold it.
bool SmallColorTargetDraw(const HW::Context& context) {
	static const uint32_t limit = [] {
		const char* value = std::getenv("KYTY_SMALL_TARGET_SYNC");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 1024u;
	}();
	if (limit == 0) {
		return false;
	}
	const auto mask  = context.GetRenderTargetMask();
	bool       found = false;
	for (uint32_t slot = 0; slot < 8; slot++) {
		const auto& target = context.GetRenderTarget(slot);
		if (((mask >> (slot * 4u)) & 0x0fu) == 0 || target.base.addr == 0) {
			continue;
		}
		if (target.attrib2.width + 1u > limit || target.attrib2.height + 1u > limit) {
			return false;
		}
		found = true;
	}
	return found;
}

// Round 87: which targets the game draws into every frame. The rule above also caught the game's
// reflection cubes (512x512 and 1024x1024, six layers), which it redraws every frame with the
// scenery around the car: each new object in them translated and compiled on the GPU thread, and
// a pipeline of those takes the driver about a second, so the race stopped for a second at a
// time, several times every 10 s, all race long (2026-10-18, warm86: up to 38 s of 42). A draw
// into such a target may be skipped while its shaders translate or its pipeline compiles: it
// misses a frame or a few, not a picture drawn once. Such a target is a reflection cube (learned
// by renderDraw.cpp; drawn in two frames or more until round 146), or one drawn in at least three
// of six frames in a row at some time, when the draw tests against a depth target bound with it,
// with a test that can fail or a depth write (a 3D pass: the depth test enable alone is not
// enough, the lost glyphs above had it set). The second holds until the target goes unused for
// 32 frames; the cube's count is kept for good (round 88): the race start had none of the frames
// before (a loading screen), and round 87 compiled 46 cube pipelines on the GPU thread there,
// 44 s. KYTY_DEFER_REDRAWN=0 turns the rule off. GPU thread only.
namespace {

const bool g_defer_redrawn = [] {
	const char* value = std::getenv("KYTY_DEFER_REDRAWN");
	return value == nullptr || value[0] != '0';
}();

std::array<uint64_t, 4> g_reflection_cubes {};
uint32_t                g_reflection_cube_count = 0;

// The frames a target was drawn in: bit i, i frames before `last`; how many frames in all (up to
// a few), and whether it was ever drawn in three of six frames in a row.
struct TargetFrames {
	uint64_t last   = UINT64_MAX;
	uint32_t bits   = 0;
	uint32_t frames = 0;
	bool     steady = false;
};
std::unordered_map<uint64_t, TargetFrames> g_target_frames;
const TargetFrames                         g_no_target_frames {};

// Notes a draw into its target (its first color target, else its depth target) in this frame.
const TargetFrames& NoteTargetFrame(const HW::Context& context) {
	const auto mask = context.GetRenderTargetMask();
	uint64_t   key  = 0;
	for (uint32_t slot = 0; slot < 8 && key == 0; slot++) {
		const auto& target = context.GetRenderTarget(slot);
		if (((mask >> (slot * 4u)) & 0x0fu) != 0) {
			key = target.base.addr;
		}
	}
	if (key == 0) {
		const auto address = context.GetDepthRenderTarget().z_read_base_addr;
		key                = address != 0 ? address | (uint64_t {1} << 63u) : 0;
	}
	if (key == 0) {
		return g_no_target_frames;
	}
	if (g_target_frames.size() >= 16384 && !g_target_frames.contains(key)) {
		g_target_frames.clear();
	}
	const auto frame = FrameGraph::GpuThreadFrames();
	auto&      entry = g_target_frames[key];
	if (frame != entry.last) {
		const auto delta = frame - entry.last;
		const bool gap   = entry.last == UINT64_MAX || delta >= 32;
		entry.bits       = gap ? 0u : entry.bits << delta;
		// A target unused for 32 frames may be memory the game uses for something else now
		// (one drawn once): it must show three frames of six again. A reflection cube keeps its
		// count (a race start comes after a loading screen of many frames).
		entry.steady = entry.steady && !gap;
		entry.last   = frame;
		entry.frames = std::min(entry.frames + 1u, 16u);
	}
	entry.bits |= 1u;
	entry.steady = entry.steady || std::popcount(entry.bits & 0x7eu) >= 3;
	return entry;
}

bool ReflectionCubeDraw(const HW::Context& context) {
	const auto address = context.GetRenderTarget(0).base.addr;
	const auto known   = std::span(g_reflection_cubes.data(), g_reflection_cube_count);
	return address != 0 && (context.GetRenderTargetMask() & 0x0fu) != 0 &&
	       std::ranges::find(known, address) != known.end();
}

// A depth test against a depth target bound with the draw, one that can fail (or writes depth).
bool DepthTestedDraw(const HW::Context& context) {
	constexpr uint8_t CompareAlways = 7;
	const auto&       control       = context.GetDepthControl();
	const auto&       depth         = context.GetDepthRenderTarget();
	return control.z_enable && depth.z_info.format != Prospero::DepthFormat::kInvalid &&
	       depth.z_read_base_addr != 0 &&
	       (control.zfunc != CompareAlways || control.z_write_enable);
}

// Round 91: a 3D pass into two color targets or more, each at least 256 texels each way, with a
// depth test that can fail and a depth write: a G-buffer pass into a texture, not a UI or layer
// picture. A game's car view shown before an event drew one into 1024x512 targets the first
// frame it showed: 64 pipelines compiled on the GPU thread, 50 s of black screens in a session,
// ~5 s at a time (2026-10-19). Such a draw may be skipped while its shaders translate and its
// pipeline compiles, from its first frame: a picture drawn only once would miss what was skipped.
// A pass into one target is left out: a picture rendered once (a preview, a thumbnail) would lose
// what was skipped for good; a race start's rear-view mirror (one 1024x256 target, 4 pipelines,
// 4.5 s) still compiles them at once, the first time only now that new pipelines are kept again
// (round 92). KYTY_DEFER_GBUFFER=0 (or KYTY_DEFER_REDRAWN=0, all of these rules) turns this off;
// the F8 step gbuffer:sync too.
bool GbufferPassDraw(const HW::Context& context) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DEFER_GBUFFER");
		return value == nullptr || value[0] != '0';
	}();
	constexpr uint8_t CompareAlways = 7;
	const auto&       control       = context.GetDepthControl();
	if (!enabled || Bisect::GbufferSync() || !control.z_write_enable ||
	    control.zfunc == CompareAlways || !DepthTestedDraw(context)) {
		return false;
	}
	constexpr uint32_t MinExtent = 256;
	const auto         mask      = context.GetRenderTargetMask();
	uint32_t           targets   = 0;
	for (uint32_t slot = 0; slot < 8; slot++) {
		const auto& target = context.GetRenderTarget(slot);
		if (((mask >> (slot * 4u)) & 0x0fu) == 0 || target.base.addr == 0) {
			continue;
		}
		if (target.attrib2.width + 1u < MinExtent || target.attrib2.height + 1u < MinExtent) {
			return false;
		}
		targets++;
	}
	return targets >= 2;
}

// Whether this draw's target is one the game draws every frame (see above), noting the draw.
bool RedrawnTargetDraw(const HW::Context& context) {
	const auto& frames = NoteTargetFrame(context);
	if (!g_defer_redrawn) {
		return false;
	}
	if (ReflectionCubeDraw(context)) {
		// Round 146: from its first frame. A race start can draw its cube for the first time in
		// the session: on a cold cache (a new car or track after a shader reset) its draws
		// compiled 33 pipelines on the GPU thread, 26 s of a 51 s freeze before the countdown.
		// The cube is drawn again the next frame.
		return true;
	}
	return (frames.steady && DepthTestedDraw(context)) || GbufferPassDraw(context);
}

} // namespace

void NoteReflectionCubeTarget(uint64_t address) {
	const auto known = std::span(g_reflection_cubes.data(), g_reflection_cube_count);
	if (address == 0 || g_reflection_cube_count >= g_reflection_cubes.size() ||
	    std::ranges::find(known, address) != known.end()) {
		return;
	}
	g_reflection_cubes[g_reflection_cube_count++] = address;
}

// Round 143: KYTY_PS_BARYCENTRICS=0 leaves a pixel shader's raw barycentric registers (the I/J
// VGPRs the hardware fills) unset, so they read as 0 and the shader declares no gl_BaryCoordKHR
// for them; "mesh" does it only for pixel shaders drawn with a mesh shader. Interpolated inputs
// do not need these registers (V_INTERP_P2 reads the attribute itself); shaders that interpolate
// by hand from them come out flat. A test for drivers whose shader compiler fails on
// barycentrics: an AMD driver crashes building the first mesh pipeline (issue 21).
enum class RawBarycentrics : uint8_t { Keep, Mesh, Off };

static RawBarycentrics RawBarycentricsMode() {
	static const RawBarycentrics mode = [] {
		const char*     value  = std::getenv("KYTY_PS_BARYCENTRICS");
		RawBarycentrics result = RawBarycentrics::Keep;
		if (value != nullptr && std::strcmp(value, "0") == 0) {
			result = RawBarycentrics::Off;
		} else if (value != nullptr && std::strcmp(value, "mesh") == 0) {
			result = RawBarycentrics::Mesh;
		}
		const char* text = result == RawBarycentrics::Off    ? "read as 0 in every pixel shader"
		                   : result == RawBarycentrics::Mesh ? "read as 0 with mesh shaders"
		                                                     : "from the host's barycentrics";
		::printf("Pixel shaders: raw barycentric registers %s (KYTY_PS_BARYCENTRICS)\n", text);
		LOGF("Pixel shaders: raw barycentric registers %s (KYTY_PS_BARYCENTRICS)\n", text);
		return result;
	}();
	return mode;
}

bool PipelineCache::DescribeGraphicsStages(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    std::array<ShaderParams, 3>& vertex_params, ShaderParams& pixel_params, bool scan) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	// The scanner's register copy can be stale: a user SGPR count past 32 (a 6-bit field) ends the
	// emulator in GetShaderParams, so the scanner leaves such a draw to the GPU thread.
	constexpr uint32_t MaxUserSgprs = HW::UserSgprInfo::SGPRS_MAX;
	if (scan && (vertex_regs.gs_regs.rsrc2.user_sgpr > MaxUserSgprs ||
	             (pixel_active && pixel_regs.ps_regs.rsrc2.user_sgpr > MaxUserSgprs))) {
		return false;
	}
	if (tess_active && scan) {
		if (!PrepareTessellationProgramsForScan(vertex_regs, context, vertex_info, vertex_params)) {
			return false;
		}
	} else if (tess_active) {
		if (!PrepareTessellationPrograms(vertex_regs, context, vertex_info, vertex_params)) {
			if (!ShaderFailureNonFatal()) {
				EXIT("unsupported tessellation programs\n");
			}
			return false;
		}
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
		if (vertex_info[0].prepare_failed) {
			// Skipped like the draws of a shader that gives up, and reported once per shader.
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			if (logged.insert(vertex_params[0].hash).second) {
				LOGF("vertex shader 0x%016" PRIx64 " at 0x%016" PRIx64
				     ": its vertex inputs could not be described, draws skipped\n",
				     vertex_params[0].hash, vertex_regs.es_regs.data_addr);
			}
			return false;
		}
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		// The pipeline requires the wave size where it can (PrepareGraphicsPipeline).
		const auto required =
		    m_graphics.GraphicsSubgroupSize(vk::ShaderStageFlagBits::eMeshEXT, mesh.wave_size);
		mesh.host_subgroup_size = required != 0 ? required : m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		// A subgroup with more threads than a mesh workgroup may have (NVIDIA: 128) runs its
		// waves in passes; see EmitMeshEntryPoint.
		mesh.passes = mesh.PassesFor(
		    std::min(limits.maxMeshWorkGroupInvocations, limits.maxMeshWorkGroupSize[0]));
		if (mesh.passes == 0 || mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			// Skipped like the draws of a shader that gives up, and reported once per shader.
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			if (logged.insert(vertex_params[0].hash).second) {
				LOGF("mesh shader 0x%016" PRIx64 " exceeds host limits, draws skipped: wave%u "
				     "threads=%u vertices=%u primitives=%u LDS=%u\n",
				     vertex_params[0].hash, mesh.wave_size, mesh.HostThreads(), mesh.max_vertices,
				     mesh.max_primitives, mesh.lds_size_dwords);
			}
			return false;
		}
	}
	if (pixel_active) {
		static std::once_flag float_store_logged;
		std::call_once(float_store_logged, [] {
			const char* mode = ShaderRecompiler::IR::FloatStoreClampEnabled()
			                       ? "beyond the half range, the largest half, as the guest GPU "
			                         "(KYTY_FLOAT_STORE_CLAMP=0 leaves the host's conversion)"
			                       : "the host's conversion (KYTY_FLOAT_STORE_CLAMP=0)";
			::printf("Half-float stores: %s\n", mode);
			LOGF("Half-float stores: %s\n", mode);
		});
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		if (const auto raw = RawBarycentricsMode();
		    raw == RawBarycentrics::Off || (raw == RawBarycentrics::Mesh && mesh_active)) {
			// Part of the program key: the default keeps its translations.
			pixel_info.ps_perspective_center_vgpr   = UINT32_MAX;
			pixel_info.ps_perspective_centroid_vgpr = UINT32_MAX;
		}
		// SPI_SHADER_COL_FORMAT describes export packing, not the attachment numeric type.
		// In particular, 32-bit exports can carry raw integer material data.
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if (rt.base.addr == 0 ||
			    render_target_mask_slot(context.GetRenderTargetMask(), slot) == 0) {
				continue;
			}
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << slot;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << slot;
			} else if (rt.info.channel_type == Prospero::ChannelType::kFloat &&
			           (rt.info.format == Prospero::ChannelLayout::k16 ||
			            rt.info.format == Prospero::ChannelLayout::k16_16 ||
			            rt.info.format == Prospero::ChannelLayout::k16_16_16_16) &&
			           ShaderRecompiler::IR::FloatStoreClampEnabled()) {
				pixel_info.target_half_float_mask |= 1u << slot;
			}
		}
		// Exports to unbound targets are dead; they need not block the dual-source takeover of
		// MRT1.
		const auto target_live = [&](uint32_t slot) {
			return context.GetRenderTarget(slot).base.addr != 0 &&
			       render_target_mask_slot(context.GetRenderTargetMask(), slot) != 0;
		};
		bool other_targets_free = true;
		for (uint32_t slot = 1; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if (pixel_info.target_output_mode[slot] != 0 && target_live(slot)) {
				other_targets_free = false;
			}
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           ((pixel_info.target_uint_mask | pixel_info.target_sint_mask) & 1u) == 0 &&
		           other_targets_free) {
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
				case BlendMappingSupport::OneAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::OneAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				for (uint32_t slot = 2; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
					pixel_info.target_output_mode[slot] = 0;
				}
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
		if (pixel_info.dual_source_blending) {
			// The second source takes the numeric type of target 0.
			pixel_info.target_uint_mask =
			    (pixel_info.target_uint_mask & ~2u) | ((pixel_info.target_uint_mask & 1u) << 1u);
			pixel_info.target_sint_mask =
			    (pixel_info.target_sint_mask & ~2u) | ((pixel_info.target_sint_mask & 1u) << 1u);
			pixel_info.target_half_float_mask = (pixel_info.target_half_float_mask & ~2u) |
			                                    ((pixel_info.target_half_float_mask & 1u) << 1u);
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	return true;
}

// The vertex-side identity of a draw for DrawPrep::NoteGoodPair / IsGoodPair.
static uint64_t DrawPrepVertexIdentity(const HW::VertexShaderInfo& vertex_regs) {
	return vertex_regs.es_regs.data_addr ^ (vertex_regs.gs_regs.data_addr << 1u);
}

// A tessellated draw's (round 78): its three programs and the registers that configure the
// tessellator, tagged apart from the identities above.
static uint64_t DrawPrepTessIdentity(const HW::VertexShaderInfo& vertex_regs,
                                     const HW::ShaderRegisters&  sh) {
	const uint64_t words[6] = {0x7e55u,
	                           vertex_regs.ls_regs.data_addr,
	                           vertex_regs.hs_regs.data_addr,
	                           vertex_regs.es_regs.data_addr,
	                           sh.m_vgtLsHsConfig,
	                           sh.m_vgtTfParam};
	return XXH3_64bits(words, sizeof(words));
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	// KYTY_TABLE (tableMode.h). The key is the registers describe reads: the shader registers are
	// the context's own (RefreshShaders); anything else, or a bisect step, takes the normal path.
	if (!TableMode::Enabled() || &sh != &context.GetShaderRegisters() || Bisect::Active()) {
		return GetGraphicsProgramsFull(vertex_regs, pixel_regs, sh, context, user_config,
		                               target_export_mapping, pixel_active, vertex_info, pixel_info);
	}
	TableMode::LogIfDue();
	TableMode::ProgramHitFlag() = false;
	if (m_program_table == nullptr) {
		m_program_table = std::make_unique<ProgramTable>();
	}
	auto&                table = *m_program_table;
	HW::VertexShaderInfo vertex_key;
	HW::PixelShaderInfo  pixel_key;
	ProgramTable::KeyRegs(vertex_regs, pixel_regs, vertex_key, pixel_key);
	const uint64_t hash = ProgramTable::Hash(vertex_key, pixel_key, context, user_config,
	                                         target_export_mapping, pixel_active);
	const uint64_t program_gen    = m_program_cache->next_shader_id;
	const uint64_t shader_map_gen = ShaderMapGeneration();
	auto&          slot           = table.slots[hash & (ProgramTable::Slots - 1u)];
	const bool     tess_active    = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;

	// The table's answer, when the entry holds: this draw is described (its user data, its
	// vertex tables), what the program keys take from that must be the entry's, and each stage's
	// refresh must be the scanner's with the specialization of the entry's permutation.
	ProgramTable::Entry* hit = nullptr;
	GraphicsPrograms     hit_programs;
	if (slot != nullptr && slot->hash == hash &&
	    ProgramTable::SameKey(*slot, vertex_key, pixel_key, context, user_config,
	                          target_export_mapping, pixel_active)) {
		auto& e = *slot;
		if (e.program_gen != program_gen || e.shader_map_gen != shader_map_gen) {
			TableMode::Add(TableMode::Counter::ProgramMissStale);
		} else {
			std::array<ShaderParams, 3> vertex_params;
			ShaderParams                pixel_params;
			GpuPhases::Scope            describe_phase(GpuPhases::Phase::Describe);
			bool ok = DescribeGraphicsStages(vertex_regs, pixel_regs, sh, context, user_config,
			                                 target_export_mapping, pixel_active, vertex_info,
			                                 pixel_info, vertex_params, pixel_params);
			describe_phase.End();
			const uint32_t vertex_count = tess_active ? 3u : 1u;
			ok = ok && e.vertex_count == vertex_count &&
			     e.stage_count == vertex_count + (pixel_active ? 1u : 0u);
			// The stages in Get's order: pixel first.
			const auto same_inputs = [&](uint32_t k, const ShaderParams& params,
			                             const auto& input_info) {
				if (params.code.data() != e.code[k] || params.hash != e.code_hash[k] ||
				    params.user_data_count != e.user_data_count[k]) {
					return false;
				}
				BuildStageStaticKey(input_info, table.static_key_scratch);
				return table.static_key_scratch == e.static_key[k];
			};
			uint32_t k = 0;
			if (ok && pixel_active) {
				ok = same_inputs(k++, pixel_params, pixel_info);
			}
			for (uint32_t i = 0; ok && i < vertex_count; i++) {
				ok = same_inputs(k++, vertex_params[i], vertex_info[i]);
			}
			if (!ok) {
				TableMode::Add(TableMode::Counter::ProgramMissInputs);
			} else {
				k = 0;
				if (pixel_active) {
					ok = m_program_cache->TableRefresh(*e.sources[k], ShaderType::Pixel,
					                                   pixel_params, e.specialization[k]);
					pixel_info.stage = {.program   = e.pixel_info.stage.program,
					                    .resources = &e.sources[k]->resources};
					k++;
				}
				for (uint32_t i = 0; ok && i < vertex_count; i++, k++) {
					ok = m_program_cache->TableRefresh(*e.sources[k], vertex_info[i].logical_stage,
					                                   vertex_params[i], e.specialization[k]);
					vertex_info[i].stage = {.program   = e.vertex_info[i].stage.program,
					                        .resources = &e.sources[k]->resources};
					// The hull shader's normalized copy follows its samplers (GetGraphicsProgramsFull).
					const auto& program = e.programs.vertex[i];
					ok = ok && (program.normalized_module != nullptr && UnnormalizedSamplingFix() &&
					            StageSamplersAllUnnormalized(vertex_info[i].stage)) ==
					               program.normalized_samplers;
				}
				if (ok) {
					hit          = &e;
					hit_programs = e.programs;
				} else {
					TableMode::Add(TableMode::Counter::ProgramMissReads);
				}
			}
		}
	} else {
		TableMode::Add(TableMode::Counter::ProgramMissNew);
	}

	if (hit != nullptr && !TableMode::Verifying()) {
		TableMode::Add(TableMode::Counter::ProgramHits);
		TableMode::ProgramHitFlag() = true;
		for (uint32_t i = 0; i < hit->vertex_count; i++) {
			if (vertex_info[i].fetch_embedded || vertex_info[i].fetch_external) {
				TableMode::Add(TableMode::Counter::ProgramFetchTables);
			}
		}
		hit_programs.redrawn_target = RedrawnTargetDraw(context);
		return hit_programs;
	}
	// KYTY_TABLE=2: what the hit gave, before the normal path describes again.
	std::array<ShaderStageRuntime, 3> hit_vertex_stages {};
	ShaderStageRuntime                hit_pixel_stage {};
	if (hit != nullptr) {
		for (uint32_t i = 0; i < hit->vertex_count; i++) {
			hit_vertex_stages[i] = vertex_info[i].stage;
		}
		hit_pixel_stage = pixel_info.stage;
	}

	// The normal path, noting the source entries and specializations its stages were found with.
	m_program_cache->table_capture              = true;
	m_program_cache->table_capture_count        = 0;
	m_program_cache->table_capture_source_count = 0;
	auto result = GetGraphicsProgramsFull(vertex_regs, pixel_regs, sh, context, user_config,
	                                      target_export_mapping, pixel_active, vertex_info,
	                                      pixel_info);
	m_program_cache->table_capture = false;
	const uint32_t captured        = m_program_cache->table_capture_source_count;
	const bool     complete        = static_cast<bool>(result.vertex[0]) &&
	                      (!pixel_active || static_cast<bool>(result.pixel));
	const uint32_t vertex_count = complete ? result.VertexStageCount() : 0u;

	if (hit != nullptr) {
		// KYTY_TABLE=2: the table's answer against the normal path's; the normal one draws.
		TableMode::Add(TableMode::Counter::ProgramHits);
		TableMode::ProgramHitFlag() = true;
		TableMode::Add(TableMode::Counter::ProgramVerifyChecks);
		const char* diff = nullptr;
		if (!complete) {
			diff = "the normal path made no programs";
		} else if (vertex_count != hit->vertex_count) {
			diff = "vertex stage count";
		} else if (pixel_active && (result.pixel.id != hit_programs.pixel.id ||
		                            pixel_info.stage.program != hit_pixel_stage.program)) {
			diff = "pixel program";
		} else if (pixel_active && pixel_info.stage.resources != hit_pixel_stage.resources) {
			diff = "pixel resources";
		} else {
			for (uint32_t i = 0; i < vertex_count && diff == nullptr; i++) {
				if (result.vertex[i].id != hit_programs.vertex[i].id ||
				    result.vertex[i].normalized_samplers !=
				        hit_programs.vertex[i].normalized_samplers ||
				    vertex_info[i].stage.program != hit_vertex_stages[i].program) {
					diff = "vertex program";
				} else if (vertex_info[i].stage.resources != hit_vertex_stages[i].resources) {
					diff = "vertex resources";
				}
			}
		}
		if (diff != nullptr) {
			TableMode::Add(TableMode::Counter::ProgramVerifyMismatches);
			if (TableMode::VerifyLogAllowed()) {
				LOGF_UNMUTED("Table verify: programs of vs 0x%012" PRIx64 " ps 0x%012" PRIx64
				     " differ from the normal path: %s (entry dropped)\n",
				     static_cast<uint64_t>(vertex_regs.es_regs.data_addr),
				     static_cast<uint64_t>(pixel_regs.ps_regs.data_addr), diff);
			}
			slot.reset();
		}
		return result;
	}
	TableMode::Add(TableMode::Counter::ProgramMisses);
	if (!complete) {
		return result;
	}
	const uint32_t stage_count = vertex_count + (pixel_active ? 1u : 0u);
	bool           specialized = captured == stage_count;
	for (uint32_t k = 0; specialized && k < captured; k++) {
		specialized = m_program_cache->table_capture_sources[k].second != 0;
	}
	if (!specialized) {
		// A stage built or walked here, or refreshed without the scanner's specialization hash.
		TableMode::Add(TableMode::Counter::ProgramNotStored);
		return result;
	}
	// Kept the second time its key comes.
	auto& seen = table.seen[hash & (ProgramTable::SeenSlots - 1u)];
	if (seen != hash) {
		seen = hash;
		return result;
	}
	// What the program keys took from describe, as Get built them (its params: describe's again,
	// which is deterministic in the registers; recomputed here once per store).
	std::array<ShaderParams, 3> vertex_params;
	ShaderParams                pixel_params;
	{
		std::array<ShaderVertexInputInfo, 3> info_copy = vertex_info;
		ShaderPixelInputInfo                 pixel_copy = pixel_info;
		if (!DescribeGraphicsStages(vertex_regs, pixel_regs, sh, context, user_config,
		                            target_export_mapping, pixel_active, info_copy, pixel_copy,
		                            vertex_params, pixel_params)) {
			TableMode::Add(TableMode::Counter::ProgramNotStored);
			return result;
		}
	}
	if (slot == nullptr) {
		slot = std::make_unique<ProgramTable::Entry>();
	}
	auto& e          = *slot;
	e.hash           = hash;
	e.program_gen    = program_gen;
	e.shader_map_gen = shader_map_gen;
	std::memcpy(&e.vertex_regs, &vertex_key, sizeof(vertex_key));
	std::memcpy(&e.pixel_regs, &pixel_key, sizeof(pixel_key));
	std::memcpy(&e.context, &context, sizeof(context));
	std::memcpy(&e.user_config, &user_config, sizeof(user_config));
	std::memcpy(e.export_mapping.data(), target_export_mapping.data(),
	            target_export_mapping.size_bytes());
	e.pixel_active = pixel_active;
	e.programs     = result;
	e.vertex_count = vertex_count;
	e.stage_count  = stage_count;
	const auto keep = [&](uint32_t k, const ShaderParams& params, const auto& input_info) {
		e.sources[k]         = m_program_cache->table_capture_sources[k].first;
		e.specialization[k]  = m_program_cache->table_capture_sources[k].second;
		e.code[k]            = params.code.data();
		e.code_hash[k]       = params.hash;
		e.user_data_count[k] = params.user_data_count;
		BuildStageStaticKey(input_info, e.static_key[k]);
	};
	uint32_t k = 0;
	if (pixel_active) {
		e.pixel_info = pixel_info;
		keep(k++, pixel_params, pixel_info);
	}
	for (uint32_t i = 0; i < vertex_count; i++) {
		e.vertex_info[i] = vertex_info[i];
		keep(k++, vertex_params[i], vertex_info[i]);
	}
	TableMode::Add(TableMode::Counter::ProgramStored);
	return result;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsProgramsFull(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	std::array<ShaderParams, 3> vertex_params;
	ShaderParams                pixel_params;
	GpuPhases::Scope            describe_phase(GpuPhases::Phase::Describe);
	if (!DescribeGraphicsStages(vertex_regs, pixel_regs, sh, context, user_config,
	                            target_export_mapping, pixel_active, vertex_info, pixel_info,
	                            vertex_params, pixel_params)) {
		return {};
	}
	describe_phase.End();
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	result.redrawn_target = RedrawnTargetDraw(context);
	// New shaders of 3D draws (depth test on) translate on the workers while the draw is skipped;
	// UI and layer draws translate at once (TranslationWorkers), unless the game draws their
	// target every frame (round 87: its reflection cubes).
	struct DeferScope {
		ProgramCache& cache;
		~DeferScope() { cache.SetDeferTranslation(false); }
	} defer_scope {*m_program_cache};
	m_program_cache->SetDeferTranslation(
	    (context.GetDepthControl().z_enable && !SmallColorTargetDraw(context)) ||
	    result.redrawn_target);
	struct PreparedScope {
		ProgramCache& cache;
		~PreparedScope() { cache.prepared = {}; }
	} prepared_scope {*m_program_cache};
	if (pixel_active && !tess_active) {
		m_program_cache->PrepareStagesInParallel(pixel_params, pixel_info, vertex_params[0],
		                                         vertex_info[0]);
	}
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
		if (!result.vertex[i]) {
			return {};
		}
		// A hull shader sampling through unnormalized samplers only: its copy with normalized
		// coordinates, with an id of its own (its pipelines differ), and its samplers made
		// normalized when the draw binds them (normalized_samplers).
		auto& program = result.vertex[i];
		if (program.normalized_module != nullptr && UnnormalizedSamplingFix() &&
		    StageSamplersAllUnnormalized(vertex_info[i].stage)) {
			program.module              = program.normalized_module;
			program.id                  = program.normalized_id;
			program.normalized_samplers = true;
		}
	}
	if (DrawPrep::Enabled() && (!pixel_active || result.pixel)) {
		DrawPrep::NoteGoodPair(tess_active ? DrawPrepTessIdentity(vertex_regs, sh)
		                                   : DrawPrepVertexIdentity(vertex_regs),
		                       pixel_active ? pixel_regs.ps_regs.data_addr : 0u);
	}
	return result;
}

bool PipelineCache::DrawPrepDescribeGraphics(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    DrawPrepInputs& inputs) {
	// Tessellated draws are prepared too since round 78 (GT7's car bodies: four stages a draw, all
	// walked on the GPU thread before); KYTY_BISECT's drawprep:notess leaves them to it again.
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	if (tess_active && Bisect::DrawPrepNoTess()) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return false;
	}
	if (!DrawPrep::IsGoodPair(tess_active ? DrawPrepTessIdentity(vertex_regs, sh)
	                                      : DrawPrepVertexIdentity(vertex_regs),
	                          pixel_active ? pixel_regs.ps_regs.data_addr : 0u)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnknownPair);
		return false;
	}
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	std::array<ShaderParams, 3>          vertex_params;
	if (!DescribeGraphicsStages(vertex_regs, pixel_regs, sh, context, user_config,
	                            target_export_mapping, pixel_active, vertex_info, inputs.pixel_info,
	                            vertex_params, inputs.pixel_params, true)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return false;
	}
	inputs.compute = false;
	inputs.pixel   = pixel_active;
	if (tess_active) {
		inputs.tess_params.assign(vertex_params.begin(), vertex_params.end());
		inputs.tess_info.assign(vertex_info.begin(), vertex_info.end());
		DrawPrep::Add(DrawPrep::Counter::TessScanned);
	} else {
		inputs.tess_params.clear();
		inputs.tess_info.clear();
		inputs.vertex_params = vertex_params[0];
		inputs.vertex_info   = vertex_info[0];
	}
	return true;
}

bool PipelineCache::DrawPrepDescribeCompute(const HW::ComputeShaderInfo&  regs,
                                            const HW::ShaderRegisters&    sh,
                                            const ShaderComputeInputInfo& input_info,
                                            DrawPrepInputs&               inputs) {
	if (!DrawPrep::IsGoodPair(regs.cs_regs.data_addr, UINT64_MAX)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnknownPair);
		return false;
	}
	// As DescribeGraphicsStages: a count GetShaderParams would end the emulator on.
	if (regs.cs_regs.user_sgpr > HW::UserSgprInfo::SGPRS_MAX) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return false;
	}
	inputs.compute                         = true;
	inputs.compute_info                    = input_info;
	inputs.compute_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	inputs.compute_params                  = PrepareProgram(regs, sh, inputs.compute_info);
	inputs.compute_info.lds_storage =
	    inputs.compute_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	return true;
}

bool PipelineCache::DrawPrepWalk(const DrawPrepInputs& inputs, DrawPrep::PacketResult& out) {
	out.stage_count = 0;
	if (inputs.compute) {
		if (m_program_cache->ScanStage(inputs.compute_params, inputs.compute_info,
		                               ShaderType::Compute, out.stages[0])) {
			out.stage_count = 1;
		}
		return out.stage_count != 0;
	}
	if (inputs.pixel && m_program_cache->ScanStage(inputs.pixel_params, inputs.pixel_info,
	                                               ShaderType::Pixel, out.stages[0])) {
		out.stage_count++;
	}
	if (!inputs.tess_info.empty()) {
		// LS, HS and ES, each taken by its own stage (DrawPrep::Take).
		static_assert(std::tuple_size_v<decltype(out.stages)> >= 4);
		for (size_t i = 0; i < inputs.tess_info.size() && i < inputs.tess_params.size(); i++) {
			if (m_program_cache->ScanStage(inputs.tess_params[i], inputs.tess_info[i],
			                               inputs.tess_info[i].logical_stage,
			                               out.stages[out.stage_count])) {
				out.stage_count++;
			}
		}
		return out.stage_count != 0;
	}
	if (m_program_cache->ScanStage(inputs.vertex_params, inputs.vertex_info,
	                               inputs.vertex_info.logical_stage, out.stages[out.stage_count])) {
		out.stage_count++;
	}
	return out.stage_count != 0;
}

bool PipelineCache::IsSkipListed(uint64_t shader_addr) {
	return SkipShaderRequested(ShaderDeclaredHash(shader_addr));
}

// Round 94: a compute pipeline compiling on a thread of its own (TryGetComputePipeline).
struct PipelineCache::BackgroundCompute {
	std::unique_ptr<Pipeline> pipeline;
	// The program info the compile reads (its hash and wave size), copied: the program cache's
	// entry is an element of a vector a later permutation of the same source may move.
	ShaderRecompiler::IR::CompiledShaderInfo program;
	ShaderComputeInputInfo                   input_info {};
	uint64_t                                 hash  = 0;
	uint32_t                                 words = 0;
	std::chrono::steady_clock::time_point    begin;
	uint64_t                                 skipped = 0; // dispatches skipped meanwhile
	// Round 98: the expanded code (empty when not given) and the start order, for
	// m_ready_expansions once done.
	std::vector<uint32_t> expansion;
	uint64_t              serial = 0;
	// Round 99: a partition's pipeline (GetComputeProgram), which takes half a compile slot.
	bool                  partition = false;
	// Round 117: started by StartWaitingPartitions, not by a dispatch of its own.
	bool                  ahead = false;
	std::atomic<bool>     done {false};
	std::thread           thread;
};

// Round 117: a partition waiting for a compile slot, with what its compile needs, so that a free
// slot starts it without waiting for its next dispatch (StartWaitingPartitions). Its layouts are in
// m_compute_waiting.
struct PipelineCache::WaitingPartition {
	ShaderRecompiler::IR::CompiledShaderInfo program;
	ShaderComputeInputInfo                   input_info {};
	vk::ShaderModule                         module = nullptr;
	uint32_t                                 words  = 0;
	// The dispatches that asked for it meanwhile (skipped), and the order it first waited in.
	uint64_t asks   = 0;
	uint64_t serial = 0;
};

// Round 117: KYTY_PARTITION_QUICK=1: the partitions of the ray tracing material passes are built
// with VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT (a test of how much faster the driver compiles
// them so; their dispatches may run slower). Looked up so as well: a run without it compiles them
// again.
static bool PartitionQuick() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_PARTITION_QUICK");
		return text != nullptr && *text != '\0' && *text != '0';
	}();
	return value;
}

// Round 118: the partitioned shaders' dispatches, every 10 s a line (GPU thread): for the busiest
// ones, how many partitions their dispatches ran and how many materials they know (more when
// KYTY_PARTITION_IDLE_MS leaves out the ones not in view).
static void
NotePartitionedDispatch(uint64_t                                              hash,
                        const ShaderRecompiler::Decoder::ShaderCallPartition& partition) {
	struct Counts {
		uint64_t groups     = 0;
		uint64_t dispatches = 0;
		size_t   known      = 0;
		size_t   run        = 0;
	};
	static std::mutex                            mutex;
	static std::unordered_map<uint64_t, Counts>  shaders;
	static std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
	const std::lock_guard                        lock(mutex);
	auto&                                        counts = shaders[hash];
	counts.groups++;
	counts.dispatches += partition.targets.size();
	counts.known   = std::max(partition.known, partition.targets.size());
	counts.run     = partition.targets.size();
	const auto now = std::chrono::steady_clock::now();
	if (now - since < std::chrono::seconds(10)) {
		return;
	}
	std::vector<std::pair<uint64_t, Counts>> busiest(shaders.begin(), shaders.end());
	std::sort(busiest.begin(), busiest.end(), [](const auto& a, const auto& b) {
		return a.second.dispatches > b.second.dispatches;
	});
	uint64_t total = 0;
	for (const auto& [shader, entry]: busiest) {
		(void)shader;
		total += entry.dispatches;
	}
	std::string text;
	for (size_t i = 0; i < busiest.size() && i < 8; i++) {
		const auto& [shader, entry] = busiest[i];
		text += fmt::format(
		    "{}{:016x}: {} dispatches in {} groups, the last ran {} of {} materials",
		    i == 0 ? "" : "; ", shader, entry.dispatches, entry.groups, entry.run, entry.known);
	}
	LOGF("Partitioned dispatches (%.0f s): %" PRIu64 " in all%s; %s\n",
	     std::chrono::duration<double>(now - since).count(), total,
	     ShaderRecompiler::Decoder::ShaderCallPartitionIdle() != 0u
	         ? fmt::format(" (materials reported in the last {} ms)",
	                       ShaderRecompiler::Decoder::ShaderCallPartitionIdle())
	               .c_str()
	         : " (every material known)",
	     text.c_str());
	shaders.clear();
	since = now;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               ComputePartition*            partition) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	const auto        input_before     = input_info;
	uint32_t          push_data_cursor = 0;
	const bool        keep_last        = KeepLastExpansion() && params.hash != 0;
	if (partition != nullptr) {
		partition->targets.clear();
	}
	// Round 99: a partition's program (ShaderCallPartition). The partitions' expansions never
	// change (each inlines its target alone), so there is no earlier one to run meanwhile: a
	// partition whose program translates or compiles is skipped. A new partition's SPIR-V takes
	// the GPU thread up to a quarter of the time (m_partition_next_build).
	const auto get_partition = [&]() {
		const auto begin                  = std::chrono::steady_clock::now();
		m_program_cache->call_partition   = partition;
		m_program_cache->partition_budget = begin >= m_partition_next_build;
		m_program_cache->partition_built  = false;
		auto program = m_program_cache->Get(params, input_info, push_data_cursor);
		m_program_cache->call_partition = nullptr;
		if (m_program_cache->partition_built) {
			const auto end         = std::chrono::steady_clock::now();
			m_partition_next_build = end + (end - begin) * 3;
		}
		return program;
	};
	if (partition != nullptr && m_partitioned_shaders.contains(params.hash)) {
		const bool first_of_dispatch = partition->target == 0;
		auto       program           = get_partition();
		if (first_of_dispatch) {
			NotePartitionedDispatch(params.hash, *partition);
		}
		if (partition->targets.empty()) {
			// Its call no longer partitions (the table call took another shape): the usual way
			// from its next dispatch.
			m_partitioned_shaders.erase(params.hash);
		}
		if (program) {
			m_partition_programs.insert(program.id);
			if (DrawPrep::Enabled()) {
				DrawPrep::NoteGoodPair(regs.cs_regs.data_addr, UINT64_MAX);
			}
		}
		return program;
	}
	if (keep_last) {
		// Round 98: while one expansion of the shader compiles in the background, its newer ones
		// (with the targets reported meanwhile) are neither translated nor compiled: the last
		// ready one runs, or the dispatch waits for the one compiling, and the newest expansion is
		// translated once that compile is done. Each new expansion of a material pass took the GPU
		// thread up to a second to translate (14 million SPIR-V words), and rt97 made 71.
		ReapBackgroundCompute();
		// Round 99: or waits for a compile slot (PendingExpansion). Only a dispatch of its
		// program starts that compile (TryGetComputePipeline), and while it waits the dispatches
		// run the ready one: each of them first tries again to start it.
		const auto* pending = PendingExpansion(params.hash);
		if (pending != nullptr && CompilingCompute(params.hash) == nullptr) {
			// A copy: starting the compile takes the entry the pointer is into.
			const std::vector<uint32_t> waiting_code = *pending;
			m_program_cache->forced_function_code    = &waiting_code;
			const auto waiting = m_program_cache->Get(params, input_info, push_data_cursor);
			m_program_cache->forced_function_code = nullptr;
			if (waiting) {
				(void)TryGetComputePipeline(input_info, waiting, waiting_code);
			}
			input_info       = input_before;
			push_data_cursor = 0;
			pending          = PendingExpansion(params.hash);
		}
		if (pending != nullptr) {
			const auto  ready = m_ready_expansions.find(params.hash);
			const auto& held  = ready != m_ready_expansions.end() ? ready->second.code : *pending;
			if (!held.empty()) {
				m_program_cache->forced_function_code = &held;
				auto program = m_program_cache->Get(params, input_info, push_data_cursor);
				m_program_cache->forced_function_code = nullptr;
				if (program && (m_compute_pipelines.contains(program.id) ||
				                m_compute_background.contains(program.id) ||
				                m_compute_waiting.contains(program.id))) {
					if (DrawPrep::Enabled()) {
						DrawPrep::NoteGoodPair(regs.cs_regs.data_addr, UINT64_MAX);
					}
					return program;
				}
				input_info       = input_before;
				push_data_cursor = 0;
			}
		}
	}
	auto program = partition != nullptr && params.hash != 0
	                   ? get_partition()
	                   : m_program_cache->Get(params, input_info, push_data_cursor);
	if (program && DrawPrep::Enabled()) {
		// Compute shaders join the known-good list with a vertex identity no draw has.
		DrawPrep::NoteGoodPair(regs.cs_regs.data_addr, UINT64_MAX);
	}
	if (partition != nullptr && !partition->targets.empty()) {
		m_partitioned_shaders.insert(params.hash);
		LOGF("Compute shader %016" PRIx64 ": its call is partitioned, one dispatch per target "
		     "(%zu so far)\n",
		     params.hash, partition->targets.size());
		if (program) {
			m_partition_programs.insert(program.id);
		}
		return program;
	}
	// Round 97: a shader with call expansions keeps running its last expansion whose pipeline is
	// ready while a newer one, with more targets inlined, compiles in the background (a minute or
	// more for the ray tracing mode's material passes). Before, its dispatches were skipped all
	// that while, and the reflections they shade showed garbage (the rainbow came back whenever a
	// new material appeared).
	const auto& code = m_program_cache->lookup_key.function_code;
	if (!program || code.empty() || !keep_last) {
		return program;
	}
	// Round 98: the expanded code goes with a background compile it starts (TryGetComputePipeline).
	const bool runnable = TryGetComputePipeline(input_info, program, code) != nullptr;
	auto       ready    = m_ready_expansions.find(params.hash);
	if (runnable) {
		// Compiled now, or found compiled: the shader's newest ready expansion.
		if (ready == m_ready_expansions.end() || ready->second.id != program.id) {
			auto& entry  = m_ready_expansions[params.hash];
			entry.id     = program.id;
			entry.serial = ++m_expansion_serial;
			entry.code   = code;
		}
		return program;
	}
	if (ready == m_ready_expansions.end() || ready->second.id == program.id) {
		// Nothing to run meanwhile: its dispatches wait for the compile.
		return program;
	}
	const auto newer                      = program;
	input_info                            = input_before;
	push_data_cursor                      = 0;
	m_program_cache->forced_function_code = &ready->second.code;
	auto earlier = m_program_cache->Get(params, input_info, push_data_cursor);
	m_program_cache->forced_function_code = nullptr;
	if (!earlier || !m_compute_pipelines.contains(earlier.id)) {
		// Not runnable after all: the newer one, whose dispatches wait for its compile.
		input_info       = input_before;
		push_data_cursor = 0;
		return m_program_cache->Get(params, input_info, push_data_cursor);
	}
	if (m_last_expansion_logged.insert(newer.id).second && m_last_expansion_logged.size() < 4096) {
		LOGF("Compute shader %016" PRIx64 ": its earlier expansion runs while the newer one "
		     "compiles\n",
		     params.hash);
	}
	return earlier;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	auto* pipeline =
	    TryGetGraphicsPipeline(colors, depth, vertex_info, command, ps_input_info, topology,
	                           primitive_restart_enable, programs, PipelineDefer::Never);
	EXIT_IF(pipeline == nullptr);
	return *pipeline;
}

bool PipelineCache::FinishPending(Pipeline& pipeline) {
	auto& job = *pipeline.pending;
	if (!job.done.load(std::memory_order_acquire)) {
		if (m_compiler != nullptr && !m_compiler->Stopped()) {
			// The unoptimized pipeline meanwhile; `pending` stays until the optimized one.
			if (job.quick_ready.load(std::memory_order_acquire)) {
				pipeline.pipeline = job.quick;
				return true;
			}
			return false;
		}
		// The workers stopped before this job started: compile it here.
		job.result =
		    CreateThroughCacheFiles(*job.build, m_driver_older, m_driver_cache, &job.pipeline);
		Profiler::Add(Profiler::Counter::PipelinesPending, -1);
		job.done.store(true, std::memory_order_release);
	}
	EXIT_NOT_IMPLEMENTED(job.result != vk::Result::eSuccess || job.pipeline == nullptr);
	pipeline.pipeline   = job.pipeline;
	const bool compiled = job.compiled;
	if (job.quick != nullptr) {
		// ponytail: kept until shutdown, command buffers in flight may still use it; destroy it
		// after a frame fence if the driver memory of thousands of pipelines starts to matter.
		m_quick_retired.push_back(job.quick);
		job.quick = nullptr;
	}
	pipeline.pending.reset();
	// Round 134: a new pipeline in the driver cache, kept by a save during play.
	if (compiled) {
		MaybeSaveDuringPlay();
	}
	return true;
}

PipelineCache::Pipeline* PipelineCache::TryGetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, PipelineDefer defer) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
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
				case ShaderAlphaBlendSource::OneAlphaZero:
					// The second source: 1 or 0 for each physical colour channel (0 where logical
					// alpha lies), logical Sa in its A for the destination factor. Physical A
					// holds a logical colour channel (alpha lies elsewhere, or this mapping would
					// be direct), so its own source factor is One.
					blend.color_srcblend  = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.alpha_srcblend  = static_cast<uint8_t>(Prospero::BlendFactor::kOne);
					blend.alpha_destblend = blend.color_destblend;
					blend.alpha_comb_fcn  = blend.color_comb_fcn;
					blend.separate_alpha_blend = true;
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
		const uint32_t depth_samples = depth.desc.info.samples;
		if (attachment_samples == 0) {
			attachment_samples = depth_samples;
		} else if (attachment_samples != depth_samples) {
			// PS5 EQAA: color holds the fragment count, depth the coverage count.
			using MixedSamples       = GraphicContext::MixedAttachmentSamples;
			const auto     mixed     = m_graphics.mixed_attachment_samples;
			const uint32_t color_samples = attachment_samples;
			uint32_t       resolvable_samples = 0;
			if (m_graphics.AcceptsMixedSamples(color_samples, depth_samples)) {
				rendering.color_samples        = color_samples;
				rendering.depth_samples        = depth_samples;
				attachment_samples             = std::max(color_samples, depth_samples);
				static std::atomic_bool logged = false;
				if (!logged.exchange(true, std::memory_order_relaxed)) {
					LOGF("Pipeline: mixed attachment samples (%s): color %u, depth %u, raster %u\n",
					     mixed == MixedSamples::Amd ? "AMD" : "NV", color_samples, depth_samples,
					     attachment_samples);
				}
			} else if (MsColorTargetsResolvable(colors, resolvable_samples) &&
			           EmulatesColorAtDepthSamples(false, resolvable_samples, depth_samples)) {
				// The color is 1-sample and the depth is not: render the color into a temporary
				// image at the depth sample count and resolve it (renderDraw.cpp). Every
				// attachment then shares one sample count, so depth stays attached.
				attachment_samples             = depth_samples;
				static std::atomic_bool logged = false;
				if (!logged.exchange(true, std::memory_order_relaxed)) {
					LOGF("Pipeline: emulated EQAA: color %u rendered at %u samples and resolved\n",
					     color_samples, depth_samples);
				}
			} else {
				// ponytail: no mixed-sample extension on the host (RenderDoc hides
				// VK_NV_framebuffer_mixed_samples). Vulkan needs one sample count for all
				// attachments, so render this draw without depth instead of ending the emulator.
				// Ceiling: no depth test/write for these draws; renderDraw.cpp drops the
				// attachment to match.
				rendering.depth_format         = vk::Format::eUndefined;
				rendering.stencil_format       = vk::Format::eUndefined;
				static std::atomic_bool warned = false;
				if (!warned.exchange(true, std::memory_order_relaxed)) {
					LOGF("Pipeline: no mixed-sample support: rendering color %u / depth %u draws "
					     "without depth\n",
					     color_samples, depth_samples);
				}
			}
		}
	}
	{
		const bool exports_layer = std::ranges::any_of(vertex_info, [](const auto& info) {
			return info.stage && std::ranges::any_of(info.stage.program->info.outputs, [](const auto& o) {
				       return o.kind == ShaderRecompiler::IR::StageOutputKind::Layer;
			       });
		});
		uint32_t layers = UINT32_MAX;
		for (uint32_t i = 0; i < color_count; i++)
			layers = std::min(layers, colors[i].desc.view_info.layer_count);
		if (with_depth) layers = std::min(layers, depth.desc.view_info.layer_count);
		rendering.layer_limit = exports_layer && layers != UINT32_MAX ? std::max(layers, 1u) : 0u;
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
	if (static_params.sample_shading_enable && rendering.color_samples != 0 &&
	    m_graphics.mixed_attachment_samples == GraphicContext::MixedAttachmentSamples::Nv) {
		// NV mixed samples shades per pixel with coverage modulation; per-sample shading
		// against fewer color samples than raster samples is not something it defines.
		static_params.sample_shading_enable = false;
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("Pipeline: temporary: per-sample shading disabled for NV mixed-sample draw\n");
		}
	}
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	// Round 159: PA_SC_AA_MASK reduced to the draw's samples (spirvCoverageMask.h). Until the copy
	// of the pixel shader that applies it is made, the draw runs as before (all ones in the key).
	vk::ShaderModule coverage_module = nullptr;
	if (m_coverage != nullptr && rendering.color_samples == 0) {
		const uint32_t samples = std::clamp<uint32_t>(attachment_samples, 1u, 16u);
		const uint32_t full    = samples >= 16u ? 0xffffu : (1u << samples) - 1u;
		const uint32_t both    = full | (full << 16u);
		const uint32_t mask01  = ctx.GetAaMask(0) & both;
		const uint32_t mask23  = ctx.GetAaMask(1) & both;
		if ((mask01 != both || mask23 != both) && AaMaskOn()) {
			coverage_module = m_coverage->Find(
			    ps_active ? pixel_program.id : 0u,
			    ps_active ? ps_input_info->stage.program->shader_hash : 0u, mask01, mask23);
			if (coverage_module != nullptr) {
				static_params.aa_mask[0] = mask01;
				static_params.aa_mask[1] = mask23;
			}
		}
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);
	NormalizeGraphicsPipelineKey(static_params);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	const auto skip_draw = [&](const PendingGraphicsPipeline& job) -> Pipeline* {
		job.asked.store(AskClockMs(), std::memory_order_relaxed);
		if (job.looking_up.load(std::memory_order_acquire)) {
			// Round 130: its lookup is not done yet (milliseconds).
			GpuPhases::Count(GpuPhases::Phase::PipelineLookupSkip, 1);
			// Round 133: for the flip of this frame (GpuPhases::TakeFrameHold).
			GpuPhases::NoteLookupSkip();
			Profiler::CommandZone::Annotate("skipped: pipeline looked up");
			return nullptr;
		}
		GpuPhases::Count(GpuPhases::Phase::PipelineSkip, 1);
		// For the flip of this frame, with KYTY_HOLD_COMPILING=1 (GpuPhases::TakeFrameHold).
		GpuPhases::NoteCompileSkip();
		Profiler::Add(Profiler::Counter::SkippedPipelinePending);
		Profiler::CommandZone::Annotate("skipped: pipeline compiling");
		// Diagnostics: each pipeline whose draws are skipped, once (the first 128).
		static std::unordered_set<uint64_t> logged;
		const auto pipeline_key = (vs_id << 32u) ^ ps_id;
		if (logged.size() < 128 && logged.insert(pipeline_key).second) {
			uint32_t vs_words = 0;
			for (const auto& program: programs.vertex) {
				vs_words += program.spirv_words;
			}
			LOGF("PipelineCache: draw skipped, pipeline compiling: VS 0x%016" PRIx64
			     " (%u words) PS 0x%016" PRIx64
			     " (%u words) colors=%u depth=%d/%d stencil=%d\n",
			     vertex_program.hash, vs_words, ps_active ? pixel_program.hash : 0,
			     ps_active ? pixel_program.spirv_words : 0, color_count,
			     depth.depth_test_enable ? 1 : 0, depth.depth_write_enable ? 1 : 0,
			     depth.stencil_test_enable ? 1 : 0);
		}
		if (++m_deferred_draws % 256 == 1) {
			LOGF("PipelineCache: %" PRIu64 " draws skipped while their pipeline compiled in the "
			     "background\n",
			     m_deferred_draws);
		}
		return nullptr;
	};
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		if (found.pending && !FinishPending(found)) {
			if (defer == PipelineDefer::Allowed) {
				return skip_draw(*found.pending);
			}
			// A draw that may not be skipped waits for the compile another draw of the same
			// pipeline sent to the background (round 87: more of those are sent now).
			const GpuPhases::Scope wait_phase(GpuPhases::Phase::PipelineWait);
			bool                   stalled = false;
			while (!FinishPending(found)) {
				m_compiler->Ask(*found.pending, stalled);
				(void)m_compiler->Wait(*found.pending, std::chrono::milliseconds(100));
				stalled = true;
			}
		}
		return &found;
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

	uint8_t seed_state = SeedUnrecorded;
	if (PipelineSeeds::Recording() && coverage_module == nullptr) {
		// Guest pipeline seeds (pipelineSeeds.h): once per new pipeline, its programs by their
		// seeds and the key's state.
		seed_state = RecordPipelineSeed(key, programs, static_cast<uint32_t>(vertex_info.size()), ps_active);
	}
	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	auto build_programs            = programs;
	build_programs.coverage_module = coverage_module;
	auto build = PrepareGraphicsPipeline(m_graphics, *cached, rendering, key.vertex_input,
	                                     vertex_info, ps_input_info, build_programs, static_params);
	build->seed_state = seed_state;
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	uint64_t vertex_words = 0;
	for (const auto& program: programs.vertex) {
		vertex_words += program.spirv_words;
	}
	const uint64_t spirv_words = vertex_words + pixel_program.spirv_words;
	const uint64_t vs_hash     = vs_input_info.stage.program->shader_hash;
	const uint64_t ps_hash     = ps_active ? ps_input_info->stage.program->shader_hash : 0;
	// Stencil does not count: UI libraries clip their panels with stencil masks. A target the game
	// draws every frame is no layer drawn once (round 87).
	const bool layer_draw = !programs.redrawn_target && !depth.depth_test_enable &&
	                        !depth.depth_write_enable &&
	                        vertex_words <= g_pipeline_sync_ui_vs_words;
	const bool background = defer == PipelineDefer::Allowed &&
	                        spirv_words > g_pipeline_sync_words && !layer_draw &&
	                        m_compiler != nullptr && !m_compiler->Stopped();
	// Round 130: a new pipeline of a draw into a target the game draws every frame is looked up
	// on a worker (F8's probe:sync: on the GPU thread, as before), and the draw left out until
	// that is done; a pipeline the caches do not hold is compiled next, as before.
	if (const auto unanswered = g_lookup_unanswered.exchange(0, std::memory_order_relaxed);
	    unanswered != 0 && m_probe) {
		m_probe = false;
		PipelineCacheLog("Vulkan pipelines: the driver's cache lookup returned {}; looking up no "
		                 "more",
		                 vk::to_string(static_cast<vk::Result>(unanswered)));
	}
	const bool lookup_async = background && m_probe && programs.redrawn_target &&
	                          m_compiler->LooksUp() && !Bisect::PipelineProbeSync();
	bool probed = false;
	if (background && m_probe && !lookup_async) {
		// Round 87: a pipeline the driver has in its caches is created here at once, and one it
		// would have to compile goes to a worker at once. Before, the GPU thread waited up to
		// 20 ms for each new pipeline in case it was cached, and those of a new track take the
		// driver about a second each: the waits took 2-4 s of every 10 s of a race on one.
		const auto       probe_begin = std::chrono::steady_clock::now();
		GpuPhases::Scope probe_phase(GpuPhases::Phase::PipelineProbe);
		// The full cache files first (round 92), then the one new pipelines go to.
		const auto result =
		    CreateFromOlderFiles(*build, m_driver_older, &cached->pipeline)
		        ? vk::Result::eSuccess
		        : CreateGraphicsPipeline(*build, m_driver_cache, &cached->pipeline, true);
		probe_phase.End();
		if (result == vk::Result::eSuccess && cached->pipeline != nullptr) {
			GpuPhases::Count(GpuPhases::Phase::PipelineProbeHit, 1);
			ReportGraphicsPipelineBuilt(vs_hash, ps_hash, probe_begin);
			LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
			auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
			EXIT_IF(!inserted);
			return iter->second.get();
		}
		if (cached->pipeline != nullptr) {
			m_graphics.device.destroyPipeline(cached->pipeline, nullptr);
			cached->pipeline = nullptr;
		}
		if (result != vk::Result::ePipelineCompileRequired) {
			// A driver that cannot answer: compile as before, without asking again.
			m_probe = false;
			PipelineCacheLog("Vulkan pipelines: the driver's cache lookup returned {}; looking "
			                 "up no more",
			                 vk::to_string(result));
		}
		probed = true;
	}
	if (background) {
		// A new pipeline is compiled in the background, and its draws are skipped until it is
		// ready; a short wait catches one the driver had cached (KYTY_PIPELINE_FRAME_WAIT_MS a
		// frame in all, at most KYTY_PIPELINE_WAIT_MS for one).
		auto job          = std::make_shared<PendingGraphicsPipeline>();
		job->build        = std::move(build);
		job->driver_cache = m_driver_cache;
		if (!probed) {
			job->older = m_driver_older;
		}
		job->vs_hash      = vs_hash;
		job->ps_hash      = ps_hash;
		job->looking_up.store(lookup_async, std::memory_order_relaxed);
		cached->pending   = job;
		Profiler::Add(Profiler::Counter::PipelinesPending);
		m_compiler->Submit(job);
		if (probed) {
			GpuPhases::Count(GpuPhases::Phase::PipelineProbeMiss, 1);
		}
		WaitForPipeline(*job);
		auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
		EXIT_IF(!inserted);
		auto& pipeline = *iter->second;
		if (!FinishPending(pipeline)) {
			return skip_draw(*job);
		}
		LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
		return &pipeline;
	}
	const auto       build_begin = std::chrono::steady_clock::now();
	GpuPhases::Scope compile_phase(GpuPhases::Phase::PipelineCompile);
	const auto       result =
	    CreateThroughCacheFiles(*build, m_driver_older, m_driver_cache, &cached->pipeline);
	compile_phase.End();
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	ReportGraphicsPipelineBuilt(vs_hash, ps_hash, build_begin);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
	if (const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
	                                                              build_begin)
	                        .count();
	    ms >= 100.0) {
		// Round 87: why a slow one was compiled on the GPU thread (it stopped the game that long).
		const char*     reason = defer == PipelineDefer::WritesMemory ? "its draw writes memory"
		                         : defer == PipelineDefer::SmallTarget
		                             ? "a small target not drawn every frame"
		                         : defer == PipelineDefer::Never        ? "asked for at once"
		                         : spirv_words <= g_pipeline_sync_words ? "small shaders"
		                         : layer_draw ? "a draw without depth, a UI or layer draw"
		                                      : "no compile threads";
		const auto&     target = color_count > 0 ? colors[0].desc.info : depth.desc.info;
		static uint64_t slow   = 0;
		slow++;
		if (slow <= 64 || std::has_single_bit(slow)) {
			LOGF("PipelineCache: compiled on the GPU thread in %.0f ms (%s): VS 0x%016" PRIx64
			     " (%" PRIu64 " words) PS 0x%016" PRIx64 " (%u words), target 0x%" PRIx64
			     " %ux%u layers %u (%u colors), depth test %d write %d, redrawn %d (%" PRIu64
			     " so far)\n",
			     ms, reason, vs_hash, vertex_words, ps_hash,
			     ps_active ? pixel_program.spirv_words : 0u, target.data.address,
			     target.extent.width, target.extent.height, target.resources.layers, color_count,
			     depth.depth_test_enable ? 1 : 0, depth.depth_write_enable ? 1 : 0,
			     programs.redrawn_target ? 1 : 0, slow);
		}
	}

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return iter->second.get();
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

	auto cached = std::make_unique<Pipeline>();
	{
		const auto hash = input_info.stage.program != nullptr ? input_info.stage.program->shader_hash : 0;
		Profiler::Phases zone;
		zone.SetText("CS 0x%016" PRIx64, hash);
		KYTY_PROFILER_PHASE(zone, "Compute pipeline build", profiler::colors::RedA100);
		const auto             begin = std::chrono::steady_clock::now();
		const GpuPhases::Scope phase(GpuPhases::Phase::PipelineCompile);
		if (CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                           m_driver_cache, m_driver_older)) {
			ReportPipelineBuilt(Profiler::Counter::ComputePipelinesBuilt, "Compute", hash, 0,
			                    begin);
		}
	}

	// Round 105: one the driver could not build is kept as such, and its dispatches skipped.
	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr && !cached->build_failed);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}

void PipelineCache::ReapBackgroundCompute() {
	for (auto it = m_compute_background.begin(); it != m_compute_background.end();) {
		auto& job = *it->second;
		if (!job.done.load(std::memory_order_acquire)) {
			++it;
			continue;
		}
		job.thread.join();
		const auto seconds =
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - job.begin).count();
		if (job.pipeline->build_failed) {
			// Round 105: kept as failed (its dispatches skipped); the shader's ready expansion,
			// if any, stays the one it runs.
			LOGF("Compute pipeline CS=0x%016" PRIx64 " (%u SPIR-V words)%s failed to compile in "
			     "the background after %.1f s; %" PRIu64 " of its dispatches were skipped "
			     "meanwhile, and the rest are too\n",
			     job.hash, job.words, job.partition ? ", a partition," : "", seconds, job.skipped);
			m_compute_pipelines.emplace(it->first, std::move(job.pipeline));
			it = m_compute_background.erase(it);
			m_background_finished++;
			continue;
		}
		LOGF("Compute pipeline CS=0x%016" PRIx64 " (%u SPIR-V words)%s compiled in the background "
		     "in %.1f s; %" PRIu64 " of its dispatches were skipped meanwhile%s%s\n",
		     job.hash, job.words, job.partition ? ", a partition," : "", seconds, job.skipped,
		     job.ahead ? " (started while it waited)" : "",
		     job.pipeline->disable_optimization ? " (without optimization)" : "");
		ReportPipelineBuilt(Profiler::Counter::ComputePipelinesBuilt, "Compute", job.hash, 0,
		                    job.begin);
		if (job.hash != 0 && !job.expansion.empty() && !job.partition) {
			// Newer than the shader's ready expansion: it runs from now on, until a newer one is
			// ready (GetComputeProgram). A partition's is not one: it runs its target alone.
			auto& ready = m_ready_expansions[job.hash];
			if (ready.serial < job.serial) {
				ready.id     = it->first;
				ready.serial = job.serial;
				ready.code   = std::move(job.expansion);
			}
		}
		m_compute_pipelines.emplace(it->first, std::move(job.pipeline));
		it = m_compute_background.erase(it);
		m_background_finished++;
	}
	StartWaitingPartitions();
	ReportBackgroundProgress();
}

const PipelineCache::BackgroundCompute*
PipelineCache::CompilingCompute(uint64_t shader_hash) const {
	for (const auto& [id, job]: m_compute_background) {
		(void)id;
		// Round 99: partitions compile alongside each other, and none runs for the whole shader.
		if (job->hash == shader_hash && !job->partition) {
			return job.get();
		}
	}
	return nullptr;
}

const std::vector<uint32_t>* PipelineCache::PendingExpansion(uint64_t shader_hash) const {
	if (const auto* compiling = CompilingCompute(shader_hash); compiling != nullptr) {
		return &compiling->expansion;
	}
	for (const auto& [id, waiting]: m_compute_waiting_expansions) {
		(void)id;
		if (waiting.hash == shader_hash && !waiting.code.empty()) {
			return &waiting.code;
		}
	}
	return nullptr;
}

PipelineCache::Pipeline*
PipelineCache::TryGetComputePipeline(const ShaderComputeInputInfo& input_info,
                                     const ShaderProgram&          compute_program,
                                     std::span<const uint32_t>     expansion) {
	EXIT_IF(!compute_program);
	ReapBackgroundCompute();
	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		// Round 105: the driver could not build it; its dispatches are skipped.
		return iter->second->build_failed ? nullptr : iter->second.get();
	}
	if (auto iter = m_compute_background.find(compute_program.id);
	    iter != m_compute_background.end()) {
		// GetComputeProgram's own look (with the expansion) is not a dispatch.
		if (expansion.empty()) {
			iter->second->skipped++;
		}
		return nullptr;
	}
	const auto threshold = ComputeBackgroundWords();
	// Round 95: an expansion that inlined called functions from a tenth of that: one of 563,699
	// words took the driver 15.6 s, one of 387,140 5.9 s (unexpanded shaders of 400,000 words
	// compile in well under a second).
	const auto background_from =
	    compute_program.inlined_calls ? std::max<uint32_t>(threshold / 10u, 1u) : threshold;
	if (threshold == 0 || compute_program.spirv_words < background_from) {
		auto& built = GetComputePipeline(input_info, compute_program);
		return built.build_failed ? nullptr : &built;
	}
	const auto hash =
	    input_info.stage.program != nullptr ? input_info.stage.program->shader_hash : 0;
	// Round 99: the partitions of a shader (GetComputeProgram) compile alongside each other, and
	// each takes half a slot: a partition inlines one material function, a fraction of the
	// whole expansions the slots were sized for (rt97 ran out of memory with 27 of those).
	const bool                partition = m_partition_programs.contains(compute_program.id);
	std::unique_ptr<Pipeline> cached;
	std::vector<uint32_t>     waiting_code;
	uint64_t                  asks        = 0;
	uint64_t                  wait_serial = 0;
	if (auto waiting = m_compute_waiting.find(compute_program.id);
	    waiting != m_compute_waiting.end()) {
		// Looked up before, in no cache: its layouts are made.
		cached = std::move(waiting->second);
		m_compute_waiting.erase(waiting);
		if (auto code = m_compute_waiting_expansions.find(compute_program.id);
		    code != m_compute_waiting_expansions.end()) {
			waiting_code = std::move(code->second.code);
			m_compute_waiting_expansions.erase(code);
		}
		if (auto entry = m_partition_waiting.find(compute_program.id);
		    entry != m_partition_waiting.end()) {
			asks        = entry->second->asks;
			wait_serial = entry->second->serial;
			m_partition_waiting.erase(entry);
		}
	} else {
		cached                       = std::make_unique<Pipeline>();
		cached->disable_optimization = partition && PartitionQuick();
		const GpuPhases::Scope phase(GpuPhases::Phase::PipelineCompile);
		if (CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                           m_driver_cache, m_driver_older, ComputeBuild::LookUp)) {
			auto [iter, inserted] =
			    m_compute_pipelines.emplace(compute_program.id, std::move(cached));
			EXIT_IF(!inserted);
			return iter->second.get();
		}
	}
	// Round 98: one compile per shader at a time, and ComputeBackgroundThreads in all; the
	// program waits meanwhile (GetComputeProgram runs the shader's ready expansion, if any). A
	// shader with a ready expansion leaves one of them to the shaders with nothing to run.
	const auto slots       = ComputeBackgroundThreads();
	const bool same        = !partition && CompilingCompute(hash) != nullptr;
	const bool has_ready   = !partition && hash != 0 && m_ready_expansions.contains(hash);
	uint32_t   used_halves = 0;
	for (const auto& [id, job]: m_compute_background) {
		(void)id;
		used_halves += job->partition ? 1u : 2u;
	}
	const uint32_t halves  = partition ? 1u : 2u;
	const bool     no_slot = used_halves + halves > 2u * slots;
	// The expanded code goes with the program, whether the dispatch or GetComputeProgram asks.
	if (expansion.empty() && !waiting_code.empty()) {
		expansion = waiting_code;
	}
	if (same || no_slot || (has_ready && slots > 1 && used_halves + halves + 2u > 2u * slots)) {
		m_compute_waiting.emplace(compute_program.id, std::move(cached));
		if (hash != 0 && !partition && !expansion.empty()) {
			// Round 99: its shader's newer expansions wait for it (GetComputeProgram), as they
			// do while it compiles: rt-race98 translated eight ever larger ones on the GPU
			// thread meanwhile, the largest for minutes.
			m_compute_waiting_expansions[compute_program.id] = {
			    hash, std::vector<uint32_t>(expansion.begin(), expansion.end())};
		}
		if (partition) {
			// Round 117: what StartWaitingPartitions needs to start it in a free slot.
			auto entry                      = std::make_unique<WaitingPartition>();
			entry->program                  = *input_info.stage.program;
			entry->input_info               = input_info;
			entry->input_info.stage.program = &entry->program;
			entry->module                   = compute_program.module;
			entry->words                    = compute_program.spirv_words;
			entry->asks                     = asks + (expansion.empty() ? 1u : 0u);
			entry->serial = wait_serial != 0 ? wait_serial : ++m_partition_wait_serial;
			m_partition_waiting[compute_program.id] = std::move(entry);
		}
		if (m_compute_wait_logged.size() < 4096 &&
		    m_compute_wait_logged.insert(compute_program.id).second) {
			LOGF("Compute pipeline CS=0x%016" PRIx64 " (%u SPIR-V words) waits to compile in the "
			     "background: %s\n",
			     hash, compute_program.spirv_words,
			     same ? "another expansion of it compiles" : "the other compiles run");
		}
		return nullptr;
	}
	StartBackgroundCompute(compute_program.id, std::move(cached), input_info,
	                       *input_info.stage.program, compute_program.module,
	                       compute_program.spirv_words, partition, expansion,
	                       asks + (expansion.empty() ? 1u : 0u), false);
	return nullptr;
}

void PipelineCache::StartBackgroundCompute(uint64_t id, std::unique_ptr<Pipeline> cached,
                                           const ShaderComputeInputInfo& input_info,
                                           const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                           vk::ShaderModule module, uint32_t words, bool partition,
                                           std::span<const uint32_t> expansion, uint64_t skipped,
                                           bool ahead) {
	const auto running            = m_compute_background.size();
	auto       job                = std::make_unique<BackgroundCompute>();
	job->pipeline                 = std::move(cached);
	job->program                  = program;
	job->input_info               = input_info;
	job->input_info.stage.program = &job->program;
	job->hash                     = program.shader_hash;
	job->words                    = words;
	job->begin                    = std::chrono::steady_clock::now();
	job->expansion.assign(expansion.begin(), expansion.end());
	job->serial    = ++m_expansion_serial;
	job->partition = partition;
	job->ahead     = ahead;
	// The dispatch that asked for it now is skipped too (and those while it waited).
	job->skipped = skipped;
	auto* raw    = job.get();
	if (partition) {
		LOGF("Compute pipeline CS=0x%016" PRIx64 " (%u SPIR-V words), a partition, is compiled in "
		     "the background (%zu compiling, %u slots, a partition takes half of one%s%s); until "
		     "then its partition's dispatch is skipped\n",
		     job->hash, words, running + 1u, ComputeBackgroundThreads(),
		     ahead ? "; started while it waited, after " : "",
		     ahead ? fmt::format("{} dispatches asked for it", skipped).c_str() : "");
	} else {
		LOGF("Compute pipeline CS=0x%016" PRIx64 " (%u SPIR-V words) is compiled in the "
		     "background (%zu of %u running); until then its shader runs an earlier expansion, if "
		     "one is ready\n",
		     job->hash, words, running + 1u, ComputeBackgroundThreads());
	}
	if (job->pipeline->disable_optimization) {
		LOGF("Compute pipeline CS=0x%016" PRIx64 " (%u SPIR-V words) is compiled without "
		     "optimization (KYTY_PARTITION_QUICK)\n",
		     job->hash, words);
	}
	m_background_started++;
	// The driver cache stays until Save, which waits for this thread first.
	raw->thread = std::thread([this, raw, module, cache = m_driver_cache] {
		KYTY_PROFILER_THREAD("ComputePipelineCompile");
		// Round 117: the game's threads first; up to 16 of these compile at once.
		Common::LowerThreadPriority();
		(void)CreatePipelineInternal(m_graphics, *raw->pipeline, raw->input_info, module, cache, {},
		                             ComputeBuild::Compile);
		raw->done.store(true, std::memory_order_release);
	});
	m_compute_background.emplace(id, std::move(job));
}

void PipelineCache::StartWaitingPartitions() {
	// After Save (the emulator closes), none starts unasked.
	if (m_partition_waiting.empty() || m_saved) {
		return;
	}
	const auto slots       = ComputeBackgroundThreads();
	uint32_t   used_halves = 0;
	for (const auto& [id, job]: m_compute_background) {
		(void)id;
		used_halves += job->partition ? 1u : 2u;
	}
	// One slot is left to a dispatch of a shader that is not a partition (a whole expansion,
	// which waits otherwise), when there are four or more.
	const uint32_t limit = 2u * slots - (slots >= 4u ? 2u : 0u);
	while (used_halves + 1u <= limit && !m_partition_waiting.empty()) {
		auto best = m_partition_waiting.begin();
		for (auto it = m_partition_waiting.begin(); it != m_partition_waiting.end(); ++it) {
			if (it->second->asks > best->second->asks ||
			    (it->second->asks == best->second->asks &&
			     it->second->serial < best->second->serial)) {
				best = it;
			}
		}
		const auto id    = best->first;
		auto       entry = std::move(best->second);
		m_partition_waiting.erase(best);
		auto layouts = m_compute_waiting.find(id);
		if (layouts == m_compute_waiting.end()) {
			continue;
		}
		auto cached = std::move(layouts->second);
		m_compute_waiting.erase(layouts);
		StartBackgroundCompute(id, std::move(cached), entry->input_info, entry->program,
		                       entry->module, entry->words, true, {}, entry->asks, true);
		used_halves += 1u;
	}
}

// The minutes and seconds of a span, as "4 min 10 s" or "25 s".
static std::string MinutesAndSeconds(std::chrono::steady_clock::duration span) {
	const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(span).count();
	if (seconds < 60) {
		return fmt::format("{} s", seconds);
	}
	return fmt::format("{} min {} s", seconds / 60, seconds % 60);
}

void PipelineCache::ReportBackgroundProgress() {
	const size_t compiling = m_compute_background.size();
	const size_t waiting   = m_compute_waiting.size();
	const auto   now       = std::chrono::steady_clock::now();
	// What waits and is not a partition starts at its next dispatch; with nothing compiling, no
	// dispatch asks for it now.
	if (compiling != 0 || !m_partition_waiting.empty()) {
		if (!m_backlog) {
			m_backlog             = true;
			m_backlog_reported    = false;
			m_backlog_begin       = now;
			m_backlog_last_report = now;
			m_backlog_started_at  = m_background_started - compiling;
			m_backlog_finished_at = m_background_finished;
		}
		if (now - m_backlog_last_report < std::chrono::seconds(10)) {
			return;
		}
		m_backlog_last_report = now;
		m_backlog_reported    = true;
		const auto done       = m_background_finished - m_backlog_finished_at;
		uint64_t   total = 0, available = 0, commit_limit = 0, commit_free = 0;
		const bool physical = Common::VirtualMemory::QueryPhysical(total, available);
		const bool commit   = Common::VirtualMemory::QueryCommit(commit_limit, commit_free);
		const auto memory =
		    physical ? fmt::format("; memory: {:.1f} GB free of {:.1f} GB",
		                           static_cast<double>(available) / double(uint64_t {1} << 30u),
		                           static_cast<double>(total) / double(uint64_t {1} << 30u))
		             : std::string();
		::printf("Shaders compiling in the background: %zu now, %zu waiting, %" PRIu64
		         " done (%s so far%s). Until they are done, parts of the picture can be missing or "
		         "wrong (ray tracing: blotches, white glow).\n",
		         compiling, waiting, done, MinutesAndSeconds(now - m_backlog_begin).c_str(),
		         memory.c_str());
		std::fflush(stdout);
		LOGF("Background compiles: %zu compiling, %zu waiting (%zu of them partitions), %" PRIu64
		     " done in %.0f s%s%s\n",
		     compiling, waiting, m_partition_waiting.size(), done,
		     std::chrono::duration<double>(now - m_backlog_begin).count(), memory.c_str(),
		     commit ? fmt::format("; commit: {} MB free of {} MB", commit_free >> 20u,
		                          commit_limit >> 20u)
		                  .c_str()
		            : "");
		return;
	}
	if (!m_backlog) {
		return;
	}
	m_backlog       = false;
	const auto done = m_background_finished - m_backlog_finished_at;
	if (m_backlog_reported) {
		::printf("Shaders compiling in the background: all done (%" PRIu64 " in %s).\n", done,
		         MinutesAndSeconds(now - m_backlog_begin).c_str());
		std::fflush(stdout);
		LOGF("Background compiles: all done, %" PRIu64 " in %.0f s (%zu others wait for a "
		     "dispatch)\n",
		     done, std::chrono::duration<double>(now - m_backlog_begin).count(), waiting);
	}
	m_backlog_reported = false;
}

size_t PipelineCache::FinishBackgroundCompute(std::chrono::milliseconds budget) {
	if (m_compute_background.empty()) {
		return 0;
	}
	const auto deadline = std::chrono::steady_clock::now() + budget;
	size_t     waiting  = 0;
	for (const auto& [id, job]: m_compute_background) {
		(void)id;
		waiting += job->done.load(std::memory_order_acquire) ? 0u : 1u;
	}
	if (waiting != 0) {
		const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(budget).count();
		::printf("Closing: waiting up to %lld s for %zu shader pipeline(s) still compiling in the "
		         "background, to keep them for the next run...\n",
		         static_cast<long long>(seconds), waiting);
		LOGF("Compute pipelines: %zu still compile in the background; waiting up to %lld s\n",
		     waiting, static_cast<long long>(seconds));
	}
	for (auto it = m_compute_background.begin(); it != m_compute_background.end();) {
		auto& job = *it->second;
		while (!job.done.load(std::memory_order_acquire) &&
		       std::chrono::steady_clock::now() < deadline) {
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
		if (!job.done.load(std::memory_order_acquire)) {
			++it;
			continue;
		}
		job.thread.join();
		m_compute_pipelines.emplace(it->first, std::move(job.pipeline));
		it = m_compute_background.erase(it);
	}
	if (!m_compute_background.empty()) {
		::printf("Closing: %zu shader pipeline(s) still compiling are not kept; they compile again "
		         "next run\n",
		         m_compute_background.size());
		LOGF("Compute pipelines: %zu still compiling are not kept (they compile again next run)\n",
		     m_compute_background.size());
	}
	return m_compute_background.size();
}
} // namespace Libs::Graphics
