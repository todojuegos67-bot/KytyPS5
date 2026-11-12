#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/gpuPhases.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/drawRepeats.h"
#include "graphics/host_gpu/renderer/frameGraph.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/rendererLock.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

// KYTY_LOG_SKIPPED_DISPATCHES=<n>: log the first n dispatches of compute shaders that gave up,
// with their user data, to see which guest buffers they would have written.
static bool EarlyDispatchSkip() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EARLY_DISPATCH_SKIP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// Round 94: a dispatch whose pipeline compiles in the background (TryGetComputePipeline), or
// (round 105) one the driver could not build.
static void SkipDispatchWhileCompiling(uint64_t shader_hash) {
	if (FrameGraph::Logging()) {
		FrameGraph::Log("  not run: cs %016" PRIx64
		                " has no pipeline (it compiles in the background, or the driver could not "
		                "build it)",
		                shader_hash);
	}
}

// gt7-nort-1112, KYTY_NO_RT (on by default; KYTY_NO_RT=0 runs ray tracing as before): the
// dispatches of compute shaders that intersect rays with a BVH (IMAGE_BVH_INTERSECT_RAY) are not
// run, and their pipelines are never compiled. Their hit lists stay empty, so the material passes
// that shade the hits have no lanes to shade and report no call targets: no new table call
// expansions. The picture is the one of the dispatches skipped while compiling (reflections
// without ray-traced content). GPU only: nothing reaches the game's memory beyond what the pass
// would have written, and no save data is touched.
static bool NoRayTracing() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NO_RT");
		const bool  on    = value == nullptr || value[0] != '0';
		LOGF("Ray tracing: %s (KYTY_NO_RT=%s)\n",
		     on ? "off, BVH dispatches are not run" : "on", on ? "1, =0 turns it on" : "0");
		return on;
	}();
	return enabled;
}

static bool SkipRayTracingDispatch(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	if (!program.info.uses_bvh || !NoRayTracing()) {
		return false;
	}
	static std::atomic<uint64_t> skipped {0};
	const auto                   n = skipped.fetch_add(1, std::memory_order_relaxed);
	if (n == 0) {
		LOGF("Ray tracing: first BVH dispatch not run (cs %016" PRIx64 ", KYTY_NO_RT)\n",
		     program.shader_hash);
	}
	if (FrameGraph::Logging()) {
		FrameGraph::Log("  not run: KYTY_NO_RT leaves out cs %016" PRIx64, program.shader_hash);
	}
	return true;
}

static void LogSkippedDispatch(const HW::ComputeShaderInfo& cs, const char* groups) {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_LOG_SKIPPED_DISPATCHES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	static std::atomic<uint32_t> used {0};
	if (budget == 0 || used.fetch_add(1, std::memory_order_relaxed) >= budget) {
		return;
	}
	std::string user_data;
	for (uint32_t i = 0; i < cs.cs_user_sgpr.count && i < HW::UserSgprInfo::SGPRS_MAX; i++) {
		char word[12];
		std::snprintf(word, sizeof(word), " %08x", cs.cs_user_sgpr.value[i]);
		user_data += word;
	}
	LOGF("Skipped dispatch: hash=0x%016" PRIx64 " groups=%s user_data=%s\n",
	     ShaderDeclaredHash(cs.cs_regs.data_addr), groups, user_data.c_str());
}
namespace {

// A host GPU zone for a dispatch (gpuProfiler.h), named by its shader.
int64_t BeginGpuZone(RenderContext& context, CommandBuffer& buffer, uint64_t hash) {
	if (!Profiler::g_counters) {
		return -1;
	}
	char      name[32];
	const int size = std::snprintf(name, sizeof(name), "CS %016" PRIx64, hash);
	Profiler::CommandZone::Annotate("CS %016" PRIx64, hash);
	return GpuProfiler::Begin(context, buffer, name, static_cast<size_t>(size), 0xf28e2b);
}

// The rollback below drains the GPU twice per dispatch and almost never fires once a shader's
// pages are resident (one retry in a whole run, against ~2500 protected dispatches a frame). A
// shader is trusted after trust_runs clean attempts and dispatched without it; any fault that
// unprotected work reports re-arms every shader. KYTY_DISPATCH_RECOVERY=always keeps it on every
// dispatch; KYTY_DISPATCH_RECOVERY_TRUST=<n> sets the clean runs needed.
struct DispatchRecoveryPolicy {
	std::unordered_map<uint64_t, uint32_t> clean_runs;
	uint64_t                               seen_faults          = 0;
	uint32_t                               trust_runs           = 4;
	bool                                   always               = false;
	uint64_t                               protected_dispatches = 0;
	uint64_t                               trusted_dispatches   = 0;
	uint64_t                               retries              = 0;
	uint64_t                               rearms               = 0;

	void Count(bool trusted) {
		(trusted ? trusted_dispatches : protected_dispatches)++;
		if (((protected_dispatches + trusted_dispatches) & 0x3fff) == 0) {
			LOGF("DispatchRecovery: protected=%" PRIu64 " trusted=%" PRIu64 " retries=%" PRIu64
			     " rearms=%" PRIu64 " shaders=%zu\n",
			     protected_dispatches, trusted_dispatches, retries, rearms, clean_runs.size());
		}
	}
};

// KYTY_TRACE_DISPATCHES=<hash>[,<hash>...]: every dispatch of these compute shaders (a hash of 16
// hex digits, or its first 8) is logged as it is recorded, "Trace dispatch <n> ...": its frame,
// groups, user data and the words its user-data pointers point at (its constants and
// descriptors; KYTY_TRACE_PTR_BYTES, default 256), its textures with their cache states, and its
// buffers (the words of the small ones it reads). Once the GPU has run it: mip 0 of each float
// image it writes, up to KYTY_TRACE_SCAN_KB (default 1024), a glyph a layer with the texels that
// are not finite, and the same for the images the GPU wrote that it reads (once a frame each); the
// first words of each buffer it writes. A texture it reads that the GPU did not write is compared
// with its guest bytes (TextureCache::TraceVerify: once each, up to KYTY_TRACE_VERIFY_MB, default
// 2048). Up to KYTY_TRACE_MAX (default 20000) dispatches. KYTY_TRACE_FROM=<hash>: the trace starts
// at that shader's first dispatch (a shader that runs only where the trace is wanted).
//
// KYTY_TRACE_NAN=1 instead follows the texels that are not finite (infinity or NaN) through the
// traced dispatches, for passes of tens of thousands of them (a light-probe bake): after each one,
// every mip of the float images it writes, and of the ones it reads once their contents changed,
// is scanned on the GPU (TextureCache::ScanNonFiniteGpu). A layer of a mip that is not finite now
// and was not at the image's last scan, or the other way round, is logged "Trace nan: dispatch
// <n> ...". Only the first KYTY_TRACE_FIRST (default 3) dispatches of each shader are logged as
// they are recorded; the others' lines are kept until their scans land and logged for the first
// KYTY_TRACE_NAN_DESCRIBE (default 64) dispatches whose scans found a change. Up to
// KYTY_TRACE_NAN_MAX (default 4000) change lines; KYTY_TRACE_MAX defaults to 1000000, and the
// pointers' words to 1024 bytes.
struct ShaderHashes {
	std::vector<uint64_t> hashes;
	std::vector<uint32_t> prefixes; // the first 8 hex digits: the high 32 bits

	[[nodiscard]] bool Empty() const { return hashes.empty() && prefixes.empty(); }
	[[nodiscard]] bool Contains(uint64_t hash) const {
		return std::find(hashes.begin(), hashes.end(), hash) != hashes.end() ||
		       std::find(prefixes.begin(), prefixes.end(), static_cast<uint32_t>(hash >> 32u)) !=
		           prefixes.end();
	}
	// "<hash>[,<hash>...]", each of 16 hex digits or the first 8.
	[[nodiscard]] static ShaderHashes Parse(const char* value) {
		ShaderHashes      result;
		const std::string text = value != nullptr ? value : "";
		for (size_t begin = 0; begin < text.size();) {
			const auto end   = std::min(text.find(',', begin), text.size());
			auto       token = text.substr(begin, end - begin);
			begin            = end + 1u;
			std::erase_if(token, [](char c) { return c == ' ' || c == '\t'; });
			if (token.starts_with("0x") || token.starts_with("0X")) {
				token = token.substr(2);
			}
			if (token.empty() || token.size() > 16) {
				continue;
			}
			char*      stop = nullptr;
			const auto hash = std::strtoull(token.c_str(), &stop, 16);
			if (stop == nullptr || *stop != '\0') {
				continue;
			}
			if (token.size() <= 8) {
				result.prefixes.push_back(static_cast<uint32_t>(hash));
			} else {
				result.hashes.push_back(hash);
			}
		}
		return result;
	}
};

struct DispatchTraceConfig {
	ShaderHashes traced;
	ShaderHashes from;
	uint64_t     max_dispatches = 20000;
	uint64_t     scan_bytes     = 1024ull << 10u;
	uint64_t     pointer_bytes  = 256;
	bool         nan            = false;
	uint32_t     first          = 3;
	uint32_t     describe       = 64;
	uint64_t     max_changes    = 4000;
};

[[nodiscard]] uint64_t EnvironmentNumber(const char* name, uint64_t fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : fallback;
}

// Leaked, as NanTrace(): the scans' callbacks may still run while the process exits.
const DispatchTraceConfig& DispatchTrace() {
	static const auto* config = new DispatchTraceConfig([] {
		DispatchTraceConfig result;
		result.traced = ShaderHashes::Parse(std::getenv("KYTY_TRACE_DISPATCHES"));
		if (result.traced.Empty()) {
			return result;
		}
		result.from = ShaderHashes::Parse(std::getenv("KYTY_TRACE_FROM"));
		const char* nan = std::getenv("KYTY_TRACE_NAN");
		result.nan      = nan != nullptr && nan[0] == '1';
		result.max_dispatches =
		    EnvironmentNumber("KYTY_TRACE_MAX", result.nan ? 1000000u : result.max_dispatches);
		result.scan_bytes    = EnvironmentNumber("KYTY_TRACE_SCAN_KB", 1024u) << 10u;
		result.pointer_bytes = std::clamp<uint64_t>(
		    EnvironmentNumber("KYTY_TRACE_PTR_BYTES", result.nan ? 1024u : 256u), 4u, 65536u);
		result.first    = static_cast<uint32_t>(EnvironmentNumber("KYTY_TRACE_FIRST", 3u));
		result.describe = static_cast<uint32_t>(EnvironmentNumber("KYTY_TRACE_NAN_DESCRIBE", 64u));
		result.max_changes = EnvironmentNumber("KYTY_TRACE_NAN_MAX", 4000u);
		LOGF("Trace dispatch: tracing the dispatches of %zu shaders (KYTY_TRACE_DISPATCHES)%s, up "
		     "to %" PRIu64 " dispatches (KYTY_TRACE_MAX); %s\n",
		     result.traced.hashes.size() + result.traced.prefixes.size(),
		     result.from.Empty() ? "" : " from the first dispatch of KYTY_TRACE_FROM's shader",
		     result.max_dispatches,
		     result.nan ? fmt::format("following texels that are not finite on the GPU "
		                              "(KYTY_TRACE_NAN), the first {} dispatches of each shader "
		                              "logged as they are recorded (KYTY_TRACE_FIRST)",
		                              result.first)
		                      .c_str()
		                : fmt::format("images up to {} KB are scanned (KYTY_TRACE_SCAN_KB)",
		                              result.scan_bytes >> 10u)
		                      .c_str());
		return result;
	}());
	return *config;
}

// GPU thread: whether this dispatch is traced (KYTY_TRACE_FROM's shader starts the trace).
[[nodiscard]] bool TracedShader(uint64_t hash) {
	const auto& config = DispatchTrace();
	if (config.traced.Empty()) {
		return false;
	}
	static bool started = config.from.Empty();
	if (!started) {
		if (!config.from.Contains(hash)) {
			return false;
		}
		started = true;
		LOGF("Trace dispatch: the trace starts at this dispatch of cs %016" PRIx64
		     " (KYTY_TRACE_FROM)\n",
		     hash);
	}
	return config.traced.Contains(hash);
}

// Where a traced dispatch's lines go: the log, or (KYTY_TRACE_NAN) the dispatch's description,
// kept until its scans land.
struct TraceLines {
	std::vector<std::string>* kept = nullptr;

	template <typename... Args>
	void operator()(const char* format, const Args&... args) const {
		if (kept != nullptr) {
			kept->push_back(fmt::sprintf(format, args...));
		} else if (!Log::IsSilent()) {
			Log::Write(fmt::sprintf(format, args...));
		}
	}
};

void LogTraceWords(const TraceLines& out, const std::string& prefix, const uint32_t* words,
                   size_t count) {
	for (size_t row = 0; row < count; row += 16) {
		std::string line;
		for (size_t k = row; k < std::min(count, row + 16); k++) {
			line += fmt::format(" {:08x}", words[k]);
		}
		out("%s +0x%03zx:%s\n", prefix.c_str(), row * 4u, line.c_str());
	}
}

// A texture the GPU wrote that a traced dispatch reads: scanned once a frame (many dispatches
// read the same ones). GPU thread.
[[nodiscard]] bool ClaimInputScan(uint64_t address, uint64_t frame) {
	static std::unordered_map<uint64_t, uint64_t> scanned;
	const auto [found, inserted] = scanned.emplace(address, frame);
	if (!inserted && found->second == frame) {
		return false;
	}
	found->second = frame;
	return true;
}

// The compute shader whose dispatch the GPU thread records (TextureCache::s_trace_shader).
struct TraceShaderScope {
	explicit TraceShaderScope(uint64_t hash) { TextureCache::s_trace_shader = hash; }
	~TraceShaderScope() { TextureCache::s_trace_shader = 0; }
	KYTY_CLASS_NO_COPY(TraceShaderScope);
};

// KYTY_TRACE_NAN: a traced dispatch, its lines kept until its scans have landed (none when they
// were logged as it was recorded).
struct NanTraceDispatch {
	uint64_t                 serial = 0;
	uint64_t                 shader = 0;
	uint64_t                 frame  = 0;
	std::vector<std::string> lines;
	bool                     logged = false; // the priority thread's
};

// The last scan of each image, by guest address, shape and format, and what was logged (priority
// thread).
struct NanTraceState {
	using Key = std::tuple<uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, int>;
	struct Scanned {
		uint64_t              serial = 0; // the dispatch the scan followed
		std::vector<uint32_t> counts;
	};
	std::mutex             mutex;
	std::map<Key, Scanned> images;
	uint64_t               changes   = 0;
	uint32_t               described = 0;
};

NanTraceState& NanTrace() {
	static auto* state = new NanTraceState;
	return *state;
}

// " mip <m> layers <runs> (<words>/<of> words at the first)" for each mip with a layer `pick`
// selects, up to `most` runs a mip.
template <typename Pick>
[[nodiscard]] std::string NanMipLayers(const TextureCache::NonFiniteCounts& counts, size_t most,
                                       Pick&& pick, uint32_t& total) {
	std::string text;
	total = 0;
	for (uint32_t mip = 0; mip < counts.mips; mip++) {
		std::string runs;
		size_t      run_count = 0;
		uint32_t    first     = UINT32_MAX;
		for (uint32_t layer = 0; layer < counts.layers;) {
			if (!pick(mip, layer)) {
				layer++;
				continue;
			}
			uint32_t last = layer;
			while (last + 1u < counts.layers && pick(mip, last + 1u)) {
				last++;
			}
			total += last - layer + 1u;
			first = std::min(first, layer);
			if (run_count++ < most) {
				runs +=
				    last == layer ? fmt::format(" {}", layer) : fmt::format(" {}-{}", layer, last);
			}
			layer = last + 1u;
		}
		if (run_count == 0) {
			continue;
		}
		if (run_count > most) {
			runs += fmt::format(" (and {} more runs)", run_count - most);
		}
		text += fmt::format("{} mip {} layers{} ({}/{} words at layer {})", text.empty() ? "" : ";",
		                    mip, runs, counts.Count(mip, first), counts.words[mip], first);
	}
	return text;
}

// A scan of an image a traced dispatch wrote or read has landed (priority thread): what changed
// since the image's last scan.
void NanScanLanded(const std::shared_ptr<NanTraceDispatch>& dispatch, const std::string& what,
                   TextureCache::NonFiniteCounts&& counts) {
	const auto&      config = DispatchTrace();
	auto&            state  = NanTrace();
	std::scoped_lock lock {state.mutex};
	auto&      last  = state.images[NanTraceState::Key {counts.address, counts.width, counts.height,
                                                  counts.layers, counts.mips, counts.format}];
	const bool known = last.serial != 0 && last.counts.size() == counts.counts.size();
	bool       changed = !known;
	for (size_t k = 0; !changed && k < counts.counts.size(); k++) {
		changed = (counts.counts[k] != 0) != (last.counts[k] != 0);
	}
	const auto keep = [&] {
		last.serial = dispatch->serial;
		last.counts = std::move(counts.counts);
	};
	if (!changed) {
		keep();
		return;
	}
	const bool shown = state.changes++ < config.max_changes;
	if (state.changes == config.max_changes + 1u) {
		LOGF("Trace nan: the change lines stop here (KYTY_TRACE_NAN_MAX=%" PRIu64 ")\n",
		     config.max_changes);
	}
	if (!shown) {
		keep();
		return;
	}
	const auto header = fmt::format(
	    "Trace nan: dispatch {} (cs {:016x}, frame {}) {} 0x{:012x} {}x{}x{} mips {} fmt {} "
	    "(guest {})",
	    dispatch->serial, dispatch->shader, dispatch->frame, what, counts.address, counts.width,
	    counts.height, counts.layers, counts.mips, counts.format, counts.guest_format);
	const uint32_t mip_layers = counts.layers * counts.mips;
	uint32_t       now        = 0;
	for (const auto count: counts.counts) {
		now += count != 0 ? 1u : 0u;
	}
	const auto not_finite = [&](uint32_t mip, uint32_t layer) {
		return counts.Count(mip, layer) != 0;
	};
	if (!known) {
		// The first scan of the image: what it held before the trace could see it change.
		uint32_t   listed = 0;
		const auto layers = NanMipLayers(counts, 16, not_finite, listed);
		LOGF("%s: the first scan of it: %u of %u mip-layers hold words that are not finite%s%s\n",
		     header.c_str(), now, mip_layers, now != 0 ? ":" : "", layers.c_str());
	} else {
		const auto was_not_finite = [&](uint32_t mip, uint32_t layer) {
			return last.counts[static_cast<size_t>(mip) * counts.layers + layer] != 0;
		};
		const auto became_not_finite = [&](uint32_t mip, uint32_t layer) {
			return not_finite(mip, layer) && !was_not_finite(mip, layer);
		};
		const auto became_finite = [&](uint32_t mip, uint32_t layer) {
			return !not_finite(mip, layer) && was_not_finite(mip, layer);
		};
		uint32_t   became_count  = 0;
		uint32_t   cleared_count = 0;
		const auto became        = NanMipLayers(counts, 16, became_not_finite, became_count);
		const auto cleared       = NanMipLayers(counts, 16, became_finite, cleared_count);
		LOGF("%s: since the scan after dispatch %" PRIu64 "%s%s%s%s; %u of %u mip-layers hold "
		     "words that are not finite now\n",
		     header.c_str(), last.serial, became_count != 0 ? ", not finite now:" : "",
		     became.c_str(), cleared_count != 0 ? ", finite again:" : "", cleared.c_str(), now,
		     mip_layers);
	}
	if (!dispatch->logged && !dispatch->lines.empty() && state.described < config.describe) {
		dispatch->logged = true;
		state.described++;
		LOGF("Trace nan: dispatch %" PRIu64 " as it was recorded (%u of %u described, "
		     "KYTY_TRACE_NAN_DESCRIBE):\n",
		     dispatch->serial, state.described, config.describe);
		for (const auto& line: dispatch->lines) {
			if (!Log::IsSilent()) {
				Log::Write(line);
			}
		}
	}
	keep();
}

// KYTY_DRAW_REPEATS=1 (round 113, DrawRepeats): how much of each frame's dispatches the frame
// before had too.
void NoteDispatchRepeat(const ShaderStageRuntime& stage, const ShaderComputeInputInfo& input_info,
                        uint64_t image_generation) {
	if (!DrawRepeats::Enabled() || !stage) {
		return;
	}
	static DrawRepeats::Stats stats("Dispatch");
	const auto                mix = [](uint64_t seed, const void* data, size_t size) {
        return size == 0 ? seed : XXH3_64bits_withSeed(data, size, seed);
	};
	const auto  hash   = stage.program->shader_hash;
	const auto& snap   = *stage.resources;
	uint64_t    inputs = mix(0x5eed, &hash, sizeof(hash));
	inputs             = mix(inputs, snap.user_data.data(), snap.user_data.size() * 4u);
	uint64_t textures  = mix(0x7e47, &hash, sizeof(hash));
	textures           = mix(textures, snap.images.data(),
	                         snap.images.size() * sizeof(ShaderRecompiler::IR::DescriptorValue));
	textures           = mix(textures, snap.samplers.data(),
	                         snap.samplers.size() * sizeof(ShaderRecompiler::IR::DescriptorValue));
	uint64_t resolved  = mix(textures, snap.buffers.data(),
	                         snap.buffers.size() * sizeof(ShaderRecompiler::IR::DescriptorValue));
	resolved           = mix(resolved, snap.flattened_srt.data(), snap.flattened_srt.size() * 4u);
	resolved = mix(resolved, input_info.workgroup_counts, sizeof(input_info.workgroup_counts));
	resolved ^= inputs * 0x9e3779b97f4a7c15ull;
	stats.Note(FrameGraph::GpuThreadFrames(), inputs, resolved, textures, image_generation);
}

// KYTY_TRACE_NAN: records the GPU scans of a traced dispatch's float images (GPU thread).
void ScanDispatchImages(RenderContext& context, const ShaderStageRuntime& stage,
                        const PreparedBindings&                  bindings,
                        const std::shared_ptr<NanTraceDispatch>& dispatch) {
	// The content serial each image was last scanned at, and the addresses that cannot be
	// scanned (said once).
	static std::unordered_map<ImageId, uint64_t> scanned;
	static std::unordered_set<uint64_t>          unscanned;

	const auto&                 program  = *stage.program;
	auto&                       textures = context.GetTextureCache();
	std::unordered_set<ImageId> done;
	// The images it writes first: one bound twice is scanned once, as written.
	for (const bool outputs: {true, false}) {
		for (size_t i = 0; i < bindings.images.size() && i < program.info.images.size(); i++) {
			const auto& binding = bindings.images[i];
			const auto& info    = binding.desc.info;
			if (program.info.images[i].written != outputs || !binding.image_id ||
			    info.data.Empty() || done.contains(binding.image_id)) {
				continue;
			}
			const auto serial = textures.ContentSerial(binding.image_id);
			const auto last   = scanned.find(binding.image_id);
			if (!outputs && last != scanned.end() && last->second == serial) {
				continue;
			}
			done.insert(binding.image_id);
			const auto what = fmt::format("tex[{}] {}", i, outputs ? "write" : "read");
			if (!textures.ScanNonFiniteGpu(
			        binding.image_id, [dispatch, what](TextureCache::NonFiniteCounts&& counts) {
				        NanScanLanded(dispatch, what, std::move(counts));
			        })) {
				if (unscanned.insert(info.data.address).second) {
					LOGF("Trace nan: dispatch %" PRIu64 " tex[%zu] 0x%012" PRIx64
					     " fmt %d is not scanned (not a float format of whole words, a volume, "
					     "or over KYTY_GPU_SCAN_MB); said once\n",
					     dispatch->serial, i, info.data.address,
					     static_cast<int>(info.pixel_format));
				}
				continue;
			}
			scanned[binding.image_id] = serial;
		}
	}
}

void TraceDispatch(RenderContext& context, const ShaderStageRuntime& stage,
                   const PreparedBindings& bindings, const std::string& groups) {
	static std::atomic<uint64_t> traced {0};
	const auto&                  config = DispatchTrace();
	const auto                   n      = ++traced;
	if (n > config.max_dispatches) {
		if (n == config.max_dispatches + 1u) {
			LOGF("Trace dispatch: the trace stops here (KYTY_TRACE_MAX=%" PRIu64 ")\n",
			     config.max_dispatches);
		}
		return;
	}
	const auto& program  = *stage.program;
	const auto& snapshot = *stage.resources;
	auto&       textures = context.GetTextureCache();
	auto&       buffers  = context.GetBufferCache();
	const auto  frame    = context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	// KYTY_TRACE_NAN logs the first dispatches of each shader as they are recorded, and keeps the
	// others' lines (GPU thread).
	static std::unordered_map<uint64_t, uint32_t> logged_per_shader;
	const bool full = !config.nan || logged_per_shader[program.shader_hash]++ < config.first;
	std::shared_ptr<NanTraceDispatch> dispatch;
	if (config.nan) {
		dispatch         = std::make_shared<NanTraceDispatch>();
		dispatch->serial = n;
		dispatch->shader = program.shader_hash;
		dispatch->frame  = frame;
	}
	const TraceLines out {full ? nullptr : &dispatch->lines};
	out("Trace dispatch %" PRIu64 " frame %" PRIu64 " cs %016" PRIx64
	    " groups %s: %zu textures, %zu buffers, %zu user data\n",
	    n, frame, program.shader_hash, groups.c_str(), program.info.images.size(),
	    program.info.buffers.size(), snapshot.user_data.size());
	const auto& user = snapshot.user_data;
	for (size_t row = 0; row < user.size(); row += 16) {
		std::string line;
		for (size_t k = row; k < std::min(user.size(), row + 16); k++) {
			line += fmt::format(" {:08x}", user[k]);
		}
		out("Trace dispatch %" PRIu64 " user s%zu:%s\n", n, program.user_data_base + row,
		    line.c_str());
	}
	// What the user-data pointers point at: the shader's constants and descriptors, as guest
	// memory holds them.
	for (size_t reg = 0; reg + 1 < user.size(); reg += 2) {
		const uint64_t address = user[reg] | (static_cast<uint64_t>(user[reg + 1]) << 32u);
		if ((user[reg] & 3u) != 0 || user[reg + 1] == 0 || user[reg + 1] > 0xffu) {
			continue;
		}
		const auto bytes =
		    Libs::LibKernel::Memory::TryClampRangeSize(address, config.pointer_bytes) &
		    ~uint64_t {3};
		if (bytes == 0) {
			continue;
		}
		const auto prefix =
		    fmt::format("Trace dispatch {} ptr s[{}:{}] 0x{:012x}", n, program.user_data_base + reg,
		                program.user_data_base + reg + 1u, address);
		if (buffers.HasGpuDirtyBytes(address, bytes)) {
			out("%s: the GPU wrote these bytes (not shown)\n", prefix.c_str());
			continue;
		}
		std::vector<uint32_t> words(bytes / sizeof(uint32_t));
		if (Libs::LibKernel::Memory::TryReadBacking(address, words.data(), bytes)) {
			LogTraceWords(out, prefix, words.data(), words.size());
		}
	}
	for (size_t i = 0; i < bindings.images.size() && i < program.info.images.size(); i++) {
		const auto& binding  = bindings.images[i];
		const auto& resource = program.info.images[i];
		const auto& info     = binding.desc.info;
		const auto& view     = binding.desc.view_info;
		const auto  state    = binding.image_id ? textures.DiagnosticImageState(binding.image_id)
		                                        : TextureCache::DiagnosticState {};
		out("Trace dispatch %" PRIu64 " tex[%zu] %s%s 0x%012" PRIx64 "+0x%" PRIx64
		    " %ux%ux%u layers %u mips %u view %u+%u/%u+%u fmt %d guest %u tile %u [%c%c%c]\n",
		    n, i, resource.written ? "write" : "read",
		    binding.desc.type == TextureCache::BindingType::Storage ? " storage" : "",
		    info.data.address, info.data.size, info.extent.width, info.extent.height,
		    info.extent.depth, info.resources.layers, info.resources.levels, view.base_level,
		    view.level_count, view.base_layer, view.layer_count,
		    static_cast<int>(info.pixel_format), static_cast<uint32_t>(info.guest_format),
		    static_cast<uint32_t>(info.tile_mode), state.gpu_written ? 'G' : '-',
		    state.buffer_write ? 'B' : '-', state.cpu_dirty ? 'C' : '-');
		if (!binding.image_id || info.data.Empty()) {
			continue;
		}
		const auto label = fmt::format("Trace dispatch {} tex[{}]", n, i);
		if (config.nan) {
			// The GPU scans below; a texture the GPU did not write is still compared with its
			// guest bytes (once each).
			if (!resource.written && !state.gpu_written && !state.buffer_write) {
				textures.TraceVerify(binding.image_id, fmt::format("Trace dispatch {}", n),
				                     fmt::format("tex[{}]", i));
			}
		} else if (resource.written) {
			// One that is not scanned says so once (its address).
			static std::unordered_set<uint64_t> unscanned;
			if (!textures.TraceScanImage(binding.image_id, label + " out", config.scan_bytes) &&
			    unscanned.insert(info.data.address).second) {
				LOGF("%s out 0x%012" PRIx64 ": not scanned (over KYTY_TRACE_SCAN_KB=%" PRIu64
				     ", or a format the scan does not read); said once\n",
				     label.c_str(), info.data.address, config.scan_bytes >> 10u);
			}
		} else if (state.gpu_written || state.buffer_write) {
			if (ClaimInputScan(info.data.address, frame)) {
				(void)textures.TraceScanImage(binding.image_id, label + " in", config.scan_bytes);
			}
		} else {
			textures.TraceVerify(binding.image_id, fmt::format("Trace dispatch {}", n),
			                     fmt::format("tex[{}]", i));
		}
	}
	for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
		const auto     r = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
		const uint64_t bytes =
		    std::max<uint64_t>(r.Stride(), 1u) * static_cast<uint64_t>(r.NumRecords());
		const uint64_t mapped  = r.Base48() >= BufferCache::CACHING_PAGESIZE && bytes != 0
		                             ? Libs::LibKernel::Memory::TryClampRangeSize(r.Base48(), bytes)
		                             : 0;
		const bool     gpu     = mapped != 0 && buffers.HasGpuDirtyBytes(r.Base48(), mapped);
		const bool     written = program.info.buffers[i].written;
		out("Trace dispatch %" PRIu64 " buf[%zu] %s 0x%012" PRIx64 "+0x%" PRIx64
		    " (mapped 0x%" PRIx64 ") stride %u records %u fmt %u [%c]\n",
		    n, i, written ? "write" : "read", r.Base48(), bytes, mapped, r.Stride(), r.NumRecords(),
		    r.RawFormat(), gpu ? 'G' : '-');
		if (mapped == 0) {
			continue;
		}
		const auto prefix = fmt::format("Trace dispatch {} buf[{}]", n, i);
		if (written) {
			if (full) {
				buffers.TraceContents(r.Base48(), mapped, prefix + " out");
			}
		} else if (!gpu && mapped <= 1024) {
			std::vector<uint32_t> words(mapped / sizeof(uint32_t));
			if (!words.empty() && Libs::LibKernel::Memory::TryReadBacking(
			                          r.Base48(), words.data(), words.size() * sizeof(uint32_t))) {
				LogTraceWords(out, prefix, words.data(), words.size());
			}
		}
	}
	if (config.nan) {
		ScanDispatchImages(context, stage, bindings, dispatch);
	}
}

// Dispatches are recorded on the GPU thread only.
DispatchRecoveryPolicy& RecoveryPolicy() {
	static DispatchRecoveryPolicy policy = [] {
		DispatchRecoveryPolicy p;
		if (const char* value = std::getenv("KYTY_DISPATCH_RECOVERY")) {
			p.always = std::strcmp(value, "always") == 0;
		}
		if (const char* value = std::getenv("KYTY_DISPATCH_RECOVERY_TRUST")) {
			p.trust_runs = static_cast<uint32_t>(std::strtoul(value, nullptr, 0));
		}
		return p;
	}();
	return policy;
}

// A missing BDA page returns zero during the first attempt. Preserve every
// externally writable buffer, including atomic destinations, so that attempt
// can be rolled back before any dependent dispatch observes its results.
// A ray tracing shader is left out: its traversal finds the BVH a level per attempt, up to
// MaxPageFaults pages each, so a scene the GPU has not touched yet would drain the GPU dozens
// of times in one dispatch and could pass the 128 attempts Retry allows (the emulator exits
// there). It runs once instead, as one writing images does: a ray that reaches a missing node
// misses for that frame, and the pages it faulted on are mapped after the submission.
class DispatchBufferRecovery {
public:
	DispatchBufferRecovery(RenderContext& context, const PreparedBindings& bindings,
	                       uint64_t shader_address)
	    : m_context(context), m_shader_address(shader_address) {
		const auto& program = *bindings.runtime->program;
		const bool  writes_images =
		    std::any_of(program.info.images.begin(), program.info.images.end(),
		                [](const auto& image) { return image.written; });
		m_enabled = program.info.uses_dma && !program.info.uses_bvh && !writes_images;
		if (!m_enabled) return;
		auto& cache     = context.GetBufferCache();
		auto& scheduler = context.GetCommandScheduler();
		auto& policy    = RecoveryPolicy();
		if (!policy.always) {
			if (const auto faults = cache.UnattributedFaults(); faults != policy.seen_faults) {
				policy.seen_faults = faults;
				policy.clean_runs.clear();
				policy.rearms++;
			}
			if (policy.clean_runs[shader_address] >= policy.trust_runs) {
				m_enabled = false;
				policy.Count(true);
				return;
			}
		}
		policy.Count(false);
		// Reports from earlier work cannot be attributed to this transaction.
		cache.ProcessFaultBuffer();
		scheduler.Finish();
		RangeSet ranges;
		// buffer_sources is indexed by memory slot (FindBuffers), not by program.info.buffers:
		// map each slot to its resource. Indexing it by resource read past the slots and
		// handed FindBuffer garbage ranges (the GT7 Cafe "invalid buffer discovery request").
		for (size_t i = 0; i < bindings.buffer_sources.size(); ++i) {
			const auto  resource = program.bindings.descriptors.front().resources[i];
			const auto& source   = bindings.buffer_sources[i];
			if (resource >= program.info.buffers.size() || !program.info.buffers[resource].written ||
			    source.size == 0) {
				continue;
			}
			const auto start = source.address & ~uint64_t {3};
			const auto end   = (source.address + source.size + 3) & ~uint64_t {3};
			if (!GuestRange {start, end - start}.Valid()) {
				static std::atomic<uint32_t> reported = 0;
				if (reported.fetch_add(1) < 8) {
					LOGF("DispatchBufferRecovery: temporary: skipping invalid written range "
					     "0x%016" PRIx64 " size 0x%016" PRIx64 " (slot %zu)\n",
					     source.address, source.size, i);
				}
				continue;
			}
			ranges.Add(start, end - start);
		}
		ranges.ForEach([&](uint64_t start, uint64_t end) {
			const auto id = cache.FindBuffer(start, end - start);
			cache.SynchronizeBuffersInRange(start, end - start);
			const auto& source = cache.GetBuffer(id);
			Save(source, source.Offset(start), start, end - start, false);
		});
		if (std::any_of(program.bindings.descriptors.begin(), program.bindings.descriptors.end(),
		                [](const auto& binding) {
			                return binding.kind == ShaderRecompiler::IR::DescriptorBindingKind::Gds;
		                })) {
			const auto& source = *cache.GetGdsBuffer();
			Save(source, 0, 0, source.Size(), true);
		}
	}

	// Whether this dispatch runs as a transaction that collects its own faults.
	[[nodiscard]] bool Enabled() const { return m_enabled; }

	bool Retry() {
		if (!m_enabled) return false;
		auto&      cache  = m_context.GetBufferCache();
		const auto report = cache.CollectFaults();
		auto&      policy = RecoveryPolicy();
		if (report.page_count == 0) {
			if (report.trap.claimed != 0) {
				const auto hash =
				    (uint64_t {report.trap.shader_hash_high} << 32) | report.trap.shader_hash_low;
				EXIT("GPU shader trap after page residency: hash=0x%016" PRIx64
				     " pc=0x%08x code=0x%02x\n",
				     hash, report.trap.pc, report.trap.code);
			}
			if (m_attempts == 0) {
				policy.clean_runs[m_shader_address]++;
			}
			return false;
		}
		policy.clean_runs[m_shader_address] = 0;
		policy.retries++;
		if (++m_attempts > 128) {
			EXIT("GPU shader paging did not converge: shader=0x%016" PRIx64 " pages=%" PRIu64 "\n",
			     m_shader_address, report.page_count);
		}
		static uint32_t reports = 0;
		if (reports++ < 32) {
			::printf("GPU dispatch retry: shader=0x%016" PRIx64 " attempt=%u missing_pages=%" PRIu64
			         "\n",
			         m_shader_address, m_attempts, report.page_count);
		}
		for (const auto& saved: m_saved) {
			if (saved.gds) {
				cache.GetGdsBuffer()->CopyFrom(m_context.GetCommandScheduler().Current(),
				                               *saved.buffer, 0, 0, saved.buffer->Size());
			} else {
				// Resolving a fault can merge owners; never retain a native destination
				// handle or BufferId across CollectFaults().
				auto [destination, offset] =
				    cache.ObtainBuffer(saved.address, saved.buffer->Size(), true);
				destination->CopyFrom(m_context.GetCommandScheduler().Current(), *saved.buffer, 0,
				                      offset, saved.buffer->Size());
			}
		}
		m_context.PrepareBda();
		return true;
	}

private:
	void Save(const Buffer& source, uint64_t offset, uint64_t address, uint64_t size, bool gds) {
		auto saved =
		    std::make_unique<Buffer>(m_context.GetGraphics(), m_context.GetCommandScheduler(),
		                             MemoryUsage::DeviceLocal, 0, AllFlags, size);
		saved->CopyFrom(m_context.GetCommandScheduler().Current(), source, offset, 0, size);
		m_saved.push_back({address, gds, std::move(saved)});
	}
	struct Saved {
		uint64_t                address;
		bool                    gds;
		std::unique_ptr<Buffer> buffer;
	};
	RenderContext&     m_context;
	uint64_t           m_shader_address;
	bool               m_enabled  = false;
	uint32_t           m_attempts = 0;
	std::vector<Saved> m_saved;
};

} // namespace

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

// The SDK's buffer fill shader. With 64 threads a group, invocation i stores pattern[i & 3] (user
// data s4..s7) to element i of the V# in s[0:3] while i < s8, with buffer_store_format_x. The
// recompiler's uniform fill does not describe it (each invocation picks its word from a VGPR
// array with v_movrels), so its fills ran as buffer writes. A buffer write to a multisampled
// stencil plane never reaches the host's stencil (TextureCache::RefreshImage): the course maps'
// stencil kept what the frames before left in it, and their track, drawn with a stencil test that
// lets each pixel be drawn once, was not drawn at all. A buffer write over a render target makes
// its next use upload the written bytes over it.
constexpr std::array<uint32_t, 23> SdkPatternFillCode {
    0xbfa00001u,              // s_inst_prefetch 0x1
    0xd7460004u, 0x04010c09u, // v_lshl_add_u32 v4, s9, 6, v0
    0x7e000204u,              // v_mov_b32 v0, s4
    0x7e020205u,              // v_mov_b32 v1, s5
    0x7e040206u,              // v_mov_b32 v2, s6
    0x7e060207u,              // v_mov_b32 v3, s7
    0x7da80808u,              // v_cmpx_gt_u32 s8, v4
    0xbf88000du,              // s_cbranch_execz (to s_endpgm)
    0x360a0883u,              // v_and_b32 v5, 3, v4
    0xbe84047eu,              // s_mov_b64 s[4:5], exec
    0xbe86047eu,              // s_mov_b64 s[6:7], exec
    0x7e100505u,              // v_readfirstlane_b32 s8, v5
    0x7da40a08u,              // v_cmpx_eq_u32 s8, v5
    0xbefc0308u,              // s_mov_b32 m0, s8
    0x7e0c8700u,              // v_movrels_b32 v6, v0
    0x8a867e06u,              // s_andn2_b64 s[6:7], s[6:7], exec
    0xbefe0406u,              // s_mov_b64 exec, s[6:7]
    0xbf85fff9u,              // s_cbranch_scc1 (to v_readfirstlane_b32)
    0xbefe0404u,              // s_mov_b64 exec, s[4:5]
    0xe0102000u, 0x80000604u, // buffer_store_format_x v6, v4, s[0:3], 0 idxen
    0xbf810000u,              // s_endpgm
};

// KYTY_PATTERN_FILLS=0 (or the F8 step "patternfill:off"): the SDK's pattern fills run as
// dispatches, as before round 66.
static bool PatternFillsOn() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PATTERN_FILLS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && !Bisect::PatternFillsOff();
}

// Whether the guest code at `code_address` (the program `shader_hash` was made from) is the SDK's
// pattern fill: compared once a program.
static bool IsSdkPatternFillCode(uint64_t shader_hash, uint64_t code_address) {
	static std::mutex                         mutex;
	static std::unordered_map<uint64_t, bool> checked;
	std::scoped_lock                          lock(mutex);
	if (const auto found = checked.find(shader_hash); found != checked.end()) {
		return found->second;
	}
	constexpr uint64_t Bytes = sizeof(SdkPatternFillCode);
	const bool         match = ShaderCodeSizeBytes(code_address) >= Bytes &&
	                   Libs::LibKernel::Memory::TryClampRangeSize(code_address, Bytes) == Bytes &&
	                   std::memcmp(reinterpret_cast<const void*>(code_address),
	                               SdkPatternFillCode.data(), Bytes) == 0;
	if (checked.size() < 65536) {
		checked.emplace(shader_hash, match);
	}
	if (match) {
		LOGF("Dispatch: cs %016" PRIx64 " at 0x%012" PRIx64
		     " is the SDK's pattern fill: a fill of one word clears the image whose memory it "
		     "fills (KYTY_PATTERN_FILLS=0 runs it as a dispatch)\n",
		     shader_hash, code_address);
	}
	return match;
}

namespace {

struct SdkPatternFill {
	ShaderBufferResource    descriptor; // the V# its store writes through (s[0:3])
	std::array<uint32_t, 4> pattern {}; // s4..s7
	uint32_t                count = 0;  // s8: the elements it stores at most
};

// A dispatch fill of one 32-bit word: the recompiler's uniform fill, or the SDK's pattern fill.
struct DispatchFill {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint32_t value   = 0;
	bool     pattern = false; // the SDK's pattern fill
};

} // namespace

// The dispatch's program as the SDK's pattern fill, with its V#, pattern and count; none when it
// is another program (or KYTY_PATTERN_FILLS=0).
static std::optional<SdkPatternFill> FindSdkPatternFill(const ShaderComputeInputInfo& input,
                                                        uint64_t code_address) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	// Its one buffer is stored to and never read; s9 is the group's index (after 9 user SGPRs).
	if (code_address == 0 || program.info.buffers.size() != 1 || resources.buffers.size() != 1 ||
	    !program.info.buffers[0].written || program.info.buffers[0].read ||
	    program.info.buffers[0].atomic || !program.info.images.empty() ||
	    program.user_data_base != 0 || resources.user_data.size() < 9 ||
	    input.workgroup_register != 9 || !input.group_id[0] || !PatternFillsOn() ||
	    !IsSdkPatternFillCode(program.shader_hash, code_address)) {
		return std::nullopt;
	}
	SdkPatternFill fill;
	fill.descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[0]);
	std::copy_n(resources.user_data.begin() + 4, 4, fill.pattern.begin());
	fill.count = resources.user_data[8];
	// The V# bound for its store is the one in s[0:3].
	ShaderBufferResource user {};
	std::copy_n(resources.user_data.begin(), 4, user.fields);
	if (user.Base48() != fill.descriptor.Base48() || user.Stride() != fill.descriptor.Stride() ||
	    user.NumRecords() != fill.descriptor.NumRecords() ||
	    user.RawFormat() != fill.descriptor.RawFormat()) {
		return std::nullopt;
	}
	return fill;
}

// The word an SDK pattern fill writes throughout, when its four pattern words are the same.
static std::optional<uint32_t> PatternFillWord(const SdkPatternFill& fill) {
	const auto& pattern = fill.pattern;
	if (pattern[1] != pattern[0] || pattern[2] != pattern[0] || pattern[3] != pattern[0]) {
		return std::nullopt;
	}
	return pattern[0];
}

// The memory an SDK pattern fill of one word writes, as ResolveComputeBufferFill resolves the
// recompiler's uniform fills: its first min(s8, invocations, records) dwords.
static bool ResolveSdkPatternFill(const ShaderComputeInputInfo& input, uint64_t code_address,
                                  uint32_t group_x, uint32_t group_y, uint32_t group_z,
                                  uint32_t mode, DispatchFill& resolved, const char** why) {
	const auto fail = [why](const char* reason) {
		if (why != nullptr) {
			*why = reason;
		}
		return false;
	};
	const auto fill = FindSdkPatternFill(input, code_address);
	if (!fill) {
		return false;
	}
	const auto word = PatternFillWord(*fill);
	if (!word) {
		return fail("the SDK's pattern fill of more than one word");
	}
	const auto& descriptor = fill->descriptor;
	const auto  format     = descriptor.Format();
	// Bounds checked by index (OOB_SELECT 0 or 1), so its records bound the elements it stores.
	if (descriptor.Stride() != 4 || descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 ||
	    descriptor.AddTid() || descriptor.Base48() == 0 || descriptor.OutOfBounds() > 1 ||
	    descriptor.Type() != 0 ||
	    (format != Prospero::BufferFormat::k32UInt && format != Prospero::BufferFormat::k32SInt &&
	     format != Prospero::BufferFormat::k32Float)) {
		return fail("the SDK's pattern fill through a V# that is not a plain array of dwords");
	}
	if (input.threads_num[0] != 64 || input.threads_num[1] != 1 || input.threads_num[2] != 1 ||
	    input.dispatch_thread_dimensions || mode != 0x41u || group_x == 0 || group_y != 1 ||
	    group_z != 1) {
		return fail("the SDK's pattern fill in a dispatch of another shape");
	}
	const uint64_t elements = std::min(
	    {uint64_t {fill->count}, uint64_t {group_x} * 64u, uint64_t {descriptor.NumRecords()}});
	if (elements == 0 || elements * 4u > UINT32_MAX ||
	    !GuestRange {descriptor.Base48(), elements * 4u}.Valid()) {
		return fail("the SDK's pattern fill of no memory, of too much, or outside guest memory");
	}
	resolved = {descriptor.Base48(), elements * 4u, *word, true};
	return true;
}

// The fill of one word a dispatch is, if any: the recompiler's uniform fill, or else the SDK's
// pattern fill (its code at `code_address`; 0 leaves that out). `why`: as for
// ResolveComputeBufferFill.
static bool ResolveDispatchFill(const ShaderComputeInputInfo& input, uint64_t code_address,
                                uint32_t group_x, uint32_t group_y, uint32_t group_z, uint32_t mode,
                                DispatchFill& resolved, const char** why) {
	ShaderBufferResource descriptor;
	uint32_t             value = 0;
	uint64_t             size  = 0;
	if (ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, value, size,
	                             why)) {
		resolved = {descriptor.Base48(), size, value, false};
		return true;
	}
	return input.stage.resources->uniform_fill.kind ==
	           ShaderRecompiler::IR::UniformFillKind::None &&
	       ResolveSdkPatternFill(input, code_address, group_x, group_y, group_z, mode, resolved,
	                             why);
}

// What became of the SDK's pattern fills: a "Dispatch: SDK pattern fills" line every 10 s
// (while there are any) counts them. The GPU thread's (the renderer's lock held).
enum class PatternFillOutcome : uint32_t { Stencil, Image, Meta, MappedParts, Ran, Count };

static void NotePatternFill(PatternFillOutcome outcome) {
	static std::array<uint64_t, static_cast<size_t>(PatternFillOutcome::Count)> counts {};
	static auto since = std::chrono::steady_clock::now();
	counts[static_cast<size_t>(outcome)]++;
	const auto now     = std::chrono::steady_clock::now();
	const auto elapsed = std::chrono::duration<double>(now - since).count();
	if (elapsed < 10.0) {
		return;
	}
	LOGF("Dispatch: SDK pattern fills in %.0f s: %" PRIu64 " cleared a stencil plane, %" PRIu64
	     " another image, %" PRIu64 " metadata (an HTile, CMASK or DCC), %" PRIu64
	     " filled the mapped parts of memory with holes, %" PRIu64 " ran as dispatches\n",
	     elapsed, counts[0], counts[1], counts[2], counts[3], counts[4]);
	counts = {};
	since  = now;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer& buffer, uint64_t code_address,
                                                uint32_t group_x, uint32_t group_y,
                                                uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		// An SDK pattern fill of one word, whose value an HTile it covers whole takes, as from the
		// recompiler's uniform fills (group_x 0: the dispatch is not known, no value).
		DispatchFill pattern_fill;
		const bool   pattern =
		    resources.uniform_fill.kind == ShaderRecompiler::IR::UniformFillKind::None &&
		    group_x != 0 &&
		    ResolveSdkPatternFill(input, code_address, group_x, group_y, group_z, mode,
		                          pattern_fill, nullptr);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				const auto base = descriptor.Base48();
				// The value a uniform fill of this buffer writes, which says what the HTile clears
				// (a depth target's stencil too, or not: TextureCache::MetaFillValue).
				const auto&             fill = resources.uniform_fill;
				std::optional<uint32_t> value;
				if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Buffer &&
				    fill.resource == i && fill.words == 1) {
					value = fill.value;
				} else if (pattern && i == 0 && pattern_fill.address == base) {
					const auto meta_size = cache.MetaSize(base);
					if (meta_size != 0 && pattern_fill.size >= meta_size) {
						value = pattern_fill.value;
					}
				}
				if (!cache.ClearMeta(base, value)) {
					continue;
				}
				if (!cache.IsColorMeta(base)) {
					if (pattern && i == 0) {
						NotePatternFill(PatternFillOutcome::Meta);
					}
					return true;
				}
				// A color target's CMASK or DCC starts here as well (memory reused within the
				// frame): the depth target's clear is noted and the fill still runs, so the color
				// target finds it in its metadata, as the hardware's memory would hold it for both.
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
					LOGF("Dispatch: a fill at 0x%012" PRIx64 " by cs %016" PRIx64
					     " clears a depth target's HTile and a color target's metadata there: it "
					     "still runs\n",
					     base, program.shader_hash);
				}
				return false;
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size, const char** why) {
	const auto fail = [why](const char* reason) {
		if (why != nullptr) {
			*why = reason;
		}
		return false;
	};
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return fail("not a uniform buffer fill (or its value is not known for this dispatch)");
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return fail("its V# is not a plain array of the stored elements");
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1) {
		return fail("its workgroup shape does not match the fill's index");
	}
	if (group_x == 0 || group_y != 1 || group_z != 1) {
		return fail("not a one-dimensional dispatch");
	}
	if (mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return fail("dispatch initiator bits");
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords()) {
		return fail("its invocations are not one per record of its V#");
	}
	if (size == 0 || size > UINT32_MAX) {
		return fail("its size");
	}
	if (input.dispatch_thread_dimensions &&
	    (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	     input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1)) {
		return fail("its thread counts are not whole workgroups");
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return fail("another of its buffers overlaps the destination");
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

// KYTY_SPARSE_FILLS=0: a compute fill over memory with holes runs as a dispatch, which writes
// only up to the first hole.
static bool SparseFills() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPARSE_FILLS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// A uniform fill of [base, base + size) whose mappings have holes: its mapped parts are filled on
// the GPU, as the hardware's writes land there (the rest it drops). False when the range has no
// hole (the dispatch runs) or too many pieces to fill one by one.
static bool FillMappedParts(RenderContext& context, uint64_t shader, uint64_t base, uint64_t size,
                            uint32_t value) {
	const auto prefix = Libs::LibKernel::Memory::TryClampRangeSize(base, size);
	if (prefix == size || (base & 3u) != 0 || (size & 3u) != 0) {
		return false;
	}
	constexpr size_t                           MaxSpans = 16384;
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	if (!Libs::LibKernel::Memory::CommittedSpans(base, size, MaxSpans, &spans)) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("GraphicsRenderDispatchDirect: buffer fill 0x%012" PRIx64 "+0x%" PRIx64
			     " by cs %016" PRIx64 " over more than %zu mapped pieces: run as a dispatch, "
			     "which stops at its first hole\n",
			     base, size, shader, MaxSpans);
		}
		return false;
	}
	uint64_t mapped = 0;
	for (const auto& span: spans) {
		mapped += span.second;
	}
	context.GetBufferCache().FillMappedSpans(spans, value);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: buffer fill 0x%012" PRIx64 "+0x%" PRIx64
		     " with 0x%08x by cs %016" PRIx64 " over memory with holes: its %zu mapped pieces "
		     "(0x%" PRIx64 " bytes) filled, the holes left alone (a dispatch stops at the "
		     "first, 0x%" PRIx64 " in)\n",
		     base, size, value, shader, spans.size(), mapped, prefix);
	}
	if (Libs::LibKernel::Memory::TracesMemory(base, size)) {
		Libs::LibKernel::Memory::TraceMemory(
		    "compute fill 0x%016" PRIx64 "+0x%" PRIx64 " with 0x%08x by cs %016" PRIx64
		    " over holes: %zu mapped pieces, 0x%" PRIx64 " bytes, filled on the GPU",
		    base, size, value, shader, spans.size(), mapped);
	}
	return true;
}

// Diagnostics: a compute shader's buffer write over memory with holes in its mappings (bound
// across them, SparseWriteSize in descriptors.cpp). Each such (shader, address) is logged once
// (the first 32) with the dispatch, its V# and user data, and why it was not taken as a fill.
// `groups`: the dispatch's group counts, or null for an indirect one (`args` holds its address).
static void ReportWritesOverHoles(const ShaderComputeInputInfo& input, const char* fill_why,
                                  const uint32_t* groups, uint64_t args, uint32_t mode) {
	// `seen`: the (shader, address) pairs checked, logged or not; `reports`: those logged.
	constexpr size_t                               MaxReports = 32;
	constexpr size_t                               MaxSeen    = 4096;
	static std::atomic<bool>                       full {false};
	static std::mutex                              mutex;
	static std::set<std::pair<uint64_t, uint64_t>> seen;
	static size_t                                  reports = 0;
	if (full.load(std::memory_order_relaxed)) {
		return;
	}
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	for (uint32_t i = 0; i < program.info.buffers.size() && i < resources.buffers.size(); i++) {
		if (!program.info.buffers[i].written) {
			continue;
		}
		const auto     r     = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		const uint64_t bytes = r.GetSize();
		if (r.Base48() < BufferCache::CACHING_PAGESIZE || bytes == 0 ||
		    bytes > UINT64_MAX - r.Base48()) {
			continue;
		}
		const auto mapped = Libs::LibKernel::Memory::TryClampRangeSize(r.Base48(), bytes);
		if (mapped == bytes) {
			continue;
		}
		std::lock_guard lock(mutex);
		if (!seen.insert({program.shader_hash, r.Base48()}).second) {
			continue;
		}
		if (seen.size() >= MaxSeen) {
			full.store(true, std::memory_order_relaxed);
		}
		// Anything mapped past the first hole (the binding loses only that)?
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		const bool few = Libs::LibKernel::Memory::CommittedSpans(r.Base48(), bytes, 2, &spans);
		if (few && (spans.empty() || (spans.size() == 1 && spans[0].first == r.Base48()))) {
			continue;
		}
		if (++reports >= MaxReports) {
			full.store(true, std::memory_order_relaxed);
		}
		const auto shape =
		    groups != nullptr
		        ? fmt::format("groups {}x{}x{} mode 0x{:x}", groups[0], groups[1], groups[2], mode)
		        : fmt::format("indirect args 0x{:x} mode 0x{:x}", args, mode);
		std::string user;
		for (size_t k = 0; k < resources.user_data.size() && k < 16; k++) {
			user += fmt::format(" {:08x}", resources.user_data[k]);
		}
		const auto& fill = resources.uniform_fill;
		LOGF("GraphicsRenderDispatch: cs %016" PRIx64 " %s threads %ux%ux%u%s writes buf[%u] "
		     "0x%012" PRIx64 "+0x%" PRIx64 " over memory with holes (its first 0x%" PRIx64
		     " bytes mapped, more mapped pieces past them); V# [%08x %08x %08x %08x] stride %u "
		     "records %u fmt %u; user data%s; uniform fill: kind %u words %u value 0x%08x; not "
		     "taken as a fill: %s\n",
		     program.shader_hash, shape.c_str(), input.threads_num[0], input.threads_num[1],
		     input.threads_num[2], input.dispatch_thread_dimensions ? " (thread counts)" : "", i,
		     r.Base48(), bytes, mapped, r.fields[0], r.fields[1], r.fields[2], r.fields[3],
		     r.Stride(), r.NumRecords(), r.RawFormat(), user.c_str(),
		     static_cast<uint32_t>(fill.kind), fill.words, fill.value,
		     fill_why != nullptr ? fill_why : "(not checked)");
	}
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                 CommandBuffer& command, uint32_t group_x,
                                                 uint32_t group_y, uint32_t group_z, uint32_t mode,
                                                 uint64_t code_address) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX)
				return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1])
			return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue                  clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	DispatchFill buffer_fill;
	if (!ResolveDispatchFill(input, code_address, group_x, group_y, group_z, mode, buffer_fill,
	                         nullptr)) {
		return false;
	}
	// An SDK pattern fill clears an image only when no other image starts at its address (the
	// dispatch's buffer write gives those the written bytes too), and its word is the guest
	// memory's. What it cleared, or why not, is logged once per address (the first 64).
	BufferFillClear how;
	std::string     what;
	bool            also_run = false;
	if (buffer_fill.pattern) {
		how = {.alone = true, .also_run = &also_run, .guest_word = true, .what = &what};
	}
	const bool cleared = cache.ClearImageFromBuffer(command, buffer_fill.address, buffer_fill.size,
	                                                buffer_fill.value, how);
	// (Fills that find no image are not logged: most fill metadata, an FMASK or a buffer.)
	if (buffer_fill.pattern && (cleared || !what.starts_with("no image"))) {
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint64_t> logged_patterns;
		std::lock_guard                     lock(logged_mutex);
		if (logged_patterns.size() < 64 &&
		    logged_patterns.insert(buffer_fill.address | (cleared ? 1u : 0u)).second) {
			LOGF("Dispatch: the SDK's pattern fill 0x%012" PRIx64 "+0x%" PRIx64
			     " with 0x%08x %s%s\n",
			     buffer_fill.address, buffer_fill.size, buffer_fill.value,
			     cleared ? "clears " : "runs as a dispatch: ", what.c_str());
		}
	}
	if (!cleared) {
		return false;
	}
	if (buffer_fill.pattern) {
		NotePatternFill(what.starts_with("the stencil") ? PatternFillOutcome::Stencil
		                                                : PatternFillOutcome::Image);
		// A stencil plane no buffer write reaches is cleared, and the fill still runs for another
		// image at its address.
		return !also_run;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, buffer_fill.address, buffer_fill.size,
		     buffer_fill.value);
	}
	return true;
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (ShaderRecompiler::IR::FindBinding(input.stage.program->bindings,
	        ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory) == nullptr) {
		return;
	}
	auto& cache = context.GetBufferCache();
	if (indirect_args != 0) {
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

// Round 118: the target of the partition this dispatch runs, for the expansion's compares of
// the call's target (GetPartitionTarget): the same code serves the material at any address.
static void FillPartitionTarget(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                const PipelineCache::ComputePartition*          partition,
                                PreparedBindings&                               bindings) {
	const auto dword = program.bindings.partition_target_dword;
	if (dword == ShaderRecompiler::IR::PushData::NoStart ||
	    dword + 2u > bindings.shader_data.size()) {
		return;
	}
	const uint64_t target            = partition != nullptr ? partition->built : 0;
	bindings.shader_data[dword]      = static_cast<uint32_t>(target);
	bindings.shader_data[dword + 1u] = static_cast<uint32_t>(target >> 32u);
}

// Round 99: a shader whose call is partitioned (ShaderCallPartition) runs once per partition,
// each partition's lanes in its own dispatch; the first one also tells which partitions there are.
// A partition whose program translates or compiles is skipped meanwhile (its lanes keep what the
// last dispatch wrote there), as a dispatch is.
void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode) {
	PipelineCache::ComputePartition partition;
	DispatchDirectPart(submit_id, buffer, thread_group_x, thread_group_y, thread_group_z, mode,
	                   &partition);
	const auto targets = partition.targets;
	for (size_t i = 1; i < targets.size(); i++) {
		partition.target = targets[i];
		DispatchDirectPart(submit_id, buffer, thread_group_x, thread_group_y, thread_group_z, mode,
		                   &partition);
	}
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
                                      uint32_t mode) {
	PipelineCache::ComputePartition partition;
	DispatchIndirectPart(submit_id, buffer, args_addr, mode, &partition);
	const auto targets = partition.targets;
	for (size_t i = 1; i < targets.size(); i++) {
		partition.target = targets[i];
		DispatchIndirectPart(submit_id, buffer, args_addr, mode, &partition);
	}
}

void RenderExecutor::DispatchDirectPart(uint64_t submit_id, CommandBuffer& buffer,
                                        uint32_t thread_group_x, uint32_t thread_group_y,
                                        uint32_t thread_group_z, uint32_t mode,
                                        PipelineCache::ComputePartition* partition) {
	const GpuPhases::Scope dispatch_phase(GpuPhases::Phase::Dispatch);
	const GpuPhases::OpStats op_stats(true);
	// KYTY_BISECT's slow:gpu (round 114).
	Bisect::SpinMicroseconds(Bisect::SlowGpuMicroseconds());
	EXIT_IF(buffer.IsInvalid());
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchPending);
		m_context.GetCommandScheduler().PopPendingOperations(false);
	}
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);
	if (FrameGraph::Logging()) {
		FrameGraph::BeginOp();
		FrameGraph::Log("dispatch groups=%ux%ux%u mode=0x%x cs at 0x%012" PRIx64 " queue %s",
		                thread_group_x, thread_group_y, thread_group_z, mode,
		                sh_ctx.GetCs().cs_regs.data_addr, FrameGraph::GuestQueueName().c_str());
	}

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	const RendererLock lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return;
	}
	// The skip list's ray-tracing dispatches are most of this title's dispatches in the jungle
	// (~2,600 skip-list lookups a frame); drop them before preparing their inputs and program.
	// KYTY_EARLY_DISPATCH_SKIP=0 drops them in ProgramCache::Get as before.
	if (EarlyDispatchSkip() && PipelineCache::IsSkipListed(sh_ctx.GetCs().cs_regs.data_addr)) {
		Profiler::Add(Profiler::Counter::SkippedSkipList);
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not run: skip-listed");
		}
		ResetBindings();
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions      = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	ShaderProgram compute_program;
	{
		KYTY_PROFILER_BLOCK("Dispatch::GetComputeProgram");
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchProgram);
		compute_program =
		    m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info, partition);
	}
	if (FrameGraph::Logging() && partition != nullptr && !partition->targets.empty()) {
		FrameGraph::Log("  partition for target 0x%012" PRIx64 " (%zu partitions)",
		                partition->target != 0 ? partition->target : partition->targets.front(),
		                partition->targets.size());
	}
	if (!compute_program) {
		char groups[48];
		std::snprintf(groups, sizeof(groups), "%ux%ux%u", thread_group_x, thread_group_y,
		              thread_group_z);
		LogSkippedDispatch(cs_regs, groups);
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not run: no program (%016" PRIx64 ")",
			                ShaderDeclaredHash(cs_regs.cs_regs.data_addr));
			FrameGraph::NoteShader("CS", ShaderDeclaredHash(cs_regs.cs_regs.data_addr),
			                       cs_regs.cs_regs.data_addr, cs_regs.cs_user_sgpr.value,
			                       cs_regs.cs_user_sgpr.count);
		}
		ResetBindings();
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0] = thread_group_x;
		input_info.dispatch_threads_num[1] = thread_group_y;
		input_info.dispatch_threads_num[2] = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	NoteDispatchRepeat(input_info.stage, input_info,
	                   m_context.GetTextureCache().ImageSetGeneration());
	if (GpuPhases::OpStatsOn()) {
		GpuPhases::SetOpKey(program.shader_hash, uint64_t {input_info.workgroup_counts[0]} *
		                                             input_info.workgroup_counts[1] *
		                                             input_info.workgroup_counts[2]);
	}
	if (SkipRayTracingDispatch(program)) {
		ResetBindings();
		return;
	}
	// KYTY_BISECT (bisect.h): a step that leaves out this shader's dispatches.
	if (Bisect::SkipsShader(program.shader_hash)) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not run: KYTY_BISECT leaves out cs %016" PRIx64,
			                program.shader_hash);
		}
		ResetBindings();
		return;
	}
	{
		KYTY_PROFILER_BLOCK("Dispatch::TryConsumeClears");
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchClears);
		if (resources.specialization_reads.empty() &&
		    (TryConsumeComputeMetaClear(input_info, buffer, cs_regs.cs_regs.data_addr,
		                                thread_group_x, thread_group_y, thread_group_z, mode) ||
		     TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
		                                 thread_group_z, mode, cs_regs.cs_regs.data_addr))) {
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  consumed as a clear: cs %016" PRIx64, program.shader_hash);
			}
			ResetBindings();
			return;
		}
	}
	// A buffer fill that found no image to clear. Over memory with holes in its mappings (sparse
	// memory, as the top-down depth maps of a track), the hardware writes the mapped pages and
	// drops the rest; a binding stops at the first hole, so its mapped pages after it are filled
	// one by one instead of running the dispatch (KYTY_SPARSE_FILLS=0 runs it as before).
	// Otherwise it runs as a buffer write, noted once recorded (its own write forgets older fills
	// there): a target created inside it later starts as the fill (ClearsTargetsFromFills).
	struct {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint32_t value   = 0;
	} fill_note;
	const char* fill_why = "its program is specialized on guest memory it reads";
	GpuPhases::Scope fill_phase(GpuPhases::Phase::DispatchFill);
	if (resources.specialization_reads.empty()) {
		DispatchFill       fill_found;
		constexpr uint64_t SmallestNoted = 64u * 1024u;
		const bool         fill =
		    ResolveDispatchFill(input_info, cs_regs.cs_regs.data_addr, thread_group_x,
		                        thread_group_y, thread_group_z, mode, fill_found, &fill_why);
		if (fill && SparseFills() &&
		    FillMappedParts(m_context, program.shader_hash, fill_found.address, fill_found.size,
		                    fill_found.value)) {
			if (fill_found.pattern) {
				NotePatternFill(PatternFillOutcome::MappedParts);
			}
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  consumed as a fill of the mapped parts of 0x%012" PRIx64
				                "+0x%" PRIx64 ": cs %016" PRIx64,
				                fill_found.address, fill_found.size, program.shader_hash);
			}
			ResetBindings();
			return;
		}
		if (fill && fill_found.pattern) {
			NotePatternFill(PatternFillOutcome::Ran);
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  the SDK's pattern fill of 0x%012" PRIx64 "+0x%" PRIx64
				                " with 0x%08x: it clears no image, it runs",
				                fill_found.address, fill_found.size, fill_found.value);
			}
		}
		if (fill && TextureCache::ClearsTargetsFromFills() && fill_found.size >= SmallestNoted) {
			fill_note = {fill_found.address, fill_found.size, fill_found.value};
		}
		if (fill) {
			fill_why = nullptr;
		}
	}
	if (fill_why != nullptr) {
		const uint32_t groups[] = {thread_group_x, thread_group_y, thread_group_z};
		ReportWritesOverHoles(input_info, fill_why, groups, 0, mode);
	}
	fill_phase.End();

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	buffer.EndRendering();
	GpuPhases::Scope pipeline_phase(GpuPhases::Phase::DispatchPipeline);
	auto*            pipeline_found =
	    m_context.GetPipelineCache().TryGetComputePipeline(input_info, compute_program);
	pipeline_phase.End();
	if (pipeline_found == nullptr) {
		SkipDispatchWhileCompiling(program.shader_hash);
		ResetBindings();
		return;
	}
	auto& pipeline = *pipeline_found;
	auto& bindings = m_compute_bindings;
	if (FrameGraph::Logging()) {
		FrameGraph::Log("  cs %016" PRIx64 " groups %ux%ux%u threads %ux%ux%u", program.shader_hash,
		                thread_group_x, thread_group_y, thread_group_z, input_info.threads_num[0],
		                input_info.threads_num[1], input_info.threads_num[2]);
		FrameGraph::NoteShader("CS", ShaderDeclaredHash(cs_regs.cs_regs.data_addr),
		                       cs_regs.cs_regs.data_addr, cs_regs.cs_user_sgpr.value,
		                       cs_regs.cs_user_sgpr.count);
		FrameGraph::SetShowBuffers(true);
	}
	const TraceShaderScope trace_shader(program.shader_hash);
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchTextures);
		PrepareBindings(input_info.stage, bindings);
	}
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	FillPartitionTarget(program, partition, bindings);
	if (program.info.uses_dma) {
		PrefaultDmaPointers(m_context.GetBufferCache(), input_info.stage);
	}
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchFind);
		FindBuffers(bindings);
	}
	if (program.info.uses_dma) {
		KYTY_PROFILER_BLOCK("Dispatch::PrepareBda");
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchBda);
		m_context.PrepareBda();
	}
	// Resolving one image binding can replace or free the image of an earlier one (an image
	// expanded or merged into a larger one): the image set's generation then moves, and the
	// bindings are made again below.
	const auto image_generation = m_context.GetTextureCache().ImageSetGeneration();
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchImages);
		RebindImages(bindings);
	}
	BindSharedMemory(m_context, input_info, bindings);
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchBuffers);
		RebindBuffers(bindings);
	}

	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	DispatchBufferRecovery recovery(m_context, bindings, cs_regs.cs_regs.data_addr);
	// The bindings made above are made again only when something may have changed them since: a
	// dispatch that runs as a transaction (the recovery's copies and its drain may merge buffers),
	// each retry, and the images when the image set changed while they were made. Before round 68
	// every dispatch made them twice (KYTY_BISECT's rebind2).
	bool rebind = recovery.Enabled() || Bisect::RebindTwice();
	bool rebind_images =
	    rebind || m_context.GetTextureCache().ImageSetGeneration() != image_generation;
	do {
		if (rebind_images) {
			const GpuPhases::Scope phase(GpuPhases::Phase::DispatchImages);
			RebindImages(bindings);
		}
		if (rebind) {
			const GpuPhases::Scope phase(GpuPhases::Phase::DispatchBuffers);
			RebindBuffers(bindings);
		}
		rebind                             = true;
		rebind_images                      = true;
		const auto        vk_buffer        = buffer.Recorder();
		PreparedBindings* descriptor_stage = &bindings;
		GpuPhases::Scope  commit_phase(GpuPhases::Phase::DispatchCommit);
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
		commit_phase.End();
		KYTY_PROFILER_BLOCK("Dispatch::Record");
		bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
		    HasShaderBufferWrites(input_info.stage);
		has_storage_writes =
		    std::any_of(program.info.images.begin(), program.info.images.end(),
		                [](const auto& image) {
			                return image.written &&
			                       image.resource_class ==
			                           ShaderRecompiler::IR::ImageResourceClass::Storage;
		                }) ||
		    has_storage_writes;
		// Round 166: with barriers:guest the guest's own sync points order dispatches instead.
		const bool guest_barriers = GuestGpu::GuestPm4Barriers();
		if (has_storage_writes && !guest_barriers) {
			// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
			// while allowing the queue to execute asynchronously.
			ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		// A checkpoint right before the dispatch: preparing its bindings may have recorded
		// transfers (and their checkpoints) after the first one.
		buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
		                    thread_group_x, thread_group_y, thread_group_z, mode,
		                    sh_ctx.GetCs().cs_regs.data_addr);
		// Device loss reports name the dispatch's shader (the CS hash in the VS slot).
		buffer.AnnotateDiagnosticCheckpoint(program.shader_hash, 0, 0, 0);
		GpuTiming::Before(m_context, buffer);
		const auto gpu_zone = BeginGpuZone(m_context, buffer, program.shader_hash);
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
		Profiler::Add(Profiler::Counter::Dispatches);
		Common::PerfStats::AddDispatch();
		m_context.GetCommandScheduler().NoteDispatch(
		    program.shader_hash, false,
		    uint64_t {thread_group_x} * uint64_t {thread_group_y} * uint64_t {thread_group_z});
		GpuProfiler::End(buffer, gpu_zone);
		GpuTiming::After(m_context, buffer, program.shader_hash, GpuTiming::Dispatch);
		Timeline::Mark("dispatch", program.shader_hash,
		               (static_cast<uint64_t>(thread_group_x) << 32u) |
		                   (thread_group_y << 16u) | thread_group_z);

		// The removed host fence also ordered read-only dispatches before later writers.
		if (!guest_barriers) {
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
	} while (recovery.Retry());
	// The dispatch's writes are in this command buffer now.
	m_context.GetBufferCache().WritesRecorded(program.has_address_writes, program.shader_hash);
	// Round 106, a capture: the chosen shaders' written buffers as the dispatch left them.
	FrameGraph::SnapshotWritten(m_context.GetBufferCache());
	if (TracedShader(program.shader_hash)) {
		TraceDispatch(m_context, input_info.stage, bindings,
		              fmt::format("{}x{}x{}", thread_group_x, thread_group_y, thread_group_z));
	}
	if (program.info.uses_dma && DmaFaultReport() && !recovery.Enabled()) {
		m_context.GetBufferCache().ProcessFaultBuffer();
	}
	if (fill_note.size != 0) {
		const auto mapped =
		    Libs::LibKernel::Memory::TryClampRangeSize(fill_note.address, fill_note.size);
		m_context.GetBufferCache().NoteFill(fill_note.address, fill_note.size, fill_note.value,
		                                    mapped, true);
		static std::atomic<uint32_t> logged {0};
		if (mapped != fill_note.size && logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: buffer fill 0x%012" PRIx64 "+0x%" PRIx64
			     " with 0x%08x by cs %016" PRIx64 " found no image, and 0x%" PRIx64
			     " bytes of it are mapped: a target created inside it starts as the fill\n",
			     fill_note.address, fill_note.size, fill_note.value, program.shader_hash, mapped);
		}
		if (Libs::LibKernel::Memory::TracesMemory(fill_note.address, fill_note.size)) {
			Libs::LibKernel::Memory::TraceMemory(
			    "compute fill 0x%016" PRIx64 "+0x%" PRIx64 " with 0x%08x by cs %016" PRIx64
			    " (no image there; 0x%" PRIx64 " bytes mapped): noted for targets made inside it",
			    fill_note.address, fill_note.size, fill_note.value, program.shader_hash, mapped);
		}
	}
	ResetBindings();
}

void RenderExecutor::DispatchIndirectPart(uint64_t submit_id, CommandBuffer& buffer,
                                          uint64_t args_addr, uint32_t mode,
                                          PipelineCache::ComputePartition* partition) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0);
	// The arguments are thread counts: IndirectDispatchGroups converts them on the GPU, and the
	// shader bounds its threads by the counts it reads from the same memory.
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	const GpuPhases::Scope dispatch_phase(GpuPhases::Phase::Dispatch);
	const GpuPhases::OpStats op_stats(true);
	// KYTY_BISECT's slow:gpu (round 114).
	Bisect::SpinMicroseconds(Bisect::SlowGpuMicroseconds());
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchPending);
		m_context.GetCommandScheduler().PopPendingOperations(false);
	}
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	const RendererLock lock(m_context.GetMutex());
	const auto&       cs_regs = buffer.GetShaders().GetCs();
	if (FrameGraph::Logging()) {
		FrameGraph::BeginOp();
		FrameGraph::Log(
		    "dispatch indirect args 0x%012" PRIx64 " mode=0x%x cs at 0x%012" PRIx64 " queue %s",
		    args_addr, mode, cs_regs.cs_regs.data_addr, FrameGraph::GuestQueueName().c_str());
		FrameGraph::SetShowBuffers(true);
		FrameGraph::NoteBuffer(args_addr, 12);
	}
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	// Empty dispatch whose arguments no GPU write can still change: nothing to prepare (no program
	// lookup, bindings or argument sync). Read from the backing only when no byte is GPU-dirty and
	// no command storing through shader addresses (which marks no range) is still on the GPU.
	if (vk::DispatchIndirectCommand skip_args {};
	    !m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(skip_args)) &&
	    !m_context.GetBufferCache().AddressWritesInFlight() &&
	    Libs::LibKernel::Memory::TryReadBacking(args_addr, &skip_args, sizeof(skip_args)) &&
	    (skip_args.x == 0 || skip_args.y == 0 || skip_args.z == 0)) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not run: empty arguments");
		}
		return;
	}
	if (EarlyDispatchSkip() && PipelineCache::IsSkipListed(cs_regs.cs_regs.data_addr)) {
		Profiler::Add(Profiler::Counter::SkippedSkipList);
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not run: skip-listed");
		}
		ResetBindings();
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.dispatch_thread_dimensions   = use_thread_dimensions;
	input_info.dispatch_dimensions_indirect = use_thread_dimensions;
	GpuPhases::Scope program_phase(GpuPhases::Phase::DispatchProgram);
	const auto       compute_program = m_context.GetPipelineCache().GetComputeProgram(
        cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info, partition);
	program_phase.End();
	if (FrameGraph::Logging() && partition != nullptr && !partition->targets.empty()) {
		FrameGraph::Log("  partition for target 0x%012" PRIx64 " (%zu partitions)",
		                partition->target != 0 ? partition->target : partition->targets.front(),
		                partition->targets.size());
	}
	if (!compute_program) {
		LogSkippedDispatch(cs_regs, "indirect");
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not run: no program (%016" PRIx64 ")",
			                ShaderDeclaredHash(cs_regs.cs_regs.data_addr));
			FrameGraph::NoteShader("CS", ShaderDeclaredHash(cs_regs.cs_regs.data_addr),
			                       cs_regs.cs_regs.data_addr, cs_regs.cs_user_sgpr.value,
			                       cs_regs.cs_user_sgpr.count);
		}
		ResetBindings();
		return;
	}
	// Indirect: its thread groups are on the GPU (counted as none).
	if (GpuPhases::OpStatsOn()) {
		GpuPhases::SetOpKey(input_info.stage.program->shader_hash, 0);
	}
	if (SkipRayTracingDispatch(*input_info.stage.program)) {
		ResetBindings();
		return;
	}
	// KYTY_BISECT (bisect.h): a step that leaves out this shader's dispatches.
	if (Bisect::SkipsShader(input_info.stage.program->shader_hash)) {
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  not run: KYTY_BISECT leaves out cs %016" PRIx64,
			                input_info.stage.program->shader_hash);
		}
		ResetBindings();
		return;
	}
	if (use_thread_dimensions && m_indirect_groups == nullptr) {
		m_indirect_groups = std::make_unique<IndirectDispatchGroups>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler());
	}
	buffer.EndRendering();
	GpuPhases::Scope pipeline_phase(GpuPhases::Phase::DispatchPipeline);
	auto*            pipeline_found =
	    m_context.GetPipelineCache().TryGetComputePipeline(input_info, compute_program);
	pipeline_phase.End();
	if (pipeline_found == nullptr) {
		SkipDispatchWhileCompiling(input_info.stage.program->shader_hash);
		ResetBindings();
		return;
	}
	auto& pipeline = *pipeline_found;
	auto& bindings = m_compute_bindings;
	if (FrameGraph::Logging()) {
		FrameGraph::Log("  cs %016" PRIx64 " threads %ux%ux%u",
		                input_info.stage.program->shader_hash, input_info.threads_num[0],
		                input_info.threads_num[1], input_info.threads_num[2]);
		FrameGraph::NoteShader("CS", ShaderDeclaredHash(cs_regs.cs_regs.data_addr),
		                       cs_regs.cs_regs.data_addr, cs_regs.cs_user_sgpr.value,
		                       cs_regs.cs_user_sgpr.count);
	}
	const TraceShaderScope trace_shader(input_info.stage.program->shader_hash);
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchTextures);
		PrepareBindings(input_info.stage, bindings);
	}
	const auto& program = *input_info.stage.program;
	FillPartitionTarget(program, partition, bindings);
	ReportWritesOverHoles(input_info, "an indirect dispatch is not checked for a fill", nullptr,
	                      args_addr, mode);
	if (program.info.uses_dma) {
		PrefaultDmaPointers(m_context.GetBufferCache(), input_info.stage);
	}
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchFind);
		FindBuffers(bindings);
	}
	if (program.info.uses_dma) {
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchBda);
		m_context.PrepareBda();
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchImages);
		RebindImages(bindings);
	}
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DispatchBuffers);
		RebindBuffers(bindings);
	}
	DispatchBufferRecovery recovery(m_context, bindings, cs_regs.cs_regs.data_addr);
	do {
		GpuPhases::Scope images_phase(GpuPhases::Phase::DispatchImages);
		RebindImages(bindings);
		images_phase.End();
		GpuPhases::Scope buffers_phase(GpuPhases::Phase::DispatchBuffers);
		const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
		    args_addr, sizeof(vk::DispatchIndirectCommand), false, false, {},
		    use_thread_dimensions);
		EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
		vk::Buffer     indirect_buffer = args_buffer->Handle();
		vk::DeviceSize indirect_offset = args_offset;
		if (use_thread_dimensions) {
			EXIT_IF(!args_buffer->HasDeviceAddress());
			const std::array<uint32_t, 3> local_size {
			    std::max(cs_regs.cs_regs.num_thread_x, 1u),
			    std::max(cs_regs.cs_regs.num_thread_y, 1u),
			    std::max(cs_regs.cs_regs.num_thread_z, 1u)};
			const auto converted = m_indirect_groups->Convert(
			    buffer.Recorder(), args_buffer->BufferDeviceAddress() + args_offset, local_size);
			indirect_buffer = converted.groups_buffer;
			indirect_offset = converted.groups_offset;
			// The shader reads the thread counts through this device address (upstream's
			// dispatch-thread words hold the address instead of the counts).
			const auto dword = program.bindings.dispatch_thread_dword;
			if (dword != ShaderRecompiler::IR::PushData::NoStart) {
				bindings.shader_data[dword]      = static_cast<uint32_t>(converted.threads);
				bindings.shader_data[dword + 1u] = static_cast<uint32_t>(converted.threads >> 32u);
			}
		}
		RebindBuffers(bindings);
		buffers_phase.End();
		PreparedBindings* descriptor_stage = &bindings;
		GpuPhases::Scope  commit_phase(GpuPhases::Phase::DispatchCommit);
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
		commit_phase.End();
		const auto vk_buffer = buffer.Recorder();
		const bool has_storage_writes =
		    bindings.shared_memory.buffer != nullptr || HasShaderBufferWrites(input_info.stage) ||
		    std::any_of(
		        program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			        return image.written && image.resource_class ==
			                                    ShaderRecompiler::IR::ImageResourceClass::Storage;
		        });
		const bool guest_barriers = GuestGpu::GuestPm4Barriers();
		if (has_storage_writes && !guest_barriers) {
			ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask =
		    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(
		    vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eComputeShader |
		        vk::PipelineStageFlagBits::eTransfer,
		    vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr, 0, nullptr);
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		// A checkpoint right before the dispatch: preparing its bindings may have recorded
		// transfers (and their checkpoints) after the first one.
		buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect),
		                    submit_id, static_cast<uint32_t>(args_addr),
		                    static_cast<uint32_t>(args_addr >> 32u), 0, mode,
		                    cs_regs.cs_regs.data_addr);
		// Device loss reports name the dispatch's shader (the CS hash in the VS slot).
		buffer.AnnotateDiagnosticCheckpoint(program.shader_hash, 0, 0, 0);
		GpuTiming::Before(m_context, buffer);
		const auto gpu_zone = BeginGpuZone(m_context, buffer, program.shader_hash);
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
		Profiler::Add(Profiler::Counter::Dispatches);
		Common::PerfStats::AddDispatch();
		m_context.GetCommandScheduler().NoteDispatch(program.shader_hash, true, 0);
		GpuProfiler::End(buffer, gpu_zone);
		GpuTiming::After(m_context, buffer, program.shader_hash, GpuTiming::Dispatch);
		Timeline::Mark("dispatch-indirect", program.shader_hash, 0);
		if (!guest_barriers) {
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
	} while (recovery.Retry());
	// The dispatch's writes are in this command buffer now.
	m_context.GetBufferCache().WritesRecorded(program.has_address_writes, program.shader_hash);
	// Round 106, a capture: the chosen shaders' written buffers as the dispatch left them.
	FrameGraph::SnapshotWritten(m_context.GetBufferCache());
	if (TracedShader(program.shader_hash)) {
		TraceDispatch(m_context, input_info.stage, bindings,
		              fmt::format("indirect (arguments at 0x{:012x})", args_addr));
	}
	if (program.info.uses_dma && DmaFaultReport() && !recovery.Enabled()) {
		m_context.GetBufferCache().ProcessFaultBuffer();
	}
	ResetBindings();
}

} // namespace Libs::Graphics
