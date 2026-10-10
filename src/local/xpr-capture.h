#ifndef EMULATOR_SRC_LOCAL_XPR_CAPTURE_H_
#define EMULATOR_SRC_LOCAL_XPR_CAPTURE_H_
// XPR draw capture for the offline replay tests (tests/XprReplayTests.cpp).
//
// An XPR draw is an indexed indirect draw whose arguments lie in a range an XPR
// culling dispatch writes; the renderer learns those ranges from the culling
// programs while a capture is pending. KYTY_XPR_CAPTURE=N writes N XPR draws,
// one of every KYTY_XPR_CAPTURE_STRIDE (default 1, so a capture can span all
// passes), one file each, to $KYTY_XPR_CAPTURE_DIR (default _Build/xpr-capture).
// With KYTY_XPR_CAPTURE_TRIGGER=<file> the capture starts once that file exists.
// A file holds, per stage, the compiler inputs (the shader-warmup Record
// encoding), user data and push-data start, the runtime resource snapshot, every
// guest range the SRT evaluation read with its contents, and the contents of
// every bound buffer range. Nothing is changed in rendering.
#include "kernel/memory.h"
#include "shader-warmup-cache.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Libs::Graphics::XprCapture {

inline void Words64(std::vector<uint32_t>& out, uint64_t value) {
	out.push_back(static_cast<uint32_t>(value));
	out.push_back(static_cast<uint32_t>(value >> 32u));
}

inline void Snapshot(std::vector<uint32_t>& out, const ShaderRecompiler::IR::ResourceSnapshot& s) {
	for (const auto* values: {&s.buffers, &s.images, &s.samplers}) {
		out.push_back(static_cast<uint32_t>(values->size()));
		for (const auto& value: *values) {
			out.push_back(value.dword_count);
			out.insert(out.end(), value.dwords.begin(), value.dwords.end());
		}
	}
	for (const auto* words: {&s.flattened_srt, &s.user_data}) {
		out.push_back(static_cast<uint32_t>(words->size()));
		out.insert(out.end(), words->begin(), words->end());
	}
	out.push_back(static_cast<uint32_t>(s.uniform_fill.kind));
	out.push_back(s.uniform_fill.resource);
	out.insert(out.end(), s.uniform_fill.group_stride.begin(), s.uniform_fill.group_stride.end());
	out.push_back(s.uniform_fill.words);
	out.push_back(s.uniform_fill.value);
}

// Guest bytes as the renderer would upload them (backing, not a faulting read).
inline void GuestBytes(std::vector<uint32_t>& out, uint64_t address, uint64_t size) {
	Words64(out, address);
	Words64(out, size);
	std::vector<uint32_t> data((size + 3) / 4, 0);
	if (!Libs::LibKernel::Memory::TryReadBackingToHost(address, data.data(), size))
		std::memcpy(data.data(), reinterpret_cast<const void*>(address), size);
	out.insert(out.end(), data.begin(), data.end());
}

struct StageCapture {
	LocalShaderWarmup::Record              record;
	std::vector<uint32_t>                  user_data;
	uint32_t                               push_start = 0;
	uint64_t                               code_address = 0; // SrtRuntime::shader_base
	ShaderRecompiler::IR::ResourceSnapshot resources;
	std::vector<std::pair<uint64_t, uint64_t>> buffers; // (address, size) of bound ranges
};

struct ShaderEntry {
	uint64_t address = 0, semantics = 0, user_data = 0;
	uint32_t type = 0, code_size = 0, scratch = 0, semantics_count = 0;
};

// Guest memory the replay must provide without its original contents.
enum class FillKind : uint32_t { Texture = 1, Metadata = 2, ColorTarget = 3, DepthTarget = 4, Stencil = 5 };
struct Fill {
	uint64_t address = 0, size = 0;
	FillKind kind    = FillKind::Texture;
	uint32_t value   = 0; // depth fill word for DepthTarget
};

struct PendingDraw {
	std::array<uint32_t, 5>                    draw {}; // index_count, instances, base_vertex, first_instance, index type
	StageCapture                               vertex;
	bool                                       has_pixel = false;
	StageCapture                               pixel;
	std::vector<uint32_t>                      context, shaders, user_config;
	std::vector<ShaderEntry>                   entries;
	std::vector<std::pair<uint64_t, uint64_t>> exact;   // further guest ranges captured verbatim
	std::vector<std::pair<uint64_t, uint64_t>> reads;   // SRT evaluation reads
	std::vector<Fill>                          fills;
	std::vector<Fill>                          targets; // ranges read back after the draw
};

// The command processor and the renderer run on the same GPU thread.
struct State {
	uint32_t                                   remaining = 0; // draws still to write
	uint32_t                                   stride    = 1;
	const char*                                trigger   = nullptr; // file that starts the capture
	uint64_t                                   waiting   = 0;       // XPR draws seen before it
	std::map<uint64_t, uint64_t>               cull_outputs;        // begin -> end
	bool                                       current_xpr  = false;
	bool                                       capture_this = false; // stride selection
	uint64_t                                   xpr_seen     = 0;
	std::vector<std::pair<uint64_t, uint64_t>> reads;
	uint32_t                                   written = 0;
	std::unique_ptr<PendingDraw>               pending; // completed in ExecutePreparedDraw
};
inline thread_local State g_state;
// KYTY_XPR_CAPTURE is set (Initialize): Enabled() looks at the thread's state only then (the command processor asks at
// every packet, and the state's thread-local access with its initialization check cost 0.25% of the render thread).
inline bool g_configured = false;

// Called once by the GPU thread before it consumes commands.
inline void Initialize() {
	auto& s = g_state;
	if (const auto* count = std::getenv("KYTY_XPR_CAPTURE")) {
		g_configured = true;
		s.remaining  = static_cast<uint32_t>(std::strtoul(count, nullptr, 10));
	}
	if (const auto* stride = std::getenv("KYTY_XPR_CAPTURE_STRIDE")) {
		s.stride = std::max<uint32_t>(static_cast<uint32_t>(std::strtoul(stride, nullptr, 10)), 1);
	}
	s.trigger = std::getenv("KYTY_XPR_CAPTURE_TRIGGER");
}

[[nodiscard]] inline bool Enabled() {
	return g_configured && g_state.remaining != 0;
}

// The three XPR culling compute programs (xprindexculling variants).
[[nodiscard]] inline bool IsCullProgram(uint64_t shader_hash) {
	return shader_hash == 0x71449641cf0d0d33ull || shader_hash == 0x0794beec8ee30949ull ||
	       shader_hash == 0xb78df07eda02dab9ull;
}

inline void LearnCullOutput(uint64_t address, uint64_t size) {
	if (size == 0) {
		return;
	}
	auto& outputs = g_state.cull_outputs;
	auto  end     = address + size;
	auto  it      = outputs.upper_bound(address);
	if (it != outputs.begin() && std::prev(it)->second >= address) {
		--it;
		address = it->first;
		end     = std::max(end, it->second);
		it      = outputs.erase(it);
	}
	while (it != outputs.end() && it->first <= end) {
		end = std::max(end, it->second);
		it  = outputs.erase(it);
	}
	outputs.emplace(address, end);
}

[[nodiscard]] inline bool IsCullOutput(uint64_t address) {
	const auto& outputs = g_state.cull_outputs;
	auto        it      = outputs.upper_bound(address);
	return it != outputs.begin() && std::prev(it)->second > address;
}

// Command-processor side, before a draw packet runs; `args_address` is the
// argument address of an indexed indirect draw and 0 for any other draw.
inline void ObserveDraw(uint64_t args_address) {
	auto& s        = g_state;
	s.current_xpr  = args_address != 0 && IsCullOutput(args_address);
	s.capture_this = false;
	if (!s.current_xpr) {
		return;
	}
	if (s.trigger != nullptr) {
		// Poll the trigger file only now and then.
		if (s.waiting++ % 4096 != 0 || !std::filesystem::exists(s.trigger)) {
			return;
		}
		s.trigger = nullptr;
	}
	s.capture_this = s.xpr_seen++ % s.stride == 0;
}

// Called after a captured draw was written.
inline void Written() {
	--g_state.remaining;
}

inline void Stage(std::vector<uint32_t>& out, StageCapture& stage) {
	LocalShaderWarmup::Writer writer;
	LocalShaderWarmup::Visit(writer, stage.record);
	out.push_back(static_cast<uint32_t>(writer.words.size()));
	out.insert(out.end(), writer.words.begin(), writer.words.end());
	out.push_back(static_cast<uint32_t>(stage.user_data.size()));
	out.insert(out.end(), stage.user_data.begin(), stage.user_data.end());
	out.push_back(stage.push_start);
	Words64(out, stage.code_address);
	Snapshot(out, stage.resources);
	out.push_back(static_cast<uint32_t>(stage.buffers.size()));
	for (const auto& [address, size]: stage.buffers) GuestBytes(out, address, size);
}

// Version 3: v2 layout (draw words now 5, adding the index type) followed by the
// register blocks, shader map entries, further verbatim ranges, fill ranges and
// read-back targets.
inline void WriteDraw(PendingDraw& d) {
	auto& s = g_state;
	std::vector<uint32_t> out {0x43525058u /* XPRC */, 3u, d.has_pixel ? 1u : 0u};
	out.insert(out.end(), d.draw.begin(), d.draw.end());
	Stage(out, d.vertex);
	if (d.has_pixel) Stage(out, d.pixel);
	out.push_back(static_cast<uint32_t>(d.reads.size()));
	for (const auto& [address, size]: d.reads) GuestBytes(out, address, size);
	for (const auto* block: {&d.context, &d.shaders, &d.user_config}) {
		out.push_back(static_cast<uint32_t>(block->size()));
		out.insert(out.end(), block->begin(), block->end());
	}
	out.push_back(static_cast<uint32_t>(d.entries.size()));
	for (const auto& e: d.entries) {
		Words64(out, e.address);
		out.push_back(e.type);
		out.push_back(e.code_size);
		out.push_back(e.scratch);
		out.push_back(e.semantics_count);
		Words64(out, e.semantics);
		Words64(out, e.user_data);
	}
	out.push_back(static_cast<uint32_t>(d.exact.size()));
	for (const auto& [address, size]: d.exact) GuestBytes(out, address, size);
	for (const auto* fills: {&d.fills, &d.targets}) {
		out.push_back(static_cast<uint32_t>(fills->size()));
		for (const auto& f: *fills) {
			Words64(out, f.address);
			Words64(out, f.size);
			out.push_back(static_cast<uint32_t>(f.kind));
			out.push_back(f.value);
		}
	}

	const char* dir_env = std::getenv("KYTY_XPR_CAPTURE_DIR");
	const std::filesystem::path dir = dir_env != nullptr ? dir_env : "_Build/xpr-capture";
	std::error_code error;
	std::filesystem::create_directories(dir, error);
	char name[64];
	std::snprintf(name, sizeof(name), "draw_%05u.xprc", s.written++);
	if (std::FILE* file = std::fopen((dir / name).string().c_str(), "wb")) {
		std::fwrite(out.data(), sizeof(uint32_t), out.size(), file);
		std::fclose(file);
	}
}

} // namespace Libs::Graphics::XprCapture

#endif // EMULATOR_SRC_LOCAL_XPR_CAPTURE_H_
