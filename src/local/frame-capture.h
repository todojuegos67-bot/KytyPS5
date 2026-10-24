#ifndef EMULATOR_SRC_LOCAL_FRAME_CAPTURE_H_
#define EMULATOR_SRC_LOCAL_FRAME_CAPTURE_H_
// One-shot record of every guest draw and dispatch, for mapping a title's render
// passes. The live command `capture <dir> [frames]` arms it; recording starts at the
// next guest flip and covers whole frames, one JSON line per call, one file per
// frame (<dir>/frame-N.jsonl). A line carries the call's shaders (hash and code
// address), counts or workgroups, bound targets, textures and compute buffers with
// their guest address, format and extent, key fixed-function state, and the render
// thread's wall time for the call. Formats are raw VkFormat / guest BufferFormat
// numbers. `capture <dir> <frames> <hash,hash,...>` also stores, for calls running one of
// those shaders, user data, the flattened SRT and bound buffers up to 16 KiB as hex words
// (constants: jitter, matrices). Rendering is unchanged, except that the batched draw
// paths (native XPR, indirect draw runs) step aside while a frame is recorded so every
// draw is seen.
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace FrameCapture {

enum : uint8_t { ColorTarget = 0, DepthTarget = 1, Texture = 2 };
enum : uint8_t { StageVs = 0, StagePs = 1, StageCs = 2 };

struct Image {
	uint64_t address = 0, size = 0;
	uint32_t format = 0, guest_format = 0, type = 0, width = 0, height = 0, depth = 0;
	uint32_t levels = 0, layers = 0, pitch = 0, tile = 0, samples = 0, slot = 0;
	uint8_t  kind = Texture, stage = StagePs, storage = 0, written = 0;
};

struct Buffer {
	uint64_t address = 0, size = 0;
	uint8_t  stage = StageCs, written = 0;
};

struct Blob {
	std::string           name;
	uint64_t              address = 0;
	std::vector<uint32_t> words;
};

struct Call {
	const char* name     = nullptr;
	const char* consumed = nullptr; // handled without running the shader (clear, copy)
	bool        dispatch = false;
	bool        prepared = false;   // reached binding preparation
	uint64_t    vs = 0, ps = 0, cs = 0, vs_address = 0, ps_address = 0, cs_address = 0;
	uint32_t    count = 0, instances = 0, groups[3] {}, local[3] {};
	uint64_t    indirect = 0;     // a dispatch's arguments, or the arguments a draw's GPU reads
	uint64_t    index    = 0;     // a draw's index buffer
	uint32_t    index_size = 0;   // INDEX_TYPE (0: 16-bit, 1: 32-bit)
	int32_t     base_vertex = 0;
	float       viewport[6] {}; // xscale xoffset yscale yoffset zscale zoffset
	int32_t     scissor[4] {};  // screen scissor left top right bottom
	uint32_t    target_mask = 0, blend_mask = 0, blend0 = 0; // blend0: src | dst<<8 | op<<16
	uint8_t     z_enable = 0, z_write = 0, z_func = 0;
	std::vector<Image>  images;
	std::vector<Buffer> buffers;
	std::vector<Blob>   blobs;
};

inline std::atomic<int>  g_pending {0}; // frames still to record, set by Arm()
inline std::vector<uint64_t> g_data_hashes; // set by Arm() before recording starts
inline std::vector<uint32_t> g_data_counts; // calls stored per hash in the current frame
constexpr uint32_t           DataCallsPerFrame = 4;
inline std::atomic<bool> g_active {false};
inline std::atomic<int>  g_done {0};     // frames finished by the current request
inline std::mutex        g_mutex;        // guards g_dir (live thread -> Kyty.Gpu)
inline std::string       g_dir;
inline FILE*             g_file  = nullptr; // Kyty.Gpu only from here on
inline uint32_t          g_frame = 0, g_seq = 0;
inline thread_local Call g_call;

[[nodiscard]] inline bool Active() { return g_active.load(std::memory_order_relaxed); }

// True for the first DataCallsPerFrame calls of a listed shader in each frame.
[[nodiscard]] inline bool WantsData(uint64_t hash) {
	const auto it = std::find(g_data_hashes.begin(), g_data_hashes.end(), hash);
	if (hash == 0 || it == g_data_hashes.end()) return false;
	auto& count = g_data_counts[static_cast<size_t>(it - g_data_hashes.begin())];
	return count < DataCallsPerFrame && ++count != 0;
}

inline void AddWords(std::string name, uint64_t address, const std::vector<uint32_t>& words) {
	g_call.blobs.push_back({std::move(name), address, words});
}

// Guest memory as the renderer uploads it (the backing copy, no faulting read).
inline void AddGuest(std::string name, uint64_t address, uint64_t size) {
	if (address == 0 || size == 0) return;
	size = std::min<uint64_t>(size, 16 * 1024);
	std::vector<uint32_t> words((size + 3) / 4, 0);
	if (!Libs::LibKernel::Memory::TryReadBackingToHost(address, words.data(), size)) return;
	g_call.blobs.push_back({std::move(name), address, std::move(words)});
}

// Frame-generation discovery (KYTY_FG_DISCOVER, F10): a summary of each recorded frame in the log ([fgdisc] lines), to
// find a title's scene color, depth and motion vectors, its TAA-like pass and its final composite (the flip image's
// draws and what they sample: the 3D scene and the HUD layer). The full calls are in the frame's .jsonl.
struct SummaryTarget {
	uint64_t address = 0;
	uint32_t format = 0, width = 0, height = 0, layers = 0;
	uint32_t calls = 0, first = 0, last = 0;
	uint64_t last_shader = 0;
	uint8_t  kind = 0; // ColorTarget, DepthTarget, or Texture for a storage image a dispatch writes
};
struct SummaryCandidate {
	uint32_t    index = 0;
	uint64_t    shader = 0;
	bool        dispatch = false;
	std::string images;
};
inline std::vector<SummaryTarget>    g_sum_targets;
inline std::vector<SummaryCandidate> g_sum_taa, g_sum_composite;
inline uint32_t                      g_sum_draws = 0, g_sum_dispatches = 0;
// The flip images seen (guest address, size, format), from the presenter (any thread).
inline std::atomic<uint64_t> g_flip_address[4] {};
inline std::atomic<uint32_t> g_flip_w {0}, g_flip_h {0}, g_flip_format {0};
inline void NoteFlipImage(uint64_t address, uint32_t width, uint32_t height, uint32_t format) {
	for (auto& slot: g_flip_address)
		if (slot.load(std::memory_order_relaxed) == address) return;
	static std::atomic<uint32_t> next {0};
	g_flip_address[next.fetch_add(1, std::memory_order_relaxed) % 4].store(address, std::memory_order_relaxed);
	g_flip_w.store(width, std::memory_order_relaxed);
	g_flip_h.store(height, std::memory_order_relaxed);
	g_flip_format.store(format, std::memory_order_relaxed);
}
[[nodiscard]] inline bool IsFlipAddress(uint64_t address) {
	if (address == 0) return false;
	for (auto& slot: g_flip_address)
		if (slot.load(std::memory_order_relaxed) == address) return true;
	return false;
}
// VkFormat names of the formats render targets usually have.
[[nodiscard]] inline const char* FormatName(uint32_t f) {
	switch (f) {
		case 9: return "R8";
		case 16: return "RG8";
		case 17: return "RG8_SNORM";
		case 37: return "RGBA8";
		case 38: return "RGBA8_SNORM";
		case 43: return "RGBA8_SRGB";
		case 44: return "BGRA8";
		case 50: return "BGRA8_SRGB";
		case 58: return "A2R10G10B10";
		case 64: return "A2B10G10R10";
		case 70: return "R16";
		case 76: return "R16F";
		case 77: return "RG16";
		case 78: return "RG16_SNORM";
		case 81: return "RG16_UINT";
		case 83: return "RG16F";
		case 91: return "RGBA16";
		case 92: return "RGBA16_SNORM";
		case 97: return "RGBA16F";
		case 98: return "R32_UINT";
		case 100: return "R32F";
		case 101: return "RG32_UINT";
		case 103: return "RG32F";
		case 109: return "RGBA32F";
		case 122: return "R11G11B10F";
		case 124: return "D16";
		case 125: return "X8D24";
		case 126: return "D32F";
		case 129: return "D24S8";
		case 130: return "D32FS8";
		default: return "?";
	}
}
[[nodiscard]] inline bool DepthLike(uint32_t f) { return f == 124 || f == 125 || f == 126 || f == 129 || f == 130; }
[[nodiscard]] inline bool MotionLike(uint32_t f) { return f == 16 || f == 17 || f == 77 || f == 78 || f == 83 || f == 103; }

inline void SummaryReset() {
	g_sum_targets.clear();
	g_sum_taa.clear();
	g_sum_composite.clear();
	g_sum_draws = g_sum_dispatches = 0;
}

inline std::string DescribeImages(const std::vector<Image>& images) {
	std::string out;
	char        text[160];
	for (const auto& m: images) {
		const char* kind = m.kind == ColorTarget ? "rt" : m.kind == DepthTarget ? "ds" : m.storage ? (m.written ? "uav-w" : "uav") : "tex";
		std::snprintf(text, sizeof(text), " %s%u:%" PRIx64 ":%s:%ux%u%s", kind, m.slot, m.address, FormatName(m.format),
		              m.width, m.height, IsFlipAddress(m.address) ? ":FLIP" : "");
		out += text;
	}
	return out;
}

inline void SummaryAdd(const Call& call, uint32_t index) {
	(call.dispatch ? g_sum_dispatches : g_sum_draws)++;
	const auto note = [&](const Image& m, uint8_t kind) {
		for (auto& t: g_sum_targets)
			if (t.address == m.address && t.kind == kind && t.format == m.format && t.width == m.width) {
				++t.calls;
				t.last        = index;
				t.last_shader = call.dispatch ? call.cs : call.ps;
				return;
			}
		if (g_sum_targets.size() >= 512) return;
		g_sum_targets.push_back({m.address, m.format, m.width, m.height, m.layers, 1, index, index,
		                         call.dispatch ? call.cs : call.ps, kind});
	};
	bool                     flip = false;
	uint32_t                 out_w = 0, out_h = 0;
	for (const auto& m: call.images) {
		if (m.kind == ColorTarget || m.kind == DepthTarget) note(m, m.kind);
		else if (m.storage && m.written) note(m, Texture);
		if (m.kind == ColorTarget && IsFlipAddress(m.address)) flip = true;
		if ((m.kind == ColorTarget || (m.storage && m.written)) && out_w == 0) out_w = m.width, out_h = m.height;
	}
	// A TAA-like pass: it samples a depth-like and a motion-like image of the size it writes.
	bool depth = false, motion = false;
	for (const auto& m: call.images) {
		if (m.kind != Texture || (m.storage && m.written)) continue;
		if (out_w != 0 && (m.width != out_w || m.height != out_h)) continue;
		depth  = depth || DepthLike(m.format) || m.format == 100;
		motion = motion || MotionLike(m.format);
	}
	if (depth && motion && g_sum_taa.size() < 16)
		g_sum_taa.push_back({index, call.dispatch ? call.cs : call.ps, call.dispatch, DescribeImages(call.images)});
	if (flip && g_sum_composite.size() < 16)
		g_sum_composite.push_back({index, call.ps, false, DescribeImages(call.images)});
}

inline void SummaryPrint(uint32_t frame, uint32_t calls) {
	std::printf("[fgdisc] frame %u: %u calls (%u draws, %u dispatches); flip image %ux%u %s at", frame, calls, g_sum_draws,
	            g_sum_dispatches, g_flip_w.load(), g_flip_h.load(), FormatName(g_flip_format.load()));
	for (auto& slot: g_flip_address)
		if (const auto a = slot.load(std::memory_order_relaxed); a != 0) std::printf(" %" PRIx64, a);
	std::printf("\n");
	auto targets = g_sum_targets;
	std::sort(targets.begin(), targets.end(), [](const auto& a, const auto& b) { return a.calls > b.calls; });
	uint32_t shown = 0;
	for (const auto& t: targets) {
		if (shown++ == 40) break;
		std::printf("[fgdisc]  %s %" PRIx64 " %s %ux%u%s: %u calls (#%u..#%u), last %s %016" PRIx64 "%s%s\n",
		            t.kind == ColorTarget ? "RT " : t.kind == DepthTarget ? "DS " : "UAV", t.address, FormatName(t.format),
		            t.width, t.height, t.layers > 1 ? " (layers)" : "", t.calls, t.first, t.last,
		            t.kind == Texture ? "cs" : "ps", t.last_shader, IsFlipAddress(t.address) ? "  <- FLIP IMAGE" : "",
		            MotionLike(t.format) ? "  <- motion-like" : DepthLike(t.format) ? "  <- depth" : "");
	}
	for (const auto& c: g_sum_taa)
		std::printf("[fgdisc]  TAA-like %s #%u %s %016" PRIx64 ":%s\n", c.dispatch ? "dispatch" : "draw", c.index,
		            c.dispatch ? "cs" : "ps", c.shader, c.images.c_str());
	for (const auto& c: g_sum_composite)
		std::printf("[fgdisc]  into the flip image: draw #%u ps %016" PRIx64 ":%s\n", c.index, c.shader, c.images.c_str());
	std::fflush(stdout);
}

// Live thread or the F10 key: record the next `frames` frames into `dir` without waiting (false: one is in progress).
[[nodiscard]] inline bool ArmAsync(const std::string& dir, int frames) {
	if (g_active.load() || g_pending.load() > 0) return false;
	{
		const std::lock_guard lock(g_mutex);
		g_dir = dir;
	}
	g_data_hashes.clear();
	g_data_counts.clear();
	g_done.store(0);
	g_pending.store(frames, std::memory_order_release);
	std::printf("[fgdisc] recording %d frame(s) into %s\n", frames, dir.c_str());
	std::fflush(stdout);
	return true;
}

inline void OpenFrame() {
	const std::string path = g_dir + "/frame-" + std::to_string(g_frame) + ".jsonl";
	g_file = std::fopen(path.c_str(), "wb");
	g_seq  = 0;
	SummaryReset();
	std::fill(g_data_counts.begin(), g_data_counts.end(), 0u);
	if (g_file != nullptr) std::fprintf(g_file, "{\"t\":\"frame\",\"frame\":%u}\n", g_frame);
}


// KYTY_FG_DISCOVER=1: two frames recorded 20, 60 and 180 s after the frame-generation gate opened (`ready`: the
// player's first button), into _DebugSnapshots/fgdisc-<n> (KYTY_FG_DISCOVER_AT=<s>,<s>,...: other times). Kyty.Gpu.
inline void AutoTick(bool ready) {
	static const std::vector<double> at = [] {
		std::vector<double> v;
		const char*         on = std::getenv("KYTY_FG_DISCOVER");
		if (on == nullptr || *on == '\0' || *on == '0') return v;
		const char* list = std::getenv("KYTY_FG_DISCOVER_AT");
		std::string text = list != nullptr && *list != '\0' ? list : "20,60,180";
		for (size_t begin = 0; begin < text.size();) {
			const auto end = std::min(text.find(',', begin), text.size());
			if (end > begin) v.push_back(std::atof(text.substr(begin, end - begin).c_str()));
			begin = end + 1;
		}
		std::printf("[fgdisc] on: frames recorded %zu time(s) after the first button (F10: now)\n", v.size());
		std::fflush(stdout);
		return v;
	}();
	if (at.empty() || !ready) return;
	static std::chrono::steady_clock::time_point start {};
	static size_t                                next = 0;
	const auto                                   now  = std::chrono::steady_clock::now();
	if (start == std::chrono::steady_clock::time_point {}) start = now;
	if (next >= at.size() || std::chrono::duration<double>(now - start).count() < at[next]) return;
	const auto dir = std::filesystem::absolute(std::filesystem::path("_DebugSnapshots") / ("fgdisc-" + std::to_string(next + 1)));
	if (ArmAsync(dir.string(), 2)) ++next;
}

// Kyty.Gpu, at every guest flip.
inline void OnFlip() {
	if (g_active.load(std::memory_order_relaxed)) {
		SummaryPrint(g_frame, g_seq);
		if (g_file != nullptr) {
			std::fprintf(g_file, "{\"t\":\"flip\",\"frame\":%u,\"calls\":%u}\n", g_frame, g_seq);
			std::fclose(g_file);
			g_file = nullptr;
		}
		++g_frame;
		g_done.fetch_add(1, std::memory_order_release);
		if (g_pending.fetch_sub(1, std::memory_order_acq_rel) <= 1) {
			g_active.store(false, std::memory_order_relaxed);
			return;
		}
		OpenFrame();
		return;
	}
	if (g_pending.load(std::memory_order_acquire) <= 0) return;
	{
		const std::lock_guard lock(g_mutex);
		std::error_code       error;
		std::filesystem::create_directories(g_dir, error);
	}
	g_frame = 0;
	OpenFrame();
	g_active.store(true, std::memory_order_relaxed);
}

// Live thread: record the next `frames` frames into `dir`, wait for them. `hashes` is a
// comma-separated list of shader hashes whose constants are stored too.
[[nodiscard]] inline bool Arm(const std::string& dir, int frames, const std::string& hashes,
                              double timeout_seconds) {
	if (g_active.load() || g_pending.load() > 0) return false;
	{
		const std::lock_guard lock(g_mutex);
		g_dir = dir;
	}
	g_data_hashes.clear();
	for (size_t begin = 0; begin < hashes.size();) {
		const auto end = std::min(hashes.find(',', begin), hashes.size());
		if (end > begin) g_data_hashes.push_back(std::strtoull(hashes.substr(begin, end - begin).c_str(), nullptr, 16));
		begin = end + 1;
	}
	g_data_counts.assign(g_data_hashes.size(), 0u);
	g_done.store(0);
	g_pending.store(frames, std::memory_order_release);
	const auto deadline = std::chrono::steady_clock::now() +
	                      std::chrono::duration<double>(timeout_seconds);
	while (g_done.load(std::memory_order_acquire) < frames) {
		if (std::chrono::steady_clock::now() > deadline) return false;
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
	return true;
}

// Renderer ImageInfo (template: this header stays free of renderer types).
template <typename Info>
[[nodiscard]] Image FromInfo(const Info& info, uint8_t kind, uint8_t stage, uint32_t slot) {
	Image m;
	m.address      = info.data.address;
	m.size         = info.data.size;
	m.format       = static_cast<uint32_t>(info.pixel_format);
	m.guest_format = static_cast<uint32_t>(info.guest_format);
	m.type         = static_cast<uint32_t>(info.type);
	m.width        = info.extent.width;
	m.height       = info.extent.height;
	m.depth        = info.extent.depth;
	m.levels       = info.resources.levels;
	m.layers       = info.resources.layers;
	m.pitch        = info.pitch;
	m.tile         = static_cast<uint32_t>(info.tile_mode);
	m.samples      = info.samples;
	m.kind         = kind;
	m.stage        = stage;
	m.slot         = slot;
	return m;
}

inline void Append(std::string& out, const char* format, auto... args) {
	char text[512];
	const int n = std::snprintf(text, sizeof(text), format, args...);
	if (n > 0) out.append(text, static_cast<size_t>(n < static_cast<int>(sizeof(text)) ? n : sizeof(text) - 1));
}

inline void Write(const Call& call, double microseconds) {
	if (g_file == nullptr) return;
	SummaryAdd(call, g_seq);
	std::string line;
	line.reserve(512 + call.images.size() * 160 + call.buffers.size() * 64);
	Append(line, "{\"i\":%u,\"t\":\"%s\",\"us\":%.2f,\"prep\":%d", g_seq++, call.name ? call.name : "?",
	       microseconds, call.prepared ? 1 : 0);
	if (call.consumed) Append(line, ",\"consumed\":\"%s\"", call.consumed);
	if (call.dispatch) {
		Append(line, ",\"cs\":\"%016" PRIx64 "\",\"cs_va\":\"%" PRIx64 "\",\"groups\":[%u,%u,%u],\"local\":[%u,%u,%u]",
		       call.cs, call.cs_address, call.groups[0], call.groups[1], call.groups[2], call.local[0],
		       call.local[1], call.local[2]);
		if (call.indirect != 0) Append(line, ",\"indirect\":\"%" PRIx64 "\"", call.indirect);
	} else {
		Append(line, ",\"vs\":\"%016" PRIx64 "\",\"ps\":\"%016" PRIx64 "\",\"vs_va\":\"%" PRIx64 "\",\"ps_va\":\"%" PRIx64
		       "\",\"n\":%u,\"inst\":%u",
		       call.vs, call.ps, call.vs_address, call.ps_address, call.count, call.instances);
		if (call.index != 0)
			Append(line, ",\"ib\":\"%" PRIx64 "\",\"it\":%u,\"bv\":%d", call.index, call.index_size, call.base_vertex);
		if (call.indirect != 0) Append(line, ",\"indirect\":\"%" PRIx64 "\"", call.indirect);
		Append(line, ",\"vp\":[%g,%g,%g,%g,%g,%g],\"sc\":[%d,%d,%d,%d]", call.viewport[0], call.viewport[1],
		       call.viewport[2], call.viewport[3], call.viewport[4], call.viewport[5], call.scissor[0],
		       call.scissor[1], call.scissor[2], call.scissor[3]);
		Append(line, ",\"mask\":\"%x\",\"blend\":\"%x\",\"blend0\":\"%x\",\"z\":[%u,%u,%u]", call.target_mask,
		       call.blend_mask, call.blend0, call.z_enable, call.z_write, call.z_func);
	}
	line += ",\"img\":[";
	for (size_t i = 0; i < call.images.size(); ++i) {
		const auto& m = call.images[i];
		Append(line,
		       "%s{\"k\":%u,\"st\":%u,\"slot\":%u,\"a\":\"%" PRIx64 "\",\"sz\":%" PRIu64
		       ",\"f\":%u,\"gf\":%u,\"ty\":%u,\"w\":%u,\"h\":%u,\"d\":%u,\"mip\":%u,\"arr\":%u,\"pitch\":%u,"
		       "\"tile\":%u,\"ms\":%u,\"rw\":%u,\"uav\":%u}",
		       i == 0 ? "" : ",", m.kind, m.stage, m.slot, m.address, m.size, m.format, m.guest_format, m.type,
		       m.width, m.height, m.depth, m.levels, m.layers, m.pitch, m.tile, m.samples, m.written, m.storage);
	}
	line += "],\"buf\":[";
	for (size_t i = 0; i < call.buffers.size(); ++i) {
		const auto& b = call.buffers[i];
		Append(line, "%s{\"st\":%u,\"a\":\"%" PRIx64 "\",\"sz\":%" PRIu64 ",\"rw\":%u}", i == 0 ? "" : ",", b.stage,
		       b.address, b.size, b.written);
	}
	line += "]";
	if (!call.blobs.empty()) {
		line += ",\"data\":[";
		for (size_t i = 0; i < call.blobs.size(); ++i) {
			const auto& blob = call.blobs[i];
			Append(line, "%s{\"n\":\"%s\",\"a\":\"%" PRIx64 "\",\"w\":\"", i == 0 ? "" : ",", blob.name.c_str(),
			       blob.address);
			for (const auto word: blob.words) Append(line, "%08x", word);
			line += "\"}";
		}
		line += "]";
	}
	line += "}\n";
	std::fwrite(line.data(), 1, line.size(), g_file);
}

// Brackets one guest draw or dispatch on Kyty.Gpu; the renderer fills g_call.
class Scope {
public:
	Scope(const char* name, bool dispatch): m_on(Active()) {
		if (!m_on) return;
		g_call.name     = name;
		g_call.consumed = nullptr;
		g_call.dispatch = dispatch;
		g_call.prepared = false;
		g_call.vs = g_call.ps = g_call.cs = 0;
		g_call.vs_address = g_call.ps_address = g_call.cs_address = 0;
		g_call.count = g_call.instances = 0;
		g_call.indirect                 = 0;
		g_call.index = 0, g_call.index_size = 0, g_call.base_vertex = 0;
		for (auto& value: g_call.groups) value = 0;
		for (auto& value: g_call.local) value = 0;
		g_call.images.clear();
		g_call.buffers.clear();
		g_call.blobs.clear();
		m_start = std::chrono::steady_clock::now();
	}
	~Scope() {
		if (!m_on) return;
		const auto elapsed = std::chrono::steady_clock::now() - m_start;
		Write(g_call, std::chrono::duration<double, std::micro>(elapsed).count());
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool                                  m_on;
	std::chrono::steady_clock::time_point m_start {};
};

} // namespace FrameCapture

#endif // EMULATOR_SRC_LOCAL_FRAME_CAPTURE_H_
