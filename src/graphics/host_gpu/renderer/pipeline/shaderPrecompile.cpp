#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"

#include "common/archive.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "graphics/shader/shaderCompiler.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <map>
#include <string_view>
#include <unordered_map>
#include <xxhash.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics::Precompile {

namespace {

// Raised by hand when a recorded struct changes without changing its size.
// 2: a program without push data is recorded at cursor PushData::NoStart. 3: a shader hash covers only the program
// code and literal buffer (upstream ae67e2db), so every recorded key changed.
constexpr uint32_t LayoutVersion  = 3;
constexpr uint32_t JournalVersion = 1;
constexpr uint32_t HintsVersion   = 1;
constexpr uint32_t WarmVersion    = 2;

constexpr uint32_t Tag(const char (&text)[5]) {
	return static_cast<uint32_t>(text[0]) | (static_cast<uint32_t>(text[1]) << 8u) |
	       (static_cast<uint32_t>(text[2]) << 16u) | (static_cast<uint32_t>(text[3]) << 24u);
}
constexpr uint32_t TagStage     = Tag("STGE");
constexpr uint32_t TagPipeline  = Tag("PIPE");
constexpr uint32_t TagHintInfo  = Tag("HINF");
constexpr uint32_t TagHintStage = Tag("HSTG");
constexpr uint32_t TagTemplate  = Tag("TMPL");
// A stage record with callees (a shader with calls): its own tags, so readers before them skip it.
constexpr uint32_t TagStageCalls     = Tag("STGC");
constexpr uint32_t TagHintStageCalls = Tag("HSTC");

constexpr std::string_view JournalMagic = "KYPJRNL1";
constexpr std::string_view HintsMagic   = "KYHINTS1";
constexpr std::string_view WarmMagic    = "KYWARML2";

// Every shader of Marvel's Wolverine ends in a 48-byte footer that starts with this tag
// (Tools/assets/shaders.py: the code ends 0x30 bytes after "barefoot").
constexpr std::string_view FooterMarker = "barefoot";
constexpr size_t           FooterBytes  = 48;

class Writer {
public:
	void U32(uint32_t value) { m_words.push_back(value); }
	void U64(uint64_t value) {
		U32(static_cast<uint32_t>(value));
		U32(static_cast<uint32_t>(value >> 32u));
	}
	void Words(std::span<const uint32_t> words) {
		U32(static_cast<uint32_t>(words.size()));
		m_words.insert(m_words.end(), words.begin(), words.end());
	}
	// A byte length, then the bytes zero-padded to whole words.
	void Bytes(const void* data, size_t size) {
		U32(static_cast<uint32_t>(size));
		const auto at = m_words.size();
		m_words.resize(at + (size + 3u) / 4u, 0u);
		if (size != 0) std::memcpy(m_words.data() + at, data, size);
	}
	void Text(std::string_view text) { Bytes(text.data(), text.size()); }
	void Magic(std::string_view magic) { Bytes(magic.data(), magic.size()); }

	[[nodiscard]] std::vector<uint32_t>&       Data() { return m_words; }
	[[nodiscard]] const std::vector<uint32_t>& Data() const { return m_words; }

private:
	std::vector<uint32_t> m_words;
};

class Reader {
public:
	explicit Reader(std::span<const uint32_t> words): m_words(words) {}
	uint32_t U32() {
		if (m_at >= m_words.size()) {
			m_ok = false;
			return 0;
		}
		return m_words[m_at++];
	}
	uint64_t U64() {
		const uint64_t low = U32();
		return low | (static_cast<uint64_t>(U32()) << 32u);
	}
	std::vector<uint32_t> Words() {
		const auto count = U32();
		if (!m_ok || count > m_words.size() - m_at) {
			m_ok = false;
			return {};
		}
		std::vector<uint32_t> words(m_words.begin() + static_cast<ptrdiff_t>(m_at),
		                            m_words.begin() + static_cast<ptrdiff_t>(m_at + count));
		m_at += count;
		return words;
	}
	// False unless exactly size bytes follow.
	bool Bytes(void* data, size_t size) {
		const auto stored = U32();
		const auto words  = (static_cast<size_t>(stored) + 3u) / 4u;
		if (!m_ok || stored != size || words > m_words.size() - m_at) {
			m_ok = false;
			return false;
		}
		if (size != 0) std::memcpy(data, m_words.data() + m_at, size);
		m_at += words;
		return true;
	}
	std::string Text() {
		const auto size  = U32();
		const auto words = (static_cast<size_t>(size) + 3u) / 4u;
		if (!m_ok || words > m_words.size() - m_at) {
			m_ok = false;
			return {};
		}
		std::string text(reinterpret_cast<const char*>(m_words.data() + m_at), size);
		m_at += words;
		return text;
	}
	[[nodiscard]] bool   Ok() const { return m_ok; }
	[[nodiscard]] bool   Done() const { return m_at == m_words.size(); }
	[[nodiscard]] size_t At() const { return m_at; }

private:
	std::span<const uint32_t> m_words;
	size_t                    m_at = 0;
	bool                      m_ok = true;
};

// A framed payload: u32 size in words, the words, u64 XXH3 of them.
void Frame(Writer& file, std::span<const uint32_t> payload) {
	file.U32(static_cast<uint32_t>(payload.size()));
	file.Data().insert(file.Data().end(), payload.begin(), payload.end());
	file.U64(XXH3_64bits(payload.data(), payload.size_bytes()));
}

// Walks the frames after a header. A frame whose hash fails is skipped; a size past the end tears the file there.
template <typename Visit>
void ForEachFrame(std::span<const uint32_t> words, size_t at, uint32_t& rejected, bool& torn, Visit&& visit) {
	while (at < words.size()) {
		const auto size = words[at];
		if (size > words.size() - at - 1u || words.size() - at - 1u - size < 2u) {
			torn = true;
			return;
		}
		const auto payload = words.subspan(at + 1u, size);
		const auto hash    = static_cast<uint64_t>(words[at + 1u + size]) |
		                  (static_cast<uint64_t>(words[at + 2u + size]) << 32u);
		at += size + 3u;
		if (XXH3_64bits(payload.data(), payload.size_bytes()) != hash) {
			rejected++;
			continue;
		}
		visit(payload);
	}
}

bool ReadWords(const std::filesystem::path& path, std::vector<uint32_t>& words, bool& torn) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) return false;
	const auto size = static_cast<size_t>(file.tellg());
	file.seekg(0);
	words.assign(size / 4u, 0u);
	torn = size % 4u != 0;
	return static_cast<bool>(file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * 4u))) ||
	       words.empty();
}

bool WriteWordsAtomically(const std::filesystem::path& path, std::span<const uint32_t> words) {
	std::error_code error;
	if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), error);
	auto temp = path;
	temp += ".tmp";
	{
		std::ofstream file(temp, std::ios::binary | std::ios::trunc);
		if (!file || !file.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size_bytes())) ||
		    !file.flush()) {
			return false;
		}
	}
	std::filesystem::rename(temp, path, error);
	return !error;
}

std::span<const uint8_t> AsBytes(std::span<const uint32_t> words) {
	return {reinterpret_cast<const uint8_t*>(words.data()), words.size_bytes()};
}

const std::span<const uint8_t> InputBytes(const InputInfo& input) {
	return std::visit([](const auto& info) {
		return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&info), sizeof(info));
	}, input);
}

// Two bits of an image's flags: an image that doesn't decode writes the same bytes, so its stage key stays.
uint32_t SrgbDecodeBits(Prospero::BufferFormat format) {
	return format == Prospero::BufferFormat::k8Srgb ? 1u : format == Prospero::BufferFormat::k8_8Srgb ? 2u : 0u;
}

Prospero::BufferFormat SrgbDecodeFormat(uint32_t bits) {
	return bits == 1u   ? Prospero::BufferFormat::k8Srgb
	       : bits == 2u ? Prospero::BufferFormat::k8_8Srgb
	                    : Prospero::BufferFormat::kInvalid;
}

void WriteSpecialization(Writer& out, const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	// the table variant in the count's top bit; normal keys stay as they were.
	out.U32(static_cast<uint32_t>(specialization.buffers.size()) | (specialization.buffer_table ? 0x80000000u : 0u));
	for (const auto& buffer: specialization.buffers) {
		for (const uint32_t value: {buffer.packed_stride, static_cast<uint32_t>(buffer.descriptor_format), buffer.descriptor_swizzle,
		                            static_cast<uint32_t>(buffer.zero_stride_oob), static_cast<uint32_t>(buffer.scalar_empty),
		                            buffer.indirect_root, buffer.indirect_mapping_offset, buffer.indirect_search_iterations}) {
			out.U32(value);
		}
	}
	out.U32(static_cast<uint32_t>(specialization.images.size()));
	for (const auto& image: specialization.images) {
		const uint32_t flags = static_cast<uint32_t>(image.cube) | (static_cast<uint32_t>(image.fmask) << 1u) |
		                       (static_cast<uint32_t>(image.heap) << 2u) | (static_cast<uint32_t>(image.guest_heap) << 3u) |
		                       (static_cast<uint32_t>(image.skip_degamma) << 4u) | (SrgbDecodeBits(image.srgb_decode) << 5u);
		for (const uint32_t value: {static_cast<uint32_t>(image.numeric_class), static_cast<uint32_t>(image.dimension),
		                            image.mip_count, static_cast<uint32_t>(image.conversion_format), image.shader_swizzle,
		                            image.indirect_root, image.indirect_mapping_offset, image.indirect_search_iterations,
		                            image.indirect_capacity, flags}) {
			out.U32(value);
		}
	}
	out.U32(static_cast<uint32_t>(specialization.samplers.size()));
	for (const auto& sampler: specialization.samplers) {
		for (const uint32_t value: {sampler.indirect_root, sampler.indirect_mapping_offset, sampler.indirect_search_iterations,
		                            static_cast<uint32_t>(sampler.guest_heap)}) {
			out.U32(value);
		}
	}
}

bool ReadSpecialization(Reader& in, ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	// shortcut: 4096 of each at most, against a damaged count; the game's largest have a few hundred.
	constexpr uint32_t Most = 4096;
	const auto         counted = in.U32();
	const auto         buffers = counted & 0x7fffffffu;
	specialization.buffer_table = (counted & 0x80000000u) != 0u;
	if (!in.Ok() || buffers > Most) return false;
	specialization.buffers.resize(buffers);
	for (auto& buffer: specialization.buffers) {
		buffer.packed_stride              = in.U32();
		buffer.descriptor_format          = static_cast<Prospero::BufferFormat>(in.U32());
		buffer.descriptor_swizzle         = in.U32();
		buffer.zero_stride_oob            = in.U32() != 0;
		buffer.scalar_empty               = in.U32() != 0;
		buffer.indirect_root              = in.U32();
		buffer.indirect_mapping_offset    = in.U32();
		buffer.indirect_search_iterations = in.U32();
	}
	const auto images = in.U32();
	if (!in.Ok() || images > Most) return false;
	specialization.images.resize(images);
	for (auto& image: specialization.images) {
		image.numeric_class              = static_cast<Prospero::TextureNumericClass>(in.U32());
		image.dimension                  = static_cast<ShaderRecompiler::Decoder::ImageDimension>(in.U32());
		image.mip_count                  = in.U32();
		image.conversion_format          = static_cast<Prospero::BufferFormat>(in.U32());
		image.shader_swizzle             = in.U32();
		image.indirect_root              = in.U32();
		image.indirect_mapping_offset    = in.U32();
		image.indirect_search_iterations = in.U32();
		image.indirect_capacity          = in.U32();
		const auto flags                 = in.U32();
		image.cube                       = (flags & 1u) != 0;
		image.fmask                      = (flags & 2u) != 0;
		image.heap                       = (flags & 4u) != 0;
		image.guest_heap                 = (flags & 8u) != 0;
		image.skip_degamma               = (flags & 16u) != 0;
		image.srgb_decode                = SrgbDecodeFormat((flags >> 5u) & 3u);
	}
	const auto samplers = in.U32();
	if (!in.Ok() || samplers > Most) return false;
	specialization.samplers.resize(samplers);
	for (auto& sampler: specialization.samplers) {
		sampler.indirect_root              = in.U32();
		sampler.indirect_mapping_offset    = in.U32();
		sampler.indirect_search_iterations = in.U32();
		sampler.guest_heap                 = in.U32() != 0;
	}
	return in.Ok();
}

void WriteStage(Writer& out, const StageRecord& record, bool with_code) {
	out.U32(static_cast<uint32_t>(record.stage));
	out.U64(record.shader_hash);
	out.U32(record.user_data_count);
	out.Words(with_code ? std::span<const uint32_t>(record.code) : std::span<const uint32_t> {});
	out.Words(with_code ? std::span<const uint32_t>(record.back_code) : std::span<const uint32_t> {});
	const auto input = InputBytes(record.input);
	out.Bytes(input.data(), input.size());
	out.U32(record.push_data_cursor);
	WriteSpecialization(out, record.specialization);
	out.U64(record.spirv_hash);
	out.U64(record.origin);
}

// A stage record's journal payload: STGE, or STGC with the callees' code after the record.
void WriteStagePayload(Writer& payload, const StageRecord& record) {
	payload.U32(record.callees.empty() ? TagStage : TagStageCalls);
	WriteStage(payload, record, true);
	if (record.callees.empty()) return;
	payload.U32(static_cast<uint32_t>(record.callees.size()));
	for (const auto& callee: record.callees) payload.Words(callee);
}

InputInfo InputFor(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return ShaderComputeInputInfo {};
		case ShaderType::Pixel: return ShaderPixelInputInfo {};
		default: return ShaderVertexInputInfo {};
	}
}

bool KnownStage(uint32_t stage) {
	switch (static_cast<ShaderType>(stage)) {
		case ShaderType::Vertex:
		case ShaderType::Pixel:
		case ShaderType::Compute:
		case ShaderType::Mesh:
		case ShaderType::Local:
		case ShaderType::TessellationControl:
		case ShaderType::TessellationEvaluation: return true;
		default: return false;
	}
}

bool ReadStage(Reader& in, StageRecord& record) {
	const auto stage = in.U32();
	if (!in.Ok() || !KnownStage(stage)) return false;
	record.stage           = static_cast<ShaderType>(stage);
	record.shader_hash     = in.U64();
	record.user_data_count = in.U32();
	record.code            = in.Words();
	record.back_code       = in.Words();
	record.input           = InputFor(record.stage);
	bool input_read        = std::visit([&](auto& info) { return in.Bytes(&info, sizeof(info)); }, record.input);
	record.push_data_cursor = in.U32();
	if (!input_read || !ReadSpecialization(in, record.specialization)) return false;
	record.spirv_hash = in.U64();
	record.origin     = in.U64();
	return in.Ok();
}

void WritePipeline(Writer& out, const PipelineRecord& record) {
	for (const auto key: record.vertex) out.U64(key);
	out.U64(record.pixel);
	out.U32(record.rendering.color_count);
	for (const auto format: record.rendering.color_formats) out.U32(static_cast<uint32_t>(format));
	out.U32(static_cast<uint32_t>(record.rendering.depth_format));
	out.U32(static_cast<uint32_t>(record.rendering.stencil_format));
	out.U32(record.vertex_input.binding_count);
	for (uint32_t i = 0; i < record.vertex_input.binding_count; i++) {
		out.U32(record.vertex_input.bindings[i].stride);
		out.U32(static_cast<uint32_t>(record.vertex_input.bindings[i].instance));
	}
	out.U32(record.vertex_input.attribute_count);
	for (uint32_t i = 0; i < record.vertex_input.attribute_count; i++) {
		out.U32(record.vertex_input.attributes[i].offset);
		out.U32(record.vertex_input.attributes[i].binding);
	}
	out.Bytes(&record.static_params, sizeof(record.static_params));
}

bool ReadPipeline(Reader& in, PipelineRecord& record) {
	for (auto& key: record.vertex) key = in.U64();
	record.pixel                 = in.U64();
	record.rendering.color_count = in.U32();
	for (auto& format: record.rendering.color_formats) format = static_cast<vk::Format>(in.U32());
	record.rendering.depth_format   = static_cast<vk::Format>(in.U32());
	record.rendering.stencil_format = static_cast<vk::Format>(in.U32());
	const auto bindings             = in.U32();
	if (!in.Ok() || bindings > ShaderVertexInputInfo::RES_MAX ||
	    record.rendering.color_count > RENDER_COLOR_ATTACHMENTS_MAX) {
		return false;
	}
	record.vertex_input.binding_count = static_cast<uint8_t>(bindings);
	for (uint32_t i = 0; i < bindings; i++) {
		record.vertex_input.bindings[i].stride   = in.U32();
		record.vertex_input.bindings[i].instance = in.U32() != 0;
	}
	const auto attributes = in.U32();
	if (!in.Ok() || attributes > ShaderVertexInputInfo::RES_MAX) return false;
	record.vertex_input.attribute_count = static_cast<uint8_t>(attributes);
	for (uint32_t i = 0; i < attributes; i++) {
		record.vertex_input.attributes[i].offset  = in.U32();
		record.vertex_input.attributes[i].binding = static_cast<uint8_t>(in.U32());
	}
	return in.Bytes(&record.static_params, sizeof(record.static_params)) && in.Ok() && record.vertex[0] != 0;
}

void WriteTemplate(Writer& out, const TemplateRecord& record) {
	out.U64(record.pixel);
	out.U64(record.pipeline);
	out.Bytes(record.context.data(), record.context.size());
	out.Bytes(record.user_config.data(), record.user_config.size());
	out.Bytes(record.pixel_regs.data(), record.pixel_regs.size());
	out.Bytes(record.export_mapping.data(), record.export_mapping.size());
	out.U32(static_cast<uint32_t>(record.colors.size()));
	for (const auto& color: record.colors) {
		for (const uint32_t value: {color.slot, color.format, color.guest_format, color.samples, uint32_t {color.mapping}}) {
			out.U32(value);
		}
	}
	for (const uint32_t value: {record.depth_format, record.depth_samples, uint32_t {record.depth_bound},
	                            uint32_t {record.depth_test}, record.topology, uint32_t {record.primitive_restart}}) {
		out.U32(value);
	}
}

// The register blocks keep their stored size; a reader of another layout checks it against its own.
bool ReadBlock(Reader& in, std::vector<uint8_t>& block) {
	const auto size  = in.U32();
	const auto words = (static_cast<size_t>(size) + 3u) / 4u;
	if (!in.Ok() || size > (1u << 20u)) return false;
	block.resize(size);
	std::vector<uint32_t> packed(words);
	for (auto& word: packed) word = in.U32();
	if (size != 0) std::memcpy(block.data(), packed.data(), size);
	return in.Ok();
}

bool ReadTemplate(Reader& in, TemplateRecord& record) {
	record.pixel    = in.U64();
	record.pipeline = in.U64();
	std::vector<uint8_t> mapping;
	if (!ReadBlock(in, record.context) || !ReadBlock(in, record.user_config) || !ReadBlock(in, record.pixel_regs) ||
	    !ReadBlock(in, mapping) || mapping.size() != record.export_mapping.size()) {
		return false;
	}
	std::ranges::copy(mapping, record.export_mapping.begin());
	const auto colors = in.U32();
	if (!in.Ok() || colors > RENDER_COLOR_ATTACHMENTS_MAX) return false;
	record.colors.resize(colors);
	for (auto& color: record.colors) {
		color.slot         = in.U32();
		color.format       = in.U32();
		color.guest_format = in.U32();
		color.samples      = in.U32();
		color.mapping      = static_cast<uint8_t>(in.U32());
	}
	record.depth_format      = in.U32();
	record.depth_samples     = in.U32();
	record.depth_bound       = in.U32() != 0;
	record.depth_test        = in.U32() != 0;
	record.topology          = in.U32();
	record.primitive_restart = in.U32() != 0;
	return in.Ok();
}

std::vector<uint32_t> Header(std::string_view magic, uint32_t version, uint64_t extra) {
	Writer header;
	header.Magic(magic);
	header.U32(version);
	header.U64(SchemaFingerprint());
	header.U64(extra);
	return std::move(header.Data());
}

// The size of a header written by Header().
size_t HeaderWords(std::string_view magic) {
	return 1u + (magic.size() + 3u) / 4u + 1u + 2u + 2u;
}

bool CheckHeader(std::span<const uint32_t> words, std::string_view magic, uint32_t version, uint64_t& extra) {
	if (words.size() < HeaderWords(magic)) return false;
	Reader      in(words.first(HeaderWords(magic)));
	std::string read(magic.size(), '\0');
	if (!in.Bytes(read.data(), read.size()) || read != magic || in.U32() != version || in.U64() != SchemaFingerprint()) {
		return false;
	}
	extra = in.U64();
	return in.Ok();
}

} // namespace

uint64_t CodeHash(std::span<const uint32_t> code, std::span<const uint32_t> back_code) {
	const auto hash = XXH3_64bits(code.data(), code.size_bytes());
	if (back_code.empty()) return hash;
	const uint64_t hashes[] = {hash, XXH3_64bits(back_code.data(), back_code.size_bytes())};
	return XXH3_64bits(hashes, sizeof(hashes));
}

uint64_t SchemaFingerprint() {
	static const uint64_t fingerprint = [] {
		const auto text = fmt::format(
		    "L{} V{} P{} C{} B{} I{} S{} R{} A{} SP{} RES{}", LayoutVersion, sizeof(ShaderVertexInputInfo),
		    sizeof(ShaderPixelInputInfo), sizeof(ShaderComputeInputInfo),
		    sizeof(ShaderRecompiler::IR::ResourceSpecialization::Buffer), sizeof(ShaderRecompiler::IR::ResourceSpecialization::Image),
		    sizeof(ShaderRecompiler::IR::ResourceSpecialization::Sampler), sizeof(PipelineRenderingState),
		    RENDER_COLOR_ATTACHMENTS_MAX, sizeof(PipelineStaticParameters), ShaderVertexInputInfo::RES_MAX);
		return XXH3_64bits(text.data(), text.size());
	}();
	return fingerprint;
}

uint64_t Origin() {
	static const uint64_t origin = [] {
		// The switches that change a module's SPIR-V or its pipelines (a grep of the recompiler and pipeline code for
		// getenv("KYTY_, 2026-10-29), and the options that do. Not every KYTY_ variable: the Yotei fork's lesson 4.1.
		// The ctest precompile_env_guard reads this list: a new switch must be here or in its output-only list. Of the
		// options only the debug dump changes SPIR-V (debug info); the shader log direction only logs.
		std::string text = KYTY_GIT_REVISION;
		// KYTY_UPSTREAM_PATHS and KYTY_LEAN_PATHS (common/upstreamPaths.h) pick between upstream's and lean's code,
		// some of it in the recompiler.
		for (const char* name: {"KYTY_GPL", "KYTY_IMAGE_HEAP", "KYTY_IMAGE_FORMS", "KYTY_NULL_IMAGE_DIMS",
		                         "KYTY_UPSTREAM_PATHS", "KYTY_LEAN_PATHS"}) {
			const char* value = std::getenv(name);
			text += fmt::format("|{}={}", name, value != nullptr ? value : "");
		}
		text += fmt::format("|dump={}", Config::GraphicsDebugDumpEnabled());
		return XXH3_64bits(text.data(), text.size());
	}();
	return origin;
}

uint64_t DeviceFingerprint(uint32_t vendor_id, uint32_t device_id) {
	const auto text = fmt::format("{:08x}:{:08x}", vendor_id, device_id);
	return XXH3_64bits(text.data(), text.size());
}

template <typename Info>
static InputInfo Cleared(const Info& input) {
	Info copy  = input;
	copy.stage = {};
	return copy;
}
InputInfo RecordedInput(const ShaderVertexInputInfo& input) { return Cleared(input); }
InputInfo RecordedInput(const ShaderPixelInputInfo& input) { return Cleared(input); }
InputInfo RecordedInput(const ShaderComputeInputInfo& input) { return Cleared(input); }

static void WriteStageKey(Writer& key, ShaderType stage, uint64_t shader_hash, uint32_t user_data_count, size_t code_size,
                          size_t back_code_size, std::span<const uint32_t> static_key, uint32_t push_data_cursor,
                          const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	key.U32(static_cast<uint32_t>(stage));
	key.U64(shader_hash);
	key.U32(user_data_count);
	key.U32(static_cast<uint32_t>(code_size));
	key.U32(static_cast<uint32_t>(back_code_size));
	key.Words(static_key);
	key.U32(push_data_cursor);
	WriteSpecialization(key, specialization);
}

uint64_t StageKey(ShaderType stage, uint64_t shader_hash, uint32_t user_data_count, size_t code_size, size_t back_code_size,
                  std::span<const uint32_t> static_key, uint32_t push_data_cursor,
                  const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	Writer key;
	WriteStageKey(key, stage, shader_hash, user_data_count, code_size, back_code_size, static_key, push_data_cursor,
	              specialization);
	const auto hash = XXH3_64bits(key.Data().data(), key.Data().size() * sizeof(uint32_t));
	return hash != 0 ? hash : 1;
}

uint64_t StageRecord::Key() const {
	Writer                key;
	std::vector<uint32_t> static_key;
	std::visit([&](const auto& info) { BuildStageStaticKey(info, static_key); }, input);
	WriteStageKey(key, stage, shader_hash, user_data_count, code.size(), back_code.size(), static_key, push_data_cursor,
	              specialization);
	if (!callees.empty()) {
		key.U32(static_cast<uint32_t>(callees.size()));
		for (const auto& callee: callees) key.U64(XXH3_64bits(callee.data(), callee.size() * sizeof(uint32_t)));
	}
	const auto hash = XXH3_64bits(key.Data().data(), key.Data().size() * sizeof(uint32_t));
	return hash != 0 ? hash : 1;
}

uint64_t PipelineRecord::Key() const {
	Writer key;
	WritePipeline(key, *this);
	const auto hash = XXH3_64bits(key.Data().data(), key.Data().size() * sizeof(uint32_t));
	return hash != 0 ? hash : 1;
}

uint64_t TemplateRecord::Key() const {
	const uint64_t keys[] = {pixel, pipeline};
	const auto     hash   = XXH3_64bits(keys, sizeof(keys));
	return hash != 0 ? hash : 1;
}

void Journal::Dedupe() {
	std::unordered_set<uint64_t> seen;
	std::erase_if(stages, [&](const StageRecord& record) { return !seen.insert(record.Key()).second; });
	seen.clear();
	std::erase_if(pipelines, [&](const PipelineRecord& record) { return !seen.insert(record.Key()).second; });
	seen.clear();
	std::erase_if(templates, [&](const TemplateRecord& record) { return !seen.insert(record.Key()).second; });
}

// A recorded stage in the variant this run draws with: the buffer table for a stage with buffers. Records of the
// other variant are rewritten (the shipped hints are recorded in the normal one), and
// the pipeline and template records that name them follow the new keys.
bool MigrateBufferTable(Journal& journal) {
	// Old stage key -> its key in this run's variant, for the stages that moved.
	std::unordered_map<uint64_t, uint64_t> keys;
	for (auto& record: journal.stages) {
		const bool want = !record.specialization.buffers.empty();
		if (record.specialization.buffer_table == want) continue;
		const auto key                    = record.Key();
		record.specialization.buffer_table = want;
		record.spirv_hash                  = 0;
		record.origin                      = 0; // no recorded translation to check it against
		keys[key]                          = record.Key();
	}
	if (keys.empty()) return false;
	const auto remap = [&](uint64_t key) {
		const auto found = keys.find(key);
		return found != keys.end() ? found->second : key;
	};
	std::unordered_map<uint64_t, uint64_t> pipeline_keys;
	for (auto& record: journal.pipelines) {
		const auto key = record.Key();
		for (auto& stage: record.vertex) stage = remap(stage);
		record.pixel       = remap(record.pixel);
		pipeline_keys[key] = record.Key();
	}
	for (auto& record: journal.templates) {
		record.pixel = remap(record.pixel);
		if (const auto found = pipeline_keys.find(record.pipeline); found != pipeline_keys.end()) {
			record.pipeline = found->second;
		}
	}
	journal.Dedupe();
	return true;
}

Journal ReadJournal(const std::filesystem::path& path, uint64_t device) {
	Journal               journal;
	std::vector<uint32_t> words;
	if (!ReadWords(path, words, journal.torn)) return journal;
	journal.found  = true;
	uint64_t found_device = 0;
	if (!CheckHeader(words, JournalMagic, JournalVersion, found_device) || (device != 0 && found_device != device)) {
		journal.foreign = true;
		return journal;
	}
	ForEachFrame(words, HeaderWords(JournalMagic), journal.rejected, journal.torn, [&](std::span<const uint32_t> payload) {
		Reader     in(payload);
		const auto tag = in.U32();
		if (tag == TagStage || tag == TagStageCalls) {
			StageRecord record;
			bool        ok = ReadStage(in, record);
			if (ok && tag == TagStageCalls) {
				const auto count = in.U32();
				ok               = in.Ok() && count != 0 && count <= 64;
				for (uint32_t i = 0; ok && i < count; i++) {
					record.callees.push_back(in.Words());
					ok = in.Ok() && !record.callees.back().empty();
				}
			}
			if (ok && in.Done()) {
				journal.stages.push_back(std::move(record));
				return;
			}
		} else if (tag == TagPipeline) {
			PipelineRecord record;
			if (ReadPipeline(in, record) && in.Done()) {
				journal.pipelines.push_back(record);
				return;
			}
		} else if (tag == TagTemplate) {
			TemplateRecord record;
			if (ReadTemplate(in, record) && in.Done()) {
				journal.templates.push_back(std::move(record));
				return;
			}
		}
		journal.rejected++;
	});
	return journal;
}

static void AppendFrames(Writer& file, const Journal& journal) {
	for (const auto& record: journal.stages) {
		Writer payload;
		WriteStagePayload(payload, record);
		Frame(file, payload.Data());
	}
	for (const auto& record: journal.pipelines) {
		Writer payload;
		payload.U32(TagPipeline);
		WritePipeline(payload, record);
		Frame(file, payload.Data());
	}
	for (const auto& record: journal.templates) {
		Writer payload;
		payload.U32(TagTemplate);
		WriteTemplate(payload, record);
		Frame(file, payload.Data());
	}
}

bool WriteJournal(const std::filesystem::path& path, uint64_t device, const Journal& journal) {
	Writer file;
	file.Data() = Header(JournalMagic, JournalVersion, device);
	AppendFrames(file, journal);
	return WriteWordsAtomically(path, file.Data());
}

bool JournalWriter::Open(const std::filesystem::path& path, uint64_t device, const Journal& journal, bool rewrite) {
	std::lock_guard lock(m_mutex);
	std::error_code error;
	if (!rewrite && std::filesystem::exists(path, error)) {
		m_file.open(path, std::ios::binary | std::ios::app);
		return m_file.is_open();
	}
	if (!WriteJournal(path, device, journal)) return false;
	m_file.open(path, std::ios::binary | std::ios::app);
	return m_file.is_open();
}

void JournalWriter::Write(std::span<const uint32_t> payload) {
	Writer frame;
	Frame(frame, payload);
	std::lock_guard lock(m_mutex);
	if (!m_file.is_open()) return;
	m_file.write(reinterpret_cast<const char*>(frame.Data().data()), static_cast<std::streamsize>(frame.Data().size() * 4u));
	m_file.flush();
}

void JournalWriter::Append(const StageRecord& record) {
	Writer payload;
	WriteStagePayload(payload, record);
	Write(payload.Data());
}

void JournalWriter::Append(const PipelineRecord& record) {
	Writer payload;
	payload.U32(TagPipeline);
	WritePipeline(payload, record);
	Write(payload.Data());
}

void JournalWriter::Append(const TemplateRecord& record) {
	Writer payload;
	payload.U32(TagTemplate);
	WriteTemplate(payload, record);
	Write(payload.Data());
}

uint64_t FooterHash(std::span<const uint8_t> block) {
	if (block.size() < FooterBytes) return 0;
	const auto footer = block.last(FooterBytes);
	if (std::memcmp(footer.data(), FooterMarker.data(), FooterMarker.size()) != 0) return 0;
	return XXH3_64bits(footer.data(), footer.size()) | 1u;
}

namespace {

// A block of code in the game's files. footer: the hash of the footer the block ends in; 0 for a program the shader
// map trimmed (upstream ae67e2db: code and literal buffer), which ends where its metadata starts, before the footer.
struct HintCode {
	uint32_t bytes  = 0;
	uint64_t footer = 0;
	uint64_t hash   = 0;
};

HintCode MakeHintCode(std::span<const uint32_t> code) {
	const auto bytes = AsBytes(code);
	return {.bytes = static_cast<uint32_t>(bytes.size()), .footer = FooterHash(bytes),
	        .hash = XXH3_64bits(bytes.data(), bytes.size())};
}

void WriteHintCode(Writer& out, const HintCode& code) {
	out.U32(code.bytes);
	out.U64(code.footer);
	out.U64(code.hash);
}

HintCode ReadHintCode(Reader& in) {
	HintCode code;
	code.bytes  = in.U32();
	code.footer = in.U64();
	code.hash   = in.U64();
	return code;
}

} // namespace

ExportResult ExportHints(const Journal& journal, const HintsInfo& info, const std::filesystem::path& path) {
	ExportResult result;
	Writer       file;
	file.Data() = Header(HintsMagic, HintsVersion, 0);
	{
		Writer payload;
		payload.U32(TagHintInfo);
		payload.Text(info.title_id);
		payload.Text(info.app_version);
		Frame(file, payload.Data());
	}
	std::unordered_set<uint64_t> exported;
	for (const auto& record: journal.stages) {
		const auto code = MakeHintCode(record.code);
		const auto back = record.back_code.empty() ? HintCode {} : MakeHintCode(record.back_code);
		std::vector<HintCode> callees;
		for (const auto& callee: record.callees) callees.push_back(MakeHintCode(callee));
		if (code.footer == 0) result.no_footer++;
		Writer payload;
		payload.U32(callees.empty() ? TagHintStage : TagHintStageCalls);
		WriteHintCode(payload, code);
		WriteHintCode(payload, back);
		if (!callees.empty()) {
			payload.U32(static_cast<uint32_t>(callees.size()));
			for (const auto& callee: callees) WriteHintCode(payload, callee);
		}
		WriteStage(payload, record, false);
		Frame(file, payload.Data());
		exported.insert(record.Key());
		result.stages++;
	}
	for (const auto& record: journal.pipelines) {
		const bool resolved = exported.contains(record.vertex[0]) &&
		                      std::ranges::all_of(record.vertex, [&](uint64_t key) { return key == 0 || exported.contains(key); }) &&
		                      (record.pixel == 0 || exported.contains(record.pixel));
		if (!resolved) continue;
		Writer payload;
		payload.U32(TagPipeline);
		WritePipeline(payload, record);
		Frame(file, payload.Data());
		result.pipelines++;
	}
	result.ok = WriteWordsAtomically(path, file.Data());
	return result;
}

namespace {

// The files of a game folder in the order an import reads them: eboot.bin, Marvel's Wolverine's material archives
// (where the rest of its shaders are), then every other file, smallest first.
std::vector<std::filesystem::path> ScanOrder(const std::filesystem::path& game_dir) {
	std::vector<std::pair<int, std::pair<uintmax_t, std::filesystem::path>>> files;
	const auto rank_of = [](const std::string& name) {
		return name == "eboot.bin" ? 0 : (name == "material" || name == "materialgraph") ? 1 : 2;
	};
	// A game loaded from a ZArchive: its files are virtual paths inside the archive.
	if (Common::IsArchivePath(game_dir)) {
		std::vector<std::filesystem::path> folders {game_dir};
		while (!folders.empty()) {
			const auto folder = std::move(folders.back());
			folders.pop_back();
			for (const auto& entry: Common::File::GetDirEntries(folder)) {
				const auto path = folder / entry.name;
				if (!entry.is_file) {
					folders.push_back(path);
					continue;
				}
				const auto info = Common::File::GetInfo(path);
				files.push_back({rank_of(entry.name), {info ? info->size : 0u, path}});
			}
		}
		std::ranges::sort(files);
		std::vector<std::filesystem::path> order;
		for (auto& [rank, file]: files) order.push_back(std::move(file.second));
		return order;
	}
	std::error_code error;
	for (auto it = std::filesystem::recursive_directory_iterator(
	         game_dir, std::filesystem::directory_options::follow_directory_symlink |
	                       std::filesystem::directory_options::skip_permission_denied,
	         error);
	     !error && it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
		if (!it->is_regular_file(error)) continue;
		const auto name = it->path().filename().string();
		const int  rank = name == "eboot.bin" ? 0 : (name == "material" || name == "materialgraph") ? 1 : 2;
		files.push_back({rank, {it->file_size(error), it->path()}});
	}
	std::ranges::sort(files);
	std::vector<std::filesystem::path> order;
	for (auto& [rank, file]: files) order.push_back(std::move(file.second));
	return order;
}

} // namespace

std::filesystem::path ShippedHints(const std::string& title_id) {
	if (title_id.empty()) return {};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::array<wchar_t, 32768> name {};
	const auto length = GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	if (length == 0 || length >= name.size()) return {};
	const std::filesystem::path exe = name.data();
#else
	std::error_code             link_error;
	const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", link_error);
	if (link_error) return {};
#endif
	auto path = exe.parent_path() / "hints" / (title_id + ".kyhints");
	std::error_code error;
	return std::filesystem::is_regular_file(path, error) ? path : std::filesystem::path {};
}

std::string HintsStamp(const std::filesystem::path& path) {
	std::error_code error;
	const auto      size = std::filesystem::file_size(path, error);
	if (error) return {};
	const auto time = std::filesystem::last_write_time(path, error);
	if (error) return {};
	return fmt::format("{}:{}", size, time.time_since_epoch().count());
}

ImportResult ImportHints(const std::filesystem::path& path, const std::filesystem::path& game_dir, const HintsInfo& info,
                         const std::function<void(StageRecord&)>& adjust,
                         const std::function<void(uint64_t done, uint64_t total)>& progress) {
	ImportResult          result;
	const auto            start = std::chrono::steady_clock::now();
	std::vector<uint32_t> words;
	bool                  torn = false;
	uint64_t              extra = 0;
	if (!ReadWords(path, words, torn)) {
		result.reason = "unreadable";
		return result;
	}
	if (!CheckHeader(words, HintsMagic, HintsVersion, extra)) {
		result.reason = "other_emulator_version";
		return result;
	}
	struct Hint {
		HintCode              code, back;
		std::vector<HintCode> callees;
		StageRecord           record;
	};
	std::vector<Hint>           hints;
	std::vector<PipelineRecord> pipelines;
	HintsInfo                   file_info;
	uint32_t                    rejected = 0;
	ForEachFrame(words, HeaderWords(HintsMagic), rejected, torn, [&](std::span<const uint32_t> payload) {
		Reader     in(payload);
		const auto tag = in.U32();
		if (tag == TagHintInfo) {
			file_info.title_id    = in.Text();
			file_info.app_version = in.Text();
		} else if (tag == TagHintStage || tag == TagHintStageCalls) {
			Hint hint;
			hint.code = ReadHintCode(in);
			hint.back = ReadHintCode(in);
			if (tag == TagHintStageCalls) {
				const auto count = in.U32();
				if (!in.Ok() || count == 0 || count > 64) return;
				for (uint32_t i = 0; i < count; i++) hint.callees.push_back(ReadHintCode(in));
			}
			if (ReadStage(in, hint.record) && in.Done()) hints.push_back(std::move(hint));
		} else if (tag == TagPipeline) {
			PipelineRecord record;
			if (ReadPipeline(in, record) && in.Done()) pipelines.push_back(record);
		}
	});
	if (!file_info.title_id.empty() && !info.title_id.empty() && file_info.title_id != info.title_id) {
		result.reason = "other_game";
		return result;
	}

	// Footer hash -> (bytes, block hash) of each block wanted that ends in its footer; the programs wanted, by hash, and
	// their sizes; the blocks found, by block hash.
	std::unordered_multimap<uint64_t, std::pair<uint32_t, uint64_t>> wanted;
	std::unordered_set<uint64_t>                                     programs;
	std::unordered_set<uint32_t>                                     program_bytes;
	std::unordered_map<uint64_t, std::vector<uint32_t>>              found;
	std::unordered_set<uint64_t>                                     distinct;
	size_t                                                           max_bytes = 0;
	for (const auto& hint: hints) {
		std::vector<HintCode> blocks {hint.code, hint.back};
		blocks.insert(blocks.end(), hint.callees.begin(), hint.callees.end());
		for (const auto& code: blocks) {
			if (code.bytes == 0 || code.bytes % 4u != 0 || !distinct.insert(code.hash).second) continue;
			if (code.footer != 0) {
				wanted.emplace(code.footer, std::make_pair(code.bytes, code.hash));
			} else {
				programs.insert(code.hash);
				program_bytes.insert(code.bytes);
			}
			max_bytes = std::max<size_t>(max_bytes, code.bytes);
		}
	}
	// A program's footer gives its size and its metadata's (ShaderProgramSize): code, literal buffer, metadata and
	// footer, each of the first three from a 16-byte boundary. All 14,586 Marvel's Wolverine binaries are laid out so.
	constexpr size_t ProgramMetadataMax = 1u << 20u; // the metadata a chunk's kept tail allows for
	const auto       align16            = [](uint64_t value) { return (value + 15u) & ~uint64_t {15u}; };
	const auto files   = ScanOrder(game_dir);
	const bool archive = Common::IsArchivePath(game_dir);
	uint64_t   total   = 0;
	for (const auto& file: files) {
		if (archive) {
			total += Common::File::Size(file);
			continue;
		}
		std::error_code error;
		total += std::filesystem::file_size(file, error);
	}
	constexpr size_t Chunk   = 64u << 20u;
	const size_t     overlap = max_bytes + FooterBytes + (programs.empty() ? 0u : ProgramMetadataMax);
	std::vector<uint8_t> buffer;
	const auto*                              marker = reinterpret_cast<const uint8_t*>(FooterMarker.data());
	const std::boyer_moore_horspool_searcher searcher(marker, marker + FooterMarker.size());
	for (const auto& file: files) {
		if (found.size() == distinct.size()) break;
		std::ifstream stream;
		Common::File  archived;
		if (archive ? !archived.Open(file, Common::File::Mode::Read)
		            : (stream.open(file, std::ios::binary), !stream)) {
			continue;
		}
		buffer.clear();
		for (;;) {
			const auto kept = buffer.size();
			buffer.resize(kept + Chunk);
			size_t read = 0;
			if (archive) {
				uint32_t got = 0;
				archived.Read(buffer.data() + kept, static_cast<uint32_t>(Chunk), &got);
				read = got;
			} else {
				stream.read(reinterpret_cast<char*>(buffer.data() + kept), static_cast<std::streamsize>(Chunk));
				read = static_cast<size_t>(stream.gcount());
			}
			buffer.resize(kept + read);
			result.bytes_read += read;
			if (progress) progress(result.bytes_read, total);
			const auto* begin = buffer.data();
			const auto* end   = buffer.data() + buffer.size();
			for (auto at = std::search(begin, end, searcher); at != end; at = std::search(at + 1, end, searcher)) {
				if (static_cast<size_t>(end - at) < FooterBytes) break;
				const auto footer = XXH3_64bits(at, FooterBytes) | 1u;
				const auto block_end = static_cast<size_t>(at - begin) + FooterBytes;
				for (auto [it, last] = wanted.equal_range(footer); it != last; ++it) {
					const auto [bytes, hash] = it->second;
					if (bytes > block_end || found.contains(hash)) continue;
					const auto* block = begin + (block_end - bytes);
					if (XXH3_64bits(block, bytes) != hash) continue;
					std::vector<uint32_t> code(bytes / 4u);
					std::memcpy(code.data(), block, bytes);
					found.emplace(hash, std::move(code));
				}
				if (!programs.empty()) {
					std::array<uint32_t, FooterBytes / 4u> words {};
					std::memcpy(words.data(), at, FooterBytes);
					const auto bytes = align16(words[5]) + words[6];
					const auto whole = align16(bytes) + align16(words[7]) + FooterBytes;
					if (words[4] == 1u && words[5] != 0u && bytes % 4u == 0 && words[7] <= ProgramMetadataMax &&
					    whole <= block_end && program_bytes.contains(static_cast<uint32_t>(bytes))) {
						const auto* block = begin + (block_end - whole);
						const auto  hash  = XXH3_64bits(block, bytes);
						if (programs.contains(hash) && !found.contains(hash)) {
							std::vector<uint32_t> code(bytes / 4u);
							std::memcpy(code.data(), block, bytes);
							found.emplace(hash, std::move(code));
						}
					}
				}
			}
			if (read < Chunk || found.size() == distinct.size()) break;
			// The tail stays: a block that ends in the next chunk starts within it.
			if (buffer.size() > overlap) buffer.erase(buffer.begin(), buffer.end() - static_cast<ptrdiff_t>(overlap));
		}
	}

	std::unordered_map<uint64_t, uint64_t> renamed; // a hint's stage key -> the restored record's (adjust may change it)
	for (auto& hint: hints) {
		const auto code = found.find(hint.code.hash);
		const auto back = hint.back.bytes == 0 ? found.end() : found.find(hint.back.hash);
		if (code == found.end() || (hint.back.bytes != 0 && back == found.end())) {
			result.missing++;
			continue;
		}
		auto& record = hint.record;
		bool  callees_found = true;
		for (const auto& callee: hint.callees) {
			const auto block = found.find(callee.hash);
			callees_found    = callees_found && block != found.end();
			if (callees_found) record.callees.push_back(block->second);
		}
		if (!callees_found) {
			result.missing++;
			continue;
		}
		record.code  = code->second;
		if (back != found.end()) record.back_code = back->second;
		if (CodeHash(record.code, record.back_code) != record.shader_hash) {
			result.missing++;
			continue;
		}
		const auto key = record.Key();
		if (adjust) adjust(record);
		renamed[key] = record.Key();
		result.journal.stages.push_back(std::move(record));
		result.stages++;
	}
	for (auto& record: pipelines) {
		bool resolved = true;
		for (auto* key: {&record.vertex[0], &record.vertex[1], &record.vertex[2], &record.pixel}) {
			if (*key == 0) continue;
			const auto found_key = renamed.find(*key);
			if (found_key == renamed.end()) {
				resolved = false;
				break;
			}
			*key = found_key->second;
		}
		if (!resolved) continue;
		result.journal.pipelines.push_back(record);
		result.pipelines++;
	}
	result.journal.Dedupe();
	result.ok      = true;
	result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	return result;
}

uint64_t WarmList::Entry(uint64_t key, bool graphics) {
	const uint64_t words[] = {key, Origin(), graphics ? 1u : 0u};
	return XXH3_64bits(words, sizeof(words));
}

void WarmList::Load(const std::filesystem::path& path, const std::string& signature, uint64_t cache_bytes) {
	std::lock_guard lock(m_mutex);
	m_path      = path;
	m_signature = signature;
	m_saved.clear();
	m_older.clear();
	std::vector<uint32_t> words;
	bool                  torn = false;
	if (cache_bytes == 0 || !ReadWords(path, words, torn) || torn || words.size() < 2u) return;
	const auto hash = static_cast<uint64_t>(words[words.size() - 2u]) | (static_cast<uint64_t>(words.back()) << 32u);
	if (XXH3_64bits(words.data(), (words.size() - 2u) * 4u) != hash) return;
	Reader      in(std::span(words).first(words.size() - 2u));
	std::string magic(WarmMagic.size(), '\0');
	if (!in.Bytes(magic.data(), magic.size()) || magic != WarmMagic || in.U32() != WarmVersion ||
	    in.Text() != signature) {
		return;
	}
	// The cache shrank since: another run started it over, and the entries no longer say what it holds.
	if (in.U64() > cache_bytes) return;
	const auto sections = in.U32();
	for (uint32_t i = 0; in.Ok() && i < sections; i++) {
		const auto origin = in.U64();
		const auto count  = in.U64();
		if (!in.Ok() || count > words.size()) return;
		std::vector<uint64_t> entries(count);
		for (auto& entry: entries) entry = in.U64();
		if (!in.Ok()) return;
		if (origin == Origin()) {
			m_saved.insert(entries.begin(), entries.end());
		} else if (m_older.size() < 2u) {
			m_older.emplace_back(origin, std::move(entries));
		}
	}
}

bool WarmList::Contains(uint64_t key, bool graphics) const {
	std::lock_guard lock(m_mutex);
	return m_saved.contains(Entry(key, graphics));
}

void WarmList::Mark(uint64_t key, bool graphics) {
	std::lock_guard lock(m_mutex);
	m_marked.push_back(Entry(key, graphics));
}

std::vector<uint64_t> WarmList::TakeMarked() {
	std::lock_guard lock(m_mutex);
	return std::exchange(m_marked, {});
}

void WarmList::Commit(std::vector<uint64_t> entries, bool written, uint64_t cache_bytes) {
	std::lock_guard lock(m_mutex);
	if (!written) {
		m_marked.insert(m_marked.end(), entries.begin(), entries.end());
		return;
	}
	m_saved.insert(entries.begin(), entries.end());
	if (m_path.empty()) return;
	Writer file;
	file.Magic(WarmMagic);
	file.U32(WarmVersion);
	file.Text(m_signature);
	file.U64(cache_bytes);
	file.U32(static_cast<uint32_t>(1u + m_older.size()));
	file.U64(Origin());
	file.U64(m_saved.size());
	for (const auto entry: m_saved) file.U64(entry);
	for (const auto& [origin, older]: m_older) {
		file.U64(origin);
		file.U64(older.size());
		for (const auto entry: older) file.U64(entry);
	}
	file.U64(XXH3_64bits(file.Data().data(), file.Data().size() * 4u));
	WriteWordsAtomically(m_path, file.Data());
}

size_t WarmList::Size() const {
	std::lock_guard lock(m_mutex);
	return m_saved.size();
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
void LowerCurrentThreadPriority() {
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
}
void* CurrentThreadHandle() {
	return OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, GetCurrentThreadId());
}
void SetThreadPriorityNormal(void* handle, bool normal) {
	if (handle != nullptr) SetThreadPriority(handle, normal ? THREAD_PRIORITY_NORMAL : THREAD_PRIORITY_BELOW_NORMAL);
}
void CloseThreadHandle(void* handle) {
	if (handle != nullptr) CloseHandle(handle);
}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
void LowerCurrentThreadPriority() {
	setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 5);
}
void* CurrentThreadHandle() {
	return reinterpret_cast<void*>(static_cast<intptr_t>(syscall(SYS_gettid)));
}
void SetThreadPriorityNormal(void* handle, bool normal) {
	setpriority(PRIO_PROCESS, static_cast<id_t>(reinterpret_cast<intptr_t>(handle)), normal ? 0 : 5);
}
void CloseThreadHandle(void* /*handle*/) {}
#else
void  LowerCurrentThreadPriority() {}
void* CurrentThreadHandle() {
	return nullptr;
}
void SetThreadPriorityNormal(void* /*handle*/, bool /*normal*/) {}
void CloseThreadHandle(void* /*handle*/) {}
#endif

namespace {

std::filesystem::path g_inflight_path;
// The exit handlers run with workers still going: once cleared, no worker writes the file again.
std::mutex g_inflight_exit_mutex;
bool       g_inflight_cleared = false;

void ClearInflight() {
	std::lock_guard lock(g_inflight_exit_mutex);
	g_inflight_cleared = true;
	if (!g_inflight_path.empty()) {
		std::error_code error;
		std::filesystem::remove(g_inflight_path, error);
	}
}

std::vector<std::pair<uint64_t, uint64_t>> ReadKeyLines(const std::filesystem::path& path) {
	std::vector<std::pair<uint64_t, uint64_t>> lines;
	std::ifstream                              file(path);
	std::string                                origin, key;
	while (file >> origin >> key) {
		lines.emplace_back(std::strtoull(origin.c_str(), nullptr, 16), std::strtoull(key.c_str(), nullptr, 16));
	}
	return lines;
}

} // namespace

std::unordered_set<uint64_t> LoadSuspects(const std::filesystem::path& inflight, const std::filesystem::path& suspects) {
	auto left = ReadKeyLines(inflight);
	if (!left.empty()) {
		std::ofstream file(suspects, std::ios::app);
		for (const auto& [origin, key]: left) {
			file << fmt::format("{:016x} {:016x}\n", origin, key);
			Log::WriteToConsoleAndLog(fmt::format("SHADER_PRECOMPILE suspect {:016x}: a run ended while it compiled\n", key));
		}
	}
	std::error_code error;
	std::filesystem::remove(inflight, error);
	// One end of a run proves little: every other job in flight is listed with the one that crashed.
	std::unordered_map<uint64_t, uint32_t> ends;
	std::unordered_set<uint64_t>           keys;
	for (const auto& [origin, key]: ReadKeyLines(suspects)) {
		if (origin == Origin() && ++ends[key] == 2) keys.insert(key);
	}
	return keys;
}

void RegisterInflightCleanup(const std::filesystem::path& inflight) {
	static bool registered = false;
	g_inflight_path        = inflight;
	if (!registered) {
		registered = true;
		at_quick_exit(ClearInflight);
		std::atexit(ClearInflight);
	}
}

static void DropCode(StageRecord& record) {
	std::vector<uint32_t>().swap(record.code);
	std::vector<uint32_t>().swap(record.back_code);
	std::vector<std::vector<uint32_t>>().swap(record.callees);
}

// Options::predict_volume_pixels: a pixel record's specialization as a later draw of the shader may want it. The game
// binds a plain 2D image and an empty buffer, and later a 3D image and a buffer of stride 4 (a guess at what it is: a
// placeholder swapped for a real volume texture). False: not a record of that form.
static bool PredictVolumeForm(ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	using Image = ShaderRecompiler::IR::ResourceSpecialization::Image;
	if (specialization.buffers.empty()) return false;
	auto& buffer = specialization.buffers.back();
	if (buffer.packed_stride != 0 || !buffer.zero_stride_oob || !buffer.scalar_empty) return false;
	const Image plain {};
	for (auto image = specialization.images.rbegin(); image != specialization.images.rend(); ++image) {
		if (image->numeric_class != decltype(Image::numeric_class)::Float ||
		    image->dimension != decltype(Image::dimension)::Dim2D || image->mip_count != plain.mip_count ||
		    image->conversion_format != plain.conversion_format || image->shader_swizzle != plain.shader_swizzle ||
		    image->indirect_root != plain.indirect_root) {
			continue;
		}
		image->dimension       = decltype(Image::dimension)::Dim3D;
		buffer.packed_stride   = 4;
		buffer.zero_stride_oob = false;
		buffer.scalar_empty    = false;
		return true;
	}
	return false;
}

Precompiler::Precompiler(Compiler& compiler, Options options, Journal journal, const WarmList* warm)
    : m_compiler(compiler), m_options(std::move(options)), m_journal(std::move(journal)) {
	for (auto& record: m_journal.stages) {
		if (record.code.empty()) continue;
		m_stages.try_emplace(record.Key()).first->second.record = &record;
	}
	const auto add = [&](Job job) {
		if (m_options.suspects.contains(job.key) ||
		    std::ranges::any_of(job.stages, [&](uint64_t key) { return key != 0 && m_options.suspects.contains(key); })) {
			m_progress.suspects++;
			return;
		}
		if (warm != nullptr && warm->Contains(job.key, job.graphics)) {
			m_progress.warm++;
			if (job.paired || !m_options.reuse || m_options.only) return;
			// A warm pipeline is a driver-cache hit: created only to be handed over to the GPU thread.
			job.create = m_options.hand_over;
		}
		if (!m_by_key.try_emplace(job.key, m_jobs.size()).second) return;
		job.table = job.graphics && std::ranges::any_of(job.stages, [&](uint64_t key) {
			if (key == 0) return false;
			const auto* record = m_stages[key].record;
			return record != nullptr && record->specialization.buffer_table;
		});
		for (const auto key: job.stages) {
			if (key == 0) continue;
			auto& stage = m_stages[key];
			stage.users++;
			stage.keep_code |= job.table;
		}
		(job.graphics ? m_progress.graphics : m_progress.compute)++;
		if (job.paired) m_progress.paired++;
		m_jobs.push_back(job);
	};
	// Compute records and graphics pipelines, each list in journal order (the order a run first needed them),
	// interleaved in proportion: the jobs done then approximate the share of time spent.
	std::vector<Job> compute, graphics;
	for (const auto& record: m_journal.stages) {
		if (record.stage != ShaderType::Compute || record.code.empty()) continue;
		const auto key = record.Key();
		compute.push_back({.key = key, .graphics = false, .stages = {key, 0, 0, 0}});
	}
	uint32_t unresolved = 0;
	for (const auto& record: m_journal.pipelines) {
		const std::array<uint64_t, 4> stages {record.vertex[0], record.vertex[1], record.vertex[2], record.pixel};
		if (std::ranges::any_of(stages, [&](uint64_t key) { return key != 0 && !m_stages.contains(key); })) {
			unresolved++;
			continue;
		}
		graphics.push_back({.key = record.Key(), .graphics = true, .pipeline = &record, .stages = stages});
	}
	// Pairing: a recorded pipeline of a vertex shader in Options::pair_vertex_shaders, with each other vertex record set
	// recorded for that shader in the same state. Such a shader's records differ in specialization, push cursor and
	// indirect draw (Marvel's Wolverine's b1b8b8b4ce99b497: 10 of them): one program, so the pixel stage's inputs are the
	// ones it was recorded with. A run's new pipelines of recorded parts were mostly such swaps (session 14).
	std::vector<Job> paired;
	if (!m_options.pair_vertex_shaders.empty()) {
		const auto shader_of = [&](const PipelineRecord& record) {
			const auto found = m_stages.find(record.vertex[0]);
			return found != m_stages.end() && found->second.record != nullptr ? found->second.record->shader_hash : 0;
		};
		const auto state_of = [](PipelineRecord record) {
			record.vertex = {};
			record.pixel  = 0;
			return record.Key();
		};
		// By state: the vertex record sets recorded in it, with their shader.
		std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, std::array<uint64_t, 3>>>> sets;
		std::unordered_set<uint64_t>                                                         keys;
		for (const auto& job: graphics) {
			const auto shader = shader_of(*job.pipeline);
			if (!m_options.pair_vertex_shaders.contains(shader)) continue;
			auto&      in_state = sets[state_of(*job.pipeline)];
			const auto set      = std::make_pair(shader, job.pipeline->vertex);
			if (std::ranges::find(in_state, set) == in_state.end()) in_state.push_back(set);
			keys.insert(job.key);
		}
		for (const auto& job: graphics) {
			const auto shader = shader_of(*job.pipeline);
			if (job.pipeline->pixel == 0 || !m_options.pair_vertex_shaders.contains(shader)) continue;
			for (const auto& [other, vertex]: sets[state_of(*job.pipeline)]) {
				if (other != shader || vertex == job.pipeline->vertex) continue;
				auto record   = *job.pipeline;
				record.vertex = vertex;
				const auto key = record.Key();
				if (!keys.insert(key).second) continue;
				m_paired.push_back(record);
				paired.push_back({.key      = key,
				                  .graphics = true,
				                  .pipeline = &m_paired.back(),
				                  .stages   = {vertex[0], vertex[1], vertex[2], record.pixel},
				                  .paired   = true});
			}
		}
	}
	// Prediction: each recorded pipeline over the predicted form of its pixel record. A run's new pipelines were mostly
	// such records of known shaders (session 14); a wrong guess costs its compile and no draw changes.
	std::vector<Job> predicted;
	uint64_t         predicted_stages = 0;
	if (m_options.predict_volume_pixels) {
		const auto predict = [&](uint64_t pixel) -> uint64_t {
			const auto* source = m_stages.at(pixel).record;
			if (source == nullptr || source->stage != ShaderType::Pixel) return 0;
			auto specialization = source->specialization;
			if (!PredictVolumeForm(specialization)) return 0;
			auto record           = *source;
			record.specialization = std::move(specialization);
			record.spirv_hash     = 0;
			record.origin         = 0; // no recorded translation to check it against
			const auto key        = record.Key();
			auto&      stage      = m_stages[key];
			if (stage.record == nullptr) {
				m_predicted_stages.push_back(std::move(record));
				stage.record = &m_predicted_stages.back();
				predicted_stages++;
			}
			return key;
		};
		std::unordered_map<uint64_t, uint64_t> pixels; // a recorded pixel stage's predicted one, 0 for none
		std::unordered_set<uint64_t>           keys;
		for (const auto& job: graphics) keys.insert(job.key);
		for (const auto& job: graphics) {
			if (job.pipeline->pixel == 0) continue;
			const auto [found, fresh] = pixels.try_emplace(job.pipeline->pixel, 0);
			if (fresh) found->second = predict(job.pipeline->pixel);
			if (found->second == 0) continue;
			auto record    = *job.pipeline;
			record.pixel   = found->second;
			const auto key = record.Key();
			if (!keys.insert(key).second) continue;
			m_paired.push_back(record);
			predicted.push_back({.key      = key,
			                     .graphics = true,
			                     .pipeline = &m_paired.back(),
			                     .stages   = {record.vertex[0], record.vertex[1], record.vertex[2], record.pixel},
			                     .paired   = true});
		}
	}
	for (size_t c = 0, g = 0; c < compute.size() || g < graphics.size();) {
		const bool take_compute = g == graphics.size() ||
		                          (c < compute.size() && c * graphics.size() <= g * compute.size());
		add(take_compute ? compute[c++] : graphics[g++]);
	}
	// After every recorded job: a prediction and a pairing only guess that a draw will need them.
	for (const auto& job: predicted) add(job);
	const uint64_t predicted_jobs = m_progress.paired.load();
	if (!predicted.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("SHADER_PRECOMPILE predicted: {} pipelines of {} pixel records queued after the recorded jobs\n",
		                                      predicted_jobs, predicted_stages));
	}
	for (const auto& job: paired) add(job);
	if (!paired.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("SHADER_PRECOMPILE paired: {} pipelines of {} vertex shaders queued last\n",
		                                      m_progress.paired.load() - predicted_jobs, m_options.pair_vertex_shaders.size()));
	}
	m_progress.total = m_jobs.size();
	// The code of a stage no job uses is not needed (a journal holds hundreds of MB of it).
	for (auto& [key, stage]: m_stages) {
		if (stage.users == 0 && stage.record != nullptr) DropCode(*stage.record);
	}
	if (unresolved != 0) {
		LOGF("SHADER_PRECOMPILE %u pipelines without their stage records left out\n", unresolved);
	}
}

Precompiler::~Precompiler() {
	Stop();
	for (auto& [key, stage]: m_stages) {
		if (stage.module != nullptr) m_compiler.DestroyStage(*stage.module);
	}
}

void Precompiler::Start() {
	std::lock_guard lock(m_mutex);
	if (!m_workers.empty() || m_jobs.empty()) {
		if (m_jobs.empty()) m_progress.done = true;
		m_progress.started = true;
		return;
	}
	m_start = std::chrono::steady_clock::now();
	const auto threads =
	    std::clamp<uint32_t>(m_options.threads, 1u, static_cast<uint32_t>(std::min<size_t>(m_jobs.size(), 64u)));
	m_progress.threads = threads;
	m_progress.started = true;
	m_worker_handles.assign(threads, nullptr);
	m_live = threads;
	for (uint32_t i = 0; i < threads; i++) m_workers.emplace_back([this, i] { Worker(i); });
}

void Precompiler::Stop() {
	{
		std::lock_guard lock(m_mutex);
		m_stop = true;
	}
	m_changed.notify_all();
	for (auto& worker: m_workers) {
		if (worker.joinable()) worker.join();
	}
	m_workers.clear();
}

void Precompiler::Wait() {
	std::unique_lock lock(m_mutex);
	m_changed.wait(lock, [&] { return m_live == 0; });
}

bool Precompiler::Finished() {
	std::lock_guard lock(m_mutex);
	return m_live == 0;
}

// The running keys as of the caller's last change (NoteInflightLocked) or later, written before it returns; without
// m_mutex, which the GPU thread takes for every pipeline (BeforeCreate): with the file's open under it, Marvel's
// Wolverine's boot spent ~5 ms a pipeline waiting for it, ~1 s for 155. A change another writer wrote returns at once.
void Precompiler::WriteInflight() {
	if (m_options.inflight.empty()) return;
	std::lock_guard file_lock(m_inflight_mutex);
	std::string     text;
	{
		std::lock_guard lock(m_mutex);
		if (m_inflight_written == m_inflight_version) return;
		m_inflight_written = m_inflight_version;
		for (const auto key: m_running_keys) text += fmt::format("{:016x} {:016x}\n", Origin(), key);
	}
	std::lock_guard exit_lock(g_inflight_exit_mutex);
	if (g_inflight_cleared) return;
	std::ofstream file(m_options.inflight, std::ios::trunc);
	file << text;
}

void Precompiler::ReleaseStageLocked(uint64_t key) {
	auto& stage = m_stages[key];
	if (stage.users == 0 || --stage.users != 0) return;
	if (stage.module != nullptr) {
		m_compiler.DestroyStage(*stage.module);
		stage.module.reset();
	}
	if (stage.keep_code && stage.record != nullptr) DropCode(*stage.record);
}

void Precompiler::ReleaseStagesLocked(const Job& job) {
	for (const auto key: job.stages) {
		if (key != 0) ReleaseStageLocked(key);
	}
}

StageModule* Precompiler::AcquireStage(uint64_t key) {
	std::unique_lock lock(m_mutex);
	auto&            stage = m_stages[key];
	m_changed.wait(lock, [&] { return !stage.busy; });
	if (stage.ready || stage.failed) return stage.failed ? nullptr : stage.module.get();
	stage.busy = true;
	m_running_keys.insert(key);
	NoteInflightLocked();
	lock.unlock();
	WriteInflight();
	auto module = std::make_unique<StageModule>();
	bool ok     = false;
	try {
		Common::ScopedExitThrows throws;
		ok = m_compiler.CompileStage(*stage.record, m_options.reuse, *module);
	} catch (const Common::ExitException& error) {
		LOGF("SHADER_PRECOMPILE gave up stage %016" PRIx64 ": %s\n", key, error.message.c_str());
	}
	if (ok && stage.record->origin == Origin() && module->spirv_hash != stage.record->spirv_hash) {
		m_progress.mismatches++;
		Log::WriteToConsoleAndLog(fmt::format("SHADER_PRECOMPILE_MISMATCH stage {:016x} shader {:016x}\n", key,
		                                      stage.record->shader_hash));
	}
	if (!stage.keep_code) DropCode(*stage.record); // translated once: never again
	lock.lock();
	stage.busy = false;
	m_running_keys.erase(key);
	if (ok) {
		stage.module = std::move(module);
		stage.ready  = true;
	} else {
		stage.failed = true;
		if (module->module != nullptr) m_compiler.DestroyStage(*module);
	}
	NoteInflightLocked();
	auto* const result = ok ? stage.module.get() : nullptr;
	lock.unlock();
	m_changed.notify_all();
	WriteInflight();
	return result;
}

// A table job's stage at the push cursor the draw gives it: its record when the draw's key is the record's, else a
// copy at that cursor (NoStart when the stage's push data does not fit there, as Record keys it; the push data size
// is the same at every cursor). draw_key: the key the draw records. A copy is one more translation, held in extra.
StageModule* Precompiler::AcquireAt(uint64_t key, uint32_t cursor, uint64_t& draw_key, std::vector<uint64_t>& extra) {
	auto* const module = AcquireStage(key);
	if (module == nullptr) return nullptr;
	using ShaderRecompiler::IR::PushData;
	const auto  want = PushData::StartFor(cursor, module->info.bindings.ShaderDataDwords()) == PushData::NoStart
	                       ? PushData::NoStart
	                       : cursor;
	StageRecord copy;
	{
		std::lock_guard lock(m_mutex);
		const auto*     record = m_stages.at(key).record;
		// A record without its code (not a table job's) or with calls stays as recorded.
		if (record->push_data_cursor == want || record->code.empty() || !record->callees.empty()) {
			draw_key = key;
			return module;
		}
		copy = *record;
	}
	copy.push_data_cursor = want;
	copy.spirv_hash       = 0;
	copy.origin           = 0; // no recorded translation to check it against
	draw_key              = copy.Key();
	{
		std::lock_guard lock(m_mutex);
		auto&           stage = m_stages[draw_key];
		// New, or a stage whose code and module are gone (no job left used it): translated again from the copy.
		if (stage.record == nullptr ||
		    (!stage.busy && !stage.failed && stage.module == nullptr && stage.record->code.empty())) {
			m_predicted_stages.push_back(std::move(copy));
			stage.record = &m_predicted_stages.back();
			stage.ready  = false;
		}
		stage.users++;
		extra.push_back(draw_key);
	}
	return AcquireStage(draw_key);
}

void Precompiler::Run(Job& job, uint32_t /*index*/) {
	if (!job.graphics) {
		auto* stage = AcquireStage(job.stages[0]);
		{
			std::lock_guard lock(m_mutex);
			job.creating = job.create && stage != nullptr;
		}
		if (!job.creating) {
			if (stage == nullptr) m_progress.failed++;
			return;
		}
		const auto start = std::chrono::steady_clock::now();
		bool       ok    = false;
		try {
			Common::ScopedExitThrows throws;
			ok = m_compiler.CreateCompute(job.key, *stage);
		} catch (const Common::ExitException& error) {
			LOGF("SHADER_PRECOMPILE gave up compute %016" PRIx64 ": %s\n", job.key, error.message.c_str());
		}
		if (!ok) {
			m_progress.failed++;
			return;
		}
		if (std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(10)) m_progress.compiled++;
		if (on_created) on_created(job.key, false);
		std::lock_guard lock(m_mutex);
		job.created = true;
		return;
	}
	std::array<StageModule*, 3> vertex {};
	StageModule*                pixel = nullptr;
	uint32_t                    vertex_count = 0;
	bool                        ok           = true;
	PipelineRecord              pipeline     = *job.pipeline; // its stage keys as the draw records them
	std::vector<uint64_t>       extra;                         // copies AcquireAt made for this job
	const auto release_extra = [&] {
		std::lock_guard lock(m_mutex);
		for (const auto key: extra) ReleaseStageLocked(key);
	};
	if (!job.table) {
		for (uint32_t i = 0; i < 3 && ok; i++) {
			if (job.stages[i] == 0) continue;
			vertex[vertex_count++] = AcquireStage(job.stages[i]);
			ok                     = vertex[vertex_count - 1] != nullptr;
		}
		if (ok && job.stages[3] != 0) {
			pixel = AcquireStage(job.stages[3]);
			ok    = pixel != nullptr;
		}
	} else {
		// The buffer table grows a stage's push data (2 dwords a buffer, or past the 32), and MigrateBufferTable cannot
		// move the next stages' cursors: the size needs the translation. Here each stage goes where GetGraphicsPrograms
		// puts it, the pixel stage first, so its key and SPIR-V are the draw's own.
		using ShaderRecompiler::IR::PushData;
		uint32_t cursor = 0; // where the vertex stages' push data starts without the pixel stage's (PrepareVertexStages)
		{
			std::lock_guard lock(m_mutex);
			const auto*     first = std::get_if<ShaderVertexInputInfo>(&m_stages.at(job.stages[0]).record->input);
			const auto      last  = job.stages[2] != 0 ? job.stages[2] : job.stages[1] != 0 ? job.stages[1] : job.stages[0];
			const auto*     tail  = std::get_if<ShaderVertexInputInfo>(&m_stages.at(last).record->input);
			const auto*     pixel_input =
			    job.stages[3] != 0 ? std::get_if<ShaderPixelInputInfo>(&m_stages.at(job.stages[3]).record->input) : nullptr;
			ok = first != nullptr && tail != nullptr && (job.stages[3] == 0 || pixel_input != nullptr);
			if (ok) {
				cursor = PushData::GraphicsDrawDwords(first->logical_stage == ShaderType::Mesh,
				                                     pixel_input != nullptr && pixel_input->object_primitive_id,
				                                     tail->user_clip_planes);
			}
		}
		if (ok && job.stages[3] != 0) {
			pixel = AcquireAt(job.stages[3], PushData::MeshDrawDwordCount, pipeline.pixel, extra);
			ok    = pixel != nullptr;
			uint32_t pixel_cursor = PushData::MeshDrawDwordCount;
			if (ok) pixel->info.bindings.AdvancePushData(pixel_cursor);
			if (pixel_cursor != PushData::MeshDrawDwordCount) cursor = pixel_cursor;
		}
		for (uint32_t i = 0; i < 3 && ok; i++) {
			if (job.stages[i] == 0) continue;
			auto* const stage      = AcquireAt(job.stages[i], cursor, pipeline.vertex[i], extra);
			vertex[vertex_count++] = stage;
			ok                     = stage != nullptr;
			if (ok) stage->info.bindings.AdvancePushData(cursor);
		}
	}
	const auto key = pipeline.Key();
	{
		std::lock_guard lock(m_mutex);
		// The GPU thread asks for it by the key it records (BeforeCreate).
		if (ok && key != job.key) {
			m_by_key.try_emplace(key, static_cast<size_t>(&job - m_jobs.data()));
			m_progress.recursored++;
		}
		job.creating = job.create && ok;
	}
	if (!job.creating) {
		if (!ok) m_progress.failed++;
		release_extra();
		return;
	}
	const auto start   = std::chrono::steady_clock::now();
	bool       created = false;
	try {
		Common::ScopedExitThrows throws;
		created = m_compiler.CreateGraphics(pipeline, std::span(vertex.data(), vertex_count), pixel, !job.paired);
	} catch (const Common::ExitException& error) {
		LOGF("SHADER_PRECOMPILE gave up pipeline %016" PRIx64 ": %s\n", key, error.message.c_str());
	}
	release_extra();
	if (!created) {
		m_progress.failed++;
		return;
	}
	if (std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(10)) m_progress.compiled++;
	if (on_created) on_created(key, true);
	std::lock_guard lock(m_mutex);
	job.created = true;
}

void Precompiler::Hurry() {
	if (m_hurry.exchange(true)) return;
	std::lock_guard lock(m_mutex);
	if (m_next >= m_jobs.size()) return;
	LOGF("SHADER_PRECOMPILE hurried: %zu of %zu jobs left, the GPU thread compiles itself\n", m_jobs.size() - m_next,
	     m_jobs.size());
	for (auto* handle: m_worker_handles) {
		if (handle != nullptr) SetThreadPriorityNormal(handle, true);
	}
}

void Precompiler::Worker(uint32_t index) {
	if (!m_hurry) LowerCurrentThreadPriority();
	{
		std::lock_guard lock(m_mutex);
		m_worker_handles[index] = CurrentThreadHandle();
	}
	for (;;) {
		Job* job = nullptr;
		{
			std::unique_lock lock(m_mutex);
			while (m_next < m_jobs.size() && m_jobs[m_next].state != State::Queued) m_next++;
			if (m_stop || m_next == m_jobs.size()) break;
			job         = &m_jobs[m_next++];
			job->state  = State::Running;
			job->worker = index;
			m_running_keys.insert(job->key);
			NoteInflightLocked();
		}
		WriteInflight();
		// The GPU thread may have raised this worker while it waited on its last job; once hurried, all stay raised.
		if (!m_hurry) LowerCurrentThreadPriority();
		Run(*job, index);
		{
			std::lock_guard lock(m_mutex);
			job->state = State::Done;
			m_running_keys.erase(job->key);
			ReleaseStagesLocked(*job);
			m_progress.finished++;
			(job->graphics ? m_progress.graphics_finished : m_progress.compute_finished)++;
			NoteInflightLocked();
		}
		m_changed.notify_all();
		WriteInflight();
	}
	std::lock_guard lock(m_mutex);
	CloseThreadHandle(m_worker_handles[index]);
	m_worker_handles[index] = nullptr;
	if (--m_live == 0) {
		const bool complete = m_next == m_jobs.size();
		m_progress.done     = complete;
		if (complete) {
			for (auto& [key, stage]: m_stages) {
				if (stage.module != nullptr) {
					m_compiler.DestroyStage(*stage.module);
					stage.module.reset();
				}
			}
			std::error_code error;
			if (!m_options.inflight.empty()) std::filesystem::remove(m_options.inflight, error);
		}
		const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count();
		Log::WriteToConsoleAndLog(fmt::format(
		    "SHADER_PRECOMPILE {}: {} of {} jobs in {:.1f} s ({} compiled, {} dropped for the GPU thread, {} failed, "
		    "{} mismatches, {} warm, {} suspects, {} moved to the draw's push cursors)\n",
		    complete ? "done" : "stopped", m_progress.finished.load(), m_jobs.size(), seconds, m_progress.compiled.load(),
		    m_progress.dropped.load(), m_progress.failed.load(), m_progress.mismatches.load(), m_progress.warm.load(),
		    m_progress.suspects.load(), m_progress.recursored.load()));
	}
	m_changed.notify_all();
}

bool Precompiler::BeforeCreate(uint64_t key) {
	std::unique_lock lock(m_mutex);
	const auto       found = m_by_key.find(key);
	if (found == m_by_key.end()) return false;
	auto& job = m_jobs[found->second];
	if (job.state == State::Queued) {
		job.state = State::Dropped;
		ReleaseStagesLocked(job);
		m_progress.dropped++;
		m_progress.finished++;
		(job.graphics ? m_progress.graphics_finished : m_progress.compute_finished)++;
		return false;
	}
	if (job.state != State::Running) return job.created;
	// Still translating its stages: the GPU thread has its own modules and creates the pipeline itself.
	if (!job.creating) {
		job.create = false;
		return false;
	}
	// The driver is creating it: a second create would cost more than the wait.
	const auto start = std::chrono::steady_clock::now();
	SetThreadPriorityNormal(m_worker_handles[job.worker], true);
	m_changed.wait(lock, [&] { return job.state != State::Running; });
	LOGF("SHADER_PRECOMPILE waited %" PRId64 " ms for %016" PRIx64 "\n",
	     static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()),
	     key);
	return job.created;
}

} // namespace Libs::Graphics::Precompile
