// kyty_shader_precompile --make-hints and --apply-hints: specialization hints (shader-hints.h).
#include "shader-hints.h"

#include "common/file.h"
#include "shader-warmup-cache.h"
#include "static-seeds.h"

#include <nlohmann/json.hpp>
#include <xxhash.h>
#include <zstd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ShaderHints {

namespace {

namespace fs = std::filesystem;
using Lists  = std::vector<std::vector<uint32_t>>;
using LocalShaderWarmup::Cache;
using LocalShaderWarmup::NoShader;
using LocalShaderWarmup::Reader;
using LocalShaderWarmup::Record;
using LocalShaderWarmup::Writer;
using Libs::Graphics::ShaderType;

constexpr std::string_view Identity = "KytyShaderHints1:";
// A hint file's body, uncompressed (a recorded seed file's records without code: ~13 MiB for 15000 pipelines).
constexpr size_t MaxBodyBytes = Cache::MaxBytes;

std::string Utf8(const fs::path& path) {
	const auto text = path.u8string();
	return {reinterpret_cast<const char*>(text.data()), text.size()};
}

uint64_t Hash(const std::vector<uint32_t>& words) {
	return XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
}

// A program's code as a hint names it: the stage, and the code's and the back code's word counts and XXH3-64s.
struct Code {
	uint32_t stage = 0, words = 0, back_words = 0;
	uint64_t hash = 0, back_hash = 0;
	bool     operator==(const Code&) const = default;
};
struct CodeHash {
	size_t operator()(const Code& code) const { return code.hash ^ (code.back_hash * 0x9e3779b97f4a7c15ull) ^ code.stage; }
};

Code CodeOf(const Record& record) {
	return {static_cast<uint32_t>(record.stage), static_cast<uint32_t>(record.code.size()),
	        static_cast<uint32_t>(record.back_code.size()), Hash(record.code), Hash(record.back_code)};
}

bool Decode(const std::vector<uint32_t>& words, Record& record) {
	Reader reader {words};
	return LocalShaderWarmup::Visit(reader, record) && reader.cursor == words.size();
}

std::vector<uint32_t> Encode(Record& record) {
	Writer writer;
	LocalShaderWarmup::Visit(writer, record);
	return std::move(writer.words);
}

// A record's hint: its code and back code as [word count, XXH3-64 low, high], no static key.
std::vector<uint32_t> HintOf(Record record) {
	const auto code   = CodeOf(record);
	record.code       = {code.words, static_cast<uint32_t>(code.hash), static_cast<uint32_t>(code.hash >> 32u)};
	record.back_code  = {code.back_words, static_cast<uint32_t>(code.back_hash), static_cast<uint32_t>(code.back_hash >> 32u)};
	record.static_key = {};
	return Encode(record);
}

// The code a hint names, or nullopt where its record does not decode.
std::optional<Code> NamedCode(const std::vector<uint32_t>& hint, Record& record) {
	if (!Decode(hint, record) || record.code.size() != 3 || record.back_code.size() != 3) return std::nullopt;
	return Code {static_cast<uint32_t>(record.stage), record.code[0], record.back_code[0],
	             record.code[1] | (uint64_t(record.code[2]) << 32u), record.back_code[1] | (uint64_t(record.back_code[2]) << 32u)};
}

// The record a hint stands for: its own fields with the program's code and the static key of its input info.
std::vector<uint32_t> Completed(Record& hint, const Record& program) {
	hint.code      = program.code;
	hint.back_code = program.back_code;
	if (hint.stage == ShaderType::Vertex || hint.stage == ShaderType::Mesh) {
		BuildStageStaticKey(hint.vertex, hint.static_key);
	} else if (hint.stage == ShaderType::Pixel) {
		BuildStageStaticKey(hint.pixel, hint.static_key);
	} else {
		BuildStageStaticKey(hint.compute, hint.static_key);
	}
	return Encode(hint);
}

// Entries kept once, in the order first added (precompile.py SeedFile's lists).
struct Unique {
	Lists                                        entries;
	std::unordered_multimap<uint64_t, uint32_t> index;

	uint32_t Add(std::vector<uint32_t> words) {
		const auto hash         = Hash(words);
		const auto [begin, end] = index.equal_range(hash);
		for (auto it = begin; it != end; ++it) {
			if (entries[it->second] == words) return it->second;
		}
		const auto id = static_cast<uint32_t>(entries.size());
		index.emplace(hash, id);
		entries.push_back(std::move(words));
		return id;
	}
};

// A pipeline recipe with its shaders (its first three words) renumbered; false where one of them is not kept.
bool Renumber(std::vector<uint32_t>& pipeline, const std::vector<uint32_t>& numbers) {
	if (pipeline.size() < 3) return false;
	for (size_t i = 0; i < 3; i++) {
		if (pipeline[i] == NoShader) continue;
		if (pipeline[i] >= numbers.size() || numbers[pipeline[i]] == NoShader) return false;
		pipeline[i] = numbers[pipeline[i]];
	}
	return true;
}

// The programs of a seed file by their code (the first record of each).
bool ReadPrograms(const fs::path& seeds, Lists& records, std::unordered_map<Code, uint32_t, CodeHash>& programs) {
	Lists pipelines;
	if (!Cache::ReadSeeds(seeds, records, pipelines)) {
		std::fprintf(stderr, "%s: not a seed file (kyty_shader_precompile --make-seeds)\n", Utf8(seeds).c_str());
		return false;
	}
	for (uint32_t i = 0; i < records.size(); i++) {
		Record record;
		if (Decode(records[i], record)) programs.emplace(CodeOf(record), i);
	}
	return true;
}

std::vector<uint32_t> Body(const Lists& records, const Lists& pipelines) {
	std::vector<uint32_t> body;
	for (const auto* list: {&records, &pipelines}) {
		body.push_back(static_cast<uint32_t>(list->size()));
		for (const auto& words: *list) {
			body.push_back(static_cast<uint32_t>(words.size()));
			body.insert(body.end(), words.begin(), words.end());
		}
	}
	return body;
}

bool ParseBody(const std::vector<uint32_t>& body, Lists& records, Lists& pipelines) {
	Reader reader {body};
	for (auto* list: {&records, &pipelines}) {
		uint32_t count = 0;
		reader(count);
		if (!reader.Good() || count > Cache::MaxPipelines) return false;
		list->resize(count);
		for (auto& words: *list) {
			if (!LocalShaderWarmup::Words(reader, words, 1024 * 1024)) return false;
		}
	}
	return reader.Good() && reader.cursor == body.size();
}

const char* StageName(uint32_t stage) {
	switch (static_cast<ShaderType>(stage)) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Compute: return "compute";
		case ShaderType::Mesh: return "mesh";
		default: return "other";
	}
}

std::string ByStage(const Lists& records) {
	std::map<uint32_t, size_t> counts;
	for (const auto& words: records) counts[words[0]]++;
	std::string text;
	for (const auto& [stage, count]: counts) {
		text += (text.empty() ? "" : ", ") + std::to_string(count) + " " + StageName(stage);
	}
	return text;
}

} // namespace

std::string GameId(const fs::path& game) {
	Common::File file;
	if (!file.Open(game / "sce_sys" / "param.json", Common::File::Mode::Read) || file.Size() > 16 * 1024 * 1024) return {};
	std::string text(static_cast<size_t>(file.Size()), '\0');
	uint32_t    read = 0;
	if (!text.empty()) file.Read(text.data(), static_cast<uint32_t>(text.size()), &read);
	if (read != text.size()) return {};
	const auto param = nlohmann::json::parse(text, nullptr, false);
	if (param.is_discarded() || !param.is_object() || !param.contains("titleId") || !param.contains("contentVersion") ||
	    !param["titleId"].is_string() || !param["contentVersion"].is_string())
		return {};
	return param["titleId"].get<std::string>() + "_" + param["contentVersion"].get<std::string>();
}

int Make(const fs::path& out, const fs::path& seeds, const std::vector<fs::path>& recordings, const std::string& game_id) {
	Lists                                        seed_records;
	std::unordered_map<Code, uint32_t, CodeHash> programs;
	if (game_id.empty() || !ReadPrograms(seeds, seed_records, programs)) return 1;
	Unique hints, pipelines;
	size_t other_records = 0, other_pipelines = 0;
	for (const auto& recording: recordings) {
		Lists records, recipes;
		if (!Cache::ReadInputs(recording, records, recipes)) {
			std::fprintf(stderr, "%s: not a warmup recording or a seed file\n", Utf8(recording).c_str());
			return 1;
		}
		std::vector<uint32_t> numbers(records.size(), NoShader);
		for (size_t i = 0; i < records.size(); i++) {
			Record record;
			if (!Decode(records[i], record) || !programs.contains(CodeOf(record))) {
				other_records++;
				continue;
			}
			numbers[i] = hints.Add(HintOf(std::move(record)));
		}
		for (auto& recipe: recipes) {
			if (Renumber(recipe, numbers)) {
				pipelines.Add(std::move(recipe));
			} else {
				other_pipelines++;
			}
		}
	}
	const auto body = Body(hints.entries, pipelines.entries);
	const auto size = body.size() * sizeof(uint32_t);
	std::vector<char> packed(ZSTD_compressBound(size));
	const auto        packed_size = ZSTD_compress(packed.data(), packed.size(), body.data(), size, 19);
	if (ZSTD_isError(packed_size) != 0 || size > MaxBodyBytes) {
		std::fprintf(stderr, "cannot compress the hints\n");
		return 1;
	}
	const std::string identity = std::string(Identity) + game_id + "\n";
	const uint64_t    checksum = XXH3_64bits(body.data(), size);
	std::error_code   error;
	if (out.has_parent_path()) fs::create_directories(out.parent_path(), error);
	auto temporary = out;
	temporary += ".tmp";
	std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
	file.write(identity.data(), static_cast<std::streamsize>(identity.size()));
	file.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
	file.write(packed.data(), static_cast<std::streamsize>(packed_size));
	file.close();
	if (file) fs::rename(temporary, out, error);
	if (!file || error) {
		std::fprintf(stderr, "cannot write %s\n", Utf8(out).c_str());
		return 1;
	}
	std::printf("%s: %zu programs (%s), %zu pipelines; %.1f MiB, %.0f KiB compressed; not this version's (not in the "
	            "seed file): %zu records, %zu pipelines\n",
	            Utf8(out).c_str(), hints.entries.size(), ByStage(hints.entries).c_str(), pipelines.entries.size(),
	            static_cast<double>(size) / (1024.0 * 1024.0), static_cast<double>(packed_size) / 1024.0, other_records,
	            other_pipelines);
	return 0;
}

int Apply(const fs::path& hints, const fs::path& seeds, const fs::path& out, const std::string& game_id) {
	// The hint file: identity line, checksum, one zstd frame.
	const auto identity = Cache::FileIdentity(hints);
	if (!identity.starts_with(Identity) || (!game_id.empty() && identity != std::string(Identity) + game_id + "\n")) {
		std::fprintf(stderr, "%s: not a hint file of %s\n", Utf8(hints).c_str(), game_id.c_str());
		return 1;
	}
	std::ifstream file(hints, std::ios::binary | std::ios::ate);
	const auto    file_size = static_cast<std::streamoff>(file.tellg());
	if (!file || file_size < static_cast<std::streamoff>(identity.size() + 8) || file_size > std::streamoff(MaxBodyBytes)) {
		std::fprintf(stderr, "cannot read %s\n", Utf8(hints).c_str());
		return 1;
	}
	uint64_t          checksum = 0;
	std::vector<char> packed(static_cast<size_t>(file_size) - identity.size() - 8);
	file.seekg(static_cast<std::streamoff>(identity.size()));
	file.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
	file.read(packed.data(), static_cast<std::streamsize>(packed.size()));
	const auto size = ZSTD_getFrameContentSize(packed.data(), packed.size());
	if (!file || size == ZSTD_CONTENTSIZE_UNKNOWN || size == ZSTD_CONTENTSIZE_ERROR || size > MaxBodyBytes || size % 4 != 0) {
		std::fprintf(stderr, "%s: damaged\n", Utf8(hints).c_str());
		return 1;
	}
	std::vector<uint32_t> body(static_cast<size_t>(size) / 4);
	Lists                 records, recipes;
	if (ZSTD_decompress(body.data(), static_cast<size_t>(size), packed.data(), packed.size()) != size ||
	    XXH3_64bits(body.data(), static_cast<size_t>(size)) != checksum || !ParseBody(body, records, recipes)) {
		std::fprintf(stderr, "%s: damaged\n", Utf8(hints).c_str());
		return 1;
	}
	// Each hint with the code of the seed file's program it names.
	Lists                                        seed_records;
	std::unordered_map<Code, uint32_t, CodeHash> programs;
	if (!ReadPrograms(seeds, seed_records, programs)) return 1;
	Lists                 completed;
	std::vector<uint32_t> numbers(records.size(), NoShader);
	size_t                missing = 0;
	for (size_t i = 0; i < records.size(); i++) {
		Record     hint;
		const auto code  = NamedCode(records[i], hint);
		const auto found = code ? programs.find(*code) : programs.end();
		Record     program;
		if (found == programs.end() || !Decode(seed_records[found->second], program)) {
			missing++;
			continue;
		}
		numbers[i] = static_cast<uint32_t>(completed.size());
		completed.push_back(Completed(hint, program));
	}
	Lists  pipelines;
	size_t left_out = 0;
	for (auto& recipe: recipes) {
		if (Renumber(recipe, numbers)) {
			pipelines.push_back(std::move(recipe));
		} else {
			left_out++;
		}
	}
	const auto completed_count = completed.size();
	const auto pipeline_count  = pipelines.size();
	if (StaticSeeds::Write(out, std::move(completed), std::move(pipelines)) == 0) return 1;
	// What the precompile reads of it (Cache::ReadSeeds: every record's static key, every recipe's shaders).
	Lists check_records, check_pipelines;
	if (!Cache::ReadSeeds(out, check_records, check_pipelines)) {
		std::fprintf(stderr, "%s: the completed records do not load\n", Utf8(out).c_str());
		std::error_code error;
		fs::remove(out, error);
		return 1;
	}
	std::printf("%s: %zu programs (%s), %zu pipelines from %s; without their code in the seed file: %zu programs, "
	            "%zu pipelines\n",
	            Utf8(out).c_str(), completed_count, ByStage(check_records).c_str(), pipeline_count, Utf8(hints).c_str(),
	            missing, left_out);
	return 0;
}

} // namespace ShaderHints
