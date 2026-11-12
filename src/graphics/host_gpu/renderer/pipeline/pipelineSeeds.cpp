#include "graphics/host_gpu/renderer/pipeline/pipelineSeeds.h"

#include "common/logging/log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <xxhash.h>

namespace Libs::Graphics::PipelineSeeds {

namespace fs = std::filesystem;

namespace {

constexpr char     FileMagic[8]  = {'K', 'Y', 'G', 'S', 'E', 'E', 'D', '1'};
constexpr size_t   HeaderSize    = sizeof(FileMagic) + sizeof(uint64_t);
constexpr uint32_t MaxRecordSize = 64u << 20u;

struct State {
	std::mutex                             mutex;
	fs::path                               directory;
	uint64_t                               format    = 0;
	bool                                   recording = false;
	std::unordered_set<uint64_t>           records; // hashes in guest.seeds
	std::unordered_map<uint64_t, uint64_t> programs; // program id -> record hash
	uint32_t                               new_records = 0;
	std::unordered_set<uint64_t>           sources; // SourceId of the file's program records

	std::deque<std::pair<uint64_t, std::vector<uint8_t>>> queue;
	std::condition_variable                               wake, idle;
	bool                                                  writing = false;
	bool                                                  writer_started = false;
};

State& S() {
	static State s;
	return s;
}

fs::path SeedsPath(const fs::path& dir) {
	return dir / "guest.seeds";
}

void WriterLoop() {
	auto& s = S();
	for (;;) {
		std::pair<uint64_t, std::vector<uint8_t>> job;
		{
			std::unique_lock lock(s.mutex);
			s.wake.wait(lock, [&] { return !s.queue.empty(); });
			job = std::move(s.queue.front());
			s.queue.pop_front();
			s.writing = true;
		}
		const auto      path = SeedsPath(s.directory);
		std::error_code ec;
		const bool      fresh = !fs::exists(path, ec) || fs::file_size(path, ec) == 0;
		{
			std::ofstream out(path, std::ios::binary | std::ios::app);
			if (fresh) {
				out.write(FileMagic, sizeof(FileMagic));
				out.write(reinterpret_cast<const char*>(&s.format), sizeof(s.format));
			}
			const auto size = static_cast<uint32_t>(job.second.size());
			out.write(reinterpret_cast<const char*>(&size), sizeof(size));
			out.write(reinterpret_cast<const char*>(&job.first), sizeof(job.first));
			const uint64_t check = XXH3_64bits(job.second.data(), job.second.size());
			out.write(reinterpret_cast<const char*>(&check), sizeof(check));
			out.write(reinterpret_cast<const char*>(job.second.data()), size);
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

// Under s.mutex.
void Enqueue(uint64_t hash, std::vector<uint8_t> payload) {
	auto& s = S();
	s.queue.emplace_back(hash, std::move(payload));
	if (!s.writer_started) {
		s.writer_started = true;
		std::thread(WriterLoop).detach();
	}
	s.wake.notify_one();
}

enum class FileState { Missing, Read, OtherFormat };

// Calls fn(hash, payload) for each intact record of the file; headers may repeat (joined files).
template <typename F>
FileState ReadSeeds(const fs::path& path, uint64_t format, F&& fn) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		return FileState::Missing;
	}
	const std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	size_t                     at = 0;
	bool                       any_header = false;
	while (at < file.size()) {
		if (at + HeaderSize <= file.size() && std::memcmp(file.data() + at, FileMagic, sizeof(FileMagic)) == 0) {
			uint64_t f = 0;
			std::memcpy(&f, file.data() + at + sizeof(FileMagic), sizeof(f));
			if (f != format) {
				return any_header ? FileState::Read : FileState::OtherFormat;
			}
			any_header = true;
			at += HeaderSize;
			continue;
		}
		if (!any_header || at + 20 > file.size()) {
			break;
		}
		uint32_t size  = 0;
		uint64_t hash  = 0;
		uint64_t check = 0;
		std::memcpy(&size, file.data() + at, 4);
		std::memcpy(&hash, file.data() + at + 4, 8);
		std::memcpy(&check, file.data() + at + 12, 8);
		at += 20;
		if (size == 0 || size > MaxRecordSize || at + size > file.size()) {
			break; // a cut tail (a run that ended while writing)
		}
		std::span<const uint8_t> payload(file.data() + at, size);
		at += size;
		if (XXH3_64bits(payload.data(), payload.size()) == check) {
			fn(hash, payload);
		}
	}
	return any_header ? FileState::Read : (file.empty() ? FileState::Missing : FileState::OtherFormat);
}

fs::path ProgressPath() {
	const char* env = std::getenv("KYTY_PRECOMPILE_PROGRESS");
	return env != nullptr && *env != '\0' ? fs::u8path(env) : fs::path();
}

fs::path StopPath() {
	const auto p = ProgressPath();
	return p.empty() ? p : fs::path(p.native() + fs::path(".stop").native());
}

} // namespace

bool PrecompileRequested() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PRECOMPILE");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

void Open(const fs::path& directory, uint64_t format) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	s.directory     = directory;
	s.format        = format;
	const char* env = std::getenv("KYTY_PIPELINE_SEEDS");
	// Recording goes on in precompile mode too: nothing new is made there, so nothing is written.
	s.recording = !directory.empty() && !(env != nullptr && env[0] == '0');
	if (directory.empty()) {
		return;
	}
	const auto path   = SeedsPath(directory);
	uint32_t   total  = 0;
	const bool miss_check = MissCheck();
	const auto state  = ReadSeeds(path, format, [&](uint64_t hash, std::span<const uint8_t> payload) {
		s.records.insert(hash);
		total++;
		// A program record: tag, u32 stage, u64 shader hash, u32 wave, user data base, push
		// cursor, u8 kind, then the code (u64 count, words).
		constexpr size_t code_at = 1 + 4 + 8 + 4 + 4 + 4 + 1;
		if (miss_check && payload.size() >= code_at + 8 && payload[0] == KindProgram) {
			uint32_t stage = 0;
			uint64_t shader_hash = 0, words = 0;
			std::memcpy(&stage, payload.data() + 1, 4);
			std::memcpy(&shader_hash, payload.data() + 5, 8);
			std::memcpy(&words, payload.data() + code_at, 8);
			if (words <= (payload.size() - code_at - 8) / 4) {
				std::vector<uint32_t> code(static_cast<size_t>(words));
				std::memcpy(code.data(), payload.data() + code_at + 8, code.size() * 4);
				s.sources.insert(SourceId(stage, shader_hash, code));
			}
		}
	});
	if (state == FileState::OtherFormat) {
		// Records of another emulator build's structures: set aside, a new file starts.
		std::error_code ec;
		fs::rename(path, fs::path(path.native() + fs::path(".old").native()), ec);
		LOGF("Pipeline seeds: %s is of another emulator build; set aside as .old\n", path.string().c_str());
	}
	LOGF("Pipeline seeds: %u guest records in %s; recording %s\n", total, path.string().c_str(),
	     s.recording ? "on" : "off (KYTY_PIPELINE_SEEDS=0)");
}

namespace {
std::atomic<uint64_t> g_seeded_misses {0};
std::atomic<uint64_t> g_unseeded_misses {0};
} // namespace

uint64_t SourceId(uint32_t stage, uint64_t shader_hash, std::span<const uint32_t> code) {
	return XXH3_64bits_withSeed(code.data(), code.size_bytes(),
	                            shader_hash ^ (static_cast<uint64_t>(stage) << 56u));
}

bool MissCheck() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_SEED_MISS_CHECK");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

bool HasSource(uint64_t source_id) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	return s.sources.contains(source_id);
}

void NoteSourceMiss(bool seeded) {
	(seeded ? g_seeded_misses : g_unseeded_misses).fetch_add(1, std::memory_order_relaxed);
}

void MissCounts(uint64_t& seeded, uint64_t& unseeded) {
	seeded   = g_seeded_misses.load(std::memory_order_relaxed);
	unseeded = g_unseeded_misses.load(std::memory_order_relaxed);
}

bool Recording() {
	return S().recording;
}

void RecordProgram(uint64_t hash, std::vector<uint8_t> payload, std::span<const uint64_t> ids) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	for (const auto id: ids) {
		if (id != 0) {
			s.programs[id] = hash;
		}
	}
	if (s.records.insert(hash).second) {
		s.new_records++;
		Enqueue(hash, std::move(payload));
	}
}

uint64_t ProgramHash(uint64_t id) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	const auto      it = s.programs.find(id);
	return it == s.programs.end() ? 0 : it->second;
}

void RecordPipeline(std::vector<uint8_t> payload) {
	auto&           s    = S();
	const auto      hash = XXH3_64bits(payload.data(), payload.size());
	std::lock_guard lock(s.mutex);
	if (s.records.insert(hash).second) {
		s.new_records++;
		Enqueue(hash, std::move(payload));
	}
}

void Flush() {
	auto&            s = S();
	std::unique_lock lock(s.mutex);
	s.idle.wait_for(lock, std::chrono::seconds(10), [&] { return s.queue.empty() && !s.writing; });
	if (s.new_records != 0) {
		LOGF("Pipeline seeds: %u new guest records written this run\n", s.new_records);
	}
}

std::vector<Record> LoadAll(uint32_t* other_format) {
	auto&                        s = S();
	std::vector<Record>          out;
	std::unordered_set<uint64_t> seen;
	std::vector<fs::path>        files;
	if (!s.directory.empty()) {
		files.push_back(SeedsPath(s.directory));
	}
	if (const char* extra = std::getenv("KYTY_PRECOMPILE_SEEDS"); extra != nullptr) {
		std::string list = extra;
		size_t      at   = 0;
		while (at <= list.size()) {
			const auto end = std::min(list.find(';', at), list.size());
			if (end > at) {
				files.push_back(fs::u8path(list.substr(at, end - at)));
			}
			at = end + 1;
		}
	}
	for (const auto& file: files) {
		uint32_t   count = 0;
		const auto state = ReadSeeds(file, s.format, [&](uint64_t hash, std::span<const uint8_t> payload) {
			if (seen.insert(hash).second) {
				out.push_back({hash, std::vector<uint8_t>(payload.begin(), payload.end())});
				count++;
			}
		});
		if (state == FileState::OtherFormat && other_format != nullptr) {
			(*other_format)++;
		}
		LOGF("Pipeline seeds: %s: %s, %u new records\n", file.string().c_str(),
		     state == FileState::Missing       ? "not found"
		     : state == FileState::OtherFormat ? "another emulator build's format, skipped"
		                                       : "read",
		     count);
	}
	return out;
}

void WriteProgress(uint32_t done, uint32_t total, uint32_t compiled, uint32_t failed, bool final) {
	const auto progress = ProgressPath();
	if (progress.empty()) {
		return;
	}
	const auto tmp = fs::path(progress.native() + fs::path(".tmp").native());
	{
		std::ofstream out(tmp, std::ios::trunc);
		out << done << ' ' << total << ' ' << compiled << ' ' << failed << ' ' << (final ? 1 : 0) << '\n';
	}
	std::error_code ec;
	fs::rename(tmp, progress, ec);
}

bool StopRequested() {
	const auto      stop = StopPath();
	std::error_code ec;
	return !stop.empty() && fs::exists(stop, ec);
}

void RemoveStopFile() {
	const auto      stop = StopPath();
	std::error_code ec;
	if (!stop.empty()) {
		fs::remove(stop, ec);
	}
}

} // namespace Libs::Graphics::PipelineSeeds
