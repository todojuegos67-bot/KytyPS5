#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERDISKCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERDISKCACHE_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// The shader disk cache: translations of guest shaders kept on disk between runs, so that a warm
// start does not translate every shader again (_ShaderCache/<title id>/, next to _PipelineCache).
//
// Two kinds of entries, both keyed by the exact bytes of everything the result depends on:
// - a source entry: what ProgramCache keeps per shader source (ProgramKey) before any
//   permutation: whether the shader gave up, whether it reads the shader clock, and its resource
//   plan (the descriptor/SRT value graph the renderer evaluates at every draw);
// - a permutation entry: the source key plus the resource specialization and push-data start:
//   the SPIR-V and the compiled program info (CompiledShaderInfo) the renderer binds with.
// Every key starts with the session key: the recompiler source version (a hash of every file the
// recompiler is built from, computed at build time: src/shader_cache_version.cmake), this file
// format, the device, the emitter's device switches and every KYTY_* switch the recompiler reads.
// A file stores its whole key and is used only when the key matches byte for byte and the payload
// hash matches, so neither a hash collision nor a damaged file can give a wrong program.
//
// KYTY_SHADER_DISK_CACHE=0 turns it off. KYTY_SHADER_DISK_CACHE_VERIFY=1 translates every shader
// anyway and compares the result with the cached one (a check of the cache, not a speed-up).
namespace Libs::Graphics::ShaderDiskCache {

class ByteWriter {
public:
	template <typename T>
	requires(std::is_arithmetic_v<T> || std::is_enum_v<T>)
	void Put(T value) {
		if constexpr (std::is_same_v<T, bool>) {
			bytes.push_back(value ? 1u : 0u);
		} else {
			const auto offset = bytes.size();
			bytes.resize(offset + sizeof(T));
			std::memcpy(bytes.data() + offset, &value, sizeof(T));
		}
	}
	void PutWords(std::span<const uint32_t> words) {
		Put<uint64_t>(words.size());
		const auto offset = bytes.size();
		bytes.resize(offset + words.size_bytes());
		if (!words.empty()) {
			std::memcpy(bytes.data() + offset, words.data(), words.size_bytes());
		}
	}
	void PutBytes(std::span<const uint8_t> data) {
		bytes.insert(bytes.end(), data.begin(), data.end());
	}

	std::vector<uint8_t> bytes;
};

class ByteReader {
public:
	explicit ByteReader(std::span<const uint8_t> data): m_data(data) {}

	template <typename T>
	requires(std::is_arithmetic_v<T> || std::is_enum_v<T>)
	[[nodiscard]] bool Get(T& value) {
		if constexpr (std::is_same_v<T, bool>) {
			uint8_t byte = 0;
			if (!Get(byte) || byte > 1u) {
				return Fail();
			}
			value = byte != 0;
			return true;
		} else {
			if (!m_ok || m_data.size() - m_offset < sizeof(T)) {
				return Fail();
			}
			std::memcpy(&value, m_data.data() + m_offset, sizeof(T));
			m_offset += sizeof(T);
			return true;
		}
	}
	// A count of elements that each take at least one byte: more than the bytes left is damage.
	[[nodiscard]] bool GetCount(uint64_t& count) {
		return Get(count) && (count <= m_data.size() - m_offset || Fail());
	}
	[[nodiscard]] bool GetWords(std::vector<uint32_t>& words) {
		uint64_t count = 0;
		if (!Get(count) || count > (m_data.size() - m_offset) / sizeof(uint32_t)) {
			return Fail();
		}
		words.resize(count);
		if (count != 0) {
			std::memcpy(words.data(), m_data.data() + m_offset, count * sizeof(uint32_t));
		}
		m_offset += count * sizeof(uint32_t);
		return true;
	}
	bool Fail() {
		m_ok = false;
		return false;
	}
	[[nodiscard]] bool Ok() const { return m_ok; }
	[[nodiscard]] size_t Remaining() const { return m_data.size() - m_offset; }
	[[nodiscard]] bool AtEnd() const { return m_ok && m_offset == m_data.size(); }

private:
	std::span<const uint8_t> m_data;
	size_t                   m_offset = 0;
	bool                     m_ok     = true;
};

// What a translation depends on besides the guest code and the stage input: fixed for a session.
struct SessionInfo {
	uint32_t vendor_id           = 0;
	uint32_t device_id           = 0;
	bool     bindless_images     = false;
	bool     float_image_atomics = false;
};

[[nodiscard]] uint64_t             SourceVersion();
[[nodiscard]] std::vector<uint8_t> SessionKey(const SessionInfo& session);
[[nodiscard]] bool                 VerifyEnabled();

// The stage input fields a translation reads (a superset of BuildStageStaticKey's).
void AppendInputKey(ByteWriter& key, const ShaderVertexInputInfo& info);
void AppendInputKey(ByteWriter& key, const ShaderPixelInputInfo& info);
void AppendInputKey(ByteWriter& key, const ShaderComputeInputInfo& info);
void AppendSpecialization(ByteWriter& key, const ShaderRecompiler::IR::ResourceSpecialization& value);
// AppendSpecialization's bytes back (guest pipeline seeds, pipelineSeeds.h).
[[nodiscard]] bool ReadSpecialization(ByteReader& in, ShaderRecompiler::IR::ResourceSpecialization& value);

struct SourceRecord {
	bool                             unsupported = false;
	bool                             uses_clock  = false;
	ShaderRecompiler::IR::ResourcePlan plan;
};

// False when the plan refers to an instruction it does not own (then it is not cached).
[[nodiscard]] bool EncodeSourceRecord(bool unsupported, bool uses_clock,
                                      const ShaderRecompiler::IR::ResourcePlan* plan,
                                      std::vector<uint8_t>& out);
[[nodiscard]] bool DecodeSourceRecord(std::span<const uint8_t> bytes, SourceRecord& record);

struct PermutationRecord {
	std::vector<uint32_t>                    spirv;
	ShaderRecompiler::IR::CompiledShaderInfo program;
};

void EncodePermutationRecord(std::span<const uint32_t>                       spirv,
                             const ShaderRecompiler::IR::CompiledShaderInfo& program,
                             std::vector<uint8_t>&                           out);
[[nodiscard]] bool DecodePermutationRecord(std::span<const uint8_t> bytes, PermutationRecord& record);

// Decodes a payload and encodes the result again: true when that gives the same bytes. Run on
// every entry before it is stored, so an encoding that loses something is never cached.
[[nodiscard]] bool SourceRecordRoundTrips(std::span<const uint8_t> bytes);
[[nodiscard]] bool PermutationRecordRoundTrips(std::span<const uint8_t> bytes);

// The entry files of one title. Loads run on the caller's thread; stores are written by a
// background thread (Flush waits for them).
class Store {
public:
	struct Counters {
		std::atomic<uint64_t> source_hits {0};
		std::atomic<uint64_t> source_misses {0};
		std::atomic<uint64_t> permutation_hits {0};
		std::atomic<uint64_t> permutation_misses {0};
		std::atomic<uint64_t> rejected {0};
		std::atomic<uint64_t> writes {0};
		std::atomic<uint64_t> write_failures {0};
		std::atomic<uint64_t> bytes_read {0};
		std::atomic<uint64_t> bytes_written {0};
		std::atomic<uint64_t> verified {0};
		std::atomic<uint64_t> verify_mismatches {0};
		std::atomic<uint64_t> round_trip_failures {0};
	};

	// Null when the cache is off (KYTY_SHADER_DISK_CACHE=0) or there is no title id.
	[[nodiscard]] static std::unique_ptr<Store> Open(std::filesystem::path directory);

	explicit Store(std::filesystem::path directory);
	~Store();
	Store(const Store&)            = delete;
	Store& operator=(const Store&) = delete;

	// The payload stored under exactly this key. A file under the key's name with another key or
	// a damaged payload counts as rejected and is a miss.
	[[nodiscard]] bool Load(std::span<const uint8_t> key, std::vector<uint8_t>& payload);
	void               Save(std::span<const uint8_t> key, std::span<const uint8_t> payload);
	void               Flush();
	void               LogTotals(std::string_view when);

	[[nodiscard]] const std::filesystem::path& Directory() const { return m_directory; }
	Counters                                    counters;

private:
	struct Pending {
		std::filesystem::path path;
		std::vector<uint8_t>  bytes;
	};
	void WriterLoop();
	[[nodiscard]] std::filesystem::path EntryPath(std::span<const uint8_t> key) const;

	std::filesystem::path   m_directory;
	std::mutex              m_mutex;
	std::condition_variable m_wake;
	std::condition_variable m_idle;
	std::deque<Pending>     m_queue;
	bool                    m_writing   = false;
	bool                    m_stop      = false;
	bool                    m_directory_ready = false;
	std::thread             m_writer;
};

} // namespace Libs::Graphics::ShaderDiskCache

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERDISKCACHE_H_ */
