#pragma once
// VK_KHR_pipeline_binary: the static precompile's pipelines as the driver's own binaries, in one file
// (_PipelineCache/static/<title>_<version>.binaries) indexed by the driver's pipeline keys. A pipeline's
// binaries are read when the game first needs it; a whole static VkPipelineCache is instead copied by
// the driver into memory when it is loaded (4 GB for Demon's Souls).
//
// File: magic, signature (the GPU and driver, and the driver's global pipeline key), the binaries' data,
// then the tables (binaries: key, offset, size, XXH3; pipelines: key and their binaries) and a footer
// pointing at them.

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

class PipelineBinaries {
public:
	struct Key {
		uint32_t                                                size = 0;
		std::array<uint8_t, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR> bytes {};
		bool operator==(const Key& other) const { return size == other.size && bytes == other.bytes; }
	};
	struct KeyHash {
		size_t operator()(const Key& key) const;
	};
	// A pipeline's binaries (in order), and of those the ones made for it alone (destroyed with this).
	struct Handles {
		vk::Device                       device;
		std::vector<VkPipelineBinaryKHR> binaries;
		std::vector<VkPipelineBinaryKHR> owned;
		Handles() = default;
		Handles(Handles&& other) noexcept
		    : device(other.device), binaries(std::move(other.binaries)), owned(std::move(other.owned)) {}
		Handles& operator=(Handles&&) = delete;
		Handles(const Handles&)       = delete;
		~Handles();
	};

	// The store at `path` when it was made on this GPU and driver (signature), else none.
	static std::unique_ptr<PipelineBinaries> Open(vk::Device device, const std::filesystem::path& path,
	                                              const std::string& signature);
	// The signature binaries of this device carry: `device` (ShaderInputDeviceSignature) and the
	// driver's global pipeline key; empty when the driver gives none.
	static std::string Signature(vk::Device device, const std::string& device_signature);
	// The driver's key of a pipeline (a Vk*PipelineCreateInfo); false when it gives none.
	static bool PipelineKey(vk::Device device, const void* create_info, Key& key);
	~PipelineBinaries();

	// The binaries stored for a pipeline key (empty: none, or their data is damaged).
	[[nodiscard]] Handles Load(const Key& key) const;
	[[nodiscard]] bool    Holds(const Key& key) const { return m_pipelines.contains(key); }
	// Its binaries' data reads back whole (their hashes hold): what a copy of them needs (the precompile's).
	[[nodiscard]] bool Intact(const Key& key) const;
	[[nodiscard]] size_t  Pipelines() const { return m_pipelines.size(); }
	[[nodiscard]] size_t  Binaries() const { return m_binaries.size(); }
	[[nodiscard]] uint64_t DataBytes() const;

private:
	friend class PipelineBinaryWriter;
	struct Binary {
		Key      key;
		uint64_t offset = 0;
		uint64_t size   = 0;
		uint64_t hash   = 0;
	};
	struct Pipeline {
		uint32_t first = 0; // into m_refs
		uint32_t count = 0;
	};
	// Binaries this many pipelines use are made once and kept.
	static constexpr uint32_t SharedUsers = 16;
	bool ReadData(const Binary& binary, std::vector<uint8_t>& data) const;
	// The binaries of these indices into out[] (all or none).
	bool CreateBinaries(std::span<const uint32_t> indices, VkPipelineBinaryKHR* out) const;

	vk::Device                                       m_device;
	uint64_t                                         m_file = 0;
	std::vector<Binary>                              m_binaries;
	std::vector<uint32_t>                            m_users; // pipelines that use each binary
	std::vector<uint32_t>                            m_refs;
	std::unordered_map<Key, Pipeline, KeyHash>       m_pipelines;
	mutable std::mutex                               m_shared_mutex;
	mutable std::vector<VkPipelineBinaryKHR>         m_shared; // made binaries of SharedUsers or more
};

// The static precompile's output: pipelines' binaries, captured from pipelines created with
// VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR or copied from another store; binaries shared by several
// pipelines are kept once. Thread-safe.
//
// NVIDIA compresses binaries with a dictionary ("_NVDICT_" key) once a process has made a few hundred
// pipelines, trained from them and kept in its disk cache; the driver then reads only binaries made with
// the dictionary its own disk cache holds (the one among the binaries is not used): on another PC, or
// after that cache changes, they are compiled again. Such captures are left out: the precompile runs
// small shards, each with an empty disk cache of its own (precompile-windows.ps1), and one that gets
// there anyway is split.
class PipelineBinaryWriter {
public:
	explicit PipelineBinaryWriter(vk::Device device);
	// The binaries of a pipeline created with VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR (its captured
	// data is released), or with `create_info` (a Vk*PipelineCreateInfo) those the driver's internal
	// cache holds for it; false when the driver gives none.
	bool Capture(const PipelineBinaries::Key& key, vk::Pipeline pipeline, const void* create_info = nullptr);
	// A pipeline another store holds (its data is read when this is saved).
	void Copy(const PipelineBinaries::Key& key, const PipelineBinaries& from);
	// Every pipeline of another store.
	void CopyAll(const PipelineBinaries& from);
	// What Save did besides writing: the pipelines left out because a binary of theirs could not be read back from the
	// store holding it (its file damaged on the disk), and why it failed.
	struct SaveReport {
		size_t      unreadable_binaries = 0, dropped_pipelines = 0;
		std::string failure;
	};
	// Written whole to a temporary file renamed to `path`.
	bool Save(const std::filesystem::path& path, const std::string& signature, SaveReport* report = nullptr) const;
	[[nodiscard]] size_t Pipelines() const;
	// Pipelines copied from another store, captured (compiled, or from a static cache's hit), and left out
	// (compressed with NVIDIA's dictionary).
	[[nodiscard]] size_t Copied() const { return m_copied.load(std::memory_order_relaxed); }
	[[nodiscard]] size_t Captured() const { return m_captured.load(std::memory_order_relaxed); }
	[[nodiscard]] size_t LeftOut() const { return m_left_out.load(std::memory_order_relaxed); }
	// KYTY_PRECOMPILE_CLAIMS=<dir>: the shards of one precompile share the keys they hold (a file each),
	// so a pipeline the seeds of several shards make is compiled by one of them, not by each.
	[[nodiscard]] bool   Claimed(const PipelineBinaries::Key& key) const;
	[[nodiscard]] size_t ClaimedElsewhere() const { return m_claimed.load(std::memory_order_relaxed); }

private:
	struct Binary {
		PipelineBinaries::Key   key;
		uint64_t                hash = 0;        // XXH3 of the data: keys alone are not unique (see AddBinary)
		std::vector<uint8_t>    data;            // captured, or
		const PipelineBinaries* from  = nullptr; // copied: that store's binary
		uint32_t                index = 0;
	};
	struct BinaryId {
		PipelineBinaries::Key key;
		uint64_t              hash = 0;
		bool operator==(const BinaryId& other) const { return hash == other.hash && key == other.key; }
	};
	struct BinaryIdHash {
		size_t operator()(const BinaryId& id) const { return PipelineBinaries::KeyHash {}(id.key) ^ id.hash; }
	};
	uint32_t AddBinary(Binary binary);
	void     Claim(const PipelineBinaries::Key& key) const;

	vk::Device                                                                  m_device;
	std::filesystem::path                                                       m_claims;
	mutable std::mutex                                                          m_mutex;
	std::vector<Binary>                                                         m_binaries;
	std::unordered_map<BinaryId, uint32_t, BinaryIdHash>                        m_binary_index;
	std::unordered_map<PipelineBinaries::Key, std::vector<uint32_t>, PipelineBinaries::KeyHash> m_pipelines;
	std::atomic<size_t> m_copied {0}, m_captured {0}, m_left_out {0};
	mutable std::atomic<size_t> m_claimed {0};
};

// Pipelines compiled in play (none of the static store's) kept as their binaries for the sessions after: a driver's
// own cache is not enough (AMD's Windows driver adds nothing to its 1 GiB cache once it is full: such pipelines, 478-1051
// a session, compiled again in every session, issue #78). Beside the driver cache: <name>.played.binaries and the
// captures saved since, <name>.played.<n>.binaries, merged into it at a start once there are more than MergeAt.
class PlayedPipelines {
public:
	// Null when the driver gives no keys (`signature` empty). Files of another GPU or driver are removed, and all of
	// them past Limit bytes (pipelines of shaders long changed pile up): their pipelines compile again.
	static std::unique_ptr<PlayedPipelines> Open(vk::Device device, const std::filesystem::path& base,
	                                             const std::string& signature);
	~PlayedPipelines();

	// The binaries an earlier session kept for a pipeline key.
	[[nodiscard]] PipelineBinaries::Handles Load(const PipelineBinaries::Key& key) const;
	// Whether pipelines compiled now are made with VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR and kept: not once the
	// driver compresses binaries with NVIDIA's dictionary (PipelineBinaryWriter), which a later process may not read.
	[[nodiscard]] bool Capturing() const { return m_capturing.load(std::memory_order_relaxed); }
	// A pipeline created with the capture flag (its captured data is released).
	void Capture(const PipelineBinaries::Key& key, vk::Pipeline pipeline);
	// The captures since the last save into a file of their own, when there are any.
	bool Save();
	void Report() const;
	// The report once its counts changed (a session that is ended, not exited, reports none at its end).
	void ReportChanged();

	std::atomic<uint64_t> hits {0}, refused {0};

private:
	PlayedPipelines() = default;
	static constexpr uint64_t Limit   = uint64_t {1} << 30u;
	static constexpr size_t   MergeAt = 8;

	vk::Device                                     m_device;
	std::filesystem::path                          m_base;
	std::string                                    m_signature;
	std::vector<std::unique_ptr<PipelineBinaries>> m_stores; // newest first
	size_t                                         m_held = 0;
	uint32_t                                       m_next = 0; // the next file's <n>
	// Captures go into the current writer; a save takes it (exclusively) and writes it without blocking them.
	mutable std::shared_mutex             m_session_mutex;
	std::unique_ptr<PipelineBinaryWriter> m_session;
	std::mutex                            m_save_mutex;
	std::atomic<bool>                     m_capturing {true};
	std::atomic<uint64_t>                 m_captured {0}, m_saved {0}, m_save_failures {0};
	uint64_t                              m_reported = 0;
};

} // namespace Libs::Graphics
