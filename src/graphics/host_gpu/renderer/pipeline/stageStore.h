#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_STAGESTORE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_STAGESTORE_H_

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::Precompile {

// The stage store (_PipelineCache/<title>.stages): what the launcher's precompile translated, by stage record key, so
// that the game's precompile on start takes each stage's plan and SPIR-V instead of translating it again (session 9's
// M7, for the first game start after a precompile). A header, then records, each a frame and its zstd bytes; records
// are only appended, so a store cut short keeps every whole record. A store of another build, switch set or GPU (the
// header) is ignored, and the next precompile writes it anew.
class StageStore {
public:
	StageStore() = default;
	StageStore(const StageStore&)            = delete;
	StageStore& operator=(const StageStore&) = delete;

	// Indexes path's records when its header is header. writing (the precompile): appends from here on, and starts
	// the file over when its header is another. False when nothing can be read (reading) or written (writing).
	bool Open(const std::filesystem::path& path, std::span<const uint8_t> header, bool writing);
	// The record of key; any thread.
	std::optional<std::vector<uint8_t>> Find(uint64_t key);
	// Adds key's record unless it has one; any thread, writing only.
	void Append(uint64_t key, std::span<const uint8_t> record);
	[[nodiscard]] size_t   Size();
	[[nodiscard]] uint64_t Bytes();
	[[nodiscard]] bool     Writing() const { return m_writing; }

	// Changes with every build of the emulator (its file time), as upstream's store header has it: a local build of
	// the same revision may translate differently. 0 when the executable cannot be found.
	static uint64_t BuildStamp();

private:
	struct Frame {
		uint32_t compressed = 0;
		uint32_t raw        = 0;
		uint64_t key        = 0;
		uint64_t checksum   = 0; // XXH3 of the compressed bytes
	};
	struct Entry {
		uint64_t offset     = 0; // of the compressed bytes
		uint32_t compressed = 0;
		uint32_t raw        = 0;
		uint64_t checksum   = 0;
	};

	std::mutex                          m_mutex;
	std::fstream                        m_file;
	std::unordered_map<uint64_t, Entry> m_index;
	uint64_t                            m_end     = 0;
	std::atomic<bool>                   m_writing = false; // Append reads it before taking the lock
};

} // namespace Libs::Graphics::Precompile

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_STAGESTORE_H_ */
