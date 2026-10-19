#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>

namespace Libs::SaveData {

struct SaveDataCapacity {
	static constexpr uint64_t BlockBytes    = 65536;
	static constexpr uint64_t MinimumBlocks = 48;
	uint64_t                  blocks;
	uint64_t                  free_blocks;
};

inline std::optional<SaveDataCapacity> ReadSaveDataCapacity(const std::filesystem::path& directory,
                                                            const std::filesystem::path& metadata) {
	uint64_t                                      bytes = 0;
	std::error_code                               ec;
	std::filesystem::recursive_directory_iterator it(directory, ec), end;
	if (ec) {
		return std::nullopt;
	}
	for (; it != end; it.increment(ec)) {
		if (ec) {
			return std::nullopt;
		}
		if (it->is_regular_file(ec)) {
			bytes += it->file_size(ec);
		}
		if (ec) {
			return std::nullopt;
		}
	}
	if (ec) {
		return std::nullopt;
	}
	const uint64_t used =
	    bytes / SaveDataCapacity::BlockBytes + (bytes % SaveDataCapacity::BlockBytes != 0);
	uint64_t      allocated = 0;
	// The emulator's own bookkeeping: unreadable (a file of zeros after a power cut) counts as none, it is
	// rewritten at the mount. Refusing the mount for it made every later save of the game fail.
	std::ifstream file(metadata);
	if (file) {
		// The emulator's own bookkeeping: unreadable (a file of zeros after a power cut) counts as none, and the
		// mount writes it again. Refused, it failed every later mount of that save.
		if (!(file >> allocated)) {
			allocated = 0;
		}
	} else if (std::filesystem::exists(metadata, ec) || ec) {
		return std::nullopt;
	}
	allocated = std::max({allocated, used, SaveDataCapacity::MinimumBlocks});
	return SaveDataCapacity {allocated, allocated - used};
}

inline bool WriteSaveDataCapacity(const std::filesystem::path& metadata, uint64_t blocks) {
	std::error_code ec;
	std::filesystem::create_directories(metadata.parent_path(), ec);
	if (ec) {
		return false;
	}
	auto temporary = metadata;
	temporary += ".tmp";
	{
		std::ofstream file(temporary, std::ios::trunc);
		file << blocks << '\n';
		file.close();
		if (!file) {
			return false;
		}
	}
	std::filesystem::rename(temporary, metadata, ec);
	return !ec;
}

} // namespace Libs::SaveData
