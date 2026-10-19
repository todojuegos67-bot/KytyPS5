#include "libs/saveDataCapacity.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>

void Check(bool ok) {
	if (!ok) { std::fputs("save capacity regression\n", stderr); std::abort(); }
}

int main() {
	using namespace Libs::SaveData;
	const auto root = std::filesystem::temp_directory_path() /
	                  ("kyty-save-capacity-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	const auto data = root / "save";
	const auto meta = root / "metadata" / "save.blocks";
	std::filesystem::create_directories(data);
	std::ofstream(data / "payload").put('a');
	auto value = ReadSaveDataCapacity(data, meta);
	Check(value->blocks == 48 && value->free_blocks == 47);
	Check(WriteSaveDataCapacity(meta, 96));
	value = ReadSaveDataCapacity(data, meta);
	Check(value->blocks == 96 && value->free_blocks == 95);
	std::filesystem::resize_file(data / "payload", 65536 * 2 + 1);
	value = ReadSaveDataCapacity(data, meta);
	Check(value->blocks == 96 && value->free_blocks == 93);
	// Existing data larger than recorded capacity must not underflow free blocks.
	std::filesystem::resize_file(data / "payload", 65536 * 100);
	value = ReadSaveDataCapacity(data, meta);
	Check(value->blocks == 100 && value->free_blocks == 0);
	// Unreadable bookkeeping (text, or the zeros a power cut leaves) counts as none: the save still mounts.
	std::ofstream(meta, std::ios::trunc) << "invalid";
	value = ReadSaveDataCapacity(data, meta);
	Check(value && value->blocks == 100 && value->free_blocks == 0);
	{
		std::ofstream zeros(meta, std::ios::binary | std::ios::trunc);
		zeros.write("\0\0\0\0", 4);
	}
	std::filesystem::resize_file(data / "payload", 1);
	value = ReadSaveDataCapacity(data, meta);
	Check(value && value->blocks == 48 && value->free_blocks == 47);
	std::filesystem::remove_all(root);
}
