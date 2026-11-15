#include "graphics/host_gpu/renderer/pipeline/stageStore.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace {

using Libs::Graphics::Precompile::StageStore;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "StageStoreTests: failed: %s\n", message);
		std::abort();
	}
}

std::vector<uint8_t> Record(uint8_t seed) {
	std::vector<uint8_t> record(1000);
	for (size_t i = 0; i < record.size(); i++) {
		record[i] = static_cast<uint8_t>(seed + i * 7u);
	}
	return record;
}

} // namespace

int main() {
	const auto                 path = std::filesystem::temp_directory_path() / "kyty_stage_store_tests.stages";
	const std::vector<uint8_t> header {1, 2, 3, 4};
	std::filesystem::remove(path);

	{
		StageStore store;
		Check(store.Open(path, header, true), "a new store opens for writing");
		for (uint8_t key = 1; key <= 3; key++) {
			store.Append(key, Record(key));
		}
		Check(store.Size() == 3, "three records written");
	}
	// The last record cut mid-frame, as a precompile killed while writing leaves it.
	std::filesystem::resize_file(path, std::filesystem::file_size(path) - 5);
	{
		StageStore store;
		Check(store.Open(path, header, false), "a cut store opens for reading");
		Check(store.Size() == 2, "the whole records are indexed");
		Check(store.Find(1) == Record(1) && store.Find(2) == Record(2), "the whole records read back");
		Check(!store.Find(3), "the cut record is not found");
	}
	{
		StageStore store;
		Check(store.Open(path, header, true), "a cut store opens for writing");
		store.Append(4, Record(4));
		Check(store.Find(4) == Record(4), "a record appended after the cut reads back");
	}
	{
		StageStore store;
		Check(store.Open(path, header, false), "the store opens again");
		Check(store.Size() == 3 && store.Find(1) == Record(1) && store.Find(4) == Record(4), "the appended record stays");
		StageStore other;
		Check(!other.Open(path, std::vector<uint8_t> {9}, false), "another header is not read");
	}
	std::filesystem::remove(path);
	std::printf("StageStoreTests: ok\n");
	return 0;
}
