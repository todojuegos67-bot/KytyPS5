#include "libs/saveDataBackup.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "save backup regression: %s\n", what);
		std::abort();
	}
}

void Write(const std::filesystem::path& path, const std::string& bytes) {
	std::filesystem::create_directories(path.parent_path());
	std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
}

std::string Read(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

size_t Count(const std::filesystem::path& root, const char* prefix = "2") {
	size_t count = 0;
	for (const auto& entry: std::filesystem::directory_iterator(root)) {
		count += entry.path().filename().string().rfind(prefix, 0) == 0 ? 1 : 0;
	}
	return count;
}

} // namespace

int main() {
	using namespace Libs::SaveData::Backup;
	const auto root = std::filesystem::temp_directory_path() /
	                  ("kyty-save-backup-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	const SlotFiles slot {root / "save" / "SLOT0", root / "param" / "SLOT0.param", root / "blocks" / "SLOT0.blocks"};
	const auto      backups = root / "backup" / "SLOT0";
	Write(slot.directory / "USR-DATA", "progress 1");
	Write(slot.directory / "sub" / "extra", "x");
	Write(slot.param, "title");
	Write(slot.blocks, "48\n");
	std::time_t now = 1792370000;

	// A first save: a backup; the same files again: none.
	Check(Make(slot, backups, now).has_value(), "first backup");
	Check(!Make(slot, backups, now + 1).has_value(), "unchanged slot backed up again");
	Check(Count(backups) == 1, "one backup");

	// Saves in a row keep their last; one after the interval is a new generation.
	Write(slot.directory / "USR-DATA", "progress 2");
	Check(Make(slot, backups, now + 60).has_value(), "second backup");
	Check(Count(backups) == 1, "a save within the interval replaces the newest");
	now += ReplaceSeconds + 60;
	for (int i = 0; i < 6; i++) {
		Write(slot.directory / "USR-DATA", "progress " + std::to_string(3 + i));
		Check(Make(slot, backups, now).has_value(), "later backup");
		now += ReplaceSeconds + 1;
	}
	Check(Count(backups) == Generations, "generations kept");
	Check(!Broken(slot), "a good slot is not broken");

	// A crash while saving: the save file at its size, all zeros. Not backed up over the good ones; restored on mount.
	Write(slot.directory / "USR-DATA", std::string(48649, '\0'));
	Check(Broken(slot), "zeros are broken");
	Check(!Make(slot, backups, now).has_value(), "a broken slot backed up");
	const auto restored = Restore(slot, backups, now);
	Check(restored.has_value(), "restore");
	Check(Read(slot.directory / "USR-DATA") == "progress 8", "the newest good save is back");
	Check(Read(slot.directory / "sub" / "extra") == "x" && Read(slot.param) == "title" && Read(slot.blocks) == "48\n",
	      "the slot's other files are back");
	Check(Count(backups, "broken-") == 1, "the broken files are kept");
	Check(!Broken(slot), "restored slot is good");

	// A .param of zeros counts too; with no good backup nothing changes.
	Write(slot.param, std::string(1288, '\0'));
	Check(Broken(slot), "param zeros are broken");
	Check(Restore(slot, backups, now).has_value() && Read(slot.param) == "title", "param restored");
	const SlotFiles lone {root / "save" / "SLOT1", root / "param" / "SLOT1.param", root / "blocks" / "SLOT1.blocks"};
	Write(lone.directory / "USR-DATA", std::string(16, '\0'));
	Check(!Restore(lone, root / "backup" / "SLOT1", now).has_value(), "no backup, no restore");
	Check(Read(lone.directory / "USR-DATA").size() == 16, "the slot without a backup is left as it is");

	std::filesystem::remove_all(root);
	std::puts("save backup tests passed");
}
