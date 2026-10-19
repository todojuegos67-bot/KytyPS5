#include "libs/saveDataBackup.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace Libs::SaveData::Backup {

namespace fs = std::filesystem;

namespace {

constexpr char ParamName[]  = "slot.param";
constexpr char BlocksName[] = "slot.blocks";
constexpr char DataName[]   = "data";

// A slot's files by their place in a backup: data/<path in the slot>, slot.param, slot.blocks.
std::map<std::string, fs::path> Files(const fs::path& directory, const fs::path& param, const fs::path& blocks) {
	std::map<std::string, fs::path> files;
	std::error_code                 ec;
	for (fs::recursive_directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
		if (it->is_regular_file(ec)) {
			files[(fs::path(DataName) / it->path().lexically_relative(directory)).generic_string()] = it->path();
		}
	}
	if (fs::is_regular_file(param, ec)) files[ParamName] = param;
	if (fs::is_regular_file(blocks, ec)) files[BlocksName] = blocks;
	return files;
}

std::map<std::string, fs::path> Files(const SlotFiles& slot) {
	return Files(slot.directory, slot.param, slot.blocks);
}

std::map<std::string, fs::path> BackupFiles(const fs::path& backup) {
	return Files(backup / DataName, backup / ParamName, backup / BlocksName);
}

bool SameBytes(const fs::path& a, const fs::path& b) {
	std::error_code ec;
	if (fs::file_size(a, ec) != fs::file_size(b, ec) || ec) return false;
	std::ifstream     fa(a, std::ios::binary), fb(b, std::ios::binary);
	std::vector<char> ba(65536), bb(65536);
	while (fa && fb) {
		fa.read(ba.data(), static_cast<std::streamsize>(ba.size()));
		fb.read(bb.data(), static_cast<std::streamsize>(bb.size()));
		if (fa.gcount() != fb.gcount() || std::memcmp(ba.data(), bb.data(), static_cast<size_t>(fa.gcount())) != 0) {
			return false;
		}
	}
	return fa.eof() && fb.eof();
}

bool Same(const std::map<std::string, fs::path>& a, const std::map<std::string, fs::path>& b) {
	if (a.size() != b.size()) return false;
	for (const auto& [name, path]: a) {
		const auto other = b.find(name);
		if (other == b.end() || !SameBytes(path, other->second)) return false;
	}
	return true;
}

// The local date and time as a backup's name: 20261019-101500 (with -2, -3 ... when taken).
std::string Stamp(const fs::path& root, std::time_t now, const char* prefix = "") {
	std::tm local {};
#if defined(_WIN32)
	localtime_s(&local, &now);
#else
	localtime_r(&now, &local);
#endif
	char text[32];
	std::strftime(text, sizeof(text), "%Y%m%d-%H%M%S", &local);
	std::string     name = std::string(prefix) + text;
	std::error_code ec;
	for (int n = 2; fs::exists(root / name, ec); n++) {
		name = std::string(prefix) + text + "-" + std::to_string(n);
	}
	return name;
}

// When a backup named by Stamp was made (0: not such a name).
std::time_t StampTime(const std::string& name) {
	std::tm local {};
	if (std::sscanf(name.c_str(), "%4d%2d%2d-%2d%2d%2d", &local.tm_year, &local.tm_mon, &local.tm_mday, &local.tm_hour,
	                &local.tm_min, &local.tm_sec) != 6) {
		return 0;
	}
	local.tm_year -= 1900;
	local.tm_mon -= 1;
	local.tm_isdst = -1;
	return std::mktime(&local);
}

// The backups in `root`, oldest first.
std::vector<std::string> Backups(const fs::path& root) {
	std::vector<std::string> names;
	std::error_code          ec;
	for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
		const auto name = it->path().filename().string();
		if (it->is_directory(ec) && StampTime(name) != 0) names.push_back(name);
	}
	std::sort(names.begin(), names.end());
	return names;
}

bool CopyFile(const fs::path& from, const fs::path& to) {
	std::error_code ec;
	fs::create_directories(to.parent_path(), ec);
	return !ec && fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec) && !ec;
}

// The slot's files copied to `target` in a backup's layout (data/..., slot.param, slot.blocks), each flushed.
bool CopyAll(const std::map<std::string, fs::path>& files, const fs::path& target) {
	for (const auto& [name, path]: files) {
		if (!CopyFile(path, target / name) || !FlushFile(target / name)) return false;
	}
	return true;
}

} // namespace

bool FlushFile(const fs::path& path) {
#if defined(_WIN32)
	const HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
	                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (file == INVALID_HANDLE_VALUE) return false;
	const bool flushed = FlushFileBuffers(file) != 0;
	CloseHandle(file);
	return flushed;
#else
	const int file = open(path.c_str(), O_RDWR);
	if (file < 0) return false;
	const bool flushed = fsync(file) == 0;
	close(file);
	return flushed;
#endif
}

void FlushSlot(const SlotFiles& slot) {
	for (const auto& [name, path]: Files(slot)) {
		FlushFile(path);
	}
}

bool AllZeros(const fs::path& path) {
	std::ifstream     file(path, std::ios::binary);
	std::vector<char> buffer(65536);
	bool              any = false;
	while (file) {
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		const auto count = static_cast<size_t>(file.gcount());
		if (std::any_of(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(count),
		                [](char c) { return c != 0; })) {
			return false;
		}
		any = any || count != 0;
	}
	return any;
}

bool Broken(const SlotFiles& slot) {
	const auto files = Files(slot);
	return std::any_of(files.begin(), files.end(), [](const auto& file) { return AllZeros(file.second); });
}

std::optional<std::string> Make(const SlotFiles& slot, const fs::path& root, std::time_t now) {
	std::error_code ec;
	if (!fs::is_directory(slot.directory, ec)) return std::nullopt;
	const auto files   = Files(slot);
	auto       backups = Backups(root);
	if (!backups.empty()) {
		const auto newest = BackupFiles(root / backups.back());
		if (Same(files, newest)) return std::nullopt;
		// A broken slot is not copied over the good backups: a file of zeros that the newest backup has with data.
		for (const auto& [name, path]: files) {
			if (const auto kept = newest.find(name); kept != newest.end() && AllZeros(path) && !AllZeros(kept->second)) {
				return std::nullopt;
			}
		}
	}
	const auto name      = Stamp(root, now);
	const auto temporary = root / (".new-" + name);
	fs::remove_all(temporary, ec);
	if (!CopyAll(files, temporary)) {
		fs::remove_all(temporary, ec);
		return std::nullopt;
	}
	fs::rename(temporary, root / name, ec);
	if (ec) {
		fs::remove_all(temporary, ec);
		return std::nullopt;
	}
	// Saves in a row keep their last: the newest before this one goes when it is recent.
	if (!backups.empty() && now - StampTime(backups.back()) < ReplaceSeconds) {
		fs::remove_all(root / backups.back(), ec);
		backups.pop_back();
	}
	backups.push_back(name);
	for (size_t i = 0; i + Generations < backups.size(); i++) {
		fs::remove_all(root / backups[i], ec);
	}
	return name;
}

std::optional<std::string> Restore(const SlotFiles& slot, const fs::path& root, std::time_t now) {
	const auto files = Files(slot);
	std::vector<std::string> zeros;
	for (const auto& [name, path]: files) {
		if (AllZeros(path)) zeros.push_back(name);
	}
	if (zeros.empty()) return std::nullopt;
	auto backups = Backups(root);
	for (auto it = backups.rbegin(); it != backups.rend(); ++it) {
		const auto backup = BackupFiles(root / *it);
		// A good backup: it has every file the slot has in zeros, and none of its files is zeros (a file that is
		// zeros there too may be meant that way: then nothing is restored).
		const bool has_all = std::all_of(zeros.begin(), zeros.end(), [&](const auto& name) { return backup.count(name) != 0; });
		if (!has_all ||
		    std::any_of(backup.begin(), backup.end(), [](const auto& file) { return AllZeros(file.second); })) {
			continue;
		}
		// The slot as it was kept beside the backups, then the backup copied in its place.
		std::error_code ec;
		const auto      broken = root / Stamp(root, now, "broken-");
		fs::create_directories(broken, ec);
		if (ec) return std::nullopt;
		fs::rename(slot.directory, broken / DataName, ec);
		if (ec) return std::nullopt;
		if (fs::exists(slot.param, ec)) fs::rename(slot.param, broken / ParamName, ec);
		if (fs::exists(slot.blocks, ec)) fs::rename(slot.blocks, broken / BlocksName, ec);
		fs::create_directories(slot.directory, ec);
		for (const auto& [name, path]: backup) {
			const auto target = name == ParamName    ? slot.param
			                    : name == BlocksName ? slot.blocks
			                                         : slot.directory / fs::path(name).lexically_relative(DataName);
			CopyFile(path, target);
			FlushFile(target);
		}
		return *it;
	}
	return std::nullopt;
}

} // namespace Libs::SaveData::Backup
