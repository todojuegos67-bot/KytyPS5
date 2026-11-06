#include "keyboard_prompts.h"

#include "prompt_assets.h"
#include "prompt_texture.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace KeyboardPrompts {

namespace {

const char* const BackupName = ".kyty-keyboard-prompts-v1";

std::vector<Icon>                               g_test_icons;
bool                                            g_use_test_icons = false;
std::function<void(const std::string& path)>    g_write_hook;

using Bytes::Buffer;

std::string Hash(const Buffer& data) {
	return Bytes::Sha256Hex(data);
}

std::string Errno(int code) {
	return std::strerror(code);
}

// ------------------------------------------------------------- files

// What is at `path` (an existing file or folder) is not a link: the files this changes are inside the
// game folder and nothing of the game folder may lead out of it.
void NoLinks(const fs::path& root, const fs::path& path) {
	std::error_code error;
	auto            relative = path.lexically_relative(root);
	if (relative.empty() || *relative.begin() == "..") {
		throw PromptError("Keyboard prompts cannot modify files outside the game folder: " + path.string());
	}
	fs::path current = root;
	for (const auto& part: relative) {
		if (part == ".") continue;
		current /= part;
		const auto status = fs::symlink_status(current, error);
		if (error) break; // (does not exist yet)
		if (fs::is_symlink(status))
			throw PromptError("Keyboard prompts cannot modify linked files or folders: " + current.string());
	}
}

// `name` in `folder`: as it is, else the one that differs only in case (a dump made on Windows
// keeps its own, a case-sensitive file system does not match "Ui" with "ui"), else as it is.
fs::path Insensitive(const fs::path& folder, const std::string& name) {
	std::error_code error;
	if (fs::exists(folder / name, error) || !fs::is_directory(folder, error)) return folder / name;
	auto lower = [](std::string text) {
		for (auto& c: text)
			if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
		return text;
	};
	const auto wanted = lower(name);
	fs::path   found;
	int        matches = 0;
	for (const auto& entry: fs::directory_iterator(folder, error)) {
		if (lower(entry.path().filename().string()) == wanted) {
			found = entry.path();
			++matches;
		}
	}
	return matches == 1 ? found : folder / name;
}

Buffer ReadFile(const fs::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) throw PromptError("Cannot read " + path.string());
	Buffer data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (file.bad()) throw PromptError("Cannot read " + path.string());
	return data;
}

// A file written whole or not at all: a temporary file next to it, flushed, renamed over it (the
// old file's permissions kept).
void WriteAtomic(const fs::path& path, const Buffer& bytes) {
	if (g_write_hook) g_write_hook(path.string());
	static std::mt19937_64 random {std::random_device {}()};
	char                   suffix[32];
	std::snprintf(suffix, sizeof(suffix), ".%016llx.tmp", static_cast<unsigned long long>(random()));
	const std::string temporary = path.string() + suffix;
	struct stat       old {};
	const bool        had = ::stat(path.c_str(), &old) == 0;
	const int         fd  = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, had ? (old.st_mode & 0777) : 0644);
	if (fd < 0) throw PromptError("Cannot write " + path.string() + ": " + Errno(errno));
	auto fail = [&](const char* what, int code) {
		::close(fd);
		::unlink(temporary.c_str());
		throw PromptError(std::string("Cannot ") + what + " " + path.string() + ": " + Errno(code));
	};
	size_t at = 0;
	while (at < bytes.size()) {
		const auto n = ::write(fd, bytes.data() + at, bytes.size() - at);
		if (n < 0) {
			if (errno == EINTR) continue;
			fail("write", errno);
		}
		at += static_cast<size_t>(n);
	}
	if (had) ::fchmod(fd, old.st_mode & 0777);
	if (::fsync(fd) != 0) fail("flush", errno);
	if (::close(fd) != 0) {
		::unlink(temporary.c_str());
		throw PromptError("Cannot write " + path.string() + ": " + Errno(errno));
	}
	if (::rename(temporary.c_str(), path.c_str()) != 0) {
		const int code = errno;
		::unlink(temporary.c_str());
		throw PromptError("Cannot replace " + path.string() + ": " + Errno(code));
	}
}

// The emulator runs: a game file it has open must not change under it.
bool EmulatorRunning() {
	DIR* proc = ::opendir("/proc");
	if (proc == nullptr) return false;
	bool running = false;
	while (const dirent* entry = ::readdir(proc)) {
		const std::string name = entry->d_name;
		if (name.empty() || !std::all_of(name.begin(), name.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
		std::ifstream comm("/proc/" + name + "/comm");
		std::string   text;
		if (comm && std::getline(comm, text) && text == "kyty_emulator") {
			running = true;
			break;
		}
	}
	::closedir(proc);
	return running;
}

// ------------------------------------------------------------- the pack

std::vector<Icon> LoadIcons() {
	if (g_use_test_icons) return g_test_icons;
	return PackIcons();
}

fs::path GeneratedPath(const fs::path& backup, const std::string& name, const std::string& hash) {
	return backup / "generated" / name / (hash + ".gz");
}

// `current` (whose hash is `hash`) is a variant this launcher generated and recorded.
bool KnownGenerated(const fs::path& root, const fs::path& backup, const std::string& name,
                    const std::string& hash, const Buffer& current) {
	const auto path = GeneratedPath(backup, name, hash);
	NoLinks(root, path);
	std::error_code error;
	if (!fs::is_regular_file(path, error)) return false;
	std::string why;
	const auto  bytes = Bytes::Gunzip(ReadFile(path), &why);
	return why.empty() && bytes.size() == current.size() && Hash(bytes) == hash;
}

struct LockFile {
	int fd = -1;
	~LockFile() {
		if (fd >= 0) ::close(fd); // (closing releases the lock)
	}
};

void ApplyCore(const Game& game, bool enabled, const KeyboardLayout* layout) {
	if (game.path.empty()) {
		if (enabled) throw PromptError("Select an extracted Demon's Souls game folder for keyboard prompts.");
		return;
	}
	std::error_code error;
	// (The game folder itself may be reached through a link; what is inside it may not.)
	const fs::path given = game.path;
	const fs::path root  = fs::is_directory(given, error) ? fs::canonical(given, error) : given;
	const fs::path backup = root / BackupName;
	if (!enabled && !fs::is_directory(backup, error)) return;
	if (!enabled) {
		bool any = false;
		for (const auto& entry: fs::directory_iterator(backup, error))
			if (entry.path().extension() == ".ctxr") any = true;
		if (!any) return;
	}
	if (!fs::is_directory(root, error) || game.title_id != "PPSA01341" || (enabled && game.version != "01.007.000"))
		throw PromptError("Keyboard prompts support extracted PPSA01341 version 01.007.000 only; ZArchive and other versions are not supported.");
	if (EmulatorRunning()) throw PromptError("Close the game before changing keyboard prompts.");

	fs::path folder = root;
	for (const char* part: {"ui", "textures", "buttonicons", "_ps5"})
		folder = Insensitive(folder, part);
	NoLinks(root, folder);
	NoLinks(root, backup);
	fs::create_directories(backup, error);
	if (error) throw PromptError("Cannot create " + backup.string() + ": " + error.message());
	const auto lock_path = backup / "lock";
	NoLinks(root, lock_path);
	LockFile lock;
	lock.fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0644);
	if (lock.fd < 0) throw PromptError("Cannot open " + lock_path.string() + ": " + Errno(errno));
	if (::flock(lock.fd, LOCK_EX | LOCK_NB) != 0)
		throw PromptError("Another launcher is changing the keyboard prompts of this game folder.");

	auto icons = LoadIcons();
	const KeyboardLayout defaults;
	if (enabled && layout != nullptr)
		for (auto& icon: icons) {
			const auto label = layout->IconLabel(icon.name);
			if (label != defaults.IconLabel(icon.name))
				icon.replacement = PromptTexture::GenerateWithTemplate(label, icon.replacement);
		}

	struct Step {
		fs::path target;
		Buffer   before, desired;
	};
	std::vector<Step> steps;
	// Validate the whole batch first. Existing mods and damaged backups are never overwritten.
	for (const auto& icon: icons) {
		const auto target   = Insensitive(folder, icon.name);
		const auto original = backup / icon.name;
		NoLinks(root, target);
		NoLinks(root, original);
		if (!fs::is_regular_file(target, error)) throw PromptError("Cannot read " + target.string());
		auto       current = ReadFile(target);
		const auto hash    = Hash(current);
		if (hash != icon.original_hash && hash != icon.replacement_hash && !KnownGenerated(root, backup, icon.name, hash, current))
			throw PromptError("Unrecognized or modified prompt: " + icon.name + ". Restore the original game icon before changing this option.");
		Buffer saved;
		if (fs::exists(original, error)) {
			saved = ReadFile(original);
			if (Hash(saved) != icon.original_hash) throw PromptError("Invalid original prompt backup: " + original.string());
		} else {
			if (hash != icon.original_hash)
				throw PromptError("Original prompt backup is missing: " + icon.name + ". Restore the original icon from your own backup first.");
			saved = current;
		}
		steps.push_back({target, current, enabled ? icon.replacement : saved});
	}
	// Persist every original before replacing anything. An interrupted batch can be completed or
	// restored on the next Play; originals are kept when the option is off too.
	for (size_t i = 0; i < icons.size(); ++i) {
		const auto original = backup / icons[i].name;
		if (!fs::exists(original, error)) WriteAtomic(original, steps[i].before);
	}
	// Record each generated variant before any game write: it recognizes a mix of old and new
	// profiles after an interruption, without accepting unrelated texture mods.
	if (enabled)
		for (const auto& icon: icons) {
			const auto hash = Hash(icon.replacement);
			if (hash == icon.replacement_hash) continue;
			const auto path = GeneratedPath(backup, icon.name, hash);
			NoLinks(root, path);
			fs::create_directories(path.parent_path(), error);
			if (error) throw PromptError("Cannot create " + path.parent_path().string() + ": " + error.message());
			WriteAtomic(path, Bytes::GzipStored(icon.replacement));
		}
	std::vector<size_t> changed;
	try {
		for (size_t i = 0; i < steps.size(); ++i) {
			if (Hash(steps[i].before) == Hash(steps[i].desired)) continue;
			changed.push_back(i);
			WriteAtomic(steps[i].target, steps[i].desired);
		}
	} catch (const PromptError& failure) {
		std::string message = std::string("Could not finish changing keyboard prompts: ") + failure.what();
		// A file that will not go back must not stop the others from going back.
		for (const size_t i: changed) {
			try {
				WriteAtomic(steps[i].target, steps[i].before);
			} catch (const PromptError& again) {
				message += "\n(Putting " + steps[i].target.string() + " back also failed: " + again.what() + ")";
			}
		}
		throw PromptError(message);
	}
}

} // namespace

std::vector<Icon> PackIcons() {
	const auto* manifest = PromptAssets::Find("manifest.txt");
	if (manifest == nullptr) throw PromptError("Missing keyboard prompt resource: manifest.txt");
	const std::string text(reinterpret_cast<const char*>(manifest->data), manifest->size);
	std::vector<Icon> icons;
	size_t            start = 0;
	while (start < text.size()) {
		size_t end = text.find('\n', start);
		if (end == std::string::npos) end = text.size();
		std::string line = text.substr(start, end - start);
		start            = end + 1;
		while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
		if (line.empty()) continue;
		std::vector<std::string> fields;
		size_t                   from = 0;
		for (size_t at; (at = line.find(' ', from)) != std::string::npos; from = at + 1)
			fields.push_back(line.substr(from, at - from));
		fields.push_back(line.substr(from));
		if (fields.size() != 3 || fields[0].find('/') != std::string::npos || fields[0] == "..")
			throw PromptError("Invalid keyboard prompt manifest.");
		const auto* asset = PromptAssets::Find(fields[0] + ".gz");
		if (asset == nullptr) throw PromptError("Missing keyboard prompt resource: " + fields[0]);
		std::string why;
		Icon        icon {fields[0], fields[1], fields[2], Bytes::Gunzip(asset->data, asset->size, &why)};
		if (!why.empty() || Hash(icon.replacement) != icon.replacement_hash)
			throw PromptError("Damaged keyboard prompt: " + icon.name);
		icons.push_back(std::move(icon));
	}
	if (icons.size() != 22) throw PromptError("Incomplete keyboard prompt pack.");
	return icons;
}

void SetIconsForTesting(std::vector<Icon> icons) {
	g_use_test_icons = !icons.empty();
	g_test_icons     = std::move(icons);
}

void SetWriteHookForTesting(std::function<void(const std::string& path)> hook) {
	g_write_hook = std::move(hook);
}

void Apply(const Game& game, bool enabled, const KeyboardLayout* layout) {
	try {
		if (enabled && layout != nullptr) layout->Validate();
		ApplyCore(game, enabled, layout);
	} catch (const std::exception& error) {
		throw PromptError(std::string(error.what()) +
		                  "\n\nIf prompts were partly applied, close the game, turn off "
		                  "Game > Keyboard and mouse prompts and press Play for the same game folder to restore "
		                  "originals from .kyty-keyboard-prompts-v1. Keep that folder; missing or damaged "
		                  "backups must be recovered from your own original game files first.");
	}
}

} // namespace KeyboardPrompts
