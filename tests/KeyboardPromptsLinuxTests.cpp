// SPDX-License-Identifier: GPL-2.0-or-later
// The Linux launcher's keyboard prompts (launcher-linux/keyboard_prompts.cpp), the counterpart of
// KeyboardPromptsTests.cs. No game files are needed: the "originals" are made from the pack's own
// replacement textures (one byte changed), and the pack's hashes are replaced for the test.
//
//   g++ -std=c++20 -iquote launcher-linux -I3rdparty/stb -DKYTY_PROMPT_ASSETS_DIR='"<repo>/launcher-windows/KeyboardPrompts"'
//       tests/KeyboardPromptsLinuxTests.cpp launcher-linux/keyboard_prompts.cpp launcher-linux/keyboard_layout.cpp
//       launcher-linux/prompt_texture.cpp launcher-linux/prompt_assets.cpp launcher-linux/bytes.cpp -o prompts_test
#include "keyboard_layout.h"
#include "keyboard_prompts.h"
#include "prompt_texture.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Bytes::Buffer;
using namespace KeyboardPrompts;

static int g_checks = 0;
static void Check(bool value, const std::string& message) {
	++g_checks;
	if (!value) {
		std::printf("FAIL: %s\n", message.c_str());
		std::exit(1);
	}
}

static Buffer Read(const fs::path& path) {
	std::ifstream file(path, std::ios::binary);
	return Buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static void Write(const fs::path& path, const Buffer& data) {
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

static void Reject(const std::function<void()>& action, const std::string& what) {
	try {
		action();
	} catch (const PromptError& error) {
		const std::string text = error.what();
		Check(text.find("turn off") != std::string::npos && text.find("press Play") != std::string::npos &&
		          text.find(".kyty-keyboard-prompts-v1") != std::string::npos,
		      "missing recovery instructions: " + what);
		return;
	}
	Check(false, "expected a rejection: " + what);
}

using Snapshot = std::map<std::string, Buffer>;
static Snapshot Take(const fs::path& folder) {
	Snapshot all;
	for (const auto& entry: fs::directory_iterator(folder))
		all[entry.path().filename().string()] = Read(entry.path());
	return all;
}

int main() {
	const fs::path root = fs::temp_directory_path() / ("kyty-prompts-linux-test-" + std::to_string(::getpid()));
	fs::remove_all(root);
	fs::create_directories(root);
	try {
		// The pack's real icons, and test originals derived from them.
		const auto        pack = PackIcons();
		Check(pack.size() == 22, "pack has 22 icons");
		std::vector<Icon> icons;
		Snapshot          originals;
		for (const auto& icon: pack) {
			Buffer original = icon.replacement;
			original[original.size() - 1] ^= 0x5a;
			originals[icon.name] = original;
			icons.push_back({icon.name, Bytes::Sha256Hex(original), icon.replacement_hash, icon.replacement});
		}
		SetIconsForTesting(icons);

		const fs::path game_root = root / "game";
		const fs::path folder    = game_root / "ui" / "textures" / "buttonicons" / "_ps5";
		fs::create_directories(folder);
		Game game {game_root.string(), "PPSA01341", "01.007.000"};

		// Off by default: no backup, nothing written.
		Apply(game, false);
		Check(!fs::exists(game_root / ".kyty-keyboard-prompts-v1"), "default off wrote a backup");
		for (const auto& [name, data]: originals) Write(folder / name, data);

		// The version guard.
		game.version = "01.005.000";
		Reject([&] { Apply(game, true); }, "wrong version");
		game.version = "01.007.000";
		Game other = game;
		other.title_id = "PPSA00000";
		Reject([&] { Apply(other, true); }, "wrong title");
		Check(Take(folder) == originals, "a rejected apply changed a file");

		// Apply, idempotence, exact restore.
		Apply(game, true);
		const auto installed = Take(folder);
		int        changed   = 0;
		for (const auto& [name, data]: installed) changed += data != originals[name] ? 1 : 0;
		Check(changed == 22, "exactly 22 changed icons");
		for (const auto& icon: pack) Check(installed.at(icon.name) == icon.replacement, "default artwork is the pack's");
		Apply(game, true);
		Check(Take(folder) == installed, "second enable changed an icon");
		Apply(game, false);
		Check(Take(folder) == originals, "restore did not recover the exact original bytes");
		Check(fs::exists(game_root / ".kyty-keyboard-prompts-v1" / "ps4_l1.ctxr"), "the backup stays when off");
		Apply(game, false);
		Check(Take(folder) == originals, "second disable changed an icon");

		// An interrupted install: one replaced icon, then restore.
		const std::string first = "ps4_button_cross.ctxr", second = "ps4_button_circle.ctxr";
		Write(folder / first, installed.at(first));
		Apply(game, false);
		Check(Read(folder / first) == originals[first], "a partial install did not restore");
		// ... and complete.
		Write(folder / first, installed.at(first));
		Apply(game, true);
		Check(Take(folder) == installed, "a partial install did not complete");
		Apply(game, false);

		// An unrelated edit (another mod) is never overwritten.
		Buffer modded = originals[first];
		modded[modded.size() - 1] ^= 1;
		Write(folder / first, modded);
		Reject([&] { Apply(game, true); }, "modified icon");
		Check(Read(folder / first) == modded, "an existing mod was overwritten");
		Write(folder / first, originals[first]);

		// A damaged backup stops the batch before any write.
		const fs::path backup = game_root / ".kyty-keyboard-prompts-v1";
		Write(backup / first, modded);
		Reject([&] { Apply(game, true); }, "damaged backup");
		Check(Take(folder) == originals, "a rejected batch changed a game file");
		Write(backup / first, originals[first]);

		// A write that fails after the first icon: the batch rolls back.
		int writes = 0;
		SetWriteHookForTesting([&](const std::string& path) {
			if (path.find("/_ps5/") != std::string::npos && ++writes == 2) throw PromptError("simulated write failure");
		});
		Reject([&] { Apply(game, true); }, "failed write");
		SetWriteHookForTesting({});
		Check(Take(folder) == originals, "a failed batch left a modified icon");
		// A rollback that fails too: reported, the others still go back.
		writes = 0;
		SetWriteHookForTesting([&](const std::string& path) {
			if (path.find("/_ps5/") == std::string::npos) return;
			++writes;
			if (writes == 3 || writes == 4) throw PromptError("simulated write failure");
		});
		try {
			Apply(game, true);
			Check(false, "failed write with a failed rollback was accepted");
		} catch (const PromptError& error) {
			Check(std::string(error.what()).find("also failed") != std::string::npos, "the failed rollback is reported");
		}
		SetWriteHookForTesting({});
		// (One icon is left changed, as reported; restoring from the backup recovers it.)
		Apply(game, false);
		Check(Take(folder) == originals, "recovery after a failed rollback");

		// Custom layouts.
		KeyboardLayout layout;
		Check(layout.Assign("Cross", "R") == "Square", "a duplicate input swaps the actions");
		Check(layout.Binding("Square") == "E", "the swap kept the previous input");
		layout.Assign("LeftStickUp", "I");
		layout.Assign("R3", "Mouse:X1");
		layout.sensitivity = "2.0";
		try {
			layout.Assign("Cross", "F7");
			Check(false, "a reserved key was accepted");
		} catch (const PromptError&) {}
		Apply(game, true, &layout);
		const auto custom = Take(folder);
		for (const auto& [name, data]: custom) {
			Check(Buffer(data.begin(), data.begin() + 65) == Buffer(installed.at(name).begin(), installed.at(name).begin() + 65) &&
			          Buffer(data.begin() + 131137, data.end()) == Buffer(installed.at(name).begin() + 131137, installed.at(name).end()),
			      "a custom texture changed its metadata: " + name);
		}
		Check(custom.at(first) != installed.at(first), "the custom label was generated");
		Check(custom.at("ps4_l1.ctxr") == installed.at("ps4_l1.ctxr"), "an unchanged label keeps the pack's artwork");
		Check(custom.at("ps4_stick_r.ctxr") == installed.at("ps4_stick_r.ctxr"), "the mouse icon stays");
		Check(custom.at("ps4_stick_l.ctxr") != installed.at("ps4_stick_l.ctxr"), "the stick shows its new keys");
		Apply(game, true, &layout);
		Check(Take(folder) == custom, "reapplying a custom layout changed artwork");
		// Another profile (the defaults) over it, and back.
		Apply(game, true);
		Check(Take(folder) == installed, "switching to the default profile");
		Apply(game, true, &layout);
		Check(Take(folder) == custom, "switching to the custom profile");
		// An unrelated edit of a generated icon is still refused.
		Buffer edited = custom.at(first);
		edited[80] ^= 1;
		Write(folder / first, edited);
		Reject([&] { Apply(game, false); }, "edited generated icon");
		Write(folder / first, custom.at(first));
		// A mix of generated and default states after an interruption restores exactly.
		Apply(game, true);
		Write(folder / first, custom.at(first));
		Apply(game, false);
		Check(Take(folder) == originals, "a custom profile restore lost original bytes");
		// A failed custom batch rolls back.
		writes = 0;
		SetWriteHookForTesting([&](const std::string& path) {
			if (path.find("/_ps5/") != std::string::npos && ++writes == 2) throw PromptError("simulated write failure");
		});
		Reject([&] { Apply(game, true, &layout); }, "failed custom batch");
		SetWriteHookForTesting({});
		Check(Take(folder) == originals, "a failed custom batch left an icon changed");
		// Generated variants are readable gzip files (the Windows launcher reads them too).
		std::string why;
		bool        found = false;
		for (const auto& entry: fs::recursive_directory_iterator(backup / "generated")) {
			if (entry.path().extension() != ".gz") continue;
			found = true;
			const auto bytes = Bytes::Gunzip(Read(entry.path()), &why);
			Check(why.empty() && bytes.size() == 131149 && Bytes::Sha256Hex(bytes) == entry.path().stem().string(), "generated record is valid");
		}
		Check(found, "generated variants were recorded");

		// A missing backup stops a restore before anything is written.
		Apply(game, true);
		fs::remove(backup / first);
		Reject([&] { Apply(game, false); }, "missing backup");
		Check(Take(folder) == installed, "a missing backup caused a partial restore");
		Write(backup / first, originals[first]);
		Apply(game, false);

		// Linux specifics: a link inside the game folder is refused; the game folder reached through
		// a link is fine; names that differ only in case are found.
		fs::rename(folder / first, root / "moved.ctxr");
		fs::create_symlink(root / "moved.ctxr", folder / first);
		Reject([&] { Apply(game, true); }, "a linked icon");
		fs::remove(folder / first);
		fs::rename(root / "moved.ctxr", folder / first);
		const fs::path link = root / "game-link";
		fs::create_directory_symlink(game_root, link);
		Game linked = game;
		linked.path = link.string();
		Apply(linked, true);
		Check(Take(folder) == installed, "applied through a linked game folder");
		Apply(linked, false);
		Check(Take(folder) == originals, "restored through a linked game folder");
		fs::rename(game_root / "ui", game_root / "UI");
		Apply(game, true);
		Check(Take(game_root / "UI" / "textures" / "buttonicons" / "_ps5") == installed, "a folder of another case");
		Apply(game, false);
		fs::rename(game_root / "UI", game_root / "ui");

		// Two launchers at once: the second one waits for nobody.
		const int fd = ::open((backup / "lock").c_str(), O_RDWR);
		Check(fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) == 0, "test lock");
		Reject([&] { Apply(game, true); }, "locked by another launcher");
		::close(fd);
		Apply(game, true);
		Apply(game, false);

		// A game that is not extracted (a ZArchive) is refused to enable, and ignored when off.
		Game zar {(root / "game.zar").string(), "", ""};
		Write(root / "game.zar", {1, 2, 3});
		Reject([&] { Apply(zar, true); }, "a ZArchive");
		Apply(zar, false);
		Game none {"", "", ""};
		Apply(none, false);
		Reject([&] { Apply(none, true); }, "no game");

		// Layout rules.
		KeyboardLayout rules;
		Check(rules.Pairs().size() == 21, "21 pairs without a sensitivity");
		rules.sensitivity = "0.05";
		try {
			rules.Validate();
			Check(false, "a sensitivity below 0.1 was accepted");
		} catch (const PromptError&) {}
		for (const char* good: {"0.1", "5", "2.", ".5", "1.25"}) {
			rules.sensitivity = good;
			rules.Validate();
		}
		for (const char* bad: {"5.1", "-1", "1e2", "1,5", "abc", ".", " 1"}) {
			rules.sensitivity = bad;
			try {
				rules.Validate();
				Check(false, std::string("sensitivity accepted: ") + bad);
			} catch (const PromptError&) {}
		}
		for (const char* good: {";", "'", "\\", "Keypad +", "F13", "F24", "Mouse:WheelUp", "Left Alt"})
			Check(KeyboardLayout::Supported(good), std::string("supported: ") + good);
		for (const char* bad: {"F7", "F8", "F11", "F25", "F0", "F05", "Pause", "a", "Keypad 55", "Mouse:Foo", ""})
			Check(!KeyboardLayout::Supported(bad), std::string("not supported: ") + bad);
		KeyboardLayout twice;
		twice.SetBinding("Circle", "E");
		try {
			twice.Validate();
			Check(false, "a duplicate input was accepted");
		} catch (const PromptError& error) {
			Check(std::string(error.what()).find("Duplicate binding: E") == 0, "the duplicate is named");
		}
		Check(KeyboardLayout::Label("Left Shift") == "SHIFT" && KeyboardLayout::Label("Keypad 5") == "NUM 5" &&
		          KeyboardLayout::Label("Mouse:WheelDown") == "WHEEL\nDOWN" && KeyboardLayout::Label("q") == "Q",
		      "labels");
		KeyboardLayout defaults;
		Check(defaults.IconLabel("ps4_stick_l.ctxr") == "W A\nS D" && defaults.IconLabel("ps4_dpad_ud") == "\xE2\x86\x91 \xE2\x86\x93", "icon labels");
		const auto pairs = defaults.Pairs();
		Check(pairs[0] == "Cross=E" && pairs[7] == "R1=Mouse:Left" && pairs.back() == "LeftStickRight=D", "pairs");

		std::printf("PASS (%d checks): default off, version guard, apply, idempotence, exact restore, partial recovery, mod/corrupt/missing backup protection, write-failure rollback, custom profile swaps/artwork/restore, links, case, locking, layout rules\n", g_checks);
	} catch (const std::exception& error) {
		std::printf("EXCEPTION: %s\n", error.what());
		fs::remove_all(root);
		return 1;
	}
	fs::remove_all(root);
	return 0;
}
