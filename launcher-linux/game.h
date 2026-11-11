#ifndef KYTY_LAUNCHER_LINUX_GAME_H
#define KYTY_LAUNCHER_LINUX_GAME_H

#include "platform.h"

#include <string>
#include <vector>

// A game: its folder (eboot.bin and sce_sys) or that folder packed into a ZArchive (.zar), and what
// its param.json says (launcher-windows' Game.cs).
struct GameInfo {
	std::string path;
	std::string title_id;
	std::string version;
	std::string name;
	fs::path    picture; // sce_sys/pic0.png or pic1.png of a folder (the game's own art)

	[[nodiscard]] bool        Found() const { return !title_id.empty(); }
	[[nodiscard]] std::string GameId() const { return title_id + "_" + version; }
	// Demon's Souls (its engine settings: the graphics presets).
	[[nodiscard]] bool DemonsSouls() const;
	// _Patches/<title id>.json, the GoldHEN-style cheat file of this title, or empty.
	[[nodiscard]] fs::path CheatFile(const fs::path& root) const;

	// A .zar's param.json comes from the precompile program (`tool`, in `root`), up to 15 s: not on
	// the window's thread.
	static GameInfo Read(const std::string& path, const fs::path& tool, const fs::path& root);
};

enum class StoreState { Unknown, Checking, None, Current, Outdated, OutdatedAll };

// The precompiled shaders of a game version (_PipelineCache/static/<title>_<version>.*), as the
// precompile program's --status reports them; up to a minute.
StoreState CheckShaderStore(const fs::path& root, const fs::path& precompile_tool,
                            const GameInfo& game);

// A GoldHEN-style mods file: every mod has a name, an "enabled" flag and the memory writes it
// makes. The emulator honours the flag (src/loader/gamePatch.cpp), so choosing the mods only sets
// those flags.
struct CheatMod {
	std::string name;
	std::string description;
	bool        enabled = true;
};

struct CheatFile {
	std::string           title;  // its "name", else its "id"
	std::string           before; // the text read (a write checks the file did not change since)
	std::vector<CheatMod> mods;
	std::string           error; // why it could not be read

	static CheatFile Load(const fs::path& file);
	// The flags written (a .bak first, read back after); empty, or the key of what went wrong.
	std::string Save(const fs::path& file) const;
};

#endif
