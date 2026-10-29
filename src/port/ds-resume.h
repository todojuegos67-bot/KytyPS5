#ifndef EMULATOR_SRC_PORT_DS_RESUME_H_
#define EMULATOR_SRC_PORT_DS_RESUME_H_

// "Demon's Souls" PC port (ds-pc-port8): F1 = save and quit, and JUGAR = back where F1 left.
//
// The emulator cannot snapshot the whole PS5 (HLE: the game's threads, GPU and memory are host objects), so a true
// savestate is impossible. What F1 does instead:
//   1. reads the player's place (position and facing, 01.007.000: loader/demonsSoulsWarp.h PlayerPlace);
//   2. waits until no save of the game is in progress (no save slot mounted, no save file open for writing), then
//      refuses new saves until the process ends, so the game can never be cut in the middle of writing one
//      (Libs::SaveData::BlockForExit), and flushes the save files to disk;
//   3. copies _SaveData to _SaveData-copias\F1-<date-time> (written as .tmp, renamed when complete; the last 3 kept);
//   4. writes _SaveData\port-resume.ini (the character's save, its size / time / checksum, the place), then quits.
// The next start (JUGAR): when the character's save is still that one, the game's next reads of it get a copy whose
// position is the place of F1 (the debug warp's patched copy; the file on disk is never written by the port) and the
// title screen is passed by itself: Cross pressed about once a second until the game loads the character (its second
// mount of the character's save slot), at most 75 s.

#include <cstdint>
#include <filesystem>
#include <string>

namespace Port::Resume {

struct Info {
	std::string profile;   // SAVEDATA0PlayerProfile<n>
	uint64_t    size  = 0; // its USR-DATA at F1
	int64_t     mtime = 0;
	uint32_t    hash  = 0; // FNV-1a 32 of the whole file
	uint32_t    map_uid = 0;
	bool        place   = false; // the position below is known
	float       x = 0, y = 0, z = 0, yaw = 0; // yaw in degrees
	std::string when;    // local time of F1 (YYYY-MM-DD HH:MM)
};

std::filesystem::path File(const std::filesystem::path& root); // <root>\_SaveData\port-resume.ini
bool                  Read(const std::filesystem::path& root, Info* info);
// The launcher's line about it ("" when there is none).
std::string Summary(const std::filesystem::path& root);

// In the game (the F1 key, or R3 held + OPTIONS): starts the save and quit once (a worker thread).
void StartSaveAndQuit(const std::filesystem::path& root);
bool SaveAndQuitActive();
// Its progress text (Spanish key for I18n::T) and whether it ended.
std::string SaveAndQuitText();

// Before the emulator starts (JUGAR): with `enabled` (settings.ini continuar_f1) and a port-resume.ini that matches the
// save, the resume and the automatic Continue are prepared; they start at the game's first frame (Presented()).
void Prepare(const std::filesystem::path& root, const std::string& title_id, bool enabled);
void Presented(); // the present thread, each frame

} // namespace Port::Resume

#endif // EMULATOR_SRC_PORT_DS_RESUME_H_
