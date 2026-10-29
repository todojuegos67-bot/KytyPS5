#ifndef EMULATOR_SRC_LIBS_SAVEDATAINFO_H_
#define EMULATOR_SRC_LIBS_SAVEDATAINFO_H_

#include <cstdint>
#include <string>
#include <string_view>

namespace Libs::SaveData {

// What the system's save data list shows of a slot: the texts the game last gave it (SaveDataSetParam; empty for a
// slot no session of this emulator has saved since they were kept) and its modification time (0: no such slot).
struct SaveSlotInfo {
	std::string title;
	std::string sub_title;
	std::string detail;
	int64_t     mtime = 0;
};

SaveSlotInfo GetSaveSlotInfo(std::string_view dir_name);

// (ds-pc-port8) The PC port's save-and-quit (F1):
int         MountedSlots();        // the slots the game has mounted now
uint64_t    PlayerProfileMounts(); // successful mounts of a SAVEDATA0PlayerProfile* slot so far (the title reads it, Continue loads it)
std::string TitleId();
// Waits up to `timeout_ms` for no slot mounted and no save file open for writing during `settle_ms`, then refuses every
// new mount and save data memory write (SAVE_DATA_ERROR_BUSY) until the process ends: no save starts after it. True:
// the game was idle; false: the time ran out (new saves refused all the same).
bool        BlockForExit(int timeout_ms, int settle_ms);
// Every slot of this title flushed to disk.
void        FlushAll();

} // namespace Libs::SaveData

#endif /* EMULATOR_SRC_LIBS_SAVEDATAINFO_H_ */
