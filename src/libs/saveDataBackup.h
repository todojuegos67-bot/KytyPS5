#pragma once

#include <cstddef>
#include <ctime>
#include <filesystem>
#include <optional>
#include <string>

// Save data kept from crashes (issue #73). The game rewrites its save files in place; a crash (a blue screen, a power
// cut) just after can leave them at their new size full of zeros, since NTFS journals a file's size but not its data,
// and there was no other copy. So:
// - a save's files are flushed to disk when the game commits or unmounts it (FlushSlot);
// - each save the game makes is copied, with the slot's .param and .blocks, into a backup next to it
//   (_SaveData/.kyty-backup/<title>/<slot>/<date-time>): written into a temporary folder, flushed, then renamed
//   into place, so a crash while copying loses only that copy (Make);
// - a slot found broken when the game opens it (a save file, .param or .blocks of zeros) gets the newest good backup
//   back, the broken files kept beside the backups (Restore).
namespace Libs::SaveData::Backup {

// A slot's files: its folder (the game's files) and the emulator's two bookkeeping files for it.
struct SlotFiles {
	std::filesystem::path directory, param, blocks;
};

// Backups made within this many seconds of the newest one replace it: saves in a row keep their last, and earlier
// generations are at least this far apart.
inline constexpr std::time_t ReplaceSeconds = 5 * 60;
// Backups kept per slot.
inline constexpr size_t Generations = 4;

// The file's data written to the disk (FlushFileBuffers, fsync); false when it could not be opened.
bool FlushFile(const std::filesystem::path& path);
// Every file of the slot flushed.
void FlushSlot(const SlotFiles& slot);
// A file with bytes, all of them zero: what a crash leaves of a file rewritten just before it.
[[nodiscard]] bool AllZeros(const std::filesystem::path& path);
// The slot is not what the game wrote: one of its files, its .param or its .blocks is all zeros.
[[nodiscard]] bool Broken(const SlotFiles& slot);

// A backup of the slot in `root` (its slot's backup folder), unless the newest one already holds the same files.
// Returns the backup's name, or nothing (no change, or the copy failed: then the partial copy is removed).
std::optional<std::string> Make(const SlotFiles& slot, const std::filesystem::path& root, std::time_t now);
// The newest backup in `root` that is not broken put back in place of the slot's files; the slot's files as they
// were moved to `root`/broken-<date-time>. Returns the backup's name, or nothing (no good backup: nothing changed).
std::optional<std::string> Restore(const SlotFiles& slot, const std::filesystem::path& root, std::time_t now);

} // namespace Libs::SaveData::Backup
