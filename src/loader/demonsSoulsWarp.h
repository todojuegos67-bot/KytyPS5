#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Debug warp for Demon's Souls (PPSA01341): start the character on another map. A warp is armed
// (the live command `warp MAP SPAWN`; Select/ToggleSelected have no keys now: F8 is the overlay, F9 a debug
// snapshot); the next time the game reads its save (Continue on the title screen) it gets a copy whose map and position
// are those
// of the chosen spawn point, a player part of the map's MSB file. The save on disk is not touched:
// the game writes the new location itself when it next saves, which ends the warp.
namespace Loader::DemonsSoulsWarp {

struct Spawn {
	std::string map;     // m04_01_00_00
	std::string name;    // c0000_0001
	uint32_t    map_uid; // 0x04010000
	float       x, y, z;
	float       yaw; // degrees
};

// The player spawn points of every map (read from the game's MSB files once).
const std::vector<Spawn>& Spawns();

// F6 (01.007.000): the player's position appended to _Build/waypoints/route-<start time>.txt as "x y z label" (the
// format automated tours read); RecordedWaypoints counts them (the window title shows it).
void     RecordWaypoint();
uint32_t RecordedWaypoints();
// The player's position and yaw (01.007.000; false: another build, or no character loaded).
bool PlayerPlace(float xyz[3], float* yaw);
// A line in the panel (HudText) for `seconds`.
void ShowNotice(std::string message, int seconds);
// Live `glide X Y Z SPEED` (01.007.000, automated tours): the player moved there in a straight line at SPEED m/s by the
// game's warp request each 16 ms, facing the way it goes; returns the seconds it takes (-1: no player, another build).
double Glide(float x, float y, float z, float speed);

// Live `gamecmd TEXT` (01.007.000, test runs with KYTY_LIVE_FILE): an engine console command, e.g.
// "+cp11_unlockAllArchstones" or "+cp11_loadLevel 2 1" (world, level), run on the game's pad-reading thread the next time it
// reads the pad, through the engine's own command entry (the one its debug "golden path" helper calls). False: another
// build (nothing queued).
bool QueueCommand(std::string_view text);
// The pad read (libs/controller.cpp): runs a queued command.
void RunQueuedCommand();

bool        Arm(std::string_view map, std::string_view spawn);
void        Disarm();
void        Select(int step);    // the spawn point ToggleSelected arms
void        ToggleSelected();
std::string HudText();           // the panel text (empty: none)

// USR-DATA with the spawn's map and position (the size and FNV-1a checksum recomputed; a save without
// a position, a character still in the tutorial, gets the entries); false when its layout is unexpected.
bool PatchSave(std::vector<uint8_t>& data, const Spawn& spawn);

// File system hooks: the host file a read-only open of `real` should open instead (a patched copy
// of the save while a warp is armed; empty: `real` itself), and a write-open.
std::filesystem::path RedirectRead(const std::filesystem::path& real);
void                  NoteWrite(const std::filesystem::path& real);
// A stat of `real`: the size of the copy a read would get (nothing: its own), without making one.
std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real);

} // namespace Loader::DemonsSoulsWarp
