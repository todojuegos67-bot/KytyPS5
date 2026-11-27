#ifndef EMULATOR_SRC_LIBS_TROPHY_H_
#define EMULATOR_SRC_LIBS_TROPHY_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// The game's trophies (sce_sys/trophy2/*.ucp: names, grades and icons in the game's language), the unlocks kept in
// _SaveData/<title>/trophies.json, and the unlock popups the presenter draws (SystemOverlay).
namespace Libs::Trophy {

// grade: 1 platinum, 2 gold, 3 silver, 4 bronze (as the NpTrophy2 API's), 0 unknown
// (1030c) The port's own trophy (not the game's): the Nexus reached for the first time. Not needed for the platinum.
constexpr int kNexusTrophyId = 100;
// (rando1) "Trofeos Randomizer": the port's own, one per archdemon defeated with the randomizer on (200 .. 216, the
// order of kRandoBosses in trophy.cpp) and one for all of them (217). Outside the game's set: never count for the
// platinum nor in the game's "X / 38".
constexpr int kRandoFirst = 200;
constexpr int kRandoAll   = 217;
inline bool   IsRando(int id) { return id >= kRandoFirst && id <= kRandoAll; }
// The randomizer trophy of a boss fight (the game's UDS activity id: "Phlx", "ToKn"...), -1 for any other activity.
int RandoTrophyForActivity(const std::string& activity);
// (rando-trofeos) That boss's "defeated" event flag (FUN_002a7550's numbering), 0 for any other activity.
uint32_t RandoFlagForActivity(const std::string& activity);
// (rando-trofeos) The randomizer's unlocks: _SaveData/<title>/<this>, never trophies.json.
constexpr const char* kRandoStoreName = "trophies-randomizer.json";
// Every boss trophy (200 .. 216) unlocked: the last one (217) is due.
bool RandoAllBosses();
// The randomizer medal (original art drawn in code), RGBA8 bytes, size x size.
void RandoMedal(int size, std::vector<uint8_t>* rgba);
void RandoMedal(int size, int trophy_id, std::vector<uint8_t>* rgba); // (rando3) that trophy's own medal

struct Info {
	int         id    = -1;
	int         grade = 0;
	bool        hidden = false;
	std::string name, detail;
};

struct Toast {
	int                   id    = -1;
	int                   grade = 0;
	std::string           label;      // "Trofeo obtenido", in the game's language
	std::string           grade_text; // "Bronce" ...
	std::string           name;
	std::string           detail; // (10-28) the description, in the game's language
	int                   icon_w = 0, icon_h = 0;
	std::vector<uint32_t> icon; // RGBA8 (IM_COL32 order), icon_w * icon_h; empty: none
	float                 seconds = 6.0f; // (1109) how long it shows (TOAST_SECONDS; the save sync's are shorter)
};

// sceNpTrophy2UnlockTrophy: false when the trophy was already unlocked (no popup). platinum_id: the platinum's id when
// this unlock completed the set, else -1.
bool Unlock(int trophy_id, int* platinum_id);
bool IsUnlocked(int trophy_id);
bool GetInfo(int trophy_id, Info* info);
int  Count();
// (F1 Trofeos) Every trophy and the unlocked ids, only when the module already loaded them (never loads): false if not.
bool Snapshot(std::vector<Info>* all, std::vector<int>* unlocked);
void Preload();

// (port34 Ver trofeos) The F1 trophy page's data. Never waits on the trophy mutex (the present thread calls it): 1 the
// list and the unlock times (id -> unix time, 0 unknown) were copied, 0 busy (keep the last copy), -1 not loaded yet,
// -2 loaded but no trophies (no trophy package).
int  TrySnapshot(std::vector<Info>* all, std::map<int, int64_t>* unlock_times);
bool UnlockTimes(std::map<int, int64_t>* unlock_times); // (waits on the mutex) id -> unix time of every unlock
std::string GameTitle(); // param.sfo's TITLE once loaded ("" before)

// The trophy icons at full quality, packed into one RGBA8 atlas (one texture, one descriptor). Each cell is cell x cell
// with a pad-pixel edge-extended gutter; slot i is at ((i % cols) * stride + pad, (i / cols) * stride + pad).
struct Art {
	int                   cell = 0, pad = 0, stride = 0, cols = 0, width = 0, height = 0;
	std::vector<uint32_t> pixels;    // RGBA8 (IM_COL32 order), width * height
	std::map<int, int>    slots;     // trophy id -> slot
	int                   game_slot = -1; // sce_sys/icon0.png
};
// Reads the package again and decodes every icon (slow: call it on a worker thread). The mutex is held only to copy
// the package path. false: no package (art left empty).
bool BuildArt(int cell, Art* art); // the package read in the background (the emulator, once /app0 is mounted)

// The popup to draw now (nullptr: none) and its age in seconds; the presenter thread calls it each frame.
std::shared_ptr<const Toast> CurrentToast(float* seconds);
constexpr float TOAST_SECONDS = 6.0f; // slide in 0.35 s, shown, fade out the last 0.6 s

} // namespace Libs::Trophy

#endif
