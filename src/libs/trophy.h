#ifndef EMULATOR_SRC_LIBS_TROPHY_H_
#define EMULATOR_SRC_LIBS_TROPHY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The game's trophies (sce_sys/trophy2/*.ucp: names, grades and icons in the game's language), the unlocks kept in
// _SaveData/<title>/trophies.json, and the unlock popups the presenter draws (SystemOverlay).
namespace Libs::Trophy {

// grade: 1 platinum, 2 gold, 3 silver, 4 bronze (as the NpTrophy2 API's), 0 unknown
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
	int                   icon_w = 0, icon_h = 0;
	std::vector<uint32_t> icon; // RGBA8 (IM_COL32 order), icon_w * icon_h; empty: none
};

// sceNpTrophy2UnlockTrophy: false when the trophy was already unlocked (no popup). platinum_id: the platinum's id when
// this unlock completed the set, else -1.
bool Unlock(int trophy_id, int* platinum_id);
bool IsUnlocked(int trophy_id);
bool GetInfo(int trophy_id, Info* info);
int  Count();
// (F1 Trofeos) Every trophy and the unlocked ids, only when the module already loaded them (never loads): false if not.
bool Snapshot(std::vector<Info>* all, std::vector<int>* unlocked);
void Preload(); // the package read in the background (the emulator, once /app0 is mounted)

// The popup to draw now (nullptr: none) and its age in seconds; the presenter thread calls it each frame.
std::shared_ptr<const Toast> CurrentToast(float* seconds);
constexpr float TOAST_SECONDS = 5.6f; // slide in 0.35 s, shown, fade out the last 0.6 s

} // namespace Libs::Trophy

#endif
