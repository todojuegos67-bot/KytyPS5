#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Loader {
struct Program;
// (trucos3) F2 > Trucos > Inventario: items and spells added BY THE GAME'S OWN CODE on the GAME'S THREAD
// (demonsSoulsInventory.cpp). The menu queues; a detour on a per-frame call of the game thread runs the queue.
namespace DemonsSoulsInventory {
enum class Kind : int { Weapon = 0, Armor = 1, Ring = 2, Goods = 3, Magic = 4, Souls = 5 };
void Install(Program* program);
// True when the detour is in and ran in the last minute (a game was in play): only then Add is offered.
bool Ready();
// Queued for the game thread; false when not Ready or the queue is full. The result shows in Last().
bool Add(Kind kind, int id, int quantity);
// The last result, for the menu: 0 none yet, 1 added, -1 refused by the game (unknown id / no player data).
int  Last(int* id);
// (trucos8) Counts the results the game thread gave (each added / refused request): a change means a new Last().
int  ResultSeq();
// (trucos4) The cheat switches, written ON THE GAME THREAD each frame (the port thread's writes to the eboot's data were
// refused: "[port] trucos: write refused at eboot+0x27cbd50", log 20261120-032627). `vaddr` is an eboot vaddr; the byte
// is kept at `value` while `on`; turned off it is written 0 once. Logged with the prefix [trucos].
void SetFlag(uint64_t vaddr, bool on);
// The soul count set on the game thread (same checks as cp11_setSouls 0x4529c0); the result in Last() (id = souls).
bool SetSouls(int souls);
// (trucos4) Names from the game's own text tables (in memory, the player's language): the item list of a category is
// built on the game thread by asking the game's text lookup 0x6d1660 for "<Category>Name_<id>" over an id range, a
// budget a frame. ScanStart begins (or restarts) it; Scan gives the progress 0..100 (-1 not started); List the ids found
// with their names (UTF-8), ids with the same name as the previous one left out (weapon upgrades).
void ScanStart(Kind kind);
int  Scan(Kind kind);
std::vector<std::pair<int, std::string>> List(Kind kind);
// (trucos21) "Sin limite de peso": on the game thread each frame, the equipment weight class forced to Light (the
// debug convar "Override Equipment Weight Class": int at eboot+0x27c1268, 5 = no override, 0 = Light; read by
// 0x3011f0 for the roll / speed class) and the item-burden limit (float at *(pgd+0x130+0x250)+0x604, the one
// AddItem 0x432be0 compares with) held at 1e9; off: 5 written back and the limit restored.
void SetNoWeight(bool on);
// Hook installed (false: the game's bytes differ or no cave).
bool Installed();
} // namespace DemonsSoulsInventory
} // namespace Loader
