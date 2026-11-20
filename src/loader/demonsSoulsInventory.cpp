// (trucos3) Inventory cheats: the game's own "add item" and "learn spell" routines, called on the game's thread.
//
// [G] eboot 01.007.000 (vaddr; run time + program base). The console commands of the game (bpe::ConCommand, registered
// in 0x43b6bd / 0x43ba8d / 0x43bf11 / 0x43c4c9 / 0x44e9fa with their usage strings "Equip Game Data - Add Weapon to
// Inventory - Usage: cp11_addWeapon <id> [quantity] [durability]" ...) run:
//   pgd = *(*(0x27e6948) + 0x10)   (player game data; null on the title screen: the command does nothing)
//   params = *(0x27cf378), must be non-null with params+0x258 != 0 (the item tables loaded)
//   AddItem 0x432be0(pgd + 0x130, int* out (=0), type, id, quantity, 1, 1, 1, 0, 1)
//     type: 0 weapon (cp11_addWeapon, 0x42f650 -> call 0x42f742), 0x10000000 armor (cp11_addArmor, 0x42fbf0 -> 0x42fce4),
//     0x20000000 ring (cp11_addAccessory, 0x430880 -> 0x43093c), 0x40000000 goods (cp11_addGoods, 0x4300b0 -> 0x43016c);
//     the last four arguments are pushed 1, 0, 1, 1 (stack args 7..10 = 1, 1, 0, 1) by all four commands.
//   LearnMagic (cp11_learnMagic, 0x44c590): if 0x44c8d0(*(pgd + 0x5d0), id, 2) then 0x44ce20(*(pgd + 0x5d0), pgd + 0x2b0).
// The game thread: the follow camera's context step 0x191d70 is called once a frame from 0x1825bc and 0x182713
// (demonsSoulsCamera.cpp, "eboot+0x191d70 builds ctx ... runs the follow camera's steps"). Both 5-byte calls go to a
// cave that runs Drain() (registers, flags and FP state kept) and then jumps to 0x191d70 as the call did.
// Names (trucos4): the game's text lookup 0x6d1660(bpe string* out, category, id, 0) (its callers 0x6d52ea etc. fall
// back to "?WeaponName?" / "?GoodsName?" / "?ProtectorName?" / "?AccessoryName?" / "?MagicName?" when out+8 == 0; the
// categories they pass: 0xb weapon, 0xc armor, 0xd ring, 0xa goods, 0xe magic). Inside, snprintf("%s_%d", prefix, id),
// a hash (0x863ff0) and the localisation table *(0x2eed5a8) (0x866240). The bpe string: +0 the bytes, +8 the length,
// +0x18 a reference count; released by 0x7f4b90 (as the callers do).
// Each queued request re-checks pgd / params at the moment it runs (the same checks the commands do).
#include "loader/demonsSoulsInventory.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace Loader::DemonsSoulsInventory {
namespace {

constexpr uint64_t CavePage = 0x4000, CaveOffset = 0x8000000 + 24 * 0x4000; // (beside the other caves: +0 .. +16 pages)
constexpr uint64_t SiteA = 0x1825bc, SiteB = 0x182713, Step = 0x191d70;
constexpr uint64_t PgdHolder = 0x27e6948, Params = 0x27cf378, AddItem = 0x432be0, Learn = 0x44c8d0, LearnApply = 0x44ce20;

struct Req {
	Kind kind;
	int  id, quantity;
};
std::mutex           g_mutex;
Req                  g_queue[16];
int                  g_count = 0;
uint64_t             g_base  = 0;
std::atomic<bool>    g_installed {false};
std::atomic<int64_t> g_seen_ns {0};
std::atomic<int>     g_last {0}, g_last_id {0}, g_seq {0};
bool                 g_said_alive = false;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

using AddFn        = int(KYTY_SYSV_ABI*)(uint64_t, int*, uint32_t, int, int, int, int, int, int, int);
using LearnFn      = bool(KYTY_SYSV_ABI*)(uint64_t, int, int);
using LearnApplyFn = void(KYTY_SYSV_ABI*)(uint64_t, uint64_t);

struct FlagState {
	uint64_t vaddr = 0;
	std::atomic<int> want {-1}; // -1 unused, 0 off (write 0 once), 1 on
	bool     written = false;
	int64_t  next_log = 0;
};
FlagState g_flags[8];

std::atomic<int> g_noweight {-1}; // (trucos21) -1 never set, 0 off (restore once), 1 on
bool             g_noweight_applied = false;
float            g_item_limit_saved = 0.0f;
uint64_t         g_item_limit_at    = 0;

void NoWeight() {
	const int want = g_noweight.load(std::memory_order_relaxed);
	if (want < 0 || (want == 0 && !g_noweight_applied)) return;
	auto* wclass = reinterpret_cast<volatile int32_t*>(g_base + 0x27c1268);
	uint64_t limit_at = 0;
	const uint64_t holder = *reinterpret_cast<const uint64_t*>(g_base + PgdHolder);
	if (holder != 0) {
		const uint64_t pgd = *reinterpret_cast<const uint64_t*>(holder + 0x10);
		if (pgd != 0) {
			const uint64_t chr = *reinterpret_cast<const uint64_t*>(pgd + 0x130 + 0x250);
			if (chr != 0) limit_at = chr + 0x604;
		}
	}
	if (want == 1) {
		if (!g_noweight_applied) std::printf("[trucos] sin limite de peso: activado (clase de peso %d -> 0 Ligero)\n", static_cast<int>(*wclass));
		*wclass = 0;
		if (limit_at != 0) {
			auto* lim = reinterpret_cast<volatile float*>(limit_at);
			if (limit_at != g_item_limit_at) {
				g_item_limit_at    = limit_at;
				g_item_limit_saved = *lim;
				std::printf("[trucos] sin limite de peso: limite de objetos %.1f -> sin limite\n", g_item_limit_saved);
			}
			*lim = 1.0e9f;
		}
		g_noweight_applied = true;
	} else {
		*wclass = 5;
		if (limit_at != 0 && limit_at == g_item_limit_at) *reinterpret_cast<volatile float*>(limit_at) = g_item_limit_saved;
		std::printf("[trucos] sin limite de peso: desactivado (clase de peso 5, limite de objetos %.1f)\n", g_item_limit_saved);
		g_item_limit_at    = 0;
		g_noweight_applied = false;
	}
}

constexpr uint64_t TextLookup = 0x6d1660, StringRelease = 0x7f4b90;
constexpr int      kScanEnd[] = {2000000, 2000000, 2000000, 2000000, 2000000}; // (ids above are not used by the game's params)
constexpr int      kScanPerFrame = 1500;
struct ScanState {
	int  cursor = -1; // -1 not started
	std::vector<std::pair<int, std::string>> found;
};
ScanState      g_scan[5];
std::mutex     g_scan_mutex;
std::atomic<int> g_scan_kind {-1};
int            g_names_said = 0;

using TextFn    = void*(KYTY_SYSV_ABI*)(void*, int, int, int);
using ReleaseFn = void(KYTY_SYSV_ABI*)(void*);

bool Name(int kind, int id, std::string* out) {
	static constexpr int kCat[] = {0xb, 0xc, 0xd, 0xa, 0xe};
	alignas(16) uint8_t str[32] {};
	reinterpret_cast<TextFn>(g_base + TextLookup)(str, kCat[kind], id, 0);
	const char* data = nullptr;
	uint64_t    len  = 0;
	std::memcpy(&data, str, 8);
	std::memcpy(&len, str + 8, 8);
	const bool ok = data != nullptr && len > 0 && len < 256;
	if (ok) out->assign(data, len);
	reinterpret_cast<ReleaseFn>(g_base + StringRelease)(str);
	return ok && !out->empty() && (*out)[0] != '?';
}

void ScanStep() {
	const int kind = g_scan_kind.load(std::memory_order_relaxed);
	if (kind < 0) return;
	std::vector<std::pair<int, std::string>> batch;
	int cursor;
	std::string last;
	{
		std::scoped_lock lock(g_scan_mutex);
		cursor = g_scan[kind].cursor;
		if (!g_scan[kind].found.empty()) last = g_scan[kind].found.back().second;
	}
	if (cursor < 0) return;
	// The lookup's own tables (0x6d167c reads *(0x2832918), 0x6d170d *(0x2eed5a8)): not yet made -> wait.
	if (*reinterpret_cast<const uint64_t*>(g_base + 0x2832918) == 0 || *reinterpret_cast<const uint64_t*>(g_base + 0x2eed5a8) == 0) return;
	const int end = std::min(cursor + kScanPerFrame, kScanEnd[kind]);
	std::string name;
	// (trucos5) Weapons and shields: every id the game names is kept (the upgraded / other-path variants are separate ids
	// of EquipParamWeapon); a run of ids with the same name is shown as "name +k" (k = id - the run's first id), as the
	// game lists the level apart from the name. Other kinds: the repeats left out.
	static int         run_first = -1;
	static std::string run_name; // (the raw name of the run: `last` from found.back() carries the " +k")
	if (kind == 0 && cursor == 0) run_name.clear();
	if (kind == 0) last = run_name;
	for (int id = cursor; id < end; id++)
		if (Name(kind, id, &name) && (kind == 0 || name != last)) {
			if (kind == 0) {
				if (name != last) run_first = id;
				run_name = name;
				batch.emplace_back(id, id == run_first ? name : name + " +" + std::to_string(id - run_first));
			} else {
				batch.emplace_back(id, name);
			}
			last = name;
			if (g_names_said < 5) {
				g_names_said++;
				std::printf("[trucos] nombre: categoria %d id %d = %s\n", kind, id, name.c_str());
			}
		}
	std::scoped_lock lock(g_scan_mutex);
	auto& st = g_scan[kind];
	st.found.insert(st.found.end(), batch.begin(), batch.end());
	st.cursor = end;
	if (end >= kScanEnd[kind]) {
		std::printf("[trucos] lista categoria %d: %zu nombres\n", kind, st.found.size());
		g_scan_kind = -1;
	}
}

void Flags() {
	const int64_t now = NowNs();
	for (auto& f: g_flags) {
		const int want = f.want.load(std::memory_order_relaxed);
		if (f.vaddr == 0 || want < 0 || (want == 0 && !f.written)) continue;
		auto* p = reinterpret_cast<volatile uint8_t*>(g_base + f.vaddr);
		const uint8_t before = *p;
		if (before != static_cast<uint8_t>(want)) {
			*p = static_cast<uint8_t>(want);
			std::printf("[trucos] eboot+0x%llx (0x%llx): %s %u -> %u%s\n", static_cast<unsigned long long>(f.vaddr),
			            static_cast<unsigned long long>(g_base + f.vaddr), want ? "activado" : "desactivado", before, *p,
			            (want && f.written) ? " (el juego lo habia devuelto a 0)" : "");
			f.written  = want != 0;
			f.next_log = now + 5000000000LL;
		} else if (want && now >= f.next_log) {
			std::printf("[trucos] eboot+0x%llx sigue a 1\n", static_cast<unsigned long long>(f.vaddr));
			f.next_log = now + 5000000000LL;
		}
		if (want == 0 && before == 0) f.written = false;
	}
}

bool Run(const Req& r) {
	const uint64_t holder = *reinterpret_cast<const uint64_t*>(g_base + PgdHolder);
	if (holder == 0) return false;
	const uint64_t pgd = *reinterpret_cast<const uint64_t*>(holder + 0x10);
	if (pgd == 0) return false;
	if (r.kind == Kind::Souls) {
		auto* souls = reinterpret_cast<int32_t*>(pgd + 0x80);
		const int before = *souls;
		*souls = r.id < 0 ? 0 : (r.id > 999999999 ? 999999999 : r.id); // (cp11_setSouls' cap, 0x3b9ac9ff)
		std::printf("[trucos] almas (0x%llx): %d -> %d\n", static_cast<unsigned long long>(pgd + 0x80), before, *souls);
		return true;
	}
	if (r.kind == Kind::Magic) {
		const uint64_t magic = *reinterpret_cast<const uint64_t*>(pgd + 0x5d0);
		if (magic == 0) return false;
		if (!reinterpret_cast<LearnFn>(g_base + Learn)(magic, r.id, 2)) return false;
		reinterpret_cast<LearnApplyFn>(g_base + LearnApply)(magic, pgd + 0x2b0);
		return true;
	}
	const uint64_t params = *reinterpret_cast<const uint64_t*>(g_base + Params);
	if (params == 0 || *reinterpret_cast<const uint64_t*>(params + 0x258) == 0 || r.quantity <= 0) return false;
	static constexpr uint32_t kType[] = {0x00000000u, 0x10000000u, 0x20000000u, 0x40000000u};
	// (trucos20) 0x432be0 refuses (returns -1) when the item would go over the ITEM BURDEN: it sums the weight of every
	// item held (0x4329d0: EquipParamWeapon +0x24 / Protector / Accessory +0xc / Goods +4 weight x quantity), compares
	// with the limit float at *(inv+0x250)+0x604 (0x432ded..0x432e1a) and, when over, sets the flag inv+0x25c = 1 and
	// returns -1. Light items passed (a dagger), heavy ones (shields, armour) did not. What the shop wants is the item in
	// the inventory, so on exactly that refusal the limit is raised for this one call and put back at once (the player
	// may then be over the burden, as when the game itself hands an item over).
	const uint64_t inv  = pgd + 0x130;
	const uint32_t type = kType[static_cast<int>(r.kind)];
	int            out  = 0;
	int            slot = reinterpret_cast<AddFn>(g_base + AddItem)(inv, &out, type, r.id, r.quantity, 1, 1, 1, 0, 1);
	const int      over = *reinterpret_cast<const int32_t*>(inv + 0x25c);
	std::printf("[port] inventario: AddItem(inv=0x%llx, type=0x%08x, id=%d, qty=%d, 1,1,1,0,1) = %d (out %d, sobrepeso %d)\n",
	            static_cast<unsigned long long>(inv), type, r.id, r.quantity, slot, out, over);
	if (slot == -1 && over == 1) {
		const uint64_t chr = *reinterpret_cast<const uint64_t*>(inv + 0x250);
		if (chr != 0) {
			auto*       limit = reinterpret_cast<float*>(chr + 0x604);
			const float old   = *limit;
			*limit            = 1.0e9f;
			out               = 0;
			slot              = reinterpret_cast<AddFn>(g_base + AddItem)(inv, &out, type, r.id, r.quantity, 1, 1, 1, 0, 1);
			*limit            = old;
			std::printf("[port] inventario: por encima del peso de objetos (limite %.1f): reintento sin limite = %d\n", old, slot);
		}
	}
	return slot != -1;
}

// On the game thread, once a frame (in play).
void KYTY_SYSV_ABI Drain() {
	g_seen_ns.store(NowNs(), std::memory_order_relaxed);
	if (!g_said_alive) {
		g_said_alive = true;
		std::printf("[port] inventario: game-thread hook alive\n");
	}
	Flags();
	NoWeight(); // (trucos21)
	ScanStep();
	Req  local[16];
	int  n = 0;
	{
		std::unique_lock lock(g_mutex, std::try_to_lock);
		if (!lock.owns_lock() || g_count == 0) return;
		n = g_count;
		std::memcpy(local, g_queue, sizeof(Req) * n);
		g_count = 0;
	}
	for (int i = 0; i < n; i++) {
		const bool ok = Run(local[i]);
		g_last_id     = local[i].id;
		g_last        = ok ? 1 : -1;
		if (local[i].kind != Kind::Souls) g_seq.fetch_add(1); // (trucos8: the shop waits on this)
		std::printf("[port] inventario: kind %d id %d x%d -> %s\n", static_cast<int>(local[i].kind), local[i].id, local[i].quantity,
		            ok ? "added" : "refused");
	}
}

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin") return;
	const uint64_t base = program->base_vaddr;
	// The game's bytes as expected: both sites are "call 0x191d70" (e8 rel32), the step starts push rbp; mov rbp, rsp.
	const auto calls_step = [&](uint64_t site) {
		const auto* p = reinterpret_cast<const uint8_t*>(base + site);
		int32_t     rel = 0;
		std::memcpy(&rel, p + 1, 4);
		return p[0] == 0xe8 && base + site + 5 + static_cast<int64_t>(rel) == base + Step;
	};
	const auto* step = reinterpret_cast<const uint8_t*>(base + Step);
	if (!calls_step(SiteA) || !calls_step(SiteB) || step[0] != 0x55 || step[1] != 0x48 || step[2] != 0x89 || step[3] != 0xe5) {
		std::printf("[port] inventario: off (the game's bytes differ from 01.007.000)\n");
		return;
	}
	const uint64_t cave = Libs::LibKernel::Memory::AllocateRuntimeMemory(base + CaveOffset, CavePage, Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                     "demons_souls_inventory", true);
	const auto is_near = [&](uint64_t a) {
		const int64_t d = static_cast<int64_t>(a) - static_cast<int64_t>(cave);
		return d > -0x70000000LL && d < 0x70000000LL;
	};
	if (cave == 0 || !is_near(base + SiteA)) {
		if (cave != 0) Libs::LibKernel::Memory::FreeGuestMemory(cave, CavePage);
		std::printf("[port] inventario: off (no executable memory near the eboot)\n");
		return;
	}
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(CavePage, reinterpret_cast<void*>(cave));
	constexpr uint8_t   save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	const uint8_t*      entry = c.getCurr();
	c.lea(rsp, ptr[rsp - 128]);
	c.pushfq();
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
		c.push(reg);
	c.mov(rbx, rsp);
	c.and_(rsp, -16);
	c.sub(rsp, 512);
	c.db(save_fp, sizeof(save_fp));
	c.mov(rax, reinterpret_cast<uint64_t>(&Drain));
	c.call(rax);
	c.fxrstor64(ptr[rsp]);
	c.mov(rsp, rbx);
	for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.popfq();
	c.lea(rsp, ptr[rsp + 128]);
	c.jmp(reinterpret_cast<const void*>(base + Step), Xbyak::CodeGenerator::T_NEAR); // (the return address is the call site's)
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(cave, c.getSize())) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CavePage);
		std::printf("[port] inventario: off (hook code not generated)\n");
		return;
	}
	g_base = base;
	for (const uint64_t site: {SiteA, SiteB}) {
		uint8_t       j[5] {0xe8, 0, 0, 0, 0};
		const int32_t rel = static_cast<int32_t>(reinterpret_cast<int64_t>(entry) - static_cast<int64_t>(base + site + 5));
		std::memcpy(j + 1, &rel, 4);
		std::memcpy(reinterpret_cast<void*>(base + site), j, 5);
		Common::VirtualMemory::FlushInstructionCache(base + site, 5);
	}
	g_installed = true;
	std::printf("[port] inventario: hook at eboot+0x%llx / +0x%llx (game thread)\n", static_cast<unsigned long long>(SiteA),
	            static_cast<unsigned long long>(SiteB));
#else
	(void)program;
#endif
}

void ScanStart(Kind kind) {
	const int k = static_cast<int>(kind);
	if (k < 0 || k > 4) return;
	std::scoped_lock lock(g_scan_mutex);
	g_scan[k].found.clear();
	g_scan[k].cursor = 0;
	g_scan_kind      = k;
}

int Scan(Kind kind) {
	const int k = static_cast<int>(kind);
	if (k < 0 || k > 4) return -1;
	std::scoped_lock lock(g_scan_mutex);
	return g_scan[k].cursor < 0 ? -1 : static_cast<int>(static_cast<int64_t>(g_scan[k].cursor) * 100 / kScanEnd[k]);
}

std::vector<std::pair<int, std::string>> List(Kind kind) {
	const int k = static_cast<int>(kind);
	if (k < 0 || k > 4) return {};
	std::scoped_lock lock(g_scan_mutex);
	return g_scan[k].found;
}

void SetNoWeight(bool on) {
	g_noweight = on ? 1 : 0;
}

bool Installed() {
	return g_installed.load();
}

void SetFlag(uint64_t vaddr, bool on) {
	for (auto& f: g_flags)
		if (f.vaddr == vaddr) {
			f.want = on ? 1 : 0;
			return;
		}
	for (auto& f: g_flags)
		if (f.vaddr == 0) {
			f.vaddr = vaddr; // (menu thread only; the game thread skips want -1)
			f.want  = on ? 1 : 0;
			return;
		}
}

bool SetSouls(int souls) {
	if (!Ready()) return false;
	std::scoped_lock lock(g_mutex);
	if (g_count >= 16) return false;
	g_queue[g_count++] = {Kind::Souls, souls, 1};
	return true;
}

bool Ready() {
	return g_installed.load() && g_seen_ns.load(std::memory_order_relaxed) != 0 &&
	       NowNs() - g_seen_ns.load(std::memory_order_relaxed) < 60000000000LL; // (in play in the last minute; a queued one runs on the next game frame)
}

bool Add(Kind kind, int id, int quantity) {
	if (!Ready() || id < 0 || kind == Kind::Souls) return false;
	std::scoped_lock lock(g_mutex);
	if (g_count >= 16) return false;
	g_queue[g_count++] = {kind, id, quantity < 1 ? 1 : (quantity > 99 ? 99 : quantity)};
	return true;
}

int ResultSeq() {
	return g_seq.load();
}

int Last(int* id) {
	if (id != nullptr) *id = g_last_id.load();
	return g_last.load();
}

} // namespace Loader::DemonsSoulsInventory
