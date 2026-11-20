// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// F1 > ARMAS Y OBJETOS (see bb_shop.h). Formats (checked against the 1.09 files):
//  - DCX: "DCX\0", DCS (+0x1c uncompressed size, BE), DCP "DFLT", DCA, then a zlib stream.
//  - BND4: count +0x0c, entry header size +0x20 (36), unicode +0x30; entries from +0x40:
//    flags, -1, size (i64), uncompressed size (i64), data offset (u32), id (i32), name (u32).
//  - PARAM: row count u16 +0x0a, type name +0x0c, rows from +0x40: id (i32), pad, data (i64),
//    name (i64); row size = the distance between the first two data offsets.
//      EQUIP_PARAM_GOODS_ST (108 B): +0x2c icon (u16), +0x34 max count (s16), +0x3a type
//        (0 normal, 1 key, 2 upgrade material, 6 chalice).
//      EQUIP_PARAM_WEAPON_ST (316 B): +0x5c origin (+0) weapon id, +0xba icon (u16).
//      SHOP_LINEUP_PARAM (32 B): +0 id, +4 price, +0x16 currency (0 echoes, 5 insight),
//        +0x17 type (0 weapon, 3 goods).
//      ITEMLOT_PARAM_ST (148 B): 8 ids at +0, 8 categories at +0x20 (0 weapon, 4 goods).
//  - FMG (v2, 64-bit): group count +0x0c, string offsets table (i64) at +0x18, groups from +0x28
//    {offset index, first id, last id, pad}; UTF-16LE strings. item.msgbnd ids: 10 goods
//    names, 11 weapon names, 20 goods captions, 21 weapon captions, 24 goods lore, 25 weapon lore.
//  - TPF (PS4): +0x10 data offset, size, format (102 = BC7), type, mips, flags, width, height;
//    the BC7 blocks are stored in 8x8-block tiles (row-major), Morton order inside a tile,
//    padded to whole tiles. menu_knowledge_NNNNN uses NNNNN = the param icon id.
// Item list: weapons = +0 ids (id % 10000 == 0) that are their own origin and are not an
// uncanny / lost variant (id / 10000 % 10 == 0); goods = normal consumables (900..1594, 1700),
// hunter tools (2000..2999) and upgrade materials (type 2). Every entry must have a name (not
// "*", "NOT NEEDED", "(innecesario)"), be obtainable in the game (in ItemLotParam or sold for
// echoes in ShopLineupParam), have no duplicate name, and have a readable icon.
// Prices: the lowest echoes price in ShopLineupParam; items the game does not sell for echoes
// get a fixed price: weapons 15,000, items 2,000, tools 20,000, materials 10,000.
//
// Add-item (vaddr; Ghidra = +0x400000): 0x1310370 AddItem(unused, category, id, count, -1)
// returns 0 on success. Lua "AddInventoryItem" (0x132FBE0) calls it with category 0 (weapon) /
// 0x40000000 (goods); it checks the param row (0x1F1E3D0 goods, 0x1F29AA0 weapon), adds to the
// inventory PlayerGameData+0x1D0 (0x14CDFD0) and queues the game's "obtained" message through
// the manager at [0x5546C88] (null: nothing happens, so it is checked first).
#include "core/bb_shop.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <fmt/format.h>
#include <zlib.h>
#include "common/logging/log.h"
#include "common/singleton.h"
#include "core/bb_coop_config.h"
#include "core/emulator_settings.h"
#include "core/file_sys/fs.h"
#include "core/libraries/np/np_handler/np_handler.h"

namespace BB::Shop {
namespace {

constexpr u64 AddItemVa = 0x1310370;
constexpr u64 ItemGetManVa = 0x5546C88;
constexpr u32 PgdEchoes = 0x94;
constexpr s32 EchoesMax = 999999999;
constexpr s32 CatWeapon = 0, CatGoods = 0x40000000;
constexpr s32 kFallbackPrice[] = {15000, 2000, 20000, 10000};

// AddItem prologue (1.09) and its "cmpq $0, [rip -> 0x5546C88]" at +0x2e.
constexpr u8 kAddItemSig[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41,
                              0x54, 0x53, 0x48, 0x81, 0xec, 0x38, 0x01, 0x00, 0x00, 0x45, 0x89,
                              0xc4, 0x41, 0x89, 0xce, 0x41, 0x89, 0xd7};
using AddItemFn = s32(PS4_SYSV_ABI*)(u64, s32, s32, s32, s32);

uintptr_t g_base = 0;
bool g_eboot_ok = false;

u16 Le16(const u8* p) {
    return u16(p[0] | p[1] << 8);
}
u32 Le32(const u8* p) {
    return u32(p[0]) | u32(p[1]) << 8 | u32(p[2]) << 16 | u32(p[3]) << 24;
}
u64 Le64(const u8* p) {
    return u64(Le32(p)) | u64(Le32(p + 4)) << 32;
}
u32 Be32(const u8* p) {
    return u32(p[3]) | u32(p[2]) << 8 | u32(p[1]) << 16 | u32(p[0]) << 24;
}

using Bytes = std::vector<u8>;

bool Undcx(const Bytes& f, Bytes& out) {
    if (f.size() < 0x50 || std::memcmp(f.data(), "DCX\0", 4) != 0) return false;
    if (std::memcmp(f.data() + 0x18, "DCS\0", 4) != 0 ||
        std::memcmp(f.data() + 0x28, "DFLT", 4) != 0) {
        return false;
    }
    const u64 usize = Be32(f.data() + 0x1c);
    size_t data = 0;
    for (size_t i = 0x2c; i + 8 < std::min<size_t>(f.size(), 0x100); i += 4) {
        if (std::memcmp(f.data() + i, "DCA\0", 4) == 0) {
            data = i + Be32(f.data() + i + 4);
            break;
        }
    }
    if (data == 0 || data >= f.size() || usize == 0 || usize > (256u << 20)) return false;
    out.assign(usize, 0);
    z_stream zs{};
    if (inflateInit(&zs) != Z_OK) return false;
    zs.next_in = const_cast<Bytef*>(f.data() + data);
    zs.avail_in = uInt(f.size() - data);
    zs.next_out = out.data();
    zs.avail_out = uInt(usize);
    const int r = inflate(&zs, Z_FINISH);
    const u64 got = zs.total_out;
    inflateEnd(&zs);
    return (r == Z_STREAM_END || got == usize) && got == usize;
}

struct BndFile {
    std::string name; // base name, ASCII part
    s32 id;
    const u8* data;
    u64 size;
};
// Only the 36-byte entry layout of the BB 1.09 binders (checked: param and msg binders).
bool Bnd4(const Bytes& b, std::vector<BndFile>& out) {
    const u8* p = b.data();
    if (b.size() < 0x40 || std::memcmp(p, "BND4", 4) != 0 || Le64(p + 0x20) != 36) return false;
    const u32 count = Le32(p + 0x0c);
    const bool unicode = p[0x30] == 1;
    if (count == 0 || count > 4096 || 0x40 + u64(count) * 36 > b.size()) return false;
    for (u32 i = 0; i < count; ++i) {
        const u8* e = p + 0x40 + u64(i) * 36;
        BndFile f;
        f.size = Le64(e + 8);
        const u64 off = Le32(e + 24);
        f.id = s32(Le32(e + 28));
        const u64 name = Le32(e + 32);
        if (off + f.size > b.size() || name >= b.size()) return false;
        f.data = p + off;
        std::string n;
        for (u64 k = name; k + 1 < b.size(); k += unicode ? 2 : 1) {
            const u32 c = unicode ? Le16(p + k) : p[k];
            if (c == 0) break;
            n.push_back(c < 0x80 ? char(c) : '?');
        }
        const size_t slash = n.find_last_of("\\/");
        f.name = slash == std::string::npos ? n : n.substr(slash + 1);
        out.push_back(std::move(f));
    }
    return true;
}

struct Param {
    std::string type;
    u32 row_size = 0;
    std::map<s32, const u8*> rows;
};
bool ReadParam(const BndFile& f, Param& out, u32 want_size) {
    const u8* p = f.data;
    if (f.size < 0x40) return false;
    const u32 n = Le16(p + 0x0a);
    if (0x40 + u64(n) * 24 > f.size || n < 2) return false;
    out.type.assign(reinterpret_cast<const char*>(p + 0x0c),
                    strnlen(reinterpret_cast<const char*>(p + 0x0c), 32));
    const u64 d0 = Le64(p + 0x40 + 8), d1 = Le64(p + 0x40 + 24 + 8);
    out.row_size = u32(d1 - d0);
    if (out.row_size != want_size) return false;
    for (u32 i = 0; i < n; ++i) {
        const u8* r = p + 0x40 + u64(i) * 24;
        const u64 d = Le64(r + 8);
        if (d + out.row_size > f.size) return false;
        out.rows[s32(Le32(r))] = p + d;
    }
    return true;
}

void AppendUtf8(std::string& s, u32 c) {
    if (c < 0x80) {
        s.push_back(char(c));
    } else if (c < 0x800) {
        s.push_back(char(0xC0 | c >> 6)), s.push_back(char(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        s.push_back(char(0xE0 | c >> 12)), s.push_back(char(0x80 | (c >> 6 & 0x3F)));
        s.push_back(char(0x80 | (c & 0x3F)));
    } else {
        s.push_back(char(0xF0 | c >> 18)), s.push_back(char(0x80 | (c >> 12 & 0x3F)));
        s.push_back(char(0x80 | (c >> 6 & 0x3F))), s.push_back(char(0x80 | (c & 0x3F)));
    }
}
using Fmg = std::map<s32, std::string>;
bool ReadFmg(const BndFile& f, Fmg& out) {
    const u8* p = f.data;
    const u64 n = f.size;
    if (n < 0x28 || p[2] != 2) return false;
    const u32 groups = Le32(p + 0x0c);
    const u64 table = Le64(p + 0x18);
    if (0x28 + u64(groups) * 16 > n || table > n) return false;
    for (u32 g = 0; g < groups; ++g) {
        const u8* e = p + 0x28 + u64(g) * 16;
        const s32 idx = s32(Le32(e)), first = s32(Le32(e + 4)), last = s32(Le32(e + 8));
        if (idx < 0 || last < first || last - first > 100000) return false;
        for (s32 k = 0; k <= last - first; ++k) {
            const u64 slot = table + u64(idx + k) * 8;
            if (slot + 8 > n) return false;
            const u64 off = Le64(p + slot);
            if (off == 0 || off >= n) continue;
            std::string s;
            for (u64 q = off; q + 1 < n;) {
                u32 c = Le16(p + q);
                q += 2;
                if (c == 0) break;
                if (c >= 0xD800 && c < 0xDC00 && q + 1 < n) {
                    const u32 lo = Le16(p + q);
                    if (lo >= 0xDC00 && lo < 0xE000) {
                        c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00), q += 2;
                    }
                }
                AppendUtf8(s, c);
            }
            out[first + k] = std::move(s);
        }
    }
    return true;
}

// PS4 TPF, one BC7 texture: returns the linear BC7 blocks.
bool ReadTpfBc7(const Bytes& t, Bytes& blocks, u32& w, u32& h) {
    if (t.size() < 0x30 || std::memcmp(t.data(), "TPF\0", 4) != 0 || Le32(t.data() + 8) != 1) {
        return false;
    }
    const u8* e = t.data() + 0x10;
    const u64 off = Le32(e), size = Le32(e + 4);
    const u8 format = e[8];
    w = Le16(e + 12), h = Le16(e + 14);
    if (format != 102 || w == 0 || h == 0 || w > 2048 || h > 2048 || w % 4 || h % 4 ||
        off + size > t.size()) {
        return false;
    }
    const u32 bw = w / 4, bh = h / 4, pw = (bw + 7) / 8 * 8, ph = (bh + 7) / 8 * 8;
    if (u64(pw) * ph * 16 > size) return false;
    blocks.assign(u64(bw) * bh * 16, 0);
    const u8* src = t.data() + off;
    const u32 tiles_w = pw / 8;
    for (u32 tile = 0; tile < tiles_w * (ph / 8); ++tile) {
        const u32 tx = tile % tiles_w, ty = tile / tiles_w;
        for (u32 i = 0; i < 64; ++i) {
            u32 x = 0, y = 0;
            for (u32 b = 0; b < 3; ++b) {
                x |= ((i >> (2 * b)) & 1) << b;
                y |= ((i >> (2 * b + 1)) & 1) << b;
            }
            const u32 X = tx * 8 + x, Y = ty * 8 + y;
            if (X < bw && Y < bh) {
                std::memcpy(&blocks[(u64(Y) * bw + X) * 16], src + (u64(tile) * 64 + i) * 16, 16);
            }
        }
    }
    return true;
}

bool BadName(const std::string& n) {
    return n.empty() || n[0] == '*' || n.find("NOT NEEDED") != std::string::npos ||
           n.find("innecesario") != std::string::npos || n.find("(unneeded)") != std::string::npos;
}
std::string Trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n')) s.pop_back();
    return s;
}

// ---------------------------------------------------------------- state
std::mutex g_mtx; // catalog, state, message
State g_state = State::Idle;
std::string g_message;
std::deque<Item> g_items; // stable addresses
std::atomic<int> g_progress{0}, g_total{0};
std::jthread g_loader;

struct Purchase {
    s32 id, category, price;
    std::string name;
};
std::mutex g_buy_mtx;
std::deque<Purchase> g_queue;
std::atomic<bool> g_pending{false};
std::string g_result;
std::atomic<u32> g_result_serial{0};

std::optional<Bytes> ReadGame(const std::string& rel) {
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    auto raw = mnt->ReadFile("/app0/dvdroot_ps4/" + rel);
    if (!raw) return std::nullopt;
    Bytes out;
    if (!Undcx(*raw, out)) return std::nullopt;
    return out;
}

void SetState(State s, std::string msg) {
    std::scoped_lock lk{g_mtx};
    g_state = s;
    g_message = std::move(msg);
}

void Load(std::stop_token stop) {
    const auto fail = [](std::string why) {
        LOG_WARNING(Frontend, "[shop] {}", why);
        SetState(State::Failed, std::move(why));
    };
    // params
    const auto pb = ReadGame("param/gameparam/gameparam.parambnd.dcx");
    std::vector<BndFile> pf;
    if (!pb || !Bnd4(*pb, pf)) return fail("No se pudo leer gameparam.parambnd.dcx");
    Param goods, weapons, shop, lots;
    for (const auto& f : pf) {
        if (f.name == "EquipParamGoods.param") ReadParam(f, goods, 108);
        if (f.name == "EquipParamWeapon.param") ReadParam(f, weapons, 316);
        if (f.name == "ShopLineupParam.param") ReadParam(f, shop, 32);
        if (f.name == "ItemLotParam.param") ReadParam(f, lots, 148);
    }
    if (goods.rows.empty() || weapons.rows.empty() || shop.rows.empty() || lots.rows.empty()) {
        return fail("Formato de parámetros no reconocido (¿no es Bloodborne 1.09?)");
    }
    // text: Spanish (Spain), else English (US)
    std::map<s32, Fmg> text;
    const char* lang = nullptr;
    for (const char* l : {"spaes", "engus"}) {
        const auto mb = ReadGame(fmt::format("msg/{}/item.msgbnd.dcx", l));
        std::vector<BndFile> mf;
        if (!mb || !Bnd4(*mb, mf)) continue;
        for (const auto& f : mf) {
            if (f.id == 10 || f.id == 11 || f.id == 20 || f.id == 21 || f.id == 24 || f.id == 25) {
                ReadFmg(f, text[f.id]);
            }
        }
        if (!text[10].empty() && !text[11].empty()) {
            lang = l;
            break;
        }
        text.clear();
    }
    if (lang == nullptr) return fail("No se pudieron leer los textos (msg/spaes o msg/engus)");
    if (stop.stop_requested()) return;

    // obtainable: item lots and echo shop lineups
    std::set<s32> lot_weapon, lot_goods;
    for (const auto& [id, r] : lots.rows) {
        for (int k = 0; k < 8; ++k) {
            const s32 item = s32(Le32(r + 4 * k)), cat = s32(Le32(r + 0x20 + 4 * k));
            if (item <= 0) continue;
            if (cat == 0) lot_weapon.insert(item);
            if (cat == 4) lot_goods.insert(item);
        }
    }
    std::map<s32, s32> price_weapon, price_goods;
    for (const auto& [id, r] : shop.rows) {
        const s32 item = s32(Le32(r)), value = s32(Le32(r + 4));
        if (value <= 0 || r[0x16] != 0) continue; // echoes only
        auto& m = r[0x17] == 0 ? price_weapon : r[0x17] == 3 ? price_goods : price_weapon;
        if (r[0x17] != 0 && r[0x17] != 3) continue;
        auto it = m.find(item);
        if (it == m.end() || value < it->second) m[item] = value;
    }
    std::vector<Item> list;
    const auto text_of = [&](s32 fmg, s32 id) {
        auto t = text.find(fmg);
        if (t == text.end()) return std::string{};
        auto it = t->second.find(id);
        return it == t->second.end() || BadName(it->second) ? std::string{} : Trim(it->second);
    };
    for (const auto& [id, r] : weapons.rows) {
        if (id <= 0 || id % 10000 != 0 || (id / 10000) % 10 != 0 || id == 1000000) continue;
        if (s32(Le32(r + 0x5c)) != id) continue;
        const std::string name = text_of(11, id);
        if (name.empty()) continue;
        const auto pr = price_weapon.find(id);
        if (pr == price_weapon.end() && !lot_weapon.contains(id)) continue;
        Item it;
        it.id = id, it.tab = Tab::Weapons, it.icon = Le16(r + 0xba), it.name = name;
        it.caption = text_of(21, id), it.lore = text_of(25, id);
        it.game_price = pr != price_weapon.end();
        it.price = it.game_price ? pr->second : kFallbackPrice[0];
        list.push_back(std::move(it));
    }
    std::set<std::string> names;
    for (const auto& [id, r] : goods.rows) {
        const u8 type = r[0x3a];
        Tab tab;
        if (type == 0 && ((id >= 900 && id <= 1594 && id != 901) || id == 1700)) {
            tab = Tab::Items;
        } else if (type == 0 && id >= 2000 && id < 3000) {
            tab = Tab::Tools;
        } else if (type == 2) {
            tab = Tab::Materials;
        } else {
            continue;
        }
        const std::string name = text_of(10, id);
        if (name.empty() || names.contains(name)) continue;
        const auto pr = price_goods.find(id);
        if (pr == price_goods.end() && !lot_goods.contains(id)) continue;
        names.insert(name);
        Item it;
        it.id = id, it.tab = tab, it.icon = Le16(r + 0x2c), it.name = name;
        it.caption = text_of(20, id), it.lore = text_of(24, id);
        it.game_price = pr != price_goods.end();
        it.price = it.game_price ? pr->second : kFallbackPrice[static_cast<int>(tab)];
        list.push_back(std::move(it));
    }
    g_total = static_cast<int>(list.size());
    // icons: an item without a readable icon is left out
    int kept = 0, no_icon = 0;
    for (auto& it : list) {
        if (stop.stop_requested()) return;
        const auto tb = ReadGame(fmt::format("menu/menu_knowledge_{:05}.tpf.dcx", it.icon));
        Bytes blocks;
        u32 w = 0, h = 0;
        ++g_progress;
        if (!tb || !ReadTpfBc7(*tb, blocks, w, h)) {
            ++no_icon;
            LOG_INFO(Frontend, "[shop] {} ({}): icon {} not readable, left out", it.name, it.id,
                     it.icon);
            continue;
        }
        it.tex = ImGui::RefCountedTexture::UploadBc7(std::move(blocks), w, h);
        std::scoped_lock lk{g_mtx};
        g_items.push_back(std::move(it));
        ++kept;
    }
    LOG_INFO(Frontend, "[shop] {} items ({}), {} without an icon left out", kept, lang, no_icon);
    SetState(State::Ready, "");
}

bool CoopRoom() {
    return Core::BBCoop::IsEnabled() && EmulatorSettings.IsShadNetEnabled() &&
           Libraries::Np::NpHandler::GetInstance().GetCoopSnapshot().in_room;
}

void Result(std::string s) {
    LOG_INFO(Frontend, "[shop] {}", s);
    std::scoped_lock lk{g_buy_mtx};
    g_result = std::move(s);
    ++g_result_serial;
}

} // namespace

void OnEbootLoaded(uintptr_t eboot_base, u64 image_size) {
    g_base = eboot_base;
    g_eboot_ok = AddItemVa + 0x40 < image_size && ItemGetManVa + 8 < image_size &&
                 std::memcmp(reinterpret_cast<const void*>(eboot_base + AddItemVa), kAddItemSig,
                             sizeof(kAddItemSig)) == 0;
    if (g_eboot_ok) {
        s32 d = 0;
        std::memcpy(&d, reinterpret_cast<const void*>(eboot_base + AddItemVa + 0x2e + 3), 4);
        g_eboot_ok = s64(AddItemVa) + 0x2e + 8 + d == s64(ItemGetManVa);
    }
    if (!g_eboot_ok) {
        LOG_INFO(Frontend, "[shop] add-item function not found (not 1.09): buying off");
    }
}

void Open() {
    std::scoped_lock lk{g_mtx};
    if (g_state != State::Idle) return;
    g_state = State::Loading;
    g_message.clear();
    g_progress = 0, g_total = 0;
    g_loader = std::jthread([](std::stop_token st) { Load(st); });
}

void Close() {
    if (g_loader.joinable()) {
        g_loader.request_stop();
        g_loader.join();
    }
    std::scoped_lock lk{g_mtx};
    g_items.clear(); // frees the textures
    g_state = State::Idle;
    g_message.clear();
}

State GetState(std::string* message, int* progress, int* total) {
    std::scoped_lock lk{g_mtx};
    if (message) *message = g_message;
    if (progress) *progress = g_progress;
    if (total) *total = g_total;
    return g_state;
}

std::vector<const Item*> Items(Tab tab) {
    std::scoped_lock lk{g_mtx};
    std::vector<const Item*> out;
    for (const auto& it : g_items) {
        if (it.tab == tab) out.push_back(&it);
    }
    return out;
}

bool BuyAvailable(std::string* why) {
    if (!g_eboot_ok) {
        if (why) *why = "No disponible en este ejecutable";
        return false;
    }
    if (CoopRoom()) {
        if (why) *why = "En multijugador no se puede comprar";
        return false;
    }
    return true;
}

bool Buy(const Item& item, std::string& why) {
    if (!BuyAvailable(&why)) return false;
    std::scoped_lock lk{g_buy_mtx};
    if (g_queue.size() >= 32) {
        why = "Demasiadas compras pendientes";
        return false;
    }
    g_queue.push_back({item.id, item.tab == Tab::Weapons ? CatWeapon : CatGoods, item.price,
                       item.name});
    g_pending = true;
    return true;
}

std::string LastResult(u32* serial) {
    std::scoped_lock lk{g_buy_mtx};
    if (serial) *serial = g_result_serial;
    return g_result;
}

bool HasPending() {
    return g_pending.load();
}

void OnGuestFlip(u64 pgd) {
    if (!g_pending.load() || !g_eboot_ok || pgd == 0) return;
    std::deque<Purchase> q;
    {
        std::scoped_lock lk{g_buy_mtx};
        q.swap(g_queue);
        g_pending = false;
    }
    u64 man = 0;
    std::memcpy(&man, reinterpret_cast<const void*>(g_base + ItemGetManVa), 8);
    for (const auto& p : q) {
        if (man == 0) {
            Result(fmt::format("{}: el juego no está listo, no se ha cobrado", p.name));
            continue;
        }
        // take the echoes (refused when there are not enough)
        std::atomic_ref<s32> echoes{*reinterpret_cast<s32*>(pgd + PgdEchoes)};
        s32 cur = echoes.load();
        bool paid = false;
        while (cur >= p.price) {
            if (echoes.compare_exchange_weak(cur, cur - p.price)) {
                paid = true;
                break;
            }
        }
        if (!paid) {
            Result(fmt::format("{}: no tienes ecos suficientes ({} de {})", p.name, cur, p.price));
            continue;
        }
        const auto add = reinterpret_cast<AddItemFn>(g_base + AddItemVa);
        const s32 r = add(0, p.category, p.id, 1, -1);
        if (r != 0) {
            // refund (clamped as the game clamps)
            s32 c = echoes.load();
            while (!echoes.compare_exchange_weak(
                c, s32(std::min<s64>(s64(c) + p.price, EchoesMax)))) {
            }
            Result(fmt::format("{}: el juego no lo ha aceptado (inventario lleno, código {}); "
                               "ecos devueltos",
                               p.name, r));
            continue;
        }
        Result(fmt::format("Comprado: {} por {} ecos", p.name, p.price));
    }
}

} // namespace BB::Shop
