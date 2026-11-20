// (f5) The pass registry (pass-registry.h).
#include "native/pass-registry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace NativeRender {
namespace {

constexpr StaticPass StaticTable[] = {
#include "native/pass-table.inc"
};
constexpr size_t StaticCount = sizeof(StaticTable) / sizeof(StaticTable[0]);
static_assert(StaticCount < MaxPasses, "the static table leaves no room for computed names");

struct Row {
	uint64_t  name_offset = 0;
	char      name[64]    = {};
	uint16_t  category    = 0xffff;
	uint8_t   queues      = 0;
	bool      dynamic     = false;
	PassState requested   = PassState::Emulated;
};

struct Requested {
	std::string name;
	PassState   state = PassState::Emulated;
};

Row                    g_rows[MaxPasses];
std::atomic<uint32_t>  g_count {0};
std::atomic<bool>      g_ready {false};
uint64_t               g_base = 0;
uint64_t               g_size = 0;
std::vector<Requested> g_requested; // KYTY_NATIVE_PASSES, parsed once in Init

PassState RequestedFor(const char* name) {
	for (const auto& r: g_requested)
		if (PassRegistry::SameName(r.name.c_str(), name)) return r.state;
	return PassState::Emulated;
}

void ParseRequested() {
	const char* v = std::getenv("KYTY_NATIVE_PASSES");
	if (v == nullptr) return;
	std::string s(v);
	size_t      at = 0;
	while (at < s.size()) {
		size_t end = s.find(',', at);
		if (end == std::string::npos) end = s.size();
		const std::string item  = s.substr(at, end - at);
		const size_t      colon = item.rfind(':');
		Requested         r;
		r.name = colon == std::string::npos ? item : item.substr(0, colon);
		while (!r.name.empty() && r.name.front() == ' ') r.name.erase(r.name.begin());
		const int st = colon == std::string::npos ? 1 : std::atoi(item.c_str() + colon + 1);
		r.state      = st == 2 ? PassState::Verify : st == 1 ? PassState::Native : PassState::Emulated;
		if (!r.name.empty()) g_requested.push_back(r);
		at = end + 1;
	}
}

// A printable name of 2..63 characters at an eboot address (read-only data of the module image).
bool ReadName(uint64_t ptr, char* out) {
	if (!PassRegistry::InEboot(ptr)) return false;
	const uint64_t room = g_base + g_size - ptr;
	const auto*    p    = reinterpret_cast<const char*>(ptr);
	size_t         n    = 0;
	for (; n < 63 && n < room; ++n) {
		const char c = p[n];
		if (c == '\0') break;
		if (c < 0x20 || c > 0x7e) return false;
		out[n] = c;
	}
	if (n < 2 || n >= 63 || n >= room || p[n] != '\0') return false;
	out[n] = '\0';
	return true;
}

} // namespace

void PassRegistry::Init(uint64_t eboot_base, uint64_t eboot_size) {
	if (g_ready.load(std::memory_order_acquire)) return;
	g_base = eboot_base;
	g_size = eboot_size;
	ParseRequested();
	uint32_t n = 0;
	for (const auto& s: StaticTable) {
		Row& r        = g_rows[n++];
		r.name_offset = s.name_offset;
		std::snprintf(r.name, sizeof(r.name), "%s", s.name);
		r.category  = s.category;
		r.queues    = s.static_queues;
		r.dynamic   = false;
		r.requested = RequestedFor(r.name);
	}
	g_count.store(n, std::memory_order_release);
	g_ready.store(true, std::memory_order_release);
	// (f5b) The only implementation in this build is the verification framework's control pass (pass-verify.cpp): it
	// exists for every pass, in Verify only.
	for (const auto& q: g_requested)
		std::printf("[f5] KYTY_NATIVE_PASSES: %s -> %s%s\n", q.name.c_str(),
		            q.state == PassState::Verify ? "verificar" : q.state == PassState::Native ? "nativa" : "emulada",
		            q.state == PassState::Verify ? " con la pasada de CONTROL (tiene que dar 0 DISCREPANCIA)"
		            : q.state == PassState::Native ? "; ninguna pasada nativa real en esta build: sigue emulada"
		                                            : "");
}

bool PassRegistry::Ready() noexcept { return g_ready.load(std::memory_order_acquire); }
uint64_t PassRegistry::EbootBase() noexcept { return g_base; }
bool PassRegistry::InEboot(uint64_t address) noexcept {
	return g_size != 0 && address >= g_base && address - g_base < g_size;
}

uint32_t PassRegistry::IdOfName(uint64_t name_ptr) {
	if (!Ready() || !InEboot(name_ptr)) return NoPass;
	const uint64_t off = name_ptr - g_base;
	// Static rows are sorted by offset (mapa.py writes them so).
	const auto* it = std::lower_bound(StaticTable, StaticTable + StaticCount, off,
	                                  [](const StaticPass& s, uint64_t o) { return s.name_offset < o; });
	if (it != StaticTable + StaticCount && it->name_offset == off) return static_cast<uint32_t>(it - StaticTable);
	const uint32_t n = g_count.load(std::memory_order_relaxed);
	for (uint32_t i = static_cast<uint32_t>(StaticCount); i < n; ++i)
		if (g_rows[i].name_offset == off) return i;
	if (n >= MaxPasses) return NoPass;
	Row r;
	if (!ReadName(name_ptr, r.name)) return NoPass;
	r.name_offset = off;
	r.dynamic     = true;
	r.requested   = RequestedFor(r.name);
	g_rows[n]     = r;
	g_count.store(n + 1, std::memory_order_release);
	return n;
}

uint32_t PassRegistry::Count() noexcept { return g_count.load(std::memory_order_acquire); }
const char* PassRegistry::Name(uint32_t id) noexcept { return id < Count() ? g_rows[id].name : "?"; }
uint64_t PassRegistry::NameOffset(uint32_t id) noexcept { return id < Count() ? g_rows[id].name_offset : 0; }
bool PassRegistry::IsDynamic(uint32_t id) noexcept { return id < Count() && g_rows[id].dynamic; }
uint16_t PassRegistry::Category(uint32_t id) noexcept { return id < Count() ? g_rows[id].category : 0xffff; }
uint8_t PassRegistry::StaticQueues(uint32_t id) noexcept { return id < Count() ? g_rows[id].queues : 0; }
PassState PassRegistry::Requested(uint32_t id) noexcept { return id < Count() ? g_rows[id].requested : PassState::Emulated; }
// (f5b) The control pass (pass-verify.cpp) is the only implementation: Verify only.
bool PassRegistry::HasNative(uint32_t id) noexcept { return Requested(id) == PassState::Verify; }
bool PassRegistry::SameName(const char* a, const char* b) noexcept {
	const auto skip = [](const char*& p) {
		while (*p == ' ' || *p == '_') ++p;
	};
	const auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; };
	if (a == nullptr || b == nullptr) return false;
	for (;;) {
		skip(a);
		skip(b);
		if (*a == '\0' || *b == '\0') return *a == *b;
		if (lower(*a) != lower(*b)) return false;
		++a;
		++b;
	}
}
PassState PassRegistry::Effective(uint32_t id) noexcept {
	return HasNative(id) ? Requested(id) : PassState::Emulated;
}
const StaticPass* PassRegistry::StaticRows(size_t* count) noexcept {
	if (count != nullptr) *count = StaticCount;
	return StaticTable;
}

} // namespace NativeRender
