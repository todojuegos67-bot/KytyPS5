// KYTY_NATIVE_TEXTURE_PC: see nativeTexturePc.h. Vulkan-free on purpose (the lab's model tests link this file alone).
#include "graphics/host_gpu/nativeTexturePc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::NativeTexturePc {

namespace {

using Bytes = std::shared_ptr<const std::vector<uint8_t>>;

struct Chunk {
	uint64_t end    = 0;
	Bytes    bytes;
	uint64_t offset = 0; // of `begin` in *bytes
	uint64_t owner  = 0;
	bool     materialized = false; // written into guest memory by a fallback (BeforeGuestRead)
};

struct Record {
	Texture  texture;
	uint64_t lo = UINT64_MAX, hi = 0; // span of the chunks staged for it
};

struct Extent {
	uint64_t end   = 0;
	uint64_t owner = 0;
};

// A piece of a range: staged bytes, owned-undefined (zeros) or guest memory.
struct Part {
	uint64_t begin = 0, end = 0;
	Bytes    bytes;
	uint64_t offset = 0;
	enum : uint8_t { Staged, Zero, Guest } kind = Guest;
};

struct State {
	std::mutex                           m;
	std::map<uint64_t, Chunk>            chunks;  // by begin; never overlapping
	std::map<uint64_t, Extent>           extents; // engine texture memory with a known size (factory textures), by base
	std::map<uint64_t, uint64_t>         bases;   // every registered texture's base -> owner
	std::unordered_map<uint64_t, Record> owners;
	Host                                 host;
};
State&             S() {
	static State s;
	return s;
}
std::atomic<bool>     g_enabled {false};
std::atomic<uint64_t> g_any {0}; // chunks + extents: the cheap "nothing here" test

struct Counters {
	std::atomic<uint64_t> textures {0}, textures_loaded {0}, textures_created {0}, released {0};
	std::atomic<uint64_t> loads {0}, load_bytes {0}, mips {0}, mip_bytes {0}, slice_copies {0}, slice_bytes {0},
	    slice_guest_reads {0};
	std::atomic<uint64_t> dropped_bytes {0}, held_bytes {0}, held_chunks {0};
	std::atomic<uint64_t> fills {0}, fill_bytes {0}, fill_ns {0}, frames {0};
	std::atomic<uint64_t> fallbacks {0}, fallback_bytes {0}, declined {0};
};
Counters g_c;

thread_local int t_inside = 0; // BeforeGuestRead re-entered from our own guest reads/writes: nothing to do

void Recount(State& s) { g_any.store(s.chunks.size() + s.extents.size(), std::memory_order_relaxed); }

void Held(int64_t bytes, int64_t chunks) {
	g_c.held_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
	g_c.held_chunks.fetch_add(static_cast<uint64_t>(chunks), std::memory_order_relaxed);
}

// Removes staged bytes in [b, e) (pieces outside kept). Returns the bytes removed.
uint64_t CutLocked(State& s, uint64_t b, uint64_t e, uint64_t only_owner = 0) {
	if (b >= e) return 0;
	uint64_t removed = 0;
	auto     it      = s.chunks.upper_bound(b);
	if (it != s.chunks.begin()) --it;
	while (it != s.chunks.end() && it->first < e) {
		const uint64_t cb = it->first;
		Chunk          c  = it->second;
		if (c.end <= b || (only_owner != 0 && c.owner != only_owner)) {
			++it;
			continue;
		}
		it = s.chunks.erase(it);
		Held(-static_cast<int64_t>(c.end - cb), -1);
		if (cb < b) { // left piece stays
			Chunk left = c;
			left.end   = b;
			s.chunks.emplace(cb, left);
			Held(static_cast<int64_t>(b - cb), 1);
		}
		if (c.end > e) { // right piece stays
			Chunk right  = c;
			right.offset = c.offset + (e - cb);
			it           = s.chunks.emplace(e, right).first;
			Held(static_cast<int64_t>(c.end - e), 1);
			++it;
		}
		removed += std::min(c.end, e) - std::max(cb, b);
	}
	return removed;
}

void InsertLocked(State& s, uint64_t begin, uint64_t size, Bytes bytes, uint64_t offset, uint64_t owner) {
	(void)CutLocked(s, begin, begin + size);
	Chunk c;
	c.end    = begin + size;
	c.bytes  = std::move(bytes);
	c.offset = offset;
	c.owner  = owner;
	s.chunks.emplace(begin, std::move(c));
	Held(static_cast<int64_t>(size), 1);
	if (owner != 0) {
		auto& r = s.owners[owner];
		r.lo    = std::min(r.lo, begin);
		r.hi    = std::max(r.hi, begin + size);
	}
	Recount(s);
}

bool InExtentLocked(const State& s, uint64_t addr, uint64_t* end) {
	auto it = s.extents.upper_bound(addr);
	if (it == s.extents.begin()) return false;
	--it;
	if (addr >= it->second.end) return false;
	*end = it->second.end;
	return true;
}

// The parts of [b, e): staged / zero (engine texture memory, nothing staged) / guest.
std::vector<Part> PartsLocked(const State& s, uint64_t b, uint64_t e) {
	std::vector<Part> parts;
	auto              it = s.chunks.upper_bound(b);
	if (it != s.chunks.begin()) --it;
	uint64_t at = b;
	const auto gap = [&](uint64_t gb, uint64_t ge) {
		while (gb < ge) {
			uint64_t xend = 0;
			if (InExtentLocked(s, gb, &xend)) {
				const uint64_t pe = std::min(ge, xend);
				parts.push_back({gb, pe, nullptr, 0, Part::Zero});
				gb = pe;
				continue;
			}
			auto next = s.extents.upper_bound(gb);
			const uint64_t pe = next != s.extents.end() ? std::min(ge, next->first) : ge;
			parts.push_back({gb, pe, nullptr, 0, Part::Guest});
			gb = pe;
		}
	};
	for (; it != s.chunks.end() && it->first < e; ++it) {
		const uint64_t cb = it->first, ce = it->second.end;
		if (ce <= at) continue;
		const uint64_t pb = std::max(cb, at), pe = std::min(ce, e);
		if (pb > at) gap(at, pb);
		parts.push_back({pb, pe, it->second.bytes, it->second.offset + (pb - cb), Part::Staged});
		at = pe;
	}
	if (at < e) gap(at, e);
	return parts;
}

// Copies the parts into out (out[0] = byte b). False: a guest read failed.
bool Assemble(const std::vector<Part>& parts, uint64_t b, uint8_t* out,
              const std::function<bool(uint64_t, void*, uint64_t)>& read_guest, uint64_t* guest_reads) {
	bool ok = true;
	for (const auto& p: parts) {
		uint8_t* to = out + (p.begin - b);
		const uint64_t n = p.end - p.begin;
		switch (p.kind) {
			case Part::Staged: std::memcpy(to, p.bytes->data() + p.offset, n); break;
			case Part::Zero: std::memset(to, 0, n); break;
			case Part::Guest:
				if (guest_reads != nullptr) ++*guest_reads;
				t_inside++;
				if (!read_guest || !read_guest(p.begin, to, n)) {
					std::memset(to, 0, n);
					ok = false;
				}
				t_inside--;
				break;
		}
	}
	return ok;
}

double MiB(uint64_t v) { return static_cast<double>(v) / (1024.0 * 1024.0); }

} // namespace

bool Requested() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_TEXTURE_PC");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

bool Enabled() noexcept { return g_enabled.load(std::memory_order_acquire); }
void SetEnabled(bool on) noexcept { g_enabled.store(on, std::memory_order_release); }

void SetHost(const Host& host) noexcept {
	auto&      s = S();
	const std::lock_guard lock(s.m);
	s.host = host;
}

void OnTexture(const Texture& texture) noexcept {
	if (!Enabled() || texture.owner == 0 || texture.base == 0) return;
	auto& s = S();
	{
		const std::lock_guard lock(s.m);
		auto&                 r = s.owners[texture.owner];
		r.texture               = texture;
		s.bases[texture.base]   = texture.owner;
		if (texture.size != 0 && texture.kind != Kind::Created) s.extents[texture.base] = {texture.base + texture.size, texture.owner};
		Recount(s);
	}
	g_c.textures.fetch_add(1, std::memory_order_relaxed);
	(texture.kind == Kind::Created ? g_c.textures_created : g_c.textures_loaded).fetch_add(1, std::memory_order_relaxed);
}

bool OnLoad(uint64_t owner, uint64_t owner_base, uint64_t dst, const void* src, uint64_t size, bool mip) noexcept {
	if (!Enabled() || src == nullptr || size == 0 || dst == 0) {
		g_c.declined.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	std::shared_ptr<std::vector<uint8_t>> copy;
	{
		auto* v = new (std::nothrow) std::vector<uint8_t>();
		if (v == nullptr) return false;
		copy.reset(v);
	}
	copy->resize(size);
	if (copy->size() != size) {
		g_c.declined.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	std::memcpy(copy->data(), src, size); // (the one CPU copy of the load: resource heap block -> host staging)
	auto& s = S();
	Host  host;
	{
		const std::lock_guard lock(s.m);
		if (owner != 0 && owner_base != 0 && s.bases.find(owner_base) == s.bases.end()) s.bases[owner_base] = owner;
		InsertLocked(s, dst, size, std::move(copy), 0, owner);
		host = s.host;
	}
	if (mip) {
		g_c.mips.fetch_add(1, std::memory_order_relaxed);
		g_c.mip_bytes.fetch_add(size, std::memory_order_relaxed);
	} else {
		g_c.loads.fetch_add(1, std::memory_order_relaxed);
		g_c.load_bytes.fetch_add(size, std::memory_order_relaxed);
	}
	if (host.staged != nullptr) host.staged(dst, size); // (after the bytes are here: an upload racing it re-runs)
	return true;
}

void OnDrop(uint64_t addr, uint64_t size) noexcept {
	if (!Enabled() || size == 0 || g_any.load(std::memory_order_relaxed) == 0) return;
	auto& s = S();
	const std::lock_guard lock(s.m);
	g_c.dropped_bytes.fetch_add(CutLocked(s, addr, addr + size), std::memory_order_relaxed);
	Recount(s);
}

bool OnSliceCopy(uint64_t dst, uint64_t src, uint64_t size,
                 const std::function<bool(uint64_t, void*, uint64_t)>& read_guest) noexcept {
	if (!Enabled() || size == 0) return false;
	auto&             s = S();
	std::vector<Part> parts;
	uint64_t          dst_owner = 0;
	{
		const std::lock_guard lock(s.m);
		parts            = PartsLocked(s, src, src + size);
		const bool ours  = std::any_of(parts.begin(), parts.end(), [](const Part& p) { return p.kind != Part::Guest; });
		if (!ours) {
			// Not an engine texture this path loaded: guest memory is authoritative for the source, and becomes so for the
			// destination (the caller copies guest memory).
			(void)CutLocked(s, dst, dst + size);
			Recount(s);
			g_c.declined.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		auto b = s.bases.upper_bound(dst);
		if (b != s.bases.begin()) dst_owner = std::prev(b)->second;
	}
	Bytes    bytes;
	uint64_t offset = 0, guest_reads = 0;
	if (parts.size() == 1 && parts[0].kind == Part::Staged) {
		bytes  = parts[0].bytes; // (shared: no byte copied)
		offset = parts[0].offset;
	} else {
		auto* v = new (std::nothrow) std::vector<uint8_t>();
		if (v == nullptr) return false;
		std::shared_ptr<std::vector<uint8_t>> copy(v);
		copy->resize(size);
		if (copy->size() != size) return false;
		(void)Assemble(parts, src, copy->data(), read_guest, &guest_reads);
		bytes = std::move(copy);
	}
	Host host;
	{
		const std::lock_guard lock(s.m);
		InsertLocked(s, dst, size, std::move(bytes), offset, dst_owner);
		host = s.host;
	}
	g_c.slice_copies.fetch_add(1, std::memory_order_relaxed);
	g_c.slice_bytes.fetch_add(size, std::memory_order_relaxed);
	g_c.slice_guest_reads.fetch_add(guest_reads, std::memory_order_relaxed);
	if (host.staged != nullptr) host.staged(dst, size);
	return true;
}

void OnRelease(uint64_t owner, uint64_t base) noexcept {
	if (!Enabled() || owner == 0) return;
	auto& s = S();
	const std::lock_guard lock(s.m);
	auto it = s.owners.find(owner);
	if (it != s.owners.end()) {
		if (it->second.lo < it->second.hi)
			g_c.dropped_bytes.fetch_add(CutLocked(s, it->second.lo, it->second.hi, owner), std::memory_order_relaxed);
		s.owners.erase(it);
		g_c.released.fetch_add(1, std::memory_order_relaxed);
	}
	if (auto b = s.bases.find(base); b != s.bases.end() && b->second == owner) s.bases.erase(b);
	if (auto x = s.extents.find(base); x != s.extents.end() && x->second.owner == owner) s.extents.erase(x);
	Recount(s);
}

bool Overlaps(uint64_t addr, uint64_t size) noexcept {
	if (!Enabled() || size == 0 || g_any.load(std::memory_order_relaxed) == 0) return false;
	auto& s = S();
	const std::lock_guard lock(s.m);
	const auto parts = PartsLocked(s, addr, addr + size);
	return std::any_of(parts.begin(), parts.end(), [](const Part& p) { return p.kind != Part::Guest; });
}

bool Fill(uint64_t addr, uint64_t size, void* out, const std::function<bool(uint64_t, void*, uint64_t)>& read_guest) noexcept {
	const auto start = std::chrono::steady_clock::now();
	auto&      s     = S();
	std::vector<Part> parts;
	{
		const std::lock_guard lock(s.m);
		parts = PartsLocked(s, addr, addr + size);
	}
	const bool ok = Assemble(parts, addr, static_cast<uint8_t*>(out), read_guest, nullptr);
	g_c.fills.fetch_add(1, std::memory_order_relaxed);
	g_c.fill_bytes.fetch_add(size, std::memory_order_relaxed);
	g_c.fill_ns.fetch_add(static_cast<uint64_t>(
	                          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()),
	                      std::memory_order_relaxed);
	return ok;
}

void BeforeGuestRead(uint64_t addr, uint64_t size) noexcept {
	if (t_inside != 0 || !Enabled() || size == 0 || g_any.load(std::memory_order_relaxed) == 0) return;
	auto&                                       s = S();
	std::vector<std::pair<uint64_t, Chunk>>     todo;
	Host                                        host;
	{
		const std::lock_guard lock(s.m);
		host    = s.host;
		auto it = s.chunks.upper_bound(addr);
		if (it != s.chunks.begin()) --it;
		for (; it != s.chunks.end() && it->first < addr + size; ++it) {
			if (it->second.end <= addr || it->second.materialized) continue;
			it->second.materialized = true;
			todo.emplace_back(it->first, it->second);
		}
	}
	if (todo.empty() || host.write_guest == nullptr) return;
	t_inside++;
	for (const auto& [b, c]: todo) {
		host.write_guest(b, c.bytes->data() + c.offset, c.end - b);
		g_c.fallbacks.fetch_add(1, std::memory_order_relaxed);
		g_c.fallback_bytes.fetch_add(c.end - b, std::memory_order_relaxed);
	}
	t_inside--;
}

Stats GetStats() noexcept {
	Stats r;
	r.textures          = g_c.textures.load();
	r.textures_loaded   = g_c.textures_loaded.load();
	r.textures_created  = g_c.textures_created.load();
	r.released          = g_c.released.load();
	r.loads             = g_c.loads.load();
	r.load_bytes        = g_c.load_bytes.load();
	r.mips              = g_c.mips.load();
	r.mip_bytes         = g_c.mip_bytes.load();
	r.slice_copies      = g_c.slice_copies.load();
	r.slice_bytes       = g_c.slice_bytes.load();
	r.slice_guest_reads = g_c.slice_guest_reads.load();
	r.dropped_bytes     = g_c.dropped_bytes.load();
	r.held_bytes        = g_c.held_bytes.load();
	r.held_chunks       = g_c.held_chunks.load();
	r.fills             = g_c.fills.load();
	r.fill_bytes        = g_c.fill_bytes.load();
	r.fill_ns           = g_c.fill_ns.load();
	r.frames            = g_c.frames.load();
	r.fallbacks         = g_c.fallbacks.load();
	r.fallback_bytes    = g_c.fallback_bytes.load();
	r.declined          = g_c.declined.load();
	return r;
}

void Pump() noexcept {
	if (!Enabled()) return;
	g_c.frames.fetch_add(1, std::memory_order_relaxed);
	using Clock = std::chrono::steady_clock;
	static Clock::time_point last = Clock::now();
	static Stats             prev {};
	const auto               now = Clock::now();
	if (now - last < std::chrono::seconds(30)) return;
	last          = now;
	const Stats c = GetStats();
	const uint64_t frames = c.frames - prev.frames;
	std::printf("NativeTexturePC: textures created natively %llu (loaded %llu, engine-made %llu, destroyed %llu) | file bytes "
	            "staged natively %llu loads %.1f MiB | mips streamed natively %llu (%.1f MiB) | slice copies host-side %llu "
	            "(%.1f MiB, %llu guest reads) | streamed out/freed %.1f MiB | guest MiB never written %.1f | held on host "
	            "%.1f MiB in %llu blocks | uploads from host staging %llu (%.1f MiB; %.2f ms/frame over %llu frames) | CPU "
	            "fallbacks %llu (%.1f MiB) | declined %llu\n",
	            static_cast<unsigned long long>(c.textures), static_cast<unsigned long long>(c.textures_loaded),
	            static_cast<unsigned long long>(c.textures_created), static_cast<unsigned long long>(c.released),
	            static_cast<unsigned long long>(c.loads), MiB(c.load_bytes), static_cast<unsigned long long>(c.mips),
	            MiB(c.mip_bytes), static_cast<unsigned long long>(c.slice_copies), MiB(c.slice_bytes),
	            static_cast<unsigned long long>(c.slice_guest_reads), MiB(c.dropped_bytes),
	            MiB(c.load_bytes + c.mip_bytes + c.slice_bytes - c.fallback_bytes), MiB(c.held_bytes),
	            static_cast<unsigned long long>(c.held_chunks), static_cast<unsigned long long>(c.fills), MiB(c.fill_bytes),
	            frames != 0 ? static_cast<double>(c.fill_ns - prev.fill_ns) / 1e6 / static_cast<double>(frames) : 0.0,
	            static_cast<unsigned long long>(frames), static_cast<unsigned long long>(c.fallbacks), MiB(c.fallback_bytes),
	            static_cast<unsigned long long>(c.declined));
	std::fflush(stdout);
	prev = c;
}

void ResetForTests() noexcept {
	auto& s = S();
	{
		const std::lock_guard lock(s.m);
		s.chunks.clear();
		s.extents.clear();
		s.bases.clear();
		s.owners.clear();
		s.host = {};
		Recount(s);
	}
	for (auto* a: {&g_c.textures, &g_c.textures_loaded, &g_c.textures_created, &g_c.released, &g_c.loads, &g_c.load_bytes,
	               &g_c.mips, &g_c.mip_bytes, &g_c.slice_copies, &g_c.slice_bytes, &g_c.slice_guest_reads,
	               &g_c.dropped_bytes, &g_c.held_bytes, &g_c.held_chunks, &g_c.fills, &g_c.fill_bytes, &g_c.fill_ns,
	               &g_c.frames, &g_c.fallbacks, &g_c.fallback_bytes, &g_c.declined})
		a->store(0);
}

} // namespace Libs::Graphics::NativeTexturePc
