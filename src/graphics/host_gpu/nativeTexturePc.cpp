// KYTY_NATIVE_TEXTURE_PC: see nativeTexturePc.h. Vulkan-free on purpose (the lab's model tests link this file alone with
// a fake device); the device side is the texture cache's (textureCache.cpp InitNativeTexturePc, bufferCache.cpp).
#include "graphics/host_gpu/nativeTexturePc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::NativeTexturePc {

namespace {

struct Counters {
	std::atomic<uint64_t> textures {0}, textures_loaded {0}, textures_created {0}, released {0};
	std::atomic<uint64_t> loads {0}, load_bytes {0}, mips {0}, mip_bytes {0}, slice_copies {0}, slice_bytes {0},
	    slice_guest_reads {0};
	std::atomic<uint64_t> dropped_bytes {0}, host_bytes {0}, host_blocks {0}, vram_bytes {0}, vram_buffers {0};
	std::atomic<uint64_t> to_vram {0}, to_vram_bytes {0}, upload_fail {0};
	std::atomic<uint64_t> fills {0}, fill_bytes {0}, fill_ns {0}, gathered {0}, frames {0};
	std::atomic<uint64_t> fallbacks {0}, fallback_bytes {0}, faults {0}, fallback_refused {0}, declined {0},
	    watched_bytes {0}, guest_writes {0}, guest_write_bytes {0};
	std::atomic<uint64_t> demoted {0}, demoted_bytes {0}, demote_passes {0};
};
Counters g_c;

// A device buffer: released (on the render thread, Flush) when nothing refers to it any more.
std::mutex            g_release_m;
std::vector<uint64_t> g_release;
struct Dev {
	uint64_t handle = 0, size = 0;
	Dev(uint64_t h, uint64_t s): handle(h), size(s) {
		g_c.vram_bytes.fetch_add(s, std::memory_order_relaxed);
		g_c.vram_buffers.fetch_add(1, std::memory_order_relaxed);
	}
	~Dev() {
		g_c.vram_bytes.fetch_sub(size, std::memory_order_relaxed);
		g_c.vram_buffers.fetch_sub(1, std::memory_order_relaxed);
		const std::lock_guard lock(g_release_m);
		g_release.push_back(handle);
	}
	Dev(const Dev&)            = delete;
	Dev& operator=(const Dev&) = delete;
};
using DevRef = std::shared_ptr<Dev>;
using Bytes  = std::shared_ptr<std::vector<uint8_t>>;

Bytes NewBytes(uint64_t size) {
	auto* v = new (std::nothrow) std::vector<uint8_t>();
	if (v == nullptr) return nullptr;
	v->resize(size);
	if (v->size() != size) {
		delete v;
		return nullptr;
	}
	g_c.host_bytes.fetch_add(size, std::memory_order_relaxed);
	g_c.host_blocks.fetch_add(1, std::memory_order_relaxed);
	return Bytes(v, [size](std::vector<uint8_t>* p) {
		g_c.host_bytes.fetch_sub(size, std::memory_order_relaxed);
		g_c.host_blocks.fetch_sub(1, std::memory_order_relaxed);
		delete p;
	});
}

struct Chunk {
	uint64_t end = 0;
	Bytes    host;          // the bytes on the host (until Flush), or
	DevRef   dev;           // in a device buffer, or
	bool     zero = false;  // undefined content (zeros)
	uint64_t offset = 0;    // of the chunk's begin in host / dev
	uint64_t owner  = 0;
	bool     materialized = false; // its bytes are in guest memory too (a fallback): its pages are not watched
	uint64_t used = 0;             // KYTY_VRAM_BUDGET: the frame of its last image upload (Plan) or of its load
};

struct Record {
	Texture  texture;
	uint64_t lo = UINT64_MAX, hi = 0;
};
struct Extent {
	uint64_t end = 0, owner = 0;
};
struct Op {
	uint64_t addr = 0, size = 0;
	bool     on   = false;
};

struct State {
	std::mutex                           m;
	std::mutex                           watch_m;
	std::condition_variable              materializing_cv; // a materialization over a range ended
	std::vector<std::pair<uint64_t, uint64_t>> materializing; // [b, e) being read back + written to guest (no lock held) // the watch calls in the order the state changed
	std::map<uint64_t, Chunk>            chunks;  // by begin; never overlapping
	std::map<uint64_t, Extent>           extents; // factory textures' memory, by base
	std::map<uint64_t, uint64_t>         bases;   // registered texture base -> owner
	std::unordered_map<uint64_t, Record> owners;
	std::map<uint64_t, uint64_t>         watched; // page-aligned intervals [b, e) we hold read watchers on
	struct Pending {
		std::weak_ptr<std::vector<uint8_t>> block;
		uint64_t                            b = 0, e = 0;
	};
	std::vector<Pending> pending; // host blocks Flush moves to the device (and the ranges that use them)
	Host                 host;
};
State& S() {
	static State s;
	return s;
}
std::atomic<bool>     g_enabled {false};
std::atomic<uint64_t> g_any {0};
std::atomic<int>      g_level_override {-1};
bool                  KeepGuest() noexcept;
thread_local int      t_inside = 0; // our own guest reads/writes: BeforeGuestRead does nothing

uint64_t PD(const State& s, uint64_t a) { return a / s.host.page_size * s.host.page_size; }
uint64_t PU(const State& s, uint64_t a) { return (a + s.host.page_size - 1) / s.host.page_size * s.host.page_size; }

void Recount(State& s) { g_any.store(s.chunks.size() + s.extents.size(), std::memory_order_relaxed); }

// Interval helpers over [b, e) lists.
using Ranges = std::vector<std::pair<uint64_t, uint64_t>>;
Ranges Subtract(const Ranges& a, const Ranges& b) {
	Ranges out;
	for (auto [ab, ae]: a) {
		for (const auto& [bb, be]: b) {
			if (be <= ab || bb >= ae) continue;
			if (bb > ab) out.emplace_back(ab, bb);
			ab = std::max(ab, be);
			if (ab >= ae) break;
		}
		if (ab < ae) out.emplace_back(ab, ae);
	}
	return out;
}

// The watchers over the pages of [b, e) made what the chunks there want: the whole pages of each run of adjacent
// non-materialized chunks. The calls are appended to `ops`.
void RewatchLocked(State& s, uint64_t b, uint64_t e, std::vector<Op>& ops) {
	if (s.host.watch == nullptr || KeepGuest()) return;
	const uint64_t wb = PD(s, b), we = PU(s, e);
	// Runs of chunks touching the window (extended by whole runs so a page is judged with all its chunks).
	Ranges runs;
	auto   it = s.chunks.upper_bound(wb);
	if (it != s.chunks.begin()) --it;
	for (; it != s.chunks.end() && it->first < we; ++it) {
		if (it->second.end <= wb || it->second.materialized) continue;
		if (!runs.empty() && runs.back().second == it->first) runs.back().second = it->second.end;
		else runs.emplace_back(it->first, it->second.end);
	}
	Ranges want;
	for (const auto& [rb, re]: runs) {
		const uint64_t pb = std::max(PU(s, rb), wb), pe = std::min(PD(s, re), we);
		if (pb < pe) want.emplace_back(pb, pe);
	}
	Ranges have;
	auto   w = s.watched.upper_bound(wb);
	if (w != s.watched.begin()) --w;
	for (; w != s.watched.end() && w->first < we; ++w) {
		const uint64_t hb = std::max(w->first, wb), he = std::min(w->second, we);
		if (hb < he) have.emplace_back(hb, he);
	}
	for (const auto& [ob, oe]: Subtract(have, want)) {
		ops.push_back({ob, oe - ob, false});
		g_c.watched_bytes.fetch_sub(oe - ob, std::memory_order_relaxed);
	}
	for (const auto& [ob, oe]: Subtract(want, have)) {
		ops.push_back({ob, oe - ob, true});
		g_c.watched_bytes.fetch_add(oe - ob, std::memory_order_relaxed);
	}
	// The window's part of `watched` replaced by `want`.
	Ranges keep;
	w = s.watched.upper_bound(wb);
	if (w != s.watched.begin()) --w;
	while (w != s.watched.end() && w->first < we) {
		if (w->second <= wb) {
			++w;
			continue;
		}
		if (w->first < wb) keep.emplace_back(w->first, wb);
		if (w->second > we) keep.emplace_back(we, w->second);
		w = s.watched.erase(w);
	}
	for (const auto& r: keep) s.watched.emplace(r.first, r.second);
	for (const auto& r: want) s.watched.emplace(r.first, r.second);
	// (coalescing is not needed: lookups are by containment)
}

// The calls made in state order: the watch lock taken before the state lock is let go.
void ApplyOps(State& s, std::unique_lock<std::mutex>& state, const std::vector<Op>& ops) {
	if (ops.empty()) {
		state.unlock();
		return;
	}
	std::unique_lock watch(s.watch_m);
	const auto       fn = s.host.watch;
	state.unlock();
	for (const auto& op: ops) fn(op.addr, op.size, op.on);
}

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
		if (cb < b) {
			Chunk left = c;
			left.end   = b;
			s.chunks.emplace(cb, left);
		}
		if (c.end > e) {
			Chunk right  = c;
			right.offset = c.offset + (e - cb);
			it           = s.chunks.emplace(e, right).first;
			++it;
		}
		removed += std::min(c.end, e) - std::max(cb, b);
	}
	return removed;
}

void InsertLocked(State& s, uint64_t begin, Chunk c) {
	(void)CutLocked(s, begin, c.end);
	if (c.owner != 0) {
		auto& r = s.owners[c.owner];
		r.lo    = std::min(r.lo, begin);
		r.hi    = std::max(r.hi, c.end);
	}
	if (c.used == 0) c.used = g_c.frames.load(std::memory_order_relaxed) + 1;
	s.chunks.emplace(begin, std::move(c));
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

std::vector<Part> PartsLocked(const State& s, uint64_t b, uint64_t e) {
	std::vector<Part> parts;
	auto              it = s.chunks.upper_bound(b);
	if (it != s.chunks.begin()) --it;
	uint64_t   at  = b;
	const auto gap = [&](uint64_t gb, uint64_t ge) {
		while (gb < ge) {
			uint64_t xend = 0;
			Part     p;
			p.begin = gb;
			if (InExtentLocked(s, gb, &xend)) {
				p.end  = std::min(ge, xend);
				p.kind = Part::Zero;
			} else {
				auto next = s.extents.upper_bound(gb);
				p.end     = next != s.extents.end() ? std::min(ge, next->first) : ge;
				p.kind    = Part::Guest;
			}
			gb = p.end;
			parts.push_back(std::move(p));
		}
	};
	for (; it != s.chunks.end() && it->first < e; ++it) {
		const uint64_t cb = it->first, ce = it->second.end;
		if (ce <= at) continue;
		const uint64_t pb = std::max(cb, at), pe = std::min(ce, e);
		if (pb > at) gap(at, pb);
		if (it->second.materialized) { // (its bytes are the guest's now: a CPU write may follow)
			Part g;
			g.begin = pb;
			g.end   = pe;
			g.kind  = Part::Guest;
			parts.push_back(std::move(g));
			at = pe;
			continue;
		}
		const Chunk& c = it->second;
		Part         p;
		p.begin = pb;
		p.end   = pe;
		if (c.dev) {
			p.kind   = Part::Device;
			p.handle = c.dev->handle;
			p.offset = c.offset + (pb - cb);
			p.keep   = c.dev;
		} else if (c.host) {
			p.kind   = Part::Host;
			p.host   = c.host->data() + c.offset + (pb - cb);
			p.keep   = c.host;
		} else {
			p.kind = Part::Zero;
		}
		parts.push_back(std::move(p));
		at = pe;
	}
	if (at < e) gap(at, e);
	return parts;
}

// The bytes of `parts` into out (out[0] = byte `b`). Device parts read back. False: something could not be read.
bool Assemble(const State& s, const std::vector<Part>& parts, uint64_t b, uint8_t* out,
              const std::function<bool(uint64_t, void*, uint64_t)>& read_guest, uint64_t* guest_reads) {
	bool ok = true;
	t_inside++;
	for (const auto& p: parts) {
		uint8_t*       to = out + (p.begin - b);
		const uint64_t n  = p.end - p.begin;
		switch (p.kind) {
			case Part::Host: std::memcpy(to, p.host, n); break;
			case Part::Zero: std::memset(to, 0, n); break;
			case Part::Device:
				if (s.host.read_device == nullptr || !s.host.read_device(p.handle, p.offset, to, n)) ok = false;
				break;
			case Part::Guest:
				if (guest_reads != nullptr) ++*guest_reads;
				if (!read_guest || !read_guest(p.begin, to, n)) {
					std::memset(to, 0, n);
					ok = false;
				}
				break;
		}
	}
	t_inside--;
	return ok;
}

// Level 2: a read-back this thread cannot do would leave wrong bytes: stop instead (never wrong bytes silently).
[[noreturn]] void Refused(const char* what, uint64_t addr, uint64_t size) {
	std::printf("NativeTexturePC: FATAL: KYTY_NATIVE_TEXTURE_PC=2 could not read back %s 0x%llx+0x%llx through its own "
	            "copy queue (device error: should not happen); use KYTY_NATIVE_TEXTURE_PC=1\n",
	            what, static_cast<unsigned long long>(addr), static_cast<unsigned long long>(size));
	std::fflush(stdout);
	std::abort();
}

// Edge pages of [b, e) shared with other data: the chunk's bytes written into guest memory there (they are never watched).
void WriteEdges(State& s, uint64_t b, uint64_t e) {
	if (s.host.write_guest == nullptr || s.host.watch == nullptr) return;
	Ranges edges;
	if (b % s.host.page_size != 0) edges.emplace_back(b, std::min(e, PU(s, b)));
	if (e % s.host.page_size != 0 && PD(s, e) >= PU(s, b)) edges.emplace_back(std::max(b, PD(s, e)), e);
	for (const auto& [eb, ee]: edges) {
		std::vector<Part> parts;
		{
			const std::lock_guard lock(s.m);
			parts = PartsLocked(s, eb, ee);
		}
		std::vector<uint8_t> bytes(ee - eb);
		if (Assemble(s, parts, eb, bytes.data(), nullptr, nullptr)) {
			t_inside++;
			s.host.write_guest(eb, bytes.data(), ee - eb);
			t_inside--;
		} else {
			g_c.fallback_refused.fetch_add(1, std::memory_order_relaxed);
			Refused("an edge page", eb, ee - eb);
		}
	}
}

// The non-materialized chunks over [b, e) written into guest memory and unwatched. False: some could not be read here.
bool Materialize(uint64_t b, uint64_t e, bool demote = false) {
	auto&                                   s = S();
	std::vector<std::pair<uint64_t, Chunk>> todo;
	std::pair<uint64_t, uint64_t>           mine {};
	{
		// One materialization at a time over any range: a second one (another fault, an emulator read) waits for the
		// first to end and then finds its chunks materialized - it never writes older bytes over what the game wrote
		// after the first. (No lock is held across the read-back itself.)
		std::unique_lock lock(s.m);
		const auto       busy = [&] {
            for (const auto& [mb, me]: s.materializing)
                if (mb < e && b < me) return true;
            return false;
		};
		s.materializing_cv.wait(lock, [&] { return !busy(); });
		auto it = s.chunks.upper_bound(b);
		if (it != s.chunks.begin()) --it;
		uint64_t lo = UINT64_MAX, hi = 0;
		for (; it != s.chunks.end() && it->first < e; ++it)
			if (it->second.end > b && !it->second.materialized && (!demote || it->second.dev)) {
				todo.emplace_back(it->first, it->second);
				lo = std::min(lo, it->first);
				hi = std::max(hi, it->second.end);
			}
		if (todo.empty()) return true;
		// (The whole chunks' span: a materialization of a neighbouring range touching them waits too.)
		mine = {std::min(lo, b), std::max(hi, e)};
		s.materializing.push_back(mine);
	}
	struct Done {
		State&                         s;
		std::pair<uint64_t, uint64_t>  range;
		~Done() {
			{
				const std::lock_guard lock(s.m);
				auto it = std::find(s.materializing.begin(), s.materializing.end(), range);
				if (it != s.materializing.end()) s.materializing.erase(it);
			}
			s.materializing_cv.notify_all();
		}
	} done {s, mine};
	bool all = true;
	for (const auto& [cb, c]: todo) {
		const uint64_t n = c.end - cb;
		std::vector<uint8_t> bytes;
		const uint8_t*       data = nullptr;
		if (demote && KeepGuest()) {
			// (Level 1 demotion: guest memory already holds these bytes; no read-back, no write.)
		} else if (c.host) {
			data = c.host->data() + c.offset;
		} else {
			bytes.resize(n); // (zero chunks: zeros)
			if (c.dev && (s.host.read_device == nullptr || !s.host.read_device(c.dev->handle, c.offset, bytes.data(), n))) {
				g_c.fallback_refused.fetch_add(1, std::memory_order_relaxed);
				Refused("a guest read/touch", cb, n);
				all = false;
				continue;
			}
			data = bytes.data();
		}
		// (Level 1 demotion: guest memory already holds these bytes.)
		if (!(demote && KeepGuest())) {
			t_inside++;
			if (s.host.write_guest != nullptr) s.host.write_guest(cb, data, n);
			t_inside--;
		}
		std::unique_lock lock(s.m);
		auto             it = s.chunks.find(cb);
		// (Only if it is still the same chunk: a newer load replaced it meanwhile - its own pages stay watched.)
		if (it == s.chunks.end() || it->second.end != c.end || it->second.dev != c.dev || it->second.host != c.host)
			continue;
		it->second.materialized = true;
		if (demote) {
			// KYTY_VRAM_BUDGET: guest memory holds the bytes now; the device buffer goes when no other range shares it.
			// The texture cache's image stays the only video memory copy; a re-created image is uploaded from guest.
			it->second.dev.reset();
			it->second.offset = 0;
			g_c.demoted.fetch_add(1, std::memory_order_relaxed);
			g_c.demoted_bytes.fetch_add(n, std::memory_order_relaxed);
		} else {
			g_c.fallbacks.fetch_add(1, std::memory_order_relaxed);
			g_c.fallback_bytes.fetch_add(n, std::memory_order_relaxed);
		}
		std::vector<Op> ops;
		RewatchLocked(s, cb, c.end, ops);
		ApplyOps(s, lock, ops);
	}
	return all;
}

double MiB(uint64_t v) { return static_cast<double>(v) / (1024.0 * 1024.0); }

} // namespace

int Level() noexcept {
	const int forced = g_level_override.load(std::memory_order_relaxed);
	if (forced >= 0) return forced;
	static const int level = [] {
		const char* v = std::getenv("KYTY_NATIVE_TEXTURE_PC");
		return v != nullptr && (v[0] == '1' || v[0] == '2') ? v[0] - '0' : 0;
	}();
	return level;
}
bool Requested() noexcept { return Level() != 0; }
namespace {
bool KeepGuest() noexcept { return Level() != 2; }
} // namespace
void SetLevelForTests(int level) noexcept { g_level_override.store(level, std::memory_order_relaxed); }

bool Enabled() noexcept { return g_enabled.load(std::memory_order_acquire); }
void SetEnabled(bool on) noexcept { g_enabled.store(on, std::memory_order_release); }

void SetHost(const Host& host) noexcept {
	auto&                 s = S();
	const std::lock_guard lock(s.m);
	// (Two setters: the loader's guest-side members, the texture cache's device-side ones; null members are kept.)
	if (host.staged != nullptr) s.host.staged = host.staged;
	if (host.write_guest != nullptr) s.host.write_guest = host.write_guest;
	if (host.watch != nullptr) s.host.watch = host.watch;
	if (host.upload != nullptr) s.host.upload = host.upload;
	if (host.upload_commit != nullptr) s.host.upload_commit = host.upload_commit;
	if (host.release != nullptr) s.host.release = host.release;
	if (host.read_device != nullptr) s.host.read_device = host.read_device;
	if (host.page_size != 0 && host.page_size != 4096) s.host.page_size = host.page_size;
	if (s.host.page_size == 0) s.host.page_size = 4096;
}

void HostReady() noexcept {
	auto&            s = S();
	std::unique_lock lock(s.m);
	std::vector<Op>  ops;
	if (!s.chunks.empty()) RewatchLocked(s, s.chunks.begin()->first, std::prev(s.chunks.end())->second.end, ops);
	ApplyOps(s, lock, ops);
}

void OnTexture(const Texture& texture) noexcept {
	if (!Enabled() || texture.owner == 0 || texture.base == 0) return;
	auto& s = S();
	{
		const std::lock_guard lock(s.m);
		s.owners[texture.owner].texture = texture;
		s.bases[texture.base]           = texture.owner;
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
	Bytes bytes = NewBytes(size);
	if (!bytes) {
		g_c.declined.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	std::memcpy(bytes->data(), src, size); // (the one CPU copy: resource heap block -> host block)
	auto& s = S();
	Host  host;
	{
		std::unique_lock lock(s.m);
		if (owner != 0 && owner_base != 0 && s.bases.find(owner_base) == s.bases.end()) s.bases[owner_base] = owner;
		Chunk c;
		c.end   = dst + size;
		c.host  = std::move(bytes);
		c.owner = owner;
		s.pending.push_back({c.host, dst, dst + size});
		InsertLocked(s, dst, std::move(c));
		host = s.host;
		std::vector<Op> ops;
		RewatchLocked(s, dst, dst + size, ops);
		ApplyOps(s, lock, ops);
	}
	if (KeepGuest()) {
		// The guest's copy, written once (the host's write_guest tells the trackers); our own invalidation ignored.
		t_inside++;
		if (host.write_guest != nullptr) host.write_guest(dst, src, size);
		t_inside--;
	} else {
		WriteEdges(s, dst, dst + size);
	}
	(mip ? g_c.mips : g_c.loads).fetch_add(1, std::memory_order_relaxed);
	(mip ? g_c.mip_bytes : g_c.load_bytes).fetch_add(size, std::memory_order_relaxed);
	t_inside++; // (the images' invalidation is ours: OnGuestWrite must not drop what was just staged)
	if (host.staged != nullptr) host.staged(dst, size); // (after the bytes are here: an upload racing it re-runs)
	t_inside--;
	return true;
}

void OnDrop(uint64_t addr, uint64_t size) noexcept {
	if (!Enabled() || size == 0 || g_any.load(std::memory_order_relaxed) == 0) return;
	auto&            s = S();
	std::unique_lock lock(s.m);
	g_c.dropped_bytes.fetch_add(CutLocked(s, addr, addr + size), std::memory_order_relaxed);
	Recount(s);
	std::vector<Op> ops;
	RewatchLocked(s, addr, addr + size, ops);
	ApplyOps(s, lock, ops);
}

bool OnSliceCopy(uint64_t dst, uint64_t src, uint64_t size,
                 const std::function<bool(uint64_t, void*, uint64_t)>& read_guest) noexcept {
	if (!Enabled() || size == 0) return false;
	auto&             s = S();
	std::vector<Part> parts;
	uint64_t          dst_owner = 0;
	{
		std::unique_lock lock(s.m);
		parts           = PartsLocked(s, src, src + size);
		const bool ours = !KeepGuest() && std::any_of(parts.begin(), parts.end(), [](const Part& p) { return p.kind != Part::Guest; });
		if (!ours) {
			// Not engine texture memory this path holds: guest memory is authoritative for both (the caller copies).
			(void)CutLocked(s, dst, dst + size);
			Recount(s);
			g_c.declined.fetch_add(1, std::memory_order_relaxed);
			std::vector<Op> ops;
			RewatchLocked(s, dst, dst + size, ops);
			ApplyOps(s, lock, ops);
			return false;
		}
		auto b = s.bases.upper_bound(dst);
		if (b != s.bases.begin()) dst_owner = std::prev(b)->second;
	}
	// The destination shares each part of the source (device buffer / host block / zeros); guest parts are read now.
	std::vector<std::pair<uint64_t, Chunk>> made;
	uint64_t                                guest_reads = 0;
	for (const auto& p: parts) {
		Chunk c;
		c.end   = dst + (p.end - src);
		c.owner = dst_owner;
		if (p.kind == Part::Device) {
			c.dev    = std::static_pointer_cast<Dev>(std::const_pointer_cast<void>(p.keep));
			c.offset = p.offset;
		} else if (p.kind == Part::Host) {
			c.host   = std::static_pointer_cast<std::vector<uint8_t>>(std::const_pointer_cast<void>(p.keep));
			c.offset = static_cast<uint64_t>(p.host - c.host->data());
		} else if (p.kind == Part::Zero) {
			c.zero = true;
		} else {
			Bytes bytes = NewBytes(p.end - p.begin);
			if (!bytes) return false;
			guest_reads++;
			t_inside++;
			if (!read_guest || !read_guest(p.begin, bytes->data(), p.end - p.begin)) std::memset(bytes->data(), 0, bytes->size());
			t_inside--;
			c.host = std::move(bytes);
		}
		made.emplace_back(dst + (p.begin - src), std::move(c));
	}
	Host host;
	{
		std::unique_lock lock(s.m);
		for (auto& [b, c]: made) {
			if (c.host) s.pending.push_back({c.host, b, c.end});
			InsertLocked(s, b, std::move(c));
		}
		host = s.host;
		std::vector<Op> ops;
		RewatchLocked(s, dst, dst + size, ops);
		ApplyOps(s, lock, ops);
	}
	WriteEdges(s, dst, dst + size);
	g_c.slice_copies.fetch_add(1, std::memory_order_relaxed);
	g_c.slice_bytes.fetch_add(size, std::memory_order_relaxed);
	g_c.slice_guest_reads.fetch_add(guest_reads, std::memory_order_relaxed);
	t_inside++;
	if (host.staged != nullptr) host.staged(dst, size);
	t_inside--;
	return true;
}

void OnRelease(uint64_t owner, uint64_t base) noexcept {
	if (!Enabled() || owner == 0) return;
	auto&            s = S();
	std::unique_lock lock(s.m);
	std::vector<Op>  ops;
	auto             it = s.owners.find(owner);
	if (it != s.owners.end()) {
		if (it->second.lo < it->second.hi) {
			g_c.dropped_bytes.fetch_add(CutLocked(s, it->second.lo, it->second.hi, owner), std::memory_order_relaxed);
			RewatchLocked(s, it->second.lo, it->second.hi, ops);
		}
		s.owners.erase(it);
		g_c.released.fetch_add(1, std::memory_order_relaxed);
	}
	if (auto b = s.bases.find(base); b != s.bases.end() && b->second == owner) s.bases.erase(b);
	if (auto x = s.extents.find(base); x != s.extents.end() && x->second.owner == owner) s.extents.erase(x);
	Recount(s);
	ApplyOps(s, lock, ops);
}

bool Overlaps(uint64_t addr, uint64_t size) noexcept {
	if (!Enabled() || size == 0 || g_any.load(std::memory_order_relaxed) == 0) return false;
	auto&                 s = S();
	const std::lock_guard lock(s.m);
	const auto            parts = PartsLocked(s, addr, addr + size);
	return std::any_of(parts.begin(), parts.end(), [](const Part& p) { return p.kind != Part::Guest; });
}

void Flush() noexcept {
	if (!Enabled()) return;
	static std::mutex     flush_m; // (the host's recorded uploads are one list: one Flush at a time)
	const std::lock_guard flush_lock(flush_m);
	auto&                     s = S();
	std::vector<State::Pending> pending;
	Host                      host;
	{
		const std::lock_guard lock(s.m);
		host = s.host;
		if (host.upload != nullptr) pending.swap(s.pending);
	}
	// One upload per host block (a block shared by a slice copy is listed once per range). The uploads are recorded and
	// submitted together (host.upload_commit: one submit and one wait per batch of up to kBatch bytes), and a batch's
	// buffers are published to the chunks only after its wait: a read-back never sees a buffer not yet written.
	constexpr uint64_t                        kBatch = 64ull << 20;
	std::unordered_map<const void*, DevRef>   done;
	std::vector<std::pair<Bytes, uint64_t>>   batch; // block, handle (recorded, not yet committed)
	std::vector<const State::Pending*>        users; // the pending ranges of this batch
	uint64_t                                  batch_bytes = 0;
	const auto commit = [&] {
		if (users.empty()) return;
		const bool ok = batch.empty() || host.upload_commit == nullptr || host.upload_commit();
		if (!ok) {
			g_c.upload_fail.fetch_add(batch.size(), std::memory_order_relaxed);
			for (const auto& [block, handle]: batch) {
				done.erase(block.get());
				if (host.release != nullptr) host.release(handle);
			}
		} else {
			for (const auto& [block, handle]: batch) {
				done[block.get()] = std::make_shared<Dev>(handle, block->size());
				g_c.to_vram.fetch_add(1, std::memory_order_relaxed);
				g_c.to_vram_bytes.fetch_add(block->size(), std::memory_order_relaxed);
			}
		}
		const std::lock_guard lock(s.m);
		for (const auto* p: users) {
			Bytes block = p->block.lock();
			if (!block) continue;
			const auto d = done.find(block.get());
			if (d == done.end() || !d->second) {
				s.pending.push_back(*p); // (again next time)
				continue;
			}
			auto it = s.chunks.upper_bound(p->b);
			if (it != s.chunks.begin()) --it;
			for (; it != s.chunks.end() && it->first < p->e; ++it) {
				if (it->second.host != block) continue;
				it->second.dev = d->second; // (same layout: offsets stay)
				it->second.host.reset();
			}
		}
		batch.clear();
		users.clear();
		batch_bytes = 0;
	};
	for (auto& p: pending) {
		Bytes block = p.block.lock();
		if (!block) continue; // (cut or replaced meanwhile: nothing to move)
		if (done.find(block.get()) == done.end()) {
			if (batch_bytes != 0 && batch_bytes + block->size() > kBatch) commit();
			const uint64_t handle = host.upload(block->data(), block->size());
			if (handle == 0) {
				g_c.upload_fail.fetch_add(1, std::memory_order_relaxed);
				const std::lock_guard lock(s.m);
				s.pending.push_back(p); // (again next time)
				continue;
			}
			done[block.get()] = nullptr; // (recorded; set when its batch is committed)
			batch.emplace_back(block, handle);
			batch_bytes += block->size();
		}
		users.push_back(&p);
	}
	commit();
	pending.clear();
	done.clear(); // (host blocks are freed as their last chunk lets go; device buffers nobody took are released below)
	std::vector<uint64_t> gone;
	{
		const std::lock_guard lock(g_release_m);
		gone.swap(g_release);
	}
	if (host.release != nullptr)
		for (const auto h: gone) host.release(h);
}

std::vector<Part> Plan(uint64_t addr, uint64_t size) noexcept {
	auto&                 s = S();
	const std::lock_guard lock(s.m);
	const uint64_t        frame = g_c.frames.load(std::memory_order_relaxed) + 1;
	auto                  it    = s.chunks.upper_bound(addr);
	if (it != s.chunks.begin()) --it;
	for (; it != s.chunks.end() && it->first < addr + size; ++it)
		if (it->second.end > addr) it->second.used = frame;
	return PartsLocked(s, addr, addr + size);
}

void NoteUpload(uint64_t bytes, uint64_t ns, bool gathered) noexcept {
	g_c.fills.fetch_add(1, std::memory_order_relaxed);
	g_c.fill_bytes.fetch_add(bytes, std::memory_order_relaxed);
	g_c.fill_ns.fetch_add(ns, std::memory_order_relaxed);
	if (gathered) g_c.gathered.fetch_add(1, std::memory_order_relaxed);
}

bool Fill(uint64_t addr, uint64_t size, void* out, const std::function<bool(uint64_t, void*, uint64_t)>& read_guest) noexcept {
	const auto start = std::chrono::steady_clock::now();
	auto&      s     = S();
	const auto parts = Plan(addr, size);
	const bool ok    = Assemble(s, parts, addr, static_cast<uint8_t*>(out), read_guest, nullptr);
	NoteUpload(size,
	           static_cast<uint64_t>(
	               std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()),
	           false);
	return ok;
}

void BeforeGuestRead(uint64_t addr, uint64_t size) noexcept {
	if (t_inside != 0 || !Enabled() || KeepGuest() || size == 0 || g_any.load(std::memory_order_relaxed) == 0) return;
	(void)Materialize(addr, addr + size); // (a refusal stops the process: Refused)
}

void OnGuestWrite(uint64_t addr, uint64_t size) noexcept {
	if (t_inside != 0 || !Enabled() || size == 0 || g_any.load(std::memory_order_relaxed) == 0) return;
	auto& s = S();
	if (!KeepGuest()) (void)Materialize(addr, addr + size); // (the parts the write leaves alone stay right)
	std::unique_lock lock(s.m);
	const uint64_t   cut = CutLocked(s, addr, addr + size);
	if (cut != 0) {
		g_c.guest_writes.fetch_add(1, std::memory_order_relaxed);
		g_c.guest_write_bytes.fetch_add(cut, std::memory_order_relaxed);
	}
	uint64_t xend = 0;
	if (cut != 0 || InExtentLocked(s, addr, &xend)) {
		// The range is the guest's now (written): a guest-marked chunk, so an engine texture's memory there is read from
		// the guest instead of taken as undefined (zeros).
		Chunk g;
		g.end          = addr + size;
		g.materialized = true;
		auto b         = s.bases.upper_bound(addr);
		if (b != s.bases.begin()) g.owner = std::prev(b)->second;
		InsertLocked(s, addr, std::move(g));
	}
	Recount(s);
	std::vector<Op> ops;
	RewatchLocked(s, addr, addr + size, ops);
	ApplyOps(s, lock, ops);
}

bool OnFault(uint64_t address) noexcept {
	if (t_inside != 0 || !Enabled() || KeepGuest() || g_any.load(std::memory_order_relaxed) == 0) return false;
	auto&    s = S();
	uint64_t page = 0, page_end = 0;
	{
		const std::lock_guard lock(s.m);
		page     = PD(s, address);
		page_end = page + s.host.page_size;
		auto w   = s.watched.upper_bound(page);
		if (w == s.watched.begin()) return false;
		--w;
		if (page >= w->second) return false; // not a page we watch: not our fault
	}
	g_c.faults.fetch_add(1, std::memory_order_relaxed);
	return Materialize(page, page_end);
}

// ---- KYTY_VRAM_BUDGET -------------------------------------------------------------------------------------------------

std::atomic<int> g_budget_default {0};

int VramBudgetMode() noexcept {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_VRAM_BUDGET");
		if (v == nullptr || *v == 0) return -1;
		if (std::strcmp(v, "auto") == 0 || std::strcmp(v, "1") == 0) return 1;
		return 0;
	}();
	return mode;
}
void SetVramBudgetDefault(bool on) noexcept { g_budget_default.store(on ? 1 : 0, std::memory_order_relaxed); }
bool VramBudgetOn() noexcept {
	const int m = VramBudgetMode();
	return m >= 0 ? m == 1 : g_budget_default.load(std::memory_order_relaxed) == 1;
}

namespace {
uint64_t EnvPercent(const char* name, uint64_t fallback, uint64_t lo, uint64_t hi) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == 0) return fallback;
	return std::clamp<uint64_t>(std::strtoull(v, nullptr, 10), lo, hi);
}
struct BudgetLog {
	uint64_t usage = 0, budget = 0, target = 0, passes = 0;
};
BudgetLog g_budget_log;

// Once a second: over KYTY_VRAM_BUDGET_HIGH_PCT (default 90) of the device-local budget (the OS's: DXGI on Windows),
// the least recently used engine textures' device buffers (no image upload for at least 2 s) are demoted: their bytes
// to guest memory (as level 1 keeps them), their buffers freed, until the estimate is at KYTY_VRAM_BUDGET_LOW_PCT
// (default 82); at most 256 MiB a pass (one wait per buffer read back on this path's own queue).
void BudgetPass() {
	auto& s = S();
	Host  host;
	{
		const std::lock_guard lock(s.m);
		host = s.host;
	}
	uint64_t usage = 0, budget = 0;
	if (host.vram == nullptr || !host.vram(&usage, &budget) || budget == 0) return;
	static const uint64_t high = EnvPercent("KYTY_VRAM_BUDGET_HIGH_PCT", 90, 50, 98);
	static const uint64_t low  = std::min(high, EnvPercent("KYTY_VRAM_BUDGET_LOW_PCT", 82, 40, 98));
	g_budget_log.usage         = usage;
	g_budget_log.budget        = budget;
	g_budget_log.target        = budget / 100 * low;
	if (usage <= budget / 100 * high) return;
	const uint64_t want  = std::min<uint64_t>(usage - budget / 100 * low, 256ull << 20);
	const uint64_t frame = g_c.frames.load(std::memory_order_relaxed);
	struct Cand {
		uint64_t b, e, used;
		DevRef   dev; // (held: alive while this pass looks at it)
	};
	std::vector<Cand> cand;
	{
		const std::lock_guard lock(s.m);
		for (const auto& [b, c]: s.chunks)
			if (c.dev && !c.materialized && c.used + 120 <= frame) cand.push_back({b, c.end, c.used, c.dev});
	}
	std::stable_sort(cand.begin(), cand.end(), [](const Cand& x, const Cand& y) { return x.used < y.used; });
	std::unordered_set<Dev*> counted;
	uint64_t                 freed = 0;
	std::vector<Cand>        pick;
	for (const auto& c: cand) {
		if (freed >= want) break;
		if (counted.insert(c.dev.get()).second) freed += c.dev->size; // (an estimate: a buffer a slice still shares stays)
		pick.push_back(c);
	}
	// (Every range sharing a picked buffer, so that the buffer is really freed.)
	for (const auto& c: cand)
		if (counted.count(c.dev.get()) != 0 && std::none_of(pick.begin(), pick.end(), [&](const Cand& p) { return p.b == c.b; }))
			pick.push_back(c);
	for (const auto& c: pick) (void)Materialize(c.b, c.e, true);
	g_c.demote_passes.fetch_add(1, std::memory_order_relaxed);
	g_budget_log.passes++;
}
} // namespace

void Pump() noexcept {
	if (!Enabled()) return;
	Flush();
	g_c.frames.fetch_add(1, std::memory_order_relaxed);
	if (VramBudgetOn() && g_c.frames.load(std::memory_order_relaxed) % 60 == 0) BudgetPass();
	using Clock = std::chrono::steady_clock;
	static Clock::time_point last = Clock::now();
	static Stats             prev {};
	const auto               now = Clock::now();
	if (now - last < std::chrono::seconds(30)) return;
	last                  = now;
	const Stats    c      = GetStats();
	const uint64_t frames = c.frames - prev.frames;
	std::printf("NativeTexturePC (level %d, %s): textures created natively %llu (loaded %llu, engine-made %llu, destroyed %llu) | file loads "
	            "%llu (%.1f MiB) | mips streamed natively %llu (%.1f MiB) | slice copies shared %llu (%.1f MiB, %llu guest "
	            "reads) | streamed out/freed %.1f MiB | host MiB held %.1f (%llu blocks) | VRAM MiB %.1f (%llu buffers; to VRAM "
	            "%llu, %.1f MiB, failed %llu) | guest pages no-access %.1f MiB | image uploads from VRAM %llu (%.1f MiB, "
	            "%llu gathered; %.2f ms/frame over %llu frames) | CPU fallbacks %llu (%.1f MiB; faults %llu, refused %llu) | "
	            "declined %llu | CPU/host writes over staged bytes %llu (%.1f MiB dropped)\n",
	            Level(), KeepGuest() ? "VRAM + guest copy" : "VRAM only, guest no-access", static_cast<unsigned long long>(c.textures), static_cast<unsigned long long>(c.textures_loaded),
	            static_cast<unsigned long long>(c.textures_created), static_cast<unsigned long long>(c.released),
	            static_cast<unsigned long long>(c.loads), MiB(c.load_bytes), static_cast<unsigned long long>(c.mips),
	            MiB(c.mip_bytes), static_cast<unsigned long long>(c.slice_copies), MiB(c.slice_bytes),
	            static_cast<unsigned long long>(c.slice_guest_reads), MiB(c.dropped_bytes), MiB(c.host_bytes),
	            static_cast<unsigned long long>(c.host_blocks), MiB(c.vram_bytes), static_cast<unsigned long long>(c.vram_buffers),
	            static_cast<unsigned long long>(c.to_vram), MiB(c.to_vram_bytes), static_cast<unsigned long long>(c.upload_fail),
	            MiB(c.watched_bytes), static_cast<unsigned long long>(c.fills), MiB(c.fill_bytes),
	            static_cast<unsigned long long>(c.gathered),
	            frames != 0 ? static_cast<double>(c.fill_ns - prev.fill_ns) / 1e6 / static_cast<double>(frames) : 0.0,
	            static_cast<unsigned long long>(frames), static_cast<unsigned long long>(c.fallbacks), MiB(c.fallback_bytes),
	            static_cast<unsigned long long>(c.faults), static_cast<unsigned long long>(c.fallback_refused),
	            static_cast<unsigned long long>(c.declined), static_cast<unsigned long long>(c.guest_writes),
	            MiB(c.guest_write_bytes));
	if (VramBudgetOn())
		std::printf("NativeTexturePC VRAM budget (KYTY_VRAM_BUDGET auto): device-local %.0f MiB in use of a %.0f MiB budget "
		            "(target %.0f) | engine texture buffers %.1f MiB | demoted to guest memory %llu ranges (%.1f MiB; %llu "
		            "passes)\n",
		            MiB(g_budget_log.usage), MiB(g_budget_log.budget), MiB(g_budget_log.target), MiB(c.vram_bytes),
		            static_cast<unsigned long long>(g_c.demoted.load()), MiB(g_c.demoted_bytes.load()),
		            static_cast<unsigned long long>(g_c.demote_passes.load()));
	std::fflush(stdout);
	prev = c;
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
	r.host_bytes        = g_c.host_bytes.load();
	r.host_blocks       = g_c.host_blocks.load();
	r.vram_bytes        = g_c.vram_bytes.load();
	r.vram_buffers      = g_c.vram_buffers.load();
	r.to_vram           = g_c.to_vram.load();
	r.to_vram_bytes     = g_c.to_vram_bytes.load();
	r.upload_fail       = g_c.upload_fail.load();
	r.fills             = g_c.fills.load();
	r.fill_bytes        = g_c.fill_bytes.load();
	r.fill_ns           = g_c.fill_ns.load();
	r.gathered          = g_c.gathered.load();
	r.frames            = g_c.frames.load();
	r.fallbacks         = g_c.fallbacks.load();
	r.fallback_bytes    = g_c.fallback_bytes.load();
	r.faults            = g_c.faults.load();
	r.fallback_refused  = g_c.fallback_refused.load();
	r.declined          = g_c.declined.load();
	r.watched_bytes     = g_c.watched_bytes.load();
	r.guest_writes      = g_c.guest_writes.load();
	r.guest_write_bytes = g_c.guest_write_bytes.load();
	return r;
}

void ResetForTests() noexcept {
	auto& s = S();
	{
		const std::lock_guard lock(s.m);
		s.chunks.clear();
		s.extents.clear();
		s.bases.clear();
		s.owners.clear();
		s.watched.clear();
		s.pending.clear();
		s.host = {};
		Recount(s);
	}
	{
		const std::lock_guard lock(g_release_m);
		g_release.clear();
	}
	for (auto* a: {&g_c.textures, &g_c.textures_loaded, &g_c.textures_created, &g_c.released, &g_c.loads, &g_c.load_bytes,
	               &g_c.mips, &g_c.mip_bytes, &g_c.slice_copies, &g_c.slice_bytes, &g_c.slice_guest_reads,
	               &g_c.dropped_bytes, &g_c.to_vram, &g_c.to_vram_bytes, &g_c.upload_fail, &g_c.fills, &g_c.fill_bytes,
	               &g_c.fill_ns, &g_c.gathered, &g_c.frames, &g_c.fallbacks, &g_c.fallback_bytes, &g_c.faults,
	               &g_c.fallback_refused, &g_c.declined, &g_c.watched_bytes, &g_c.guest_writes,
	               &g_c.guest_write_bytes, &g_c.demoted, &g_c.demoted_bytes, &g_c.demote_passes})
		a->store(0);
}

} // namespace Libs::Graphics::NativeTexturePc
