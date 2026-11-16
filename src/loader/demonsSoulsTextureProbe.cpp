// Texture streaming probe (KYTY_TEXTURE_PROBE, on by default; =0 off). Measurement only: no guest byte, no call result
// and no memory operation is changed.
//
// Demon's Souls 01.007.000 (static notes port-texturas/NOTAS.md): CTexture::SetCurrentStreamingChunk(int) at
// eboot+0x901330 commits the next mip chunk of a streamed texture and copies its bytes there with
//   memcpy(tex+0x98 (base) + chunk.offset, chunkData, chunk.size)     call at eboot+0x901459, returns to 0x90145e
// The call goes through the eboot's PLT stub (01.007.000: eboot+0x1e73d00) to the guest libc's memcpy, whose body
// DemonsSoulsCopy replaces with CoherentMove (or to Kyty's LibcInternal::memcpy): both call OnCopy with their return
// address. Here: the bytes and copies of that one call site, and its destination ranges kept as live (overlaps merged)
// until KernelMunmap unmaps them (the engine's decommit: BatchMap unmap, op 1). The texture's first chunk (the
// eboot+0x813130 read) and single-chunk textures do not go through this call: they are not counted.
//
// Checked at start-up: the bytes at eboot+0x901459 must be a `call rel32`, or the probe stays off and says why. The
// call's target (stub, GOT slot, module) and a `lea reg, [rip+disp32]` to a string naming SetCurrentStreamingChunk in
// the function are printed only.
//
// KYTY_TEXTURE_RELEASE (off by default; =1 on: tried, it freed no real RAM): each copied chunk range, once ReleaseDelayVblanks
// vblanks (~2 s, and at least ReleaseMinMs) have passed - by then the TextureCache has uploaded it to the GPU -, is taken
// out of the process's working set by a thread of its own (Kyty.TexRelease): VirtualUnlock on pages not locked, as
// KYTY_OFFLOAD_PAGEOUT and the parked-range trim do. Windows keeps the contents (standby list / page file) and faults
// them back in on any later read (a re-upload, a CPU read): no byte is lost, no protection changes, nothing is
// discarded. 1 MiB per call, through LocalPlatform::RunTrimQuiet (no guest map, unmap or protection change meanwhile:
// the bug check 0x1A), at most ~200 MiB/s. Only the parts still live (not unmapped by the game) are released.
#include "loader/demonsSoulsTextureProbe.h"

#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "live-counters.h"
#include "local-platform.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Loader::DemonsSoulsTextureProbe {
namespace {

constexpr uint64_t FunctionStart = 0x901330; // CTexture::SetCurrentStreamingChunk (01.007.000)
constexpr uint64_t CallSite      = 0x901459; // its memcpy of the chunk's bytes
constexpr uint64_t AuditedStub   = 0x1e73d00;
// KYTY_BUFFER_PROBE: CBuffer::Create (01.007.000) and its memcpy(buffer memory, initial data, elementSize * count)
constexpr uint64_t BufferFunctionStart = 0x909460;
constexpr uint64_t BufferCallSite      = 0x90954b;
constexpr uint64_t ReportSeconds = 5;
constexpr double   MiB           = 1024.0 * 1024.0;
// KYTY_TEXTURE_RELEASE
constexpr uint64_t ReleaseDelayVblanks = 120;
constexpr int64_t  ReleaseMinMs        = 1000; // (and at most ReleaseMaxMs when vblanks are not counted)
constexpr int64_t  ReleaseMaxMs        = 4000;
constexpr uint64_t ReleaseChunk        = uint64_t {1} << 20u;
constexpr uint64_t PageMask            = 0xfffu;

struct Pending {
	uint64_t                              begin = 0, end = 0, vblank = 0;
	std::chrono::steady_clock::time_point at;
};

// Never destroyed: the reporter thread is detached and may still run while the process exits.
struct State {
	std::mutex                   mutex;
	std::map<uint64_t, uint64_t> live; // begin -> end; disjoint, overlapping or touching ranges merged
	uint64_t                     live_bytes   = 0;
	uint64_t                     peak_bytes   = 0;
	uint64_t                     window_bytes = 0, window_copies = 0, window_unmapped = 0;
	uint64_t                     total_bytes  = 0, total_copies = 0;
	// KYTY_TEXTURE_RELEASE (under `mutex` too)
	std::deque<Pending>          pending;
	uint64_t                     released_window = 0, released_total = 0, released_ranges = 0;
	bool                         release_reported = false;
};
State& GetState() {
	static auto* state = new State();
	return *state;
}
// KYTY_BUFFER_PROBE's live CBuffer ranges (only live/live_bytes/peak_bytes/total_* and mutex are used).
State& GetBufferState() {
	static auto* state = new State();
	return *state;
}
bool              g_installed = false;
std::atomic<bool> g_release {false};

int32_t Rel32(const uint8_t* at) {
	int32_t value = 0;
	std::memcpy(&value, at, sizeof(value));
	return value;
}

void AddLive(State& s, uint64_t begin, uint64_t end) {
	auto it = s.live.upper_bound(begin);
	if (it != s.live.begin()) {
		auto previous = std::prev(it);
		if (previous->second >= begin) {
			begin = previous->first;
			if (previous->second > end) end = previous->second;
			s.live_bytes -= previous->second - previous->first;
			it = s.live.erase(previous);
		}
	}
	while (it != s.live.end() && it->first <= end) {
		if (it->second > end) end = it->second;
		s.live_bytes -= it->second - it->first;
		it = s.live.erase(it);
	}
	s.live.emplace(begin, end);
	s.live_bytes += end - begin;
	if (s.live_bytes > s.peak_bytes) s.peak_bytes = s.live_bytes;
}

uint64_t RemoveLive(State& s, uint64_t begin, uint64_t end) {
	uint64_t removed = 0;
	auto     it      = s.live.upper_bound(begin);
	if (it != s.live.begin()) {
		auto previous = std::prev(it);
		if (previous->second > begin) it = previous;
	}
	while (it != s.live.end() && it->first < end) {
		const uint64_t range_begin = it->first, range_end = it->second;
		s.live_bytes -= range_end - range_begin;
		it = s.live.erase(it);
		const uint64_t cut_begin = range_begin > begin ? range_begin : begin;
		const uint64_t cut_end   = range_end < end ? range_end : end;
		removed += cut_end - cut_begin;
		if (range_begin < begin) {
			s.live.emplace(range_begin, begin);
			s.live_bytes += begin - range_begin;
		}
		if (range_end > end) {
			s.live.emplace(end, range_end);
			s.live_bytes += range_end - end;
		}
	}
	return removed;
}

uint64_t UploadBytes() {
	return LiveCounters::Value(LiveCounters::FullUploadBytes) + LiveCounters::Value(LiveCounters::PartialUploadBytes);
}

void Reporter() {
	auto     last    = std::chrono::steady_clock::now();
	uint64_t uploads = UploadBytes();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(ReportSeconds));
		const auto   now     = std::chrono::steady_clock::now();
		const double seconds = std::chrono::duration<double>(now - last).count();
		last                 = now;
		const uint64_t uploads_now = UploadBytes();
		const uint64_t upload_window = uploads_now - uploads;
		uploads = uploads_now;
		uint64_t bytes = 0, copies = 0, unmapped = 0, live = 0, peak = 0, ranges = 0, total = 0, total_copies = 0;
		{
			auto&                       s = GetState();
			std::lock_guard<std::mutex> lock(s.mutex);
			bytes        = s.window_bytes;
			copies       = s.window_copies;
			unmapped     = s.window_unmapped;
			live         = s.live_bytes;
			peak         = s.peak_bytes;
			ranges       = s.live.size();
			total        = s.total_bytes;
			total_copies = s.total_copies;
			s.window_bytes = s.window_copies = s.window_unmapped = 0;
		}
		uint64_t released = 0, released_total = 0, released_ranges = 0;
		bool     release_line = false;
		if (g_release.load(std::memory_order_relaxed)) {
			auto&                       s = GetState();
			std::lock_guard<std::mutex> lock(s.mutex);
			released          = s.released_window;
			released_total    = s.released_total;
			released_ranges   = s.released_ranges;
			s.released_window = 0;
			release_line      = s.release_reported = s.release_reported || released_total != 0;
		}
		if (release_line) {
			std::printf("Texture release (5 s): %.1f MiB released, %.1f MiB total, %llu ranges\n",
			            static_cast<double>(released) / MiB, static_cast<double>(released_total) / MiB,
			            static_cast<unsigned long long>(released_ranges));
			std::fflush(stdout);
		}
		if (total_copies == 0) continue; // nothing streamed yet (start-up, menus): no line
		std::printf("Texture probe (5 s): %llu copies, %.1f MiB copied (%.1f MiB/s), live texture ranges %.1f MiB "
		            "(%llu ranges, peak %.1f MiB, %.1f MiB unmapped in the window, %.1f MiB copied since start in %llu "
		            "copies); TextureCache uploads (all images) %.1f MiB\n",
		            static_cast<unsigned long long>(copies), static_cast<double>(bytes) / MiB,
		            seconds > 0.0 ? static_cast<double>(bytes) / MiB / seconds : 0.0, static_cast<double>(live) / MiB,
		            static_cast<unsigned long long>(ranges), static_cast<double>(peak) / MiB,
		            static_cast<double>(unmapped) / MiB, static_cast<double>(total) / MiB,
		            static_cast<unsigned long long>(total_copies), static_cast<double>(upload_window) / MiB);
		std::fflush(stdout);
	}
}

// A `lea r64, [rip+disp32]` in [begin, end) whose target holds a string naming SetCurrentStreamingChunk.
bool NamesFunction(uint64_t begin, uint64_t end) {
	if (!Libs::Graphics::HostMemoryRangeIsReadable(begin, end - begin)) return false;
	const auto* code = reinterpret_cast<const uint8_t*>(begin);
	for (uint64_t i = 0; begin + i + 7 <= end; i++) {
		if ((code[i] != 0x48 && code[i] != 0x4c) || code[i + 1] != 0x8d || (code[i + 2] & 0xc7) != 0x05) continue;
		const uint64_t target = begin + i + 7 + static_cast<uint64_t>(static_cast<int64_t>(Rel32(code + i + 3)));
		char           text[128] {};
		if (!Libs::Graphics::HostMemoryRangeIsReadable(target, sizeof(text) - 1)) continue;
		std::memcpy(text, reinterpret_cast<const void*>(target), sizeof(text) - 1);
		if (std::strstr(text, "SetCurrentStreamingChunk") != nullptr) return true;
	}
	return false;
}

#if defined(_WIN32)
// One 1 MiB piece out of the working set, through the trim runner (false: the address space stayed busy; retried).
thread_local uint64_t t_guest = 0, t_backing = 0, t_size = 0;
thread_local bool     t_ran   = false;
bool ReleasePiece(uint64_t guest, uint64_t size) {
	// (Looked up before the quiet call: the lookup takes the address space's lock. A stale pointer is harmless:
	// VirtualUnlock only moves pages out of the working set, their contents stay.)
	const auto* backing = Libs::LibKernel::Memory::TryGetBackingPointer(guest, size);
	t_guest   = guest;
	t_size    = size;
	t_backing = backing != nullptr && reinterpret_cast<uint64_t>(backing) != guest ? reinterpret_cast<uint64_t>(backing)
	                                                                                 : 0;
	t_ran     = false;
	LocalPlatform::RunTrimQuiet([] {
		t_ran = true;
		// (Pages not locked: the call "fails" with ERROR_NOT_LOCKED and takes them out of the working set.)
		(void)VirtualUnlock(reinterpret_cast<void*>(t_guest), t_size);
		if (t_backing != 0) (void)VirtualUnlock(reinterpret_cast<void*>(t_backing), t_size);
	});
	return t_ran;
}

void Releaser() {
	LocalPlatform::SetThreadName("Kyty.TexRelease");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	std::vector<std::pair<uint64_t, uint64_t>> parts;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		const uint64_t vblank = LiveCounters::Value(LiveCounters::Vblanks);
		const auto     now    = std::chrono::steady_clock::now();
		for (;;) {
			parts.clear();
			{
				auto&                       s = GetState();
				std::lock_guard<std::mutex> lock(s.mutex);
				if (s.pending.empty()) break;
				const Pending& p  = s.pending.front();
				const int64_t  ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - p.at).count();
				if (ms < ReleaseMinMs || (vblank - p.vblank < ReleaseDelayVblanks && ms < ReleaseMaxMs)) break;
				// The parts of it still live (the game unmaps what it decommits).
				auto it = s.live.upper_bound(p.begin);
				if (it != s.live.begin() && std::prev(it)->second > p.begin) --it;
				for (; it != s.live.end() && it->first < p.end; ++it) {
					const uint64_t b = it->first > p.begin ? it->first : p.begin;
					const uint64_t e = it->second < p.end ? it->second : p.end;
					if (b < e) parts.emplace_back(b, e);
				}
				s.pending.pop_front();
			}
			uint64_t done = 0;
			for (const auto& [begin, end]: parts) {
				// Whole pages only: a partial page at an edge stays (it may hold other data still in use).
				const uint64_t first = (begin + PageMask) & ~PageMask;
				const uint64_t last  = end & ~PageMask;
				for (uint64_t at = first; at < last;) {
					const uint64_t size = last - at < ReleaseChunk ? last - at : ReleaseChunk;
					if (!ReleasePiece(at, size)) {
						std::this_thread::sleep_for(std::chrono::milliseconds(20));
						continue; // (busy: the same piece again)
					}
					done += size;
					at += size;
					std::this_thread::sleep_for(std::chrono::milliseconds(5)); // at most ~200 MiB/s
				}
			}
			if (done != 0) {
				auto&                       s = GetState();
				std::lock_guard<std::mutex> lock(s.mutex);
				s.released_window += done;
				s.released_total += done;
				s.released_ranges++;
			}
		}
	}
}
#endif

} // namespace

void Record(uint64_t destination, uint64_t size) {
	if (destination > UINT64_MAX - size) return;
	::Libs::Graphics::PageManager::NoteZoneSignal(0, size); // (KYTY_PROTECT_PAUSE; before the probe's lock)
	auto&                       s = GetState();
	std::lock_guard<std::mutex> lock(s.mutex);
	s.window_bytes += size;
	s.window_copies++;
	s.total_bytes += size;
	s.total_copies++;
	AddLive(s, destination, destination + size);
	if (g_release.load(std::memory_order_relaxed)) {
		Pending p;
		p.begin  = destination;
		p.end    = destination + size;
		p.vblank = LiveCounters::Value(LiveCounters::Vblanks);
		p.at     = std::chrono::steady_clock::now();
		s.pending.push_back(p);
	}
}

uint64_t LiveOverlap(uint64_t begin, uint64_t end) {
	if (g_return_address.load(std::memory_order_relaxed) == 0) return UINT64_MAX;
	auto&                        s = GetState();
	std::unique_lock<std::mutex> lock(s.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return UINT64_MAX;
	uint64_t bytes = 0;
	auto     it    = s.live.upper_bound(begin);
	if (it != s.live.begin()) --it;
	for (; it != s.live.end() && it->first < end; ++it) {
		const uint64_t b = std::max(begin, it->first), e = std::min(end, it->second);
		if (e > b) bytes += e - b;
	}
	return bytes;
}

void RecordBuffer(uint64_t destination, uint64_t size) {
	if (destination > UINT64_MAX - size) return;
	auto&                       s = GetBufferState();
	std::lock_guard<std::mutex> lock(s.mutex);
	s.total_bytes += size;
	s.total_copies++;
	AddLive(s, destination, destination + size);
}

uint64_t BufferOverlap(uint64_t begin, uint64_t end) {
	if (g_buffer_return_address.load(std::memory_order_relaxed) == 0) return UINT64_MAX;
	auto&                        s = GetBufferState();
	std::unique_lock<std::mutex> lock(s.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return UINT64_MAX;
	uint64_t bytes = 0;
	auto     it    = s.live.upper_bound(begin);
	if (it != s.live.begin()) --it;
	for (; it != s.live.end() && it->first < end; ++it) {
		const uint64_t b = std::max(begin, it->first), e = std::min(end, it->second);
		if (e > b) bytes += e - b;
	}
	return bytes;
}

void Unmapped(uint64_t vaddr, uint64_t size) {
	if (vaddr > UINT64_MAX - size) return;
	if (g_buffer_return_address.load(std::memory_order_relaxed) != 0) {
		auto&                       b = GetBufferState();
		std::lock_guard<std::mutex> lock(b.mutex);
		if (!b.live.empty()) RemoveLive(b, vaddr, vaddr + size);
	}
	if (g_return_address.load(std::memory_order_relaxed) == 0) return;
	auto&    s       = GetState();
	uint64_t removed = 0;
	{
		std::lock_guard<std::mutex> lock(s.mutex);
		if (s.live.empty()) return;
		removed = RemoveLive(s, vaddr, vaddr + size);
		s.window_unmapped += removed;
	}
	::Libs::Graphics::PageManager::NoteZoneSignal(removed, 0); // (KYTY_PROTECT_PAUSE)
}

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (g_installed || program == nullptr || program->file_name.filename() != "eboot.bin" ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame())
		return;
	g_installed = true;
	{
		// KYTY_BUFFER_PROBE: independent of the texture probe; only the call's return address is armed.
		const char* bv = std::getenv("KYTY_BUFFER_PROBE");
		const uint64_t bcall = program->base_vaddr + BufferCallSite;
		if (bv != nullptr && bv[0] == '0') {
			std::printf("Buffer probe: off (KYTY_BUFFER_PROBE=0)\n");
		} else if (program->mapped_size < BufferCallSite + 5 || !Libs::Graphics::HostMemoryRangeIsReadable(bcall, 5)) {
			std::printf("Buffer probe: OFF (eboot+0x%llx is not mapped: another build?)\n",
			            static_cast<unsigned long long>(BufferCallSite));
		} else {
			const auto*    bb    = reinterpret_cast<const uint8_t*>(bcall);
			const uint64_t bstub = bb[0] == 0xe8 ? bcall + 5 + static_cast<uint64_t>(static_cast<int64_t>(Rel32(bb + 1))) : 0;
			if (bstub != program->base_vaddr + AuditedStub) {
				std::printf("Buffer probe: OFF (eboot+0x%llx is not a call to the memcpy stub eboot+0x%llx: %02x %02x %02x "
				            "%02x %02x; another build or patched code)\n",
				            static_cast<unsigned long long>(BufferCallSite), static_cast<unsigned long long>(AuditedStub),
				            bb[0], bb[1], bb[2], bb[3], bb[4]);
			} else {
				(void)GetBufferState();
				g_buffer_return_address.store(bcall + 5, std::memory_order_relaxed);
				std::printf("Buffer probe: on (measurement only; KYTY_BUFFER_PROBE=0 turns it off): CBuffer::Create "
				            "(eboot+0x%llx) memcpy call at eboot+0x%llx; tags \"buffer N%%\" in the [faults] lines\n",
				            static_cast<unsigned long long>(BufferFunctionStart),
				            static_cast<unsigned long long>(BufferCallSite));
			}
		}
	}
	const char* v = std::getenv("KYTY_TEXTURE_PROBE");
	if (v != nullptr && v[0] == '0') {
		std::printf("Texture probe: off (KYTY_TEXTURE_PROBE=0)\n");
		return;
	}
	const uint64_t base = program->base_vaddr;
	const uint64_t call = base + CallSite;
	if (program->mapped_size < CallSite + 5 || !Libs::Graphics::HostMemoryRangeIsReadable(call, 5)) {
		std::printf("Texture probe: OFF (eboot+0x%llx is not mapped: another build?)\n",
		            static_cast<unsigned long long>(CallSite));
		return;
	}
	const auto* bytes = reinterpret_cast<const uint8_t*>(call);
	if (bytes[0] != 0xe8) {
		std::printf("Texture probe: OFF (eboot+0x%llx is not a call: %02x %02x %02x %02x %02x; another build or "
		            "patched code)\n",
		            static_cast<unsigned long long>(CallSite), bytes[0], bytes[1], bytes[2], bytes[3], bytes[4]);
		return;
	}
	const uint64_t stub = call + 5 + static_cast<uint64_t>(static_cast<int64_t>(Rel32(bytes + 1)));
	// Where the call goes: the stub's `jmp qword [rip+disp32]` slot and the module its value lies in (printed only).
	uint64_t    slot_value = 0;
	const char* stub_kind  = "not a PLT stub";
	if (Libs::Graphics::HostMemoryRangeIsReadable(stub, 6)) {
		const auto* stub_code = reinterpret_cast<const uint8_t*>(stub);
		if (stub_code[0] == 0xff && stub_code[1] == 0x25) {
			const uint64_t slot = stub + 6 + static_cast<uint64_t>(static_cast<int64_t>(Rel32(stub_code + 2)));
			if (Libs::Graphics::HostMemoryRangeIsReadable(slot, sizeof(slot_value))) {
				std::memcpy(&slot_value, reinterpret_cast<const void*>(slot), sizeof(slot_value));
				stub_kind = "PLT stub";
			} else {
				stub_kind = "PLT stub, GOT slot unreadable";
			}
		}
	}
	char target[160] {};
	std::snprintf(target, sizeof(target), "0x%llx (host: Kyty HLE)", static_cast<unsigned long long>(slot_value));
	if (slot_value != 0 && program->rt != nullptr) {
		if (auto* owner = program->rt->FindProgramByAddr(slot_value); owner != nullptr) {
			const std::string module = owner->file_name.filename().string();
			std::snprintf(target, sizeof(target), "%s+0x%llx", module.c_str(),
			              static_cast<unsigned long long>(slot_value - owner->base_vaddr));
		}
	}
	const bool named = NamesFunction(base + FunctionStart, call);
	(void)GetState();
	g_return_address.store(call + 5, std::memory_order_relaxed);
	std::thread(Reporter).detach();
#if defined(_WIN32)
	{
		const char* r = std::getenv("KYTY_TEXTURE_RELEASE");
		if (r == nullptr || r[0] != '1') {
			std::printf("Texture release: off (KYTY_TEXTURE_RELEASE=1 turns it on)\n");
		} else {
			g_release.store(true, std::memory_order_relaxed);
			std::thread(Releaser).detach();
			std::printf("Texture release: on (KYTY_TEXTURE_RELEASE=1): streamed chunk ranges still live "
			            "%llu vblanks (min %lld ms, max %lld ms) after their copy leave the working set (VirtualUnlock, "
			            "contents kept), 1 MiB at a time, <= ~200 MiB/s; a \"Texture release (5 s)\" line every 5 s\n",
			            static_cast<unsigned long long>(ReleaseDelayVblanks), static_cast<long long>(ReleaseMinMs),
			            static_cast<long long>(ReleaseMaxMs));
		}
	}
#else
	std::printf("Texture release: off (Windows only)\n");
#endif
	std::printf("Texture probe: on (measurement only; KYTY_TEXTURE_PROBE=0 turns it off): memcpy call at eboot+0x%llx "
	            "-> eboot+0x%llx (%s; 01.007.000: 0x%llx, %s) -> %s; SetCurrentStreamingChunk string %s; a "
	            "\"Texture probe (5 s)\" line every 5 s once a chunk is copied\n",
	            static_cast<unsigned long long>(CallSite), static_cast<unsigned long long>(stub - base), stub_kind,
	            static_cast<unsigned long long>(AuditedStub), stub - base == AuditedStub ? "same" : "differs",
	            target,
	            named ? "found in the function" : "NOT found (another build? the counts may be of another call)");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

} // namespace Loader::DemonsSoulsTextureProbe
