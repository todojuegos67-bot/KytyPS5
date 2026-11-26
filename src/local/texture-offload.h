#pragma once
// Texture offload (Windows). The guest-memory copy of a texture GPU work never wrote, that only shaders sample, uploaded
// and clean for a while, duplicates its image in video memory: on a PS5 the game's memory holds it once.
//
// KYTY_TEXTURE_OFFLOAD_PROBE=1 (diagnostics): such a texture's pages are made inaccessible in both views of guest memory
// (the guest's mapping: read watchers of the page manager; the backing view the emulator reads and writes: no access) and
// nothing else. Every later touch faults, is counted with who made it and how long after, and gives the range its
// protection back for good: what keeping the texture only in video memory would cost.
//
// KYTY_TEXTURE_OFFLOAD=1 (default with 20 GiB of RAM or less): the same for textures whose download was checked to give
// their guest bytes back exactly (16 MiB of images a scan: the downloads go with the frame's GPU work), and their pages
// are dropped by the offload worker (DiscardVirtualMemory in the backing view a MiB at a time, the guest view's pages
// trimmed: the RAM given back, no pagefile write). On a 16 GB PC with a 1-1 -> Stonefang -> 1-1 route: up to 2 GB of
// guest memory out of RAM. Whoever needs the bytes again gets them back first, downloaded from the image by the render
// thread (TextureCache::RestoreOffloaded, waiting for the GPU): an upload of an image over them (InitializeImage), a host
// write over part of them (InvalidateMemory), the game's code touching them (its fault: the render thread through
// SendCommandSync); the image going (UnregisterImage: the collector retires images no draw used for 20 s) has them
// written back by the offload worker when the GPU is done, touches waiting till then. A host overwrite of all of them or
// an unmap only drops them. A touch by a thread that cannot wait for the render thread (the upload worker, deferred GPU
// work, a speculation's recorder) is a miss: counted and logged, the bytes it sees zeros, and the render thread restores
// them after (Drain).
//
// KYTY_GPU_ONLY_OFFLOAD=1|0 (default: on with the low-RAM profile, or while the RAM guard's pressure is critical; needs
// the offload on; =2 also formats no check proved, diagnostics): the guest bytes of GPU-written images too (render
// targets, storage images; not depth, linear ones, ones dirty, read back or under GPU buffer writes), clean at two scans
// in a row and of a format (guest format + tiling) a texture check proved the download of. Their guest copy is stale or
// the same: a touch gets the image's bytes as a CPU read back of it would. Under critical pressure images from 64 KiB.
// Logged: "GPU-only offload: ..." (ranges, MiB, restores and their time), and every 30 s "Guest RAM by GPU cache" (what
// the GPU caches know of the guest bytes: what could be dropped and why the rest cannot).
//
// (ram15) KYTY_NATIVE_TEXTURE_OFFLOAD=1 (off by default) with native guest memory (KYTY_NATIVE_MEMORY / KYTY_RAIZ, where
// the offload was off: no backing view). The same textures are taken, their pages made no-access by the page manager's
// read watchers (never protected here: every protection change of guest memory goes through the page manager or the
// gate), and their RAM dropped at their guest address by the offload worker (MEM_RESET + out of the working set,
// memory.cpp NativeOffloadDrop: no page-file or compression copy). A touch waits for the render thread to download the
// bytes back from the image (written through a write window, NativeOffloadWrite) - a game thread's fault, a window of
// the emulator's over them (NativeGuardedAccess's hook: its copies do not fault) -; a touch that cannot wait (the upload
// worker, deferred GPU work, a recorder, a window inside a window) is a miss: counted, its pages opened as they are,
// restored after (Drain). An image gone (evicted) gives its bytes back first, as before; the collector takes the stale
// images whose bytes are still in RAM before those whose bytes it would have to download (KYTY_NATIVE_OFFLOAD_EVICT_LAST=0:
// LRU order only). Ghidra 01.007.000: the streamed textures are written by the file read (0x8fed14) and the chunk memcpy
// 0x901459 (CTexture::SetCurrentStreamingChunk 0x901330) into VirtualMemoryGPU (0x2f64438) and decommitted (op 2,
// 0x9014f4); no read of their bytes by the CPU found in CTexture's code. Measured (backing view, 11-15, 18 runs): 0 game
// reads of taken ranges, 1-15 game writes (memory reused) a run. Every 30 s: "Native texture offload (30 s ...)" (RAM
// saved now in GB, downloads back and why, misses, windows checked, the worker's time).
//
// Read watchers go at once after a touch by game code (a guest thread holds no lock of the emulator's) or a host write or
// unmap (normal context); after one by the emulator's own code the fault handler only gives the pages a protection that
// lets the access through, and the render thread removes them (Drain).

#include "guest-lazy-commit.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "local-platform.h"
#include "mem1.h" // (mem1: the emulator's windows over the streamed textures' zone, stamped)
#include "ram-native-tex.h" // (ram1b: KYTY_RAM_NATIVE_TEX)
#include "tiron4-counters.h" // (tiron4)
#include "vram-store.h"     // (mem1 etapa 2: the store in video memory, KYTY_MEM1_STAGE2)
#include "buf-nativo.h"    // (BUF-NATIVO: KYTY_BUF_NATIVO, las mallas)
#include "raiz.h" // (ram15: KYTY_RAIZ forces KYTY_NATIVE_MEMORY=1 at its first read)

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h> // (ram5: K32QueryWorkingSetEx)
#include <winioctl.h> // (nativo-ram-2: IOCTL_STORAGE_QUERY_PROPERTY)
#endif

namespace TextureOffload {

enum class Mode { Off, Probe, Offload };

// KYTY_TEXTURE_OFFLOAD=1|0 (default: on with 20 GiB of RAM or less, where the game's memory and the emulator's caches
// press on it; with more, its restores' stalls buy nothing).
inline Mode GetMode() {
#if defined(_WIN32)
	static const Mode mode = [] {
		// KYTY_NATIVE_MEMORY=1 (kernel/memoryAddressSpace.inc, NativeMemoryRequested): no backing view - the "backing
		// pieces" are the guest addresses themselves -, so the offload would change the guest view's protection (which
		// that mode never does), drop the game's own pages (DiscardVirtualMemory, KYTY_OFFLOAD_PAGEOUT=0) and restore
		// into guest memory the game may have unmapped since. Off then.
		(void)Raiz::On(); // (ram15: sets KYTY_NATIVE_MEMORY=1 with KYTY_RAIZ=1 first)
		if (const char* native = std::getenv("KYTY_NATIVE_MEMORY"); native != nullptr && native[0] == '1') {
			// (ram15) KYTY_NATIVE_TEXTURE_OFFLOAD=1: on, at the guest addresses themselves (above).
			const char* on = std::getenv("KYTY_NATIVE_TEXTURE_OFFLOAD");
			if (on != nullptr && on[0] == '1') {
				std::printf("Texture offload: on with native memory (KYTY_NATIVE_TEXTURE_OFFLOAD=1): verified clean textures' guest "
				            "pages dropped from RAM at their guest address, downloaded back from the image on any touch\n");
				std::fflush(stdout);
				return Mode::Offload;
			}
			std::printf("Texture offload: off (KYTY_NATIVE_MEMORY: no backing view; KYTY_NATIVE_TEXTURE_OFFLOAD=1 turns it on)\n");
			return Mode::Off;
		}
		const char* offload = std::getenv("KYTY_TEXTURE_OFFLOAD");
		if (offload != nullptr && offload[0] != '\0') {
			if (offload[0] == '1') return Mode::Offload;
		} else if (const auto ram = LocalPlatform::PhysicalMemory(); ram != 0 && ram <= (uint64_t {20} << 30u)) {
			return Mode::Offload;
		}
		const char* probe = std::getenv("KYTY_TEXTURE_OFFLOAD_PROBE");
		return probe != nullptr && probe[0] == '1' ? Mode::Probe : Mode::Off;
	}();
	return mode;
#else
	return Mode::Off;
#endif
}

inline bool Enabled() {
	return GetMode() != Mode::Off;
}

// (ram15) The offload over native guest memory: its "backing pieces" are the guest pages themselves.
inline bool Native() {
#if defined(_WIN32)
	static const bool native = [] {
		if (GetMode() != Mode::Offload) return false;
		const char* value = std::getenv("KYTY_NATIVE_MEMORY");
		return value != nullptr && value[0] == '1';
	}();
	return native;
#else
	return false;
#endif
}

enum class Touch : uint8_t {
	GameRead,
	GameWrite,
	EmulatorGuestRead,
	EmulatorGuestWrite,
	BackingRead,
	BackingWrite,
	HostWrite,
	HostOverwrite,
	Unmap,
	Upload,
	ImageGone,
	Readback, // (KYTY_GPU_ONLY_OFFLOAD: a GPU-written image's download to guest memory over it, TryDownloadImage)
	Count
};

inline const char* TouchName(Touch touch) {
	static const char* names[] = {"game read",
	                              "game write",
	                              "emulator read (guest view)",
	                              "emulator write (guest view)",
	                              "emulator read (backing view)",
	                              "emulator write (backing view)",
	                              "host write over part",
	                              "host write over all",
	                              "unmap",
	                              "image upload over it",
	                              "image gone",
	                              "GPU result written back"};
	return names[static_cast<int>(touch)];
}

struct Piece {
	uint64_t backing = 0, size = 0;
};

struct Range {
	uint64_t                              guest_end = 0;
	std::vector<Piece>                    pieces;
	uint64_t                              serial = 0, image = 0; // (the image's serial and packed id)
	std::chrono::steady_clock::time_point since;
	bool                                  discarded = false;
	bool                                  touched   = false; // (protection given back; its read watchers go: Drain)
	bool                                  missed    = false; // (discarded and touched by a thread that could not wait)
	bool                                  restoring = false; // (its restore runs: other touches wait for its end)
	std::thread::id                       restorer {};       // (the thread restoring it; none: the offload worker)
	bool                                  gpu       = false; // (KYTY_GPU_ONLY_OFFLOAD: a GPU-written image's range)
	uint64_t                              dropped   = 0;     // (ram15, native: bytes the worker reset, not given back yet)
	// (ram17, KYTY_RAM17) Its guest bytes in the disk store (file_len 0: none), guest byte begin + i at file_off + i; an
	// orphan: its image went, its bytes are only there (no download, nothing back in RAM until a touch).
	uint64_t                              file_off  = 0, file_len = 0;
	bool                                  orphan    = false;
	// (ram18, KYTY_RAM18) A cold range of the game's memory (no image: RAM18 below): its bytes only ever come back from
	// the disk store, never from the GPU; if they cannot be kept there it is not dropped.
	bool                                  cold      = false;
	// (ram20, KYTY_RAM20) A cold range taken because video memory holds a copy of it: 1 + its kind (GpuResidency::Kind:
	// texture, render target, GPU-written image, buffer, GPU-written buffer); 0: none (a RAM18 range or a texture).
	uint8_t                               r20       = 0;
	// (ram1b, KYTY_RAM_NATIVE_TEX=1) A cold range taken because a GPU cache's image holds it: 1 a texture array's layers
	// (VirtualMemoryGPU), 2 a render target (RTShared/RTPersist); 0: none.
	// (ram2, KYTY_RAM2=1) 3 (RntRam2): memory of that zone no image has used for KYTY_RAM2_AGE_S since it was mapped. Its
	// bytes' hash as written to the disk store (r2_hashed: kept whole since; a range cut by an unmap is not verified).
	uint8_t                               rnt       = 0;
	uint64_t                              r2_hash   = 0;
	bool                                  r2_hashed = false;
};
constexpr uint8_t RntRam2 = 3; // (ram2: Range::rnt of its ranges; not one of State::RntKinds, whose counters stay ram1b's)
constexpr uint8_t RntMesh = 4; // (BUF-NATIVO: los datos de una malla, buf-nativo.h; fuera de State::RntKinds, sus contadores mesh_*)

// Set by the texture cache: the page manager's read watchers of a guest range removed; a restore of the discarded ranges
// over a guest range by the render thread (any thread CanWait allows: it returns when they are back); whether the calling
// thread may wait for the render thread.
using RemoveWatchers = void (*)(uint64_t vaddr, uint64_t size);
using Restore        = void (*)(uint64_t vaddr, uint64_t size, Touch touch);
using CanWait        = bool (*)();
// (ram15, native memory; set by the texture cache: memory.cpp's NativeOffloadDrop / Write / Open.)
using NativeDrop  = uint64_t (*)(uint64_t vaddr, uint64_t size, uint64_t* refused);
using NativeWrite = bool (*)(uint64_t vaddr, const void* data, uint64_t size);
using NativeOpen  = bool (*)(uint64_t vaddr, uint64_t size, bool write);
// (ram17; set by the texture cache: memory.cpp's NativeGuardedRead / NativeForgetWrites.)
using NativeRead   = bool (*)(void* destination, const void* guest, uint64_t size);
using NativeForget = void (*)(uint64_t vaddr, uint64_t size);

struct State {
	std::mutex                   mutex;
	std::map<uint64_t, Range>    guest;   // by guest start
	std::map<uint64_t, uint64_t> backing; // backing piece start -> guest start
	std::vector<uint64_t>        pending; // touched ranges (guest starts) whose read watchers go (Drain)
	std::unordered_set<uint64_t> touched_serials; // images touched once: not taken again
	// (ram16, KYTY_RAM16: KYTY_RAM16_RETAKE_S) When each image of touched_serials was touched: taken again once it has been
	// that long since (TouchedLocked). Only filled with the retake on.
	std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> touched_at;
	uint64_t retaken = 0; // (ram16: images let back in after their wait, in all)
	std::atomic<RemoveWatchers>  remove {nullptr};
	std::atomic<Restore>         restore {nullptr};
	std::atomic<CanWait>         can_wait {nullptr};
	std::atomic<uint64_t>        guest_lo {~uint64_t {0}}, guest_hi {0}, backing_lo {~uint64_t {0}}, backing_hi {0};
	// Statistics.
	uint64_t now = 0, peak = 0, total = 0, ranges_total = 0;
	uint64_t touches[static_cast<int>(Touch::Count)] {}, touch_bytes[static_cast<int>(Touch::Count)] {};
	uint64_t ages[4] {}; // <1 s, <10 s, <60 s, later
	uint64_t pageout_bytes = 0; // (KYTY_OFFLOAD_PAGEOUT: bytes taken out of the working set)
	uint64_t unmap_splits  = 0; // (ram3, KYTY_OFFLOAD_UNMAP_SPLIT: ranges an unmap took only a part of)
	// (ram5, KYTY_OFFLOAD_PAGEOUT=0) Bytes DiscardVirtualMemory took, refused (then MEM_RESET + VirtualUnlock: taken by
	// that or still refused), and the backing view's pages still in the working set right after (sampled, a page a MiB).
	uint64_t discard_ok = 0, discard_refused = 0, reset_ok = 0, reset_refused = 0, resident_after = 0, sampled_after = 0;
	uint32_t discard_error = 0;
	uint64_t restores = 0, restore_bytes = 0, restore_us = 0, restore_us_max = 0, lost = 0, missed = 0, missed_bytes = 0;
	uint64_t checked = 0, checked_bytes = 0, check_failed = 0;
	// KYTY_GPU_ONLY_OFFLOAD: the GPU-written images' ranges (render targets, storage images) among the above.
	uint64_t gpu_now = 0, gpu_peak = 0, gpu_total = 0, gpu_ranges_total = 0, gpu_restores = 0, gpu_restore_bytes = 0;
	uint64_t gpu_restore_us = 0, gpu_restore_us_max = 0;
	// Ranges being restored (any thread), and buffer uploads that waited for an asynchronous restore (WaitBeforeRead).
	std::atomic<uint64_t> restoring_ranges {0}, read_waits {0}, read_wait_bytes {0};
	std::unordered_map<uint64_t, uint64_t> rips;    // the emulator's instructions -> touches
	std::unordered_map<uint64_t, uint64_t> callers; // their callers (one frame unwound) -> touches
	// (ram15) Native memory: its calls, and what it dropped and gave back.
	std::atomic<NativeDrop>  native_drop {nullptr};
	std::atomic<NativeWrite> native_write {nullptr};
	std::atomic<NativeOpen>  native_open {nullptr};
	uint64_t native_now = 0, native_peak = 0, native_total = 0, native_refused = 0, native_drops = 0, native_drop_us = 0;
	uint64_t native_drop_us_max = 0, native_write_failed = 0, native_opened = 0;
	uint64_t restores_by[static_cast<int>(Touch::Count) + 1] {}, restore_bytes_by[static_cast<int>(Touch::Count) + 1] {};
	uint64_t evict_offloaded = 0, evict_deferred = 0; // (the collector: offloaded images evicted, put after the others)
	uint64_t trusted = 0, trusted_bytes = 0;          // (KYTY_NATIVE_OFFLOAD_TRUST_FORMAT: taken by their format's proof)
	std::atomic<uint64_t> window_calls {0}, window_lookups {0}, window_hits {0}, window_missed {0}, window_ns {0};
	// (ram17, KYTY_RAM17) The disk store: its calls, what it wrote and read back, the images gone with no download (their
	// ranges left as orphans), the takes held back after a quick return, and when each range read back did so.
	std::atomic<NativeRead>   native_read {nullptr};
	std::atomic<NativeForget> native_forget {nullptr};
	uint64_t file_writes = 0, file_write_bytes = 0, file_write_us = 0, file_write_failed = 0;
	uint64_t file_reads = 0, file_read_bytes = 0, file_read_us = 0, file_read_us_max = 0, file_read_failed = 0, file_quick = 0;
	uint64_t orphaned = 0, orphaned_bytes = 0, held = 0;
	std::map<uint64_t, std::pair<uint64_t, std::chrono::steady_clock::time_point>> recent; // begin -> (end, when back)
	// (ram18, KYTY_RAM18) Cold ranges: proposed by the RAM18 thread (cold_queue), taken by the render thread (watchers on:
	// cold_drop), written to the disk store and dropped by the RAM18 thread. Their numbers apart from the textures'.
	std::vector<std::pair<uint64_t, uint64_t>> cold_queue;
	std::vector<uint64_t>                      cold_drop;
	uint64_t cold_now = 0, cold_peak = 0, cold_ranges = 0, cold_taken = 0, cold_taken_bytes = 0, cold_written = 0;
	uint64_t cold_written_bytes = 0, cold_write_us = 0, cold_zero = 0, cold_zero_bytes = 0, cold_failed = 0, cold_reads = 0;
	uint64_t cold_read_bytes = 0, cold_read_us = 0, cold_read_us_max = 0, cold_lost = 0, cold_unmapped = 0, cold_released = 0;
	// (ram20, KYTY_RAM20) Its own proposals (begin, size, 1 + kind) and its numbers by kind (index r20 - 1), within the
	// cold ranges' above.
	std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> r20_queue;
	static constexpr size_t R20Kinds = 5;
	uint64_t r20_now[R20Kinds] {}, r20_peak = 0, r20_taken[R20Kinds] {}, r20_taken_bytes[R20Kinds] {};
	uint64_t r20_reads[R20Kinds] {}, r20_read_bytes[R20Kinds] {}, r20_read_us = 0, r20_read_us_max = 0;
	uint64_t r20_zero_bytes = 0, r20_written_bytes = 0, r20_lost = 0, r20_failed = 0, r20_released = 0;
	// (ram1b, KYTY_RAM_NATIVE_TEX=1) Its ranges by kind (index rnt - 1: arrays, render targets), within the cold ones; the
	// touches that brought them back by how long after the take (<10 s, <60 s, later).
	static constexpr size_t RntKinds = 2;
	uint64_t rnt_now[RntKinds] {}, rnt_peak = 0, rnt_taken[RntKinds] {}, rnt_taken_bytes[RntKinds] {};
	uint64_t rnt_reads[RntKinds] {}, rnt_read_bytes[RntKinds] {}, rnt_read_us = 0, rnt_read_us_max = 0;
	uint64_t rnt_zero_bytes = 0, rnt_written_bytes = 0, rnt_lost = 0, rnt_failed = 0, rnt_released = 0, rnt_ages[3] {};
	// (ram2, KYTY_RAM2=1) Its ranges' numbers (Range::rnt == RntRam2), within the cold ones: out of RAM now and peak, taken,
	// written / all zero / not written, read back by a touch (bytes, time, the longest, by age since the take: <10 s,
	// <60 s, later), unmapped while out, lost; the verification of each read back (checked, equal, DISCREPANCIA, time).
	uint64_t r2_now = 0, r2_peak = 0, r2_taken = 0, r2_taken_bytes = 0, r2_written_bytes = 0, r2_zero_bytes = 0, r2_failed = 0;
	uint64_t r2_reads = 0, r2_read_bytes = 0, r2_read_us = 0, r2_read_us_max = 0, r2_ages[3] {}, r2_unmapped = 0, r2_lost = 0;
	uint64_t r2_checked = 0, r2_checked_bytes = 0, r2_equal = 0, r2_discrepancy = 0, r2_unchecked = 0, r2_check_us = 0;
	// (BUF-NATIVO) Las mallas (Range::rnt == RntMesh), dentro de los frios: fuera de RAM ahora y pico, tomadas, fallos de
	// la CPU sobre ellas (juego / emulador), leidas de vuelta (restauraciones), olvidadas al destruir la malla, perdidas.
	uint64_t mesh_now = 0, mesh_peak = 0, mesh_taken = 0, mesh_taken_bytes = 0, mesh_refused = 0, mesh_faults_game = 0;
	uint64_t mesh_faults_emu = 0, mesh_reads = 0, mesh_read_bytes = 0, mesh_read_us = 0, mesh_read_us_max = 0;
	uint64_t mesh_forgot = 0, mesh_forgot_bytes = 0, mesh_lost = 0, mesh_failed = 0, mesh_written_bytes = 0;
};

inline State& Get() {
	static State state;
	return state;
}

// (ram17) KYTY_RAM17=1 (off by default; only with the native offload: KYTY_RAM16=1 / KYTY_NATIVE_TEXTURE_OFFLOAD=1, which
// ram-guard.cpp turns on with it). The guest bytes of a texture taken are written to a file of the emulator's own (the
// disk store, kyty-ram17-<pid>.tmp next to the exe or in KYTY_RAM17_DIR, unbuffered, deleted on close; at most
// KYTY_RAM17_FILE_MB, default 8192, and 2 GB of the disk left free) before their RAM is dropped. Then:
//  - an image gone (evicted) leaves its bytes there: no download from the GPU, nothing back in RAM (its ranges stay
//    taken, as orphans, until something touches them);
//  - a touch, by any thread (a game thread's fault, a window of the emulator's, the upload worker), reads them back from
//    the file itself: no wait for the render thread or the GPU, never a miss;
//  - a range read back is not taken again for KYTY_RAM17_HOLD_S (default 30; 0 off): the textures that leave and come
//    back at once stay in RAM meanwhile.
// Ranges over KYTY_RAM17_MAX_MB (default 32) are not written (a touch would wait for the write). A write or read of the
// file that fails leaves the range as before (GPU download; a read failure is a miss, counted). Measured before
// (11-16 11:01, RAM16): 1633 of the 1657 downloads back were images gone (1080 MiB, 6.9 s), 1321 of 1937 touches came
// less than 10 s after the take, and VirtualMemo held >= 0.87 GB in RAM in no image of the cache. Line: "[ram17]".
inline bool Ram17On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_RAM17");
		return value != nullptr && value[0] == '1' && Native();
	}();
	return on;
}
inline int Ram17HoldSeconds() {
	static const int seconds = [] {
		if (!Ram17On()) return 0;
		const char* value = std::getenv("KYTY_RAM17_HOLD_S");
		if (value == nullptr || value[0] == '\0') return Tiron4::TexOn() ? 300 : 30; // (tiron5: KYTY_TIRON5_TEX=1)
		const long asked = std::strtol(value, nullptr, 10);
		return asked <= 0 ? 0 : static_cast<int>(std::min<long>(asked, 3600));
	}();
	return seconds;
}
inline void FileFree(uint64_t offset, uint64_t bytes); // (any thread; never takes the state's lock)

// (ram18) KYTY_RAM18=1 (off by default; implies KYTY_RAM17=1, ram-guard.cpp, and needs it on): the game's memory that
// Windows itself found cold - out of the working set at every sample for KYTY_RAM18_COLD_S (default 10) -, in the labels
// of KYTY_RAM18_LABELS (default VirtualMemo: the streamed textures' zone, which the CPU never reads, Ghidra 01.007.000
// 0x8fed14 / 0x901459 write it, no read found), taken as ranges of up to 1 MiB with no image: written to the disk store
// and dropped (MEM_RESET: the working-set page and the compressed copy in Windows' store go). Any touch reads them back
// from the file on its own thread (the RAM17 paths); an all-zero range is only reset and let go; a range that cannot be
// written is not dropped. Measured (11-16 18:33, RAM17): Windows' compression 4.37-4.60 GB with 0.4-0.9 GB available, and
// VirtualMemo in RAM going 2.19 -> 1.27 GB while the textures out of RAM stayed at 0.6-0.7 GB: its pages went to the
// compression store. Line "[ram18]" every 5 s.
// (ram20) KYTY_RAM20=1 (gpu-residency.h, ram-guard.cpp) uses the cold-range path too: Ram18On is true with either; the
// RAM18 thread proposes its own (VirtualMemo) cold ranges only with KYTY_RAM18=1 (Ram18Own).
inline bool Ram20Asked() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_RAM20");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}
inline bool Ram18Own() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_RAM18");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}
// (ram1b) KYTY_RAM_NATIVE_TEX=1 takes its ranges through the cold-range path too (=2 takes none).
inline bool RntAsked() {
	return RamNativeTex::Mode() == 1;
}
inline bool Ram18On() {
	static const bool on = (Ram18Own() || Ram20Asked() || RntAsked() || BufNativo::On()) && Ram17On(); // (BUF-NATIVO: su hilo suelta las mallas)
	return on;
}
inline bool Ram20Active() {
	return Ram20Asked() && Ram18On();
}
// (ram18) A cold range's serial (the RAM16 retake rule and touched_serials: per range start, never an image's).
constexpr uint64_t ColdSerial(uint64_t begin) {
	return (uint64_t {1} << 63u) | begin;
}

// (state.mutex held) The range's bytes in the disk store forgotten.
inline void DropFileLocked(Range& range) {
	if (range.file_len == 0) return;
	FileFree(range.file_off, range.file_len);
	range.file_len = 0;
	range.file_off = 0;
}

// (ram16) KYTY_RAM16=1 (KYTY_RAM16_RETAKE_S=<s>, default 60; 0 off): an image touched once (its bytes given back) may be
// taken again once it has been clean that long after the touch, instead of never (touched_serials). It has to pass the
// same gates as the first time (registered, clean at two scans, sampled only, a format a check proved exact), so its
// bytes in RAM are again those of its image; the wait bounds a texture the game keeps rewriting to one take and one
// restore a minute. Measured before (11-16 07:41, "touched once"): 140 MiB of sampled textures held in RAM that way.
// Without KYTY_RAM16 nothing changes.
inline int Ram16RetakeSeconds() {
	static const int seconds = [] {
		const char* on = std::getenv("KYTY_RAM16");
		if (on == nullptr || on[0] != '1') return 0;
		const char* value = std::getenv("KYTY_RAM16_RETAKE_S");
		if (value == nullptr || value[0] == '\0') return Tiron4::TexOn() ? 300 : 60; // (tiron5: KYTY_TIRON5_TEX=1)
		const long asked = std::strtol(value, nullptr, 10);
		return asked <= 0 ? 0 : static_cast<int>(std::min<long>(asked, 3600));
	}();
	return seconds;
}

// (state.mutex held) Whether image `serial` was touched once and may not be taken now (ram16: its wait over, it is let
// back in and forgotten).
inline bool TouchedLocked(State& state, uint64_t serial) {
	if (state.touched_serials.count(serial) == 0) return false;
	const int seconds = Ram16RetakeSeconds();
	if (seconds == 0) return true;
	const auto at = state.touched_at.find(serial);
	if (at == state.touched_at.end() || std::chrono::steady_clock::now() - at->second < std::chrono::seconds(seconds)) return true;
	state.touched_at.erase(at);
	state.touched_serials.erase(serial);
	if ((serial >> 63u) == 0) state.retaken++; // (ram18: a cold range's serial is not an image's)
	return false;
}

// (ram16) The "[ram16]" line's numbers (ram-guard.cpp, its thread): the native offload's bytes out of RAM now and its
// counters so far. Never waits: false (nothing read) when the state's lock is busy or the native offload is off.
struct Ram16View {
	uint64_t saved = 0, total = 0, restores = 0, restore_bytes = 0, restore_us = 0, missed = 0, retaken = 0, trusted_bytes = 0;
	// (ram17) The disk store (FileUse below: in use / file size), its writes, reads back, orphans, holds, failures.
	uint64_t file_used = 0, file_size = 0, file_writes = 0, file_write_bytes = 0, file_write_us = 0, file_reads = 0;
	uint64_t file_read_bytes = 0, file_read_us = 0, file_read_us_max = 0, file_quick = 0, orphaned = 0, orphaned_bytes = 0;
	uint64_t held = 0, file_failed = 0;
	uint64_t cold_now = 0; // (ram18: the cold ranges out of RAM now, not in `saved`)
};
inline void FileUse(uint64_t* used, uint64_t* size);
inline bool Ram16Read(Ram16View* out) {
	if (!Native()) return false;
	auto&                        state = Get();
	std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return false;
	out->saved         = state.native_now;
	out->total         = state.native_total;
	out->restores      = state.restores;
	out->restore_bytes = state.restore_bytes;
	out->restore_us    = state.restore_us;
	out->missed        = state.missed;
	out->retaken       = state.retaken;
	out->trusted_bytes = state.trusted_bytes;
	// (ram17)
	out->file_writes      = state.file_writes;
	out->file_write_bytes = state.file_write_bytes;
	out->file_write_us    = state.file_write_us;
	out->file_reads       = state.file_reads;
	out->file_read_bytes  = state.file_read_bytes;
	out->file_read_us     = state.file_read_us;
	out->file_read_us_max = state.file_read_us_max;
	out->file_quick       = state.file_quick;
	out->orphaned         = state.orphaned;
	out->orphaned_bytes   = state.orphaned_bytes;
	out->held             = state.held;
	out->file_failed      = state.file_write_failed + state.file_read_failed;
	out->cold_now         = state.cold_now; // (ram18)
	lock.unlock();
	FileUse(&out->file_used, &out->file_size);
	return true;
}

// (ram17, state.mutex held) Whether [vaddr, vaddr + size) was read back from the disk store less than KYTY_RAM17_HOLD_S
// ago: not taken again yet (Taken). Old entries forgotten as the map grows.
inline bool HeldLocked(State& state, uint64_t vaddr, uint64_t size) {
	const int hold = Ram17HoldSeconds();
	if (hold == 0 || state.recent.empty()) return false;
	const auto now    = std::chrono::steady_clock::now();
	const auto window = std::chrono::seconds(hold);
	if (state.recent.size() > 4096)
		std::erase_if(state.recent, [&](const auto& item) { return now - item.second.second >= window; });
	auto it = state.recent.upper_bound(vaddr);
	if (it != state.recent.begin()) --it;
	for (; it != state.recent.end() && it->first < vaddr + size; ++it) {
		if (it->second.first <= vaddr) continue;
		if (now - it->second.second < window) return true;
	}
	return false;
}

// ---- (ram17) The disk store: space in 4 KiB units (the file is unbuffered: offsets, sizes and buffers aligned) ----
constexpr uint64_t FileUnit  = 4096;
constexpr uint64_t FileChunk = uint64_t {1} << 20u; // (a read or write of the file, and a window of guest memory)
// (nativo-ram-3) The read back's cost split (KYTY_NATIVE_TEX_DISK): the file read, the write over the guest pages
// (native_write: commit/protection), the watchers forgotten, the ram2 check, the waits for another thread's restore or a
// cold range being written; and how many read backs ran from an emulator window (render thread or uploads: the touch was
// the emulator's own) and their time. Relaxed counters, read by the "[nativo-ram]" line.
struct NrSplit {
	std::atomic<uint64_t> io_us {0}, guest_us {0}, forget_us {0}, check_us {0}, wait_us {0}, emu_n {0}, emu_us {0}, emu_max_us {0};
};
inline NrSplit& NrCost() {
	static NrSplit split;
	return split;
}

#if defined(_WIN32)
struct FileStore {
	std::mutex                   mutex;
	HANDLE                       handle = INVALID_HANDLE_VALUE;
	bool                         opened = false;
	bool                         overlapped = false; // (nativo-ram-3: opened FILE_FLAG_OVERLAPPED, no write-through)
	uint64_t                     end = 0, cap = 0, used = 0;
	std::map<uint64_t, uint64_t> holes; // offset -> bytes (free space below end)
};
inline FileStore& Files() {
	static FileStore files;
	return files;
}

// (nativo-ram-2) 1: the volume of `path` has a seek penalty (hard disk), 0: none (SSD), -1: unknown.
inline int SeekPenalty(const wchar_t* path) {
	wchar_t root[MAX_PATH] {};
	if (GetVolumePathNameW(path, root, MAX_PATH) == 0) return -1;
	wchar_t volume[MAX_PATH] {};
	if (GetVolumeNameForVolumeMountPointW(root, volume, MAX_PATH) == 0) return -1;
	const size_t len = std::wcslen(volume);
	if (len != 0 && volume[len - 1] == L'\\') volume[len - 1] = L'\0'; // (the volume device, without the trailing slash)
	const HANDLE h = CreateFileW(volume, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
	if (h == INVALID_HANDLE_VALUE) return -1;
	STORAGE_PROPERTY_QUERY query {};
	query.PropertyId = StorageDeviceSeekPenaltyProperty;
	query.QueryType  = PropertyStandardQuery;
	DEVICE_SEEK_PENALTY_DESCRIPTOR out {};
	DWORD                          got = 0;
	const BOOL ok = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), &out, sizeof(out), &got, nullptr);
	CloseHandle(h);
	if (ok == 0 || got < sizeof(out)) return -1;
	return out.IncursSeekPenalty ? 1 : 0;
}
inline bool NativeDiskSsdWanted() {
	const char* disk = std::getenv("KYTY_NATIVE_TEX_DISK");
	if (disk == nullptr || disk[0] != '1') return false;
	const char* ssd = std::getenv("KYTY_NATIVE_TEX_DISK_SSD");
	return ssd == nullptr || ssd[0] != '0';
}

// (files.mutex held) The file, opened at the first use (once: a failure leaves the store off).
inline bool FileOpenLocked(FileStore& files) {
	if (files.opened) return files.handle != INVALID_HANDLE_VALUE;
	files.opened = true;
	wchar_t     dir[MAX_PATH] {};
	const char* asked = std::getenv("KYTY_RAM17_DIR");
	if (asked != nullptr && asked[0] != '\0') {
		if (MultiByteToWideChar(CP_ACP, 0, asked, -1, dir, MAX_PATH) == 0) return false;
	} else {
		const DWORD n = GetModuleFileNameW(nullptr, dir, MAX_PATH);
		if (n == 0 || n >= MAX_PATH) return false;
		if (wchar_t* slash = std::wcsrchr(dir, L'\\'); slash != nullptr) slash[1] = L'\0';
	}
	if (const size_t length = std::wcslen(dir); length != 0 && dir[length - 1] != L'\\' && dir[length - 1] != L'/' && length + 1 < MAX_PATH) {
		dir[length]     = L'\\';
		dir[length + 1] = L'\0';
	}
	// (nativo-ram-2) With KYTY_NATIVE_TEX_DISK the store is read back while the game plays: on a disk with a seek penalty
	// (a hard disk: 20-60 ms per read back, the 11-23 tirones) it moves to the temp folder when that one has none (an SSD).
	// KYTY_RAM17_DIR still wins; KYTY_NATIVE_TEX_DISK_SSD=0 keeps the old place.
	if ((asked == nullptr || asked[0] == '\0') && NativeDiskSsdWanted()) {
		const int   here = SeekPenalty(dir);
		wchar_t     temp[MAX_PATH] {};
		const DWORD tn    = GetTempPathW(MAX_PATH, temp);
		const int   there = (tn != 0 && tn < MAX_PATH) ? SeekPenalty(temp) : -1;
		std::printf("[nativo-ram] almacen: %ls %s; temporal %ls %s\n", dir, here == 1 ? "con busqueda (disco duro)" : here == 0 ? "SSD" : "desconocido",
		            temp, there == 1 ? "con busqueda (disco duro)" : there == 0 ? "SSD" : "desconocido");
		if (here != 0 && there == 0) {
			std::wcscpy(dir, temp);
			std::printf("[nativo-ram] almacen movido al temporal (SSD)\n");
		}
		std::fflush(stdout);
	}
	const char* mb = std::getenv("KYTY_RAM17_FILE_MB");
	files.cap      = (mb != nullptr && mb[0] != '\0' ? std::strtoull(mb, nullptr, 10) : uint64_t {8192}) << 20u;
	ULARGE_INTEGER free_bytes {};
	constexpr uint64_t Spare = uint64_t {2} << 30u; // (left free on the disk)
	if (GetDiskFreeSpaceExW(dir, &free_bytes, nullptr, nullptr) != 0)
		files.cap = std::min<uint64_t>(files.cap, free_bytes.QuadPart > Spare ? free_bytes.QuadPart - Spare : 0);
	wchar_t path[MAX_PATH + 64] {};
	std::swprintf(path, sizeof(path) / sizeof(path[0]), L"%lskyty-ram17-%lu.tmp", dir, static_cast<unsigned long>(GetCurrentProcessId()));
	// (nativo-ram-3) With KYTY_NATIVE_TEX_DISK the store is read back while the game plays. A synchronous handle makes
	// Windows run its reads and writes one after another (one file object), and FILE_FLAG_WRITE_THROUGH makes every 1 MiB
	// write of the offload worker wait for the device's cache flush: a read back queued behind them waited 15-66 ms on an
	// SSD (11-23 11:04). The file is deleted on close (nothing to keep through a crash): opened overlapped (reads and writes
	// run side by side) and without write-through. KYTY_NATIVE_TEX_DISK_OVL=0: as before.
	const char* ovl_env = std::getenv("KYTY_NATIVE_TEX_DISK_OVL");
	const char* nd_env  = std::getenv("KYTY_NATIVE_TEX_DISK");
	files.overlapped    = nd_env != nullptr && nd_env[0] == '1' && (ovl_env == nullptr || ovl_env[0] != '0');
	if (files.cap >= (uint64_t {64} << 20u))
		files.handle = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
		                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_NO_BUFFERING |
		                               (files.overlapped ? DWORD {FILE_FLAG_OVERLAPPED} : DWORD {FILE_FLAG_WRITE_THROUGH}),
		                           nullptr);
	if (files.overlapped)
		std::printf("[nativo-ram] almacen abierto en modo solapado, sin escritura directa (lecturas y escrituras a la vez)\n");
	std::printf("RAM17: disk store %s: %ls (up to %llu MiB; error %lu)\n", files.handle != INVALID_HANDLE_VALUE ? "on" : "OFF", path,
	            static_cast<unsigned long long>(files.cap >> 20u),
	            static_cast<unsigned long>(files.handle != INVALID_HANDLE_VALUE ? 0 : GetLastError()));
	std::fflush(stdout);
	return files.handle != INVALID_HANDLE_VALUE;
}

// (mem1 etapa 2) The store's space in video memory (vram-store.h): the same allocator as the file's, its offsets
// carrying VramStore::OffsetBit. Taken first while the store is on and has room; the file after.
inline FileStore& VramSpace() {
	static FileStore space;
	return space;
}
inline uint64_t VramAlloc(uint64_t bytes) {
	if (!VramStore::Active() || bytes == 0 || bytes % FileUnit != 0) return UINT64_MAX;
	auto&           space = VramSpace();
	std::lock_guard lock(space.mutex);
	space.cap = VramStore::Get().cap.load(std::memory_order_relaxed);
	for (auto it = space.holes.begin(); it != space.holes.end(); ++it) {
		if (it->second < bytes) continue;
		const uint64_t offset = it->first, rest = it->second - bytes;
		space.holes.erase(it);
		if (rest != 0) space.holes.emplace(offset + bytes, rest);
		space.used += bytes;
		return offset | VramStore::OffsetBit;
	}
	if (space.end + bytes > space.cap) return UINT64_MAX;
	const uint64_t offset = space.end;
	space.end += bytes;
	space.used += bytes;
	return offset | VramStore::OffsetBit;
}
inline void VramUse(uint64_t* used, uint64_t* size) {
	auto&           space = VramSpace();
	std::lock_guard lock(space.mutex);
	*used = space.used;
	*size = space.end;
}

// `bytes` (a multiple of FileUnit) of the file: its offset, or UINT64_MAX (store off or full).
inline uint64_t FileAlloc(uint64_t bytes) {
	if (const uint64_t in_vram = VramAlloc(bytes); in_vram != UINT64_MAX) return in_vram; // (mem1 etapa 2)
	auto&           files = Files();
	std::lock_guard lock(files.mutex);
	if (bytes == 0 || bytes % FileUnit != 0 || !FileOpenLocked(files)) return UINT64_MAX;
	for (auto it = files.holes.begin(); it != files.holes.end(); ++it) {
		if (it->second < bytes) continue;
		const uint64_t offset = it->first, rest = it->second - bytes;
		files.holes.erase(it);
		if (rest != 0) files.holes.emplace(offset + bytes, rest);
		files.used += bytes;
		return offset;
	}
	if (files.end + bytes > files.cap) return UINT64_MAX;
	const uint64_t offset = files.end;
	files.end += bytes;
	files.used += bytes;
	return offset;
}

inline void FileFree(uint64_t offset, uint64_t bytes) {
	if (bytes == 0) return;
	auto&           files = (offset & VramStore::OffsetBit) != 0 ? VramSpace() : Files(); // (mem1 etapa 2)
	offset &= ~VramStore::OffsetBit;
	std::lock_guard lock(files.mutex);
	files.used -= std::min(files.used, bytes);
	uint64_t begin = offset, end = offset + bytes;
	if (auto next = files.holes.find(end); next != files.holes.end()) {
		end += next->second;
		files.holes.erase(next);
	}
	if (auto prev = files.holes.lower_bound(begin); prev != files.holes.begin()) {
		--prev;
		if (prev->first + prev->second == begin) {
			begin = prev->first;
			files.holes.erase(prev);
		}
	}
	if (end == files.end) files.end = begin; // (the tail of the file free: reused from there; the file keeps its size)
	else files.holes.emplace(begin, end - begin);
}

inline void FileUse(uint64_t* used, uint64_t* size) {
	auto&           files = Files();
	std::lock_guard lock(files.mutex);
	*used = files.used;
	*size = files.end;
}

// This thread's buffer for the file's unbuffered reads and writes (page-aligned, FileChunk bytes).
inline uint8_t* FileStaging() {
	struct Buffer {
		uint8_t* data = nullptr;
		~Buffer() {
			if (data != nullptr) VirtualFree(data, 0, MEM_RELEASE);
		}
	};
	static thread_local Buffer buffer;
	if (buffer.data == nullptr)
		buffer.data = static_cast<uint8_t*>(VirtualAlloc(nullptr, FileChunk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
	return buffer.data;
}

inline bool FileIo(bool write, uint64_t offset, uint8_t* data, uint64_t bytes) {
	if ((offset & VramStore::OffsetBit) != 0) { // (mem1 etapa 2: a GPU copy to or from the store's video memory)
		const VramStore::IoFn io = VramStore::Get().io.load(std::memory_order_acquire);
		return io != nullptr && bytes != 0 && bytes <= FileChunk && io(write, offset & ~VramStore::OffsetBit, data, bytes);
	}
	const HANDLE handle = Files().handle; // (set once, before any offset was given out)
	if (handle == INVALID_HANDLE_VALUE || bytes == 0 || bytes > FileChunk) return false;
	OVERLAPPED at {};
	at.Offset     = static_cast<DWORD>(offset);
	at.OffsetHigh = static_cast<DWORD>(offset >> 32u);
	DWORD done = 0;
	if (Files().overlapped) { // (nativo-ram-3: this thread's own event; the call waits for its own operation only)
		struct Event {
			HANDLE h = nullptr;
			~Event() {
				if (h != nullptr) CloseHandle(h);
			}
		};
		static thread_local Event event;
		if (event.h == nullptr) event.h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
		if (event.h == nullptr) return false;
		ResetEvent(event.h);
		at.hEvent     = event.h;
		const BOOL ok = write ? WriteFile(handle, data, static_cast<DWORD>(bytes), nullptr, &at)
		                      : ReadFile(handle, data, static_cast<DWORD>(bytes), nullptr, &at);
		if (ok == 0 && GetLastError() != ERROR_IO_PENDING) return false;
		if (GetOverlappedResult(handle, &at, &done, TRUE) == 0) return false;
		return done == bytes;
	}
	const BOOL ok = write ? WriteFile(handle, data, static_cast<DWORD>(bytes), &done, &at)
	                      : ReadFile(handle, data, static_cast<DWORD>(bytes), &done, &at);
	return ok != 0 && done == bytes;
}
#else
inline void FileFree(uint64_t /*offset*/, uint64_t /*bytes*/) {}
inline void FileUse(uint64_t* used, uint64_t* size) {
	*used = 0;
	*size = 0;
}
#endif

#if defined(_WIN32)
inline bool Protect(uint64_t address, uint64_t size, DWORD protection) {
	// (ram15) Never with native memory: the pages are the game's own, protected by the page manager and the gate only.
	if (Native()) return false;
	DWORD old = 0;
	return GuestLazyCommit::Protect(reinterpret_cast<void*>(address), size, protection, &old); // (reserved pages first)
}

inline bool IsHostCode(uint64_t rip) {
	HMODULE module = nullptr;
	return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                          reinterpret_cast<LPCWSTR>(rip), &module) != 0;
}

// The return address of the faulting function (one frame unwound from the fault's context; a leaf's at rsp).
inline uint64_t CallerOf(const void* native_context) {
	CONTEXT context    = *static_cast<const CONTEXT*>(native_context);
	DWORD64 image_base = 0;
	auto*   function   = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
	if (function == nullptr) return *reinterpret_cast<const DWORD64*>(context.Rsp);
	void*   handler_data = nullptr;
	DWORD64 establisher  = 0;
	RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context, &handler_data, &establisher, nullptr);
	return context.Rip;
}
#endif

// A guest range [vaddr, vaddr + size) taken, with its backing pieces and image (texture cache, render thread; the
// caller protects, and discards, after).
inline void Add(uint64_t vaddr, uint64_t size, std::vector<Piece> pieces, uint64_t serial, uint64_t image, bool discarded,
                bool gpu = false) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	Range           range;
	range.guest_end = vaddr + size;
	range.pieces    = std::move(pieces);
	range.serial    = serial;
	range.image     = image;
	range.since     = std::chrono::steady_clock::now();
	range.discarded = discarded;
	range.gpu       = gpu;
	for (const auto& piece: range.pieces) {
		state.backing[piece.backing] = vaddr;
		if (piece.backing < state.backing_lo.load(std::memory_order_relaxed)) state.backing_lo.store(piece.backing, std::memory_order_relaxed);
		if (piece.backing + piece.size > state.backing_hi.load(std::memory_order_relaxed))
			state.backing_hi.store(piece.backing + piece.size, std::memory_order_relaxed);
	}
	if (vaddr < state.guest_lo.load(std::memory_order_relaxed)) state.guest_lo.store(vaddr, std::memory_order_relaxed);
	if (vaddr + size > state.guest_hi.load(std::memory_order_relaxed)) state.guest_hi.store(vaddr + size, std::memory_order_relaxed);
	state.guest.emplace(vaddr, std::move(range));
	state.now += size;
	state.peak = std::max(state.peak, state.now);
	state.total += size;
	state.ranges_total++;
	if (gpu) {
		state.gpu_now += size;
		state.gpu_peak = std::max(state.gpu_peak, state.gpu_now);
		state.gpu_total += size;
		state.gpu_ranges_total++;
	}
}

// (Diagnostics, KYTY_GPU_ONLY_OFFLOAD's breakdown) 1: a range over [vaddr, vaddr + size) is taken now; 2: the image was
// touched once (not taken again); 0: neither.
[[nodiscard]] inline int RangeStatus(uint64_t vaddr, uint64_t size, uint64_t serial) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	auto            it = state.guest.upper_bound(vaddr);
	if (it != state.guest.begin() && std::prev(it)->second.guest_end > vaddr) return 1;
	if (it != state.guest.end() && it->first < vaddr + size) return 1;
	return TouchedLocked(state, serial) ? 2 : 0;
}

#if defined(_WIN32)
// The worker thread's drop of range `begin`'s pages (discarded in the backing view, then no access there): unless it was
// touched, restored or dropped since it was taken. Under the state's lock: no touch or restore of it meanwhile.
inline bool DiscardRangeNative(uint64_t begin);
inline bool FileWriteRange(const std::vector<Piece>& pieces, uint64_t* file_off, uint64_t* file_len,
                           bool* zero = nullptr, uint64_t* hash = nullptr); // (ram17, below; ram2: `hash`)
inline bool DiscardRange(uint64_t begin) {
	if (Native()) return DiscardRangeNative(begin);
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	const auto      it = state.guest.find(begin);
	if (it == state.guest.end() || it->second.touched || it->second.restoring || !it->second.discarded) return false;
	// A MiB at a time: a discard holds the process's working set lock over its pages, which every page fault and
	// protection change of the game's threads waits for (whole ranges: frames of 30-60 ms after each scan).
	constexpr uint64_t Chunk = uint64_t {1} << 20u;
	// KYTY_OFFLOAD_PAGEOUT=1 (default on): the pages are not dropped but taken out of the working set (VirtualUnlock):
	// Windows keeps their contents (standby list / page file) and gives them back on any touch by any thread, so no
	// thread ever sees zeros and nothing waits for a restore. The range is then no longer "discarded".
	static const bool pageout = [] {
		const char* v = std::getenv("KYTY_OFFLOAD_PAGEOUT");
		return v == nullptr || v[0] != '0';
	}();
	if (pageout) {
		for (const auto& piece: it->second.pieces) {
			for (uint64_t at = 0; at < piece.size; at += Chunk) {
				// Through the trim runner: no guest map, unmap or protection change while the pages leave (bug check 0x1A
				// otherwise). A busy address space skips the chunk (it stays in RAM), never stalls.
				static thread_local uint64_t t_at = 0, t_size = 0;
				t_at   = piece.backing + at;
				t_size = std::min(Chunk, piece.size - at);
				LocalPlatform::RunTrimQuiet([] { (void)VirtualUnlock(reinterpret_cast<void*>(t_at), t_size); });
				SwitchToThread();
			}
		}
		it->second.discarded = false;
		state.pageout_bytes += it->second.guest_end - begin;
		return true;
	}
	for (const auto& piece: it->second.pieces) {
		// (A discard is refused on pages without access.)
		for (uint64_t at = 0; at < piece.size; at += Chunk) {
			// (ram5) Its result was ignored: the backing view is a view of a shared section, where a discard may be refused
			// [not measured]; then nothing was dropped and the log said "pages dropped". Now counted, and a refused chunk
			// goes by MEM_RESET (no write to the page file) and out of the working set.
			auto*          chunk = reinterpret_cast<void*>(piece.backing + at);
			const uint64_t bytes = std::min(Chunk, piece.size - at);
			const DWORD    error = DiscardVirtualMemory(chunk, bytes);
			if (error == ERROR_SUCCESS) {
				state.discard_ok += bytes;
			} else {
				state.discard_refused += bytes;
				if (state.discard_error == 0) state.discard_error = error;
				if (VirtualAlloc(chunk, bytes, MEM_RESET, PAGE_READWRITE) != nullptr) {
					state.reset_ok += bytes;
					// (Fails "not locked" but takes the pages out of the working set; through the trim runner, as the
					// page-out above: no guest map, unmap or protection change meanwhile.)
					static thread_local uint64_t t_chunk = 0, t_bytes = 0;
					t_chunk = reinterpret_cast<uint64_t>(chunk);
					t_bytes = bytes;
					LocalPlatform::RunTrimQuiet([] { (void)VirtualUnlock(reinterpret_cast<void*>(t_chunk), t_bytes); });
				} else {
					state.reset_refused += bytes;
				}
			}
			// (A page a MiB: still in the working set right after?)
			PSAPI_WORKING_SET_EX_INFORMATION page {};
			page.VirtualAddress = chunk;
			if (K32QueryWorkingSetEx(GetCurrentProcess(), &page, sizeof(page)) != 0) {
				state.sampled_after++;
				if (page.VirtualAttributes.Valid != 0) state.resident_after++;
			}
			SwitchToThread();
		}
		(void)Protect(piece.backing, piece.size, PAGE_NOACCESS);
	}
	return true;
}
#endif

#if defined(_WIN32)
// (ram15, native memory; the offload worker) Range `begin`'s RAM dropped at its guest address (its pages no-access by
// then: the read watchers went on at the scan). Marked as restoring by the worker meanwhile, out of the state's lock: a
// touch waits for the drop to end (then downloads the bytes back), a restore skips it until then. Untouched, not
// restoring and not dropped yet only.
inline void CountLocked(State& state, std::map<uint64_t, Range>::iterator it, Touch touch, uint64_t rip, uint64_t caller); // (ram18: below)
inline bool DiscardRangeNative(uint64_t begin) {
	auto&              state = Get();
	const auto         drop  = state.native_drop.load(std::memory_order_acquire);
	std::vector<Piece> pieces;
	bool               cold = false; // (ram18)
	uint8_t            r20  = 0;     // (ram20)
	uint8_t            rnt  = 0;     // (ram1b)
	{
		std::lock_guard lock(state.mutex);
		const auto      it = state.guest.find(begin);
		if (drop == nullptr || it == state.guest.end() || it->second.touched || it->second.restoring || !it->second.discarded ||
		    it->second.dropped != 0)
			return false;
		it->second.restoring = true;
		it->second.restorer  = std::thread::id {};
		state.restoring_ranges.fetch_add(1, std::memory_order_acq_rel);
		pieces = it->second.pieces;
		cold   = it->second.cold;
		r20    = it->second.r20;
		rnt    = it->second.rnt;
	}
	// (ram18) A cold range is let go (its bytes left in RAM, its watchers removed: Drain) unless its bytes are in the disk
	// store; all zero, it is only reset (a reset page reads as its old bytes or zeros: zeros either way) and let go too.
	const auto let_go = [&](bool failed, uint64_t zero_bytes) {
		std::lock_guard lock(state.mutex);
		const auto      it = state.guest.find(begin);
		if (it == state.guest.end()) return;
		if (it->second.restoring && it->second.restorer == std::thread::id {}) {
			it->second.restoring = false;
			state.restoring_ranges.fetch_sub(1, std::memory_order_acq_rel);
		}
		if (failed) state.cold_failed++;
		if (failed && r20 != 0) state.r20_failed++;
		if (failed && rnt != 0 && rnt <= State::RntKinds) state.rnt_failed++; // (ram1b)
		if (failed && rnt == RntRam2) state.r2_failed++;                        // (ram2)
		if (failed && rnt == RntMesh) state.mesh_failed++;                      // (BUF-NATIVO)
		if (zero_bytes != 0) {
			state.cold_zero++;
			state.cold_zero_bytes += zero_bytes;
			if (r20 != 0) state.r20_zero_bytes += zero_bytes;
			if (rnt != 0 && rnt <= State::RntKinds) state.rnt_zero_bytes += zero_bytes; // (ram1b)
			if (rnt == RntRam2) state.r2_zero_bytes += zero_bytes;                     // (ram2)
		}
		it->second.dropped = 0;
		if (!it->second.touched) {
			it->second.touched = true;
			CountLocked(state, it, Touch::Count, 0, 0);
			state.pending.push_back(begin);
			// (All zero - never written, most likely -: not looked at again for 10 minutes, with the RAM16 retake on.)
			if (zero_bytes != 0 && Ram16RetakeSeconds() != 0)
				state.touched_at[it->second.serial] = std::chrono::steady_clock::now() + std::chrono::minutes(10);
		}
	};
	// (ram17, KYTY_RAM17) Its bytes to the disk store first (still marked as restoring: a touch meanwhile waits).
	// (At most KYTY_RAM17_MAX_MB, default 32, a range: a touch waits for the write, ~1 ms a MiB; larger ones as before.)
	static const uint64_t file_max = [] {
		const char* value = std::getenv("KYTY_RAM17_MAX_MB");
		return (value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : uint64_t {32}) << 20u;
	}();
	uint64_t range_bytes = 0;
	for (const auto& piece: pieces) range_bytes += piece.size;
	uint64_t file_off = 0, file_len = 0;
	bool     zero     = false; // (ram18: a cold range all zeros)
	uint64_t hash     = 0;     // (ram2: its bytes' hash as written)
	if (cold && (!Ram17On() || range_bytes > FileChunk)) {
		let_go(true, 0);
		return true;
	}
	if (Ram17On() && (range_bytes <= file_max || cold)) {
		const auto file_start = std::chrono::steady_clock::now();
		const bool written    = FileWriteRange(pieces, &file_off, &file_len, cold ? &zero : nullptr, rnt == RntRam2 ? &hash : nullptr);
		const auto file_us    = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - file_start).count());
		if (cold && !written && !zero) {
			let_go(true, 0);
			return true;
		}
		std::lock_guard lock(state.mutex);
		if (cold) {
			if (written) {
				state.cold_written++;
				state.cold_written_bytes += file_len;
				state.cold_write_us += file_us;
				if (r20 != 0) state.r20_written_bytes += file_len;
				if (rnt != 0 && rnt <= State::RntKinds) state.rnt_written_bytes += file_len; // (ram1b)
				if (rnt == RntRam2) state.r2_written_bytes += file_len;                     // (ram2)
				if (rnt == RntMesh) state.mesh_written_bytes += file_len;                   // (BUF-NATIVO)
			}
		} else if (written) {
			state.file_writes++;
			state.file_write_bytes += file_len;
			state.file_write_us += file_us;
		} else {
			state.file_write_failed++;
		}
	}
	const auto start   = std::chrono::steady_clock::now();
	uint64_t   dropped = 0, refused = 0;
	for (const auto& piece: pieces) {
		uint64_t not_done = 0;
		dropped += drop(piece.backing, piece.size, &not_done);
		refused += not_done;
	}
	// (A page a MiB: still in the working set right after?)
	uint64_t sampled = 0, resident = 0;
	for (const auto& piece: pieces) {
		for (uint64_t at = 0; at < piece.size; at += uint64_t {1} << 20u) {
			PSAPI_WORKING_SET_EX_INFORMATION page {};
			page.VirtualAddress = reinterpret_cast<void*>(piece.backing + at);
			if (K32QueryWorkingSetEx(GetCurrentProcess(), &page, sizeof(page)) == 0) continue;
			sampled++;
			if (page.VirtualAttributes.Valid != 0) resident++;
		}
	}
	const auto      us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	if (cold && zero) { // (ram18: reset, nothing kept on disk: let go)
		let_go(false, range_bytes);
		return true;
	}
	std::lock_guard lock(state.mutex);
	state.native_drops++;
	state.native_drop_us += us;
	state.native_drop_us_max = std::max(state.native_drop_us_max, us);
	state.native_refused += refused;
	if (!cold) {
		state.sampled_after += sampled;
		state.resident_after += resident;
	}
	const auto it = state.guest.find(begin);
	if (it == state.guest.end()) {
		FileFree(file_off, file_len); // (ram17: gone meanwhile)
		return true;
	}
	if (it->second.restoring && it->second.restorer == std::thread::id {}) {
		it->second.restoring = false;
		state.restoring_ranges.fetch_sub(1, std::memory_order_acq_rel);
	}
	it->second.dropped = dropped;
	// (ram17) The copy on disk kept only for bytes really out of RAM, of a range untouched since (a touch waited).
	if (file_len != 0) {
		if (dropped != 0 && !it->second.touched && it->second.file_len == 0 && file_len == it->second.guest_end - begin) {
			it->second.file_off = file_off;
			it->second.file_len = file_len;
			if (it->second.rnt == RntRam2) { // (ram2: verified when read back)
				it->second.r2_hash   = hash;
				it->second.r2_hashed = true;
			}
		} else {
			FileFree(file_off, file_len);
		}
	}
	if (cold) { // (ram18: its own count; a cold range dropped with no copy kept would lose bytes: never, see above)
		if (it->second.file_len == 0) { // (the reset refused: its bytes still in RAM; let go)
			it->second.dropped = 0;
			if (!it->second.touched) {
				it->second.touched = true;
				CountLocked(state, it, Touch::Count, 0, 0);
				state.pending.push_back(begin);
			}
			return true;
		}
		state.cold_now += it->second.dropped;
		state.cold_peak = std::max(state.cold_peak, state.cold_now);
		if (it->second.r20 != 0) { // (ram20)
			state.r20_now[it->second.r20 - 1] += it->second.dropped;
			uint64_t all = 0;
			for (const auto v: state.r20_now) all += v;
			state.r20_peak = std::max(state.r20_peak, all);
		}
		if (it->second.rnt != 0 && it->second.rnt <= State::RntKinds) { // (ram1b)
			state.rnt_now[it->second.rnt - 1] += it->second.dropped;
			uint64_t all = 0;
			for (const auto v: state.rnt_now) all += v;
			state.rnt_peak = std::max(state.rnt_peak, all);
		}
		if (it->second.rnt == RntRam2) { // (ram2)
			state.r2_now += it->second.dropped;
			state.r2_peak = std::max(state.r2_peak, state.r2_now);
		}
		if (it->second.rnt == RntMesh) { // (BUF-NATIVO)
			state.mesh_now += it->second.dropped;
			state.mesh_peak = std::max(state.mesh_peak, state.mesh_now);
		}
		return true;
	}
	state.native_now += dropped;
	state.native_peak = std::max(state.native_peak, state.native_now);
	state.native_total += dropped;
	return true;
}
#endif

// (11-23, tiron) KYTY_OFFLOAD_MAX_IMAGE_MB=<n> (default 32; 0 = no limit): an image larger than that is never taken out
// of RAM. Bringing one back costs ~1.5-2.4 ms per MiB on the thread that touched it while the game waits: the 320 MiB
// global texture (global_pages.ctxr, 0x358500000) was taken and brought back 7 times in one 15-minute session
// (458-767 ms each), and a 64 MiB one once (155 ms): 4.2 s of the session's 16 s of frames over 100 ms (11-23 00:46,
// 8 GB card simulated, 16 GB RAM). Capped, a restore is at most ~50-75 ms; the RAM kept is that of the big images.
inline uint64_t MaxImageBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_OFFLOAD_MAX_IMAGE_MB");
		const long  mb    = value != nullptr && value[0] != '\0' ? std::strtol(value, nullptr, 10) : 32;
		return mb <= 0 ? uint64_t {0} : static_cast<uint64_t>(mb) << 20u;
	}();
	return bytes;
}

// Not to be taken: a range of it is taken, or the image was touched once (or it is larger than MaxImageBytes).
[[nodiscard]] inline bool Taken(uint64_t vaddr, uint64_t size, uint64_t serial) {
	if (const uint64_t cap = MaxImageBytes(); cap != 0 && size > cap) return true; // (11-23, tiron)
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	if (TouchedLocked(state, serial)) return true;
	auto it = state.guest.upper_bound(vaddr);
	if (it != state.guest.begin() && std::prev(it)->second.guest_end > vaddr) return true;
	if (it != state.guest.end() && it->first < vaddr + size) return true;
	// (ram17) Read back from the disk store a moment ago: held in RAM for KYTY_RAM17_HOLD_S (scans asked, not images).
	if (HeldLocked(state, vaddr, size)) {
		state.held++;
		return true;
	}
	return false;
}

// (state.mutex held) The first range over [vaddr, ...), or end.
inline std::map<uint64_t, Range>::iterator FirstOver(State& state, uint64_t vaddr) {
	auto it = state.guest.upper_bound(vaddr);
	if (it != state.guest.begin() && std::prev(it)->second.guest_end > vaddr) --it;
	return it;
}

// A discarded range not yet given back over [vaddr, vaddr + size) (cheap when none is near).
[[nodiscard]] inline bool Overlaps(uint64_t vaddr, uint64_t size) {
	auto& state = Get();
	if (size == 0 || vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return false;
	std::lock_guard lock(state.mutex);
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it)
		if (it->second.discarded && !it->second.touched) return true;
	return false;
}

// (state.mutex held) A restore by another thread (or the offload worker) runs over [vaddr, vaddr + size).
inline bool RestoringLocked(State& state, uint64_t vaddr, uint64_t size) {
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it)
		if (it->second.restoring && it->second.restorer != std::this_thread::get_id()) return true;
	return false;
}

// Whether a restore by another thread (or the offload worker) runs over [vaddr, vaddr + size).
[[nodiscard]] inline bool IsRestoring(uint64_t vaddr, uint64_t size) {
	auto& state = Get();
	if (size == 0 || vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return false;
	std::lock_guard lock(state.mutex);
	return RestoringLocked(state, vaddr, size);
}

// Until no other thread (nor the offload worker) restores a range over [vaddr, vaddr + size).
inline void WaitRestores(uint64_t vaddr, uint64_t size) {
	while (IsRestoring(vaddr, size)) std::this_thread::yield();
}

// Whether a read of [vaddr, vaddr + size) through the backing view must wait for a restore first (cheap when none
// runs). An image gone (UnregisterImage) is restored asynchronously: its backing pieces are given read-write access at
// once and its bytes land when the GPU's download is done, so a reader of the backing view in between (a buffer upload,
// the async upload worker) read the discarded pages: zeros. At a death and reload under a small video memory budget
// the collector takes hundreds of offloaded textures at once and the reload reads their memory again: dark textures.
[[nodiscard]] inline bool MustWaitBeforeRead(uint64_t vaddr, uint64_t size) {
	auto& state = Get();
	return state.restoring_ranges.load(std::memory_order_acquire) != 0 && IsRestoring(vaddr, size);
}
inline void NoteReadWait(uint64_t size) {
	auto& state = Get();
	state.read_waits.fetch_add(1, std::memory_order_relaxed);
	state.read_wait_bytes.fetch_add(size, std::memory_order_relaxed);
}

// (state.mutex held) Counted as touched by `touch`.
inline void CountLocked(State& state, std::map<uint64_t, Range>::iterator it, Touch touch, uint64_t rip = 0, uint64_t caller = 0) {
	auto&      range = it->second;
	const auto size  = range.guest_end - it->first;
	if (range.cold) { // (ram18: its own numbers; the RAM16 retake rule as a texture's)
		(void)touch, (void)rip, (void)caller;
		state.touched_serials.insert(range.serial);
		if (Ram16RetakeSeconds() != 0) state.touched_at[range.serial] = std::chrono::steady_clock::now();
		state.cold_now -= std::min(state.cold_now, range.dropped);
		if (range.r20 != 0) { // (ram20)
			auto& now = state.r20_now[range.r20 - 1];
			now -= std::min(now, range.dropped);
			state.r20_released++;
		}
		if (range.rnt != 0 && range.rnt <= State::RntKinds) { // (ram1b)
			auto& now = state.rnt_now[range.rnt - 1];
			now -= std::min(now, range.dropped);
			state.rnt_released++;
			const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - range.since).count();
			state.rnt_ages[seconds < 10 ? 0 : seconds < 60 ? 1 : 2]++;
		}
		if (range.rnt == RntMesh) state.mesh_now -= std::min(state.mesh_now, range.dropped); // (BUF-NATIVO)
		if (range.rnt == RntRam2) { // (ram2: its touches by age are counted where it is read back)
			state.r2_now -= std::min(state.r2_now, range.dropped);
			if (touch == Touch::Unmap || touch == Touch::HostOverwrite) state.r2_unmapped += size;
		}
		state.cold_ranges -= std::min<uint64_t>(state.cold_ranges, 1);
		state.cold_released++;
		if (touch == Touch::Unmap || touch == Touch::HostOverwrite) state.cold_unmapped += size;
		range.dropped = 0;
		return;
	}
	state.touches[static_cast<int>(touch)]++;
	state.touch_bytes[static_cast<int>(touch)] += size;
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - range.since).count();
	state.ages[seconds < 1 ? 0 : seconds < 10 ? 1 : seconds < 60 ? 2 : 3]++;
	if (rip != 0) state.rips[rip]++;
	if (caller != 0) state.callers[caller]++;
	state.touched_serials.insert(range.serial);
	if (Ram16RetakeSeconds() != 0) state.touched_at[range.serial] = std::chrono::steady_clock::now(); // (ram16)
	state.now -= std::min(state.now, size);
	if (range.gpu) state.gpu_now -= std::min(state.gpu_now, size);
	// (ram15, native: its RAM comes back, by a download or a drop with the memory.)
	state.native_now -= std::min(state.native_now, range.dropped);
	range.dropped = 0;
}

// (state.mutex held) The touch of range `it`: counted, its views given a protection that lets accesses through (the
// guest's read-only: its write watchers stay) and queued for its read watchers to go.
inline void TouchLocked(State& state, std::map<uint64_t, Range>::iterator it, Touch touch, uint64_t rip = 0, uint64_t caller = 0) {
	auto& range = it->second;
	if (range.touched) return;
	range.touched = true;
	CountLocked(state, it, touch, rip, caller);
#if defined(_WIN32)
	for (const auto& piece: range.pieces) (void)Protect(piece.backing, piece.size, PAGE_READWRITE);
	(void)Protect(it->first, range.guest_end - it->first, PAGE_READONLY);
#endif
	state.pending.push_back(it->first);
}

// (state.mutex held) The queued ranges out of the maps (missed discarded ones stay for their restore: Drain): their guest
// ranges, for the caller to remove the watchers of.
inline std::vector<std::pair<uint64_t, uint64_t>> TakePendingLocked(State& state) {
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	std::vector<uint64_t>                      kept;
	for (const auto begin: state.pending) {
		auto range = state.guest.find(begin);
		if (range == state.guest.end()) continue;
		if (range->second.missed) {
			kept.push_back(begin);
			continue;
		}
		ranges.emplace_back(begin, range->second.guest_end - begin);
		for (const auto& piece: range->second.pieces) state.backing.erase(piece.backing);
		DropFileLocked(range->second); // (ram17)
		state.guest.erase(range);
	}
	state.pending.swap(kept);
	return ranges;
}

inline void RemoveAll(State& state, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	const auto remove = state.remove.load(std::memory_order_acquire);
	if (remove == nullptr) return;
	for (const auto& [begin, size]: ranges) remove(begin, size);
}

// A restore's ranges (TextureCache::RestoreOffloaded, render thread): the discarded ranges over [vaddr, vaddr + size) not
// given back yet (and the missed ones: `late`), marked as restored by this thread and counted as touched by `touch`.
struct Restoring {
	uint64_t           begin = 0, end = 0, serial = 0, image = 0;
	std::vector<Piece> pieces;
};
// `async`: the bytes are written back by the offload worker (EndRestore there).
inline std::vector<Restoring> BeginRestore(uint64_t vaddr, uint64_t size, Touch touch, bool late = false, bool async = false) {
	auto&                  state = Get();
	std::vector<Restoring> ranges;
	std::lock_guard        lock(state.mutex);
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it) {
		auto& range = it->second;
		if (!range.discarded || range.restoring || (range.touched && !(late && range.missed))) continue;
		if (range.file_len != 0) continue; // (ram17: read back from the disk store, FileRestore; an orphan stays)
		if (range.cold) continue;          // (ram18: no image; its bytes come back from the disk store only)
		if (!range.touched && touch != Touch::Count) CountLocked(state, it, touch);
		range.touched   = true;
		range.restoring = true;
		range.restorer  = async ? std::thread::id {} : std::this_thread::get_id();
		state.restoring_ranges.fetch_add(1, std::memory_order_acq_rel);
		ranges.push_back({it->first, range.guest_end, range.serial, range.image, range.pieces});
	}
	return ranges;
}

// (ram15) A restore's downloads counted by why (Touch::Count: after a miss).
inline void NoteRestoreKind(Touch touch, uint64_t bytes) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.restores_by[static_cast<int>(touch)]++;
	state.restore_bytes_by[static_cast<int>(touch)] += bytes;
}

// The restored ranges out of the maps (their bytes back and read watchers removed: other touches go on).
inline void EndRestore(const std::vector<Restoring>& ranges, uint64_t bytes, uint64_t us, uint64_t lost) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	bool            gpu = false;
	for (const auto& restored: ranges) {
		auto range = state.guest.find(restored.begin);
		if (range == state.guest.end()) continue;
		if (range->second.gpu) {
			gpu = true;
			state.gpu_restores++;
			state.gpu_restore_bytes += range->second.guest_end - restored.begin;
		}
		if (range->second.restoring) state.restoring_ranges.fetch_sub(1, std::memory_order_acq_rel);
		for (const auto& piece: range->second.pieces) state.backing.erase(piece.backing);
		DropFileLocked(range->second); // (ram17)
		state.guest.erase(range);
	}
	std::erase_if(state.pending, [&](uint64_t begin) { return state.guest.find(begin) == state.guest.end(); });
	state.restores++;
	state.restore_bytes += bytes;
	state.restore_us += us;
	state.restore_us_max = std::max(state.restore_us_max, us);
	state.lost += lost;
	if (gpu) {
		state.gpu_restore_us += us;
		state.gpu_restore_us_max = std::max(state.gpu_restore_us_max, us);
	}
}

// KYTY_GPU_ONLY_OFFLOAD=1|0|2 (-1: unset; the texture cache turns it on with the low-RAM profile or critical RAM
// pressure; 2: also images whose format no texture check proved: diagnostics).
inline int GpuOnlySetting() {
	static const int setting = [] {
		const char* value = std::getenv("KYTY_GPU_ONLY_OFFLOAD");
		if (value == nullptr || value[0] == '\0') return -1;
		return value[0] == '0' ? 0 : value[0] == '2' ? 2 : 1;
	}();
	return setting;
}

// (ram15) A texture taken by its format's proof, without its own check.
inline void NoteTrusted(uint64_t bytes) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.trusted++;
	state.trusted_bytes += bytes;
}

inline void NoteCheck(uint64_t images, uint64_t bytes, uint64_t failed) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.checked += images;
	state.checked_bytes += bytes;
	state.check_failed += failed;
}

#if defined(_WIN32)
// (ram15) Native memory: this thread's write of restored bytes (its own window must not ask for them again).
inline thread_local bool t_native_writing = false;
// (ram2b) This thread's last fault on a ram2 range: its page, how many times in a row (within a second), when.
inline thread_local uint64_t                              t_r2_page   = 0;
inline thread_local uint32_t                              t_r2_repeat = 0;
inline thread_local std::chrono::steady_clock::time_point t_r2_when {};
constexpr uint32_t                                        Ram2RepeatMax = 8;

// (ram15, native) Restored bytes written over guest [vaddr, vaddr + size) through a write window (memory.cpp
// NativeOffloadWrite); without native memory, a plain copy as before.
inline void WriteBack(uint64_t vaddr, const void* data, uint64_t size) {
	auto& state = Get();
	if (!Native()) {
		std::memcpy(reinterpret_cast<void*>(vaddr), data, size);
		return;
	}
	const auto write = state.native_write.load(std::memory_order_acquire);
	t_native_writing = true;
	const bool ok    = write != nullptr && write(vaddr, data, size);
	t_native_writing = false;
	if (!ok) {
		std::lock_guard lock(state.mutex);
		state.native_write_failed++;
	}
}

// (ram17; the offload worker, its range marked as restoring) The range's guest bytes, piece after piece, to a place of
// the disk store of their size: read through a window (memory.cpp NativeGuardedRead: the pages are no-access by then;
// this thread's window does not ask the offload for them, t_native_writing), a MiB at a time. False: nothing kept.
// (ram18) `zero`: asked for a range of one chunk at most; set when its bytes are all zero (nothing written, false).
// (ram2) `hash`: the bytes written, hashed (RamNativeTex::Ram2Hash, chunk after chunk).
inline bool FileWriteRange(const std::vector<Piece>& pieces, uint64_t* file_off, uint64_t* file_len, bool* zero, uint64_t* hash) {
	*file_off = 0;
	*file_len = 0;
	if (zero != nullptr) *zero = false;
	if (hash != nullptr) *hash = 0;
	auto&      state = Get();
	const auto read  = state.native_read.load(std::memory_order_acquire);
	uint64_t   bytes = 0;
	for (const auto& piece: pieces) {
		if (piece.size % FileUnit != 0 || piece.backing % FileUnit != 0) return false;
		bytes += piece.size;
	}
	uint8_t* staging = FileStaging();
	if (read == nullptr || bytes == 0 || staging == nullptr) return false;
	const uint64_t offset = FileAlloc(bytes);
	if (offset == UINT64_MAX) return false;
	uint64_t at = 0;
	bool     ok = true;
	for (const auto& piece: pieces) {
		for (uint64_t done = 0; ok && done < piece.size;) {
			const uint64_t n = std::min(FileChunk, piece.size - done);
			t_native_writing = true;
			ok               = read(staging, reinterpret_cast<const void*>(piece.backing + done), n);
			t_native_writing = false;
			// (ram18) One chunk all zero: nothing to keep.
			if (ok && zero != nullptr && pieces.size() == 1 && n == bytes) {
				bool all = true;
				for (uint64_t i = 0; i < n && all; i += 8) {
					uint64_t word = 0;
					std::memcpy(&word, staging + i, 8);
					all = word == 0;
				}
				if (all) {
					FileFree(offset, bytes);
					*zero = true;
					return false;
				}
			}
			if (ok && hash != nullptr) *hash = RamNativeTex::Ram2Hash(*hash, staging, n); // (ram2)
			ok               = ok && FileIo(true, offset + at, staging, n);
			done += n;
			at += n;
		}
		if (!ok) break;
	}
	if (!ok) {
		FileFree(offset, bytes);
		return false;
	}
	*file_off = offset;
	*file_len = bytes;
	return true;
}

// (ram17; any thread) The discarded ranges over [vaddr, vaddr + size) with a copy on disk, not touched and not being
// restored: counted as touched by `touch`, read back from the file and written over their guest pages (NativeOffloadWrite)
// by this thread, then queued for their read watchers to go - removed here when `remove_now` (a game thread's fault, the
// render thread, a host write: no lock of the emulator's held), else by the render thread (Drain) or a later fault.
// Ranges over it another thread restores are waited for. A read of the file that fails: the range is a miss (restored
// from its image by the render thread later, as before). Returned: the ranges read back.
inline size_t FileRestoreOnce(uint64_t vaddr, uint64_t size, Touch touch, bool remove_now, bool* waited);
inline size_t FileRestore(uint64_t vaddr, uint64_t size, Touch touch, bool remove_now) {
	// (A range the offload worker was writing and dropping meanwhile has its copy on disk once that is over: looked at
	// again after the wait, so that a thread that cannot wait for the render thread finds it there.)
	const auto   tiron5_begin = std::chrono::steady_clock::now(); // (tiron5: the [tiron3] line's RAM17 restore ms)
	bool         waited = false;
	size_t       first  = FileRestoreOnce(vaddr, size, touch, remove_now, &waited);
	if (waited) {
		const auto w0 = std::chrono::steady_clock::now(); // (nativo-ram-3: the wait for another thread's restore)
		first += FileRestoreOnce(vaddr, size, touch, remove_now, &waited);
		NrCost().wait_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                               std::chrono::steady_clock::now() - w0).count()),
		                           std::memory_order_relaxed);
	}
	if (first != 0) {
		const auto us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		    std::chrono::steady_clock::now() - tiron5_begin).count());
		Tiron4::g_restore_us.fetch_add(us, std::memory_order_relaxed);
		Tiron4::g_restores.fetch_add(first, std::memory_order_relaxed);
		// (nativo-ram-3) A read back for an emulator window (the render thread's or the uploads' own touch of the guest).
		if (touch == Touch::EmulatorGuestRead || touch == Touch::EmulatorGuestWrite) {
			auto& c = NrCost();
			c.emu_n.fetch_add(1, std::memory_order_relaxed);
			c.emu_us.fetch_add(us, std::memory_order_relaxed);
			uint64_t prev = c.emu_max_us.load(std::memory_order_relaxed);
			while (us > prev && !c.emu_max_us.compare_exchange_weak(prev, us, std::memory_order_relaxed)) {}
		}
	}
	return first;
}
inline size_t FileRestoreOnce(uint64_t vaddr, uint64_t size, Touch touch, bool remove_now, bool* waited) {
	*waited    = false;
	auto& state = Get();
	if (!Ram17On() || size == 0 || vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) ||
	    vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return 0;
	struct Job {
		uint64_t                              begin = 0, end = 0, file_off = 0;
		std::vector<Piece>                    pieces;
		std::chrono::steady_clock::time_point since;
		bool                                  cold = false; // (ram18)
		uint8_t                               r20  = 0;     // (ram20)
		uint8_t                               rnt  = 0;     // (ram1b)
		uint64_t                              r2_hash   = 0;     // (ram2)
		bool                                  r2_hashed = false; // (ram2)
	};
	std::vector<Job> jobs;
	bool             wait = false;
	{
		std::lock_guard lock(state.mutex);
		for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it) {
			auto& range = it->second;
			if (range.file_len == 0) continue;
			if (range.restoring) {
				wait = wait || range.restorer != std::this_thread::get_id();
				continue;
			}
			if (range.touched || !range.discarded) continue;
			CountLocked(state, it, touch == Touch::Count ? Touch::EmulatorGuestRead : touch);
			range.touched   = true;
			range.restoring = true;
			range.restorer  = std::this_thread::get_id();
			state.restoring_ranges.fetch_add(1, std::memory_order_acq_rel);
			jobs.push_back({it->first, range.guest_end, range.file_off, range.pieces, range.since, range.cold, range.r20, range.rnt,
			                range.r2_hash, range.r2_hashed});
		}
	}
	const auto write  = state.native_write.load(std::memory_order_acquire);
	const auto forget = state.native_forget.load(std::memory_order_acquire);
	std::vector<uint64_t> r2_slow; // (ram2b: ram2 ranges whose read back took over KYTY_RAM2_SLOW_MS)
	for (const auto& job: jobs) {
		const auto start   = std::chrono::steady_clock::now();
		uint8_t*   staging = FileStaging();
		bool       ok      = staging != nullptr && write != nullptr;
		uint64_t   at      = 0;
		for (const auto& piece: job.pieces) {
			for (uint64_t done = 0; ok && done < piece.size;) {
				const uint64_t n = std::min(FileChunk, piece.size - done);
				const auto     t0 = std::chrono::steady_clock::now(); // (nativo-ram-3: the split)
				ok               = FileIo(false, job.file_off + at, staging, n);
				// (ram18: a cold range has no image to come back from: the read tried twice more.)
				for (int retry = 0; !ok && job.cold && retry < 2; retry++) {
					Sleep(1);
					ok = FileIo(false, job.file_off + at, staging, n);
				}
				const auto t1 = std::chrono::steady_clock::now();
				auto       t2 = t1, t3 = t1;
				if (ok) {
					t_native_writing = true;
					ok               = write(piece.backing + done, staging, n);
					t_native_writing = false;
					t2               = std::chrono::steady_clock::now();
					if (ok && forget != nullptr) forget(piece.backing + done, n);
					t3 = std::chrono::steady_clock::now();
				}
				{
					auto&      c   = NrCost();
					const auto usd = [](auto a, auto b) {
						return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
					};
					c.io_us.fetch_add(usd(t0, t1), std::memory_order_relaxed);
					c.guest_us.fetch_add(usd(t1, t2), std::memory_order_relaxed);
					c.forget_us.fetch_add(usd(t2, t3), std::memory_order_relaxed);
				}
				done += n;
				at += n;
			}
		}
		const auto now = std::chrono::steady_clock::now();
		const auto us  = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
		if (job.rnt == RntRam2 && us > static_cast<uint64_t>(RamNativeTex::Ram2SlowMs()) * 1000u) r2_slow.push_back(job.begin);
		// (ram2) Verification: the guest bytes now (read through this thread's window: the range is still marked as being
		// restored by it, no other touch meanwhile) hashed against those written to the disk store.
		int      r2_check    = 0; // 0 not checked, 1 equal, 2 different
		uint64_t r2_check_us = 0;
		if (ok && job.rnt == RntRam2 && job.r2_hashed) {
			const auto check_start = std::chrono::steady_clock::now();
			const auto read        = state.native_read.load(std::memory_order_acquire);
			uint64_t   h           = 0;
			bool       read_ok     = read != nullptr && staging != nullptr;
			for (const auto& piece: job.pieces) {
				for (uint64_t done = 0; read_ok && done < piece.size;) {
					const uint64_t n = std::min(FileChunk, piece.size - done);
					t_native_writing = true;
					read_ok          = read(staging, reinterpret_cast<const void*>(piece.backing + done), n);
					t_native_writing = false;
					if (read_ok) h = RamNativeTex::Ram2Hash(h, staging, n);
					done += n;
				}
			}
			r2_check    = !read_ok ? 0 : h == job.r2_hash ? 1 : 2;
			r2_check_us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - check_start).count());
			NrCost().check_us.fetch_add(r2_check_us, std::memory_order_relaxed); // (nativo-ram-3)
			if (r2_check == 2)
				std::printf("[ram2] DISCREPANCIA: 0x%llx..0x%llx leido del disco no coincide con lo escrito (hash %016llx, ahora %016llx)\n",
				            static_cast<unsigned long long>(job.begin), static_cast<unsigned long long>(job.end),
				            static_cast<unsigned long long>(job.r2_hash), static_cast<unsigned long long>(h));
		}
		std::lock_guard lock(state.mutex);
		auto            it = state.guest.find(job.begin);
		if (it != state.guest.end()) {
			auto& range = it->second;
			if (range.restoring && range.restorer == std::this_thread::get_id()) {
				range.restoring = false;
				state.restoring_ranges.fetch_sub(1, std::memory_order_acq_rel);
			}
			DropFileLocked(range);
			if (!ok && job.cold) {
				// (ram18) No image to restore it from: its bytes are lost (never seen: 0 file failures in RAM17's runs).
				state.cold_lost++;
				if (job.r20 != 0) state.r20_lost++; // (ram20)
				if (job.rnt != 0 && job.rnt <= State::RntKinds) state.rnt_lost++; // (ram1b)
				if (job.rnt == RntRam2) state.r2_lost++;                          // (ram2)
				if (job.rnt == RntMesh) state.mesh_lost++;                        // (BUF-NATIVO)
				std::printf("[ram18] PERDIDO: lectura del disco fallida de 0x%llx..0x%llx (sin imagen de la que volver)\n",
				            static_cast<unsigned long long>(job.begin), static_cast<unsigned long long>(job.end));
			} else if (!ok) {
				range.missed = true;
				state.missed++;
				state.missed_bytes += job.end - job.begin;
				std::printf("Texture offload: MISSED disk read back of 0x%llx..0x%llx (restored from its image later)\n",
				            static_cast<unsigned long long>(job.begin), static_cast<unsigned long long>(job.end));
			}
			state.pending.push_back(job.begin);
		}
		if (ok && job.cold) {
			state.cold_reads++;
			state.cold_read_bytes += job.end - job.begin;
			state.cold_read_us += us;
			state.cold_read_us_max = std::max(state.cold_read_us_max, us);
			if (job.r20 != 0) { // (ram20)
				state.r20_reads[job.r20 - 1]++;
				state.r20_read_bytes[job.r20 - 1] += job.end - job.begin;
				state.r20_read_us += us;
				state.r20_read_us_max = std::max(state.r20_read_us_max, us);
			}
			if (job.rnt != 0 && job.rnt <= State::RntKinds) { // (ram1b)
				state.rnt_reads[job.rnt - 1]++;
				state.rnt_read_bytes[job.rnt - 1] += job.end - job.begin;
				state.rnt_read_us += us;
				state.rnt_read_us_max = std::max(state.rnt_read_us_max, us);
			}
			if (job.rnt == RntMesh) { // (BUF-NATIVO: una restauracion)
				state.mesh_reads++;
				state.mesh_read_bytes += job.end - job.begin;
				state.mesh_read_us += us;
				state.mesh_read_us_max = std::max(state.mesh_read_us_max, us);
			}
			if (job.rnt == RntRam2) { // (ram2)
				const double seconds = std::chrono::duration<double>(now - job.since).count();
				state.r2_reads++;
				state.r2_read_bytes += job.end - job.begin;
				state.r2_read_us += us;
				state.r2_read_us_max = std::max(state.r2_read_us_max, us);
				state.r2_ages[seconds < 10 ? 0 : seconds < 60 ? 1 : 2]++;
				if (r2_check == 0) state.r2_unchecked++;
				else {
					state.r2_checked++;
					state.r2_checked_bytes += job.end - job.begin;
					state.r2_check_us += r2_check_us;
					if (r2_check == 1) state.r2_equal++;
					else state.r2_discrepancy++;
				}
			}
			if (Ram17HoldSeconds() != 0) state.recent[job.begin] = {job.end, now};
		} else if (ok) {
			state.file_reads++;
			state.file_read_bytes += job.end - job.begin;
			state.file_read_us += us;
			Tiron4::g_ram17_us.fetch_add(us, std::memory_order_relaxed), Tiron4::g_ram17_reads.fetch_add(1, std::memory_order_relaxed); // (tiron4)
			state.file_read_us_max = std::max(state.file_read_us_max, us);
			if (now - job.since < std::chrono::seconds(30)) state.file_quick++;
			if (Ram17HoldSeconds() != 0) state.recent[job.begin] = {job.end, now};
		} else {
			state.file_read_failed++;
		}
	}
	for (const auto begin: r2_slow) { // (ram2b)
		RamNativeTex::Ram2Guards().slow.fetch_add(1, std::memory_order_relaxed);
		RamNativeTex::Ram2Block(begin, "lectura del disco lenta", "");
	}
	if (remove_now && !jobs.empty()) {
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		{
			std::lock_guard lock(state.mutex);
			ranges = TakePendingLocked(state);
		}
		RemoveAll(state, ranges);
	}
	if (wait) {
		WaitRestores(vaddr, size);
		*waited = true;
	}
	return jobs.size();
}

// (ram17; render thread, an image going: UnregisterImage) Its ranges with a copy on disk are left as they are - out of
// RAM, watched - with no download: orphans, read back from the file at their first touch. Returned: bytes left so.
inline uint64_t OrphanFileBacked(uint64_t vaddr, uint64_t size) {
	if (!Ram17On() || size == 0) return 0;
	auto&           state = Get();
	uint64_t        bytes = 0;
	std::lock_guard lock(state.mutex);
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it) {
		auto& range = it->second;
		if (range.file_len == 0 || range.touched || range.restoring || !range.discarded) continue;
		if (!range.orphan) {
			range.orphan = true;
			state.orphaned++;
			state.orphaned_bytes += range.guest_end - it->first;
		}
		range.image = 0;
		bytes += range.guest_end - it->first;
	}
	return bytes;
}

// (ram15, native; render thread, first thing of a restore) The taken ranges over [vaddr, vaddr + size) whose bytes are
// still in RAM (not dropped yet: no download needed) touched by `touch`, and every range waiting for its read watchers
// to go: out of the maps. Returned: their guest ranges, whose read watchers the caller removes.
inline std::vector<std::pair<uint64_t, uint64_t>> ReleaseIntactNative(uint64_t vaddr, uint64_t size, Touch touch) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it) {
		auto& range = it->second;
		if (range.restoring || range.touched || range.dropped != 0 || range.file_len != 0) continue;
		TouchLocked(state, it, touch == Touch::Count ? Touch::EmulatorGuestRead : touch);
	}
	return TakePendingLocked(state);
}

// (ram15) The probe handler with native memory (OnFault's first step): a game thread's or the emulator's own fault on a
// taken range's guest pages (no-access by its read watchers).
inline bool OnFaultNative(uint64_t address, uint64_t rip, const void* native_context, bool write) {
	auto& state = Get();
	if (address < state.guest_lo.load(std::memory_order_relaxed) || address >= state.guest_hi.load(std::memory_order_relaxed))
		return false;
	enum class Act { Wait, Download, Remove, Open, File, Escape } act = Act::Wait;
	std::vector<std::pair<uint64_t, uint64_t>> now;
	uint64_t                                   begin = 0, size = 0;
	Touch                                      touch = Touch::GameRead;
	bool                                       host_code = false;
	bool                                       ram2      = false; // (ram2b)
	const auto                                 r2_start  = std::chrono::steady_clock::now();
	{
		std::unique_lock lock(state.mutex);
		auto             found = state.guest.upper_bound(address);
		if (found == state.guest.begin()) return false;
		--found;
		auto& range = found->second;
		if (address >= range.guest_end) return false;
		begin = found->first;
		size  = range.guest_end - begin;
		const bool host = IsHostCode(rip);
		host_code       = host;
		touch           = host ? (write ? Touch::EmulatorGuestWrite : Touch::EmulatorGuestRead) : (write ? Touch::GameWrite : Touch::GameRead);
		if (range.rnt == RntMesh && !range.touched) (host ? state.mesh_faults_emu : state.mesh_faults_game)++; // (BUF-NATIVO)
		const auto can_wait = state.can_wait.load(std::memory_order_acquire);
		const bool can = state.restore.load(std::memory_order_acquire) != nullptr && can_wait != nullptr && can_wait();
		if (range.rnt == RntRam2 || range.rnt == RntMesh) {
			// (ram2b) A ram2 range (cold: only the disk store gives it back; the GPU path never does - a "Download" or a miss
			// would leave the page protected and the fault coming back for ever). The same page faulting again and again on
			// this thread (more than Ram2RepeatMax times within a second): let go whatever its state (Escape).
			// (BUF-NATIVO fix) Las mallas son frias igual (image 0): por el camino general un rango MISSED iba a Download
			// (GPU) y el hilo del juego fallaba en bucle para siempre (cuelgue 20261124-073717-127).
			ram2 = range.rnt == RntRam2;
			const uint64_t page = address >> 12u;
			if (page == t_r2_page && r2_start - t_r2_when < std::chrono::seconds(1)) t_r2_repeat++;
			else t_r2_page = page, t_r2_repeat = 1;
			t_r2_when = r2_start;
			if (t_r2_repeat > Ram2RepeatMax) {
				act = Act::Escape;
			} else if (range.rnt == RntMesh) {
				// (BUF-NATIVO fix2) Una malla: SIEMPRE suelta en el primer fallo (leida de vuelta si esta en el almacen, tocada,
				// sus vigilantes fuera, abierta) y fuera del rastreo. Los caminos Wait/Remove/Open podian volver sin
				// desproteger (pendientes ya tomados por otro hilo) y el hilo del juego fallaba sin fin (20261124-074903-128).
				if (range.restoring && range.restorer == std::this_thread::get_id()) return false;
				act = Act::Escape;
			} else if (range.restoring) {
				if (range.restorer == std::this_thread::get_id()) return false; // (not ours: this thread restores it)
				act = Act::Wait; // (the RAM18 thread writing it to the disk store, or another thread reading it back)
			} else if (range.touched) {
				range.missed = false; // (never a miss: its bytes are in RAM or were read back)
				if (!host) act = Act::Remove, now = TakePendingLocked(state);
				else act = Act::Open;
			} else if (range.file_len != 0) {
				act = Act::File;
			} else {
				// Not dropped yet: its bytes are there. Let go.
				TouchLocked(state, found, touch, host ? rip : 0, host ? CallerOf(native_context) : 0);
				if (!host) act = Act::Remove, now = TakePendingLocked(state);
				else act = Act::Open;
			}
		} else if (range.restoring) {
			if (range.restorer == std::this_thread::get_id()) return false; // (not ours: this thread restores it)
			act = can ? Act::Download : Act::Wait; // (a restore waits for the other one first, its GPU work submitted)
		} else if (range.touched) {
			// Its read watchers not gone yet (Drain), or missed and not restored yet.
			if (range.missed) act = can ? Act::Download : Act::Open;
			else if (!host) act = Act::Remove, now = TakePendingLocked(state);
			else act = can ? Act::Download : Act::Open;
			if (range.missed) touch = Touch::Count; // (a late restore)
		} else if (range.file_len != 0) {
			act = Act::File; // (ram17: read back from the disk store by this thread, whatever it is)
		} else if (range.dropped == 0 || range.cold) {
			// Its bytes still there (the worker has not dropped them): no download. (ram18: a cold range never has one.)
			if (!host) {
				TouchLocked(state, found, touch);
				now = TakePendingLocked(state);
				act = Act::Remove;
			} else if (can) {
				act = Act::Download;
			} else {
				TouchLocked(state, found, touch, rip, CallerOf(native_context));
				act = Act::Open;
			}
		} else if (can) {
			act = Act::Download;
		} else {
			range.missed = true;
			state.missed++;
			state.missed_bytes += size;
			std::printf("Texture offload: MISSED %s of 0x%llx..0x%llx (native: its dropped bytes read as they are until restored) "
			            "at rip 0x%llx\n",
			            TouchName(touch), static_cast<unsigned long long>(begin), static_cast<unsigned long long>(begin + size),
			            static_cast<unsigned long long>(rip));
			TouchLocked(state, found, touch, host ? rip : 0, host ? CallerOf(native_context) : 0);
			act = Act::Open;
		}
	}
	switch (act) {
		case Act::Wait: WaitRestores(begin, size); break;
		case Act::Download: state.restore.load(std::memory_order_acquire)(begin, size, touch); break;
		case Act::Remove: RemoveAll(state, now); break; // (a game thread: no lock of the emulator's held)
		case Act::File: {
			// (ram17) A game thread removes the read watchers itself (no lock of the emulator's held); the emulator's own
			// code gets the pages opened for its access, the watchers going on the render thread (Drain), as a miss's.
			if (FileRestore(begin, size, touch, !host_code) != 0 && host_code) {
				const auto open = state.native_open.load(std::memory_order_acquire);
				if (open != nullptr && open(begin, size, write)) {
					std::lock_guard lock(state.mutex);
					state.native_opened++;
				}
			}
			break;
		}
		case Act::Open: {
			const auto open = state.native_open.load(std::memory_order_acquire);
			if (open != nullptr && open(begin, size, write)) {
				std::lock_guard lock(state.mutex);
				state.native_opened++;
			}
			break;
		}
		case Act::Escape: {
			// (ram2b) Its bytes back from the disk store if they are there, then let go and opened, whatever its state; its
			// window out of ram2.
			t_r2_repeat = 0;
			if (!ram2) WaitRestores(begin, size); // (BUF-NATIVO fix2: el hilo RAM18 escribiendola: su copia en el almacen antes)
			(void)FileRestore(begin, size, touch, !host_code);
			{
				std::lock_guard lock(state.mutex);
				const auto      it = state.guest.find(begin);
				if (it != state.guest.end() && (it->second.rnt == RntRam2 || it->second.rnt == RntMesh) && !it->second.restoring) {
					if (!it->second.touched) TouchLocked(state, it, touch);
					it->second.missed = false;
					now               = TakePendingLocked(state);
				}
			}
			RemoveAll(state, now);
			if (!ram2) { // (BUF-NATIVO fix2: la malla nunca vuelve a tomarse; sus paginas con la proteccion del juego)
				std::lock_guard lock(state.mutex);
				state.touched_serials.insert(ColdSerial(begin));
			}
			const auto open = state.native_open.load(std::memory_order_acquire);
			if (open != nullptr) (void)open(begin, size, write);
			if (!ram2) break; // (BUF-NATIVO: una malla no tiene ventana de ram2)
			auto& guard = RamNativeTex::Ram2Guards();
			guard.repeats.fetch_add(1, std::memory_order_relaxed);
			guard.escapes.fetch_add(1, std::memory_order_relaxed);
			char label[24] {};
			(void)Loader::DemonsSoulsMemMap::LabelAt(address, label, sizeof(label));
			RamNativeTex::Ram2Block(address, "fallo repetido en la misma pagina", label);
			break;
		}
	}
	if (ram2) { // (ram2b) A fault that took too long: its window out of ram2.
		const auto us = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - r2_start).count());
		auto& guard = RamNativeTex::Ram2Guards();
		if (us > guard.slow_us_max.load(std::memory_order_relaxed)) guard.slow_us_max.store(us, std::memory_order_relaxed);
		if (us > static_cast<uint64_t>(RamNativeTex::Ram2SlowMs()) * 1000u) {
			guard.slow.fetch_add(1, std::memory_order_relaxed);
			char label[24] {};
			(void)Loader::DemonsSoulsMemMap::LabelAt(address, label, sizeof(label));
			RamNativeTex::Ram2Block(address, "fallo lento", label);
		}
	}
	return true;
}

// (ram17, diagnostics of a miss) This thread's name (SetThreadDescription), whether it held the gate, and its first
// callers as offsets in the exe (0x... outside it).
inline void ThreadWho(char* out, size_t size, bool gate_held) {
	char     name[64] = "?";
	wchar_t* wide     = nullptr;
	if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &wide)) && wide != nullptr) {
		if (wide[0] != L'\0') (void)WideCharToMultiByte(CP_UTF8, 0, wide, -1, name, sizeof(name), nullptr, nullptr);
		LocalFree(wide);
	}
	void*          frames[6] {};
	const USHORT   count = RtlCaptureStackBackTrace(2, 6, frames, nullptr);
	const auto     exe   = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	int            used  = std::snprintf(out, size, "hilo %s (id %lu), puerta %s, llamadas", name,
	                                     static_cast<unsigned long>(GetCurrentThreadId()), gate_held ? "tomada" : "libre");
	for (USHORT i = 0; i < count && used > 0 && static_cast<size_t>(used) < size; i++) {
		const auto at = reinterpret_cast<uint64_t>(frames[i]);
		used += std::snprintf(out + used, size - static_cast<size_t>(used), " %s0x%llx", at >= exe && at - exe < (uint64_t {256} << 20u) ? "exe+" : "",
		                      static_cast<unsigned long long>(at >= exe && at - exe < (uint64_t {256} << 20u) ? at - exe : at));
	}
}

// (ram15, native) The hook of every window of the emulator's over guest memory (NativeGuardedAccess, memory.cpp): its
// copies do not fault, so the dropped bytes they would read (or partly write over) come back first, the window's thread
// waiting for the render thread. One that cannot wait (the upload worker, deferred GPU work, a recorder, a window inside
// a window) is a miss: counted; the window copies what is there, the range restored after (Drain).
inline void NativeBeforeWindow(uint64_t vaddr, uint64_t bytes, bool writes, bool gate_held) {
	Mem1::NoteWindow(vaddr, bytes); // (mem1, KYTY_MEM1=1: when the emulator last read or wrote VirtualMemo; off: a bool)
	auto& state = Get();
	state.window_calls.fetch_add(1, std::memory_order_relaxed);
	if (t_native_writing || bytes == 0 || vaddr + bytes <= state.guest_lo.load(std::memory_order_relaxed) ||
	    vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return;
	const auto start = std::chrono::steady_clock::now();
	state.window_lookups.fetch_add(1, std::memory_order_relaxed);
	bool any = false;
	{
		std::lock_guard lock(state.mutex);
		for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + bytes && !any; ++it)
			any = !(it->second.restoring && it->second.restorer == std::this_thread::get_id()) &&
			      !(it->second.touched && !it->second.missed && it->second.dropped == 0);
	}
	state.window_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()),
	                          std::memory_order_relaxed);
	if (!any) return;
	state.window_hits.fetch_add(1, std::memory_order_relaxed);
	const Touch touch    = writes ? Touch::EmulatorGuestWrite : Touch::EmulatorGuestRead;
	// (ram17) The ranges with a copy on disk read back by this thread first, whatever it is (never a miss); then only
	// what is left asks the render thread.
	if (Ram17On()) {
		// (ram18) A cold range being written and dropped by the RAM18 thread (a MiB at most): waited for, then read back
		// like the others - unless this thread holds the gate, which that thread's read window needs.
		const auto cold_busy = [&] {
			std::lock_guard lock(state.mutex);
			for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + bytes; ++it)
				if (it->second.cold && it->second.restoring && it->second.restorer != std::this_thread::get_id()) return true;
			return false;
		};
		if (Ram18On() && !gate_held) {
			const auto w0 = std::chrono::steady_clock::now(); // (nativo-ram-3: the wait for a cold range being written)
			while (cold_busy()) std::this_thread::yield();
			NrCost().wait_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                               std::chrono::steady_clock::now() - w0).count()),
			                           std::memory_order_relaxed);
		}
		(void)FileRestore(vaddr, bytes, touch, false);
		any = false;
		std::lock_guard lock(state.mutex);
		for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + bytes && !any; ++it)
			any = !(it->second.restoring && it->second.restorer == std::this_thread::get_id()) &&
			      !(it->second.touched && !it->second.missed && it->second.dropped == 0);
		if (!any) return;
	}
	const auto  restore  = state.restore.load(std::memory_order_acquire);
	const auto  can_wait = state.can_wait.load(std::memory_order_acquire);
	if (!gate_held && restore != nullptr && can_wait != nullptr && can_wait()) {
		restore(vaddr, bytes, touch);
		return;
	}
	state.window_missed.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard lock(state.mutex);
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + bytes; ++it) {
		auto& range = it->second;
		if (range.restoring || range.touched) continue;
		if (range.dropped != 0) {
			range.missed = true;
			state.missed++;
			state.missed_bytes += range.guest_end - it->first;
			// (ram17) Who: the thread's name, whether it held the gate, and its callers (as exe offsets).
			char who[192];
			ThreadWho(who, sizeof(who), gate_held);
			std::printf("Texture offload: MISSED %s window over 0x%llx..0x%llx (native: copied as it is until restored; %s)\n",
			            writes ? "write" : "read", static_cast<unsigned long long>(it->first),
			            static_cast<unsigned long long>(range.guest_end), who);
		}
		TouchLocked(state, it, touch);
	}
}

// The exception filter's probe handler (any thread): a touch of a taken range, either view.
inline bool OnFault(uint64_t address, uint64_t rip, const void* native_context, bool write) {
	if (Native()) return OnFaultNative(address, rip, native_context, write);
	auto&      state      = Get();
	const bool in_guest   = address >= state.guest_lo.load(std::memory_order_relaxed) && address < state.guest_hi.load(std::memory_order_relaxed);
	const bool in_backing = address >= state.backing_lo.load(std::memory_order_relaxed) && address < state.backing_hi.load(std::memory_order_relaxed);
	if (!in_guest && !in_backing) return false;
	// A fault raised while its range was still taken, served after another thread's touch gave the pages back (and the
	// range left the maps): the access goes again. (Only this protects the backing view.)
	const auto stale = [address] {
		MEMORY_BASIC_INFORMATION info {};
		return VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == sizeof(info) && info.Protect == PAGE_READWRITE;
	};
	std::vector<std::pair<uint64_t, uint64_t>> now;
	uint64_t restore_begin = 0, restore_size = 0, wait_begin = 0, wait_size = 0;
	Touch    restore_touch = Touch::GameRead;
	{
		std::unique_lock lock(state.mutex);
		std::map<uint64_t, Range>::iterator it = state.guest.end();
		bool                                backing_hit = false;
		if (in_guest) {
			auto found = state.guest.upper_bound(address);
			if (found != state.guest.begin() && address < std::prev(found)->second.guest_end) it = std::prev(found);
		}
		if (it == state.guest.end() && in_backing) {
			auto piece = state.backing.upper_bound(address);
			if (piece == state.backing.begin()) return stale();
			--piece;
			auto range = state.guest.find(piece->second);
			if (range == state.guest.end()) return stale();
			const bool inside = std::any_of(range->second.pieces.begin(), range->second.pieces.end(), [&](const Piece& p) {
				return address >= p.backing && address < p.backing + p.size;
			});
			if (!inside) return stale();
			it          = range;
			backing_hit = true;
		}
		if (it == state.guest.end()) return false;
		auto&      range = it->second;
		const bool host  = backing_hit || IsHostCode(rip);
		if (range.restoring && range.restorer != std::this_thread::get_id()) {
			// Being restored (the render thread or the offload worker): the access goes again after.
			wait_begin = it->first;
			wait_size  = range.guest_end - it->first;
		} else if (range.touched) {
			// (Its protection came back before its watchers went, or this thread restores it.)
			if (host) {
				if (backing_hit) {
					for (const auto& piece: range.pieces) (void)Protect(piece.backing, piece.size, PAGE_READWRITE);
				} else {
					(void)Protect(it->first, range.guest_end - it->first, PAGE_READONLY);
				}
				return true;
			}
			const auto can_wait = state.can_wait.load(std::memory_order_acquire);
			if (range.missed && state.restore.load(std::memory_order_acquire) != nullptr && can_wait != nullptr && can_wait()) {
				// (A missed one keeps its watchers until its restore: it comes back now.)
				restore_begin = it->first;
				restore_size  = range.guest_end - it->first;
				restore_touch = Touch::Count;
			} else {
				now = TakePendingLocked(state);
			}
		} else {
			const Touch touch = backing_hit ? (write ? Touch::BackingWrite : Touch::BackingRead)
			                    : host      ? (write ? Touch::EmulatorGuestWrite : Touch::EmulatorGuestRead)
			                                : (write ? Touch::GameWrite : Touch::GameRead);
			const auto  can_wait = state.can_wait.load(std::memory_order_acquire);
			if (range.discarded && state.restore.load(std::memory_order_acquire) != nullptr && can_wait != nullptr && can_wait()) {
				restore_begin = it->first;
				restore_size  = range.guest_end - it->first;
				restore_touch = touch;
			} else {
				if (range.discarded) {
					range.missed = true;
					state.missed++;
					state.missed_bytes += range.guest_end - it->first;
					std::printf("Texture offload: MISSED %s of 0x%llx..0x%llx (zeros until restored) at rip 0x%llx\n", TouchName(touch),
					            static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(range.guest_end),
					            static_cast<unsigned long long>(rip));
				}
				TouchLocked(state, it, touch, host ? rip : 0, host ? CallerOf(native_context) : 0);
				if (host) return true;
				now = TakePendingLocked(state);
			}
		}
	}
	if (wait_size != 0) {
		WaitRestores(wait_begin, wait_size);
		return true;
	}
	if (restore_size != 0) {
		state.restore.load(std::memory_order_acquire)(restore_begin, restore_size, restore_touch);
		return true;
	}
	RemoveAll(state, now); // (a game thread: no lock of the emulator's held)
	return true;
}
#else
inline size_t FileRestore(uint64_t /*vaddr*/, uint64_t /*size*/, Touch /*touch*/, bool /*remove_now*/) {
	return 0;
}
inline uint64_t OrphanFileBacked(uint64_t /*vaddr*/, uint64_t /*size*/) {
	return 0;
}
#endif

// (ram3) KYTY_OFFLOAD_UNMAP_SPLIT=1 (off by default; only matters with KYTY_OFFLOAD_PAGEOUT=0, where a taken range's
// pages are really dropped): an unmap over PART of a discarded range no longer drops the whole range. Before, the part
// still mapped lost its protection with its bytes gone (zeros, never restored): Demon's Souls unmaps a streamed
// texture's largest mips (decommit, eboot+0x9014f4) while the smaller ones stay mapped, and the image the unmap
// deletes was not asked for them back (UnregisterImage finds no range left). Now only the unmapped part is dropped;
// the rest stays taken (same image, serial and pieces, cut at the unmap's bounds), so the dying image gives it back
// (its ImageGone restore) or it stays offloaded under a surviving one.
inline bool UnmapSplit() {
	static const bool on = [] {
		const char* v       = std::getenv("KYTY_OFFLOAD_UNMAP_SPLIT");
		const char* pageout = std::getenv("KYTY_OFFLOAD_PAGEOUT");
		// (ram15) Native memory drops the pages always: on unless KYTY_OFFLOAD_UNMAP_SPLIT=0 (the part still mapped would
		// keep no bytes and no restore).
		if (Native()) return !(v != nullptr && v[0] == '0');
		// (Only where pages are dropped: paged out, a range keeps its bytes and the old path is right.)
		return v != nullptr && v[0] == '1' && pageout != nullptr && pageout[0] == '0';
	}();
	return on;
}

// (state.mutex held; ram3, UnmapSplit) Range `begin`, discarded and not touched, loses [vaddr, vaddr + size): what is
// left of it before and after stays as ranges of their own. Returns the guest part dropped (its read watchers go).
inline std::pair<uint64_t, uint64_t> SplitLocked(State& state, uint64_t begin, uint64_t vaddr, uint64_t size) {
	const auto found = state.guest.find(begin);
	if (found == state.guest.end()) return {0, 0};
	Range          whole = std::move(found->second);
	const uint64_t end   = whole.guest_end;
	const uint64_t whole_size = end > begin ? end - begin : 1;
	for (const auto& piece: whole.pieces) state.backing.erase(piece.backing);
	state.guest.erase(found);
	const uint64_t cut_begin = std::max(begin, vaddr), cut_end = std::min(end, vaddr + size);
	Range          head, tail;
	for (Range* part: {&head, &tail}) {
		part->serial    = whole.serial;
		part->image     = whole.image;
		part->since     = whole.since;
		part->discarded = whole.discarded;
		part->gpu       = whole.gpu;
		part->orphan    = whole.orphan;
		part->cold      = whole.cold; // (ram18)
		part->r20       = whole.r20;  // (ram20)
		part->rnt       = whole.rnt;  // (ram1b)
		part->r2_hashed = false;      // (ram2: a part's bytes are not the hash's: not verified)
	}
	uint64_t at = begin;
	for (const auto& piece: whole.pieces) {
		const uint64_t piece_begin = at, piece_end = at + piece.size;
		if (piece_begin < cut_begin) head.pieces.push_back({piece.backing, std::min(piece_end, cut_begin) - piece_begin});
#if defined(_WIN32)
		const uint64_t gone_begin = std::max(piece_begin, cut_begin), gone_end = std::min(piece_end, cut_end);
		if (gone_begin < gone_end) (void)Protect(piece.backing + (gone_begin - piece_begin), gone_end - gone_begin, PAGE_READWRITE);
#endif
		if (piece_end > cut_end) {
			const uint64_t from = std::max(piece_begin, cut_end);
			tail.pieces.push_back({piece.backing + (from - piece_begin), piece_end - from});
		}
		at = piece_end;
	}
	const bool keep_head = cut_begin > begin && !head.pieces.empty(), keep_tail = end > cut_end && !tail.pieces.empty();
	// (ram15, native: the dropped bytes shared out by size; the unmapped part's leave the count.)
	const auto share = [&](uint64_t bytes) {
		return whole.dropped / whole_size * bytes + whole.dropped % whole_size * bytes / whole_size;
	};
	if (keep_head) head.dropped = share(cut_begin - begin);
	if (keep_tail) tail.dropped = share(end - cut_end);
	// (ram17) Its copy on disk shared out the same way (guest byte begin + i at file_off + i); the unmapped part's freed.
	if (whole.file_len == end - begin && end > begin) {
		const uint64_t head_bytes = cut_begin - begin, gone_bytes = cut_end > cut_begin ? cut_end - cut_begin : 0;
		const uint64_t tail_bytes = end - std::max(cut_end, cut_begin);
		if (keep_head) head.file_off = whole.file_off, head.file_len = head_bytes;
		else FileFree(whole.file_off, head_bytes);
		FileFree(whole.file_off + head_bytes, gone_bytes);
		if (keep_tail) tail.file_off = whole.file_off + (end - begin - tail_bytes), tail.file_len = tail_bytes;
		else FileFree(whole.file_off + (end - begin - tail_bytes), tail_bytes);
	} else {
		FileFree(whole.file_off, whole.file_len);
	}
	const uint64_t lost_dropped = whole.dropped - std::min(whole.dropped, head.dropped + tail.dropped);
	if (whole.cold) { // (ram18: its own count)
		state.cold_now -= std::min(state.cold_now, lost_dropped);
		if (whole.r20 != 0) state.r20_now[whole.r20 - 1] -= std::min(state.r20_now[whole.r20 - 1], lost_dropped); // (ram20)
		if (whole.rnt != 0 && whole.rnt <= State::RntKinds) // (ram1b)
			state.rnt_now[whole.rnt - 1] -= std::min(state.rnt_now[whole.rnt - 1], lost_dropped);
		if (whole.rnt == RntMesh) state.mesh_now -= std::min(state.mesh_now, lost_dropped); // (BUF-NATIVO)
		if (whole.rnt == RntRam2) { // (ram2)
			state.r2_now -= std::min(state.r2_now, lost_dropped);
			state.r2_unmapped += cut_end > cut_begin ? cut_end - cut_begin : 0;
		}
		state.cold_ranges += (keep_head ? 1 : 0) + (keep_tail ? 1 : 0);
		state.cold_ranges -= std::min<uint64_t>(state.cold_ranges, 1);
		state.cold_unmapped += cut_end > cut_begin ? cut_end - cut_begin : 0;
	} else {
		state.native_now -= std::min(state.native_now, lost_dropped);
	}
	if (keep_head) {
		head.guest_end = cut_begin;
		for (const auto& piece: head.pieces) state.backing[piece.backing] = begin;
		state.guest.emplace(begin, std::move(head));
	}
	if (keep_tail) {
		tail.guest_end = end;
		for (const auto& piece: tail.pieces) state.backing[piece.backing] = cut_end;
		state.guest.emplace(cut_end, std::move(tail));
	}
	const uint64_t gone = cut_end > cut_begin ? cut_end - cut_begin : 0;
	if (whole.cold) return {cut_begin, gone}; // (ram18: not a texture's numbers)
	state.touches[static_cast<int>(Touch::Unmap)]++;
	state.touch_bytes[static_cast<int>(Touch::Unmap)] += gone;
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - whole.since).count();
	state.ages[seconds < 1 ? 0 : seconds < 10 ? 1 : seconds < 60 ? 2 : 3]++;
	state.now -= std::min(state.now, gone);
	if (whole.gpu) state.gpu_now -= std::min(state.gpu_now, gone);
	state.unmap_splits++;
	return {cut_begin, gone};
}

// A host write (InvalidateMemory: file reads, network; `over_all`: it writes all of [vaddr, vaddr + size)) or an unmap
// over [vaddr, vaddr + size): the taken ranges there touched. A discarded one a write covers part of comes back first
// (the restore); the others are dropped. Normal context: their read watchers go at once.
inline void OnHostAccess(uint64_t vaddr, uint64_t size, bool unmap, bool over_all) {
	if (!Enabled() || size == 0) return;
	auto& state = Get();
	if (vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed)) return;
	WaitRestores(vaddr, size);
	std::vector<std::pair<uint64_t, uint64_t>> ranges, restore;
	std::vector<uint64_t>                      ram2_hits; // (ram2b: ram2 ranges a host write meets)
	std::vector<uint64_t>                      mesh_hits; // (BUF-NATIVO fix: mallas frias que toca una escritura del emulador)
	{
		std::lock_guard lock(state.mutex);
		std::vector<uint64_t> cut; // (ram3, UnmapSplit: discarded ranges the unmap covers a part of)
		for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it) {
			if (!unmap && it->second.rnt == RntRam2) ram2_hits.push_back(it->first); // (ram2b)
			if (!unmap && it->second.rnt == RntMesh) mesh_hits.push_back(it->first); // (BUF-NATIVO fix: fria como ram2)
			if (it->second.restoring) continue; // (this thread's own restore)
			const bool covered = over_all && it->first >= vaddr && it->second.guest_end <= vaddr + size;
			if (it->second.touched) {
				// A missed one: dropped with the memory, or written over whole.
				if (it->second.missed && (unmap || covered)) {
					it->second.missed = false;
					state.pending.push_back(it->first);
				}
				continue;
			}
			if (unmap && it->second.discarded && UnmapSplit() && !(it->first >= vaddr && it->second.guest_end <= vaddr + size)) {
				cut.push_back(it->first);
				continue;
			}
			if (!unmap && !covered && it->second.discarded) {
				restore.emplace_back(it->first, it->second.guest_end - it->first);
				continue;
			}
			TouchLocked(state, it, unmap ? Touch::Unmap : covered ? Touch::HostOverwrite : Touch::HostWrite);
		}
		ranges = TakePendingLocked(state);
		for (const auto begin: cut) {
			const auto gone = SplitLocked(state, begin, vaddr, size);
			if (gone.second != 0) ranges.push_back(gone);
		}
	}
	RemoveAll(state, ranges);
	const auto restore_fn = state.restore.load(std::memory_order_acquire);
	const auto can_wait   = state.can_wait.load(std::memory_order_acquire);
	for (const auto& [begin, bytes]: restore) {
		(void)FileRestore(begin, bytes, Touch::HostWrite, true); // (ram17: any thread, from the disk store; normal context)
		// (ram2b) A ram2 range (cold: no image, the GPU never gives it back) the disk store did not give back has its bytes
		// in RAM still (taken, not dropped yet: a cold range is only dropped with its copy on disk): let go, never a miss -
		// a missed cold range could never be restored, and a game thread's write fault on it looped for ever (ram2's hang,
		// 11-18 21:26). The window is taken out: the host writes there, it is not the streamed textures' memory.
		if (std::find(ram2_hits.begin(), ram2_hits.end(), begin) != ram2_hits.end() ||
		    std::find(mesh_hits.begin(), mesh_hits.end(), begin) != mesh_hits.end()) {
			{
				std::lock_guard lock(state.mutex);
				auto            it = state.guest.find(begin);
				if (it != state.guest.end() && (it->second.rnt == RntRam2 || it->second.rnt == RntMesh)) {
					if (!it->second.touched && !it->second.restoring) TouchLocked(state, it, Touch::HostWrite);
					it->second.missed = false;
					ranges = TakePendingLocked(state);
				} else {
					ranges.clear();
				}
			}
			RemoveAll(state, ranges);
			continue;
		}
		if (restore_fn != nullptr && can_wait != nullptr && can_wait()) {
			restore_fn(begin, bytes, Touch::HostWrite);
			continue;
		}
		std::lock_guard lock(state.mutex);
		auto            it = state.guest.find(begin);
		if (it == state.guest.end() || it->second.touched) continue;
		it->second.missed = true;
		state.missed++;
		state.missed_bytes += bytes;
		std::printf("Texture offload: MISSED host write over part of 0x%llx..0x%llx (zeros until restored)\n",
		            static_cast<unsigned long long>(begin), static_cast<unsigned long long>(begin + bytes));
		TouchLocked(state, it, Touch::HostWrite);
	}
	// (ram2b) The windows of the ram2 ranges a host write met: out of ram2 (it is not the streamed textures' memory).
	for (const auto begin: ram2_hits) {
		RamNativeTex::Ram2Guards().host_writes.fetch_add(1, std::memory_order_relaxed);
		char label[24] {};
		(void)Loader::DemonsSoulsMemMap::LabelAt(begin, label, sizeof(label));
		RamNativeTex::Ram2Block(begin, "escritura del emulador sobre memoria tomada", label);
	}
}

// Render thread: the read watchers of ranges the emulator's code touched go; missed discarded ranges come back first
// (`restore` with Touch::Count: a late restore, on the render thread itself).
inline void Drain() {
	auto&                                      state = Get();
	std::vector<std::pair<uint64_t, uint64_t>> ranges, missed;
	{
		std::lock_guard lock(state.mutex);
		if (state.pending.empty()) return;
		for (const auto begin: state.pending) {
			auto it = state.guest.find(begin);
			if (it != state.guest.end() && it->second.missed && !it->second.restoring)
				missed.emplace_back(begin, it->second.guest_end - begin);
		}
		ranges = TakePendingLocked(state);
	}
	RemoveAll(state, ranges);
	const auto restore = state.restore.load(std::memory_order_acquire);
	for (const auto& [begin, size]: missed)
		if (restore != nullptr) restore(begin, size, Touch::Count);
}

// ---- (ram18) Cold ranges (Ram18On above) ----
// The RAM18 thread: the cold ranges it found (whole pages, at most a chunk each), in place of the ones not taken yet.
inline void Ram18Propose(std::vector<std::pair<uint64_t, uint64_t>> ranges) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.cold_queue.swap(ranges);
}

// Render thread (TextureCache::OffloadScan, where the textures are taken: no other thread adds ranges): the proposed
// ranges still free - no range over them, not touched in the RAM16 retake time, not read back in the RAM17 hold time -
// added as cold ranges, discarded (dropped by the RAM18 thread once written: Ram18TakeDrops). Returned: those added,
// whose read watchers the caller puts on (before the drop: the drop waits for the next RAM18 pass).
// (ram20) The RAM18 thread's RAM20 pass: ranges with a copy in video memory, (begin, size, 1 + kind), in place of the ones
// not taken yet. Taken by Ram18TakeQueued with the same rules as the cold ones.
inline void Ram20Propose(std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> ranges) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.r20_queue.swap(ranges);
}

inline std::vector<std::pair<uint64_t, uint64_t>> Ram18TakeQueued() {
	std::vector<std::pair<uint64_t, uint64_t>> added;
	if (!Ram18On()) return added;
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	const auto take = [&](uint64_t begin, uint64_t size, uint8_t r20) {
		if (size == 0 || size > FileChunk || begin % FileUnit != 0 || size % FileUnit != 0) return;
		if (TouchedLocked(state, ColdSerial(begin))) return;
		auto it = state.guest.upper_bound(begin);
		if (it != state.guest.begin() && std::prev(it)->second.guest_end > begin) return;
		if (it != state.guest.end() && it->first < begin + size) return;
		if (HeldLocked(state, begin, size)) return;
		Range range;
		range.guest_end = begin + size;
		range.pieces    = {{begin, size}}; // (native memory: the guest pages themselves)
		range.serial    = ColdSerial(begin);
		range.image     = 0;
		range.since     = std::chrono::steady_clock::now();
		range.discarded = true;
		range.orphan    = true;
		range.cold      = true;
		range.r20       = r20;
		state.backing[begin] = begin;
		if (begin < state.guest_lo.load(std::memory_order_relaxed)) state.guest_lo.store(begin, std::memory_order_relaxed);
		if (begin + size > state.guest_hi.load(std::memory_order_relaxed)) state.guest_hi.store(begin + size, std::memory_order_relaxed);
		state.guest.emplace(begin, std::move(range));
		state.cold_ranges++;
		state.cold_taken++;
		state.cold_taken_bytes += size;
		if (r20 != 0 && r20 <= State::R20Kinds) {
			state.r20_taken[r20 - 1]++;
			state.r20_taken_bytes[r20 - 1] += size;
		}
		state.cold_drop.push_back(begin);
		added.emplace_back(begin, size);
	};
	for (const auto& [begin, size]: state.cold_queue) take(begin, size, 0);
	for (const auto& [begin, size, r20]: state.r20_queue) take(begin, size, r20 <= State::R20Kinds ? r20 : 0);
	state.cold_queue.clear();
	state.r20_queue.clear();
	return added;
}

// (ram1b, KYTY_RAM_NATIVE_TEX=1) Render thread (TextureCache::RntScan, where the textures are): ranges (begin, size,
// kind 1 arrays / 2 render targets) a GPU cache's image holds, taken as cold ranges now with the RAM18 rules (whole
// pages, at most a chunk; none over a range already taken; not touched within the RAM16 retake time; not read back
// within the RAM17 hold time): written to the disk store and dropped by the RAM18 thread at its next pass. Returned: those
// added, whose read watchers the caller puts on.
inline std::vector<std::pair<uint64_t, uint64_t>> RntTakeNow(const std::vector<std::tuple<uint64_t, uint64_t, uint8_t>>& ranges) {
	std::vector<std::pair<uint64_t, uint64_t>> added;
	if (!RntAsked() || !Ram18On() || !Native()) return added;
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	for (const auto& [begin, size, kind]: ranges) {
		if (size == 0 || size > FileChunk || begin % FileUnit != 0 || size % FileUnit != 0 || kind == 0 || kind > State::RntKinds) continue;
		if (TouchedLocked(state, ColdSerial(begin))) continue;
		auto it = state.guest.upper_bound(begin);
		if (it != state.guest.begin() && std::prev(it)->second.guest_end > begin) continue;
		if (it != state.guest.end() && it->first < begin + size) continue;
		if (HeldLocked(state, begin, size)) continue;
		Range range;
		range.guest_end = begin + size;
		range.pieces    = {{begin, size}}; // (native memory: the guest pages themselves)
		range.serial    = ColdSerial(begin);
		range.image     = 0;
		range.since     = std::chrono::steady_clock::now();
		range.discarded = true;
		range.orphan    = true;
		range.cold      = true;
		range.rnt       = kind;
		state.backing[begin] = begin;
		if (begin < state.guest_lo.load(std::memory_order_relaxed)) state.guest_lo.store(begin, std::memory_order_relaxed);
		if (begin + size > state.guest_hi.load(std::memory_order_relaxed)) state.guest_hi.store(begin + size, std::memory_order_relaxed);
		state.guest.emplace(begin, std::move(range));
		state.cold_ranges++;
		state.cold_taken++;
		state.cold_taken_bytes += size;
		state.rnt_taken[kind - 1]++;
		state.rnt_taken_bytes[kind - 1] += size;
		state.cold_drop.push_back(begin);
		added.emplace_back(begin, size);
	}
	return added;
}

// (ram2, KYTY_RAM2=1) Render thread (TextureCache::RntScan): ranges (begin, size) of the VirtualMemoryGPU zone no image
// has used for KYTY_RAM2_AGE_S since they were mapped, in RAM, outside every image of the cache, taken as cold ranges now
// with the RAM18 rules (as RntTakeNow): written to the disk store (hashed) and dropped by the RAM18 thread at its next
// pass; any touch reads them back from the file and verifies them. Returned: those added, whose read watchers the caller
// puts on.
inline std::vector<std::pair<uint64_t, uint64_t>> Ram2TakeNow(const std::vector<std::tuple<uint64_t, uint64_t, uint8_t>>& ranges) {
	std::vector<std::pair<uint64_t, uint64_t>> added;
	if (RamNativeTex::Ram2Mode() != 1 || !Ram18On() || !Native()) return added;
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	for (const auto& [begin, size, kind]: ranges) {
		if (size == 0 || size > FileChunk || begin % FileUnit != 0 || size % FileUnit != 0 || kind != RntRam2) continue;
		if (TouchedLocked(state, ColdSerial(begin))) continue;
		auto it = state.guest.upper_bound(begin);
		if (it != state.guest.begin() && std::prev(it)->second.guest_end > begin) continue;
		if (it != state.guest.end() && it->first < begin + size) continue;
		if (HeldLocked(state, begin, size)) continue;
		Range range;
		range.guest_end = begin + size;
		range.pieces    = {{begin, size}}; // (native memory: the guest pages themselves)
		range.serial    = ColdSerial(begin);
		range.image     = 0;
		range.since     = std::chrono::steady_clock::now();
		range.discarded = true;
		range.orphan    = true;
		range.cold      = true;
		range.rnt       = RntRam2;
		state.backing[begin] = begin;
		if (begin < state.guest_lo.load(std::memory_order_relaxed)) state.guest_lo.store(begin, std::memory_order_relaxed);
		if (begin + size > state.guest_hi.load(std::memory_order_relaxed)) state.guest_hi.store(begin + size, std::memory_order_relaxed);
		state.guest.emplace(begin, std::move(range));
		state.cold_ranges++;
		state.cold_taken++;
		state.cold_taken_bytes += size;
		state.r2_taken++;
		state.r2_taken_bytes += size;
		state.cold_drop.push_back(begin);
		added.emplace_back(begin, size);
	}
	return added;
}

// (ram2; render thread, its scan) 0: [begin, begin + size) may be taken (Ram2TakeNow's rules); 1: a range over it is
// taken; 2: held back (touched within the RAM16 retake time, or read back within the RAM17 hold time).
[[nodiscard]] inline int Ram2Status(uint64_t begin, uint64_t size) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	auto            it = state.guest.upper_bound(begin);
	if (it != state.guest.begin() && std::prev(it)->second.guest_end > begin) return 1;
	if (it != state.guest.end() && it->first < begin + size) return 1;
	if (TouchedLocked(state, ColdSerial(begin)) || HeldLocked(state, begin, size)) return 2;
	return 0;
}

// (BUF-NATIVO) Hilo de render: los trozos de las mallas (buf-nativo.h Due: paginas enteras, <= 1 MiB) tomados como rangos
// frios con las reglas de RAM18 (ninguno sobre otro rango tomado; no tocado en el tiempo de retoma de RAM16; no leido de
// vuelta en el de RAM17): el hilo RAM18 los copia al almacen (VRAM con mem1 etapa 2, si no el archivo) y suelta su RAM.
// Devueltos: los tomados, a los que el llamador pone los vigilantes de lectura (antes de soltarlos).
inline std::vector<std::pair<uint64_t, uint64_t>> MeshTakeNow(const std::vector<std::tuple<uint64_t, uint64_t, uint8_t>>& ranges) {
	std::vector<std::pair<uint64_t, uint64_t>> added;
	if (!BufNativo::On() || !Ram18On() || !Native()) return added;
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	for (const auto& [begin, size, kind]: ranges) {
		if (size == 0 || size > FileChunk || begin % FileUnit != 0 || size % FileUnit != 0 || kind != RntMesh) {
			state.mesh_refused++;
			continue;
		}
		auto it = state.guest.upper_bound(begin);
		if ((it != state.guest.begin() && std::prev(it)->second.guest_end > begin) || (it != state.guest.end() && it->first < begin + size) ||
		    TouchedLocked(state, ColdSerial(begin)) || HeldLocked(state, begin, size)) {
			state.mesh_refused++;
			continue;
		}
		Range range;
		range.guest_end = begin + size;
		range.pieces    = {{begin, size}}; // (memoria nativa: las paginas del juego mismas)
		range.serial    = ColdSerial(begin);
		range.image     = 0;
		range.since     = std::chrono::steady_clock::now();
		range.discarded = true;
		range.orphan    = true;
		range.cold      = true;
		range.rnt       = RntMesh;
		state.backing[begin] = begin;
		if (begin < state.guest_lo.load(std::memory_order_relaxed)) state.guest_lo.store(begin, std::memory_order_relaxed);
		if (begin + size > state.guest_hi.load(std::memory_order_relaxed)) state.guest_hi.store(begin + size, std::memory_order_relaxed);
		state.guest.emplace(begin, std::move(range));
		state.cold_ranges++;
		state.cold_taken++;
		state.cold_taken_bytes += size;
		state.mesh_taken++;
		state.mesh_taken_bytes += size;
		state.cold_drop.push_back(begin);
		added.emplace_back(begin, size);
	}
	return added;
}

// (BUF-NATIVO) Hilo del juego (destructor de la malla 0xaa1d10, antes de liberar su memoria): sus trozos tomados se
// sueltan SIN restaurar y sin escribir en la RAM del juego (la memoria se libera: nadie lee esos bytes otra vez); su copia
// en el almacen se libera, sus vigilantes se quitan, y la direccion puede volver a tomarse (otra malla en ella). Uno que
// se este escribiendo al almacen ahora mismo (restoring) se deja: si la memoria se reutiliza, el primer toque lo lee de
// vuelta como cualquier otro.
inline void MeshForget(const std::vector<std::pair<uint64_t, uint64_t>>& chunks) {
	auto&                                      state = Get();
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	std::vector<std::pair<uint64_t, uint64_t>> reopen; // (buf8)
	// (buf7) Uno que el hilo RAM18 este escribiendo al almacen (o leyendose de vuelta): se espera a que acabe. Si se
	// dejaba vigilado, liberado el bloque, dlmalloc u otro objeto en esas paginas leia bytes viejos o fallaba sin fin
	// (cuelgue 0x803528, bucle en 0xbcaede).
	for (const auto& [begin, size]: chunks) WaitRestores(begin, size);
	{
		std::lock_guard lock(state.mutex);
		for (const auto& [begin, size]: chunks) {
			auto it = state.guest.find(begin);
			if (it == state.guest.end() || it->second.rnt != RntMesh || it->second.guest_end != begin + size) continue;
			if (it->second.restoring) continue; // (solo si otro lo empezo justo ahora: la toma esta cerrada por take_mutex)
			it->second.missed = false;
			if (!it->second.touched) {
				state.mesh_forgot++;
				state.mesh_forgot_bytes += size;
				TouchLocked(state, it, Touch::HostOverwrite);
			}
			state.touched_serials.erase(ColdSerial(begin));
			state.touched_at.erase(ColdSerial(begin));
			reopen.emplace_back(begin, size);
		}
		ranges = TakePendingLocked(state);
	}
	RemoveAll(state, ranges);
	// (buf8, cuelgue 11-24 08:55: hilo del juego al 100% en el memcpy de 0x1d0cfd0 llamado desde 0xa94dbc, mismo pc 7.6 s)
	// TouchLocked deja la pagina del juego en PAGE_READONLY (memoria nativa: backing == guest) y la memoria nativa la
	// sigue dando por soltada (sin acceso); olvidado el rango, el fault de una escritura del nuevo dueno de esas paginas
	// (dlmalloc la reutiliza) no encuentra rango y vuelve sin abrir nada: bucle de fallos infinito. Como Act::Escape:
	// se abren lectura+escritura.
	if (const auto open = state.native_open.load(std::memory_order_acquire); open != nullptr) {
		for (const auto& [begin, size]: reopen) (void)open(begin, size, true);
	}
}

// (BUF-NATIVO) Los numeros de la linea "[buf-nativo]" (nunca espera: false si el cerrojo esta ocupado).
struct MeshView {
	uint64_t now = 0, peak = 0, taken = 0, taken_bytes = 0, refused = 0, faults_game = 0, faults_emu = 0, reads = 0;
	uint64_t read_bytes = 0, read_us = 0, read_us_max = 0, forgot = 0, forgot_bytes = 0, lost = 0, failed = 0, written_bytes = 0;
};
inline bool MeshRead(MeshView* out) {
	auto&                        state = Get();
	std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return false;
	out->now           = state.mesh_now;
	out->peak          = state.mesh_peak;
	out->taken         = state.mesh_taken;
	out->taken_bytes   = state.mesh_taken_bytes;
	out->refused       = state.mesh_refused;
	out->faults_game   = state.mesh_faults_game;
	out->faults_emu    = state.mesh_faults_emu;
	out->reads         = state.mesh_reads;
	out->read_bytes    = state.mesh_read_bytes;
	out->read_us       = state.mesh_read_us;
	out->read_us_max   = state.mesh_read_us_max;
	out->forgot        = state.mesh_forgot;
	out->forgot_bytes  = state.mesh_forgot_bytes;
	out->lost          = state.mesh_lost;
	out->failed        = state.mesh_failed;
	out->written_bytes = state.mesh_written_bytes;
	return true;
}

// (ram2) The "[ram2]" line's numbers (never waits: false when the state's lock is busy).
struct Ram2View {
	uint64_t now = 0, peak = 0, taken = 0, taken_bytes = 0, written_bytes = 0, zero_bytes = 0, failed = 0, reads = 0;
	uint64_t read_bytes = 0, read_us = 0, read_us_max = 0, ages[3] {}, unmapped = 0, lost = 0, checked = 0, checked_bytes = 0;
	uint64_t equal = 0, discrepancy = 0, unchecked = 0, check_us = 0, file_used = 0, file_size = 0;
};
inline bool Ram2Read(Ram2View* out) {
	auto&                        state = Get();
	std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return false;
	out->now           = state.r2_now;
	out->peak          = state.r2_peak;
	out->taken         = state.r2_taken;
	out->taken_bytes   = state.r2_taken_bytes;
	out->written_bytes = state.r2_written_bytes;
	out->zero_bytes    = state.r2_zero_bytes;
	out->failed        = state.r2_failed;
	out->reads         = state.r2_reads;
	out->read_bytes    = state.r2_read_bytes;
	out->read_us       = state.r2_read_us;
	out->read_us_max   = state.r2_read_us_max;
	for (size_t k = 0; k < 3; k++) out->ages[k] = state.r2_ages[k];
	out->unmapped      = state.r2_unmapped;
	out->lost          = state.r2_lost;
	out->checked       = state.r2_checked;
	out->checked_bytes = state.r2_checked_bytes;
	out->equal         = state.r2_equal;
	out->discrepancy   = state.r2_discrepancy;
	out->unchecked     = state.r2_unchecked;
	out->check_us      = state.r2_check_us;
	lock.unlock();
	FileUse(&out->file_used, &out->file_size);
	return true;
}

// (ram1b) Whether [vaddr, vaddr + size) meets a taken range (any kind).
[[nodiscard]] inline bool AnyTakenOver(uint64_t vaddr, uint64_t size) {
	auto& state = Get();
	if (size == 0 || vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return false;
	std::lock_guard lock(state.mutex);
	auto            it = state.guest.upper_bound(vaddr);
	if (it != state.guest.begin() && std::prev(it)->second.guest_end > vaddr) return true;
	return it != state.guest.end() && it->first < vaddr + size;
}

// (ram1b) The "[rnt]" line's numbers (never waits: false when the state's lock is busy).
struct RntView {
	static constexpr size_t Kinds = State::RntKinds;
	uint64_t now[Kinds] {}, peak = 0, taken[Kinds] {}, taken_bytes[Kinds] {}, reads[Kinds] {}, read_bytes[Kinds] {};
	uint64_t read_us = 0, read_us_max = 0, zero_bytes = 0, written_bytes = 0, lost = 0, failed = 0, released = 0, ages[3] {};
	uint64_t file_used = 0, file_size = 0;
};
inline bool RntRead(RntView* out) {
	auto&                        state = Get();
	std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return false;
	for (size_t k = 0; k < State::RntKinds; k++) {
		out->now[k]         = state.rnt_now[k];
		out->taken[k]       = state.rnt_taken[k];
		out->taken_bytes[k] = state.rnt_taken_bytes[k];
		out->reads[k]       = state.rnt_reads[k];
		out->read_bytes[k]  = state.rnt_read_bytes[k];
	}
	out->peak          = state.rnt_peak;
	out->read_us       = state.rnt_read_us;
	out->read_us_max   = state.rnt_read_us_max;
	out->zero_bytes    = state.rnt_zero_bytes;
	out->written_bytes = state.rnt_written_bytes;
	out->lost          = state.rnt_lost;
	out->failed        = state.rnt_failed;
	out->released      = state.rnt_released;
	for (size_t k = 0; k < 3; k++) out->ages[k] = state.rnt_ages[k];
	lock.unlock();
	FileUse(&out->file_used, &out->file_size);
	return true;
}

// The RAM18 thread: the cold ranges taken (watchers on) and not dropped yet, to write and drop (DiscardRangeNative).
inline std::vector<uint64_t> Ram18TakeDrops() {
	std::vector<uint64_t> drops;
	auto&                 state = Get();
	std::lock_guard       lock(state.mutex);
	drops.swap(state.cold_drop);
	return drops;
}

// The "[ram18]" line's numbers (never waits: false when the state's lock is busy).
struct Ram18View {
	uint64_t now = 0, peak = 0, ranges = 0, taken = 0, taken_bytes = 0, written = 0, written_bytes = 0, write_us = 0;
	uint64_t zero = 0, zero_bytes = 0, failed = 0, reads = 0, read_bytes = 0, read_us = 0, read_us_max = 0, lost = 0;
	uint64_t unmapped = 0, released = 0;
};
inline bool Ram18Read(Ram18View* out) {
	if (!Ram18On()) return false;
	auto&                        state = Get();
	std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return false;
	out->now           = state.cold_now;
	out->peak          = state.cold_peak;
	out->ranges        = state.cold_ranges;
	out->taken         = state.cold_taken;
	out->taken_bytes   = state.cold_taken_bytes;
	out->written       = state.cold_written;
	out->written_bytes = state.cold_written_bytes;
	out->write_us      = state.cold_write_us;
	out->zero          = state.cold_zero;
	out->zero_bytes    = state.cold_zero_bytes;
	out->failed        = state.cold_failed;
	out->reads         = state.cold_reads;
	out->read_bytes    = state.cold_read_bytes;
	out->read_us       = state.cold_read_us;
	out->read_us_max   = state.cold_read_us_max;
	out->lost          = state.cold_lost;
	out->unmapped      = state.cold_unmapped;
	out->released      = state.cold_released;
	return true;
}

// (ram20) The "[ram20]" line's numbers (never waits: false when the state's lock is busy).
struct Ram20View {
	static constexpr size_t Kinds = State::R20Kinds;
	uint64_t now[Kinds] {}, peak = 0, taken[Kinds] {}, taken_bytes[Kinds] {}, reads[Kinds] {}, read_bytes[Kinds] {};
	uint64_t read_us = 0, read_us_max = 0, zero_bytes = 0, written_bytes = 0, lost = 0, failed = 0, released = 0;
};
inline bool Ram20Read(Ram20View* out) {
	if (!Ram20Active()) return false;
	auto&                        state = Get();
	std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return false;
	for (size_t k = 0; k < State::R20Kinds; k++) {
		out->now[k]         = state.r20_now[k];
		out->taken[k]       = state.r20_taken[k];
		out->taken_bytes[k] = state.r20_taken_bytes[k];
		out->reads[k]       = state.r20_reads[k];
		out->read_bytes[k]  = state.r20_read_bytes[k];
	}
	out->peak          = state.r20_peak;
	out->read_us       = state.r20_read_us;
	out->read_us_max   = state.r20_read_us_max;
	out->zero_bytes    = state.r20_zero_bytes;
	out->written_bytes = state.r20_written_bytes;
	out->lost          = state.r20_lost;
	out->failed        = state.r20_failed;
	out->released      = state.r20_released;
	return true;
}

inline void Report(bool force) {
	auto&       state = Get();
	static auto last  = std::chrono::steady_clock::now();
	const auto  time  = std::chrono::steady_clock::now();
	if (!force && time - last < std::chrono::seconds(30)) return;
	last = time;
	std::lock_guard  lock(state.mutex);
	constexpr double MiB = 1024.0 * 1024.0;
	std::string      line;
	for (int i = 0; i < static_cast<int>(Touch::Count); i++) {
		if (state.touches[i] == 0) continue;
		char part[160];
		std::snprintf(part, sizeof(part), "%s%s %llu (%.0f MiB)", line.empty() ? "" : ", ", TouchName(static_cast<Touch>(i)),
		              static_cast<unsigned long long>(state.touches[i]), state.touch_bytes[i] / MiB);
		line += part;
	}
	const bool offload = GetMode() == Mode::Offload;
	std::printf("Texture offload: %.0f MiB %s now (peak %.0f, %llu ranges / %.0f MiB ever); touched: %s; after <1 s %llu, <10 s %llu, "
	            "<60 s %llu, later %llu\n",
	            state.now / MiB, offload ? "out of RAM" : "probed", state.peak / MiB, static_cast<unsigned long long>(state.ranges_total),
	            state.total / MiB, line.empty() ? "none" : line.c_str(), static_cast<unsigned long long>(state.ages[0]),
	            static_cast<unsigned long long>(state.ages[1]), static_cast<unsigned long long>(state.ages[2]),
	            static_cast<unsigned long long>(state.ages[3]));
	if (offload) {
		std::printf("Texture offload: checked %llu images (%.0f MiB, %llu differ); %llu restores (%.0f MiB, %.1f ms, at most %.1f ms), "
		            "%llu lost, %llu missed (%.0f MiB); %llu buffer reads waited for a restore (%.0f MiB)\n",
		            static_cast<unsigned long long>(state.checked), state.checked_bytes / MiB,
		            static_cast<unsigned long long>(state.check_failed), static_cast<unsigned long long>(state.restores),
		            state.restore_bytes / MiB, state.restore_us / 1000.0, state.restore_us_max / 1000.0,
		            static_cast<unsigned long long>(state.lost), static_cast<unsigned long long>(state.missed), state.missed_bytes / MiB,
		            static_cast<unsigned long long>(state.read_waits.load()), state.read_wait_bytes.load() / MiB);
		// (ram3) Which of the two it is: paged out (the bytes go on living in the page file or Windows' compression
		// store) or really dropped; and the partial unmaps split.
		if (state.pageout_bytes != 0)
			std::printf("Texture offload: paged out, not dropped (KYTY_OFFLOAD_PAGEOUT=1): %.0f MiB sent to the page file / "
			            "memory compression so far (KYTY_OFFLOAD_PAGEOUT=0 drops them)\n",
			            state.pageout_bytes / MiB);
		else
			std::printf("Texture offload: pages dropped (KYTY_OFFLOAD_PAGEOUT=0); unmaps over part of a range split %llu "
			            "(KYTY_OFFLOAD_UNMAP_SPLIT=%d); discard took %.0f MiB, refused %.0f MiB (error %lu; of those MEM_RESET "
			            "took %.0f, refused %.0f); still in RAM right after: %llu of %llu sampled pages\n",
			            static_cast<unsigned long long>(state.unmap_splits), UnmapSplit() ? 1 : 0, state.discard_ok / MiB,
			            state.discard_refused / MiB, static_cast<unsigned long>(state.discard_error), state.reset_ok / MiB,
			            state.reset_refused / MiB, static_cast<unsigned long long>(state.resident_after),
			            static_cast<unsigned long long>(state.sampled_after));
		if (state.gpu_ranges_total != 0) {
			std::printf("GPU-only offload: %.0f MiB of GPU-written images out of RAM now (peak %.0f, %llu ranges / %.0f MiB ever); "
			            "restored on a touch %llu ranges (%.0f MiB, %.1f ms, at most %.1f ms)\n",
			            state.gpu_now / MiB, state.gpu_peak / MiB, static_cast<unsigned long long>(state.gpu_ranges_total),
			            state.gpu_total / MiB, static_cast<unsigned long long>(state.gpu_restores), state.gpu_restore_bytes / MiB,
			            state.gpu_restore_us / 1000.0, state.gpu_restore_us_max / 1000.0);
		}
	}
	if (Native()) {
		constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
		std::string      why;
		for (int i = 0; i <= static_cast<int>(Touch::Count); i++) {
			if (state.restores_by[i] == 0) continue;
			char part[128];
			std::snprintf(part, sizeof(part), "%s%s %llu (%.0f MiB)", why.empty() ? "" : ", ",
			              i == static_cast<int>(Touch::Count) ? "tras un fallo perdido" : TouchName(static_cast<Touch>(i)),
			              static_cast<unsigned long long>(state.restores_by[i]), state.restore_bytes_by[i] / MiB);
			why += part;
		}
		const uint64_t lookups = state.window_lookups.load();
		std::printf("Native texture offload (30 s, KYTY_NATIVE_TEXTURE_OFFLOAD=1): RAM ahorrada ahora %.2f GB (pico %.2f; dejado "
		            "en total %.2f GB, rechazado %.0f MiB; tras dejar seguian en RAM %llu de %llu paginas muestreadas); "
		            "re-descargas %llu (%.0f MiB, %.1f ms, la mayor %.1f ms) por: %s; perdidas %llu (%.0f MiB), escrituras de "
		            "vuelta fallidas %llu, abiertas sin esperar %llu; ventanas del emulador %llu (miradas %llu, %.2f us cada una; "
		            "con rango tomado %llu, sin poder esperar %llu); hilo de descarte %llu pasadas, %.1f ms (la mayor %.1f ms); "
		            "expulsiones de imagenes con bytes fuera de RAM %llu (puestas al final %llu); comprobadas %llu (%.0f MiB, "
		            "distintas %llu), tomadas por su formato sin comprobar %llu (%.0f MiB)\n",
		            state.native_now / GiB, state.native_peak / GiB, state.native_total / GiB, state.native_refused / MiB,
		            static_cast<unsigned long long>(state.resident_after), static_cast<unsigned long long>(state.sampled_after),
		            static_cast<unsigned long long>(state.restores), state.restore_bytes / MiB, state.restore_us / 1000.0,
		            state.restore_us_max / 1000.0, why.empty() ? "nada" : why.c_str(), static_cast<unsigned long long>(state.missed),
		            state.missed_bytes / MiB, static_cast<unsigned long long>(state.native_write_failed),
		            static_cast<unsigned long long>(state.native_opened), static_cast<unsigned long long>(state.window_calls.load()),
		            static_cast<unsigned long long>(lookups), lookups != 0 ? state.window_ns.load() / 1000.0 / static_cast<double>(lookups) : 0.0,
		            static_cast<unsigned long long>(state.window_hits.load()), static_cast<unsigned long long>(state.window_missed.load()),
		            static_cast<unsigned long long>(state.native_drops), state.native_drop_us / 1000.0, state.native_drop_us_max / 1000.0,
		            static_cast<unsigned long long>(state.evict_offloaded), static_cast<unsigned long long>(state.evict_deferred),
		            static_cast<unsigned long long>(state.checked), state.checked_bytes / MiB,
		            static_cast<unsigned long long>(state.check_failed), static_cast<unsigned long long>(state.trusted),
		            state.trusted_bytes / MiB);
		// (texnativo 11-23) The native texture path in one line: RAM let go now, ranges written over whole or unmapped
		// (dropped, no download), and downloads back from the image (every reason) with the largest.
		const uint64_t no_dl = state.touches[static_cast<int>(Touch::HostOverwrite)] + state.touches[static_cast<int>(Touch::Unmap)];
		std::printf("[texnativo] RAM soltada %.0f MB, reescrituras sin descarga %llu, descargas por escritura parcial %llu "
		            "(la mayor %.1f ms; por subida de imagen %llu, por escritura del juego %llu)\n",
		            state.native_now / MiB, static_cast<unsigned long long>(no_dl), static_cast<unsigned long long>(state.restores),
		            state.restore_us_max / 1000.0, static_cast<unsigned long long>(state.restores_by[static_cast<int>(Touch::Upload)]),
		            static_cast<unsigned long long>(state.restores_by[static_cast<int>(Touch::GameWrite)]));
	}
#if defined(_WIN32)
	const auto exe  = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	const auto list = [exe](const std::unordered_map<uint64_t, uint64_t>& addresses, const char* what) {
		std::vector<std::pair<uint64_t, uint64_t>> top(addresses.begin(), addresses.end());
		std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		for (size_t i = 0; i < std::min<size_t>(top.size(), 10); i++) {
			HMODULE module = nullptr;
			GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			                   reinterpret_cast<LPCWSTR>(top[i].first), &module);
			wchar_t name[MAX_PATH] {};
			if (module != nullptr) GetModuleFileNameW(module, name, MAX_PATH);
			const wchar_t* base = wcsrchr(name, L'\\');
			std::printf("Texture offload:   %s %llu touches at %ls+0x%llx%s\n", what, static_cast<unsigned long long>(top[i].second),
			            base != nullptr ? base + 1 : name, static_cast<unsigned long long>(top[i].first - reinterpret_cast<uint64_t>(module)),
			            reinterpret_cast<uint64_t>(module) == exe ? " (the emulator)" : "");
		}
	};
	list(state.rips, "at");
	list(state.callers, "called from");
#endif
	std::fflush(stdout);
}

} // namespace TextureOffload
