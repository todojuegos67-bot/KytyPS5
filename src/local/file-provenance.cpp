#include "file-provenance.h"

#include "common/file.h"
#include "common/stringUtils.h"
#include "fresh-memory.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
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

namespace FileProvenance {

namespace {

namespace Memory = Libs::LibKernel::Memory;

constexpr uint64_t PagesPerGranule = uint64_t {1} << (GranuleBits - PageBits);
constexpr uint64_t MaxFiles        = IdMask + 1; // (id 0: none)
constexpr int64_t  ReportSeconds   = 30;
constexpr uint64_t VerifyPages     = (uint64_t {16} << 20u) / Page; // 16 MiB a pass
constexpr uint64_t VerifyRunPages  = 256;                           // read 1 MiB at most at once

struct FileTable {
	std::mutex                                mutex;
	std::unordered_map<std::string, uint16_t> ids;
	std::vector<std::string>                  paths {std::string()}; // [id]
	bool                                      full_logged = false;
};

FileTable& Files() {
	static FileTable table;
	return table;
}

struct Ranges {
	std::mutex                                             mutex;
	std::map<std::pair<uint64_t, uint64_t>, uint32_t>      texture, buffer; // (address, size) -> registrations
};

Ranges& Registered() {
	static Ranges ranges;
	return ranges;
}

bool EnvOn(const char* name, bool fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') return fallback;
	return value[0] != '0';
}

void* AllocZeroed(uint64_t bytes) {
#if defined(_WIN32)
	// (Zero pages from the system, resident only where written.)
	return VirtualAlloc(nullptr, static_cast<SIZE_T>(bytes), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
	return std::calloc(static_cast<size_t>(bytes), 1);
#endif
}

// Calls piece(backing_address, guest_offset, size) for the parts of guest range [vaddr, vaddr + size) in the backing
// store (in order).
template <typename Piece>
void ForEachBackingPiece(uint64_t vaddr, uint64_t size, Piece&& piece) {
	for (uint64_t done = 0; done < size;) {
		const uint64_t at  = vaddr + done;
		const auto     end = Memory::BackingMappingEnd(at);
		if (end <= at) {
			done += std::min(size - done, Page - (at & (Page - 1)));
			continue;
		}
		const auto bytes = std::min(end - at, size - done);
		if (const auto* backing = Memory::TryGetBackingPointer(at, bytes); backing != nullptr)
			piece(reinterpret_cast<uint64_t>(backing), done, bytes);
		done += bytes;
	}
}

bool Tainted(const State& state, uint64_t granule) {
	const auto mark   = state.granule_mark[granule].load(std::memory_order_acquire);
	const auto record = state.granule_record[granule].load(std::memory_order_acquire);
	return mark != 0 && mark >= record;
}

void RecordPiece(State& state, uint64_t begin, uint64_t bytes, uint16_t file_id, uint64_t file_offset) {
	if (begin >= state.size || bytes > state.size - begin) return;
	const uint64_t first = (begin + Page - 1) >> PageBits;
	const uint64_t last  = std::min((begin + bytes) >> PageBits, state.page_count);
	if (first >= last) return;
	uint32_t sequence = state.sequence.fetch_add(1, std::memory_order_acq_rel) + 1;
	if (sequence == 0) sequence = state.sequence.fetch_add(1, std::memory_order_acq_rel) + 1; // (wrapped)
	// A granule's record sequence: this record's when it holds no other intact page, else its older one kept (a write
	// after that one then leaves out these pages too: conservative).
	for (uint64_t granule = first / PagesPerGranule, end = (last - 1) / PagesPerGranule;
	     granule <= end && granule < state.granules; ++granule) {
		const uint64_t gp0 = granule * PagesPerGranule;
		const uint64_t gp1 = std::min(gp0 + PagesPerGranule, state.page_count);
		bool           others = false;
		for (uint64_t page = gp0; page < gp1 && !others; ++page) {
			if (page >= first && page < last) continue;
			others = (state.pages[page].load(std::memory_order_relaxed) & ValidBit) != 0;
		}
		if (!others || state.granule_record[granule].load(std::memory_order_relaxed) == 0)
			state.granule_record[granule].store(sequence, std::memory_order_release);
	}
	for (uint64_t page = first; page < last; ++page) {
		const uint64_t offset = file_offset + ((page << PageBits) - begin);
		const uint64_t word   = ValidBit | (uint64_t {file_id} << IdShift) | (offset & OffsetMask);
		if ((state.pages[page].exchange(word, std::memory_order_acq_rel) & ValidBit) == 0)
			state.valid_pages.fetch_add(1, std::memory_order_relaxed);
	}
	state.recorded.fetch_add(last - first, std::memory_order_relaxed);
}

std::string ShortName(const std::string& path) {
	const auto slash = path.find_last_of("/\\");
	return slash == std::string::npos ? path : path.substr(slash + 1);
}

struct Verify {
	uint64_t cursor = 0, compared = 0, mismatched = 0, unreadable = 0, logged = 0;
	std::vector<uint8_t> buffer;
};

void VerifyPass(State& state, Verify& verify, const std::vector<std::string>& paths, uint64_t* pass_compared,
                uint64_t* pass_mismatched) {
	struct Run {
		uint64_t first_page = 0, count = 0, file_offset = 0;
		uint16_t id = 0;
	};
	std::vector<Run> runs;
	uint64_t         sampled = 0;
	for (uint64_t walked = 0; walked < state.page_count && sampled < VerifyPages; ++walked) {
		const uint64_t page = verify.cursor;
		verify.cursor       = verify.cursor + 1 < state.page_count ? verify.cursor + 1 : 0;
		const uint64_t word = state.pages[page].load(std::memory_order_acquire);
		if ((word & ValidBit) == 0) continue;
		const auto id     = static_cast<uint16_t>((word >> IdShift) & IdMask);
		const auto offset = word & OffsetMask;
		if (!runs.empty()) {
			auto& run = runs.back();
			if (run.id == id && run.first_page + run.count == page && run.file_offset + (run.count << PageBits) == offset &&
			    run.count < VerifyRunPages) {
				++run.count;
				++sampled;
				continue;
			}
		}
		runs.push_back({page, 1, offset, id});
		++sampled;
	}
	std::unordered_map<uint16_t, std::unique_ptr<Common::File>> opened;
	for (const auto& run: runs) {
		if (run.id >= paths.size() || paths[run.id].empty()) {
			verify.unreadable += run.count;
			continue;
		}
		// (Open for the pass: an archive member's open reads the archive's index.)
		auto& file = opened[run.id];
		if (file == nullptr) {
			file = std::make_unique<Common::File>();
			if (!file->Open(Common::PathFromUtf8(paths[run.id]), Common::File::Mode::Read)) file->Close();
		}
		if (file->IsInvalid() || !file->Seek(run.file_offset)) {
			verify.unreadable += run.count;
			continue;
		}
		verify.buffer.resize(static_cast<size_t>(run.count << PageBits));
		uint32_t got = 0;
		file->Read(verify.buffer.data(), static_cast<uint32_t>(verify.buffer.size()), &got);
		for (uint64_t i = 0; i < run.count; ++i) {
			if (((i + 1) << PageBits) > got) {
				verify.unreadable += run.count - i;
				break;
			}
			const uint64_t page     = run.first_page + i;
			const uint64_t expected = ValidBit | (uint64_t {run.id} << IdShift) | ((run.file_offset + (i << PageBits)) & OffsetMask);
			if ((state.pages[page].load(std::memory_order_acquire) & ~DiscardedBit) != expected) continue;
			const bool same = std::memcmp(reinterpret_cast<const void*>(state.base + (page << PageBits)),
			                              verify.buffer.data() + (i << PageBits), Page) == 0;
			// (Written meanwhile, by a write that was seen: not counted either way.)
			if ((state.pages[page].load(std::memory_order_acquire) & ~DiscardedBit) != expected) continue;
			++verify.compared;
			++*pass_compared;
			if (same) continue;
			++verify.mismatched;
			++*pass_mismatched;
			if (verify.logged++ < 16) {
				std::printf("File provenance verify: mismatch, backing offset 0x%" PRIx64 " vs %s at 0x%" PRIx64 "\n",
				            page << PageBits, ShortName(paths[run.id]).c_str(), run.file_offset + (i << PageBits));
			}
		}
	}
}

void Report(State& state, bool verify_on) {
	static std::vector<uint8_t>  flags;
	static std::vector<uint64_t> per_file;
	static Verify                verify;
	constexpr uint8_t            TextureFlag = 1, BufferFlag = 2;

	std::vector<std::pair<uint64_t, uint64_t>> textures, buffers;
	{
		auto&           ranges = Registered();
		std::lock_guard lock(ranges.mutex);
		textures.reserve(ranges.texture.size());
		buffers.reserve(ranges.buffer.size());
		for (const auto& entry: ranges.texture) textures.push_back(entry.first);
		for (const auto& entry: ranges.buffer) buffers.push_back(entry.first);
	}
	flags.assign(static_cast<size_t>(state.page_count), 0);
	const auto mark = [&](const std::vector<std::pair<uint64_t, uint64_t>>& list, uint8_t flag) {
		for (const auto& [vaddr, size]: list) {
			ForEachBackingPiece(vaddr, size, [&](uint64_t backing, uint64_t /*guest_offset*/, uint64_t bytes) {
				if (backing < state.base || backing - state.base >= state.size) return;
				const uint64_t begin = backing - state.base;
				const uint64_t end   = std::min(begin + bytes, state.size);
				for (uint64_t page = begin >> PageBits, last = (end + Page - 1) >> PageBits; page < last && page < state.page_count;
				     ++page)
					flags[page] |= flag;
			});
		}
	};
	mark(textures, TextureFlag);
	mark(buffers, BufferFlag);

	std::vector<std::string> paths;
	{
		auto&           files = Files();
		std::lock_guard lock(files.mutex);
		paths = files.paths;
	}
	per_file.assign(paths.size(), 0);
	uint64_t intact = 0, conservative = 0, on_textures = 0, on_buffers = 0;
	for (uint64_t page = 0; page < state.page_count; ++page) {
		const uint64_t word = state.pages[page].load(std::memory_order_relaxed);
		if ((word & ValidBit) == 0) continue;
		++intact;
		const auto id = static_cast<size_t>((word >> IdShift) & IdMask);
		if (id < per_file.size()) ++per_file[id];
		if (!Tainted(state, page / PagesPerGranule)) ++conservative;
		if ((flags[page] & TextureFlag) != 0) ++on_textures;
		if ((flags[page] & BufferFlag) != 0) ++on_buffers;
	}
	std::vector<size_t> order;
	for (size_t id = 1; id < per_file.size(); ++id)
		if (per_file[id] != 0) order.push_back(id);
	const size_t top = std::min<size_t>(8, order.size());
	std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top), order.end(),
	                  [&](size_t a, size_t b) { return per_file[a] > per_file[b]; });
	std::string files_text;
	for (size_t i = 0; i < top; ++i) {
		char text[32];
		std::snprintf(text, sizeof(text), "=%.1f", static_cast<double>(per_file[order[i]] << PageBits) / (1024.0 * 1024.0));
		files_text += (i == 0 ? "" : ", ") + ShortName(paths[order[i]]) + text;
	}
	const auto mib = [](uint64_t pages) { return static_cast<double>(pages << PageBits) / (1024.0 * 1024.0); };
	std::printf("File provenance: intact %.1f MiB (conservative %.1f MiB: no write mark in the granule since), on "
	            "textures %.1f MiB, on buffers %.1f MiB, not CPU-read recently: not measured (CPU reads are not seen); "
	            "recorded %.1f MiB, cleared %.1f MiB in all, %llu reads not recorded; %zu files; top: %s "
	            "(guest CPU stores are not seen: KYTY_PROVENANCE_VERIFY=1 measures them)\n",
	            mib(intact), mib(conservative), mib(on_textures), mib(on_buffers),
	            mib(state.recorded.load(std::memory_order_relaxed)), mib(state.cleared.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(state.unrecorded.load(std::memory_order_relaxed)), paths.size() - 1,
	            files_text.empty() ? "none" : files_text.c_str());
	if (verify_on) {
		uint64_t compared = 0, mismatched = 0;
		VerifyPass(state, verify, paths, &compared, &mismatched);
		std::printf("File provenance verify: %.1f MiB compared, %llu pages differ from the file; in all %llu of %llu "
		            "pages (%.2f%%), %llu not readable\n",
		            mib(compared), static_cast<unsigned long long>(mismatched),
		            static_cast<unsigned long long>(verify.mismatched), static_cast<unsigned long long>(verify.compared),
		            verify.compared != 0 ? 100.0 * static_cast<double>(verify.mismatched) / static_cast<double>(verify.compared)
		                                 : 0.0,
		            static_cast<unsigned long long>(verify.unreadable));
	}
	std::fflush(stdout);
}

void Watch(State* state, bool verify_on) {
#if defined(_WIN32)
	SetThreadDescription(GetCurrentThread(), L"Kyty.FileProvenance");
	// (Low CPU, I/O and memory priority: the verify's reads wait behind the game's.)
	SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
#endif
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(ReportSeconds));
		Report(*state, verify_on);
	}
}

} // namespace

State* Init() {
	static std::mutex mutex;
	std::lock_guard   lock(mutex);
	if (auto* state = g_state.load(std::memory_order_acquire); state != nullptr) return state;
	if (g_off.load(std::memory_order_relaxed)) return nullptr;
	if (!EnvOn("KYTY_PROVENANCE", false) || !FreshMemory::Enabled()) {
		g_off.store(true, std::memory_order_relaxed);
		std::printf("File provenance: off (%s)\n", FreshMemory::Enabled() ? "default; KYTY_PROVENANCE=1 turns it on" : "needs fresh memory");
		std::fflush(stdout);
		return nullptr;
	}
	uint64_t base = 0, size = 0;
	if (!Memory::BackingView(&base, &size) || base == 0 || size == 0) return nullptr; // (not yet)
	static State state;
	state.base       = base;
	state.size       = size;
	state.page_count = (size + Page - 1) >> PageBits;
	state.granules   = (size + (uint64_t {1} << GranuleBits) - 1) >> GranuleBits;
	auto* pages      = AllocZeroed(state.page_count * sizeof(uint64_t));
	auto* marks      = AllocZeroed(state.granules * sizeof(uint32_t));
	auto* records    = AllocZeroed(state.granules * sizeof(uint32_t));
	if (pages == nullptr || marks == nullptr || records == nullptr) {
		g_off.store(true, std::memory_order_relaxed);
		std::printf("File provenance: off (no memory for its table)\n");
		std::fflush(stdout);
		return nullptr;
	}
	// (Zeroed memory: the atomics' initial values. std::atomic of these types is lock-free and has their layout.)
	static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t) && sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
	state.pages          = static_cast<std::atomic<uint64_t>*>(pages);
	state.granule_mark   = static_cast<std::atomic<uint32_t>*>(marks);
	state.granule_record = static_cast<std::atomic<uint32_t>*>(records);
	const bool verify_on = EnvOn("KYTY_PROVENANCE_VERIFY", false);
	std::printf("File provenance: on (KYTY_PROVENANCE=0 turns it off; census only), %llu MiB of backing store, a %llu "
	            "KiB table; verify %s (KYTY_PROVENANCE_VERIFY=1: 16 MiB of intact pages compared with their files every "
	            "%lld s); a line every %lld s\n",
	            static_cast<unsigned long long>(size >> 20u),
	            static_cast<unsigned long long>((state.page_count * sizeof(uint64_t)) >> 10u), verify_on ? "on" : "off",
	            static_cast<long long>(ReportSeconds), static_cast<long long>(ReportSeconds));
	std::fflush(stdout);
	g_state.store(&state, std::memory_order_release);
	std::thread(Watch, &state, verify_on).detach();
	return &state;
}

void ClearGuestRange(uint64_t vaddr, uint64_t size) {
	auto* state = Get();
	if (state == nullptr || size == 0 || state->valid_pages.load(std::memory_order_relaxed) <= 0) return;
	ForEachBackingPiece(vaddr, size, [](uint64_t backing, uint64_t /*guest_offset*/, uint64_t bytes) {
		ClearBacking(backing, bytes);
	});
}

uint16_t FileId(const std::string& path) {
	if (Get() == nullptr || path.empty()) return 0;
	auto&           files = Files();
	std::lock_guard lock(files.mutex);
	if (const auto found = files.ids.find(path); found != files.ids.end()) return found->second;
	if (files.paths.size() >= MaxFiles) {
		if (!files.full_logged) {
			files.full_logged = true;
			std::printf("File provenance: file table full (%llu files): later files are not recorded\n",
			            static_cast<unsigned long long>(MaxFiles - 1));
			std::fflush(stdout);
		}
		return 0;
	}
	const auto id = static_cast<uint16_t>(files.paths.size());
	files.paths.push_back(path);
	files.ids.emplace(path, id);
	return id;
}

void Record(uint64_t vaddr, uint64_t size, uint16_t file_id, uint64_t file_offset) {
	auto* state = Get();
	if (state == nullptr || size == 0) return;
	if (file_id == 0 || file_offset > OffsetMask || size > OffsetMask + 1 - file_offset) {
		state->unrecorded.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	ForEachBackingPiece(vaddr, size, [&](uint64_t backing, uint64_t guest_offset, uint64_t bytes) {
		if (backing < state->base) return;
		RecordPiece(*state, backing - state->base, bytes, file_id, file_offset + guest_offset);
	});
}

void NoteRange(Kind kind, uint64_t vaddr, uint64_t size, bool registered) {
	if (size == 0 || Get() == nullptr) return;
	auto&           ranges = Registered();
	std::lock_guard lock(ranges.mutex);
	auto&           map = kind == Kind::Texture ? ranges.texture : ranges.buffer;
	const auto      key = std::make_pair(vaddr, size);
	if (registered) {
		++map[key];
	} else if (const auto found = map.find(key); found != map.end() && --found->second == 0) {
		map.erase(found);
	}
}

} // namespace FileProvenance
