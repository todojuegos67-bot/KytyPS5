// Adversarial simulation of KYTY_NATIVE_MEM=3 item 1 (RegionManager::ForEachCpuUploadRange, uploadHeap.h), built
// against a worktree's real regionManager.h / uploadHeap.h (run: uploadheap-adversarial.sh <worktree>).
//
// Unlike tests/UploadHeapSim.cpp (which only compares at quiet checkpoints), this follows the guest/GPU protocol and
// checks EVERY draw's bytes at the moment the GPU thread reads them:
//   - a guest thread builds frames of submissions; each draw reads a chunk of 16..3000 bytes bump-allocated from a
//     per-frame slot of an arena, so pages are shared by chunks of different submissions (and frames);
//   - the guest writes a chunk only before submitting the submission that reads it (as the engine fills its
//     GameToGPU / RenderSingleFrameGPU memory), and reuses a frame slot only after the GPU thread passed that frame's
//     end-of-frame label (as a label wait makes it);
//   - the GPU thread runs concurrently: one observation epoch per submission slice (graphicsRun.cpp loop top
//     FlushDeferredProtection) and one before each label (non-state packet); draws of one submission re-read
//     pages other draws copied in the same epoch (the "once per epoch" skip is exercised all the time);
//   - write faults (real mprotect + SIGSEGV) mark pages CPU-dirty as HandleFault does;
//   - the arena is marked (UploadHeap::Mark) only after MARK_FRAME frames, from another thread, mid-flight
//     (the census thread's 10 s delayed marking);
//   - GPU writes: (a) written uses of page-aligned GPU output chunks inside the arena (upload is_written, GPU-dirty,
//     the GPU's bytes, readback at the label, given back), (b) a GPU mark directly over kept pages
//     (MarkRegionAsGpuModified after a read-only use), (c) buffer deletion mid-epoch (shadow scribbled + UntrackMemory
//     = ChangeState<Cpu,true>): the next reads in the same epoch must not be served from the scribbled shadow;
//   - a second uploader thread (not the GPU thread: stamp 0) uploads random ranges of the marked arena with keep;
//   - noise writers hammer the untracked-for-the-GPU half of the region (faults, protection changes on other pages).
// Negative controls (must FAIL): ADV_NO_SUBMIT_EPOCH=1 (no epoch at a submission start), ADV_NO_DELETE_INVALIDATE=1
// (a deleted buffer's pages not made CPU-dirty).
#include "graphics/host_gpu/regionManager.h"

#include <signal.h>
#include <sys/mman.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace Common {
int DbgExitHandler(const char* file, int line, std::string_view msg) {
	std::fprintf(stderr, "EXIT %s:%d %.*s", file, line, static_cast<int>(msg.size()), msg.data());
	std::abort();
}
void DbgExit(int) { std::abort(); }
} // namespace Common
namespace Libs::LibKernel::Memory {
const uint8_t* TryGetBackingPointer(uint64_t, uint64_t) { return nullptr; }
}
namespace Libs::Graphics {
struct PageManager::Impl {};
PageManager::PageManager()  = default;
PageManager::~PageManager() = default;
thread_local uint64_t t_epoch    = 0;
thread_local bool     t_deferred = false;
bool                  PageManager::ReadProtectionDeferred() noexcept { return t_deferred; }
uint64_t              PageManager::ObservationEpoch() noexcept { return t_epoch; }
void                  PageManager::ReapplyProtection(uint64_t, uint64_t) {}
std::atomic<uint64_t> g_protects {0};
template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if constexpr (is_read) return; // (GPU-dirty pages: the guest never touches the GPU output pages before readback)
	for (const auto [start, end]: mask) {
		mprotect(reinterpret_cast<void*>(base_addr + start * TRACKER_PAGE_SIZE), (end - start) * TRACKER_PAGE_SIZE,
		         track ? PROT_READ : PROT_READ | PROT_WRITE);
		g_protects.fetch_add(1, std::memory_order_relaxed);
	}
}
template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);
} // namespace Libs::Graphics

using namespace Libs::Graphics;

namespace {
constexpr uint64_t Page = TRACKER_PAGE_SIZE, Region = TRACKER_REGION_SIZE;
constexpr uint64_t ArenaSize = Region / 2;               // [0, 2 MiB): the arena; [2, 4 MiB): noise
constexpr uint64_t Slots = 3, SlotSize = ArenaSize / Slots & ~(Page - 1);
constexpr uint64_t GpuOutPages = 4;                      // page-aligned GPU output chunk at each slot's end
uint64_t           g_base = 0;
RegionManager*     g_manager = nullptr;
std::atomic<uint64_t> g_faults {0};

void OnFault(int, siginfo_t* info, void*) {
	const uint64_t a = reinterpret_cast<uint64_t>(info->si_addr) & ~(Page - 1);
	if (a < g_base || a >= g_base + Region) {
		signal(SIGSEGV, SIG_DFL);
		return;
	}
	g_faults.fetch_add(1, std::memory_order_relaxed);
	std::scoped_lock lock(g_manager->lock);
	g_manager->ChangeState<DirtySource::Cpu, true>(a, Page); // (HandleFault -> MarkRegionAsCpuModified)
}

struct Draw {
	uint64_t offset, size, seed;
};
struct Submission {
	uint64_t          frame;
	uint32_t          index;
	std::vector<Draw> draws;
	bool              last; // ends with the frame's label
};

uint8_t Byte(uint64_t seed, uint64_t i) {
	uint64_t x = seed * 0x9e3779b97f4a7c15ull + i * 0xbf58476d1ce4e5b9ull;
	x ^= x >> 31;
	return static_cast<uint8_t>(x);
}

std::mutex              g_queue_mutex;
std::condition_variable g_queue_cv;
std::deque<Submission>  g_queue;
std::atomic<uint64_t>   g_label {0}; // frames whose end-of-frame label the GPU thread passed (+1)
std::atomic<bool>       g_stop {false};
} // namespace

int main() {
	const bool no_submit_epoch = std::getenv("ADV_NO_SUBMIT_EPOCH") != nullptr;
	const bool no_delete_inval = std::getenv("ADV_NO_DELETE_INVALIDATE") != nullptr;
	const uint64_t frames      = std::getenv("ADV_FRAMES") ? std::strtoull(std::getenv("ADV_FRAMES"), nullptr, 10) : 400;
	const uint64_t mark_frame  = 40;
	void* p = mmap(reinterpret_cast<void*>(0xa000000000ull), Region, PROT_READ | PROT_WRITE,
	               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
	if (p == MAP_FAILED) return 2;
	g_base = reinterpret_cast<uint64_t>(p);
	struct sigaction sa {};
	sa.sa_sigaction = OnFault;
	sa.sa_flags     = SA_SIGINFO;
	sigaction(SIGSEGV, &sa, nullptr);
	PageManager   pm;
	RegionManager manager(pm, g_base);
	g_manager        = &manager;
	UploadHeap::g_on = 1;
	std::vector<uint8_t> shadow(Region, 0xcd); // the GPU's copy (the buffer cache's buffer)

	// GPU-side copy of [offset, offset + size): the upload (lock, ranges), then the copies after the lock.
	const auto upload = [&](uint64_t offset, uint64_t size, bool keep, uint8_t* into) {
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		{
			std::scoped_lock lock(manager.lock);
			manager.ForEachCpuUploadRange(g_base + offset, size, keep, [&](uint64_t a, uint64_t s) { ranges.emplace_back(a, s); });
		}
		for (const auto& [a, s]: ranges) std::memcpy(into + (a - g_base), reinterpret_cast<const void*>(a), s);
	};

	std::atomic<uint64_t> stale_draws {0}, stale_bytes {0}, draws_checked {0}, gpu_writes {0}, deletes {0}, gpu_marks {0};

	// Noise writers: the other half (watched, faulting, reprotected by the uploads below).
	std::vector<std::thread> threads;
	for (int w = 0; w < 2; w++) {
		threads.emplace_back([w] {
			std::mt19937_64 rng(1000 + w);
			while (!g_stop.load(std::memory_order_relaxed)) {
				const uint64_t at = g_base + ArenaSize + (rng() % ((Region - ArenaSize) / 8)) * 8;
				reinterpret_cast<std::atomic<uint64_t>*>(at)->store(rng(), std::memory_order_relaxed);
			}
		});
	}
	// Second uploader (not the GPU thread: no observation epoch, stamp 0) into a throwaway copy.
	threads.emplace_back([&] {
		std::mt19937_64      rng(77);
		std::vector<uint8_t> other(Region);
		while (!g_stop.load(std::memory_order_relaxed)) {
			// (Inside the marked arena only, into a throwaway copy: its uploads keep every page CPU-dirty there, so all
			// they change is the epoch stamp. Elsewhere a copy into another buffer would consume dirty pages the GPU's
			// copy never got: not what the buffer cache does, one buffer per address.)
			if (UploadHeap::RegionGranules(g_base) == 0) {
				std::this_thread::yield();
				continue;
			}
			const uint64_t first = rng() % (ArenaSize / Page), count = 1 + rng() % std::min<uint64_t>(32, ArenaSize / Page - first);
			// (Only ranges no GPU output chunk lies in: a non-GPU thread uploading them while GPU-dirty is a written use.)
			bool touches_out = false;
			for (uint64_t s = 0; s < Slots; s++) {
				const uint64_t out = s * SlotSize + SlotSize - GpuOutPages * Page;
				if (first * Page < out + GpuOutPages * Page && out < (first + count) * Page) touches_out = true;
			}
			if (!touches_out && !std::getenv("ADV_NO_SECOND")) upload(first * Page, count * Page, true, other.data());
			std::this_thread::yield();
		}
	});

	// GPU thread.
	threads.emplace_back([&] {
		t_deferred = true;
		std::mt19937_64 rng(5);
		uint64_t        own_epoch = 1;
		t_epoch                   = own_epoch;
		for (;;) {
			Submission s;
			{
				std::unique_lock lock(g_queue_mutex);
				g_queue_cv.wait(lock, [&] { return !g_queue.empty() || g_stop.load(); });
				if (g_queue.empty()) return;
				s = std::move(g_queue.front());
				g_queue.pop_front();
			}
			// (GuestGpu::Process: BdaDirtyRegions::Publish at each submission taken - the upload heap's epoch - and the loop
			// top's FlushDeferredProtection.)
			if (!no_submit_epoch) {
				t_epoch = ++own_epoch;
				UploadHeap::Publish();
			}
			for (size_t d = 0; d < s.draws.size(); d++) {
				const auto& draw = s.draws[d];
				// (c) a buffer deleted mid-epoch: its bytes gone, its pages untracked (CPU-dirty).
				if (rng() % 97 == 0 && !std::getenv("ADV_NO_DELETE")) {
					const uint64_t at = (draw.offset & ~(Page - 1));
					std::memset(shadow.data() + at, 0xee, Page);
					if (!no_delete_inval) {
						std::scoped_lock lock(manager.lock);
						manager.ChangeState<DirtySource::Cpu, true>(g_base + at, Page); // (UntrackMemory)
					}
					deletes++;
				}
				upload(draw.offset, draw.size, true, shadow.data());
				uint64_t bad = 0;
				for (uint64_t i = 0; i < draw.size; i++) bad += shadow[draw.offset + i] != Byte(draw.seed, i);
				draws_checked++;
				if (bad != 0) {
					if (stale_draws.fetch_add(1) < 5)
						std::printf("  STALE draw: frame %llu sub %u draw %zu offset 0x%llx size %llu: %llu bytes differ\n",
						            (unsigned long long)s.frame, s.index, d, (unsigned long long)draw.offset,
						            (unsigned long long)draw.size, (unsigned long long)bad);
					stale_bytes += bad;
				}
			}
			const uint64_t slot = s.frame % Slots;
			const uint64_t out  = slot * SlotSize + SlotSize - GpuOutPages * Page;
			if (s.index == 1 && !std::getenv("ADV_NO_GPUOUT")) {
				// (a) a written use of the slot's GPU output pages: GPU-owned, the GPU's bytes (read back at the label).
				{
					std::scoped_lock lock(manager.lock);
					manager.ForEachCpuUploadRange(g_base + out, GpuOutPages * Page, false, [&](uint64_t a, uint64_t sz) {
						std::memcpy(shadow.data() + (a - g_base), reinterpret_cast<const void*>(a), sz);
					});
					manager.ChangeState<DirtySource::Gpu, true>(g_base + out, GpuOutPages * Page);
				}
				for (uint64_t i = 0; i < GpuOutPages * Page; i++) shadow[out + i] = Byte(s.frame * 7 + 3, i);
				gpu_writes++;
			}
			if (s.index == 2 && UploadHeap::RegionGranules(g_base) != 0 && rng() % 3 == 0 && !std::getenv("ADV_NO_MARKGPU")) {
				// (b) a GPU mark straight over a page a read-only use just kept (MarkRegionAsGpuModified after the
				// use synchronized it): CPU chunks of this and the next submissions share the page, the guest may be
				// writing it right now. The GPU writes no byte of it here (it rewrote what it read); it is given back
				// at once (the readback a guest fault would make), then CPU-owned: later uses must copy it again.
				const uint64_t at = s.draws.front().offset & ~(Page - 1);
				std::vector<std::pair<uint64_t, uint64_t>> ranges;
				{
					std::scoped_lock lock(manager.lock);
					manager.ForEachCpuUploadRange(g_base + at, Page, true, [&](uint64_t a, uint64_t sz) { ranges.emplace_back(a, sz); });
					manager.ChangeState<DirtySource::Gpu, true>(g_base + at, Page);
					for (const auto& [a, sz]: ranges) std::memcpy(shadow.data() + (a - g_base), reinterpret_cast<const void*>(a), sz);
					manager.ChangeState<DirtySource::Gpu, false>(g_base + at, Page);
					manager.ChangeState<DirtySource::Cpu, true>(g_base + at, Page);
				}
				gpu_marks++;
			}
			if (s.last) {
				// The end-of-frame label: a non-state packet (an epoch), the GPU output read back, then the label.
				t_epoch = ++own_epoch;
				mprotect(reinterpret_cast<void*>(g_base + out), GpuOutPages * Page, PROT_READ | PROT_WRITE);
				for (uint64_t i = 0; i < GpuOutPages * Page; i++) {
					if (shadow[out + i] != Byte(s.frame * 7 + 3, i)) stale_bytes++; // (the GPU's own bytes kept)
				}
				std::memcpy(reinterpret_cast<void*>(g_base + out), shadow.data() + out, GpuOutPages * Page);
				{
					std::scoped_lock lock(manager.lock);
					manager.ChangeState<DirtySource::Gpu, false>(g_base + out, GpuOutPages * Page);
					manager.ChangeState<DirtySource::Cpu, true>(g_base + out, GpuOutPages * Page);
				}
				UploadHeap::g_frame.fetch_add(1, std::memory_order_acq_rel); // (the FrameBoundary submission)
				g_label.store(s.frame + 1, std::memory_order_release);
			}
		}
	});

	// Delayed marking (the census thread).
	std::atomic<uint64_t> g_frame {0};
	threads.emplace_back([&] {
		while (g_frame.load() < mark_frame && !g_stop.load()) std::this_thread::yield();
		if (!std::getenv("ADV_NO_MARK")) UploadHeap::Mark(g_base, ArenaSize);
	});

	// The guest's render thread.
	std::mt19937_64 rng(42);
	uint64_t        seed = 1;
	for (uint64_t frame = 0; frame < frames; frame++) {
		g_frame.store(frame);
		// The slot of frame - Slots is reused only after its label.
		while (frame >= Slots && g_label.load(std::memory_order_acquire) < frame - Slots + 1) std::this_thread::yield();
		const uint64_t slot = frame % Slots;
		uint64_t       cursor = slot * SlotSize;
		const uint64_t limit  = slot * SlotSize + SlotSize - GpuOutPages * Page;
		const uint32_t subs   = 3 + rng() % 4;
		for (uint32_t sub = 0; sub < subs; sub++) {
			Submission s {frame, sub, {}, sub + 1 == subs};
			const uint32_t draws = 2 + rng() % 24;
			for (uint32_t d = 0; d < draws && cursor < limit - 4096; d++) {
				Draw draw {cursor, 16 + rng() % 3000, ++seed};
				if (rng() % 4 == 0 && !s.draws.empty()) {
					draw = s.draws[rng() % s.draws.size()]; // (a chunk read again by a later draw of the submission)
				} else {
					auto* at = reinterpret_cast<uint8_t*>(g_base + draw.offset);
					for (uint64_t i = 0; i < draw.size; i++) at[i] = Byte(draw.seed, i);
					cursor += (draw.size + 15) & ~uint64_t {15};
				}
				s.draws.push_back(draw);
			}
			{
				std::scoped_lock lock(g_queue_mutex);
				g_queue.push_back(std::move(s));
			}
			g_queue_cv.notify_one();
			if (rng() % 3 == 0) std::this_thread::yield();
		}
	}
	while (g_label.load() < frames) std::this_thread::yield();
	g_stop = true;
	g_queue_cv.notify_all();
	for (auto& t: threads) t.join();
	std::printf("frames %llu, draws checked %llu, stale draws %llu (%llu bytes); write faults %llu, protection calls %llu; "
	            "kept-copies %llu deduped %llu promoted %llu demoted %llu GPU-taken %llu; GPU written uses %llu, GPU marks over kept %llu, deletions %llu; "
	            "marked %llu KiB\n",
	            (unsigned long long)frames, (unsigned long long)draws_checked.load(), (unsigned long long)stale_draws.load(),
	            (unsigned long long)stale_bytes.load(), (unsigned long long)g_faults.load(),
	            (unsigned long long)g_protects.load(), (unsigned long long)UploadHeap::g_kept_copies.load(),
	            (unsigned long long)UploadHeap::g_deduped_pages.load(), (unsigned long long)UploadHeap::g_promoted_pages.load(),
	            (unsigned long long)UploadHeap::g_demoted_pages.load(), (unsigned long long)UploadHeap::g_gpu_taken_pages.load(),
	            (unsigned long long)gpu_writes.load(), (unsigned long long)gpu_marks.load(), (unsigned long long)deletes.load(),
	            (unsigned long long)(UploadHeap::g_marked_bytes.load() >> 10));
	const bool ok = stale_draws.load() == 0 && stale_bytes.load() == 0;
	std::printf("result: %s\n", ok ? "PASS (no stale byte)" : "FAIL (stale GPU data)");
	return ok ? 0 : 1;
}
