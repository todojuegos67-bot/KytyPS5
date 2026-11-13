// Build and run on Linux (no CMake target; behaviour check of the KYTY_NATIVE_MEM=3 tracking change):
//   clang++ -std=c++20 -O2 -mavx2 -mbmi2 -DKYTY_PLATFORM=KYTY_PLATFORM_LINUX -DKYTY_COMPILER=KYTY_COMPILER_CLANG \
//     -DKYTY_BUILD=KYTY_BUILD_RELEASE -Isrc -Isrc/local -I3rdparty/fmt/include -I3rdparty/magic_enum/include/magic_enum \
//     -I<dir with cmake_config.h> -DFMT_HEADER_ONLY tests/UploadHeapSim.cpp -o upload-heap-sim -lpthread
//   ./upload-heap-sim marked; ./upload-heap-sim unmarked; SIM_BROKEN=1 ./upload-heap-sim marked (must report mismatches)
// Simulation of the UploadHeap tracking change (RegionManager::ForEachCpuUploadRange) with real page protection:
// writers store into a 4 MiB tracker region while an uploader copies CPU-dirty pages to a "GPU" shadow; a write fault
// (SIGSEGV on a write-protected page) marks the page CPU-dirty as GpuResourceManager::HandleFault does. At each
// checkpoint (writers quiet, a new observation epoch, as at a submission) one more upload must leave the shadow equal
// to memory: any difference is a missed write.
#include "graphics/host_gpu/regionManager.h"

#include <signal.h>
#include <sys/mman.h>
#include <thread>
#include <random>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>

namespace Common {
int DbgExitHandler(const char* file, int line, std::string_view msg) {
	std::fprintf(stderr, "EXIT %s:%d %.*s", file, line, static_cast<int>(msg.size()), msg.data());
	std::abort();
}
void DbgExit(int) { std::abort(); }
}
namespace Libs::LibKernel::Memory {
const uint8_t* TryGetBackingPointer(uint64_t, uint64_t) { return nullptr; }
}
namespace Libs::Graphics {
struct PageManager::Impl {};
PageManager::PageManager() = default;
PageManager::~PageManager() = default;
thread_local uint64_t t_epoch = 0;
thread_local bool     t_deferred = false;
std::atomic<uint64_t> g_protect_calls[2]; // [0] marked half, [1] unmarked half
uint64_t              g_base = 0;
bool     PageManager::ReadProtectionDeferred() noexcept { return t_deferred; }
uint64_t PageManager::ObservationEpoch() noexcept { return t_epoch; }
void     PageManager::ReapplyProtection(uint64_t, uint64_t) {}
template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if constexpr (is_read) return;
	for (const auto [start, end]: mask) {
		const uint64_t a = base_addr + start * TRACKER_PAGE_SIZE;
		mprotect(reinterpret_cast<void*>(a), (end - start) * TRACKER_PAGE_SIZE, track ? PROT_READ : PROT_READ | PROT_WRITE);
		g_protect_calls[a - g_base >= TRACKER_REGION_SIZE / 2 ? 1 : 0].fetch_add(1);
	}
}
template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);
} // namespace Libs::Graphics

using namespace Libs::Graphics;
RegionManager*        g_manager = nullptr;
std::atomic<uint64_t> g_faults[2];
std::atomic<bool>     g_pause {false}, g_stop {false};
std::atomic<int>      g_paused {0};

void OnFault(int, siginfo_t* info, void*) {
	const uint64_t a = reinterpret_cast<uint64_t>(info->si_addr) & ~(TRACKER_PAGE_SIZE - 1);
	if (a < g_base || a >= g_base + TRACKER_REGION_SIZE) std::abort();
	g_faults[a - g_base >= TRACKER_REGION_SIZE / 2 ? 1 : 0].fetch_add(1);
	std::scoped_lock lock(g_manager->lock);
	g_manager->ChangeState<DirtySource::Cpu, true>(a, TRACKER_PAGE_SIZE); // (as MarkRegionAsCpuModified)
}

int main(int argc, char** argv) {
	const bool marked = argc < 2 || std::strcmp(argv[1], "unmarked") != 0;
	void* p = mmap(reinterpret_cast<void*>(0xa000000000ull), TRACKER_REGION_SIZE, PROT_READ | PROT_WRITE,
	               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
	if (p == MAP_FAILED) return 2;
	g_base = reinterpret_cast<uint64_t>(p);
	std::vector<uint8_t> gpu(TRACKER_REGION_SIZE);
	struct sigaction sa {};
	sa.sa_sigaction = OnFault;
	sa.sa_flags     = SA_SIGINFO;
	sigaction(SIGSEGV, &sa, nullptr);
	PageManager pm;
	RegionManager manager(pm, g_base); // (all pages start CPU-dirty and writable)
	g_manager = &manager;
	UploadHeap::g_on = 1;
	if (marked) UploadHeap::Mark(g_base, TRACKER_REGION_SIZE / 2); // the first half: an upload heap
	constexpr int Writers = 4;
	std::vector<std::thread> writers;
	for (int w = 0; w < Writers; w++) {
		writers.emplace_back([w] {
			std::mt19937_64 rng(w + 1);
			uint64_t        value = static_cast<uint64_t>(w) << 56;
			while (!g_stop.load()) {
				if (g_pause.load()) {
					g_paused.fetch_add(1);
					while (g_pause.load() && !g_stop.load()) std::this_thread::yield();
					g_paused.fetch_sub(1);
					continue;
				}
				const uint64_t at = g_base + (rng() % (TRACKER_REGION_SIZE / 8)) * 8;
				reinterpret_cast<std::atomic<uint64_t>*>(at)->store(++value, std::memory_order_relaxed);
			}
		});
	}
	std::mt19937_64 rng(99);
	uint64_t        uploads = 0, copied_pages = 0, mismatches = 0, gpu_writes = 0;
	t_deferred              = true;
	const auto upload = [&](uint64_t offset, uint64_t size, bool keep) {
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		{
			std::scoped_lock lock(manager.lock);
			manager.ForEachCpuUploadRange(g_base + offset, size, keep, [&](uint64_t a, uint64_t s) { ranges.emplace_back(a, s); });
		}
		for (const auto& [a, s]: ranges) { // (the copies after the lock, as the upload worker's)
			std::memcpy(gpu.data() + (a - g_base), reinterpret_cast<const void*>(a), s);
			copied_pages += s / TRACKER_PAGE_SIZE;
		}
		uploads++;
	};
	for (int round = 0; round < 60; round++) {
		// Concurrent phase: uploads of random ranges, a few epochs.
		for (int i = 0; i < 4000; i++) {
			if (i % 500 == 0) t_epoch++;
			const uint64_t first = rng() % (TRACKER_REGION_SIZE / TRACKER_PAGE_SIZE);
			const uint64_t count = 1 + rng() % std::min<uint64_t>(64, TRACKER_REGION_SIZE / TRACKER_PAGE_SIZE - first);
			upload(first * TRACKER_PAGE_SIZE, count * TRACKER_PAGE_SIZE, true);
		}
		// Checkpoint: writers quiet, a new epoch, one full upload; the shadow must equal memory.
		g_pause = true;
		while (g_paused.load() != Writers) std::this_thread::yield();
		if (std::getenv("SIM_BROKEN") == nullptr) t_epoch++; // (negative control: no new epoch -> stale copies kept)
		upload(0, TRACKER_REGION_SIZE, true);
		for (uint64_t page = 0; page < TRACKER_REGION_SIZE; page += TRACKER_PAGE_SIZE)
			if (std::memcmp(gpu.data() + page, reinterpret_cast<const void*>(g_base + page), TRACKER_PAGE_SIZE) != 0) mismatches++;
		// A GPU write of a range (kept pages included): the written use, then GPU-owned, its bytes back, given back.
		{
			const uint64_t first = (rng() % 900) * TRACKER_PAGE_SIZE, size = 16 * TRACKER_PAGE_SIZE;
			std::vector<std::pair<uint64_t, uint64_t>> ranges;
			{
				std::scoped_lock lock(manager.lock);
				manager.ForEachCpuUploadRange(g_base + first, size, false, [&](uint64_t a, uint64_t s) { ranges.emplace_back(a, s); });
				manager.ChangeState<DirtySource::Gpu, true>(g_base + first, size);
			}
			for (uint64_t at = first; at < first + size; at += 8) gpu[at] ^= 0x5a; // the GPU's write
			mprotect(reinterpret_cast<void*>(g_base + first), size, PROT_READ | PROT_WRITE);
			std::memcpy(reinterpret_cast<void*>(g_base + first), gpu.data() + first, size); // the download
			{
				std::scoped_lock lock(manager.lock);
				manager.ChangeState<DirtySource::Gpu, false>(g_base + first, size);
				manager.ChangeState<DirtySource::Cpu, true>(g_base + first, size); // (given back to the CPU)
			}
			gpu_writes++;
		}
		// A GPU write marked directly over kept pages (MarkRegionAsGpuModified after a read-only use synchronized them).
		{
			const uint64_t first = (rng() % 400) * TRACKER_PAGE_SIZE, size = 8 * TRACKER_PAGE_SIZE;
			upload(first, size, true);
			{
				std::scoped_lock lock(manager.lock);
				manager.ChangeState<DirtySource::Gpu, true>(g_base + first, size);
			}
			for (uint64_t at = first; at < first + size; at += 8) gpu[at] ^= 0xa5;
			mprotect(reinterpret_cast<void*>(g_base + first), size, PROT_READ | PROT_WRITE);
			std::memcpy(reinterpret_cast<void*>(g_base + first), gpu.data() + first, size);
			{
				std::scoped_lock lock(manager.lock);
				manager.ChangeState<DirtySource::Gpu, false>(g_base + first, size);
				manager.ChangeState<DirtySource::Cpu, true>(g_base + first, size);
			}
			gpu_writes++;
		}
		g_pause = false;
	}
	g_stop = true;
	for (auto& t: writers) t.join();
	std::printf("%s: %llu uploads, %llu pages copied; write faults marked half %llu, other half %llu; protection calls "
	            "marked half %llu, other half %llu; kept %llu, deduped %llu, GPU-taken %llu; %llu GPU writes; MISMATCHES %llu\n",
	            marked ? "marked" : "unmarked", (unsigned long long)uploads, (unsigned long long)copied_pages,
	            (unsigned long long)g_faults[0].load(), (unsigned long long)g_faults[1].load(),
	            (unsigned long long)g_protect_calls[0].load(), (unsigned long long)g_protect_calls[1].load(),
	            (unsigned long long)UploadHeap::g_kept_pages.load(), (unsigned long long)UploadHeap::g_deduped_pages.load(),
	            (unsigned long long)UploadHeap::g_gpu_taken_pages.load(), (unsigned long long)gpu_writes,
	            (unsigned long long)mismatches);
	return mismatches == 0 ? 0 : 1;
}
