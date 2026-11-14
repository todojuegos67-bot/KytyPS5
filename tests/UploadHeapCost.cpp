// Cost model of the KYTY_NATIVE_MEM=3 upload heap against plain write tracking (same harness as UploadHeapSim.cpp:
// real mprotect, SIGSEGV write faults). Workloads, 256 pages, 600 frames, one or more submissions (publish epochs) a
// frame, every upload checked against memory (any stale byte fails):
//   write1-read1: each page written once a frame, read by one submission (the engine's per-frame constants);
//   write1-read3: written once, read by three submissions;
//   write0-read1: never written after the first frame, read every frame.
// Prints, per workload, marked vs unmarked: write faults, protection calls and pages copied.
#include "graphics/host_gpu/regionManager.h"

#include <signal.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
uint64_t g_protects = 0;
bool     PageManager::ReadProtectionDeferred() noexcept { return true; }
uint64_t PageManager::ObservationEpoch() noexcept { return 0; }
void     PageManager::ReapplyProtection(uint64_t, uint64_t) {}
template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if constexpr (is_read) return;
	for (const auto [start, end]: mask) {
		mprotect(reinterpret_cast<void*>(base_addr + start * TRACKER_PAGE_SIZE), (end - start) * TRACKER_PAGE_SIZE,
		         track ? PROT_READ : PROT_READ | PROT_WRITE);
		g_protects++;
	}
}
template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);
} // namespace Libs::Graphics
using namespace Libs::Graphics;

uint64_t       g_base = 0, g_faults = 0;
RegionManager* g_manager = nullptr;
void OnFault(int, siginfo_t* info, void*) {
	const uint64_t a = reinterpret_cast<uint64_t>(info->si_addr) & ~(TRACKER_PAGE_SIZE - 1);
	g_faults++;
	g_manager->ChangeState<DirtySource::Cpu, true>(a, TRACKER_PAGE_SIZE);
}

int main() {
	constexpr uint64_t Pages = 256, Frames = 600;
	struct sigaction sa {};
	sa.sa_sigaction = OnFault;
	sa.sa_flags     = SA_SIGINFO;
	sigaction(SIGSEGV, &sa, nullptr);
	bool ok = true;
	struct Work {
		const char* name;
		int         writes, reads;
	};
	for (const Work w: {Work {"write1-read1", 1, 1}, Work {"write1-read3", 1, 3}, Work {"write0-read1", 0, 1}}) {
		for (const bool marked: {false, true}) {
			// (Below the tracker's 2^40 address space, region aligned.)
			void* p = mmap(reinterpret_cast<void*>(0xb000000000ull), TRACKER_REGION_SIZE, PROT_READ | PROT_WRITE,
			               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
			if (p == MAP_FAILED) return 2;
			g_base = reinterpret_cast<uint64_t>(p);
			PageManager   pm;
			RegionManager manager(pm, g_base);
			g_manager = &manager;
			UploadHeap::g_on = 1;
			for (auto& g: UploadHeap::g_granules) g.store(0);
			if (marked) UploadHeap::Mark(g_base, Pages * TRACKER_PAGE_SIZE);
			std::vector<uint8_t> gpu(Pages * TRACKER_PAGE_SIZE);
			g_faults = g_protects = 0;
			uint64_t copied = 0, stale = 0;
			for (uint64_t frame = 0; frame < Frames; frame++) {
				if (w.writes != 0 || frame == 0)
					for (uint64_t page = 0; page < Pages; page++)
						reinterpret_cast<volatile uint64_t*>(g_base + page * TRACKER_PAGE_SIZE)[frame % 512] = frame * 1000 + page;
				for (int r = 0; r < w.reads; r++) {
					UploadHeap::Publish(); // (a submission)
					manager.ForEachCpuUploadRange(g_base, Pages * TRACKER_PAGE_SIZE, true, [&](uint64_t a, uint64_t s) {
						std::memcpy(gpu.data() + (a - g_base), reinterpret_cast<const void*>(a), s);
						copied += s / TRACKER_PAGE_SIZE;
					});
					stale += std::memcmp(gpu.data(), reinterpret_cast<const void*>(g_base), gpu.size()) != 0;
				}
				UploadHeap::g_frame.fetch_add(1);
			}
			std::printf("%-13s %-8s write faults %6llu  protection calls %6llu  pages copied %7llu  stale uploads %llu\n", w.name,
			            marked ? "marked" : "tracked", (unsigned long long)g_faults, (unsigned long long)g_protects,
			            (unsigned long long)copied, (unsigned long long)stale);
			ok = ok && stale == 0;
			munmap(p, TRACKER_REGION_SIZE);
		}
	}
	return ok ? 0 : 1;
}
