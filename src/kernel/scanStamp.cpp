#include "kernel/scanStamp.h"

#include "common/perfStats.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>

namespace Libs::ScanStamp {

namespace {

constexpr uint64_t PageBits   = 12u;              // 4 KiB pages, as the page tracker's
constexpr uint32_t SlotBits   = 20u;              // 1 Mi stamps (8 MiB): 4 GiB without aliasing
constexpr uint64_t MaxPages   = 4096u;            // a wider stamp moves the floor (every range)
constexpr size_t   MaxStacks  = 256u;

std::atomic<uint64_t>                    g_epoch {1};
std::atomic<uint64_t>                    g_floor {0};
std::atomic<uint64_t>                    g_layout {1};
std::unique_ptr<std::atomic<uint64_t>[]> g_slots;

struct Stack {
	std::atomic<uint64_t> start {0};
	std::atomic<uint64_t> end {0};
};
std::array<Stack, MaxStacks> g_stacks;
std::atomic<bool>            g_stacks_overflow {false};

size_t Slot(uint64_t page) {
	return static_cast<size_t>((page * 0x9E3779B97F4A7C15ull) >> (64u - SlotBits));
}

void Raise(std::atomic<uint64_t>& slot, uint64_t stamp) {
	auto current = slot.load(std::memory_order_relaxed);
	while (current < stamp &&
	       !slot.compare_exchange_weak(current, stamp, std::memory_order_seq_cst,
	                                   std::memory_order_relaxed)) {
	}
}

} // namespace

int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_SCAN_STAMP");
		const int   m     = value != nullptr ? std::atoi(value) : 0;
		const int   clamped = m < 0 || m > 2 ? 0 : m;
		if (clamped != 0) {
			g_slots = std::make_unique<std::atomic<uint64_t>[]>(size_t {1} << SlotBits);
		}
		Log::WriteUnmuted(clamped == 0 ? "Sello del escaner: apagado (KYTY_SCAN_STAMP=1 lo usa, =2 "
		                                 "lo verifica contra los bytes)\n"
		                  : clamped == 1
		                      ? "Sello del escaner: encendido (KYTY_SCAN_STAMP=1)\n"
		                      : "Sello del escaner: verificador (KYTY_SCAN_STAMP=2: sello y bytes, "
		                        "decide la comparacion de bytes)\n");
		return clamped;
	}();
	return mode;
}

bool On() {
	static const bool on = Mode() != 0;
	return on;
}

uint64_t Epoch() {
	return g_epoch.load(std::memory_order_seq_cst);
}

void Note(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !On()) {
		return;
	}
	const uint64_t stamp = g_epoch.fetch_add(1, std::memory_order_seq_cst) + 1u;
	const uint64_t first = vaddr >> PageBits;
	const uint64_t last  = (vaddr + size - 1u) >> PageBits;
	if (last - first >= MaxPages) {
		Raise(g_floor, stamp);
		return;
	}
	for (uint64_t page = first; page <= last; page++) {
		Raise(g_slots[Slot(page)], stamp);
	}
}

bool WrittenSince(uint64_t vaddr, uint64_t size, uint64_t epoch) {
	if (g_floor.load(std::memory_order_seq_cst) > epoch) {
		return true;
	}
	if (size == 0) {
		return false;
	}
	const uint64_t first = vaddr >> PageBits;
	const uint64_t last  = (vaddr + size - 1u) >> PageBits;
	if (last - first >= MaxPages) {
		return true;
	}
	for (uint64_t page = first; page <= last; page++) {
		if (g_slots[Slot(page)].load(std::memory_order_seq_cst) > epoch) {
			return true;
		}
	}
	return false;
}

void LayoutChanging() {
	if (On()) {
		g_layout.fetch_add(1, std::memory_order_seq_cst);
	}
}

uint64_t LayoutEpoch() {
	return g_layout.load(std::memory_order_seq_cst);
}

void AddLiveStack(uint64_t start, uint64_t end) {
	if (!On()) {
		return;
	}
	for (auto& stack: g_stacks) {
		uint64_t expected = 0;
		if (stack.end.load(std::memory_order_relaxed) == 0 &&
		    stack.start.compare_exchange_strong(expected, start, std::memory_order_seq_cst)) {
			stack.end.store(end, std::memory_order_seq_cst);
			LayoutChanging();
			return;
		}
	}
	g_stacks_overflow.store(true, std::memory_order_seq_cst);
	LayoutChanging();
}

void RemoveLiveStack(uint64_t start, uint64_t end) {
	if (!On()) {
		return;
	}
	for (auto& stack: g_stacks) {
		if (stack.start.load(std::memory_order_relaxed) == start &&
		    stack.end.load(std::memory_order_relaxed) == end) {
			stack.end.store(0, std::memory_order_seq_cst);
			stack.start.store(0, std::memory_order_seq_cst);
			return;
		}
	}
}

bool OverlapsLiveStack(uint64_t vaddr, uint64_t size) {
	if (g_stacks_overflow.load(std::memory_order_seq_cst)) {
		return true;
	}
	for (const auto& stack: g_stacks) {
		const auto end = stack.end.load(std::memory_order_seq_cst);
		if (end == 0) {
			continue;
		}
		const auto start = stack.start.load(std::memory_order_seq_cst);
		if (vaddr < end && start < vaddr + size) {
			return true;
		}
	}
	return false;
}

} // namespace Libs::ScanStamp
