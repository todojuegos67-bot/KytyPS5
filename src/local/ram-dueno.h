#pragma once
// (ram-dueno) Who owns each big block of the "[ram-emu]" line. A cheap registry (base, size, label) filled where the
// emulator itself reserves or commits >= 1 MiB outside the guest (common/platform/sysWindowsVirtual.cpp: every
// Common::VirtualMemory path; the file reader's bounce buffer); "[ram-emu]" prints the label next to each of its
// "mayores". A block not in the registry was not reserved by the emulator's own VirtualAlloc calls: it is the NVIDIA
// driver / Vulkan loader / Streamline-DLSS or a C runtime large block, labeled "desconocido (driver?)".
// Cost: one spinlock and a linear slot search per reservation of >= 1 MiB (a few hundred in a whole session).
#include <atomic>
#include <cstdint>

namespace RamDueno {

constexpr uint64_t MinBytes = uint64_t {1} << 20u;
constexpr int      Slots    = 4096;

struct Entry {
	uint64_t    base;
	uint64_t    size;
	const char* label;
};

inline Entry* Table() {
	static Entry t[Slots] {};
	return t;
}
inline std::atomic_flag& Lock() {
	static std::atomic_flag f = ATOMIC_FLAG_INIT;
	return f;
}

inline void Add(uint64_t base, uint64_t size, const char* label) {
	if (base == 0 || size < MinBytes) return;
	while (Lock().test_and_set(std::memory_order_acquire)) {
	}
	Entry* t    = Table();
	int    free = -1;
	for (int i = 0; i < Slots; i++) {
		if (t[i].base == base) {
			free = i;
			break;
		}
		if (free < 0 && t[i].base == 0) free = i;
	}
	if (free >= 0) t[free] = {base, size, label};
	Lock().clear(std::memory_order_release);
}

inline void Remove(uint64_t base) {
	if (base == 0) return;
	while (Lock().test_and_set(std::memory_order_acquire)) {
	}
	Entry* t = Table();
	for (int i = 0; i < Slots; i++)
		if (t[i].base == base) {
			t[i] = {0, 0, nullptr};
			break;
		}
	Lock().clear(std::memory_order_release);
}

// Label of the block that contains 'address' (the allocation base the "[ram-emu]" walk finds), or nullptr.
inline const char* Find(uint64_t address) {
	const char* label = nullptr;
	while (Lock().test_and_set(std::memory_order_acquire)) {
	}
	const Entry* t = Table();
	for (int i = 0; i < Slots; i++)
		if (t[i].base != 0 && address >= t[i].base && address < t[i].base + t[i].size) {
			label = t[i].label;
			break;
		}
	Lock().clear(std::memory_order_release);
	return label;
}

} // namespace RamDueno
