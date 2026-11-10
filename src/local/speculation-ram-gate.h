#pragma once
// KYTY_SPECULATE=4 on a PC with less physical memory than KYTY_SPECULATE_RAM_GB (GiB; default 28, 0: no gate): mode 4
// stands down at boot and the run stays in mode 0. Mode 4 holds more private memory than mode 0, nearly all of it
// mimalloc's arenas (main.cpp): at the arena 3.2 GiB committed of 7 GiB reserved, ~1.3 GiB of it what the CRT heap
// holds in mode 0; over a long session ~3.6 GB more (a tour of the worlds, 21.1 against 17.5 GB private). A 16 or 24 GB
// PC with the system and the driver on top would sit at the 1.5 GB-left line where single black frames start (README).
// 28 admits a 32 GB PC (Windows reports ~31.9 GiB) and refuses 16 and 24 GB ones. Read where the heap is chosen (main.cpp,
// at the first allocation) and where the switches are applied (performance-switches.h), so both agree.

#include "local-platform.h"

#include <cstdint>
#include <cstdlib>

namespace SpeculationRamGate {

inline uint32_t ThresholdGib() {
	const char* value = std::getenv("KYTY_SPECULATE_RAM_GB");
	if (value == nullptr || *value == '\0') return 28;
	char*      end    = nullptr;
	const auto parsed = std::strtoul(value, &end, 10);
	return *end == '\0' && parsed <= 4096 ? static_cast<uint32_t>(parsed) : 28;
}

// True when KYTY_SPECULATE=4 is to stand down on this PC.
inline bool StandsDown() {
	const uint32_t gib = ThresholdGib();
	return gib != 0 && LocalPlatform::PhysicalMemory() < (uint64_t {gib} << 30u);
}

} // namespace SpeculationRamGate
