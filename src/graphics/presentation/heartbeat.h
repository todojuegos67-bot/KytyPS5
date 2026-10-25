#ifndef KYTY_GRAPHICS_PRESENTATION_HEARTBEAT_H_
#define KYTY_GRAPHICS_PRESENTATION_HEARTBEAT_H_

#include <atomic>
#include <cstdint>

// Counters printed every 10 s by the flip queue (videoOut.cpp) for black-screen diagnosis.
inline std::atomic<uint64_t> g_kyty_skipped_dispatches {0};
inline std::atomic<uint64_t> g_kyty_unsupported_dispatches {0};

#endif
