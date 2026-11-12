#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_SYNC_ON_ADDRESS_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_SYNC_ON_ADDRESS_H_

#include "common/abi.h"
#include "common/common.h"

#include <chrono>

namespace Libs::LibKernel::SyncOnAddress {

using signal_poll_func_t = void (*)();

int Wait32(volatile uint32_t* address, uint32_t expected, const uint32_t* timeout_micros,
           signal_poll_func_t signal_poll = nullptr);
int Wait64(volatile uint64_t* address, uint64_t expected, const uint32_t* timeout_micros,
           signal_poll_func_t signal_poll = nullptr);
int Wait64(volatile uint64_t* address, uint64_t expected, std::chrono::nanoseconds timeout,
           signal_poll_func_t signal_poll = nullptr);
int Wake(volatile void* address, int32_t count);

// Round 132 (KYTY_BISECT's syncwait:old): the waits of round 131 and before, on Windows (with
// KYTY_SYNC_WAIT=old they are used throughout).
void SetOldWaits(bool old);

} // namespace Libs::LibKernel::SyncOnAddress

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_SYNC_ON_ADDRESS_H_ */
