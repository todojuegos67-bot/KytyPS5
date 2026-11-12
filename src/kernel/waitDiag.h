#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_WAIT_DIAG_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_WAIT_DIAG_H_

#include <cstdint>
#include <string_view>

// Guest CPU wait diagnostics (Gran Turismo 7, or KYTY_CPU_WAIT_DIAG=1): every 5 s the guest threads
// blocked for more than 5 s in an event queue, semaphore, event flag, condition variable or
// address wait are printed and logged (what, its object, its name, since when, the timeout), so a
// black screen with the GPU idle names what the game waits for.
namespace Libs::LibKernel::WaitDiag {

class Scope {
public:
	// `timeout_us` < 0: no timeout.
	Scope(const char* what, uint64_t object, std::string_view name, int64_t timeout_us);
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool m_active = false;
};

} // namespace Libs::LibKernel::WaitDiag

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_WAIT_DIAG_H_ */
