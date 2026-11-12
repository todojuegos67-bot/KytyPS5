#ifndef KYTY_COMMON_FP_ENV_H_
#define KYTY_COMMON_FP_ENV_H_

#include <cstdint>
#include <cstdlib>
#include <xmmintrin.h>

namespace Common {

// The floating-point control state a thread runs with: MXCSR (SSE: rounding, exception masks, flush-to-zero,
// denormals-are-zero) and the x87 control word.
struct FpEnv {
	uint32_t mxcsr;
	uint16_t control;
};

// The console's, as its threads and fibers start with it (the guest's code was built and tuned for it): every exception
// masked, round to nearest, flush-to-zero and denormals-are-zero (MXCSR 0x9fc0), x87 at 64-bit precision (0x037f).
// The host's (Windows: MXCSR 0x1f80, x87 0x027f) keeps denormals, which cost an Intel core a microcode assist (~100+
// cycles) per operation that makes or reads one.
inline constexpr FpEnv GuestFpEnv {0x9fc0, 0x037f};

// KYTY_GUEST_FP_ENV=0: guest threads keep the host's environment (as before 12 Nov 2026).
inline bool GuestFpEnvOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_GUEST_FP_ENV");
		return value == nullptr || *value == '\0' || value[0] != '0';
	}();
	return on;
}

inline FpEnv ReadFpEnv() {
	FpEnv env {};
	env.mxcsr = _mm_getcsr();
	asm volatile("fnstcw %0" : "=m"(env.control) : : "memory");
	return env;
}

inline void WriteFpEnv(FpEnv env) {
	_mm_setcsr(env.mxcsr);
	asm volatile("fldcw %0" : : "m"(env.control) : "memory");
}

// A thread about to run guest code: the console's environment (unless KYTY_GUEST_FP_ENV=0).
inline void EnterGuestFpEnv() {
	if (GuestFpEnvOn()) WriteFpEnv(GuestFpEnv);
}

} // namespace Common

#endif /* KYTY_COMMON_FP_ENV_H_ */
