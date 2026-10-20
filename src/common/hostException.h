#ifndef KYTY_COMMON_HOST_EXCEPTION_H_
#define KYTY_COMMON_HOST_EXCEPTION_H_

#include "common/common.h"

namespace Common::HostException {

enum class ExceptionType { Unknown, AccessViolation, IllegalInstruction };

enum class AccessViolationType { Unknown, Read, Write, Execute };

struct ExceptionInfo {
	ExceptionType       type                   = ExceptionType::Unknown;
	AccessViolationType access_violation_type  = AccessViolationType::Unknown;
	uint64_t            access_violation_vaddr = 0;
	uint64_t            exception_address      = 0;
	uint64_t            rax                    = 0;
	uint64_t            rbx                    = 0;
	uint64_t            rcx                    = 0;
	uint64_t            rdx                    = 0;
	uint64_t            rsi                    = 0;
	uint64_t            rdi                    = 0;
	uint64_t            rbp                    = 0;
	uint64_t            rsp                    = 0;
	uint64_t            r8                     = 0;
	uint64_t            r9                     = 0;
	uint64_t            r10                    = 0;
	uint64_t            r11                    = 0;
	uint64_t            r12                    = 0;
	uint64_t            r13                    = 0;
	uint64_t            r14                    = 0;
	uint64_t            r15                    = 0;
	uint32_t            native_code            = 0;
	// Platform-specific mutable context, valid only for the duration of the handler call.
	void* native_context = nullptr;
};

using Handler = bool (*)(const ExceptionInfo&);

bool InstallHandler(Handler handler);

// Guard-page violations (Windows): the touched address and the native context; true resumes the faulting thread.
using GuardPageHandler = bool (*)(uint64_t address, void* native_context);

void SetGuardPageHandler(GuardPageHandler handler);

// Windows, the texture offload (src/local/texture-offload.h): an access violation outside the guest's fault handling is
// first offered to this handler (the touched address, the faulting instruction, whether it wrote); true resumes the
// thread.
using ProbeHandler = bool (*)(uint64_t address, uint64_t rip, const void* native_context, bool write);

void SetProbeHandler(ProbeHandler handler);

// Windows: the stack range the TEB gives while guest code runs on a guest or fiber stack (RunOnGuestStack, RunEntry):
// the region every guest and fiber stack lies in (pthread.cpp PTHREAD_STACK_BOTTOM..TOP, MapGuestStack), where the
// thread's host stack lies above it, else none (zeros). With Control Flow Guard on for the process, ntdll checks that an
// exception's continuation (RtlGuardRestoreContext) has its rsp within the TEB's range and fails fast (0xC0000409, code
// 13) otherwise: every write-tracking fault on a guest thread ended the process - issue #56, a "lite" Windows that
// turns CFG on for all programs (reproduced with PROCESS_CREATION_MITIGATION_POLICY_CONTROL_FLOW_GUARD_ALWAYS_ON).
// __chkstk probes only below the range's low end: never, as with zeros. The kernel's stack growth needs the faulting
// address between the TEB's DeallocationStack and the range's top: never, with the host stack above it. Scratch
// registers: rcx, rdx and r<zero> (a number: "10").
inline constexpr uint64_t GuestStacksLow  = 0x0000040000ull;
inline constexpr uint64_t GuestStacksHigh = 0x07efff8000ull;
#define KYTY_GUEST_TEB_STACK_RANGE_ASM(zero)                                                                           \
	"xorq %%r" zero ", %%r" zero "\n\t"                                                                                 \
	"movq %%gs:0x1478, %%rcx\n\t"                                                                                       \
	"movabsq $0x7efff8000, %%rdx\n\t"                                                                                   \
	"cmpq %%rdx, %%rcx\n\t"                                                                                             \
	"cmovbq %%r" zero ", %%rdx\n\t"                                                                                     \
	"movq %%rdx, %%gs:0x08\n\t"                                                                                         \
	"movl $0x40000, %%ecx\n\t"                                                                                          \
	"testq %%rdx, %%rdx\n\t"                                                                                            \
	"cmovzq %%r" zero ", %%rcx\n\t"                                                                                     \
	"movq %%rcx, %%gs:0x10\n\t"

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
bool InitializeThreadSignalStack();
#endif

} // namespace Common::HostException

#endif /* KYTY_COMMON_HOST_EXCEPTION_H_ */
