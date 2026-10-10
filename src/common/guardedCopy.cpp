#include "common/guardedCopy.h"

#if defined(_WIN32)

#include <windows.h>

extern "C" {
uint64_t   kyty_guarded_copy(void* destination, const void* source, uint64_t size);
extern const char kyty_guarded_copy_fault[];
extern const char kyty_guarded_copy_resume[];
}

// rep movsb: a fault leaves rip on the instruction, rcx the bytes not copied and rsi, rdi past the copied ones.
// Win64: destination rcx, source rdx, size r8 (the direction flag clear); rdi and rsi are the caller's, kept in r9
// and r10: a leaf function without a frame needs no unwind data.
__asm__(".text\n"
        ".globl kyty_guarded_copy\n"
        ".p2align 4\n"
        "kyty_guarded_copy:\n"
        "  movq %rdi, %r9\n"
        "  movq %rsi, %r10\n"
        "  movq %rcx, %rdi\n"
        "  movq %rdx, %rsi\n"
        "  movq %r8, %rcx\n"
        ".globl kyty_guarded_copy_fault\n"
        "kyty_guarded_copy_fault:\n"
        "  rep movsb\n"
        ".globl kyty_guarded_copy_resume\n"
        "kyty_guarded_copy_resume:\n"
        "  movq %rcx, %rax\n"
        "  movq %r9, %rdi\n"
        "  movq %r10, %rsi\n"
        "  ret\n");

namespace Common {

uint64_t GuardedCopy(void* destination, const void* source, uint64_t size) noexcept {
	return size == 0 ? 0 : kyty_guarded_copy(destination, source, size);
}

bool ResumeGuardedCopy(void* context) noexcept {
	auto* registers = static_cast<CONTEXT*>(context);
	if (registers->Rip != reinterpret_cast<DWORD64>(kyty_guarded_copy_fault)) {
		return false;
	}
	registers->Rip = reinterpret_cast<DWORD64>(kyty_guarded_copy_resume);
	return true;
}

} // namespace Common

#endif
