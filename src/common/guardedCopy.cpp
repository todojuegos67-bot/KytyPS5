#include "common/guardedCopy.h"

#if defined(_WIN32)

#include "common/hostException.h"

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

// (fix7) Before HostException::InstallHandler nothing resumes a fault at kyty_guarded_copy_fault (a thread such as
// the RAM line can read the game's image while the start waits, e.g. KYTY_TIRON6's warm-up): copy only the readable
// prefix VirtualQuery reports, the same "bytes not copied" contract.
static uint64_t ReadablePrefix(const void* source, uint64_t size) noexcept {
	auto       at  = reinterpret_cast<uintptr_t>(source);
	const auto end = at + size;
	while (at < end) {
		MEMORY_BASIC_INFORMATION mbi {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT ||
		    (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0 || mbi.Protect == 0) {
			break;
		}
		at = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
	}
	return (at < end ? at : end) - reinterpret_cast<uintptr_t>(source);
}

uint64_t GuardedCopy(void* destination, const void* source, uint64_t size) noexcept {
	if (size == 0) return 0;
	if (!HostException::HandlerInstalled()) {
		const uint64_t readable = ReadablePrefix(source, size);
		return readable == 0 ? size : (size - readable) + kyty_guarded_copy(destination, source, readable);
	}
	return kyty_guarded_copy(destination, source, size);
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

#else

#include <fcntl.h>
#include <unistd.h>

namespace Common {

uint64_t GuardedCopy(void* destination, const void* source, uint64_t size) noexcept {
	if (size == 0) {
		return 0;
	}
	// /proc/self/mem serves this process's whole address space: a read of an unmapped page fails with EIO and a read
	// that spans into one returns a short count, so what was copied is exactly what was readable where it started.
	static const int mem = ::open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
	if (mem < 0) {
		return size;
	}
	auto*       dst    = static_cast<uint8_t*>(destination);
	const auto* src    = static_cast<const uint8_t*>(source);
	uint64_t    copied = 0;
	while (copied < size) {
		const auto done = ::pread(mem, dst + copied, static_cast<size_t>(size - copied),
		                          static_cast<off_t>(reinterpret_cast<uintptr_t>(src + copied)));
		if (done <= 0) {
			break;
		}
		copied += static_cast<uint64_t>(done);
	}
	return size - copied;
}

} // namespace Common

#endif
