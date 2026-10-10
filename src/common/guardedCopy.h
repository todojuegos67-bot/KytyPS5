#ifndef KYTY_COMMON_GUARDED_COPY_H_
#define KYTY_COMMON_GUARDED_COPY_H_

#include <cstdint>

namespace Common {

#if defined(_WIN32)
// A copy that stops at the first byte it cannot read or write: the access violation resumes after the copy
// (hostException.cpp asks ResumeGuardedCopy before the guest-memory fault handler). Returns the bytes not copied:
// 0 when all were, else the copy ended at byte `size` - returned (a page without access there).
uint64_t GuardedCopy(void* destination, const void* source, uint64_t size) noexcept;
// The vectored exception handler's check (`context`: the fault's CONTEXT): true when the access violation is
// GuardedCopy's, with the context moved past its copy.
bool ResumeGuardedCopy(void* context) noexcept;
#endif

} // namespace Common

#endif /* KYTY_COMMON_GUARDED_COPY_H_ */
