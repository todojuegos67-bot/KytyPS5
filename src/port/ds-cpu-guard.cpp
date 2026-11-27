// (11-27) CPUs without AVX2 (Intel before Haswell, AMD before Excavator): "Demon's Souls.exe" is built with
// -march=x86-64-v3 and would die with 0xc000001d. This file is compiled at plain x86-64 (no LTO, see CMakeLists)
// and runs as the exe's FIRST TLS callback (.CRT$XLB: before the CRT's thread_local init at .CRT$XLC, the C++
// static initializers and main), so nothing built at v3 has run yet. Without the v3 features it starts
// "Demon's Souls-compat.exe" (same folder, built at x86-64-v2) with the same command line, waits for it and exits
// with its code (Steam and the launcher see one process). With AVX2 it returns at once: zero cost at runtime.
// Only Win32 calls and inline asm: no CRT, no intrinsics that could pull v3 code.
#if defined(_WIN32) && defined(KYTY_DS_CPU_GUARD)

#include <windows.h>

namespace {

void Cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{
	unsigned a, b, c, d;
	__asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(leaf), "c"(sub));
	r[0] = a; r[1] = b; r[2] = c; r[3] = d;
}

// x86-64-v3 = AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, XSAVE (+ the OS saving the YMM state).
bool HasV3()
{
	unsigned r[4];
	Cpuid(0, 0, r);
	if (r[0] < 7) return false;
	Cpuid(1, 0, r);
	const unsigned c1 = r[2];
	const unsigned need1 = (1u << 12) | (1u << 22) | (1u << 26) | (1u << 27) | (1u << 28) | (1u << 29); // FMA MOVBE XSAVE OSXSAVE AVX F16C
	if ((c1 & need1) != need1) return false;
	unsigned lo, hi;
	__asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
	if ((lo & 6u) != 6u) return false; // XMM + YMM state enabled by the OS
	Cpuid(7, 0, r);
	const unsigned need7 = (1u << 3) | (1u << 5) | (1u << 8); // BMI1 AVX2 BMI2
	if ((r[1] & need7) != need7) return false;
	Cpuid(0x80000000u, 0, r);
	if (r[0] < 0x80000001u) return false;
	Cpuid(0x80000001u, 0, r);
	return (r[2] & (1u << 5)) != 0; // LZCNT
}

void NTAPI CpuGuard(PVOID /*module*/, DWORD reason, PVOID /*reserved*/)
{
	if (reason != DLL_PROCESS_ATTACH || HasV3()) return;

	wchar_t exe[MAX_PATH * 2];
	DWORD   n = GetModuleFileNameW(nullptr, exe, MAX_PATH * 2);
	if (n == 0 || n >= MAX_PATH * 2 - 32) TerminateProcess(GetCurrentProcess(), 0xc000001d);
	DWORD slash = n;
	while (slash > 0 && exe[slash - 1] != L'\\' && exe[slash - 1] != L'/') slash--;
	const wchar_t* leaf = L"Demon's Souls-compat.exe";
	DWORD          k    = 0;
	for (; leaf[k] != 0; k++) exe[slash + k] = leaf[k];
	exe[slash + k] = 0;

	// (A loop guard: the compat exe never has this file active, but an exe renamed by hand could.)
	if (GetEnvironmentVariableW(L"KYTY_DS_COMPAT_CHILD", nullptr, 0) != 0 ||
	    GetFileAttributesW(exe) == INVALID_FILE_ATTRIBUTES)
	{
		MessageBoxW(nullptr,
		            L"Tu procesador no tiene AVX2 y falta \"Demon's Souls-compat.exe\" junto al juego.\n"
		            L"Your CPU has no AVX2 and \"Demon's Souls-compat.exe\" is missing next to the game.",
		            L"Demon's Souls", MB_OK | MB_ICONERROR);
		TerminateProcess(GetCurrentProcess(), 0xc000001d);
	}
	SetEnvironmentVariableW(L"KYTY_DS_COMPAT_CHILD", L"1");

	STARTUPINFOW si {};
	si.cb = sizeof(si);
	PROCESS_INFORMATION pi {};
	// The same command line (argv[0] is ignored by the child's CRT for the program path: lpApplicationName wins).
	if (!CreateProcessW(exe, GetCommandLineW(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi))
		TerminateProcess(GetCurrentProcess(), GetLastError());
	CloseHandle(pi.hThread);
	WaitForSingleObject(pi.hProcess, INFINITE);
	DWORD code = 0;
	GetExitCodeProcess(pi.hProcess, &code);
	// TerminateProcess, not ExitProcess: we are inside the loader (TLS callback); no DLL detach, no CRT.
	TerminateProcess(GetCurrentProcess(), code);
}

} // namespace

#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:kyty_ds_cpu_guard_cb")
#pragma section(".CRT$XLB", long, read)
extern "C" __declspec(allocate(".CRT$XLB")) const PIMAGE_TLS_CALLBACK kyty_ds_cpu_guard_cb;
extern "C" __declspec(allocate(".CRT$XLB")) const PIMAGE_TLS_CALLBACK kyty_ds_cpu_guard_cb = CpuGuard;

#endif
