#include "local/local-platform.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <softpub.h>
#include <wincrypt.h>
#include <wintrust.h>
#pragma comment(lib, "wintrust.lib")
#pragma comment(lib, "crypt32.lib")
#include <vector>
#else
#include <cstdio>
#include <unistd.h>
#endif

namespace LocalPlatform {

#if defined(_WIN32)

bool IsNvidiaSigned(const std::wstring& path) {
	WINTRUST_FILE_INFO file {};
	file.cbStruct      = sizeof(file);
	file.pcwszFilePath = path.c_str();
	WINTRUST_DATA data {};
	data.cbStruct            = sizeof(data);
	data.dwUIChoice          = WTD_UI_NONE;
	data.fdwRevocationChecks = WTD_REVOKE_NONE;
	data.dwUnionChoice       = WTD_CHOICE_FILE;
	data.pFile               = &file;
	data.dwStateAction       = WTD_STATEACTION_VERIFY;
	data.dwProvFlags         = WTD_CACHE_ONLY_URL_RETRIEVAL;
	GUID       action        = WINTRUST_ACTION_GENERIC_VERIFY_V2;
	const LONG status        = WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &data);
	bool       nvidia        = false;
	if (status == ERROR_SUCCESS) {
		auto* provider = WTHelperProvDataFromStateData(data.hWVTStateData);
		auto* signer   = provider != nullptr ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
		auto* cert     = signer != nullptr ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
		wchar_t name[256] {};
		if (cert != nullptr && cert->pCert != nullptr &&
		    CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name, 256) > 1)
			nvidia = std::wstring(name) == L"NVIDIA Corporation";
	}
	data.dwStateAction = WTD_STATEACTION_CLOSE;
	(void)WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &data);
	return nvidia;
}

MemoryStatus QueryMemory() {
	MemoryStatus    out {};
	MEMORYSTATUSEX  mem {};
	mem.dwLength = sizeof(mem);
	if (GlobalMemoryStatusEx(&mem) == 0) return out;
	out.total_phys = mem.ullTotalPhys;
	out.avail_phys = mem.ullAvailPhys;
	PERFORMANCE_INFORMATION perf {};
	perf.cb = sizeof(perf);
	if (GetPerformanceInfo(&perf, sizeof(perf)) != 0) {
		out.commit_limit = static_cast<uint64_t>(perf.CommitLimit) * perf.PageSize;
		out.commit_total = static_cast<uint64_t>(perf.CommitTotal) * perf.PageSize;
	}
	PROCESS_MEMORY_COUNTERS_EX counters {};
	counters.cb = sizeof(counters);
	if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
	                         sizeof(counters)) != 0) {
		out.private_bytes = counters.PrivateUsage;
		out.working_set   = counters.WorkingSetSize;
	}
	out.valid = true;
	return out;
}

bool LowMemorySignaled() {
	static HANDLE handle = CreateMemoryResourceNotification(LowMemoryResourceNotification);
	BOOL          low    = FALSE;
	return handle != nullptr && QueryMemoryResourceNotification(handle, &low) != 0 && low != FALSE;
}

uint64_t CompactHeaps() {
	std::vector<HANDLE> heaps(GetProcessHeaps(0, nullptr) + 16u);
	const DWORD         n       = GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data());
	uint64_t            largest = 0;
	for (DWORD i = 0; i < n && i < heaps.size(); ++i) largest += HeapCompact(heaps[i], 0);
	return largest;
}

#else

bool IsNvidiaSigned(const std::wstring& /*path*/) {
	return false;
}

MemoryStatus QueryMemory() {
	MemoryStatus out {};
	const long   pages = sysconf(_SC_PHYS_PAGES), avail = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGESIZE);
	if (pages <= 0 || size <= 0) return out;
	out.total_phys = static_cast<uint64_t>(pages) * static_cast<uint64_t>(size);
	out.avail_phys = static_cast<uint64_t>(avail) * static_cast<uint64_t>(size);
	if (FILE* f = std::fopen("/proc/self/statm", "r"); f != nullptr) {
		unsigned long total = 0, resident = 0;
		if (std::fscanf(f, "%lu %lu", &total, &resident) == 2) {
			out.private_bytes = static_cast<uint64_t>(total) * static_cast<uint64_t>(size);
			out.working_set   = static_cast<uint64_t>(resident) * static_cast<uint64_t>(size);
		}
		std::fclose(f);
	}
	out.commit_limit = out.total_phys;
	out.commit_total = out.total_phys - out.avail_phys;
	out.valid        = true;
	return out;
}

bool LowMemorySignaled() {
	return false;
}

uint64_t CompactHeaps() {
	return 0;
}

#endif

} // namespace LocalPlatform
