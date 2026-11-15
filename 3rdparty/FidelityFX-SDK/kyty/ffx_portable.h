#pragma once

// Kyty: the MSVC CRT names that FSR 3.1's portable code calls, for the builds outside Windows (Linux, macOS).
// 3rdparty/CMakeLists.txt force-includes this into every FidelityFX source there; the SDK files stay as vendored.

#include <algorithm>
#include <bit>    // ffx_util.h calls std::popcount without including <bit>
#include <cerrno>
#include <cmath>  // log2 (ffx_vk.cpp)
#include <codecvt>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring> // memset, memcpy
#include <cwchar>  // wcscmp, wcslen
#include <locale>  // std::wstring_convert (ffx_vk.cpp's UTF-8/UTF-16 fallback)

#ifndef _WIN32

// ffx_message.cpp's non-Windows branch uses it without ffx_util.h; the same definition as there.
#ifndef FFX_UNUSED
#define FFX_UNUSED(x) ((void)(x))
#endif

#ifndef _countof
#define _countof(a) (sizeof(a) / sizeof((a)[0]))
#endif

// Copies with truncation where MSVC's would fail; every call in the SDK copies a short literal name.
inline int wcscpy_s(wchar_t* dst, size_t size, const wchar_t* src)
{
	if (dst == nullptr || size == 0) return EINVAL;
	if (src == nullptr)
	{
		dst[0] = L'\0';
		return EINVAL;
	}
	const size_t len = std::wcslen(src);
	const size_t n   = len < size ? len : size - 1;
	std::wmemcpy(dst, src, n);
	dst[n] = L'\0';
	return len < size ? 0 : ERANGE;
}

template <size_t N>
inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t* src)
{
	return wcscpy_s(dst, N, src);
}

inline int strcpy_s(char* dst, size_t size, const char* src)
{
	if (dst == nullptr || size == 0) return EINVAL;
	if (src == nullptr)
	{
		dst[0] = '\0';
		return EINVAL;
	}
	const size_t len = std::strlen(src);
	const size_t n   = len < size ? len : size - 1;
	std::memcpy(dst, src, n);
	dst[n] = '\0';
	return len < size ? 0 : ERANGE;
}

template <typename... Args>
inline int sprintf_s(char* dst, size_t size, const char* format, Args... args)
{
	return std::snprintf(dst, size, format, args...);
}

#endif // _WIN32
