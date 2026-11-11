#include "graphics/host_gpu/hostMemory.h"

#include <cinttypes>
#include <cstdio>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <atomic>
#include <cerrno>
#include <sys/uio.h>
#include <unistd.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace Libs::Graphics {
namespace {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool IsAccessible(DWORD protect, HostMemoryAccess access) {
	constexpr DWORD blocked  = PAGE_NOACCESS | PAGE_GUARD;
	constexpr DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
	                           PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
	if (access == HostMemoryAccess::Mapped) {
		return (protect & PAGE_GUARD) == 0;
	}
	return (protect & blocked) == 0 && (protect & readable) != 0;
}
#endif

} // namespace

bool HostMemoryQueryRange(uint64_t addr, uint64_t requested_size, HostMemoryAccess access,
                          uint64_t& accessible_size) {
	accessible_size = 0;
	if (addr == 0 || requested_size == 0) {
		return false;
	}

	const auto end     = UINT64_MAX - addr < requested_size ? UINT64_MAX : addr + requested_size;
	uint64_t   current = addr;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	while (current < end) {
		MEMORY_BASIC_INFORMATION region {};
		if (::VirtualQuery(reinterpret_cast<const void*>(static_cast<uintptr_t>(current)), &region,
		                   sizeof(region)) == 0) {
			break;
		}
		const auto begin  = reinterpret_cast<uint64_t>(region.BaseAddress);
		auto       finish = begin + region.RegionSize;
		if (finish < begin) {
			finish = UINT64_MAX;
		}
		if (finish <= current || (region.State & MEM_COMMIT) == 0 ||
		    !IsAccessible(region.Protect, access)) {
			break;
		}
		current = finish < end ? finish : end;
	}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// process_vm_readv can be refused by a seccomp filter (sandboxes, some containers): from then on, /proc/self/maps.
	static std::atomic_bool vm_readv_unavailable {false};
	if (access == HostMemoryAccess::Read && !vm_readv_unavailable.load(std::memory_order_relaxed)) {
		// One byte of each page through process_vm_readv on this process: the kernel checks the mappings (no
		// fault handler, no signal) and stops at the first page it cannot read. Reading /proc/self/maps at each
		// call, as below, took most of the render thread's time with KYTY_RENDER_ASPECT (src/local/hud-anchor.h
		// asks several times a draw; ~7400 mappings: 1-5 fps).
		static const uint64_t page  = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
		constexpr int         batch = 256;
		const pid_t           self  = getpid();
		unsigned char         sink[batch];
		struct iovec          local[batch];
		struct iovec          remote[batch];
		while (current < end) {
			int      count = 0;
			uint64_t probe = current;
			while (count < batch && probe < end) {
				local[count]  = {&sink[count], 1};
				remote[count] = {reinterpret_cast<void*>(static_cast<uintptr_t>(probe)), 1};
				count++;
				const uint64_t next = (probe & ~(page - 1)) + page;
				probe               = next > probe ? next : end; // wrapped at the top of the address space
			}
			const auto copied = process_vm_readv(self, local, static_cast<unsigned long>(count), remote,
			                                     static_cast<unsigned long>(count), 0);
			if (copied < 0 && errno != EFAULT) {
				vm_readv_unavailable.store(true, std::memory_order_relaxed);
				break;
			}
			if (copied <= 0) {
				break;
			}
			if (copied < count) {
				current = reinterpret_cast<uint64_t>(remote[copied].iov_base);
				break;
			}
			current = probe < end ? probe : end;
		}
		if (!vm_readv_unavailable.load(std::memory_order_relaxed)) {
			accessible_size = current - addr;
			return accessible_size != 0;
		}
		current = addr;
	}
	auto* maps = std::fopen("/proc/self/maps", "r");
	if (maps == nullptr) {
		return false;
	}
	char line[1024] = {};
	while (current < end && std::fgets(line, sizeof(line), maps) != nullptr) {
		uint64_t begin          = 0;
		uint64_t finish         = 0;
		char     permissions[5] = {};
		if (std::sscanf(line, "%" SCNx64 "-%" SCNx64 " %4s", &begin, &finish, permissions) != 3 ||
		    finish <= current) {
			continue;
		}
		if (begin > current) {
			break;
		}
		const bool allowed = access == HostMemoryAccess::Mapped || permissions[0] == 'r';
		if (!allowed) {
			break;
		}
		current = finish < end ? finish : end;
	}
	std::fclose(maps);
#else
	(void)access;
#endif

	accessible_size = current - addr;
	return accessible_size != 0;
}

bool HostMemoryQueryReadable(uint64_t addr, uint64_t requested_size, uint64_t& readable_size) {
	return HostMemoryQueryRange(addr, requested_size, HostMemoryAccess::Read, readable_size);
}

bool HostMemoryIsReadable(uint64_t addr) {
	return HostMemoryRangeIsReadable(addr, 1);
}

bool HostMemoryRangeIsReadable(uint64_t addr, uint64_t size) {
	if (addr == 0 || size == 0 || UINT64_MAX - addr < size) {
		return false;
	}
	uint64_t readable_size = 0;
	return HostMemoryQueryReadable(addr, size, readable_size) && readable_size >= size;
}

} // namespace Libs::Graphics
