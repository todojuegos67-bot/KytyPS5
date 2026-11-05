#include "graphics/host_gpu/renderer/pipeline/compileGuard.h"

#include "common/common.h"
#include "graphics/shader/recompiler/backend/spirv/HelperInlinePolicy.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#endif

namespace Libs::Graphics::CompileGuard {
namespace {

constexpr int kSlots = 64;

struct Slot {
	std::atomic<uint32_t>    busy {0};
	std::atomic<uint64_t>    thread {0};
	std::atomic<uint64_t>    key {0};
	std::atomic<uint64_t>    guest {0};
	std::atomic<uint64_t>    words {0};
	std::atomic<int64_t>     start_ms {0};
	std::atomic<const char*> what {nullptr};
};

Slot g_slots[kSlots];

uint64_t ThreadId() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return GetCurrentThreadId();
#else
	return std::hash<std::thread::id> {}(std::this_thread::get_id());
#endif
}

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

struct ModuleInfo {
	uint64_t guest;
	uint64_t words;
};

std::mutex& ModulesMutex() {
	static std::mutex m;
	return m;
}
std::unordered_map<uint64_t, ModuleInfo>& Modules() {
	static std::unordered_map<uint64_t, ModuleInfo> m;
	return m;
}

} // namespace

void NoteModule(VkShaderModule module, uint64_t guest_hash, uint64_t words) {
	if (module == VK_NULL_HANDLE) return;
	std::lock_guard lock(ModulesMutex());
	Modules()[reinterpret_cast<uint64_t>(module)] = {guest_hash, words};
}

Scope::Scope(const char* what, uint64_t key, VkShaderModule module, uint64_t guest_hash, uint64_t words) {
	if (module != VK_NULL_HANDLE && (guest_hash == 0 || words == 0)) {
		std::lock_guard lock(ModulesMutex());
		if (const auto it = Modules().find(reinterpret_cast<uint64_t>(module)); it != Modules().end()) {
			if (guest_hash == 0) guest_hash = it->second.guest;
			if (words == 0) words = it->second.words;
		}
	}
	for (int i = 0; i < kSlots; i++) {
		uint32_t expected = 0;
		if (g_slots[i].busy.compare_exchange_strong(expected, 2u)) {
			g_slots[i].thread.store(ThreadId());
			g_slots[i].key.store(key);
			g_slots[i].guest.store(guest_hash);
			g_slots[i].words.store(words);
			g_slots[i].start_ms.store(NowMs());
			g_slots[i].what.store(what);
			g_slots[i].busy.store(1u, std::memory_order_release);
			m_slot = i;
			return;
		}
	}
}

Scope::~Scope() {
	if (m_slot >= 0) g_slots[m_slot].busy.store(0u, std::memory_order_release);
}

int ReportAndBlacklist() {
	const uint64_t self  = ThreadId();
	const int64_t  now   = NowMs();
	int            count = 0;
	uint64_t       guests[kSlots];
	int            nguests = 0;
	bool           self_open = false;
	for (auto& slot: g_slots) {
		if (slot.busy.load(std::memory_order_acquire) != 1u) continue;
		const uint64_t tid   = slot.thread.load();
		const uint64_t guest = slot.guest.load();
		const char*    what  = slot.what.load();
		std::printf("compile in flight: thread %" PRIu64 "%s %s key=%016" PRIx64 " guest=%016" PRIx64
		            " words=%" PRIu64 " for %" PRId64 " ms\n",
		            tid, tid == self ? " (faulting thread)" : "", what != nullptr ? what : "?",
		            slot.key.load(), guest, slot.words.load(), now - slot.start_ms.load());
		count++;
		if (tid == self) self_open = true;
		if (guest != 0) guests[nguests++] = guest;
	}
	if (count == 0) {
		std::printf("compile in flight: none\n");
	}
	// Blacklist the faulting thread's shader when it is the one compiling; otherwise every shader
	// in flight (the fault may be on a driver worker thread).
	if (nguests > 0) {
		if (std::FILE* f = std::fopen(ShaderRecompiler::HelperInlinePolicy::kBlacklistPath, "a"); f != nullptr) {
			for (auto& slot: g_slots) {
				if (slot.busy.load(std::memory_order_acquire) != 1u) continue;
				const uint64_t guest = slot.guest.load();
				if (guest == 0 || (self_open && slot.thread.load() != self)) continue;
				std::fprintf(f, "guest %016" PRIx64 " words=%" PRIu64 " key=%016" PRIx64 "\n", guest,
				             slot.words.load(), slot.key.load());
				std::printf("compile in flight: guest %016" PRIx64 " added to %s (next run inlines its helpers)\n",
				            guest, ShaderRecompiler::HelperInlinePolicy::kBlacklistPath);
			}
			std::fclose(f);
		}
	}
	std::fflush(stdout);
	return count;
}

void DescribeAddress(uint64_t address, char* out, unsigned size) {
	if (size == 0) return;
	out[0] = '\0';
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HMODULE module = nullptr;
	if (address == 0 ||
	    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(address), &module) == 0 ||
	    module == nullptr) {
		return;
	}
	char path[MAX_PATH] {};
	if (GetModuleFileNameA(module, path, MAX_PATH) == 0) return;
	const char* name = path;
	for (const char* p = path; *p != '\0'; p++) {
		if (*p == '\\' || *p == '/') name = p + 1;
	}
	std::snprintf(out, size, "%s+0x%" PRIx64, name, address - reinterpret_cast<uint64_t>(module));
#else
	(void)address;
#endif
}

} // namespace Libs::Graphics::CompileGuard
