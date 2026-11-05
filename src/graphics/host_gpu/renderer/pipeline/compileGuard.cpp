#include "graphics/host_gpu/renderer/pipeline/compileGuard.h"

#include "common/common.h"
#include "graphics/shader/recompiler/backend/spirv/HelperInlinePolicy.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
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
	std::atomic<int32_t>     dont_inline {-1}; // -1 = unknown (module not noted)
	std::atomic<uint32_t>    dumped {0};
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
	uint32_t dont_inline;
	bool     dumped;
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

void NoteModule(VkShaderModule module, uint64_t guest_hash, std::span<const uint32_t> spirv) {
	if (module == VK_NULL_HANDLE) return;
	// OpFunction (opcode 54): word 3 is the function control; bit 1 = DontInline.
	uint32_t functions = 0, dont_inline = 0;
	for (size_t i = 5; i < spirv.size();) {
		const uint32_t count = spirv[i] >> 16u;
		if (count == 0u) break;
		if ((spirv[i] & 0xffffu) == 54u && i + 3 < spirv.size()) {
			functions++;
			if ((spirv[i + 3] & 2u) != 0u) dont_inline++;
		}
		i += count;
	}
	if (functions > 1) {
		std::printf("HelperInline: module guest=%016" PRIx64 " words=%zu functions=%u DontInline=%u\n",
		            guest_hash, spirv.size(), functions, dont_inline);
	}
	static const bool dump = [] {
		const char* v = std::getenv("KYTY_DUMP_INFLIGHT");
		return v != nullptr && v[0] == '1';
	}();
	bool dumped = false;
	if (dump) {
		std::error_code ec;
		std::filesystem::create_directories("_shaderdump", ec);
		char name[96];
		std::snprintf(name, sizeof(name), "_shaderdump/mod_%016" PRIx64 "_%zu.spv", guest_hash, spirv.size());
		if (std::FILE* f = std::fopen(name, "wb"); f != nullptr) {
			dumped = std::fwrite(spirv.data(), 4, spirv.size(), f) == spirv.size();
			std::fclose(f);
		}
	}
	std::lock_guard lock(ModulesMutex());
	Modules()[reinterpret_cast<uint64_t>(module)] = {guest_hash, spirv.size(), dont_inline, dumped};
}

Scope::Scope(const char* what, uint64_t key, VkShaderModule module, uint64_t guest_hash, uint64_t words) {
	int32_t  dont_inline = -1;
	uint32_t dumped      = 0;
	if (module != VK_NULL_HANDLE) {
		std::lock_guard lock(ModulesMutex());
		if (const auto it = Modules().find(reinterpret_cast<uint64_t>(module)); it != Modules().end()) {
			if (guest_hash == 0) guest_hash = it->second.guest;
			if (words == 0) words = it->second.words;
			dont_inline = static_cast<int32_t>(it->second.dont_inline);
			dumped      = it->second.dumped ? 1u : 0u;
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
			g_slots[i].dont_inline.store(dont_inline);
			g_slots[i].dumped.store(dumped);
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
		            " words=%" PRIu64 " DontInline functions=%d for %" PRId64 " ms%s\n",
		            tid, tid == self ? " (faulting thread)" : "", what != nullptr ? what : "?",
		            slot.key.load(), guest, slot.words.load(), slot.dont_inline.load(),
		            now - slot.start_ms.load(),
		            slot.dumped.load() != 0u ? " (SPIR-V in _shaderdump/mod_<guest>_<words>.spv)" : "");
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
