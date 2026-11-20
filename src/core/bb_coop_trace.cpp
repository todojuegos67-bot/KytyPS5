// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// (bb-coop) Trace of Bloodborne 1.09's HTTP manager, to find why the ss.info request is
// created but never sent. Installed only with BB_COOP=1 on CUSA03173; observer hooks only.
//
// Eboot code (vaddr; Ghidra = vaddr + 0x400000), from the disassembly:
//   0xfd10d0  HTTP manager tick: sceHttpWaitRequest(eh, ev, 128, 0) on every epoll, then the
//             request update below on every live request.
//   0xfd2970  request update: state [this+0x84] 0/1 -> sceHttpSendRequest (call at 0xfd2ac4).
//   0xfd2740  request setup: CreateConnectionWithURL ... SetEpoll (the calls seen in the log).
//   0x1e7df20 / 0x1fe4d70  the two functions that call the tick (0x1e7e2f0, 0x1fe4d96).

#include <array>
#include <atomic>
#include <cstring>

#include "common/arch.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/thread.h"
#include "core/bb_coop_config.h"
#include "core/bb_coop_trace.h"
#ifdef ARCH_X86_64
#include "core/cpu_patches.h"
#endif

namespace Core::BBCoop {
#ifdef ARCH_X86_64
namespace {

struct Site {
    const char* name;
    u64 offset;
};
constexpr std::array<Site, 5> kSites{{
    {"http-tick", 0xfd10d0},
    {"http-request-update", 0xfd2970},
    {"http-request-setup", 0xfd2740},
    {"tick-caller-a", 0x1e7df20},
    {"tick-caller-b", 0x1fe4d70},
}};
// push rbp; mov rbp,rsp; push r15; push r14 (all five sites)
constexpr std::array<u8, 8> kPrologue{0x55, 0x48, 0x89, 0xE5, 0x41, 0x57, 0x41, 0x56};

uintptr_t g_base = 0;
u64 g_size = 0;
std::array<std::atomic<u64>, kSites.size()> g_calls{};
std::atomic<int> g_update_logs{0};

void PS4_SYSV_ABI OnSite(u64 tag, const Core::GuestRegisterSnapshot* regs) {
    if (tag >= kSites.size()) {
        return;
    }
    const u64 n = g_calls[tag].fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1 || ((n & (n - 1)) == 0 && n >= 1024)) {
        u64 ret = 0;
        std::memcpy(&ret, reinterpret_cast<const void*>(regs->rsp), sizeof(ret));
        const u64 ret_off = ret >= g_base && ret - g_base < g_size ? ret - g_base : 0;
        LOG_INFO(Lib_Http, "[coop-http] {} call #{} on thread '{}' from eboot+{:#x}",
                 kSites[tag].name, n, Common::GetCurrentThreadName(), ret_off);
    }
    if (tag == 1 && regs->rdi > 0x10000 && g_update_logs.load() < 40) {
        s32 state = 0, req = 0;
        std::memcpy(&state, reinterpret_cast<const void*>(regs->rdi + 0x84), sizeof(state));
        std::memcpy(&req, reinterpret_cast<const void*>(regs->rdi + 0x8c), sizeof(req));
        static std::atomic<s64> last{-1};
        const s64 key = (static_cast<s64>(req) << 8) | (state & 0xff);
        if (last.exchange(key) != key) {
            g_update_logs.fetch_add(1);
            LOG_INFO(Lib_Http, "[coop-http] request update: reqId={} state={}", req, state);
        }
    }
}

} // namespace

void InstallHttpTrace(uintptr_t base, u64 size) {
    if (!IsEnabled() || MemoryPatcher::g_game_serial != "CUSA03173") {
        return;
    }
    g_base = base;
    g_size = size;
    int installed = 0;
    for (size_t i = 0; i < kSites.size(); ++i) {
        const auto& s = kSites[i];
        if (s.offset + kPrologue.size() > size ||
            std::memcmp(reinterpret_cast<const void*>(base + s.offset), kPrologue.data(),
                        kPrologue.size()) != 0) {
            LOG_WARNING(Lib_Http, "[coop-http] {} at {:#x}: bytes differ, not traced", s.name,
                        s.offset);
            continue;
        }
        if (Core::InstallGuestCodeHook(reinterpret_cast<void*>(base + s.offset), kPrologue, i,
                                       OnSite)) {
            ++installed;
        }
    }
    LOG_INFO(Lib_Http, "[coop-http] HTTP manager trace: {}/{} sites hooked", installed,
             kSites.size());
}
#else
void InstallHttpTrace(uintptr_t, u64) {}
#endif

} // namespace Core::BBCoop
