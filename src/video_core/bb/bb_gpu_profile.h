// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-gpu-prof: GPU timestamp profiling per pass (BB_GPU_PROFILE=1, default off).
//
// Every scope writes two Vulkan timestamps (stage ALL_COMMANDS: the begin stamp is taken when
// the work recorded before it has finished, the end stamp when the scope's own work has), so the
// durations are the scopes' serialized GPU times; nested scopes are never used (the game's
// render passes, compute dispatches, the emulator's copies and our own passes are disjoint).
// Results are read without waiting, several frames later (host query reset, VK 1.2), and summed
// per label; once a second one "[gpu-prof]" line lists the labels sorted by cost in ms per guest
// frame, plus the GPU busy time of the submits themselves ("submit:" envelopes).
//
// Off (BB_GPU_PROFILE unset or 0): On() is one cached bool, Begin() returns 0 and End(0) does
// nothing; no query pool is created. Work recorded inside a game pass (objmv2's extra velocity
// work in the game's own draws) is part of that pass's time: measure it by A/B (BB_OBJMV2=0).
// The timings are serialized (no overlap between scopes), so their sum can exceed what the GPU
// spends when passes overlap; the "submit" envelopes are the GPU's real busy time.

#pragma once

#include <string>
#include <string_view>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace BB::GpuProf {

enum class Kind : u8 {
    Pass = 0,     // a timed piece of work (summed into "measured")
    Envelope = 1, // a whole command buffer (submit): GPU busy time, listed apart
    Group = 2,    // a sum of other labels (e.g. every pass into the same targets)
};

/// BB_GPU_PROFILE >= 1 (read once).
bool On();
/// BB_GPU_PROFILE value: 0 off, 1 per pass, 2 submits only, 3 per pass with a drain barrier
/// before each scope (exact own cost, serialized GPU).
int Level();

/// Creates the query pool (once; needs hostQueryReset). Safe to call repeatedly.
void Init(const Vulkan::Instance& instance);
/// True when profiling is active and queue family `family` supports timestamps (a command
/// buffer submitted to another queue than the graphics one, e.g. FFX's async compute queue).
bool FamilyHasTimestamps(u32 family);

/// A label id for a fixed name (cache it in a function-local static at the call site).
u32 Label(std::string_view name, Kind kind = Kind::Pass);
/// A label id for a dynamic key; `make` builds the name the first time the key is seen.
/// `group` (optional): another label the time is also summed into ("by target" view).
template <typename F>
u32 KeyLabel(u64 key, F&& make, u32 group = 0);
u32 KeyLookup(u64 key);
u32 KeyInsert(u64 key, const std::string& name, u32 group);

/// Starts a scope: one timestamp into `cmd`. Returns a token (0: not timed).
u32 Begin(vk::CommandBuffer cmd, u32 label);
/// Ends the scope of `token` (0: nothing).
void End(vk::CommandBuffer cmd, u32 token);

/// Once per guest frame (the presenter's flip): reads finished results, logs once a second.
void Frame();

struct Scope {
    Scope(vk::CommandBuffer cmd_, u32 label) : cmd{cmd_}, token{On() ? Begin(cmd_, label) : 0u} {}
    ~Scope() {
        if (token != 0) End(cmd, token);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    vk::CommandBuffer cmd;
    u32 token;
};

template <typename F>
u32 KeyLabel(u64 key, F&& make, u32 group) {
    if (const u32 id = KeyLookup(key); id != 0) return id;
    return KeyInsert(key, make(), group);
}

} // namespace BB::GpuProf

// One scope with a fixed name: BB_GPU_SCOPE(cmd, "name"); the label is interned once.
#define BB_GPU_SCOPE_CAT2(a, b) a##b
#define BB_GPU_SCOPE_CAT(a, b) BB_GPU_SCOPE_CAT2(a, b)
#define BB_GPU_SCOPE(cmd, name)                                                                    \
    static const u32 BB_GPU_SCOPE_CAT(bb_gp_label_, __LINE__) = ::BB::GpuProf::Label(name);        \
    const ::BB::GpuProf::Scope BB_GPU_SCOPE_CAT(bb_gp_scope_, __LINE__)(                           \
        (cmd), BB_GPU_SCOPE_CAT(bb_gp_label_, __LINE__))
