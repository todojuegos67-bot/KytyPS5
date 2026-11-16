// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// bb-precompile: removes "Compiling graphics pipeline" hitches for first-time players.
//
//  * Shipped cache: bb-shadercache.bin (next to the exe, or BB_SHADERCACHE=<path>) holds the
//    shader/pipeline cache (guest shader meta + pipeline keys + translated SPIR-V) per GPU
//    profile. On boot its entries that the user cache lacks are copied into
//    user/cache/<serial>/. On close it is (re)written when BB_SHADERCACHE_EXPORT=1, or
//    automatically when the shipped file is missing a section for this GPU profile / was made
//    by an older cache version (so the shipped cache regenerates after version bumps).
//  * Startup precompilation: every cached pipeline is built into a VkPipeline on worker threads
//    at boot (the game thread does not wait); the driver VkPipelineCache is persisted too.
//
// BB_PRECOMPILE=0 disables all of it (default on). Implies the pipeline cache is enabled even
// when pipelineCacheEnable=false in config.toml (directory mode only for the shipped cache).

#include <filesystem>
#include <functional>
#include <span>

#include "common/types.h"

namespace BBPrecompile {

bool Enabled();
// Worker thread count for the startup precompile (BB_PRECOMPILE_THREADS, default cores/2, 1..8).
u32 NumThreads();

struct CacheVersions {
    u32 shader_binary;
    u32 shader_meta;
    u32 pipeline_key;
    bool operator==(const CacheVersions&) const = default;
};

using ProfileMatcher = std::function<bool(std::span<const u8>)>;

// Called from WarmUp once the user cache directory is open. Copies the shipped entries for this
// GPU profile that the cache directory lacks. Returns the number of files imported.
u32 ImportShipped(const std::filesystem::path& cache_dir, std::span<const u8> profile,
                  const CacheVersions& versions, const ProfileMatcher& matches);

// Called from Storage::DataBase::Close after all pending cache writes are flushed.
void OnCacheClosed();

// Writes the driver VkPipelineCache blob atomically (tmp + rename).
bool WriteFileAtomic(const std::filesystem::path& path, std::span<const u8> data,
                     const char* tmp_suffix);

} // namespace BBPrecompile
