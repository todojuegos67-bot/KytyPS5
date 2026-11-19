// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Read-only network view for the F1 menu's MULTIJUGADOR card (Bloodborne co-op).

#include <cstring>
#include <mutex>

#include "core/libraries/np/np_handler/np_handler.h"
#include "core/libraries/np/np_types2.h"

namespace Libraries::Np {

NpHandler::CoopSnapshot NpHandler::GetCoopSnapshot() const {
    CoopSnapshot out;
    out.server = static_cast<CoopServerState>(m_coop_server_state.load());
    out.error_code = m_coop_error.load();
    const auto id_string = [](const OrbisNpOnlineId& id) {
        return std::string(id.data, strnlen(id.data, ORBIS_NP_ONLINEID_MAX_LENGTH));
    };
    {
        std::lock_guard lock(m_mutex_clients);
        for (const auto& [uid, np_id] : m_np_ids) {
            if (m_clients.contains(uid)) {
                out.npid = id_string(np_id.handle);
                break;
            }
        }
    }
    std::vector<std::shared_ptr<NpMatching2::Matching2ContextCache>> caches;
    {
        std::lock_guard lock(m_mutex_matching2_cache);
        for (const auto& [ctx, cache] : m_matching2_cache) {
            if (cache) {
                caches.push_back(cache);
            }
        }
    }
    for (const auto& cache : caches) {
        std::scoped_lock lock(cache->mutex);
        for (const auto& [room_id, room] : cache->rooms) {
            out.in_room = true;
            out.is_owner = out.is_owner || room.owner;
            for (const auto& [member_id, member] : room.members) {
                const std::string name = id_string(member.np_id.handle);
                if (name.empty() || name == out.npid) {
                    continue;
                }
                CoopPeer peer;
                peer.name = name;
                peer.is_owner = (member.flag_attr &
                                 NpMatching2::ORBIS_NP_MATCHING2_ROOMMEMBER_FLAG_ATTR_OWNER) != 0;
                if (const auto it = cache->peers.find(member_id); it != cache->peers.end()) {
                    peer.active = it->second.status != 0;
                    if (it->second.ping_us != 0) {
                        peer.ping_ms = static_cast<s32>(it->second.ping_us / 1000);
                    }
                }
                out.peers.push_back(std::move(peer));
            }
        }
    }
    return out;
}

} // namespace Libraries::Np
