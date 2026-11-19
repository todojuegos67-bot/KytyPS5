// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/bb_coop_config.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "core/user_manager.h"
#include "core/user_settings.h"

namespace Core::BBCoop {
namespace {

bool EnvFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::string_view{value} != "0";
}

std::string EnvString(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string{value} : std::string{};
}

// Both layers: the base value and the game-specific one, so a per-game profile that sets the
// same key does not undo the launcher's choice. Neither is saved unless a settings UI saves.
template <typename Setter, typename T>
void SetBoth(Setter setter, const T& value) {
    (EmulatorSettings.*setter)(value, false);
    (EmulatorSettings.*setter)(value, true);
}

void ApplyAccount() {
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "bb_coop.json";
    std::ifstream file{path};
    if (!file.is_open()) {
        LOG_WARNING(Config, "BB co-op: {} not found; using the shadNet account in users.json",
                    Common::FS::PathToUTF8String(path));
        return;
    }
    std::string npid;
    std::string password;
    try {
        const auto json = nlohmann::json::parse(file);
        npid = json.value("npid", std::string{});
        password = json.value("password", std::string{});
    } catch (const std::exception&) {
        // The parser message can quote the file's text, so it is not logged.
        LOG_ERROR(Config, "BB co-op: bb_coop.json is not valid JSON; account not applied");
        return;
    }
    if (npid.empty() || password.empty()) {
        LOG_ERROR(Config, "BB co-op: bb_coop.json needs both \"npid\" and \"password\"");
        return;
    }
    User* user = UserManagement.GetUserByPlayerIndex(1);
    if (user == nullptr) {
        LOG_ERROR(Config, "BB co-op: no default user; account not applied");
        return;
    }
    user->shadnet_enabled = true;
    user->shadnet_npid = std::move(npid);
    user->shadnet_password = std::move(password);
    LOG_INFO(Config, "BB co-op: shadNet account from bb_coop.json given to user {}",
             user->user_id);
}

std::string HostOf(const std::string& server) {
    const auto colon = server.rfind(':');
    return colon == std::string::npos ? server : server.substr(0, colon);
}

} // namespace

bool IsEnabled() {
    static const bool enabled = EnvFlag("BB_COOP");
    return enabled;
}

bool IsSeamlessEnabled() {
    static const bool enabled =
        EnvFlag("BB_COOP_SEAMLESS") || EnvFlag("SHADPS4_BLOODBORNE_SEAMLESS_COOP");
    return enabled;
}

void ApplyToSettings() {
    if (!IsEnabled()) {
        return;
    }
    SetBoth(&EmulatorSettingsImpl::SetConnectedToNetwork, true);
    SetBoth(&EmulatorSettingsImpl::SetShadNetEnabled, true);
    SetBoth(&EmulatorSettingsImpl::SetUPnPEnabled, EnvFlag("BB_COOP_UPNP"));

    const std::string server = EnvString("BB_COOP_SERVER");
    if (!server.empty()) {
        SetBoth(&EmulatorSettingsImpl::SetShadNetServer, server);
    }
    std::string webapi = EnvString("BB_COOP_WEBAPI");
    if (webapi.empty() && !server.empty()) {
        webapi = "http://" + HostOf(server) + ":31315";
    }
    if (!webapi.empty()) {
        SetBoth(&EmulatorSettingsImpl::SetShadNetWebApiServer, webapi);
    }
    if (const std::string port = EnvString("BB_COOP_P2P_PORT"); !port.empty()) {
        try {
            const int value = std::stoi(port);
            if (value < 0 || value > 65535) {
                throw std::out_of_range{"port"};
            }
            SetBoth(&EmulatorSettingsImpl::SetP2PPort, value);
        } catch (const std::exception&) {
            LOG_ERROR(Config, "BB co-op: invalid BB_COOP_P2P_PORT '{}'", port);
        }
    }
    ApplyAccount();
    LOG_INFO(Config, "BB co-op on: server={} webapi={} p2p_port={} upnp={} seamless={}",
             EmulatorSettings.GetShadNetServer(), EmulatorSettings.GetShadNetWebApiServer(),
             EmulatorSettings.GetP2PPort(), EmulatorSettings.IsUPnPEnabled(), IsSeamlessEnabled());
}

} // namespace Core::BBCoop
