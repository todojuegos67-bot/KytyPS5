// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// `--coop-register`: create a shadNet account (same Create request as shadnet-p2p's
// clientsample "register") and print one result line. The password and key are never logged.

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <optional>
#include <string>

#include "core/bb_coop_config.h"
#include "shadnet/client.h"

namespace Core::BBCoop {
namespace {

const char* ErrorName(ShadNet::ErrorType e) {
    using E = ShadNet::ErrorType;
    switch (e) {
    case E::NoError: return "OK";
    case E::Malformed: return "Malformed";
    case E::Invalid: return "Invalid";
    case E::InvalidInput: return "InvalidInput";
    case E::TooSoon: return "TooSoon";
    case E::CreationError: return "CreationError";
    case E::CreationExistingUsername: return "CreationExistingUsername";
    case E::CreationBannedEmailProvider: return "CreationBannedEmailProvider";
    case E::CreationExistingEmail: return "CreationExistingEmail";
    case E::Unauthorized: return "Unauthorized";
    case E::DbFail: return "DbFail";
    case E::EmailFail: return "EmailFail";
    case E::Unsupported: return "Unsupported";
    default: return "Error";
    }
}

} // namespace

int RunRegister(const std::string& server, const std::string& npid, const std::string& password,
                const std::string& email, const std::string& key) {
    std::string host = server;
    u16 port = 31313;
    if (const auto colon = server.rfind(':'); colon != std::string::npos) {
        host = server.substr(0, colon);
        try {
            const int p = std::stoi(server.substr(colon + 1));
            if (p <= 0 || p > 65535) {
                throw std::out_of_range{"port"};
            }
            port = static_cast<u16>(p);
        } catch (const std::exception&) {
            std::printf("InvalidServer\n");
            return 4;
        }
    }
    if (host.empty() || npid.empty() || password.empty() || email.empty()) {
        std::printf("MissingArguments\n");
        return 4;
    }

    ShadNet::ShadNetClient client;
    std::mutex mutex;
    std::condition_variable cv;
    std::optional<ShadNet::ErrorType> result;
    client.onAsyncReply = [&](ShadNet::CommandType cmd, u64, ShadNet::ErrorType err,
                              const std::vector<u8>&) {
        if (cmd != ShadNet::CommandType::Create) {
            return;
        }
        std::scoped_lock lock{mutex};
        result = err;
        cv.notify_all();
    };
    client.StartRegister(host, port, npid, password, email, key);
    if (client.WaitForConnection() != ShadNet::ShadNetState::Ok) {
        client.Stop();
        std::printf("ConnectionFailed\n");
        return 2;
    }
    {
        std::unique_lock lock{mutex};
        cv.wait_for(lock, std::chrono::seconds(15), [&] { return result.has_value(); });
    }
    client.Stop();
    if (!result) {
        std::printf("Timeout\n");
        return 3;
    }
    std::printf("%s\n", ErrorName(*result));
    std::fflush(stdout);
    return *result == ShadNet::ErrorType::NoError ? 0 : 1;
}

} // namespace Core::BBCoop
