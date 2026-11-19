// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Bloodborne co-op switches for the launcher. Everything here is off unless the launcher sets
// the environment variables below; with none set, nothing is read and no setting is changed.
//
//   BB_COOP=1                 co-op on: network connected + shadNet on for this run, and the
//                             account from user/bb_coop.json given to the default user.
//   BB_COOP_SERVER=h:p        shadNet server (default: the config's shadnet_server).
//   BB_COOP_WEBAPI=url        shadNet WebAPI (default: http://<BB_COOP_SERVER host>:31315 when
//                             BB_COOP_SERVER is set, else the config's shadnet_webapi_server).
//   BB_COOP_P2P_PORT=n        UDP P2P port (default: the config's p2p_port, 3658).
//   BB_COOP_UPNP=1            keep UPnP on (default with BB_COOP=1: off, for Tailscale).
//   BB_COOP_SEAMLESS=1        experimental seamless co-op guest patches (shadp2p's
//                             SHADPS4_BLOODBORNE_SEAMLESS_COOP=1 is accepted too).
//
// user/bb_coop.json: {"npid": "HunterOne", "password": "..."}. Never logged.
// host_overrides.json (user/, or SHADPS4_HTTP_HOST_OVERRIDES_JSON) is read by libSceHttp.

#pragma once

namespace Core::BBCoop {

/// BB_COOP=1 (read once).
bool IsEnabled();

/// BB_COOP_SEAMLESS=1 or SHADPS4_BLOODBORNE_SEAMLESS_COOP=1 (read once).
bool IsSeamlessEnabled();

/// Applies the BB_COOP_* overrides to the loaded settings and the default user (in memory
/// only). Call after the game's settings profile has been loaded. No-op unless BB_COOP=1.
void ApplyToSettings();

} // namespace Core::BBCoop
