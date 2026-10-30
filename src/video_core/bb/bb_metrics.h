// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// BB_METRICS_OUT=<file>: once a second, the same lines as Kyty writes for Demon's Souls, read by
// metricas-bb.ps1 (-FpsFile):
//   [fps] N frames: mean X ms, max Y ms
//   [lat] input->present mean A ms, p95 B ms
//   [up] <upscaler> <mode> <render> -> <output>   (or "[up] off")
// Appended and flushed per line; the previous run's file is kept as <file>.1.
#pragma once

#include <string>

namespace BB::Metrics {
bool Enabled();
void OnInput();                // the game read the pad
void OnPresent();              // a new guest frame was presented
void SetUpscaleLine(const std::string& text);
void SetResLine(const std::string& text); // written each second as "[res] game WxH -> output WxH ..."
} // namespace BB::Metrics
