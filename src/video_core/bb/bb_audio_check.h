// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Windows audio check (only with an upscaler on, or BB_AUDIO_CHECK=1; BB_AUDIO_CHECK=0 turns it
// off): 25 s after start and twice more, logs to bb_upscale_init.log this process's Windows audio
// session on every output device (app volume, mute, the peak Windows really mixes) and the
// default device's system volume. Windows keeps the per-app volume and output device per exe
// path, so a fork in its own folder can be silent while the official exe is not. A muted / zero
// per-app volume is set back to 100% for this run (BB_AUDIO_FIX=0 leaves it).
#pragma once

namespace BB::Audio {
void StartCheck();
// Called by the SDL audio back end after each stream opens: the first one triggers pass 1.
void OnStreamOpened();
}
