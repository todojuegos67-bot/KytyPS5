// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// BB DLSS 5 (NVIDIA Neural Rendering, NGX feature 18) on the upscaler's output, with the foveated
// working grid. Ported from the Demon's Souls fork (src/local/dlss-nr.cpp / dlss-nr.comp,
// ds-pc-port10..43; after RenoDX's DLSS 5 add-on and FoveatedDLSS5, MIT:
// LICENSES/FoveatedDLSS5-RenoDX-MIT.txt). NVIDIA's nvngx_dlssnr.dll is never shipped: the user's
// own file beside shadPS4.exe (or in its streamline folder / BB_STREAMLINE_DIR / BB_DLSS5_DIR).
//
// Switches (the launcher's "DLSS 5" group, as the Demon's Souls settings.ini):
//   BB_DLSS5             0 off (default: nothing is loaded, no device extension, no pass),
//                        1 Suave, 2 Medio, 3 Fuerte (DS: the model's intensity 0.75 / 1.1 / 1.5 and
//                        the foveated grid's edge density 0.75 / 0.5 / 0.35)
//   BB_DLSS5_INTENSITY   baja / media / alta (default): the share of the model's edit applied,
//                        0.5 / 1 / 1.15 (alta also caps the model's intensity at 1, as DS port43)
//   BB_DLSS5_UNOFFICIAL  1: also load an nvngx_dlssnr.dll without NVIDIA's signature (default 0)
// Only NVIDIA GPUs, only after a successful DLSS / XeSS / FSR 3.1 / FSR 4 evaluate (the pass is
// called from the rasterizer after the back end); DLSS-G and FSR 3 FG are untouched (they read
// the same output image afterwards). Anything missing: one line in bb_upscale_init.log, the game
// keeps the upscaler's picture.

#pragma once

#include <string>
#include <vector>
#include "video_core/bb/bb_sr.h"

namespace BB::Dlss5 {

int Level();   // BB_DLSS5 0..3
bool Wanted(); // Level() > 0
void AddDeviceExtensions(vk::PhysicalDevice physical, const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              VmaAllocator allocator);
// After the upscaler wrote desc.output (RGBA16F, GENERAL): NR in place. False: the frame unchanged.
bool Apply(const SR::Desc& desc);
// 0.89: one call per upscaler run (before Apply): ok = the back end wrote the output, window =
// the window / present-time path (post-processed input), for the "[dlss5] last second" line.
void NoteUpscale(bool ok, bool window);
std::string StatusLine();

} // namespace BB::Dlss5
