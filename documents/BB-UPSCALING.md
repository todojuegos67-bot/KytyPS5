# Bloodborne upscaling (phase 1)

DLSS / DLAA (NVIDIA Streamline), XeSS, FSR 3.1 and FSR 4 run on the game's 3D image before its HUD.
The game draws the HUD itself afterwards, at the output size, so the HUD never goes through the
upscaler. The code is in `src/video_core/bb/`. It reuses the Demon's Souls fork's `src/local` back ends.

## Environment variables (set by the launcher)

| Variable | Values | Default |
|---|---|---|
| `BB_UPSCALER` | `off`, `native`, `dlss`, `dlaa`, `xess`, `fsr3`, `fsr4` | `off` |
| `BB_UPSCALE_MODE` | `native` (DLAA/AA, 1.0), `ultraquality+` (XeSS 1.3), `ultraquality` (XeSS 1.5), `quality` (1.5, XeSS 1.7), `balanced` (1.72, XeSS 2.0), `performance` (2.0, XeSS 2.3), `ultraperf` (3.0), `drs` (FSR 4 dynamic model) | `quality` |
| `BB_UPSCALE_SHARPNESS` (alias `BB_SHARPNESS`) | 0..1: FSR 3.1 / FSR 4 RCAS; after DLSS / DLAA / XeSS the presenter runs RCAS (attenuation (1-s)*2) | `0.3` |
| `BB_UPSCALE_JITTER` | `0` / `1`: sub-pixel viewport jitter on the scene passes (only while a camera is found) | `1` |
| `BB_REACTIVE` | `0` turns off the reactive mask (luminance change at the reprojected position: particles, fire, fog) sent to FSR 3.1 (reactive + T&C), XeSS (responsive mask) and DLSS (bias-current-colour hint); FSR 4's model has no reactive input | `1` |
| `BB_SCENE_UPSCALE` | 0.93b default (unset / `auto` = `1`): DCport placement. The game renders its 3D scene chain, including its own scene-size post effects, at the internal size (resolution patch of the scene values, Bloodborne 1.09), its UI at the output size; the upscaler runs where the game composites the scene into the output-size image, before the HUD, so the HUD is never an input and never resets history. `native`: the generic per-target chain (opt-in; in 0.93 its later consumers still read internal-size images). `0`: off | `1` |
| `BB_DEPTH_INVERT` | `1` / `0`: forces the depth convention given to every upscaler (default: the G-buffer pass's depth test) | auto |
| `BB_MV` | `reconstruct` (from depth plus camera) / `zero` | `reconstruct` |
| `BB_UPSCALE_AT` | `auto`, `scene` (HDR before post-processing), `composite` (where render size changes to output size) | `auto` |
| `BB_UPSCALE_TO_WINDOW` | `1`: when the game's whole frame is smaller than the window, upscale to the window size (before the HUD; the HUD is added on top when presenting); `0`: in place at the game's size (native AA) | `1` |
| `BB_UPSCALE_SRC` | guest address (e.g. `0x2720c0000`), overrides the detected source image | detected |
| `BB_HUDLESS_COPY` | `1`: copy the presented image just before the first HUD draw (for phase 2 frame generation) | `0` |
| `BB_UPSCALE_LOG_EVERY` | frames between `[BB_UPSCALE]` status lines | `600` |
| `BB_ANALYZE` | `1`: also dumps every draw's targets and sampled textures for one frame every 1800 | off |
| `BB_STREAMLINE_DIR` | folder with `sl.interposer.dll` and its plugins | `streamline\` next to the exe |
| `BB_DLSS_PRESET` | `A`..`O`: force a DLSS model | driver default |
| `BB_FSR4_PRESET`, `BB_FSR4_DIR`, `BB_FSR4_SHARPNESS`, `BB_FSR4_DEPTH_INVERT`, `BB_FSR4_OPT` | the same as the Demon's Souls `KYTY_FSR4_*` variables | from mode |
| `BB_XESS_MV_SIGN`, `BB_FSR3_MV_SIGN`, `BB_JITTER_SIGN` | `-1` flips the sign convention (for testing) | `1` |

The internal resolution is the resolution the game renders at, which the game's resolution patch
sets. The log prints the internal size that the selected mode needs (output / ratio). When the 3D
renders at the output size, the upscaler runs as native AA (DLAA), unless the window is larger:
then the upscaler's output is the window size (game size -> window size).

## Package

`shadPS4.exe`, `streamline\*.dll`, `libxess.dll`, `amd_fidelityfx_vk.dll`, `fsr4_shaders\`, and
`LICENSES\` (GPL-2.0, Streamline MIT, XeSS, FidelityFX MIT, FSR 4 v07).

## Frame generation (BB_FG, branch bb-fg-022)

Ported from the Demon's Souls fork (src/local/frame-gen.cpp, fsr3-fg.cpp, anti-lag.cpp) into
src/video_core/bb/bb_fg.cpp, bb_fg_fsr3.cpp and bb_antilag.cpp. It needs an upscaler
(`BB_UPSCALER`): its inputs are the upscaler's (the upscaled image before the HUD, the scene depth,
the camera motion vectors, the jitter and the camera matrices). Frames without a 3D scene (menus,
movies, loading screens) are shown as they are; DLSS-G turns off after 30 such presents and on
again after 10 3D frames.

| Variable | Values | Effect |
|---|---|---|
| `BB_FG` | `off` (default), `dlss-x2`, `dlss-x3`, `dlss-x4`, `dlss-x5`, `dlss-x6`, `dlss-dyn` | NVIDIA DLSS Frame Generation through Streamline 2.14.16 (NVIDIA-signed DLLs only, checked before loading). x5 / x6 only where the GPU makes them (RTX 50), else capped to its maximum. `dlss-dyn`: the app-side dynamic multiplier up to x6, target refresh x 14/15. |
| `BB_FG` | `amd-x2`, `amd-x3`, `amd-x4`, `amd-dyn` | AMD FSR 3 frame generation (FidelityFX frame interpolation swapchain). x3 / x4 / dyn need the patched amd_fidelityfx_vk.dll the workflow builds (AMD's signed one stays x2). `amd-dyn`: x2..x4, target refresh - refresh^2/3600. |
| `BB_DLSSG_TARGET_FPS`, `BB_FSR3_TARGET_FPS` | fps | The dynamic multiplier's shown-fps target. |
| `BB_VRR_CAP_FPS` | fps or `auto` | Fixed multipliers: real frames held to cap / multiplier (G-SYNC / FreeSync range). |
| `BB_FSR3_PACING_FLOOR` | factor (default 1.0, 0 off) | Patched DLL: no FSR 3 present closer than one refresh period x factor. |
| `BB_REFLEX` | `on` / `low`, `boost`, `off` | NVIDIA Reflex. With DLSS-G: Streamline's Reflex mode (default low latency). Without Streamline: VK_NV_low_latency2 (as `BB_ANTILAG=1`). |
| `BB_ANTILAG` | `0`, `1`, `2` | 1: AMD Anti-Lag 2 (VK_AMD_anti_lag) or NVIDIA Reflex (VK_NV_low_latency2) when Streamline is not loaded; 2: also the emulator's latency limiter where neither exists. |
| `BB_LATENCY` | `1` | `[lat] input->present ...` once a second. |
| `BB_METRICS_OUT` | file | The `[dlssg] ... real X fps, shown Y fps; generation on N of M frames`, `[fsr3] real X fps, shown Y fps; generation on N of M frames`, `[fsr3-mfg] xN ...; real X fps (free F), shown Y fps` and `[dyn] ...` lines go there too (metricas-bb.ps1 reads real / shown). |

Every other `KYTY_*` knob of the Demon's Souls code is read as `BB_*` first (`BB_DLSSG_REFLEX`,
`BB_DLSSG_SLEEP`, `BB_DLSSG_PACE=reflex`, `BB_DLSSG_REFLEX_LIMIT`, `BB_FRAMEGEN_HUDLESS=0`,
`BB_FSR3_ASYNC`, `BB_FSR3_DEBUG_VIEW`, `BB_FSR3_MV_SIGN`, `BB_DYN_QUALITY`, ...). nvngx_dlssnr.dll is
never shipped.
