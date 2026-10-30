# Bloodborne upscaling (phase 1)

DLSS / DLAA (NVIDIA Streamline), XeSS, FSR 3.1 and FSR 4 run on the game's 3D image before its HUD.
The game draws the HUD itself afterwards, at the output size, so the HUD never goes through the
upscaler. The code is in `src/video_core/bb/`. It reuses the Demon's Souls fork's `src/local` back ends.

## Environment variables (set by the launcher)

| Variable | Values | Default |
|---|---|---|
| `BB_UPSCALER` | `off`, `native`, `dlss`, `dlaa`, `xess`, `fsr3`, `fsr4` | `off` |
| `BB_UPSCALE_MODE` | `native` (DLAA/AA, 1.0), `ultraquality+` (XeSS 1.3), `ultraquality` (XeSS 1.5), `quality` (1.5, XeSS 1.7), `balanced` (1.72, XeSS 2.0), `performance` (2.0, XeSS 2.3), `ultraperf` (3.0), `drs` (FSR 4 dynamic model) | `quality` |
| `BB_UPSCALE_SHARPNESS` | 0..1 (FSR 3.1 / FSR 4 RCAS) | `0.3` |
| `BB_UPSCALE_JITTER` | `0` / `1`: sub-pixel viewport jitter on the scene passes (only while a camera is found) | `1` |
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
