# DLSS on Linux (NVIDIA NGX bridge)

On Windows the emulator reaches NVIDIA DLSS through the Streamline runtime (`src/local/frame-gen.cpp`,
`KYTY_FRAMEGEN_STREAMLINE`). Streamline exists for Windows only, so on Linux `frame-gen.cpp` (`KYTY_FRAMEGEN_NGX`)
loads `libkyty_dlss.so` at run time instead: this bridge alone links NVIDIA's DLSS (NGX) SDK, and NGX loads
`libnvidia-ngx-dlss.so.*` from the bridge's folder. The emulator itself contains no NVIDIA code and runs as before
without the bridge (the game's own TAA).

Adapted from bbport's bridge (spacepepperoni/uncharted_Linux `gpu/dlss_bridge`, MIT), itself from
IFreemz/shadPS4-Bloodborne-DLSS-FSR. ABI 3 (`src/local/kyty-dlss-bridge.h`) adds UTF-8 paths, the motion vectors'
scale for DLSS, and the emulator's own camera for frame generation.

## Build

```
./build-linux.sh                              # kyty_emulator in _Build/linux
tools/local/dlss-bridge/build-dlss-linux.sh   # libkyty_dlss.so and NVIDIA's libraries next to it
```

`build-dlss-linux.sh` fetches the DLSS SDK (github.com/NVIDIA/DLSS, NVIDIA's license) into `_Build/deps/dlss-sdk`
the first time. Other folders can get the libraries too: `build-dlss-linux.sh _Build/linux /path/to/release`.

The Linux package carries none of NVIDIA's files: it has the bridge's source (`dlss-bridge/`) and
`get-dlss-linux.sh`, which downloads the SDK and builds the bridge on the user's PC with git and a C++ compiler alone
(no CMake; when missing, it offers to install them with the distribution's package manager), next to the package's
`kyty_emulator`.

## Run

DLSS replaces Sony's PS5 Pro upscaler on Demon's Souls' PS5 Pro path, whose output is 3840x2160; the mode follows the
render height's share of it (`KYTY_RENDER_HEIGHT`: 2160 DLAA, 1440 and 1512 Quality, 1080 Performance; 1080 at least
on that path):

```
KYTY_UPSCALER=dlss KYTY_RENDER_HEIGHT=1080 KYTY_GAME=/path/to/game tools/local/run_ds.sh
```

(From the repository, `run_ds.sh` takes `_Build/linux/kyty_emulator`; a release folder's copy takes
`KYTY_BIN=/path/to/_Build/linux/kyty_emulator`, and the bridge is then looked for next to that executable.)

The log says `DLSS: bridge ... loaded`, `DLSS super resolution: ready (NVIDIA NGX)` and
`DLSS super resolution: 1920x1080 -> 3840x2160, Performance (NVIDIA NGX)`.

| Variable | |
| --- | --- |
| `KYTY_DLSS_DIR` | the bridge's folder (default: next to `kyty_emulator`, or its `dlss` folder) |
| `KYTY_DLSS_PRESET` | a DLSS model letter for every mode (`K`, `L`, `M` ...) |
| `KYTY_DLSS_SHARPNESS` | NGX's sharpening, 0 to 1 (default 0) |
| `KYTY_DLSS_NGX_LOG=1` | NGX's own verbose log |
| `KYTY_SR_LOG=1` | the first evaluations' images and constants |

## Frame generation

`KYTY_FRAMEGEN=<n>` (with or without `KYTY_UPSCALER=dlss`): NGX's DLSS-G makes `n` frames between two of the game's
(1: 2x; up to 3, 4x, on GeForce RTX 50), and the presenter shows them before the game's frame, evenly spaced over the
game's frame time (`PresentGenerated`, swapchain.cpp: on Windows Streamline does this inside the present). The log says
`Frame generation: DLSS-G ready (NVIDIA NGX, 1 generated frame(s) per frame, 2x)`.

The generated frames are spaced by the game's frame time, so they are as even as the game's flips: at the console's
60 Hz vblank a game between 30 and 60 fps flips after one vblank or two by turns (16.7 or 33.3 ms), and so do the frames
in between. `KYTY_VBLANK=240` (run_ds.sh adds the game's `gameMinDeltaTime` above 60 Hz) lets each flip go 4.2 ms after
it is ready instead, and the frame times follow the game's own.
