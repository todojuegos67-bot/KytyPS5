# Marvel's Wolverine on KytyPS5: performance and stability work

Branch `wolverine-v1` (release
[wolverine-v1](https://github.com/IDXTRI/KytyPS5/releases/tag/wolverine-v1)): KytyPS5 release
2026-10-03 (f53e5d2) merged with the community Wolverine work (PR #937: Mac/itsmemac, Senaxx, Ali
Almohaya) and the changes below. The earlier branch `wolverine-perf` was not merged with upstream.

Tested with PPSA03671 v01.001.005 on Windows 11, Ryzen 7 7800X3D, RTX 4070 Ti (12 GB),
clang-cl Release build. Run with
`--bindless --redzone --tessellation --readback-linear-images true --amd-cpu` (AMD CPUs only) and
Mac's skip list `--skip-shaders bad108e74fb72e9f,4e7f2c6bb9b158a1,8bfd230b9cd875a2,c4df2a00067e0666,dc76e1223a9bf673,e6d76d24f59f8015`.
No environment variables are needed: `KYTY_LABELS_AFTER_GPU` is on by default, and upstream's
written-overlap check only warns (`KYTY_IGNORE_WRITTEN_OVERLAP` is gone). The release package
(`Play Wolverine.bat`, launcher preset) sets all of this.

## Results

| Scene | `wolverine-perf` (before the merge) | `wolverine-v1` |
|---|---|---|
| Main menu | ~30 fps | ~30 fps |
| Gameplay spot, standing still | ~29 fps (benchmark) | ~29-30 fps |
| Dense jungle | not measured | ~12-15 fps |
| VRAM (device usage, 12 GB card) | 12.7 GiB, ~1.8 GiB in shared memory | 10.3-11.1 GiB, ~0.7 GiB shared |

Same spot (slot 2 save); `wolverine-v1` numbers with the user driving, the user judged it the
best of the builds compared. Without Mac's skip list the spot runs at
21-22 fps. In busy scenes the GPU is only ~36-40 % busy: the emulator's CPU side is the limit.

Changes were measured with same-process A/B (live switches, Tracy) where possible, and only
changes with a measured gain or a fix are on by default.

## Main changes (default on)

**Memory**
- **Two-level BDA page table**: shader memory accesses translate guest pages through a directory
  and 2,048 chunks (64 MiB) instead of a flat 768 MiB table.
- **Image pool** (`KYTY_IMAGE_RECYCLE_MB=256`): freed images are parked and reused by images of
  the same shape; parked images are trimmed after 2 s or above the cap.
- **Garbage collection tuning**: image-aware buffer age (`KYTY_BUFFER_GC_AGE=120`), collector every
  4 ms (`KYTY_GC_INTERVAL_US=4000`), combined buffer+image budget (`KYTY_GC_COMBINED=1`), critical
  ages 60 (buffers) / 30 (images).
- **Streamed textures** (`KYTY_BINDLESS_REVALIDATE=1`, from Senaxx 0563e10e): a bindless key whose
  heap T# changed is settled again and its old image is unpinned.

**Performance**
- **Recording thread** (`KYTY_RECORD_THREAD=1`): Vulkan command recording moves off Thread_Gpu.
- **Native SRT walker** (`KYTY_SRT_NATIVE=1`) and compiled plans (`KYTY_SRT_COMPILED=1`), with the
  replay of the last reachability walk.
- **Page read cache** (`KYTY_SHADER_READ_CHUNKS=2`).
- **Bindless generation skip** (`KYTY_BINDLESS_SKIP=1`).
- **Async readbacks** (`KYTY_ASYNC_READBACK=1`, `KYTY_COPY_QUEUE_READBACK=1`).
- **Driver pipeline cache kept across builds** (`KYTY_PIPELINE_CACHE_ANY_REVISION=1`).

**Stability**
- **GPU fault handling**: a host fault on a page the GPU cache no longer maps is retried once the
  page is accessible; guest pages whose protection was orphaned get their access restored.
- **Suspend points** block only on the previous suspend point, as on the console.
- **Labels after GPU** (`KYTY_LABELS_AFTER_GPU=1`): EOP labels, timestamps and flips are published
  when the GPU has executed the preceding work.

**Input**: the Xbox View/Back button acts as the PS5 touchpad.

## Merge with KytyPS5 2026-10-03 (e6cb880, f53e5d2)

Upstream versions were taken where they replace earlier copies of the same work (BVH execution,
FLAT apertures and local stacks, code-hash shader identities, IR operand storage, explicit gather
LOD, 64-bit atomics). Kept or adjusted for Wolverine:

- `KYTY_SKIP_BVH_DISPATCHES=1`: programs with BVH intersections are still skipped (Wolverine's
  CS 0x29be8047c4e86afe ran skipped; upstream's BVH GPU test loses the device on this build).
- Upstream's scalar-read overlap check (6409be28) warns once per shader pair instead of exiting.
- A scalar read through a null base evaluates to 0 (CS 0x0b4b91abfed42248).
- Finite images only for sampled handles (Senaxx edc4532a); fast-launch mesh instance VGPR and
  primitive capacity (Senaxx e6f33fdb).
- Explicit-LOD gathers with linear mip filtering or LOD bias read the selected mip instead of
  failing the specialization (upstream skipped the draw: 3,130 failures in the first minutes).
- Our suspend points, `KYTY_BOOL_PREDICATION_WAIT`, windowed async downloads and the
  once-per-submission BDA sync are kept.

## Experimental (default off)

- `KYTY_FUNCTION_ARRAY_SHRINK=1` (from Jetsku/KytyPS5): shrinks Function-storage arrays to their
  proven index bound before the driver sees the module. In Wolverine it changes 7 modules by 32
  elements each: no VRAM gain, more stutter (the driver recompiles them).
- `KYTY_SKIP_BVH_DISPATCHES=0`: runs the ray-tracing programs (untested in the game).
- `KYTY_BINDLESS_EVICT=1`: unpins bindless textures the shaders stop sampling. It exposes a read of
  an image whose memory was already released (device lost within minutes); under investigation.
- `KYTY_VMA_BLOCK_MB=64`: smaller device memory blocks; same stale-image read.

## Diagnostics

- Live switches: set `KYTY_LIVE_FILE` to a text file and write `NAME=value` lines while the game
  runs (A/B in one process).
- `KYTY_MEMORY_STATS`, `KYTY_IMAGE_STATS`, `KYTY_IMAGE_GC_STATS`, `KYTY_BUFFER_STATS`,
  `KYTY_BUFFER_AUDIT`, `KYTY_GPU_TIME`, `KYTY_FAULT_STATS`: periodic reports in the log.
- Device loss: `VK_EXT_device_fault` + `VK_EXT_device_address_binding_report` name the buffer or
  image a faulting GPU address belonged to, its lifecycle events (created, taken from or parked in
  the image pool, destroyed) and the draws that bound it (`KYTY_ADDRESS_BINDING_REPORT=0` turns
  the binding report off). `KYTY_BDA_VERIFY=N` checks the BDA table against the buffer cache.

## Known issues

- Resizing, minimizing or full-screening the window can lose the device (from PR #937).
- Rare device loss in long sessions: an image's memory is released while the GPU can still read
  it (seen with 64 MiB blocks; under investigation).
- Flickering light/shadow patches on the ground and decals appearing and disappearing (not
  texture streaming, readback timing or recording order; probably cached shadow maps).
- Moon tiled across the night sky.
- Ray-traced effects are skipped.
- Pixel shader 0x7dce3261e625949c can hang the GPU in the night scene (from PR #937).

## wolverine-turbo-1025 (test build, untested on hardware)

`wolverine-perf` (151be85, the engine last run in the unified launcher) plus:

- **FSR 3 frame generation for any game** (`KYTY_FSR3_FG=1`, off unless set;
  `src/graphics/presentation/window/fsr3FrameGen.cpp`), ported from the Demon's Souls fork's
  `src/local/fsr3-fg.cpp`. That version feeds the game's TAA depth and motion vectors (found by
  shader hash); Wolverine has no such inputs here, so this one gives FSR 3 a constant far depth and
  zero motion vectors and relies on its optical flow. Generation only between new guest flips.
  `KYTY_FSR3_MFG=3|4` with `KYTY_FSR3_DLL=<patched DLL>` (CI input `ffx_mfg=true`, packaged in
  `emulator\fsr3-mfg\`) for x3/x4. `[fsr3]` lines every 5 s give real and shown fps.
- `KYTY_ASYNC_WRITE_READBACK` default 1 and `KYTY_WRITE_WINDOW_KB` default 16 (same-process A/B of
  10-15 in docs/EXPERIMENTS.md: -1.5 ms and -1.7 ms per frame).
- `packaging/wolverine-turbo/Play Wolverine TURBO.bat` (Spanish), CI builds only the emulator.

Not ported: DLSS-G/Streamline and DLSS/XeSS/FSR super resolution (all need the game's depth, motion
vectors and camera, found per title), the Demon's Souls recorder (`KYTY_VULKAN_RECORDING`, ported
in c13d6a0 and reverted in 470be8c), table mode and the static shader precompile (Demon's Souls
tree only, not mergeable here: the trees diverged on 2026-09-10 by ~1,100 commits each side).
