# Static shader / pipeline precompile (Demon's Souls, PPSA01341)

Goal: collect every shader and pipeline the game can use straight from the game files (no recording
of actual play) and compile them into the driver cache ahead of time, so the first encounter of a new
shader / pipeline in the game no longer stalls for 1–3 seconds.

## Flow

1. `kyty_shader_precompile --game <dir> --make-seeds OUT` (`src/local/static-seeds.cpp`): static
   collection into a seed file (`KytyShaderSeeds2`, the warmup cache's format), without Python. It is
   the C++ port of `precompile.py seeds OUT` and writes the same file byte for byte (`--stages` and
   `--limit` as there); the Python tools (Python 3 with numpy) stay the reference and do the analyses
   below. `pass-states.json` is read from `tools/local/static-precompile` below the program's folder (a
   release package), two folders above it (`_Build/<build>/`) or below the working directory;
   `--states <file>` names another.
   - Shaders: every CSDR package (`_trinity` is the PS5 Pro variant, skipped) plus the AGC headers and
     code embedded in eboot.bin's data segment.
   - Compile keys: derived from the AGC headers' register tables (`keys.py`, ported from
     `shader.cpp`/`pm4Handlers.cpp`/`agc.cpp`; CS use the dispatch_modifier, PS interpolators come from
     pairing VS/PS semantics).
   - VS/PS pairs: the technique lists of the `.cmat` materials (pass id + package entry); the engine's
     own full-screen/UI/particle passes pair by their `coredata/enginesupport/shaders/<effect>/`
     folders.
   - Pipeline state: each material pass's render target/blend/depth-stencil state table
     (`pass-states.json`, the "engine render pass configurations" `learn-states` learned from recorded
     caches), each state with culling on and off.
2. `precompile-windows.ps1` (repository root): the standalone program `kyty_shader_precompile` (no game,
   no window; build it with `build-windows.cmd kyty_shader_precompile`) translates every seed into
   SPIR-V (portable specialization, see below), creates every pipeline and writes the **static pipeline
   cache** `_PipelineCache/static/PPSA01341_<version>.bin` (one per game version). It runs step 1 first when the seed file is missing.
   - The static cache is keyed by the GPU/driver signature only and survives emulator rebuilds; the
     game looks a pipeline up there first without compiling
     (`VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT`) and returns on a hit.
   - The NVIDIA driver compiles large CS almost serially within one process (22 threads kept only 1.3
     cores busy, about 7 s each for the large ones), so the work is split by `--shard i/n` into several
     low-priority processes running in parallel (`-Jobs`, default one per allowed CPU) and merged at the
     end.
   - A rerun compiles only what is missing (what the static cache holds hits directly). Each process
     saves a checkpoint every 10 minutes; after an interruption a rerun merges those checkpoints first.
   - `-Coverage`: builds no pipelines, only writes what was compiled (`<seeds>.compiled.shaders` and
     `.spirv.txt`/`.rejected.txt`).
   - A full run finally writes `_PipelineCache/static/PPSA01341_<version>.shaders` (the compile inputs, warmup
     format, 136 MB) for the game's **background pre-translation** (`KYTY_SHADER_PREFETCH`, on by
     default): after start-up, low-priority threads translate all of these programs (about 60–100 s,
     SPIR-V kept in a temporary system file), and the first encounter takes them instead of
     translating on the spot. `-InputsOnly` writes just this file (about 30 s). It holds compile
     inputs (the translation is done by the emulator in use), so it only needs rewriting when the seeds
     or the specialization inference (`SeedSpecializations`) change.
   - For comparisons: `KYTY_STATIC_PIPELINE_CACHE=0` keeps the game from loading the static cache,
     `KYTY_SHADER_PREFETCH=0` turns off the background pre-translation.
3. `precompile.py coverage COMPILED --recorded-compiled R`: comparison with a recorded cache (at the
   SPIR-V level = whether the driver cache can hit). `recorded-seeds` turns a recorded cache into
   seeds, compiled the same way with `-Coverage` to get R.
4. `stalls.py LOG`: compile stall statistics of a run log (`KYTY_SLOW_LOG_MS=30`).

## Portable specialization (recompiler, `KYTY_PORTABLE_SHADERS`, on by default)

Information that is not in the shader binary but only in runtime descriptors used to be baked into the
SPIR-V, which a static pass cannot predict:

- Buffer stride (≤4095, not swizzled): read from each buffer's descriptor word in the shader data
  (`IR::BufferWord`) instead.
- Format/dst_sel of read-only formatted buffers (vertex attributes and the like): two levels. The first
  encounter uses the portable module P that decodes the format at run time (exactly what the static
  precompile compiles: no driver compile stall); the warm cache records the dedicated specialization S,
  and the next start-up's warmup compiles S (full speed).
- Null textures: by the dimension the shader declares (formerly always 2D, which made one more
  permutation of the same program).
- Indirect texture tables: the candidate count padded to a power of two, the key mapping offset read
  through the flattened SRT header, the binary search turned into a runtime loop. The table's type comes
  from the textures in it: the particle CS's table instructions declare 3D while the game puts only 2D
  textures there, so the seeds add a 2D-table version of every variant.

Still baked in: the numeric class of integer textures (`PredictImageNumericClasses` guesses only part
of them from their use), storage image swizzles, cube maps, the format of written formatted buffers
(guessed as the most common 32UInt), buffers with a stride ≥4096 or swizzled.

## In-game measurements (09-30)

Method: `tools\local\windows\bench-run.ps1 -NoPrecompile -Set KYTY_SHADER_WARMUP=0,KYTY_SLOW_LOG_MS=30`
(cold start: no warmup, empty driver cache), then
- `stalls.py LOG`: the number and total time of compile stalls;
- `precompile.py runtime-coverage LOG COMPILED`: how many of the modules compiled at run time (the log's
  `MODULE` lines, SPIR-V hashes) are in the precompiled set (COMPILED is the output of
  `precompile-windows.ps1 -Coverage`).
- `KYTY_SHADER_WARMUP=record` + `KYTY_SHADER_WARMUP_FILE=<new file>` record the inputs one run compiled
  to a separate file; `recorded-seeds` + `-Coverage` + `coverage --recorded` then show which
  specialization field a missing module differs in.

Result on the fixed route: of the modules compiled at run time, CS 82%, PS 91% and VS 100% are in the
precompiled set (every program used at run time is in the seeds; the misses are all differently
inferred specializations). The missing CS cost about 36 pipelines, 22 s in total, on a truly cold start;
NVIDIA's driver disk cache and the game's own `_PipelineCache/local/<exe SHA256>` both make repeated
tests "warm" (the same pipelines take 30 ms or 100+ ms), so module coverage is the more reliable
comparison.

Pipelines that miss no longer stall: they are first compiled unoptimized (milliseconds on NVIDIA) and
used right away, and an optimized build replaces them in the background (`KYTY_PIPELINE_FAST_BUILD=0`
turns this off). Pipeline stalls on the cold route went from 22 s to 0.

Missing specializations (mostly CS): integer textures judged as float (now "every direct use before a
merge is integer arithmetic means uint"), single-channel swizzles of storage images (X001), a declared
2D array bound to a cube or a plain 2D texture, written formatted buffers that are not actually 32UInt.

## Scale

- The game ships 20.6K programs (10.7K CS, 9.5K of them particles; 71 MB of PS5 machine code in total)
  and about 46K pipelines; one play-through uses only about 3K. The particle CS are genuinely different
  programs (even allowing 10% differing instructions merges them only into 7.6K groups): they cannot be
  saved.
- In the NVIDIA cache each pipeline is about 26 times its PS5 code (translation overhead, fixed 16-byte
  instructions, a VS+PS copy in every graphics pipeline): an estimated 3–5 GB in total.

## Known gaps

- Numeric class of integer textures: the usage prediction recognizes only about 23% (about 10% of the CS
  programs miss because of it).
- Pass 10 (1799 VS/PS pairs) never appeared in a recording: its render target formats/blending are
  unknown and no pipelines are made for it.
- Programs with 3 or more indirect texture sources are compiled only in their single-texture form.
- 8 seeds are refused by the recompiler (code paths the game never uses), see `*.rejected.txt`.

## Checking the portable format decode

`shader_recompiler_compute_tests --portable-formats-only` compares, on the GPU, the portable (runtime)
decode of every format, wave32/64, with the decode specialized for the descriptor; normalized formats
have a known FDiv rounding difference (at most 2 ULP), which is reported as it is
(`--portable-specialization-parity-only` compares bit for bit and fails because of it at the moment).
`--portable-formats-emission-only` compiles without running.

Moving the decode of rare formats into a shared function (`DontInline`, night of 09-29) was tried:
compile times of large shaders went up for some and down for others with about the same total, and
modules such as `b4a64911fc8e88db` crashed the NVIDIA compiler (nvgpucomp64.dll), so the game quit on
their cold encounter: removed. The driver-level shader cache of Windows NVIDIA drivers here ignores
`__GL_SHADER_DISK_CACHE=0`: compiling the same SPIR-V again is no cold compile for comparisons.
