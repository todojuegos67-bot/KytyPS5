# Path tracing (opt-in, work in progress)

Real-time path tracing for Bloodborne by **João Vitor Alves**, from
<https://github.com/Joaovitor404silva/bloodborne-pc-path-tracing> (commit `04d398f`,
GPL-2.0). He wrote it for SuperMedo's Bloodborne PC port 1.9.8.6; this branch carries it into
our renderer. All credit for the path tracer — BVH capture and cache, ReSTIR GI and DI, traced
moon and lamp shadows, reflections, fog, the light-data reverse engineering, the denoiser and the
DLSS Ray Reconstruction guides — goes to him. The port, presets and bugs in the glue are ours.
Licence: GPL-2.0 (`LICENSES/GPL-2.0-only.txt`, `LICENSES/PathTracing-NOTICE.txt`). His full design
notes are in `PATH_TRACING-upstream.md` next to this file (knob names there are `BB_RT_*`, here
`BB_PT_*`), his engine patch for SuperMedo's tree in `supermedo-1.9.8.6-pathtracing.patch`.

## Status in this build

**Runs in game** (run_pt1: RTX 5080, preset low, DLSS quality 1708x960 -> 2560x1440): geometry
capture, BVH, GI, traced direct light and moon shadows work. bb-pathtracing-2 (this commit)
answers that run's report:

| problem (run_pt1) | cause found | change | state |
|---|---|---|---|
| noisy shadows | no denoiser on the direct light; RR not hosted | DLSS Ray Reconstruction hosted (`rt/pt_rr.cpp`); soft shadows give the temporal filters something to average | implemented, not run |
| no reflections | preset low had `BB_PT_REFLECT=0` | low: reflections on for roughness <= 0.25 (wet floors, puddles on the game's water-floor shaders, metal, glass); `REFLECT_MAX_ROUGH` 0.25 / 0.4 / 0.5 / 0.6 per preset | implemented, not run |
| lamp / fire / foliage shadows too hard and dark | every local light was a point: razor shadows, as dark at 5 m as at 5 cm; alpha-tested surfaces all cut at 0.5 | lights are small spheres (`BB_PT_LIGHT_SOURCE`, rt_direct.comp `LightSourceRadius`); the game's own alpha-test reference read from each pixel shader (literal or material constant, `FindAlphaRef`) | implemented, not run |
| "alpha-tested surfaces 0" | that line was printed in the menu, before the first G-buffer pass; in game it counts 37 k-240 k alpha-tested surfaces | none (alpha capture works) | investigated |
| store overflow (6.29 M of 6 M, 7782 draws dropped) | store too small for Central Yharnam | 8 / 10 / 12 / 14 M per preset; past 90 % of a frame's half, draws under 384 vertices are left out first (`BB_PT_STORE_RESERVE`) | implemented, not run |
| `0.000 ms/frame` | the line printed in menus / loading (no traced frame); in game it was 4.8-6.3 ms; TLAS counted twice | per traced frame, with the per-pass split; "no traced frame" said | implemented, not run |
| too expensive on low | direct 2.2 ms (4 shadow rays per pixel), GI 2.4 ms | low: 2 direct rays, shorter GI rays, smaller BVH budgets, nearer far cut-off (table below) | implemented, not measured |

### How the capture works here (ours, not the mod's)

- **Vertex shader variant** (`emit_spirv_bb_pt.cpp`): `HwVertexRuntimeInfo::bb_pt_store`
  (in what was padding: no shader cache layout change) makes a vertex shader also write, per
  vertex, its clip-space position (vec4, as the guest computed it: before the depth-mode / clip
  conversions and the jitter), and when the pixel shader's albedo / alpha source was found a
  half2 UV and two snorm4x8 normals (his `EmitVertexMotion` store and `uv_select` layout). The
  addresses are in the push constants (`PushData::Pt*`, 88..120 bytes, still <= 128), not baked
  into SPIR-V as upstream (cached SPIR-V stays valid across runs).
- **Which draws** (`vk_pipeline_cache.cpp`): triangle lists into 5+ colour targets with depth
  (the mod's `gbuffer_draw`; Ghidra: the eboot's debug target list names the deferred MRT
  "GBuffer RT0..RT3" + "Glow" + light accumulation, strings at 0x49d4166..0x49d431e, the
  engine's default emissive target is index 6). Only opaque G-buffer pipelines get the variant.
- **Store** (`rt/pt_capture.*`): one device buffer, 2 x `BB_PT_STORE_MVERTS` (default 6) M
  vertices (28 B each: 336 MB at the default), halves alternate per frame; per draw the index
  range is scanned (cached per index list, re-verified every 32 frames) and range x instances
  slots allocated. Overflow drops the draw from that frame's BVH (logged).
- **Surface info** (`shader_info_collection_pass.cpp`, his `FindAlbedo / FindAlphaSource /
  FindNormalMap`): kept in `Info` and saved as an optional trailing record after the cache meta
  (`Info::Serialize`), so **`ShaderMetaVersion` is not bumped** and caches stay valid. A pixel
  shader cached before this record existed is, *only with `BB_PATHTRACING=1`*, skipped at
  preload and compiled once at first use (a one-time stutter per G-buffer material on the
  first PT run), then saved with the record.
- **Camera** (`bb_upscale.cpp PtCamera`): the frame's first 864-byte scene block (inverse view
  floats 180-191, projection 52/57/62/63), as SuperMedo's `CameraMotion`; captured whenever path
  tracing is on (also without an upscaler). Jitter: ours (render pixels). Scene depth: the depth
  of the G-buffer pass.
- **Rasterizer hooks** (`vk_rasterizer_pt.cpp`): frame start = first draw into a display buffer
  (`OnDisplayPass`, store flip); `OnGBufferPass`, `OnSceneColor`, `AddDraw` + `OnDrawBound`
  (albedo / normal / alpha-test views); light accumulation draws (VS 0x8747a367): state,
  stencil, targets, constants, vertex params, `SkipLightDraw`/`SkipHalfResLight`;
  `OnLightComposite` before the direct-hook draw; `OnAfterComposite` after the light composite
  (PS 0x32e47788: GI); `OnAoDraw/OnAoInput` (PS 0x3fe9e964, the draw our tree calls the
  composite), `OnDirShadowInput`, `OnCompositeInput/Buffer`, `OnDispatch` (post trigger CS
  0x9a9cf8a9), `OnSkyDome`, `SkipPs`.
- **F9 / zero cost:** paused, the store base pushed is 0 (the variant only takes a branch per
  vertex), no index scan, no BVH build, no pass.

### Not wired (all default off in every preset)

Water / fog sheets, smoke / flame / glow particle capture (`BB_PT_FX_*`), alpha inheritance from
depth-only prepasses (`BB_PT_ALPHA_INHERIT`: depth prepasses are not stored), indirect draws in
the BVH, the liverpool light-set CB write hook (`OnLightSetWrite`; the light draw constants are
captured), device-lost hook.

## How to enable

    BB_PATHTRACING=1                     # master switch, default off
    BB_PATHTRACING_QUALITY=low           # low (default) | medium | high | ultra
    BB_PT_<knob>=<value>                 # any knob below or in PATH_TRACING-upstream.md

Needs a GPU with `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` + bufferDeviceAddress
(GeForce RTX, Radeon RX 6000+, Arc). CMake option `ENABLE_PATHTRACING` (default ON) builds the
engine and its 34 shaders; `-DENABLE_PATHTRACING=OFF` leaves them out entirely.

**F9** (only with `BB_PATHTRACING=1`) switches path tracing off/on in game, as in the mod. It
takes F9 from shadPS4's `hotkey_pause` while the switch is on (remap that hotkey if you need it).
Off costs no GPU time: the passes and, unlike upstream, the BVH build are skipped
(`BB_PT_PAUSED_BUILD=1` restores upstream's "keep the BVH current while paused" for an instant
switch back).

**What the log should show** with `BB_PATHTRACING=1`: `[pt] geometry store: 8 M vertices a
frame x 2 ...` (the preset's size), `[pt] path tracing engine on ...`, `[pt] DLSS Ray
Reconstruction: BB_PT_DLSS_RR=auto (wanted) ...`, `Ray tracing: reflections on ... roughness <=
0.25` (low), once in game `Ray tracing: G-buffer targets: 0 ... 6 ...` and `Ray tracing: GI
albedo target ...`, at the first upscaled traced frame `[pt] DLSS Ray Reconstruction on:
1708x960 -> 2560x1440, linearised ...` (or `... off for this session: <why>`), then every 5 s
`[pt] geometry store, last 5 s: ... draws stored ..., N draws over the store, M small draws left
out ...`, `[pt] alpha-tested G-buffer draws, last 5 s: a with the shader's own reference literal,
b with it read from the material constants, c at 0.50 ...` and the GPU line `[pt] preset low,
on: path tracing GPU x.xxx ms/frame total over N traced frames (BVH build ..., GI ..., direct
..., reflections ...)`. "draws over the store" > 0: raise `BB_PT_STORE_MVERTS`. "GI skipped
(...)" lines name what is missing (scene colour, depth, camera, geometry).

## Knob names (clash with our older RT features)

Our tree already had `BB_RT_REFLECT`, `BB_RT_VOLUMETRIC`, `BB_RT_SHADOWS`, `BB_RT_DEBUG_VIEW`,
`BB_RT_REFLECT_DEBUG`, `BB_RT_REFLECT_STRENGTH`, ... (bb_rt_reflect.cpp). The mod uses
`BB_RT_REFLECT`, `BB_RT_SHADOWS`, `BB_RT_REFLECT_DEBUG`, `BB_RT_REFLECT_STRENGTH` too, with other
meanings. So **every mod knob is renamed `BB_RT_*` → `BB_PT_*`**, its master switch `BB_RT=1` →
`BB_PATHTRACING=1`, and its live config file `BB_RT_CFG` → `BB_PT_CFG`. Ours keep their names.
Running both (`BB_PATHTRACING=1` with one of ours) works but traces twice; the log warns.

## Quality presets (GPU cost)

`BB_PATHTRACING_QUALITY=low|medium|high|ultra` (default `low`; the launcher's selector writes this
variable). A preset only fills knobs you did not set (environment or the `BB_PT_CFG` live file
win). All presets first set the mod launcher's look defaults: `BB_PT_DIRECT=1 BB_PT_GI=1
BB_PT_COMPACT=0 BB_PT_FILL_LIGHT=off BB_PT_FILL_ENCLOSED=0.1 BB_PT_SKIN_ALPHA=1 BB_PT_FX_FOG=0
BB_PT_FX_FOG_G=0 BB_PT_GI_LAST_BOUNCE_CACHE=1 BB_PT_GI_SER=1 BB_PT_GI_PIPELINE=1
BB_PT_RESTIR_GI=1 BB_PT_KEEP_GI=1`. The table is `PtPresetTable` in `rt/pt_config.h`.

| knob | low | medium | high | ultra | what |
|---|---|---|---|---|---|
| `BB_PT_GI_SPARSE` | 4 | 2 | 1 | 1 | GI trace rate: 1 pixel of 2x2 / 2x1 / every pixel (ReSTIR + denoiser fill the rest) |
| `BB_PT_SPP` | 1 | 1 | 1 | 2 | GI paths per traced pixel |
| `BB_PT_BOUNCES` | 1 | 2 | 2 | 3 | GI bounces |
| `BB_PT_BOUNCE_RAYS` | 1 | 1 | 1 | 2 | shadow rays per bounce hit |
| `BB_PT_GI_TMAX` | 25 | 60 | 0 | 0 | GI ray length, metres (0 unlimited) |
| `BB_PT_GI_DENOISE` | 2 | 2 | 2 | 3 | built-in GI denoiser passes |
| `BB_PT_REFLECT` | 1 | 1 | 1 | 1 | traced reflections (one GGX ray per glossy pixel) |
| `BB_PT_REFLECT_MAX_ROUGH` | 0.25 | 0.4 | 0.5 | 0.6 | roughest surface that gets a reflection ray (G-buffer 1 gloss: roughness = 1 - gloss) |
| `BB_PT_RESTIR` | 0 | 0 | 1 | 1 | ReSTIR DI |
| `BB_PT_DIRECT_RAYS` | 2 | 8 | 32 | 48 | shadow rays per pixel for the local lights (strongest first) |
| `BB_PT_SHADOW_LIGHTS` | 2 | 3 | 4 | 6 | lights of the separate shadow-mask pass (off unless `BB_PT_SHADOWS=1`) |
| `BB_PT_LIGHT_SOURCE` | 0.02 | 0.02 | 0.02 | 0.02 | lamp / fire radius as a share of its range: soft shadows (0 = point) |
| `BB_PT_AO` | 0 | 0 | 0 | 0 | traced AO (the game's AO is kept) |
| `BB_PT_FAR_START` | 50 | 120 | 0 | 0 | beyond this distance (+10 m fade) no rays, stock lighting |
| `BB_PT_KEEP_DIST` | 20 | 40 | 40 | 60 | off-screen instances kept in the BVH within (m) |
| `BB_PT_MIN_TRIS` | 8 | 2 | 0 | 0 | opaque draws below this many triangles stay out of the BVH |
| `BB_PT_REFIT_BUDGET` | 131072 | 524288 | 0 | 0 | skinned BLAS refit triangles/frame (0 unlimited) |
| `BB_PT_REBUILD_BUDGET` | 393216 | 1048576 | 1048576 | 2097152 | BLAS rebuild triangles/frame |
| `BB_PT_STORE_MVERTS` | 8 | 10 | 12 | 14 | geometry store, M vertices a frame (x2 halves, 28 B each: 448 / 560 / 672 / 784 MB) |
| `BB_PT_MIST_STEPS` | 4 | 8 | 8 | 12 | water mist march steps |
| `BB_PT_MOON_HALF` | 2 | 2 | 1 | 1 | moon rays: 2 = half the pixels + shared penumbra, 1 = every pixel |
| `BB_PT_ADAPTIVE_MS` | 12 | 12 | 12 | 16 | frame budget: GI bounces drop while the frame is slower |
| `BB_PT_DLSS_RR` | auto | auto | auto | auto | DLSS Ray Reconstruction (below) |

"high" is the mod as its launcher shipped it, plus reflections up to roughness 0.5. Ray tracing
pipeline + SER (`BB_PT_GI_PIPELINE`, `BB_PT_GI_SER`) and opacity micromaps (`BB_PT_OMM`, default
on where supported) stay on in every preset: they make the same rays cheaper.

**Measured (run_pt1, RTX 5080, 1708x960 render, Central Yharnam, the old low preset):** path
tracing GPU 4.8-6.3 ms/frame = BVH build 0.7 (TLAS 0.16) + GI 1.9-2.7 + direct 1.7-2.8; reflections
0 (they were off in low). The new low halves the direct rays (4 -> 2) and trims GI ray length, BVH
budgets and trace distance, and turns on mirror-like reflections (wet floors, puddles, metal). Not measured yet: the next `[pt]` line gives the new numbers per pass.

**BVH reuse (already in the mod, kept):** BLAS are cached per draw identity across frames
(`rt_blas_cache.h`): static geometry is built once and only rebuilt when its vertices change;
skinned meshes are refit, not rebuilt (`BB_PT_MOVING_REFIT=1`, rebuilt after `BB_PT_REFIT_MAX`
refits), with per-frame refit/rebuild budgets above; the TLAS is rebuilt from the instance list
each frame (or partitioned updates with `BB_PT_PTLAS=1` on NVIDIA). Distance culling:
`BB_PT_KEEP_DIST`, `BB_PT_FAR_START`; small-geometry cull: `BB_PT_MIN_TRIS` (ours).

**Cost estimate by design (not measured yet in game).** Upper bound of
rays per pixel per frame, ignoring pixels beyond `BB_PT_FAR_START` and sky:

| preset | GI | direct (local lights + moon) | reflections | total, rays/px | 1080p internal | 1440p internal |
|---|---|---|---|---|---|---|
| low | 1/4 × (1 path + 1 bounce + 1 shadow) ≈ 0.75 | ≤ 4 + 1 | 0 | ≲ 6 | ≲ 12 M rays | ≲ 22 M rays |
| medium | 1/2 × (1 + 2 + 2) ≈ 2.5 | ≤ 12 + 1 | ≤ 1 | ≲ 17 | ≲ 35 M | ≲ 63 M |
| high | 1 × (1 + 2 + 2) = 5 + ReSTIR | ≤ 32 + 1 | ≤ 1 | ≲ 40 | ≲ 83 M | ≲ 150 M |

Direct rays are only spent where lights reach, so real counts are well below the bound. At a
practical 2–4 G incoherent rays/s (RTX 3070 class) that is roughly ≤ 3–6 ms (low), ≤ 9–18 ms
(medium), ≥ 20 ms (high) at 1080p, plus BVH maintenance (≈ 0.3–1 ms with the cache warm, a spike
on area loads). Treat these as ordering only; the log gives the real numbers.

**Measuring:** every 5 s (or every `BB_PT_GPU_LINE_FRAMES` frames if set) the log prints

    [pt] preset low, on: path tracing GPU 3.412 ms/frame total over 300 traced frames (BVH build 0.41 incl. TLAS 0.12, GI 1.20, direct 0.95, ReSTIR DI 0.00, reflections 0.40, shadow mask 0.00, AO 0.00, fog 0.00; ...)
    Ray tracing GPU ms/frame: build 0.41, GI 1.20, direct 0.95, ...

from GPU timestamp queries around each pass (`BB_PT_GPU_TIME=0` turns the queries off). The total
is per traced frame (all passes summed over the frames that ran any, the TLAS counted once inside
the build). Without a traced frame (menus, loading, F9) the line says "no traced frame since the
last line" instead of the old `0.000 ms/frame`. Not in it: the G-buffer vertex stores (the
geometry capture's vertex shader writes, part of the game's own G-buffer pass) and DLSS RR (the
`[perf]` line's `sr` ms is the upscaler pass, RR included). The whole-frame cost is the `[perf]`
line's `gpu` with path tracing on vs. paused (F9).

## Denoiser / DLSS Ray Reconstruction

**DLSS Ray Reconstruction (`rt/pt_rr.cpp`, bb-pathtracing-2).** With `BB_UPSCALER=dlss` and
NVIDIA's `nvngx_dlssd.dll` beside `shadPS4.exe` (or in its `streamline` folder, or
`BB_PT_DLSS_RR_DIR`), path traced frames go through DLSS-D instead of DLSS SR: RR denoises the
noisy traced light and upscales in one network. `BB_PT_DLSS_RR=auto` (every preset) or `1`: on
when possible; `0`: off. It is fed what the mod's RR path fed SuperMedo's bridge:

- colour: the game's tonemapped image, inverse-tonemapped to linear HDR first (the mod's
  `dlss_rr_tonemap.comp` linearize; `IsHDR` + auto exposure) and tonemapped back after
  (`BB_PT_DLSS_RR_HDR=0` gives RR the LDR image as is);
- guides (`vk_dlss_rr.cpp`, `dlss_rr_guides.comp`): diffuse albedo (G-buffer albedo), specular
  albedo (G-buffer 3 specular colour), normals + roughness packed (`Roughness.Mode` packed,
  roughness = 1 - G-buffer 1 gloss), the reflections' specular hit distance when the
  reflection pass ran this frame;
- depth (hardware, inverted as the game's), motion vectors (render pixels, `MVLowRes`), jitter
  (ours, `BB_DLSS_JITTER_SIGN` applies), world-to-view and view-to-clip matrices from the scene
  block camera;
- quality from the DLSS mode (DLAA / Quality / Balanced / Performance / Ultra Performance);
  `BB_PT_DLSS_RR_PRESET=D|E` picks the model (default: the runtime's).

The runtime is loaded directly, as `bb_dlss5.cpp` loads `nvngx_dlssnr.dll`: `nvngx_dlssd.dll`
(310.x) exports `NVSDK_NGX_VULKAN_Init_Ext / CreateFeature / EvaluateFeature / ReleaseFeature`
(checked on the 310.x DLL of our streamline folder), its caller check is answered the same way,
and its signature must be NVIDIA's (`BB_PT_DLSS_RR_UNOFFICIAL=1` skips that). Anything missing
(DLL, init, feature creation, an evaluate error) turns RR off for the session with one
`[pt] DLSS Ray Reconstruction off for this session: <why>` line and DLSS SR runs as before. With
Streamline DLSS-G on (`BB_FG=dlss-sl*`), RR stays off unless `BB_PT_DLSS_RR=1`: DLSS-G takes its
camera constants from the SR evaluate. F9 (paused) frames and frames without a G-buffer go
through SR; a switch resets RR's history.

The mod's own GI denoiser (temporal + `BB_PT_GI_DENOISE` à-trous passes) keeps running in both
cases; it is the fallback when RR is not available (other GPUs, other upscalers).

## What was ported

- The whole engine, unchanged in logic: `src/video_core/renderer_vulkan/rt/` (vk_ray_tracing.cpp
  11k lines, PTLAS, 25 headers, RR guides), 22 shader sources = 34 variants in
  `host_shaders/CMakeLists.txt` (`add_host_shader_stage` added for rgen/rchit/rahit/rmiss), its 17
  unit tests in `tests/pathtracing/`, glass masks in `data/pathtracing/glass_masks/`.
- `pt_compat.h`: stand-ins for SuperMedo's `ObjectMotion`, `SceneTargets` (we render at guest
  size, `reduced` is always false), `GpuProfiler` (none; the mod's own timestamps are used),
  `Dlss::Resource/RrGuides`, and `PtCapture` (where the vertex store addresses will go).
- `vk_instance.*` (from his vk_instance hunk): with `BB_PATHTRACING=1` the AS + ray query base we
  already had for bb-rt-reflect, plus descriptor indexing (partially bound, runtime arrays,
  non-uniform sampled-image indexing), opacity micromaps, ray tracing pipeline, SER and (opt-in)
  partitioned TLAS, and the `IsRayQuerySupported()`… queries the engine asks.
- `texture_cache.*`, `image.*` (his texture_cache hunk): `rt_hold_tick`, `HoldImage`,
  `EraseHeldImages` (a texture a cached BLAS still samples is not freed under the GPU), plus
  `TryGetImageRt` (his `TryGetImage` semantics: uid 0 = any) and `UniqueImage::TryCreate`.
  Without path tracing `rt_hold_tick` stays 0: unchanged behaviour.
- `vk_scheduler.h`: `WaitSubmitted()`.
- `sdl_window.cpp`: F9 (his window.cpp hunk).
- `vk_rasterizer.*`: owns `RayTracing` (created only with the switch).
- Ours: presets (`pt_config.h`), `BB_PT_MIN_TRIS`, zero-cost F9 pause, 5 s `[pt]` GPU line, log
  routed to our logger, `BB_PT_DLSS_RR` refused.

## Remaining work (precise; items 1-3 now done, see "Status")

1. **Geometry capture** (blocking). SuperMedo's shader recompiler, when a pipeline key has
   `motion_vectors` set, appends to every vertex shader a store of the clip-space position
   (vec4) into a device buffer at `positions_address + (store_base + vertex) * 16`, and the mod
   adds a half2 UV (`uvs_address`) and two snorm4x8 normals (`normals_address`) selected by a
   `uv_select` word (his `emit_spirv_special.cpp` hunk, `EmitVertexMotion`; `runtime_info.h`
   `MotionVectors`; `vk_object_motion.cpp` `PrepareDraw`/`StoreBase` allocates `store_base`
   per draw, 8 M positions per frame half). Our recompiler has a different per-draw variant
   (bb_objmv2, two clip positions to attachment 7) and no store. Needed: a pipeline variant that
   stores positions/UV/normals, a per-frame store allocator, and setting `PtCapture::*_address`.
   Plus his `shader_info_collection_pass.cpp` hunk (`FindAlbedo`, `FindAlphaSource`,
   `FindNormalMap`, persisted `rt_*` fields in `InfoPersistent`, pipeline-cache serialization).
2. **Camera**: `SceneInputs` wants the world camera (`inv_view` 3x4), projection, jitter and the
   scene depth image (his `CameraMotion::WorldCamera/Depth/CurrentProjection`). Our equivalents
   live in bb_upscale / bb_rt_reflect (the 864-byte scene block); an adapter is needed.
3. **Rasterizer hooks** (his vk_rasterizer.cpp hunk, ~1800 lines): `AddDraw` per triangle-list
   draw with `DrawTag`, alpha-test / prepass alpha inheritance, `OnDrawBound` (albedo / alpha /
   normal views of the draw), `OnGBufferPass`, `OnSceneColor`, the light-accumulation hooks
   (`OnLightDraw/Stencil/State/Targets/VtxParam/Buffer`, `SkipLightDraw`, `SkipHalfResLight`,
   `OnAccTargetDraw`, `OnFastClearEliminate`, `OnLightSet*Write`), `OnLightComposite` /
   `OnCompositeDraw/Input/Buffer`, AO (`OnAoInput/OnAoDraw/SkipGameAo`), shadow maps
   (`SkipShadowMaps`, `OnDirShadowInput`), `OnDispatch` (compute trigger), `OnDisplayPass` +
   `InvalidateHistory` at frame start, `OnSkyDome`, FX capture (`CaptureSmokePatch`,
   `CaptureGlow`, `AddSmoke/Flames/Fog/Glows`, water and fog sheets), `SkipPs`. Each maps to a
   spot in our `Rasterizer::Draw/DispatchDirect/PrepareRenderState`; the shader hashes in
   `rt_direct.h`/`rt_gi.h` are game constants and stay valid.
4. **Upscaler / presenter** (his vk_temporal_upscaler.cpp, vk_presenter.cpp hunks): the composite
   ordering hook and the RR evaluate described above.
5. Smaller: `liverpool.cpp` (light-set CB writes), `driver.cpp`, `vk_pipeline_cache.cpp` (UV
   select variant), `vk_staging_buffer_pool.cpp`, device-lost hook (`Instance::lost_hook` exists,
   our device-lost path does not call it yet), GPU checkpoints (not enabled).

## Verification done

bb-pathtracing-2 commit (RR, presets, soft shadows, alpha reference, store, timing):

- glslang 15 (`--target-env vulkan1.3`): `rt_direct.comp` (3 variants: direct, ReSTIR temporal,
  ReSTIR spatial) and `dlss_rr_guides.comp` compile.
- `clang++ -std=c++23 -fsyntax-only` clean with `-DENABLE_PATHTRACING`: pt_rr.cpp,
  vk_dlss_rr.cpp, vk_ray_tracing.cpp, vk_rasterizer_pt.cpp, pt_capture.cpp, vk_rasterizer.cpp,
  bb_present_sr.cpp, vk_pipeline_serialization.cpp, vk_pipeline_cache.cpp,
  shader_info_collection_pass.cpp; and without it: vk_rasterizer.cpp, bb_present_sr.cpp,
  vk_pipeline_serialization.cpp, vk_pipeline_cache.cpp, shader_info_collection_pass.cpp.
  pt_rr.cpp's Windows half against a Windows-target clang (`x86_64-pc-windows-msvc`) with our stub
  headers: only the stubs' missing PE / loader declarations fail, exactly as for bb_dlss5.cpp.
- `tests/pathtracing`: test_pt_config (ultra, unique rows), test_rt_restir_gi,
  test_rt_direct, test_rt_live pass.
- `nvngx_dlssd.dll` 310.x (our streamline folder) exports checked with a PE reader:
  `NVSDK_NGX_VULKAN_Init_Ext`, `CreateFeature`, `EvaluateFeature`, `ReleaseFeature` present, the
  same set as `nvngx_dlssnr.dll`. The NGX parameter names are the DLSS SDK 310.9 headers'
  (`nvsdk_ngx_helpers_dlssd_vk.h`, `nvsdk_ngx_defs*.h`).
- Ghidra (eboot 1.09, `FUN_026deeb0`, the light-set builder): each set colour is `rgb^2 x w`
  (squares of the four floats at +0x50, +0x60, +0xb0, +0xc0, +0x1e0, +0x1f0 times their w), as the
  mod's notes say; the point-light record the light shaders read (`[3]` = 1 / radius^2) is the
  mod's reading of the DPointLight shader and is what `LegacyAtten` uses.
- Not run in game, not built on Windows.

Capture / hooks commit: every changed or new C++ file `clang++ -std=c++23 -fsyntax-only` clean
with and without `-DENABLE_PATHTRACING` (vk_rasterizer.cpp, vk_rasterizer_pt.cpp, pt_capture.cpp,
emit_spirv_bb_pt.cpp, spirv_emit_context.cpp, emit_spirv*.cpp, shader_info_collection_pass.cpp,
vk_pipeline_cache.cpp, vk_pipeline_serialization.cpp, vk_graphics_pipeline.cpp, bb_upscale.cpp,
vk_ray_tracing.cpp). No host shader changed. The SPIR-V the new vertex variant emits has not been
through spirv-val or a driver yet (first run: watch for pipeline creation errors on G-buffer
draws; `BB_PATHTRACING=0` is the way back).

Port commit:

- All 34 shader variants compile with glslang 15.2 (`--target-env vulkan1.3`), through our real
  `host_shaders/CMakeLists.txt`.
- `vk_ray_tracing.cpp`, `vk_ray_tracing_ptlas.cpp`, `vk_dlss_rr.cpp`, `vk_rasterizer.cpp` (with and
  without `ENABLE_PATHTRACING`), `vk_instance.cpp`, `texture_cache.cpp`, `image.cpp`,
  `vk_scheduler.cpp`, `sdl_window.cpp`: `clang++ -std=c++23 -fsyntax-only` clean against our tree
  (and g++ 13 for the engine). Not linked, not run on Windows / clang-cl, not run in game.
- `tests/pathtracing`: 17 of 18 pass (g++); `test_rt_geometry` fails at
  `TestKeepInstances` exactly as it does on the untouched upstream commit.
