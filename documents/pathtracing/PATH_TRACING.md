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

**Geometry capture, camera and the rasterizer hooks are wired; not yet run in game.** With
`BB_PATHTRACING=1` the engine starts, a geometry store is allocated, and opaque G-buffer draws
feed the BVH every frame; GI is traced after the light composite, direct light at the light
accumulation hook, shadows / sky / fog / exposure at the post trigger dispatch. Nothing was run on
hardware from this tree (built and checked only: see "Verification"). With the switch off (the
default) nothing of it runs: no object, no extension, no shader variant, no memory.

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
captured), DLSS Ray Reconstruction (below), device-lost hook.

## How to enable

    BB_PATHTRACING=1                     # master switch, default off
    BB_PATHTRACING_QUALITY=low           # low (default) | medium | high
    BB_PT_<knob>=<value>                 # any knob below or in PATH_TRACING-upstream.md

Needs a GPU with `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` + bufferDeviceAddress
(GeForce RTX, Radeon RX 6000+, Arc). CMake option `ENABLE_PATHTRACING` (default ON) builds the
engine and its 34 shaders; `-DENABLE_PATHTRACING=OFF` leaves them out entirely.

**F9** (only with `BB_PATHTRACING=1`) switches path tracing off/on in game, as in the mod. It
takes F9 from shadPS4's `hotkey_pause` while the switch is on (remap that hotkey if you need it).
Off costs no GPU time: the passes and, unlike upstream, the BVH build are skipped
(`BB_PT_PAUSED_BUILD=1` restores upstream's "keep the BVH current while paused" for an instant
switch back).

**What the log should show** with `BB_PATHTRACING=1`: `[pt] geometry store: 6 M vertices a
frame x 2 ...`, `[pt] path tracing engine on ...`, once in game `Ray tracing: G-buffer targets:
0 ... 6 ...` and `Ray tracing: GI albedo target ...`, then every 5 s `[pt] geometry store, last
5 s: ... draws stored ...` and the GPU line `[pt] preset low, on: path tracing GPU x.xxx
ms/frame`. "draws over the store" > 0: raise `BB_PT_STORE_MVERTS`. "GI skipped (...)" lines name
what is missing (scene colour, depth, camera, geometry).

## Knob names (clash with our older RT features)

Our tree already had `BB_RT_REFLECT`, `BB_RT_VOLUMETRIC`, `BB_RT_SHADOWS`, `BB_RT_DEBUG_VIEW`,
`BB_RT_REFLECT_DEBUG`, `BB_RT_REFLECT_STRENGTH`, ... (bb_rt_reflect.cpp). The mod uses
`BB_RT_REFLECT`, `BB_RT_SHADOWS`, `BB_RT_REFLECT_DEBUG`, `BB_RT_REFLECT_STRENGTH` too, with other
meanings. So **every mod knob is renamed `BB_RT_*` → `BB_PT_*`**, its master switch `BB_RT=1` →
`BB_PATHTRACING=1`, and its live config file `BB_RT_CFG` → `BB_PT_CFG`. Ours keep their names.
Running both (`BB_PATHTRACING=1` with one of ours) works but traces twice; the log warns.

## Quality presets (GPU cost)

A preset only fills knobs you did not set (environment or the `BB_PT_CFG` live file win). All
presets first set the mod launcher's look defaults: `BB_PT_DIRECT=1 BB_PT_GI=1 BB_PT_COMPACT=0
BB_PT_FILL_LIGHT=off BB_PT_FILL_ENCLOSED=0.1 BB_PT_SKIN_ALPHA=1 BB_PT_FX_FOG=0 BB_PT_FX_FOG_G=0
BB_PT_GI_LAST_BOUNCE_CACHE=1 BB_PT_GI_SER=1 BB_PT_GI_PIPELINE=1 BB_PT_RESTIR_GI=1 BB_PT_KEEP_GI=1`.

| knob | low | medium | high | what |
|---|---|---|---|---|
| `BB_PT_GI_SPARSE` | 4 | 2 | 1 | GI trace rate: 1 pixel of 2x2 / 2x1 / every pixel (ReSTIR + denoiser fill the rest) |
| `BB_PT_SPP` | 1 | 1 | 1 | GI paths per traced pixel |
| `BB_PT_BOUNCES` | 1 | 2 | 2 | GI bounces |
| `BB_PT_BOUNCE_RAYS` | 1 | 1 | 1 | shadow rays per bounce hit |
| `BB_PT_GI_TMAX` | 30 | 60 | 0 | GI ray length, metres (0 unlimited) |
| `BB_PT_REFLECT` | 0 | 1 | 1 | traced reflections (water, puddles, glass) |
| `BB_PT_RESTIR` | 0 | 0 | 1 | ReSTIR DI |
| `BB_PT_DIRECT_RAYS` | 4 | 12 | 32 | shadow rays per pixel for the local lights |
| `BB_PT_SHADOW_LIGHTS` | 2 | 3 | 4 | lights with traced shadows per pixel |
| `BB_PT_AO` | 0 | 0 | 0 | traced AO (the game's AO is kept) |
| `BB_PT_FAR_START` | 60 | 120 | 0 | beyond this distance (+10 m fade) no rays, stock lighting |
| `BB_PT_KEEP_DIST` | 25 | 40 | 40 | off-screen instances kept in the BVH within (m) |
| `BB_PT_MIN_TRIS` | 4 | 2 | 0 | opaque draws below this many triangles stay out of the BVH (new) |
| `BB_PT_REFIT_BUDGET` | 262144 | 524288 | 0 | skinned BLAS refit triangles/frame (0 unlimited) |
| `BB_PT_REBUILD_BUDGET` | 524288 | 1048576 | 1048576 | BLAS rebuild triangles/frame |
| `BB_PT_MIST_STEPS` | 4 | 8 | 8 | water mist march steps |
| `BB_PT_ADAPTIVE_MS` | 12 | 12 | 12 | frame budget: GI bounces drop while the frame is slower |

"high" is the mod as its launcher shipped it. Ray tracing pipeline + SER (`BB_PT_GI_PIPELINE`,
`BB_PT_GI_SER`) and opacity micromaps (`BB_PT_OMM`, default on where supported) stay on in every
preset: they make the same rays cheaper.

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

    [pt] preset low, on: path tracing GPU 3.412 ms/frame total (0 frames paused since the last line)
    Ray tracing GPU ms/frame: build 0.41, GI 1.20, direct 0.95, ReSTIR 0.00, reflect 0.00, shadows ..., TLAS ...

from GPU timestamp queries around each pass (`BB_PT_GPU_TIME=0` turns the queries off).
The mod's lines go to our log (they went to stdout upstream).

## Denoiser / DLSS Ray Reconstruction

Not hosted yet: `BB_PT_DLSS_RR=1` is ignored with a log line, and the mod's own GI denoiser
(temporal + `BB_PT_GI_DENOISE` à-trous passes, default 2) is used. The RR guide generation
(`rt/vk_dlss_rr.cpp`, `dlss_rr_guides.comp`, `dlss_rr_tonemap.comp`) is ported and compiles; what is
missing is the DLSS-D evaluate: SuperMedo's NGX bridge got `CreateRrFeature/EvaluateRr`
(`NVSDK_NGX_Feature_RayReconstruction`, `nvngx_dlssd.dll`), our DLSS SR goes through Streamline
(`bb_sr_dlss.cpp`), which would need the `sl.dlss_d` plugin and `slDLSSDSetOptions` /
`slEvaluateFeature(kFeatureDLSS_RR)` with the guides tagged (diffuse/specular albedo,
normal-roughness, specular hit distance, world-to-view / view-to-clip matrices), replacing SR for
frames where path tracing ran. `nvngx_dlssd.dll` stays user-supplied.

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
