# Injector (generic, phases 1 and 2)

An OptiScaler-style injector that works in any game. Its files go next to the game's exe. Press **F1** in game to open the menu.

It covers **DX11 games through DXVK**, with these features:

- an ImGui overlay;
- FSR 3.1 frame generation x2 with pacing;
- a live "Analizar vectores" analyzer that finds the depth buffer, the motion vectors and the HUD point;
- (phase 2) upscaling: **DLSS** (NVIDIA Streamline), **FSR 4**, **FSR 3.1** and **XeSS**, with quality modes, RCAS sharpening
  at the display size and a live A/B (F3);
- a log next to the exe.

The code contains no game-specific values: no offsets, no shader hashes, no exe names. Knowledge about a particular game lives only in the profiles that the analyzer writes (`injector_profiles\<exe>.ini`). Those profiles hold image signatures (format and size), never handles or addresses.

## How it works

```
game (DX11) -> d3d11.dll / dxgi.dll (DXVK 2.6.1, official) -> vulkan-1.dll (OURS, next to the exe)
                                                               -> %SystemRoot%\System32\vulkan-1.dll (system loader)
```

DXVK loads `vulkan-1.dll` from the exe's folder first. Our proxy forwards every call to the system loader and intercepts only a few:

- **Device creation:** adds the three queues that FSR 3 frame generation needs (an extra graphics queue for present, plus two in a compute-only family).
- **Swapchain:** replaced by the FidelityFX frame interpolation swapchain (the same mechanism as KytyPS5's `src/local/fsr3-fg.cpp`).
- **Present:**
  1. copies the analyzer's resources;
  2. draws the overlay;
  3. configures FSR 3 and records its prepare pass;
  4. paces the frame;
  5. presents.
- **Image creation, `vkCmdBeginRendering` and pipeline barriers:** feed the live analyzer.

## Install (InjectorSetup.exe)

1. Unzip the release anywhere.
2. Run `InjectorSetup.exe` and pick the game's exe. For Wildlands that is `GRW.exe`, in the Ubisoft Connect / Steam install folder.
3. The setup reads the exe's import table and detects the API:
   - **DX11:** copies `d3d11.dll`, `dxgi.dll`, `vulkan-1.dll`, `amd_fidelityfx_vk.dll`, `dxvk.conf` and `injector.ini` next to the exe, and (phase 2, when the package has them) `libxess.dll`, the `streamline\` folder (NVIDIA's `sl.interposer.dll`, `sl.common.dll`, `sl.dlss.dll`, `nvngx_dlss.dll`) and the `fsr4_shaders\` folder (AMD's FSR 4 model). Files already there are renamed to `*.injector-bak`. Everything is recorded in `injector_install.txt`.
   - **DX12:** nothing is copied. DX12 is a later phase: the same DLL behind vkd3d-proton.
   - **Vulkan:** copies only `vulkan-1.dll`, `amd_fidelityfx_vk.dll` and `injector.ini`, and only if the exe gets Vulkan functions through `vkGetInstanceProcAddr`. An exe that imports other `vk*` functions directly is refused.
   - **32-bit, DX9 or unknown:** refused.
4. Uninstall: run `InjectorSetup.exe` again on the same exe. It deletes the listed files and restores the backups. `injector.ini`, `injector.log` and the profiles are kept.

The setup can also be scripted:

```
InjectorSetup.exe detect|install|uninstall <game.exe>
```

To install by hand for a DX11 game, copy everything in `payload\` (files and the `streamline\` and `fsr4_shaders\` folders) next to the exe.

## The menu (F1)

F2 toggles frame generation without opening the menu. Settings are saved to `injector.ini` when the menu closes.

- **Frame generation:**
  - on/off;
  - the VRR cap (`-1` = auto, refresh − refresh²/3600: 240 Hz gives 224 shown fps, so at most 112 real fps);
  - state, real fps vs shown fps, and generated frames per second;
  - which inputs are in use: the game's depth and motion vectors, or neutral inputs.
- **Analizar vectores** (live, every frame): for depth, motion vectors and HUD it shows a confidence %, the detected format and size, and a live preview. You can also set:
  - the minimum confidence;
  - "use the game's depth/MV in FG";
  - MV units (UV or pixels) and sign;
  - the depth convention (auto, which reads the depth clear value: 0 means reversed-Z, 1 means normal).

  Once both depth and motion vectors are above the minimum confidence and have the same size, FSR 3 uses them on the spot. Until then it uses neutral inputs (far depth, zero motion), so FSR 3 interpolates from its own optical flow only.

  The profile is rewritten while you play, at most every 5 s and only when something changed. On the next start it locks on within about 10 frames.
- **Reescalado** (phase 2): see below.

## Upscaling (phase 2, F1 > Reescalado)

### What to do in the game (Wildlands)

1. In the game's own video options, keep its anti-aliasing / TAA on and set its **resolution scale below 100%**
   (Quality ~67%, Balanced ~58%, Performance 50%, Ultra Performance 33%). The injector does not shrink the game's render
   targets itself (see "Render size" below): the game renders smaller, the injector upscales.
2. F1 > Reescalado: pick **DLSS**, **FSR 4**, **FSR 3.1** or **XeSS**; "Modo" Auto follows the game's scale.
3. Watch the status lines in the menu (render size -> display size, inputs, jitter, GPU ms, sharpness metric).
   **F3** (or the checkbox) toggles A/B: the game's own image on screen while the upscaler keeps running.
   "Pantalla dividida": left half upscaled, right half the game's.

### How it works

- **Render size, path (b).** The game's own resolution-scale option sets the render size; the analyzer sees a main
  depth smaller than the swapchain and the scene image at that size. Path (a) (OptiScaler-style: shrinking the game's
  render targets, viewports and scissors underneath it) was **not** taken: under DXVK the injector only sees Vulkan
  calls, and the game's shaders address their targets through constants and `SV_Position` math sized for what they
  asked for (half-resolution effects, full-screen passes, UV clamps), so shrinking them below DXVK breaks
  post-processing in ways no generic rule can find. Path (b) is the simple robust one.
- **Inputs (analyzer).** Depth and motion vectors as for frame generation (same confidence, MV units / sign, depth
  convention); the **scene image** = the last render-size colour target written before the composite (the
  backbuffer's first pass). Optional **pre-TAA** input: the analyzer finds the game's TAA by its history ping-pong
  (two render-size images written on alternate frames) and takes the colour image written just before it.
- **Jitter (camera finder, `src/camera.cpp`).** The constant buffers the scene draws read (DXVK binds them through
  descriptor update templates; we map buffers to their host memory) are scanned for a perspective matrix (either
  memory order). Its jitter terms over the last 32 frames: several distinct sub-pixel values = the game's TAA jitter,
  "detected". Into a post-TAA scene image the jitter is already resolved, so 0 is given unless the game's TAA is not
  seen; pre-TAA input takes the detected jitter. "Inyectar Halton" (with the game's TAA off): a Halton(2,3) offset is
  written into the projection in mapped memory before the GPU reads it. The same matrix gives the FOV and near / far.
  A game that only uploads a combined view-projection matrix is not seen (then: no jitter known, 0).
- **Placement (HUD point).** Right before the second pass into the backbuffer begins (the first UI pass), inside DXVK's
  command buffer: the scene image, depth and motion vectors are copied into our images, the upscaler runs into a
  display-size RGBA16F image, RCAS sharpens it at the display size, and it is blitted over the game's composite. The
  UI is then drawn on top as before. With no separate UI pass it is done right before DXVK's present blit (the HUD is
  covered; the menu says so). DXVK's compute pipeline / descriptor sets / push constants are re-bound afterwards.
- **Backends, all on Vulkan:**
  - **DLSS:** NVIDIA Streamline v2.14.16 (public SDK release; only its production DLLs, no others) in manual-hooking
    mode: `slInit` before the game's instance, the extensions / features `slGetFeatureRequirements` lists, then
    `slSetVulkanInfo` with the device that presents. Tags, constants and evaluate as KytyPS5's DLSS path.
  - **FSR 4:** KytyPS5's own runtime (`src/local/fsr4-sr.cpp`) ported to plain Vulkan: AMD's v07 INT8 model (`fsr4_shaders\`,
    MIT), every module checked against its manifest, 14 passes. Needs shaderIntegerDotProduct, int8, int16, fp16 and
    compute derivatives (RTX 40/50 and RDNA 3/4 have them).
  - **FSR 3.1:** the FidelityFX API upscaler in `amd_fidelityfx_vk.dll` (the same DLL frame generation uses).
  - **XeSS:** Intel XeSS SDK v3.0.2 `libxess.dll` (DP4a path on non-Intel GPUs); its instance / device extensions and
    features are added at creation.
- **Sharpening:** one RCAS pass (FSR 1's algorithm, our GLSL) at the display size after every upscaler; the
  upscalers' own sharpening is off.
- **Conventions (KytyPS5's):** motion vectors scaled to render pixels toward the previous frame (UV units x size x sign),
  jitter in pixels (FidelityFX convention), depth inverted flags from the analyzer (FSR 4 flips reversed depth),
  exposure 1 (auto-exposure for HDR input).

## Log (`injector.log` next to the exe): the lines that show it works

```
... device: NVIDIA GeForce RTX 5080 (Vulkan 1.4.x ...)
... frame generation queues: created (graphics family 0 present index 1, compute family 2 from index 0)
... [fg] FidelityFX API loaded (amd_fidelityfx_vk.dll)
... [fg] frame interpolation swapchain 1920x1080, format 44, present mode 0, 3 images
... swapchain 1920x1080 format 44, 3 images, FidelityFX frame interpolation swapchain
... [overlay] ImGui ready (F1 menu, F2 frame generation on/off)
... [fg] generation context 1920x1080 (render up to 1920x1080, format 44, depth normal)
... [fg] inputs: neutral (far depth, zero motion): FSR 3 interpolates from optical flow only
... [fg] generation on: x2, render 1920x1080 -> display 1920x1080
... [fg] real 90.0 fps -> shown 180.0 fps (x2.00) | generated 90 | on (optical flow only) | present 0.40 ms | cap 224 fps, held 0.0 ms
... [analyze] frame 300: 140 passes; depth found 0.93 (fmt 130 1920x1080, cleared to 0: reversed Z); motion found 0.85 (...)
... [fg] inputs: the game's depth (fmt 130 1920x1080, reversed) and motion vectors (fmt 83) from the analyzer
```

Upscaling (phase 2) lines, per upscaler:

```
... [dlss] Streamline 2.14.16 loaded from ...\streamline\ (manual hooking): DLSS asks for ...
... [up] device extension added: VK_KHR_push_descriptor (our passes, FSR 4)
... [fsr4] device NVIDIA GeForce RTX 5080: shaderIntegerDotProduct yes (...), computeDerivativeGroupLinear yes
... [dlss] DLSS ready on this device (Streamline told about it with slSetVulkanInfo)
... [camera] projection found (form T, 412 draws, 384 ranges scanned): fov 60.0 deg, near 0.100 far inf, reversed depth, jitter detected (...)
... [up] DLSS on: 1280x720 -> 1920x1080 (x1.50, Quality, DLSS mode Quality ...) | colour scene (post-TAA) fmt 97 1280x720 HDR | depth fmt 130 1280x720 reversed | motion fmt 83 1280x720, scale (-1280, -720) | jitter none (...) | RCAS 0.30 at 1920x1080 | placement: before the first UI pass (backbuffer pass 2)
... [dlss] first evaluate: 1280x720 -> 1920x1080, ...
... [up] DLSS frame 60: 1280x720 -> 1920x1080 (Quality) | gpu 0.85 ms (copies + upscale + RCAS + blit) | jitter none (...) | sharpness metric (mean |laplacian| of luma, centre 256x256): game 0.01234 -> ours 0.01890 (+53%)
```

The same with `[fsr4] first frame`, `[fsr3.1] first dispatch` or `[xess] first execute`. The proof that an upscaler
ran and its image reached the screen is the pair `[<backend>] first ...` + `[up] <backend> frame N: ... sharpness
metric: game X -> ours Y`: the metric is measured on the backbuffer before our blit (the game's own stretch) and after
it (what is on screen), so a different value means our image is there. With F3 (A/B original) on, "ours" is not measured.
(Again, the values above only show the format: nothing has run on a real PC yet.)

The `[fg] real … -> shown …` line is the proof that generated frames reach the screen:

- `shown` is the FidelityFX swapchain's own count of presents to the display (real plus generated);
- `generated` is the number of generation dispatches FidelityFX asked for in that second.

With frame generation working, `shown` is about 2 × `real` and `generated` is about `real`. With FG off, `shown` equals `real` and `generated` is 0.

(The values above are only an example of the format. Nothing has been run on a real PC yet; see "Untested".)

## Anti-cheat (Wildlands)

Ghost Recon Wildlands ships **EasyAntiCheat** (Ghost War PvP, error 30005 reports). The setup warns you whenever it finds an EasyAntiCheat or BattlEye folder or launcher next to the exe.

- Use the injector **offline / single-player only**: Ubisoft Connect offline mode, campaign only, never Ghost War.
- Replacing `d3d11.dll` / `dxgi.dll` can make EAC refuse to start the game, or flag the account when you play online.
- **Not verified:** whether GRW.exe starts at all without EAC, and whether the campaign runs with EAC active plus DXVK. It has to be tried on the owner's PC, offline first.

## Untested / known limits (be honest)

- **Untested.** Nothing has run in a game: only the CI build is checked. No frame of Wildlands has been seen with this injector.
- **Analyzer scope.** It only sees `vkCmdBeginRendering` passes (DXVK 2.x) and barriers:
  - motion vectors written only by compute shaders are found only through their barrier to GENERAL (weaker);
  - there is no HUD-less image injection yet. The HUD point is used by the upscalers (phase 2) but not fed to frame generation as HUD-less colour.
- **Camera for frame generation.** FOV is fixed at 1 rad and near / far are defaults for FSR 3 FG; the upscalers use the camera finder (phase 2).
- **Overlay inside generated frames.** The overlay and OSD are drawn into the real frame, so FSR interpolates them too.
- **Input is not blocked.** Mouse clicks in the menu also reach the game.
- **Pacing.**
  - **What it does:** FidelityFX paces the generated frames, our cap holds the real ones (KytyPS5's VRR rule), and `dxvk.conf` sets `dxgi.maxFrameLatency = 1`.
  - **What it lacks:** no Reflex / Anti-Lag 2 markers.
- **Single swapchain.** One swapchain and one device per process are assumed.

## Upscaling: untested / known limits (be honest)

- **Nothing has run in a game.** Only the CI build (MSVC) and a local syntax check are done. No upscaled frame of
  Wildlands has been seen; every log line above is the expected format, not an observed one.
- **DLSS:** Streamline in manual-hooking mode on Vulkan without its present / swapchain hooks is not NVIDIA's usual
  integration; whether `slSetVulkanInfo` after DXVK's probe device, and NGX on a device DXVK made, work is unverified.
  Streamline loads "vulkan-1.dll" by name and gets ours (we pass its calls straight through).
- **Placement:** relies on the analyzer's backbuffer / HUD point; a game that draws its UI in the same pass as the
  composite gets its HUD covered (logged). Image layouts at that point are taken from the barriers DXVK recorded.
- **Inputs:** the default input is the game's post-TAA scene image (so the upscaler mostly does a better stretch + its
  own temporal AA, jitter 0). Pre-TAA input is experimental: the game's tone mapping / grading after its TAA is then
  skipped. Motion vectors only when the analyzer is confident and they are at the render size (else zero vectors).
- **Jitter:** detection needs the projection matrix on its own in a constant buffer; a combined view-projection is not
  recognised. Halton injection writes into the game's mapped constant memory (only into a matrix with no jitter of its
  own) and is only meaningful with the game's TAA off.
- **No reactive / transparency masks, no exposure texture,** no HUD-less colour for frame generation yet.
- **DX12:** still phase 3 (vkd3d-proton).

## Licenses

- DXVK: zlib.
- FidelityFX SDK and its signed DLL: MIT. FSR 4 v07 model (`fsr4_shaders\`): MIT (AMD).
- NVIDIA Streamline / DLSS: NVIDIA's license (in the Streamline release; `licenses\`).
- Intel XeSS SDK: Intel's license (`licenses\XeSS-LICENSE.txt`).
- nlohmann/json: MIT.
- Dear ImGui: MIT.
- Vulkan-Headers: Apache-2.0 / MIT.

Only public SDKs and official releases are used. No leaked DLLs.
