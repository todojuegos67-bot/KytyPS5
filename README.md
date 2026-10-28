# Injector (generic, phase 1)

An OptiScaler-style injector that works in any game. Its files go next to the game's exe. Press **F2** in game to open the menu and **F3** to toggle frame generation (both set in `injector.ini`: `menu_key`, `fg_key`).

Phase 1 covers **DX11 games through DXVK**, with these features:

- an ImGui overlay;
- FSR 3.1 frame generation x2 with pacing;
- a live "Analizar vectores" analyzer that finds the depth buffer, the motion vectors and the HUD point;
- a log next to the exe.

The code contains no game-specific values: no offsets, no shader hashes, no exe names. Knowledge about a particular game lives only in the profiles that the analyzer writes (`injector_profiles\<exe>.ini`). Those profiles hold image signatures (format and size), never handles or addresses.

## How it works

```
game (DX11) -> d3d11.dll / dxgi.dll (DXVK 2.6.1, official) -> winevulkan.dll (OURS, next to the exe)
                                                               -> %SystemRoot%\System32\vulkan-1.dll (system loader)
```

DXVK tries `winevulkan.dll` before `vulkan-1.dll`. No Windows module has that name, so ours is always the one DXVK gets, even when an overlay (RTSS, Ubisoft) already loaded the system `vulkan-1.dll` (the 10-28 Wildlands failure). Native Vulkan games get the same binary as `vulkan-1.dll`. Our proxy forwards every call to the system loader and intercepts only a few:

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
   - **DX11:** copies `d3d11.dll`, `dxgi.dll`, `winevulkan.dll`, `amd_fidelityfx_vk.dll`, `dxvk.conf` and `injector.ini` next to the exe. A `vulkan-1.dll` left by an older install is moved aside. Files already there are renamed to `*.injector-bak`. Everything is recorded in `injector_install.txt`.
   - **DX12:** nothing is copied. DX12 is phase 2: the same DLL behind vkd3d-proton.
   - **Vulkan:** copies only `vulkan-1.dll`, `amd_fidelityfx_vk.dll` and `injector.ini`, and only if the exe gets Vulkan functions through `vkGetInstanceProcAddr`. An exe that imports other `vk*` functions directly is refused.
   - **32-bit, DX9 or unknown:** refused.
4. Uninstall: run `InjectorSetup.exe` again on the same exe. It deletes the listed files and restores the backups. `injector.ini`, `injector.log` and the profiles are kept.

The setup can also be scripted:

```
InjectorSetup.exe detect|install|uninstall <game.exe>
```

To install by hand for a DX11 game, copy the six files from `payload\` next to the exe.

## The menu (F2)

F3 toggles frame generation without opening the menu. Settings are saved to `injector.ini` when the menu closes.

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
- **Upscaler:** phase 2. It will be fed from the same profile data.

## Log (`injector.log` next to the exe): the lines that show it works

```
... device: NVIDIA GeForce RTX 5080 (Vulkan 1.4.x ...)
... frame generation queues: created (graphics family 0 present index 1, compute family 2 from index 0)
... [fg] FidelityFX API loaded (amd_fidelityfx_vk.dll)
... [fg] frame interpolation swapchain 1920x1080, format 44, present mode 0, 3 images
... swapchain 1920x1080 format 44, 3 images, FidelityFX frame interpolation swapchain
... [overlay] ImGui ready (F2 menu, F3 frame generation on/off)
... [fg] generation context 1920x1080 (render up to 1920x1080, format 44, depth normal)
... [fg] inputs: neutral (far depth, zero motion): FSR 3 interpolates from optical flow only
... [fg] generation on: x2, render 1920x1080 -> display 1920x1080
... [fg] real 90.0 fps -> shown 180.0 fps (x2.00) | generated 90 | on (optical flow only) | present 0.40 ms | cap 224 fps, held 0.0 ms
... [analyze] frame 300: 140 passes; depth found 0.93 (fmt 130 1920x1080, cleared to 0: reversed Z); motion found 0.85 (...)
... [fg] inputs: the game's depth (fmt 130 1920x1080, reversed) and motion vectors (fmt 83) from the analyzer
```

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
  - there is no HUD-less image injection yet. The HUD point is found and reported but not fed to FSR (phase 2).
- **Unknown camera.** FOV is fixed at 1 rad, and the near/far planes are defaults. A per-profile camera is phase 2.
- **Overlay inside generated frames.** The overlay and OSD are drawn into the real frame, so FSR interpolates them too.
- **Input is not blocked.** Mouse clicks in the menu also reach the game.
- **Pacing.**
  - **What it does:** FidelityFX paces the generated frames, our cap holds the real ones (KytyPS5's VRR rule), and `dxvk.conf` sets `dxgi.maxFrameLatency = 1`.
  - **What it lacks:** no Reflex / Anti-Lag 2 markers.
- **Single swapchain.** One swapchain and one device per process are assumed.

## Phase 2 plan (in code: `analyzer.h`, `overlay.cpp` "Upscaler (fase 2)", the launcher's DX12 branch)

- **DX12:** vkd3d-proton (official release) in place of DXVK, with the same `vulkan-1.dll`.
- **Upscaler:** FSR 3.1 upscaling (FidelityFX, public SDK), XeSS (Intel's public SDK) and DLSS through Streamline (NVIDIA's public SDK). They will be fed from the profile: depth, motion vectors, render size, and the jitter estimated from the projection.
- **HUD-less:** a copy of the backbuffer at the HUD point, made inside the game's command buffer, registered as `HUDLessColor`.
- **Profiles:** add the camera FOV / near / far and the MV scale found by reprojection error.

## Licenses

- DXVK: zlib.
- FidelityFX SDK and its signed DLL: MIT.
- Dear ImGui: MIT.
- Vulkan-Headers: Apache-2.0 / MIT.

Only public SDKs and official releases are used. No leaked DLLs.
