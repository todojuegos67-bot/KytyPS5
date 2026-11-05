# KytyPS5 portable test build (Windows)

A Windows test build of the Kyty PS5 emulator. So far it has been tuned only for
**Demon's Souls (the PS5 remake)**. It is built by GitHub Actions from the source at
https://github.com/chenxiao07/KytyPS5, where newer builds are under Releases.

## Supported game versions

| Item | Supported |
| --- | --- |
| Game | Demon's Souls |
| Title ID | PPSA01341 |
| Content ID | EP9000-PPSA01341_00-DEMONSSOULS00000 |
| Versions | 01.007.000 (1.07) and 01.005.000 (1.05) |

Both versions get the same performance patches. Each version has shader caches of its own (versions do
not share shaders), so the shader precompile is done once per version. Other versions, other regions
and other games have not been tested and may not run or may fail. At start-up the launcher reads the
game's `sce_sys\param.json` and warns when the version is not one of these.

This dump is the European release, whose language list has no Chinese, Japanese or Korean; the game files
hold every language, and one value turns the others on: see [Game language](#game-language-chinese-japanese-korean-and-others).

## System requirements

| Item | Requirement | Notes |
| --- | --- | --- |
| OS | Windows 10 (1803 or later) or Windows 11, 64-bit | |
| CPU | AVX2 required (Intel Core 4th generation / AMD Ryzen or later) | 8 cores or more recommended; the test PC has an i9-14900K |
| RAM | **32 GB recommended, 16 GB works** | The game process takes about 20 GB while a save loads but uses only about 4 GB of it in play, so with 16 GB Windows moves the rest to the page file and loading takes longer. It needs about 34 GB of RAM plus page file: keep the page file "system managed" or set it to at least 32 GB (the launcher warns when Windows cannot provide it). The MEMORY_MANAGEMENT blue screen that PCs whose RAM ran full got (issue #50) is fixed; a full RAM still makes the game slow, so close other large programs (a browser) while playing |
| GPU | NVIDIA RTX with **12 GB of VRAM recommended, 8 GB works** | About 11 GB of VRAM in use in the heaviest areas. With less than 12 GB, render targets and buffers keep the video memory and the textures that do not fit are read from system memory. The test PC with its video memory limited to 8 GB, walking past every enemy of a map (up to 120 fps): Boletaria's gate about 72 fps, Boletaria 1-2 98, 1-3 64 (its own limit on the processor, the same with all the video memory), Stonefang 2-1 87, Shrine of Storms 96, about what it runs with all its video memory. That measures where the textures go, not a slower GPU: an 8 GB card is usually a slower one too, and gives less. The test PC has an RTX 5090; users run it on AMD Radeon (RX 6000 and newer) and Intel Arc cards too, with problems of their own tracked in the issues |
| GPU driver | The latest | The emulator uses Vulkan, which comes with the driver |
| Disk | About 83 GB for the game, an SSD recommended | The emulator takes about 270 MB, the shader caches about 2.5–5 GB, and while playing the system temp folder needs another 4–5 GB |

On the test PC (i9-14900K + RTX 5090, 2560×1440 window) the game holds its 60 fps in most places; the
heaviest spots (Boletaria's gate area, the Tower of Latria) run at about 57–60 fps. With "Up to 120 fps"
on (see the launcher below), most areas run at 70–100 fps; the release notes list every area. Much
slower PCs may not run it or may stutter badly.

On a CPU with performance and efficiency cores (Intel Core 12th generation and later), the launcher
keeps the emulator's render threads on the performance cores (all but the one of CPU 0) by itself,
which was about 2% faster than leaving them to Windows; the console shows them as "render threads on
CPUs ...". To leave everything to Windows instead: `run.cmd -Set KYTY_RECORDING_CPUS=,KYTY_RENDER_CPUS=`.

The game starts with the table path: most draws and compute dispatches read their resources straight
from GPU memory instead of descriptor sets the emulator builds for each of them (Boletaria standing
about 45 → 50–53 fps). The first visit to an area compiles its pipelines in the background, so the
gain grows as you play. To start without it: `run.cmd -Set KYTY_TABLE_XPR=0,KYTY_TABLE_DISPATCH=0`.

## Where to put the game files

You need the unpacked PS5 game folder: the level that directly contains `eboot.bin` and the `sce_sys`
folder (usually named `PPSA01341-app0`).

- It can be anywhere; it does not need to be inside the emulator folder. An SSD is recommended.
- The first launch opens a folder picker: select that folder. It is remembered in `game-path.txt`,
  so you only choose it once.
- To use another location: delete `game-path.txt` and launch again, or run
  `run.cmd -Game "D:\Games\PPSA01341-app0"`.
- Optional: the folder packed into one ZArchive file (`.zar`, about two thirds of its size) works
  too, read without extracting it: choose it with the launcher's `.zar...` button, or run
  `run.cmd -Game "D:\Games\PPSA01341-app0.zar"`. Pack it with `zarchive.exe <folder> <file.zar>`
  (ZArchive, https://github.com/Exzap/ZArchive/releases). The folder stays the main way.
- On a hard disk (HDD): pack the game into a `.zar` there and keep the emulator folder (with its
  `_PipelineCache`, about 6 GB) on an SSD. Unpacked on a busy hard disk, every small read of the game
  waited 100-500 ms: long area loads and freezes while walking; packed, with the emulator on the SSD, it
  ran at full speed ([#91](https://github.com/chenxiao07/KytyPS5/issues/91)).
- Microsoft Defender scans each game file the first time it is opened in a session (about 117,000 files
  in 6 minutes of play), and while the game streams those scans take CPU from it: runs of slow frames. A
  `.zar` is one file, scanned once. Or exclude the game folder: Windows Security > Virus & threat
  protection > Manage settings > Add or remove exclusions > Add an exclusion > Folder (Defender then no
  longer scans that folder: only for a game copy you trust). Measured with the folder excluded: the 99th
  percentile frame time 31.5 -> 26 ms, the 1% lows 30 -> 36 fps
  ([#240](https://github.com/chenxiao07/KytyPS5/issues/240), @DevList1454).

### Add-on content (DLC)

- Add-ons with files of their own: one folder each in `Content\PPSA01341\` inside the emulator folder,
  holding the add-on's unpacked files (with its `sce_sys\param.json`). The game is told it owns them.
- Add-ons without files (a deluxe edition's extras, say): one section each in `dlc_emu.ini` in the
  emulator folder, with the add-on's content id:

  ```ini
  [PSAL]
  content_id=EP9000-PPSA01341_00-XXXXXXXXXXXXXXXX
  download_status=NO_EXTRA_DATA
  service_label=0
  ```

- The run log lists each add-on the game is told it owns (`NpEntitlementAccess: ... reported as owned`).

## Starting the game

**Double-click `KytyLauncher.exe`** (or `launcher.cmd`) for the settings window, in English or Chinese. Its
pages: **Play** (the game folder or `.zar`; checks of the game, graphics card, RAM, precompiled shaders and
DLSS/FSR; the Play button), **Display** (resolution, aspect ratio, fullscreen, HDR, up to 120 fps, VSync, low
latency), **Graphics** (upscaler, render height with each height's DLSS mode, game output, DLSS model,
sharpening, graphics preset, frame generation), **Game** (console
language, cheats), **Shaders** (precompile, optionally on the efficiency cores only so the PC stays responsive;
export shader hints), **Advanced** (performance overlay, red-zone protection, safe GPU mode, the graphics card on a
PC with more than one, extra command-line options, the launcher's language, the logs folder) and **Experimental** (options still being tried out, each with
the issue that collects how it goes: a 30 fps lock, direct presentation for VRR displays, keep memory resident,
AMD Anti-Lag). It remembers the settings (`launcher-settings.json`) and needs nothing installed (.NET
Framework 4.7.2, part of Windows 10 and 11). Windows may ask before its first start ("More info", "Run anyway"):
it is not signed. The earlier PowerShell window (`launcher.ps1`) still works.
`KytyLauncher.exe --play` starts the game right away with the launcher's settings, without its window: for a
desktop shortcut, or Steam (Add a Non-Steam Game, pick `KytyLauncher.exe`, then Launch Options `--play` in its
properties; Steam counts the game as running until it closes, so the controller layout applies). It shows no
dialogs (after a driver update, precompile from the Shaders page) and opens the window when no game folder is set.
"Up to 120 fps" (off by default) lets the game render up to 120 frames a second instead of the
console's 60 where the PC is fast enough: play keeps its speed, and movies, menus and loading screens
stay at 60.
"VSync" (off by default) waits for the display's refresh, so the picture does not tear; displays with
adaptive sync (G-SYNC, FreeSync) keep it. "HDR" (off by default) outputs HDR10 when "Use HDR" is on for the
display in Windows' display settings (the game is told the TV takes HDR); otherwise the picture stays SDR.
"Low latency" (off by default) keeps one finished frame waiting for the screen instead of about two, so what you
press shows sooner: at 60 Hz about 15-23 ms from the game's frame to the screen instead of about 30. A frame that
comes late is then no longer covered by a waiting one, so the frame pacing is a little less even.
"Render height" (1440 by default, as on the PS5) is the height the game renders its 3D at before it
upscales it to its 4K picture; lower heights (1296, 1152, 1080, 900) are faster where the graphics card
limits the frame rate (a laptop RTX 3080 at Boletaria's start: 20.7 fps at 1440, 24.8 at 1080) and
less sharp. The interface keeps the game output's size either way.
"Aspect ratio" (16:9 by default, as on the PS5) renders the 3D wider for ultrawide displays: 21:9
(3416x1440 at render height 1440) or 32:9, seeing more to the sides, not a stretched 16:9 picture. In
play the HUD keeps its shape: health, magic, stamina and the equipment at the left edge, the souls at
the right edge, the boss's health in the middle. The intro, the movies, the title screen, the loading screens
and the in-game menus are stretched to the width, unless "Movies and menus at 16:9 (black bars)" keeps them at
16:9 between black bars; the lock-on and enemy health bars stay on their targets. The
"Resolution" list follows the aspect ratio (2560x1080, 3440x1440, 3840x1600, 5120x2160 at 21:9), and
"Render height" offers the heights whose 3D the game's render target memory holds (it was made for its own
3840x2160): up to 1800 at 21:9 (4264x1800), 1512 at 32:9 (5376x1512). Larger ones drew the 3D red or broken, and
2160 at 21:9 ended the game at start-up; the emulator now lowers such a height by itself.
"Graphics" (Standard by default, as on the PS5) sets how much the game draws. Low picks the level of
detail as for a 720-line picture and ends sun shadows at 125 m (250 m); Lowest picks detail as for 480
lines, ends sun shadows at 60 m and turns off ambient shadowing, the shadows of torches and other lamps, and the
decals laid onto surfaces (blood, scorch marks). Each thing drawn costs the emulator processor time, so these
help where the processor limits the frame rate (a laptop Ryzen 9 5900HX standing at the Gates of
Boletaria: 21.9 fps at Standard, 25.6 at Lowest); the decals alone took Lowest on an RTX 5090 at Boletaria from 86
to 96 fps standing, the graphics card's work a frame 22% less.
"Upscaler" (Game by default, as on the PS5) with "DLSS (PS5 Pro path)" runs the game's own PS5 Pro mode:
its 3D at the render height (1080 at least, or 720 for DLSS's Ultra Performance) is made a real 3840x2160 image by NVIDIA DLSS in place of Sony's
PS5 Pro upscaler, and the game finishes it at 4K (bloom, depth of field, the HUD), instead of anti-aliasing
at the render height and stretching to 4K. The launcher keeps a render height for each upscaler: DLSS starts
at 1080, DLSS's Performance (on an RTX 5090 about the frame rate of the game's own TAA at 1440, with a
sharper picture); 1440 is its Quality, 1512 the PS5 Pro's own 2688x1512, 2160 DLAA. NVIDIA RTX only, with
NVIDIA's Streamline runtime as for frame generation below: on a PC without an NVIDIA RTX graphics card (AMD,
Intel, GeForce GTX) the launcher offers no DLSS. "AMD FSR (PS5 Pro path)" does the same with AMD's FSR 3 in place of
DLSS, on any graphics card, AMD and Intel included: it is built into the emulator, nothing to download
(`run.cmd -Upscaler fsr`; on an RTX 5090 at Boletaria's start 75.8 fps from 1080 to 4K, a picture close to DLSS's).
The render heights show their mode as with DLSS (FSR's "Native AA" at the output's own size). The in-game
"Cinematic" mode stays native 4K without DLSS or FSR.
The game's own TAA sharpens its picture; DLSS in its place does not, so fine detail (armour, cloth) looks softer
with it. "Sharpening" (Off, Low, Medium, High) sharpens the upscaler's picture with RCAS, the sharpening FSR 3 uses:
Off by default with DLSS (`-Set KYTY_DLSS_SHARPNESS=0.6` on a `run.cmd` line, 0 to 1), Medium with FSR 3, its own
(`-Set KYTY_FSR_SHARPNESS=0.3`; 0 none); each upscaler keeps its own. The higher levels can show halos on edges.
With DLSS or FSR 3 below their output's size, the game's textures are read with the detail of the output's size
rather than the render height's (a texture LOD bias of log2(render height / output height), as NVIDIA's DLSS guide
recommends; `-Set KYTY_UPSCALE_LOD_BIAS=0` turns it off).
With an "Aspect ratio" wider than 16:9, DLSS makes the picture at its full width: 5120x2160 at 21:9 (from
2560x1080 at render height 1080), and the game finishes it at that size, with the HUD at the screen's edges,
instead of a 3840x2160 picture widened on the screen. On an RTX 5090 at Boletaria's start that ran at 64.5 fps,
the game's TAA at 21:9 at 59.3. Screens without the 3D (movies, the title, loading) are stretched to the
width, as without DLSS. With "Game output: 1440p" that picture is made at 1440 lines instead (3416x1440 at 21:9, for
a 3440x1440 screen): the game's post-processing at the screen's size rather than at 5120x2160 scaled down.
"Game output" ("Automatic" by default: 1080p for a 16:9 game window, or screen in full screen, at most 1080 lines
high, else 4K as on the PS5) with "1080p" runs the game's own HD mode, the one a PS5 takes on
a 1080p TV: its post-processing and HUD at 1920x1080 instead of 3840x2160 scaled down to the window afterwards,
and its 3D at most 1080 high (a higher render height is lowered to 1080). Less work for the graphics card: at
Boletaria's start on an RTX 5090, 69.8 fps at 4K, 78.3 at 1080p. It suits 1080p screens; on larger ones the
picture is softer than the 4K output's. With "DLSS (PS5 Pro path)" too, the game's PS5 Pro mode makes a
1920x1080 output instead of its 4K one: DLSS turns the 3D at the render height (360 = Ultra Performance, 540 = Performance, 630 = Balanced, 720 and 900 = Quality, 1080 = DLAA)
into the 1080p picture, and the game finishes it at 1080p, HUD included. For 1080p screens and graphics cards
that find DLSS's 4K output too heavy (RTX 20/30): the game's own 1080p mode has no upscaler (its 3D, anti-aliasing
and effects at the render height, stretched at the end). On an RTX 5090 at Boletaria's start (its processor the
limit there): DLSS from 720 to 1080p 75.2 fps, DLSS from 1080 to 4K 68.8, the game's TAA at 1080p 76.8.
"1440p" (with DLSS or FSR 3 only; `-Upscaler dlss -OutputHeight 1440` on a `run.cmd` line, contributed by Tutuzaooo)
makes that output 2560x1440, for 1440p screens: the upscaler's own picture at the screen's size, neither the 4K one
scaled down nor the 1080p one scaled up, the HUD and menus in the game's 4K layout drawn at 1440 (as the 4K output's,
scaled down; `-Set KYTY_OUTPUT_LAYOUT=hd`: the game's HD layout scaled up, as before). Its render heights:
1440 = DLAA, 960 = Quality, 840 = Balanced, 720 = Performance, 480 = Ultra Performance. With the game's TAA the
output stays 4K.
"Frame generation" (DLSS, 2x to 6x; NVIDIA RTX 40 or newer, 3x and up RTX 50) uses NVIDIA's Streamline
runtime, which this package does not include: download the Streamline SDK (a release zip from
https://github.com/NVIDIA-RTX/Streamline/releases), unpack it and choose its folder with the launcher's
"Streamline..." button (only files signed by NVIDIA are loaded). The frames shown are the game's times the
multiplier; with VSync they are capped at the display's refresh rate (a 60 Hz display at 2x runs the game at 30 fps).
"AMD FSR frame generation" (2x, any graphics card, AMD included; `run.cmd -Fsr`) uses AMD's FSR runtime, which this
package does not include either: download AMD's FSR SDK 2.4 (the "Minimal package" of
https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.4.0) and put `amd_fidelityfx_loader_vk.dll`,
`amd_fidelityfx_framegeneration_vk.dll` and `amd_fidelityfx_upscaler_vk.dll` from its `Kits\FidelityFX\signedbin` next
to `kyty_emulator.exe` (only DLLs signed by AMD are loaded); the launcher's FSR checkbox is enabled then. It takes the
place of DLSS frame generation. With these DLLs a Radeon RX 9000 card gets FSR 4 frame generation, and with the
upscaler set to FSR, RX 7000 and 9000 cards (RX 6000 too) get FSR 4 upscaling; other cards get FSR 3.1. FSR 4 needs a
current AMD driver: with Adrenalin 26.9.2 an RX 9070 showed a black 3D image with FSR 4 upscaling and black generated
frames with FSR 4 frame generation, and a driver update fixed both (issue #271; AMD's SDK 2.4 names 26.10.1). The older
`amd_fidelityfx_vk.dll` of the FidelityFX SDK 1.1.4 (FSR 3.1 frame generation) still works as well.
"Overlay" (Off by default) shows the emulator's own panel at the top right of the game image: "Frame rate"
the frames per second and a graph of the last frames' times (60 and 30 fps lines, hitches in red, the 1% low
and the longest frame of the last 10 seconds); "Detailed" adds the render thread's time in each frame (orange),
the graphics card (load, clock, temperature, power and video memory on NVIDIA cards), the processor's load
with a bar for each core, and the memory in use. (The overlays seen in videos with temperatures and voltages
of every part come from separate tools such as MSI Afterburner's RivaTuner Statistics Server.) In the
game, F8 switches the panel between Off, "Frame rate" and "Detailed", whatever the launcher started it with
(also when started with `run.cmd`).
"Safe GPU mode" (off by default) prevents the crashes and blue screens that come from the graphics card
hanging ("device lost", the screen going black while the graphics driver restarts, a VIDEO_TDR_FAILURE blue
screen), but **the frame rate is about 37% lower**. The emulator normally lets the graphics card overlap
Demon's Souls' compute work; in this mode it runs the steps one after another. On a laptop RTX 3080 (driver
566.07) that had these hangs, 3 of 4 runs through Boletaria hung without it and none of 3 with it, at 17 fps
instead of 27. It does not help the MEMORY_MANAGEMENT blue screens of a full RAM (see RAM under System
requirements). Leave it off when the game runs without such crashes; a current graphics driver can help as well.
(With `run.cmd` or `run-windows.ps1`: `-Set KYTY_COMPUTE_CHAIN=0`.)
Its settings and language list follow the KytyPS5 launcher (`src/launcher` in the source, by the
KytyPS5 developers).

Or **double-click `run.cmd`** to start with the default settings (do not double-click
`kyty_emulator.exe` directly: it would lack its launch options).

The first launch goes like this:

1. Choose the game folder (see above).
2. The console lists the game's shaders from its files (once; nothing to install for it).
3. A dialog shows whether the game version is a supported one and offers to precompile the shaders
   first (see the next section).
4. The console window makes the input files for the background shader preparation (once per
   graphics card and driver version): about 50 seconds on the test PC's 22 threads, a few minutes on
   PCs with fewer threads.
5. The game window appears. The start-up screen shows the loading progress; in the game, the top
   right corner shows "Preparing shaders xx%".

Later launches go straight into the game. The console window stays open and shows the game's log as
it runs; when the game crashes it shows the exit code and the end of the error log and waits for a key
(the log files are in the `logs` folder). To quit, close the game window.

Common options (add them after `run.cmd` on a command line, or put them into the "Target" of a
shortcut):

| Option | Effect |
| --- | --- |
| `-Fullscreen` | Full screen (F11 or Alt+Enter also toggle it in the game) |
| `-Fullscreen -AspectFit` | Full screen keeping 16:9, with black bars instead of stretching |
| `-Width 1920 -Height 1080` | The window size (default 2560×1440, shrunk when the screen is smaller) |
| `-Game <folder>` | The game folder (or a `.zar` archive of it) |
| `-Language 11` | The console language, as the PS5 numbers them (0 Japanese, 1 English (US), 11 Chinese (Simplified), ...; the launcher lists them) |
| `-Set KYTY_ALL_LANGUAGES=0` | Demon's Souls' language settings offer every language the package has, in every region: the text of 23 (a US game Russian, Japanese or Korean too) and the voices of 9 (English, French, German, Italian, Spanish, Japanese, Brazilian Portuguese, Latin American Spanish; British English speaks English); this keeps only the console region's |
| `-RenderHeight 1080` | The height the game renders its 3D at (360–2160; below 720 is meant for the 1080p output, `-HdOutput`, e.g. 540 = DLSS Performance there): lower is faster on a slow graphics card and less sharp. Without it the height follows the window, at most 1440 as on the PS5 (1080 on a 1080p screen) |
| `-HdOutput` | The game's own 1080p output (the launcher's "Game output: 1080p"): post-processing and HUD at 1920×1080, the 3D at most 1080 high; with `-Upscaler dlss`, DLSS makes the 1080p picture from it. Without it (and without `-RenderHeight`, `-Upscaler` or `-Aspect`) the game takes it by itself for a window or screen at most 1080 lines high; `-Set KYTY_HD_OUTPUT=0` keeps the 4K output |
| `-OutputHeight 1440` | With `-Upscaler dlss` or `-Upscaler fsr`: the PS5 Pro path's output at that height (1080–2159, 16:9; 2560×1440 for 1440, the launcher's "Game output: 1440p"), the HUD in the game's 4K layout at that size; `-RenderHeight` picks the mode (1440 DLAA, 960 Quality, 840 Balanced, 720 Performance, 480 Ultra Performance) |
| `-Aspect 21:9` | The 3D rendered wider (`21:9` or `32:9`) for ultrawide displays; the HUD keeps its shape at the sides; in-game menus, movies and loading screens are stretched to the width (with `-AspectFit` too: movies, the title and loading screens keep their 16:9, with black bars at the sides) |
| `-Quality low` | How much the game draws: `low` or `lowest` (shorter shadows, simpler distant detail; `lowest` also without ambient and torch shadows and decals), faster where the processor or the graphics card limits the frame rate |
| `-TextureReduce 2` | Large textures without their largest levels: `1` half the size per side (the default with less than 12 GB of video memory), `2` a quarter, `0` whole |
| `-Fps120` | The game's frame rate up to 120 fps instead of 60 (the launcher's "Up to 120 fps") |
| `-Set KYTY_FLIP_RATE=1` | The game locked to a steady 30 fps, at the same game speed: half the work, even frame times; movies and menus stay at 60 (the launcher's Experimental "Lock to 30 fps") |
| `-PresentMode Fifo` | VSync on (the launcher's "VSync") |
| `-LowLatency` | Less input lag: one frame waiting for the screen instead of about two (the launcher's "Low latency") |
| `-Hdr` | HDR10 output when Windows' "Use HDR" is on for the display |
| `-Upscaler dlss` | DLSS super resolution through the game's PS5 Pro path (NVIDIA RTX, needs Streamline: see "Upscaler" above) |
| `-Upscaler fsr` | The same with AMD FSR 3, built in: any graphics card |
| `-Set KYTY_DLSS_SHARPNESS=0.6` | With `-Upscaler dlss`: DLSS's picture sharpened, 0 to 1 (the launcher's "Sharpening": Low 0.3, Medium 0.6, High 0.9; none by default) |
| `-Set KYTY_FSR_SHARPNESS=0.3` | With `-Upscaler fsr`: FSR 3's own sharpening, 0 (none) to 1 (the launcher's "Sharpening"; 0.6, Medium, by default) |
| `-FrameGen 1` | DLSS frame generation, the number of generated frames per rendered one (`1` = 2x, up to `5`; RTX 40 or newer, needs Streamline) |
| `-Gpu 1` | The graphics card on a PC with more than one: its number in the log's `Vulkan GPU <n>:` lines, or a part of its name (`-Gpu 4090`); give `precompile.cmd` the same `-Gpu`, so the shaders are compiled for that card. Without it the first discrete GPU, in the order Windows' graphics settings give |
| `-Set KEY=VALUE` | An emulator switch, e.g. `-Set KYTY_SHADER_PREFETCH=0` (several: `-Set A=1,B=2`); the issues name the ones worth trying |

The launcher has a box for these too (Advanced, "Command-line options"), written as after `run.cmd`: its `-Set`
pairs join the launcher's own (one of the same name replaces the launcher's), and another option replaces the
launcher's of the same name, e.g. `-RenderHeight 900` or `-Set KYTY_DLSS_PRESET=E`. If the game no longer starts,
clear the box.

## Game language (Chinese, Japanese, Korean and others)

The emulator makes the game's language settings offer every language in every release: the text of 23
(Chinese, Japanese and Korean too) and the voices of 9 (Settings > Language in the game; the voice
language from the title screen's settings). `-Set KYTY_ALL_LANGUAGES=0` turns that off. The edit below
is only needed with it off, or for a game version where the emulator does not find the language sets
(the log says so: `Demon's Souls languages: ...`).

The game files hold every language of the game, but the game offers only those of its release. It
picks the release from `userDefinedParam1` in the game folder's `sce_sys\param.json`, an index into a
table inside the game. This dump (the European release) has `2`, whose list has no Chinese, Japanese
or Korean. The emulator stores saves by the title ID in `param.json`, which does not change, so the
saves stay where they are.

| `userDefinedParam1` | Release | The game's language list |
| --- | --- | --- |
| `0` | Global | Every language: Arabic, Chinese (Traditional and Simplified), Danish, Dutch, English, English (UK), Finnish, French, German, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Portuguese (Brazil), Russian, Spanish, Spanish (Latin America), Swedish, Turkish, Thai |
| `1` | America | English, French, Portuguese (Brazil), Spanish (Latin America) |
| `2` | Europe (as dumped) | Arabic, Danish, Dutch, English (UK), Finnish, French, German, Italian, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Turkish |
| `3` | Japan | English (UK), Japanese |
| `4` | Asia | Chinese (Traditional and Simplified), English (UK), Korean, Thai |

To switch:

1. Close the game. Copy `sce_sys\param.json` somewhere as a backup, open it with Notepad, change
   `"userDefinedParam1": 2,` to `0` (every language) or to `4` (the Asian release), and save it.
2. In the launcher, set "Console language" to the language you want (`run.cmd -Language 11` for
   Chinese (Simplified), `10` Traditional, `0` Japanese, `9` Korean): when the release has it, the game's
   list starts on it.
3. The game asks for its language only while there is no options save:
   - No saves yet (a new installation): it asks at the first start.
   - With saves: back up the `_SaveData` folder, then move the folder
     `_SaveData\PPSA01341\SAVEDATA0OptionsProfile0` out of `_SaveData` (to the desktop, say). The characters
     are in the `SAVEDATA0PlayerProfile...` folders and stay as they are; only the game's options
     (brightness, controls, camera, subtitles) go back to their defaults.
4. Start the game: "LANGUAGE SELECT" lists the languages of that release. Choose one and confirm with ×
   (J on the keyboard); the brightness and the speech and subtitle settings follow in that language.

中文说明：游戏文件里其实包含全部语言，只是游戏按 `sce_sys\param.json` 里的 `userDefinedParam1`（发行区域）决定
语言选择列表。本 dump 是欧版（值为 `2`，列表里没有中文、日文、韩文）。用记事本把它改成 `0`（全部语言，含简繁中文和日文）
或 `4`（亚洲版：简繁中文、英文、韩文、泰文）并保存（先备份原文件）；启动器的 "Console language" 选中文（简体）。
没有存档时首次启动就会出现语言选择；已有存档时，先备份 `_SaveData`，再把 `_SaveData\PPSA01341\SAVEDATA0OptionsProfile0`
这个文件夹移出 `_SaveData`（角色存档在 `SAVEDATA0PlayerProfile...` 里，不受影响，只有亮度、按键等游戏设置会恢复默认），
下次启动时在 "LANGUAGE SELECT" 里选择简体中文即可。存档按 `param.json` 里的 Title ID 保存，改这个值不影响存档位置。

## Shader precompile (without it the game stutters)

The emulator translates the game's shaders and compiles them into a form the graphics card can run.
When a scene, effect or enemy appears for the first time and its shaders are not ready yet, the game
**stutters for a fraction of a second to a few seconds**, most noticeably when entering a new area or
seeing a new effect for the first time. What you have already played is remembered and does not
stutter again.

There are two levels of preparation, both specific to the graphics card and driver:

1. **Background shader preparation (automatic)**: the first launch translates the whole game's shaders
   in the background while the top right corner shows the progress. It takes less than half a minute
   on the test PC and a few minutes on PCs with fewer threads, with a high CPU load meanwhile. The
   translations are kept in `_PipelineCache\static\<title>_<version>.prefetch` (1-2 GB), and later
   launches read them from there in about a second. After an emulator update or a new graphics driver
   they are translated once more. The first launch also makes the input files it needs.
2. **Full precompile (recommended once)**: compiles every shader and pipeline of the game into
   `_PipelineCache`. Afterwards shader stutters are essentially gone. With a recent NVIDIA driver the
   result is a store the game reads pipeline by pipeline (`<title>_<version>.binaries`, about 2 GB, nothing to
   load at launch); with other drivers a cache each launch loads (with a progress bar).
   - Choose "Precompile first" in the dialog at launch, or double-click `precompile.cmd` on its own.
   - The time depends on the CPU: under an hour with 22 threads, about 1 hour 50 minutes on a 16-thread
     laptop (Ryzen 9 5900HX), longer with fewer threads. The CPU is fully loaded meanwhile: running it
     overnight is a good idea. Once "ready to play" shows (about halfway), what play needs is in place
     and the game can be started; the rest follows.
   - Closing the window stops it; running it again continues where it stopped.
   - If the graphics driver never finishes compiling a pipeline (reported on an AMD card), the precompile leaves
     that pipeline out after 5 minutes and carries on; at the end it lists them in the logs folder
     (`stuck-pipelines.txt`). Please update the graphics driver, and if it happens again, report that file on GitHub.
   - Coming from an older package with a precompiled `_PipelineCache\static\<title>.bin`: the precompile
     turns it into the store in a few minutes; the `.bin` is then unused and can be deleted (about 4 GB).
   - It compiles the shaders as the game uses them in play too: the package carries what was recorded
     while playing every world on the test PC (`tools\local\static-precompile\hints-<title>_<version>.hints`,
     without the game's shader code), together with what your own play recorded, completed with the code of
     your own game files. For those shaders it leaves out the guessed variants the game did not use (a fifth
     of the time); `precompile.cmd -Full` compiles them too.
   - Places and effects the test PC never recorded (bosses' deaths, for one) still compile when they first
     appear. The launcher's "Export shader hints" button (or `precompile.cmd -ExportHints`) writes what your
     own play recorded, without the game's shader code, to `my-hints-<title>_<version>.hints.zip` (a few hundred
     KB, nothing is compiled) and shows it in Explorer: attached to a GitHub issue, it goes into the hints of the
     next packages.
   - **After a graphics driver update or a new graphics card it has to be done again**; the launcher
     asks again. It also asks after an update of the emulator that precompiles more: running it again
     then compiles only what is new, unless that version translates the shaders differently (the launcher
     then says the precompile compiles all of them once; until then the old ones stay in use).
   - To stop being asked: tick "Don't ask about precompiling again" (delete
     `no-precompile-prompt.txt` to be asked again).

## Controls

The Windows launcher also offers **Game → Keyboard and mouse prompts**, off by default.
For an extracted PPSA01341 v1.07 game folder, it applies 22 English key/mouse icons and matching
keyboard/mouse bindings on Play. **Game → Keyboard controls → Edit controls…** lets you change
keys, mouse buttons, wheel directions and sensitivity; the prompts follow the saved layout.
Press **F7** to capture/release the mouse for camera control. Disabling the option restores
backed-up original icons on the next Play and stops applying these bindings; `launch.json`
is not rewritten. The keyboard labels remain visible when using a controller.
Other game versions and ZArchive files are not supported by this option. See the
[prompt pack and label mapping](https://github.com/chenxiao07/KytyPS5/tree/experiment/perf-40fps-20260926/tools/keyboard-prompts)
for installation, restoration and customization details.

A gamepad is recommended (DualSense, DualShock 4 and Xbox controllers work; just plug them in). An Xbox pad's View
button is the touchpad. Holding the PS button (an Xbox pad's Guide button) for two seconds quits the game, as closing
the window does (in Steam Big Picture, Steam keeps that button for its overlay unless set to pass it to the game).
Where Steam or Windows' Game Bar takes that button, `-Set KYTY_PAD_QUIT=both` makes holding Create and Options together
(an Xbox pad's View and Menu) for two seconds quit too. It is off by default because some pads switch their own mode on
that hold. `-Set KYTY_PAD_QUIT=off` turns off quitting from the pad.

Default keyboard layout:

| Key | Maps to |
| --- | --- |
| W A S D | Left stick (move) |
| T F G H | Right stick (camera: T up, G down, F left, H right) |
| J / L / K / I | × / ○ / □ / △ |
| Q / E | L1 / R1 |
| Left Shift / Left Ctrl | L3 / R3 |
| Arrow keys | D-pad |
| Enter | OPTIONS |
| Backspace / Tab | Left / right half of the touch pad |
| F7 | Camera with the mouse (press again to release the mouse) |
| Space, Pause | **Pause / resume the emulation** (careful not to hit Space by accident; once your own keys give Space to the game, only Pause) |
| F8 | The emulator's overlay: Off, Frame rate, Detailed (the launcher's "Overlay" is where it starts) |
| F11, Alt+Enter | Toggle full screen |

The keyboard has no L2 / R2 by default: use a gamepad when you need them.

### Your own keys

Add `"--keymap", "<control>=<key>"` pairs to the `"command"` list in `launch.json` (after `"kyty_emulator.exe"`).
With any of them, only your own keys work on the keyboard (none of the layout above):

```json
"--keymap", "LeftStickUp=W",
"--keymap", "Circle=Space",
"--keymap", "L1=Left Shift",
"--keymap", "R1=Mouse:Left",
"--keymap", "MouseSensitivity=1.5",
```

- Controls: `Cross Circle Square Triangle L1 R1 L2 R2 L3 R3 Options Up Down Left Right TouchPad TouchPadRight`,
  `LeftStickUp/Down/Left/Right`, `RightStickUp/Down/Left/Right`.
- Keys: their names on the keyboard (`E`, `Tab`, `Space`, `Escape`, `Left Shift`, `Left Ctrl`, `Return`, ...; also
  written without the space, as `LeftShift`), or `Mouse:Left`, `Mouse:Right`, `Mouse:Middle`, `Mouse:X1`, `Mouse:X2`,
  `Mouse:WheelUp`, `Mouse:WheelDown` (each notch of the wheel a short press: `"--keymap", "Up=Mouse:WheelUp"` and
  `"--keymap", "Down=Mouse:WheelDown"` switch spells and items).
- A control takes one key and one mouse input (`Up=Up` with `Up=Mouse:WheelUp`: both work); a second key, or a second
  mouse input, for the same control replaces the first.
- F7, F8 and F11 stay the emulator's. A line it cannot use is skipped, and the run log says which and why
  (`Input mapping skipped: ...`).
- An update's `launch.json` replaces yours: keep a copy of your lines.

## Files and folders

| File / folder | Contents |
| --- | --- |
| `run.cmd` | Starts the game |
| `precompile.cmd` | Full shader precompile |
| `_SaveData` | **Saves** (created by the first run; back them up now and then). Each save the game makes is also kept in `_SaveData\.kyty-backup\<game>\<slot>\<date-time>` (the last 4, at least 5 minutes apart); a save left broken by a crash while saving (a file of zeros) gets the newest good one back when the game opens it, the broken files kept beside the backups, and the log says so |
| `_PipelineCache` | Shader caches (valid only for this PC's graphics card and driver: do not copy them to other PCs) |
| `logs` | Run logs |
| `game-path.txt` | The remembered game location |
| `seeds-<title>_<version>.seeds` | The list of shaders the first launch collects from the game files (one per game version) (it contains the game's shader code: do not share it), for the precompile |
| `hinted-<title>_<version>.seeds` | The recorded play of the package's hints and of your own play with the code of your game files (made by each precompile, from `released-<title>_<version>.seeds` and `recorded-play-<title>_<version>.hints`; the `.seeds` files contain the game's shader code: do not share them) |
| `launch.json` | The emulator's command line and extra switches; normally left alone |
| `launcher-settings.json` | The launcher's settings |
| `Content\<title>`, `dlc_emu.ini` | Add-on content (DLC), if you add it (see "Add-on content") |

## Updating to a newer package

Close the game and the launcher, then extract the new zip over the old folder and let it replace the
files. The zip holds only the program files (the emulator, its DLLs, the scripts, `launch.json`, this
README, the licenses and the shader hints under `tools`). Your saves, shader caches, logs and settings
are in files it does not have (`_SaveData`, `_PipelineCache`, `logs`, `game-path.txt`,
`launcher-settings.json`, the `seeds-...` files), so they stay as they are. The launcher asks for a new
precompile only when the new version needs one, and that run compiles only what is new (or, when it says so,
all of them once, for a newer translation). If you edited
`launch.json` yourself, make the edit again in the new one. A copy of `_SaveData` before an update never
hurts.

## Known issues

- It crashes now and then.
- On some PCs the graphics card hangs ("device lost", the screen goes black and the driver restarts, or a
  VIDEO_TDR_FAILURE blue screen), in a laptop RTX 3080's case within minutes at Boletaria: turn on "Safe GPU
  mode" in the launcher (see above). The frame rate is then about 37% lower.
- Sound: 3D sound has no direction, and there is no reverb. On a busy PC it may crackle: the sound buffer
  (20 ms) grows by itself after each dropout; to start with a bigger one: `run.cmd -Set KYTY_AUDIO_BUFFER_MS=60`
  (up to 200). Where the CPU is at its limit and the game makes its sound too late, the sound is stretched a
  little to fill the gaps instead of cutting out (`-Set KYTY_AUDIO_STRETCH=0` turns that off).
- It uses a lot of memory and video memory: on PCs with too little it may just exit.
- NVIDIA's Low Latency Mode **Ultra** (NVIDIA Control Panel) makes the emulator's presents wait: uneven frame times
  (RTSS and the emulator's panel disagree) and fewer frames (issue #129). Set Low Latency Mode to Off or On for
  `kyty_emulator.exe` (Manage 3D settings > Program Settings).
- The MEMORY_MANAGEMENT blue screen (0x1A, 0x41202) that PCs whose RAM ran full got after 10 minutes to an hour
  of play (issue #50, mostly 16 GB with a browser open) is fixed: Windows mishandles no-access pages of a mapped
  file when it pages them out, and the emulator now keeps such pages from reads with guard pages instead
  (`run.cmd -Set KYTY_READ_GUARD=0` goes back to no access, for comparison). If you still get this blue screen,
  please report it in issue #50. A full RAM still makes the game slow: close other large programs while playing
  and keep a page file of at least 32 GB.
- On some TVs the picture tears or stutters with VSync on. A user's fix on a 60 Hz TV with an NVIDIA
  card (issue #30): in the NVIDIA Control Panel, Manage 3D settings > Program Settings, add
  `kyty_emulator.exe` and set "Vertical sync" to On and "Vulkan/OpenGL present method" to "Prefer native".
- With DLSS frame generation on, the game may stop at start-up with "Could not create the Vulkan device with DLSS
  frame generation". A user's fix on an RTX 5080 (issue #45): in Windows Security, App & browser control > Exploit
  protection > Program settings, add `kyty_emulator.exe` and, under "Randomize memory allocations (Bottom-up ASLR)",
  tick "Override system settings" and turn it On.
- Windows may say "Windows protected your PC" (the programs are not signed): click
  "More info → Run anyway". For a zip downloaded from the internet, you can tick "Unblock" in the
  zip file's properties before extracting it.

## When something goes wrong

Send the newest `.out.log` and `.err.log` from the `logs` folder to whoever gave you this package,
and describe where you were and what you were doing when it happened.
