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

## System requirements

| Item | Requirement | Notes |
| --- | --- | --- |
| OS | Windows 10 (1803 or later) or Windows 11, 64-bit | |
| CPU | AVX2 required (Intel Core 4th generation / AMD Ryzen or later) | 8 cores or more recommended; the test PC has an i9-14900K |
| RAM | **32 GB recommended, 16 GB works** | The game process takes about 20 GB while a save loads but uses only about 4 GB of it in play, so with 16 GB Windows moves the rest to the page file and loading takes longer. It needs about 34 GB of RAM plus page file: keep the page file "system managed" (the launcher warns when Windows cannot provide it) |
| GPU | NVIDIA RTX with **12 GB of VRAM recommended, 8 GB works** | About 11 GB of VRAM in use in the heaviest areas. With less than 12 GB, render targets and buffers keep the video memory and the textures that do not fit are read from system memory. The test PC with its video memory limited to 8 GB: Boletaria's gate 45–52 fps, most areas 40–60, the heaviest (Shrine of Storms, Stonefang 2-1, Boletaria 1-2 and 1-3) 10–35, changing from run to run with the textures the driver keeps in video memory; a slower GPU or PCIe link gives less. The test PC has an RTX 5090; AMD/Intel GPUs are untested |
| GPU driver | The latest | The emulator uses Vulkan, which comes with the driver |
| Disk | About 83 GB for the game, an SSD recommended | The emulator takes about 270 MB, the shader caches up to about 2.5 GB, and while playing the system temp folder needs another 4–5 GB |

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
gain grows as you play. To start without it: `run.cmd -Set KYTY_TABLE_XPR=,KYTY_TABLE_DISPATCH=`.

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

## Starting the game

**Double-click `launcher.cmd`** for a small settings window: the game folder, the resolution,
fullscreen (optionally keeping 16:9 with black bars), the console language and red-zone protection,
with buttons to play, to precompile the shaders (optionally on the efficiency cores only, so the PC
stays responsive), and to open the logs folder. It remembers the settings (`launcher-settings.json`).
"Up to 120 fps" (off by default) lets the game render up to 120 frames a second instead of the
console's 60 where the PC is fast enough: play keeps its speed, but pre-rendered movies play faster.
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

## Shader precompile (without it the game stutters)

The emulator translates the game's shaders and compiles them into a form the graphics card can run.
When a scene, effect or enemy appears for the first time and its shaders are not ready yet, the game
**stutters for a fraction of a second to a few seconds**, most noticeably when entering a new area or
seeing a new effect for the first time. What you have already played is remembered and does not
stutter again.

There are two levels of preparation, both specific to the graphics card and driver:

1. **Background shader preparation (automatic)**: after every launch, the whole game's shaders are
   translated in the background while the top right corner shows the progress. It takes less than
   half a minute on the test PC and a few minutes on PCs with fewer threads, with a high CPU load
   meanwhile. The first launch makes the input files it needs.
2. **Full precompile (recommended once)**: compiles every shader and pipeline of the game into
   `_PipelineCache`. Afterwards shader stutters are essentially gone. With a recent NVIDIA driver the
   result is a store the game reads pipeline by pipeline (`<title>_<version>.binaries`, about 2 GB, nothing to
   load at launch); with other drivers a cache each launch loads (with a progress bar).
   - Choose "Precompile first" in the dialog at launch, or double-click `precompile.cmd` on its own.
   - The time depends on the CPU threads: about 55 minutes with 22 threads, about 1 hour 15 minutes with
     16, about 2.5 hours with 8. The CPU is fully loaded meanwhile: running it overnight is a good idea.
   - Closing the window stops it; running it again continues where it stopped.
   - Coming from an older package with a precompiled `_PipelineCache\static\<title>.bin`: the precompile
     turns it into the store in a few minutes; the `.bin` is then unused and can be deleted (about 4 GB).
   - It compiles the shaders as the game uses them in play too: the package carries what was recorded
     while playing every world on the test PC (`tools\local\static-precompile\hints-<title>_<version>.hints`,
     without the game's shader code), completed with the code of your own game files.
   - **After a graphics driver update or a new graphics card it has to be done again**; the launcher
     asks again. It also asks after an update of the emulator that precompiles more: running it again
     then compiles only what is new.
   - To stop being asked: tick "Don't ask about precompiling again" (delete
     `no-precompile-prompt.txt` to be asked again).

## Controls

A gamepad is recommended (DualSense, DualShock 4 and Xbox controllers work; just plug them in).

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
| Space | **Pause / resume the emulation** (careful not to hit it by accident) |
| F11, Alt+Enter | Toggle full screen |
| F9 / F10, F8 | Debug warp: pick a map's spawn point, arm (or cancel) the warp |

The keyboard has no L2 / R2 by default: use a gamepad when you need them.

**Debug warp** (for testing other areas): F9 / F10 show the list of spawn points of every map (read
from the game's map files) and move in it, F8 arms the shown one. Then leave to the title (OPTIONS >
Settings, the gear tab > Exit Game > Save and Exit Game) and Continue: the game loads on that map.
The save on disk is only changed by the game itself when it saves on the new map (the panel then says
"Warp done"), so a character can be warped back the same way; it keeps everything else (level,
items, world progress). Some spawn points are at a boss's fog gate. A character still in the tutorial
can be warped too; one in the character creation cannot (it has no save yet).

## Files and folders

| File / folder | Contents |
| --- | --- |
| `run.cmd` | Starts the game |
| `precompile.cmd` | Full shader precompile |
| `_SaveData` | **Saves** (created by the first run; back them up now and then) |
| `_PipelineCache` | Shader caches (valid only for this PC's graphics card and driver: do not copy them to other PCs) |
| `logs` | Run logs |
| `game-path.txt` | The remembered game location |
| `seeds-<title>_<version>.seeds` | The list of shaders the first launch collects from the game files (one per game version) (it contains the game's shader code: do not share it), for the precompile |
| `hinted-<title>_<version>.seeds` | The recorded play of the package's hints with the code of your game files (made by each precompile; it contains the game's shader code: do not share it) |
| `launch.json` | The emulator's switches; normally left alone |

## Known issues

- It crashes now and then.
- Sound: 3D sound has no direction, and there is no reverb.
- It uses a lot of memory and video memory: on PCs with too little it may just exit.
- Windows may say "Windows protected your PC" (the programs are not signed): click
  "More info → Run anyway". For a zip downloaded from the internet, you can tick "Unblock" in the
  zip file's properties before extracting it.

## When something goes wrong

Send the newest `.out.log` and `.err.log` from the `logs` folder to whoever gave you this package,
and describe where you were and what you were doing when it happened.
