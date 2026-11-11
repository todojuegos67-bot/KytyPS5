# Demon's Souls on the Steam Deck

KytyPS5 runs Demon's Souls (PPSA01341: 01.007.000, the version tested on the Deck, or 01.005.000) on a Steam Deck at
about 26-30 frames a second in the worlds and 40 in the Nexus, with the settings below. This page tells how to get the
emulator and the game onto the Deck, how to start it from Steam, and what to expect. (中文版:
[STEAM-DECK.zh-CN.md](STEAM-DECK.zh-CN.md).)

## What it runs at

Measured on a Steam Deck LCD (SteamOS 3.9) with the Steam Deck build (below) at `run-deck.sh`'s settings and the GPU
clock at 1300 MHz: 15 seconds at each world's first archstone, 40 seconds after it loaded, after travelling there from
the previous one.

| Where | Frames a second |
|---|---|
| Boletaria (1-1), at the start of the level | 37-38 |
| Boletaria (1-1), arriving from the Nexus | 25-26 |
| Stonefang Tunnel (2-1) | 29-30 |
| Tower of Latria (3-1) | 26 |
| Shrine of Storms (4-1) | 29 |
| Valley of Defilement (5-1) | 28-29 |
| Nexus | 40-41 |

The frame rate is set by the Deck's 15 W shared between its CPU and its GPU: the emulator needs both. At 1300 MHz the
GPU leaves the CPU enough power; at 1600 MHz (the most) the whole frame was slower, and with the clock left on auto
(Steam's default) a later tour ran 1-3 frames a second slower in the worlds (the Nexus the same). The first visit to a
place compiles new shaders, which shows as short stutters (up to ~1.5 s); they are kept in `_PipelineCache`, so the
next visit is smooth.

The settings `run-deck.sh` uses: the launcher's **Lowest** graphics preset (shorter sun shadows, no ambient or torch
shadows, simpler distant detail, textures a quarter the size), the 3D drawn at 640x360, no projected decals (blood,
scorch marks), sun shadows within 20 m, the game's 1080p output shown full screen with its 16:9 kept (bars at the top
and bottom).

Memory: the Deck's 16 GB are shared with its GPU. After visiting every world, 1.2 GB of RAM was left and 2.7 GB of the
emulator's memory was in the swap file: close other programs, and keep SteamOS's swap file.

## 1. The emulator

In **Desktop Mode** (hold the power button > Switch to Desktop):

1. Open the browser and download `KytyPS5-steamdeck-<version>.zip` from the
   [releases page](https://github.com/chenxiao07/KytyPS5/releases): the Steam Deck package. (The Linux package,
   `KytyPS5-linux-<version>.zip`, works too, with the same `run-deck.sh`; not the Windows one.)
2. In the file manager (Dolphin), right-click it > Extract > Extract archive here, and rename the folder to `KytyPS5`
   (any name works; this page uses `~/KytyPS5`).
3. Open Konsole and make the programs executable (the unpacker may drop that):

   ```
   chmod +x ~/KytyPS5/kyty_emulator ~/KytyPS5/run-deck.sh
   ```

The package's `run-deck.sh` sets everything below; nothing else needs installing.

**Updating** to a newer version: extract its Steam Deck package the same way, copy everything inside the new folder
into `~/KytyPS5` and let it replace the files (Dolphin: Overwrite All; in Konsole `cp -r <new folder>/. ~/KytyPS5/`),
then run the `chmod +x` line above again. Your saves (`_SaveData`), the shader cache (`_PipelineCache`) and cheat files
are not in the package, so they stay, and the Steam shortcut keeps its launch options. The first play after an update
may compile some shaders again.

## 2. The game

The game is the folder with `eboot.bin` (`PPSA01341-app0`, about 82 GB) or the same packed as a ZArchive
(`PPSA01341-app0.zar`, about 54 GB). Put it in `~/Games/` on the Deck as `~/Games/PPSA01341-app0.zar` or the folder
`~/Games/PPSA01341-app0`: `run-deck.sh` looks there. (Elsewhere: set `KYTY_GAME`, below.)

Ways to copy it from a PC:

- **A USB drive** formatted **exFAT** on the PC (54-82 GB of room): copy the game onto it, plug it into the Deck (a USB-C
  adapter or dock) in Desktop Mode, and copy it into `~/Games` with Dolphin. A microSD card SteamOS formatted is ext4,
  which Windows cannot write; a card formatted exFAT on the PC works like the USB drive.
- **Over the network** from Windows: on the Deck, in Konsole, give the `deck` user a password and start SSH:

  ```
  passwd
  sudo systemctl start sshd
  mkdir -p ~/Games
  ```

  Find the Deck's address (Settings > Internet, or `ip -4 addr` in Konsole), then on the PC in PowerShell:

  ```
  scp -r "D:\Games\PPSA01341-app0" deck@192.168.0.50:/home/deck/Games/
  ```

  (Your folder and the Deck's address in place of these.) On 5 GHz Wi-Fi this ran at 30-40 MB/s (the folder in about
  40 minutes); on 2.4 GHz Wi-Fi at 2-7 MB/s. From Linux or WSL, `rsync -av --partial` can resume a copy that broke off.
  `sudo systemctl stop sshd` turns SSH off again.

**Saves**: to go on with a game saved on the PC, copy the PC's `_SaveData` folder (next to the emulator) into
`~/KytyPS5/` on the Deck.

## 3. Add it to Steam

Still in Desktop Mode: open Steam > Games > **Add a Non-Steam Game to My Library** > Browse, show all files, pick
`~/KytyPS5/run-deck.sh`, and add it. In its Properties, give it a name (Demon's Souls).

Back in **Game Mode**, start it from the library (Non-Steam). Then set its performance profile once: the Quick Access
menu (the `...` button) > Performance > **Use per-game profile**, then (Advanced View) Manual GPU Clock Control **on**,
**1300 MHz**. Game Mode keeps it for this game.

Start the game from Game Mode: there the Deck's controls reach the game as a controller only. Started outside Steam
(Desktop Mode), Steam's desktop layout also sends keyboard keys for the Deck's sticks and buttons: the arrow keys change
equipment, and the Space it sends pauses the emulator (Space again resumes; the Steam Deck build ignores those keys).

## 4. Options

`run-deck.sh` takes switches in front of `%command%` in the shortcut's Properties > Launch Options, for example
`KYTY_DECK_SUN_SHADOWS=full %command%`:

| Switch | What it does |
|---|---|
| `KYTY_DECK_SUN_SHADOWS=full` | Sun shadows to 60 m, the Lowest preset's (slower) |
| `KYTY_DECK_SUN_SHADOWS=off` | No sun shadows (fastest, but the sun lights the character in dark places) |
| `KYTY_DECK_DECALS=1` | Projected decals (blood, scorch marks) back on (~7% slower) |
| `KYTY_RENDER_HEIGHT=432` | A sharper 3D (768x432; ~5% slower than 360) |
| `KYTY_FPS_HUD=0` | No frame-rate panel at the top right (F8 switches it) |
| `KYTY_GAME=/path/to/game` | The game somewhere else than `~/Games` |
| `KYTY_CONSOLE_LANGUAGE=0` | The console's language (1 English, the default; 0 Japanese) |

**Cheats.** The emulator's own options go after `%command%`. A GoldHEN-style cheat file for the game's version (the
package has none; see the Windows README, "Cheats") saved as `~/KytyPS5/_Patches/PPSA01341.json` applies with
`%command% --game-patch /home/deck/KytyPS5/_Patches/PPSA01341.json`; in the file, a cheat with `"enabled": false` is
left out. A wrapper, such as a frame-generation tool's, goes in front of `%command%` and the switches in front of
it: `KYTY_FPS_HUD=0 /path/to/wrapper %command% --game-patch /home/deck/KytyPS5/_Patches/PPSA01341.json`. The log has a
`Successfully applied cheat: <name>` line for each cheat applied.

The log of each run is in `~/KytyPS5/logs/`; attach it to an issue when something goes wrong.

## The Steam Deck package and the Linux package

The Steam Deck package holds the Steam Deck build: code tuned for the Deck's Zen 2 CPU, the Deck's defaults built in,
and the keyboard keys Steam's desktop layout sends left out. The Linux package with its `run-deck.sh` runs the same way
except for those keys and the CPU tuning. Measured side by side on a Deck (the start of Boletaria at `run-deck.sh`'s
settings, the GPU clock on auto, six boots taking turns), the two ran at the same frame rate: 36.0-36.7 frames a second
standing and 34.2-34.7 turning the camera, within the spread between runs. A Steam Deck build with the PGO profile
trained on a Deck (what `build-deck.sh` makes) measured the same. So the Linux package is as fast; the Steam Deck
package differs in the keys. To build the Steam Deck version yourself: `tools/local/deck/build-deck.sh` on Linux, or in
WSL with a distribution no newer than SteamOS's (glibc 2.43; Ubuntu 26.04 matches); `tools/local/deck/README.md` has
the steps and the tools used to measure on the Deck.
