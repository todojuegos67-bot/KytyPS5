# kyty_launcher (Linux)

The settings window for Linux: the pages and settings of the Windows launcher (`launcher-windows/`), drawn with
Dear ImGui on SDL2's renderer. Both are linked in statically (the emulator's own `3rdparty/` copies), so the program
needs nothing installed beyond what the emulator needs. `build-linux.sh` builds it next to `kyty_emulator`, and the
Linux package carries it.

It reads and writes `launcher-settings.json` next to `run_ds.sh`, with the Windows launcher's keys (unknown keys are kept),
or in `$XDG_CONFIG_HOME/KytyPS5/` when that folder is read-only. PLAY starts `run_ds.sh` in a terminal window with
the settings as the emulator's switches, its output also in `logs/<time>.log` (`_Build/run-logs/` in a build tree).
The precompile and DLSS's build (`precompile-linux.sh`, `get-dlss-linux.sh`) run in a terminal window too, where they
can ask y/N and for the sudo password.

```
./kyty_launcher                the window
./kyty_launcher --play         the game with the saved settings, no window (Steam: a non-Steam game); its exit code
./kyty_launcher --lang pt      the window in English (en), Chinese (zh) or Portuguese (pt) for this run
./kyty_launcher --dry-run      the scripts the buttons run, printed
./kyty_launcher --screenshot <folder>   each page as <n>-<page>.bmp
```

## Settings and what they set

| Page | Setting | Emulator switch / option |
| --- | --- | --- |
| Display | Resolution | `--screen-width`, `--screen-height` |
| | Aspect ratio 21:9, 32:9 | `KYTY_RENDER_ASPECT` |
| | Movies and menus at 16:9 (black bars), with 21:9 or 32:9 | `KYTY_PRESENT_ASPECT=fit` |
| | Fullscreen, keep the aspect ratio | `--fullscreen`, `KYTY_PRESENT_ASPECT=fit` |
| | HDR | `--hdr` |
| | Up to 120 fps | `KYTY_VBLANK=120` (run_ds.sh adds `gameMinDeltaTime`) |
| | VSync | `PRESENT_MODE=Fifo` (run_ds.sh's mailbox otherwise) |
| | Low latency | `KYTY_LOW_LATENCY=1` |
| Graphics | Upscaler DLSS / FSR 3 | `KYTY_UPSCALER=dlss` / `fsr` |
| | 3D render height | `KYTY_RENDER_HEIGHT` |
| | Game output 1080p (or Automatic for a 16:9 window up to 1080 lines) | `KYTY_HD_OUTPUT=1` |
| | Game output 1440p (with DLSS or FSR 3) | `KYTY_OUTPUT_HEIGHT=1440` |
| | DLSS model | `KYTY_DLSS_PRESET` |
| | Sharpening Off / Low / Medium / High: DLSS (Off by default) | `KYTY_DLSS_SHARPNESS=0.3` / `0.6` / `0.9` (Off: none) |
| | Sharpening Off / Low / Medium / High: FSR 3 (Medium by default) | `KYTY_FSR_SHARPNESS=0` / `0.3` / `0.9` (Medium: none) |
| | Graphics preset Low / Lowest (Lowest: no decals too) | `KYTY_GAME_CONVARS` (Demon's Souls), `KYTY_TEXTURE_REDUCE=1` / `2` |
| | DLSS frame generation 2x-6x | `KYTY_FRAMEGEN=1`-`5` |
| | DLSS runtime | runs `get-dlss-linux.sh <the emulator's folder>` |
| | AMD FSR 3 frame generation | `KYTY_FRAMEGEN=1`, `KYTY_FRAMEGEN_BACKEND=fsr` |
| Game | Console language | `--console-language` |
| | Audio output | `KYTY_AUDIO_OUTPUT` |
| | Cheats | `--game-patch _Patches/<title id>.json`; Choose… sets the mods' `enabled` flags |
| Shaders | Precompile, Quick, efficiency cores | `precompile-linux.sh --game … [--quick] [--cpus <E-cores>]` |
| | Export shader hints | `precompile-linux.sh --export-hints` |
| Advanced | Performance overlay | `KYTY_FPS_HUD=1` / `2` |
| | MangoHud | `MANGO=1` / `0` |
| | Safe GPU mode | `KYTY_COMPUTE_CHAIN=0` |
| | Graphics card (more than one) | `KYTY_GPU=<its name>` (the precompile's too) |
| | Extra options | `KEY=VALUE` (replaces the launcher's; `KEY=` removes it, or sets an emulator default to `0`), `--option value` (replaces the launcher's option of that name; a repeated one such as `--keymap L1=Q --keymap R1=E` keeps each), also the Windows form `-Set A=1,B=2` |
| Experimental | Lock to 30 fps | `KYTY_FLIP_RATE=1` |
| | Direct presentation | `KYTY_PRESENT_DIRECT=1` |
| | AMD Anti-Lag | `KYTY_ANTI_LAG=1`, `KYTY_LOW_LATENCY=1` |

The game is `KYTY_GAME`, the emulator `KYTY_BIN` (the `kyty_emulator` next to the launcher, or `KYTY_BIN` when the
launcher was started with it). The variables above that a start does not set are unset, so what the shell exported
does not change a start.

The terminal is the desktop's own (Konsole on KDE, Ptyxis, Console or GNOME Terminal on GNOME, and so on), else the
first of the usual ones found, or the one chosen on the Advanced page. The folder dialog is kdialog, zenity or qarma when
one is installed, else a built-in one.

The Windows launcher's red-zone protection, Streamline folder and "Keep memory resident" are Windows-only and not shown;
their keys stay in the file.
