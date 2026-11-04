# Demon's Souls keyboard and mouse prompts

An optional set of 22 replacement button icons for Demon's Souls (PS5). Tested visually with PPSA01341, version 01.007.000, on Windows using KytyPS5. Other releases are untested.

![Interact: E](icons/ps4_button_cross.png)
![Dodge/back: Space](icons/ps4_button_circle.png)
![Lock-on: middle mouse](icons/ps4_r3.png)

The launcher generates English labels matching its saved keyboard layout and applies the same controls for that launch. The labels remain visible when using a controller; there is no automatic input-device switching. **Press F7 to capture/release the mouse for camera control.** Standalone texture installation does not remap controls: configure them separately to match the labels below.

| Original prompt | Default replacement |
| --- | --- |
| Cross / Circle / Square / Triangle | E / Space / R / F |
| L1 / L2 / L3 | Shift / Ctrl / X |
| R1 / R2 / R3 | Left mouse / Right mouse / Middle mouse |
| Left stick / Right stick | WASD / Mouse |
| D-pad | Arrow keys |
| Options | Esc |
| Touchpad / left touchpad / right touchpad | Tab / Tab / Backspace |

Share and touchpad-swipe textures are left alone. `labels.json` documents the shipped artwork; changing that file's label strings alone does not redraw the PNGs.

## Launcher option

![Optional keyboard prompt setting on the Game page](https://github.com/user-attachments/assets/10920d72-a5b8-4034-8291-473dac9e1434)

In the Windows launcher, open **Game → Keyboard and mouse prompts**. It is off by default. Enabling it applies the bundled icons and matching controls when you press **Play**, including launcher's `--play` shortcut mode. Disabling it restores the original icons on the next **Play** and stops adding the bindings. The launcher passes `-KeyboardMouseControls` and an encoded layout through `-KeyboardMouseBindings` to `run-windows.ps1`, which appends the matching `--keymap` pairs after the configured bindings. These take precedence for conflicting controls/keys; mouse sensitivity and non-conflicting wheel bindings remain unchanged. It never rewrites `launch.json`, so disabling the option returns to your configured controls (or emulator defaults). With this layout, Space goes to the game; use the Pause key to pause emulation. Simply changing the toggle does not modify any files.

### Edit controls locally

Open **Game → Keyboard controls → Edit controls…**. Click an action and press a key, mouse button, or wheel direction. Assigning an input already used by another action swaps the two bindings. The preview shows the new key label. **Save** stores the layout in `launcher-settings.json`; **Cancel** leaves it unchanged. **Reset defaults** resets the draft, so you can still cancel. F7, F8, F11 and Pause are reserved for emulator functions. Single keys, including modifiers, navigation keys, function keys, numpad and punctuation, are supported; key combinations are not.

Mouse sensitivity accepts 0.1–5. Leave it blank to keep the existing `launch.json` sensitivity or emulator default. Enable the keyboard-prompts option and press **Play** to apply the saved layout and its icons. Editing alone does not change game files. The original default artwork is reused for unchanged labels; changed labels are rendered locally, including all eight mip levels. Mouse-look still requires **F7**.

![Editable keyboard controls with label previews](https://github.com/user-attachments/assets/95f21c41-1784-4cc1-8ca5-25975f1da207)

In-game example with Select rebound to **N**:

![Custom N Select prompt in the Archstone Travel menu](https://github.com/user-attachments/assets/50f26d8d-a5db-4c18-9b33-ea5b49d82702)

Generated variants are recorded in `.kyty-keyboard-prompts-v1/generated` before any game icon is replaced. This allows exact restoration after changing profiles or an interrupted update, while unrelated texture modifications are still rejected. Keep the backup folder. To roll back to a launcher without custom-layout support, first disable keyboard prompts and press Play with this launcher so it restores the original icons.

This option supports an extracted **PPSA01341 version 01.007.000** folder with the original icon files. ZArchive games and other versions are not supported. It verifies every original icon, saves originals in `.kyty-keyboard-prompts-v1` inside the game folder, and retains that backup when disabled. Modified icons, missing originals, and damaged backups stop the operation instead of overwriting other mods. Close the game before applying either state.

If applying stops halfway, close the game, turn this option off and press **Play** for the same folder to restore originals from `.kyty-keyboard-prompts-v1`. Keep that backup folder. Missing or damaged backups must be recovered from your own original game files first.

If you previously installed these icons manually, restore your original icons first so the launcher can create a verified backup. Switching to another game folder does not restore the old folder automatically: disable the option and press Play for that old folder first. Directly starting the emulator or `run-windows.ps1` bypasses the launcher option.

The launcher includes generated CTXR artwork and needs no extra tools or downloads. No original controller artwork is bundled. The standalone instructions below are for rebuilding/customizing the assets.

## Build the game textures

Requirements: Windows, PowerShell 7, your own game dump, and the extracted [Aqua Toolset 9_19_2026](https://github.com/Shadowth117/Aqua-Toolset/releases/tag/9_19_2026), including its .NET runtime requirements. Point `AquaToolsetDirectory` at the folder containing `AquaModelLibrary.Data.dll` (the tested package uses `net9.0-windows7.0`). No Aqua binaries or original game textures are included here.

```powershell
pwsh -File .\Build-Textures.ps1 `
  -IconDirectory 'D:\Games\DemonsSouls\PPSA01341\ui\textures\buttonicons\_ps5' `
  -AquaToolsetDirectory 'C:\Tools\AquaToolset\net9.0-windows7.0' `
  -OutputDirectory 'C:\Mods\DemonsSoulsKeyboardPrompts'
```

The script reads the original CTXR headers from your dump, replaces all pixel/mipmap data with the supplied artwork, and writes to a new directory. It checks all eight mip levels after decoding the result. It never writes to the game directory. Only install after it reports all 22 prompts built successfully; discard incomplete output after an error and retry with a new output directory.

## Install and restore

1. Close the game. Back up the complete `ui/textures/buttonicons/_ps5` directory somewhere outside the game folder.
2. Copy the 22 generated `.ctxr` files into that directory, replacing the matching files.
3. Start the game. Your bindings, launcher settings, saves and shader caches are unchanged.

To uninstall, close the game and restore the original files from your backup. Keep the backup: running the converter on already modified textures will not recover the original artwork.

## Artwork and attribution

The PNGs were drawn procedurally as white keycaps with text and arrow labels using Windows System.Drawing and Segoe UI. They contain no original controller icon pixels; no font files are bundled. The artwork and conversion script are supplied under the repository's GPL-2.0-or-later license. CTXR decoding/encoding is provided by [Shadowth117's Aqua Toolset](https://github.com/Shadowth117/Aqua-Toolset) and [PSO2 Aqua Library](https://github.com/Shadowth117/PSO2-Aqua-Library), which are downloaded separately under their respective licenses. The generated textures retain the compatible CTXR layout from the user's templates; every pixel and mip level is replaced.

To regenerate the launcher's compressed resources after conversion, run Python 3.10 or newer:

```powershell
python .\pack-launcher-assets.py --originals 'D:\OriginalIcons' --converted 'C:\Mods\DemonsSoulsKeyboardPrompts'
```

Commit the resulting `launcher-windows/KeyboardPrompts` files alongside any artwork changes. The manifest records original and replacement SHA-256 hashes. Local integration tests use your originals as temporary fixtures and do not modify them:

```powershell
.\tests\Test-KeyboardPrompts.ps1 -OriginalIconDirectory 'D:\OriginalIcons'
# After building the launcher (no game files needed):
.\tests\Test-KeyboardPromptControls.ps1
# Brief editor UI tests with temporary settings:
powershell -STA -File .\tests\Test-KeyboardControlsEditor.ps1
# Independent verification of all generated mip levels (PowerShell 7 + Aqua):
.\tests\Test-PromptTextureRoundtrip.ps1 -AquaToolsetDirectory 'C:\Tools\AquaToolset'
```

Run that last command from the repository root with Visual Studio Build Tools installed.

### Runtime texture layout resources

`Build-PixelMap.ps1` uses Aqua Toolset and one verified 128×128 original icon to generate a zero-pixel CTXR template and the pixel-offset map embedded in the launcher. No original pixel content is retained. The runtime renderer uses only .NET drawing APIs and these resources; Aqua is not required by users. Each icon keeps its own metadata, and custom mip data is validated against Aqua's independent decoder by `Test-PromptTextureRoundtrip.ps1`.
