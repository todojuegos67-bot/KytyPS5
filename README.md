> [!NOTE]
> **This fork is tuned for Demon's Souls.** Its renderer, shader and memory paths are optimized for
> **Demon's Souls (PS5, PPSA01341), game versions 1.07 (01.007.000) and 1.05 (01.005.000)**: both are
> supported, with the same performance patches (each version precompiles its own shader caches); other
> versions and games are untested here and may not run. Tested on Windows 11 with an Intel i9-14900K and
> an NVIDIA RTX 5090 (about 45–60 fps at 2560×1440); AMD and Intel GPUs are untested. 32 GB of RAM and a
> GPU with 12 GB of video memory are recommended; it also runs with 16 GB of RAM (slower loading) and 8 GB
> of video memory (the test PC limited to 8 GB: most areas 40–60 fps, the heaviest 10–35).
>
> **Download: [the latest release](https://github.com/chenxiao07/KytyPS5/releases/latest)**, a Windows
> x64 build made by GitHub Actions from this branch. Unzip it, double-click `launcher.cmd` (or
> `run.cmd`) and choose the game folder (the one with `eboot.bin`); the `README.md` inside explains
> the rest. Use only a game dump made from your own copy of the game. To build it yourself, see
> [Building with build-windows.cmd](#building-with-build-windowscmd-windows).
>
> 中文：本分支专为《恶魔之魂》(PS5 版 PPSA01341) 优化，支持游戏版本 1.07 (01.007.000) 和 1.05 (01.005.000)，两者性能补丁相同
> （着色器缓存按版本分别预编译）；其他版本和游戏未测试。
> 下载[最新 release](https://github.com/chenxiao07/KytyPS5/releases/latest)，解压后双击 `launcher.cmd`
> 或 `run.cmd`，选择游戏目录（含 `eboot.bin` 的那一层）。推荐 32 GB 内存、12 GB 以上显存的显卡；16 GB 内存
> （加载较慢）和 8 GB 显存（测试机限制到 8 GB 时：多数区域 40–60 帧，最重的场景 10–35 帧）也能运行。

## Fixes worth cherry-picking

Most of this fork is performance work tied to its own caches, but these three fixes stand on their
own and are useful in other KytyPS5 trees:

**1. Game audio: the game was silent, then positional sounds were nearly inaudible**

- [`f203d0f3`](https://github.com/chenxiao07/KytyPS5/commit/f203d0f3), its audio part only (the commit
  also holds Windows packaging): ATRAC9 streams in a RIFF container required the whole `data` chunk in
  the first stream piece, so every decode returned partial input and the game played no sound besides
  its videos; AudioOut2 object ports are mixed into the main port. Take just its two audio files:
  `git show f203d0f3 -- src/libs/ajm/atrac9_decoder.h src/libs/libAudio2.cpp | git apply`
  (upstream has rewritten its RIFF parsing since, in `e40438f9`: compare before taking that part).
- [`c2b48aeb`](https://github.com/chenxiao07/KytyPS5/commit/c2b48aeb), `git cherry-pick c2b48aeb` after
  the above: the game sends its whole mix as two fifth-order ambisonics scenes on AudioOut2 object
  ports (attribute 8 = 0x40 | ACN, SN3D). Summed into the front pair they cancelled out; they are now
  decoded to stereo from the first order, and 7.1 ports are downmixed on stereo devices without SDL's
  -13.5 dB front level.

**2. Nexus: a ring of blue "?" models** ([`3248e372`](https://github.com/chenxiao07/KytyPS5/commit/3248e372),
`src/libs/libAmpr.cpp` only)

APR file ids were hashes of the guest path. The game resolves all of its ~250K files at start-up, 13
pairs collided, and one file of each pair read the other's bytes: the Nexus arch model failed to load
and the engine drew its default "?" model instead (some LODs, a collision mesh and a texture elsewhere
were affected too). Ids are now assigned one per path. Upstream main (cdb64bfd, 2026-10-06) still hashes.

![The Nexus ceiling before and after 3248e372: the ring of "?" models is gone](docs/images/nexus-question-marks.jpg)

**3. Boletaria 1-1: the brazier's stone pillar and other missing objects**
([`3b351932`](https://github.com/chenxiao07/KytyPS5/commit/3b351932))

Indirect draws name a user SGPR (START_INST_LOC) that the command processor fills with the draw's
start instance, and the shader's instance ID counts from 0. Both were ignored (the ID included
firstInstance), so shaders that index per-object data with that SGPR read stale values: the stone
pillar under the brazier at 1-1, rock faces there, the hall behind the cell windows of the Tower of
Latria. The fix is in the shader recompiler (`Translate.cpp`, the SPIR-V emitter), `shader.cpp/.h`
(the static key), `graphicsRun.cpp` / `hardwareContext.h` (the SGPR per draw) and `vulkanWindow.cpp`
(`shaderDrawParameters`, `drawIndirectFirstInstance`); its `src/local` and `tools` changes only follow
this fork's caches.

![Boletaria 1-1 before and after 3b351932: the brazier floats without its stone pillar before the fix](docs/images/boletaria-pillar.jpg)

# KytyPS5

[![Build KytyPS5 (Windows)](https://img.shields.io/github/actions/workflow/status/KytyPS5/KytyPS5/build.yml?branch=main&event=push&label=Build%20KytyPS5%20%28Windows%29)](https://github.com/KytyPS5/KytyPS5/actions/workflows/build.yml)
[![Build KytyPS5 (Linux)](https://img.shields.io/github/actions/workflow/status/KytyPS5/KytyPS5/build.yml?branch=main&event=push&label=Build%20KytyPS5%20%28Linux%29)](https://github.com/KytyPS5/KytyPS5/actions/workflows/build.yml)
[![Build KytyPS5 (macOS)](https://img.shields.io/github/actions/workflow/status/KytyPS5/KytyPS5/build.yml?branch=main&event=push&label=Build%20KytyPS5%20%28macOS%29)](https://github.com/KytyPS5/KytyPS5/actions/workflows/build.yml)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64%20%7C%20Linux%20x64%20%7C%20macOS%20x86__64-0078D4.svg)](#system-requirements)
[![Status](https://img.shields.io/badge/status-early%20development-orange.svg)](#current-status)
[![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)

KytyPS5 is a free and open-source PlayStation 5 emulator written in C++ for Windows and Linux,
with experimental macOS support. It is based on a heavily modified version of
[Kyty](https://github.com/InoriRus/Kyty). The project is in an early stage of development, so
compatibility is limited and behavior may change significantly between builds.

> [!IMPORTANT]
> KytyPS5 is not affiliated with Sony Interactive Entertainment or PlayStation. The project does
> not distribute games or copyrighted system software. Use only game files that you have obtained
> legally.

## Current Status

KytyPS5 can boot 2D games and a selection of 3D games, including titles built with Unreal Engine
4/5, Unity, and custom engines. No external low-level emulation modules are currently required.

Development is focused on compatibility and boot reliability.

Windows is the primary platform and receives the most testing. Linux builds and runs; see
[Building on Linux](#building-on-linux).

macOS support is experimental. The emulator is built for x86-64 and runs on Apple Silicon under
Rosetta 2, with Vulkan provided by MoltenVK. A small number of titles have been verified in-game
on Apple Silicon hardware; see [Building on macOS](#building-on-macos).

Community game test results are available in the
[KytyPS5 Compatibility List](https://kytyps5.github.io/).

## Bugs and Issues

The project is in an early stage, so please be mindful when opening new issues. Expect crashes,
graphical glitches, low compatibility, and poor performance.

## Screenshots

<table align="center">
  <tr>
    <td align="center">
      <strong>Disgaea 6</strong><br>
      <img src="docs/screenshots/ps5-01.png" width="300" alt="Disgaea 6 running in KytyPS5">
    </td>
    <td align="center">
      <strong>Dreaming Sarah</strong><br>
      <img src="docs/screenshots/ps5-03.png" width="300" alt="Dreaming Sarah running in KytyPS5">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Neptunia ReVerse</strong><br>
      <img src="docs/screenshots/ps5-04.png" width="300" alt="Neptunia ReVerse running in KytyPS5">
    </td>
    <td align="center">
      <strong>SILENT HILL: The Short Message</strong><br>
      <img src="docs/screenshots/ps5-05.png" width="300" alt="SILENT HILL: The Short Message running in KytyPS5">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Hellboy</strong><br>
      <img src="docs/screenshots/ps5-02.png" width="300" alt="Hellboy running in KytyPS5">
    </td>
    <td align="center">
      <strong>Paleo Pines</strong><br>
      <img src="docs/screenshots/ps5-06.png" width="300" alt="Paleo Pines running in KytyPS5">
    </td>
  </tr>
</table>

<p align="center"><em>And many more...</em></p>

## Contributing

Testing games and submitting detailed bug reports are useful ways to contribute. Search existing
issues first, then use the **Game Emulation Bug Report** template and attach the complete log file.

Code contributions should be focused, build successfully on the platforms they touch, and include
relevant tests where practical. Windows is the primary target, so a change that alters shared code
should not regress it; changes confined to a platform's own code paths only need to build there. Because KytyPS5 is still evolving quickly, consider opening an issue before
starting a large change.

### Formatting

Set up the clang-format hook after cloning:

```powershell
python -m pip install pre-commit
python -m pre_commit install --install-hooks
```

It formats staged `.cpp`, `.h`, and `.inc` files in `src`.

## Developer Information

The PS5 graphics architecture is based on AMD RDNA 2. Use AMD's
[RDNA 2 Instruction Set Architecture Reference Guide (document 70648)](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)
as the primary instruction-encoding reference when working on shader decoding and recompilation.

Important areas of the codebase:

- [`src/graphics/shader/recompiler`](src/graphics/shader/recompiler) — instruction decoding,
  intermediate representation, control flow, resource tracking, and SPIR-V emission
- [`src/graphics/guest_gpu`](src/graphics/guest_gpu) — PS5 (Prospero) GPU formats and command processing
- [`src/graphics/host_gpu`](src/graphics/host_gpu) — Vulkan host backend and resource management
- [`tests`](tests) — focused memory, shader, and resource-tracking regression tests

The renderer targets Vulkan 1.3. Keep shader changes aligned with both the RDNA 2 ISA semantics and
the Vulkan/SPIR-V validation rules.

## Building

### System requirements

- Windows 10 version 1803, a current Linux distribution, or macOS on Apple Silicon
- A 64-bit x86 processor (on macOS, an Apple Silicon processor with Rosetta 2)
- A Vulkan 1.3-capable GPU with current drivers (on macOS, Vulkan is provided by the bundled
  MoltenVK)

### Building with build-windows.cmd (Windows)

The shortest way to a Release build of the emulator (without the Qt launcher):

1. Install the tools once:
   - Git: `winget install Git.Git`
   - Visual Studio 2022 or Build Tools 2022 with the **Desktop development with C++** workload (its
     **C++ CMake tools for Windows** component brings CMake and Ninja)
   - LLVM 19 for `clang-cl`: `winget install LLVM.LLVM --version 19.1.7` (19.1.7 is tested; the
     **C++ Clang tools for Windows** component of Visual Studio is used when LLVM is not installed;
     LLVM 23.1.2 crashes compiling `agc.cpp`)
   - Vulkan SDK, for `glslangValidator`: `winget install KhronosGroup.VulkanSDK`
   - Python 3: `winget install Python.Python.3.12`
2. Clone and build from a new terminal (so that the tools are on `PATH`):

   ```powershell
   git clone --recurse-submodules https://github.com/chenxiao07/KytyPS5.git
   cd KytyPS5
   .\build-windows.cmd
   ```

   The script finds Visual Studio, `clang-cl` and the Vulkan SDK, checks out submodules that are
   missing, configures `_Build\windows` the first time (CMake downloads xbyak and zydis) and builds
   `_Build\windows\kyty_emulator.exe`. Run it again after changes for an incremental build;
   `.\build-windows.cmd all` builds every target.
3. Start a game with its folder (the one with `eboot.bin`; it is remembered in `game-path.txt`):

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run-windows.ps1 -Game "D:\Games\PPSA01341-app0"
   ```

   The emulator switches come from `run-windows.json`; `-DryRun` prints the environment and the
   command, `-Fullscreen`, `-Width`/`-Height` and the others are listed at the top of the script.

The build is optimized for the CPU it is built on (`-march=native`). For an executable that other
PCs can run, set `KYTY_CMAKE_ARGS=-DKYTY_MARCH=x86-64-v3` before the first configure. The release
package is built that way by [`.github/workflows/build.yml`](.github/workflows/build.yml) when
**Actions > Build KytyPS5 (Windows) > Run workflow** is started or a `v*` tag is pushed, and published as a
release (pushes build nothing).

### Build requirements (Windows)

- Git
- CMake 3.12 or newer
- Ninja
- Visual Studio 2022 or Build Tools 2022 with the **Desktop development with C++** workload and
  **C++ Clang tools for Windows** component
- The Vulkan SDK (`glslangValidator`) and Python 3
- Qt 6 for MSVC 2022 64-bit, including Concurrent, Network, and Widgets

The Microsoft C++ compiler (`cl.exe`) is not supported; use `clang-cl`.

Open an **x64 Native Tools Command Prompt for Visual Studio 2022** (or the equivalent Developer
PowerShell), change to the repository root, and initialize the dependencies:

```powershell
git submodule update --init --recursive
```

Configure the project. Replace the Qt path with the version installed on your system:

```powershell
cmake -S . -B _Build/windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2022_64"
```

Build the launcher and stage a runnable installation:

```powershell
cmake --build _Build/windows --target launcher
cmake --install _Build/windows --prefix _Build/windows/install
```

The finished application and its runtime dependencies will be placed in
`_Build/windows/install`.

### Building on Linux

Install the toolchain and the libraries the bundled SDL2 needs. Without the audio, Wayland and
udev development packages SDL2 quietly configures itself without those backends, and the resulting
build has no working sound and no gamepad hotplug:

```bash
sudo apt-get install --no-install-recommends \
  clang lld ninja-build cmake git glslang-tools \
  libgl1-mesa-dev libx11-dev libxcursor-dev libxext-dev libxfixes-dev \
  libxi-dev libxrandr-dev libxss-dev libxkbcommon-dev \
  libasound2-dev libpulse-dev libudev-dev libdbus-1-dev libwayland-dev wayland-protocols
```

Qt 6 (Concurrent, Network, Widgets) is also required — either the distribution packages
(`qt6-base-dev`) or an official Qt installation.

```bash
git submodule update --init --recursive

cmake -S . -B _Build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_PREFIX_PATH="$Qt6_DIR"

cmake --build _Build/linux --target launcher --parallel
cmake --install _Build/linux --prefix _Build/linux/install
```

The install step copies the Qt libraries and plugins next to the binaries, so
`_Build/linux/install` runs without a matching system Qt.

As on Windows, the MSVC compiler is not used; Clang is required. `cl.exe` is rejected at configure
time.

The CMake source root is the repository root.

### Building on macOS

macOS builds target x86-64 and run under Rosetta 2 on Apple Silicon, so the PS5's x86-64 game
code executes through the same translation layer as the emulator itself. Prebuilt archives are
attached to releases; the steps below are for building from source.

Requirements:

- An Apple Silicon Mac with Rosetta 2 installed (`softwareupdate --install-rosetta`)
- Xcode (or the Command Line Tools)
- Homebrew packages: `brew install cmake ninja glslang`
- Qt 6 (Concurrent, Network, Widgets) with x86-64 support. The official Qt installation is
  universal and works; Homebrew's Qt is arm64-only and will not link

```bash
git submodule update --init --recursive

cmake -S . -B _Build/macos -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_PREFIX_PATH="$Qt6_DIR"

cmake --build _Build/macos --target launcher --parallel
cmake --install _Build/macos --prefix _Build/macos/install
```

The build re-signs `kyty_emulator` with the JIT entitlements it needs to execute translated
guest code; no manual signing step is required. When the launcher is built, the install
also produces `_Build/macos/install/KytyPS5.app` — double-click to launch the GUI.
A flat `kyty_emulator` is kept for CLI usage.

Vulkan comes from MoltenVK. Download `MoltenVK-macos.tar` from the
[MoltenVK releases](https://github.com/KhronosGroup/MoltenVK/releases), then copy
`MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib` next to the flat `kyty_emulator`
(and, for the bundle, into `KytyPS5.app/Contents/Frameworks/`) and ad-hoc sign it:

```bash
codesign --force --sign - _Build/macos/install/libMoltenVK.dylib
# For the bundle (if present):
codesign --force --sign - _Build/macos/install/KytyPS5.app/Contents/Frameworks/libMoltenVK.dylib
codesign --force --sign - _Build/macos/install/KytyPS5.app
```

Release archives already include a signed `libMoltenVK.dylib` (both flat and inside the bundle).

### Regression tests

Build every regression executable and run the registered tests with:

```powershell
cmake --build _Build/windows --target kyty_tests
ctest --test-dir _Build/windows --output-on-failure
```

Use `_Build/linux` instead of `_Build/windows` for a Linux build.

### Visual Studio Code

A ready-made Visual Studio Code setup is included in [`.vscode`](.vscode). It configures CMake
Tools to build the project with Ninja and `clang-cl` and provides launch profiles for both
`launcher.exe` and `kyty_emulator.exe`. It is Windows-only: VS Code settings cannot select a
compiler per platform, so on Linux configure from the command line as shown above.

Before using it:

1. Install the **CMake Tools** and **C/C++** extensions in Visual Studio Code.
2. Update `CMAKE_PREFIX_PATH` in [`.vscode/settings.json`](.vscode/settings.json) to point to your
   Qt 6 MSVC installation.
3. Update the `--game` path in [`.vscode/launch.json`](.vscode/launch.json) for the
   **Debug kyty_emulator** profile.
4. Open the repository in an x64 Visual Studio developer environment, configure the CMake project,
   and select a launch profile from **Run and Debug**.

## Running

Update your graphics driver before reporting rendering problems.

To use the graphical launcher:

```powershell
.\_Build\windows\install\launcher.exe
```

```bash
./_Build/linux/install/launcher
```

```bash
open _Build/macos/install/KytyPS5.app  # or double-click in Finder
```

On first launch, add one or more game folders in the global settings. The launcher searches those
folders recursively for game directories containing `eboot.bin`. Select a detected game and run it
from the game list.

The emulator can also be started directly with a legally obtained game directory or ELF file:

```powershell
.\_Build\windows\install\kyty_emulator.exe --game "D:\Games\ExampleGame"
```

```bash
./_Build/linux/install/kyty_emulator --game "/games/ExampleGame"
```

On macOS, the adjacent flat or app-bundled `libMoltenVK.dylib` is found automatically; no
environment variable is required:

```bash
./_Build/macos/install/kyty_emulator --game "/games/ExampleGame"
```

To override the Vulkan loader, set `SDL_VULKAN_LIBRARY`:

```bash
SDL_VULKAN_LIBRARY=/path/to/libMoltenVK.dylib ./kyty_emulator --game "/games/ExampleGame"
```

Run `kyty_emulator --help` to see the available graphics, logging, validation, profiling, and
debugging options.

### AI Use

AI tools may be used for research, reverse engineering, and development assistance. Contributors
must fully understand, review, and test all code they submit and remain responsible for its
correctness. Repository communication, including pull-request descriptions, code comments, and
issue comments, must come from the human contributor rather than an autonomous AI agent.

Pull requests that include AI-assisted or AI-generated work should disclose the scope of the AI
involvement and describe the human review and testing performed before submission. Unverified or
untested generated changes may be closed without review.

## License

KytyPS5 is licensed under the [GNU General Public License version 2](LICENSE)
(`GPL-2.0-only`).

This project is based on the original [Kyty](https://github.com/InoriRus/Kyty), which was released
under the MIT License. Kyty's original copyright and license notice are preserved in
[`LICENSES/Kyty-MIT.txt`](LICENSES/Kyty-MIT.txt). Third-party components remain subject to the
licenses included with those components.

## Special Thanks

- [InoriRus/Kyty](https://github.com/InoriRus/Kyty) — KytyPS5 is based on a heavily modified version
  of the original Kyty project.
- [shadps4-emu/shadPS4](https://github.com/shadps4-emu/shadPS4) — reference for memory-model
  understanding and the AVPlayer implementation.
