## The shader precompile is made per PC and cannot be shared

The precompiled shader caches belong to **one graphics card model and one driver version**, on the PC that
made them:

- Every player precompiles once on their own PC: choose "Precompile first" when the launcher asks, or
  double-click `precompile.cmd`. It takes about 55 minutes with 22 CPU threads and about 2.5 hours with 8;
  closing the window stops it and running it again continues.
- A `_PipelineCache` folder or cache files from another PC do not work, even for the same game version:
  do not share them or download them from others.
- After a graphics driver update or a change of graphics card, precompile again (the launcher asks).
- Game versions 1.07 and 1.05 each have caches of their own.
- Precompiled with v20261010 or older? Run the precompile again: this build compiles many more of the
  shaders the game uses (what is already compiled is kept, so it takes a fraction of the first run). The
  launcher asks.
- The game also runs without the precompile, but stutters the first time an area or effect appears.
- The package holds no list of the game's shaders: the first launch makes it from your own game files.

## New in this build

- **Far fewer shader stutters after the precompile**: it now also compiles the vertex shaders of the game's
  indirect draws (most of its scenery), which it left out before, and the shaders as the game specialized
  them while playing every world on the test PC. Those come with the package as "hints" without the game's
  code (`tools\local\static-precompile\hints-*.hints`) and are completed with your own game files. Game
  version 1.05 on a PC with fresh caches, after the precompile: at Boletarian Palace 1-1 the shaders
  compiled during play while the game waits went from 274 (60 s in all) to 0, and the HUD shows after 45 s
  instead of 93; walking there, the longest frame was 43 ms instead of 1.4 s. At the Tower of Latria: 150 to
  0, the HUD after 43 s instead of 77. The precompile takes about 10 minutes longer and needs about 2.5 GB
  more disk space.
- **The launcher asks to precompile again** when a new build precompiles more (only what is new is
  compiled).
- **AMD RDNA3 and later**: the precompile now compiles the compute shaders as these cards run them (64-lane
  waves); before, none of them was used. (Untested here: the test PC has no AMD GPU.)
- **Fewer hitches while walking**: large texture uploads are copied in parallel and no longer hold up
  recording the frame (1-3 walk: the longest frames 44 -> 40 ms), new draws no longer wait for a per-frame
  quota before they take the fast path (1-3 walk 1% low +12%), and fewer memory protection changes.

## Frame rates on the test PC

i9-14900K + RTX 5090, 2560×1440, precompiled, with "Up to 120 fps" on so the numbers show the headroom:
average fps over 10 seconds (1% low in brackets) standing still, walking around and turning the camera at
one spot of each area. With the default 60 fps cap, values of about 62 and more are a steady 60.

| World | Standing | Moving | Turning the camera |
| --- | ---: | ---: | ---: |
| 1-1 Boletarian Palace | 60 (54) | 80 (50) | 83 (53) |
| 1-2 Boletarian Palace | 62 (55) | 74 (48) | 76 (28) |
| 1-3 Boletarian Palace | 69 (59) | 67 (36) | 76 (57) |
| 1-4 Boletarian Palace | 66 (59) | 72 (57) | 77 (64) |
| 2-1 Stonefang Tunnel | 85 (71) | 86 (60) | 92 (65) |
| 2-2 Stonefang Tunnel | 120 (90) | 120 (75) | 120 (88) |
| 3-1 Tower of Latria | 60 (52) | 58 (37) | 64 (44) |
| 3-2 Tower of Latria | 84 (69) | 75 (42) | 73 (51) |
| 4-1 Shrine of Storms | 74 (62) | 84 (58) | 85 (48) |
| 4-2 Shrine of Storms | 98 (75) | 89 (61) | 92 (66) |
| 5-1 Valley of Defilement | 80 (67) | 97 (60) | 102 (67) |
| 5-2 Valley of Defilement | 103 (81) | 91 (49) | 93 (63) |
| Nexus | 87 (69) | 82 (45) | 83 (62) |

2-2: the test spot is at a fog gate where almost nothing is drawn. 1-2 turning: the camera turns where the walk ended, which differs from run to run; some runs see one heavy view (1% low 28 to 58 in runs of the same build).

## 着色器预编译只对本机有效，不能共享

预编译生成的着色器缓存只对应**生成它的那台电脑的显卡型号和驱动版本**：

- 每位玩家都需要在自己的电脑上预编译一次：启动器询问时选择 "Precompile first"，或者双击 `precompile.cmd`。
  22 线程约 55 分钟，8 线程约 2.5 小时；关闭窗口会中止，再次运行会从中断处继续。
- 从别人电脑复制来的 `_PipelineCache` 文件夹或缓存文件无法使用（即使游戏版本相同），请不要分享或下载。
- 更新显卡驱动或更换显卡后需要重新预编译（启动器会提示）。
- 游戏 1.07 和 1.05 各有自己的缓存。
- 用 v20261010 或更早的版本预编译过的，请重新运行一次预编译：本版本会编译多得多的游戏实际使用的着色器（已编译的部分会保留，所以只需第一次的一小部分时间）。启动器会提示。
- 不预编译也能玩，但第一次进入新区域或出现新特效时会卡顿。
- 发布包里不包含游戏的着色器列表：第一次启动时会从你自己的游戏文件生成。

## 本版本新增

- **预编译后的着色器卡顿大幅减少**：预编译现在也会编译游戏间接绘制（大部分场景物体）用到的顶点着色器（此前漏掉了），以及在测试机上游玩各个世界时游戏实际特化出的着色器。后者以不含游戏代码的“提示”文件随发布包提供（`tools\local\static-precompile\hints-*.hints`），预编译时用你自己的游戏文件补全。以 1.05 版在缓存全新的电脑上预编译后为例：在 1-1 波雷塔利亚王城，游戏运行中需要等待的着色器编译从 274 个（共 60 秒）降到 0 个，HUD 出现从 93 秒提前到 45 秒；在那里走动时最长的一帧从 1.4 秒降到 43 毫秒。在拉特利亚之塔：从 150 个降到 0 个，HUD 从 77 秒提前到 43 秒。预编译时间约增加 10 分钟，磁盘占用约增加 2.5 GB。
- **新版本预编译的内容更多时，启动器会提示重新预编译**（只编译新增的部分）。
- **AMD RDNA3 及更新的显卡**：预编译现在按这些显卡运行计算着色器的方式（64 线程波）编译；此前预编译的计算着色器在这些显卡上一个都用不上。（测试机没有 AMD 显卡，未实测。）
- **走动时的卡顿更少**：大纹理上传并行拷贝，不再拖住帧的录制（1-3 走动：最长帧 44 -> 40 毫秒）；新出现的绘制不再受每帧配额限制即可走快速路径（1-3 走动 1% low +12%）；内存保护的切换也更少了。

## 测试机上的帧率

i9-14900K + RTX 5090，2560×1440，已预编译，开启 "Up to 120 fps" 以显示性能余量：每个区域的一个位置上站立、走动、转动镜头各 10 秒的平均帧率（括号内为 1% low）。默认 60 帧上限时，约 62 以上的数值都是稳定的 60 帧。

| 世界 | 站立 | 移动 | 转动镜头 |
| --- | ---: | ---: | ---: |
| 1-1 Boletarian Palace | 60 (54) | 80 (50) | 83 (53) |
| 1-2 Boletarian Palace | 62 (55) | 74 (48) | 76 (28) |
| 1-3 Boletarian Palace | 69 (59) | 67 (36) | 76 (57) |
| 1-4 Boletarian Palace | 66 (59) | 72 (57) | 77 (64) |
| 2-1 Stonefang Tunnel | 85 (71) | 86 (60) | 92 (65) |
| 2-2 Stonefang Tunnel | 120 (90) | 120 (75) | 120 (88) |
| 3-1 Tower of Latria | 60 (52) | 58 (37) | 64 (44) |
| 3-2 Tower of Latria | 84 (69) | 75 (42) | 73 (51) |
| 4-1 Shrine of Storms | 74 (62) | 84 (58) | 85 (48) |
| 4-2 Shrine of Storms | 98 (75) | 89 (61) | 92 (66) |
| 5-1 Valley of Defilement | 80 (67) | 97 (60) | 102 (67) |
| 5-2 Valley of Defilement | 103 (81) | 91 (49) | 93 (63) |
| Nexus | 87 (69) | 82 (45) | 83 (62) |

2-2：测试位置在雾门前，几乎不绘制任何东西。1-2 转动镜头：在走动结束的位置转动，每次运行位置不同，有的运行会转到一个很重的视角（同一版本的多次运行中 1% low 在 28 到 58 之间）。
