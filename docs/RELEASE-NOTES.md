## The shader precompile is made per PC and cannot be shared

The precompiled shader caches belong to **one graphics card model and one driver version**, on the PC that
made them:

- Every player precompiles once on their own PC: choose "Precompile first" when the launcher asks, or
  double-click `precompile.cmd`. It takes about 45 minutes with 22 CPU threads and about 2 hours with 8;
  closing the window stops it and running it again continues.
- A `_PipelineCache` folder or cache files from another PC do not work, even for the same game version:
  do not share them or download them from others.
- After a graphics driver update or a change of graphics card, precompile again (the launcher asks).
- Game versions 1.07 and 1.05 each have caches of their own.
- Precompiled with an older build? Run the precompile again: this build compiles more pipeline variants
  (what is already compiled is kept).
- The game also runs without the precompile, but stutters the first time an area or effect appears.
- The package holds no list of the game's shaders: the first launch makes it from your own game files.

## New in this build

- **"Up to 120 fps"** in the launcher (off by default): the game renders up to 120 frames a second
  instead of 60. Movies play faster while it is on.
- Faster everywhere (see the table): compute shaders that exchange nothing between lanes run one GPU lane
  per invocation (about 10% less GPU time in the Tower of Latria), more draws and compute work take the
  table path, and many smaller savings on the render thread.
- Fewer hitches: texture streaming updates only the parts the game rewrote, the precompile also covers the
  table and native pipeline variants, and the driver cache is saved while you play.
- Fixes: GPU memory garbage collection never ran (the game presents from the GPU), so video memory use
  kept growing over a session on cards with less memory; above the display's refresh rate every refresh
  now shows the newest frame; a partial texture update no longer loses the GPU's other bytes of the texture.

## Frame rates on the test PC

i9-14900K + RTX 5090, 2560×1440, precompiled, with "Up to 120 fps" on so the numbers show the headroom:
average fps over 10 seconds (1% low in brackets) standing still, walking around and turning the camera at
one spot of each area. With the default 60 fps cap, values of about 62 and more are a steady 60.

| World | Standing | Moving | Turning the camera |
| --- | ---: | ---: | ---: |
| 1-1 Boletarian Palace | 58 (54) | 72 (44) | 82 (51) |
| 1-2 Boletarian Palace | 65 (57) | 72 (43) | 79 (53) |
| 1-3 Boletarian Palace | 72 (61) | 69 (27) | 79 (59) |
| 1-4 Boletarian Palace | 68 (61) | 73 (51) | 76 (23) |
| 2-1 Stonefang Tunnel | 87 (72) | 86 (53) | 93 (65) |
| 2-2 Stonefang Tunnel | 120 (95) | 119 (59) | 120 (88) |
| 3-1 Tower of Latria | 63 (54) | 60 (31) | 62 (44) |
| 3-2 Tower of Latria | 87 (70) | 76 (35) | 74 (53) |
| 4-1 Shrine of Storms | 77 (67) | 89 (35) | 87 (44) |
| 4-2 Shrine of Storms | 95 (71) | 87 (34) | 83 (29) |
| 5-1 Valley of Defilement | 85 (70) | 95 (40) | 104 (69) |
| 5-2 Valley of Defilement | 104 (84) | 92 (38) | 93 (63) |
| Nexus | 88 (70) | 81 (36) | 84 (58) |

2-2: the test spot is at a fog gate where almost nothing is drawn.

## 着色器预编译只对本机有效，不能共享

预编译生成的着色器缓存只对应**生成它的那台电脑的显卡型号和驱动版本**：

- 每位玩家都需要在自己的电脑上预编译一次：启动器询问时选择 "Precompile first"，或者双击 `precompile.cmd`。
  22 线程约 45 分钟，8 线程约 2 小时；关闭窗口会中止，再次运行会从中断处继续。
- 从别人电脑复制来的 `_PipelineCache` 文件夹或缓存文件无法使用（即使游戏版本相同），请不要分享或下载。
- 更新显卡驱动或更换显卡后需要重新预编译（启动器会提示）。
- 游戏 1.07 和 1.05 各有自己的缓存。
- 用旧版本预编译过的，请重新运行一次预编译：这个版本会编译更多的管线变体（已编译的部分会保留）。
- 不预编译也能玩，但第一次进入新区域或出现新特效时会卡顿。
- 发布包里不包含游戏的着色器列表：第一次启动时会从你自己的游戏文件生成。

## 本版本新增

- 启动器新增 **"Up to 120 fps"** 选项（默认关闭）：游戏最高以 120 帧渲染，而不是 60 帧。开启时过场动画会播放得更快。
- 各处更快（见下表）：不需要在线程之间交换数据的计算着色器改为每个调用只运行一个 GPU 线程（拉特利亚之塔 GPU 时间约减少 10%），更多绘制和计算走表路径，以及渲染线程上的许多小优化。
- 卡顿更少：纹理流送只更新游戏改写的部分，预编译也覆盖表路径和原生管线的变体，游戏过程中会保存驱动缓存。
- 修复：GPU 内存回收此前从未执行（游戏从 GPU 提交画面），显存较小的显卡上显存占用会随游戏时间不断增长；渲染帧率高于显示器刷新率时，每次刷新都显示最新的一帧；纹理局部更新不再丢失 GPU 上该纹理的其他数据。

## 测试机上的帧率

i9-14900K + RTX 5090，2560×1440，已预编译，开启 "Up to 120 fps" 以显示性能余量：每个区域的一个位置上站立、走动、转动镜头各 10 秒的平均帧率（括号内为 1% low）。默认 60 帧上限时，约 62 以上的数值都是稳定的 60 帧。

| 世界 | 站立 | 移动 | 转动镜头 |
| --- | ---: | ---: | ---: |
| 1-1 Boletarian Palace | 58 (54) | 72 (44) | 82 (51) |
| 1-2 Boletarian Palace | 65 (57) | 72 (43) | 79 (53) |
| 1-3 Boletarian Palace | 72 (61) | 69 (27) | 79 (59) |
| 1-4 Boletarian Palace | 68 (61) | 73 (51) | 76 (23) |
| 2-1 Stonefang Tunnel | 87 (72) | 86 (53) | 93 (65) |
| 2-2 Stonefang Tunnel | 120 (95) | 119 (59) | 120 (88) |
| 3-1 Tower of Latria | 63 (54) | 60 (31) | 62 (44) |
| 3-2 Tower of Latria | 87 (70) | 76 (35) | 74 (53) |
| 4-1 Shrine of Storms | 77 (67) | 89 (35) | 87 (44) |
| 4-2 Shrine of Storms | 95 (71) | 87 (34) | 83 (29) |
| 5-1 Valley of Defilement | 85 (70) | 95 (40) | 104 (69) |
| 5-2 Valley of Defilement | 104 (84) | 92 (38) | 93 (63) |
| Nexus | 88 (70) | 81 (36) | 84 (58) |

2-2：测试位置在雾门前，几乎不绘制任何东西。
