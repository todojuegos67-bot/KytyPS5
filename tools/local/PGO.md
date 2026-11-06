# PGO

发布构建（GitHub Actions 和本地 `build-windows.cmd`）都用 profile 引导优化。不要在测帧率时编译或训练，
也不要覆盖正在运行或已发布的可执行文件。

## Windows

`build-windows.cmd` 自动带上 `-DKYTY_PGO_USE`：有 `_Build\pgo\windows\kyty.profdata` 就用它，否则用仓库里的
`tools\pgo\kyty.profdata`（GitHub 上的发布构建用这个；重新训练后把新 profile 也复制过去提交）。源码改动多了之后，
profile 里改过的函数会因哈希不符而失去 PGO，需要重新训练：

1. `python tools\local\bench-windows.py prepare`，然后 `tools\local\windows\pgo-train.ps1`：
   它在 `_Build\windows-pgogen` 构建插桩版本（`-DKYTY_PGO_GENERATE=ON`），用它跑训练场景
   （基准路线走过去站立、转镜头、走进遗迹、横移），用 live 命令 `pgo <file>` 写出 profile，
   再用 `llvm-profdata merge` 得到 `_Build\pgo\windows\kyty-<日期>.profdata`。
2. 把旧的 `kyty.profdata` 另存一份，新文件复制成 `kyty.profdata`；路径不变时 Ninja 不会重编，
   所以先 `cmake --build _Build\windows --target clean` 再 `build-windows.cmd`。
3. 用 `ab-exe.ps1` 对比旧 profile 的可执行文件（先复制到自己的目录）和新构建。

2026-10-01：9 月 27 日的 profile 换成新训练的之后，固定场景 43.83 → 45.17 fps
（ab-exe 三轮，B 组每个窗口都高于 A 组）。

2026-10-16（笔记本，16 GB 内存）：10-08 的 profile 已落后 232 个提交，在当前 HEAD 上重新训练三个场景后合并：
- 默认基准存档（序章；它有两个角色，出生点会在隧道和雾中平原之间变化，对比前先看截图）；
- `KYTY_BENCH_BASELINE=_Build\bench-baselines\user-20261001`（1-1 城门广场）；
- `world-latria`。

1-1 站立，ab-exe 两轮 ABBA（需要 `-StartupSeconds 240`：16 GB 下 1-1 要约 180 s 才到 HUD）：
渲染线程 33.60 → 32.00 ms/帧（-4.8%），fps 22.87 → 23.12（站立时帧还在等游戏侧）。
清理构建可以用 `ninja -C _Build\windows -t clean` 代替 `cmake --build ... --target clean`。

2026-10-18（台式机，RTX 5090）：6bae2db5 之后的提交让 29 个 TU 里 75 个函数失配（扫描：用 `ninja -t compdb -x` 取编译命令，
把 `-Wno-profile-instr-out-of-date` 换成 `-Wprofile-instr-out-of-date`，-O0、不带 LTO 逐个编译，看每个 TU 的
"of N functions, M have mismatched data"；前端的 CFG 哈希与优化级别无关）。热函数里加任何 if/循环/&&/||/?: 都会让它整个
失去 profile，诊断代码要放到函数外、用无条件调用。在 670eee86 上训练四个场景后合并：默认存档（序章）、`user-20261001`、
`world-latria`、`rec-m02_00_00_00-c0000_0006`（1-1 竞技场，光照循环和 fe93 那类小 dispatch 多）。重扫只剩两处，都是
按设计的：graphicsRun.cpp 的 live 命令处理（有 `KYTY_PGO_GENERATE` 才有的分支）和 redZonePatcher.cpp（加载器）。
ab-exe ABBA，120 Hz 上限（`-Vblank 120`），走 10 s 后两个 6 s 窗口：
- 1-1：渲染线程 13.52 → 12.81 ms/帧（-5.3%），fps 65.3 → 69.3（A 的第一轮带预热，去掉它 +2.7%）；
- Latria：12.47 → 12.15 ms/帧（-2.6%），fps 75.2 → 76.1。
2026-10-20（台式机，发布前）：10-18 之后的提交让 20 个 TU 里 62 个函数失配（bufferCache 6、renderDraw 6、graphicsRun 4、
textureCache 4 等）。在 ddc59a8e 上用同样四个场景重新训练（`tools\local\windows\pgo-multi.ps1`），ab-exe ABBA，120 Hz 上限：
- 1-1：渲染线程 13.71 → 13.04 ms/帧（-4.9%），fps 70.0 → 72.8；
- Latria：12.23 → 12.17 ms/帧，fps 75.8 → 76.5。

2026-10-25（台式机，v20261025 发布前）：扫描 22 个 TU 里 82 个函数失配。v20261024 的可执行文件对比当前 HEAD（同一份 10-20
profile），1-1、120 Hz 上限：fps 85.3 / 84.9，渲染线程 9.71 / 10.44 ms。在 HEAD 上用同样四个场景重新训练后，ab-exe ABBA（旧
profile 为 A）：渲染线程 10.30 → 10.23 ms/帧，但四轮里 B 的 fps 都更低（84.3、86.5、83.5、84.2 对 81.0、80.7、78.3、78.7，
平均 84.6 → 79.7）：新 profile 拖慢了渲染线程以外的线程。所以 v20261025 仍用 10-20 的 profile；新的另存为
`_Build\pgo\windows\kyty-20261025.profdata`，没有提交。

2026-11-05（台式机，v20261105 发布前）：PR #280（scout）等改动之后，在 aff2a569 上训练五个场景：原来的四个（scout 默认开），
加上 `user-20261001` 配 `KYTY_SCOUT=0`（空闲核心不够、scout 停下的电脑上，渲染线程自己求值表；issue #283）。ab-exe ABBA，
120 Hz 上限（旧的 10-21 profile 为 A）：
- Latria：渲染线程 12.61 → 11.97 ms/帧，fps 76.0 → 80.0（B 的四个窗口都高于 A 的）；
- 1-1：14.19 / 14.26 ms，fps 69.0 / 69.2（持平）；
- 1-1 加 `KYTY_SCOUT=0`，两次 ABBA 合计：14.32 / 14.55 ms，fps 69.1 / 68.1（这个位置单次启动之间相差 ±5%，在噪声内）。
@DevList1454 在 Ryzen 7 9700X + RX 9070 XT 上按同样的方法重新训练（scout 开），王城内城 +5-6%。新 profile 提交为
`tools\pgo\kyty.profdata`，旧的另存为 `_Build\pgo\windows\kyty-before-20261105.profdata`。

2026-11-06（台式机）：合并 PR #285、#287、#297-#304、#308、#309 之后，扫描 10 个 TU 里 65 个函数失配（pm4Handlers 30、redZonePatcher 17、
bufferCache 7、renderDraw 3 等）。旧的场景存档已随 `_Build` 删除，所以这次的存档全部由 1-1 竞技场的
`rec-m02_00_00_00-c0000_0006` 派生（`tools\local\demons-souls-save.py set-map`；游戏只有 `.zar` 时，先把
`cp11demonssouls/dvdroot/map/mapstudio/m0*.msb` 取到一个文件夹，用 `--game` 指过去）。`tools\local\windows\pgo-multi.ps1`
现在默认跑这五次：
- `world-prologue`（m08_00_00_00 c0000_0000，教程的隧道）；
- `rec-m02_00_00_00-c0000_0001`（1-1 城门广场，代替 `user-20261001`）；
- `world-latria`（m04_00_00_00 c0000_0000）；
- `rec-m02_00_00_00-c0000_0006`；
- `rec-m02_00_00_00-c0000_0001` 配 `KYTY_SCOUT=0`。

ab-exe ABBA，120 Hz 上限，源码相同（11-05 的 profile 为 A）：
- 1-1：渲染线程 14.68 → 13.65 ms/帧，fps 65.95 → 70.03（4 个窗口）；
- 1-1 加 `KYTY_SCOUT=0`：14.50 → 14.36 ms，fps 68.35 → 69.33（8 个窗口）；
- Latria：12.69 → 12.48 ms，fps 76.91 → 75.98（8 个窗口；这里帧率不只取决于渲染线程）。

重扫只剩 redZonePatcher.cpp（加载器）的 17 个。新 profile 提交为 `tools\pgo\kyty.profdata`，旧的另存为
`_Build\pgo\windows\kyty-before-20261106.profdata`。
