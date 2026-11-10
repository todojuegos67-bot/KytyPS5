## New in this build

Your precompiled shaders stay valid: the shader translation is the same as in v20261109, except on AMD's older driver
for Radeon RX Vega and Polaris cards (the first item below).

- **RX Vega / Polaris with AMD's older driver: the black title screen** (#305, credit @FKF-GR): that driver ("AMD
  proprietary shader compiler" in the log's GPU line) refused every pipeline of 173 of the game's pixel shaders, the
  ones that read the GPU's barycentric coordinates without per-vertex values, the picture's post-processing among them.
  On that driver these shaders now get the same coordinates through AMD's own extension
  (VK_AMD_shader_explicit_vertex_parameter), which it lists. Other graphics cards translate them as before. Precompile
  again after updating on such a card. We could not run AMD's older driver here (the shaders pass validation and AMD's
  own compiler for Vega takes them), so please tell us in #305 whether the title shows.
- **New, experimental: parallel translation** (PR #293, credit @CarpetHead): with `-Set KYTY_SPECULATE=4` (run.cmd, or
  the launcher's Advanced > Extra options, "Command-line options"), worker threads translate the game's GPU work ahead
  of the render thread, which then only checks and commits it. Where the render thread limits the frame rate it is
  faster: +6% to +16% over the worlds in the author's test at 120 fps (Core i9-12900K, RTX 5090). It needs 28 GB of RAM
  or more (about 3.6 GB more than without it) and at least 9 logical CPUs on the performance cores; on other PCs it
  stays off and the log says why. Off by default; without it the emulator runs as before. Tested on Windows only.
- **A little faster**: the build's profile-guided optimization was trained again on this code. 1-1 at 120 fps on an
  RTX 5090: 70.3 -> 72.5 fps (the render thread 13.9 -> 13.4 ms a frame); Latria and the arena unchanged.
- **Thanks** to @CarpetHead for the pull request and the many rounds of measurements, and to @FKF-GR for the logs.

## New in v20261109

Your precompiled shaders stay valid: the shader translation is the same as in v20261108.

- **Long sessions on 8 GB graphics cards no longer slow down after many area loads, on 16 GB PCs too**: the game's
  page allocator frees blocks of memory but keeps them mapped and hands them to the next part of the game that asks,
  and the emulator kept the GPU buffers it had made over such a block for good. Over many world loads they grew from
  about 2 GB to over 3 GB, past an 8 GB card's budget, and Boletaria fell from about 60 to 48-52 fps. Now the buffers over a
  block the game frees go. In our runs on an 8 GB card with 16 GB of RAM (simulated; 42 world loads in 50 minutes) the
  buffers stayed at 1.5-1.9 GB and every world kept 60 fps; on an RTX 5090 they are about 0.4 GB smaller, at the same
  frame rate. What v20261108's notes said about 16 GB PCs (about 10% slower after 20 or more world changes) no longer
  applies. `-Set KYTY_ENGINE_FREE_BUFFERS=0` turns it off.
- **A pipeline the driver refuses gets a second try** (#305, credit @FKF-GR): an RX Vega 56 with AMD 26.5.2 refused some
  of the game's graphics pipelines. Since v20261108 their draws were left out, and at the title these included the
  picture's post-processing, so the screen stayed black. Now a refused pipeline is made again without the driver's
  optimizations before its draws are left out (a compute pipeline: before the game ends), and the log counts the
  outcomes at the end ("Pipelines the driver refused in play"). Also, on NVIDIA and on Linux with RADV, where pipelines
  are first made unoptimized for speed, a refused optimized build no longer replaces the pipeline that works.
- **An option against square holes in overlapping decals** (#315, credit @robbely, @zezoman): small square holes came
  and went in the Nexus candles' wax and at the bottom of some castle walls in 1-3. Those are decals that each read the
  scene's material data under them and write it back blended in. The game puts no wait between them: the PS5 runs them
  nearly in order, a PC graphics card runs overlapping ones at the same time, and pixels get lost in 2x2 blocks.
  `-Set KYTY_DRAW_READ_AFTER_WRITE=1` (launcher: Advanced > Extra options, "Command-line options") makes each such decal
  wait for the one before it, and the holes go. It is off by default because the waiting costs frame rate where there
  are hundreds of decals: at Boletaria's start without the 60 fps cap an RTX 5090 went from 73 to 61 fps (with the cap
  it held 60).
- **README** (#240, credit @blazouf): how to use a cheat file (`_Patches\PPSA01341.json`; `"enabled": false` leaves a
  cheat out; the file must be for the game's version), cheats on the Steam Deck (with a wrapper too), packing a `.zar`
  stops at paths over 260 characters (put the game folder near the drive root, or turn on Windows' long paths), and
  the black flicker on 16 GB PCs with little RAM left (#159, #319, credit @YC4R0): close the browser and other large
  programs; `-Set KYTY_HOT_READS_MIN_AVAIL_MB=0` keeps the flicker away, at the price of weaker blue-screen protection
  when the RAM runs completely full.
- **Long runs**: about 4.5 hours of world changes on this code (8 GB of video memory with and without 16 GB of RAM
  simulated, and the RTX 5090 pressing F11 at every stop): no memory leak, no hang, no crash.
- **Thanks** to @FKF-GR, @robbely, @zezoman, @blazouf and @YC4R0 for their reports and tests.

## 本版本的变化

已有的预编译着色器继续有效：着色器的翻译与 v20261109 相同，只有 Radeon RX Vega、Polaris 显卡使用 AMD 旧驱动时例外
（见第一条）。

- **RX Vega / Polaris 配 AMD 旧驱动：标题画面黑屏**（#305，感谢 @FKF-GR）：这个驱动（日志 GPU 一行写着“AMD
  proprietary shader compiler”）会拒绝游戏中 173 个像素着色器的全部管线，也就是读取 GPU 重心坐标、但不读逐顶点数据的
  那些，画面的后处理也在其中。现在在这个驱动上，这些着色器改用 AMD 自己的扩展（VK_AMD_shader_explicit_vertex_parameter，
  驱动有列出）取得同样的坐标，其他显卡的翻译不变。这类显卡更新后请重新预编译。我们这里没法运行 AMD 的旧驱动（这些着色器
  通过了校验，AMD 自己的 Vega 编译器也能编译），所以请在 #305 告诉我们标题画面是否正常显示。
- **新增实验功能：并行翻译**（PR #293，感谢 @CarpetHead）：加上 `-Set KYTY_SPECULATE=4`（run.cmd，或启动器的高级 >
  附加参数“命令行参数”）后，工作线程会提前翻译游戏的 GPU 工作，渲染线程只需检查并提交。在渲染线程限制帧率的地方会更快：
  作者在 120 fps 下逐个世界测试，提升 6% 到 16%（Core i9-12900K、RTX 5090）。需要 28 GB 以上内存（比不开多用约 3.6 GB），
  以及性能核上至少 9 个逻辑处理器；不满足的电脑上它不会开启，日志会说明原因。默认关闭，不开时模拟器和以前一样。目前只在 Windows 上测试过。
- **稍快一些**：重新训练了这个版本的 PGO（按运行情况优化）。RTX 5090 在 1-1、120 fps 下：70.3 -> 72.5 fps（渲染线程
  每帧 13.9 -> 13.4 ms）；Latria 和竞技场不变。
- **感谢** @CarpetHead 的 PR 和多轮测量，以及 @FKF-GR 提供的日志。

## v20261109 的变化

已有的预编译着色器继续有效：着色器的翻译与 v20261108 相同。

- **8 GB 显卡长时间游玩、多次读图后不再变慢（16 GB 内存的电脑也一样）**：游戏的页分配器释放内存块后并不解除映射，
  而是交给下一个申请内存的模块使用，模拟器在这种块上建立的 GPU 缓冲却一直保留。多次加载世界后，缓冲从约 2 GB 涨到
  3 GB 以上，超出 8 GB 显卡的预算，Boletaria 从 60 fps 掉到 48-52。现在游戏释放一个块时，块上的缓冲随之释放。在 8 GB
  显卡 + 16 GB 内存（模拟；50 分钟加载世界 42 次）的测试中，缓冲保持在 1.5-1.9 GB，每个世界都保持 60 fps；RTX 5090 上
  缓冲约少 0.4 GB，帧率不变。v20261108 说明里关于 16 GB 内存电脑“换图 20 次以上后帧率仍可能下降约 10%”的情况不再存在。
  `-Set KYTY_ENGINE_FREE_BUFFERS=0` 可以关闭。
- **驱动拒绝的管线会再试一次**（#305，感谢 @FKF-GR）：RX Vega 56（AMD 26.5.2）的驱动拒绝了游戏的部分图形管线。从
  v20261108 起这些管线的绘制会被跳过，而标题画面里被跳过的包括画面的后处理，所以一直黑屏。现在被拒绝的管线会先关掉
  驱动的优化重新创建一次，仍然失败才跳过它的绘制（计算管线：在结束游戏之前），日志结尾会统计结果（“Pipelines the
  driver refused in play”）。另外，在 NVIDIA 和 Linux 的 RADV 上（管线先以不优化的方式快速创建），驱动拒绝优化版本时
  不再用它替换掉已经能用的管线。
- **贴花重叠处的方块洞：新增一个选项**（#315，感谢 @robbely、@zezoman）：Nexus 蜡烛的蜡油和 1-3 部分城墙底部会出现
  时有时无的小方块洞。这些是贴花：每个贴花读出底下的场景材质数据，混合后再写回。游戏在贴花之间没有等待：PS5 基本按
  顺序执行，PC 显卡会同时执行重叠的贴花，像素就以 2x2 的块丢失。`-Set KYTY_DRAW_READ_AFTER_WRITE=1`（启动器：高级 >
  附加参数，“命令行参数”）让每个这样的贴花等前一个完成，方块洞就会消失。默认关闭，因为在有几百个贴花的地方，等待会
  降低帧率：Boletaria 起点不锁 60 帧时，RTX 5090 从 73 fps 降到 61（锁 60 帧时仍是 60）。
- **README**（#240，感谢 @blazouf）：金手指文件的用法（`_Patches\PPSA01341.json`；`"enabled": false` 的条目不生效；文件
  必须与游戏版本对应）、在 Steam Deck 上使用金手指（也可以配合包装程序）、打包 `.zar` 时路径超过 260 个字符会失败（把
  游戏文件夹放到靠近盘符根目录的位置，或开启 Windows 的长路径），以及 16 GB 内存的电脑剩余内存不足时的黑色闪烁
  （#159、#319，感谢 @YC4R0）：关闭浏览器等占内存的程序；`-Set KYTY_HOT_READS_MIN_AVAIL_MB=0` 可以消除闪烁，代价是内存
  完全用满时防蓝屏的保护变弱。
- **长时间测试**：在这版代码上进行了约 4.5 小时的连续换图（8 GB 显存，模拟 16 GB 内存和不限内存各一组，以及每个
  停留点都按 F11 的 RTX 5090）：没有内存泄漏，没有卡死，没有崩溃。
- **感谢** @FKF-GR、@robbely、@zezoman、@blazouf 和 @YC4R0 的反馈与测试。
