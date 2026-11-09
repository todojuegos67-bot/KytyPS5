## New in this build

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

## New in v20261108

Your precompiled shaders stay valid: the shader translation is the same as in v20261106.

- **A crash while a world loads fixed**: the game's physics sometimes stepped a job whose object was not there, and the
  emulator closed with an error at `eboot+0x1ab6047`. We saw it at least five times since October 11, replaying a
  player's route at 1-1 and in long runs of world changes. Such a job now leaves its step the way the game's own code
  does when it has nothing to work on.
- **8 GB graphics cards no longer slow down after many world changes, on PCs with more than 16 GB of RAM**: since
  v20261027 the game has streamed its textures without the largest levels that cards under 12 GB leave out, on every
  PC, to save about 1 GB of RAM. In long play with many loads this made the game's GPU buffers grow from about 2 GB to
  3-3.4 GB, past an 8 GB card's budget. In our runs (six worlds, 48 loads an hour) Boletaria fell to 48 fps against 53,
  and standing still to 49 against 60. With more than 16 GB of RAM the game now streams whole textures again, which
  costs only RAM. PCs with 16 GB keep the saving, so there the frame rate can still drop by about 10% after 20 or more
  world changes in one session; restarting the game brings it back. `-Set KYTY_TEXTURE_REDUCE_GAME=0` streams whole
  textures on any PC, `=1` leaves those levels out on any PC.
- **F11 with AMD FSR frame generation no longer freezes the picture in menus and loads** (#174, credit @ft1704): with
  FidelityFX SDK 1.1.4's DLL (`amd_fidelityfx_vk.dll`), while no frames were being generated (menus, loads, the start),
  AMD's swapchain silently dropped a frame when the window had just changed (F11, a resize), and the next wait for it
  never ended. The emulator now makes the swapchain again itself in that case. On our RTX 5090: 144 presses of F11 at
  the title, 12 resizes, F11 in the world and during six loads, all fine. FSR SDK 2's DLLs were not affected.
- **A graphics pipeline the driver refuses no longer ends the game** (#305, credit @FKF-GR): an RX Vega 56 with AMD
  26.5.2 turned down about 6,500 of the game's pipelines (it built 73,571), made with 71 of its pixel shaders, and the
  game stopped at the first one it needed. Now the draws that need a refused pipeline are left out and the game goes
  on, and the log names the refused pipelines. What goes missing from the picture depends on the card: on our RTX
  5090, with 30 such pixel shaders refused on purpose, 1-1 ran at 60 fps and looked right apart from slightly cooler
  light. A refused compute pipeline still ends the game.
- **Stereo with Audio output on Headphones** (#294, credit @robbely): the game hands its stereo mix to the system on two
  separate channels, left and right, and the emulator played both in the centre, so the music sounded nearly mono. Each
  now plays on its own side (how alike left and right are at the Nexus, 1 = mono: 0.86 -> 0.53). TV output is
  unchanged.
- **Linux launcher: keyboard and mouse prompts** (PR #311, credit @Bipboop-beep): the Windows launcher's option from
  v20261105, now on Linux too: button prompts as keyboard keys and mouse buttons, and the keyboard controls editor, with
  the same files and settings.
- **Long runs**: about 17 hours of world changes (six worlds, 48 loads an hour) with 8, 10 and 32 GB of video memory,
  DLSS and FSR frame generation found no memory leak and no hang. The crash and the 8 GB slowdown above came out of them.
- **Thanks** to @Bipboop-beep for the pull request, and to @robbely, @ft1704 and @FKF-GR for their reports and tests.

## 本版本的变化

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

## v20261108 的变化

已有的预编译着色器继续有效：着色器的翻译与 v20261106 相同。

- **修复加载世界时的崩溃**：游戏的物理系统有时会处理一个对象并不存在的任务，模拟器随即报错退出（`eboot+0x1ab6047`）。
  从 10 月 11 日起我们至少遇到过 5 次，包括重放玩家在 1-1 的路线和长时间连续换图。现在这种任务会按游戏自己代码里
  “无事可做”时的方式结束这一步。
- **8 GB 显卡多次换图后不再变慢（内存大于 16 GB 的电脑）**：从 v20261027 起，为了省大约 1 GB 内存，游戏在所有电脑上都
  不再流送 12 GB 以下显卡会丢掉的最大纹理层级。长时间、多次加载后，这会让游戏的 GPU 缓冲从约 2 GB 涨到 3-3.4 GB，
  超出 8 GB 显卡的预算。我们的测试（6 个世界轮换，每小时 48 次加载）中，Boletaria 从 53 fps 掉到 48，站着不动从 60 掉到
  49。内存大于 16 GB 时，游戏现在重新完整流送纹理，代价只是多占内存。16 GB 内存的电脑保留这项节省，所以一次游戏中换图
  20 次以上后，帧率仍可能下降约 10%，重启游戏即可恢复。`-Set KYTY_TEXTURE_REDUCE_GAME=0` 在任何电脑上都完整流送，
  `=1` 在任何电脑上都不流送这些层级。
- **AMD FSR 插帧时，在菜单和加载中按 F11 不再卡住画面**（#174，感谢 @ft1704）：使用 FidelityFX SDK 1.1.4 的 DLL
  （`amd_fidelityfx_vk.dll`）时，在不插帧的阶段（菜单、加载、刚启动），如果窗口刚刚改变（F11、调整大小），AMD 的交换链
  会悄悄丢掉一帧，之后等待这一帧的操作永远等不到，画面就卡住了。现在遇到这种情况，模拟器会自己重建交换链。我们的
  RTX 5090 上：标题画面按 F11 144 次、调整窗口大小 12 次，在世界里和 6 次加载中按 F11，全部正常。FSR SDK 2 的 DLL 没有
  这个问题。
- **驱动拒绝某条图形管线时游戏不再停止**（#305，感谢 @FKF-GR）：一块 RX Vega 56（AMD 26.5.2）拒绝了游戏约 6,500 条管线
  （建成 73,571 条），涉及 71 个像素着色器，游戏会停在第一条要用到的被拒管线上。现在需要被拒管线的绘制会被跳过，游戏继续
  运行，日志会写出被拒的管线。画面会少掉什么取决于显卡：在我们的 RTX 5090 上故意拒绝 30 个这类像素着色器，1-1 仍是
  60 fps，画面除了光线略偏冷以外都正常。计算管线被拒时游戏仍会停止。
- **音频输出选“耳机”时恢复立体声**（#294，感谢 @robbely）：游戏会把立体声混音分成左右两个通道交给系统，模拟器之前把
  两个通道都放在正中，音乐听起来几乎是单声道。现在左右各自在自己那一侧播放（Nexus 处左右声道的相似度，1 = 单声道：
  0.86 -> 0.53）。电视输出不变。
- **Linux 启动器：键盘和鼠标按键提示**（PR #311，感谢 @Bipboop-beep）：v20261105 加入 Windows 启动器的选项，现在 Linux
  上也有了：按键提示可以显示为键盘按键和鼠标按钮，还有键盘操作设置，使用相同的文件和设置。
- **长时间测试**：在 8、10 和 32 GB 显存、DLSS 和 FSR 插帧等设置下，共约 17 小时连续换图（6 个世界，每小时 48 次加载），
  没有发现内存泄漏或卡死。上面的崩溃和 8 GB 变慢就是这些测试发现的。
- **感谢** @Bipboop-beep 的 PR，以及 @robbely、@ft1704 和 @FKF-GR 的反馈与测试。
