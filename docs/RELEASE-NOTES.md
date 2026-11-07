## New in this build

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

## New in v20261106

Your precompiled shaders stay valid: the shader translation is the same as in v20261105.

- **The game's own stereo sound** (#294, credit @robbely): music and ambience no longer sound nearly mono, and speech is
  clearer. The emulator used to fold the game's 3D sound down to its first order, which made left and right almost the
  same. Now the game mixes for stereo itself. Left and right differ far more (correlation at the Nexus 0.74 -> 0.27, in
  Latria 0.93 -> 0.42), and speech stands out about 7 dB more from the ambience. `-Set
  KYTY_GAME_CONVARS=sound_enableAmbisonics=true` brings back the old sound.
- **F11 with AMD FSR frame generation no longer loses the graphics device** (#174, credit @ft1704): when AMD's runtime
  made or destroyed its swapchain, or turned frame generation on or off, it waited for the whole GPU while the emulator
  kept submitting work, which Vulkan does not allow. The emulator now holds its queues during that wait. With AMD's
  runtime doing both the upscaling and the frame generation, F11 lost the device about 1 time in 25 on our RTX 5090 (AMD's
  FSR 3.1 there); now 135 presses went through without one, and 30 with DLSS and DLSS frame generation. `-Set
  KYTY_FSR_RUNTIME=0` is no longer needed for this. FSR 4 on a Radeon is untested, so please tell us if it still happens.
- **A draw that could run without its index buffer**: in rare cases a draw after a non-indexed draw in a new command
  buffer skipped binding its indices, which is undefined on any GPU. It went unnoticed since v20261007; Vulkan's
  validation layer found it on a tour of the worlds.
- **Optimized again after the merged pull requests**: the profile-guided optimization was trained again. 1-1 on our RTX
  5090: 6% more frames at the 120 Hz cap (render thread 14.7 -> 13.7 ms a frame); Latria about the same.
- **AMD FSR frame generation set-up corrected**: Vulkan's validation layer found four more mistakes in how the emulator
  prepared FSR's images and its HUD pass (image usage, a depth barrier, an access mask, a shader feature that was not
  enabled). The picture is the same here; other drivers may be less forgiving.
- **Linux: no more crash at boot with NVIDIA 620.70.06** (PR #308, credit @ponhei9502): the driver's pipeline key call
  overwrote registers it should keep. `KYTY_PIPELINE_KEY_THUNK=0` turns the workaround off.
- **When a driver refuses a shader** (#305, credit @FKF-GR): a precompile leaves that pipeline out, goes on and says how
  many it left out, instead of splitting into hundreds of retries. The log names the driver's answer and the shaders,
  and says when a GPU has no fragment shader barycentrics, which nearly all of the game's pixel shaders need. PR #281's
  empty fragment shader is now NVIDIA only, the one place it was measured to help (`-Set KYTY_EMPTY_FS=1` for any GPU).
- **Fixes from pull requests** (credit @CarpetHead, @ponhei9502): the GPU's fault buffer is read and cleared word by word
  in one step (PR #297), guarded writes reach the scout thread as such (PR #298), the lane-local proof refuses a DPP row
  mask that treats the wave's halves apart (PR #299), and the write guard works with the Linux userfaultfd write watch
  (PR #309, `KYTY_WRITE_GUARD_UFFD=1`).
- **For testers, off by default** (credit @CarpetHead, @ponhei9502): `KYTY_PM4_REGFILE` (PR #285), `KYTY_UPLOAD_LANES`
  (PR #287), `KYTY_EAGER_FEEDBACK`, `KYTY_FEEDBACK_BATCH`, `KYTY_TABLE_SERIALS`, `KYTY_TABLE_MEMO` (PRs #300-#303) and
  `KYTY_LANE_LOCAL_WIDE` (PR #304). They made no difference to the frame rate on our PCs, or moved it both ways.
- **Linux package README**: AMD's FSR SDK DLLs are Windows files, so FSR 4 is not available on Linux (#307).
- **Thanks** to @CarpetHead and @ponhei9502 for their pull requests, and to @robbely, @ft1704, @FKF-GR and
  @blubberbozz-wq for their reports and tests.

## 本版本的变化

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

## v20261106 的变化

已有的预编译着色器继续有效：着色器的翻译与 v20261105 相同。

- **改用游戏自己的立体声**（#294，感谢 @robbely）：音乐和环境音不再接近单声道，人声也更清楚。模拟器之前把游戏的 3D 声音只解到
  一阶，左右声道几乎一样；现在由游戏自己混成立体声。左右声道的差别大了很多（相关度：Nexus 0.74 -> 0.27，Latria 0.93 -> 0.42），
  人声比环境音突出约 7 dB。`-Set KYTY_GAME_CONVARS=sound_enableAmbisonics=true` 可恢复原来的声音。
- **AMD FSR 插帧时按 F11 不再丢失显卡设备**（#174，感谢 @ft1704）：AMD 运行库创建或销毁交换链、开关插帧时，会等待整个 GPU，
  而模拟器还在继续提交工作，这是 Vulkan 不允许的。现在等待期间模拟器会让出所有队列。超分和插帧都由 AMD 运行库完成时，
  我们的 RTX 5090 上（那里是 AMD 的 FSR 3.1）大约每 25 次 F11 丢失一次设备；现在按了 135 次都没有发生，DLSS 加 DLSS 插帧也按了
  30 次没有问题。不再需要为此加 `-Set KYTY_FSR_RUNTIME=0`。Radeon 上的 FSR 4 尚未测试，如果还会发生请告诉我们。
- **绘制可能缺少索引缓冲**：少数情况下，新命令缓冲里紧跟在非索引绘制之后的绘制会跳过索引绑定，这在任何 GPU 上都是未定义行为。
  这个问题从 v20261007 起就存在，这次由 Vulkan 的校验层在各世界巡游时发现。
- **合并 PR 后重新做了编译优化**：程序的 profile 引导优化（PGO）重新训练。我们的 RTX 5090 在 1-1、120 Hz 上限下多 6% 帧数
  （渲染线程每帧 14.7 -> 13.7 ms），Latria 基本持平。
- **修正 AMD FSR 插帧的设置**：Vulkan 校验层还发现模拟器准备 FSR 图像和 HUD 处理时的四处错误（图像用途、一处深度屏障、一处访问掩码、
  一项没有启用的着色器功能）。我们这里画面没有变化，其他驱动可能没这么宽容。
- **Linux：NVIDIA 620.70.06 不再启动时崩溃**（PR #308，感谢 @ponhei9502）：驱动的管线密钥调用会改写本应保留的寄存器。
  `KYTY_PIPELINE_KEY_THUNK=0` 可关闭这项规避。
- **驱动拒绝某个着色器时**（#305，感谢 @FKF-GR）：预编译会跳过这条管线继续进行，并报告跳过了多少条，不再拆分成几百次重试。
  日志会写出驱动的返回结果和对应的着色器；如果显卡不支持 fragment shader barycentrics（游戏几乎所有像素着色器都需要），日志也会写明。
  PR #281 的空像素着色器现在只用于 NVIDIA，这是唯一测到收益的地方（`-Set KYTY_EMPTY_FS=1` 可用于任何显卡）。
- **来自 PR 的修复**（感谢 @CarpetHead、@ponhei9502）：GPU 故障缓冲的每个字一步完成读取和清除（PR #297）；受保护的写入如实告知
  scout 线程（PR #298）；lane-local 证明拒绝把 wave 两半分开处理的 DPP 行掩码（PR #299）；写保护可与 Linux 的 userfaultfd 写监视
  一起使用（PR #309，`KYTY_WRITE_GUARD_UFFD=1`）。
- **供测试用，默认关闭**（感谢 @CarpetHead、@ponhei9502）：`KYTY_PM4_REGFILE`（PR #285）、`KYTY_UPLOAD_LANES`（PR #287）、
  `KYTY_EAGER_FEEDBACK`、`KYTY_FEEDBACK_BATCH`、`KYTY_TABLE_SERIALS`、`KYTY_TABLE_MEMO`（PR #300-#303）和
  `KYTY_LANE_LOCAL_WIDE`（PR #304）。它们在我们的电脑上对帧率没有影响，或者有升有降。
- **Linux 包的 README**：AMD FSR SDK 的 DLL 是 Windows 文件，Linux 上没有 FSR 4（#307）。
- **感谢** @CarpetHead 和 @ponhei9502 的 PR，以及 @robbely、@ft1704、@FKF-GR 和 @blubberbozz-wq 的反馈与测试。
