## New in this build

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

## New in v20261105

Your precompiled shaders stay valid: the shader translation is the same as in v20261103.

- **Faster on most PCs: a scout thread** (PR #280, credit @DevList1454 and @opcod3; fixes in PR #288 by @ponhei9502):
  a helper thread reads the game's graphics commands ahead of the render thread and prepares the tables its draws read,
  which the render thread checks and uses. At 1-1 on our RTX 5090: about 5% more frames (render thread 14.1 -> 13.3 ms a
  frame); the Boletarian Palace's Inner Ward on an RX 9070 XT +7-9% (@DevList1454); Linux with an RX 7900 XTX +11%
  (@opcod3). It takes about one more processor core, and on PCs without idle cores it stands down by itself. Off in the
  Steam Deck build until it is measured there. `-Set KYTY_SCOUT=0` turns it off.
- **Optimized again for the code as it runs now** (#283, credit @DevList1454): the profile-guided optimization of the
  build was trained again, with the scout. Latria on our RTX 5090: 5% more frames (render thread 12.6 -> 12.0 ms a
  frame), 1-1 the same; @DevList1454 measured +5-6% at the Inner Ward on a Ryzen 7 9700X.
- **NVIDIA: no GPU drain between the game's stencil passes** (PR #281, credit @CarpetHead): pipelines without a pixel
  shader get an empty one, so the GPU no longer drains between such draws; 0.57 ms less GPU time a frame at the stone
  arch. `-Set KYTY_EMPTY_FS=0` turns it off.
- **Fewer stutters on 8 GB graphics cards whose memory budget drops in play** (#276): below 12 GB, the texture
  collector now follows the budget when the driver lowers it during play (an 8 GB Radeon RX 6650 XT went from 7378 to
  about 5570 MiB in the world) instead of keeping the start-up figure.
- **The launcher no longer says "Shaders not precompiled yet" after a precompile** (#290): with graphics drivers
  without pipeline binaries the precompile keeps the shaders in a `.blobs` file, which the Windows launcher did not
  count. The game used them all along.
- **Keyboard and mouse prompts** (PR #277, credit @miroslavdanutalexandru-alt): the launcher's Game page can show the
  game's button prompts as keyboard keys and mouse buttons (English), matching the default keyboard controls, and
  "Keyboard controls" changes the keys and the mouse sensitivity. Off by default; it needs the extracted 1.07 game
  folder (not a `.zar`) and keeps a backup of the game files it changes.
- **For testers, off by default**: `-Set KYTY_WRITE_GUARD=1` (PR #282, credit @CarpetHead) lets the render thread's
  writes to game memory go ahead of older uploads (+1-2% where the game reads back a lot). It changes page protections
  often, the kind of churn behind the MEMORY_MANAGEMENT blue screens on PCs short of RAM, so use it only with plenty of
  RAM (on Linux it stays off with `KYTY_UFFD_WP`, #289). `-Set KYTY_TEXTURE_HOST_MODE=pd` (PR #284, credit
  @ponhei9502): on cards under 12 GB, textures past half the budget prefer video memory (it helps on Linux; NVIDIA's
  Windows driver keeps them there already).
- **FSR 4 needs a current AMD driver** (#271): with Adrenalin 26.9.2 an RX 9070 showed a black 3D picture with FSR 4
  upscaling and black generated frames with FSR 4 frame generation; a driver update fixed both.
- **For reports**: the log names the graphics driver and its version next to the GPU in use.
- **Thanks** to @DevList1454, @opcod3, @CarpetHead, @ponhei9502 and @miroslavdanutalexandru-alt for their pull requests,
  reviews and measurements, and to @STALKERSK, @yura1234 and @robbely for their reports and tests.

## 本版本的变化

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

## v20261105 的变化

已有的预编译着色器继续有效：着色器的翻译与 v20261103 相同。

- **大多数电脑上更快：scout 线程**（PR #280，感谢 @DevList1454 和 @opcod3；PR #288 的修复来自 @ponhei9502）：一个辅助线程
  先于渲染线程读取游戏的图形命令，提前准备好绘制要读取的表，渲染线程检查后直接使用。我们的 RTX 5090 在 1-1 约多 5% 帧数
  （渲染线程每帧 14.1 -> 13.3 ms）；RX 9070 XT 在波雷塔利亚王城内城快 7-9%（@DevList1454）；Linux 加 RX 7900 XTX 快 11%
  （@opcod3）。它大约多占一个处理器核心，没有空闲核心的电脑上会自动停下。Steam Deck 版在测量之前默认关闭。
  `-Set KYTY_SCOUT=0` 关闭。
- **按现在的代码重新做了编译优化**（#283，感谢 @DevList1454）：程序的 profile 引导优化（PGO）带着 scout 重新训练。我们的
  RTX 5090 在 Latria 快 5%（渲染线程每帧 12.6 -> 12.0 ms），1-1 持平；@DevList1454 在 Ryzen 7 9700X 上测得王城内城
  快 5-6%。
- **NVIDIA：游戏的模板阴影绘制之间不再让 GPU 排空**（PR #281，感谢 @CarpetHead）：没有像素着色器的管线加上一个空的像素
  着色器，GPU 不再在这类绘制之间排空；石拱处每帧少 0.57 ms 的 GPU 时间。`-Set KYTY_EMPTY_FS=0` 关闭。
- **游戏中显存预算下降的 8 GB 显卡卡顿更少**（#276）：显存低于 12 GB 时，驱动在游戏中调低预算后，贴图回收会跟着调整
  （一张 8 GB 的 Radeon RX 6650 XT 进入世界后从 7378 降到约 5570 MiB），而不是一直按启动时的数值。
- **预编译完成后，启动器不再提示「着色器尚未预编译」**（#290）：驱动不支持 pipeline binary 时，预编译把着色器存成
  `.blobs` 文件，Windows 启动器之前没有把它算进去。游戏本身一直在使用它们。
- **「键盘和鼠标按键提示」**（PR #277，感谢 @miroslavdanutalexandru-alt）：启动器的「游戏」页可以把游戏里的手柄按键提示
  换成键盘按键和鼠标按钮的图标（英文），与默认键位一致；「键盘控制」可以修改键位和鼠标灵敏度。默认关闭；需要解压后的
  1.07 游戏文件夹（不支持 `.zar`），并会备份它修改的游戏文件。
- **供测试用，默认关闭**：`-Set KYTY_WRITE_GUARD=1`（PR #282，感谢 @CarpetHead）让渲染线程写入游戏内存时不必等待之前的
  上传（游戏大量读回数据的地方快 1-2%）。它会频繁更改页面保护，正是内存不足的电脑上 MEMORY_MANAGEMENT 蓝屏背后的那类操作，
  所以只在内存充足时使用（Linux 上与 `KYTY_UFFD_WP` 同时开启时会自动关闭，#289）。`-Set KYTY_TEXTURE_HOST_MODE=pd`
  （PR #284，感谢 @ponhei9502）：显存低于 12 GB 时，超过一半预算后的贴图优先放在显存（对 Linux 有帮助；NVIDIA 的 Windows
  驱动本来就把它们放在显存）。
- **FSR 4 需要较新的 AMD 驱动**（#271）：使用 Adrenalin 26.9.2 时，RX 9070 开 FSR 4 超分画面是黑的，开 FSR 4 插帧时生成的帧
  是黑的；更新驱动后两者都正常了。
- **便于反馈问题**：日志会在所用显卡旁写出显卡驱动及其版本。
- **感谢** @DevList1454、@opcod3、@CarpetHead、@ponhei9502 和 @miroslavdanutalexandru-alt 的 PR、审阅与测量，以及
  @STALKERSK、@yura1234 和 @robbely 的反馈与测试。
