## New in this build

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

## New in v20261103

Your precompiled shaders stay valid: the shaders are the same as in v20261102.

- **FSR 4 frame generation: F11 and window resizes no longer close the game** (#270): with AMD's FSR SDK 2.4 DLLs,
  switching to or from full screen, or changing the window's size, while frame generation was on ended with "FSR frame
  generation swapchain failed". AMD's newer runtime leaves the old swapchain for the program to destroy, and the
  emulator didn't. The performance overlay now also says "FSR 4 Frame Generation" when AMD's runtime generates with
  FSR 4. One case is left: with AMD's runtime doing both the upscaling and the frame generation, about 1 F11 in 25 still
  lost the graphics device on our RTX 5090 (AMD's FSR 3.1 there; FSR 4 on a Radeon is untested). If that happens to
  you, `-Set KYTY_FSR_RUNTIME=0` keeps the frame generation and upscales with the built-in FSR 3, which never did here.
- **The launchers call the FSR upscaler "AMD FSR"**: with the SDK 2.4 DLLs it is FSR 4 on Radeon RX 7000 and 9000
  cards and FSR 3.1 on others; without them the built-in FSR 3. The log line `Super resolution: AMD FSR ...` says which
  one runs.
- **A rare freeze at start-up** (PR #269, credit @CarpetHead): the GPU thread could wait for a flip that only it could
  prepare. It showed most with the start-up logos skipped.
- **The game's threads wait less for the GPU's data** (PR #273, credit @CarpetHead): their requests are now also served
  while the render thread waits for its uploads. Up to 2-4% where the game reads a lot back (the Boletarian Palace's
  Inner Ward on an RX 9070 XT, measured by @DevList1454), the same elsewhere; `-Set KYTY_SERVE_IN_WAITS=0` turns it off.
- **For reports about AMD FSR** (#271, #276): the log now says how much video memory AMD's runtime takes for frame
  generation, its swapchain and the upscaler, and what the FSR upscaler is given.
- **Tip for 8 GB Radeon RX 6000 cards with frame generation** (#276): the SDK 2.4 frame generation took about 1 GB more
  video memory there than the older `amd_fidelityfx_vk.dll`, too much at 1080p with FSR 4. Until that is settled, use
  the older DLL alone for frame generation, or add `-Set KYTY_FSR_RUNTIME=0` (the built-in FSR 3 upscales, lighter on
  those cards).
- **Thanks** to @CarpetHead for PRs #269, #273 and #274 (a flicker check for testers), to @DevList1454 for the review
  and measurements, and to @kocher, @dimitrios1977ZXZ, @yura1234, @STALKERSK, @davide20204 and @Vyncas for their
  reports and tests.

## 本版本的变化

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

## v20261103 的变化

已有的预编译着色器继续有效：着色器与 v20261102 相同。

- **FSR 4 插帧：按 F11 或改变窗口大小不再导致游戏退出**（#270）：使用 AMD FSR SDK 2.4 的 DLL 时，在插帧开启的状态下切换全屏、
  或改变窗口大小，会以 "FSR frame generation swapchain failed" 退出。AMD 的新运行库把旧交换链留给程序自己销毁，而模拟器之前
  没有这样做。另外，AMD 运行库用 FSR 4 插帧时，性能浮层现在显示 "FSR 4 Frame Generation"。还剩一种情况：超分和插帧都由 AMD
  运行库完成时，在我们的 RTX 5090 上（那里是 AMD 的 FSR 3.1；Radeon 上的 FSR 4 尚未测试）大约每 25 次 F11 仍有一次丢失显卡
  设备。如果你遇到，加上 `-Set KYTY_FSR_RUNTIME=0`：插帧照旧，超分改用内置 FSR 3，这样在我们这里从未出现过。
- **启动器把 FSR 超分改称「AMD FSR」**：放有 SDK 2.4 的 DLL 时，Radeon RX 7000 和 9000 显卡上是 FSR 4，其他显卡上是
  FSR 3.1；没有这些 DLL 时是内置的 FSR 3。日志中的 `Super resolution: AMD FSR ...` 一行会写出实际使用的版本。
- **偶发的启动卡死**（PR #269，感谢 @CarpetHead）：GPU 线程可能在等待一次只有它自己才能准备的翻转。跳过启动标志时最容易出现。
- **游戏线程等待 GPU 数据的时间更短**（PR #273，感谢 @CarpetHead）：渲染线程等待上传期间，也会处理游戏线程的请求。在游戏大量
  读回数据的地方最多快 2-4%（@DevList1454 在 RX 9070 XT 上于波雷塔利亚王城内城测得），其他地方不变；
  `-Set KYTY_SERVE_IN_WAITS=0` 关闭。
- **便于反馈 AMD FSR 的问题**（#271、#276）：日志现在会写出 AMD 运行库在插帧、插帧交换链和超分上各占用多少显存，以及 FSR 超分
  收到的输入。
- **小提示：8 GB 的 Radeon RX 6000 显卡开插帧**（#276）：在这类显卡上，SDK 2.4 的插帧比旧的 `amd_fidelityfx_vk.dll` 多占约
  1 GB 显存，1080p 加 FSR 4 时显存不够。在问题解决之前，插帧请单独使用旧 DLL，或者加上 `-Set KYTY_FSR_RUNTIME=0`（改由内置
  FSR 3 超分，在这类显卡上负担更轻）。
- **感谢** @CarpetHead 的 PR #269、#273 和 #274（供测试者使用的闪烁检查工具），@DevList1454 的审阅与测量，以及 @kocher、
  @dimitrios1977ZXZ、@yura1234、@STALKERSK、@davide20204 和 @Vyncas 的反馈与测试。
