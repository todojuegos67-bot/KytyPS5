## New in this build

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

## New in v20261102

Your precompiled shaders stay valid: the shaders are the same as in v20261101.

- **AMD FSR 4 on Radeon cards** (#62): with AMD's FSR SDK 2.4 DLLs next to the emulator (`amd_fidelityfx_loader_vk.dll`,
  `amd_fidelityfx_upscaler_vk.dll` and `amd_fidelityfx_framegeneration_vk.dll` from `Kits\FidelityFX\signedbin` of the
  v2.4.0 package; the README says where to get it), the FSR upscaler becomes FSR 4 on Radeon RX 7000 and 9000 cards (RX
  6000 too), and FSR frame generation FSR 4 on RX 9000 cards. Other cards get FSR 3.1 from the same DLLs. Without them
  FSR 3 stays built in, and the older `amd_fidelityfx_vk.dll` still works. The log names what AMD's runtime chose
  ("Super resolution: AMD FSR 4.1.2 ..."). FSR 4 is new here and not yet tested on a Radeon: reports are welcome.
- **Linux: Korean, Japanese and Chinese text in the emulator's save and message dialogs** (#264): the dialogs found no
  CJK font there and showed "???"; they now ask fontconfig for one (Noto Sans CJK and the like). On Windows, Korean
  text is drawn with Malgun Gothic.
- **The on-screen keyboard's text grows with the screen** (#265): it stayed at 13 pixels at any size, tiny on a 4K TV;
  now 20 pixels at 1080p and twice that at 4K, with the buttons sized to their labels.
- **"Skip the start-up logos" removed again** (#266): a player with it on crashed twice after about 50 minutes of play,
  and played over two hours without it. Until the reason is known, the launchers no longer offer it (`-Set
  KYTY_SKIP_LOGOS=1` still turns it on, at your own risk). If you turned it on in v20261101, it is off now.
- **Tip: a 40 fps cap** (#267): on a 120 Hz or VRR display, the 120 fps option plus `-Set KYTY_FLIP_RATE=2` in the
  command-line options flips the game's 3D frames on every third refresh (40 fps), at the game's normal speed.
- **Thanks** to @kocher, @dreamerk24m, @ns6000, @HybOj, @HemeSynth and @davide20204 for their reports and tests.

## 本版本的变化

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

## v20261102 的变化

已有的预编译着色器继续有效：着色器与 v20261101 相同。

- **Radeon 显卡上的 AMD FSR 4**（#62）：把 AMD FSR SDK 2.4 的 DLL（v2.4.0 包中 `Kits\FidelityFX\signedbin` 下的
  `amd_fidelityfx_loader_vk.dll`、`amd_fidelityfx_upscaler_vk.dll` 和 `amd_fidelityfx_framegeneration_vk.dll`；说明文档写了
  在哪里下载）放到模拟器旁后，FSR 超分在 Radeon RX 7000 和 9000 显卡（以及 RX 6000）上变为 FSR 4，FSR 插帧在 RX 9000 显卡
  上变为 FSR 4；其他显卡由同一组 DLL 提供 FSR 3.1。没有这些 DLL 时仍使用内置的 FSR 3，旧的 `amd_fidelityfx_vk.dll` 也照常
  可用。日志会写出 AMD 运行库选择的版本（"Super resolution: AMD FSR 4.1.2 ..."）。FSR 4 是新加入的，还没有在 Radeon 上测试
  过，欢迎反馈。
- **Linux：模拟器的存档和消息对话框能显示韩文、日文和中文**（#264）：之前在 Linux 上找不到 CJK 字体，只显示"???"；现在
  通过 fontconfig 查找（例如 Noto Sans CJK）。Windows 上韩文改用 Malgun Gothic 显示。
- **屏幕键盘的文字随画面放大**（#265）：之前在任何分辨率下都是 13 像素，在 4K 电视上很小；现在 1080p 下为 20 像素，4K 下
  为两倍，按钮也按文字大小调整。
- **「跳过启动标志」再次移除**（#266）：有玩家开启后两次在玩了约 50 分钟时崩溃，关闭后玩了两个多小时都没有问题。原因查明
  之前，启动器不再提供这个开关（仍可用 `-Set KYTY_SKIP_LOGOS=1` 自行开启，风险自负）。如果你在 v20261101 中开启过，现在
  已自动关闭。
- **小提示：40 帧上限**（#267）：在 120 Hz 或 VRR 显示器上，开启 120 帧选项并在命令行参数中加入 `-Set KYTY_FLIP_RATE=2`，
  游戏的 3D 画面每三次刷新翻转一次（40 帧），游戏速度正常。
- **感谢** @kocher、@dreamerk24m、@ns6000、@HybOj、@HemeSynth 和 @davide20204 的反馈与测试。
