## New in this build

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

## New in v20261101

Your precompiled shaders stay valid: the shaders are the same as in v20261031. The precompile has more shaders to
offer (last item): the launcher asks you to precompile again, and that run compiles only the new ones.

- **Skip the start-up logos** (#263): a new switch on the launcher's Game page. The game then starts without "Sony
  Interactive Entertainment presents", PlayStation Studios and Bluepoint Games, about 16 seconds that no button skips
  (the game's own switch; `-Set KYTY_SKIP_LOGOS=1` with `run.cmd`). The opening cinematic still plays, and Options
  skips it as before.
- **Nintendo Switch Pro and other Nintendo-layout pads** (#262): the face buttons now go by position, as the PlayStation
  buttons the game shows: the bottom one is Cross and the right one Circle (before, they went by label: A, on the right,
  was Cross). Joy-Cons, 8BitDo pads in Switch mode and GameCube pads too; `-Set SDL_GAMECONTROLLER_USE_BUTTON_LABELS=1`
  goes back to the labels.
- **Steam Big Picture: the controller works right after you quit the game** (#217): after a game started from Big
  Picture ended, Big Picture stayed without focus until a click or Alt+Tab. The launcher now gives it the focus back
  once Steam has seen the game end.
- **Less work on the render thread** (PR #257, credit @ponhei9502): a buffer description the game hands over unchanged
  keeps its looked-up entry instead of being hashed again, about 0.5 ms less per frame at Boletaria 1-1 on an RTX 5090
  (`-Set KYTY_TABLE_GATHER_MEMO=0` turns it off).
- **FSR 3 frame generation: F11 in menus and during loads** (#174): switching to or from full screen while frame
  generation was paused (the main menu, loading screens) could freeze the picture or lose the graphics device.
- **DualSense: an enchanted weapon no longer rumbles all through the buff** (#239): steady effects now fade from the
  motors in about half a second, while hits still rumble fully (`-Set KYTY_DUALSENSE_VIBRATION_MOTOR_SUSTAIN_MS=<ms>`,
  0 as before).
- **Direct presentation with DLSS frame generation** (#236): v20261031 ignored that Experimental switch with any frame
  generation; now only with FSR 3's, the one it held at half the display's rate.
- **Precompile: more shaders** for DLSS, frame generation and the 1440p output (#260, credit @2000Arsen, from a full
  playthrough): 859 programs and 894 pipelines more, fewer first-use stutters with those settings.
- **Thanks** to @ponhei9502 for PR #257, to @2000Arsen for the shader list, and to @HybOj, @forensicssix-debug, @ft1704,
  @jrodriguezml, @tao755-maker and @javsp for their reports and tests.

## 本版本的变化

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

## v20261101 的变化

已有的预编译着色器继续有效：着色器与 v20261031 相同。预编译可以多编译一些着色器（见最后一项）：启动器会提示再预编译一次，
这次只编译新增的部分。

- **跳过启动标志**（#263）：启动器「游戏」页新增开关。开启后游戏启动时不再播放「Sony Interactive Entertainment presents」、
  PlayStation Studios 和 Bluepoint Games 的标志，约 16 秒、任何按键都无法跳过的画面（游戏自带的开关；用 `run.cmd` 时为
  `-Set KYTY_SKIP_LOGOS=1`）。开场动画仍会播放，和以前一样可以按 Options 跳过。
- **Nintendo Switch Pro 等任天堂布局手柄**（#262）：正面按键改为按位置对应，与游戏显示的 PlayStation 按键一致：下方为 ×，右方为
  ○（以前按标签对应：右边的 A 是 ×）。Joy-Con、Switch 模式的 8BitDo 手柄和 GameCube 手柄同样如此；
  `-Set SDL_GAMECONTROLLER_USE_BUTTON_LABELS=1` 恢复按标签对应。
- **Steam Big Picture：退出游戏后手柄立即可用**（#217）：从 Big Picture 启动的游戏结束后，Big Picture 一直没有焦点，需要点一下或
  按 Alt+Tab。现在 Steam 确认游戏结束后，启动器会把焦点交还给它。
- **渲染线程的工作更少**（PR #257，感谢 @ponhei9502）：游戏交来的缓冲区描述没有变化时，沿用已查到的条目，不再重新计算哈希；
  RTX 5090 在波雷塔利亚 1-1 每帧约少 0.5 ms（`-Set KYTY_TABLE_GATHER_MEMO=0` 关闭）。
- **FSR 3 插帧：在菜单和加载时按 F11**（#174）：插帧暂停时（主菜单、加载画面）切换全屏，可能导致画面冻结或显卡设备丢失。
- **DualSense：附魔武器期间不再一直震动**（#239）：持续性的效果现在约半秒内从震动马达上渐弱，受击等仍会完整震动
  （`-Set KYTY_DUALSENSE_VIBRATION_MOTOR_SUSTAIN_MS=<毫秒>`，0 恢复原样）。
- **DLSS 插帧 + 直接呈现**（#236）：v20261031 在任何插帧下都会忽略这个「实验性」开关；现在只在 FSR 3 插帧下忽略（只有它会让
  帧率只剩显示器刷新率的一半）。
- **预编译：更多着色器**，覆盖 DLSS、插帧和 1440p 输出（#260，感谢 @2000Arsen 提供的完整通关记录）：多 859 个程序、894 条
  管线，这些设置下首次出现的卡顿更少。
- **感谢** @ponhei9502 的 PR #257、@2000Arsen 提供的着色器列表，以及 @HybOj、@forensicssix-debug、@ft1704、@jrodriguezml、
  @tao755-maker 和 @javsp 的反馈与测试。
