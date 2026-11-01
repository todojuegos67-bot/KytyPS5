## New in this build

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

## New in v20261031

Your precompiled shaders stay valid: the shaders are the same as in v20261030.

- **21:9 and 32:9: movies and menus at 16:9 between black bars** (#250): a new option on the launcher's Display page,
  "Movies and menus at 16:9 (black bars)". With 21:9 or 32:9 the intro, the movies, the title screen, the loading
  screens and the menus opened in play stay at 16:9 in the middle, with black bars at the sides, instead of being
  stretched to the width. It works with DLSS or FSR 3 and with the game's own anti-aliasing alike; the 3D view stays
  wide.
- **21:9 and 32:9 with FSR 3 frame generation** (#250): the HUD no longer leaves a ghost while you move, and the start-up
  logos are no longer cut off at the right.
- **Frame generation with "Direct presentation" on** (#236): the game ran at half the display's rate (30 fps on an RTX
  4090 at 120 Hz with FSR 3 frame generation), and each frame waited a frame longer to be shown. That Experimental
  switch is now ignored while frame generation is on.
- **AMD: smoother menus with FSR 3 frame generation and low latency** (#252): AMD Anti-Lag stays off while frame
  generation is on. With both, an RX 6650 XT's menus and shops dropped to 5-6 fps.
- **Fewer exits when video memory runs out** (#252): when the graphics card has no room for an image the game needs (an
  8 GB card with frame generation at 3840x1200), the emulator now also frees old render targets, saving their contents
  first, and tries again for up to 10 seconds before it gives up.
- **Intel Arc: no more full freezes** (#255): an Arc B580 froze completely after a while in 1-2 and 3-1. The data the
  emulator reads back from the graphics card is now copied on its main queue (`-Set KYTY_READBACK_QUEUE=1` goes back to
  the copy queue).
- **Xbox and other non-PlayStation pads** (#236): running no longer breaks off: the stick counts as fully pushed from 90%
  of its travel (`-Set KYTY_PAD_DEADZONE_OUTER=<percent>`). The long rumble at the title screen, at Continue and at each
  spawn now fades out in about half a second, while hits still rumble fully (`-Set KYTY_PAD_RUMBLE_SUSTAIN_MS=<ms>`, 0
  as before). The Steam Deck's rumble too.
- **Button presses timed in the log** (#236): every 30 seconds with presses, an `Input:` line says how long a press
  waited before the emulator took it and before the game read it, for reports of input lag.
- **Launcher: more render heights for DLSS and FSR 3** (#129): Balanced at 4K (58%, 1256 lines) and Ultra Quality (77%:
  1664 at 4K, 1104 at 1440p, 832 at 1080p).
- **Linux: DualSense haptics over USB**, as an option (PR #254, credit @ponhei9502): `KYTY_DUALSENSE_HAPTICS=1` plays
  the game's haptics on the pad's actuators (off by default).
- **Thanks** to @ponhei9502 for PR #254, and to @H0A0M0mer, @Ayonily, @labs20, @STALKERSK, @thxnk420, @markusdev and
  @2mg for their reports and tests.

## 本版本的变化

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

## v20261031 的变化

已有的预编译着色器继续有效：着色器与 v20261030 相同。

- **21:9 和 32:9：过场动画与菜单保持 16:9（黑边）**（#250）：启动器「画面」页新增选项「过场动画与菜单保持 16:9（黑边）」。
  选 21:9 或 32:9 时，片头、过场动画、标题画面、加载画面以及游戏中打开的菜单都保持 16:9 居中，两侧为黑边，不再拉伸到整个
  宽度。使用 DLSS、FSR 3 或游戏自带的抗锯齿时都有效；3D 画面仍是宽屏。
- **21:9 和 32:9 + FSR 3 插帧**（#250）：移动时 HUD 不再留下残影，启动时的标志动画右侧也不再被裁掉。
- **插帧 + 开启「直接呈现」**（#236）：游戏只能跑到显示器刷新率的一半（RTX 4090、120 Hz、FSR 3 插帧时为 30 fps），每一帧也要多等
  一帧才显示。现在开启插帧时会忽略这个「实验性」开关。
- **AMD：FSR 3 插帧 + 低延迟时菜单更流畅**（#252）：开启插帧时 AMD Anti-Lag 保持关闭。两者同时开启时，RX 6650 XT 在菜单和商店
  里会掉到 5-6 fps。
- **显存不足时更少退出**（#252）：显卡放不下游戏需要的图像时（例如 8 GB 显卡在 3840x1200 下开启插帧），模拟器现在还会释放旧的
  渲染目标（先保存其内容），并在放弃之前最多重试 10 秒。
- **Intel Arc：不再整个卡死**（#255）：Arc B580 在 1-2 和 3-1 玩一段时间后会完全卡死。模拟器从显卡读回的数据现在在主队列上复制
  （`-Set KYTY_READBACK_QUEUE=1` 恢复为复制队列）。
- **Xbox 等非 PlayStation 手柄**（#236）：跑步不再中断：摇杆推到 90% 即视为推满（`-Set KYTY_PAD_DEADZONE_OUTER=<百分比>`）。
  标题画面、「继续」和每次复活时的长时间震动现在约半秒内渐弱，受击等仍会完整震动（`-Set KYTY_PAD_RUMBLE_SUSTAIN_MS=<毫秒>`，
  0 恢复原样）。Steam Deck 的震动也一样。
- **日志记录按键耗时**（#236）：有按键时每 30 秒一行 `Input:`，写出按键等了多久才被模拟器接收、又过了多久被游戏读取，便于反馈
  输入延迟。
- **启动器：DLSS 和 FSR 3 增加渲染高度**（#129）：4K 下的「平衡」（58%，1256 行）和「超级质量」（77%：4K 为 1664，1440p 为
  1104，1080p 为 832）。
- **Linux：DualSense 通过 USB 的触觉反馈**（可选，PR #254，感谢 @ponhei9502）：`KYTY_DUALSENSE_HAPTICS=1` 让游戏的触觉效果在
  手柄的触觉马达上播放（默认关闭）。
- **感谢** @ponhei9502 的 PR #254，以及 @H0A0M0mer、@Ayonily、@labs20、@STALKERSK、@thxnk420、@markusdev 和 @2mg 的反馈与测试。
