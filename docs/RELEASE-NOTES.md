## New in this build

- **DLSS super resolution** (new launcher row "Upscaler", NVIDIA RTX only, off by default): Demon's Souls' own PS5 Pro
  mode, with NVIDIA DLSS in place of Sony's PS5 Pro upscaler. The 3D is rendered at the render height and DLSS makes a
  real 3840x2160 image of it, which the game finishes at 4K (bloom, depth of field, the HUD). It starts at render
  height 1080 (DLSS Performance): on an RTX 5090 about the frame rate of the game's own TAA at 1440, with a sharper
  picture; 1440 is DLSS Quality, 1512 the PS5 Pro's own, 2160 DLAA. It needs NVIDIA's Streamline runtime, chosen as
  for frame generation (the launcher's "Streamline..." button). The in-game Cinematic mode stays native 4K.
- **Higher render heights:** the launcher's "Render height" now goes above the PS5's 1440, up to 2160 (the 3D at the
  output's own 3840x2160). It also works on game version 01.005.000 now, where it had no effect (issue #69).
- **Ultrawide displays:**
  - Resolution, aspect ratio, render height and upscaler are side by side in the launcher. The resolutions follow the
    aspect ratio (21:9: 2560x1080, 3440x1440, 3840x1600, 5120x2160; 32:9: 3840x1080, 5120x1440, 7680x2160), and the
    render heights stop where the game's render target memory ends (21:9 up to 1800, 32:9 up to 1512).
  - A 21:9 render height of 2160 ended the game at start-up, and one of 1864 showed the 3D in red only: the game's
    render targets are made for its own 3840x2160. The emulator now lowers a height that does not fit by itself and
    says so in the log.
  - With DLSS, an ultrawide aspect makes the full-width picture (5120x2160 at 21:9) instead of a widened 16:9 one.
  - Movies, menus and loading screens are stretched to the full width instead of 16:9 between black bars.
  - Enemy health bars stay over their enemies, and the box of picked-up items stays whole.
  - Known issues: at 21:9 the lock-on marker sits a little off an enemy that is off the screen's centre; with DLSS at
    21:9 the opening movie can show at the left three quarters of the screen.
- **Saves are kept safe from a crash while saving** (issue #73): a blue screen while the game saved could leave the
  save file full of zeros. Saves are now flushed to disk when the game finishes them, and each save is also copied to
  `_SaveData\.kyty-backup` (the last 4, at least 5 minutes apart). A save found broken that way at the next start gets
  the newest good copy back automatically; the log says which one, and the broken files are kept beside the backups.
- **AMD graphics cards:**
  - Black squares all over the picture after precompiling with v20261018 (RX 6000 series, issues #64, #71): a shader
    change of that version is now made for NVIDIA cards only.
  - The GPU reset at the Vanguard (RX 9070 XT on Linux, issue #51) is fixed.
  - Both change shaders on AMD cards: the launcher offers the precompile again at the first start.
- **Fixes:**
  - The precompile uses every core on Windows 11 PCs with efficiency cores; some ran it on the efficiency cores only
    (issue #63).
  - A rare exit after long sessions with DLSS ("PageManager fatal: invalid range vaddr=0x0", issue #70).
  - A debug message from a system library no longer ends the emulator, and a fatal end writes what and where to the log
    (issue #56).
  - The MEMORY_MANAGEMENT blue screen (0x1A/0x41202, issues #5, #50): a stronger fix is on by default: every page
    protection change now waits for any memory map or unmap in progress. Please tell us if it still happens.
- **In the game:**
  - **F9** saves a debug snapshot and pauses the game: the picture shown, where you are and the run's log, in
    `_DebugSnapshots\<date-time>\`. F9 again resumes. Attach the folder to an issue when something looks wrong.
  - F8's "Frame rate" panel shows at once instead of after a second.
- **Options** (off by default): `KYTY_FG_HUDLESS=1` gives DLSS frame generation the game image without its HUD, for
  steadier HUD in generated frames (a contribution, PR #72); `KYTY_AUDIO_CATCHUP_MS` and `KYTY_AUDIO_CUSHION_MS` for
  audio dropouts with heavy GPU add-ons (a contribution, PR #68).
- **Diagnostics** (for bug reports): `KYTY_PIPELINE_STATS=1` (the driver's statistics of each pipeline),
  `KYTY_SR_TIMING=1` and `KYTY_DLSS_PRESET=<letter>` (DLSS), and a backtrace when the page tracker ends the emulator.

## 本版本的变化

- **DLSS 超分辨率**（启动器新增"Upscaler"一行，仅限 NVIDIA RTX，默认关闭）：使用《恶魔之魂》自己的 PS5 Pro 模式，用 NVIDIA DLSS 代替索尼的 PS5 Pro 超分。3D 按渲染高度渲染，由 DLSS 生成真正的 3840x2160 图像，再由游戏在 4K 下完成后期（泛光、景深、HUD）。默认从渲染高度 1080 开始（DLSS 性能档）：在 RTX 5090 上帧率与游戏自带 TAA 在 1440 时相当，画面更清晰；1440 为 DLSS 质量档，1512 是 PS5 Pro 原生设置，2160 为 DLAA。需要 NVIDIA 的 Streamline 运行库，选择方式和帧生成一样（启动器的"Streamline..."按钮）。游戏内的电影模式仍为原生 4K。
- **更高的渲染高度：** 启动器的"Render height"现在可以超过 PS5 的 1440，最高到 2160（3D 按输出本身的 3840x2160 渲染）。在游戏版本 01.005.000 上也生效了，之前在该版本上不起作用（issue #69）。
- **超宽屏：**
  - 启动器里的分辨率、画面比例、渲染高度和超分方式放在一起。分辨率列表跟随画面比例（21:9：2560x1080、3440x1440、3840x1600、5120x2160；32:9：3840x1080、5120x1440、7680x2160），渲染高度只列出游戏渲染目标内存装得下的（21:9 最高 1800，32:9 最高 1512）。
  - 21:9 下渲染高度 2160 会让游戏启动即退出，1864 会让 3D 画面只剩红色：游戏的渲染目标是按它自己的 3840x2160 分配的。现在模拟器会自动把装不下的高度降下来，并在日志里说明。
  - 使用 DLSS 时，超宽画面比例会直接生成全宽画面（21:9 下为 5120x2160），而不是把 16:9 画面拉宽。
  - 过场视频、菜单和加载画面拉伸铺满全宽，不再是 16:9 加黑边。
  - 敌人血条保持在敌人头上，拾取物品的提示框保持完整。
  - 已知问题：21:9 下，不在屏幕中央的敌人身上的锁定标记会略有偏移；DLSS 加 21:9 时，开场动画可能只显示在屏幕左侧四分之三。
- **保存时崩溃也不会丢存档**（issue #73）：以前如果在游戏保存时蓝屏，存档文件可能变成全零。现在游戏每次保存完成都会写入磁盘，并另外复制一份到 `_SaveData\.kyty-backup`（保留最近 4 份，彼此至少间隔 5 分钟）。下次启动时如果发现存档这样损坏，会自动换回最新的完好备份，日志里会写明用的是哪一份，损坏的文件保留在备份旁边。
- **AMD 显卡：**
  - 用 v20261018 预编译后画面上出现大量黑色方块（RX 6000 系列，issue #64、#71）：那个版本的一项着色器改动现在只用于 NVIDIA 显卡。
  - 先锋恶魔处的 GPU 重置（Linux 上的 RX 9070 XT，issue #51）已修正。
  - 这两项都改变了 AMD 显卡上的着色器：第一次启动时启动器会再次提示预编译。
- **修正：**
  - 在有能效核的 Windows 11 电脑上，预编译会用上所有核心；之前有些电脑只在能效核上运行（issue #63）。
  - 长时间使用 DLSS 后偶尔退出的问题（"PageManager fatal: invalid range vaddr=0x0"，issue #70）。
  - 系统库输出调试信息时不再导致模拟器退出；致命错误退出时会在日志里写明原因和位置（issue #56）。
  - MEMORY_MANAGEMENT 蓝屏（0x1A/0x41202，issue #5、#50）：更强的修正默认开启：每次修改页保护时都会等待正在进行的内存映射或解除映射。如果还会发生，请告诉我们。
- **游戏中：**
  - **F9** 保存一份调试快照并暂停游戏：当前画面、所在位置和本次运行的日志，存放在 `_DebugSnapshots\<日期-时间>\`。再按 F9 继续。画面有问题时，把这个文件夹附到 issue 里。
  - F8 的"Frame rate"面板切换后立即显示，不再等一秒。
- **选项**（默认关闭）：`KYTY_FG_HUDLESS=1` 给 DLSS 帧生成提供不含 HUD 的游戏画面，让生成帧里的 HUD 更稳定（社区贡献，PR #72）；`KYTY_AUDIO_CATCHUP_MS` 和 `KYTY_AUDIO_CUSHION_MS` 用于使用重型 GPU 插件时的音频断续（社区贡献，PR #68）。
- **诊断**（用于报告问题）：`KYTY_PIPELINE_STATS=1`（驱动对每个管线的统计）、`KYTY_SR_TIMING=1` 和 `KYTY_DLSS_PRESET=<字母>`（DLSS），以及页面跟踪器结束模拟器时输出调用栈。
