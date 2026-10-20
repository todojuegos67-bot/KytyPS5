## New in this build

v20261020b fixes a crash of v20261020 on PCs with 16 GB of RAM. Everything new in v20261020 (further below) is in it
too.

- **Crash on PCs with 16 GB of RAM (issue #91):** v20261020's new memory saving for textures (on by default with 20 GB
  of RAM or less) could end the emulator with "PageManager fatal: read-watcher overflow" while a level loaded. Fixed;
  the saving stays on. If you added `-Set KYTY_TEXTURE_OFFLOAD=0` to get around the crash, you can remove it again.
- **Four shaders that ended the emulator:** four of the game's pixel shaders could not be translated, and the emulator
  stopped where the game drew with one of them (an automated tour hit one in the Valley of Defilement). They translate
  now, and the precompile builds them.
- **More shaders precompiled:** automated tours through all six worlds, fighting and killing every enemy (bosses too),
  added 153 shader programs and 580 pipelines to the hints. Among them are three large Shrine of Storms shaders that
  froze the game for up to 40 s the first time they were needed. The launcher offers the precompile again at the first
  start; only the new part is compiled.
- **Linux with AMD graphics:** the emulator's main thread asked the graphics driver about 2,000 times a frame how far
  the GPU had got. On Linux each question is a system call (about 9 µs on RADV). Questions whose answer is already known
  are skipped now, and where a question is slow, a separate thread follows the GPU instead (`KYTY_TIMELINE_WATCHER=0`
  turns it off). Not measured on Linux yet: reports are welcome in issue #80.
- **Thanks** to @ededaelita1-afk for the crash report and to @yazuc for the Linux profiles.

## New in v20261020

- **Less memory on 16 GB PCs and 8 GB graphics cards:**
  - Game memory nothing ever wrote is no longer read when the GPU first uses it, only taken as zeros: 0.7-1.4 GB less
    RAM in use on a 16 GB PC (`KYTY_FRESH_MEMORY=0` turns it off).
  - On PCs with 16 GB of RAM, textures the game only reads keep one copy, in video memory, instead of a second one in
    RAM: about 0.5 GB less RAM in use (`KYTY_TEXTURE_OFFLOAD=0` turns it off, `=1` turns it on with more RAM).
  - The shaders of a world the game left are dropped a minute after it, on every graphics card. Before, they piled up
    for the whole session (the driver's memory went from 0.9 to 2 GB on a trip through five worlds).
  - The warm-up at start compiles the shaders recorded for the map the game loads, not those of every map (up to 4 GB
    of the driver's memory).
  - Video memory: the GPU's address table takes 64 MiB instead of 512 MiB, and game buffers no longer keep video
    memory for game memory the game freed. On cards with less than 12 GB, video memory is also allocated in smaller
    blocks and game buffers are compacted after the game frees memory: about 0.6 GB less video memory on an 8 GB card
    after travelling between worlds (`KYTY_BUFFER_DEFRAG=0` turns the compacting off).
- **AMD and Intel graphics cards:** pipelines compiled while playing are kept and loaded at the next start, instead of
  being compiled again every session (issue #78; `KYTY_PLAYED_PIPELINES=0` turns it off).
- **More shaders precompiled:** a player shared the shader hints of a whole 1.07 playthrough (issue #88, thanks
  @tscftw): 2,091 more shader programs and 2,213 more pipelines, from places our own test runs never reached. The
  launcher offers the precompile again at the first start; only the new part is compiled.
- **Smoother:** in places with many lights and enemies (Boletaria 1-3), reading a light table no longer waits for the
  GPU: a few more fps and fewer short stutters.
- **Fixes:**
  - On "lite" Windows builds that turn Control Flow Guard on for every program, the emulator ended within the first
    seconds without a line in the log (issue #56).
  - The save dialogs show Chinese, Japanese and Korean text with the system's font instead of "?" (a contribution by
    @Myoko, PR #75).
  - A save whose capacity record was all zeros (after a power cut) failed every time the game opened it; it now counts
    as empty (a contribution by @Vorshim92, PR #74).
  - The driver's shader caches: a small one is saved at least once a minute while shaders compile, and with the
    precompiled store one past 256 MiB starts over instead of growing.
- **README:** how to play in Chinese, Japanese or Korean (a value in the game's `param.json` picks the language list),
  and how to update to a newer package (extract it over the old folder; what is kept).
- **Thanks** to @nerdjb for the Linux port fixes and the Linux package, and to everyone who sent logs and tests.

## 本版本的变化

v20261020b 修正了 v20261020 在 16 GB 内存电脑上的一个崩溃。v20261020 的所有新内容（见下方）也都包含在内。

- **16 GB 内存电脑上的崩溃（issue #91）：** v20261020 新增的纹理省内存功能（内存 20 GB 及以下默认开启）可能在加载关卡时让模拟器以 "PageManager fatal: read-watcher overflow" 退出。已修正，该功能保持开启。如果你为了绕过崩溃加了 `-Set KYTY_TEXTURE_OFFLOAD=0`，现在可以去掉了。
- **会让模拟器退出的四个着色器：** 游戏里有四个像素着色器无法翻译，游戏用到它们的地方模拟器就会停止（自动测试在污秽谷（Valley of Defilement）遇到了其中一个）。现在它们可以正常翻译，预编译也会生成它们。
- **预编译的着色器更多了：** 自动测试跑遍了全部六个世界，与每个敌人（包括 Boss）战斗并击杀，提示里多了 153 个着色器程序和 580 个管线。其中包括风暴神殿（Shrine of Storms）的三个大型着色器，它们第一次用到时会让游戏卡住最长约 40 秒。第一次启动时启动器会再次提示预编译，只会编译新增的部分。
- **Linux 和 AMD 显卡：** 模拟器主线程每帧向显卡驱动询问 GPU 进度约 2,000 次。在 Linux 上每次询问都是一次系统调用（RADV 上约 9 微秒）。现在已知答案的询问会被跳过；询问较慢时，改由一个单独的线程跟踪 GPU 进度（`KYTY_TIMELINE_WATCHER=0` 可关闭）。尚未在 Linux 上实测，欢迎在 issue #80 反馈。
- **感谢** @ededaelita1-afk 报告崩溃，感谢 @yazuc 提供 Linux 性能分析数据。

## v20261020 的变化

- **16 GB 内存的电脑和 8 GB 显卡占用更少：**
  - 游戏从未写过的内存，在 GPU 第一次使用时不再去读取，直接当作全零：16 GB 内存的电脑上少用 0.7-1.4 GB 内存（`KYTY_FRESH_MEMORY=0` 可关闭）。
  - 在 16 GB 内存的电脑上，游戏只读取的纹理只在显存里保留一份，不再在内存里多存一份：少用约 0.5 GB 内存（`KYTY_TEXTURE_OFFLOAD=0` 可关闭，`=1` 可在更大内存上开启）。
  - 游戏离开一个世界一分钟后，那个世界的着色器会被释放，所有显卡都是如此。之前它们会在整个游玩过程中一直累积（走一遍五个世界，驱动内存从 0.9 GB 涨到 2 GB）。
  - 启动时的预热只编译游戏正在加载的地图录到的着色器，不再编译所有地图的（之前最多占用驱动 4 GB 内存）。
  - 显存：GPU 地址表只占 64 MiB，不再是 512 MiB；游戏缓冲区不再为游戏已释放的内存保留显存。显存小于 12 GB 的显卡上，显存还会按更小的块分配，游戏释放内存后缓冲区会被整理紧凑：8 GB 显卡在几个世界之间来回之后显存少用约 0.6 GB（`KYTY_BUFFER_DEFRAG=0` 可关闭整理）。
- **AMD 和 Intel 显卡：** 游戏过程中编译的管线会被保存，下次启动直接载入，不再每次都重新编译（issue #78；`KYTY_PLAYED_PIPELINES=0` 可关闭）。
- **预编译的着色器更多了：** 一位玩家分享了他完整通关 1.07 版的着色器提示（issue #88，感谢 @tscftw）：多了 2,091 个着色器程序和 2,213 个管线，来自我们自己的测试路线没有走到的地方。第一次启动时启动器会再次提示预编译，只会编译新增的部分。
- **更流畅：** 在灯光和敌人很多的地方（波雷塔利亚 1-3），读取灯光表不再等待 GPU：帧率高几帧，短暂卡顿更少。
- **修正：**
  - 在对所有程序都开启控制流防护（CFG）的"精简版" Windows 上，模拟器会在开头几秒内退出，日志里没有任何记录（issue #56）。
  - 存档对话框里的中文、日文、韩文改用系统字体显示，不再显示为"?"（@Myoko 的贡献，PR #75）。
  - 容量记录全为零的存档（断电之后）以前每次打开都会失败，现在按空存档处理（@Vorshim92 的贡献，PR #74）。
  - 驱动的着色器缓存：编译着色器期间，较小的缓存至少每分钟保存一次；有预编译存储时，超过 256 MiB 的缓存会重新开始，不再一直增长。
- **README：** 如何用中文、日文或韩文游玩（游戏 `param.json` 里的一个值决定可选语言），以及如何更新到新的包（解压覆盖旧文件夹，哪些内容会保留）。
- **感谢** @nerdjb 完成 Linux 移植的修正和 Linux 安装包，也感谢所有提供日志和测试的玩家。
