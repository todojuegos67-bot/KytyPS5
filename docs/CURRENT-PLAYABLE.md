# 开发手册（Windows）

构建、运行和测量模拟器的日常做法；面向玩家的说明见 [PORTABLE-README.md](PORTABLE-README.md)，
工具见 [tools/local/README.md](../tools/local/README.md)。

## 构建与运行

```powershell
.\build-windows.cmd                              # clang-cl 19 + Ninja：-O3 -march=native + ThinLTO + PGO，输出 _Build\windows\kyty_emulator.exe
.\run-windows.ps1 -Fullscreen -AspectFit         # 日常：全屏，16:9 画面 1:1 居中（5120×2160 屏两侧黑边）
.\run-windows.ps1 -Width 1920 -Height 1080       # 窗口模式（画面拉伸铺满窗口）
.\run-windows.ps1 -Precompile                    # 预编译全部已记录的 shader/管线后退出（换 exe 后首次约 7 s，冷驱动缓存约 2 min）
.\run-windows.ps1 -DryRun                        # 只打印环境变量和命令
```

其他参数：`-NoRedZone`、`-Set KEY=VALUE`（设一个模拟器开关；默认值就是发布设置，见
`src/local/performance-switches.h`，`=0` 关掉一条路径做 A/B）、
`-Patch <cheat.json>`（etaHEN 格式补丁）、`-PresentMode`、`-Vblank`、`-Game <游戏目录>`（记在 `game-path.txt`，
都找不到时弹出选择框）、`-Affinity <十六进制掩码>`。

- 依赖：VS 2022（C++ 工作负载）、LLVM 19.1.7（`winget install LLVM.LLVM --version 19.1.7`）、Vulkan SDK（glslangValidator）、
  Python 3。缺的子模块由 `build-windows.cmd` 自动检出；新 clone 的完整步骤见 README 的 "Building with build-windows.cmd"。
  LLVM 23.1.2 编译 `agc.cpp` 时编译器自身崩溃，不要用。
- 原 CPU 4、5 两个 P 核不稳定（clang 随机崩溃、系统蓝屏 0x20001），09-30 起在 BIOS 里禁用：现为 6 P 核（0–5）+ 16 E 核（6–21），
  构建与运行不再需要亲和性掩码（`build-windows.cmd` 只在设了 `KYTY_BUILD_AFFINITY` 时才绑核）。
- 渲染线程限定在 P 核 1–5（配置的 `KYTY_RECORDING_CPUS`，非独占）：比自由调度快约 2.5%。**不要像 Linux 那样独占 CPU 0**：
  Windows 的中断/DPC 集中在 CPU 0，帧率直接减半；用 CPU Set 把其他线程赶出某个 P 核也会大幅变慢。
- 给其他电脑：GitHub Actions（`.github/workflows/build.yml`）编 x86-64-v3 版（`-DKYTY_MARCH=x86-64-v3`；
  `-march=native` 会用到本机的 GFNI，别的 CPU 上直接非法指令），带仓库里的 PGO profile（`tools/pgo/kyty.profdata`），
  装配成发布包：exe、预编译程序、VC++ 运行库 DLL（随包，不用装）、libwinpthread、`run-windows.ps1`、双击用的
  `run.cmd`、`launcher.cmd`、`precompile.cmd`、`launch.json`（= `run-windows.json`：命令行）、生成种子用的渲染状态表
  （`tools/local/static-precompile/pass-states.json`）和给测试者的 `README.md`（源文件 `docs/PORTABLE-README.md`）。只在 Actions 页
  Run workflow 或推 `v*` tag 时编译并发布 release（标为 latest，README 顶部链接到它）；推送代码不触发编译。
  不含游戏、存档、着色器缓存（按 GPU+驱动区分）、Streamline 和种子文件（含游戏的 shader 代码）：
  首次启动由 `run-windows.ps1` 从玩家自己的游戏文件生成 `seeds-<标题>_<版本>.seeds`（预编译程序的 `--make-seeds`：
  `precompile.py seeds` 的 C++ 移植，输出逐字节相同，不再需要 Python + numpy）。
  种子、静态缓存、预取输入和预热录制都按游戏版本命名（`<标题>_<版本>`：1.05 与 1.07 只有 10/约 21000 个 shader
  相同）；旧的只按标题命名的文件由 `run-windows.ps1` / `precompile-windows.ps1` 改名归给上次玩的游戏（`game-path.txt`）。
  窗口比屏幕大时模拟器按比例缩进可用区域。
- 着色器准备：启动器每次先跑 `kyty_shader_precompile --status`（约 1 秒，报告预取输入和静态管线缓存是否属于当前
  GPU+驱动），预取输入缺失或过期时自动生成（首次运行、更新驱动后）。`run.cmd` 带 `-Prompt`：静态缓存没做或游戏版本不是
  已测试的 PPSA01341 01.007.000 / 01.005.000 时先弹窗（先预编译 / 直接开始 / 退出，可勾选不再提示，记在 `no-precompile-prompt.txt`）。
- PGO：见 [tools/local/PGO.md](../tools/local/PGO.md)（发布前和一批热路径改动后重新训练）。
- 驱动管线缓存与预热：所有构建共用一个固定的 `KYTY_DRIVER_CACHE_KEY`（模拟器默认值），缓存在
  `_PipelineCache\local\<key>\`；驱动按 SPIR-V 和状态找管线，新构建复用没改过的着色器的管线。预热输入按"驱动版本+UUID"
  分目录，当前签名没有文件时自动接管同一游戏、同一 GPU 最新的文件。预热在游戏旁边进行（`-Precompile` 时在前面，按
  `KYTY_SHADER_WARMUP_THREADS` 并行）。

## 测量

`tools\local\windows\bench-run.ps1 -Label <名字>`（每次重装基准存档，截图识别 HUD，live `measure`）；
先 `python tools\local\bench-windows.py prepare` 备份用户存档，结束后 `restore`（逐文件 SHA256 核对）。
- live 命令：`measure`、`trace`、`census`、`pin`、`cpuset`、`set <符号> <值>`（经 map 文件换算地址）；
  `prof`/`profw` 为挂起采样（墙钟），`profp` 不支持。

## 21:9 / 5K

- `-AspectFit`：按比例居中显示。5120×2160 屏上 3840×2160 画面 1:1 显示、两侧黑边，无缩放。
- `-Aspect 21:9`（启动器"Aspect ratio"，也可 `32:9`）：3D 按该比例原生渲染（1440 高时 3416×1440），渲染初始化里的
  16:9 投影比例（`mov dword [rbx+4], 16/9`，1.07 `0x90090ae7f`、1.05 `0x9008ef287`）改为同一比例，左右视野变宽、不是拉伸
  （与 16:9 截图对比：拱窗宽高比 0.31/0.30、人物帽宽/身高 0.243/0.244）。游戏输出仍为 3840×2160，3D 帧由呈现器按该比例
  显示（窗口比例相差 2% 内则铺满），过场视频、读盘画面（没跑 TAA 的帧）也按该比例拉伸铺满（10-19 起；之前 16:9 加黑边）。游戏中 HUD 不拉伸
  （`src/local/hud-anchor.h`：三个 UI 顶点着色器画进 3840×2160 界面画布的绘制，按顶点所在区域——左上血条、左下装备、
  右上魂、下中 Boss 血条——把视口压回 16:9 并靠左/靠右/居中；一次绘制跨多个区域时按区域分次画并加 scissor；装备图标
  用屏幕坐标取遮罩，其常量矩阵同步修正；只在 HUD 圆形图标出现的帧生效，菜单仍整体拉伸）。锁定点、敌人血条仍对准目标。
  把游戏输出也改宽（5120×2160）时，游戏把 HUD 与菜单按 16:9 靠左排版，未采用。
- 渲染尺寸上限：游戏的渲染目标内存按它自己最大的 3D（Cinematic 3840×2160）分配。按 GPU 的 64 KiB tile（每像素 2/4/8/16
  字节：256×128、128×128、128×64、64×64 像素）数，3D 占的 tile 不能多于 3840×2160（`FitsRenderTargets`，
  `src/loader/demonsSoulsRenderSize.h`），否则写出界：21:9 的 4416×1864（4 字节 525 tile，4K 510）3D 全红，4776×2016 与 32:9
  的 5760×1620 背景发黄、模型发黑，4952×2088、5120×2160 启动即崩（`eboot+0x803e3a`，游戏自己的红黑树节点被覆盖，读地址 8）；
  4264×1800（正好 510）、5120×1440 正常。像素数不是标准：4416×1864 像素比 3840×2160 少。`MaxRenderHeight` 按比例给最高
  高度（16:9 2160、21:9 1832、32:9 1512），超出时自动降到它；启动器的渲染高度列表同样过滤（21:9 最高 1800，32:9 最高 1512），
  分辨率列表跟随画面比例。
