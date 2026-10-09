# 优化现状总结（2026-09-30）

Kyty PS5 模拟器的 Windows 移植，目标游戏《恶魔之魂》（PPSA01341，内容版本 01.007.000）。
分支 `experiment/perf-40fps-20260926`，**大量改动未提交**（`git diff HEAD` 约 90 个文件 + 若干未跟踪文件）。

## 1. 环境与当前性能

- 机器：Windows 11；i9-14900K（超线程关闭；原 CPU 4、5 两个 P 核不稳定，09-30 起在 BIOS 禁用，现为 6 P 核（0–5）
  + 16 E 核（6–21）= 22 逻辑核）；128 GB 内存；RTX 5090；LLVM 19.1.7 clang-cl + ThinLTO + PGO。
- 固定场景（基准存档 Continue → 前进 10 s → 静止，开阔废墟）：**约 40.5 fps**，渲染线程约 20 ms/帧。
- 步行路线（出隧道进废墟）：平均约 39 fps，最差 1 秒窗口 29–30 fps（09-28 晚）。用户目标：移动中最低 ≥ 30 fps。
- 游戏时钟依赖 60 Hz 虚拟 vblank（`run-windows.ps1` 默认 60；240 会让游戏变速）。

## 2. 瓶颈在哪里

- **渲染线程（Kyty.Gpu）是唯一瓶颈**，约 84% 忙。每帧大头（40 fps 场景）：
  - 计算 dispatch 准备约 8 ms（每帧约 4000 次 dispatch）：`GetComputeProgram` 2.8 ms（其中 SRT 求值约 1.5）、
    `RebindBuffers` 1.7 ms、`TryLinearCopy` 0.75 ms；
  - 原生 XPR draw 约 5 ms；普通 draw 约 3.6 ms。
- 录制线程（真正调用 Vulkan 驱动）约 10 ms/帧 CPU，不是瓶颈；GPU 每帧忙 16–19 ms，帧首与帧尾一段受 GPU 限制；
  游戏自身 CPU（headless）96–100 fps，不是瓶颈。
- 结论：剩下的主要是模拟器自身逻辑（翻译 PS5 命令流、资源跟踪），换图形 API（如 DX12）不会带来明显收益。

## 3. 已完成的主要优化（大致按收益）

| 优化 | 效果 |
|---|---|
| 原生 XPR 绘制（`KYTY_NATIVE_XPR`，GPU 端参数 + 批量绑定） | 24.85 → 30.93 fps（最大一项） |
| Vulkan 录制线程 + 延迟提交（`VULKAN_RECORDING`/`DEFERRED_SUBMIT`） | 21.1 → 23.7 fps |
| 09-28/29 纯逻辑优化合集（二次见到才存记录、不可存程序对屏蔽、原生直接 draw、SRT JIT 支持 64 位地址、按范围的后备存储栅栏、深度图部分上传等） | 36.08 → 40.71 fps（同进程 A/B） |
| 回读走独立拷贝队列（`KYTY_READBACK_QUEUE`） | 34.80 → 36.05 fps |
| 异步写回读（`ASYNC_WRITE_READBACK`） | 33.0 → 34.5 fps |
| 第 3 阶段 7 个开关（范围集快路径、图像粒度、XPR 预测等） | 32.34 → 33.70 fps |
| `-O3 -march=native` + ThinLTO；PGO + SRT AOT DLL | +4%；+8.4%（AOT 约 1.5%） |
| 流式加载卡顿修复（部分图像脏区/行带上传、纹理池部分解映射、异步 XPR 管线、跳过解映射时的保护恢复） | 最差帧 200 → 约 113 ms |
| 其他：全局屏障去重 +1.4%，普通 draw 录成包 −1.3 ms，图像池 −1 ms，写窗口移交 +1.1 fps | |

大部分运行时优化默认由 `run-windows.json`（环境变量开关）和 `run-windows.ps1` 打开。

**本游戏本版本专用的适配**（按标题 + 版本号 `01.007.000` 精确匹配，换版本会静默关闭，游戏照常但变慢；
收益是早期 5–12 fps 时测的，当前未重测）：计算 dispatch 间省略屏障（+12.7%，最大）、memmove 快路径（+5%）、
一个 CS 换成线性拷贝（+1.9%）、空转等待改短睡眠（CPU 占用 1338% → 388%）、GPU buffer 按页对齐。
其中后三个在 Windows 上是否真的装上未确认（日志被静音，red-zone 补丁会先改写代码）。

## 4. 着色器 / 管线编译卡顿（09-29 的工作）

- 问题：游戏第一次遇到新 shader/管线时要现场编译，大的计算 shader 卡 1–3 s 以上。
- 做法：**静态预编译**——直接从游戏文件收集全部 shader 和管线，提前编译进"静态管线缓存"
  `_PipelineCache/static/PPSA01341_<版本>.bin`（只按 GPU/驱动做 key，跨模拟器版本保留；游戏建管线时先在其中免编译查找）。
  - 收集：`kyty_shader_precompile --make-seeds`（`src/local/static-seeds.cpp`，`tools/local/static-precompile/precompile.py seeds`
    的 C++ 移植，输出逐字节相同；CSDR 包 + eboot 内嵌 shader + 材质配对 + 学得的渲染状态）。
  - 编译：独立程序 `kyty_shader_precompile.exe`（不开游戏），`precompile-windows.ps1` 分多进程并行、可中断续跑。
  - "便携 shader"（`KYTY_PORTABLE_SHADERS`，默认开）：把只在运行时描述符里的信息（buffer stride、格式、间接纹理表大小、
    空纹理维度）改为运行时读取，使静态编出的模块与游戏运行时的一致。
- 规模：约 2 万个程序（1.07 万个 CS，其中 9500 个粒子 CS，确实各不相同）、约 5 万条管线；本机多进程全量约 1 小时，
  重跑只编缺的。静态缓存 2.27 GB，游戏启动时加载约 2.2 s。
- 实测（09-30，冷启动固定路线，不预热）：运行时编出的模块 CS 82%、PS 91%、VS 100% 已在静态缓存里。缺的 18% CS
  （特化推断不同：整数纹理数值类型、存储图像单通道 swizzle、声明 2D 数组但绑定 cube/2D、被写的格式化 buffer 格式）
  在真正冷启动时要现场编译（约 36 条共 22 s）。现在这类管线先以未优化方式编译（NVIDIA 上约快 150 倍，
  毫秒级）立即使用，后台再编优化版替换（`KYTY_PIPELINE_FAST_BUILD`，默认开）：冷启动路线管线卡顿 22 s → 0。
  方法见 `tools/local/static-precompile/README.md`。
- 翻译器（`TranslateProgram`）首遇卡顿：CFG 支配分析改为支配树后，固定路线从 28 次/2.7 s/最长 915 ms 降到
  26 次/1.4 s/最长 162 ms（SPIR-V 不变）；分派器回退路径的 SPIR-V 改为确定性（见 `docs/EXPERIMENTS.md`）。
- **后台着色器预翻译**（09-30，`KYTY_SHADER_PREFETCH`，默认开，`=0` 关）：静态预编译的全部程序（2.1 万个输入，
  `_PipelineCache/static/PPSA01341_<版本>.shaders`，由 `precompile-windows.ps1` 写出，`-InputsOnly` 只写它，约 30 s）在游戏
  启动后由 12 个低优先级线程（避开 `KYTY_RENDER_CPUS`）按大小从大到小翻译，预热里已有的跳过；游戏第一次遇到某程序时
  直接取用它的资源计划和模块，只需建 shader 模块。翻译结果的 SPIR-V 共约 4.7 GB，放在系统临时文件里
  （`FILE_ATTRIBUTE_TEMPORARY`，内存够时不落盘，进程退出即删）；计划和程序信息留在内存，进程私有内存约多 1.2 GB。
  在标题画面期间完成（约 60–100 s）。冷启动（不预热）固定路线：行走中现场翻译 44 次/916 ms → 5 次/299 ms，剩下的都是
  特化推断不同（同上面 18% 的 CS）。游戏右上角显示进度（"后台准备着色器 xx%"），完成后 3 s 消失。
- **启动进度**：启动时载入静态管线缓存、预热着色器和管线期间，窗口里显示文字和进度条（Windows 用 GDI 画在
  Vulkan 首次呈现前的窗口上，其他平台只改标题），并处理窗口消息，窗口不再"未响应"。
- 为什么编译慢（已量化，尚未修）：
  1. PS5 的 CS 全是 wave64，NVIDIA 子组 32 宽，翻译器把**每条指令发两遍**（含全 wave 一致的标量指令）→ 编译慢 2.3 倍；
  2. 便携格式解码：每个格式化 buffer 读取的每个分量内联约 3 个嵌套 `switch` → 有 300 个这种读取的 shader 变成
     4.7–9.5 MB SPIR-V，单条管线编译最长 107 s（0.6% 的管线占 29% 编译时间）；
  3. EXEC 掩码模拟：一个大 shader 约 4000 个额外 select、1.1 万个类型转换，真正的运算才约 3000 条。

## 5. 下一步候选（附预估）

- 渲染线程：剔除用 CS 按连续同程序批处理（约 1.6–2.2 ms/帧）；约 300 个 "hero" draw 用快照记录（0.6–1.35 ms）；
  每次 draw/dispatch 的 shader 代码哈希记忆化（约 0.3 ms，需要代码页写跟踪）。
- 编译卡顿：补静态缓存的特化覆盖（上面 18% 的 CS：整数纹理、cube/2D、单通道 swizzle，命中就直接是优化版，
  预翻译也随之覆盖）；把预翻译的结果（SPIR-V + 资源计划 + 程序信息）持久化成静态文件，省掉每次启动约 560 CPU·s
  的后台翻译和 1.2 GB 内存（需要序列化 IR 资源计划）；翻译器本身提速（SSA 提前封口等）。
- 编译/GPU：wave64 的一致标量指令只发一次；精简 EXEC 预测。这些会改变 SPIR-V，改完要重跑静态预编译。
  （"少见格式放共享 `DontInline` 函数"已试过：编译总量不变，且让 NVIDIA 编译器崩溃，已删除。）
- Windows 显示路径：目前为规避 NVIDIA 驱动死锁的措施约 −1.5%，可考虑经 DXGI 交换链显示。
- 不建议：换 DX12（瓶颈不在 API，且 DX12 没有着色器指针，PS5 访存要大改）；Vulkan 管线库/动态状态（实际游玩中几乎每对 VS/PS 只有一种状态，收益约 3%）。
- 09-29 夜另一个代理试了原生阴影/剔除/几何通道、命令合包等十余种方案，同进程 A/B 全部持平或回退（最差 −35%），
  代码已于 09-30 全部删除，结果与原因见 `docs/EXPERIMENTS.md` 末节，避免重复。
- 已知问题：约 1/20 次运行 red-zone 相关偶发崩溃；"灵魂出窍"（角色模型停住、玩家隐形）09-28 复现过，疑似未跟踪的写，已修几个可疑路径，待确认。

## 6. 工作规则（用户要求）

- 只做能说清"去掉了什么工作/等待"的逻辑优化，不做按场景调阈值的"玄学调参"；无效的实验代码要删掉。
- 测量：同进程 A/B（`ab-spot.ps1` 切换临时开关），多个场景（固定场景 + 步行路线），配截图核对画面。
- 保护用户存档：测前 `python tools\local\bench-windows.py prepare`，测后 `restore`（逐文件 SHA256 核对）。
- 同一时间只开一个模拟器实例；不抢用户窗口焦点；启动超过 1 分钟即异常。
- 未经允许不提交；NVIDIA Streamline 等二进制不得提交或分发。

## 7. 常用命令

```powershell
.\build-windows.cmd [target]                  # clang-cl + Ninja，默认 kyty_emulator；kyty_shader_precompile 为预编译程序
.\run-windows.ps1                             # 按 run-windows.json 配置运行（-Precompile 预编译已录制的 shader；-Set KEY=VALUE 覆盖开关）
.\precompile-windows.ps1                      # 全量静态预编译（多进程，低优先级，可续跑）
tools\local\windows\bench-run.ps1 -Label x    # 固定场景 3×20 s 测帧率（自动装基准存档、识别 HUD）
tools\local\windows\walk-run.ps1 -Plan "w:down:0,h:down:4000,h:up:4500,w:up:25000"   # 步行路线
tools\local\windows\ab-spot.ps1 / prof-spot.ps1   # 同进程 A/B / 渲染线程采样剖析
```

## 8. 详细文档

`docs/CURRENT-PLAYABLE.md`（运行方式与 Windows 专有问题）、`docs/EXPERIMENTS.md`（实验记录）、`docs/PERF-40FPS-*.md`、
`docs/RENDER-SPLIT-NOTES.md`、`docs/BENCHMARKING.md`、`tools/local/README.md`、`tools/local/static-precompile/README.md`。
