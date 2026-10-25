# 性能实验简记

更新：2026-09-20。平台 i9-14900K / RTX 5090，游戏《恶魔之魂》PPSA01341。
固定场景：固定存档 Continue → 前进 10 秒 → 停止并稳定 → 2560×1440 静止测量。
当前正式版约 22～23 FPS，30/60 FPS 均未达到。启动见 [CURRENT-PLAYABLE.md](CURRENT-PLAYABLE.md)，复测见 [BENCHMARKING.md](BENCHMARKING.md)。

以下数值属于各自实验，不能跨会话拼接推算收益。历史长文、重复结论、旧二进制、抓取文件和截图已清理；仅本轮 PBR 留少量机器可读凭据及回退文件。

## 最新：完整 PBR 准备路径（2026-09-20）

实现跨布局资源定位复用、颜色/深度附件描述复用、完整动态状态输入复用、native descriptor template，以及按完整新鲜字节比较的上传切片复用。保留当次 SRT 求值、内容更新、附件清除、同步和真实绘制。

| 同进程交替测试 | 原路径 FPS | 完整候选 FPS | 变化 |
|---|---:|---:|---:|
| 6 × 20 秒，0/1/1/0/1/0 | 22.438 | 21.657 | −3.48% |
| 5 × 12 秒，0/1/0/1/0 | 22.551 | 22.023 | −2.34% |

结论：没有净收益，每帧反而增加约 1.1～1.6 ms；没有接近 30 FPS。高命中率不等于消除了整段 15～20 ms 工作，新增检查、哈希、比较和副本维护抵消了部分查找/提交节省。

- 静止及转动视角共 24,002,030 次资源/附件/状态结果对照，差异 0；候选画面抽查正常。不是全游戏正确性证明，也不包含 descriptor template 的逐项对照。
- 诊断窗口命中：纹理 91.02%、buffer 92.87%、颜色附件 35.79%、深度附件 69.75%、状态 66.69%、小块上传 62.32%、SRT/常量 52.01%。正式 FPS 窗口关闭实验计数。
- 首版 22.605 → 12.766 FPS：直接读取非 CPU cached 的 Vulkan 映射进行比较导致降速；改成普通内存副本后重新验证。首版全寄存器状态键零命中，修正为状态写入器实际使用的完整输入集合。
- 子项拆分测试首尾参考漂移 6.9%，不据此给子项排名。转镜后的轻场景超过 30 FPS 不算原重场景提升。
- 8 秒 CPU 采样提示：FindImage、CommitBindings 自身占比下降，memcmp、短块哈希占比上升；不能把采样百分比直接相减成绝对毫秒。
- 两次游戏会话均正常关闭，34 个原存档文件逐文件 SHA256 一致；正式二进制、配置、AOT 和 `run.sh` 未改动。

原型源码（`src/local/pbr-full.*` 及渲染器条件编译接入处；编译选项 `KYTY_LOCAL_PBR_FULL`、运行开关均默认关闭）未部署到正式版，已在 2026-09-25 历史整理中删除，保留在分支 `experiment/render-preparation-20260912`（提交 `4134ca6`）。运行模式：0 原路径、1 候选无计数、2 原路径对照、3 候选计数；mask：资源 1、附件 2、状态 4、descriptor 8、上传 16，完整为 31。

最小凭据在 [`_Build/experiment-records/pbr-full-20260920/`](../_Build/experiment-records/pbr-full-20260920/)：`results.json`、构建/源码指纹、跟踪文件补丁及修改前文件。测试二进制和大附件已删除；按构建收据重建用的 `tools/local/build-pbr-full-overlay.py` 已于 2026-09-28 移除，需要复测时从上述分支构建。

## 2026-09-25 路线 A 上限：重复 draw 快路径（`src/local/draw-delta-ceiling.h`，默认关闭）

`kyty_local_draw_delta_mode`（0 关 / 1 丢弃干净 draw / 2 只计数），`LIVE_DELTA_DRAW` 输出。
"干净"= 自上一个 DRAW_INDEX_INDIRECT 以来，除图形 user-data SGPR（XPR 几何根 R）、索引状态、间接偏移外，
CX/SH/UC 寄存器**值**均未改变（影子寄存器逐值比较）；dispatch、同步包、索引包不算改变。依据 ds_decompile 活动 T5/T7。

| 同进程交替（官方场景，4 × 10 秒） | 正常 | 丢弃干净 draw（上限，画面错误） |
|---|---:|---:|
| FPS | 24.92 | **30.80**（28.9～31.6） |
| 渲染线程 CPU | 33.59 ms | **26.31 ms（−7.3 ms）** |
| 录制线程 CPU | 8.85 ms | 6.94 ms |

- 每帧约 6,790 个 indexed indirect draw（计数模式下被 draw-run 批处理吞掉一部分，显示 4,190），干净 5,440（80%），
  1,340 个因寄存器值真实改变（换 shader/材质）而不干净；没有任何其他包打断间隙。约 650 个干净 draw 前索引状态变过。
- 这是**上限**：真实快路径仍需每 draw 绑定按 R 缓存的资源、必要时重开渲染（每个 draw 之间有 cull dispatch）和记录 draw。
  上限比 A1（<1 ms）大一个数量级，值得实现；需先测"保留不可省部分"的现实上限。

### 路线 A 现实上限（模式 3）

干净且索引状态未变的 draw 只做真实快路径保留的工作：重开上一 draw 的渲染、提交上一 draw 绑定的副本
（图像布局检查、描述符推送、push constants）、在 buffer cache 查 GPU 写的间接参数、记录 vkCmdDrawIndexedIndirect。
资源仍是上一 draw 的，画面错误。`src/local/repeat-draw.inc`。

| 同进程交替（官方场景，4 × 10 秒） | 正常 | 模式 3 |
|---|---:|---:|
| FPS | 25.08 | **28.82**（28.6～29.2） |
| 渲染线程 CPU | 33.43 ms | **28.24 ms（−5.2 ms）** |

- 候选中 99.8% 走通快路径（每帧约 4,060 个）；另有约 650 个干净 draw 因索引状态变化未覆盖。
- 观察器自身（逐值影子寄存器）约 0.25 ms/帧；ObtainBuffer 查间接参数、CommitBindings 副本为主要剩余开销。
- 结论：正确实现（按 R 缓存资源）预计 28.5～29.5 FPS；再覆盖索引变化的 650 个、把观察器并入寄存器写入口，才有望到 30。

### 路线 A 正式实现第一阶段（模式 4）：只复用状态，结果无收益

复用上一 draw 的渲染目标及其布局、pipeline、动态状态；每个 draw 照常做 SRT 求值与选 shader（RefreshShaders）、
资源准备、绑定提交，并按 CPU 读出的参数绑定索引区间、直接 draw（`repeat-draw.inc` 的 `RepeatDrawFast`）。

- 首版画面错误：GPU 间接 draw 复用了普通路径按 `base + start_index` 绑定的索引缓冲，再叠加参数里的 `start_index`，偏移算了两次。
  改为与普通路径一致后，开/关截图一致（仅植被动画差异）。
- 性能：候选 95% 走通快路径，但 23.5 FPS / 36.3 ms，对照约 25 FPS / 33.5 ms——**更慢**。
- 结论：每 draw 的"状态"工作（目标、pipeline、动态状态）很便宜；模式 3 的 5.2 ms 几乎全部来自跳过
  **SRT 求值 + 资源准备**。收益只能来自第二阶段：按几何根 R 缓存 SRT 求值结果与准备好的绑定，并以低成本证明其仍有效。

### 路线 A 第二阶段：按几何根缓存 SRT 求值与绑定（模式 5/6），正确但无净收益

跨帧缓存（键 = 程序对 + 各阶段 user-data），逐项证明有效：SRT 读取的每个字按原读法重读比对；映射纪元不变；
缓冲仍由同一活跃游戏缓冲持有且未被 CPU 改写（否则该槽走 ObtainBuffer 重解析并回填对齐偏移）；
texel 缓冲无重叠图像（纹理解析纪元变动时复查）；采样图像所有者仍注册且 backing 相同。
每帧仍做 BindImage/RebindImages、两次小上传、CommitBindings。

- 迭代中排除的障碍：格式化（texel）缓冲占满拒绝原因 → 加图像重叠守卫；备份别名纪元每次 mprotect 都变（~700/帧）→ 改为只查映射纪元；
  CPU 每帧改写的小缓冲走上传环 → 按槽每 draw 重解析；一个全局时间常量（`0x2088830c0`，递增浮点）每帧使所有条目失效 →
  学习"只落到 flattened SRT 的常量字"并直接打补丁。
- 最终：命中 2,650/帧（69%），其中 645 次靠补丁；校验模式快照差异 0，唯一的绑定"差异"是校验比较了尚未重解析的 CPU 改写槽。
- 性能：35.35 ms / 24.0 FPS，对照 33.26 ms / 25.17 FPS——**仍慢约 2 ms**。
- 剖析差分：省下的（纹理发现、目标解析、pipeline 查找、采样器等）约 1.6 ms；新增的分散在拷贝记忆状态与缓存条目
  （memmove/new）、寄存器观察器、参数 CPU 读取、索引区间 ObtainBuffer、每 draw 图像刷新与上传、校验。
- 结论：模式 3 的 5.2 ms 上限是在不做任何正确性工作的前提下得到的；按正确性要求补回的每 draw 工作
  （索引、图像内容刷新、CPU 改写缓冲、常量、参数）本身就占原路径成本的大头。在"逐 draw 翻译"架构内，路线 A 不能带来净收益。
  代码（`kyty_local_draw_delta_mode` 4/5/6，默认关闭）已在 2026-09-25 历史整理中删除，保留在分支 `experiment/render-preparation-20260912`（提交 `4134ca6`）。

## 2026-09-25 路线 B 上限：原生 GPU 驱动 XPR 通道（`src/local/route-b-ceiling.h`，默认关闭）

`kyty_local_route_b_mode`：7 丢弃深度预处理 XPR draw（VS 7833dd/daa847 + PS 0a206f，按渲染器翻译时学到的代码地址识别）
及其间的 XPR 剔除 dispatch；8 丢弃全部 XPR 剔除 dispatch 与跟在剔除后的 XPR draw。画面错误（静态世界消失），只测主机侧上限。

| 同进程交替（官方场景，3 轮 × 10 秒） | FPS | 渲染线程 CPU | 录制线程 CPU |
|---|---:|---:|---:|
| 正常 | 24.83 | 33.96 ms | 8.55 ms |
| 7：只去掉深度预处理（每帧约 3,470 个 draw） | 26.57 | 31.33 ms（−2.6 ms） | 7.58 ms |
| 8：去掉全部 XPR 剔除（每帧约 3,020 个 dispatch）及 XPR draw | **38.67** | **19.83 ms（−14.1 ms）** | 4.99 ms |

- 深度预处理 draw 之间没有剔除 dispatch（模式 7 丢弃剔除数为 0），深度通道复用其他通道的剔除结果。
- 模式 8 实际只命中了剔除 dispatch（draw 计数为 0：剔除可能在另一条命令流上，打断了"剔除后 draw"的判定）；
  剔除被丢弃后 XPR draw 读到的参数为 0 或旧值，大多在普通路径开头以 0 计数提前返回。所以 8 的数字是"没有剔除 + XPR draw 基本不做事"。
- 结论：只做深度预处理 +1.7 FPS，不够；覆盖全部 XPR 通道（剔除 + draw）的上限是 38.7 FPS，余量足够达到 30。

### 路线 B：XPR 的 GPU 耗时与剔除/draw 拆分

模式 9：只丢弃参数位于 XPR 剔除输出范围内的 indexed indirect draw（每帧约 5,316 个），保留全部剔除。
GPU 忙碌时间用 `nvidia-smi` 利用率 × 帧时间估算（粗略，利用率是"有内核在运行"的时间比例）。

| 同进程（2 轮 × 8 秒） | FPS | 渲染 CPU | 录制 CPU | GPU 利用率 | ≈GPU 忙/帧 |
|---|---:|---:|---:|---:|---:|
| 正常 | 24.75 | 33.25 ms | 8.61 ms | 47% | ≈19 ms |
| 8：去掉 XPR 剔除（draw 近乎空转） | 38.9 | 19.1 ms | 5.1 ms | 62% | ≈16 ms |
| 9：只去掉 XPR draw，保留剔除 | 33.1 | 23.9 ms | 6.6 ms | 57% | ≈17 ms |

- 拆分：XPR draw 约 9.3 ms 渲染 CPU，剔除约 5 ms。
- GPU：整帧约 19 ms，XPR 约占 2～3 ms；距 30 FPS 预算 33 ms 余量很大，GPU 不会成为瓶颈。
- 推论：只把 draw 做成原生 GPU 驱动，上限就有 33 FPS；再做剔除，上限 38～39 FPS。

### 原生 draw 里程碑 1（模式 10）：GPU 读参数 + 批次绑定索引池 + 只处理命中，画面正确但未提速

`kyty_local_draw_delta_mode` 10：缓存未命中走普通路径并由普通路径发布条目（带 SRT 读取日志与补丁学习）；
命中时 GPU 直接读间接参数（不再 CPU 回读），索引池按 base 每命令缓冲绑定一次（GPU 使用参数里的 start_index），
图像内容每帧每图至多刷新一次；描述符提交与两次小上传仍每 draw 进行。

- 画面与关闭时一致。
- 23.2 FPS / 36.6 ms，命中每帧约 450（查找 1,880：只有"寄存器干净且索引未变"的 draw 进入）；XPR draw 每帧约 5,300。
- 与模式 5 合看：即使命中 2,650/帧，每次命中仍需校验、按槽重解析、图像记账、两次上传、描述符提交、开渲染与记录，
  每 draw 约 1 µs 量级，接近普通路径每 draw 约 1.75 µs 的大半。
- 结论：只要 XPR 仍是"每个对象一个 Vulkan draw + 每 draw 一次绑定提交"，就到不了 30。
  原生 draw 必须改成每通道/材质一次 multi-draw，每 draw 的资源由 GPU 端表按 draw 序号取用
  （着色器翻译需支持：按 draw 序号取 user data/R，经 BDA 读 R 链与缓冲，图像走描述符索引）。

### 原生 draw 预算：模式 11（不做绑定提交的下限）

模式 11 = 模式 3 去掉 CommitBindings（只重开渲染、查间接参数、记录 draw），图像错误，测每 draw 的下限。

| 同进程交替（3 轮 × 10 秒，本次启动基线偏低） | FPS | 渲染 CPU |
|---|---:|---:|
| 正常 | 22.97 | 35.88 ms |
| 模式 3（含绑定提交） | 26.00 | 30.83 ms |
| 模式 11（无绑定提交） | 26.43 | 30.19 ms |

- 绑定提交每 draw 只约 0.16 µs（约 4,000 个重复 draw 共 0.64 ms），不是瓶颈。
- 不绑定时每个重复 draw 仍约 0.35 µs：cull dispatch 之间每 draw 都要重开渲染、逐 draw 查参数、记录 draw。
- 预算：30 FPS 约对应渲染 27.3 ms，XPR draw 总成本须 ≤ 约 3.4 ms（约 0.64 µs/draw）。重复 draw 按每对象描述符集约 0.6 µs，
  而约 1,340 个链头（换材质）仍走普通路径约 1.8 µs → 合计约 4.8 ms，约 28～29 FPS。
- 结论：链头也要走原生路径（按寄存器状态哈希选 pipeline），并压低下限（参数按批次查找）；原生剔除（M6）可去掉每 draw 重开渲染，作为余量。

### 原生 XPR draw 里程碑 2～3（离线）：每对象描述符集 + 动态偏移，400/400 逐像素一致

`src/local/native-xpr.inc`（仅本地录制构建）。普通路径按请求在 CommitBindings 之后存两项：
draw state（渲染目标、附件布局、动态状态；键 = 全部图形寄存器，不含 user data 值）与对象记录
（键 = 着色器地址 + user data：原生 pipeline 变体、写一次的描述符集、常量）。原生 pipeline 变体与普通 pipeline 同一 SPIR-V，
只是描述符集非 push、flattened SRT 与 shader data 为动态存储缓冲（指向着色器上传环）。原生 draw：常量拷入上传环、
绑集（动态偏移）、push constants、按需切 pipeline/动态状态、GPU 读参数的 `vkCmdDrawIndexedIndirect`。

- 离线 A/B：`xpr_replay_tests <dir> --native --once`，每个捕获 draw 先普通路径（A），GPU 清目标回填充值，再原生路径（B）。
  步长捕获全部 400 个 draw（深度预处理、GBuffer、阴影）**A == B**（354 个有输出，46 个空）。
- 负对照（`KYTY_NATIVE_XPR_DEBUG`）：1 清零顶点阶段常量、2 跳过 draw，都在有输出的 draw 上报差异，说明比较与重置有效。
- 运行时（校验、失效、命令缓冲内状态去重）是 M4。

### 原生 XPR draw 里程碑 4（运行时，进行中）：发射很便宜，慢在每帧记录维护

`kyty_local_native_xpr_mode`（0 关 / 1 原生 / 2 校验，见下一节；本节的"上限"模式当时编号为 2：记录存后永不再校验，画面错误，仅测量，现已删除），`LIVE_NATIVE` 输出。
命令处理器把每个 DRAW_INDEX_INDIRECT 串（同一对象的连续包）交给原生路径；未命中时普通路径带 SRT 读取日志重存。
每帧每记录校验一次：SRT 读取字比对（学到的常量字直接打补丁，其余重跑 SRT 求值、结构不变只换常量）、
缓冲属主存活且 CPU 干净（区域纪元快速证明）、图像注册且非 CPU 脏；渲染目标每 draw 检查仍为 GPU 拥有、无待处理快速清除。

沿途修正（都来自游戏内统计/日志）：
- 渲染目标不经纹理缓存"获取"记账时，只被原生路径使用的目标会被 GC 回收，普通路径随后按客体内存重建一个 176 MB 的 PRT 图像而崩溃 → 每 draw 检查目标仍 GPU 拥有并 touch LRU，缓冲/图像每帧 touch。
- 每帧约 1,580 次"动态缓冲"拒存：普通路径对 CPU 脏页上的小只读范围每 draw 流式拷贝到上传环、从不同步回属主，页永远是脏的 → 存储时同步回属主一次（之后再被写会触发写保护并被证明发现），拒存降到约 13。
- texel 缓冲重叠图像的判据改成与普通路径相同（只有恰好从缓冲起点开始且 GPU 拥有的图像才会同步），拒存 800 → 0。
- SRT 字失效集中在每帧常量块（0x208883b20 每帧 +0xC000 像环形偏移，0x2088831c0 等浮点）；学习规则补上"变化但不影响任何产物的字可忽略"，求值成功时顺便学习补丁。

同进程交替（3 轮 × 8 秒，官方场景）：

| 模式 | FPS | 渲染 CPU |
|---|---:|---:|
| 0 关 | 25.00 | 33.31 ms |
| 1 原生（完整校验） | 22.79 | 37.10 ms |
| 上限（当时的 2，已删除；不维护，画面错误） | 29.16（26.75 → 29.37 → 31.37，覆盖逐轮增加） | 26.85 ms（末轮 25.44） |

- 模式 1 每帧：原生 draw 约 6,000，记录校验约 3,160，补丁约 1,380，重求值成功约 430，
  重求值失败（描述符变了）约 750 → 普通路径重存约 860。重存是最大开销（约 6 µs/次）。
- 描述符变化：约 88% 是图像（按帧轮换的纹理基址、流送纹理 T# 的 dword5/6 LOD 字段），约 10% 是缓冲（每帧前进 0x2000 的环形常量缓冲）；
  只有 25～43% 的新结构在该对象最近 8 个结构里出现过，描述符集变体缓存只能解决一小部分。
- 上限说明原生发射本身约 0.25 µs/draw，满足预算；达标取决于把每帧维护压到约 2 ms：
  重存改成"只重新解析变化的槽位"，校验按页变化/共享常量块集中处理（或移到辅助线程），必要时再做原生剔除（M6）。
- 上限模式最后因过期数据让游戏逻辑拿到错误 GPU 结果，普通路径读到非法采样器描述符而退出（仅诊断模式的预期风险）。

### 原生 XPR draw 里程碑 4 第二部分：画面正确（校验 0 差异），+1.4 FPS

- **校验模式**（`kyty_local_native_xpr_mode` 2；本节记录时编号为 3）：原生路径本要发射的 draw 改走普通路径，普通路径把自己准备出的程序、
  描述符快照、flattened 常量、shader data、图像视图、采样器、缓冲来源、渲染目标、动态状态逐项与记录比对。
  - 第一次运行每帧约 232 次不一致，全部是 flattened 常量（时间类常量旧一帧）：同一记录一帧内被多次使用，
    游戏在帧中途改写了每帧常量块（帧流水线下 CPU 提前准备下一帧），普通路径每次 draw 都读当时的值。
    → SRT 读取字改为**每次使用都比对**（尝试数只比校验数多约 10%），缓冲/图像证明仍按帧。
  - 修正后每帧约 3,550～3,690 次核对，**0 不一致**。
- **截图**：修正前开/关差异集中在天空/远景（雾与大气），修正后开/关差异落入同模式相隔 2 秒的动画噪声底
  （平均绝对差 1.85～2.2，噪声底 1.65～2.0）。
- **描述符变化不再走普通路径**：重求值发现结构变了时，用普通路径同样的逐阶段函数重新解析全部槽位、换新描述符集
  （相同则保留旧集），每帧重存从约 860 次降到约 120 次。
- **T# 的 LOD 统计字段**（MIN_LOD_WARN / MIP_STATS_CNT_EN / MIP_STATS_CNT_ID，游戏每帧轮换计数器）只影响 shader data 的
  LodStats 字，不影响图像解析：作为结构变化的例外，并学成"图像 LOD 字补丁"。
- 映射纪元变化改为只强制重做缓冲/图像证明（每次使用比对字之后，全局纪元会让帧中途所有记录失效）。

同进程交替（4 轮 × 8 秒）：关 **24.34 FPS / 34.10 ms**，原生 **25.71 FPS / 31.80 ms**（+1.37 FPS）。

分段计时（每帧，TSC 周期）：原生尝试共约 34 M（其中重求值约 16 M、重解析约 6 M、发射约 5.7 M、字比对约 4.3 M），存储约 3 M。
- 重求值每帧约 1,100 次：结构变化约 485（环形纹理：同一纹理 dword0 基址每帧前进 0xC000，环很长，只有约 53% 的结构在该记录
  最近 8 次里出现过）、学习因同值多位置放弃约 300、学成后下一帧又遇到新变化字约 330。
- 下一步：用线性 SRT 计划的节点图直接得到每个读取字的用途（只流向 flattened 的数据字每次直接收集、
  参与地址/描述符的结构字才比对），取代"比对 + 学补丁"；描述符变化做到只解析变化槽位。

### 原生 XPR draw 里程碑 4 第三部分：按线性 SRT 节点图区分数据字与结构字

- 新增 `TraceLinearSrtReads`（SrtWalker）：跑一次线性 SRT 计划，得到每个读取的地址与用途——只流向 flattened 的"数据字"
  （源地址）与"结构字"（喂地址、描述符 dword、计算值或控制流谓词；带用途位、对应的描述符 dword 与 flattened 副本位置）。
  支持控制流变体（谓词读取记为结构字）。离线：800 个捕获阶段中 649 个可跟踪，每个数据字/结构字/描述符 feed 都与求值结果一致。
- 原生记录每次使用：比对结构字；数据字直接按地址收集进 flattened；结构字变化若只喂描述符 dword，就直接补丁
  （T# LOD 统计字段 → 只更新 shader data；图像/缓冲基址 → 改快照后重解析换集；flattened 里的副本一并更新），
  否则才完整重求值。游戏内结构字变化 100% 是描述符字（环形纹理基址约 45%，其余多为 LOD 统计字段），补丁后重求值从每帧约 1,100 次降到个位数。
- 校验模式发现一处：T# 的 LOD 字段也被着色器当常量读进 flattened，补丁须同时改 flattened 副本；修正后 0 不一致（每帧约 3,400～4,500 次核对）。
- 记录按帧 LRU 回收（两帧未用），存了却从不复用的着色器对（user data 每帧变）停止存储；哈希表查找热点（缓存未命中）随之缓解，每帧存储从约 130 降到约 48。
- 同进程交替（3 轮 × 8 秒）：关 **24.79 FPS / 33.24 ms**，原生 **26.62 FPS / 30.12 ms**；另一会话关 23.54 / 原生 26.75。
- 剩余（每帧，TSC 周期）：原生尝试约 27 M（比对/收集约 6 M、重解析约 6.8 M、发射约 5.7 M、查找与状态等约 8 M），存储约 3 M。
  上限模式（不维护）渲染约 25.4 ms，达标需再省约 4 ms：环形纹理的重解析改为每帧解析一次、按槽更新集合；查找/发射路径精简；之后是原生剔除（M6）。

### 原生 XPR draw 里程碑 4 第四部分：记录维护按槽位、按变化字处理；冷数据预取

- **按槽重解析**：环形纹理（三缓冲，T# 基址在 3 个位置轮换）与缓冲基址变化只重解析变化的槽位，改存储的描述符载荷后换新集（全有或全无，
  避免"载荷已改、集未写"骗过完整重绑的同载荷判断）；完整重绑每帧约 90 → 0，每帧约 400 次图像、约 80 次缓冲按槽重绑。
- **CPU 改写的缓冲依赖就地重新同步**（与存储时相同的 `EnsureBufferContents` + `ObtainBuffer`），不再作废记录走普通路径再重存：
  每帧存储约 50 → 3～10，存储耗时 3.2 M → 0.2 M 周期。
- **跟踪收集改成页内连续段**：结构字与数据字按页合并成段（小空隙一并读），与记录"持有的字"逐段比较，只处理变化字；持有的字取自记录本身
  （描述符 dword、flattened），不是跟踪时再读一次内存——跟踪与求值之间客体改写的字下次必被看到（原先可能被跟踪值掩盖）。
  clean 读合并成段、失败逐字回退（与求值读法同结果）。
- **冷数据预取**：记录大多每帧只用一次，数据全在缓存外（每次使用约 15～25 处分散的堆分配各一次缺失）。先预取记录热字段所在行，
  状态查找后再预取各数组，收集前预取所有客体段的行。
- 发射：参数 `ObtainBuffer` 按连续组一次（每帧 7,650 → 4,100 次）；动态状态与已绑定的相同时跳过（每帧 1,310 → 300 次）；LRU 每 8 帧扫一次。
- 纹理缓存（语义不变的提速）：`FindImageFromRange` 加"起始地址精确索引"，候选只查起始字节；新增"新起始地址日志"，
  texel 缓冲检查改为与日志比对（原先每记录每帧上锁查询，锁被客体缺页处理 `InvalidateMemory` 争用）；
  `FindDepthTarget` 只在元数据真正变化时推进元数据纪元（原先每次深度目标获取都推进，原生路径每次发射都要上锁复查快清）。
- 正确性：校验模式 0 不一致（每帧约 3,600～3,900 次核对）；截图开/关差异与同模式间隔 4 秒的动画噪声一致（>48 的像素约 1,050～1,190）。
- 同进程交替（4 × 10 秒）：稳定会话 关 **24.23 / 33.61 ms**，原生 **28.95 FPS / 26.56 ms**（各轮 28.7～29.1）；
  另两会话 CPU 过热降频（封装 79～89 °C，两组都降约 7%）：关 23.45 / 原生 28.18、关 23.07 / 原生 28.40。单次 8 秒原生 29.25～29.62 FPS。
- 离 30 FPS：渲染线程每帧仍有约 8 ms 空闲（off-CPU：约 2.7 ms 等客体提交、约 1.5 ms 普通计算 dispatch 的 `DrainGuestReadback` 等 GPU 回读），
  30 FPS 需渲染 ≤ 约 25.3 ms（现 26.5）。原生尝试约 21 M 周期（收集 6.2 含重绑 2.4、资源证明 3.4、发射 4.7）。

### 原生 XPR draw 里程碑 5：官方场景 ≥ 30 FPS（目标达成）

- 描述符集变体：记录保留至多 3 个"只在图像上不同"的旧集，图像签名（去掉 LOD 统计字段）命中且各图像仍当前、CPU 干净时直接切回
  （每帧约 390 次），新建集每帧 ~500 → ~100；缓冲变化或完整重绑时丢弃变体。
- 记录查找改开放寻址索引（一次缓存行），键的前 36 个字内联比较（精确；更长的键其余部分再比向量）；逐字数组一并预取。
- 同进程交替：4 × 10 秒 关 25.15 / 原生 **31.10 FPS**；6 × 10 秒 关 24.85 / 原生 **30.93 FPS**（每轮 30.7～31.1，渲染 33.97 → 25.85 ms）。
- 验收：离线 400/400 逐像素一致；官方机位与另 2 个机位截图开/关差异在动画噪声内，校验模式 0 不一致；
  10 分钟脚本游玩（`tools/local/native-xpr-playtest.py play 10`）无崩溃、无错误日志，存档还原。

## 2026-09-25 A1：按程序复用上一次调用的阶段绑定（默认关闭）

`src/local/stage-reuse.inc`，开关 `kyty_local_stage_reuse_mode`（0 关 / 1 复用 / 2 校验），`live-bench.py sym` 实时切换，`measure` 输出 `LIVE_REUSE` 计数。
同一编译程序与上一次调用比较快照：图像走 `TryReuseSampledImage` 同一证明，采样器按原始描述符，缓冲整阶段复用（沿用 XPR culling 快路径已校验的纪元守卫），上传区复用限同一 tick。

| 同进程交替（官方场景，4 × 10 秒） | 关 | 开 |
|---|---:|---:|
| 首版（线性查找 + 原子计数） | 24.02 FPS / 33.90 ms | 23.07 / 35.77 |
| 只拷贝变化部分 | 23.93 / 34.73 | 23.05 / 36.50 |
| 直接映射表 + 单写者计数 + 无锁 MappingEpoch | 23.95 / 34.10 | 23.55 / 34.98 |

- 正确性：校验模式每帧约 3,425 个阶段的缓冲复用、7,620 个图像复用，逐项对照差异 0。
- 每帧约 10,100 次阶段准备，快照整体相同约 3,340，缓冲复用约 3,830，图像复用约 8,260。
- 采样差分：跳过的工作实际只省约 1 ms/帧（PrepareBindingsInto 0.25、ResolveTexture 0.15、FindBuffer 0.09、GetSampler 0.07 等），而簿记自身约 0.77 ms，且分散在查表与冷数据比较上，无单一热点。
- 基线中所有可跳过类别合计约 5.9 ms/帧，但大部分不在"重复 draw"路径上。即使簿记为零，A1 上限也不足 1 ms。
- 结论：**净负（−0.4 FPS），不采用**，与 09-20 完整 PBR 准备路径、诚实 buffer memo 的教训一致：命中率高不等于可省成本大。
  `GpuResourceManager::MappingEpoch` 改为无锁原子读（写仍持锁），语义不变，保留。

## 历史实验

| 日期 | 实验 | 最简结果与结论 |
|---|---|---|
| 09-12～18 | 配置与基础优化 | 保留已验证的组合优化，历史合计约省 2.8 ms/帧；当前正式配置仍约 22～23 FPS。 |
| 09-12～18 | 陈旧绑定/跳过资源准备的天花板 | 图形盲跳曾 22.657 → 26.128 FPS，但跳过内容更新，不是正确优化；激进 compute 复用曾崩溃/挂起。 |
| 09-12～18 | 诚实 buffer memo、stream memo | buffer 复用簿记约 2.006 ms，节省约 1.05 ms；stream memo 无稳定收益。地址/dirty/映射 epoch 都不能单独证明内容不变。 |
| 09-12～18 | 页查询与位图微优化 | 页查询合并上限约 0.244 ms/帧；位图检查已很便宜，不是大幅提速入口。 |
| 09-12～18 | 更换渲染核心、客体忙等退避 | CPU 0 换到 CPU 6 没更快；客体退避降 CPU 占用但不提 FPS。保留 renderer CPU 0。 |
| 09-12～18 | 外部宿主内存零拷贝别名 | 独立正确性/带宽测试通过，游戏内图像分配报 OOM；已撤回，未形成可用路径。 |
| 09-12～18 | 单一 HLE/AGC 钩子、Draw/SRT 叶子微优化 | AGC 主要打包 PM4，缺材质/pass 语义；SRT 合计曾约 5.292 ms，局部叶子不足填补整帧缺口。 |
| 09-12～18 | 降 LOD 回读频率、纹理/I/O/解压等 | 无可靠 FPS 提升，部分更慢；不采用。 |
| 09-18～19 | shader/pipeline warmup | 已收集区域的预热收敛，常态无持续 shader JIT 证据；扩大同一预热集合不是稳态 FPS 解法。 |
| 09-18～19 | `VK_EXT_shader_object` | 冷预热 77.1 vs 80.9 秒近似；热态 pipeline cache 1358 ms 优于 SO 3582 ms；SO draw 路径片头错误、无法正常进入游戏。原型已移除。 |
| 09-19 | 地址空间读缓存、readback/批量/descriptor 等十项开关复测 | CPU/帧仪器噪声约 0.5 ms；十项差值在 −0.27～+0.43 ms，无可分辨净收益。包括 unique_range_cache、guest_readback_guards、stream_batch、draw_run_precheck、specialization_memo、descriptor_plan、bda_ranges、direct_multi_draw、image_query、lod_copy_feedback。 |
| 09-19 | mprotect 与拷贝归因 | 曾测 mprotect 603 次/帧、2.43 ms；小拷贝约 10,900 次、11.1 MB、3.14 ms。说明固定调用/保护成本重要，不能只按总字节判断。旧“mprotect 近乎为零”的口径不完整。 |
| 09-19 | 720p 对照 1440p | renderer CPU 约 38.8～39.8 vs 39.8～40.5 ms，主机成本没有随像素数成比例下降；该场景主要 CPU 受限，不代表 GPU 永远无瓶颈。 |
| 09-19 | 并行依赖审计 | 快照/队列、提前求值、完整后台准备、Vulkan 录制均曾尝试；输入版本、GPU 生产者与共享缓存所有权是主要难点。旧早期双线程曾退化至 6.68 FPS 并死锁，不代表所有并行方案都不可能。 |
| 09-19 | Intel PT 单帧捕获 | 基线 22.538 FPS；选取 48.596 ms 窗口，但丢轨迹、匿名 JIT 缺代码、解码不匹配；没有获得完整可离线执行的 CPU 帧，不能用污染的排行推断收益。 |
| 09-19 | SRT 捕获/离线回放 | 3 个呈现间隔、37,033 次调用、573 个计划、531,394 次读取；100 轮逐项回放一致。仅覆盖资源求值接口，未来输入已预先提供，不是整帧回放。 |
| 09-19 | SRT 静态依赖与离线并行 | 样本读取地址依赖最多两层；热点 19 次读取可条件性合并为 9 次。37,033 调用整体 1/2/4/8 worker 中位 7.789/4.094/2.688/2.193 ms；不是每帧耗时，更不是实时 FPS 收益。 |
| 09-19 | 在线合并/后台准备 | 严格热点合并 21.923 → 21.715 FPS；后台 + 合并 20.418 FPS；主动页跟踪后合并 20.361、后台 18.615，均慢于约 21.5～21.9 参考。证明/保护/投递/提交成本抵消并行收益；已回退。 |
| 09-19 | GPU 直接消费间接绘制参数 | 22.1384 → 22.1387 FPS，视为无收益。累计 7,432 次 GPU 转换、错误 0；仅覆盖观察参数，不代表完整 GPU 往返已消除。已回退。 |
| 09-20 | 单一资源族准入 | 全 PBR 约 17.13 ms，但 307 shader、222 配对、138 资源形状；最热同模板仅 4.09 ms，XPR culling 5.88 ms、decal 2.03 ms。单一同布局族不足 15～20 ms。 |
| 09-20 | 跨布局 PBR 成本/复用观测 | PBR 图形总计 18.511 ms：资源核心 11.028、附件 3.373、顶点索引 0.670、pipeline 0.621、draw 0.351、其余 2.467。buffer 身份重复 95.70%、image 94.13%、sampler 99.89%，但 flat SRT 62.22% 改变；不能复用整套旧绑定。成本覆盖不等于可净省成本。 |

## 2026-09-22 渲染层级开关与逐阶段归因

补齐引擎注册但未绑定的渲染开关（阴影、灯光、环境遮蔽、虚拟几何、DoF 等 10 个），并测了两个"地板"。
场景是 `demons-souls-layer-bench.py` 的出生点静止窗口（`--walk-forward-seconds 0`，基线 29 FPS /
28.98 ms），**比本文其余数据用的官方场景（前进 10 秒、22～23 FPS / 44.4 ms）轻约 30%，不能直接相减**。

| 方案 | 结果 | 是否保留 |
|---|---|---|
| 关阴影（`r_enableCascadeShadows`/`SpotShadows`/`contactShadows`） | 28.98 → 24.03 ms，省 4.95 ms | 仅作定位，不是优化手段 |
| 关灯光（`r_enableAllLights`） | 28.98 → 24.85 ms，省 4.13 ms | 同上 |
| 关全部 16 层 | 14.27 ms | 同上 |
| **隐藏整个场景**（`drawJustParticles`） | **11.11 ms** | **上界结论：约 11 ms/帧与画面内容无关** |

逐阶段归因（`render-cost-report.py --last-blocks 30 --per-block --groups`，本轮为此加了这三个选项）
**必须扣除插桩开销**：带插桩 38.36 ms vs 不带 28.98 ms，插桩占 9.38 ms（+32%），
136,325 个作用域 × 约 69 ns。扣除后资源绑定 9.29 ms(32.1%)、提交同步 8.88 ms(30.6%)、
录制命令 8.78 ms(30.3%)——三者平分，**不存在单一热点**。未扣除时 `ObtainBuffer` 读数 3.34 ms，
其中 2.06 ms 是计时器自己，真实约 1.29 ms。

失败一次：据未扣除的归因提出"优先优化每 draw 绑定开销 / 合批提交"，
对照本文后发现两条都已被 2026-09-20 完整 PBR 准备路径（−3.48% FPS）和 09-19 十项开关复测
（−0.27～+0.43 ms）否定，且"图形盲跳"给出的作弊天花板只有 5.87 ms < 30fps 所需的 11.1 ms。
原因是把"成本覆盖"当成了"可省成本"，正是本文既有教训。详见
[渲染层级文档第十节](compatibility/demons-souls-layers.md)。

## 2026-09-22 游戏逻辑是否依赖 GPU 产出

用 `LOCAL_READBACK` 页故障记录判定数据流方向（本配置 `guest_readback_guards_mode=0`，
所以客体线程的读一定会被记录；`TryReadCopyFeedback` 只在故障路径内部省下载，不掩盖故障）。

两次完整会话（启动+菜单+载入+游玩）共 48,703 次跟踪故障：
**客体线程读 GPU 写过的内存 = 0 次**，客体侧故障全是写（36,533 次）；
所有读（12,170 次）都发生在模拟器自己的 GPU 线程上。

结论：**游戏逻辑 → GPU 是单向的**，CPU 层与渲染层在数据流上可以解耦，headless 运行架构上成立。
剩下的耦合是 (a) 渲染器自身每帧 4.1 次 GPU→CPU 回读——**每帧只需 104 字节却下载 660 KB，
放大约 6,300 倍，停顿 1.13 ms**；(b) 客体对 GPU 的同步等待（WAIT_REG_MEM / EOP 标签、忙等）。

这给 09-19「并行依赖审计」里"GPU 生产者"那条难点一个具体形状：挡住拆分的不是游戏读结果，
是渲染器自己的回读。先收掉那个 6,300 倍放大，再谈两层并行。
边界：事件驱动的回读（遮挡查询等）未被这两个场景覆盖，不是全游戏证明。详见
[渲染层级文档第十二节](compatibility/demons-souls-layers.md)。

## 2026-09-22 headless 模式:CPU 层的上限

在 `graphicsRun.cpp` 加运行时开关 `kyty_local_headless_mode`(默认 0,关闭时只多一次 relaxed load),
为 1 时四个"客体 → 渲染器"入口直接返回;PM4 解析、EOP 标签、`WAIT_REG_MEM`、GDS、中断、flip 全部保留,
所以客体状态机不变(依据见上一条:客体从不读 GPU 写过的内存)。
开关可由 `KYTY_HEADLESS` 或每次 flip 轮询的 `KYTY_HEADLESS_FILE` 控制——后者是必须的,
因为 `switch-control.py` 需要 ptrace 而本机 `ptrace_scope=1`,且 Continue 自动化靠看屏幕。

同进程 A/B/A(固定存档机位,240 Hz vblank + Immediate,否则撞 60 上限),两次独立复现:

| 阶段 | FPS | ms/帧 | 进程 CPU | 客体主线程 | `Kyty.Gpu` |
|---|---|---|---|---|---|
| A 渲染开 | 29.4 / 28.7 | 34.0 | 3.65 核 | 100% | 85%(28.8 ms/帧) |
| **B headless** | **96.1 / 99.6** | **10.4** | 5.60 核 | 100% | 74%(7.7 ms/帧) |
| A 渲染开 | 29.4 / 27.4 | 34.0 | 3.66 核 | 100% | 84% |

结论:**CPU 层只要 10.4 ms/帧(96～100 FPS),不到 30 FPS 预算的三分之一,瓶颈完全在渲染层。**
客体主线程两种模式都钉在 100% 一个核而每帧 CPU 从 34 ms 掉到 10.4 ms,说明它大部分在自旋等渲染器。
渲染线程即使 headless 也要 7.7 ms/帧(PM4/标签/flip/GC),渲染再加约 21 ms。

两层已经部分重叠(28.8 + 10.4 = 39.2 > 整帧 34.0,约 5 ms)。完美解耦上限
= max(10.4, 28.8) = 28.8 ms = **34.7 FPS**(本轻场景够 30,官方重场景按渲染线程约 40 ms 估算仍只有约 25 FPS)。
**所以解耦本身不足以达标,还需要渲染线程自己再降约 18%。** 真正的余量在于整机只用了 3.65～5.60/24 核
而渲染层是单个饱和线程——方向是把渲染层并行化,阻碍是它自身每帧 4.1 次 GPU→CPU 回读。
正式冻结二进制未改动(已校验 `78a0d08f…`),实验构建在 `_Build/agent30fps-20260917/headless/`。

## 2026-09-23 更正：游戏确实读 GPU 结果；真正的耦合是约 73 个同步点/帧

**撤回 09-22 "客体从不读 GPU 写过的内存"。** 那是插桩盲区：`KYTY_FRAME_PIPELINE=2` 时
`GuestReadbackMode()` 为 6，客体读走未被记录的异步 `BeginGuestReadback` 路径。补记录后官方场景每帧：
客体读 GPU 内存 62.5 次（下 539 KB）、客体写 GPU 已改页 8.1 次（下 2.9 MB）、渲染线程读 3.1 次（下 661 KB）。
依赖是结构性的：让读取不等 GPU 的实验开关一开，客体在 `0xa2e820` 空指针崩溃（开关已删）。

同进程官方场景 A/B/A：渲染开 22.3/23.0 FPS、headless 107.7 FPS。渲染线程 39.3 ms/帧 CPU，
off-CPU 约 5.1 ms/帧（`DrainGuestReadback` 1.7、队列阻塞轮询 1.6、驱动 poll 1.0）。
按帧年龄统计客体读：本帧 30.5%、上一帧 29.3%、两帧前 39.2%；且客体与渲染线程锁步在同一帧。
**约 70% 的同步点读的是早已定稿的数据，却排在当前整条 GPU 队列后等**——这也是过去所有渲染流水线
方案（含 09-13 录制线程约 2%）收益被吃掉的共同原因。零空等上限约 25 FPS；30 FPS 还需渲染线程 CPU 降约 6 ms。
详见[渲染层级文档第十四节](compatibility/demons-souls-layers.md)。

## 2026-09-23 反馈预取（提前快照客体要读的 GPU 结果）

复用 CopyFeedback：从客体读故障学习热页，每帧开始对其 GPU 已改字节做不等待的快照，客体读先试快照；
任何后续 GPU 写都会失效快照，所以命中必为当前真实字节。默认关闭（`KYTY_FEEDBACK_PREFETCH`）。
正确性：校验模式下 195,072 次命中、195,074 段 193 MB 逐字节比对，**0 失败**，全程无崩溃。
同进程官方场景交替：关 22.7/22.8，开 **23.1/23.0**（每段计数证明生效：约 59/82 次读/帧命中）。
**约 +0.3 FPS（+1.3%），提前定稿的反馈往返不在关键路径上。** 当时保留、默认关闭；代码已在 2026-09-25 历史整理中删除，保留在分支 `experiment/render-preparation-20260912`（提交 `4134ca6`）。

## 2026-09-23 逆向：客体读的 GPU 结果属于哪些子系统

用断言/性能标记字符串 + gdb 实时调用链给全部热点读点命名。每帧约 62.5 次里：
**粒子 `ParticleSystemPool::ProcessStats` 约 53%**、**XPR 虚拟几何索引剔除约 32%**（含 Decal；
`0xad91de` 检查 20 字节间接绘制参数的 `index_count` 是否为 0 以跳过空绘制）、
**GPU 碰撞约 5%**、自动曝光约 2%，其余为统计与杂项。
GPU 碰撞只给脚部 IK：`GpuCollisionManager::CastRay` 全镜像唯一调用者是
`FootIkRaycastGPU::castRay`，对**渲染网格**打射线；真正的物理/碰撞是 CPU 上的 Havok。
**玩法层（移动、战斗、AI、物理）不读 GPU 结果；读的全是表现层逻辑**，因此是可按标题专门处理的对象。
详见[渲染层级文档第十六节](compatibility/demons-souls-layers.md)。

## 2026-09-23 完全解耦（latent 模式）：不更快，也不能做成通用的

`KYTY_FEEDBACK_PREFETCH=2`：客体读 GPU 结果时从不等待，直接用上一帧已完成的快照；精确消费者先 `RestoreLatent`。
同进程官方场景，两次独立运行：关 21.7/21.6、22.0/21.7，latent **19.9/19.7、20.2/19.9/19.9**（约 71% 读被服务），
**约 −8%**：快照拷贝加重了已饱和的渲染线程。定时采样显示客体本来就约半数时间领先一帧，
09-23 "锁步"是只在故障时采样的偏差。零拷贝的"干净页"变体（v2～v4）每版都在切换后 4 秒内崩在
`TaggedFrameAllocator` `0x8120b8`：帧分配器每帧复用同一批页（一帧是 CPU 链表，另一帧是 GPU 输出），
按页给旧数据会把 GPU 的 float 当成链表指针。二分（可复现构建，哈希逐字节一致）确认 `03d4` 版未崩。
**安全的解耦只能按读点白名单做；即使完美解耦，渲染线程约 39 ms/帧 CPU 仍把上限限制在约 25 FPS。**
`03d4` 版当时保留（默认关闭、仅供实验）；代码已在 2026-09-25 历史整理中删除，保留在分支 `experiment/render-preparation-20260912`（提交 `4134ca6`）。详见[渲染层级文档第十七节](compatibility/demons-souls-layers.md)。

## 2026-09-23 渲染线程提速与高效迭代工具

录制线程 + 延迟提交（`vkBegin/End/QueueSubmit` 交给回放线程，排空 191→5 次/帧）+ 回放线程限定 P 核：同进程 21.1 → **23.7 FPS**；
图像回收池消除临时渲染目标每帧重建（渲染 CPU −1 ms）；`launch-fast.json` 开机即启用，实测 23.75 FPS。
新增实时控制线程与 `live-bench.py`：一次启动后每个 A/B 约 30 秒（原来每次重启 5～8 分钟），并有进程内采样器、页保护调用栈、客体等待拆分。
排除：GPU 等待自旋、呈现模式、提前 EOP 中断（游戏不用）、热页不保护（客体反而变慢）、运行中改 convar（引擎锁存）。
关键发现：客体线程每帧合计约 34 ms 等本帧 GPU 反馈结果，客体产出被渲染翻译吞吐绑定；渲染 CPU 约 35 ms 是上限，
30 FPS 仍需再砍约 5 ms。详见[渲染层级文档第十八、十九节](compatibility/demons-souls-layers.md)。

## 留下的判断与测量教训

- 以约 44.4 ms/帧作预算，30 FPS 需净省约 11.1 ms，60 FPS 需约 27.7 ms。必须扣除替代路径本身的成本。
- 现有实验没有证明“本架构绝不可能 30 FPS”，也没有证明“只剩并行化”。旧文这两种绝对结论不沿用。游戏专用批量处理、改变资源消费方式仍是待验证方向，不是已取得收益。
- 同进程也有时间漂移；跨会话绝对 FPS 更不能直接相减。用交替窗口、实际呈现帧数/总时间、CPU/帧和必要的间隔分布交叉判断。
- FIFO 量化可能掩盖小收益，但不能因此把所有无收益都归因于 vsync。显示器关闭曾使 present 约 1 秒一次，不能误诊为 guest job 停滞；保持显示器活动、60 Hz。
- 高频细粒度插桩曾显著放大耗时；诊断窗口不用来宣称 FPS 收益。GPU dummy 也必须保留 CPU 能观察到的输出和同步语义。

## 后续记录约定

每个新实验只追加“日期、方案、同进程参考/候选结果、正确性检查、是否保留”一行；重要失败补一句原因。只有结论需要原始数据时保留小型凭据，不再为每轮新建分析长文。正式构建、存档、AOT/PGO、分配器和预热缓存按启动说明保护。

## 2026-09-20 整理

顶层文档收敛为导航、当前版本、复测方法和本实验简记 4 份；历史长文合并，过期捕获/回放工具和约 8.02 GiB 旧产物移出项目，其中约 2.96 GiB 在用户回收站，可恢复。清理清单及回收站映射见 `_Build/experiment-records/cleanup-20260920.json`。正式程序、配置、运行库、PGO、存档及关键缓存共 64 项指纹一致，两个日常启动入口 dry-run 通过；保留当前默认关闭的 PBR 原型及最小实验凭据。

## 2026-09-26 40 FPS 目标 M0：剖析、两项提交、原生 dispatch 否定

整理后的分支 `experiment/perf-40fps-20260926`（基于 `experiment/perf-clean-20260925`）。新增本地测量工具（同进程 A/B、带帧指针调用链的
渲染线程墙钟/CPU 采样、按 shader 与阶段计时的 census、每帧事件计数）。主要数据与结论见 `docs/PERF-40FPS-GOAL.md` 进度日志。
提交两项（默认关）：写窗口故障交出图像锁（+1.1 FPS）、分离被 GPU 写覆盖的客体回读（+0.56 FPS），合计同进程 28.12 → 30.10 FPS。
否定：写窗口加大、只读缓冲 clean proof、原生 dispatch（剔除输入每帧重定位，重求值开销超过普通路径）。

### 2026-09-26 原生 XPR 画面错误（乱三角形）修复

现象：原生 XPR 开着时墙体发黑成剪影、黑色碎三角；关掉原生 XPR 或把 `KYTY_STREAM_UPLOAD` 调成 0（原生路径基本失效）就正常。
排查：运行时开关逐个切（三项 33 FPS 新开关、延迟提交、帧流水线、屏障去重都无关）→ -O2 构建同样错（不是编译器问题）→
历次会话起始截图显示整理后的第一个构建（9 月 25 日 20:22）起就错 → 对比原分支 `4134ca6`：`CommitGraphicsState()` 经
`PrepareGraphicsState()` 在录制构建里总是 `InvalidateGraphicsState()`，整理时随 `GraphicsStateCache` 一起删掉了。
普通单个 draw 固定走直接路径（`set_bind_debug`/`set_auto_debug` 为 true），不让代数加一，原生 draw 的 emit 缓存便沿用它绑的管线、动态状态、索引缓冲。
修复：恢复这一行（`0b50cd7` / `a5d2314`）。同进程 A/B：-2.6% FPS，渲染线程 CPU 不变。教训：只看 FPS 的基准会掩盖渲染错误，
起始截图要看；跨会话 FPS 受前走停下位置影响 ±1～3，只信同进程 A/B。

## 2026-09-29～30 另一个代理（Codex）的夜间实验：均无收益，09-30 已删除

09-29 17:24 至 09-30 07:27，另一个代理在本分支试了下列方案（开关默认关）。它们即使不开也让日常路径变慢
（同位置 40.85 → 39.95 FPS，渲染线程 +0.7 ms），09-30 全部撤回：改过的 72 个已有文件按它的会话日志逆向还原为
动手前内容（与它开工时写进 git 对象库的 9 个原文件逐字节一致），新增的 71 个文件移入回收站。完整改动存为
`_Build/experiment-records/codex-native-render-20260930/agent-work.patch`（可在还原后的树上 `git apply`），
各轮日志与截图在 `_Build/{portable-decode,fps50,render-effects,native-*}-2026092x/`。下表均为同进程、同站位的
新画面 present FPS（参考 → 候选），完整画质、60 Hz vblank、插帧关。

| 方案 | 参考 → 候选 | 结论 |
|---|---:|---|
| 便携格式解码：32 位格式直读 + 其余放共享 `DontInline` 函数 | 两个重 CS 首次编译 278.8 → 198.7 s；另测 5 个最重模块总量不变 | NVIDIA 编译器在 CS `b4a64911fc8e88db` 上崩溃（游戏冷遇到即闪退），删除 |
| XPR 多段常量上传合成一次 Map/Commit | 39.35 → 39.43 | 噪声内 |
| 描述符后备计数、clear/reserve 延后 | 39.33 → 38.38 | 无收益 |
| compute 描述符/屏障/绑定/dispatch 合包 | 39.85 → 38.85（近墙轻场景 60.2 → 60.2） | 无收益 |
| NativeXprReader 页内快读复用于校验 | 40.20 → 40.20 | 无收益 |
| 相邻剔除 dispatch 复用 PreparedBindings | 39.73 → 39.50 | 查找省下的被校验与存储抵消 |
| 连续剔除 dispatch 整段原生提交（DispatchBase 选参数行）v1 / v2 | 39.35 → 32.08 / 39.95 → 32.98 | v1 有 290 万次尝试在完整准备后因 CPU 脏或非持久 buffer 被拒；段均只有 2.8–3.3 个对象 |
| 原生阴影通道（DrawID 对象表 + multi-draw），首版 / 固定协议后 | 39.33 → 35.28 / 39.88 → 35.55 | 回退 |
| 资源表 v2 + 原生剔除算法 | 40.17 → 35.25 | 回退 |
| 四族完整 GPU 几何链（初始化→剔除→间接绘制） | 38.45 → 25.05 | 回退 |
| 仅替换游戏的 CPU 对象准备函数（机器码指纹认证） | 38.45 → 38.73 | 未证明稳定收益 |
| 整段几何准备（修正后） / 固定原生对象输入程序 | 39.68 → 33.05 / 37.93 → 32.80 | 回退 |
| 实际 mesh16 阴影通道 / 最后一版阴影输入 + 小索引上传 | 39.85 → 30.03 / 37.75 → 30.43 | 渲染线程 20.97 → 28.36 ms/帧 |

- 诊断（降画质，不是优化）：游戏自己的效果开关是 convar 对象 `+0x60` 指向的 bool 字节。同进程关掉：体积雾
  40.30 → 42.15，全部粒子 40.45 → 40.65，太阳级联阴影 40.60 → 44.05，雾 + 三项阴影 40.43 → 47.07（约 −3.5 ms/帧）。
  阴影是最大的一项：原画质每帧 8977 draw / 4244 dispatch，两张 1024² 阴影图共约 2600 draw。
- 原因：瓶颈不只在 GPU 绘制，渲染线程还要翻译命令、读对象数据、解析资源、处理写入与同步。"原生"替换省下的
  模拟准备被新增的读取证明、资格检查和失败回退抵消；逐对象保持 guest 内存语义的查询与校验仍然昂贵。
- 方形 D32 阴影图用的是 Uint16 mesh 索引，与四族剔除的输出无关（剔除输出属于主深度预通道），不能用剔除结果驱动阴影。
- 游戏大量复用内存，旧图像登记仍覆盖新用途（顶点描述符、代码、索引确实落在旧图像范围内，不只是同一个 4 KiB 页），
  所以每帧约 2500 次图像重叠检查全部拒绝；要扩大接管得有逐字节的 backing 所有权证明（未实现）。
- 三个高频 q0 剔除 CS（约 3000 dispatch/帧、约 3.85 ms）都用 6 个 buffer + 1 个 sampled image + 1 个 sampler，
  "只有 buffer 的准备快路径"覆盖率为 0。
- 剖析要用本次构建实际的 lld map：误用 PDB 邻近符号曾把 `NativeXprValidate` 报成 XXH3（真正的 shader 代码哈希约 0.28 ms/帧）。
- ABBA 窗口里偶有 35–36 FPS 的异常低值，原因未定位；全部保留，不挑窗口。

## 2026-09-30 翻译器首遇卡顿：CFG 支配分析与确定性

- 冷启动固定路线（不预热、`KYTY_SLOW_LOG_MS=30`）翻译卡顿 28 次共 2.7 s、最长 915 ms（CS `55d35da67c9a387a`）。
  分阶段计时：其中 769 ms 是 CFG 结构化；结构化每做一次图变换就重算全部分析（该 CS 52 次），后支配集用"每块一份
  完整集合、按正序迭代求交"的朴素算法，占 843/862 ms；`Dominates` 在集合里线性查找，最近公共后支配者 O(k³)。
- 改为 Cooper-Harvey-Kennedy 支配树 + DFS 区间（O(1) 查询），保留旧不动点在边界情况下的答案（根不可达的块视为
  被所有块支配，上次分析后新增的块没有支配关系）；循环体、区域成员改位图/二分。21,003 个种子的 SPIR-V 逐字节不变，
  `shader_cfg_tests` 全过。该 CS 翻译 862 → 77 ms；冷启动翻译卡顿 26 次共 1.4 s、最长 162 ms。
- 分派器回退路径（不可结构化 CFG）在 wave64 下按指针哈希顺序给高半波 spill 变量分配 id，同一输入每次 SPIR-V 都不同，
  静态缓存永远命中不了（CS `8fb6871e64c650c9`）。改为程序顺序后两次全量翻译完全一致。
- 剩余：翻译 30–160 ms 的程序约 25 个，成本分散在 IR 翻译、SSA（所有块到最后才封口，每块每变量先建临时 Phi）、
  SRT 规划、SPIR-V 生成；同一程序遇到新特化会整段重翻（IR 不可复制）。真正的冷启动大头是没进静态缓存的管线：
  CS 约 36 条共 22 s（特化猜不中的 18%），驱动缓存冷热会让同一批图形管线在 30 ms 与 100+ ms 之间波动。

## 2026-09-30 首遇管线：先编未优化版，后台换优化版（`KYTY_PIPELINE_FAST_BUILD`，默认开）

- 冷启动固定路线上，没进静态缓存的 113 条 CS 管线离线正常编译共 89 s、单条最长 6.3 s；加
  `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT` 只要 0.6 s、最长 26 ms（NVIDIA，约 150 倍）。
- 做法：静态缓存和本机驱动缓存都没有的管线，立刻以未优化方式编译并使用；两个编译线程用同一管线布局编优化版，
  编完后渲染线程在下一次管线查找时原地替换（原生 XPR 记录保存的是 Pipeline 指针，所以按完成计数统一替换，
  不能只在查到时换）；被换下的句柄留到退出再销毁。`KYTY_PIPELINE_FAST_BUILD=0` 恢复原行为。
- 冷启动同路线（不预热）：CS 管线卡顿 36–38 次共 22 s → 1 次 32 ms，图形管线 8–32 次 → 0，出现 HUD 88 s → 61 s；
  冷步行逐秒最低 24 fps，无秒级冻结。画面截图与之前一致。
- 日常设置（新 exe、本机缓存空）：40.25 / 40.40 / 40.60 fps，逐窗口回升到还原后的 40.8 附近——测量开始时后台还在
  替换原生 XPR 变体的优化版。正常退出会把优化版存进 `_PipelineCache/local/<exe SHA256>`，下次直接命中。

## 2026-10-17 08:10 — "Sin TAA" modes darker in some zones (pending)
- User: with the game's TAA the image is fine; without it (sin TAA) whole scenes are darker in some zones.
- TAA shader a44e580f4f0cfb2b (dumped with KYTY_SHADER_DUMP): works in tonemapped space (c/(1+luma 0.3/0.59/0.11), inverse at the end); multiplies the history (5-tap bicubic of the previous output) by a constant (S_BUFFER_LOAD offset 24 of the buffer at s30+240), likely an exposure ratio. Steady state out = w*cur/(1-(1-w)K): brighter than the raw frame when K > 1. Skipping the TAA loses it.
- Next: log that constant per frame in those zones; then apply the same factor before the upscaler. Also a bug found: DlssUpscale passes jitter 0 (meant for the TAA's resolved output), but with KYTY_TAA_REPLACE=upscaler the input is the raw jittered frame: it needs the camera jitter like the TAA path.

## 2026-10-25 shadows-1025 — local-light / shadow-map quality (non-sun)
- Evidence (today's logs 091911 / 130923 / 134203 / 172949, build c3e58bd): the 1024x1024x80 D32 spot pool renders 5-14 layers a
  frame (binds 300+ at the lamps), "layers refreshed from guest memory" only equal the sampled streamer layers (kept 0 /
  changed 0: no GPU-rendered layer was dirtied, KYTY_SHADOW_REFRESH_FIX never had to act); no "RTLS:" line in any log: the
  RT local-shadow light-list discovery (KYTY_RTLS_DEBUG, rtlocal step 1) has not been run yet, so step 2 (per-pixel rays to
  the N nearest lights) stays blocked on that proof.
- The game has no convar for the shadow filter, penumbra, spot resolution or distance (eboot strings + the shipped
  coredata/enginesupport/debugmenu/debugrendermenulighting.txt): the shadow convars are the launcher's sun cascades,
  r_enableSpotShadows, r_contactShadows{Enabled,Distance,Thickness,Softness} (0..1 each) and the ambient (capsule)
  shadowing: r_ambientShadowMinAmbient / MinDirectional (0..1), ConeAngle (1..89), RadiusToFadeRange (5..50),
  RadiusToFadeRangePow (0.1..4), r_enableAmbientShadowingLocalLightSource (one local light's capsule shadows). The
  console's renderconfig_ps5_{high,perf}.txt only set r_virtualGeometryNumShadowCascades=2.
- KYTY_SHADOW_FILTER=1|2|3 (spirvEmitterImage.cpp): every depth-compare sample / gather of a non-cube 2D-array shadow map
  widened at translation (4-tap rotated grid / 9-tap Poisson / PCSS: 8-tap blocker search + 12 taps at the penumbra;
  KYTY_SHADOW_FILTER_RADIUS, _SCALE (1000), _MAX (7)). Default 0: SPIR-V unchanged. A level changes the SPIR-V of the
  shaders reading shadow maps: their pipelines compile at first use (fast build or a stall with KYTY_PIPELINE_FAST_BUILD=0).
- KYTY_GTAO_FLOOR (0.3..0.95, 0.7): the GTAO composite's darkest multiplier ("profundidad").
- Not done, with the reason: shadow-atlas upscale (no render-target scaling in the emulator; the cube pool is written as
  an R32 storage image with integer coordinates; the spot pool's per-layer guest hashes assume the guest layout);
  local-light contact shadows (need the light positions: step 1's proof).
