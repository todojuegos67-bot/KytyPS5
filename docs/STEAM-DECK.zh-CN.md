# 在 Steam Deck 上玩恶魔之魂

按下文的设置，KytyPS5 在 Steam Deck 上运行恶魔之魂（PPSA01341：Deck 上测试的是 01.007.000，01.005.000 也支持），
各世界约 26-30 帧，连结之地（Nexus）约 40 帧。本文说明怎样把模拟器和游戏传到 Deck、怎样从 Steam 启动，以及能跑到多少帧。
（English: [STEAM-DECK.md](STEAM-DECK.md)）

## 帧数

在 Steam Deck LCD（SteamOS 3.9）上用 Steam Deck 版（见文末）、按 `run-deck.sh` 的设置、GPU 频率 1300 MHz 测得：从上一个世界传送过去，在各世界的第一块
要石处、载入 40 秒后测 15 秒。

| 地点 | 帧数 |
|---|---|
| 波雷塔利亚 Boletaria（1-1），关卡起点 | 37-38 |
| 波雷塔利亚 Boletaria（1-1），从连结之地传送过来时 | 25-26 |
| 石牙坑道 Stonefang Tunnel（2-1） | 29-30 |
| 拉特利亚之塔 Tower of Latria（3-1） | 26 |
| 风暴祭祀场 Shrine of Storms（4-1） | 29 |
| 腐朽之谷 Valley of Defilement（5-1） | 28-29 |
| 连结之地 Nexus | 40-41 |

决定帧数的是 Deck 的 15 W 功耗：CPU 和 GPU 共用，而模拟器两者都要用。GPU 1300 MHz 时留给 CPU 的功耗足够；
GPU 1600 MHz（最高）时整体反而更慢；频率保持自动（Steam 的默认）时，之后一次巡游各世界慢 1-3 帧（连结之地相同）。第一次到一个地方会编译新的着色器，表现为短暂卡顿（Deck 上最长约 1.5 秒）；
编译结果保存在 `_PipelineCache`，再到那里就流畅了。

`run-deck.sh` 用的设置：启动器的**最低**画质预设（阴影距离更短，无环境光遮蔽和火把阴影，远处细节简化，贴图为四分之一），
3D 画面以 640x360 渲染，关闭投射贴花（血迹、焦痕），太阳阴影只到 20 米，游戏的 1080p 输出全屏显示并保持 16:9
（上下黑边）。

内存：Deck 的 16 GB 与 GPU 共用。走遍所有世界后，剩余内存 1.2 GB，模拟器有 2.7 GB 的内存在交换文件里：
请关闭其他程序，保留 SteamOS 的交换文件。

## 1. 模拟器

在**桌面模式**下（长按电源键 > 切换到桌面）：

1. 打开浏览器，从[发布页](https://github.com/chenxiao07/KytyPS5/releases)下载 `KytyPS5-steamdeck-<版本>.zip`
   （Steam Deck 版的包；Linux 版 `KytyPS5-linux-<版本>.zip` 配合同一个 `run-deck.sh` 也能用；不要下载 Windows 版）。
2. 在文件管理器（Dolphin）中右键它 > 解压 > 解压到此处，把文件夹改名为 `KytyPS5`（叫什么都可以；本文用 `~/KytyPS5`）。
3. 打开 Konsole，给程序加上可执行权限（解压工具可能把它丢掉）：

   ```
   chmod +x ~/KytyPS5/kyty_emulator ~/KytyPS5/run-deck.sh
   ```

下面的设置都由压缩包里的 `run-deck.sh` 完成，不需要另外安装任何东西。

**更新**到新版本：用同样的方法解压新版的 Steam Deck 包，把新文件夹里的全部内容复制到 `~/KytyPS5`，覆盖原有文件
（Dolphin：全部覆盖；Konsole 中为 `cp -r <新文件夹>/. ~/KytyPS5/`），然后再执行一次上面的 `chmod +x` 那一行。存档
（`_SaveData`）、着色器缓存（`_PipelineCache`）和金手指文件不在包里，会原样保留；Steam 里的快捷方式及其启动选项也
不受影响。更新后第一次游玩时，部分着色器可能会重新编译一次。

## 2. 游戏

游戏是含有 `eboot.bin` 的文件夹（`PPSA01341-app0`，约 82 GB），或者打包成 ZArchive 的同一内容（`PPSA01341-app0.zar`，
约 54 GB）。放到 Deck 的 `~/Games/` 下，即 `~/Games/PPSA01341-app0.zar` 或文件夹 `~/Games/PPSA01341-app0`：
`run-deck.sh` 会在这里找。（放在别处时设置 `KYTY_GAME`，见下文。）

从电脑复制过去的方法：

- **U 盘**：在电脑上格式化为 **exFAT**（要有 54-82 GB 空间），把游戏复制进去，在桌面模式下接到 Deck（USB-C 转接头或扩展坞），
  用 Dolphin 复制到 `~/Games`。SteamOS 格式化过的 microSD 卡是 ext4，Windows 写不了；在电脑上格式化为 exFAT 的卡可以像 U 盘一样用。
- **通过网络**从 Windows 复制：在 Deck 的 Konsole 里给 `deck` 用户设置密码并启动 SSH：

  ```
  passwd
  sudo systemctl start sshd
  mkdir -p ~/Games
  ```

  查到 Deck 的地址（设置 > 互联网，或在 Konsole 里运行 `ip -4 addr`），然后在电脑的 PowerShell 里：

  ```
  scp -r "D:\Games\PPSA01341-app0" deck@192.168.0.50:/home/deck/Games/
  ```

  （换成你的文件夹和 Deck 的地址。）5 GHz Wi-Fi 下约 30-40 MB/s（整个文件夹约 40 分钟）；2.4 GHz Wi-Fi 只有 2-7 MB/s。
  从 Linux 或 WSL 用 `rsync -av --partial` 复制，中断后可以续传。用完后 `sudo systemctl stop sshd` 关闭 SSH。

**存档**：要接着玩电脑上的存档，把电脑上模拟器旁边的 `_SaveData` 文件夹复制到 Deck 的 `~/KytyPS5/` 里。

## 3. 加入 Steam

仍在桌面模式：打开 Steam > 游戏 > **添加非 Steam 游戏到我的库** > 浏览，显示所有文件，选择 `~/KytyPS5/run-deck.sh` 并添加。
在它的属性里起个名字（恶魔之魂）。

回到**游戏模式**，从库（非 Steam）里启动。然后设置一次它的性能配置：快速访问菜单（`...` 键）> 性能 > **使用游戏配置文件**，
再在高级视图里打开手动 GPU 时钟控制，设为 **1300 MHz**。游戏模式会为这个游戏记住它。

请从游戏模式启动：这样 Deck 的按键只会作为手柄传给游戏。在 Steam 之外（桌面模式）启动时，Steam 的桌面布局还会为 Deck 的
摇杆和按键发送键盘按键：方向键会切换装备，它发送的空格键会让模拟器暂停（再按一次空格恢复；Steam Deck 版会忽略这些按键）。

## 4. 选项

`run-deck.sh` 的开关写在快捷方式属性 > 启动选项里的 `%command%` 前面，例如 `KYTY_DECK_SUN_SHADOWS=full %command%`：

| 开关 | 作用 |
|---|---|
| `KYTY_DECK_SUN_SHADOWS=full` | 太阳阴影到 60 米，即最低预设的值（更慢） |
| `KYTY_DECK_SUN_SHADOWS=off` | 没有太阳阴影（最快，但在暗处阳光会照亮角色） |
| `KYTY_DECK_DECALS=1` | 恢复投射贴花（血迹、焦痕）（约慢 7%） |
| `KYTY_RENDER_HEIGHT=432` | 更清晰的 3D（768x432；比 360 约慢 5%） |
| `KYTY_FPS_HUD=0` | 不显示右上角的帧数面板（F8 切换） |
| `KYTY_GAME=/游戏/路径` | 游戏不在 `~/Games` 时 |
| `KYTY_CONSOLE_LANGUAGE=0` | 主机语言（1 英语，默认；0 日语） |

**金手指。** 模拟器自己的参数写在 `%command%` 后面。与游戏版本对应的 GoldHEN 格式金手指文件（包里不带；见 Windows 版
README 的 "Cheats"）放在 `~/KytyPS5/_Patches/PPSA01341.json`，用
`%command% --game-patch /home/deck/KytyPS5/_Patches/PPSA01341.json` 启用；文件里 `"enabled": false` 的条目不生效。
包装程序（例如补帧工具的）写在 `%command%` 前面，开关再写在它前面：
`KYTY_FPS_HUD=0 /路径/包装程序 %command% --game-patch /home/deck/KytyPS5/_Patches/PPSA01341.json`。每条生效的金手指在日志里
有一行 `Successfully applied cheat: <名称>`。

每次运行的日志在 `~/KytyPS5/logs/`；遇到问题时请附在 issue 里。

## Steam Deck 版的包与 Linux 版的包

Steam Deck 版的包里是 Steam Deck 版：针对 Deck 的 Zen 2 CPU 优化，内置 Deck 的默认设置，并忽略 Steam 桌面布局发送的键盘按键。
Linux 版的包加上其中的 `run-deck.sh` 运行方式相同，只是没有这两点。在 Deck 上并排测过（波雷塔利亚关卡起点，`run-deck.sh` 的设置，
GPU 频率自动，交替启动六次）：两者帧数相同，站立 36.0-36.7 帧，转动视角 34.2-34.7 帧，差别在每次运行的波动之内。用 Deck 上训练的
PGO 构建的 Steam Deck 版（`build-deck.sh` 构建的）也一样。所以 Linux 版的包一样快，Steam Deck 版的包的区别在按键。想自己构建
Steam Deck 版：在 Linux 上，或在不比 SteamOS 新的发行版的 WSL 里（glibc 2.43；Ubuntu 26.04 与之相符）运行
`tools/local/deck/build-deck.sh`；步骤和在 Deck 上测量用的工具见 `tools/local/deck/README.md`。
