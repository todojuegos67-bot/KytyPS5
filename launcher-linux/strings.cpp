#include "strings.h"

#include <cstdlib>
#include <cstring>
#include <string_view>
#include <unordered_map>

namespace Strings {

Language Current = Language::English;

namespace {

using Table = std::unordered_map<std::string_view, const char*>;

const Table& English() {
	static const Table table {
	    {"output.1440", "1440p (DLSS / FSR 3)"},
	    {"output.1440NoUpscaler",
	     "1440p needs DLSS or FSR 3 as the upscaler; with the game's TAA the output stays 4K."},
	    {"render.capped1440", "Demon's Souls' 3D at {0}x{1} (1440 at most with the 1440p output)."},
	    {"render.raised1440", "Demon's Souls' 3D at {0}x{1} (with the 1440p output 720 at least, "
	                          "or 480 for its Ultra Performance)."},
	    {"summary.1440", "1440p output"},
	    {"button.desktop", "Add to the applications menu"},
	    {"desktop.done", "KytyPS5 is in the applications menu now ({0})."},
	    {"desktop.failed", "The menu entry could not be written: {0}"},
	    {"app.subtitle", "PS5 emulator for Linux"},
	    {"nav.play", "Play"},
	    {"nav.display", "Display"},
	    {"nav.graphics", "Graphics"},
	    {"nav.game", "Game"},
	    {"nav.shaders", "Shaders"},
	    {"nav.advanced", "Advanced"},
	    {"nav.experimental", "Experimental"},
	    {"nav.hint",
	     "In the game: F8 switches the performance overlay."},

	    {"play.choose", "Choose your game"},
	    {"play.checks", "Checks"},
	    {"play.gamebox", "Game"},
	    {"play.summary", "Current settings"},
	    {"button.folder", "Folder…"},
	    {"button.zar", ".zar…"},
	    {"button.play", "PLAY"},
	    {"button.precompile", "Precompile shaders"},
	    {"button.hints", "Export shader hints"},
	    {"button.logs", "Logs folder"},
	    {"button.readme", "README"},
	    {"button.root", "Emulator folder"},
	    {"button.cheats", "Choose…"},
	    {"button.getdlss", "Download and build DLSS…"},
	    {"game.none", "No game chosen: choose the folder with eboot.bin and sce_sys (the game "
	                  "dump), or a .zar archive of it."},
	    {"game.bad", "Not a game: no sce_sys/param.json there (a .zar needs kyty_shader_precompile "
	                 "next to the launcher)."},
	    {"game.loading", "Reading the game…"},
	    {"game.ok", "{0} · {1} · version {2}"},
	    {"gpu.ok", "{0} · {1} GB video memory"},
	    {"gpu.okNoMemory", "{0}"},
	    {"gpu.none", "No Vulkan graphics card found (is its Vulkan driver installed?)"},
	    {"ram.ok", "{0} GB of RAM"},
	    {"ram.low", "{0} GB of RAM: close other programs while playing (the emulator takes about "
	                "14 GB while a save loads)"},
	    {"shaders.checking", "Precompiled shaders: checking…"},
	    {"shaders.current", "Shaders precompiled for this graphics card"},
	    {"shaders.outdated", "Shaders: precompile again (only what is new is compiled)"},
	    {"shaders.outdatedAll", "Shaders: precompile again (this time all of them, once: a newer translation; the current ones work until then)"},
	    {"shaders.none",
	     "Shaders not precompiled yet: the game stutters the first time each effect appears"},
	    {"shaders.unknown", "Precompiled shaders: not known (start the game once)"},
	    {"dlss.ok", "DLSS ready (libkyty_dlss.so found)"},
	    {"dlss.noRtx", "DLSS needs an NVIDIA RTX graphics card"},
	    {"dlss.noLibrary", "DLSS: download and build its runtime once (Graphics page)"},
	    {"fsr.ok", "AMD FSR 3 ready (built in)"},
	    {"emulator.missing",
	     "No kyty_emulator next to the launcher (build it with ./build-linux.sh)"},

	    {"summary.window", "{0} window"},
	    {"summary.fullscreen", "{0} fullscreen"},
	    {"summary.render", "3D {0}p"},
	    {"summary.taa", "game's TAA"},
	    {"summary.noAa", "no anti-aliasing"},
	    {"summary.dlss", "DLSS"},
	    {"summary.fsr3", "FSR 3"},
	    {"summary.fsr", "FSR 3 frame generation"},
	    {"summary.hd", "1080p output"},
	    {"summary.hdr", "HDR"},
	    {"summary.vsync", "VSync"},
	    {"summary.fps120", "up to 120 fps"},
	    {"summary.options", "Options"},
	    {"summary.presentDirect", "direct presentation"},
	    {"summary.vrrFlips", "VRR flips"},
	    {"summary.wideBars", "16:9 movies and menus"},
	    {"summary.antilag", "Anti-Lag"},
	    {"summary.fps30", "30 fps lock"},
	    {"summary.extra", "extra options"},
	    {"summary.mangohud", "MangoHud"},

	    {"display.title", "Display"},
	    {"display.window", "Window"},
	    {"display.rate", "Frame rate"},
	    {"resolution", "Resolution"},
	    {"resolution.desc", "The window's size; in fullscreen the screen mode."},
	    {"ratio", "Aspect ratio"},
	    {"ratio.desc", "21:9 and 32:9 show more to the sides; the HUD keeps its shape."},
	    {"ratio.169", "16:9 (PS5)"},
	    {"widebars", "Movies and menus at 16:9 (black bars)"},
	    {"widebars.desc", "With 21:9 or 32:9: the intro, the movies, the title screen, the loading screens and "
	                      "the menus at 16:9 with black bars at the sides, not stretched (off: stretched to the "
	                      "width)."},
	    {"fullscreen", "Fullscreen"},
	    {"aspect", "Keep the aspect ratio (black bars)"},
	    {"aspect.desc", "In fullscreen on a screen of another shape."},
	    {"hdr", "HDR"},
	    {"hdr.desc", "Needs HDR turned on in the desktop's display settings (for example KDE "
	                 "Plasma 6, GNOME 48 or newer, or "
	                 "gamescope) and a Vulkan driver that offers it; SDR otherwise."},
	    {"fps120", "Up to 120 fps"},
	    {"fps120.desc", "The game's own frames above 60; movies stay at 60."},
	    {"vsync", "VSync"},
	    {"vsync.desc", "Frames wait for the display's refresh (Vulkan FIFO); VRR displays keep "
	                   "adapting. Off: mailbox, which "
	                   "doesn't tear either but renders past the refresh rate."},
	    {"lowlatency", "Low latency"},
	    {"lowlatency.desc", "One finished frame waits for the screen instead of about two: at 60 "
	                        "Hz about 15-23 ms from the "
	                        "game's frame to the screen instead of about 30. A late frame is then "
	                        "no longer covered, so the "
	                        "pacing is a little less even."},

	    {"graphics.title", "Graphics"},
	    {"graphics.resolution", "Resolution and upscaling"},
	    {"graphics.detail", "Detail"},
	    {"graphics.fg", "Frame generation"},
	    {"render", "3D render height"},
	    {"render.desc",
	     "Demon's Souls' 3D at {0}x{1}; higher is sharper and needs more of the graphics card."},
	    {"render.capped", "Demon's Souls' 3D at {0}x{1} (1080 at most with the 1080p output)."},
	    {"render.raised",
	     "Demon's Souls' 3D at {0}x{1} (with DLSS's 4K output 1080 at least, or 720 for its Ultra "
	     "Performance)."},
	    {"render.native", "native 4K"},
	    {"render.pro", "PS5 Pro"},
	    {"render.ps5", "PS5"},
	    {"upscaler", "Upscaler"},
	    {"upscaler.game", "Game (PS5)"},
	    {"upscaler.dlss", "DLSS (PS5 Pro path)"},
	    {"upscaler.fsr", "FSR 3 (PS5 Pro path)"},
	    {"upscaler.fsrNote", "AMD FSR 3 in place of Sony's PS5 Pro upscaler, on any graphics card "
	                         "(built in, nothing to "
	                         "download): sharper than the game's stretch. Each render height shows "
	                         "the mode it runs in with "
	                         "the chosen game output."},
	    {"fsr.native", "Native AA"},
	    {"upscaler.off", "Off (no anti-aliasing)"},
	    {"upscaler.offNote", "Demon's Souls without its own anti-aliasing (TAA) and the camera "
	                         "jitter that comes with it: a sharper, steadier picture, with jagged "
	                         "edges and shimmer on fine detail. The 3D at the render height below, "
	                         "stretched to the output as the game does."},
	    {"upscaler.ok", "DLSS in place of Sony's PS5 Pro upscaler: sharper than the game's "
	                    "stretch. Each render height "
	                    "shows the DLSS mode it runs in with the chosen game output."},
	    {"upscaler.noRtx",
	     "DLSS needs an NVIDIA RTX graphics card: here the game's own TAA makes the picture."},
	    {"upscaler.noLibrary",
	     "DLSS needs its runtime: Download and build DLSS under Frame generation (once)."},
	    {"output", "Game output"},
	    {"output.auto", "Automatic"},
	    {"output.4k", "4K (PS5)"},
	    {"output.1080", "1080p"},
	    {"output.autoNow", "Automatic: 1080p for a 16:9 game window (in full screen, the screen) "
	                       "at most 1080 lines high, "
	                       "4K otherwise. Here: {0}."},
	    {"output.desc", "1080p: the game's HD layout, its effects and HUD at 1920x1080; less work "
	                    "for the graphics card. With DLSS, DLSS makes the 1080p picture from the "
	                    "3D (for 1080p screens). 1440p: DLSS or FSR 3 makes a 2560x1440 picture "
	                    "(for 1440p screens), the HUD in its HD layout scaled to it."},
	    {"dlssmodel", "DLSS model"},
	    {"dlssmodel.auto", "Automatic"},
	    {"dlssmodel.k", "Transformer (preset K)"},
	    {"dlssmodel.e", "CNN (preset E)"},
	    {"dlssmodel.m", "Transformer 2 (preset M, DLSS 4.5)"},
	    {"dlssmodel.l", "Transformer 2, sharpest (preset L, DLSS 4.5)"},
	    {"dlssmodel.desc", "With DLSS: Automatic leaves the model to the DLSS version. Transformer "
	                       "is the newer, sharper "
	                       "model; CNN the older one, lighter on RTX 20 and 30 cards, where the "
	                       "transformer model costs more. M and L are DLSS 4.5's second transformer "
	                       "models (L sharper, M faster), with a DLSS 4.5 runtime."},
	    {"sharpen", "Sharpening"},
	    {"sharpen.off", "Off"},
	    {"sharpen.low", "Low"},
	    {"sharpen.medium", "Medium"},
	    {"sharpen.high", "High"},
	    {"sharpen.desc", "With DLSS or FSR 3: sharpens the upscaler's picture (RCAS). DLSS is Off by "
	                     "default: the game's own TAA sharpens its image, DLSS doesn't, so fine detail "
	                     "looks softer. FSR 3 is Medium by default, its own sharpening. Each upscaler "
	                     "keeps its own setting. Higher levels can show halos on edges."},
	    {"dlss.dlaa", "DLAA"},
	    {"dlss.ultraquality", "Ultra Quality"},
	    {"dlss.quality", "Quality"},
	    {"dlss.balanced", "Balanced"},
	    {"dlss.performance", "Performance"},
	    {"dlss.ultra", "Ultra Performance"},
	    {"quality", "Graphics preset"},
	    {"quality.standard", "Standard (PS5)"},
	    {"quality.low", "Low"},
	    {"quality.lowest", "Lowest"},
	    {"quality.desc", "Standard: the PS5's picture (below 12 GB of video memory large textures "
	                     "leave out their largest "
	                     "level). Low: shorter sun shadows, simpler distant detail, textures half "
	                     "the size per side. Lowest: "
	                     "shorter still, no ambient or torch shadows, no decals (blood, scorch marks), "
	                     "textures a quarter the size. For PCs whose processor "
	                     "or graphics card limits the frame rate."},
	    {"framegen", "DLSS frame generation"},
	    {"framegen.off", "Off"},
	    {"framegen.ok", "NVIDIA RTX 40 or newer (3x and up: RTX 50). Shown frames = the game's "
	                    "frames x the multiplier; "
	                    "with VSync they are capped at the display's refresh (a 60 Hz display at "
	                    "2x runs the game at 30 "
	                    "fps)."},
	    {"framegen.noRtx", "Needs an NVIDIA RTX 40 or newer graphics card."},
	    {"framegen.noLibrary", "Needs DLSS's runtime: Download and build DLSS below (once)."},
	    {"dlssruntime", "DLSS runtime"},
	    {"dlssruntime.desc",
	     "Runs get-dlss-linux.sh in a terminal: it downloads NVIDIA's DLSS SDK and builds "
	     "libkyty_dlss.so next to the emulator. When git or a C++ compiler is missing it offers to "
	     "install them (y/N, then your password). Nothing of NVIDIA's comes with the emulator."},
	    {"dlssruntime.found", "found in {0}"},
	    {"dlssruntime.none", "not built yet"},
	    {"dlssruntime.noScript", "get-dlss-linux.sh is not next to the launcher"},
	    {"fsr", "AMD FSR 3 frame generation (2x, any graphics card; instead of DLSS frame generation)"},
	    {"fsr.desc", "One generated frame between two of the game's, from its depth and motion. "
	                 "With VSync the shown "
	                 "frames are capped at the display's refresh."},

	    {"game.title", "Game"},
	    {"game.console", "Console"},
	    {"game.cheats", "Cheats"},
	    {"language", "Console language"},
	    {"language.desc",
	     "The language the game is told; it decides which text and voices it has."},
	    {"audiooutput", "Audio output"},
	    {"audiooutput.tv", "TV"},
	    {"audiooutput.headphones", "Headphones"},
	    {"audiooutput.amplifier", "AV amplifier"},
	    {"audiooutput.soundbar", "Soundbar"},
	    {"audiooutput.desc", "What the game is told its sound goes to, as on a PS5. The game mixes its sound "
	                         "for it and shows it in Settings > Sound."},
	    {"cheats", "Cheats"},
	    {"cheats.desc", "_Patches/<title id>.json (GoldHEN style): the emulator applies the cheats "
	                    "left on in it."},
	    {"cheats.count", "{0} of {1} on"},
	    {"cheats.noFile", "No cheat file for this game"},
	    {"cheats.noGame", "Choose the game first"},

	    {"shaders.title", "Shaders"},
	    {"shaders.precompile", "Precompile"},
	    {"shaders.share", "Share"},
	    {"duration.minutes", "about {0} minutes"},
	    {"duration.hours", "about {0} hours"},
	    {"precompile.desc", "Runs precompile-linux.sh in a terminal: compiles every shader of the "
	                        "game for this graphics "
	                        "card once, {0} on this PC with the processor fully loaded. Closing "
	                        "its window stops it, and "
	                        "the next run continues. Needed again after a graphics driver update."},
	    {"precompile.quick",
	     "Quick: only the shaders as recorded play used them (about half the time)"},
	    {"ecores", "Precompile on the efficiency cores only (slower, the PC stays responsive)"},
	    {"ecores.none", "This processor has one kind of core."},
	    {"hints.desc", "The shaders your play recorded, without the game's code, as "
	                   "my-hints-<game>.hints: attached to a "
	                   "GitHub issue, the next packages precompile them for everyone."},

	    {"advanced.title", "Advanced"},
	    {"advanced.stability", "Overlay and stability"},
	    {"advanced.launcher", "Launcher"},
	    {"hud", "Performance overlay"},
	    {"hud.off", "Off"},
	    {"hud.rate", "Frame rate"},
	    {"hud.detailed", "Detailed"},
	    {"hud.desc", "Top right of the game image: the frame rate and frame times; Detailed adds "
	                 "the graphics card, "
	                 "processor and memory. F8 in the game switches it."},
	    {"mangohud", "MangoHud"},
	    {"mangohud.desc", "MangoHud's overlay (frame rate, processor and graphics card load, "
	                      "temperatures, power) over the "
	                      "game."},
	    {"mangohud.missing", "MangoHud is not installed (your distribution's mangohud package)."},
	    {"safegpu", "Safe GPU mode"},
	    {"safegpu.desc", "Prevents the crashes that come from the graphics card hanging ('device "
	                     "lost', a black screen while "
	                     "the driver recovers): the game's compute work runs one step after "
	                     "another. About 37% lower frame "
	                     "rate. Leave it off if the game runs without such crashes."},
	    {"uilang", "Launcher language"},
	    {"uilang.auto", "Automatic"},
	    {"uiscale", "Interface size"},
	    {"uiscale.auto", "Automatic"},
	    {"terminal", "Terminal"},
	    {"terminal.auto", "Automatic"},
	    {"terminal.desc", "The window the game's log, the precompile and DLSS's build run in."},
	    {"terminal.none",
	     "No terminal program found: the game starts without its log window (its log goes to the "
	     "logs "
	     "folder), and the precompile and DLSS's build can't be started from here."},
	    {"credit", "The settings follow the Windows launcher (launcher-windows)."},
	    {"advanced.gpu", "Graphics card"},
	    {"gpu", "Graphics card to use"},
	    {"gpu.auto", "Automatic"},
	    {"gpu.desc", "For PCs with more than one graphics card: the game and the shader precompile "
	                 "both use the one chosen. "
	                 "Automatic takes the first dedicated graphics card. After changing it, "
	                 "precompile the shaders again "
	                 "(Shaders page)."},
	    {"advanced.extra", "Extra options"},
	    {"extra", "Extra options"},
	    {"extra.desc", "Added to the game's start: KEY=VALUE sets an emulator switch (KYTY_...; "
	                   "KEY= turns one off) and "
	                   "replaces the launcher's, an --option value goes to the emulator's command "
	                   "line. \"Double quotes\" "
	                   "keep spaces. If the game no longer starts, clear this box."},

	    {"experimental.title", "Experimental"},
	    {"experimental.intro",
	     "Options still being tried out. They can help on some PCs and make things worse on "
	     "others: "
	     "turn one on, play a while, and turn it off again if it doesn't help. A note in the "
	     "linked "
	     "issue (what you saw, with your graphics card and display) helps decide what becomes a "
	     "default."},
	    {"experimental.display", "Display"},
	    {"experimental.amd", "AMD graphics cards"},
	    {"experimental.framerate", "Frame rate"},
	    {"fps30", "Lock to 30 fps (slower PCs)"},
	    {"fps30.desc",
	     "The game runs at a steady 30 fps (like the PS5's cinematic mode) instead of up to 60: "
	     "half the work "
	     "for the processor and the graphics card, even frame times, and the same game speed. For "
	     "PCs that "
	     "swing between 30 and 60; with 2x frame generation it shows 60. Movies and menus stay at "
	     "60. "
	     "Smoothest on 60 Hz and 120 Hz displays or with VRR. Replaces \"Up to 120 fps\"."},
	    {"vrrflips", "Flip frames when ready (VRR displays)"},
	    {"vrrflips.desc",
	     "For a display with G-SYNC, FreeSync or VRR on: each frame goes to the screen as soon as it is "
	     "done instead of on the game's 60 Hz beat, so below 60 fps the motion stays even. A "
	     "display without VRR shows each frame at its next refresh, as before."},
	    {"presentdirect", "Direct presentation (VRR displays)"},
	    {"presentdirect.desc",
	     "Each frame goes to the display as soon as it is ready, instead of through the presenter "
	     "thread that keeps VSync even. On some displays VRR engages only this way. Use it with "
	     "VSync "
	     "off; without VRR the frame pacing can be less even."},
	    {"antilag", "AMD Anti-Lag"},
	    {"antilag.desc",
	     "AMD's driver times the game's frames to cut input lag further, on top of Low latency "
	     "(which it "
	     "turns on too). Tested on Linux; without the driver's support Low latency works alone."},
	    {"antilag.notAmd", "AMD Radeon graphics cards only."},
	    {"feedback", "Feedback: issue #{0}"},

	    {"dialog.game", "The game folder (eboot.bin, sce_sys)"},
	    {"dialog.zar", "The game folder packed into a ZArchive"},
	    {"picker.up", "Up"},
	    {"picker.choose", "Choose this folder"},
	    {"picker.open", "Open"},
	    {"picker.home", "Home"},
	    {"cheatsWindow", "Cheats - {0}"},
	    {"cheats.all", "Select all"},
	    {"cheats.none", "Select none"},
	    {"ok", "OK"},
	    {"cancel", "Cancel"},
	    {"close", "Close"},
	    {"cheats.unreadable", "The cheat file cannot be read:\n{0}\n\n{1}"},
	    {"cheats.empty", "That file holds no cheats:\n{0}"},
	    {"cheats.changed",
	     "The cheat file changed while the window was open; nothing was written."},
	    {"cheats.countChanged", "The number of cheats does not match; nothing was written."},
	    {"cheats.verifyFailed", "The write could not be verified; the backup was put back."},
	    {"error.title", "KytyPS5 launcher"},
	    {"error.start", "Could not start:\n{0}"},
	    {"run.terminal", "Started in a terminal window ({0})."},
	};
	return table;
}

const Table& Chinese() {
	static const Table table {
	    {"output.1440", "1440p（DLSS / FSR 3）"},
	    {"output.1440NoUpscaler",
	     "1440p 需要选择 DLSS 或 FSR 3 作为放大方式；使用游戏自带 TAA 时输出仍为 4K。"},
	    {"render.capped1440", "3D 画面为 {0}x{1}（1440p 输出时最高 1440）。"},
	    {"render.raised1440",
	     "3D 画面为 {0}x{1}（1440p 输出时最低 720，或 480 用于超级性能模式）。"},
	    {"summary.1440", "1440p 输出"},
	    {"button.desktop", "添加到应用程序菜单"},
	    {"desktop.done", "KytyPS5 已添加到应用程序菜单（{0}）。"},
	    {"desktop.failed", "无法写入菜单项：{0}"},
	    {"app.subtitle", "Linux 版 PS5 模拟器"},
	    {"nav.play", "开始游戏"},
	    {"nav.display", "画面"},
	    {"nav.graphics", "画质"},
	    {"nav.game", "游戏"},
	    {"nav.shaders", "着色器"},
	    {"nav.advanced", "高级"},
	    {"nav.experimental", "实验性"},
	    {"nav.hint", "游戏中：F8 切换性能面板。"},

	    {"play.choose", "选择游戏"},
	    {"play.checks", "检查"},
	    {"play.gamebox", "游戏"},
	    {"play.summary", "当前设置"},
	    {"button.folder", "文件夹…"},
	    {"button.zar", ".zar…"},
	    {"button.play", "开始游戏"},
	    {"button.precompile", "预编译着色器"},
	    {"button.hints", "导出着色器提示"},
	    {"button.logs", "日志文件夹"},
	    {"button.readme", "说明文档"},
	    {"button.root", "模拟器文件夹"},
	    {"button.cheats", "选择…"},
	    {"button.getdlss", "下载并构建 DLSS…"},
	    {"game.none", "未选择游戏：请选择包含 eboot.bin 和 sce_sys 的游戏文件夹（游戏 dump），或其 "
	                  ".zar 压缩包。"},
	    {"game.bad",
	     "不是游戏：缺少 sce_sys/param.json（使用 .zar 时启动器旁需要 kyty_shader_precompile）。"},
	    {"game.loading", "正在读取游戏…"},
	    {"game.ok", "{0} · {1} · 版本 {2}"},
	    {"gpu.ok", "{0} · 显存 {1} GB"},
	    {"gpu.okNoMemory", "{0}"},
	    {"gpu.none", "未找到 Vulkan 显卡（是否安装了 Vulkan 驱动？）"},
	    {"ram.ok", "内存 {0} GB"},
	    {"ram.low", "内存 {0} GB：游戏时请关闭其他程序（读取存档时模拟器约占用 14 GB）"},
	    {"shaders.checking", "预编译着色器：检查中…"},
	    {"shaders.current", "着色器已为本显卡预编译"},
	    {"shaders.outdated", "着色器需要重新预编译（只编译新增部分）"},
	    {"shaders.outdatedAll", "着色器需要重新预编译（这次会全部重新编译一遍：换成新的翻译方式；在此之前现有的仍可使用）"},
	    {"shaders.none", "着色器尚未预编译：每种效果第一次出现时会卡顿"},
	    {"shaders.unknown", "预编译着色器：状态未知（请先启动一次游戏）"},
	    {"dlss.ok", "DLSS 可用（已找到 libkyty_dlss.so）"},
	    {"dlss.noRtx", "DLSS 需要 NVIDIA RTX 显卡"},
	    {"dlss.noLibrary", "DLSS：请先下载并构建其运行库（画质页，只需一次）"},
	    {"fsr.ok", "AMD FSR 3 可用（已内置）"},
	    {"emulator.missing", "启动器所在文件夹中没有 kyty_emulator（请用 ./build-linux.sh 构建）"},

	    {"summary.window", "窗口 {0}"},
	    {"summary.fullscreen", "全屏 {0}"},
	    {"summary.render", "3D {0}p"},
	    {"summary.taa", "游戏 TAA"},
	    {"summary.noAa", "无抗锯齿"},
	    {"summary.dlss", "DLSS"},
	    {"summary.fsr3", "FSR 3"},
	    {"summary.fsr", "FSR 3 帧生成"},
	    {"summary.hd", "1080p 输出"},
	    {"summary.hdr", "HDR"},
	    {"summary.vsync", "垂直同步"},
	    {"summary.fps120", "最高 120 帧"},
	    {"summary.options", "选项"},
	    {"summary.presentDirect", "直接呈现"},
	    {"summary.vrrFlips", "VRR 即时翻转"},
	    {"summary.wideBars", "过场与菜单 16:9"},
	    {"summary.antilag", "Anti-Lag"},
	    {"summary.fps30", "锁定 30 帧"},
	    {"summary.extra", "附加参数"},
	    {"summary.mangohud", "MangoHud"},

	    {"display.title", "画面"},
	    {"display.window", "窗口"},
	    {"display.rate", "帧率"},
	    {"resolution", "分辨率"},
	    {"resolution.desc", "窗口大小；全屏时为屏幕分辨率。"},
	    {"ratio", "画面比例"},
	    {"ratio.desc", "21:9 与 32:9 能看到两侧更多画面；HUD 保持比例。"},
	    {"ratio.169", "16:9（PS5）"},
	    {"widebars", "过场动画与菜单保持 16:9（黑边）"},
	    {"widebars.desc", "选 21:9 或 32:9 时：片头、过场动画、标题画面、加载画面与菜单保持 16:9，两侧为黑边，"
	                      "不拉伸（关闭时拉伸到整个宽度）。"},
	    {"fullscreen", "全屏"},
	    {"aspect", "保持画面比例（黑边）"},
	    {"aspect.desc", "全屏且屏幕比例不同时生效。"},
	    {"hdr", "HDR"},
	    {"hdr.desc", "需要在桌面的显示设置中打开 HDR（例如 KDE Plasma 6、GNOME 48 或更新版本，或 "
	                 "gamescope），且 Vulkan 驱动支持；否则为 SDR。"},
	    {"fps120", "最高 120 帧"},
	    {"fps120.desc", "游戏本身以超过 60 帧运行；过场动画仍为 60 帧。"},
	    {"vsync", "垂直同步"},
	    {"vsync.desc", "每帧等待显示器刷新（Vulkan FIFO）；VRR 显示器仍可自适应刷新。关闭时为 "
	                   "mailbox：同样无撕裂，但会渲染超过刷新率的帧。"},
	    {"lowlatency", "低延迟"},
	    {"lowlatency.desc", "只让一帧等待显示（而不是约两帧）：60 Hz 下从游戏出帧到显示约 15–23 "
	                        "ms（原约 30 ms）。迟到的帧不再有缓冲，帧间隔会稍不均匀。"},

	    {"graphics.title", "画质"},
	    {"graphics.resolution", "分辨率与超分"},
	    {"graphics.detail", "细节"},
	    {"graphics.fg", "帧生成"},
	    {"render", "3D 渲染高度"},
	    {"render.desc", "3D 画面为 {0}x{1}；越高越清晰，显卡负担越大。"},
	    {"render.capped", "3D 画面为 {0}x{1}（1080p 输出时最高 1080）。"},
	    {"render.raised",
	     "3D 画面为 {0}x{1}（DLSS 的 4K 输出时最低 1080，或 720 用于超级性能模式）。"},
	    {"render.native", "原生 4K"},
	    {"render.pro", "PS5 Pro"},
	    {"render.ps5", "PS5"},
	    {"upscaler", "超分辨率"},
	    {"upscaler.game", "游戏自带（PS5）"},
	    {"upscaler.dlss", "DLSS（PS5 Pro 路径）"},
	    {"upscaler.fsr", "FSR 3（PS5 Pro 路径）"},
	    {"upscaler.fsrNote", "用 AMD FSR 3 替代索尼的 PS5 Pro "
	                         "超分，任何显卡都能用（已内置，无需下载）：比游戏自带的拉伸更清晰。渲"
	                         "染高度列表标出了每个高度在所选游戏输出下对应的模式。"},
	    {"fsr.native", "原生抗锯齿"},
	    {"upscaler.off", "关闭（无抗锯齿）"},
	    {"upscaler.offNote", "关闭《恶魔之魂》自带的抗锯齿（TAA）及随之而来的镜头抖动：画面更锐利、更稳定，"
	                         "但边缘有锯齿、细节会闪烁。3D 按下方的渲染高度渲染，再像游戏自带那样拉伸到输出分辨率。"},
	    {"upscaler.ok", "用 DLSS 替代索尼的 PS5 Pro "
	                    "超分：比游戏自带的拉伸更清晰。渲染高度列表标出了每个高度在所选游戏输出下对"
	                    "应的 DLSS 模式。"},
	    {"upscaler.noRtx", "DLSS 需要 NVIDIA RTX 显卡：当前由游戏自带的 TAA 生成画面。"},
	    {"upscaler.noLibrary",
	     "DLSS 需要其运行库：请在下方「帧生成」中下载并构建 DLSS（只需一次）。"},
	    {"output", "游戏输出"},
	    {"output.auto", "自动"},
	    {"output.4k", "4K（PS5）"},
	    {"output.1080", "1080p"},
	    {"output.autoNow",
	     "自动：16:9 且游戏窗口（全屏时为屏幕）高度不超过 1080 时用 1080p，否则用 4K。当前：{0}。"},
	    {"output.desc", "1080p：游戏的 HD 布局，特效与 HUD 为 1920x1080；显卡负担更低。配合 DLSS "
	                    "时由 DLSS 把 3D 画面放大到 1080p（适合 1080p 屏幕）。1440p：由 DLSS 或 "
	                    "FSR 3 输出 2560x1440 画面（适合 1440p 屏幕），HUD 按 HD 布局放大。"},
	    {"dlssmodel", "DLSS 模型"},
	    {"dlssmodel.auto", "自动"},
	    {"dlssmodel.k", "Transformer（预设 K）"},
	    {"dlssmodel.e", "CNN（预设 E）"},
	    {"dlssmodel.m", "Transformer 2（预设 M，DLSS 4.5）"},
	    {"dlssmodel.l", "Transformer 2，最清晰（预设 L，DLSS 4.5）"},
	    {"dlssmodel.desc", "使用 DLSS 时生效。自动：由 DLSS "
	                       "版本决定模型。Transformer：更新、更清晰的模型；CNN：旧模型，在 RTX 20 "
	                       "和 30 系显卡上更轻（Transformer 模型在这些显卡上开销更大）。"},
	    {"sharpen", "锐化"},
	    {"sharpen.off", "关闭"},
	    {"sharpen.low", "低"},
	    {"sharpen.medium", "中"},
	    {"sharpen.high", "高"},
	    {"sharpen.desc", "使用 DLSS 或 FSR 3 时生效：锐化超分后的画面（RCAS）。DLSS 默认关闭：游戏自带的 TAA "
	                     "会锐化画面，DLSS 不会，所以细节看起来偏软。FSR 3 默认为「中」，即它自带的锐化。"
	                     "两种超分各自保存设置。等级越高，边缘越可能出现光晕。"},
	    {"dlss.dlaa", "DLAA"},
	    {"dlss.ultraquality", "超级质量"},
	    {"dlss.quality", "质量"},
	    {"dlss.balanced", "平衡"},
	    {"dlss.performance", "性能"},
	    {"dlss.ultra", "超级性能"},
	    {"quality", "画质预设"},
	    {"quality.standard", "标准（PS5）"},
	    {"quality.low", "低"},
	    {"quality.lowest", "最低"},
	    {"quality.desc",
	     "标准：PS5 的画面（显存低于 12 GB "
	     "时大贴图会去掉最大一级）。低：日光阴影更短，远景细节更简单，贴图边长减半。最低：阴影更短"
	     "，无环境光遮蔽与火把阴影，无贴花（血迹、焦痕），贴图为四分之一。适合 CPU 或显卡限制帧率的电脑。"},
	    {"framegen", "DLSS 帧生成"},
	    {"framegen.off", "关闭"},
	    {"framegen.ok",
	     "需 NVIDIA RTX 40 或更新（3 倍及以上需 RTX 50）。显示帧数 = 游戏帧数 × "
	     "倍数；开启垂直同步时不超过显示器刷新率（60 Hz 显示器用 2 倍时游戏为 30 帧）。"},
	    {"framegen.noRtx", "需要 NVIDIA RTX 40 或更新的显卡。"},
	    {"framegen.noLibrary", "需要 DLSS 运行库：请在下方下载并构建 DLSS（只需一次）。"},
	    {"dlssruntime", "DLSS 运行库"},
	    {"dlssruntime.desc",
	     "在终端中运行 get-dlss-linux.sh：下载 NVIDIA 的 DLSS SDK，并在模拟器旁构建 "
	     "libkyty_dlss.so。缺少 git 或 C++ "
	     "编译器时会询问是否安装（y/N，然后输入密码）。模拟器不附带任何 NVIDIA 文件。"},
	    {"dlssruntime.found", "位于 {0}"},
	    {"dlssruntime.none", "尚未构建"},
	    {"dlssruntime.noScript", "启动器所在文件夹中没有 get-dlss-linux.sh"},
	    {"fsr", "AMD FSR 3 帧生成（2 倍，任意显卡；替代 DLSS 帧生成）"},
	    {"fsr.desc",
	     "根据游戏的深度与运动，在两帧之间生成一帧。开启垂直同步时显示帧数不超过显示器刷新率。"},

	    {"game.title", "游戏"},
	    {"game.console", "主机"},
	    {"game.cheats", "作弊"},
	    {"language", "主机语言"},
	    {"language.desc", "告诉游戏的系统语言；游戏支持哪些文字与语音由游戏决定。"},
	    {"audiooutput", "音频输出"},
	    {"audiooutput.tv", "电视"},
	    {"audiooutput.headphones", "耳机"},
	    {"audiooutput.amplifier", "AV 功放"},
	    {"audiooutput.soundbar", "条形音箱"},
	    {"audiooutput.desc", "告诉游戏声音输出到哪里（与 PS5 相同）。游戏会按此混音，并显示在“设置 > 声音”中。"},
	    {"cheats", "作弊"},
	    {"cheats.desc", "_Patches/<title id>.json（GoldHEN 格式）：模拟器会应用其中开启的作弊项。"},
	    {"cheats.count", "已开启 {0}/{1} 项"},
	    {"cheats.noFile", "本游戏没有作弊文件"},
	    {"cheats.noGame", "请先选择游戏"},

	    {"shaders.title", "着色器"},
	    {"shaders.precompile", "预编译"},
	    {"shaders.share", "分享"},
	    {"duration.minutes", "约 {0} 分钟"},
	    {"duration.hours", "约 {0} 小时"},
	    {"precompile.desc",
	     "在终端中运行 "
	     "precompile-linux.sh：为本显卡把游戏的所有着色器编译一次，在这台电脑上{0}，期间 CPU "
	     "满载。关闭其窗口即可停止，下次会继续。更新显卡驱动后需要重新预编译。"},
	    {"precompile.quick", "快速：只编译游玩记录中用到的着色器（约一半时间）"},
	    {"ecores", "仅在能效核上预编译（较慢，但电脑仍保持流畅）"},
	    {"ecores.none", "此处理器只有一种核心。"},
	    {"hints.desc", "把你游玩时记录的着色器（不含游戏代码）导出为 my-hints-<game>.hints：附在 "
	                   "GitHub issue 里，之后的版本会为所有人预编译它们。"},

	    {"advanced.title", "高级"},
	    {"advanced.stability", "性能面板与稳定性"},
	    {"advanced.launcher", "启动器"},
	    {"hud", "性能面板"},
	    {"hud.off", "关闭"},
	    {"hud.rate", "帧率"},
	    {"hud.detailed", "详细"},
	    {"hud.desc",
	     "显示在游戏画面右上角：帧率与帧时间；「详细」另含显卡、处理器与内存。游戏中按 F8 切换。"},
	    {"mangohud", "MangoHud"},
	    {"mangohud.desc", "在游戏画面上显示 MangoHud（帧率、处理器与显卡负载、温度、功耗）。"},
	    {"mangohud.missing", "未安装 MangoHud（请安装发行版中的 mangohud 软件包）。"},
	    {"safegpu", "安全 GPU 模式"},
	    {"safegpu.desc", "防止显卡挂起导致的崩溃（device "
	                     "lost、驱动恢复时黑屏）：游戏的计算任务逐个执行。帧率约降低 "
	                     "37%。若没有此类崩溃请保持关闭。"},
	    {"uilang", "启动器语言"},
	    {"uilang.auto", "自动"},
	    {"uiscale", "界面大小"},
	    {"uiscale.auto", "自动"},
	    {"terminal", "终端"},
	    {"terminal.auto", "自动"},
	    {"terminal.desc", "显示游戏日志、预编译和 DLSS 构建过程的窗口。"},
	    {"terminal.none", "未找到终端程序：游戏会在没有日志窗口的情况下启动（日志写入日志文件夹），"
	                      "预编译和 DLSS 构建无法从这里启动。"},
	    {"credit", "设置项沿用 Windows 启动器（launcher-windows）。"},
	    {"advanced.gpu", "显卡"},
	    {"gpu", "使用的显卡"},
	    {"gpu.auto", "自动"},
	    {"gpu.desc", "适用于装有多块显卡的电脑：游戏和着色器预编译都使用所选显卡。自动：选第一块独"
	                 "立显卡。更改后请重新预编译着色器（着色器页）。"},
	    {"advanced.extra", "附加参数"},
	    {"extra", "附加参数"},
	    {"extra.desc",
	     "启动游戏时附加：KEY=VALUE 设置模拟器开关（KYTY_...；KEY= "
	     "关闭），并替换启动器自己的同名设置；--选项 值 "
	     "会传给模拟器命令行。用\"双引号\"保留空格。如果游戏因此无法启动，清空此框即可。"},

	    {"experimental.title", "实验性"},
	    {"experimental.intro",
	     "仍在试验中的选项。它们在一些电脑上有帮助，在另一些电脑上可能更糟：打开一项、玩一会儿，没"
	     "有帮助就再关掉。在对应的 issue "
	     "里留言（看到了什么，以及显卡和显示器型号）能帮助决定哪些将来成为默认。"},
	    {"experimental.display", "显示"},
	    {"experimental.amd", "AMD 显卡"},
	    {"experimental.framerate", "帧率"},
	    {"fps30", "锁定 30 帧（较慢的电脑）"},
	    {"fps30.desc",
	     "游戏稳定在 30 帧（和 PS5 的电影模式一样），而不是最高 60 "
	     "帧：处理器和显卡的工作量减半，帧间隔均匀，游戏速度不变。适合帧率在 30 到 60 "
	     "之间来回跳的电脑；配合 2 倍插帧显示 60 帧。过场动画和菜单仍为 60 帧。在 60 Hz、120 Hz "
	     "显示器上或开启 VRR 时最平稳。打开后「最高 120 帧」不再生效。"},
	    {"vrrflips", "画面完成即翻转（VRR 显示器）"},
	    {"vrrflips.desc",
	     "适用于开启了 G-SYNC、FreeSync 或 VRR 的显示器：每一帧完成后立即送去显示，而不是按游戏 60 Hz 的节拍，帧数低于 60 时画面运动依然均匀。"
	     "不支持 VRR 的显示器仍在下一次刷新时显示每一帧，和以前一样。"},
	    {"presentdirect", "直接呈现（VRR 显示器）"},
	    {"presentdirect.desc",
	     "每一帧准备好后立即送到显示器，不经过为保证垂直同步平稳而设的呈现线程。在某些显示器上，只"
	     "有这样 VRR 才会生效。请在关闭垂直同步时使用；没有 VRR 时帧间隔可能不够均匀。"},
	    {"antilag", "AMD Anti-Lag"},
	    {"antilag.desc", "由 AMD "
	                     "驱动安排游戏帧的时机，在「低延迟」的基础上进一步降低输入延迟（会同时打开"
	                     "低延迟）。已在 Linux 上测试；驱动不支持时只有低延迟生效。"},
	    {"antilag.notAmd", "仅限 AMD Radeon 显卡。"},
	    {"feedback", "反馈：issue #{0}"},

	    {"dialog.game", "游戏文件夹（eboot.bin、sce_sys）"},
	    {"dialog.zar", "打包为 ZArchive 的游戏文件夹"},
	    {"picker.up", "上一级"},
	    {"picker.choose", "选择此文件夹"},
	    {"picker.open", "打开"},
	    {"picker.home", "主目录"},
	    {"cheatsWindow", "作弊 - {0}"},
	    {"cheats.all", "全选"},
	    {"cheats.none", "全不选"},
	    {"ok", "确定"},
	    {"cancel", "取消"},
	    {"close", "关闭"},
	    {"cheats.unreadable", "无法读取作弊文件：\n{0}\n\n{1}"},
	    {"cheats.empty", "该文件中没有作弊项：\n{0}"},
	    {"cheats.changed", "窗口打开期间作弊文件被修改，未写入任何内容。"},
	    {"cheats.countChanged", "作弊项数量不一致，未写入任何内容。"},
	    {"cheats.verifyFailed", "写入后校验失败，已恢复备份。"},
	    {"error.title", "KytyPS5 启动器"},
	    {"error.start", "无法启动：\n{0}"},
	    {"run.terminal", "已在终端窗口中启动（{0}）。"},
	};
	return table;
}

const Table& Portuguese() {
	static const Table table {
	    {"output.1440", "1440p (DLSS / FSR 3)"},
	    {"output.1440NoUpscaler", "O 1440p precisa do DLSS ou do FSR 3 como upscaler; com o TAA do "
	                              "jogo a saída fica em 4K."},
	    {"render.capped1440",
	     "O 3D de Demon's Souls em {0}x{1} (no máximo 1440 com a saída 1440p)."},
	    {"render.raised1440", "O 3D de Demon's Souls em {0}x{1} (com a saída 1440p, no mínimo 720, "
	                          "ou 480 para o Ultra Desempenho)."},
	    {"summary.1440", "saída 1440p"},
	    {"button.desktop", "Adicionar ao menu de aplicativos"},
	    {"desktop.done", "O KytyPS5 está no menu de aplicativos agora ({0})."},
	    {"desktop.failed", "Não foi possível gravar a entrada do menu: {0}"},
	    {"app.subtitle", "PS5 Launcher For Linux"},
	    {"nav.play", "Jogar"},
	    {"nav.display", "Tela"},
	    {"nav.graphics", "Gráficos"},
	    {"nav.game", "Jogo"},
	    {"nav.shaders", "Shaders"},
	    {"nav.advanced", "Avançado"},
	    {"nav.experimental", "Experimental"},
	    {"nav.hint",
	     "No jogo: F8 alterna o painel de desempenho."},

	    {"play.choose", "Escolha o seu jogo"},
	    {"play.checks", "Verificações"},
	    {"play.gamebox", "Jogo"},
	    {"play.summary", "Configuração atual"},
	    {"button.folder", "Pasta…"},
	    {"button.zar", ".zar…"},
	    {"button.play", "JOGAR"},
	    {"button.precompile", "Pré-compilar shaders"},
	    {"button.hints", "Exportar dicas de shaders"},
	    {"button.logs", "Pasta de logs"},
	    {"button.readme", "README"},
	    {"button.root", "Pasta do emulador"},
	    {"button.cheats", "Escolher…"},
	    {"button.getdlss", "Baixar e compilar o DLSS…"},
	    {"game.none", "Nenhum jogo escolhido: escolha a pasta com eboot.bin e sce_sys (o dump do "
	                  "jogo) ou um arquivo .zar dela."},
	    {"game.bad", "Não é um jogo: não há sce_sys/param.json ali (um .zar precisa do "
	                 "kyty_shader_precompile ao lado do launcher)."},
	    {"game.loading", "Lendo o jogo…"},
	    {"game.ok", "{0} · {1} · versão {2}"},
	    {"gpu.ok", "{0} · {1} GB de memória de vídeo"},
	    {"gpu.okNoMemory", "{0}"},
	    {"gpu.none", "Nenhuma placa de vídeo Vulkan encontrada (o driver Vulkan está instalado?)"},
	    {"ram.ok", "{0} GB de RAM"},
	    {"ram.low", "{0} GB de RAM: feche outros programas ao jogar (o emulador usa cerca de 14 GB "
	                "ao carregar um save)"},
	    {"shaders.checking", "Shaders pré-compilados: verificando…"},
	    {"shaders.current", "Shaders pré-compilados para esta placa de vídeo"},
	    {"shaders.outdated", "Shaders: pré-compile de novo (só o que é novo é compilado)"},
	    {"shaders.outdatedAll", "Shaders: pré-compile de novo (desta vez todos, uma vez: uma tradução mais nova; os atuais funcionam até lá)"},
	    {"shaders.none", "Shaders ainda não pré-compilados: o jogo engasga na primeira vez que "
	                     "cada efeito aparece"},
	    {"shaders.unknown", "Shaders pré-compilados: desconhecido (inicie o jogo uma vez)"},
	    {"dlss.ok", "DLSS pronto (libkyty_dlss.so encontrada)"},
	    {"dlss.noRtx", "O DLSS precisa de uma placa NVIDIA RTX"},
	    {"dlss.noLibrary", "DLSS: baixe e compile o runtime uma vez (página Gráficos)"},
	    {"fsr.ok", "AMD FSR 3 pronto (embutido)"},
	    {"emulator.missing",
	     "Não há kyty_emulator ao lado do launcher (compile com ./build-linux.sh)"},

	    {"summary.window", "janela {0}"},
	    {"summary.fullscreen", "tela cheia {0}"},
	    {"summary.render", "3D {0}p"},
	    {"summary.taa", "TAA do jogo"},
	    {"summary.noAa", "sem anti-aliasing"},
	    {"summary.dlss", "DLSS"},
	    {"summary.fsr3", "FSR 3"},
	    {"summary.fsr", "geração de quadros FSR 3"},
	    {"summary.hd", "saída 1080p"},
	    {"summary.hdr", "HDR"},
	    {"summary.vsync", "VSync"},
	    {"summary.fps120", "até 120 fps"},
	    {"summary.options", "Opções"},
	    {"summary.presentDirect", "apresentação direta"},
	    {"summary.vrrFlips", "trocas VRR"},
	    {"summary.wideBars", "vídeos e menus em 16:9"},
	    {"summary.antilag", "Anti-Lag"},
	    {"summary.fps30", "trava em 30 fps"},
	    {"summary.extra", "opções extras"},
	    {"summary.mangohud", "MangoHud"},

	    {"display.title", "Tela"},
	    {"display.window", "Janela"},
	    {"display.rate", "Taxa de quadros"},
	    {"resolution", "Resolução"},
	    {"resolution.desc", "O tamanho da janela; em tela cheia, o modo da tela."},
	    {"ratio", "Proporção"},
	    {"ratio.desc", "21:9 e 32:9 mostram mais dos lados; o HUD mantém a forma."},
	    {"ratio.169", "16:9 (PS5)"},
	    {"widebars", "Vídeos e menus em 16:9 (barras pretas)"},
	    {"widebars.desc", "Com 21:9 ou 32:9: a abertura, os vídeos, a tela de título, as telas de carregamento "
	                      "e os menus em 16:9 com barras pretas nos lados, sem esticar (desligado: esticados na "
	                      "largura)."},
	    {"fullscreen", "Tela cheia"},
	    {"aspect", "Manter a proporção (barras pretas)"},
	    {"aspect.desc", "Em tela cheia numa tela de outro formato."},
	    {"hdr", "HDR"},
	    {"hdr.desc", "Precisa do HDR ligado nas configurações de tela do desktop (por exemplo KDE "
	                 "Plasma 6, GNOME 48 ou mais "
	                 "novo, ou gamescope) e de um driver Vulkan que o ofereça; senão, SDR."},
	    {"fps120", "Até 120 fps"},
	    {"fps120.desc", "Os quadros do próprio jogo acima de 60; os vídeos ficam em 60."},
	    {"vsync", "VSync"},
	    {"vsync.desc",
	     "Os quadros esperam a atualização da tela (Vulkan FIFO); telas VRR continuam se "
	     "adaptando. Desligado: "
	     "mailbox, que também não rasga a imagem, mas renderiza acima da taxa de atualização."},
	    {"lowlatency", "Baixa latência"},
	    {"lowlatency.desc", "Um quadro pronto espera a tela em vez de cerca de dois: a 60 Hz uns "
	                        "15-23 ms do quadro do jogo até "
	                        "a tela em vez de uns 30. Um quadro atrasado deixa de ser coberto, "
	                        "então o ritmo fica um pouco "
	                        "menos uniforme."},

	    {"graphics.title", "Gráficos"},
	    {"graphics.resolution", "Resolução e upscaling"},
	    {"graphics.detail", "Detalhe"},
	    {"graphics.fg", "Geração de quadros"},
	    {"render", "Altura de renderização 3D"},
	    {"render.desc", "O 3D de Demon's Souls em {0}x{1}; mais alto é mais nítido e exige mais da "
	                    "placa de vídeo."},
	    {"render.capped", "O 3D de Demon's Souls em {0}x{1} (no máximo 1080 com a saída 1080p)."},
	    {"render.raised", "O 3D de Demon's Souls em {0}x{1} (com a saída 4K do DLSS, no mínimo "
	                      "1080, ou 720 para o Ultra "
	                      "Performance)."},
	    {"render.native", "4K nativo"},
	    {"render.pro", "PS5 Pro"},
	    {"render.ps5", "PS5"},
	    {"upscaler", "Upscaler"},
	    {"upscaler.game", "Jogo (PS5)"},
	    {"upscaler.dlss", "DLSS (caminho do PS5 Pro)"},
	    {"upscaler.fsr", "FSR 3 (caminho do PS5 Pro)"},
	    {"upscaler.fsrNote", "AMD FSR 3 no lugar do upscaler do PS5 Pro da Sony, em qualquer placa "
	                         "de vídeo (embutido, nada "
	                         "a baixar): mais nítido que o esticado do jogo. Cada altura mostra o "
	                         "modo em que roda com a saída "
	                         "escolhida."},
	    {"fsr.native", "AA nativo"},
	    {"upscaler.off", "Desligado (sem anti-aliasing)"},
	    {"upscaler.offNote", "Demon's Souls sem o seu anti-aliasing (TAA) e sem o tremor de câmera "
	                         "que vem com ele: imagem mais nítida e estável, com bordas serrilhadas e "
	                         "cintilação em detalhes finos. O 3D na altura de renderização abaixo, "
	                         "esticado até a saída como o jogo faz."},
	    {"upscaler.ok", "DLSS no lugar do upscaler do PS5 Pro da Sony: mais nítido que o esticado "
	                    "do jogo. Cada altura mostra o "
	                    "modo do DLSS em que roda com a saída escolhida."},
	    {"upscaler.noRtx",
	     "O DLSS precisa de uma placa NVIDIA RTX: aqui o TAA do próprio jogo faz a imagem."},
	    {"upscaler.noLibrary",
	     "O DLSS precisa do runtime: Baixar e compilar o DLSS, em Geração de quadros (uma vez)."},
	    {"output", "Saída do jogo"},
	    {"output.auto", "Automático"},
	    {"output.4k", "4K (PS5)"},
	    {"output.1080", "1080p"},
	    {"output.autoNow", "Automático: 1080p para uma janela 16:9 (em tela cheia, a tela) de no "
	                       "máximo 1080 linhas, 4K nos "
	                       "outros casos. Aqui: {0}."},
	    {"output.desc", "1080p: o layout HD do jogo, efeitos e HUD em 1920x1080; menos trabalho "
	                    "para a placa de vídeo. Com DLSS, o DLSS faz a imagem 1080p a partir do 3D "
	                    "(para telas 1080p). 1440p: o DLSS ou o FSR 3 faz uma imagem 2560x1440 "
	                    "(para telas 1440p), com o HUD do layout HD ampliado para ela."},
	    {"dlssmodel", "Modelo do DLSS"},
	    {"dlssmodel.auto", "Automático"},
	    {"dlssmodel.k", "Transformer (preset K)"},
	    {"dlssmodel.e", "CNN (preset E)"},
	    {"dlssmodel.m", "Transformer 2 (preset M, DLSS 4.5)"},
	    {"dlssmodel.l", "Transformer 2, mais nítido (preset L, DLSS 4.5)"},
	    {"dlssmodel.desc",
	     "Com DLSS: Automático deixa o modelo para a versão do DLSS. Transformer é o modelo mais "
	     "novo e "
	     "nítido; CNN o antigo, mais leve nas RTX 20 e 30, onde o transformer custa mais."},
	    {"sharpen", "Nitidez"},
	    {"sharpen.off", "Desligada"},
	    {"sharpen.low", "Baixa"},
	    {"sharpen.medium", "Média"},
	    {"sharpen.high", "Alta"},
	    {"sharpen.desc",
	     "Com DLSS ou FSR 3: aumenta a nitidez da imagem do upscaler (RCAS). No DLSS vem desligada: o "
	     "TAA do próprio jogo aumenta a nitidez da imagem, o DLSS não, então os detalhes finos parecem "
	     "mais suaves. No FSR 3 vem em Média, a nitidez dele mesmo. Cada upscaler guarda a sua. Níveis "
	     "mais altos podem mostrar halos nas bordas."},
	    {"dlss.dlaa", "DLAA"},
	    {"dlss.ultraquality", "Ultraqualidade"},
	    {"dlss.quality", "Qualidade"},
	    {"dlss.balanced", "Equilibrado"},
	    {"dlss.performance", "Desempenho"},
	    {"dlss.ultra", "Ultra Desempenho"},
	    {"quality", "Predefinição gráfica"},
	    {"quality.standard", "Padrão (PS5)"},
	    {"quality.low", "Baixa"},
	    {"quality.lowest", "Mínima"},
	    {"quality.desc", "Padrão: a imagem do PS5 (abaixo de 12 GB de memória de vídeo as texturas "
	                     "grandes deixam de fora o "
	                     "maior nível). Baixa: sombras do sol mais curtas, detalhe distante mais "
	                     "simples, texturas com metade "
	                     "do tamanho por lado. Mínima: ainda mais curtas, sem sombras ambientes "
	                     "nem de tochas, sem decalques (sangue, marcas de queimado), texturas com um "
	                     "quarto do tamanho. Para PCs cujo processador ou placa de vídeo limita a "
	                     "taxa de quadros."},
	    {"framegen", "Geração de quadros DLSS"},
	    {"framegen.off", "Desligada"},
	    {"framegen.ok",
	     "NVIDIA RTX 40 ou mais nova (3x ou mais: RTX 50). Quadros mostrados = quadros do jogo x o "
	     "multiplicador; com VSync ficam limitados à taxa da tela (uma tela de 60 Hz em 2x roda o "
	     "jogo a 30 "
	     "fps)."},
	    {"framegen.noRtx", "Precisa de uma placa NVIDIA RTX 40 ou mais nova."},
	    {"framegen.noLibrary",
	     "Precisa do runtime do DLSS: Baixar e compilar o DLSS, abaixo (uma vez)."},
	    {"dlssruntime", "Runtime do DLSS"},
	    {"dlssruntime.desc",
	     "Roda o get-dlss-linux.sh num terminal: baixa o SDK do DLSS da NVIDIA e compila a "
	     "libkyty_dlss.so ao lado do emulador. Se faltar git ou um compilador C++, ele oferece "
	     "instalar "
	     "(y/N e depois a sua senha). Nada da NVIDIA vem com o emulador."},
	    {"dlssruntime.found", "encontrado em {0}"},
	    {"dlssruntime.none", "ainda não compilado"},
	    {"dlssruntime.noScript", "o get-dlss-linux.sh não está ao lado do launcher"},
	    {"fsr", "Geração de quadros AMD FSR 3 (2x, qualquer placa de vídeo; no lugar da geração de quadros DLSS)"},
	    {"fsr.desc", "Um quadro gerado entre dois do jogo, a partir da profundidade e do "
	                 "movimento. Com VSync os quadros "
	                 "mostrados ficam limitados à taxa da tela."},

	    {"game.title", "Jogo"},
	    {"game.console", "Console"},
	    {"game.cheats", "Cheats"},
	    {"language", "Idioma do console"},
	    {"language.desc",
	     "O idioma informado ao jogo; ele decide quais textos e vozes o jogo tem."},
	    {"audiooutput", "Saída de áudio"},
	    {"audiooutput.tv", "TV"},
	    {"audiooutput.headphones", "Fones de ouvido"},
	    {"audiooutput.amplifier", "Amplificador AV"},
	    {"audiooutput.soundbar", "Soundbar"},
	    {"audiooutput.desc", "Para onde o jogo é informado que o som vai, como num PS5. O jogo mixa o som "
	                         "para isso e mostra em Configurações > Som."},
	    {"cheats", "Cheats"},
	    {"cheats.desc", "_Patches/<title id>.json (estilo GoldHEN): o emulador aplica os cheats "
	                    "deixados ligados nele."},
	    {"cheats.count", "{0} de {1} ligados"},
	    {"cheats.noFile", "Não há arquivo de cheats para este jogo"},
	    {"cheats.noGame", "Escolha o jogo primeiro"},

	    {"shaders.title", "Shaders"},
	    {"shaders.precompile", "Pré-compilar"},
	    {"shaders.share", "Compartilhar"},
	    {"duration.minutes", "cerca de {0} minutos"},
	    {"duration.hours", "cerca de {0} horas"},
	    {"precompile.desc",
	     "Roda o precompile-linux.sh num terminal: compila uma vez todos os shaders do jogo para "
	     "esta placa "
	     "de vídeo, {0} neste PC com o processador no máximo. Fechar a janela para; a próxima vez "
	     "continua. Precisa de novo depois de atualizar o driver de vídeo."},
	    {"precompile.quick",
	     "Rápido: só os shaders como o jogo gravado os usou (cerca de metade do tempo)"},
	    {"ecores",
	     "Pré-compilar só nos núcleos de eficiência (mais lento, o PC continua responsivo)"},
	    {"ecores.none", "Este processador tem um só tipo de núcleo."},
	    {"hints.desc", "Os shaders que o seu jogo gravou, sem o código do jogo, como "
	                   "my-hints-<jogo>.hints: anexado a uma "
	                   "issue no GitHub, os próximos pacotes os pré-compilam para todos."},

	    {"advanced.title", "Avançado"},
	    {"advanced.stability", "Painel e estabilidade"},
	    {"advanced.launcher", "Launcher"},
	    {"hud", "Painel de desempenho"},
	    {"hud.off", "Desligado"},
	    {"hud.rate", "Taxa de quadros"},
	    {"hud.detailed", "Detalhado"},
	    {"hud.desc", "No canto superior direito da imagem: taxa e tempos de quadro; Detalhado "
	                 "acrescenta placa de vídeo, "
	                 "processador e memória. F8 no jogo alterna."},
	    {"mangohud", "MangoHud"},
	    {"mangohud.desc", "O painel do MangoHud (taxa de quadros, uso do processador e da placa de "
	                      "vídeo, temperaturas, "
	                      "consumo) sobre o jogo."},
	    {"mangohud.missing",
	     "O MangoHud não está instalado (pacote mangohud da sua distribuição)."},
	    {"safegpu", "Modo GPU seguro"},
	    {"safegpu.desc", "Evita os travamentos que vêm da placa de vídeo parar ('device lost', "
	                     "tela preta enquanto o driver se "
	                     "recupera): o trabalho de computação do jogo roda um passo após o outro. "
	                     "Cerca de 37% menos quadros. "
	                     "Deixe desligado se o jogo roda sem esses travamentos."},
	    {"uilang", "Idioma do launcher"},
	    {"uilang.auto", "Automático"},
	    {"uiscale", "Tamanho da interface"},
	    {"uiscale.auto", "Automático"},
	    {"terminal", "Terminal"},
	    {"terminal.auto", "Automático"},
	    {"terminal.desc",
	     "A janela onde rodam o log do jogo, a pré-compilação e a compilação do DLSS."},
	    {"terminal.none",
	     "Nenhum programa de terminal encontrado: o jogo inicia sem a janela de log (o log vai "
	     "para a pasta "
	     "de logs), e a pré-compilação e a compilação do DLSS não podem ser iniciadas daqui."},
	    {"credit", "As configurações seguem o launcher do Windows (launcher-windows)."},
	    {"advanced.gpu", "Placa de vídeo"},
	    {"gpu", "Placa de vídeo a usar"},
	    {"gpu.auto", "Automático"},
	    {"gpu.desc", "Para PCs com mais de uma placa de vídeo: o jogo e a pré-compilação usam a "
	                 "escolhida. Automático pega a "
	                 "primeira placa dedicada. Depois de mudar, pré-compile os shaders de novo "
	                 "(página Shaders)."},
	    {"advanced.extra", "Opções extras"},
	    {"extra", "Opções extras"},
	    {"extra.desc", "Somadas ao início do jogo: KEY=VALOR define uma chave do emulador "
	                   "(KYTY_...; KEY= desliga uma) e "
	                   "substitui a do launcher; um --opção valor vai para a linha de comando do "
	                   "emulador. \"Aspas duplas\" "
	                   "mantêm espaços. Se o jogo deixar de iniciar, limpe esta caixa."},

	    {"experimental.title", "Experimental"},
	    {"experimental.intro", "Opções ainda em teste. Podem ajudar em alguns PCs e piorar em "
	                           "outros: ligue uma, jogue um "
	                           "pouco e desligue de novo se não ajudar. Um comentário na issue "
	                           "indicada (o que você viu, com a "
	                           "sua placa de vídeo e tela) ajuda a decidir o que vira padrão."},
	    {"experimental.display", "Tela"},
	    {"experimental.amd", "Placas de vídeo AMD"},
	    {"experimental.framerate", "Taxa de quadros"},
	    {"fps30", "Travar em 30 fps (PCs mais lentos)"},
	    {"fps30.desc", "O jogo roda a 30 fps estáveis (como o modo cinema do PS5) em vez de até "
	                   "60: metade do trabalho para "
	                   "o processador e a placa de vídeo, tempos de quadro uniformes e a mesma "
	                   "velocidade de jogo. Para PCs "
	                   "que oscilam entre 30 e 60; com geração de quadros 2x mostra 60. Vídeos e "
	                   "menus ficam em 60. Mais "
	                   "suave em telas de 60 Hz e 120 Hz ou com VRR. Substitui \"Até 120 fps\"."},
	    {"vrrflips", "Quadros trocados quando prontos (telas VRR)"},
	    {"vrrflips.desc",
	     "Para telas com G-SYNC, FreeSync ou VRR ligado: cada quadro vai para a tela assim que fica "
	     "pronto, em vez de no ritmo de 60 Hz do jogo, e abaixo de 60 fps o movimento fica regular. "
	     "Uma tela sem VRR mostra cada quadro na próxima atualização, como antes."},
	    {"presentdirect", "Apresentação direta (telas VRR)"},
	    {"presentdirect.desc",
	     "Cada quadro vai para a tela assim que fica pronto, em vez de passar pela thread de "
	     "apresentação que mantém o VSync uniforme. Em algumas telas o VRR só engata assim. Use "
	     "com o "
	     "VSync desligado; sem VRR o ritmo dos quadros pode ficar menos uniforme."},
	    {"antilag", "AMD Anti-Lag"},
	    {"antilag.desc", "O driver da AMD sincroniza os quadros do jogo para cortar ainda mais o "
	                     "atraso dos comandos, além da "
	                     "Baixa latência (que ele também liga). Testado no Linux; sem suporte do "
	                     "driver, só a Baixa latência "
	                     "funciona."},
	    {"antilag.notAmd", "Só placas AMD Radeon."},
	    {"feedback", "Comentários: issue #{0}"},

	    {"dialog.game", "A pasta do jogo (eboot.bin, sce_sys)"},
	    {"dialog.zar", "A pasta do jogo compactada num ZArchive"},
	    {"picker.up", "Subir"},
	    {"picker.choose", "Escolher esta pasta"},
	    {"picker.open", "Abrir"},
	    {"picker.home", "Pasta pessoal"},
	    {"cheatsWindow", "Cheats - {0}"},
	    {"cheats.all", "Marcar todos"},
	    {"cheats.none", "Desmarcar todos"},
	    {"ok", "OK"},
	    {"cancel", "Cancelar"},
	    {"close", "Fechar"},
	    {"cheats.unreadable", "O arquivo de cheats não pode ser lido:\n{0}\n\n{1}"},
	    {"cheats.empty", "Esse arquivo não tem cheats:\n{0}"},
	    {"cheats.changed",
	     "O arquivo de cheats mudou enquanto a janela estava aberta; nada foi gravado."},
	    {"cheats.countChanged", "O número de cheats não confere; nada foi gravado."},
	    {"cheats.verifyFailed", "Não foi possível conferir a gravação; o backup foi restaurado."},
	    {"error.title", "Launcher do KytyPS5"},
	    {"error.start", "Não foi possível iniciar:\n{0}"},
	    {"run.terminal", "Iniciado numa janela de terminal ({0})."},
	};
	return table;
}

} // namespace

Language FromCode(const std::string& code) {
	if (code == "zh") return Language::Chinese;
	if (code == "pt") return Language::Portuguese;
	if (code == "en") return Language::English;
	return SystemLanguage();
}

Language SystemLanguage() {
	for (const char* name: {"LC_ALL", "LC_MESSAGES", "LANG"}) {
		const char* value = std::getenv(name);
		if (value == nullptr || *value == 0) continue;
		if (std::strncmp(value, "zh", 2) == 0) return Language::Chinese;
		if (std::strncmp(value, "pt", 2) == 0) return Language::Portuguese;
		return Language::English;
	}
	return Language::English;
}

const char* T(const char* key) {
	const Table* own = Current == Language::Chinese      ? &Chinese()
	                   : Current == Language::Portuguese ? &Portuguese()
	                                                     : nullptr;
	if (own != nullptr) {
		if (const auto it = own->find(key); it != own->end()) return it->second;
	}
	if (const auto it = English().find(key); it != English().end()) return it->second;
	return key;
}

std::string F(const char* key, const std::vector<std::string>& values) {
	const std::string_view text = T(key);
	std::string            out;
	for (size_t i = 0; i < text.size(); ++i) {
		if (text[i] == '{' && i + 2 < text.size() && text[i + 1] >= '0' && text[i + 1] <= '9') {
			const size_t close = text.find('}', i);
			const auto   index = static_cast<size_t>(text[i + 1] - '0');
			if (close != std::string_view::npos && index < values.size()) {
				out += values[index];
				i = close;
				continue;
			}
		}
		out += text[i];
	}
	return out;
}

const std::vector<std::string>& ConsoleLanguages() {
	static const std::vector<std::string> english {"Japanese",
	                                               "English (United States)",
	                                               "French (France)",
	                                               "Spanish (Spain)",
	                                               "German",
	                                               "Italian",
	                                               "Dutch",
	                                               "Portuguese (Portugal)",
	                                               "Russian",
	                                               "Korean",
	                                               "Chinese (Traditional)",
	                                               "Chinese (Simplified)",
	                                               "Finnish",
	                                               "Swedish",
	                                               "Danish",
	                                               "Norwegian",
	                                               "Polish",
	                                               "Portuguese (Brazil)",
	                                               "English (United Kingdom)",
	                                               "Turkish",
	                                               "Spanish (Latin America)",
	                                               "Arabic",
	                                               "French (Canada)",
	                                               "Czech",
	                                               "Hungarian",
	                                               "Greek",
	                                               "Romanian",
	                                               "Thai",
	                                               "Vietnamese",
	                                               "Indonesian"};
	static const std::vector<std::string> chinese {"日语",
	                                               "英语（美国）",
	                                               "法语（法国）",
	                                               "西班牙语（西班牙）",
	                                               "德语",
	                                               "意大利语",
	                                               "荷兰语",
	                                               "葡萄牙语（葡萄牙）",
	                                               "俄语",
	                                               "韩语",
	                                               "中文（繁体）",
	                                               "中文（简体）",
	                                               "芬兰语",
	                                               "瑞典语",
	                                               "丹麦语",
	                                               "挪威语",
	                                               "波兰语",
	                                               "葡萄牙语（巴西）",
	                                               "英语（英国）",
	                                               "土耳其语",
	                                               "西班牙语（拉丁美洲）",
	                                               "阿拉伯语",
	                                               "法语（加拿大）",
	                                               "捷克语",
	                                               "匈牙利语",
	                                               "希腊语",
	                                               "罗马尼亚语",
	                                               "泰语",
	                                               "越南语",
	                                               "印尼语"};
	static const std::vector<std::string> portuguese {"Japonês",
	                                                  "Inglês (Estados Unidos)",
	                                                  "Francês (França)",
	                                                  "Espanhol (Espanha)",
	                                                  "Alemão",
	                                                  "Italiano",
	                                                  "Holandês",
	                                                  "Português (Portugal)",
	                                                  "Russo",
	                                                  "Coreano",
	                                                  "Chinês (Tradicional)",
	                                                  "Chinês (Simplificado)",
	                                                  "Finlandês",
	                                                  "Sueco",
	                                                  "Dinamarquês",
	                                                  "Norueguês",
	                                                  "Polonês",
	                                                  "Português (Brasil)",
	                                                  "Inglês (Reino Unido)",
	                                                  "Turco",
	                                                  "Espanhol (América Latina)",
	                                                  "Árabe",
	                                                  "Francês (Canadá)",
	                                                  "Tcheco",
	                                                  "Húngaro",
	                                                  "Grego",
	                                                  "Romeno",
	                                                  "Tailandês",
	                                                  "Vietnamita",
	                                                  "Indonésio"};
	return Current == Language::Chinese      ? chinese
	       : Current == Language::Portuguese ? portuguese
	                                         : english;
}

} // namespace Strings
