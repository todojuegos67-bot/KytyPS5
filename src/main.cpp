#include "common/archive.h"
#include "common/common.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/file.h"
#include "common/settingsFile.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "emulator.h"
#include "kytyGitVersion.h"
#include "local/ram-guard.h"

#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

#include <shellapi.h>
#endif

// KYTY_DEBUG_HEAP_FILL=<byte>: every operator new block starts filled with that byte, to tell
// reads of uninitialized memory apart from timing. Defaults to 0 (zero-fill) to prevent uninitialized
// garbage in Vulkan/graphics allocations (restoring pitch/menus). "=off" disables filling.
struct HeapFill {
	int         byte       = 0;
	std::size_t min        = 0;
	std::size_t max        = SIZE_MAX;
	uintptr_t   caller_min = 0;
	uintptr_t   caller_max = UINTPTR_MAX;
	uintptr_t   base       = 0;
	bool        sites      = false;
};

static void ParseRange(const char* name, uint64_t& min, uint64_t& max) {
	if (const char* range = std::getenv(name); range != nullptr) {
		char* end = nullptr;
		min       = std::strtoull(range, &end, 0);
		if (end != nullptr && *end == '-') {
			max = std::strtoull(end + 1, nullptr, 0);
		}
	}
}

static const HeapFill& GetHeapFill() {
	static const HeapFill fill = [] {
		HeapFill    f;
		f.byte = 0;
		const char* value = std::getenv("KYTY_DEBUG_HEAP_FILL");
		if (value != nullptr) {
			f.byte = std::strcmp(value, "off") == 0
			             ? -1
			             : static_cast<int>(std::strtol(value, nullptr, 0) & 0xff);
		}
		uint64_t min = f.min, max = f.max, caller_min = f.caller_min, caller_max = f.caller_max;
		ParseRange("KYTY_DEBUG_HEAP_FILL_RANGE", min, max);
		ParseRange("KYTY_DEBUG_HEAP_FILL_CALLER", caller_min, caller_max);
		f.min        = min;
		f.max        = max;
		f.caller_min = caller_min;
		f.caller_max = caller_max;
		f.sites      = std::getenv("KYTY_DEBUG_HEAP_SITES") != nullptr;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		f.base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
#endif
		return f;
	}();
	return fill;
}

static void NoteHeapSite(uintptr_t offset, std::size_t size) {
	static std::atomic<uintptr_t> sites[1u << 14u];
	for (uintptr_t i = (offset * 0x9e3779b97f4a7c15ull) >> 50u;; i = (i + 1) % std::size(sites)) {
		uintptr_t current = sites[i].load(std::memory_order_relaxed);
		if (current == offset) {
			return;
		}
		if (current == 0 && sites[i].compare_exchange_strong(current, offset)) {
			std::printf("heap-site 0x%llx size %zu\n", static_cast<unsigned long long>(offset),
			            size);
			std::fflush(stdout);
			return;
		}
		if (current == offset) {
			return;
		}
	}
}

void* operator new(std::size_t size) {
	void* block = std::malloc(size != 0 ? size : 1);
	if (block == nullptr) {
		throw std::bad_alloc();
	}
	if (const auto& fill = GetHeapFill(); fill.byte >= 0) {
		const auto caller = reinterpret_cast<uintptr_t>(__builtin_return_address(0)) - fill.base;
		if (fill.sites && size >= 1024) {
			NoteHeapSite(caller, size);
		}
		if (size >= fill.min && size < fill.max && caller >= fill.caller_min &&
		    caller < fill.caller_max) {
			std::memset(block, fill.byte, size);
		}
	}
	return block;
}

void operator delete(void* p) noexcept {
	std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
	std::free(p);
}

void* operator new[](std::size_t size) {
	return ::operator new(size);
}

void operator delete[](void* p) noexcept {
	::operator delete(p);
}

void operator delete[](void* p, std::size_t) noexcept {
	::operator delete(p);
}

using namespace Common;
using namespace Emulator;

static std::string GetBuildString() {
	Date date = Date::FromMacros(std::string(__DATE__));

#if KYTY_BUILD == KYTY_BUILD_DEBUG
	std::string type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
	std::string type = "Release";
#else
	std::string type = "????";
#endif

	std::string compiler = Debug::GetCompiler() + "-" + Debug::GetLinker();

	std::string str =
	    fmt::format("{}, {}, ver = {}, git = {}, date = {}", type.c_str(), compiler.c_str(),
	                KYTY_VERSION, KYTY_GIT_VERSION, date.ToString().c_str());

	return str;
}

static void PrintUsage() {
	::printf("%s\n", GetBuildString().c_str());
	::printf("kyty_emulator --game <dir|elf|zar> [options]\n\n");
	::printf("Options can also be set in kyty_settings.ini in the working directory, one per\n"
	         "line without \"--\" (e.g. gpu-timestamp-scale = 115). The command line overrides it.\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf|zar>                 Game directory, ELF, or ZArchive to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
	::printf("  --controller-color <#RRGGBB>        Override the controller lightbar color.\n");
	::printf("  --controller-volume <0-100>         DualSense speaker volume. Default: 50.\n");
	::printf("  --controller-vibration <0-100>      DualSense vibration intensity. Default: 100.\n");
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
	::printf("  --bda-sync <value>                   Selective, Legacy, or SelectiveChecked. "
	         "Default: Selective.\n");
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf(
	    "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
	::printf("  --no-splash                          Disable the startup splash screen.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --reported-resolution <0|1|2>        Console output resolution reported to the game:\n"
	         "                                       1 = 1080p, 2 = 2160p, 0 = automatic. Default: 0.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
	::printf("  --gpu-assisted-validation <t|f>      Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --shader-validation <true|false>     Enable shader validation.\n");
	::printf("  --shader-precompile <true|false>     Replay recorded shaders before drawing. "
	         "Default: true.\n");
	::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
	::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump <true|false>   Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump <true|false>   Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profile                            Enable the Tracy profiler.\n");
	::printf("  --spirv-debug-printf <true|false>    Enable SPIR-V debug printf.\n");
	::printf(
	    "  --readback-linear-images <true|false> Read back writable linear images on submit.\n");
	::printf(
	    "  --sync-raw-image-buffers <true|false> Synchronize raw reads of GPU images. Default: false.\n");
	::printf(
	    "  --trophy-notifications <true|false>   Show trophy unlock toasts and play their sound.\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
	::printf("  --drain-stats <seconds>              Report GPU waits by cause every N seconds.\n");
	::printf("  --dcc-gpu-clear <true|false>         Apply GPU-written DCC clears on the GPU. "
	         "Default: true.\n");
	::printf("  --async-submit <true|false>          Submit GPU work from a dedicated queue thread. "
	         "Default: true.\n");
	::printf("  --gpu-mesh-indirect <true|false>     Build mesh-emulated indirect draws on the GPU. "
	         "Default: true.\n");
	::printf("  --gpu-frames-ahead <0-3>             Frames the game may build ahead of the GPU "
	         "thread. 0 waits for idle. Default: 0.\n");
	::printf("  --label-flush-interval-us <us>       Minimum time between RELEASE_MEM submits. "
	         "Default: 2000.\n");
	::printf("  --gpu-timestamp-scale <100-200>      Stretch GPU time the game measures, in "
	         "percent, so dynamic resolution keeps headroom. Default: 100 (off).\n");
	::printf("  --pipeline-libraries <true|false>    Build new graphics pipelines from cached, "
	         "separately compiled parts where the driver supports it. Default: true.\n");
	::printf("  --async-pipelines <true|false>       Skip draws whose new pipeline is still "
	         "compiling instead of stalling; they appear a few frames late. Default: false.\n");
	::printf("  --fsr-upscaling <true|false>         Scale a frame smaller than the window with AMD "
	         "FSR 1.\n");
	::printf("  --fsr-softness <0-20>                How soft FSR 1 leaves edges: 0 the sharpest, "
	         "10 by default.\n");
	::printf("  --frame-generation <true|false>      Show a generated frame between every two of "
	         "the game's (AMD FSR 3). Default: false.\n");
	::printf("  --relaxed-readback <true|false>      Let game threads read memory the GPU is still "
	         "writing without waiting; a value can be a frame old. Default: false.\n");
	::printf("  --speculative-draws <true|false>     Prepare draws' shader resources on a second "
	         "thread ahead of the GPU thread. Default: true.\n");
	::printf("  --record-thread <true|false>         Record the GPU thread's Vulkan commands on the "
	         "submit thread. Default: true.\n");
	::printf("  --hardware-buffer-bounds <true|false> Let the GPU check shader buffer ranges where "
	         "it supports that, instead of checks compiled into each shader. Default: true.\n");
	::printf(
	    "  --skip-notice-screen <true|false>    Skip startup logos and notices in supported games.\n"
	    "                                      Default: false.\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

static bool ParseBool(const std::string& value, bool& out) {
	if (Common::EqualNoCase(value, "true") || value == "1" || Common::EqualNoCase(value, "yes") ||
	    Common::EqualNoCase(value, "on")) {
		out = true;
		return true;
	}

	if (Common::EqualNoCase(value, "false") || value == "0" || Common::EqualNoCase(value, "no") ||
	    Common::EqualNoCase(value, "off")) {
		out = false;
		return true;
	}

	return false;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value.c_str());
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUint32(const std::string& value, uint32_t& out) {
	uint32_t number   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseControllerColor(const std::string& value, Config::ControllerColor& out) {
	if (value.size() != 7 || value[0] != '#') {
		return false;
	}
	uint32_t rgb = 0;
	auto [end, error] = std::from_chars(value.data() + 1, value.data() + value.size(), rgb, 16);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8),
	       static_cast<uint8_t>(rgb)};
	return true;
}

static bool ParseInt32(const std::string& value, int32_t& out) {
	int32_t number    = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;
	options.config.shader_optimization_type = Config::ShaderOptimizationType::Performance;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (arg == "--rd") {
			options.config.renderdoc_enabled = true;
			continue;
		}

		if (arg == "--fullscreen") {
			options.config.fullscreen_enabled = true;
			continue;
		}

		if (arg == "--hide-cursor") {
			options.config.hide_cursor_enabled = true;
			continue;
		}

		if (arg == "--no-splash") {
			options.config.splash_enabled = false;
			continue;
		}

		if (arg == "--vr") {
			options.config.vr_enabled = true;
			continue;
		}

		if (arg == "--amd-cpu") {
			options.config.amd_cpu_enabled = true;
			continue;
		}

		if (arg == "--playgo-hack") {
			options.config.playgo_hack_enabled = true;
			continue;
		}

		if (arg == "--tessellation") {
			options.config.tessellation_enabled = true;
			continue;
		}

		if (arg == "--profile") {
			options.config.profiler_enabled = true;
			continue;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (arg == "--redzone") {
			options.config.red_zone_protection_enabled = true;
			continue;
		}
#endif

		if (!arg.starts_with("--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value           = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (Common::File::IsDirectoryExisting(path)) {
				options.app0_dir = path;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::IsSupportedArchive(path) && Common::File::IsFileExisting(path)) {
				const auto root = Common::MakeArchivePath(path);
				if (!Common::File::IsFileExisting(root / "eboot.bin")) {
					::printf("Archive does not contain eboot.bin: %s\n", value.c_str());
					return false;
				}
				options.app0_dir = root;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(path)) {
				options.app0_dir = path.parent_path();

				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}

				options.elf = std::filesystem::path("/app0") / path.filename();
			} else {
				::printf("--game must point to an existing directory, ELF, or archive: %s\n",
				         value.c_str());
				return false;
			}
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (!Common::File::IsFileExisting(path)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = path;
		} else if (arg == "--screen-width") {
			if (!ParseUint32(value, options.config.screen_width) ||
			    options.config.screen_width == 0) {
				::printf("invalid screen width: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--screen-height") {
			if (!ParseUint32(value, options.config.screen_height) ||
			    options.config.screen_height == 0) {
				::printf("invalid screen height: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mic") {
			options.config.audio_input_device = value;
		} else if (arg == "--controller-color") {
			Config::ControllerColor color {};
			if (!ParseControllerColor(value, color)) {
				::printf("invalid controller color (expected #RRGGBB): %s\n", value.c_str());
				return false;
			}
			options.config.controller_color = color;
		} else if (arg == "--controller-volume") {
			if (!ParseUint32(value, options.config.controller_speaker_volume) ||
			    options.config.controller_speaker_volume > 100) {
				::printf("invalid controller volume: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--controller-vibration") {
			if (!ParseUint32(value, options.config.controller_vibration_intensity) ||
			    options.config.controller_vibration_intensity > 100) {
				::printf("invalid controller vibration intensity: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--bda-sync") {
			if (!ParseEnum(value, options.config.bda_sync_mode)) {
				::printf("invalid BDA synchronization mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--gpu") {
			if (!ParseInt32(value, options.config.gpu_index)) {
				::printf("invalid gpu index: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vblank-frequency") {
			if (!ParseUint32(value, options.config.vblank_frequency)) {
				::printf("invalid vblank frequency: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--reported-resolution") {
			if (!ParseUint32(value, options.config.reported_resolution) ||
			    options.config.reported_resolution > 2) {
				::printf("invalid reported resolution: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vulkan-validation") {
			if (!ParseBool(value, options.config.vulkan_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-assisted-validation") {
			if (!ParseBool(value, options.config.gpu_assisted_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-validation") {
			if (!ParseBool(value, options.config.shader_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-precompile") {
			if (!ParseBool(value, options.config.shader_precompile_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-optimization-type") {
			if (!ParseEnum(value, options.config.shader_optimization_type)) {
				::printf("invalid shader optimization type: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = Common::PathFromUtf8(value);
		} else if (arg == "--command-buffer-dump") {
			if (!ParseBool(value, options.config.command_buffer_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
		} else if (arg == "--graphics-debug-dump") {
			if (!ParseBool(value, options.config.graphics_debug_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--spirv-debug-printf") {
			if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--drain-stats") {
			uint32_t interval = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), interval);
			if (error != std::errc {} || end != value.data() + value.size() || interval > 3600) {
				::printf("invalid drain-stats interval: %s\n", value.c_str());
				return false;
			}
			options.config.drain_stats_interval = interval;
		} else if (arg == "--label-flush-interval-us") {
			uint32_t interval = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), interval);
			if (error != std::errc {} || end != value.data() + value.size() || interval > 100000) {
				::printf("invalid label-flush-interval-us: %s\n", value.c_str());
				return false;
			}
			options.config.label_flush_interval_us = interval;
		} else if (arg == "--gpu-frames-ahead") {
			uint32_t frames = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), frames);
			if (error != std::errc {} || end != value.data() + value.size() || frames > 3) {
				::printf("invalid gpu-frames-ahead (0-3): %s\n", value.c_str());
				return false;
			}
			options.config.gpu_frames_ahead = frames;
		} else if (arg == "--dcc-gpu-clear") {
			if (!ParseBool(value, options.config.dcc_gpu_clear_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-mesh-indirect") {
			if (!ParseBool(value, options.config.gpu_mesh_indirect_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-timestamp-scale") {
			uint32_t percent = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), percent);
			if (error != std::errc {} || end != value.data() + value.size() || percent < 100 ||
			    percent > 200) {
				::printf("invalid gpu-timestamp-scale (100-200): %s\n", value.c_str());
				return false;
			}
			options.config.gpu_timestamp_scale_percent = percent;
		} else if (arg == "--pipeline-libraries") {
			if (!ParseBool(value, options.config.pipeline_libraries_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--async-pipelines") {
			if (!ParseBool(value, options.config.async_pipelines_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--speculative-draws") {
			if (!ParseBool(value, options.config.speculative_draws_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--record-thread") {
			if (!ParseBool(value, options.config.record_thread_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--hardware-buffer-bounds") {
			if (!ParseBool(value, options.config.hardware_buffer_bounds)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--fsr-upscaling") {
			if (!ParseBool(value, options.config.fsr_upscaling_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--fsr-softness") {
			const auto tenths = std::strtoul(value.c_str(), nullptr, 10);
			if (tenths > 20) {
				::printf("invalid fsr-softness (0-20): %s\n", value.c_str());
				return false;
			}
			options.config.fsr_softness_tenths = static_cast<uint32_t>(tenths);
		} else if (arg == "--frame-generation") {
			if (!ParseBool(value, options.config.frame_generation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--relaxed-readback") {
			if (!ParseBool(value, options.config.relaxed_readback_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--async-submit") {
			if (!ParseBool(value, options.config.async_submit_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-linear-images") {
			if (!ParseBool(value, options.config.readback_linear_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--sync-raw-image-buffers") {
			if (!ParseBool(value, options.config.sync_raw_image_buffers)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--trophy-notifications") {
			if (!ParseBool(value, options.config.trophy_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--skip-notice-screen") {
			if (!ParseBool(value, options.config.skip_notice_screen)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Started with nothing to run, which is what a double click does: the player wanted the launcher
// that sits beside this executable. Without one, say what this executable needs instead of
// flashing a console that closes. Returns false when the usage text is the answer: from a shell,
// and for a program reading our output, which is how the launcher asks for the version. A
// launcher opened from here marks its environment, so that nothing it starts opens another.
static bool OpenLauncherInstead() {
	static constexpr wchar_t opened_mark[] = L"KYTY_OPENED_LAUNCHER";
	DWORD                    attached[2] {};
	if (GetConsoleProcessList(attached, 2) != 1 ||
	    GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) != FILE_TYPE_CHAR ||
	    GetEnvironmentVariableW(opened_mark, nullptr, 0) != 0) {
		return false;
	}
	std::wstring path(MAX_PATH, L'\0');
	for (;;) {
		const DWORD length =
		    GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
		if (length == 0) {
			return false;
		}
		if (length < path.size()) {
			path.resize(length);
			break;
		}
		path.resize(path.size() * 2);
	}
	const auto      folder   = std::filesystem::path(path).parent_path();
	const auto      launcher = folder / L"launcher.exe";
	std::error_code error;
	if (std::filesystem::exists(launcher, error)) {
		SetEnvironmentVariableW(opened_mark, L"1");
		const auto opened = reinterpret_cast<INT_PTR>(ShellExecuteW(
		    nullptr, L"open", launcher.c_str(), nullptr, folder.c_str(), SW_SHOWNORMAL));
		if (opened > 32) {
			return true;
		}
	}
	MessageBoxW(nullptr,
	            L"This is the emulator itself, and it needs a game to run.\n\n"
	            L"Open launcher.exe and start the game from there, or run:\n"
	            L"kyty_emulator.exe --game <game folder>",
	            L"brunoKytyPs5", MB_OK | MB_ICONINFORMATION);
	return true;
}
#endif

static int Main(int argc, char* argv[]) {
	VirtualMemory::Init();
	InitializeThreads();

	RunOptions options;
	bool       show_help = false;

	std::vector<std::string> settings;
	if (!SettingsFile::LoadArguments(settings)) {
		return 1;
	}
	if (!settings.empty()) {
		::printf("Settings: %s\n", SettingsFile::FileName);
	}

	if (argc < 2 && settings.empty()) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (OpenLauncherInstead()) {
			return 0;
		}
#endif
		PrintUsage();
		return 0;
	}

	// The settings file's options go first, so the same option on the command line overrides it.
	std::vector<char*> arguments {argv[0]};
	for (auto& setting: settings) {
		arguments.push_back(setting.data());
	}
	arguments.insert(arguments.end(), argv + 1, argv + argc);
	argc = static_cast<int>(arguments.size());
	argv = arguments.data();

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

	// bruno-astro-1025: the RAM guard (local/ram-guard.h): the low-RAM profile logged, the pressure watcher started.
	RamGuard::ApplyLowRamDefaults();
	RamGuard::Start();

	Run(options);

	return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

int wmain(int argc, wchar_t* argv[]) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
