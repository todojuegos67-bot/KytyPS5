#include "common/archive.h"
#include "common/common.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/file.h"
#include "common/lastWords.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "common/wheelConfig.h"
#include "emulator.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <string_view>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(KYTY_PGO_GENERATE)
#include <chrono>
#include <cstdlib>
#include <string>
#include <system_error>
#include <thread>

// compiler-rt profile runtime (instrumented builds, CMake KYTY_PGO_GENERATE).
extern "C" void __llvm_profile_set_filename(const char*);
extern "C" int  __llvm_profile_write_file(void);
#endif

using namespace Common;
using namespace Emulator;

#if defined(KYTY_PGO_GENERATE)
// The profile runtime writes its counters only at a normal exit, but the run scripts may stop the
// emulator with TerminateProcess. With KYTY_PGO_FILE=<file.profraw>, a thread writes them every
// KYTY_PGO_DUMP_SECONDS (default 20) to <file>.tmp and renames that over <file>, so the file always
// holds the last complete dump.
static void StartPgoDumps() {
	const char* file = std::getenv("KYTY_PGO_FILE");
	if (file == nullptr || *file == '\0') {
		::printf("PGO: instrumented build without KYTY_PGO_FILE: no profile is written\n");
		return;
	}
	const char* seconds_text = std::getenv("KYTY_PGO_DUMP_SECONDS");
	const int   seconds =
	    (seconds_text != nullptr && std::atoi(seconds_text) > 0) ? std::atoi(seconds_text) : 20;
	static const std::string target = file;
	static const std::string temp   = target + ".tmp";
	::printf("PGO: writing the profile to %s every %d s\n", target.c_str(), seconds);
	std::thread([seconds]() {
		for (uint32_t dump = 1;; dump++) {
			std::this_thread::sleep_for(std::chrono::seconds(seconds));
			__llvm_profile_set_filename(temp.c_str());
			const int       written = __llvm_profile_write_file();
			std::error_code error;
			if (written == 0) {
				std::filesystem::rename(temp, target, error);
			}
			::printf("PGO: dump %u %s\n", dump,
			         written != 0 ? "failed to write" : (error ? "failed to rename" : "written"));
			::fflush(stdout);
		}
	}).detach();
}
#endif

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
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf(
	    "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --wheel <true|false>                 Show the game a racing wheel fed by host\n"
	         "                                       devices (see --wheel-config).\n");
	::printf("  --wheel-config <json>                Wheel mapping file. Default: wheel.json in\n"
	         "                                       the working directory.\n");
	::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
	::printf("  --gpu-assisted-validation <t|f>      Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --vulkan-validation-fatal <t|f>      Exit on the first validation error (default\n"
	         "                                       true); false logs errors and continues.\n");
	::printf("  --shader-validation <true|false>     Enable shader validation.\n");
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
	    "  --trophy-notifications <true|false>   Show trophy unlock toasts and play their sound.\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
	::printf("  --bindless                           Sample heap-indexed images through bindless arrays.\n");
	::printf("  --reflections-every <n>              Redraw reflection cubes every nth frame.\n");
	::printf("  --no-float-image-atomics             Skip shaders that run integer image atomics on R32\n"
	         "                                       float images (they run by default).\n");
	::printf("  --skip-shaders <hash,...>            Skip the draws and dispatches of these guest shaders\n"
	         "                                       (hex hashes, as logged).\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone (default).\n");
	::printf("  --no-redzone                         Do not protect the guest SysV red zone.\n");
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
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Round 149: red zone protection is on unless --no-redzone is given. Guest code may keep data
	// in the 128 bytes below rsp (the SysV red zone); Windows writes its exception frames there,
	// and the emulator raises exceptions all the time for memory tracking (issue 25). Set here,
	// not in ConfigOptions, whose header the shader cache key covers.
	options.config.red_zone_protection_enabled = true;
#endif

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
		if (arg == "--bindless") {
			options.config.bindless_images_enabled = true;
			continue;
		}
		if (arg == "--float-image-atomics" || arg == "--no-float-image-atomics") {
			options.config.float_image_atomics_enabled = arg == "--float-image-atomics";
			continue;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (arg == "--redzone") {
			options.config.red_zone_protection_enabled = true;
			continue;
		}
		if (arg == "--no-redzone") {
			options.config.red_zone_protection_enabled = false;
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
		} else if (arg == "--gpu") {
			if (!ParseInt32(value, options.config.gpu_index)) {
				::printf("invalid gpu index: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--reflections-every") {
			// Round 86: the launcher's "Reflections every other frame". It reaches the renderer as
			// KYTY_CUBE_EVERY (which, set by hand, wins), so the configuration stays as it was.
			uint32_t every = 0;
			if (!ParseUint32(value, every) || every < 1 || every > 8) {
				::printf("invalid reflection cube interval (1 to 8): %s\n", value.c_str());
				return false;
			}
			if (std::getenv("KYTY_CUBE_EVERY") == nullptr) {
#if defined(_WIN32)
				_putenv_s("KYTY_CUBE_EVERY", value.c_str());
#else
				setenv("KYTY_CUBE_EVERY", value.c_str(), 0);
#endif
			}
		} else if (arg == "--vblank-frequency") {
			if (!ParseUint32(value, options.config.vblank_frequency)) {
				::printf("invalid vblank frequency: %s\n", value.c_str());
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
		} else if (arg == "--vulkan-validation-fatal") {
			if (!ParseBool(value, options.config.vulkan_validation_fatal)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-validation") {
			if (!ParseBool(value, options.config.shader_validation_enabled)) {
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
		} else if (arg == "--wheel") {
			bool enabled = false;
			if (!ParseBool(value, enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
			Config::Wheel::SetEnabled(enabled);
		} else if (arg == "--wheel-config") {
			Config::Wheel::SetConfigPath(Common::PathFromUtf8(value));
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--skip-shaders") {
			options.config.skip_shader_hashes = value;
		} else if (arg == "--spirv-debug-printf") {
			if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-linear-images") {
			if (!ParseBool(value, options.config.readback_linear_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--trophy-notifications") {
			if (!ParseBool(value, options.config.trophy_enabled)) {
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
// NVIDIA Smooth Motion (the driver's frame generation on RTX 40 and 50 cards) makes its frames
// with CUDA in the driver's present layer, which initializes CUDA only when the swapchain is
// created. By then the guest address space owns every free range below HOST_USER_MAX
// (memoryAddressSpace.inc), CUDA finds no room for its address range there, and cuInit fails with
// CUDA_ERROR_OUT_OF_MEMORY (2): Smooth Motion stays off (the layer's log is in C:/NvPresent/logs).
// So CUDA is initialized here, before any reservation, with the free ranges below a floor held
// for the duration of cuInit and released right after it: CUDA's range lands above the floor, and
// the guest address space later reserves everything around it. The floor sits above the whole
// guest range (HOST_USER_MAX + 1 = 1008 GiB): whatever CUDA reserved below it would be a hole in
// guest memory that a fixed-address mapping could hit.
// KYTY_CUDA_EARLY_INIT=1 turns it on (off by default: it adds a DLL, CUDA threads and startup
// time, and only drivers with Smooth Motion gain anything). KYTY_CUDA_VA_FLOOR_GB: the floor in
// GiB, 1024 (the default) to 65536. Without nvcuda.dll (no NVIDIA driver) nothing happens.
// From Jetsku/KytyPS5 (Luiz Henrique, PR 58, and Jetsku's opt-in, ec9943e6).
static void InitCudaEarly() {
	if (const char* value = std::getenv("KYTY_CUDA_EARLY_INIT");
	    value == nullptr || std::strcmp(value, "1") != 0) {
		return;
	}
	HMODULE cuda = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
	if (cuda == nullptr) {
		::printf("CUDA: nvcuda.dll not found (KYTY_CUDA_EARLY_INIT): no NVIDIA driver\n");
		_putenv_s("KYTY_CUDA_EARLY_RESULT", "nvcuda.dll not found");
		return;
	}
	using CuInit    = int(__stdcall*)(unsigned int);
	const auto init = reinterpret_cast<CuInit>(GetProcAddress(cuda, "cuInit"));
	if (init == nullptr) {
		return;
	}
	constexpr uint64_t MinFloorGb = 1024;
	constexpr uint64_t MaxFloorGb = 65536;
	uint64_t           floor_gb   = MinFloorGb;
	if (const char* value = std::getenv("KYTY_CUDA_VA_FLOOR_GB");
	    value != nullptr && std::atoi(value) > 0) {
		floor_gb =
		    std::clamp<uint64_t>(static_cast<uint64_t>(std::atoi(value)), MinFloorGb, MaxFloorGb);
	}
	const uint64_t floor = floor_gb << 30u;
	// Hold the free ranges below the floor.
	std::vector<void*> held;
	uint64_t           next = 0x10000;
	while (next < floor) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(next), &info, sizeof(info)) == 0) {
			break;
		}
		const auto base = reinterpret_cast<uint64_t>(info.BaseAddress);
		const auto end  = std::min<uint64_t>(floor, base + static_cast<uint64_t>(info.RegionSize));
		if (end <= next) {
			break;
		}
		// Reservations start on the 64 KiB allocation granularity.
		const auto start = (std::max<uint64_t>(base, next) + 0xffffu) & ~uint64_t {0xffff};
		if (info.State == MEM_FREE && end > start) {
			void* ptr = VirtualAlloc(reinterpret_cast<void*>(start), end - start, MEM_RESERVE,
			                         PAGE_NOACCESS);
			if (ptr != nullptr) {
				held.push_back(ptr);
			}
		}
		next = end;
	}
	const int result = init(0);
	for (void* ptr: held) {
		VirtualFree(ptr, 0, MEM_RELEASE);
	}
	const auto text =
	    fmt::format("initialized before the guest address space, above {} GiB "
	                "(cuInit {}{})",
	                floor_gb, result, result == 0 ? "" : ", NVIDIA Smooth Motion will not start");
	::printf("CUDA: %s\n", text.c_str());
	// For the log, which starts later (vma.cpp logs it with the commit limit).
	_putenv_s("KYTY_CUDA_EARLY_RESULT", text.c_str());
}
#endif

// KYTY_TIMER_RESOLUTION (default 1 ms; 0 leaves Windows' default, often 15.6 ms): the waits with a
// timeout (semaphores, condition variables, SyncOnAddress slices, the GPU thread's 100 us poll)
// otherwise round up to the system tick when no wake arrives. KYTY_POWER_THROTTLING=1 keeps
// Windows' EcoQoS (by default it is turned off for this process, so its threads are not moved to
// efficiency cores or slowed when the window is in the background). Both are once per run.
static void SetupWindowsScheduling() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const char* res_env = std::getenv("KYTY_TIMER_RESOLUTION");
	const auto  res_ms  = res_env != nullptr ? std::strtoul(res_env, nullptr, 10) : 1ul;
	std::string text;
	if (res_ms != 0) {
		using TimeBeginPeriod = UINT(WINAPI*)(UINT);
		HMODULE winmm         = LoadLibraryW(L"winmm.dll");
		auto*   begin = winmm != nullptr ? reinterpret_cast<TimeBeginPeriod>(
		                                     reinterpret_cast<void*>(GetProcAddress(winmm, "timeBeginPeriod")))
		                                 : nullptr;
		const bool ok = begin != nullptr && begin(static_cast<UINT>(res_ms)) == 0;
		text += fmt::format("timer resolution {} ms {}", res_ms, ok ? "set" : "NOT set");
	} else {
		text += "timer resolution left alone";
	}
	const char* pt_env = std::getenv("KYTY_POWER_THROTTLING");
	if (pt_env == nullptr || pt_env[0] != '1') {
		struct PowerThrottlingState {
			ULONG version;
			ULONG control_mask;
			ULONG state_mask;
		};
		// ProcessPowerThrottling = 4; EXECUTION_SPEED = 1, IGNORE_TIMER_RESOLUTION = 4.
		PowerThrottlingState state {1, 0x1u | 0x4u, 0};
		using SetProcessInformationFn = BOOL(WINAPI*)(HANDLE, int, LPVOID, DWORD);
		auto* set = reinterpret_cast<SetProcessInformationFn>(reinterpret_cast<void*>(
		    GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetProcessInformation")));
		const bool ok = set != nullptr && set(GetCurrentProcess(), 4, &state, sizeof(state)) != 0;
		text += fmt::format(", power throttling (EcoQoS) {}", ok ? "off" : "NOT changed");
	} else {
		text += ", power throttling left alone";
	}
	std::printf("Scheduling: %s (KYTY_TIMER_RESOLUTION=0, KYTY_POWER_THROTTLING=1 for Windows' defaults)\n",
	            text.c_str());
	std::fflush(stdout);
#endif
}

static int Main(int argc, char* argv[]) {
	SetupWindowsScheduling();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	InitCudaEarly();
#endif
	VirtualMemory::Init();
	InitializeThreads();
	// After the address-space reservations, as its thread's stack must not land in the guest
	// range: the report of an end no fault handler sees (round 125).
	Common::LastWords::Install();
#if defined(KYTY_PGO_GENERATE)
	// After the address-space reservations: the thread's stack must not land in the guest range.
	StartPgoDumps();
#endif

	RunOptions options;
	bool       show_help = false;

	if (argc < 2) {
		PrintUsage();
		return 0;
	}

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

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
