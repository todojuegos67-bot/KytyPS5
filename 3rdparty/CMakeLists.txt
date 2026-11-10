set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(cpuinfo)

# The Windows guest red-zone patcher needs a decoder/encoder and a small
# trampoline assembler. Pin the same revisions used by shadPS4's implementation.
include(FetchContent)

set(XBYAK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
	xbyak
	GIT_REPOSITORY https://github.com/herumi/xbyak.git
	GIT_TAG 44a72f369268f7d552650891b296693e91db86bb
	GIT_SHALLOW FALSE
)

set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_DOXYGEN OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
	zydis
	GIT_REPOSITORY https://github.com/zyantific/zydis.git
	GIT_TAG 120e0e705f8e3b507dc49377ac2879979f0d545c
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW FALSE
)

FetchContent_MakeAvailable(xbyak zydis)

# ZArchive (.zar): random-access reads of compressed game dumps (upstream's pinned revisions and reader patch). The
# library and its zstd dependency; zarchive_tool (not built by default) packs a game folder into a .zar.
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
	zstd
	GIT_REPOSITORY https://github.com/facebook/zstd.git
	GIT_TAG f8745da6ff1ad1e7bab384bd1f9d742439278e99
	GIT_SHALLOW FALSE
	SOURCE_SUBDIR build/cmake
)
FetchContent_Declare(
	zarchive_source
	GIT_REPOSITORY https://github.com/Exzap/ZArchive.git
	GIT_TAG d2c717730092c7bf8cbb033b12fd4001b7c4d932
	GIT_SHALLOW FALSE
	PATCH_COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace
		"${CMAKE_CURRENT_SOURCE_DIR}/patches/zarchive-reader.patch"
	SOURCE_SUBDIR kyty-library-only
)
FetchContent_MakeAvailable(zstd zarchive_source)

# Opus: the DualSense's Bluetooth link carries the pad speaker as Opus packets inside HID reports (there is no
# USB audio endpoint over Bluetooth), so src/libs/dualSenseBluetooth.cpp encodes them with this.
set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
	opus
	GIT_REPOSITORY https://github.com/xiph/opus.git
	GIT_TAG ddbe48383984d56acd9e1ab6a090c54ca6b735a6 # v1.5.2
	GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(opus)

# KYTY_MIMALLOC: the emulator's operator new/delete (src/main.cpp) from mimalloc instead of the CRT heap.
if(KYTY_MIMALLOC)
	# Vendored: 3rdparty/mimalloc is mimalloc v2.5.2 (2406cc5f), the files src/static.c includes (Windows only),
	# include/ and LICENSE, unchanged. Built as its own CMake build of the static library would build it here: one
	# translation unit (static.c) compiled as C++ (MI_USE_CXX, as with MSVC), MI_STATIC_LIB, the release defines.
	add_library(mimalloc-static STATIC mimalloc/src/static.c)
	set_source_files_properties(mimalloc/src/static.c PROPERTIES LANGUAGE CXX)
	set_target_properties(mimalloc-static PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
	target_include_directories(mimalloc-static PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/mimalloc/include")
	target_compile_definitions(mimalloc-static PRIVATE MI_BUILD_RELEASE MI_CMAKE_BUILD_TYPE=release MI_STATIC_LIB NOMINMAX)
	target_link_libraries(mimalloc-static PUBLIC psapi shell32 user32 advapi32 bcrypt)
	if(MSVC)
		target_compile_options(mimalloc-static PRIVATE /W0 /Zc:__cplusplus) # (/Zc:__cplusplus: as mimalloc's own build)
	else()
		target_compile_options(mimalloc-static PRIVATE -w)
	endif()
	# SPIRV-Tools would add mimalloc a second time.
	set(SPIRV_TOOLS_USE_MIMALLOC OFF CACHE BOOL "" FORCE)
endif()

add_library(zarchive STATIC
	"${zarchive_source_SOURCE_DIR}/src/zarchivereader.cpp"
	"${zarchive_source_SOURCE_DIR}/src/zarchivewriter.cpp"
	"${zarchive_source_SOURCE_DIR}/src/sha_256.c"
)
add_library(ZArchive::zarchive ALIAS zarchive)
target_include_directories(zarchive SYSTEM PUBLIC "${zarchive_source_SOURCE_DIR}/include")
target_include_directories(zarchive PRIVATE "${zstd_SOURCE_DIR}/lib")
target_link_libraries(zarchive PRIVATE libzstd_static)
target_compile_features(zarchive PRIVATE cxx_std_20)
if(MSVC)
	target_compile_options(zarchive PRIVATE /W0)
else()
	target_compile_options(zarchive PRIVATE -w)
endif()

add_executable(zarchive_tool EXCLUDE_FROM_ALL "${zarchive_source_SOURCE_DIR}/src/main.cpp")
target_link_libraries(zarchive_tool PRIVATE zarchive)
target_compile_features(zarchive_tool PRIVATE cxx_std_20)
set_target_properties(zarchive_tool PROPERTIES OUTPUT_NAME zarchive)
if(MSVC)
	target_compile_options(zarchive_tool PRIVATE /W0)
else()
	target_compile_options(zarchive_tool PRIVATE -w)
endif()

set(SDL_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(SDL2)
# SDL without its C library (SDL_LIBC off, its MSVC default) is compiled with /Gs1048576: no stack probe for a frame
# under 1 MiB, as it has no __chkstk of its own. Each SDL function carries that probe size, and one that ThinLTO inlines
# into the emulator's code hands it to the caller (LLVM's adjustCallerStackProbeSize, for a caller without its own):
# the caller's frame then skips the stack's guard page. The DualSense Bluetooth sender thread's 16 KiB frame did, and
# the process ended at start-up with 0xC0000005 and no word in the log (the fault cannot be delivered on that stack).
# The emulator links the C runtime, whose __chkstk serves SDL too.
if(MSVC AND NOT SDL_LIBC AND TARGET SDL2-static)
	get_target_property(sdl_compile_options SDL2-static COMPILE_OPTIONS)
	list(REMOVE_ITEM sdl_compile_options "/Gs1048576")
	set_target_properties(SDL2-static PROPERTIES COMPILE_OPTIONS "${sdl_compile_options}")
endif()

add_library(Vulkan-Headers INTERFACE)
add_library(Vulkan::Headers ALIAS Vulkan-Headers)
target_include_directories(Vulkan-Headers INTERFACE
	"${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include"
)

add_library(imgui STATIC
	imgui/imgui.cpp
	imgui/imgui_draw.cpp
	imgui/imgui_tables.cpp
	imgui/imgui_widgets.cpp
	imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/imgui"
	"${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends"
)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_VULKAN_NO_PROTOTYPES)
target_link_libraries(imgui PUBLIC Vulkan::Headers)
if(MSVC)
	target_compile_options(imgui PRIVATE /W0)
else()
	target_compile_options(imgui PRIVATE -w)
endif()

set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
set(SPIRV-Headers_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Headers" CACHE PATH "" FORCE)
add_subdirectory(SPIRV-Tools)

add_library(spirv-tools ALIAS SPIRV-Tools-static)
add_library(spirv-tools-opt ALIAS SPIRV-Tools-opt)

add_subdirectory(xxHash/build/cmake)

# AMD FidelityFX SDK 1.1.4 sources (FSR 3 upscaling, and frame interpolation on Linux) for Linux and Windows builds.
if (LINUX OR WIN32)
	add_subdirectory(fidelityfx)
endif()

set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_MODULE OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt EXCLUDE_FROM_ALL)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
add_subdirectory(nlohmann_json EXCLUDE_FROM_ALL)

set(SPDLOG_BUILD_ALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_WARNINGS OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog EXCLUDE_FROM_ALL)

if (NOT TARGET FFmpeg::ffmpeg)
	# On macOS the target arch is driven by CMAKE_OSX_ARCHITECTURES, not the host
	# CMAKE_SYSTEM_PROCESSOR (which reports arm64 even for an x86_64/Rosetta build).
	if(APPLE AND CMAKE_OSX_ARCHITECTURES)
		if(CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64)$")
			set(ARCHITECTURE x86_64)
		elseif(CMAKE_OSX_ARCHITECTURES MATCHES "^(arm64)$")
			set(ARCHITECTURE arm64)
		else()
			set(ARCHITECTURE x86_64 arm64) # universal
		endif()
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
		set(ARCHITECTURE arm64)
	else()
		set(ARCHITECTURE x86_64)
	endif()
	add_subdirectory(ffmpeg-core)
	add_library(FFmpeg::ffmpeg ALIAS ffmpeg)
endif()

file(GLOB LIBATRAC9_SOURCES LibAtrac9/C/src/*.c)
# Its decinit.c from overrides/: the shared tables initialized once (the submodule stays at an upstream commit).
list(FILTER LIBATRAC9_SOURCES EXCLUDE REGEX "/decinit\\.c$")
add_library(LibAtrac9 STATIC ${LIBATRAC9_SOURCES} overrides/LibAtrac9/decinit.c)
target_include_directories(LibAtrac9 PUBLIC LibAtrac9/C/src)
find_package(Threads REQUIRED)
target_link_libraries(LibAtrac9 PRIVATE Threads::Threads)
if(MSVC)
	target_compile_options(LibAtrac9 PRIVATE /W0)
endif()

set(TRACY_STATIC ON CACHE BOOL "" FORCE)
set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
set(TRACY_DELAYED_INIT ON CACHE BOOL "" FORCE)
set(TRACY_MANUAL_LIFETIME ON CACHE BOOL "" FORCE)
add_subdirectory(tracy EXCLUDE_FROM_ALL)
if (TARGET TracyClient)
	if (MSVC)
		target_compile_options(TracyClient PRIVATE /WX-)
	endif()
	if (CLANG)
		target_compile_options(TracyClient PRIVATE
			-Wno-error
			-Wno-braced-scalar-init
			-Wno-format
			-Wno-unused-variable
		)
	endif()
endif()

if (CLANG AND MSVC)
add_subdirectory(winpthread)
endif()
