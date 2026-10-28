cmake_minimum_required(VERSION 3.20)
project(injector CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # static CRT: nothing else to install

# Fetched by CI (see .github/workflows/injector.yml): Vulkan-Headers and Dear ImGui at fixed tags.
set(VULKAN_HEADERS_DIR "${CMAKE_SOURCE_DIR}/third_party/Vulkan-Headers" CACHE PATH "")
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui" CACHE PATH "")

# Phase 2 (upscalers), each built in when its public SDK is there (CI fetches them, see the workflow):
#   Streamline (NVIDIA, DLSS): third_party/streamline/include/sl.h
#   XeSS (Intel): third_party/xess/inc/xess/xess_vk.h
#   nlohmann/json (FSR 4's manifest reader): third_party/json/nlohmann/json.hpp
set(STREAMLINE_DIR "${CMAKE_SOURCE_DIR}/third_party/streamline" CACHE PATH "")
set(XESS_DIR "${CMAKE_SOURCE_DIR}/third_party/xess" CACHE PATH "")
set(JSON_DIR "${CMAKE_SOURCE_DIR}/third_party/json" CACHE PATH "")
option(INJ_REQUIRE_UPSCALERS "Fail when an upscaler's SDK is missing (CI)" OFF)
if(INJ_REQUIRE_UPSCALERS)
  set(INJ_MISSING FATAL_ERROR)
else()
  set(INJ_MISSING WARNING)
endif()

add_library(injector_vulkan SHARED
  src/proxy.cpp src/analyzer.cpp src/framegen.cpp src/overlay.cpp src/vulkan-1.def
  src/camera.cpp src/upscale.cpp src/up_dlss.cpp src/up_fsr3.cpp src/up_fsr4.cpp src/up_xess.cpp
  ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
set_target_properties(injector_vulkan PROPERTIES OUTPUT_NAME "vulkan-1")
target_include_directories(injector_vulkan PRIVATE src ${VULKAN_HEADERS_DIR}/include ${IMGUI_DIR} ${IMGUI_DIR}/backends
  ${CMAKE_SOURCE_DIR}/third_party/ffx_api/include)
target_compile_definitions(injector_vulkan PRIVATE VK_NO_PROTOTYPES IMGUI_IMPL_VULKAN_NO_PROTOTYPES VK_USE_PLATFORM_WIN32_KHR
  WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
target_compile_options(injector_vulkan PRIVATE /W3 /EHsc /utf-8)

add_executable(InjectorSetup WIN32 launcher/launcher.cpp)
target_compile_definitions(InjectorSetup PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
target_compile_options(InjectorSetup PRIVATE /W3 /EHsc /utf-8)
target_link_libraries(InjectorSetup PRIVATE comdlg32 shell32)

if(EXISTS "${STREAMLINE_DIR}/include/sl.h")
  target_compile_definitions(injector_vulkan PRIVATE INJ_STREAMLINE)
  target_include_directories(injector_vulkan PRIVATE "${STREAMLINE_DIR}/include")
  message(STATUS "upscalers: DLSS (Streamline) built in")
else()
  message(${INJ_MISSING} "upscalers: no Streamline SDK in ${STREAMLINE_DIR}: DLSS left out")
endif()
if(EXISTS "${XESS_DIR}/inc/xess/xess_vk.h")
  target_compile_definitions(injector_vulkan PRIVATE INJ_XESS)
  target_include_directories(injector_vulkan PRIVATE "${XESS_DIR}/inc")
  message(STATUS "upscalers: XeSS built in")
else()
  message(${INJ_MISSING} "upscalers: no XeSS SDK in ${XESS_DIR}: XeSS left out")
endif()
if(EXISTS "${JSON_DIR}/nlohmann/json.hpp")
  target_compile_definitions(injector_vulkan PRIVATE INJ_FSR4)
  target_include_directories(injector_vulkan PRIVATE "${JSON_DIR}")
  message(STATUS "upscalers: FSR 4 built in")
else()
  message(${INJ_MISSING} "upscalers: no nlohmann/json in ${JSON_DIR}: FSR 4 left out")
endif()
