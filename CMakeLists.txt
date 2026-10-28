cmake_minimum_required(VERSION 3.20)
project(injector CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # static CRT: nothing else to install

# Fetched by CI (see .github/workflows/injector.yml): Vulkan-Headers and Dear ImGui at fixed tags.
set(VULKAN_HEADERS_DIR "${CMAKE_SOURCE_DIR}/third_party/Vulkan-Headers" CACHE PATH "")
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui" CACHE PATH "")

add_library(injector_vulkan SHARED
  src/proxy.cpp src/analyzer.cpp src/framegen.cpp src/overlay.cpp src/vulkan-1.def
  ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
set_target_properties(injector_vulkan PROPERTIES OUTPUT_NAME "winevulkan")
target_include_directories(injector_vulkan PRIVATE src ${VULKAN_HEADERS_DIR}/include ${IMGUI_DIR} ${IMGUI_DIR}/backends
  ${CMAKE_SOURCE_DIR}/third_party/ffx_api/include)
target_compile_definitions(injector_vulkan PRIVATE VK_NO_PROTOTYPES IMGUI_IMPL_VULKAN_NO_PROTOTYPES VK_USE_PLATFORM_WIN32_KHR
  WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
target_compile_options(injector_vulkan PRIVATE /W3 /EHsc /utf-8)

add_executable(InjectorSetup WIN32 launcher/launcher.cpp)
target_compile_definitions(InjectorSetup PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
target_compile_options(InjectorSetup PRIVATE /W3 /EHsc /utf-8)
target_link_libraries(InjectorSetup PRIVATE comdlg32 shell32)
