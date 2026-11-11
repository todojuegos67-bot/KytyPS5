# SPDX-FileCopyrightText: Copyright 2026 IFreemz (shadps4_dlss bridge), bbport contributors, KytyPS5 contributors
# SPDX-License-Identifier: MIT
#
# libkyty_dlss.so: the Linux DLSS bridge (src/local/kyty-dlss-bridge.h). It links NVIDIA's DLSS SDK and is built on its
# own, not with the emulator (build-dlss-linux.sh):
#   cmake -S tools/local/dlss-bridge -B _Build/dlss-bridge -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DDLSS_SDK_ROOT=<github.com/NVIDIA/DLSS checkout> -DVULKAN_INCLUDE=3rdparty/Vulkan-Headers/include
#   ninja -C _Build/dlss-bridge
# Ship it next to kyty_emulator with lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.<version> (and libnvidia-ngx-dlssg.so)
# from the SDK.

cmake_minimum_required(VERSION 3.20)
project(kyty_dlss LANGUAGES CXX)

set(DLSS_SDK_ROOT "" CACHE PATH "NVIDIA DLSS SDK root (contains include/ and lib/)")
set(VULKAN_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/Vulkan-Headers/include" CACHE PATH
    "Vulkan-Headers include directory")

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(FATAL_ERROR "The DLSS bridge is for Linux x86_64 (Windows uses NVIDIA Streamline)")
endif()
if (NOT EXISTS "${DLSS_SDK_ROOT}/include/nvsdk_ngx_vk.h")
	message(FATAL_ERROR "Set DLSS_SDK_ROOT to the NVIDIA DLSS SDK (github.com/NVIDIA/DLSS)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(kyty_dlss SHARED kyty_dlss_bridge.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/local/kyty-dlss-bridge.h")
target_include_directories(kyty_dlss PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/local" "${DLSS_SDK_ROOT}/include"
                           "${VULKAN_INCLUDE}")
# libnvsdk_ngx.a loads the driver's libnvidia-ngx.so.1 and the DLSS snippet at run time.
set_target_properties(kyty_dlss PROPERTIES CXX_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
target_link_libraries(kyty_dlss PRIVATE "${DLSS_SDK_ROOT}/lib/Linux_x86_64/libnvsdk_ngx.a" ${CMAKE_DL_LIBS})
target_link_options(kyty_dlss PRIVATE -Wl,--no-undefined -static-libstdc++ -static-libgcc)
