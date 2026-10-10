#
# Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


cmake_minimum_required(VERSION 3.11)

project(nvrhi VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CMakeDependentOption)

option(NVRHI_BUILD_SHARED "Build NVRHI as a shared library (DLL or .so)" OFF)
option(NVRHI_INSTALL "Generate install rules for NVRHI" ON)
cmake_dependent_option(NVRHI_INSTALL_EXPORTS "Install CMake exports" OFF "NVRHI_INSTALL" OFF)

option(NVRHI_WITH_VALIDATION "Build NVRHI the validation layer" ON)
option(NVRHI_WITH_VULKAN "Build the NVRHI Vulkan backend" ON)
option(NVRHI_WITH_RTXMU "Use RTXMU for acceleration structure management" OFF)
option(NVRHI_WITH_AFTERMATH "Include Aftermath support (requires NSight Aftermath SDK)" OFF)

cmake_dependent_option(NVRHI_WITH_NVAPI "Include NVAPI support (requires NVAPI SDK)" OFF "WIN32" OFF)
cmake_dependent_option(NVRHI_WITH_DX11 "Build the NVRHI D3D11 backend" ON "WIN32" OFF)
cmake_dependent_option(NVRHI_WITH_DX12 "Build the NVRHI D3D12 backend" ON "WIN32" OFF)
option(NVRHI_WITH_METAL3 "Build the NVRHI Metal 3 backend" OFF)
cmake_dependent_option(NVRHI_METAL3_WITH_TRACY "Enable Tracy GPU timestamp instrumentation in the Metal 3 backend" OFF "NVRHI_WITH_METAL3" OFF)
cmake_dependent_option(NVRHI_METAL3_BUILD_TESTS "Build native Metal command-list contract tests" OFF "NVRHI_WITH_METAL3" OFF)
cmake_dependent_option(NVRHI_VULKAN_BUILD_TESTS "Build Vulkan timer query contract tests" OFF "NVRHI_WITH_VULKAN" OFF)


if(NVRHI_WITH_VULKAN)
    option(NVRHI_FETCH_VULKAN_HEADERS "Fetch Vulkan-Headers when building the Vulkan backend and no external targets are provided" ON)
    set(NVRHI_VULKAN_HEADERS_GIT_REPOSITORY "https://github.com/KhronosGroup/Vulkan-Headers.git" CACHE STRING "Vulkan-Headers git repository used by NVRHI_FETCH_VULKAN_HEADERS")
    set(NVRHI_VULKAN_HEADERS_GIT_TAG "v1.4.352" CACHE STRING "Vulkan-Headers git tag or commit used by NVRHI_FETCH_VULKAN_HEADERS")
    set(NVRHI_VULKAN_HEADERS_FETCH_DIR "" CACHE STRING "Source directory for fetched Vulkan-Headers; empty uses CMake FetchContent default")
endif()

if(NVRHI_WITH_DX12)
    option(NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP "Use D3D12 native Opacity Micromaps from DXR 1.2" OFF)
    option(NVRHI_FETCH_DIRECTX_HEADERS "Fetch DirectX-Headers when building the D3D12 backend and no external targets are provided" ON)
    set(NVRHI_DIRECTX_HEADERS_GIT_REPOSITORY "https://github.com/microsoft/DirectX-Headers.git" CACHE STRING "DirectX-Headers git repository used by NVRHI_FETCH_DIRECTX_HEADERS")
    set(NVRHI_DIRECTX_HEADERS_GIT_TAG "v1.717.0-preview" CACHE STRING "DirectX-Headers git tag or commit used by NVRHI_FETCH_DIRECTX_HEADERS")
    set(NVRHI_DIRECTX_HEADERS_FETCH_DIR "" CACHE STRING "Source directory for fetched DirectX-Headers; empty uses CMake FetchContent default")
endif()

if(NVRHI_WITH_METAL3)
    if(NOT APPLE)
        message(FATAL_ERROR "NVRHI_WITH_METAL3 requires an Apple platform and a Metal-capable SDK.")
    endif()
    enable_language(OBJCXX)
endif()
if(NVRHI_WITH_RTXMU)
    set(NVRHI_RTXMU_GIT_REPOSITORY "https://github.com/NVIDIA-RTX/RTXMU.git" CACHE STRING "RTXMU git repository")
    set(NVRHI_RTXMU_GIT_TAG "v1.4" CACHE STRING "RTXMU git tag or commit")
    set(NVRHI_RTXMU_FETCH_DIR "" CACHE STRING "Source directory for fetched RTXMU; empty uses CMake FetchContent default")
endif()

if (NVRHI_BUILD_SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if (MSVC)
    string(REGEX REPLACE "/W[0-4]\\s*" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " /W4")
    if (NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        string(APPEND CMAKE_CXX_FLAGS " /MP")
    endif()
endif()


set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NVRHI_WITH_VULKAN)
    set(nvrhi_vulkan_headers_target "")

    if(TARGET Vulkan::Headers)
        set(nvrhi_vulkan_headers_target Vulkan::Headers)
    elseif(TARGET Vulkan-Headers)
        set(nvrhi_vulkan_headers_target Vulkan-Headers)
    endif()

    if((NOT nvrhi_vulkan_headers_target) AND NVRHI_FETCH_VULKAN_HEADERS)
        include(FetchContent)

        message(STATUS "Fetching Vulkan-Headers from ${NVRHI_VULKAN_HEADERS_GIT_REPOSITORY}, tag ${NVRHI_VULKAN_HEADERS_GIT_TAG}")

        if(NVRHI_VULKAN_HEADERS_FETCH_DIR)
            FetchContent_Declare(
                vulkan_headers
                GIT_REPOSITORY ${NVRHI_VULKAN_HEADERS_GIT_REPOSITORY}
                GIT_TAG ${NVRHI_VULKAN_HEADERS_GIT_TAG}
                SOURCE_DIR ${NVRHI_VULKAN_HEADERS_FETCH_DIR})
        else()
            FetchContent_Declare(
                vulkan_headers
                GIT_REPOSITORY ${NVRHI_VULKAN_HEADERS_GIT_REPOSITORY}
                GIT_TAG ${NVRHI_VULKAN_HEADERS_GIT_TAG})
        endif()

        FetchContent_MakeAvailable(vulkan_headers)

        if(TARGET Vulkan::Headers)
            set(nvrhi_vulkan_headers_target Vulkan::Headers)
        elseif(TARGET Vulkan-Headers)
            set(nvrhi_vulkan_headers_target Vulkan-Headers)
        endif()
    endif()

    if(NOT nvrhi_vulkan_headers_target)
        message(FATAL_ERROR "NVRHI Vulkan backend requires Vulkan-Headers CMake target (Vulkan::Headers or Vulkan-Headers), or enable NVRHI_FETCH_VULKAN_HEADERS.")
    endif()
endif()

if(NVRHI_WITH_DX12)
    set(nvrhi_directx_headers_target "")
    set(nvrhi_directx_guids_target "")

    if(TARGET Microsoft::DirectX-Headers)
        set(nvrhi_directx_headers_target Microsoft::DirectX-Headers)
    elseif(TARGET DirectX-Headers)
        set(nvrhi_directx_headers_target DirectX-Headers)
    endif()

    if(TARGET Microsoft::DirectX-Guids)
        set(nvrhi_directx_guids_target Microsoft::DirectX-Guids)
    elseif(TARGET DirectX-Guids)
        set(nvrhi_directx_guids_target DirectX-Guids)
    endif()

    if((NOT nvrhi_directx_headers_target OR NOT nvrhi_directx_guids_target) AND NVRHI_FETCH_DIRECTX_HEADERS)
        include(FetchContent)

        message(STATUS "Fetching DirectX-Headers from ${NVRHI_DIRECTX_HEADERS_GIT_REPOSITORY}, tag ${NVRHI_DIRECTX_HEADERS_GIT_TAG}")

        if(NVRHI_DIRECTX_HEADERS_FETCH_DIR)
            FetchContent_Declare(
                directx_headers
                GIT_REPOSITORY ${NVRHI_DIRECTX_HEADERS_GIT_REPOSITORY}
                GIT_TAG ${NVRHI_DIRECTX_HEADERS_GIT_TAG}
                SOURCE_DIR ${NVRHI_DIRECTX_HEADERS_FETCH_DIR})
        else()
            FetchContent_Declare(
                directx_headers
                GIT_REPOSITORY ${NVRHI_DIRECTX_HEADERS_GIT_REPOSITORY}
                GIT_TAG ${NVRHI_DIRECTX_HEADERS_GIT_TAG})
        endif()

        FetchContent_MakeAvailable(directx_headers)

        if(TARGET Microsoft::DirectX-Headers)
            set(nvrhi_directx_headers_target Microsoft::DirectX-Headers)
        endif()

        if(TARGET Microsoft::DirectX-Guids)
            set(nvrhi_directx_guids_target Microsoft::DirectX-Guids)
        endif()
    endif()

    if(NOT nvrhi_directx_headers_target OR NOT nvrhi_directx_guids_target)
        message(FATAL_ERROR "NVRHI D3D12 requires DirectX-Headers and DirectX-Guids CMake targets. Provide Microsoft::DirectX-Headers and Microsoft::DirectX-Guids, or enable NVRHI_FETCH_DIRECTX_HEADERS.")
    endif()
endif()

if (NVRHI_WITH_RTXMU)
    include(FetchContent)

    get_target_property(RTXMU_VULKAN_INCLUDE_DIR ${nvrhi_vulkan_headers_target} INTERFACE_INCLUDE_DIRECTORIES)

    option(RTXMU_WITH_VULKAN "" ${NVRHI_WITH_VULKAN})
    option(RTXMU_WITH_D3D12 "" ${NVRHI_WITH_DX12})

    message(STATUS "Fetching RTXMU from ${NVRHI_RTXMU_GIT_REPOSITORY}, tag ${NVRHI_RTXMU_GIT_TAG}")

    if(NVRHI_RTXMU_FETCH_DIR)
        FetchContent_Declare(
            rtxmu
            GIT_REPOSITORY ${NVRHI_RTXMU_GIT_REPOSITORY}
            GIT_TAG ${NVRHI_RTXMU_GIT_TAG}
            SOURCE_DIR ${NVRHI_RTXMU_FETCH_DIR})
    else()
        FetchContent_Declare(
            rtxmu
            GIT_REPOSITORY ${NVRHI_RTXMU_GIT_REPOSITORY}
            GIT_TAG ${NVRHI_RTXMU_GIT_TAG})
    endif()

    FetchContent_MakeAvailable(rtxmu)
endif()

if (NVRHI_WITH_NVAPI AND NOT TARGET nvapi)
    find_package(NVAPI REQUIRED)

    if (NVAPI_FOUND)
        add_library(nvapi STATIC IMPORTED GLOBAL)
        target_include_directories(nvapi INTERFACE "${NVAPI_INCLUDE_DIR}")
        set_property(TARGET nvapi PROPERTY IMPORTED_LOCATION "${NVAPI_LIBRARY}")
    endif()
endif()

if (NVRHI_WITH_AFTERMATH AND NOT TARGET aftermath)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchAftermath.cmake")
endif()

set(include_common
    include/nvrhi/nvrhi.h
    include/nvrhi/nvrhiHLSL.h
    include/nvrhi/utils.h
    include/nvrhi/common/containers.h
    include/nvrhi/common/misc.h
    include/nvrhi/common/resource.h
    include/nvrhi/common/aftermath.h)
set(src_common
    src/common/format-info.cpp
    src/common/misc.cpp
    src/common/state-tracking.cpp
    src/common/state-tracking.h
    src/common/utils.cpp
    src/common/aftermath.cpp)

if(MSVC)
    set(misc_common 
        tools/nvrhi.natvis)
else()
    set(misc_common "")
endif()

set(include_validation
    include/nvrhi/validation.h)
set(src_validation
    src/validation/validation-commandlist.cpp
    src/validation/validation-device.cpp
    src/validation/validation-backend.h)

set(include_d3d11
    include/nvrhi/d3d11.h)
set(src_d3d11
    src/common/dxgi-format.h
    src/common/dxgi-format.cpp
    src/d3d11/d3d11-buffer.cpp
    src/d3d11/d3d11-commandlist.cpp
    src/d3d11/d3d11-compute.cpp
    src/d3d11/d3d11-constants.cpp
    src/d3d11/d3d11-backend.h
    src/d3d11/d3d11-device.cpp
    src/d3d11/d3d11-graphics.cpp
    src/d3d11/d3d11-queries.cpp
    src/d3d11/d3d11-resource-bindings.cpp
    src/d3d11/d3d11-shader.cpp
    src/d3d11/d3d11-texture.cpp)

set(include_d3d12
    include/nvrhi/d3d12.h)
set(src_d3d12
    src/common/dxgi-format.h
    src/common/dxgi-format.cpp
    src/common/versioning.h
    src/d3d12/d3d12-buffer.cpp
    src/d3d12/d3d12-commandlist.cpp
    src/d3d12/d3d12-compute.cpp
    src/d3d12/d3d12-constants.cpp
    src/d3d12/d3d12-backend.h
    src/d3d12/d3d12-descriptor-heap.cpp
    src/d3d12/d3d12-device.cpp
    src/d3d12/d3d12-graphics.cpp
    src/d3d12/d3d12-meshlets.cpp
    src/d3d12/d3d12-queries.cpp
    src/d3d12/d3d12-raytracing.cpp
    src/d3d12/d3d12-resource-bindings.cpp
    src/d3d12/d3d12-shader.cpp
    src/d3d12/d3d12-state-tracking.cpp
    src/d3d12/d3d12-texture.cpp
    src/d3d12/d3d12-upload.cpp)

set(include_vk
    include/nvrhi/vulkan.h)
set(src_vk
    src/common/versioning.h
    src/vulkan/vulkan-allocator.cpp
    src/vulkan/vulkan-buffer.cpp
    src/vulkan/vulkan-commandlist.cpp
    src/vulkan/vulkan-compute.cpp
    src/vulkan/vulkan-constants.cpp
    src/vulkan/vulkan-device.cpp
    src/vulkan/vulkan-graphics.cpp
    src/vulkan/vulkan-meshlets.cpp
    src/vulkan/vulkan-queries.cpp
    src/vulkan/vulkan-queue.cpp
    src/vulkan/vulkan-raytracing.cpp
    src/vulkan/vulkan-resource-bindings.cpp
    src/vulkan/vulkan-shader.cpp
    src/vulkan/vulkan-staging-texture.cpp
    src/vulkan/vulkan-state-tracking.cpp
    src/vulkan/vulkan-texture.cpp
    src/vulkan/vulkan-upload.cpp
    src/vulkan/vulkan-backend.h)

set(include_metal3
    include/nvrhi/metal3.h)
set(src_metal3_impl
    src/metal3/metal3-buffer.cpp
    src/metal3/metal3-commandlist.cpp
    src/metal3/metal3-compute.cpp
    src/metal3/metal3-constants.cpp
    src/metal3/metal3-device.cpp
    src/metal3/metal3-graphics.cpp
    src/metal3/metal3-meshlets.cpp
    src/metal3/metal3-queries.cpp
    src/metal3/metal3-raytracing.cpp
    src/metal3/metal3-resource-bindings.cpp
    src/metal3/metal3-shader.cpp
    src/metal3/metal3-state-tracking.cpp
    src/metal3/metal3-texture.cpp)
set(src_metal3
    src/metal3/metal3-backend.h
    ${src_metal3_impl})

# NVRHI interface and common implementation functions

if (NVRHI_BUILD_SHARED)
    add_library(nvrhi SHARED
        ${include_common}
        ${src_common}
        ${misc_common})
else()
    add_library(nvrhi STATIC
        ${include_common}
        ${src_common}
        ${misc_common})
endif()


if (NVRHI_BUILD_SHARED)
    target_compile_definitions(nvrhi PRIVATE NVRHI_SHARED_LIBRARY_BUILD=1)
    if(NVRHI_WITH_VULKAN)
        target_compile_definitions(nvrhi PRIVATE VULKAN_HPP_STORAGE_SHARED VULKAN_HPP_STORAGE_SHARED_EXPORT)
    endif()
    set_property(TARGET nvrhi PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if (NVRHI_WITH_VALIDATION)
    target_sources(nvrhi PRIVATE
        ${include_validation}
        ${src_validation})
endif()

target_include_directories(nvrhi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)

set_target_properties(nvrhi PROPERTIES FOLDER "NVRHI")

target_compile_definitions(nvrhi PRIVATE NVRHI_WITH_AFTERMATH=$<BOOL:${NVRHI_WITH_AFTERMATH}>)

# implementations

if (NVRHI_WITH_DX11)
    if (NVRHI_BUILD_SHARED)
        set(nvrhi_d3d11_target nvrhi)

        target_sources(${nvrhi_d3d11_target} PRIVATE
            ${include_d3d11}
            ${src_d3d11})
    else()
        set(nvrhi_d3d11_target nvrhi_d3d11)

        add_library(${nvrhi_d3d11_target} STATIC
            ${include_d3d11}
            ${src_d3d11})

        set_target_properties(${nvrhi_d3d11_target} PROPERTIES FOLDER "NVRHI")
        target_include_directories(${nvrhi_d3d11_target} PRIVATE include)
    endif()

    target_link_libraries(${nvrhi_d3d11_target} PUBLIC d3d11 dxguid)

    if (NVRHI_WITH_NVAPI)
        target_link_libraries(${nvrhi_d3d11_target} PUBLIC nvapi)
    endif()
    target_compile_definitions(${nvrhi_d3d11_target} PRIVATE NVRHI_D3D11_WITH_NVAPI=$<BOOL:${NVRHI_WITH_NVAPI}>)

    if (NVRHI_WITH_AFTERMATH)
        target_link_libraries(${nvrhi_d3d11_target} PUBLIC aftermath)
    endif()
    target_compile_definitions(${nvrhi_d3d11_target} PRIVATE NVRHI_WITH_AFTERMATH=$<BOOL:${NVRHI_WITH_AFTERMATH}>)
endif()

if (NVRHI_WITH_DX12)
    if (NVRHI_BUILD_SHARED)
        set(nvrhi_d3d12_target nvrhi)

        target_sources(${nvrhi_d3d12_target} PRIVATE
            ${include_d3d12}
            ${src_d3d12})
    else()
        set(nvrhi_d3d12_target nvrhi_d3d12)

        add_library(${nvrhi_d3d12_target} STATIC
            ${include_d3d12}
            ${src_d3d12})

        set_target_properties(${nvrhi_d3d12_target} PROPERTIES FOLDER "NVRHI")
        target_include_directories(${nvrhi_d3d12_target} PRIVATE include)
    endif()

    if (NVRHI_WITH_RTXMU)
        target_compile_definitions(${nvrhi_d3d12_target} PUBLIC NVRHI_WITH_RTXMU=1)
        target_link_libraries(${nvrhi_d3d12_target} PUBLIC rtxmu)
    endif()

    target_compile_definitions(${nvrhi_d3d12_target} PRIVATE NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP=$<BOOL:${NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP}>)

    target_link_libraries(${nvrhi_d3d12_target} PUBLIC ${nvrhi_directx_headers_target} ${nvrhi_directx_guids_target} d3d12)

    if (NVRHI_WITH_NVAPI)
        target_link_libraries(${nvrhi_d3d12_target} PUBLIC nvapi)
    endif()
    target_compile_definitions(${nvrhi_d3d12_target} PRIVATE NVRHI_D3D12_WITH_NVAPI=$<BOOL:${NVRHI_WITH_NVAPI}>)

    if (NVRHI_WITH_AFTERMATH)
        target_link_libraries(${nvrhi_d3d12_target} PUBLIC aftermath)
    endif()
    target_compile_definitions(${nvrhi_d3d12_target} PRIVATE NVRHI_WITH_AFTERMATH=$<BOOL:${NVRHI_WITH_AFTERMATH}>)
endif()

if (NVRHI_WITH_VULKAN)
    if (NVRHI_BUILD_SHARED)
        set(nvrhi_vulkan_target nvrhi)

        target_sources(${nvrhi_vulkan_target} PRIVATE
            ${include_vk}
            ${src_vk})
    else()
        set(nvrhi_vulkan_target nvrhi_vk)

        add_library(${nvrhi_vulkan_target} STATIC
            ${include_vk}
            ${src_vk})

        set_target_properties(${nvrhi_vulkan_target} PROPERTIES FOLDER "NVRHI")
        target_include_directories(${nvrhi_vulkan_target} PRIVATE include)
    endif()
    
    if(WIN32)
        target_compile_definitions(${nvrhi_vulkan_target} PUBLIC VK_USE_PLATFORM_WIN32_KHR)
        target_compile_definitions(${nvrhi_vulkan_target} PRIVATE NOMINMAX)
    endif()

    if (NVRHI_WITH_RTXMU)
        target_compile_definitions(${nvrhi_vulkan_target} PUBLIC NVRHI_WITH_RTXMU=1)
        target_link_libraries(${nvrhi_vulkan_target} PUBLIC rtxmu)
    endif()

    target_link_libraries(${nvrhi_vulkan_target} PRIVATE ${nvrhi_vulkan_headers_target})

    if (NVRHI_WITH_AFTERMATH)
        target_link_libraries(${nvrhi_vulkan_target} PUBLIC aftermath)
    endif()
    target_compile_definitions(${nvrhi_vulkan_target} PRIVATE NVRHI_WITH_AFTERMATH=$<BOOL:${NVRHI_WITH_AFTERMATH}>)

    if(NVRHI_VULKAN_BUILD_TESTS)
        enable_testing()
        set(NVRHI_VULKAN_TEST_LIBRARY "" CACHE FILEPATH "Vulkan library used by timer query tests")
        add_executable(nvrhi_vulkan_timer_query_tests tests/vulkan-timer-queries.cpp)
        target_compile_features(nvrhi_vulkan_timer_query_tests PRIVATE cxx_std_17)
        target_link_libraries(nvrhi_vulkan_timer_query_tests PRIVATE nvrhi ${nvrhi_vulkan_headers_target} ${CMAKE_DL_LIBS})
        if(NOT NVRHI_BUILD_SHARED)
            target_link_libraries(nvrhi_vulkan_timer_query_tests PRIVATE nvrhi_vk)
        endif()
        add_test(NAME nvrhi_vulkan_timer_queries COMMAND nvrhi_vulkan_timer_query_tests "${NVRHI_VULKAN_TEST_LIBRARY}")
        set_tests_properties(nvrhi_vulkan_timer_queries PROPERTIES TIMEOUT 30)
    endif()
endif()

if (NVRHI_WITH_METAL3)
    if (NVRHI_BUILD_SHARED)
        set(nvrhi_metal3_target nvrhi)

        target_sources(${nvrhi_metal3_target} PRIVATE
            ${include_metal3}
            ${src_metal3})
    else()
        set(nvrhi_metal3_target nvrhi_metal3)

        add_library(${nvrhi_metal3_target} STATIC
            ${include_metal3}
            ${src_metal3})

        set_target_properties(${nvrhi_metal3_target} PROPERTIES FOLDER "NVRHI")
        target_include_directories(${nvrhi_metal3_target} PRIVATE include)
    endif()

    find_library(NVRHI_METAL_FRAMEWORK Metal REQUIRED)
    find_library(NVRHI_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(NVRHI_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
    set(NVRHI_METAL_SHADER_CONVERTER_ROOT "" CACHE PATH "Metal Shader Converter SDK prefix")
    find_path(NVRHI_METAL_IRCONVERTER_RUNTIME_INCLUDE_DIR
        NAMES metal_irconverter_runtime/metal_irconverter_runtime.h
        HINTS "${NVRHI_METAL_SHADER_CONVERTER_ROOT}"
        PATHS /usr/local /opt/homebrew
        PATH_SUFFIXES include)
    if(NOT EXISTS "${NVRHI_METAL_IRCONVERTER_RUNTIME_INCLUDE_DIR}/metal_irconverter_runtime/metal_irconverter_runtime.h")
        message(FATAL_ERROR
            "The native Metal backend requires Apple's Metal Shader Converter runtime headers.\n"
            "Set NVRHI_METAL_SHADER_CONVERTER_ROOT to the SDK prefix containing include/, "
            "or set NVRHI_METAL_IRCONVERTER_RUNTIME_INCLUDE_DIR to the include directory "
            "containing metal_irconverter_runtime/metal_irconverter_runtime.h.\n"
            "Download the SDK from https://developer.apple.com/metal/shader-converter/ "
            "(Apple Account sign-in required). No SDK is downloaded automatically.")
    endif()
    find_path(NVRHI_METAL_IRCONVERTER_INCLUDE_DIR
        NAMES metal_irconverter/metal_irconverter.h
        HINTS "${NVRHI_METAL_SHADER_CONVERTER_ROOT}"
        PATHS /usr/local /opt/homebrew
        PATH_SUFFIXES include
        REQUIRED)
    find_library(NVRHI_METAL_IRCONVERTER_LIBRARY
        NAMES metalirconverter
        HINTS "${NVRHI_METAL_SHADER_CONVERTER_ROOT}"
        PATHS /usr/local /opt/homebrew
        PATH_SUFFIXES lib
        REQUIRED)
    set(NVRHI_METAL_DXC_ROOT "" CACHE PATH "DXC SDK prefix for native Metal shader reflection")
    find_path(NVRHI_METAL_DXC_INCLUDE_DIR dxcapi.h
        HINTS "${NVRHI_METAL_DXC_ROOT}" PATH_SUFFIXES include/dxc include REQUIRED)
    find_path(NVRHI_METAL_DXC_REFLECTION_INCLUDE_DIR d3d12shader.h
        HINTS "${NVRHI_METAL_DXC_ROOT}" PATH_SUFFIXES include/directx include REQUIRED)
    find_path(NVRHI_METAL_DXC_STUB_INCLUDE_DIR rpc.h
        HINTS "${NVRHI_METAL_DXC_ROOT}" PATH_SUFFIXES include/wsl/stubs REQUIRED)
    find_library(NVRHI_METAL_DXC_LIBRARY dxcompiler
        HINTS "${NVRHI_METAL_DXC_ROOT}" PATH_SUFFIXES lib REQUIRED)
    if(TARGET copy-dxcompiler)
        add_dependencies(${nvrhi_metal3_target} copy-dxcompiler)
    endif()
    message(STATUS "Metal Shader Converter runtime: ${NVRHI_METAL_IRCONVERTER_RUNTIME_INCLUDE_DIR}")
    message(STATUS "Metal Shader Converter library: ${NVRHI_METAL_IRCONVERTER_LIBRARY}")

    set_source_files_properties(${src_metal3_impl} PROPERTIES LANGUAGE OBJCXX)
    target_compile_features(${nvrhi_metal3_target} PRIVATE cxx_std_17)
    target_compile_options(${nvrhi_metal3_target} PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
    target_include_directories(${nvrhi_metal3_target}
        PRIVATE
            ${NVRHI_METAL_IRCONVERTER_RUNTIME_INCLUDE_DIR}
            ${NVRHI_METAL_IRCONVERTER_INCLUDE_DIR}
            ${NVRHI_METAL_DXC_INCLUDE_DIR}
            ${NVRHI_METAL_DXC_REFLECTION_INCLUDE_DIR}
            ${NVRHI_METAL_DXC_STUB_INCLUDE_DIR})
    target_link_libraries(${nvrhi_metal3_target}
        PUBLIC
            ${NVRHI_METAL_IRCONVERTER_LIBRARY}
            ${NVRHI_METAL_DXC_LIBRARY}
            ${NVRHI_METAL_FRAMEWORK}
            ${NVRHI_FOUNDATION_FRAMEWORK}
            ${NVRHI_QUARTZCORE_FRAMEWORK})

    if (NVRHI_METAL3_WITH_TRACY)
        if (NOT TARGET Tracy::TracyClient)
            message(FATAL_ERROR "NVRHI_METAL3_WITH_TRACY requires the Tracy::TracyClient target")
        endif()
        target_link_libraries(${nvrhi_metal3_target} PRIVATE Tracy::TracyClient)
        target_compile_definitions(${nvrhi_metal3_target} PRIVATE NVRHI_METAL3_WITH_TRACY=1)
    endif()

    if(NVRHI_METAL3_BUILD_TESTS)
        enable_testing()
        add_executable(nvrhi_metal3_commandlist_tests tests/metal3-commandlist.mm)
        target_compile_features(nvrhi_metal3_commandlist_tests PRIVATE cxx_std_17)
        target_compile_options(nvrhi_metal3_commandlist_tests PRIVATE -fobjc-arc)
        target_include_directories(nvrhi_metal3_commandlist_tests PRIVATE ${NVRHI_METAL_DXC_INCLUDE_DIR})
        target_link_libraries(nvrhi_metal3_commandlist_tests PRIVATE nvrhi)
        if(NOT NVRHI_BUILD_SHARED)
            target_link_libraries(nvrhi_metal3_commandlist_tests PRIVATE nvrhi_metal3)
        endif()
        add_test(NAME nvrhi_metal3_commandlist COMMAND nvrhi_metal3_commandlist_tests)
        set_tests_properties(nvrhi_metal3_commandlist PROPERTIES TIMEOUT 30 ENVIRONMENT "MTL_DEBUG_LAYER=1")
    endif()
endif()


if (NVRHI_INSTALL)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/nvrhi
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

    install(TARGETS nvrhi
        EXPORT "nvrhiTargets"
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

    if (NOT NVRHI_BUILD_SHARED)
        if (NVRHI_WITH_DX11)
            install(TARGETS ${nvrhi_d3d11_target} DESTINATION "lib" EXPORT "nvrhiTargets")
        endif()

        if (NVRHI_WITH_DX12)
            install(TARGETS ${nvrhi_d3d12_target} DESTINATION "lib" EXPORT "nvrhiTargets")
        endif()

        if (NVRHI_WITH_VULKAN)
            install(TARGETS ${nvrhi_vulkan_target} DESTINATION "lib" EXPORT "nvrhiTargets")
        endif()

        if (NVRHI_WITH_METAL3)
            install(TARGETS ${nvrhi_metal3_target} DESTINATION "lib" EXPORT "nvrhiTargets")
        endif()
    endif()

    if (NVRHI_INSTALL_EXPORTS)
        set(nvrhi_CONFIG_PATH "lib/cmake/nvrhi")

        include(CMakePackageConfigHelpers)

        configure_package_config_file(
            src/nvrhiConfig.cmake.in
            src/nvrhiConfig.cmake
            INSTALL_DESTINATION "${nvrhi_CONFIG_PATH}"
            NO_CHECK_REQUIRED_COMPONENTS_MACRO)

        write_basic_package_version_file(
            src/nvrhiConfigVersion.cmake 
            VERSION ${nvrhi_VERSION} 
            COMPATIBILITY ExactVersion)

        install(FILES
            "${nvrhi_BINARY_DIR}/src/nvrhiConfig.cmake"
            "${nvrhi_BINARY_DIR}/src/nvrhiConfigVersion.cmake"
            DESTINATION "${nvrhi_CONFIG_PATH}")

        install(EXPORT "nvrhiTargets" 
            FILE "nvrhiTargets.cmake"
            EXPORT_LINK_INTERFACE_LIBRARIES
            DESTINATION "${nvrhi_CONFIG_PATH}")
    endif()
endif()
