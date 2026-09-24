# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.19)
project (HARTDiscoveryTest LANGUAGES NONE)

list (PREPEND CMAKE_MODULE_PATH "${OSL_SOURCE_DIR}/src/cmake")
include (hart_macros)

set (OSL_USE_HART ON)
set (HART_TARGET_ARCHITECTURES "gfx1201;gfx1100;gfx1151")
set (LLVM_TARGETS "X86 NVPTX AMDGPU")
set (LLVM_VERSION "20.1.8")
set (USE_FAST_MATH ON)
set (bitcode_test_cases bitcode-output-names bitcode-reordered-sources
                       bitcode-duplicate-basename)
foreach (var LLVM_DIRECTORY LLVM_LIBRARIES LLVM_BC_GENERATOR LLVM_AS_TOOL
             LLVM_LINK_TOOL LLVM_LLC_TOOL LLVM_OPT_TOOL CMAKE_CXX_COMPILER)
    set (${var} "osl-${var}")
endforeach ()
set (CMAKE_PREFIX_PATH "original-prefix")

set (HART_ROOT "${CMAKE_CURRENT_BINARY_DIR}/hart sdk")
set (ROCM_ROOT "${CMAKE_CURRENT_BINARY_DIR}/rocm sdk")
set (ENV{ROCM_DEVICE_LIB_PATH} "")
set (expected_device_lib_path "${ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
set (device_libraries hip ocml ockl oclc_isa_version_1201
                      oclc_isa_version_1100 oclc_isa_version_1151)
foreach (lib IN LISTS device_libraries)
    file (WRITE "${expected_device_lib_path}/${lib}.bc" "mock bitcode\n")
endforeach ()
file (WRITE "${ROCM_ROOT}/lib/cmake/hip/hip-config.cmake"
    "set(HIP_PACKAGE_PREFIX_DIR \"${ROCM_ROOT}\")\n"
    "set(HIP_PLATFORM amd)\n"
    "set(hip_VERSION 7.14.0)\n"
    "add_library(hip::host INTERFACE IMPORTED)\n")
file (WRITE "${HART_ROOT}/lib/cmake/amd.hart/amd.hart-config.cmake"
    "include(CMakeFindDependencyMacro)\n"
    "find_dependency(amd.shader_compiler CONFIG REQUIRED)\n"
    "add_library(amd::hart INTERFACE IMPORTED)\n")
file (WRITE "${HART_ROOT}/lib/cmake/amd.shader_compiler/amd.shader_compiler-config.cmake"
    "add_library(amd::shader_compiler INTERFACE IMPORTED)\n")
if (WIN32)
    set (clang "${ROCM_ROOT}/lib/llvm/bin/clang.bat")
    file (WRITE "${clang}" "@echo off\necho AMD clang version 23.0.0git\n")
else ()
    set (clang "${ROCM_ROOT}/lib/llvm/bin/clang")
    file (WRITE "${clang}" "#!/bin/sh\necho 'AMD clang version 23.0.0git'\n")
    file (CHMOD "${clang}" PERMISSIONS
          OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
          WORLD_READ WORLD_EXECUTE)
endif ()
# Use the script explicitly; find_program need not recognize script suffixes.
set (ROCM_CLANG_EXECUTABLE "${clang}")

if (TEST_CASE STREQUAL "disabled")
    set (OSL_USE_HART OFF)
    set (HART_TARGET_ARCHITECTURES "")
    set (LLVM_TARGETS "")
    set (CMAKE_DISABLE_FIND_PACKAGE_hip TRUE)
    set (CMAKE_DISABLE_FIND_PACKAGE_amd.hart TRUE)
    unset (ROCM_CLANG_EXECUTABLE)
elseif (TEST_CASE STREQUAL "empty-architectures")
    set (HART_TARGET_ARCHITECTURES "")
elseif (TEST_CASE STREQUAL "bad-architecture")
    set (HART_TARGET_ARCHITECTURES "gfx1201;gfx9999")
elseif (TEST_CASE STREQUAL "comma-architectures")
    set (HART_TARGET_ARCHITECTURES "gfx1201,gfx1100")
elseif (TEST_CASE STREQUAL "missing-amdgpu")
    set (LLVM_TARGETS "X86 NVPTX")
elseif (TEST_CASE STREQUAL "missing-hart")
    set (CMAKE_DISABLE_FIND_PACKAGE_amd.hart TRUE)
elseif (TEST_CASE STREQUAL "missing-hip")
    set (CMAKE_DISABLE_FIND_PACKAGE_hip TRUE)
elseif (TEST_CASE STREQUAL "missing-clang")
    set (ROCM_CLANG_EXECUTABLE "${ROCM_ROOT}/missing-clang")
elseif (TEST_CASE STREQUAL "no-clang")
    unset (ROCM_CLANG_EXECUTABLE)
    file (APPEND "${ROCM_ROOT}/lib/cmake/hip/hip-config.cmake"
          "set(HIP_PACKAGE_PREFIX_DIR \"${ROCM_ROOT}/without-compiler\")\n")
elseif (TEST_CASE STREQUAL "wrong-platform")
    file (APPEND "${ROCM_ROOT}/lib/cmake/hip/hip-config.cmake"
          "set(HIP_PLATFORM nvidia)\n")
elseif (TEST_CASE STREQUAL "missing-target")
    file (WRITE "${HART_ROOT}/lib/cmake/amd.hart/amd.hart-config.cmake" "")
elseif (TEST_CASE STREQUAL "subset")
    set (HART_TARGET_ARCHITECTURES "gfx1151;gfx1100")
    file (REMOVE "${expected_device_lib_path}/oclc_isa_version_1201.bc")
elseif (TEST_CASE STREQUAL "device-lib-override"
        OR TEST_CASE STREQUAL "device-lib-environment"
        OR TEST_CASE STREQUAL "device-lib-layout")
    if (TEST_CASE STREQUAL "device-lib-layout")
        foreach (lib IN LISTS device_libraries)
            file (REMOVE "${expected_device_lib_path}/${lib}.bc")
        endforeach ()
        set (expected_device_lib_path "${ROCM_ROOT}/amdgcn/bitcode")
    else ()
        set (expected_device_lib_path "${CMAKE_CURRENT_BINARY_DIR}/device libs override")
    endif ()
    foreach (lib IN LISTS device_libraries)
        file (WRITE "${expected_device_lib_path}/${lib}.bc" "mock bitcode\n")
    endforeach ()
    if (TEST_CASE STREQUAL "device-lib-environment")
        set (ENV{ROCM_DEVICE_LIB_PATH} "${expected_device_lib_path}")
    elseif (TEST_CASE STREQUAL "device-lib-override")
        set (ROCM_DEVICE_LIB_PATH "${expected_device_lib_path}" CACHE PATH "")
        set (ENV{ROCM_DEVICE_LIB_PATH} "${ROCM_ROOT}/ignored-environment-override")
    endif ()
elseif (TEST_CASE STREQUAL "missing-device-libs")
    set (ROCM_DEVICE_LIB_PATH "${ROCM_ROOT}/missing-device-libs" CACHE PATH "")
elseif (TEST_CASE STREQUAL "incomplete-device-libs")
    set (ROCM_DEVICE_LIB_PATH "${expected_device_lib_path}" CACHE PATH "")
    file (REMOVE "${expected_device_lib_path}/ockl.bc")
elseif (TEST_CASE STREQUAL "missing-isa-library")
    file (REMOVE "${expected_device_lib_path}/oclc_isa_version_1151.bc")
elseif (TEST_CASE STREQUAL "precise-math")
    set (USE_LLVM_BITCODE ON)
    set (USE_FAST_MATH OFF)
elseif (TEST_CASE STREQUAL "matching-tools" OR TEST_CASE STREQUAL "mismatched-tools"
        OR TEST_CASE IN_LIST bitcode_test_cases)
    set (USE_LLVM_BITCODE ON)
    set (LLVM_BC_GENERATOR "${clang}")
    set (LLVM_LINK_TOOL "${clang}")
    set (LLVM_OPT_TOOL "${clang}")
    if (NOT TEST_CASE STREQUAL "mismatched-tools")
        set (LLVM_VERSION "23.0.0git")
    endif ()
elseif (NOT TEST_CASE STREQUAL "enabled")
    message (FATAL_ERROR "Unknown test case: ${TEST_CASE}")
endif ()

foreach (var LLVM_DIRECTORY LLVM_LIBRARIES LLVM_BC_GENERATOR LLVM_AS_TOOL
             LLVM_LINK_TOOL LLVM_LLC_TOOL LLVM_OPT_TOOL CMAKE_CXX_COMPILER)
    set (expected_${var} "${${var}}")
endforeach ()
osl_find_hart ()

foreach (var LLVM_DIRECTORY LLVM_LIBRARIES LLVM_BC_GENERATOR LLVM_AS_TOOL
             LLVM_LINK_TOOL LLVM_LLC_TOOL LLVM_OPT_TOOL CMAKE_CXX_COMPILER)
    if (NOT "${${var}}" STREQUAL "${expected_${var}}")
        message (FATAL_ERROR "HART discovery changed ${var}")
    endif ()
endforeach ()
if (NOT CMAKE_PREFIX_PATH STREQUAL "original-prefix")
    message (FATAL_ERROR "HART discovery changed CMAKE_PREFIX_PATH")
endif ()
if (OSL_USE_HART)
    if (NOT TARGET amd::hart OR NOT TARGET hip::host OR NOT TARGET amd::shader_compiler
        OR NOT ROCM_CLANG_EXECUTABLE STREQUAL clang
        OR NOT ROCM_CLANG_VERSION STREQUAL "23.0.0"
        OR NOT ROCM_DEVICE_LIB_PATH STREQUAL expected_device_lib_path)
        message (FATAL_ERROR "HART/ROCm discovery did not find the expected SDKs")
    endif ()
elseif (TARGET amd::hart OR TARGET hip::host OR ROCM_CLANG_EXECUTABLE
        OR ROCM_DEVICE_LIB_PATH)
    message (FATAL_ERROR "Disabled HART discovery searched for SDKs")
endif ()

if (TEST_CASE IN_LIST bitcode_test_cases)
    set (CMAKE_CXX_STANDARD 17)
    set (sources "${CMAKE_CURRENT_BINARY_DIR}/sources/first.cpp"
                 "${CMAKE_CURRENT_BINARY_DIR}/sources/second.cpp")
    if (TEST_CASE STREQUAL "bitcode-reordered-sources")
        list (REVERSE sources)
    elseif (TEST_CASE STREQUAL "bitcode-duplicate-basename")
        list (APPEND sources "${CMAKE_CURRENT_BINARY_DIR}/other/first.cpp")
    endif ()
    foreach (src IN LISTS sources)
        file (WRITE "${src}" "")
    endforeach ()
    HART_SHADEOPS_COMPILE ("test_shadeops" linked_bitcode "${sources}" "" "")
    set (expected_linked_bitcode)
    foreach (arch IN LISTS HART_TARGET_ARCHITECTURES)
        foreach (name first second)
            set (bc "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/${name}_test_shadeops_${arch}.bc")
            get_source_file_property (generated "${bc}" GENERATED)
            if (NOT generated)
                message (FATAL_ERROR "Missing architecture-named bitcode rule: ${bc}")
            endif ()
        endforeach ()
        list (APPEND expected_linked_bitcode
            "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/test_shadeops.bc")
    endforeach ()
    if (NOT linked_bitcode STREQUAL expected_linked_bitcode)
        message (FATAL_ERROR "Unexpected linked bitcode paths: ${linked_bitcode}")
    endif ()
    add_custom_target (test_bitcode DEPENDS ${linked_bitcode})
endif ()
