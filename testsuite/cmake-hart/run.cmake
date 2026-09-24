# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.19)

function (check_discovery name expected_error)
    execute_process (
        COMMAND "${CMAKE_COMMAND}"
            -S "${OSL_SOURCE_DIR}/testsuite/cmake-hart"
            -B "${TEST_BINARY_DIR}/${name}"
            -U hip_DIR -U amd.hart_DIR -U amd.shader_compiler_DIR
            -U ROCM_CLANG_EXECUTABLE -U ROCM_DEVICE_LIB_PATH
            "-DOSL_SOURCE_DIR=${OSL_SOURCE_DIR}" "-DTEST_CASE=${name}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error)
    set (log "${output}\n${error}")
    file (WRITE "${TEST_BINARY_DIR}/${name}.log" "${log}")
    if (expected_error STREQUAL "")
        if (NOT result STREQUAL "0")
            message (FATAL_ERROR "${name} unexpectedly failed:\n${log}")
        endif ()
    elseif (result STREQUAL "0" OR NOT log MATCHES "${expected_error}")
        message (FATAL_ERROR
            "${name}: expected failure '${expected_error}', got:\n${log}")
    endif ()
    message (STATUS "HART discovery test passed: ${name}")
endfunction ()

check_discovery (disabled "")
check_discovery (enabled "")
check_discovery (subset "")
check_discovery (empty-architectures "HART_TARGET_ARCHITECTURES must not be empty")
check_discovery (bad-architecture "Unsupported HART architecture")
check_discovery (comma-architectures "Unsupported HART architecture")
check_discovery (missing-amdgpu "requires AMDGPU")
check_discovery (missing-hart "CMAKE_DISABLE_FIND_PACKAGE_amd.hart")
check_discovery (missing-hip "CMAKE_DISABLE_FIND_PACKAGE_hip")
check_discovery (missing-clang "Cannot query ROCm clang")
check_discovery (no-clang "Cannot find ROCm clang")
check_discovery (wrong-platform "requires the AMD HIP platform")
check_discovery (missing-target "must provide amd::hart and hip::host")
check_discovery (device-lib-override "")
check_discovery (device-lib-environment "")
check_discovery (device-lib-layout "")
check_discovery (missing-device-libs "Missing ROCm device library hip.bc")
check_discovery (incomplete-device-libs "Missing ROCm device library ockl.bc")
check_discovery (missing-isa-library "Missing ROCm device library for gfx1151")
check_discovery (precise-math "requires USE_FAST_MATH=ON")
check_discovery (matching-tools "")
check_discovery (mismatched-tools "LLVM_BC_GENERATOR must match OSL LLVM")
check_discovery (bitcode-output-names "")
check_discovery (bitcode-reordered-sources "")
check_discovery (bitcode-duplicate-basename "HART shadeops sources must have unique basenames")
