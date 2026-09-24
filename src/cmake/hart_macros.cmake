# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Discover the HART runtime and ROCm tools without selecting HIP as OSL's
# host language or replacing any of OSL's LLVM_* tools/libraries.
macro (osl_find_hart)
    if (OSL_USE_HART)
        if ("${HART_TARGET_ARCHITECTURES}" STREQUAL "")
            message (FATAL_ERROR "HART_TARGET_ARCHITECTURES must not be empty")
        endif ()
        foreach (_hart_arch IN LISTS HART_TARGET_ARCHITECTURES)
            if (NOT _hart_arch MATCHES "^(gfx1201|gfx1100|gfx1151)$")
                message (FATAL_ERROR
                    "Unsupported HART architecture '${_hart_arch}'. "
                    "Use a semicolon-separated subset of gfx1201;gfx1100;gfx1151.")
            endif ()
        endforeach ()

        if (NOT "${LLVM_TARGETS}" MATCHES "(^|[ ;])AMDGPU([ ;]|$)")
            message (FATAL_ERROR
                "OSL_USE_HART requires AMDGPU in OSL's LLVM build. "
                "Select that build with LLVM_DIRECTORY/LLVM_ROOT; "
                "ROCm's LLVM does not replace it.")
        endif ()

        if (NOT DEFINED HART_ROOT)
            set (HART_ROOT "$ENV{HART_ROOT}")
        endif ()
        if (NOT DEFINED ROCM_ROOT)
            if (NOT "$ENV{ROCM_ROOT}" STREQUAL "")
                set (ROCM_ROOT "$ENV{ROCM_ROOT}")
            else ()
                set (ROCM_ROOT "$ENV{ROCM_PATH}")
            endif ()
        endif ()
        set (HART_ROOT "${HART_ROOT}" CACHE PATH "HART installation prefix")
        set (ROCM_ROOT "${ROCM_ROOT}" CACHE PATH "ROCm installation prefix")

        # Keep the SDK prefixes available to transitive package discovery,
        # but do not let ROCm influence subsequent searches for host tools.
        set (_hart_saved_prefix_path "${CMAKE_PREFIX_PATH}")
        list (PREPEND CMAKE_PREFIX_PATH "${HART_ROOT}" "${ROCM_ROOT}")
        find_package (hip CONFIG REQUIRED)
        find_package (amd.hart CONFIG REQUIRED)
        set (CMAKE_PREFIX_PATH "${_hart_saved_prefix_path}")
        unset (_hart_saved_prefix_path)

        if (NOT TARGET amd::hart OR NOT TARGET hip::host)
            message (FATAL_ERROR
                "HART/ROCm packages must provide amd::hart and hip::host")
        endif ()
        if (NOT HIP_PLATFORM STREQUAL "amd")
            message (FATAL_ERROR "OSL_USE_HART requires the AMD HIP platform")
        endif ()

        # Search only the selected HIP installation, never an unrelated
        # clang (particularly OSL's LLVM) on PATH.
        find_program (ROCM_CLANG_EXECUTABLE NAMES clang++ clang
            HINTS "${HIP_PACKAGE_PREFIX_DIR}/lib/llvm/bin"
                  "${HIP_PACKAGE_PREFIX_DIR}/llvm/bin"
                  "${HIP_PACKAGE_PREFIX_DIR}/bin"
            NO_DEFAULT_PATH)
        if (NOT ROCM_CLANG_EXECUTABLE)
            message (FATAL_ERROR
                "Cannot find ROCm clang in ${HIP_PACKAGE_PREFIX_DIR}. "
                "Select a ROCm development installation with ROCM_ROOT or hip_DIR.")
        endif ()
        mark_as_advanced (ROCM_CLANG_EXECUTABLE)
        execute_process (COMMAND "${ROCM_CLANG_EXECUTABLE}" --version
            RESULT_VARIABLE _hart_clang_result
            OUTPUT_VARIABLE _hart_clang_output
            ERROR_VARIABLE _hart_clang_error)
        if (NOT _hart_clang_result STREQUAL "0"
            OR NOT _hart_clang_output MATCHES "clang version ([0-9]+\\.[0-9]+\\.[0-9]+)")
            message (FATAL_ERROR
                "Cannot query ROCm clang: ${ROCM_CLANG_EXECUTABLE}\n"
                "${_hart_clang_output}\n${_hart_clang_error}")
        endif ()
        set (ROCM_CLANG_VERSION "${CMAKE_MATCH_1}")

        message (STATUS "HART package: ${amd.hart_DIR}")
        message (STATUS "ROCm HIP ${hip_VERSION}: ${hip_DIR}")
        message (STATUS "HART GPU architectures: ${HART_TARGET_ARCHITECTURES}")
        message (STATUS "OSL LLVM ${LLVM_VERSION}: ${LLVM_DIRECTORY}")
        message (STATUS "ROCm clang ${ROCM_CLANG_VERSION}: ${ROCM_CLANG_EXECUTABLE}")
        message (STATUS
            "HART dependency discovery enabled; device compilation and shader "
            "execution are not implemented yet.")
        message (WARNING
            "OSL LLVM and ROCm clang are separate toolchains. Discovery does not "
            "establish LLVM bitcode or C++ ABI compatibility. Do not link ROCm "
            "LLVM libraries into OSL or feed ROCm bitcode to OSL's LLVM without "
            "a separately validated compatibility path.")
    else ()
        message (STATUS "HART/ROCm support disabled")
    endif ()
endmacro ()


# Only future HART consumers opt into the runtime's include/link requirements.
function (osl_hart_target target)
    if (OSL_USE_HART)
        target_link_libraries (${target} PRIVATE amd::hart hip::host)
    endif ()
endfunction ()
