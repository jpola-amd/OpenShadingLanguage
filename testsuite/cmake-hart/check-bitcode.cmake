# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.19)

foreach (arch IN LISTS HART_TARGET_ARCHITECTURES)
    set (bc "${BITCODE_DIR}/${arch}/shadeops_hart.bc")
    execute_process (
        COMMAND "${LLVM_OPT_TOOL}" -passes=verify -S "${bc}" -o -
        RESULT_VARIABLE result OUTPUT_VARIABLE ir ERROR_VARIABLE error)
    if (NOT result STREQUAL "0")
        message (FATAL_ERROR "Cannot verify ${bc}:\n${error}")
    endif ()
    if (NOT ir MATCHES "target triple = \"amdgcn-amd-amdhsa\"")
        message (FATAL_ERROR "${bc} is not AMDGCN device bitcode")
    endif ()
    string (REGEX MATCHALL "\"target-cpu\"=\"[^\"]+\"" cpus "${ir}")
    list (REMOVE_DUPLICATES cpus)
    if (NOT cpus STREQUAL "\"target-cpu\"=\"${arch}\"")
        message (FATAL_ERROR "${bc} has incorrect or mixed architectures: ${cpus}")
    endif ()
    foreach (symbol osl_sin_ff osl_mul_mmm osl_noise_ff osl_blackbody_vf
                    osl_gabornoise_dfdf osl_simplexnoise_ff osl_pointcloud_get
                    osl_spline_fff osl_texture osl_allocate_closure_component)
        if (NOT ir MATCHES "define [^\n]+@${symbol}\\(")
            message (FATAL_ERROR "${bc} is missing device definition ${symbol}")
        endif ()
    endforeach ()
    string (REGEX MATCHALL "define [^\n]+@osl_[a-zA-Z0-9_]+\\(" definitions "${ir}")
    list (LENGTH definitions definition_count)
    if (definition_count LESS 400)
        message (FATAL_ERROR "${bc} contains only ${definition_count} shadeops")
    endif ()
    message (STATUS "${arch}: verified ${definition_count} OSL device definitions")
endforeach ()
