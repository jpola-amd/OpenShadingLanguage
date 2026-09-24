# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.19)

foreach (bc IN LISTS BITCODE_FILES)
    execute_process (
        COMMAND "${LLVM_OPT_TOOL}" -passes=verify -S "${bc}" -o -
        RESULT_VARIABLE result OUTPUT_VARIABLE ir ERROR_VARIABLE error)
    if (NOT result STREQUAL "0")
        message (FATAL_ERROR "Cannot verify ${bc}:\n${error}")
    endif ()
    if (NOT ir MATCHES "target triple = \"amdgcn-amd-amdhsa\""
        OR NOT ir MATCHES "define [^\n]*@__raygen__testshade\\("
        OR NOT ir MATCHES "@testshade_hart_params = external")
        message (FATAL_ERROR "${bc} does not implement the HART grid contract")
    endif ()
    message (STATUS "Verified HART grid module: ${bc}")
endforeach ()

foreach (bc IN LISTS CALLABLE_FILES)
    execute_process (
        COMMAND "${LLVM_OPT_TOOL}" -passes=verify -S "${bc}" -o -
        RESULT_VARIABLE result OUTPUT_VARIABLE ir ERROR_VARIABLE error)
    if (NOT result STREQUAL "0")
        message (FATAL_ERROR "Cannot verify ${bc}:\n${error}")
    endif ()
    if (NOT ir MATCHES "target triple = \"amdgcn-amd-amdhsa\""
        OR NOT ir MATCHES "define [^\n]*@__direct_callable__testshade_init\\("
        OR NOT ir MATCHES "define [^\n]*@__direct_callable__testshade_entry\\("
        OR NOT ir MATCHES "define [^\n]*@osl_sin_ff\\("
        OR NOT ir MATCHES "call [^\n]*@osl_sin_ff\\(")
        message (FATAL_ERROR "${bc} must define both callables and call the linked OSL shadeop")
    endif ()
    message (STATUS "Verified HART callable/shadeops module: ${bc}")
endforeach ()
