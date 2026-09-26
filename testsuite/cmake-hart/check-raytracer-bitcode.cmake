# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.19)

execute_process (
    COMMAND "${LLVM_OPT_TOOL}" -passes=verify -S "${BITCODE_FILE}" -o -
    RESULT_VARIABLE result OUTPUT_VARIABLE ir ERROR_VARIABLE error)
if (NOT result STREQUAL "0")
    message (FATAL_ERROR "Cannot verify ${BITCODE_FILE}:\n${error}")
endif ()
if (NOT DEFINED RAYGEN)
    set (RAYGEN __raygen__osl_hart_probe)
endif ()
foreach (entry ${RAYGEN} __closesthit__osl_hart __miss__osl_hart)
    if (NOT ir MATCHES "define [^\n]*@${entry}\\(")
        message (FATAL_ERROR "${BITCODE_FILE} does not define ${entry}")
    endif ()
endforeach ()
if (NOT ir MATCHES "target triple = \"amdgcn-amd-amdhsa\""
    OR NOT ir MATCHES "\"target-cpu\"=\"${ARCH}\""
    OR NOT ir MATCHES "@osl_hart_render_params = external"
    OR NOT ir MATCHES "@llvm.compiler.used"
    OR NOT ir MATCHES "__hart")
    message (FATAL_ERROR "${BITCODE_FILE} does not implement the HART trace ABI")
endif ()
message (STATUS "Verified HART raytracer device module for ${ARCH}")
