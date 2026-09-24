# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set (exec_headers "${CMAKE_CURRENT_SOURCE_DIR}/hashes.h")
CUDA_SHADEOPS_COMPILE ("cuda_bitcode_link" linked_bc linked_ptx
    "${CMAKE_CURRENT_SOURCE_DIR}/first.cu;${CMAKE_CURRENT_SOURCE_DIR}/second.cu"
    "${exec_headers}")

add_custom_target (cuda_bitcode_link DEPENDS "${linked_bc}" "${linked_ptx}")
add_test (NAME cuda-bitcode-link
    COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
        --config "$<CONFIG>" --target cuda_bitcode_link)
