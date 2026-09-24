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

        if (NOT DEFINED ROCM_DEVICE_LIB_PATH
            AND NOT "$ENV{ROCM_DEVICE_LIB_PATH}" STREQUAL "")
            set (ROCM_DEVICE_LIB_PATH "$ENV{ROCM_DEVICE_LIB_PATH}" CACHE PATH
                 "ROCm device bitcode library directory")
        endif ()
        find_path (ROCM_DEVICE_LIB_PATH NAMES ocml.bc
            HINTS "${HIP_PACKAGE_PREFIX_DIR}/lib/llvm/amdgcn/bitcode"
                  "${HIP_PACKAGE_PREFIX_DIR}/llvm/amdgcn/bitcode"
                  "${HIP_PACKAGE_PREFIX_DIR}/amdgcn/bitcode"
                  "${HIP_PACKAGE_PREFIX_DIR}/lib/amdgcn/bitcode"
            NO_DEFAULT_PATH)
        foreach (_hart_lib hip ocml ockl)
            if (NOT EXISTS "${ROCM_DEVICE_LIB_PATH}/${_hart_lib}.bc")
                message (FATAL_ERROR
                    "Missing ROCm device library ${_hart_lib}.bc. "
                    "Set ROCM_DEVICE_LIB_PATH to a complete, LLVM-compatible "
                    "ROCm device bitcode installation.")
            endif ()
        endforeach ()
        foreach (_hart_arch IN LISTS HART_TARGET_ARCHITECTURES)
            string (REPLACE "gfx" "" _hart_isa "${_hart_arch}")
            if (NOT EXISTS "${ROCM_DEVICE_LIB_PATH}/oclc_isa_version_${_hart_isa}.bc")
                message (FATAL_ERROR
                    "Missing ROCm device library for ${_hart_arch}: "
                    "${ROCM_DEVICE_LIB_PATH}/oclc_isa_version_${_hart_isa}.bc")
            endif ()
        endforeach ()
        file (GLOB ROCM_DEVICE_LIBRARIES CONFIGURE_DEPENDS
              "${ROCM_DEVICE_LIB_PATH}/*.bc")

        if (USE_LLVM_BITCODE)
            if (NOT USE_FAST_MATH)
                message (FATAL_ERROR
                    "Compiling HART shadeops currently requires USE_FAST_MATH=ON, "
                    "as does the shared CUDA shadeops implementation.")
            endif ()
            find_program (LLVM_BC_GENERATOR NAMES clang++
                HINTS "${LLVM_DIRECTORY}/bin" "${LLVM_DIRECTORY}/tools/llvm"
                NO_DEFAULT_PATH REQUIRED)
            find_program (LLVM_LINK_TOOL NAMES llvm-link
                HINTS "${LLVM_DIRECTORY}/bin" "${LLVM_DIRECTORY}/tools/llvm"
                NO_DEFAULT_PATH REQUIRED)
            find_program (LLVM_OPT_TOOL NAMES opt
                HINTS "${LLVM_DIRECTORY}/bin" "${LLVM_DIRECTORY}/tools/llvm"
                NO_DEFAULT_PATH REQUIRED)
            foreach (_hart_tool LLVM_BC_GENERATOR LLVM_LINK_TOOL LLVM_OPT_TOOL)
                execute_process (COMMAND "${${_hart_tool}}" --version
                    RESULT_VARIABLE _hart_tool_result
                    OUTPUT_VARIABLE _hart_tool_output
                    ERROR_VARIABLE _hart_tool_error)
                if (NOT _hart_tool_result STREQUAL "0"
                    OR NOT _hart_tool_output MATCHES "(clang|LLVM) version ([0-9]+\\.[0-9]+\\.[0-9]+)")
                    message (FATAL_ERROR
                        "Cannot query ${_hart_tool}: ${${_hart_tool}}\n"
                        "${_hart_tool_output}\n${_hart_tool_error}")
                endif ()
                if (NOT CMAKE_MATCH_2 VERSION_EQUAL LLVM_VERSION)
                    message (FATAL_ERROR
                        "${_hart_tool} must match OSL LLVM ${LLVM_VERSION}, "
                        "but ${${_hart_tool}} reports ${CMAKE_MATCH_2}. "
                        "Clear stale LLVM tool cache entries when changing LLVM.")
                endif ()
            endforeach ()
            message (STATUS "HART bitcode compiler: ${LLVM_BC_GENERATOR} -x hip")
        endif ()

        message (STATUS "HART package: ${amd.hart_DIR}")
        message (STATUS "ROCm HIP ${hip_VERSION}: ${hip_DIR}")
        message (STATUS "ROCm device libraries: ${ROCM_DEVICE_LIB_PATH}")
        message (STATUS "HART GPU architectures: ${HART_TARGET_ARCHITECTURES}")
        message (STATUS "OSL LLVM ${LLVM_VERSION}: ${LLVM_DIRECTORY}")
        message (STATUS "ROCm clang ${ROCM_CLANG_VERSION}: ${ROCM_CLANG_EXECUTABLE}")
        message (STATUS
            "HART build infrastructure enabled; runtime shader execution "
            "is not implemented yet.")
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


# Compile one HIP device translation unit to raw AMDGCN LLVM bitcode.
function (MAKE_HART_BITCODE src suffix arch generated_bc extra_clang_args include_dirs)
    if (NOT OSL_USE_HART OR NOT USE_LLVM_BITCODE OR NOT LLVM_BC_GENERATOR)
        message (FATAL_ERROR "MAKE_HART_BITCODE requires OSL_USE_HART and USE_LLVM_BITCODE")
    endif ()
    if (NOT arch IN_LIST HART_TARGET_ARCHITECTURES)
        message (FATAL_ERROR "HART architecture '${arch}' was not configured")
    endif ()
    get_filename_component (src "${src}" ABSOLUTE)
    get_filename_component (src_we "${src}" NAME_WE)
    set (output_dir "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}")
    set (bc "${output_dir}/${src_we}${suffix}.bc")
    set (${generated_bc} "${bc}" PARENT_SCOPE)

    get_property (definitions DIRECTORY PROPERTY COMPILE_DEFINITIONS)
    list (REMOVE_ITEM definitions OSL_LLVM_CUDA_BITCODE)
    list (TRANSFORM definitions PREPEND "-D")
    list (TRANSFORM include_dirs PREPEND "-I")
    set (math_flags -fno-math-errno)
    if (USE_FAST_MATH)
        # Keep Inf/NaN checks. ROCm also uses -ffast-math to select
        # finite-only device libraries, regardless of individual overrides.
        list (APPEND math_flags
            -fapprox-func -fassociative-math -freciprocal-math
            -fno-signed-zeros -fno-trapping-math -fno-rounding-math
            -ffp-contract=fast)
    endif ()
    set (depfile_args)
    set (dependency_flags)
    if (CMAKE_GENERATOR MATCHES "Ninja" OR CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        set (depfile_args DEPFILE "${bc}.d")
        # The HIP driver drops -MMD/-MF in device-only mode.
        string (REPLACE "$" "$$" depfile_target "${bc}")
        string (REPLACE "#" "\\#" depfile_target "${depfile_target}")
        string (REPLACE " " "\\ " depfile_target "${depfile_target}")
        set (dependency_flags -Xclang -dependency-file -Xclang "${bc}.d"
                              -Xclang -MT -Xclang "${depfile_target}")
    endif ()

    add_custom_command (OUTPUT "${bc}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
        COMMAND "${LLVM_BC_GENERATOR}"
            -x hip --offload-device-only --no-gpu-bundle-output
            "--offload-arch=${arch}"
            "--hip-path=${HIP_PACKAGE_PREFIX_DIR}"
            "--rocm-device-lib-path=${ROCM_DEVICE_LIB_PATH}"
            -fgpu-rdc "-std=c++${CMAKE_CXX_STANDARD}"
            ${LLVM_COMPILE_FLAGS} ${definitions} ${include_dirs}
            -DOSL_COMPILING_TO_BITCODE=1 -DNDEBUG -DOIIO_NO_SSE
            -Wno-ignored-attributes -Wno-unknown-attributes
            ${math_flags} -O3 ${extra_clang_args}
            ${dependency_flags} -emit-llvm -c "${src}" -o "${bc}"
        DEPENDS "${src}" "${LLVM_BC_GENERATOR}" ${exec_headers}
                ${PROJECT_PUBLIC_HEADERS} ${ROCM_DEVICE_LIBRARIES}
        ${depfile_args}
        VERBATIM)
endfunction ()


# Keep each architecture's modules separate; packaging them is a later stage.
function (HART_SHADEOPS_COMPILE prefix output_bc input_srcs headers include_dirs)
    if (NOT input_srcs)
        message (FATAL_ERROR "HART_SHADEOPS_COMPILE requires source files")
    endif ()
    list (APPEND exec_headers ${headers})
    set (architectures ${HART_TARGET_ARCHITECTURES})
    list (REMOVE_DUPLICATES architectures)
    set (all_bitcode)
    foreach (arch IN LISTS architectures)
        set (source_bitcode)
        foreach (src IN LISTS input_srcs)
            MAKE_HART_BITCODE ("${src}" "_${prefix}_${arch}" "${arch}"
                              bc "" "${include_dirs}")
            if (bc IN_LIST source_bitcode)
                message (FATAL_ERROR
                    "HART shadeops sources must have unique basenames: ${src}")
            endif ()
            list (APPEND source_bitcode "${bc}")
        endforeach ()
        set (unoptimized "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/${prefix}_unoptimized.bc")
        set (linked "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/${prefix}.bc")
        add_custom_command (OUTPUT "${linked}"
            BYPRODUCTS "${unoptimized}"
            COMMAND "${LLVM_LINK_TOOL}" ${source_bitcode} -o "${unoptimized}"
            COMMAND "${LLVM_OPT_TOOL}" "-passes=default<O3>,verify"
                "${unoptimized}" -o "${linked}"
            DEPENDS ${source_bitcode} "${LLVM_LINK_TOOL}" "${LLVM_OPT_TOOL}"
            VERBATIM)
        list (APPEND all_bitcode "${linked}")
    endforeach ()
    set (${output_bc} "${all_bitcode}" PARENT_SCOPE)
endfunction ()
