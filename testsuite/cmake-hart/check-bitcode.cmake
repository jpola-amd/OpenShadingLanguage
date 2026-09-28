# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.19)

# Let LLVM evaluate calls into the actual device shadeops without a GPU.
set (fp_probes "")
set (fp_expectations)
foreach (case IN ITEMS
         "finite;1.000000e+00;0;0;1"
         "zero;0.000000e+00;0;0;1"
         "negative_zero;-0.000000e+00;0;0;1"
         "positive_inf;0x7FF0000000000000;0;1;0"
         "negative_inf;0xFFF0000000000000;0;1;0"
         "nan;0x7FF8000000000000;1;0;0")
    list (POP_FRONT case name value)
    foreach (op isnan isinf isfinite)
        list (POP_FRONT case expected)
        set (probe "hart_test_${op}_${name}")
        string (APPEND fp_probes
            "\ndefine i32 @${probe}() {\n"
            "  %result = call i32 @osl_${op}_if(float ${value})\n"
            "  ret i32 %result\n}\n")
        list (APPEND fp_expectations "${probe},${expected}")
    endforeach ()
endforeach ()
string (APPEND fp_probes [=[

define i32 @hart_test_safe_div() {
  %inf = call float @osl_safe_div_fff(float 1.000000e+00, float 0.000000e+00)
  %nan = call float @osl_safe_div_fff(float 0.000000e+00, float 0.000000e+00)
  %inf_ok = fcmp oeq float %inf, 0.000000e+00
  %nan_ok = fcmp oeq float %nan, 0.000000e+00
  %ok = and i1 %inf_ok, %nan_ok
  %result = zext i1 %ok to i32
  ret i32 %result
}
]=])
list (APPEND fp_expectations "hart_test_safe_div,1")

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
    set (probe_file "${BITCODE_DIR}/${arch}/shadeops_fp_test.ll")
    file (WRITE "${probe_file}" "${ir}\n${fp_probes}")
    execute_process (
        COMMAND "${LLVM_OPT_TOOL}" "-passes=default<O3>,verify" -S "${probe_file}" -o -
        RESULT_VARIABLE result OUTPUT_VARIABLE probe_ir ERROR_VARIABLE error)
    file (REMOVE "${probe_file}")
    if (NOT result STREQUAL "0")
        message (FATAL_ERROR "Cannot evaluate ${arch} floating-point checks:\n${error}")
    endif ()
    foreach (check IN LISTS fp_expectations)
        string (REPLACE "," ";" check "${check}")
        list (GET check 0 probe)
        list (GET check 1 expected)
        string (REGEX MATCH "define [^\n]+@${probe}\\(\\)[^{]*\\{[^}]*\\}"
                body "${probe_ir}")
        if (NOT body MATCHES "ret i32 ${expected}[\r\n]")
            message (FATAL_ERROR
                "${arch}: ${probe} should return ${expected}, got:\n${body}")
        endif ()
    endforeach ()
    message (STATUS "${arch}: verified ${definition_count} OSL device definitions and Inf/NaN checks")
endforeach ()
