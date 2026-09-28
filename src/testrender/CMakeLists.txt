# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# The 'testrender' executable
set (testrender_srcs
     shading.cpp
     simpleraytracer.cpp
     scene.cpp
     bvh.cpp
     testrender.cpp)

find_package(Threads REQUIRED)

if (OSL_USE_OPTIX)
    list (APPEND testrender_srcs optixraytracer.cpp)
    set (testrender_cuda_srcs
        cuda/optix_raytracer.cu
        )

    set (testrender_rend_lib_srcs
        cuda/rend_lib.cu
        ../testshade/rs_simplerend.cpp
        )

    # We need to make sure that the PTX files are regenerated whenever these
    # files change.
    set (testrender_cuda_headers
        cuda/rend_lib.h
        background.h
        optics.h
        render_params.h
        raytracer.h
        sampling.h
        shading.h
        shading.cpp
        simpleraytracer.cpp
        )

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testrender_cuda_srcs})
        NVCC_COMPILE ( ${cudasrc} "${testrender_cuda_headers}" ptx_generated "" BSDL)
        list (APPEND ptx_list ${ptx_generated})
    endforeach ()

    # Compile the renderer-supplied shadeops (rend_lib) to LLVM bitcode and PTX
    add_compile_definitions (OSL_LLVM_CUDA_BITCODE)
    CUDA_SHADEOPS_COMPILE ( "rend_lib_testrender"
        rend_lib_bc
        rend_lib_ptx
        "${testrender_rend_lib_srcs}"
        "${testrender_cuda_headers}"
    )

    # Serialize the linked bitcode into a CPP file to be embedded in the current target binary
    set (rend_lib_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/rend_lib_cuda.bc.cpp")
    MAKE_EMBEDDED_CPP( "rend_lib_llvm_compiled_ops" ${rend_lib_bc_cuda_cpp} ${rend_lib_bc} )
    list (APPEND testrender_srcs ${rend_lib_bc_cuda_cpp})
    list (APPEND ptx_list ${rend_lib_ptx})

    add_custom_target (testrender_ptx ALL
        DEPENDS ${ptx_list} ${testrender_cuda_headers}
        SOURCES ${testrender_cuda_srcs} )

    # Install the PTX files in a fixed location so that they can be
    # loaded at run time
    install (FILES ${ptx_list}
             DESTINATION ${OSL_PTX_INSTALL_DIR})
endif()

if (CMAKE_COMPILER_IS_INTELCLANG)
    # To better match existing test results
    add_compile_options("-fp-model=precise")
    # Better performance is likely by not requiring a precise floating point
    # model, although with slightly different numerical results.
endif ()

add_executable (testrender ${testrender_srcs})
target_compile_definitions (testrender PRIVATE
    OSL_TESTRENDER_HART=$<AND:$<BOOL:${OSL_USE_HART}>,$<BOOL:${USE_LLVM_BITCODE}>>)

target_include_directories (testrender BEFORE PRIVATE ${OpenImageIO_INCLUDES})

target_link_libraries (testrender
    PRIVATE
        oslexec oslquery oslcomp BSDL
        pugixml::pugixml
        Threads::Threads)

if (OSL_USE_OPTIX)
    add_dependencies(testrender testrender_ptx)
endif ()

osl_optix_target (testrender)

install ( TARGETS testrender RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

if (OSL_USE_HART AND USE_LLVM_BITCODE)
    get_target_property (hart_path_includes amd::hart INTERFACE_INCLUDE_DIRECTORIES)
    list (APPEND hart_path_includes
          "${PROJECT_SOURCE_DIR}/src/include" "${PROJECT_BINARY_DIR}/include"
          ${IMATH_INCLUDES} ${OpenImageIO_INCLUDES})
    get_target_property (hart_bsdl_includes BSDL INTERFACE_INCLUDE_DIRECTORIES)
    list (APPEND hart_path_includes ${hart_bsdl_includes})
    list (REMOVE_DUPLICATES hart_path_includes)
    set (hart_path_declarations "")
    set (hart_path_entries "")
    set (hart_path_sources)
    set (exec_headers "${CMAKE_CURRENT_SOURCE_DIR}/hartpathparams.h")
    set (hart_path_architectures ${HART_TARGET_ARCHITECTURES})
    list (REMOVE_DUPLICATES hart_path_architectures)
    foreach (arch IN LISTS hart_path_architectures)
        MAKE_HART_BITCODE ("hart/pathtracer.hip" "_hart_${arch}" "${arch}"
                          path_bc "" "${hart_path_includes}")
        set (symbol "osl_hart_pathtracer_${arch}")
        set (embedded "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/pathtracer.bc.cpp")
        MAKE_EMBEDDED_CPP ("${symbol}" "${embedded}" "${path_bc}")
        list (APPEND hart_path_sources "${embedded}")
        string (APPEND hart_path_declarations
            "extern unsigned char ${symbol}_block[];\nextern int ${symbol}_size;\n")
        string (APPEND hart_path_entries
            "    { \"${arch}\", ${symbol}_block, &${symbol}_size },\n")
        if (BUILD_TESTING)
            add_test (NAME hart-pathtracer-codegen-${arch}
                COMMAND "${CMAKE_COMMAND}"
                    "-DLLVM_OPT_TOOL=${LLVM_OPT_TOOL}" "-DBITCODE_FILE=${path_bc}"
                    "-DARCH=${arch}" "-DRAYGEN=__raygen__osl_hart_path"
                    -P "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-raytracer-bitcode.cmake")
            add_test (NAME hart-background-codegen-${arch}
                COMMAND "${CMAKE_COMMAND}"
                    "-DLLVM_OPT_TOOL=${LLVM_OPT_TOOL}" "-DBITCODE_FILE=${path_bc}"
                    "-DARCH=${arch}" "-DRAYGEN=__raygen__osl_hart_background"
                    -P "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-raytracer-bitcode.cmake")
        endif ()
    endforeach ()
    file (WRITE "${CMAKE_CURRENT_BINARY_DIR}/hart_pathtracer_bitcode.h"
        "${hart_path_declarations}"
        "struct HartPathtracerModule { const char* arch; const unsigned char* data; const int* size; };\n"
        "static const HartPathtracerModule hart_pathtracer_modules[] = {\n"
        "${hart_path_entries}};\n")
    target_sources (testrender PRIVATE hartcontext.cpp hartraytracer.cpp
                    ../testshade/harttexture.cpp ${hart_path_sources})
    target_include_directories (testrender PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    osl_hart_target (testrender)
    osl_hart_runtime_target (testrender)
    if (BUILD_TESTING)
        add_executable (hart_lifecycle_test hart_lifecycle_test.cpp
                        hartcontext.cpp hartraytracer.cpp
                        shading.cpp simpleraytracer.cpp scene.cpp bvh.cpp
                        ../testshade/harttexture.cpp ${hart_path_sources})
        target_include_directories (hart_lifecycle_test BEFORE PRIVATE ${OpenImageIO_INCLUDES})
        target_include_directories (hart_lifecycle_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_link_libraries (hart_lifecycle_test PRIVATE oslexec oslquery oslcomp BSDL
                              pugixml::pugixml Threads::Threads)
        osl_hart_target (hart_lifecycle_test)
        osl_optix_target (hart_lifecycle_test)
        osl_hart_runtime_target (hart_lifecycle_test)
        set_target_properties (hart_lifecycle_test PROPERTIES FOLDER "Unit Tests")
        set (hart_material_declarations "")
        set (hart_material_entries "")
        set (hart_material_sources)
        foreach (arch IN LISTS hart_path_architectures)
            MAKE_HART_BITCODE ("hart/material_test.hip" "_hart_${arch}" "${arch}"
                              material_bc "" "${hart_path_includes}")
            set (symbol "osl_hart_material_test_${arch}")
            set (embedded "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/material_test.bc.cpp")
            MAKE_EMBEDDED_CPP ("${symbol}" "${embedded}" "${material_bc}")
            list (APPEND hart_material_sources "${embedded}")
            string (APPEND hart_material_declarations
                "extern unsigned char ${symbol}_block[];\nextern int ${symbol}_size;\n")
            string (APPEND hart_material_entries
                "    { \"${arch}\", ${symbol}_block, &${symbol}_size },\n")
            add_test (NAME hart-material-codegen-${arch}
                COMMAND "${CMAKE_COMMAND}"
                    "-DLLVM_OPT_TOOL=${LLVM_OPT_TOOL}" "-DBITCODE_FILE=${material_bc}"
                    "-DARCH=${arch}" "-DRAYGEN=__raygen__osl_hart_material_test"
                    -P "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-raytracer-bitcode.cmake")
        endforeach ()
        file (WRITE "${CMAKE_CURRENT_BINARY_DIR}/hart_material_bitcode.h"
            "${hart_material_declarations}"
            "struct HartMaterialModule { const char* arch; const unsigned char* data; const int* size; };\n"
            "static const HartMaterialModule hart_material_modules[] = {\n"
            "${hart_material_entries}};\n")
        add_executable (hart_material_test hart_material_test.cpp hartcontext.cpp
                        ${hart_material_sources})
        target_include_directories (hart_material_test BEFORE PRIVATE ${OpenImageIO_INCLUDES})
        target_include_directories (hart_material_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_link_libraries (hart_material_test PRIVATE oslexec BSDL)
        osl_hart_target (hart_material_test)
        osl_optix_target (hart_material_test)
        osl_hart_runtime_target (hart_material_test)
        set_target_properties (hart_material_test PROPERTIES FOLDER "Unit Tests")
        if ("$ENV{TESTSUITE_HART}" STREQUAL "1")
            add_test (NAME hart-material-components COMMAND hart_material_test)
            set_tests_properties (hart-material-components PROPERTIES
                                  LABELS "gpu;hart" RUN_SERIAL TRUE)
        endif ()
    endif ()
    if (BUILD_TESTING AND "$ENV{TESTSUITE_HART}" STREQUAL "1")
        foreach (mode split fused fused-local unoptimized)
            add_test (NAME hart-native-lifecycle-${mode}
                COMMAND hart_lifecycle_test
                    "${PROJECT_SOURCE_DIR}/src/shaders/stdosl.h" "${mode}")
            set_tests_properties (hart-native-lifecycle-${mode} PROPERTIES
                                  LABELS "gpu;hart" RUN_SERIAL TRUE TIMEOUT 900)
            add_test (NAME hart-pathtracer-${mode}
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-pathtracer.py"
                    "$<TARGET_FILE:testrender>" "$<TARGET_FILE:oslc>"
                    "${PROJECT_SOURCE_DIR}/src/shaders/stdosl.h" "${mode}")
            set_tests_properties (hart-pathtracer-${mode} PROPERTIES
                                  LABELS "gpu;hart" RUN_SERIAL TRUE)
            add_test (NAME hart-material-render-${mode}
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-pathtracer.py"
                    "$<TARGET_FILE:testrender>" "$<TARGET_FILE:oslc>"
                    "${PROJECT_SOURCE_DIR}/src/shaders/stdosl.h" "${mode}" --materials)
            set_tests_properties (hart-material-render-${mode} PROPERTIES
                                  LABELS "gpu;hart" RUN_SERIAL TRUE)
            add_test (NAME hart-lighting-render-${mode}
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-pathtracer.py"
                    "$<TARGET_FILE:testrender>" "$<TARGET_FILE:oslc>"
                    "${PROJECT_SOURCE_DIR}/src/shaders/stdosl.h" "${mode}" --lighting)
            set_tests_properties (hart-lighting-render-${mode} PROPERTIES
                                  LABELS "gpu;hart" RUN_SERIAL TRUE)
            add_test (NAME hart-volume-render-${mode}
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-pathtracer.py"
                    "$<TARGET_FILE:testrender>" "$<TARGET_FILE:oslc>"
                    "${PROJECT_SOURCE_DIR}/src/shaders/stdosl.h" "${mode}" --volumes)
            set_tests_properties (hart-volume-render-${mode} PROPERTIES
                                  LABELS "gpu;hart" RUN_SERIAL TRUE)
            add_test (NAME hart-texture-render-${mode}
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-pathtracer.py"
                    "$<TARGET_FILE:testrender>" "$<TARGET_FILE:oslc>"
                    "${PROJECT_SOURCE_DIR}/src/shaders/stdosl.h" "${mode}" --textures)
            set_tests_properties (hart-texture-render-${mode} PROPERTIES
                                  LABELS "gpu;hart" RUN_SERIAL TRUE)
        endforeach ()
    endif ()
endif ()

if (OSL_USE_HART AND USE_LLVM_BITCODE AND BUILD_TESTING)
    get_target_property (hart_include_dirs amd::hart INTERFACE_INCLUDE_DIRECTORIES)
    list (APPEND hart_include_dirs
          "${PROJECT_SOURCE_DIR}/src/include" "${PROJECT_BINARY_DIR}/include"
          ${IMATH_INCLUDES} ${OpenImageIO_INCLUDES})
    list (REMOVE_DUPLICATES hart_include_dirs)
    set (hart_trace_declarations "")
    set (hart_trace_entries "")
    set (hart_trace_sources)
    set (exec_headers "${CMAKE_CURRENT_SOURCE_DIR}/hartparams.h")
    set (hart_trace_architectures ${HART_TARGET_ARCHITECTURES})
    list (REMOVE_DUPLICATES hart_trace_architectures)
    foreach (arch IN LISTS hart_trace_architectures)
        MAKE_HART_BITCODE ("hart/raytracer.hip" "_hart_${arch}" "${arch}"
                          trace_bc "" "${hart_include_dirs}")
        set (symbol "osl_hart_raytracer_${arch}")
        set (embedded "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/raytracer.bc.cpp")
        MAKE_EMBEDDED_CPP ("${symbol}" "${embedded}" "${trace_bc}")
        list (APPEND hart_trace_sources "${embedded}")
        string (APPEND hart_trace_declarations
            "extern unsigned char ${symbol}_block[];\nextern int ${symbol}_size;\n")
        string (APPEND hart_trace_entries
            "    { \"${arch}\", ${symbol}_block, &${symbol}_size },\n")
        add_test (NAME hart-raytracer-codegen-${arch}
            COMMAND "${CMAKE_COMMAND}"
                "-DLLVM_OPT_TOOL=${LLVM_OPT_TOOL}" "-DBITCODE_FILE=${trace_bc}"
                "-DARCH=${arch}"
                -P "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-raytracer-bitcode.cmake")
    endforeach ()
    file (WRITE "${CMAKE_CURRENT_BINARY_DIR}/hart_raytracer_bitcode.h"
        "${hart_trace_declarations}"
        "struct HartRaytracerModule { const char* arch; const unsigned char* data; const int* size; };\n"
        "static const HartRaytracerModule hart_raytracer_modules[] = {\n"
        "${hart_trace_entries}};\n")
    add_executable (hart_trace_test hart_trace_test.cpp hartcontext.cpp bvh.cpp
                    ${hart_trace_sources})
    target_include_directories (hart_trace_test BEFORE PRIVATE ${OpenImageIO_INCLUDES})
    target_include_directories (hart_trace_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries (hart_trace_test PRIVATE OpenImageIO::OpenImageIO)
    osl_hart_target (hart_trace_test)
    osl_optix_target (hart_trace_test)
    # testshade deploys the selected runtime beside the shared test binaries.
    add_dependencies (hart_trace_test testshade)
    set_target_properties (hart_trace_test PROPERTIES FOLDER "Unit Tests")
    if ("$ENV{TESTSUITE_HART}" STREQUAL "1")
        add_test (NAME hart-raytracer-traversal COMMAND hart_trace_test)
        set_tests_properties (hart-raytracer-traversal PROPERTIES
                              LABELS "gpu;hart" RUN_SERIAL TRUE)
        add_test (NAME hart-context-lifecycle COMMAND hart_trace_test --lifecycle)
        set_tests_properties (hart-context-lifecycle PROPERTIES
                              LABELS "gpu;hart" RUN_SERIAL TRUE TIMEOUT 900)
    endif ()
endif ()
