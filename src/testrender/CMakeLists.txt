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
    endif ()
endif ()
