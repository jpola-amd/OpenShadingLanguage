# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# The 'testshade' executable
set ( testshade_srcs
      testshade.cpp
      simplerend.cpp )

if (OSL_BUILD_BATCHED)
    list(APPEND testshade_srcs
        batched_simplerend.cpp)
endif()

if (OSL_USE_HART)
    list (APPEND testshade_srcs hartgridrender.cpp harttexture.cpp)
endif ()

if (OSL_USE_OPTIX)
    list (APPEND testshade_srcs optixgridrender.cpp)
    set ( testshade_cuda_srcs
        cuda/optix_grid_renderer.cu
        )

    set (testshade_rend_lib_srcs
        ../testrender/cuda/rend_lib.cu
        rs_simplerend.cpp
        )

    set ( testshade_cuda_headers
        ../testrender/cuda/rend_lib.h
        ../testrender/raytracer.h
        )

    # We need to make sure that the PTX files are regenerated whenever these
    # headers change.
    set ( extra_cuda_headers
        render_params.h )

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testshade_cuda_srcs})
        NVCC_COMPILE ( ${cudasrc} ${extra_cuda_headers} ptx_generated "-I../testrender/cuda" "")
        list (APPEND ptx_list ${ptx_generated})
    endforeach ()

    # Compile the renderer-supplied shadeops (rend_lib) to LLVM bitcode and PTX
    add_compile_definitions (OSL_LLVM_CUDA_BITCODE)
    CUDA_SHADEOPS_COMPILE ( "rend_lib_testshade"
        rend_lib_bc
        rend_lib_ptx
        "${testshade_rend_lib_srcs}"
        "${testshade_cuda_headers}"
    )

    # Serialize the rend_lib bitcode into a CPP file to be embedded in the current target binary
    set (rend_lib_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/rend_lib_cuda.bc.cpp")
    MAKE_EMBEDDED_CPP( "rend_lib_llvm_compiled_ops" ${rend_lib_bc_cuda_cpp} ${rend_lib_bc} )
    list (APPEND testshade_srcs ${rend_lib_bc_cuda_cpp})
    list (APPEND ptx_list ${rend_lib_ptx})

    add_custom_target (testshade_ptx ALL
        DEPENDS ${ptx_list}
        SOURCES ${testshade_cuda_srcs} )

    # Install the PTX files in a fixed location so that they can be
    # loaded at run time
    install (FILES ${ptx_list}
             DESTINATION ${OSL_PTX_INSTALL_DIR})
endif()

set ( rs_srcs
    rs_simplerend.cpp )

set(include_dirs ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND include_dirs ${CMAKE_SOURCE_DIR}/src/include)
list(APPEND include_dirs ${CMAKE_BINARY_DIR}/include)
list(APPEND include_dirs ${IMATH_INCLUDES})
list(APPEND include_dirs ${OpenImageIO_INCLUDES})

if (OSL_USE_HART)
    set (hart_generated_declarations "")
    set (hart_generated_entries "")
    set (hart_generated_grids)
    if (USE_LLVM_BITCODE)
        get_target_property (hart_include_dirs amd::hart INTERFACE_INCLUDE_DIRECTORIES)
        list (APPEND hart_include_dirs ${include_dirs})
        list (REMOVE_DUPLICATES hart_include_dirs)
        set (exec_headers "${CMAKE_CURRENT_SOURCE_DIR}/hartgeneratedparams.h")
        set (hart_grid_architectures ${HART_TARGET_ARCHITECTURES})
        list (REMOVE_DUPLICATES hart_grid_architectures)
        foreach (arch IN LISTS hart_grid_architectures)
            MAKE_HART_BITCODE ("hart/generated_grid.hip" "_hart_${arch}" "${arch}"
                              generated_grid_bc "" "${hart_include_dirs}")
            set (symbol "testshade_hart_raygen_${arch}")
            set (embedded "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/generated_grid.bc.cpp")
            MAKE_EMBEDDED_CPP ("${symbol}" "${embedded}" "${generated_grid_bc}")
            list (APPEND testshade_srcs "${embedded}")
            list (APPEND hart_generated_grids "${generated_grid_bc}")
            string (APPEND hart_generated_declarations
                "extern unsigned char ${symbol}_block[];\nextern int ${symbol}_size;\n")
            string (APPEND hart_generated_entries
                "    { \"${arch}\", ${symbol}_block, &${symbol}_size },\n")
        endforeach ()
    endif ()
    file (WRITE "${CMAKE_CURRENT_BINARY_DIR}/hart_generated_bitcode.h"
        "${hart_generated_declarations}"
        "struct HartEmbeddedRaygen { const char* arch; const unsigned char* data; const int* size; };\n"
        "static const HartEmbeddedRaygen hart_generated_raygens[] = {\n"
        "${hart_generated_entries}    { nullptr, nullptr, nullptr }\n};\n")
endif ()

EMBED_LLVM_BITCODE_IN_CPP ( "${rs_srcs}" "_host" "testshade_llvm_compiled_rs" testshade_srcs "-DOSL_HOST_RS_BITCODE=1" "${include_dirs}")

add_executable ( testshade ${testshade_srcs} testshademain.cpp )

target_include_directories (testshade BEFORE PRIVATE ${OpenImageIO_INCLUDES})
target_include_directories (testshade PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries (testshade
                       PRIVATE
                           oslexec oslquery oslcomp)
target_compile_definitions (testshade PRIVATE OSL_TESTSHADE_HART=$<BOOL:${OSL_USE_HART}>)
osl_hart_target (testshade)
if (OSL_USE_HART)
    target_link_libraries (testshade PRIVATE
        ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES})
endif ()
if (OSL_USE_HART AND WIN32)
    find_package (hiprtc CONFIG REQUIRED
                  PATHS "${HIP_PACKAGE_PREFIX_DIR}" NO_DEFAULT_PATH)
    find_package (rocm-kpack CONFIG QUIET
                  PATHS "${HIP_PACKAGE_PREFIX_DIR}" NO_DEFAULT_PATH)
    # System32 precedes PATH and may contain incompatible display-driver copies.
    set (hart_comgr_dll "${HIP_PACKAGE_PREFIX_DIR}/bin/amd_comgr.dll")
    if (NOT EXISTS "${hart_comgr_dll}")
        message (FATAL_ERROR "Cannot deploy the selected ROCm COMGR: ${hart_comgr_dll}")
    endif ()
    set (hart_runtime_files
        "$<TARGET_FILE:amd::hart>"
        "$<TARGET_FILE:amd::shader_compiler>"
        "$<TARGET_FILE:amd::shader_stack_runtime>"
        "$<TARGET_FILE_DIR:amd::hart>/hart-native-codegen.exe"
        "$<TARGET_FILE:hip::amdhip64>"
        "$<TARGET_FILE:hiprtc::hiprtc>"
        "${hart_comgr_dll}")
    file (GLOB hart_hiprtc_builtins CONFIGURE_DEPENDS
          "${HIP_PACKAGE_PREFIX_DIR}/bin/hiprtc-builtins*.dll")
    list (APPEND hart_runtime_files ${hart_hiprtc_builtins})
    if (TARGET rocm::rocm_kpack)
        list (APPEND hart_runtime_files "$<TARGET_FILE:rocm::rocm_kpack>")
    endif ()
    add_custom_command (TARGET testshade POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${hart_runtime_files} "$<TARGET_FILE_DIR:testshade>"
        COMMAND_EXPAND_LISTS VERBATIM)
    install (FILES ${hart_runtime_files} DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif ()
if (OSL_USE_OPTIX)
    add_dependencies(testshade testshade_ptx)
endif ()

install (TARGETS testshade RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

osl_optix_target(testshade)


# As explained in PR #39, problems were reported with calling OSL from a
# Houdini plugin.  So this section sets up a version of testshade as a DSO, to
# be sure we can link it into an executable properly.
#
# But the fact that we use testshade.cpp twice -- once for regular testshade,
# once for the DSO version -- is throwing off the code coverage test, so at
# least for now, disable the dso version entirely when CODECOV is on.
if (NOT CODECOV)
    # The 'libtestshade' library
    add_library ( "libtestshade" ${testshade_srcs} )
    add_dependencies(libtestshade testshade)

    set_target_properties (libtestshade
                           PROPERTIES
                           VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
                           SOVERSION ${SOVERSION}
                           OUTPUT_NAME libtestshade${OSL_LIBNAME_SUFFIX}
                           )

    # Only libtestshade exports test_shade; keep it out of the global map.
    if (VISIBILITY_MAP_COMMAND)
        set_property (TARGET libtestshade
                      APPEND PROPERTY LINK_FLAGS
                      -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/testshade_symbols.map)
    endif ()

    target_link_libraries (libtestshade
                           PRIVATE
                               oslexec oslquery oslcomp)
    target_compile_definitions (libtestshade PRIVATE OSL_TESTSHADE_HART=$<BOOL:${OSL_USE_HART}>)
    osl_hart_target (libtestshade)
    if (OSL_USE_HART)
        target_link_libraries (libtestshade PRIVATE
            ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES})
    endif ()
    target_include_directories (libtestshade BEFORE PRIVATE ${OpenImageIO_INCLUDES})
    target_include_directories (libtestshade PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    set_target_properties (libtestshade PROPERTIES PREFIX "")

    install_targets ( libtestshade )

    # The 'testshade_dso' executable
    add_executable ( testshade_dso testshade_dso.cpp )
    target_include_directories (testshade_dso BEFORE PRIVATE ${OpenImageIO_INCLUDES})
    target_link_libraries (testshade_dso
                           PRIVATE
                               OpenImageIO::OpenImageIO
                               ${CMAKE_DL_LIBS} )
    install (TARGETS testshade_dso RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
    osl_optix_target(libtestshade)
endif ()

if (OSL_USE_HART AND USE_LLVM_BITCODE)
    get_target_property (hart_include_dirs amd::hart INTERFACE_INCLUDE_DIRECTORIES)
    list (APPEND hart_include_dirs ${include_dirs})
    list (REMOVE_DUPLICATES hart_include_dirs)
    set (exec_headers "${CMAKE_CURRENT_SOURCE_DIR}/hartgridparams.h"
                      "${CMAKE_CURRENT_SOURCE_DIR}/hart/callable_smoke.h")
    set (hart_grid_bitcode)
    set (hart_callable_grids)
    set (hart_callable_bitcode)
    get_target_property (shadeops_binary_dir osl_hart_bitcode BINARY_DIR)
    set (hart_grid_architectures ${HART_TARGET_ARCHITECTURES})
    list (REMOVE_DUPLICATES hart_grid_architectures)
    foreach (arch IN LISTS hart_grid_architectures)
        MAKE_HART_BITCODE ("hart/grid_smoke.hip" "_hart_${arch}" "${arch}"
                          grid_bc "" "${hart_include_dirs}")
        list (APPEND hart_grid_bitcode "${grid_bc}")
        MAKE_HART_BITCODE ("hart/callable_grid.hip" "_hart_${arch}" "${arch}"
                          callable_grid_bc "" "${hart_include_dirs}")
        MAKE_HART_BITCODE ("hart/callable_shader.hip" "_hart_${arch}" "${arch}"
                          callable_shader_bc "" "${hart_include_dirs}")
        set (shadeops_bc "${shadeops_binary_dir}/hart/${arch}/shadeops_hart.bc")
        set (callable_bc "${CMAKE_CURRENT_BINARY_DIR}/hart/${arch}/callable_shadeops_hart_${arch}.bc")
        add_custom_command (OUTPUT "${callable_bc}"
            COMMAND "${LLVM_LINK_TOOL}" --only-needed
                "${callable_shader_bc}" "${shadeops_bc}" -o "${callable_bc}"
            DEPENDS "${callable_shader_bc}" "${shadeops_bc}" "${LLVM_LINK_TOOL}"
            VERBATIM)
        list (APPEND hart_callable_grids "${callable_grid_bc}")
        list (APPEND hart_callable_bitcode "${callable_bc}")
    endforeach ()
    add_custom_target (testshade_hart_bitcode ALL
        DEPENDS ${hart_grid_bitcode} ${hart_callable_grids} ${hart_callable_bitcode})
    add_dependencies (testshade_hart_bitcode osl_hart_bitcode)
    add_dependencies (testshade testshade_hart_bitcode)
    if (BUILD_TESTING)
        add_test (NAME hart-grid-bitcode
            COMMAND "${CMAKE_COMMAND}"
                "-DLLVM_OPT_TOOL=${LLVM_OPT_TOOL}"
                "-DBITCODE_FILES=${hart_grid_bitcode};${hart_callable_grids}"
                "-DGENERATED_GRID_FILES=${hart_generated_grids}"
                "-DCALLABLE_FILES=${hart_callable_bitcode}"
                -P "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-grid-bitcode.cmake")
        if ("$ENV{TESTSUITE_HART}")
            list (GET hart_grid_bitcode 0 hart_grid_test_module)
            list (GET hart_callable_grids 0 hart_callable_test_grid)
            list (GET hart_callable_bitcode 0 hart_callable_test_module)
            add_test (NAME hart-grid-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-grid.py"
                    "$<TARGET_FILE:testshade>" --module "${hart_grid_test_module}"
                    --callable-grid "${hart_callable_test_grid}"
                    --callable-module "${hart_callable_test_module}"
                    --llvm-opt "${LLVM_OPT_TOOL}")
            set_tests_properties (hart-grid-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-generated-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu)
            set_tests_properties (hart-generated-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-loops-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --loops)
            set_tests_properties (hart-loops-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-derivatives-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --derivatives)
            set_tests_properties (hart-derivatives-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-surface-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --surface)
            set_tests_properties (hart-surface-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-filterwidth-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --filterwidth)
            set_tests_properties (hart-filterwidth-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-noise-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --noise)
            set_tests_properties (hart-noise-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-noise-families-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --noise-families)
            set_tests_properties (hart-noise-families-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-math-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --math)
            set_tests_properties (hart-math-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-procedural-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --procedural)
            set_tests_properties (hart-procedural-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-texture-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --textures)
            set_tests_properties (hart-texture-runtime PROPERTIES LABELS "hart;gpu")
            add_test (NAME hart-matrix-runtime
                COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
                    "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>" --gpu --matrices)
            set_tests_properties (hart-matrix-runtime PROPERTIES LABELS "hart;gpu")
        endif ()
    endif ()
endif ()

if (BUILD_TESTING AND OSL_USE_HART)
    if (OSL_BUILD_TESTS)
        add_executable (hart_texture_test hart_texture_test.cpp harttexture.cpp)
        target_include_directories (hart_texture_test BEFORE PRIVATE ${OpenImageIO_INCLUDES})
        target_link_libraries (hart_texture_test PRIVATE OpenImageIO::OpenImageIO hip::host)
        add_dependencies (hart_texture_test testshade)
        set_target_properties (hart_texture_test PROPERTIES FOLDER "Unit Tests")
        if ("$ENV{TESTSUITE_HART}" STREQUAL "1")
            add_test (NAME hart-texture-resources COMMAND hart_texture_test)
            set_tests_properties (hart-texture-resources PROPERTIES
                LABELS "hart;gpu" WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        endif ()
    endif ()
    add_test (NAME hart-grid-cli
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-grid.py"
            "$<TARGET_FILE:testshade>" --llvm-opt "${LLVM_OPT_TOOL}")
    add_test (NAME hart-generated-cli
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/testsuite/cmake-hart/check-generated.py"
            "$<TARGET_FILE:testshade>" --oslc "$<TARGET_FILE:oslc>")
endif ()
