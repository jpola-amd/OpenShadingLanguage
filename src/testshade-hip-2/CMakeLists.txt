# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
function(HIP_COMPILE_TO_BC sources headers out_bitcode out_llvm extra_options)
    get_filename_component ( source_we ${sources} NAME_WE )
    get_filename_component ( source_dir ${sources} DIRECTORY )
    get_filename_component ( source_abs ${sources} ABSOLUTE)

    set(bitcode "${CMAKE_CURRENT_BINARY_DIR}/${source_we}.bc")
    set(llvm_bc "${CMAKE_CURRENT_BINARY_DIR}/${source_we}.llvm")

    set(${out_bitcode} ${bitcode} PARENT_SCOPE)
    set(${out_llvm} ${llvm_bc} PARENT_SCOPE)

    message(STATUS "Compiling HIP source file ${sources} to LLVM bitcode ${out_bitcode} for HIP_TARGET_ARCH=${HIP_TARGET_ARCH}")

    file(GLOB hip_headers "${hip_src_dir}/*.h")
    if(headers)
        list(APPEND hip_headers ${headers})
    endif()

    set(options
        -x hip
        -emit-llvm
        -ffast-math
        -fgpu-rdc
        -S 
        --cuda-device-only
        --offload-arch=${HIP_TARGET_ARCH}
        -D__HIP_PLATFORM_AMD__
        -DUSE_HIP
        -DHIP
        -DOSL_USE_FAST_MATH=1
    )

    if(extra_options)
        list(APPEND options ${extra_options})
    endif()
    
    list (TRANSFORM IMATH_INCLUDES PREPEND -I
    OUTPUT_VARIABLE ALL_IMATH_INCLUDES)
    
    list (TRANSFORM OPENEXR_INCLUDES PREPEND -I
    OUTPUT_VARIABLE ALL_OPENEXR_INCLUDES)
    
    list (TRANSFORM OpenImageIO_INCLUDES PREPEND -I
    OUTPUT_VARIABLE ALL_OpenImageIO_INCLUDES)
    set (include_dirs   
    "-I${HIP_INCLUDE_DIRS}"
    "-I${CMAKE_CURRENT_SOURCE_DIR}"
    "-I${CMAKE_BINARY_DIR}/include"
    "-I${PROJECT_SOURCE_DIR}/src/include"
    "-I${PROJECT_SOURCE_DIR}/src/cuda_common"
    ${ALL_OpenImageIO_INCLUDES}
    ${ALL_IMATH_INCLUDES}
    
    )

    message(STATUS "include_dirs: ${include_dirs}")
    
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        list(APPEND options "-g")
    endif()

    list(JOIN options " " optionsStr)
    separate_arguments(compilerOptionsList NATIVE_COMMAND ${optionsStr})

    message(STATUS "\t\t* Bitcode: ${bitcode} ")
    message(STATUS "\t\t* LLVM BC: ${llvm_bc} ")
    message(STATUS "\t\t* Compiler options: ${compilerOptionsList} ")

    add_custom_command( OUTPUT ${bitcode} ${llvm_bc}
        COMMAND ${LLVM_BC_GENERATOR}
        ARGS  ${compilerOptionsList} ${source_abs} -o ${llvm_bc} ${include_dirs}
        #COMMAND ${LLVM_OPT} ${llvm_bc} -o ${llvm_bc}
        COMMAND ${LLVM_AS_TOOL} ${llvm_bc} -f -o ${bitcode}
        DEPENDS ${source_abs}
        COMMENT "Compiling HIP source file ${source_abs} to LLVM bitcode ${bitcode}"
        VERBATIM
    )
    
endfunction()

# The 'testshade' executable
set ( testshade_srcs
      testshade.cpp
      simplerend.cpp )

if (OSL_BUILD_BATCHED)
    list(APPEND testshade_srcs
        batched_simplerend.cpp)
endif()

if (OSL_USE_HIP)
    list (APPEND testshade_srcs optixgridrender.hip.cpp)
    set ( testshade_cuda_srcs
        cuda/optix_grid_renderer.hip.cu
        )

    set (testshade_rend_lib_srcs
        cuda/rend_lib.hip.cu
        rs_simplerend.hip.cpp
        )

    set ( testshade_cuda_headers
        cuda/rend_lib.hip.h
        raytracer.h
        )

    # We need to make sure that the PTX files are regenerated whenever these
    # headers change.
    set ( extra_cuda_headers
        render_params.hip.h )

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testshade_cuda_srcs})
        HIP_COMPILE_TO_BC ( ${cudasrc} ${extra_cuda_headers} generated_bc generated_llvm "-I cuda -DOSL_TEST_SHADE_HIP -D__HIP_DEVICE_COMPILE__ --offload-device-only" )
        list (APPEND bitcode_list ${generated_bc})
        
    endforeach ()

    # Compile the renderer-supplied shadeops (rend_lib) to LLVM bitcode and PTX
    add_compile_definitions (OSL_LLVM_CUDA_BITCODE)
    HIP_SHADEOPS_COMPILE ( "rend_lib_testshade"
        rend_lib_bc
        rend_lib_llvm
        "${testshade_rend_lib_srcs}"
        "${testshade_cuda_headers}"
    )

    message(STATUS "testshade-hip-2 bitcode_sources: rend_lib_bc: ${rend_lib_bc}")
    message(STATUS "testshade-hip-2 bitcode_sources: rend_lib_llvm: ${rend_lib_llvm}")

    # Serialize the rend_lib bitcode into a CPP file to be embedded in the current target binary
    set (rend_lib_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/rend_lib_cuda.bc.cpp")
    MAKE_EMBEDDED_CPP( "rend_lib_llvm_compiled_ops" ${rend_lib_bc_cuda_cpp} ${rend_lib_bc} )
    list (APPEND testshade_srcs ${rend_lib_bc_cuda_cpp})
    list (APPEND bitcode_list ${rend_lib_bc})

    add_custom_target (testshade_ptx_hip ALL
        DEPENDS ${bitcode_list}
        SOURCES ${testshade_cuda_srcs} )

    # Install the PTX files in a fixed location so that they can be
    # loaded at run time
    install (FILES ${bitcode_list}
             DESTINATION ${OSL_BC_INSTALL_DIR})
endif()

set ( rs_srcs
    rs_simplerend.hip.cpp )

set(include_dirs ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND include_dirs ${CMAKE_SOURCE_DIR}/src/include)
list(APPEND include_dirs ${CMAKE_BINARY_DIR}/include)
list(APPEND include_dirs ${IMATH_INCLUDES})
list(APPEND include_dirs ${OpenImageIO_INCLUDES})
list(APPEND include_dirs ${HIP_INCLUDE_DIRS})

list(APPEND definitions "-DOSL_HOST_RS_BITCODE=1" "-D__HIP_PLATFORM_AMD__")


EMBED_LLVM_BITCODE_IN_CPP ( "${rs_srcs}" "_host" "testshade_llvm_compiled_rs" testshade_srcs "${definitions}" "${include_dirs}")


function(hip_target TARGET)
    target_compile_definitions(${TARGET} PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(${TARGET} PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE hip::host)
    target_compile_options(${TARGET} PRIVATE ${HIP_COMPILE_OPTIONS})
    target_compile_definitions (${TARGET} PRIVATE PTX_PATH="${OSL_BC_INSTALL_DIR}")
    # add -Wno-deprecated-declarations to suppress warnings from hip headers
    target_compile_options(${TARGET} PRIVATE -Wno-deprecated-declarations)
    target_link_options(${TARGET} PRIVATE ${HIP_LINK_OPTIONS})
endfunction()


add_executable ( testshade-hip-2 ${testshade_srcs} testshademain.cpp )
target_compile_definitions(testshade-hip-2 PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries (testshade-hip-2 PRIVATE
                        oslexec
                        oslquery
                        oslcomp
                    )
hip_target(testshade-hip-2)
install (TARGETS testshade-hip-2 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )


# As explained in PR #39, problems were reported with calling OSL from a
# Houdini plugin.  So this section sets up a version of testshade as a DSO, to
# be sure we can link it into an executable properly.
#
# But the fact that we use testshade.cpp twice -- once for regular testshade,
# once for the DSO version -- is throwing off the code coverage test, so at
# least for now, disable the dso version entirely when CODECOV is on.
# if (NOT CODECOV)
#     # The 'libtestshade' library
#     add_library ( "libtestshade-hip-2" ${testshade_srcs} )

#     set_target_properties (libtestshade-hip-2
#                            PROPERTIES
#                            VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
#                            SOVERSION ${SOVERSION}
#                            OUTPUT_NAME libtestshade-hip-2${OSL_LIBNAME_SUFFIX}
#                            )

#     target_link_libraries (libtestshade-hip-2
#                            PRIVATE
#                                oslexec oslquery oslcomp)
#     set_target_properties (libtestshade-hip-2 PROPERTIES PREFIX "")

#     install_targets ( libtestshade-hip-2 )

#     # The 'testshade_dso' executable
#     add_executable ( testshade_dso-hip-2 testshade_dso.cpp )
#     target_link_libraries (testshade_dso-hip-2
#                            PRIVATE
#                                OpenImageIO::OpenImageIO
#                                ${CMAKE_DL_LIBS} )
#     install (TARGETS testshade_dso-hip-2 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
#     #osl_optix_target(libtestshade-hip-2)
# endif ()
