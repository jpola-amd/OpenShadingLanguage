


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
    )

    if(extra_options)
        list(APPEND options ${extra_options})
    endif()
    
    
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
        ARGS  ${compilerOptionsList} ${source_abs} -o ${llvm_bc}
        #COMMAND ${LLVM_OPT} ${llvm_bc} -o ${llvm_bc}
        COMMAND ${LLVM_AS_TOOL} ${llvm_bc} -f -o ${bitcode}
        DEPENDS ${source_abs}
        COMMENT "Compiling HIP source file ${source_abs} to LLVM bitcode ${bitcode}"
        VERBATIM
    )
    
endfunction()


set (testshade-hip_srcs
    assert_hip.hpp
    argparse.hpp
    argparse.cpp     
    utils.hpp
    utils.cpp
    oslcompiler.hpp
    oslcompiler.cpp
    renderservices_strdecls.hpp
    renderstate.hpp
    simplerenderer.hpp
    simplerenderer.cpp
    hiprenderer.hpp
    hiprenderer.cpp
    main.cpp )
 
if (OSL_BUILD_BATCHED)
    status(FATAL_ERROR "OSL_BUILD_BATCHED is not supported for HIP")
endif()


if (NOT OSL_USE_HIP)
    status(FATAL_ERROR "OSL_USE_HIP is not enabled")
endif()

list( APPEND include_dirs 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_BINARY_DIR}/include
    ${IMATH_INCLUDES}
    ${OpenImageIO_INCLUDES}
)

# Handle all the HIP deps including HIPRT but not right now.
message(STATUS "HIP_LIBRARIES: ${HIP_LIBRARIES}")
message(STATUS "HIP_INCLUDE_DIRS: ${HIP_INCLUDE_DIRS}")


set(bitcode_sources 
    ${CMAKE_CURRENT_SOURCE_DIR}/device/hip_grid_renderer.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/rend_lib.hip.cpp
)

message(STATUS "testshde-hip bitcode_sources:")
foreach (src ${bitcode_sources})
    message(STATUS "\t\t* ${src}")
endforeach()

# add a global definition to the bitcode sources
# this helps to identify the bitcode sources in the code
set(device_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/device)
set(device_bc_dir ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "device_src_dir: ${device_src_dir}")
message(STATUS "device_bc_dir: ${device_bc_dir}")

foreach(bitcode_src ${bitcode_sources})
    HIP_COMPILE_TO_BC("${bitcode_src}"
        "" 
        bitcode bitcode_llvm 
        "-I${CMAKE_SOURCE_DIR}/src/include -I${CMAKE_BINARY_DIR}/include -I${IMATH_INCLUDES} -I${CMAKE_SOURCE_DIR}/..")
        # here instead of the series of the -I flags we should use just -I${OSL_INCLUDES} but it is not defined check example-hip
    list(APPEND bitcodes ${bitcode})
    list(APPEND bitcode_llvms ${bitcode_llvm})
endforeach()

add_custom_target(amdgcn_bitcodes ALL 
    DEPENDS ${bitcodes}
)


# # compile the bitcodes
# foreach (src ${bitcode_sources})
#     string(REPLACE ".cpp" ".bc" bc ${src})
#     string(REPLACE ".cpp" ".hip" hip ${src})
#     add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bc}
#                        COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} -emit-llvm -c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${CMAKE_CURRENT_BINARY_DIR}/${bc}
#                        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
#     add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${hip}
#                        COMMAND ${HIP_CLANG_EXECUTABLE} -x hip -c ${CMAKE_CURRENT_BINARY_DIR}/${bc} -o ${CMAKE_CURRENT_BINARY_DIR}/${hip}
#                        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bc})
#     list(APPEND testshade-hip_bitcodes ${CMAKE_CURRENT_BINARY_DIR}/${bc})
#     list(APPEND testshade-hip_hipfiles ${CMAKE_CURRENT_BINARY_DIR}/${hip})

add_executable(testshade-hip ${testshade-hip_srcs})
target_include_directories(testshade-hip PRIVATE ${include_dirs})
target_link_libraries(testshade-hip PRIVATE oslexec oslquery oslcomp hip::host)
target_compile_definitions(testshade-hip PRIVATE __HIP_PLATFORM_AMD__)
target_compile_definitions(testshade-hip PRIVATE HIP_BC_DIR="${device_bc_dir}" HIP_SRC_DIR="${device_src_dir}")

add_dependencies(testshade-hip amdgcn_bitcodes)

install (TARGETS testshade-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
