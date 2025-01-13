cmake_minimum_required(VERSION 3.15)

project(example-hip CXX)

if (NOT GPU_TARGET_ARCH)
    set(GPU_TARGET_ARCH "gfx1030")
endif()

message(STATUS "GPU_TARGET_ARCH: ${GPU_TARGET_ARCH}")

if (NOT LLVM_HOME)
    message(FATAL_ERROR "Please define the path to LLVM 19 with the LLVM_HOME variable")
else()
    message(STATUS "LLVM_HOME: ${LLVM_HOME}")
endif()

# check if env variable HIP_PATH is set
if (NOT DEFINED ENV{HIP_PATH})
    message(STATUS "Setting HIP_PATH to /opt/rocm/")
    set(ENV{HIP_PATH} "/opt/rocm/")
else()
    message(STATUS "HIP_PATH: $ENV{HIP_PATH}")
endif()

if (NOT ARTIFACTS_HOME)
    message(FATAL_ERROR "Please define the path to the artifacts folder with the ARTIFACTS_HOME variable")
else()
    message(STATUS "ARTIFACTS_HOME: ${ARTIFACTS_HOME}")
endif()

# To find OSL dependencies
list(APPEND CMAKE_MODULE_PATH ${ARTIFACTS_HOME}/artifacts/lib/cmake)

find_package(LLVM 19.1 REQUIRED CONFIG HINTS ${LLVM_HOME})
# find_package(HIP REQUIRED HINTS $ENV{HIP_PATH}) # this is fucked up. It alters the add_executable and adds the --offload-arch to the invocation
set(HIP_INCLUDE_DIRS "/opt/rocm/include")
set(HIP_LIBRARIES "/opt/rocm/lib/libamdhip64.so")



find_package(Imath REQUIRED HINTS ${ARTIFACTS_HOME})
find_package(OpenImageIO REQUIRED HINTS ${ARTIFACTS_HOME})
find_package(OpenColorIO REQUIRED HINTS ${ARTIFACTS_HOME})
find_package(OSL REQUIRED HINTS ${OSL_HOME}/lib/cmake)
find_package(CUDA REQUIRED)



message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found HIP ${HIP_VERSION}")
message(STATUS "Found Imath ${Imath_VERSION}")
message(STATUS "Found OpenImageIO ${OpenImageIO_VERSION}")
message(STATUS "Found OpenColorIO ${OpenColorIO_VERSION}")
message(STATUS "Found OSL ${OSL_VERSION}")

message(STATUS "OSL INCLUDE DIRS: ${OSL_INCLUDES}")


# Get Imath::Imath INTERFACE_INCLUDE_DIRECTORIES property to include in bitcode compilation
get_property(IMATH_INCLUDES TARGET Imath::Imath PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "IMATH INCLUDE DIRS: ${IMATH_INCLUDES}")

function(find_llvm_program OUT_VARIABLE program_name description)
    if (NOT ${OUT_VARIABLE})
        find_program(${OUT_VARIABLE} 
            NAMES "${program_name}" 
            PATHS "${LLVM_HOME}/bin" 
            NO_DEFAULT_PATH 
            NO_CMAKE_SYSTEM_PATH 
            NO_SYSTEM_ENVIRONMENT_PATH 
            NO_CMAKE_ENVIRONMENT_PATH 
            NO_CMAKE_PATH
        )
    endif()

    if (${OUT_VARIABLE})
        message(STATUS "Using ${description}: ${${OUT_VARIABLE}}")
    else()
        message(FATAL_ERROR "Can't find ${description} (${program_name})")
    endif()
endfunction()

find_llvm_program(LLVM_BC_GENERATOR "clang++" "bitcode generator")
find_llvm_program(LLVM_LINK "llvm-link" "LLVM linker")
find_llvm_program(LLVM_OPT "opt" "LLVM optimizer")
find_llvm_program(LLVM_AS_TOOL "llvm-as" "LLVM assembler")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(hip_macros)


set(bitcode_sources 
    "${CMAKE_CURRENT_SOURCE_DIR}/op.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/RenderLib.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/HipRenderer.hip.cpp"
)

foreach(bitcode_src ${bitcode_sources})
    HIP_COMPILE_TO_BC("${bitcode_src}" "" bitcode bitcode_llvm "-I${OSL_INCLUDES} -I${IMATH_INCLUDES}")
    list(APPEND bitcodes ${bitcode})
    list(APPEND bitcode_llvs ${bitcode_llvm})
endforeach()


add_custom_target(amdgcn_bitcodes ALL 
    DEPENDS ${bitcodes}
)

set(sources
    main.cpp
    HipRenderer.cpp
    HipRenderer.hpp
    RenderLib.hpp
    StringHash.hpp
    Assert.hpp
)

add_executable(example ${sources})
target_compile_definitions(example PRIVATE -D__HIP_PLATFORM_AMD__)
target_include_directories(example PRIVATE ${HIP_INCLUDE_DIRS})
add_dependencies(example amdgcn_bitcodes)

include_directories(
    ${OSL_INCLUDES}
    ${OpenImageIO_INCLUDES}
    ${OpenColorIO_INCLUDES}
)
target_include_directories(example PRIVATE 
    ${CUDA_INCLUDE_DIRS}
    ${HIP_INCLUDES}
    ${OSL_INCLUDES}
    ${OpenImageIO_INCLUDES}
    ${OpenColorIO_INCLUDES}
)

target_link_libraries(example PRIVATE ${HIP_LIBRARIES})
target_link_libraries(example PRIVATE OSL::oslcomp OSL::oslexec OSL::osl.imageio)
target_link_libraries(example PRIVATE ${OpenImageIO_LIBRARIES})
target_link_libraries(example PRIVATE ${OpenColorIO_LIBRARIES})


add_custom_command(TARGET example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/shader/test_add.oso
    ${CMAKE_BINARY_DIR}/shader/test_add.oso
)