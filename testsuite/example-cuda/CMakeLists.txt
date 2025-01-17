# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required(VERSION 3.15)
project(examplecuda LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CC compiler: ${CMAKE_C_COMPILER}")
if (NOT LLVM_HOME)
    message(FATAL_ERROR "Please define the path to LLVM 19 with the LLVM_HOME variable")
else()
    message(STATUS "LLVM_HOME: ${LLVM_HOME}")
endif()

if (NOT ARTIFACTS_HOME)
    message(FATAL_ERROR "Please define the path to the artifacts folder with the ARTIFACTS_HOME variable")
else()
    message(STATUS "ARTIFACTS_HOME: ${ARTIFACTS_HOME}")
endif()

message(STATUS "OSL_HOME: ${OSL_HOME}")
set(CMAKE_MODULE_PATH "${OSL_LIST_DIR}/src/cmake;${OSL_LIST_DIR}/src/cmake/modules")

find_package(LLVM 19.1 REQUIRED CONFIG HINTS ${LLVM_HOME})

find_package(Imath REQUIRED HINTS ${ARTIFACTS_HOME})
find_package(OpenImageIO REQUIRED HINTS ${ARTIFACTS_HOME})
find_package(OpenColorIO REQUIRED HINTS ${ARTIFACTS_HOME})
find_package(OSL REQUIRED HINTS ${OSL_HOME}/lib/cmake)
find_package(CUDA REQUIRED)

# Make the build area layout look like we expect
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# TODO: what are the path suffixes on other platforms?
find_library(CUDA_nvrtc_LIBRARY nvrtc HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(CUDA_cuda_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib/stubs lib64/stubs)

# TODO: move to sm_60?
set(CUDA_TARGET_ARCH sm_86)

set (CMAKE_CXX_STANDARD 17 CACHE STRING
     "C++ standard to build with (17, 20, etc.)")
# set env variable NVCC_CCBIN to the path of the gcc compiler
# to be used by nvcc

# Get Imath::Imath INTERFACE_INCLUDE_DIRECTORIES property to include in bitcode compilation
get_property(IMATH_INCLUDES TARGET Imath::Imath PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "IMATH INCLUDE DIRS: ${IMATH_INCLUDES}")

message(STATUS "NVCC_CCBIN: $ENV{NVCC_CCBIN}")
# Compile our "renderer" to PTX
cuda_compile_ptx(CUDA_PTX_FILES cuda_grid_renderer.cu
    OPTIONS --gpu-architecture=${CUDA_TARGET_ARCH} --use_fast_math -dc
            --std=c++${CMAKE_CXX_STANDARD}
            --expt-relaxed-constexpr
            -I${OSL_INCLUDES}
            -I${IMATH_INCLUDES}
            -I${OpenImageIO_INCLUDES}
            -I${OPTIX_INCLUDES}
)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/cuda_grid_renderer.ptx
    COMMAND ${CMAKE_COMMAND} -E rename ${CUDA_PTX_FILES} ${CMAKE_BINARY_DIR}/cuda_grid_renderer.ptx
    DEPENDS ${CUDA_PTX_FILES}
)
add_custom_target(cuda_grid_renderer_ptx ALL
    DEPENDS ${CMAKE_BINARY_DIR}/cuda_grid_renderer.ptx cuda_grid_renderer.cu
    SOURCES cuda_grid_renderer.cu
)

# Compile the rend_lib shadeops to PTX
cuda_compile_ptx(CUDA_PTX_FILES rend_lib.cu
    OPTIONS --gpu-architecture=${CUDA_TARGET_ARCH} --use_fast_math -dc
            --std=c++${CMAKE_CXX_STANDARD}
            --expt-relaxed-constexpr
            -I${OSL_INCLUDES}
            -I${IMATH_INCLUDES}
            -I${OpenImageIO_INCLUDES}
            -I${OPTIX_INCLUDES}
)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/rend_lib.ptx
    COMMAND ${CMAKE_COMMAND} -E rename ${CUDA_PTX_FILES} ${CMAKE_BINARY_DIR}/rend_lib.ptx
    DEPENDS ${CUDA_PTX_FILES}
)
add_custom_target(rend_lib_ptx ALL
    DEPENDS ${CMAKE_BINARY_DIR}/rend_lib.ptx rend_lib.cu
    SOURCES rend_lib.cu
)

# Compile and link the main executable
add_executable(example-cuda example-cuda.cpp cuda_grid_renderer.cpp)

target_link_libraries(example-cuda PRIVATE OSL::oslexec OSL::oslquery ${CUDA_LIBRARIES} ${CUDA_nvrtc_LIBRARY} ${CUDA_cuda_LIBRARY})
target_include_directories(example-cuda PRIVATE ${CUDA_INCLUDE_DIRS})
set_property(TARGET example-cuda PROPERTY CXX_STANDARD 17)

install(TARGETS example-cuda DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
