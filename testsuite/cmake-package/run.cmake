# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required (VERSION 3.19)
if (NOT OSL_SOURCE_DIR OR NOT TEST_BINARY_DIR)
    message (FATAL_ERROR "Set OSL_SOURCE_DIR and TEST_BINARY_DIR")
endif ()
file (TO_CMAKE_PATH "${OSL_SOURCE_DIR}" OSL_SOURCE_DIR)
file (TO_CMAKE_PATH "${TEST_BINARY_DIR}" TEST_BINARY_DIR)
include (CMakePackageConfigHelpers)

foreach (scenario shared static static-hart static-partio)
    set (root "${TEST_BINARY_DIR}/${scenario}")
    file (MAKE_DIRECTORY "${root}/include" "${root}/lib/cmake/OSL")
    set (BUILD_SHARED_LIBS OFF)
    set (OSL_USE_HART OFF)
    set (OSL_USE_OPTIX ON)
    set (OSL_BUILD_TESTS ON)
    set (partio_FOUND OFF)
    set (LLVM_VERSION 23.0.0git)
    set (PROJECT_NAME OSL)
    set (CMAKE_INSTALL_INCLUDEDIR include)
    set (CMAKE_INSTALL_LIBDIR lib)
    if (scenario STREQUAL shared)
        set (BUILD_SHARED_LIBS ON)
        set (OSL_USE_HART ON)
    elseif (scenario STREQUAL static-hart)
        set (OSL_USE_HART ON)
    elseif (scenario STREQUAL static-partio)
        set (partio_FOUND ON)
    endif ()

    foreach (dep Imath OpenImageIO pugixml hip amd.hart partio ZLIB)
        file (MAKE_DIRECTORY "${root}/lib/cmake/${dep}")
        file (WRITE "${root}/lib/cmake/${dep}/${dep}Config.cmake"
              "set(${dep}_FOUND TRUE)\nset(found_${dep} TRUE)\n")
    endforeach ()
    # No SDK exists outside this fixture. The imported archive references
    # targets which OSLConfig must discover for its consumers.
    file (APPEND "${root}/lib/cmake/hip/hipConfig.cmake"
          "add_library(hip::host INTERFACE IMPORTED)\n")
    file (APPEND "${root}/lib/cmake/amd.hart/amd.hartConfig.cmake"
          "add_library(amd::hart INTERFACE IMPORTED)\n")
    file (WRITE "${root}/lib/cmake/OSL/OSLTargets.cmake"
          "add_library(OSL::oslexec INTERFACE IMPORTED)\n")
    configure_package_config_file (
        "${OSL_SOURCE_DIR}/src/cmake/Config.cmake.in"
        "${root}/lib/cmake/OSL/OSLConfig.cmake"
        INSTALL_DESTINATION lib/cmake/OSL)
    file (WRITE "${root}/CMakeLists.txt"
        "cmake_minimum_required(VERSION 3.19)\n"
        "project(package_consumer NONE)\n"
        "set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)\n"
        "find_package(OSL CONFIG REQUIRED PATHS \"${root}/lib/cmake/OSL\" NO_DEFAULT_PATH)\n"
        "if (NOT OSL_USE_OPTIX OR NOT OSL_LLVM_VERSION STREQUAL \"23.0.0git\")\n"
        "  message(FATAL_ERROR \"Missing configured backend/LLVM metadata\")\n"
        "endif()\n")
    if (NOT BUILD_SHARED_LIBS)
        file (APPEND "${root}/CMakeLists.txt"
              "if (NOT found_pugixml)\n"
              "  message(FATAL_ERROR \"Missing static pugixml dependency\")\n"
              "endif()\n")
        if (OSL_USE_HART)
            file (APPEND "${root}/CMakeLists.txt"
                  "if (NOT TARGET hip::host OR NOT TARGET amd::hart)\n"
                  "  message(FATAL_ERROR \"Missing static HART SDK targets\")\n"
                  "endif()\n")
        endif ()
        if (partio_FOUND)
            file (APPEND "${root}/CMakeLists.txt"
                  "if (NOT found_partio OR NOT found_ZLIB)\n"
                  "  message(FATAL_ERROR \"Missing static Partio dependencies\")\n"
                  "endif()\n")
        endif ()
    else ()
        file (APPEND "${root}/CMakeLists.txt"
              "if (found_pugixml OR found_hip OR found_amd.hart)\n"
              "  message(FATAL_ERROR \"Shared consumers should not need private SDKs\")\n"
              "endif()\n")
    endif ()
    execute_process (COMMAND "${CMAKE_COMMAND}"
        -S "${root}" -B "${root}/build" "-DCMAKE_PREFIX_PATH=${root}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if (NOT result EQUAL 0)
        message (FATAL_ERROR "${scenario}: ${output}\n${error}")
    endif ()
    message (STATUS "Package export fixture passed: ${scenario}")
endforeach ()
