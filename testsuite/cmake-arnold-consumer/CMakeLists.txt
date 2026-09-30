# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.20)
project (OSLArnoldInstalledConsumer LANGUAGES CXX)

set (CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set (CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
find_package (OSL CONFIG REQUIRED)
find_package (Threads REQUIRED)

# Compile exactly the same public-API acceptance checks without build-tree
# headers, libraries, or private target dependencies.
add_executable (arnold_compat_consumer
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/liboslexec/arnold_compat_test.cpp")
add_executable (oiio_compat_consumer
    "${CMAKE_CURRENT_SOURCE_DIR}/../oiio-compat/consumer.cpp")
get_target_property (_osl_type OSL::oslexec TYPE)
foreach (consumer arnold_compat_consumer oiio_compat_consumer)
    target_compile_features (${consumer} PRIVATE cxx_std_17)
    target_link_libraries (${consumer} PRIVATE OSL::oslexec OSL::oslcomp)
    set_target_properties (${consumer} PROPERTIES
        CXX_EXTENSIONS OFF VS_GLOBAL_VcpkgEnabled FALSE)
    if (_osl_type STREQUAL "STATIC_LIBRARY")
        target_compile_definitions (${consumer} PRIVATE EXPECT_STATIC_OSL=1)
    endif ()
endforeach ()
target_link_libraries (arnold_compat_consumer PRIVATE Threads::Threads)
target_link_libraries (oiio_compat_consumer PRIVATE OSL::oslquery)

find_file (OSL_STDOSL stdosl.h
    HINTS "${OSL_INCLUDE_DIR}/../share/OSL/shaders" REQUIRED)
enable_testing ()
add_test (NAME installed-arnold-compat
    COMMAND arnold_compat_consumer "${OSL_STDOSL}")
add_test (NAME installed-oiio-compat
    COMMAND oiio_compat_consumer "${OSL_STDOSL}")
