# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required (VERSION 3.20)
project (OSLInstalledConsumer LANGUAGES CXX)
find_package (OSL CONFIG REQUIRED)

add_executable (oiio_compat_consumer consumer.cpp)
target_link_libraries (oiio_compat_consumer PRIVATE OSL::oslexec OSL::oslcomp)
set_target_properties (oiio_compat_consumer PROPERTIES
    VS_GLOBAL_VcpkgEnabled FALSE)
get_target_property (_osl_type OSL::oslexec TYPE)
if (_osl_type STREQUAL "STATIC_LIBRARY")
    target_compile_definitions (oiio_compat_consumer PRIVATE EXPECT_STATIC_OSL=1)
endif ()

find_file (OSL_STDOSL stdosl.h
    HINTS "${OSL_INCLUDE_DIR}/../share/OSL/shaders" REQUIRED)
enable_testing ()
add_test (NAME installed-oiio-compat
    COMMAND oiio_compat_consumer "${OSL_STDOSL}")
