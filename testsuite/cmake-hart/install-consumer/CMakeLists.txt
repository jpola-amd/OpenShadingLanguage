# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.15)
project (osl_install_consumer LANGUAGES CXX)

set (CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set (CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
find_package (OSL CONFIG REQUIRED)

add_executable (osl_install_consumer main.cpp)
target_compile_features (osl_install_consumer PRIVATE cxx_std_17)
# Runtime paths are supplied by the checker, not user-wide VS integration.
set_target_properties (osl_install_consumer PROPERTIES CXX_EXTENSIONS OFF
                      VS_GLOBAL_VcpkgEnabled FALSE)
target_link_libraries (osl_install_consumer PRIVATE
                      OSL::oslcomp OSL::oslquery OSL::oslexec)
