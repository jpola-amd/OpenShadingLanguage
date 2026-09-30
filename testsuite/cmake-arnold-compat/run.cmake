# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Run after installation, independently of the OSL build:
# cmake -DOSL_SOURCE_DIR=... -DOSL_INSTALL_PREFIX=... -DTEST_BINARY_DIR=...
#       [-DCMAKE_PREFIX_PATH=dependency-prefixes] [-DTEST_GENERATOR=...]
#       [-DOSL_DEPENDENCY_BUILD_DIR=build-with-dependency-cache]
#       [-DTEST_CONFIG=Release] -P testsuite/cmake-arnold-compat/run.cmake
# The caller supplies any external dependency runtime paths, just as for the
# oiio-compat installed consumer. No OSL build-tree path is used for discovery.

cmake_minimum_required (VERSION 3.20)

foreach (required OSL_SOURCE_DIR OSL_INSTALL_PREFIX TEST_BINARY_DIR)
    if (NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message (FATAL_ERROR "${required} is required")
    endif ()
endforeach ()
if (NOT TEST_CONFIG)
    set (TEST_CONFIG Release)
endif ()
find_file (_osl_config OSLConfig.cmake
    PATHS "${OSL_INSTALL_PREFIX}/lib/cmake/OSL"
          "${OSL_INSTALL_PREFIX}/lib64/cmake/OSL"
    NO_DEFAULT_PATH REQUIRED)
get_filename_component (_osl_dir "${_osl_config}" DIRECTORY)
file (MAKE_DIRECTORY "${TEST_BINARY_DIR}")

function (checked_command name)
    execute_process (COMMAND ${ARGN}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    file (WRITE "${TEST_BINARY_DIR}/${name}.log" "${output}\n${error}")
    if (NOT result STREQUAL "0")
        message (FATAL_ERROR "${name} failed (${result}):\n${output}\n${error}")
    endif ()
    message (STATUS "Arnold installed consumer: ${name} passed")
endfunction ()

set (_generator)
if (TEST_GENERATOR)
    list (APPEND _generator -G "${TEST_GENERATOR}")
endif ()
if (TEST_GENERATOR_PLATFORM)
    list (APPEND _generator -A "${TEST_GENERATOR_PLATFORM}")
endif ()
if (TEST_GENERATOR_TOOLSET)
    list (APPEND _generator -T "${TEST_GENERATOR_TOOLSET}")
endif ()

# Static profiles may need exact dependency libraries rather than discovery
# by prefix (notably the /MD JPEG build on Windows). Import only dependency
# hints, never OSL_DIR, OSL include paths, or OSL libraries from a build cache.
set (_dependency_hints
    OpenImageIO_DIR Imath_DIR
    ZLIB_INCLUDE_DIR ZLIB_LIBRARY_RELEASE
    PNG_PNG_INCLUDE_DIR PNG_LIBRARY_RELEASE
    TIFF_INCLUDE_DIR TIFF_LIBRARY_RELEASE
    JPEG_INCLUDE_DIR JPEG_LIBRARY_RELEASE
    FREETYPE_INCLUDE_DIR_freetype2 FREETYPE_INCLUDE_DIR_ft2build
    FREETYPE_LIBRARY_RELEASE CMAKE_MSVC_RUNTIME_LIBRARY)
if (OSL_DEPENDENCY_BUILD_DIR)
    load_cache ("${OSL_DEPENDENCY_BUILD_DIR}" READ_WITH_PREFIX _dependency_
        CMAKE_PREFIX_PATH ${_dependency_hints})
endif ()
set (_dependency_args)
foreach (hint IN LISTS _dependency_hints)
    if (DEFINED ${hint})
        set (_value "${${hint}}")
    elseif (DEFINED _dependency_${hint})
        set (_value "${_dependency_${hint}}")
    else ()
        continue ()
    endif ()
    string (REPLACE ";" "\\;" _value "${_value}")
    list (APPEND _dependency_args "-D${hint}=${_value}")
endforeach ()

# Preserve semicolons inside the dependency-prefix argument when forwarding
# through the function's argument list.
set (_prefix_path
    "${OSL_INSTALL_PREFIX};${CMAKE_PREFIX_PATH};${_dependency_CMAKE_PREFIX_PATH}")
string (REPLACE ";" "\\;" _prefix_path "${_prefix_path}")
checked_command (configure "${CMAKE_COMMAND}"
    -S "${OSL_SOURCE_DIR}/testsuite/cmake-arnold-consumer"
    -B "${TEST_BINARY_DIR}/consumer" ${_generator} ${_dependency_args}
    "-DOSL_DIR=${_osl_dir}" "-DCMAKE_PREFIX_PATH=${_prefix_path}"
    "-DCMAKE_BUILD_TYPE=${TEST_CONFIG}")
checked_command (build "${CMAKE_COMMAND}"
    --build "${TEST_BINARY_DIR}/consumer" --config "${TEST_CONFIG}"
    --target arnold_compat_consumer oiio_compat_consumer)

get_filename_component (_cmake_bin "${CMAKE_COMMAND}" DIRECTORY)
find_program (_ctest ctest HINTS "${_cmake_bin}" REQUIRED)
if (WIN32)
    set (ENV{PATH} "${OSL_INSTALL_PREFIX}/bin;$ENV{PATH}")
endif ()
checked_command (ctest "${_ctest}"
    --test-dir "${TEST_BINARY_DIR}/consumer" -C "${TEST_CONFIG}"
    -R "^installed-(arnold|oiio)-compat$" --output-on-failure)
