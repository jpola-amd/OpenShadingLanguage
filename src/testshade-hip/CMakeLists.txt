

set (testshade-hip_srcs
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

# # print all the properties of ${HIP_LIBRARIES}
# get_property(HIP_LIBRARIES_LIST TARGET ${HIP_LIBRARIES} PROPERTY LINK_LIBRARIES)
# message(STATUS "*********HIP_LIBRARIES_LIST: ${HIP_LIBRARIES_LIST}")

add_executable(testshade-hip ${testshade-hip_srcs})
target_include_directories(testshade-hip PRIVATE ${include_dirs})
target_link_libraries(testshade-hip PRIVATE oslexec oslquery oslcomp hip::host)
target_compile_definitions(testshade-hip PRIVATE __HIP_PLATFORM_AMD__)

install (TARGETS testshade-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
