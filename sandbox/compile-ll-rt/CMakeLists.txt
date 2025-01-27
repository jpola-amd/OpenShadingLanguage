cmake_minimum_required(VERSION 3.15)

project(example-hip CXX)

set(CMAKE_CXX_STANDARD 17)

if (NOT GPU_TARGET_ARCH)
    set(GPU_TARGET_ARCH "gfx1036")
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

find_package(LLVM 19.1 REQUIRED CONFIG HINTS ${LLVM_HOME})
find_package(LLD 19.1 REQUIRED CONFIG HINTS ${LLVM_HOME} )

find_package(HIP REQUIRED HINTS $ENV{HIP_PATH})

# how to use the llvm package
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found LLD ${LLD_VERSION}")
message(STATUS "Found HIP ${HIP_VERSION}")

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
message(STATUS "LLVM_TOOLS_BINARY_DIR: ${LLVM_TOOLS_BINARY_DIR}")

message(STATUS "LLD_LIBRARIES ${LLD_EXPORTED_TARGETS}")
message(STATUS "LLD_INCLUDE_DIRS ${LLD_INCLUDE_DIRS}")

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
find_llvm_program(LLVM_LLC "llc" "LLVM static compiler")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(hip_macros)


set(ir_sources 
    "${CMAKE_CURRENT_SOURCE_DIR}/test-ll/test.ll"
)

# compile the IR files to asm and obj to compare later with the output of the code
foreach(ir ${ir_sources})
    LLVM_COMPILE_LLC(${ir} output_asm "amdgcn" "${GPU_TARGET_ARCH}" "asm")
    LLVM_COMPILE_LLC(${ir} output_obj "amdgcn" "${GPU_TARGET_ARCH}" "obj")
    list(APPEND llc_outputs ${output_asm})
    list(APPEND llc_outputs ${output_obj})

endforeach()


add_custom_target(amdgcn_outputs ALL 
    DEPENDS ${llc_outputs}
)

set(sources
    main.cpp
)

llvm_map_components_to_libnames(llvm_libs 
    AllTargetsAsmParsers
    AllTargetsCodeGens
    AllTargetsDescs
    AllTargetsInfos
    Analysis
    AsmPrinter
    CodeGen
    CodeGenTypes
    Core
    IRPrinter
    IRReader
    MC
    MIRParser
    Passes
    Remarks
    ScalarOpts
    SelectionDAG
    Support
    Target
    TargetParser
    TransformUtils
    Vectorize
    Option
    Demangle
)

add_executable(example ${sources})
target_include_directories(example PRIVATE ${LLVM_INCLUDE_DIRS} ${LLD_INCLUDE_DIRS})
target_link_libraries(example PRIVATE ${llvm_libs} ${LLD_EXPORTED_TARGETS})
add_dependencies(example amdgcn_outputs)


add_custom_command(TARGET example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/test-ll
    ${CMAKE_BINARY_DIR}/test-ll
)