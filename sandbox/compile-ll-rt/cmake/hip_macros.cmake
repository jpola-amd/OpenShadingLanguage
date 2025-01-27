

function(LLVM_COMPILE_LLC source output march mcpu filetype)
    get_filename_component(source_we ${source} NAME_WE)

    if (filetype STREQUAL "obj")
        set(extension "o")
    elseif (filetype STREQUAL "asm")
        set(extension "s")
    else()
        message(FATAL_ERROR "Unknown filetype ${filetype}")
    endif()

    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/llc_${source_we}.${filetype}")
    set(${output} ${output_file} PARENT_SCOPE)
    message(STATUS "Compiling ${source} to ${output} with llc")

    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${LLVM_LLC}
        ARGS ${source} -o ${output_file} -march=${march} -mcpu=${mcpu} -filetype=${filetype}
        DEPENDS ${source}
        COMMENT "Compiling ${source} to ${output} with llc"
        VERBATIM
    )
endfunction()

function(HIP_COMPILE_TO_BC sources headers out_bitcode out_llvm extra_options)
    get_filename_component ( source_we ${sources} NAME_WE )
    get_filename_component ( source_dir ${sources} DIRECTORY )
    get_filename_component ( source_abs ${sources} ABSOLUTE)

    set(bitcode "${CMAKE_CURRENT_BINARY_DIR}/${source_we}.bc")
    set(llvm_bc "${CMAKE_CURRENT_BINARY_DIR}/${source_we}.llvm")

    set(${out_bitcode} ${bitcode} PARENT_SCOPE)
    set(${out_llvm} ${llvm_bc} PARENT_SCOPE)

    message(STATUS "Compiling HIP source file ${sources} to LLVM bitcode ${out_bitcode}")

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
        --offload-arch=${GPU_TARGET_ARCH}
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

    message(STATUS " Bitcode: ${bitcode} ")
    message(STATUS " LLVM BC: ${llvm_bc} ")
    message(STATUS " Compiler options: ${compilerOptionsList} ")

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