```c++
// OptiX Callables:
//  Builds three OptiX callables: an init wrapper, an entry layer wrapper,
//  and a "fused" callable that wraps both and owns the groupdata params buffer.
//
//  Clients can either call both init + entry, or use the fused callable.
//
//  The init and entry layer wrappers exist instead of keeping the underlying
//  functions as direct callables because the fused callable can't call other
//  direct callables.
//
std::vector<llvm::Function*>
BackendLLVM::build_llvm_optix_callables()
{
    std::vector<llvm::Function*> funcs;

    // Build a callable for the entry layer function
    {
        int nlayers               = group().nlayers();
        ShaderInstance* inst      = group()[nlayers - 1];
        std::string dc_entry_name = layer_function_name(group(), *inst, true);

        ll.current_function(
            ll.make_function(dc_entry_name, false,
                             ll.type_void(),  // return type
                             {
                                 llvm_type_sg_ptr(), llvm_type_groupdata_ptr(),
                                 ll.type_void_ptr(),  // userdata_base_ptr
                                 ll.type_void_ptr(),  // output_base_ptr
                                 ll.type_int(),
                                 ll.type_void_ptr(),  // interactive params
                             }));

        llvm::BasicBlock* entry_bb = ll.new_basic_block(dc_entry_name);
        ll.new_builder(entry_bb);

        llvm::Value* args[] = {
            ll.current_function_arg(0), ll.current_function_arg(1),
            ll.current_function_arg(2), ll.current_function_arg(3),
            ll.current_function_arg(4), ll.current_function_arg(5),
        };

        // Call layer
        std::string layer_name = layer_function_name(group(), *inst);
        ll.call_function(layer_name.c_str(), args);

        ll.op_return();
        ll.end_builder();

        funcs.push_back(ll.current_function());
    }

    // Build a callable for the init function
    {
        std::string dc_init_name = init_function_name(shadingsys(), group(),
                                                      true);

        ll.current_function(
            ll.make_function(dc_init_name, false,
                             ll.type_void(),  // return type
                             {
                                 llvm_type_sg_ptr(), llvm_type_groupdata_ptr(),
                                 ll.type_void_ptr(),  // userdata_base_ptr
                                 ll.type_void_ptr(),  // output_base_ptr
                                 ll.type_int(),
                                 ll.type_void_ptr(),  // interactive params
                             }));

        llvm::BasicBlock* init_bb = ll.new_basic_block(dc_init_name);
        ll.new_builder(init_bb);

        llvm::Value* args[] = {
            ll.current_function_arg(0), ll.current_function_arg(1),
            ll.current_function_arg(2), ll.current_function_arg(3),
            ll.current_function_arg(4), ll.current_function_arg(5),
        };

        // Call init
        std::string init_name = init_function_name(shadingsys(), group());
        ll.call_function(init_name.c_str(), args);

        ll.op_return();
        ll.end_builder();

        funcs.push_back(ll.current_function());
    }

    funcs.push_back(build_llvm_fused_callable());
    return funcs;
}
```

1. **Purpose**:
   - Creates three OptiX callable functions for shader execution
   - Builds entry wrapper, init wrapper, and a fused callable
   - Manages shader group execution in a GPU context

2. **Function Components**:
   - Entry layer wrapper: Wraps the main shader execution
   - Init wrapper: Wraps initialization code
   - Fused callable: Combines both init and entry functionality

3. **Key Implementation Details**:
   - Returns vector of LLVM Function pointers
   - Each callable has same signature with 6 parameters:
     - Shader globals pointer
     - Group data pointer 
     - User data base pointer
     - Output base pointer
     - Integer parameter
     - Interactive params pointer

This code is part of a shader compilation system, that generates LLVM IR for OptiX ray tracing. The code creates wrapper functions that enable shader execution on the GPU.

The function uses a builder pattern through `ll` helper object to construct LLVM IR, creating function definitions with proper signatures, basic blocks, and function calls.

Key patterns:
```cpp
// Function creation pattern
ll.current_function(ll.make_function(name, false,
                     ll.type_void(),
                     { /* parameter types */ }));

// Basic block and builder setup
llvm::BasicBlock* bb = ll.new_basic_block(name);
ll.new_builder(bb);

// Function call setup
llvm::Value* args[] = { /* function arguments */ };
ll.call_function(name.c_str(), args);
```


# HIP / ROCm

1. Structure Changes Needed:
 - Replace OptiX-specific names with HIP equivalents
 - Modify function signatures for HIP kernel calling convention
 - Add HIP kernel attributes
 - Keep LLVM IR generation but target AMD GPU
2. Implementation Steps:
 - /Create new class BackendLLVMHIP/
 - Implement HIP callable builders
 - Add ROCm-specific attributes
 - Modify parameter types for HIP ABI

```c++
lass BackendLLVMHIP : public BackendLLVM {
public:
    std::vector<llvm::Function*> build_llvm_hip_callables() {
        std::vector<llvm::Function*> funcs;

        // Build kernel for the entry layer function
        {
            int nlayers = group().nlayers();
            ShaderInstance* inst = group()[nlayers - 1];
            std::string kernel_entry_name = hip_kernel_name(group(), *inst);

            // HIP kernel attributes
            std::vector<llvm::Attribute> attrs;
            attrs.push_back(ll.create_attribute("amdgpu-flat-work-group-size", "1, 1024"));
            attrs.push_back(ll.create_attribute("amdgpu-waves-per-eu", "1"));

            ll.current_function(
                ll.make_function(kernel_entry_name, false,
                               ll.type_void(),  // return type
                               {
                                   llvm_type_sg_ptr(),         // shader globals
                                   llvm_type_groupdata_ptr(),  // group data
                                   ll.type_void_ptr(),         // userdata_ptr
                                   ll.type_void_ptr(),         // output_ptr
                                   ll.type_int(),             // thread_id
                               },
                               attrs));

            llvm::BasicBlock* entry_bb = ll.new_basic_block(kernel_entry_name);
            ll.new_builder(entry_bb);

            // Call shader implementation
            std::string impl_name = layer_function_name(group(), *inst);
            llvm::Value* args[] = {
                ll.current_function_arg(0),
                ll.current_function_arg(1),
                ll.current_function_arg(2),
                ll.current_function_arg(3),
                ll.current_function_arg(4),
            };
            ll.call_function(impl_name.c_str(), args);

            ll.op_return();
            ll.end_builder();

            funcs.push_back(ll.current_function());
        }

        // Build kernel for initialization
        {
            std::string kernel_init_name = hip_init_kernel_name(group());
            
            std::vector<llvm::Attribute> attrs;
            attrs.push_back(ll.create_attribute("amdgpu-flat-work-group-size", "1, 1"));

            ll.current_function(
                ll.make_function(kernel_init_name, false,
                               ll.type_void(),
                               {
                                   llvm_type_sg_ptr(),
                                   llvm_type_groupdata_ptr(),
                                   ll.type_void_ptr(),
                                   ll.type_void_ptr(),
                                   ll.type_int(),
                               },
                               attrs));

            // Similar implementation as entry kernel...
            // ...

            funcs.push_back(ll.current_function());
        }

        return funcs;
    }

private:
    std::string hip_kernel_name(const ShaderGroup& group, const ShaderInstance& inst) {
        return std::string("__hip_kernel_") + inst.layername();
    }

    std::string hip_init_kernel_name(const ShaderGroup& group) {
        return std::string("__hip_init_") + group.name();
    }
};
```

Key differences from OptiX version:

- Uses HIP kernel attributes for AMD GPU
- Simplified parameter list (removed interactive params)
- Added AMD-specific work group size attributes
- Changed naming convention to HIP/kernel style
- Removed fused callable (not needed in HIP model)
- Added thread_id parameter for GPU thread identification

### Attributes

```c++
std::vector<llvm::Attribute> attrs;
attrs.push_back(ll.create_attribute("amdgpu-flat-work-group-size", "1, 1024"));
```
This single attribute is sufficient for basic HIP kernel functionality while allowing the compiler to optimize wave occupancy.

#### Layer function names

```c++
layer_function_name(const ShaderGroup& group, const ShaderInstance& inst,
                    bool api)
{
    bool use_optix     = inst.shadingsys().use_optix();
    const char* prefix = use_optix && api ? "__direct_callable__" : "";
    return fmtformat("{}osl_layer_group_{}_name_{}", prefix, group.name(),
                     inst.layername());
}
```

For HIP it should be something like

```c++
std::string
layer_function_name(const ShaderGroup& group, const ShaderInstance& inst,
                   bool is_kernel = false)
{
    const char* prefix = is_kernel ? "__global__ " : "__device__ ";
    return fmtformat("{}osl_layer_group_{}_name_{}", prefix, group.name(),
                    inst.layername());
}
```

But which of the parsed functions is the kernel and which one just a device function?

## How to query the module

`llvm-nm <file.hsaco/bc>`

We can query the output of llvm both for BC and the assembly (HSACO) file 

```shell
lvm-nm build/debug/src/liboslexec/simplexnoise_hip.bc 
---------------- D _ZN11OSL_v1_14_33pvt12_GLOBAL__N_14zeroE.static.6ef3aa44047f6a18
---------------- D _ZN11OSL_v1_14_33pvt12_GLOBAL__N_17simplexE.static.6ef3aa44047f6a18
---------------- D _ZN11OSL_v1_14_33pvt12_GLOBAL__N_18grad2lutE.static.6ef3aa44047f6a18
---------------- D _ZN11OSL_v1_14_33pvt12_GLOBAL__N_18grad3lutE.static.6ef3aa44047f6a18
---------------- D _ZN11OSL_v1_14_33pvt12_GLOBAL__N_18grad4lutE.static.6ef3aa44047f6a18
---------------- T _ZN11OSL_v1_14_33pvt13simplexnoise1EfiPf
---------------- T _ZN11OSL_v1_14_33pvt13simplexnoise2EffiPfS1_
---------------- T _ZN11OSL_v1_14_33pvt13simplexnoise3EfffiPfS1_S1_
---------------- T _ZN11OSL_v1_14_33pvt13simplexnoise4EffffiPfS1_S1_S1_
---------------- W __assert_fail
---------------- W __assertfail
---------------- W __cxa_deleted_virtual
---------------- W __cxa_pure_virtual
---------------- D __hip_cuid_6ef3aa44047f6a18
---------------- t __ockl_fprintf_append_string_n
---------------- W __oclc_ABI_version
```

Here is an example of shadeops_hip.hsaco,


```shell
000000000012e944 T _ZN11OSL_v1_14_33pvt11ColorSystem10fromStringEN18OpenImageIO_v3_1_011ustringhashE
000000000012f05c T _ZN11OSL_v1_14_33pvt11ColorSystem14set_colorspaceEN18OpenImageIO_v3_1_011ustringhashE
00000000000001e0 R _ZN11OSL_v1_14_33pvt12_GLOBAL__N_112_GLOBAL__N_116cie_colour_matchE.static.2e9094b412c0dce6
00000000000005b0 R _ZN11OSL_v1_14_33pvt12_GLOBAL__N_115k_color_systemsE.static.2e9094b412c0dce6
0000000000000750 R _ZN11OSL_v1_14_33pvt12_GLOBAL__N_14zeroE.static.6ef3aa44047f6a18
0000000000000a60 R _ZN11OSL_v1_14_33pvt12_GLOBAL__N_17simplexE.static.6ef3aa44047f6a18
0000000000000760 R _ZN11OSL_v1_14_33pvt12_GLOBAL__N_18grad2lutE.static.6ef3aa44047f6a18
00000000000007a0 R _ZN11OSL_v1_14_33pvt12_GLOBAL__N_18grad3lutE.static.6ef3aa44047f6a18
0000000000000860 R _ZN11OSL_v1_14_33pvt12_GLOBAL__N_18grad4lutE.static.6ef3aa44047f6a18
00000000001604b0 T _ZN11OSL_v1_14_33pvt13simplexnoise1EfiPf
0000000000160654 T _ZN11OSL_v1_14_33pvt13simplexnoise2EffiPfS1_
0000000000160be0 T _ZN11OSL_v1_14_33pvt13simplexnoise3EfffiPfS1_S1_
0000000000161604 T _ZN11OSL_v1_14_33pvt13simplexnoise4EffffiPfS1_S1_S1_
000000000013f834 T _ZN11OSL_v1_14_33pvt5gaborERKNS_4DualIN9Imath_3_24Vec3IfEELi2EEEPKNS_11NoiseParamsE
000000000013e780 T _ZN11OSL_v1_14_33pvt5gaborERKNS_4DualIfLi2EEEPKNS_11NoiseParamsE
00000000001416e4 T _ZN11OSL_v1_14_33pvt5gaborERKNS_4DualIfLi2EEES4_PKNS_11NoiseParamsE
0000000000000040 R _ZN11OSL_v1_14_33pvt6Spline12_GLOBAL__N_19gBasisSetE.static.2c9ce11dff145b9b
000000000014606c T _ZN11OSL_v1_14_33pvt6gabor3ERKNS_4DualIN9Imath_3_24Vec3IfEELi2EEEPKNS_11NoiseParamsE
0000000000143268 T _ZN11OSL_v1_14_33pvt6gabor3ERKNS_4DualIfLi2EEEPKNS_11NoiseParamsE
000000000014ac98 T _ZN11OSL_v1_14_33pvt6gabor3ERKNS_4DualIfLi2EEES4_PKNS_11NoiseParamsE
0000000000150584 T _ZN11OSL_v1_14_33pvt6pgaborERKNS_4DualIN9Imath_3_24Vec3IfEELi2EEERKS4_PKNS_11NoiseParamsE
00000000001524f4 T _ZN11OSL_v1_14_33pvt6pgaborERKNS_4DualIfLi2EEES4_ffPKNS_11NoiseParamsE
000000000014f4cc T _ZN11OSL_v1_14_33pvt6pgaborERKNS_4DualIfLi2EEEfPKNS_11NoiseParamsE
0000000000156ec4 T _ZN11OSL_v1_14_33pvt7pgabor3ERKNS_4DualIN9Imath_3_24Vec3IfEELi2EEERKS4_PKNS_11NoiseParamsE
000000000015bc10 T _ZN11OSL_v1_14_33pvt7pgabor3ERKNS_4DualIfLi2EEES4_ffPKNS_11NoiseParamsE
00000000001540cc T _ZN11OSL_v1_14_33pvt7pgabor3ERKNS_4DualIfLi2EEEfPKNS_11NoiseParamsE

<MUCH MORE HERE>
...

```

## Manual compilation stages

1. `llvm-ir` code generation
```shell
clang++ -x hip -D__HIP_PLATFORM_AMD__ <all> -S -emit-llvm -fgpu-rdc -o output.ll <input.cpp>
```
 - "-S" - human readable llvm 

2. `llvm-as` assembly the llvm-ir into binary version of the code (bitcode) 
```shell
llvm-as output.ll -o output.bc
```

3. `opt` optional optimization step
```shell
opt -O3 output.bc -o output.opt.bc
```

4. `llc` backend compilation
```shell
llc -march=amdgcn -mcpu=gfx1030 output.opt.bc -o kernel.s
```
 - For Assembly Output: The output of llc is typically a textual GPU-specific assembly file (e.g., .s).
This needs to be assembled into a binary object file using an assembler (e.g., clang or a platform-specific assembler like ld.lld).
 - For Object Code Output: If llc directly generates an object file (e.g., .o), it is closer to being runtime-loadable. However, it may still require linking or packaging into a format compatible with the runtime (e.g., a shared object library or a binary blob).

 s human-readable. This command generates GPU-specific assembly code in a textual format for AMD GPUs (specifically, the GCN architecture for the gfx1103 target).

What You'll See in the Output
The .s file generated by this command contains low-level instructions in AMD GCN ISA (Instruction Set Architecture). It is a human-readable representation of the machine instructions that the GPU will execute.

Here is an example of what AMD GCN assembly might look like:

```c
    .text
    .amdgpu_target "gfx1103"
    .amdgpu_hsa_kernel kernel_function_name

kernel_function_name:
    s_load_dwordx4  s[0:3], s[4:5], 0x0
    v_add_u32       v0, v0, s0
    s_waitcnt       vmcnt(0)
    v_mul_f32       v0, v0, s1
    s_endpgm
```

### Key Characteristics of the Output
1. Assembly Directives: Lines starting with . (e.g., .text, .amdgpu_target) are directives providing metadata to the assembler.

2. Readable Instructions: Instructions like v_add_u32 (vector add), s_load_dwordx4 (scalar load), and s_endpgm (end program) are explicitly readable.
3. Metadata: It contains information about the target architecture, kernel names, and other attributes necessary for the assembler.

Note: If we are working with something that is a kernel then we do need a final assembly usign clang

```shell
clang -target amdgcn-amd-amdhsa -mcpu=gfx1030 kernel.s -c -o kernel.hsaco 
```
and that can be loaded by the runtime.

When working with device code compiled using the -fgpu-rdc (Relocatable Device Code) flag, the workflow changes slightly because this flag enables separate compilation for device code. Here's what happens and whether a final assembly step is necessary:

### **What Does `-fgpu-rdc` Do?**
1. **Separate Compilation**:
   - The `-fgpu-rdc` flag allows device code to be compiled into intermediate relocatable objects, which can be linked later with other object files. This is especially useful for large applications where device code is spread across multiple source files.

2. **Output**:
   - The device code is not directly embedded into the host executable. Instead, it generates intermediate `.o` files containing device-specific object code and metadata.

---

### **Do You Need Final Assembly?**
No, you do **not** need to perform a final manual assembly step after using `-fgpu-rdc`. The workflow automatically handles the necessary assembly and linking stages. Here's how it works:

1. **Compile Each Source File**:
   - The `-fgpu-rdc` flag ensures that both host and device code are compiled into `.o` files.
   - Example:
     ```bash
     clang++ -x hip -fgpu-rdc -c file1.hip -o file1.o
     clang++ -x hip -fgpu-rdc -c file2.hip -o file2.o
     ```

2. **Linking**:
   - Use the HIP/ROCm linker to combine these object files and generate the final executable or shared object.
   - Example:
     ```bash
     clang++ -x hip -fgpu-rdc file1.o file2.o -o final_executable
     ```

3. **Behind the Scenes**:
   - During the linking step, the device code is:
     - **Assembled**: The `.o` files contain device assembly code, which is converted into GPU machine code.
     - **Linked**: Relocations between device kernels or references to device symbols across files are resolved.
   - The result is a fully linked device binary (HSA code object for AMD GPUs) embedded in the host executable.

---

### **Key Differences with and without `-fgpu-rdc`**

| Feature                   | Without `-fgpu-rdc`            | With `-fgpu-rdc`              |
|---------------------------|---------------------------------|--------------------------------|
| Device Code Compilation   | Embedded directly in the host executable | Compiled into relocatable `.o` files |
| Separate Compilation      | Not supported                  | Supported                     |
| Linking                   | Simple linking of host code    | Requires linking of device code during the final step |
| Manual Assembly           | May be needed for intermediate steps | Not required; handled during final linking |

---

### **Practical Considerations**
- **When to Use `-fgpu-rdc`**:
  - Use it for modular or large projects where device code is spread across multiple files and relies on cross-file device symbols (e.g., global variables, device functions).
- **Final Assembly**:
  - The final linking step automatically assembles and links all device code. You don’t need to perform manual assembly.

---

### **Conclusion**
If you're using `-fgpu-rdc`, the final assembly is already integrated into the linking process. Simply compile each source file with `-fgpu-rdc`, and ensure that all `.o` files are linked together to produce the final executable or shared library. The toolchain will handle assembling the device code into a runtime-loadable format.
