<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright Contributors to the Open Shading Language Project. -->

Building OSL
============

OSL currently compiles and runs cleanly on Linux (x86_64), Mac OS X (x86_64
and aarch64), and Windows (x86_64). It may build and run on other platforms as
well, but we don't officially support or test other than these platforms.

Shader execution is supported on the native architectures of those x86_64 and
aarch64 platforms, a special batched 4-, 8- or 16-wide SIMD execution mode
requiring x86_64 with SSE2, AVX/AVX2 or AVX-512 instructions, as well as on
NVIDIA GPUs using Cuda+OptiX.

Dependencies
------------

OSL requires the following dependencies or tools.
NEW or CHANGED minimum dependencies since the last major release are **bold**.

* Build system: [CMake](https://cmake.org/) 3.19 or newer (tested
  through 4.2)

* A suitable C++17 compiler to build OSL itself, which may be any of:
   - GCC 9.3 or newer (tested through gcc 14)
   - Clang 5 or newer (tested through clang 23)
   - Microsoft Visual Studio 2017 or newer
   - **Intel LLVM-based icx compiler version 2022 or newer** (note: the classic `icc` compiler is no longer supported).

* [OpenImageIO](http://openimageio.org) 3.0 or newer (tested through 3.1
  and main)

    OSL uses OIIO both for its texture mapping functionality as well as
    numerous utility classes.  If you are integrating OSL into an existing
    renderer, you may use your own favorite texturing system rather than
    OpenImageIO with a little minor surgery.  There are only a few places
    where OIIO texturing calls are made, and they could easily be bypassed.
    But it is probably not possible to remove OIIO completely as a
    dependency, since we so heavily rely on a number of other utility classes
    that it provides (for which there was no point reinventing redundantly
    for OSL).

    After building OpenImageIO, if you don't have it installed in a
    "standard" place (like /usr/include), you should set the environment
    variable `$OpenImageIO_ROOT` to point to the compiled distribution, and
    then OSL's build scripts will be able to find it. You should also have
    $OpenImageIO_ROOT/lib to be in your LD_LIBRARY_PATH (or
    DYLD_LIBRARY_PATH on OS X).

* [LLVM](http://www.llvm.org) **14.0 or newer**, 15, 16, 17, 18, 19, 20, 21,
  22, 23, including clang libraries.

* (optional) For GPU rendering on NVIDIA GPUs:
    * [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) 7.0 or higher.
    * [Cuda](https://developer.nvidia.com/cuda-downloads) 9.0 or higher. It is
      recommended that you use 11.0 or higher.

* (optional) For experimental AMD GPU build infrastructure: HART and a ROCm
  development installation with HIP and clang. See [HART/ROCm](#hartrocm)
  below. Dependency discovery alone does not enable AMD shader execution.

* [Imath](https://github.com/AcademySoftwareFoundation/Imath) 3.1 or newer.
* [Flex](https://github.com/westes/flex) 2.5.35 or newer and
  [GNU Bison](https://www.gnu.org/software/bison/) 2.7 or newer.
  Note that on some MacOS/xcode releases, the system-installed Bison is too
  old, and it's better to install a newer Bison (via Homebrew is one way to
  do this easily).
* [PugiXML](http://pugixml.org/) >= 1.8 (we have tested through 1.16).
* (optional) [Partio](https://www.disneyanimation.com/technology/partio.html)
  If it is not found at build time, the OSL `pointcloud` functions will not
  be operative.
* (optional) Python: If you are building the Python bindings or running the
  testsuite:
    * **Python >= 3.9** (tested through 3.14)
    * NumPy (tested through 2.4)
    * A binding framework, depending on `OSL_PYTHON_BINDINGS_BACKEND` (see
      [Python binding backends](#python-binding-backends) below):
        * pybind11 >= 2.7 (tested through 3.0) -- needed for the `pybind11`
          backend and for `both`. It is the auto-selected default when
          OpenImageIO is older than 3.2 or Python is older than 3.10.
        * nanobind >= 2.8.0 (tested through 3.0), with Python >= 3.10 -- needed
          for the `nanobind` backend and for `both`. It is the auto-selected
          default when OpenImageIO is 3.2 or newer and Python is 3.10 or newer.
          Usually installed as a Python package (`pip install nanobind`, or
          `brew install nanobind`), which is enough: the build locates it by
          asking the interpreter. If it is not installed, the build fetches and
          builds it locally (it is a small header/CMake package, not a
          compiled library).
* (optional) Qt5 >= 5.6 or Qt6 (tested Qt5 through 5.15 and Qt6 through 6.10).
  If not found at build time, the `osltoy` application will be disabled.



Build process
-------------

Here are the steps to check out, build, and test the OSL distribution:

1. Install and build dependencies.

2. Check out a copy of the source code from the Git repository:

        git clone https://github.com/AcademySoftwareFoundation/OpenShadingLanguage.git osl

3. Change to the distribution directory and 'make'

        cd osl
        make

   Note: OSL uses 'CMake' for its cross-platform build system.  But for
   simplicity, we have made a "make wrapper" around it, so that by just
   typing 'make' everything will build.  Type 'make help' for other
   options, and note that 'make nuke' will blow everything away for the
   freshest possible compile.

   You can also ignore the top level Makefile wrapper, and instead use
   CMake directly:

       cmake -B build -S .
       cmake --build build --target install

   NOTE: If the build breaks due to compiler warnings which have been elevated
   to errors, you can try "make clean" followed by "make STOP_ON_WARNING=0",
   or if using cmake directly, add `-DSTOP_ON_WARNING=0` to the cmake
   configuration command. That will create a build that will only stop for
   full errors, not warnings.

4. After compilation, you'll end up with a full OSL distribution in
   dist/

5. Add the "dist/bin" to your `$PATH`, and "dist/lib" to your
   `$LD_LIBRARY_PATH` (or `$DYLD_LIBRARY_PATH` on MacOS), or copy the contents
   of those files to appropriate directories.  Public include files
   (those needed when building applications that incorporate OSL)
   can be found in "dist/include", and documentation can be found
   in "dist/share/doc".

6. After building (and setting your library path), you can run the
   test suite with:

        make test

CUDA/OptiX on Windows
--------------------

Enable both `OSL_USE_OPTIX` and `USE_LLVM_BITCODE` when configuring a GPU
build. `USE_LLVM_BITCODE` defaults to `OFF` on Windows, but is required for
OptiX.

With CUDA 12.9 and fmt 12, nvcc's PTX compilation can fail with
`Unicode support requires compiling with /utf-8`. Forwarding `/utf-8` to
MSVC with `-Xcompiler=/utf-8` does not fix nvcc's device-side encoding
check. Add this option to the CMake configuration command as a workaround:

    "-DOSL_EXTRA_NVCC_ARGS=-DFMT_UNICODE=0"

This opts out of fmt's Unicode requirement only for the nvcc PTX commands;
it does not change the ordinary host C++ compilation. Reconfigure before
rebuilding. The CUDA link setup also omits the Unix libraries `dl` and `rt`
on Windows; no Windows replacements for these libraries are required.

Clang-generated CUDA bitcode uses `-fgpu-rdc` because OSL links multiple
device translation units. This also gives host-referenced device constants
distinct names with LLVM 23 when defined separately in each translation
unit, avoiding duplicate definitions when linking the shadeops bitcode.

When running the executables, ensure that the CUDA toolkit's `bin` directory
and any other dependency DLL directories are on `PATH`.

HART/ROCm
---------

`OSL_USE_HART=ON` opts into experimental HART/ROCm build infrastructure.
It defaults to `OFF`, so CPU-only and CUDA/OptiX builds do not need either
SDK. With `USE_LLVM_BITCODE=ON`, it also compiles the GPU shadeops to
architecture-specific AMDGCN LLVM bitcode using **direct Clang `-x hip`**,
not hipcc, and embeds each architecture's linked module in `liboslexec`.
It does not yet generate shader bundles, install standalone AMD bitcode
files, or implement an OSL HART execution backend.
It does not require `USE_LLVM_BITCODE` just to discover the dependencies.

Add these options to your normal CMake configuration command (in addition
to your existing host dependency and `LLVM_DIRECTORY`/`LLVM_ROOT` settings):

```powershell
 -DOSL_USE_HART=ON `
 -DUSE_LLVM_BITCODE=ON `
 "-DHART_ROOT=D:\hart-repos\hart-radeon-pro\install" `
 "-DROCM_ROOT=D:\opt\rocm\therock-dist-windows-gfx120X-all-7.14.0rc3" `
 "-DHART_TARGET_ARCHITECTURES=gfx1201;gfx1100;gfx1151"
```

`HART_ROOT` and `ROCM_ROOT` may also be supplied as environment variables;
`ROCM_PATH` is a fallback for the latter. Standard CMake package locations
(`amd.hart_DIR`, `hip_DIR`, or `CMAKE_PREFIX_PATH`) can also be used, provided
all transitive packages are discoverable. Prefer the root hints for
co-installed dependencies: `amd.hart_DIR` alone does not locate its sibling
shader compiler/runtime packages. Discovery uses the SDKs' own `amd.hart`
and `hip` config packages rather than guessing library filenames.
The imported `amd::hart` and `hip::host` targets carry the host link
requirements; `osl_hart_target(target)` applies them privately to a future
HART consumer. No existing OSL target is linked to HART yet.

`HART_TARGET_ARCHITECTURES` is a **semicolon-separated list**, not a single
CUDA-style architecture. Its default is `gfx1201;gfx1100;gfx1151`; a nonempty
subset of those targets may be selected. These are the initial OSL target
policy, not a guarantee of runtime support from every ROCm/HART release.
Verify the selected SDK and driver support the target GPUs. Compilation
produces separate bitcode for each selected architecture; code-object
generation and packaging those images in a bundle remain future work.
Unlike PTX, an AMD GPU machine-code image is not
a portable intermediate representation for all GPU architectures. No PTX
output or PTX installation path is reused for HART.

### Compiling the shadeops

The `osl_hart_bitcode` target is included in the default build when both
HART and LLVM bitcode are enabled. Like the existing CUDA shadeops, this
initial port requires `USE_FAST_MATH=ON` (the default). It can also be built
independently:

```powershell
cmake --build build --config Release --target osl_hart_bitcode
ctest --test-dir build -C Release -R hart-shadeops-bitcode --output-on-failure
```

HART keeps OSL's fast math implementations but enables the compiler's
optimizations individually:

```text
-fapprox-func -fno-math-errno -fassociative-math -freciprocal-math
-fno-signed-zeros -fno-trapping-math -fno-rounding-math -ffp-contract=fast
```

It deliberately omits `-fno-honor-infinities`, `-fno-honor-nans`, and
`-ffinite-math-only`, so `isinf`, `isnan`, `isfinite`, and non-finite-result
guards remain meaningful. Do not replace this list with `-ffast-math` plus
overrides: the tested ROCm LLVM 23 driver still selects finite-only device
libraries when that umbrella option is enabled. `__FAST_MATH__` is not
defined by this flag set; `OSL_FAST_MATH=1` still selects OSL's approximate
math implementations. Reassociation, approximate functions, and signed-zero
relaxation remain enabled, so this is not a strict IEEE floating-point mode.
The bitcode regression constant-folds calls with finite values, infinities,
and NaNs into the actual shadeops and checks the results without GPU execution.

The ten GPU shadeops translation units are compiled separately for each
architecture, then linked and optimized with OSL's `llvm-link` and `opt`.
Outputs are `build/src/liboslexec/hart/<arch>/shadeops_hart.bc`.
Per-source intermediates use `<source>_shadeops_hart_<arch>.bc` in the same
directory, so their names remain stable when the source list is reordered.
Sources passed to `HART_SHADEOPS_COMPILE` must have unique basenames.
Modules from different architectures are never linked together.
Building `oslexec` also serializes each linked module to `shadeops_hart.bc.cpp`
alongside its bitcode and embeds it in the library, using the same serializer
as CUDA. A small generated registry supplies the internal
`OSL::pvt::hart_shadeops_bitcode(arch, errhandler)` lookup. It returns a
read-only view of the embedded bytes for an exact architecture match, or
reports an error and returns an empty view when that target was not built.
There is no bundle parser, runtime file loading, or implicit architecture
fallback, and embedding adds no HART/HIP runtime linkage.

The `hart-embedded-bitcode-*` tests compare these views with the original
bitcode files and check unavailable-target errors. These are host-only tests,
but a CUDA-enabled `oslexec` still requires its CUDA DLLs; use a build with
`OSL_USE_OPTIX=OFF` to run them on machines without those runtime dependencies.

### Testing external HART device code

`testshade --hart` is an experimental grid runner for externally compiled
AMDGPU bitcode. It does **not** compile or execute OSL shaders yet, and does
not change `liboslexec`'s LLVM-IR generation. Its purpose is to validate the
HIP/HART execution path before connecting the OSL AMDGPU code generator.
OSL shader arguments and unsupported options are errors, not CPU fallbacks.

Build with `OSL_USE_HART=ON`; use `OSL_USE_OPTIX=OFF` on machines without the
NVIDIA runtime. Both backends may be compiled into the same executable, but
their SDK headers and compatibility aliases are isolated in separate source
files. The HART runner uses HIP allocations, transfers, and streams directly.
HART's compatibility aliases are used for the OptiX-shaped pipeline API.
Unlike CUDA's integer `CUdeviceptr`, the HIP/HART device addresses are pointers.

When LLVM bitcode is enabled, the `testshade_hart_bitcode` target builds a
small example for each configured architecture. For a `gfx1201` device:

```powershell
cmake --build build --config Release --target testshade
.\build\bin\Release\testshade.exe --hart `
  --hart-module .\build\src\testshade\hart\gfx1201\grid_smoke_hart_gfx1201.bc `
  -g 3 2 --print
```

The example writes `(u, v, u+v)` for each pixel, with grid endpoints at 0 and
1 (or 0.5 for a singleton dimension). Choose the module matching your GPU.
The runner reads the module's actual `target-cpu` attributes, not its
filename, and rejects architecture mismatches before creating a HART context
or pipeline. For example, `gfx1100` bitcode is rejected on a `gfx1201` device,
even though HART's final compilation log names the detected GPU. Changing
the final compilation target does not make architecture-specific input IR
portable. Mixed-architecture and non-AMDGPU modules are also rejected.
Modules without `target-cpu` attributes may still be compiled by HART for
the selected device; their authors remain responsible for target-independent
IR and device-library compatibility.
Use `--hart-device INDEX` to select a HIP device and `-v` for device and HART
compiler diagnostics. On Windows, the build places `amd.hart.dll`, its shader
compiler and stack-runtime DLLs, and `hart-native-codegen.exe` beside the
executable, together with the selected SDK's HIP runtime, HIPRTC (including
builtins), COMGR, and (when provided) ROCm kpack DLLs. Installation deploys
these files to the executable directory as well. HART locates its native
codegen worker relative to its DLL, not through `PATH`.
Windows searches `System32` before `PATH`, and
display-driver copies there can be incompatible with the selected SDK.
Merely adding the SDK to `PATH` does not override those copies.

The initial external-module contract is intentionally small:

- Raw, unbundled AMDGPU LLVM bitcode with a nullary raygen entry named
  `__raygen__testshade`, or the name supplied by `--hart-entry NAME`.
- An external constant launch-parameter symbol `testshade_hart_params` with
  the layout in [hartgridparams.h](src/testshade/hartgridparams.h): one
  64-bit `float*` pointing to the device output buffer.
- Write three consecutive floats per pixel at
  `3 * (launch_index.y * launch_width + launch_index.x)`. The single RGB
  output is called `Cout`. The buffer initially contains NaNs to expose
  unwritten pixels.
- The module must contain its required device definitions; additional
  bitcode libraries can be linked beforehand with the matching `llvm-link`.
  No acceleration structure, miss program, or hit program is required.

The supported options are `--hart-module`, `--hart-entry`, `--hart-device`,
`--res`/`-g`, `--iters`, `--warmup`, `--print`, `-v`/`--debug`, and
`-o Cout FILE`. As in ordinary testshade, `--print` suppresses image writing
and the filename `null` suppresses it as well. Image output is linear RGB
float data; no display color conversion is performed.

Compile external sources with matching Clang using `-x hip`, the HART
include directory, and the device-bitcode flags below. Textual LLVM IR must
first be assembled with the matching `llvm-as`; raw text IR, PTX, and HIP
code objects are not accepted by this input mode. HART module creation takes
bitcode; pipeline creation compiles/links it into an HSACO and loads that
with HIP's module API. The runner does not duplicate HART's compiler or
load bitcode directly through `hipModuleLoadData`.

`hart-grid-bitcode` verifies the example modules without GPU execution.
`hart-grid-cli` checks validation and error reporting without initializing
a GPU. To enable `hart-grid-runtime` during configuration, set the environment
variable `TESTSUITE_HART=1`. That test executes the example for the **first**
configured HART architecture on HIP device 0; those must match. It checks
numeric results, rectangular grids, repeated launches, image output, and
runtime errors. With `--llvm-opt`, the test also verifies malformed,
non-AMDGPU, mixed-target, and device-mismatched bitcode rejection, including
misleading filenames. It can also be run directly:

```powershell
python .\testsuite\cmake-hart\check-grid.py .\build\bin\Release\testshade.exe `
  --module .\build\src\testshade\hart\gfx1201\grid_smoke_hart_gfx1201.bc
```

### Device compilation details

`MAKE_HART_BITCODE` implements compilation and `HART_SHADEOPS_COMPILE`
implements this per-architecture pipeline. The compiler uses:

```text
-x hip --offload-device-only --no-gpu-bundle-output -fgpu-rdc
--offload-arch=<arch>
--hip-path=<selected HIP SDK>
--rocm-device-lib-path=<ROCM_DEVICE_LIB_PATH>
```

These are raw LLVM bitcode files, not HIP fat binaries or executable code
objects. Relocatable-device compilation preserves device entry points and
translation-unit identity for internal device globals during LLVM linking.
The regression verifies AMDGCN triples, architecture attributes, and actual
`osl_*` function definitions; it does not execute GPU code.

`ROCM_DEVICE_LIB_PATH` is discovered under the selected HIP installation,
including the `lib/llvm/amdgcn/bitcode`, `llvm/amdgcn/bitcode`, and
`amdgcn/bitcode` layouts. Override it with a CMake cache setting or environment
variable (the cache takes precedence), for example:

```powershell
 "-DROCM_DEVICE_LIB_PATH:PATH=D:\opt\rocm\therock-dist-windows-gfx120X-all-7.14.0rc3\lib\llvm\amdgcn\bitcode"
```

Discovery checks the core libraries and each selected architecture's ISA
library. It does not silently substitute a different SDK or omit device
libraries when compilation fails. Header and device-library changes are
tracked as build dependencies; automatic transitive header tracking requires
Ninja or CMake 3.21+ for the other generators.

The OpenImageIO headers must support genuine HIP host/device annotations,
device math/hash functions, and exclusion of CPU SIMD in device compilation.
Stock OpenImageIO 3.1.14.0 needs the companion HIP header port. OSL does not
pretend HIP is CUDA by defining CUDA compiler macros. Existing renderer-side
GPU callbacks still need a HART implementation; compiling this bitcode alone
does not provide texture services, closure allocation, or shader execution.

#### Applying the OpenImageIO patch after a fresh vcpkg install

The [OpenImageIO 3.1.14.0 HIP patch](src/build-scripts/OpenImageIO-3.1.14.0-hip.patch)
contains the seven required header changes. It targets **3.1.14.0**; do not
force it onto another version if the check fails. From the OSL source root,
run the [PowerShell helper](src/build-scripts/apply-openimageio-hip-patch.ps1)
with your vcpkg installation's triplet directory:

```powershell
.\src\build-scripts\apply-openimageio-hip-patch.ps1 `
    -Prefix "D:\OSL\dependencies\x64-windows"
```

The helper requires Git and PowerShell 5.1 or newer. The default prefix is
`D:\OSL\dependencies\x64-windows`. Add `-CheckOnly` for a dry run.
It checks the installed version and patch applicability, verifies the result,
and leaves an already-patched installation unchanged. It resolves the patch
relative to the script, so it can be invoked from any working directory.

The script uses `git apply -p2` to map the patch's source paths to
`include/OpenImageIO` in the installed triplet.
The explicit work tree and `--no-index` avoid accidentally skipping
the patch when the install directory is nested inside another Git checkout;
`core.autocrlf=false` preserves the installed headers' LF line endings.
For an OpenImageIO source checkout instead, apply this same patch from its
root with `git apply --check` followed by `git apply` (the default `-p1`).

No OpenImageIO library rebuild is required for this header-only port. Rebuild
OSL's `osl_hart_bitcode` target afterward. A vcpkg reinstall or binary-cache
restore can overwrite these edits, so reapply the patch after installation;
use a patched overlay port if the change must be part of the package build.

### Keep OSL LLVM and ROCm LLVM separate

OSL still discovers and links its own LLVM/Clang libraries through
`LLVM_DIRECTORY`/`LLVM_ROOT`. That LLVM build must include `AMDGPU` for
`OSL_USE_HART`; keep `X86` for x64 CPU execution and `NVPTX` if also enabling
CUDA/OptiX. ROCm's device compiler is discovered separately as
`ROCM_CLANG_EXECUTABLE` for diagnostics. **Shadeops compilation uses
`LLVM_BC_GENERATOR` (Clang from OSL's selected LLVM installation)**, not this
separately discovered SDK compiler. HART discovery does not change OSL's
host compiler, override selected `LLVM_*` tools/libraries, or enable CMake's
HIP language. Missing bitcode tools are found only in OSL's selected LLVM
installation; their versions must match its LLVM library. Clear stale
`LLVM_BC_GENERATOR`, `LLVM_LINK_TOOL`, and `LLVM_OPT_TOOL` cache entries when
changing LLVM installations. HART and OptiX can be enabled together.

For example, OSL LLVM 20.1.8 and ROCm clang 23.0.0git are **not assumed
bitcode compatible**. LLVM's ability to read older bitcode does not promise that
OSL's older LLVM can read newer ROCm bitcode, or that AMD-specific intrinsics,
data layouts, calling conventions, and device libraries agree even with
matching version numbers. Likewise, do not mix LLVM C++ objects/libraries
from the two installations in one OSL link.

Use a ROCm-compatible LLVM/Clang build matching the SDK's device libraries
as OSL's LLVM installation. The matched ROCm LLVM 23 build is the tested
producer/linker/optimizer path for the ROCm 7.14 SDK above. Version checks
catch stale tools but cannot establish compatibility between different
forks or revisions with identical version numbers. Runtime linking with HART
still needs separate ABI validation. Do not link the SDK's LLVM C++
libraries into OSL alongside its own LLVM.

Python binding backends
-----------------------

OSL's Python bindings (the `oslquery` module, wrapping `OSLQuery`) can be
built with either [pybind11](https://github.com/pybind/pybind11) or
[nanobind](https://github.com/wjakob/nanobind). Both are generated from one
set of sources and expose exactly the same Python API; which one you get is a
build-time choice:

    cmake -B build -S .                                          # auto (see below)
    cmake -B build -S . -DOSL_PYTHON_BINDINGS_BACKEND=nanobind
    cmake -B build -S . -DOSL_PYTHON_BINDINGS_BACKEND=pybind11
    cmake -B build -S . -DOSL_PYTHON_BINDINGS_BACKEND=both

or equivalently by setting an environment variable of the same name.

When `OSL_PYTHON_BINDINGS_BACKEND` is left unset, OSL auto-selects `nanobind`
when both of these hold, and `pybind11` otherwise:

* OpenImageIO is 3.2 or newer. Reading `OSLQuery.Parameter.type` (see below)
  needs OSL's and OpenImageIO's Python modules to have been built with the
  same binding framework, and OpenImageIO switched its own default to nanobind
  in 3.2; matching it keeps `type` working out of the box.
* Python is 3.10 or newer (nanobind's minimum).

nanobind does not have to be installed for this -- if it is missing the build
fetches and builds it locally. If that local build is not possible in your
environment, configure with `-DOSL_PYTHON_BINDINGS_BACKEND=pybind11`.

With `pybind11` or `nanobind`, you get a single `oslquery` module installed in
the usual place, and it makes no difference to Python code which one it is.
With `both`, the pybind11 module keeps the ordinary location and the nanobind
one is installed alongside it under a `nanobind/` subdirectory of the
site-packages directory; put that subdirectory on `PYTHONPATH` to import it
instead. `both` exists so that the testsuite can run against each backend and
confirm they agree; it is not intended for deployment.

Why this is a choice at all: `OSLQuery.Parameter.type` returns an OpenImageIO
`TypeDesc`, and reading that attribute only works if OpenImageIO's own Python
module has been imported *and* was built with the same binding framework as
OSL's. (Each framework keeps its own registry of bound C++ types, and they
cannot see each other's.) So if you use that attribute, build OSL's bindings
to match whatever OpenImageIO you are pairing them with. Otherwise the
attribute raises `TypeError`.

Everything else in the module is free of that constraint, and
`Parameter.type_name` -- a plain string such as `"color"` or `"float[4]"` --
gives you the same information with no coupling to OpenImageIO at all. Prefer
it. `type` is retained for backward compatibility.

Conda Environment
-----------------

To simplify installation of Python and other dependencies, you can use
the provided Conda environment setup script located at `src/build-scripts/` 
by running:

    source src/build-scripts/configure_conda_env.bash

**This script will:**
  * Check for Miniconda installation.
  * Create a Conda environment named `osl-env` if it doesn't exist.
  * Install all required dependencies into the environment.
  * Activate the environment for the current shell session.

After running this script, the `osl-env` environment will be created, and
all you need to do when opening a new shell session is simply activate the 
Conda environment.

**When to use it:**  
Run this script after cloning the repository and before building OSL. It 
sets up a consistent development environment without manually installing 
all dependencies. If you already have all required dependencies installed, 
running it is optional.

Troubleshooting
----------------

- [Build issues on macOS Catalina (fatal error: 'wchar.h' file not found)](https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/issues/1055#issuecomment-581920327)
