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
It also enables a numeric multi-layer OSL HART execution path in
`testshade`. It does not generate shader bundles or install standalone
AMD bitcode files.
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
HART consumer. `testshade` links to HART when this option is enabled.

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

### Running a generated OSL shader with HART

With `OSL_USE_HART=ON` and `USE_LLVM_BITCODE=ON`, `testshade --hart` accepts
an ordinary compiled OSL shader. `oslc` and the `.oso` format are unchanged:

```osl
shader hart_first(output color Cout = 0)
{
    Cout = color(u, v, u + v);
}
```

```powershell
oslc hart_first.osl
testshade --hart -g 3 2 --print hart_first
testshade --hart --hart-no-cache --warmup --iters 3 -g 37 5 `
  -o Cout first.exr hart_first
```

The frontend uses its normal shader/parameter/layer setup. It selects the
actual HIP device architecture before optimizing the shader group, then
passes verified AMDGPU bitcode and the generated init/entry names to HART.
`--inbuffer` uses the existing source-buffer compiler and memory-loaded shader
API for each named layer; it does not require or write `.oso` files. A missing
or invalid `.osl` source is an error even if a compiled file exists.
Serialized `--group` (or `-group`) specifications use the same inline-string
or file parser as CPU testshade. `--oslquery` prints the actual group
serialization, layers and parameter types before optimization; `-v` also
prints group serialization and layer names.
`--print-groupdata` reports the compiled group's logical storage size through
the same checked query as CPU testshade; this is not hardware stack usage.
HART also accepts `--options opt_groupdata=0` or `opt_groupdata=1` to control
that layout optimization. Other `--options` settings remain unsupported and
are rejected before launch. The original `groupdata-opt` fixture checks the
optimized layout and preservation of connected output parameters.
The renderer embeds a separate raygen module for every configured
architecture; execution does not read device code from build-tree paths.
Each point receives real `ShaderGlobals` and separately aligned group
storage. Generated output placement uses a checked, contiguous output arena;
the renderer does not assume offsets inside the group.

With no `-o` options, this path selects `output color Cout` on the final layer
when present. If there is no output named `Cout`, the group executes without an
output arena or image; this includes diagnostic-only and empty shaders.
Diagnostics, warmup, repeated launches and error reporting still run normally.
An explicitly requested missing output remains an error.
Explicit `-o NAME FILE` options select one or more numeric outputs:
integers, floats, triples, matrices, numeric arrays, and numeric struct fields.
Use `layer.parameter` to disambiguate layers; unqualified names select the last
matching layer. Struct fields retain their dotted parameter names. Whole
structs, strings and closures are not image outputs. As on the CPU, shader
output arrays must have a fixed size (unsized input arrays remain supported).
Generated grids can construct the supported diffuse/emission closure trees,
including closure parameters and layer connections. Groups that need closures
use a separate raygen with a 1024-byte caller-owned pool per point, shared by
all entries until the call sequence returns. Numeric-only groups retain the
pool-free path. Pool exhaustion reports a device error and publishes no image;
closure pointers are never copied into image outputs.

`--center` places `u`, `v` and `P` at pixel centers and uses `1/width` and
`1/height` UV derivatives. As in CPU testshade, `P` derivatives retain the
grid spacing `1/max(1, dimension-1)`. Without this option, border samples stay
at zero and one; a one-point axis stays at one half in either mode.

`isconstant` uses OSL's existing compile-time symbol classification, including
numeric and string operands. Uniform runtime data is not necessarily constant;
results can change when OSL optimization proves an expression constant.

Each point owns one packed record containing each distinct selected symbol,
in first-request order. Aliases of the same symbol share storage but may write
different files. Integer printing and integer image buffers retain int32
values; `-d float`, `-d half`, and `-d uint8` explicitly request file conversion.
The existing `-od` alias selects the same conversions.
Arrays and matrices become flattened image channels. Display conversion for
JPEG/GIF/PNG applies to scalar color outputs, not numeric data outputs.
`--print` suppresses all image files, and `null` suppresses individual files.

Output mappings must be selected before group optimization. A compiled group
may be rendered again with the same distinct outputs in the same order and
different filenames, but a different layout requires a new group. The renderer
checks existing mappings before allocating or launching; the shading system
rejects changes to a compiled HART group's symbol locations or renderer-output
selection. The arena belongs to the renderer and remains alive through
synchronized readback and image writing. The six-argument callable ABI and
the separate external-module RGB contract are unchanged.

Generated HART mode also accepts ordered `--entry LAYER` options. Initialization
runs once, followed by the selected entry layers with shared per-point Groupdata.
Existing execution flags and lazy connections prevent repeated earlier layers
and shared producers from running again. As with CPU execution, the last layer
does not have an already-run guard. `--entryoutput layer.parameter` options
override the execution order by selecting parameters in the declared entry
layers; they require `--entry`. An unselected layer's output arena stays zero.
Split, fused-scratch and fused-local modes share these semantics.

At the API level, set the group's `entry_layers` before optimization. This
declares entry points and supplies their default call order. The HART-specific
`hart_entry_layers` attribute can select or reorder these entries before code
generation; `num_hart_entry_layers` and `hart_entry_layers` report the effective
sequence without optimizing. A compiled sequence is immutable. The init, entry,
and fused exports keep their six-argument ABI; entry/fused wrappers execute the
whole selected sequence, rather than initializing separately for each layer.

A connected two-layer example is:

```osl
shader hart_group_producer(float scale = 1, output float value = 0)
{
    value = scale * (u + v);
}

// Compile each shader in its own .osl file.
shader hart_group_consumer(float value = 42, output color Cout = 0)
{
    Cout = color(u, v, sin(value));
}
```

```powershell
oslc hart_group_producer.osl
oslc hart_group_consumer.osl
testshade --hart --hart-no-cache --warmup --iters 3 -g 3 2 --print `
  --shader hart_group_producer producer `
  --shader hart_group_consumer consumer `
  --connect producer value consumer value
```

The producer is an internal LLVM function, not a separate HART callable.
OSL's existing layer scheduling and connection copies share the per-point
group storage. HART exports init, final-entry and fused init+entry wrappers,
not a callable per layer. Output placement is qualified to the final layer, so producer
outputs remain internal. Both layers are validated before optimization,
including unused layers and parameters.

Generated mode defaults to two direct calls (init then entry) within one GPU
launch. `--hart-fused` selects one callable that invokes the same internal
init and entry functions, still using renderer-supplied per-point group
storage. Both paths keep the six-argument callable ABI and lazy layer
execution. This switch is not supported for external modules; their ABI and
two fixed callable entries remain unchanged. The fused path is opt-in, not
an assumed performance improvement.

`--hart-fused --hart-local-groupdata BYTES` additionally permits private
group storage inside the fused callable when the complete group fits the
nonnegative byte budget. Zero (the default) disables this allocation;
larger groups retain renderer scratch, without truncating their storage.
The threshold includes equality. Split callables always use caller storage.
Private storage uses the AMDGPU target's alignment and a private-to-generic
pointer conversion. The generated and external launch-parameter layouts are
unchanged. Verbose output reports logical group size, alignment, selected
local bytes and renderer scratch bytes (zero in local mode).

Renderer integrations set `max_hart_groupdata_alloc` before compilation and
query `hart_groupdata_alloc` on the successfully compiled group. The latter
is the logical allocation selected for its fused wrapper, not final hardware
stack usage; subsequent attribute changes do not rewrite compiled groups.
Only a renderer selecting that fused wrapper may omit its scratch buffer.
Private storage may become registers or spill, so a larger budget is not
necessarily faster.

For generated groups, `--runstats` separates OSL group preparation, pipeline
construction, warmup and synchronized launch latency. Pipeline construction
includes module/program-group creation, linking, deferred compilation triggered
by stack queries, and SBT setup. Earlier reporting stopped before the stack
query, so those older pipeline measurements are not directly comparable.
Group preparation includes optimization and compiled-artifact extraction, not
the separate `oslc` source-compilation process.
Use `--warmup --iters N` to exclude one warmup launch and average the following
N launches. Warmup includes its clears and error checks. The launch timer includes
HART submission and waiting for GPU completion, but excludes output clears,
resource setup, compilation, error-buffer readback and image copies. This
is not a pure GPU kernel time. The tested Windows ROCm SDK returned negative
HIP event intervals even after event synchronization, so event timing is
not used.

The stack estimates come from HART's program-group stack query: raygen bytes
and the maximum direct-callable bytes among the selected callable records.
Callable estimates may be conservative floors; they are not measured VGPR,
spill, or total per-thread physical stack usage. The current public API does
not expose those hardware metrics.

The manual comparison uses a nine-layer numeric chain, a connected textured
material and an eight-lobe closure-construction shader. It compares full float
images with CPU references and across split, fused-scratch and fused-local
modes:

```powershell
python .\testsuite\cmake-hart\check-generated.py `
    .\build\hart-validation\bin\Release\testshade.exe `
    --oslc .\build\hart-validation\bin\Release\oslc.exe --gpu --fused-benchmark
```

For each workload/mode it runs a single cache-disabled sample, a separate
cache-enabled priming process, then three trials in rotated mode order.
Trials use a 256x256 grid, one warmup and 100 measured launches per process.
JSON rows retain individual trials, medians and ranges for source-compiler
subprocess wall time, in-process group preparation, full pipeline construction,
warmup, synchronized launch latency and process wall time. OSLC wall time
includes process startup, not just parsing. Observed cache-hit keys are separate
from the cache-enabled policy; disabling the HART pipeline cache does not flush
OS or driver caches. Logical storage and SDK stack estimates are also recorded.
There is no timing-based pass/fail threshold or automatic mode selection;
compare on the intended GPU and workload.

Numeric comparisons and `if`/`else` can also use connected inputs
conditionally. For example, replace the consumer with:

```osl
shader hart_branch(float value = 42, float bias = 0, output color Cout = 0)
{
    if (u > v + bias)
        Cout = color(u, v, sin(value));
    else
        Cout = color(u, v, 0);
}
```

```powershell
oslc hart_branch.osl
testshade --hart --llvm_opt 10 --warmup --iters 3 -g 37 5 --print `
  --shader hart_group_producer producer `
  --shader hart_branch consumer `
  --connect producer value consumer value
```

With the default bias, points where `u > v` use the producer's value;
the other points, including `u == v`, write zero to the blue channel.
The generated code uses the existing OSL conditional and lazy-layer
lowering, without adding HART callable types or changing the group ABI.

`for` and `while` loops use the same shared LLVM lowering. A consumer can,
for example, accumulate a shadeop result over a varying number of iterations:

```osl
int count = 1;
if (u > v)
    count = 4;
else if (u < v)
    count = 0;

float sum = 0;
for (int i = 0; i < count; i += 1)
    sum += sin(value + i);
Cout = color(u, v, sum);
```

The connected input can be used inside the loop and again afterward,
including when the loop executes zero times. Loop tests use deliberately
small bounds; the backend checks supported operations and types, not
termination, and does not impose an iteration limit. Shader authors remain
responsible for terminating their loops.

`do`/`while`, `break`, `continue`, helper-function early returns and shader
`exit()` use the same shared LLVM control-flow lowering. They preserve lazy
connected inputs and derivative propagation through function results and
output parameters. Logical `&&`, `||`, `!` (and `and`, `or`, `not`) retain
short-circuit evaluation. Integer bitwise `&`, `|`, `^`, `~`, shifts, signed
remainder and prefix/postfix increments and decrements are also supported.
Shift tests exercise counts 0 through 31; HART does not add semantics for
out-of-range counts.

The GPU-opt-in `hart-control-flow-runtime` test checks packed integer results
exactly, nested loop exits, early shader exits with hazardous unused producers,
and short-circuit texture calls that would otherwise raise device errors.
It covers OSL 0 / LLVM 10 and optimized split, fused and callable-local storage.
The existing `hart-loops-runtime` additionally checks loop-carried values and
zero/one/multiple iterations.

The numeric HART subset also includes trigonometric and hyperbolic functions,
logarithms, exponentials, error functions, cube root, inverse square root,
rounding, sign and IEEE finite/Inf/NaN classification. Available scalar and triple
overloads reuse OSL's existing safe/fast math and derivative implementations.
Cross products, distance, area and `calculatenormal` are supported, along with
standard-library reflection, refraction, faceforward and rotation. Area and
`calculatenormal` retain OSL's zero output-derivative convention; the latter
does not normalize the resulting normal. `hart-numeric-math-runtime` compares
full-precision values and derivatives against CPU and independent references,
including connected groups, sincos output aliasing, domain boundaries and actual
nonfinite input classification. HIP `asin`/`acos` explicitly retain OSL's domain
clamp. CPU-only approximation tolerances do not loosen GPU reference or
derivative checks. The existing shared `atan2` duals' reversed derivative signs
are preserved for backend parity, not corrected by this change.
Additional noise selectors still require explicit support.

Nonconstant HART float division uses unrelaxed LLVM division, rather than the
HIP shadeop's approximate reciprocal path, so runtime and constant operands
retain the same precision. Nonfinite quotients become positive zero, as in
OSL's safe division; finite signed zeros are retained. Derivative reciprocals
use the same guarded division. CPU, OptiX, integer and matrix division, and the
existing nonzero constant-denominator path are unchanged.

HART also supports `spline` and `splineinverse` with `catmull-rom`,
`bezier`, `bspline`, `hermite`, `linear` and `constant` bases. Selectors may be
literals, immutable input string parameters (including instance overrides),
or locals whose initialization dominates their reads and whose writes all
resolve to the same value. Dynamic, connected, output, interactive and
interpolated selectors remain unsupported. Renderers opt in
with `HARTSplineErrors` and provide `rs_hart_spline_error` in addition to array
services. Knot arrays must have at least four resolved elements; the selected
count must fit the array and the basis's segment cardinality. Statically invalid
arrays/counts are rejected before launch. Dynamic counts are checked even when
shader range checking is disabled; invalid counts or NaN inputs report a device
error before evaluation and prevent output publication. Infinite inputs retain
endpoint clamping. The existing constant basis discards derivatives. Inverse
evaluation uses OSL's bounded solver, retains knot-based endpoint clamping,
ignores knot derivatives, and can drop derivatives at solver segment boundaries.
`hart-spline-runtime` checks nonlinear bases, derivatives, connected/resized
arrays, endpoints and pre/post-launch failures. Path-tracer tests also compose
spline weights with textures and verify device failure propagation.
`hart-spline-division-runtime` additionally checks live division equality,
finite guards, signed zeros, derivatives, and default/overridden forward and
inverse spline selectors against independent values in all four dispatch and
storage modes. The original aggregate math and spline-boundary fixtures also
run through the standard HART runner.

The grid and path renderers bind color-system data through their device service
state, not host addresses or general userdata. `HARTColorSystem` renderers
provide `rs_hart_get_colorsystem` and `rs_hart_color_error`. This enables
`luminance`, `blackbody`, `wavelength_color` and supported literal built-in color
conversions: RGB/rgb, hsv/hsl, YIQ, XYZ, xyY and the current working-space name.
`transformc` also accepts linear and sRGB; these are not additional named
constructor spaces. Custom OCIO conversions and dynamic space names are not enabled.
Unsupported literal spaces are rejected before optimization, including in
unused layers. A conversion that becomes unsupported after rebinding reports a
device error and prevents output publication rather than returning an identity
transform silently.

Color-system-dependent HART expressions retain runtime data access even with
constant shader inputs. The shared resource owner refreshes the POD data and
trailing string hashes before each render, reusing its device allocation.
`hart-color-rebind-*` tests use one optimized group through A-B-A working-space
changes, check values and derivatives independently, and require actual native
pipeline cache hits. `hart-color-runtime` covers numerical conversions and
connected values; path tests compose color operations with textured spline
weights. Named color constructors, blackbody and wavelength shadeops retain
OSL's zero output-derivative convention. The existing Rec709 D65 white point
uses y=0.3291; tests derive matrices from that value, not a different standard
white point. HIP wavelength lookup returns zero outside the existing table
domain (including nonfinite inputs), without undefined float-to-index conversion;
the existing first-bin extrapolation between 375 and 380 nm is retained.
Blackbody tests use an independent double-precision Planck integral over the
same CIE sampling data. The existing interpolated lookup table is approximate;
direct HIP integration is checked more tightly than CPU `fast_expm1`.
CPU fast-math can turn a NaN temperature into a finite result, while HIP's
libm path propagates NaN; classification tests preserve that backend distinction.

`Dx` and `Dy` expose OSL's propagated first-order derivatives. For example:

```osl
float value = sin(u * v);
Cout = color(value, Dx(value), Dy(value));
```

The result is `(sin(u*v), cos(u*v)*v*dudx, cos(u*v)*u*dvdy)`.
The endpoint grid supplies `dudx = 1/max(1,width-1)` and
`dvdy = 1/max(1,height-1)`, with `dudy = dvdx = 0`, matching CPU testshade.
Singleton dimensions use coordinate 0.5 and derivative 1, not zero.
These are propagated derivatives, not finite differences between GPU threads.
The existing arithmetic and derivative-aware sine lowering also handles
color components, executed branches and loops, and connected layer inputs.
For example, a producer can compute `value = sin(u*v)` and its consumer
can read `Dx(value)` and `Dy(value)`; OSL propagates derivative requirements
upstream and copies value/dx/dy through group storage. Constant and uniform
parameter derivatives are zero. No new callable or launch ABI is needed.

Explicit matrix transforms are supported for points, vectors and normals:

```osl
matrix M = matrix(2,0,0,0, 0,0.5,0,0, 0,0,1,0, 1,2,0,1);
point p = transform(M, P);
Cout = color(p);
```

Points include translation and homogeneous division; vectors ignore translation.
Normals use the inverse transpose without automatic normalization. Matrix
construction, multiplication/division, transpose, determinant, assignment,
uniform matrix parameters and matrix connections are supported. Component
reads/writes support integer row and column indices in `[0,3]`, including
runtime indices when the renderer advertises `HARTArrayBounds`. Invalid
runtime indices report a device error and prevent image publication.
Transforms propagate the input triple's derivatives using existing OSL
shadeops. OSL does not track matrix-element derivatives, even when matrix
values vary across the grid. Singular matrices and zero homogeneous
denominators retain existing OSL behavior rather than introducing a new
inverse or projection policy. Named spaces remain a separate capability.

Literal `"common"`, `"object"` and `"shader"` coordinate spaces are supported
by spatial constructors, `transform`, `matrix`, and `getmatrix`. For example:

```osl
point p = transform("object", "shader", P);
normal n = normal("object", 0, 0, 1);
matrix M = matrix("shader", "common");
Cout = color(transform(M, p));
```

The test renderer uses the same static transforms as CPU testshade: shader
space has a 45-degree Z rotation and translation `(1,0,0)`, and object space
has a 90-degree Z rotation and translation `(0,1,0)`. Forward/inverse matrices
are uploaded per render and referenced through `ShaderGlobals`; neither
matrix values nor host addresses are baked into the compiled shader group.
The generated launch parameters change, but the external-module ABI does not.
Renderers must advertise `HARTTransforms` and supply the matrix callbacks.
Renderers advertising `HARTTransforms` without `HARTNamedTransforms` retain
this literal-only subset and reject other names before optimization. The
native path tracer currently advertises neither coordinate-service capability.

Generated testshade also advertises `HARTNamedTransforms`. Spatial
constructors, `transform`, `matrix` and `getmatrix` accept literal or runtime
scalar string names, including names selected from string arrays. Its named
transforms (normally `myspace`, with scale `(1,2,1)`) come from the actual
`SimpleRenderer::name_transform` bindings. Camera, screen, NDC and raster
inverse matrices use the current camera settings. As on `SimpleRenderer`,
those camera names do not have a forward transform unless explicitly
registered as a named transform. No forward camera transform is fabricated.

Named matrices, the current `commonspace` synonym (normally `world`), and
`unknown_coordsys_error` are bound before every render, never folded from host
renderer data into HART code. The common synonym takes precedence over a named
entry of the same name, including after A-B-A rebinding. Unknown names or
unavailable directions return false and identity matrices, matching CPU lookup
behavior; composed `getmatrix` results still include any successfully resolved
direction. When `unknown_coordsys_error` is enabled, the lookup also reports a
device error and prevents image publication. Malformed or nonfinite *queried*
bindings always report an error; unused invalid matrices do not poison a
render. Existing inverse-transpose normal and input-derivative shadeops are
reused. These bindings are static and ignore time, as the reference renderer
does; animated and nonlinear transforms are not implemented.

`HartTextureState::transforms` points to a 32-byte `HartTransformState` with
144-byte typed forward/inverse records. The texture state is now 64 bytes,
with the transform pointer at offset 56 and camera pointer still at offset 48.
The six-argument callable ABI is unchanged. The four `hart-space-rebind-*`
tests compare 33 cases at three rows against CPU execution, including
projection, point/vector/normal gradients, common-synonym changes, query misses
and error recovery. Value comparisons use `2e-6 + 2e-6*abs(CPU)`; return codes,
A-B-A artifact identity, repeated images and cache reuse are checked exactly.
Genuine HIP layout probes and compiler tests cover all configured targets.

`filterwidth` computes `sqrt(Dx(x)*Dx(x) + Dy(x)*Dy(x))` for a float input,
or the same expression component-wise for a color, point, vector, or normal.
It uses propagated derivatives, including those copied across layer
connections. Constants and inputs without derivatives have zero width.
`Dx` and `Dy` of the width itself are zero: OSL does not compute the
second-order derivatives needed to differentiate it. This exposes a
footprint for shader filtering; it does not automatically antialias a shader.
For example:

```osl
float w = filterwidth(sin(u * v));
Cout = color(w, Dx(w), Dy(w));
```

Here the red channel is `abs(cos(u*v))*sqrt((v*dudx)^2+(u*dvdy)^2)`;
green and blue are zero.

Surface geometry is also available: `P`, `N`, `Ng`, `dPdu`, `dPdv`,
`I`, and `time`, in addition to `u` and `v`. The test grid is still a synthetic
flat patch, not a ray-traced scene. It supplies `P=(u,v,1)`, `N=Ng=(0,0,1)`,
`dPdu=(1,0,0)`, and `dPdv=(0,1,0)`. Position derivatives are
`Dx(P)=(dudx,0,0)` and `Dy(P)=(0,dvdy,0)`.
Like CPU testshade, this grid supplies `I=(0,0,0)` and `time=0`, with zero
derivatives and filter widths. These are explicit synthetic-grid defaults,
not camera-ray directions or animated sampling; no new camera or time
controls are implied.

Renderers advertising `HARTGeometry` also support `surfacearea()`,
`backfacing()` and `raytype()`, and writes to `P`, `I`, `N`, `Ng`, `dPdu`,
`dPdv`, `u` and `v`. Writes and their supported derivative storage are visible
to subsequent executed layers through the existing shared shader globals.
`time`, `dtime` and `dPdtime` remain read-only; the supplied static grid and
native renderer set motion intervals and velocities to zero, without claiming
motion-blur support. `Ps` is still unsupported.

Both supplied HART renderers opt in. The grid's actual unit patch has surface
area one and is front-facing. The native path tracer supplies the hit mesh's
area, the original hit's backfacing flag, incident direction and current ray
mask. Changing a shading normal does not recalculate that captured hit flag.
Live geometry queries report their required fields through `globals_needed`.
Constant and dynamic `raytype` names use the shading system's configured
name-to-bit mapping without calling host renderer services. Unknown names
return zero; duplicate names retain first-match precedence. Configure up to
32 names before compiling groups, as for constant ray queries; changing the
mapping requires recompilation. The supplied path tracer uses the standard
ray-bit mapping, while generated testshade's `--raytype` selects the launch
mask. The opt-in `hart-geometry-state-runtime` test compares live ray masks,
shared-global writes and gradients with CPU and independent references in
split, fused, callable-local and unoptimized modes.
The four `hart-raytype-rebind-*` tests use 32 configured names, bit 31,
duplicate-name precedence and unknown/empty names on the GPU, and require
A-B-A mask updates to preserve compiled artifacts and pipeline-cache identity.

Unqualified `point`, `vector`, and `normal` constructors work in common
space, together with `dot`, `length`, and `normalize`. Construction does
not transform coordinate spaces or normalize a normal automatically.
Vector values and their derivatives can cross layer connections, using the
existing OSL storage and shadeops. For example, a producer can construct
`vector(P[0], 2*P[1], P[2]+P[0]*P[1])`; its consumer can compute `length(value)`
and extract the result's `Dx` and `Dy`. Normalizing a zero vector returns
zero, including zero derivatives, following ordinary OSL behavior.

Numeric, non-periodic Perlin noise is supported through `noise` (unsigned)
and `snoise` (signed). The coordinate forms are `noise(x)`, `noise(x,y)`,
`noise(p)`, and `noise(p,t)` for 1D through 4D, with the same forms for
`snoise`. The third form uses a point; the fourth adds a numeric coordinate
that need not be the `time` global.
Select float, color, or vector results with an explicitly typed temporary.
For example:

```osl
float n = noise(P * 3.7);
Cout = color(n, Dx(n), filterwidth(n));
```

Values and first-order derivatives can cross layer connections and feed
`Dx`, `Dy`, and `filterwidth`. Constant coordinates produce constant noise
values with zero derivatives. This is unfiltered Perlin noise: neither
derivative support nor `filterwidth` automatically antialiases it.

The procedural noise family also includes:

| Family | Non-periodic forms | Periodic forms |
| --- | --- | --- |
| Signed Perlin | `snoise(...)`, `noise("perlin",...)` | `psnoise(...)`, `pnoise("perlin",...)` |
| Unsigned Perlin | `noise(...)`, `noise("uperlin",...)` | `pnoise(...)`, `pnoise("uperlin",...)` |
| Cell | `cellnoise(...)`, `noise("cell",...)` | `pnoise("cell",...)` |
| Hash | `hashnoise(...)`, `noise("hash",...)` | `pnoise("hash",...)` |
| Signed simplex | `noise("simplex",...)` | Unsupported |
| Unsigned simplex | `noise("usimplex",...)` | Unsupported |
| Gabor | `noise("gabor",...)` | `pnoise("gabor",...)` |

These forms support 1D-4D coordinates and float/color/vector results.
Periodic calls append matching period arguments: `(x,px)`, `(x,y,px,py)`,
`(p,pp)`, or `(p,t,pp,tp)`. Period handling follows existing OSL semantics:
periods are floored to integers with a minimum of one. Period derivatives
are not propagated. Cell and hash noise have zero output derivatives,
including at discontinuities. Hash noise depends on the coordinate bit
patterns, so rounding differences in upstream arithmetic can change its
value substantially even when the coordinates are numerically close.

The selectors `"noise"` and `"snoise"` also select unsigned and signed Perlin,
respectively, including periodic calls. Literal selectors resolve to existing
device shadeops. Runtime selectors (including parameters, arrays and connected
strings) use the existing generic noise dispatch with a checked hash selector.
They require renderer support for `HARTNoiseErrors`. The seven named families
in the table plus these two aliases are accepted dynamically; periodic simplex
and options on non-Gabor noise remain unsupported. Unknown literals and malformed
option lists reject before launch. Runtime invalid selectors or non-Gabor selectors
with options record a device error, initialize the result to zero and skip
evaluation; the host rejects the render rather than publishing that zero.
Parameter specialization cannot bypass selector or option validation.

Gabor accepts five literal option names with numeric, possibly varying values:
`"anisotropic"` (int, default 0), `"do_filter"` (int, default 1),
`"direction"` (triple, default `(1,0,0)`), `"bandwidth"` (float/int, default 1),
and `"impulses"` (float/int, default 16). Anisotropy 0 is isotropic, 1 is
directional, and other integers select the existing hybrid behavior. Direction
is not normalized. Finite bandwidth and impulse values clamp to `[.01,100]`
and `[1,32]`. Filtering uses coordinate derivatives and disables itself for
tiny or degenerate footprints. Options and periods do not contribute
derivatives; 4D Gabor ignores time and its period. Scalar noise matches the
first color channel.

Gabor periods count grid cells: the world-space repeat distance on each axis
is `radius * max(1,floor(period))`, where radius depends on bandwidth. They are
not ordinary world-unit periods. HIP wrapping uses a signed-corrected remainder
so reciprocal approximation cannot change a seed at exact cell multiples.
The kernel has a hard support cutoff; analytic derivatives describe its smooth
pieces, not jumps across that cutoff.

Renderers must advertise `HARTNoiseErrors` and implement `rs_hart_noise_error`.
Both supplied HART renderers do so. Device guards reject nonfinite coordinates,
coordinate derivatives, options, used periods, derived parameters and active
filter state, and check wrapped seed cells before signed-int conversion.
Invalid arguments set error bit 1024 and prevent image publication. Finite
values outside the bandwidth/impulse clamp intervals remain valid; ignored
4D time arguments are not validated.

These operations can be composed into a tileable, multi-octave color ramp:

```osl
float value = 0, amplitude = 0.5, frequency = 1;
for (int i = 0; i < 3; i += 1) {
    value += amplitude * psnoise(point(u * frequency, v * frequency, 0.375),
                                point(frequency, frequency, 1));
    amplitude *= 0.5;
    frequency *= 2;
}
float width = filterwidth(value);
float mask = smoothstep(-0.25 - width, 0.25 + width, value);
Cout = mix(color(0.1, 0.2, 0.3), color(0.8, 0.6, 0.4), mask);
```

Here the footprint explicitly broadens the ramp transition; it does not
automatically filter the noise octaves. Numeric signals and their first-order
derivatives can also pass from a producer layer into a separate ramp layer.

Its numeric instruction subset is assignment, addition, subtraction,
multiplication, division, negation, color/point/vector/normal construction,
`sin`, `dot`, `length`, `normalize`, the noise forms listed above, component
reads/writes, comparisons (`<`, `<=`, `==`, `!=`, `>=`, `>`), `if`/`else`,
`for`/`while`, `Dx`, `Dy`, and `filterwidth`. Procedural math also supports
`abs`, `min`, `max`, `clamp`, `mix`, `step`, `smoothstep`, `floor`, `ceil`,
`fmod`, `cos`, `sqrt`, and `pow`, with the usual OSL scalar/triple overloads
(plus internal structural operations). Only reads of the listed shader
globals are supported. Other instructions are rejected before
runtime optimization, even if optimization could eliminate them.
Inlined numeric function bodies, including standard-library wrappers such
as `clamp`, undergo the same checks. Their internal function-name markers
do not enable string allocation or character operations. Early function returns
remain unsupported.
Math domain boundaries and derivatives follow existing OSL semantics,
including zero derivatives for `floor`, `ceil`, and `step`; this does not
automatically filter discontinuities.
The supported command-line subset is `--hart-device`, `--hart-no-cache`,
`--hart-fused`, `--hart-local-groupdata`, `--runstats`,
`--res`/`-g`, `--warmup`, `--iters`, `--print`, `-v`/`--debug`, `-o Cout FILE`,
`-d float|half|uint8`, `--groupname`, `--layer`, `--shader`, `--connect`,
uniform `--param`, `-O0`/`-O1`/`-O2`, and `--llvm_opt`.
`--llvm_opt 10` skips OSL's LLVM passes, allowing inspection of branches and
inter-layer calls in the emitted bitcode. OSL graph specialization still
runs (use `-O0` to disable its optional optimizations), and HART still
performs final device optimization.
`--llvm_opt 3` exercises optimized bitcode.
Grid coordinates include the endpoints, with 0.5 for
singleton dimensions. `--print` suppresses image writing. As in CPU
testshade, JPEG/GIF/PNG images are converted to sRGB.
Parameter types follow the normal frontend: use `--param:type=float scale 2`
or `--param scale 2.0` for a float parameter; a bare `2` is inferred as int.

**2D image textures** use the normal OSL `texture()` lowering and a HART
renderer sampler, without CPU execution:

```osl
color albedo = texture("albedo.tx", u, v,
                       "interp", "linear", "wrap", "periodic");
Cout = albedo;
```

The filename must be a nonempty literal or an immutable scalar input string
parameter, including an instance override of an empty default. Interpolated,
interactive, connected, written and runtime-initialized filename parameters
are rejected before optimization. Resolution also works at OSL O0 and retains
the renderer's stable resource IDs; this is not dynamic filename lookup.
An explicit literal `"interp"` of
`"closest"` or `"linear"` is required, as are explicit wrap modes for both
axes: `"wrap"` sets both, or use `"swrap"` and `"twrap"`. Each supports
`"black"`, `"clamp"`, or `"periodic"`. The optional `"alpha", alpha` writes
to a scalar float variable using the normal OSL texture shadeop.
`"firstchannel", N` selects a literal nonnegative integer channel offset
(default zero). Parameter-driven, varying, negative and non-integer offsets
are rejected before optimization, including in unused code.
The default OIIO smart-bicubic/anisotropic filtering is **not** approximated
silently: omitted filtering/wrap options and unsupported options are errors,
including in branches or layers that optimization could remove.

The native `testrender --hart` renderer opts into `HARTTextureDefaults`:
omitted wrap modes use `"periodic"` and omitted interpolation uses `"linear"`,
including linear mip interpolation. These match the reference OptiX renderer's
sampler, **not** OIIO's smart-bicubic/anisotropic defaults. Explicit supported
options override these defaults, including per-axis wraps and repeated options.
Other renderers, including ordinary `testshade --hart`, retain the explicit
option requirement unless they advertise this capability.

The renderer loads raw numeric pixels from the first subimage of a non-deep
2D image with one to four channels. It performs no colorspace conversion or
alpha premultiplication. Float lookups read one channel starting at
`firstchannel`; color lookups read three consecutive channels. Any requested
channel beyond the file's channel count returns zero with zero derivatives,
even if the entire request is beyond the end. Missing channels are not filled
by replicating grayscale. Full shifted image windows work; cropped/data-window
mismatches are rejected. Stored mip levels are preserved, and missing levels are
box-resized to `max(1, floor(size/2))` down to 1x1.

Alpha is the next channel after the returned value: `firstchannel + 1` for a
float lookup or `firstchannel + 3` for a color lookup (zero-based), not a search
for a channel named "A". Missing alpha is zero, with zero derivatives; it is not
implicitly opaque. Alpha's `Dx`/`Dy` follow the same reconstruction and
coordinate-gradient chain rule as RGB, including across layer connections.
Requesting alpha does not premultiply or otherwise change the returned color.
This raw-channel contract does not emulate OIIO's optional grayscale-to-RGB
expansion.

Filtering uses an isotropic footprint in base-level texels:

```text
rho = max(length((W*dsdx, H*dtdx)), length((W*dsdy, H*dtdy)))
lod = clamp(log2(max(rho, 1)), 0, levels-1)
```

`closest` selects mip `floor(lod+0.5)` and texel
`(floor(s*width), floor(t*height))`, with zero output derivatives.
`linear` uses bilinear reconstruction centered at texel centers
`(s*width-0.5, t*height-0.5)` and linearly blends adjacent mip levels.
It computes analytical sample derivatives, holding the footprint/LOD fixed.
The existing OSL shadeop combines those with lookup gradients to produce
`Dx`/`Dy`; this is distinct from using gradients to select a mip.
Both implicit coordinate gradients and the four explicit lookup-gradient
arguments are supported, as are connected/procedural coordinates and sampled
values. This policy is deliberately not OIIO's anisotropic minification.

GPU texture objects are owned by the renderer. Generated bitcode contains
stable resource IDs, while each launch binds the current device descriptor
table, keeping resource addresses out of cached shader code. Required images
that cannot be loaded fail group compilation; nonfinite coordinates/gradients and invalid runtime
bindings fail the launch rather than returning a successful black image.
Dynamic filenames, UDIMs, texture3d/environment, dynamic channel selection,
subimage selection, colorspace, width/blur, fill/missing-color overrides, and
error-message options remain unsupported.

The GPU-opt-in tests `hart-texture-alpha-runtime`,
`hart-texture-channels-runtime` and `hart-texture-materials-runtime` cover
alpha values/derivatives, literal channel boundaries, and a six-layer material
graph with transformed procedural coordinates, channel-selected data and
sampled alpha as a blend mask. The material tests compare CPU and independent
sampling/product-rule references, exercise split and fused storage modes,
and change alpha image contents across A-B-A cache reuse without changing
shader code or filenames.

The four `hart-texture-render-{split,fused,fused-local,unoptimized}` tests
exercise immutable default/overridden filenames, renderer-gated defaults and
explicit option precedence, two simultaneous textures with swapped bindings,
repeated rendering, and missing/empty resource failures without image output.
Their CPU oracles explicitly select matching magnifying samplers.

`Dz`, unlisted noise forms and closure constructors,
tracing, shader printing, writes to read-only or unlisted shader globals,
other globals such as `Ps`, unlisted coordinate spaces and transforms,
string allocation and character operations,
other renderer-service callbacks, batched execution, instrumentation,
and unlisted frontend
options are unsupported.
They fail explicitly; there is **no CPU fallback**.
`--hart-entry` and `--hart-callable-module` belong only to external-module
mode and cannot override generated callables.
The selected architecture is fixed for the lifetime of a `ShadingSystem`;
its `hart_arch` attribute may be set again only to the same architecture.

Generated HART testshade and the native path tracer support interactive numeric
and string parameters, including fixed/resolved arrays. Declare them with
`[[int interactive=1]]` or `--param:interactive=1`. Testshade applies `--reparam`
updates between measured iterations, after each completed launch; warmup does
not update parameters. The API can update A-B-A values on the same compiled
group, without changing shader bitcode or pipeline-cache identity.
Parameter overrides must repeat the interactive hint; an ordinary static
override replaces the shader default's interactive setting.

Custom renderers opt in with `HARTInteractive` and implement the existing
`device_alloc`, `device_free`, and `copy_to_device` hooks. The group allocates and
owns its arena; the sixth callable argument borrows `device_interactive_params`.
The parameter layout, including array lengths, is fixed by optimization. Strings
in both host and device arenas are hashes, not host pointers. Finish GPU work
before calling `ReParameter` or destroying the group; keep its renderer alive
until group destruction. Allocation/copy failures report errors and invalidate
the device binding (the query returns false/null), preventing test-renderer
launches and image publication. A successful update restores the full arena
from the last committed host values before accepting new values.
`hart-interactive-runtime` and the four `hart-interactive-rebind-*` GPU tests
cover CLI updates, derivatives, scalar/array/string values, warmup, errors,
immutable artifacts and A-B-A cache reuse.

Generated testshade also supports interpolated parameters, including int32,
float-based scalar/aggregate/array values and string hashes. Use
`[[int interpolated=1]]` or `--param:interpolated=1`; interpolated closures are
unsupported.
The grid supplies the same `s`, `t`, `face_idx` and conditional `red`, `green`,
`blue` userdata as CPU testshade, with their meaningful derivatives.
`--userdata[:type=TYPE] NAME VALUE` adds uniform values. Missing, absent or
type-mismatched entries use each shader parameter's own default; they are not
device errors. Values supplied without derivatives have zero gradients.

Numeric input parameters can be both interpolated and interactive when the
renderer supports both capabilities. A userdata hit takes precedence; a miss
copies that layer's current interactive default into per-point Groupdata and
zeros its derivatives. Shader execution never writes these resolved values
into the shared interactive arena. Reparameterization changes subsequent
fallback values without recompiling the group or changing cached userdata
hits. Combined string/output/closure parameters and runtime-initialized
defaults without an instance override are rejected. Immutable texture
filenames still cannot be interpolated or interactive.

`hart-interactive-userdata-runtime` checks typed and partial hits, missing and
type-mismatched userdata, distinct defaults in two layers, zero/default and
supplied derivatives, and in-place updates in all four execution modes.
The numerical CPU controls use ordinary interpolated parameters with staged
defaults; the CPU's eager combined-parameter path is not an equivalent oracle.

The generated-renderer API accepts `HartOptions::userdata_bindings`. Each
`HartUserdataBinding` borrows host bytes for one typed record (`stride=0`) or
one record per grid point, with an optional per-point 0/1 presence mask.
Derivative records contain the complete value, Dx and Dy blocks in that order.
String input elements must be host `ustring` values, not character pointers or
precomputed hashes; upload converts each to a device hash. Bindings are copied
to renderer-owned device tables before launch, after previous GPU work has
finished. A-B-A rebinding does not modify compiled group artifacts or cache
identity. Invalid extents, strides, types, presence masks or duplicate names
fail before launch; malformed device bindings report a service error and
prevent output publication.

Custom renderers opt in with `HARTUserdata` and `build_interpolated_getter`,
using an `InterpolatedGetterSpec` to call the typed shadeop
`osl_hart_get_userdata`; they supply its device callback `rs_hart_get_userdata`.
The callback receives execution context, shade index, name hash, encoded type,
derivative demand and destination. It must return true only for supplied data
and fill all requested derivative blocks. Derivative demand includes every
layer sharing that cached userdata, even if the first layer only needs its
value. The six callable arguments are unchanged; tables are reached through
renderer state. Raw `SymArena::UserData` pre-placement and native path-tracer
userdata are not yet enabled. The `hart-userdata-runtime` and four
`hart-userdata-rebind-*` opt-in GPU tests cover typed/default values, gradients,
per-point presence, A-B-A artifact/cache reuse and invalid host bindings.

Generated HART testshade supports all object/index forms of `getattribute`.
Its camera resolution, projection, pixel aspect, screen window, field of view,
clipping planes and shutter come from the renderer's current camera settings,
not constants baked into shader code. It also supplies `osl:version`,
`shading:index`, the test `options`/`blahblah` value, and the same empty-object,
index-minus-one userdata fallback as ordinary CPU `SimpleRenderer`.
Registered camera getters ignore the supplied object and index, as on CPU.
Missing names or mismatched types/extents return false without modifying the
destination or its gradients. Uniform values have zero gradients; the CPU
demo's `options` getter also clears them rather than retaining old gradients.

Custom renderers opt in with `HARTAttributes` and implement the device callback
`rs_hart_get_attribute`. Its arguments are execution context, int32 shade index,
object hash, attribute hash, int64 encoded `TypeDesc`, bool derivative demand,
int32 index (minus one when omitted), and destination; it returns bool.
The corresponding shadeop is `osl_hart_get_attribute`. No host renderer callback
is invoked during HART attribute lowering or mutable-attribute constant folding.
Existing immutable `osl:version` and literal `shader:*name` folds remain;
dynamic shader metadata and unoptimized shader-name queries are not invented
as runtime services. The native path tracer does not yet opt into this service.

The grid uploads a fresh camera `RenderContext` with no host journal pointer
before each render. Its pointer is at offset 48 in `HartTextureState`;
the six-argument callable ABI is unchanged. Genuine HIP probes
cross-check this record and camera layout for every configured architecture.
The four `hart-attribute-rebind-*` tests check 33 query cases at three grid rows
against exact numerical expectations and CPU execution, including return codes,
typed arrays/strings/matrices, derivatives and preserved destinations on misses.
They rebind camera and userdata A-B-A on one compiled group and verify unchanged
artifact bytes/addresses, matching cache hits and actual launches.

HART also accepts a device renderer library through the existing
`ShadingSystem::attribute("lib_bitcode", TypeDesc(TypeDesc::UINT8, byte_count),
bytes)` API. Supply raw HIP/AMDGPU bitcode for the selected `hart_arch`, before
compiling the group; a zero byte count clears the setting. Nonempty buffers
must be non-null, with scalar byte elements and a representable array length.
Read bitcode files in binary mode, including on Windows. The shading system
copies the bytes.

Libraries must match the embedded shadeops' target triple and data layout.
Build against matching OSL and HART headers, including
`amd/hart/hart_device.h` to record the device-storage ABI provenance;
missing or conflicting provenance is rejected before linking.
Defined functions must target the selected architecture, and shared symbols
must have compatible types, calling conventions and ABI attributes. Compatible
weak HIP definitions can be merged normally, but libraries cannot replace
existing strong shadeop definitions. Imports must resolve within the library
or existing shadeops (LLVM intrinsics are also allowed). Kernels, callable
exports, indirect calls, inline assembly, aliases and global initialization
are not supported by this library interface. Rejected libraries report an
error and do not publish a group artifact. Host `rs_bitcode` and NVPTX
libraries are not substitutes for HART device code.

A library can provide genuine renderer services such as
`rs_hart_get_userdata`; the renderer must still advertise the required
capabilities and supply valid runtime state. Linked functions become private
to the compiled group and use the unchanged callable ABI. Existing optimized
groups retain their library when the setting changes; compile a new group
to use a new library. A SHA-256 identity of the complete input is retained in
the group's `osl.hart.renderer_library` metadata, including when unused
library code is pruned, so the bitcode-based pipeline cache includes that
identity.

With `OSL_BUILD_TESTS` enabled, the four opt-in `hart-renderer-library-*`
GPU tests compile two real HIP service libraries, verify shader-global and
shade-index-dependent values and gradients, switch between their immutable
groups A-B-A, and require cache hits when returning to A. Compiler tests use
the actual per-architecture libraries for linkage and malformed-input checks.

Numeric and string arrays and flattened structs support initialization, copying,
length queries, runtime indexing, nested members and whole-aggregate connections.
Unsized input arrays use the existing group-resolved parameter length; an empty
initializer retains OSL's one-zero-element default. Unresolved or zero-length
storage cannot be indexed. Differentiable members retain
their normal derivative storage. Closure aggregates require `HARTClosures`
as well as the array capability; generated grids provide bounded caller-owned
storage but still reject closure-valued image outputs.
Recompile older shaders with arrays-of-struct parameters to include their
struct-field metadata in the bytecode; missing metadata is rejected before
launch rather than losing connected values.

Array storage and potentially out-of-range component accesses require
`HARTArrayBounds`, supplied by both generated testshade and the HART path tracer.
Accesses not statically proven in range also require shader `range_checking`.
The device range-error callback records an error before the existing OSL clamp
keeps the failing invocation memory-safe; it is not a successful clamp-only
fallback. The renderer must check the error before publishing output.
`hart-aggregate-runtime` compares exact values and derivatives with CPU and
independent references in split, fused, fused-local and unoptimized modes,
and distinguishes pre-launch rejection from device bounds errors.

String literals, ordinary locked parameters, copies, scalar equality/inequality
and layer connections use 64-bit `ustringhash` values, including arrays and nested
struct members. String constant arrays contain integer hashes, not fabricated
pointers. No character data or host addresses are needed on the device.
Empty strings compare equally whether supplied as a null-backed empty parameter
or an interned literal; scalar constant folding now uses the same hash comparison
as execution. String arrays use the existing bounds checks and group-resolved
lengths. Final image output remains a single numeric `Cout`.

`hart-string-values-runtime` checks exact CPU/GPU/independent results for
parameters, array copies, nested structs and connected strings in all four
storage/optimization modes, plus explicit bounds failures. `string-empty-compare`
separately covers empty equality and inequality in ordinary CPU builds.
The batched backend retains character-pointer comparisons and is excluded from
this empty-representation regression pending separate qualification.

`hash(string)` returns the signed low 32 bits of the stored hash, matching scalar
CPU and the reference CUDA implementation without reading character memory.
Numeric `hash` overloads for int, float, float/float, triple and triple/float
reuse the existing shadeops. `hart-string-selectors-runtime` checks all six
overloads with lossless 16-bit output packing, including empty/case-distinct
strings, connected strings and published numeric regression vectors. It also
compares dynamic and literal noise in 1D-4D, scalar/color values and derivatives,
all storage modes, and explicit runtime selector/option failures.

Generated-grid UVs and their derivatives use non-approximate division, including
on grids with non-power-of-two interval counts. This prevents rounding below a
selection boundary (even the final UV of 1) from selecting the wrong string.
The test checks exact CPU/GPU grid values and discrete indices; ordinary shader
shadeops retain their existing fast-math settings.
Character operations, allocation and other dynamic service selectors remain
explicit rejections; no placeholder `strlen`/`getchar` results are used.

`printf`, `warning` and `error` use bounded, caller-owned device records when the
renderer advertises `HARTDiagnostics`. Both testshade's generated grid and
testrender's HART path tracer support this service. Records carry shader/source
names, source line and the explicit callable shade index; host decoding resolves
real hashes, including empty and connected strings. Reports are drained after
each synchronized launch, including warmup. Each contextual report is newline
terminated, even when its format has no final newline. Ordering between shading
points follows the full 64-bit shade index. Within each point, reports retain
their execution order, including reports from connected layers. Host ordering
does not change the bounded device storage or diagnostic payload.

Formats must be literal strings. Numeric width (at most 1024), precision (at most
128), ordinary flags and OSL's `cdefgimnopsvxX` conversions are accepted, with at
most 120 characters per specification. Argument types use OSL's usual format
coercion and array/triple/matrix expansion. Dynamic
width/precision, length modifiers, positional arguments, closures and structs
are rejected. OSL's existing frontend exclusion of `%u` is unchanged. File
printing, regex, general string allocation and character operations remain
unsupported.

Each launch holds at most 256 records. Each record has at most 256 flattened
scalar arguments and 2048 packed value bytes. Original and expanded formats and
decoded messages are limited to 4096 bytes, as are shader/source names; each
string argument and formatted field is limited to 1024 bytes. Overflow, invalid
payload/formatting, oversized fields/messages and shader `error` explicitly fail
the render before image
publication, rather than silently dropping or successfully truncating output.
`warning` and `printf` alone do not fail a valid render. The six-argument shader
callable ABI is unchanged. `hart-diagnostics-runtime`, the path-tracer tests and
`unit_journal` cover payloads, bounds, context and repeated-launch draining.

`hart-generated-cli` checks the supported CLI boundary without GPU execution.
The standard test runner also recognizes `HART` marker files. With
`TESTSUITE_HART=1` at configuration, marked fixtures get `.hart` (OSL 0 /
LLVM 10), `.hart.opt` (OSL 2 / LLVM 3), and `.hart.fused` variants, subject to
the existing optimization/fusion markers. These tests retain their original
shaders and numerical thresholds, have `hart;gpu` labels, and run serially.
For CPU-only selections in a mixed build, exclude both backends with
`-LE "hart|optix"`. `testshade` and `testrender` also honor
`TESTSHADE_HART=1` and, with HART selected, `TESTSHADE_FUSED=1`.
The runner rejects conflicting backend selections rather than falling back
to CPU. A dedicated `out-hart.txt` reference is backend-exclusive and can
preserve HART's existing shader/source/point diagnostic prefix; CPU references
are not changed or accepted in its place.
An optional `out-noopt-hart.txt` is exclusive to the OSL 0 variant when
optimization legitimately changes executed diagnostics; it cannot fall back
to the optimized reference.
An optional `out-fused-hart.txt` is exclusive to optimized fused dispatch,
for example when it preserves a different floating-point signed zero. OSL 0
keeps precedence over this variant. CPU reference selection excludes all HART
variants; no numerical tolerance or diagnostic metadata is removed.
Fixtures with included-header diagnostics may opt into `relative_source_paths`:
text comparison removes only the current test directory prefix from HART
source paths. Filenames, line numbers, point indices and payloads remain checked,
and the raw diagnostic output is unchanged.

The standard native-render fixtures also opt into HART. Existing
`OPTIX_OPTIMIZEONLY` markers retain their GPU optimization restriction.
The bump and Cornell fixtures preserve their explicit LLVM modes 13 and 12
while still exercising the selected OSL optimization and dispatch mode.
The runner selects `--hart-bounces 64`, rather than the four-bounce preview
default; explicit scene `max_bounces` settings still take precedence.
The native HART limit remains 64, not the CPU renderer's larger default.

Image references use the same exclusive naming convention, for example
`out-hart.exr`. CPU and OptiX never consume HART references. Backend-specific
images are qualified by visual equivalence with identical display conversion
and exposure, retaining finite/nonblack output, geometry/material/lighting
checks and independent analytic tests. Pixel differences are diagnostic data,
not an expectation of CPU/GPU bit identity; existing comparison thresholds
and CPU/OptiX references remain unchanged.

For an enabled HART build, a core development smoke and the extended selection
can be run separately:

```powershell
ctest --test-dir build\hart-validation -C Release --output-on-failure `
  -R "^(hart-(codegen-gfx1201|generated-cli|reference-selection|generated-runtime)|render-cornell\.hart\.fused)$"
ctest --test-dir build\hart-validation -C Release --output-on-failure -j 1 `
  -R "^hart-|\.hart(\.|$)"
```

The extended selection includes all configured compiler architectures and the
longer numeric, spline, material and resource tests, not just the smoke set.

Set `TESTSUITE_HART=1` when configuring to enable `hart-generated-runtime`,
which compares arithmetic, `sin(u+v)`, and straight-line and conditional
two-layer groups
against CPU execution
on `1x1`, `3x2`, and `37x5` grids. It checks numerical and image output,
cold-cache compilation, warmup, and repeated launches. Absolute tolerances
are `2e-6` for GPU/image values and `6e-6` for comparison with CPU text's
six-significant-digit formatting plus float rounding (relative tolerance
`1e-6`). The CPU text margin does not relax full-precision image checks.
The two-layer tests cover LLVM levels 10 and 3 and an overridden producer
parameter. Control-flow checks cover mixed/all-true/all-false outcomes,
equality boundaries, signed integer and float comparisons, and reuse of a
connected input after branches join. The GPU-independent `hart-codegen-*`
tests check the internal producer call, its six arguments and calling
convention, and exactly three exported HART callables for each configured
architecture. At LLVM level 10, they also verify a varying branch and that
the conditional consumer's producer call is confined to the true branch.

`hart-loops-runtime` is a separate GPU test, enabled by the same
`TESTSUITE_HART=1` configuration. It reuses the generated-shader test helpers
to check `for` and `while`, zero/one/four iterations, varying loop lengths,
loop-carried accumulation, and connected inputs inside and after loops.
It covers LLVM levels 10 and 3, numerical/image output, cold/cache-enabled
compilation, and repeated launches. The `hart-codegen-*` tests use LLVM loop
analysis at level 10 to confirm that real loops remain, containing both a
sine shadeop and, for connected groups, an internal producer call.

`hart-derivatives-runtime` separately checks `Dx`/`Dy` against analytical
values and CPU execution at LLVM levels 10 and 3. It covers single-layer
sine, connected derivative propagation, and a producer with varying
zero/one/three-iteration loops on `1x1`, `3x2`, and `37x5` grids, plus
arithmetic and color derivatives and zero derivatives for uniform parameters
and constant-valued connections. It uses the same numerical/image,
cold/cache-enabled and repeated-launch checks. The `hart-codegen-*` tests
verify linked scalar/color derivative-aware sine calls and derivative-sized
connected storage at level 10, and reject the still-unsupported `Dz`
operation.

`hart-surface-runtime` checks the original surface globals, position
derivatives, and value-only and derivative-aware vector math. A connected
producer exercises point/normal/vector construction and a consumer selects
normalization, length, or dot products, including mixed derivative/non-derivative
operands. It compares analytical and CPU/GPU results at LLVM levels 10 and 3,
with connected groups on `1x1`, `3x2`, and `37x5` grids. Additional cases
check each global and operation, uniform vector parameters, zero-length
vectors with nonzero input derivatives, normal writes, and rejection of
unsupported globals and named spaces. It uses the same image, cache and
repeated-launch checks. The compiler tests verify the linked shadeop variants,
vector value/dx/dy storage and unchanged callable ABI on every configured
architecture.

`hart-filterwidth-runtime` checks scalar and component-wise triple footprints
against analytical values and CPU execution. It covers standalone scalar
shaders and scalar/vector connections on `1x1`, `3x2`, and `37x5` grids,
all four triple types, negative input derivatives, zero widths for uniform
parameters and constant-valued connections, and zero derivatives of the
width result. Tests run at LLVM levels 10 and 3, retaining the existing
numerical tolerances, image comparisons, cache modes and repeated launches.
Compiler checks verify the linked scalar/triple shadeops and connected
derivative storage on every configured architecture. Unlisted globals remain
rejected.

`hart-noise-runtime` compares numeric Perlin values and derivatives with CPU
execution at LLVM levels 10 and 3. A packed `12x3` matrix covers 1D-4D inputs,
float/color/vector results, and all three result components for both signed
and unsigned noise. Checks include the signed/unsigned relationship,
constant-input derivatives, and composition with `filterwidth`.
Noise values retain the `2e-6` comparison tolerance. Only noise derivatives
and footprints use `4e-6`: CPU SIMD and scalar HIP interpolate corners in
different orders, and measured rounding differences reached `3.16e-6` after
connected arithmetic. Production floating-point settings are unchanged.
Independent derivative checks use full-precision CPU central differences
with coordinate offsets of `1/512`; their separate `5e-4` approximation
tolerance is in unscaled u/v derivative units, not the direct CPU/GPU
comparison tolerance. Representative connected 4D arithmetic groups also
exercise `1x1`, `3x2`, and `37x5` grids, images, cache modes and repeated launches.
Compiler checks verify all plain and derivative shadeop signatures,
including mixed varying/constant coordinates and connected storage.

`hart-math-runtime` packs scalar/color/vector math and integer remapping
probes into image comparisons at LLVM levels 10 and 3. Independent analytical
checks cover values and connected derivatives, negative and zero inputs,
transition boundaries, and constant-input zero derivatives. A representative
connected group also exercises cache modes and repeated launches.
The existing numerical tolerances are unchanged.

`hart-noise-families-runtime` covers periodic Perlin, cell/hash noise, and
literal-name selection including simplex at LLVM levels 10 and 3.
Packed float/color/vector cases exercise 1D-4D inputs, connected derivatives,
named/numeric aliases, signed/unsigned relationships, periodic shifts around
negative coordinates and tile seams, and period flooring/minimum semantics.
Cell/hash derivatives are checked against exact zero. Simplex derivatives
are also checked with CPU central differences away from discontinuities.
Hash inputs use binary-exact coordinates so the CPU/GPU comparisons test the
same input bits. Values and simplex derivatives retain `2e-6`; only Perlin
derivatives use `4e-6`. No production floating-point settings are changed.
Dynamic/unknown/empty names, non-Gabor options, and periodic simplex are rejected.

`hart-gabor-runtime` covers literal Perlin aliases, real scalar/color Gabor in
1D-4D, values and derivatives, option defaults/reset/clamps, filtering,
embeddings, ignored time, connected groups and explicit prelaunch/device
failures. Split, fused and callable-local variants include a compound periodic
regression at exact seed-cell multiples. Separate unfiltered central differences
check all points in a small smooth patch at two steps; periodic shifts use the
bandwidth-derived radius and include a nonzero world-unit-shift control.
Default CPU/HIP comparisons remain `2e-6`. CPU/GPU radius constants differ by
one ULP and transcendental approximations differ: measured directional/hybrid
cases use `3e-6`, the high-frequency unfiltered directional case uses `3e-5`,
and bandwidth `.01` uses `1e-5`. Exact clamp identities are checked separately.
These are localized numerical allowances, not changes to production math flags
or other noise tests. The path tracer also exercises connected Gabor materials
and verifies that invalid noise prevents output publication.

`hart-texture-resources` checks the HART test renderer's image resource layer
on a HIP device. It verifies float image uploads, existing and box-generated
mips (including non-power-of-two dimensions), stable texture IDs, descriptor
table replacement, channel zero-fill, raw numeric image data, explicit load
errors, and repeated cleanup. Resource addresses are held in a launch-time
device table rather than embedded in shader bitcode.

`hart-texture-runtime` exercises real OSL-generated GPU shaders at LLVM levels
10 and 3. A packed matrix checks float/color lookups, all supported wraps,
texel boundaries, magnification, integer/fractional mip selection, implicit
and explicit gradients, and sampled `Dx`/`Dy`. It includes connected
noise-transformed UVs, zero-filled monochrome images, and exact-zero
constant/closest derivatives. An independent sampler oracle checks the
specified mip policy; CPU comparisons use compatible magnifying lookups.
Three textures coexist in a cached group whose file contents change A-B-A,
with confirmed cache hits and repeated launches. Rejections cover unsupported
options in unused code, missing/UDIM resources, and nonfinite device inputs.

`hart-matrix-runtime` checks explicit matrix transforms and arithmetic on the
GPU, including derivative propagation, non-uniform scale, shear, composition,
inverse/transpose, projective points, and matrix-valued layer connections.
It compares analytical cases and CPU results at LLVM levels 10 and 3.

`hart-spaces-runtime` checks named point/vector/normal transforms, constructors,
matrix queries, values and derivatives against CPU and analytical references.
`hart-transform-rebind` uses one already optimized group with A-B-A runtime
matrices, checks the artifact remains unchanged, and requires matching pipeline
cache hits while the rendered values and derivatives change and change back.

The four `hart-grid-lifecycle-*` tests repeat three complete renderer/system/group
lifetimes in one process. Two live renderer instances deliberately reuse shader
and group names with different shader code and independent interactive arenas.
Exact pixels and derivatives check isolation and changed bindings; artifact and
arena addresses remain stable within each lifetime. Bounded device-array errors
and failed between-iteration updates must publish no partial image. Another
renderer remains usable after failure, and the failed renderer must recover with
matching pipeline-cache keys. These are serialized GPU tests, not a guarantee
of concurrent host submission or multi-GPU support. They do not infer leak
freedom from process-wide free VRAM, which other applications and SDK caches
can change.

`hart-context-lifecycle` repeatedly initializes, traces, clears and reuses one
native context while a second context remains live. It checks owned allocation
counts and bytes, module/program-group/pipeline ownership, cross-context pointer
rejection, invalid bounded operations and cleanup of a partially created
pipeline. The four `hart-native-lifecycle-*` tests add two materials per renderer,
exact emission images, interactive A-B-A updates, output growth/reuse/shrinkage,
and failure after one material has compiled. Native renderer errors are terminal:
`clear()` releases resources and invalidates the published image but does not
erase the error history. Recovery uses a fresh renderer in the same process.
Preparation also invalidates an old image before any possible failure, so a
failed operation cannot expose an earlier successful frame as its result.
Ownership snapshots cover the native context's resources, not SDK caches,
texture storage or group-owned interactive allocations.

`hart-geometry-runtime` checks initial `I` and `time`, their derivatives and
filter widths against the grid's exact-zero defaults. It also composes named
transforms, procedural UVs and texture sampling in standalone and connected
groups at LLVM levels 10 and 3, including disabled OSL optimization. Values
and derivatives are compared with CPU execution and the sampler oracle.
It also verifies an incident-vector component write and its gradients;
unsupported globals and writes to time still fail before launch.

`hart-groups-runtime` checks longer numeric chains with float, color, point,
vector, normal and matrix connections. Group storage, derivative propagation,
and the six-argument callable ABI use OSL's existing layer machinery, without
a HART-specific scheduler or fixed layer-count cap. The last layer remains
the sole default entry point; every original layer is validated before
optimization, including unused layers.

`hart-topology-runtime` checks fan-in/fan-out graphs, shared producers, and
conditional inputs reused after branches. Layer execution remains lazy:
only required dependencies execute, and a shared producer runs at most once
per shading point. Compiler tests verify distinct execution flags, guarded
calls and complete initialization, including remapping past unused layers.
GPU probes put overflowing texture coordinates in unselected producers:
lazy execution must succeed, while explicitly consuming both branches must
report a device sampling error. This checks execution, not just matching pixels.

`hart-materials-runtime` composes named transforms, procedural UV distortion,
texture lookups, masks and mixing across a larger connected material graph.
It checks values and derivatives against CPU execution and reference sampling,
including parameter overrides, LLVM levels 10/3 and disabled OSL optimization.
Compiler checks cover the corresponding six-layer composition on every
configured architecture.

`hart-procedural-runtime` combines these operations in connected groups:
a bounded multi-octave periodic-noise loop with a color ramp, a repeating
cell/hash pattern, and an explicitly footprint-filtered transition.
The tests compare CPU/GPU values and derivatives at LLVM levels 10 and 3,
check opposite tile edges along both axes on a `17x9` grid, and verify that
additional octaves and explicit filtering change the output. A singleton
group also runs with OSL optimization disabled, cache modes, and repeated
launches. Compiler checks cover the loop/footprint/ramp composition on all
configured architectures. The fixture's octave limit is only a test-workload
bound, not a backend limit.

```powershell
ctest --test-dir build\hart-validation -C Release `
  -R "hart-(generated|loops|derivatives|surface|filterwidth|noise|math|procedural|texture|matrix|spaces|transform|geometry|groups|topology|materials|grid)" --output-on-failure
ctest --test-dir build\hart-validation -C Release `
  -R "^hart-(codegen-.*|texture-(resources|runtime))$" --output-on-failure
```

Prefer an OptiX-disabled build on an AMD-only machine. A mixed HART/OptiX build
needs the CUDA runtime dependencies even when executing HART; such a run is
not validation of the NVIDIA backend.

### Inspecting generated HART closures

Numerical closure-tree inspection is separate from ordinary RGB `testshade --hart`
output. A renderer advertising `HARTClosures` may compile scalar closure values,
closure connections and `Ci`, using registered `diffuse(N)` and `emission()`
components, addition, scalar/color multiplication and null closures. Without
further renderer capabilities, other constructors and keyword arguments are
rejected before optimization.

A renderer also advertising `HARTClosureParameters` may use its registered
closures with numeric, scalar string-hash and nested closure parameters.
Literal registered keyword names may have varying values. HART validates
registration sizes, alignment, field bounds, overlaps and types before
optimization. With `HARTArrayBounds`, fixed-length numeric arrays are supported
as formal and keyword parameters, including the `color[8]` fields used by
Blender's `diffuse_ramp` and `phong_ramp`. Supported element types are int,
float, color, point, vector, normal and matrix. Arrays are copied inline into
the registered record, not stored as pointers. The argument's resolved length
must match the registered length; unsized shader inputs use their initializer
or instance binding to resolve it. Unsized registered arrays, string arrays,
closure-pointer arrays and host prepare/setup callbacks remain unsupported.
The renderer must actually consume these device records: this capability does
not provide an implementation of an arbitrary registered closure.

The test renderer binds a combined `HartRenderState` through `ShaderGlobals`
`renderstate`. It contains the existing texture descriptor/error state and a
bounded closure pool. Pool storage belongs to the raygen caller and outlives
both split and fused shader calls, including callable-local Groupdata.
`rs_allocate_closure` checks size and alignment without overflowing and makes
allocation failure sticky until reset. Exhaustion sets a device error and
fails the launch without returning a partial result.

With `OSL_BUILD_TESTS`, `USE_LLVM_BITCODE` and `OSL_USE_HART` enabled,
`hart_closure_test` inspects trees on the GPU **after shader return** and
transfers only numerical component counts, weights, parameter payloads and
allocation sizes to the host. Its test pool is 1024 bytes per shading point;
this is not a promise about physical GPU stack use or a general renderer
allocation limit.
The unit covers a diffuse/emission material, null and conditional trees,
weighted operations, connected texture/procedural weights, allocator
boundaries, exact-fit storage, repeated resets and post-launch exhaustion.
Ramp cases compare all 24 components of constant and varying `color[8]`
parameters against CPU and independent expectations, including weighted
closures, an array keyword and a following scalar field. Compiler checks cover
full-field copies of each numeric array type, formal/keyword argument type and
length mismatches, resolved unsized inputs and invalid registrations.
It does not implement path tracing or change the ordinary RGB output interface.

Set `TESTSUITE_HART=1` during configuration, then run:

```powershell
cmake --build build\hart-validation --config Release `
  --target hart_closure_test hart_codegen_test --parallel 8
ctest --test-dir build\hart-validation -C Release `
  -R "^hart-(closures-.*|codegen-.*|grid-bitcode)$" --output-on-failure
```

The closure runtime variants are `split`, `fused`, `fused-local` and
`unoptimized` (OSL optimization 0, LLVM mode 10). Compiler checks and generated
device modules cover all configured architectures. Compilation alone is not
evidence of execution on Linux, gfx1100, gfx1151 or NVIDIA hardware.

### Testing native HART triangle traversal

`hart_trace_test` exercises the native HART acceleration and pipeline API,
without HART's OptiX compatibility aliases or NVIDIA runtime dependencies.
It uploads indexed triangles, builds a triangle acceleration structure, and
selects a material record through each primitive's SBT offset. Raygen traces
real rays; closest-hit returns primitive IDs, distance, barycentrics and the
SBT material ID. The GPU also computes geometric normals from the uploaded
world-space vertices.

With `OSL_BUILD_TESTS`, `BUILD_TESTING`, `USE_LLVM_BITCODE` and `OSL_USE_HART`
enabled, set `TESTSUITE_HART=1` when configuring and run:

```powershell
cmake --build build\hart-validation --config Release `
  --target hart_trace_test --parallel 8
ctest --test-dir build\hart-validation -C Release `
  -R "^hart-raytracer-" --output-on-failure
```

The runtime test compares the existing CPU scene/BVH with analytical and GPU
hit/miss results for zero, one and multiple triangles, overlapping geometry,
near/far clipping, back faces, non-unit directions, sheared geometry normals,
distinct material records, A-B-A ray rebinding, and an empty scene. Invalid
geometry produces an explicit error before acceleration construction.
Codegen tests verify the device module for every configured architecture.
Only execution on the selected physical device is runtime evidence.

### Rendering materials with HART

`testrender --hart scene.xml output.exr` uses the existing XML scene parser,
triangle meshes, camera and sampling helpers with native HART traversal and
OSL-generated material callables. It reuses the reference renderer's BSDF/BSDL
implementations for diffuse, Oren-Nayar, Phong, Ward, GGX/Beckmann microfacet,
reflection/refraction, translucent/transparent, MaterialX diffuse, sheen,
conductor, dielectric, generalized Schlick, layers and SPI thinlayer surfaces.
Emission includes `emission()` and MaterialX uniform EDF. Closure parameters
and keywords, nested closures, connected textured/procedural weights and
modified shading normals use the same generated callable path.
MaterialX subsurface uses the reference's diffuse approximation, not a BSSRDF.

The integrator uses the reference renderer's BSDF mixtures, explicit triangle
light sampling, power-heuristic MIS and Russian roulette, maintaining medium
boundary state for dielectric surfaces. Mark emissive shader groups with
`is_light="yes"` to include their triangles in direct-light sampling.
Background shaders illuminate misses and can participate in importance
sampling. Homogeneous MaterialX anisotropic and medium VDFs use the shared
reference volume integrator. Displacement, heterogeneous volumes and diagnostic
visualization modes remain explicitly unsupported.
`--hart-bounces N` limits surface and volume events to 0..64 (default 4);
`-aa N` uses N squared samples per pixel, with N in 1..64. A limit of zero
shows directly visible emission and background only. `--no-jitter` fixes primary
rays at pixel centers while retaining deterministic BSDF and light sampling.
The scene's `rr_depth` option controls when roulette starts (default 5).

Native `--runstats --warmup --iters N` also reports separate OSL material-group
preparation, pipeline construction, synchronized launches and full frame times,
all in milliseconds. Pipeline time includes modules, groups, linking, deferred
stack-query compilation and SBT setup, but not context initialization or
acceleration construction. Group preparation can include compilation-time
texture/binding work; it is not a source-parser-only measurement.
Launch counters and timing reset after warmup. They include background-table
launches as well as the main frame, but exclude parameter uploads, output clears,
error readbacks and image downloads. Full frame time includes the entire
`render()` operation, including those operations, background preparation and
pixel publication, but excludes the final image-file write. These are
instrumented host timings, including submission/wait overhead, not GPU event
times. Verbose diagnostics also contribute to process and frame wall time.

`--hart-no-cache` disables the native SDK pipeline cache. Otherwise the SDK
cache remains enabled; native statistics report the policy, **not an observed
cache hit**. Warmup rendering and cross-process cache priming are different
operations. Native memory statistics report context-owned allocation bytes,
caller Groupdata stride and the configured local-storage budget, not physical
VRAM usage. Textures, group-owned interactive allocations and opaque SDK/driver
caches are excluded. Traversal/state/continuation stack estimates are SDK
requirements, not measured registers, spills or physical per-thread stack use.

The native benchmark reuses the analytic two-material white-furnace fixture:

```powershell
python .\testsuite\cmake-hart\check-pathtracer.py `
  (Resolve-Path .\build\hart-validation\bin\Release\testrender.exe) `
  (Resolve-Path .\build\hart-validation\bin\Release\oslc.exe) `
  (Resolve-Path .\src\shaders\stdosl.h) split --benchmark
```

It runs all three optimized modes (the positional mode chooses the starting
rotation), at 128x96, 16 samples per pixel and one bounce. Each mode has a
cache-disabled single-frame sample, a separate cache-enabled priming frame,
and three trials of one warmup plus 20 measured frames. Local mode uses a
4096-byte budget. Finite bounded radiance, analytic mean furnace energy and
cross-mode energy checks remain required. Per-pixel CPU/GPU differences are
recorded as diagnostics, not equality gates; qualify visual equivalence under
the same display conversion/exposure. JSON records separate individual timings,
medians/ranges, cache policy, logical memory and SDK stack estimates; there are
no speed thresholds.
Neither benchmark is a cross-vendor throughput comparison. Record the build,
device, driver/SDK, flags and trial ranges with any reported results.

A positive `<Background resolution="N"/>` uses at least 32 samples per axis.
OSL background values are evaluated on HART in batches of at most 65,536 texels;
the shared host CDF builder prepares importance tables from those values, not
from CPU shader execution. Tables refresh after interactive parameter uploads.
Nonpositive resolution disables importance sampling but preserves direct
background shader evaluation on misses. Importance sampling requires finite,
nonnegative radiance. Black maps and zero-energy rows remain black and use
valid sampling distributions, without divisions by zero. The scene parser
still requires geometry, even if every camera ray misses it.

Volumes support absorption, scattering, anisotropy, weighted layers and
dielectric boundaries. Up to eight nested medium entries, including vacuum
entries, retain the reference priority and IOR rules. Equal highest-priority
media combine their coefficients. Coefficients must be finite and nonnegative;
IOR must be finite and positive, and anisotropy must satisfy `abs(g) < 1`.
Medium VDF transmission colors must lie in `[0,1]` with positive transmission
depth, except that the reference black-albedo/black-transmission vacuum may
use zero depth. Albedo is clamped to `[0,1]` when forming scattering coefficients.
Zero-extinction channels remain transparent, and complete absorption terminates
a path without being mistaken for an error. Invalid parameters, samples or
medium-capacity exhaustion fail explicitly without publishing an image.
As in the reference renderer, volume integration occurs on finite-hit segments;
this is not a new infinite-medium or transmissive-shadow implementation.

Hit-point ShaderGlobals use real incident rays, interpolated normals/UVs,
mesh surface areas, camera/ray-cone differentials and camera/diffuse/shadow ray
flags. Secondary ray origins use triangle-scaled offsets to avoid shared-edge
self-intersections. The bounded 1024-byte closure pool belongs to the raygen
caller and is reset only after consuming each material. Closure trees are
checked for valid records, bounded depth and cycles before reference traversal.
A separate aligned 1024-byte BSDF arena holds up to 32 lobes, subject to their
sizes. Invalid closure records, distributions, PDFs and exhausted arenas
produce device errors rather than partial images. Light evaluation has a
separate closure pool so it cannot overwrite an active surface's closures.
Background preparation and image rendering use distinct raygen records in
one pipeline; stack-overflow, trace-depth and stack-size validation remain
enabled for both. `--hart-fused` selects fused
callables; `--hart-local-groupdata BYTES` additionally permits bounded
callable-local storage. Otherwise Groupdata uses per-pixel caller storage.
Allocation, compilation, launch and device errors fail without writing an
image. This traversal does not implement the OSL `trace()` renderer service.

```powershell
cmake --build build\hart-validation --config Release `
  --target testrender oslc hart_trace_test hart_material_test --parallel 8
ctest --test-dir build\hart-validation -C Release `
  -R "^hart-(pathtracer|raytracer|material|background|lighting|volume)-" --output-on-failure
```

The path tests compare connected textured emissive materials and geometry
derivatives with CPU rendering, verify a diffuse furnace against known RGB values, and
compare multiple bounded bounces with the CPU integrator with direct-light
sampling disabled. They cover repeated launches, shared edges at three scene
scales, rejected empty scenes, and closure-pool exhaustion without image
publication in split, fused, fused-local and unoptimized (OSL 0 / LLVM 10)
modes. Groupdata queries verify caller versus callable-local storage rather
than inferring it from matching pixels. CPU/OptiX renders can use
`--max-bounces N` to match the HART depth limit; their existing
default is unchanged.

The material component probe compares unquantized GPU/CPU albedo, BSDF
evaluation and sampling, PDFs, directions and roughness for 20 surface cases
at three inputs. Twelve additional volume cases check independent absorption
and scattering values, clear channels, phase directions, layers, IOR and
priorities. Fourteen invalid/boundary cases cover cycles, truncated records,
the 32/33 lobe limit, overflowing PDFs, malformed media and the eight-entry
medium limit, including rollback after a failed update.
Grazing microfacet cases check stable density and masking; subnormal MIS
densities have independent balance/power-heuristic expectations. Captured
near-normal anisotropic samples check finite PDFs and unit transmitted rays.
The renderer bounds normalized sampling cosines, avoids underflowed squared
cotangents, and normalizes subnormal PDF pairs before HIP reciprocal division.
Its float tolerance is `2e-6 + 2e-5*abs(CPU)`; only the re-evaluated PDFs of the
two captured `.001`-roughness sampling cases use `1e-4` relative tolerance for
direction-rounding amplification. Delta PDFs and rejection flags have exact
checks. Material rendering tests exercise 21 CPU/HART scene pairs
in all four dispatch/storage modes with direct-light sampling disabled.
Image comparisons use `3e-5` absolute tolerance or adjacent HALF values:
`testrender` quantizes output to HALF even for float PFM files. Constant
reflection/transparent references and zero-bounce output remain exact.
Invalid runtime distributions must fail without publishing an image.
Weighted layer records preserve the scaling of explicit closure multiplication,
including base lobes and opacity. The layered scene also compares CPU OSL 0
and OSL 2 to catch optimizer-dependent energy changes.

Lighting tests compare 17 CPU/HART render pairs per execution mode, including
direct and importance-sampled backgrounds, black/partially black maps,
delta-BSDF MIS, designated area lights, visibility, multiple lights, combined
environment/area lighting and multibounce paths. A 257-by-257 textured
background crosses the prepass batch boundary; forced early roulette and
repeated renders exercise their respective paths. Invalid background closure
types must fail without publishing an image. Codegen checks verify both raygen
entries for every configured architecture.

Volume tests compare 15 CPU/HART scenes per execution mode, with independent
Beer-Lambert references for absorption, vacuum, weighted layers and nested
priorities. They cover clear channels, forward/backward scattering, dielectric
boundaries, repeated rendering, CPU optimizer invariance and invalid-media
rejection without image publication. Shared phase sampling uses paired
sine/cosine evaluation to preserve unit ray directions; coarse independent
`sinpi`/`cospi` approximations do not meet that invariant. A scatter event clears
the previous-triangle exclusion so rays can exit through the entered face.
Transmitted CPU/CUDA boundaries reuse HART's triangle-scaled origin offset to
prevent neighboring triangles from recording the same medium entry repeatedly.
Windows reference variants account for rare path changes from these corrections;
existing references and comparison thresholds are retained.

### Testing external HART device code

`testshade --hart --hart-module FILE.bc` preserves the independent external
AMDGPU bitcode runner. It bypasses OSL group compilation. Mixing an external
module with OSL shader arguments, or using unsupported options, is an error.

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

### Installing and checking a relocated HART build

Shader and PTX data destinations default to paths relative to the installation
prefix. `cmake --install BUILD --config Release --prefix PREFIX` therefore
relocates them along with the public headers, libraries, tools and CMake
exports. An explicit absolute `OSL_SHADER_INSTALL_DIR` or
`OSL_PTX_INSTALL_DIR` remains an override. Existing build caches may retain
the old absolute defaults; remove just these two cache entries when
reconfiguring to adopt the relative defaults. Compiled-in shader/PTX fallback
paths still refer to the configured prefix; pass an explicit installed shader
include path to `oslc` after relocation.

The Windows runtime DLLs and native codegen worker are app-local, but this is
**not an SDK-free deployment**. Cold HART compilation also needs the selected
HIP SDK's compiler resources. Set `HIP_PATH` to that compatible SDK root,
not to its `bin` directory. A cached pipeline can hide this dependency; the
installed check disables caching for generated grid pipelines. Keep the
selected HART/ROCm development installations available, including any compiler
resource paths required by that HART build. Do not substitute driver DLLs or
another ROCm version.

On Linux, HART executables retain the selected SDK runtime paths and gain an
executable-relative OSL library RPATH (normally `$ORIGIN/../lib`). This respects
`CMAKE_SKIP_INSTALL_RPATH` and nonstandard `CMAKE_INSTALL_BINDIR` /
`CMAKE_INSTALL_LIBDIR` layouts. SDK libraries are not bundled on Linux; preserve
their package layout, including the worker beside the HART library or in its
sibling `bin` directory. Configuration tests are not a Linux build/run result.

Build the standalone consumer against the **installed** CMake package, not
build-tree targets, with matching OIIO/Imath dependencies. For example:

```powershell
cmake -S .\testsuite\cmake-hart\install-consumer -B C:\temp\osl-consumer `
  "-DOSL_DIR=C:\temp\osl-install\lib\cmake\OSL" `
  "-DCMAKE_PREFIX_PATH=D:\OSL\dependencies\x64-windows"
cmake --build C:\temp\osl-consumer --config Release
python .\testsuite\cmake-hart\check-install.py C:\temp\osl-install `
  --consumer C:\temp\osl-consumer\Release\osl_install_consumer.exe `
  --oiio-runtime-dir D:\OSL\dependencies\x64-windows\bin `
  --hip-root D:\opt\rocm\therock-dist-windows-gfx120X-all-7.14.0rc3 --gpu
```

The checker uses a fresh temporary working directory outside source, build and
installation trees. It removes inherited loader/backend settings, then uses
only the install, explicit dependency runtime directories and explicit HIP
compiler-resource root. On Windows it checks app-local runtime families and
exact PE imports, including optional ROCm kpack, without adding SDK binaries
to `PATH`. It verifies compiler metadata, exact CPU and split/fused GPU grid
values, native emissive rendering, and the consumer's compile/query/CPU APIs.
This is not a loaded-module audit. Mixed HART/OptiX installations also need
`--cuda-runtime-dir` naming their CUDA runtime directory, even for CPU work.
For CPU-only installations omit `--gpu` and `--hip-root`; GPU work is reported
as **NOT RUN**, not as a successful GPU test. The checker deliberately requires
the default installed `share/OSL/shaders/stdosl.h` layout.

The consumer disables user-wide Visual Studio/vcpkg integration so it cannot
auto-copy unrelated OIIO/Imath DLLs beside the executable and shadow the
explicit runtime directory. This does not change system integration settings.
For incremental native builds, generated BSDL lookup headers are declared
outputs: a no-op build leaves headers and embedded native bitcode untouched,
while changing a real dependency or deleting a generated header rebuilds the
affected outputs.

### External-module contract

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

The supported options are `--hart-module`, `--hart-callable-module`,
`--hart-entry`, `--hart-device`, `--hart-no-cache`,
`--res`/`-g`, `--iters`, `--warmup`, `--print`, `-v`/`--debug`, and
`-o Cout FILE`. As in ordinary testshade, `--print` suppresses image writing
and the filename `null` suppresses it as well. Image output is linear RGB
float data; no display color conversion is performed.

For an OSL-shaped callable integration test, supply a separate callable module:

```powershell
.\build\bin\Release\testshade.exe --hart `
  --hart-module .\build\src\testshade\hart\gfx1201\callable_grid_hart_gfx1201.bc `
  --hart-callable-module .\build\src\testshade\hart\gfx1201\callable_shadeops_hart_gfx1201.bc `
  --hart-no-cache -g 3 2 --print -v
```

The optional callable module provides two fixed entries:
`__direct_callable__testshade_init` at callable SBT index 0 and
`__direct_callable__testshade_entry` at index 1. Both use OSL's existing
six-argument group calling pattern:
`void (ShaderGlobals*, void* groupdata, void* userdata, void* output,
int shadeindex, void* interactive_params)`.
The supplied raygen must invoke these entries; adding the module does not
change a raygen's behavior automatically. Both modules undergo the same
bitcode and architecture checks before HART initialization.
Compile callable sources with HART's device header even when they use no HART
intrinsics: it emits device-storage ABI provenance required by HART.

The hand-written fixture uses the actual OSL `ShaderGlobals` header and a
small, typed per-point group structure, not a generated shader-group layout.
Raygen passes private shader globals, group storage, userdata and interactive
parameters, plus a global output pointer. Init populates the group, then entry
calls the real `osl_sin_ff` shadeop and writes
`(u, v, 0.25 + 2 * sin(u+v))` through the output pointer and shade index.
The build links only the needed definitions from the matching architecture's
shadeops bitcode into the callable module; HART links it with raygen at runtime.
This tests callable dispatch, pointer passing and shadeop integration
independently of OSL AMDGPU code generation or renderer services.
Callable grids are limited to `INT_MAX` pixels by the shade-index argument.

`--hart-no-cache` disables HART's pipeline cache for this invocation, forcing
pipeline compilation without deleting or changing the user's cache contents.

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
runtime errors. It also runs the callable/shadeops fixture with caching disabled,
including singleton and rectangular grids, image output and missing callable
entries. With `--llvm-opt`, the test also verifies malformed,
non-AMDGPU, mixed-target, and device-mismatched bitcode rejection, including
misleading filenames. It can also be run directly:

```powershell
python .\testsuite\cmake-hart\check-grid.py .\build\bin\Release\testshade.exe `
  --module .\build\src\testshade\hart\gfx1201\grid_smoke_hart_gfx1201.bc `
  --callable-grid .\build\src\testshade\hart\gfx1201\callable_grid_hart_gfx1201.bc `
  --callable-module .\build\src\testshade\hart\gfx1201\callable_shadeops_hart_gfx1201.bc
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
