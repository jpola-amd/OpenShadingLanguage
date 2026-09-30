# Arnold-compatible Windows static profile

This is an isolated **Release, x64, `/MD`** profile with static OSL and Arnold's
static OpenImageIO 2.6. It enables `OSL_ARNOLD_COMPAT`, **both**
`OSL_USE_HART` and `OSL_USE_OPTIX`, and LLVM bitcode. Select HART and OptiX on
**separate `ShadingSystem` instances**; they are not interchangeable modes on
one instance. This is a rebuilt-consumer contract, not binary compatibility
with a previously built Arnold OSL integration.
Successful configuration is not acceptance of the integration: the completed
ShaderGlobals initialization contract and CPU/GPU consumer tests are required
before claiming compatibility.
With this profile enabled, OptiX init, entry, and fused exports use Arnold's
five-argument callable ABI. They forward a null `interactive_params` pointer to
OSL's six-argument internals; groups requiring an interactive-parameter arena
are rejected before PTX cache lookup or code generation. CPU, HART, and
profile-disabled OptiX retain their six-argument ABI. Rebuild renderer CPU code
and GPU renderer-service bitcode against the matching full-layout
ShaderGlobals headers: existing compact-layout bitcode is not compatible.
NVIDIA runtime acceptance remains unvalidated; enabling both backends or
successfully generating PTX does not establish it. See
[the profile-specific callable ABI](ArnoldCompatibility.md#profile-specific-arnold-optix-callable-abi)
before replacing Arnold's OSL package.

The helper does not modify Arnold, its dependency packages, the root
configuration scripts, or `build\hart-current` / `install\hart-current`.
The Current profile remains shared OSL with the current OIIO and HART only.

## Reproduce the dependency and configuration

Run from the OSL source directory. All output is beneath
`build\hart-arnold` or `install\hart-arnold`. Neither helper downloads source.
The official JPEG 3.1.0 source can be obtained separately:

```powershell
$jpegSources = ".\build\hart-arnold\dependencies\jpeg-source"
New-Item -ItemType Directory -Force $jpegSources | Out-Null
curl.exe --fail --location --output "$jpegSources\libjpeg-turbo-3.1.0.tar.gz" `
  https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.0/libjpeg-turbo-3.1.0.tar.gz
$hash = (Get-FileHash "$jpegSources\libjpeg-turbo-3.1.0.tar.gz" -Algorithm SHA256).Hash
if ($hash -ne "9564c72b1dfd1d6fe6274c5f95a8d989b59854575d4bbee44ade7bc17aa9bc93") {
  throw "Unexpected JPEG source archive"
}
Push-Location $jpegSources
cmake -E tar xzf .\libjpeg-turbo-3.1.0.tar.gz
Pop-Location

.\src\build-scripts\build-arnold-jpeg.ps1 `
  -JpegSource "$jpegSources\libjpeg-turbo-3.1.0" `
  -Nasm D:\vcpkg\downloads\tools\nasm\nasm-3.01\nasm.exe

.\src\build-scripts\configure-hart.ps1 -DependencyProfile Arnold `
  -LLVMRoot D:\OSL\LLVM\llvm-23.0.0-rocm-install `
  -HartRoot D:\hart-repos\hart-radeon-pro\install `
  -RocmRoot D:\opt\rocm\therock-dist-windows-gfx120X-all-7.14.0rc3 `
  -Architectures gfx1201 `
  -CudaRoot D:\OSL\dependencies\cuda\12.9 `
  -OptixRoot D:\OSL\dependencies\optix\8.0.0 `
  -CudaArchitecture sm_60
```

The first command builds, tests, and installs **JPEG only**. The second command
only configures OSL. For a different compatible `/MD` JPEG installation, pass
`-JpegRoot`; by default it uses
`build\hart-arnold\dependencies\jpeg-md`. The profile verifies the archive's
CRT directives before using it.

After completing source changes, build and install separately:

```powershell
cmake --build .\build\hart-arnold --config Release --parallel 8
ctest --test-dir .\build\hart-arnold -C Release --output-on-failure `
  -R '^(cmake-package-export|cmake-hart-discovery)$' --no-tests=error
# Run the HART/OptiX runtime and rebuilt-consumer acceptance tests as well.
cmake --install .\build\hart-arnold --config Release
```

Alternatively, `configure-hart.ps1 -DependencyProfile Arnold -Test -Install`
reconfigures, builds, runs its selected compatibility/HART tests, and installs.
This includes the Arnold API/CPU/PTX acceptance and custom HART closures at
O0/O2. Its selected tests are not a substitute for OptiX hardware acceptance
or the separate installed-consumer test.
Do not copy into Arnold's dependency tree as part of this procedure. Stage the
validated installation separately (for example `D:\OSL\artifacts\hart-arnold`).

## Selected toolchains and ABI

Configuration verified on 2026-09-30:

| Component | Selected version / architecture |
| --- | --- |
| Host | Visual Studio 2022, toolset 14.44.35207, compiler 19.44.35228.0, x64, Windows SDK 10.0.26100.0 |
| CMake | 4.3.3 |
| OSL LLVM | `23.0.0git`, targets X86, NVPTX, AMDGPU |
| HART | package 0.1.0, `D:\hart-repos\hart-radeon-pro\install` |
| ROCm / HIP | TheRock `7.14.0rc3`; HIP headers `7.14.60850`, hash `2b22ab01`; ROCm clang 23.0.0 |
| HART device | `gfx1201` |
| CUDA | SDK 12.9.1, nvcc 12.9.86, cudart 12.9.79 |
| OptiX | 8.0.0 (`OPTIX_VERSION=80000`), PTX target `sm_60` |
| OIIO | Arnold package `autodesk-arnold-2.6.3.2-1-3`; exported CMake version 2.6.3.0 |
| Imath / OpenEXR | 3.1.10 / Arnold 3.3.4 |
| JPEG replacement | Official libjpeg-turbo 3.1.0, libjpeg ABI 62, SIMD with NASM 3.01, `/MD` |
| Other image dependencies | libdeflate 1.24, PNG 1.6.55, TIFF 4.7.0, zlib 1.3.1, Freetype 2.13.3 |
| pugixml | 1.15, from the existing `x64-windows` prefix |

The selected OSL LLVM supplies **all host LLVM and Clang archives and bitcode
tools**. ROCm is a separate SDK for HIP headers, runtime, and device libraries.
Never add Arnold's LLVM 20 archives to an executable containing these LLVM 23
archives. Rebuild the consuming integration against a consistent LLVM 23
toolchain rather than treating these OSL archives as a drop-in LLVM 20 package.
Likewise, do not substitute OIIO 3 headers for the staged OIIO 2.6 headers.

Development version labels such as `23.0.0git` and HART `0.1.0` do not identify
unique builds. The selected OSL Clang reports LLVM source commit
`46fcb339fb61119b337f973c7ca9e710a319fdd0`; the distinct ROCm Clang reports the
same base plus `PATCHED:1efe4605317a40d2170d33617075bd5675d0d558`.
The dependency manifest preserves both complete `--version` outputs and
executable paths. It also records paths and SHA256 fingerprints of the
installed SDK HART and HIP runtime DLLs selected by CMake's generated deployment
rules. These fingerprint the actual selected binaries, not source repositories:
never infer that a nearby checkout's `HEAD` produced an installed DLL. Re-run
the manifest generator after SDK replacement and before staging.

## JPEG CRT conflict

Read-only inspection of Arnold's
`libjpeg-turbo\3.1.0-2\lib\jpeg-static.lib` found 102 COFF directives requesting
`/DEFAULTLIB:LIBCMT` (`/MT`). Linking this into the `/MD` profile causes LNK4098.
Do **not** hide it with `/NODEFAULTLIB` or `/IGNORE:4098`, and do not switch the
rest of OSL to `/MT`.

No local 3.1.0 source archive was found. An initial local vcpkg 3.1.4.1 ABI-62
replacement passed 211 tests, but the selected replacement now uses the
**official 3.1.0 source release**, matching Arnold's JPEG version without
touching its packages. The helper explicitly keeps `WITH_JPEG7=OFF`,
`WITH_JPEG8=OFF`, `WITH_CRT_DLL=ON`, and
`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`. It uses version-specific build
directories and verifies that the resulting archive requests MSVCRT, not
LIBCMT. All 211 upstream tests passed for the selected 3.1.0 build, and its
`jpeglib.h`, `jmorecfg.h`, and `jerror.h` match Arnold's public headers after
newline normalization. The helper configures explicitly and disables automatic
regeneration to avoid older JPEG Visual Studio projects racing over
`generate.stamp` during a parallel build. The official source archive SHA256 is
`9564c72b1dfd1d6fe6274c5f95a8d989b59854575d4bbee44ade7bc17aa9bc93`.

## Static link and package exports

Configuration writes `build\hart-arnold\arnold-dependencies.json` using CMake's
**actual Release codemodel**. It records SDK/header versions, roots,
architectures, complete compiler build identities, HART/HIP runtime and JPEG
binary hashes, and the complete ordered transitive link
fragments for every executable/shared target. Relative paths are resolved from
each target's recorded `build_directory`. The manifest is installed into
`share\OSL`; regenerate it by rerunning the profile helper after changing
dependencies. It is the authoritative full library list, not a manually
maintained abbreviated link command.

In addition to the OSL libraries, the configured `testshade` link contains:

* All selected LLVM 23 archives and the Clang frontend/driver/parser/AST/support
  archives needed by `oslcomp` (including LLVM 23's additional Clang libraries).
* `OpenImageIO`, `OpenImageIO_Util`, `Imath-3_1`, `OpenEXR-3_3`,
  `OpenEXRCore-3_3`, `Iex-3_3`, `IlmThread-3_3`, `deflatestatic`,
  `libpng16_static`, `tiff`, `zlibstatic`, the isolated `jpeg-static`,
  `freetype`, and `pugixml`.
* For standalone consumers, Arnold's real `ai.lib` / `ai.dll` supplies its
  OIIO allocator symbols. No allocator shim is supplied. When linking Arnold
  itself with SCons, its own objects provide these symbols: do not add a
  dependency on a previously built copy of Arnold merely to resolve them.
* `amd.hart.lib`, `amdhip64.lib`, and CUDA `cudart.lib`. OptiX 8 uses SDK
  headers and the driver function-table loader, not a separate OptiX library.
* Windows system libraries: `psapi`, `shell32`, `ole32`, `uuid`, `advapi32`,
  `ws2_32`, `ntdll`, `Version`, and CMake's platform defaults (`kernel32`,
  `user32`, `gdi32`, `winspool`, `oleaut32`, `comdlg32`).

The CUDA runtime remains dynamic: CUDA 12.9's `cudart_static.lib` also requests
LIBCMT and must not be substituted into this `/MD` profile. The selected
`cudart64_12.dll` is copied next to built OptiX targets and installed into
`bin`, alongside the existing HART runtime deployment. The NVIDIA driver
(including `nvoptix.dll`) is needed only for NVIDIA execution; it is not
redistributed in the OSL package. CPU and HART execution do not require an
NVIDIA device or driver.

Use `find_package(OSL CONFIG REQUIRED)` and imported `OSL::` targets rather than
copying that abbreviated list. Static exports discover pugixml and, when HART
test tools are exported, HIP and HART; optional Partio also discovers ZLIB.
The staged OIIO configuration discovers OpenEXR and Freetype and exports the
real Arnold runtime target. Supply the same isolated dependency prefixes plus
HART and ROCm to downstream `CMAKE_PREFIX_PATH`. The exports retain absolute
paths to selected LLVM and SDK libraries: the OSL install alone is **not** a
self-contained redistributable SDK.

For SCons, compile the OSL-facing code as C++17 with `/MD`,
`OSL_STATIC_DEFINE=1`, and `OIIO_STATIC_DEFINE=1`, as propagated by the actual
imported targets. Use this installation's OSL headers and the matching staged
OIIO 2.6/Imath headers. The generated OSL configuration header supplies the
Arnold profile macro; do not define it independently for another build.
Use the manifest's resolved LLVM 23 link inputs and remove the old LLVM 20
include/library set together, rather than appending the new archives to it.

Shared-package consumers do not acquire the private HART dependency.
The `cmake-package-export` test covers shared, static, static HART, and static
Partio package discovery without needing a GPU or building OSL.
