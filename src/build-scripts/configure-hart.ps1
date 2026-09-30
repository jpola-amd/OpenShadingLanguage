# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause

#requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet("Current", "Arnold")]
    [string]$DependencyProfile = "Current",
    [string]$Dependencies = "D:\OSL\dependencies",
    [string]$ArnoldDependencies = "D:\autodesk\arnold-core\build\windows\dependencies",
    [string]$ArnoldRoot = "D:\autodesk\arnold-core\build\windows\x86_64\icx_dev\dist",
    [string]$LLVMRoot = "D:\OSL\LLVM\llvm-23.0.0-rocm-install",
    [string]$HartRoot = "D:\hart-repos\hart-radeon-pro\install",
    [string]$RocmRoot = $env:ROCM_PATH,
    [string]$Architectures = "gfx1201",
    [string]$CudaRoot = "",
    [string]$OptixRoot = "",
    [string]$CudaArchitecture = "sm_60",
    [string]$JpegRoot = "",
    [switch]$Build,
    [switch]$Install,
    [switch]$SmokeTest,
    [switch]$Test
)

$ErrorActionPreference = "Stop"
if (-not $RocmRoot) { $RocmRoot = $env:ROCM_ROOT }
if (-not $RocmRoot) { $RocmRoot = $env:HIP_PATH }
$source = (Resolve-Path "$PSScriptRoot\..\..").Path
$profile = $DependencyProfile.ToLowerInvariant()
$buildDir = "$source\build\hart-$profile"
$installDir = "$source\install\hart-$profile"
$current = (Resolve-Path "$Dependencies\x64-windows").Path
$prefixes = @($current)
$argsCmake = @(
    "-S", $source, "-B", $buildDir, "-G", "Visual Studio 17 2022", "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CONFIGURATION_TYPES=Release",
    "-DCMAKE_INSTALL_PREFIX=$installDir",
    "-DCMAKE_VS_GLOBALS=VcpkgEnabled=false",
    "-DLLVM_ROOT=$LLVMRoot", "-DLLVM_DIRECTORY=$LLVMRoot",
    "-DFLEX_EXECUTABLE=$current\bin\win_flex.exe",
    "-DBISON_EXECUTABLE=$current\bin\win_bison.exe",
    "-DUSE_QT=OFF", "-DUSE_PYTHON=OFF", "-DUSE_PARTIO=OFF",
    "-DUSE_LLVM_BITCODE=ON", "-DSTOP_ON_WARNING=OFF",
    "-DOSL_USE_HART=ON", "-DHART_TARGET_ARCHITECTURES=$Architectures",
    "-DHART_ROOT=$HartRoot"
)
if ($RocmRoot) {
    $argsCmake += "-DROCM_ROOT=$RocmRoot"
}

if ($DependencyProfile -eq "Arnold") {
    $llvmVersion = & "$LLVMRoot\bin\llvm-config.exe" --version
    if ($LASTEXITCODE -ne 0 -or $llvmVersion -notmatch '^23\.') {
        throw "The Arnold profile requires OSL LLVM 23; do not mix LLVM 20 archives."
    }
    if (-not $CudaRoot) { $CudaRoot = "$Dependencies\cuda\12.9" }
    if (-not $OptixRoot) { $OptixRoot = "$Dependencies\optix\8.0.0" }
    if (-not $JpegRoot) { $JpegRoot = "$buildDir\dependencies\jpeg-md" }
    foreach ($required in @("$CudaRoot\bin\nvcc.exe", "$OptixRoot\include\optix.h",
                            "$JpegRoot\lib\jpeg-static.lib", "$JpegRoot\include\jpeglib.h")) {
        if (-not (Test-Path $required)) {
            throw "Missing $required. Build an isolated /MD JPEG with build-arnold-jpeg.ps1, or select -JpegRoot."
        }
    }
    $jpegDirectives = & "$LLVMRoot\bin\llvm-readobj.exe" --coff-directives "$JpegRoot\lib\jpeg-static.lib"
    if ($LASTEXITCODE -ne 0 -or ($jpegDirectives -match 'DEFAULTLIB:.*LIBCMT') -or
        -not ($jpegDirectives -match 'DEFAULTLIB:.*MSVCRT')) {
        throw "Arnold's OSL profile requires a /MD JPEG archive. Do not suppress LNK4098."
    }
    $stage = "$buildDir\dependencies"
    & python "$PSScriptRoot\prepare-arnold-deps.py" $ArnoldDependencies $stage `
        --arnold-root $ArnoldRoot
    if ($LASTEXITCODE -ne 0) { throw "Arnold dependency preparation failed" }
    $oiio = "$stage\openimageio"
    $imath = "$stage\imath"
    $packages = @(
        "openexr\3.3.4-arnold-dev1-2", "libdeflate\1.24-2",
        "libpng\1.6.55-0", "libtiff\4.7.0-2", "libjpeg-turbo\3.1.0-2",
        "zlib\1.3.1-9", "freetype\2.13.3-2"
    )
    $prefixes = @($oiio, $imath, $JpegRoot)
    foreach ($package in $packages) {
        $prefixes += (Resolve-Path "$ArnoldDependencies\$package").Path
    }
    $prefixes += $current
    $argsCmake += @(
        "-DBUILD_SHARED_LIBS=OFF",
        "-DOSL_ARNOLD_COMPAT=ON",
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL",
        "-DOSL_USE_OPTIX=ON",
        "-DCUDA_TOOLKIT_ROOT_DIR:PATH=$CudaRoot",
        "-DCUDA_TARGET_ARCH=$CudaArchitecture",
        "-DCUDA_PREFER_STATIC_LIBS=OFF",
        "-U", "cudart_lib",
        "-DOptiX_ROOT=$OptixRoot",
        "-DOPTIX_INCLUDE_DIR=$OptixRoot\include",
        "-DOSL_EXTRA_NVCC_ARGS=-DFMT_UNICODE=0",
        "-DLLVM_STATIC=ON",
        "-U", "_CLANG_*", "-U", "LLVM_LIBRARY", "-U", "LLVM_MCJIT_LIBRARY",
        "-DLLVM_CONFIG=$LLVMRoot\bin\llvm-config.exe",
        "-DLLVM_BC_GENERATOR=$LLVMRoot\bin\clang++.exe",
        "-DLLVM_AS_TOOL=$LLVMRoot\bin\llvm-as.exe",
        "-DLLVM_LINK_TOOL=$LLVMRoot\bin\llvm-link.exe",
        "-DLLVM_OPT_TOOL=$LLVMRoot\bin\opt.exe",
        "-DLLVM_LLC_TOOL=$LLVMRoot\bin\llc.exe",
        "-DOSL_DEPENDENCY_MANIFEST:FILEPATH=$buildDir\arnold-dependencies.json",
        "-DOSL_ALLOW_OIIO_26=ON",
        "-DOpenImageIO_ROOT=$oiio",
        "-DOpenImageIO_DIR=$oiio\lib\cmake\OpenImageIO",
        "-DImath_DIR=$imath\lib\cmake\Imath",
        "-DZLIB_INCLUDE_DIR=$ArnoldDependencies\zlib\1.3.1-9\include",
        "-DZLIB_LIBRARY_RELEASE=$ArnoldDependencies\zlib\1.3.1-9\lib\zlibstatic.lib",
        "-DPNG_PNG_INCLUDE_DIR=$ArnoldDependencies\libpng\1.6.55-0\include",
        "-DPNG_LIBRARY_RELEASE=$ArnoldDependencies\libpng\1.6.55-0\lib\libpng16_static.lib",
        "-DTIFF_INCLUDE_DIR=$ArnoldDependencies\libtiff\4.7.0-2\include",
        "-DTIFF_LIBRARY_RELEASE=$ArnoldDependencies\libtiff\4.7.0-2\lib\tiff.lib",
        "-DJPEG_INCLUDE_DIR=$JpegRoot\include",
        "-DJPEG_LIBRARY_RELEASE=$JpegRoot\lib\jpeg-static.lib",
        "-DFREETYPE_INCLUDE_DIR_freetype2=$ArnoldDependencies\freetype\2.13.3-2\include\freetype2",
        "-DFREETYPE_INCLUDE_DIR_ft2build=$ArnoldDependencies\freetype\2.13.3-2\include\freetype2",
        "-DFREETYPE_LIBRARY_RELEASE=$ArnoldDependencies\freetype\2.13.3-2\lib\freetype.lib"
    )
    $oiioBin = "$oiio\bin;$ArnoldRoot\bin"
    $query = "$buildDir\.cmake\api\v1\query"
    New-Item -ItemType Directory -Force $query | Out-Null
    New-Item -ItemType File -Force "$query\codemodel-v2" | Out-Null
} else {
    $argsCmake += @(
        "-DBUILD_SHARED_LIBS=ON",
        "-DOSL_ARNOLD_COMPAT=OFF",
        "-DOSL_USE_OPTIX=OFF",
        "-DOSL_ALLOW_OIIO_26=OFF",
        "-DOpenImageIO_ROOT=",
        "-DOpenImageIO_DIR=$current\share\openimageio",
        "-DImath_DIR=$current\share\imath"
    )
    $oiioBin = "$current\tools\openimageio"
}
$argsCmake += "-DCMAKE_PREFIX_PATH=$($prefixes -join ';')"
$savedPath = $env:PATH
$savedTests = $env:TESTSUITE_HART
$savedHipPath = $env:HIP_PATH
try {
    $env:PATH = "$oiioBin;$current\bin;$savedPath"
    $env:TESTSUITE_HART = "1"
    if ($RocmRoot) { $env:HIP_PATH = $RocmRoot }
    & cmake @argsCmake
    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
    if ($DependencyProfile -eq "Arnold") {
        & python "$PSScriptRoot\write-arnold-manifest.py" $buildDir
        if ($LASTEXITCODE -ne 0) { throw "Dependency manifest generation failed" }
    }
    if ($Build -or $Install -or $SmokeTest -or $Test) {
        & cmake --build $buildDir --config Release --parallel 8
        if ($LASTEXITCODE -ne 0) { throw "OSL build failed" }
    }
    if ($SmokeTest -or $Test) {
        $runtimeTests = "hart-oiio-compat-smoke|arnold-compat-cpu"
        if ($Test) { $runtimeTests += "|hart-generated-runtime|hart-custom-closures-O[02]|hart-texture-colorspaces-O[02]" }
        & ctest --test-dir $buildDir -C Release --output-on-failure `
            -R "^(oiio-compat-.*|cmake-hart-discovery|cmake-package-export|hart-codegen-.*|hart-.*bitcode.*|$runtimeTests)$" `
            --timeout 600 --no-tests=error
        if ($LASTEXITCODE -ne 0) { throw "OSL tests failed" }
    }
    if ($Install) {
        & cmake --install $buildDir --config Release
        if ($LASTEXITCODE -ne 0) { throw "OSL installation failed" }
    }
} finally {
    $env:PATH = $savedPath
    $env:TESTSUITE_HART = $savedTests
    $env:HIP_PATH = $savedHipPath
}
