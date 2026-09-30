# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause

#requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JpegSource,
    [Parameter(Mandatory = $true)]
    [string]$Nasm,
    [string]$LLVMRoot = "D:\OSL\LLVM\llvm-23.0.0-rocm-install"
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path "$PSScriptRoot\..\..").Path
$jpegSourcePath = (Resolve-Path $JpegSource).Path
$nasmPath = (Resolve-Path $Nasm).Path
$jpegCmake = Get-Content "$jpegSourcePath\CMakeLists.txt" -Raw
if ($jpegCmake -notmatch 'set\(VERSION ([0-9]+(?:\.[0-9]+)+)\)') {
    throw "Cannot identify the libjpeg-turbo source version"
}
$jpegVersion = $Matches[1]
$buildDir = "$source\build\hart-arnold\jpeg-$jpegVersion-md-build"
$installDir = "$source\build\hart-arnold\dependencies\jpeg-md"
# These outputs are deliberately outside Arnold and the input source tree.
# Configure explicitly to avoid concurrent VS regeneration of generate.stamp.
& cmake -S $jpegSourcePath -B $buildDir -G "Visual Studio 17 2022" -A x64 `
    "-DCMAKE_CONFIGURATION_TYPES=Release" "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_INSTALL_PREFIX=$installDir" "-DCMAKE_INSTALL_LIBDIR=lib" `
    "-DCMAKE_VS_GLOBALS=VcpkgEnabled=false" `
    "-DCMAKE_SUPPRESS_REGENERATION=ON" `
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL" "-DWITH_CRT_DLL=ON" `
    "-DENABLE_SHARED=OFF" "-DENABLE_STATIC=ON" "-DWITH_TURBOJPEG=OFF" `
    "-DWITH_JPEG7=OFF" "-DWITH_JPEG8=OFF" "-DWITH_SIMD=ON" `
    "-DCMAKE_ASM_NASM_COMPILER=$nasmPath"
if ($LASTEXITCODE -ne 0) { throw "JPEG configuration failed" }
& cmake --build $buildDir --config Release --parallel 8
if ($LASTEXITCODE -ne 0) { throw "JPEG build failed" }
& ctest --test-dir $buildDir -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "JPEG tests failed" }
& cmake --install $buildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "JPEG installation failed" }
$directives = & "$LLVMRoot\bin\llvm-readobj.exe" --coff-directives "$installDir\lib\jpeg-static.lib"
if ($LASTEXITCODE -ne 0) { throw "Cannot inspect JPEG CRT directives" }
if (($directives -match 'DEFAULTLIB:.*LIBCMT') -or
    -not ($directives -match 'DEFAULTLIB:.*MSVCRT')) {
    throw "JPEG must request the dynamic Release CRT (MSVCRT), not LIBCMT"
}
Write-Host "Verified /MD JPEG: $installDir"
