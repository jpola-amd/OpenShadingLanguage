cmake -S . -B .\build -G "Visual Studio 17 2022" -A x64 `
 -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake `
 -DVCPKG_TARGET_TRIPLET=x64-windows `
 -DVCPKG_INSTALLED_DIR=D:\OSL\dependencies `
 -DCMAKE_BUILD_TYPE=Release `
 -DCMAKE_INSTALL_PREFIX="./install" `
 -DLLVM_ROOT="D:\OSL\LLVM\llvm-23.0.0-rocm-install" `
 -DLLVM_DIRECTORY="D:\OSL\LLVM\llvm-23.0.0-rocm-install" `
 -DFLEX_EXECUTABLE=D:\OSL\dependencies\x64-windows\bin\win_flex.exe `
 -DBISON_EXECUTABLE=D:\OSL\dependencies\x64-windows\bin\win_bison.exe `
 -DUSE_QT=OFF `
 -DUSE_PYTHON=OFF `
 -DUSE_PARTIO=OFF `
 -DOSL_USE_OPTIX=OFF `
 -DUSE_LLVM_BITCODE=ON `
 -DSTOP_ON_WARNING=OFF `
 -DOSL_USE_HART=ON `
 -DHART_TARGET_ARCHITECTURES="gfx1201" `
 "-Damd.hart_ROOT=D:\hart-repos\amd-cae-hart\install"
 