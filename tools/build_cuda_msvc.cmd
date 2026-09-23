@echo off
rem Configure + build the CUDA-enabled tree with MSVC (nvcc needs cl.exe as host compiler on Windows).
rem Usage: tools\build_cuda_msvc.cmd [build-dir]
rem   NEXUSDATA_CUDA_ARCH   CMAKE_CUDA_ARCHITECTURES (default: native)
rem   NEXUSDATA_CMAKE_ARGS  extra configure arguments (cmd splits "=" in plain arguments)
setlocal
set "BUILD_DIR=%~1"
if "%BUILD_DIR%"=="" set "BUILD_DIR=build-cuda"
if not defined NEXUSDATA_CUDA_ARCH set "NEXUSDATA_CUDA_ARCH=native"

if defined VSCMD_VER goto :have_vs
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`call "%VSWHERE%" -latest -prerelease -products * -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (
    echo Visual Studio with C++ tools not found
    exit /b 1
)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
:have_vs

where cmake >nul 2>nul || set "PATH=%ProgramFiles%\JetBrains\CLion 2026.2.1\bin\cmake\win\x64\bin;%PATH%"
where ninja >nul 2>nul || set "PATH=%ProgramFiles%\JetBrains\CLion 2026.2.1\bin\ninja\win\x64;%PATH%"

cmake -S "%~dp0.." -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DNEXUSDATA_WITH_CUDA=ON ^
    "-DCMAKE_CUDA_ARCHITECTURES=%NEXUSDATA_CUDA_ARCH%" %NEXUSDATA_CMAKE_ARGS% || exit /b 1
cmake --build "%BUILD_DIR%" -j 8 || exit /b 1
