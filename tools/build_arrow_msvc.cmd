@echo off
rem Configure + build the MSVC tree with Apache Arrow / Parquet (conda-forge).
rem Usage: tools\build_arrow_msvc.cmd [build-dir]
rem   NEXUSDATA_ARROW_PREFIX  CMake prefix of the env's Library dir
rem                           (default: %USERPROFILE%\.conda\envs\nexusdata-arrow\Library)
setlocal
set "BUILD_DIR=%~1"
if "%BUILD_DIR%"=="" set "BUILD_DIR=build-arrow"
if not defined NEXUSDATA_ARROW_PREFIX set "NEXUSDATA_ARROW_PREFIX=%USERPROFILE%\.conda\envs\nexusdata-arrow\Library"

rem cmd AutoRun may call doskey; keep System32 visible even if the parent PATH dropped it.
set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"

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
rem vcvars does not always put rc.exe / mt.exe on PATH. Point CMake at the newest kit.
set "RC_EXE="
set "MT_EXE="
for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles(x86)%\Windows Kits\10\bin\10.*" 2^>nul') do (
    if not defined RC_EXE if exist "%ProgramFiles(x86)%\Windows Kits\10\bin\%%D\x64\rc.exe" (
        set "RC_EXE=%ProgramFiles(x86)%\Windows Kits\10\bin\%%D\x64\rc.exe"
        set "MT_EXE=%ProgramFiles(x86)%\Windows Kits\10\bin\%%D\x64\mt.exe"
    )
)
if not defined RC_EXE (
    echo Windows SDK rc.exe not found
    exit /b 1
)
set "RC_EXE=%RC_EXE:\=/%"
set "MT_EXE=%MT_EXE:\=/%"
set "NEXUSDATA_ARROW_PREFIX=%NEXUSDATA_ARROW_PREFIX:\=/%"
rem vcvars on this machine does not add the Windows SDK. Link and includes need it.
for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles(x86)%\Windows Kits\10\Lib\10.*" 2^>nul') do (
    if not defined SDKVER if exist "%ProgramFiles(x86)%\Windows Kits\10\Lib\%%D\um\x64\kernel32.lib" set "SDKVER=%%D"
)
if not defined SDKVER (
    echo Windows SDK libraries not found
    exit /b 1
)
set "SDKROOT=%ProgramFiles(x86)%\Windows Kits\10"
set "INCLUDE=%INCLUDE%;%SDKROOT%\Include\%SDKVER%\ucrt;%SDKROOT%\Include\%SDKVER%\um;%SDKROOT%\Include\%SDKVER%\shared"
set "LIB=%LIB%;%SDKROOT%\Lib\%SDKVER%\ucrt\x64;%SDKROOT%\Lib\%SDKVER%\um\x64"

cmake -S "%~dp0.." -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DNEXUSDATA_WITH_ARROW=ON ^
    "-DCMAKE_RC_COMPILER=%RC_EXE%" "-DCMAKE_MT=%MT_EXE%" ^
    "-DCMAKE_PREFIX_PATH=%NEXUSDATA_ARROW_PREFIX%" || exit /b 1
cmake --build "%BUILD_DIR%" -j 8 --target nexusdata test_arrow test_v06 || exit /b 1
