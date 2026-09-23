@echo off
rem Usage: run_ctest.cmd [bench]  -- rebuild the CUDA tree in the MSVC env, then run ctest (or bench_gpu).
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -prerelease -products * -property installationPath`) do set "VSROOT=%%i"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%ProgramFiles%\JetBrains\CLion 2026.2.1\bin\cmake\win\x64\bin;%ProgramFiles%\JetBrains\CLion 2026.2.1\bin\ninja\win\x64;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%"
cmake --build "%~dp0." -j 8 >nul || exit /b 1
if "%~1"=="bench" ( "%~dp0bench_gpu.exe" & exit /b )
ctest --test-dir "%~dp0." -j 4 --output-on-failure
