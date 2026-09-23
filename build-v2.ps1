$ErrorActionPreference = "Continue"
$clion = "C:\Program Files\JetBrains\CLion 2026.2.1\bin"
$env:PATH = "$clion\mingw\bin;$clion\ninja\win\x64;" + $env:PATH
$cmake = "$clion\cmake\win\x64\bin\cmake.exe"
$ctest = "$clion\cmake\win\x64\bin\ctest.exe"
Set-Location $PSScriptRoot
if (-not (Test-Path build-v2\build.ninja)) {
    & $cmake -S . -B build-v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=c++ 2>&1 | Select-Object -Last 8
}
& $cmake --build build-v2 -j 8 2>&1 | Where-Object { $_ -match 'error|warning|FAILED|\[\d+/\d+\] Linking' } | Select-Object -First 60
if ($args -contains "test") {
    & $ctest --test-dir build-v2 --output-on-failure -E install_find_package_smoke 2>&1 | Select-Object -Last 40
}
