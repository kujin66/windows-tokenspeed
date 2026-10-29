# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Exercise activation from an ordinary PowerShell terminal, without run.py.
. (Join-Path $PSScriptRoot 'Activate-Toolchain.ps1')
$tokenSpeedSource = Join-Path $PSScriptRoot 'cuda_smoke.cu'
$tokenSpeedExecutable = Join-Path $tokenSpeedRoot '.port/nvcc1451-activated-smoke.exe'
$tokenSpeedBuildLog = Join-Path $tokenSpeedRoot '.port/nvcc1451-activated-smoke-build.log'
& nvcc --use-local-env -std=c++17 -arch=sm_121 -Xcompiler /MD,/EHsc,/utf-8,/Zc:__cplusplus,/Zc:preprocessor $tokenSpeedSource -o $tokenSpeedExecutable -v 2>&1 | Out-File -Encoding utf8 $tokenSpeedBuildLog
if ($LASTEXITCODE -ne 0) {
    throw ('Activated-terminal nvcc build failed; see ' + $tokenSpeedBuildLog)
}
$tokenSpeedGpuJson = & $tokenSpeedExecutable
if ($LASTEXITCODE -ne 0) {
    throw 'Activated-terminal CUDA GPU validation failed.'
}
$tokenSpeedGpuResult = $tokenSpeedGpuJson | ConvertFrom-Json
if (-not $tokenSpeedGpuResult.passed -or $tokenSpeedGpuResult.msc_ver -ne 1951 -or $tokenSpeedGpuResult.msc_full_ver -ne 195136260 -or $tokenSpeedGpuResult.cuda_runtime_version -ne 13040 -or $tokenSpeedGpuResult.elements -ne 4097) {
    throw 'Activated-terminal compiler or CUDA result differs from the requested toolchain.'
}
$tokenSpeedGpuJson | Out-File -Encoding utf8 (Join-Path $tokenSpeedRoot '.port/nvcc1451-activated-smoke-result.json')
Write-Host 'PASS: terminal activation, direct MSVC 14.51 nvcc build and 4097 GPU results.'
