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

param([Parameter(Mandatory=$true)][string]$BuildId)
$ErrorActionPreference = 'Stop'
if ($BuildId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'BuildId must contain only letters, digits, underscore or hyphen' }
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $root
$py = Join-Path $root '.venv/Scripts/python.exe'
$env:PATH = (Join-Path $root '.port/rust-toolchain/bin') + ';' + (Join-Path $root '.venv/Scripts') + ';' + $env:PATH
$env:CARGO_HOME = Join-Path $root '.port/cargo'
$env:CARGO_NET_OFFLINE = 'true'
$env:PYO3_PYTHON = $py
$env:FETCHCONTENT_SOURCE_DIR_XGRAMMAR = Join-Path $root '.port/xgrammar-0.2.8'
$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_ARGS = '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release'
$env:PYTHONPATH = ''
$env:TOKENSPEED_KERNEL_BACKEND = 'cuda'
$env:TOKENSPEED_KERNEL_WINDOWS_PROFILE = 'dense-triton'
$env:TOKENSPEED_CUDA_ARCH = '121'
$env:TOKENSPEED_KERNEL_CUDA_FRONTEND_CCBIN = ''
$buildDir = Join-Path $root ('.port/scheduler-' + $BuildId)
New-Item -ItemType Directory -Force dist | Out-Null
function Invoke-Prepared {
    & $py tools/windows/run.py @args
    if ($LASTEXITCODE -ne 0) { throw "Build command failed ($LASTEXITCODE)" }
}
Invoke-Prepared $py -m pip wheel ./tokenspeed-scheduler --no-build-isolation --no-deps --wheel-dir dist --config-settings=build-dir=$buildDir
Invoke-Prepared $py -m pip wheel ./tokenspeed-engine --no-build-isolation --no-deps --wheel-dir dist
Invoke-Prepared $py -m pip wheel ./tokenspeed-kernel/python --no-build-isolation --no-deps --wheel-dir dist
Invoke-Prepared $py -m pip wheel ./python --no-build-isolation --no-deps --wheel-dir dist
# Refuse ambiguous directories rather than installing every historical wheel.
$wheels = @(Get-ChildItem -LiteralPath (Join-Path $root 'dist') -Filter '*.whl')
if ($wheels.Count -ne 4) { throw 'Expected exactly four project wheels in dist. Select/archive older builds explicitly.' }
foreach ($wheel in $wheels) {
    & $py -m pip install --no-deps --no-index --force-reinstall $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "Installation failed: $($wheel.Name)" }
}
Invoke-Prepared $py tools/windows/check_native_artifacts.py
Get-ChildItem dist/*.whl | Get-FileHash -Algorithm SHA256 | Format-Table -AutoSize
