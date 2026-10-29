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

param(
    [Parameter(Mandatory=$true)][string]$SourceRepo,
    [Parameter(Mandatory=$true)][string]$RuntimeSitePackages,
    [Parameter(Mandatory=$true)][string]$NativeWheelhouse
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$source = (Resolve-Path -LiteralPath $SourceRepo).Path
if ($root -eq $source) { throw 'Source and destination must differ' }
$runtime = (Resolve-Path -LiteralPath $RuntimeSitePackages).Path
$wheelhouse = (Resolve-Path -LiteralPath $NativeWheelhouse).Path
$python = Join-Path $source '.venv/Scripts/python.exe'
$targetPython = Join-Path $root '.venv/Scripts/python.exe'
# Reuse dependencies explicitly; never copy a non-relocatable venv.
if (-not (Test-Path -LiteralPath $targetPython)) {
    & $python -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $root '.venv/Lib/site-packages/local-runtime.pth'), $runtime + [Environment]::NewLine, $utf8)
& $python (Join-Path $root 'tools/windows/copy_python_dependencies.py') --source-site (Join-Path $source '.venv/Lib/site-packages') --target-site (Join-Path $root '.venv/Lib/site-packages') --manifest (Join-Path $root '.port/python-dependencies.json')
if ($LASTEXITCODE -ne 0) { throw 'Python dependency snapshot failed' }
foreach ($part in @('.port/msvc', '.port/rust-toolchain', '.port/xgrammar-0.2.8', '.port/cargo/registry', '.port/triton', '.port/flashinfer', '.port/autotune', '.port/models/Qwen3-0.6B')) {
    $from = Join-Path $source $part
    $to = Join-Path $root $part
    if (-not (Test-Path -LiteralPath $to)) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $to) | Out-Null
        Copy-Item -LiteralPath $from -Destination $to -Recurse
    }
}
New-Item -ItemType Directory -Force (Join-Path $root '.port/downloads'), (Join-Path $root '.port/wheelhouse') | Out-Null
$auditPath = Join-Path $root 'docs/validation/windows-arm64-fp8-20261027/inventory/download-cache-audit.json'
$audit = Get-Content -Raw -Encoding UTF8 -LiteralPath $auditPath | ConvertFrom-Json
$records = @()
foreach ($archive in $audit.archives) {
    if ($archive.filename -like '*Store*') { continue }
    $from = Join-Path $source ('.port/downloads/' + $archive.filename)
    $to = Join-Path $root ('.port/downloads/' + $archive.filename)
    if (-not (Test-Path -LiteralPath $to)) { Copy-Item -LiteralPath $from -Destination $to }
    $hash = (Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $archive.expected_sha256) { throw "Archive hash mismatch: $to" }
    $records += [ordered]@{path=('.port/downloads/' + $archive.filename); sha256=$hash; source=$archive.public_source; verified_against_manifest=$true}
}
foreach ($filename in @('torch-2.14.0a0+git20b0f57-cp313-cp313-win_arm64.whl', 'apache_tvm_ffi-0.1.13.post3-cp313-cp313-win_arm64.whl')) {
    $from = Join-Path $wheelhouse $filename
    $to = Join-Path $root ('.port/wheelhouse/' + $filename)
    if (-not (Test-Path -LiteralPath $to)) { Copy-Item -LiteralPath $from -Destination $to }
    $hash = (Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne (Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash.ToLowerInvariant()) { throw "Wheel copy mismatch: $filename" }
    $records += [ordered]@{path=('.port/wheelhouse/' + $filename); sha256=$hash; source='local ARM64 wheelhouse'; verified_against_manifest=$false}
}
[IO.File]::WriteAllText((Join-Path $root '.port/local-dependencies.json'), ($records | ConvertTo-Json -Depth 6), $utf8)
Write-Output 'Prepared local toolchains, archive hashes, ARM64 wheels, model and project venv.'
Write-Output 'Runtime dependencies are shared via local-runtime.pth; this is not a clean-machine installation.'
