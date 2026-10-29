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
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][string]$IndexUrl
)

$ErrorActionPreference = 'Stop'
$compilerRoot = [System.IO.Path]::GetFullPath($Destination)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$pythonTag = 'cpython-3.13.17-windows-x86_64-none'
& uv python install $pythonTag --install-dir (Join-Path $compilerRoot 'python') --no-bin --no-registry --cache-dir (Join-Path $compilerRoot 'uv-cache')
if ($LASTEXITCODE -ne 0) { throw 'Compiler Python download failed' }
$compilerBase = Join-Path $compilerRoot "python/$pythonTag/python.exe"
$compilerEnv = Join-Path $compilerRoot 'venv'
if (-not (Test-Path -LiteralPath (Join-Path $compilerEnv 'Scripts/python.exe'))) {
    & $compilerBase -m venv $compilerEnv
    if ($LASTEXITCODE -ne 0) { throw 'Compiler venv creation failed' }
}
$compilerPython = Join-Path $compilerEnv 'Scripts/python.exe'
& uv pip install --python $compilerPython --index-url $IndexUrl --only-binary ':all:' --cache-dir (Join-Path $compilerRoot 'uv-cache') -r (Join-Path $repoRoot 'requirements/windows-cutedsl-compiler-x64.txt')
if ($LASTEXITCODE -ne 0) { throw 'CuTe compiler dependency installation failed' }
& $compilerPython -I -c 'import sys; import cutlass.cute; print(sys.version); print(cutlass.cute.__file__)'
if ($LASTEXITCODE -ne 0) { throw 'Windows CuTe compiler import failed' }
Write-Output "Compiler Python: $compilerPython"
