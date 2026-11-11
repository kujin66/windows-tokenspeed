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

# Windows ARM64 compiler environment shared by CPU and CUDA CI.
param([Parameter(Mandatory)][ValidateSet('cpu', 'cuda')][string]$Profile)

$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -property installationPath
if (-not $vs) { throw 'Visual Studio with ARM64 C++ tools is required' }
# VS2022's PowerShell wrapper rejects HostArch=arm64; the batch
# entrypoint supports the native ARM64 host and target environment.
$vcvarsall = Join-Path $vs 'VC/Auxiliary/Build/vcvarsall.bat'
# A prepared runner may point VCINSTALLDIR at a portable toolset.
# Let the installed activation script discover its own VC tree first.
$compilerEnv = & $env:ComSpec /d /c ('set "VCINSTALLDIR=" && set "VCToolsInstallDir=" && set "VCToolsVersion=" && call "{0}" arm64 && set' -f $vcvarsall)
if ($LASTEXITCODE -ne 0) {
  $compilerEnv | Write-Output
  throw 'Native ARM64 compiler environment initialization failed'
}
foreach ($line in $compilerEnv) {
  if ($line -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
  }
}
# ring uses Clang for its Windows ARM64 assembly sources.
$llvm = Join-Path $vs 'VC/Tools/Llvm/ARM64/bin'
if (Test-Path (Join-Path $llvm 'clang.exe')) { $env:PATH = $llvm + ';' + $env:PATH }
if ($Profile -eq 'cuda' -and -not (Get-Command clang.exe -ErrorAction SilentlyContinue)) { throw 'LLVM clang.exe is required by ring on Windows ARM64; add its bin directory to PATH' }
if ($env:TOKENSPEED_MSVC_ROOT) {
  $installedMsvc = $env:VCToolsInstallDir.TrimEnd([char]92)
  $preparedMsvc = $env:TOKENSPEED_MSVC_ROOT.TrimEnd([char]92)
  if (-not (Test-Path (Join-Path $preparedMsvc 'bin/Hostarm64/arm64/cl.exe'))) { throw 'Prepared ARM64 MSVC compiler is missing' }
  foreach ($name in @('PATH', 'INCLUDE', 'LIB', 'LIBPATH')) {
    $value = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $value.Replace($installedMsvc, $preparedMsvc), 'Process')
  }
  $env:VCToolsInstallDir = $preparedMsvc + [char]92
  # Vendor JIT discovery uses VCINSTALLDIR + Tools/MSVC + version.
  # Keep those variables consistent with the selected compiler.
  $env:VCToolsVersion = Split-Path $preparedMsvc -Leaf
  $env:VCINSTALLDIR = (Get-Item -LiteralPath $preparedMsvc).Parent.Parent.Parent.FullName + [char]92
}
$nativeTools = Join-Path $env:VCToolsInstallDir 'bin/Hostarm64/arm64'
if (-not (Test-Path (Join-Path $nativeTools 'cl.exe'))) { throw 'Native ARM64 MSVC host tools are required' }
$env:PATH = $nativeTools + ';' + $env:PATH
$env:VSCMD_ARG_HOST_ARCH = 'arm64'
$env:VSCMD_ARG_TGT_ARCH = 'arm64'
if ($Profile -eq 'cuda' -and (-not $env:OPENSSL_LICENSE_FILE -or -not (Test-Path -LiteralPath $env:OPENSSL_LICENSE_FILE))) { throw 'OPENSSL_LICENSE_FILE must identify the external OpenSSL license' }
if ($Profile -eq 'cuda' -and -not $env:OPENSSL_ROOT_DIR) { throw 'OPENSSL_ROOT_DIR must identify the prepared ARM64 static OpenSSL /MD install' }
foreach ($item in (Get-ChildItem Env:)) {
  if ($item.Name -in @('PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'VCToolsInstallDir', 'VCToolsVersion', 'VCINSTALLDIR', 'WindowsSdkDir', 'WindowsSDKVersion', 'WindowsSDKLibVersion', 'UniversalCRTSdkDir', 'UCRTVersion', 'OPENSSL_ROOT_DIR', 'VSCMD_ARG_HOST_ARCH', 'VSCMD_ARG_TGT_ARCH')) {
    ('{0}={1}' -f $item.Name, $item.Value) >> $env:GITHUB_ENV
  }
}
