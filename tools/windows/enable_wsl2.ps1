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

param([Parameter(Mandatory=$true)][string]$ReportDirectory)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $ReportDirectory | Out-Null
try {
    $before = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    $result = Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
    [ordered]@{before=$before.State.ToString(); restart_needed=$result.RestartNeeded; after=(Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State.ToString(); time=(Get-Date -Format o)} | ConvertTo-Json | Out-File -LiteralPath (Join-Path $ReportDirectory 'virtual-machine-platform.json') -Encoding utf8
} catch {
    @{error=$_.Exception.Message} | ConvertTo-Json | Out-File -LiteralPath (Join-Path $ReportDirectory 'virtual-machine-platform.json') -Encoding utf8
    exit 1
}
