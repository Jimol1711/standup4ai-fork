param(
    [string]$DataRoot = "",
    [ValidateSet('cpu', 'cuda:0')][string]$Device = 'cpu'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $DataRoot) { $DataRoot = Join-Path $projectRoot 'data\standup' }
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null
$pidFile = Join-Path $DataRoot 'pipeline.pid'
if (Test-Path -LiteralPath $pidFile) {
    $previousPid = [int](Get-Content -LiteralPath $pidFile)
    if (Get-Process -Id $previousPid -ErrorAction SilentlyContinue) {
        throw "Dataset pipeline already running with PID $previousPid"
    }
}
$python = Join-Path $projectRoot 'venv\Scripts\python.exe'
$pipelineFile = Join-Path $projectRoot 'run_dataset.py'
$arguments = @('-u', ('"' + $pipelineFile + '"'), '--data-root', ('"' + $DataRoot + '"'), '--device', $Device)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $DataRoot 'pipeline.stdout.log') -RedirectStandardError (Join-Path $DataRoot 'pipeline.stderr.log')
Set-Content -LiteralPath $pidFile -Value $process.Id
Write-Output "Dataset pipeline started: PID $($process.Id)"
Write-Output "Data and logs: $DataRoot"
