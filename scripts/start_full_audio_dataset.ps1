$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputFolder = Join-Path $projectRoot 'data\audio_segments'
New-Item -ItemType Directory -Force -Path $outputFolder | Out-Null
$pidFile = Join-Path $outputFolder 'full_pipeline.pid'
if (Test-Path -LiteralPath $pidFile) {
    $recordedProcessId = [int](Get-Content -LiteralPath $pidFile)
    $previous = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedProcessId"
    if ($previous -and $previous.CommandLine -like '*finish_dataset.py*') {
        throw "Full dataset job is already running (PID $recordedProcessId)"
    }
}
$arguments = @('-u', ('"' + (Join-Path $projectRoot 'finish_dataset.py') + '"'))
$job = Start-Process -FilePath (Join-Path $projectRoot 'venv\Scripts\python.exe') -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputFolder 'full_pipeline.stdout.log') -RedirectStandardError (Join-Path $outputFolder 'full_pipeline.log')
Set-Content -LiteralPath $pidFile -Value $job.Id
Write-Output "Full audio dataset job started (PID $($job.Id))"
Write-Output "Download log: $outputFolder\download.log"
