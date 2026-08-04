param(
    [int]$Port = 50160,
    [double]$ClockRate = 20,
    [string]$TaskNamePrefix = "AFSIM-ActionAudit-",
    [string]$UdpAddress = "127.0.0.1",
    [string]$SourceScenario = "aam_action3_air_target_test.txt"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "provision_parallel_warlock_tasks.ps1") `
    -Workers 1 -BasePort $Port -ClockRate $ClockRate -StateUpdateInterval 1 `
    -DecisionPauseInterval 0 -UdpAddress $UdpAddress -Runner warlock `
    -TaskPrefix $TaskNamePrefix -SourceScenario $SourceScenario
$task = "${TaskNamePrefix}0"
Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-ScheduledTask -TaskName $task
Write-Host "started $task as Warlock GUI on UDP $UdpAddress`:$Port scenario=$SourceScenario"