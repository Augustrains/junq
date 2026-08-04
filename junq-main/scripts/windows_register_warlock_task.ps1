param(
    [string]$TaskName = "AFSIM-Warlock",
    [string]$WarlockPath = "D:\junq\afsim_work\afsim-2.9.0-win64_bin\bin_release\warlock.exe",
    [string]$WorkingDirectory = "D:\junq\afsim_work\afsim-2.9.0-win64_bin\demos\air_to_air",
    [string]$ScenarioFile = "scenarios\island_assault_linux_train.txt"
)

$ErrorActionPreference = "Stop"
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an Administrator PowerShell window."
}

foreach ($path in @($WarlockPath, (Join-Path $WorkingDirectory $ScenarioFile))) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required file does not exist: $path"
    }
}

$arguments = '-log-server-host localhost -log-server-port 18888 "{0}"' -f $ScenarioFile
$action = New-ScheduledTaskAction `
    -Execute $WarlockPath `
    -Argument $arguments `
    -WorkingDirectory $WorkingDirectory
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew
$userId = "{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME
$taskPrincipal = New-ScheduledTaskPrincipal `
    -UserId $userId `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Settings $settings `
    -Principal $taskPrincipal `
    -Description "Start Warlock with the Linux-training UDP scenario." `
    -Force | Out-Host

Write-Host "Registered $TaskName for $userId"
Write-Host "Scenario: $(Join-Path $WorkingDirectory $ScenarioFile)"
