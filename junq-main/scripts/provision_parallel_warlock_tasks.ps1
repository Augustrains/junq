param(
    [int]$Workers = 4,
    [int]$BasePort = 50050,
    [double]$ClockRate = 60,
    [double]$StateUpdateInterval = 5,
    [string]$UdpAddress = "127.0.0.1",
    [ValidateSet("mission", "warlock")]
    [string]$Runner = "mission",
    [switch]$SkipTaskRegistration,
    [string]$TaskPrefix = "AFSIM-Warlock-",
    [string]$BaseTaskName = "AFSIM-Warlock",
    [string]$ScenarioDirectory = "D:\junq\afsim_work\afsim-2.9.0-win64_bin\demos\air_to_air\scenarios",
    [string]$SourceScenario = "island_assault_linux_train.txt"
)
$ErrorActionPreference = "Stop"
if ($Workers -lt 1) { throw "Workers must be positive" }
if ($ClockRate -le 0) { throw "ClockRate must be positive" }
if ($StateUpdateInterval -le 0) { throw "StateUpdateInterval must be positive" }
$base = if ($SkipTaskRegistration) { $null } else { Get-ScheduledTask -TaskName $BaseTaskName }
$sourcePath = Join-Path $ScenarioDirectory $SourceScenario
$source = [IO.File]::ReadAllText($sourcePath)
for ($worker = 0; $worker -lt $Workers; $worker++) {
    $port = $BasePort + $worker
    $scenarioName = "{0}.worker_{1}.txt" -f [IO.Path]::GetFileNameWithoutExtension($SourceScenario), $worker
    $scenarioPath = Join-Path $ScenarioDirectory $scenarioName
    $udpBlock = @(
        "udpnet"
        "   port $port"
        "   address $UdpAddress"
        "   state_update_interval $StateUpdateInterval sec"
        $(if ($DecisionPauseInterval -gt 0) { "   decision_pause_interval $DecisionPauseInterval sec" })
        "   recon_range 100000"
        "end_udpnet"
    ) -join "`n"
    $udpPattern = '(?ms)^udpnet\s*\r?\n.*?^end_udpnet\s*$'
    $rewritten = [regex]::Replace($source, $udpPattern, $udpBlock, 1)
    if ($rewritten -eq $source) { throw "Active udpnet block not found" }
    $clockPattern = '(?m)^\s*clock_rate\s+\S+.*$'
    $clockMatches = [regex]::Matches($rewritten, $clockPattern)
    if ($clockMatches.Count -ne 1) { throw "Expected exactly one clock_rate, found $($clockMatches.Count)" }
    $rewritten = [regex]::Replace($rewritten, $clockPattern, "clock_rate $ClockRate", 1)
    [IO.File]::WriteAllText($scenarioPath, $rewritten, [Text.UTF8Encoding]::new($false))
    $taskName = "$TaskPrefix$worker"
    if (-not $SkipTaskRegistration) {
        $executable = $base.Actions[0].Execute
        $arguments = '-log-server-host localhost -log-server-port 18888 "scenarios\' + $scenarioName + '"'
        if ($Runner -eq "mission") {
            $executable = Join-Path ([IO.Path]::GetDirectoryName($executable)) "mission.exe"
            if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
                throw "mission.exe not found next to the base Warlock executable: $executable"
            }
            $arguments = '"scenarios\' + $scenarioName + '"'
        }
        $action = New-ScheduledTaskAction -Execute $executable -Argument $arguments -WorkingDirectory $base.Actions[0].WorkingDirectory
        Register-ScheduledTask -TaskName $taskName -Action $action -Principal $base.Principal -Settings $base.Settings -Description "Parallel AFSIM $Runner worker $worker UDP $port" -Force | Out-Null
    }
    $verb = if ($SkipTaskRegistration) { "generated" } else { "created" }
    Write-Host "$verb $taskName -> $scenarioName -> UDP ${UdpAddress}:$port clock_rate=$ClockRate state_update_interval=${StateUpdateInterval}s runner=$Runner"
}