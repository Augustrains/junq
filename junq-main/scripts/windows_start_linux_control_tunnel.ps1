param(
    [string]$LinuxHost = "gzc133",
    [int]$LinuxForwardPort = 2222
)

$ErrorActionPreference = "Stop"
$signature = "127.0.0.1:{0}:127.0.0.1:22" -f $LinuxForwardPort
$existing = Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*$signature*" }
if ($existing) {
    Write-Host "Linux-to-Windows control tunnel is already running (PID $($existing.ProcessId -join ','))."
    exit 0
}

$arguments = @(
    "-N",
    "-T",
    "-o", "BatchMode=yes",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-R", $signature,
    $LinuxHost
)
$process = Start-Process `
    -FilePath "$env:WINDIR\System32\OpenSSH\ssh.exe" `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -PassThru
Start-Sleep -Seconds 2
if ($process.HasExited) {
    throw "SSH tunnel exited immediately with code $($process.ExitCode). Verify the '$LinuxHost' SSH alias and key."
}
Write-Host "Linux-to-Windows control tunnel started (PID $($process.Id), Linux 127.0.0.1:$LinuxForwardPort -> Windows 127.0.0.1:22)."
