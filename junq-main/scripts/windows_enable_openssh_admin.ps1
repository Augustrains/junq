param(
    [Parameter(Mandatory = $false)]
    [string]$LinuxPublicKey = ""
)

$ErrorActionPreference = "Stop"
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an Administrator PowerShell window."
}

$capability = Get-WindowsCapability -Online -Name "OpenSSH.Server~~~~0.0.1.0"
if ($capability.State -ne "Installed") {
    Add-WindowsCapability -Online -Name "OpenSSH.Server~~~~0.0.1.0" | Out-Host
}

Set-Service -Name sshd -StartupType Automatic
Start-Service sshd

$firewallRule = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
if ($null -eq $firewallRule) {
    New-NetFirewallRule `
        -Name "OpenSSH-Server-In-TCP" `
        -DisplayName "OpenSSH Server (sshd)" `
        -Enabled True `
        -Direction Inbound `
        -Protocol TCP `
        -Action Allow `
        -LocalPort 22 | Out-Host
} else {
    Enable-NetFirewallRule -Name "OpenSSH-Server-In-TCP"
}

if ($LinuxPublicKey.Trim()) {
    $authorizedKeys = "C:\ProgramData\ssh\administrators_authorized_keys"
    $existing = @()
    if (Test-Path -LiteralPath $authorizedKeys) {
        $existing = @(Get-Content -LiteralPath $authorizedKeys)
    }
    if ($existing -notcontains $LinuxPublicKey.Trim()) {
        Add-Content -LiteralPath $authorizedKeys -Value $LinuxPublicKey.Trim() -Encoding ascii
    }
    & icacls.exe $authorizedKeys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F" | Out-Host
}

Get-Service sshd | Select-Object Name, Status, StartType
Write-Host "OpenSSH Server is ready on TCP port 22."
if (-not $LinuxPublicKey.Trim()) {
    Write-Host "No Linux public key was supplied; configure key authentication before unattended training."
}