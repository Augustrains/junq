param(
    [switch]$SampleActions,
    [int]$MaxSteps = 5000,
    [int]$LogEvery = 10
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$checkpoint = Join-Path $projectRoot "checkpoints\windows_happo_45km\bottom_happo_recon_attack_parallel_eval_update_000025.pt"
$config = Join-Path $PSScriptRoot "afsim_demo_units.json"

if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "Best evaluated checkpoint not found: $checkpoint"
}

Set-Location $projectRoot
python -m show.demo_happo_warlock --checkpoint $checkpoint --config-path $config --device cpu --check-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$arguments = @(
    "-m", "show.demo_happo_warlock",
    "--checkpoint", $checkpoint,
    "--config-path", $config,
    "--device", "cpu",
    "--max-steps", $MaxSteps,
    "--log-every", $LogEvery,
    "--simulation-clock-rate", "60",
    "--bottom-decisions-per-hour", "50"
)
if ($SampleActions) { $arguments += "--sample-actions" }

python @arguments
exit $LASTEXITCODE
