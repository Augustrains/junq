Set-Location 'D:\junq\dppo'
[Environment]::CurrentDirectory = 'D:\junq\dppo'
function Write-RewardJson([string]$Path, [object]$Value) {
  New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
  $json=$Value | ConvertTo-Json -Depth 12
  [IO.File]::WriteAllText($Path,$json+"`n",[Text.UTF8Encoding]::new($false))
}
$rulesPath='envs\reward_rules.json'
$backupDir='test\reward_rules_legacy_backup'
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
$backup=Join-Path $backupDir 'reward_rules.before_split_20260731.json'
if(Test-Path -LiteralPath $backup){ Remove-Item -LiteralPath $backup -Force }
Copy-Item -LiteralPath $rulesPath -Destination $backup
Write-RewardJson 'envs\rewards\global\rules.json' ([ordered]@{
  description='Shared team reward, combat value, milestones, limits, and reward switches.'
  negative_rewards_enabled=$false
  delta_rewards=[ordered]@{landing_window_opened=25.0;capture_condition_reached=50.0}
  combat_positive_rewards=[ordered]@{
    description='Target-value-aware shared combat shaping.'
    recon_contributor_fraction=0.2
    by_target_role=[ordered]@{
      radar=[ordered]@{damage_per_full_hp=10.0;destroyed=10.0}
      sam=[ordered]@{damage_per_full_hp=8.0;destroyed=8.0}
      attack_aircraft=[ordered]@{damage_per_full_hp=6.0;destroyed=6.0}
      ground_force=[ordered]@{damage_per_full_hp=2.0;destroyed=2.0}
    }
  }
  limits=[ordered]@{min_reward=-100.0;max_reward=100.0}
})
Write-RewardJson 'envs\rewards\recon\rules.json' ([ordered]@{event_rewards=[ordered]@{recon_task_detection=0.0;maintained_detection=0.0;new_detection=1.0;wait=-0.01}})
Write-RewardJson 'envs\rewards\attack\rules.json' ([ordered]@{
  event_rewards=[ordered]@{invalid_action=-2.0;udp_not_ready=-2.0;task_rejected=-1.0;platform_not_ready=-1.0;attack_unknown_target=-1.0;attack_target_not_known=-1.0;attack_failed=-1.0}
  attack_result_rewards=[ordered]@{NO_ENEMY=-1.0;NO_COMPATIBLE_WEAPON=-1.0;AMMO_EMPTY=-1.0}
})
Write-RewardJson 'envs\rewards\landing\rules.json' ([ordered]@{event_rewards=[ordered]@{landing_action_masked=-1.0;retreat_not_ready=-1.0}})
Write-RewardJson 'envs\rewards\ground\rules.json' ([ordered]@{event_rewards=[ordered]@{ground_action_masked=-1.0}})
Write-RewardJson $rulesPath ([ordered]@{
 description='Composed reward design. Fragments are merged at environment startup.'
 fragments=@('rewards/global/rules.json','rewards/recon/rules.json','rewards/attack/rules.json','rewards/landing/rules.json','rewards/ground/rules.json')
})