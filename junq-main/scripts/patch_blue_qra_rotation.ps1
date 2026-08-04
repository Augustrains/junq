param([Parameter(Mandatory = $true)][string]$ScenarioPath)

$ErrorActionPreference = "Stop"
$s = [IO.File]::ReadAllText($ScenarioPath)

if ($s -notmatch 'script int CountBlueActiveCAP') {
    $marker = 'script void RecordDestroyedPlatform(string platformName)'
    $helper = @'
script int CountBlueActiveCAP(string excludedName)
   int activeCount = 0;
   for (int fighterIndex = 0; fighterIndex < BLUE_AIR_TARGET_NAMES.Size(); fighterIndex = fighterIndex + 1)
   {
      WsfPlatform fighter = WsfSimulation.FindPlatform(BLUE_AIR_TARGET_NAMES[fighterIndex]);
      if (fighter != null && fighter.Name() != excludedName && fighter.DamageFactor() < 1.0 &&
          fighter.AuxDataExists("BLUE_CAP_ENABLED") && fighter.AuxDataBool("BLUE_CAP_ENABLED") &&
          fighter.AuxDataExists("TASK_STATUS"))
      {
         string fighterStatus = fighter.AuxDataString("TASK_STATUS");
         if (fighterStatus == "BLUE_CAP_PATROL" || fighterStatus == "BLUE_INTERCEPT" ||
             fighterStatus == "BLUE_WEAPON_SUPPORT" || fighterStatus == "BLUE_GROUND_ATTACK")
         {
            activeCount = activeCount + 1;
         }
      }
   }
   return activeCount;
end_script

'@
    if (-not $s.Contains($marker)) { throw "RecordDestroyed marker missing" }
    $s = $s.Replace($marker, $helper + $marker)
}

if ($s -notmatch 'string BLUE_FORCE_ROLE') {
    $marker = '      bool BLUE_CAP_ENABLED = false'
    $replacement = $marker + "`n" + '      string BLUE_FORCE_ROLE = "UNASSIGNED"' + "`n" + '      int BLUE_DESIRED_CAP_COUNT = 4'
    if (-not $s.Contains($marker)) { throw "BLUE_CAP_ENABLED marker missing" }
    $s = $s.Replace($marker, $replacement)
}

$rolePattern = '(?ms)(else if \(PLATFORM\.Name\(\) == "blue_attack_4"\).*?lane=east_south"\);\s*\})(\s*\}\s*string status = PLATFORM\.AuxDataString\("TASK_STATUS"\);)'
$role = @'

         if (PLATFORM.AuxDataBool("BLUE_CAP_ENABLED"))
         {
            PLATFORM.SetAuxData("BLUE_FORCE_ROLE", "CAP_INITIAL");
         }
         else if (PLATFORM.Name() == "blue_attack_5" || PLATFORM.Name() == "blue_attack_6" ||
                  PLATFORM.Name() == "blue_attack_7" || PLATFORM.Name() == "blue_attack_8")
         {
            PLATFORM.SetAuxData("BLUE_FORCE_ROLE", "QRA_READY");
            writeln("[BLUE_QRA] READY T=", TIME_NOW, " fighter=", PLATFORM.Name());
         }
         else
         {
            PLATFORM.SetAuxData("BLUE_FORCE_ROLE", "RESERVE");
            writeln("[BLUE_QRA] RESERVE T=", TIME_NOW, " fighter=", PLATFORM.Name());
         }
'@
$m = [regex]::Match($s, $rolePattern)
if (-not $m.Success) { throw "role insertion location missing" }
$replacement = $m.Groups[1].Value + $role + $m.Groups[2].Value
$s = [regex]::Replace($s, $rolePattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($x) $replacement }, 1)

$weaponMarker = '      WsfWeapon agm = PLATFORM.Weapon("agm");'
$dynamic = @'

      if (!PLATFORM.AuxDataBool("BLUE_CAP_ENABLED") &&
          (status == "IDLE" || status == "PARKED") && TIME_NOW >= 5.0 &&
          CountBlueActiveCAP(PLATFORM.Name()) < PLATFORM.AuxDataInt("BLUE_DESIRED_CAP_COUNT"))
      {
         int activeBefore = CountBlueActiveCAP(PLATFORM.Name());
         PLATFORM.SetAuxData("BLUE_CAP_ENABLED", true);
         PLATFORM.SetAuxData("BLUE_FORCE_ROLE", "CAP_REPLACEMENT");
         PLATFORM.SetAuxData("EXTERNAL_TASK", "CAP_PATROL");
         PLATFORM.SetAuxData("TASK_STATUS", "BLUE_CAP_PATROL");
         PLATFORM.SetAuxData("BLUE_CAP_SORTIE_START_TIME", TIME_NOW);
         PLATFORM.SetAuxData("AT_HOME_BASE", false);
         status = "BLUE_CAP_PATROL";
         writeln("[BLUE_QRA] SCRAMBLE_REPLACEMENT T=", TIME_NOW,
                 " fighter=", PLATFORM.Name(), " active_before=", activeBefore,
                 " desired=", PLATFORM.AuxDataInt("BLUE_DESIRED_CAP_COUNT"));
      }
'@
if (-not $s.Contains($weaponMarker)) { throw "weapon marker missing" }
$blueAircraftStart = $s.IndexOf('platform_type BLUE_ATTACK_AIRCRAFT')
if ($blueAircraftStart -lt 0) { throw "BLUE_ATTACK_AIRCRAFT section missing" }
$weaponIndex = $s.IndexOf($weaponMarker, $blueAircraftStart)
if ($weaponIndex -lt 0) { throw "blue aircraft weapon marker missing" }
$s = $s.Substring(0, $weaponIndex) + $weaponMarker + $dynamic +
     $s.Substring($weaponIndex + $weaponMarker.Length)

$rearmPattern = '(?ms)\s*if \(PLATFORM\.AuxDataBool\("BLUE_CAP_ENABLED"\)\)\s*\{\s*PLATFORM\.SetAuxData\("EXTERNAL_TASK", "CAP_PATROL"\);.*?\}\s*else\s*\{\s*PLATFORM\.SetAuxData\("EXTERNAL_TASK", "PARKED"\);.*?\}'
$rearm = @'

            bool capSlotOpen = CountBlueActiveCAP(PLATFORM.Name()) < PLATFORM.AuxDataInt("BLUE_DESIRED_CAP_COUNT");
            if (capSlotOpen)
            {
               PLATFORM.SetAuxData("BLUE_CAP_ENABLED", true);
               PLATFORM.SetAuxData("BLUE_FORCE_ROLE", "CAP_REPLACEMENT");
               PLATFORM.SetAuxData("EXTERNAL_TASK", "CAP_PATROL");
               PLATFORM.SetAuxData("TASK_STATUS", "BLUE_CAP_PATROL");
               PLATFORM.SetAuxData("BLUE_CAP_SORTIE_START_TIME", TIME_NOW);
               PLATFORM.SetAuxData("AT_HOME_BASE", false);
            }
            else
            {
               PLATFORM.SetAuxData("BLUE_CAP_ENABLED", false);
               PLATFORM.SetAuxData("BLUE_FORCE_ROLE", "QRA_READY");
               PLATFORM.SetAuxData("EXTERNAL_TASK", "PARKED");
               PLATFORM.SetAuxData("TASK_STATUS", "IDLE");
               PLATFORM.SetAuxData("AT_HOME_BASE", true);
            }
'@
$matches = [regex]::Matches($s, $rearmPattern)
if ($matches.Count -ne 1) { throw "expected one rearm role block, found $($matches.Count)" }
$s = [regex]::Replace($s, $rearmPattern, $rearm, 1)

[IO.File]::WriteAllText($ScenarioPath, $s, [Text.UTF8Encoding]::new($false))
Write-Host "patched blue CAP/QRA rotation: $ScenarioPath"
