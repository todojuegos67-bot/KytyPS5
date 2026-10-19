param([Parameter(Mandatory)][string]$A, [Parameter(Mandatory)][string]$B, [string]$Label = 'abexe',
      [string]$Plan = 'w:down:0,w:up:10000', [int]$Rounds = 2, [int]$Windows = 2, [double]$Seconds = 6,
      [string[]]$Set = @(), [string[]]$SetA = @(), [string[]]$SetB = @(), [int]$StartupSeconds = 100,
      [int]$Vblank = 0)
# A/B of two builds at one place of the walk route, for changes a runtime switch cannot select (data
# structures, code generation): separate launches in ABBA order (each one: bench-run.ps1 with -Exe,
# walk -Plan, -Windows measurement windows of -Seconds). Each executable sits in its own directory
# with libwinpthread-1.dll; its first launch also builds its driver pipeline cache (bench-run's
# precompile), so the first round is a warm-up worth repeating when it differs. -SetA / -SetB add
# switches to one side only (one executable as both A and B compares launch-time switches). -StartupSeconds: each
# start-up attempt's limit (a 1-1 save on a 16 GB laptop needs ~180). -Vblank 120: both sides at the 120 Hz cap.
#   ab-exe.ps1 -A _Build\ab\base\kyty_emulator.exe -B _Build\windows\kyty_emulator.exe -Label rangeset
$S = $PSScriptRoot
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $root
$out = "$root\_Build\ab\$Label"
New-Item -ItemType Directory -Force $out | Out-Null
function Emulator { Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 } }
$sums = @{}
$order = @()
for ($round = 1; $round -le $Rounds; $round++) { $order += if ($round % 2) { @('A', 'B') } else { @('B', 'A') } }
foreach ($side in $order) {
	$exe = (Resolve-Path $(if ($side -eq 'A') { $A } else { $B })).Path
	$params = @{ Label = "$Label-$side"; Exe = $exe; KeepRunning = $true; NoWalk = $true; ShotOnly = $true; StartupSeconds = $StartupSeconds }
	$switches = @($Set) + $(if ($side -eq 'A') { @($SetA) } else { @($SetB) })
	if ($switches.Count) { $params['Set'] = $switches }
	if ($Vblank -gt 0) { $params['Vblank'] = $Vblank }
	& "$S\bench-run.ps1" @params | Select-Object -Last 1
	if (!(Emulator)) { "$side`: emulator not running"; continue }
	$walk = Start-Process powershell -ArgumentList '-NoProfile', '-File', "$S\keys.ps1", '-Plan', $Plan -PassThru -WindowStyle Hidden -RedirectStandardOutput "$out\keys.txt"
	$walk.WaitForExit()
	Start-Sleep -Seconds 2
	$commands = 1..$Windows | ForEach-Object { "measure $Seconds $side$_" }
	$lines = python tools\local\bench-windows.py live ($commands -join ' ; ') | Select-String 'LIVE_MEASURE' | ForEach-Object { $_.Line }
	$lines | Add-Content "$out\live.txt" -Encoding ascii
	foreach ($line in $lines) {
		if ($line -match 'label=(\S+) .*fps=([\d.]+) render_ms=([\d.]+)') {
			'{0}: {1,6} fps  render {2,6} ms' -f $Matches[1], $Matches[2], $Matches[3]
			if (!$sums.Contains($side)) { $sums[$side] = @(0.0, 0.0, 0) }
			$sums[$side] = @(($sums[$side][0] + [double]$Matches[2]), ($sums[$side][1] + [double]$Matches[3]), ($sums[$side][2] + 1))
		}
	}
	Emulator | Stop-Process -Force
	for ($i = 0; $i -lt 30 -and (Emulator); $i++) { Start-Sleep 1 }
}
foreach ($side in @('A', 'B')) {
	if ($sums.Contains($side)) {
		$n = $sums[$side][2]
		'{0}: {1:N2} fps, render {2:N2} ms/frame over {3} windows' -f $side, ($sums[$side][0] / $n), ($sums[$side][1] / $n), $n
	}
}
