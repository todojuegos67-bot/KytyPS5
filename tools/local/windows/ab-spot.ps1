param([string]$Label = 'ab', [string]$Plan = 'w:down:0,w:up:10000',
      [string]$Symbol = '', [string]$A = '0', [string]$B = '1',
      [int]$Rounds = 4, [double]$Seconds = 6, [string[]]$Set = @(), [switch]$KeepRunning,
      [int]$Attempts = 3, [double]$MaxFps = 55)
# Same-process A/B at one place of the walk route: launch (bench-run.ps1), walk -Plan and stop
# there, then switch a 32-bit variable (a kyty_local_* switch, by its linker-map address) between
# the hex values -A and -B with the live poke32 command, measuring one window per setting in
# ABBA order. Prints each window and the per-setting means of fps and render-thread ms/frame.
#   ab-spot.ps1 -Label srt -Symbol kyty_local_srt_predicate_mode -A 0 -B 1
if (!$Symbol) { 'usage: ab-spot.ps1 -Symbol <kyty_local_* switch> -A <hex> -B <hex>'; exit 1 }
$S = $PSScriptRoot
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $root
$out = "$root\_Build\ab\$Label"
New-Item -ItemType Directory -Force $out | Out-Null
$map = "$root\_Build\windows\kyty_emulator_clang_lld_link.map"
$entry = Select-String -Path $map -Pattern "\s$Symbol\s*$" | Select-Object -First 1
if (!$entry) { "symbol $Symbol not in $map"; exit 1 }
# The Windows build links at a fixed base (0x140000000); map addresses are relative to it.
$address = '{0:x}' -f (0x140000000 + [Convert]::ToUInt64(($entry.Line.Trim() -split '\s+')[0], 16))
$params = @{ Label = $Label; KeepRunning = $true; NoWalk = $true; ShotOnly = $true }
if ($Set.Count) { $params['Set'] = $Set }
# The walk sometimes ends facing a wall of the tunnel (a light spot at the 60 fps cap): walk again
# from a fresh start then, up to -Attempts times (-MaxFps 0 keeps the first spot).
for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
	& "$S\bench-run.ps1" @params | Select-Object -Last 3
	if (!(Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 })) { 'emulator not running'; exit 1 }
	$walk = Start-Process powershell -ArgumentList '-NoProfile', '-File', "$S\keys.ps1", '-Plan', $Plan -PassThru -WindowStyle Hidden -RedirectStandardOutput "$out\keys.txt"
	$walk.WaitForExit()
	Get-Content "$out\keys.txt" | Select-Object -Last 1
	Start-Sleep -Seconds 2
	if ($MaxFps -le 0 -or $attempt -eq $Attempts) { break }
	$probe = python tools\local\bench-windows.py live 'measure 2 probe' | Select-String 'LIVE_MEASURE' | Select-Object -First 1
	if ($probe -and $probe.Line -match 'fps=([\d.]+)' -and [double]$Matches[1] -le $MaxFps) { "spot: $($Matches[1]) fps"; break }
	"light spot ($($Matches[1]) fps): walking again"
}
powershell -NoProfile -File "$S\screen.ps1" -Out "$out\spot.png" -Scale 0.25 | Out-Null
$commands = New-Object System.Collections.Generic.List[string]
for ($round = 1; $round -le $Rounds; $round++) {
	$order = if ($round % 2) { @('A', 'B') } else { @('B', 'A') }
	foreach ($side in $order) {
		$value = if ($side -eq 'A') { $A } else { $B }
		$commands.Add("poke32 $address $value")
		$commands.Add('sleep 1')
		$commands.Add("measure $Seconds $side$round")
	}
}
$commands.Add("poke32 $address $B")
$lines = python tools\local\bench-windows.py live ($commands -join ' ; ') | Select-String 'LIVE_MEASURE|LIVE_COUNTERS|LIVE_ERROR' | ForEach-Object { $_.Line }
$lines | Set-Content "$out\live.txt" -Encoding ascii
# One client-area picture under each setting (standing still: compare them for rendering errors).
foreach ($side in @('A', 'B')) {
	$value = if ($side -eq 'A') { $A } else { $B }
	python tools\local\bench-windows.py live "poke32 $address $value ; sleep 2" | Out-Null
	powershell -NoProfile -File "$S\screen.ps1" -Out "$out\shot-$side.png" -Client | Out-Null
}
python tools\local\bench-windows.py live "poke32 $address $B" | Out-Null
$sums = @{}
foreach ($line in $lines) {
	if ($line -match 'label=([AB])(\d+) .*fps=([\d.]+) render_ms=([\d.]+) record_ms=([\d.]+) render_busy=([\d.]+)') {
		'{0}{1}: {2,6} fps  render {3,6} ms  record {4,6} ms  busy {5}%' -f $Matches[1], $Matches[2], $Matches[3], $Matches[4], $Matches[5], $Matches[6]
		if (!$sums.Contains($Matches[1])) { $sums[$Matches[1]] = @(0.0, 0.0, 0) }
		$sums[$Matches[1]] = @(($sums[$Matches[1]][0] + [double]$Matches[3]), ($sums[$Matches[1]][1] + [double]$Matches[4]), ($sums[$Matches[1]][2] + 1))
	} elseif ($line -match 'LIVE_ERROR') { $line }
}
foreach ($side in @('A', 'B')) {
	if ($sums.Contains($side)) {
		$n = $sums[$side][2]
		'{0} ({1}={2}): {3:N2} fps, render {4:N2} ms/frame over {5} windows' -f $side, $Symbol, $(if ($side -eq 'A') { $A } else { $B }), ($sums[$side][0] / $n), ($sums[$side][1] / $n), $n
	}
}
if (!$KeepRunning) { Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 } | Stop-Process -Force }
