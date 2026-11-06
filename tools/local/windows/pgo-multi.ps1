param([string]$Scenes = 'world-prologue,rec-m02_00_00_00-c0000_0001,world-latria,rec-m02_00_00_00-c0000_0006,rec-m02_00_00_00-c0000_0001+KYTY_SCOUT=0')
# (-File passes one string: scenes separated by commas; <save>+<SWITCH=value> = that save with a launch switch.)
$SceneList = @($Scenes -split ',')
# PGO training over several scenes (tools/local/PGO.md; by default the release profile's five runs): the user's save moved aside once, each scene's
# baseline (_Build\bench-baselines\<save>) installed by bench-run (KYTY_BENCH_BASELINE), pgo-train.ps1 per scene, the raw
# profiles merged into _Build\pgo\windows\kyty-<date>.profdata. The user's save is restored at the end.
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $root
# (A single-thread process is an exit leftover; only this checkout's emulators are ever stopped.)
function Running { Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 } }
if (Running) { 'an emulator is running; nothing started'; exit 1 }
Get-ChildItem env: | Where-Object { $_.Name -like 'KYTY_*' } | ForEach-Object { Remove-Item "env:$($_.Name)" }
$env:KYTY_BENCH_BASELINE = "$root\_Build\bench-baselines\$($SceneList[0].Split('+')[0])"
python tools\local\bench-windows.py prepare
if ($LASTEXITCODE -ne 0) { 'prepare failed'; exit 1 }
$date = Get-Date -Format yyyyMMdd
try {
	foreach ($scene in $SceneList) {
		$save, $switch = $scene.Split('+', 2)
		$tag = $scene -replace '[^\w.-]', '_'
		if (!(Test-Path "$root\_Build\bench-baselines\$save")) { "no baseline save $save"; continue }
		$env:KYTY_BENCH_BASELINE = "$root\_Build\bench-baselines\$save"
		$out = "$root\_Build\pgo\windows\kyty-$date-$tag"
		# (The previous scene's emulator may still be exiting: its executable is copied again.)
		for ($i = 0; $i -lt 90 -and (Running); $i++) { Start-Sleep 1 }
		"== $tag $(Get-Date -Format HH:mm:ss)"
		$params = @{ Out = $out; NoPrecompile = $true }
		if ($switch) { $params['Set'] = @($switch) }
		& "$root\tools\local\windows\pgo-train.ps1" @params
		if (Test-Path "$out.profraw") { "raw: $((Get-Item "$out.profraw").Length) bytes" }
		else { "NO PROFILE for $tag" }
	}
} finally {
	Remove-Item env:KYTY_BENCH_BASELINE -ErrorAction SilentlyContinue
	Running | Where-Object { $_.Path -like "$root*" } | Stop-Process -Force
	Start-Sleep 3
	# (Another checkout's emulator refuses the restore: every 20 s until it is gone, up to 40 min.)
	for ($r = 0; $r -lt 120; $r++) { $line = "$(python tools\local\bench-windows.py restore 2>&1 | Select-Object -Last 1)"; if ($line -notmatch 'emulator process is running') { break }; Start-Sleep 20 }; $line
}
$raws = @(Get-ChildItem "$root\_Build\pgo\windows\kyty-$date-*.profraw" | ForEach-Object { $_.FullName })
if ($raws.Count) {
	& 'C:\Program Files\LLVM\bin\llvm-profdata.exe' merge -o "$root\_Build\pgo\windows\kyty-$date.profdata" @raws
	"merged $($raws.Count) scenes: $root\_Build\pgo\windows\kyty-$date.profdata"
}
