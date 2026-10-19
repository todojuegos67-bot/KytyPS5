param([string]$Out = '', [string[]]$Set = @(), [switch]$NoPrecompile)
# Windows PGO refresh (tools/local/PGO.md): build an instrumented emulator in _Build\windows-pgogen,
# play the training scene with it (the benchmark walk and standing, camera turns, on into the ruins,
# strafing), write the profile with the live `pgo` command and merge it. Installing it as the release
# profile and rebuilding are left to the caller (see PGO.md). Needs the baseline save installed
# (bench-windows.py prepare); -Set passes launch switches (e.g. KYTY_DRIVER_CACHE_KEY=<sha>). -NoPrecompile: no
# precompile pass first (its process's profile is not the one merged; the driver cache is shared by key).
$S = $PSScriptRoot
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $root
if (!$Out) { $Out = "$root\_Build\pgo\windows\kyty-$(Get-Date -Format yyyyMMdd)" }
$env:KYTY_BUILD_DIR   = "$root\_Build\windows-pgogen"
$env:KYTY_CMAKE_ARGS  = '-DKYTY_PGO_GENERATE=ON'
cmd /c "$root\build-windows.cmd" | Select-String -Pattern ' error|FAILED' | Select-Object -Last 5
Remove-Item env:KYTY_BUILD_DIR, env:KYTY_CMAKE_ARGS
# bench-run finds the game by process name: the instrumented executable runs from its own directory.
$exe = "$root\_Build\ab\pgogen"
New-Item -ItemType Directory -Force $exe | Out-Null
Copy-Item "$root\_Build\windows-pgogen\kyty_emulator.exe", "$root\_Build\windows-pgogen\libwinpthread-1.dll" $exe -Force
$params = @{ Label = 'pgotrain'; Exe = "$exe\kyty_emulator.exe"; KeepRunning = $true; NoWalk = $true; ShotOnly = $true; StartupSeconds = 180 }
if ($Set.Count) { $params['Set'] = $Set }
if ($NoPrecompile) { $params['NoPrecompile'] = $true }
$started = & "$S\bench-run.ps1" @params | Select-Object -Last 1
$started
# A start-up that never reached the HUD would train the title screen: no profile then.
if ("$started" -match 'no HUD') {
	Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 } | Stop-Process -Force
	'no profile: the game did not reach the HUD'
	exit 1
}
function Plan([string]$plan) {
	$walk = Start-Process powershell -ArgumentList '-NoProfile', '-File', "$S\keys.ps1", '-Plan', $plan -PassThru -WindowStyle Hidden
	$walk.WaitForExit()
}
Start-Sleep 10
Plan 'w:down:0,w:up:10000'   # the benchmark path
Start-Sleep 20               # standing where the benchmark measures
Plan 'h:down:0,h:up:1500'; Start-Sleep 4
Plan 'f:down:0,f:up:3000'; Start-Sleep 4
Plan 'h:down:0,h:up:1500'; Start-Sleep 3
Plan 'w:down:0,w:up:12000'   # on into the ruins
Start-Sleep 15
Plan 'd:down:0,d:up:2000,w:down:2000,w:up:8000,a:down:8000,a:up:10000'
Start-Sleep 15
$raw = "$Out.profraw" -replace '\\', '/'
python tools\local\bench-windows.py live "pgo $raw" | Select-String 'LIVE_PGO|LIVE_ERROR'
Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 } | Stop-Process -Force
& 'C:\Program Files\LLVM\bin\llvm-profdata.exe' merge -o "$Out.profdata" "$Out.profraw"
"profile: $Out.profdata"
