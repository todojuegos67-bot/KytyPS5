param([string]$Out = '', [string[]]$Set = @(), [switch]$NoPrecompile, [ValidateSet('fe', 'ir')][string]$Kind = 'fe')
# Windows PGO refresh (tools/local/PGO.md): build an instrumented emulator in _Build\windows-pgogen,
# play the training scene with it (the benchmark walk and standing, camera turns, on into the ruins,
# strafing), write the profile with the live `pgo` command and merge it. Installing it as the release
# profile and rebuilding are left to the caller (see PGO.md). Needs the baseline save installed
# (bench-windows.py prepare); -Set passes launch switches (e.g. KYTY_DRIVER_CACHE_KEY=<sha>). -NoPrecompile: no
# precompile pass first (its process's profile is not the one merged; the driver cache is shared by key).
# -Kind ir: LLVM's IR-level counters (KYTY_PGO_KIND=ir) instead of clang's front-end ones, built in a folder of its own,
# _Build\windows-pgogen-ir (KYTY_CMAKE_ARGS only counts at a folder's first configure), with the profile named
# kyty-ir-<date> (profiles of the two kinds don't merge).
$S = $PSScriptRoot
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $root
# (Only this checkout's emulators: another checkout's session may be running one.)
function Ours { Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 -and $_.Path -like "$root*" } }
New-Item -ItemType Directory -Force "$root\_Build\pgo\windows" | Out-Null
$suffix = if ($Kind -eq 'ir') { '-ir' } else { '' }
if (!$Out) { $Out = "$root\_Build\pgo\windows\kyty$suffix-$(Get-Date -Format yyyyMMdd)" }
$gen = "$root\_Build\windows-pgogen$suffix"
$env:KYTY_BUILD_DIR   = $gen
$env:KYTY_CMAKE_ARGS  = "-DKYTY_PGO_GENERATE=ON -DKYTY_PGO_KIND=$Kind"
cmd /c "$root\build-windows.cmd" | Select-String -Pattern ' error|FAILED' | Select-Object -Last 5
Remove-Item env:KYTY_BUILD_DIR, env:KYTY_CMAKE_ARGS
# (A folder configured before KYTY_PGO_KIND existed holds front-end counters, as the option's default does.)
$cached = Select-String -Path "$gen\CMakeCache.txt" -Pattern '^KYTY_PGO_KIND:STRING=(\w+)' -ErrorAction SilentlyContinue
$built = if ($cached) { $cached.Matches[0].Groups[1].Value } else { 'fe' }
if ($built -ne $Kind) { "no profile: $gen holds $built counters, not $Kind"; exit 1 }
# bench-run finds the game by process name: the instrumented executable runs from its own directory.
$exe = "$root\_Build\ab\pgogen$suffix"
New-Item -ItemType Directory -Force $exe | Out-Null
Copy-Item "$gen\kyty_emulator.exe", "$gen\libwinpthread-1.dll" $exe -Force
$params = @{ Label = 'pgotrain'; Exe = "$exe\kyty_emulator.exe"; KeepRunning = $true; NoWalk = $true; ShotOnly = $true; StartupSeconds = 180 }
if ($Set.Count) { $params['Set'] = $Set }
if ($NoPrecompile) { $params['NoPrecompile'] = $true }
# (The instrumented runtime's own file, default.profraw where it runs, made empty at start-up: under _Build instead.)
$env:LLVM_PROFILE_FILE = "$root\_Build\pgo\windows\default-%p.profraw"
$started = & "$S\bench-run.ps1" @params | Select-Object -Last 1
$started
# A start-up that never reached the HUD would train the title screen: no profile then.
if ("$started" -match 'no HUD') {
	Ours | Stop-Process -Force
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
Ours | Stop-Process -Force
& 'C:\Program Files\LLVM\bin\llvm-profdata.exe' merge -o "$Out.profdata" "$Out.profraw"
"profile: $Out.profdata"
