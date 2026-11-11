param([string]$Label = 'prof', [string]$Plan = 'w:down:0,w:up:10000',
      [double]$Seconds = 8, [int]$Top = 60, [string[]]$Focus = @(), [string[]]$Poke = @(),
      [string]$Live = '', [string[]]$Set = @(), [switch]$KeepRunning, [int]$Vblank = 0)
# Render-thread profile at one place of the walk route: launch (bench-run.ps1), walk -Plan and
# stop there, sample the render thread (live `prof`) and symbolize it with profreport.exe against
# this build's linker map while the process is still alive. Writes _Build/prof/<label>/prof.txt,
# and prof-<focus>.txt per -Focus text (hottest addresses and emulator callers of those leaves).
#   prof-spot.ps1 -Label heavy1 -Seconds 8 -Top 80 -Focus _NLG_Return2,memchr
# -Poke symbol=hexvalue: poke32 a kyty_local_* variable (linker-map address) before sampling;
# -Live: more live commands after the profile (e.g. 'sleep 2'). -Vblank 120: the 120 Hz cap (bench-run's), so the render
# thread works most of the frame instead of waiting for the 60 Hz flip.
$S = $PSScriptRoot
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $root
$out = "$root\_Build\prof\$Label"
New-Item -ItemType Directory -Force $out | Out-Null
$map = "$root\_Build\windows\kyty_emulator_clang_lld_link.map"
Copy-Item $map "$out\map.map"
# prof-hot-lines.py disassembles the build that ran; prof-inline.py symbolizes it with its PDB.
Copy-Item "$root\_Build\windows\kyty_emulator.exe" "$out\kyty_emulator.exe"
Copy-Item "$root\_Build\windows\kyty_emulator.pdb" "$out\kyty_emulator.pdb"
$pokes = foreach ($item in ($Poke | ForEach-Object { $_ -split ',' } | Where-Object { $_ })) {
	$symbol, $value = $item -split '=', 2
	$entry = Select-String -Path $map -Pattern "\s$symbol\s*$" | Select-Object -First 1
	if (!$entry) { "symbol $symbol not in $map"; exit 1 }
	# The Windows build links at a fixed base (0x140000000); map addresses are relative to it.
	"poke32 {0:x} $value" -f (0x140000000 + [Convert]::ToUInt64(($entry.Line.Trim() -split '\s+')[0], 16))
}
$params = @{ Label = $Label; KeepRunning = $true; NoWalk = $true; ShotOnly = $true }
if ($Set.Count) { $params['Set'] = $Set }
if ($Vblank -gt 0) { $params['Vblank'] = $Vblank }
& "$S\bench-run.ps1" @params | Select-Object -Last 3
$process = Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 } | Select-Object -First 1
if (!$process) { 'emulator not running'; exit 1 }
$walk = Start-Process powershell -ArgumentList '-NoProfile', '-File', "$S\keys.ps1", '-Plan', $Plan -PassThru -WindowStyle Hidden -RedirectStandardOutput "$out\keys.txt"
$walk.WaitForExit()
Get-Content "$out\keys.txt" | Select-Object -Last 1
Start-Sleep -Seconds 2
powershell -NoProfile -File "$S\screen.ps1" -Out "$out\spot.png" -Scale 0.25 | Out-Null
# Frames outside the executable by module (prof-inline.py --modules).
$process.Modules | ForEach-Object { '{0:x} {1:x} {2}' -f $_.BaseAddress.ToInt64(), $_.ModuleMemorySize, $_.ModuleName } |
	Set-Content "$out\modules.txt" -Encoding ascii
$bin = "$out\prof.bin" -replace '\\', '/'
$commands = @('measure 4 before') + @($pokes) + @("prof $Seconds $bin", 'measure 4 after')
if ($Live) { $commands += $Live }
python tools\local\bench-windows.py live ($commands -join ' ; ') |
	Select-String 'LIVE_PROF|LIVE_MEASURE|LIVE_POKE|LIVE_ERROR' | ForEach-Object { $_.Line } | Tee-Object -FilePath "$out\live.txt"
$env:KYTY_MAP = "$out\map.map"
& "$root\_Build\windows-tools\profreport.exe" $process.Id "$out\prof.bin" $Top | Set-Content "$out\prof.txt" -Encoding utf8
"report: $out\prof.txt"
$env:KYTY_PROF_CHAIN = '4'
$env:KYTY_PROF_EMU = '1'
# powershell -File passes a comma list as one string: split it.
foreach ($text in ($Focus | ForEach-Object { $_ -split ',' } | Where-Object { $_ })) {
	$name = $text -replace '[^A-Za-z0-9_]', '_'
	& "$root\_Build\windows-tools\profreport.exe" $process.Id "$out\prof.bin" 5 $text |
		Select-String -Pattern '== focus' -Context 0, 40 | ForEach-Object { $_.Line; $_.Context.PostContext } |
		Set-Content "$out\prof-$name.txt" -Encoding utf8
	"focus: $out\prof-$name.txt"
}
Remove-Item env:KYTY_PROF_CHAIN, env:KYTY_PROF_EMU -ErrorAction SilentlyContinue
if (!$KeepRunning) { Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 } | Stop-Process -Force }
