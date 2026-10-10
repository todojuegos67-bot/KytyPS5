# Static shader and pipeline precompile (tools\local\static-precompile), a program of its own:
#   .\precompile-windows.ps1              every shader and pipeline of the game into the static pipeline
#                                         cache _PipelineCache\static\<title>_<version>.bin, which the emulator
#                                         looks up before compiling (hours the first time: run it overnight); with
#                                         a driver that has VK_KHR_pipeline_binary (NVIDIA 5xx) the pipelines'
#                                         binaries instead, <title>_<version>.binaries, read only when a pipeline is
#                                         needed (the driver copies a whole .bin into memory: GBs)
#   .\precompile-windows.ps1 -Jobs 8      8 processes at a time (default: 3 threads each on the allowed CPUs)
#   .\precompile-windows.ps1 -Coverage    no pipelines, only what the seeds compile to, for
#                                         tools\local\static-precompile\precompile.py coverage
#   .\precompile-windows.ps1 -InputsOnly  only the compiled inputs the emulator's shader prefetch
#                                         translates in the background (_PipelineCache\static\<title>_<version>.shaders;
#                                         every full run writes them too)
# The NVIDIA driver compiles big compute shaders nearly one at a time per process, so the work is split
# into shards, a below-normal-priority process each (kyty_shader_precompile --shard i/n), whose caches
# are merged into the static cache at the end. The shards are small (-Shards, about 100 pipelines each):
# after a few hundred pipelines NVIDIA compresses binaries with a dictionary of its own, which only that
# PC reads (pipelineBinaries.h); a shard that got there anyway left the rest out (exit code 3) and runs
# again as two. recorded-<title>_<version>.seeds next to the seed file (-Recorded) adds the specializations of
# recorded play; a PC without a recording of its own (a release) gets those of the specialization hints the release
# ships (tools\local\static-precompile\hints-<title>_<version>.hints: recordings without the game's code), completed
# with the code of the seed file into hinted-<title>_<version>.seeds (kyty_shader_precompile --apply-hints, a second).
# An interrupted run resumes: finished shards are merged first, and what the static cache
# holds is not compiled again (binaries: the final merge keeps only what this run's shards made, so
# pipelines no seed makes any more are dropped; a .bin left from before is where the first binaries run
# takes them from without compiling). Build the program with build-windows.cmd kyty_shader_precompile; it also
# makes the seed file when there is none (--make-seeds: the game's shaders from its files, with the render-pass
# states of tools\local\static-precompile\pass-states.json). A release package (.github/workflows/build.yml) has
# the program and launch.json next to this script (precompile.cmd) and pass-states.json in that folder below it;
# the seed file is made there. The files are the game version's (seeds-<title>_<version>.seeds): versions need
# not share shaders.
param(
	[string]$Game = '',
	# Default: seeds-<title>_<version>.seeds next to this script (a release) or in _Build\static-precompile.
	[string]$Seeds = '',
	# The shaders as the game specialized them in recorded play (tools\local\static-precompile\precompile.py
	# recorded-seeds): the specializations the seeds' guesses miss (a new PC compiled ~55 compute
	# pipelines, up to 6 s each, before the HUD). '' = none; default: recorded-<title>_<version>.seeds next to the seed file,
	# else the release's hints completed with the seed file's code (hinted-<title>_<version>.seeds).
	[string]$Recorded = '*',
	[int]$Jobs = 0,
	[int]$Threads = 3,
	[int]$Shards = 512,
	[int64]$Affinity = 0, # 0: the launch config's CPUs (this PC's leave out 4 and 5, where the compiler crashes), else all
	[switch]$Coverage,
	[switch]$InputsOnly,
	[string]$Exe = $(if (Test-Path "$PSScriptRoot\kyty_shader_precompile.exe") { "$PSScriptRoot\kyty_shader_precompile.exe" } else { "$PSScriptRoot\_Build\windows\kyty_shader_precompile.exe" })
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path $Exe)) { throw "missing $Exe; build it with build-windows.cmd kyty_shader_precompile" }
# The game's sce_sys\param.json (its title and version). The game is its folder (eboot.bin and sce_sys) or the
# folder packed into a ZArchive (a .zar file, read without extracting it): a .zar's is copied out by the program.
function Read-GameParam([string]$game) {
	if (!$game) { return $null }
	if (Test-Path -LiteralPath "$game\sce_sys\param.json") { return Get-Content -LiteralPath "$game\sce_sys\param.json" -Raw -Encoding UTF8 | ConvertFrom-Json }
	$tool = $Exe
	if ($game -notmatch '\.zar$' -or !(Test-Path -LiteralPath $game -PathType Leaf) -or !(Test-Path $tool)) { return $null }
	$copy = [IO.Path]::GetTempFileName()
	try {
		& $tool --game $game --param $copy 2>$null | Out-Null
		if ($LASTEXITCODE -eq 0) { return Get-Content -LiteralPath $copy -Raw -Encoding UTF8 | ConvertFrom-Json }
	} catch {
	} finally {
		Remove-Item -LiteralPath $copy -ErrorAction SilentlyContinue
	}
	return $null
}

# Caches made from the game's files are named by its title and version (seeds-<title>_<version>.seeds,
# _PipelineCache\static\<title>_<version>.*, the warmup recordings; versions need not share shaders). Those named
# by the title alone are from before: the game's that was played last (the remembered one), so they take its
# name (run-windows.ps1 has the same).
function Rename-TitleCaches([string]$game) {
	$info = Read-GameParam $game
	if (!$info -or !$info.titleId -or !$info.contentVersion) { return }
	$title = $info.titleId
	$id = "$($title)_$($info.contentVersion)"
	$moves = @()
	foreach ($dir in @($PSScriptRoot, "$PSScriptRoot\_Build\static-precompile")) {
		$moves += , @("$dir\seeds.seeds", "$dir\seeds-$id.seeds")
		$moves += , @("$dir\recorded.seeds", "$dir\recorded-$id.seeds")
	}
	foreach ($file in @(Get-ChildItem "$PSScriptRoot\_PipelineCache\static\$title.*" -File -ErrorAction SilentlyContinue)) {
		$moves += , @($file.FullName, (Join-Path $file.DirectoryName ($id + $file.Name.Substring($title.Length))))
	}
	foreach ($file in @(Get-ChildItem "$PSScriptRoot\_PipelineCache\warmup-v2\*\$title.shaders" -File -ErrorAction SilentlyContinue)) {
		$moves += , @($file.FullName, (Join-Path $file.DirectoryName "$id.shaders"))
	}
	foreach ($move in $moves) {
		if (!(Test-Path $move[0]) -or (Test-Path $move[1])) { continue }
		Move-Item $move[0] $move[1]
		Write-Host "caches:   $($move[0].Substring($PSScriptRoot.Length + 1)) is $id's"
	}
}

# The game run-windows.ps1 was given last, else the default folder.
$lastGame = if (Test-Path "$PSScriptRoot\game-path.txt") { "$(Get-Content "$PSScriptRoot\game-path.txt" -Raw)".Trim() }
if (!$lastGame) { $lastGame = "$env:USERPROFILE\Documents\PPSA01341-app0" }
Rename-TitleCaches $lastGame
if (!$Game) { $Game = $lastGame }
$info = Read-GameParam $Game
if (!$info) { throw "no sce_sys\param.json in $Game (-Game <folder or .zar>, or start the game once to choose it)" }
$gameId = "$($info.titleId)_$($info.contentVersion)"
$defaultSeeds = !$Seeds
if (!$Seeds) {
	$Seeds = if ((Test-Path "$PSScriptRoot\seeds-$gameId.seeds") -or !(Test-Path "$PSScriptRoot\_Build")) { "$PSScriptRoot\seeds-$gameId.seeds" } else { "$PSScriptRoot\_Build\static-precompile\seeds-$gameId.seeds" }
}
# The seed file another program made (<file>.program: its size and time) is made again (a release unpacked over the
# last), and replaced where it lists something else (the vertex shaders' indirect draws' records since 10-10): the
# static cache is then older than it, out of date (--status), and the precompile compiles what is new.
$programStamp = "$((Get-Item $Exe).Length) $((Get-Item $Exe).LastWriteTimeUtc.Ticks)"
$madeBy = "$Seeds.program"
$remake = $defaultSeeds -and (Test-Path $Seeds) -and (!(Test-Path $madeBy) -or "$(Get-Content $madeBy -Raw)".Trim() -ne $programStamp)
if (!(Test-Path $Seeds) -or $remake) {
	# Every shader the game ships, with the pipelines it draws them with (from the game files, by the program:
	# precompile.py seeds without Python).
	New-Item -ItemType Directory -Force (Split-Path $Seeds) | Out-Null
	$made = if ($remake) { "$Seeds.new" } else { $Seeds }
	& $Exe --game $Game --make-seeds $made --states "$PSScriptRoot\tools\local\static-precompile\pass-states.json"
	if ($LASTEXITCODE) { throw 'making the seed file failed (kyty_shader_precompile --make-seeds)' }
	if ($remake -and (Get-FileHash $made).Hash -eq (Get-FileHash $Seeds).Hash) { Remove-Item $made } elseif ($remake) { Move-Item -Force $made $Seeds }
	Set-Content $madeBy $programStamp
}
if ($Recorded -eq '*') {
	$Recorded = Join-Path (Split-Path $Seeds) "recorded-$gameId.seeds"
	$hints = "$PSScriptRoot\tools\local\static-precompile\hints-$gameId.hints"
	if (!(Test-Path $Recorded) -and (Test-Path $hints)) {
		# (Made again each run: the hints of this release, the code of this seed file.)
		$Recorded = Join-Path (Split-Path $Seeds) "hinted-$gameId.seeds"
		& $Exe --game $Game --seeds $Seeds --apply-hints $hints --out $Recorded
		if ($LASTEXITCODE) {
			Write-Host 'the specialization hints could not be applied (kyty_shader_precompile --apply-hints): the seeds alone'
			$Recorded = ''
		}
	}
}
# The pipeline recipes of a seed file (its body after the identity line and checksum: the records, each its word
# count and words, then the recipes' count).
function Get-SeedPipelines([string]$file) {
	$stream = [IO.File]::OpenRead($file)
	try {
		$reader = [IO.BinaryReader]::new($stream)
		while ($reader.ReadByte() -ne 10) {}
		$null = $reader.ReadUInt64()
		$records = $reader.ReadUInt32()
		for ($i = 0; $i -lt $records; $i++) { $null = $stream.Seek([int64]$reader.ReadUInt32() * 4, [IO.SeekOrigin]::Current) }
		return $reader.ReadUInt32()
	} finally {
		$stream.Dispose()
	}
}
if ($Affinity -eq 0) {
	$config = @("$PSScriptRoot\launch.json", "$PSScriptRoot\run-windows.json") | Where-Object { Test-Path $_ } |
		Select-Object -First 1
	foreach ($cpu in (Get-Content $config -Raw | ConvertFrom-Json).cpu_affinity) { $Affinity = $Affinity -bor ([int64]1 -shl [int]$cpu) }
}
$all = if ([Environment]::ProcessorCount -ge 64) { [int64]-1 } else { ([int64]1 -shl [Environment]::ProcessorCount) - 1 }
$mask = $Affinity -band $all
if ($mask -eq 0) { $mask = $all }
$cpus = 0
for ($bit = 0; $bit -lt 64; $bit++) { if ($mask -band ([int64]1 -shl $bit)) { $cpus++ } }
# A process scales to a few threads (4: 85%), and each translates the programs its share needs.
if ($Jobs -le 0) { $Jobs = [math]::Max(1, [math]::Ceiling($cpus / $Threads)) }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logs = if (Test-Path "$PSScriptRoot\_Build") { "$PSScriptRoot\_Build\run-logs" } else { "$PSScriptRoot\logs" }
$logs = "$logs\$stamp-precompile"
$logName = $logs.Substring($PSScriptRoot.Length + 1)
New-Item -ItemType Directory -Force $logs | Out-Null
$begin = Get-Date

# Each process gets an empty NVIDIA disk cache of its own (nothing trained yet), removed when it is done.
$nvidiaCaches = Join-Path ([IO.Path]::GetTempPath()) "kyty-precompile-$stamp"
function Start-Precompile([string]$name, [string[]]$arguments) {
	$env:__GL_SHADER_DISK_CACHE_PATH = Join-Path $nvidiaCaches $name
	New-Item -ItemType Directory -Force $env:__GL_SHADER_DISK_CACHE_PATH | Out-Null
	$process = Start-Process -FilePath $Exe -ArgumentList (@('--game', "`"$Game`"") + $arguments) -NoNewWindow -PassThru `
		-WorkingDirectory $PSScriptRoot -RedirectStandardOutput "$logs\$name.out.log" -RedirectStandardError "$logs\$name.err.log"
	$process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
	$process.ProcessorAffinity = [IntPtr]$mask
	$null = $process.Handle # keeps the exit code readable after the process ends
	$process | Add-Member NoteProperty NvidiaCache $env:__GL_SHADER_DISK_CACHE_PATH -PassThru
}
function Remove-NvidiaCache($process) { Remove-Item -Recurse -Force $process.NvidiaCache -ErrorAction SilentlyContinue }
function Wait-Precompile($process, [string]$what) {
	$process.WaitForExit()
	Remove-NvidiaCache $process
	if ($process.ExitCode -ne 0) { throw "$what failed; logs: $logName" }
}

Write-Host "precompile: $Seeds, $Shards shards, $Jobs at a time, affinity 0x$('{0:X}' -f $mask); logs $logName"
try {
if ($Coverage) {
	$out = [IO.Path]::ChangeExtension($Seeds, '.compiled.shaders')
	Wait-Precompile (Start-Precompile 'coverage' @('--seeds', "`"$Seeds`"", '--no-pipelines', '--threads', "$cpus",
		'--out', "`"$out`"")) 'coverage'
	Write-Host "wrote $out"
	return
}
# The compiled inputs for the emulator's shader prefetch: the seeds translated as the full run does,
# without pipelines (half a minute).
$inputs = @('--seeds', "`"$Seeds`"", '--no-pipelines', '--threads', "$cpus", '--static-inputs')
if ($InputsOnly) {
	Wait-Precompile (Start-Precompile 'inputs' $inputs) 'inputs'
	Write-Host "wrote _PipelineCache\static\$gameId.shaders"
	return
}
# The shards an interrupted run finished first: what they hold is not compiled again.
Wait-Precompile (Start-Precompile 'merge-before' @('--merge')) 'merge'
$pending = [System.Collections.Generic.Queue[object]]::new()
for ($i = 0; $i -lt $Shards; $i++) { $pending.Enqueue(@($Seeds, "$i/$Shards")) }
if ($Recorded -and (Test-Path $Recorded)) {
	# About as many pipelines a shard as the seeds' (100: a recording of every world holds about 16000).
	$count = [math]::Max(1, [math]::Ceiling((Get-SeedPipelines $Recorded) / 100))
	for ($i = 0; $i -lt $count; $i++) { $pending.Enqueue(@($Recorded, "$i/$count")) }
}
# The shards share the pipelines they hold (PipelineBinaryWriter::Claimed): one that the seeds of several
# shards make (a third of them) is compiled once.
$env:KYTY_PRECOMPILE_CLAIMS = Join-Path $nvidiaCaches 'claims'
$running = @{}
$finished = 0
$split = 0
$shown = Get-Date
while ($pending.Count -or $running.Count) {
	while ($pending.Count -and $running.Count -lt $Jobs) {
		$seedFile, $shard = $pending.Dequeue()
		$name = 'shard' + ($shard -replace '/', 'of') + '-' + [IO.Path]::GetFileNameWithoutExtension($seedFile)
		$running[$name] = @($seedFile, $shard, (Start-Precompile $name @('--seeds', "`"$seedFile`"", '--shard', $shard, '--threads', "$Threads")))
	}
	Start-Sleep -Milliseconds 250
	foreach ($name in @($running.Keys)) {
		$seedFile, $shard, $process = $running[$name]
		if (!$process.HasExited) { continue }
		$running.Remove($name)
		Remove-NvidiaCache $process
		if ($process.ExitCode -eq 3) {
			$i, $n = [int[]]($shard -split '/')
			if ($n -ge 65536) { throw "$name cannot be split further; logs: $logName" }
			$pending.Enqueue(@($seedFile, "$i/$(2 * $n)"))
			$pending.Enqueue(@($seedFile, "$($i + $n)/$(2 * $n)"))
			$split++
		} elseif ($process.ExitCode -ne 0) {
			throw "$name failed; logs: $logName"
		} else {
			$finished++
		}
	}
	if (((Get-Date) - $shown).TotalSeconds -ge 60) {
		$shown = Get-Date
		Write-Host ("  {0:hh\:mm\:ss} shards: {1} done, {2} running, {3} waiting ({4} split)" -f ((Get-Date) - $begin), $finished,
			$running.Count, $pending.Count, $split)
	}
}
$env:KYTY_PRECOMPILE_CLAIMS = $null
Wait-Precompile (Start-Precompile 'merge' @('--merge', '--prune')) 'merge'
Wait-Precompile (Start-Precompile 'inputs' $inputs) 'inputs'
$cache = Get-ChildItem "$PSScriptRoot\_PipelineCache\static\$gameId.bin", "$PSScriptRoot\_PipelineCache\static\$gameId.binaries" -ErrorAction SilentlyContinue |
	Sort-Object LastWriteTime | Select-Object -Last 1
Write-Host ("done in {0:hh\:mm\:ss}: {1} ({2:N0} MB; {3} shards split)" -f ((Get-Date) - $begin), $cache.FullName, ($cache.Length / 1MB), $split)
} finally {
	# A failed shard ends the run: the others are stopped too.
	if ($running) { foreach ($entry in $running.Values) { if (!$entry[2].HasExited) { $entry[2].Kill(); $entry[2].WaitForExit() } } }
	$env:__GL_SHADER_DISK_CACHE_PATH = $null
	$env:KYTY_PRECOMPILE_CLAIMS = $null
	Remove-Item -Recurse -Force $nvidiaCaches -ErrorAction SilentlyContinue
}
