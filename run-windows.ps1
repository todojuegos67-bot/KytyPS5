# Windows launcher: runs the Windows build with the switches of a launch config.
#   .\run-windows.ps1                        the release package's switches (run-windows.json), 2560x1440
#   .\run-windows.ps1 -Precompile            compile every recorded shader and pipeline, then exit
#                                            (every shader of the game: precompile-windows.ps1)
#   .\run-windows.ps1 -Baseline              no performance switches
#   .\run-windows.ps1 -Config <launch.json>  another checkpoint's switches
#   .\run-windows.ps1 -Width 1920 -Height 1080 -Fullscreen
#   .\run-windows.ps1 -NoRedZone             without guest red-zone protection
#   .\run-windows.ps1 -Patch <cheat.json>    apply an etaHEN-style game patch
#   .\run-windows.ps1 -Fullscreen -AspectFit keep the game's 16:9 (black bars) instead of stretching
#   .\run-windows.ps1 -Language 11           the console language (0-29 as the PS5 numbers them: 1 English (US),
#                                            11 Chinese (Simplified); default: the config's)
#   .\run-windows.ps1 -Set KEY=VALUE         override a switch of the config (KEY= removes it)
#   .\run-windows.ps1 -NoAsyncShaders       new pipelines compile before their first draw (stutters; the
#                                            default compiles them on workers without stopping the frame)
#   .\run-windows.ps1 -FrameGen 1            DLSS frame generation, 1 generated frame per rendered
#                                            frame (2x); needs _Build\deps\streamline\sdk
#   .\run-windows.ps1 -Fps120                the game's frame rate up to 120 fps instead of 60 (a 120 Hz
#                                            virtual vblank; movies keep 60 flips a second)
#   .\run-windows.ps1 -Vblank 240            another virtual vblank rate (default 60, the console's)
#   .\run-windows.ps1 -Game <folder>         the game (the folder with eboot.bin); remembered in
#                                            game-path.txt, a folder dialog when none is known
#   .\run-windows.ps1 -Affinity FFFFCF       only these CPUs (hex mask; default: the config's list, else all)
#   .\run-windows.ps1 -Prompt                a dialog first when the game version is untested or the
#                                            shaders are not precompiled for this GPU (run.cmd)
#   .\run-windows.ps1 -Follow                stay open showing the run log until the game exits, then its
#                                            exit code (run.cmd)
#   .\run-windows.ps1 -DryRun                print environment and command only
# A release package (.github/workflows/build.yml) has its kyty_emulator.exe and launch.json next to this
# script: those are used instead of the build tree's. A build tree takes run-windows.json (the release
# package's switches; an older _Build\release-stage1-20260927\launch.json lacks the table modes).
param(
	[string]$Config = $(@("$PSScriptRoot\launch.json", "$PSScriptRoot\run-windows.json") | Where-Object { Test-Path $_ } |
	                    Select-Object -First 1),
	[string]$Game = '',
	[string]$Exe = $(if (Test-Path "$PSScriptRoot\kyty_emulator.exe") { "$PSScriptRoot\kyty_emulator.exe" } else { "$PSScriptRoot\_Build\windows\kyty_emulator.exe" }),
	[int]$Width = 0,
	[int]$Height = 0,
	[switch]$Fullscreen,
	[switch]$Baseline,
	[switch]$Precompile,
	[int]$Threads = 0,
	[switch]$NoRedZone,
	[switch]$NoAot,
	[string]$PresentMode = '',
	[int]$Vblank = 0,
	[switch]$Fps120,
	[string[]]$Set = @(),
	[string]$Patch = '',
	[switch]$AspectFit,
	[int]$Language = -1,
	[int]$FrameGen = 0,
	[switch]$NoAsyncShaders,
	[string]$Affinity = '',
	[switch]$Prompt,
	[switch]$Follow,
	[switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path $Exe)) { throw "missing $Exe; build it with build-windows.cmd" }

# Dialogs: sharp on scaled displays (the process is DPI aware before its first window).
function Initialize-Dialogs {
	Add-Type -AssemblyName System.Windows.Forms, System.Drawing
	Add-Type -Namespace Kyty -Name Dpi -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
	[Kyty.Dpi]::SetProcessDPIAware() | Out-Null
	[System.Windows.Forms.Application]::EnableVisualStyles()
}
# A message with a button per choice and an optional check box: the chosen index (-1: closed) and
# whether the box was checked.
function Show-Choice([string]$message, [string[]]$choices, [string]$checkbox = '') {
	$form = New-Object System.Windows.Forms.Form -Property @{
		Text = 'KytyPS5'; FormBorderStyle = 'FixedDialog'; MaximizeBox = $false; MinimizeBox = $false; StartPosition = 'CenterScreen'
		AutoSize = $true; AutoSizeMode = 'GrowAndShrink'; Padding = (New-Object System.Windows.Forms.Padding 16)
		Font = (New-Object System.Drawing.Font 'Segoe UI', 10); TopMost = $true; Tag = -1 }
	$panel = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ FlowDirection = 'TopDown'; WrapContents = $false; AutoSize = $true }
	# Lines of about 70 characters (the font's height follows the display's scale).
	$panel.Controls.Add((New-Object System.Windows.Forms.Label -Property @{ Text = $message; AutoSize = $true
		MaximumSize = (New-Object System.Drawing.Size ($form.Font.Height * 30), 0) }))
	$check = New-Object System.Windows.Forms.CheckBox -Property @{ Text = $checkbox; AutoSize = $true; Visible = [bool]$checkbox }
	$panel.Controls.Add($check)
	$row = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ AutoSize = $true }
	for ($i = 0; $i -lt $choices.Count; $i++) {
		$button = New-Object System.Windows.Forms.Button -Property @{ Text = $choices[$i]; Tag = $i; AutoSize = $true; Padding = (New-Object System.Windows.Forms.Padding 8, 4, 8, 4) }
		$button.Add_Click({ $form.Tag = $this.Tag; $form.Close() })
		$row.Controls.Add($button)
	}
	$panel.Controls.Add($row)
	$form.Controls.Add($panel)
	[void]$form.ShowDialog()
	return $form.Tag, $check.Checked
}

# Caches made from the game's files are named by its title and version (seeds-<title>_<version>.seeds,
# _PipelineCache\static\<title>_<version>.*, the warmup recordings; versions need not share shaders). Those named
# by the title alone are from before: the game's that was played last (the remembered one), so they take its
# name (precompile-windows.ps1 has the same).
function Rename-TitleCaches([string]$game) {
	$info = if (Test-Path "$game\sce_sys\param.json") { Get-Content "$game\sce_sys\param.json" -Raw -Encoding UTF8 | ConvertFrom-Json }
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

# The game: -Game, else the last one given, else the default folder; a folder dialog when that has
# no eboot.bin (a portable package started by double-clicking run.cmd).
$gameFile = "$PSScriptRoot\game-path.txt"
$remember = [bool]$Game
$lastGame = if (Test-Path $gameFile) { "$(Get-Content $gameFile -Raw)".Trim() }
if (!$lastGame) { $lastGame = "$env:USERPROFILE\Documents\PPSA01341-app0" }
if (!$DryRun) { Rename-TitleCaches $lastGame }
if (!$Game) { $Game = $lastGame }
if ($Prompt -or (!$remember -and !(Test-Path "$Game\eboot.bin"))) { Initialize-Dialogs }
if (!$remember -and !(Test-Path "$Game\eboot.bin")) {
	$dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{ Description = 'Choose the game folder (the one with eboot.bin and sce_sys)' }
	if ($dialog.ShowDialog() -eq 'OK') { $Game = $dialog.SelectedPath; $remember = $true }
}
if (!(Test-Path "$Game\eboot.bin")) { throw "no eboot.bin in $Game" }
if ($remember) { Set-Content $gameFile $Game -Encoding UTF8 }
# Its title and version (sce_sys\param.json): the emulator is tested with one of them.
$testedVersions = @('PPSA01341 01.007.000', 'PPSA01341 01.005.000')
$param = if (Test-Path "$Game\sce_sys\param.json") { Get-Content "$Game\sce_sys\param.json" -Raw -Encoding UTF8 | ConvertFrom-Json }
$titleId = if ($param) { $param.titleId } else { '' }
$version = if ($param) { "$titleId $($param.contentVersion)" } else { 'unknown' }
$titleName = if ($param) { $param.localizedParameters.($param.localizedParameters.defaultLanguage).titleName }
$tested = $testedVersions -contains $version

$launch = Get-Content $Config -Raw | ConvertFrom-Json

# Emulator options: everything after the binary in the Linux command, minus --game.
$command = @($launch.command)
$start = [Array]::IndexOf($command, '--') + 2
$options = New-Object System.Collections.Generic.List[string]
for ($i = $start; $i -lt $command.Count; $i++) {
	if ($command[$i] -eq '--game') { $i++; continue }
	$options.Add($command[$i])
}
function Set-Option([string]$name, [string]$value) {
	$at = $options.IndexOf($name)
	if ($at -ge 0) { $options[$at + 1] = $value } else { $options.Add($name); $options.Add($value) }
}
if ($Width -gt 0) { Set-Option '--screen-width' "$Width" }
if ($Height -gt 0) { Set-Option '--screen-height' "$Height" }
if ($Fullscreen) { $options.Add('--fullscreen') }
if ($PresentMode) { Set-Option '--present-mode' $PresentMode }
if ($Language -ge 0) { Set-Option '--console-language' "$Language" }
# The flip queue, one flip per vblank, is what caps the game's frame rate: it never waits for a
# vblank or a flip on the CPU, its submissions block once the flips queue up. At the console's
# 60 Hz the game runs at most 60 fps; frames that miss a vblank still queue (no 30 fps lock).
# -Fps120: a 120 Hz vblank, the game's own frames up to 120 fps (Stonefang standing 60 -> 79 fps).
# Play keeps its speed (its clock is real time: a 0.7 s camera turn and a 0.6 s walk end in the
# same place at 79 and 60 fps), but movies advance a frame per flip and play faster (start-up to
# the title 70 -> 54 s). With the Linux configs' 240 Hz the intro played about four times too fast.
Set-Option '--vblank-frequency' "$(if ($Vblank -gt 0) { $Vblank } elseif ($Fps120) { 120 } else { 60 })"
if ($Patch) { Set-Option '--game-patch' (Resolve-Path $Patch).Path }
# Windows dispatches exceptions on the faulting thread's stack, over the guest's SysV red
# zone (Linux signal delivery skips it). The write-tracking faults of the performance paths
# then corrupt guest locals; --redzone rewrites the guest's red-zone accesses at load time.
if (!$NoRedZone) { $options.Add('--redzone') }

# Environment switches. The ahead-of-time SRT library of a Linux config is a .so: use the DLL
# built from the same plan sources (tools\local\compile-srt-aot-windows.py <library dir>).
$environment = [ordered]@{}
if (!$Baseline) {
	foreach ($property in $launch.environment.PSObject.Properties) {
		if ($property.Name -eq 'KYTY_SRT_AOT_LIBRARY') {
			if ($NoAot) { continue }
			$relative = ([string]$property.Value -replace '^.*?/_Build/', '_Build/') -replace 'srt-aot\.so$', 'srt-aot.dll'
			$dll = Join-Path $PSScriptRoot $relative
			# A library built on Windows from this tree's own plan exports (KYTY_SRT_AOT_EXPORT, then
			# compile-srt-aot-windows.py) matches plans the Linux one misses: the walk's plans
			# changed since it was built (28% of them ran in the interpreter).
			$windows = Get-ChildItem "$PSScriptRoot\_Build\srt-aot\windows-libraries\*\srt-aot.dll" -ErrorAction SilentlyContinue |
				Sort-Object LastWriteTime -Descending | Select-Object -First 1
			if ($windows) { $dll = $windows.FullName }
			if (Test-Path $dll) {
				$environment[$property.Name] = (Resolve-Path $dll).Path
			} else {
				Write-Host "SRT AOT: $relative missing; build it with tools\local\compile-srt-aot-windows.py"
			}
			continue
		}
		$environment[$property.Name] = [string]$property.Value
	}
	# A build tree with an SRT AOT library built here (tools\local\compile-srt-aot-windows.py) uses it also when
	# the config names none (run-windows.json: release packages have none).
	if (!$NoAot -and !$environment.Contains('KYTY_SRT_AOT_LIBRARY')) {
		$windows = Get-ChildItem "$PSScriptRoot\_Build\srt-aot\windows-libraries\*\srt-aot.dll" -ErrorAction SilentlyContinue |
			Sort-Object LastWriteTime -Descending | Select-Object -First 1
		if ($windows) { $environment['KYTY_SRT_AOT_LIBRARY'] = $windows.FullName }
	}
}

# CPU affinity: -Affinity, else the config's list (this PC's leaves CPUs 4 and 5 out), else every
# CPU; only CPUs this PC has.
$all = if ([Environment]::ProcessorCount -ge 64) { [int64]-1 } else { ([int64]1 -shl [Environment]::ProcessorCount) - 1 }
$mask = [int64]0
if ($Affinity) { $mask = [Convert]::ToInt64(($Affinity -replace '^0x'), 16) } else { foreach ($cpu in $launch.cpu_affinity) { $mask = $mask -bor ([int64]1 -shl [int]$cpu) } }
$mask = $mask -band $all
if ($mask -eq 0) { $mask = $all }
$cpus = 0
for ($bit = 0; $bit -lt 64; $bit++) { if ($mask -band ([int64]1 -shl $bit)) { $cpus++ } }

# The render thread on the recording worker's CPUs (this PC's P-cores without CPU 0, which takes
# most interrupts on Windows): +2.5% over free placement in the fixed scene. Pinning it to CPU 0
# alone, as on Linux, halved the frame rate here. A config without them leaves both to Windows;
# "auto" (the portable package's config) picks the performance cores of this PC (below).
if (!$environment.Contains('KYTY_RENDER_CPUS') -and $environment.Contains('KYTY_RECORDING_CPUS')) {
	$environment['KYTY_RENDER_CPUS'] = $environment['KYTY_RECORDING_CPUS']
}

# Image staging copies on the upload worker too (KYTY_ASYNC_UPLOAD=2): streamed textures were
# up to 4 MB of backing reads a frame on the render thread while walking.
if ($environment['KYTY_ASYNC_UPLOAD'] -eq '1') { $environment['KYTY_ASYNC_UPLOAD'] = '2' }
# Retired buffers' memory is freed on a worker (a kernel call per dedicated allocation).
if (!$environment.Contains('KYTY_BUFFER_RECLAIM') -and !$Baseline) { $environment['KYTY_BUFFER_RECLAIM'] = '1' }
# Ordinary indexed/auto draws recorded as packets too (about 1.3 ms a frame less render time).
if (!$environment.Contains('KYTY_DRAW_PACKETS') -and !$Baseline) { $environment['KYTY_DRAW_PACKETS'] = '1' }
# A CPU write into a large image re-uploads only the written part: the game streams textures one
# layer at a time into 320-352 MiB arrays, and each layer re-uploaded the whole array (20-70 ms).
if (!$environment.Contains('KYTY_PARTIAL_IMAGE_DIRTY') -and !$Baseline) { $environment['KYTY_PARTIAL_IMAGE_DIRTY'] = '1' }
# ... and inside a large subresource only the rows of tile blocks over the written ranges: an
# 8192x8192 streamed texture's mip 0 is 64 MiB of 85, and picking it whole re-uploaded all of it.
if (!$environment.Contains('KYTY_PARTIAL_ROW_BANDS') -and !$Baseline) { $environment['KYTY_PARTIAL_ROW_BANDS'] = '1' }
# Native XPR pipeline variants compile on worker threads; their draws take the normal path until
# then. Entering a new area compiled dozens at once on the render thread (a 200 ms frame).
if (!$environment.Contains('KYTY_ASYNC_XPR_PIPELINES') -and !$Baseline) { $environment['KYTY_ASYNC_XPR_PIPELINES'] = '1' }
# Direct draws whose objects point at a new copy of their tables every frame keep their native XPR
# records (the pointer leaves the key): Latria 33 -> 37 fps, Boletaria 1-1 31 -> 32.
if (!$environment.Contains('KYTY_NATIVE_XPR_RELOCATE') -and !$Baseline) { $environment['KYTY_NATIVE_XPR_RELOCATE'] = '1' }
# At most 256 native XPR record stores a frame (about 16 us each): walking brings a thousand or more new objects
# and LODs a frame, and storing them all at once made 60-90 ms frames (Latria walk 1% low 14.1 -> 16.4 fps).
if (!$environment.Contains('KYTY_NATIVE_XPR_STORE_BUDGET') -and !$Baseline) { $environment['KYTY_NATIVE_XPR_STORE_BUDGET'] = '256' }
# Native XPR records stay 600 frames unused (2 before): a turn of the camera no longer drops every record
# behind it (camera turning in place: Latria 35.2 -> 39.1 fps, Boletaria 1-1 42.0 -> 47.0).
if (!$environment.Contains('KYTY_NATIVE_XPR_KEEP_FRAMES') -and !$Baseline) { $environment['KYTY_NATIVE_XPR_KEEP_FRAMES'] = '600' }
# The instances of a mesh keep their native XPR records whatever order a frame draws them in: once an object's
# record rebinds its descriptors, its relocated key holds its descriptor bases (Latria camera turning in place
# 42.95 -> 45.33 fps, 1% low 27.95 -> 30.02; standing 39.13 -> 38.73).
if (!$environment.Contains('KYTY_NATIVE_XPR_INSTANCES') -and !$Baseline) { $environment['KYTY_NATIVE_XPR_INSTANCES'] = '1' }
# Demon's Souls' own engine settings, added to the command-line file it reads at boot (src/loader/gameArgs.h):
# no XPR index-buffer culling, a GPU pass that culls every object's triangles (~2800 compute dispatches a
# frame at Boletaria 1-1) before drawing them; the PC GPU draws them all instead, the image unchanged
# (same process at Boletaria 1-1: standing 34.30 -> 36.48 fps, camera turning +0.8%; GPU 60 -> 71% busy).
if (!$environment.Contains('KYTY_GAME_CONVARS') -and !$Baseline -and $titleId -eq 'PPSA01341') { $environment['KYTY_GAME_CONVARS'] = 'doIndexBufferCulling=false' }
# Memory the game releases (texture pool layers, 64 KiB mappings each) is not unprotected mapping
# by mapping before its unmap: VirtualProtect was ~15% of the render thread in the open area.
if (!$environment.Contains('KYTY_UNMAP_PROTECT_SKIP') -and !$Baseline) { $environment['KYTY_UNMAP_PROTECT_SKIP'] = '1' }
# The key of the driver pipeline cache (_PipelineCache\local\<key>); it also enables the shader
# warmup, which a modified source tree runs without otherwise. One key for every build: the driver
# finds a pipeline by its whole input (SPIR-V and state), so a new build reuses the pipelines of the
# shaders it left alone. Keyed by the executable's SHA-256, every update compiled the ~3700 warmup
# pipelines again (2-2.5 minutes of "Preparing pipelines").
$environment['KYTY_DRIVER_CACHE_KEY'] = '1f3c29b70e536c8f5a5e812e793b33ac2eb6858b4810ed98ca2a51b9fa5eed97'
if ($Precompile) {
	# Translate every recorded shader and create every recorded pipeline on all allowed CPUs,
	# save the driver cache, exit. Later launches warm up from that cache in seconds.
	$environment['KYTY_SHADER_WARMUP'] = '1'
	$environment['KYTY_SHADER_WARMUP_ONLY'] = '1'
	$environment['KYTY_SHADER_WARMUP_THREADS'] = "$cpus"
	$environment.Remove('KYTY_SHADER_WARMUP_SECONDS')
} else {
	# An earlier -Precompile in the same shell leaves this set; the game must not exit after warmup.
	Remove-Item env:KYTY_SHADER_WARMUP_ONLY -ErrorAction SilentlyContinue
	# At most a minute of start-up warmup: with an empty driver cache (the first launch, a driver
	# update) its ~4400 pipelines took 4.5 minutes; the rest are built at their first use (a fast
	# unoptimized build at once, the optimized one in the background) and cached from then on.
	if (!$environment.Contains('KYTY_SHADER_WARMUP_SECONDS')) { $environment['KYTY_SHADER_WARMUP_SECONDS'] = '60' }
}
if ($Threads -gt 0) { $environment['KYTY_SHADER_WARMUP_THREADS'] = "$Threads" }
# Stutters in the log: frames, shader translations and pipeline creations of 100 ms or more (SLOW lines;
# what runs per frame or on a cache miss only, so nothing measurable).
if (!$environment.Contains('KYTY_HITCH_LOG_MS')) { $environment['KYTY_HITCH_LOG_MS'] = '100' }
if ($AspectFit) { $environment['KYTY_PRESENT_ASPECT'] = 'fit' }
# Async shaders (renderDraw.cpp, KYTY_ASYNC_DRAW_PIPELINES): a draw whose pipeline is not compiled yet does
# not stop the frame; a worker compiles it and the object appears a few frames later. -NoAsyncShaders: off.
if (!$NoAsyncShaders -and !$Baseline -and !$environment.Contains('KYTY_ASYNC_DRAW_PIPELINES')) { $environment['KYTY_ASYNC_DRAW_PIPELINES'] = '1' }
if ($FrameGen -gt 0) { $environment['KYTY_FRAMEGEN'] = "$FrameGen" }
# -Set KEY=VALUE overrides a switch of the config; KEY= drops it. Several: -Set A=1,B=2 (a comma
# starts a new pair only before KEY=, so values such as 1,2,3,6,7 stay whole).
foreach ($pair in ($Set | ForEach-Object { $_ -split ',(?=[A-Za-z_][A-Za-z0-9_]*=)' })) {
	$key, $value = $pair -split '=', 2
	if ($value) { $environment[$key] = $value } else { $environment.Remove($key); Remove-Item "env:$key" -ErrorAction SilentlyContinue }
}

# The performance cores of a hybrid CPU (Windows' CPU set efficiency classes: an Intel Core with
# P- and E-cores), without the core of CPU 0, within the launch's CPUs: a list such as "1,2,3,4,5",
# or nothing when every core is alike (the threads are left to Windows there).
function Get-PerformanceCpus([int64]$allowed) {
	Add-Type -Namespace Kyty -Name CpuSets -MemberDefinition @'
[DllImport("kernel32.dll")]
static extern bool GetSystemCpuSetInformation(IntPtr information, uint length, out uint returned, IntPtr process, uint flags);
// Group 0's logical processors as (index, core, efficiency class) triples.
public static int[] Query() {
	var result = new System.Collections.Generic.List<int>();
	uint length;
	GetSystemCpuSetInformation(IntPtr.Zero, 0, out length, IntPtr.Zero, 0);
	if (length == 0) return result.ToArray();
	IntPtr buffer = Marshal.AllocHGlobal((int)length);
	try {
		if (!GetSystemCpuSetInformation(buffer, length, out length, IntPtr.Zero, 0)) return result.ToArray();
		for (int offset = 0; offset < length; offset += Marshal.ReadInt32(buffer, offset)) {
			if (Marshal.ReadInt32(buffer, offset + 4) != 0 || Marshal.ReadInt16(buffer, offset + 12) != 0) continue;
			result.Add(Marshal.ReadByte(buffer, offset + 14));
			result.Add(Marshal.ReadByte(buffer, offset + 15));
			result.Add(Marshal.ReadByte(buffer, offset + 18));
		}
	} finally {
		Marshal.FreeHGlobal(buffer);
	}
	return result.ToArray();
}
'@
	$values = [Kyty.CpuSets]::Query()
	$sets = for ($i = 0; $i + 2 -lt $values.Count; $i += 3) { [pscustomobject]@{ Cpu = $values[$i]; Core = $values[$i + 1]; Class = $values[$i + 2] } }
	if (@($sets | ForEach-Object Class | Sort-Object -Unique).Count -lt 2) { return '' }
	$fastest = ($sets | Measure-Object Class -Maximum).Maximum
	$interrupts = ($sets | Where-Object Cpu -eq 0).Core
	return (($sets | Where-Object { $_.Class -eq $fastest -and $_.Core -ne $interrupts -and $_.Cpu -lt 64 -and
		($allowed -band ([int64]1 -shl $_.Cpu)) } | ForEach-Object Cpu) -join ',')
}
if ('KYTY_RECORDING_CPUS', 'KYTY_RENDER_CPUS' | Where-Object { $environment[$_] -eq 'auto' }) {
	$performance = Get-PerformanceCpus $mask
	foreach ($key in 'KYTY_RECORDING_CPUS', 'KYTY_RENDER_CPUS') {
		if ($environment[$key] -ne 'auto') { continue }
		if ($performance) { $environment[$key] = $performance } else { $environment.Remove($key) }
	}
}

$quoted = @('--game', "`"$Game`"") + ($options | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } })
$logDir = if (Test-Path "$PSScriptRoot\_Build") { "$PSScriptRoot\_Build\run-logs" } else { "$PSScriptRoot\logs" }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
Write-Host "game:     $(if ($titleName) { "$titleName, " })$version$(if (!$tested) { " (untested: the emulator is tested with $($testedVersions -join ' and '))" })"
Write-Host "config:   $Config$(if ($Baseline) { ' (baseline: no switches)' })$(if ($Precompile) { ' (precompile)' })"
Write-Host ("affinity: 0x{0:X} ({1} CPUs){2}" -f $mask, $cpus, $(if ($environment['KYTY_RENDER_CPUS']) { "; render threads on CPUs $($environment['KYTY_RENDER_CPUS'])" }))
Write-Host "switches: $($environment.Count)"
Write-Host "command:  $Exe $($quoted -join ' ')"
if ($DryRun) { $environment.GetEnumerator() | ForEach-Object { "  $($_.Key)=$($_.Value)" }; return }

# Shaders, from the seed file (every shader of the game) by the precompile program: the shader
# prefetch's inputs for this GPU and driver (_PipelineCache\static\<title>_<version>.shaders; new shaders
# are then translated in the background while playing), made when they are missing or stale (the first
# launch, a driver update: half a minute on 22 CPUs); and the static pipeline cache, the whole game
# compiled ahead by precompile-windows.ps1 (44 minutes on 22 CPUs), offered by -Prompt.
$tool = Join-Path (Split-Path $Exe) 'kyty_shader_precompile.exe'
# The seed file of this game version (Rename-TitleCaches).
$seedName = if ($param -and $titleId -and $param.contentVersion) { "seeds-$($titleId)_$($param.contentVersion).seeds" } else { 'seeds.seeds' }
$seeds = @("$PSScriptRoot\$seedName", "$PSScriptRoot\_Build\static-precompile\$seedName") | Where-Object { Test-Path $_ } | Select-Object -First 1
# No seed file yet (a release has none: it holds the game's shader code): made from the game's files once, as
# precompile-windows.ps1 makes it (Python 3 with numpy; 16 s on 22 CPUs).
$generator = "$PSScriptRoot\tools\local\static-precompile\precompile.py"
if (!$seeds -and !$Precompile -and (Test-Path $tool) -and (Test-Path $generator)) {
	cmd /c 'python -c "import numpy" >nul 2>nul'
	if ($LASTEXITCODE -eq 0) {
		$made = if (Test-Path "$PSScriptRoot\_Build") { "$PSScriptRoot\_Build\static-precompile\$seedName" } else { "$PSScriptRoot\$seedName" }
		New-Item -ItemType Directory -Force (Split-Path $made) | Out-Null
		Write-Host "shaders:  listing the game's shaders from its files (once)"
		python $generator --game $Game seeds $made | Out-Null
		if ($LASTEXITCODE -eq 0 -and (Test-Path $made)) { $seeds = $made } else { Write-Host 'shaders:  listing failed: the game compiles its shaders when they first appear' }
	} else {
		Write-Host 'shaders:  no Python 3 with numpy (winget install Python.Python.3.12, then pip install numpy): the game compiles its shaders when they first appear'
	}
}
$shaders = !$Precompile -and $seeds -and (Test-Path $tool)
# The precompile program on the launch's CPUs, its output in the console or a file: its exit code.
function Invoke-Tool([string[]]$arguments, [string]$output = '') {
	$parameters = @{ FilePath = $tool; WorkingDirectory = $PSScriptRoot; NoNewWindow = $true; PassThru = $true
		ArgumentList = @('--game', "`"$Game`"", '--seeds', "`"$seeds`"") + $arguments }
	if ($output) { $parameters['RedirectStandardOutput'] = $output }
	$process = Start-Process @parameters
	$null = $process.Handle # keeps the exit code readable
	if ($mask -ne $all) { $process.ProcessorAffinity = [IntPtr]$mask }
	$process.WaitForExit()
	return $process.ExitCode
}
$inputsReady = $cacheReady = $false
if ($shaders) {
	$statusFile = [IO.Path]::GetTempFileName()
	$shaders = (Invoke-Tool @('--status') $statusFile) -eq 0
	$status = Get-Content $statusFile -Raw
	Remove-Item $statusFile
	$inputsReady = $status -match 'inputs current'
	# A precompile that stopped before its last merge keeps its shards' checkpoints (small ones stay
	# behind after a merge).
	$cacheReady = $status -match 'static cache current' -and
		!(Get-ChildItem "$PSScriptRoot\_PipelineCache\static\*.shard*" -ErrorAction SilentlyContinue | Where-Object Length -gt 1MB)
	Write-Host "shaders:  prefetch inputs $(if ($inputsReady) { 'ready' } else { 'to be made' }), static pipeline cache $(if ($cacheReady) { 'ready' } else { 'not made (precompile-windows.ps1)' })"
}

# -Prompt: the game version and the precompile, before anything starts.
$noPrompt = "$PSScriptRoot\no-precompile-prompt.txt"
$offer = $shaders -and !$cacheReady -and !(Test-Path $noPrompt)
if ($Prompt -and ($offer -or !$tested)) {
	$info = "Game: $(if ($titleName) { $titleName } else { 'unknown' }) ($version)"
	$info += if ($tested) { ', a tested version.' } else { "`n⚠ This version is untested (the emulator was tested with $($testedVersions -join ' and ')): it may not run or may fail." }
	if ($offer) {
		$minutes = [math]::Ceiling(44 * 22 / $cpus / 10) * 10
		$time = if ($minutes -lt 90) { "about $minutes minutes" } else { 'about {0:N1} hours' -f ($minutes / 60) }
		$message = "$info`n`nThe shaders are not precompiled for this graphics card yet: scenes and effects stutter the first time they appear in the game (a fraction of a second to a few seconds).`n" +
			"The precompile compiles every shader of the game once: $time on this PC (with the CPU fully loaded; closing its window stops it, and the next run continues). " +
			"It is needed only once, and again after a graphics driver update."
		$choice, $never = Show-Choice $message @('Precompile first, then start the game', 'Start the game now', 'Quit') "Don't ask about precompiling again"
		if ($never) { Set-Content $noPrompt 'run-windows.ps1 -Prompt: no precompile dialog (delete this file to get it back)' }
		if ($choice -ne 0 -and $choice -ne 1) { Write-Host 'cancelled'; return }
		if ($choice -eq 0) {
			try {
				& "$PSScriptRoot\precompile-windows.ps1" -Game $Game -Affinity $mask
				$inputsReady = $true # the full run ends with them
			} catch {
				Write-Host "precompile failed ($($_.Exception.Message)); starting the game anyway"
			}
		}
	} else {
		$choice, $never = Show-Choice $info @('Start the game', 'Quit')
		if ($choice -ne 0) { Write-Host 'cancelled'; return }
	}
}
# Memory: Windows ends a program that asks for more than RAM and the page file can hold. The emulator
# commits about 34 GB (the game's 13.5 GiB of PS5 memory, about 11 GB Windows sets aside to back the
# video memory in use, the emulator's own) and keeps about 20 GB in RAM.
$os = Get-CimInstance Win32_OperatingSystem
[double]$ram = $os.TotalVisibleMemorySize * 1KB
[double]$commit = $os.FreeVirtualMemory * 1KB
if ((Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile) {
	# A system-managed page file grows to 3 x RAM (at most an eighth of its disk) while the disk has room.
	$disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$env:SystemDrive'"
	[double]$paged = (Get-CimInstance Win32_PageFileUsage | Measure-Object AllocatedBaseSize -Sum).Sum * 1MB
	[double]$room = [math]::Min([math]::Min(3 * $ram, [double]$disk.Size / 8) - $paged, [double]$disk.FreeSpace - 5GB)
	$commit += [math]::Max([double]0, $room)
}
$memory = @()
if ($ram -lt 24GB) {
	$memory += "This PC has {0:N0} GB of RAM and the emulator keeps about 20 GB in use (32 GB recommended): expect long stutters." -f ($ram / 1GB)
}
if ($commit -lt 34GB) {
	$memory += ("Windows can give programs only {0:N0} GB more memory (RAM plus page file) and the emulator needs about 34 GB: " +
		"it may close in the middle of the game. Close other programs, or set the page file to 'System managed size' " +
		"(Settings > System > About > Advanced system settings > Performance > Settings > Advanced > Virtual memory).") -f ($commit / 1GB)
}
foreach ($line in $memory) { Write-Host "memory:   $line" }
if ($memory -and $Prompt) {
	$choice, $never = Show-Choice ($memory -join "`n`n") @('Start the game anyway', 'Quit')
	if ($choice -ne 0) { Write-Host 'cancelled'; return }
}

if ($shaders -and !$inputsReady) {
	Write-Host 'shaders:  making the background shader preparation inputs for this graphics card (first run or after a driver update: about a minute)'
	if ((Invoke-Tool @('--no-pipelines', '--static-inputs', '--threads', "$cpus")) -ne 0) {
		Write-Host 'shaders:  making them failed; no background shader preparation this time'
	}
}

foreach ($entry in $environment.GetEnumerator()) { Set-Item "env:$($entry.Key)" $entry.Value }
New-Item -ItemType Directory -Force $logDir | Out-Null
$out = "$logDir\$stamp.out.log"
$process = Start-Process -FilePath $Exe -ArgumentList $quoted -WorkingDirectory $PSScriptRoot -PassThru `
	-RedirectStandardOutput $out -RedirectStandardError "$logDir\$stamp.err.log"
if ($mask -ne $all) { $process.ProcessorAffinity = [IntPtr]$mask }
$null = $process.Handle # keeps the exit code readable after the process ends
Write-Host "pid $($process.Id); logs: $($logDir.Substring($PSScriptRoot.Length + 1))\$stamp.*.log"
if (!$Precompile -and $Follow) {
	# The run log in this window as it is written; the exit code (and the error log's end) when the game ends.
	$reader = $null
	$pending = ''
	$show = {
		$text = $reader.ReadToEnd()
		if ($text) {
			$lines = ($pending + $text) -split "`n"
			$script:pending = $lines[-1]
			for ($i = 0; $i -lt $lines.Count - 1; $i++) { Write-Host $lines[$i].TrimEnd("`r") }
		}
	}
	while ($true) {
		$exited = $process.HasExited
		if (!$reader -and (Test-Path $out)) {
			$reader = New-Object System.IO.StreamReader([System.IO.FileStream]::new($out, 'Open', 'Read', 'ReadWrite'))
		}
		if ($reader) { & $show }
		if ($exited) { break }
		Start-Sleep -Milliseconds 250
	}
	if ($pending) { Write-Host $pending }
	if ($reader) { $reader.Dispose() }
	Write-Host ''
	Write-Host ("The game exited with code {0} (0x{0:X8}); logs: {1}\{2}.*.log" -f $process.ExitCode, $logDir, $stamp)
	if ($process.ExitCode -ne 0) {
		Get-Content "$logDir\$stamp.err.log" -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
	}
	exit $process.ExitCode
}
if (!$Precompile) { return }

# Precompile: follow the warmup progress until the emulator exits.
$begin = Get-Date
$shown = 0
while (!$process.HasExited) {
	Start-Sleep -Milliseconds 500
	$lines = @(Get-Content $out -ErrorAction SilentlyContinue | Where-Object { $_ -match 'warmup|precompile|pipeline cache' })
	for (; $shown -lt $lines.Count; $shown++) { Write-Host "  $($lines[$shown])" }
}
$lines = @(Get-Content $out -ErrorAction SilentlyContinue | Where-Object { $_ -match 'warmup|precompile|pipeline cache' })
for (; $shown -lt $lines.Count; $shown++) { Write-Host "  $($lines[$shown])" }
Write-Host ("exit code {0} after {1:N0} s" -f $process.ExitCode, ((Get-Date) - $begin).TotalSeconds)
exit $process.ExitCode
