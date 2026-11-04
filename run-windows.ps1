# Windows launcher: runs the Windows build with the command line of a launch config.
#   .\run-windows.ps1                        the release package's command line (run-windows.json), 2560x1440
#   .\run-windows.ps1 -Precompile            compile every recorded shader and pipeline, then exit
#                                            (every shader of the game: precompile-windows.ps1)
#   .\run-windows.ps1 -Config <launch.json>  another launch config
#   .\run-windows.ps1 -Width 1920 -Height 1080 -Fullscreen
#   .\run-windows.ps1 -NoRedZone             without guest red-zone protection
#   .\run-windows.ps1 -Patch <cheat.json>    apply an etaHEN-style game patch
#   .\run-windows.ps1 -Fullscreen -AspectFit keep the game's 16:9 (black bars) instead of stretching
#   .\run-windows.ps1 -Hdr                   HDR10 output when Windows HDR is on for the display (the game is
#                                            told the TV takes HDR and outputs it; SDR otherwise)
#   .\run-windows.ps1 -Language 11           the console language (0-29 as the PS5 numbers them: 1 English (US),
#                                            11 Chinese (Simplified); default: the config's)
#   .\run-windows.ps1 -Set KEY=VALUE         an emulator switch (KYTY_*; the defaults are the emulator's own,
#                                            src/local/performance-switches.h; KEY= removes one the config sets)
#   .\run-windows.ps1 -FrameGen 1            DLSS frame generation, 1 generated frame per rendered
#                                            frame (2x; up to 5, at most what the GPU makes); needs
#                                            NVIDIA's Streamline runtime (-StreamlineDir)
#   .\run-windows.ps1 -Upscaler dlss         DLSS super resolution: Demon's Souls' own PS5 Pro path (its 3D at
#                                            2688x1512 or -RenderHeight, 1080 and up or 720), DLSS in place of Sony's
#                                            upscaler making the 3840x2160 image; NVIDIA's Streamline runtime
#   .\run-windows.ps1 -Upscaler fsr          the same with AMD's FSR 3 built in (any graphics card, nothing to download);
#                                            with AMD's FSR SDK 2.4 DLLs next to the emulator, FSR 4 on Radeon RX 7000/9000
#   .\run-windows.ps1 -StreamlineDir <dir>   where NVIDIA's Streamline runtime is (the Streamline SDK
#                                            or its bin\x64; default: a "streamline" folder next to
#                                            the emulator)
#   .\run-windows.ps1 -Fsr                   AMD FSR frame generation instead (2x, any GPU): AMD's FSR SDK 2.4 DLLs
#                                            (FSR 4 on Radeon RX 9000) or amd_fidelityfx_vk.dll next to the emulator
#   .\run-windows.ps1 -Fps120                the game's frame rate up to 120 fps instead of 60 (a 120 Hz
#                                            virtual vblank; movies, menus and loading screens keep 60)
#   .\run-windows.ps1 -Vblank 240            another virtual vblank rate (default 60, the console's)
#   .\run-windows.ps1 -LowLatency            one frame queued instead of about two: what you press shows sooner (at
#                                            60 Hz 15-23 ms from the game's frame to the screen instead of ~30); a late
#                                            frame is no longer covered by a queued one (KYTY_LOW_LATENCY)
#   .\run-windows.ps1 -RenderHeight 1080     Demon's Souls renders its 3D at 1920x1080 instead of 2560x1440
#                                            (720-1440; its 4K output and UI stay): faster where the GPU limits
#   .\run-windows.ps1 -HdOutput              Demon's Souls' own 1080p mode (a PS5's on a 1080p TV): its post-processing
#                                            and HUD at 1920x1080 instead of 3840x2160, its 3D at most 1080 high; less
#                                            work for the GPU (not with -Upscaler dlss, whose output is 4K)
#   .\run-windows.ps1 -Upscaler dlss -OutputHeight 1440   the upscaler's output (and the frame on it) at 2560x1440
#                                            instead of 3840x2160, for a 1440p screen (1080-2159; the HUD in its HD
#                                            layout scaled to it); -RenderHeight picks the mode (1440 DLAA, 960 Quality)
#   .\run-windows.ps1 -Aspect 21:9           Demon's Souls renders its 3D wider (21:9 = 64:27, up to 32:9; 3416x1440
#                                            at 21:9), seeing more to the sides; shown at that aspect (the HUD at
#                                            its sides unstretched; menus and movies stretched, or with -AspectFit
#                                            movies, the title and loading screens kept at 16:9 with black bars)
#   .\run-windows.ps1 -Quality low            Demon's Souls draws less (low / lowest: shorter shadows and
#                                            distant detail): faster where the CPU limits
#   .\run-windows.ps1 -Antialiasing off      Demon's Souls without its own anti-aliasing (TAA) and the camera jitter
#                                            that goes with it: a sharp picture with jagged edges (with the game's own
#                                            upscaler only; at a render height equal to the output, nothing is scaled)
#   .\run-windows.ps1 -TextureReduce 1       large textures without their largest level: half the size per side
#                                            (2: a quarter; 0: whole), less video memory and upload; default 1
#                                            with less than 12 GB of video memory, else 0 (the launcher's Graphics
#                                            Low / Lowest pass 1 / 2)
#   .\run-windows.ps1 -Game <folder>         the game (the folder with eboot.bin); remembered in
#                                            game-path.txt, a folder dialog when none is known
#   .\run-windows.ps1 -Game <file.zar>       the game folder packed into a ZArchive (read without extracting it)
#   .\run-windows.ps1 -Affinity FFFFCF       only these CPUs (hex mask; default: the config's list, else all)
#   .\run-windows.ps1 -Prompt                a dialog first when the game version is untested or the
#                                            shaders are not precompiled for this GPU (run.cmd)
#   .\run-windows.ps1 -Follow                stay open showing the run log until the game exits, then its
#                                            exit code (run.cmd)
#   .\run-windows.ps1 -DryRun                print environment and command only
# A release package (.github/workflows/build.yml) has its kyty_emulator.exe and launch.json next to this
# script: those are used instead of the build tree's. A build tree takes run-windows.json (the release
# package's launch.json).
param(
	[string]$Config = $(@("$PSScriptRoot\launch.json", "$PSScriptRoot\run-windows.json") | Where-Object { Test-Path $_ } |
	                    Select-Object -First 1),
	[string]$Game = '',
	[string]$Exe = $(if (Test-Path "$PSScriptRoot\kyty_emulator.exe") { "$PSScriptRoot\kyty_emulator.exe" } else { "$PSScriptRoot\_Build\windows\kyty_emulator.exe" }),
	[int]$Width = 0,
	[int]$Height = 0,
	[switch]$Fullscreen,
	[switch]$Hdr,
	[switch]$Precompile,
	[int]$Threads = 0,
	[switch]$NoRedZone,
	[string]$PresentMode = '',
	[int]$Vblank = 0,
	[switch]$Fps120,
	[switch]$LowLatency,
	[switch]$KeyboardMouseControls,
	[string]$KeyboardMouseBindings = '',
	[int]$RenderHeight = 0,
	[switch]$HdOutput,
	[int]$OutputHeight = 0,
	[string]$Aspect = '',
	[string]$Quality = '',
	[ValidateSet('', 'off')][string]$Antialiasing = '',
	[string]$TextureReduce = '',
	[string[]]$Set = @(),
	[string]$Patch = '',
	[switch]$AspectFit,
	[int]$Language = -1,
	[int]$FrameGen = 0,
	[ValidateSet('', 'dlss', 'fsr')][string]$Upscaler = '',
	[string]$StreamlineDir = '',
	[switch]$Fsr,
	[string]$Affinity = '',
	[string]$Gpu = '',
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

# A game: its folder (eboot.bin and sce_sys) or the folder packed into a ZArchive (a .zar file, read by the emulator
# without extracting it).
function Test-Game([string]$game) {
	$game -and ((Test-Path -LiteralPath "$game\eboot.bin") -or ($game -match '\.zar$' -and (Test-Path -LiteralPath $game -PathType Leaf)))
}
# The game's sce_sys\param.json (its title and version): a .zar's is copied out by the precompile program.
function Read-GameParam([string]$game) {
	if (!$game) { return $null }
	if (Test-Path -LiteralPath "$game\sce_sys\param.json") { return Get-Content -LiteralPath "$game\sce_sys\param.json" -Raw -Encoding UTF8 | ConvertFrom-Json }
	$tool = Join-Path (Split-Path $Exe) 'kyty_shader_precompile.exe'
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
# name (precompile-windows.ps1 has the same).
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

# The game: -Game, else the last one given, else the default folder; a folder dialog when that has
# no eboot.bin (a portable package started by double-clicking run.cmd). A .zar archive of the folder
# is a game too (-Game, or the launcher's settings window).
$gameFile = "$PSScriptRoot\game-path.txt"
$remember = [bool]$Game
$lastGame = if (Test-Path $gameFile) { "$(Get-Content $gameFile -Raw)".Trim() }
if (!$lastGame) { $lastGame = "$env:USERPROFILE\Documents\PPSA01341-app0" }
if (!$DryRun) { Rename-TitleCaches $lastGame }
if (!$Game) { $Game = $lastGame }
if ($Prompt -or (!$remember -and !(Test-Game $Game))) { Initialize-Dialogs }
if (!$remember -and !(Test-Game $Game)) {
	$dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{ Description = 'Choose the game folder (the one with eboot.bin and sce_sys)' }
	if ($dialog.ShowDialog() -eq 'OK') { $Game = $dialog.SelectedPath; $remember = $true }
}
if (!(Test-Game $Game)) { throw "no eboot.bin in $Game (nor a .zar archive)" }
if ($remember) { Set-Content $gameFile $Game -Encoding UTF8 }
# Its title and version (sce_sys\param.json): the emulator is tested with one of them.
$testedVersions = @('PPSA01341 01.007.000', 'PPSA01341 01.005.000')
$param = Read-GameParam $Game
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
# The launcher's keyboard prompt artwork matches the requested layout. Append its bindings only
# for this launch: the emulator resolves conflicts in favour of the last mapping.
# Existing sensitivity and non-conflicting mouse-wheel bindings remain intact.
# launch.json is never rewritten; switching off restores the user's normal bindings.
if ($KeyboardMouseControls) {
	$bindings = @(
		'Cross=E', 'Circle=Space', 'Square=R', 'Triangle=F',
		'L1=Left Shift', 'L2=Left Ctrl', 'L3=X',
		'R1=Mouse:Left', 'R2=Mouse:Right', 'R3=Mouse:Middle', 'Options=Escape',
		'Up=Up', 'Down=Down', 'Left=Left', 'Right=Right',
		'TouchPad=Tab', 'TouchPadRight=Backspace',
		'LeftStickUp=W', 'LeftStickDown=S', 'LeftStickLeft=A', 'LeftStickRight=D'
	)
	if ($KeyboardMouseBindings) {
		$custom = ConvertFrom-Json -InputObject ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($KeyboardMouseBindings)))
		if ($custom -isnot [array]) { throw 'Invalid keyboard layout' }
		$controls = @($bindings | ForEach-Object { ($_ -split '=',2)[0] })
		$seen = @{}; $inputs = @{}
		foreach ($binding in $custom) {
			if ($binding -isnot [string] -or $binding -notmatch '^([A-Za-z][A-Za-z0-9]*)=([A-Za-z0-9: .+*/=\\;,\[\]''`-]+)$') { throw 'Invalid keyboard binding' }
			$control,$inputName = $binding -split '=',2
			if ($seen.ContainsKey($control)) { throw "Duplicate keyboard control: $control" }
			$seen[$control] = $true
			if ($control -eq 'MouseSensitivity') {
				$number = 0.0
				if (![double]::TryParse($inputName,[Globalization.NumberStyles]::AllowDecimalPoint,[Globalization.CultureInfo]::InvariantCulture,[ref]$number) -or $number -lt 0.1 -or $number -gt 5) { throw 'Invalid mouse sensitivity' }
			} else {
				if ($control -cnotin $controls -or $inputs.ContainsKey($inputName) -or $inputName -in @('F7','F8','F11','Pause')) { throw 'Invalid or conflicting keyboard binding' }
				$inputs[$inputName] = $true
			}
		}
		if (@($controls | Where-Object { !$seen.ContainsKey($_) }).Count) { throw 'Incomplete keyboard layout' }
		$bindings = $custom
	}
	foreach ($binding in $bindings) { $options.Add('--keymap'); $options.Add($binding) }
}
if ($Width -gt 0) { Set-Option '--screen-width' "$Width" }
if ($Height -gt 0) { Set-Option '--screen-height' "$Height" }
if ($Fullscreen) { $options.Add('--fullscreen') }
if ($Hdr) { $options.Add('--hdr') }
if ($PresentMode) { Set-Option '--present-mode' $PresentMode }
if ($Language -ge 0) { Set-Option '--console-language' "$Language" }
# The flip queue, one flip per vblank, is what caps the game's frame rate: it never waits for a
# vblank or a flip on the CPU, its submissions block once the flips queue up. At the console's
# 60 Hz the game runs at most 60 fps; frames that miss a vblank still queue (no 30 fps lock).
# -Fps120: a 120 Hz vblank, the game's own frames up to 120 fps (Stonefang standing 60 -> 79 fps).
# Play keeps its speed with gameMinDeltaTime (below), but movies advance a frame per flip and play
# faster (start-up to the title 70 -> 54 s). With the Linux configs' 240 Hz the intro played about
# four times too fast.
Set-Option '--vblank-frequency' "$(if ($Vblank -gt 0) { $Vblank } elseif ($Fps120) { 120 } else { 60 })"
if ($Patch) { Set-Option '--game-patch' (Resolve-Path $Patch).Path }
# Windows dispatches exceptions on the faulting thread's stack, over the guest's SysV red
# zone (Linux signal delivery skips it). The write-tracking faults of the performance paths
# then corrupt guest locals; --redzone rewrites the guest's red-zone accesses at load time.
if (!$NoRedZone) { $options.Add('--redzone') }

# Environment switches: the config's (the emulator's own defaults are the release settings).
$environment = [ordered]@{}
foreach ($property in $launch.environment.PSObject.Properties) {
	$environment[$property.Name] = [string]$property.Value
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
# alone, as on Linux, halved the frame rate here. "auto" (the default) picks the performance cores of this PC
# (below); a CPU of one kind of core leaves both to Windows.
if (!$environment.Contains('KYTY_RECORDING_CPUS')) { $environment['KYTY_RECORDING_CPUS'] = 'auto' }
if (!$environment.Contains('KYTY_RENDER_CPUS')) { $environment['KYTY_RENDER_CPUS'] = $environment['KYTY_RECORDING_CPUS'] }
# -RenderHeight: the height Demon's Souls' performance mode renders its 3D at (src/loader/demonsSoulsRenderSize.h);
# the game upscales it to its 3840x2160 output as it does 1440 (a laptop's RTX 3080 at Boletaria 1-1: standing
# 20.7 -> 24.5 fps at 1080, walking 18.7 -> 22.2).
# Without it the height follows the window: its own height, 720 to the game's 1440 (with -Upscaler the window's / 1.5,
# at least 1080). A height given is passed as given, 1440 too, so the window's size does not change a choice.
if ($RenderHeight -gt 0) { $environment['KYTY_RENDER_HEIGHT'] = "$RenderHeight" }
# -HdOutput: the game's 1080p output (src/loader/demonsSoulsHdOutput.h). A 1440 render height (the default) becomes 1080.
if ($HdOutput) { $environment['KYTY_HD_OUTPUT'] = '1' }
# -OutputHeight: the PS5 Pro path's output at that height (KYTY_OUTPUT_HEIGHT, src/loader/demonsSoulsHdOutput.h), with
# -Upscaler only.
if ($OutputHeight -gt 0) { $environment['KYTY_OUTPUT_HEIGHT'] = "$OutputHeight" }
# -Gpu: the graphics card on a PC with more than one, its number in the log's "Vulkan GPU <n>:" list or a part of its
# name (-Gpu 4090); without it the first discrete GPU, in the order Windows' graphics settings give (issue #132). Give
# precompile-windows.ps1 the same -Gpu: the pipelines are compiled for one card.
# (Set here already: the precompile program's runs below, the prefetch inputs, are made for that card too.)
if ($Gpu) { $environment['KYTY_GPU'] = $Gpu; $env:KYTY_GPU = $Gpu }
# -Aspect 21:9 (or 32:9, w:h): the 3D rendered at that aspect, its width at the render height (3416x1440), the game's
# projection widened to match; the presenter shows every frame at the aspect (movies and menus stretched).
if ($Aspect -and $Aspect -ne '16:9') { $environment['KYTY_RENDER_ASPECT'] = $Aspect }
# -Quality low|lowest: Demon's Souls engine settings that cut what a frame draws, for PCs whose CPU limits the frame
# rate (each draw costs the emulator CPU time). low: levels of detail chosen as for a 720-line picture (1080), sun
# shadows to 125 m (250); lowest: detail as for 480 lines, sun shadows to 60 m, no ambient shadowing, no shadows from
# torches and other spot lights, no projected decals (blood, scorch marks: 10-28, RTX 5090 at 1-1 on lowest, standing
# 86.1 -> 96.1 fps, walking 89.2 -> 95.6, the GPU's frame -22%; on a Steam Deck +7% at 360p). A laptop Ryzen 9 5900HX
# + RTX 3080 standing at the Gates of Boletaria, at 1440: standard 21.9 fps, lowest 25.6 (before the decals). Not
# r_forceShadowLOD (until 10-17 in both): it draws objects with their shadow meshes, and objects without one vanish
# (carts and barricades at the Gates of Boletaria; issue #40), for 6% more.
$qualityConvars = @{
	low    = 'r_forceViewportHeightForPixelSizeCalc=720;sunShadowMaxDistance=125'
	lowest = 'r_forceViewportHeightForPixelSizeCalc=480;sunShadowMaxDistance=60;r_enableAmbientShadowing=false;r_enableSpotShadows=false;r_projectedDecalEnable=false'
}
if ($Quality -and $Quality -ne 'standard') {
	if (!$qualityConvars.Contains($Quality)) { throw "-Quality takes low or lowest (or standard), not '$Quality'" }
	if ($titleId -match '^PPSA0134[0-2]$') {
		$environment['KYTY_GAME_CONVARS'] = (@($environment['KYTY_GAME_CONVARS'], $qualityConvars[$Quality]) | Where-Object { $_ }) -join ';'
	}
}
# -Antialiasing off (issue #220): Demon's Souls' TAA off (r_aaMethod 0) and its projection jitter with it, which TAA
# alone left on: a still scene's edges moved from frame to frame (8.5% of its pixels changed, 1% with the jitter off).
# Not with -Upscaler: DLSS and FSR take the game's PS5 Pro path, whose upscaler is the anti-aliasing.
if ($Antialiasing -eq 'off' -and $titleId -match '^PPSA0134[0-2]$') {
	if ($Upscaler) {
		Write-Warning "-Antialiasing off is for the game's own upscaler: left out with -Upscaler $Upscaler"
	} else {
		$environment['KYTY_GAME_CONVARS'] = (@($environment['KYTY_GAME_CONVARS'], 'r_aaMethod=0;r_aaEnableTemporalProjectMatrixJitter=false') |
			Where-Object { $_ }) -join ';'
	}
}
# -TextureReduce 0|1|2: large block-compressed textures held without their 1 or 2 largest levels (KYTY_TEXTURE_REDUCE;
# without it the emulator takes 1 below 12 GB of video memory, else 0). A simulated 8 GB card at Boletaria 1-1: textures
# 4.1 -> 2.8 GiB, 1% low 37-49 -> 53-58 fps; close surfaces a little softer.
if ($TextureReduce -ne '') {
	if ($TextureReduce -notin '0', '1', '2') { throw "-TextureReduce takes 0, 1 or 2, not '$TextureReduce'" }
	$environment['KYTY_TEXTURE_REDUCE'] = $TextureReduce
}
if ($Precompile) {
	# Translate every recorded shader and create every recorded pipeline on all allowed CPUs,
	# save the driver cache, exit. Later launches warm up from that cache in seconds.
	$environment['KYTY_SHADER_WARMUP'] = '1'
	$environment['KYTY_SHADER_WARMUP_ONLY'] = '1'
	$environment['KYTY_SHADER_WARMUP_THREADS'] = "$cpus"
} else {
	# An earlier -Precompile in the same shell leaves this set; the game must not exit after warmup.
	Remove-Item env:KYTY_SHADER_WARMUP_ONLY -ErrorAction SilentlyContinue
}
if ($Threads -gt 0) { $environment['KYTY_SHADER_WARMUP_THREADS'] = "$Threads" }
if ($AspectFit) { $environment['KYTY_PRESENT_ASPECT'] = 'fit' }
# -LowLatency: src/graphics/presentation/videoOut.cpp's FlipQueue (fc1e9cf4).
if ($LowLatency) { $environment['KYTY_LOW_LATENCY'] = '1' }
if ($FrameGen -gt 0) { $environment['KYTY_FRAMEGEN'] = "$FrameGen" }
# -Upscaler dlss|fsr: src/loader/demonsSoulsSuperResolution.h (fsr: src/local/fsr-native.h).
if ($Upscaler) { $environment['KYTY_UPSCALER'] = $Upscaler }
# -Fsr: FSR 3.1 (one generated frame; no Streamline).
if ($Fsr) { $environment['KYTY_FRAMEGEN'] = '1'; $environment['KYTY_FRAMEGEN_BACKEND'] = 'fsr'; $FrameGen = 0 }
# Without -StreamlineDir, the Streamline folder chosen in the launcher (launcher-settings.json), as the launcher passes
# it: run.cmd -FrameGen 1 had started with frame generation off, its runtime not found (issue #38).
if (!$StreamlineDir -and ($FrameGen -gt 0 -or $Upscaler -eq 'dlss') -and (Test-Path "$PSScriptRoot\launcher-settings.json")) {
	$chosen = try { [string](Get-Content "$PSScriptRoot\launcher-settings.json" -Raw | ConvertFrom-Json).streamline } catch { '' }
	if ($chosen -and (Test-Path $chosen)) { $StreamlineDir = $chosen }
}
if ($StreamlineDir) { $environment['KYTY_STREAMLINE_DIR'] = $StreamlineDir }
# -Set KEY=VALUE overrides a switch of the config; KEY= drops it. Several: -Set A=1,B=2 (a comma
# starts a new pair only before KEY=, so values such as 1,2,3,6,7 stay whole).
foreach ($pair in ($Set | ForEach-Object { $_ -split ',(?=[A-Za-z_][A-Za-z0-9_]*=)' })) {
	$key, $value = $pair -split '=', 2
	# (KEY= removes what the config or this script set; for the emulator's own defaults it turns the path off, 0, as it
	# did when they were set here: -Set KYTY_TABLE_XPR=,KYTY_TABLE_DISPATCH= in issue replies.)
	if ($value) { $environment[$key] = $value }
	elseif ($environment.Contains($key)) { $environment.Remove($key); Remove-Item "env:$key" -ErrorAction SilentlyContinue }
	else { $environment[$key] = '0' }
}
# Above 60 Hz (-Fps120, -Vblank; after -Set, whose engine settings get it too): Demon's Souls steps its game by at
# least a frame of its target rate each frame (gameMinDeltaTime below 0, its default: 1/60 s, the cinematic mode's
# 1/30), so frames past that rate played the game that much faster (issue #22: Stonefang 2-3 at 74 fps walked 4.52 m/s,
# 3.66 at 60; the cinematic mode several times over). A 4 ms least step keeps it real time (the same run, the same
# 74 fps: 3.54-3.68 m/s).
$vblankHz = if ($Vblank -gt 0) { $Vblank } elseif ($Fps120) { 120 } else { 60 }
if ($vblankHz -gt 60 -and $titleId -match '^PPSA0134[0-2]$' -and "$($environment['KYTY_GAME_CONVARS'])" -notmatch 'gameMinDeltaTime') {
	$environment['KYTY_GAME_CONVARS'] = (@($environment['KYTY_GAME_CONVARS'], 'gameMinDeltaTime=0.004') | Where-Object { $_ }) -join ';'
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
Write-Host "config:   $Config$(if ($Precompile) { ' (precompile)' })"
Write-Host ("affinity: 0x{0:X} ({1} CPUs){2}" -f $mask, $cpus, $(if ($environment['KYTY_RENDER_CPUS']) { "; render threads on CPUs $($environment['KYTY_RENDER_CPUS'])" }))
Write-Host "switches: $($environment.Count)"
Write-Host "command:  $Exe $($quoted -join ' ')"
if ($DryRun) { $environment.GetEnumerator() | ForEach-Object { "  $($_.Key)=$($_.Value)" }; return }

# Shaders, from the seed file (every shader of the game) by the precompile program: the shader
# prefetch's inputs for this GPU and driver (_PipelineCache\static\<title>_<version>.shaders; new shaders
# are then translated in the background while playing), made when they are missing or stale (the first
# launch, a driver update: half a minute on 22 CPUs); and the static pipeline cache, the whole game
# compiled ahead by precompile-windows.ps1 (about 55 minutes on 22 CPUs), offered by -Prompt.
$tool = Join-Path (Split-Path $Exe) 'kyty_shader_precompile.exe'
# The seed file of this game version (Rename-TitleCaches).
$seedName = if ($param -and $titleId -and $param.contentVersion) { "seeds-$($titleId)_$($param.contentVersion).seeds" } else { 'seeds.seeds' }
$seeds = @("$PSScriptRoot\$seedName", "$PSScriptRoot\_Build\static-precompile\$seedName") | Where-Object { Test-Path $_ } | Select-Object -First 1
# No seed file yet (a release has none: it holds the game's shader code): made from the game's files once by the
# precompile program, as precompile-windows.ps1 makes it (--make-seeds, with the render-pass states of
# pass-states.json; precompile.py seeds without Python).
$passStates = "$PSScriptRoot\tools\local\static-precompile\pass-states.json"
# One another program made (<file>.program: its size and time) is made again (a release unpacked over the last) and
# replaced where it lists something else (precompile-windows.ps1 has the same).
$programStamp = if (Test-Path $tool) { "$((Get-Item $tool).Length) $((Get-Item $tool).LastWriteTimeUtc.Ticks)" }
$staleSeeds = $seeds -and $programStamp -and (!(Test-Path "$seeds.program") -or "$(Get-Content "$seeds.program" -Raw)".Trim() -ne $programStamp)
if ((!$seeds -or $staleSeeds) -and !$Precompile -and (Test-Path $tool) -and (Test-Path $passStates)) {
	$target = if ($seeds) { $seeds } elseif (Test-Path "$PSScriptRoot\_Build") { "$PSScriptRoot\_Build\static-precompile\$seedName" } else { "$PSScriptRoot\$seedName" }
	$made = if ($staleSeeds) { "$target.new" } else { $target }
	New-Item -ItemType Directory -Force (Split-Path $made) | Out-Null
	Write-Host "shaders:  listing the game's shaders from its files $(if ($staleSeeds) { 'again (another precompile program)' } else { '(once)' })"
	& $tool --game $Game --make-seeds $made --states $passStates | Out-Null
	if ($LASTEXITCODE -eq 0 -and (Test-Path $made)) {
		if ($staleSeeds -and (Get-FileHash $made).Hash -eq (Get-FileHash $target).Hash) { Remove-Item $made } elseif ($staleSeeds) { Move-Item -Force $made $target }
		Set-Content "$target.program" $programStamp
		$seeds = $target
	} elseif (!$staleSeeds) {
		Write-Host 'shaders:  listing failed: the game compiles its shaders when they first appear'
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
	# Outdated: made from the seeds of an older precompile program, which listed fewer of the game's shaders.
	$cacheOutdated = $status -match 'static cache outdated'
	# A shard older than its store is left from an earlier run, not an unfinished one: another graphics driver's, which
	# no merge takes (issue #126: six of them made every start offer the precompile again).
	$unfinished = Get-ChildItem "$PSScriptRoot\_PipelineCache\static\*.shard*" -ErrorAction SilentlyContinue | Where-Object {
		$store = Join-Path $_.DirectoryName ($_.Name -replace '\.shard.*$', '')
		$_.Length -gt 1MB -and !((Test-Path -LiteralPath $store) -and (Get-Item -LiteralPath $store).LastWriteTime -gt $_.LastWriteTime)
	}
	$cacheReady = $status -match 'static cache current' -and !$unfinished
	Write-Host "shaders:  prefetch inputs $(if ($inputsReady) { 'ready' } else { 'to be made' }), static pipeline cache $(if ($cacheReady) { 'ready' } elseif ($cacheOutdated) { 'out of date (precompile-windows.ps1 compiles what is new)' } else { 'not made (precompile-windows.ps1)' })"
}

# -Prompt: the game version and the precompile, before anything starts.
$noPrompt = "$PSScriptRoot\no-precompile-prompt.txt"
$offer = $shaders -and !$cacheReady -and !(Test-Path $noPrompt)
if ($Prompt -and ($offer -or !$tested)) {
	$info = "Game: $(if ($titleName) { $titleName } else { 'unknown' }) ($version)"
	$info += if ($tested) { ', a tested version.' } else { "`n⚠ This version is untested (the emulator was tested with $($testedVersions -join ' and ')): it may not run or may fail." }
	if ($offer) {
		$minutes = [math]::Ceiling(55 * 22 / $cpus / 10) * 10
		$time = if ($minutes -lt 90) { "about $minutes minutes" } else { 'about {0:N1} hours' -f ($minutes / 60) }
		# (next precompile all: the precompiled shaders lack a translation that changes nearly every shader, issue #246.)
		$message = if ($cacheOutdated -and $status -match 'next precompile all') {
			"$info`n`nThis version of the emulator translates the game's shaders differently from the one that made the precompiled shaders on this PC (a newer translation), so a precompile now compiles all of them again, once: $time on this PC, with the CPU fully loaded (closing its window stops it, and the next run continues).`n" +
				"Until then the current precompiled shaders stay in use: starting the game now is fine."
		} elseif ($cacheOutdated) {
			"$info`n`nThis version of the emulator precompiles more of the game's shaders than the one that made the precompiled shaders on this PC: the new ones would stutter the first time they appear.`n" +
				"Precompiling again compiles only what is new (minutes; at most $time on this PC, with the CPU fully loaded; closing its window stops it, and the next run continues)."
		} else {
			"$info`n`nThe shaders are not precompiled for this graphics card yet: scenes and effects stutter the first time they appear in the game (a fraction of a second to a few seconds).`n" +
				"The precompile compiles every shader of the game once: $time on this PC (with the CPU fully loaded; closing its window stops it, and the next run continues). " +
				"It is needed only once, and again after a graphics driver update."
		}
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
# video memory in use, the emulator's own). Under 24 GB of RAM it compiles no shaders or pipelines at
# start-up and keeps its pipeline caches small: about 14 GB in use after a save loads at 1-1 (most of it
# the game's own memory), of which play keeps using only a few GB (10-09: emptied after loading, the
# working set grew back to 3.5 GB in a minute of play at 60 fps): with 16 GB Windows pages the rest out.
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
	$memory += ("This PC has {0:N0} GB of RAM: the game runs, but loading takes longer (the emulator takes about 14 GB " +
		"while a save loads and keeps using a few GB in play; 32 GB recommended).") -f ($ram / 1GB)
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
