param([switch]$DryRun)
# A small settings window in front of run-windows.ps1 and precompile-windows.ps1 (launcher.cmd opens it). The
# settings and the console language names follow the KytyPS5 launcher (src/launcher, by the KytyPS5 developers);
# they are kept in launcher-settings.json next to this script. -DryRun prints the commands without a window.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
# cmd.exe by its full path: Start-Process with a bare "cmd" opens whatever "cmd" it finds first, and a file of that
# name without an extension (in the working folder or on the PATH) made Windows ask which app should open it (issue #47).
$cmd = if ($env:ComSpec -and (Test-Path -LiteralPath $env:ComSpec)) { $env:ComSpec } else { Join-Path ([Environment]::SystemDirectory) 'cmd.exe' }
# Windows PowerShell by its path too: on a PC whose PATH lacks its folder, cmd.exe did not know "powershell" (issue #55).
$powershellExe = Join-Path ([Environment]::SystemDirectory) 'WindowsPowerShell\v1.0\powershell.exe'
$powershellCommand = if (Test-Path -LiteralPath $powershellExe) { "`"$powershellExe`"" } else { 'powershell' }
$settingsPath = Join-Path $root 'launcher-settings.json'
$languages = 'Japanese', 'English (United States)', 'French (France)', 'Spanish (Spain)', 'German', 'Italian', 'Dutch',
	'Portuguese (Portugal)', 'Russian', 'Korean', 'Chinese (Traditional)', 'Chinese (Simplified)', 'Finnish', 'Swedish',
	'Danish', 'Norwegian', 'Polish', 'Portuguese (Brazil)', 'English (United Kingdom)', 'Turkish', 'Spanish (Latin America)',
	'Arabic', 'French (Canada)', 'Czech', 'Hungarian', 'Greek', 'Romanian', 'Thai', 'Vietnamese', 'Indonesian'
# The window's sizes for each aspect ratio of the 3D (below).
$resolutionsByRatio = @{
	'16:9' = '1280x720', '1920x1080', '2560x1440', '3840x2160'
	'21:9' = '2560x1080', '3440x1440', '3840x1600', '5120x2160'
	'32:9' = '3840x1080', '5120x1440', '7680x2160' }
# The height Demon's Souls renders its 3D at (run-windows.ps1 -RenderHeight); its output and UI stay 3840x2160.
$renderHeights = '2160 (native 4K)', '1800', '1620', '1512 (PS5 Pro)', '1440 (PS5)', '1296', '1152', '1080', '900'
# What makes the 3840x2160 image of the 3D (run-windows.ps1 -Upscaler): the game's TAA and stretch (the PS5's), or DLSS in
# place of Sony's PS5 Pro upscaler on the game's PS5 Pro path (NVIDIA's Streamline runtime, as frame generation).
$upscalers = 'Game (PS5)', 'DLSS (PS5 Pro path)', 'XeSS (PS5 Pro path)', 'FSR 3.1 (PS5 Pro path)'
$upscalerKinds = '', 'dlss', 'xess', 'fsr'
# The aspect ratio of Demon's Souls' 3D (run-windows.ps1 -Aspect): wider sees more to the sides.
$ratios = '16:9 (PS5)', '21:9', '32:9'
# Demon's Souls' own settings for what a frame draws (run-windows.ps1 -Quality): shorter shadows, simpler distant detail;
# and smaller textures (run-windows.ps1 -TextureReduce: Low 1, Lowest 2; Standard leaves it to the emulator: 1 below
# 12 GB of video memory, else 0).
$qualities = 'Standard (PS5)', 'Low', 'Lowest'
# DLSS frame generation (run-windows.ps1 -FrameGen: the generated frames per rendered one, the multiplier less one).
$frameGens = 'Off', '2x', '3x', '4x', '5x', '6x'
# The emulator's own panel at the top right of the game image (KYTY_FPS_HUD 0, 1, 2).
$overlays = 'Off', 'Frame rate', 'Detailed'
# Safe GPU mode (KYTY_COMPUTE_CHAIN=0): Demon's Souls' compute work runs one dispatch after another instead of letting
# the GPU overlap them. Slower, but it stops the GPU hangs ("device lost", the driver resetting the display) that some
# PCs have: on an RTX 3080 Laptop with driver 566.07, 3 of 4 runs through 1-1 hung with overlap, none of 3 without it,
# at 25-48% less frame rate there (13.9-17.7 fps instead of 26.8 at 1-1, 21-22 instead of 28.8 at 2-2).
$safeGpuTip = "Prevents the crashes and blue screens that come from the graphics card hanging: 'device lost', the screen going`n" +
	"black while the graphics driver restarts (a TDR), or a VIDEO_TDR_FAILURE blue screen. The emulator then runs Demon's Souls'`n" +
	"compute work one step after another instead of letting the graphics card overlap the steps. The frame rate is about 37%`n" +
	"lower (on a laptop RTX 3080 at Boletaria: 17 fps instead of 27). It does not help MEMORY_MANAGEMENT blue screens, which`n" +
	"come from the RAM being full (see the README). Leave it off if the game runs without such crashes; a current graphics`n" +
	"driver can help too."

$settings = [ordered]@{ game = ''; resolution = '2560x1440'; fullscreen = $false; aspect = $true; language = 1; redzone = $true;
	ecores = $false; fps120 = $false; render = 1440; ratio = '16:9'; quality = 'standard'; vsync = $false; hdr = $false; framegen = 0
	hud = 0; streamline = ''; safegpu = $false; upscaler = ''; render_dlss = 1080; turbo = $false; gentle = $false; amdfg = $false }
if (Test-Path $settingsPath) {
	$saved = Get-Content $settingsPath -Raw | ConvertFrom-Json
	foreach ($property in $saved.PSObject.Properties) { if ($settings.Contains($property.Name)) { $settings[$property.Name] = $property.Value } }
}
if (!$settings.game -and (Test-Path "$root\game-path.txt")) { $settings.game = (Get-Content "$root\game-path.txt" -TotalCount 1).Trim() }

# The efficiency cores (the lowest efficiency class of a hybrid CPU) as an affinity mask, or 0.
function Get-EfficiencyMask {
	Add-Type -Namespace KytyLauncher -Name CpuSets -MemberDefinition @'
[DllImport("kernel32.dll")]
static extern bool GetSystemCpuSetInformation(IntPtr information, uint length, out uint returned, IntPtr process, uint flags);
// Group 0's logical processors as (index, efficiency class) pairs.
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
			result.Add(Marshal.ReadByte(buffer, offset + 18));
		}
	} finally {
		Marshal.FreeHGlobal(buffer);
	}
	return result.ToArray();
}
'@
	$values = [KytyLauncher.CpuSets]::Query()
	$sets = for ($i = 0; $i + 1 -lt $values.Count; $i += 2) { [pscustomobject]@{ Cpu = $values[$i]; Class = $values[$i + 1] } }
	if (@($sets | ForEach-Object Class | Sort-Object -Unique).Count -lt 2) { return [int64]0 }
	$slowest = ($sets | Measure-Object Class -Minimum).Minimum
	$mask = [int64]0
	foreach ($set in $sets | Where-Object { $_.Class -eq $slowest -and $_.Cpu -lt 63 }) { $mask = $mask -bor ([int64]1 -shl $set.Cpu) }
	return $mask
}
$efficiencyMask = Get-EfficiencyMask
# Frame generation needs NVIDIA's Streamline runtime, which no release ships: the Streamline SDK
# (github.com/NVIDIA-RTX/Streamline/releases) chosen with "Streamline..." (settings: streamline), else a "streamline"
# folder next to the emulator, else the SDK beside a local build (_Build\deps\streamline\sdk). The emulator loads
# only NVIDIA's own sl.interposer.dll (run-windows -StreamlineDir).
$exe = if (Test-Path "$root\kyty_emulator.exe") { "$root\kyty_emulator.exe" } else { "$root\_Build\windows\kyty_emulator.exe" }
$streamlineSdkUrl = 'https://github.com/NVIDIA-RTX/Streamline/releases'
# The runtime's folder in $dir (the folder itself or its bin\x64) when its sl.interposer.dll is signed by NVIDIA;
# $null otherwise.
function Find-Streamline([string]$dir) {
	if (!$dir) { return $null }
	foreach ($folder in @($dir, (Join-Path $dir 'bin\x64'))) {
		$file = Join-Path $folder 'sl.interposer.dll'
		if (Test-Path -LiteralPath $file) {
			$signature = Get-AuthenticodeSignature -LiteralPath $file
			if ($signature.Status -eq 'Valid' -and $signature.SignerCertificate.Subject -match 'O=NVIDIA Corporation') { return $folder }
			return $null
		}
	}
	return $null
}
function Resolve-Streamline {
	if ($settings.streamline) { return Find-Streamline $settings.streamline }
	if ($env:KYTY_STREAMLINE_DIR) { return Find-Streamline $env:KYTY_STREAMLINE_DIR }
	foreach ($dir in @((Join-Path (Split-Path $exe) 'streamline'), (Join-Path (Split-Path (Split-Path $exe)) 'deps\streamline\sdk'))) {
		$found = Find-Streamline $dir
		if ($found) { return $found }
	}
	return $null
}
$streamline = Resolve-Streamline
# DLSS (the upscaler and frame generation) needs an NVIDIA RTX graphics card: with none (AMD, Intel, GeForce GTX) both
# rows stay off and the game's own TAA makes the picture, at the render height the game's TAA keeps.
$rtxGpu = try {
	[bool](Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' -and $_.Name -match '\bRTX\b' })
} catch { $false }
$frameGenAvailable = [bool]$streamline -and $rtxGpu

function Get-PlayCommand {
	$size = $settings.resolution -split 'x'
	$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$root\run-windows.ps1`"", '-Prompt', '-Follow',
		'-Width', $size[0], '-Height', $size[1], '-Language', $settings.language)
	if ($settings.fullscreen) { $arguments += '-Fullscreen'; if ($settings.aspect) { $arguments += '-AspectFit' } }
	if (!$settings.redzone) { $arguments += '-NoRedZone' }
	if ($settings.fps120) { $arguments += '-Fps120' }
	# Presents wait for the display's refresh (Vulkan FIFO): no tearing, and G-SYNC/FreeSync displays keep adapting.
	if ($settings.vsync) { $arguments += @('-PresentMode', 'Fifo') }
	# HDR10 output when Windows HDR is on for the display; SDR otherwise.
	if ($settings.hdr) { $arguments += '-Hdr' }
	$dlss = $frameGenAvailable -and $settings.upscaler -eq 'dlss'
	# The PS5 Pro path with DLSS, XeSS or FSR 3.1 (XeSS and FSR: any graphics card).
	$sr = $dlss -or $settings.upscaler -in 'xess', 'fsr'
	# The upscaler's own render height (with the PS5 Pro path always passed: its own height is 1512).
	$height = if ($sr) { [int]$settings.render_dlss } else { [int]$settings.render }
	if ($height -gt 0 -and ($height -ne 1440 -or $sr)) { $arguments += @('-RenderHeight', $height) }
	if ($settings.ratio -in '21:9', '32:9') { $arguments += @('-Aspect', $settings.ratio) }
	if ($settings.quality -in 'low', 'lowest') {
		$arguments += @('-Quality', $settings.quality, '-TextureReduce', $(if ($settings.quality -eq 'lowest') { 2 } else { 1 }))
	}
	if ($frameGenAvailable -and [int]$settings.framegen -ge 2) { $arguments += @('-FrameGen', ([int]$settings.framegen - 1)) }
	if ($dlss) { $arguments += @('-Upscaler', 'dlss') } elseif ($sr) { $arguments += @('-Upscaler', $settings.upscaler) }
	# AMD FSR 3 frame generation (any graphics card; with DLSS-G off).
	if ($settings.amdfg -and !($frameGenAvailable -and [int]$settings.framegen -ge 2)) { $arguments += '-Fsr3FrameGen' }
	# Turbo (16 GB PCs: about 6 GB of video memory) and the gentle precompile (half the CPU).
	if ($settings.turbo) { $arguments += '-Turbo' }
	if ($settings.gentle) { $arguments += '-GentlePrecompile' }
	if ($frameGenAvailable -and ([int]$settings.framegen -ge 2 -or $dlss)) { $arguments += @('-StreamlineDir', "`"$streamline`"") }
	# Switches of the config, one -Set argument (run-windows.ps1 splits it at the commas).
	$sets = @()
	# The emulator's panel (KYTY_FPS_HUD) at the top right of the game image: the frame rate and frame times, or with
	# the GPU, processor and memory too.
	if ([int]$settings.hud -ge 1) { $sets += "KYTY_FPS_HUD=$([Math]::Min([int]$settings.hud, 2))" }
	# Safe GPU mode: no overlapping compute dispatches ($safeGpuTip).
	if ($settings.safegpu) { $sets += 'KYTY_COMPUTE_CHAIN=0' }
	if ($sets.Count) { $arguments += @('-Set', ($sets -join ',')) }
	if ($settings.game) { $arguments += @('-Game', "`"$($settings.game)`"") }
	# The console stays for the live log; after a crash it waits for a key.
	return "$powershellCommand " + ($arguments -join ' ') + ' & if !errorlevel! neq 0 pause'
}
function Get-PrecompileCommand([string]$switch = '') {
	$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$root\precompile-windows.ps1`"")
	if ($switch) { $arguments += $switch }
	if ($settings.gentle) { $arguments += '-Gentle' }
	if ($settings.ecores -and $efficiencyMask -ne 0) { $arguments += @('-Affinity', $efficiencyMask) }
	if ($settings.game) { $arguments += @('-Game', "`"$($settings.game)`"") }
	return "$powershellCommand " + ($arguments -join ' ') + ' & pause'
}
function Save-Settings { $settings | ConvertTo-Json | Set-Content $settingsPath -Encoding UTF8 }
if ($DryRun) {
	"play:       cmd /v:on /s /c `"$(Get-PlayCommand)`""
	"precompile: cmd /s /c `"$(Get-PrecompileCommand)`""
	"hints:      cmd /s /c `"$(Get-PrecompileCommand '-ExportHints')`""
	"efficiency cores: 0x{0:X}" -f $efficiencyMask
	"frame generation: $(if ($frameGenAvailable) { "NVIDIA's Streamline runtime in $streamline" } elseif (!$rtxGpu) { 'no NVIDIA RTX graphics card (DLSS off)' } else { "no NVIDIA-signed Streamline runtime (settings: '$($settings.streamline)')" })"
	return
}

Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type -Namespace KytyLauncher -Name Dpi -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
[KytyLauncher.Dpi]::SetProcessDPIAware() | Out-Null
[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form -Property @{
	Text = 'KytyPS5 launcher'; FormBorderStyle = 'FixedDialog'; MaximizeBox = $false; StartPosition = 'CenterScreen'
	AutoSize = $true; AutoSizeMode = 'GrowAndShrink'; Font = New-Object System.Drawing.Font('Segoe UI', 9); Padding = '12,12,12,8'
}
$grid = New-Object System.Windows.Forms.TableLayoutPanel -Property @{ ColumnCount = 2; AutoSize = $true; Dock = 'Fill' }
$form.Controls.Add($grid)
# A row: a label, then the controls side by side.
function Add-Row($label, [object[]]$controls) {
	$grid.Controls.Add((New-Object System.Windows.Forms.Label -Property @{ Text = $label; AutoSize = $true; Anchor = 'Left'; Margin = '0,6,8,6' }))
	$row = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ AutoSize = $true; WrapContents = $false; Margin = '0,0,0,0' }
	$row.Controls.AddRange($controls)
	$grid.Controls.Add($row)
}

$game = New-Object System.Windows.Forms.TextBox -Property @{ Text = $settings.game; Width = 440 }
$browse = New-Object System.Windows.Forms.Button -Property @{ Text = 'Browse...'; AutoSize = $true }
$browseZar = New-Object System.Windows.Forms.Button -Property @{ Text = '.zar...'; AutoSize = $true }
Add-Row 'Game folder' @($game, $browse, $browseZar)
$title = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray' }
Add-Row '' @($title)
# A game: its folder (eboot.bin and sce_sys) or the folder packed into a ZArchive (a .zar file).
function Read-GameParam([string]$game) {
	if (!$game) { return $null }
	if (Test-Path -LiteralPath "$game\sce_sys\param.json") { return Get-Content -LiteralPath "$game\sce_sys\param.json" -Raw -Encoding UTF8 | ConvertFrom-Json }
	$tool = @("$root\kyty_shader_precompile.exe", "$root\_Build\windows\kyty_shader_precompile.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
	if ($game -notmatch '\.zar$' -or !(Test-Path -LiteralPath $game -PathType Leaf) -or !$tool) { return $null }
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
function Update-Title {
	$title.Text = 'Choose the folder with eboot.bin and sce_sys (the game dump), or a .zar archive of it.'
	try {
		$json = Read-GameParam $game.Text.Trim()
		if ($json) {
			$title.Text = '{0} ({1} {2})' -f $json.localizedParameters.($json.localizedParameters.defaultLanguage).titleName,
				$json.titleId, $json.contentVersion
		}
	} catch {}
}
$browse.Add_Click({
	$dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{ Description = 'The game folder (eboot.bin, sce_sys)' }
	if ($game.Text -and (Test-Path -LiteralPath $game.Text -PathType Container)) { $dialog.SelectedPath = $game.Text }
	if ($dialog.ShowDialog($form) -eq 'OK') { $game.Text = $dialog.SelectedPath }
})
# The game folder packed into a ZArchive (zarchive.exe <folder> <file.zar>): read without extracting it.
$browseZar.Add_Click({
	$dialog = New-Object System.Windows.Forms.OpenFileDialog -Property @{ Title = 'The game folder packed into a ZArchive'; Filter = 'ZArchive (*.zar)|*.zar' }
	if ($game.Text -and (Test-Path -LiteralPath $game.Text -PathType Leaf)) { $dialog.InitialDirectory = Split-Path $game.Text }
	if ($dialog.ShowDialog($form) -eq 'OK') { $game.Text = $dialog.FileName }
})
$game.Add_TextChanged({ Update-Title })
Update-Title

# The picture's size, shape and detail, side by side: the aspect ratio offers the resolutions of its shape (a 21:9
# picture in a 3840x2160 window was stretched back to 16:9) and the render heights whose 3D the game holds (no larger
# than its own 3840x2160 in GPU tiles: 21:9 at 2160 ended it at start-up, 1864 drew it red; src/loader/demonsSoulsRenderSize.h).
$ratio = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 205 }
$ratio.Items.AddRange($ratios)
$ratio.SelectedItem = ($ratios | Where-Object { ($_ -split ' ')[0] -eq [string]$settings.ratio } | Select-Object -First 1)
if ($ratio.SelectedIndex -lt 0) { $ratio.SelectedIndex = 0 }
$ratioNote = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '6,6,0,0'
	Text = '21:9 and 32:9 show more to the sides; the HUD keeps its shape, menus and movies are stretched' }
function Get-Ratio { ([string]$ratio.SelectedItem -split ' ')[0] }
function Get-RatioAspect { switch (Get-Ratio) { '21:9' { 64.0 / 27.0 } '32:9' { 32.0 / 9.0 } default { 16.0 / 9.0 } } }
$resolution = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 205 }
# The ratio's resolutions; `wanted` kept when it is one of them, else the one nearest its height.
function Update-Resolutions([string]$wanted) {
	$list = $resolutionsByRatio[(Get-Ratio)]
	$resolution.Items.Clear()
	$resolution.Items.AddRange($list)
	$height = [int](($wanted -split 'x')[1])
	$resolution.SelectedItem = if ($list -contains $wanted) { $wanted } else {
		$list | Sort-Object { [Math]::Abs([int](($_ -split 'x')[1]) - $height) } | Select-Object -First 1
	}
}
Update-Resolutions $settings.resolution
$render = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 205 }
# The highest render height whose 3D at the ratio fits the game's render target memory, made for 3840x2160: no more of
# the GPU's 64 KiB tiles (256x128, 128x128, 128x64, 64x64 pixels) than 3840x2160 (the emulator's FitsRenderTargets).
function Get-MaxRenderHeight {
	$aspectValue = Get-RatioAspect
	for ($h = 2160; $h -ge 720; $h -= 8) {
		$w = [Math]::Floor($h * $aspectValue / 8.0 + 0.5) * 8
		$fits = $true
		foreach ($tile in @(@(256, 128), @(128, 128), @(128, 64), @(64, 64))) {
			$tiles = [Math]::Ceiling($w / $tile[0]) * [Math]::Ceiling($h / $tile[1])
			if ($tiles -gt [Math]::Ceiling(3840 / $tile[0]) * [Math]::Ceiling(2160 / $tile[1])) { $fits = $false }
		}
		if ($fits) { return $h }
	}
	720
}
# Each upscaler keeps its own render height (settings render, render_dlss): the row shows the height last chosen for the
# chosen upscaler ($renderWanted) among those the ratio allows; one above them shows as the highest allowed (and comes
# back with a ratio that allows it), one not in the list as the upscaler's default.
function Get-RenderFallback { if ($script:renderForDlss) { 1080 } else { 1440 } }
function Select-RenderHeight {
	$most = Get-MaxRenderHeight
	$allowed = @($renderHeights | Where-Object { [int](($_ -split ' ')[0]) -le $most })
	$script:fillingRender = $true
	try {
		$render.Items.Clear()
		$render.Items.AddRange($allowed)
		if ($script:renderWanted -gt $most) { $render.SelectedIndex = 0; return }
		foreach ($wanted in @($script:renderWanted, (Get-RenderFallback))) {
			$render.SelectedItem = ($allowed | Where-Object { ($_ -split ' ')[0] -eq [string]$wanted } | Select-Object -First 1)
			if ($render.SelectedIndex -ge 0) { return }
		}
		$render.SelectedIndex = 0
	} finally {
		$script:fillingRender = $false
	}
}
$renderNote = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '6,6,0,0' }
function Update-RenderNote {
	$h = [int](([string]$render.SelectedItem -split ' ')[0])
	$w = [Math]::Floor($h * (Get-RatioAspect) / 8.0 + 0.5) * 8
	$renderNote.Text = "Demon's Souls' 3D at ${w}x${h}; higher = sharper, needs more GPU"
}
$render.Add_SelectedIndexChanged({
	if (!$script:fillingRender) { $script:renderWanted = [int](([string]$render.SelectedItem -split ' ')[0]) }
	Update-RenderNote
})
$renderForDlss = ($frameGenAvailable -and $settings.upscaler -eq 'dlss') -or $settings.upscaler -in 'xess', 'fsr'
$renderWanted = [int]$(if ($renderForDlss) { $settings.render_dlss } else { $settings.render })
Select-RenderHeight
Update-RenderNote
$ratio.Add_SelectedIndexChanged({
	Update-Resolutions ([string]$resolution.SelectedItem)
	Select-RenderHeight
	Update-RenderNote
})
$upscaler = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 205 }
$upscaler.Items.AddRange($upscalers)
$upscaler.SelectedIndex = if ($settings.upscaler -in 'xess', 'fsr') { [array]::IndexOf($upscalerKinds, [string]$settings.upscaler) } elseif ($frameGenAvailable -and $settings.upscaler -eq 'dlss') { 1 } else { 0 }
$upscalerNote = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '6,6,0,0' }
function Update-UpscalerNote {
	$upscalerNote.Text = if ($frameGenAvailable) { 'DLSS: NVIDIA RTX; XeSS, FSR: any GPU; 1440 = Quality, 1080 = Performance' }
		elseif (!$rtxGpu) { 'XeSS and FSR: any graphics card; DLSS needs an NVIDIA RTX' }
		else { 'XeSS, FSR: any GPU; DLSS needs NVIDIA''s Streamline SDK (folder under Frame generation)' }
}
Update-UpscalerNote
(New-Object System.Windows.Forms.ToolTip -Property @{ AutoPopDelay = 30000 }).SetToolTip($upscaler,
	"Game: the PS5's performance mode, its 3D anti-aliased by the game's TAA at the render height and stretched to 4K.`n" +
	"DLSS: the game's own PS5 Pro mode, Sony's upscaler (PSSR) replaced by NVIDIA DLSS: the 3D at the render height (2688x1512`n" +
	"at 1512, the PS5 Pro's; 1080 at least) made a 3840x2160 image the game finishes at 4K (bloom, depth of field, the HUD).`n" +
	"Sharper than the stretch, at some GPU cost. The in-game Cinematic mode stays native 4K without DLSS.`n" +
	"Each upscaler keeps its own render height. DLSS starts at 1080: on an RTX 5090 about the frame rate of the game's TAA`n" +
	"at 1440, with a sharper 4K image.")
$upscaler.Add_SelectedIndexChanged({
	# The height chosen goes back to the upscaler it was for; the newly chosen one's is shown.
	if ($script:renderForDlss) { $settings.render_dlss = $script:renderWanted } else { $settings.render = $script:renderWanted }
	$script:renderForDlss = $upscaler.SelectedIndex -ge 1
	$script:renderWanted = [int]$(if ($script:renderForDlss) { $settings.render_dlss } else { $settings.render })
	Select-RenderHeight
	Update-RenderNote
})
Add-Row 'Resolution' @($resolution)
Add-Row 'Aspect ratio' @($ratio, $ratioNote)
Add-Row 'Render height' @($render, $renderNote)
Add-Row 'Upscaler' @($upscaler, $upscalerNote)
$fullscreen = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Fullscreen'; AutoSize = $true; Checked = [bool]$settings.fullscreen }
$aspect = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Keep the aspect ratio (black bars)'; AutoSize = $true; Checked = [bool]$settings.aspect }
$aspect.Enabled = $fullscreen.Checked
$fullscreen.Add_CheckedChanged({ $aspect.Enabled = $fullscreen.Checked })
Add-Row '' @($fullscreen, $aspect)
$language = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 320 }
$language.Items.AddRange($languages)
$language.SelectedIndex = [Math]::Max(0, [Math]::Min($languages.Count - 1, [int]$settings.language))
Add-Row 'Console language' @($language)
$redzone = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Red-zone protection (recommended)'; AutoSize = $true; Checked = [bool]$settings.redzone }
Add-Row '' @($redzone)
$fps120 = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Up to 120 fps (the game''s own frames; movies stay at 60)'; AutoSize = $true; Checked = [bool]$settings.fps120 }
Add-Row '' @($fps120)
$vsync = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'VSync (no screen tearing; adaptive sync displays keep it)'; AutoSize = $true; Checked = [bool]$settings.vsync }
Add-Row '' @($vsync)
$hdr = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'HDR (turn on "Use HDR" in Windows display settings first)'; AutoSize = $true; Checked = [bool]$settings.hdr }
Add-Row '' @($hdr)
$safeGpu = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Safe GPU mode: prevents GPU-hang crashes and blue screens, but the frame rate is about 37% lower'; AutoSize = $true; Checked = [bool]$settings.safegpu }
(New-Object System.Windows.Forms.ToolTip -Property @{ AutoPopDelay = 30000 }).SetToolTip($safeGpu, $safeGpuTip)
Add-Row '' @($safeGpu)
$turbo = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'TURBO for PCs with 16 GB of RAM or less (about 6 GB of video memory, smaller textures, less detail)'; AutoSize = $true; Checked = [bool]$settings.turbo }
(New-Object System.Windows.Forms.ToolTip -Property @{ AutoPopDelay = 30000 }).SetToolTip($turbo,
	"For PCs that crash or restart in heavy zones (Boletaria, the Nexus) with 16 GB of RAM or 8 GB of video memory or less:`n" +
	"the emulator keeps its video memory near 6 GB, textures at half size, the game's low detail, and empties RAM every 15 s.")
Add-Row '' @($turbo)
$gentle = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Gentle shader precompile (half the CPU: slower, for PCs that restart while precompiling)'; AutoSize = $true; Checked = [bool]$settings.gentle }
Add-Row '' @($gentle)
$amdFg = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'AMD frame generation x2 (FSR 3, any graphics card; works with DLSS, XeSS or FSR above)'; AutoSize = $true; Checked = [bool]$settings.amdfg }
Add-Row '' @($amdFg)
$quality = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 205 }
$quality.Items.AddRange($qualities)
$quality.SelectedItem = ($qualities | Where-Object { ($_ -split ' ')[0] -eq [string]$settings.quality } | Select-Object -First 1)
if ($quality.SelectedIndex -lt 0) { $quality.SelectedIndex = 0 }
$qualityNote = New-Object System.Windows.Forms.Label -Property @{ Text = 'shorter shadows, simpler detail, smaller textures; faster on weaker PCs'; AutoSize = $true; ForeColor = 'Gray'; Margin = '6,6,0,0' }
$qualityTips = New-Object System.Windows.Forms.ToolTip
$qualityTips.SetToolTip($quality, "Standard: the PS5's picture. With less than 12 GB of video memory, large textures leave out their largest level`n(half the size per side) so that they fit.`nLow: shorter sun shadows, simpler distant detail, textures half the size per side.`nLowest: shorter still, no ambient shadowing or torch shadows, textures a quarter the size per side.")
Add-Row 'Graphics' @($quality, $qualityNote)
$frameGen = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 160; Enabled = $frameGenAvailable }
$frameGen.Items.AddRange($frameGens)
$frameGen.SelectedItem = if ($frameGenAvailable -and [int]$settings.framegen -ge 2) { "$([int]$settings.framegen)x" } else { 'Off' }
if ($frameGen.SelectedIndex -lt 0) { $frameGen.SelectedIndex = 0 }
$frameGenFolder = New-Object System.Windows.Forms.Button -Property @{ Text = 'Streamline...'; AutoSize = $true }
$frameGenNote = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '6,6,0,0' }
function Update-FrameGenNote {
	$frameGenNote.Text = if ($frameGenAvailable) { 'DLSS; NVIDIA RTX 40 or newer (3x and up: RTX 50)' }
		elseif (!$rtxGpu) { 'DLSS: needs an NVIDIA RTX 40 or newer graphics card' }
		else { 'needs NVIDIA''s Streamline SDK: download it, then choose its folder' }
}
Update-FrameGenNote
$frameGenTips = New-Object System.Windows.Forms.ToolTip
$frameGenTips.SetToolTip($frameGen, "Shown frames = the game's frames x the multiplier. With VSync on, they are capped at the display's refresh`n(a 60 Hz display at 2x runs the game at 30 fps). The game's frames stay at 60 with 'Up to 120 fps' too:`nframe generation needs them evenly spaced.")
$frameGenTips.SetToolTip($frameGenFolder, "The NVIDIA Streamline SDK (a release zip from $streamlineSdkUrl), unpacked: its folder or its bin\x64.`nOnly NVIDIA-signed files are used; nothing of NVIDIA's comes with the emulator.")
$frameGenFolder.Add_Click({
	$dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{
		Description = "The unpacked NVIDIA Streamline SDK ($streamlineSdkUrl), or its bin\x64 folder" }
	if ($settings.streamline -and (Test-Path -LiteralPath $settings.streamline)) { $dialog.SelectedPath = $settings.streamline }
	if ($dialog.ShowDialog($form) -ne 'OK') { return }
	$found = Find-Streamline $dialog.SelectedPath
	if (!$found) {
		[void][System.Windows.Forms.MessageBox]::Show($form, "No sl.interposer.dll signed by NVIDIA in that folder or its bin\x64.`nDownload the Streamline SDK from $streamlineSdkUrl and choose the unpacked folder.", 'Frame generation')
		return
	}
	$settings.streamline = $dialog.SelectedPath
	$script:streamline = $found
	# (Without an NVIDIA RTX graphics card the folder is kept, the rows stay off.)
	$script:frameGenAvailable = $rtxGpu
	$frameGen.Enabled = $rtxGpu
	Update-FrameGenNote
	Update-UpscalerNote
})
Add-Row 'Frame generation' @($frameGen, $frameGenFolder, $frameGenNote)
$hud = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 160 }
$hud.Items.AddRange($overlays)
# (An older launcher saved true for the frame rate.)
$hud.SelectedIndex = [Math]::Max(0, [Math]::Min($overlays.Count - 1, [int]$settings.hud))
$hudNote = New-Object System.Windows.Forms.Label -Property @{ Text = 'top right: frame rate and frame times; detailed: GPU, CPU, memory. F8 in the game switches it'; AutoSize = $true; ForeColor = 'Gray'; Margin = '6,6,0,0' }
Add-Row 'Overlay' @($hud, $hudNote)
$ecores = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Precompile on the efficiency cores only (slower, the PC stays responsive)'; AutoSize = $true
	Checked = ([bool]$settings.ecores -and $efficiencyMask -ne 0); Enabled = ($efficiencyMask -ne 0) }
Add-Row '' @($ecores)

$buttons = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ AutoSize = $true; Margin = '0,10,0,0' }
$play = New-Object System.Windows.Forms.Button -Property @{ Text = 'Play'; AutoSize = $true; Font = New-Object System.Drawing.Font('Segoe UI', 9, [System.Drawing.FontStyle]::Bold) }
$precompile = New-Object System.Windows.Forms.Button -Property @{ Text = 'Precompile shaders'; AutoSize = $true }
$share = New-Object System.Windows.Forms.Button -Property @{ Text = 'Export shader hints'; AutoSize = $true }
(New-Object System.Windows.Forms.ToolTip).SetToolTip($share, ('The shaders your play recorded, without the game''s code, as my-hints-<game>.hints: ' +
	'attached to a GitHub issue, the next packages precompile them for everyone.'))
$logs = New-Object System.Windows.Forms.Button -Property @{ Text = 'Logs folder'; AutoSize = $true }
$readme = New-Object System.Windows.Forms.Button -Property @{ Text = 'README'; AutoSize = $true }
$buttons.Controls.AddRange(@($play, $precompile, $share, $logs, $readme))
$grid.Controls.Add($buttons)
$grid.SetColumnSpan($buttons, 2)
$credit = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '0,8,0,0'
	Text = 'Settings follow the KytyPS5 launcher (src/launcher) by the KytyPS5 developers.' }
$grid.Controls.Add($credit)
$grid.SetColumnSpan($credit, 2)
$form.AcceptButton = $play

function Read-Form {
	$settings.game       = $game.Text.Trim()
	$settings.resolution = [string]$resolution.SelectedItem
	$settings.fullscreen = $fullscreen.Checked
	$settings.aspect     = $aspect.Checked
	$settings.language   = $language.SelectedIndex
	$settings.redzone    = $redzone.Checked
	$settings.fps120     = $fps120.Checked
	$settings.vsync      = $vsync.Checked
	$settings.hdr        = $hdr.Checked
	$settings.safegpu    = $safeGpu.Checked
	$settings.hud        = $hud.SelectedIndex
	$height              = [int](([string]$render.SelectedItem -split ' ')[0])
	if ($renderForDlss) { $settings.render_dlss = $height } else { $settings.render = $height }
	$settings.ratio      = ([string]$ratio.SelectedItem -split ' ')[0]
	$settings.quality    = ([string]$quality.SelectedItem -split ' ')[0].ToLowerInvariant()
	# (Kept as it was where the DLLs are missing.)
	if ($frameGenAvailable) { $settings.framegen = if ($frameGen.SelectedIndex -gt 0) { $frameGen.SelectedIndex + 1 } else { 0 } }
	$kind = $upscalerKinds[[Math]::Max(0, $upscaler.SelectedIndex)]
	if ($kind -ne 'dlss' -or $frameGenAvailable) { $settings.upscaler = $kind }
	$settings.turbo      = $turbo.Checked
	$settings.gentle     = $gentle.Checked
	$settings.amdfg      = $amdFg.Checked
	$settings.ecores     = $ecores.Checked
	Save-Settings
}
$play.Add_Click({
	Read-Form
	Start-Process $cmd -ArgumentList "/v:on /s /c `"$(Get-PlayCommand)`"" -WorkingDirectory $root
	$form.Close()
})
$precompile.Add_Click({
	Read-Form
	Start-Process $cmd -ArgumentList "/s /c `"$(Get-PrecompileCommand)`"" -WorkingDirectory $root
})
$share.Add_Click({
	Read-Form
	Start-Process $cmd -ArgumentList "/s /c `"$(Get-PrecompileCommand '-ExportHints')`"" -WorkingDirectory $root
})
$logs.Add_Click({
	$folder = if (Test-Path "$root\_Build") { "$root\_Build\run-logs" } else { "$root\logs" }
	New-Item -ItemType Directory -Force $folder | Out-Null
	Start-Process explorer.exe $folder
})
$readme.Add_Click({
	$file = @("$root\README.md", "$root\docs\PORTABLE-README.md") | Where-Object { Test-Path $_ } | Select-Object -First 1
	if ($file) { Start-Process notepad.exe "`"$file`"" }
})
$form.Add_FormClosing({ Read-Form })
[void]$form.ShowDialog()
