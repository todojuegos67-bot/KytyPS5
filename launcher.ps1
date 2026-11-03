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
	'16:9' = '1280x720', '1600x900', '1920x1080', '2560x1440', '3840x2160'
	'21:9' = '2560x1080', '3440x1440', '3840x1600', '5120x2160'
	'32:9' = '3840x1080', '5120x1440', '7680x2160' }
# The height Demon's Souls renders its 3D at (run-windows.ps1 -RenderHeight); its output and UI stay 3840x2160.
$renderHeights = '2160 (native 4K)', '1800', '1620', '1512 (PS5 Pro)', '1440 (PS5)', '1296', '1152', '1080', '900', '720'
# What makes the 3840x2160 image of the 3D (run-windows.ps1 -Upscaler): the game's TAA and stretch (the PS5's), or DLSS in
# place of Sony's PS5 Pro upscaler on the game's PS5 Pro path (NVIDIA's Streamline runtime, as frame generation).
$upscalers = 'Game (PS5)', 'DLSS (PS5 Pro path)', 'AMD FSR (PS5 Pro path)', 'Off (no anti-aliasing)'
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

$settings = [ordered]@{ game = ''; resolution = '2560x1440'; fullscreen = $false; aspect = $true; language = 1; redzone = $true; cheats = $false;
	ecores = $false; fps120 = $false; render = 1440; ratio = '16:9'; quality = 'standard'; vsync = $false; hdr = $false; framegen = 0
	hud = 0; streamline = ''; safegpu = $false; upscaler = ''; render_dlss = 1080; fsr = $false; output = 'auto'
	lowlatency = $false; present_direct = $false; vrr_flips = $false; antilag = $false; keep_resident = $false; fps30 = $false; pad_speaker = $true
	extra_args = ''; dlss_preset = ''; dlss_sharpness = ''; fsr_sharpness = ''; gpu = ''; output_auto = $true; audio_output = '' }
if (Test-Path $settingsPath) {
	$saved = Get-Content $settingsPath -Raw | ConvertFrom-Json
	foreach ($property in $saved.PSObject.Properties) { if ($settings.Contains($property.Name)) { $settings[$property.Name] = $property.Value } }
	# 4K was the game output's default, written by every save: a file from before Automatic has it as Automatic once
	# (KytyLauncher.exe's Settings does the same); a 4K chosen since stays.
	if (!$saved.PSObject.Properties['output_auto'] -and $settings.output -eq '4k') { $settings.output = 'auto' }
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

# The primary display's height in pixels, its current mode's (not scaled by Windows' display scaling), or 0
# (KytyLauncher.exe's Native.DisplayHeight).
function Get-DisplayHeight {
	Add-Type -Namespace KytyLauncher -Name Display -MemberDefinition @'
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
public struct DevMode {
	[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
	public ushort dmSpecVersion, dmDriverVersion, dmSize, dmDriverExtra;
	public uint dmFields;
	public int dmPositionX, dmPositionY;
	public uint dmDisplayOrientation, dmDisplayFixedOutput;
	public short dmColor, dmDuplex, dmYResolution, dmTTOption, dmCollate;
	[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
	public ushort dmLogPixels;
	public uint dmBitsPerPel, dmPelsWidth, dmPelsHeight, dmDisplayFlags, dmDisplayFrequency;
}
[DllImport("user32.dll", CharSet = CharSet.Unicode)]
static extern bool EnumDisplaySettingsW(string deviceName, int modeNumber, ref DevMode mode);
public static int Height() {
	var mode = new DevMode();
	mode.dmSize = (ushort)Marshal.SizeOf(typeof(DevMode));
	return EnumDisplaySettingsW(null, -1, ref mode) ? (int)mode.dmPelsHeight : 0;
}
'@
	return [KytyLauncher.Display]::Height()
}
$displayHeight = Get-DisplayHeight

# Game output "Automatic" (KytyLauncher.exe's Commands.AutoHdOutput): the game's 1080p output for a 16:9 game window (in
# full screen, the screen) at most 1080 lines high, 4K otherwise. A window larger than the display is shrunk to it.
function Test-AutoHdOutput([string]$size, [bool]$full, [string]$shape) {
	$height = 0
	if ($full) {
		$height = $displayHeight
	} else {
		$parts = $size -split 'x'
		if ($parts.Count -lt 2 -or ![int]::TryParse($parts[1], [ref]$height)) { $height = 0 }
		elseif ($displayHeight -gt 0 -and $height -gt $displayHeight) { $height = $displayHeight }
	}
	return $shape -eq '16:9' -and $height -gt 0 -and $height -le 1080
}
function Test-HdOutput {
	return $settings.output -eq '1080p' -or
		($settings.output -eq 'auto' -and (Test-AutoHdOutput ([string]$settings.resolution) ([bool]$settings.fullscreen) ([string]$settings.ratio)))
}
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
# FSR frame generation (run-windows -Fsr), any GPU: AMD's signed runtime next to the emulator, FSR SDK 2's loader with
# its frame generation DLL, or FidelityFX SDK 1.1.4's amd_fidelityfx_vk.dll.
function Test-AmdSigned([string]$name) {
	$path = Join-Path (Split-Path $exe) $name
	(Test-Path -LiteralPath $path) -and
		((Get-AuthenticodeSignature -LiteralPath $path).SignerCertificate.Subject -match 'O=Advanced Micro Devices')
}
$fsrAvailable = ((Test-AmdSigned 'amd_fidelityfx_loader_vk.dll') -and (Test-AmdSigned 'amd_fidelityfx_framegeneration_vk.dll')) -or
	(Test-AmdSigned 'amd_fidelityfx_vk.dll')

# Cheats (_Patches\<title id>.json)
# A GoldHEN-style mods file: every mod has a name, an enabled flag and the memory writes it makes.
# The emulator already honours "enabled" (src/loader/gamePatch.cpp skips a mod whose flag is false),
# so choosing which mods to apply only means setting those flags.
$cheatTitleIds = @{}

function Get-CheatFile([string]$game) {
	if (!$game) { return '' }
	try {
		if (!$cheatTitleIds.ContainsKey($game)) {
			$param = Read-GameParam $game
			$cheatTitleIds[$game] = if ($param -and $param.titleId) { [string]$param.titleId } else { '' }
		}
		$id = $cheatTitleIds[$game]
		if (!$id) { return '' }
		$file = Join-Path $root "_Patches\$id.json"
		if (Test-Path -LiteralPath $file) { return $file }
	} catch {}
	return ''
}

function Read-CheatFile([string]$file) {
	# The file is UTF-8; Get-Content would read it as ANSI and can break the quotes of its text.
	return [System.IO.File]::ReadAllText($file, [System.Text.Encoding]::UTF8)
}

function Get-CheatData([string]$file) {
	return (Read-CheatFile $file | ConvertFrom-Json)
}

function Get-CheatCounts([string]$file) {
	$counts = @{ on = 0; total = 0 }
	if (!$file -or !(Test-Path -LiteralPath $file)) { return $counts }
	try {
		$mods = @((Get-CheatData $file).mods)
		$counts.total = $mods.Count
		foreach ($mod in $mods) {
			if ($mod.enabled -ne $false) { $counts.on++ }
		}
	} catch {}
	return $counts
}

function Save-CheatMods([string]$file, [string]$before, [bool[]]$wanted) {
	# Refuse to write over a file that changed while the dialog was open.
	if ((Read-CheatFile $file) -ne $before) { return 'file-changed' }
	$data = Read-CheatFile $file | ConvertFrom-Json
	$mods = @($data.mods)
	if ($mods.Count -ne $wanted.Count) { return 'count-changed' }
	for ($i = 0; $i -lt $mods.Count; $i++) {
		$mods[$i] | Add-Member -NotePropertyName enabled -NotePropertyValue ([bool]$wanted[$i]) -Force
	}
	$data.mods = $mods
	$json = $data | ConvertTo-Json -Depth 32
	Copy-Item -LiteralPath $file -Destination ($file + '.bak') -Force
	[System.IO.File]::WriteAllText($file, $json, (New-Object System.Text.UTF8Encoding($false)))
	# Read the file back: if it no longer holds the same names and flags, put the backup back.
	try {
		$check = @((Read-CheatFile $file | ConvertFrom-Json).mods)
		if ($check.Count -ne $mods.Count) { throw 'count' }
		for ($i = 0; $i -lt $check.Count; $i++) {
			if ([string]$check[$i].name -ne [string]$mods[$i].name) { throw 'name' }
			if ([bool]$check[$i].enabled -ne [bool]$wanted[$i]) { throw 'enabled' }
		}
	} catch {
		Copy-Item -LiteralPath ($file + '.bak') -Destination $file -Force
		return 'verify-failed'
	}
	return ''
}

function Show-CheatDialog([string]$file) {
	$data = $null
	try { $data = Get-CheatData $file } catch {
		[void][System.Windows.Forms.MessageBox]::Show("The cheat file cannot be read:`n$file`n`n$($_.Exception.Message)", 'Cheats')
		return
	}
	$mods = @($data.mods)
	if ($mods.Count -eq 0) {
		[void][System.Windows.Forms.MessageBox]::Show("That file holds no cheats:`n$file", 'Cheats')
		return
	}
	$before = Read-CheatFile $file

	$cheatForm = New-Object System.Windows.Forms.Form -Property @{
		Text = 'Cheats - ' + $(if ($data.name) { [string]$data.name } else { [string]$data.id })
		StartPosition = 'CenterParent'
		FormBorderStyle = 'FixedDialog'
		MaximizeBox = $false
		MinimizeBox = $false
		ClientSize = New-Object System.Drawing.Size(600, 480)
		Font = New-Object System.Drawing.Font('Segoe UI', 9)
	}
	$cheatList = New-Object System.Windows.Forms.CheckedListBox -Property @{
		Location = New-Object System.Drawing.Point(12, 12)
		Size = New-Object System.Drawing.Size(576, 330)
		CheckOnClick = $true
	}
	foreach ($mod in $mods) { [void]$cheatList.Items.Add([string]$mod.name, ($mod.enabled -ne $false)) }
	$cheatForm.Controls.Add($cheatList)

	$cheatDesc = New-Object System.Windows.Forms.TextBox -Property @{
		Location = New-Object System.Drawing.Point(12, 348)
		Size = New-Object System.Drawing.Size(576, 78)
		Multiline = $true
		ReadOnly = $true
		ScrollBars = 'Vertical'
	}
	$cheatForm.Controls.Add($cheatDesc)

	# Event handlers run in their own scope, so what they read is kept in the script's.
	$script:cheatDialogList = $cheatList
	$script:cheatDialogDesc = $cheatDesc
	$script:cheatDialogMods = $mods
	$cheatList.Add_SelectedIndexChanged({
		$index = $script:cheatDialogList.SelectedIndex
		if ($index -ge 0) {
			$script:cheatDialogDesc.Text = [string]$script:cheatDialogMods[$index].description
		}
	})

	$all = New-Object System.Windows.Forms.Button -Property @{ Text = 'Select all'; Location = New-Object System.Drawing.Point(12, 436); Size = New-Object System.Drawing.Size(100, 28) }
	$all.Add_Click({
		for ($i = 0; $i -lt $script:cheatDialogList.Items.Count; $i++) { $script:cheatDialogList.SetItemChecked($i, $true) }
	})
	$none = New-Object System.Windows.Forms.Button -Property @{ Text = 'Select none'; Location = New-Object System.Drawing.Point(120, 436); Size = New-Object System.Drawing.Size(100, 28) }
	$none.Add_Click({
		for ($i = 0; $i -lt $script:cheatDialogList.Items.Count; $i++) { $script:cheatDialogList.SetItemChecked($i, $false) }
	})
	$ok = New-Object System.Windows.Forms.Button -Property @{ Text = 'OK'; Location = New-Object System.Drawing.Point(408, 436); Size = New-Object System.Drawing.Size(90, 28); DialogResult = 'OK' }
	$cancel = New-Object System.Windows.Forms.Button -Property @{ Text = 'Cancel'; Location = New-Object System.Drawing.Point(504, 436); Size = New-Object System.Drawing.Size(90, 28); DialogResult = 'Cancel' }
	$cheatForm.Controls.AddRange(@($all, $none, $ok, $cancel))
	$cheatForm.AcceptButton = $ok
	$cheatForm.CancelButton = $cancel

	if ($cheatForm.ShowDialog($form) -ne 'OK') { return }

	$wanted = @()
	foreach ($i in 0..($mods.Count - 1)) { $wanted += [bool]$cheatList.GetItemChecked($i) }
	$result = Save-CheatMods $file $before $wanted
	if ($result -eq 'file-changed') {
		[void][System.Windows.Forms.MessageBox]::Show('The cheat file changed while the dialog was open; nothing was written.', 'Cheats')
	} elseif ($result -eq 'count-changed') {
		[void][System.Windows.Forms.MessageBox]::Show('The number of cheats does not match; nothing was written.', 'Cheats')
	} elseif ($result -eq 'verify-failed') {
		[void][System.Windows.Forms.MessageBox]::Show('The write could not be verified; the backup was restored.', 'Cheats')
	}
}

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

function Get-PlayCommand {
	$size = $settings.resolution -split 'x'
	$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$root\run-windows.ps1`"", '-Prompt', '-Follow',
		'-Width', $size[0], '-Height', $size[1], '-Language', $settings.language)
	# (-AspectFit: the 16:9 picture between black bars on a screen of another shape; with 21:9 or 32:9 the movies, the title
	# and the loading screens, KytyLauncher.exe's "wide_bars".)
	if ($settings.fullscreen) {
		$arguments += '-Fullscreen'
		if ($settings.aspect -and $settings.ratio -notin '21:9', '32:9') { $arguments += '-AspectFit' }
	}
	if ($settings.wide_bars -and $settings.ratio -in '21:9', '32:9') { $arguments += '-AspectFit' }
	if (!$settings.redzone) { $arguments += '-NoRedZone' }
	# (KytyLauncher.exe's 30 fps lock, Experimental: KYTY_FLIP_RATE=1 below, at 60 Hz.)
	if ($settings.fps120 -and !$settings.fps30) { $arguments += '-Fps120' }
	# Presents wait for the display's refresh (Vulkan FIFO): no tearing, and G-SYNC/FreeSync displays keep adapting.
	if ($settings.vsync) { $arguments += @('-PresentMode', 'Fifo') }
	# One frame queued instead of about two ($lowLatencyTip).
	# (AMD Anti-Lag, an experiment of KytyLauncher.exe, works on top of low latency.)
	if ($settings.lowlatency -or $settings.antilag) { $arguments += '-LowLatency' }
	# HDR10 output when Windows HDR is on for the display; SDR otherwise.
	if ($settings.hdr) { $arguments += '-Hdr' }
	# The cheats file of this title, when it has one: the emulator applies the mods left enabled in it.
	if ($settings.cheats) {
		$patch = Get-CheatFile $settings.game
		if ($patch) { $arguments += @('-Patch', "`"$patch`"") }
	}
	$dlss = $frameGenAvailable -and $settings.upscaler -eq 'dlss'
	# FSR 3 upscaling, built into the emulator: any graphics card (KytyLauncher.exe's Commands.FsrUpscaler).
	$fsrUpscaler = $settings.upscaler -eq 'fsr'
	# The upscaler's own render height (with DLSS or FSR 3 always passed: the PS5 Pro path's own height is 1512).
	$height = if ($dlss -or $fsrUpscaler) { [int]$settings.render_dlss } else { [int]$settings.render }
	if ($height -gt 0) { $arguments += @('-RenderHeight', $height) }
	if ($settings.ratio -in '21:9', '32:9') { $arguments += @('-Aspect', $settings.ratio) }
	if ($settings.quality -in 'low', 'lowest') {
		$arguments += @('-Quality', $settings.quality, '-TextureReduce', $(if ($settings.quality -eq 'lowest') { 2 } else { 1 }))
	}
	$fsrOn = $fsrAvailable -and $settings.fsr
	if ($fsrOn) { $arguments += '-Fsr' }
	if (!$fsrOn -and $frameGenAvailable -and [int]$settings.framegen -ge 2) { $arguments += @('-FrameGen', ([int]$settings.framegen - 1)) }
	if ($dlss) { $arguments += @('-Upscaler', 'dlss') } elseif ($fsrUpscaler) { $arguments += @('-Upscaler', 'fsr') }
	# (Off: Demon's Souls' TAA and its camera jitter off, the game's own upscaling kept; issue #220.)
	elseif ($settings.upscaler -eq 'off') { $arguments += @('-Antialiasing', 'off') }
	# (DLSS makes the 4K output: the 1080p output only with the game's own upscale.)
	if (Test-HdOutput) { $arguments += '-HdOutput' }
	# Game output 1440p, with DLSS or FSR 3 only: the PS5 Pro path's output at 2560x1440 (KytyLauncher.exe's Commands.Output1440).
	elseif ($settings.output -eq '1440p' -and ($dlss -or $fsrUpscaler)) { $arguments += @('-OutputHeight', 1440) }
	if ($frameGenAvailable -and ((!$fsrOn -and [int]$settings.framegen -ge 2) -or $dlss)) { $arguments += @('-StreamlineDir', "`"$streamline`"") }
	$arguments += @(Get-GpuArguments)
	# Switches of the config, one -Set argument (run-windows.ps1 splits it at the commas).
	$sets = @()
	# The emulator's panel (KYTY_FPS_HUD) at the top right of the game image: the frame rate and frame times, or with
	# the GPU, processor and memory too.
	if ([int]$settings.hud -ge 1) { $sets += "KYTY_FPS_HUD=$([Math]::Min([int]$settings.hud, 2))" }
	# Safe GPU mode: no overlapping compute dispatches ($safeGpuTip).
	if ($settings.safegpu) { $sets += 'KYTY_COMPUTE_CHAIN=0' }
	# KytyLauncher.exe's experiments: direct presentation (issue #37), AMD Anti-Lag and memory kept resident (issue #50).
	if ($settings.present_direct) { $sets += 'KYTY_PRESENT_DIRECT=1' }
	if ($settings.vrr_flips) { $sets += 'KYTY_VRR=1' }
	if ($settings.antilag) { $sets += 'KYTY_ANTI_LAG=1' }
	if ($settings.fps30) { $sets += 'KYTY_FLIP_RATE=1' }
	if ($settings.keep_resident) { $sets += @('KYTY_RESIDENT_MB=6144', 'KYTY_WORKING_SET_TRIM=0') }
	# The DualSense's speaker silent, over USB or Bluetooth: the pad's own volume 0 (the haptics stay).
	if (-not $settings.pad_speaker) { $sets += 'KYTY_DUALSENSE_SPEAKER_VOLUME=0' }
	# KytyLauncher.exe's DLSS model (a preset letter; none: the DLSS version's own choice).
	if ($dlss -and "$($settings.dlss_preset)" -cmatch '^[A-O]$') { $sets += "KYTY_DLSS_PRESET=$($settings.dlss_preset)" }
	if ($dlss -and "$($settings.dlss_sharpness)" -in @('0.3', '0.6', '0.9')) { $sets += "KYTY_DLSS_SHARPNESS=$($settings.dlss_sharpness)" }
	if ($fsrUpscaler -and "$($settings.fsr_sharpness)" -in @('0', '0.3', '0.9')) { $sets += "KYTY_FSR_SHARPNESS=$($settings.fsr_sharpness)" }
	# KytyLauncher.exe's audio output the game is told (issue #207; none: a TV).
	if ("$($settings.audio_output)" -cmatch '^(headphones|amplifier|soundbar)$') { $sets += "KYTY_AUDIO_OUTPUT=$($settings.audio_output)" }
	# KytyLauncher.exe's extra options (Advanced page): their -Set pairs join (or replace) the launcher's own.
	$extra = Get-ExtraOptions
	foreach ($option in $extra) {
		if ($option[0] -ne '-Set' -or $null -eq $option[1]) { continue }
		foreach ($pair in ($option[1] -split ',(?=[A-Za-z_][A-Za-z0-9_]*=)')) {
			$key = ($pair -split '=')[0]
			$sets = @($sets | Where-Object { ($_ -split '=')[0] -ne $key })
			$sets += $pair
		}
	}
	if ($sets.Count) { $arguments += @('-Set', ($sets -join ',')) }
	if ($settings.game) { $arguments += @('-Game', "`"$($settings.game)`"") }
	# Their other options in place of the launcher's of the same name (run-windows.ps1 takes each once), else added
	# (its own arguments from the 6th on: after -NoProfile -ExecutionPolicy Bypass -File <script>).
	foreach ($option in $extra) {
		if ($option[0] -eq '-Set') { continue }
		$list = [System.Collections.Generic.List[string]]@($arguments | ForEach-Object { [string]$_ })
		$at = -1
		for ($j = 5; $j -lt $list.Count; $j++) { if ($list[$j] -eq $option[0]) { $at = $j; break } }
		if ($at -ge 0) { $list.RemoveRange($at, $(if ($at + 1 -lt $list.Count -and !$list[$at + 1].StartsWith('-')) { 2 } else { 1 })) }
		$list.Add($option[0])
		if ($null -ne $option[1]) { $list.Add($(if ($option[1].Contains(' ')) { "`"$($option[1])`"" } else { $option[1] })) }
		$arguments = $list.ToArray()
	}
	# The console stays for the live log; after a crash it waits for a key.
	return "$powershellCommand " + ($arguments -join ' ') + ' & if !errorlevel! neq 0 pause'
}
# KytyLauncher.exe's extra options (extra_args), as on a run.cmd line (its Commands.ExtraOptions): each -Name with its
# value (the next part when it does not start with -), "double quotes" keeping spaces in one; a value without a name,
# or an option holding a character cmd would take itself, is left out.
$cmdCharacters = [char[]]@('&', '|', '<', '>', '^', '!', '%', '"')
function Get-ExtraOptions {
	$parts = New-Object System.Collections.Generic.List[string]
	$part = New-Object System.Text.StringBuilder
	$quoted = $false
	$any = $false
	foreach ($c in ([string]$settings.extra_args).ToCharArray()) {
		if ($c -eq [char]'"') { $quoted = !$quoted; $any = $true }
		elseif ([char]::IsWhiteSpace($c) -and !$quoted) { if ($any) { $parts.Add($part.ToString()) }; [void]$part.Clear(); $any = $false }
		else { [void]$part.Append($c); $any = $true }
	}
	if ($any) { $parts.Add($part.ToString()) }
	$options = @()
	for ($i = 0; $i -lt $parts.Count; $i++) {
		$name = $parts[$i]
		if ($name.Length -lt 2 -or $name[0] -ne [char]'-') { continue }
		$value = $null
		if ($i + 1 -lt $parts.Count -and !$parts[$i + 1].StartsWith('-')) { $i++; $value = $parts[$i] }
		if ($name.IndexOfAny($cmdCharacters) -ge 0 -or ($null -ne $value -and $value.IndexOfAny($cmdCharacters) -ge 0)) { continue }
		$options += , @($name, $value)
	}
	return , $options
}
# KytyLauncher.exe's graphics card (Advanced page, a PC with more than one), by its name, for the game and the precompile
# alike (issue #132); one holding a character cmd would act on is left out.
function Get-GpuArguments {
	$gpu = "$($settings.gpu)"
	if ($gpu -and $gpu.IndexOfAny([char[]]'&|<>^!%"') -lt 0) { return @('-Gpu', "`"$gpu`"") }
	return @()
}
function Get-PrecompileCommand([string]$switch = '') {
	$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$root\precompile-windows.ps1`"")
	if ($switch) { $arguments += $switch }
	if ($settings.ecores -and $efficiencyMask -ne 0) { $arguments += @('-Affinity', $efficiencyMask) }
	$arguments += @(Get-GpuArguments)
	if ($settings.game) { $arguments += @('-Game', "`"$($settings.game)`"") }
	return "$powershellCommand " + ($arguments -join ' ') + ' & pause'
}
function Save-Settings { $settings | ConvertTo-Json | Set-Content $settingsPath -Encoding UTF8 }
if ($DryRun) {
	"play:       cmd /v:on /s /c `"$(Get-PlayCommand)`""
	"precompile: cmd /s /c `"$(Get-PrecompileCommand)`""
	"hints:      cmd /s /c `"$(Get-PrecompileCommand '-ExportHints')`""
	"efficiency cores: 0x{0:X}" -f $efficiencyMask
	"display: $displayHeight lines"
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
# The game output shown: 1080p chosen, or Automatic for the window chosen above (Test-AutoHdOutput).
function Test-HdOutputShown {
	if (!$output) { return $false }
	return $output.SelectedIndex -eq 2 -or ($output.SelectedIndex -eq 0 -and
		(Test-AutoHdOutput ([string]$resolution.SelectedItem) ([bool]($fullscreen -and $fullscreen.Checked)) (Get-Ratio)))
}
function Update-RenderNote {
	$h = [int](([string]$render.SelectedItem -split ' ')[0])
	# (The 1080p output's 3D is at most 1080 high: $output below.)
	$capped = (Test-HdOutputShown) -and $h -gt 1080
	if ($capped) { $h = 1080 }
	# (DLSS's 4K output takes 1080 at least.)
	$raised = $upscaler.SelectedIndex -in 1, 2 -and !(Test-HdOutputShown) -and $h -lt 1080 -and $h -ne 720
	if ($raised) { $h = 1080 }
	$w = [Math]::Floor($h * (Get-RatioAspect) / 8.0 + 0.5) * 8
	$renderNote.Text = if ($capped) { "Demon's Souls' 3D at ${w}x${h} (1080 at most with the 1080p output)" } elseif ($raised) { "Demon's Souls' 3D at ${w}x${h} (1080 at least with DLSS's 4K output)" } else { "Demon's Souls' 3D at ${w}x${h}; higher = sharper, needs more GPU" }
}
$render.Add_SelectedIndexChanged({
	if (!$script:fillingRender) { $script:renderWanted = [int](([string]$render.SelectedItem -split ' ')[0]) }
	Update-RenderNote
})
$renderForDlss = ($frameGenAvailable -and $settings.upscaler -eq 'dlss') -or $settings.upscaler -eq 'fsr'
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
$upscaler.SelectedIndex = if ($frameGenAvailable -and $settings.upscaler -eq 'dlss') { 1 } elseif ($settings.upscaler -eq 'fsr') { 2 } elseif ($settings.upscaler -eq 'off') { 3 } else { 0 }
$upscalerNote = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '6,6,0,0' }
function Update-UpscalerNote {
	$upscalerNote.Text = if ($frameGenAvailable) { 'DLSS: NVIDIA RTX; 1440 = Quality, 1080 = Performance, 2160 = DLAA' }
		elseif (!$rtxGpu) { 'DLSS needs an NVIDIA RTX graphics card: here the game''s own TAA' }
		else { 'needs NVIDIA''s Streamline SDK (choose its folder under Frame generation)' }
}
Update-UpscalerNote
(New-Object System.Windows.Forms.ToolTip -Property @{ AutoPopDelay = 30000 }).SetToolTip($upscaler,
	"Game: the PS5's performance mode, its 3D anti-aliased by the game's TAA at the render height and stretched to 4K.`n" +
	"DLSS: the game's own PS5 Pro mode, Sony's upscaler (PSSR) replaced by NVIDIA DLSS: the 3D at the render height (2688x1512`n" +
	"at 1512, the PS5 Pro's; 1080 at least) made a 3840x2160 image the game finishes at 4K (bloom, depth of field, the HUD).`n" +
	"Sharper than the stretch, at some GPU cost. The in-game Cinematic mode stays native 4K without DLSS.`n" +
	"Each upscaler keeps its own render height. DLSS starts at 1080: on an RTX 5090 about the frame rate of the game's TAA`n" +
	"at 1440, with a sharper 4K image.`n" +
	"Off: the game's TAA and its camera jitter off: sharper and steadier, with jagged edges and shimmer on fine detail.")
$upscaler.Add_SelectedIndexChanged({
	# The height chosen goes back to the upscaler it was for; the newly chosen one's is shown.
	if ($script:renderForDlss) { $settings.render_dlss = $script:renderWanted } else { $settings.render = $script:renderWanted }
	$script:renderForDlss = $upscaler.SelectedIndex -in 1, 2
	$script:renderWanted = [int]$(if ($script:renderForDlss) { $settings.render_dlss } else { $settings.render })
	Select-RenderHeight
	Update-RenderNote
})
Add-Row 'Resolution' @($resolution)
Add-Row 'Aspect ratio' @($ratio, $ratioNote)
Add-Row 'Render height' @($render, $renderNote)
Add-Row 'Upscaler' @($upscaler, $upscalerNote)
# The game's output (run-windows.ps1 -HdOutput, src/loader/demonsSoulsHdOutput.h): 4K, or the game's own 1080p mode, its
# post-processing and HUD at 1920x1080 and its 3D at most 1080 high (with DLSS: DLSS upscales the 3D to that 1080p).
$output = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 205 }
$output.Items.AddRange(@('Automatic', '4K (PS5)', '1080p', '1440p (DLSS / FSR)'))
$output.SelectedIndex = [Math]::Max(0, [Array]::IndexOf(@('auto', '4k', '1080p', '1440p'), [string]$settings.output))
$outputNote = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '6,6,0,0'
	Text = 'Automatic: 1080p for a 16:9 window (full screen: the screen) up to 1080 lines, else 4K. 1080p: the game''s HD layout, effects and HUD at 1920x1080; less GPU work; with DLSS, DLSS makes the 1080p picture' }
$output.Add_SelectedIndexChanged({ Update-RenderNote })
$upscaler.Add_SelectedIndexChanged({ Update-RenderNote })
Add-Row 'Game output' @($output, $outputNote)
# (Automatic follows the window's size.)
$resolution.Add_SelectedIndexChanged({ Update-RenderNote })
Update-RenderNote
$fullscreen = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Fullscreen'; AutoSize = $true; Checked = [bool]$settings.fullscreen }
$aspect = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Keep the aspect ratio (black bars)'; AutoSize = $true; Checked = [bool]$settings.aspect }
$aspect.Enabled = $fullscreen.Checked
$fullscreen.Add_CheckedChanged({ $aspect.Enabled = $fullscreen.Checked; Update-RenderNote })
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
$lowLatencyTip = "The game keeps one finished frame waiting for the screen instead of about two, so what you press shows sooner:`n" +
	"at 60 Hz about 15-23 ms from the game's frame to the screen instead of about 30. A frame that comes late is then no`n" +
	"longer covered by a waiting one, so the frame pacing is a little less even."
$lowLatency = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Low latency (less input lag, slightly less even frame pacing)'; AutoSize = $true; Checked = [bool]$settings.lowlatency }
(New-Object System.Windows.Forms.ToolTip -Property @{ AutoPopDelay = 30000 }).SetToolTip($lowLatency, $lowLatencyTip)
Add-Row '' @($lowLatency)
$hdr = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'HDR (turn on "Use HDR" in Windows display settings first)'; AutoSize = $true; Checked = [bool]$settings.hdr }
Add-Row '' @($hdr)
$safeGpu = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Safe GPU mode: prevents GPU-hang crashes and blue screens, but the frame rate is about 37% lower'; AutoSize = $true; Checked = [bool]$settings.safegpu }
(New-Object System.Windows.Forms.ToolTip -Property @{ AutoPopDelay = 30000 }).SetToolTip($safeGpu, $safeGpuTip)
Add-Row '' @($safeGpu)
$cheats = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Cheats (_Patches\<title id>.json)'; AutoSize = $true; Checked = [bool]$settings.cheats }
Add-Row '' @($cheats)
$cheatButton = New-Object System.Windows.Forms.Button -Property @{ Text = 'Select...'; AutoSize = $true }
$cheatButton.Add_Click({
	$cheatFile = Get-CheatFile $game.Text.Trim()
	if (!$cheatFile) {
		[void][System.Windows.Forms.MessageBox]::Show('There is no _Patches\<title id>.json for this game.', 'Cheats')
		return
	}
	Show-CheatDialog $cheatFile
})
Add-Row '' @($cheatButton)
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
$fsr = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'AMD FSR frame generation (2x, any GPU; instead of DLSS)'; AutoSize = $true
	Checked = ($fsrAvailable -and [bool]$settings.fsr); Enabled = $fsrAvailable }
(New-Object System.Windows.Forms.ToolTip).SetToolTip($fsr, "AMD FSR: one generated frame between two of the game's, from its depth and motion (FSR 4 on Radeon RX 9000 with the FSR SDK 2.4 runtime, FSR 3.1 otherwise).`n" +
	"With VSync on, the shown frames are capped at the display's refresh (a 60 Hz display runs the game at 30 fps).`nNeeds AMD's FSR runtime next to the emulator: the FSR SDK 2.4 DLLs or amd_fidelityfx_vk.dll (README).")
Add-Row '' @($fsr)
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
(New-Object System.Windows.Forms.ToolTip).SetToolTip($share, ('The shaders your play recorded, without the game''s code, as my-hints-<game>.hints.zip: ' +
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
	$settings.lowlatency = $lowLatency.Checked
	$settings.hdr        = $hdr.Checked
	$settings.cheats     = $cheats.Checked
	$settings.safegpu    = $safeGpu.Checked
	$settings.hud        = $hud.SelectedIndex
	$height              = [int](([string]$render.SelectedItem -split ' ')[0])
	if ($renderForDlss) { $settings.render_dlss = $height } else { $settings.render = $height }
	$settings.ratio      = ([string]$ratio.SelectedItem -split ' ')[0]
	$settings.quality    = ([string]$quality.SelectedItem -split ' ')[0].ToLowerInvariant()
	$settings.output     = @('auto', '4k', '1080p', '1440p')[[Math]::Max(0, $output.SelectedIndex)]
	# (Kept as it was where the DLLs are missing.)
	if ($frameGenAvailable) { $settings.framegen = if ($frameGen.SelectedIndex -gt 0) { $frameGen.SelectedIndex + 1 } else { 0 } }
	# (DLSS kept as it was where its DLLs are missing; FSR 3 and the game's own anywhere.)
	if ($upscaler.SelectedIndex -eq 3) { $settings.upscaler = 'off' }
	elseif ($upscaler.SelectedIndex -eq 2) { $settings.upscaler = 'fsr' }
	elseif ($upscaler.SelectedIndex -eq 1) { if ($frameGenAvailable) { $settings.upscaler = 'dlss' } }
	elseif ($frameGenAvailable -or $settings.upscaler -ne 'dlss') { $settings.upscaler = '' }
	if ($fsrAvailable) { $settings.fsr = $fsr.Checked }
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
