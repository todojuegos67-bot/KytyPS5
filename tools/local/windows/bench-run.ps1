param([string]$Label = 'run', [int]$Rounds = 3, [string]$Exe = '', [string[]]$Extra = @(),
      [switch]$KeepRunning, [string]$PresentMode = '', [int]$Vblank = 0, [string[]]$Set = @(),
      [string]$Patch = '', [switch]$Fullscreen, [switch]$Hdr, [switch]$ShotOnly, [switch]$NoWalk,
      [int]$StartupSeconds = 100, [int]$StartupAttempts = 3, [switch]$NoPrecompile, [string]$Config = '',
      [string]$Game = '', [switch]$LoadGame, [switch]$KeepCinematics)
# End to end: fresh baseline save, launch, skip to the game, walk the fixed path once the HUD
# is really up, measure. Screen states are told apart by pixel statistics.
$S = $PSScriptRoot
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $root
# run-windows.ps1 puts the launch's switches into this PowerShell process's environment for its Start-Process and leaves
# them there: the KYTY_ variables as they were here are put back after each launch (Restore-KytyEnvironment), or a -Set of
# one launch reached every later one started from the same session (ab-exe's -SetB KYTY_SPECULATE=4 ran its A side's
# later launches in mode 4 too, 11-09).
$kytyEnvironment = @{}
Get-ChildItem env: | Where-Object { $_.Name -like 'KYTY_*' } | ForEach-Object { $kytyEnvironment[$_.Name] = $_.Value }
function Restore-KytyEnvironment {
	Get-ChildItem env: | Where-Object { $_.Name -like 'KYTY_*' -and !$kytyEnvironment.ContainsKey($_.Name) } |
		ForEach-Object { Remove-Item "env:$($_.Name)" }
	foreach ($name in $kytyEnvironment.Keys) { Set-Item "env:$name" $kytyEnvironment[$name] }
}
# Screenshots and the start-up states go to _Build\bench.
$out = "$root\_Build\bench"
New-Item -ItemType Directory -Force $out | Out-Null
function Key([string]$key, [int]$ms = 250) { powershell -NoProfile -File "$S\key.ps1" -Key $key -HoldMs $ms | Out-Null }
function Shot([string]$name) { powershell -NoProfile -File "$S\screen.ps1" -Out "$out\$name" -Scale 0.25 | Out-Null; "$out\$name" }
Add-Type -AssemblyName System.Drawing
# In-process capture and input for the start-up loop: a PowerShell process per screenshot or key
# (compiling its helper each time) took seconds, so cinematics played long before a skip.
Add-Type -ReferencedAssemblies System.Drawing @"
using System; using System.Drawing; using System.Runtime.InteropServices;
public class BenchWin {
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint code, uint type);
  [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, IntPtr extra);
  [DllImport("imm32.dll")] public static extern IntPtr ImmGetDefaultIMEWnd(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, uint msg, IntPtr w, IntPtr l);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [StructLayout(LayoutKind.Sequential)] public struct KEYBDINPUT { public ushort wVk, wScan; public uint dwFlags, time; public IntPtr dwExtraInfo; }
  [StructLayout(LayoutKind.Explicit, Size=40)] public struct INPUT { [FieldOffset(0)] public uint type; [FieldOffset(8)] public KEYBDINPUT ki; }
  [DllImport("user32.dll")] public static extern uint SendInput(uint n, INPUT[] i, int size);
  public static bool Focus(IntPtr h) {
    if (IsIconic(h)) ShowWindow(h, 9);
    if (GetForegroundWindow() != h) { keybd_event(0x12, 0, 0, IntPtr.Zero); keybd_event(0x12, 0, 2, IntPtr.Zero); SetForegroundWindow(h); System.Threading.Thread.Sleep(150); }
    return GetForegroundWindow() == h;
  }
  public static void Key(ushort vk, bool up) {
    var i = new INPUT[1]; i[0].type = 1; i[0].ki.wScan = (ushort)MapVirtualKey(vk, 0);
    i[0].ki.dwFlags = 8u | (up ? 2u : 0u);
    SendInput(1, i, Marshal.SizeOf(typeof(INPUT)));
  }
  // The client area (the game image) scaled to 640x360, or null: State's positions are in that image, whatever the
  // display's resolution or the window's place.
  public static Bitmap Client(IntPtr h) {
    RECT r; POINT o = new POINT();
    if (!GetClientRect(h, out r) || !ClientToScreen(h, ref o) || r.Right <= 0 || r.Bottom <= 0) return null;
    using (var full = new Bitmap(r.Right, r.Bottom)) {
      using (var g = Graphics.FromImage(full)) g.CopyFromScreen(o.X, o.Y, 0, 0, full.Size);
      var small = new Bitmap(640, 360);
      using (var g2 = Graphics.FromImage(small)) { g2.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bilinear; g2.DrawImage(full, 0, 0, 640, 360); }
      return small;
    }
  }
}
"@
[BenchWin]::SetProcessDPIAware() | Out-Null
function Window { Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object MainWindowHandle -ne 0 | Select-Object -First 1 }
# Press in the emulator window only (never type into another window); Options = Enter, Cross = J.
function Press([string]$key, [int]$ms = 150) {
	$w = Window; if (!$w -or ![BenchWin]::Focus($w.MainWindowHandle)) { $script:unfocused++; return }
	$ime = [BenchWin]::ImmGetDefaultIMEWnd($w.MainWindowHandle)
	if ($ime -ne [IntPtr]::Zero) { [BenchWin]::SendMessage($ime, 0x283, [IntPtr]6, [IntPtr]::Zero) | Out-Null }
	$code = if ($key -eq 'enter') { 0x0D } elseif ($key -eq 'down') { 0x28 } else { [int][char]$key.ToUpper() }
	[BenchWin]::Key([uint16]$code, $false); Start-Sleep -Milliseconds $ms; [BenchWin]::Key([uint16]$code, $true)
}
function State($source) {
	$bmp = if ($source -is [System.Drawing.Bitmap]) { $source } else { [System.Drawing.Bitmap]::FromFile($source) }
	try {
		# HUD: the red-edged item slots in the bottom-left corner, and the red health bar at the top
		# left (the intro cinematic's fire alone matched the corner, with no bar).
		$red = 0
		for ($y = 250; $y -lt 360; $y += 1) { for ($x = 0; $x -lt 130; $x += 1) {
			$c = $bmp.GetPixel($x, $y); if ($c.R -gt 90 -and $c.R -gt 2 * $c.G -and $c.R -gt 2 * $c.B) { $red++ } } }
		$bar = 0
		for ($y = 2; $y -lt 22; $y += 1) { for ($x = 10; $x -lt 140; $x += 1) {
			$c = $bmp.GetPixel($x, $y); if ($c.R -gt 90 -and $c.R -gt 2 * $c.G -and $c.R -gt 2 * $c.B) { $bar++ } } }
		# In game: 109 slot and 151 bar pixels (the full-screen captures this replaced also counted
		# red desktop icons next to the window).
		if ($red -gt 70 -and $bar -gt 100) { return 'hud' }
		# A character in soul form (half health, a short bar: 35 slot and 12 bar pixels) has the green
		# stamina bar under it (180 pixels; none in the menus, prompt, fog or cinematics).
		$stamina = 0
		if ($red -gt 20) {
			for ($y = 2; $y -lt 30; $y += 1) { for ($x = 10; $x -lt 200; $x += 1) {
				$c = $bmp.GetPixel($x, $y); if ($c.G -gt 80 -and $c.G -gt 1.5 * $c.R -and $c.G -gt 1.3 * $c.B) { $stamina++ } } }
		}
		if ($stamina -gt 120) { return 'hud' }
		# Offline prompt: black screen with a lit box in the middle. Menu: green-tinted.
		$dark = 0; $total = 0; $sr = 0; $sg = 0; $sb = 0; $center = 0; $sat = 0
		for ($y = 0; $y -lt 360; $y += 4) { for ($x = 0; $x -lt 640; $x += 4) {
			$c = $bmp.GetPixel($x, $y); $total++; $sr += $c.R; $sg += $c.G; $sb += $c.B
			$sat += [Math]::Max([Math]::Max($c.R, $c.G), $c.B) - [Math]::Min([Math]::Min($c.R, $c.G), $c.B)
			if ($c.R + $c.G + $c.B -lt 30) { $dark++ }
			if ($x -gt 200 -and $x -lt 440 -and $y -gt 110 -and $y -lt 240 -and ($c.R + $c.G + $c.B) -gt 400) { $center++ } } }
		# The prompt's selected option ("Continue Offline") glows teal; the title logo is grey.
		$teal = 0
		for ($y = 178; $y -le 210; $y += 2) { for ($x = 220; $x -lt 420; $x += 2) {
			$c = $bmp.GetPixel($x, $y); if ($c.G -gt 45 -and $c.G -gt $c.R + 25 -and [Math]::Abs([int]$c.G - [int]$c.B) -le 12) { $teal++ } } }
		if ($dark / $total -gt 0.75 -and $center -gt 3 -and $teal -gt 20) { return 'prompt' }
		if ($sg -gt 1.25 * $sr -and $sg -gt 1.1 * $sb) { return 'menu' }
		if ($dark / $total -gt 0.97) { return 'dark' }
		# Cinematics are letterboxed: black bands over and under a lit picture.
		$bands = 0; $bandTotal = 0; $middle = 0; $middleTotal = 0
		for ($x = 0; $x -lt 640; $x += 8) {
			foreach ($y in @(4, 12, 20, 28, 332, 340, 348, 356)) { $c = $bmp.GetPixel($x, $y); $bandTotal++; if ($c.R + $c.G + $c.B -lt 30) { $bands++ } }
			for ($y = 80; $y -lt 280; $y += 20) { $c = $bmp.GetPixel($x, $y); $middleTotal++; if ($c.R + $c.G + $c.B -lt 30) { $middle++ } }
		}
		if ($bands / $bandTotal -gt 0.97 -and $middle / $middleTotal -lt 0.8) { return 'cinematic' }
		# After Continue: the loading fog is light grey without dark pixels; the cinematic after it
		# is dim and nearly colourless (game pictures have colour), long before it is letterboxed.
		$lum = ($sr + $sg + $sb) / (3.0 * $total); $sat = $sat / $total
		if ($lum -gt 60 -and $dark -eq 0 -and $sat -lt 5) { return 'fog' }
		if ($lum -lt 30 -and $sat -lt 2.5) { return 'mono' }
		return 'other'
	} finally { if ($source -isnot [System.Drawing.Bitmap]) { $bmp.Dispose() } }
}

# (Only this checkout's emulators: another checkout's session may be running one.)
function Live { Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 -and $_.Path -like "$root*" } }
for ($i = 0; $i -lt 60 -and (Live); $i++) {
	Live | Stop-Process -Force -ErrorAction SilentlyContinue
	Start-Sleep 1
}
python tools\local\bench-windows.py reset
# Never start on the user's own save: without an installed baseline (bench-windows.py prepare)
# reset refuses, and the run stops here.
if ($LASTEXITCODE -ne 0) { 'baseline save not installed: run python tools\local\bench-windows.py prepare'; exit 1 }
$params = @{}
if ($Exe) { $params['Exe'] = $Exe }
if ($PresentMode) { $params['PresentMode'] = $PresentMode }
if ($Vblank -gt 0) { $params['Vblank'] = $Vblank }
if ($Set.Count) { $params['Set'] = $Set }
if ($Patch) { $params['Patch'] = $Patch }
if ($Fullscreen) { $params['Fullscreen'] = $true }
if ($Hdr) { $params['Hdr'] = $true }
if ($Config) { $params['Config'] = $Config } # another launch config (run-windows.ps1 -Config)
# -Game <folder>: another game (version) for this run; game-path.txt keeps the folder remembered before
# it (run-windows.ps1 remembers a -Game).
$gameFile = "$root\game-path.txt"
$remembered = if (Test-Path $gameFile) { [IO.File]::ReadAllBytes($gameFile) }
function Restore-GamePath {
	if (!$Game) { return }
	if ($remembered) { [IO.File]::WriteAllBytes($gameFile, $remembered) } else { Remove-Item $gameFile -ErrorAction SilentlyContinue }
}
if ($Game) { $params['Game'] = $Game }
# -NoPrecompile: the driver cache as it is (first-encounter measurements with KYTY_SHADER_WARMUP=0).
if (!$NoPrecompile) { & "$root\run-windows.ps1" -Precompile -Width 1280 -Height 720 @params *> $null; Restore-GamePath; Restore-KytyEnvironment }
# The HUD is up about 60 s after launch: a start-up still short of it after -StartupSeconds is stuck
# (seen once in the attract loop), so it starts again from a fresh baseline save.
for ($attempt = 1; $attempt -le $StartupAttempts; $attempt++) {
if ($attempt -gt 1) {
	for ($i = 0; $i -lt 60 -and (Live); $i++) {
		Live | Stop-Process -Force -ErrorAction SilentlyContinue
		Start-Sleep 1
	}
	python tools\local\bench-windows.py reset | Out-Null
	if ($LASTEXITCODE -ne 0) { 'baseline save not installed: run python tools\local\bench-windows.py prepare'; exit 1 }
}
$env:KYTY_LIVE_FILE = "$root\_Build\windows-bench\live-commands.txt"
foreach ($pair in $Extra) { $k, $v = $pair -split '=', 2; Set-Item "env:$k" $v }
& "$root\run-windows.ps1" @params 6>&1 | Select-Object -Last 1
Restore-GamePath
# (The emulator has its environment: the switches, -Extra's and KYTY_LIVE_FILE leave this session's.)
Restore-KytyEnvironment

$start = Get-Date
$state = ''
# KYTY_BENCH_SHOTS=<dir>: keep a picture of the start-up screen about every second, named by time
# and state (for tuning the skips).
$shots = $env:KYTY_BENCH_SHOTS
if ($shots) { New-Item -ItemType Directory -Force $shots | Out-Null }
$lastShot = Get-Date '2000-01-01'
# Captures and presses skipped because the emulator window could not be brought to the front (a
# start-up that waits at the title for input it never got is not a hang).
$script:unfocused = 0
$answered = $false  # the offline prompt was answered: the game loads (fog), then a cinematic
$fogSeen = $false; $earlySkips = 0; $lastSkip = Get-Date '2000-01-01'
$log = New-Object System.Collections.Generic.List[string]
$last = ''
while (((Get-Date) - $start).TotalSeconds -lt $StartupSeconds) {
	if (!(Get-Process kyty_emulator -ErrorAction SilentlyContinue)) { $state = 'exited'; break }
	$w = Window
	# The capture reads the screen where the window is: it must be in front (as screen.ps1 did).
	$bmp = if ($w -and [BenchWin]::Focus($w.MainWindowHandle)) { [BenchWin]::Client($w.MainWindowHandle) } else { $null }
	if (!$bmp) { $script:unfocused++; Start-Sleep -Milliseconds 500; continue }
	try {
		$state = State $bmp
		if ($shots -and ((Get-Date) - $lastShot).TotalMilliseconds -ge 1000) {
			$bmp.Save(("{0}\{1:000.0}-{2}.png" -f $shots, ((Get-Date) - $start).TotalSeconds, $state))
			$lastShot = Get-Date
		}
	} finally { $bmp.Dispose() }
	if ($state -ne $last) { $log.Add(("{0,6:N1} s {1}" -f ((Get-Date) - $start).TotalSeconds, $state)); $last = $state }
	if ($state -eq 'hud') { break }
	if ($state -eq 'prompt') { Press j; $answered = $true; Start-Sleep -Milliseconds 1500; continue }
	# -LoadGame: the main menu's Load Game (its second item) instead of Continue, once.
	if ($state -eq 'menu' -and $LoadGame -and !$script:loadChosen) { Start-Sleep -Milliseconds 1500; Press s 300; Start-Sleep -Milliseconds 1500; $script:loadChosen = $true; continue }
	if ($state -eq 'menu') { Press j; Start-Sleep -Milliseconds 1500; continue }
	# A cinematic, or before the prompt the title: Options opens the skip prompt (or leaves the
	# title), Cross held confirms (250 ms let go just before its bar filled). After the prompt only
	# a cinematic is skipped (Options elsewhere would open the in-game menu): a letterboxed picture,
	# or the dim colourless one that follows the loading fog (skipped at most 3 times, 4 s apart,
	# so a fade into the game is never taken for it).
	# -LoadGame: in the save list that follows, the second save (one move down) before it is chosen.
	if ($LoadGame -and $answered -and !$script:listMoved -and $state -eq 'cinematic') { Press s 200; Start-Sleep -Milliseconds 700; $script:listMoved = $true }
	if ($state -eq 'fog') { $fogSeen = $true }
	$early = $answered -and $fogSeen -and $state -eq 'mono' -and $earlySkips -lt 3 -and
	         ((Get-Date) - $lastSkip).TotalSeconds -ge 4
	# -KeepCinematics: what follows the offline prompt plays (a save that starts in a cutscene).
	if (!($KeepCinematics -and $answered) -and ($state -eq 'cinematic' -or (($state -eq 'other' -or $state -eq 'mono') -and !$answered) -or $early)) {
		if ($early) { $earlySkips++ }
		Press enter; Start-Sleep -Milliseconds 700; Press j 1500; Start-Sleep -Milliseconds 800
		$lastSkip = Get-Date
		continue
	}
	Start-Sleep -Milliseconds 500
}
$log | Set-Content "$out\bench-states.txt" -Encoding ascii
if ($state -eq 'hud') { break }
"no HUD after $StartupSeconds s (attempt $attempt, last state $state, window not in front $($script:unfocused)x; states: $($log -join ' | '))"
}
if ($state -ne 'hud') { "no HUD in $StartupAttempts start-ups"; exit 1 }
"HUD after {0:N0} s ({1})" -f ((Get-Date) - $start).TotalSeconds, ($log -join ' | ')
Start-Sleep 10
if (!$NoWalk) { Key w 10000; Start-Sleep 10 }
Shot 'scene.png' | Out-Null
if ($ShotOnly) { powershell -NoProfile -File "$S\screen.ps1" -Out "$out\scene-$Label.png" -Scale 0.25 | Out-Null; "screenshot: $out\scene-$Label.png"; if (!$KeepRunning) { Live | Stop-Process -Force -ErrorAction SilentlyContinue }; return }
$commands = (1..$Rounds | ForEach-Object { "measure 20 $Label-$_" }) -join ' ; '
python tools\local\bench-windows.py live $commands | Select-String 'LIVE_MEASURE' | ForEach-Object { $_.Line }
if (!$KeepRunning) { Live | Stop-Process -Force -ErrorAction SilentlyContinue }
