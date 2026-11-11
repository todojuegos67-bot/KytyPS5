# Recording launcher: the game capped at 30 fps (KYTY_FLIP_RATE=1: at least two 60 Hz vblanks
# between flips; the game's clock keeps following the vblanks, so play keeps its speed) plus DLSS
# frame generation on top, with every performance switch of run-windows.ps1. Only the game's 3D
# view is capped: movies advance a frame per flip and would play at half speed.
#   tools\local\windows\record-fg.ps1 30 fps x2 = 60 fps presented (from the repository root)
#   .\record-fg.ps1 -Multiplier 4     30 fps x4 = 120 fps presented (for a 120 fps recording; the
#                                     75 Hz monitor itself shows at most 75 of them)
#   .\record-fg.ps1 -Fullscreen       fullscreen (16:9 image with black bars on the 21:9 monitor)
#   .\record-fg.ps1 -Width 3840 -Height 2160
#   .\record-fg.ps1 -NoHud            without the frame rate panel (KYTY_FPS_HUD, top right)
# Record with OBS (Game Capture source, Settings > Video > Common FPS Values 60 or 120) or the
# NVIDIA app overlay (Alt+Z, Settings > Video Capture > Frame rate 60 FPS).
param(
	[ValidateSet(2, 3, 4)][int]$Multiplier = 2,
	[switch]$Fullscreen,
	[int]$Width = 0,
	[int]$Height = 0,
	[switch]$NoHud
)
$ErrorActionPreference = 'Stop'
if (Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 }) {
	throw 'kyty_emulator.exe is already running: close it first'
}
$params = @{
	Set      = @('KYTY_FLIP_RATE=1', "KYTY_FPS_HUD=$(if ($NoHud) { '0' } else { '1' })")
	FrameGen = $Multiplier - 1
}
if ($Fullscreen) { $params['Fullscreen'] = $true; $params['AspectFit'] = $true }
if ($Width -gt 0) { $params['Width'] = $Width }
if ($Height -gt 0) { $params['Height'] = $Height }
Write-Host "Game capped at 30 fps, DLSS frame generation x$Multiplier -> $(30 * $Multiplier) fps presented"
& "$PSScriptRoot\..\..\..\run-windows.ps1" @params
