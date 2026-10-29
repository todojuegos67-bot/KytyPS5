# Arranca este shadPS4.exe como lo hace Launcher-bloodborne.ps1 al pulsar JUGAR (modo seguro), usando las
# funciones del propio lanzador (se carga con -NoUi: no abre ventana ni cambia launcher.ini).
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$launcherDir = Split-Path -Parent $here
. (Join-Path $launcherDir 'Launcher-bloodborne.ps1') -NoUi
$userDir = Join-Path $launcherDir 'user'
$emu = Join-Path $here 'shadPS4.exe'
$eboot = Join-Path $launcherDir 'games\CUSA03173\eboot.bin'
$bb = (Get-ChildItem Env: | Where-Object { $_.Name -like 'BB_*' } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' '
Write-Log "---- PRUEBA REESCALADO ---- $bb"
Invoke-SoundFix $userDir
$eargs = @('-g', ('"' + $eboot + '"'))
Write-Log "Prueba reescalado: arranque `"$emu`" $($eargs -join ' ') (trabajo: $launcherDir, user: $userDir)"
$p = Start-Process -FilePath $emu -ArgumentList $eargs -WorkingDirectory $launcherDir -PassThru
Write-Log "Prueba reescalado: shadPS4 arrancado (PID $($p.Id))"
