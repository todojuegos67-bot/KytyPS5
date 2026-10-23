# KytyPS5 - Demon's Souls: lanzador (ventana). Launcher.cmd lo abre.
#   Jugar                -> arranca el juego; con "Frame generation AMD x2" (config\fsr3.json) o
#                           "x3" (config\fsr3-x3.json, 2 frames generados), si no, sin FG (config\sin-fg.json). Copia de las partidas antes.
#   Settings             -> carpeta del juego volcado (game-path.txt). La casilla se recuerda en launcher.ini.
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public static class FgWin {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
"@
[System.Windows.Forms.Application]::EnableVisualStyles()
$ErrorActionPreference = 'Stop'
# (Any error that stops this script, with where it came from, in launcher-error.txt next to it.)
trap { "$(Get-Date -Format s) $_`n$($_.ScriptStackTrace)" | Out-File -Append -LiteralPath (Join-Path $PSScriptRoot 'launcher-error.txt'); break }
$root     = $PSScriptRoot
$exeName  = 'kyty_emulator.exe'
$pathFile = Join-Path $root 'game-path.txt'
$iniFile  = Join-Path $root 'launcher.ini'
# Modos de frame generation: texto del desplegable -> config. Se recuerda el indice en launcher.ini.
$fgModes = @(
	@('Sin frame generation',                          'sin-fg.json'),
	@('---------- AMD (todas las graficas) ----------', ''),
	@('   Frame generation AMD x2',                     'fsr3.json'),
	@('   Frame generation AMD x3',                     'fsr3-x3.json'),
	@('   Frame generation AMD x4',                     'fsr3-x4.json'),
	@('   AMD MFG dinamico (x2-x4 automatico)',         'fsr3-mfg-dynamic.json'),
	@('---------- NVIDIA (RTX) ----------',             ''),
	@('   Frame generation DLSS x2',                    'dlss-x2.json'),
	@('   Frame generation DLSS x3',                    'dlss-x3.json'),
	@('   Frame generation DLSS x4',                    'dlss-x4.json'),
	@('   DLSS MFG dinamico (x2-x4 automatico)',        'dlss-dynamic.json')
)
# (10-23) Grafica y equipo: con una grafica que no es NVIDIA no se ofrece nada de DLSS (ni reescalado ni frame generation).
$script:gpuName = ''; $script:pcInfo = ''
try {
	$vcs = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name })
	$pick = $vcs | Where-Object { $_.Name -match 'NVIDIA|GeForce|Radeon|AMD|Arc' } | Select-Object -First 1
	if (-not $pick) { $pick = $vcs | Select-Object -First 1 }
	if ($pick) { $script:gpuName = "$($pick.Name)".Trim() }
	$cpu = (Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1).Name
	$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory / 1GB)
	$script:pcInfo = "$($cpu -replace '\s+', ' ')".Trim() + "  |  $ram GB RAM"
} catch { }
$script:isNvidia = (-not $script:gpuName) -or ($script:gpuName -match 'NVIDIA|GeForce|RTX|GTX|Quadro')
# (10-23) Base de datos de graficas: el modo recomendado y lo que se puede esperar. Medido en una RTX 5080 (1440p:
# ~75 reales, 224 en pantalla con frame generation dinamico; 4K cine ~40-48 reales) y en una de 8 GB simulada; el
# resto es una ESTIMACION por potencia relativa (el emulador tambien depende mucho de la CPU).
$script:gpuDb = @(
	@('RTX 509|RTX 508|RTX 409|RTX 408|RX 7900 XTX', 'Modo Souls extremo', '4K cine: ~40-50 FPS reales (150-224 en pantalla con FG). 1440p: 60-75 reales'),
	@('RTX 507|RTX 4070 Ti|RTX 4070 SUPER|RTX 309|RTX 3080 Ti|RX 907|RX 7900 XT|RX 7900 GRE|RX 6950', 'Modo Souls ultra', '1440p: ~50-65 FPS reales (150-224 en pantalla con FG)'),
	@('RTX 4070|RTX 4060 Ti 16|RTX 3080|RTX 506|RX 6800|RX 6900|RX 7800|RX 7700|Arc.*B5[78]0', 'Modo Souls equilibrado', '1440p: 45 FPS reales fijos (90 en pantalla con AMD x2)'),
	@('RTX 4060|RTX 3070|RTX 3060|RTX 207|RTX 208|RTX 2060|RX 66|RX 7600|RX 6700|Arc.*A7', 'Modo Souls ligero', '1080p: 30 FPS reales fijos (60 en pantalla con AMD x2); 8 GB de VRAM justos'),
	@('GTX|RX 5[0-9]|RX 6500|RX 6400|Arc.*A[35]|Radeon\(TM\) Graphics|Iris|UHD', 'Modo Souls PC low profile', '900p: 30 FPS reales fijos (60 con AMD x2); puede no ir fluido')
)
$script:gpuRec = $null
foreach ($row in $script:gpuDb) { if ($script:gpuName -match $row[0]) { $script:gpuRec = $row; break } }
if (-not $script:isNvidia) { $fgModes = @($fgModes | Where-Object { $_[1] -notlike 'dlss*' -and $_[0] -notmatch 'NVIDIA' }) }
# Reescalado: texto -> altura del 3D (run-windows.ps1 -RenderHeight), reescalador (-Upscale) y altura de salida
# (-UpscaleHeight; 0 = 1440). DLSS (NVIDIA RTX): DLAA 1x, Ultra calidad 1.3x, Calidad 1.5x, Equilibrado 1.72x,
# Rendimiento 2x, Ultra rendimiento 3x (720 -> 2160). XeSS 3 (todas las graficas): AA 1x, Ultra calidad plus 1.3x,
# Ultra calidad 1.5x, Calidad 1.7x, Equilibrado 2x, Rendimiento 2.3x (940 -> 2160). FSR 3.1 (graficas de bajo rendimiento).
# "juego": el 3D a menos resolucion y lo sube el propio juego.
$resModes = @(
	# Upscalers (DLSS, XeSS, FSR): the second column is the ratio of the monitor's height (detected by the emulator)
	# (the game's 4K output needs 720 at least: 720). Others: the render height itself.
	@('1440p nativo',   1440, '',     0),
	@('DLSS DLAA',      1.0,  'dlss', 0),
	@('DLSS U.calidad', 1.3,  'dlss', 0),
	# (10-21) DLSS sale a 4K por dentro (camino PS5 Pro): solo acepta 3D de 1080 lineas o mas, o 720 exacto
	# (rendimiento). 960 y 837 (los antiguos Calidad/Equilibrado) dejaban la imagen negra.
	@('DLSS Calidad',   1.333, 'dlss', 0),
	@('DLSS Equilib.',  1.72, 'dlss', 0),  # (10-22) solo con Modo 4K (2160/1.72 = 1256; en 1440p darian 837: pantalla negra)
	@('DLSS Rendim.',   2.0,  'dlss', 0),
	@('DLSS U.rendim.', 3.0,  'dlss', 0),  # (10-22) solo con Modo 4K (2160/3 = 720 justo; en 1440p darian 480)
	@('XeSS AA',        1.0,  'xess', 0),
	@('XeSS U.cal.+',   1.3,  'xess', 0),
	@('XeSS U.calidad', 1.5,  'xess', 0),
	@('XeSS Calidad',   1.7,  'xess', 0),
	@('XeSS Equilib.',  2.0,  'xess', 0),
	@('XeSS Rendim.',   2.3,  'xess', 0),
	@('FSR AA nativo',  1.0,  'fsr',  0),
	@('FSR U.calidad',  1.3,  'fsr',  0),
	@('FSR Calidad',    1.5,  'fsr',  0),
	@('FSR Equilib.',   1.72, 'fsr',  0),
	@('FSR Rendim.',    2.0,  'fsr',  0),
	@('FSR 900p (AA)',  1.6,  'fsr',  0),  # (10-23) PC low profile: el 3D a 900 (1440 / 1.6) con FSR y nitidez
	@('1080p juego',    1080, '',     0),
	@('900p juego',     900,  '',     0),
	@('720p juego',     720,  '',     0)
)
# Preset de DLSS: Auto (el de NVIDIA para cada modo), K (DLSS 4), L y M (DLSS 4.5).
$dlssPresets = @(
	@('DLSS auto',    ''),
	@('DLSS 4 (K)',   'K'),
	@('DLSS 4.5 (L)', 'L'),
	@('DLSS 4.5 (M)', 'M')
)
function Get-FgIndex {
	if (Test-Path -LiteralPath $iniFile) {
		$text = Get-Content -LiteralPath $iniFile -Raw
		# (10-22) Se guarda tambien el config (fgcfg=): el indice cambio al quitar los modos Smooth x5/x6/120.
		if ($text -match '(?m)^fgcfg=(.+)$') {
			$cfg = $Matches[1].Trim()
			for ($i = 0; $i -lt $fgModes.Count; $i++) { if ($fgModes[$i][1] -eq $cfg) { return $i } }
		}
		if ($text -match 'fgmode=(\d+)') {
			# Indice antiguo (lista de 15) -> su config; los quitados al mas parecido.
			$old = @('sin-fg.json', '', 'fsr3.json', 'fsr3-x3.json', 'fsr3-x4.json', 'fsr3-x3.json', 'fsr3-mfg-dynamic.json', '',
			         'dlss-x2.json', 'dlss-x3.json', 'dlss-x4.json', 'dlss-x4.json', 'dlss-x4.json', 'dlss-x3.json', 'dlss-dynamic.json')
			$n = [int]$Matches[1]
			if ($text -notmatch '(?m)^fgcfg=' -and $n -ge 0 -and $n -lt $old.Count -and $old[$n]) {
				for ($i = 0; $i -lt $fgModes.Count; $i++) { if ($fgModes[$i][1] -eq $old[$n]) { return $i } }
			}
			return $n
		}
		if ($text -match 'framegen=1') { return 1 }
		if ($text -match 'framegen=0') { return 0 }
	}
	return 1
}
function Get-IniValue([string]$key, [string]$default) {
	if (Test-Path -LiteralPath $iniFile) {
		$text = Get-Content -LiteralPath $iniFile -Raw
		if ($text -match "(?m)^$key=(.*)$") { return $Matches[1].Trim() }
	}
	return $default
}
function Set-IniValue([string]$key, [string]$value) {
	$lines = @()
	if (Test-Path -LiteralPath $iniFile) {
		$lines = @(Get-Content -LiteralPath $iniFile | Where-Object { $_ -and ($_ -notmatch "^$key=") -and ($_ -notmatch '^framegen=') })
	}
	$lines += "$key=$value"
	[IO.File]::WriteAllText($iniFile, (($lines -join "`r`n") + "`r`n"), (New-Object System.Text.UTF8Encoding $false))
}
function Set-FgIndex([int]$i) { Set-IniValue 'fgmode' "$i"; if ($i -ge 0 -and $i -lt $fgModes.Count -and $fgModes[$i][1]) { Set-IniValue 'fgcfg' $fgModes[$i][1] } }
# (10-22) Frame generation fijo (AMD x2): Modo graficas de 8 GB (30 FPS, 1080p, solo AA) o Modo 45 FPS (1440p, reescalado
# normal). Lo de 1080p y el filtro de AA siguen mirando solo chkLowEnd.
function Is-FixedFg { return ($chkLowEnd -and $chkLowEnd.Checked) -or ($chkLowEnd45 -and $chkLowEnd45.Checked) -or ($chkSoulsLow -and $chkSoulsLow.Checked) }  # (10-23) low profile: 30 reales con AMD x2
# (10-22) FPS desbloqueados siempre, salvo DLSS frame generation (Reflex) y los modos Souls ligero / equilibrado.
function Is-Unlocked { if (Is-FixedFg) { return $false }; $c = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]; return $true }  # (10-22 d) DLSS tambien desbloqueado: reflexfree (Reflex como unico limite)

function Get-GamePath {
	if (Test-Path -LiteralPath $pathFile) { return ((Get-Content -LiteralPath $pathFile -Raw) -replace '^\uFEFF', '').Trim() }
	return ''
}
function Set-GamePath([string]$path) {
	[IO.File]::WriteAllText($pathFile, $path, (New-Object System.Text.UTF8Encoding $false))
}
function Backup-Partidas {
	# Copia de seguridad de las partidas antes de jugar. Solo copia; nunca borra nada.
	$src = Join-Path $root '_SaveData'
	if (Test-Path -LiteralPath $src) {
		$dst = Join-Path $root ('_SaveData-copias\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
		New-Item -ItemType Directory -Force $dst | Out-Null
		Copy-Item -Recurse -Force (Join-Path $src '*') $dst
	}
}
function Start-Game {
	# (10-23) Aviso de RAM libre: solo LEE la memoria; no cierra ni vacia nada. Con poca RAM libre recomienda cerrar programas.
	try {
		$os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
		$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1); $totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB)
		if ($freeGB -lt 10) {
			$mine = @($PID)
			$top = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Id -notin $mine -and $_.ProcessName -notmatch '^(System|Idle|Memory Compression|Registry|csrss|wininit|winlogon|services|lsass|smss|svchost|dwm|explorer|powershell)$' } |
				Group-Object ProcessName | ForEach-Object { [pscustomobject]@{ Name = $_.Name; MB = [math]::Round((($_.Group | Measure-Object WorkingSet64 -Sum).Sum) / 1MB) } } |
				Sort-Object MB -Descending | Select-Object -First 5
			$list = ($top | ForEach-Object { "   - $($_.Name): $([math]::Round($_.MB / 1024, 1)) GB" }) -join "`n"
			$msg = "Solo quedan $freeGB GB de RAM libres (de $totalGB GB). El juego necesita unos 12 GB y puede ir a tirones o cerrarse.`n`n" +
				"Los programas que mas RAM usan ahora:`n$list`n`nCierralos si puedes (navegador, Discord, etc.).`n`nAceptar: jugar igualmente    Cancelar: volver"
			if ([System.Windows.Forms.MessageBox]::Show($msg, 'Memoria RAM', 'OKCancel', 'Warning') -ne 'OK') { return }
		}
	} catch { }
	if (-not (Get-GamePath) -or -not (Test-Path -LiteralPath (Get-GamePath))) {
		[System.Windows.Forms.MessageBox]::Show('Pon la carpeta del juego en Settings.', 'Launcher', 'OK', 'Warning') | Out-Null
		return
	}
	if (-not (Test-Path -LiteralPath (Join-Path $root $exeName))) {
		[System.Windows.Forms.MessageBox]::Show("Falta $exeName", 'Launcher', 'OK', 'Error') | Out-Null
		return
	}
	Backup-Partidas
	# (A folder with ram16.txt: the 16 GB profile, run-windows -Ram16.)
	$ram16 = Test-Path -LiteralPath (Join-Path $root 'ram16.txt')
	$config = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	# (10-21) PC gama baja: el juego a 30 reales (repartidos por igual) y frame generation de AMD x2 = 60.
	if (Is-FixedFg) { $config = 'fsr3.json' }
	# Rutas entre comillas: Start-Process une los argumentos con espacios, y una carpeta con espacios
	# (C:\Users\Juan Perez\...) partia la ruta y PowerShell se cerraba sin arrancar nada.
	$psargs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$(Join-Path $root 'run-windows.ps1')`"",
	          '-Exe', "`"$(Join-Path $root $exeName)`"", '-Config', "`"$(Join-Path $root "config\$config")`"", '-Fullscreen')
	if ($ram16 -or $chkPc16.Checked) { $psargs += '-Ram16' }
	# HDR10 (solo si "Usar HDR" esta activado en la pantalla de Windows) y VSync (modo FIFO: sin tearing).
	if ($chkHdr.Checked)   { $psargs += '-Hdr' }
	if ($chkVsync.Checked) { $psargs += @('-PresentMode', 'Fifo') }
	# Modo PC de pocos recursos: el emulador se comporta como con 16 GB de RAM (KYTY_SIMULATE_RAM_MB: sus perfiles de
	# poca memoria, sin precarga de pipelines, y run-windows vacia cada 30 s la RAM en uso por encima de 8 GB). La VRAM
	# la mide el emulador en la grafica de verdad. (Con menos de 24 GB de RAM pasa solo, sin la casilla.)
	# Modo PC de pocos recursos (16 GB de RAM, graficas de 8 GB o menos): perfiles de 16 GB de RAM y vaciado de la RAM
	# cada 30 s (KYTY_SIMULATE_RAM_MB), las caches de video limitadas a 6.5 GB (KYTY_VRAM_BUDGET_MB: con lo que el
	# juego y Windows usan aparte, por debajo de 8 GB) y texturas a la mitad y sombras mas cortas (-Quality low).
	# Modo GPU seguro: el trabajo de compute del juego sin solaparse (KYTY_COMPUTE_CHAIN=0), para graficas que se
	# cuelgan o dan pantallazo azul (~37% menos FPS).
	$sets = @()
	# (10-23) Reserva PS5 reducida retirada: el juego pide unos 11,8 GB fijos y con 12 GB fallaban asignaciones  # (10-23) reserva de memoria PS5 reducida
	# Modo TURBO (run-windows -Turbo): caches de video a 4.5 GB (unos 6 GB en total), texturas a un cuarto, detalle bajo
	# del juego y la RAM vaciada cada 15 s por encima de 5 GB: para PCs de 16 GB que se cerraban en zonas pesadas.
	if ($chkTurbo2.Checked) { $psargs += @('-Turbo2', '-Quality', 'low', '-TextureReduce', '2') }
	elseif ($chkLowPc.Checked) { $sets += 'KYTY_SIMULATE_RAM_MB=16384'; $psargs += @('-Turbo', '-Quality', 'low', '-TextureReduce', '2') }
	# (10-23) Modo GPU seguro retirado del launcher
	# Graficos: texturas (0 entera, 1 mitad, 2 cuarto) y distancia de dibujado (-Quality low/lowest: solo el LOD).
	$psargs += @('-TextureReduce', @('0', 'h', '1', '2')[$cmbTex.SelectedIndex])  # (10-23) Alto = h
	# (10-23) Anti-Lag solo en los modos Souls ligero y PC low profile (frame generation de AMD + Reflex); en los demas
	# no mejoraba y quitaba FPS.
	# (10-23 noche) Anti-Lag solo con graficas AMD (AMD Anti-Lag 2) en Souls ligero y PC low profile. En NVIDIA no: Reflex
	# sobre la frame generation de AMD cerro el juego (20261023-205956, en Fsr3::QueuePresent).
	# NVIDIA + frame generation de AMD: casilla "Anti-Lag (experimental)", solo sin reescalado DLSS (con DLSS cargado fue
	# el crash); usa Reflex por Vulkan, el camino anterior a hoy.
	$resNow = $script:resShown[[math]::Max(0, $cmbRes.SelectedIndex)]
	$nvAmd  = $script:isNvidia -and $chkAntiLag.Checked -and $config -like 'fsr3*' -and $resNow[2] -in 'xess', 'fsr'
	if ($nvAmd -or ((-not $script:isNvidia) -and ($chkLowEnd.Checked -or $chkSoulsLow.Checked))) {
		$psargs += '-AntiLag'
		# (10-23) Baja latencia: la CPU no prepara un frame por adelantado (~ -18 ms de retraso, algo menos de FPS) y
		# con DLSS Frame Generation, Reflex en modo Boost.
		# (10-23 tarde) KYTY_FRAME_PIPELINE=0 quitado: a 4K bajaba de ~45 a 35 reales (y x3 a 20): limitaba los FPS.
		if ($config -like 'dlss*') { $sets += 'KYTY_DLSSG_REFLEX=boost' }
	}
	if ($cmbRt.SelectedIndex -gt 0) { $psargs += @('-Ssgi', "$($cmbRt.SelectedIndex)") }
	# (10-23) Sombras RT retiradas: el juego tiene las suyas (Sombras Ultra)
	if ($cmbShadow.SelectedIndex -gt 0) { $psargs += @('-Shadows', @('0', '3', '1', '2')[$cmbShadow.SelectedIndex]) }  # (10-23) Alto = 3
	$rN = @('Ultra', '', 'Normal', 'Low'); if ($rN[$cmbRefl.SelectedIndex]) { $psargs += @('-Reflections', $rN[$cmbRefl.SelectedIndex]) }
	if ($rN[$cmbFade.SelectedIndex]) { $psargs += @('-ObjectFade', $rN[$cmbFade.SelectedIndex]) }
	$aN = @('Ultra', 'High', '', 'Low', 'Off'); if ($aN[$cmbAaS.SelectedIndex]) { $psargs += @('-AaSharp', $aN[$cmbAaS.SelectedIndex]) }
	$postOff = @($script:postChecks | Where-Object { -not $_[0].Checked } | ForEach-Object { $_[1] }) -join ','; if ($postOff) { $psargs += @('-PostOff', $postOff) }  # (10-23)
	$detNames = @('Ultra', 'VeryHigh', 'High', 'Normal', 'Low'); if ($cmbDet.SelectedIndex -ne 2) { $psargs += @('-Detail', $detNames[$cmbDet.SelectedIndex]) }  # (10-23) preset de detalle
	if ($cmbDist.SelectedIndex -ge 1) { $psargs += @('-Quality', @('', 'high', 'low', 'lowest')[$cmbDist.SelectedIndex]) }  # (10-23) Alto = high
	if ($chkExcl.Checked) { $psargs += '-ExclusiveFullscreen' }
	if ($cmbAspect.SelectedIndex -gt 0) { $psargs += @('-Aspect', @('16:9', '21:9', '32:9')[$cmbAspect.SelectedIndex]) }  # (10-23) ultrawide
	# Preset C con la imagen sin HUD (hudless) daba pantalla negra con solo el HUD (10-21): sin ese buffer.
	# (10-21) Preset C del frame generation de NVIDIA siempre (mas rapido; el emulador solo lo aplica a DLSS-G).
	$sets += 'KYTY_DLSSG_PRESET=C'
	$sets += 'KYTY_FG_PACING=0'  # (10-21) el reparto de frames del emulador (se quedaba en 20 FPS) apagado
	# (10-22) Modo 4K: los reescaladores calculan su resolucion sobre 2160 (salida 4K), no sobre los 1440 del monitor.
	$outH = Get-OutHeight
	if ($outH -ne 1440) { $sets += "KYTY_DISPLAY_HEIGHT=$outH" }
	$sets += 'KYTY_DS_SIXTY_STEP=0'  # (10-22) el paso de 60 en cinematico no arreglaba las cinematicas: el modo como en PS5 y cinematicas a 30 reales
	$sets += 'KYTY_DS_VARIABLE_STEP=1'  # (10-21) fisica al tiempo real de cada frame (physicsUseFixedTimestep=false)
	# Nitidez del reescalado (KYTY_UPSCALE_SHARPNESS): FSR con su propio RCAS, DLSS y XeSS con un paso de nitidez despues.
	$sharp = $sharpLevels[[math]::Max(0, $cmbSharp.SelectedIndex)][1]
	if ($sharp -ne '') { $sets += "KYTY_UPSCALE_SHARPNESS=$sharp" }
	# FPS desbloqueados con frame generation: el juego deja de estar limitado a 60 reales (KYTY_FRAMEGEN_BASE=0); los
	# modos Smooth siguen en sus FPS fijos (KYTY_FLIP_FPS) y DLSS-G se limita solo con Reflex al tope del monitor.
	$fgConfig = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	if (Is-FixedFg) { $fgConfig = 'fsr3.json'; $sets += $(if ($chkLowEnd45.Checked) { 'KYTY_FLIP_FPS=45' } else { 'KYTY_FLIP_FPS=30' }) }
	# Nombre del modo de frame generation para las metricas.
	# (10-23) El modo grafico para las metricas.
	Set-IniValue 'soulsmodo' $(if ($chkLowEnd.Checked) { 'Modo Souls ligero' } elseif ($chkLowEnd45.Checked) { 'Modo Souls equilibrado' } elseif ($chkSoulsUltra.Checked) { 'Modo Souls ultra' } elseif ($chkSoulsExt.Checked) { 'Modo Souls extremo' } elseif ($chkSoulsLow.Checked) { 'Modo Souls PC low profile' } else { 'Personalizado' })
	if ($chkLowEnd45.Checked) { Set-IniValue 'fglabel' 'Modo Souls equilibrado' } elseif ($chkLowEnd.Checked) { Set-IniValue 'fglabel' 'Modo Souls ligero' } elseif ($chkSoulsLow.Checked) { Set-IniValue 'fglabel' 'Modo Souls PC low profile' } else { Set-IniValue 'fglabel' ($fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][0].Trim()) }
	# (10-22) FPS desbloqueados solo sin frame generation: con frame generation el juego se queda en su base (60) y el
	# multiplicador llega al monitor; desbloqueado (110 reales x3 = 330) se pasaba de los 240 Hz y salia de G-Sync.
	# AMD x4: 56 FPS reales (224 mostrados), por debajo de 240 Hz para que G-SYNC vaya suave.
	# (10-22 b) FPS desbloqueados + frame generation DINAMICO: como en PC, el juego pasa de 60 reales y el dinamico
	# elige x2/x3/x4 con el tope de 224 a la salida (limite real = 224/x). Con frame generation fijo sigue a 60.
	if ((Is-Unlocked) -and -not (Is-FixedFg) -and ($fgConfig -like 'fsr3*' -or $fgConfig -like 'dlss*')) { $sets += 'KYTY_FRAMEGEN_BASE=0' }  # (10-22) AMD y sin FG: siempre desbloqueados; DLSS y los modos Souls/8 GB bloqueados
	if ($chkAf16.Checked) { $sets += 'KYTY_FORCE_AF=16' }
	if ($chkLod.Checked) { $sets += 'KYTY_TEXTURE_LOD_BIAS=auto' }  # (10-23) solo texturas 2D comprimidas de superficie  # (10-23) texturas con el detalle de la resolucion final
	$sets += 'KYTY_DLSSG_PACE=grid'  # (10-22 e) reflexfree esperaba el limite de Reflex en cada fotograma mostrado: 14 reales
	if ($cmbAo.SelectedIndex -gt 0) { $sets += "KYTY_GTAO=$($cmbAo.SelectedIndex)"; $sets += "KYTY_GTAO_STRENGTH=0.35" }  # (10-23) de cerca oscurecia demasiado (0.6)
	# (10-23) KYTY_SSR retirado (reflejos del juego en Reflections)
	# (10-23) KYTY_GUEST_LAZY_COMMIT retirado: bloqueaba el arranque y solo ahorraba ~0,5 GB
	# (10-23) KYTY_LOWMEM_VRAM retirado: con graficas de 8 GB bajaba el presupuesto a 4 GB y daba tirones (13-47% de tramos frente a 2-7%)
	$sets += 'KYTY_FAULT_STATS=1'
	$sets += 'KYTY_ALIAS_STATS=0'  # (10-23) el diagnostico de alias recorria todos los mapeos en cada map: tirones al cambiar de zona que empeoraban con la sesion
	if ($fgConfig -eq 'fsr3-x4.json') { $sets += 'KYTY_VRR_CAP_FPS=224' }  # (10-22 d) en DLSS el tope se aplicaba dos veces: 14 reales x4 = 56
	if ($fgConfig -like '*dyn*') { $sets += 'KYTY_DYN_QUALITY=0.95' }  # (10-22 g) llenar el monitor: el menor multiplicador que llega a ~224 (80 reales = x3, 120 = x2)
	if ($sets.Count -gt 0) { $psargs += @('-Set', ($sets -join ',')) }
	# Con frame generation: vblank virtual de 360 Hz (run-windows -Vblank; corrige la velocidad del juego). Por
	# debajo de 60 cada frame sale a su tiempo (20 ms -> ~20, no 33 como a 60 Hz: los trompicones); el juego sigue
	# hasta 60 reales (FrameGenFloor), videos y menus a 60, y los Smooth (KYTY_FLIP_FPS 40/45) en vblanks justos.
	# DLSS x4: solo Reflex (56 reales, 224 en pantalla; el modo mas suave, 10-19). x5 (40) y x6 (36)
	# caen justos en los 360 Hz (9 y 10 vblanks): el juego no dibuja frames de mas que se tirarian antes del DLSS.
	# (10-21) Modos 224: el juego a 56 reales repartidos por igual (vblank 224, flip cada 4) y x4 = 224; con 360 Hz y
	# un tope a 56 los 60 del juego se comian 4 frames por segundo (tirones cada 0.25 s).
	# (10-21 b) Con frame generation siempre 840 Hz y cada frame sale en cuanto esta listo (FLIP_WHEN_READY): por debajo
	# de 60 los frames ya no se redondean a la rejilla de vblanks (a 40 reales salian con +-3 ms de vaiven: tirones con G-Sync).
	if ($config -like 'dlss*' -or $config -eq 'fsr3-mfg-dynamic.json') { $psargs += @('-Vblank', '840') }  # (10-21: 60 = 14 pulsos, 56 = 15)
	elseif ($config -ne 'sin-fg.json') { $psargs += @('-Vblank', '360') }
	# Sin frame generation, "FPS desbloqueados": vblank de 120 Hz (-Fps120, el modo probado de chenxiao07; a 240 los menus y el intro iban acelerados, 10-18), el juego pasa de 60 (a velocidad real:
	# gameMinDeltaTime en run-windows); videos, menus y pantallas de carga siguen a 60.
	# (Desbloqueado sin frame generation: vblank virtual de 360 Hz y cada frame se muestra en cuanto esta listo
	# (-FlipWhenReady), no al siguiente tick de 120 Hz: a ~80 FPS los frames alternaban 8 y 17 ms (tirones).)
	elseif ((Is-Unlocked)) { $psargs += @('-Vblank', '960') }
	# (10-23) 900 en cualquier monitor: la altura REAL del monitor (con el escalado de Windows al 150 % la de la ventana
	# sale 960 y daba 1350 lineas).
	$realH = 0; try { $realH = [int](@(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.CurrentVerticalResolution } | Sort-Object CurrentVerticalResolution -Descending)[0].CurrentVerticalResolution) } catch { }
	if ($realH -le 0) { $realH = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height }
	foreach ($rm in $resModes) { if ($rm[0] -eq 'FSR 900p (AA)') { $rm[1] = [math]::Max(1.0, [math]::Round($realH / 900.0, 4)) } }
	$res    = $script:resShown[[math]::Max(0, $cmbRes.SelectedIndex)]
	if ($res[1] -eq 0) { $res = $resModes[0] }
	# (10-22) Modo 4K: el modo cinematico del juego a 4K nativo (sin reescalado).
	# (10-22) El modo del juego siempre segun la salida: 4K = cinematico, lo demas = rendimiento (el juego guarda el
	# ultimo en la partida y sin esto se quedaba en cinematico).
	$psargs += @('-ForceMode', $(if ($chk4K.Checked -and -not (Is-FixedFg)) { 'cinematic' } else { 'performance' }))  # (10-22) modos Souls: siempre rendimiento
	# 1080p / 720p sin reescalador: el juego dibuja a esa altura.
	if ($outH -lt 1440 -and -not $res[2]) { $res = @("${outH}p", $outH, '', 0) }
	# DLSS (reescalado) carga Streamline de NVIDIA; con la generacion de frames de AMD el emulador se cerraba al crear
	# el dispositivo grafico. Con AMD: FSR o XeSS para reescalar.
	if ($res[2] -like '*dlss' -and $config -like 'fsr3*') {
		# (10-23) Prueba: DLSS + frame generation de AMD permitido (antes se cerraba al crear el dispositivo)
	}
	# DLSS, XeSS and FSR go through the game's PS5 Pro path (its output 4K): the ratio to the monitor's height, the
	# emulator detecting it (KYTY_UPSCALE_RATIO); the others their render height.
	if ($res[2] -in 'dlss', 'xess', 'fsr') {
		$psargs += @('-UpscaleRatio', ([double]$res[1]).ToString([System.Globalization.CultureInfo]::InvariantCulture))
	} elseif ($res[1] -ne 1440) { $psargs += @('-RenderHeight', "$($res[1])") }
	# "sin TAA": el TAA del juego no se ejecuta; DLSS (DLAA) en su lugar, o (taa-up-*) el reescalador elegido toma la
	# imagen sin filtrar y hace el antialiasing el solo (encima del TAA salia como una acuarela, con estelas).
	if ($res[2] -like 'taa-up-*') { $psargs += @('-Upscale', $res[2].Substring(7), '-TaaReplace', 'upscaler') }
	elseif ($res[2] -like 'taa-*') { $psargs += @('-TaaReplace', $res[2].Substring(4)) }
	elseif ($res[2]) { $psargs += @('-Upscale', $res[2]) }
	if ($res[3] -gt 0) { $psargs += @('-UpscaleHeight', "$($res[3])") }
	$preset = $dlssPresets[[math]::Max(0, $cmbPreset.SelectedIndex)][1]
	if ($res[2] -like '*dlss' -and $preset) { $psargs += @('-DlssPreset', $preset) }
	if ($chkVram8.Checked) { $env:KYTY_SIMULATE_VRAM_MB = '8192' } else { Remove-Item Env:KYTY_SIMULATE_VRAM_MB -ErrorAction SilentlyContinue }
	# Cinematico a 60 quitado (no funcionaba): el modo cinematico queda como en PS5.
	$env:KYTY_UNLOCK_30 = '0'
	# (10-23) Mando remapeado (KYTY_PAD_MAP) y teclado y raton (KYTY_KBM): ver "Mapear mando". Sin nada, como siempre.
	foreach ($n in 'KYTY_PAD_MAP', 'KYTY_KBM', 'KYTY_KBM_MAP', 'KYTY_KBM_SENS', 'KYTY_KBM_ANTIDZ', 'KYTY_KBM_CALIBRATE') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
	$padMap = Get-IniValue 'padmap' ''; if ($padMap) { $env:KYTY_PAD_MAP = $padMap }
	# (10-23) "Calibrar camara": el emulador mide cuanto gira la camara del juego con cada posicion del stick y lo guarda
	# en kbm_curve.txt junto al exe; desde entonces el raton gira la camara exactamente lo que se mueve.
	if ($script:kbmCalibrate) { $env:KYTY_KBM_CALIBRATE = '1' }
	if ($chkKbm.Checked -or $script:kbmCalibrate) {
		$env:KYTY_KBM = '1'
		$kbmMap = Get-IniValue 'kbmmap' ''; if ($kbmMap) { $env:KYTY_KBM_MAP = $kbmMap }
		$env:KYTY_KBM_SENS   = ($trkSens.Value / 100).ToString([System.Globalization.CultureInfo]::InvariantCulture)
		$env:KYTY_KBM_ANTIDZ = ($trkAdz.Value / 100).ToString([System.Globalization.CultureInfo]::InvariantCulture)
	}
	foreach ($n in 'KYTY_VSYNC_LOCK', 'KYTY_FPS_CAP', 'KYTY_FLIP_WHEN_READY') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
	# G-Sync: reloj del emulador suelto del monitor. Mostrar cada frame al momento solo con FPS desbloqueados y sin
	# frame generation; con el juego a 60 (o con frame generation) el ritmo de 60 lo marcan los vblanks y no se toca.
	if ($chkGsync.Checked) {
		$env:KYTY_VSYNC_LOCK = '0'
		if ((Is-Unlocked) -and $config -eq 'sin-fg.json') { $env:KYTY_FPS_CAP = 'auto'; $env:KYTY_FLIP_WHEN_READY = '1' }
	}
	Start-Process -FilePath 'powershell' -ArgumentList $psargs -WorkingDirectory $root -WindowStyle Hidden | Out-Null
	$metArgs = @(); if ($chkRam16.Checked) { $metArgs = @('-RamGB', '16') }
	if ($chkVram8.Checked) { $metArgs += @('-VramMB', '8192') }
	# Como administrador solo con el limite de RAM (lo necesita); si no, normal (no molesta a la grabacion de NVIDIA).
	$metVerb = if ($chkRam16.Checked) { @{ Verb = 'RunAs' } } else { @{} }
	if ($chkMetricas.Checked -or $chkRam16.Checked -or $chkVram8.Checked) { Start-Process -FilePath 'powershell' -ArgumentList (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', ('"' + (Join-Path $root 'metricas.ps1') + '"'), '-LogDir', ('"' + (Join-Path $root 'logs') + '"')) + $metArgs) -WindowStyle Hidden @metVerb -ErrorAction SilentlyContinue | Out-Null }
	$form.WindowState = 'Minimized'
	# (10-22) La ventana del juego al frente (con el foco) en cuanto aparece, y otra vez a los pocos segundos (por si las
	# metricas o el lanzador se quedaron delante): sin el foco, NVIDIA App cortaba la grabacion.
	$script:fgTries = 0; $script:fgOk = 0
	$script:fgTimer = New-Object System.Windows.Forms.Timer
	$script:fgTimer.Interval = 1000
	$script:fgTimer.Add_Tick({
		$script:fgTries++
		$w = Get-Process -Name 'kyty_emulator*' -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
		if ($w) {
			if ([FgWin]::GetForegroundWindow() -ne $w.MainWindowHandle) { [void][FgWin]::SetForegroundWindow($w.MainWindowHandle); $script:fgOk = 0 }
			else { $script:fgOk++; if ($script:fgOk -ge 5) { $script:fgTimer.Stop() } }
		}
		if ($script:fgTries -ge 90) { $script:fgTimer.Stop() }
	})
	$script:fgTimer.Start()
}

# ---- ventana ----
# (10-22) Pantalla completa sin bordes en la pantalla principal; la columna de ajustes pegada a la derecha.
$scr = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$W = $scr.Width; $H = $scr.Height
$form = New-Object System.Windows.Forms.Form
$form.Text            = 'Launcher'
$form.FormBorderStyle = 'None'
$form.StartPosition   = 'Manual'
$form.Bounds          = $scr
$form.MaximizeBox     = $false
# Esc cierra el lanzador (como Salir).
$form.KeyPreview = $true
$form.Add_KeyDown({ param($sender, $e) if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Escape) { $form.Close() } })
$form.BackColor       = [System.Drawing.Color]::FromArgb(14, 16, 22)
$fondo = Join-Path $root 'fondo-souls.jpg'
if (Test-Path -LiteralPath $fondo) {
	# Fondo "cover": llena la pantalla sin deformarse (se recorta lo que sobra por los lados o arriba/abajo).
	$img = [System.Drawing.Image]::FromFile($fondo)
	$scale = [math]::Max($W / $img.Width, $H / $img.Height)
	$sw = [int][math]::Round($W / $scale); $sh = [int][math]::Round($H / $scale)
	$src = New-Object System.Drawing.Rectangle([int](($img.Width - $sw) / 2), [int](($img.Height - $sh) / 2), $sw, $sh)
	$bmp = New-Object System.Drawing.Bitmap($W, $H)
	$gr = [System.Drawing.Graphics]::FromImage($bmp)
	$gr.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
	$gr.DrawImage($img, (New-Object System.Drawing.Rectangle(0, 0, $W, $H)), $src, [System.Drawing.GraphicsUnit]::Pixel)
	$gr.Dispose(); $img.Dispose()
	$form.BackgroundImage       = $bmp
	$form.BackgroundImageLayout = 'None'
}
$ico = Join-Path $root 'kyty_emulator.exe'
if (Test-Path -LiteralPath $ico) { try { $form.Icon = [System.Drawing.Icon]::ExtractAssociatedIcon($ico) } catch {} }

$titleFont  = New-Object System.Drawing.Font('Segoe UI Semibold', 40, [System.Drawing.FontStyle]::Bold)
$subFont    = New-Object System.Drawing.Font('Segoe UI', 12)
$buttonFont = New-Object System.Drawing.Font('Segoe UI Semibold', 15, [System.Drawing.FontStyle]::Bold)
$smallFont  = New-Object System.Drawing.Font('Segoe UI', 10)
$gold       = [System.Drawing.Color]::FromArgb(222, 196, 140)
$ink        = [System.Drawing.Color]::FromArgb(236, 232, 224)
$panelBack  = [System.Drawing.Color]::FromArgb(170, 8, 10, 14)
$pcFont     = New-Object System.Drawing.Font('Segoe UI Black', 34, [System.Drawing.FontStyle]::Bold)

function New-Label([string]$text, $font, $color, $x, $y, $w, $h, $align = 'MiddleLeft') {
	$l = New-Object System.Windows.Forms.Label
	$l.Text = $text; $l.Font = $font; $l.ForeColor = $color
	$l.BackColor = [System.Drawing.Color]::Transparent
	$l.Location = New-Object System.Drawing.Point($x, $y); $l.Size = New-Object System.Drawing.Size($w, $h)
	$l.TextAlign = $align
	return $l
}
function New-Button([string]$text, $x, $y, $w, $h, $primary) {
	$b = New-Object System.Windows.Forms.Button
	$b.Text = $text; $b.Font = $buttonFont
	$b.Location = New-Object System.Drawing.Point($x, $y); $b.Size = New-Object System.Drawing.Size($w, $h)
	$b.FlatStyle = 'Flat'; $b.FlatAppearance.BorderSize = 1
	$b.Cursor = [System.Windows.Forms.Cursors]::Hand
	if ($primary) {
		$b.BackColor = [System.Drawing.Color]::FromArgb(230, 176, 140, 70); $b.ForeColor = [System.Drawing.Color]::FromArgb(20, 16, 10)
		$b.FlatAppearance.BorderColor = $gold
		$b.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(240, 206, 170, 96)
	} else {
		$b.BackColor = [System.Drawing.Color]::FromArgb(200, 24, 28, 38); $b.ForeColor = $ink
		$b.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(110, 110, 100)
		$b.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(220, 44, 50, 64)
	}
	return $b
}

# ("PC" va pintado en la propia imagen de fondo.)

# Barra inferior semitransparente: Play, la casilla del frame generation, Settings y Salir.
$panel = New-Object System.Windows.Forms.Panel
$panelW = 880  # (10-23) dos columnas
$panel.Location  = New-Object System.Drawing.Point(($W - $panelW), 0)
$panel.Size      = New-Object System.Drawing.Size($panelW, $H)
$panel.AutoScroll = $true
$panel.Anchor = 'Top, Bottom, Right'
$panel.BackColor = $panelBack
$form.Controls.Add($panel)

$btnPlay = New-Button 'PLAY' 50 38 260 74 $true
$btnPlay.Font = New-Object System.Drawing.Font('Segoe UI Black', 22, [System.Drawing.FontStyle]::Bold)
$btnPlay.Add_Click({ Start-Game })
$panel.Controls.Add($btnPlay)

$cmbFg = New-Object System.Windows.Forms.ComboBox
$cmbFg.DropDownStyle = 'DropDownList'
$cmbFg.Font      = $subFont
$cmbFg.Location  = New-Object System.Drawing.Point(350, 56)
$cmbFg.Size      = New-Object System.Drawing.Size(300, 34)
foreach ($m in $fgModes) { [void]$cmbFg.Items.Add($m[0]) }
$cmbFg.DrawMode = 'OwnerDrawFixed'
$cmbFg.ItemHeight = 24
$cmbFg.BackColor = [System.Drawing.Color]::FromArgb(24, 26, 34)
$cmbFg.ForeColor = $ink
$cmbFg.SelectedIndex = [math]::Min([math]::Max(0, (Get-FgIndex)), $fgModes.Count - 1)
if (-not $fgModes[$cmbFg.SelectedIndex][1]) { $cmbFg.SelectedIndex = [math]::Min($cmbFg.SelectedIndex + 1, $fgModes.Count - 1) }
$cmbFg.Add_SelectedIndexChanged({
	# Section titles are not modes: move to the first mode under them.
	if ($cmbFg.SelectedIndex -ge 0 -and -not $fgModes[$cmbFg.SelectedIndex][1]) {
		$cmbFg.SelectedIndex = [math]::Min($cmbFg.SelectedIndex + 1, $fgModes.Count - 1)
		return
	}
	if (-not $script:fgNoSave) { Set-FgIndex $cmbFg.SelectedIndex }
})
$panel.Controls.Add($cmbFg)

function New-Check([string]$text, $x, $y, $w, [string]$key) {
	$c = New-Object System.Windows.Forms.CheckBox
	$c.Text = $text; $c.Font = $smallFont; $c.ForeColor = $ink
	$c.BackColor = [System.Drawing.Color]::Transparent
	$c.Location = New-Object System.Drawing.Point($x, $y); $c.Size = New-Object System.Drawing.Size($w, 26)
	$c.Checked = (Get-IniValue $key '0') -eq '1'
	$c.Add_CheckedChanged({ param($sender, $e) Set-IniValue $sender.Tag $(if ($sender.Checked) { '1' } else { '0' }) })
	$c.Tag = $key
	return $c
}
$chkHdr   = New-Check 'HDR' 350 100 90 'hdr'
$chkVsync = New-Check 'VSync' 450 100 100 'vsync'
$chkLowPc = New-Check 'Modo TURBO (16 GB)' 670 14 230 'lowpc'
$chkUnlock = New-Check 'FPS desbloqueados' 910 14 180 'unlock'
$chkTurbo2 = New-Check 'TURBO (VRAM 8 GB o menos)' 670 14 230 'turbo2'
$chkPc16 = New-Check 'Equipo de 16 GB RAM' 670 14 230 'pc16'
$chkRamOpt = New-Check '16 GB de RAM optimizacion' 0 0 396 'ramopt'
$chkLowEnd = New-Check 'Modo Souls ligero' 910 44 220 'lowend'
$chkLowEnd45 = New-Check 'Modo Souls equilibrado' 0 0 260 'lowend45'
$chkAntiLag = New-Check 'Anti-Lag' 500 4 140 'antilag'
$chkExcl = New-Check 'Pantalla exclusiva' 50 30 180 'exclusive'
$chkPresetC = New-Check 'DLSS FG Preset C' 1100 44 200 'presetc'
$chkSafeGpu = New-Check 'Modo GPU seguro' 1100 14 170 'safegpu'
$tips = New-Object System.Windows.Forms.ToolTip
$tips.SetToolTip($chkHdr, 'HDR10: activa antes "Usar HDR" en la configuracion de pantalla de Windows')
$tips.SetToolTip($chkVsync, 'VSync: sin cortes de imagen (tearing); con FreeSync/G-Sync sigue funcionando')
$panel.Controls.Add($chkHdr)
$panel.Controls.Add($chkVsync)
$chkGsync = New-Check 'G-Sync' 50 116 120 'gsync'
$tips.SetToolTip($chkGsync, 'G-Sync / FreeSync: el emulador no ata su reloj a los Hz del monitor (usalo con FPS desbloqueados)')
$panel.Controls.Add($chkGsync)
# (10-21: quitada del lanzador; se deja como estaba en uso, activada.)
$chkGsync.Checked = $true; $chkGsync.Visible = $false
$chkMetricas = New-Check 'Metricas' 180 116 130 'metricas'
$tips.SetToolTip($chkMetricas, 'Muestra encima del juego GPU, VRAM, CPU, RAM, FPS y la grafica del frametime')
$panel.Controls.Add($chkMetricas)
$chkAf16 = New-Check 'Filtrado x16' 320 116 150 'af16'
$tips.SetToolTip($chkAf16, 'Filtrado anisotropico x16 en las texturas: suelos y paredes nitidos a lo lejos y en angulo. Cuesta muy poco.')
$panel.Controls.Add($chkAf16)
$chkLod = New-Check 'Texturas nitidas' 0 0 190 'lodbias'
$tips.SetToolTip($chkLod, 'Con reescalado (DLSS, XeSS, FSR): las texturas usan el detalle de la resolucion final. Mas nitidas, cuesta pocos FPS. Puede brillar algo en rejas.')
$panel.Controls.Add($chkLod)
$chkRam16 = New-Check 'Simular 16 GB RAM' 50 4 200 'ram16sim'
$tips.SetToolTip($chkRam16, 'Solo el juego: como mucho 12 GB en RAM (4 GB para Windows), el resto a disco, como un PC de 16 GB. Abre las metricas.')
$panel.Controls.Add($chkRam16)
$chkVram8 = New-Check 'Simular grafica 8 GB' 260 4 220 'vram8sim'
$tips.SetToolTip($chkVram8, 'El juego solo puede usar 8 GB de VRAM, como una grafica de 8 GB. Abre las metricas.')
$panel.Controls.Add($chkVram8)
$tips.SetToolTip($chkExcl, 'Pantalla completa de toda la vida (exclusiva) en vez de sin bordes: puede ir mejor con G-Sync')
$panel.Controls.Add($chkExcl)
# (10-21: quitada; no aportaba nada.)
$chkExcl.Checked = $false; $chkExcl.Visible = $false
$tips.SetToolTip($chkPresetC, 'Frame generation de NVIDIA con el Preset C: hasta un 35% mas rapido, algo menos de calidad')
$panel.Controls.Add($chkPresetC)
$chkPresetC.Visible = $false
# (10-21: ocultas hasta arreglar la imagen negra con DLSS-G: pantalla exclusiva y Preset C.)
$tips.SetToolTip($chkLowPc, 'PCs con 16 GB de RAM o menos o graficas de 8 GB o menos: unos 6 GB de VRAM como maximo, RAM devuelta a Windows cada 15 s, texturas a la mitad y sombras mas cortas. Para que no se cierre en zonas pesadas.')
# (Un solo TURBO desde 10-20: sin vaciado de RAM los dos solo se diferenciaban en la VRAM.)
$chkLowPc.Checked = $false
$tips.SetToolTip($chkTurbo2, 'Graficas de 8 GB o menos: el juego en unos 5.5-6 GB de VRAM (3D a 1080 como maximo, texturas a un cuarto, detalle bajo). Se ve peor, pero no se queda sin memoria de video.')
$panel.Controls.Add($chkTurbo2)
# (10-21) TURBO quitado (se veia raro); en su lugar un modo solo de RAM para equipos de 16 GB.
$chkTurbo2.Checked = $false; $chkTurbo2.Visible = $false
$tips.SetToolTip($chkPc16, 'PCs con 16 GB de RAM: el emulador guarda menos en RAM (6 GB fijos) y precarga menos. No cambia los graficos.')
$panel.Controls.Add($chkPc16)
$tips.SetToolTip($chkRamOpt, 'Experimental: el emulador solo reserva la memoria que el juego usa de verdad (unos 3 GB menos). Si da problemas, quitala.')
$panel.Controls.Add($chkRamOpt)
$tips.SetToolTip($chkLowEnd, 'Adaptado para graficas de 8 GB como RTX 3060, RTX 3060 Ti, RTX 3070, RTX 2070/2080, RX 6600, RX 6600 XT o RX 6650 XT: el juego a 30 FPS estables y frame generation de AMD x2 (60 en pantalla). A 1080p, con el antialiasing de XeSS o de FSR (sin reescalar).')
$panel.Controls.Add($chkLowEnd)
$tips.SetToolTip($chkLowEnd45, 'Para graficas de la potencia de una RTX 3080: RTX 2080 Ti, RTX 3070 Ti, RTX 3080, RTX 3080 Ti, RTX 4060 Ti, RTX 4070, RX 6800, RX 6800 XT, RX 6900 XT, RX 6950 XT, RX 7700 XT, RX 7800 XT o Intel Arc B580: el juego a 45 FPS estables y frame generation de AMD x2 (90 en pantalla). A 1440p, con el reescalado que elijas.')
$panel.Controls.Add($chkLowEnd45)
# (10-22) Solo uno de los dos modos a la vez (si ya estaban los dos en el ini, gana el de 30).
if ($chkLowEnd.Checked -and $chkLowEnd45.Checked) { $chkLowEnd45.Checked = $false }
$tips.SetToolTip($chkAntiLag, 'Menos retraso en los controles y el raton: AMD Anti-Lag 2 en graficas AMD y Reflex Boost con DLSS Frame Generation.')
$panel.Controls.Add($chkAntiLag)
$chk4K = New-Check 'Modo 4K' 0 0 120 'modo4k'
$tips.SetToolTip($chk4K, 'El juego arranca en su modo cinematico a 4K (como en PS5). Con un reescalador, su resolucion se calcula para 4K.')
$panel.Controls.Add($chk4K)
$chk4K.Add_CheckedChanged({ Fill-Res })
$tips.SetToolTip($chkUnlock, 'El juego pasa de 60 FPS reales, tambien con frame generation (videos, menus y cargas siguen a 60)')
# (Oculta otra vez, 10-19: a 120 Hz tirones (frames de 8 y 17 ms alternos a ~80 FPS); a 360 Hz con flip al estar listo,
# fluido pero una cinematica del motor iba acelerada: el juego ata algunas cosas a los frames.)
$chkUnlock.Checked = $false
$panel.Controls.Add($chkUnlock)
$tips.SetToolTip($chkSafeGpu, 'Para graficas que se cuelgan o dan pantallazo azul: mas estable, ~37% menos FPS')
$panel.Controls.Add($chkSafeGpu)

# Desplegables con color por marca: titulos de apartado en negrita (NVIDIA verde, Intel azul, AMD rojo) y sus modos
# en el mismo tono.
$boldSmall = New-Object System.Drawing.Font('Segoe UI Semibold', 10, [System.Drawing.FontStyle]::Bold)
function Get-BrandColor([string]$text) {
	if ($text -match 'NVIDIA|DLSS')      { return [System.Drawing.Color]::FromArgb(118, 185, 0) }
	if ($text -match 'Intel|XeSS')       { return [System.Drawing.Color]::FromArgb(70, 160, 255) }
	if ($text -match 'AMD|FSR')          { return [System.Drawing.Color]::FromArgb(237, 70, 70) }
	if ($text -match 'juego|Del juego')  { return [System.Drawing.Color]::FromArgb(170, 170, 170) }
	return $ink
}
function Draw-ColoredItem($combo, $e) {
	if ($e.Index -lt 0) { return }
	$text   = [string]$combo.Items[$e.Index]
	$header = $text.StartsWith('---')
	$selected = ((([int]$e.State) -band [int][System.Windows.Forms.DrawItemState]::Selected) -ne 0) -and -not $header
	$back = if ($selected) { [System.Drawing.Color]::FromArgb(60, 64, 80) } elseif ($header) { [System.Drawing.Color]::FromArgb(14, 16, 22) } else { [System.Drawing.Color]::FromArgb(24, 26, 34) }
	$brush = New-Object System.Drawing.SolidBrush($back)
	$e.Graphics.FillRectangle($brush, $e.Bounds)
	$brush.Dispose()
	$font  = if ($header) { $boldSmall } else { $combo.Font }
	$label = if ($header) { $text.Trim('-', ' ') } else { $text.Trim() }
	$color = Get-BrandColor $text
	$x = if ($header) { 4 } else { 12 }
	$textBrush = New-Object System.Drawing.SolidBrush($color)
	$e.Graphics.DrawString($label, $font, $textBrush, [single]($e.Bounds.X + $x), [single]($e.Bounds.Y + 2))
	$textBrush.Dispose()
}

# Reescalado por apartados (NVIDIA DLSS, Intel XeSS, AMD FSR, el juego), sin mezclar con la generacion de frames:
# con la de NVIDIA solo DLSS; con la de AMD, FSR y XeSS; sin generacion, todos.
$script:resShown = @()
function Get-ResFamilies {
	# (10-22) Modo graficas de 8 GB: frame generation de AMD, asi que XeSS y FSR (aunque la lista tenga uno de NVIDIA).
	if ($chkLowEnd -and $chkLowEnd.Checked) { return @('dlss', 'xess', 'fsr') }  # (10-23) DLSS tambien (va con el FG de AMD)
	# (10-22) Modo Souls equilibrado: frame generation de AMD, asi que sin DLSS (XeSS, FSR o el del juego).
	if ($chkLowEnd45 -and $chkLowEnd45.Checked) { return @('', 'dlss', 'xess', 'fsr', 'juego') }  # (10-23) DLSS tambien
	$cfg = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	if ($cfg -like 'dlss*') { return @('', 'dlss', 'juego') }
	if ($cfg -like 'fsr3*') { return @('', 'dlss', 'xess', 'fsr', 'juego') }  # (10-23) DLSS + frame generation de AMD (prueba)
	return @('', 'dlss', 'xess', 'fsr', 'juego')
}
function Get-OutHeight {
	# (10-22) Modo graficas de 8 GB: siempre 1080p.
	if ($chkLowEnd -and $chkLowEnd.Checked) { return 1080 }
	# (10-22) Modo 45 FPS: siempre 1440p.
	if ($chkLowEnd45 -and $chkLowEnd45.Checked) { return 1440 }
	if ($cmbOut) { switch ($cmbOut.SelectedIndex) { 1 { return 2160 } 2 { return 1080 } 3 { return 720 } } }
	if ($chk4K -and $chk4K.Checked) { return 2160 }
	return 1440
}
function Fill-Res {
	$wanted = Get-IniValue 'scalelabel' '1440p nativo'
	$titles = @{ 'dlss' = '--- NVIDIA DLSS ---'; 'xess' = '--- Intel XeSS ---'; 'fsr' = '--- AMD FSR 3.1 ---'; 'juego' = '--- Del juego ---' }
	$shown  = @()
	foreach ($family in (Get-ResFamilies)) {
		$modes = @($resModes | Where-Object { $(if ($_[2] -eq '' -and $_[1] -ne 1440) { 'juego' } elseif ($_[2] -like 'taa-*') { $_[2] -replace '^taa-(up-)?', '' } else { $_[2] }) -eq $family })
		if ($family -eq '') { $modes = @($resModes | Where-Object { $_[0] -eq '1440p nativo' }) }
		# Con TURBO (1080p) solo el antialiasing de cada uno a resolucion nativa (DLAA, XeSS AA, FSR AA): reescalar desde
		# menos de 1080 se veia peor.
		# (10-21) TURBO ya no limita la resolucion ni los modos de reescalado.
		# PC bajos requisitos: sin DLSS (usa el frame generation de AMD x2; solo XeSS, FSR o nativo).
		# (10-22) DLSS Equilibrado y Ultra rendimiento solo a 4K: sobre 1440 caen por debajo de lo que acepta DLSS.
		# Modo graficas de 8 GB: frame generation de AMD, asi que sin DLSS (todos los de XeSS y FSR).
		if ($family -eq 'dlss' -and -not $script:isNvidia) { continue }  # (10-23) sin NVIDIA no hay DLSS
		$modes = @($modes | Where-Object { $_[0] -ne 'FSR 900p (AA)' -or ($chkSoulsLow -and $chkSoulsLow.Checked) })  # (10-23) solo en low profile
		# (10-23) Modo Souls ligero: solo el antialiasing de cada uno (DLSS DLAA, XeSS AA, FSR AA) o nativo.
		if ($chkLowEnd -and $chkLowEnd.Checked) { $modes = @($modes | Where-Object { $_[0] -in 'DLSS DLAA', 'XeSS AA', 'FSR AA nativo' }) }
		# (10-22) Modo graficas de 8 GB: tambien sin reescalado (1080p nativo, sin XeSS ni FSR), lo primero de la lista.
		if ($chkLowEnd -and $chkLowEnd.Checked -and $family -eq 'xess') { $shown += , @('Sin reescalado (1080p nativo)', 1080, '', 0) }
		$outH = Get-OutHeight
		if ($family -eq 'dlss') { $modes = @($modes | Where-Object { $h = [math]::Round($outH / $_[1]); $h -ge 1080 -or $h -eq 720 }) }
		elseif ($family -in 'xess', 'fsr' -and $outH -lt 1440) { $modes = @($modes | Where-Object { $outH / $_[1] -ge 719 }) }
		if ($family -eq 'juego' -and $outH -lt 1440) { $modes = @() }
		if ($modes.Count -eq 0) { continue }
		if ($titles.ContainsKey($family)) { $shown += , @($titles[$family], 0, '', 0) }
		foreach ($m in $modes) { $shown += , $m }
	}
	if ($shown.Count -eq 0) { $shown = @(, $resModes[0]) }
	$script:resShown = $shown
	$script:resFilling = $true
	$cmbRes.Items.Clear()
	foreach ($m in $shown) { [void]$cmbRes.Items.Add($m[0]) }
	$index = 0
	for ($i = 0; $i -lt $shown.Count; $i++) { if ($shown[$i][0] -eq $wanted) { $index = $i } }
	$cmbRes.SelectedIndex = $index
	$script:resFilling = $false
}
$cmbRes = New-Object System.Windows.Forms.ComboBox
$cmbRes.DropDownStyle = 'DropDownList'
$cmbRes.Font     = $smallFont
$cmbRes.Location = New-Object System.Drawing.Point(555, 100)
$cmbRes.Size     = New-Object System.Drawing.Size(110, 26)
$cmbRes.DropDownWidth = 230
$cmbRes.DrawMode = 'OwnerDrawFixed'
$cmbRes.ItemHeight = 20
$cmbRes.BackColor = [System.Drawing.Color]::FromArgb(24, 26, 34)
$cmbRes.ForeColor = $ink
$cmbRes.Add_DrawItem({ param($sender, $e) Draw-ColoredItem $sender $e })
$cmbRes.Add_SelectedIndexChanged({
	if ($script:resFilling -or $cmbRes.SelectedIndex -lt 0) { return }
	# Los titulos de apartado no son modos: al siguiente.
	if ($script:resShown[$cmbRes.SelectedIndex][1] -eq 0) {
		$cmbRes.SelectedIndex = [math]::Min($cmbRes.SelectedIndex + 1, $script:resShown.Count - 1)
		return
	}
	Set-IniValue 'scalelabel' $script:resShown[$cmbRes.SelectedIndex][0]
})
$tips.SetToolTip($cmbRes, 'Reescalado: dibuja el 3D a menos resolucion y lo sube a 1440 (mas fps). DLSS: NVIDIA RTX. XeSS y FSR: todas las graficas. "juego": lo sube el propio juego.')
$panel.Controls.Add($cmbRes)
Fill-Res
$cmbFg.Add_SelectedIndexChanged({ Fill-Res })
# (10-22) Con frame generation dinamico los FPS quedan bloqueados (el juego a su base): la casilla se desmarca y no se puede marcar.
function Update-UnlockForFg {
	$cfg = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	$chkUnlock.Enabled = $true
}
$cmbFg.Add_SelectedIndexChanged({ Update-UnlockForFg })
Update-UnlockForFg
$chkTurbo2.Add_CheckedChanged({ Fill-Res })
# PC bajos requisitos: la lista de frame generation bloqueada (va con AMD x2) y sin modos DLSS.
function Update-LowEnd { $low = Is-FixedFg; $cmbFg.Enabled = -not $low
	# (10-23) Con un modo de FG fija (Souls ligero, equilibrado, low profile) el desplegable muestra AMD x2; al quitarlo vuelve
	# el que habia (sin guardarlo en launcher.ini).
	$fixIdx = -1; for ($q = 0; $q -lt $fgModes.Count; $q++) { if ($fgModes[$q][1] -eq 'fsr3.json') { $fixIdx = $q } }
	$script:fgNoSave = $true
	if ($low -and $fixIdx -ge 0) { if ($null -eq $script:fgBefore) { $script:fgBefore = $cmbFg.SelectedIndex }; $cmbFg.SelectedIndex = $fixIdx }
	elseif (-not $low -and $null -ne $script:fgBefore) { $cmbFg.SelectedIndex = $script:fgBefore; $script:fgBefore = $null }
	$script:fgNoSave = $false
	; if ($cmbOut) { if ($chkLowEnd45.Checked) { $cmbOut.SelectedIndex = 0 }; $cmbOut.Enabled = -not $low; $chk4K.Checked = (-not $low) -and $cmbOut.SelectedIndex -eq 1 }; if ($cmbPreset) { $cmbPreset.Enabled = $script:isNvidia }; Fill-Res }
$chkLowEnd.Add_CheckedChanged({ if ($chkLowEnd.Checked) { $chkLowEnd45.Checked = $false }; Update-LowEnd })
$chkLowEnd45.Add_CheckedChanged({ if ($chkLowEnd45.Checked) { $chkLowEnd.Checked = $false }; Update-LowEnd })
$cmbFg.Enabled = -not (Is-FixedFg)
$cmbFg.Add_DrawItem({ param($sender, $e) Draw-ColoredItem $sender $e })

$cmbPreset = New-Object System.Windows.Forms.ComboBox
$cmbPreset.DropDownStyle = 'DropDownList'
$cmbPreset.Font     = $smallFont
$cmbPreset.Location = New-Object System.Drawing.Point(($W - 610), 110)
$cmbPreset.Size     = New-Object System.Drawing.Size(150, 26)
foreach ($m in $dlssPresets) { [void]$cmbPreset.Items.Add($m[0]) }
$presetIndex = 0
[void][int]::TryParse((Get-IniValue 'dlsspreset' '0'), [ref]$presetIndex)
$cmbPreset.SelectedIndex = [math]::Min([math]::Max(0, $presetIndex), $dlssPresets.Count - 1)
$cmbPreset.Add_SelectedIndexChanged({ Set-IniValue 'dlsspreset' "$($cmbPreset.SelectedIndex)" })
$tips.SetToolTip($cmbPreset, 'Modelo de DLSS: auto (el de NVIDIA), K = DLSS 4, L y M = DLSS 4.5 (mas nitidos, algo mas lentos)')
$panel.Controls.Add($cmbPreset)
$cmbPreset.Enabled = $script:isNvidia  # (10-23) modos Souls: tambien con DLSS (solo NVIDIA)

# Nitidez del reescalado: Auto (FSR 0.3, DLSS y XeSS sin nada) o un nivel fijo para todos.
$sharpLevels = @(@('Nitidez: auto', ''), @('Nitidez: 0', '0'), @('Nitidez: 25%', '0.25'), @('Nitidez: 50%', '0.5'),
                 @('Nitidez: 75%', '0.75'), @('Nitidez: 100%', '1'))
$cmbSharp = New-Object System.Windows.Forms.ComboBox
$cmbSharp.DropDownStyle = 'DropDownList'
$cmbSharp.Font     = $smallFont
$cmbSharp.Location = New-Object System.Drawing.Point(($W - 450), 110)
$cmbSharp.Size     = New-Object System.Drawing.Size(140, 26)
foreach ($m in $sharpLevels) { [void]$cmbSharp.Items.Add($m[0]) }
$sharpIndex = 0
[void][int]::TryParse((Get-IniValue 'sharpness' '0'), [ref]$sharpIndex)
$cmbSharp.SelectedIndex = [math]::Min([math]::Max(0, $sharpIndex), $sharpLevels.Count - 1)
$cmbSharp.Add_SelectedIndexChanged({ Set-IniValue 'sharpness' "$($cmbSharp.SelectedIndex)" })
$tips.SetToolTip($cmbSharp, 'Nitidez del reescalado (DLSS, XeSS y FSR). Mas alto: mas nitido, pero puede marcar bordes y ruido.')
$panel.Controls.Add($cmbSharp)

# (10-21) Graficos tipo PC: Maximo = el juego tal cual (PS5); nunca baja la resolucion ni toca las sombras.
$gfxLevels = @('Maximo', 'Alto', 'Medio', 'Bajo')
# (10-23) Nivel 'Alto' nuevo: los indices guardados de antes (1 Medio, 2 Bajo) pasan a 2 y 3 una sola vez.
if ((Get-IniValue 'gfxlv' '3') -ne '4') { foreach ($k in 'gfxtex', 'gfxdist', 'gfxshadow') { $o = 0; [void][int]::TryParse((Get-IniValue $k '0'), [ref]$o); if ($o -gt 0) { Set-IniValue $k "$($o + 1)" } }; Set-IniValue 'gfxlv' '4' }
$script:gfxTop = @()
function New-GfxCombo([string]$label, [string]$key, $x, [string]$tip) {
	$lb = New-Label $label $smallFont $ink $x 6 90 22; $script:gfxTop += $lb
	$panel.Controls.Add($lb)
	$c = New-Object System.Windows.Forms.ComboBox
	$c.DropDownStyle = 'DropDownList'; $c.Font = $smallFont
	$c.Location = New-Object System.Drawing.Point(($x + 92), 2); $c.Size = New-Object System.Drawing.Size(100, 26)
	foreach ($l in $gfxLevels) { [void]$c.Items.Add($l) }
	$i = 0; [void][int]::TryParse((Get-IniValue $key '0'), [ref]$i)
	$c.SelectedIndex = [math]::Min([math]::Max(0, $i), 3)
	$c.Tag = $key
	$c.Add_SelectedIndexChanged({ param($sender, $e) Set-IniValue $sender.Tag "$($sender.SelectedIndex)" })
	$tips.SetToolTip($c, $tip)
	$script:gfxTop += $c
	$panel.Controls.Add($c)
	return $c
}
$cmbTex  = New-GfxCombo 'Texturas' 'gfxtex' 290 'Texturas: Medio = mitad de tamano, Bajo = un cuarto. Lo que mas VRAM ahorra.'
$cmbShadow = New-GfxCombo 'Sombras' 'gfxshadow' 710 'Sombras: Medio = las del sol hasta 125 m, Bajo = hasta 60 m y sin sombras de focos ni ambientales. Mas FPS, pero pueden aparecer de golpe.'
$cmbDist = New-GfxCombo 'Distancia' 'gfxdist' 500 'Distancia de dibujado: a menos nivel, los objetos lejanos pasan antes a modelos mas simples. Mas FPS si va justo el procesador.'

# Precompilar los shaders del juego (todos, de los archivos del juego): la primera vez puede tardar horas.
$btnPre = New-Button 'Precompilar shaders' ($W - 610) 46 230 58 $false
$btnPre.Add_Click({
	if (-not (Get-GamePath) -or -not (Test-Path -LiteralPath (Get-GamePath))) {
		[System.Windows.Forms.MessageBox]::Show('Pon la carpeta del juego en Settings.', 'Launcher', 'OK', 'Warning') | Out-Null
		return
	}
	$pre = Join-Path $root 'precompile-windows.ps1'
	if (-not (Test-Path -LiteralPath $pre)) { [System.Windows.Forms.MessageBox]::Show('Falta precompile-windows.ps1', 'Launcher', 'OK', 'Error') | Out-Null; return }
	$a = @('-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$pre`"", '-Game', "`"$(Get-GamePath)`"")
	# Suave: la mitad de la CPU a prioridad minima (PCs que se reiniciaban o se calentaban precompilando).
	$suave = [System.Windows.Forms.MessageBox]::Show("Precompilar en modo SUAVE?`n`nSi: usa la mitad de la CPU, tarda mas y el PC no se calienta ni se reinicia.`nNo: a toda potencia (lo normal).", 'Precompilar shaders', 'YesNoCancel', 'Question')
	if ($suave -eq 'Cancel') { return }
	if ($suave -eq 'Yes') { $a += '-Gentle' }
	Start-Process -FilePath 'powershell' -ArgumentList $a -WorkingDirectory $root | Out-Null
})
$panel.Controls.Add($btnPre)

$btnSettings = New-Button 'Settings' ($W - 365) 46 160 58 $false
$panel.Controls.Add($btnSettings)

$btnExit = New-Button 'Salir' ($W - 190) 46 140 58 $false
$btnExit.Add_Click({ $form.Close() })
# (10-23) Cruz arriba a la derecha para cerrar el launcher.
$btnX = New-Object System.Windows.Forms.Button
$btnX.Text = [string][char]0x2715; $btnX.Font = New-Object System.Drawing.Font('Segoe UI Symbol', 14, [System.Drawing.FontStyle]::Bold)
$btnX.FlatStyle = 'Flat'; $btnX.FlatAppearance.BorderSize = 0
$btnX.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22); $btnX.ForeColor = [System.Drawing.Color]::FromArgb(222, 196, 140)
$btnX.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(180, 40, 40)
$btnX.Size = New-Object System.Drawing.Size(48, 40); $btnX.Location = New-Object System.Drawing.Point(($W - $panelW - 56), 8)  # a la izquierda del panel: no tapa nada
$btnX.Add_Click({ $form.Close() })
$form.Controls.Add($btnX); $btnX.BringToFront()
# (10-23) Abajo: la grafica y el equipo detectados.
$lbPc = New-Object System.Windows.Forms.Label
$lbPc.AutoSize = $false; $lbPc.TextAlign = 'MiddleLeft'
$lbPc.Font = $smallFont; $lbPc.ForeColor = [System.Drawing.Color]::FromArgb(222, 196, 140)
$lbPc.BackColor = [System.Drawing.Color]::FromArgb(170, 14, 16, 22)
$lbPc.Text = "  Grafica: $(if ($script:gpuName) { $script:gpuName } else { 'no detectada' })$(if (-not $script:isNvidia) { '  (sin DLSS: solo NVIDIA)' })     Equipo: $($script:pcInfo)" +
	$(if ($script:gpuRec) { "`n  Recomendado: $($script:gpuRec[1]).  Esperado (estimacion): $($script:gpuRec[2])" } else { "`n  Grafica sin datos: empieza por el Modo Souls equilibrado y mira las metricas." })
$lbPc.Location = New-Object System.Drawing.Point(16, ($H - 60)); $lbPc.Size = New-Object System.Drawing.Size(($W - $panelW - 32), 48)
$form.Controls.Add($lbPc); $lbPc.BringToFront()
if ($cmbPreset -and -not $script:isNvidia) { $cmbPreset.Enabled = $false }
$panel.Controls.Add($btnExit)
$chk4K.Visible = $false
# (10-22) Salida: 1440p (el monitor), 4K (modo cinematico del juego), 1080p y 720p (graficas mas modestas). Los
# reescaladores calculan su resolucion sobre esa altura.
$lbOut = New-Label 'Salida' $smallFont $ink 925 6 60 22; $panel.Controls.Add($lbOut)
$cmbOut = New-Object System.Windows.Forms.ComboBox
$cmbOut.DropDownStyle = 'DropDownList'; $cmbOut.Font = $smallFont
$cmbOut.Location = New-Object System.Drawing.Point(985, 2); $cmbOut.Size = New-Object System.Drawing.Size(130, 26)
foreach ($l in '1440p (monitor)', '4K (cinematico)', '1080p', '720p') { [void]$cmbOut.Items.Add($l) }
$oi = 0; [void][int]::TryParse((Get-IniValue 'salida' $(if ($chk4K.Checked) { '1' } else { '0' })), [ref]$oi)
$cmbOut.SelectedIndex = [math]::Min([math]::Max(0, $oi), 3)
$chk4K.Checked = ($cmbOut.SelectedIndex -eq 1)
$cmbOut.Add_SelectedIndexChanged({ Set-IniValue 'salida' "$($cmbOut.SelectedIndex)"; $chk4K.Checked = ($cmbOut.SelectedIndex -eq 1) -and -not (Is-FixedFg); Fill-Res })
$tips.SetToolTip($cmbOut, 'Resolucion de salida. 4K: el juego arranca en su modo cinematico a 4K. 1080p y 720p: para graficas mas modestas.')
$panel.Controls.Add($cmbOut)
# (10-22) Iluminacion RT (rayos sobre la imagen: luz rebotada y sombreado de contacto). Solo con DLSS, XeSS o FSR.
$lbRt = New-Label 'Luz RT' $smallFont $ink 1125 6 60 22; $panel.Controls.Add($lbRt)
$cmbRt = New-Object System.Windows.Forms.ComboBox
$cmbRt.DropDownStyle = 'DropDownList'; $cmbRt.Font = $smallFont
$cmbRt.Location = New-Object System.Drawing.Point(1185, 2); $cmbRt.Size = New-Object System.Drawing.Size(85, 26)
foreach ($l in 'No', 'Media', 'Alta', 'Ultra') { [void]$cmbRt.Items.Add($l) }
$ri = 0; [void][int]::TryParse((Get-IniValue 'ssgi' '0'), [ref]$ri)
$cmbRt.SelectedIndex = [math]::Min([math]::Max(0, $ri), 3)
$cmbRt.Add_SelectedIndexChanged({ Set-IniValue 'ssgi' "$($cmbRt.SelectedIndex)" })
$tips.SetToolTip($cmbRt, 'Iluminacion por rayos sobre la imagen (luz rebotada y sombreado de contacto). Solo funciona con DLSS, XeSS o FSR. Cuesta FPS. Experimental.')
$panel.Controls.Add($cmbRt)
$cmbOut.Enabled = -not (Is-FixedFg)
if ($chkLowEnd45.Checked) { $cmbOut.SelectedIndex = 0 }
if (Is-FixedFg) { $chk4K.Checked = $false }

# ---- (10-22) Disposicion por apartados: cada control a su grupo (mismo nombre, clave, tooltip y eventos) ----
function New-Group([string]$text, $x, $y, $w, $h) {
	$g = New-Object System.Windows.Forms.GroupBox
	$g.Text = $text; $g.Font = $boldSmall; $g.ForeColor = $gold
	$g.BackColor = [System.Drawing.Color]::Transparent
	$g.Location = New-Object System.Drawing.Point($x, $y); $g.Size = New-Object System.Drawing.Size($w, $h)
	$panel.Controls.Add($g)
	return $g
}
function Move-To($ctl, $grp, $x, $y, $w) {
	# Colores y letra de antes (si no, los heredaria del grupo).
	$fc = $ctl.ForeColor; $fn = $ctl.Font
	$grp.Controls.Add($ctl)
	$ctl.ForeColor = $fc; $ctl.Font = $fn
	$ctl.Location = New-Object System.Drawing.Point($x, $y)
	if ($w) { $ctl.Width = $w }
}
$gImg = New-Group 'Imagen' 10 8 420 152
$gFg  = New-Group 'Frame generation' 450 8 420 94
$gGfx = New-Group 'Calidad grafica' 10 168 420 410
$gScr = New-Group 'Pantalla y otros' 450 110 420 56
$gModes = New-Group 'Modos Souls' 450 174 420 114
$gPc  = New-Group 'Rendimiento / PC' 450 294 420 90
# Imagen: salida, reescalado, preset y nitidez.
Move-To $lbOut $gImg 10 26 98
Move-To $cmbOut $gImg 110 24 296
$lbRes = New-Label 'Reescalado' $smallFont $ink 0 0 88 22; Move-To $lbRes $gImg 10 58 98
Move-To $cmbRes $gImg 110 56 296
$lbPreset = New-Label 'Preset' $smallFont $ink 0 0 88 22; Move-To $lbPreset $gImg 10 90 98
Move-To $cmbPreset $gImg 110 88 296
$lbSharp = New-Label 'Nitidez' $smallFont $ink 0 0 88 22; Move-To $lbSharp $gImg 10 122 98
Move-To $cmbSharp $gImg 110 120 296
# Frame generation: modo, FPS desbloqueados y Anti-Lag.
Move-To $cmbFg $gFg 12 24 396
Move-To $chkUnlock $gFg 12 60 200
$chkUnlock.Visible = $false  # (10-22) sin casilla: siempre desbloqueados salvo DLSS y modos Souls
Move-To $chkAntiLag $gFg 222 60 186
# (10-23 noche) Visible solo con NVIDIA: "Anti-Lag (experimental)" para la frame generation de AMD sin DLSS.
$chkAntiLag.Visible = $script:isNvidia; $chkAntiLag.Text = 'Anti-Lag (experimental)'
# Solo se puede marcar con la frame generation de AMD (con la de NVIDIA ya va Reflex); fuera de ella se desmarca.
function Update-AntiLagBox {
	$cfgNow = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	$resSel = if ($script:resShown.Count) { $script:resShown[[math]::Max(0, $cmbRes.SelectedIndex)] } else { $null }
	$ok = $cfgNow -like 'fsr3*' -and -not (Is-FixedFg) -and $resSel -and $resSel[2] -in 'xess', 'fsr'  # (10-23) solo con XeSS o FSR
	if (-not $ok) { $chkAntiLag.Checked = $false }
	$chkAntiLag.Visible = $ok -and $script:isNvidia  # (10-23) oculta si no se puede usar
}
$cmbFg.Add_SelectedIndexChanged({ Update-AntiLagBox })
$cmbRes.Add_SelectedIndexChanged({ Update-AntiLagBox })
$form.Add_Shown({ Update-AntiLagBox })
$tips.SetToolTip($chkAntiLag, 'Solo NVIDIA con la frame generation de AMD y reescalado XeSS o FSR: NVIDIA Reflex para menos retraso. Experimental: si el juego se cierra, quitala.')
# Calidad grafica: texturas, distancia, sombras, Luz RT y filtrado x16 ($script:gfxTop: etiqueta y desplegable de Texturas, Sombras, Distancia).
Move-To $script:gfxTop[0] $gGfx 10 26 98; Move-To $cmbTex $gGfx 110 24 296
Move-To $script:gfxTop[4] $gGfx 10 58 98; Move-To $cmbDist $gGfx 110 56 296
Move-To $script:gfxTop[2] $gGfx 10 90 98; Move-To $cmbShadow $gGfx 110 88 296
Move-To $lbRt $gGfx 10 122 98; Move-To $cmbRt $gGfx 110 120 296
$lbCs = New-Label 'Sombras RT' $smallFont $ink 0 0 88 22
$cmbCs = New-Object System.Windows.Forms.ComboBox
$cmbCs.DropDownStyle = 'DropDownList'; $cmbCs.Font = $smallFont
foreach ($l in 'No', 'Normal', 'Alta', 'Ultra') { [void]$cmbCs.Items.Add($l) }
$ci = 0; [void][int]::TryParse((Get-IniValue 'cshadow' '0'), [ref]$ci)
$cmbCs.SelectedIndex = [math]::Min([math]::Max(0, $ci), 3)
$cmbCs.Add_SelectedIndexChanged({ Set-IniValue 'cshadow' "$($cmbCs.SelectedIndex)" })
$tips.SetToolTip($cmbCs, 'Sombras de contacto por rayos sobre la imagen hacia el sol (donde los objetos tocan suelo y paredes). Solo con DLSS, XeSS o FSR. Cuesta FPS. Experimental.')
$lbCs.Visible = $false; $cmbCs.Visible = $false  # (10-23) Sombras RT sustituidas por las del juego en Ultra
$lbAo = New-Label 'Oclusion' $smallFont $ink 0 0 98 22
$cmbAo = New-Object System.Windows.Forms.ComboBox
$cmbAo.DropDownStyle = 'DropDownList'; $cmbAo.Font = $smallFont
foreach ($l in 'No', 'Normal', 'Alta', 'Ultra') { [void]$cmbAo.Items.Add($l) }
$iAo = 0; [void][int]::TryParse((Get-IniValue 'gtao' '0'), [ref]$iAo)
$cmbAo.SelectedIndex = [math]::Min([math]::Max(0, $iAo), 3)
$cmbAo.Add_SelectedIndexChanged({ Set-IniValue 'gtao' "$($cmbAo.SelectedIndex)" })
$tips.SetToolTip($cmbAo, 'Oclusion ambiental avanzada (GTAO): rincones, grietas y contactos mas realistas. Solo con DLSS, XeSS o FSR. Cuesta FPS. Experimental.')
Move-To $lbAo $gGfx 10 186 98; Move-To $cmbAo $gGfx 110 184 296
$lbSsr = New-Label 'Reflejos RT' $smallFont $ink 0 0 98 22
$cmbSsr = New-Object System.Windows.Forms.ComboBox
$cmbSsr.DropDownStyle = 'DropDownList'; $cmbSsr.Font = $smallFont
foreach ($l in 'No', 'Normal', 'Alta', 'Ultra') { [void]$cmbSsr.Items.Add($l) }
$iSsr = 0; [void][int]::TryParse((Get-IniValue 'ssr' '0'), [ref]$iSsr)
$cmbSsr.SelectedIndex = [math]::Min([math]::Max(0, $iSsr), 3)
$cmbSsr.Add_SelectedIndexChanged({ Set-IniValue 'ssr' "$($cmbSsr.SelectedIndex)" })
$lbSsr.Visible = $false; $cmbSsr.Visible = $false  # (10-23) Reflejos RT retirados: se usan los del juego
# (10-23) Reflections / Object Fade / AA Sharpness: ajustes del propio juego (valores de fabrica = High / Normal).
function New-GameCombo([string]$label, [string]$key, [string[]]$items, [int]$def, $y, [string]$tip) {
	$lb = New-Label $label $smallFont $ink 0 0 98 22; $panel.Controls.Add($lb)
	$c = New-Object System.Windows.Forms.ComboBox
	$c.DropDownStyle = 'DropDownList'; $c.Font = $smallFont; $c.Tag = $key
	foreach ($it in $items) { [void]$c.Items.Add($it) }
	$v = $def; [void][int]::TryParse((Get-IniValue $key "$def"), [ref]$v); $c.SelectedIndex = [math]::Min([math]::Max(0, $v), $items.Count - 1)
	$c.Add_SelectedIndexChanged({ param($sender, $e) Set-IniValue $sender.Tag "$($sender.SelectedIndex)" })
	$tips.SetToolTip($c, $tip); $panel.Controls.Add($c)
	Move-To $lb $gGfx 10 ($y + 2) 98; Move-To $c $gGfx 110 $y 296
	return $c
}
$cmbRefl = New-GameCombo 'Reflections' 'greflect' @('Ultra', 'High (default)', 'Normal', 'Low') 1 216 'Reflejos del propio juego (agua, charcos, suelos): mas pasos y mas alcance en Ultra. Cuesta algo de GPU.'
$cmbFade = New-GameCombo 'Object Fade' 'gfade' @('Ultra', 'High (default)', 'Normal', 'Low') 1 280 'Distancia a la que los objetos lejanos se desvanecen. Ultra: aparecen de golpe mucho menos.'
$cmbAaS = New-GameCombo 'AA Sharpness' 'gaasharp' @('Ultra', 'High', 'Normal (default)', 'Low', 'Off') 2 312 'Nitidez del antialiasing del juego. No cuesta FPS.'
Move-To $chkAf16 $gGfx 10 248 200
Move-To $chkLod $gGfx 216 248 190  # (10-23) vuelve: decide por textura (no toca el humo)
$lbDet = New-Label 'Detail' $smallFont $ink 0 0 88 22; $panel.Controls.Add($lbDet)
$cmbDet = New-Object System.Windows.Forms.ComboBox
$cmbDet.DropDownStyle = 'DropDownList'; $cmbDet.Font = $smallFont
foreach ($d in @('Ultra', 'Very High (cinematic)', 'High (PS5 performance)', 'Normal', 'Low')) { [void]$cmbDet.Items.Add($d) }
$iDet = 2; [void][int]::TryParse((Get-IniValue 'detail' '2'), [ref]$iDet); $cmbDet.SelectedIndex = [math]::Min([math]::Max(0, $iDet), 4)
$cmbDet.Add_SelectedIndexChanged({ Set-IniValue 'detail' "$($cmbDet.SelectedIndex)" })
$tips.SetToolTip($cmbDet, 'Detalle del juego: sombras (cascadas y distancia), iluminacion de entorno y relieve (teselado). Ultra > Very High (modo cine) > High (modo rendimiento de PS5) > Normal > Low (graficas modestas).')
$panel.Controls.Add($cmbDet)
Move-To $lbDet $gGfx 10 154 98; Move-To $cmbDet $gGfx 110 152 296
# Rendimiento / PC (chkLowPc no se muestra: sigue fuera de la ventana como antes).
Move-To $chkLowEnd $gModes 12 24 186
Move-To $chkLowEnd45 $gModes 202 24 206
Move-To $chkPc16 $gPc 12 24 190
$chkSafeGpu.Visible = $false; $chkSafeGpu.Checked = $false
Move-To $chkRam16 $gPc 12 54 190
Move-To $chkVram8 $gPc 202 54 206
# (10-23) Optimizacion tarjetas de 8 GB: texturas bajo, distancia y sombras medio (7,4 -> 6,3 GB de VRAM, tirones 26% -> 4%).
$chkGpu8 = New-Check 'Optimizacion tarjetas 8 GB' 0 0 206 'gpu8opt'
$tips.SetToolTip($chkGpu8, 'Graficas de 8 GB: pone texturas en Bajo y distancia y sombras en Medio para que el juego quepa en la memoria de la grafica.')
$panel.Controls.Add($chkGpu8)
Move-To $chkGpu8 $gPc 202 24 206
$chkGpu8.Visible = $false; $chkGpu8.Checked = $false  # (10-23) retirada: hacia lo mismo que los graficos del Modo Souls ligero
# (10-23) Presets: Souls ligero = Optimizacion 8 GB (texturas bajo, distancia y sombras medio, Luz RT no, Detail Normal,
# Oclusion Normal); Souls equilibrado (todo Maximo, Luz RT no, Detail High, Oclusion Alta). Al quitarlos: todo Maximo.
function Set-PresetLight { $cmbTex.SelectedIndex = 3; $cmbDist.SelectedIndex = 2; $cmbShadow.SelectedIndex = 2; $cmbRt.SelectedIndex = 0; $cmbDet.SelectedIndex = 3; $cmbAo.SelectedIndex = 1; $cmbAaS.SelectedIndex = 1; $cmbFade.SelectedIndex = 2; $cmbRefl.SelectedIndex = 2 }
function Set-PresetMax { $cmbTex.SelectedIndex = 0; $cmbDist.SelectedIndex = 0; $cmbShadow.SelectedIndex = 0; $cmbDet.SelectedIndex = 2; $cmbAaS.SelectedIndex = 2; $cmbFade.SelectedIndex = 1; $cmbRefl.SelectedIndex = 1 }
# (10-23) Modo Souls ultra y Modo Souls extremo (graficas potentes).
$chkSoulsUltra = New-Check 'Modo Souls ultra' 0 0 186 'soulsultra'
$tips.SetToolTip($chkSoulsUltra, 'Para RTX 5060 / 5060 Ti / 5070 / 5070 Ti y RX 9070 / 9070 XT: todo al maximo, Detail Very High, Luz RT y Oclusion en Alta. Los FPS dependen de la potencia de tu GPU y tu CPU.')
$panel.Controls.Add($chkSoulsUltra); Move-To $chkSoulsUltra $gModes 12 54 186
$chkSoulsExt = New-Check 'Modo Souls extremo' 0 0 206 'soulsext'
$tips.SetToolTip($chkSoulsExt, 'Para enfermos de Souls. RTX 5080, RTX 5090 o superior: modo cinematico del juego a 4K y absolutamente todo al maximo (Ultra en todo, filtrado x16 y texturas nitidas). Los FPS dependen de la potencia de tu GPU y tu CPU.')
$panel.Controls.Add($chkSoulsExt); Move-To $chkSoulsExt $gModes 202 54 206
function Set-PresetUltra { $cmbTex.SelectedIndex = 0; $cmbDist.SelectedIndex = 0; $cmbShadow.SelectedIndex = 0; $cmbDet.SelectedIndex = 1; $cmbRt.SelectedIndex = 2; $cmbAo.SelectedIndex = 2; $cmbRefl.SelectedIndex = 1; $cmbFade.SelectedIndex = 1; $cmbAaS.SelectedIndex = 0 }  # (10-23) nitidez del AA al maximo (no cuesta)
function Set-PresetExtreme { $cmbTex.SelectedIndex = 0; $cmbDist.SelectedIndex = 0; $cmbShadow.SelectedIndex = 0; $cmbDet.SelectedIndex = 0; $cmbRt.SelectedIndex = 3; $cmbAo.SelectedIndex = 3; $cmbRefl.SelectedIndex = 0; $cmbFade.SelectedIndex = 0; $cmbAaS.SelectedIndex = 0; $chkAf16.Checked = $true; $chkLod.Checked = $true
	if ($cmbOut) { $cmbOut.SelectedIndex = 1 }; $chk4K.Checked = $true }  # (10-23) extremo: modo cinematico del juego a 4K
# (10-23) Modo Souls PC low profile: 900p y todo en bajo, para PCs modestos.
$chkSoulsLow = New-Check 'Modo Souls PC low profile (900p, todo en bajo)' 0 0 396 'soulslow'
$tips.SetToolTip($chkSoulsLow, 'Para PCs modestos: 30 FPS reales con frame generation de AMD x2 (60 en pantalla), el juego a 900p reescalado con FSR (nitidez 50 %) y todos los graficos en bajo (texturas, distancia, sombras, detalle, reflejos, objetos y nitidez). Al quitarlo vuelve todo a Maximo y a 1440p.')
$panel.Controls.Add($chkSoulsLow); Move-To $chkSoulsLow $gModes 12 84 396
function Set-ResByName([string]$name) {
	for ($i = 0; $i -lt $script:resShown.Count; $i++) { if ($script:resShown[$i][0] -eq $name) { $cmbRes.SelectedIndex = $i; Set-IniValue 'scalelabel' $name; return } }
}
function Set-PresetLowProfile {
	$cmbTex.SelectedIndex = 3; $cmbDist.SelectedIndex = 3; $cmbShadow.SelectedIndex = 3; $cmbDet.SelectedIndex = 4
	$cmbRefl.SelectedIndex = 3; $cmbFade.SelectedIndex = 3; $cmbAaS.SelectedIndex = 3; $cmbRt.SelectedIndex = 0; $cmbAo.SelectedIndex = 0
	# (10-23) Lo que casi no cuesta, al maximo: filtrado x16, texturas nitidas y el postprocesado del juego.
	$chkAf16.Checked = $true; $chkLod.Checked = $true
	foreach ($pc in @($script:postChecks)) { if ($pc -and $pc[0]) { $pc[0].Checked = $true } }
	if ($cmbOut) { $cmbOut.SelectedIndex = 0 }; $chk4K.Checked = $false
	Fill-Res; Set-ResByName 'FSR 900p (AA)'
	$cmbSharp.SelectedIndex = 3  # (10-23) FSR con nitidez al 50 %, fija
}
$script:soulsModes = @($chkLowEnd, $chkLowEnd45, $chkSoulsUltra, $chkSoulsExt, $chkSoulsLow)
$chkSoulsLow.Add_CheckedChanged({
	if ($chkSoulsLow.Checked) { foreach ($o in $script:soulsModes) { if ($o -ne $chkSoulsLow -and $o.Checked) { $o.Checked = $false } }; Set-PresetLowProfile }
	else { if (-not (@($script:soulsModes | Where-Object { $_.Checked }).Count)) { Set-PresetMax }; Fill-Res; Set-ResByName '1440p nativo' }
})
foreach ($m in @($chkSoulsUltra, $chkSoulsExt)) {
	$m.Add_CheckedChanged({ param($sender, $e)
		if ($sender.Checked) {
			foreach ($o in $script:soulsModes) { if ($o -ne $sender -and $o.Checked) { $o.Checked = $false } }
			if ($sender -eq $chkSoulsExt) { Set-PresetExtreme } else { Set-PresetUltra }
		} elseif (-not (@($script:soulsModes | Where-Object { $_.Checked }).Count)) { Set-PresetMax; if ($sender -eq $chkSoulsExt -and $cmbOut -and $cmbOut.SelectedIndex -eq 1) { $cmbOut.SelectedIndex = 0; $chk4K.Checked = $false } }
	})
}
foreach ($m in @($chkLowEnd, $chkLowEnd45)) { $m.Add_CheckedChanged({ param($sender, $e) if ($sender.Checked) { $chkSoulsUltra.Checked = $false; $chkSoulsExt.Checked = $false; $chkSoulsLow.Checked = $false } }) }
$chkGpu8.Add_CheckedChanged({ if ($chkGpu8.Checked) { Set-PresetLight } else { Set-PresetMax } })
# (10-23) Modo Souls ligero y PC low profile: resolucion fija (no se puede cambiar el reescalado).
function Update-ResLock { $lock = $chkSoulsLow.Checked; if ($lock -and $chkLowEnd.Checked) { Set-ResByName 'Sin reescalado (1080p nativo)' }; $cmbRes.Enabled = -not $lock 
	# (10-23) PC low profile: todo fijo (salida, graficos y casillas), nada se puede cambiar.
	$low = $chkSoulsLow.Checked
	foreach ($c in @($cmbSharp, $cmbFg, $cmbOut, $cmbTex, $cmbDist, $cmbShadow, $cmbDet, $cmbRefl, $cmbFade, $cmbAaS, $cmbRt, $cmbAo, $chkAf16, $chkLod)) { if ($c) { $c.Enabled = -not $low } }
	foreach ($p in @($script:postChecks)) { if ($p -and $p[0]) { $p[0].Enabled = -not $low } }
}
$chkLowEnd.Add_CheckedChanged({ Update-ResLock })
$chkSoulsLow.Add_CheckedChanged({ Update-ResLock; Update-LowEnd })
$form.Add_Shown({ Update-ResLock; Update-LowEnd })
$chkLowEnd.Add_CheckedChanged({ if ($chkLowEnd.Checked) { Set-PresetLight } elseif (-not $chkLowEnd45.Checked) { Set-PresetMax } })
$chkLowEnd45.Add_CheckedChanged({ if ($chkLowEnd45.Checked) { Set-PresetMax; $cmbTex.SelectedIndex = 1; $cmbDist.SelectedIndex = 1; $cmbShadow.SelectedIndex = 1; $cmbRt.SelectedIndex = 0; $cmbAo.SelectedIndex = 2; $cmbAaS.SelectedIndex = 1 } elseif (-not $chkLowEnd.Checked) { Set-PresetMax } })
Move-To $chkTurbo2 $gPc 202 54 206
$chkStore = New-Check 'Reserva PS5 reducida (12 GB)' 0 0 396 'guest12'
$tips.SetToolTip($chkStore, 'Reserva 12 GB para la memoria de la PS5 en vez de 13,8 GB (el juego usa como mucho unos 10,7). Ayuda en PCs de 16 GB. Si el juego se cierra con un aviso de memoria, quitala.')
$chkStore.Visible = $false; $chkStore.Checked = $false
$chkRamOpt.Visible = $false
# Pantalla y otros: HDR, VSync, Metricas en una fila (pantalla exclusiva, Modo 4K, Preset C y G-Sync siguen ocultas,
# en el mismo sitio: no se ven).
Move-To $chkHdr $gScr 12 24 72
Move-To $chkVsync $gScr 88 24 84
Move-To $chkMetricas $gScr 176 24 100
# (10-23) Monitores panoramicos: el 3D a 21:9 o 32:9 (run-windows -Aspect; con DLSS / XeSS / FSR la salida es ancha).
$cmbAspect = New-Object System.Windows.Forms.ComboBox
$cmbAspect.DropDownStyle = 'DropDownList'; $cmbAspect.Font = $smallFont
foreach ($l in 'Monitor 16:9', 'Monitor 21:9', 'Monitor 32:9') { [void]$cmbAspect.Items.Add($l) }
$ai = 0; [void][int]::TryParse((Get-IniValue 'aspect' '0'), [ref]$ai); $cmbAspect.SelectedIndex = [math]::Min([math]::Max(0, $ai), 2)
$cmbAspect.Add_SelectedIndexChanged({ Set-IniValue 'aspect' "$($cmbAspect.SelectedIndex)" })
$tips.SetToolTip($cmbAspect, 'Monitores panoramicos (ultrawide): el juego se dibuja a 21:9 o 32:9 en vez de estirarse. Funciona con DLSS, XeSS y FSR. Los videos y algunos menus se ven estirados.')
$gScr.Controls.Add($cmbAspect); $cmbAspect.Location = New-Object System.Drawing.Point(280, 21); $cmbAspect.Size = New-Object System.Drawing.Size(130, 26)
Move-To $chkExcl $gScr 12 24 120
Move-To $chk4K $gScr 142 24 120
Move-To $chkPresetC $gScr 272 24 136
Move-To $chkGsync $gScr 12 24 120
# Botones abajo.
$btnPlay.Location = New-Object System.Drawing.Point(450, 390); $btnPlay.Size = New-Object System.Drawing.Size(420, 64)
$btnPre.Location = New-Object System.Drawing.Point(450, 462); $btnPre.Size = New-Object System.Drawing.Size(420, 44)
$btnSettings.Location = New-Object System.Drawing.Point(450, 514); $btnSettings.Size = New-Object System.Drawing.Size(205, 44)
$btnExit.Location = New-Object System.Drawing.Point(665, 514); $btnExit.Size = New-Object System.Drawing.Size(205, 44)
# (10-23) que nada tape los botones de abajo
foreach ($b in @($btnPlay, $btnPre, $btnSettings, $btnExit)) { $b.BringToFront() }

# ---- (10-23) Mandos: mapeo del mando (KYTY_PAD_MAP) y teclado y raton (KYTY_KBM) ----
# El emulador lee el mapeo de las variables de entorno al arrancar: KYTY_PAD_MAP="a=Cross;b=Circle;..." (nombres de
# SDL del boton fisico = control de PS5) y, con "Teclado y raton", KYTY_KBM=1, KYTY_KBM_MAP="Cross=F;R1=Mouse:Left;...",
# KYTY_KBM_SENS y KYTY_KBM_ANTIDZ. Se guardan en launcher.ini (padmap, kbm, kbmmap, kbmsens, kbmadz). Vacio = por defecto.
if (-not ([System.Management.Automation.PSTypeName]'PadXi').Type) {
	Add-Type @"
using System; using System.Runtime.InteropServices;
public static class PadXi {
  [StructLayout(LayoutKind.Sequential)] public struct XGamepad { public ushort wButtons; public byte bLeftTrigger; public byte bRightTrigger; public short sThumbLX; public short sThumbLY; public short sThumbRX; public short sThumbRY; }
  [StructLayout(LayoutKind.Sequential)] public struct XState { public uint dwPacketNumber; public XGamepad Gamepad; }
  [DllImport("xinput1_4.dll", EntryPoint="XInputGetState")] static extern uint Get14(uint i, out XState s);
  [DllImport("xinput9_1_0.dll", EntryPoint="XInputGetState")] static extern uint Get910(uint i, out XState s);
  static int lib = 0;
  // Botones (bits de XInput) y gatillos en un solo valor: bit 16 = LT, bit 17 = RT. -1: sin mando.
  public static int Read(uint i) {
    XState s = new XState();
    uint r = 1167;
    if (lib == 0) { try { r = Get14(i, out s); } catch (Exception) { lib = 1; } }
    if (lib == 1) { try { r = Get910(i, out s); } catch (Exception) { lib = 2; } }
    if (lib == 2) return -1;
    if (r != 0) return -1;
    int v = s.Gamepad.wButtons;
    if (s.Gamepad.bLeftTrigger > 100) v |= 0x10000;
    if (s.Gamepad.bRightTrigger > 100) v |= 0x20000;
    return v;
  }
}
public static class PadSdl {
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr LoadLibraryW(string p);
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern int SDL_Init(uint f);
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern void SDL_QuitSubSystem(uint f);
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern int SDL_SetHint([MarshalAs(UnmanagedType.LPStr)] string n, [MarshalAs(UnmanagedType.LPStr)] string v);
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern int SDL_NumJoysticks();
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern int SDL_IsGameController(int i);
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr SDL_GameControllerOpen(int i);
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern void SDL_GameControllerClose(IntPtr p);
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern void SDL_GameControllerUpdate();
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern byte SDL_GameControllerGetButton(IntPtr p, int b);
  [DllImport("SDL2.dll", CallingConvention=CallingConvention.Cdecl)] public static extern short SDL_GameControllerGetAxis(IntPtr p, int a);
}
"@
}
# (10-23) Lo que hace cada control en Demon's Souls (para el mapeo).
$script:padAction = @{
	'R1' = 'Ataque'; 'R2' = 'Ataque fuerte'; 'L1' = 'Bloquear'; 'L2' = 'Parry'; 'L3' = 'Pulsar stick izquierdo'; 'R3' = 'Fijar objetivo'
	'Cross' = 'Interactuar'; 'Circle' = 'Rodar / correr'; 'Square' = 'Usar objeto'; 'Triangle' = 'Arma a dos manos'
	'Up' = 'Cambiar magia'; 'Down' = 'Cambiar objeto'; 'Left' = 'Arma mano izquierda'; 'Right' = 'Arma mano derecha'
	'Options' = 'Menu'; 'TouchPad' = 'Gestos'
}
$script:padGlyph = @{
	'Cross' = [string][char]0x2715; 'Circle' = [string][char]0x25CB; 'Triangle' = [string][char]0x25B3; 'Square' = [string][char]0x25A1
	'L1' = 'L1'; 'R1' = 'R1'; 'L2' = 'L2'; 'R2' = 'R2'; 'L3' = 'L3'; 'R3' = 'R3'; 'Options' = 'Options'; 'TouchPad' = 'Touchpad'
	'Up' = [string][char]0x25B2; 'Down' = [string][char]0x25BC; 'Left' = [string][char]0x25C0; 'Right' = [string][char]0x25B6; 'None' = '-'
}
$script:xbLabel = @{
	'a' = "A $([char]0x2715)"; 'b' = "B $([char]0x25CB)"; 'x' = "X $([char]0x25A1)"; 'y' = "Y $([char]0x25B3)"; 'leftshoulder' = 'LB / L1'; 'rightshoulder' = 'RB / R1'; 'lefttrigger' = 'LT / L2'; 'righttrigger' = 'RT / R2'
	'leftstick' = 'LS L3'; 'rightstick' = 'RS R3'; 'start' = 'Start'; 'back' = 'Select'
	'dpup' = [string][char]0x25B2; 'dpdown' = [string][char]0x25BC; 'dpleft' = [string][char]0x25C0; 'dpright' = [string][char]0x25B6
}
$script:xbSources = @('a', 'b', 'x', 'y', 'leftshoulder', 'rightshoulder', 'lefttrigger', 'righttrigger', 'leftstick', 'rightstick', 'start', 'back', 'dpup', 'dpdown', 'dpleft', 'dpright')
$script:padDefault = [ordered]@{
	'a' = 'Cross'; 'b' = 'Circle'; 'x' = 'Square'; 'y' = 'Triangle'; 'leftshoulder' = 'L1'; 'rightshoulder' = 'R1'; 'lefttrigger' = 'L2'; 'righttrigger' = 'R2'
	'leftstick' = 'L3'; 'rightstick' = 'R3'; 'start' = 'Options'; 'back' = 'None'; 'dpup' = 'Up'; 'dpdown' = 'Down'; 'dpleft' = 'Left'; 'dpright' = 'Right'
}
# XInput: bit -> nombre de SDL. SDL: indice de boton -> nombre (orden de SDL_GameControllerButton).
$script:xiBits = [ordered]@{ 0x1000 = 'a'; 0x2000 = 'b'; 0x4000 = 'x'; 0x8000 = 'y'; 0x100 = 'leftshoulder'; 0x200 = 'rightshoulder'; 0x10000 = 'lefttrigger'; 0x20000 = 'righttrigger'
	0x40 = 'leftstick'; 0x80 = 'rightstick'; 0x10 = 'start'; 0x20 = 'back'; 0x1 = 'dpup'; 0x2 = 'dpdown'; 0x4 = 'dpleft'; 0x8 = 'dpright' }
$script:sdlNames = @('a', 'b', 'x', 'y', 'back', 'guide', 'start', 'leftstick', 'rightstick', 'leftshoulder', 'rightshoulder', 'dpup', 'dpdown', 'dpleft', 'dpright')
# Teclado y raton: los mismos de serie que el emulador (hostInput.cpp, KBM_DEFAULTS), estilo Souls.
$script:kbmDefault = 'LeftStickUp=W;LeftStickDown=S;LeftStickLeft=A;LeftStickRight=D;Walk=Left Ctrl;R1=Mouse:Left;L1=Mouse:Right;R2=Shift+Mouse:Left;L2=Shift+Mouse:Right;R3=Mouse:Middle;R3=Mouse:WheelUp;R3=Mouse:WheelDown;Circle=Space;Square=E;Cross=F;Triangle=G;Left=Q;Right=R;Up=1;Down=2;Left=3;Right=4;Options=Tab;TouchPad=Backspace'
$script:kbmNames = @{
	'LeftStickUp' = 'Mover adelante'; 'LeftStickDown' = 'Mover atras'; 'LeftStickLeft' = 'Mover a la izquierda'; 'LeftStickRight' = 'Mover a la derecha'
	'Walk' = 'Andar (media velocidad)'; 'R1' = 'R1 (ataque)'; 'L1' = 'L1 (bloquear)'; 'R2' = 'R2 (ataque fuerte)'; 'L2' = 'L2 (parry / fuerte izq.)'
	'R3' = 'R3 (fijar objetivo)'; 'Circle' = "$([char]0x25CB) (rodar / atras)"; 'Square' = "$([char]0x25A1) (usar objeto)"; 'Cross' = "$([char]0x2715) (interactuar)"
	'Triangle' = "$([char]0x25B3) (dos manos)"; 'Left' = "$([char]0x25C0) (cambiar mano izq.)"; 'Right' = "$([char]0x25B6) (cambiar mano der.)"
	'Up' = "$([char]0x25B2) (cambiar magia)"; 'Down' = "$([char]0x25BC) (cambiar objeto)"; 'Options' = 'Options (menu)'; 'TouchPad' = 'Touchpad'
}
function Get-PadMap {
	$m = [ordered]@{}; foreach ($k in $script:padDefault.Keys) { $m[$k] = $script:padDefault[$k] }
	foreach ($e in ((Get-IniValue 'padmap' '') -split ';')) { if ($e -match '^([a-z]+)=(\w+)$' -and $m.Contains($Matches[1])) { $m[$Matches[1]] = $Matches[2] } }
	return $m
}
function Format-PadMap($m) {
	$same = $true; foreach ($k in $script:padDefault.Keys) { if ($m[$k] -ne $script:padDefault[$k]) { $same = $false } }
	if ($same) { return '' }
	return (($m.Keys | ForEach-Object { "$_=$($m[$_])" }) -join ';')
}
function Get-KbmRows([string]$text) {
	$rows = New-Object System.Collections.ArrayList
	foreach ($e in ($text -split ';')) { $i = $e.IndexOf('='); if ($i -gt 0) { [void]$rows.Add(@{ c = $e.Substring(0, $i); k = $e.Substring($i + 1) }) } }
	return , $rows
}
function Format-KbmInput([string]$k) {
	if (-not $k -or $k -eq 'None') { return '(sin asignar)' }
	$pre = ''; if ($k.StartsWith('Shift+')) { $pre = 'Mayus + '; $k = $k.Substring(6) }
	$n = @{ 'Mouse:Left' = 'Clic izquierdo'; 'Mouse:Right' = 'Clic derecho'; 'Mouse:Middle' = 'Clic central'; 'Mouse:X1' = 'Boton lateral 1'; 'Mouse:X2' = 'Boton lateral 2'
		'Mouse:WheelUp' = 'Rueda arriba'; 'Mouse:WheelDown' = 'Rueda abajo'; 'Space' = 'Espacio'; 'Left Ctrl' = 'Ctrl izq.'; 'Left Shift' = 'Mayus izq.'; 'Left Alt' = 'Alt izq.'
		'Backspace' = 'Retroceso'; 'Return' = 'Intro'; 'Up' = 'Flecha arriba'; 'Down' = 'Flecha abajo'; 'Left' = 'Flecha izquierda'; 'Right' = 'Flecha derecha' }
	if ($n.ContainsKey($k)) { return $pre + $n[$k] }
	return $pre + $k
}
# Tecla de WinForms -> nombre de tecla de SDL (SDL_GetKeyFromName). $null: no se puede usar (Esc, F7-F11, ';').
function ConvertTo-SdlKey([System.Windows.Forms.Keys]$key) {
	$k = $key -band [System.Windows.Forms.Keys]::KeyCode
	$s = [string]$k
	if ($s -match '^[A-Z]$') { return $s }
	if ($s -match '^D(\d)$') { return $Matches[1] }
	if ($s -match '^NumPad(\d)$') { return "Keypad $($Matches[1])" }
	if ($s -match '^F(\d+)$') { if ([int]$Matches[1] -in 7, 8, 9, 11) { return $null }; return $s }
	$map = @{ 'Space' = 'Space'; 'Tab' = 'Tab'; 'Back' = 'Backspace'; 'Return' = 'Return'; 'ShiftKey' = 'Left Shift'; 'LShiftKey' = 'Left Shift'; 'RShiftKey' = 'Left Shift'
		'ControlKey' = 'Left Ctrl'; 'LControlKey' = 'Left Ctrl'; 'RControlKey' = 'Left Ctrl'; 'Menu' = 'Left Alt'; 'LMenu' = 'Left Alt'; 'RMenu' = 'Left Alt'
		'Up' = 'Up'; 'Down' = 'Down'; 'Left' = 'Left'; 'Right' = 'Right'; 'Capital' = 'CapsLock'; 'Insert' = 'Insert'; 'Delete' = 'Delete'; 'Home' = 'Home'; 'End' = 'End'
		'PageUp' = 'PageUp'; 'Next' = 'PageDown'; 'Oemcomma' = ','; 'OemPeriod' = '.'; 'OemMinus' = '-'; 'Oemplus' = '='; 'OemQuestion' = '/'; 'Oem7' = "'"
		'OemOpenBrackets' = '['; 'Oem6' = ']'; 'Oem5' = '\'; 'Oemtilde' = '`'; 'Oem102' = '<' }
	if ($map.ContainsKey($s)) { return $map[$s] }
	return $null
}

function Show-PadMapper {
	$back = [System.Drawing.Color]::FromArgb(14, 16, 22)
	$f = New-Object System.Windows.Forms.Form
	$f.Text = 'Mapear mando'; $f.StartPosition = 'CenterScreen'; $f.FormBorderStyle = 'FixedDialog'; $f.MaximizeBox = $false; $f.MinimizeBox = $false
	$f.ClientSize = New-Object System.Drawing.Size(1280, 760); $f.BackColor = $back; $f.ForeColor = $ink; $f.Font = $smallFont
	$f.KeyPreview = $true
	$tabs = New-Object System.Windows.Forms.TabControl
	$tabs.Location = New-Object System.Drawing.Point(10, 10); $tabs.Size = New-Object System.Drawing.Size(1260, 680); $tabs.Font = $boldSmall
	$f.Controls.Add($tabs)
	$tPad = New-Object System.Windows.Forms.TabPage; $tPad.Text = 'Mando'; $tPad.BackColor = $back
	$tKbm = New-Object System.Windows.Forms.TabPage; $tKbm.Text = 'Teclado y raton'; $tKbm.BackColor = $back
	$tabs.TabPages.Add($tPad); $tabs.TabPages.Add($tKbm)

	$script:pmMap = Get-PadMap
	$script:pmSel = $null; $script:pmHover = $null; $script:pmFlash = $null
	$script:pmRectPs = @{}; $script:pmRectXb = @{}
	$script:pmAll = (Get-IniValue 'padlines' '0') -eq '1'

	$info = New-Label 'Paso 1: haz clic en el boton del juego que quieres mover (izquierda).   Paso 2: pulsa ese boton en tu mando, o haz clic en el de la derecha.   Las flechas doradas son los cambios.' $smallFont $gold 12 8 1220 24
	$tPad.Controls.Add($info)
	$canvas = New-Object System.Windows.Forms.PictureBox
	$canvas.Location = New-Object System.Drawing.Point(8, 36); $canvas.Size = New-Object System.Drawing.Size(1236, 500); $canvas.BackColor = $back
	$tPad.Controls.Add($canvas)
	$status = New-Label '' $subFont $ink 12 544 1220 30
	$tPad.Controls.Add($status)
	$chkAll = New-Object System.Windows.Forms.CheckBox
	$chkAll.Text = 'Ver todas las conexiones'; $chkAll.ForeColor = $ink; $chkAll.Font = $smallFont; $chkAll.BackColor = $back
	$chkAll.Location = New-Object System.Drawing.Point(12, 584); $chkAll.Size = New-Object System.Drawing.Size(260, 26); $chkAll.Checked = $script:pmAll
	$chkAll.Add_CheckedChanged({ $script:pmAll = $chkAll.Checked; Set-IniValue 'padlines' $(if ($chkAll.Checked) { '1' } else { '0' }); $canvas.Invalidate() })
	$tPad.Controls.Add($chkAll)
	$lbDev = New-Label 'Buscando mandos...' $smallFont ([System.Drawing.Color]::FromArgb(170, 170, 170)) 290 584 940 26
	$tPad.Controls.Add($lbDev)

	# Rectangulos de cada boton (coordenadas relativas a cada mando).
	$ps = @{ 'L2' = @(80, 18, 110, 30, 'r'); 'L1' = @(80, 54, 110, 24, 'r'); 'R2' = @(350, 18, 110, 30, 'r'); 'R1' = @(350, 54, 110, 24, 'r')
		'Up' = @(125, 128, 30, 30, 'r'); 'Down' = @(125, 188, 30, 30, 'r'); 'Left' = @(95, 158, 30, 30, 'r'); 'Right' = @(155, 158, 30, 30, 'r')
		'Triangle' = @(384, 120, 34, 34, 'c'); 'Cross' = @(384, 190, 34, 34, 'c'); 'Square' = @(349, 155, 34, 34, 'c'); 'Circle' = @(419, 155, 34, 34, 'c')
		'TouchPad' = @(196, 100, 148, 74, 'r'); 'Options' = @(350, 98, 52, 18, 'r'); 'L3' = @(186, 214, 66, 66, 'c'); 'R3' = @(288, 214, 66, 66, 'c') }
	$xb = @{ 'lefttrigger' = @(80, 18, 110, 30, 'r'); 'leftshoulder' = @(80, 54, 110, 24, 'r'); 'righttrigger' = @(350, 18, 110, 30, 'r'); 'rightshoulder' = @(350, 54, 110, 24, 'r')
		'leftstick' = @(96, 126, 66, 66, 'c'); 'rightstick' = @(296, 214, 66, 66, 'c')
		'dpup' = @(190, 198, 30, 30, 'r'); 'dpdown' = @(190, 258, 30, 30, 'r'); 'dpleft' = @(160, 228, 30, 30, 'r'); 'dpright' = @(220, 228, 30, 30, 'r')
		'y' = @(384, 112, 34, 34, 'c'); 'a' = @(384, 182, 34, 34, 'c'); 'x' = @(349, 147, 34, 34, 'c'); 'b' = @(419, 147, 34, 34, 'c')
		'back' = @(206, 138, 52, 18, 'r'); 'start' = @(282, 138, 52, 18, 'r') }
	# (10-23) Fotos propias de los mandos (mando-ps5.png / mando-8bitdo.png junto al launcher; no van en la release):
	# los botones se colocan sobre la foto (centros medidos en la foto a 350 de alto).
	$script:pmPhotoPs = $null; $script:pmPhotoXb = $null
	try {
		$fp = Join-Path $root 'mando-ps5.png'; if (Test-Path -LiteralPath $fp) { $script:pmPhotoPs = [System.Drawing.Image]::FromFile($fp) }
		$fx = Join-Path $root 'mando-8bitdo.png'; if (Test-Path -LiteralPath $fx) { $script:pmPhotoXb = [System.Drawing.Image]::FromFile($fx) }
	} catch { $script:pmPhotoPs = $null; $script:pmPhotoXb = $null }
	$photoTop = 84
	if ($script:pmPhotoPs) {
		$k = 350 / 371; $offX = (540 - $script:pmPhotoPs.Width) / 2
		$c = @{ 'Up' = @(100, 72, 26); 'Down' = @(100, 132, 26); 'Left' = @(70, 102, 26); 'Right' = @(130, 102, 26)
			'Triangle' = @(438, 62, 34); 'Circle' = @(478, 102, 34); 'Cross' = @(438, 142, 34); 'Square' = @(400, 102, 34)
			'L3' = @(180, 175, 60); 'R3' = @(355, 175, 60); 'Options' = @(395, 42, 22); 'TouchPad' = @(270, 60, 90) }
		foreach ($n in $c.Keys) { $v = $c[$n]; $d = $v[2] * $k; $h = if ($n -eq 'TouchPad') { 80 * $k } else { $d }
			$ps[$n] = @(($offX + $v[0] * $k - $d / 2), ($photoTop + $v[1] * $k - $h / 2), $d, $h, $(if ($n -eq 'TouchPad') { 'r' } else { 'c' })) }
	}
	if ($script:pmPhotoXb) {
		$k = 350 / 393; $offX = (540 - $script:pmPhotoXb.Width) / 2
		$c = @{ 'y' = @(420, 70, 36); 'x' = @(380, 108, 36); 'b' = @(460, 108, 36); 'a' = @(420, 150, 36)
			'leftstick' = @(97, 110, 76); 'rightstick' = @(337, 198, 76)
			'dpup' = @(183, 170, 26); 'dpdown' = @(183, 226, 26); 'dpleft' = @(155, 198, 26); 'dpright' = @(211, 198, 26)
			'back' = @(186, 62, 26); 'start' = @(332, 62, 26) }
		foreach ($n in $c.Keys) { $v = $c[$n]; $d = $v[2] * $k
			$xb[$n] = @(($offX + $v[0] * $k - $d / 2), ($photoTop + $v[1] * $k - $d / 2), $d, $d, 'c') }
	}
	$script:pmPsDef = $ps; $script:pmXbDef = $xb
	$psX = 30; $xbX = 690; $padY = 60
	foreach ($k in $ps.Keys) { $v = $ps[$k]; $script:pmRectPs[$k] = New-Object System.Drawing.RectangleF(($psX + $v[0]), ($padY + $v[1]), $v[2], $v[3]) }
	foreach ($k in $xb.Keys) { $v = $xb[$k]; $script:pmRectXb[$k] = New-Object System.Drawing.RectangleF(($xbX + $v[0]), ($padY + $v[1]), $v[2], $v[3]) }

	$script:pmFontBtn = New-Object System.Drawing.Font('Segoe UI Semibold', 10, [System.Drawing.FontStyle]::Bold)
	$script:pmFontGlyph = New-Object System.Drawing.Font('Segoe UI Symbol', 13, [System.Drawing.FontStyle]::Bold)
	$script:pmFontTiny = New-Object System.Drawing.Font('Segoe UI Symbol', 8, [System.Drawing.FontStyle]::Bold)
	$script:pmFontTitle = New-Object System.Drawing.Font('Segoe UI Semibold', 13, [System.Drawing.FontStyle]::Bold)

	$canvas.Add_Paint({
		param($s, $e)
		$g = $e.Graphics
		$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
		$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
		$center = New-Object System.Drawing.StringFormat; $center.Alignment = 'Center'; $center.LineAlignment = 'Center'
		$rr = {
			param($x, $y, $w, $h, $r)
			$p = New-Object System.Drawing.Drawing2D.GraphicsPath
			$p.AddArc($x, $y, $r, $r, 180, 90); $p.AddArc($x + $w - $r, $y, $r, $r, 270, 90)
			$p.AddArc($x + $w - $r, $y + $h - $r, $r, $r, 0, 90); $p.AddArc($x, $y + $h - $r, $r, $r, 90, 90); $p.CloseFigure()
			return $p
		}
		$bodyFill = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(34, 38, 50))
		$bodyPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(150, 222, 196, 140), 2)
		# (10-23) Silueta de mando con empunaduras (dibujada; un contorno continuo) y gatillos detras.
		$trigFill = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(26, 29, 38))
		foreach ($ox in $psX, $xbX) {
			$oy = $padY
			$photo = if ($ox -eq $psX) { $script:pmPhotoPs } else { $script:pmPhotoXb }
			if ($photo) {
				foreach ($tx in ($ox + 70), ($ox + 340)) { $tp = & $rr $tx ($oy + 10) 130 74 26; $g.FillPath($trigFill, $tp); $g.DrawPath($bodyPen, $tp); $tp.Dispose() }
				$g.DrawImage($photo, [float]($ox + (540 - $photo.Width) / 2), [float]($oy + 84), [float]$photo.Width, [float]$photo.Height)
				continue
			}
			foreach ($tx in ($ox + 70), ($ox + 340)) { $tp = & $rr $tx ($oy + 10) 130 90 26; $g.FillPath($trigFill, $tp); $g.DrawPath($bodyPen, $tp); $tp.Dispose() }
			$body = New-Object System.Drawing.Drawing2D.GraphicsPath
			$pts = @(@(110, 84), @(160, 76), @(380, 76), @(430, 84), @(480, 92), @(505, 120), @(512, 170), @(525, 250), @(540, 330), @(520, 375),
				@(505, 405), @(460, 410), @(440, 385), @(420, 360), @(400, 310), @(370, 295), @(330, 285), @(210, 285), @(170, 295),
				@(140, 310), @(120, 360), @(100, 385), @(80, 410), @(35, 405), @(20, 375), @(0, 330), @(15, 250), @(28, 170),
				@(35, 120), @(60, 92), @(110, 84))
			$pf = New-Object 'System.Drawing.PointF[]' $pts.Count
			for ($q = 0; $q -lt $pts.Count; $q++) { $pf[$q] = New-Object System.Drawing.PointF(($ox + $pts[$q][0]), ($oy + $pts[$q][1])) }
			$body.AddBeziers($pf); $body.CloseFigure()
			$g.FillPath($bodyFill, $body); $g.DrawPath($bodyPen, $body); $body.Dispose()
		}
		$trigFill.Dispose()
		$tb = New-Object System.Drawing.SolidBrush($gold)
		$g.DrawString('1. Boton del juego (mando de PS5)', $script:pmFontTitle, $tb, ($psX + 270), 22, $center)
		$g.DrawString('2. Tu mando (Xbox, 8BitDo o DualSense)', $script:pmFontTitle, $tb, ($xbX + 270), 22, $center)
		$g.DrawString([string][char]0x2192, $script:pmFontTitle, $tb, 630, 22, $center)
		$g.FillEllipse($bodyFill, ($xbX + 250), ($padY + 96), 40, 40)
		$g.DrawEllipse($bodyPen, ($xbX + 250), ($padY + 96), 40, 40)
		$dim = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(120, 120, 120))
		$g.DrawString('Create', $script:pmFontTiny, $dim, (New-Object System.Drawing.RectangleF(($psX + 138), ($padY + 98), 52, 18)), $center)
		# Que controles de PS5 tienen boton fisico (para el aviso de "sin asignar").
		$used = @{}; foreach ($k in $script:pmMap.Keys) { $used[$script:pmMap[$k]] = $k }
		$glyphColor = @{ 'Cross' = [System.Drawing.Color]::FromArgb(120, 170, 255); 'Circle' = [System.Drawing.Color]::FromArgb(255, 100, 110)
			'Triangle' = [System.Drawing.Color]::FromArgb(80, 220, 170); 'Square' = [System.Drawing.Color]::FromArgb(240, 130, 210)
			'y' = [System.Drawing.Color]::FromArgb(240, 210, 80); 'a' = [System.Drawing.Color]::FromArgb(110, 210, 110); 'x' = [System.Drawing.Color]::FromArgb(100, 160, 255); 'b' = [System.Drawing.Color]::FromArgb(255, 100, 100) }
		$drawBtn = {
			param($name, $rect, $shape, $text, $hot, $faded, $onPhoto)
			if ($onPhoto) {
				$ring = New-Object System.Drawing.Pen($(if ($hot) { [System.Drawing.Color]::FromArgb(255, 236, 200, 120) } else { [System.Drawing.Color]::FromArgb(110, 236, 200, 120) }), $(if ($hot) { 4 } else { 2 }))
				if ($hot) { $hb = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(110, 236, 200, 120)); if ($shape -eq 'c') { $g.FillEllipse($hb, $rect) } else { $g.FillRectangle($hb, $rect) }; $hb.Dispose() }
				if ($shape -eq 'c') { $g.DrawEllipse($ring, $rect) } else { $g.DrawRectangle($ring, $rect.X, $rect.Y, $rect.Width, $rect.Height) }
				$ring.Dispose(); return
			}
			$fill = if ($hot) { [System.Drawing.Color]::FromArgb(222, 196, 140) } elseif ($faded) { [System.Drawing.Color]::FromArgb(40, 44, 54) } else { [System.Drawing.Color]::FromArgb(58, 64, 80) }
			$br = New-Object System.Drawing.SolidBrush($fill)
			$pen = New-Object System.Drawing.Pen($(if ($hot) { [System.Drawing.Color]::White } else { [System.Drawing.Color]::FromArgb(110, 110, 100) }), $(if ($hot) { 2 } else { 1 }))
			if ($shape -eq 'c') { $g.FillEllipse($br, $rect); $g.DrawEllipse($pen, $rect) }
			else { $p = & $rr $rect.X $rect.Y $rect.Width $rect.Height ([math]::Min(10, $rect.Height)); $g.FillPath($br, $p); $g.DrawPath($pen, $p) }
			$col = if ($hot) { [System.Drawing.Color]::FromArgb(20, 16, 10) } elseif ($glyphColor.ContainsKey($name)) { $glyphColor[$name] } else { $ink }
			$tbr = New-Object System.Drawing.SolidBrush($col)
			$fnt = if ($text.Length -le 2) { $script:pmFontGlyph } else { $script:pmFontBtn }
			if ($rect.Height -lt 20) { $fnt = $script:pmFontTiny }
			$g.DrawString($text, $fnt, $tbr, $rect, $center)
			$br.Dispose(); $pen.Dispose(); $tbr.Dispose()
		}
		# Conexiones: la del boton elegido y las cambiadas (o todas), del boton fisico al control de PS5.
		foreach ($src in $script:xbSources) {
			$t = $script:pmMap[$src]
			if ($t -eq 'None' -or -not $script:pmRectPs.ContainsKey($t)) { continue }
			$changed = $t -ne $script:padDefault[$src]
			$isSel = ($script:pmSel -eq $t) -or ($script:pmHover -eq $src) -or ($script:pmHover -eq $t) -or ($script:pmFlash -eq $src)
			if (-not ($isSel -or $changed -or $script:pmAll)) { continue }
			$a = $script:pmRectXb[$src]; $b = $script:pmRectPs[$t]
			$col = if ($isSel) { [System.Drawing.Color]::FromArgb(255, 236, 200, 120) } elseif ($changed) { [System.Drawing.Color]::FromArgb(200, 222, 160, 70) } else { [System.Drawing.Color]::FromArgb(70, 200, 200, 210) }
			$pen = New-Object System.Drawing.Pen($col, $(if ($isSel) { 3 } elseif ($changed) { 2 } else { 1 }))
			$pen.EndCap = [System.Drawing.Drawing2D.LineCap]::ArrowAnchor
			$x1 = $a.X + $a.Width / 2; $y1 = $a.Y + $a.Height / 2; $x2 = $b.X + $b.Width / 2; $y2 = $b.Y + $b.Height / 2
			$g.DrawBezier($pen, $x1, $y1, ($x1 - 160), ($y1 - 60), ($x2 + 160), ($y2 - 60), $x2, $y2)
			$pen.Dispose()
		}
		# (10-23) Que hace cada control en Demon's Souls: los gatillos al lado, el resto al pasar el raton (abajo).
		$capBr = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(190, 190, 180))
		$capL = New-Object System.Drawing.StringFormat; $capL.Alignment = 'Far'; $capL.LineAlignment = 'Center'
		$capR = New-Object System.Drawing.StringFormat; $capR.Alignment = 'Near'; $capR.LineAlignment = 'Center'
		foreach ($k in 'L1', 'L2') { $r = $script:pmRectPs[$k]; $g.DrawString($script:padAction[$k], $script:pmFontTiny, $capBr, (New-Object System.Drawing.RectangleF(($r.X - 82), $r.Y, 78, $r.Height)), $capL) }
		foreach ($k in 'R1', 'R2') { $r = $script:pmRectPs[$k]; $g.DrawString($script:padAction[$k], $script:pmFontTiny, $capBr, (New-Object System.Drawing.RectangleF(($r.Right + 4), $r.Y, 90, $r.Height)), $capR) }
		$focus = if ($script:pmSel) { $script:pmSel } elseif ($script:pmHover -and $script:pmRectPs.ContainsKey($script:pmHover)) { $script:pmHover } elseif ($script:pmHover) { $script:pmMap[$script:pmHover] } else { $null }
		if ($focus -and $script:padAction.ContainsKey($focus)) {
			$cap = "$($script:padGlyph[$focus])  =  $($script:padAction[$focus])"
			$cb = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(230, 40, 34, 22))
			$cr = if ($script:pmPhotoPs) { New-Object System.Drawing.RectangleF(($psX + 110), ($padY + 84 + 290), 320, 30) } else { New-Object System.Drawing.RectangleF(($psX + 110), ($padY + 410), 320, 30) }
			$g.FillRectangle($cb, $cr); $g.DrawString($cap, $script:pmFontTitle, $tb, $cr, $center); $cb.Dispose()
		}
		$capBr.Dispose()
		foreach ($k in $script:pmRectPs.Keys) {
			$hot = ($script:pmSel -eq $k) -or ($script:pmHover -eq $k)
			$onP = $script:pmPhotoPs -and $k -notin 'L1', 'L2', 'R1', 'R2'
			& $drawBtn $k $script:pmRectPs[$k] $script:pmPsDef[$k][4] $script:padGlyph[$k] $hot (-not $used.ContainsKey($k)) $onP
		}
		foreach ($k in $script:pmRectXb.Keys) {
			$hot = ($script:pmHover -eq $k) -or ($script:pmFlash -eq $k) -or ($script:pmSel -and $script:pmMap[$k] -eq $script:pmSel)
			$onP = $script:pmPhotoXb -and $k -notin 'leftshoulder', 'rightshoulder', 'lefttrigger', 'righttrigger'
			& $drawBtn $k $script:pmRectXb[$k] $script:pmXbDef[$k][4] $script:xbLabel[$k] $hot $false $onP
			# Insignia con el control de PS5 que hace este boton.
			$r = $script:pmRectXb[$k]; $t = $script:pmMap[$k]
			$bb = New-Object System.Drawing.RectangleF(($r.Right - 6), ($r.Y - 12), 40, 18)
			$bg = New-Object System.Drawing.SolidBrush($(if ($t -ne $script:padDefault[$k]) { [System.Drawing.Color]::FromArgb(230, 176, 140, 70) } else { [System.Drawing.Color]::FromArgb(200, 14, 16, 22) }))
			$g.FillRectangle($bg, $bb)
			$g.DrawString($script:padGlyph[$t], $script:pmFontTiny, $tb, $bb, $center)
			$bg.Dispose()
		}
		$bodyFill.Dispose(); $bodyPen.Dispose(); $tb.Dispose(); $dim.Dispose()
	})
	$hit = {
		param($pt)
		foreach ($k in $script:pmRectPs.Keys) { if ($script:pmRectPs[$k].Contains($pt)) { return @('ps', $k) } }
		foreach ($k in $script:pmRectXb.Keys) { if ($script:pmRectXb[$k].Contains($pt)) { return @('xb', $k) } }
		return $null
	}
	$script:pmAssign = {
		param($src)
		if (-not $script:pmSel) {
			$t = $script:pmMap[$src]
			$status.Text = "Tu boton $($script:xbLabel[$src]) hace $($script:padGlyph[$t]) en el juego. Elige primero un boton a la izquierda para cambiarlo."
			return
		}
		$old = $script:pmMap[$src]
		# Intercambio: el boton que tenia este control se queda con el que tenia el nuevo (nada se queda sin hacer).
		foreach ($k in @($script:pmMap.Keys)) { if ($k -ne $src -and $script:pmMap[$k] -eq $script:pmSel) { $script:pmMap[$k] = $old } }
		$script:pmMap[$src] = $script:pmSel
		$status.Text = "Hecho: $($script:padGlyph[$script:pmSel]) ($($script:padAction[$script:pmSel])) = boton $($script:xbLabel[$src]) de tu mando."
		$script:pmFlash = $src; $script:pmSel = $null
		$canvas.Invalidate()
	}
	$canvas.Add_MouseMove({
		param($s, $e)
		$h = & $hit (New-Object System.Drawing.PointF($e.X, $e.Y))
		$n = if ($h) { $h[1] } else { $null }
		if ($n -ne $script:pmHover) { $script:pmHover = $n; $canvas.Cursor = $(if ($n) { [System.Windows.Forms.Cursors]::Hand } else { [System.Windows.Forms.Cursors]::Default }); $canvas.Invalidate() }
	})
	$canvas.Add_MouseClick({
		param($s, $e)
		$h = & $hit (New-Object System.Drawing.PointF($e.X, $e.Y))
		if (-not $h) { $script:pmSel = $null; $status.Text = ''; $canvas.Invalidate(); return }
		if ($h[0] -eq 'ps') {
			$script:pmSel = $h[1]; $script:pmFlash = $null
			$status.Text = "Elegido $($script:padGlyph[$h[1]]) ($($script:padAction[$h[1]])). Pulsa ahora el boton de tu mando (o haz clic en el de la derecha)."
			$canvas.Invalidate()
		} else { & $script:pmAssign $h[1] }
	})

	# Lectura del mando fisico: XInput (Xbox, 8BitDo en modo X) y, si esta SDL2.dll junto al emulador, SDL (DualSense y otros).
	$script:pmXiPrev = @{}; $script:pmSdlPrev = @{}; $script:pmSdlPads = @(); $script:pmSdlOk = $false; $script:pmScan = 0
	$sdlDll = Join-Path $root 'SDL2.dll'
	if (Test-Path -LiteralPath $sdlDll) {
		try {
			if ([PadSdl]::LoadLibraryW($sdlDll) -ne [IntPtr]::Zero) {
				[void][PadSdl]::SDL_SetHint('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')
				if ([PadSdl]::SDL_Init(0x2200) -eq 0) { $script:pmSdlOk = $true }
			}
		} catch { $script:pmSdlOk = $false }
	}
	$script:pmTimer = New-Object System.Windows.Forms.Timer
	$script:pmTimer.Interval = 25
	$script:pmTimer.Add_Tick({
		$pressed = $null; $names = @()
		for ($i = 0; $i -lt 4; $i++) {
			$v = [PadXi]::Read($i)
			if ($v -lt 0) { continue }
			$names += "XInput $($i + 1)"
			$prev = if ($script:pmXiPrev.ContainsKey($i)) { $script:pmXiPrev[$i] } else { $v }
			# (10-23) Por pares clave/valor: un diccionario [ordered] indexado con un numero devuelve por POSICION (0x1000 = A
			# daba nada y la cruceta daba botones cambiados): solo funcionaba la cruceta.
			foreach ($kv in $script:xiBits.GetEnumerator()) { $bit = [int]$kv.Key; if ((($v -band $bit) -ne 0) -and (($prev -band $bit) -eq 0)) { $pressed = $kv.Value } }
			$script:pmXiPrev[$i] = $v
		}
		if ($script:pmSdlOk) {
			try {
				$script:pmScan++
				if ($script:pmScan % 80 -eq 1) {
					foreach ($p in $script:pmSdlPads) { [PadSdl]::SDL_GameControllerClose($p) }
					$script:pmSdlPads = @()
					$n = [PadSdl]::SDL_NumJoysticks()
					for ($j = 0; $j -lt $n; $j++) { if ([PadSdl]::SDL_IsGameController($j) -ne 0) { $p = [PadSdl]::SDL_GameControllerOpen($j); if ($p -ne [IntPtr]::Zero) { $script:pmSdlPads += $p } } }
				}
				[PadSdl]::SDL_GameControllerUpdate()
				$j = 0
				foreach ($p in $script:pmSdlPads) {
					$names += "SDL $($j + 1)"
					$v = 0
					for ($b = 0; $b -lt $script:sdlNames.Count; $b++) { if ([PadSdl]::SDL_GameControllerGetButton($p, $b) -ne 0) { $v = $v -bor (1 -shl $b) } }
					if ([PadSdl]::SDL_GameControllerGetAxis($p, 4) -gt 13000) { $v = $v -bor (1 -shl 20) }
					if ([PadSdl]::SDL_GameControllerGetAxis($p, 5) -gt 13000) { $v = $v -bor (1 -shl 21) }
					$prev = if ($script:pmSdlPrev.ContainsKey($j)) { $script:pmSdlPrev[$j] } else { $v }
					for ($b = 0; $b -lt 22; $b++) {
						if ((($v -band (1 -shl $b)) -ne 0) -and (($prev -band (1 -shl $b)) -eq 0)) {
							$nm = if ($b -eq 20) { 'lefttrigger' } elseif ($b -eq 21) { 'righttrigger' } elseif ($b -lt $script:sdlNames.Count) { $script:sdlNames[$b] } else { $null }
							if ($nm -and $script:xbLabel.ContainsKey($nm)) { $pressed = $nm }
						}
					}
					$script:pmSdlPrev[$j] = $v; $j++
				}
			} catch { $script:pmSdlOk = $false }
		}
		$txt = if ($names.Count) { 'Mandos detectados: ' + ($names -join ', ') } else { 'No se detecta ningun mando (puedes hacer clic en los botones de la derecha).' }
		if ($lbDev.Text -ne $txt) { $lbDev.Text = $txt }
		if ($pressed -and $tabs.SelectedTab -eq $tPad) {
			if ($script:pmSel) { & $script:pmAssign $pressed }
			else { $script:pmFlash = $pressed; $status.Text = "Has pulsado $($script:xbLabel[$pressed]): en el juego hace $($script:padGlyph[$script:pmMap[$pressed]])."; $canvas.Invalidate() }
		}
	})
	$script:pmTimer.Start()

	# ---- Pestana Teclado y raton ----
	$script:pmKbm = Get-KbmRows $(if (Get-IniValue 'kbmmap' '') { Get-IniValue 'kbmmap' '' } else { $script:kbmDefault })
	$tKbm.Controls.Add((New-Label 'Elige una fila y pulsa "Cambiar" (o doble clic); luego pulsa la tecla o haz clic con el boton del raton en el recuadro.' $smallFont $gold 12 8 1220 24))
	$lv = New-Object System.Windows.Forms.ListView
	$lv.View = 'Details'; $lv.FullRowSelect = $true; $lv.MultiSelect = $false; $lv.HideSelection = $false
	$lv.Location = New-Object System.Drawing.Point(12, 38); $lv.Size = New-Object System.Drawing.Size(640, 560)
	$lv.BackColor = [System.Drawing.Color]::FromArgb(24, 26, 34); $lv.ForeColor = $ink; $lv.Font = New-Object System.Drawing.Font('Segoe UI Symbol', 10)
	[void]$lv.Columns.Add('Accion en el juego', 330); [void]$lv.Columns.Add('Tecla / raton', 280)
	$tKbm.Controls.Add($lv)
	$script:pmFillLv = {
		$sel = if ($lv.SelectedIndices.Count) { $lv.SelectedIndices[0] } else { -1 }
		$lv.BeginUpdate(); $lv.Items.Clear()
		foreach ($r in $script:pmKbm) {
			$name = if ($script:kbmNames.ContainsKey($r.c)) { $script:kbmNames[$r.c] } else { $r.c }
			$it = New-Object System.Windows.Forms.ListViewItem($name); [void]$it.SubItems.Add((Format-KbmInput $r.k)); [void]$lv.Items.Add($it)
		}
		if ($sel -ge 0 -and $sel -lt $lv.Items.Count) { $lv.Items[$sel].Selected = $true }
		$lv.EndUpdate()
		if ($mouseBox) { $mouseBox.Invalidate() }
	}
	& $script:pmFillLv
	$cap = New-Object System.Windows.Forms.Button
	$cap.Location = New-Object System.Drawing.Point(670, 38); $cap.Size = New-Object System.Drawing.Size(560, 200)
	$cap.FlatStyle = 'Flat'; $cap.BackColor = [System.Drawing.Color]::FromArgb(24, 28, 38); $cap.ForeColor = $ink; $cap.Font = $subFont
	$cap.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(110, 110, 100)
	$cap.Text = 'Elige una accion de la lista.'
	$tKbm.Controls.Add($cap)
	$script:pmWait = -1
	$btnChange = New-Button 'Cambiar' 670 250 270 44 $false; $btnChange.Font = $boldSmall
	$btnClear = New-Button 'Quitar tecla' 960 250 270 44 $false; $btnClear.Font = $boldSmall
	$tKbm.Controls.Add($btnChange); $tKbm.Controls.Add($btnClear)
	$tKbm.Controls.Add((New-Label ("Raton: mueve la camara (stick derecho). Mayus + clic: ataques fuertes (R2 / L2).`nEn el juego: Esc suelta el raton y un clic lo vuelve a capturar. F7 tambien lo suelta o lo captura.`nLa sensibilidad y la anti-zona muerta estan en el lanzador (apartado Controles).") $smallFont ([System.Drawing.Color]::FromArgb(190, 190, 190)) 670 310 560 90 'TopLeft'))
	# (10-23) Dibujo del raton con lo que hace cada boton (se actualiza al cambiar la lista).
	$mouseBox = New-Object System.Windows.Forms.PictureBox
	$mouseBox.Location = New-Object System.Drawing.Point(670, 404); $mouseBox.Size = New-Object System.Drawing.Size(560, 236); $mouseBox.BackColor = $back
	$tKbm.Controls.Add($mouseBox)
	$mouseBox.Add_Paint({
		param($s, $e)
		$g = $e.Graphics; $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
		$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
		$fill = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(34, 38, 50))
		$btnFill = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(58, 64, 80))
		$pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(150, 222, 196, 140), 2)
		$thin = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(120, 222, 196, 140), 1)
		$gold2 = New-Object System.Drawing.SolidBrush($gold); $txt = New-Object System.Drawing.SolidBrush($ink)
		$f1 = New-Object System.Drawing.Font('Segoe UI Semibold', 9, [System.Drawing.FontStyle]::Bold); $f2 = New-Object System.Drawing.Font('Segoe UI Symbol', 9)
		# Cuerpo: dos arcos y la base redondeada; botones izquierdo y derecho, rueda y dos laterales.
		$mphoto = $null; $mp = Join-Path $root 'raton.png'
		if (Test-Path -LiteralPath $mp) { try { $mphoto = [System.Drawing.Image]::FromFile($mp) } catch { $mphoto = $null } }
		$cx = 90; $top = 10; $w = 130; $h = 214
		$body = New-Object System.Drawing.Drawing2D.GraphicsPath
		$body.AddArc(($cx - $w / 2), $top, $w, 120, 180, 180); $body.AddArc(($cx - $w / 2), ($top + $h - 140), $w, 140, 0, 180); $body.CloseFigure()
		if ($mphoto) { $g.DrawImage($mphoto, 10, 8, $mphoto.Width, $mphoto.Height) } else { $g.FillPath($fill, $body); $g.DrawPath($pen, $body) }
		$lb = New-Object System.Drawing.Drawing2D.GraphicsPath; $lb.AddArc(($cx - $w / 2 + 4), ($top + 4), ($w - 8), 112, 180, 90); $lb.AddLine(($cx - 2), ($top + 4), ($cx - 2), ($top + 80)); $lb.AddLine(($cx - 2), ($top + 80), ($cx - $w / 2 + 4), ($top + 80)); $lb.CloseFigure()
		$rb = New-Object System.Drawing.Drawing2D.GraphicsPath; $rb.AddArc(($cx - $w / 2 + 4), ($top + 4), ($w - 8), 112, 270, 90); $rb.AddLine(($cx + $w / 2 - 4), ($top + 80), ($cx + 2), ($top + 80)); $rb.AddLine(($cx + 2), ($top + 80), ($cx + 2), ($top + 4)); $rb.CloseFigure()
		if (-not $mphoto) {
			$g.FillPath($btnFill, $lb); $g.DrawPath($thin, $lb); $g.FillPath($btnFill, $rb); $g.DrawPath($thin, $rb)
			$wheel = New-Object System.Drawing.RectangleF(($cx - 9), ($top + 22), 18, 40); $g.FillRectangle($fill, $wheel); $g.DrawRectangle($thin, $wheel.X, $wheel.Y, $wheel.Width, $wheel.Height)
			$x1r = New-Object System.Drawing.RectangleF(($cx - $w / 2 - 6), ($top + 104), 10, 26); $x2r = New-Object System.Drawing.RectangleF(($cx - $w / 2 - 6), ($top + 136), 10, 26)
			foreach ($r in $x1r, $x2r) { $g.FillRectangle($btnFill, $r); $g.DrawRectangle($thin, $r.X, $r.Y, $r.Width, $r.Height) }
		}
		# Que hace cada entrada del raton (de la lista).
		$what = { param($inp)
			$acts = @(); foreach ($r in $script:pmKbm) { if ($r.k -eq $inp -or $r.k -eq "Shift+$inp") { $n = if ($script:kbmNames.ContainsKey($r.c)) { $script:kbmNames[$r.c] } else { $r.c }; $acts += $(if ($r.k -like 'Shift+*') { "Mayus+: $n" } else { $n }) } }
			if ($acts.Count) { return ($acts -join '   ') } else { return '(sin asignar)' } }
		$rows = @(@('Clic izquierdo', 'Mouse:Left', ($cx - 34), ($top + 40)), @('Clic derecho', 'Mouse:Right', ($cx + 34), ($top + 40)),
			@('Rueda (clic)', 'Mouse:Middle', $cx, ($top + 42)), @('Rueda arriba', 'Mouse:WheelUp', $cx, ($top + 26)), @('Rueda abajo', 'Mouse:WheelDown', $cx, ($top + 58)),
			@('Lateral 1', 'Mouse:X1', ($cx - $w / 2 - 1), ($top + 117)), @('Lateral 2', 'Mouse:X2', ($cx - $w / 2 - 1), ($top + 149)))
		$lx = 200
		if ($mphoto) {
			# Puntos medidos en la foto (raton.png a 220 de alto, dibujada en 10,8).
			$pt = @{ 'Mouse:Left' = @(55, 83); 'Mouse:Right' = @(125, 78); 'Mouse:Middle' = @(85, 60); 'Mouse:WheelUp' = @(80, 54); 'Mouse:WheelDown' = @(90, 66); 'Mouse:X1' = @(52, 113); 'Mouse:X2' = @(58, 136) }
			foreach ($row in $rows) { $row[2] = $pt[$row[1]][0]; $row[3] = $pt[$row[1]][1] }
			$lx = 236
		}
		$y = 6
		foreach ($row in $rows) {
			$ly = $y + 12
			$g.DrawLine($thin, [float]$row[2], [float]$row[3], [float]($lx - 4), [float]$ly); $g.FillEllipse($gold2, ([float]$row[2] - 3), ([float]$row[3] - 3), 6, 6)
			$g.DrawString($row[0], $f1, $gold2, $lx, $y + 3)
			$g.DrawString((& $what $row[1]), $f2, $txt, (New-Object System.Drawing.RectangleF(($lx + 100), ($y + 3), (558 - $lx - 100), 30)))
			$y += 32
		}
		$g.DrawString('Mover el raton = camara', $f1, $gold2, 14, 216)
		foreach ($o in $fill, $btnFill, $pen, $thin, $gold2, $txt, $f1, $f2, $body, $lb, $rb) { $o.Dispose() }
		if ($mphoto) { $mphoto.Dispose() }
	})
	$script:pmStartWait = {
		if ($lv.SelectedIndices.Count -eq 0) { $cap.Text = 'Elige primero una accion de la lista.'; return }
		$script:pmWait = $lv.SelectedIndices[0]
		$cap.Text = "Pulsa una tecla o haz clic aqui con un boton del raton (rueda: girala aqui).`nMayus + clic = combinacion con Mayus.   Esc: cancelar."
		$cap.BackColor = [System.Drawing.Color]::FromArgb(70, 60, 30)
		[void]$cap.Focus()
	}
	$script:pmSetInput = {
		param($k)
		$i = $script:pmWait; $script:pmWait = -1
		$cap.BackColor = [System.Drawing.Color]::FromArgb(24, 28, 38)
		if ($i -lt 0 -or $i -ge $script:pmKbm.Count) { return }
		# Una tecla solo hace una cosa: si ya estaba en otra fila, esa se queda sin asignar.
		for ($j = 0; $j -lt $script:pmKbm.Count; $j++) { if ($j -ne $i -and $script:pmKbm[$j].k -eq $k) { $script:pmKbm[$j].k = 'None' } }
		$script:pmKbm[$i].k = $k
		$cap.Text = "Asignado: $(Format-KbmInput $k)"
		& $script:pmFillLv
	}
	$btnChange.Add_Click({ & $script:pmStartWait })
	$lv.Add_DoubleClick({ & $script:pmStartWait })
	$btnClear.Add_Click({ if ($lv.SelectedIndices.Count) { $script:pmWait = $lv.SelectedIndices[0]; & $script:pmSetInput 'None' } })
	$cap.Add_PreviewKeyDown({ param($s, $e) $e.IsInputKey = $true })
	$cap.Add_KeyDown({
		param($s, $e)
		$e.Handled = $true; $e.SuppressKeyPress = $true
		if ($script:pmWait -lt 0) { return }
		if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Escape) { $script:pmWait = -1; $cap.BackColor = [System.Drawing.Color]::FromArgb(24, 28, 38); $cap.Text = 'Cancelado.'; return }
		$k = ConvertTo-SdlKey $e.KeyData
		if (-not $k) { $cap.Text = "Esa tecla no se puede usar (Esc, F7, F8, F9 y F11 son del emulador). Prueba otra.`nPulsa una tecla o haz clic aqui."; return }
		& $script:pmSetInput $k
	})
	$cap.Add_MouseDown({
		param($s, $e)
		if ($script:pmWait -lt 0) { return }
		$n = switch ($e.Button) { 'Left' { 'Mouse:Left' } 'Right' { 'Mouse:Right' } 'Middle' { 'Mouse:Middle' } 'XButton1' { 'Mouse:X1' } 'XButton2' { 'Mouse:X2' } default { $null } }
		if (-not $n) { return }
		if (([System.Windows.Forms.Control]::ModifierKeys -band [System.Windows.Forms.Keys]::Shift) -ne 0) { $n = "Shift+$n" }
		& $script:pmSetInput $n
	})
	$cap.Add_MouseWheel({
		param($s, $e)
		if ($script:pmWait -lt 0 -or $e.Delta -eq 0) { return }
		& $script:pmSetInput $(if ($e.Delta -gt 0) { 'Mouse:WheelUp' } else { 'Mouse:WheelDown' })
	})

	# ---- Botones de abajo ----
	$btnOk = New-Button 'Guardar' 10 702 220 48 $true
	$btnReset = New-Button 'Restablecer' 240 702 220 48 $false
	$btnCancel = New-Button 'Cancelar' 1050 702 220 48 $false
	foreach ($b in $btnOk, $btnReset, $btnCancel) { $f.Controls.Add($b) }
	$btnReset.Add_Click({
		$script:pmMap = [ordered]@{}; foreach ($k in $script:padDefault.Keys) { $script:pmMap[$k] = $script:padDefault[$k] }
		$script:pmKbm = Get-KbmRows $script:kbmDefault
		$script:pmSel = $null; $script:pmFlash = $null
		$status.Text = 'Restablecido: los botones de serie (mando y teclado). Pulsa Guardar para quedartelo.'
		& $script:pmFillLv; $canvas.Invalidate()
	})
	$btnOk.Add_Click({
		Set-IniValue 'padmap' (Format-PadMap $script:pmMap)
		$k = (($script:pmKbm | ForEach-Object { "$($_.c)=$($_.k)" }) -join ';')
		Set-IniValue 'kbmmap' $(if ($k -eq $script:kbmDefault) { '' } else { $k })
		$f.Close()
	})
	$btnCancel.Add_Click({ $f.Close() })
	$f.Add_KeyDown({ param($s, $e) if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Escape -and $script:pmWait -lt 0 -and $tabs.SelectedTab -eq $tPad) { $f.Close() } })
	$f.Add_FormClosed({
		$script:pmTimer.Stop(); $script:pmTimer.Dispose()
		if ($script:pmSdlOk) { try { foreach ($p in $script:pmSdlPads) { [PadSdl]::SDL_GameControllerClose($p) }; [PadSdl]::SDL_QuitSubSystem(0x2200) } catch {} }
		$script:pmSdlPads = @()
	})
	[void]$f.ShowDialog($form)
	$f.Dispose()
}

# Apartado Controles (debajo de los botones): teclado y raton, mapear el mando y los dos deslizadores del raton.
# (10-23) Post-processing del propio juego: marcado = como viene el juego; desmarcado = quitado.
$script:postChecks = @()
$px = 12; $py = 342
foreach ($pp in @(@('Motion Blur', 'motionblur'), @('Depth of Field', 'dof'), @('Vignette', 'vignette'), @('Bloom', 'bloom'), @('Eye Adaptation', 'eyeadapt'))) {
	$c = New-Object System.Windows.Forms.CheckBox
	$c.Text = $pp[0]; $c.Font = $smallFont; $c.ForeColor = $ink; $c.BackColor = [System.Drawing.Color]::Transparent
	$c.Tag = "post_$($pp[1])"; $c.Checked = (Get-IniValue $c.Tag '1') -eq '1'
	$c.Add_CheckedChanged({ param($sender, $e) Set-IniValue $sender.Tag $(if ($sender.Checked) { '1' } else { '0' }) })
	$panel.Controls.Add($c); Move-To $c $gGfx $px $py 130
	$script:postChecks += ,@($c, $pp[1])
	$px += 134; if ($px -gt 300) { $px = 12; $py += 30 }
}
$gCtl = New-Group 'Controles' 450 604 420 168
# (10-23) Controles en un desplegable: el boton los muestra u oculta.
$gCtl.Visible = $false
$btnCtl = New-Button 'Controles  v' 450 566 420 32 $false
$btnCtl.Font = $smallFont
$btnCtl.Add_Click({ $gCtl.Visible = -not $gCtl.Visible; $btnCtl.Text = if ($gCtl.Visible) { 'Controles  ^' } else { 'Controles  v' } })
$panel.Controls.Add($btnCtl)
# (10-23) Donaciones por PayPal (abre la pagina de PayPal en el navegador; no se envia nada desde el lanzador).
$lnkDon = New-Object System.Windows.Forms.LinkLabel
$lnkDon.Text = ([string][char]0x2665) + '  Dona tu alma  -  PayPal: todojuegos67@gmail.com'
$lnkDon.Font = $boldSmall; $lnkDon.LinkColor = $gold; $lnkDon.ActiveLinkColor = [System.Drawing.Color]::White
$lnkDon.BackColor = [System.Drawing.Color]::Transparent
$lnkDon.Location = New-Object System.Drawing.Point(450, 790); $lnkDon.Size = New-Object System.Drawing.Size(420, 28)
$lnkDon.TextAlign = 'MiddleCenter'
$lnkDon.Add_LinkClicked({ Start-Process 'https://www.paypal.com/donate?business=todojuegos67%40gmail.com&currency_code=EUR' })
$tips.SetToolTip($lnkDon, 'Si te gusta el proyecto, puedes donar por PayPal. Gracias, portador de la marca.')
$panel.Controls.Add($lnkDon)
# (10-23) Boton oficial 'Donar' de PayPal, servido por PayPal (paypalobjects.com); si no hay internet, solo queda el texto.
$picDon = New-Object System.Windows.Forms.PictureBox
$picDon.Size = New-Object System.Drawing.Size(92, 26); $picDon.Location = New-Object System.Drawing.Point(614, 822)
$picDon.BackColor = [System.Drawing.Color]::Transparent; $picDon.Cursor = [System.Windows.Forms.Cursors]::Hand
$picDon.SizeMode = 'Zoom'; $picDon.WaitOnLoad = $false
try { $picDon.LoadAsync('https://www.paypalobjects.com/es_ES/ES/i/btn/btn_donate_LG.gif') } catch { }
$picDon.Add_Click({ Start-Process 'https://www.paypal.com/donate?business=todojuegos67%40gmail.com&currency_code=EUR' })
$tips.SetToolTip($picDon, 'Donar con PayPal')
$panel.Controls.Add($picDon)
$chkKbm = New-Check 'Teclado y raton' 0 0 190 'kbm'
$tips.SetToolTip($chkKbm, 'Jugar con teclado y raton: WASD para moverse, el raton mueve la camara, clic izquierdo R1, clic derecho L1, rueda o clic central fija objetivo. Las teclas se cambian en "Mapear mando".')
Move-To $chkKbm $gCtl 12 26 190
$btnMap = New-Button 'Mapear mando' 0 0 186 34 $false
$btnMap.Font = $boldSmall
$tips.SetToolTip($btnMap, 'Ver y cambiar que boton de tu mando (o que tecla) hace cada boton del mando de PS5.')
$btnMap.Add_Click({ Show-PadMapper })
Move-To $btnMap $gCtl 222 20 186
function New-Slider([string]$key, [int]$min, [int]$max, [int]$def, $x, $y, [scriptblock]$fmt) {
	$t = New-Object System.Windows.Forms.TrackBar
	$t.Minimum = $min; $t.Maximum = $max; $t.TickStyle = 'None'; $t.AutoSize = $false
	$t.Size = New-Object System.Drawing.Size(190, 30); $t.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22)
	$v = $def; [void][int]::TryParse((Get-IniValue $key "$def"), [ref]$v)
	$t.Value = [math]::Min([math]::Max($min, $v), $max)
	$t.Tag = @{ key = $key; fmt = $fmt }
	$lb = New-Label (& $fmt $t.Value) $smallFont $ink 0 0 190 22
	$t.Tag.label = $lb
	$t.Add_ValueChanged({ param($s, $e) $s.Tag.label.Text = (& $s.Tag.fmt $s.Value); Set-IniValue $s.Tag.key "$($s.Value)" })
	Move-To $lb $gCtl $x $y 190
	Move-To $t $gCtl $x ($y + 24) 190
	return $t
}
$trkSens = New-Slider 'kbmsens' 5 400 100 12 62 { param($v) 'Sensibilidad del raton: ' + ($v / 100).ToString('0.00', [System.Globalization.CultureInfo]::InvariantCulture) }
$trkAdz = New-Slider 'kbmadz' 0 50 15 218 62 { param($v) "Anti-zona muerta: $v%" }
$tips.SetToolTip($trkSens, 'Cuanto gira la camara por cada movimiento del raton (1.00 = normal).')
$tips.SetToolTip($trkAdz, 'Salta la zona muerta del stick del juego: con mas, un movimiento pequeno del raton ya gira la camara. Si la camara se mueve sola, bajalo.')
function Update-KbmEnabled { $trkSens.Enabled = $chkKbm.Checked; $trkAdz.Enabled = $chkKbm.Checked }
$chkKbm.Add_CheckedChanged({ Update-KbmEnabled })
Update-KbmEnabled
# (10-23) Calibrar la camara para el raton: arranca el juego con KYTY_KBM_CALIBRATE=1 (y teclado y raton).
$script:kbmCalibrate = $false
$btnCal = New-Button 'Calibrar camara (raton)' 0 0 396 34 $false
$btnCal.Font = $boldSmall
$tips.SetToolTip($btnCal, 'Mide como gira la camara del juego para que el raton la mueva sin retraso. Una vez basta (se guarda en kbm_curve.txt).')
$btnCal.Add_Click({
	$msg = "Se abrira el juego para calibrar la camara del raton.`n`n1. Carga tu partida.`n2. Quedate quieto en un sitio abierto (sin enemigos cerca), sin tocar el raton ni el stick derecho.`n3. A los 3 segundos la camara girara sola a izquierda y derecha unos 13 segundos. No toques nada.`n4. Cuando pare, ya esta: cierra el juego o sigue jugando. Queda guardado en kbm_curve.txt.`n`nNecesita un reescalador (DLSS, XeSS o FSR) o frame generation activado."
	if ([System.Windows.Forms.MessageBox]::Show($msg, 'Calibrar camara', 'OKCancel', 'Information') -ne 'OK') { return }
	$script:kbmCalibrate = $true
	try { Start-Game } finally { $script:kbmCalibrate = $false }
})
Move-To $btnCal $gCtl 12 124 396

# ---- Settings (panel desplegable) ----
$settings = New-Object System.Windows.Forms.Panel
$settings.Location  = New-Object System.Drawing.Point(($W - $panelW - 620), ($H - 210))  # a la izquierda de la columna, abajo
$settings.Anchor = 'Bottom, Right'
$settings.Size      = New-Object System.Drawing.Size(610, 190)
$settings.BackColor = $panelBack
$settings.Visible   = $false
$form.Controls.Add($settings)
$settings.Controls.Add((New-Label 'Carpeta del juego volcado (PPSA01341)' $subFont $gold 20 16 560 28))
$pathBox = New-Object System.Windows.Forms.TextBox
$pathBox.Location  = New-Object System.Drawing.Point(20, 56); $pathBox.Size = New-Object System.Drawing.Size(450, 30)
$pathBox.Font      = $smallFont
$pathBox.Text      = Get-GamePath
$settings.Controls.Add($pathBox)
$btnBrowse = New-Button '...' 480 52 50 32 $false
$btnBrowse.Font = $smallFont
$btnBrowse.Add_Click({
	$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
	$dialog.Description = 'Carpeta donde esta el juego volcado (la que contiene eboot.bin o sce_sys)'
	if ($pathBox.Text -and (Test-Path -LiteralPath $pathBox.Text)) { $dialog.SelectedPath = $pathBox.Text }
	if ($dialog.ShowDialog($form) -eq 'OK') { $pathBox.Text = $dialog.SelectedPath }
})
$settings.Controls.Add($btnBrowse)
# (10-23) El juego comprimido en un solo archivo .zar (unos dos tercios del tamano): el emulador lo lee sin descomprimir.
$btnZarPick = New-Button '.zar' 536 52 54 32 $false
$btnZarPick.Font = $smallFont
$tips.SetToolTip($btnZarPick, 'Elegir el juego comprimido (.zar) en vez de la carpeta')
$btnZarPick.Add_Click({
	$d = New-Object System.Windows.Forms.OpenFileDialog
	$d.Filter = 'Juego comprimido (*.zar)|*.zar'; $d.Title = 'Elige el archivo .zar del juego'
	if ($d.ShowDialog($form) -eq 'OK') { $pathBox.Text = $d.FileName }
})
$settings.Controls.Add($btnZarPick)
$btnZarMake = New-Button 'Comprimir a .zar' 200 96 200 36 $false
$btnZarMake.Font = $smallFont
$tips.SetToolTip($btnZarMake, 'Crea un .zar con el juego (unos dos tercios de lo que ocupa la carpeta). La carpeta original no se borra.')
$btnZarMake.Add_Click({
	$src = $pathBox.Text.Trim()
	$tool = Join-Path $root 'zarchive.exe'
	if (-not (Test-Path -LiteralPath $tool)) { [System.Windows.Forms.MessageBox]::Show('Falta zarchive.exe junto al launcher (viene en la release).', 'Comprimir', 'OK', 'Warning') | Out-Null; return }
	if (-not $src -or -not (Test-Path -LiteralPath $src -PathType Container) -or -not (Test-Path -LiteralPath (Join-Path $src 'eboot.bin'))) {
		[System.Windows.Forms.MessageBox]::Show('Pon primero la carpeta del juego (la que tiene eboot.bin).', 'Comprimir', 'OK', 'Warning') | Out-Null; return }
	$dst = $src.TrimEnd('\') + '.zar'
	if (Test-Path -LiteralPath $dst) { [System.Windows.Forms.MessageBox]::Show("Ya existe:`n$dst", 'Comprimir', 'OK', 'Warning') | Out-Null; return }
	$msg = "Se va a crear:`n$dst`n`nOcupara unos dos tercios de la carpeta del juego y tarda varios minutos. Necesitas ese espacio libre en el disco.`nLa carpeta original NO se borra: cuando compruebes que el .zar funciona, puedes borrarla tu.`n`nAl terminar, el launcher usara el .zar."
	if ([System.Windows.Forms.MessageBox]::Show($msg, 'Comprimir a .zar', 'OKCancel', 'Information') -ne 'OK') { return }
	$status.Text = 'Comprimiendo... (ventana aparte)'
	$p = Start-Process -FilePath $tool -ArgumentList @("`"$src`"", "`"$dst`"") -PassThru -Wait
	if ($p.ExitCode -eq 0 -and (Test-Path -LiteralPath $dst)) {
		$pathBox.Text = $dst; Set-GamePath $dst
		$gb = [math]::Round((Get-Item -LiteralPath $dst).Length / 1GB, 1)
		$status.Text = "Hecho: $gb GB. El launcher usa ahora el .zar."
	} else { $status.Text = "No se pudo comprimir (codigo $($p.ExitCode))."; Remove-Item -LiteralPath $dst -ErrorAction SilentlyContinue }
})
$settings.Controls.Add($btnZarMake)
$status = New-Label '' $smallFont $ink 20 140 560 24
$settings.Controls.Add($status)
$btnSave = New-Button 'Guardar' 20 96 160 36 $true
$btnSave.Font = $smallFont
$btnSave.Add_Click({
	$p = $pathBox.Text.Trim()
	if (-not $p -or -not (Test-Path -LiteralPath $p)) { $status.Text = 'Esa carpeta (o ese .zar) no existe.'; return }
	Set-GamePath $p
	$status.Text = 'Guardado en game-path.txt'
	$settings.Visible = $false
})
$settings.Controls.Add($btnSave)
$btnSettings.Add_Click({ $settings.Visible = -not $settings.Visible })
# (10-23) Cruz para cerrar Settings; al guardar bien tambien se cierra.
$btnCloseSet = New-Button ([string][char]0x2715) 572 6 32 30 $false
$btnCloseSet.Font = $smallFont
$tips.SetToolTip($btnCloseSet, 'Cerrar')
$btnCloseSet.Add_Click({ $settings.Visible = $false })
$settings.Controls.Add($btnCloseSet); $btnCloseSet.BringToFront()

# (10-23) Aviso del archivo de paginacion: solo lo LEE; nunca cambia ajustes de Windows. Abre la ventana de Windows
# para que el usuario lo cambie el mismo si quiere. No vuelve a avisar si se pulsa "No volver a avisar".
$form.Add_Shown({
	try {
		return  # (10-23) aviso del archivo de paginacion desactivado
		$cs = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
		if ($cs.AutomaticManagedPagefile) { return }
		$ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB)
		$msg = "Tu archivo de paginacion de Windows no esta en automatico.`n`n" +
			"Con $ramGB GB de RAM el juego puede cerrarse por falta de memoria. Para arreglarlo:`n`n" +
			"  1. Pulsa 'Si': se abre la ventana de Windows 'Opciones de rendimiento'.`n" +
			"  2. Pestana 'Opciones avanzadas' -> boton 'Cambiar...'.`n" +
			"  3. Marca 'Administrar automaticamente el tamano del archivo de paginacion'.`n" +
			"  4. Aceptar y reinicia el PC.`n`n" +
			"Si: abrir ajustes    No: ahora no    Cancelar: no volver a avisar"
		$r = [System.Windows.Forms.MessageBox]::Show($form, $msg, 'Memoria de Windows', 'YesNoCancel', 'Warning')
		if ($r -eq 'Yes') { Start-Process 'SystemPropertiesPerformance.exe' }
		elseif ($r -eq 'Cancel') { Set-IniValue 'pagefilewarn' '0' }
	} catch { }
})
[void]$form.ShowDialog()
