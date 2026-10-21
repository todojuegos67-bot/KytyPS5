# KytyPS5 - Demon's Souls: lanzador (ventana). Launcher.cmd lo abre.
#   Jugar                -> arranca el juego; con "Frame generation AMD x2" (config\fsr3.json) o
#                           "x3" (config\fsr3-x3.json, 2 frames generados), si no, sin FG (config\sin-fg.json). Copia de las partidas antes.
#   Settings             -> carpeta del juego volcado (game-path.txt). La casilla se recuerda en launcher.ini.
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
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
	@('   120 FPS AMD Smooth Souls',                    'fsr3-x3-120.json'),
	@('   AMD MFG dinamico (x2-x4 automatico)',         'fsr3-mfg-dynamic.json'),
	@('---------- NVIDIA (RTX) ----------',             ''),
	@('   Frame generation DLSS x2',                    'dlss-x2.json'),
	@('   Frame generation DLSS x3',                    'dlss-x3.json'),
	@('   Frame generation DLSS x4',                    'dlss-x4.json'),
	@('   200 FPS Nvidia Smooth x5',                    'dlss-x5.json'),
	@('   216 FPS Nvidia Smooth x6',                    'dlss-x6.json'),
	@('   120 FPS Nvidia Smooth Souls',                 'dlss-x3-120.json'),
	@('   DLSS MFG dinamico (x2-x4 automatico)',        'dlss-dynamic.json')
)
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
	@('DLSS Rendim.',   2.0,  'dlss', 0),
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
		if ($text -match 'fgmode=(\d+)') { return [int]$Matches[1] }
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
function Set-FgIndex([int]$i) { Set-IniValue 'fgmode' "$i" }

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
	if ($chkLowEnd.Checked) { $config = 'fsr3.json' }
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
	# Modo TURBO (run-windows -Turbo): caches de video a 4.5 GB (unos 6 GB en total), texturas a un cuarto, detalle bajo
	# del juego y la RAM vaciada cada 15 s por encima de 5 GB: para PCs de 16 GB que se cerraban en zonas pesadas.
	if ($chkTurbo2.Checked) { $psargs += @('-Turbo2', '-Quality', 'low', '-TextureReduce', '2') }
	elseif ($chkLowPc.Checked) { $sets += 'KYTY_SIMULATE_RAM_MB=16384'; $psargs += @('-Turbo', '-Quality', 'low', '-TextureReduce', '2') }
	if ($chkSafeGpu.Checked) { $sets += 'KYTY_COMPUTE_CHAIN=0' }
	# Graficos: texturas (0 entera, 1 mitad, 2 cuarto) y distancia de dibujado (-Quality low/lowest: solo el LOD).
	$psargs += @('-TextureReduce', "$($cmbTex.SelectedIndex)")
	if ($cmbDist.SelectedIndex -eq 1) { $psargs += @('-Quality', 'low') } elseif ($cmbDist.SelectedIndex -eq 2) { $psargs += @('-Quality', 'lowest') }
	if ($chkExcl.Checked) { $psargs += '-ExclusiveFullscreen' }
	# Preset C con la imagen sin HUD (hudless) daba pantalla negra con solo el HUD (10-21): sin ese buffer.
	# (10-21) Preset C del frame generation de NVIDIA siempre (mas rapido; el emulador solo lo aplica a DLSS-G).
	$sets += 'KYTY_DLSSG_PRESET=C'
	# Nitidez del reescalado (KYTY_UPSCALE_SHARPNESS): FSR con su propio RCAS, DLSS y XeSS con un paso de nitidez despues.
	$sharp = $sharpLevels[[math]::Max(0, $cmbSharp.SelectedIndex)][1]
	if ($sharp -ne '') { $sets += "KYTY_UPSCALE_SHARPNESS=$sharp" }
	# FPS desbloqueados con frame generation: el juego deja de estar limitado a 60 reales (KYTY_FRAMEGEN_BASE=0); los
	# modos Smooth siguen en sus FPS fijos (KYTY_FLIP_FPS) y DLSS-G se limita solo con Reflex al tope del monitor.
	$fgConfig = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	if ($chkLowEnd.Checked) { $fgConfig = 'fsr3.json'; $sets += 'KYTY_FLIP_FPS=30' }
	# Nombre del modo de frame generation para las metricas.
	if ($chkLowEnd.Checked) { Set-IniValue 'fglabel' 'MODO PC BAJOS REQUISITOS' } else { Set-IniValue 'fglabel' ($fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][0].Trim()) }
	if ($chkUnlock.Checked -and $fgConfig -and $fgConfig -ne 'sin-fg.json') { $sets += 'KYTY_FRAMEGEN_BASE=0' }
	# AMD x4: 56 FPS reales (224 mostrados), por debajo de 240 Hz para que G-SYNC vaya suave.
	$sets += 'KYTY_FAULT_STATS=1'
	if ($fgConfig -eq 'fsr3-x4.json') { $sets += 'KYTY_VRR_CAP_FPS=224' }
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
	if ($config -in 'dlss-dynamic.json', 'fsr3-mfg-dynamic.json') { $psargs += @('-Vblank', '840') }  # (10-21: 60 = 14 pulsos, 56 = 15)
	elseif ($config -ne 'sin-fg.json') { $psargs += @('-Vblank', '360') }
	# Sin frame generation, "FPS desbloqueados": vblank de 120 Hz (-Fps120, el modo probado de chenxiao07; a 240 los menus y el intro iban acelerados, 10-18), el juego pasa de 60 (a velocidad real:
	# gameMinDeltaTime en run-windows); videos, menus y pantallas de carga siguen a 60.
	# (Desbloqueado sin frame generation: vblank virtual de 360 Hz y cada frame se muestra en cuanto esta listo
	# (-FlipWhenReady), no al siguiente tick de 120 Hz: a ~80 FPS los frames alternaban 8 y 17 ms (tirones).)
	elseif ($chkUnlock.Checked) { $psargs += @('-Vblank', '960') }
	$res    = $script:resShown[[math]::Max(0, $cmbRes.SelectedIndex)]
	if ($res[1] -eq 0) { $res = $resModes[0] }
	# DLSS (reescalado) carga Streamline de NVIDIA; con la generacion de frames de AMD el emulador se cerraba al crear
	# el dispositivo grafico. Con AMD: FSR o XeSS para reescalar.
	if ($res[2] -like '*dlss' -and $config -like 'fsr3*') {
		[System.Windows.Forms.MessageBox]::Show('El reescalado DLSS va con "Sin frame generation" o con DLSS frame generation. Con la generacion de AMD usa FSR o XeSS para reescalar.', 'Launcher', 'OK', 'Warning') | Out-Null
		return
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
	foreach ($n in 'KYTY_VSYNC_LOCK', 'KYTY_FPS_CAP', 'KYTY_FLIP_WHEN_READY') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
	# G-Sync: reloj del emulador suelto del monitor. Mostrar cada frame al momento solo con FPS desbloqueados y sin
	# frame generation; con el juego a 60 (o con frame generation) el ritmo de 60 lo marcan los vblanks y no se toca.
	if ($chkGsync.Checked) {
		$env:KYTY_VSYNC_LOCK = '0'
		if ($chkUnlock.Checked -and $config -eq 'sin-fg.json') { $env:KYTY_FPS_CAP = 'auto'; $env:KYTY_FLIP_WHEN_READY = '1' }
	}
	Start-Process -FilePath 'powershell' -ArgumentList $psargs -WorkingDirectory $root -WindowStyle Hidden | Out-Null
	$metArgs = @(); if ($chkRam16.Checked) { $metArgs = @('-RamGB', '16') }
	if ($chkVram8.Checked) { $metArgs += @('-VramMB', '8192') }
	# Como administrador solo con el limite de RAM (lo necesita); si no, normal (no molesta a la grabacion de NVIDIA).
	$metVerb = if ($chkRam16.Checked) { @{ Verb = 'RunAs' } } else { @{} }
	if ($chkMetricas.Checked -or $chkRam16.Checked -or $chkVram8.Checked) { Start-Process -FilePath 'powershell' -ArgumentList (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', ('"' + (Join-Path $root 'metricas.ps1') + '"'), '-LogDir', ('"' + (Join-Path $root 'logs') + '"')) + $metArgs) -WindowStyle Hidden @metVerb -ErrorAction SilentlyContinue | Out-Null }
	$form.WindowState = 'Minimized'
}

# ---- ventana ----
$W = 1280; $H = 720
$form = New-Object System.Windows.Forms.Form
$form.Text            = 'Launcher'
$form.ClientSize      = New-Object System.Drawing.Size($W, $H)
$form.StartPosition   = 'CenterScreen'
$form.FormBorderStyle = 'FixedSingle'
$form.MaximizeBox     = $false
$form.BackColor       = [System.Drawing.Color]::FromArgb(14, 16, 22)
$fondo = Join-Path $root 'fondo-souls.jpg'
if (Test-Path -LiteralPath $fondo) {
	$form.BackgroundImage       = [System.Drawing.Image]::FromFile($fondo)
	$form.BackgroundImageLayout = 'Stretch'
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
$panel.Location  = New-Object System.Drawing.Point(0, ($H - 178))
$panel.Size      = New-Object System.Drawing.Size($W, 178)
$panel.BackColor = $panelBack
$form.Controls.Add($panel)

$btnPlay = New-Button 'PLAY' 50 38 260 74 $true
$btnPlay.Font = New-Object System.Drawing.Font('Segoe UI Black', 22, [System.Drawing.FontStyle]::Bold)
$btnPlay.Add_Click({ Start-Game })
$panel.Controls.Add($btnPlay)

$panel.Controls.Add((New-Label 'Frame generation' $smallFont $ink 350 30 300 22))
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
	Set-FgIndex $cmbFg.SelectedIndex
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
$chkLowEnd = New-Check 'Modo graficas de 8 GB' 910 44 220 'lowend'
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
$tips.SetToolTip($chkLowEnd, 'Adaptado para graficas de 8 GB como RTX 3060, RTX 3060 Ti, RTX 3070, RTX 2070/2080, RX 6600, RX 6600 XT o RX 6650 XT: el juego a 30 FPS estables y frame generation de AMD x2 (60 en pantalla). Reescalado solo con XeSS o FSR.')
$panel.Controls.Add($chkLowEnd)
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
	$cfg = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	if ($cfg -like 'dlss*') { return @('', 'dlss', 'juego') }
	if ($cfg -like 'fsr3*') { return @('', 'xess', 'fsr', 'juego') }
	return @('', 'dlss', 'xess', 'fsr', 'juego')
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
		if ($chkLowEnd -and $chkLowEnd.Checked -and $family -eq 'dlss') { $modes = @() }
		if ($modes.Count -eq 0) { continue }
		if ($titles.ContainsKey($family)) { $shown += , @($titles[$family], 0, '', 0) }
		foreach ($m in $modes) { $shown += , $m }
	}
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
$cmbRes.DropDownWidth = 170
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
$chkTurbo2.Add_CheckedChanged({ Fill-Res })
# PC bajos requisitos: la lista de frame generation bloqueada (va con AMD x2) y sin modos DLSS.
$chkLowEnd.Add_CheckedChanged({ $cmbFg.Enabled = -not $chkLowEnd.Checked; Fill-Res })
$cmbFg.Enabled = -not $chkLowEnd.Checked
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
$gfxLevels = @('Maximo', 'Medio', 'Bajo')
$script:gfxTop = @()
function New-GfxCombo([string]$label, [string]$key, $x, [string]$tip) {
	$lb = New-Label $label $smallFont $ink $x 6 90 22; $script:gfxTop += $lb
	$panel.Controls.Add($lb)
	$c = New-Object System.Windows.Forms.ComboBox
	$c.DropDownStyle = 'DropDownList'; $c.Font = $smallFont
	$c.Location = New-Object System.Drawing.Point(($x + 92), 2); $c.Size = New-Object System.Drawing.Size(100, 26)
	foreach ($l in $gfxLevels) { [void]$c.Items.Add($l) }
	$i = 0; [void][int]::TryParse((Get-IniValue $key '0'), [ref]$i)
	$c.SelectedIndex = [math]::Min([math]::Max(0, $i), 2)
	$c.Tag = $key
	$c.Add_SelectedIndexChanged({ param($sender, $e) Set-IniValue $sender.Tag "$($sender.SelectedIndex)" })
	$tips.SetToolTip($c, $tip)
	$script:gfxTop += $c
	$panel.Controls.Add($c)
	return $c
}
$cmbTex  = New-GfxCombo 'Texturas' 'gfxtex' 290 'Texturas: Medio = mitad de tamano, Bajo = un cuarto. Lo que mas VRAM ahorra.'
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
$panel.Controls.Add($btnExit)
# Fila de arriba (10-21): pantalla exclusiva y Preset C; todo lo demas baja 28 px (antes se tapaban con PLAY y Salir).
foreach ($c in @($panel.Controls)) { if ($c -ne $chkExcl -and $c -ne $chkPresetC -and $script:gfxTop -notcontains $c) { $c.Top += 28 } }
$chkExcl.Location = New-Object System.Drawing.Point(50, 4)
$chkPresetC.Location = New-Object System.Drawing.Point(260, 4)
$chkLowEnd.Location = New-Object System.Drawing.Point(50, 4)

# ---- Settings (panel desplegable) ----
$settings = New-Object System.Windows.Forms.Panel
$settings.Location  = New-Object System.Drawing.Point(($W - 650), ($H - 388))
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
$status = New-Label '' $smallFont $ink 20 140 560 24
$settings.Controls.Add($status)
$btnSave = New-Button 'Guardar' 20 96 160 36 $true
$btnSave.Font = $smallFont
$btnSave.Add_Click({
	$p = $pathBox.Text.Trim()
	if (-not $p -or -not (Test-Path -LiteralPath $p)) { $status.Text = 'Esa carpeta no existe.'; return }
	Set-GamePath $p
	$status.Text = 'Guardado en game-path.txt'
})
$settings.Controls.Add($btnSave)
$btnSettings.Add_Click({ $settings.Visible = -not $settings.Visible })

[void]$form.ShowDialog()
