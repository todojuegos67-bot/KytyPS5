# KytyPS5 - Demon's Souls: lanzador (ventana). Launcher.cmd lo abre.
#   Jugar                -> arranca el juego; con "Frame generation AMD x2" (config\fsr3.json) o
#                           "x3" (config\fsr3-x3.json, 2 frames generados), si no, sin FG (config\sin-fg.json). Copia de las partidas antes.
#   Settings             -> carpeta del juego volcado (game-path.txt). La casilla se recuerda en launcher.ini.
# (10-26) Texto nitido con el escalado de Windows (125/150 %): GDI-scaled DPI. La ventana sigue en pixeles logicos
# (la maquetacion absoluta no cambia) pero Windows dibuja el texto GDI de etiquetas, botones y desplegables a la
# resolucion real del monitor en vez de estirar un bitmap borroso. (Per-monitor v2 obligaria a reescalar a mano toda
# la maquetacion y los lienzos dibujados.) Windows 10 1809+; si no, sin cambios.
try {
	Add-Type -Namespace KytyDpi -Name Native -MemberDefinition '[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(System.IntPtr v);' -ErrorAction Stop
	[void][KytyDpi.Native]::SetProcessDpiAwarenessContext([System.IntPtr](-5))  # DPI_AWARENESS_CONTEXT_UNAWARE_GDISCALED
} catch { }
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
try { [System.Windows.Forms.Application]::SetCompatibleTextRenderingDefault($false) } catch { }  # (10-26) etiquetas con GDI (ClearType)
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
# (10-27) Grafica dedicada antes que la integrada (portatiles e iGPU + dGPU): NVIDIA, AMD Radeon RX / Pro o Intel Arc
# antes que Intel UHD / Iris o AMD "Radeon(TM) Graphics"; si no, la de mas VRAM. La VRAM real sale del registro
# (HardwareInformation.qwMemorySize): AdapterRAM de Win32_VideoController se queda en 4 GB como maximo.
function Get-GpuVramBytes($vc, $regList) {
	$best = [uint64]0
	foreach ($k in @($regList)) {
		if (-not $k) { continue }
		$desc = "$($k.DriverDesc)"; $pnp = "$($k.MatchingDeviceId)"
		$same = ($desc -and $desc.Trim() -eq "$($vc.Name)".Trim()) -or ($pnp -and "$($vc.PNPDeviceID)" -like "$pnp*")
		if (-not $same) { continue }
		$q = $k.'HardwareInformation.qwMemorySize'
		if ($q) { try { $v = [uint64]$q; if ($v -gt $best) { $best = $v } } catch { } }
	}
	if ($best -eq 0 -and $vc.AdapterRAM) { try { $best = [uint64][uint32]$vc.AdapterRAM } catch { } }
	return $best
}
function Select-BestGpu($vcs, $regList) {
	$list = @($vcs | Where-Object { $_ -and $_.Name })
	if ($list.Count -eq 0) { return $null }
	$scored = foreach ($vc in $list) {
		$n = "$($vc.Name)"; $tier = 1
		if ($n -match 'Microsoft Basic|Remote Display|Virtual|Parsec|Meta Virtual|DisplayLink') { $tier = 0 }
		elseif ($n -match 'Radeon\(TM\)\s*(\d+\w*\s*)?Graphics|Radeon Vega|Vega \d+ Graphics|UHD|Iris|HD Graphics') { $tier = 1 }
		elseif ($n -match 'NVIDIA|GeForce|Quadro|RTX|GTX|Radeon\s*(RX|Pro|VII)|\bRX\s*\d|Intel.*Arc|\bArc\s*(A|B)\d') { $tier = 2 }
		[pscustomobject]@{ Vc = $vc; Tier = $tier; Vram = (Get-GpuVramBytes $vc $regList) }
	}
	return (@($scored | Sort-Object -Property @{ Expression = 'Tier'; Descending = $true }, @{ Expression = 'Vram'; Descending = $true }))[0]
}
$script:gpuName = ''; $script:pcInfo = ''; $script:ramGB = 0; $script:gpuVramBytes = [uint64]0
$script:cimS = $null; try { $script:cimS = New-CimSession -ErrorAction Stop } catch { }  # (10-27 startup) una sola sesion WMI para las 3 consultas
$cimArg = @{}; if ($script:cimS) { $cimArg = @{ CimSession = $script:cimS } }
try {
	$vcs = @(Get-CimInstance Win32_VideoController @cimArg -ErrorAction Stop | Where-Object { $_.Name })
	$gpuReg = @(); try { $gpuReg = @(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue) } catch { }
	$best = Select-BestGpu $vcs $gpuReg
	if ($best) { $script:gpuName = "$($best.Vc.Name)".Trim(); $script:gpuVramBytes = [uint64]$best.Vram }
	$cpu = (Get-CimInstance Win32_Processor @cimArg -ErrorAction Stop | Select-Object -First 1).Name
	$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem @cimArg -ErrorAction Stop).TotalPhysicalMemory / 1GB)
	$script:pcInfo = "$($cpu -replace '\s+', ' ')".Trim() + "  |  $ram GB RAM"
	$script:ramGB = [int]$ram
} catch { }
if ($script:cimS) { try { Remove-CimSession $script:cimS } catch { } }; $script:cimS = $null
$script:isNvidia = (-not $script:gpuName) -or ($script:gpuName -match 'NVIDIA|GeForce|RTX|GTX|Quadro')
$script:isAmd = [bool]($script:gpuName -match 'AMD|Radeon')  # (10-24) AMD: su Anti-Lag 2 de serie
# (10-23) Base de datos de graficas: el modo recomendado y lo que se puede esperar. Medido en una RTX 5080 (1440p:
# ~75 reales, 224 en pantalla con frame generation dinamico; 4K cine ~40-48 reales) y en una de 8 GB simulada; el
# resto es una ESTIMACION por potencia relativa (el emulador tambien depende mucho de la CPU).
$script:gpuDb = @(
	# (10-23) Sin VK_KHR_fragment_shader_barycentric (lo exige el emulador para los shaders del juego): no arrancan.
	@('GTX (9|10)[0-9]{2}|GTX TITAN|RX (4|5)[0-9]{2}\b|RX 5[0-9]{3}|Vega|Radeon VII', 'NO COMPATIBLE', 'tu grafica no tiene fragmentShaderBarycentric (hace falta NVIDIA RTX 20 / GTX 16 o AMD RX 6000 o mas nuevas)'),
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
	# AMD FSR 4 (modelo v07 INT8 de AMD, src/local/fsr4-sr.h): el quinto valor es el modelo (KYTY_FSR4_PRESET), uno por
	# preset; el de 1080 o el de 2160 segun la salida. Necesita una grafica con producto escalar INT8 (RTX 20+, RX 6000+).
	@('FSR 4 Nativo (AA)',       1.0, 'fsr4', 0, 'native'),
	@('FSR 4 Calidad',           1.5, 'fsr4', 0, 'quality'),
	@('FSR 4 Equilibrado',       1.7, 'fsr4', 0, 'balanced'),
	@('FSR 4 Rendimiento',       2.0, 'fsr4', 0, 'performance'),
	@('FSR 4 Ultra rendimiento', 3.0, 'fsr4', 0, 'ultraperf'),
	@('FSR 4 Dinámico',          1.5, 'fsr4', 0, 'drs'),
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
	if ($cmbTest -and $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] -eq 'TEST30') { $config = 'sin-fg.json' }  # (10-24) prueba de latencia
	if ($cmbTest -and $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] -eq 'TESTCINE') { $config = 'fsr3.json' }
	if ($cmbTest -and $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] -eq 'TESTNV30') { $config = 'dlss-x2.json' }  # (10-24) prueba: modo Souls ligero con frame generation DLSS x2 + Reflex  # (10-24) prueba: juego en cinematico tal cual + FG x2
	# Rutas entre comillas: Start-Process une los argumentos con espacios, y una carpeta con espacios
	# (C:\Users\Juan Perez\...) partia la ruta y PowerShell se cerraba sin arrancar nada.
	$psargs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$(Join-Path $root 'run-windows.ps1')`"",
	          '-Exe', "`"$(Join-Path $root $exeName)`"", '-Config', "`"$(Join-Path $root "config\$config")`"", '-Fullscreen')
	if ($ram16 -or ($chkPc16.Checked -and (Test-LowRam))) { $psargs += '-Ram16' }
	# (10-24) Perfil seguro de 16 GB (Update-LowRam: 20 GB de RAM o menos, o "Simular 16 GB RAM"): -Ram16 de siempre y
	# KYTY_LOWRAM=1 (los limites de memoria del emulador para 16 GB). Con la simulacion, el emulador cree tener 16 GB.
	$lowRamSets = @()
	if ($ram16 -or (Test-LowRam)) { $lowRamSets += 'KYTY_LOWRAM=1' }
	if ($chkRam16.Checked) { $lowRamSets += 'KYTY_SIMULATE_RAM_MB=16384' }
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
	$sets = @() + $lowRamSets
	# (10-24) Vaciado RAM: cada 5 s el emulador suelta hasta 1 GB de lo que puede rehacer (caches, pools libres) y la
	# memoria que el juego ya libero (KYTY_RAM_FLUSH_SECONDS / KYTY_RAM_FLUSH_MB; una linea "RAM flush" por pasada en el log).
	if (Is-FixedFg) { $sets += $(if ($chkLowEnd45 -and $chkLowEnd45.Checked) { 'KYTY_PAD_LATE_MS=29' } else { 'KYTY_PAD_LATE_MS=43' }) }  # (10-25) Anti-Lag+ incluido siempre en ligero, equilibrado y low profile  # (10-25) Anti-Lag+: 30 fps 43 ms, 45 fps (equilibrado) 29 ms
	# (10-23) Reserva PS5 reducida retirada: el juego pide unos 11,8 GB fijos y con 12 GB fallaban asignaciones  # (10-23) reserva de memoria PS5 reducida
	# Modo TURBO (run-windows -Turbo): caches de video a 4.5 GB (unos 6 GB en total), texturas a un cuarto, detalle bajo
	# del juego y la RAM vaciada cada 15 s por encima de 5 GB: para PCs de 16 GB que se cerraban en zonas pesadas.
	if ($chkTurbo2.Checked) { $psargs += @('-Turbo2', '-Quality', 'low', '-TextureReduce', '2') }
	elseif ($chkLowPc.Checked) { if ($sets -notcontains 'KYTY_SIMULATE_RAM_MB=16384') { $sets += 'KYTY_SIMULATE_RAM_MB=16384' }; $psargs += @('-Turbo', '-Quality', 'low', '-TextureReduce', '2') }
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
	# (10-23 21:30) NVIDIA Reflex sobre la frame generation de AMD crashea tambien sin DLSS (20261023-213004, Fsr3::QueuePresent): nunca.
	$nvAmd  = $false
	# Modos Souls con FG fija (ligero, equilibrado, low profile): Anti-Lag automatico en AMD, y en NVIDIA solo con XeSS o FSR.
	# (10-24) Graficas AMD: su Anti-Lag 2 de serie con la frame generation de AMD (y nada mas); NVIDIA / Intel: el propio.
	# (10-24) Se elige en el desplegable: en AMD, Anti-Lag 2 de serie (por defecto) o el nuestro; en NVIDIA el nuestro
	# (Reflex sobre la frame generation de AMD se cerraba). Modos Souls fijos: el nuestro en Ultra.
	$fixedAl = $script:isAmd -and ($config -like 'fsr3*') -and $cmbAl.SelectedIndex -eq 0 -and -not (Is-FixedFg)
	# (10-23 noche) Anti-Lag propio del emulador (KYTY_LOWLAT): sin Reflex ni FSR3 tocado; con la frame generation de AMD.
	if (Is-FixedFg) { $cmbAl.SelectedIndex = 2 }  # (10-24) siempre Ultra en los modos fijos
	$ownLat = ($cmbAl.SelectedIndex -eq 1 -or $cmbAl.SelectedIndex -eq 2) -and $config -like 'fsr3*'
	# (10-24) Frame generation de NVIDIA (DLSS-G): Reflex (por defecto) o el nuestro; con el nuestro Reflex se apaga.
	$dgOwn = $config -like 'dlss*' -and $cmbAlNv.SelectedIndex -gt 0
	if ($dgOwn) {
		$sets += 'KYTY_DLSSG_REFLEX=off'; $sets += 'KYTY_LOWLAT=1'
		$sets += $(if ($cmbAlNv.SelectedIndex -eq 2) { 'KYTY_LOWLAT_QUEUE=1' } else { 'KYTY_LOWLAT_QUEUE=2' })
	}
	if ($ownLat) { $sets += 'KYTY_LOWLAT=1'; $sets += $(if ($cmbAl.SelectedIndex -eq 2) { 'KYTY_LOWLAT_QUEUE=1' } else { 'KYTY_LOWLAT_QUEUE=2' }) }  # (10-23) Anti-Lag: 2 frames en cola; Ultra: 1
	# (10-23) Para las metricas: si va Anti-Lag y cual.
	Set-IniValue 'antilagon' $(if ($dgOwn) { $(if ($cmbAlNv.SelectedIndex -eq 2) { 'Ultra (Reflex apagado)' } else { 'Normal (Reflex apagado)' }) } elseif ($ownLat) { $(if ($cmbAl.SelectedIndex -eq 2) { 'Ultra' } else { 'Normal' }) } elseif ($nvAmd -or $fixedAl) { if ($script:isNvidia) { 'Reflex (NVIDIA)' } else { 'Anti-Lag 2 (AMD)' } } elseif ($config -like 'dlss*') { 'Reflex (DLSS-G)' } else { 'No' })
	if ($nvAmd -or $fixedAl) {
		$psargs += '-AntiLag'
		# (10-23) Baja latencia: la CPU no prepara un frame por adelantado (~ -18 ms de retraso, algo menos de FPS) y
		# con DLSS Frame Generation, Reflex en modo Boost.
		# (10-23 tarde) KYTY_FRAME_PIPELINE=0 quitado: a 4K bajaba de ~45 a 35 reales (y x3 a 20): limitaba los FPS.
		if ($config -like 'dlss*') { $sets += 'KYTY_DLSSG_REFLEX=boost' }
	}
	if ($cmbRt.SelectedIndex -gt 0) { $psargs += @('-Ssgi', "$([math]::Min($cmbRt.SelectedIndex, 3))") }
	if ($cmbRt.SelectedIndex -eq 4) { $sets += 'KYTY_SSGI_INTENSITY=1.6'; $sets += 'KYTY_SSGI_AO=0.4' }  # (10-25) Extremo: Ultra con mas luz rebotada y mas sombreado de contacto  # (10-24) iluminacion RT (vuelve)
	foreach ($n in 'KYTY_RT_SHADOWS', 'KYTY_RT_REFLECTIONS') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }  # (10-24) RT beta
	if ($cmbRtRf -and $cmbRtRf.SelectedIndex -gt 0) { $env:KYTY_RT_REFLECTIONS = "$($cmbRtRf.SelectedIndex)" }
	foreach ($n in 'KYTY_RT_REFLECTIONS_MIN', 'KYTY_RT_REFLECTIONS_GUIDE_STRENGTH') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
	if ($cmbRtRf -and $cmbRtRf.SelectedIndex -ge 4) { $env:KYTY_RT_REFLECTIONS_MIN = '0.55'; $env:KYTY_RT_REFLECTIONS_GUIDE_STRENGTH = '1.3' }  # (10-25) Ultra: nivel 4 del emulador, mas presencia
	# (10-25 noche) combo22: sol RT, calidad de sombra, profundidad, particulas RT
	foreach ($n in 'KYTY_GTAO', 'KYTY_GTAO_STRENGTH', 'KYTY_GTAO_BENT') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
	# (10-26) Oclusion (GTAO) retirada: nunca se pasa KYTY_GTAO*. Profundidad va al AO de RT real (ver hwMap).
	if ($cmbAo) { $cmbAo.SelectedIndex = 0 }
	# (s25) RT real (hardware): solo Reflejos, Sombras sol y Luces antorcha/lamparas (No/Si). Solo KYTY_HWRT_REFL / _MATERIAL /
	# _SHADOW / _PARTICLES (motor que viene); nada de volumetrica, metal ni AO RT.
	$hwMap = [ordered]@{}
	if ($cmbHRef -and $cmbHRef.SelectedIndex -gt 0) { $hwMap['KYTY_HWRT_REFL'] = '1'; $hwMap['KYTY_HWRT_REFL_MATERIAL'] = '1' }
	$sunOn = ($cmbHSun -and $cmbHSun.SelectedIndex -gt 0)
	if ($sunOn) { $hwMap['KYTY_HWRT_SHADOW'] = '2' }
	if ($cmbHLamp -and $cmbHLamp.SelectedIndex -gt 0) { $hwMap['KYTY_HWRT_PARTICLES'] = '2'; $hwMap['KYTY_HWRT_LIGHTS'] = "$([math]::Min(3, $cmbHLamp.SelectedIndex))"; }
	if ($cmbHGi -and $cmbHGi.SelectedIndex -gt 0) { $hwMap['KYTY_HWRT_GI'] = "$([math]::Min(3, $cmbHGi.SelectedIndex))" }  # (fx6)
	if ($cmbHCon -and $cmbHCon.SelectedIndex -gt 0) { $hwMap['KYTY_HWRT_CONTACT'] = "$([math]::Min(3, $cmbHCon.SelectedIndex))" }  # (fx6)
	if ($cmbHSoft -and $cmbHSoft.SelectedIndex -gt 0) { $hwMap['KYTY_HWRT_SOFTSUN'] = "$([math]::Min(3, $cmbHSoft.SelectedIndex))" }  # (fx6)
	if ($cmbHWater -and $cmbHWater.SelectedIndex -gt 0) { $hwMap['KYTY_HWRT_WATER'] = "$([math]::Min(3, $cmbHWater.SelectedIndex))" }  # (fx6)
	if ($cmbHVolSh -and $cmbHVolSh.SelectedIndex -gt 0) { $hwMap['KYTY_HWRT_VOLSHADOW'] = "$([math]::Min(3, $cmbHVolSh.SelectedIndex))" }  # (fx6)

	$hw = @(foreach ($n in $hwMap.Keys) { "$n=$($hwMap[$n])" })
	# (ds-rt1026-restore) RT real (hardware) solo como experimental: launcher.ini rthwexp=1 (el motor lo exige con KYTY_RT_HW_EXPERIMENTAL=1)
	if ($hw.Count -gt 0 -and (Get-IniValue 'rthwexp' '0') -eq '1') { $sets += 'KYTY_RT_HW_EXPERIMENTAL=1'; $sets += 'KYTY_HWRT=1'; $sets += $hw }
	# (10-24) Sombras RT retiradas
	# (10-25) Reflejos RT retirados definitivamente
	# (10-23) Sombras RT retiradas: el juego tiene las suyas (Sombras Ultra)
	if ($cmbShadow.SelectedIndex -gt 0) { $psargs += @('-Shadows', @('0', '3', '1', '2')[$cmbShadow.SelectedIndex]) }  # (10-23) Alto = 3
	$rN = @('Ultra', '', 'Normal', 'Low'); if ($rN[$cmbRefl.SelectedIndex]) { $psargs += @('-Reflections', $rN[$cmbRefl.SelectedIndex]) }
	if ($rN[$cmbFade.SelectedIndex]) { $psargs += @('-ObjectFade', $rN[$cmbFade.SelectedIndex]) }
	$aN = @('Ultra', 'High', '', 'Low', 'Off'); if ($aN[$cmbAaS.SelectedIndex]) { $psargs += @('-AaSharp', $aN[$cmbAaS.SelectedIndex]) }
	$postOff = @($script:postChecks | Where-Object { -not $_[0].Checked } | ForEach-Object { $_[1] }) -join ','; if ($postOff) { $psargs += @('-PostOff', $postOff) }  # (10-23)
	$detNames = @('Lejana', 'VeryHigh', 'High', 'Normal', 'Low')  # (10-24) Ultra = Lejana: sombras del juego a 1000 m, 4 cascadas, objetos lejanos, SSR largo; if ($cmbDet.SelectedIndex -ne 2) { $psargs += @('-Detail', $detNames[$cmbDet.SelectedIndex]) }  # (10-23) preset de detalle
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
	if ($chkSoulsLow.Checked) { $sharp = '1' } elseif ($chkLowEnd.Checked) { $sharp = '0.75' }  # (10-24) ligero y low profile: nitidez 75%
	if ($sharp -ne '') { $sets += "KYTY_UPSCALE_SHARPNESS=$sharp" }
	# FPS desbloqueados con frame generation: el juego deja de estar limitado a 60 reales (KYTY_FRAMEGEN_BASE=0); los
	# modos Smooth siguen en sus FPS fijos (KYTY_FLIP_FPS) y DLSS-G se limita solo con Reflex al tope del monitor.
	$fgConfig = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	if (Is-FixedFg) { $fgConfig = 'fsr3.json'; $sets += $(if ($chkLowEnd45.Checked) { 'KYTY_FLIP_FPS=45' } else { 'KYTY_FLIP_FPS=30' }) }
	# (10-24) Prueba de latencia: 30 fps bloqueados sin frame generation (temporal).
	if ($cmbTest -and $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] -eq 'TEST30') { $fgConfig = 'sin-fg.json'; $sets += 'KYTY_FLIP_FPS=30' }
	if ($cmbTest -and $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] -eq 'TESTNV30') { $fgConfig = 'dlss-x2.json'; if (-not (Is-FixedFg)) { $sets += 'KYTY_FLIP_FPS=30' } }
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
	if ($cmbLod -and $cmbLod.SelectedIndex -eq 2) {
		# (10-25) Extra nitidas: log2(3D / salida) - 1 con reescalador (guia de NVIDIA para DLSS), -0,5 sin reescalador; tope -2.
		$rq = $script:resShown[[math]::Max(0, $cmbRes.SelectedIndex)]; $rr = 1.0; if ($rq -and $rq[2] -in 'dlss', 'xess', 'fsr') { $rr = [double]$rq[1] }
		$lb = if ($rr -gt 1.0) { -([math]::Log($rr, 2)) - 1.0 } else { -0.5 }; $lb = [math]::Max(-2.0, $lb)
		$sets += ('KYTY_TEXTURE_LOD_BIAS=' + $lb.ToString('0.000', [System.Globalization.CultureInfo]::InvariantCulture))
	} elseif ($chkLod.Checked) { $sets += 'KYTY_TEXTURE_LOD_BIAS=auto' }  # (10-23) solo texturas 2D comprimidas de superficie  # (10-23) texturas con el detalle de la resolucion final
	$sets += 'KYTY_DLSSG_PACE=grid'  # (10-22 e) reflexfree esperaba el limite de Reflex en cada fotograma mostrado: 14 reales
	# (10-26) GTAO retirado del launcher: sin KYTY_GTAO / STRENGTH / BENT
	# (10-23) KYTY_SSR retirado (reflejos del juego en Reflections)
	# (10-23) KYTY_GUEST_LAZY_COMMIT retirado: bloqueaba el arranque y solo ahorraba ~0,5 GB
	# (10-23) KYTY_LOWMEM_VRAM retirado: con graficas de 8 GB bajaba el presupuesto a 4 GB y daba tirones (13-47% de tramos frente a 2-7%)
	$sets += 'KYTY_FAULT_STATS=1'
	$sets += 'KYTY_ALIAS_STATS=0'  # (10-23) el diagnostico de alias recorria todos los mapeos en cada map: tirones al cambiar de zona que empeoraban con la sesion
	if ($fgConfig -eq 'fsr3-x4.json') { $sets += 'KYTY_VRR_CAP_FPS=224' }  # (10-22 d) en DLSS el tope se aplicaba dos veces: 14 reales x4 = 56
	if ($fgConfig -like '*dyn*') { $sets += 'KYTY_DYN_QUALITY=0.95' }  # (10-22 g) llenar el monitor: el menor multiplicador que llega a ~224 (80 reales = x3, 120 = x2)
	# (10-23) Teclado y raton: velocidad de camara del juego (sin teclado y raton no se toca: la del menu del juego).
	# (10-24) Convars extra: velocidad de camara (raton) y la prueba elegida en Ajustes (manchas negras).
	$extraCv = @()
	if ($chkKbm.Checked) { $extraCv += "cp11_Setting_CameraSpeed=$($trkCam.Value)" }
	$tsel = if ($cmbTest) { $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] } else { '' }
	if ($tsel -eq 'TEST30' -or $tsel -eq 'TESTCINE' -or $tsel -eq 'TESTNV30') { $tsel = '' }
	if ($tsel -like 'KYTY_*') { $sets += @($tsel -split ';') } elseif ($tsel) { $extraCv += $tsel }  # (KYTY_*: por -Set, que manda sobre la config)
	if ($extraCv.Count) { $psargs += @('-ExtraConvars', ($extraCv -join ';')) }
	if ($chkMetricas.Checked) { $sets += 'KYTY_LATENCY=1' }
	$sets += 'KYTY_DEPTH_ARRAY_LOG=1'  # (10-24) diagnostico de las sombras de focos
	if (Test-Path 'C:\kyty-live\medir.on') { $sets += 'KYTY_LIVE_FILE=C:\kyty-live\cmd.txt'; $sets += 'KYTY_PROF_HZ=1000' }  # (medir) solo con Medir-rendimiento.cmd
	if (Test-Path (Join-Path $root 'rtcapture.on')) { $sets += 'KYTY_RT_CAPTURE=1' }  # (rtcapture) solo con Capturar-RT.cmd
	if (-not (Test-Path (Join-Path $root 'fsr4_shaders')) -and (Test-Path "$env:USERPROFILE\OneDrive\Desktop\bbport-windows\fsr4_shaders")) { $sets += "KYTY_FSR4_DIR=$env:USERPROFILE\OneDrive\Desktop\bbport-windows\fsr4_shaders" }  # (fsr4) modelo FSR 4 de la carpeta de Bloodborne
	$sets += 'KYTY_LOWLAT_PACE_BEFORE_INPUT=0'; $sets += 'KYTY_FLIP_QUEUE_LIMIT=0'; $sets += 'KYTY_PHASE_PACE=0'; if (Is-FixedFg) { $sets += 'KYTY_DISPLAYED_WAIT=1' }  # (10-25) cola de 1 fotograma como en consola: solo en los modos de 30/45 fijos (desbloqueado lo limitaba a ~90)  # (10-24) el limite de cola daba trompicones (juego sd 11 ms) y no bajaba la latencia: apagado
	$pureGame = ($chkPure -and $chkPure.Checked)  # (audit) Juego original: ningun pase visual anadido
	if ($cmbAoNew -and -not $pureGame) { $sets += "KYTY_AO=$([math]::Max(0, $cmbAoNew.SelectedIndex))" } else { $sets += 'KYTY_AO=0' }  # (ao-1028) oclusion ambiental nueva: No/Bajo/Medio/Alto
	# (ds-shadow-1028) Sombras RT de pantalla (cshadow.comp, nunca RT real): No / Bajo / Medio / Alto = nivel 1 / 2 / 3 a fuerza
	# 0,35 / 0,5 / 0,7. Con la Luz volumetrica y sin Sombras RT el pase sigue a fuerza 0 (solo da el sol a la volumetrica).
	$shLv = if (-not $pureGame -and $cmbRtSh) { [math]::Min([math]::Max(0, $cmbRtSh.SelectedIndex), 3) } else { 0 }
	$raysOn = (-not $pureGame -and $cmbRays -and $cmbRays.SelectedIndex -gt 0)
	if ($raysOn) { $sets += "KYTY_GODRAYS=$($cmbRays.SelectedIndex)" }  # (10-25) luz volumetrica Bajo/Medio/Alto
	if ($shLv -gt 0) {
		$env:KYTY_RT_SHADOWS = "$shLv"; $env:KYTY_RT_SHADOWS_STRENGTH = @('0', '0.35', '0.5', '0.7')[$shLv]
		$sets += @('KYTY_RT_SUN=0', 'KYTY_HWRT_SHADOW=0')  # (solo el camino de pantalla: sin cascadas ni TLAS)
	} elseif ($raysOn) {
		$env:KYTY_RT_SHADOWS = '1'; $env:KYTY_RT_SHADOWS_STRENGTH = '0'
		# (vol25) KYTY_RT_SUN=0: la sombra RT de pantalla a fuerza 0 da el sol a la volumetrica en cada frame (sin niebla que salta).
		$sets += 'KYTY_RT_SUN=0'
	} else { foreach ($n in 'KYTY_RT_SHADOWS', 'KYTY_RT_SHADOWS_STRENGTH') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue } }  # (audit) ni volumetrica ni Sombras RT: nada
	if ($pureGame) {
		# (audit) Juego original: fuera todo pase visual del emulador (RT, volumetrica, sombra RT, brillo, oclusion especular, GTAO,
		# SSGI, reflejos RT de pantalla, contacto, particulas RT); -Set manda sobre el entorno. Rendimiento, reescalador, FG y latencia igual.
		$fxPat = '^KYTY_(HWRT|RT_REFLECTIONS|RT_SHADOWS|RT_SUN|RT_PARTICLES|RT_LOCAL|GODRAYS|SHINE|SPEC_OCCLUSION|GTAO|SSGI|SSR|CONTACT_SHADOWS|AO)'
		$sets = @($sets | Where-Object { $_ -notmatch $fxPat })
		foreach ($n in @(Get-ChildItem Env: | Where-Object { $_.Name -match $fxPat } | ForEach-Object { $_.Name })) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
		$sets += @('KYTY_HWRT=0', 'KYTY_HWRT_REFL=0', 'KYTY_HWRT_SHADOW=0', 'KYTY_HWRT_LIGHTS=0', 'KYTY_HWRT_PARTICLES=0', 'KYTY_HWRT_GI=0', 'KYTY_HWRT_AO=0', 'KYTY_HWRT_CONTACT=0', 'KYTY_HWRT_SOFTSUN=0', 'KYTY_HWRT_WATER=0', 'KYTY_HWRT_VOLSHADOW=0', 'KYTY_RT_REFLECTIONS=0',
			'KYTY_RT_SHADOWS=0', 'KYTY_CONTACT_SHADOWS=0', 'KYTY_RT_SUN=0', 'KYTY_RT_PARTICLES=0', 'KYTY_GODRAYS=0', 'KYTY_SHINE=0', 'KYTY_SPEC_OCCLUSION=0',
			'KYTY_GTAO=0', 'KYTY_SSGI=0', 'KYTY_SSR=0', 'KYTY_AO=0')
		$pa = @(); for ($i = 0; $i -lt $psargs.Count; $i++) { if ($psargs[$i] -eq '-Ssgi') { $i++; continue }; $pa += $psargs[$i] }; $psargs = $pa
	}
	# (10-24) Brillo metales retirado (no aportaba)
	if ($cmbTest -and $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] -eq 'TESTCINE') { $sets = @($sets | Where-Object { $_ -notlike 'KYTY_FLIP_FPS=*' -and $_ -notlike 'KYTY_FRAMEGEN_BASE=*' -and $_ -notlike 'KYTY_LOWLAT*' -and $_ -notlike 'KYTY_DS_*' -and $_ -notlike 'KYTY_VRR_CAP_FPS=*' -and $_ -notlike 'KYTY_DYN_QUALITY=*' }) }  # (10-24) cinematico: el juego manda en sus 30
	if ($cmbTest -and $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] -eq 'TESTNV30') { $sets = @($sets | Where-Object { $_ -notlike 'KYTY_LOWLAT*' -and $_ -ne 'KYTY_DLSSG_REFLEX=off' -and $_ -notlike 'KYTY_FRAMEGEN_BASE=*' }) }  # (10-24) Reflex de NVIDIA encendido
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
	# (10-27) DLSS en Modo Souls ligero / PC low profile (base 1080 / 900): DLSS sale a 4K (camino PS5 Pro) y solo acepta el 3D a
	# 2160 / ratio exacto (1440, 1256, 1080, 720). Con base 1080, Calidad 1.333 daba 810: DLSS lo rechazaba y el juego se subia
	# sin DLSS. Ahora cualquier calidad de DLSS en esos modos va a 720 (1080 / 1.5 = DLSS Calidad de 1080p; 900 / 1.25).
	if ($script:isNvidia -and $res[2] -eq 'dlss' -and (($chkLowEnd -and $chkLowEnd.Checked) -or ($chkSoulsLow -and $chkSoulsLow.Checked))) {
		if (-not ($outH -eq 1080 -and [double]$res[1] -eq 1.0)) { $res = @($res[0], [math]::Round($outH / 720.0, 4), 'dlss', 0) }  # DLAA a 1080 (valido) se queda
	}
	# (10-22) Modo 4K: el modo cinematico del juego a 4K nativo (sin reescalado).
	# (10-22) El modo del juego siempre segun la salida: 4K = cinematico, lo demas = rendimiento (el juego guarda el
	# ultimo en la partida y sin esto se quedaba en cinematico).
	$psargs += @('-ForceMode', $(if (($cmbTest -and $script:testModes[[math]::Max(0, $cmbTest.SelectedIndex)][1] -eq 'TESTCINE') -or $false) { 'cinematic' } else { 'performance' }))  # (10-25) 4K ya no fuerza el cinematico (bloqueaba el juego a 30 reales)
	#  # (10-22) modos Souls: siempre rendimiento
	# 1080p / 720p sin reescalador: el juego dibuja a esa altura.
	if ($outH -lt 1440 -and -not $res[2]) { $res = @("${outH}p", $outH, '', 0) }
	# DLSS (reescalado) carga Streamline de NVIDIA; con la generacion de frames de AMD el emulador se cerraba al crear
	# el dispositivo grafico. Con AMD: FSR o XeSS para reescalar.
	if ($res[2] -like '*dlss' -and $config -like 'fsr3*') {
		# (10-23) Prueba: DLSS + frame generation de AMD permitido (antes se cerraba al crear el dispositivo)
	}
	# DLSS, XeSS and FSR go through the game's PS5 Pro path (its output 4K): the ratio to the monitor's height, the
	# emulator detecting it (KYTY_UPSCALE_RATIO); the others their render height.
	if ($res[2] -in 'dlss', 'xess', 'fsr', 'fsr4') {
		$psargs += @('-UpscaleRatio', ([double]$res[1]).ToString([System.Globalization.CultureInfo]::InvariantCulture))
	} elseif ($res[1] -ne 1440) { $psargs += @('-RenderHeight', "$($res[1])") }
	# "sin TAA": el TAA del juego no se ejecuta; DLSS (DLAA) en su lugar, o (taa-up-*) el reescalador elegido toma la
	# imagen sin filtrar y hace el antialiasing el solo (encima del TAA salia como una acuarela, con estelas).
	if ($res[2] -like 'taa-up-*') { $psargs += @('-Upscale', $res[2].Substring(7), '-TaaReplace', 'upscaler') }
	elseif ($res[2] -like 'taa-*') { $psargs += @('-TaaReplace', $res[2].Substring(4)) }
	elseif ($res[2]) { $psargs += @('-Upscale', $res[2]) }
	if ($res[3] -gt 0) { $psargs += @('-UpscaleHeight', "$($res[3])") }
	$preset = $dlssPresets[[math]::Max(0, $cmbPreset.SelectedIndex)][1]
	if ($res[2] -eq 'fsr4' -and $res.Count -gt 4 -and $res[4]) { $psargs += @('-Fsr4Preset', $res[4]) }  # (FSR 4: el modelo del preset)
	if ($res[2] -like '*dlss' -and $preset) { $psargs += @('-DlssPreset', $preset) }
	if ($chkVram8.Checked) { $env:KYTY_SIMULATE_VRAM_MB = '8192' } else { Remove-Item Env:KYTY_SIMULATE_VRAM_MB -ErrorAction SilentlyContinue }
	# Cinematico a 60 quitado (no funcionaba): el modo cinematico queda como en PS5.
	$env:KYTY_UNLOCK_30 = '0'
	# (10-23) Mando remapeado (KYTY_PAD_MAP) y teclado y raton (KYTY_KBM): ver "Mapear mando". Sin nada, como siempre.
	foreach ($n in 'KYTY_PAD_MAP', 'KYTY_KBM', 'KYTY_KBM_MAP', 'KYTY_KBM_SENS', 'KYTY_KBM_ANTIDZ', 'KYTY_KBM_CALIBRATE', 'KYTY_KBM_WARP', 'KYTY_KBM_NO_RAMP') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
	$padMap = Get-IniValue 'padmap' ''; if ($padMap) { $env:KYTY_PAD_MAP = $padMap }
	# (10-23) "Calibrar camara": el emulador mide cuanto gira la camara del juego con cada posicion del stick y lo guarda
	# en kbm_curve.txt junto al exe; desde entonces el raton gira la camara exactamente lo que se mueve.
	if ($script:kbmCalibrate) { $env:KYTY_KBM_CALIBRATE = '1' }
	if ($chkKbm.Checked -or $script:kbmCalibrate) {
		$env:KYTY_KBM = '1'
		$kbmMap = Get-IniValue 'kbmmap' ''; if ($kbmMap) { $env:KYTY_KBM_MAP = $kbmMap }
		$ks = $trkSens.Value / 100
		if (Is-FixedFg) { $ks = $ks * 0.92 }  # (10-25) un poco menos de sensibilidad en 30x2 / 45x2 (antes 0.85 + velocidad base: giraba lento)
		$env:KYTY_KBM_SENS   = $ks.ToString([System.Globalization.CultureInfo]::InvariantCulture)
		$env:KYTY_KBM_ANTIDZ = ($trkAdz.Value / 100).ToString([System.Globalization.CultureInfo]::InvariantCulture)
		if ($chkWarp.Checked) { $env:KYTY_KBM_WARP = '1' } else { Remove-Item Env:KYTY_KBM_WARP -ErrorAction SilentlyContinue }
	}
	foreach ($n in 'KYTY_VSYNC_LOCK', 'KYTY_FPS_CAP', 'KYTY_FLIP_WHEN_READY') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
	# G-Sync: reloj del emulador suelto del monitor. Mostrar cada frame al momento solo con FPS desbloqueados y sin
	# frame generation; con el juego a 60 (o con frame generation) el ritmo de 60 lo marcan los vblanks y no se toca.
	if ($chkGsync.Checked) {
		$env:KYTY_VSYNC_LOCK = '0'
		if ((Is-Unlocked) -and $config -eq 'sin-fg.json') { $env:KYTY_FPS_CAP = 'auto'; $env:KYTY_FLIP_WHEN_READY = '1' }
	}
	# (10-24) Ajustes del juego (ventana "Ajustes del juego"): en KYTY_GAME_CONVARS_EXTRA, que el emulador lee despues de
	# KYTY_GAME_CONVARS (todo lo de run-windows): mandan. La variable que tuviera el usuario va detras (sus pruebas mandan).
	$gsAll = @((Get-GameSettingsConvars), $script:userCvExtra) | Where-Object { $_ }
	if ($gsAll) { $env:KYTY_GAME_CONVARS_EXTRA = ($gsAll -join ';') } else { Remove-Item Env:KYTY_GAME_CONVARS_EXTRA -ErrorAction SilentlyContinue }
	Start-Process -FilePath 'powershell' -ArgumentList $psargs -WorkingDirectory $root -WindowStyle Hidden | Out-Null
	# (10-26) Equipos de 16 GB: limpiador de memoria en espera (standby) en tiempo real, como administrador. Si se rechaza el UAC, el juego sigue.
	$metArgs = @(); if ($chkRam16.Checked) { $metArgs = @('-RamGB', '16') }
	if ($chkVram8.Checked) { $metArgs += @('-VramMB', '8192') }
	# Como administrador solo con el limite de RAM (lo necesita); si no, normal (no molesta a la grabacion de NVIDIA).
	$metVerb = if ($chkRam16.Checked) { @{ Verb = 'RunAs' } } else { @{} }
	if (-not $chkMetricas.Checked) { $metArgs += '-NoOverlay' }
	if ($chkMetricas.Checked -or $chkRam16.Checked) { Start-Process -FilePath 'powershell' -ArgumentList (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', ('"' + (Join-Path $root 'metricas.ps1') + '"'), '-LogDir', ('"' + (Join-Path $root 'logs') + '"')) + $metArgs) -WindowStyle Hidden @metVerb -ErrorAction SilentlyContinue | Out-Null }
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
# (10-27 startup) Sin parpadeo al abrir: invisible (Opacity 0) hasta que todo esta colocado; doble buffer.
$form.Opacity = 0
function Set-DoubleBuffer($c) { try { $c.GetType().GetProperty('DoubleBuffered', [System.Reflection.BindingFlags]'Instance,NonPublic').SetValue($c, $true, $null) } catch { } }
Set-DoubleBuffer $form
$form.SuspendLayout()
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
$subFont    = New-Object System.Drawing.Font('Segoe UI', 12.5)
$buttonFont = New-Object System.Drawing.Font('Segoe UI Semibold', 15, [System.Drawing.FontStyle]::Bold)
$smallFont  = New-Object System.Drawing.Font('Segoe UI', 11)  # (10-26) 10 -> 11
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
$panel.SuspendLayout(); Set-DoubleBuffer $panel
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
$cmbFg.ItemHeight = 26
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
$tips.SetToolTip($chkRam16, 'Solo el juego: como mucho 12 GB en RAM (4 GB para Windows), el resto a disco, como un PC de 16 GB.')
$panel.Controls.Add($chkRam16)
$chkVram8 = New-Check 'Simular grafica 8 GB' 260 4 220 'vram8sim'
$tips.SetToolTip($chkVram8, 'El juego solo puede usar 8 GB de VRAM, como una grafica de 8 GB.')
$panel.Controls.Add($chkVram8)
$chkRamFlush = New-Check 'Vaciado RAM (4 GB)' 0 0 190 'ramflush'
$tips.SetToolTip($chkRamFlush, 'Cada 5 s el emulador devuelve a Windows hasta 2 GB que puede rehacer (caches y memoria libre) y la que el juego ya solto. No toca la memoria del juego. Prueba para PCs de 16 GB: mira las lineas "RAM flush" del log.')
$panel.Controls.Add($chkRamFlush)
$chkRamFlush2 = New-Check 'Vaciado RAM fuerte' 0 0 200 'ramflush2'
$tips.SetToolTip($chkRamFlush2, 'Como el vaciado normal y ademas suelta las texturas que no se han usado en 2 minutos y aparta de la RAM las que solo se leen. Libera mas, a cambio de que alguna textura se vuelva a cargar al regresar a un sitio.')
$panel.Controls.Add($chkRamFlush2)
$chkRamFlush2.Add_CheckedChanged({ if ($chkRamFlush2.Checked) { $chkRamFlush.Checked = $false } })
$chkRamFlush.Add_CheckedChanged({ if ($chkRamFlush.Checked -and $chkRamFlush2) { $chkRamFlush2.Checked = $false } })
# (10-25) Latencia minima 30x2: lee el mando 43 ms antes de que salga el fotograma (solo con limite de 30). Menos latencia,
# menos margen para picos.
$chkPadLate = New-Check 'Anti-Lag+' 0 0 206 'padlate'
$tips.SetToolTip($chkPadLate, 'Anti-Lag+: el juego lee el mando lo mas tarde posible antes de dibujar. Modos de 30 FPS con frame generation (30x2): unos 61 ms en vez de unos 79. Modo Souls equilibrado (45x2): tambien baja. En los modos desbloqueados no hace nada. Si notas tirones, quitalo.')
$panel.Controls.Add($chkPadLate)
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
$boldSmall = New-Object System.Drawing.Font('Segoe UI Semibold', 11, [System.Drawing.FontStyle]::Bold)  # (10-26) 10 -> 11
function Get-BrandColor([string]$text) {
	if ($text -match 'NVIDIA|DLSS')      { return [System.Drawing.Color]::FromArgb(118, 185, 0) }
	if ($text -match 'Intel|XeSS')       { return [System.Drawing.Color]::FromArgb(70, 160, 255) }
	if ($text -match 'AMD|FSR')          { return [System.Drawing.Color]::FromArgb(237, 70, 70) }
	if ($text -match 'juego|Del juego')  { return [System.Drawing.Color]::FromArgb(200, 200, 200) }
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
	[System.Windows.Forms.TextRenderer]::DrawText($e.Graphics, $label, $font, (New-Object System.Drawing.Rectangle(($e.Bounds.X + $x), $e.Bounds.Y, ($e.Bounds.Width - $x), $e.Bounds.Height)), $color, ([System.Windows.Forms.TextFormatFlags]::VerticalCenter -bor [System.Windows.Forms.TextFormatFlags]::NoPrefix -bor [System.Windows.Forms.TextFormatFlags]::EndEllipsis))  # (10-26) GDI/ClearType como el resto
	$textBrush.Dispose()
}

# Reescalado por apartados (NVIDIA DLSS, Intel XeSS, AMD FSR, el juego), sin mezclar con la generacion de frames:
# con la de NVIDIA solo DLSS; con la de AMD, FSR y XeSS; sin generacion, todos.
$script:resShown = @()
function Get-ResFamilies {
	# (10-22) Modo graficas de 8 GB: frame generation de AMD, asi que XeSS y FSR (aunque la lista tenga uno de NVIDIA).
	if ($chkLowEnd -and $chkLowEnd.Checked) { return @('dlss', 'xess', 'fsr', 'fsr4') }  # (10-23) DLSS tambien (va con el FG de AMD)
	# (10-22) Modo Souls equilibrado: frame generation de AMD, asi que sin DLSS (XeSS, FSR o el del juego).
	if ($chkLowEnd45 -and $chkLowEnd45.Checked) { return @('', 'dlss', 'xess', 'fsr', 'fsr4', 'juego') }  # (10-23) DLSS tambien
	$cfg = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	if ($cfg -like 'dlss*') { return @('', 'dlss', 'juego') }
	if ($cfg -like 'fsr3*') { return @('', 'dlss', 'xess', 'fsr', 'fsr4', 'juego') }  # (10-23) DLSS + frame generation de AMD (prueba)
	return @('', 'dlss', 'xess', 'fsr', 'fsr4', 'juego')
}
function Get-OutHeight {
	# (10-22) Modo graficas de 8 GB: siempre 1080p.
	if ($chkSoulsLow -and $chkSoulsLow.Checked) { return 900 }  # (10-23) PC low profile: salida de 900, FSR 3.1 AA nativo a 900
	if ($chkLowEnd -and $chkLowEnd.Checked) { return 1080 }
	# (10-22) Modo 45 FPS: siempre 1440p.
	if ($chkLowEnd45 -and $chkLowEnd45.Checked) { return 1440 }
	if ($cmbOut) { switch ($cmbOut.SelectedIndex) { 1 { return 2160 } 2 { return 1080 } 3 { return 720 } } }
	if ($chk4K -and $chk4K.Checked) { return 2160 }
	return 1440
}
function Fill-Res {
	$wanted = Get-IniValue 'scalelabel' '1440p nativo'
	$titles = @{ 'dlss' = '--- NVIDIA DLSS ---'; 'xess' = '--- Intel XeSS ---'; 'fsr' = '--- AMD FSR 3.1 ---'; 'fsr4' = '--- AMD FSR 4 ---'; 'juego' = '--- Del juego ---' }
	$shown  = @()
	# (10-24) Modos Souls ligero y low profile: sin reescalado (los AA de reescalado gastan mucha memoria), fijo.
	if ($chkSoulsLow -and $chkSoulsLow.Checked) { $shown += , @('Sin reescalado (900p nativo)', 900, '', 0) }
	elseif ($chkLowEnd -and $chkLowEnd.Checked) { $shown += , @('Sin reescalado (1080p nativo)', 1080, '', 0) }
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
		$modes = @($modes | Where-Object { $_[0] -ne 'FSR 900p (AA)' })  # (10-23) ya no se usa: low profile = FSR AA nativo con salida de 900
		# (10-23) Modo Souls ligero: solo el antialiasing de cada uno (DLSS DLAA, XeSS AA, FSR AA) o nativo.
		# (10-25) ligero y low profile: todas las calidades de reescalado (antes solo el AA)
		# (10-22) Modo graficas de 8 GB: tambien sin reescalado (1080p nativo, sin XeSS ni FSR), lo primero de la lista.
		$outH = Get-OutHeight
		$lowAll = ($chkLowEnd -and $chkLowEnd.Checked) -or ($chkSoulsLow -and $chkSoulsLow.Checked)  # (10-25) ligero y low profile: todas las calidades de DLSS (Ultra rendimiento incluido)
		if ($family -eq 'dlss' -and -not $lowAll) { $modes = @($modes | Where-Object { $h = [math]::Round($outH / $_[1]); $h -ge 1080 -or $h -eq 720 }) }
		elseif ($family -in 'xess', 'fsr' -and $outH -lt 1440 -and -not $lowAll) { $modes = @($modes | Where-Object { $outH / $_[1] -ge 719 }) }
		elseif ($family -eq 'fsr4') { $modes = @($modes | Where-Object { $outH / $_[1] -ge 359 }) }  # (10-28) FSR 4: el 3D a 360 como minimo (su modelo sale al tamano de la pantalla: en 1080p Equilibrado 635, Rendimiento 540, Ultra rendimiento 360)
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
$cmbRes.ItemHeight = 24
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
$cmbRes.Add_SelectedIndexChanged({ $r0 = $script:resShown[[math]::Max(0, $cmbRes.SelectedIndex)]; $base = 'Reescalado: dibuja el 3D a menos resolucion y lo sube a 1440 (mas fps). DLSS: NVIDIA RTX. XeSS y FSR: todas las graficas. ''juego'': lo sube el propio juego.'; if ($r0 -and $r0[2] -eq 'dlss' -and (($chkLowEnd -and $chkLowEnd.Checked) -or ($chkSoulsLow -and $chkSoulsLow.Checked))) { $tips.SetToolTip($cmbRes, $base + ' En Modo Souls ligero / PC low profile DLSS siempre dibuja el 3D a 720p (DLSS Calidad de 1080p): es lo unico que DLSS acepta en su salida 4K.') } else { $tips.SetToolTip($cmbRes, $base) } })  # (10-27)
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
$chkLowEnd.Add_CheckedChanged({ if ($chkLowEnd.Checked) { $chkLowEnd45.Checked = $false }; Update-LowEnd; if (Get-Command Update-AntiLagBox -ErrorAction SilentlyContinue) { Update-AntiLagBox } })
$chkLowEnd45.Add_CheckedChanged({ if ($chkLowEnd45.Checked) { $chkLowEnd.Checked = $false }; Update-LowEnd; if (Get-Command Update-AntiLagBox -ErrorAction SilentlyContinue) { Update-AntiLagBox } })
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
$form.Controls.Add($lbPc); $lbPc.BringToFront(); $lbPc.Visible = $false  # (10-24) ahora en la tarjeta 'Tu equipo'
if ($cmbPreset -and -not $script:isNvidia) { $cmbPreset.Enabled = $false }
$panel.Controls.Add($btnExit)
$chk4K.Visible = $false
# (10-22) Salida: 1440p (el monitor), 4K (modo cinematico del juego), 1080p y 720p (graficas mas modestas). Los
# reescaladores calculan su resolucion sobre esa altura.
$lbOut = New-Label 'Resolucion' $smallFont $ink 925 6 60 22; $panel.Controls.Add($lbOut)
$cmbOut = New-Object System.Windows.Forms.ComboBox
$cmbOut.DropDownStyle = 'DropDownList'; $cmbOut.Font = $smallFont
$cmbOut.Location = New-Object System.Drawing.Point(985, 2); $cmbOut.Size = New-Object System.Drawing.Size(130, 26)
foreach ($l in '1440p', '4K', '1080p', '720p') { [void]$cmbOut.Items.Add($l) }
$oi = 0; [void][int]::TryParse((Get-IniValue 'salida' $(if ($chk4K.Checked) { '1' } else { '0' })), [ref]$oi)
$cmbOut.SelectedIndex = [math]::Min([math]::Max(0, $oi), 3)
$chk4K.Checked = ($cmbOut.SelectedIndex -eq 1)
$cmbOut.Add_SelectedIndexChanged({ Set-IniValue 'salida' "$($cmbOut.SelectedIndex)"; $chk4K.Checked = ($cmbOut.SelectedIndex -eq 1) -and -not (Is-FixedFg); Fill-Res })
$tips.SetToolTip($cmbOut, 'Resolucion de salida: 1440p, 4K, 1080p o 720p (estas dos para graficas mas modestas). Los reescaladores calculan su resolucion sobre esta.')
$panel.Controls.Add($cmbOut)
# (10-22) Iluminacion RT (rayos sobre la imagen: luz rebotada y sombreado de contacto). Solo con DLSS, XeSS o FSR.
$lbRt = New-Label 'Luz RT' $smallFont $ink 1125 6 60 22; $panel.Controls.Add($lbRt)
$cmbRt = New-Object System.Windows.Forms.ComboBox
$cmbRt.DropDownStyle = 'DropDownList'; $cmbRt.Font = $smallFont
$cmbRt.Location = New-Object System.Drawing.Point(1185, 2); $cmbRt.Size = New-Object System.Drawing.Size(85, 26)
foreach ($l in 'No', 'Media', 'Alta', 'Ultra', 'Extremo') { [void]$cmbRt.Items.Add($l) }
$ri = 0; [void][int]::TryParse((Get-IniValue 'ssgi' '0'), [ref]$ri)
$cmbRt.SelectedIndex = [math]::Min([math]::Max(0, $ri), 4)
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
$gPc  = New-Group 'Rendimiento / PC' 450 294 420 118  # (10-24) +28: fila del vaciado de RAM
# Imagen: salida, reescalado, preset y nitidez.
Move-To $lbOut $gImg 10 26 112
Move-To $cmbOut $gImg 124 24 282
$lbRes = New-Label 'Reescalado' $smallFont $ink 0 0 88 22; Move-To $lbRes $gImg 10 58 112
Move-To $cmbRes $gImg 124 56 282
$lbPreset = New-Label 'Preset DLSS' $smallFont $ink 0 0 88 22; Move-To $lbPreset $gImg 10 90 112
Move-To $cmbPreset $gImg 124 88 282
$lbSharp = New-Label 'Nitidez' $smallFont $ink 0 0 88 22; Move-To $lbSharp $gImg 10 122 112
Move-To $cmbSharp $gImg 124 120 282
# Frame generation: modo, FPS desbloqueados y Anti-Lag.
Move-To $cmbFg $gFg 12 24 396
Move-To $chkUnlock $gFg 12 60 200
$chkUnlock.Visible = $false  # (10-22) sin casilla: siempre desbloqueados salvo DLSS y modos Souls
Move-To $chkAntiLag $gFg 222 60 186
# (10-23 noche) Visible solo con NVIDIA: "Anti-Lag (experimental)" para la frame generation de AMD sin DLSS.
$chkAntiLag.Text = 'Anti-Lag'
# (10-23) Desplegable Anti-Lag: No / Anti-Lag (2 frames en cola) / Anti-Lag Ultra (1), en el sitio de la casilla.
$cmbAl = New-Object System.Windows.Forms.ComboBox
$cmbAl.DropDownStyle = 'DropDownList'; $cmbAl.Font = $smallFont
# (10-24) AMD: Anti-Lag 2 de serie (por defecto), el nuestro, el nuestro Ultra o ninguno. NVIDIA: ninguno / el nuestro / Ultra.
$script:alKey = if ($script:isAmd) { 'antilagamd' } else { 'antilagmode2' }
$alItems = if ($script:isAmd) { @('Anti-Lag 2 (AMD)', 'Anti-Lag', 'Anti-Lag Ultra', 'Anti-Lag: no') } else { @('Anti-Lag: no', 'Anti-Lag', 'Anti-Lag Ultra') }
foreach ($l in $alItems) { [void]$cmbAl.Items.Add($l) }
$ai2 = if ($script:isAmd) { 0 } else { 1 }; [void][int]::TryParse((Get-IniValue $script:alKey "$ai2"), [ref]$ai2); $cmbAl.SelectedIndex = [math]::Min([math]::Max(0, $ai2), $cmbAl.Items.Count - 1)
$cmbAl.Add_SelectedIndexChanged({ if (-not $script:alLocked) { Set-IniValue $script:alKey "$($cmbAl.SelectedIndex)" } })
$gFg.Controls.Add($cmbAl); $cmbAl.Location = $chkAntiLag.Location; $cmbAl.Size = New-Object System.Drawing.Size(186, 26)
# (10-24) Con la frame generation de NVIDIA (DLSS-G) va siempre NVIDIA Reflex: se muestra para que se sepa.
$lbReflex = New-Label 'Reflex: activado (DLSS-G)' $smallFont ([System.Drawing.Color]::FromArgb(120, 210, 120)) 0 0 186 24
$lbReflex.TextAlign = 'MiddleLeft'; $gFg.Controls.Add($lbReflex); $lbReflex.Location = $chkAntiLag.Location; $lbReflex.Visible = $false
$tips.SetToolTip($lbReflex, 'La frame generation de NVIDIA (DLSS-G) lleva NVIDIA Reflex siempre activado: menos retraso de forma automatica.')
# (10-24) Con DLSS-G: elegir Reflex (por defecto) o nuestro Anti-Lag (Reflex apagado), para comparar.
$cmbAlNv = New-Object System.Windows.Forms.ComboBox
$cmbAlNv.DropDownStyle = 'DropDownList'; $cmbAlNv.Font = $smallFont
foreach ($l in 'Reflex (NVIDIA)', 'Anti-Lag', 'Anti-Lag Ultra') { [void]$cmbAlNv.Items.Add($l) }
$an = 0; [void][int]::TryParse((Get-IniValue 'antilagdlssg' '0'), [ref]$an); $cmbAlNv.SelectedIndex = [math]::Min([math]::Max(0, $an), 2)
$cmbAlNv.Add_SelectedIndexChanged({ Set-IniValue 'antilagdlssg' "$($cmbAlNv.SelectedIndex)" })
$gFg.Controls.Add($cmbAlNv); $cmbAlNv.Location = $chkAntiLag.Location; $cmbAlNv.Size = New-Object System.Drawing.Size(186, 26); $cmbAlNv.Visible = $false
$tips.SetToolTip($cmbAlNv, 'Con la frame generation de NVIDIA (DLSS-G). Reflex: el de NVIDIA (recomendado). Anti-Lag / Ultra: el nuestro con Reflex apagado, para comparar cual va mejor. Sin Reflex DLSS-G puede ir con otro ritmo o avisar de que falta Reflex.')
# Solo se puede marcar con la frame generation de AMD (con la de NVIDIA ya va Reflex); fuera de ella se desmarca.
function Update-AntiLagBox {
	$cfgNow = $fgModes[[math]::Max(0, $cmbFg.SelectedIndex)][1]
	$resSel = if ($script:resShown.Count) { $script:resShown[[math]::Max(0, $cmbRes.SelectedIndex)] } else { $null }
	$ok = $cfgNow -like 'fsr3*' -or (Is-FixedFg)  # (10-23 noche) anti-lag propio: con la frame generation de AMD, cualquier reescalado
	# (10-23) Modos Souls con FG fija: se ve marcada y gris cuando se activa sola (AMD siempre; NVIDIA con XeSS o FSR).
	$autoAl = $false
	if ($autoAl) { $chkAntiLag.Checked = $true; $chkAntiLag.Enabled = $false; $chkAntiLag.Visible = $true; $script:alAuto = $true; return }
	if ($script:alAuto) { $chkAntiLag.Checked = $false; $script:alAuto = $false }
	$chkAntiLag.Enabled = $true
	if (-not $ok) { $chkAntiLag.Checked = $false }
	$chkAntiLag.Visible = $false; $cmbAl.Visible = $ok
	# (10-24) Modos Souls con FG fija (ligero, equilibrado, low profile): Anti-Lag Ultra siempre, sin poder cambiarlo.
	if (Is-FixedFg) {
		if (-not $script:alLocked) { $script:alLocked = $true; $cmbAl.SelectedIndex = 2 }
		$cmbAl.Enabled = $false
	} elseif ($script:alLocked) {
		$script:alLocked = $false; $cmbAl.Enabled = $true
		$v = if ($script:isAmd) { 0 } else { 1 }; [void][int]::TryParse((Get-IniValue $script:alKey "$v"), [ref]$v); $cmbAl.SelectedIndex = [math]::Min([math]::Max(0, $v), $cmbAl.Items.Count - 1)
	}
	$lbReflex.Visible = $false
	$cmbAlNv.Visible = (-not $ok) -and $cfgNow -like 'dlss*'
}
$cmbFg.Add_SelectedIndexChanged({ Update-AntiLagBox })
$cmbRes.Add_SelectedIndexChanged({ Update-AntiLagBox })
# (10-27 startup) se aplica antes de mostrar la ventana (ver el final)
$tips.SetToolTip($cmbAl, 'Con la frame generation de AMD. Anti-Lag 2 (AMD): el de serie de las graficas AMD. Anti-Lag: el nuestro, como mucho 2 frames en cola antes de leer el mando. Ultra: 1 (aun menos retraso). En los modos Souls ligero, equilibrado y low profile va siempre el nuestro en Ultra.')
# Calidad grafica: texturas, distancia, sombras, Luz RT y filtrado x16 ($script:gfxTop: etiqueta y desplegable de Texturas, Sombras, Distancia).
Move-To $script:gfxTop[0] $gGfx 10 26 112; Move-To $cmbTex $gGfx 124 24 282
Move-To $script:gfxTop[4] $gGfx 10 58 112; Move-To $cmbDist $gGfx 124 56 282
Move-To $script:gfxTop[2] $gGfx 10 90 112; Move-To $cmbShadow $gGfx 124 88 282
Move-To $lbRt $gGfx 10 122 112; Move-To $cmbRt $gGfx 124 120 282
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
foreach ($l in 'No', 'Normal', 'Alta', 'Ultra', 'Extremo') { [void]$cmbAo.Items.Add($l) }
$iAo = 0; [void][int]::TryParse((Get-IniValue 'gtao' '0'), [ref]$iAo)
$cmbAo.SelectedIndex = 0  # (10-26) Oclusion (GTAO) retirada: siempre No y oculta
$cmbAo.Add_SelectedIndexChanged({ Set-IniValue 'gtao' "$($cmbAo.SelectedIndex)" })
$tips.SetToolTip($cmbAo, 'Oclusion ambiental avanzada (GTAO): rincones, grietas y contactos mas realistas. Solo con DLSS, XeSS o FSR. Cuesta FPS. Experimental.')
Move-To $lbAo $gGfx 10 186 112; Move-To $cmbAo $gGfx 124 184 282
$lbAo.Visible = $false; $cmbAo.Visible = $false
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
	Move-To $lb $gGfx 10 ($y + 2) 112; Move-To $c $gGfx 124 $y 282
	return $c
}
$cmbRefl = New-GameCombo 'Reflections' 'greflect' @('Ultra', 'High (default)', 'Normal', 'Low') 1 216 'Reflejos del propio juego (agua, charcos, suelos): mas pasos y mas alcance en Ultra. Cuesta algo de GPU.'
$cmbFade = New-GameCombo 'Object Fade' 'gfade' @('Ultra', 'High (default)', 'Normal', 'Low') 1 280 'Distancia a la que los objetos lejanos se desvanecen. Ultra: aparecen de golpe mucho menos.'
$cmbAaS = New-GameCombo 'AA Sharpness' 'gaasharp' @('Ultra', 'High', 'Normal (default)', 'Low', 'Off') 2 312 'Nitidez del antialiasing del juego. No cuesta FPS.'
Move-To $chkAf16 $gGfx 10 248 200
Move-To $chkLod $gGfx 216 248 190  # (10-23) vuelve: decide por textura (no toca el humo)
# (10-25) Texturas: Normal / Nitidas / Extra nitidas (desplegable en lugar de la casilla; la casilla queda oculta y sincronizada).
$cmbLod = New-Object System.Windows.Forms.ComboBox
$cmbLod.DropDownStyle = 'DropDownList'; $cmbLod.Font = $smallFont
foreach ($it in @('Texturas: normales', 'Texturas: nitidas', 'Texturas: extra nitidas')) { [void]$cmbLod.Items.Add($it) }
$lv0 = 1; [void][int]::TryParse((Get-IniValue 'lodbias' '1'), [ref]$lv0); $cmbLod.SelectedIndex = [math]::Min([math]::Max(0, $lv0), 2)
$cmbLod.Add_SelectedIndexChanged({ Set-IniValue 'lodbias' "$($cmbLod.SelectedIndex)"; $chkLod.Checked = $cmbLod.SelectedIndex -ge 1 })
$tips.SetToolTip($cmbLod, 'Nitidez de las texturas. Nitidas: el detalle de la resolucion final con reescalado. Extra nitidas: un escalon mas (lo que recomienda NVIDIA para DLSS); mas detalle en suelos y paredes, puede brillar algo en rejas y hierba. Cuesta pocos FPS.')
$panel.Controls.Add($cmbLod); Move-To $cmbLod $gGfx 216 246 190
$chkLod.Visible = $false; $chkLod.Checked = $cmbLod.SelectedIndex -ge 1
$lbDet = New-Label 'Detail' $smallFont $ink 0 0 88 22; $panel.Controls.Add($lbDet)
$cmbDet = New-Object System.Windows.Forms.ComboBox
$cmbDet.DropDownStyle = 'DropDownList'; $cmbDet.Font = $smallFont
foreach ($d in @('Ultra', 'Very High (cinematic)', 'High (PS5 performance)', 'Normal', 'Low')) { [void]$cmbDet.Items.Add($d) }
$iDet = 2; [void][int]::TryParse((Get-IniValue 'detail' '2'), [ref]$iDet); $cmbDet.SelectedIndex = [math]::Min([math]::Max(0, $iDet), 4)
$cmbDet.Add_SelectedIndexChanged({ Set-IniValue 'detail' "$($cmbDet.SelectedIndex)" })
$tips.SetToolTip($cmbDet, 'Detalle del juego: sombras (cascadas y distancia), iluminacion de entorno y relieve (teselado). Ultra > Very High (modo cine) > High (modo rendimiento de PS5) > Normal > Low (graficas modestas).')
$panel.Controls.Add($cmbDet)
Move-To $lbDet $gGfx 10 154 112; Move-To $cmbDet $gGfx 124 152 282
# Rendimiento / PC (chkLowPc no se muestra: sigue fuera de la ventana como antes).
Move-To $chkLowEnd $gModes 12 24 186
Move-To $chkLowEnd45 $gModes 202 24 206
Move-To $chkPc16 $gPc 12 24 190
$chkSafeGpu.Visible = $false; $chkSafeGpu.Checked = $false
Move-To $chkRam16 $gPc 12 54 190
Move-To $chkVram8 $gPc 202 54 206
Move-To $chkRamFlush $gPc 12 84 190; Move-To $chkPadLate $gPc 202 84 206; $chkPadLate.Visible = $false; Move-To $chkRamFlush2 $gPc 202 84 206  # (10-25) Anti-Lag+ ya va incluido en los modos 30x2 / 45x2
# (10-24) PCs de 20 GB de RAM o menos (o con "Simular 16 GB RAM"): el perfil seguro de 16 GB solo (la casilla del equipo
# de 16 GB marcada y fija, KYTY_LOWRAM=1) y un aviso de una linea. Lo que el usuario tenia marcado se guarda aparte: al
# quitar la simulacion vuelve tal cual (mientras esta forzada la casilla escribe en otra clave del ini).
function Test-LowRam { return ($chkRam16.Checked -or ($script:ramGB -gt 0 -and $script:ramGB -le 20)) }
$script:pc16User = $chkPc16.Checked
$lbRamNote = New-Label '' (New-Object System.Drawing.Font('Segoe UI', 9)) $gold 0 0 212 34
$panel.Controls.Add($lbRamNote)
Move-To $lbRamNote $gPc 202 19 212  # (hasta dos lineas cortas, sin tapar la fila de abajo)
$tips.SetToolTip($lbRamNote, 'Con 16 GB de RAM el juego va justo de memoria: el launcher usa el perfil seguro de 16 GB (menos cosas guardadas en RAM, sin bajar la resolucion). Cierra el navegador (Chrome, Edge, Firefox), Discord y otros launchers antes de jugar.')
function Update-LowRam {
	if (Test-LowRam) {
		if ($chkPc16.Enabled) { $script:pc16User = $chkPc16.Checked }
		$chkPc16.Tag = 'pc16forced'; $chkPc16.Checked = $true; $chkPc16.Enabled = $false
		$ramTxt = if ($chkRam16.Checked) { '16 GB (simulado)' } else { "$($script:ramGB) GB" }
		$lbRamNote.Text = "RAM $($ramTxt): perfil seguro. Cierra navegadores"
		$lbRamNote.Visible = $true
	} else {
		if (-not $chkPc16.Enabled) { $chkPc16.Enabled = $true; $chkPc16.Checked = [bool]$script:pc16User; $chkPc16.Tag = 'pc16' }
		$lbRamNote.Visible = $false
	}
}
Update-LowRam
$chkRam16.Add_CheckedChanged({ Update-LowRam })
# (10-23) Optimizacion tarjetas de 8 GB: texturas bajo, distancia y sombras medio (7,4 -> 6,3 GB de VRAM, tirones 26% -> 4%).
$chkGpu8 = New-Check 'Optimizacion tarjetas 8 GB' 0 0 206 'gpu8opt'
$tips.SetToolTip($chkGpu8, 'Graficas de 8 GB: pone texturas en Bajo y distancia y sombras en Medio para que el juego quepa en la memoria de la grafica.')
$panel.Controls.Add($chkGpu8)
Move-To $chkGpu8 $gPc 202 24 206
$chkGpu8.Visible = $false; $chkGpu8.Checked = $false  # (10-23) retirada: hacia lo mismo que los graficos del Modo Souls ligero
# (10-23) Presets: Souls ligero = Optimizacion 8 GB (texturas bajo, distancia y sombras medio, Luz RT no, Detail Normal,
# Oclusion Normal); Souls equilibrado (todo Maximo, Luz RT no, Detail High, Oclusion Alta). Al quitarlos: todo Maximo.
function Set-PresetLight { $cmbTex.SelectedIndex = 2; $cmbDist.SelectedIndex = 1; $cmbShadow.SelectedIndex = 2; $cmbRt.SelectedIndex = 0; $cmbDet.SelectedIndex = 3; $cmbAo.SelectedIndex = 1; $cmbAaS.SelectedIndex = 1; $cmbFade.SelectedIndex = 2; $cmbRefl.SelectedIndex = 2 }
function Set-PresetMax { $cmbTex.SelectedIndex = 0; $cmbDist.SelectedIndex = 0; $cmbShadow.SelectedIndex = 0; $cmbDet.SelectedIndex = 2; $cmbAaS.SelectedIndex = 2; $cmbFade.SelectedIndex = 1; $cmbRefl.SelectedIndex = 1 }
# (10-23) Modo Souls ultra y Modo Souls extremo (graficas potentes).
$chkSoulsUltra = New-Check 'Modo Souls ultra' 0 0 186 'soulsultra'
$tips.SetToolTip($chkSoulsUltra, 'Para RTX 5060 / 5060 Ti / 5070 / 5070 Ti y RX 9070 / 9070 XT: todo al maximo, Detail Very High, Luz RT y Oclusion en Alta. Los FPS dependen de la potencia de tu GPU y tu CPU.')
$panel.Controls.Add($chkSoulsUltra); Move-To $chkSoulsUltra $gModes 12 54 186
$chkSoulsExt = New-Check 'Modo Souls extremo' 0 0 206 'soulsext'
$tips.SetToolTip($chkSoulsExt, 'Para enfermos de Souls. RTX 5080, RTX 5090 o superior: modo cinematico del juego a 4K y absolutamente todo al maximo (Ultra en todo, filtrado x16 y texturas nitidas). Los FPS dependen de la potencia de tu GPU y tu CPU.')
$panel.Controls.Add($chkSoulsExt); Move-To $chkSoulsExt $gModes 202 54 206
function Set-PresetUltra { $cmbTex.SelectedIndex = 0; $cmbDist.SelectedIndex = 0; $cmbShadow.SelectedIndex = 0; $cmbDet.SelectedIndex = 1; $cmbRt.SelectedIndex = 0; $cmbAo.SelectedIndex = 2; $cmbRefl.SelectedIndex = 1; $cmbFade.SelectedIndex = 1; $cmbAaS.SelectedIndex = 0 }  # (10-23) nitidez del AA al maximo (no cuesta)
function Set-PresetExtreme { $cmbTex.SelectedIndex = 0; $cmbDist.SelectedIndex = 0; $cmbShadow.SelectedIndex = 0; $cmbDet.SelectedIndex = 0; $cmbRt.SelectedIndex = 0; $cmbAo.SelectedIndex = 3; $cmbRefl.SelectedIndex = 0; $cmbFade.SelectedIndex = 0; $cmbAaS.SelectedIndex = 0; $chkAf16.Checked = $true; $chkLod.Checked = $true
	if ($cmbOut) { $cmbOut.SelectedIndex = 1 }; $chk4K.Checked = $true }  # (10-23) extremo: modo cinematico del juego a 4K
# (10-23) Modo Souls PC low profile: 900p y todo en bajo, para PCs modestos.
$chkSoulsLow = New-Check 'Modo Souls PC low profile (900p, todo en bajo)' 0 0 396 'soulslow'
$tips.SetToolTip($chkSoulsLow, 'Para PCs modestos: 30 FPS reales con frame generation de AMD x2 (60 en pantalla), el juego a 900p reescalado con FSR (nitidez 50 %) y todos los graficos en bajo (texturas, distancia, sombras, detalle, reflejos, objetos y nitidez). Al quitarlo vuelve todo a Maximo y a 1440p.')
$panel.Controls.Add($chkSoulsLow); Move-To $chkSoulsLow $gModes 12 84 396
function Set-ResByName([string]$name) {
	for ($i = 0; $i -lt $script:resShown.Count; $i++) { if ($script:resShown[$i][0] -eq $name) { $cmbRes.SelectedIndex = $i; Set-IniValue 'scalelabel' $name; return } }
}
function Set-PresetLowProfile {
	$cmbTex.SelectedIndex = 3; $cmbDist.SelectedIndex = 1; $cmbShadow.SelectedIndex = 2; $cmbDet.SelectedIndex = 4  # (10-25) distancia Alto, sombras Medio (fijas)
	$cmbRefl.SelectedIndex = 3; $cmbFade.SelectedIndex = 3; $cmbAaS.SelectedIndex = 3; $cmbRt.SelectedIndex = 0; $cmbAo.SelectedIndex = 0
	# (10-23) Lo que casi no cuesta, al maximo: filtrado x16, texturas nitidas y el postprocesado del juego.
	$chkAf16.Checked = $true; $chkLod.Checked = $true
	foreach ($pc in @($script:postChecks)) { if ($pc -and $pc[0]) { $pc[0].Checked = $true } }
	if ($cmbOut) { $cmbOut.SelectedIndex = 0 }; $chk4K.Checked = $false
	Fill-Res; Set-ResByName 'FSR Calidad'  # (10-25) recomendado FSR 3.1 Calidad
	if ($cmbRays) { $cmbRays.SelectedIndex = 2 }  # (10-25) luz volumetrica Medio
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
function Update-ResLock { $lock = $false
	Fill-Res; $cmbRes.Enabled = $true  # (10-25) ligero y low profile: el reescalado se elige libremente (todas las calidades)
	# (10-23) PC low profile: todo fijo (salida, graficos y casillas), nada se puede cambiar.
	$low = $chkSoulsLow.Checked
	$cmbFg.Enabled = -not (Is-FixedFg)  # (10-23) ligero / equilibrado / low profile: FG fija (antes esta funcion la volvia a activar)
	if ($cmbOut) { $cmbOut.Enabled = -not (Is-FixedFg) }
	foreach ($c in @($cmbSharp, $cmbTex, $cmbDist, $cmbShadow, $cmbDet, $cmbRefl, $cmbFade, $cmbAaS, $cmbRt, $cmbAo, $chkAf16, $chkLod)) { if ($c) { $c.Enabled = -not $low } }
	foreach ($p in @($script:postChecks)) { if ($p -and $p[0]) { $p[0].Enabled = -not $low } }
}
$chkLowEnd.Add_CheckedChanged({ Update-ResLock })
$chkSoulsLow.Add_CheckedChanged({ Update-ResLock; Update-LowEnd; Update-AntiLagBox })
$chkLowEnd.Add_CheckedChanged({ if ($chkLowEnd.Checked) { Set-PresetLight } elseif (-not $chkLowEnd45.Checked) { Set-PresetMax } })
$chkLowEnd.Add_CheckedChanged({ if ($chkLowEnd.Checked) { $cmbSharp.SelectedIndex = 4; Set-GsUserPreset } })
$chkLowEnd.Add_CheckedChanged({ if ($chkLowEnd.Checked) { Fill-Res; Set-ResByName 'FSR Calidad'; if ($cmbRays) { $cmbRays.SelectedIndex = 2 } } })  # (10-25) ligero: recomendado FSR 3.1 Calidad, distancia Alto, luz volumetrica Medio  # (10-24) nitidez 75%
$chkSoulsLow.Add_CheckedChanged({ if ($chkSoulsLow.Checked) { $cmbSharp.SelectedIndex = 5; Set-GsUserPreset } })  # (10-24) low profile: nitidez 100%
$chkLowEnd45.Add_CheckedChanged({ if ($chkLowEnd45.Checked) { Set-GsUserPreset } })  # (10-24) equilibrado: Alto, focos No
$chkSoulsUltra.Add_CheckedChanged({ if ($chkSoulsUltra.Checked) { Set-GsAll 4 2 } })  # ultra: Ultra, focos No
$chkSoulsExt.Add_CheckedChanged({ if ($chkSoulsExt.Checked) { Set-GsAll 5 2 } })  # extremo: Extremo, focos Si
$chkLowEnd45.Add_CheckedChanged({ if ($chkLowEnd45.Checked) { Set-PresetMax; $cmbTex.SelectedIndex = 1; $cmbDist.SelectedIndex = 1; $cmbShadow.SelectedIndex = 1; $cmbRt.SelectedIndex = 0; $cmbAo.SelectedIndex = 2; $cmbAaS.SelectedIndex = 1 } elseif (-not $chkLowEnd.Checked) { Set-PresetMax } })
# (10-25) Modo Souls equilibrado por defecto (captura del usuario): DLSS Calidad (K), nitidez 100 %, texturas Maximo, distancia y sombras
# Alto, Detail High, Reflections / Object Fade High, AA Sharpness High, filtrado x16, texturas extra nitidas, postprocesado
# todo activo, Luz RT No, Oclusion Normal, Luz volumetrica Bajo, Reflejos RT No.
$chkLowEnd45.Add_CheckedChanged({ if ($chkLowEnd45.Checked) {
	$cmbTex.SelectedIndex = 0; $cmbDist.SelectedIndex = 1; $cmbShadow.SelectedIndex = 1; $cmbDet.SelectedIndex = 2
	$cmbRefl.SelectedIndex = 1; $cmbFade.SelectedIndex = 1; $cmbAaS.SelectedIndex = 1
	$chkAf16.Checked = $true; if ($cmbLod) { $cmbLod.SelectedIndex = 2 }
	foreach ($pc in @($script:postChecks)) { if ($pc -and $pc[0]) { $pc[0].Checked = $true } }
	$cmbRt.SelectedIndex = 0; $cmbAo.SelectedIndex = 1; if ($cmbRays) { $cmbRays.SelectedIndex = 1 }; if ($cmbRtRf) { $cmbRtRf.SelectedIndex = 0 }
	$cmbSharp.SelectedIndex = 5; if ($cmbPreset -and $script:isNvidia) { $cmbPreset.SelectedIndex = 1 }
	Fill-Res; Set-ResByName $(if ($script:isNvidia) { 'DLSS Calidad' } else { 'FSR Calidad' })
} })
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
$btnPlay.Location = New-Object System.Drawing.Point(450, 418); $btnPlay.Size = New-Object System.Drawing.Size(420, 56)
$btnPre.Location = New-Object System.Drawing.Point(450, 480); $btnPre.Size = New-Object System.Drawing.Size(420, 44)
$btnSettings.Location = New-Object System.Drawing.Point(450, 530); $btnSettings.Size = New-Object System.Drawing.Size(205, 44)
$btnExit.Location = New-Object System.Drawing.Point(665, 530); $btnExit.Size = New-Object System.Drawing.Size(205, 44)
# (10-23) que nada tape los botones de abajo
foreach ($b in @($btnPlay, $btnPre, $btnSettings, $btnExit)) { $b.BringToFront() }

# ---- (10-27) Actualizar version / Update version (EN EL SITIO) ----
# Busca la ultima release PUBLICADA (releases/latest) de todojuegos67-bot/KytyPS5. Si es mas nueva que version.txt,
# pregunta y un actualizador (en TEMP; con UAC solo si la carpeta no se puede escribir) la descarga, la comprueba,
# mueve los archivos del motor actuales a _backup_update\<version vieja>\ (se guardan las 2 ultimas) y copia la
# release ENCIMA de esta carpeta sin tocar los datos del usuario. Si falla una copia, deshace todo desde la copia.
# Solo HTTPS, sin tokens.
$script:updLang = 'es'
function UT([string]$es, [string]$en) { if ($script:updLang -eq 'en') { return $en } return $es }
$script:updVerFile = Join-Path $root 'version.txt'
function Get-InstalledVersion {
	try { if (Test-Path -LiteralPath $script:updVerFile) { $v = ((Get-Content -LiteralPath $script:updVerFile -Raw) -replace '^\uFEFF', '').Trim(); if ($v) { return $v } } } catch { }
	return ''
}
function Get-LatestRelease {
	[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
	$rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/todojuegos67-bot/KytyPS5/releases/latest' -Headers @{ 'User-Agent' = 'DS-Launcher'; 'Accept' = 'application/vnd.github+json' } -UseBasicParsing -TimeoutSec 30
	if ($rel.draft -or $rel.prerelease) { throw 'no published release' }
	$zips = @($rel.assets | Where-Object { $_.name -like '*.zip' -and $_.name -notmatch '(?i)linux|macos|source|symbols|pdb' -and $_.browser_download_url -like 'https://*' })
	$win = @($zips | Where-Object { $_.name -match '(?i)windows|win64|x64|DemonsSouls|KytyPS5' })
	if ($win.Count) { $zips = $win }
	$asset = $zips | Sort-Object size -Descending | Select-Object -First 1
	if (-not $asset) { throw 'no zip asset' }
	return [pscustomobject]@{ Tag = [string]$rel.tag_name; Name = [string]$rel.name; Asset = [string]$asset.name; Url = [string]$asset.browser_download_url; Size = [int64]$asset.size }
}
function Test-FolderWritable([string]$p) {
	try { $t = Join-Path $p ('.ds-write-test-' + [guid]::NewGuid().ToString('N')); [IO.File]::WriteAllText($t, 'x'); Remove-Item -LiteralPath $t -Force; return $true } catch { return $false }
}
# El actualizador (se escribe en TEMP). -NoGui -ZipIn <zip> lo usa para pruebas sin ventanas ni descarga.
# Codigos de salida: 0 = actualizado, 1 = error (todo deshecho), 2 = cancelado (nada cambiado).
$script:updScript = @'
param([string]$Root, [string]$Url, [string]$Tag, [string]$Asset, [int64]$Size, [string]$Lang, [string]$ZipIn, [string]$RestoreFrom, [switch]$NoGui)
$ErrorActionPreference = 'Stop'
function T([string]$es, [string]$en) { if ($Lang -eq 'en') { return $en } return $es }
$title = T 'Actualizar version' 'Update version'
$BackupDirName = '_backup_update'
# Datos del usuario: NUNCA se copian encima ni se mueven (carpetas raiz y patrones de archivo).
$ExcludeDirs = @('_PipelineCache', '_SaveData', '_SaveData-copias', 'config', 'logs', 'profiles', $BackupDirName)
$ExcludeFiles = @('*.ini', '*.seeds', '*.seeds.program', '*.prewarm2', 'game-path.txt', 'ram16.txt', 'version.txt')
# Archivos del motor (solo raiz) que se mueven a la copia de seguridad.
$EnginePatterns = @('*.exe', '*.dll', '*.ps1')

function Get-RelPath([string]$base, [string]$full) {
	$b = [IO.Path]::GetFullPath($base).TrimEnd('\', '/')
	return [IO.Path]::GetFullPath($full).Substring($b.Length + 1)
}
function Test-Excluded([string]$rel) {
	$parts = $rel -split '[\\/]'
	if ($parts.Count -gt 1 -and ($ExcludeDirs -contains $parts[0])) { return $true }
	if ($parts.Count -eq 1 -and ($ExcludeDirs -contains $parts[0])) { return $true }
	$leaf = $parts[-1]
	foreach ($p in $ExcludeFiles) { if ($leaf -like $p) { return $true } }
	return $false
}
function Expand-Release([string]$zip, [string]$dir) {
	Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
	if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force }
	New-Item -ItemType Directory -Path $dir | Out-Null
	[IO.Compression.ZipFile]::ExtractToDirectory($zip, $dir)
	# Si el zip trae una sola carpeta raiz, esa es la release.
	$top = @(Get-ChildItem -LiteralPath $dir -Force)
	if ($top.Count -eq 1 -and $top[0].PSIsContainer) { return $top[0].FullName }
	return $dir
}
function Test-Release([string]$src) {
	if (-not (Test-Path -LiteralPath (Join-Path $src 'kyty_emulator.exe'))) { throw (T 'el zip no trae kyty_emulator.exe' 'the zip has no kyty_emulator.exe') }
	if (-not @(Get-ChildItem -LiteralPath $src -Filter '*.dll' -File).Count) { throw (T 'el zip no trae ninguna dll' 'the zip has no dll') }
}
function Get-OldVersion([string]$root) {
	$v = ''
	try { $vf = Join-Path $root 'version.txt'; if (Test-Path -LiteralPath $vf) { $v = ((Get-Content -LiteralPath $vf -Raw) -replace '^\uFEFF', '').Trim() } } catch { }
	if (-not $v) { $v = 'sin-version-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
	return ($v -replace '[\\/:*?"<>|]', '_')
}
function Backup-Engine([string]$root, [string]$oldVer) {
	# Mueve exe, dll y .ps1 de la raiz a _backup_update\<version vieja>\. Devuelve el estado para deshacer.
	$bdir = Join-Path (Join-Path $root $BackupDirName) $oldVer
	if (Test-Path -LiteralPath $bdir) { $bdir = $bdir + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
	New-Item -ItemType Directory -Force -Path $bdir | Out-Null
	New-Item -ItemType Directory -Force -Path (Join-Path $bdir '_overwritten') | Out-Null
	[IO.File]::WriteAllText((Join-Path $bdir '.version'), $oldVer, (New-Object System.Text.UTF8Encoding $false))
	$st = [pscustomobject]@{ Root = $root; Dir = $bdir; Moved = New-Object System.Collections.ArrayList; Overwritten = New-Object System.Collections.ArrayList; Created = New-Object System.Collections.ArrayList }
	foreach ($pat in $EnginePatterns) {
		foreach ($it in @(Get-ChildItem -LiteralPath $root -Filter $pat -File -Force)) {
			if (Test-Excluded $it.Name) { continue }
			Move-Item -LiteralPath $it.FullName -Destination (Join-Path $bdir $it.Name) -Force
			[void]$st.Moved.Add($it.Name)
		}
	}
	return $st
}
function Copy-Release([string]$src, $st) {
	# Copia la release encima. Lo que se pisa (fuera de lo movido) se guarda antes en _overwritten.
	$root = $st.Root
	foreach ($f in @(Get-ChildItem -LiteralPath $src -Recurse -File -Force)) {
		$rel = Get-RelPath $src $f.FullName
		if (Test-Excluded $rel) { continue }
		if ($rel -match '^_overwritten[\\/]' -or $rel -eq '.version') { continue }
		$dst = Join-Path $root $rel
		$dd = [IO.Path]::GetDirectoryName($dst)
		if (-not (Test-Path -LiteralPath $dd)) { New-Item -ItemType Directory -Force -Path $dd | Out-Null }
		if (Test-Path -LiteralPath $dst) {
			$ov = Join-Path (Join-Path $st.Dir '_overwritten') $rel
			New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($ov)) | Out-Null
			Copy-Item -LiteralPath $dst -Destination $ov -Force
			[void]$st.Overwritten.Add($rel)
		} else { [void]$st.Created.Add($rel) }
		Copy-Item -LiteralPath $f.FullName -Destination $dst -Force
	}
	# Los scripts movidos que la release no trae (p.ej. el launcher del usuario) vuelven a su sitio (exe/dll viejos no).
	foreach ($n in @($st.Moved | Where-Object { $_ -like '*.ps1' })) {
		$dst = Join-Path $root $n
		if (-not (Test-Path -LiteralPath $dst)) { Copy-Item -LiteralPath (Join-Path $st.Dir $n) -Destination $dst -Force; [void]$st.Created.Add($n) }
	}
}
function Undo-Update($st) {
	# Deshace: borra lo creado, devuelve lo pisado y lo movido.
	$errs = 0
	foreach ($rel in $st.Created) { try { $p = Join-Path $st.Root $rel; if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force } } catch { $errs++ } }
	foreach ($rel in $st.Overwritten) { try { Copy-Item -LiteralPath (Join-Path (Join-Path $st.Dir '_overwritten') $rel) -Destination (Join-Path $st.Root $rel) -Force } catch { $errs++ } }
	foreach ($n in $st.Moved) { try { Copy-Item -LiteralPath (Join-Path $st.Dir $n) -Destination (Join-Path $st.Root $n) -Force } catch { $errs++ } }
	if ($errs -eq 0) { try { Remove-Item -LiteralPath $st.Dir -Recurse -Force } catch { } }
	return $errs
}
function Remove-OldBackups([string]$root, [int]$keep = 2, [string]$protect = '') {
	$b = Join-Path $root $BackupDirName
	if (-not (Test-Path -LiteralPath $b)) { return }
	$all = @(Get-ChildItem -LiteralPath $b -Directory -Force | Where-Object { $_.Name -ne $protect } | Sort-Object CreationTimeUtc, LastWriteTimeUtc -Descending)
	if ($protect -and (Test-Path -LiteralPath (Join-Path $b $protect))) { $keep = [Math]::Max(0, $keep - 1) }
	foreach ($d in ($all | Select-Object -Skip $keep)) { try { Remove-Item -LiteralPath $d.FullName -Recurse -Force } catch { } }
}
function Invoke-InPlaceUpdate([string]$root, [string]$zip, [string]$tag) {
	# Nucleo: extraer, comprobar, copia de seguridad, copiar, deshacer si falla, version.txt.
	$tmp = Join-Path ([IO.Path]::GetTempPath()) ('ds-update-' + [guid]::NewGuid().ToString('N'))
	try {
		$src = Expand-Release $zip $tmp
		Test-Release $src
		$st = Backup-Engine $root (Get-OldVersion $root)
		try { Copy-Release $src $st }
		catch {
			$m = $_.Exception.Message
			$e = Undo-Update $st
			if ($e) { throw ((T 'fallo la copia y NO se pudo deshacer todo (copia en ' 'copy failed and could NOT fully roll back (backup in ') + $st.Dir + '): ' + $m) }
			throw ((T 'fallo la copia (deshecho, la carpeta sigue como estaba): ' 'copy failed (rolled back, folder unchanged): ') + $m)
		}
		Remove-Item -LiteralPath (Join-Path $st.Dir '_overwritten') -Recurse -Force -ErrorAction SilentlyContinue
		[IO.File]::WriteAllText((Join-Path $root 'version.txt'), $tag, (New-Object System.Text.UTF8Encoding $false))
		Remove-OldBackups $root 2
	} finally { try { if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force } } catch { } }
}
function Get-BackupVersion([string]$dir) {
	$v = ''
	try { $vf = Join-Path $dir '.version'; if (Test-Path -LiteralPath $vf) { $v = ((Get-Content -LiteralPath $vf -Raw) -replace '^\uFEFF', '').Trim() } } catch { }
	if (-not $v) { $v = Split-Path -Leaf $dir }
	return $v
}
function Invoke-RestoreBackup([string]$root, [string]$name) {
	# Vuelve a una copia de _backup_update. Antes guarda lo actual (asi tambien se puede deshacer).
	if (-not $name -or $name -match '[\\/]|^\.+$') { throw 'bad backup name' }
	$src = Join-Path (Join-Path $root $BackupDirName) $name
	if (-not (Test-Path -LiteralPath (Join-Path $src 'kyty_emulator.exe'))) { throw (T 'la copia no tiene kyty_emulator.exe' 'the backup has no kyty_emulator.exe') }
	$ver = Get-BackupVersion $src
	$st = Backup-Engine $root (Get-OldVersion $root)
	try { Copy-Release $src $st }
	catch {
		$m = $_.Exception.Message
		$e = Undo-Update $st
		if ($e) { throw ((T 'fallo la copia y NO se pudo deshacer todo (copia en ' 'copy failed and could NOT fully roll back (backup in ') + $st.Dir + '): ' + $m) }
		throw ((T 'fallo la copia (deshecho, la carpeta sigue como estaba): ' 'copy failed (rolled back, folder unchanged): ') + $m)
	}
	Remove-Item -LiteralPath (Join-Path $st.Dir '_overwritten') -Recurse -Force -ErrorAction SilentlyContinue
	[IO.File]::WriteAllText((Join-Path $root 'version.txt'), $ver, (New-Object System.Text.UTF8Encoding $false))
	Remove-OldBackups $root 2 $name
	return $ver
}
if ($env:DS_UPDATER_LIBONLY -eq '1') { return }

if ($RestoreFrom) {
	try { [void](Invoke-RestoreBackup $Root $RestoreFrom); exit 0 }
	catch {
		if ($NoGui) { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }
		Add-Type -AssemblyName System.Windows.Forms
		[System.Windows.Forms.MessageBox]::Show((T 'No se pudo restaurar: ' 'Restore failed: ') + $_.Exception.Message, (T 'Volver a la version anterior' 'Restore previous version'), 'OK', 'Error') | Out-Null
		exit 1
	}
}

if ($NoGui) {
	try { Invoke-InPlaceUpdate $Root $ZipIn $Tag; exit 0 } catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }
}

Add-Type -AssemblyName System.Windows.Forms, System.Drawing
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$zip = Join-Path ([IO.Path]::GetTempPath()) ('ds-update-' + ($Tag -replace '[\\/:*?"<>|]', '_') + '.zip')
$script:cancel = $false; $script:done = $false; $script:code = 1
$f = New-Object System.Windows.Forms.Form
$f.Text = "$title - $Tag"; $f.FormBorderStyle = 'FixedDialog'; $f.StartPosition = 'CenterScreen'; $f.MaximizeBox = $false; $f.MinimizeBox = $false
$f.ClientSize = New-Object System.Drawing.Size(460, 120); $f.TopMost = $true
$lb = New-Object System.Windows.Forms.Label; $lb.SetBounds(12, 12, 436, 36); $lb.Text = T 'Preparando...' 'Preparing...'
$pb = New-Object System.Windows.Forms.ProgressBar; $pb.SetBounds(12, 52, 436, 22); $pb.Maximum = 1000
$bc = New-Object System.Windows.Forms.Button; $bc.SetBounds(358, 82, 90, 28); $bc.Text = T 'Cancelar' 'Cancel'
$bc.Add_Click({ $script:cancel = $true; $bc.Enabled = $false; $lb.Text = T 'Cancelando...' 'Cancelling...'; try { $script:wc.CancelAsync() } catch { } })
$f.Controls.AddRange(@($lb, $pb, $bc))
$f.Add_FormClosing({ param($s, $e) if (-not $script:done) { $e.Cancel = $true } })
function Install {
	try {
		$lb.Text = (T 'Descargando ' 'Downloading ') + $Asset
		$script:wc = New-Object System.Net.WebClient
		$script:wc.Headers.Add('User-Agent', 'DS-Launcher')
		$script:dlDone = $false; $script:dlErr = $null
		$script:wc.add_DownloadProgressChanged({ param($s, $e)
			$tot = if ($e.TotalBytesToReceive -gt 0) { $e.TotalBytesToReceive } else { $Size }
			$p = if ($tot -gt 0) { $e.BytesReceived / $tot } else { 0 }
			$lb.Text = (T 'Descargando ' 'Downloading ') + ('{0:N0} / {1:N0} MB  ({2:P0})' -f ($e.BytesReceived / 1MB), ($tot / 1MB), $p)
			$pb.Value = [int][Math]::Min(1000, $p * 800) })
		$script:wc.add_DownloadFileCompleted({ param($s, $e) $script:dlErr = if ($e.Cancelled) { 'CANCEL' } elseif ($e.Error) { $e.Error.Message } else { $null }; $script:dlDone = $true })
		$script:wc.DownloadFileAsync([Uri]$Url, $zip)
		while (-not $script:dlDone) { [System.Windows.Forms.Application]::DoEvents(); Start-Sleep -Milliseconds 50 }
		if ($script:dlErr) { throw $script:dlErr }
		if ($script:cancel) { throw 'CANCEL' }
		if ($Size -gt 0 -and (Get-Item -LiteralPath $zip).Length -ne $Size) { throw (T 'la descarga esta incompleta' 'the download is incomplete') }
		# Desde aqui ya no se puede cancelar (se esta cambiando la carpeta).
		$bc.Enabled = $false; $pb.Value = 850
		$lb.Text = T 'Instalando (copia de seguridad y archivos nuevos)...' 'Installing (backup and new files)...'
		[System.Windows.Forms.Application]::DoEvents()
		Invoke-InPlaceUpdate $Root $zip $Tag
		$pb.Value = 1000; $script:code = 0
	} catch {
		$m = "$($_.Exception.Message)"
		if ($m -like '*CANCEL*' -or $script:cancel) { $script:code = 2 }
		else {
			$f.Hide()
			[System.Windows.Forms.MessageBox]::Show((T 'No se pudo actualizar: ' 'Update failed: ') + $m, $title, 'OK', 'Error') | Out-Null
			$script:code = 1
		}
	} finally {
		try { if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force } } catch { }
		$script:done = $true; $f.Close()
	}
}
$f.Add_Shown({ $f.Activate(); Install })
[void]$f.ShowDialog()
exit $script:code
'@
function Start-UpdateVersion {
	$t = UT 'Actualizar version' 'Update version'
	# 1) Comprobar version
	$cur = Get-InstalledVersion
	try { $form.Cursor = 'WaitCursor'; $rel = Get-LatestRelease }
	catch { [System.Windows.Forms.MessageBox]::Show((UT 'No se pudo consultar GitHub: ' 'Could not check GitHub: ') + $_.Exception.Message, $t, 'OK', 'Error') | Out-Null; return }
	finally { $form.Cursor = 'Default' }
	if ($cur -and $cur -eq $rel.Tag) {
		[System.Windows.Forms.MessageBox]::Show((UT 'Ya tienes la última versión: ' 'You already have the latest version: ') + $rel.Tag, $t, 'OK', 'Information') | Out-Null; return
	}
	# 2) Confirmar
	$curTxt = if ($cur) { $cur } else { UT 'desconocida (no hay version.txt)' 'unknown (no version.txt)' }
	$newTxt = if ($rel.Name -and $rel.Name -ne $rel.Tag) { "$($rel.Tag) - $($rel.Name)" } else { $rel.Tag }
	$msg = (UT 'Version instalada: ' 'Installed version: ') + $curTxt + "`n" + (UT 'Version nueva: ' 'New version: ') + $newTxt + "`n" + $rel.Asset + ('  ({0:N0} MB)' -f ($rel.Size / 1MB)) + "`n`n" +
		(UT 'Se actualiza en esta misma carpeta. Los archivos actuales del motor se guardan en _backup_update. La cache de shaders, las partidas, los seeds y los ajustes no se tocan. Actualizar?' 'It updates in this same folder. The current engine files are kept in _backup_update. The shader cache, save games, seeds and settings are not touched. Update?')
	if ([System.Windows.Forms.MessageBox]::Show($msg, $t, 'YesNo', 'Question') -ne 'Yes') { return }
	# 3) El juego debe estar cerrado
	if (@(Get-Process -Name 'kyty_emulator' -ErrorAction SilentlyContinue).Count) {
		[System.Windows.Forms.MessageBox]::Show((UT 'Cierra el juego primero y vuelve a pulsar Actualizar version.' 'Close the game first, then press Update version again.'), $t, 'OK', 'Warning') | Out-Null; return
	}
	# 4) Administrador solo si hace falta
	$admin = -not (Test-FolderWritable $root)
	$us = Join-Path $env:TEMP 'ds-actualizar-version.ps1'
	[IO.File]::WriteAllText($us, $script:updScript, (New-Object System.Text.UTF8Encoding $true))
	$q = { param($s) '"' + ($s -replace '"', '') + '"' }
	$args2 = '-NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File ' + (& $q $us) + ' -Root ' + (& $q $root) + ' -Url ' + (& $q $rel.Url) + ' -Tag ' + (& $q $rel.Tag) + ' -Asset ' + (& $q $rel.Asset) + ' -Size ' + $rel.Size + ' -Lang ' + $script:updLang
	try {
		if ($admin) { $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $args2 -Verb RunAs -PassThru }
		else { $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $args2 -PassThru }
	} catch {
		[System.Windows.Forms.MessageBox]::Show((UT 'Sin permiso de administrador no se puede escribir en esta carpeta. No se ha cambiado nada.' 'Without administrator permission this folder cannot be written. Nothing was changed.'), $t, 'OK', 'Warning') | Out-Null; return
	}
	$form.Enabled = $false
	try { while (-not $p.HasExited) { [System.Windows.Forms.Application]::DoEvents(); Start-Sleep -Milliseconds 100 } }
	finally { $form.Enabled = $true }
	# 5) Final: un solo aviso (no se reinicia solo)
	if ($p.ExitCode -eq 0) {
		[System.Windows.Forms.MessageBox]::Show(((UT 'Actualización completada a la versión {0}. Cierra el launcher y vuelve a abrirlo para usar la nueva versión. Tus shaders compilados y partidas guardadas se mantienen.' 'Update to version {0} complete. Close the launcher and open it again to use the new version. Your compiled shaders and saved games are kept.') -f $rel.Tag), $t, 'OK', 'Information') | Out-Null
	}
}
function Get-UpdateBackups {
	$b = Join-Path $root '_backup_update'
	if (-not (Test-Path -LiteralPath $b)) { return @() }
	return @(Get-ChildItem -LiteralPath $b -Directory -Force | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'kyty_emulator.exe') } | Sort-Object CreationTimeUtc -Descending | ForEach-Object {
		$v = ''; try { $vf = Join-Path $_.FullName '.version'; if (Test-Path -LiteralPath $vf) { $v = ((Get-Content -LiteralPath $vf -Raw) -replace '^\uFEFF', '').Trim() } } catch { }
		if (-not $v) { $v = $_.Name }
		[pscustomobject]@{ Name = $_.Name; Version = $v; Date = $_.CreationTime } })
}
function Start-RestoreVersion {
	$t = UT 'Volver a la versión anterior' 'Restore previous version'
	$list = @(Get-UpdateBackups)
	if (-not $list.Count) { [System.Windows.Forms.MessageBox]::Show((UT 'No hay versiones anteriores guardadas' 'No previous versions saved'), $t, 'OK', 'Information') | Out-Null; return }
	# Elegir copia
	$f = New-Object System.Windows.Forms.Form
	$f.Text = $t; $f.FormBorderStyle = 'FixedDialog'; $f.StartPosition = 'CenterParent'; $f.MaximizeBox = $false; $f.MinimizeBox = $false
	$f.ClientSize = New-Object System.Drawing.Size(420, 200)
	$l = New-Object System.Windows.Forms.Label; $l.Text = (UT 'Version instalada: ' 'Installed version: ') + (Get-InstalledVersion) + "`n" + (UT 'Elige la version a restaurar:' 'Choose the version to restore:'); $l.SetBounds(12, 8, 396, 34)
	$lst = New-Object System.Windows.Forms.ListBox; $lst.SetBounds(12, 44, 396, 108)
	foreach ($it in $list) { [void]$lst.Items.Add(('{0}   ({1:yyyy-MM-dd HH:mm})' -f $it.Version, $it.Date)) }
	$lst.SelectedIndex = 0
	$ok = New-Object System.Windows.Forms.Button; $ok.Text = UT 'Restaurar' 'Restore'; $ok.SetBounds(218, 162, 90, 30); $ok.DialogResult = 'OK'
	$cn = New-Object System.Windows.Forms.Button; $cn.Text = UT 'Cancelar' 'Cancel'; $cn.SetBounds(318, 162, 90, 30); $cn.DialogResult = 'Cancel'
	$f.Controls.AddRange(@($l, $lst, $ok, $cn)); $f.AcceptButton = $ok; $f.CancelButton = $cn
	$r = $f.ShowDialog($form); $idx = $lst.SelectedIndex; $f.Dispose()
	if ($r -ne 'OK' -or $idx -lt 0) { return }
	$sel = $list[$idx]
	$msg = (UT 'Se restaura la version ' 'Version ') + $sel.Version + (UT '. Los archivos actuales del motor se guardan antes en _backup_update (se puede deshacer). La cache de shaders, las partidas, los seeds y los ajustes no se tocan. Continuar?' ' will be restored. The current engine files are saved to _backup_update first (it can be undone). The shader cache, save games, seeds and settings are not touched. Continue?')
	if ([System.Windows.Forms.MessageBox]::Show($msg, $t, 'YesNo', 'Question') -ne 'Yes') { return }
	if (@(Get-Process -Name 'kyty_emulator' -ErrorAction SilentlyContinue).Count) {
		[System.Windows.Forms.MessageBox]::Show((UT 'Cierra el juego primero y vuelve a intentarlo.' 'Close the game first, then try again.'), $t, 'OK', 'Warning') | Out-Null; return
	}
	$admin = -not (Test-FolderWritable $root)
	$us = Join-Path $env:TEMP 'ds-actualizar-version.ps1'
	[IO.File]::WriteAllText($us, $script:updScript, (New-Object System.Text.UTF8Encoding $true))
	$q = { param($s) '"' + ($s -replace '"', '') + '"' }
	$args2 = '-NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File ' + (& $q $us) + ' -Root ' + (& $q $root) + ' -RestoreFrom ' + (& $q $sel.Name) + ' -Lang ' + $script:updLang
	try {
		if ($admin) { $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $args2 -Verb RunAs -PassThru }
		else { $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $args2 -PassThru }
	} catch {
		[System.Windows.Forms.MessageBox]::Show((UT 'Sin permiso de administrador no se puede escribir en esta carpeta. No se ha cambiado nada.' 'Without administrator permission this folder cannot be written. Nothing was changed.'), $t, 'OK', 'Warning') | Out-Null; return
	}
	$form.Enabled = $false; $form.Cursor = 'WaitCursor'
	try { while (-not $p.HasExited) { [System.Windows.Forms.Application]::DoEvents(); Start-Sleep -Milliseconds 100 } }
	finally { $form.Enabled = $true; $form.Cursor = 'Default' }
	if ($p.ExitCode -eq 0) {
		[System.Windows.Forms.MessageBox]::Show(((UT 'Restaurada la versión {0}. Cierra el launcher y vuelve a abrirlo.' 'Restored version {0}. Close the launcher and open it again.') -f $sel.Version), $t, 'OK', 'Information') | Out-Null
	}
}
$btnUpd = New-Button 'Actualizar version' 730 480 140 44 $false
$panel.Controls.Add($btnUpd)
$btnUpd.Add_Click({ try { Start-UpdateVersion } catch { [System.Windows.Forms.MessageBox]::Show("$_", 'Launcher', 'OK', 'Error') | Out-Null } })
$tips.SetToolTip($btnUpd, 'Busca la ultima version publicada en GitHub y la instala en esta misma carpeta. Guarda los archivos del motor actuales en _backup_update (las 2 ultimas) y no toca la cache de shaders, las partidas, los seeds ni los ajustes.')
$btnRes = New-Button 'Volver a la versión anterior' 610 480 125 44 $false
$panel.Controls.Add($btnRes)
$btnRes.Add_Click({ try { Start-RestoreVersion } catch { [System.Windows.Forms.MessageBox]::Show("$_", 'Launcher', 'OK', 'Error') | Out-Null } })
$tips.SetToolTip($btnRes, 'Vuelve a una version guardada en _backup_update. Antes guarda la actual (se puede deshacer). No toca la cache de shaders, las partidas, los seeds ni los ajustes.')
$btnUpd.Location = New-Object System.Drawing.Point(745, 480); $btnUpd.Size = New-Object System.Drawing.Size(125, 44)
$btnPre.Size = New-Object System.Drawing.Size(150, 44)
$btnUpd.BringToFront(); $btnRes.BringToFront()

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
	$lbDev = New-Label 'Buscando mandos...' $smallFont ([System.Drawing.Color]::FromArgb(205, 205, 205)) 290 584 940 26
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
		# (10-24) Dibujo del mando (sin logos, va en la release), 350 de alto: centros medidos en el.
		$k = 1.0; $offX = (540 - $script:pmPhotoPs.Width) / 2
		$c = @{ 'Up' = @(96, 62, 29); 'Down' = @(96, 126, 29); 'Left' = @(65, 93, 29); 'Right' = @(128, 93, 29)
			'Triangle' = @(435, 60, 38); 'Circle' = @(472, 94, 38); 'Cross' = @(435, 131, 38); 'Square' = @(398, 94, 38)
			'L3' = @(178, 160, 67); 'R3' = @(346, 160, 67); 'Options' = @(389, 32, 20); 'TouchPad' = @(265, 52, 150) }
		foreach ($n in $c.Keys) { $v = $c[$n]; $d = $v[2] * $k; $h = if ($n -eq 'TouchPad') { 80 * $k } else { $d }
			$ps[$n] = @(($offX + $v[0] * $k - $d / 2), ($photoTop + $v[1] * $k - $h / 2), $d, $h, $(if ($n -eq 'TouchPad') { 'r' } else { 'c' })) }
	}
	if ($script:pmPhotoXb) {
		$k = 1.0; $offX = (540 - $script:pmPhotoXb.Width) / 2
		$c = @{ 'y' = @(416, 58, 38); 'x' = @(377, 95, 38); 'b' = @(456, 95, 38); 'a' = @(416, 132, 38)
			'leftstick' = @(125, 86, 72); 'rightstick' = @(339, 156, 72)
			'dpup' = @(193, 140, 25); 'dpdown' = @(193, 192, 25); 'dpleft' = @(167, 166, 25); 'dpright' = @(219, 166, 25)
			'back' = @(226, 89, 22); 'start' = @(312, 89, 22) }
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
		$dim = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(165, 165, 165))
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
			& $drawBtn $k $script:pmRectXb[$k] $script:pmXbDef[$k][4] $(if ($onP) { '' } else { $script:xbLabel[$k] }) $hot $false $onP  # (10-25) sobre la foto, sin etiqueta (se solapaban)
			# Insignia con el control de PS5 que hace este boton: solo si se ha cambiado (las de serie se solapaban).
			$r = $script:pmRectXb[$k]; $t = $script:pmMap[$k]
			if ($t -eq $script:padDefault[$k]) { continue }
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
			foreach ($kv in $script:xiBits.GetEnumerator()) { $bit = [int]$kv.Key; if ((($v -band $bit) -ne 0) -and (($prev -band $bit) -eq 0)) { $pressed = $kv.Value; $script:pmUsed = "XInput $($i + 1)" } }  # (pad-1024) slot en uso
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
							if ($nm -and $script:xbLabel.ContainsKey($nm)) { $pressed = $nm; $script:pmUsed = "SDL $($j + 1)" }  # (pad-1024)
						}
					}
					$script:pmSdlPrev[$j] = $v; $j++
				}
			} catch { $script:pmSdlOk = $false }
		}
		$txt = if ($names.Count) { 'Mandos detectados: ' + ($names -join ', ') + $(if ($script:pmUsed -and ($names -contains $script:pmUsed)) { " - en uso: $($script:pmUsed)" } else { '' }) } else { 'No se detecta ningun mando (puedes hacer clic en los botones de la derecha).' }  # (pad-1024) slot en uso
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
	$tKbm.Controls.Add((New-Label ("Raton: mueve la camara (stick derecho). Mayus + clic: ataques fuertes (R2 / L2).`nEn el juego: Esc suelta el raton y un clic lo vuelve a capturar. F7 tambien lo suelta o lo captura.`nLa sensibilidad y la anti-zona muerta estan en el lanzador (apartado Controles).") $smallFont ([System.Drawing.Color]::FromArgb(215, 215, 215)) 670 310 560 90 'TopLeft'))
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
# (10-24) Ray tracing en pantalla (beta): sombras del sol y reflejos con denoise. Solo con DLSS, XeSS o FSR.
$gRt = New-Group 'Efectos' 10 586 420 178
function New-RtCombo([string]$label, [string]$key, $y, [string]$tip, $items = @('No', 'Bajo', 'Alto', 'Ultra')) {
	$lb = New-Label $label $smallFont $ink 0 0 98 22; $panel.Controls.Add($lb)
	$c = New-Object System.Windows.Forms.ComboBox
	$c.DropDownStyle = 'DropDownList'; $c.Font = $smallFont; $c.Tag = $key
	foreach ($it in $items) { [void]$c.Items.Add($it) }
	$v = 0; [void][int]::TryParse((Get-IniValue $key '0'), [ref]$v); $c.SelectedIndex = [math]::Min([math]::Max(0, $v), $items.Count - 1)
	$c.Add_SelectedIndexChanged({ param($sender, $e) Set-IniValue $sender.Tag "$($sender.SelectedIndex)" })
	$tips.SetToolTip($c, $tip); $panel.Controls.Add($c)
	Move-To $lb $gRt 10 ($y + 2) 112; Move-To $c $gRt 124 $y 282
	return $c
}
$cmbRtSh = New-RtCombo 'Sombras RT' 'rtshadow2' 120 'Sombras del sol por rayos sobre la imagen (de pantalla, sin RT real): sombras de contacto que las del juego pierden (bajo brazos, entre piernas, junto a muros), con borde suave y estable. Solo donde el juego ya muestra sol. Solo con DLSS, XeSS o FSR. Bajo ~0,3 ms, Medio ~0,6 ms, Alto ~1 ms.' @('No', 'Bajo', 'Medio', 'Alto')  # (ds-shadow-1028)
$cmbAoNew = New-RtCombo 'Oclusión ambiental' 'ao1028' 88 'Oclusion ambiental nueva (ao-1028, sin RT por hardware): oscurece solo la luz ambiental en rincones, juntas y bajo objetos, con radio en metros, sin halos ni ruido; nunca el cielo, el fuego ni lo emisivo. Sustituye a la oclusion del juego (no oscurece dos veces). Solo con DLSS, XeSS o FSR. Bajo ~0,3 ms, Medio ~0,5 ms, Alto ~1 ms (resolucion completa).' @('No', 'Bajo', 'Medio', 'Alto')  # (ao-1028) oclusion ambiental nueva (KYTY_AO), solo en Ray tracing
$cmbRays = New-RtCombo 'Luz volumetrica' 'godrays2' 24 'Luz volumetrica por rayos: rayos de sol que entran por ventanas y puertas, y halo de luz en la niebla alrededor de antorchas. Solo con DLSS, XeSS o FSR. Bajo ~0,4 ms, Medio ~0,8 ms, Alto ~1,2 ms (mas rayos y mas pasos de sombra).' @('No', 'Bajo', 'Medio', 'Alto')
$cmbRtRf = New-RtCombo 'Reflejos RT' 'rtrefl2' 56 'Reflejos trazados por rayos guiados por los del propio juego: solo donde el juego refleja y con su intensidad, mas nitidos (agua como espejo). Donde el rayo no encuentra nada queda el reflejo del juego. Solo con DLSS, XeSS o FSR. Bajo ~0,6 ms, Medio ~1 ms, Alto ~2 ms, Ultra ~3 ms (rayos a resolucion completa y mas fuertes). Beta.' @('No', 'Bajo', 'Medio', 'Alto', 'Ultra', 'Extremo')  # (10-25 noche) Extremo = nivel 5 (refl10)  # (10-25) refl8; Ultra vuelve: el cierre era del frame generation AMD, no de los reflejos
$cmbShine = $null  # (10-24) Brillo metales retirado
# (10-24) Ajustes del juego: convars del propio motor, cada uno en Juego (no se toca) o en un nivel. Nombres, tipos,
# rangos y valores de fabrica sacados del registro de convars del eboot 1.07 (constructor de cada convar: min / max;
# constructor de la estructura de ajustes: valor de fabrica). Ninguno baja la resolucion. Van en KYTY_GAME_CONVARS_EXTRA,
# que el emulador lee DESPUES de todo lo que pone run-windows (-Shadows, -Quality, -Reflections, -ObjectFade, -AaSharp,
# -PostOff, -Detail, TURBO y las pruebas): mandan sobre esos desplegables y casillas. Ojo: el juego carga su renderconfig
# (modo rendimiento / cine) despues de la linea de comandos y en el vuelve a poner r_virtualGeometryNumShadowCascades,
# r_iblResolution y r_tessellation*. Cortes de cascada: 75 % logaritmico + 25 % lineal desde 0,5 m, la ultima = la maxima
# (como run-windows; si no, lo lejano sale negro). No existen convars de grano, aberracion cromatica, particulas, LOD de
# texturas ni distancia / resolucion de las sombras de focos.
$script:userCvExtra = "$env:KYTY_GAME_CONVARS_EXTRA"  # (la del usuario, si la tiene, va detras: manda para sus pruebas)
$script:gsLevels = @('Predeterminado', 'Bajo', 'Medio', 'Alto', 'Ultra', 'Extremo')
$script:gsOnOff = @('Predeterminado', 'No', 'Si')
$script:gameSettings = @(
	@{ id = 'gs_sun'; sec = 'Sombras'; onoff = $false; name = 'Sombras del sol'; lv = @('r_virtualGeometryNumShadowCascades=1;sunShadowMaxDistance=60;sunShadowCascadeDistance0=60;sunShadowLightDistance=1000', 'r_virtualGeometryNumShadowCascades=2;sunShadowMaxDistance=125;sunShadowCascadeDistance0=22;sunShadowCascadeDistance1=125;sunShadowLightDistance=1000', 'r_virtualGeometryNumShadowCascades=2;sunShadowMaxDistance=250;sunShadowCascadeDistance0=40;sunShadowCascadeDistance1=250;sunShadowLightDistance=1000', 'r_virtualGeometryNumShadowCascades=4;sunShadowMaxDistance=400;sunShadowCascadeDistance0=27;sunShadowCascadeDistance1=61;sunShadowCascadeDistance2=131;sunShadowCascadeDistance3=400;sunShadowLightDistance=1000', 'r_virtualGeometryNumShadowCascades=4;sunShadowMaxDistance=1000;sunShadowCascadeDistance0=65;sunShadowCascadeDistance1=142;sunShadowCascadeDistance2=300;sunShadowCascadeDistance3=1000;sunShadowLightDistance=1000')
		tip = 'Sombras del sol del juego: hasta donde llegan y en cuantas cascadas se reparten. Bajo 60 m (1 cascada), Medio 125 m (2), Alto 250 m (2, el valor de fabrica), Ultra 400 m (4), Extremo 1000 m (4, el maximo del juego). Cuesta GPU y CPU (las sombras son unos 2600 dibujados por frame); Extremo puede costar un 10-15 % de FPS. Manda sobre el desplegable Sombras.' }
	@{ id = 'gs_spot'; sec = 'Sombras'; onoff = $true; name = 'Sombras de focos (antorchas)'; lv = @('r_enableSpotShadows=false', 'r_enableSpotShadows=true')
		tip = 'Sombras que proyectan antorchas, hogueras y otras luces (unas 7 por escena en 1-1). No: mas FPS, pero sin esas sombras. El juego no tiene ajuste de distancia ni de resolucion para ellas.' }
	@{ id = 'gs_contact'; sec = 'Sombras'; onoff = $false; name = 'Sombras de contacto'; lv = @('r_contactShadowsEnabled=false', 'r_contactShadowsEnabled=true;r_contactShadowsDistance=0.03', 'r_contactShadowsEnabled=true;r_contactShadowsDistance=0.06', 'r_contactShadowsEnabled=true;r_contactShadowsDistance=0.1', 'r_contactShadowsEnabled=true;r_contactShadowsDistance=0.3;r_contactShadowsThickness=0.12;r_contactShadowsSoftness=0.2')
		tip = 'Sombras de contacto del juego: sombras finas donde los objetos tocan el suelo y las paredes. Bajo: apagadas. Medio, Alto (de fabrica), Ultra y Extremo: alcance 0,03 / 0,06 / 0,10 / 0,30 (Extremo tambien mas gruesas y mas definidas). Cuesta poco; con mucho alcance pueden salir halos.' }
	@{ id = 'gs_ambient'; sec = 'Iluminacion'; onoff = $false; name = 'Sombreado ambiental del juego'; lv = @('r_enableAmbientShadowing=false', 'r_enableAmbientShadowing=true;r_ambientShadowCutoffDistance=50;r_ambientShadowCutoffFadeDistance=10', 'r_enableAmbientShadowing=true;r_ambientShadowCutoffDistance=100;r_ambientShadowCutoffFadeDistance=20', 'r_enableAmbientShadowing=true;r_ambientShadowCutoffDistance=250;r_ambientShadowCutoffFadeDistance=40', 'r_enableAmbientShadowing=true;r_ambientShadowCutoffDistance=1000;r_ambientShadowCutoffFadeDistance=100')
		tip = 'Sombreado ambiental del juego: oscurece bajo personajes y objetos con la luz del entorno. Bajo: apagado (mas FPS). Medio 50 m, Alto 100 m (de fabrica), Ultra 250 m, Extremo 1000 m (el maximo): hasta donde se dibuja. Mas distancia = mas GPU en zonas abiertas.' }
	@{ id = 'gs_gi'; sec = 'Iluminacion'; onoff = $false; name = 'Iluminacion global'; lv = @('GIEnableMultiBounce=false;GIUpdateNumIteraions=1;r_iblResolution=HalfResX', 'GIEnableMultiBounce=true;GIUpdateNumIteraions=1;r_iblResolution=HalfResX', 'GIEnableMultiBounce=true;GIUpdateNumIteraions=1;r_iblResolution=Full', 'GIEnableMultiBounce=true;GIUpdateNumIteraions=2;GILightCutoffIntensity=0.0005;r_iblResolution=Full', 'GIEnableMultiBounce=true;GIUpdateNumIteraions=4;GILightCutoffIntensity=0.0001;GILightMinSubtendedAngle=3;r_iblResolution=Full')
		tip = 'Iluminacion global del juego (luz que rebota). Bajo: sin rebote multiple y reflejos de entorno a media resolucion. Medio: como el modo rendimiento de PS5. Alto: reflejos de entorno completos (modo cine). Ultra y Extremo: 2 o 4 pasadas de actualizacion por frame y cuentan luces mas debiles y pequenas. Cuesta GPU (Extremo bastante). El modo del juego puede volver a fijar los reflejos de entorno al cargar.' }
	@{ id = 'gs_dist'; sec = 'Distancia'; onoff = $false; name = 'Distancia de dibujado'; lv = @('r_forceViewportHeightForPixelSizeCalc=480;r_fadeOutInstanceOverDistance=6', 'r_forceViewportHeightForPixelSizeCalc=720;r_fadeOutInstanceOverDistance=10', 'r_forceViewportHeightForPixelSizeCalc=1080;r_fadeOutInstanceOverDistance=15', 'r_forceViewportHeightForPixelSizeCalc=1440;r_fadeOutInstanceOverDistance=35', 'r_forceViewportHeightForPixelSizeCalc=2160;r_fadeOutInstanceOverDistance=60')
		tip = 'Distancia de dibujado: el detalle de los modelos (LOD) se elige como para una imagen de 480 / 720 / 1080 (de fabrica) / 1440 / 2160 lineas, y el fundido de los objetos lejanos pasa a 6 / 10 / 15 / 35 / 60. No cambia la resolucion. Bajo y Medio ayudan si va justo el procesador; Ultra y Extremo cuestan CPU y GPU. Manda sobre Distancia y Object Fade.' }
	@{ id = 'gs_ssr'; sec = 'Efectos'; onoff = $false; name = 'Reflejos del juego (SSR)'; lv = @('r_ssrEnableOverrides=true;r_ssrMaxIterations=32;r_ssrMaxRayLength=300;r_ssrCutoffDistance=250', 'r_ssrEnableOverrides=true;r_ssrMaxIterations=64;r_ssrMaxRayLength=600;r_ssrCutoffDistance=500', 'r_ssrEnableOverrides=true;r_ssrMaxIterations=128;r_ssrMaxRayLength=1000;r_ssrCutoffDistance=1000', 'r_ssrEnableOverrides=true;r_ssrMaxIterations=192;r_ssrMaxRayLength=1400;r_ssrCutoffDistance=1500', 'r_ssrEnableOverrides=true;r_ssrMaxIterations=256;r_ssrMaxRayLength=1600;r_ssrCutoffDistance=5000')
		tip = 'Reflejos en pantalla del juego (agua, charcos, suelos mojados). Pasos / largo del rayo / distancia: Bajo 32 / 300 / 250, Medio 64 / 600 / 500, Alto 128 / 1000 / 1000 (de fabrica), Ultra 192 / 1400 / 1500, Extremo 256 / 1600 / 5000 (el maximo). Cuesta GPU, sobre todo con lluvia. Manda sobre Reflections.' }
	@{ id = 'gs_tess'; sec = 'Efectos'; onoff = $false; name = 'Teselado / relieve'; lv = @('r_tessellationDensity=4.0;r_tessellationSmallestEdgeLength=0.08;r_tessellationFadeOutStart=6.0;r_tessellationFadeOutRange=2.0', 'r_tessellationDensity=8.0;r_tessellationSmallestEdgeLength=0.04;r_tessellationFadeOutStart=12.0;r_tessellationFadeOutRange=3.0', 'r_tessellationDensity=18.0;r_tessellationSmallestEdgeLength=0.02;r_tessellationFadeOutStart=15.0;r_tessellationFadeOutRange=3.0', 'r_tessellationDensity=20.0;r_tessellationSmallestEdgeLength=0.015;r_tessellationFadeOutStart=25.0;r_tessellationFadeOutRange=5.0', 'r_tessellationDensity=20.0;r_tessellationSmallestEdgeLength=0.01;r_tessellationFadeOutStart=40.0;r_tessellationFadeOutRange=8.0')
		tip = 'Teselado: relieve de verdad en suelos de piedra, barro y paredes. Bajo: menos relieve y solo de cerca. Medio: modo rendimiento de PS5. Alto: modo cine de PS5. Ultra y Extremo: mas denso y hasta 25 / 40 m. Cuesta GPU. Ojo: el juego vuelve a poner el de su modo al cargar su configuracion; si no ves cambio, es eso.' }
	@{ id = 'gs_fog'; sec = 'Efectos'; onoff = $true; name = 'Niebla volumetrica'; lv = @('fog_enable=false', 'fog_enable=true')
		tip = 'Niebla volumetrica: la bruma iluminada por el sol y las antorchas. No: unos 4-5 % mas FPS, pero cambia mucho el aspecto del juego.' }
	@{ id = 'gs_fogden'; sec = 'Efectos'; onoff = $false; name = 'Densidad de niebla'; lv = @('fog_density=0.6', 'fog_density=1.0', 'fog_density=1.5', 'fog_density=2.5', 'fog_density=4.0')
		tip = 'Cuanta niebla hay en el aire (fog_density del juego, 1.0 = como viene). Bajo 0.6, Medio 1.0, Alto 1.5, Ultra 2.5, Extremo 4.0. Mas niebla = mas ambiente y rayos de luz mas visibles. Casi no cuesta.' }
	@{ id = 'gs_fogpart'; sec = 'Efectos'; onoff = $true; name = 'Niebla con particulas'; lv = @('fog_enableParticles=false', 'fog_enableParticles=true')
		tip = 'Particulas dentro de la niebla volumetrica (motas y polvo en el aire, sobre todo con luz). Si: mas ambiente, cuesta poco. Necesita la niebla volumetrica activa.' }
	@{ id = 'gs_decal'; sec = 'Efectos'; onoff = $true; name = 'Calcomanias (sangre, suciedad)'; lv = @('r_projectedDecalEnable=false', 'r_projectedDecalEnable=true')
		tip = 'Calcomanias proyectadas: sangre, manchas, suciedad y marcas en el suelo y las paredes. No: casi la mitad de dibujados menos en 1-1 y bastantes mas FPS, pero se pierde mucho detalle.' }
	@{ id = 'gs_mblur'; sec = 'Postprocesado'; onoff = $true; name = 'Desenfoque de movimiento'; lv = @('r_motionBlurEnable=false', 'r_motionBlurEnable=true')
		tip = 'Desenfoque de movimiento del juego al girar la camara. No: imagen mas nitida en movimiento. Casi no cuesta. Manda sobre la casilla Motion Blur.' }
	@{ id = 'gs_dof'; sec = 'Postprocesado'; onoff = $true; name = 'Profundidad de campo'; lv = @('r_dofEnable=false', 'r_dofEnable=true')
		tip = 'Profundidad de campo: desenfoque del fondo en cinematicas y menus. No: todo enfocado; ahorra un poco. Manda sobre la casilla Depth of Field.' }
	@{ id = 'gs_vignette'; sec = 'Postprocesado'; onoff = $true; name = 'Vineta (bordes oscuros)'; lv = @('r_vignetteEnable=false', 'r_vignetteEnable=true')
		tip = 'Vineta: oscurecimiento de las esquinas de la imagen. No cuesta FPS. Manda sobre la casilla Vignette.' }
	@{ id = 'gs_bloom'; sec = 'Postprocesado'; onoff = $true; name = 'Bloom (resplandor)'; lv = @('r_bloomEnable=false', 'r_bloomEnable=true')
		tip = 'Bloom: resplandor alrededor de las luces y las zonas brillantes. Cuesta muy poco. Manda sobre la casilla Bloom.' }
	@{ id = 'gs_eyeadapt'; sec = 'Postprocesado'; onoff = $true; name = 'Adaptacion de la vista'; lv = @('r_exposureEyeAdaptEnable=false', 'r_exposureEyeAdaptEnable=true')
		tip = 'Adaptacion de la vista: la exposicion cambia poco a poco al pasar de zonas oscuras a claras. No: exposicion fija. No cuesta FPS. Manda sobre la casilla Eye Adaptation.' }
	@{ id = 'gs_aasharp'; sec = 'Postprocesado'; onoff = $false; name = 'Nitidez del antialiasing'; lv = @('r_aaSharpenStrengthStatic=1.2;r_aaSharpenStrengthMotion=1.2;r_aaHistorySharpeningStrength=0.5', 'r_aaSharpenStrengthStatic=2.0;r_aaSharpenStrengthMotion=2.0;r_aaHistorySharpeningStrength=0.7', 'r_aaSharpenStrengthStatic=2.5;r_aaSharpenStrengthMotion=2.5;r_aaHistorySharpeningStrength=0.8', 'r_aaSharpenStrengthStatic=3.5;r_aaSharpenStrengthMotion=3.5;r_aaHistorySharpeningStrength=0.9', 'r_aaSharpenStrengthStatic=5.0;r_aaSharpenStrengthMotion=5.0;r_aaHistorySharpeningStrength=1.0')
		tip = 'Nitidez del antialiasing temporal del juego: Bajo 1,2, Medio 2,0, Alto 2,5 (de fabrica), Ultra 3,5, Extremo 5,0 (la del historial de 0,5 a 1,0). No cuesta FPS; mucha puede marcar bordes y brillos. Manda sobre AA Sharpness.' }
)
function Get-GsIndex($s) { $i = 0; [void][int]::TryParse((Get-IniValue $s.id '0'), [ref]$i); if ($i -lt 0 -or $i -gt $s.lv.Count) { $i = 0 }; return $i }
function Get-GameSettingsConvars { return (@($script:gameSettings | ForEach-Object { $i = Get-GsIndex $_; if ($i -gt 0) { $_.lv[$i - 1] } }) -join ';') }
# (10-24) Modo Souls ligero: ajustes del juego por defecto (sombras del sol, de contacto, sombreado ambiental, iluminacion
# global, distancia, reflejos y teselado en Medio; sombras de focos en No; lo demas como viene el juego).
function Set-GsLowProfile { Set-GsLevels @{ gs_spot = 1 } }  # (10-25) contacto y sombreado ambiental en Juego  # (10-24) PC low profile
function Set-GsLight { Set-GsLevels @{ gs_spot = 1 } }
function Set-GsAll([int]$lv, [int]$spot) { Set-GsLevels @{ gs_spot = 1 } }  # (10-24) equilibrado/ultra/extremo
# (gsp) ligero / low profile / equilibrado: sol Juego, focos No, contacto Bajo, ambiental/GI/distancia/SSR/teselado Medio, resto Juego
function Set-GsUserPreset { Set-GsLevels @{ gs_spot = 1 } }
function Set-GsLevels($v) {
	foreach ($s in $script:gameSettings) { Set-IniValue $s.id "$(if ($v.ContainsKey($s.id)) { $v[$s.id] } else { 0 })" }
	if (Get-Command Update-GsLabel -ErrorAction SilentlyContinue) { Update-GsLabel }
}
function Update-GsLabel {
	$n = @($script:gameSettings | Where-Object { (Get-GsIndex $_) -gt 0 }).Count
	$lbGs.Text = if ($n) { "$n ajuste(s) del juego cambiados" } else { 'Todo predeterminado (PS5)' }
}
function Show-GameSettings {
	$f = New-Object System.Windows.Forms.Form
	$f.Text = 'Ajustes del juego'; $f.StartPosition = 'CenterScreen'; $f.FormBorderStyle = 'FixedDialog'
	$f.MaximizeBox = $false; $f.MinimizeBox = $false; $f.ShowInTaskbar = $false
	$f.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22); $f.ForeColor = $ink
	$fh = [math]::Min(860, $H - 60)
	$f.ClientSize = New-Object System.Drawing.Size(640, $fh)
	$tt = New-Object System.Windows.Forms.ToolTip; $tt.AutoPopDelay = 30000; $tt.InitialDelay = 300
	$f.Controls.Add((New-Label 'Ajustes del propio juego. Predeterminado = como viene en PS5, sin tocar nada. Pasa el raton por encima de cada uno para ver que hace y lo que cuesta.' $smallFont $gold 16 8 608 40 'TopLeft'))
	$sp = New-Object System.Windows.Forms.Panel
	$sp.AutoScroll = $true; $sp.BackColor = [System.Drawing.Color]::FromArgb(20, 22, 30)
	$sp.Location = New-Object System.Drawing.Point(16, 52); $sp.Size = New-Object System.Drawing.Size(608, ($fh - 52 - 62))
	$f.Controls.Add($sp)
	$y = 8; $sec = ''; $script:gsCombos = @()
	foreach ($s in $script:gameSettings) {
		if ($s.sec -ne $sec) { $sec = $s.sec; if ($y -gt 8) { $y += 8 }; $sp.Controls.Add((New-Label $sec $boldSmall $gold 8 $y 560 24)); $y += 28 }
		$lb = New-Label $s.name $smallFont $ink 24 ($y + 2) 300 22; $sp.Controls.Add($lb)
		$c = New-Object System.Windows.Forms.ComboBox
		$c.DropDownStyle = 'DropDownList'; $c.Font = $smallFont; $c.Tag = $s.id
		$c.Location = New-Object System.Drawing.Point(330, $y); $c.Size = New-Object System.Drawing.Size(240, 26)
		foreach ($it in $(if ($s.onoff) { $script:gsOnOff } else { $script:gsLevels })) { [void]$c.Items.Add($it) }
		$c.SelectedIndex = Get-GsIndex $s
		$c.Add_SelectedIndexChanged({ param($sender, $e) Set-IniValue $sender.Tag "$($sender.SelectedIndex)"; Update-GsLabel })
		$tt.SetToolTip($c, $s.tip); $tt.SetToolTip($lb, $s.tip)
		$sp.Controls.Add($c); $script:gsCombos += $c
		$y += 32
	}
	$sp.Controls.Add((New-Label '' $smallFont $ink 0 $y 10 8))  # (margen al final del desplazamiento)
	$btnGsReset = New-Button 'Todo predeterminado' 16 ($fh - 52) 200 40 $false; $btnGsReset.Font = $boldSmall
	$tt.SetToolTip($btnGsReset, 'Pone todos en Juego: el juego como viene, sin tocar nada.')
	$btnGsReset.Add_Click({ foreach ($c in $script:gsCombos) { $c.SelectedIndex = 0 } })
	$btnGsOk = New-Button 'Cerrar' 424 ($fh - 52) 200 40 $true; $btnGsOk.Font = $boldSmall
	$btnGsOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
	$f.Controls.Add($btnGsReset); $f.Controls.Add($btnGsOk); $f.AcceptButton = $btnGsOk; $f.CancelButton = $btnGsOk
	[void]$f.ShowDialog($form)
	$f.Dispose(); $script:gsCombos = @()
}
$btnGs = New-Button 'Ajustes del juego...' 0 0 396 34 $false
$btnGs.Font = $boldSmall
$tips.SetToolTip($btnGs, 'Ajustes del propio motor del juego, uno a uno (sombras, iluminacion, distancia, efectos y postprocesado): Juego / Bajo / Medio / Alto / Ultra / Extremo. Mandan sobre los desplegables de Calidad grafica.')
$btnGs.Add_Click({ Show-GameSettings })
$panel.Controls.Add($btnGs); Move-To $btnGs $gRt 10 90 396
$lbGs = New-Label '' $smallFont $gold 0 0 396 22
$panel.Controls.Add($lbGs); Move-To $lbGs $gRt 10 128 396
Update-GsLabel

# (10-24) Tu equipo: tarjeta con el caballero senalando la grafica y el equipo detectados (antes, una linea abajo).
function New-PcCard([int]$cw = 420, [int]$ch = 292, [double]$fs = 1.0, [int]$px = 10, [int]$py = 772, [bool]$noKnight = $false) {
	# (10-26) tamano y escala de letra por parametro: se pinta al tamano final (sin escalar el bitmap); $script:pcEndY = alto usado
	$bmp = New-Object System.Drawing.Bitmap($cw, $ch, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
	$gc = [System.Drawing.Graphics]::FromImage($bmp)
	$gc.SmoothingMode = 'AntiAlias'; $gc.TextRenderingHint = 'AntiAliasGridFit'; $gc.InterpolationMode = 'HighQualityBicubic'; $gc.PixelOffsetMode = 'HighQuality'; $gc.CompositingQuality = 'HighQuality'; $gc.TextContrast = 0  # (10-26) bitmap con alfa: AntiAliasGridFit (ClearType sale con bordes de color sobre transparente); contraste maximo
	$path = New-Object System.Drawing.Drawing2D.GraphicsPath; $r = 14
	$path.AddArc(0, 0, $r * 2, $r * 2, 180, 90); $path.AddArc($cw - 1 - $r * 2, 0, $r * 2, $r * 2, 270, 90)
	$path.AddArc($cw - 1 - $r * 2, $ch - 1 - $r * 2, $r * 2, $r * 2, 0, 90); $path.AddArc(0, $ch - 1 - $r * 2, $r * 2, $r * 2, 90, 90); $path.CloseFigure()
	$fill = New-Object System.Drawing.Drawing2D.LinearGradientBrush((New-Object System.Drawing.Point(0, 0)), (New-Object System.Drawing.Point($cw, $ch)),
		[System.Drawing.Color]::FromArgb(215, 30, 26, 20), [System.Drawing.Color]::FromArgb(215, 12, 13, 18))
	$gc.FillPath($fill, $path)
	$pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(150, 222, 196, 140), 1.5); $gc.DrawPath($pen, $path)
	$knight = $null; $kp = $(if ($root) { Join-Path $root 'caballero-espada.png' } else { '' })  # (10-26) caballero dentro de la tarjeta, espada hacia el texto
	try { if (-not $noKnight -and $kp -and (Test-Path -LiteralPath $kp)) { $knight = [System.Drawing.Image]::FromFile($kp) } } catch { $knight = $null }
	if ($knight) {
		# (10-25) mas pequeno: no cabia (como mucho el 70 % de la altura de la tarjeta)
		$ks = [math]::Min([math]::Min(1.0, ($ch - 12) / [double]$knight.Height), ($cw * 0.62) / [double]$knight.Width);  # alto = tarjeta - 12, sin pasar del 62 % del ancho
		 $kw = $knight.Width * $ks; $kh = $knight.Height * $ks
		$gc.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic; $gc.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
		$gc.DrawImage($knight, [float]($cw - $kw - 6), [float]($ch - $kh - 6), [float]$kw, [float]$kh)
	}
	$goldB = New-Object System.Drawing.SolidBrush($gold); $inkB = New-Object System.Drawing.SolidBrush($ink)
	$dimB = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(205, 195, 180))  # (10-26) mas claro
	$redB = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(235, 90, 80))
	$fHead = New-Object System.Drawing.Font('Segoe UI Black', [float](16 * $fs), [System.Drawing.FontStyle]::Bold)
	$fCap = New-Object System.Drawing.Font('Segoe UI Semibold', [float](10.5 * $fs)); $fVal = New-Object System.Drawing.Font('Segoe UI Semibold', [float](13 * $fs), [System.Drawing.FontStyle]::Bold)
	$fTxt = New-Object System.Drawing.Font('Segoe UI', [float](12 * $fs))
	$tw = $cw - 32; if ($knight) { $tw = [int]($cw - $kw * 0.55 - 32) }; $x = 16; $y = [int](12 * $fs)
	$gc.DrawString('TU EQUIPO', $fHead, $goldB, [float]$x, [float]$y); $y += [int](40 * $fs)
	$put = {
		param($cap, $val, $font, $brush)
		$gc.DrawString($cap, $fCap, $dimB, [float]$x, [float]$y); $script:pcY = $y + [int](21 * $fs)
		$sz = $gc.MeasureString($val, $font, $tw)
		$gc.DrawString($val, $font, $brush, (New-Object System.Drawing.RectangleF($x, ($y + [int](21 * $fs)), $tw, ($sz.Height + 2))))
		return ($y + [int](23 * $fs) + $sz.Height + [int](7 * $fs))
	}
	$gpu = if ($script:gpuName) { $script:gpuName } else { 'no detectada' }
	if (-not $script:isNvidia -and $script:gpuName) { $gpu += '  (sin DLSS: solo NVIDIA)' }
	$y = & $put 'Grafica' $gpu $fVal $inkB
	$y = & $put 'Procesador y RAM' "$($script:pcInfo)" $fTxt $inkB
	if ($script:gpuRec) {
		$bad = $script:gpuRec[1] -eq 'NO COMPATIBLE'
		$y = & $put 'Modo recomendado' "$($script:gpuRec[1])" $fVal $(if ($bad) { $redB } else { $goldB })
		$y = (& $put 'Esperado (estimacion)' "$($script:gpuRec[2])" $fTxt $(if ($bad) { $redB } else { $inkB }))
	} else {
		$y = (& $put 'Modo recomendado' 'Grafica sin datos: empieza por el Modo Souls equilibrado y mira las metricas.' $fTxt $inkB)
	}
	$script:pcEndY = $y
	foreach ($o in $fill, $pen, $goldB, $inkB, $dimB, $redB, $fHead, $fCap, $fVal, $fTxt, $path) { $o.Dispose() }
	if ($knight) { $knight.Dispose() }
	$gc.Dispose()
	$pb = New-Object System.Windows.Forms.PictureBox
	$pb.BackColor = [System.Drawing.Color]::Transparent; $pb.Image = $bmp
	$pb.Location = New-Object System.Drawing.Point($px, $py); $pb.Size = New-Object System.Drawing.Size($cw, $ch)
	$panel.Controls.Add($pb)
	return $pb
}
$pcCard = New-PcCard
$gCtl = New-Group 'Controles' 450 604 420 214
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
$lnkDon.Location = New-Object System.Drawing.Point(450, 830); $lnkDon.Size = New-Object System.Drawing.Size(420, 28)
$lnkDon.TextAlign = 'MiddleCenter'
$lnkDon.Add_LinkClicked({ Start-Process 'https://www.paypal.com/donate?business=todojuegos67%40gmail.com&currency_code=EUR' })
$tips.SetToolTip($lnkDon, 'Si te gusta el proyecto, puedes donar por PayPal. Gracias, portador de la marca.')
$panel.Controls.Add($lnkDon)
# (10-23) Boton oficial 'Donar' de PayPal, servido por PayPal (paypalobjects.com); si no hay internet, solo queda el texto.
$picDon = New-Object System.Windows.Forms.PictureBox
$picDon.Size = New-Object System.Drawing.Size(92, 26); $picDon.Location = New-Object System.Drawing.Point(614, 862)
$picDon.BackColor = [System.Drawing.Color]::Transparent; $picDon.Cursor = [System.Windows.Forms.Cursors]::Hand
$picDon.SizeMode = 'Zoom'; $picDon.WaitOnLoad = $false
try { $picDon.LoadAsync('https://www.paypalobjects.com/es_ES/ES/i/btn/btn_donate_LG.gif') } catch { }
$picDon.Add_Click({ Start-Process 'https://www.paypal.com/donate?business=todojuegos67%40gmail.com&currency_code=EUR' })
$tips.SetToolTip($picDon, 'Donar con PayPal')
$panel.Controls.Add($picDon)
$chkKbm = New-Check 'Teclado y raton' 0 0 190 'kbm'
$tips.SetToolTip($chkKbm, 'Jugar con teclado y raton: WASD para moverse, el raton mueve la camara, clic izquierdo R1, clic derecho L1, rueda o clic central fija objetivo. Las teclas se cambian en "Mapear mando".')
Move-To $chkKbm $gCtl 12 26 190
# (10-25) Aviso al activarlo: la camara sin aceleracion tambien afecta al stick derecho del mando.
$chkKbm.Add_CheckedChanged({ if ($chkKbm.Checked -and $chkKbm.Focused) { [void][System.Windows.Forms.MessageBox]::Show('Con Teclado y raton activado, la camara gira sin la aceleracion del juego y a su velocidad maxima (en 30x2 con Anti-Lag+, a velocidad normal). Tambien afecta al stick derecho del mando: si juegas con mando, dejalo desactivado.', 'Teclado y raton', 'OK', 'Information') } })
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
# (10-23) Velocidad de camara del juego con teclado y raton (cp11_Setting_CameraSpeed): mas alta = los giros grandes del
# raton no se quedan en el tope del stick. Sin teclado y raton, la del juego (la de su menu).
$trkCam = New-Slider 'kbmcam2' 5 15 10 12 116 { param($v) "Velocidad de camara (raton): $v$(if ($v -gt 10) { ' (extra)' })" }  # (10-24) 11-15: por encima del menu del juego
$tips.SetToolTip($trkCam, 'Solo con teclado y raton: la velocidad maxima de giro de la camara del juego. 10 = el maximo del menu del juego. 11-15 (extra, experimental): mas rapido que el menu; si la camara hace cosas raras, vuelve a 10. Al quitar teclado y raton vuelve la del menu del juego.')
$tips.SetToolTip($trkAdz, 'Salta la zona muerta del stick del juego: con mas, un movimiento pequeno del raton ya gira la camara. Si la camara se mueve sola, bajalo.')
# (10-24) Frame warp: la camara del raton desplaza el 3D al momento (HUD quieto), tambien con la frame generation de AMD.
$chkWarp = New-Check 'Camara instantanea (frame warp)' 0 0 300 'kbmwarp'
$tips.SetToolTip($chkWarp, 'Solo con teclado y raton: justo antes de mostrar cada imagen, el 3D se desplaza lo que has movido el raton (el HUD no se mueve). Funciona sin frame generation y con la de AMD; con la de NVIDIA no hace nada. Experimental.')
$panel.Controls.Add($chkWarp); Move-To $chkWarp $gCtl 214 120 200
function Update-KbmEnabled { $trkSens.Enabled = $chkKbm.Checked; $trkAdz.Enabled = $chkKbm.Checked; $trkCam.Enabled = $chkKbm.Checked; $chkWarp.Enabled = $chkKbm.Checked }
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
Move-To $btnCal $gCtl 12 170 396

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
# (10-24) Prueba para las manchas negras del suelo mojado: una cosa cada vez (se guarda; 'Ninguna' para volver).
$script:testModes = @(
	@('Prueba: ninguna', ''), @('Sin compute asincrono', 'enableAsyncCompute=false'), @('Sin sombras de focos', 'r_enableSpotShadows=false'),
	@('Sin ambient shadowing', 'r_enableAmbientShadowing=false'), @('Sin SSR del juego', 'r_ssrCutoffDistance=0'),
	@('Emu: fresh memory off', 'KYTY_FRESH_MEMORY=0'), @('Emu: partial dirty off', 'KYTY_PARTIAL_IMAGE_DIRTY=0'),
	@('Emu: resolve pages off', 'KYTY_TEXTURE_RESOLVE_PAGES=0'), @('Emu: image pool off', 'KYTY_IMAGE_POOL=0'), @('Emu: texture reduce off', 'KYTY_TEXTURE_REDUCE=0'), @('Emu: sin arreglo focos', 'KYTY_DEPTH_ARRAY_GUARD=0'), @('Emu: sin limite de capas', 'KYTY_TEXTURE_LAYER_BOUND=0'),
	@('Lejos: cascadas 400 m', 'sunShadowMaxDistance=400;r_virtualGeometryNumShadowCascades=4;sunShadowCascadeDistance0=27;sunShadowCascadeDistance1=61;sunShadowCascadeDistance2=131;sunShadowCascadeDistance3=400;sunShadowLightDistance=1000'),
	@('Lejos: sin sombras cascada', 'r_enableCascadeShadows=false'), @('Lejos: sombra con fundido', 'r_EnableShadowFadeout=true'),
	@('Lejos: GI sin areas', 'GIEnableAreasOfInterest=false'), @('Lejos: GI areas de fondo', 'GIEnableBackgroundAOI=true;GIEnableDefaultCameraAOI=true'),
	@('Lejos: sin oclusion IBL', 'r_enableIBLOcclusion=false'), @('Lejos: sin culling luces', 'r_enableLightOcclusionCulling=false'),
	@('Lejos: sombreado amb. 1000', 'r_ambientShadowCutoffDistance=1000;r_ambientShadowCutoffFadeDistance=100'))
$cmbTest = New-Object System.Windows.Forms.ComboBox
$cmbTest.DropDownStyle = 'DropDownList'; $cmbTest.Font = $smallFont
foreach ($t in $script:testModes) { [void]$cmbTest.Items.Add($t[0]) }
$tv = 0; [void][int]::TryParse((Get-IniValue 'pruebamanchas' '0'), [ref]$tv); $cmbTest.SelectedIndex = [math]::Min([math]::Max(0, $tv), $script:testModes.Count - 1)
$cmbTest.Add_SelectedIndexChanged({ Set-IniValue 'pruebamanchas' "$($cmbTest.SelectedIndex)" })
$cmbTest.Location = New-Object System.Drawing.Point(410, 100); $cmbTest.Size = New-Object System.Drawing.Size(180, 26)
$tips.SetToolTip($cmbTest, 'Solo para pruebas: quita una cosa del juego o del emulador para encontrar el fallo de las manchas negras. Deja Prueba: ninguna para jugar normal.')
$settings.Controls.Add($cmbTest)
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
# (10-25) Las opciones de ray tracing juntas: Luz RT y Oclusion pasan de "Calidad grafica" al grupo "Ray tracing"
# (con Luz volumetrica y Reflejos RT). Las filas de debajo suben para no dejar huecos.
try {
	foreach ($c in @($lbRt, $cmbRt, $lbAo, $cmbAo)) { if ($c) { $gGfx.Controls.Remove($c) } }
	foreach ($c in @($gGfx.Controls)) {
		$yy = $c.Location.Y
		if ($yy -ge 148 -and $yy -le 162) { $c.Top = $yy - 32 } elseif ($yy -ge 210) { $c.Top = $yy - 64 }
	}
	$gGfx.Height = $gGfx.Height - 64
	foreach ($c in @($gRt.Controls)) { $c.Top = $c.Top + 64 }
	$gRt.Top = $gRt.Top - 64; $gRt.Height = $gRt.Height + 64; $gRt.Text = 'Ray tracing'
	Move-To $lbRt $gRt 10 26 112; Move-To $cmbRt $gRt 124 24 282
	Move-To $lbAo $gRt 10 58 112; Move-To $cmbAo $gRt 124 56 282
} catch { }
# (10-25 noche) combo22: cuatro filas nuevas en "Ray tracing" (debajo de Reflejos RT); el boton de ajustes del juego y la tarjeta del PC bajan
$cmbSun = $null; $cmbShQ = $null; $cmbShD = $null; $cmbPart = $null
try {
	$btnGs.Top = $btnGs.Top + 128; $lbGs.Top = $lbGs.Top + 128; $gRt.Height = $gRt.Height + 160
	if ($pcCard) { $pcCard.Top = $pcCard.Top + 160 }
} catch { }
# (10-26) RT real: cuatro filas mas en 'Ray tracing' (solo efectos probados)
$cmbHRef = $null; $cmbHSun = $null; $cmbHLamp = $null; $cmbHVol = $null; $cmbHPart = $null; $cmbHMetal = $null; $cmbHAo = $null
try {
	$btnGs.Top = $btnGs.Top + 128; $lbGs.Top = $lbGs.Top + 128; $gRt.Height = $gRt.Height + 128
	if ($pcCard) { $pcCard.Top = $pcCard.Top + 128 }
	# (10-26) 5 niveles: claves nuevas '<clave>5'; si faltan se convierte el indice antiguo
	$hwMig = @{ hwrefl = @(0, 3); hwsun = @(0, 3, 4) }
	foreach ($k in $hwMig.Keys) {
		if ((Get-IniValue "${k}5" '') -ne '') { continue }
		$ov = 0; [void][int]::TryParse((Get-IniValue $k '0'), [ref]$ov)
		$m = $hwMig[$k]; $nv = $m[[math]::Min([math]::Max(0, $ov), $m.Count - 1)]
		Set-IniValue "${k}5" "$nv"
	}
	# (10-27) migra niveles (*5) a No/Si (*2): nivel > 0 -> 1
	foreach ($p in @('hwrefl', 'hwsun')) { if ((Get-IniValue "${p}2" '') -eq '') { $o = 0; [void][int]::TryParse((Get-IniValue "${p}5" '0'), [ref]$o); Set-IniValue "${p}2" $(if ($o -gt 0) { '1' } else { '0' }) } }
	$cmbHRef = New-RtCombo 'Reflejos' 'hwrefl2' 280 'Reflejos con trazado de rayos: refleja tambien lo que esta fuera de pantalla o tapado, con sus texturas.' @('No', 'Sí')
	$cmbHSun = New-RtCombo 'Sombras sol' 'hwsun2' 312 'Sombras del sol con trazado de rayos: sombras de contacto y las que faltaban.' @('No', 'Sí')
	if ((Get-IniValue 'hwlamp4' '') -eq '') { $o = 0; [void][int]::TryParse((Get-IniValue 'hwlamp2' '0'), [ref]$o); Set-IniValue 'hwlamp4' $(if ($o -gt 0) { '2' } else { '0' }) }  # (fx6) No/Si -> niveles
	$cmbHLamp = New-RtCombo 'Antorchas' 'hwlamp4' 344 'Antorchas, lamparas, fuego y magia iluminan lo que tienen cerca con trazado de rayos y sombras reales.' @('No', 'Bajo', 'Medio', 'Alto')
	$cmbHGi = New-RtCombo 'Luz rebotada' 'hwgi' 376 'Luz que rebota con trazado de rayos: el fuego tiñe las paredes, la luz de las aperturas llena las salas. Incluye oclusion ambiental estable.' @('No', 'Bajo', 'Medio', 'Alto')  # (fx6)
	$cmbHCon = New-RtCombo 'Contacto' 'hwcontact' 408 'Sombras finas y nitidas donde personajes, armaduras y objetos tocan suelo y paredes (sol y luz mas fuerte).' @('No', 'Bajo', 'Medio', 'Alto')  # (fx6)
	$cmbHSoft = New-RtCombo 'Sol suave' 'hwsoftsun' 440 'Sombras del sol con penumbra real: nitidas al contacto y suaves con la distancia; el sol RT funciona siempre en exteriores.' @('No', 'Bajo', 'Medio', 'Alto')  # (fx6)
	$cmbHWater = New-RtCombo 'Agua y charcos' 'hwwater' 472 'Reflejos nitidos en agua, charcos y piedra mojada con trazado de rayos.' @('No', 'Bajo', 'Medio', 'Alto')  # (fx6)
	$cmbHVolSh = New-RtCombo 'Rayos de luz RT' 'hwvolsh' 504 'Los haces de la Luz volumetrica se cortan con columnas, ventanas y rejas (necesita Luz volumetrica en Medio o Alto).' @('No', 'Bajo', 'Medio', 'Alto')  # (fx6)
} catch { }
# (10-26) Memoria: un solo vaciado (el fuerte). Las simulaciones de 16 GB / grafica 8 GB se quedan (pruebas).
if ($chkRamFlush -and $chkRamFlush.Checked -and $chkRamFlush2) { $chkRamFlush2.Checked = $true }
foreach ($c in @($chkRamFlush)) { if ($c) { $c.Checked = $false; $c.Visible = $false } }
if ($chkRamFlush2) { $chkRamFlush2.Text = 'Vaciado RAM' }
# (10-26) Layout: el vaciado en el hueco izquierdo; Controles debajo de Settings/Salir sin pisarlos.
if ($chkRamFlush2 -and $gPc) { $chkRamFlush2.Location = New-Object System.Drawing.Point(12, 84) }
if ($btnCtl -and $btnSettings) { $btnCtl.Top = $btnSettings.Bottom + 10; $btnCtl.Height = $btnSettings.Height; $btnCtl.Font = $btnSettings.Font }
if ($gCtl -and $btnCtl) { $gCtl.Top = $btnCtl.Bottom + 6 }
# (audit) Casilla Juego original: al marcarla las filas de efectos quedan inactivas (sus valores se guardan).
$chkPure = New-Check 'Juego original (sin efectos añadidos)' 0 0 396 'puregame'
$tips.SetToolTip($chkPure, 'Apaga todos los efectos visuales que anade el emulador (ray tracing, luz volumetrica, sombras RT, brillo, oclusion, SSGI, reflejos RT). Se ve el juego tal cual. Rendimiento, reescalador, frame generation y latencia no cambian.')
function Update-PureGame { foreach ($c in @($cmbRtRf, $cmbRt, $cmbHRef, $cmbHSun, $cmbHLamp, $cmbHGi, $cmbHCon, $cmbHSoft, $cmbHWater, $cmbHVolSh, $cmbRays, $cmbAoNew, $cmbRtSh)) { if ($c) { $c.Enabled = -not $chkPure.Checked } } }
$chkPure.Add_CheckedChanged({ Update-PureGame })
# (10-26) "Ray tracing" solo con RT real: fuera Luz RT (SSGI), Luz volumetrica de pantalla, Reflejos RT, Sol RT y Particulas RT de pantalla.
# (Visible no sirve para leer antes de mostrar la ventana: se trabaja con listas.)
try {
	# (ds-rt1026-restore) el RT del 10-25/26 (c3e58bd5): Reflejos RT, Luz RT (SSGI) y Luz volumetrica; el RT real oculto (rthwexp=1 lo muestra)
	$rtHwExp = (Get-IniValue 'rthwexp' '0') -eq '1'
	$fake = @($cmbSun, $cmbPart, $cmbHGi, $cmbHCon, $cmbHSoft, $cmbHWater) | Where-Object { $_ }  # (10-27) solo reflejos, sombras RT y rayos volumetricos
	$hideOnly = @(if (-not $rtHwExp) { @($cmbHRef, $cmbHSun, $cmbHLamp, $cmbHVolSh) | Where-Object { $_ } })  # (vol25) la Luz volumetrica ya no se oculta
	$keep = @(@($cmbRtRf, $cmbRt, $cmbRays, $cmbAoNew, $cmbRtSh) | Where-Object { $_ }) + @(if ($rtHwExp) { @($cmbHRef, $cmbHSun, $cmbHLamp, $cmbHVolSh) | Where-Object { $_ } | Sort-Object Top })  # (vol25) Luz volumetrica del 10-25 visible otra vez, debajo
	$lblOf = @{}
	foreach ($c in @($fake + $hideOnly + $keep + @($cmbShQ, $cmbShD, $cmbAo) | Where-Object { $_ })) {
		$lblOf[$c.GetHashCode()] = @($gRt.Controls | Where-Object { $_ -is [System.Windows.Forms.Label] -and [math]::Abs($_.Top - ($c.Top + 2)) -le 4 })
	}
	foreach ($c in $fake) {
		if ($c.Items.Count -gt 0) { $c.SelectedIndex = 0 }
		$c.Visible = $false
		foreach ($o in $lblOf[$c.GetHashCode()]) { $o.Visible = $false }
	}
	foreach ($c in $hideOnly) { $c.Visible = $false; foreach ($o in $lblOf[$c.GetHashCode()]) { $o.Visible = $false } }
	# (ds-rt1026-restore) Luz RT visible otra vez
	# (10-26) Oclusion (GTAO) fuera; Calidad sombra y Profundidad pasan a 'Calidad grafica' (debajo del post-proceso)
	if ($cmbAo) { $cmbAo.SelectedIndex = 0; $cmbAo.Visible = $false; foreach ($o in $lblOf[$cmbAo.GetHashCode()]) { $o.Visible = $false } }
	$gy = 0; foreach ($pc in @($script:postChecks)) { if ($pc[0].Bottom -gt $gy) { $gy = $pc[0].Bottom } }
	if ($gy -le 0) { $gy = $gGfx.Height - 10 }
	$gy += 8; $gAdd = 0
	foreach ($c in @($cmbShQ, $cmbShD) | Where-Object { $_ }) {
		foreach ($o in $lblOf[$c.GetHashCode()]) { $gRt.Controls.Remove($o); $gGfx.Controls.Add($o); $o.Top = $gy + 2; $o.Left = 10; $o.Width = 112 }
		$gRt.Controls.Remove($c); $gGfx.Controls.Add($c); $c.Top = $gy; $c.Left = 124; $c.Width = 282; $gy += 32; $gAdd += 32
	}
	if ($gAdd -gt 0) { $nh = $gy + 6; $d = $nh - $gGfx.Height; if ($d -gt 0) { $gGfx.Height = $nh; $gRt.Top = $gRt.Top + $d } }
	$y = 24
	foreach ($c in $keep) {
		foreach ($o in $lblOf[$c.GetHashCode()]) { $o.Top = $y + 2; $o.Left = 10; $o.Width = 128 }
		$c.Top = $y; $c.Left = 142; $c.Width = 264; $y += 30
	}
	if ($chkPure) { Move-To $chkPure $gRt 10 $y 396; $y += 30; Update-PureGame }  # (audit)
	# Boton de ajustes del juego y su aviso debajo de la ultima fila, dentro del grupo; la tarjeta del PC justo debajo del grupo.
	if ($btnGs) { $btnGs.Top = $y + 6; $y = $btnGs.Top + $btnGs.Height }
	if ($lbGs) { $lbGs.Top = $y + 6; $y = $lbGs.Top + $lbGs.Height }
	$gRt.Height = $y + 10
	if ($pcCard) { $pcCard.Top = $gRt.Top + $gRt.Height + 12 }
} catch { }
# (10-26) Al elegir cualquier Modo Souls: todo el ray tracing en No (el usuario lo enciende a mano despues).
function Set-RtAllOff { foreach ($c in @($cmbHRef, $cmbHSun, $cmbHLamp, $cmbHGi, $cmbHCon, $cmbHSoft, $cmbHWater, $cmbHVolSh, $cmbRt, $cmbRtRf, $cmbAoNew, $cmbRtSh)) { if ($c -and $c.Items.Count -gt 0) { $c.SelectedIndex = 0 } } }
# (vol25) Luz volumetrica y Sombras sol RT comparten el pase del sol en el motor: con las dos, el sol de la volumetrica
# llega a ratos (niebla que salta). Se excluyen: elegir una pone la otra en No.
if ($false -and $cmbRays -and $cmbHSun) {  # (rt-vol 1ac512fe) ya pueden ir juntas
	if ($cmbRays.SelectedIndex -gt 0 -and $cmbHSun.SelectedIndex -gt 0) { $cmbHSun.SelectedIndex = 0 }
	$cmbRays.Add_SelectedIndexChanged({ if ($cmbRays.SelectedIndex -gt 0 -and $cmbHSun.SelectedIndex -gt 0) { $cmbHSun.SelectedIndex = 0 } })
	$cmbHSun.Add_SelectedIndexChanged({ if ($cmbHSun.SelectedIndex -gt 0 -and $cmbRays.SelectedIndex -gt 0) { $cmbRays.SelectedIndex = 0 } })
}
foreach ($m in @($script:soulsModes | Where-Object { $_ })) { $m.Add_CheckedChanged({ param($sender, $e) if ($sender.Checked) { Set-RtAllOff } }) }
# (10-26) El RT real va en su apartado: la Prueba arranca siempre en 'ninguna' (los indices guardados ya no coinciden).
if ($cmbTest -and $cmbTest.Items.Count -gt 0) { $cmbTest.SelectedIndex = 0 }
# (10-26) Una sola casilla: perfil 16 GB + vaciado RAM
if ($chkPc16 -and $chkRamFlush2) {
	if ($chkRamFlush2.Checked) { if ($chkPc16.Enabled) { $chkPc16.Checked = $true } }
	$chkRamFlush2.Checked = $chkPc16.Checked
	$chkRamFlush2.Visible = $false
	$chkPc16.Text = 'Optimizacion equipos 16 GB RAM'; $chkPc16.Width = 300
	$tips.SetToolTip($chkPc16, 'Solo aplica el perfil de memoria de 16 GB (KYTY_LOWRAM / -Ram16) en PCs reales de 20 GB de RAM o menos, o con "Simular 16 GB RAM". No cambia los graficos.')
	$chkPc16.Add_CheckedChanged({ $chkRamFlush2.Checked = $chkPc16.Checked })
}
# (10-26) PayPal debajo de Controles (antes quedaba tapado) y boton con respaldo si la imagen no carga
if ($lnkDon -and $gCtl) {
	$lnkDon.Left = $gCtl.Left; $lnkDon.Width = $gCtl.Width; $lnkDon.Top = $gCtl.Bottom + 8; $lnkDon.BringToFront()
	if ($picDon) {
		$picDon.Size = New-Object System.Drawing.Size(120, 34)
		$picDon.Left = $gCtl.Left + [int](($gCtl.Width - 120) / 2); $picDon.Top = $lnkDon.Bottom + 4; $picDon.BringToFront()
		$picDon.ErrorImage = $null; $picDon.InitialImage = $null
		$script:donTry = 0
		$picDon.Add_LoadCompleted({ param($s, $e)
			if ($e.Error) {
				$script:donTry++
				if ($script:donTry -eq 1) { try { $picDon.LoadAsync('https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif') } catch { } }
				else { $picDon.Visible = $false; if ($btnPay) { $btnPay.Visible = $true } }
			}
		})
		$btnPay = New-Object System.Windows.Forms.Button
		$btnPay.Text = 'PayPal  Donar'; $btnPay.FlatStyle = 'Flat'; $btnPay.FlatAppearance.BorderSize = 0
		$btnPay.BackColor = [System.Drawing.Color]::FromArgb(255, 196, 57); $btnPay.ForeColor = [System.Drawing.Color]::FromArgb(0, 48, 135)
		$btnPay.Font = New-Object System.Drawing.Font('Segoe UI', 10, [System.Drawing.FontStyle]::Bold)
		$btnPay.Size = New-Object System.Drawing.Size(140, 34); $btnPay.Location = New-Object System.Drawing.Point(($gCtl.Left + [int](($gCtl.Width - 140) / 2)), $picDon.Top)
		$btnPay.Cursor = [System.Windows.Forms.Cursors]::Hand; $btnPay.Visible = $false
		$btnPay.Add_Click({ Start-Process 'https://www.paypal.com/donate?business=todojuegos67%40gmail.com&currency_code=EUR' })
		$picDon.Parent.Controls.Add($btnPay); $btnPay.BringToFront()
	}
}
# (10-26) Sin scroll. Columna derecha: boton Controles, [grupo Controles si esta abierto], TU EQUIPO (llena el hueco,
# caballero dentro), PayPal fijo abajo. Se recoloca al abrir/cerrar Controles. La columna izquierda queda vacia bajo 'Ray tracing'.
function Update-RightCol([bool]$ctlOpen) {
	try {
		$par = $btnCtl.Parent; if (-not $par) { return }
		$bot = $par.ClientSize.Height - 8
		# PayPal abajo del todo
		$payTop = $bot
		if ($picDon) { $picDon.Top = $bot - 34; $payTop = $picDon.Top }
		if ($btnPay) { $btnPay.Top = $bot - 34; $payTop = [Math]::Min($payTop, $btnPay.Top) }
		if ($lnkDon) { $lnkDon.Top = $payTop - 4 - $lnkDon.Height; $payTop = $lnkDon.Top }
		$top = $btnCtl.Bottom + 6
		if ($gCtl) { $gCtl.Top = $top; if ($ctlOpen) { $top = $gCtl.Bottom + 6 } }
		$cH = $payTop - 8 - $top
		if ($script:pcCard) { $par.Controls.Remove($script:pcCard); if ($script:pcCard.Image) { $script:pcCard.Image.Dispose() }; $script:pcCard.Dispose(); $script:pcCard = $null }
		if ($cH -lt 60) { return }
		$card = $null
		foreach ($nk in $false, $true) {
			foreach ($fs in 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55) {
				if ($card) { $panel.Controls.Remove($card); $card.Image.Dispose(); $card.Dispose(); $card = $null }
				$card = New-PcCard $btnCtl.Width $cH $fs $btnCtl.Left $top $nk
				if ($script:pcEndY -le $cH - 6) { break }
			}
			if ($script:pcEndY -le $cH - 6) { break }
		}
		if ($card.Parent -ne $par) { $par.Controls.Add($card) }
		$card.Location = New-Object System.Drawing.Point($btnCtl.Left, $top)
		$script:pcCard = $card
		$card.SendToBack()
		if ($gCtl) { $gCtl.BringToFront() }
		foreach ($o in @($lnkDon, $picDon, $btnPay)) { if ($o) { $o.BringToFront() } }
	} catch { }
}
if ($btnCtl) {
	if ($gCtl) { $gCtl.Visible = $false }; $btnCtl.Text = 'Controles  v'
	Update-RightCol $false
	$btnCtl.Add_Click({ Update-RightCol ($btnCtl.Text -eq 'Controles  ^') })
}
# Sin barra horizontal: todo cabe en 880 de ancho (la vertical aparecia y empujaba el ancho)
try { $panel.AutoScroll = $false; $panel.HorizontalScroll.Maximum = 0; $panel.HorizontalScroll.Visible = $false; $panel.HorizontalScroll.Enabled = $false; $panel.AutoScroll = $true } catch { }
# (10-27 startup) Estado final ANTES de mostrar (antes iba en Shown y se veia recolocar). Mismo orden de antes.
try { Update-AntiLagBox } catch { }
try { Update-ResLock; Update-LowEnd } catch { }
# (10-27 trofeos) Seccion desplegable Trofeos: los obtenidos (grado y nombre en el idioma del juego) y el % completado.
# Lo escribe el emulador: _SaveData\PPSA01341\trophies.json (desbloqueos) y trophies-list.json (la lista del .ucp).
$trophyDir    = Join-Path $root '_SaveData\PPSA01341'
$trophyGrades = @{ 'P' = 'Platino'; 'G' = 'Oro'; 'S' = 'Plata'; 'B' = 'Bronce' }
$trophyPts    = @{ 'P' = 300; 'G' = 90; 'S' = 30; 'B' = 15 }  # (los puntos de PSN: el % como el de la consola)
$trophyColors = @{ 'P' = [System.Drawing.Color]::FromArgb(170, 200, 235); 'G' = [System.Drawing.Color]::FromArgb(236, 196, 72); 'S' = [System.Drawing.Color]::FromArgb(200, 205, 212); 'B' = [System.Drawing.Color]::FromArgb(205, 127, 70) }
function Get-TrophySummary {
	$r = @{ Earned = @(); Total = 0; Count = 0; Pct = 0 }
	try {
		$listFile = Join-Path $trophyDir 'trophies-list.json'; $unFile = Join-Path $trophyDir 'trophies.json'
		$list = @(); if (Test-Path -LiteralPath $listFile) { $list = @((Get-Content -LiteralPath $listFile -Raw -Encoding UTF8 | ConvertFrom-Json).trophies) }
		$un = @{}
		if (Test-Path -LiteralPath $unFile) { $j = Get-Content -LiteralPath $unFile -Raw -Encoding UTF8 | ConvertFrom-Json; if ($j.unlocked) { foreach ($p in $j.unlocked.PSObject.Properties) { $un[[string]$p.Name] = [long]$p.Value } } }
		$r.Total  = $list.Count
		$r.Earned = @($list | Where-Object { $un.ContainsKey([string]$_.id) } | Sort-Object { $un[[string]$_.id] })
		$r.Count  = $r.Earned.Count
		$all = 0; $got = 0
		foreach ($t in $list) { $p = $trophyPts[[string]$t.grade]; if (-not $p) { $p = 15 }; $all += $p; if ($un.ContainsKey([string]$t.id)) { $got += $p } }
		if ($all -gt 0) { $r.Pct = [math]::Floor(100 * $got / $all) }
	} catch { }
	return $r
}
$trophyW = 420; $trophyListH = 360
$btnTrophy = New-Button 'Trofeos  v' 24 ($H - 70) $trophyW 46 $false
$btnTrophy.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 12, [System.Drawing.FontStyle]::Bold)
$btnTrophy.TextAlign = 'MiddleLeft'
try {  # (10-27 trofeos) icono de copa dibujado (dorado)
	$ti = New-Object System.Drawing.Bitmap 30, 30; $tg = [System.Drawing.Graphics]::FromImage($ti); $tg.SmoothingMode = 'AntiAlias'
	$tb = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(236, 196, 72)); $tp = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(236, 196, 72)), 2.4
	$tg.FillPie($tb, 7, -6, 16, 22, 0, 180); $tg.DrawArc($tp, 2, 2, 8, 9, 90, 180); $tg.DrawArc($tp, 20, 2, 8, 9, 270, 180)
	$tg.FillRectangle($tb, 13, 15, 4, 7); $tg.FillRectangle($tb, 9, 22, 12, 3); $tg.FillRectangle($tb, 7, 25, 16, 3)
	$tg.Dispose(); $btnTrophy.Image = $ti; $btnTrophy.ImageAlign = 'MiddleLeft'; $btnTrophy.TextImageRelation = 'ImageBeforeText'
} catch { }
$form.Controls.Add($btnTrophy)
$gTrophy = New-Object System.Windows.Forms.Panel
$gTrophy.Location = New-Object System.Drawing.Point(24, ($H - 76 - $trophyListH)); $gTrophy.Size = New-Object System.Drawing.Size($trophyW, $trophyListH)
$gTrophy.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22); $gTrophy.Visible = $false
$form.Controls.Add($gTrophy)
$lbTrophyPct = New-Label '' $smallFont $gold 10 6 ($trophyW - 20) 26
$gTrophy.Controls.Add($lbTrophyPct)
$pbTrophy = New-Object System.Windows.Forms.ProgressBar
$pbTrophy.Location = New-Object System.Drawing.Point(10, 34); $pbTrophy.Size = New-Object System.Drawing.Size(($trophyW - 20), 8); $pbTrophy.Maximum = 100
$gTrophy.Controls.Add($pbTrophy)
$lvTrophy = New-Object System.Windows.Forms.ListView
$lvTrophy.View = 'Details'; $lvTrophy.FullRowSelect = $true; $lvTrophy.HeaderStyle = 'Nonclickable'; $lvTrophy.BorderStyle = 'None'
$lvTrophy.Location = New-Object System.Drawing.Point(10, 50); $lvTrophy.Size = New-Object System.Drawing.Size(($trophyW - 20), ($trophyListH - 60))
$lvTrophy.BackColor = [System.Drawing.Color]::FromArgb(22, 25, 33); $lvTrophy.ForeColor = $ink; $lvTrophy.Font = New-Object System.Drawing.Font('Segoe UI', 10)
[void]$lvTrophy.Columns.Add('Grado', 90); [void]$lvTrophy.Columns.Add('Nombre', ($trophyW - 20 - 90 - 22))
$gTrophy.Controls.Add($lvTrophy)
function Update-TrophySection([bool]$open) {
	$s = Get-TrophySummary
	$arrow = if ($open) { '^' } else { 'v' }
	$btnTrophy.Text = if ($s.Total -gt 0) { 'Trofeos   ' + $s.Count + ' / ' + $s.Total + '   (' + $s.Pct + '%)  ' + $arrow } else { 'Trofeos  ' + $arrow }
	$gTrophy.Visible = $open
	if (-not $open) { return }
	$lbTrophyPct.Text = if ($s.Total -gt 0) { '' + $s.Pct + '% completado   -   ' + $s.Count + ' / ' + $s.Total } else { 'Aun no hay trofeos (se crean al jugar)' }
	$pbTrophy.Value = [math]::Max(0, [math]::Min(100, $s.Pct))
	$lvTrophy.BeginUpdate(); $lvTrophy.Items.Clear()
	foreach ($t in $s.Earned) {
		$g = [string]$t.grade; $gt = $trophyGrades[$g]; if (-not $gt) { $gt = [string]$t.grade_text }
		$it = New-Object System.Windows.Forms.ListViewItem($gt)
		[void]$it.SubItems.Add([string]$t.name)
		$it.UseItemStyleForSubItems = $false
		if ($trophyColors[$g]) { $it.ForeColor = $trophyColors[$g] }
		$it.SubItems[1].ForeColor = $ink
		$it.ToolTipText = [string]$t.detail
		[void]$lvTrophy.Items.Add($it)
	}
	$lvTrophy.ShowItemToolTips = $true
	$lvTrophy.EndUpdate()
	$gTrophy.BringToFront(); $btnTrophy.BringToFront()
}
$btnTrophy.Add_Click({ Update-TrophySection (-not $gTrophy.Visible) })
try { Update-TrophySection $false } catch { }
function Set-DoubleBufferAll($p) { foreach ($c in @($p.Controls)) { if ($c -is [System.Windows.Forms.Panel] -or $c -is [System.Windows.Forms.GroupBox] -or $c -is [System.Windows.Forms.PictureBox]) { Set-DoubleBuffer $c }; if ($c.Controls.Count) { Set-DoubleBufferAll $c } } }
Set-DoubleBufferAll $form
$panel.ResumeLayout($true); $form.ResumeLayout($true)
[void]$form.Handle; $form.PerformLayout()
$form.Add_Shown({ $form.Refresh(); $form.Opacity = 1 })
[void]$form.ShowDialog()
