# Metricas: ventana pequena siempre encima con GPU, VRAM, CPU, RAM y FPS del emulador.
# Uso: powershell -ExecutionPolicy Bypass -File metricas.ps1 [-VramMB 8192] [-LogDir "C:\DEMON SOULS TURBO\logs"]
# -VramMB: si el emulador simula una grafica mas pequena (KYTY_SIMULATE_VRAM_MB), la VRAM se muestra
#          sobre ese maximo, restando lo que el emulador reserva para la simulacion.
param(
	[int]$VramMB = 0,
	[int]$MaxMB = 0,
	[int]$RamGB = 0,
	[switch]$LossLess,
	[string]$LogDir = "",
	[string]$Titulo = "DEMON'S SOULS REMAKE"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
# Limite de RAM solo para el emulador: como mucho ($RamGB - 4) GB en memoria fisica (4 GB para Windows), el resto
# a disco, como en un PC con esa RAM. Windows y los demas programas no se tocan.
Add-Type @"
using System; using System.Runtime.InteropServices;
public static class RamLimit {
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
}
"@
$script:limitedPid = 0

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
	foreach ($p in @("$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe", "$env:SystemRoot\System32\nvidia-smi.exe")) {
		if (Test-Path $p) { $smi = $p; break }
	}
} else { $smi = $smi.Source }


# (10-22) Nitido en 4K y 1440p: el proceso trabaja en pixeles reales (sin el escalado borroso de Windows, que
# ademas agrandaba las letras pero no los huecos y las metricas se pisaban). Solo afecta a esta ventana.
Add-Type @"
using System; using System.Runtime.InteropServices;
public static class DpiFix { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }
"@
try { [void][DpiFix]::SetProcessDPIAware() } catch { }
$scr = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
# Tamano segun la altura de la pantalla: 1 en 1080p, 1.33 en 1440p, 2 en 4K.
$S = [Math]::Max(1.0, [Math]::Min(2.5, $scr.Height / 1080.0))
function Px([double]$v) { [int][Math]::Round($v * $S) }
$fontRow   = New-Object System.Drawing.Font("Consolas", [single](15 * $S), [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$fontTitle = New-Object System.Drawing.Font("Consolas", [single](16 * $S), [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$fontFps   = New-Object System.Drawing.Font("Consolas", [single](34 * $S), [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$fontSmall = New-Object System.Drawing.Font("Consolas", [single](10 * $S), [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
$DOT = ' ' + [char]0x00B7 + ' '   # separador (punto medio)
$ARROW = [string][char]0x2192     # flecha "->"

$form = New-Object System.Windows.Forms.Form
$form.Text = "Metricas"
$form.FormBorderStyle = 'None'
$form.TopMost = $true
$form.ShowInTaskbar = $false
$form.BackColor = [System.Drawing.Color]::Black
$form.Opacity = 0.8
$form.StartPosition = 'Manual'
$form.AutoScaleMode = 'None'
$form.Location = New-Object System.Drawing.Point((Px 10), (Px 10))
$form.Size = New-Object System.Drawing.Size((Px 420), (Px 400))

# Todas las metricas se dibujan en una sola superficie, en dos columnas (nombre | valor) de ancho fijo:
# una fila por metrica, alto de fila medido con la fuente, sin controles apilados que se pisen.
# Cada fila: @(nombre, valor, color). Nombre vacio: el valor ocupa las dos columnas (titulo, modo FG...).
$script:rows = @(,@('', 'Cargando...', [System.Drawing.Color]::Orange))
$script:fpsText = ''; $script:fpsColor = [System.Drawing.Color]::White
$label = New-Object System.Windows.Forms.PictureBox
$label.Dock = 'Fill'; $label.BackColor = [System.Drawing.Color]::Black
$colName = [System.Drawing.Color]::FromArgb(150, 150, 150)
$fmtEll = New-Object System.Drawing.StringFormat
$fmtEll.Trimming = [System.Drawing.StringTrimming]::EllipsisCharacter
$fmtEll.FormatFlags = [System.Drawing.StringFormatFlags]::NoWrap
$fmtCenter = New-Object System.Drawing.StringFormat
$fmtCenter.Alignment = 'Center'; $fmtCenter.Trimming = 'EllipsisCharacter'; $fmtCenter.FormatFlags = 'NoWrap'
$script:layout = @{ nameW = 0; rowH = 0 }
function Measure-Layout {
	# Ancho de cada columna: el texto mas largo (con un minimo, para que no baile cada segundo).
	$g = $label.CreateGraphics()
	try {
		$rowH = [Math]::Ceiling($fontRow.GetHeight($g) * 1.12)
		$nameW = (Px 90); $valW = (Px 230); $wideW = 0
		foreach ($r in $script:rows) {
			$w = [Math]::Ceiling($g.MeasureString($r[1], $fontRow).Width)
			if ($r[0]) { $nameW = [Math]::Max($nameW, [Math]::Ceiling($g.MeasureString($r[0], $fontRow).Width) + (Px 10)); $valW = [Math]::Max($valW, $w) }
			else { $wideW = [Math]::Max($wideW, $w) }
		}
		$titleH = [Math]::Ceiling($fontTitle.GetHeight($g) * 1.4)
		$fpsH = if ($script:fpsText) { [Math]::Ceiling($fontFps.GetHeight($g) * 1.1) } else { 0 }
		$pad = (Px 8)
		$W = [Math]::Max($nameW + $valW, $wideW) + 2 * $pad
		$W = [int][Math]::Min($W, $scr.Width * 0.4)
		$H = [int]($titleH + $script:rows.Count * $rowH + $fpsH + $graph.Height + 2 * $pad)
		$script:layout = @{ nameW = $nameW; rowH = $rowH; titleH = $titleH; fpsH = $fpsH; pad = $pad }
		# Solo se redimensiona si cambia de verdad (sin parpadeo).
		if ([Math]::Abs($form.Width - $W) -gt (Px 6) -or $form.Height -ne $H) { $form.Size = New-Object System.Drawing.Size($W, $H) }
	} finally { $g.Dispose() }
}
$label.Add_Paint({ param($s, $e)
	$g = $e.Graphics; $L = $script:layout; if (-not $L.rowH) { return }
	$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
	$pad = $L.pad; $w = $label.Width - 2 * $pad
	$b = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
	$g.DrawString($Titulo, $fontTitle, $b, (New-Object System.Drawing.RectangleF([single]$pad, [single]$pad, [single]$w, [single]$L.titleH)), $fmtCenter)
	$y = $pad + $L.titleH
	$bn = New-Object System.Drawing.SolidBrush($colName)
	foreach ($r in $script:rows) {
		$bv = New-Object System.Drawing.SolidBrush($r[2])
		if ($r[0]) {
			$g.DrawString($r[0], $fontRow, $bn, (New-Object System.Drawing.RectangleF([single]$pad, [single]$y, [single]$L.nameW, [single]$L.rowH)), $fmtEll)
			$g.DrawString($r[1], $fontRow, $bv, (New-Object System.Drawing.RectangleF([single]($pad + $L.nameW), [single]$y, [single]($w - $L.nameW), [single]$L.rowH)), $fmtEll)
		} else {
			$g.DrawString($r[1], $fontRow, $bv, (New-Object System.Drawing.RectangleF([single]$pad, [single]$y, [single]$w, [single]$L.rowH)), $fmtEll)
		}
		$bv.Dispose(); $y += $L.rowH
	}
	if ($script:fpsText) {
		$bf = New-Object System.Drawing.SolidBrush($script:fpsColor)
		$g.DrawString($script:fpsText, $fontFps, $bf, (New-Object System.Drawing.RectangleF([single]$pad, [single]$y, [single]$w, [single]$L.fpsH)), $fmtEll)
		$bf.Dispose()
	}
	$b.Dispose(); $bn.Dispose()
})

$script:ft = New-Object System.Collections.Generic.List[double]
$script:ftMax = New-Object System.Collections.Generic.List[double]
$graph = New-Object System.Windows.Forms.PictureBox
$graph.Dock = 'Bottom'; $graph.Height = (Px 90); $graph.BackColor = [System.Drawing.Color]::Black
$graph.Add_Paint({ param($s, $e)
	$g = $e.Graphics; $w = $graph.Width; $h = $graph.Height
	$top = 40.0
	$grid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(60, 60, 60))
	foreach ($ms in @(8.3, 16.7, 33.3)) { $y = $h - ($ms / $top) * $h; $g.DrawLine($grid, [single]0, [single]$y, [single]$w, [single]$y) }
	$g.DrawString("frametime (ms)  40", $fontSmall, [System.Drawing.Brushes]::Gray, [single]0, [single]0)
	$g.DrawString("16.7", $fontSmall, [System.Drawing.Brushes]::Gray, [single]0, [single]($h - (16.7 / $top) * $h - $fontSmall.Height))
	foreach ($pair in @(,@($script:ft, [System.Drawing.Color]::Lime))) {
		$list = $pair[0]; if ($list.Count -lt 2) { continue }
		$pen = New-Object System.Drawing.Pen($pair[1], [single](1.5 * $S))
		$step = $w / 59.0; $x0 = $w - ($list.Count - 1) * $step
		for ($i = 1; $i -lt $list.Count; $i++) {
			$y1 = $h - [Math]::Min($list[$i-1], $top) / $top * $h; $y2 = $h - [Math]::Min($list[$i], $top) / $top * $h
			$g.DrawLine($pen, [single]($x0 + ($i-1) * $step), [single]$y1, [single]($x0 + $i * $step), [single]$y2)
		}
	}
})
$form.Controls.Add($label)
$form.Controls.Add($graph)
$graph.Add_DoubleClick({ $form.Close() })

# Relojes (MHz) de GPU y CPU: sin instalar nada y sin administrador, como mucho una vez por segundo y guardados.
# GPU: nvidia-smi (clocks.gr, en la misma consulta de siempre). Sin NVIDIA: el uso con los contadores de Windows
# "GPU Engine" (no dan reloj); AMD sale "n/a", otras graficas sin linea de reloj.
# CPU: "% Processor Performance" x MaxClockSpeed / 100 = MHz efectivos (turbo incluido).
$script:gpuVendor = ''
try { $vc = Get-CimInstance Win32_VideoController -ErrorAction Stop | Select-Object -First 1; $script:gpuVendor = "$($vc.Name) $($vc.AdapterCompatibility)" } catch { }
$script:isAmd = $script:gpuVendor -match 'AMD|Radeon|Advanced Micro'
$script:cpuMaxMHz = 0
try { $script:cpuMaxMHz = [int](Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1).MaxClockSpeed } catch { }
$script:cpuPerfCtr = $null
try { $script:cpuPerfCtr = New-Object System.Diagnostics.PerformanceCounter('Processor Information', '% Processor Performance', '_Total', $true); [void]$script:cpuPerfCtr.NextValue() } catch { $script:cpuPerfCtr = $null }
$script:clk = @{ stamp = [datetime]::MinValue; cpuMHz = $null; gpuUse = $null }
function Update-Clocks {
	if (((Get-Date) - $script:clk.stamp).TotalMilliseconds -lt 950) { return }
	$script:clk.stamp = Get-Date
	if ($script:cpuMaxMHz -gt 0) {
		$perf = $null
		if ($script:cpuPerfCtr) { try { $perf = [double]$script:cpuPerfCtr.NextValue() } catch { $script:cpuPerfCtr = $null } }
		if (-not $perf) { try { $perf = [double](Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'" -ErrorAction Stop).PercentProcessorPerformance } catch { } }
		$script:clk.cpuMHz = if ($perf) { [int][Math]::Round($perf * $script:cpuMaxMHz / 100.0) } else { $null }
	}
	if (-not $smi) {
		try {
			$eng = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine -ErrorAction Stop | Where-Object { $_.Name -like '*engtype_3D' }
			$script:clk.gpuUse = [int][Math]::Min(100, ($eng | Measure-Object -Property UtilizationPercentage -Sum).Sum)
		} catch { $script:clk.gpuUse = $null }
	}
}

# Lossless Scaling: FPS medidos con PresentMon (si PresentMon.exe esta junto a este script).
$script:lsFps = $null; $script:lsTimes = New-Object System.Collections.Generic.Queue[double]; $script:lsCol = -1
# PresentMon (FPS de Lossless Scaling) solo con -LossLess: su sesion de seguimiento cortaba la grabacion de NVIDIA.
$pm = if ($LossLess) { Get-ChildItem -Path $PSScriptRoot -Filter "PresentMon*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 } else { $null }
if ($pm) {
	$psi = New-Object System.Diagnostics.ProcessStartInfo
	$psi.FileName = $pm.FullName
	$psi.Arguments = "--process_name LosslessScaling.exe --output_stdout --stop_existing_session --session_name MetricasLS"
	$psi.UseShellExecute = $false; $psi.RedirectStandardOutput = $true; $psi.CreateNoWindow = $true
	$script:pmProc = New-Object System.Diagnostics.Process; $script:pmProc.StartInfo = $psi
	$script:lsLines = [System.Collections.Concurrent.ConcurrentQueue[string]]::new()
	try {
		$script:pmProc.Start() | Out-Null
		$rs = [runspacefactory]::CreateRunspace(); $rs.Open()
		$rs.SessionStateProxy.SetVariable('reader', $script:pmProc.StandardOutput)
		$rs.SessionStateProxy.SetVariable('q', $script:lsLines)
		$ps = [powershell]::Create(); $ps.Runspace = $rs
		$ps.AddScript({ while ($true) { $l = $reader.ReadLine(); if ($l -eq $null) { break }; $q.Enqueue($l) } }) | Out-Null
		$ps.BeginInvoke() | Out-Null
	} catch { $script:pmProc = $null }
	$form.Add_FormClosed({ try { if ($script:pmProc -and -not $script:pmProc.HasExited) { $script:pmProc.Kill() } } catch { } })
}
function Update-Ls {
	if (-not $script:lsLines) { return }
	$line = $null; $sum = 0.0; $n = 0
	while ($script:lsLines.TryDequeue([ref]$line)) {
		$c = $line -split ','
		if ($script:lsCol -lt 0) {
			for ($i = 0; $i -lt $c.Count; $i++) { if ($c[$i] -eq 'MsBetweenPresents' -or $c[$i] -eq 'MsBetweenDisplayChange' -or $c[$i] -eq 'FrameTime') { $script:lsCol = $i; break } }
			continue
		}
		if ($c.Count -gt $script:lsCol) { $v = 0.0; if ([double]::TryParse($c[$script:lsCol], [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$v) -and $v -gt 0) { $sum += $v; $n++ } }
	}
	if ($n -gt 0) { $script:lsFps = [Math]::Round(1000.0 * $n / $sum); $script:lsStamp = Get-Date }
	elseif ($script:lsStamp -and ((Get-Date) - $script:lsStamp).TotalSeconds -gt 3) { $script:lsFps = $null }
}

# Arrastrar con el raton; doble clic cierra.
$script:drag = $null
$label.Add_MouseDown({ $script:drag = [System.Windows.Forms.Cursor]::Position; $script:start = $form.Location })
$label.Add_MouseUp({ $script:drag = $null })
$label.Add_MouseMove({
	if ($script:drag) {
		$p = [System.Windows.Forms.Cursor]::Position
		$form.Location = New-Object System.Drawing.Point(($script:start.X + $p.X - $script:drag.X), ($script:start.Y + $p.Y - $script:drag.Y))
	}
})
$label.Add_DoubleClick({ $form.Close() })

function Get-Fps {
	if (-not $LogDir -or -not (Test-Path $LogDir)) { return $null }
	$log = Get-ChildItem -Path $LogDir -Filter "*.out.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
	if (-not $log -or ((Get-Date) - $log.LastWriteTime).TotalSeconds -gt 10) { return $null }
	$inv = [System.Globalization.CultureInfo]::InvariantCulture
	try {
		$fs = [System.IO.File]::Open($log.FullName, 'Open', 'Read', 'ReadWrite')
		$len = $fs.Length; $take = [Math]::Min($len, 65536)
		$fs.Seek($len - $take, 'Begin') | Out-Null
		$buf = New-Object byte[] $take
		$fs.Read($buf, 0, $take) | Out-Null
		$fs.Close()
		$text = [System.Text.Encoding]::UTF8.GetString($buf)
		$r = @{}
		$m = [regex]::Matches($text, '\[fps\] \d+ frames: mean ([0-9.]+) ms, max ([0-9.]+)')
		if ($m.Count -gt 0) {
			$last = $m[$m.Count - 1]
			$r.mean = [double]::Parse($last.Groups[1].Value, $inv); $r.max = [double]::Parse($last.Groups[2].Value, $inv)
			if ($r.mean -gt 0) { $r.fps = [Math]::Round(1000.0 / $r.mean) }
			$r.key = ($len - $take) + $last.Index  # (10-23) posicion absoluta en el log: la relativa se repetia y la grafica no avanzaba
		} else {
			$m = [regex]::Matches($text, 'Frames \(5 s\): ([0-9.]+) fps')
			if ($m.Count -gt 0) { $r.fps = [Math]::Round([double]::Parse($m[$m.Count - 1].Groups[1].Value, $inv)) }
		}
		$m = [regex]::Matches($text, '\[[a-z0-9]+\][^\r\n]*?real ([0-9.]+) fps, shown ([0-9.]+) fps')
		$x = [regex]::Matches($text, '\[(dlssg|fsr3-mfg)\][^\r\n]*?x(\d)')
		if ($x.Count -gt 0) { $r.fgMult = [int]$x[$x.Count - 1].Groups[2].Value }
		if ($m.Count -gt 0) { $r.fgShown = [Math]::Round([double]::Parse($m[$m.Count - 1].Groups[2].Value, $inv)); $r.fgReal = [Math]::Round([double]::Parse($m[$m.Count - 1].Groups[1].Value, $inv)) }
		return $r
	} catch { }
	return $null
}
$script:lastKey = -1; $script:lastLog = ''

# launcher.ini una vez por segundo, como tabla (antes se leia cuatro veces por tick).
function Read-Ini {
	$h = @{}
	try {
		foreach ($l in (Get-Content (Join-Path (Split-Path $LogDir -Parent) 'launcher.ini') -ErrorAction Stop)) {
			$i = $l.IndexOf('='); if ($i -gt 0) { $h[$l.Substring(0, $i).Trim()] = $l.Substring($i + 1).Trim() }
		}
	} catch { }
	return $h
}

# Reescalado en uso. Del log del emulador (leido por partes, solo lo nuevo):
#  "<kind> ... WxH -> WxH" (DLSS, XeSS, FSR 3.1; se ignoran las imagenes pequenas de los menus),
#  "render size: KYTY_UPSCALE_RATIO=3 on a 2160 high display: 720", "DLSS super resolution ...: ready",
#  "XeSS: libxess.dll 2.0.2 loaded". Si el log aun no lo dice: el modo del lanzador (scalelabel, dlsspreset).
# (10-22) Antes el modelo no salia: el texto solo se formaba con la linea "WxH -> WxH", el "Nativo" iba dentro
# del bucle (llave mal puesta) y la fila de 28 px cortaba el texto largo en la 2a linea, que quedaba tapada.
$script:sc = @{ log = ''; pos = 0 }
function Reset-Scale { $script:sc = @{ log = ''; pos = 0; kind = ''; inH = 0; outH = 0; ratio = ''; xessVer = ''; dlssReady = $false; force4k = $false; native = 0; logPreset = '' } }
Reset-Scale
function Get-ScaleModeName([string]$label) {
	$mode = ($label -split ' ', 2)[1]
	if (-not $mode) { return '' }
	return ($mode -replace 'U\.cal\.\+', 'Ultra calidad+' -replace 'U\.calidad', 'Ultra calidad' -replace 'Equilib\.', 'Equilibrado' -replace 'U\.rendim\.', 'Ultra rendimiento' -replace 'Rendim\.', 'Rendimiento')
}
function Update-ScaleLog {
	if (-not $LogDir -or -not (Test-Path $LogDir)) { return }
	$log = Get-ChildItem -Path $LogDir -Filter "*.out.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
	if (-not $log) { return }
	if ($log.FullName -ne $script:sc.log) { Reset-Scale; $script:sc.log = $log.FullName }
	try {
		$fs = [System.IO.File]::Open($log.FullName, 'Open', 'Read', 'ReadWrite')
		if ($fs.Length -lt $script:sc.pos) { $script:sc.pos = 0 }
		$fs.Seek($script:sc.pos, 'Begin') | Out-Null
		$take = [int][Math]::Min($fs.Length - $script:sc.pos, 8MB)
		$buf = New-Object byte[] $take
		$n = $fs.Read($buf, 0, $take); $script:sc.pos += $n
		$fs.Close()
		if ($n -le 0) { return }
		$text = [System.Text.Encoding]::UTF8.GetString($buf, 0, $n)
		$m = [regex]::Matches($text, 'KYTY_UPSCALE_RATIO=([0-9.]+) on a (\d{3,4}) high display: (\d{3,4})')
		if ($m.Count -gt 0) { $l = $m[$m.Count - 1]; $script:sc.ratio = $l.Groups[1].Value; $script:sc.outH = [int]$l.Groups[2].Value; $script:sc.inH = [int]$l.Groups[3].Value }
		else {
			# (10-22) Modo 4K: la salida es 2160 (KYTY_DISPLAY_HEIGHT), no la altura del monitor.
			$d = [regex]::Matches($text, 'on a (\d{3,4}) high display')
			if ($d.Count -gt 0) { $script:sc.outH = [int]$d[$d.Count - 1].Groups[1].Value }
		}
		if ($text -match 'starts in its cinematic mode \(KYTY_DS_FORCE_MODE') { $script:sc.force4k = $true; if (-not $script:sc.outH) { $script:sc.outH = 2160 } }
		if ($text -match 'DLSS super resolution[^\r\n]*ready') { $script:sc.dlssReady = $true; $script:sc.kind = 'DLSS' }
		$x = [regex]::Match($text, 'XeSS: libxess\.dll ([0-9.]+) loaded')
		if ($x.Success) { $script:sc.xessVer = $x.Groups[1].Value; if (-not $script:sc.kind) { $script:sc.kind = 'XeSS' } }
		$p = [regex]::Matches($text, 'DLSS[^\r\n]*?(?:preset|model[o]?)[ =:]+([A-Z])\b')
		if ($p.Count -gt 0) { $script:sc.logPreset = $p[$p.Count - 1].Groups[1].Value }
		$m = [regex]::Matches($text, '(DLSS|XeSS|FSR 3\.1)[^\r\n]*?(\d{3,4})x(\d{3,4}) -> (\d{3,4})x(\d{3,4})')
		for ($i = $m.Count - 1; $i -ge 0; $i--) {
			$h = [int]$m[$i].Groups[3].Value
			if ($h -ge 600) { $script:sc.kind = $m[$i].Groups[1].Value -replace ' 3\.1', ''; $script:sc.inH = $h; if (-not $script:sc.outH) { $script:sc.outH = [int]$m[$i].Groups[5].Value }; break }
		}
		# Sin reescalador: la resolucion a la que dibuja el juego (con TURBO, 1080 aunque pongas 1440).
		$r = [regex]::Matches($text, 'performance mode renders (\d{3,4})x(\d{3,4})')
		if ($r.Count -gt 0) { $script:sc.native = [int]$r[$r.Count - 1].Groups[2].Value }
	} catch { }
}
# Devuelve @{ kind; mode; res; model } para las filas Reescalado / Resolucion / Modelo.
function Get-Scale($ini) {
	Update-ScaleLog
	$s = $script:sc
	$lbl = "$($ini['scalelabel'])"
	$lk = ($lbl -split ' ', 2)[0]
	$kind = $s.kind
	if (-not $kind -and $lk -in 'DLSS', 'XeSS', 'FSR') { $kind = $lk }
	if (-not $script:screenH) { try { $script:screenH = (Get-CimInstance Win32_VideoController | Where-Object { $_.CurrentVerticalResolution } | Select-Object -First 1).CurrentVerticalResolution } catch { } ; if (-not $script:screenH) { $script:screenH = $scr.Height } }
	$outH = if ($s.outH) { $s.outH } else { $script:screenH }
	$r = @{ kind = ''; mode = ''; res = ''; model = '' }
	if ($kind) {
		$r.kind = $kind
		if ($lk -eq $kind) { $r.mode = Get-ScaleModeName $lbl }
		$inH = $s.inH
		if (-not $inH -and $lk -eq $kind -and $s.ratio) { $inH = [int][Math]::Round($outH / [double]::Parse($s.ratio, [System.Globalization.CultureInfo]::InvariantCulture)) }
		$ratio = if ($s.ratio) { " (x$($s.ratio))" } elseif ($inH) { " (x{0:0.##})" -f ($outH / [double]$inH) } else { '' }
		$r.res = if ($inH) { "{0}p $ARROW {1}p{2}" -f $inH, $outH, $ratio } else { "$ARROW ${outH}p" }
		switch ($kind) {
			'DLSS' {
				$pk = $s.logPreset
				if (-not $pk) { $pi2 = 0; [void][int]::TryParse("$($ini['dlsspreset'])", [ref]$pi2); $pk = @('', 'K', 'L', 'M')[[Math]::Min([Math]::Max(0, $pi2), 3)] }
				$r.model = switch ($pk) { 'K' { 'K (Transformer, DLSS 4)' } 'L' { 'L (Transformer, DLSS 4.5)' } 'M' { 'M (Transformer, DLSS 4.5)' } '' { 'auto (NVIDIA)' } default { "$pk" } }
				if (-not $s.dlssReady -and -not $s.inH) { $r.model += ' ...' }
			}
			'XeSS' { $r.model = if ($s.xessVer -eq '2.0.2') { 'XeSS 3.0.2' } elseif ($s.xessVer) { "XeSS $($s.xessVer)" } else { 'XeSS' } }  # el SDK 3.0.2 trae libxess 2.0.2
			'FSR'  { $r.model = 'FSR 3.1' }
		}
	} elseif ($s.force4k) { $r.kind = 'Nativo'; $r.res = '2160p (4K)' }
	elseif ($lbl -like '*juego*') { $h = ($lbl -split 'p', 2)[0]; $r.kind = 'Juego'; $r.res = "{0}p $ARROW {1}p" -f $h, $outH; $r.model = 'reescalado del juego' }
	elseif ($s.native) { $r.kind = 'Nativo'; $r.res = "{0}p $ARROW {1}p" -f $s.native, $outH }
	elseif ($lbl) { $r.kind = $lbl }
	return $r
}

$cGreen = [System.Drawing.Color]::LimeGreen; $cRed = [System.Drawing.Color]::Red; $cBlue = [System.Drawing.Color]::DeepSkyBlue
$cOrange = [System.Drawing.Color]::Orange; $cWhite = [System.Drawing.Color]::White; $cKhaki = [System.Drawing.Color]::Khaki
# (10-23) Color segun lo lleno que este (RAM y VRAM): verde < 70 %, amarillo < 85 %, naranja < 95 %, rojo al limite.
function Get-UseColor([double]$used, [double]$total) {
	if ($total -le 0) { return $cOrange }
	$f = $used / $total
	if ($f -lt 0.70) { return $cGreen } elseif ($f -lt 0.85) { return [System.Drawing.Color]::Gold } elseif ($f -lt 0.95) { return $cOrange } else { return $cRed }
}
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 1000
$timer.Add_Tick({
	# Se cierra sola cuando se cierra el juego (despues de haberlo visto abierto).
	$emu = Get-Process -Name 'kyty_emulator*' -ErrorAction SilentlyContinue
	$emuWin = $emu | Where-Object { $_.MainWindowHandle -ne 0 }
	# (Despues de que salga la ventana del juego: antes reserva su memoria de golpe, y con el limite ya puesto fallaba.)
	if ($RamGB -gt 0 -and $emuWin) {
		$p = $emuWin | Select-Object -First 1
		if ($p.Id -ne $script:limitedPid) {
			$max = [uint64](($RamGB - 4) * 1GB)
			$ok = [RamLimit]::SetProcessWorkingSetSizeEx($p.Handle, [UIntPtr][uint64]200MB, [UIntPtr]$max, 6)
			$script:limitedPid = $p.Id; $script:limitOk = $ok
		}
	}
	if ($emuWin) { $script:emuSeen = $true; $script:emuGone = 0 }
	elseif ($script:emuSeen) { $script:emuGone++; if ($script:emuGone -ge 2) { $form.Close(); return } }
	Update-Clocks
	$ini = Read-Ini
	$rows = New-Object System.Collections.Generic.List[object]
	if ($smi) {
		$q = & $smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.gr --format=csv,noheader,nounits 2>$null | Select-Object -First 1
		if ($q) {
			$v = $q -split ',\s*'
			$gpu = [int]$v[0]; $used = [int]$v[1]; $total = [int]$v[2]; $temp = [int]$v[3]
			$gclk = 0; if ($v.Count -gt 4) { [void][int]::TryParse($v[4], [ref]$gclk) }
			if ($VramMB -gt 0 -and $VramMB -lt $total) {
				$reserved = $total - $VramMB
				$shown = [Math]::Max(0, $used - $reserved)
				$max = $VramMB
			} elseif ($MaxMB -gt 0) { $shown = $used; $max = $MaxMB } else { $shown = $used; $max = $total }
			$rows.Add(@('GPU', ("{0,3} %   {1} C" -f $gpu, $temp), $cOrange))
			if ($gclk -gt 0) { $rows.Add(@('GPU reloj', ("{0,4} MHz" -f $gclk), $cOrange)) }
			$rows.Add(@('VRAM', ("{0:N1} / {1:N0} GB" -f ($shown / 1024.0), ($max / 1024.0)), (Get-UseColor $shown $max)))
		}
	} else {
		$rows.Add(@('GPU', $(if ($script:clk.gpuUse -ne $null) { "{0,3} %" -f $script:clk.gpuUse } else { '(sin nvidia-smi)' }), $cOrange))
		if ($script:isAmd) { $rows.Add(@('GPU reloj', 'n/a', $cOrange)) }
	}
	try {
		$cpu = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
		$os = Get-CimInstance Win32_OperatingSystem
		$ramUsed = ($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB
		$ramTotal = $os.TotalVisibleMemorySize / 1MB
		$rows.Add(@('CPU', ("{0,3} %" -f $cpu), $cOrange))
		if ($script:clk.cpuMHz) { $rows.Add(@('CPU reloj', ("{0,4} MHz" -f $script:clk.cpuMHz), $cOrange)) }
		if ($RamGB -gt 0) {
			$ws = 0; $e2 = $emu | Select-Object -First 1
			if ($e2) { $ws = $e2.WorkingSet64 / 1GB }
			$rows.Add(@('RAM', ("{0:N1} / {1} GB{2}" -f ($ws + 4), $RamGB, $(if ($script:limitedPid -and -not $script:limitOk) { ' (sin limite!)' } else { '' })), (Get-UseColor ($ws + 4) $RamGB)))
		} else {
			$rows.Add(@('RAM', ("{0:N1} / {1:N0} GB" -f $ramUsed, $ramTotal), (Get-UseColor $ramUsed $ramTotal)))
		}
		# (10-24) Commit (memoria comprometida / limite de Windows = RAM + archivo de paginacion), de la misma consulta
		# de Win32_OperatingSystem (nada mas que leer). Al limite Windows niega memoria y el juego se cierra: rojo por
		# encima del 90 %. Con -RamGB (simular un PC de N GB): el emulador + 4 GB de Windows contra N GB + la paginacion.
		$commitLimit = [double]$os.TotalVirtualMemorySize / 1MB
		$commitUsed = $commitLimit - [double]$os.FreeVirtualMemory / 1MB
		$commitSim = ''
		if ($RamGB -gt 0) {
			$pageFile = [Math]::Max(0.0, $commitLimit - $ramTotal)
			$commitLimit = $RamGB + $pageFile
			$priv = 0; $e3 = $emu | Select-Object -First 1
			if ($e3) { $priv = $e3.PrivateMemorySize64 / 1GB }
			$commitUsed = $priv + 4
			$commitSim = ' (simulado)'
		}
		if ($commitLimit -gt 0) {
			$cc = if ($commitUsed / $commitLimit -gt 0.90) { $cRed } else { Get-UseColor $commitUsed $commitLimit }
			$rows.Add(@('Commit', ("{0:N1} / {1:N0} GB{2}" -f $commitUsed, $commitLimit, $commitSim), $cc))
		}
	} catch { }
	$shownFps = $null
	$fps = Get-Fps
	if ($fps) {
		$shownFps = if ($fps.fgShown) { $fps.fgShown } elseif ($fps.fps) { $fps.fps } else { $null }
		if ($fps.mean) {
			$rows.Add(@('Frame', ("{0:N1} ms" -f $fps.mean), $cOrange))
			if ($fps.key -ne $script:lastKey) {
				$script:lastKey = $fps.key
				$script:ft.Add($fps.mean); $script:ftMax.Add($fps.max)
				while ($script:ft.Count -gt 60) { $script:ft.RemoveAt(0); $script:ftMax.RemoveAt(0) }
			}
		}
	}
	Update-Ls
	if ($script:lsFps) { $rows.Add(@('FPS LS', "$($script:lsFps)", $cOrange)) }
	# Reescalador en su color: DLSS (NVIDIA) verde, FSR (AMD) rojo, XeSS (Intel) azul.
	$sc = Get-Scale $ini
	if ($sc.kind) {
		$col = switch ($sc.kind) { 'DLSS' { $cGreen } 'FSR' { $cRed } 'XeSS' { $cBlue } default { $cOrange } }
		$rows.Add(@('Escala', ($(if ($sc.mode) { $sc.kind + $DOT + $sc.mode } else { $sc.kind })), $col))
		if ($sc.res) { $rows.Add(@('Resol.', $sc.res, $col)) }
		if ($sc.model) { $rows.Add(@('Modelo', $sc.model, $col)) }
	}
	# Frame generation: el modo del lanzador y el multiplicador que lleva ahora (DLSS verde, AMD rojo).
	$fgName = "$($ini['fglabel'])"
	$fgNv = $fgName -match 'DLSS|Nvidia'
	if ($fgName -and $fgName -ne 'Sin frame generation' -and $fgName -ne 'No frame generation') {
		$fgCol = if ($fgNv) { $cGreen } else { $cRed }
		if ($fgName -like 'MODO *' -or $fgName -like 'MODE *') { $rows.Add(@('', $fgName, $fgCol)) }
		else {
			$rows.Add(@('FG', ($fgName + $(if ($fgNv) { ' (modelo C)' } else { '' })), $fgCol))
			if ($fps -and $fps.fgMult) { $rows.Add(@('FG ahora', "x$($fps.fgMult)", $fgCol)) }
		}
	}
	# (10-23) Solo el modo grafico elegido en el lanzador (Modo Souls o Personalizado).
	# (10-23) Anti-Lag que se pidio al arrancar (Reflex / Anti-Lag 2 / No).
	$al = "$($ini['antilagon'])"
	if ($al) { $rows.Add(@('Anti-Lag', $al, $(if ($al -eq 'No') { $cKhaki } else { $cGreen }))) }
	$modo = "$($ini['soulsmodo'])"
	if ($modo) { $rows.Add(@('', $modo, $cKhaki)) }
	$script:rows = $rows.ToArray()
	$script:fpsText = $(if ($fps -and $shownFps) { "FPS $shownFps" } else { '' })
	# FPS en el color del frame generation: verde NVIDIA, rojo AMD, blanco sin frame generation.
	$script:fpsColor = if ($fgName -and $fgName -ne 'Sin frame generation' -and $fgName -ne 'No frame generation') { if ($fgNv) { $cGreen } else { $cRed } } else { $cWhite }
	Measure-Layout
	$label.Invalidate(); $graph.Invalidate()
})
$form.Add_Shown({ Measure-Layout; $label.Invalidate() })
$timer.Start()
[System.Windows.Forms.Application]::Run($form)
