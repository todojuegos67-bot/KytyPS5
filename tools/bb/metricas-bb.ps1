# Metricas de Bloodborne (shadPS4): ventana pequena siempre encima, el mismo aspecto que la de Demon's Souls.
# Solo LEE: no toca config, parches, partidas ni el proceso del juego (sin limites de RAM).
# Uso: powershell -ExecutionPolicy Bypass -File metricas-bb.ps1 [-ProcessId N] [-UserDir "...\user"] [-LauncherIni "...\launcher.ini"]
#        [-FpsFile "...\logs\bb-fps.log"] [-PresentMon]
# FPS / frametime, por orden:
#   1) shadPS4: no escribe los FPS en su log ni exporta los datos de --show-fps (su overlay es ImGui dentro del juego).
#      No hay nada que leer.
#   2) El emulador propio: archivo -FpsFile (BB_METRICS_OUT) con las mismas lineas que Kyty en Demon's Souls:
#      "[fps] N frames: mean X ms, max Y" y "[lat] input->present mean ...".
#   3) -PresentMon: PresentMon*.exe junto a este script (portable, sin instalar). Usa ETW: sin administrador o sin el
#      grupo "Usuarios del registro de rendimiento" falla. Es opcional porque en Demon's Souls su sesion cortaba la
#      grabacion de NVIDIA.
#   Si nada funciona: "FPS —" (no se inventa).
param(
	[int]$ProcessId = 0,
	[string]$ProcessName = 'shadPS4',
	[string]$UserDir = '',
	[string]$LauncherIni = '',
	[string]$FpsFile = '',
	[switch]$PresentMon,
	[string]$Titulo = 'BLOODBORNE'
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

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

# (10-27) Grabacion de NVIDIA (ShadowPlay / NVIDIA App, Alt+F9): la ventana de metricas nunca toma el foco.
# Antes Application.Run la activaba (el juego perdia el primer plano) y NVIDIA cortaba la grabacion al empezar.
# Ahora: se muestra sin activarse (ShowWithoutActivation + WS_EX_NOACTIVATE), es de herramienta (fuera de la barra
# de tareas y de Alt-Tab), transparente al raton, topmost una sola vez con SWP_NOACTIVATE, y excluida de la captura
# (WDA_EXCLUDEFROMCAPTURE): no sale en el video y no molesta a la captura del juego.
Add-Type -ReferencedAssemblies System.Windows.Forms, System.Drawing @"
using System; using System.Runtime.InteropServices; using System.Windows.Forms;
public class NoActForm : Form {
  public const int WS_EX_TOPMOST = 0x8, WS_EX_TRANSPARENT = 0x20, WS_EX_TOOLWINDOW = 0x80, WS_EX_LAYERED = 0x80000, WS_EX_NOACTIVATE = 0x8000000;
  const int WM_MOUSEACTIVATE = 0x21, MA_NOACTIVATE = 3;
  protected override bool ShowWithoutActivation { get { return true; } }
  protected override CreateParams CreateParams { get {
    CreateParams cp = base.CreateParams;
    cp.ExStyle |= WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOPMOST;
    return cp; } }
  protected override void WndProc(ref Message m) {
    if (m.Msg == WM_MOUSEACTIVATE) { m.Result = (IntPtr)MA_NOACTIVATE; return; }
    base.WndProc(ref m);
  }
}
public static class OvWin {
  [DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr h, int i);
  [DllImport("user32.dll")] public static extern int SetWindowLong(IntPtr h, int i, int v);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint f);
  [DllImport("user32.dll")] public static extern bool SetWindowDisplayAffinity(IntPtr h, uint a);
  public static void Apply(IntPtr h) {
    int ex = GetWindowLong(h, -20);
    SetWindowLong(h, -20, ex | NoActForm.WS_EX_TOOLWINDOW | NoActForm.WS_EX_LAYERED | NoActForm.WS_EX_TRANSPARENT | NoActForm.WS_EX_NOACTIVATE);
    // HWND_TOPMOST, SWP_NOSIZE|SWP_NOMOVE|SWP_NOACTIVATE|SWP_NOOWNERZORDER|SWP_FRAMECHANGED
    SetWindowPos(h, new IntPtr(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010 | 0x0200 | 0x0020);
    // WDA_EXCLUDEFROMCAPTURE (Windows 10 2004+); si falla, WDA_MONITOR no: se deja como esta.
  }
}
"@
$form = New-Object NoActForm
$form.Text = "Metricas Bloodborne"
$form.FormBorderStyle = 'None'
$form.ShowInTaskbar = $false
$keyCol = [System.Drawing.Color]::FromArgb(255, 1, 0, 1)
$form.BackColor = $keyCol; $form.TransparencyKey = $keyCol
$form.Opacity = 1.0
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
$label.Dock = 'Fill'; $label.BackColor = $keyCol
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
	$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::SingleBitPerPixelGridFit
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
$graph.Dock = 'Bottom'; $graph.Height = (Px 90); $graph.BackColor = $keyCol
$graph.Add_Paint({ param($s, $e)
	$g = $e.Graphics; $w = $graph.Width; $h = $graph.Height
	$top = 40.0
	$grid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(60, 60, 60))
	foreach ($ms in @(8.3, 16.7, 33.3)) { $y = $h - ($ms / $top) * $h; $g.DrawLine($grid, [single]0, [single]$y, [single]$w, [single]$y) }
	$g.DrawString("frametime (ms)  40   verde media, naranja picos", $fontSmall, [System.Drawing.Brushes]::Gray, [single]0, [single]0)
	$g.DrawString("16.7", $fontSmall, [System.Drawing.Brushes]::Gray, [single]0, [single]($h - (16.7 / $top) * $h - $fontSmall.Height))
	# (10-24) Dos lineas: el peor frame de cada segundo (picos, naranja) y la media (verde). Antes solo la media: con
	# 30 FPS fijos quedaba encima de la raya de 33.3 y no se veia.
	$sc = if ($S) { [double]$S } else { 1.0 }
	foreach ($pair in @(@($script:ftMax, [System.Drawing.Color]::FromArgb(255, 150, 40)), @($script:ft, [System.Drawing.Color]::Lime))) {
		$list = $pair[0]; if ($list.Count -lt 2) { continue }
		$pen = New-Object System.Drawing.Pen($pair[1], [single](2.0 * $sc))
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

# PresentMon (opcional, -PresentMon): FPS y frametime del proceso de shadPS4 por ETW.
$script:pmLines = $null; $script:pmCol = -1; $script:pmFps = $null; $script:pmMean = $null; $script:pmMax = $null; $script:pmStamp = $null; $script:pmErr = ''
$pmExe = if ($PresentMon) { Get-ChildItem -Path $PSScriptRoot -Filter 'PresentMon*.exe' -ErrorAction SilentlyContinue | Select-Object -First 1 } else { $null }
if ($pmExe) {
	$psi = New-Object System.Diagnostics.ProcessStartInfo
	$psi.FileName = $pmExe.FullName
	$target = if ($ProcessId -gt 0) { "--process_id $ProcessId" } else { "--process_name $ProcessName.exe" }
	$psi.Arguments = "$target --output_stdout --stop_existing_session --session_name MetricasBB"
	$psi.UseShellExecute = $false; $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.CreateNoWindow = $true
	$script:pmProc = New-Object System.Diagnostics.Process; $script:pmProc.StartInfo = $psi
	$script:pmLines = [System.Collections.Concurrent.ConcurrentQueue[string]]::new()
	try {
		$script:pmProc.Start() | Out-Null
		$rs = [runspacefactory]::CreateRunspace(); $rs.Open()
		$rs.SessionStateProxy.SetVariable('reader', $script:pmProc.StandardOutput)
		$rs.SessionStateProxy.SetVariable('q', $script:pmLines)
		$ps = [powershell]::Create(); $ps.Runspace = $rs
		$ps.AddScript({ while ($true) { $l = $reader.ReadLine(); if ($l -eq $null) { break }; $q.Enqueue($l) } }) | Out-Null
		$ps.BeginInvoke() | Out-Null
	} catch { $script:pmProc = $null; $script:pmErr = 'PresentMon no arranco' }
	$form.Add_FormClosed({ try { if ($script:pmProc -and -not $script:pmProc.HasExited) { $script:pmProc.Kill() } } catch { } })
}
function Update-Pm {
	if (-not $script:pmLines) { return }
	if ($script:pmProc -and $script:pmProc.HasExited -and -not $script:pmFps) { $script:pmErr = 'PresentMon: sin permiso ETW (hace falta administrador)' }
	$line = $null; $vals = New-Object System.Collections.Generic.List[double]
	while ($script:pmLines.TryDequeue([ref]$line)) {
		$c = $line -split ','
		if ($script:pmCol -lt 0) { for ($i = 0; $i -lt $c.Count; $i++) { if ($c[$i] -eq 'MsBetweenPresents' -or $c[$i] -eq 'FrameTime') { $script:pmCol = $i; break } }; continue }
		if ($c.Count -gt $script:pmCol) { $v = 0.0; if ([double]::TryParse($c[$script:pmCol], [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$v) -and $v -gt 0) { $vals.Add($v) } }
	}
	if ($vals.Count -gt 0) {
		$sum = 0.0; $mx = 0.0; foreach ($v in $vals) { $sum += $v; if ($v -gt $mx) { $mx = $v } }
		$script:pmMean = $sum / $vals.Count; $script:pmMax = $mx; $script:pmFps = [Math]::Round(1000.0 / $script:pmMean); $script:pmStamp = Get-Date
	} elseif ($script:pmStamp -and ((Get-Date) - $script:pmStamp).TotalSeconds -gt 3) { $script:pmFps = $null; $script:pmMean = $null }
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

# Archivo de FPS del emulador propio (mismo formato que Kyty). Se lee solo el final.
function Get-FileFps {
	if (-not $FpsFile -or -not (Test-Path -LiteralPath $FpsFile)) { return $null }
	$fi = Get-Item -LiteralPath $FpsFile
	if ([Math]::Abs(([DateTime]::UtcNow - $fi.LastWriteTimeUtc).TotalSeconds) -gt 10) { return $null }
	$inv = [System.Globalization.CultureInfo]::InvariantCulture
	try {
		$fs = [System.IO.File]::Open($fi.FullName, 'Open', 'Read', 'ReadWrite')
		$len = $fs.Length; $take = [Math]::Min($len, 65536); $fs.Seek($len - $take, 'Begin') | Out-Null
		$buf = New-Object byte[] $take; $fs.Read($buf, 0, $take) | Out-Null; $fs.Close()
		$text = [System.Text.Encoding]::UTF8.GetString($buf)
		$r = @{}
		$m = [regex]::Matches($text, '\[fps\] \d+ frames: mean ([0-9.]+) ms, max ([0-9.]+)')
		if ($m.Count -gt 0) {
			$last = $m[$m.Count - 1]
			$r.mean = [double]::Parse($last.Groups[1].Value, $inv); $r.max = [double]::Parse($last.Groups[2].Value, $inv)
			if ($r.mean -gt 0) { $r.fps = [Math]::Round(1000.0 / $r.mean) }
			$r.key = ($len - $take) + $last.Index
		}
		$m = [regex]::Matches($text, '\[[a-z0-9-]+\][^\r\n]*?real ([0-9.]+) fps, shown ([0-9.]+) fps')
		if ($m.Count -gt 0 -and $m[$m.Count - 1].Groups[2].Value -ne $m[$m.Count - 1].Groups[1].Value) { $r.fgShown = [Math]::Round([double]::Parse($m[$m.Count - 1].Groups[2].Value, $inv)) }
		$m = [regex]::Matches($text, '\[lat\] input->present mean ([0-9.]+) ms, p95 ([0-9.]+)[^\r\n]*?(?:; total ([0-9.]+), p95 ([0-9.]+))?')
		if ($m.Count -gt 0) { $q = $m[$m.Count - 1].Groups; $r.latMean = [double]::Parse($(if ($q[3].Success) { $q[3].Value } else { $q[1].Value }), $inv); $r.latP95 = [double]::Parse($(if ($q[4].Success) { $q[4].Value } else { $q[2].Value }), $inv) }
		# (bb-upscale) "[up] <reescalador> <modo> <interna> -> <salida>" del emulador propio.
		$m = [regex]::Matches($text, '\[up\] ([^\r\n]+)')
		if ($m.Count -gt 0) { $r.up = $m[$m.Count - 1].Groups[1].Value.Trim() }
		if ($r.Count) { return $r }
	} catch { }
	return $null
}
$script:lastKey = -1; $script:lastKeyAt = $null

function Read-Ini {
	$h = @{}
	if (-not $LauncherIni) { return $h }
	try { foreach ($l in [IO.File]::ReadAllLines($LauncherIni)) { $i = $l.IndexOf('='); if ($i -gt 0) { $h[$l.Substring(0, $i).Trim()] = $l.Substring($i + 1).Trim() } } } catch { }
	return $h
}
# Config de shadPS4 (solo lectura, cada 5 s): FSR / RCAS y resoluciones. config.json (versiones nuevas) o config.toml.
$script:cfg = @{ stamp = [datetime]::MinValue }
function Get-ShadCfg {
	if (((Get-Date) - $script:cfg.stamp).TotalSeconds -lt 5) { return $script:cfg }
	$c = @{ stamp = (Get-Date) }
	try {
		$j = Join-Path $UserDir 'config.json'; $t = Join-Path $UserDir 'config.toml'
		if ($UserDir -and (Test-Path -LiteralPath $j)) {
			$x = [IO.File]::ReadAllText($j)
			foreach ($k in @(@('fsr', 'fsr_enabled'), @('rcas', 'rcas_enabled'), @('att', 'rcas_attenuation'), @('iw', 'internal_screen_width'), @('ih', 'internal_screen_height'), @('ww', 'window_width'), @('wh', 'window_height'))) {
				$m = [regex]::Match($x, '"' + $k[1] + '"\s*:\s*("?)([^",\r\n}]*)\1'); if ($m.Success) { $c[$k[0]] = $m.Groups[2].Value.Trim() }
			}
		} elseif ($UserDir -and (Test-Path -LiteralPath $t)) {
			$x = [IO.File]::ReadAllText($t)
			foreach ($k in @(@('fsr', 'fsrEnabled'), @('rcas', 'rcasEnabled'), @('att', 'rcasAttenuation'), @('iw', 'internalScreenWidth'), @('ih', 'internalScreenHeight'), @('ww', 'screenWidth'), @('wh', 'screenHeight'))) {
				$m = [regex]::Match($x, '(?m)^\s*' + $k[1] + '\s*=\s*"?([^"\r\n#]*)'); if ($m.Success) { $c[$k[0]] = $m.Groups[1].Value.Trim() }
			}
		}
	} catch { }
	$script:cfg = $c
	return $c
}
$script:modeNames = @{ '0' = 'Modo Ligero'; '1' = 'Modo Equilibrado'; '2' = 'Modo Ultra'; '3' = 'Modo Extremo'; '4' = 'Personalizado' }

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
	# Se cierra sola cuando se cierra shadPS4 (el PID que paso el lanzador, o por nombre).
	$emu = if ($ProcessId -gt 0) { Get-Process -Id $ProcessId -ErrorAction SilentlyContinue } else { Get-Process -Name "$ProcessName*" -ErrorAction SilentlyContinue }
	if ($emu) { $script:emuSeen = $true; $script:emuGone = 0 }
	elseif ($script:emuSeen -or $ProcessId -gt 0) { $script:emuGone++; if ($script:emuGone -ge 2) { $form.Close(); return } }
	Update-Clocks
	$ini = Read-Ini
	$rows = New-Object System.Collections.Generic.List[object]
	if ($smi) {
		$q = & $smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.gr --format=csv,noheader,nounits 2>$null | Select-Object -First 1
		if ($q) {
			$v = $q -split ',\s*'
			$gpu = [int]$v[0]; $used = [int]$v[1]; $total = [int]$v[2]; $temp = [int]$v[3]
			$gclk = 0; if ($v.Count -gt 4) { [void][int]::TryParse($v[4], [ref]$gclk) }
			$rows.Add(@('GPU', ("{0,3} %   {1} C" -f $gpu, $temp), $cOrange))
			if ($gclk -gt 0) { $rows.Add(@('GPU reloj', ("{0,4} MHz" -f $gclk), $cOrange)) }
			$rows.Add(@('VRAM', ("{0:N1} / {1:N0} GB" -f ($used / 1024.0), ($total / 1024.0)), (Get-UseColor $used $total)))
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
		$rows.Add(@('RAM', ("{0:N1} / {1:N0} GB" -f $ramUsed, $ramTotal), (Get-UseColor $ramUsed $ramTotal)))
		$commitLimit = [double]$os.TotalVirtualMemorySize / 1MB
		$commitUsed = $commitLimit - [double]$os.FreeVirtualMemory / 1MB
		if ($commitLimit -gt 0) {
			$cc = if ($commitUsed / $commitLimit -gt 0.90) { $cRed } else { Get-UseColor $commitUsed $commitLimit }
			$rows.Add(@('Commit', ("{0:N1} / {1:N0} GB" -f $commitUsed, $commitLimit), $cc))
		}
		$e2 = $emu | Select-Object -First 1
		if ($e2) { $rows.Add(@('shadPS4', ("{0:N1} GB" -f ($e2.WorkingSet64 / 1GB)), $cOrange)) }
	} catch { }
	# FPS: archivo del emulador propio, si no PresentMon, si no "—".
	$fps = Get-FileFps; $src = ''
	$mean = $null; $max = $null; $shownFps = $null
	if ($fps -and $fps.fps) { $src = 'emulador'; $mean = $fps.mean; $max = $fps.max; $shownFps = $(if ($fps.fgShown) { $fps.fgShown } else { $fps.fps })
		if ($fps.key -ne $script:lastKey) { $script:lastKey = $fps.key; $script:lastKeyAt = Get-Date; $script:ft.Add($mean); $script:ftMax.Add($max) }
		if ($script:lastKeyAt -and ((Get-Date) - $script:lastKeyAt).TotalSeconds -gt 3) { $shownFps = $null }
	} else {
		Update-Pm
		if ($script:pmFps) { $src = 'PresentMon'; $mean = $script:pmMean; $max = $script:pmMax; $shownFps = $script:pmFps; $script:ft.Add($mean); $script:ftMax.Add($max) }
	}
	while ($script:ft.Count -gt 60) { $script:ft.RemoveAt(0); $script:ftMax.RemoveAt(0) }
	if ($mean) { $rows.Add(@('Frame', ("{0:N1} ms" -f $mean), $cOrange)) }
	if ($fps -and $fps.latMean) {
		$lc = if ($fps.latMean -lt 40) { $cGreen } elseif ($fps.latMean -lt 60) { [System.Drawing.Color]::Gold } elseif ($fps.latMean -lt 80) { $cOrange } else { $cRed }
		$rows.Add(@('Latencia', ("{0:N0} ms (p95 {1:N0})" -f $fps.latMean, $fps.latP95), $lc))
	}
	# Equivalentes de shadPS4: FSR 1 + RCAS y resoluciones (leidos de su config, sin tocarla).
	$c = Get-ShadCfg
	if ($c.Count -gt 1) {
		$fsrOn = ($c.fsr -eq 'true'); $rcasOn = ($c.rcas -eq 'true')
		$esc = if ($fsrOn) { 'FSR 1' + $(if ($rcasOn) { ' + RCAS' + $(if ($c.att) { $a = 0.0; if ([double]::TryParse($c.att, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$a)) { $DOT + ('{0:0} %' -f [Math]::Max(0, 100 - $a / 10)) } }) } else { '' }) } else { 'Nativo (sin FSR)' }
		$rows.Add(@('Escala', $esc, $(if ($fsrOn) { $cRed } else { $cOrange })))
		if ($c.iw -and $c.ih) { $rows.Add(@('Resol.', ("{0}x{1}{2}" -f $c.iw, $c.ih, $(if ($c.ww -and $c.wh) { " $ARROW $($c.ww)x$($c.wh)" } else { '' })), $(if ($fsrOn) { $cRed } else { $cOrange }))) }
	}
	if ($fps -and $fps.up -and $fps.up -ne 'off') { $rows.Add(@('Reesc.', $fps.up, $cRed)) }
	if ("$($ini['safemode'])" -ne '0') { $rows.Add(@('', 'Modo seguro (config de shadPS4 tal cual)', $cKhaki)) }
	else { $mn = $script:modeNames["$($ini['modo'])"]; if ($mn) { $rows.Add(@('', $mn, $cKhaki)) } }
	if (-not $src) { $rows.Add(@('', $(if ($script:pmErr) { $script:pmErr } else { 'FPS: shadPS4 no los da fuera de su overlay' }), $colName)) }
	$script:rows = $rows.ToArray()
	$script:fpsText = $(if ($shownFps) { "FPS $shownFps" } else { 'FPS ' + [char]0x2014 })
	$script:fpsColor = $cWhite
	Measure-Layout
	$label.Invalidate(); $graph.Invalidate()
})
$form.Add_Shown({ try { [OvWin]::Apply($form.Handle) } catch { }; Measure-Layout; $label.Invalidate() })
$timer.Start()
# (10-29) F12: muestra / oculta las metricas (otra vez F12 las vuelve a poner).
try { Add-Type @"
using System; using System.Runtime.InteropServices;
public static class MetricsKey { [DllImport("user32.dll")] public static extern short GetAsyncKeyState(int vKey); }
"@ } catch { }
$script:f12Down = $false
$keyTimer = New-Object System.Windows.Forms.Timer
$keyTimer.Interval = 50
$keyTimer.Add_Tick({
	try {
		$down = ([MetricsKey]::GetAsyncKeyState(0x7B) -band 0x8000) -ne 0
		if ($down -and -not $script:f12Down) { $form.Visible = -not $form.Visible; if ($form.Visible) { $form.TopMost = $true } }
		$script:f12Down = $down
	} catch { }
})
$keyTimer.Start()
[System.Windows.Forms.Application]::Run($form)
