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

$form = New-Object System.Windows.Forms.Form
$form.Text = "Metricas"
$form.FormBorderStyle = 'None'
$form.TopMost = $true
$form.ShowInTaskbar = $false
$form.BackColor = [System.Drawing.Color]::Black
$form.Opacity = 0.8
$form.StartPosition = 'Manual'
$form.Location = New-Object System.Drawing.Point(10, 10)
$form.Size = New-Object System.Drawing.Size(560, 465)

$label = New-Object System.Windows.Forms.Label
$label.Dock = 'Top'; $label.Height = 200
$title = New-Object System.Windows.Forms.Label
$title.Dock = 'Top'; $title.Height = 30; $title.TextAlign = 'MiddleCenter'
$title.ForeColor = [System.Drawing.Color]::White
$title.Font = New-Object System.Drawing.Font("Consolas", 14, [System.Drawing.FontStyle]::Bold)
$title.Text = $Titulo
$label.ForeColor = [System.Drawing.Color]::Orange
$label.Font = New-Object System.Drawing.Font("Consolas", 14, [System.Drawing.FontStyle]::Bold)
$label.Text = "Cargando..."
# Reescalador en su color: DLSS (NVIDIA) verde, FSR (AMD) rojo, XeSS (Intel) azul.
$scaleLabel = New-Object System.Windows.Forms.Label
$scaleLabel.Dock = 'Top'; $scaleLabel.Height = 28; $scaleLabel.Text = ''
$scaleLabel.Font = New-Object System.Drawing.Font("Consolas", 14, [System.Drawing.FontStyle]::Bold)
# Opciones graficas del lanzador que no estan en Maximo (debajo del frame generation).
$gfxLabel = New-Object System.Windows.Forms.Label
$gfxLabel.Dock = 'Top'; $gfxLabel.Height = 24; $gfxLabel.Text = ''
$gfxLabel.ForeColor = [System.Drawing.Color]::Khaki
$gfxLabel.Font = New-Object System.Drawing.Font("Consolas", 11, [System.Drawing.FontStyle]::Bold)
$form.Controls.Add($gfxLabel)
# Frame generation: el modo del lanzador y el multiplicador que lleva ahora (DLSS verde, AMD rojo).
$fgLabel = New-Object System.Windows.Forms.Label
$fgLabel.Dock = 'Top'; $fgLabel.Height = 28; $fgLabel.Text = ''
$fgLabel.Font = New-Object System.Drawing.Font("Consolas", 12, [System.Drawing.FontStyle]::Bold)
$form.Controls.Add($fgLabel)
$fgLabel.Add_DoubleClick({ $form.Close() })
$form.Controls.Add($scaleLabel)
$scaleLabel.Add_DoubleClick({ $form.Close() })
$form.Controls.Add($label)
$form.Controls.Add($title)
$script:ft = New-Object System.Collections.Generic.List[double]
$script:ftMax = New-Object System.Collections.Generic.List[double]
$graph = New-Object System.Windows.Forms.PictureBox
$graph.Dock = 'Bottom'; $graph.Height = 100; $graph.BackColor = [System.Drawing.Color]::Black
$graph.Add_Paint({ param($s, $e)
	$g = $e.Graphics; $w = $graph.Width; $h = $graph.Height
	$top = 40.0
	$grid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(60, 60, 60))
	foreach ($ms in @(8.3, 16.7, 33.3)) { $y = $h - ($ms / $top) * $h; $g.DrawLine($grid, [single]0, [single]$y, [single]$w, [single]$y) }
	$f = New-Object System.Drawing.Font("Consolas", 7)
	$g.DrawString("frametime (ms)  40", $f, [System.Drawing.Brushes]::Gray, [single]0, [single]0)
	$g.DrawString("16.7", $f, [System.Drawing.Brushes]::Gray, [single]0, [single]($h - (16.7 / $top) * $h - 10))
	foreach ($pair in @(,@($script:ft, [System.Drawing.Color]::Lime))) {
		$list = $pair[0]; if ($list.Count -lt 2) { continue }
		$pen = New-Object System.Drawing.Pen($pair[1], [single]1.5)
		$step = $w / 59.0; $x0 = $w - ($list.Count - 1) * $step
		for ($i = 1; $i -lt $list.Count; $i++) {
			$y1 = $h - [Math]::Min($list[$i-1], $top) / $top * $h; $y2 = $h - [Math]::Min($list[$i], $top) / $top * $h
			$g.DrawLine($pen, [single]($x0 + ($i-1) * $step), [single]$y1, [single]($x0 + $i * $step), [single]$y2)
		}
	}
})
$form.Controls.Add($graph)
# FPS aparte: grandes y en rojo, encima de la grafica.
$fpsLabel = New-Object System.Windows.Forms.Label
$fpsLabel.Dock = 'Bottom'; $fpsLabel.Height = 56; $fpsLabel.TextAlign = 'MiddleLeft'
$fpsLabel.ForeColor = [System.Drawing.Color]::Red
$fpsLabel.Font = New-Object System.Drawing.Font("Consolas", 28, [System.Drawing.FontStyle]::Bold)
$fpsLabel.Text = ''
$form.Controls.Add($fpsLabel)
$fpsLabel.Add_DoubleClick({ $form.Close() })
$graph.Add_DoubleClick({ $form.Close() })

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
			$r.key = $last.Index
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

# Reescalado en uso: la ultima linea del log con "<NNN>x<NNN> -> <NNN>x<NNN>" de DLSS, XeSS o FSR (se ignoran las
# imagenes pequenas de los menus).
$script:scaleText = $null; $script:scaleLog = ''; $script:scalePos = 0
function Get-ScaleMode {
	# El modo elegido en el lanzador (launcher.ini, scalelabel), con su nombre completo.
	try {
		$ini = Join-Path (Split-Path $LogDir -Parent) 'launcher.ini'
		$line = Get-Content $ini -ErrorAction Stop | Where-Object { $_ -like 'scalelabel=*' } | Select-Object -First 1
		if (-not $line) { return '' }
		$label = $line.Substring(11)
		$mode = ($label -split ' ', 2)[1]
		if (-not $mode) { return '' }
		$mode = $mode -replace 'U\.cal\.\+', 'Ultra calidad+' -replace 'U\.calidad', 'Ultra calidad' -replace 'Equilib\.', 'Equilibrado' -replace 'U\.rendim\.', 'Ultra rendimiento' -replace 'Rendim\.', 'Rendimiento'
		return ' ' + $mode
	} catch { return '' }
}
function Get-Scale {
	if (-not $LogDir -or -not (Test-Path $LogDir)) { return $null }
	$log = Get-ChildItem -Path $LogDir -Filter "*.out.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
	if (-not $log) { return $null }
	if ($log.FullName -ne $script:scaleLog) { $script:scaleLog = $log.FullName; $script:scalePos = 0; $script:scaleText = $null }
	try {
		$fs = [System.IO.File]::Open($log.FullName, 'Open', 'Read', 'ReadWrite')
		if ($fs.Length -lt $script:scalePos) { $script:scalePos = 0 }
		$fs.Seek($script:scalePos, 'Begin') | Out-Null
		$take = [Math]::Min($fs.Length - $script:scalePos, 8MB)
		$buf = New-Object byte[] $take
		$n = $fs.Read($buf, 0, $take); $script:scalePos += $n
		$fs.Close()
		$text = [System.Text.Encoding]::UTF8.GetString($buf, 0, $n)
		$m = [regex]::Matches($text, '(DLSS|XeSS|FSR 3\.1)[^\r\n]*?(\d{3,4})x(\d{3,4}) -> (\d{3,4})x(\d{3,4})')
		for ($i = $m.Count - 1; $i -ge 0; $i--) {
			$h = [int]$m[$i].Groups[3].Value
			if ($h -ge 600) {
				$kind = $m[$i].Groups[1].Value -replace ' 3\.1', ''
				if (-not $script:screenH) { $script:screenH = (Get-CimInstance Win32_VideoController | Where-Object { $_.CurrentVerticalResolution } | Select-Object -First 1).CurrentVerticalResolution;  }
				$screen = $script:screenH  # (resolucion real: Screen.Bounds daba 960 con el escalado de Windows al 150 %)
				$script:scaleText = ("{0}{1} {2}p -> {3}p" -f $kind, (Get-ScaleMode), $h, $screen)
				break
			}
		if (-not $script:scaleText) {
			# Sin reescalador: la resolucion a la que dibuja el juego (con TURBO, 1080 aunque pongas 1440).
			$n = [regex]::Matches($text, 'performance mode renders (\d{3,4})x(\d{3,4})')
			if ($n.Count -gt 0) {
				if (-not $script:screenH) { $script:screenH = (Get-CimInstance Win32_VideoController | Where-Object { $_.CurrentVerticalResolution } | Select-Object -First 1).CurrentVerticalResolution;  }
				$script:scaleText = ("Nativo {0}p -> {1}p" -f $n[$n.Count - 1].Groups[2].Value, $script:screenH)
			}
		}
		}
	} catch { }
	return $script:scaleText
}

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 1000
$timer.Add_Tick({
	# Se cierra sola cuando se cierra el juego (despues de haberlo visto abierto).
	$emu = Get-Process -Name 'kyty_emulator*' -ErrorAction SilentlyContinue
	$emuWin = $emu | Where-Object { $_.MainWindowHandle -ne 0 }
	# (Despues de que salga la ventana del juego: antes reserva su memoria de golpe, y con el limite ya puesto fallaba.)
	$emuW = $emu | Where-Object { $_.MainWindowHandle -ne 0 }
	if ($RamGB -gt 0 -and $emuW) {
		$p = $emuW | Select-Object -First 1
		if ($p.Id -ne $script:limitedPid) {
			$max = [uint64](($RamGB - 4) * 1GB)
			$ok = [RamLimit]::SetProcessWorkingSetSizeEx($p.Handle, [UIntPtr][uint64]200MB, [UIntPtr]$max, 6)
			$script:limitedPid = $p.Id; $script:limitOk = $ok
		}
	}
	if ($emuWin) { $script:emuSeen = $true; $script:emuGone = 0 }
	elseif ($script:emuSeen) { $script:emuGone++; if ($script:emuGone -ge 2) { $form.Close(); return } }
	$lines = @()
	if ($smi) {
		$q = & $smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
		if ($q) {
			$v = $q -split ',\s*'
			$gpu = [int]$v[0]; $used = [int]$v[1]; $total = [int]$v[2]; $temp = [int]$v[3]
			if ($VramMB -gt 0 -and $VramMB -lt $total) {
				$reserved = $total - $VramMB
				$shown = [Math]::Max(0, $used - $reserved)
				$max = $VramMB
			} elseif ($MaxMB -gt 0) { $shown = $used; $max = $MaxMB } else { $shown = $used; $max = $total }
			$lines += ("GPU   {0,3} %  {1} C" -f $gpu, $temp)
			$lines += ("VRAM  {0:N1} / {1:N0} GB" -f ($shown / 1024.0), ($max / 1024.0))
			if ($shown -gt $max) { $label.ForeColor = [System.Drawing.Color]::Red } else { $label.ForeColor = [System.Drawing.Color]::Orange }
		}
	} else { $lines += "GPU   (sin nvidia-smi)" }
	try {
		$cpu = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
		$os = Get-CimInstance Win32_OperatingSystem
		$ramUsed = ($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB
		$ramTotal = $os.TotalVisibleMemorySize / 1MB
		$lines += ("CPU   {0,3} %" -f $cpu)
		if ($RamGB -gt 0) {
			$ws = 0; $e2 = Get-Process -Name 'kyty_emulator*' -ErrorAction SilentlyContinue | Select-Object -First 1
			if ($e2) { $ws = $e2.WorkingSet64 / 1GB }
			$lines += ("RAM   {0:N1} / {1} GB{2}" -f ($ws + 4), $RamGB, $(if ($script:limitedPid -and -not $script:limitOk) { ' (sin limite!)' } else { '' }))
		} else {
			$lines += ("RAM   {0:N1} / {1:N0} GB" -f $ramUsed, $ramTotal)
		}
	} catch { }
	$shownFps = $null
	$fps = Get-Fps
	if ($fps) {
		$shownFps = if ($fps.fgShown) { $fps.fgShown } elseif ($fps.fps) { $fps.fps } else { $null }
		if ($fps.mean) {
			$lines += ("Frame {0:N1} ms" -f $fps.mean)
			if ($fps.key -ne $script:lastKey) {
				$script:lastKey = $fps.key
				$script:ft.Add($fps.mean); $script:ftMax.Add($fps.max)
				while ($script:ft.Count -gt 60) { $script:ft.RemoveAt(0); $script:ftMax.RemoveAt(0) }
			}
		}
	}
	$sc = Get-Scale
	if ($sc) {
		$scaleLabel.Text = "Escala $sc"
		$scaleLabel.ForeColor = if ($sc -like 'DLSS*') { [System.Drawing.Color]::LimeGreen } elseif ($sc -like 'FSR*') { [System.Drawing.Color]::Red } elseif ($sc -like 'XeSS*') { [System.Drawing.Color]::DeepSkyBlue } else { [System.Drawing.Color]::Orange }
	} else { $scaleLabel.Text = '' }
	$fgName = ''
	try { $l = Get-Content (Join-Path (Split-Path $LogDir -Parent) 'launcher.ini') -ErrorAction Stop | Where-Object { $_ -like 'fglabel=*' } | Select-Object -First 1; if ($l) { $fgName = $l.Substring(8) } } catch { }
	if ($fgName -and $fgName -ne 'Sin frame generation') {
		$mult = if ($fps -and $fps.fgMult) { " (ahora x$($fps.fgMult))" } else { '' }
		$fgLabel.Text = $(if ($fgName -like 'MODO *') { $fgName } else { "FG $fgName$mult" })
		$fgLabel.ForeColor = if ($fgName -match 'DLSS|Nvidia') { [System.Drawing.Color]::LimeGreen } else { [System.Drawing.Color]::Red }
	} else { $fgLabel.Text = '' }
	$gfx = @()
	try {
		$ini = Get-Content (Join-Path (Split-Path $LogDir -Parent) 'launcher.ini') -ErrorAction Stop
		$names = @('Maximo', 'Medio', 'Bajo')
		foreach ($pair in @(@('gfxtex', 'Texturas'), @('gfxdist', 'Distancia de dibujado'))) {
			$l = $ini | Where-Object { $_ -like "$($pair[0])=*" } | Select-Object -First 1
			$v = 0; if ($l) { [void][int]::TryParse($l.Substring($pair[0].Length + 1), [ref]$v) }
			$gfx += "$($pair[1]) $($names[[math]::Min([math]::Max($v, 0), 2)])"
		}
	} catch { }
	$gfxLabel.Text = ($gfx -join ', ')
	Update-Ls
	if ($script:lsFps) { $lines += ("FPS LS {0}" -f $script:lsFps) }
	$graph.Invalidate()
	$label.Text = ($lines -join "`n")
	$fpsLabel.Text = $(if ($fps -and $shownFps) { "FPS $shownFps" } else { '' })
	# FPS en el color del frame generation: verde NVIDIA, rojo AMD, blanco sin frame generation.
	$fpsLabel.ForeColor = if ($fgName -and $fgName -ne 'Sin frame generation') { if ($fgName -match 'DLSS|Nvidia') { [System.Drawing.Color]::LimeGreen } else { [System.Drawing.Color]::Red } } else { [System.Drawing.Color]::White }
})
$timer.Start()
[System.Windows.Forms.Application]::Run($form)
