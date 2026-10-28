# Anade "Demon's Souls" (el port) a Steam como juego que no es de Steam,
# con el fondo-souls.jpg como banner/portada/fondo y el icono del exe.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe  = Join-Path $dir "Demon's Souls.exe"
$bg   = Join-Path $dir 'fondo-souls.jpg'
if (Test-Path -LiteralPath (Join-Path $dir 'fondo-steam.jpg')) { $bg = Join-Path $dir 'fondo-steam.jpg' }  # titulo a la derecha
$name = "Demon's Souls"
if (-not (Test-Path -LiteralPath $exe)) { Write-Host "No encuentro $exe"; exit 1 }
if (-not (Test-Path -LiteralPath $bg))  { Write-Host "No encuentro $bg"; exit 1 }

$steam = (Get-ItemProperty 'HKCU:\Software\Valve\Steam' -ErrorAction SilentlyContinue).SteamPath
if (-not $steam) { Write-Host 'No encuentro Steam instalado.'; exit 1 }
$steam = $steam -replace '/', '\'
$steamExe = Join-Path $steam 'steam.exe'

# Steam debe estar cerrado para que no sobrescriba la lista
$wasOpen = $false
if (Get-Process steam -ErrorAction SilentlyContinue) {
	$wasOpen = $true
	Write-Host 'Cerrando Steam...'
	Start-Process $steamExe '-shutdown'
	for ($i = 0; $i -lt 30 -and (Get-Process steam -ErrorAction SilentlyContinue); $i++) { Start-Sleep 1 }
	Get-Process steam -ErrorAction SilentlyContinue | Stop-Process -Force
	Start-Sleep 2
}

# appid de Steam para accesos directos: crc32(exe entre comillas + nombre) | 0x80000000
$table = New-Object 'uint32[]' 256
for ($n = 0; $n -lt 256; $n++) { $c = [uint32]$n; for ($k = 0; $k -lt 8; $k++) { if ($c -band 1) { $c = [uint32]([uint32]3988292384 -bxor ($c -shr 1)) } else { $c = [uint32]($c -shr 1) } }; $table[$n] = $c }
function Get-Crc32([byte[]]$b) { $c = [uint32]4294967295; foreach ($x in $b) { $c = [uint32]($table[($c -bxor $x) -band 0xFF] -bxor ($c -shr 8)) }; return [uint32]($c -bxor [uint32]4294967295) }
$exeQ  = '"' + $exe + '"'
$utf8  = New-Object System.Text.UTF8Encoding($false)
$appid = [uint32]((Get-Crc32 ($utf8.GetBytes($exeQ + $name))) -bor [uint32]2147483648)
$appidS = [BitConverter]::ToInt32([BitConverter]::GetBytes($appid), 0)

function Add-Str($l, [string]$k, [string]$v) { $l.Add(1); $l.AddRange($utf8.GetBytes($k)); $l.Add(0); $l.AddRange($utf8.GetBytes($v)); $l.Add(0) }
function Add-Int($l, [string]$k, [int]$v)    { $l.Add(2); $l.AddRange($utf8.GetBytes($k)); $l.Add(0); $l.AddRange([BitConverter]::GetBytes($v)) }

function Save-Fit([string]$src, [string]$dst, [int]$w, [int]$h) {
	$im = [System.Drawing.Image]::FromFile($src)
	$sc = [math]::Max($w / $im.Width, $h / $im.Height); $sw = [int]($w / $sc); $sh = [int]($h / $sc)
	$sx = [int](($im.Width - $sw) / 2); $sy = [int](($im.Height - $sh) / 2)
	$bm = New-Object System.Drawing.Bitmap $w, $h; $g = [System.Drawing.Graphics]::FromImage($bm)
	$g.InterpolationMode = 'HighQualityBicubic'
	$g.DrawImage($im, (New-Object System.Drawing.Rectangle 0, 0, $w, $h), (New-Object System.Drawing.Rectangle $sx, $sy, $sw, $sh), [System.Drawing.GraphicsUnit]::Pixel)
	$g.Dispose(); $im.Dispose(); $bm.Save($dst, [System.Drawing.Imaging.ImageFormat]::Jpeg); $bm.Dispose()
}

$users = Get-ChildItem (Join-Path $steam 'userdata') -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^\d+$' -and $_.Name -ne '0' }
if (-not $users) { Write-Host 'No hay usuarios de Steam (inicia sesion en Steam una vez).'; exit 1 }

foreach ($u in $users) {
	$cfg = Join-Path $u.FullName 'config'; New-Item -ItemType Directory -Force $cfg | Out-Null
	$grid = Join-Path $cfg 'grid'; New-Item -ItemType Directory -Force $grid | Out-Null
	$vdf = Join-Path $cfg 'shortcuts.vdf'

	# Imagenes: banner, portada, fondo y logo/icono
	# Cada imagen recortada al tamano que pide Steam, centrada (sin cortar lo importante)
	Save-Fit $bg (Join-Path $grid "$appid.jpg") 920 430
	Save-Fit $bg (Join-Path $grid "${appid}p.jpg") 600 900
	Save-Fit $bg (Join-Path $grid "${appid}_hero.jpg") 3840 1240
	# Logo transparente: Steam no escribe el nombre encima del fondo (sin letras)
	$lg = New-Object System.Drawing.Bitmap 16, 16; $lg.Save((Join-Path $grid "${appid}_logo.png"), [System.Drawing.Imaging.ImageFormat]::Png); $lg.Dispose()
	$ico = Join-Path $grid "${appid}_icon.png"
	try { $i = [System.Drawing.Icon]::ExtractAssociatedIcon($exe); $i.ToBitmap().Save($ico, [System.Drawing.Imaging.ImageFormat]::Png) } catch { $ico = $exe }

	$old = $null
	if (Test-Path -LiteralPath $vdf) {
		$old = [IO.File]::ReadAllBytes($vdf)
		Copy-Item -LiteralPath $vdf "$vdf.bak-souls" -Force
		$lat = [Text.Encoding]::GetEncoding(28591).GetString($old)
		if ($lat.Contains([Text.Encoding]::GetEncoding(28591).GetString($utf8.GetBytes($exeQ)))) { Write-Host "Usuario $($u.Name): ya estaba en Steam (imagenes actualizadas)."; continue }
		$idx = ([regex]::Matches($lat, "(?i)\x02appid\x00")).Count
	} else { $idx = 0 }

	$e = New-Object 'System.Collections.Generic.List[byte]'
	$e.Add(0); $e.AddRange($utf8.GetBytes("$idx")); $e.Add(0)
	Add-Int $e 'appid' $appidS
	Add-Str $e 'AppName' $name
	Add-Str $e 'Exe' $exeQ
	Add-Str $e 'StartDir' ('"' + $dir + '\"')
	Add-Str $e 'icon' $ico
	Add-Str $e 'ShortcutPath' ''
	Add-Str $e 'LaunchOptions' ''
	Add-Int $e 'IsHidden' 0
	Add-Int $e 'AllowDesktopConfig' 1
	Add-Int $e 'AllowOverlay' 1
	Add-Int $e 'OpenVR' 0
	Add-Int $e 'Devkit' 0
	Add-Str $e 'DevkitGameID' ''
	Add-Int $e 'DevkitOverrideAppID' 0
	Add-Int $e 'LastPlayTime' 0
	Add-Str $e 'FlatpakAppID' ''
	$e.Add(0); $e.AddRange($utf8.GetBytes('tags')); $e.Add(0); $e.Add(8)
	$e.Add(8)

	$out = New-Object 'System.Collections.Generic.List[byte]'
	if ($old -and $old.Length -ge 2 -and $old[$old.Length-1] -eq 8 -and $old[$old.Length-2] -eq 8) {
		$out.AddRange([byte[]]$old[0..($old.Length-3)])
	} else {
		$out.Add(0); $out.AddRange($utf8.GetBytes('shortcuts')); $out.Add(0)
	}
	$out.AddRange($e); $out.Add(8); $out.Add(8)
	[IO.File]::WriteAllBytes($vdf, $out.ToArray())
	Write-Host "Usuario $($u.Name): Demon's Souls anadido a Steam."
}

if ($wasOpen) { Start-Process $steamExe }
Write-Host ''
Write-Host 'Listo. Abre Steam: aparece en tu biblioteca como "Demon''s Souls".'
