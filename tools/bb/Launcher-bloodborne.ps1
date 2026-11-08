# Bloodborne PC - lanzador para shadPS4 (ventana). Launcher-bloodborne.cmd lo abre.
#   JUGAR    -> escribe los ajustes en la config de shadPS4 (user\config.toml de la 0.15 o user\config.json de las
#               versiones nuevas; copia de seguridad antes), prepara las variables BB_* para el emulador propio que
#               viene y arranca shadPS4.exe -g "<juego>\eboot.bin" -f true|false.
#   -Play    -> sin ventana: arranca con los ultimos ajustes guardados (lo usa el acceso de Steam).
#   -Lang en -> interfaz en ingles (Launcher-bloodborne-EN.ps1 es esta misma copia con 'en' por defecto).
#   -NoUi    -> solo carga las funciones (pruebas).
# Ajustes del lanzador en launcher.ini; registro en launcher.log (los dos junto a este script).
param([string]$Lang = 'es', [switch]$Play, [switch]$NoUi)

$ErrorActionPreference = 'Stop'
$root    = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$iniFile = Join-Path $root 'launcher.ini'
$logFile = Join-Path $root 'launcher.log'
$utf8NoBom = New-Object System.Text.UTF8Encoding $false

function T([string]$es, [string]$en) { if ($Lang -eq 'en') { return $en } return $es }

# ---------------------------------------------------------------- registro
function Write-Log([string]$msg) {
	try {
		if ((Test-Path -LiteralPath $logFile) -and ((Get-Item -LiteralPath $logFile).Length -gt 2MB)) {
			Move-Item -LiteralPath $logFile ($logFile + '.old') -Force
		}
		[IO.File]::AppendAllText($logFile, "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg`r`n", $utf8NoBom)
	} catch { }
}
trap { Write-Log "ERROR: $_ `n$($_.ScriptStackTrace)"; break }

# ---------------------------------------------------------------- launcher.ini
$script:defaults = [ordered]@{
	game = ''; emu = ''; userdir = ''
	winres = 'auto'; intres = '1920x1080'; fsmode = '1'; present = '1'; fps = '1'; patches = '1'; useupdate = '1'
	hdr = '0'; showfps = '0'; metricas = '0'
	upscaler = '0'; upmode = '1'; sharp = '75'; framegen = '0'; sceneopt = '0'; bbfg = '0'; antilag = '0'; reflex = '1'
	tex = '1'; q_shadows = '0'; q_ao = '0'; q_fog = '0'; q_aniso = '0'; q_ca = '0'; q_blur = '0'; q_grain = '0'
	input = '0'; motion = '0'; bginput = '0'
	ram16 = '0'; experimental = '0'; modo = '4'; safemode = '1'; soundfix = '1'
}
$script:S = [ordered]@{}
function Read-Settings {
	foreach ($k in $script:defaults.Keys) { $script:S[$k] = $script:defaults[$k] }
	if (Test-Path -LiteralPath $iniFile) {
		foreach ($line in [IO.File]::ReadAllLines($iniFile)) {
			if ($line -match '^\s*([A-Za-z0-9_]+)\s*=(.*)$') { $script:S[$Matches[1]] = $Matches[2].Trim() }
		}
	}
}
function Save-Settings {
	try {
		$lines = foreach ($k in $script:S.Keys) { "$k=$($script:S[$k])" }
		[IO.File]::WriteAllText($iniFile, (($lines -join "`r`n") + "`r`n"), $utf8NoBom)
	} catch { Write-Log "No se pudo guardar launcher.ini: $_" }
}
# (no tocar lo del dueno) Solo se escriben en config.toml / config.json las claves que el usuario ha cambiado A MANO
# en ESTA sesion (marcas en memoria, nunca en launcher.ini). Cambios hechos por el programa no marcan nada.
$script:cfgSettingKeys = @{ winres = @('winW', 'winH'); intres = @('intW', 'intH'); fsmode = @('full', 'fsMode'); present = @('present'); hdr = @('hdr'); upscaler = @('fsr', 'rcas', 'rcasAtt'); sharp = @('fsr', 'rcas', 'rcasAtt'); motion = @('motion'); bginput = @('bgInput') }
$script:chg = @{}
function Set-Setting([string]$k, $v) { $script:S[$k] = "$v"; Save-Settings }
$script:chgAuto = @{}  # cambios hechos por un modo / Optimizar (no cuentan en modo seguro)
function Mark-Changed([string]$k, [switch]$Auto) { if ($Auto) { $script:chgAuto[$k] = $true } else { $script:chg[$k] = $true } }

# ---------------------------------------------------------------- param.sfo
# Formato SFO: cabecera (magic \0PSF, version, tabla de claves, tabla de datos, n entradas) y entradas de 16 bytes.
function Read-ParamSfo([string]$path) {
	$r = @{}
	try {
		$b = [IO.File]::ReadAllBytes($path)
		if ($b.Length -lt 20 -or $b[0] -ne 0 -or $b[1] -ne 0x50 -or $b[2] -ne 0x53 -or $b[3] -ne 0x46) { return $r }
		$keyT = [BitConverter]::ToUInt32($b, 8); $datT = [BitConverter]::ToUInt32($b, 12); $n = [BitConverter]::ToUInt32($b, 16)
		for ($i = 0; $i -lt $n; $i++) {
			$e = 20 + 16 * $i
			$ko = [BitConverter]::ToUInt16($b, $e); $fmt = [BitConverter]::ToUInt16($b, $e + 2)
			$len = [BitConverter]::ToUInt32($b, $e + 4); $dof = [BitConverter]::ToUInt32($b, $e + 12)
			$ks = $keyT + $ko; $ke = $ks; while ($ke -lt $b.Length -and $b[$ke] -ne 0) { $ke++ }
			$key = [Text.Encoding]::ASCII.GetString($b, $ks, $ke - $ks)
			$ds = $datT + $dof
			if ($fmt -eq 0x0404) { $r[$key] = [BitConverter]::ToUInt32($b, $ds) }
			else { $r[$key] = [Text.Encoding]::UTF8.GetString($b, $ds, $len).TrimEnd([char]0) }
		}
	} catch { Write-Log "param.sfo ilegible ($path): $_" }
	return $r
}

# Bloodborne: series conocidas. Otra serie se acepta con aviso.
# Versiones compatibles (fuentes publicas, 28-10-2026). Solo lo comprobado; lo de una sola fuente va como "sin confirmar".
#   CUSA00207: EP9000-CUSA00207_00-BLOODBORNE0000EU (orbispatches.com, PlayStation Store EU) -> Europa.
#   CUSA03173: EP9000-CUSA03173_00-BLOODBORNE0000EU (ps.konsoloyun.gen.tr) -> Europa, GOTY (con The Old Hunters).
#   CUSA00900: EE. UU. (README de github.com/deadinside28/bloodborne_pc; bloodbornepc.wiki la nombra) -> sin confirmar en
#              una base de datos de PS4.
#   CUSA03023: Asia (solo el README de deadinside28/bloodborne_pc) -> sin confirmar.
#   Japon y otras ediciones asiaticas: no se encontro la serie -> no se listan (se aceptan con aviso).
#   Version: 1.09 (01.09), la ultima; la base 1.00 se cierra al arrancar en los ports (deadinside28, bloodbornepc.wiki).
$script:bbVersion = '01.09'
$script:bbSerials = @{
	'CUSA00207' = (T 'Europa' 'Europe')
	'CUSA03173' = (T 'Europa - GOTY (The Old Hunters)' 'Europe - GOTY (The Old Hunters)')
	'CUSA00900' = (T 'EE. UU. (sin confirmar)' 'USA (unconfirmed)')
	'CUSA03023' = (T 'Asia (sin confirmar)' 'Asia (unconfirmed)')
}
# Estado de compatibilidad: version efectiva = la de la actualizacion si se usa, si no la base.
function Get-Compat($g) {
	if (-not $g.Ok) { return $null }
	$v = $g.Version; if ($g.Update -and $g.UpdateVersion -and $script:S.useupdate -ne '0') { $v = $g.UpdateVersion }
	if (-not $g.Known) { return [pscustomobject]@{ Ok = $false; Text = (T "Serie $($g.Serial) no listada: puede funcionar (sin comprobar)" "Title ID $($g.Serial) not listed: may work (unverified)") } }
	if ($v -eq $script:bbVersion) { return [pscustomobject]@{ Ok = $true; Text = (T 'Compatible (1.09)' 'Compatible (1.09)') } }
	return [pscustomobject]@{ Ok = $false; Text = (T "Versión ${v}: instala la actualización 1.09" "Version $v`: install update 1.09") }
}

# Comprueba una carpeta de juego. Devuelve un objeto con Ok, Serial, Title, Version, Update, Problems.
function Test-GameFolder([string]$dir) {
	$o = [pscustomobject]@{ Ok = $false; Dir = $dir; Serial = ''; Title = ''; Version = ''; Region = ''; Update = ''; UpdateVersion = ''; Known = $false; Problems = @() }
	if (-not $dir) { $o.Problems += (T 'No has elegido la carpeta del juego.' 'No game folder selected.'); return $o }
	if (-not (Test-Path -LiteralPath $dir -PathType Container)) { $o.Problems += (T "No existe la carpeta: $dir" "Folder not found: $dir"); return $o }
	$eboot = Join-Path $dir 'eboot.bin'; $sfo = Join-Path (Join-Path $dir 'sce_sys') 'param.sfo'
	if (-not (Test-Path -LiteralPath $eboot)) { $o.Problems += (T 'Falta eboot.bin en la carpeta.' 'eboot.bin is missing.') }
	if (-not (Test-Path -LiteralPath $sfo)) { $o.Problems += (T 'Falta sce_sys\param.sfo.' 'sce_sys\param.sfo is missing.') }
	if (Test-Path -LiteralPath $sfo) {
		$p = Read-ParamSfo $sfo
		$o.Serial = "$($p['TITLE_ID'])"; $o.Title = "$($p['TITLE'])"; $o.Version = "$($p['APP_VER'])"
	}
	if (-not $o.Serial) { $leaf = Split-Path -Leaf $dir; if ($leaf -match '^(CUSA\d{5})') { $o.Serial = $Matches[1] } }
	if ($o.Serial -and $script:bbSerials.ContainsKey($o.Serial)) { $o.Known = $true; $o.Region = $script:bbSerials[$o.Serial] }
	# Actualizacion: shadPS4 usa sola la carpeta hermana <carpeta>-UPDATE o <carpeta>-patch.
	foreach ($suf in '-UPDATE', '-patch') {
		$u = $dir.TrimEnd('\', '/') + $suf
		if (Test-Path -LiteralPath (Join-Path (Join-Path $u 'sce_sys') 'param.sfo')) {
			$o.Update = $u; $o.UpdateVersion = "$((Read-ParamSfo (Join-Path (Join-Path $u 'sce_sys') 'param.sfo'))['APP_VER'])"; break
		}
	}
	$o.Ok = ($o.Problems.Count -eq 0)
	return $o
}

# Busca juegos (carpetas con eboot.bin, sin las de actualizacion) y emuladores (shadPS4.exe) bajo la carpeta del lanzador.
function Find-Games {
	$res = @()
	foreach ($base in @($root, (Join-Path $root 'games'), (Join-Path $root 'juegos'))) {
		if (-not (Test-Path -LiteralPath $base)) { continue }
		foreach ($d in @(Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)) {
			if ($d.Name -match '-(UPDATE|patch)$') { continue }
			if (Test-Path -LiteralPath (Join-Path $d.FullName 'eboot.bin')) { $res += $d.FullName }
		}
	}
	return @($res | Select-Object -Unique)
}
function Find-Emulators {
	$res = @()
	try {
		$res = @(Get-ChildItem -LiteralPath $root -Filter 'shadPS4.exe' -Recurse -Depth 3 -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
	} catch {
		# PowerShell 5.1 sin -Depth en algunas versiones: a mano, 3 niveles.
		$res = @(Get-ChildItem -LiteralPath $root -Filter 'shadPS4.exe' -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 20 | ForEach-Object { $_.FullName })
	}
	return @($res | Select-Object -Unique)
}
function Get-EmuLabel([string]$exe) {
	$v = ''; try { $v = (Get-Item -LiteralPath $exe).VersionInfo.ProductVersion } catch { }
	$rel = $exe; if ($exe.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { $rel = $exe.Substring($root.Length).TrimStart('\') }
	if ($v) { return "$rel  ($v)" } return $rel
}

# Carpeta "user" de shadPS4: la portable (carpeta de trabajo\user). shadPS4 la busca en la carpeta desde la que se
# arranca; si no existe, en %APPDATA%\shadPS4 (en Windows la crea en la carpeta de trabajo). Orden: la elegida en
# launcher.ini, junto al exe, junto al lanzador, %APPDATA%\shadPS4.
# (10-29) Resolucion interna: en Bloodborne la manda el "Resolution Patch WxH" de los parches de shadPS4, no la config.
# Se activa el de la resolucion elegida y se desactivan los demas "Resolution Patch" (1920x1080 = ninguno: la nativa).
# Antes se guarda copia en user\launcher-backups. El resto de parches no se toca.
function Set-ResolutionPatch([string]$userDir, [string]$res) {
	if (-not $res) { return }
	foreach ($rel in @('patches\shadPS4\Bloodborne.xml', 'patches\GoldHEN\Bloodborne-Orbis.xml')) {
		$x = Join-Path $userDir $rel
		if (-not (Test-Path -LiteralPath $x)) { continue }
		try {
			$txt = [IO.File]::ReadAllText($x)
			if ($txt -notmatch 'Name="Resolution Patch ') { continue }
			$hasTarget = $txt -match ('Name="Resolution Patch ' + [regex]::Escape($res) + ' \(')
			$new = [regex]::Replace($txt, '(<Metadata[^>]*Name="Resolution Patch ([0-9]+x[0-9]+) [^"]*"[^>]*isEnabled=")(true|false)(")', {
				param($m)
				$on = if ($hasTarget -and $m.Groups[2].Value -eq $res) { 'true' } else { 'false' }
				$m.Groups[1].Value + $on + $m.Groups[4].Value
			})
			if ($new -ne $txt) {
				$bd = Join-Path $userDir 'launcher-backups'; New-Item -ItemType Directory -Force -Path $bd | Out-Null
				Copy-Item -LiteralPath $x -Destination (Join-Path $bd ((Split-Path -Leaf $x) + '.' + (Get-Date -Format 'yyyyMMdd-HHmmss'))) -Force
				[IO.File]::WriteAllText($x, $new, (New-Object System.Text.UTF8Encoding $false))
				Write-Log "Resolucion interna $res -> $rel ($(if ($hasTarget) { "Resolution Patch $res activado" } else { 'sin parche de resolucion: la nativa del juego' }))"
			}
		} catch { Write-Log "No se pudo cambiar el parche de resolucion en ${rel}: $_" }
	}
}
function Get-UserDir([string]$emuExe) {
	if ($script:S.userdir -and (Test-Path -LiteralPath $script:S.userdir)) { return $script:S.userdir }
	$cands = @((Join-Path $root 'user'))
	if ($emuExe) { $cands += (Join-Path (Split-Path -Parent $emuExe) 'user') }
	if ($env:APPDATA) { $cands += (Join-Path $env:APPDATA 'shadPS4') }
	foreach ($c in $cands) { if (Test-Path -LiteralPath (Join-Path $c 'config.toml')) { return $c }; if (Test-Path -LiteralPath (Join-Path $c 'config.json')) { return $c } }
	foreach ($c in $cands) { if (Test-Path -LiteralPath $c) { return $c } }
	if ($emuExe) { return (Join-Path (Split-Path -Parent $emuExe) 'user') }
	return (Join-Path $root 'user')
}

# ---------------------------------------------------------------- config de shadPS4
# TOML (0.15 y anteriores): se cambia solo la linea de cada clave dentro de su [seccion]; el resto queda igual.
function Format-TomlValue($val) {
	if ($val -is [bool]) { return $(if ($val) { 'true' } else { 'false' }) }
	if ($val -is [int] -or $val -is [long] -or $val -is [double]) { return ([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $val)) }
	return '"' + ("$val" -replace '\\', '\\' -replace '"', '\"') + '"'
}
# Cambia SOLO el valor de cada clave en su [seccion]; el resto del archivo queda igual byte a byte. Si la clave no
# existe, se anade una linea al final de su seccion (no se crean secciones nuevas).
function Set-TomlValues([string]$text, $changes) {
	$nl = "`n"; if ($text -match "`r`n") { $nl = "`r`n" }
	foreach ($sec in $changes.Keys) {
		$hm = [regex]::Match($text, '(?m)^[ \t]*\[' + [regex]::Escape($sec) + '\][ \t]*\r?$')
		if (-not $hm.Success) { Write-Log "config.toml sin [$sec]: no se cambia"; continue }
		foreach ($key in $changes[$sec].Keys) {
			$hm = [regex]::Match($text, '(?m)^[ \t]*\[' + [regex]::Escape($sec) + '\][ \t]*\r?$')
			$start = $hm.Index + $hm.Length
			$nx = [regex]::Match($text.Substring($start), '(?m)^[ \t]*\[[^\]\r\n]+\][ \t]*\r?$')
			$end = $(if ($nx.Success) { $start + $nx.Index } else { $text.Length })
			$body = $text.Substring($start, $end - $start)
			$vs = Format-TomlValue $changes[$sec][$key]
			$km = [regex]::Match($body, '(?m)^([ \t]*' + [regex]::Escape($key) + '[ \t]*=[ \t]*)("(?:[^"\\\r\n]|\\.)*"|[^ \t#\r\n]*)')
			if ($km.Success) {
				if ($km.Groups[2].Value -eq $vs) { continue }
				$body = $body.Substring(0, $km.Groups[2].Index) + $vs + $body.Substring($km.Groups[2].Index + $km.Groups[2].Length)
			} else {
				$trim = $body.TrimEnd("`r", "`n", ' ', "`t"); $body = $trim + $nl + "$key = $vs" + $body.Substring($trim.Length)
				if (-not $nx.Success -and -not $body.EndsWith("`n")) { $body += $nl }
			}
			$text = $text.Substring(0, $start) + $body + $text.Substring($end)
		}
	}
	return $text
}
# JSON: igual, se cambia solo el valor de la clave (sin reformatear). Si la clave no esta, no se toca.
function Set-JsonValues([string]$text, $vals) {
	foreach ($k in $vals.Keys) {
		$v = $vals[$k]
		if ($v -is [bool]) { $vs = $(if ($v) { 'true' } else { 'false' }) } elseif ($v -is [int] -or $v -is [long] -or $v -is [double]) { $vs = ([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $v)) } else { $vs = '"' + ("$v" -replace '\\', '\\' -replace '"', '\"') + '"' }
		$m = [regex]::Match($text, '("' + [regex]::Escape($k) + '"\s*:\s*)("(?:[^"\\]|\\.)*"|-?[0-9.eE+]+|true|false|null)')
		if (-not $m.Success) { Write-Log "config.json sin '$k': no se anade"; continue }
		if ($m.Groups[2].Value -eq $vs) { continue }
		$text = $text.Substring(0, $m.Groups[2].Index) + $vs + $text.Substring($m.Groups[2].Index + $m.Groups[2].Length)
	}
	return $text
}

# Copia de seguridad: la primera vez config.toml.original (no se toca nunca mas) y en cada cambio una con fecha
# en user\launcher-backups (se guardan las 10 ultimas).
function Backup-ShadConfig([string]$file) {
	$orig = $file + '.original'
	if (-not (Test-Path -LiteralPath $orig)) { Copy-Item -LiteralPath $file $orig }
	$bd = Join-Path (Split-Path -Parent $file) 'launcher-backups'
	New-Item -ItemType Directory -Force -Path $bd | Out-Null
	Copy-Item -LiteralPath $file (Join-Path $bd ((Split-Path -Leaf $file) + '.' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
	Get-ChildItem -LiteralPath $bd -File | Where-Object { $_.Name -like ((Split-Path -Leaf $file) + '.*') } | Sort-Object Name -Descending | Select-Object -Skip 10 | Remove-Item -Force -ErrorAction SilentlyContinue
}

# Valores que el shadPS4 actual entiende: nombre comun -> (seccion, clave TOML 0.15, clave JSON nueva).
$script:shadKeys = @(
	@('winW',    'GPU',   'screenWidth',               'window_width'),
	@('winH',    'GPU',   'screenHeight',              'window_height'),
	@('intW',    'GPU',   'internalScreenWidth',       'internal_screen_width'),
	@('intH',    'GPU',   'internalScreenHeight',      'internal_screen_height'),
	@('full',    'GPU',   'Fullscreen',                'full_screen'),
	@('fsMode',  'GPU',   'FullscreenMode',            'full_screen_mode'),
	@('present', 'GPU',   'presentMode',               'present_mode'),
	@('hdr',     'GPU',   'allowHDR',                  'hdr_allowed'),
	@('fsr',     'GPU',   'fsrEnabled',                'fsr_enabled'),
	@('rcas',    'GPU',   'rcasEnabled',               'rcas_enabled'),
	@('rcasAtt', 'GPU',   'rcasAttenuation',           'rcas_attenuation'),
	@('motion',  'Input', 'isMotionControlsEnabled',   'motion_controls_enabled'),
	@('bgInput', 'Input', 'backgroundControllerInput', 'background_controller_input')
)
# Que config lee la version elegida: 0.15.x y anteriores config.toml; 0.20 y posteriores config.json (comprobado:
# 0.20.0 usa config.json). Entre medias o sin version legible: las dos (solo las mismas claves). Version del exe
# (ProductVersion / FileVersion) o del nombre de carpeta (v.0.15.0).
function Get-EmuVersion([string]$exe) {
	$cands = @()
	try { $vi = (Get-Item -LiteralPath $exe).VersionInfo; $cands += "$($vi.ProductVersion)"; $cands += "$($vi.FileVersion)" } catch { }
	try { $cands += (Split-Path -Leaf (Split-Path -Parent $exe)) } catch { }
	foreach ($c in $cands) { if ($c -match '(?<!\d)v?\.?(\d+)\.(\d+)(?:\.(\d+))?') { return [version]("$($Matches[1]).$($Matches[2]).$(if ($Matches[3]) { $Matches[3] } else { 0 })") } }
	return $null
}
function Get-ConfigKind([string]$exe) {
	$v = Get-EmuVersion $exe
	if (-not $v) { return 'both' }
	if ($v -lt [version]'0.16.0') { return 'toml' }
	if ($v -ge [version]'0.20.0') { return 'json' }
	return 'both'
}
function Write-ShadConfig([string]$userDir, $vals, [string]$Only = '') {
	$written = @()
	$toml = Join-Path $userDir 'config.toml'; $json = Join-Path $userDir 'config.json'
	if ($Only -eq 'json') { $toml = '' } elseif ($Only -eq 'toml') { $json = '' }  # 'both' o '': las dos
	if ($toml -and (Test-Path -LiteralPath $toml)) {
		$ch = [ordered]@{}
		foreach ($k in $script:shadKeys) { if ($vals.Contains($k[0])) { if (-not $ch.Contains($k[1])) { $ch[$k[1]] = [ordered]@{} }; $ch[$k[1]][$k[2]] = $vals[$k[0]] } }
		$old = [IO.File]::ReadAllText($toml)
		$new = Set-TomlValues $old $ch
		if ($new -ne $old) { Backup-ShadConfig $toml; [IO.File]::WriteAllText($toml, $new, $utf8NoBom); $written += $toml }
	}
	if ($json -and (Test-Path -LiteralPath $json)) {
		$old = [IO.File]::ReadAllText($json)
		$jv = [ordered]@{}; foreach ($k in $script:shadKeys) { if ($vals.Contains($k[0])) { $jv[$k[3]] = $vals[$k[0]] } }
		$new = Set-JsonValues $old $jv
		if ($new -ne $old) { Backup-ShadConfig $json; [IO.File]::WriteAllText($json, $new, $utf8NoBom); $written += $json }
	}
	return $written
}

# ---------------------------------------------------------------- parches de shadPS4 (user\patches\<repo>\*.xml)
# Cada repo trae files.json (archivo -> series) y XML con <Metadata Name=".." isEnabled="true|false" AppVer="..">.
function Find-PatchFiles([string]$userDir, [string]$serial) {
	$res = @(); $pd = Join-Path $userDir 'patches'
	if (-not $serial -or -not (Test-Path -LiteralPath $pd)) { return $res }
	foreach ($repo in @(Get-ChildItem -LiteralPath $pd -Directory -ErrorAction SilentlyContinue)) {
		$fj = Join-Path $repo.FullName 'files.json'
		if (Test-Path -LiteralPath $fj) {
			try {
				$m = [IO.File]::ReadAllText($fj) | ConvertFrom-Json
				foreach ($p in $m.PSObject.Properties) { if (@($p.Value) -contains $serial) { $f = Join-Path $repo.FullName $p.Name; if (Test-Path -LiteralPath $f) { $res += $f } } }
			} catch { Write-Log "files.json ilegible: $fj" }
		}
	}
	return @($res | Select-Object -Unique)
}
function Get-Patches([string]$userDir, [string]$serial, [string]$appVer) {
	$list = @()
	foreach ($f in (Find-PatchFiles $userDir $serial)) {
		try {
			$x = New-Object System.Xml.XmlDocument; $x.PreserveWhitespace = $true; $x.Load($f)
			foreach ($md in @($x.SelectNodes('//Metadata'))) {
				$ver = $md.GetAttribute('AppVer')
				$list += [pscustomobject]@{ File = $f; Name = $md.GetAttribute('Name'); AppVer = $ver; Enabled = ($md.GetAttribute('isEnabled') -eq 'true'); Note = $md.GetAttribute('Note'); VerOk = (-not $appVer -or -not $ver -or $ver -eq $appVer) }
			}
		} catch { Write-Log "Parche ilegible: $f ($_)" }
	}
	return $list
}
# Los xml de parches NUNCA se escriben (solo lectura). Para cambiarlos: shadPS4 / BB_Launcher.
$script:pat60   = '60\s*-?\s*FPS|60\s*fps|60fps|60 Frames'
$script:patUnlock = '(?i)fps|unlock|uncap|\b60\b'  # parche activo que desbloquea / quita el limite de FPS
$script:patCA   = '(?i)chromatic|aberra'
$script:patBlur = '(?i)motion\s*blur|desenfoque'
$script:patGrain= '(?i)film\s*grain|grain'

# ---------------------------------------------------------------- trofeos (user\game_data\<serie>\TrophyFiles\trophyNN\Xml\TROP.XML)
# Carpeta de partidas: la que diga config.json "home_dir" (o su equivalente en config.toml); si no, la carpeta user.
# Solo se lee: nunca se crean ni se copian carpetas de partidas.
function Get-HomeDir([string]$userDir) {
	$j = Join-Path $userDir 'config.json'
	if (Test-Path -LiteralPath $j) { $t = [IO.File]::ReadAllText($j); if ($t -match '"home_dir"\s*:\s*"((?:[^"\\]|\\.)*)"') { $h = $Matches[1] -replace '\\\\', '\' -replace '\\/', '/'; if ($h) { return $h } } }
	$tm = Join-Path $userDir 'config.toml'
	if (Test-Path -LiteralPath $tm) { $t = [IO.File]::ReadAllText($tm); if ($t -match '(?mi)^\s*(home_dir|homeDir|homeDirectory)\s*=\s*[''"]([^''"]*)[''"]') { if ($Matches[2]) { return $Matches[2] } } }
	return ''
}
function Get-Trophies([string]$userDir, [string]$serial) {
	$r = [pscustomobject]@{ Total = 0; Count = 0; Pct = 0; Items = @(); Found = $false }
	if (-not $serial) { return $r }
	$pts = @{ 'P' = 300; 'G' = 90; 'S' = 30; 'B' = 15 }; $all = 0; $got = 0
	$found = @(); $hdir = Get-HomeDir $userDir
	if ($hdir -and (Test-Path -LiteralPath $hdir)) { $found = @(Get-ChildItem -LiteralPath $hdir -Recurse -Depth 6 -Filter 'TROP.XML' -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match [regex]::Escape($serial) } | Select-Object -First 1) }
	if (-not $found.Count) { $td = Join-Path (Join-Path (Join-Path $userDir 'game_data') $serial) 'TrophyFiles'; if (Test-Path -LiteralPath $td) { $found = @(Get-ChildItem -LiteralPath $td -Recurse -Filter 'TROP.XML' -File -ErrorAction SilentlyContinue | Select-Object -First 1) } }
	foreach ($x in $found) {
		try {
			$doc = New-Object System.Xml.XmlDocument; $doc.Load($x.FullName); $r.Found = $true
			foreach ($t in @($doc.SelectNodes('//trophy'))) {
				$g = "$($t.GetAttribute('ttype'))"; if (-not $g) { continue }
				$un = ($t.GetAttribute('unlockstate') -eq 'true')
				$name = "$($t.SelectSingleNode('name').InnerText)"; $det = ''; $dn = $t.SelectSingleNode('detail'); if ($dn) { $det = $dn.InnerText }
				$r.Total++; $p = $pts[$g.Substring(0, 1)]; if (-not $p) { $p = 15 }; $all += $p
				if ($un) { $r.Count++; $got += $p }
				$r.Items += [pscustomobject]@{ Grade = $g.Substring(0, 1); Name = $name; Detail = $det; Unlocked = $un }
			}
		} catch { Write-Log "TROP.XML ilegible: $($x.FullName) ($_)" }
	}
	if ($all -gt 0) { $r.Pct = [int][math]::Floor(100 * $got / $all) }
	return $r
}

# ---------------------------------------------------------------- equipo (grafica, CPU, RAM)
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
function Get-PcInfo {
	$o = [pscustomobject]@{ Gpu = ''; VramGB = 0; Cpu = ''; Cores = 0; Threads = 0; RamGB = 0 }
	if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') { return $o }
	try {
		$vcs = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name })
		$reg = @(); try { $reg = @(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue) } catch { }
		$scored = foreach ($vc in $vcs) {
			$n = "$($vc.Name)"; $tier = 1
			if ($n -match 'Microsoft Basic|Remote Display|Virtual|Parsec|DisplayLink') { $tier = 0 }
			elseif ($n -match 'Radeon\(TM\)\s*(\d+\w*\s*)?Graphics|Radeon Vega|Vega \d+ Graphics|UHD|Iris|HD Graphics') { $tier = 1 }
			elseif ($n -match 'NVIDIA|GeForce|Quadro|RTX|GTX|Radeon\s*(RX|Pro|VII)|\bRX\s*\d|Intel.*Arc|\bArc\s*(A|B)\d') { $tier = 2 }
			[pscustomobject]@{ Vc = $vc; Tier = $tier; Vram = (Get-GpuVramBytes $vc $reg) }
		}
		$best = @($scored | Sort-Object -Property @{ Expression = 'Tier'; Descending = $true }, @{ Expression = 'Vram'; Descending = $true })[0]
		if ($best) { $o.Gpu = "$($best.Vc.Name)".Trim(); $o.VramGB = [math]::Round($best.Vram / 1GB) }
		$cpu = Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1
		$o.Cpu = ("$($cpu.Name)" -replace '\s+', ' ').Trim(); $o.Cores = [int]$cpu.NumberOfCores; $o.Threads = [int]$cpu.NumberOfLogicalProcessors
		$o.RamGB = [int][math]::Round((Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory / 1GB)
	} catch { Write-Log "Deteccion del equipo incompleta: $_" }
	return $o
}
# Tabla de graficas (idea del lanzador de Demon's Souls). ESTIMACION: no medido aun con Bloodborne en shadPS4, que
# depende mucho de la CPU. Columnas: patron, nombre del perfil, fps (0 = 30, 1 = 60), reescalado (0 nativo, 1 FSR 1),
# resolucion interna, explicacion.
$script:tierDb = @(
	@('RTX 50[89]0|RTX 40[89]0|RX 79[0-9]0|RX 90[67]0', (T 'Cazador veterano' 'Veteran hunter'), 1, 0, '1920x1080', (T '60 FPS con parche, imagen nativa (estimación)' '60 FPS with patch, native image (estimate)')),
	@('RTX 50[67]0|RTX 40[67]0|RTX 30[789]0|RX 6[89][05]0|RX 7[78]00', (T 'Cazador' 'Hunter'), 1, 1, '1920x1080', (T '60 FPS con parche, FSR 1 al monitor (estimación)' '60 FPS with patch, FSR 1 to the monitor (estimate)')),
	@('RTX 3060|RTX 4060|RTX 5060|RTX 20[678]0|RX 6[67]00|RX 7600|Arc.*(A7|B5)', (T 'Equilibrado' 'Balanced'), 1, 1, '1280x720', (T '60 FPS con parche puede bajar en zonas pesadas; si no va fluido, 30 FPS' '60 FPS with patch may drop in heavy areas; use 30 if not smooth')),
	@('GTX 16|GTX 10[678]0|RX 5[67]00|RX 5[78]0|RX 6600|Arc.*A[35]', (T 'Ligero' 'Light'), 0, 1, '1280x720', (T '30 FPS (como en PS4), 720p interna con FSR 1' '30 FPS (like PS4), 720p internal with FSR 1')),
	@('Radeon\(TM\)|Vega|Iris|UHD|HD Graphics|GTX 9|GTX 10[35]0', (T 'Mínimo' 'Minimum'), 0, 1, '1280x720', (T 'Gráfica integrada o antigua: 30 FPS, 720p; puede no ir fluido' 'Integrated or old GPU: 30 FPS, 720p; may not be smooth'))
)
function Get-Tier($pc) { foreach ($row in $script:tierDb) { if ($pc.Gpu -and $pc.Gpu -match $row[0]) { return $row } }; return $null }

# ---------------------------------------------------------------- listas de opciones
$script:resList = @('1280x720', '1600x900', '1920x1080', '2560x1440', '3840x2160')
$script:upscalers = @(
	@((T 'Nativo (sin reescalado)' 'Native (no upscaling)'), 'native', $true),
	@('AMD FSR 1 + RCAS (shadPS4)', 'fsr1', $true)
)
# (10-29) Como en Demon's Souls: cada reescalador con su calidad en el mismo desplegable. [3] = modo (BB_UPSCALE_MODE).
# (ui-1030) DLSS ya funciona en bb-port-medicion: va con sus calidades como los demas (indices 2-6; launcher.ini migrado, upver=2).
foreach ($un in @(@('DLSS', 'dlss'), @('FSR 4', 'fsr4'), @('FSR 3.1', 'fsr3'), @('XeSS', 'xess'))) {
	foreach ($um in @(@((T 'Nativo (AA)' 'Native (AA)'), 'native'), @((T 'Calidad' 'Quality'), 'quality'), @((T 'Equilibrado' 'Balanced'), 'balanced'), @((T 'Rendimiento' 'Performance'), 'performance'), @((T 'Ultra rendimiento' 'Ultra performance'), 'ultraperf'))) {
		$script:upscalers += ,@("$($un[0]) $($um[0])", $un[1], $true, $um[1])
	}
}
$script:upModes = @((T 'Calidad nativa (AA)' 'Native AA'), (T 'Calidad' 'Quality'), (T 'Equilibrado' 'Balanced'), (T 'Rendimiento' 'Performance'), (T 'Ultra rendimiento' 'Ultra performance'))
$script:upModeKeys = @('native', 'quality', 'balanced', 'performance', 'ultraperf')
# (bb-launcher-022) Frame generation como en Demon's Souls. [1] = BB_FG (el emulador lo hace en bb-fg-022; hasta entonces solo se pasa la variable).
$script:fgModes = @(
	@((T 'Desactivado' 'Off'), 'off'),
	@('AMD FSR 3 x2', 'amd-x2'),
	@((T 'AMD FSR 3 dinámico' 'AMD FSR 3 dynamic'), 'amd-dyn'),
	@('NVIDIA DLSS-G x2', 'dlss-x2'),
	@('NVIDIA DLSS-G x3', 'dlss-x3'),
	@('NVIDIA DLSS-G x4', 'dlss-x4'),
	@((T 'NVIDIA DLSS-G dinámico' 'NVIDIA DLSS-G dynamic'), 'dlss-dyn'),
	@('AMD FSR 3 x3', 'amd-x3'),  # (bb-fg-022) al final: bbfg 0-6 siguen siendo los mismos
	@('AMD FSR 3 x4', 'amd-x4')
)
$script:fgList = @($script:fgModes | ForEach-Object { $_[0] })
$script:quality = @(
	# clave ini, texto, opciones, variable BB_*, patron de parche de shadPS4 (si existe, la opcion es real)
	@('tex',       (T 'Calidad de texturas' 'Texture quality'), @((T 'Bajo' 'Low'), (T 'Normal' 'Normal'), (T 'Alto' 'High'), (T 'Ultra' 'Ultra')), 'BB_TEX_QUALITY', ''),
	@('q_shadows', (T 'Sombras' 'Shadows'), @((T 'Como PS4' 'PS4'), (T 'Altas' 'High'), (T 'Ultra' 'Ultra')), 'BB_SHADOWS', ''),
	@('q_ao',      (T 'Oclusión ambiental' 'Ambient occlusion'), @((T 'Como PS4' 'PS4'), (T 'Mejorada' 'Improved')), 'BB_AO', ''),
	@('q_fog',     (T 'Niebla volumétrica' 'Volumetric fog'), @((T 'Como PS4' 'PS4'), (T 'Alta' 'High')), 'BB_FOG', ''),
	@('q_aniso',   (T 'Filtrado anisotrópico' 'Anisotropic filtering'), @((T 'Como PS4' 'PS4'), 'x8', 'x16'), 'BB_ANISO', ''),
	@('q_ca',      (T 'Aberración cromática' 'Chromatic aberration'), @((T 'Activada' 'On'), (T 'Desactivada' 'Off')), 'BB_NO_CA', $script:patCA),
	@('q_blur',    (T 'Desenfoque de movimiento' 'Motion blur'), @((T 'Activado' 'On'), (T 'Desactivado' 'Off')), 'BB_NO_BLUR', $script:patBlur),
	@('q_grain',   (T 'Grano de película' 'Film grain'), @((T 'Activado' 'On'), (T 'Desactivado' 'Off')), 'BB_NO_GRAIN', $script:patGrain)
)
# ---------------------------------------------------------------- MODOS (como los del lanzador de Demon's Souls)
# Cada modo fija FPS reales, frame generation, resolucion base y reescalado. Personalizado no toca nada.
# Columnas: clave, texto, fps reales (BB_FPS_CAP), FG (BB_FG), en pantalla, res. interna, fps (0 = 30, 1 = parche 60),
# reescalado (indice de $upscalers), modo de reescalado (indice de $upModes), nitidez, calidad (sombras, AO, niebla, aniso).
# Hoy shadPS4 hace: resolucion (config.toml), 30/60 con el parche. FG y el tope de 45 son del emulador propio (proximamente).
$script:modes = @(
	@('ligero',      (T 'Ligero: 30 FPS x2 = 60, 1080p' 'Light: 30 FPS x2 = 60, 1080p'),            '30', 'amd-x2', 60,  '1920x1080', 0, 1, 0, 75, @(0, 0, 0, 0)),
	@('equilibrado', (T 'Equilibrado: 45 FPS x2 = 90, 1440p' 'Balanced: 45 FPS x2 = 90, 1440p'),     '45', 'amd-x2', 90,  '2560x1440', 1, 1, 1, 75, @(1, 0, 0, 1)),
	@('ultra',       (T 'Ultra: 60 FPS x2 = 120, 1440p' 'Ultra: 60 FPS x2 = 120, 1440p'),           '60', 'amd-x2', 120, '2560x1440', 1, 1, 1, 75, @(2, 1, 1, 2)),
	@('extremo',     (T 'Extremo: 60 FPS x2 = 120, 4K nativo' 'Extreme: 60 FPS x2 = 120, native 4K'), '60', 'amd-x2', 120, '3840x2160', 1, 0, 0, 50, @(2, 1, 1, 2)),
	@('personalizado', (T 'Personalizado (lo que elijas)' 'Custom (your choice)'), '', '', 0, '', -1, -1, -1, -1, $null)
)
# Requisitos de cada modo: PENDIENTE (los dara el dueno). Texto vacio = "pendiente".
$script:modeReq = @{ ligero = ''; equilibrado = ''; ultra = ''; extremo = ''; personalizado = '' }
# Gancho: opciones graficas DENTRO del juego por modo (llegaran de los logs del dueno). Formato previsto:
# ligero = [ordered]@{ 'Brillo' = '5'; ... }. Vacias: no se aplica nada. El emulador propio las leera de BB_GAME_SETTINGS.
$script:modeGameSettings = @{ ligero = [ordered]@{}; equilibrado = [ordered]@{}; ultra = [ordered]@{}; extremo = [ordered]@{}; personalizado = [ordered]@{} }
function Get-Mode { return $script:modes[[math]::Max(0, [math]::Min([int]$script:S.modo, $script:modes.Count - 1))] }
function Get-ModeGameSettings {
	$gs = $script:modeGameSettings[(Get-Mode)[0]]
	if (-not $gs -or $gs.Count -eq 0) { return '' }
	return (($gs.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ';')
}
function Get-ScreenRes {
	# (bb-fg-022) El modo fisico del monitor (EnumDisplaySettings), no los limites de WinForms: con la escala
	# de Windows al 150 % el lanzador (no consciente de DPI) veia 1707x960 en un monitor 2560x1440.
	try {
		if (-not ('BbDisp.Native' -as [type])) {
			Add-Type -Namespace BbDisp -Name Native -MemberDefinition '[System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, CharSet = System.Runtime.InteropServices.CharSet.Unicode)] public struct DEVMODE { [System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName; public short dmSpecVersion; public short dmDriverVersion; public short dmSize; public short dmDriverExtra; public int dmFields; public int dmPositionX; public int dmPositionY; public int dmDisplayOrientation; public int dmDisplayFixedOutput; public short dmColor; public short dmDuplex; public short dmYResolution; public short dmTTOption; public short dmCollate; [System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName; public short dmLogPixels; public int dmBitsPerPel; public int dmPelsWidth; public int dmPelsHeight; public int dmDisplayFlags; public int dmDisplayFrequency; }
[System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)] public static extern bool EnumDisplaySettings(string deviceName, int modeNum, ref DEVMODE devMode);' -ErrorAction Stop
		}
		$dm = New-Object BbDisp.Native+DEVMODE
		$dm.dmSize = [int16][System.Runtime.InteropServices.Marshal]::SizeOf($dm)
		if ([BbDisp.Native]::EnumDisplaySettings($null, -1, [ref]$dm) -and $dm.dmPelsWidth -gt 0) { return "$($dm.dmPelsWidth)x$($dm.dmPelsHeight)" }
	} catch { }
	try { $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; return "$($b.Width)x$($b.Height)" } catch { return '1920x1080' }
}
function Split-Res([string]$r) { if ($r -match '^(\d+)\s*x\s*(\d+)$') { return @([int]$Matches[1], [int]$Matches[2]) }; return @(1920, 1080) }

# ---------------------------------------------------------------- variables para el emulador propio (BB_*)
# (bb-launcher-022) Frame generation, Anti-Lag / Reflex y nitidez (0-1) para todos los reescaladores.
function Get-FgKey { $i = 0; [void][int]::TryParse("$($script:S.bbfg)", [ref]$i); return $script:fgModes[[math]::Max(0, [math]::Min($i, $script:fgModes.Count - 1))][1] }
function Get-SharpText { $v = 0; [void][int]::TryParse("$($script:S.sharp)", [ref]$v); return ([double][math]::Max(0, [math]::Min(100, $v)) / 100).ToString('0.00', [Globalization.CultureInfo]::InvariantCulture) }
# Nitidez EN VIVO: el emulador lee BB_UPSCALE_SHARPNESS al arrancar y puede releer este archivo (BB_UPSCALE_SHARPNESS_FILE)
# mientras juegas; el lanzador lo reescribe cada vez que cambias la Nitidez.
$script:sharpFile = Join-Path $root 'bb-live-sharpness.txt'
function Write-SharpLive { try { [IO.File]::WriteAllText($script:sharpFile, "BB_UPSCALE_SHARPNESS=$(Get-SharpText)`r`n", $utf8NoBom) } catch { } }
function Get-FgEnv {
	$e = [ordered]@{}
	$fg = Get-FgKey; $e['BB_FG'] = $fg
	$e['BB_HUD_SHARP'] = '1'  # (1.55) HUD canvas: HUD drawn at the output resolution, never through the upscaler (BB_HUD_SHARP=0 = old present-time composite)
	$e['BB_RES_EXACT'] = $(if (Test-ResExact) { '1' } else { '0' })  # (upin-158) el emulador pone la interna exacta del modo sobre el parche de la salida
	$e['BB_SCENE_UPSCALE'] = $(if (Get-SceneUp) { '1' } else { '0' })  # (0.93b) metodo DCport: escena y su postproceso a la interna (parche del juego), reescalado en la composicion antes del HUD  # (bb-native-071) el reescalador dentro de la cadena de render del juego  # (bb-fg-022) aqui: el arranque del emulador de reescalado solo pasa Get-FgEnv
	$dl = ($fg -like 'dlss*')
	$al = 0; [void][int]::TryParse("$($script:S.antilag)", [ref]$al); $rx = 0; [void][int]::TryParse("$($script:S.reflex)", [ref]$rx)
	$e['BB_ANTILAG'] = $(if (-not $dl -and $al -gt 0) { '1' } else { '0' })
	$e['BB_REFLEX'] = $(if ($dl) { "$([math]::Max(0, [math]::Min($rx, 2)))" } else { '0' })
	$e['BB_UPSCALE_SHARPNESS'] = Get-SharpText
	$e['BB_UPSCALE_SHARPNESS_FILE'] = $script:sharpFile
	return $e
}
function Get-SceneUp {
	# (bb-fg-022) Reescalado de escena: marcado y DLSS / FSR 4 / FSR 3.1 / XeSS en un modo reducido.
	$u = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]
	return ($script:S.sceneopt -eq '1' -and $u.Count -gt 3 -and @('dlss', 'fsr4', 'fsr3', 'xess') -contains $u[1] -and $u[3] -ne 'native')
}
function Get-BbEnv {
	$e = [ordered]@{}
	$up = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]
	$e['BB_UPSCALER'] = $up[1]
	$e['BB_UPSCALE_MODE'] = $(if ($up.Count -gt 3) { $up[3] } else { $script:upModeKeys[[math]::Min([int]$script:S.upmode, 4)] })
	$e['BB_UPSCALE_SHARPNESS'] = ([double]$script:S.sharp / 100).ToString('0.00', [Globalization.CultureInfo]::InvariantCulture)
	$m = Get-Mode
	$e['BB_MODE'] = $m[0]
	foreach ($kv in @((Get-FgEnv).GetEnumerator())) { $e[$kv.Key] = $kv.Value }  # (bb-launcher-022) BB_FG, BB_ANTILAG, BB_REFLEX, nitidez
	Write-SharpLive
	$e['BB_FPS_LIMIT'] = @('30', '60', 'unlocked')[[math]::Min([int]$script:S.fps, 2)]
	$e['BB_FPS_CAP'] = $(if ($m[2]) { $m[2] } else { @('30', '60', '0')[[math]::Min([int]$script:S.fps, 2)] })
	$gs = Get-ModeGameSettings; if ($gs) { $e['BB_GAME_SETTINGS'] = $gs }
	foreach ($q in $script:quality) { $e[$q[3]] = "$($script:S[$q[0]])" }
	$e['BB_INPUT'] = $(if ($script:S.input -eq '1') { 'kbm' } else { 'pad' })
	if ($script:S.ram16 -eq '1') { $e['BB_LOWRAM'] = '1' }
	if ($script:S.metricas -eq '1') { $e['BB_METRICS'] = '1'; $e['BB_LATENCY'] = '1'; $e['BB_METRICS_OUT'] = (Join-Path $root 'logs\bb-fps.log') }  # (bb-metrics-043) panel F8 dentro del juego, como Demon's Souls; BB_LATENCY: su fila Latencia
	  # el emulador propio escribira ahi [fps]/[lat]
	# (medir) Gancho de medicion como en Demon's Souls: con medir.on junto al lanzador, el emulador propio leera
	# ordenes de BB_LIVE_FILE y muestreara a BB_PROF_HZ.
	if (Test-Path -LiteralPath (Join-Path $root 'medir.on')) { $e['BB_LIVE_FILE'] = (Join-Path $root 'medir-cmd.txt'); $e['BB_PROF_HZ'] = '1000' }
	return $e
}

# ---------------------------------------------------------------- JUGAR
# (10-29) Resolucion interna segun la calidad del reescalado (como en Demon's Souls): la salida es la de la ventana
# (winres) y el juego renderiza a la del "Resolution Patch" 16:9 mas alta que cabe en salida / factor.
# (10-30) La salida nunca pasa de la pantalla real: 4K en un monitor de 1440p = 1440p (si no, no habia nada que reescalar).
function Limit-ToScreen([string]$res) { try { if (-not $script:realScreen) { $vc = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.CurrentVerticalResolution } | Sort-Object CurrentVerticalResolution -Descending)[0]; $script:realScreen = $(if ($vc) { "$($vc.CurrentHorizontalResolution)x$($vc.CurrentVerticalResolution)" } else { Get-ScreenRes }) }; $s = Split-Res $script:realScreen; $r = Split-Res $res; if ([int]$r[1] -gt [int]$s[1]) { return $script:realScreen } } catch { }; return $res }
function Get-UpRes([string]$mode) {
	$out = $script:S.winres; if (-not $out -or $out -eq 'auto') { $out = Get-ScreenRes }
	$o = Split-Res $out
	$f = switch ($mode) { 'native' { 1.0 } 'quality' { 1.5 } 'balanced' { 1.7 } 'performance' { 2.0 } 'ultraperf' { 3.0 } default { 1.5 } }
	$best = '640x360'
	foreach ($r in @('640x360', '960x540', '1280x720', '1440x810', '1600x900', '1920x1080', '2560x1440', '3840x2160')) { $wh = Split-Res $r; if ($wh[1] -le [math]::Ceiling($o[1] / $f) + 1) { $best = $r } }
	return $best
}
# (upin-158) Con DLSS / FSR 4 / FSR 3.1 / XeSS: UN solo parche de resolucion, el de la salida, y el emulador
# (BB_RES_EXACT=1) reescribe las medidas del juego a salida / factor del modo, alineadas a 4 como el emulador:
# Calidad en 2560x1440 = 1708x960 (antes el parche XML mas cercano, 1600x900 = x1.6, y un segundo parche).
# Factores como el emulador (bb_upscale.cpp QualityRatio): XeSS 1.7 / 2.0 / 2.3, los demas 1.5 / 1.724 / 2.0, ultra 3.0.
function Test-Temporal([string]$k) { return (@('dlss', 'fsr4', 'fsr3', 'xess') -contains $k) }
function Get-UpFactor([string]$mode, [string]$k) {
	$x = ($k -eq 'xess')
	switch ($mode) { 'native' { return 1.0 } 'quality' { if ($x) { return 1.7 } return 1.5 } 'balanced' { if ($x) { return 2.0 } return 1.724 } 'performance' { if ($x) { return 2.3 } return 2.0 } 'ultraperf' { return 3.0 } default { return 1.5 } }
}
function Get-UpInternal([string]$mode, [string]$k) {
	if (-not (Test-Temporal $k)) { return Get-UpRes $mode }
	$out = $script:S.winres; if (-not $out -or $out -eq 'auto') { $out = Get-ScreenRes }
	$o = Split-Res $out; $f = Get-UpFactor $mode $k
	if ($f -le 1.01) { return "$($o[0])x$($o[1])" }
	$w = [math]::Max(4, [int][math]::Round($o[0] / $f / 4, [MidpointRounding]::AwayFromZero) * 4)
	$h = [math]::Max(4, [int][math]::Round($o[1] / $f / 4, [MidpointRounding]::AwayFromZero) * 4)
	return "${w}x${h}"
}
function Test-ResExact {
	$u = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]
	return ((Test-Temporal $u[1]) -and -not (Get-SceneUp))
}
function Show-Msg([string]$text, [string]$title = 'Bloodborne', [string]$icon = 'Warning', [string]$buttons = 'OK') {
	Write-Log "Aviso: $($text -replace "`r?`n", ' | ')"
	if ($script:headless) { if ($buttons -eq 'YesNo') { return 'Yes' } return 'OK' }
	return [System.Windows.Forms.MessageBox]::Show($text, $title, $buttons, $icon)
}
function Start-Game {
	Write-Log '---- JUGAR ----'
	$emu = $script:S.emu
	if (-not $emu -or -not (Test-Path -LiteralPath $emu)) { Show-Msg (T 'Elige la carpeta del emulador (donde está shadPS4.exe).' 'Choose the emulator folder (where shadPS4.exe is).') | Out-Null; return $false }
	$g = Test-GameFolder $script:S.game
	if (-not $g.Ok) { Show-Msg ((T 'La carpeta del juego no vale:' 'The game folder is not valid:') + "`n`n" + ($g.Problems -join "`n")) | Out-Null; return $false }
	if (-not $g.Known) {
		if ((Show-Msg ((T "La serie '$($g.Serial)' no es una de Bloodborne conocidas. ¿Jugar igualmente?" "Title ID '$($g.Serial)' is not a known Bloodborne one. Play anyway?")) 'Bloodborne' 'Question' 'YesNo') -ne 'Yes') { return $false }
	}
	# Aviso de RAM libre (solo lee; no cierra nada).
	try {
		$os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
		$free = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
		if ($free -lt 6) {
			if ((Show-Msg (T "Solo quedan $free GB de RAM libres. Cierra programas (navegador, Discord...) si puedes.`n`nAceptar: jugar igualmente" "Only $free GB of RAM free. Close programs (browser, Discord...) if you can.`n`nOK: play anyway") 'RAM' 'Warning' 'OKCancel') -ne 'OK') { return $false }
		}
	} catch { }
	$userDir = Get-UserDir $emu
	$workDir = Split-Path -Parent $userDir
	if ((Split-Path -Leaf $userDir) -ne 'user') { $workDir = Split-Path -Parent $emu }  # %APPDATA%\shadPS4: se arranca desde la carpeta del exe
	Write-Log "Modo: $((Get-Mode)[0])"
	Write-Log "Emulador: $emu | user: $userDir | trabajo: $workDir | juego: $($g.Dir) [$($g.Serial) v$($g.Version)] actualizacion: $($g.Update) v$($g.UpdateVersion)"
	$eboot = Join-Path $g.Dir 'eboot.bin'
	Invoke-SoundFix $userDir  # como BB_Launcher, tambien en modo seguro
	$outRes = $script:S.winres; if (-not $outRes -or $outRes -eq 'auto') { $outRes = Get-ScreenRes }
	Set-ResolutionPatch $userDir $outRes  # (10-29) sin reescalado el juego va a la resolucion elegida (4K = 4K)  # (10-29) tambien en modo seguro  # (upin-158) el UNICO parche de la sesion: con reescalado el emulador baja a la interna (BB_RES_EXACT)
	# (10-29 b) La resolucion interna de shadPS4 (internalScreenWidth/Height) = la resolucion elegida: antes se quedaba en
	# 2560x1440 y el juego, aunque saliera a 4K, se veia a 1440. Solo FSR 1 (shadPS4) usa una interna menor.
	try {
		$upK = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)][1]
		if ($upK -ne 'fsr1') {
			$fr = Split-Res $outRes; $script:S.intres = $outRes
			$fw = @(Write-ShadConfig $userDir ([ordered]@{ winW = $fr[0]; winH = $fr[1]; intW = $fr[0]; intH = $fr[1]; fsr = $false }) 'both')
			Write-Log "Resolucion forzada: ventana e interna de shadPS4 = $outRes ($($fw -join ', '))"
			# (10-30) La config propia del juego (custom_configs\<serie>.json) manda sobre la general: tambien su ventana.
			try { foreach ($cc in @(Get-ChildItem -LiteralPath (Join-Path $userDir 'custom_configs') -Filter 'CUSA*.json' -File -ErrorAction SilentlyContinue)) { $t0 = [IO.File]::ReadAllText($cc.FullName); $t1 = Set-JsonValues $t0 ([ordered]@{ window_width = [int]$fr[0]; window_height = [int]$fr[1] }); if ($t1 -ne $t0) { [IO.File]::WriteAllText($cc.FullName, $t1, $utf8NoBom); Write-Log "Resolucion forzada en $($cc.Name): $outRes" } } } catch { Write-Log "custom_configs: $_" }
		}
	} catch { Write-Log "No se pudo forzar la resolucion: $_" }
	# MODO SEGURO (por defecto): no se modifica nada; solo se arranca shadPS4 con el juego, como BB_Launcher.
	if ($script:S.safemode -ne '0') {
		# Solo la pantalla que el usuario haya cambiado A MANO en esta sesion (no modos ni Optimizar), solo en la config
		# de la version elegida (config.json si existe, si no config.toml), solo esos valores en su sitio.
		$sv = [ordered]@{}
		if ($script:chg['winres'] -and $script:S.winres -and $script:S.winres -ne 'auto') { $wr = Split-Res $script:S.winres; $sv.winW = $wr[0]; $sv.winH = $wr[1] }
		if ($script:chg['intres']) { $ir = Split-Res $script:S.intres; $sv.intW = $ir[0]; $sv.intH = $ir[1] }
		if ($script:chg['fsmode']) { $sv.full = ($script:S.fsmode -ne '2'); $sv.fsMode = @('Fullscreen', 'Fullscreen (Borderless)', 'Windowed')[[math]::Min([int]$script:S.fsmode, 2)] }
		if ($script:chg['present']) { $sv.present = @('Fifo', 'Mailbox', 'Immediate')[[math]::Min([int]$script:S.present, 2)] }
		if ($sv.Count) {
			$only = Get-ConfigKind $emu
			try {
				$w = @(Write-ShadConfig $userDir $sv $only)
				Write-Log "Modo seguro: version $(Get-EmuVersion $emu) -> config $only";
				Write-Log "Modo seguro: pantalla escrita en $(if ($w.Count) { $w -join ', ' } else { "config.$only (sin cambios o no existe)" }) :: $(($sv.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')"
				foreach ($k in @('winres', 'intres', 'fsmode', 'present')) { $script:chg.Remove($k) }
			} catch { Write-Log "Modo seguro: no se pudo escribir la pantalla: $_" }
		}
		# (10-29) Reescalado FSR 3.1 / FSR 4 / XeSS: el mismo shadPS4 (0.17.1, revision del Pre-release) con los
		# reescaladores, en bb-port-medicion. Misma carpeta de trabajo, mismo user, mismo sound fix; solo cambia el exe y las BB_*.
		foreach ($k in @(Get-ChildItem Env: | Where-Object { $_.Name -like 'BB_*' } | ForEach-Object { $_.Name })) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
		$upKey = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)][1]
		$upExe = Join-Path $root 'bb-port-medicion\shadPS4.exe'
		# Si hay una version nueva en trozos (shadPS4.part0..), se une sola antes de jugar.
		$pdir = Split-Path -Parent $upExe
		if (Test-Path -LiteralPath (Join-Path $pdir 'shadPS4.part0')) {
			try {
				$parts = @(Get-ChildItem -LiteralPath $pdir -Filter 'shadPS4.part*' | Sort-Object { [int]($_.Name -replace '\D', '') })
				$tmp = Join-Path $pdir 'shadPS4.nuevo'
				$fs = [System.IO.File]::Create($tmp); foreach ($pp in $parts) { $b = [System.IO.File]::ReadAllBytes($pp.FullName); $fs.Write($b, 0, $b.Length) }; $fs.Close()
				if (Test-Path -LiteralPath $upExe) { Move-Item -LiteralPath $upExe -Destination (Join-Path $pdir 'shadPS4-anterior.exe') -Force }
				Move-Item -LiteralPath $tmp -Destination $upExe -Force; $parts | Remove-Item -Force
				Write-Log "Emulador de reescalado actualizado: MD5 $((Get-FileHash -LiteralPath $upExe -Algorithm MD5).Hash)"
			} catch { Write-Log "No se pudo unir la version nueva: $_" }
		}
		if (@('dlss', 'fsr3', 'fsr4', 'xess') -contains $upKey -and (Test-Path -LiteralPath $upExe)) {
			$emu = $upExe
			$env:BB_UPSCALER = $upKey
			if ($upKey -eq 'dlss') {
				# (ui-1030) Como Probar-DLSS.cmd: sin el guardia de la ultima prueba y con reintento.
				$guard = Join-Path $root 'bb_upscale_guard.txt'; if (Test-Path -LiteralPath $guard) { Remove-Item -LiteralPath $guard -Force -ErrorAction SilentlyContinue }
				$env:BB_UPSCALE_RETRY = '1'
			}
			$upE = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]
			$env:BB_UPSCALE_MODE = $(if ($upE.Count -gt 3) { $upE[3] } else { 'quality' })
			$env:BB_UPSCALE_SHARPNESS = ([double]$script:S.sharp / 100).ToString('0.00', [Globalization.CultureInfo]::InvariantCulture)
			$upRes = Get-UpInternal $env:BB_UPSCALE_MODE $upKey  # (upin-158) sin segundo parche: el de $outRes ya esta puesto
			if (Get-SceneUp) { Write-Log "Reescalado de escena (BB_SCENE_UPSCALE=1): parche $outRes, la escena 3D a la interna $upRes" }  # (bb-fg-022) intres = winres
			Write-Log "Reescalado: $upKey $($env:BB_UPSCALE_MODE) con $upExe (parche $outRes, interna $upRes, BB_RES_EXACT=$(if (Test-ResExact) { '1' } else { '0' }))"
		}
		# (bb-launcher-022) Frame generation + Anti-Lag/Reflex + nitidez (todos los reescaladores) como variables BB_*.
		$fge = Get-FgEnv
		if ($fge['BB_FG'] -ne 'off' -and $emu -ne $upExe -and (Test-Path -LiteralPath $upExe)) { $emu = $upExe; Write-Log "Frame generation $($fge['BB_FG']): con $upExe" }
		foreach ($k in $fge.Keys) { Set-Item "Env:$k" $fge[$k] }
		Write-SharpLive
		Write-Log "Variables: $(($fge.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')"
		$fpsLog = Join-Path $root 'logs\bb-fps.log'; New-Item -ItemType Directory -Force -Path (Split-Path -Parent $fpsLog) | Out-Null; $env:BB_METRICS_OUT = $fpsLog
		$eargs = @('-g', ('"' + $eboot + '"'))
		Write-Log "Modo seguro: sin cambios. Arranque: `"$emu`" $($eargs -join ' ') (trabajo: $workDir)"
		try { $p = Start-Process -FilePath $emu -ArgumentList $eargs -WorkingDirectory $workDir -PassThru } catch { Show-Msg ((T 'No se pudo arrancar shadPS4:' 'Could not start shadPS4:') + "`n$_") 'Bloodborne' 'Error' | Out-Null; return $false }
		Write-Log "shadPS4 arrancado (PID $($p.Id))"; Start-Metrics $p $userDir; return $true
	}
	# 1) Config de shadPS4 (lo que entiende hoy).
	$win = $script:S.winres; if ($win -eq 'auto' -or -not $win) { $win = Get-ScreenRes }
	$wr = Split-Res $win; $ir = Split-Res $script:S.intres
	$up = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]
	$sharp = [int]$script:S.sharp
	$vals = [ordered]@{
		winW = $wr[0]; winH = $wr[1]; intW = $ir[0]; intH = $ir[1]
		full = ($script:S.fsmode -ne '2'); fsMode = @('Fullscreen', 'Fullscreen (Borderless)', 'Windowed')[[math]::Min([int]$script:S.fsmode, 2)]
		present = @('Fifo', 'Mailbox', 'Immediate')[[math]::Min([int]$script:S.present, 2)]
		hdr = ($script:S.hdr -eq '1')
		fsr = ($up[1] -eq 'fsr1'); rcas = ($up[1] -eq 'fsr1' -and $sharp -gt 0); rcasAtt = [int](1000 - 10 * [math]::Max(0, [math]::Min(100, $sharp)))
		motion = ($script:S.motion -eq '1'); bgInput = ($script:S.bginput -eq '1')
	}
	# Solo las claves que el usuario ha cambiado en el lanzador.
	$want = @(); foreach ($sk in $script:cfgSettingKeys.Keys) { if ($script:chg[$sk] -or $script:chgAuto[$sk]) { $want += $script:cfgSettingKeys[$sk] } }
	if ($script:S.winres -eq 'auto' -or -not $script:S.winres) { $want = @($want | Where-Object { $_ -ne 'winW' -and $_ -ne 'winH' }) }  # nunca una resolucion sacada del monitor
	$all = $vals; $vals = [ordered]@{}; foreach ($k in $all.Keys) { if ($want -contains $k) { $vals[$k] = $all[$k] } }
	$written = @()
	try { if ($vals.Count) { $written = Write-ShadConfig $userDir $vals }; $script:chg = @{}; $script:chgAuto = @{} } catch { Write-Log "No se pudo escribir la config de shadPS4: $_"; Show-Msg ((T 'No se pudo escribir la configuración de shadPS4 (¿solo lectura?). Se juega con la que tenga.' 'Could not write the shadPS4 config (read-only?). Playing with its current one.') + "`n$_") | Out-Null }
	if (-not $vals.Count) { Write-Log 'Config de shadPS4: nada cambiado en el lanzador, no se escribe.' }
	if ($written.Count) { Write-Log "Config escrita: $($written -join ', ') :: $(($vals.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')" }
	elseif (-not (Test-Path -LiteralPath (Join-Path $userDir 'config.toml')) -and -not (Test-Path -LiteralPath (Join-Path $userDir 'config.json'))) {
		Write-Log 'Primera vez: shadPS4 aun no tiene config; se crea al arrancar. Los ajustes se aplican desde la siguiente partida.'
		if (-not $script:headless -and (Get-Setting 'firstrunwarned') -ne '1') { Show-Msg (T 'Primera vez con este emulador: shadPS4 crea su configuración al arrancar. Pantalla, reescalado y controles se aplicarán desde la próxima vez que pulses JUGAR.' 'First run with this emulator: shadPS4 creates its config on start. Display, upscaling and controls apply from the next time you press PLAY.') 'Bloodborne' 'Information' | Out-Null; Set-Setting 'firstrunwarned' '1' }
	}
	# 2) Parches: solo el "Resolution Patch" de la resolucion interna (Set-ResolutionPatch); el resto nunca se escribe.
	# 3) Variables BB_* para el emulador propio (shadPS4 de serie las ignora).
	$bb = Get-BbEnv
	foreach ($k in @(Get-ChildItem Env: | Where-Object { $_.Name -like 'BB_*' } | ForEach-Object { $_.Name })) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
	foreach ($k in $bb.Keys) { Set-Item "Env:$k" $bb[$k] }
	Write-Log "Variables: $(($bb.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')"
	# 4) Arranque. Rutas entre comillas (carpetas con espacios).
	$eargs = @('-g', ('"' + $eboot + '"'))
	if ($script:chg['fsmode']) { $eargs += @('-f', $(if ($all.full) { 'true' } else { 'false' })) }  # solo si el usuario eligio el modo de pantalla
	if ($script:S.showfps -eq '1') { $eargs += '--show-fps' }
	if ($script:S.useupdate -ne '1') { $eargs += '-i' }
	Write-Log "Arranque: `"$emu`" $($eargs -join ' ')"
	try { $p = Start-Process -FilePath $emu -ArgumentList $eargs -WorkingDirectory $workDir -PassThru }
	catch { Show-Msg ((T 'No se pudo arrancar shadPS4:' 'Could not start shadPS4:') + "`n$_") 'Bloodborne' 'Error' | Out-Null; return $false }
	Write-Log "shadPS4 arrancado (PID $($p.Id))"
	Start-Metrics $p $userDir
	return $true
}
# Metricas en pantalla (metricas-bb.ps1): solo lee, asi que funciona tambien en modo seguro. Se cierra sola cuando
# termina ese PID de shadPS4.
function Start-Metrics($p, [string]$userDir) {
	if ($p) { $script:gameProc = $p }
	if ($script:S.metricas -ne '1' -or -not $p) { return }
	# (bb-metrics-043) Fuera del modo seguro las metricas van dentro del juego (F8, el panel de Demon's Souls): sin ventana aparte.
	if ($script:S.safemode -eq '0') { Write-Log 'Metricas: panel dentro del juego (F8)'; return }
	$met = Join-Path $root 'metricas-bb.ps1'
	if (-not (Test-Path -LiteralPath $met)) { Write-Log 'Metricas: falta metricas-bb.ps1 junto al lanzador'; return }
	$a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', ('"' + $met + '"'), '-ProcessId', "$($p.Id)", '-UserDir', ('"' + $userDir + '"'), '-LauncherIni', ('"' + $iniFile + '"'), '-FpsFile', ('"' + (Join-Path $root 'logs\bb-fps.log') + '"'))
	if ($script:S.presentmon -eq '1') { $a += '-PresentMon' }
	try { Start-Process powershell -ArgumentList $a -WindowStyle Hidden | Out-Null; Write-Log "Metricas arrancadas (PID juego $($p.Id))" } catch { Write-Log "metricas-bb.ps1 no arranco: $_" }
}
function Get-Setting([string]$k) { if ($script:S.Contains($k)) { return $script:S[$k] } return '' }

# ---------------------------------------------------------------- Anadir a Steam (adaptado de Anadir-a-Steam.ps1)
# Acceso de Steam a este lanzador con -Play (arranca directo con los ultimos ajustes). Imagenes del fondo recortadas.
function Add-ToSteam {
	Add-Type -AssemblyName System.Drawing
	$name = 'Bloodborne'
	$ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
	$me = Join-Path $root $(if ($Lang -eq 'en') { 'Launcher-bloodborne-EN.ps1' } else { 'Launcher-bloodborne.ps1' })
	$bgWide = $null; $bgTall = $null
	foreach ($c in 'fondo-bloodborne2.png', 'fondo-bloodborne2.jpg', 'fondo-bloodborne.png', 'fondo-bloodborne.jpg') { $p = Join-Path $root $c; if (Test-Path -LiteralPath $p) { if (-not $bgWide) { $bgWide = $p }; if (-not $bgTall) { $bgTall = $p } } }
	if (-not $bgWide) { return (T 'Falta la imagen fondo-bloodborne2.png junto al lanzador.' 'fondo-bloodborne2.png is missing next to the launcher.') }
	$steam = (Get-ItemProperty 'HKCU:\Software\Valve\Steam' -ErrorAction SilentlyContinue).SteamPath
	if (-not $steam) { return (T 'No encuentro Steam instalado.' 'Steam is not installed.') }
	$steam = $steam -replace '/', '\'; $steamExe = Join-Path $steam 'steam.exe'
	$users = @(Get-ChildItem (Join-Path $steam 'userdata') -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^\d+$' -and $_.Name -ne '0' })
	if (-not $users) { return (T 'No hay usuarios de Steam (inicia sesión en Steam una vez).' 'No Steam users (log in to Steam once).') }
	$wasOpen = $false
	if (Get-Process steam -ErrorAction SilentlyContinue) {
		$wasOpen = $true; Write-Log 'Steam: cerrando para editar shortcuts.vdf'
		Start-Process $steamExe '-shutdown'
		for ($i = 0; $i -lt 30 -and (Get-Process steam -ErrorAction SilentlyContinue); $i++) { Start-Sleep 1; [System.Windows.Forms.Application]::DoEvents() }
		Get-Process steam -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2
	}
	$table = New-Object 'uint32[]' 256
	for ($n = 0; $n -lt 256; $n++) { $c = [uint32]$n; for ($k = 0; $k -lt 8; $k++) { if ($c -band 1) { $c = [uint32]([uint32]3988292384 -bxor ($c -shr 1)) } else { $c = [uint32]($c -shr 1) } }; $table[$n] = $c }
	$crc = { param([byte[]]$b) $c = [uint32]4294967295; foreach ($x in $b) { $c = [uint32]($table[($c -bxor $x) -band 0xFF] -bxor ($c -shr 8)) }; [uint32]($c -bxor [uint32]4294967295) }
	$exeQ = '"' + $ps + '"'
	$appid = [uint32]((& $crc ($utf8NoBom.GetBytes($exeQ + $name))) -bor [uint32]2147483648)
	$appidS = [BitConverter]::ToInt32([BitConverter]::GetBytes($appid), 0)
	$addStr = { param($l, [string]$k, [string]$v) $l.Add(1); $l.AddRange($utf8NoBom.GetBytes($k)); $l.Add(0); $l.AddRange($utf8NoBom.GetBytes($v)); $l.Add(0) }
	$addInt = { param($l, [string]$k, [int]$v) $l.Add(2); $l.AddRange($utf8NoBom.GetBytes($k)); $l.Add(0); $l.AddRange([BitConverter]::GetBytes($v)) }
	$fit = { param([string]$src, [string]$dst, [int]$w, [int]$h)
		$im = [System.Drawing.Image]::FromFile($src)
		$sc = [math]::Max($w / $im.Width, $h / $im.Height); $sw = [int]($w / $sc); $sh = [int]($h / $sc)
		$sx = [int](($im.Width - $sw) / 2); $sy = [int](($im.Height - $sh) / 2)
		$bm = New-Object System.Drawing.Bitmap $w, $h; $gr = [System.Drawing.Graphics]::FromImage($bm); $gr.InterpolationMode = 'HighQualityBicubic'
		$gr.DrawImage($im, (New-Object System.Drawing.Rectangle 0, 0, $w, $h), (New-Object System.Drawing.Rectangle $sx, $sy, $sw, $sh), [System.Drawing.GraphicsUnit]::Pixel)
		$gr.Dispose(); $im.Dispose(); $bm.Save($dst, [System.Drawing.Imaging.ImageFormat]::Jpeg); $bm.Dispose() }
	$launch = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $me + '" -Play'
	if ($Lang -eq 'en') { $launch += ' -Lang en' }
	$msgs = @()
	foreach ($u in $users) {
		try {
			$cfg = Join-Path $u.FullName 'config'; New-Item -ItemType Directory -Force $cfg | Out-Null
			$grid = Join-Path $cfg 'grid'; New-Item -ItemType Directory -Force $grid | Out-Null
			$vdf = Join-Path $cfg 'shortcuts.vdf'
			& $fit $bgWide (Join-Path $grid "$appid.jpg") 920 430
			& $fit $bgTall (Join-Path $grid "${appid}p.jpg") 600 900
			& $fit $bgWide (Join-Path $grid "${appid}_hero.jpg") 3840 1240
			$lg = New-Object System.Drawing.Bitmap 16, 16; $lg.Save((Join-Path $grid "${appid}_logo.png"), [System.Drawing.Imaging.ImageFormat]::Png); $lg.Dispose()
			$ico = Join-Path $grid "${appid}_icon.png"
			try { $src = $(if ($script:S.emu -and (Test-Path -LiteralPath $script:S.emu)) { $script:S.emu } else { $ps }); ([System.Drawing.Icon]::ExtractAssociatedIcon($src)).ToBitmap().Save($ico, [System.Drawing.Imaging.ImageFormat]::Png) } catch { $ico = '' }
			$old = $null; $idx = 0
			if (Test-Path -LiteralPath $vdf) {
				$old = [IO.File]::ReadAllBytes($vdf)
				Copy-Item -LiteralPath $vdf "$vdf.bak-bloodborne" -Force
				$lat = [Text.Encoding]::GetEncoding(28591).GetString($old)
				if ($lat.Contains([Text.Encoding]::GetEncoding(28591).GetString($utf8NoBom.GetBytes($me)))) { $msgs += (T "Usuario $($u.Name): ya estaba (imágenes actualizadas)." "User $($u.Name): already there (images updated)."); continue }
				$idx = ([regex]::Matches($lat, "(?i)\x02appid\x00")).Count
			}
			$e = New-Object 'System.Collections.Generic.List[byte]'
			$e.Add(0); $e.AddRange($utf8NoBom.GetBytes("$idx")); $e.Add(0)
			& $addInt $e 'appid' $appidS
			& $addStr $e 'AppName' $name
			& $addStr $e 'Exe' $exeQ
			& $addStr $e 'StartDir' ('"' + $root + '\"')
			& $addStr $e 'icon' $ico
			& $addStr $e 'ShortcutPath' ''
			& $addStr $e 'LaunchOptions' $launch
			& $addInt $e 'IsHidden' 0; & $addInt $e 'AllowDesktopConfig' 1; & $addInt $e 'AllowOverlay' 1; & $addInt $e 'OpenVR' 0
			& $addInt $e 'Devkit' 0; & $addStr $e 'DevkitGameID' ''; & $addInt $e 'DevkitOverrideAppID' 0; & $addInt $e 'LastPlayTime' 0
			& $addStr $e 'FlatpakAppID' ''
			$e.Add(0); $e.AddRange($utf8NoBom.GetBytes('tags')); $e.Add(0); $e.Add(8)
			$e.Add(8)
			$out = New-Object 'System.Collections.Generic.List[byte]'
			if ($old -and $old.Length -ge 2 -and $old[$old.Length - 1] -eq 8 -and $old[$old.Length - 2] -eq 8) { $out.AddRange([byte[]]$old[0..($old.Length - 3)]) }
			else { $out.Add(0); $out.AddRange($utf8NoBom.GetBytes('shortcuts')); $out.Add(0) }
			$out.AddRange($e); $out.Add(8); $out.Add(8)
			[IO.File]::WriteAllBytes($vdf, $out.ToArray())
			$msgs += (T "Usuario $($u.Name): Bloodborne añadido." "User $($u.Name): Bloodborne added.")
		} catch { $msgs += "Usuario $($u.Name): error $_"; Write-Log "Steam: $_" }
	}
	if ($wasOpen) { Start-Process $steamExe }
	Write-Log ("Steam: " + ($msgs -join ' | '))
	return (($msgs -join "`n") + "`n`n" + (T 'Abre Steam: aparece en tu biblioteca como "Bloodborne".' 'Open Steam: it shows in your library as "Bloodborne".'))
}

# ---------------------------------------------------------------- controles (user\input_config\*.ini de shadPS4)
# Formato: "salida = entrada" por linea; varias lineas para la misma salida son alternativas; "a, b" en una linea es
# una combinacion. Las lineas que el panel no conoce (comentarios, mouse_to_joystick, deadzones...) se dejan igual.
$script:ps4Outputs = @(
	@('cross', (T 'Cruz (X)' 'Cross (X)')), @('circle', (T 'Círculo' 'Circle')), @('square', (T 'Cuadrado' 'Square')), @('triangle', (T 'Triángulo' 'Triangle')),
	@('l1', 'L1'), @('r1', 'R1'), @('l2', 'L2'), @('r2', 'R2'), @('l3', 'L3'), @('r3', 'R3'),
	@('options', 'OPTIONS'), @('touchpad_center', (T 'Panel táctil' 'Touchpad')),
	@('pad_up', (T 'Cruceta arriba' 'D-pad up')), @('pad_down', (T 'Cruceta abajo' 'D-pad down')), @('pad_left', (T 'Cruceta izquierda' 'D-pad left')), @('pad_right', (T 'Cruceta derecha' 'D-pad right')),
	@('axis_left_x_minus', (T 'Stick izq. izquierda' 'Left stick left')), @('axis_left_x_plus', (T 'Stick izq. derecha' 'Left stick right')),
	@('axis_left_y_minus', (T 'Stick izq. arriba' 'Left stick up')), @('axis_left_y_plus', (T 'Stick izq. abajo' 'Left stick down')),
	@('axis_right_x_minus', (T 'Stick der. izquierda' 'Right stick left')), @('axis_right_x_plus', (T 'Stick der. derecha' 'Right stick right')),
	@('axis_right_y_minus', (T 'Stick der. arriba' 'Right stick up')), @('axis_right_y_plus', (T 'Stick der. abajo' 'Right stick down')),
	@('axis_left_x', (T 'Stick izq. eje X (mando)' 'Left stick X axis (pad)')), @('axis_left_y', (T 'Stick izq. eje Y (mando)' 'Left stick Y axis (pad)')),
	@('axis_right_x', (T 'Stick der. eje X (mando)' 'Right stick X axis (pad)')), @('axis_right_y', (T 'Stick der. eje Y (mando)' 'Right stick Y axis (pad)'))
)
$script:padInputs = @('', 'cross', 'circle', 'square', 'triangle', 'l1', 'r1', 'l2', 'r2', 'l3', 'r3', 'options', 'back', 'touchpad', 'pad_up', 'pad_down', 'pad_left', 'pad_right', 'axis_left_x', 'axis_left_y', 'axis_right_x', 'axis_right_y', 'lpaddle_high', 'lpaddle_low', 'rpaddle_high', 'rpaddle_low')
function Test-PadBinding([string]$v) { foreach ($p in ($v -split ',')) { if ($script:padInputs -notcontains $p.Trim().ToLower()) { return $false } }; return $true }
function Get-InputConfigFile([string]$userDir, [string]$serial) {
	$dir = Join-Path $userDir 'input_config'; $unified = $true
	$toml = Join-Path $userDir 'config.toml'; if (Test-Path -LiteralPath $toml) { if ([IO.File]::ReadAllText($toml) -match '(?m)^\s*useUnifiedInputConfig\s*=\s*false') { $unified = $false } }
	$f = Join-Path $dir 'default.ini'; if (-not $unified -and $serial -and (Test-Path -LiteralPath (Join-Path $dir "$serial.ini"))) { $f = Join-Path $dir "$serial.ini" }
	return $f
}
# Devuelve @{ salida = @{ Kb = @(...); Pad = @(...) } } para las salidas conocidas.
function Read-InputConfig([string]$file) {
	$map = @{}; foreach ($o in $script:ps4Outputs) { $map[$o[0]] = @{ Kb = @(); Pad = @() } }
	if (-not (Test-Path -LiteralPath $file)) { return $map }
	foreach ($l in [IO.File]::ReadAllLines($file)) {
		if ($l -match '^\s*#' -or $l -notmatch '^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$') { continue }
		$o = $Matches[1].ToLower(); $v = $Matches[2]; if (-not $map.ContainsKey($o) -or -not $v) { continue }
		if (Test-PadBinding $v) { $map[$o].Pad += $v } else { $map[$o].Kb += $v }
	}
	return $map
}
# Reescribe solo las lineas de las salidas conocidas (en el sitio de la primera); copia antes (.original y con fecha).
function Write-InputConfig([string]$file, $map) {
	$text = ''; if (Test-Path -LiteralPath $file) { $text = [IO.File]::ReadAllText($file) }
	$nl = "`r`n"; if ($text -and $text -notmatch "`r`n") { $nl = "`n" }
	$new = New-Object System.Collections.Generic.List[string]
	foreach ($o in $script:ps4Outputs) { foreach ($v in @($map[$o[0]].Kb) + @($map[$o[0]].Pad)) { if ("$v".Trim()) { $new.Add("$($o[0]) = $("$v".Trim())") } } }
	$out = New-Object System.Collections.Generic.List[string]; $placed = $false
	foreach ($l in ($text -split "`r?`n")) {
		if ($l -notmatch '^\s*#' -and $l -match '^\s*([A-Za-z0-9_]+)\s*=' -and $map.ContainsKey($Matches[1].ToLower())) { if (-not $placed) { $out.AddRange($new); $placed = $true }; continue }
		$out.Add($l)
	}
	while ($out.Count -gt 0 -and $out[$out.Count - 1] -eq '') { $out.RemoveAt($out.Count - 1) }
	if (-not $placed) { $out.AddRange($new) }
	$result = ($out -join $nl) + $nl
	if ($result -eq $text) { return $false }
	if (Test-Path -LiteralPath $file) {
		if (-not (Test-Path -LiteralPath ($file + '.original'))) { Copy-Item -LiteralPath $file ($file + '.original') }
		$bd = Join-Path (Split-Path -Parent $file) 'launcher-backups'; New-Item -ItemType Directory -Force -Path $bd | Out-Null
		Copy-Item -LiteralPath $file (Join-Path $bd ((Split-Path -Leaf $file) + '.' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
	} else { New-Item -ItemType Directory -Force -Path (Split-Path -Parent $file) | Out-Null }
	[IO.File]::WriteAllText($file, $result, $utf8NoBom)
	return $true
}
function Get-ShadKeyName([string]$k) {
	if ($k -match '^[A-Z]$') { return $k.ToLower() }
	if ($k -match '^D(\d)$') { return $Matches[1] }
	if ($k -match '^NumPad(\d)$') { return "kp$($Matches[1])" }
	if ($k -match '^F(\d+)$') { return "f$($Matches[1])" }
	$m = @{ Space = 'space'; Return = 'enter'; Escape = 'escape'; Tab = 'tab'; Back = 'backspace'; ShiftKey = 'lshift'; LShiftKey = 'lshift'; RShiftKey = 'rshift'; ControlKey = 'lctrl'; LControlKey = 'lctrl'; RControlKey = 'rctrl'; Menu = 'lalt'; LMenu = 'lalt'; RMenu = 'ralt'; Up = 'up'; Down = 'down'; Left = 'left'; Right = 'right'; Add = 'kpplus'; Subtract = 'kpminus'; Multiply = 'kpasterisk'; Divide = 'kpslash'; Decimal = 'kpperiod'; Insert = 'insert'; Delete = 'delete'; Home = 'home'; End = 'end'; PageUp = 'pgup'; Next = 'pgdown'; Oemcomma = 'comma'; OemPeriod = 'period'; OemMinus = 'minus'; Oemplus = 'plus'; Capital = 'capslock' }
	if ($m.ContainsKey($k)) { return $m[$k] }
	return $k.ToLower()
}

# ---------------------------------------------------------------- actualizar shadPS4 (solo releases oficiales)
# Se instala en una carpeta NUEVA junto a las otras (v.X.Y.Z); nunca se borra ni se sobrescribe ninguna existente.
$script:ghApi = 'https://api.github.com/repos/shadps4-emu/shadPS4/releases/latest'
function Test-OfficialUrl([string]$url, [string]$prefix) {
	try { $u = [uri]$url } catch { return $false }
	return ($u.Scheme -eq 'https' -and $u.Host -eq 'github.com' -and $u.AbsolutePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase))
}
function Get-ShadRelease {
	try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }
	$rel = Invoke-RestMethod -Uri $script:ghApi -Headers @{ 'User-Agent' = 'bb-launcher'; 'Accept' = 'application/vnd.github+json' } -UseBasicParsing
	$ver = ("$($rel.tag_name)" -replace '^[vV]\.?', '') -replace '[^0-9A-Za-z\.\-]', ''
	$zips = @($rel.assets | Where-Object { $_.name -match '(?i)win64' -and $_.name -match '(?i)\.zip$' -and (Test-OfficialUrl $_.browser_download_url '/shadps4-emu/shadPS4/releases/download/') })
	$asset = @(@($zips | Where-Object { $_.name -match '(?i)sdl' }) + @($zips | Where-Object { $_.name -notmatch '(?i)qt' }) + $zips)[0]
	return [pscustomobject]@{ Version = $ver; Tag = "$($rel.tag_name)"; Name = "$($rel.name)"; Date = "$($rel.published_at)"; Url = $(if (Test-OfficialUrl $rel.html_url '/shadps4-emu/') { "$($rel.html_url)" } else { '' }); Asset = $asset }
}
function Install-ShadRelease($rel, $progress) {
	$a = $rel.Asset
	if (-not $a -or -not (Test-OfficialUrl $a.browser_download_url '/shadps4-emu/shadPS4/releases/download/')) { throw (T 'La descarga no viene de github.com/shadps4-emu.' 'The download is not from github.com/shadps4-emu.') }
	$dest = Join-Path $root "v.$($rel.Version)"; $i = 2; while (Test-Path -LiteralPath $dest) { $dest = Join-Path $root "v.$($rel.Version)-$i"; $i++ }
	$tmp = Join-Path ([IO.Path]::GetTempPath()) ("shadps4-" + [guid]::NewGuid().ToString('N') + '.zip')
	Write-Log "Actualizar shadPS4: $($a.browser_download_url) -> $dest"
	$req = [Net.HttpWebRequest]::Create($a.browser_download_url); $req.UserAgent = 'bb-launcher'; $req.AllowAutoRedirect = $true
	$resp = $req.GetResponse()
	try {
		$h = $resp.ResponseUri.Host
		if ($h -ne 'github.com' -and $h -notmatch '\.githubusercontent\.com$') { throw "Servidor inesperado: $h" }
		$total = $resp.ContentLength; $in = $resp.GetResponseStream(); $out = [IO.File]::Create($tmp)
		try {
			$buf = New-Object byte[] 262144; $done = 0
			while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
				$out.Write($buf, 0, $n); $done += $n
				if ($progress -and $total -gt 0) { $progress.Value = [int][math]::Min(100, 100 * $done / $total); [System.Windows.Forms.Application]::DoEvents() }
			}
		} finally { $out.Close(); $in.Close() }
	} finally { $resp.Close() }
	try {
		if ($a.size -and (Get-Item -LiteralPath $tmp).Length -ne [long]$a.size) { throw (T 'Descarga incompleta.' 'Incomplete download.') }
		if ("$($a.digest)" -match '^sha256:([0-9a-fA-F]{64})$') {
			if ((Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash -ne $Matches[1].ToUpper()) { throw (T 'El SHA-256 no coincide con el de GitHub.' 'SHA-256 does not match GitHub.') }
		}
		Expand-Archive -LiteralPath $tmp -DestinationPath $dest
		$exe = @(Get-ChildItem -LiteralPath $dest -Filter 'shadPS4.exe' -Recurse -File -ErrorAction SilentlyContinue)[0]
		if (-not $exe) { Remove-Item -LiteralPath $dest -Recurse -Force -ErrorAction SilentlyContinue; throw (T 'El zip no trae shadPS4.exe.' 'The zip has no shadPS4.exe.') }
		Write-Log "shadPS4 $($rel.Version) instalado: $($exe.FullName)"
		return $exe.FullName
	} finally { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
}

# ---------------------------------------------------------------- Sound fix (SFX)
# Igual que el "Sound Fix" de BB_Launcher (comprobado en el PC del dueno: BB_Launcher deja 0x204E = 0x01 en cada
# partida y el juego lo vuelve a 0x00): antes de cada JUGAR, en SPRJ0005\userdata0010 de la carpeta de partidas
# efectiva, el byte 0x204E pasa a 0x01. Solo ese byte; solo archivos de 262144 bytes; nunca con shadPS4 abierto.
# Copia userdata0010.bak-sfx-original la primera vez (no se vuelve a tocar). Opcion soundfix (activada por defecto).
# Fuentes: github.com/LL4nc33/SoundCrashFix, nexusmods.com/bloodborne/mods/332.
function Find-SaveFiles([string]$userDir) {
	$h = Get-HomeDir $userDir; $roots = @($(if ($h) { $h } else { $userDir }))  # la carpeta de partidas efectiva
	$res = @()
	foreach ($r in $roots) { if ($r -and (Test-Path -LiteralPath $r)) { $res += @(Get-ChildItem -LiteralPath $r -Recurse -Depth 8 -Filter 'userdata0010' -File -ErrorAction SilentlyContinue | Where-Object { $_.DirectoryName -match 'SPRJ0005$' } | ForEach-Object { $_.FullName }) } }
	return @($res | Select-Object -Unique)
}
function Get-SfxByte([string]$file) {
	try { $fs = [IO.File]::Open($file, 'Open', 'Read', 'ReadWrite'); try { if ($fs.Length -le 0x204E) { return -1 }; [void]$fs.Seek(0x204E, 'Begin'); return $fs.ReadByte() } finally { $fs.Close() } } catch { return -2 }
}
function Set-SfxByte([string]$file) {
	if ((Get-Item -LiteralPath $file).Length -ne 262144) { Write-Log "Sound fix: $file no mide 262144 bytes, no se toca"; return -3 }
	$b = Get-SfxByte $file
	if ($b -ne 0) { return $b }
	$bak = $file + '.bak-sfx-original'; if (-not (Test-Path -LiteralPath $bak)) { Copy-Item -LiteralPath $file $bak }
	$fs = [IO.File]::Open($file, 'Open', 'ReadWrite', 'None'); try { [void]$fs.Seek(0x204E, 'Begin'); $fs.WriteByte(1) } finally { $fs.Close() }
	Write-Log "Sound fix: $file 0x204E 0x00 -> 0x01"
	return 256
}

function Invoke-SoundFix([string]$userDir) {
	if ($script:S.soundfix -eq '0') { return }
	if (Get-Process -Name 'shadPS4*' -ErrorAction SilentlyContinue) { Write-Log 'Sound fix: shadPS4 ya esta abierto, no se toca la partida'; return }
	$files = @(Find-SaveFiles $userDir)
	if (-not $files.Count) { Write-Log 'Sound fix: no hay partidas (SPRJ0005\userdata0010) todavia'; return }
	foreach ($f in $files) { try { $r = Set-SfxByte $f; if ($r -eq 1) { Write-Log "Sound fix: $f ya estaba en 0x01" } } catch { Write-Log "Sound fix: $f $_" } }
}

# ================================================================ inicio
Read-Settings
# (ui-1030) Indices de "upscaler" de antes (2 = DLSS proximamente, 3-17 = FSR 4 / FSR 3.1 / XeSS): DLSS entra con 5 calidades
# en 2-6, asi que los de 3 en adelante suben 4 y el 2 pasa a DLSS Calidad. Una sola vez (upver=2).
if ($script:S['upver'] -ne '2') {
	$uo = 0; [void][int]::TryParse("$($script:S.upscaler)", [ref]$uo)
	if ($uo -ge 3) { $uo += 4 } elseif ($uo -eq 2) { $uo = 3 }
	$script:S.upscaler = "$uo"; $script:S['upver'] = '2'
	if (Test-Path -LiteralPath $iniFile) { Save-Settings }
}
# (bb-launcher-022) framegen=1 de antes (FSR 3 x2) pasa a bbfg=1 (AMD x2). Una sola vez (fgver=1); framegen se sigue guardando.
if ($script:S['fgver'] -ne '1') {
	if ($script:S.framegen -eq '1' -and $script:S.bbfg -eq '0') { $script:S.bbfg = '1' }
	$script:S['fgver'] = '1'
	if (Test-Path -LiteralPath $iniFile) { Save-Settings }
}
Write-Log "Lanzador abierto (v1, idioma $Lang, PowerShell $($PSVersionTable.PSVersion))"
if ($NoUi) { return }
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
# Primera vez: se rellenan solos el emulador y el juego si estan bajo la carpeta del lanzador.
if (-not $script:S.emu -or -not (Test-Path -LiteralPath $script:S.emu)) { $f = @(Find-Emulators); if ($f.Count) { $script:S.emu = $f[0]; Save-Settings; Write-Log "Emulador encontrado: $($f[0])" } }
if (-not $script:S.game -or -not (Test-Path -LiteralPath $script:S.game)) { $f = @(Find-Games); if ($f.Count) { $script:S.game = $f[0]; Save-Settings; Write-Log "Juego encontrado: $($f[0])" } }

if ($Play) {
	$script:headless = $true
	if (-not (Start-Game)) {
		# Algo falta: se abre el lanzador normal para arreglarlo.
		$script:headless = $false
	} else { return }
}

# ================================================================ ventana
try {
	Add-Type -Namespace BbDpi -Name Native -MemberDefinition '[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(System.IntPtr v);' -ErrorAction Stop
	[void][BbDpi.Native]::SetProcessDpiAwarenessContext([System.IntPtr](-5))
} catch { }
[System.Windows.Forms.Application]::EnableVisualStyles()
try { [System.Windows.Forms.Application]::SetCompatibleTextRenderingDefault($false) } catch { }  # etiquetas con GDI (ClearType), como Demon's Souls
$script:pc = Get-PcInfo
Write-Log "Equipo: $($script:pc.Gpu) ($($script:pc.VramGB) GB) | $($script:pc.Cpu) $($script:pc.Cores)c/$($script:pc.Threads)h | $($script:pc.RamGB) GB RAM"

$scr = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$W = $scr.Width; $H = $scr.Height
$form = New-Object System.Windows.Forms.Form
$form.Opacity = 0
function Set-DoubleBuffer($c) { try { $c.GetType().GetProperty('DoubleBuffered', [System.Reflection.BindingFlags]'Instance,NonPublic').SetValue($c, $true, $null) } catch { } }
Set-DoubleBuffer $form
$form.SuspendLayout()
$form.Text = 'Bloodborne'
$form.FormBorderStyle = 'None'; $form.StartPosition = 'Manual'; $form.Bounds = $scr; $form.MaximizeBox = $false
$form.KeyPreview = $true
$form.Add_KeyDown({ param($s, $e) if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Escape) { $form.Close() } })
$form.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22)
$fondo = $null
foreach ($c in 'fondo-bloodborne2.jpg', 'fondo-bloodborne2.png', 'fondo-bloodborne.jpg', 'fondo-bloodborne.png') { $p = Join-Path $root $c; if (Test-Path -LiteralPath $p) { $fondo = $p; break } }
if ($fondo) {
	try {
		# Fondo "cover" (sin deformar), anclado arriba para no cortar el titulo.
		$img = [System.Drawing.Image]::FromFile($fondo)
		$scale = [math]::Max($W / $img.Width, $H / $img.Height)
		$sw = [int][math]::Round($W / $scale); $sh = [int][math]::Round($H / $scale)
		$src = New-Object System.Drawing.Rectangle([int](($img.Width - $sw) / 2), 0, $sw, $sh)
		$bmp = New-Object System.Drawing.Bitmap($W, $H)
		$gr = [System.Drawing.Graphics]::FromImage($bmp); $gr.InterpolationMode = 'HighQualityBicubic'
		$gr.DrawImage($img, (New-Object System.Drawing.Rectangle(0, 0, $W, $H)), $src, [System.Drawing.GraphicsUnit]::Pixel)
		$gr.Dispose(); $img.Dispose()
		$form.BackgroundImage = $bmp; $form.BackgroundImageLayout = 'None'
	} catch { Write-Log "Fondo ilegible: $_" }
} else { Write-Log 'Sin fondo (fondo-bloodborne2.png no esta junto al lanzador)' }
if ($script:S.emu -and (Test-Path -LiteralPath $script:S.emu)) { try { $form.Icon = [System.Drawing.Icon]::ExtractAssociatedIcon($script:S.emu) } catch { } }

# ---------------------------------------------------------------- tema (bb-launcher-022: igual que el lanzador de Demon's Souls)
# Panel derecho semitransparente de 880 con dos columnas de apartados (GroupBox, titulo dorado), JUGAR grande arriba a la
# izquierda y Precompilar / Actualizar / Settings / Salir a su lado; TU EQUIPO llena la columna derecha; Trofeos y la
# version abajo a la izquierda. Juego, emulador, parches y controles van en la ventana Settings (como en Demon's Souls).
$gold    = [System.Drawing.Color]::FromArgb(222, 196, 140)
$bone    = $gold
$ink     = [System.Drawing.Color]::FromArgb(236, 232, 224)
$dim     = [System.Drawing.Color]::FromArgb(165, 160, 150)
$blood   = [System.Drawing.Color]::FromArgb(176, 140, 70)
$field   = [System.Drawing.Color]::FromArgb(24, 26, 34)
$grpBack = [System.Drawing.Color]::FromArgb(18, 20, 26)
$panelBack = [System.Drawing.Color]::FromArgb(170, 8, 10, 14)
$buttonFont = New-Object System.Drawing.Font('Segoe UI Semibold', 15, [System.Drawing.FontStyle]::Bold)
$smallFont  = New-Object System.Drawing.Font('Segoe UI', 11)
$boldSmall  = New-Object System.Drawing.Font('Segoe UI Semibold', 11, [System.Drawing.FontStyle]::Bold)
$tinyFont   = New-Object System.Drawing.Font('Segoe UI', 9)
$fText  = $smallFont
$fSmall = New-Object System.Drawing.Font('Segoe UI', 10)
$fBtn   = New-Object System.Drawing.Font('Segoe UI Semibold', 11, [System.Drawing.FontStyle]::Bold)
$fBar   = New-Object System.Drawing.Font('Segoe UI Semibold', 12, [System.Drawing.FontStyle]::Bold)
$tips = New-Object System.Windows.Forms.ToolTip; $tips.AutoPopDelay = 20000; $tips.InitialDelay = 400; $tips.ReshowDelay = 100

function New-Label([string]$text, $x, $y, $w, $h = 24, $color = $ink, $font = $fText) {
	$l = New-Object System.Windows.Forms.Label
	$l.Text = $text; $l.Font = $font; $l.ForeColor = $color; $l.BackColor = [System.Drawing.Color]::Transparent
	$l.Location = New-Object System.Drawing.Point($x, $y); $l.Size = New-Object System.Drawing.Size($w, $h); $l.TextAlign = 'MiddleLeft'
	return $l
}
function New-Btn([string]$text, $x, $y, $w, $h, [bool]$primary = $false) {
	$b = New-Object System.Windows.Forms.Button
	$b.Text = $text; $b.Font = $fBtn; $b.FlatStyle = 'Flat'; $b.FlatAppearance.BorderSize = 1; $b.Cursor = [System.Windows.Forms.Cursors]::Hand
	$b.Location = New-Object System.Drawing.Point($x, $y); $b.Size = New-Object System.Drawing.Size($w, $h)
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
function New-Combo($items, $x, $y, $w, [string]$key, $enabledFlags = $null) {
	$c = New-Object System.Windows.Forms.ComboBox
	$c.DropDownStyle = 'DropDownList'; $c.FlatStyle = 'Flat'; $c.Font = $fText
	$c.BackColor = $field; $c.ForeColor = $ink
	$c.Location = New-Object System.Drawing.Point($x, $y); $c.Width = $w
	foreach ($i in $items) { [void]$c.Items.Add($i) }
	$v = 0; [void][int]::TryParse("$($script:S[$key])", [ref]$v)
	$c.SelectedIndex = [math]::Max(0, [math]::Min($v, $c.Items.Count - 1))
	$c.Tag = $key
	$c.Add_SelectedIndexChanged({ param($s, $e) Set-Setting $s.Tag $s.SelectedIndex })
	return $c
}
function New-Check([string]$text, $x, $y, $w, [string]$key) {
	$c = New-Object System.Windows.Forms.CheckBox
	$c.Text = $text; $c.Font = $fText; $c.ForeColor = $ink; $c.BackColor = [System.Drawing.Color]::Transparent
	$c.Location = New-Object System.Drawing.Point($x, $y); $c.Size = New-Object System.Drawing.Size($w, 26)
	$c.Checked = ($script:S[$key] -eq '1'); $c.Tag = $key
	$c.Add_CheckedChanged({ param($s, $e) Set-Setting $s.Tag $(if ($s.Checked) { '1' } else { '0' }) })
	return $c
}
# Desplegables dibujados a mano como en Demon's Souls (fondo oscuro, colores de marca, gris si no esta disponible).
function Draw-ComboItem($combo, $e) {
	if ($e.Index -lt 0) { return }
	$text = [string]$combo.Items[$e.Index]
	$selected = (([int]$e.State) -band [int][System.Windows.Forms.DrawItemState]::Selected) -ne 0
	$back = $(if ($selected -and $combo.Enabled) { [System.Drawing.Color]::FromArgb(60, 64, 80) } else { $field })
	$brush = New-Object System.Drawing.SolidBrush($back); $e.Graphics.FillRectangle($brush, $e.Bounds); $brush.Dispose()
	$color = $ink
	if (-not $combo.Enabled) { $color = $dim }
	elseif ($combo -eq $script:cmbUpRef -and $e.Index -lt $script:upscalers.Count -and -not $script:upscalers[$e.Index][2] -and $script:S.experimental -ne '1') { $color = $dim }
	elseif (($combo -eq $script:cmbUpRef -or $combo -eq $script:cmbFgRef) -and -not $selected) {
		if ($text -match 'DLSS|NVIDIA') { $color = [System.Drawing.Color]::FromArgb(118, 185, 0) } elseif ($text -match 'XeSS') { $color = [System.Drawing.Color]::FromArgb(70, 160, 255) } elseif ($text -match 'FSR|AMD') { $color = [System.Drawing.Color]::FromArgb(237, 70, 70) }
	}
	$r = New-Object System.Drawing.Rectangle(($e.Bounds.X + 8), $e.Bounds.Y, ($e.Bounds.Width - 8), $e.Bounds.Height)
	[System.Windows.Forms.TextRenderer]::DrawText($e.Graphics, $text, $combo.Font, $r, $color, ([System.Windows.Forms.TextFormatFlags]::VerticalCenter -bor [System.Windows.Forms.TextFormatFlags]::NoPrefix -bor [System.Windows.Forms.TextFormatFlags]::EndEllipsis))
}
function Set-ComboStyle($c) {
	$c.DrawMode = 'OwnerDrawFixed'; $c.ItemHeight = 24; $c.BackColor = $field; $c.ForeColor = $ink
	$c.Add_DrawItem({ param($s, $e) Draw-ComboItem $s $e })
	$c.Add_EnabledChanged({ param($s, $e) $s.Invalidate() })
}
function New-Group([string]$text, $parent, $x, $y, $w, $h) {
	$g = New-Object System.Windows.Forms.GroupBox
	$g.Text = $text; $g.Font = $boldSmall; $g.ForeColor = $gold
	$g.BackColor = [System.Drawing.Color]::Transparent
	$g.Location = New-Object System.Drawing.Point($x, $y); $g.Size = New-Object System.Drawing.Size($w, $h)
	$parent.Controls.Add($g)
	return $g
}
function Put($grp, $ctl, $x, $y, $w) {
	# Colores y letra propios (si no, los heredaria del grupo).
	$fc = $ctl.ForeColor; $fn = $ctl.Font
	$grp.Controls.Add($ctl)
	$ctl.ForeColor = $fc; $ctl.Font = $fn
	$ctl.Location = New-Object System.Drawing.Point($x, $y)
	if ($w) { $ctl.Width = $w }
	return $ctl
}
# Fila como en Demon's Souls: etiqueta a x=10 (112), desplegable a x=124 (282), paso de 32.
function Add-Row($grp, [string]$label, $ctl, $y, [string]$tip = '') {
	$lb = New-Label $label 10 ($y + 2) 112 24
	[void](Put $grp $lb 10 ($y + 2) 112)
	[void](Put $grp $ctl 124 $y 282)
	if ($tip) { $tips.SetToolTip($ctl, $tip); $tips.SetToolTip($lb, $tip) }
	return $lb
}
function New-DarkCombo($items, [string]$font = 'text') {
	$c = New-Object System.Windows.Forms.ComboBox
	$c.DropDownStyle = 'DropDownList'; $c.FlatStyle = 'Flat'; $c.BackColor = $field; $c.ForeColor = $ink; $c.Font = $(if ($font -eq 'small') { $fSmall } else { $fText })
	foreach ($i in $items) { [void]$c.Items.Add($i) }
	return $c
}

# ---------------------------------------------------------------- ventana Settings (juego, emulador, parches, controles, sistema)
# Apartados redondeados de antes, en una sola columna que se desplaza. Se abre con el boton Settings.
$fSec = New-Object System.Drawing.Font('Georgia', 12, [System.Drawing.FontStyle]::Bold)
$script:secY = 10
function New-RoundPath($x, $y, $w, $h, $r) {
	$p = New-Object System.Drawing.Drawing2D.GraphicsPath
	$p.AddArc($x, $y, $r, $r, 180, 90); $p.AddArc(($x + $w - $r), $y, $r, $r, 270, 90)
	$p.AddArc(($x + $w - $r), ($y + $h - $r), $r, $r, 0, 90); $p.AddArc($x, ($y + $h - $r), $r, $r, 90, 90); $p.CloseFigure()
	return $p
}
function New-Section([string]$text, $h) {
	$g = New-Object System.Windows.Forms.Panel
	$g.Text = $text; $g.Font = $fText; $g.ForeColor = $ink; $g.BackColor = $grpBack
	$g.Location = New-Object System.Drawing.Point(12, $script:secY); $g.Size = New-Object System.Drawing.Size($colW, $h)
	Set-DoubleBuffer $g
	$rp = New-RoundPath 0 0 $colW $h 16; $g.Region = New-Object System.Drawing.Region($rp); $rp.Dispose()
	$g.Add_Paint({
		param($s, $e)
		$gr = $e.Graphics; $gr.SmoothingMode = 'AntiAlias'
		$p = New-RoundPath 0 0 ($s.Width - 1) ($s.Height - 1) 16
		$pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(110, 222, 196, 140), 1); $gr.DrawPath($pen, $p); $pen.Dispose(); $p.Dispose()
		[System.Windows.Forms.TextRenderer]::DrawText($gr, $s.Text, $fSec, (New-Object System.Drawing.Point(12, 7)), $gold)
		$ln = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(70, 222, 196, 140), 1); $gr.DrawLine($ln, 12, 33, ($s.Width - 12), 33); $ln.Dispose()
	})
	$scroll.Controls.Add($g)
	$script:secY += $h + 12
	return $g
}
function Add-To($grp, $ctl) {
	$fc = $ctl.ForeColor; $fn = $ctl.Font; $grp.Controls.Add($ctl); $ctl.ForeColor = $fc; $ctl.Font = $fn
	# Fondo solido del apartado: texto ClearType nitido.
	if (($ctl -is [System.Windows.Forms.Label]) -or (($ctl -is [System.Windows.Forms.ButtonBase]) -and -not ($ctl -is [System.Windows.Forms.Button]))) { $ctl.BackColor = $grp.BackColor }
	return $ctl
}
$colW = 460
$fSet = New-Object System.Windows.Forms.Form
$fSet.Text = 'Settings - Bloodborne'; $fSet.StartPosition = 'CenterParent'; $fSet.FormBorderStyle = 'FixedDialog'; $fSet.MaximizeBox = $false; $fSet.MinimizeBox = $false; $fSet.ShowInTaskbar = $false
$fSet.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22); $fSet.ForeColor = $ink; $fSet.Font = $fText
$fSet.ClientSize = New-Object System.Drawing.Size(($colW + 24 + [System.Windows.Forms.SystemInformation]::VerticalScrollBarWidth), [math]::Min(980, $H - 60))
$fSet.KeyPreview = $true
$fSet.Add_KeyDown({ param($s, $e) if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Escape) { $e.Handled = $true; $fSet.Hide() } })
Set-DoubleBuffer $fSet
$setBar = New-Object System.Windows.Forms.Panel; $setBar.Dock = 'Bottom'; $setBar.Height = 60; $setBar.BackColor = $fSet.BackColor
$scroll = New-Object System.Windows.Forms.Panel; $scroll.Dock = 'Fill'; $scroll.AutoScroll = $true; $scroll.BackColor = $fSet.BackColor
Set-DoubleBuffer $scroll
$fSet.Controls.Add($scroll); $fSet.Controls.Add($setBar); $scroll.BringToFront()
$btnSetOk = New-Btn (T 'Cerrar' 'Close') 12 10 $colW 40 $true
$btnSetOk.Add_Click({ $fSet.Hide() }); $setBar.Controls.Add($btnSetOk)
$c1 = 12; $iw = $colW - 24
$lw = 180; $cx = 196; $cw = $colW - $cx - 12

# ---- Juego
$gGame = New-Section (T 'Juego' 'Game') 172
$cmbGame = Add-To $gGame (New-Object System.Windows.Forms.ComboBox)
$cmbGame.DropDownStyle = 'DropDownList'; $cmbGame.FlatStyle = 'Flat'; $cmbGame.BackColor = $field; $cmbGame.ForeColor = $ink; $cmbGame.Font = $fSmall
$cmbGame.Location = New-Object System.Drawing.Point(12, 42); $cmbGame.Width = $iw - 98; $cmbGame.DropDownWidth = 520
$btnGame = Add-To $gGame (New-Btn (T 'Examinar' 'Browse') ($iw - 78) 40 90 30); $btnGame.Font = $fSmall
$lbGame = Add-To $gGame (New-Label '' 12 76 $iw 62 $dim $fSmall); $lbGame.TextAlign = 'TopLeft'
$lbCompat = Add-To $gGame (New-Label '' 12 140 ($iw - 34) 24 $dim $fSmall)
$btnCompat = Add-To $gGame (New-Btn '?' ($iw - 16) 139 28 26); $btnCompat.Font = $fSmall
$tips.SetToolTip($btnCompat, (T 'Versiones compatibles' 'Compatible versions'))
$tips.SetToolTip($cmbGame, (T 'Carpeta del juego (la que tiene eboot.bin). La actualización 1.09 va en <serie>-patch / -UPDATE al lado.' 'Game folder (with eboot.bin). The 1.09 update goes in <serial>-patch / -UPDATE next to it.'))
$btnCompat.Add_Click({
	$l = ($script:bbSerials.GetEnumerator() | Sort-Object Name | ForEach-Object { "  $($_.Name)  -  $($_.Value)" }) -join "`n"
	[void][System.Windows.Forms.MessageBox]::Show((T "Versiones compatibles`n`n$l`n`nVersión del juego: 1.09 (01.09), la última. Con la base 1.00 sola se cierra al arrancar.`nLa actualización va en una carpeta <serie>-patch / -UPDATE junto al juego (o copiada encima).`n`n'Sin confirmar': una sola fuente pública. Japón y otras ediciones asiáticas: serie no comprobada; se pueden usar con aviso." "Compatible versions`n`n$l`n`nGame version: 1.09 (01.09), the latest. The 1.00 base alone crashes on start.`nThe update goes in a <serial>-patch / -UPDATE folder next to the game (or copied over it).`n`n'Unconfirmed': only one public source. Japan and other Asian editions: serial not verified; usable with a warning."), (T 'Versiones compatibles' 'Compatible versions'), 'OK', 'Information')
})

# ---- Emulador
$gEmu = New-Section (T 'Emulador (shadPS4)' 'Emulator (shadPS4)') 142
$cmbEmu = Add-To $gEmu (New-Object System.Windows.Forms.ComboBox)
$cmbEmu.DropDownStyle = 'DropDownList'; $cmbEmu.FlatStyle = 'Flat'; $cmbEmu.BackColor = $field; $cmbEmu.ForeColor = $ink; $cmbEmu.Font = $fSmall
$cmbEmu.Location = New-Object System.Drawing.Point(12, 42); $cmbEmu.Width = $iw - 98; $cmbEmu.DropDownWidth = 520
$btnEmu = Add-To $gEmu (New-Btn (T 'Examinar' 'Browse') ($iw - 78) 40 90 30); $btnEmu.Font = $fSmall
$lbEmu = Add-To $gEmu (New-Label '' 12 76 ($iw - 104) 46 $dim $fSmall); $lbEmu.TextAlign = 'TopLeft'
$lnkUser = Add-To $gEmu (New-Object System.Windows.Forms.LinkLabel); $lnkUser.Text = (T 'cambiar user' 'change user'); $lnkUser.Font = $fSmall
$lnkUser.LinkColor = $gold; $lnkUser.BackColor = $grpBack; $lnkUser.Location = New-Object System.Drawing.Point(($iw - 78), 76); $lnkUser.Size = New-Object System.Drawing.Size(90, 22)
$tips.SetToolTip($lnkUser, (T 'Carpeta "user" de shadPS4 (config, parches, trofeos). Se detecta sola; cámbiala si usas otra.' 'shadPS4 "user" folder (config, patches, trophies). Auto-detected; change it if you use another.'))
$lnkUpd = Add-To $gEmu (New-Object System.Windows.Forms.LinkLabel); $lnkUpd.Text = (T 'actualizar' 'update'); $lnkUpd.Font = $fSmall
$lnkUpd.LinkColor = $gold; $lnkUpd.BackColor = $grpBack; $lnkUpd.Location = New-Object System.Drawing.Point(($iw - 78), 100); $lnkUpd.Size = New-Object System.Drawing.Size(90, 22)
$updTip = (T 'Actualizar shadPS4: busca la última versión oficial (github.com/shadps4-emu) y la instala en una carpeta nueva. La actual no se toca: para volver, elígela en la lista de Settings.' 'Update shadPS4: finds the latest official release (github.com/shadps4-emu) and installs it to a new folder. The current one is kept: to go back, pick it in the Settings list.')
$tips.SetToolTip($lnkUpd, $updTip)

# ---- Parches
$gPat = New-Section (T 'Parches y mods' 'Patches and mods') 142
$chkUpd = Add-To $gPat (New-Check (T 'Usar actualización (-patch)' 'Use update (-patch)') 12 40 ($iw) 'useupdate')
$tips.SetToolTip($chkUpd, (T 'Desactivado: arranca sin la carpeta de actualización (-i). Solo fuera del modo seguro.' 'Off: starts without the update folder (-i). Only outside safe mode.'))
$btnPatches = Add-To $gPat (New-Btn (T 'Abrir parches en shadPS4' 'Open patches in shadPS4') 12 72 $iw 32); $btnPatches.Font = $fSmall
$tips.SetToolTip($btnPatches, (T 'El lanzador no cambia parches nunca (solo muestra su estado). Abre la carpeta user\patches; cámbialos con shadPS4 / BB_Launcher.' 'The launcher never changes patches (it only shows their state). Opens user\patches; change them with shadPS4 / BB_Launcher.'))
$btnPatches.Add_Click({ $d = Join-Path (Get-UserDir $script:S.emu) 'patches'; if (-not (Test-Path -LiteralPath $d)) { $d = Get-UserDir $script:S.emu }; if (Test-Path -LiteralPath $d) { Start-Process explorer.exe -ArgumentList ('"' + $d + '"') } })
$lbPatch = Add-To $gPat (New-Label '' 12 110 $iw 24 $dim $fSmall)

# ---- Controles
$gCtl = New-Section (T 'Controles' 'Controls') 176
$rbPad = Add-To $gCtl (New-Object System.Windows.Forms.RadioButton); $rbPad.Text = (T 'Mando' 'Gamepad'); $rbPad.Location = New-Object System.Drawing.Point(12, 40); $rbPad.Size = New-Object System.Drawing.Size(($lw - 4), 26); $rbPad.ForeColor = $ink; $rbPad.Font = $fText
$rbKbm = Add-To $gCtl (New-Object System.Windows.Forms.RadioButton); $rbKbm.Text = (T 'Teclado y ratón' 'Keyboard & mouse'); $rbKbm.Location = New-Object System.Drawing.Point($cx, 40); $rbKbm.Size = New-Object System.Drawing.Size($cw, 26); $rbKbm.ForeColor = $ink; $rbKbm.Font = $fText
if ($script:S.input -eq '1') { $rbKbm.Checked = $true } else { $rbPad.Checked = $true }
$rbKbm.Add_CheckedChanged({ Set-Setting 'input' $(if ($rbKbm.Checked) { '1' } else { '0' }) })
$tips.SetToolTip($rbKbm, (T 'shadPS4 acepta mando y teclado a la vez; las teclas se cambian en su configuración de controles.' 'shadPS4 takes pad and keyboard at once; keys are set in its input config.'))
$chkMotion = Add-To $gCtl (New-Check (T 'Sensor de movimiento' 'Motion sensor') 12 70 ($iw) 'motion')
$chkBg = Add-To $gCtl (New-Check (T 'Mando con la ventana en segundo plano' 'Pad input in background') 12 98 ($iw) 'bginput')
$btnInput = Add-To $gCtl (New-Btn (T 'Configurar controles (teclado / mando)' 'Configure controls (keyboard / pad)') 12 130 $iw 34)

# ---- Sistema
$gSys = New-Section (T 'Sistema' 'System') 132
$bw3 = [int](($iw - 16) / 3)
$btnLogs = Add-To $gSys (New-Btn (T 'Logs' 'Logs') 12 42 $bw3 32); $btnLogs.Font = $fSmall
$btnUserDir = Add-To $gSys (New-Btn (T 'Carpeta user' 'User folder') (20 + $bw3) 42 $bw3 32); $btnUserDir.Font = $fSmall
$btnSfx = Add-To $gSys (New-Btn 'Sound fix' (28 + 2 * $bw3) 42 $bw3 32); $btnSfx.Font = $fSmall
$tips.SetToolTip($btnSfx, (T 'Estado del sound fix en tus partidas (byte 0x204E de userdata0010). Solo muestra.' 'Sound fix state of your saves (byte 0x204E of userdata0010). View only.'))
$btnSfx.Add_Click({
	$files = @(Find-SaveFiles (Get-UserDir $script:S.emu))
	if (-not $files.Count) { [void][System.Windows.Forms.MessageBox]::Show((T 'No encuentro ninguna partida (SPRJ0005\userdata0010) en home_dir ni en la carpeta user.' 'No save found (SPRJ0005\userdata0010) in home_dir or the user folder.'), 'Sound fix', 'OK', 'Information'); return }
	$st = foreach ($f in $files) { $b = Get-SfxByte $f; "$f`n   0x204E = $(if ($b -ge 0) { '0x{0:X2}' -f $b } else { T 'ilegible' 'unreadable' })$(if ($b -eq 1) { T '  (ya aplicado)' '  (already applied)' })" }
	$msg = (T "Partidas encontradas:`n`n" "Saves found:`n`n") + ($st -join "`n`n")
	$msg += "`n`n" + $(if ($script:S.soundfix -eq '0') { T 'Sound fix automático: DESACTIVADO.' 'Automatic sound fix: OFF.' } else { T 'Sound fix automático: activado (se pone a 0x01 antes de cada JUGAR; solo archivos de 262144 bytes).' 'Automatic sound fix: on (set to 0x01 before each PLAY; 262144-byte files only).' })
	[void][System.Windows.Forms.MessageBox]::Show($msg, 'Sound fix', 'OK', 'Information')
})
$btnSteam = Add-To $gSys (New-Btn (T 'Añadir a Steam' 'Add to Steam') 12 84 $iw 34)
$tips.SetToolTip($btnSteam, (T 'Añade "Bloodborne" a tu biblioteca de Steam con imágenes (arranca con los últimos ajustes).' 'Adds "Bloodborne" to your Steam library with artwork (starts with the last settings).'))
$padEnd = New-Object System.Windows.Forms.Label; $padEnd.Location = New-Object System.Drawing.Point(0, $script:secY); $padEnd.Size = New-Object System.Drawing.Size(1, 4); $padEnd.BackColor = [System.Drawing.Color]::Transparent; $scroll.Controls.Add($padEnd)

# ---------------------------------------------------------------- panel derecho (como Demon's Souls)
$panelW = [math]::Min(880, $W)
$panel = New-Object System.Windows.Forms.Panel
$panel.Location = New-Object System.Drawing.Point(($W - $panelW), 0)
$panel.Size = New-Object System.Drawing.Size($panelW, $H)
$panel.AutoScroll = $true
$panel.Anchor = 'Top, Bottom, Right'
$panel.BackColor = $panelBack
$panel.SuspendLayout(); Set-DoubleBuffer $panel
$form.Controls.Add($panel)

# Fila de arriba: JUGAR grande; Precompilar / Actualizar / Settings / Salir a su lado.
$btnPlay = New-Btn (T 'JUGAR' 'PLAY') 10 12 300 74 $true
$btnPlay.Font = New-Object System.Drawing.Font('Segoe UI Black', 22, [System.Drawing.FontStyle]::Bold)
$tips.SetToolTip($btnPlay, (T 'Arranca Bloodborne con estos ajustes (Esc cierra el lanzador).' 'Starts Bloodborne with these settings (Esc closes the launcher).'))
$panel.Controls.Add($btnPlay)
$tbw = 131; $tbx = 320
$btnPre = New-Btn (T 'Precompilar' 'Precompile') $tbx 22 $tbw 54; $btnPre.Font = $fBar
$btnUpd = New-Btn (T 'Actualizar' 'Update') ($tbx + 139) 22 $tbw 54; $btnUpd.Font = $fBar
$btnSettings = New-Btn 'Settings' ($tbx + 278) 22 $tbw 54; $btnSettings.Font = $fBar
$btnExit = New-Btn (T 'Salir' 'Exit') ($tbx + 417) 22 $tbw 54; $btnExit.Font = $fBar
foreach ($b in @($btnPre, $btnUpd, $btnSettings, $btnExit)) { $panel.Controls.Add($b) }
$tips.SetToolTip($btnPre, (T 'shadPS4 compila los shaders mientras juegas y los guarda en su caché (los tirones solo pasan la primera vez). Aquí ves el estado de la caché.' 'shadPS4 compiles shaders while you play and keeps them in its cache (stutters only happen the first time). Shows the cache state.'))
$tips.SetToolTip($btnUpd, $updTip)
$tips.SetToolTip($btnSettings, (T 'Juego, emulador, parches y mods, controles, Steam y carpetas.' 'Game, emulator, patches and mods, controls, Steam and folders.'))
$tips.SetToolTip($btnExit, (T 'Cerrar el lanzador (Esc).' 'Close the launcher (Esc).'))
$btnExit.Add_Click({ $form.Close() })
$btnSettings.Add_Click({ [void]$fSet.ShowDialog($form); try { Update-Emu; Update-Game } catch { } })
$pbUpd = New-Object System.Windows.Forms.ProgressBar; $pbUpd.Location = New-Object System.Drawing.Point(320, 80); $pbUpd.Size = New-Object System.Drawing.Size(548, 6); $pbUpd.Maximum = 100; $pbUpd.Visible = $false
$panel.Controls.Add($pbUpd)

# ---- Imagen: Resolución, Reescalado, Preset, Nitidez
$gImg = New-Group (T 'Imagen' 'Image') $panel 10 100 420 222
$resItems = @((T "Monitor ($(Get-ScreenRes))" "Monitor ($(Get-ScreenRes))")) + $script:resList
$cmbWin = New-DarkCombo $resItems
$cmbWin.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:resList, $script:S.winres) + 1)
$cmbWin.Add_SelectedIndexChanged({ Set-Setting 'winres' $(if ($cmbWin.SelectedIndex -le 0) { 'auto' } else { $script:resList[$cmbWin.SelectedIndex - 1] }) })
$cmbWin.Add_SelectedIndexChanged({ if ($cmbUp) { for ($k = 0; $k -lt $script:upscalers.Count; $k++) { $u = $script:upscalers[$k]; if ($u.Count -gt 3) { $cmbUp.Items[$k] = "$($u[0])  ($(Get-UpInternal $u[3] $u[1]))" } } } })  # (10-29) la interna de cada calidad sigue a la resolucion
[void](Add-Row $gImg (T 'Resolución' 'Resolution') $cmbWin 26 (T 'Resolución de salida (ventana y pantalla completa exclusiva). Puede ser mayor que el monitor (DSR / VSR): el juego se dibuja a esa y la gráfica la reduce. Se escribe en la config de shadPS4 y en custom_configs\<serie>.json.' 'Output resolution (window and exclusive fullscreen). It can be above the monitor (DSR / VSR): the game renders at it and the GPU scales it down. Written to the shadPS4 config and custom_configs\<serial>.json.'))
# Res. interna y modo de reescalado aparte: ocultos (la interna la pone la calidad del reescalado), se conservan para los modos.
$cmbInt = New-DarkCombo $script:resList
$cmbInt.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:resList, $script:S.intres))
$cmbInt.Add_SelectedIndexChanged({ Set-Setting 'intres' $script:resList[$cmbInt.SelectedIndex] })
[void](Put $gImg $cmbInt 124 26 282); $cmbInt.Visible = $false
$cmbUp = New-DarkCombo @(); $cmbUp.DropDownWidth = 360; $cmbUp.MaxDropDownItems = 16
foreach ($u in $script:upscalers) { $lab = $u[0]; if ($u.Count -gt 3) { $lab = "$lab  ($(Get-UpInternal $u[3] $u[1]))" }; [void]$cmbUp.Items.Add($lab) }
$cmbUp.SelectedIndex = [math]::Max(0, [math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1))
[void](Add-Row $gImg (T 'Reescalado' 'Upscaling') $cmbUp 58 (T 'FSR 1 (shadPS4), DLSS, FSR 4, FSR 3.1 y XeSS con su calidad (entre paréntesis la resolución interna). DLSS necesita una gráfica NVIDIA RTX; FSR 4, una AMD RX 9000.' 'FSR 1 (shadPS4), DLSS, FSR 4, FSR 3.1 and XeSS with their quality (internal resolution in brackets). DLSS needs an NVIDIA RTX GPU; FSR 4, an AMD RX 9000.'))
$cmbUpMode = New-Combo $script:upModes 124 90 282 'upmode'
[void](Put $gImg $cmbUpMode 124 90 282); $cmbUpMode.Visible = $false
$cmbMode = New-Combo @($script:modes | ForEach-Object { $_[1] }) 0 0 282 'modo'; $cmbMode.DropDownWidth = 340
[void](Add-Row $gImg 'Preset' $cmbMode 90)
$cmbSharp = New-DarkCombo @(0..20 | ForEach-Object { "$($_ * 5) %" })
function Set-Sharp([int]$v) { $cmbSharp.SelectedIndex = [math]::Max(0, [math]::Min(20, [int][math]::Round([math]::Max(0, [math]::Min(100, $v)) / 5))) }
$sv0 = 75; [void][int]::TryParse("$($script:S.sharp)", [ref]$sv0); Set-Sharp $sv0
$cmbSharp.Add_SelectedIndexChanged({ Set-Setting 'sharp' ($cmbSharp.SelectedIndex * 5); Write-SharpLive })
[void](Add-Row $gImg (T 'Nitidez' 'Sharpness') $cmbSharp 122 (T 'Nitidez para TODOS los reescaladores (BB_UPSCALE_SHARPNESS 0-1): RCAS en FSR 1 / FSR 3.1 / FSR 4, paso de nitidez tras DLSS y XeSS. Se aplica en vivo: puedes cambiarla con el juego abierto.' 'Sharpness for EVERY upscaler (BB_UPSCALE_SHARPNESS 0-1): RCAS in FSR 1 / FSR 3.1 / FSR 4, a sharpening pass after DLSS and XeSS. Applied live: you can change it while the game runs.'))
$lbUp = New-Label '' 10 182 400 36 $dim $tinyFont; $lbUp.TextAlign = 'TopLeft'; [void](Put $gImg $lbUp 10 182 400)
# (bb-fg-022) Reescalado de escena experimental (BB_SCENE_UPSCALE): solo la escena 3D a la interna; el parche de
# resolucion se queda en la Resolucion de salida mientras esta marcado (el emulador reduce la escena el mismo).
$chkScene = Put $gImg (New-Check (T 'Reescalado de escena (experimental)' 'Scene upscale (experimental)') 0 0 396 'sceneopt') 12 154 396
$tips.SetToolTip($chkScene, (T 'BB_SCENE_UPSCALE=1: solo la escena 3D se dibuja a la resolución interna del reescalado y el reescalador actúa antes del postprocesado; postprocesado y HUD a la Resolución de salida. Solo con DLSS / FSR 4 / FSR 3.1 / XeSS en un modo reducido (no Nativo). Mientras está marcado, la resolución del juego (parche) se queda en la Resolución de salida. Experimental.' 'BB_SCENE_UPSCALE=1: only the 3D scene renders at the upscaler''s internal resolution and the upscaler runs before post-processing; post-processing and HUD at the output Resolution. Only with DLSS / FSR 4 / FSR 3.1 / XeSS in a reduced mode (not Native). While checked, the game resolution (patch) stays at the output Resolution. Experimental.'))

# ---- Frame generation + Anti-Lag / Reflex
$gFg = New-Group 'Frame generation' $panel 450 100 420 124
$cmbFg = New-Combo $script:fgList 0 0 282 'bbfg'
[void](Add-Row $gFg (T 'Modo' 'Mode') $cmbFg 26 (T 'Genera fotogramas intermedios (BB_FG): AMD FSR 3 x2 o dinámico; NVIDIA DLSS-G x2 / x3 / x4 o dinámico (RTX 40/50). Dinámico ajusta el multiplicador a tu monitor.' 'Generates in-between frames (BB_FG): AMD FSR 3 x2 or dynamic; NVIDIA DLSS-G x2 / x3 / x4 or dynamic (RTX 40/50). Dynamic matches the multiplier to your monitor.'))
$cmbAl = New-Combo @((T 'Desactivado' 'Off'), (T 'Activado' 'On')) 0 0 282 'antilag'
$cmbRx = New-Combo @((T 'Desactivado' 'Off'), (T 'Activado' 'On'), (T 'Activado + Boost' 'On + Boost')) 0 0 282 'reflex'
$lbLat = Add-Row $gFg 'Anti-Lag' $cmbAl 58 (T 'AMD Anti-Lag (BB_ANTILAG): menos retraso entre el mando y la imagen, sobre todo con frame generation.' 'AMD Anti-Lag (BB_ANTILAG): less delay between input and image, mostly with frame generation.')
[void](Put $gFg $cmbRx 124 58 282)
$tips.SetToolTip($cmbRx, (T 'NVIDIA Reflex (BB_REFLEX): menos retraso con DLSS-G. Boost mantiene la gráfica a más reloj.' 'NVIDIA Reflex (BB_REFLEX): less delay with DLSS-G. Boost keeps the GPU clocked higher.'))
$lbFg = New-Label '' 10 90 400 28 $dim $tinyFont; $lbFg.TextAlign = 'TopLeft'; [void](Put $gFg $lbFg 10 90 400)
function Update-LatBox {
	$fg = Get-FgKey; $dl = ($fg -like 'dlss*')
	$cmbAl.Visible = -not $dl; $cmbRx.Visible = $dl; $lbLat.Text = $(if ($dl) { 'Reflex' } else { 'Anti-Lag' })
	$lbFg.Text = $(if ($fg -eq 'off') { T 'Sin frame generation: los FPS reales del juego.' 'No frame generation: the real game FPS.' } elseif ($dl) { T "BB_FG=$fg  |  Reflex con DLSS-G (NVIDIA RTX 40/50)." "BB_FG=$fg  |  Reflex with DLSS-G (NVIDIA RTX 40/50)." } else { T "BB_FG=$fg  |  Anti-Lag con FSR 3 (cualquier gráfica)." "BB_FG=$fg  |  Anti-Lag with FSR 3 (any GPU)." })
}
$cmbFg.Add_SelectedIndexChanged({ Set-Setting 'framegen' $(if ($cmbFg.SelectedIndex -gt 0) { '1' } else { '0' }); Update-LatBox })

# ---- Pantalla y otros: Monitor, VSync, Límite FPS, HDR, Contador, Métricas
$gScr = New-Group (T 'Pantalla y otros' 'Display and more') $panel 450 232 420 158
$cmbFs = New-Combo @((T 'Pantalla completa (exclusiva)' 'Fullscreen (exclusive)'), (T 'Completa sin bordes' 'Borderless'), (T 'Ventana' 'Windowed')) 0 0 282 'fsmode'
[void](Add-Row $gScr 'Monitor' $cmbFs 26 (T 'Exclusiva: el monitor cambia a la Resolución elegida (útil con DSR / VSR). Sin bordes: ventana a pantalla completa.' 'Exclusive: the monitor switches to the chosen Resolution (useful with DSR / VSR). Borderless: fullscreen window.'))
$cmbPres = New-Combo @((T 'Activado (Fifo)' 'On (Fifo)'), (T 'Mailbox (por defecto)' 'Mailbox (default)'), (T 'Desactivado (Immediate)' 'Off (Immediate)')) 0 0 282 'present'
[void](Add-Row $gScr 'VSync' $cmbPres 58 (T 'Fifo: sin cortes, algo más de retraso. Mailbox: sin cortes y poco retraso. Immediate: el menor retraso, puede cortarse la imagen.' 'Fifo: no tearing, a bit more delay. Mailbox: no tearing, low delay. Immediate: lowest delay, may tear.'))
$cmbFps = New-Combo @('30 (PS4)', (T '60 (parche 60 FPS)' '60 (60 FPS patch)'), (T 'Sin límite (próximamente)' 'Uncapped (coming soon)')) 0 0 282 'fps'
[void](Add-Row $gScr (T 'Límite FPS' 'FPS limit') $cmbFps 90 (T 'Bloodborne va a 30. Para 60 hace falta el parche de 60 FPS de shadPS4 (user\patches).' 'Bloodborne runs at 30. 60 needs the shadPS4 60 FPS patch (user\patches).'))
$chkHdr = Put $gScr (New-Check 'HDR' 0 0 80 'hdr') 12 124 80
$chkFpsC = Put $gScr (New-Check (T 'Contador FPS' 'FPS counter') 0 0 150 'showfps') 96 124 150
$chkMet = Put $gScr (New-Check (T 'Métricas' 'Metrics') 0 0 150 'metricas') 252 124 150
$tips.SetToolTip($chkHdr, (T 'HDR de shadPS4 (necesita un monitor HDR con HDR activado en Windows).' 'shadPS4 HDR (needs an HDR monitor with HDR on in Windows).'))
$tips.SetToolTip($chkFpsC, (T 'Contador de FPS de shadPS4 (--show-fps, fuera del modo seguro).' 'shadPS4 FPS counter (--show-fps, outside safe mode).'))
$tips.SetToolTip($chkMet, (T "Ventana de métricas como en Demon's Souls (GPU, VRAM, CPU, RAM, FPS) con metricas-bb.ps1. Solo lee: funciona también en modo seguro. Se cierra al cerrar shadPS4." "Metrics window like Demon's Souls (GPU, VRAM, CPU, RAM, FPS) with metricas-bb.ps1. Read-only: works in safe mode too. Closes when shadPS4 exits."))

# ---- Calidad
$script:qCombos = @()
$gQ = New-Group (T 'Calidad' 'Quality') $panel 10 ($gImg.Bottom + 8) 420 (26 + 32 * $script:quality.Count + 30)
$qy = 26
foreach ($q in $script:quality) {
	$cb = New-Combo $q[2] 0 0 282 $q[0]
	[void](Add-Row $gQ $q[1] $cb $qy)
	$script:qCombos += , @($cb, $q)
	$qy += 32
}
$lbQ = New-Label (T 'Las filas en gris llegarán con el emulador propio (próximamente).' 'Greyed rows come with our own emulator (coming soon).') 10 ($qy + 2) 400 22 $dim $tinyFont; [void](Put $gQ $lbQ 10 ($qy + 2) 400)

# ---- Rendimiento / PC
$gPc = New-Group (T 'Rendimiento / PC' 'Performance / PC') $panel 10 ($gQ.Bottom + 8) 420 132
$chkSafe = Put $gPc (New-Check (T 'Modo seguro: solo cambia la resolución que elijas' 'Safe mode: only changes the resolution you pick') 0 0 396 'safemode') 12 24 396; $chkSafe.ForeColor = $gold
$tips.SetToolTip($chkSafe, (T 'Activado: JUGAR arranca shadPS4 con el juego (como BB_Launcher). Solo escribe la resolución, el modo de pantalla y el VSync que hayas cambiado tú en esta sesión, más las variables BB_* (reescalado, nitidez, frame generation). Nunca parches ni mods. Partidas: solo el byte del sound fix.' 'On: PLAY starts shadPS4 with the game (like BB_Launcher). Only writes the resolution, screen mode and VSync you changed this session, plus the BB_* variables (upscaling, sharpness, frame generation). Never patches or mods. Saves: only the sound-fix byte.'))
$chk16 = Put $gPc (New-Check (T '16 GB de RAM' '16 GB RAM') 0 0 190 'ram16') 12 54 190
$tips.SetToolTip($chk16, (T 'Hoy: aviso de RAM libre antes de jugar. Los límites de memoria para 16 GB llegarán con el emulador propio (BB_LOWRAM).' 'Today: free-RAM warning before playing. 16 GB memory limits come with our own emulator (BB_LOWRAM).'))
$chkSfx = Put $gPc (New-Check (T 'Sound fix' 'Sound fix') 0 0 196 'soundfix') 210 54 196
$tips.SetToolTip($chkSfx, (T 'Antes de cada JUGAR (también en modo seguro) pone a 1 el byte 0x204E de la partida (userdata0010), como BB_Launcher: sin él no suenan los golpes. Copia .bak-sfx-original la primera vez.' 'Before each PLAY (safe mode too) sets save byte 0x204E (userdata0010) to 1, like BB_Launcher: without it hit sounds are silent. Makes a .bak-sfx-original copy the first time.'))
$btnOpt = Put $gPc (New-Btn (T 'Optimizar según tu equipo' 'Optimize for your PC') 0 0 396 34) 12 86 396
$tips.SetToolTip($btnOpt, (T 'Elige resolución, reescalado y RAM según tu gráfica (te pregunta antes).' 'Picks resolution, upscaling and RAM for your GPU (asks first).'))

# ---- TU EQUIPO (tarjeta dibujada, llena la columna derecha)
function New-PcCard([int]$cw, [int]$ch, [double]$fs) {
	$bmp = New-Object System.Drawing.Bitmap($cw, $ch, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
	$gc = [System.Drawing.Graphics]::FromImage($bmp)
	$gc.SmoothingMode = 'AntiAlias'; $gc.TextRenderingHint = 'AntiAliasGridFit'; $gc.TextContrast = 0
	$path = New-RoundPath 0 0 ($cw - 1) ($ch - 1) 28
	$fill = New-Object System.Drawing.Drawing2D.LinearGradientBrush((New-Object System.Drawing.Point(0, 0)), (New-Object System.Drawing.Point($cw, $ch)), [System.Drawing.Color]::FromArgb(215, 30, 26, 20), [System.Drawing.Color]::FromArgb(215, 12, 13, 18))
	$gc.FillPath($fill, $path)
	$pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(150, 222, 196, 140), 1.5); $gc.DrawPath($pen, $path)
	$goldB = New-Object System.Drawing.SolidBrush($gold); $inkB = New-Object System.Drawing.SolidBrush($ink)
	$dimB = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(205, 195, 180))
	$fHead = New-Object System.Drawing.Font('Segoe UI Black', [float](16 * $fs), [System.Drawing.FontStyle]::Bold)
	$fCap = New-Object System.Drawing.Font('Segoe UI Semibold', [float](10.5 * $fs)); $fVal = New-Object System.Drawing.Font('Segoe UI Semibold', [float](13 * $fs), [System.Drawing.FontStyle]::Bold)
	$fTxt = New-Object System.Drawing.Font('Segoe UI', [float](12 * $fs))
	$tw = $cw - 32; $x = 16; $y = [int](12 * $fs)
	$gc.DrawString((T 'TU EQUIPO' 'YOUR PC'), $fHead, $goldB, [float]$x, [float]$y); $y += [int](40 * $fs)
	$put = {
		param($cap, $val, $font, $brush)
		$gc.DrawString($cap, $fCap, $dimB, [float]$x, [float]$y)
		$sz = $gc.MeasureString($val, $font, $tw)
		$gc.DrawString($val, $font, $brush, (New-Object System.Drawing.RectangleF($x, ($y + [int](21 * $fs)), $tw, ($sz.Height + 2))))
		return ($y + [int](23 * $fs) + $sz.Height + [int](7 * $fs))
	}
	$pc = $script:pc
	$gpu = $(if ($pc.Gpu) { $pc.Gpu + $(if ($pc.VramGB -gt 0) { "  ($($pc.VramGB) GB)" } else { '' }) } else { T 'no detectada' 'not detected' })
	if ($pc.Gpu -and $pc.Gpu -notmatch 'NVIDIA|GeForce|RTX') { $gpu += (T '  (sin DLSS: solo NVIDIA)' '  (no DLSS: NVIDIA only)') }
	$y = & $put (T 'Gráfica' 'Graphics card') $gpu $fVal $inkB
	$cpuTxt = "$(if ($pc.Cpu) { $pc.Cpu } else { '?' })$(if ($pc.Cores -gt 0) { "  ($($pc.Cores)c / $($pc.Threads)h)" })  |  $($pc.RamGB) GB RAM"
	$y = & $put (T 'Procesador y RAM' 'CPU and RAM') $cpuTxt $fTxt $inkB
	$t = Get-Tier $pc
	if ($t) {
		$y = & $put (T 'Perfil recomendado' 'Recommended profile') "$($t[1])" $fVal $goldB
		$y = & $put (T 'Esperado (estimación)' 'Expected (estimate)') "$($t[5])" $fTxt $inkB
	} else {
		$y = & $put (T 'Perfil recomendado' 'Recommended profile') (T 'Gráfica sin datos: empieza con 30 FPS y FSR 1, y mira las métricas.' 'No data for this GPU: start with 30 FPS and FSR 1, and watch the metrics.') $fTxt $inkB
	}
	$script:pcEndY = $y
	foreach ($o in $fill, $pen, $goldB, $inkB, $dimB, $fHead, $fCap, $fVal, $fTxt, $path) { $o.Dispose() }
	$gc.Dispose()
	return $bmp
}
$cardY = $gScr.Bottom + 10; $cardH = [math]::Max(240, $H - $cardY - 12)
$pcCard = New-Object System.Windows.Forms.PictureBox
$pcCard.BackColor = [System.Drawing.Color]::Transparent
$pcCard.Location = New-Object System.Drawing.Point(450, $cardY); $pcCard.Size = New-Object System.Drawing.Size(420, $cardH)
try {
	foreach ($fs in 1.0, 0.9, 0.8, 0.7, 0.6) {
		if ($pcCard.Image) { $pcCard.Image.Dispose() }
		$pcCard.Image = New-PcCard 420 $cardH $fs
		if ($script:pcEndY -le $cardH - 6) { break }
	}
} catch { Write-Log "Tarjeta TU EQUIPO: $_" }
$panel.Controls.Add($pcCard)

# ---- Trofeos (abajo a la izquierda, desplegable como en Demon's Souls)
$trophyW = 420; $trophyListH = 360
$btnTrophy = New-Btn (T 'Trofeos  v' 'Trophies  v') 24 ($H - 70) $trophyW 46
$btnTrophy.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 12, [System.Drawing.FontStyle]::Bold)
$btnTrophy.TextAlign = 'MiddleLeft'
try {
	$ti = New-Object System.Drawing.Bitmap 30, 30; $tg = [System.Drawing.Graphics]::FromImage($ti); $tg.SmoothingMode = 'AntiAlias'
	$tb = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(236, 196, 72)); $tp = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(236, 196, 72)), 2.4
	$tg.FillPie($tb, 7, -6, 16, 22, 0, 180); $tg.DrawArc($tp, 2, 2, 8, 9, 90, 180); $tg.DrawArc($tp, 20, 2, 8, 9, 270, 180)
	$tg.FillRectangle($tb, 13, 15, 4, 7); $tg.FillRectangle($tb, 9, 22, 12, 3); $tg.FillRectangle($tb, 7, 25, 16, 3)
	$tg.Dispose(); $btnTrophy.Image = $ti; $btnTrophy.ImageAlign = 'MiddleLeft'; $btnTrophy.TextImageRelation = 'ImageBeforeText'
} catch { }
$tips.SetToolTip($btnTrophy, (T 'Trofeos de shadPS4 para este juego (grado y nombre) y el % completado.' 'shadPS4 trophies for this game (grade and name) and the completion %.'))
$form.Controls.Add($btnTrophy)
$gTrophy = New-Object System.Windows.Forms.Panel
$gTrophy.Location = New-Object System.Drawing.Point(24, ($H - 76 - $trophyListH)); $gTrophy.Size = New-Object System.Drawing.Size($trophyW, $trophyListH)
$gTrophy.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22); $gTrophy.Visible = $false
$form.Controls.Add($gTrophy)
$lbTr = New-Label '' 10 6 ($trophyW - 20) 26 $gold $smallFont; $lbTr.BackColor = $gTrophy.BackColor; $gTrophy.Controls.Add($lbTr)
$pbTr = New-Object System.Windows.Forms.ProgressBar; $pbTr.Location = New-Object System.Drawing.Point(10, 34); $pbTr.Size = New-Object System.Drawing.Size(($trophyW - 20), 8); $pbTr.Maximum = 100
$gTrophy.Controls.Add($pbTr)
$lvTr = New-Object System.Windows.Forms.ListView
$lvTr.View = 'Details'; $lvTr.FullRowSelect = $true; $lvTr.HeaderStyle = 'None'; $lvTr.BorderStyle = 'None'; $lvTr.ShowItemToolTips = $true
$lvTr.Location = New-Object System.Drawing.Point(10, 50); $lvTr.Size = New-Object System.Drawing.Size(($trophyW - 20), ($trophyListH - 60))
$lvTr.BackColor = [System.Drawing.Color]::FromArgb(22, 25, 33); $lvTr.ForeColor = $ink; $lvTr.Font = $fSmall
[void]$lvTr.Columns.Add('', 90); [void]$lvTr.Columns.Add('', ($trophyW - 20 - 90 - 22))
$gTrophy.Controls.Add($lvTr)
$script:trSummary = ''
function Update-TrophyButton { $arrow = $(if ($gTrophy.Visible) { '^' } else { 'v' }); $btnTrophy.Text = (T 'Trofeos' 'Trophies') + $(if ($script:trSummary) { '   ' + $script:trSummary } else { '' }) + '  ' + $arrow }
$btnTrophy.Add_Click({ $gTrophy.Visible = -not $gTrophy.Visible; if ($gTrophy.Visible) { $gTrophy.BringToFront(); $btnTrophy.BringToFront() }; Update-TrophyButton })

# ---- version abajo a la izquierda
$lblVer = New-Object System.Windows.Forms.Label
$lblVer.AutoSize = $true; $lblVer.BackColor = [System.Drawing.Color]::Transparent; $lblVer.ForeColor = [System.Drawing.Color]::FromArgb(200, 194, 180)
$lblVer.Font = New-Object System.Drawing.Font('Segoe UI', 8)
$lblVer.Location = New-Object System.Drawing.Point(8, ($H - 20))
$form.Controls.Add($lblVer)
function Update-VerLabel {
	$ev = ''; try { if ($script:S.emu -and (Test-Path -LiteralPath $script:S.emu)) { $ev = Get-EmuLabel $script:S.emu } } catch { }
	$lblVer.Text = (T 'Lanzador Bloodborne v2' 'Bloodborne launcher v2') + $(if ($ev) { '  -  shadPS4: ' + $ev } else { '' }) + (T '   |   Esc: salir   |   launcher.log junto al lanzador' '   |   Esc: exit   |   launcher.log next to the launcher')
}

# Desplegables dibujados a mano (estilo Demon's Souls).
$script:cmbUpRef = $cmbUp; $script:cmbFgRef = $cmbFg
foreach ($c in @($cmbGame, $cmbEmu, $cmbWin, $cmbInt, $cmbFs, $cmbPres, $cmbFps, $cmbUp, $cmbUpMode, $cmbMode, $cmbFg, $cmbAl, $cmbRx, $cmbSharp)) { Set-ComboStyle $c }
foreach ($pair in $script:qCombos) { Set-ComboStyle $pair[0] }

function Update-Up {
	$i = $cmbUp.SelectedIndex; $u = $script:upscalers[$i]
	$exp = ($script:S.experimental -eq '1')
	if (-not $u[2] -and -not $exp) {
		# Opcion aun no disponible: vuelve a la ultima real.
		$cmbUp.SelectedIndex = [math]::Max(0, [math]::Min([int]$script:S.upscaler, 1)); return
	}
	Set-Setting 'upscaler' $i
	if ($cmbInt) { $cmbInt.Enabled = -not ($u.Count -gt 3) }  # (10-29) con DLSS / FSR 4 / FSR 3.1 / XeSS la interna la pone la calidad
	$cmbUpMode.Enabled = ($exp -and -not $u[2])
	$cmbSharp.Enabled = $true  # (bb-launcher-022) Nitidez desbloqueada para todos los reescaladores
	$lbUp.Text = $(if ($u[1] -eq 'fsr1') { T 'FSR 1 + RCAS de shadPS4. Nitidez = RCAS (y BB_UPSCALE_SHARPNESS).' 'shadPS4 FSR 1 + RCAS. Sharpness = RCAS (and BB_UPSCALE_SHARPNESS).' } elseif ($u[1] -eq 'native') { T 'Sin reescalado. La Nitidez se envía igual (BB_UPSCALE_SHARPNESS).' 'No upscaling. Sharpness is still sent (BB_UPSCALE_SHARPNESS).' } else { T 'Reescalado real antes del HUD. Nitidez: RCAS en FSR 3.1/FSR 4, paso de nitidez tras DLSS/XeSS (en vivo).' 'Real upscaling before the HUD. Sharpness: RCAS in FSR 3.1/FSR 4, a pass after DLSS/XeSS (live).' })
}
$cmbUp.Add_SelectedIndexChanged({ Update-Up })

# ---- logica de la ventana
$tierCol = @{ 'P' = [System.Drawing.Color]::FromArgb(170, 200, 235); 'G' = [System.Drawing.Color]::FromArgb(236, 196, 72); 'S' = [System.Drawing.Color]::FromArgb(200, 205, 212); 'B' = [System.Drawing.Color]::FromArgb(205, 127, 70) }
$gradeTxt = @{ 'P' = (T 'Platino' 'Platinum'); 'G' = (T 'Oro' 'Gold'); 'S' = (T 'Plata' 'Silver'); 'B' = (T 'Bronce' 'Bronze') }
function Update-Trophies($g) {
	$lvTr.BeginUpdate(); $lvTr.Items.Clear()
	$t = Get-Trophies (Get-UserDir $script:S.emu) $g.Serial
	if (-not $t.Found) { $lbTr.Text = (T 'Aún no hay trofeos (shadPS4 los crea al jugar).' 'No trophies yet (shadPS4 creates them when playing).'); $pbTr.Value = 0 }
	else {
		$lbTr.Text = "$($t.Count) / $($t.Total)   ($($t.Pct) %)"; $pbTr.Value = [math]::Max(0, [math]::Min(100, $t.Pct))
		foreach ($it in @($t.Items | Sort-Object @{ Expression = 'Unlocked'; Descending = $true })) {
			$li = New-Object System.Windows.Forms.ListViewItem($gradeTxt[$it.Grade]); [void]$li.SubItems.Add($it.Name)
			$li.UseItemStyleForSubItems = $false
			if ($it.Unlocked) { if ($tierCol[$it.Grade]) { $li.ForeColor = $tierCol[$it.Grade] }; $li.SubItems[1].ForeColor = $ink }
			else { $li.ForeColor = [System.Drawing.Color]::FromArgb(90, 86, 82); $li.SubItems[1].ForeColor = [System.Drawing.Color]::FromArgb(110, 106, 100) }
			$li.ToolTipText = $it.Detail
			[void]$lvTr.Items.Add($li)
		}
	}
	$lvTr.EndUpdate()
	$script:trSummary = $(if ($t.Found) { "$($t.Count) / $($t.Total)   ($($t.Pct) %)" } else { '' }); Update-TrophyButton
}
function Update-Quality($g) {
	$exp = ($script:S.experimental -eq '1')
	$patches = @(); if ($g.Serial) { $patches = @(Get-Patches (Get-UserDir $script:S.emu) $g.Serial $g.Version) }
	foreach ($pair in $script:qCombos) {
		$cb = $pair[0]; $q = $pair[1]
		$real = ($q[4] -and @($patches | Where-Object { $_.Name -match $q[4] }).Count -gt 0)
		$cb.Enabled = ((-not $q[4]) -and $exp)  # filas de parche: solo lectura
		$tips.SetToolTip($cb, $(if ($real) { T 'Estado del parche de shadPS4 (solo lectura). Se cambia en shadPS4 / BB_Launcher.' 'shadPS4 patch state (read-only). Change it in shadPS4 / BB_Launcher.' } else { T "Próximamente ($($q[3]) para el emulador propio)." "Coming soon ($($q[3]) for our own emulator)." }))
	}
	# Estado ACTUAL de los parches leido de los xml (si el usuario no ha cambiado la fila en esta sesion).
	$script:applying = $true
	try {
		foreach ($pair in $script:qCombos) {
			$q = $pair[1]; if (-not $q[4] -or $script:S["$($q[0])_set"] -eq '1') { continue }
			$m = @($patches | Where-Object { $_.Name -match $q[4] -and $_.VerOk }); if (-not $m.Count) { $m = @($patches | Where-Object { $_.Name -match $q[4] }) }
			if ($m.Count) { $pair[0].SelectedIndex = $(if (@($m | Where-Object { $_.Enabled }).Count) { 1 } else { 0 }) }
		}
		$m60 = @($patches | Where-Object { $_.Name -match $script:pat60 -and $_.VerOk }); if (-not $m60.Count) { $m60 = @($patches | Where-Object { $_.Name -match $script:pat60 }) }
		$unl = @($patches | Where-Object { $_.Enabled -and $_.Name -match $script:patUnlock })
		$script:fpsPatchActive = ($unl.Count -gt 0)
		$cmbFps.Items[2] = $(if ($script:fpsPatchActive) { T 'Sin límite (parche activo)' 'Uncapped (patch active)' } else { T 'Sin límite (próximamente)' 'Uncapped (coming soon)' })
		if ($script:fpsPatchActive) { $cmbFps.SelectedIndex = 2 } elseif ($m60.Count) { $cmbFps.SelectedIndex = $(if (@($m60 | Where-Object { $_.Enabled }).Count) { 1 } else { 0 }) }
		$cmbFps.Enabled = $false
		$tips.SetToolTip($cmbFps, (T "Estado leído de los parches activos (solo lectura)$(if ($unl.Count) { ': ' + (($unl | ForEach-Object { $_.Name } | Select-Object -Unique) -join ', ') })." "Read from the enabled patches (read-only)$(if ($unl.Count) { ': ' + (($unl | ForEach-Object { $_.Name } | Select-Object -Unique) -join ', ') })."))
	} finally { $script:applying = $false }
	$has60 = @($patches | Where-Object { $_.Name -match $script:pat60 }).Count -gt 0
	$lbPatch.Text = $(if (-not $g.Serial) { '' } elseif ($patches.Count -eq 0) { T 'Sin parches en user\patches (descárgalos en shadPS4).' 'No patches in user\patches (download them in shadPS4).' } elseif ($script:fpsPatchActive) { T "Límite FPS: Sin límite (parche activo)" "FPS limit: Uncapped (patch active)" } elseif ($has60) { T "Parches: $($patches.Count) (incluye 60 FPS)" "Patches: $($patches.Count) (60 FPS included)" } else { T "Parches: $($patches.Count) (sin 60 FPS)" "Patches: $($patches.Count) (no 60 FPS)" })
}
function Update-Game {
	$g = Test-GameFolder $script:S.game
	$lbCompat.Text = ''
	if (-not $script:S.game) { $lbGame.Text = (T 'Elige la carpeta con eboot.bin (CUSA00900, CUSA03173...).' 'Choose the folder with eboot.bin (CUSA00900, CUSA03173...).'); $lbGame.ForeColor = $dim }
	elseif (-not $g.Ok) { $lbGame.Text = ($g.Problems -join "`n"); $lbGame.ForeColor = [System.Drawing.Color]::FromArgb(230, 110, 100) }
	else {
		$t = "$($g.Serial)  $(if ($g.Region) { '- ' + $g.Region })  v$($g.Version)"
		if (-not $g.Known) { $t += "`n" + (T '(serie no reconocida como Bloodborne)' '(title ID not recognised as Bloodborne)') }
		if ($g.Update) { $t += "`n" + (T "Actualización: $(Split-Path -Leaf $g.Update)  v$($g.UpdateVersion)" "Update: $(Split-Path -Leaf $g.Update)  v$($g.UpdateVersion)") } else { $t += "`n" + (T 'Sin actualización (carpeta -patch / -UPDATE)' 'No update (-patch / -UPDATE folder)') }
		$lbGame.Text = $t; $lbGame.ForeColor = $dim
		$c = Get-Compat $g; $lbCompat.Text = $c.Text
		$lbCompat.ForeColor = $(if ($c.Ok) { [System.Drawing.Color]::FromArgb(110, 200, 120) } else { [System.Drawing.Color]::FromArgb(236, 196, 72) })
	}
	try { Update-Trophies $g } catch { Write-Log "Trofeos: $_" }
	try { Update-Quality $g } catch { Write-Log "Parches: $_" }
}
function Update-Emu {
	if ($script:S.emu -and (Test-Path -LiteralPath $script:S.emu)) {
		$u = Get-UserDir $script:S.emu
		$cfg = $(if (Test-Path -LiteralPath (Join-Path $u 'config.json')) { 'config.json' } elseif (Test-Path -LiteralPath (Join-Path $u 'config.toml')) { 'config.toml' } else { T 'sin config (primera vez)' 'no config yet (first run)' })
		$hd = Get-HomeDir $u; $lbEmu.Text = "user: $u`n$cfg$(if ($hd) { (T '  |  partidas: ' '  |  saves: ') + $hd })"; $lbEmu.ForeColor = $dim
	} else { $lbEmu.Text = (T 'Elige la carpeta donde está shadPS4.exe.' 'Choose the folder with shadPS4.exe.'); $lbEmu.ForeColor = [System.Drawing.Color]::FromArgb(230, 110, 100) }
	try { Update-VerLabel } catch { }
}
$script:filling = $false
function Fill-Lists {
	$script:filling = $true
	$cmbGame.Items.Clear(); $script:gameList = @(Find-Games); if ($script:S.game -and $script:gameList -notcontains $script:S.game) { $script:gameList = @($script:S.game) + $script:gameList }
	foreach ($g in $script:gameList) { $rel = $g; if ($g.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { $rel = $g.Substring($root.Length).TrimStart('\') }; [void]$cmbGame.Items.Add($rel) }
	if ($script:gameList.Count) { $cmbGame.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:gameList, $script:S.game)) }
	$cmbEmu.Items.Clear(); $script:emuList = @(Find-Emulators); if ($script:S.emu -and $script:emuList -notcontains $script:S.emu) { $script:emuList = @($script:S.emu) + $script:emuList }
	foreach ($e in $script:emuList) { [void]$cmbEmu.Items.Add((Get-EmuLabel $e)) }
	if ($script:emuList.Count) { $cmbEmu.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:emuList, $script:S.emu)) }
	$script:filling = $false
	Update-Emu; Update-Game
}
$cmbGame.Add_SelectedIndexChanged({ if ($script:filling) { return }; Set-Setting 'game' $script:gameList[$cmbGame.SelectedIndex]; Update-Game })
$cmbEmu.Add_SelectedIndexChanged({ if ($script:filling) { return }; Set-Setting 'emu' $script:emuList[$cmbEmu.SelectedIndex]; Update-Emu; Update-Game })
$btnGame.Add_Click({
	$d = New-Object System.Windows.Forms.FolderBrowserDialog
	$d.Description = (T 'Carpeta del juego (la que tiene eboot.bin y sce_sys)' 'Game folder (with eboot.bin and sce_sys)')
	if ($script:S.game -and (Test-Path -LiteralPath $script:S.game)) { $d.SelectedPath = $script:S.game } else { $d.SelectedPath = $root }
	if ($d.ShowDialog() -eq 'OK') {
		$p = $d.SelectedPath
		# Si eligen la carpeta de encima (games), se busca dentro.
		if (-not (Test-Path -LiteralPath (Join-Path $p 'eboot.bin'))) { $in = @(Get-ChildItem -LiteralPath $p -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch '-(UPDATE|patch)$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'eboot.bin')) }); if ($in.Count -ge 1) { $p = $in[0].FullName } }
		if ($p -match '-(UPDATE|patch)$') { $b = $p -replace '-(UPDATE|patch)$', ''; if (Test-Path -LiteralPath (Join-Path $b 'eboot.bin')) { $p = $b } }
		$g = Test-GameFolder $p
		if (-not $g.Ok) { [void][System.Windows.Forms.MessageBox]::Show(((T 'Esa carpeta no vale:' 'That folder is not valid:') + "`n`n" + ($g.Problems -join "`n")), 'Bloodborne', 'OK', 'Warning') }
		Set-Setting 'game' $p; Write-Log "Juego elegido: $p"; Fill-Lists
	}
})
$btnEmu.Add_Click({
	$d = New-Object System.Windows.Forms.OpenFileDialog
	$d.Filter = 'shadPS4.exe|shadPS4.exe;shadps4.exe|*.exe|*.exe'; $d.Title = (T 'Elige shadPS4.exe' 'Choose shadPS4.exe')
	if ($script:S.emu) { try { $d.InitialDirectory = Split-Path -Parent $script:S.emu } catch { } } else { $d.InitialDirectory = $root }
	if ($d.ShowDialog() -eq 'OK') { Set-Setting 'emu' $d.FileName; Write-Log "Emulador elegido: $($d.FileName)"; Fill-Lists }
})
function Show-ControlsPanel {
	$g = Test-GameFolder $script:S.game
	$f = Get-InputConfigFile (Get-UserDir $script:S.emu) $g.Serial
	if (-not (Test-Path -LiteralPath $f)) { [void][System.Windows.Forms.MessageBox]::Show((T "Aún no existe $f.`nshadPS4 lo crea la primera vez que arranca un juego." "$f does not exist yet.`nshadPS4 creates it the first time it runs a game."), 'Bloodborne', 'OK', 'Information'); return }
	$map = Read-InputConfig $f
	$d = New-Object System.Windows.Forms.Form
	$d.Text = (T 'Controles' 'Controls') + " - $(Split-Path -Leaf $f)"; $d.StartPosition = 'CenterParent'; $d.Size = New-Object System.Drawing.Size(780, 680)
	$d.BackColor = [System.Drawing.Color]::FromArgb(14, 12, 16); $d.ForeColor = $ink; $d.Font = $fSmall; $d.KeyPreview = $true; $d.FormBorderStyle = 'FixedDialog'; $d.MaximizeBox = $false
	$gv = New-Object System.Windows.Forms.DataGridView
	$gv.Location = New-Object System.Drawing.Point(12, 12); $gv.Size = New-Object System.Drawing.Size(740, 520)
	$gv.AllowUserToAddRows = $false; $gv.AllowUserToDeleteRows = $false; $gv.RowHeadersVisible = $false; $gv.SelectionMode = 'CellSelect'; $gv.MultiSelect = $false
	$gv.BackgroundColor = $d.BackColor; $gv.GridColor = [System.Drawing.Color]::FromArgb(60, 50, 50); $gv.BorderStyle = 'None'; $gv.EnableHeadersVisualStyles = $false
	$gv.DefaultCellStyle.BackColor = [System.Drawing.Color]::FromArgb(22, 19, 24); $gv.DefaultCellStyle.ForeColor = $ink; $gv.DefaultCellStyle.SelectionBackColor = $blood
	$gv.ColumnHeadersDefaultCellStyle.BackColor = [System.Drawing.Color]::FromArgb(30, 24, 28); $gv.ColumnHeadersDefaultCellStyle.ForeColor = $bone
	$c0 = New-Object System.Windows.Forms.DataGridViewTextBoxColumn; $c0.HeaderText = (T 'Botón de PS4' 'PS4 button'); $c0.ReadOnly = $true; $c0.Width = 200
	$c1x = New-Object System.Windows.Forms.DataGridViewTextBoxColumn; $c1x.HeaderText = (T 'Teclado / ratón  (alternativas con |, combinación con ,)' 'Keyboard / mouse  (alternatives with |, combo with ,)'); $c1x.Width = 340
	$c2x = New-Object System.Windows.Forms.DataGridViewComboBoxColumn; $c2x.HeaderText = (T 'Mando' 'Controller'); $c2x.Width = 180; $c2x.FlatStyle = 'Flat'
	foreach ($p in $script:padInputs) { [void]$c2x.Items.Add($p) }
	[void]$gv.Columns.AddRange(@($c0, $c1x, $c2x))
	foreach ($o in $script:ps4Outputs) {
		$e = $map[$o[0]]; $pad = "$(@($e.Pad)[0])"; if ($pad -and -not $c2x.Items.Contains($pad)) { [void]$c2x.Items.Add($pad) }
		$i = $gv.Rows.Add($o[1], (@($e.Kb) -join ' | '), $pad); $gv.Rows[$i].Tag = @(@($e.Pad) | Select-Object -Skip 1)  # otras del mando: se conservan
	}
	$d.Controls.Add($gv)
	$hint = New-Object System.Windows.Forms.Label; $hint.Location = New-Object System.Drawing.Point(12, 538); $hint.Size = New-Object System.Drawing.Size(740, 34); $hint.ForeColor = $dim
	$hint.Text = (T 'Elige una fila y pulsa "Asignar tecla", luego la tecla. También puedes escribir el nombre de shadPS4 (kp2, lshift, leftbutton...). Antes de guardar se hace copia (.original y input_config\launcher-backups).' 'Pick a row, press "Assign key" and then the key. You can also type the shadPS4 name (kp2, lshift, leftbutton...). A backup is made before saving (.original and input_config\launcher-backups).')
	$d.Controls.Add($hint)
	$script:capturing = $false
	function New-DBtn($t, $x, $w) { $b = New-Btn $t $x 580 $w 32; $b.Font = $fSmall; $d.Controls.Add($b); return $b }
	$bKey = New-DBtn (T 'Asignar tecla' 'Assign key') 12 120
	$bL = New-DBtn (T 'Clic izq.' 'Left click') 138 90; $bR = New-DBtn (T 'Clic der.' 'Right click') 234 90; $bM = New-DBtn (T 'Clic central' 'Middle click') 330 100
	$bClr = New-DBtn (T 'Borrar' 'Clear') 436 80; $bNote = New-DBtn (T 'Abrir .ini' 'Open .ini') 522 80
	$bSave = New-DBtn (T 'Guardar' 'Save') 608 70; $bCancel = New-DBtn (T 'Cancelar' 'Cancel') 682 70
	$setKb = { param($v) $r = $gv.CurrentCell.RowIndex; if ($r -ge 0) { $gv.Rows[$r].Cells[1].Value = $v } }
	$bKey.Add_Click({ $script:capturing = $true; $bKey.Text = (T 'Pulsa una tecla...' 'Press a key...') })
	$bL.Add_Click({ & $setKb 'leftbutton' }); $bR.Add_Click({ & $setKb 'rightbutton' }); $bM.Add_Click({ & $setKb 'middlebutton' })
	$bClr.Add_Click({ $r = $gv.CurrentCell.RowIndex; if ($r -ge 0) { if ($gv.CurrentCell.ColumnIndex -eq 2) { $gv.Rows[$r].Cells[2].Value = '' } else { $gv.Rows[$r].Cells[1].Value = '' } } })
	$bNote.Add_Click({ Start-Process notepad.exe -ArgumentList ('"' + $f + '"') })
	$d.Add_KeyDown({ param($s, $e)
		if (-not $script:capturing) { return }
		$e.Handled = $true; $e.SuppressKeyPress = $true; $script:capturing = $false; $bKey.Text = (T 'Asignar tecla' 'Assign key')
		& $setKb (Get-ShadKeyName "$($e.KeyCode)")
	})
	$bCancel.Add_Click({ $d.Close() })
	$bSave.Add_Click({
		$gv.EndEdit()
		$new = @{}
		for ($i = 0; $i -lt $script:ps4Outputs.Count; $i++) {
			$row = $gv.Rows[$i]
			$kb = @("$($row.Cells[1].Value)" -split '\|' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
			$pad = @(); if ("$($row.Cells[2].Value)") { $pad += "$($row.Cells[2].Value)" }; $pad += @($row.Tag)
			$new[$script:ps4Outputs[$i][0]] = @{ Kb = $kb; Pad = @($pad | Where-Object { $_ }) }
		}
		try { if (Write-InputConfig $f $new) { Write-Log "Controles guardados: $f" }; $d.Close() }
		catch { [void][System.Windows.Forms.MessageBox]::Show((T "No se pudo guardar:`n$_" "Could not save:`n$_"), 'Bloodborne', 'OK', 'Error') }
	})
	[void]$d.ShowDialog($fSet)
}
$btnInput.Add_Click({ try { Show-ControlsPanel } catch { Write-Log "Controles: $_"; [void][System.Windows.Forms.MessageBox]::Show("$_", 'Bloodborne', 'OK', 'Error') } })
$btnLogs.Add_Click({
	$d = Join-Path (Get-UserDir $script:S.emu) 'log'
	if (-not (Test-Path -LiteralPath $d)) { $d = $root }
	Start-Process explorer.exe -ArgumentList ('"' + $d + '"')
})
$btnUserDir.Add_Click({ $d = Get-UserDir $script:S.emu; if (Test-Path -LiteralPath $d) { Start-Process explorer.exe -ArgumentList ('"' + $d + '"') } })
$btnOpt.Add_Click({
	$t = Get-Tier $script:pc
	if (-not $t) { [void][System.Windows.Forms.MessageBox]::Show((T "No tengo datos de tu gráfica ($($script:pc.Gpu)). Empieza con 30 FPS y FSR 1." "No data for your GPU ($($script:pc.Gpu)). Start with 30 FPS and FSR 1."), 'Bloodborne', 'OK', 'Information'); return }
	$fps = $t[2]; $note = ''
	if ($script:pc.Cores -gt 0 -and $script:pc.Cores -lt 6 -and $fps -eq 1) { $fps = 0; $note = "`n" + (T "Tu CPU tiene $($script:pc.Cores) núcleos: shadPS4 depende mucho de la CPU, mejor 30 FPS." "Your CPU has $($script:pc.Cores) cores: shadPS4 is CPU-heavy, 30 FPS is safer.") }
	$msg = (T "Perfil: $($t[1])`n$($t[5])$note`n`n¿Aplicar? (resolución, FPS, reescalado y RAM)" "Profile: $($t[1])`n$($t[5])$note`n`nApply? (resolution, FPS, upscaling and RAM)")
	if ([System.Windows.Forms.MessageBox]::Show($msg, (T 'Optimizar según tu equipo' 'Optimize for your PC'), 'YesNo', 'Question') -ne 'Yes') { return }
	$cmbWin.SelectedIndex = 0
	$cmbInt.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:resList, $t[4]))
	foreach ($k in @('intres', 'upscaler', 'sharp', 'fsmode')) { Mark-Changed $k -Auto }  # el usuario ha pulsado Sí
	$cmbUp.SelectedIndex = $t[3]; Set-Sharp 75
	$cmbFs.SelectedIndex = 1
	$chk16.Checked = ($script:pc.RamGB -gt 0 -and $script:pc.RamGB -le 20)
	Write-Log "Optimizar: perfil $($t[1]) fps=$fps"
})
$btnSteam.Add_Click({
	if ([System.Windows.Forms.MessageBox]::Show((T 'Se cerrará Steam un momento para añadir "Bloodborne" a tu biblioteca (con imágenes). ¿Seguir?' 'Steam will close for a moment to add "Bloodborne" to your library (with artwork). Continue?'), 'Steam', 'YesNo', 'Question') -ne 'Yes') { return }
	$form.Cursor = 'WaitCursor'
	try { $r = Add-ToSteam } catch { $r = "$_"; Write-Log "Steam: $_" }
	$form.Cursor = 'Default'
	[void][System.Windows.Forms.MessageBox]::Show($r, 'Steam', 'OK', 'Information')
})
$script:doUpdate = {
	$form.Cursor = 'WaitCursor'
	try { $rel = Get-ShadRelease } catch { $form.Cursor = 'Default'; Write-Log "GitHub: $_"; [void][System.Windows.Forms.MessageBox]::Show((T "No se pudo consultar GitHub:`n$_" "Could not query GitHub:`n$_"), 'shadPS4', 'OK', 'Warning'); return }
	$form.Cursor = 'Default'
	if (-not $rel.Asset) { [void][System.Windows.Forms.MessageBox]::Show((T "La versión $($rel.Version) no tiene zip de Windows oficial." "Release $($rel.Version) has no official Windows zip."), 'shadPS4', 'OK', 'Information'); return }
	$have = @($script:emuList | Where-Object { (Split-Path -Leaf (Split-Path -Parent $_)) -match ('^v\.?' + [regex]::Escape($rel.Version) + '(-\d+)?$') })
	$msg = (T "Última versión oficial: $($rel.Version)  ($($rel.Date.Substring(0, [math]::Min(10, $rel.Date.Length))))`n$($rel.Asset.name)  ($([math]::Round($rel.Asset.size / 1MB)) MB)`nCambios: $($rel.Url)" "Latest official release: $($rel.Version)  ($($rel.Date.Substring(0, [math]::Min(10, $rel.Date.Length))))`n$($rel.Asset.name)  ($([math]::Round($rel.Asset.size / 1MB)) MB)`nChangelog: $($rel.Url)")
	if ($have.Count) { $msg += "`n`n" + (T "Ya la tienes: $($have[0])" "You already have it: $($have[0])") }
	$msg += "`n`n" + (T "AVISO: las versiones nuevas pueden guardar las partidas en otro sitio (parecerá partida nueva) y usar otro formato de parches. Tu versión actual no se toca y sigue siendo la elegida." "WARNING: newer versions may store saves elsewhere (it will look like a new game) and use a different patch format. Your current build is kept and stays selected.")
	$msg += "`n`n" + (T "Sí: instalar en una carpeta nueva (la actual no se toca)`nNo: ver los cambios en el navegador`nCancelar: nada" "Yes: install to a new folder (current one is kept)`nNo: open the changelog in the browser`nCancel: nothing")
	$r = [System.Windows.Forms.MessageBox]::Show($msg, (T 'Actualizar shadPS4' 'Update shadPS4'), 'YesNoCancel', 'Question')
	if ($r -eq 'No') { if ($rel.Url) { Start-Process $rel.Url }; return }
	if ($r -ne 'Yes') { return }
	$pbUpd.Value = 0; $pbUpd.Visible = $true; $lnkUpd.Enabled = $false; $btnUpd.Enabled = $false; $form.Cursor = 'WaitCursor'
	try { $exe = Install-ShadRelease $rel $pbUpd } catch { $exe = $null; Write-Log "Actualizar shadPS4: $_"; [void][System.Windows.Forms.MessageBox]::Show((T "No se pudo instalar:`n$_" "Could not install:`n$_"), 'shadPS4', 'OK', 'Error') }
	$pbUpd.Visible = $false; $lnkUpd.Enabled = $true; $btnUpd.Enabled = $true; $form.Cursor = 'Default'
	if (-not $exe) { return }
	[void][System.Windows.Forms.MessageBox]::Show((T "Instalada en:`n$exe`n`nSigues con tu versión de siempre. Para probar la nueva, elígela en la lista (y vuelve a la tuya igual)." "Installed to:`n$exe`n`nYou stay on your usual build. To try the new one, pick it in the list (and pick yours to go back)."), 'shadPS4', 'OK', 'Information')
	Fill-Lists
}
$lnkUpd.Add_LinkClicked({ & $script:doUpdate })
$btnUpd.Add_Click({ & $script:doUpdate })
$lnkUser.Add_LinkClicked({
	$d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description = (T 'Carpeta user de shadPS4 (la que tiene config.toml)' 'shadPS4 user folder (with config.toml)')
	$d.SelectedPath = (Get-UserDir $script:S.emu)
	if ($d.ShowDialog() -eq 'OK') { Set-Setting 'userdir' $d.SelectedPath; Write-Log "Carpeta user elegida: $($d.SelectedPath)"; Update-Emu; Update-Game }
})
$btnPre.Add_Click({
	$u = Get-UserDir $script:S.emu; $g = Test-GameFolder $script:S.game
	$cands = @(); if ($g.Serial) { $cands += (Join-Path $u "cache\$($g.Serial)"); $cands += (Join-Path $u "shader\$($g.Serial)") }; $cands += (Join-Path $u 'cache'); $cands += (Join-Path $u 'shader')
	$d = @($cands | Where-Object { Test-Path -LiteralPath $_ })[0]
	$n = 0; if ($d) { try { $n = @(Get-ChildItem -LiteralPath $d -Recurse -File -ErrorAction SilentlyContinue).Count } catch { } }
	$msg = (T 'shadPS4 no tiene precompilado de shaders: los compila la primera vez que aparecen y los guarda en su caché (los tirones solo pasan esa vez).' 'shadPS4 has no shader precompile: it compiles them the first time they appear and keeps them in its cache (stutters only happen that time).')
	$msg += "`n`n" + $(if ($d) { (T "Caché: $d`nArchivos: $n" "Cache: $d`nFiles: $n") + "`n`n" + (T 'Sí: abrir la carpeta' 'Yes: open the folder') } else { T 'Aún no hay caché de shaders (se crea al jugar).' 'No shader cache yet (it is created while playing).' })
	if ($d) { if ([System.Windows.Forms.MessageBox]::Show($msg, (T 'Precompilar shaders' 'Precompile shaders'), 'YesNo', 'Information') -eq 'Yes') { Start-Process explorer.exe -ArgumentList ('"' + $d + '"') } }
	else { [void][System.Windows.Forms.MessageBox]::Show($msg, (T 'Precompilar shaders' 'Precompile shaders'), 'OK', 'Information') }
})
$btnPlay.Add_Click({
	$btnPlay.Enabled = $false
	try { if (Start-Game) { $form.WindowState = 'Minimized' } } catch { Write-Log "JUGAR: $_"; [void][System.Windows.Forms.MessageBox]::Show("$_", 'Bloodborne', 'OK', 'Error') }
	$btnPlay.Enabled = $true
})
$form.Add_Activated({ if ($script:shownOnce) { try { Update-Game } catch { } } })

$btnX = New-Object System.Windows.Forms.Button
$btnX.Text = [string][char]0x2715; $btnX.Font = New-Object System.Drawing.Font('Segoe UI Symbol', 14, [System.Drawing.FontStyle]::Bold)
$btnX.FlatStyle = 'Flat'; $btnX.FlatAppearance.BorderSize = 0
$btnX.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22); $btnX.ForeColor = $gold
$btnX.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(180, 40, 40)
$btnX.Size = New-Object System.Drawing.Size(48, 40); $btnX.Location = New-Object System.Drawing.Point([math]::Max(8, $W - $panelW - 56), 8)  # a la izquierda del panel: no tapa nada
$btnX.Add_Click({ $form.Close() }); $form.Controls.Add($btnX); $btnX.BringToFront()

# ---- modos (Preset): aplicar y volver a Personalizado si se toca algo a mano
$script:applying = $false
function Update-ModeInfo {
	$m = Get-Mode
	$req = $script:modeReq[$m[0]]; if (-not $req) { $req = (T 'pendiente' 'pending') }
	$tip = (T 'Preset: fija resolución interna, reescalado, nitidez, frame generation y calidad. Personalizado = lo que elijas. Hoy shadPS4 aplica la resolución y 30/60 FPS (parche de 60, necesario también para 45); el tope de 45 llega con el emulador propio (BB_FPS_CAP).' 'Preset: sets internal resolution, upscaling, sharpness, frame generation and quality. Custom = your choice. Today shadPS4 applies resolution and 30/60 FPS (60 patch, also needed for 45); the 45 cap comes with our own emulator (BB_FPS_CAP).')
	if ($m[0] -eq 'personalizado') { $tip += "`n" + (T "Requisitos: $req" "Requirements: $req") }
	else { $tip += "`n" + (T "$($m[2]) reales, $($m[4]) con FG. Requisitos: $req" "$($m[2]) real, $($m[4]) with FG. Requirements: $req") }
	if ($m[6] -eq 1 -and $cmbFps.SelectedIndex -eq 0) { $tip += "`n" + (T 'Necesita el parche de 60 FPS: elige 60 en Límite FPS.' 'Needs the 60 FPS patch: pick 60 in FPS limit.') }
	$tips.SetToolTip($cmbMode, $tip)
}
function Apply-Mode {
	$m = Get-Mode
	Update-ModeInfo
	if ($m[0] -eq 'personalizado') { return }
	$script:applying = $true
	try {
		$cmbInt.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:resList, $m[5]))
		# El modo NO toca parches ni mods: el parche de 60 (necesario para 45/60) solo cambia si el usuario cambia la fila FPS.
		if ($m[6] -eq 1 -and $cmbFps.SelectedIndex -eq 0) { Write-Log "Modo $($m[0]): necesita el parche de 60 FPS activado (no se activa solo)" }
		$cmbUp.SelectedIndex = $m[7]; $cmbUpMode.SelectedIndex = $m[8]; Set-Sharp $m[9]
		$cmbFg.SelectedIndex = $(if ($m[3]) { [math]::Max(0, [array]::IndexOf(@($script:fgModes | ForEach-Object { $_[1] }), $m[3])) } else { 0 })
		$qk = @('q_shadows', 'q_ao', 'q_fog', 'q_aniso')
		for ($i = 0; $i -lt 4; $i++) { foreach ($pair in $script:qCombos) { if ($pair[1][0] -eq $qk[$i]) { $pair[0].SelectedIndex = [math]::Min($m[10][$i], $pair[0].Items.Count - 1) } } }
	} finally { $script:applying = $false }
	Write-Log "Modo aplicado: $($m[0])"
}
$cmbMode.Add_SelectionChangeCommitted({ Apply-Mode; if ((Get-Mode)[0] -ne 'personalizado') { foreach ($k in @('intres', 'upscaler', 'sharp')) { Mark-Changed $k -Auto } } })  # solo al elegirlo a mano
$cmbMode.Add_SelectedIndexChanged({ Update-ModeInfo })
$toCustom = { if (-not $script:applying -and $cmbMode.SelectedIndex -ne 4) { $cmbMode.SelectedIndex = 4 } }
foreach ($ctl in @($cmbInt, $cmbFps, $cmbUp, $cmbUpMode, $cmbFg, $cmbSharp)) { $ctl.Add_SelectionChangeCommitted($toCustom) }
# Marcas de cambio: solo acciones del usuario (no los cambios hechos por el programa).
$cmbWin.Add_SelectionChangeCommitted({ Mark-Changed 'winres' }); $cmbInt.Add_SelectionChangeCommitted({ Mark-Changed 'intres' })
$cmbFs.Add_SelectionChangeCommitted({ Mark-Changed 'fsmode' }); $cmbPres.Add_SelectionChangeCommitted({ Mark-Changed 'present' })
$cmbUp.Add_SelectionChangeCommitted({ Mark-Changed 'upscaler' }); $cmbSharp.Add_SelectionChangeCommitted({ Mark-Changed 'sharp' })
$chkHdr.Add_Click({ Mark-Changed 'hdr' }); $chkMotion.Add_Click({ Mark-Changed 'motion' }); $chkBg.Add_Click({ Mark-Changed 'bginput' })
foreach ($pair in $script:qCombos) { $pair[0].Add_SelectionChangeCommitted($toCustom) }

Fill-Lists; Update-Up; Update-ModeInfo; Update-LatBox; Update-VerLabel; Update-TrophyButton
Write-SharpLive
$panel.ResumeLayout($true); $form.ResumeLayout($true)
$form.Add_Shown({ $form.Refresh(); $form.Opacity = 1; $script:shownOnce = $true })
[void]$form.ShowDialog()
Write-Log 'Lanzador cerrado'
