# Bloodborne PC - lanzador para shadPS4 (ventana). Launcher-bloodborne.cmd lo abre.
#   JUGAR    -> escribe los ajustes en la config de shadPS4 (user\config.toml de la 0.15 o user\config.json de las
#               versiones nuevas; copia de seguridad antes), prepara las variables BB_* para el emulador propio que
#               viene y arranca shadPS4.exe -g "<juego>\eboot.bin" -f true|false.
#   -Play    -> sin ventana: arranca con los ultimos ajustes guardados (lo usa el acceso de Steam).
#   -Lang es|en -> fuerza el idioma de la interfaz. Sin -Lang: (launcher132) siempre el idioma de Windows
#               (es-* -> espanol, cualquier otro -> ingles); launcher_lang del ini se ignora. Un solo script (T es en).
#   -NoUi    -> solo carga las funciones (pruebas).
#   -Menu    -> abre el lanzador aunque inicio_directo=1 (tambien: Mayus al abrirlo).
# Ajustes del lanzador en launcher.ini; registro en launcher.log (los dos junto a este script).
param([string]$Lang = '', [switch]$Play, [switch]$NoUi, [switch]$Menu, [switch]$Steam)

$ErrorActionPreference = 'Stop'
$root    = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$iniFile = Join-Path $root 'launcher.ini'
$logFile = Join-Path $root 'launcher.log'
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$reduceMotion = $false  # true = sin animaciones (las secciones se abren y cierran de golpe; sin brasas ni niebla)
# (launcher114) Zonas del fondo fondo-bloodborne2.jpg en pixeles de la imagen (W x H = tamano de referencia).
# Si cambias la imagen, ajusta aqui: el titulo nunca se tapa; Trofeos evita al cazador.
$bgLayout = @{
	W = 1672; H = 941
	Title  = @{ X0 = 50;   Y0 = 20;  X1 = 710;  Y1 = 245 }   # "Bloodborne PC" (arriba a la izquierda)
	Hunter = @{ X0 = 410;  Y0 = 240; X1 = 960;  Y1 = 941 }   # cazador (centro-izquierda)
	Moon   = @{ X0 = 1015; Y0 = 25;  X1 = 1235; Y1 = 245 }   # luna (arriba, centro-derecha)
}
$fxEmbers = 70   # particulas (brasas + ceniza)
$fxFog = 5       # capas de niebla (0 = sin niebla, max 5)
# (launcher112) Idioma: -Lang manda; si no, el idioma de Windows. (launcher132) launcher_lang del ini ya no se lee
# (la clave se deja en el ini tal cual).
# Se resuelve aqui (antes de leer el resto del ini) porque las tablas de abajo ya usan T al definirse.
$script:langArg = "$Lang".Trim().ToLower()
function Resolve-LauncherLang {
	if (@('es', 'en') -contains $script:langArg) { return $script:langArg }
	$cul = ''; try { $cul = [System.Globalization.CultureInfo]::CurrentUICulture.Name } catch { }; if (-not $cul) { try { $cul = (Get-UICulture).Name } catch { } }
	if ($cul -match '^(?i)es(-|$)') { return 'es' }; return 'en'
}
$Lang = Resolve-LauncherLang

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
$script:alHold = $false
$script:defaults = [ordered]@{
	game = ''; emu = ''; userdir = ''
	winres = 'auto'; intres = '1920x1080'; fsmode = '1'; present = '1'; fps = '1'; patches = '1'; useupdate = '1'; mvzero = '0'; vol = '0'
	hdr = '0'; showfps = '0'; metricas = '0'
	upscaler = '0'; upmode = '1'; sharp = '75'; framegen = '0'; sceneup = '0'; bbfg = '0'; antilag = '0'; reflex = '1'; almode = '0'; aldlssg = '0'
	q_shadows = '0'; q_ao = '0'; q_fog = '0'; q_aniso = '0'; q_ca = '0'; q_blur = '0'; q_grain = '0'
	cal = '1'; cal_lod = '1'; cal_grid = '1'; cal_dynsh = '1'; cal_ssao = '1'; cal_dof = '0'; cal_blur = '0'; cal_ssr = '1'; cal_ca = '0'; cal_aa = '1'
	tex = '1'; aniso = '3'
	input = '0'; motion = '0'; bginput = '0'
	experimental = '0'; modo = '2'; safemode = '1'; soundfix = '1'
	launcher_lang = 'auto'; gamelang = 'auto'; aviso_ok = '0'; inicio_directo = '0'
	dlss5 = '3'; dlss5_intensidad = 'alta'; dlss5_no_oficial = '1'  # (bb-dlss5-042) como el settings.ini de Demon's Souls
}
$script:S = [ordered]@{}
function Read-Settings {
	foreach ($k in $script:defaults.Keys) { $script:S[$k] = $script:defaults[$k] }
	if (Test-Path -LiteralPath $iniFile) {
		foreach ($line in [IO.File]::ReadAllLines($iniFile)) {
			if ($line -match '^\s*([A-Za-z0-9_]+)\s*=(.*)$') { if (@('ram16', 'rtrefl') -contains $Matches[1]) { continue }; $script:S[$Matches[1]] = $Matches[2].Trim() }  # (launcher112) opciones quitadas: se ignoran
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
# (launcher112) Unica version soportada: CUSA03173 (Europa) con la actualizacion 1.09. Devuelve el aviso o ''.
function Get-SupportWarning($g) {
	if (-not $g -or -not $g.Ok) { return '' }
	$v = "$($g.Version)"; if ($g.Update -and $g.UpdateVersion -and $script:S.useupdate -ne '0') { $v = "$($g.UpdateVersion)" }
	$okV = ($v -match '^0?1\.09$')
	if ($g.Serial -eq 'CUSA03173' -and $okV) { return '' }
	$what = @(); if ($g.Serial -ne 'CUSA03173') { $what += "$(if ($g.Serial) { $g.Serial } else { '?' })" }; if (-not $okV) { $what += "v$(if ($v) { $v } else { '?' })" }
	return (T "No soportado ($($what -join ', ')): solo CUSA03173 v1.09. Puede no arrancar o verse mal." "Unsupported ($($what -join ', ')): only CUSA03173 v1.09. May not start or may look broken.")
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
	return @($res | Where-Object { $_ -notmatch '\\_anterior\\' } | Select-Object -Unique)  # (launcher132) sin las copias de _anterior
}
function Get-EmuLabel([string]$exe) {
	$v = ''; try { $v = (Get-Item -LiteralPath $exe).VersionInfo.ProductVersion } catch { }
	$rel = $exe; if ($exe.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { $rel = $exe.Substring($root.Length).TrimStart('\') }
	if ($v) { return "$rel  ($v)" } return $rel
}

# ---------------------------------------------------------------- (launcher132) emulador: ruta, version y copia anterior
# Get-BbPortExe: el shadPS4 de bb-port-medicion (reescalado / frame generation / DLSS 5). Get-ActiveEmuExe: el exe que
# arrancaria JUGAR ahora (misma regla que Start-Game: en modo seguro el fork si existe; fuera, solo con reescalado,
# frame generation o DLSS 5). Las copias de _anterior\ nunca salen en la lista de emuladores.
function Get-BbPortExe { return (Join-Path $PSScriptRoot 'shadPS4.exe') }
function Get-ActiveEmuExe {
	$pe = Get-BbPortExe
	try {
		if (Test-Path -LiteralPath $pe) {
			if ($script:S.safemode -ne '0') { return $pe }
			$k = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)][1]
			$on = (@('dlss', 'fsr3', 'fsr4', 'xess') -contains $k)
			if (-not $on) { $d5 = 0; [void][int]::TryParse("$($script:S.dlss5)", [ref]$d5); $on = ($d5 -gt 0) }
			if (-not $on) { try { $on = ((Get-FgEnv)['BB_FG'] -ne 'off') } catch { } }
			if ($on) { return $pe }
		}
	} catch { }
	return "$($script:S.emu)"
}
# Primeros bytes de un texto aunque otro programa lo tenga abierto (shadPS4 escribe su log mientras juega).
function Read-TextHead([string]$file, [int]$max = 65536) {
	$fs = [IO.FileStream]::new($file, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
	try { $n = [int][math]::Min([long]$max, $fs.Length); $buf = New-Object byte[] $n; $r = $fs.Read($buf, 0, $n); return [Text.Encoding]::UTF8.GetString($buf, 0, $r) } finally { $fs.Close() }
}
# Version del exe: version.txt junto al exe, la linea "build ..." del ultimo log (solo si el log es posterior al exe),
# ProductVersion / FileVersion del exe; fecha del archivo y MD5 corto (8 cifras, en cache por ruta + fecha + tamano).
$script:md5Cache = @{}
function Get-EmuBuildInfo([string]$exe) {
	$o = [pscustomobject]@{ Exe = $exe; Text = ''; Short = ''; Date = ''; Md5 = '' }
	if (-not $exe -or -not (Test-Path -LiteralPath $exe)) { return $o }
	$fi = Get-Item -LiteralPath $exe; $parts = @()
	try { $vt = Join-Path $fi.DirectoryName 'version.txt'; if (Test-Path -LiteralPath $vt) { $l = @([IO.File]::ReadAllLines($vt) | Where-Object { "$_".Trim() })[0]; if ($l) { $parts += "version.txt: $("$l".Trim())" } } } catch { }
	try {
		$logs = @()
		foreach ($c in @((Join-Path (Get-UserDir $exe) 'log\shad_log.txt'), (Join-Path $fi.DirectoryName 'user\log\shad_log.txt'), (Join-Path $root 'user\log\shad_log.txt'))) { if (Test-Path -LiteralPath $c) { $logs += Get-Item -LiteralPath $c } }
		$lg = @($logs | Sort-Object LastWriteTime -Descending)[0]
		if ($lg -and $lg.LastWriteTime -ge $fi.LastWriteTime) {
			$tx = Read-TextHead $lg.FullName
			if ($tx -match '(?i)\bbuild\s+(\d+\.\d+[^\s,;]*(?:\s*\([^)\r\n]{1,80}\))?)') { $parts += (T "build $($Matches[1])  (último log, $($lg.LastWriteTime.ToString('dd/MM HH:mm')))" "build $($Matches[1])  (last log, $($lg.LastWriteTime.ToString('dd/MM HH:mm')))") }
		}
	} catch { }
	try {
		$pv = "$($fi.VersionInfo.ProductVersion)".Trim(); $fv = "$($fi.VersionInfo.FileVersion)".Trim()
		if ($pv -and $fv -and $pv -ne $fv) { $parts += "exe: $pv  ($fv)" } elseif ($pv -or $fv) { $parts += "exe: $(if ($pv) { $pv } else { $fv })" }
	} catch { }
	if (-not $parts.Count) { $parts += (T 'sin versión legible' 'no readable version') }
	$o.Text = ($parts -join "`n"); $o.Short = ($parts -join ' | ')
	$o.Date = $fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm') + "  ($([math]::Round($fi.Length / 1MB, 1)) MB)"
	$key = "$($fi.FullName)|$($fi.LastWriteTimeUtc.Ticks)|$($fi.Length)"
	if (-not $script:md5Cache.ContainsKey($key)) { try { $script:md5Cache[$key] = (Get-FileHash -LiteralPath $fi.FullName -Algorithm MD5).Hash.Substring(0, 8).ToLower() } catch { $script:md5Cache[$key] = '?' } }
	$o.Md5 = $script:md5Cache[$key]
	return $o
}
# Copia anterior: antes de cada actualizacion (Actualizar o shadPS4.part*) se copian shadPS4.exe, las .dll y version.txt
# de la carpeta del exe a <carpeta del exe>\_anterior\<fecha>\ con manifest.txt (fecha, motivo, carpeta, exe, emulador
# elegido, version, md5, archivos). Se guardan las 3 ultimas. Nunca se copia ni se toca la carpeta user (partidas y
# ajustes). Los shadPS4-xxx.exe sueltos (instalaciones a mano, shadPS4-anterior.exe) no se copian.
$script:bakKeep = 3
function Read-BackupManifest([string]$dir) {
	$m = @{ Bak = $dir; Files = @() }
	foreach ($l in [IO.File]::ReadAllLines((Join-Path $dir 'manifest.txt'))) {
		if ($l -match '^(fecha|motivo|carpeta|exe|emu|version|md5|file)=(.*)$') { if ($Matches[1] -eq 'file') { $m.Files += $Matches[2] } else { $m[$Matches[1]] = $Matches[2] } }
	}
	return $m
}
function Get-EmuBackups {
	$roots = @()
	foreach ($e in @((Get-ActiveEmuExe), "$($script:S.emu)", (Get-BbPortExe))) { if ($e) { try { $r = Join-Path (Split-Path -Parent $e) '_anterior'; if ($roots -notcontains $r) { $roots += $r } } catch { } } }
	$res = @()
	foreach ($r in $roots) {
		if (-not (Test-Path -LiteralPath $r)) { continue }
		foreach ($d in @(Get-ChildItem -LiteralPath $r -Directory -ErrorAction SilentlyContinue)) {
			if (Test-Path -LiteralPath (Join-Path $d.FullName 'manifest.txt')) { try { $m = Read-BackupManifest $d.FullName; if ($m.carpeta -and $m.Files.Count) { $res += $m } } catch { } }
		}
	}
	return @($res | Sort-Object { "$($_.fecha)" } -Descending)
}
function Get-LatestEmuBackup { return @(Get-EmuBackups)[0] }
function Backup-EmuFiles([string]$exe, [string]$reason) {
	if (-not $exe -or -not (Test-Path -LiteralPath $exe)) { return $null }
	$dir = Split-Path -Parent $exe; $leaf = Split-Path -Leaf $exe
	$info = Get-EmuBuildInfo $exe
	$broot = Join-Path $dir '_anterior'
	# La ultima copia ya es este mismo exe (mismo MD5) con el mismo emulador elegido: no se repite.
	$last = $null
	try { $last = @(@(Get-ChildItem -LiteralPath $broot -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'manifest.txt') } | Sort-Object Name -Descending) | ForEach-Object { Read-BackupManifest $_.FullName })[0] } catch { }
	if ($last -and $last.md5 -eq $info.Md5 -and $last.exe -eq $exe -and $last.emu -eq "$($script:S.emu)") { Write-Log "Copia anterior ($reason): ya existe $($last.Bak)"; return $last.Bak }
	$files = @(Get-ChildItem -LiteralPath $dir -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -ieq $leaf -or $_.Extension -ieq '.dll' -or $_.Name -ieq 'version.txt' })
	if (-not $files.Count) { return $null }
	$ts = Get-Date -Format 'yyyyMMdd-HHmmss'
	$dest = Join-Path $broot $ts; $i = 2; while (Test-Path -LiteralPath $dest) { $dest = Join-Path $broot "$ts-$i"; $i++ }
	[void](New-Item -ItemType Directory -Path $dest -Force)
	try {
		foreach ($f in $files) { Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $dest $f.Name) -Force }
		$lines = @("fecha=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "motivo=$reason", "carpeta=$dir", "exe=$exe", "emu=$($script:S.emu)", "version=$($info.Short)", "md5=$($info.Md5)")
		foreach ($f in $files) { $lines += "file=$($f.Name)" }
		[IO.File]::WriteAllLines((Join-Path $dest 'manifest.txt'), [string[]]$lines, $utf8NoBom)
	} catch { Remove-Item -LiteralPath $dest -Recurse -Force -ErrorAction SilentlyContinue; throw }
	Write-Log "Copia anterior del emulador ($reason): $dest ($($files.Count) archivos, md5 $($info.Md5))"
	$old = @(Get-ChildItem -LiteralPath $broot -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'manifest.txt') } | Sort-Object Name -Descending | Select-Object -Skip $script:bakKeep)
	foreach ($od in $old) { Remove-Item -LiteralPath $od.FullName -Recurse -Force -ErrorAction SilentlyContinue; Write-Log "Copia anterior borrada (max $($script:bakKeep)): $($od.FullName)" }
	return $dest
}
# Vuelve a dejar los archivos de la copia en su carpeta y el emulador elegido como estaba; la copia usada se borra
# (la siguiente vez se vuelve a la de antes). No toca la carpeta user.
function Restore-EmuBackup($m) {
	if (@(Get-Process -Name 'shadPS4' -ErrorAction SilentlyContinue).Count) { throw (T 'shadPS4 está abierto: ciérralo antes.' 'shadPS4 is running: close it first.') }
	$target = "$($m.carpeta)"
	if (-not $target -or -not (Test-Path -LiteralPath $target)) { throw (T "No existe la carpeta del emulador: $target" "The emulator folder does not exist: $target") }
	foreach ($n in $m.Files) {
		if ($n -match '[\\/:]|\.\.' -or -not $n) { throw (T "Nombre no válido en manifest.txt: $n" "Invalid name in manifest.txt: $n") }
		if (-not (Test-Path -LiteralPath (Join-Path $m.Bak $n))) { throw (T "Falta en la copia: $n" "Missing from the backup: $n") }
	}
	foreach ($n in $m.Files) { Copy-Item -LiteralPath (Join-Path $m.Bak $n) -Destination (Join-Path $target $n) -Force }
	if ($m.emu -and (Test-Path -LiteralPath $m.emu) -and $m.emu -ne "$($script:S.emu)") { Set-Setting 'emu' $m.emu }
	Write-Log "Volver a la version anterior: $($m.Bak) -> $target ($($m.Files -join ', ')); emulador elegido: $($script:S.emu)"
	Remove-Item -LiteralPath $m.Bak -Recurse -Force -ErrorAction SilentlyContinue
}

# Carpeta "user" de shadPS4: la portable (carpeta de trabajo\user). shadPS4 la busca en la carpeta desde la que se
# arranca; si no existe, en %APPDATA%\shadPS4 (en Windows la crea en la carpeta de trabajo). Orden: la elegida en
# launcher.ini, junto al exe, junto al lanzador, %APPDATA%\shadPS4.
# (10-29) Resolucion interna: en Bloodborne la manda el "Resolution Patch WxH" de los parches de shadPS4, no la config.
# Se activa el de la resolucion elegida y se desactivan los demas "Resolution Patch" (1920x1080 = ninguno: la nativa).
# Antes se guarda copia en user\launcher-backups. El resto de parches no se toca.
function Set-ResolutionPatch([string]$userDir, [string]$res) {
	# (sin-parches) patches=0 en launcher.ini: no se escribe ningun parche (ni calidad ni resolucion).
	if ($script:S.patches -eq '0') { return }
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
# (fpspatch) Parche de FPS: 0 = 30 FPS++, 1 = 60 FPS++, 2 = Uncap FPS++; los demas de FPS se apagan. Equilibrado (45) necesita el de 60.
function Get-FpsCapIdx {
	# (fps117) indice de limite 0 = 30, 1 = 60, 2 = sin limite, para las 5 opciones del parche de FPS
	$fi = 0; [void][int]::TryParse("$($script:S.fps)", [ref]$fi)
	return @(0, 1, 2, 1, 0, 2)[[math]::Max(0, [math]::Min($fi, 5))]
}
function Set-FpsPatch([string]$userDir) {
	if ($script:S.patches -eq '0') { return }
	$fi = 0; [void][int]::TryParse("$($script:S.fps)", [ref]$fi); $mm = Get-Mode
	if ($mm[0] -eq 'equilibrado' -and ($fi -eq 0 -or $fi -eq 4)) { $fi = 1 }
	# (fps117) 3 = 60FPS (no deltatime), 4 = sin parche de FPS (30 original: todos apagados)
	$want = @('30 FPS++', '60 FPS++', 'Uncap FPS++', '60FPS (no deltatime)', '', 'Uncap FPS (consistent dt)')[[math]::Max(0, [math]::Min($fi, 5))]
	$all = @('30 FPS++', '60 FPS++', '60FPS (no deltatime)', '90 FPS++', 'Uncap FPS++', 'Uncap FPS (consistent dt)', '40 FPS++')
	foreach ($rel in @('patches\shadPS4\Bloodborne.xml', 'patches\GoldHEN\Bloodborne-Orbis.xml')) {
		$x = Join-Path $userDir $rel
		if (-not (Test-Path -LiteralPath $x)) { continue }
		try {
			$txt = [IO.File]::ReadAllText($x)
			$new = [regex]::Replace($txt, '(<Metadata[^>]*Name="([^"]*)"[^>]*isEnabled=")(true|false)(")', {
				param($m)
				if ($all -notcontains $m.Groups[2].Value) { return $m.Value }
				$m.Groups[1].Value + $(if ($m.Groups[2].Value -eq $want) { 'true' } else { 'false' }) + $m.Groups[4].Value
			})
			# (11-12) 30 original (sin parche de FPS): 'Disable vsync' quita el tope de 30 y el juego va acelerado (~100 fps, fisica por fotograma): se apaga
			if (-not $want) { $new = [regex]::Replace($new, '(<Metadata[^>]*Name="Disable vsync"[^>]*isEnabled=")true(")', '${1}false${2}') }
			if ($new -ne $txt) {
				$bd = Join-Path $userDir 'launcher-backups'; New-Item -ItemType Directory -Force -Path $bd | Out-Null
				Copy-Item -LiteralPath $x -Destination (Join-Path $bd ((Split-Path -Leaf $x) + '.' + (Get-Date -Format 'yyyyMMdd-HHmmss'))) -Force
				[IO.File]::WriteAllText($x, $new, (New-Object System.Text.UTF8Encoding $false))
			}
			Write-Log "Parche de FPS: $(if ($want) { $want } else { 'ninguno (30 original)' }) ($rel)"
		} catch { Write-Log "No se pudo cambiar el parche de FPS en ${rel}: $_" }
	}
}
# (143) 'SAVE GAMES' y 'BLOODBORNE GAME' en la carpeta del lanzador: accesos directos (union NTFS) a donde estan de verdad.
function Ensure-Link([string]$name, [string]$tgt) {
	try {
		if (-not $tgt) { return }
		if (-not (Test-Path -LiteralPath $tgt)) { [void](New-Item -ItemType Directory -Path $tgt -Force) }
		$lnk = Join-Path $root $name
		if (Test-Path -LiteralPath $lnk) { $it = Get-Item -LiteralPath $lnk -Force; if ($it.LinkType -eq 'Junction' -and (@($it.Target)[0] -ne $tgt)) { [System.IO.Directory]::Delete($lnk) } else { return } }
		[void](New-Item -ItemType Junction -Path $lnk -Target $tgt -ErrorAction Stop)
		Write-Log "$name -> $tgt"
	} catch { try { Write-Log "$name : no se pudo crear ($($_.Exception.Message))" } catch { } }
}
function Ensure-PortLinks {
	try { $u = Get-UserDir $script:S.emu } catch { $u = $null }
	$h = $null
	if ($u) { try { $cj = Join-Path $u 'config.json'; if (Test-Path -LiteralPath $cj) { $hd = (Get-Content -LiteralPath $cj -Raw | ConvertFrom-Json).General.home_dir; if ($hd) { $h = $hd } } } catch { } }
	if (-not $h -and $u) { $h = Join-Path $u 'home' }
	if ($h) { Ensure-Link 'SAVE GAMES' (Join-Path $h '1000\savedata') }
	if ($script:S.game -and (Test-Path -LiteralPath $script:S.game)) {
		$gp = Split-Path -Parent $script:S.game
		if ($gp -and ($gp.TrimEnd('\') -ne $root.TrimEnd('\')) -and ((Split-Path -Leaf $gp) -ne 'BLOODBORNE GAME')) { Ensure-Link 'BLOODBORNE GAME' $gp }
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

# ---------------------------------------------------------------- (launcher112) idioma del juego
# Bloodborne toma el idioma de los textos Y de las voces del idioma del sistema PS4 (no hay ajuste separado en el
# juego): una sola opcion "Idioma (texto y voces)". Las voces solo existen en algunos idiomas; si no, el juego usa
# las de por defecto. IDs de idioma del sistema PS4 (SceSystemServiceParamId lang).
# (launcher131) Solo los idiomas que trae el juego (CUSA03173: dvdroot_ps4\msg\<idioma>, 21 carpetas). El tercer campo
# indica si tiene voces propias (dvdroot_ps4\sound\*_<voz>.fsb: eng, jaj, ese, esa, frf, ded, iti, ptb); el resto, voces en ingles.
$script:ps4Langs = @(
	@(3, 'Español (España)', $true), @(20, 'Español (Latinoamérica)', $true), @(1, 'English (US)', $true), @(18, 'English (UK)', $true),
	@(2, 'Français', $true), @(4, 'Deutsch', $true), @(5, 'Italiano', $true), @(17, 'Português (Brasil)', $true), @(0, '日本語 (Japanese)', $true),
	@(7, 'Português (Portugal)', $false), @(6, 'Nederlands', $false), @(8, 'Русский', $false), @(16, 'Polski', $false), @(19, 'Türkçe', $false),
	@(12, 'Suomi', $false), @(13, 'Svenska', $false), @(14, 'Dansk', $false), @(15, 'Norsk', $false), @(9, '한국어 (Korean)', $false),
	@(10, '繁體中文', $false), @(11, '简体中文', $false)
)
# Windows -> ID PS4 (para "Igual que Windows").
function Get-WinPs4Lang {
	$c = ''; try { $c = [System.Globalization.CultureInfo]::CurrentUICulture.Name } catch { }
	switch -Regex ($c) {
		'^ja' { return 0 } '^en-(GB|IE|AU|NZ|IN|ZA)' { return 18 } '^en' { return 1 } '^fr' { return 2 }
		'^es-ES' { return 3 } '^es' { return 20 } '^de' { return 4 } '^it' { return 5 } '^nl' { return 6 } '^pt-BR' { return 17 } '^pt' { return 7 }
		'^ru' { return 8 } '^ko' { return 9 } '^zh-(TW|HK|MO|Hant)' { return 10 } '^zh' { return 11 } '^fi' { return 12 } '^sv' { return 13 }
		'^da' { return 14 } '^(nb|nn|no)' { return 15 } '^pl' { return 16 } '^tr' { return 19 }
	}
	return 1
}
# (launcher131) Un id guardado que el juego no trae (22 Frances de Canada, 21 Arabe) pasa al mas cercano.
function Get-GameLangId {
	$v = -1
	if ("$($script:S.gamelang)" -ne 'auto' -and [int]::TryParse("$($script:S.gamelang)", [ref]$v)) {
		if ($v -eq 22) { return 2 }; if ($v -eq 21) { return 1 }
		foreach ($pl in $script:ps4Langs) { if ([int]$pl[0] -eq $v) { return $v } }
	}
	return (Get-WinPs4Lang)
}
# Escribe SOLO la clave de idioma que ya exista en la config de shadPS4 (toml: [Settings]/[General] consoleLanguage o
# systemLanguage; json: la clave equivalente). Copia de seguridad antes (config.*.original la primera vez).
function Set-GameLanguage([string]$userDir) {
	$id = [int](Get-GameLangId); $done = @()
	$toml = Join-Path $userDir 'config.toml'; $json = Join-Path $userDir 'config.json'
	if (Test-Path -LiteralPath $toml) {
		$old = [IO.File]::ReadAllText($toml); $ch = [ordered]@{}
		foreach ($sec in 'Settings', 'General') {
			$hm = [regex]::Match($old, '(?m)^[ \t]*\[' + $sec + '\][ \t]*\r?$'); if (-not $hm.Success) { continue }
			$st = $hm.Index + $hm.Length; $nx = [regex]::Match($old.Substring($st), '(?m)^[ \t]*\[[^\]\r\n]+\][ \t]*\r?$')
			$body = $(if ($nx.Success) { $old.Substring($st, $nx.Index) } else { $old.Substring($st) })
			foreach ($k in 'consoleLanguage', 'systemLanguage') { if ($body -match ('(?m)^[ \t]*' + $k + '[ \t]*=')) { if (-not $ch.Contains($sec)) { $ch[$sec] = [ordered]@{} }; $ch[$sec][$k] = $id } }
		}
		if ($ch.Count) { $new = Set-TomlValues $old $ch; if ($new -ne $old) { Backup-ShadConfig $toml; [IO.File]::WriteAllText($toml, $new, $utf8NoBom); $done += $toml } }
		else { Write-Log 'config.toml sin consoleLanguage/systemLanguage: idioma del juego no cambiado' }
	}
	if (Test-Path -LiteralPath $json) {
		$old = [IO.File]::ReadAllText($json); $jv = [ordered]@{}
		foreach ($k in 'console_language', 'system_language', 'consoleLanguage', 'systemLanguage') { if ($old -match ('"' + $k + '"\s*:')) { $jv[$k] = $id } }
		if ($jv.Count) { $new = Set-JsonValues $old $jv; if ($new -ne $old) { Backup-ShadConfig $json; [IO.File]::WriteAllText($json, $new, $utf8NoBom); $done += $json } }
		else { Write-Log 'config.json sin clave de idioma: idioma del juego no cambiado' }
	}
	if ($done.Count) { Write-Log "Idioma del juego: $id -> $($done -join ', ')" }
}

# (bbl2b) Parches de CALIDAD (permiso del dueno): al JUGAR (tambien en modo seguro) se pone isEnabled SOLO de las
# entradas de calidad segun las filas de Calidad. Ningun otro parche se toca (FPS, resolucion...). Copia unica
# <archivo>.bak-calidad antes del primer cambio. Mismo formato que Set-ResolutionPatch.
$script:calWhitelist = '^(Model LOD (-2|1|2) \(|(1080p|1440p|4k|1280x800) Light Grid|Disable Dynamic Light Shadows|Disable SSAO|Disable DoF|Disable Motion Blur|Enable Screen Space Reflections|Disable Chromatic Aberration|Disable AA)'
function Set-QualityPatches([string]$userDir, [string]$serial) {
	# (sin-parches) patches=0 en launcher.ini: no se escribe ningun parche (ni calidad ni resolucion).
	if ($script:S.patches -eq '0') { return }
	$rules = @()
	foreach ($q in $script:quality) {
		$lv = 1; [void][int]::TryParse("$($script:S[$q[0]])", [ref]$lv); $lv = [math]::Max(0, [math]::Min($lv, 4))  # 4 = Ultra bajo (al final)
		$want = $q[3][$lv]; if ($want -eq 'GRID') { $want = Get-GridPatch }
		$rules += , @($q[4], $want)
	}
	$files = @(Find-PatchFiles $userDir $serial)
	foreach ($rel in @('patches\shadPS4\Bloodborne.xml', 'patches\GoldHEN\Bloodborne-Orbis.xml')) { $files += (Join-Path $userDir $rel) }
	foreach ($x in @($files | Select-Object -Unique)) {
		if (-not (Test-Path -LiteralPath $x)) { continue }
		try {
			$txt = [IO.File]::ReadAllText($x)
			$new = [regex]::Replace($txt, '(<Metadata[^>]*Name="([^"]*)"[^>]*isEnabled=")(true|false)(")', {
				param($m)
				$n = $m.Groups[2].Value
				if ($n -notmatch $script:calWhitelist) { return $m.Value }
				foreach ($r in $rules) { if ($n -match $r[0]) { $on = $(if ($r[1] -and $n -match $r[1]) { 'true' } else { 'false' }); return $m.Groups[1].Value + $on + $m.Groups[4].Value } }
				return $m.Value
			})
			if ($new -ne $txt) {
				$bak = "$x.bak-calidad"; if (-not (Test-Path -LiteralPath $bak)) { Copy-Item -LiteralPath $x -Destination $bak }
				[IO.File]::WriteAllText($x, $new, (New-Object System.Text.UTF8Encoding $false))
				Write-Log "Calidad: parches de calidad actualizados en $x"
			}
		} catch { Write-Log "No se pudieron cambiar los parches de calidad en ${x}: $_" }
	}
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
# Los xml de parches no se escriben, salvo resolucion (Set-ResolutionPatch) y calidad (Set-QualityPatches).
$script:pat60   = '60\s*-?\s*FPS|60\s*fps|60fps|60 Frames'
$script:patUnlock = '(?i)uncap|unlock'  # (patunlock-052) solo los que QUITAN el limite (Uncap FPS++): '60FPS (no deltatime)' / '60 FPS++' fijan 60, no lo quitan (antes daban fps=2 = sin tope)
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
	# (launcher157b) Nombres: shadPS4 los extrae en <user>\trophy\<NPWR>\Xml\TROP_NN.XML (03 = espanol, 01 = ingles).
	try {
		$nf = Join-Path (Join-Path $userDir 'trophy') "$serial.npcommid"
		if (Test-Path -LiteralPath $nf) {
			$np = ([IO.File]::ReadAllText($nf)).Trim()
			$xd = Join-Path (Join-Path (Join-Path $userDir 'trophy') $np) 'Xml'
			$langs = $(if ($Lang -eq 'en') { @('TROP_01.XML', 'TROP.XML', 'TROP_03.XML') } else { @('TROP_03.XML', 'TROP_01.XML', 'TROP.XML') })
			foreach ($ln in $langs) { $c = Join-Path $xd $ln; if (Test-Path -LiteralPath $c) { $found = @(Get-Item -LiteralPath $c); break } }
		}
	} catch { Write-Log "Nombres de trofeos: $_" }
	if (-not $found.Count -and $hdir -and (Test-Path -LiteralPath $hdir)) { $found = @(Get-ChildItem -LiteralPath $hdir -Recurse -Depth 6 -Filter 'TROP.XML' -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match [regex]::Escape($serial) } | Select-Object -First 1) }
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
				$r.Items += [pscustomobject]@{ Id = [int]$t.GetAttribute('id'); Grade = $g.Substring(0, 1); Name = $name; Detail = $det; Unlocked = $un }
			}
		} catch { Write-Log "TROP.XML ilegible: $($x.FullName) ($_)" }
	}
	# (launcher157) Los trofeos ganados los guarda shadPS4 en <home_dir>\1000\trophy\NPWRxxxxx_00.xml (unlockstate), no en TROP.XML:
	# se leen de ahi y se aplican por id. Si no hay TROP.XML, se listan con nombre generico.
	try {
		$st = @()
		if ($hdir -and (Test-Path -LiteralPath $hdir)) { $st = @(Get-ChildItem -LiteralPath $hdir -Recurse -Depth 3 -Filter 'NPWR*.xml' -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^NPWR\d+_\d+\.xml$' -and $_.FullName -match '\\1000\\trophy\\' } | Select-Object -First 1) }
		if ($st.Count) {
			$sd = New-Object System.Xml.XmlDocument; $sd.Load($st[0].FullName)
			$un = @{}; $grades = @{}; $ids = @()
			foreach ($t in @($sd.SelectNodes('//trophy'))) { $id = [int]$t.GetAttribute('id'); $ids += $id; $grades[$id] = "$($t.GetAttribute('ttype'))"; $un[$id] = ($t.GetAttribute('unlockstate') -eq 'true') }
			if (-not $r.Found) {
				foreach ($id in $ids) { $g = $grades[$id]; if (-not $g) { continue }; $r.Items += [pscustomobject]@{ Id = $id; Grade = $g.Substring(0, 1); Name = (T "Trofeo $id" "Trophy $id"); Detail = ''; Unlocked = $false } }
				$r.Found = ($r.Items.Count -gt 0)
			}
			$i = 0; $all = 0; $got = 0; $r.Count = 0; $r.Total = 0
			foreach ($it in $r.Items) {
				$id = $(if ($null -ne $it.Id) { $it.Id } elseif ($i -lt $ids.Count) { $ids[$i] } else { $i })
				if ($un.ContainsKey($id) -and $un[$id]) { $it.Unlocked = $true }
				$p = $pts[$it.Grade]; if (-not $p) { $p = 15 }; $all += $p; $r.Total++
				if ($it.Unlocked) { $r.Count++; $got += $p }
				$i++
			}
		}
	} catch { Write-Log "Trofeos ganados ilegibles: $_" }
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
foreach ($un in @(@('NVIDIA DLSS', 'dlss'), @('AMD FSR 4', 'fsr4'), @('AMD FSR 3.1', 'fsr3'), @('Intel XeSS', 'xess'))) {
	foreach ($um in @(@((T 'Nativo (AA)' 'Native (AA)'), 'native'), @((T 'Calidad' 'Quality'), 'quality'), @((T 'Equilibrado' 'Balanced'), 'balanced'), @((T 'Rendimiento' 'Performance'), 'performance'), @((T 'Ultra rendimiento' 'Ultra performance'), 'ultraperf'))) {
		$script:upscalers += ,@("$($un[0]) $($um[0])", $un[1], $true, $um[1])
	}
}
# (taa-1109) TAA propio del emulador (BB_UPSCALER=taa), sin reescalado: indice 22, al final de la lista.
$script:upscalers += ,@((T 'TAA propio (nativo)' 'Own TAA (native)'), 'taa', $true, 'native')
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
	@('AMD FSR 3 x4', 'amd-x4'),
	@((T 'Imagen x2 (tipo Lossless)' 'Image x2 (Lossless-like)'), 'image-x2'),  # (bb-imgfg-078) FG propio solo con la imagen final
	@((T 'Imagen x3 (tipo Lossless)' 'Image x3 (Lossless-like)'), 'image-x3')
)
$script:fgList = @($script:fgModes | ForEach-Object { $_[0] })
# Orden de la lista de frame generation: DLSS arriba, luego FSR. -1 = cabecera (no se puede elegir). Se guarda el indice de $fgModes.
$script:fgDisp = @(0, -1, 3, 4, 5, 6, -1, 1, 7, 8, 2)  # (launcher123) sin Imagen x2/x3 (tipo Lossless)
$script:fgHdr = @{ 1 = '---------- NVIDIA DLSS (RTX) ----------'; 6 = '---------- AMD FSR 3 ----------' }
function Fg-Labels { return @(0..($script:fgDisp.Count - 1) | ForEach-Object { $d = $_; if ($script:fgHdr.ContainsKey($d)) { $script:fgHdr[$d] } else { $script:fgModes[$script:fgDisp[$d]][0] } }) }
# Orden de la lista de reescalado: DLSS, luego FSR 4, FSR 3.1 y XeSS. Igual: se guarda el indice de $script:upscalers.
$script:upDisp = @(0, 1, -1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, -1, 17, 18, 19, 20, 21, -1, 22)
$script:upHdr = @{ 2 = '---------- NVIDIA DLSS (RTX) ----------'; 8 = '---------- AMD FSR 4 ----------'; 14 = '---------- AMD FSR 3.1 ----------'; 20 = '---------- INTEL XeSS ----------'; 26 = '---------- TAA ----------' }
function Up-ToDisp([int]$i) { $d = [array]::IndexOf($script:upDisp, $i); if ($d -lt 0) { return 0 }; return $d }
function Up-FromDisp([int]$d) { if ($d -lt 0 -or $d -ge $script:upDisp.Count) { return 0 }; $v = $script:upDisp[$d]; if ($v -lt 0) { return 0 }; return $v }
function Up-Label([int]$d) { if ($script:upHdr.ContainsKey($d)) { return $script:upHdr[$d] }; $u = $script:upscalers[$script:upDisp[$d]]; $lab = [string]$u[0]; if ($u.Count -gt 3) { $lab = "$lab  ($(Get-UpRes $u[3]))" }; return $lab }
# (bbl2) Calidad como en Demon's Souls: Bajo / Normal (= PS4) / Alto / Ultra. Cada fila = parches REALES de la
# comunidad (patches\Bloodborne.xml). Al JUGAR, Set-QualityPatches pone isEnabled de esas entradas (solo ellas).
# La etiqueta muestra lo que cambiara. Sin perillas inventadas.
# Columnas: clave ini, texto, niveles, parche esperado por nivel (regex; '' = ninguno de la familia; 'GRID' = el de la
# resolucion de salida), familia (regex), tooltip.
$script:calLevels = @((T 'Bajo' 'Low'), (T 'Normal (PS4)' 'Normal (PS4)'), (T 'Alto' 'High'), 'Ultra')
$script:quality = @(
	@('cal_lod',   (T 'Modelos (LOD)' 'Models (LOD)'), @((T 'Bajo (LOD 1)' 'Low (LOD 1)'), 'Normal (PS4)', (T 'Alto (LOD -2)' 'High (LOD -2)'), 'Ultra (LOD -2)', (T 'Ultra bajo (LOD 2)' 'Ultra low (LOD 2)')), @('Model LOD 1 \(', '', 'Model LOD -2', 'Model LOD -2', 'Model LOD 2 \('), '(?i)Model LOD', (T 'Detalle de los modelos a distancia. Alto/Ultra: LOD -2 (máximo), cuesta GPU. Ultra bajo: LOD 2 (mínimo).' 'Model detail at distance. High/Ultra: LOD -2 (highest), costs GPU. Ultra low: LOD 2 (lowest).')),
	@('cal_grid',  (T 'Rejilla de luz' 'Light grid'), @((T 'Bajo (optimizada)' 'Low (optimized)'), 'Normal (PS4)', (T 'Alto (PS4)' 'High (PS4)'), 'Ultra (PS4)', (T 'Ultra bajo (optimizada)' 'Ultra low (optimized)')), @('GRID', '', '', '', 'GRID'), '(?i)Light Grid', (T 'Bajo: menos llamadas de dibujo de luces (la versión de tu resolución de salida). Más FPS.' 'Low: fewer light-grid draw calls (the version for your output resolution). More FPS.')),
	@('cal_dynsh', (T 'Sombras dinámicas' 'Dynamic shadows'), @((T 'Bajo (sin)' 'Low (off)'), 'Normal (PS4)', (T 'Alto (PS4)' 'High (PS4)'), 'Ultra (PS4)', (T 'Ultra bajo (sin)' 'Ultra low (off)')), @('Disable Dynamic Light Shadows', '', '', '', 'Disable Dynamic Light Shadows'), '(?i)Dynamic Light Shadows', (T 'Bajo: quita las sombras de luces dinámicas (antorchas...). Mucho más FPS.' 'Low: removes dynamic light shadows (torches...). Much more FPS.')),
	@('cal_ssao',  (T 'Oclusión (SSAO)' 'Occlusion (SSAO)'), @((T 'Bajo (sin)' 'Low (off)'), 'Normal (PS4)', (T 'Alto (PS4)' 'High (PS4)'), 'Ultra (PS4)', (T 'Ultra bajo (sin)' 'Ultra low (off)')), @('Disable SSAO', '', '', '', 'Disable SSAO'), '(?i)Disable SSAO', (T 'Sombreado de contacto en rincones. Bajo: desactivado.' 'Contact shading in corners. Low: off.')),
	@('cal_dof',   (T 'Profundidad campo' 'Depth of field'), @((T 'Bajo (sin)' 'Low (off)'), 'Normal (PS4)', (T 'Alto (PS4)' 'High (PS4)'), 'Ultra (PS4)', (T 'Ultra bajo (sin)' 'Ultra low (off)')), @('Disable DoF', '', '', '', 'Disable DoF'), '(?i)Disable DoF', (T 'Desenfoque del fondo. Bajo: desactivado.' 'Background blur. Low: off.')),
	@('cal_blur',  (T 'Desenfoque mov.' 'Motion blur'), @((T 'Bajo (sin)' 'Low (off)'), 'Normal (PS4)', (T 'Alto (PS4)' 'High (PS4)'), 'Ultra (PS4)', (T 'Ultra bajo (sin)' 'Ultra low (off)')), @('Disable Motion Blur', '', '', '', 'Disable Motion Blur'), '(?i)Motion Blur', (T 'Bajo: sin desenfoque de movimiento (más FPS, imagen más limpia).' 'Low: no motion blur (more FPS, cleaner image).')),
	@('cal_ssr',   (T 'Reflejos (SSR)' 'Reflections (SSR)'), @((T 'Bajo (PS4)' 'Low (PS4)'), 'Normal (PS4)', (T 'Alto (PS4)' 'High (PS4)'), (T 'Ultra (SSR)' 'Ultra (SSR)'), (T 'Ultra bajo (PS4)' 'Ultra low (PS4)')), @('', '', '', 'Screen Space Reflections', ''), '(?i)Screen Space Reflections', (T 'Ultra: reflejos en espacio de pantalla (no estaban en PS4). Cuesta GPU.' 'Ultra: screen-space reflections (not on PS4). Costs GPU.')),
	@('cal_ca',    (T 'Aberración crom.' 'Chromatic ab.'), @((T 'Bajo (sin)' 'Low (off)'), 'Normal (PS4)', (T 'Alto (PS4)' 'High (PS4)'), 'Ultra (PS4)', (T 'Ultra bajo (sin)' 'Ultra low (off)')), @('Disable Chromatic Aberration', '', '', '', 'Disable Chromatic Aberration'), '(?i)Chromatic Aberration', (T 'Bordes de color del juego. Bajo: desactivada.' 'Colour fringing. Low: off.')),
	@('cal_aa',    (T 'Antialiasing juego' 'Game AA'), @((T 'Bajo (sin)' 'Low (off)'), 'Normal (PS4)', (T 'Alto (PS4)' 'High (PS4)'), 'Ultra (PS4)', (T 'Ultra bajo (sin)' 'Ultra low (off)')), @('Disable AA', '', '', '', 'Disable AA'), '(?i)Disable AA', (T 'AA propio del juego. Bajo: desactivado (DLSS / FSR / XeSS ya suavizan).' 'Game''s own AA. Low: off (DLSS / FSR / XeSS already smooth).'))
)
function Get-GridPatch {
	$r = $script:S.winres; if (-not $r -or $r -eq 'auto') { $r = Get-ScreenRes }
	switch ([int](Split-Res $r)[1]) { 1080 { return '^1080p Light Grid' } 1440 { return '^1440p Light Grid' } 2160 { return '^4k Light Grid' } 800 { return '^1280x800 Light Grid' } }
	return ''
}
# ---------------------------------------------------------------- MODOS (como los del lanzador de Demon's Souls)
# Cada modo fija FPS reales, frame generation, resolucion base y reescalado. Personalizado no toca nada.
# Columnas: clave, texto, fps reales (BB_FPS_CAP), FG (BB_FG), en pantalla, res. interna, fps (0 = 30, 1 = parche 60),
# reescalado (indice de $upscalers), modo de reescalado (indice de $upModes), nitidez, preset de Calidad (0 Bajo, 1 Normal, 2 Alto, 3 Ultra).
# Hoy shadPS4 hace: resolucion (config.toml), 30/60 con el parche. FG y el tope de 45 son del emulador propio (proximamente).
# (modos-052) ligero / equilibrado / ultra: FSR 4 Calidad (indice 8) en vez de FSR 1 (1, espacial: se veia pixelado), con FSR 3 FG x2 (todo AMD, sin Streamline).
$script:modes = @(
	@('ligero',      (T 'Soulsborne ligero: 30 FPS x2 = 60' 'Soulsborne light: 30 FPS x2 = 60'),            '30', 'amd-x2', 60,  '2560x1440', 0, 2, 0, 75, 1),  # (ligero) interna 1440p: DLSS nativo (AA)
	@('equilibrado', (T 'Soulsborne equilibrado: 45 FPS x2 = 90' 'Soulsborne balanced: 45 FPS x2 = 90'),     '45', 'amd-x2', 90,  '2560x1440', 1, 2, 0, 75, 1),  # (equilibrado) interna 1440p: DLSS nativo (AA)
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
function Get-FgKey { $mm = Get-Mode; if (@('ligero', 'equilibrado') -contains $mm[0] -and $mm[3]) { return $mm[3] }; $i = 0; [void][int]::TryParse("$($script:S.bbfg)", [ref]$i); return $script:fgModes[[math]::Max(0, [math]::Min($i, $script:fgModes.Count - 1))][1] }
function Get-SharpText { $v = 0; [void][int]::TryParse("$($script:S.sharp)", [ref]$v); return ([double][math]::Max(0, [math]::Min(100, $v)) / 100).ToString('0.00', [Globalization.CultureInfo]::InvariantCulture) }
# Nitidez EN VIVO: el emulador lee BB_UPSCALE_SHARPNESS al arrancar y puede releer este archivo (BB_UPSCALE_SHARPNESS_FILE)
# mientras juegas; el lanzador lo reescribe cada vez que cambias la Nitidez.
$script:sharpFile = Join-Path $root 'bb-live-sharpness.txt'
function Write-SharpLive { try { [IO.File]::WriteAllText($script:sharpFile, "BB_UPSCALE_SHARPNESS=$(Get-SharpText)`r`n", $utf8NoBom) } catch { } }
function Get-AlIdx([string]$k, [int]$max) { $i = 0; [void][int]::TryParse("$($script:S[$k])", [ref]$i); return [math]::Max(0, [math]::Min($i, $max)) }
function Test-AlForced { return @('ligero', 'equilibrado') -contains (Get-Mode)[0] }
function Is-AmdGpu { return ([string]$script:pc.Gpu -match 'AMD|Radeon') }
function Get-AlKey { if (Test-AlForced) { return 'ultra' }; $k = @('auto', 'amd2', 'normal', 'ultra', 'off')[(Get-AlIdx 'almode' 4)]; if ($k -eq 'auto' -and -not (Is-AmdGpu) -and (Get-FgKey) -like 'amd*') { return 'ultra' }; return $k }
function Get-FgEnv {
	$e = [ordered]@{}
	$fg = Get-FgKey; $e['BB_FG'] = $fg
	$e['BB_KBM_YAW_SIGN'] = '1'; $e['BB_KBM_PITCH_SIGN'] = '1'  # (launcher137) ejes del raton: el aprendizaje los ponia al reves
	$e['BB_HUD_SHARP'] = '0'
	$e['BB_SCENE_UPSCALE'] = $(if (Get-SceneUp) { '1' } else { '0' })  # (bb-fg-022) aqui: el arranque del emulador de reescalado solo pasa Get-FgEnv
	$dl = ($fg -like 'dlss*')
	# (bbl3) Anti-Lag como en Demon's Souls (antilag_modo / antilag_dlssg): auto = Anti-Lag 2 en AMD, el nuestro en las
	# demas con FSR 3 FG, Reflex en NVIDIA sin FG; Anti-Lag = nuestro normal (2 fotogramas en cola), Anti-Lag+ = ultra (1).
	# Con DLSS-G: Reflex (recomendado) o el nuestro con Reflex apagado. Ligero / Equilibrado: siempre Anti-Lag+.
	$e['BB_ANTILAG'] = Get-AlKey
	$e['BB_ANTILAG_DLSSG'] = @('reflex', 'normal', 'ultra')[(Get-AlIdx 'aldlssg' 2)]
	$e['BB_REFLEX'] = $(if ($dl -and (Get-AlIdx 'aldlssg' 2) -eq 0) { '1' } else { '0' })
	$mk = (Get-Mode)[0]
	$e['BB_PAD_LATE_MS'] = $(if ($mk -eq 'ligero') { '43' } elseif ($mk -eq 'equilibrado') { '29' } else { '' })
	$e['BB_UPSCALE_SHARPNESS'] = Get-SharpText
	$e['BB_UPSCALE_SHARPNESS_FILE'] = $script:sharpFile
	# (bb-cap-044) Tope de FPS real del juego tambien en este arranque: el del modo (30 / 45 / 60); en Personalizado con
	# frame generation, 60; sin frame generation no se pasa (como antes).
	$mf = (Get-Mode)[2]
	# (fg-ritmo-30) Con frame generation el tope sigue la opcion de FPS (30 por defecto, como en PS4); 0 solo si eliges Sin limite.
	if ($mf) { $e['BB_FPS_CAP'] = $mf } elseif ($fg -ne 'off') { $e['BB_FPS_CAP'] = @('30', '60', '0', '40')[(Get-FpsCapIdx)] }
	$e['BB_FGPACE_V2'] = '0'  # sin tope automatico con frame generation (modo seguro)
	$e['KYTY_FG_DYN_CAP'] = '0'  # dinamico: sin techo por fps reales, apunta al refresco del monitor (x4 con 50-60 reales)
	if ($script:S.mvzero -eq '1') { $e['BB_MV_UPSCALE'] = 'zero' }  # prueba: sin vectores de movimiento de camara (mvzero=1 en launcher.ini)
	# (bb-dlss5-042) DLSS 5 (Neural Rendering de NVIDIA) como en Demon's Souls (aqui: el modo seguro solo pasa Get-FgEnv): solo tras DLSS / XeSS / FSR 3.1 / FSR 4.
	$u5 = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]
	$d5 = 0; [void][int]::TryParse("$($script:S.dlss5)", [ref]$d5); $d5 = [math]::Max(0, [math]::Min($d5, 3))
	# 0.90: DLSS 5 also without upscaler (native analysis)
	$e['BB_DLSS5'] = [string]$d5
	if ($d5 -gt 0) {
		$e['BB_DLSS5_INTENSITY'] = $(if (@('baja', 'media') -contains $script:S.dlss5_intensidad) { $script:S.dlss5_intensidad } else { 'alta' })
		$e['BB_DLSS5_UNOFFICIAL'] = $(if ($script:S.dlss5_no_oficial -eq '1') { '1' } else { '0' })
		# (bb-port-096) Suave/Medio/Fuerte ahorran GPU en los bordes (foveated); ya no se fuerza la rejilla uniforme.
	}
	$e['BB_SR_CUT_NDC'] = '3'
	return $e
}
function Get-SceneUp {
	# (bb-fg-022) Reescalado de escena: marcado y DLSS / FSR 4 / FSR 3.1 / XeSS en un modo reducido.
	$u = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]
	return ($script:S.sceneup -eq '1' -and $u.Count -gt 3 -and @('dlss', 'fsr4', 'fsr3', 'xess') -contains $u[1] -and $u[3] -ne 'native')  # (bb-native-073) marcado + DLSS / FSR 4 / FSR 3.1 / XeSS en un modo reducido
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
	$e['BB_FPS_LIMIT'] = @('30', '60', 'unlocked', '60')[(Get-FpsCapIdx)]
	$e['BB_FGPACE_V2'] = '0'  # sin tope automatico de 60 con frame generation (bb-fgpace-054 off)
	if ($script:S.fsrdebug -eq '1') { $e['KYTY_FSR3_DEBUG_VIEW'] = '1' }
	if ($script:S.fsrdepth) { $e['KYTY_FSR3_DEPTH'] = $script:S.fsrdepth }
	if ($script:S.fsrhudless -eq '0') { $e['KYTY_FSR3_HUDLESS'] = '0' }
	if ($script:S.fgplayer -eq '1') { $e['BB_FSR3_PLAYER_REAL'] = '1' }  # (11-13) personaje sin estela (AMD FSR 3)  # (11-13) prueba: FSR 3 FG sin imagen sin HUD (fsrhudless=0 en launcher.ini)  # (11-12) prueba: profundidad del FSR 3 FG (inverted / normal; fsrdepth= en launcher.ini)  # (11-12) prueba: vista de depuracion del FSR 3 FG (fsrdebug=1 en launcher.ini)
	if (-not $e.Contains('BB_FPS_CAP')) { $e['BB_FPS_CAP'] = $(if ($m[2]) { $m[2] } else { @('30', '60', '0', '40')[(Get-FpsCapIdx)] }) }  # (bb-cap-044) el de Get-FgEnv manda
	$gs = Get-ModeGameSettings; if ($gs) { $e['BB_GAME_SETTINGS'] = $gs }
	# (bbl2) La calidad va por parches de shadPS4 (los activa el usuario); el emulador aun no tiene variables de calidad.
	$dp = "$($script:S.dlsspreset)".ToUpper(); $e['BB_DLSS_PRESET'] = $(if ($dp -match '^[A-O]$') { $dp } else { 'M' })  # (dlsspre) letra A-O del ini (dlsspreset=K/L/M); por defecto M (el de Demon's Souls)
	# (prueba mvsign) XeSS y FSR 3.1 emborronan al girar: signo de los vectores invertido (mvsign=1 en el ini lo devuelve al original).
	$mvs = $(if ("$($script:S.mvsign)" -eq '1') { '1' } else { '-1' }); $e['BB_XESS_MV_SIGN'] = $mvs; $e['BB_FSR3_MV_SIGN'] = $mvs
	$e['BB_VOL'] = $(if ($script:S.vol -eq '1') { '1' } else { '0' })  # (vol) niebla y rayos de luz volumetricos
	$e['BB_TEX_QUALITY'] = [string]([math]::Max(0, [math]::Min([int]$script:S.tex, 4)))  # (tex) 0 Bajo, 1 Normal, 2 Alto, 3 Ultra, 4 Ultra bajo (al final de la lista)
	$an = 3; [void][int]::TryParse("$($script:S.aniso)", [ref]$an); $e['BB_ANISO'] = @('1', '4', '8', '16')[[math]::Max(0, [math]::Min($an, 3))]  # (aniso) 1x, 4x, 8x, 16x
	$e['BB_INPUT'] = $(if ($script:S.input -eq '1') { 'kbm' } else { 'pad' })
	if ($script:S.metricas -eq '1') { $e['BB_METRICS'] = '1'; $e['BB_METRICS_OUT'] = (Join-Path $root 'logs\bb-fps.log') }  # el emulador propio escribira ahi [fps]/[lat]
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
	$rpi = 0; [void][int]::TryParse("$($script:S.respatch)", [ref]$rpi); if ($rpi -gt 0 -and $rpi -le 3) { return @('', '3840x2160', '2560x1440', '1920x1080')[$rpi] }  # (respatch) parche de resolucion elegido a mano
	$out = $script:S.winres; if (-not $out -or $out -eq 'auto') { $out = Get-ScreenRes }
		$o = Split-Res $out
	$f = 1.0; $null = switch ($mode) { 'native' { 1.0 } 'quality' { 1.5 } 'balanced' { 1.7 } 'performance' { 2.0 } 'ultraperf' { 3.0 } default { 1.5 } }
	# La interna baja segun la calidad (salida / factor), igual que las etiquetas del desplegable.
	$best = '640x360'
	$want = $o[1] / $f; $bd = [double]::MaxValue
	foreach ($r in @('640x360', '960x540', '1280x720', '1440x810', '1600x900', '1920x1080', '2560x1440', '3840x2160')) { $wh = Split-Res $r; $dd = [math]::Abs($wh[1] - $want); if ($dd -lt $bd) { $bd = $dd; $best = $r } }
	return $best
}
function Get-CurMode { $u = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]; if ($u.Count -gt 3) { return $u[3] }; return 'native' }
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
	Set-QualityPatches $userDir $g.Serial  # (bbl2b) solo parches de calidad, tambien en modo seguro
	$outRes = $script:S.winres; if (-not $outRes -or $outRes -eq 'auto') { $outRes = Get-ScreenRes }
	$pr4 = Get-UpRes (Get-CurMode)
	Set-FpsPatch $userDir
	Set-ResolutionPatch $userDir $pr4  # (10-29) sin reescalado el juego va a la resolucion elegida (4K = 4K)  # (10-29) tambien en modo seguro
	# (10-29 b) La resolucion interna de shadPS4 (internalScreenWidth/Height) = la resolucion elegida: antes se quedaba en
	# 2560x1440 y el juego, aunque saliera a 4K, se veia a 1440. Solo FSR 1 (shadPS4) usa una interna menor.
	try {
		$upK = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)][1]
		if ($upK -ne 'fsr1') {
			$fr = Split-Res $outRes; $ii = Split-Res (Get-UpRes (Get-CurMode))
			$fw = @(Write-ShadConfig $userDir ([ordered]@{ winW = $fr[0]; winH = $fr[1]; intW = $ii[0]; intH = $ii[1]; fsr = $false }) 'both')
			Write-Log "Resolucion forzada: ventana $outRes, interna $($ii[0])x$($ii[1]) ($($fw -join ', '))"
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
		$upExe = (Join-Path $PSScriptRoot 'shadPS4.exe')
		# Si hay una version nueva en trozos (shadPS4.part0..), se une sola antes de jugar.
		$pdir = Split-Path -Parent $upExe
		# (bb-port-094) Shaders de FSR 4 en zip: se descomprimen solos la primera vez.
		$fz = Join-Path $pdir 'fsr4_shaders.zip'; if ((Test-Path -LiteralPath $fz) -and -not (Test-Path -LiteralPath (Join-Path $pdir 'fsr4_shaders'))) { try { Expand-Archive -LiteralPath $fz -DestinationPath $pdir -Force; Write-Log 'fsr4_shaders descomprimido' } catch { Write-Log "No se pudo descomprimir fsr4_shaders.zip: $_" } }
		if (Test-Path -LiteralPath (Join-Path $pdir 'shadPS4.part0')) {
			try {
				$parts = @(Get-ChildItem -LiteralPath $pdir -Filter 'shadPS4.part*' | Sort-Object { [int]($_.Name -replace '\D', '') })
				try { [void](Backup-EmuFiles $upExe 'shadPS4.part') } catch { Write-Log "Copia anterior: $_" }  # (launcher132)
				$tmp = Join-Path $pdir 'shadPS4.nuevo'
				$fs = [System.IO.File]::Create($tmp); foreach ($pp in $parts) { $b = [System.IO.File]::ReadAllBytes($pp.FullName); $fs.Write($b, 0, $b.Length) }; $fs.Close()
				if (Test-Path -LiteralPath $upExe) { Move-Item -LiteralPath $upExe -Destination (Join-Path $pdir 'shadPS4-anterior.exe') -Force }
				Move-Item -LiteralPath $tmp -Destination $upExe -Force; $parts | Remove-Item -Force
				Write-Log "Emulador de reescalado actualizado: MD5 $((Get-FileHash -LiteralPath $upExe -Algorithm MD5).Hash)"
			} catch { Write-Log "No se pudo unir la version nueva: $_" }
		}
		if ((@('dlss', 'fsr3', 'fsr4', 'xess') -contains $upKey -or [int]("0$($script:S.dlss5)") -gt 0) -and (Test-Path -LiteralPath $upExe)) {  # (bb-port-096) DLSS 5 tambien sin reescalador
			$emu = $upExe
			$env:BB_UPSCALER = $upKey
			if ($upKey -eq 'dlss') {
				# (ui-1030) Como Probar-DLSS.cmd: sin el guardia de la ultima prueba y con reintento.
				$guard = Join-Path $root 'bb_upscale_guard.txt'; if (Test-Path -LiteralPath $guard) { Remove-Item -LiteralPath $guard -Force -ErrorAction SilentlyContinue }
				$env:BB_UPSCALE_RETRY = '1'
			}
			# (obs-hook) La capa Vulkan VK_LAYER_WEOBS_HOOK (C:\ProgramData\obs-studio-hook, de Wallpaper Engine/OBS) cerraba el juego
			# al crear el dispositivo. El cargador de Vulkan la desactiva con VK_LOADER_LAYERS_DISABLE; se quita el guardia viejo.
			$env:VK_LOADER_LAYERS_DISABLE = 'VK_LAYER_WEOBS_HOOK'
			# Sin esa capa ya no hay motivo para el guardia: tras un cierre el emulador dejaba FSR 4 / XeSS en nativo para siempre.
			$gh = Join-Path $root 'bb_upscale_guard.txt'; if (Test-Path -LiteralPath $gh) { Remove-Item -LiteralPath $gh -Force -ErrorAction SilentlyContinue }
			$env:BB_UPSCALE_RETRY = '1'
			$upE = $script:upscalers[[math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)]
			$env:BB_UPSCALE_MODE = $(if ($upE.Count -gt 3) { $upE[3] } else { 'quality' })
			$env:BB_UPSCALE_SHARPNESS = ([double]$script:S.sharp / 100).ToString('0.00', [Globalization.CultureInfo]::InvariantCulture)
			$upRes = Get-UpRes $env:BB_UPSCALE_MODE
			$upRes = $outRes; if ($false) { Write-Log "Reescalado de escena (BB_SCENE_UPSCALE=1): el juego a la resolucion de salida $outRes" }  # (bb-fg-022) intres = winres
			Set-ResolutionPatch $userDir $upRes
			Write-Log "Reescalado: $upKey $($env:BB_UPSCALE_MODE) con $upExe (interna $upRes)"
		}
		# (bb-launcher-022) Frame generation + Anti-Lag/Reflex + nitidez (todos los reescaladores) como variables BB_*.
		$fge = Get-FgEnv
		# (prueba-base) Sin reescalado y sin frame generation tambien va el fork (mismas BB_*, BB_FG=off), no el de serie: linea base justa.
		if ($emu -ne $upExe -and (Test-Path -LiteralPath $upExe)) { $emu = $upExe; Write-Log "Fork ($($fge['BB_FG'])): con $upExe" }
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
	$wr = Split-Res $win; $ir = Split-Res (Get-UpRes (Get-CurMode))
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
	try { Set-GameLanguage $userDir } catch { Write-Log "Idioma del juego: $_" }  # (launcher112) solo la clave de idioma
	if ($written.Count) { Write-Log "Config escrita: $($written -join ', ') :: $(($vals.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')" }
	elseif (-not (Test-Path -LiteralPath (Join-Path $userDir 'config.toml')) -and -not (Test-Path -LiteralPath (Join-Path $userDir 'config.json'))) {
		Write-Log 'Primera vez: shadPS4 aun no tiene config; se crea al arrancar. Los ajustes se aplican desde la siguiente partida.'
		if (-not $script:headless -and (Get-Setting 'firstrunwarned') -ne '1') { Show-Msg (T 'Primera vez con este emulador: shadPS4 crea su configuración al arrancar. Pantalla, reescalado y controles se aplicarán desde la próxima vez que pulses JUGAR.' 'First run with this emulator: shadPS4 creates its config on start. Display, upscaling and controls apply from the next time you press PLAY.') 'Bloodborne' 'Information' | Out-Null; Set-Setting 'firstrunwarned' '1' }
	}
	# 2) Parches: solo el "Resolution Patch" de la resolucion interna (Set-ResolutionPatch); el resto nunca se escribe.
	# (11-02) Fuera del modo seguro tambien: con DLSS / FSR 4 / FSR 3.1 / XeSS o frame generation se arranca el
	# shadPS4 de bb-port-medicion (el de serie ignora las BB_*) y el juego va a la resolucion interna del modo.
	$upExe2 = (Join-Path $PSScriptRoot 'shadPS4.exe')
	$upKey2 = $up[1]
	if ((@('dlss', 'fsr3', 'fsr4', 'xess') -contains $upKey2 -or (Get-FgEnv)['BB_FG'] -ne 'off' -or [int]("0$($script:S.dlss5)") -gt 0) -and (Test-Path -LiteralPath $upExe2)) {  # (bb-port-096) DLSS 5 tambien sin reescalador
		$pdir2 = Split-Path -Parent $upExe2
		$fz2 = Join-Path $pdir2 'fsr4_shaders.zip'; if ((Test-Path -LiteralPath $fz2) -and -not (Test-Path -LiteralPath (Join-Path $pdir2 'fsr4_shaders'))) { try { Expand-Archive -LiteralPath $fz2 -DestinationPath $pdir2 -Force; Write-Log 'fsr4_shaders descomprimido' } catch { Write-Log "No se pudo descomprimir fsr4_shaders.zip: $_" } }
		if (Test-Path -LiteralPath (Join-Path $pdir2 'shadPS4.part0')) {
			try {
				$parts2 = @(Get-ChildItem -LiteralPath $pdir2 -Filter 'shadPS4.part*' | Sort-Object { [int]($_.Name -replace '\D', '') })
				try { [void](Backup-EmuFiles $upExe2 'shadPS4.part') } catch { Write-Log "Copia anterior: $_" }  # (launcher132)
				$tmp2 = Join-Path $pdir2 'shadPS4.nuevo'
				$fs2 = [System.IO.File]::Create($tmp2); foreach ($pp in $parts2) { $b2 = [System.IO.File]::ReadAllBytes($pp.FullName); $fs2.Write($b2, 0, $b2.Length) }; $fs2.Close()
				Move-Item -LiteralPath $upExe2 -Destination (Join-Path $pdir2 'shadPS4-anterior.exe') -Force
				Move-Item -LiteralPath $tmp2 -Destination $upExe2 -Force; $parts2 | Remove-Item -Force
				Write-Log "Emulador de reescalado actualizado: MD5 $((Get-FileHash -LiteralPath $upExe2 -Algorithm MD5).Hash)"
			} catch { Write-Log "No se pudo unir la version nueva: $_" }
		}
		$emu = $upExe2
		$env:VK_LOADER_LAYERS_DISABLE = 'VK_LAYER_WEOBS_HOOK'  # (obs-hook) capa que cerraba el juego al crear el dispositivo
		if (@('dlss', 'fsr3', 'fsr4', 'xess') -contains $upKey2) {
			$m2 = $(if ($up.Count -gt 3) { $up[3] } else { $script:upModeKeys[[math]::Min([int]$script:S.upmode, 4)] })
			$upRes2 = Get-UpRes $m2
			if (Get-SceneUp) { $upRes2 = $outRes }
			Set-ResolutionPatch $userDir $upRes2
			if ($upKey2 -eq 'dlss') { $g2 = Join-Path $root 'bb_upscale_guard.txt'; if (Test-Path -LiteralPath $g2) { Remove-Item -LiteralPath $g2 -Force -ErrorAction SilentlyContinue } }
			Write-Log "Reescalado: $upKey2 $m2 con $upExe2 (interna $upRes2)"
		} else { Write-Log "Frame generation: con $upExe2" }
	}
	# 3) Variables BB_* para el emulador propio (shadPS4 de serie las ignora).
	$bb = Get-BbEnv
	$bb['BB_OBJMV_VERTICES_MI'] = '8'
	# (0.53) Texturas: LOD extra log2(interna/salida) con reescalado, como piden DLSS y FSR.
	$ir = $(if ($upRes2) { "$upRes2" } elseif ($upRes) { "$upRes" } else { '' }); $upOn = (@('dlss', 'fsr3', 'fsr4', 'xess') -contains $upKey2) -or (@('dlss', 'fsr3', 'fsr4', 'xess') -contains $upKey)
	if ($upOn -and $ir -and $outRes) { $ih = 0; $oh = 0; [void][int]::TryParse(("$ir" -split 'x')[1], [ref]$ih); [void][int]::TryParse(("$outRes" -split 'x')[1], [ref]$oh); if ($ih -gt 0 -and $oh -gt $ih) { $bb['BB_TEX_LOD_EXTRA'] = ([math]::Log($ih / $oh, 2)).ToString('0.00', [Globalization.CultureInfo]::InvariantCulture) } }
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
	$me = $(if ($PSCommandPath) { $PSCommandPath } else { Join-Path $root 'Launcher-bloodborne.ps1' })  # (launcher112) un solo script; el idioma sale del ini / Windows
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
	if (@('es', 'en') -contains $script:langArg) { $launch += " -Lang $($script:langArg)" }
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
Ensure-PortLinks

# ---------------------------------------------------------------- (launcher112) aviso que hay que aceptar
# Solo la primera vez que se abre el lanzador (aviso_ok=1 en el ini).
$script:avisoVer = '1'
function Show-Notice {
	$d = New-Object System.Windows.Forms.Form
	$d.Text = 'Bloodborne'; $d.FormBorderStyle = 'None'; $d.StartPosition = 'CenterScreen'; $d.ShowInTaskbar = $true; $d.TopMost = $true
	$d.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22); $d.ClientSize = New-Object System.Drawing.Size(640, 520); $d.KeyPreview = $true
	try { $rp = New-Object System.Drawing.Drawing2D.GraphicsPath; $r = 18; $w = 640; $h = 520; $rp.AddArc(0, 0, $r, $r, 180, 90); $rp.AddArc($w - $r, 0, $r, $r, 270, 90); $rp.AddArc($w - $r, $h - $r, $r, $r, 0, 90); $rp.AddArc(0, $h - $r, $r, $r, 90, 90); $rp.CloseFigure(); $d.Region = New-Object System.Drawing.Region($rp) } catch { }
	$gold = [System.Drawing.Color]::FromArgb(222, 196, 140); $ink = [System.Drawing.Color]::FromArgb(236, 232, 224)
	$d.Add_Paint({ param($s, $e) $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(150, 222, 196, 140), 2); $e.Graphics.DrawRectangle($pen, 1, 1, $s.ClientSize.Width - 3, $s.ClientSize.Height - 3); $pen.Dispose()
		$ln = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(110, 222, 196, 140), 1); $e.Graphics.DrawLine($ln, 28, 66, $s.ClientSize.Width - 28, 66); $ln.Dispose() })
	$tl = New-Object System.Windows.Forms.Label; $tl.Text = (T 'AVISO' 'NOTICE'); $tl.ForeColor = $gold; $tl.BackColor = $d.BackColor
	$tl.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 16.5, [System.Drawing.FontStyle]::Bold); $tl.Location = New-Object System.Drawing.Point(26, 20); $tl.Size = New-Object System.Drawing.Size(588, 40)
	$tx = New-Object System.Windows.Forms.Label; $tx.ForeColor = $ink; $tx.BackColor = $d.BackColor; $tx.Font = New-Object System.Drawing.Font('Segoe UI', 10.5)
	$tx.Location = New-Object System.Drawing.Point(28, 80); $tx.Size = New-Object System.Drawing.Size(584, 360)
	$tx.Text = (T ("• Este es un port no oficial hecho por fans que funciona sobre una versión modificada del emulador shadPS4. No está afiliado con Sony Interactive Entertainment ni con FromSoftware.`n`n" +
		"• Debes tener el juego original y hacer tú mismo el volcado (dump) de tu copia.`n`n" +
		"• La ÚNICA versión soportada es Bloodborne CUSA03173 (Europa) con la actualización 1.09. El parche de resolución, el reescalado y los arreglos están hechos para ese ejecutable. Otras regiones o versiones (CUSA00900, CUSA00207, CUSA01363, CUSA00208 u otras versiones del parche) pueden no arrancar o verse mal.`n`n" +
		"• El DLC The Old Hunters está soportado si está instalado junto a la 1.09.`n`n" +
		"• Úsalo bajo tu propia responsabilidad y guarda copias de seguridad de tus partidas.") `
		("• This is an unofficial fan port running on a modified build of the shadPS4 emulator. It is not affiliated with Sony Interactive Entertainment or FromSoftware.`n`n" +
		"• You must own the game and dump your own copy yourself.`n`n" +
		"• The ONLY supported version is Bloodborne CUSA03173 (Europe) with update 1.09. The resolution patch, upscaling and fixes are made for that executable. Other regions or versions (CUSA00900, CUSA00207, CUSA01363, CUSA00208 or other patch versions) may not start or may break visuals.`n`n" +
		"• The Old Hunters DLC is supported if installed with 1.09.`n`n" +
		"• Use at your own risk and keep backups of your saves."))
	$bOk = New-Object System.Windows.Forms.Button; $bOk.Text = (T 'Aceptar' 'Accept'); $bOk.DialogResult = 'OK'
	$bNo = New-Object System.Windows.Forms.Button; $bNo.Text = (T 'Salir' 'Exit'); $bNo.DialogResult = 'Cancel'
	foreach ($b in @($bOk, $bNo)) { $b.FlatStyle = 'Flat'; $b.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 12, [System.Drawing.FontStyle]::Bold); $b.Size = New-Object System.Drawing.Size(180, 44); $b.Cursor = [System.Windows.Forms.Cursors]::Hand }
	$bOk.Location = New-Object System.Drawing.Point(244, 454); $bNo.Location = New-Object System.Drawing.Point(434, 454)
	$bOk.BackColor = [System.Drawing.Color]::FromArgb(150, 14, 14); $bOk.ForeColor = [System.Drawing.Color]::White; $bOk.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(90, 8, 8); $bOk.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(185, 24, 24); $bOk.FlatAppearance.MouseDownBackColor = [System.Drawing.Color]::FromArgb(120, 10, 10)
	$bNo.BackColor = [System.Drawing.Color]::FromArgb(24, 28, 38); $bNo.ForeColor = $ink; $bNo.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(110, 110, 100); $bNo.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(48, 44, 40); $bNo.FlatAppearance.MouseDownBackColor = [System.Drawing.Color]::FromArgb(30, 28, 26)
	$d.Controls.AddRange(@($tl, $tx, $bOk, $bNo)); $d.AcceptButton = $bOk; $d.CancelButton = $bNo
	# Arrastrar el aviso por cualquier sitio vacio; aparece en fundido (200 ms).
	$d.Add_MouseDown({ param($s, $e) if ($e.Button -eq 'Left') { $script:nDrag = [System.Windows.Forms.Cursor]::Position; $script:nLoc = $s.Location } })
	$d.Add_MouseMove({ param($s, $e) if ($e.Button -eq 'Left' -and $script:nDrag) { $p = [System.Windows.Forms.Cursor]::Position; $s.Location = New-Object System.Drawing.Point(($script:nLoc.X + $p.X - $script:nDrag.X), ($script:nLoc.Y + $p.Y - $script:nDrag.Y)) } })
	$d.Opacity = $(if ($reduceMotion) { 1 } else { 0.01 })
	$ft = New-Object System.Windows.Forms.Timer; $ft.Interval = 15; $script:nT0 = 0
	$ft.Add_Tick({ $t = [math]::Min(1.0, ([Environment]::TickCount - $script:nT0) / 200.0); $d.Opacity = [math]::Max(0.01, 1 - [math]::Pow(1 - $t, 3)); if ($t -ge 1) { $d.Opacity = 1; $ft.Stop() } })
	$d.Add_Shown({ $script:nT0 = [Environment]::TickCount; if (-not $reduceMotion) { $ft.Start() }; $bOk.Focus() })
	$res = $d.ShowDialog(); $ft.Stop(); $ft.Dispose(); $d.Dispose()
	return ($res -eq [System.Windows.Forms.DialogResult]::OK)
}
if ("$($script:S.aviso_ok)" -ne $script:avisoVer) {
	[System.Windows.Forms.Application]::EnableVisualStyles()
	if (-not (Show-Notice)) { Write-Log 'Aviso no aceptado: lanzador cerrado'; return }
	Set-Setting 'aviso_ok' $script:avisoVer; Write-Log "Aviso aceptado (v$($script:avisoVer))"
}
# (launcher124) -Steam: solo anade Bloodborne a Steam (Anadir-a-Steam.cmd) y sale.
if ($Steam) {
	if ([System.Windows.Forms.MessageBox]::Show((T 'Se cerrará Steam un momento para añadir "Bloodborne" a tu biblioteca (con imágenes). ¿Seguir?' 'Steam will close for a moment to add "Bloodborne" to your library (with artwork). Continue?'), 'Steam', 'YesNo', 'Question') -eq 'Yes') {
		try { $r = Add-ToSteam } catch { $r = "$_"; Write-Log "Steam: $_" }
		[void][System.Windows.Forms.MessageBox]::Show($r, 'Steam', 'OK', 'Information')
	}
	exit
}
# (launcher112) inicio_directo=1 (tambien lo cambia el menu F1 del emulador): arranca el juego sin ventana, igual que -Play.
# Para abrir el lanzador igualmente: mantener Mayus al abrirlo o pasar -menu.
if ("$($script:S.inicio_directo)" -eq '1' -and -not $Menu -and -not $Play) {
	$shift = $false; try { $shift = (([System.Windows.Forms.Control]::ModifierKeys -band [System.Windows.Forms.Keys]::Shift) -ne 0) } catch { }
	if ($shift) { Write-Log 'inicio_directo: Mayus pulsada, se abre el lanzador' } else { Write-Log 'inicio_directo=1: JUGAR directo'; $Play = [switch]$true }
}
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
# (launcher113) Fondo en una funcion (se rehace al cambiar de tamano). Guarda la escala y el recorte para calcular
# donde queda el titulo "Bloodborne PC" en la ventana (fraccion de la imagen: x 470-1200, y 20-260 de 1671x941).
# (launcher114) Las zonas salen de $bgLayout (arriba del todo).
$script:bgSrc = $null; $script:titleR = [System.Drawing.Rectangle]::Empty; $script:hunterR = [System.Drawing.Rectangle]::Empty
function Map-BgRect($z, [double]$iw, [double]$ih, [double]$sx, [double]$kx, [double]$ky) {
	$x0 = ($iw * ($z.X0 / $bgLayout.W) - $sx) * $kx; $x1 = ($iw * ($z.X1 / $bgLayout.W) - $sx) * $kx
	$y0 = ($ih * ($z.Y0 / $bgLayout.H)) * $ky; $y1 = ($ih * ($z.Y1 / $bgLayout.H)) * $ky
	return (New-Object System.Drawing.Rectangle([int][math]::Floor($x0), [int][math]::Floor($y0), [int][math]::Ceiling($x1 - $x0), [int][math]::Ceiling($y1 - $y0)))
}
function Build-Bg([int]$bw, [int]$bh) {
	$script:titleR = Map-BgRect $bgLayout.Title $bgLayout.W $bgLayout.H 0 ($bw / [double]$bgLayout.W) ($bh / [double]$bgLayout.H)
	$script:hunterR = Map-BgRect $bgLayout.Hunter $bgLayout.W $bgLayout.H 0 ($bw / [double]$bgLayout.W) ($bh / [double]$bgLayout.H)
	if (-not $fondo -or $bw -lt 16 -or $bh -lt 16) { return }
	try {
		# Fondo "cover" (sin deformar), anclado arriba para no cortar el titulo.
		$img = [System.Drawing.Image]::FromFile($fondo)
		$scale = [math]::Max($bw / $img.Width, $bh / $img.Height)
		$sw = [int][math]::Round($bw / $scale); $sh = [int][math]::Round($bh / $scale)
		# Recorte centrado, pero sin cortar el titulo (esta a la izquierda): el borde izquierdo no pasa de su X0 - 2 %.
		$sx = [int](($img.Width - $sw) / 2); $sxMax = [int]([math]::Max(0, $img.Width * (($bgLayout.Title.X0 / $bgLayout.W) - 0.02)))
		$sx = [math]::Max(0, [math]::Min($sx, $sxMax))
		$src = New-Object System.Drawing.Rectangle($sx, 0, $sw, $sh)
		$bmp = New-Object System.Drawing.Bitmap($bw, $bh, [System.Drawing.Imaging.PixelFormat]::Format32bppPArgb)  # PArgb: copia mas rapida por fotograma
		$gr = [System.Drawing.Graphics]::FromImage($bmp); $gr.InterpolationMode = 'HighQualityBicubic'
		$gr.DrawImage($img, (New-Object System.Drawing.Rectangle(0, 0, $bw, $bh)), $src, [System.Drawing.GraphicsUnit]::Pixel)
		# Vinyeta suave fija (bordes oscuros): no se anima, no cuesta nada por fotograma.
		try {
			$vp = New-Object System.Drawing.Drawing2D.GraphicsPath; $vp.AddEllipse([int](-$bw * 0.25), [int](-$bh * 0.3), [int]($bw * 1.5), [int]($bh * 1.6))
			$vb = New-Object System.Drawing.Drawing2D.PathGradientBrush($vp); $vb.CenterColor = [System.Drawing.Color]::FromArgb(0, 0, 0, 0)
			$vb.SurroundColors = [System.Drawing.Color[]]@([System.Drawing.Color]::FromArgb(150, 0, 0, 0)); $vb.FocusScales = New-Object System.Drawing.PointF(0.55, 0.5)
			$gr.FillRectangle($vb, 0, 0, $bw, $bh); $vb.Dispose(); $vp.Dispose()
		} catch { }
		$kx = $bw / [double]$sw; $ky = $bh / [double]$sh
		$script:titleR = Map-BgRect $bgLayout.Title $img.Width $img.Height $src.X $kx $ky
		$script:hunterR = Map-BgRect $bgLayout.Hunter $img.Width $img.Height $src.X $kx $ky
		$gr.Dispose(); $img.Dispose()
		$old = $script:bgBmp; $script:bgBmp = $bmp; if ($old) { $old.Dispose() }
	} catch { Write-Log "Fondo ilegible: $_" }
}
# El fondo lo pinta Paint (no BackgroundImage): permite el zoom lento de entrada y las brasas. Los controles
# semitransparentes copian este mismo dibujo (PaintParent), asi que todo cuadra.
$script:bgZoom = 1.0; $script:embers = $null; $script:fog = $null
$form.Add_Paint({ param($s, $e)
	$g = $e.Graphics
	if ($script:bgBmp) {
		if ($script:bgZoom -gt 1.0001) {
			$z = $script:bgZoom; $tr = $script:titleR; $cx = $tr.X + $tr.Width / 2.0; $cy = $tr.Y + $tr.Height / 2.0  # zoom centrado en el titulo: no se mueve
			$g.InterpolationMode = 'Bilinear'
			$g.DrawImage($script:bgBmp, (New-Object System.Drawing.RectangleF([float]($cx - $cx * $z), [float]($cy - $cy * $z), [float]($script:bgBmp.Width * $z), [float]($script:bgBmp.Height * $z))))
		} else {
			$cr = $e.ClipRectangle; $g.DrawImage($script:bgBmp, $cr, $cr, [System.Drawing.GraphicsUnit]::Pixel)
		}
	}
	if ($script:fog) { $script:fog.Draw($g) }
	if ($script:embers) { $script:embers.Draw($g) }
})
Build-Bg $W $H
if (-not $fondo) { Write-Log 'Sin fondo (fondo-bloodborne2.png no esta junto al lanzador)' }
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
$smallFont  = New-Object System.Drawing.Font('Segoe UI', 10.5)
$boldSmall  = New-Object System.Drawing.Font('Segoe UI Semibold', 10.5, [System.Drawing.FontStyle]::Bold)
$tinyFont   = New-Object System.Drawing.Font('Segoe UI', 9)
$fText  = $smallFont
$fSmall = New-Object System.Drawing.Font('Segoe UI', 10.5)
$fBtn   = New-Object System.Drawing.Font('Segoe UI Semibold', 12, [System.Drawing.FontStyle]::Bold)
$fBar   = New-Object System.Drawing.Font('Segoe UI Semibold', 12, [System.Drawing.FontStyle]::Bold)
# Titulos de apartado: Segoe UI Semibold como los de Demon's Souls (antes Georgia). 13.5 pt = 18 px a 100 %; multiplos de 0.75 pt = pixeles enteros.
$fGroup = New-Object System.Drawing.Font('Segoe UI Semibold', 13.5, [System.Drawing.FontStyle]::Bold)
# Etiquetas de fila: seminegrita algo menor y en marfil apagado; el valor (desplegable) en blanco roto.
$fLabel = New-Object System.Drawing.Font('Segoe UI Semibold', 10.5)
$lblCol = [System.Drawing.Color]::FromArgb(200, 194, 182)
$goldHi = [System.Drawing.Color]::FromArgb(246, 222, 168)
# Letra en pixeles enteros: tamanos en pt multiplos de 1,5 (= 2 px), enteros al 100 % y al 150 %.
function Snap-Px([double]$pt) { return [float]([math]::Max(7.5, [math]::Round($pt / 1.5) * 1.5)) }
$tips = New-Object System.Windows.Forms.ToolTip; $tips.AutoPopDelay = 20000; $tips.InitialDelay = 400; $tips.ReshowDelay = 100

function New-Label([string]$text, $x, $y, $w, $h = 24, $color = $ink, $font = $fText) {
	$l = New-Object System.Windows.Forms.Label
	$l.Text = $text; $l.Font = $font; $l.ForeColor = $color; $l.BackColor = [System.Drawing.Color]::Transparent
	$l.Location = New-Object System.Drawing.Point($x, $y); $l.Size = New-Object System.Drawing.Size($w, $h); $l.TextAlign = 'MiddleLeft'
	return $l
}
function New-Btn([string]$text, $x, $y, $w, $h, [bool]$primary = $false) {
	$b = New-Object BbButton  # (launcher112) redondeado, hover y pulsado animados
	$b.Text = $text; $b.Font = $fBtn; $b.FlatStyle = 'Flat'; $b.FlatAppearance.BorderSize = 1; $b.Cursor = [System.Windows.Forms.Cursors]::Hand
	$b.Location = New-Object System.Drawing.Point($x, $y); $b.Size = New-Object System.Drawing.Size($w, $h)
	if ($primary) {
		$b.BackColor = [System.Drawing.Color]::FromArgb(255, 176, 140, 70); $b.ForeColor = [System.Drawing.Color]::FromArgb(20, 16, 10)
		$b.FlatAppearance.BorderColor = $gold
		$b.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(240, 206, 170, 96); $b.FlatAppearance.MouseDownBackColor = [System.Drawing.Color]::FromArgb(240, 150, 116, 56)
	} else {
		$b.BackColor = [System.Drawing.Color]::FromArgb(255, 24, 28, 38); $b.ForeColor = $ink
		$b.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(110, 110, 100)
		$b.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(255, 48, 44, 40); $b.FlatAppearance.MouseDownBackColor = [System.Drawing.Color]::FromArgb(255, 30, 28, 26)
	}
	return $b
}
# (wheel) Los desplegables no cambian de valor con la rueda: la rueda sigue desplazando la columna.
Add-Type -ReferencedAssemblies System.Windows.Forms,System.Drawing -TypeDefinition @"
using System; using System.Drawing; using System.Windows.Forms; using System.Reflection;
public class NoWheelCombo : ComboBox {
  // La rueda del raton llega al control nativo (ComboBox) antes que al OnMouseWheel: aqui se intercepta
  // y, si el desplegable esta cerrado, se entrega al panel (que desplaza la columna) sin tocar la opcion.
  protected override void WndProc(ref Message m) {
    if (m.Msg == 0x020A && !DroppedDown) {
      int d = (short)((m.WParam.ToInt64() >> 16) & 0xffff);
      Control p = Parent;
      if (p != null) { MethodInfo mi = typeof(Control).GetMethod("OnMouseWheel", BindingFlags.Instance | BindingFlags.NonPublic); if (mi != null) mi.Invoke(p, new object[] { new MouseEventArgs(MouseButtons.None, 0, 0, 0, d) }); }
      return;
    }
    base.WndProc(ref m);
  }
  protected override void OnMouseWheel(MouseEventArgs e) {
    if (DroppedDown) { base.OnMouseWheel(e); return; }  // la rueda cambia la opcion solo con el desplegable abierto
    Control p = Parent;
    if (p != null) {
      MethodInfo m = typeof(Control).GetMethod("OnMouseWheel", BindingFlags.Instance | BindingFlags.NonPublic);
      if (m != null) m.Invoke(p, new object[] { e });
    }
  }
}
"@
# (launcher112) Botones redondeados con transicion suave (hover / pulsado) y barra de desplazamiento fina dibujada.
if (-not ('BbButton' -as [type])) {
Add-Type -ReferencedAssemblies System.Windows.Forms,System.Drawing -TypeDefinition @"
using System; using System.Collections.Generic; using System.Drawing; using System.Drawing.Drawing2D; using System.Windows.Forms;
public static class BbGfx {
  [System.Runtime.InteropServices.DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr h, int msg, IntPtr w, IntPtr l);
  [System.Runtime.InteropServices.DllImport("user32.dll")] static extern bool RedrawWindow(IntPtr h, IntPtr rc, IntPtr rgn, uint flags);
  // (launcher119) Mueve un contenedor hijo (la columna desplazable) con el repintado congelado y luego un unico
  // repintado sincrono de la vista y sus hijos: sin parpadeo ni fotogramas a medio dibujar al desplazar.
  public static void MoveTop(Control host, Control child, int top) {
    if (child.Top == top) return;
    if (host == null || !host.IsHandleCreated || !host.Visible) { child.Top = top; return; }
    SendMessage(host.Handle, 0x000B, IntPtr.Zero, IntPtr.Zero);
    try { child.Top = top; }
    finally {
      SendMessage(host.Handle, 0x000B, new IntPtr(1), IntPtr.Zero);
      RedrawWindow(host.Handle, IntPtr.Zero, IntPtr.Zero, 0x0001 | 0x0004 | 0x0080 | 0x0100);  // INVALIDATE | ERASE | ALLCHILDREN | UPDATENOW
    }
  }
  public static GraphicsPath Round(RectangleF r, float rad) {
    GraphicsPath p = new GraphicsPath(); float d = Math.Max(1f, Math.Min(rad * 2f, Math.Min(r.Width, r.Height)));
    p.AddArc(r.X, r.Y, d, d, 180, 90); p.AddArc(r.Right - d, r.Y, d, d, 270, 90);
    p.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90); p.AddArc(r.X, r.Bottom - d, d, d, 90, 90); p.CloseFigure(); return p;
  }
  public static Color Lerp(Color a, Color b, float t) {
    if (t <= 0f) return a; if (t >= 1f) return b;
    return Color.FromArgb((int)(a.A + (b.A - a.A) * t), (int)(a.R + (b.R - a.R) * t), (int)(a.G + (b.G - a.G) * t), (int)(a.B + (b.B - a.B) * t));
  }
  public static Color Shade(Color c, float f) { return Color.FromArgb(c.A, Math.Min(255, (int)(c.R * f)), Math.Min(255, (int)(c.G * f)), Math.Min(255, (int)(c.B * f))); }
  // (launcher115) Fondo del padre bajo el control, sin pintar nunca el contenido de otros controles: el formulario
  // pinta solo su fondo (imagen, niebla, brasas); cada contenedor intermedio solo su color (recortado a su Region,
  // p. ej. la tarjeta redondeada) y su propio adorno (Paint del propio contenedor). Sin el camino de transparencia
  // del sistema (que dibujaba Trofeos dentro de Actualizar / Opciones).
  public static void PaintParent(Control c, PaintEventArgs e) {
    if (c.Parent == null) { e.Graphics.Clear(SystemColors.Control); return; }
    List<Control> chain = new List<Control>(); List<Point> offs = new List<Point>();
    int ox = c.Left, oy = c.Top;
    for (Control p = c.Parent; p != null; p = p.Parent) {
      chain.Add(p); offs.Add(new Point(ox, oy));
      if (p is Form || p.Parent == null) break;
      ox += p.Left; oy += p.Top;
    }
    Graphics g = e.Graphics;
    for (int i = chain.Count - 1; i >= 0; i--) {
      Control a = chain[i]; Point o = offs[i];
      GraphicsState st = g.Save();
      try {
        g.TranslateTransform(-o.X, -o.Y);
        Rectangle r = new Rectangle(o.X, o.Y, c.Width, c.Height);
        if (i == chain.Count - 1) { BbHelper.Bg(a, new PaintEventArgs(g, r)); continue; }
        g.IntersectClip(new Rectangle(0, 0, a.Width, a.Height));
        if (a.Region != null) g.SetClip(a.Region, CombineMode.Intersect);
        Color bc = a.BackColor;
        if (bc.A > 0) { using (SolidBrush b = new SolidBrush(bc)) g.FillRectangle(b, r); }
        BbHelper.Deco(a, new PaintEventArgs(g, r));
      } finally { g.Restore(st); }
    }
  }
}
public class BbHelper : Control { static BbHelper inst = new BbHelper();
  public static void Bg(Control p, PaintEventArgs pe) { inst.Do(p, pe); }
  public static void Deco(Control p, PaintEventArgs pe) { inst.InvokePaint(p, pe); }
  void Do(Control p, PaintEventArgs pe) { InvokePaintBackground(p, pe); InvokePaint(p, pe); } }
// (launcher114) Brasas y ceniza: tamano, alfa, deriva y vaiven distintos; halo suave. Solo se invalida la caja de cada una.
public class BbEmbers {
  float[] x, y, vx, vy, life, max, size, ph, fq, amp, alpha; bool[] ash; int n; Random rnd = new Random(1871);
  public Rectangle Zone; public Rectangle Avoid; float t;
  public BbEmbers(int count) {
    n = Math.Max(1, count); x = new float[n]; y = new float[n]; vx = new float[n]; vy = new float[n]; life = new float[n]; max = new float[n];
    size = new float[n]; ph = new float[n]; fq = new float[n]; amp = new float[n]; alpha = new float[n]; ash = new bool[n];
  }
  float R() { return (float)rnd.NextDouble(); }
  void Spawn(int i, bool anywhere) {
    ash[i] = R() < 0.35f;
    x[i] = Zone.X + R() * Math.Max(1, Zone.Width);
    ph[i] = R() * 6.283f; fq[i] = 0.4f + R() * 0.9f;
    if (ash[i]) {
      // ceniza blanca tenue: grande, lenta, cae o flota de lado
      y[i] = anywhere ? Zone.Y + R() * Math.Max(1, Zone.Height) : Zone.Y + R() * Math.Max(1, Zone.Height) * 0.5f;
      vx[i] = -10f + R() * 14f; vy[i] = -3f + R() * 11f; max[i] = 9f + R() * 9f; size[i] = 1.0f + R() * 1.8f;
      amp[i] = 6f + R() * 12f; alpha[i] = 0.25f + R() * 0.35f;
    } else {
      y[i] = anywhere ? Zone.Y + R() * Math.Max(1, Zone.Height) : Zone.Bottom - R() * 40f;
      vx[i] = -6f + R() * 12f; vy[i] = -(8f + R() * 24f); max[i] = 5f + R() * 8f; size[i] = 0.7f + R() * 2.3f;
      amp[i] = 3f + R() * 9f; alpha[i] = 0.45f + R() * 0.55f;
    }
    life[i] = anywhere ? R() * max[i] : 0f;
  }
  public void Reset(Rectangle zone, Rectangle avoid) { Zone = zone; Avoid = avoid; for (int i = 0; i < n; i++) Spawn(i, true); }
  Rectangle Box(int i) { int s = (int)Math.Ceiling(size[i] * 4f) + 3; return new Rectangle((int)x[i] - s, (int)y[i] - s, 2 * s, 2 * s); }
  // invalidate = false cuando la niebla ya repinta toda la zona en este fotograma
  public void Step(float dt, Control target, bool invalidate) {
    if (Zone.Width < 8 || Zone.Height < 8) return;
    t += dt;
    for (int i = 0; i < n; i++) {
      if (invalidate) target.Invalidate(Box(i), false);
      life[i] += dt;
      float sway = (float)Math.Cos(life[i] * fq[i] * 2f + ph[i]) * amp[i] * fq[i] * 2f;
      x[i] += (vx[i] + sway + (float)Math.Sin(t * 0.13f) * 3f) * dt; y[i] += vy[i] * dt;
      if (life[i] >= max[i] || y[i] < Zone.Y - 10 || y[i] > Zone.Bottom + 10 || x[i] < Zone.X - 10 || x[i] > Zone.Right + 10) Spawn(i, false);
      if (invalidate) target.Invalidate(Box(i), false);
    }
  }
  public void Draw(Graphics g) {
    if (Zone.Width < 8 || Zone.Height < 8) return;
    SmoothingMode sm = g.SmoothingMode; g.SmoothingMode = SmoothingMode.AntiAlias;
    for (int i = 0; i < n; i++) {
      if (Avoid.Contains((int)x[i], (int)y[i]) || !Zone.Contains((int)x[i], (int)y[i])) continue;
      float a = (float)Math.Sin(Math.PI * Math.Min(1f, life[i] / max[i])) * alpha[i]; if (a <= 0.02f) continue;
      float s = size[i];
      if (ash[i]) {
        using (SolidBrush b0 = new SolidBrush(Color.FromArgb((int)(30 * a), 225, 225, 230))) g.FillEllipse(b0, x[i] - s * 1.8f, y[i] - s * 1.8f, s * 3.6f, s * 3.6f);
        using (SolidBrush b1 = new SolidBrush(Color.FromArgb((int)(120 * a), 215, 212, 210))) g.FillEllipse(b1, x[i] - s * 0.8f, y[i] - s * 0.8f, s * 1.6f, s * 1.6f);
      } else {
        float fl = 0.8f + 0.2f * (float)Math.Sin(life[i] * 9f + ph[i]);  // parpadeo
        using (SolidBrush b0 = new SolidBrush(Color.FromArgb((int)(22 * a * fl), 255, 90, 30))) g.FillEllipse(b0, x[i] - s * 4f, y[i] - s * 4f, s * 8f, s * 8f);
        using (SolidBrush b1 = new SolidBrush(Color.FromArgb((int)(60 * a * fl), 255, 125, 50))) g.FillEllipse(b1, x[i] - s * 2.2f, y[i] - s * 2.2f, s * 4.4f, s * 4.4f);
        using (SolidBrush b2 = new SolidBrush(Color.FromArgb((int)(215 * a), 255, 205, 135))) g.FillEllipse(b2, x[i] - s * 0.6f, y[i] - s * 0.6f, s * 1.2f, s * 1.2f);
      }
    }
    g.SmoothingMode = sm;
  }
}
// (launcher114) Niebla: capas pre-renderizadas una vez (ruido fBm suavizado, horizontalmente ciclica, alfa ya
// incluido en los pixeles PArgb) que se desplazan despacio a velocidades/direcciones distintas. Por fotograma solo
// DrawImageUnscaled (camino rapido de GDI+), recortado a la zona y sin pasar por el titulo.
// (launcher119) Niebla volumetrica: hasta 5 bancos con paralaje (los de delante, mas bajos, densos y rapidos; los de
// atras, altos, tenues y lentos); 2 se anclan a la altura del cazador (Focus) para envolverle piernas y cintura.
// El ruido se calcula a media resolucion y se escala una sola vez (bilineal): 4x menos trabajo al arrancar y bancos
// mas blandos. Aparece con un fundido (Fade 0..1) sin coste: hasta llegar a 1 se dibuja con ImageAttributes.
// (launcher-fog-ds) Niebla como la del launcher de Demon's Souls: tres capas de ruido fBm con deformacion de dominio
// (domain warp), que evolucionan en el tiempo (no son texturas que se desplazan), con paralaje: la de atras grande y
// lenta, la de delante pequena, rapida y pegada al suelo. Se calcula cada fotograma a baja resolucion (en paralelo)
// y se escala con filtrado bilineal: blanda, sin costuras, sin bandas (tramado).
public class BbFog {
  public Rectangle Zone; public Rectangle Avoid; public Rectangle Focus; public float Fade = 1f;
  int count; float t; Bitmap bmp; int bw, bh; int[] px; float intensity;
  public BbFog(int n) { count = Math.Max(0, Math.Min(5, n)); intensity = 0.55f + 0.09f * count; }
  static float Fr(float v) { return v - (float)Math.Floor(v); }
  static void H33(float x, float y, float z, out float a, out float b, out float c) {
    float px = Fr(x * 0.1031f), py = Fr(y * 0.1030f), pz = Fr(z * 0.0973f);
    float d = px * (py + 33.33f) + py * (px + 33.33f) + pz * (pz + 33.33f);
    px += d; py += d; pz += d;
    a = -1f + 2f * Fr((px + py) * pz); b = -1f + 2f * Fr((px + px) * py); c = -1f + 2f * Fr((py + px) * px);
  }
  static float G(float ix, float iy, float iz, float fx, float fy, float fz) { float a, b, c; H33(ix, iy, iz, out a, out b, out c); return a * fx + b * fy + c * fz; }
  static float Noise(float x, float y, float z) {
    float ix = (float)Math.Floor(x), iy = (float)Math.Floor(y), iz = (float)Math.Floor(z);
    float fx = x - ix, fy = y - iy, fz = z - iz;
    float ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10), uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10), uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10);
    float n000 = G(ix, iy, iz, fx, fy, fz), n100 = G(ix + 1, iy, iz, fx - 1, fy, fz);
    float n010 = G(ix, iy + 1, iz, fx, fy - 1, fz), n110 = G(ix + 1, iy + 1, iz, fx - 1, fy - 1, fz);
    float n001 = G(ix, iy, iz + 1, fx, fy, fz - 1), n101 = G(ix + 1, iy, iz + 1, fx - 1, fy, fz - 1);
    float n011 = G(ix, iy + 1, iz + 1, fx, fy - 1, fz - 1), n111 = G(ix + 1, iy + 1, iz + 1, fx - 1, fy - 1, fz - 1);
    float x00 = n000 + (n100 - n000) * ux, x10 = n010 + (n110 - n010) * ux, x01 = n001 + (n101 - n001) * ux, x11 = n011 + (n111 - n011) * ux;
    float y0 = x00 + (x10 - x00) * uy, y1 = x01 + (x11 - x01) * uy;
    return y0 + (y1 - y0) * uz;
  }
  static float Fbm(float x, float y, float z, int oct) {
    float s = 0f, a = 0.5f;
    for (int i = 0; i < oct; i++) { s += a * Noise(x, y, z); x = x * 2.03f + 17.1f; y = y * 2.03f - 9.7f; z *= 1.6f; a *= 0.5f; }
    return s;
  }
  static float Layer(float x, float y, float t, float scale, float wx, float wy, float seed) {
    float qx = x * scale + wx * t, qy = y * scale + wy * t, qz = seed + t * 0.04f;
    float w1 = Fbm(qx, qy, qz, 3), w2 = Fbm(qx + 5.2f, qy + 1.3f, qz + 3.1f, 3);
    float n = Fbm(qx + w1 * 1.7f - wx * t * 0.35f, qy + w2 * 1.7f - wy * t * 0.35f, qz + t * 0.025f, 4);
    return Math.Max(0f, Math.Min(1f, n * 1.55f + 0.5f));
  }
  static float SS(float a, float b, float v) { float k = Math.Max(0f, Math.Min(1f, (v - a) / (b - a))); return k * k * (3f - 2f * k); }
  public void Reset(Rectangle zone, Rectangle avoid) { Reset(zone, avoid, Focus); }
  public void Reset(Rectangle zone, Rectangle avoid, Rectangle focus) {
    Zone = zone; Avoid = avoid; Focus = focus;
    if (count == 0 || zone.Width < 16 || zone.Height < 16) return;
    int w = Math.Max(64, Math.Min(300, zone.Width / 6)), h = Math.Max(32, (int)(w * zone.Height / (float)zone.Width));
    if (bmp != null && bw == w && bh == h) return;
    if (bmp != null) bmp.Dispose();
    bw = w; bh = h; px = new int[w * h];
    bmp = new Bitmap(w, h, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
  }
  public void Step(float dt) { t += dt; }
  void Render() {
    int w = bw, h = bh; float tt = t; float aspect = Zone.Width / (float)Math.Max(1, Zone.Height); float inten = intensity * Fade;
    // altura del cazador en la zona (0 arriba .. 1 abajo): la niebla de delante le envuelve hasta la cintura
    float hy = (!Focus.IsEmpty && Focus.Height > 32) ? (Focus.Bottom - Focus.Height * 0.45f - Zone.Y) / (float)Math.Max(1, Zone.Height) : 0.55f;
    int[] o = px;
    System.Threading.Tasks.Parallel.For(0, h, yy => {
      float v = (yy + 0.5f) / h; int row = yy * w;
      float g = SS(Math.Min(hy, 0.75f) - 0.30f, 1.0f, v);  // el suelo
      for (int xx = 0; xx < w; xx++) {
        float u = (xx + 0.5f) / w; float x = u * aspect, y = v;
        float d1 = Layer(x, y, tt, 1.0f, 0.030f, 0.003f, 1.0f);
        float d2 = Layer(x, y, tt, 2.1f, 0.055f, -0.004f, 7.0f);
        float d3 = g > 0.01f ? Layer(x, y, tt, 3.8f, 0.090f, 0f, 13.0f) : 0f;
        float dens = SS(0.30f, 0.85f, d1) * 0.50f + SS(0.38f, 0.90f, d2) * 0.38f * (0.30f + 0.70f * g) + SS(0.45f, 0.95f, d3) * 0.30f * g;
        dens *= 0.30f + 0.95f * g;
        float a = Math.Max(0f, Math.Min(1f, dens * inten));
        float lit = (0.80f + 0.30f * d2) * (0.85f + 0.25f * (1f - v));
        // tramado (sin bandas)
        uint hsh = (uint)(xx * 73856093 ^ yy * 19349663 ^ (int)(tt * 60f) * 83492791); float dith = ((hsh & 255) / 255f - 0.5f) / 255f;
        a = Math.Max(0f, Math.Min(1f, a + dith));
        int A = (int)(a * 255f + 0.5f);
        int R = (int)Math.Min(A, 0.56f * lit * a * 255f), Gc = (int)Math.Min(A, 0.61f * lit * a * 255f), B = (int)Math.Min(A, 0.69f * lit * a * 255f);
        o[row + xx] = (A << 24) | (R << 16) | (Gc << 8) | B;
      }
    });
    System.Drawing.Imaging.BitmapData bd = bmp.LockBits(new Rectangle(0, 0, w, h), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
    try { for (int yy = 0; yy < h; yy++) System.Runtime.InteropServices.Marshal.Copy(o, yy * w, new IntPtr(bd.Scan0.ToInt64() + (long)yy * bd.Stride), w); }
    finally { bmp.UnlockBits(bd); }
  }
  int lastFrame = -1;
  public void Draw(Graphics g) {
    if (bmp == null || count == 0 || Zone.Width < 16 || Zone.Height < 16 || Fade <= 0.01f) return;
    int frame = (int)(t * 30f); if (frame != lastFrame) { lastFrame = frame; Render(); }
    GraphicsState st = g.Save();
    try {
      g.SetClip(Zone, CombineMode.Intersect); if (!Avoid.IsEmpty) g.SetClip(Avoid, CombineMode.Exclude);
      g.CompositingMode = CompositingMode.SourceOver; g.CompositingQuality = CompositingQuality.HighSpeed;
      g.InterpolationMode = InterpolationMode.Bilinear; g.PixelOffsetMode = PixelOffsetMode.Half;
      using (System.Drawing.Imaging.ImageAttributes ia = new System.Drawing.Imaging.ImageAttributes()) {
        ia.SetWrapMode(WrapMode.TileFlipXY);
        g.DrawImage(bmp, Zone, 0f, 0f, (float)bw, (float)bh, GraphicsUnit.Pixel, ia);
      }
    } finally { g.Restore(st); }
  }
  public Rectangle Dirty() { return (bmp == null || count == 0) ? Rectangle.Empty : Zone; }
}
public class BbButton : Button {
  public static bool ReduceMotion = false;
  public float Radius = 10f;
  float hover = 0f, press = 0f, hoverT = 0f, pressT = 0f; int last = 0;
  Timer anim = new Timer();
  public BbButton() {
    SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw | ControlStyles.SupportsTransparentBackColor, true);
    anim.Interval = 15; anim.Tick += OnAnim;
  }
  void Kick() { if (ReduceMotion) { hover = hoverT; press = pressT; Invalidate(); return; } if (!anim.Enabled) { last = Environment.TickCount - 15; anim.Start(); } }
  void OnAnim(object s, EventArgs e) {
    int now = Environment.TickCount; float dt = Math.Max(1, Math.Min(50, now - last)); last = now;
    float kh = 1f - (float)Math.Exp(-dt / (hoverT > hover ? 70.0 : 110.0)), kp = 1f - (float)Math.Exp(-dt / (pressT > press ? 28.0 : 60.0));  // (launcher119) entrada algo mas rapida que la salida
    hover += (hoverT - hover) * kh; press += (pressT - press) * kp;
    if (Math.Abs(hoverT - hover) < 0.004f && Math.Abs(pressT - press) < 0.004f) { hover = hoverT; press = pressT; anim.Stop(); }
    Invalidate();
  }
  protected override void OnMouseEnter(EventArgs e) { hoverT = 1f; Kick(); base.OnMouseEnter(e); }
  protected override void OnMouseLeave(EventArgs e) { hoverT = 0f; pressT = 0f; Kick(); base.OnMouseLeave(e); }
  protected override void OnMouseDown(MouseEventArgs e) { if (e.Button == MouseButtons.Left) { pressT = 1f; Kick(); } base.OnMouseDown(e); }
  protected override void OnMouseUp(MouseEventArgs e) { pressT = 0f; Kick(); base.OnMouseUp(e); }
  protected override void OnEnabledChanged(EventArgs e) { if (!Enabled) { hoverT = 0f; pressT = 0f; hover = 0f; press = 0f; } Invalidate(); base.OnEnabledChanged(e); }
  protected override void OnPaintBackground(PaintEventArgs e) { BbGfx.PaintParent(this, e); }
  // (launcher136) La franja inferior (sombra) ya no es parte del boton: la pinta el formulario (sin restos de texto).
  protected override void OnSizeChanged(EventArgs e) { base.OnSizeChanged(e); try { Region = new Region(new Rectangle(0, 0, Width, Math.Max(1, Height - 5))); } catch { } }
  protected override void OnPaint(PaintEventArgs e) {
    Graphics g = e.Graphics; g.SmoothingMode = SmoothingMode.AntiAlias; g.PixelOffsetMode = PixelOffsetMode.HighQuality;
    Color baseC = BackColor; if (baseC.A == 0) baseC = Color.FromArgb(200, 24, 28, 38);
    Color hot = FlatAppearance.MouseOverBackColor; if (hot.IsEmpty) hot = BbGfx.Shade(baseC, 1.25f);
    Color down = FlatAppearance.MouseDownBackColor; if (down.IsEmpty) down = BbGfx.Shade(hot, 0.78f);
    Color fill = BbGfx.Lerp(BbGfx.Lerp(baseC, hot, hover), down, press);
    if (!Enabled) fill = Color.FromArgb(Math.Max(60, fill.A / 2), BbGfx.Shade(fill, 0.7f));
    float ins = 1.6f * press;
    // (launcher113) Al pasar el raton: sube 1.5 px, crece ~1 px por lado, sombra mas larga y halo dorado.
    float lift = (ReduceMotion || !Enabled) ? 0f : hover * (1f - press);
    RectangleF r = new RectangleF(3f + ins, 3f + ins, Width - 6f - 2f * ins, Height - 7.5f - 2f * ins);
    if (lift > 0f) { r.Inflate(lift, lift * 0.5f); r.Offset(0f, -1.5f * lift); }
    if (r.Width < 4 || r.Height < 4) return;
    // Sombra suave debajo (se aplasta al pulsar, se alarga al subir).
    float sh = (2.2f + 2.4f * lift) * (1f - press);
    if (Enabled && sh > 0.2f) { using (GraphicsPath sp = BbGfx.Round(new RectangleF(r.X + 0.5f, r.Y + sh, r.Width - 1f, r.Height), Radius)) using (SolidBrush sb = new SolidBrush(Color.FromArgb((int)((90 + 50 * lift) * (1f - press)), 0, 0, 0))) g.FillPath(sb, sp); }
    if (lift > 0.02f) {
      for (int gi = 1; gi <= 3; gi++) {
        using (GraphicsPath gp = BbGfx.Round(RectangleF.Inflate(r, gi * 0.9f, gi * 0.9f), Radius + gi * 0.9f))
        using (Pen gpn = new Pen(Color.FromArgb((int)(46 * lift / gi), 246, 210, 140), 1.2f)) g.DrawPath(gpn, gp);
      }
    }
    using (GraphicsPath p = BbGfx.Round(r, Radius)) {
      using (SolidBrush b = new SolidBrush(fill)) g.FillPath(b, p);
      // Brillo arriba (mas al pasar el raton): relieve sutil.
      RectangleF top = new RectangleF(r.X, r.Y, r.Width, Math.Max(1f, r.Height / 2f));
      using (LinearGradientBrush lg = new LinearGradientBrush(new RectangleF(top.X, top.Y - 1, top.Width, top.Height + 2), Color.FromArgb((int)(22 + 20 * hover), 255, 255, 255), Color.FromArgb(0, 255, 255, 255), 90f)) {
        Region old = g.Clip; g.SetClip(p, CombineMode.Intersect); g.FillRectangle(lg, top); g.Clip = old;
      }
      if (FlatAppearance.BorderSize > 0) {
        Color bc = FlatAppearance.BorderColor; if (bc.IsEmpty) bc = Color.FromArgb(110, 110, 100);
        bc = BbGfx.Lerp(bc, Color.FromArgb(235, 222, 196, 140), hover * 0.65f);
        if (!Enabled) bc = Color.FromArgb(90, bc);
        using (Pen pen = new Pen(bc, 1f)) g.DrawPath(pen, p);
      }
      if (Focused && ShowFocusCues) { using (Pen fp = new Pen(Color.FromArgb(120, 246, 222, 168), 1f)) { fp.DashStyle = DashStyle.Dot; using (GraphicsPath fpp = BbGfx.Round(RectangleF.Inflate(r, -3f, -3f), Math.Max(2f, Radius - 3f))) g.DrawPath(fp, fpp); } }
    }
    Color fc = Enabled ? ForeColor : Color.FromArgb(140, 150, 145, 135);
    Rectangle tr = Rectangle.Round(r); tr.Offset(0, (int)Math.Round(press));
    TextFormatFlags fl = TextFormatFlags.VerticalCenter | TextFormatFlags.WordBreak | TextFormatFlags.NoPrefix;
    bool left = (TextAlign == ContentAlignment.MiddleLeft || TextAlign == ContentAlignment.TopLeft || TextAlign == ContentAlignment.BottomLeft);
    if (Image != null) {
      int ix = tr.X + 10; int iy = tr.Y + (tr.Height - Image.Height) / 2;
      g.DrawImage(Image, ix, iy, Image.Width, Image.Height);
      tr = new Rectangle(ix + Image.Width + 8, tr.Y, Math.Max(1, tr.Right - (ix + Image.Width + 8) - 6), tr.Height);
      fl |= TextFormatFlags.Left;
    } else if (left) { tr.Inflate(-10, 0); fl |= TextFormatFlags.Left; }
    else fl |= TextFormatFlags.HorizontalCenter;
    TextRenderer.DrawText(g, Text, Font, tr, fc, fl);
  }
  protected override void Dispose(bool d) { if (d) { anim.Stop(); anim.Dispose(); } base.Dispose(d); }
}
public class BbScrollBar : Control {
  public static bool ReduceMotion = false;
  public double Maximum = 0, Value = 0, Viewport = 1, Requested = 0;
  public int PageDir = 0;
  public Color ThumbColor = Color.FromArgb(150, 120, 70), ThumbHot = Color.FromArgb(222, 196, 140), TrackColor = Color.FromArgb(70, 0, 0, 0);
  public event EventHandler Scrolled; public event EventHandler PageRequested;
  float hover = 0f, hoverT = 0f; int last = 0; bool drag = false; int dragY0 = 0; double dragV0 = 0;
  Timer anim = new Timer();
  public BbScrollBar() {
    SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw | ControlStyles.SupportsTransparentBackColor, true);
    BackColor = Color.Transparent; Width = 14; Cursor = Cursors.Hand; anim.Interval = 15; anim.Tick += OnAnim;
  }
  public void SetState(double max, double val, double view) {
    bool ch = (max != Maximum || val != Value || view != Viewport); Maximum = max; Value = val; Viewport = Math.Max(1, view);
    bool vis = max > 0; if (Visible != vis) Visible = vis; if (ch) Invalidate();
  }
  void Kick() { if (ReduceMotion) { hover = hoverT; Invalidate(); return; } if (!anim.Enabled) { last = Environment.TickCount - 15; anim.Start(); } }
  void OnAnim(object s, EventArgs e) {
    int now = Environment.TickCount; float dt = Math.Max(1, Math.Min(50, now - last)); last = now;
    hover += (hoverT - hover) * (1f - (float)Math.Exp(-dt / 70.0));
    if (Math.Abs(hoverT - hover) < 0.01f) { hover = hoverT; anim.Stop(); }
    Invalidate();
  }
  float ThumbH() { double total = Maximum + Viewport; float h = Height - 8; return (float)Math.Min(h, Math.Max(36.0, h * Viewport / Math.Max(1.0, total))); }
  RectangleF ThumbRect() {
    float h = Height - 8; float th = ThumbH(); double v = drag ? Requested : Value;
    float y = 4f + (Maximum > 0 ? (float)((h - th) * Math.Max(0, Math.Min(1, v / Maximum))) : 0f);
    float maxW = Math.Max(4f, Math.Min(9f, Width - 4f)); float w = 4f + (maxW - 4f) * hover;
    return new RectangleF(Width - w - 2f, y, w, th);
  }
  protected override void OnPaint(PaintEventArgs e) {
    if (Maximum <= 0) return;
    Graphics g = e.Graphics; g.SmoothingMode = SmoothingMode.AntiAlias; g.PixelOffsetMode = PixelOffsetMode.HighQuality;
    RectangleF t = ThumbRect();
    if (hover > 0.01f) { using (GraphicsPath tp = BbGfx.Round(new RectangleF(t.X, 4f, t.Width, Height - 8f), t.Width / 2f)) using (SolidBrush tb = new SolidBrush(Color.FromArgb((int)(TrackColor.A * hover), TrackColor))) g.FillPath(tb, tp); }
    Color c = BbGfx.Lerp(Color.FromArgb(170, ThumbColor), ThumbHot, drag ? 1f : hover * 0.7f);
    using (GraphicsPath p = BbGfx.Round(t, t.Width / 2f)) using (SolidBrush b = new SolidBrush(c)) g.FillPath(b, p);
  }
  protected override void OnMouseEnter(EventArgs e) { hoverT = 1f; Kick(); base.OnMouseEnter(e); }
  protected override void OnMouseLeave(EventArgs e) { if (!drag) { hoverT = 0f; Kick(); } base.OnMouseLeave(e); }
  protected override void OnMouseDown(MouseEventArgs e) {
    base.OnMouseDown(e); if (e.Button != MouseButtons.Left || Maximum <= 0) return;
    RectangleF t = ThumbRect();
    if (e.Y >= t.Top && e.Y <= t.Bottom) { drag = true; dragY0 = Cursor.Position.Y; dragV0 = Value; Requested = Value; Capture = true; Invalidate(); }
    else { PageDir = e.Y < t.Top ? -1 : 1; if (PageRequested != null) PageRequested(this, EventArgs.Empty); }
  }
  protected override void OnMouseMove(MouseEventArgs e) {
    base.OnMouseMove(e); if (!drag) return;
    double room = Math.Max(1.0, (Height - 8) - ThumbH());
    Requested = Math.Max(0, Math.Min(Maximum, dragV0 + (Cursor.Position.Y - dragY0) * Maximum / room));
    Invalidate();  // (launcher119) la barra sigue al cursor al instante; el contenido la alcanza con suavizado
    if (Requested != Value && Scrolled != null) Scrolled(this, EventArgs.Empty);
  }
  protected override void OnMouseUp(MouseEventArgs e) {
    base.OnMouseUp(e); if (!drag) return; drag = false; Capture = false;
    if (!ClientRectangle.Contains(PointToClient(Cursor.Position))) hoverT = 0f; Kick(); Invalidate();
  }
  protected override void OnMouseCaptureChanged(EventArgs e) { if (drag && !Capture) { drag = false; hoverT = 0f; Kick(); } base.OnMouseCaptureChanged(e); }
  protected override void Dispose(bool d) { if (d) { anim.Stop(); anim.Dispose(); } base.Dispose(d); }
}
"@
}
[BbButton]::ReduceMotion = [bool]$reduceMotion; [BbScrollBar]::ReduceMotion = [bool]$reduceMotion
function New-Combo($items, $x, $y, $w, [string]$key, $enabledFlags = $null, $map = $null) {
	$c = New-Object NoWheelCombo
	$c.DropDownStyle = 'DropDownList'; $c.FlatStyle = 'Flat'; $c.Font = $fText
	$c.BackColor = $field; $c.ForeColor = $ink
	$c.Location = New-Object System.Drawing.Point($x, $y); $c.Width = $w
	foreach ($i in $items) { [void]$c.Items.Add($i) }
	$v = 0; [void][int]::TryParse("$($script:S[$key])", [ref]$v)
	if ($map) { if (-not $script:comboMaps) { $script:comboMaps = @{} }; $script:comboMaps[$key] = $map; $c.SelectedIndex = [math]::Max(0, [array]::IndexOf(@($map), $v)) } else { $c.SelectedIndex = [math]::Max(0, [math]::Min($v, $c.Items.Count - 1)) }
	$c.Tag = $key
	$c.Add_SelectedIndexChanged({ param($s, $e) $mp = $script:comboMaps[$s.Tag]; if ($mp) { if ($mp[$s.SelectedIndex] -lt 0) { if ($s.SelectedIndex + 1 -lt $s.Items.Count) { $s.SelectedIndex = $s.SelectedIndex + 1 }; return }; if (-not $script:alHold) { Set-Setting $s.Tag $mp[$s.SelectedIndex] } } elseif (-not $script:alHold) { Set-Setting $s.Tag $s.SelectedIndex } })
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
	if ($text.StartsWith('----')) { $color = [System.Drawing.Color]::White }
	elseif (-not $combo.Enabled) { $color = $dim }
	elseif ($combo -eq $script:cmbUpRef -and $e.Index -lt $script:upDisp.Count -and $script:upDisp[$e.Index] -ge 0 -and -not $script:upscalers[$script:upDisp[$e.Index]][2] -and $script:S.experimental -ne '1') { $color = $dim }
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
	# Flecha nativa tapada por un panel del color del campo con una flecha dibujada (gira al desplegar); un clic en ella abre la lista.
	if ($c.Parent) {
		$ov = New-Object System.Windows.Forms.Panel
		$ov.BackColor = $field; $ov.Cursor = [System.Windows.Forms.Cursors]::Hand
		$ov.Size = New-Object System.Drawing.Size(27, $c.Height)
		$ov.Location = New-Object System.Drawing.Point(($c.Left + $c.Width - 26), $c.Top)
		$ov.Visible = $c.Visible
		$ov.Tag = $c
		$ov.Add_Paint({ param($s, $e)
			# Flecha dibujada (dorada; gris si no esta disponible). Desplegada: apunta arriba.
			$cb = $s.Tag; $col = $(if ($cb.Enabled) { $gold } else { $dim })
			$dir = $(if ($cb.DroppedDown) { -1.0 } else { 1.0 })
			$cxh = [double]($s.Width / 2); $cyh = [double]($s.Height / 2)
			$pts = @()
			foreach ($p in @(@(-4.5, -2.5), @(0.0, 2.5), @(4.5, -2.5))) { $pts += New-Object System.Drawing.PointF([float]($cxh + $p[0]), [float]($cyh + $p[1] * $dir)) }
			$pen = New-Object System.Drawing.Pen($col, 1.8); $pen.StartCap = 'Round'; $pen.EndCap = 'Round'; $pen.LineJoin = 'Round'
			$e.Graphics.SmoothingMode = 'AntiAlias'; $e.Graphics.DrawLines($pen, [System.Drawing.PointF[]]$pts); $pen.Dispose()
		})
		$ov.Add_Click({ $c.Focus() | Out-Null; $c.DroppedDown = $true }.GetNewClosure())
		$c.Add_VisibleChanged({ $ov.Visible = $c.Visible }.GetNewClosure())
		$c.Add_DropDown({ $ov.Invalidate() }.GetNewClosure())
		$c.Add_DropDownClosed({ $ov.Invalidate() }.GetNewClosure())
		$c.Add_EnabledChanged({ $ov.Invalidate() }.GetNewClosure())
		$c.Parent.Controls.Add($ov); $ov.BringToFront()
	}
}
# Apartados de una sola columna: cabecera pulsable (pliega / despliega) con animacion de 200 ms (System.Windows.Forms.Timer).
$script:stackGroups = @(); $script:anims = @(); $script:scrollAnim = $null; $script:pcCardRef = $null
$script:animTimer = New-Object System.Windows.Forms.Timer; $script:animTimer.Interval = 15
function Relayout-Stack {
	$y = 100
	foreach ($sg in $script:stackGroups) { $sg.Top = $y; $y += $sg.Height + 12 }  # (launcher129) 12 px entre apartados
	if ($script:pcCardRef) { $script:pcCardRef.Top = $y + 2; $y = $script:pcCardRef.Bottom }
	if ($script:bottomReady) { $btnUpd.Top = $y + 14; $btnSettings.Top = $y + 14; $pbUpd.Top = $y + 66 }
}
# (launcher129) Alturas calculadas: cada apartado mide desde el borde inferior de su ultimo control visible (+10 px).
# Las ayudas (Add-Hint) se colocan bajo la ultima fila, crecen con su texto (ajuste de linea) y empujan lo que tengan debajo.
$script:hintLabels = @()
$script:getStateMI = $null
try { $script:getStateMI = [System.Windows.Forms.Control].GetMethod('GetState', [System.Reflection.BindingFlags]'Instance,NonPublic', $null, [type[]]@([int]), $null) } catch { }
function Test-OwnVisible($c) {
	# Visible propio del control (Control.Visible da $false si la ventana aun no se ha mostrado).
	if ($script:getStateMI) { try { return [bool]$script:getStateMI.Invoke($c, [object[]]@([int]2)) } catch { } }
	return $true
}
function Add-Hint($lb) {
	$lb.AutoSize = $false; $lb.TextAlign = 'TopLeft'
	$script:hintLabels += $lb
	$lb.Add_TextChanged({ param($s, $e) if ($s.Parent) { try { Fit-Group $s.Parent; Relayout-Stack } catch { Write-Log "Fit-Group: $_" } } })
}
function Fit-Group($g) {
	$info = $g.Tag
	if (-not $info -or -not $info.Header) { return }
	$kids = @($g.Controls | Where-Object { $_ -ne $info.Header })
	$hints = @($kids | Where-Object { $script:hintLabels -contains $_ } | Sort-Object Top)
	foreach ($h in $hints) {
		if (-not (Test-OwnVisible $h)) { continue }
		$top0 = $h.Top
		$ab = 26
		foreach ($c in $kids) { if ($c -ne $h -and $c.Top -lt $top0 -and (Test-OwnVisible $c) -and $c.Bottom -gt $ab) { $ab = $c.Bottom } }
		$nt = $ab + 4
		$flags = [System.Windows.Forms.TextFormatFlags]::WordBreak -bor [System.Windows.Forms.TextFormatFlags]::NoPrefix
		$sz = [System.Windows.Forms.TextRenderer]::MeasureText([string]$h.Text, $h.Font, (New-Object System.Drawing.Size([math]::Max(20, $h.Width), 4000)), $flags)
		$nh = [int][math]::Max(16, $sz.Height + 2)
		$below = @($kids | Where-Object { $_ -ne $h -and $_.Top -gt $top0 })
		$h.SetBounds($h.Left, $nt, $h.Width, $nh)
		$vis = @($below | Where-Object { Test-OwnVisible $_ })
		if ($vis.Count) {
			$minTop = ($vis | Measure-Object -Property Top -Minimum).Minimum
			$delta = [int](($nt + $nh + 6) - $minTop)
			if ($delta -ne 0) { foreach ($c in $below) { $c.Top = $c.Top + $delta } }
		}
	}
	$bot = 24
	foreach ($c in $kids) { if ((Test-OwnVisible $c) -and $c.Bottom -gt $bot) { $bot = $c.Bottom } }
	$full = [int]($bot + 10)
	$info.Full = $full
	if (-not $info.Collapsed) {
		$an = @($script:anims | Where-Object { $_.G -eq $g })
		if ($an.Count) { $an[0].To = $full } else { $g.Height = $full }
	}
}
function Fit-AllGroups { foreach ($fg in $script:stackGroups) { try { Fit-Group $fg } catch { } }; Relayout-Stack }
# Flecha de cabecera dibujada (no es un caracter): apunta a la derecha si esta plegado y gira 90 grados hasta apuntar abajo.
function Draw-Chevron($gr, [double]$cx, [double]$cy, [double]$open, $col) {
	$ang = $open * [math]::PI / 2; $cs = [math]::Cos($ang); $sn = [math]::Sin($ang)
	$pts = @()
	foreach ($p in @(@(-3.5, -4.5), @(2.5, 0.0), @(-3.5, 4.5))) { $pts += New-Object System.Drawing.PointF([float]($cx + $p[0] * $cs - $p[1] * $sn), [float]($cy + $p[0] * $sn + $p[1] * $cs)) }
	$pen = New-Object System.Drawing.Pen($col, 2.0); $pen.StartCap = 'Round'; $pen.EndCap = 'Round'; $pen.LineJoin = 'Round'
	$gr.SmoothingMode = 'AntiAlias'; $gr.DrawLines($pen, [System.Drawing.PointF[]]$pts); $pen.Dispose()
}
function Toggle-Group($grp) {
	$info = $grp.Tag; $info.Collapsed = -not $info.Collapsed
	$to = $(if ($info.Collapsed) { 34 } else { $info.Full })
	$toP = $(if ($info.Collapsed) { 0.0 } else { 1.0 })
	if ($reduceMotion) { $grp.Height = $to; $info.Progress = $toP; $info.Header.Invalidate(); Relayout-Stack; return }
	$script:wheelT.Stop()
	$script:anims = @($script:anims | Where-Object { $_.G -ne $grp }) + [pscustomobject]@{ G = $grp; From = $grp.Height; To = $to; FromP = $info.Progress; ToP = $toP; T0 = [Environment]::TickCount }
	if (-not $info.Collapsed) {
		# Al abrir, el desplazamiento suave baja hasta que la seccion entera se vea.
		$want = [math]::Max(0, ($grp.Top + $to) - ($panel.ClientSize.Height - 16))
		$script:scrollAnim = [pscustomobject]@{ From = $script:scrollY; To = $want; T0 = [Environment]::TickCount }
	}
	$script:animTimer.Start()
}
# Animacion 200 ms, ease-out cubico: la altura (recorte) y el giro de la flecha van a la vez.
$script:animTimer.Add_Tick({
	$now = [Environment]::TickCount; $done = $true
	foreach ($a in $script:anims) {
		$t = [math]::Min(1.0, ($now - $a.T0) / 240.0); $e = 1 - [math]::Pow(1 - $t, 4)  # (launcher113) 240 ms, ease-out cuartico
		$a.G.Height = [int]($a.From + ($a.To - $a.From) * $e)
		$inf = $a.G.Tag; $inf.Progress = $a.FromP + ($a.ToP - $a.FromP) * $e; $inf.Header.Invalidate()
		if ($t -lt 1) { $done = $false }
	}
	Relayout-Stack
	Set-ScrollY $script:scrollY
	if ($script:scrollAnim) {
		$st = $script:scrollAnim; $t = [math]::Min(1.0, ($now - $st.T0) / 240.0); $e = 1 - [math]::Pow(1 - $t, 4)
		Set-ScrollY ($st.From + ($st.To - $st.From) * $e)
		if ($t -ge 1) { $script:scrollAnim = $null } else { $done = $false }
	}
	if ($done) { $script:anims = @(); $script:animTimer.Stop() }
})
function New-Group([string]$text, $parent, $x, $y, $w, $h) {
	$g = New-Object System.Windows.Forms.GroupBox
	# Apartado con fondo solido (como los de Settings): el texto sale con ClearType. El titulo lo pinta una etiqueta
	# opaca (Header), no el GroupBox, que con fondo transparente dibuja la letra en escala de grises.
	$g.Text = ''; $g.Font = $fGroup; $g.ForeColor = $gold
	$g.BackColor = $grpBack
	$g.Location = New-Object System.Drawing.Point($x, $y); $g.Size = New-Object System.Drawing.Size($w, $h)
	# Franja de cabecera (solo el titulo y la flecha, sin tapar los controles de la primera fila).
	$hd = New-Object System.Windows.Forms.Label
	$hd.Location = New-Object System.Drawing.Point(6, 0); $hd.Size = New-Object System.Drawing.Size(($w - 12), 24)
	$hd.Font = $fGroup; $hd.ForeColor = $gold; $hd.BackColor = $grpBack; $hd.TextAlign = 'MiddleLeft'
	$hd.Padding = New-Object System.Windows.Forms.Padding(22, 0, 0, 0); $hd.Cursor = [System.Windows.Forms.Cursors]::Hand
	$hd.Text = $text
	$g.Tag = [pscustomobject]@{ Full = $h; Collapsed = $false; Title = $text; Header = $hd; Progress = 1.0 }
	$hd.Add_Paint({ param($s, $e) Draw-Chevron $e.Graphics 13.0 12.0 ([double]$s.Parent.Tag.Progress) $s.ForeColor })
	$hd.Add_Click({ param($s, $e) Toggle-Group $s.Parent })
	$hd.Add_MouseEnter({ param($s, $e) $s.ForeColor = $goldHi })
	$hd.Add_MouseLeave({ param($s, $e) $s.ForeColor = $gold })
	$g.Controls.Add($hd)
	$parent.Controls.Add($g)
	$script:stackGroups += $g
	return $g
}
function Put($grp, $ctl, $x, $y, $w) {
	# Colores y letra propios (si no, los heredaria del grupo).
	$fc = $ctl.ForeColor; $fn = $ctl.Font
	$grp.Controls.Add($ctl)
	$ctl.ForeColor = $fc; $ctl.Font = $fn
	# Etiquetas y casillas con el mismo fondo solido del apartado: con fondo transparente el texto sale en escala de grises (borroso).
	if ($grp.BackColor.A -eq 255 -and (($ctl -is [System.Windows.Forms.Label]) -or ($ctl -is [System.Windows.Forms.CheckBox]))) { $ctl.BackColor = $grp.BackColor }
	$ctl.Location = New-Object System.Drawing.Point($x, $y)
	if ($w) { $ctl.Width = $w }
	return $ctl
}
# Fila como en Demon's Souls: etiqueta a x=10 (112), desplegable a x=124 (282), paso de 32.
function Add-Row($grp, [string]$label, $ctl, $y, [string]$tip = '') {
	$lb = New-Label $label 12 ($y + 2) 150 24 $lblCol $fLabel
	[void](Put $grp $lb 12 ($y + 2) 150)
	[void](Put $grp $ctl 166 $y ($gw - 166 - $rowR))  # (launcher119) desplegable pegado a la derecha: termina a $rowR px del borde del apartado
	if ($tip) { $tips.SetToolTip($ctl, $tip); $tips.SetToolTip($lb, $tip) }
	return $lb
}
function New-DarkCombo($items, [string]$font = 'text') {
	$c = New-Object NoWheelCombo
	$c.DropDownStyle = 'DropDownList'; $c.FlatStyle = 'Flat'; $c.BackColor = $field; $c.ForeColor = $ink; $c.Font = $(if ($font -eq 'small') { $fSmall } else { $fText })
	foreach ($i in $items) { [void]$c.Items.Add($i) }
	return $c
}

# ---------------------------------------------------------------- ventana Settings (juego, emulador, parches, controles, sistema)
# Apartados redondeados de antes, en una sola columna que se desplaza. Se abre con el boton Settings.
$fSec = New-Object System.Drawing.Font('Segoe UI Semibold', 12, [System.Drawing.FontStyle]::Bold)
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
	$scrollC.Controls.Add($g)
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
$fSet.Text = (T 'Emulador - Bloodborne' 'Emulator - Bloodborne'); $fSet.StartPosition = 'CenterParent'; $fSet.FormBorderStyle = 'FixedDialog'; $fSet.MaximizeBox = $false; $fSet.MinimizeBox = $false; $fSet.ShowInTaskbar = $false
$fSet.BackColor = [System.Drawing.Color]::FromArgb(14, 16, 22); $fSet.ForeColor = $ink; $fSet.Font = $fText
$fSet.ClientSize = New-Object System.Drawing.Size(($colW + 24 + 14), [math]::Min(980, $H - 60))
$fSet.KeyPreview = $true
$fSet.Add_KeyDown({ param($s, $e) if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Escape) { $e.Handled = $true; $fSet.Hide() } })
Set-DoubleBuffer $fSet
$setBar = New-Object System.Windows.Forms.Panel; $setBar.Dock = 'Bottom'; $setBar.Height = 60; $setBar.BackColor = $fSet.BackColor
$scroll = New-Object System.Windows.Forms.Panel; $scroll.Dock = 'Fill'; $scroll.AutoScroll = $false; $scroll.BackColor = $fSet.BackColor
Set-DoubleBuffer $scroll
$fSet.Controls.Add($scroll); $fSet.Controls.Add($setBar); $scroll.BringToFront()
# (launcher112) Sin AutoScroll: $scrollC (contenido) se mueve a mano con rueda suave y barra fina propia (como el panel principal).
$scrollC = New-Object System.Windows.Forms.Panel; $scrollC.Location = New-Object System.Drawing.Point(0, 0); $scrollC.Size = New-Object System.Drawing.Size(($colW + 24), 200); $scrollC.BackColor = $fSet.BackColor
Set-DoubleBuffer $scrollC; $scroll.Controls.Add($scrollC)
$sbarS = New-Object BbScrollBar; $sbarS.Dock = 'Right'; $sbarS.Width = 12; $sbarS.Visible = $false
$scroll.Controls.Add($sbarS); $sbarS.BringToFront()
# Sombra suave bajo cada apartado redondeado (se pinta en el contenido, detras del apartado).
$scrollC.Add_Paint({
	param($s, $e)
	$gr = $e.Graphics; $gr.SmoothingMode = 'AntiAlias'
	foreach ($c in $s.Controls) {
		if (-not $c.Visible -or -not ($c -is [System.Windows.Forms.Panel])) { continue }
		foreach ($sh in @(@(5, 22), @(3, 38), @(1.5, 60))) {
			$p = New-RoundPath ($c.Left - $sh[0] + 2) ($c.Top - $sh[0] + 4) ($c.Width + 2 * $sh[0] - 4) ($c.Height + 2 * $sh[0] - 4) 18
			$b = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb([int]$sh[1], 0, 0, 0)); $gr.FillPath($b, $p); $b.Dispose(); $p.Dispose()
		}
	}
})
$script:setY = 0.0; $script:setPos = 0.0; $script:setTarget = 0.0; $script:setSlide = 0.0; $script:setLast = 0; $script:setTau = 90.0
function Get-SetMax { $ch = 0; foreach ($c in $scrollC.Controls) { if ((Test-OwnVisible $c) -and $c.Bottom -gt $ch) { $ch = $c.Bottom } }; return [double][math]::Max(0, ($ch + 8) - $scroll.ClientSize.Height) }
function Set-SetScrollY([double]$y) {
	$ph = $scroll.ClientSize.Height; $max = Get-SetMax
	$script:setY = [math]::Max(0.0, [math]::Min($max, $y))
	$nh = [int]($max + $ph); if ($scrollC.Height -ne $nh) { $scrollC.Height = $nh }
	$nt = -[int][math]::Round($script:setY) + [int][math]::Round($script:setSlide); if ($scrollC.Top -ne $nt) { $scrollC.Top = $nt }  # (launcher134) sin WM_SETREDRAW en esta ventana
	$sbarS.SetState($max, $script:setY, $ph)
}
$script:setWheelT = New-Object System.Windows.Forms.Timer; $script:setWheelT.Interval = 15
function Add-SetWheel([double]$px) {
	if (-not $script:setWheelT.Enabled) { $script:setPos = $script:setY; $script:setTarget = $script:setY; $script:setLast = [Environment]::TickCount - 15 }
	$script:setTau = 90.0
	$script:setTarget = [math]::Max(0.0, [math]::Min((Get-SetMax), $script:setTarget + $px))
	if ($reduceMotion) { $script:setPos = $script:setTarget; Set-SetScrollY $script:setPos; return }
	$script:setWheelT.Start()
}
# Acercamiento exponencial por tiempo (no por tick): mismo recorrido y suavidad aunque el temporizador se retrase.
$script:setWheelT.Add_Tick({
	$now = [Environment]::TickCount; $dt = [math]::Max(1, [math]::Min(50, $now - $script:setLast)); $script:setLast = $now
	$d = $script:setTarget - $script:setPos
	if ([math]::Abs($d) -lt 0.5) { $script:setPos = $script:setTarget; $script:setWheelT.Stop() } else { $script:setPos += $d * (1 - [math]::Exp(-$dt / $script:setTau)) }
	Set-SetScrollY $script:setPos
})
function Set-SetNow([double]$y) { $script:setWheelT.Stop(); Set-SetScrollY $y; $script:setPos = $script:setY; $script:setTarget = $script:setY }
$sbarS.add_Scrolled({
	if ($reduceMotion) { Set-SetNow $sbarS.Requested; return }
	if (-not $script:setWheelT.Enabled) { $script:setPos = $script:setY; $script:setLast = [Environment]::TickCount - 15 }
	$script:setTau = 35.0; $script:setTarget = [math]::Max(0.0, [math]::Min((Get-SetMax), [double]$sbarS.Requested)); $script:setWheelT.Start()
})
$sbarS.add_PageRequested({ Add-SetWheel ($sbarS.PageDir * $scroll.ClientSize.Height * 0.9) })
$scroll.Add_Resize({ Set-SetScrollY $script:setY })
# Entrada de la ventana Opciones: fundido y el contenido sube 18 px (220 ms, ease-out cubico).
# (launcher134) sin fundido ni Opacity: la ventana Emulador se ve entera desde el primer momento
$script:setFadeT0 = 0
$setFadeT = New-Object System.Windows.Forms.Timer; $setFadeT.Interval = 15
$setFadeT.Add_Tick({
	$t = [math]::Min(1.0, ([Environment]::TickCount - $script:setFadeT0) / 280.0); $e = 1 - [math]::Pow(1 - $t, 3)
	$fSet.Opacity = [math]::Max(0.01, [math]::Min(1.0, $e)); $script:setSlide = 34.0 * (1 - $e); Set-SetScrollY $script:setY
	if ($t -ge 1) { $fSet.Opacity = 1; $script:setSlide = 0.0; Set-SetScrollY $script:setY; $setFadeT.Stop() }
})
$fSet.Add_VisibleChanged({
	$setFadeT.Stop(); $script:setSlide = 0.0
	if ($fSet.Visible) { Invoke-SetLayout }  # (launcher134) sin fundido; rejilla (o disposicion simple si falla) cada vez que se abre
})
$btnSetOk = New-Btn (T 'Cerrar' 'Close') 12 10 $colW 40 $true
$btnSetOk.Add_Click({ $fSet.Hide() }); $setBar.Controls.Add($btnSetOk)
$c1 = 12; $iw = $colW - 24
$lw = 168; $cx = 190; $cw = $colW - $cx - 12
# (launcher131) Opciones con rejilla por filas: cada apartado guarda sus filas (Tag.Rows) y Fit-Sections las coloca de
# arriba abajo (primera fila a 44 px, hueco propio por fila). Las filas -Hint miden su texto con ajuste de linea y crecen;
# una ayuda vacia no ocupa sitio. Despues apila los apartados (12 px entre ellos) y rehace la barra de desplazamiento.
$script:setSections = @(); $script:curRow = $null; $script:curSec = $null
function New-SetSection([string]$text) {
	$g = New-Section $text 60
	$g.Tag = [pscustomobject]@{ Rows = (New-Object System.Collections.ArrayList) }
	$script:setSections += $g
	return $g
}
function Start-SetRow($sec, [int]$h = 32, [int]$gap = 8, [switch]$Hint) {
	$r = [pscustomobject]@{ Items = (New-Object System.Collections.ArrayList); H = $h; Gap = $gap; Hint = [bool]$Hint }
	[void]$sec.Tag.Rows.Add($r); $script:curRow = $r; $script:curSec = $sec
}
function Put-SetRow($ctl, [int]$x, [int]$w = 0, [int]$dy = 0) {
	[void](Add-To $script:curSec $ctl)
	$ctl.Left = $x; if ($w -gt 0) { $ctl.Width = $w }
	[void]$script:curRow.Items.Add([pscustomobject]@{ C = $ctl; Dy = $dy })
	if ($script:curRow.Hint -and ($ctl -is [System.Windows.Forms.Label])) { $ctl.AutoSize = $false; $ctl.TextAlign = 'TopLeft'; $ctl.Add_TextChanged({ Invoke-SetLayout }) }
	return $ctl
}
function New-SetCombo {
	$c = New-Object NoWheelCombo
	$c.DropDownStyle = 'DropDownList'; $c.FlatStyle = 'Flat'; $c.BackColor = $field; $c.ForeColor = $ink; $c.Font = $fText
	return $c
}
function Fit-Sections {
	if (-not $script:setSections.Count) { return }
	$flags = [System.Windows.Forms.TextFormatFlags]::WordBreak -bor [System.Windows.Forms.TextFormatFlags]::NoPrefix
	$y = 10
	foreach ($sec in $script:setSections) {
		$ry = 44; $first = $true
		foreach ($r in $sec.Tag.Rows) {
			$vis = @($r.Items | Where-Object { Test-OwnVisible $_.C })
			if (-not $vis.Count) { continue }
			$h = [int]$r.H
			if ($r.Hint) {
				$h = 0
				foreach ($it in $r.Items) {
					$txt = [string]$it.C.Text
					if (-not $txt) { $it.C.Height = 0; continue }
					$sz = [System.Windows.Forms.TextRenderer]::MeasureText($txt, $it.C.Font, (New-Object System.Drawing.Size([math]::Max(20, $it.C.Width), 4000)), $flags)
					$nh = [int]$sz.Height + 2; if ($nh -gt $h) { $h = $nh }
				}
				if ($h -le 0) { continue }
			}
			if (-not $first) { $ry += [int]$r.Gap }
			$first = $false
			foreach ($it in $r.Items) {
				$c = $it.C
				if ($r.Hint) { if ([string]$c.Text) { $c.SetBounds($c.Left, ($ry + $it.Dy), $c.Width, $h) } else { $c.Top = $ry } }
				else { $c.Top = $ry + $it.Dy }
				# Flecha dibujada de Set-ComboStyle (panel hermano con Tag = combo): sigue al desplegable.
				foreach ($ov in $sec.Controls) { if (($ov -is [System.Windows.Forms.Panel]) -and [object]::ReferenceEquals($ov.Tag, $c)) { $ov.SetBounds(($c.Left + $c.Width - 26), $c.Top, 27, $c.Height) } }
			}
			$ry += $h
		}
		$hh = [int]($ry + 14)
		if ($sec.Top -ne $y) { $sec.Top = $y }
		if ($sec.Height -ne $hh) {
			$sec.Height = $hh
			$rp = New-RoundPath 0 0 $colW $hh 16; $sec.Region = New-Object System.Drawing.Region($rp); $rp.Dispose()
		}
		$y += $hh + 12
	}
	$script:secY = $y
	if ($padEnd) { $padEnd.Top = $y }
	if ($script:setAutoH) { $want = [int][math]::Min(($y + $setBar.Height), ($H - 60)); if ($fSet.ClientSize.Height -ne $want) { $fSet.ClientSize = New-Object System.Drawing.Size($fSet.ClientSize.Width, $want) } }  # (launcher132)
	$scrollC.Invalidate()
	Set-SetScrollY $script:setY
}
# (launcher134) Disposicion simple de la ventana Emulador: apila filas y apartados sin depender de nada mas. Se usa si
# Fit-Sections falla o deja la ventana vacia (apartado sin alto, contenido fuera de la vista, etc.).
function Show-SetFallback {
	$y = 10
	foreach ($sec in $script:setSections) {
		$ry = 44
		foreach ($r in $sec.Tag.Rows) {
			$items = @(); foreach ($it in $r.Items) { $v = $true; try { $v = Test-OwnVisible $it.C } catch { }; if ($v) { $items += $it } }
			if (-not $items.Count) { continue }
			$h = [int]$r.H
			if ($r.Hint) {
				$h = 0
				foreach ($it in $items) {
					$tx = [string]$it.C.Text
					if ($tx) { $nh = 20; try { $nh = [int][System.Windows.Forms.TextRenderer]::MeasureText($tx, $it.C.Font, (New-Object System.Drawing.Size([math]::Max(20, $it.C.Width), 4000)), ([System.Windows.Forms.TextFormatFlags]::WordBreak -bor [System.Windows.Forms.TextFormatFlags]::NoPrefix)).Height + 2 } catch { }; if ($nh -gt $h) { $h = $nh } }
				}
				if ($h -le 0) { foreach ($it in $items) { $it.C.Top = $ry; $it.C.Height = 0 }; continue }
			}
			if ($h -le 0) { $h = 24 }
			foreach ($it in $items) {
				$c = $it.C; $c.Top = $ry + $it.Dy; if ($r.Hint) { $c.Height = $h }
				foreach ($ov in $sec.Controls) { if (($ov -is [System.Windows.Forms.Panel]) -and [object]::ReferenceEquals($ov.Tag, $c)) { $ov.SetBounds(($c.Left + $c.Width - 26), $c.Top, 27, $c.Height) } }
			}
			$ry += $h + [math]::Max(4, [int]$r.Gap)
		}
		$hh = [int]($ry + 14)
		$sec.Visible = $true; $sec.SetBounds(12, $y, $colW, $hh)
		try { $rp = New-RoundPath 0 0 $colW $hh 16; $sec.Region = New-Object System.Drawing.Region($rp); $rp.Dispose() } catch { $sec.Region = $null }
		$y += $hh + 12
	}
	$script:secY = $y; $script:setY = 0.0; $script:setPos = 0.0; $script:setTarget = 0.0; $script:setSlide = 0.0
	if ($padEnd) { $padEnd.Top = $y }
	$scroll.Visible = $true; $scrollC.Visible = $true
	if ($script:setAutoH) { $want = [int][math]::Min(($y + $setBar.Height), ($H - 60)); if ($fSet.ClientSize.Height -ne $want) { $fSet.ClientSize = New-Object System.Drawing.Size($fSet.ClientSize.Width, $want) } }
	$vh = [int][math]::Max(1, $scroll.ClientSize.Height)
	$scrollC.SetBounds(0, 0, ($colW + 24), [int][math]::Max(($y + 8), $vh))
	try { $sbarS.SetState([double][math]::Max(0, ($y + 8) - $vh), 0.0, [double]$vh) } catch { }
	$scrollC.Invalidate($true)
}
# (launcher134) Fit-Sections con red: si falla o deja algo sin ver, disposicion simple. Los errores van a launcher.log.
$script:inSetLayout = $false
function Invoke-SetLayout {
	if ($script:inSetLayout) { return }
	$script:inSetLayout = $true
	try {
		$ok = $true
		try { Fit-Sections } catch { $ok = $false; Write-Log "Emulador (ventana): Fit-Sections: $_" }
		if ($ok) {
			try {
				foreach ($sec in $script:setSections) { if ($sec.Height -lt 40 -or -not (Test-OwnVisible $sec) -or -not [object]::ReferenceEquals($sec.Parent, $scrollC)) { $ok = $false } }
				if ($script:setSections.Count -and $scrollC.Height -lt $script:setSections[-1].Bottom) { $ok = $false }
				if ($scrollC.Top -gt 0 -or -not (Test-OwnVisible $scrollC) -or -not (Test-OwnVisible $scroll)) { $ok = $false }
				if ($fSet.Visible -and $scroll.ClientSize.Height -lt 40) { $ok = $false }
			} catch { $ok = $false }
			if (-not $ok) { Write-Log 'Emulador (ventana): rejilla incompleta; se usa la disposicion simple' }
		}
		if (-not $ok) { try { Show-SetFallback } catch { Write-Log "Emulador (ventana): disposicion simple: $_" } }
	} finally { $script:inSetLayout = $false }
}

# ---- (launcher132) Ventana Emulador (antes Opciones). Quitado de aqui: Juego (ya esta "Carpeta del juego" en el panel),
# Idioma del lanzador (sigue a Windows), Usar actualizacion (-patch), Sound fix (solo mostraba; el arreglo sigue siendo
# automatico) y Anadir a Steam (esta en el panel). Estos objetos no se ven pero el resto del lanzador los sigue rellenando.
$cmbGame = New-SetCombo; $btnGame = New-Btn (T 'Examinar' 'Browse') 0 0 90 30
$lbGame = New-Label '' 0 0 10 20 $dim $fSmall; $lbCompat = New-Label '' 0 0 10 20 $dim $fSmall; $lbPatch = New-Label '' 0 0 10 20 $dim $fSmall
$red = [System.Drawing.Color]::FromArgb(230, 110, 100)

# ---- Emulador: donde esta, version, Examinar / Actualizar / Volver a la version anterior
$gEmu = New-SetSection (T 'Emulador (shadPS4)' 'Emulator (shadPS4)')
Start-SetRow $gEmu 32 0
$cmbEmu = Put-SetRow (New-SetCombo) 12 ($iw - 98)
$cmbEmu.DropDownWidth = 520
$btnEmu = Put-SetRow (New-Btn (T 'Examinar' 'Browse') 0 0 90 30) ($iw - 78) 90; $btnEmu.Font = $fSmall
$tips.SetToolTip($cmbEmu, (T 'shadPS4 elegido (los que hay junto al lanzador). Examinar: elegir otro shadPS4.exe.' 'Selected shadPS4 (the ones next to the launcher). Browse: pick another shadPS4.exe.'))
Start-SetRow $gEmu 0 10 -Hint
$lbEPathK = Put-SetRow (New-Label (T 'Instalado en (el que usa JUGAR)' 'Installed at (the one PLAY uses)') 0 0 $iw 20 $lblCol $fSmall) 12 $iw
Start-SetRow $gEmu 0 2 -Hint
$lbEmuPath = Put-SetRow (New-Label '' 0 0 $iw 20 $ink $fSmall) 12 $iw
Start-SetRow $gEmu 0 4 -Hint
$lbEVerK = Put-SetRow (New-Label (T 'Versión' 'Version') 0 0 $lw 20 $lblCol $fSmall) 12 ($cx - 20)
$lbVerV = Put-SetRow (New-Label '' 0 0 $cw 20 $ink $fSmall) $cx $cw
Start-SetRow $gEmu 0 4 -Hint
$lbEDateK = Put-SetRow (New-Label (T 'Fecha del archivo' 'File date') 0 0 $lw 20 $lblCol $fSmall) 12 ($cx - 20)
$lbDateV = Put-SetRow (New-Label '' 0 0 $cw 20 $dim $fSmall) $cx $cw
Start-SetRow $gEmu 0 4 -Hint
$lbEMd5K = Put-SetRow (New-Label 'MD5' 0 0 $lw 20 $lblCol $fSmall) 12 ($cx - 20)
$lbMd5V = Put-SetRow (New-Label '' 0 0 $cw 20 $dim $fSmall) $cx $cw
$bw2e = [int](($iw - 8) / 2)
Start-SetRow $gEmu 34 12
$btnEmuUpd = Put-SetRow (New-Btn (T 'Actualizar' 'Update') 0 0 $bw2e 34) 12 $bw2e; $btnEmuUpd.Font = $fSmall
$btnRollback = Put-SetRow (New-Btn (T 'Volver a la versión anterior' 'Back to previous version') 0 0 $bw2e 34) (20 + $bw2e) $bw2e; $btnRollback.Font = $fSmall
Start-SetRow $gEmu 6 8
$pbEmu = New-Object System.Windows.Forms.ProgressBar; $pbEmu.Height = 6; $pbEmu.Maximum = 100; $pbEmu.Visible = $false
[void](Put-SetRow $pbEmu 12 $iw)
Start-SetRow $gEmu 0 8 -Hint
$lbBak = Put-SetRow (New-Label '' 12 0 $iw 20 $dim $fSmall) 12 $iw
Start-SetRow $gEmu 0 6 -Hint
$lbEmu = Put-SetRow (New-Label '' 12 0 $iw 20 $dim $fSmall) 12 $iw
$updTip = (T 'Actualizar shadPS4: busca la última versión oficial (github.com/shadps4-emu) y la instala en una carpeta nueva. Antes guarda una copia del emulador actual en _anterior (Volver a la versión anterior).' 'Update shadPS4: finds the latest official release (github.com/shadps4-emu) and installs it to a new folder. A copy of the current emulator is saved first in _anterior (Back to previous version).')
$tips.SetToolTip($btnEmuUpd, $updTip)
$rbTipOn = (T 'Deja el emulador exactamente como estaba antes de la última actualización (shadPS4.exe, DLL y el emulador elegido). No toca partidas ni ajustes.' 'Puts the emulator back exactly as it was before the last update (shadPS4.exe, DLLs and the selected emulator). Saves and settings are not touched.')
$rbTipOff = (T 'Aún no hay copia anterior: se crea sola antes de cada actualización.' 'No previous copy yet: one is made automatically before each update.')

# ---- Carpetas
$gDirs = New-SetSection (T 'Carpetas' 'Folders')
$bw3 = [int](($iw - 16) / 3)
Start-SetRow $gDirs 32 0
$btnLogs = Put-SetRow (New-Btn (T 'Logs' 'Logs') 0 0 $bw3 32) 12 $bw3; $btnLogs.Font = $fSmall
$btnUserDir = Put-SetRow (New-Btn (T 'Carpeta user' 'User folder') 0 0 $bw3 32) (20 + $bw3) $bw3; $btnUserDir.Font = $fSmall
$btnPatches = Put-SetRow (New-Btn (T 'Parches' 'Patches') 0 0 $bw3 32) (28 + 2 * $bw3) $bw3; $btnPatches.Font = $fSmall
$tips.SetToolTip($btnLogs, (T 'Abre user\log (shad_log.txt).' 'Opens user\log (shad_log.txt).'))
$tips.SetToolTip($btnPatches, (T 'El lanzador no cambia parches nunca (solo muestra su estado). Abre la carpeta user\patches; cámbialos con shadPS4 / BB_Launcher.' 'The launcher never changes patches (it only shows their state). Opens user\patches; change them with shadPS4 / BB_Launcher.'))
$btnPatches.Add_Click({ $d = Join-Path (Get-UserDir $script:S.emu) 'patches'; if (-not (Test-Path -LiteralPath $d)) { $d = Get-UserDir $script:S.emu }; if (Test-Path -LiteralPath $d) { Start-Process explorer.exe -ArgumentList ('"' + $d + '"') } })
Start-SetRow $gDirs 24 6
$lnkUser = New-Object System.Windows.Forms.LinkLabel; $lnkUser.Text = (T 'Cambiar carpeta user' 'Change user folder'); $lnkUser.Font = $fSmall; $lnkUser.TextAlign = 'MiddleLeft'
$lnkUser.LinkColor = $gold; $lnkUser.Height = 24
[void](Put-SetRow $lnkUser 12 $iw)
$tips.SetToolTip($lnkUser, (T 'Carpeta "user" de shadPS4 (config, parches, trofeos). Se detecta sola; cámbiala si usas otra.' 'shadPS4 "user" folder (config, patches, trophies). Auto-detected; change it if you use another.'))

# ---- Controles (BB_INPUT, sensor y mando en segundo plano los usa JUGAR)
$gCtl = New-SetSection (T 'Controles' 'Controls')
Start-SetRow $gCtl 28 0
$rbPad = New-Object System.Windows.Forms.RadioButton; $rbPad.Text = (T 'Mando' 'Gamepad'); $rbPad.Size = New-Object System.Drawing.Size($lw, 26); $rbPad.ForeColor = $ink; $rbPad.Font = $fText
[void](Put-SetRow $rbPad 12 $lw 1)
$rbKbm = New-Object System.Windows.Forms.RadioButton; $rbKbm.Text = (T 'Teclado y ratón' 'Keyboard & mouse'); $rbKbm.Size = New-Object System.Drawing.Size($cw, 26); $rbKbm.ForeColor = $ink; $rbKbm.Font = $fText
[void](Put-SetRow $rbKbm $cx $cw 1)
if ($script:S.input -eq '1') { $rbKbm.Checked = $true } else { $rbPad.Checked = $true }
$rbKbm.Add_CheckedChanged({ Set-Setting 'input' $(if ($rbKbm.Checked) { '1' } else { '0' }) })
$tips.SetToolTip($rbKbm, (T 'shadPS4 acepta mando y teclado a la vez; las teclas se cambian en su configuración de controles.' 'shadPS4 takes pad and keyboard at once; keys are set in its input config.'))
Start-SetRow $gCtl 28 4
$chkMotion = Put-SetRow (New-Check (T 'Sensor de movimiento' 'Motion sensor') 0 0 $iw 'motion') 12 $iw 1
Start-SetRow $gCtl 28 4
$chkBg = Put-SetRow (New-Check (T 'Mando con la ventana en segundo plano' 'Pad input in background') 0 0 $iw 'bginput') 12 $iw 1
Start-SetRow $gCtl 34 10
$btnInput = Put-SetRow (New-Btn (T 'Configurar controles (teclado / mando)' 'Configure controls (keyboard / pad)') 0 0 $iw 34) 12 $iw
$padEnd = New-Object System.Windows.Forms.Label; $padEnd.Location = New-Object System.Drawing.Point(0, $script:secY); $padEnd.Size = New-Object System.Drawing.Size(1, 4); $padEnd.BackColor = [System.Drawing.Color]::Transparent; $scrollC.Controls.Add($padEnd)

# Rellena Instalado en / Version / Fecha / MD5 y el boton Volver a la version anterior (desactivado sin copia).
function Update-EmuInfo {
	$exe = Get-ActiveEmuExe
	if (-not $exe -or -not (Test-Path -LiteralPath $exe)) {
		$lbEmuPath.Text = (T 'No encuentro shadPS4.exe: pulsa Examinar.' 'shadPS4.exe not found: press Browse.'); $lbEmuPath.ForeColor = $red
		$lbVerV.Text = '-'; $lbDateV.Text = '-'; $lbMd5V.Text = '-'
	} else {
		$i = Get-EmuBuildInfo $exe
		$note = ''; if ($exe -ne "$($script:S.emu)") { $note = "`n" + (T '(el shadPS4 de reescalado / frame generation: JUGAR usa este con tus ajustes)' '(the upscaling / frame generation shadPS4: PLAY uses this one with your settings)') }
		$lbEmuPath.Text = $exe + $note; $lbEmuPath.ForeColor = $ink; $tips.SetToolTip($lbEmuPath, $exe)
		$lbVerV.Text = $i.Text; $lbDateV.Text = $i.Date; $lbMd5V.Text = $i.Md5
	}
	$b = $null; try { $b = Get-LatestEmuBackup } catch { }
	if ($b) {
		$btnRollback.Enabled = $true; $tips.SetToolTip($btnRollback, $rbTipOn)
		$lbBak.Text = (T "Copia anterior: $($b.fecha)  -  $($b.version)" "Previous copy: $($b.fecha)  -  $($b.version)"); $lbBak.ForeColor = $dim
	} else {
		$btnRollback.Enabled = $false; $tips.SetToolTip($btnRollback, $rbTipOff)
		$lbBak.Text = $rbTipOff; $lbBak.ForeColor = $dim
	}
}
$btnRollback.Add_Click({
	$b = $null; try { $b = Get-LatestEmuBackup } catch { }
	if (-not $b) { try { Update-EmuInfo } catch { }; return }
	$now = Get-EmuBuildInfo (Join-Path $b.carpeta (Split-Path -Leaf $b.exe))
	$msg = (T "Se vuelve a dejar el emulador como estaba antes de la última actualización.`n`nCopia del $($b.fecha)  ($($b.motivo))`n  Versión: $($b.version)`n  MD5: $($b.md5)`n  Carpeta: $($b.carpeta)`n  Archivos: $($b.Files -join ', ')`n  Emulador elegido: $($b.emu)`n`nAhora: $($now.Short)  (MD5 $($now.Md5))`n`nNo se tocan partidas ni ajustes (carpeta user). ¿Seguir?" "The emulator goes back to how it was before the last update.`n`nCopy from $($b.fecha)  ($($b.motivo))`n  Version: $($b.version)`n  MD5: $($b.md5)`n  Folder: $($b.carpeta)`n  Files: $($b.Files -join ', ')`n  Selected emulator: $($b.emu)`n`nNow: $($now.Short)  (MD5 $($now.Md5))`n`nSaves and settings (user folder) are not touched. Continue?")
	if ([System.Windows.Forms.MessageBox]::Show($msg, (T 'Volver a la versión anterior' 'Back to previous version'), 'YesNo', 'Question') -ne 'Yes') { return }
	$fSet.Cursor = 'WaitCursor'; $ok = $true
	try { Restore-EmuBackup $b } catch { $ok = $false; Write-Log "Volver a la version anterior: $_"; [void][System.Windows.Forms.MessageBox]::Show((T "No se pudo restaurar:`n$_" "Could not restore:`n$_"), 'shadPS4', 'OK', 'Error') }
	$fSet.Cursor = 'Default'
	try { Fill-Lists } catch { }
	if (-not $ok) { return }
	$ni = Get-EmuBuildInfo (Join-Path $b.carpeta (Split-Path -Leaf $b.exe))
	[void][System.Windows.Forms.MessageBox]::Show((T "Restaurada la versión anterior:`n`n$($ni.Text)`nMD5 $($ni.Md5)`n$($ni.Date)`n`nEmulador elegido: $($script:S.emu)" "Previous version restored:`n`n$($ni.Text)`nMD5 $($ni.Md5)`n$($ni.Date)`n`nSelected emulator: $($script:S.emu)"), 'shadPS4', 'OK', 'Information')
})
$btnEmuUpd.Add_Click({ & $script:doUpdate })
# La ventana se ajusta a su contenido (Fit-Sections); despues, la rejilla con la letra real.
$script:setAutoH = $true
Invoke-SetLayout
$fSet.Add_Shown({ Invoke-SetLayout })  # (launcher131) medidas con la letra real

# ---------------------------------------------------------------- panel derecho (como Demon's Souls)
$panelW = [math]::Min(520, $W)
# (launcher119) Columna pegada a la derecha: los apartados acaban 2 px antes de la barra (10 px de ancho, a 2 px del
# borde) y los desplegables a $rowR px del borde del apartado (8 px hasta la barra). Nada se cruza con la barra.
$sbW = 10; $rowR = 6
$gwTop = $panelW - 20           # JUGAR y filete (fijos arriba, sin barra al lado)
$gw = $panelW - 10 - $sbW - 2 - 2  # ancho de apartados
$panel = New-Object System.Windows.Forms.Panel
$panel.Location = New-Object System.Drawing.Point(($W - $panelW), 0)
$panel.Size = New-Object System.Drawing.Size($panelW, $H)
$panel.AutoScroll = $false
$panel.Anchor = 'None'  # (launcher113) posicion y alto los fija Apply-Layout (fuera del titulo)
$panel.BackColor = $panelBack
$panel.SuspendLayout(); Set-DoubleBuffer $panel
$form.Controls.Add($panel)
# Columna desplazable a mano: $cont se mueve (sin blit del sistema, que deja rayas con controles transparentes).
$cont = New-Object System.Windows.Forms.Panel; $cont.BackColor = [System.Drawing.Color]::Transparent; $cont.Location = New-Object System.Drawing.Point(0, 0); $cont.Size = New-Object System.Drawing.Size($panelW, $H)
# (launcher115) JUGAR fijo arriba del panel; debajo, la vista desplazable ($view) con $cont dentro (cabecera $script:hd px).
$script:hd = 100
$view = New-Object System.Windows.Forms.Panel; $view.BackColor = [System.Drawing.Color]::Transparent; $view.Location = New-Object System.Drawing.Point(0, $script:hd); $view.Size = New-Object System.Drawing.Size($panelW, [math]::Max(1, $H - $script:hd))
Set-DoubleBuffer $view; $panel.Controls.Add($view)
$cont.Top = -$script:hd
Set-DoubleBuffer $cont; $view.Controls.Add($cont)
$script:scrollY = 0
# (ui055) Barra de desplazamiento de verdad: carril + barra que se arrastra; clic en el carril = salto de pagina.
# (launcher112) Barra fina dibujada (BbScrollBar): redondeada, se ensancha al pasar el raton, arrastre 1:1 sin saltos.
$sbar = New-Object BbScrollBar; $sbar.Size = New-Object System.Drawing.Size($sbW, $H); $sbar.Visible = $false
$panel.Controls.Add($sbar); $sbar.BringToFront()
# Doble bbufer en el panel y su contenido: el desplazamiento se dibuja sin parpadeo ni trompicones.
try { $dbp = [System.Windows.Forms.Control].GetProperty('DoubleBuffered', [System.Reflection.BindingFlags]'Instance,NonPublic'); $dbp.SetValue($panel, $true, $null); $dbp.SetValue($cont, $true, $null); $dbp.SetValue($view, $true, $null) } catch { }
function Set-ScrollY([double]$y) {
	$ph = $panel.ClientSize.Height; $ch = 0
	foreach ($c in $cont.Controls) { if ($c.Visible -and $c.Bottom -gt $ch) { $ch = $c.Bottom } }
	$total = $ch + 24; $max = [math]::Max(0, $total - $ph)
	$script:scrollY = [int][math]::Max(0, [math]::Min($max, [math]::Round($y)))
	$hd = $script:hd; $vh = [math]::Max(1, $ph - $hd)
	if ($view.Top -ne $hd -or $view.Height -ne $vh -or $view.Width -ne $panelW) { $view.SetBounds(0, $hd, $panelW, $vh) }
	$nh = [int][math]::Max($ph, $total); if ($cont.Height -ne $nh) { $cont.Height = $nh }; $nt = -$hd - [int]$script:scrollY; if ($cont.Top -ne $nt) { [BbGfx]::MoveTop($view, $cont, $nt) }
	# (launcher115) barra de arriba abajo de la zona desplazable (debajo de JUGAR hasta el borde inferior)
	$sx = $panelW - $sbar.Width - 2; $sy = $hd + 4; $sh = [math]::Max(1, $vh - 12); if ($sbar.Left -ne $sx -or $sbar.Top -ne $sy -or $sbar.Height -ne $sh) { $sbar.SetBounds($sx, $sy, $sbar.Width, $sh) }
	$sbar.SetState($max, [double]$script:scrollY, $vh); if ($panel.Controls.GetChildIndex($sbar) -ne 0) { $sbar.BringToFront() }
}

# Rueda suave (sin AutoScroll): cada muesca suma 72 px (como antes) al destino; la posicion se acerca a el
# un 22 % por tick (16 ms). La posicion es fraccionaria; Set-ScrollY la redondea al pixel.
$script:scrollTarget = 0.0; $script:scrollPos = 0.0; $script:wheelLast = 0; $script:wheelTau = 90.0
$script:wheelT = New-Object System.Windows.Forms.Timer; $script:wheelT.Interval = 15
function Get-ScrollMax { $ch = 0; foreach ($c in $cont.Controls) { if ($c.Visible -and $c.Bottom -gt $ch) { $ch = $c.Bottom } }; return [double][math]::Max(0, ($ch + 24) - $panel.ClientSize.Height) }
function Add-WheelStep([double]$px) {
	if (-not $script:wheelT.Enabled) { $script:scrollPos = [double]$script:scrollY; $script:scrollTarget = [double]$script:scrollY; $script:wheelLast = [Environment]::TickCount - 16 }
	$script:scrollAnim = $null; $script:wheelTau = 90.0
	$script:scrollTarget = [math]::Max(0.0, [math]::Min((Get-ScrollMax), $script:scrollTarget + $px))
	if ($reduceMotion) { $script:scrollPos = $script:scrollTarget; Set-ScrollY $script:scrollPos; return }
	$script:wheelT.Start()
}
$script:wheelT.Add_Tick({
	# (launcher112) Acercamiento exponencial por tiempo (tau 75 ms): igual de suave aunque un tick llegue tarde.
	$now = [Environment]::TickCount; $dt = [math]::Max(1, [math]::Min(50, $now - $script:wheelLast)); $script:wheelLast = $now
	$d = $script:scrollTarget - $script:scrollPos
	if ([math]::Abs($d) -lt 0.5) { $script:scrollPos = $script:scrollTarget; $script:wheelT.Stop() }
	else { $script:scrollPos += $d * (1 - [math]::Exp(-$dt / $script:wheelTau)) }
	Set-ScrollY $script:scrollPos
})
# (ui055) Arrastrar la barra y clic en el carril.
$script:dragY = -1; $script:dragStart = 0
function Set-ScrollNow([double]$y) { $script:wheelT.Stop(); Set-ScrollY $y; $script:scrollPos = [double]$script:scrollY; $script:scrollTarget = [double]$script:scrollY }
# Arrastre 1:1: BbScrollBar captura el raton (no se pierde al moverse) y la posicion sigue al cursor sin animacion.
# (launcher119) Arrastre suave: la barra sigue al cursor y el contenido la persigue con un suavizado corto (tau 35 ms).
function Drag-ScrollTo([double]$y) {
	if ($reduceMotion) { Set-ScrollNow $y; return }
	if (-not $script:wheelT.Enabled) { $script:scrollPos = [double]$script:scrollY; $script:wheelLast = [Environment]::TickCount - 15 }
	$script:scrollAnim = $null; $script:wheelTau = 35.0
	$script:scrollTarget = [math]::Max(0.0, [math]::Min((Get-ScrollMax), $y)); $script:wheelT.Start()
}
$sbar.add_Scrolled({ Drag-ScrollTo $sbar.Requested })
$sbar.add_PageRequested({ Add-WheelStep ($sbar.PageDir * $view.ClientSize.Height * 0.9) })  # clic en el carril: salto de pagina animado
# Fila de arriba: JUGAR grande; Precompilar / Actualizar / Settings / Salir a su lado.
$fgW = 118  # (launcher121) boton Carpeta del juego a la derecha de JUGAR
$btnPlay = New-Btn (T 'JUGAR' 'PLAY') 10 12 ($gwTop - $fgW - 6) 74 $true
# (ui055) JUGAR: fondo rojo y letra gotica blanca (la primera gotica instalada; si no hay, Segoe UI Black).
$fams = @([System.Drawing.FontFamily]::Families | ForEach-Object { $_.Name })
$goth = @('Old English Text MT', 'UnifrakturMaguntia', 'UnifrakturCook', 'Blackletter', 'Fraktur', 'Blackadder ITC') | Where-Object { $fams -contains $_ } | Select-Object -First 1
if ($goth) { $btnPlay.Font = New-Object System.Drawing.Font($goth, 30, [System.Drawing.FontStyle]::Regular) } else { $btnPlay.Font = New-Object System.Drawing.Font('Segoe UI Black', 22.5, [System.Drawing.FontStyle]::Bold) }
$btnPlay.BackColor = [System.Drawing.Color]::FromArgb(150, 14, 14); $btnPlay.ForeColor = [System.Drawing.Color]::White
$btnPlay.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(90, 8, 8); $btnPlay.FlatAppearance.MouseOverBackColor = [System.Drawing.Color]::FromArgb(185, 24, 24); $btnPlay.FlatAppearance.MouseDownBackColor = [System.Drawing.Color]::FromArgb(120, 10, 10)
$tips.SetToolTip($btnPlay, (T 'Arranca Bloodborne con estos ajustes (Esc cierra el lanzador).' 'Starts Bloodborne with these settings (Esc closes the launcher).'))
$panel.Controls.Add($btnPlay)  # (launcher115) fijo arriba (no se desplaza)
$btnFolder = New-Btn (T "Carpeta`ndel juego" "Game`nfolder") (10 + $gwTop - $fgW) 12 $fgW 74; $btnFolder.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 10, [System.Drawing.FontStyle]::Bold)
$tips.SetToolTip($btnFolder, (T 'Elige la carpeta donde está el juego (la que tiene eboot.bin).' 'Choose the folder where the game is (the one with eboot.bin).'))
$btnFolder.Add_Click({ & $script:pickGame })  # (launcher122) PerformClick no funciona si Examinar esta oculto
$panel.Controls.Add($btnFolder)
$filete = New-Object System.Windows.Forms.Panel; $filete.Location = New-Object System.Drawing.Point(10, 92); $filete.Size = New-Object System.Drawing.Size($gwTop, 2); $filete.BackColor = [System.Drawing.Color]::FromArgb(150, 118, 84, 40); $panel.Controls.Add($filete)
$tbw = 131; $tbx = 320
$btnPre = New-Btn (T 'Precompilar' 'Precompile') $tbx 22 $tbw 54; $btnPre.Font = $fBar
$btnUpd = New-Btn (T 'Actualizar' 'Update') ($tbx + 139) 22 $tbw 54; $btnUpd.Font = $fBar
$btnSettings = New-Btn (T 'Emulador' 'Emulator') ($tbx + 278) 22 $tbw 54; $btnSettings.Font = $fBar
$btnExit = New-Btn (T 'Salir' 'Exit') ($tbx + 417) 22 $tbw 54; $btnExit.Font = $fBar
foreach ($b in @($btnUpd, $btnSettings)) { $cont.Controls.Add($b) }  # Precompilar y Salir quitados (Esc cierra); Actualizar y Settings van abajo
$tips.SetToolTip($btnPre, (T 'shadPS4 compila los shaders mientras juegas y los guarda en su caché (los tirones solo pasan la primera vez). Aquí ves el estado de la caché.' 'shadPS4 compiles shaders while you play and keeps them in its cache (stutters only happen the first time). Shows the cache state.'))
$tips.SetToolTip($btnUpd, $updTip)
$tips.SetToolTip($btnSettings, (T 'Dónde está shadPS4, su versión, actualizar o volver a la versión anterior; carpetas y controles.' 'Where shadPS4 is, its version, update or go back to the previous version; folders and controls.'))
$tips.SetToolTip($btnExit, (T 'Cerrar el lanzador (Esc).' 'Close the launcher (Esc).'))
$btnExit.Add_Click({ $form.Close() })
$btnSettings.Add_Click({ try { Update-EmuInfo } catch { Write-Log "Emulador (info): $_" }; Invoke-SetLayout; try { [void]$fSet.ShowDialog($form) } catch { Write-Log "Ventana Emulador: $_" }; try { Update-Emu; Update-Game } catch { } })
$pbUpd = New-Object System.Windows.Forms.ProgressBar; $pbUpd.Location = New-Object System.Drawing.Point(320, 80); $pbUpd.Size = New-Object System.Drawing.Size(548, 6); $pbUpd.Maximum = 100; $pbUpd.Visible = $false
$cont.Controls.Add($pbUpd)

# ---- Imagen: Resolución, Reescalado, Preset, Nitidez
$gImg = New-Group (T 'Imagen' 'Image') $cont 10 100 $gw 250
$resItems = @((T "Monitor ($(Get-ScreenRes))" "Monitor ($(Get-ScreenRes))")) + $script:resList
$cmbWin = New-DarkCombo $resItems
$cmbWin.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:resList, $script:S.winres) + 1)
$cmbWin.Add_SelectedIndexChanged({ Set-Setting 'winres' $(if ($cmbWin.SelectedIndex -le 0) { 'auto' } else { $script:resList[$cmbWin.SelectedIndex - 1] }) })
$cmbWin.Add_SelectedIndexChanged({ if ($cmbUp) { for ($k = 0; $k -lt $script:upDisp.Count; $k++) { if ($script:upDisp[$k] -ge 0 -and $script:upscalers[$script:upDisp[$k]].Count -gt 3) { $cmbUp.Items[$k] = Up-Label $k } } } })  # (10-29) la interna de cada calidad sigue a la resolucion
[void](Add-Row $gImg (T 'Resolución' 'Resolution') $cmbWin 26 (T 'Resolución de salida (ventana y pantalla completa exclusiva). Puede ser mayor que el monitor (DSR / VSR): el juego se dibuja a esa y la gráfica la reduce. Se escribe en la config de shadPS4 y en custom_configs\<serie>.json.' 'Output resolution (window and exclusive fullscreen). It can be above the monitor (DSR / VSR): the game renders at it and the GPU scales it down. Written to the shadPS4 config and custom_configs\<serial>.json.'))
# Res. interna y modo de reescalado aparte: ocultos (la interna la pone la calidad del reescalado), se conservan para los modos.
$cmbInt = New-DarkCombo $script:resList
$cmbInt.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:resList, $script:S.intres))
$cmbInt.Add_SelectedIndexChanged({ Set-Setting 'intres' $script:resList[$cmbInt.SelectedIndex] })
[void](Put $gImg $cmbInt 124 26 282); $cmbInt.Visible = $false
$cmbUp = New-DarkCombo @(); $cmbUp.DropDownWidth = 360; $cmbUp.MaxDropDownItems = 16
foreach ($d in 0..($script:upDisp.Count - 1)) { [void]$cmbUp.Items.Add((Up-Label $d)) }
$cmbUp.SelectedIndex = Up-ToDisp ([math]::Max(0, [math]::Min([int]$script:S.upscaler, $script:upscalers.Count - 1)))
[void](Add-Row $gImg (T 'Reescalado' 'Upscaling') $cmbUp 58 (T 'FSR 1 (shadPS4), DLSS, FSR 4, FSR 3.1 y XeSS con su calidad (entre paréntesis la resolución interna). DLSS necesita una gráfica NVIDIA RTX; FSR 4, una AMD RX 9000.' 'FSR 1 (shadPS4), DLSS, FSR 4, FSR 3.1 and XeSS with their quality (internal resolution in brackets). DLSS needs an NVIDIA RTX GPU; FSR 4, an AMD RX 9000.'))
$cmbUpMode = New-Combo $script:upModes 124 90 282 'upmode'
[void](Put $gImg $cmbUpMode 124 90 282); $cmbUpMode.Visible = $false
$script:S.modo = '2'  # (launcher124) sin Preset: siempre Personalizado
$cmbMode = New-Combo @($script:modes | ForEach-Object { $_[1] }) 0 0 282 'modo'; $cmbMode.DropDownWidth = 340
# (launcher126) Preset quitado de verdad: el desplegable existe (lo usa la logica) pero no se anade a la ventana.
# (launcher128) Fila "Steam": etiqueta como las demas filas y un boton dibujado del mismo tamano y estilo que los desplegables.
$lbSteam = New-Label 'Steam' 12 124 150 24 $lblCol $fLabel; [void](Put $gImg $lbSteam 12 124 150)
$btnSteam2 = New-Object System.Windows.Forms.Panel
$btnSteam2.Cursor = [System.Windows.Forms.Cursors]::Hand; $btnSteam2.BackColor = $field
$btnSteam2.Location = New-Object System.Drawing.Point(166, 122); $btnSteam2.Size = New-Object System.Drawing.Size(($gw - 166 - $rowR), 26)
$btnSteam2.Tag = $false
$btnSteam2.Add_Paint({ param($s, $e)
	$g = $e.Graphics; $g.SmoothingMode = 'AntiAlias'; $g.TextRenderingHint = 'ClearTypeGridFit'
	$r = New-Object System.Drawing.Rectangle(0, 0, ($s.Width - 1), ($s.Height - 1))
	$hot = [bool]$s.Tag
	$c1 = $(if ($hot) { [System.Drawing.Color]::FromArgb(255, 38, 52, 74) } else { [System.Drawing.Color]::FromArgb(255, 28, 34, 48) })
	$c2 = $(if ($hot) { [System.Drawing.Color]::FromArgb(255, 24, 32, 46) } else { [System.Drawing.Color]::FromArgb(255, 20, 22, 30) })
	$br = New-Object System.Drawing.Drawing2D.LinearGradientBrush($r, $c1, $c2, 90.0); $g.FillRectangle($br, $r); $br.Dispose()
	$pc = $(if ($hot) { [System.Drawing.Color]::FromArgb(255, 214, 178, 96) } else { [System.Drawing.Color]::FromArgb(255, 120, 100, 62) })
	$pen = New-Object System.Drawing.Pen($pc, 1); $g.DrawRectangle($pen, $r); $pen.Dispose()
	# icono: circulo con punto (estilo Steam simplificado)
	$cy = [int]($s.Height / 2); $ic = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 214, 178, 96), 1.6)
	$g.DrawEllipse($ic, 10, ($cy - 7), 14, 14); $g.FillEllipse((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 214, 178, 96))), 14, ($cy - 3), 6, 6); $ic.Dispose()
	$f = New-Object System.Drawing.Font('Segoe UI Semibold', 10, [System.Drawing.FontStyle]::Bold)
	$tc = $(if ($hot) { [System.Drawing.Color]::FromArgb(255, 245, 215, 140) } else { [System.Drawing.Color]::FromArgb(255, 226, 196, 120) })
	$tr = New-Object System.Drawing.Rectangle(32, 0, ($s.Width - 40), $s.Height)
	[System.Windows.Forms.TextRenderer]::DrawText($g, (T 'Añadir a la biblioteca de Steam' 'Add to Steam library'), $f, $tr, $tc, ([System.Windows.Forms.TextFormatFlags]::VerticalCenter -bor [System.Windows.Forms.TextFormatFlags]::Left -bor [System.Windows.Forms.TextFormatFlags]::EndEllipsis))
	$f.Dispose()
})
$btnSteam2.Add_MouseEnter({ $this.Tag = $true; $this.Invalidate() })
$btnSteam2.Add_MouseLeave({ $this.Tag = $false; $this.Invalidate() })
[void](Put $gImg $btnSteam2 166 122 ($gw - 166 - $rowR))
$tips.SetToolTip($btnSteam2, (T 'Añade "Bloodborne" a tu biblioteca de Steam con imágenes (arranca con los últimos ajustes).' 'Adds "Bloodborne" to your Steam library with artwork.'))
$cmbSharp = New-DarkCombo @(0..20 | ForEach-Object { "$($_ * 5) %" })
function Set-Sharp([int]$v) { $cmbSharp.SelectedIndex = [math]::Max(0, [math]::Min(20, [int][math]::Round([math]::Max(0, [math]::Min(100, $v)) / 5))) }
$sv0 = 75; [void][int]::TryParse("$($script:S.sharp)", [ref]$sv0); Set-Sharp $sv0
$cmbSharp.Add_SelectedIndexChanged({ Set-Setting 'sharp' ($cmbSharp.SelectedIndex * 5); Write-SharpLive })
[void](Add-Row $gImg (T 'Nitidez' 'Sharpness') $cmbSharp 154 (T 'Nitidez para TODOS los reescaladores (BB_UPSCALE_SHARPNESS 0-1): RCAS en FSR 1 / FSR 3.1 / FSR 4, paso de nitidez tras DLSS y XeSS. Se aplica en vivo: puedes cambiarla con el juego abierto.' 'Sharpness for EVERY upscaler (BB_UPSCALE_SHARPNESS 0-1): RCAS in FSR 1 / FSR 3.1 / FSR 4, a sharpening pass after DLSS and XeSS. Applied live: you can change it while the game runs.'))
$lbUp = New-Label '' 10 182 400 36 $dim $tinyFont; $lbUp.TextAlign = 'TopLeft'; [void](Put $gImg $lbUp 12 182 ($gw - 24)); Add-Hint $lbUp
# (bb-fg-022) Reescalado de escena experimental (BB_SCENE_UPSCALE): solo la escena 3D a la interna; el parche de
# resolucion se queda en la Resolucion de salida mientras esta marcado (el emulador reduce la escena el mismo).
$chkScene = Put $gImg (New-Check (T 'Reescalado de escena (experimental)' 'Scene upscale (experimental)') 0 0 396 'sceneup') 12 218 396
$chkScene.Visible = $true  # (bb-native-073) visible: BB_SCENE_UPSCALE=native
$tips.SetToolTip($chkScene, (T 'BB_SCENE_UPSCALE=1: solo la escena 3D se dibuja a la resolución interna del reescalado y el reescalador actúa antes del postprocesado; postprocesado y HUD a la Resolución de salida. Solo con DLSS / FSR 4 / FSR 3.1 / XeSS en un modo reducido (no Nativo). Mientras está marcado, la resolución del juego (parche) se queda en la Resolución de salida. Experimental.' 'BB_SCENE_UPSCALE=1: only the 3D scene renders at the upscaler''s internal resolution and the upscaler runs before post-processing; post-processing and HUD at the output Resolution. Only with DLSS / FSR 4 / FSR 3.1 / XeSS in a reduced mode (not Native). While checked, the game resolution (patch) stays at the output Resolution. Experimental.'))

# ---- Frame generation + Anti-Lag / Reflex
$gFg = New-Group 'Frame generation' $cont 10 ($gImg.Bottom + 8) $gw 190  # (launcher133) +36: boton Soulsborne ligero
$cmbFg = New-Combo (Fg-Labels) 0 0 282 'bbfg' $null $script:fgDisp
[void](Add-Row $gFg (T 'Modo' 'Mode') $cmbFg 26 (T 'Genera fotogramas intermedios (BB_FG): AMD FSR 3 x2 o dinámico; NVIDIA DLSS-G x2 / x3 / x4 o dinámico (RTX 40/50). Dinámico ajusta el multiplicador a tu monitor.' 'Generates in-between frames (BB_FG): AMD FSR 3 x2 or dynamic; NVIDIA DLSS-G x2 / x3 / x4 or dynamic (RTX 40/50). Dynamic matches the multiplier to your monitor.'))
$cmbAl = New-Combo @((T 'Auto (AMD: Anti-Lag 2; NVIDIA: Reflex; otras: el nuestro)' 'Auto (AMD: Anti-Lag 2; NVIDIA: Reflex; others: ours)'), 'AMD Anti-Lag 2', 'Anti-Lag (normal)', 'Anti-Lag+ (ultra)', 'No') 0 0 282 'almode'
$cmbRx = New-Combo @((T 'Reflex (recomendado)' 'Reflex (recommended)'), (T 'Anti-Lag (normal, Reflex apagado)' 'Anti-Lag (normal, Reflex off)'), (T 'Anti-Lag+ (ultra, Reflex apagado)' 'Anti-Lag+ (ultra, Reflex off)')) 0 0 282 'aldlssg'
$cmbAl.DropDownWidth = 380; $cmbRx.DropDownWidth = 320
$lbLat = Add-Row $gFg 'Anti-Lag' $cmbAl 58 (T 'Anti-Lag (BB_ANTILAG), como en Demon''s Souls. Auto: AMD Anti-Lag 2 en gráficas AMD; con FSR 3 frame generation en las demás el nuestro; NVIDIA sin frame generation: Reflex. Anti-Lag (normal): el nuestro, como mucho 2 fotogramas en cola. Anti-Lag+ (ultra): 1 fotograma en cola, el menor retraso (pensado para Intel y gráficas sin Anti-Lag 2 / Reflex, se puede elegir en cualquiera). Ligero y Equilibrado van siempre con Anti-Lag+.' 'Anti-Lag (BB_ANTILAG), as in Demon''s Souls. Auto: AMD Anti-Lag 2 on AMD GPUs; with FSR 3 frame generation on others ours; NVIDIA without frame generation: Reflex. Anti-Lag (normal): ours, at most 2 frames queued. Anti-Lag+ (ultra): 1 frame queued, the lowest delay (meant for Intel and GPUs without Anti-Lag 2 / Reflex, selectable on any). Light and Balanced always use Anti-Lag+.')
[void](Put $gFg $cmbRx 166 58 ($gw - 166 - $rowR))
$tips.SetToolTip($cmbRx, (T 'Con DLSS-G (BB_ANTILAG_DLSSG): Reflex (recomendado) o el nuestro, normal o ultra, con Reflex apagado.' 'With DLSS-G (BB_ANTILAG_DLSSG): Reflex (recommended) or ours, normal or ultra, with Reflex off.'))
$lbFg = New-Label '' 10 90 400 28 $dim $tinyFont; $lbFg.TextAlign = 'TopLeft'; [void](Put $gFg $lbFg 12 126 ($gw - 24)); Add-Hint $lbFg  # (launcher136) la ayuda debajo del boton Soulsborne ligero
function Update-LatBox {
	$fg = Get-FgKey; $dl = ($fg -like 'dlss*')
	$cmbAl.Visible = -not $dl; $cmbRx.Visible = $dl; $lbLat.Text = 'Anti-Lag'
	# (bbl3) Ligero / Equilibrado: Anti-Lag+ fijo, como los modos Souls de Demon's Souls.
	$forced = Test-AlForced; $cmbAl.Enabled = -not $forced
	if ($forced -and $cmbAl.SelectedIndex -ne 3) { $script:alHold = $true; $cmbAl.SelectedIndex = 3; $script:alHold = $false }
	elseif (-not $forced -and $cmbAl.SelectedIndex -ne (Get-AlIdx 'almode' 4)) { $script:alHold = $true; $cmbAl.SelectedIndex = (Get-AlIdx 'almode' 4); $script:alHold = $false }
	$lbFg.Text = $(if ($fg -eq 'off') { T 'Sin frame generation: los FPS reales del juego.' 'No frame generation: the real game FPS.' } elseif ($dl) { T "BB_FG=$fg  |  Reflex con DLSS-G (NVIDIA RTX 40/50)." "BB_FG=$fg  |  Reflex with DLSS-G (NVIDIA RTX 40/50)." } else { T "BB_FG=$fg  |  Anti-Lag con FSR 3 (cualquier gráfica)." "BB_FG=$fg  |  Anti-Lag with FSR 3 (any GPU)." })
}
$cmbFg.Enabled = -not (Test-AlForced)
$cmbFg.Add_SelectedIndexChanged({ Set-Setting 'framegen' $(if ([int]$script:S.bbfg -gt 0) { '1' } else { '0' }); Update-LatBox })
$cmbMode.Add_SelectedIndexChanged({ Update-LatBox })
# (launcher133) Soulsborne ligero (como el modo ligero del port de Demon's Souls): un clic pone 30 FPS reales (parche
# 30 FPS++: el juego sigue a su ritmo nativo de 30, la fisica igual que en PS4, solo con un ritmo de fotogramas regular),
# AMD FSR 3 x2 (= 60 en pantalla), DLSS 5 apagado y la Calidad en Normal (PS4). El reescalado se queda como lo tengas.
# Solo cambia los desplegables (sus claves de siempre: fps, bbfg, dlss5, cal); nada de Demon's Souls.
$btnLigero = New-Btn (T 'Soulsborne ligero (30 FPS x2 = 60, gráficos PS4)' 'Soulsborne light (30 FPS x2 = 60, PS4 graphics)') 0 0 ($gw - 24) 30
$btnLigero.Font = $fSmall
[void](Put $gFg $btnLigero 12 90 ($gw - 24))
# (11-13) Personaje sin estela con AMD FSR 3: el personaje sale de la imagen real (BB_FSR3_PLAYER_REAL)
$chkPlayer = New-Check (T 'Personaje sin estela (AMD FSR 3: el personaje a los FPS reales)' 'Player without ghosting (AMD FSR 3: the player at the real FPS)') 0 0 ($gw - 24) 'fgplayer'
[void](Put $gFg $chkPlayer 12 156 ($gw - 24))
$tips.SetToolTip($chkPlayer, (T 'Solo con AMD FSR 3. El personaje principal se toma de la imagen real en vez de inventarlo: sin la estela ni la transparencia en giros y ataques, a cambio de moverse a los FPS reales mientras la cámara va a los de pantalla.' 'AMD FSR 3 only. The main character comes from the real frame instead of being generated: no ghosting on turns and attacks, but it moves at the real FPS while the camera runs at the shown FPS.'))
$tips.SetToolTip($btnLigero, (T 'Un clic: 30 FPS reales del juego original (sin parches de FPS: física y ritmo como en PS4), frame generation AMD FSR 3 x2 (60 en pantalla) con nuestros vectores de movimiento, DLSS 5 No y gráficos como en PS4 (Calidad Normal) sin desenfoque de movimiento ni aberración cromática. El reescalado no cambia. Luego puedes tocar lo que quieras.' 'One click: 30 real FPS of the original game (no FPS patches: PS4 physics and pacing), frame generation AMD FSR 3 x2 (60 on screen) with our own motion vectors, DLSS 5 off and PS4 graphics (Normal quality) without motion blur or chromatic aberration. Upscaling is unchanged. You can change anything afterwards.'))
$btnLigero.Add_Click({
	try {
		$cmbFps.SelectedIndex = 4  # (11-12) 30 del juego original, sin parches de FPS (el dueno: los del parche, no)
		$fgPos = [array]::IndexOf(@($script:fgDisp), 1)  # 1 = AMD FSR 3 x2 en $fgModes
		if ($fgPos -ge 0) { $cmbFg.SelectedIndex = $fgPos }
		$cmbD5.SelectedIndex = 0
		$cmbCal.SelectedIndex = 1  # Normal (PS4)
		$script:applying = $true; try { Set-CalPreset 1 } finally { $script:applying = $false }  # todas las filas de Calidad (como al elegirlo a mano)
		foreach ($pair in $script:qCombos) { if (@('cal_ca', 'cal_blur') -contains $pair[1][0]) { $pair[0].SelectedIndex = 0 } }  # sin aberracion cromatica ni desenfoque de movimiento
		Write-Log 'Soulsborne ligero: 30 FPS originales (sin parches de FPS) / AMD FSR 3 x2 / DLSS 5 No / Calidad Normal (PS4) sin desenfoque ni aberracion cromatica'
	} catch { Write-Log "Soulsborne ligero: no se pudo aplicar: $_" }
})

# ---- DLSS 5 (bb-dlss5-042): las mismas filas, valores y ayudas que en Demon's Souls (ds-menu.cpp)
$gD5 = New-Group 'DLSS 5' $cont 10 ($gFg.Bottom + 8) $gw 158
$script:d5Tip = 'DLSS 5 (Neural Rendering de NVIDIA): mejora la imagen con la IA de NVIDIA después del reescalado. Solo NVIDIA RTX. Necesita tu propio archivo nvngx_dlssnr.dll junto a shadPS4.exe (en bb-port-medicion; no viene incluido): sin él se ve como siempre. Compatibilidad: RTX 5000 sí; RTX 4000 (Ada Lovelace) con el parche, sin estar confirmado al 100%; RTX 3000 y anteriores no (no tienen FP8). Suave, Medio y Fuerte bajan la resolución en los bordes de la pantalla para ahorrar GPU (Suave un poco, Fuerte más); el centro siempre va a resolución completa. Solo con DLSS, XeSS o FSR en Reescalado. Experimental.'
$script:d5FileTip = 'Qué nvngx_dlssnr.dll puede cargar DLSS 5. Oficial: solo si está firmado por NVIDIA (recomendado). No oficial: lo carga aunque no esté firmado (un archivo modificado); bajo tu responsabilidad.'
$script:d5StrTip = 'Intensidad del efecto de DLSS 5: Baja, Media o Alta (la de NVIDIA, por defecto). Más baja: se parece más a la imagen original del juego.'
$cmbD5 = New-Combo @('No', 'Suave', 'Medio', 'Fuerte') 0 0 282 'dlss5'
[void](Add-Row $gD5 'DLSS 5' $cmbD5 26 $script:d5Tip)
$cmbD5File = New-DarkCombo @('Oficial (firmado por NVIDIA)', 'No oficial (bajo tu responsabilidad)')
$cmbD5File.SelectedIndex = $(if ($script:S.dlss5_no_oficial -eq '1') { 1 } else { 0 })
$cmbD5File.Add_SelectedIndexChanged({ Set-Setting 'dlss5_no_oficial' $(if ($cmbD5File.SelectedIndex -eq 1) { '1' } else { '0' }) })
[void](Add-Row $gD5 'Archivo DLSS 5' $cmbD5File 58 $script:d5FileTip)
$script:d5Keys = @('baja', 'media', 'alta')
$cmbD5Str = New-DarkCombo @('Baja', 'Media', 'Alta')
$cmbD5Str.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:d5Keys, "$($script:S.dlss5_intensidad)")); if ([array]::IndexOf($script:d5Keys, "$($script:S.dlss5_intensidad)") -lt 0) { $cmbD5Str.SelectedIndex = 2 }
$cmbD5Str.Add_SelectedIndexChanged({ Set-Setting 'dlss5_intensidad' $script:d5Keys[$cmbD5Str.SelectedIndex] })
[void](Add-Row $gD5 'Intensidad DLSS 5' $cmbD5Str 90 $script:d5StrTip)
$dpKeys = @('K', 'L', 'M')
$cmbDlssP = New-DarkCombo @('K - Transformer 1a gen. (equilibrado)', 'L - Transformer 2a gen. (mas nitido, mas GPU)', 'M - Transformer 2a gen. (rapido, por defecto)')
$dpIx = [array]::IndexOf($dpKeys, "$($script:S.dlsspreset)".ToUpper()); $cmbDlssP.SelectedIndex = $(if ($dpIx -lt 0) { 2 } else { $dpIx })
$cmbDlssP.Add_SelectedIndexChanged({ Set-Setting 'dlsspreset' $dpKeys[$cmbDlssP.SelectedIndex] })
[void](Add-Row $gImg 'Modelo DLSS' $cmbDlssP 90 'Preset de DLSS (Transformer): K, L o M. Solo con DLSS en Reescalado. M es el de Demon''s Souls.')
$lbD5 = New-Label '' 10 154 400 28 $dim $tinyFont; $lbD5.TextAlign = 'TopLeft'; [void](Put $gD5 $lbD5 12 122 ($gw - 24)); Add-Hint $lbD5
function Update-D5Box {
	$on = ($cmbD5.SelectedIndex -gt 0); $cmbD5File.Enabled = $on; $cmbD5Str.Enabled = $on
	$upK = $script:upscalers[(Up-FromDisp $cmbUp.SelectedIndex)][1]
	$lbD5.Text = $(if (-not $on) { 'Sin DLSS 5.' } else { "BB_DLSS5=$($cmbD5.SelectedIndex)  |  necesita nvngx_dlssnr.dll junto a shadPS4.exe (NVIDIA RTX)." })
}
$cmbD5.Add_SelectedIndexChanged({ Update-D5Box })
$cmbUp.Add_SelectedIndexChanged({ Update-D5Box })
Update-D5Box

# ---- Pantalla y otros: Monitor, VSync, Límite FPS, HDR, Contador, Métricas
# ---- Opciones del juego: parches de resolucion y de FPS (despues de DLSS 5)
$gJuego = New-Group (T 'Opciones del juego' 'Game options') $cont 10 ($gD5.Bottom + 8) $gw 94
$cmbResP = New-Combo @((T 'Auto (según el reescalado)' 'Auto (from upscaling)'), '4K (3840x2160)', '1440p (2560x1440)', '1080p (1920x1080)') 0 0 282 'respatch'
[void](Add-Row $gJuego (T 'Parche resolución' 'Resolution patch') $cmbResP 26 (T 'Resolución a la que dibuja el juego (parche de resolución). Auto: la pone la calidad del reescalado. 4K / 1440p / 1080p: fija; el reescalado la sube a la salida.' 'Resolution the game renders at (resolution patch). Auto: set by the upscaling quality. 4K / 1440p / 1080p: fixed; upscaling takes it to the output.'))
$cmbResP.Add_SelectedIndexChanged({ if ($cmbUp) { for ($k = 0; $k -lt $script:upDisp.Count; $k++) { if ($script:upDisp[$k] -ge 0 -and $script:upscalers[$script:upDisp[$k]].Count -gt 3) { $cmbUp.Items[$k] = Up-Label $k } } } })
$gScr = New-Group (T 'Pantalla y otros' 'Display and more') $cont 10 ($gJuego.Bottom + 8) $gw 180
$cmbFs = New-Combo @((T 'Pantalla completa (exclusiva)' 'Fullscreen (exclusive)'), (T 'Completa sin bordes' 'Borderless'), (T 'Ventana' 'Windowed')) 0 0 282 'fsmode'
[void](Add-Row $gScr 'Monitor' $cmbFs 26 (T 'Exclusiva: el monitor cambia a la Resolución elegida (útil con DSR / VSR). Sin bordes: ventana a pantalla completa.' 'Exclusive: the monitor switches to the chosen Resolution (useful with DSR / VSR). Borderless: fullscreen window.'))
$cmbPres = New-Combo @((T 'Activado (Fifo)' 'On (Fifo)'), (T 'Mailbox (por defecto)' 'Mailbox (default)'), (T 'Desactivado (Immediate)' 'Off (Immediate)')) 0 0 282 'present'
[void](Add-Row $gScr 'VSync' $cmbPres 58 (T 'Fifo: sin cortes, algo más de retraso. Mailbox: sin cortes y poco retraso. Immediate: el menor retraso, puede cortarse la imagen.' 'Fifo: no tearing, a bit more delay. Mailbox: no tearing, low delay. Immediate: lowest delay, may tear.'))
$cmbFps = New-Combo @((T '30 FPS (parche 30 FPS++)' '30 FPS (30 FPS++ patch)'), (T '60 FPS (parche 60 FPS++)' '60 FPS (60 FPS++ patch)'), (T 'Desbloqueado (parche Uncap FPS++)' 'Uncapped (Uncap FPS++ patch)'), (T '60 FPS (parche sin deltatime)' '60 FPS (no deltatime patch)'), (T '30 FPS sin parches de FPS (original)' '30 FPS without FPS patches (original)'), (T 'Libre, física a velocidad real (experimental)' 'Free, real-time physics (experimental)')) 0 0 282 'fps'
[void](Add-Row $gJuego (T 'Parche FPS' 'FPS patch') $cmbFps 58 (T 'Bloodborne va a 30. Para 60 hace falta un parche de 60 FPS. 60 FPS++ ajusta la física al tiempo de cada fotograma (si los fps bajan, puede atravesar objetos). Sin deltatime: física fija, el juego va más lento si baja de 60. Sin parches: el juego original a 30.' 'Bloodborne runs at 30. 60 needs the shadPS4 60 FPS patch (user\patches).'))
$chkHdr = Put $gScr (New-Check 'HDR' 0 0 80 'hdr') 12 92 80
# Prueba: sin vectores de movimiento de camara (quita el desenfoque al girar; puede dar fantasmas con frame generation).
$chkMv = Put $gScr (New-Check (T 'Sin desenfoque al girar (prueba)' 'No blur when turning (test)') 12 0 396 'mvzero') 12 120 396
$tips.SetToolTip($chkMv, (T 'Quita el desenfoque al girar la camara (BB_MV=zero). Puede dar fantasmas o arrastre, sobre todo con frame generation. Prueba.' 'Removes the blur when turning the camera (BB_MV=zero). May cause ghosting or smearing, especially with frame generation. Test.'))
$chkVol = Put $gScr (New-Check (T 'Luces volumétricas y niebla (experimental)' 'Volumetric light and fog (experimental)') 12 0 396 'vol') 12 148 396
$tips.SetToolTip($chkVol, (T 'Niebla por distancia y rayos de luz (BB_VOL=1), antes del reescalado. Cuesta algo de GPU.' 'Distance fog and light shafts (BB_VOL=1), before upscaling. Costs some GPU.'))
$chkFpsC = Put $gScr (New-Check (T 'Contador FPS' 'FPS counter') 0 0 150 'showfps') 96 92 150
$chkMet = Put $gScr (New-Check (T 'Métricas' 'Metrics') 0 0 150 'metricas') 252 92 150
$tips.SetToolTip($chkHdr, (T 'HDR de shadPS4 (necesita un monitor HDR con HDR activado en Windows).' 'shadPS4 HDR (needs an HDR monitor with HDR on in Windows).'))
$tips.SetToolTip($chkFpsC, (T 'Contador de FPS de shadPS4 (--show-fps, fuera del modo seguro).' 'shadPS4 FPS counter (--show-fps, outside safe mode).'))
$tips.SetToolTip($chkMet, (T "Ventana de métricas como en Demon's Souls (GPU, VRAM, CPU, RAM, FPS) con metricas-bb.ps1. Solo lee: funciona también en modo seguro. Se cierra al cerrar shadPS4." "Metrics window like Demon's Souls (GPU, VRAM, CPU, RAM, FPS) with metricas-bb.ps1. Read-only: works in safe mode too. Closes when shadPS4 exits."))

# ---- (launcher131) Idioma del juego: Bloodborne no tiene ajuste de voces aparte; texto y voces siguen el idioma del
# sistema PS4 (shadPS4: [Settings] consoleLanguage / console_language). Se escribe al JUGAR (Set-GameLanguage).
$gGL = New-Group (T 'Idioma del juego' 'Game language') $cont 10 ($gJuego.Bottom + 8) $gw 94
$cmbGL = New-DarkCombo @()
[void]$cmbGL.Items.Add((T 'Igual que Windows' 'Same as Windows'))
foreach ($pl in $script:ps4Langs) { [void]$cmbGL.Items.Add("$($pl[1])" + $(if ($pl[2]) { T '  ·  texto y voces' '  ·  text and voices' } else { T '  ·  texto (voces en inglés)' '  ·  text (English voices)' })) }
$glIdx = 0
if ("$($script:S.gamelang)" -ne 'auto') { $gid = Get-GameLangId; for ($k = 0; $k -lt $script:ps4Langs.Count; $k++) { if ([int]$script:ps4Langs[$k][0] -eq $gid) { $glIdx = $k + 1 } } }
$cmbGL.SelectedIndex = $glIdx
$glTip = (T 'Idioma de textos y voces de Bloodborne (idioma del sistema PS4 en shadPS4). Igual que Windows: el de tu Windows. Se aplica al pulsar JUGAR.' 'Bloodborne text and voice language (PS4 system language in shadPS4). Same as Windows: your Windows language. Applied when you press PLAY.')
[void](Add-Row $gGL (T 'Texto y voces' 'Text and voices') $cmbGL 26 $glTip)
$cmbGL.DropDownWidth = [math]::Max($cmbGL.Width, 330); $cmbGL.MaxDropDownItems = 14
$cmbGL.Add_SelectionChangeCommitted({ Set-Setting 'gamelang' $(if ($cmbGL.SelectedIndex -le 0) { 'auto' } else { "$($script:ps4Langs[$cmbGL.SelectedIndex - 1][0])" }) })
$lbGLn = New-Label (T 'Bloodborne no separa texto y voces: los dos siguen el idioma del sistema. Voces propias en 8 idiomas (inglés, japonés, español de España y de Latinoamérica, francés, alemán, italiano y portugués de Brasil); en los demás el texto va traducido y las voces suenan en inglés. Se aplica al pulsar JUGAR.' 'Bloodborne does not split text and voices: both follow the system language. Own voices in 8 languages (English, Japanese, Spanish from Spain and Latin America, French, German, Italian and Brazilian Portuguese); the others have translated text with English voices. Applied when you press PLAY.') 12 60 ($gw - 24) 40 $dim $tinyFont
[void](Put $gGL $lbGLn 12 60 ($gw - 24)); Add-Hint $lbGLn

# ---- Calidad
$script:qCombos = @()
$gQ = New-Group (T 'Calidad' 'Quality') $cont 10 ($gScr.Bottom + 8) $gw (26 + 32 * ($script:quality.Count + 3) + 70)
$cmbCal = New-Combo @($script:calLevels + @((T 'Personalizado' 'Custom'), (T 'Ultra bajo' 'Ultra low'))) 0 0 282 'cal'  # 5 = Ultra bajo al final (no mueve Personalizado = 4)
[void](Add-Row $gQ (T 'Preset calidad' 'Quality preset') $cmbCal 26 (T 'Bajo / Normal (= PS4) / Alto / Ultra / Ultra bajo: pone todas las filas (Ultra bajo: también texturas Ultra bajo y anisotrópico 1x). Tocar una fila = Personalizado.' 'Low / Normal (= PS4) / High / Ultra / Ultra low: sets every row (Ultra low: also Ultra low textures and 1x anisotropic). Changing a row = Custom.'))
$qy = 58
foreach ($q in $script:quality) {
	$cb = New-Combo $q[2] 0 0 282 $q[0]
	[void](Add-Row $gQ $q[1] $cb $qy $q[5])
	$script:qCombos += , @($cb, $q)
	$qy += 32
}
# (tex) Calidad de texturas: sesgo de LOD, solo texturas de color. Fila propia, no es parche de calidad.
$cmbTex = New-Combo @((T 'Bajo' 'Low'), (T 'Normal (PS4)' 'Normal (PS4)'), (T 'Alto' 'High'), (T 'Ultra' 'Ultra'), (T 'Ultra bajo' 'Ultra low')) 0 0 282 'tex'
[void](Add-Row $gQ (T 'Calidad de texturas' 'Texture quality') $cmbTex $qy (T 'Sesgo de LOD de las texturas de color: Bajo = más borroso, Ultra = más nítido. Normal = PS4. Se aplica al jugar (BB_TEX_QUALITY).' 'Color texture LOD bias: Low = blurrier, Ultra = sharper. Normal = PS4. Applied on play (BB_TEX_QUALITY).'))
$qy += 32
# (aniso) Filtrado anisotropico: 1x, 4x, 8x, 16x -> BB_ANISO (1, 4, 8, 16). Fila propia, no es parche de calidad.
$cmbAniso = New-Combo @('1x', '4x', '8x', '16x') 0 0 282 'aniso'
[void](Add-Row $gQ (T 'Filtrado anisotrópico' 'Anisotropic filtering') $cmbAniso $qy (T 'Nitidez de las texturas vistas en ángulo. 1x = la de PS4 (más borrosa). 16x = máximo, cuesta poco. Se aplica al jugar (BB_ANISO).' 'Sharpness of textures seen at an angle. 1x = PS4 look (blurrier). 16x = maximum, cheap on most GPUs. Applied on play (BB_ANISO).'))
$qy += 32
$lbQ = New-Label '' 10 ($qy + 2) 400 62 $dim $tinyFont; [void](Put $gQ $lbQ 12 ($qy + 2) ($gw - 24)); Add-Hint $lbQ
$tips.SetToolTip($lbQ, (T 'Al JUGAR el lanzador activa / desactiva solo estos parches de calidad (copia .bak-calidad antes). Ningún otro parche.' 'On PLAY the launcher toggles only these quality patches (.bak-calidad copy first). No other patch.'))

# ---- Rendimiento / PC
$gPc = New-Group (T 'Rendimiento / PC' 'Performance / PC') $cont 10 ($gQ.Bottom + 8) $gw 132
$chkSafe = Put $gPc (New-Check (T 'Modo seguro: solo cambia la resolución que elijas' 'Safe mode: only changes the resolution you pick') 0 0 396 'safemode') 12 24 ($gw - 24); $chkSafe.ForeColor = $gold
$tips.SetToolTip($chkSafe, (T 'Activado: JUGAR arranca shadPS4 con el juego (como BB_Launcher). Solo escribe la resolución, el modo de pantalla y el VSync que hayas cambiado tú en esta sesión, más las variables BB_* (reescalado, nitidez, frame generation). Nunca parches ni mods. Partidas: solo el byte del sound fix.' 'On: PLAY starts shadPS4 with the game (like BB_Launcher). Only writes the resolution, screen mode and VSync you changed this session, plus the BB_* variables (upscaling, sharpness, frame generation). Never patches or mods. Saves: only the sound-fix byte.'))
$chkSfx = Put $gPc (New-Check (T 'Sound fix' 'Sound fix') 0 0 196 'soundfix') 12 54 196
$chkDirect = Put $gPc (New-Check (T 'Iniciar directamente (sin launcher)' 'Start directly (no launcher)') 0 0 196 'inicio_directo') 210 54 ($gw - 222)
$tips.SetToolTip($chkDirect, (T 'Al abrir el lanzador arranca el juego con los últimos ajustes, sin ventana. Para volver a ver el lanzador: mantén Mayús al abrirlo o usa -menu.' 'Opening the launcher starts the game with the last settings, no window. To see the launcher again: hold Shift while opening it or use -menu.'))
$tips.SetToolTip($chkSfx, (T 'Antes de cada JUGAR (también en modo seguro) pone a 1 el byte 0x204E de la partida (userdata0010), como BB_Launcher: sin él no suenan los golpes. Copia .bak-sfx-original la primera vez.' 'Before each PLAY (safe mode too) sets save byte 0x204E (userdata0010) to 1, like BB_Launcher: without it hit sounds are silent. Makes a .bak-sfx-original copy the first time.'))
$btnOpt = Put $gPc (New-Btn (T 'Optimizar según tu equipo' 'Optimize for your PC') 0 0 396 34) 12 86 ($gw - 24)
$tips.SetToolTip($btnOpt, (T 'Elige resolución y reescalado según tu gráfica (te pregunta antes).' 'Picks resolution and upscaling for your GPU (asks first).'))

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
	$fHead = New-Object System.Drawing.Font('Segoe UI Black', (Snap-Px (16 * $fs)), [System.Drawing.FontStyle]::Bold)
	$fCap = New-Object System.Drawing.Font('Segoe UI Semibold', (Snap-Px (10.5 * $fs))); $fVal = New-Object System.Drawing.Font('Segoe UI Semibold', (Snap-Px (13 * $fs)), [System.Drawing.FontStyle]::Bold)
	$fTxt = New-Object System.Drawing.Font('Segoe UI', (Snap-Px (12 * $fs)))
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
$cardY = $gPc.Bottom + 10; $cardH = 260
$pcCard = New-Object System.Windows.Forms.PictureBox
$pcCard.BackColor = [System.Drawing.Color]::Transparent
$pcCard.Location = New-Object System.Drawing.Point(10, $cardY); $pcCard.Size = New-Object System.Drawing.Size($gw, $cardH)
try {
	foreach ($fs in 1.0, 0.9, 0.8, 0.7, 0.6) {
		if ($pcCard.Image) { $pcCard.Image.Dispose() }
		$pcCard.Image = New-PcCard $gw $cardH $fs
		if ($script:pcEndY -le $cardH - 6) { break }
	}
} catch { Write-Log "Tarjeta TU EQUIPO: $_" }
$cont.Controls.Add($pcCard); $script:pcCardRef = $pcCard
# Abajo de todo, en la misma columna: Actualizar y Settings.
$bw2 = [int](($gw - 8) / 2); $byy = $pcCard.Bottom + 14
$btnUpd.Location = New-Object System.Drawing.Point(10, $byy); $btnUpd.Size = New-Object System.Drawing.Size($bw2, 46); $btnUpd.Font = $fBtn
$btnSettings.Location = New-Object System.Drawing.Point((18 + $bw2), $byy); $btnSettings.Size = New-Object System.Drawing.Size($bw2, 46); $btnSettings.Font = $fBtn
$pbUpd.Location = New-Object System.Drawing.Point(10, ($byy + 52)); $pbUpd.Size = New-Object System.Drawing.Size(($gw - 20), 6)
# Orden: lo que se toca a menudo arriba (Imagen, Frame generation, Calidad); Pantalla y Rendimiento
# plegados (un clic en el titulo los abre con animacion).
$script:bottomReady = $true
$script:stackGroups = @($gImg, $gFg, $gD5, $gJuego, $gGL, $gQ, $gScr, $gPc)  # (launcher129) Opciones del juego entra en la pila
foreach ($cg in @($gScr, $gPc)) { $cg.Tag.Collapsed = $true; $cg.Tag.Progress = 0.0; $cg.Height = 34; $cg.Tag.Header.Invalidate() }
Fit-AllGroups

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
# (launcher129) Boton "Clave de trofeos" a la derecha de Trofeos: abre siempre el cuadro de la clave (aunque ya haya una).
$tkeyW = 176
$btnTKey = New-Btn (T 'Clave de trofeos' 'Trophy key') (24 + $trophyW + 8) ($H - 70) $tkeyW 46
$btnTKey.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 11, [System.Drawing.FontStyle]::Bold)
$tips.SetToolTip($btnTKey, (T 'Ver o cambiar la clave de trofeos de PS4 (keys.json de shadPS4).' 'View or change the PS4 trophy key (shadPS4 keys.json).'))
$btnTKey.Add_Click({ try { Show-TrophyKeyPrompt -Force } catch { Write-Log "Clave de trofeos: $_" } })
$form.Controls.Add($btnTKey)
# (launcher112) Distintivo de version no soportada (no bloquea JUGAR).
$lbBadge = New-Object System.Windows.Forms.Label; $lbBadge.AutoSize = $false; $lbBadge.Location = New-Object System.Drawing.Point(24, ($H - 70 - 54)); $lbBadge.Size = New-Object System.Drawing.Size($trophyW, 44)
$lbBadge.BackColor = [System.Drawing.Color]::FromArgb(150, 14, 14); $lbBadge.ForeColor = [System.Drawing.Color]::White; $lbBadge.Font = $boldSmall; $lbBadge.TextAlign = 'MiddleLeft'
$lbBadge.Padding = New-Object System.Windows.Forms.Padding(10, 0, 6, 0); $lbBadge.Visible = $false
$lbBadge.Add_Resize({ param($s, $e) try { $rp = New-RoundPath 0 0 $s.Width $s.Height 14; $s.Region = New-Object System.Drawing.Region($rp); $rp.Dispose() } catch { } })
$form.Controls.Add($lbBadge); $lbBadge.BringToFront()
try { $rp = New-RoundPath 0 0 $lbBadge.Width $lbBadge.Height 14; $lbBadge.Region = New-Object System.Drawing.Region($rp); $rp.Dispose() } catch { }
function Update-SupportBadge($g) {
	$w = Get-SupportWarning $g
	$lbBadge.Text = $(if ($w) { [string][char]0x26A0 + '  ' + $w } else { '' }); $lbBadge.Visible = [bool]$w
	$was = $script:badgeOn; $script:badgeOn = [bool]$w
	if ($was -ne $script:badgeOn -and $script:layoutReady) { Apply-Layout }  # el distintivo cambia el hueco reservado
	if ($w) { $tips.SetToolTip($lbBadge, (T 'El parche de resolución, el reescalado y los arreglos están hechos para el ejecutable de CUSA03173 1.09. Puedes jugar igualmente.' 'The resolution patch, upscaling and fixes are made for the CUSA03173 1.09 executable. You can still play.')) }
}
$script:badgeOn = $false; $script:layoutReady = $false
$script:trW = $trophyW; $script:trTop = $H - 70; $script:trListX = 24; $script:trListBot = $H - 76; $script:trBelow = $false
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
$gTrophy.Controls.Add($lvTr); Set-DoubleBuffer $gTrophy; Set-DoubleBuffer $lvTr  # (11-13) sin parpadeo al abrir / cerrar Trofeos
$script:trSummary = ''
function Update-TrophyButton { $arrow = $(if ($gTrophy.Visible -and $script:trOpen) { '^' } else { 'v' }); $btnTrophy.Text = (T 'Trofeos' 'Trophies') + $(if ($script:trSummary) { '   ' + $script:trSummary } else { '' }) + '  ' + $arrow }
# (launcher113) La lista de trofeos sale de detras del boton (crece hacia arriba, 220 ms ease-out) y vuelve igual.
$script:trOpen = $false; $script:trAnim = $null; $script:trListH = $trophyListH
$trT = New-Object System.Windows.Forms.Timer; $trT.Interval = 15
function Set-TrophyH([int]$h) { $h = [math]::Max(1, $h); $gTrophy.SetBounds($script:trListX, ($script:trListBot - $h), $script:trW, $h) }
$trT.Add_Tick({
	$a = $script:trAnim; if (-not $a) { $trT.Stop(); return }
	$t = [math]::Min(1.0, ([Environment]::TickCount - $a.T0) / 260.0); $e = 1 - [math]::Pow(1 - $t, 5)
	Set-TrophyH ([int]($a.From + ($a.To - $a.From) * $e))
	if ($t -ge 1) { $trT.Stop(); $script:trAnim = $null; if (-not $script:trOpen) { $gTrophy.Visible = $false; Set-TrophyH $script:trListH } else { $lvTr.Visible = $true; $lvTr.Refresh() }; Update-TrophyButton }
})
$btnTrophy.Add_Click({
	$script:trOpen = -not $script:trOpen
	if ($reduceMotion) { $gTrophy.Visible = $script:trOpen; Set-TrophyH $script:trListH; if ($gTrophy.Visible) { $gTrophy.BringToFront(); $btnTrophy.BringToFront() }; Update-TrophyButton; return }
	$from = $(if ($gTrophy.Visible) { $gTrophy.Height } else { 1 }); if (-not $gTrophy.Visible) { Set-TrophyH 1 }
	if ($script:trOpen) { $gTrophy.Visible = $true; $gTrophy.BringToFront(); $btnTrophy.BringToFront() }
	$lvTr.Visible = $false; $script:trAnim = [pscustomobject]@{ From = $from; To = $(if ($script:trOpen) { $script:trListH } else { 1 }); T0 = [Environment]::TickCount }
	Update-TrophyButton; $trT.Start()
})

# ---- version abajo a la izquierda
# Version abajo a la izquierda: panel opaco que repinta el trozo de fondo debajo y dibuja el texto con ClearType
# (una etiqueta transparente sobre el fondo sale en escala de grises). Alineada al margen de la columna izquierda (24 px).
$lblVer = New-Object System.Windows.Forms.Panel
$lblVer.Size = New-Object System.Drawing.Size(600, 18); $lblVer.ForeColor = [System.Drawing.Color]::FromArgb(200, 194, 180)
$lblVer.Font = New-Object System.Drawing.Font('Segoe UI', 9)  # 9 pt = 12 px (entero a 100 %); 8 pt daba 10,7 px
$lblVer.Location = New-Object System.Drawing.Point(24, ($H - 20))
$lblVer.Add_Paint({ param($s, $e)
	$g = $e.Graphics
	if ($script:bgBmp) { $g.DrawImage($script:bgBmp, (New-Object System.Drawing.Rectangle(0, 0, $s.Width, $s.Height)), (New-Object System.Drawing.Rectangle($s.Left, $s.Top, $s.Width, $s.Height)), [System.Drawing.GraphicsUnit]::Pixel) } else { $g.Clear($form.BackColor) }
	[System.Windows.Forms.TextRenderer]::DrawText($g, [string]$s.Tag, $s.Font, (New-Object System.Drawing.Point(0, 1)), $s.ForeColor, ([System.Windows.Forms.TextFormatFlags]::NoPadding -bor [System.Windows.Forms.TextFormatFlags]::NoPrefix))
})
$lblVer.Visible = $false  # (launcher134) ya no se anade a la ventana: nada se dibuja debajo de Trofeos / Clave de trofeos
function Update-VerLabel {
	$ev = ''; try { if ($script:S.emu -and (Test-Path -LiteralPath $script:S.emu)) { $ev = Get-EmuLabel $script:S.emu } } catch { }
	$lblVer.Tag = (T 'Lanzador Bloodborne v2' 'Bloodborne launcher v2') + $(if ($ev) { '  -  shadPS4: ' + $ev } else { '' }) + (T '   |   Esc: salir   |   launcher.log junto al lanzador' '   |   Esc: exit   |   launcher.log next to the launcher')
	$lblVer.Width = [System.Windows.Forms.TextRenderer]::MeasureText([string]$lblVer.Tag, $lblVer.Font).Width + 4; $lblVer.Invalidate()
}

# Desplegables dibujados a mano (estilo Demon's Souls).
$script:cmbUpRef = $cmbUp; $script:cmbFgRef = $cmbFg
foreach ($c in @($cmbGame, $cmbEmu, $cmbWin, $cmbInt, $cmbFs, $cmbPres, $cmbFps, $cmbUp, $cmbUpMode, $cmbMode, $cmbFg, $cmbAl, $cmbRx, $cmbSharp, $cmbD5, $cmbD5File, $cmbD5Str, $cmbDlssP)) { Set-ComboStyle $c }
foreach ($pair in $script:qCombos) { Set-ComboStyle $pair[0] }
Set-ComboStyle $cmbCal; Set-ComboStyle $cmbTex; Set-ComboStyle $cmbAniso; Set-ComboStyle $cmbResP; Set-ComboStyle $cmbGL

function Update-Up {
	if ($cmbUp.SelectedIndex -ge 0 -and $script:upDisp[$cmbUp.SelectedIndex] -lt 0) { $cmbUp.SelectedIndex = [math]::Min($cmbUp.SelectedIndex + 1, $script:upDisp.Count - 1); return }
	$i = Up-FromDisp $cmbUp.SelectedIndex; $u = $script:upscalers[$i]
	$exp = ($script:S.experimental -eq '1')
	if (-not $u[2] -and -not $exp) {
		# Opcion aun no disponible: vuelve a la ultima real.
		$cmbUp.SelectedIndex = Up-ToDisp ([math]::Max(0, [math]::Min([int]$script:S.upscaler, 1))); return
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
# (bbl2) Compara cada fila de Calidad con los parches activos (solo lee) y dice que cambiar en shadPS4 / BB_Launcher.
$script:qPatches = @()
function Update-CalStatus {
	$on = @(); $off = @(); $miss = @()
	foreach ($pair in $script:qCombos) {
		$q = $pair[1]; $lv = [math]::Max(0, [math]::Min($pair[0].SelectedIndex, 4))
		$want = $q[3][$lv]; if ($want -eq 'GRID') { $want = Get-GridPatch }
		$fam = @($script:qPatches | Where-Object { $_.Name -match $q[4] -and $_.VerOk })
		$en = @($fam | Where-Object { $_.Enabled })
		foreach ($p in $en) { if (-not $want -or $p.Name -notmatch $want) { $off += $p.Name } }
		if ($want) {
			if (-not @($fam | Where-Object { $_.Name -match $want }).Count) { $miss += "$($q[1])" }
			elseif (-not @($en | Where-Object { $_.Name -match $want }).Count) { $on += @($fam | Where-Object { $_.Name -match $want })[0].Name }
		}
	}
	$t = @()
	if ($on.Count) { $t += (T 'Activa: ' 'Enable: ') + (($on | Select-Object -Unique) -join ', ') }
	if ($off.Count) { $t += (T 'Desactiva: ' 'Disable: ') + (($off | Select-Object -Unique) -join ', ') }
	if ($miss.Count) { $t += (T 'Falta el parche: ' 'Missing patch: ') + ($miss -join ', ') }
	$lbQ.Text = $(if (-not $script:qPatches.Count) { T 'Sin parches leídos: descarga los parches en shadPS4 (user\patches).' 'No patches read: download the patches in shadPS4 (user\patches).' } elseif ($t.Count) { (T 'Al JUGAR: ' 'On PLAY: ') + ($t -join '. ') } else { T 'Los parches activos coinciden con la calidad elegida.' 'Enabled patches match the chosen quality.' })
	$lbQ.ForeColor = $(if ($t.Count) { $gold } else { $dim })
}
function Update-Quality($g) {
	$exp = ($script:S.experimental -eq '1')
	$patches = @(); if ($g.Serial) { $patches = @(Get-Patches (Get-UserDir $script:S.emu) $g.Serial $g.Version) }
	$script:qPatches = $patches; Update-CalStatus
	# Estado ACTUAL de los parches leido de los xml (si el usuario no ha cambiado la fila en esta sesion).
	$script:applying = $true
	try {
		$m60 = @($patches | Where-Object { $_.Name -match $script:pat60 -and $_.VerOk }); if (-not $m60.Count) { $m60 = @($patches | Where-Object { $_.Name -match $script:pat60 }) }
		$unl = @($patches | Where-Object { $_.Enabled -and $_.Name -match $script:patUnlock })
		$script:fpsPatchActive = ($unl.Count -gt 0)
		# (fps120) indice segun el parche de FPS activo (6 opciones); si no hay ninguno, 30 original
		$fpsNames = @('30 FPS++', '60 FPS++', 'Uncap FPS++', '60FPS (no deltatime)', '', 'Uncap FPS (consistent dt)')
		$onN = @($patches | Where-Object { $_.Enabled } | ForEach-Object { $_.Name })
		$fx = -1; for ($k = 5; $k -ge 0; $k--) { if ($fpsNames[$k] -and ($onN -contains $fpsNames[$k])) { $fx = $k; break } }
		if ($fx -lt 0) { $fx = 4 }
		$fi = -1; if ("$($script:S.fps)" -match '^[0-5]$') { $fi = [int]$script:S.fps }
		$cmbFps.SelectedIndex = $(if ($fi -ge 0) { $fi } else { $fx })  # lo elegido en launcher.ini manda; los xml solo si no hay valor
		$cmbFps.Enabled = $true  # (fps120) el lanzador ya escribe el parche de FPS al JUGAR
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
	try { Update-SupportBadge $g } catch { }
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
	try { Update-EmuInfo } catch { Write-Log "Emulador (info): $_" }  # (launcher132)
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
$script:pickGame = {
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
}
$btnGame.Add_Click($script:pickGame)
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
	$msg = (T "Perfil: $($t[1])`n$($t[5])$note`n`n¿Aplicar? (resolución, FPS y reescalado)" "Profile: $($t[1])`n$($t[5])$note`n`nApply? (resolution, FPS and upscaling)")
	if ([System.Windows.Forms.MessageBox]::Show($msg, (T 'Optimizar según tu equipo' 'Optimize for your PC'), 'YesNo', 'Question') -ne 'Yes') { return }
	$cmbWin.SelectedIndex = 0
	$cmbInt.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:resList, $t[4]))
	foreach ($k in @('intres', 'upscaler', 'sharp', 'fsmode')) { Mark-Changed $k -Auto }  # el usuario ha pulsado Sí
	$cmbUp.SelectedIndex = Up-ToDisp $t[3]; Set-Sharp 75
	$cmbFs.SelectedIndex = 1
	Write-Log "Optimizar: perfil $($t[1]) fps=$fps"
})
$script:doSteam = {
	if ([System.Windows.Forms.MessageBox]::Show((T 'Se cerrará Steam un momento para añadir "Bloodborne" a tu biblioteca (con imágenes). ¿Seguir?' 'Steam will close for a moment to add "Bloodborne" to your library (with artwork). Continue?'), 'Steam', 'YesNo', 'Question') -ne 'Yes') { return }
	$form.Cursor = 'WaitCursor'
	try { $r = Add-ToSteam } catch { $r = "$_"; Write-Log "Steam: $_" }
	$form.Cursor = 'Default'
	[void][System.Windows.Forms.MessageBox]::Show($r, 'Steam', 'OK', 'Information')
}
$btnSteam2.Add_Click({ & $script:doSteam })
$script:doUpdate = {
	$form.Cursor = 'WaitCursor'
	try { $rel = Get-ShadRelease } catch { $form.Cursor = 'Default'; Write-Log "GitHub: $_"; [void][System.Windows.Forms.MessageBox]::Show((T "No se pudo consultar GitHub:`n$_" "Could not query GitHub:`n$_"), 'shadPS4', 'OK', 'Warning'); return }
	$form.Cursor = 'Default'
	if (-not $rel.Asset) { [void][System.Windows.Forms.MessageBox]::Show((T "La versión $($rel.Version) no tiene zip de Windows oficial." "Release $($rel.Version) has no official Windows zip."), 'shadPS4', 'OK', 'Information'); return }
	$have = @($script:emuList | Where-Object { (Split-Path -Leaf (Split-Path -Parent $_)) -match ('^v\.?' + [regex]::Escape($rel.Version) + '(-\d+)?$') })
	$msg = (T "Última versión oficial: $($rel.Version)  ($($rel.Date.Substring(0, [math]::Min(10, $rel.Date.Length))))`n$($rel.Asset.name)  ($([math]::Round($rel.Asset.size / 1MB)) MB)`nCambios: $($rel.Url)" "Latest official release: $($rel.Version)  ($($rel.Date.Substring(0, [math]::Min(10, $rel.Date.Length))))`n$($rel.Asset.name)  ($([math]::Round($rel.Asset.size / 1MB)) MB)`nChangelog: $($rel.Url)")
	if ($have.Count) { $msg += "`n`n" + (T "Ya la tienes: $($have[0])" "You already have it: $($have[0])") }
	$msg += "`n`n" + (T "AVISO: las versiones nuevas pueden guardar las partidas en otro sitio (parecerá partida nueva) y usar otro formato de parches. Tu versión actual no se toca y sigue siendo la elegida." "WARNING: newer versions may store saves elsewhere (it will look like a new game) and use a different patch format. Your current build is kept and stays selected.")
	$msg += "`n" + (T "Antes se guarda una copia del emulador actual (_anterior) para 'Volver a la versión anterior'." "A copy of the current emulator is saved first (_anterior) for 'Back to previous version'.")  # (launcher132)
	$msg += "`n`n" + (T "Sí: instalar en una carpeta nueva (la actual no se toca)`nNo: ver los cambios en el navegador`nCancelar: nada" "Yes: install to a new folder (current one is kept)`nNo: open the changelog in the browser`nCancel: nothing")
	$r = [System.Windows.Forms.MessageBox]::Show($msg, (T 'Actualizar shadPS4' 'Update shadPS4'), 'YesNoCancel', 'Question')
	if ($r -eq 'No') { if ($rel.Url) { Start-Process $rel.Url }; return }
	if ($r -ne 'Yes') { return }
	# (launcher132) Copia del emulador actual antes de actualizar (Volver a la version anterior).
	try { [void](Backup-EmuFiles (Get-ActiveEmuExe) "Actualizar $($rel.Version)") } catch {
		Write-Log "Copia anterior: $_"
		if ([System.Windows.Forms.MessageBox]::Show((T "No se pudo guardar la copia del emulador actual:`n$_`n`n¿Actualizar igualmente?" "Could not save a copy of the current emulator:`n$_`n`nUpdate anyway?"), 'shadPS4', 'YesNo', 'Warning') -ne 'Yes') { return }
	}
	$pb = $(if ($fSet.Visible) { $pbEmu } else { $pbUpd })
	$pb.Value = 0; $pb.Visible = $true; if ($fSet.Visible) { Invoke-SetLayout }
	$btnEmuUpd.Enabled = $false; $btnRollback.Enabled = $false; $btnUpd.Enabled = $false; $form.Cursor = 'WaitCursor'; $fSet.Cursor = 'WaitCursor'
	try { $exe = Install-ShadRelease $rel $pb } catch { $exe = $null; Write-Log "Actualizar shadPS4: $_"; [void][System.Windows.Forms.MessageBox]::Show((T "No se pudo instalar:`n$_" "Could not install:`n$_"), 'shadPS4', 'OK', 'Error') }
	$pb.Visible = $false; if ($fSet.Visible) { Invoke-SetLayout }
	$btnEmuUpd.Enabled = $true; $btnUpd.Enabled = $true; $form.Cursor = 'Default'; $fSet.Cursor = 'Default'
	try { Update-EmuInfo } catch { }
	if (-not $exe) { return }
	[void][System.Windows.Forms.MessageBox]::Show((T "Instalada en:`n$exe`n`nSigues con tu versión de siempre. Para probar la nueva, elígela en la lista (y vuelve a la tuya igual)." "Installed to:`n$exe`n`nYou stay on your usual build. To try the new one, pick it in the list (and pick yours to go back)."), 'shadPS4', 'OK', 'Information')
	Fill-Lists
}
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
$btnX.Size = New-Object System.Drawing.Size(48, 40); $btnX.Location = New-Object System.Drawing.Point(($W - 56), 8)  # (launcher113) arriba a la derecha (Apply-Layout la aparta del titulo)
$form.Controls.Add($btnX); $btnX.BringToFront(); $btnX.Add_Click({ $form.Close() })  # cruz para cerrar (Esc tambien)

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
function Set-CalPreset([int]$lv) {
	$cmbCal.SelectedIndex = $lv
	if ($lv -le 3) { foreach ($pair in $script:qCombos) { $pair[0].SelectedIndex = $lv } }
	elseif ($lv -eq 5) { foreach ($pair in $script:qCombos) { $pair[0].SelectedIndex = 4 }; $cmbTex.SelectedIndex = 4; $cmbAniso.SelectedIndex = 0 }  # Ultra bajo: un paso bajo Bajo en todo
	Update-CalStatus
}
function Apply-Mode {
	$m = Get-Mode
	Update-ModeInfo
	if ($m[0] -eq 'personalizado') { $cmbFg.Enabled = $true; return }
	$script:applying = $true
	try {
		$cmbInt.SelectedIndex = [math]::Max(0, [array]::IndexOf($script:resList, $m[5]))
		# El modo NO toca parches ni mods: el parche de 60 (necesario para 45/60) solo cambia si el usuario cambia la fila FPS.
		if ($m[6] -eq 1 -and $cmbFps.SelectedIndex -eq 0) { Write-Log "Modo $($m[0]): necesita el parche de 60 FPS activado (no se activa solo)" }
		$cmbUp.SelectedIndex = Up-ToDisp $m[7]; $cmbUpMode.SelectedIndex = $m[8]; Set-Sharp $m[9]
		$fgU = $(if ($m[3]) { [math]::Max(0, [array]::IndexOf(@($script:fgModes | ForEach-Object { $_[1] }), $m[3])) } else { 0 }); $cmbFg.SelectedIndex = [math]::Max(0, [array]::IndexOf(@($script:fgDisp), $fgU))
		$cmbFg.Enabled = -not (Test-AlForced)  # Ligero / Equilibrado: frame generation fija (como en Demon's Souls)
		Set-CalPreset ([int]$m[10])
		foreach ($pair in $script:qCombos) { if (@('cal_dof', 'cal_ca', 'cal_blur') -contains $pair[1][0]) { $pair[0].SelectedIndex = 0 } }  # profundidad de campo, aberracion cromatica y desenfoque de movimiento siempre quitados  # (bbl2) ligero Bajo, equilibrado Normal, ultra Alto, extremo Ultra
	} finally { $script:applying = $false }
	Write-Log "Modo aplicado: $($m[0])"
}
$cmbMode.Add_SelectionChangeCommitted({ Apply-Mode; if ((Get-Mode)[0] -ne 'personalizado') { foreach ($k in @('intres', 'upscaler', 'sharp')) { Mark-Changed $k -Auto } } })  # solo al elegirlo a mano
$cmbMode.Add_SelectedIndexChanged({ Update-ModeInfo; $cmbFg.Enabled = ($script:modes[[math]::Max(0, $cmbMode.SelectedIndex)][0] -eq 'personalizado') })  # frame generation fija solo en Ligero / Equilibrado
$toCustom = { if (-not $script:applying -and $cmbMode.SelectedIndex -ne ($script:modes.Count - 1)) { $cmbMode.SelectedIndex = $script:modes.Count - 1 } }
# (modos) En Ligero / Equilibrado solo la frame generation queda fija; tocar lo demas no cambia el modo.
# Marcas de cambio: solo acciones del usuario (no los cambios hechos por el programa).
$cmbWin.Add_SelectionChangeCommitted({ Mark-Changed 'winres' }); $cmbInt.Add_SelectionChangeCommitted({ Mark-Changed 'intres' })
$cmbFs.Add_SelectionChangeCommitted({ Mark-Changed 'fsmode' }); $cmbPres.Add_SelectionChangeCommitted({ Mark-Changed 'present' })
$cmbUp.Add_SelectionChangeCommitted({ Mark-Changed 'upscaler' }); $cmbSharp.Add_SelectionChangeCommitted({ Mark-Changed 'sharp' })
$chkHdr.Add_Click({ Mark-Changed 'hdr' }); $chkMotion.Add_Click({ Mark-Changed 'motion' }); $chkBg.Add_Click({ Mark-Changed 'bginput' })
foreach ($pair in $script:qCombos) { $pair[0].Add_SelectionChangeCommitted({ if ($cmbCal.SelectedIndex -ne 4) { $cmbCal.SelectedIndex = 4 } }); $pair[0].Add_SelectedIndexChanged({ if (-not $script:applying) { Update-CalStatus } }) }
$cmbCal.Add_SelectionChangeCommitted({ if ($cmbCal.SelectedIndex -le 3 -or $cmbCal.SelectedIndex -eq 5) { $script:applying = $true; try { Set-CalPreset $cmbCal.SelectedIndex } finally { $script:applying = $false } } })
$cmbWin.Add_SelectedIndexChanged({ Update-CalStatus })  # la rejilla de luz sigue a la resolucion de salida

Fill-Lists; Update-Up; Update-ModeInfo; Update-LatBox; Update-VerLabel; Update-TrophyButton
Write-SharpLive
$panel.ResumeLayout($true); $form.ResumeLayout($true)
function Hook-Wheel($ctl) { $ctl.Add_MouseWheel({ param($s, $e) Add-WheelStep (-1.0 * [double]$e.Delta * 0.6) }); foreach ($ch in $ctl.Controls) { Hook-Wheel $ch } }
Hook-Wheel $cont; Hook-Wheel $view; Hook-Wheel $btnPlay; $panel.Add_MouseWheel({ param($s, $e) Add-WheelStep (-1.0 * [double]$e.Delta * 0.6) }); $sbar.Add_MouseWheel({ param($s, $e) Add-WheelStep (-1.0 * [double]$e.Delta * 0.6) })
# (launcher112) Opciones: misma rueda suave (72 px por muesca) en todo el contenido.
function Hook-SetWheel($ctl) { $ctl.Add_MouseWheel({ param($s, $e) Add-SetWheel (-1.0 * [double]$e.Delta * 0.6) }); foreach ($ch in $ctl.Controls) { Hook-SetWheel $ch } }
Hook-SetWheel $scrollC; $scroll.Add_MouseWheel({ param($s, $e) Add-SetWheel (-1.0 * [double]$e.Delta * 0.6) }); $sbarS.Add_MouseWheel({ param($s, $e) Add-SetWheel (-1.0 * [double]$e.Delta * 0.6) })
Set-SetScrollY 0
# ---- (launcher113) Distribucion: nada tapa el titulo "Bloodborne PC" (rectangulo calculado con la misma escala y
# recorte del fondo). Panel de ajustes: tarjeta semitransparente redondeada a la derecha, bajo la franja del titulo si
# la comparte; Trofeos abajo, junto al panel (entre el cazador y el panel); la cruz arriba a la derecha.
# (launcher114) Trofeos tiene su propio hueco y nunca se cruza con el panel ni con sus botones (Actualizar / Opciones):
# 1) a la izquierda del panel, abajo, en el hueco libre a la izquierda o a la derecha del cazador (>= 260 px);
# 2) si no cabe, debajo del panel (el panel se acorta para dejarle sitio) y su lista se abre a la izquierda del panel.
# Al final se comprueba que boton, distintivo y lista no corten el panel ni el titulo; si lo hacen, se usa el caso 2.
$script:panelSlide = 0; $script:trSlide = 0; $script:panelHome = New-Object System.Drawing.Point(0, 0)
function Apply-Layout {
	$fw = $form.ClientSize.Width; $fh = $form.ClientSize.Height
	if ($fw -lt 200 -or $fh -lt 200) { return }
	$m = 16; $bH = 46; $bdH = 44; $gap = 8
	$tz = $script:titleR; $tz = New-Object System.Drawing.Rectangle(($tz.X - 20), ($tz.Y - 16), ($tz.Width + 40), ($tz.Height + 32))  # margen: cubre el zoom de entrada (<= 3 %)
	$xr = New-Object System.Drawing.Rectangle(($fw - 56), 8, 48, 40)
	if ($xr.IntersectsWith($tz)) { $xr = New-Object System.Drawing.Rectangle(8, 8, 48, 40) }
	$btnX.Bounds = $xr
	$px = [math]::Max(0, $fw - $panelW - $m); $py = $m  # (launcher115) panel de arriba (margen) abajo
	if ($px -lt $tz.Right -and ($px + $panelW) -gt $tz.Left) { $py = [math]::Max($py, $tz.Bottom + 8) }
	if ($xr.IntersectsWith((New-Object System.Drawing.Rectangle(($px - 4), $py, ($panelW + 8), ($fh - $py))))) { $xr = New-Object System.Drawing.Rectangle(($px - 56), 8, 48, 40); if ($xr.IntersectsWith($tz)) { $xr = New-Object System.Drawing.Rectangle(($px - 56), ($tz.Bottom + 8), 48, 40) }; $btnX.Bounds = $xr }
	# --- hueco de Trofeos
	$hr = $script:hunterR; $right = $px - 20
	$slots = @()
	if (-not $hr.IsEmpty) {
		$slots += , @($m, [math]::Min($right, $hr.Left - $gap))        # izquierda del cazador
		$slots += , @([math]::Max($m, $hr.Right + $gap), $right)       # entre el cazador y el panel
	}
	$slots += , @($m, $right)                                          # cualquier sitio a la izquierda del panel (ultimo recurso)
	$tw = 0; $tx = 0
	foreach ($sl in $slots) { $w = $sl[1] - $sl[0]; if ($w -ge 260) { $tw = [int][math]::Min($trophyW, $w); $tx = [int]($sl[1] - $tw); break } }
	$badgeH = $(if ($script:badgeOn) { $bdH + 10 } else { 0 })
	$below = ($tw -eq 0)
	$ty = $fh - $m - 8 - $bH
	if (-not $below) {
		# comprobacion: boton + distintivo no cortan el panel ni el titulo
		$tr = New-Object System.Drawing.Rectangle($tx, ($ty - $badgeH), $tw, ($bH + $badgeH))
		$prc = New-Object System.Drawing.Rectangle(($px - 4), $py, ($panelW + 8), ($fh - $py))
		if ($tr.IntersectsWith($prc) -or $tr.IntersectsWith($tz)) { $below = $true }
	}
	# (launcher116) Trofeos siempre abajo a la izquierda, fuera del panel: el panel va de arriba abajo.
	$below = $false; $tx = [int]$m; $tw = [int][math]::Max(200, [math]::Min($trophyW, $px - 3 * $m - $tkeyW - $gap)); $ty = $fh - $m - 8 - $bH  # (launcher129) deja sitio a Clave de trofeos
	$ph = [math]::Max(160, $fh - $py - $m)
	if ($below) {
		$tw = [int][math]::Min($trophyW, $panelW); $tx = [int]($px + $panelW - $tw)
		$ty = $fh - $m - $bH
		$ph = [math]::Max(120, $fh - $py - $m - $bH - 12 - $badgeH)
	}
	$script:trBelow = $below
	$script:panelHome = New-Object System.Drawing.Point($px, $py)
	$panel.SetBounds(($px + $script:panelSlide), $py, $panelW, $ph)
	try { $rp = New-RoundPath 0 0 $panelW $ph 18; $old = $panel.Region; $panel.Region = New-Object System.Drawing.Region($rp); $rp.Dispose(); if ($old) { $old.Dispose() } } catch { }
	Set-ScrollY $script:scrollY
	# --- boton, distintivo y lista
	$script:trW = $tw; $script:trTop = $ty
	$btnTrophy.SetBounds($tx, ($ty + $script:trSlide), $tw, $bH)
	$btnTKey.SetBounds(($tx + $tw + $gap), ($ty + $script:trSlide), $tkeyW, $bH)
	$lbBadge.SetBounds($tx, ($ty - $bdH - 10 + $script:trSlide), $tw, $bdH)
	if ($below) {
		# la lista se abre a la izquierda del panel (alineada abajo con el boton); si no cabe, encima del boton
		$lx = $px - 12 - $tw; $lb = $ty + $bH
		if ($lx -lt $m) { $lx = $tx; $lb = $ty - 6 - $badgeH }
	} else { $lx = $tx; $lb = $ty - 6 - $badgeH }
	$lh = [math]::Min($trophyListH, $lb - $m)
	if ($lx -lt $tz.Right -and ($lx + $tw) -gt $tz.Left -and ($lb - $lh) -lt ($tz.Bottom + 8)) { $lh = $lb - ($tz.Bottom + 8) }
	$lh = [math]::Max(120, $lh)
	$script:trListX = $lx; $script:trListBot = $lb; $script:trListH = $lh
	$lbTr.Width = $tw - 20; $pbTr.Width = $tw - 20; $lvTr.Width = $tw - 20; $lvTr.Height = [math]::Max(40, $lh - 60)
	try { $lvTr.Columns[1].Width = [math]::Max(60, $tw - 20 - 90 - 22) } catch { }
	if (-not $script:trAnim) { if ($gTrophy.Visible) { Set-TrophyH $lh } else { $gTrophy.SetBounds($lx, ($lb - $lh), $tw, $lh) } }
	# orden Z: los controles de abajo por encima del fondo; la lista abierta sobre todo
	$btnTrophy.Visible = $true; $btnTrophy.BringToFront(); $btnTKey.BringToFront(); $lbBadge.BringToFront(); if ($gTrophy.Visible) { $gTrophy.BringToFront(); $btnTrophy.BringToFront() }
	$lblVer.Location = New-Object System.Drawing.Point(24, ($fh - 20)); $lblVer.Width = [math]::Max(120, [math]::Min(600, $(if ($below) { $px } else { $tx }) - 32))
	# (launcher134) franja bajo Trofeos / Clave de trofeos: solo fondo (se repinta para borrar cualquier resto)
	try { $form.Invalidate((New-Object System.Drawing.Rectangle(0, [int]($ty + $bH - 6), [int][math]::Max(1, $px), [int][math]::Max(1, $fh - $ty - $bH + 6))), $false) } catch { }
	# --- efectos: solo a la izquierda del panel, fuera del titulo
	if ($script:embers) { $script:embers.Reset((New-Object System.Drawing.Rectangle(0, [int]($fh * 0.22), [math]::Max(1, $px - 12), [int]($fh * 0.78))), $tz) }
	if ($script:fog) { $script:fog.Reset((New-Object System.Drawing.Rectangle(0, [int]($fh * 0.30), [math]::Max(1, $px - 6), ($fh - [int]($fh * 0.30)))), $tz, $script:hunterR) }  # (launcher119) bancos anclados al cazador
	$script:layoutReady = $true
}
# ---- (launcher113) Animaciones de entrada: fundido de la ventana; el fondo entra con un zoom lento (3 % -> 0, centrado
# en el titulo); el panel entra deslizando desde la derecha y Trofeos desde abajo, escalonados (ease-out); el filete crece.
# Luego, brasas sutiles sobre el fondo (~30 fps, solo se repinta cada brasa; en pausa si la ventana esta minimizada).
# $reduceMotion: todo quieto.
if (-not $reduceMotion) {
	if ($fxEmbers -gt 0) { $script:embers = New-Object BbEmbers ([int]$fxEmbers) }
	if ($fxFog -gt 0) { try { $script:fog = New-Object BbFog ([int]$fxFog) } catch { $script:fog = $null; Write-Log "Niebla: $_" } }
}
Apply-Layout
$script:lastSize = $form.ClientSize
$form.Add_Resize({
	if ($form.WindowState -eq 'Minimized') { return }
	if ($form.ClientSize -eq $script:lastSize) { return }
	$script:lastSize = $form.ClientSize
	Build-Bg $form.ClientSize.Width $form.ClientSize.Height; Apply-Layout; $form.Invalidate($true)
})
function Ease([double]$t0, [double]$dur, [double]$now) { $t = [math]::Max(0.0, [math]::Min(1.0, ($now - $t0) / $dur)); return (1 - [math]::Pow(1 - $t, 3)) }
function Ease5([double]$t0, [double]$dur, [double]$now) { $t = [math]::Max(0.0, [math]::Min(1.0, ($now - $t0) / $dur)); return (1 - [math]::Pow(1 - $t, 5)) }
function EaseIO([double]$t0, [double]$dur, [double]$now) { $t = [math]::Max(0.0, [math]::Min(1.0, ($now - $t0) / $dur)); return ($t * $t * $t * ($t * ($t * 6 - 15) + 10)) }  # smootherstep
$script:slideDist = 70
$script:fadeT0 = 0
$fadeT = New-Object System.Windows.Forms.Timer; $fadeT.Interval = 15
$fadeT.Add_Tick({
	$el = [double]([Environment]::TickCount - $script:fadeT0)
	# (launcher119) fundido desde negro mas largo con curva suave en los dos extremos; el panel y Trofeos entran poco
	# trecho (menos repintado por fotograma = sin tirones) con ease-out quintico.
	$form.Opacity = [math]::Max(0.01, [math]::Min(1.0, (EaseIO 0 750 $el)))
	$ep = Ease5 200 800 $el; $script:panelSlide = [int][math]::Round($script:slideDist * (1 - $ep))
	$script:trSlide = 0  # (launcher134) Trofeos y Clave de trofeos quietos: al subir desde abajo dejaban restos de su texto debajo
	$panel.Left = $script:panelHome.X + $script:panelSlide
	$btnTrophy.Top = $script:trTop + $script:trSlide; $btnTKey.Top = $btnTrophy.Top; $lbBadge.Top = $btnTrophy.Top - 54
	$filete.Width = [math]::Max(1, [int]($gwTop * (Ease 650 500 $el)))
	if ($el -ge 1150) { $form.Opacity = 1; $script:panelSlide = 0; $script:trSlide = 0; $filete.Width = $gwTop; Apply-Layout; $fadeT.Stop(); $form.Invalidate($true) }
})
$script:fxT0 = 0; $script:fxLast = 0
$fxT = New-Object System.Windows.Forms.Timer; $fxT.Interval = 25  # (launcher119) ~40 fps
$fxT.Add_Tick({
	$now = [Environment]::TickCount; $dt = [math]::Max(1, [math]::Min(100, $now - $script:fxLast)); $script:fxLast = $now
	if ($form.WindowState -eq 'Minimized' -or -not $form.Visible) { return }
	if ($script:bgZoom -gt 1.0) {
		$z = 1.0 + 0.03 * (1 - (Ease 0 3200 ($now - $script:fxT0)))
		if ($z -le 1.0005) { $script:bgZoom = 1.0; $form.Invalidate($true) } else { $script:bgZoom = $z; $form.Invalidate($false); $panel.Invalidate($true) }
	}
	$fs = [float]($dt / 1000.0)
	if ($script:fog) {
		# (launcher119) la niebla aparece poco a poco (2.6 s, desde 0.4 s) y se mueve tambien durante el zoom de entrada
		if ($script:fog.Fade -lt 1) { $script:fog.Fade = [float](EaseIO 400 2600 ($now - $script:fxT0)) }
		$script:fog.Step($fs)
		if ($script:bgZoom -le 1.0) {
			# niebla: solo la banda que ocupa (izquierda del panel, sin el titulo); los hijos de esa zona se repintan con ella
			$dr = $script:fog.Dirty()
			if (-not $dr.IsEmpty) { $rg = New-Object System.Drawing.Region($dr); $rg.Exclude($script:fog.Avoid); $form.Invalidate($rg, $true); $rg.Dispose() }
		}
	}
	if ($script:embers) { $script:embers.Step($fs, $form, $true) }
})
$form.Add_FormClosing({ $fxT.Stop(); $fadeT.Stop(); $trT.Stop() })
# (trofeos) Clave de trofeos: si falta en <user>\keys.json, se pide una vez (no se incluye: es una clave de Sony).
function Test-TrophyKey {
	try { $ud = Get-UserDir $script:S.emu } catch { return $true }
	if (-not $ud) { return $true }
	$kf = Join-Path $ud 'keys.json'
	if (-not (Test-Path -LiteralPath $kf)) { return $false }
	try { $j = Get-Content -LiteralPath $kf -Raw | ConvertFrom-Json; return ("$($j.TrophyKeySet.ReleaseTrophyKey)" -match '^[0-9A-Fa-f]{32}$') } catch { return $false }
}
function Get-TrophyKeyValue {
	try { $ud = Get-UserDir $script:S.emu } catch { return '' }
	if (-not $ud) { return '' }
	$kf = Join-Path $ud 'keys.json'
	if (-not (Test-Path -LiteralPath $kf)) { return '' }
	try { $j = Get-Content -LiteralPath $kf -Raw | ConvertFrom-Json; $v = "$($j.TrophyKeySet.ReleaseTrophyKey)"; if ($v -match '^[0-9A-Fa-f]{32}$') { return $v } } catch { }
	return ''
}
function Show-TrophyKeyPrompt {
	# (launcher129) -Force (boton Clave de trofeos): se abre siempre, aunque haya clave o trophykey_skip=1.
	param([switch]$Force)
	if (-not $Force) {
		if ("$($script:S.trophykey_skip)" -eq '1') { return }
		if (Test-TrophyKey) { return }
	}
	$curKey = $(if ($Force) { Get-TrophyKeyValue } else { '' })
	$d = New-Object System.Windows.Forms.Form
	$d.Text = (T 'Clave de trofeos' 'Trophy key'); $d.FormBorderStyle = 'FixedDialog'; $d.StartPosition = 'CenterParent'
	$d.MaximizeBox = $false; $d.MinimizeBox = $false; $d.ClientSize = New-Object System.Drawing.Size(520, 230)
	$d.BackColor = [System.Drawing.Color]::FromArgb(255, 18, 16, 20); $d.ForeColor = [System.Drawing.Color]::White
	$l = New-Object System.Windows.Forms.Label; $l.Location = New-Object System.Drawing.Point(16, 14); $l.Size = New-Object System.Drawing.Size(488, 96)
	$l.Text = (T "Para ver los trofeos, el emulador necesita la clave de trofeos de PS4 (ReleaseTrophyKey, 32 caracteres hexadecimales).`n`nNo se incluye en el port porque es una clave de Sony: consíguela por tu cuenta y pégala aquí. Sin ella el juego funciona igual, pero sin trofeos." "To show trophies the emulator needs the PS4 trophy key (ReleaseTrophyKey, 32 hex characters).`n`nIt is not included in the port because it is a Sony key: get it yourself and paste it here. Without it the game runs the same, but without trophies.")
	$tb = New-Object System.Windows.Forms.TextBox; $tb.Location = New-Object System.Drawing.Point(16, 118); $tb.Size = New-Object System.Drawing.Size(488, 26)
	$tb.Font = New-Object System.Drawing.Font('Consolas', 11); $tb.MaxLength = 64
	$st = New-Object System.Windows.Forms.Label; $st.Location = New-Object System.Drawing.Point(16, 150); $st.Size = New-Object System.Drawing.Size(488, 20); $st.ForeColor = [System.Drawing.Color]::FromArgb(255, 235, 120, 100)
	if ($curKey) {
		$masked = $curKey.Substring(0, 4).ToUpper() + ('*' * 24) + $curKey.Substring(28, 4).ToUpper()
		$st.ForeColor = [System.Drawing.Color]::FromArgb(255, 226, 196, 120)
		$st.Text = (T "Ya hay una clave guardada: $masked. Pega otra para reemplazarla." "A key is already saved: $masked. Paste another one to replace it.")
	}
	$ok = New-Object System.Windows.Forms.Button; $ok.Text = (T 'Guardar' 'Save'); $ok.Location = New-Object System.Drawing.Point(16, 180); $ok.Size = New-Object System.Drawing.Size(120, 32)
	$later = New-Object System.Windows.Forms.Button; $later.Text = (T 'Más tarde' 'Later'); $later.Location = New-Object System.Drawing.Point(146, 180); $later.Size = New-Object System.Drawing.Size(120, 32)
	$never = New-Object System.Windows.Forms.Button; $never.Text = (T 'No volver a preguntar' 'Don''t ask again'); $never.Location = New-Object System.Drawing.Point(276, 180); $never.Size = New-Object System.Drawing.Size(228, 32)
	foreach ($b in @($ok, $later, $never)) { $b.FlatStyle = 'Flat'; $b.BackColor = [System.Drawing.Color]::FromArgb(255, 40, 34, 40); $b.ForeColor = [System.Drawing.Color]::White }
	$ok.Add_Click({
		$k = ($tb.Text -replace '\s', '')
		if ($k -notmatch '^[0-9A-Fa-f]{32}$') { $st.ForeColor = [System.Drawing.Color]::FromArgb(255, 235, 120, 100); $st.Text = (T 'La clave debe tener 32 caracteres hexadecimales (0-9, A-F).' 'The key must be 32 hex characters (0-9, A-F).'); return }
		try {
			$ud = Get-UserDir $script:S.emu; New-Item -ItemType Directory -Force -Path $ud | Out-Null
			$json = "{`n    `"TrophyKeySet`": {`n        `"ReleaseTrophyKey`": `"" + $k.ToUpper() + "`"`n    }`n}`n"
			[IO.File]::WriteAllText((Join-Path $ud 'keys.json'), $json, (New-Object System.Text.UTF8Encoding $false))
			Write-Log 'Clave de trofeos guardada en keys.json'; $d.Close()
		} catch { $st.Text = "$_" }
	})
	if ($Force) { $later.Text = (T 'Cerrar' 'Close') }
	$later.Add_Click({ $d.Close() })
	$never.Add_Click({ Set-Setting 'trophykey_skip' '1'; $d.Close() })
	$d.Controls.AddRange(@($l, $tb, $st, $ok, $later, $never)); $d.AcceptButton = $ok
	[void]$d.ShowDialog($form)
}
$form.Add_Shown({ try { Fit-AllGroups } catch { } })  # (launcher129) alturas con la letra real ya creada
$form.Add_Shown({ try { Show-TrophyKeyPrompt } catch { Write-Log "Clave de trofeos: $_" } })
$form.Add_Shown({
	$script:shownOnce = $true
	if ($reduceMotion) { $form.Opacity = 1; $form.Refresh(); return }
	$script:bgZoom = 1.03; $script:panelSlide = $script:slideDist; $script:trSlide = 0; if ($script:fog) { $script:fog.Fade = 0 }; Apply-Layout
	$form.Refresh(); $filete.Width = 1; $script:fadeT0 = [Environment]::TickCount; $script:fxT0 = $script:fadeT0; $script:fxLast = $script:fadeT0
	$form.Opacity = 0.01; $fadeT.Start(); $fxT.Start()
})
[void]$form.ShowDialog()
Write-Log 'Lanzador cerrado'
