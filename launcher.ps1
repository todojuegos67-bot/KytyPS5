param([switch]$DryRun)
# A small settings window in front of run-windows.ps1 and precompile-windows.ps1 (launcher.cmd opens it). The
# settings and the console language names follow the KytyPS5 launcher (src/launcher, by the KytyPS5 developers);
# they are kept in launcher-settings.json next to this script. -DryRun prints the commands without a window.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$settingsPath = Join-Path $root 'launcher-settings.json'
$languages = 'Japanese', 'English (United States)', 'French (France)', 'Spanish (Spain)', 'German', 'Italian', 'Dutch',
	'Portuguese (Portugal)', 'Russian', 'Korean', 'Chinese (Traditional)', 'Chinese (Simplified)', 'Finnish', 'Swedish',
	'Danish', 'Norwegian', 'Polish', 'Portuguese (Brazil)', 'English (United Kingdom)', 'Turkish', 'Spanish (Latin America)',
	'Arabic', 'French (Canada)', 'Czech', 'Hungarian', 'Greek', 'Romanian', 'Thai', 'Vietnamese', 'Indonesian'
$resolutions = '1280x720', '1920x1080', '2560x1440', '3840x2160', '2560x1080', '3440x1440', '3840x1600', '5120x2160'

$settings = [ordered]@{ game = ''; resolution = '2560x1440'; fullscreen = $false; aspect = $true; language = 1; redzone = $true }
if (Test-Path $settingsPath) {
	$saved = Get-Content $settingsPath -Raw | ConvertFrom-Json
	foreach ($property in $saved.PSObject.Properties) { if ($settings.Contains($property.Name)) { $settings[$property.Name] = $property.Value } }
}
if (!$settings.game -and (Test-Path "$root\game-path.txt")) { $settings.game = (Get-Content "$root\game-path.txt" -TotalCount 1).Trim() }

# The play / precompile command lines, run by PowerShell itself: no cmd.exe (an overlay or monitoring program that
# injects a DLL into every process can keep cmd.exe from starting, error 0xc0000142, while PowerShell starts fine).
function Get-PlayArguments {
	$size = $settings.resolution -split 'x'
	$script = @("'$root\run-windows.ps1'", '-Prompt', '-Follow', '-Width', $size[0], '-Height', $size[1], '-Language', $settings.language)
	if ($settings.fullscreen) { $script += '-Fullscreen'; if ($settings.aspect) { $script += '-AspectFit' } }
	if (!$settings.redzone) { $script += '-NoRedZone' }
	if ($settings.game) { $script += "-Game '$($settings.game.Replace("'", "''"))'" }
	# The console stays for the live log; after a crash it waits for a key.
	$command = '& ' + ($script -join ' ') + '; if ($LASTEXITCODE -ne 0) { Read-Host ''Press Enter to close'' | Out-Null }'
	return @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $command)
}
function Get-PrecompileArguments {
	$script = @("'$root\precompile-windows.ps1'")
	if ($settings.game) { $script += "-Game '$($settings.game.Replace("'", "''"))'" }
	$command = '& ' + ($script -join ' ') + '; Read-Host ''Press Enter to close'' | Out-Null'
	return @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $command)
}
function Save-Settings { $settings | ConvertTo-Json | Set-Content $settingsPath -Encoding UTF8 }
if ($DryRun) {
	"play:       powershell " + ((Get-PlayArguments) -join ' ')
	"precompile: powershell " + ((Get-PrecompileArguments) -join ' ')
	return
}

Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type -Namespace KytyLauncher -Name Dpi -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
[KytyLauncher.Dpi]::SetProcessDPIAware() | Out-Null
[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form -Property @{
	Text = 'KytyPS5 launcher'; FormBorderStyle = 'FixedDialog'; MaximizeBox = $false; StartPosition = 'CenterScreen'
	AutoSize = $true; AutoSizeMode = 'GrowAndShrink'; Font = New-Object System.Drawing.Font('Segoe UI', 9); Padding = '12,12,12,8'
}
$grid = New-Object System.Windows.Forms.TableLayoutPanel -Property @{ ColumnCount = 2; AutoSize = $true; Dock = 'Fill' }
$form.Controls.Add($grid)
# A row: a label, then the controls side by side.
function Add-Row($label, [object[]]$controls) {
	$grid.Controls.Add((New-Object System.Windows.Forms.Label -Property @{ Text = $label; AutoSize = $true; Anchor = 'Left'; Margin = '0,6,8,6' }))
	$row = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ AutoSize = $true; WrapContents = $false; Margin = '0,0,0,0' }
	$row.Controls.AddRange($controls)
	$grid.Controls.Add($row)
}

$game = New-Object System.Windows.Forms.TextBox -Property @{ Text = $settings.game; Width = 440 }
$browse = New-Object System.Windows.Forms.Button -Property @{ Text = 'Browse...'; AutoSize = $true }
$browseZar = New-Object System.Windows.Forms.Button -Property @{ Text = '.zar...'; AutoSize = $true }
Add-Row 'Game folder' @($game, $browse, $browseZar)
$title = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray' }
Add-Row '' @($title)
# A game: its folder (eboot.bin and sce_sys) or the folder packed into a ZArchive (a .zar file).
function Read-GameParam([string]$game) {
	if (!$game) { return $null }
	if (Test-Path -LiteralPath "$game\sce_sys\param.json") { return Get-Content -LiteralPath "$game\sce_sys\param.json" -Raw -Encoding UTF8 | ConvertFrom-Json }
	$tool = @("$root\kyty_shader_precompile.exe", "$root\_Build\windows\kyty_shader_precompile.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
	if ($game -notmatch '\.zar$' -or !(Test-Path -LiteralPath $game -PathType Leaf) -or !$tool) { return $null }
	$copy = [IO.Path]::GetTempFileName()
	try {
		& $tool --game $game --param $copy 2>$null | Out-Null
		if ($LASTEXITCODE -eq 0) { return Get-Content -LiteralPath $copy -Raw -Encoding UTF8 | ConvertFrom-Json }
	} catch {
	} finally {
		Remove-Item -LiteralPath $copy -ErrorAction SilentlyContinue
	}
	return $null
}
function Update-Title {
	$title.Text = 'Choose the folder with eboot.bin and sce_sys (the game dump), or a .zar archive of it.'
	try {
		$json = Read-GameParam $game.Text.Trim()
		if ($json) {
			$title.Text = '{0} ({1} {2})' -f $json.localizedParameters.($json.localizedParameters.defaultLanguage).titleName,
				$json.titleId, $json.contentVersion
		}
	} catch {}
}
$browse.Add_Click({
	$dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{ Description = 'The game folder (eboot.bin, sce_sys)' }
	if ($game.Text -and (Test-Path -LiteralPath $game.Text -PathType Container)) { $dialog.SelectedPath = $game.Text }
	if ($dialog.ShowDialog($form) -eq 'OK') { $game.Text = $dialog.SelectedPath }
})
# The game folder packed into a ZArchive (zarchive.exe <folder> <file.zar>): read without extracting it.
$browseZar.Add_Click({
	$dialog = New-Object System.Windows.Forms.OpenFileDialog -Property @{ Title = 'The game folder packed into a ZArchive'; Filter = 'ZArchive (*.zar)|*.zar' }
	if ($game.Text -and (Test-Path -LiteralPath $game.Text -PathType Leaf)) { $dialog.InitialDirectory = Split-Path $game.Text }
	if ($dialog.ShowDialog($form) -eq 'OK') { $game.Text = $dialog.FileName }
})
$game.Add_TextChanged({ Update-Title })
Update-Title

$resolution = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 160 }
$resolution.Items.AddRange($resolutions)
$resolution.SelectedItem = if ($resolutions -contains $settings.resolution) { $settings.resolution } else { '2560x1440' }
Add-Row 'Resolution' @($resolution)
$fullscreen = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Borderless fullscreen (unchecked: window)'; AutoSize = $true; Checked = [bool]$settings.fullscreen }
$aspect = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Keep 16:9 (black bars)'; AutoSize = $true; Checked = [bool]$settings.aspect }
$aspect.Enabled = $fullscreen.Checked
$fullscreen.Add_CheckedChanged({ $aspect.Enabled = $fullscreen.Checked })
Add-Row '' @($fullscreen, $aspect)
$language = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 320 }
$language.Items.AddRange($languages)
$language.SelectedIndex = [Math]::Max(0, [Math]::Min($languages.Count - 1, [int]$settings.language))
Add-Row 'Console language' @($language)
$redzone = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Red-zone protection (recommended)'; AutoSize = $true; Checked = [bool]$settings.redzone }
Add-Row '' @($redzone)
$buttons = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ AutoSize = $true; Margin = '0,10,0,0' }
$play = New-Object System.Windows.Forms.Button -Property @{ Text = 'Play'; AutoSize = $true; Font = New-Object System.Drawing.Font('Segoe UI', 9, [System.Drawing.FontStyle]::Bold) }
$precompile = New-Object System.Windows.Forms.Button -Property @{ Text = 'Precompile shaders'; AutoSize = $true }
$logs = New-Object System.Windows.Forms.Button -Property @{ Text = 'Logs folder'; AutoSize = $true }
$readme = New-Object System.Windows.Forms.Button -Property @{ Text = 'README'; AutoSize = $true }
$buttons.Controls.AddRange(@($play, $precompile, $logs, $readme))
$grid.Controls.Add($buttons)
$grid.SetColumnSpan($buttons, 2)
$credit = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '0,8,0,0'
	Text = 'Settings follow the KytyPS5 launcher (src/launcher) by the KytyPS5 developers.' }
$grid.Controls.Add($credit)
$grid.SetColumnSpan($credit, 2)
$form.AcceptButton = $play

function Read-Form {
	$settings.game       = $game.Text.Trim()
	$settings.resolution = [string]$resolution.SelectedItem
	$settings.fullscreen = $fullscreen.Checked
	$settings.aspect     = $aspect.Checked
	$settings.language   = $language.SelectedIndex
	$settings.redzone    = $redzone.Checked
	Save-Settings
}
$play.Add_Click({
	Read-Form
	Start-Process powershell -ArgumentList (Get-PlayArguments) -WorkingDirectory $root
	$form.Close()
})
$precompile.Add_Click({
	Read-Form
	Start-Process powershell -ArgumentList (Get-PrecompileArguments) -WorkingDirectory $root
})
$logs.Add_Click({
	$folder = if (Test-Path "$root\_Build") { "$root\_Build\run-logs" } else { "$root\logs" }
	New-Item -ItemType Directory -Force $folder | Out-Null
	Start-Process explorer.exe $folder
})
$readme.Add_Click({
	$file = @("$root\README.md", "$root\docs\PORTABLE-README.md") | Where-Object { Test-Path $_ } | Select-Object -First 1
	if ($file) { Start-Process notepad.exe "`"$file`"" }
})
$form.Add_FormClosing({ Read-Form })
[void]$form.ShowDialog()
