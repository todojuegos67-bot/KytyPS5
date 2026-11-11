param([string]$Launcher = (Join-Path (Split-Path $PSScriptRoot -Parent) '_Build/launcher/KytyLauncher.exe'), [string]$FixtureRoot = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$root = Join-Path ([IO.Path]::GetTempPath()) ('kyty-controls-test-' + [Guid]::NewGuid().ToString('N'))
function Check($condition, $message) { if (!$condition) { throw $message } }
if (!$FixtureRoot) {
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Launcher $Launcher -FixtureRoot $root
        if ($LASTEXITCODE -ne 0) { throw 'Control integration tests failed' }
    } finally {
        $parent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
        Check ((Split-Path ([IO.Path]::GetFullPath($root)) -Parent) -eq $parent -and (Split-Path $root -Leaf).StartsWith('kyty-controls-test-')) 'Unsafe cleanup path'
        if (Test-Path $root) { Remove-Item -LiteralPath $root -Recurse -Force }
    }
    exit
}
$root = $FixtureRoot
    New-Item -ItemType Directory -Path $root | Out-Null
    Copy-Item -LiteralPath $Launcher -Destination (Join-Path $root 'KytyLauncher.exe')
    Copy-Item -LiteralPath (Join-Path $repo 'run-windows.ps1') -Destination $root
    $game = Join-Path $root 'game'
    New-Item -ItemType Directory -Path $game | Out-Null
    [IO.File]::WriteAllText((Join-Path $game 'eboot.bin'), '')
    [IO.File]::WriteAllText((Join-Path $root 'kyty_emulator.exe'), '')
    $config = Join-Path $root 'launch.json'
    @{environment=@{}; command=@('--','kyty_emulator.exe','--keymap','Circle=L','--keymap','R3=Left Ctrl','--keymap','MouseSensitivity=2.0','--keymap','Up=Mouse:WheelUp')} | ConvertTo-Json -Depth 5 | Set-Content $config
    $before = (Get-FileHash $config).Hash
    $assembly = [Reflection.Assembly]::LoadFrom((Join-Path $root 'KytyLauncher.exe'))
    $context = [Activator]::CreateInstance($assembly.GetType('KytyLauncher.LauncherContext'), $true)
    $settings = $context.GetType().GetField('Settings').GetValue($context)
    $settings.Set('game', $game)
    $commands = [Activator]::CreateInstance($assembly.GetType('KytyLauncher.Commands'), @($context))
    foreach ($enabled in @($false,$true)) {
        $settings.Set('keyboard_prompts', $enabled)
        foreach ($direct in @($false,$true)) {
            $line = $commands.PlayCommand($null,$direct)
            Check ($line.Contains('-KeyboardMouseControls') -eq $enabled) 'GUI/direct Play did not follow the toggle'
        }
    }
    $settings.Set('extra_args', '-KeyboardMouseControls')
    Check (([regex]::Matches($commands.PlayCommand($null,$true), '-KeyboardMouseControls')).Count -eq 1) 'Duplicate control switch'
    $script = Join-Path $root 'run-windows.ps1'
    $off = & $script -Game $game -DryRun 6>&1 | Out-String -Width 10000
    $on = & $script -Game $game -KeyboardMouseControls -DryRun 6>&1 | Out-String -Width 10000
    foreach ($binding in @('Cross=E','Circle=Space','Square=R','Triangle=F','L1=Left Shift','L2=Left Ctrl','L3=X','R1=Mouse:Left','R2=Mouse:Right','R3=Mouse:Middle','Options=Escape','Up=Up','Down=Down','Left=Left','Right=Right','TouchPad=Tab','TouchPadRight=Backspace','LeftStickUp=W','LeftStickDown=S','LeftStickLeft=A','LeftStickRight=D')) {
        Check ($on.Contains($binding)) ('Missing binding: ' + $binding)
        Check (!$off.Contains($binding)) ('Disabled option added binding: ' + $binding)
    }
    Check ($on.IndexOf('Circle=Space') -gt $on.IndexOf('Circle=L')) 'Preset must override saved conflicts'
    Check ($on.Contains('MouseSensitivity=2.0') -and $on.Contains('Up=Mouse:WheelUp')) 'Lost sensitivity or wheel input'
    $settings.Set('keyboard_binding_Cross','Q')
    $settings.Set('keyboard_mouse_sensitivity','2.4')
    $line = $commands.PlayCommand($null,$true)
    Check ($line -match '-KeyboardMouseBindings ([A-Za-z0-9+/=]+)') 'Custom layout was not serialized'
    $encoded = $Matches[1]
    $custom = & $script -Game $game -KeyboardMouseControls -KeyboardMouseBindings $encoded -DryRun 6>&1 | Out-String -Width 10000
    Check ($custom.Contains('Cross=Q') -and !$custom.Contains('Cross=E')) 'Custom action did not reach emulator command'
    Check ($custom.Contains('MouseSensitivity=2.4')) 'Custom sensitivity missing'
    Check ($custom.Contains('Up=Mouse:WheelUp')) 'Existing wheel alias was lost'
    foreach ($inputName in @(';',"'",'\','Keypad +','F13')) {
        $settings.Set('keyboard_binding_Cross',$inputName)
        $line=$commands.PlayCommand($null,$true)
        Check ($line -match '-KeyboardMouseBindings ([A-Za-z0-9+/=]+)') 'Missing encoded layout'
        $special = & $script -Game $game -KeyboardMouseControls -KeyboardMouseBindings $Matches[1] -DryRun 6>&1 | Out-String -Width 10000
        Check ($special.Contains('Cross='+$inputName)) ('Special key failed: '+$inputName)
    }
    $settings.Set('keyboard_binding_Cross','R')
    $rejected = $false
    try { $commands.PlayCommand($null,$true) | Out-Null } catch { $rejected = $true }
    Check $rejected 'Duplicate bindings must stop launch before icons can disagree'
    $settings.Set('keyboard_prompts',$false)
    $settings.Set('extra_args','')
    Check (!$commands.PlayCommand($null,$true).Contains('-KeyboardMouseBindings')) 'Disabled option read or applied an invalid custom profile'
    Check ((Get-FileHash $config).Hash -eq $before) 'launch.json was changed'
    Write-Host 'PASS: GUI/direct commands, toggle off/on, all preset bindings, conflict precedence, sensitivity/wheel preservation, no config writes'
