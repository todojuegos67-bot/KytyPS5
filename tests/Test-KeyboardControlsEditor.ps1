# Runs against a built launcher with temporary settings; never touches a game or user settings.
param([string]$Launcher=(Join-Path (Split-Path $PSScriptRoot -Parent) '_Build/launcher/KytyLauncher.exe'))
$ErrorActionPreference='Stop'
Add-Type -AssemblyName PresentationFramework,PresentationCore,WindowsBase
$assembly=[Reflection.Assembly]::LoadFrom($Launcher)
$app=[Activator]::CreateInstance($assembly.GetType('KytyLauncher.App'))
$app.InitializeComponent();$app.ShutdownMode='OnExplicitShutdown'
$root=Join-Path ([IO.Path]::GetTempPath()) ('kyty-editor-test-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
$path=Join-Path $root 'settings.json'
$settings=[Activator]::CreateInstance($assembly.GetType('KytyLauncher.Settings'),[object[]]@([string]$path))
$flags=[Reflection.BindingFlags]'Instance,NonPublic'
function Check($condition,$message){if(!$condition){throw $message}}
function Field($window,$name){$window.GetType().GetField($name,$flags).GetValue($window)}
function InvokeHidden($window,$name,$arguments){$window.GetType().GetMethod($name,$flags).Invoke($window,$arguments)}
function Click($button){$button.RaiseEvent([Windows.RoutedEventArgs]::new([Windows.Controls.Button]::ClickEvent))}
function Find-Button($node,$text){
    if($node -is [Windows.Controls.Button] -and $node.Content -eq $text){return $node}
    if($node -is [Windows.DependencyObject]){
        foreach($child in [Windows.LogicalTreeHelper]::GetChildren($node)){
            $found=Find-Button $child $text
            if($found){return $found}
        }
    }
}
try {
    $window=[Activator]::CreateInstance($assembly.GetType('KytyLauncher.KeyboardControlsWindow'),[object[]]@($settings))
    $script:failure=$null
    $window.add_ContentRendered({
        try {
            $buttons=Field $window 'buttons'
            (Field $window 'layout').Assign('Cross','Z') | Out-Null
            (Field $window 'sensitivity').Text='4.0'
            Click (Find-Button $window 'Reset defaults')
            Check ((Field $window 'layout').Bindings['Cross'] -eq 'E' -and (Field $window 'sensitivity').Text -eq '') 'Reset did not restore the draft defaults'
            Click $buttons['Cross']
            Click (Find-Button $window 'Stop capture')
            Check ($null -eq (Field $window 'capturing')) 'Stop capture failed' 
            Click $buttons['Cross']
            Check ((Field $window 'capturing') -eq 'Cross') 'Click did not begin capture'
            $source=[Windows.PresentationSource]::FromVisual($window)
            $key=[Windows.Input.KeyEventArgs]::new([Windows.Input.Keyboard]::PrimaryDevice,$source,0,[Windows.Input.Key]::Q)
            $key.RoutedEvent=[Windows.Input.Keyboard]::PreviewKeyDownEvent
            $window.RaiseEvent($key)
            Check ((Field $window 'layout').Bindings['Cross'] -eq 'Q') 'Key event did not bind Q'
            Click $buttons['Square']
            $key=[Windows.Input.KeyEventArgs]::new([Windows.Input.Keyboard]::PrimaryDevice,$source,0,[Windows.Input.Key]::Q)
            $key.RoutedEvent=[Windows.Input.Keyboard]::PreviewKeyDownEvent;$window.RaiseEvent($key)
            Check ((Field $window 'layout').Bindings['Square'] -eq 'Q' -and (Field $window 'layout').Bindings['Cross'] -eq 'R') 'Used key did not swap actions'
            Click $buttons['R3']
            $mouse=[Windows.Input.MouseButtonEventArgs]::new([Windows.Input.Mouse]::PrimaryDevice,0,[Windows.Input.MouseButton]::XButton1)
            $mouse.RoutedEvent=[Windows.Input.Mouse]::PreviewMouseDownEvent;$window.RaiseEvent($mouse)
            Check ((Field $window 'layout').Bindings['R3'] -eq 'Mouse:X1') 'Mouse capture failed'
            Click $buttons['Circle']
            $key=[Windows.Input.KeyEventArgs]::new([Windows.Input.Keyboard]::PrimaryDevice,$source,0,[Windows.Input.Key]::F7)
            $key.RoutedEvent=[Windows.Input.Keyboard]::PreviewKeyDownEvent;$window.RaiseEvent($key)
            Check ((Field $window 'layout').Bindings['Circle'] -eq 'Space') 'Reserved F7 was accepted'
            (Field $window 'sensitivity').Text='2.3'
            InvokeHidden $window 'Save' @()
        } catch {$script:failure=$_;$window.Close()}
    })
    $window.ShowDialog() | Out-Null
    if($failure){throw $failure}
    $loaded=Get-Content $path -Raw | ConvertFrom-Json
    Check ($loaded.keyboard_binding_Cross -eq 'R' -and $loaded.keyboard_binding_Square -eq 'Q' -and $loaded.keyboard_binding_R3 -eq 'Mouse:X1' -and $loaded.keyboard_mouse_sensitivity -eq '2.3') 'Save did not persist the edited layout'
    $before=(Get-FileHash $path).Hash
    $cancel=[Activator]::CreateInstance($assembly.GetType('KytyLauncher.KeyboardControlsWindow'),[object[]]@($settings))
    (Field $cancel 'layout').Assign('Cross','Z') | Out-Null
    $cancel.Close()
    Check ((Get-FileHash $path).Hash -eq $before -and $settings.Str('keyboard_binding_Cross') -eq 'R') 'Cancel changed saved or active controls'
    Write-Output 'PASS: editor click/key/mouse events, duplicate swap, reserved keys, sensitivity/save/reload, cancel isolation'
} finally {
    $app.Shutdown()
    $parent=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
    Check ((Split-Path ([IO.Path]::GetFullPath($root)) -Parent) -eq $parent -and (Split-Path $root -Leaf).StartsWith('kyty-editor-test-')) 'Unsafe cleanup path'
    Remove-Item -LiteralPath $root -Recurse -Force
}

