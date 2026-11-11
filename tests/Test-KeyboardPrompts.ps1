param([Parameter(Mandatory)][string]$OriginalIconDirectory)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$compiler = & $vswhere -latest -products '*' -requires Microsoft.Component.MSBuild -find 'MSBuild/Current/Bin/Roslyn/csc.exe'
if (!$compiler) { throw 'Visual Studio C# compiler not found.' }
$out = Join-Path $root '_Build/keyboard-prompts-tests'
New-Item -ItemType Directory -Force $out | Out-Null
$exe = Join-Path $out 'KeyboardPromptsTests.exe'
$arguments = @('/nologo','/target:exe','/r:System.Drawing.dll','/r:System.Web.Extensions.dll',"/out:$exe",(Join-Path $root 'launcher-windows/KeyboardPrompts.cs'),(Join-Path $root 'launcher-windows/KeyboardLayout.cs'),(Join-Path $root 'launcher-windows/PromptTexture.cs'),(Join-Path $root 'launcher-windows/Settings.cs'),(Join-Path $PSScriptRoot 'KeyboardPromptsTests.cs'))
foreach ($file in Get-ChildItem (Join-Path $root 'launcher-windows/KeyboardPrompts') -File) {
    $arguments += "/resource:$($file.FullName),KytyLauncher.KeyboardPrompts.$($file.Name)"
}
& $compiler $arguments
if ($LASTEXITCODE -ne 0) { throw 'Test compilation failed.' }
& $exe $OriginalIconDirectory
if ($LASTEXITCODE -ne 0) { throw 'Keyboard prompt tests failed.' }
