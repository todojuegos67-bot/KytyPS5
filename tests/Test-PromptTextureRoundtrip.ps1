param([string]$AquaToolsetDirectory, [switch]$Export)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$out=Join-Path $root '_Build/prompt-roundtrip'
if($Export){
    Add-Type -AssemblyName System.Drawing
    $assembly=[Reflection.Assembly]::LoadFrom((Join-Path $root '_Build/launcher/KytyLauncher.exe'))
    $type=$assembly.GetType('KytyLauncher.PromptTexture')
    New-Item -ItemType Directory -Force $out | Out-Null
    $labels=@('Q','CTRL','WHEEL'+"`n"+'UP','NUM 9','I A'+"`n"+'S D',([char]0x2191).ToString())
    for($i=0;$i -lt $labels.Count;$i++){
        $data=$type.GetMethod('Generate').Invoke($null,@($labels[$i]))
        [IO.File]::WriteAllBytes((Join-Path $out "$i.ctxr"),$data)
        $pixels=$type.GetMethod('Pixels').Invoke($null,@($labels[$i]))
        for($level=0;$level -lt 8;$level++){[IO.File]::WriteAllBytes((Join-Path $out "$i-$level.rgba"),$pixels[$level])}
    }
    exit
}
if(!$AquaToolsetDirectory){throw 'Supply AquaToolsetDirectory for the independent CTXR decoder'}
& powershell -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Export
if($LASTEXITCODE -ne 0){throw 'Texture export failed'}
foreach($name in @('Reloaded.Memory','DrSwizzler','AquaModelLibrary.Helpers','AquaModelLibrary.Data','ZstdNet','ICSharpCode.SharpZipLib')){
    [Reflection.Assembly]::LoadFrom((Join-Path $AquaToolsetDirectory "$name.dll")) | Out-Null
}
foreach($file in Get-ChildItem $out -Filter '*.ctxr'){
    $texture=[AquaModelLibrary.Data.BluePoint.CTXR.CTXR]::new([IO.File]::ReadAllBytes($file.FullName))
    for($level=0;$level -lt 8;$level++){
        $expected=[IO.File]::ReadAllBytes((Join-Path $out "$($file.BaseName)-$level.rgba"))
        if([Convert]::ToBase64String($texture.mipMapsList[0][$level]) -cne [Convert]::ToBase64String($expected)){throw "Mismatch: $($file.Name) mip $level"}
    }
}
Write-Output 'PASS: custom CTXR textures independently decoded with exact matches at every mip level'
