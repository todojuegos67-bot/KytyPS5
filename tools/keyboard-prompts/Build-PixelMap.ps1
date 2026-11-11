# Generate the fixed 128x128 RGBA, 8-mip CTXR pixel-address map with Aqua Toolset.
# No original pixel data is retained: all mip buffers are replaced by markers or zeroes.
param([Parameter(Mandatory)][string]$AquaToolsetDirectory,
      [Parameter(Mandatory)][string]$OriginalIcon,
      [Parameter(Mandatory)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
foreach ($name in @('Reloaded.Memory','DrSwizzler','AquaModelLibrary.Helpers','AquaModelLibrary.Data','ZstdNet','ICSharpCode.SharpZipLib')) {
    [Reflection.Assembly]::LoadFrom((Join-Path $AquaToolsetDirectory "$name.dll")) | Out-Null
}
$tex=[AquaModelLibrary.Data.BluePoint.CTXR.CTXR]::new([IO.File]::ReadAllBytes($OriginalIcon))
if ($tex.textureFormat -ne 0 -or $tex.internalMipCount -ne 8 -or $tex.externalMipCount -ne 0 -or
    [AquaModelLibrary.Data.BluePoint.CTXR.CTXR]::GetDesResolutionComponent($tex.WidthBaseByte,$tex.WidthMultiplierByte,0xC0) -ne 128 -or
    [AquaModelLibrary.Data.BluePoint.CTXR.CTXR]::GetDesResolutionComponent($tex.HeightBaseByte,$tex.HeightMultiplierByte,0x80) -ne 128) { throw 'Unsupported template' }
$id=1
for ($m=0;$m -lt 8;$m++) {
    $w=[Math]::Max(1,128 -shr $m); $pitch=[Math]::Max(8,$w)
    $pixels=[byte[]]::new($pitch*$pitch*4)
    for($y=0;$y -lt $w;$y++){for($x=0;$x -lt $w;$x++){
        [Array]::Copy([BitConverter]::GetBytes([int]$id),0,$pixels,($y*$pitch+$x)*4,4);$id++
    }}
    $tex.mipMapsList[0][$m]=$pixels
}
$marked=$tex.GetBytes();$offsets=[int[]]::new($id-1);[Array]::Fill($offsets,-1)
# Fixed template's pixel storage is 128 KiB; its remaining bytes describe the texture.
if($marked.Length -ne 131149){throw 'Unexpected CTXR size'}
for($at=65;$at -lt 131137;$at+=4){
    $value=[BitConverter]::ToInt32($marked,$at)
    if($value -ge 1 -and $value -lt $id){
        if($offsets[$value-1] -ne -1){throw 'Repeated marker'}
        $offsets[$value-1]=$at
    }
}
if($offsets.Contains(-1)){throw 'Missing pixel marker'}
[Array]::Clear($marked,65,131072)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
function Gzip($name,[byte[]]$bytes){
    $file=[IO.File]::Create((Join-Path $OutputDirectory $name))
    $zip=[IO.Compression.GZipStream]::new($file,[IO.Compression.CompressionMode]::Compress)
    try{$zip.Write($bytes,0,$bytes.Length)}finally{$zip.Dispose();$file.Dispose()}
}
$map=[byte[]]::new($offsets.Length*4);[Buffer]::BlockCopy($offsets,0,$map,0,$map.Length)
Gzip 'pixel-offsets.bin.gz' $map
Gzip 'blank.ctxr.gz' $marked
Write-Output "Generated $($offsets.Length) pixel offsets; blank template contains no original pixels."

