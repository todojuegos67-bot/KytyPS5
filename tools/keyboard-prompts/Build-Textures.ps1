# SPDX-License-Identifier: GPL-2.0-or-later
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$IconDirectory,
    [Parameter(Mandatory)][string]$AquaToolsetDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $IconDirectory).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw 'Choose a new output directory; existing files are never overwritten.' }
foreach ($name in @('Reloaded.Memory','DrSwizzler','AquaModelLibrary.Helpers','AquaModelLibrary.Data','ZstdNet','ICSharpCode.SharpZipLib')) {
    [Reflection.Assembly]::LoadFrom((Join-Path $AquaToolsetDirectory "$name.dll")) | Out-Null
}
Add-Type -AssemblyName System.Drawing
$labels = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'labels.json') -Raw | ConvertFrom-Json -AsHashtable
# Validate every input before producing output. No game files are modified.
foreach ($name in $labels.Keys) {
    if ($name -notmatch '^ps4_[a-z0-9_]+$') { throw "Invalid icon name: $name" }
    foreach ($file in @((Join-Path $source "$name.ctxr"), (Join-Path $PSScriptRoot "icons/$name.png"))) {
        if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing input: $file" }
    }
}
New-Item -ItemType Directory -Path $output | Out-Null
foreach ($name in ($labels.Keys | Sort-Object)) {
    $tex = [AquaModelLibrary.Data.BluePoint.CTXR.CTXR]::new([IO.File]::ReadAllBytes((Join-Path $source "$name.ctxr")))
    $width = [AquaModelLibrary.Data.BluePoint.CTXR.CTXR]::GetDesResolutionComponent($tex.WidthBaseByte,$tex.WidthMultiplierByte,0xC0)
    $height = [AquaModelLibrary.Data.BluePoint.CTXR.CTXR]::GetDesResolutionComponent($tex.HeightBaseByte,$tex.HeightMultiplierByte,0x80)
    if ($width -ne 128 -or $height -ne 128 -or $tex.textureFormat -ne 0 -or $tex.externalMipCount -ne 0 -or $tex.internalMipCount -ne 8 -or $tex.mipMapsList.Count -ne 1) {
        throw "Unsupported texture layout: $name. Output is incomplete; do not install it."
    }
    $bitmap = [Drawing.Bitmap]::new((Join-Path $PSScriptRoot "icons/$name.png"))
    try {
        if ($bitmap.Width -ne $width -or $bitmap.Height -ne $height) { throw "Incorrect PNG dimensions: $name" }
        $expected = [Collections.Generic.List[byte[]]]::new()
        $w=$width; $h=$height
        for ($level=0; $level -lt $tex.internalMipCount; $level++) {
            $mip = [Drawing.Bitmap]::new($w,$h,[Drawing.Imaging.PixelFormat]::Format32bppArgb)
            try {
                $graphics = [Drawing.Graphics]::FromImage($mip)
                try {
                    $graphics.CompositingMode='SourceCopy'
                    $graphics.InterpolationMode='HighQualityBicubic'
                    $graphics.DrawImage($bitmap,0,0,$w,$h)
                } finally { $graphics.Dispose() }
                $pixels=[byte[]]::new($w*$h*4)
                for ($y=0; $y -lt $h; $y++) { for ($x=0; $x -lt $w; $x++) {
                    $c=$mip.GetPixel($x,$y); $i=($y*$w+$x)*4
                    $pixels[$i]=$c.R; $pixels[$i+1]=$c.G; $pixels[$i+2]=$c.B; $pixels[$i+3]=$c.A
                }}
                $expected.Add($pixels)
                # Aqua's PS5 swizzler expects at least an 8x8 tile. Pad small
                # mips by row; passing tightly packed 4x4/2x2 data loses pixels.
                if ($w -lt 8 -or $h -lt 8) {
                    $padded = [byte[]]::new(8*8*4)
                    for ($row=0; $row -lt $h; $row++) { [Array]::Copy($pixels,$row*$w*4,$padded,$row*8*4,$w*4) }
                    $tex.mipMapsList[0][$level]=$padded
                } else { $tex.mipMapsList[0][$level]=$pixels }
            } finally { $mip.Dispose() }
            $w=[Math]::Max(1,$w/2); $h=[Math]::Max(1,$h/2)
        }
        $target=Join-Path $output "$name.ctxr"
        [IO.File]::WriteAllBytes($target,$tex.GetBytes())
        $check=[AquaModelLibrary.Data.BluePoint.CTXR.CTXR]::new([IO.File]::ReadAllBytes($target))
        for ($level=0; $level -lt 8; $level++) {
            if ([Convert]::ToBase64String($check.mipMapsList[0][$level]) -cne [Convert]::ToBase64String($expected[$level])) {
                throw "Texture roundtrip failed: $name mip $level. Do not install this output."
            }
        }
        Write-Output "$name : verified 8 mip levels"
    } finally { $bitmap.Dispose() }
}
Write-Output "Built $($labels.Count) prompts in $output. Back up the original icons before copying these files into the game."
