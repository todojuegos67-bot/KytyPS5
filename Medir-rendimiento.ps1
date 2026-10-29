$ErrorActionPreference = 'Continue'
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$live = 'C:\kyty-live'
$out = Join-Path $dir 'logs'
New-Item -ItemType Directory -Force $live | Out-Null
Set-Content "$live\cmd.txt" 'id 1'
Set-Content "$live\medir.on" '1'
Start-Process -FilePath "$dir\Launcher-souls.cmd" -WorkingDirectory $dir

Write-Host ''
Write-Host 'Abre el juego desde el launcher como siempre y carga la partida.'
Write-Host 'Ve a un sitio y quedate QUIETO.'
Read-Host 'Cuando lleves 10 segundos quieto, pulsa ENTER aqui'
Set-Content "$live\cmd.txt" "id 5`nprof 10 $out\render.prof"
Write-Host 'Midiendo... NO TE MUEVAS durante 20 segundos.'
for ($i = 20; $i -gt 0; $i--) { Write-Host -NoNewline "`r  quedan $i s   "; Start-Sleep 1 }
Write-Host ''
Remove-Item "$live\medir.on" -ErrorAction SilentlyContinue
Write-Host 'Listo. Ya puedes cerrar el juego y avisar a Claude.'
Read-Host 'Pulsa ENTER para cerrar esta ventana'
