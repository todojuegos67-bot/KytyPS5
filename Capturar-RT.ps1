$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Content (Join-Path $dir 'rtcapture.on') '1'
Start-Process -FilePath "$dir\Launcher-souls.cmd" -WorkingDirectory $dir
Write-Host ''
Write-Host 'CAPTURA DE DATOS PARA EL RAY TRACING'
Write-Host '1. En el launcher pon Reflejos = Si y Sombras sol = Si, y juega.'
Write-Host '2. Ve a un sitio con agua, metal y antorchas.'
Write-Host '3. Quedate QUIETO 10 segundos.'
Write-Host '4. Luego anda 5 segundos.'
Write-Host '5. Espera 30 segundos mas y cierra el juego.'
Read-Host 'Cuando hayas cerrado el juego, pulsa ENTER aqui'
Remove-Item (Join-Path $dir 'rtcapture.on') -ErrorAction SilentlyContinue
Write-Host 'Listo. Avisa a Claude.'
Start-Sleep 3
