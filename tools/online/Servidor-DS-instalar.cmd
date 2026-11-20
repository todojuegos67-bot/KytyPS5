@echo off
rem Servidor online de Demon's Souls (online5): ponlo en la carpeta del juego (junto a "Demon's Souls.exe") y abrelo una vez.
rem Que hace: 1) reglas del firewall de Windows para el servidor (TCP 31413 y 31415, UDP 3758) - pide permiso de
rem administrador una sola vez; 2) acceso directo en la carpeta Inicio para que el servidor arranque con Windows (sin
rem admin); 3) arranca el servidor ahora (icono junto al reloj). No toca los puertos de Bloodborne (31313/31314/31315/3658).
setlocal DisableDelayedExpansion
set "EXE=%~dp0Demon's Souls.exe"
if not exist "%EXE%" (
  echo No encuentro "%EXE%". Pon este archivo en la carpeta del juego.
  pause
  exit /b 1
)
net session >nul 2>&1
if errorlevel 1 (
  echo Pidiendo permiso de administrador para el firewall...
  set "SELF=%~f0"
  powershell -NoProfile -Command "Start-Process -FilePath $env:SELF -Verb RunAs -Wait"
  goto :inicio
)
netsh advfirewall firewall delete rule name="Demon's Souls PC - servidor online (TCP)" >nul 2>&1
netsh advfirewall firewall delete rule name="Demon's Souls PC - relay P2P (UDP)" >nul 2>&1
netsh advfirewall firewall add rule name="Demon's Souls PC - servidor online (TCP)" dir=in action=allow protocol=TCP localport=31413,31415 program="%EXE%" enable=yes
netsh advfirewall firewall add rule name="Demon's Souls PC - relay P2P (UDP)" dir=in action=allow protocol=UDP localport=3758 program="%EXE%" enable=yes
echo Firewall listo.
exit /b 0
:inicio
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Startup')+'\Servidor Demon''s Souls.lnk'); $s.TargetPath=$env:EXE; $s.Arguments='--servidor 31415'; $s.WorkingDirectory=(Split-Path $env:EXE); $s.Save()"
echo Arranque con Windows listo (carpeta Inicio: "Servidor Demon's Souls").
start "" "%EXE%" --servidor 31415
echo Servidor arrancado: icono junto al reloj. En el router abre TCP 31413, TCP 31415 y UDP 3758 hacia la IP de este PC.
pause
