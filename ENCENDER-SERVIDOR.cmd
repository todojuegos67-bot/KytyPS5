@echo off
rem Enciende el servidor online de Demon's Souls (sin ventana, icono junto al reloj).
cd /d "%~dp0"
start "" "%~dp0Demon's Souls.exe" --servidor
echo Servidor encendido. Tus amigos ya pueden conectarse a bloodbornetdj67.duckdns.org
timeout /t 4
