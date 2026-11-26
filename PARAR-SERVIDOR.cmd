@echo off
rem Cierra el servidor online (y el juego si esta abierto) para poder compilar.
taskkill /f /im "Demon's Souls.exe"
taskkill /f /im "ninja.exe"
echo.
echo Servidor cerrado. Ya puedes compilar. Se vuelve a abrir solo al encender Windows o al abrir el juego.
pause
