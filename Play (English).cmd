@echo off
rem (libxess.dll ships in parts: joined on first run.)
if not exist "%~dp0libxess.dll" if exist "%~dp0libxess.dll.part0" (
  copy /b "%~dp0libxess.dll.part0"+"%~dp0libxess.dll.part1"+"%~dp0libxess.dll.part2" "%~dp0libxess.dll" >nul && del "%~dp0libxess.dll.part*"
)
start "" /min powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0Launcher-souls-EN.ps1"
