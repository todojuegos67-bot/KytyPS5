@echo off
rem The settings window (launcher.ps1): game folder, resolution, fullscreen, console language, precompile.
rem (Windows PowerShell by its path: with its folder missing from PATH, "powershell" was unknown, issue #55.)
start "" "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0launcher.ps1"
