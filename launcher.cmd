@echo off
rem The settings window: KytyLauncher.exe (launcher-windows; in a source tree _Build\launcher), else the older launcher.ps1.
rem (Windows PowerShell by its path: with its folder missing from PATH, "powershell" was unknown, issue #55.)
if exist "%~dp0KytyLauncher.exe" (
	start "" "%~dp0KytyLauncher.exe"
	exit /b 0
)
if exist "%~dp0_Build\launcher\KytyLauncher.exe" (
	start "" "%~dp0_Build\launcher\KytyLauncher.exe"
	exit /b 0
)
start "" "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0launcher.ps1"
