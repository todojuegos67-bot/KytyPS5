@echo off
rem Bloodborne con reescalador, arrancado EXACTAMENTE como Launcher-bloodborne.ps1 (JUGAR, modo seguro):
rem misma carpeta de trabajo (la del lanzador), mismo user (config, parches, mods), mismo sound fix
rem antes de arrancar, mismos argumentos (-g eboot.bin). Solo cambia el exe (este) y las variables BB_*.
rem BB_UPSCALER: dlss, dlaa, xess, fsr3, fsr4, native, off. BB_UPSCALE_MODE: native, quality, balanced, performance, ultraperf.
if "%BB_UPSCALER%"=="" set BB_UPSCALER=fsr4
if "%BB_UPSCALE_MODE%"=="" set BB_UPSCALE_MODE=quality
if "%BB_ANALYZE%"=="" set BB_ANALYZE=1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Probar-reescalado.ps1"
