@echo off
rem Marvel's Wolverine en KytyPS5 - build TURBO (rama wolverine-turbo-1025, PRUEBA).
rem Motor del fork de Wolverine (wolverine-perf) + generacion de fotogramas FSR 3 (AMD).
rem
rem Uso: doble clic. Opcional desde una consola:
rem   "Play Wolverine TURBO.bat" x1   sin generacion de fotogramas (como antes)
rem   "Play Wolverine TURBO.bat" x2   FSR 3 x2 (por defecto)
rem   "Play Wolverine TURBO.bat" x3   FSR 3 x3 (DLL parcheada de emulator\fsr3-mfg, experimental)
rem   "Play Wolverine TURBO.bat" x4   FSR 3 x4 (DLL parcheada de emulator\fsr3-mfg, experimental)
rem
rem Partidas, cache de shaders y logs van a la carpeta userdata junto a este archivo, nunca al juego.
rem La primera vez COPIA (no mueve) la cache de shaders y las partidas de ..\userdata (el paquete v1),
rem asi que la carpeta original sigue intacta como copia de seguridad.
setlocal EnableExtensions
set "HERE=%~dp0"
set "EMUDIR=%HERE%emulator"
set "EMU=%EMUDIR%\kyty_emulator.exe"
set "USERDATA=%HERE%userdata"
set "PATHFILE=%HERE%game_path.txt"
set "OLDDATA=%HERE%..\userdata"

rem Generacion de fotogramas por defecto: 2 (x2). Cambia este numero (1 = apagada, 2, 3 o 4).
set "FG=2"
if /i "%~1"=="x1" set "FG=1"
if /i "%~1"=="x2" set "FG=2"
if /i "%~1"=="x3" set "FG=3"
if /i "%~1"=="x4" set "FG=4"

if not exist "%EMU%" (
  echo No encuentro emulator\kyty_emulator.exe junto a este archivo.
  echo Descomprime el paquete entero, manteniendo sus carpetas.
  pause
  exit /b 1
)

rem Carpeta del juego: la guardada, la del paquete v1 (..\wolverine) o la que escribas.
if exist "%PATHFILE%" set /p GAME=<"%PATHFILE%"
if defined GAME if exist "%GAME%\eboot.bin" goto run
if defined GAME echo La carpeta guardada ya no tiene eboot.bin: %GAME%
set "GAME="
if exist "%HERE%..\wolverine\eboot.bin" (
  for %%i in ("%HERE%..\wolverine") do set "GAME=%%~fi"
)
if not defined GAME if exist "E:\Wolverine-KytyPS5-v1\wolverine\eboot.bin" set "GAME=E:\Wolverine-KytyPS5-v1\wolverine"
if defined GAME (
  > "%PATHFILE%" echo %GAME%
  goto run
)

:ask
echo.
echo Escribe la ruta completa de la carpeta de Marvel's Wolverine
echo ^(la que contiene eboot.bin, por ejemplo E:\Wolverine-KytyPS5-v1\wolverine^):
set "GAME="
set /p GAME=Carpeta del juego:
set GAME=%GAME:"=%
if not exist "%GAME%\eboot.bin" (
  echo No hay eboot.bin en "%GAME%". Prueba otra vez.
  goto ask
)
> "%PATHFILE%" echo %GAME%

:run
if not exist "%USERDATA%\logs" mkdir "%USERDATA%\logs"

rem Primera vez: copiar la cache de shaders y las partidas del paquete v1 (si existen).
if not exist "%USERDATA%\_PipelineCache\PPSA03671.bin" if exist "%OLDDATA%\_PipelineCache\PPSA03671.bin" (
  echo Copiando la cache de shaders de %OLDDATA% ^(solo la primera vez^)...
  if not exist "%USERDATA%\_PipelineCache" mkdir "%USERDATA%\_PipelineCache"
  copy /y "%OLDDATA%\_PipelineCache\PPSA03671.bin" "%USERDATA%\_PipelineCache\" >nul
)
if not exist "%USERDATA%\_SaveData" if exist "%OLDDATA%\_SaveData" (
  echo Copiando tus partidas de %OLDDATA% ^(solo la primera vez; las originales no se tocan^)...
  xcopy "%OLDDATA%\_SaveData" "%USERDATA%\_SaveData\" /e /i /q /y >nul
)

rem Ajustes recomendados (los del perfil de Wolverine del Emulador PS5 Unico, modo Normal).
set KYTY_LABELS_AFTER_GPU=1
set KYTY_PIPELINE_CACHE_ANY_REVISION=1
set KYTY_PIPELINE_WORKERS=12
set KYTY_CLAMP_SAMPLER_SHADERS=59bdf02950d06cb6
set KYTY_BINDLESS_SKIP=0
rem Medidos en Wolverine con A/B en el mismo proceso (docs/EXPERIMENTS.md): -1.5 ms y -1.7 ms por frame.
set KYTY_ASYNC_WRITE_READBACK=1
set KYTY_WRITE_WINDOW_KB=16
rem Ordenes en vivo: escribe lineas NOMBRE=valor en este archivo mientras juegas.
set "KYTY_LIVE_FILE=%USERDATA%\kyty_live.txt"
if not exist "%KYTY_LIVE_FILE%" type nul > "%KYTY_LIVE_FILE%"

rem Generacion de fotogramas FSR 3 (AMD, funciona en NVIDIA). Sin vectores de movimiento del juego:
rem interpola con el flujo optico de FSR 3, por eso el HUD y los bordes rapidos pueden verse con halos.
set "KYTY_FSR3_FG="
set "KYTY_FSR3_MFG="
set "KYTY_FSR3_DLL="
set "FGTEXT=apagada"
if "%FG%"=="2" (
  set KYTY_FSR3_FG=1
  set "FGTEXT=FSR 3 x2"
)
if "%FG%"=="3" goto mfg
if "%FG%"=="4" goto mfg
goto opts
:mfg
if not exist "%EMUDIR%\fsr3-mfg\amd_fidelityfx_vk.dll" (
  echo Este paquete no trae la DLL parcheada ^(emulator\fsr3-mfg^): uso FSR 3 x2.
  set KYTY_FSR3_FG=1
  set "FGTEXT=FSR 3 x2"
  goto opts
)
set KYTY_FSR3_FG=1
set "KYTY_FSR3_MFG=%FG%"
set "KYTY_FSR3_DLL=%EMUDIR%\fsr3-mfg\amd_fidelityfx_vk.dll"
set "FGTEXT=FSR 3 x%FG% (experimental)"

:opts
rem Shaders que se saltan: los 6 de Mac (PR #937) y bfacdd28a2288ac4 (el parpadeo del suelo, y cuesta fps).
set "SKIP=--skip-shaders bad108e74fb72e9f,4e7f2c6bb9b158a1,8bfd230b9cd875a2,c4df2a00067e0666,dc76e1223a9bf673,e6d76d24f59f8015,bfacdd28a2288ac4"

rem --amd-cpu (emulacion exacta de VRSQRTPS) queda apagado: en un procesador AMD la instruccion nativa ya
rem se comporta como la de la consola y la opcion solo cuesta tiempo. Ponlo aqui si ves animaciones raras.
set "CPU_OPTS="

for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmmss"') do set STAMP=%%d
set "LOG=%USERDATA%\logs\run_%STAMP%_turbo.log"

rem El emulador escribe _SaveData y sus caches relativos a la carpeta de trabajo.
cd /d "%USERDATA%" || exit /b 1

echo.
echo Marvel's Wolverine - build TURBO (prueba)
echo   Juego: %GAME%
echo   Generacion de fotogramas: %FGTEXT%
echo   Log: %LOG%
echo No cambies el tamano de la ventana ni la minimices mientras juegas.
echo Para jugar sin generacion de fotogramas: "Play Wolverine TURBO.bat" x1
echo.
"%EMU%" --game "%GAME%" --console-language 3 --bindless --redzone --tessellation --readback-linear-images true %CPU_OPTS% %SKIP% > "%LOG%" 2>&1
set CODE=%ERRORLEVEL%
echo Codigo de salida: %CODE% >> "%LOG%"
echo.
if "%CODE%"=="0" (
  echo El juego se cerro con normalidad.
) else (
  echo El emulador se paro con el codigo %CODE%. Si se colgo, pasa este log:
  echo %LOG%
)
if defined KYTY_FSR3_FG (
  echo.
  echo Lineas [fsr3] y FSR3 FG del log ^(fps reales y mostrados^):
  findstr /l /b /c:"[fsr3]" /c:"FSR3" "%LOG%"
)
pause
