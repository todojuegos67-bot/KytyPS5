@echo off
rem Windows build: clang-cl + Ninja in the Visual Studio 2022 x64 environment.
rem   build-windows.cmd            configure (first time) and build kyty_emulator
rem   build-windows.cmd <target>   build another target (for example all)
rem Needs Git, Visual Studio 2022 or its Build Tools with the C++ workload (CMake and Ninja come with
rem it), LLVM clang-cl (19.1.7 tested; else the VS "C++ Clang tools" component), the Vulkan SDK and
rem Python 3 (README.md, "Building with build-windows.cmd"). Missing submodules are checked out first.
rem Release builds use -O3 -march=native and ThinLTO, like the Linux release, and a PGO profile:
rem _Build\pgo\windows\kyty.profdata when there is one (a local training, tools\local\PGO.md), else
rem the repository's tools\pgo\kyty.profdata (which the GitHub release build uses).
rem   KYTY_BUILD_DIR    build directory (default _Build\windows)
rem   KYTY_CMAKE_ARGS   extra arguments for the first configure (e.g. -DKYTY_PGO_GENERATE=ON;
rem                     -DKYTY_MARCH=x86-64-v3 for an exe other PCs can run: native is this CPU's)
setlocal
rem KYTY_BUILD_AFFINITY=<hex mask>: the whole build on those CPUs only (a machine whose compiler
rem crashes at random on some cores; this one's two unstable cores are disabled in the BIOS).
if defined KYTY_BUILD_AFFINITY if not defined KYTY_BUILD_PINNED (
	set "KYTY_BUILD_PINNED=1"
	start "" /wait /b /affinity %KYTY_BUILD_AFFINITY% cmd /c "%~f0" %*
	exit /b
)
set "ROOT=%~dp0"
set "BUILD=%ROOT%_Build\windows"
if defined KYTY_BUILD_DIR set "BUILD=%KYTY_BUILD_DIR%"
set "PROFILE=%ROOT%_Build\pgo\windows\kyty.profdata"
if not exist "%PROFILE%" set "PROFILE=%ROOT%tools\pgo\kyty.profdata"
set "PGO_ARGS="
if exist "%PROFILE%" set "PGO_ARGS=-DKYTY_PGO_USE=%PROFILE%"

rem The submodules a clone without --recurse-submodules lacks (only those: one checked out at another
rem commit on purpose stays as it is).
if not exist "%ROOT%.git" (
	echo No Git checkout: clone the repository with git ^(a source archive has no submodules^).
	exit /b 1
)
where git >nul 2>nul || (
	echo git not found: install Git ^(winget install Git.Git^).
	exit /b 1
)
set "MISSING="
for /f "tokens=2" %%s in ('git -C "%ROOT%." submodule status ^| findstr /b /c:"-"') do call set "MISSING=%%MISSING%% %%s"
if defined MISSING (
	echo Checking out submodules:%MISSING%
	git -C "%ROOT%." submodule update --init --recursive --%MISSING% || exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
if exist "%VSWHERE%" for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 Microsoft.VisualStudio.Component.VC.Llvm.Clang -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR if exist "%VSWHERE%" for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
	echo Visual Studio 2022 or Build Tools 2022 with the "Desktop development with C++" workload not found.
	exit /b 1
)
rem (vcvars64.bat runs vswhere by name.)
set "PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
where cmake >nul 2>nul || (
	echo cmake not found: add the VS component "C++ CMake tools for Windows" or winget install Kitware.CMake.
	exit /b 1
)
where ninja >nul 2>nul || (
	echo ninja not found: add the VS component "C++ CMake tools for Windows" or winget install Ninja-build.Ninja.
	exit /b 1
)

rem clang-cl: Prefer standalone LLVM, use VS bundled Clang as fallback.
set "LLVM_BIN="
if exist "%ProgramFiles%\LLVM\bin\clang-cl.exe" set "LLVM_BIN=%ProgramFiles%\LLVM\bin"
if not defined LLVM_BIN if exist "%VSDIR%\VC\Tools\Llvm\x64\bin\clang-cl.exe" set "LLVM_BIN=%VSDIR%\VC\Tools\Llvm\x64\bin"
if defined LLVM_BIN set "PATH=%LLVM_BIN%;%PATH%"
where clang-cl >nul 2>nul || (
	echo clang-cl not found: install LLVM ^(winget install LLVM.LLVM --version 19.1.7^).
	exit /b 1
)

rem glslangValidator: Vulkan SDK (winget install KhronosGroup.VulkanSDK).
if not defined VULKAN_SDK for /d %%d in ("%SystemDrive%\VulkanSDK\*") do set "VULKAN_SDK=%%d"
if defined VULKAN_SDK set "PATH=%VULKAN_SDK%\Bin;%PATH%"
where glslangValidator >nul 2>nul || (
	echo glslangValidator not found: install the Vulkan SDK ^(winget install KhronosGroup.VulkanSDK^).
	exit /b 1
)

if not exist "%BUILD%\build.ninja" (
	call :first_configure || exit /b 1
)

rem A profile that appeared after the first configure (an instrumented build never uses one).
if defined PGO_ARGS (
	findstr /c:"KYTY_PGO_GENERATE:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul 2>nul || (
		findstr /c:"KYTY_PGO_USE:FILEPATH=%PROFILE:\=/%" "%BUILD%\CMakeCache.txt" >nul 2>nul || (
			cmake -B "%BUILD%" %PGO_ARGS% >nul || exit /b 1
		)
	)
)

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=kyty_emulator"
cmake --build "%BUILD%" --target %TARGET% || exit /b 1
exit /b 0

:first_configure
rem Python 3 runs the Vulkan recording generator at build time.
python --version >nul 2>nul || py -3 --version >nul 2>nul || (
	echo Python 3 not found: install it ^(winget install Python.Python.3.12^).
	exit /b 1
)
for /f "tokens=3" %%v in ('clang-cl --version ^| findstr /b /c:"clang version"') do set "CLANG_VERSION=%%v"
echo clang-cl %CLANG_VERSION%: %LLVM_BIN%
if not "%CLANG_VERSION:~0,3%"=="19." echo Note: built and tested with clang 19 ^(LLVM 23.1.2 crashed compiling agc.cpp^).
rem xbyak and zydis are fetched by CMake at configure time, unless copies are under _Build\deps.
set "DEPS_ARGS="
if exist "%ROOT%_Build\deps\xbyak-src\CMakeLists.txt" set DEPS_ARGS=-DFETCHCONTENT_SOURCE_DIR_XBYAK="%ROOT%_Build\deps\xbyak-src"
if exist "%ROOT%_Build\deps\zydis-src\CMakeLists.txt" set DEPS_ARGS=%DEPS_ARGS% -DFETCHCONTENT_SOURCE_DIR_ZYDIS="%ROOT%_Build\deps\zydis-src"
cmake -S "%ROOT%." -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
	-DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DKYTY_BUILD_LAUNCHER=OFF ^
	%DEPS_ARGS% %PGO_ARGS% %KYTY_CMAKE_ARGS%
exit /b %ERRORLEVEL%
